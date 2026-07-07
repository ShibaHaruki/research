"""設定辞書から入力層、リキッド層、シナプスを組み立てる中核処理。"""

# build_network.py
from __future__ import annotations

import re
from numbers import Number
from typing import Any

import numpy as np
from brian2 import Hz, NeuronGroup, PoissonGroup, Synapses, TimedArray, ms

from .models.connectivity_models import get_connection, layer_val
from .models.model_utils import merge_namespace
from .models.neuron_models import NEURON_MODELS
from .models.synapse_models import SYNAPSE_MODELS
from .weight_initialization import init_in_to_liq, init_liq_intra


PAIR_KEYS = ("EE", "EI", "IE", "II")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
# 設定はスカラー、list、層別 dict、EE/EI/IE/II dict が混ざる。
# ここで「今の層・今の接続で使う値」にそろえてから下の構築処理へ渡す。
def _layer_float(value: Any, layer_index: int) -> float:
    return float(layer_val(value, layer_index))


def _layer_sizes(cfg: dict, key: str) -> list[int]:
    sizes = cfg[key]
    if isinstance(sizes, Number):
        return [int(sizes)]
    return [int(size) for size in sizes]


def _is_pair_config(value: Any) -> bool:
    if isinstance(value, Number):
        return True
    return isinstance(value, dict) and any(key in value for key in PAIR_KEYS)


def _pair_config(value: Any, default: Any | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {key: value for key in PAIR_KEYS}

    values = {key: value.get(key, default) for key in PAIR_KEYS}
    missing = [key for key, pair_value in values.items() if pair_value is None]
    if missing:
        raise KeyError(
            f"Pair config must define {PAIR_KEYS} or use a scalar value. "
            f"Missing: {missing}"
        )
    return values


def _index_config(value: Any, index: int, name: str) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[index]
    if isinstance(value, dict):
        if index in value:
            return value[index]
        if str(index) in value:
            return value[str(index)]
    raise KeyError(f"{name} does not define index {index}.")


def _synapse_post_equations(cfg: dict) -> str:
    synapse_model = SYNAPSE_MODELS[cfg["synapse_model"]]
    return synapse_model.get("post_eqs", synapse_model["eqs"])


def _compose_neuron_equations(neuron_eqs: str, synapse_eqs: str) -> str:
    """Insert synaptic state equations before dv/dt."""
    base_eqs = re.sub(r"^\s*I_exc\s*:\s*1\s*$", "", neuron_eqs, flags=re.MULTILINE)
    base_eqs = re.sub(r"^\s*I_inh\s*:\s*1\s*$", "", base_eqs, flags=re.MULTILINE)

    match = re.search(r"^\s*dv/dt\s*=", base_eqs, flags=re.MULTILINE)
    if match is None:
        return f"{base_eqs}\n{synapse_eqs}"
    return f"{base_eqs[:match.start()]}\n{synapse_eqs}\n{base_eqs[match.start():]}"


def _pair_values(pair_dict: dict[str, Any], layer_index: int) -> tuple[float, float, float, float]:
    return tuple(_layer_float(pair_dict[key], layer_index) for key in PAIR_KEYS)


def _read_post_ei_values(params: dict[str, Any], key: str) -> tuple[float, float]:
    """Read values for post excitatory/inhibitory targets.

    Supported forms:
    - {"p": 0.1}
    - {"p": {"E": 0.1, "I": 0.2}}
    - {"p_E": 0.1, "p_I": 0.2}
    """
    value = params.get(key)
    if isinstance(value, dict):
        e_value = value.get("E", value.get("exc"))
        i_value = value.get("I", value.get("inh"))
    else:
        e_value = params.get(f"{key}_E", params.get(f"{key}_exc", value))
        i_value = params.get(f"{key}_I", params.get(f"{key}_inh", value))

    if e_value is None or i_value is None:
        raise KeyError(
            f"Input-to-liquid layer params must define '{key}' as a scalar, "
            f"'{key}' as {{'E': ..., 'I': ...}}, or '{key}_E'/'{key}_I'."
        )
    return float(e_value), float(i_value)


def _synapse_namespace(cfg: dict | None, syn_model: dict) -> dict[str, Any]:
    namespace = {}
    if cfg is None:
        return namespace

    for name in syn_model.get("ns_vars", []):
        if name not in cfg:
            raise KeyError(
                f"Missing parameter for synapse_model='{cfg.get('synapse_model')}': {name}."
            )
        value = cfg[name]
        if name.startswith("tau") and isinstance(value, Number):
            value = value * ms
        namespace[name] = value
    return namespace


def _input_current_name(filter_name: str) -> str:
    return str(filter_name).split("__", 1)[0]


# ---------------------------------------------------------------------------
# Neuron helpers
# ---------------------------------------------------------------------------
# 入力層、リキッド層を Brian2 の NeuronGroup として作る。
# 興奮/抑制ニューロンの割り当て、時定数、不応期、初期膜電位もここで決める。
def make_ei_arrays(
    N: int,
    r_inh: float,
    rng: np.random.Generator,
    tau_exc: float,
    tau_inh: float,
    ref_exc: float,
    ref_inh: float,
):
    neuron_array = np.ones(N, dtype=np.int32)
    N_inh = int(np.round(r_inh * N))
    inh_idx = rng.choice(N, size=N_inh, replace=False) if N_inh > 0 else np.array([], dtype=int)
    neuron_array[inh_idx] = -1

    tau_m = np.where(neuron_array == 1, tau_exc, tau_inh)
    t_ref = np.where(neuron_array == 1, ref_exc, ref_inh)
    return neuron_array, tau_m, t_ref


def make_in_neuron_group(N_in=None, input_ta=None, cfg: dict | None = None, name="G_in"):
    """Create the input group.

    Supported call styles:
    - make_in_neuron_group(N_in, input_ta) -> G_in
    - make_in_neuron_group(cfg) -> (G_in, input_ta)
    """
    if isinstance(N_in, dict) and input_ta is None and cfg is None:
        cfg = N_in
        N_in = None

    if cfg is not None and N_in is None and input_ta is None:
        input_rows = cfg.get("INPUT_ROWS")
        if input_rows is not None:
            N_in = len(input_rows)
        else:
            filters = cfg["USE_INPUT_FILTERS"]
            N_in = int(cfg["NUM_CHANNEL"]) * len(filters)
        input_ta = TimedArray(np.zeros((2, N_in)), dt=float(cfg.get("dt_ms", 0.1)) * ms)
        return_tuple = True
    else:
        if N_in is None or input_ta is None:
            raise ValueError("N_in and input_ta are required unless a cfg dict is passed.")
        return_tuple = False

    G_in = NeuronGroup(
        int(N_in),
        """
        t_start : second (shared)
        I = input_ta(t - t_start, i) : 1
        """,
        method="euler",
        namespace={"input_ta": input_ta},
        name=name,
    )
    G_in.t_start = 0 * ms

    if return_tuple:
        return G_in, input_ta
    return G_in


def _create_spiking_group(N: int, eqs: str, neuron_model: dict, name: str) -> NeuronGroup:
    return NeuronGroup(
        N,
        eqs,
        threshold=neuron_model["threshold"],
        reset=neuron_model["reset"],
        refractory=neuron_model["refractory"],
        method=neuron_model["method"],
        namespace=neuron_model.get("namespace", {}),
        name=name,
    )


def _init_group_state(
    group: NeuronGroup,
    cfg: dict,
    rng: np.random.Generator,
    r_inh: float,
    neuron_model: dict,
    set_position: bool = False,
) -> None:
    N = len(group)
    typ, tau_m, t_ref = make_ei_arrays(
        N=N,
        r_inh=r_inh,
        rng=rng,
        tau_exc=cfg["tau_exc"],
        tau_inh=cfg["tau_inh"],
        ref_exc=cfg["ref_exc"],
        ref_inh=cfg["ref_inh"],
    )

    group.typ = typ
    group.tau_m = tau_m * ms
    group.t_ref = t_ref * ms

    group.I_merkel = 0
    group.I_meissner = 0
    group.I_RI = 0
    group.I_SI = 0
    group.I_USI = 0
    group.I_exc = 0
    group.I_inh = 0
    group.H_exc = 0
    group.H_inh = 0
    group.tau_r = cfg["tau_r"] * ms
    group.tau_d = cfg["tau_d"] * ms

    group.bias = cfg["bias"]
    group.v_thr = cfg["v_thr"]
    group.v_reset = cfg["v_reset"]
    group.v = cfg["v_reset"]

    if set_position:
        pos = rng.uniform(0.0, 1.0, size=(N, 3))
        group.x, group.y, group.z = pos[:, 0], pos[:, 1], pos[:, 2]

    neuron_model["set_shared"](group)


def make_liquid_neuron_groups(cfg: dict, rng, name_prefix="G_liq"):
    neuron_model = NEURON_MODELS[cfg["neuron_model"]]
    eqs = _compose_neuron_equations(neuron_model["eqs"], _synapse_post_equations(cfg))

    groups = []
    for layer_index, N in enumerate(_layer_sizes(cfg, "N_liq")):
        group = _create_spiking_group(
            N=N,
            eqs=eqs,
            neuron_model=neuron_model,
            name=f"{name_prefix}{layer_index + 1}",
        )
        _init_group_state(
            group=group,
            cfg=cfg,
            rng=rng,
            r_inh=_layer_float(cfg["r_inh_liq"], layer_index),
            neuron_model=neuron_model,
            set_position=True,
        )
        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Synapse helpers
# ---------------------------------------------------------------------------
# 入力→リキッド、リキッド再帰の Synapses を作る。
# 接続確率、初期重み、学習則の式、層ごとの重みスケールをここで反映する。
def _post_type_counts(group: NeuronGroup) -> tuple[int, int]:
    typ = np.asarray(group.typ)
    return int(np.sum(typ == 1)), int(np.sum(typ == -1))


def _indices_by_post_type(synapses: Synapses) -> tuple[np.ndarray, np.ndarray]:
    post_types = np.asarray(synapses.typ_post)
    return np.where(post_types == 1)[0], np.where(post_types == -1)[0]


def _set_pair_weights(
    synapses: Synapses,
    w_attr: str,
    idx_exc_post: np.ndarray,
    idx_inh_post: np.ndarray,
    gain_exc_post: float,
    gain_inh_post: float,
    N_post_E: int,
    N_post_I: int,
    rng: np.random.Generator,
    init_fn,
) -> None:
    weights = np.zeros(len(synapses), dtype=float)
    if idx_exc_post.size:
        weights[idx_exc_post] = init_fn(
            rng,
            idx_exc_post.size,
            gain=gain_exc_post,
            N_post=N_post_E,
        )
    if idx_inh_post.size:
        weights[idx_inh_post] = init_fn(
            rng,
            idx_inh_post.size,
            gain=gain_inh_post,
            N_post=N_post_I,
        )
    setattr(synapses, w_attr, weights)
    if "x_stp" in synapses.variables:
        synapses.x_stp = 1.0
    if "u_stp" in synapses.variables:
        synapses.u_stp = 0.0


def _make_synapses(
    pre,
    post,
    syn_model: dict,
    name: str,
    learning: dict | None = None,
    namespace: dict[str, Any] | None = None,
    cfg: dict | None = None,
):
    merged_namespace = merge_namespace(
        syn_model.get("namespace"),
        _synapse_namespace(cfg, syn_model),
        namespace,
    )

    if learning is None:
        model = syn_model["eqs"]
        on_pre = syn_model["on_pre"]
        on_post = None
    else:
        model = syn_model["eqs"] + learning["eqs"]
        on_pre = syn_model["on_pre"] + learning["on_pre"]
        on_post = learning["on_post"]

    kwargs = dict(model=model, on_pre=on_pre, method="euler", name=name)
    if on_post:
        kwargs["on_post"] = on_post
    if merged_namespace:
        kwargs["namespace"] = merged_namespace
    return Synapses(pre, post, **kwargs)


def make_in_to_liq_synapses(G_in, G_liq, rng, cfg, name_prefix="S"):
    filters = cfg["USE_INPUT_FILTERS"]
    input_channels = cfg.get("USE_INPUT_CHANNELS", list(range(int(cfg.get("NUM_CHANNEL", 0)))))
    channel_to_row = {int(channel): row for row, channel in enumerate(input_channels)}
    input_rows = cfg.get("INPUT_ROWS")
    if input_rows is None:
        input_rows = [(ch, filter_name) for ch in input_channels for filter_name in filters]
    input_row_by_route = {
        (int(ch), str(filter_name)): row
        for row, (ch, filter_name) in enumerate(input_rows)
    }
    route = cfg["IN_ROUTE"]

    syn_map: dict[tuple[int, str], tuple[Synapses, int]] = {}
    meta: list[dict[str, Any]] = []

    for (ch, filter_name), info in route.items():
        if int(ch) not in channel_to_row:
            raise KeyError(
                f"IN_ROUTE uses sensor channel {ch}, but INPUT_FILTER_MAP only "
                f"defines channels {list(input_channels)}."
            )
        input_key = (int(ch), str(filter_name))
        if input_key not in input_row_by_route:
            raise KeyError(
                f"IN_ROUTE uses input {input_key}, but INPUT_FILTER_MAP does not "
                "define that channel/filter pair."
            )
        input_row = input_row_by_route[input_key]
        condition = f"(i=={input_row})"
        current_name = _input_current_name(filter_name)

        for layer_index, layer_params in info["layers"].items():
            layer_index = int(layer_index)
            p_E, p_I = _read_post_ei_values(layer_params, "p")
            scale_E, scale_I = _read_post_ei_values(layer_params, "scale")
            key = (layer_index, current_name)

            if key not in syn_map:
                post = G_liq[layer_index]
                syn_map[key] = (
                    Synapses(
                        G_in,
                        post,
                        model=f"w : 1\nI_{current_name}_post = w * I_pre : 1 (summed)\n",
                        method="euler",
                        namespace={},
                        name=f"{name_prefix}_{current_name}_liq{layer_index + 1}",
                    ),
                    _post_type_counts(post),
                )

            synapses, (N_post_E, N_post_I) = syn_map[key]

            type_specs = (
                ("E", "typ_post==1", p_E, scale_E, N_post_E),
                ("I", "typ_post==-1", p_I, scale_I, N_post_I),
            )
            idx_ranges = {}
            for post_type, post_condition, p, scale, N_post in type_specs:
                start = len(synapses)
                synapses.connect(condition=f"({condition}) and ({post_condition})", p=p)
                stop = len(synapses)

                if stop > start:
                    synapses.w[start:stop] = init_in_to_liq(
                        rng,
                        stop - start,
                        scale=scale,
                        N_post=N_post,
                    )

                idx_ranges[post_type] = (start, stop)

            meta.append(
                {
                    "pre_ch": ch,
                    "filter": filter_name,
                    "layer_index": layer_index,
                    "p": {"E": p_E, "I": p_I},
                    "scale": {"E": scale_E, "I": scale_I},
                    "S": synapses,
                    "idx_range": idx_ranges,
                }
            )

    return [synapses for synapses, _ in syn_map.values()], meta


def _poisson_input_config(cfg: dict) -> dict[str, Any]:
    noise_cfg = dict(cfg.get("poisson_input", {}))
    if "POISSON_INPUT_ENABLE" in cfg:
        noise_cfg["enabled"] = cfg["POISSON_INPUT_ENABLE"]
    return noise_cfg


def make_poisson_to_liq_synapses(G_liq, rng, cfg: dict, name_prefix="S_poisson"):
    noise_cfg = _poisson_input_config(cfg)
    if not bool(noise_cfg.get("enabled", False)):
        return [], [], []

    rate_hz = float(noise_cfg.get("rate_hz", 5.0))
    p_E, p_I = _read_post_ei_values(noise_cfg, "p")
    scale_E, scale_I = _read_post_ei_values(noise_cfg, "scale")
    current = str(noise_cfg.get("current", "exc")).lower()
    post_trace = "H_inh_post" if current in {"inh", "inhibitory", "i"} else "H_exc_post"
    on_pre = f"{post_trace} += (w / (tau_r_post * tau_d_post)) / Hz"

    groups = []
    synapse_list = []
    meta = []

    for layer_index, post in enumerate(G_liq):
        if noise_cfg.get("N_ratio") is not None:
            n_noise = int(round(len(post) * float(noise_cfg["N_ratio"])))
        else:
            n_noise = int(noise_cfg.get("N", 100))
        if n_noise <= 0:
            continue

        noise_group = PoissonGroup(
            n_noise,
            rates=rate_hz * Hz,
            name=f"G_poisson_liq{layer_index + 1}",
        )
        synapses = Synapses(
            noise_group,
            post,
            model="w : 1",
            on_pre=on_pre,
            method="euler",
            namespace={"Hz": Hz},
            name=f"{name_prefix}_liq{layer_index + 1}",
        )
        start_E = len(synapses)
        synapses.connect(condition="typ_post==1", p=p_E)
        stop_E = len(synapses)
        start_I = len(synapses)
        synapses.connect(condition="typ_post==-1", p=p_I)
        stop_I = len(synapses)

        N_post_E, N_post_I = _post_type_counts(post)
        if stop_E > start_E:
            synapses.w[start_E:stop_E] = init_in_to_liq(
                rng,
                stop_E - start_E,
                scale=scale_E,
                N_post=N_post_E,
            )
        if stop_I > start_I:
            synapses.w[start_I:stop_I] = init_in_to_liq(
                rng,
                stop_I - start_I,
                scale=scale_I,
                N_post=N_post_I,
            )
        synapses.delay = 0 * ms

        groups.append(noise_group)
        synapse_list.append(synapses)
        meta.append(
            {
                "layer_index": layer_index,
                "N": n_noise,
                "rate_hz": rate_hz,
                "current": current,
                "p": {"E": p_E, "I": p_I},
                "scale": {"E": scale_E, "I": scale_I},
                "S": synapses,
                "G": noise_group,
            }
        )

    return groups, synapse_list, meta


def make_liq_intra_synapses(G_liq, rng, cfg: dict, name_prefix="S_liq_intra_"):
    synE = SYNAPSE_MODELS[cfg["synapse_model"]]["liq_exc"]
    synI = SYNAPSE_MODELS[cfg["synapse_model"]]["liq_inh"]
    connect_fn = get_connection("liq_intra", cfg.get("liq_intra_connection", "random"))
    gain_pairs = cfg["liq_intra_gain_pairs"]

    synapse_list = []
    meta = []

    for layer_index, group in enumerate(G_liq):
        N_post_E, N_post_I = _post_type_counts(group)
        gEE, gEI, gIE, gII = _pair_values(gain_pairs, layer_index)

        sE = _make_synapses(
            group,
            group,
            synE,
            name=f"{name_prefix}E_L{layer_index + 1}",
            cfg=cfg,
        )
        connect_fn(sE, group, cfg, layer_index, rng, pairs=("EE", "EI"))
        idx_EE, idx_EI = _indices_by_post_type(sE)
        _set_pair_weights(sE, synE["w_attr"], idx_EE, idx_EI, gEE, gEI, N_post_E, N_post_I, rng, init_liq_intra)
        sE.delay = 0 * ms

        sI = _make_synapses(
            group,
            group,
            synI,
            name=f"{name_prefix}I_L{layer_index + 1}",
            cfg=cfg,
        )
        connect_fn(sI, group, cfg, layer_index, rng, pairs=("IE", "II"))
        idx_IE, idx_II = _indices_by_post_type(sI)
        _set_pair_weights(sI, synI["w_attr"], idx_IE, idx_II, gIE, gII, N_post_E, N_post_I, rng, init_liq_intra)
        sI.delay = 0 * ms

        synapse_list.extend([sE, sI])
        meta.append(
            {
                "layer_index": layer_index,
                "S_exc": sE,
                "S_inh": sI,
                "idx": {"EE": idx_EE, "EI": idx_EI, "IE": idx_IE, "II": idx_II},
            }
        )

    return synapse_list, meta
