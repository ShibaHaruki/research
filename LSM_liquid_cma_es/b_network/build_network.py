"""入力層、リキッド層、シナプスの組み立て"""

# build_network.py
from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
from brian2 import Hz, NeuronGroup, PoissonGroup, Synapses, TimedArray, ms

from .models.connectivity_models import get_connection, layer_val
from .models.model_utils import merge_namespace

from .models.neuron_models import NEURON_MODELS #ニューロンモデル呼び出し
from .models.synapse_models import SYNAPSE_MODELS #シナプスモデル呼び出し

from .weight_initialization import init_in_to_liq, init_liq_intra #重みの初期化関数呼び出し

PAIR_KEYS = ("EE", "EI", "IE", "II")


# 設定がスカラーでもリストでも、指定したLiquid層の値を取得
def _layer_float(value: Any, layer_index: int) -> float:
    return float(layer_val(value, layer_index))

# 設定された層サイズを整数リストで返す
def _layer_sizes(cfg: dict, key: str) -> list[int]:
    sizes = cfg[key]
    if isinstance(sizes, Number):
        return [int(sizes)]
    return [int(size) for size in sizes]

def _synapse_post_equations(cfg: dict) -> str:
    synapse_model = SYNAPSE_MODELS[cfg["synapse_model"]]
    return synapse_model.get("post_eqs", synapse_model["eqs"])

# 指定したLiquid層のEE・EI・IE・II設定値をまとめて取得する関数
def _pair_values(pair_dict: dict[str, Any], layer_index: int) -> tuple[float, float, float, float]:
    return tuple(_layer_float(pair_dict[key], layer_index) for key in PAIR_KEYS)

# 興奮性・抑制性の接続先ごとの結合確率や重みスケールを、設定辞書から読み取る関数
def _read_post_ei_values(params: dict[str, Any], key: str) -> tuple[float, float]:

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

# シナプスモデルが必要とする外部パラメータを設定から読み出し、単位を付けてBrian2に渡す関数
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


#興奮性と抑制性の2つのNeuronGroupを、解析時に1つのLiquid層のように読み書きするための互換クラス
class LiquidLayer:

    #興奮性・抑制性グループを結合して扱う属性
    _array_names = {
        "typ", "v", "v_thr", "v_reset", "tau_m", "t_ref",
        "x", "y", "z", "I_merkel", "I_meissner", "I_RI", "I_SI",
        "I_USI", "I_syn", "H_syn",
    }

    def __init__(self, exc: NeuronGroup, inh: NeuronGroup):
        object.__setattr__(self, "exc", exc)
        object.__setattr__(self, "inh", inh)

    def __len__(self):
        return len(self.exc) + len(self.inh) #興奮性と抑制性を合わせたLiquid層全体のニューロン数

    def _split_value(self, value):
        n_exc = len(self.exc)
        if np.isscalar(value):
            return value, value
        if hasattr(value, "unit"):
            return value[:n_exc], value[n_exc:]
        array = np.asarray(value)
        return array[:n_exc], array[n_exc:]

    def __getattr__(self, name):
        if name in self._array_names:
            if name == "H_syn":
                return np.concatenate(
                    [getattr(self.exc, name), getattr(self.inh, name)]
                )
            return np.concatenate(
                [np.asarray(getattr(self.exc, name)), np.asarray(getattr(self.inh, name))]
            )
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._array_names and "exc" in self.__dict__:
            exc_value, inh_value = self._split_value(value)
            if name == "H_syn" and not hasattr(exc_value, "unit"):
                unit = getattr(self.exc, name).unit
                exc_value = exc_value * unit
                inh_value = inh_value * unit
            setattr(self.exc, name, exc_value)
            setattr(self.inh, name, inh_value)
            return
        object.__setattr__(self, name, value)

# ニューロンを興奮性・抑制性に分類し、それぞれの膜時定数と不応期を設定する関数
# 入力のNeuronGroupを作る関数。入力のTimedArrayも同時に作ることができる。
def make_in_neuron_group(N_in=None, input_ta=None, cfg: dict | None = None, name="G_in"):
   
    """
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

# ニューロングループの作成
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

# ニューロン種別・時定数・膜電位・入力電流・位置などの初期値をまとめて設定する関数
def _init_group_state(
    group: NeuronGroup,
    cfg: dict,
    rng: np.random.Generator,
    neuron_model: dict,
    set_position: bool = False,
    typ_value: int = 1,
) -> None:

    N = len(group)
    typ = np.full(N, int(typ_value), dtype=np.int32)
    tau_m = np.full(N, cfg["tau_exc"] if typ_value == 1 else cfg["tau_inh"])
    t_ref = np.full(N, cfg["ref_exc"] if typ_value == 1 else cfg["ref_inh"])

    group.typ = typ
    group.tau_m = tau_m * ms
    group.t_ref = t_ref * ms

    active_filters = {
        str(name).split("__", 1)[0]
        for name in cfg.get("USE_INPUT_FILTERS", [])
    }
    for filter_name in active_filters:
        current_name = f"I_{filter_name}"
        if hasattr(group, current_name):
            setattr(group, current_name, 0)

    group.I_syn = 0
    group.H_syn = 0
    group.tau_r = cfg["tau_r"] * ms
    group.tau_d = cfg["tau_d"] * ms

    group.bias = cfg["bias"]
    v_thr_exc = cfg.get("v_thr_exc", cfg.get("v_thr", -40.0))
    v_thr_inh = cfg.get("v_thr_inh", cfg.get("v_thr", -40.0))
    v_reset_exc = cfg.get("v_reset_exc", cfg.get("v_reset", -65.0))
    v_reset_inh = cfg.get("v_reset_inh", cfg.get("v_reset", -65.0))
    group.v_thr = np.where(typ == 1, v_thr_exc, v_thr_inh)
    group.v_reset = np.where(typ == 1, v_reset_exc, v_reset_inh)
    group.v = group.v_reset

    if set_position:
        pos = rng.uniform(0.0, 1.0, size=(N, 3))
        group.x, group.y, group.z = pos[:, 0], pos[:, 1], pos[:, 2]

    neuron_model["set_shared"](group)

# LSMのリキッド層を構成する興奮性ニューロン群と抑制性ニューロン群を作成し，初期化して、各層を LiquidLayer として返す関数
def make_liquid_neuron_groups(cfg: dict, rng, name_prefix="G_liq"):

    #E/Iニューロンモデルを取得する
    neuron_model_e = NEURON_MODELS["LIF_E"]
    neuron_model_i = NEURON_MODELS["LIF_I"]

    filters = list(dict.fromkeys(str(name).split("__", 1)[0] for name in cfg.get("USE_INPUT_FILTERS", [])))

    if not filters:
        raise ValueError("At least one input filter must be enabled.")
    
    input_sum = " + ".join(f"I_{filter_name}" for filter_name in filters)
    input_eqs = "\n".join(
        [*(f"I_{filter_name} : 1" for filter_name in filters), f"I_in = {input_sum} : 1"]
    )
    eqs = f"{neuron_model_e['eqs']}\n{input_eqs}\n{_synapse_post_equations(cfg)}"

    groups = []
    for layer_index, N in enumerate(_layer_sizes(cfg, "N_liq")):
        if N < 2:
            raise ValueError(
                "N_liq must be at least 2 when excitatory and inhibitory "
                "neurons use separate Brian2 NeuronGroups."
            )
        r_inh_layer = _layer_float(cfg["r_inh_liq"], layer_index)
        n_inh = int(np.round(r_inh_layer * N))
        n_inh = min(max(n_inh, 1), N - 1)
        n_exc = N - n_inh
        # 興奮性のニューロングループの作成
        group_exc = _create_spiking_group(
            N=n_exc,
            eqs=eqs,
            neuron_model=neuron_model_e,
            name=f"{name_prefix}_E_L{layer_index + 1}",
        )
        _init_group_state(
            group=group_exc,
            cfg=cfg,
            rng=rng,
            neuron_model=neuron_model_e,
            set_position=False,
            typ_value=1,
        )
        # 抑制性のニューロングループの作成
        group_inh = _create_spiking_group(
            N=n_inh,
            eqs=eqs,
            neuron_model=neuron_model_i,
            name=f"{name_prefix}_I_L{layer_index + 1}",
        )
        _init_group_state(
            group=group_inh,
            cfg=cfg,
            rng=rng,
            neuron_model=neuron_model_i,
            set_position=False,
            typ_value=-1,
        )
        positions = rng.uniform(0.0, 1.0, size=(N, 3))
        group_exc.x, group_exc.y, group_exc.z = positions[:n_exc].T
        group_inh.x, group_inh.y, group_inh.z = positions[n_exc:].T
        groups.append(LiquidLayer(group_exc, group_inh))

    return groups


def _make_synapses(pre,post,syn_model: dict,name: str,learning: dict | None = None,namespace: dict[str, Any] | None = None,cfg: dict | None = None,):

    # namespaceをまとめる
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
    
    #Brian2へ渡す引数を作る
    kwargs = dict(model=model, on_pre=on_pre, method="euler", name=name)
    if on_post:
        kwargs["on_post"] = on_post
    if merged_namespace:
        kwargs["namespace"] = merged_namespace
    return Synapses(pre, post, **kwargs)

#入力からLiquid層内へのシナプスを作る
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

    syn_map: dict[tuple[int, str, str], Synapses] = {}
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
        current_name = str(filter_name).split("__", 1)[0]

        for layer_index, layer_params in info["layers"].items():
            layer_index = int(layer_index)
            p_E, p_I = _read_post_ei_values(layer_params, "p")
            scale_E, scale_I = _read_post_ei_values(layer_params, "scale")
            layer = G_liq[layer_index]
            type_specs = (
                ("E", layer.exc, p_E, scale_E),
                ("I", layer.inh, p_I, scale_I),
            )
            idx_ranges = {}
            for post_type, post, p, scale in type_specs:
                key = (layer_index, current_name, post_type)
                if key not in syn_map:
                    syn_map[key] = Synapses(
                        G_in,
                        post,
                        model=f"w : 1\nI_{current_name}_post = w * I_pre : 1 (summed)\n",
                        method="euler",
                        namespace={},
                        name=f"{name_prefix}_{current_name}_{post_type}_liq{layer_index + 1}",
                    )
                synapses = syn_map[key]
                start = len(synapses)
                synapses.connect(condition=condition, p=p)
                stop = len(synapses)
                if stop > start:
                    synapses.w[start:stop] = init_in_to_liq(
                        rng,
                        stop - start,
                        scale=scale,
                        N_post=len(post),
                    )
                idx_ranges[post_type] = (start, stop)

            meta.append(
                {
                    "pre_ch": ch,
                    "filter": filter_name,
                    "layer_index": layer_index,
                    "p": {"E": p_E, "I": p_I},
                    "scale": {"E": scale_E, "I": scale_I},
                    "idx_range": idx_ranges,
                }
            )

    return list(syn_map.values()), meta

#入力ポアソンスパイクからLiquid層へのシナプスを作る
def make_poisson_to_liq_synapses(G_liq, rng, cfg: dict, name_prefix="S_poisson"):
    noise_cfg = dict(cfg.get("poisson_input", {}))
    if not bool(noise_cfg.get("enabled", False)):
        return [], [], []

    rate_hz = float(noise_cfg.get("rate_hz", 5.0))
    p_E, p_I = _read_post_ei_values(noise_cfg, "p")
    scale_E, scale_I = _read_post_ei_values(noise_cfg, "scale")
    current = str(noise_cfg.get("current", "exc")).lower()
    on_pre = "H_syn_post += (w / (tau_r_post * tau_d_post)) / Hz"
    current_sign = -1.0 if current in {"inh", "inhibitory", "i"} else 1.0

    groups = []
    synapse_list = []
    meta = []

    for layer_index, layer in enumerate(G_liq):
        if noise_cfg.get("N_ratio") is not None:
            n_noise = int(round(len(layer) * float(noise_cfg["N_ratio"])))
        else:
            n_noise = int(noise_cfg.get("N", 100))
        if n_noise <= 0:
            continue

        noise_group = PoissonGroup(
            n_noise,
            rates=rate_hz * Hz,
            name=f"G_poisson_liq{layer_index + 1}",
        )
        groups.append(noise_group)
        for post_type, post, p, scale in (
            ("E", layer.exc, p_E, scale_E),
            ("I", layer.inh, p_I, scale_I),
        ):
            synapses = Synapses(
                noise_group,
                post,
                model="w : 1",
                on_pre=on_pre,
                method="euler",
                namespace={"Hz": Hz},
                name=f"{name_prefix}_{post_type}_liq{layer_index + 1}",
            )
            synapses.connect(p=p)
            if len(synapses):
                synapses.w = init_in_to_liq(
                    rng,
                    len(synapses),
                    scale=current_sign * scale,
                    N_post=len(post),
                )
            synapses.delay = 0 * ms
            synapse_list.append(synapses)
            meta.append(
                {
                    "layer_index": layer_index,
                    "N": n_noise,
                    "rate_hz": rate_hz,
                    "current": current,
                    "post_type": post_type,
                    "p": p,
                    "scale": scale,
                    "S": synapses,
                    "G": noise_group,
                }
            )

    return groups, synapse_list, meta

#Liquid層内のシナプスを作る
def make_liq_intra_synapses(G_liq, rng, cfg: dict, name_prefix="S_liq_intra_"):

    syn_model = SYNAPSE_MODELS[cfg["synapse_model"]]["synapse"]
    connect_fn = get_connection(
        "liq_intra", cfg.get("liq_intra_connection", "random")
    )
    gain_pairs = cfg["liq_intra_gain_pairs"]
    p_pairs = cfg.get("p_liq_intra_pairs")
    if p_pairs is None:
        p_common = cfg.get("p_liq_intra", cfg.get("p_liq", 0.1))
        p_pairs = {key: p_common for key in PAIR_KEYS}

    synapse_list = []
    meta = []

    for layer_index, layer in enumerate(G_liq):
        gEE, gEI, gIE, gII = _pair_values(gain_pairs, layer_index)
        pEE, pEI, pIE, pII = _pair_values(p_pairs, layer_index)
        specs = (
            ("EE", layer.exc, layer.exc, pEE, gEE, 1.0),
            ("EI", layer.exc, layer.inh, pEI, gEI, 1.0),
            ("IE", layer.inh, layer.exc, pIE, gIE, -1.0),
            ("II", layer.inh, layer.inh, pII, gII, -1.0),
        )
        layer_synapses = {}
        for pair, pre, post, probability, gain, sign in specs:
            s = _make_synapses(
                pre,
                post,
                syn_model,
                name=f"{name_prefix}{pair}_L{layer_index + 1}",
                cfg=cfg,
            )
            connect_fn(
                s,
                post,
                cfg,
                layer_index,
                rng,
                pairs=(pair,),
                pre_group=pre,
                post_group=post,
                probability=probability,
            )
            if len(s):
                setattr(
                    s,
                    syn_model["w_attr"],
                    init_liq_intra(
                        rng,
                        len(s),
                        gain=sign * gain,
                        N_post=len(post),
                    ),
                )
            s.delay = 0 * ms
            layer_synapses[pair] = s
            synapse_list.append(s)

        meta.append(
            {
                "layer_index": layer_index,
                "S_exc": layer_synapses["EE"],
                "S_inh": layer_synapses["IE"],
                "S_pairs": layer_synapses,
            }
        )

    return synapse_list, meta
