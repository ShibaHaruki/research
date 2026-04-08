# build_network.py
import numpy as np
import re
from brian2 import *
from typing import List, Dict, Any, Tuple

from models.connectivity_models import get_connection
from weight_initialization import init_in_to_liq, init_liq_intra, init_liq_to_out
from models.connectivity_models import layer_val
from models.synapse_models import SYNAPSE_MODELS
from models.neuron_models import NEURON_MODELS
from models.learning_rule_models import LEARNING_RULES


# -----------------------------
# 共通ユーティリティ
# -----------------------------
def _read_pair_dict_with_layer_val(
    pair_dict: Dict[str, Any],
    out_idx: int,
    liq_idx: int,
    n_out: int,
    n_liq: int,
    prefer: str = "out",
) -> Tuple[float, float, float, float]:
    """
    pair_dict = {"EE":..., "EI":..., "IE":..., "II":...}
    値がスカラーでも配列でもOKにして、layer_val で取り出す。
    prefer:
      - "out": 値が配列なら基本 out_idx を使う（長さがn_liqなら liq_idx）
      - "liq": 値が配列なら基本 liq_idx を使う（長さがn_outなら out_idx）
    """
    def pick(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            n = len(v)
            if prefer == "out":
                if n == n_out:
                    idx = out_idx
                elif n == n_liq:
                    idx = liq_idx
                else:
                    idx = 0
            else:  # prefer == "liq"
                if n == n_liq:
                    idx = liq_idx
                elif n == n_out:
                    idx = out_idx
                else:
                    idx = 0
            return float(layer_val(v, idx))
        else:
            return float(v)

    return pick(pair_dict["EE"]), pick(pair_dict["EI"]), pick(pair_dict["IE"]), pick(pair_dict["II"])


# -----------------------------
# E/I 割り当て
# -----------------------------
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
    ref = np.where(neuron_array == 1, ref_exc, ref_inh)
    return neuron_array, tau_m, ref


# -----------------------------
# NeuronGroup 作成
# -----------------------------
def make_in_neuron_group(cfg, name="G_in"):
    filters = cfg["USE_INPUT_FILTERS"]
    S = len(filters)
    num_channel = cfg["NUM_CHANNEL"]
    N_in = num_channel * S

    dt_ms = cfg.get("dt_ms", 0.1)
    input_ta = TimedArray(np.zeros((2, N_in)), dt=dt_ms * ms)

    G_in = NeuronGroup(
        N_in,
        """
        t_start : second (shared)
        I = input_ta(t - t_start, i) : 1
        """,
        method="euler",
        name=name,
    )
    G_in.t_start = 0 * ms
    return G_in, input_ta


def make_liquid_neuron_groups(cfg: dict, rng, name_prefix="G_liq"):

    neuron_models = NEURON_MODELS[cfg["neuron_model"]]
    post_eqs = SYNAPSE_MODELS[cfg["synapse_model"]]["post_eqs"]

    # base eqs から I_exc/I_inh 宣言を消して post_eqs を注入（新しい関数は作らずここで処理）
    base_eqs = neuron_models["eqs"]
    base_eqs = re.sub(r"^\s*I_exc\s*:\s*1\s*$", "", base_eqs, flags=re.MULTILINE)
    base_eqs = re.sub(r"^\s*I_inh\s*:\s*1\s*$", "", base_eqs, flags=re.MULTILINE)

    m = re.search(r"^\s*dv/dt\s*=", base_eqs, flags=re.MULTILINE)
    if m is None:
        eqs_liq = base_eqs + "\n" + post_eqs
    else:
        eqs_liq = base_eqs[:m.start()] + "\n" + post_eqs + "\n" + base_eqs[m.start():]

    G_liq = []

    for liq in range(cfg["liquid_layer"]):
        N = int(cfg["N_liq"][liq])
        g = NeuronGroup(
            N,
            eqs_liq,
            threshold=neuron_models["threshold"],
            reset=neuron_models["reset"],
            refractory=neuron_models["refractory"],
            method=neuron_models["method"],
            name=f"{name_prefix}{liq+1}",
        )

        typ, tau_m_res, ref_res = make_ei_arrays(
            N=N,
            r_inh=layer_val(cfg["r_inh_liq"], liq),
            rng=rng,
            tau_exc=cfg["tau_exc"],
            tau_inh=cfg["tau_inh"],
            ref_exc=cfg["ref_exc"],
            ref_inh=cfg["ref_inh"],
        )

        g.typ = typ
        g.tau_m = tau_m_res * ms
        g.t_ref = ref_res * ms

        # 入力（summedで入る）
        g.I_merkel = 0
        g.I_meissner = 0

        # ★I_exc/I_inh は post_eqs の ODEで状態変数になるので初期化OK
        g.I_exc = 0
        g.I_inh = 0
        # post_eqs で追加した状態（H_exc/H_inh など）を初期化
        g.H_exc = 0
        g.H_inh = 0
        # shared time constants（post_eqs 側が tau_r/tau_d を持つ前提）
        g.tau_r = cfg["tau_r"] * ms
        g.tau_d = cfg["tau_d"] * ms

        g.bias = cfg["bias"]
        g.v_thr = cfg["v_thr"]
        g.v_reset = cfg["v_reset"]
        g.v = cfg["v_reset"]

        pos = rng.uniform(0.0, 1.0, size=(N, 3))
        g.x, g.y, g.z = pos[:, 0], pos[:, 1], pos[:, 2]

        neuron_models["set_shared"](g)
        G_liq.append(g)

    return G_liq


def make_output_neuron_groups(cfg: dict, rng, name_prefix="G_out"):
    neuron_models = NEURON_MODELS[cfg["neuron_model"]]
    post_eqs = SYNAPSE_MODELS[cfg["synapse_model"]]["post_eqs"]

    base_eqs = neuron_models["eqs"]
    base_eqs = re.sub(r"^\s*I_exc\s*:\s*1\s*$", "", base_eqs, flags=re.MULTILINE)
    base_eqs = re.sub(r"^\s*I_inh\s*:\s*1\s*$", "", base_eqs, flags=re.MULTILINE)

    m = re.search(r"^\s*dv/dt\s*=", base_eqs, flags=re.MULTILINE)
    if m is None:
        eqs_out = base_eqs + "\n" + post_eqs
    else:
        eqs_out = base_eqs[:m.start()] + "\n" + post_eqs + "\n" + base_eqs[m.start():]

    G_out = []

    for out in range(cfg["output_layer"]):
        N = int(cfg["N_out"][out])
        g = NeuronGroup(
            N,
            eqs_out,
            threshold=neuron_models["threshold"],
            reset=neuron_models["reset"],
            refractory=neuron_models["refractory"],
            method=neuron_models["method"],
            name=f"{name_prefix}{out+1}",
        )

        typ, tau_m_res, ref_res = make_ei_arrays(
            N=N,
            r_inh=cfg["r_inh_out"],
            rng=rng,
            tau_exc=cfg["tau_exc"],
            tau_inh=cfg["tau_inh"],
            ref_exc=cfg["ref_exc"],
            ref_inh=cfg["ref_inh"],
        )

        g.typ = typ
        g.tau_m = tau_m_res * ms
        g.t_ref = ref_res * ms

        g.I_merkel = 0
        g.I_meissner = 0

        g.I_exc = 0
        g.I_inh = 0
        g.H_exc = 0
        g.H_inh = 0
        g.tau_r = cfg["tau_r"] * ms
        g.tau_d = cfg["tau_d"] * ms

        g.bias = cfg["bias"]
        g.v_thr = cfg["v_thr"]
        g.v_reset = cfg["v_reset"]
        g.v = cfg["v_reset"]

        neuron_models["set_shared"](g)
        G_out.append(g)

    return G_out


# -----------------------------
# Synapses 作成
# -----------------------------
def make_in_to_liq_synapses(G_in, G_liq, rng, cfg, name_prefix="S"):

    filters = cfg["USE_INPUT_FILTERS"]
    S = len(filters)
    f2mod = {f: i for i, f in enumerate(filters)}
    route = cfg["IN_ROUTE"]

    syn_map: Dict[Tuple[int, str], Tuple[Synapses, int]] = {}
    meta: List[Dict[str, Any]] = []

    for (ch, f), info in route.items():
        mod = f2mod[f]
        cond = f"((i//{S}=={ch}) and (i%{S}=={mod}))"

        for layer_index, lp in info["layers"].items():
            layer_index = int(layer_index)
            p = float(lp["p"])
            sc = float(lp["scale"])

            key = (layer_index, f)
            if key not in syn_map:
                post = G_liq[layer_index]
                N_post = int(len(post))

                s = Synapses(
                    G_in,
                    post,
                    model=f"w : 1\nI_{f}_post = w * I_pre : 1 (summed)\n",
                    method="euler",
                    name=f"{name_prefix}_{f}_liq{layer_index+1}",
                )
                syn_map[key] = (s, N_post)

            s, N_post = syn_map[key]

            n0 = len(s)
            s.connect(condition=cond, p=p)
            n1 = len(s)

            if n1 > n0:
                w_new = init_in_to_liq(rng, n1 - n0, scale=sc, N_post=N_post)
                s.w[n0:n1] = w_new

            meta.append(
                {"pre_ch": ch, "filter": f, "layer_index": layer_index, "p": p, "scale": sc, "S": s, "idx_range": (n0, n1)}
            )

    S_in_to_liq_list = [s for (s, _) in syn_map.values()]
    return S_in_to_liq_list, meta


def make_liq_intra_synapses(G_liq, rng, cfg: dict, name_prefix="S_liq_intra_"):

    synE = SYNAPSE_MODELS[cfg["synapse_model"]]["liq_exc"]
    synI = SYNAPSE_MODELS[cfg["synapse_model"]]["liq_inh"]

    conn_name = cfg.get("liq_intra_connection", "random")
    connect_fn = get_connection("liq_intra", conn_name)

    g_pairs = cfg["liq_intra_gain_pairs"]

    S_list, meta = [], []

    for layer_index, g in enumerate(G_liq):
        typ_np = np.asarray(g.typ)
        N_post_E = int(np.sum(typ_np == 1))
        N_post_I = int(np.sum(typ_np == -1))

        gEE = float(layer_val(g_pairs["EE"], layer_index))
        gEI = float(layer_val(g_pairs["EI"], layer_index))
        gIE = float(layer_val(g_pairs["IE"], layer_index))
        gII = float(layer_val(g_pairs["II"], layer_index))

        # ---- Excitatory: EE + EI in ONE Synapses ----
        sE = Synapses(
            g, g,
            model=synE["eqs"],
            on_pre=synE["on_pre"],
            method="euler",
            name=f"{name_prefix}E_L{layer_index+1}",
        )
        connect_fn(sE, g, cfg, layer_index, rng, pairs=("EE", "EI"))

        idx_EE = np.where(np.asarray(sE.typ_post) == 1)[0]
        idx_EI = np.where(np.asarray(sE.typ_post) == -1)[0]

        wE = np.zeros(len(sE), dtype=float)
        if idx_EE.size:
            wE[idx_EE] = init_liq_intra(rng, idx_EE.size, gain=gEE, N_post=N_post_E)
        if idx_EI.size:
            wE[idx_EI] = init_liq_intra(rng, idx_EI.size, gain=gEI, N_post=N_post_I)
        setattr(sE, synE["w_attr"], wE)
        sE.delay = 0 * ms

        # ---- Inhibitory: IE + II in ONE Synapses ----
        sI = Synapses(
            g, g,
            model=synI["eqs"],
            on_pre=synI["on_pre"],
            method="euler",
            name=f"{name_prefix}I_L{layer_index+1}",
        )
        connect_fn(sI, g, cfg, layer_index, rng, pairs=("IE", "II"))

        idx_IE = np.where(np.asarray(sI.typ_post) == 1)[0]
        idx_II = np.where(np.asarray(sI.typ_post) == -1)[0]

        wI = np.zeros(len(sI), dtype=float)
        if idx_IE.size:
            wI[idx_IE] = init_liq_intra(rng, idx_IE.size, gain=gIE, N_post=N_post_E)
        if idx_II.size:
            wI[idx_II] = init_liq_intra(rng, idx_II.size, gain=gII, N_post=N_post_I)
        setattr(sI, synI["w_attr"], wI)
        sI.delay = 0 * ms

        S_list.extend([sE, sI])
        meta.append({
            "layer_index": layer_index,
            "S_exc": sE, "S_inh": sI,
            "idx": {"EE": idx_EE, "EI": idx_EI, "IE": idx_IE, "II": idx_II},
        })

    return S_list, meta


def make_liq_to_out_synapses(
    G_liq, G_out_list, rng, cfg: dict, name_prefix="S"
) -> Tuple[List[Synapses], List[Dict[str, Any]]]:
    
    synE = SYNAPSE_MODELS[cfg["synapse_model"]]["liq_exc"]
    synI = SYNAPSE_MODELS[cfg["synapse_model"]]["liq_inh"]
    learning = LEARNING_RULES[cfg["learning_rule"]]

    p_pairs = cfg["p_liq_to_out_pairs"]
    g_pairs = cfg.get("gain_liq_to_out_pairs", None)
    gain_default = float(cfg.get("gain", 1.0))

    n_out = len(G_out_list)
    n_liq = len(G_liq)

    S_list: List[Synapses] = []
    meta: List[Dict[str, Any]] = []

    for out_idx, post in enumerate(G_out_list):
        typ_post_arr = np.asarray(post.typ)
        N_post_E = int(np.sum(typ_post_arr == 1))
        N_post_I = int(np.sum(typ_post_arr == -1))

        for liq_idx, pp in p_pairs[out_idx].items():
            liq_idx = int(liq_idx)
            pre = G_liq[liq_idx]

            pEE, pEI, pIE, pII = _read_pair_dict_with_layer_val(
                pp, out_idx, liq_idx, n_out=n_out, n_liq=n_liq, prefer="out"
            )

            # ---- gain ----
            if g_pairs is None:
                gEE = gEI = gIE = gII = gain_default
            else:
                gg = g_pairs[out_idx][liq_idx]
                gEE, gEI, gIE, gII = _read_pair_dict_with_layer_val(
                    {k: gg.get(k, gain_default) for k in ("EE", "EI", "IE", "II")},
                    out_idx, liq_idx, n_out=n_out, n_liq=n_liq, prefer="out"
                )

            # ===== Excitatory (learnable): EE + EI in ONE Synapses =====
            sE = Synapses(
                pre, post,
                model=synE["eqs"] + learning["eqs"],
                on_pre=synE["on_pre"] + learning["on_pre"],
                on_post=learning["on_post"],
                method="euler",
                name=f"{name_prefix}_E_liq{liq_idx+1}_to_out{out_idx+1}",
            )
            sE.connect(condition="typ_pre==1 and typ_post==1", p=pEE)
            sE.connect(condition="typ_pre==1 and typ_post==-1", p=pEI)

            idx_EE = np.where(np.asarray(sE.typ_post) == 1)[0]
            idx_EI = np.where(np.asarray(sE.typ_post) == -1)[0]

            wE = np.zeros(len(sE), dtype=float)
            if idx_EE.size:
                wE[idx_EE] = init_liq_to_out(rng, idx_EE.size, gain=gEE, N_post=N_post_E)
            if idx_EI.size:
                wE[idx_EI] = init_liq_to_out(rng, idx_EI.size, gain=gEI, N_post=N_post_I)
            setattr(sE, synE["w_attr"], wE)
            sE.delay = 0 * ms

            # ===== Inhibitory (fixed): IE + II in ONE Synapses =====
            sI = Synapses(
                pre, post,
                model=synI["eqs"],
                on_pre=synI["on_pre"],
                method="euler",
                name=f"{name_prefix}_I_liq{liq_idx+1}_to_out{out_idx+1}",
            )
            sI.connect(condition="typ_pre==-1 and typ_post==1", p=pIE)
            sI.connect(condition="typ_pre==-1 and typ_post==-1", p=pII)

            idx_IE = np.where(np.asarray(sI.typ_post) == 1)[0]
            idx_II = np.where(np.asarray(sI.typ_post) == -1)[0]

            wI = np.zeros(len(sI), dtype=float)
            if idx_IE.size:
                wI[idx_IE] = init_liq_to_out(rng, idx_IE.size, gain=gIE, N_post=N_post_E)
            if idx_II.size:
                wI[idx_II] = init_liq_to_out(rng, idx_II.size, gain=gII, N_post=N_post_I)
            setattr(sI, synI["w_attr"], wI)
            sI.delay = 0 * ms

            S_list.extend([sE, sI])
            meta.append({
                "pre_idx": liq_idx, "post_idx": out_idx,
                "S_exc": sE, "S_inh": sI,
                "p": {"EE": pEE, "EI": pEI, "IE": pIE, "II": pII},
                "gain": {"EE": gEE, "EI": gEI, "IE": gIE, "II": gII},
                "idx": {"EE": idx_EE, "EI": idx_EI, "IE": idx_IE, "II": idx_II},
            })

    return S_list, meta