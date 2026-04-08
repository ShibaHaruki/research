import numpy as np
from brian2 import *

CONNECTIONS: dict[str, dict[str, callable]] = {}

def layer_val(v, idx):
    return v[idx] if isinstance(v, (list, tuple, np.ndarray)) else v

def register(kind: str, name: str):
    def deco(fn):
        CONNECTIONS.setdefault(kind, {})[name] = fn
        return fn
    return deco

def get_connection(kind: str, name: str):
    if kind not in CONNECTIONS:
        raise KeyError(f"Unknown kind='{kind}'. kinds={list(CONNECTIONS.keys())}")
    if name not in CONNECTIONS[kind]:
        raise KeyError(
            f"Unknown connection '{name}' for kind='{kind}'. available={list(CONNECTIONS[kind].keys())}"
        )
    return CONNECTIONS[kind][name]

# --- ペアごとの条件 ---
_PAIR_COND = {
    "EE": "(i!=j) and (typ_pre==1)  and (typ_post==1)",
    "EI": "(i!=j) and (typ_pre==1)  and (typ_post==-1)",
    "IE": "(i!=j) and (typ_pre==-1) and (typ_post==1)",
    "II": "(i!=j) and (typ_pre==-1) and (typ_post==-1)",
}

def _read_intra_p(cfg: dict, layer_index: int):
    """
    優先順位：
      1) cfg["p_liq_intra_pairs"] = {"EE":..., ...} （あなたの今のcfg）
      2) cfg["p_EE"], cfg["p_EI"], cfg["p_IE"], cfg["p_II"]
      3) cfg["p_liq_intra"] or cfg["p_liq"]（共通）
    """
    if "p_liq_intra_pairs" in cfg:
        pp = cfg["p_liq_intra_pairs"]
        return (
            float(layer_val(pp["EE"], layer_index)),
            float(layer_val(pp["EI"], layer_index)),
            float(layer_val(pp["IE"], layer_index)),
            float(layer_val(pp["II"], layer_index)),
        )

    # fallback: 個別キー
    if all(k in cfg for k in ("p_EE", "p_EI", "p_IE", "p_II")):
        return (
            float(layer_val(cfg["p_EE"], layer_index)),
            float(layer_val(cfg["p_EI"], layer_index)),
            float(layer_val(cfg["p_IE"], layer_index)),
            float(layer_val(cfg["p_II"], layer_index)),
        )

    # fallback: 共通
    p_common = cfg.get("p_liq_intra", cfg.get("p_liq", 0.1))
    p_common = float(layer_val(p_common, layer_index))
    return p_common, p_common, p_common, p_common


@register("liq_intra", "random")
def connect_intra_random(
    s: Synapses, g, cfg: dict, layer_index: int, rng: np.random.Generator,
    pairs=("EE", "EI", "IE", "II")
):
    pEE, pEI, pIE, pII = _read_intra_p(cfg, layer_index)
    p_map = {"EE": pEE, "EI": pEI, "IE": pIE, "II": pII}

    for key in pairs:
        s.connect(condition=_PAIR_COND[key], p=p_map[key])


@register("liq_intra", "distance")
def connect_intra_distance(
    s: Synapses, g, cfg: dict, layer_index: int, rng: np.random.Generator,
    pairs=("EE", "EI", "IE", "II")
):
    lam = float(layer_val(cfg.get("lam", 1.0), layer_index))
    dist = "sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2)"

    pEE, pEI, pIE, pII = _read_intra_p(cfg, layer_index)
    p_map = {"EE": pEE, "EI": pEI, "IE": pIE, "II": pII}

    for key in pairs:
        s.connect(condition=_PAIR_COND[key], p=f"{p_map[key]}*exp(-({dist})/{lam})")