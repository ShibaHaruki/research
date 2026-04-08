# a.py
"""
統合版 build_network.py 用デバッグスクリプト（重み分布も保存）
- グラフは表示せず PNG と npz を保存
- 統合版（興奮1本＋抑制1本）なら、S_intra / S_lo を全部 Network に入れてOK
"""

import sys
from pathlib import Path
import numpy as np

from brian2 import (
    prefs, start_scope, defaultclock, ms, seed,
    TimedArray, SpikeMonitor, StateMonitor, Network, BrianLogger
)

# ===== matplotlib (表示しない) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

prefs.codegen.target = "numpy"

# 害のない warning 抑制（必要なければ消してOK）
BrianLogger.suppress_name("brian2.groups.group.Group.resolve.resolution_conflict")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MODELS_DIR = HERE / "models"
if MODELS_DIR.exists() and str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from build_network import (
    make_in_neuron_group,
    make_liquid_neuron_groups,
    make_output_neuron_groups,
    make_in_to_liq_synapses,
    make_liq_intra_synapses,
    make_liq_to_out_synapses,
)


def build_min_cfg():
    filters = ["merkel", "meissner"]

    # ★入力Synapsesを統合していない場合、NUM_CHANNEL>1 で input側の(summed)衝突が起きます。
    # まず動作確認なら 1 のままが安全。入力統合済みなら 4 に上げてOK。
    num_channel = 1

    cfg = {
        "dt_ms": 0.1,
        "BASE_SEED": 123,

        "neuron_model": "LIF",
        "synapse_model": "double_exp",
        "learning_rule": "off",

        "liquid_layer": 2,
        "output_layer": 1,
        "N_liq": [10,1000],
        "N_out": [100],

        "r_inh_liq": 0.2,
        "r_inh_out": 0.2,

        "tau_exc": 10.0,
        "tau_inh": 2.0,
        "ref_exc": 2.0,
        "ref_inh": 1.0,
        "bias": 0.0,
        "v_thr": -45.0,
        "v_reset": -65.0,

        # double_exp が参照する外部変数
        "tau_r": 2.0,
        "tau_d": 20.0,

        "USE_INPUT_FILTERS": filters,
        "NUM_CHANNEL": num_channel,

        # 全ch×filter を layer0 へ
        "IN_ROUTE": {
            (ch, f): {"layers": {0: {"p": 1.0, "scale": 20.0},
                                 1: {"p": 1.0, "scale": 100.0},
                                }}
            
            for ch in range(num_channel)
            for f in filters
            
        },

        "liq_intra_connection": "random",
        "p_liq_intra_pairs": {"EE": 0.1, "EI": 0.1, "IE": 0.1, "II": 0.1},
        "liq_intra_gain_pairs": {"EE": 1.0, "EI": 1.0, "IE": 1.0, "II": 1.0},

        "p_liq_to_out_pairs": [ {
                                0: {"EE": 0.3, "EI": 0.3, "IE": 0.3, "II": 0.3},
                                1: {"EE": 0.3, "EI": 0.3, "IE": 0.3, "II": 0.3}
                                }
                                ],
        "gain": 1.0,
    }
    return cfg


def make_dummy_input(cfg, T_ms=200.0, amp=5.0):
    dt_ms = float(cfg["dt_ms"])
    steps = int(round(T_ms / dt_ms))

    filters = cfg["USE_INPUT_FILTERS"]
    S = len(filters)
    C = int(cfg["NUM_CHANNEL"])
    N_in = C * S

    X = np.zeros((steps, N_in), dtype=float)

    def to_step(ms_val: float) -> int:
        return int(round(ms_val / dt_ms))

    t0, t1 = to_step(20.0), to_step(60.0)     # merkel
    t2, t3 = to_step(100.0), to_step(140.0)   # meissner

    for ch in range(C):
        for mod, f in enumerate(filters):
            i = ch * S + mod
            if f == "merkel":
                X[t0:t1, i] = amp
            elif f == "meissner":
                X[t2:t3, i] = amp

    return X


def inject_timedarray(G_in, X, cfg):
    dt = float(cfg["dt_ms"]) * ms
    ta = TimedArray(X, dt=dt)
    G_in.namespace["input_ta"] = ta
    G_in.t_start = defaultclock.t
    return ta


def set_synapse_time_constants(syn_list, cfg):
    tau_r = float(cfg["tau_r"]) * ms
    tau_d = float(cfg["tau_d"]) * ms
    for s in syn_list:
        s.namespace["tau_r"] = tau_r
        s.namespace["tau_d"] = tau_d


def save_input_plot(out_dir: Path, X: np.ndarray, cfg: dict, max_cols=8):
    dt_ms = float(cfg["dt_ms"])
    t = np.arange(X.shape[0]) * dt_ms
    cols = min(X.shape[1], max_cols)

    plt.figure()
    for i in range(cols):
        plt.plot(t, X[:, i])
    plt.xlabel("Time (ms)")
    plt.ylabel("Input amplitude")
    plt.title(f"Dummy input (first {cols}/{X.shape[1]})")
    plt.tight_layout()
    plt.savefig(out_dir / "input.png", dpi=150)
    plt.close()


def save_spike_raster(out_fp: Path, M: SpikeMonitor, title: str):
    t_ms = np.asarray(M.t / ms)
    i = np.asarray(M.i)

    plt.figure()
    if len(t_ms) > 0:
        plt.scatter(t_ms, i, s=3)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title(f"{title} | spikes={M.num_spikes}")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def save_voltage_plot(out_fp: Path, V: StateMonitor, title: str):
    t_ms = np.asarray(V.t / ms)
    plt.figure()
    for k in range(V.v.shape[0]):
        plt.plot(t_ms, np.asarray(V.v[k]), label=f"n{k}")
    plt.xlabel("Time (ms)")
    plt.ylabel("v")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def safe_key(s: str) -> str:
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def get_weights(s):
    if hasattr(s, "w"):
        return np.asarray(s.w, dtype=float).reshape(-1)
    return None


def save_weight_hist(out_fp: Path, w: np.ndarray, title: str, bins: int = 80):
    plt.figure()
    if w.size > 0:
        plt.hist(w, bins=bins)
    plt.xlabel("w")
    plt.ylabel("count")
    plt.title(f"{title} | n={w.size}")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def summarize(w: np.ndarray) -> dict:
    if w.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(w.size),
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
    }


def save_all_weight_distributions(out_dir: Path, S_in, S_intra, S_lo, tag: str):
    wdir = out_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    all_arrays = {}
    lines = [f"[{tag}] weight summary\n"]

    def handle(name: str, syn_list):
        ws = [get_weights(s) for s in syn_list if get_weights(s) is not None]
        w = np.concatenate(ws) if ws else np.array([], dtype=float)

        st = summarize(w)
        lines.append(
            f"{name}: n={st['n']} mean={st['mean']:.6g} std={st['std']:.6g} "
            f"min={st['min']:.6g} max={st['max']:.6g}\n"
        )
        all_arrays[safe_key(f"{tag}__{name}")] = w
        save_weight_hist(wdir / f"hist_{safe_key(tag)}__{safe_key(name)}.png", w, f"{tag} | {name}")

    # input: 名前ごと
    in_by_name = {}
    for s in S_in:
        in_by_name.setdefault(s.name, []).append(s)
    for name, lst in sorted(in_by_name.items()):
        handle(f"in::{name}", lst)

    # intra / liq2out: 統合版なら E と I の2本だけのはず（名前で保存）
    for s in S_intra:
        handle(f"intra::{s.name}", [s])
    for s in S_lo:
        handle(f"liq2out::{s.name}", [s])

    np.savez_compressed(wdir / f"weights_all_{tag}.npz", **all_arrays)
    (wdir / f"weights_summary_{tag}.txt").write_text("".join(lines), encoding="utf-8")


def main():
    out_dir = HERE / "debug_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_min_cfg()

    start_scope()
    defaultclock.dt = float(cfg["dt_ms"]) * ms

    base_seed = int(cfg["BASE_SEED"])
    np.random.seed(base_seed)
    seed(base_seed)
    rng = np.random.default_rng(base_seed)

    # --- build network objects ---
    G_in, input_ta0 = make_in_neuron_group(cfg)
    G_in.namespace["input_ta"] = input_ta0

    G_liq = make_liquid_neuron_groups(cfg, rng)
    G_out = make_output_neuron_groups(cfg, rng)

    S_in, _ = make_in_to_liq_synapses(G_in, G_liq, rng, cfg)
    S_intra, _ = make_liq_intra_synapses(G_liq, rng, cfg)      # 統合版なら [E,I]
    S_lo, _ = make_liq_to_out_synapses(G_liq, G_out, rng, cfg)  # 統合版なら [E,I]

    set_synapse_time_constants(S_intra + S_lo, cfg)

    # ---- 重み分布（初期値）保存 ----
    save_all_weight_distributions(out_dir, S_in, S_intra, S_lo, tag="init")

    # --- monitors ---
    M_liq = SpikeMonitor(G_liq[0], name="M_liq")
    M_out = SpikeMonitor(G_out[0], name="M_out")
    V_liq = StateMonitor(G_liq[0], "v", record=[0, 1, 2], dt=1 * ms, name="V_liq")

    # --- input ---
    X = make_dummy_input(cfg, T_ms=200.0, amp=5.0)
    input_ta = inject_timedarray(G_in, X, cfg)
    save_input_plot(out_dir, X, cfg)

    # --- explicit Network（統合版なので全部入れる）---
    net = Network()
    net.add(G_in)
    net.add(*G_liq)
    net.add(*G_out)
    net.add(*(S_in + S_intra + S_lo))
    net.add(M_liq, M_out, V_liq)

    net.run(200.0 * ms, namespace={"input_ta": input_ta})

    # ---- 重み分布（最終値）保存（学習ONなら差が出る）----
    save_all_weight_distributions(out_dir, S_in, S_intra, S_lo, tag="final")

    # --- save plots ---
    save_spike_raster(out_dir / "spike_raster_liquid.png", M_liq, "Liquid raster")
    save_spike_raster(out_dir / "spike_raster_output.png", M_out, "Output raster")
    save_voltage_plot(out_dir / "voltage_liquid.png", V_liq, "Liquid v")

    # --- save raw ---
    np.savez_compressed(out_dir / "monitors_liquid_spikes.npz",
                        t_ms=np.asarray(M_liq.t / ms), i=np.asarray(M_liq.i),
                        num_spikes=int(M_liq.num_spikes))
    np.savez_compressed(out_dir / "monitors_output_spikes.npz",
                        t_ms=np.asarray(M_out.t / ms), i=np.asarray(M_out.i),
                        num_spikes=int(M_out.num_spikes))
    np.savez_compressed(out_dir / "monitors_liquid_voltage.npz",
                        t_ms=np.asarray(V_liq.t / ms), v=np.asarray(V_liq.v))

    print("\n=== Dummy run finished ===")
    print(f"Saved to: {out_dir}")
    print(f"Liquid spikes: {M_liq.num_spikes}")
    print(f"Output spikes: {M_out.num_spikes}")
    print(f"Weight hists: {out_dir/'weights'}")


if __name__ == "__main__":
    main()