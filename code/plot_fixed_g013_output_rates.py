# -*- coding: utf-8 -*-
from pathlib import Path
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from brian2 import (
    prefs, float64, ms, Hz, second, defaultclock, seed, start_scope,
    NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network,
)


prefs.core.default_float_dtype = float64
prefs.codegen.target = "numpy"

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DATA_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = str(DATA_ROOT) + "/"

WEIGHT_PREFIX = "SRDP_1"
REP = 1
G_TAG = "g0.13"

W_IN_PATH = SCRIPT_DIR / f"{WEIGHT_PREFIX}_w_in_rep{REP}_{G_TAG}.npy"
W_RES_PATH = SCRIPT_DIR / f"{WEIGHT_PREFIX}_w_res_rep{REP}_{G_TAG}.npy"
W_OUT_PATH = SCRIPT_DIR / f"{WEIGHT_PREFIX}_w_out_rep{REP}_{G_TAG}.npy"
SAMPLE_SEQ_PATH = SCRIPT_DIR / f"sample_seq_rep{REP}.npy"

OUT_DIR = SCRIPT_DIR / f"{WEIGHT_PREFIX}_{G_TAG}_fixed_output_rate"

DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]
DEFAULT_MATERIAL = "Al_board"

DISPLAY_NAME = {
    "Al_board": "aluminum bd.",
    "buta_omote": "outer_pigskin",
    "buta_ura": "back_pigskin",
    "cork": "cork",
    "denim": "denim",
    "rubber_board": "rubber bd.",
    "washi": "japanese paper",
    "wood_board": "wood bd.",
}

LINESTYLES = [
    "-", "--", "-.", ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
]

MARKERS = ["o", "s", "^", "D", "v", "x", "*", "+"]

COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

N_TRAIN = 1
N_SAMPLE_TEST = 1
WINDOW_MS = 25.0
TOTAL_MS = 500.0


def calc_meissner(data, t, dt):
    current = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            current[0, i] = current[0, i - 1] + dF_dt + (-current[0, i - 1] * dt / (8 * 1e-3))
            current[1, i] = (
                current[1, i - 1]
                + 0.24 * dF_dt
                + (-(current[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            )
            current[2, i] = current[2, i - 1] + 0.07 * dF_dt + (-current[2, i - 1] * dt / (1744.6 * 1e-3))
            current[3, i] = current[0, i]
    return current[3, :]


def calc_merkel(data, t, dt):
    current = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            if dF_dt < 0:
                dF_dt = 0
            current[0, i] = current[0, i - 1] + 0.74 * dF_dt + (-current[0, i - 1] * dt / (8 * 1e-3))
            current[1, i] = (
                current[1, i - 1]
                + 0.24 * dF_dt
                + (-(current[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            )
            current[2, i] = current[2, i - 1] + 0.07 * dF_dt + (-current[2, i - 1] * dt / (1744.6 * 1e-3))
            current[3, i] = current[0, i] + current[1, i] + current[2, i]
    return current[3, :]


def load_input_current(material: str, sample_id: int, n_in: int, dt_s: float):
    files = glob.glob(DATA_PATH + "tactile_data/" + material + f"/data_{sample_id}_*")
    if len(files) == 0:
        raise FileNotFoundError(f"data not found: material={material}, sample_id={sample_id}")

    df = pd.read_table(files[0], header=None)
    df_np = df.to_numpy().T
    in_data_0 = df_np[:3, 3000:8000]
    nt = in_data_0.shape[1]
    t_array_s = np.arange(nt) * dt_s

    input_current = np.zeros((n_in, nt), dtype=float)
    in_data = in_data_0[0, :]
    i_merkel = calc_merkel(in_data, t_array_s, dt_s)
    i_meissner = calc_meissner(in_data, t_array_s, dt_s)
    input_current[0, :] = 0.4 * i_merkel * 0.02
    input_current[1, :] = 0.6 * 7.3 * i_meissner * 0.02
    return input_current, nt


def plot_results(rate_df: pd.DataFrame, materials: list[str], run_label: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, material in enumerate(materials):
        mat_df = rate_df[rate_df["material"] == material]
        ax.plot(
            mat_df["time_ms"],
            mat_df["output_rate_hz"],
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(mat_df) // 8),
            linewidth=1.8,
            markersize=5,
            label=DISPLAY_NAME.get(material, material),
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean firing rate of output layer (Hz)")
    ax.set_title(f"SRDP {G_TAG}")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{WEIGHT_PREFIX}_{G_TAG}_{run_label}_output_rate.png", dpi=200)
    plt.close(fig)

    for i, material in enumerate(materials):
        mat_df = rate_df[rate_df["material"] == material]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            mat_df["time_ms"],
            mat_df["output_rate_hz"],
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(mat_df) // 8),
            linewidth=1.8,
            markersize=5,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Mean firing rate of output layer (Hz)")
        ax.set_title(f"SRDP {G_TAG} | {DISPLAY_NAME.get(material, material)}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{WEIGHT_PREFIX}_{G_TAG}_{material}_output_rate.png", dpi=200)
        plt.close(fig)


def save_output_raster(material: str, sample_id: int, spike_time_ms: np.ndarray, spike_ids: np.ndarray):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(spike_time_ms, spike_ids, s=6, color="black", alpha=0.75, linewidths=0)
    ax.set_xlim(0, TOTAL_MS)
    ax.set_ylim(-0.5, max(0.5, float(np.max(spike_ids) + 0.5) if len(spike_ids) else 39.5))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("output neuron index")
    ax.set_title(f"SRDP {G_TAG} | {DISPLAY_NAME.get(material, material)} | sample {sample_id}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{WEIGHT_PREFIX}_{G_TAG}_{material}_sample{sample_id}_output_raster.png", dpi=200)
    plt.close(fig)


def save_sample_output_rate_plot(material: str,
                                 sample_id: int,
                                 time_ms: np.ndarray,
                                 output_rate_hz: np.ndarray):
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(time_ms, output_rate_hz, linewidth=2.2, color="tab:blue")
    ax.set_xlim(0, TOTAL_MS)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean firing rate of output layer (Hz)")
    ax.set_title(f"SRDP {G_TAG} | {DISPLAY_NAME.get(material, material)} | sample {sample_id}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{WEIGHT_PREFIX}_{G_TAG}_{material}_sample{sample_id}_output_rate.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material",
        default=DEFAULT_MATERIAL,
        choices=DIR_NAME + ["all"],
        help="Material to simulate. Use 'all' to run every material.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLE_TEST,
        help="Number of test samples to average.",
    )
    parser.add_argument(
        "--save-raster",
        action="store_true",
        help="Save output-layer raster plots for each simulated material/sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    materials = DIR_NAME if args.material == "all" else [args.material]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [W_IN_PATH, W_RES_PATH, W_OUT_PATH, SAMPLE_SEQ_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    w_in = np.load(W_IN_PATH)
    w_res_init = np.load(W_RES_PATH)
    w_out_init = np.load(W_OUT_PATH)
    sample_seq = np.load(SAMPLE_SEQ_PATH).astype(int)
    test_seq = sample_seq[N_TRAIN:N_TRAIN + args.n_samples]

    n_in, n_res = w_in.shape
    n_out = w_out_init.shape[1]
    if w_res_init.shape != (n_res, n_res):
        raise ValueError(f"w_res shape mismatch: {w_res_init.shape}")
    if w_out_init.shape[0] != n_res:
        raise ValueError(f"w_out shape mismatch: {w_out_init.shape}")

    start_scope()
    seed(2 + (REP - 1))
    rng = np.random.default_rng(2 + (REP - 1))

    v_reset = -65
    v_thr = -40
    tau_r = 2 * ms
    tau_d = 20 * ms
    bias = -65
    g_res = 0.25
    dt_ms = 0.1
    dt_s = dt_ms * 1e-3
    defaultclock.dt = dt_ms * ms

    lif = """
    dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
    I_exc : 1
    I_inh : 1
    tau_m : second
    t_ref : second
    """
    double_exp_res = """
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = G * R : 1
    """
    double_exp_out = """
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = R : 1
    """
    on_pre_res = "H_post += (w_res / (tau_r * tau_d)) / Hz"
    on_pre_out = "H_post += (w_out / (tau_r * tau_d)) / Hz"

    g_liquid = NeuronGroup(
        n_res,
        double_exp_res + lif,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    g_liquid.tau_m = 10 * ms
    g_liquid.t_ref = 2 * ms
    g_liquid.I_inh = 0

    g_output = NeuronGroup(
        n_out,
        double_exp_out + lif,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    g_output.tau_m = 10 * ms
    g_output.t_ref = 2 * ms
    g_output.I_inh = 0

    s_res = Synapses(g_liquid, g_liquid, model="w_res : 1", on_pre=on_pre_res, method="euler")
    s_res.connect(condition="i != j")
    s_res.w_res = w_res_init[s_res.i, s_res.j]
    s_res.delay = 0 * ms

    pre_idx, post_idx = np.where(w_out_init != 0)
    s_out = Synapses(g_liquid, g_output, model="w_out : 1", on_pre=on_pre_out, method="euler")
    s_out.connect(i=pre_idx, j=post_idx)
    s_out.w_out = w_out_init[s_out.i, s_out.j]
    s_out.delay = 0 * ms

    input_ta = TimedArray(np.zeros((1, n_in)), dt=dt_ms * ms)
    g_input = NeuronGroup(
        n_in,
        """
        t_start : second (shared)
        I = input_ta(t - t_start, i) : 1
        """,
        method="euler",
    )
    g_input.t_start = 0 * ms

    s_in = Synapses(
        g_input,
        g_liquid,
        model="""
        w : 1
        I_exc_post = w * I_pre : 1 (summed)
        """,
        method="euler",
    )
    pre_in, post_in = np.nonzero(w_in)
    s_in.connect(i=pre_in, j=post_in)
    s_in.w = w_in[pre_in, post_in]

    spike_out = SpikeMonitor(g_output)
    net = Network(g_input, g_liquid, g_output, s_in, s_res, s_out, spike_out)

    bin_edges = np.arange(0.0, TOTAL_MS + WINDOW_MS, WINDOW_MS)
    x_ms = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rate_sums = {material: np.zeros(len(x_ms), dtype=float) for material in materials}
    material_sample_ids = {material: [] for material in materials}

    t0 = 0 * ms
    ns = {
        "input_ta": input_ta,
        "tau_r": tau_r,
        "tau_d": tau_d,
        "G": g_res,
        "BIAS": bias,
        "v_reset": v_reset,
        "v_thr": v_thr,
    }

    for material in materials:
        for sample_id in tqdm(test_seq, desc=material):
            material_sample_ids[material].append(int(sample_id))
            input_current, nt = load_input_current(material, int(sample_id), n_in, dt_s)
            vals = input_current.T
            vals = np.vstack([vals, vals[-1]])
            input_ta = TimedArray(vals, dt=dt_ms * ms)
            ns["input_ta"] = input_ta
            g_input.t_start = t0

            g_liquid.v = v_reset + (v_thr - v_reset) * rng.random(n_res)
            g_output.v = v_reset + (v_thr - v_reset) * rng.random(n_out)
            g_liquid.R = 0
            g_liquid.H = 0
            g_output.R = 0
            g_output.H = 0

            start_t = t0
            start_idx = len(spike_out.t)
            duration = (nt * dt_ms) * ms
            net.run(duration, namespace=ns)
            end_idx = len(spike_out.t)
            t0 += duration

            if end_idx <= start_idx:
                continue

            rel_ms = np.asarray((spike_out.t[start_idx:end_idx] - start_t) / ms, dtype=float)
            spike_ids = np.asarray(spike_out.i[start_idx:end_idx], dtype=int)
            counts, _ = np.histogram(rel_ms, bins=bin_edges)
            sample_rate_hz = counts / (n_out * (WINDOW_MS / 1000.0))
            rate_sums[material] += sample_rate_hz

            if args.save_raster:
                save_output_raster(material, int(sample_id), rel_ms, spike_ids)
                save_sample_output_rate_plot(material, int(sample_id), x_ms, sample_rate_hz)

    rows = []
    for material in materials:
        mean_rate = rate_sums[material] / len(test_seq)
        for time_ms, rate_hz in zip(x_ms, mean_rate):
            rows.append({
                "material": material,
                "time_ms": float(time_ms),
                "output_rate_hz": float(rate_hz),
                "n_samples": int(len(test_seq)),
                "sample_ids": " ".join(str(sid) for sid in material_sample_ids[material]),
                "weight_prefix": WEIGHT_PREFIX,
                "g_tag": G_TAG,
            })

    rate_df = pd.DataFrame(rows)
    run_label = args.material if args.material != "all" else "all_materials"
    csv_path = OUT_DIR / f"{WEIGHT_PREFIX}_{G_TAG}_{run_label}_output_rate_time_transition.csv"
    rate_df.to_csv(csv_path, index=False)
    plot_results(rate_df, materials, run_label)
    print(f"[saved] {csv_path}")
    print(f"[saved] plots under {OUT_DIR}")


if __name__ == "__main__":
    main()
