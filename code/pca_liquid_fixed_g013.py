# -*- coding: utf-8 -*-
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DATA_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = str(DATA_ROOT) + "/"

REP = 1

SAMPLE_SEQ_PATH = SCRIPT_DIR / f"sample_seq_rep{REP}.npy"

OUT_DIR = SCRIPT_DIR / "liquid_pca_results"

DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]

DISPLAY_NAMES = [
    "aluminum bd.",
    "outer_pigskin",
    "back_pigskin",
    "cork",
    "denim",
    "rubber bd.",
    "japanese paper",
    "wood bd.",
]

MARKERS = ["o", "s", "^", "D", "v", "x", "*", "+"]
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

N_TRAIN = 50
N_SAMPLE = 1
N_IN = 3
N_RES = 3000
P_IN = 0.2
P_RES = 0.5
TOTAL_MS = 500.0
DEFAULT_T_N = 25
DEFAULT_CODEGEN_TARGET = "cython"
CYTHON_CACHE_DIR = SCRIPT_DIR / "brian2_cython_cache"
LIQUID_REC_PATH = SCRIPT_DIR / f"liquid_sout_rec_rep{REP}.npy"
USE_DYNAMIC_SYNAPSE = False
STP_U = 0.2
STP_TAU_REC_MS = 200.0
STP_TAU_FAC_MS = 1500.0


def calc_meissner(data, t, dt):
    current = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 25]) / (t[i] - t[i - 25])
            current[0, i] = current[0, i - 1] + dF_dt +  (-current[0, i - 1] * dt / (10 * 1e-3))
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
            dF_dt = np.abs(data[i] - data[i - 25]) / (t[i] - t[i - 25])
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


def calc_dF_dt(data, t, lag: int = 25):
    dF_dt = np.zeros(len(t), dtype=float)
    for i in range(lag, len(t)):
        dt = t[i] - t[i - lag]
        if dt > 0:
            dF_dt[i] = np.abs(data[i] - data[i - lag]) / dt
    return dF_dt


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
    dF_dt = calc_dF_dt(in_data, t_array_s, lag=25)
    input_current[0, :] = 0.4 * i_merkel * 0.02*10
    input_current[1, :] = 6 * 7.3 * i_meissner * 0.02
    input_current[2, :] = dF_dt*5
    return input_current, nt


def liquid_spikes_to_feature(spike_t_ms, spike_i, n_res: int, t_n: int) -> np.ndarray:
    n_interval = int(TOTAL_MS // t_n)
    feature = np.zeros((n_res, n_interval), dtype=np.float32)
    if len(spike_t_ms) == 0:
        return feature.reshape(-1)

    bins = np.floor(spike_t_ms / t_n).astype(int)
    valid = (bins >= 0) & (bins < n_interval)
    if np.any(valid):
        np.add.at(feature, (spike_i[valid], bins[valid]), 1.0)
    return feature.reshape(-1)


def liquid_spikes_to_rec(spike_t_ms, spike_i, n_res: int, n_bins: int = 500) -> np.ndarray:
    rec = np.zeros((n_res, n_bins), dtype=np.uint16)
    if len(spike_t_ms) == 0:
        return rec

    bin_edges = np.linspace(0, TOTAL_MS, n_bins + 1)
    valid = (spike_t_ms >= 0) & (spike_t_ms <= TOTAL_MS)
    if np.any(valid):
        bins = np.searchsorted(bin_edges, spike_t_ms[valid], side="right") - 1
        bins = np.clip(bins, 0, n_bins - 1)
        np.add.at(rec, (spike_i[valid], bins), 1)
    return rec


def make_liquid_weights(rng: np.random.Generator):
    w_in_raw = rng.lognormal(mean=0.0, sigma=1.0, size=(N_IN, N_RES))
    w_in_raw *= rng.choice([-1.0, 1.0], size=(N_IN, N_RES))
    w_in_raw /= np.std(w_in_raw)
    w_in = w_in_raw * (rng.random((N_IN, N_RES)) < P_IN) / np.sqrt(N_IN * P_IN)

    variance = (N_RES * P_RES**2) ** -1
    w_res_raw = rng.lognormal(mean=0.0, sigma=1.0, size=(N_RES, N_RES))
    w_res_raw *= rng.choice([-1.0, 1.0], size=(N_RES, N_RES))
    w_res_raw /= np.std(w_res_raw)
    w_res = w_res_raw * (rng.random((N_RES, N_RES)) < P_RES) * np.sqrt(variance)
    for k in range(N_RES):
        connected = np.where(np.abs(w_res[:, k]) > 0)[0]
        if len(connected) > 0:
            w_res[connected, k] -= np.mean(w_res[connected, k])

    return w_in, w_res


def compute_pca(features: np.ndarray, n_components: int = 3):
    x = features.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x = x / std

    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    pca = x @ vt[:n_components].T
    explained = (singular_values[:n_components] ** 2) / np.sum(singular_values ** 2)
    return pca, explained


def plot_pca(pca2: np.ndarray,
             labels: np.ndarray,
             t_n: int,
             n_sample: int,
             explained,
             dynamic_synapse: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for i, material in enumerate(DIR_NAME):
        mask = labels == i
        ax.scatter(
            pca2[mask, 0],
            pca2[mask, 1],
            c=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            s=35,
            linewidths=1.0,
            label=DISPLAY_NAMES[i],
            alpha=0.8,
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    synapse_label = "dynamic synapse" if dynamic_synapse else "static synapse"
    ax.set_title(f"Liquid layer PCA | {synapse_label} | T_n={t_n} | n={n_sample}")
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()

    dyn_tag = "dynamic" if dynamic_synapse else "static"
    out_base = OUT_DIR / f"liquid_pca_{dyn_tag}_Tn{t_n}_n{n_sample}"
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_base.with_suffix('.png')}")
    print(f"[saved] {out_base.with_suffix('.pdf')}")


def plot_pca_3d(pca3: np.ndarray,
                labels: np.ndarray,
                t_n: int,
                n_sample: int,
                explained,
                dynamic_synapse: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    for i, material in enumerate(DIR_NAME):
        mask = labels == i
        ax.scatter(
            pca3[mask, 0],
            pca3[mask, 1],
            pca3[mask, 2],
            c=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            s=28,
            label=DISPLAY_NAMES[i],
            alpha=0.8,
            depthshade=False,
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)")
    synapse_label = "dynamic synapse" if dynamic_synapse else "static synapse"
    ax.set_title(f"Liquid layer PCA 3D | {synapse_label} | T_n={t_n} | n={n_sample}")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    dyn_tag = "dynamic" if dynamic_synapse else "static"
    out_base = OUT_DIR / f"liquid_pca_3d_{dyn_tag}_Tn{t_n}_n{n_sample}"
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_base.with_suffix('.png')}")
    print(f"[saved] {out_base.with_suffix('.pdf')}")


def save_variance_summary(features: np.ndarray,
                          labels: np.ndarray,
                          t_n: int,
                          n_sample: int,
                          dynamic_synapse: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class_means = []
    rows = []
    for i, material in enumerate(DIR_NAME):
        class_features = features[labels == i]
        class_mean = class_features.mean(axis=0)
        class_means.append(class_mean)
        sq_dist = np.sum((class_features - class_mean) ** 2, axis=1)
        rows.append({
            "material": material,
            "n_samples": int(class_features.shape[0]),
            "within_variance_mean_sq_dist": float(np.mean(sq_dist)),
            "within_variance_std_sq_dist": float(np.std(sq_dist)),
        })

    class_means = np.stack(class_means, axis=0)
    global_mean = features.mean(axis=0)
    between_sq_dist = np.sum((class_means - global_mean) ** 2, axis=1)
    between_variance = float(np.mean(between_sq_dist))

    summary_df = pd.DataFrame(rows)
    summary_df["between_variance_mean_sq_dist"] = between_variance
    summary_df["fisher_ratio_between_over_within"] = (
        between_variance / summary_df["within_variance_mean_sq_dist"]
    )

    dyn_tag = "dynamic" if dynamic_synapse else "static"
    csv_path = OUT_DIR / f"liquid_variance_{dyn_tag}_Tn{t_n}_n{n_sample}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["within_variance_mean_sq_dist"], color=COLORS[:len(summary_df)])
    ax.axhline(between_variance, color="black", linestyle="--", linewidth=1.6, label="between-class variance")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["material"], rotation=35, ha="right")
    ax.set_ylabel("mean squared distance")
    ax.set_title(f"Liquid feature variance | T_n={t_n} | n={n_sample}")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / f"liquid_variance_{dyn_tag}_Tn{t_n}_n{n_sample}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_liquid_raster(material: str,
                       sample_id: int,
                       spike_t_ms: np.ndarray,
                       spike_i: np.ndarray,
                       n_res: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(spike_t_ms, spike_i, s=2, color="black", alpha=0.65, linewidths=0)
    ax.set_xlim(0, TOTAL_MS)
    ax.set_ylim(-0.5, n_res - 0.5)
    ax.set_yticks(np.arange(0, n_res, 100))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("liquid neuron index")
    ax.set_title(f"{material} | sample {sample_id} | liquid raster")
    fig.tight_layout()
    out_path = OUT_DIR / f"liquid_raster_{material}_sample{sample_id}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_liquid_rate_plot(material: str,
                          sample_id: int,
                          spike_t_ms: np.ndarray,
                          n_res: int,
                          t_n: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_edges = np.arange(0.0, TOTAL_MS + t_n, t_n, dtype=float)
    if bin_edges[-1] != TOTAL_MS:
        bin_edges = np.append(bin_edges, TOTAL_MS)

    counts, _ = np.histogram(spike_t_ms, bins=bin_edges)
    bin_width_s = np.diff(bin_edges) / 1000.0
    rate_hz = counts / (n_res * bin_width_s)
    time_ms = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(time_ms, rate_hz, linewidth=2.2, color="tab:blue")
    ax.set_xlim(0, TOTAL_MS)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean firing rate of liquid layer (Hz)")
    ax.set_title(f"{material} | sample {sample_id} | liquid rate")
    fig.tight_layout()
    out_path = OUT_DIR / f"liquid_rate_{material}_sample{sample_id}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def generate_liquid_features(n_sample: int,
                             t_n: int,
                             codegen_target: str,
                             save_sample_plots: bool,
                             save_liquid_rec: bool,
                             dynamic_synapse: bool):
    from brian2 import (
        prefs, float64, ms, Hz, defaultclock, seed, start_scope,
        NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network,
    )

    prefs.core.default_float_dtype = float64
    prefs.codegen.target = codegen_target
    if codegen_target == "cython":
        CYTHON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prefs.codegen.runtime.cython.cache_dir = str(CYTHON_CACHE_DIR)
        prefs.codegen.runtime.cython.multiprocess_safe = True

    for path in [SAMPLE_SEQ_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    sample_seq = np.load(SAMPLE_SEQ_PATH).astype(int)
    test_seq = sample_seq[N_TRAIN:N_TRAIN + n_sample]

    start_scope()
    np.random.seed(2 + (REP - 1))
    rng = np.random.default_rng(2 + (REP - 1))
    seed(2 + (REP - 1))
    w_in, w_res_init = make_liquid_weights(rng)
    n_in, n_res = w_in.shape

    v_reset = -65
    v_thr = -40
    tau_r = 2 * ms
    tau_d = 20 * ms
    bias = -65
    gain = 0.25
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
    on_pre_res_static = "H_post += (w_res / (tau_r * tau_d)) / Hz"
    dynamic_synapse_model = """
    w_res : 1
    du_stp/dt = (U_stp - u_stp) / tau_fac_stp : 1 (event-driven)
    dx_stp/dt = (1 - x_stp) / tau_rec_stp : 1 (event-driven)
    """
    on_pre_res_dynamic = """
    u_stp += U_stp * (1 - u_stp)
    H_post += ((w_res * u_stp * x_stp) / (tau_r * tau_d)) / Hz
    x_stp -= u_stp * x_stp
    """

    g_res = NeuronGroup(
        n_res, double_exp_res + lif,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    g_res.tau_m = 10 * ms
    g_res.t_ref = 2 * ms
    g_res.I_inh = 0

    if dynamic_synapse:
        s_res = Synapses(
            g_res,
            g_res,
            model=dynamic_synapse_model,
            on_pre=on_pre_res_dynamic,
            method="euler",
        )
    else:
        s_res = Synapses(
            g_res,
            g_res,
            model="w_res : 1",
            on_pre=on_pre_res_static,
            method="euler",
        )
    s_res.connect(condition="i != j")
    s_res.w_res = w_res_init[s_res.i, s_res.j]
    s_res.delay = 0 * ms
    if dynamic_synapse:
        s_res.u_stp = STP_U
        s_res.x_stp = 1.0

    input_ta = TimedArray(np.zeros((1, n_in)), dt=dt_ms * ms)
    g_in = NeuronGroup(
        n_in,
        """
        t_start : second (shared)
        I = input_ta(t - t_start, i) : 1
        """,
        method="euler",
    )
    g_in.t_start = 0 * ms

    s_in = Synapses(
        g_in, g_res,
        model="""
        w : 1
        I_exc_post = w * I_pre : 1 (summed)
        """,
        method="euler",
    )
    pre_in, post_in = np.nonzero(w_in)
    s_in.connect(i=pre_in, j=post_in)
    s_in.w = w_in[pre_in, post_in]

    spike_res = SpikeMonitor(g_res)
    net = Network(g_in, g_res, s_in, s_res, spike_res)

    n_interval = int(TOTAL_MS // t_n)
    features = np.zeros((len(DIR_NAME) * n_sample, n_res * n_interval), dtype=np.float32)
    labels = np.zeros(len(DIR_NAME) * n_sample, dtype=int)
    liquid_rec = None
    if save_liquid_rec:
        liquid_rec = np.zeros((len(DIR_NAME), n_sample, n_res, 500), dtype=np.uint16)
    namespace = {
        "input_ta": input_ta,
        "tau_r": tau_r,
        "tau_d": tau_d,
        "G": gain,
        "BIAS": bias,
        "v_reset": v_reset,
        "v_thr": v_thr,
        "U_stp": STP_U,
        "tau_rec_stp": STP_TAU_REC_MS * ms,
        "tau_fac_stp": STP_TAU_FAC_MS * ms,
    }

    t0 = 0 * ms
    row_idx = 0
    saved_plot_materials = set()
    for i_mat, material in enumerate(DIR_NAME):
        for sample_id in tqdm(test_seq, desc=material):
            input_current, nt = load_input_current(material, int(sample_id), n_in, dt_s)
            vals = input_current.T
            vals = np.vstack([vals, vals[-1]])
            input_ta = TimedArray(vals, dt=dt_ms * ms)
            namespace["input_ta"] = input_ta
            g_in.t_start = t0

            g_res.v = v_reset + (v_thr - v_reset) * rng.random(n_res)
            g_res.R = 0
            g_res.H = 0
            if dynamic_synapse:
                s_res.u_stp = STP_U
                s_res.x_stp = 1.0

            start_t = t0
            start_idx = len(spike_res.t)
            duration = (nt * dt_ms) * ms
            net.run(duration, namespace=namespace)
            end_idx = len(spike_res.t)
            t0 += duration

            rel_ms = np.asarray((spike_res.t[start_idx:end_idx] - start_t) / ms, dtype=float)
            spike_i = np.asarray(spike_res.i[start_idx:end_idx], dtype=int)
            features[row_idx, :] = liquid_spikes_to_feature(rel_ms, spike_i, n_res, t_n)
            labels[row_idx] = i_mat
            if liquid_rec is not None:
                liquid_rec[i_mat, row_idx - i_mat * n_sample, :, :] = liquid_spikes_to_rec(rel_ms, spike_i, n_res)

            if save_sample_plots and material not in saved_plot_materials:
                save_liquid_raster(material, int(sample_id), rel_ms, spike_i, n_res)
                save_liquid_rate_plot(material, int(sample_id), rel_ms, n_res, t_n)
                saved_plot_materials.add(material)

            row_idx += 1

    if liquid_rec is not None:
        np.save(LIQUID_REC_PATH, liquid_rec)
        print(f"[saved] {LIQUID_REC_PATH} shape={liquid_rec.shape}")

    return features, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--t-n", type=int, default=DEFAULT_T_N)
    parser.add_argument("--codegen", choices=["numpy", "cython"], default=DEFAULT_CODEGEN_TARGET)
    parser.add_argument(
        "--no-sample-plots",
        action="store_true",
        help="Do not save one liquid raster/rate plot per material.",
    )
    parser.add_argument(
        "--save-liquid-rec",
        action="store_true",
        help="Save liquid layer spike counts as liquid_sout_rec_rep1.npy for eval.py.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Save liquid_sout_rec_rep1.npy and run eval.py for the liquid dataset.",
    )
    parser.add_argument(
        "--run-decoders",
        action="store_true",
        help="Save liquid_sout_rec_rep1.npy and run compare_decoders.py for the liquid dataset.",
    )
    parser.add_argument(
        "--eval-t-n",
        type=int,
        default=500,
        help="T_n passed to eval.py when --run-eval is used.",
    )
    parser.add_argument(
        "--decoder-t-n",
        type=int,
        default=500,
        help="T_n passed to compare_decoders.py when --run-decoders is used.",
    )
    parser.add_argument(
        "--decoder-pca-dims",
        default="100,150,200,300",
        help="Comma-separated PCA dimensions passed to compare_decoders.py.",
    )
    parser.add_argument(
        "--static-synapse",
        action="store_true",
        help="Disable the dynamic synapse model and use the original static recurrent synapse.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features, labels = generate_liquid_features(
        n_sample=args.n_sample,
        t_n=args.t_n,
        codegen_target=args.codegen,
        save_sample_plots=not args.no_sample_plots,
        save_liquid_rec=args.save_liquid_rec or args.run_eval or args.run_decoders,
        dynamic_synapse=USE_DYNAMIC_SYNAPSE and not args.static_synapse,
    )
    pca3, explained = compute_pca(features, n_components=3)
    dynamic_synapse = USE_DYNAMIC_SYNAPSE and not args.static_synapse
    plot_pca(pca3[:, :2], labels, args.t_n, args.n_sample, explained[:2], dynamic_synapse)
    plot_pca_3d(pca3, labels, args.t_n, args.n_sample, explained, dynamic_synapse)
    save_variance_summary(features, labels, args.t_n, args.n_sample, dynamic_synapse)

    if args.run_eval:
        if args.n_sample < 10:
            raise ValueError("--run-eval requires --n-sample >= 10 for 10-fold evaluation")
        cmd = [sys.executable, str(SCRIPT_DIR / "eval.py"), str(args.eval_t_n), "liquid"]
        print("[run eval]", " ".join(cmd))
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)

    if args.run_decoders:
        if args.n_sample < 10:
            raise ValueError("--run-decoders requires --n-sample >= 10 for 10-fold evaluation")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "compare_decoders.py"),
            "--dataset", "liquid",
            "--rep", str(REP),
            "--t-n", str(args.decoder_t_n),
            "--pca-dims", args.decoder_pca_dims,
        ]
        print("[run decoders]", " ".join(cmd))
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
