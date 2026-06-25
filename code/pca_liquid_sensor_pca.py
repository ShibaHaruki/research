# -*- coding: utf-8 -*-
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

OUT_DIR = SCRIPT_DIR / "liquid_sensor_pca_results"

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
INPUT_CHANNEL_NAMES = [
    "sensor1_merkel",
    "sensor1_meissner",
    "sensor2_merkel",
    "sensor2_meissner",
    "sensor3_merkel",
    "sensor3_meissner",
]

N_TRAIN = 50
N_SAMPLE = 100
N_IN = 6
N_RES = 1000
P_IN = 0.2
P_RES = 0.5
TOTAL_MS = 500.0
DIFF_WIDTH_MS = 25.0
FAST_FILTER_TAU_MS = 0.2
DEFAULT_T_N = 25
DEFAULT_CODEGEN_TARGET = "cython"
CYTHON_CACHE_DIR = SCRIPT_DIR / "brian2_cython_cache"
USE_DYNAMIC_SYNAPSE = False
STP_U = 0.2
STP_TAU_REC_MS = 200.0
STP_TAU_FAC_MS = 1500.0
MERKEL_FILTER_SCALE = 0.1
MEISSNER_FILTER_SCALE = 0.4
SENSOR_GAIN_TABLE = [
    {
        "mode": "sensor1",
        "data_index": 0,
        "meissner_base_scale": 0.257391,
        "merkel_base_scale": 18.756983,
        "meissner_gain": 97.128388,
        "merkel_gain": 1.332837,
    },
    {
        "mode": "sensor2",
        "data_index": 1,
        "meissner_base_scale": 0.562090,
        "merkel_base_scale": 51.558203,
        "meissner_gain": 44.476862,
        "merkel_gain": 0.484889,
    },
    {
        "mode": "sensor3",
        "data_index": 2,
        "meissner_base_scale": 0.481277,
        "merkel_base_scale": 51.482391,
        "meissner_gain": 51.945128,
        "merkel_gain": 0.485603,
    },
]
SENSOR_MODE_LABELS = {
    "sensor1": "sensor 1",
    "sensor2": "sensor 2",
    "sensor3": "sensor 3",
    "all": "all sensors",
}


def sensor_out_dir(sensor_mode: str) -> Path:
    return OUT_DIR / sensor_mode


def calc_diff_rate(data, t, lag: int):
    dF_dt = np.zeros(len(t), dtype=float)
    for i in range(lag, len(t)):
        dt = t[i] - t[i - lag]
        if dt > 0:
            dF_dt[i] = np.abs(data[i] - data[i - lag]) / dt
    return dF_dt


def calc_meissner(data, t, dt, lag: int):
    current = np.zeros((4, len(t)))
    diff_rate = calc_diff_rate(data, t, lag)
    for i in range(1, len(t)):
        dF_dt = diff_rate[i]
        current[0, i] = (
            current[0, i - 1]
            + dF_dt
            + (-current[0, i - 1] * dt / (FAST_FILTER_TAU_MS * 1e-3))
        )
        current[1, i] = (
            current[1, i - 1]
            + 0.24 * dF_dt
            + (-(current[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
        )
        current[2, i] = current[2, i - 1] + 0.07 * dF_dt + (-current[2, i - 1] * dt / (1744.6 * 1e-3))
        current[3, i] = current[0, i]
    return current[3, :]


def calc_merkel(data, t, dt, lag: int):
    current = np.zeros((4, len(t)))
    diff_rate = calc_diff_rate(data, t, lag)
    for i in range(1, len(t)):
        dF_dt = diff_rate[i]
        current[0, i] = (
            current[0, i - 1]
            + 0.74 * dF_dt
            + (-current[0, i - 1] * dt / (FAST_FILTER_TAU_MS * 1e-3))
        )
        current[1, i] = (
            current[1, i - 1]
            + 0.24 * dF_dt
            + (-(current[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
        )
        current[2, i] = current[2, i - 1] + 0.07 * dF_dt + (-current[2, i - 1] * dt / (1744.6 * 1e-3))
        current[3, i] = current[0, i] + current[1, i] + current[2, i]
    return current[3, :]


def load_input_current(material: str, sample_id: int, n_in: int, dt_s: float, sensor_mode: str):
    files = glob.glob(DATA_PATH + "tactile_data/" + material + f"/data_{sample_id}_*")
    if len(files) == 0:
        raise FileNotFoundError(f"data not found: material={material}, sample_id={sample_id}")

    df = pd.read_table(files[0], header=None)
    df_np = df.to_numpy().T
    in_data_0 = df_np[:3, 3000:8000]
    nt = in_data_0.shape[1]
    t_array_s = np.arange(nt) * dt_s
    diff_lag = max(1, int(round((DIFF_WIDTH_MS * 1e-3) / dt_s)))

    input_current = np.zeros((n_in, nt), dtype=float)
    selected_sensors = SENSOR_GAIN_TABLE
    if sensor_mode != "all":
        selected_sensors = [row for row in SENSOR_GAIN_TABLE if row["mode"] == sensor_mode]
    if len(selected_sensors) == 0:
        raise ValueError(f"unknown sensor_mode: {sensor_mode}")

    for sensor in selected_sensors:
        in_data = in_data_0[sensor["data_index"], :]
        i_merkel = calc_merkel(in_data, t_array_s, dt_s, diff_lag)
        i_meissner = calc_meissner(in_data, t_array_s, dt_s, diff_lag)
        dF_dt = calc_diff_rate(in_data, t_array_s, lag=diff_lag)

        offset = sensor["data_index"] * 2
        input_current[offset + 0, :] = (
            MERKEL_FILTER_SCALE
            * sensor["merkel_base_scale"]
            * sensor["merkel_gain"]
            * i_merkel
        )
        input_current[offset + 1, :] = (
            MEISSNER_FILTER_SCALE
            * sensor["meissner_base_scale"]
            * sensor["meissner_gain"]
            * i_meissner
        )
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
    n_input_connections = int(round(N_RES * P_IN))
    mask_in = np.zeros((N_IN, N_RES), dtype=bool)
    for input_idx in range(N_IN):
        connected = rng.choice(N_RES, size=n_input_connections, replace=False)
        mask_in[input_idx, connected] = True
    w_in = w_in_raw * mask_in / np.sqrt(N_IN * P_IN)

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
             dynamic_synapse: bool,
             sensor_mode: str):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    sensor_label = SENSOR_MODE_LABELS[sensor_mode]
    ax.set_title(f"Liquid layer PCA | {sensor_label} | {synapse_label} | T_n={t_n} | n={n_sample}")
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()

    dyn_tag = "dynamic" if dynamic_synapse else "static"
    out_base = out_dir / f"liquid_pca_{dyn_tag}_Tn{t_n}_n{n_sample}"
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
                dynamic_synapse: bool,
                sensor_mode: str):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    sensor_label = SENSOR_MODE_LABELS[sensor_mode]
    ax.set_title(f"Liquid layer PCA 3D | {sensor_label} | {synapse_label} | T_n={t_n} | n={n_sample}")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    dyn_tag = "dynamic" if dynamic_synapse else "static"
    out_base = out_dir / f"liquid_pca_3d_{dyn_tag}_Tn{t_n}_n{n_sample}"
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_base.with_suffix('.png')}")
    print(f"[saved] {out_base.with_suffix('.pdf')}")


def save_variance_summary(features: np.ndarray,
                          labels: np.ndarray,
                          t_n: int,
                          n_sample: int,
                          dynamic_synapse: bool,
                          sensor_mode: str):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    csv_path = out_dir / f"liquid_variance_{dyn_tag}_Tn{t_n}_n{n_sample}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["within_variance_mean_sq_dist"], color=COLORS[:len(summary_df)])
    ax.axhline(between_variance, color="black", linestyle="--", linewidth=1.6, label="between-class variance")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["material"], rotation=35, ha="right")
    ax.set_ylabel("mean squared distance")
    ax.set_title(f"Liquid feature variance | {SENSOR_MODE_LABELS[sensor_mode]} | T_n={t_n} | n={n_sample}")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"liquid_variance_{dyn_tag}_Tn{t_n}_n{n_sample}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def make_fold_indices(n_sample: int, n_folds: int, seed_value: int):
    sample_indices = np.arange(n_sample)
    rng = np.random.default_rng(seed_value)
    rng.shuffle(sample_indices)
    return np.array_split(sample_indices, n_folds)


def flatten_class_sample_indices(sample_indices: np.ndarray, n_class: int, n_sample: int):
    out = []
    for class_idx in range(n_class):
        out.extend(class_idx * n_sample + sample_indices)
    return np.asarray(out, dtype=int)


def fit_regularized_mahalanobis_low_rank(x_train: np.ndarray,
                                         y_train: np.ndarray,
                                         ridge: float):
    models = []
    classes = np.unique(y_train)
    for class_idx in classes:
        x_class = x_train[y_train == class_idx].astype(np.float64, copy=False)
        mean = x_class.mean(axis=0)
        centered = x_class - mean
        denom = max(1, x_class.shape[0] - 1)
        u_t = centered / np.sqrt(denom)
        gram = u_t @ u_t.T
        a = np.eye(gram.shape[0], dtype=np.float64) + gram / ridge
        a_inv = np.linalg.pinv(a)
        models.append((int(class_idx), mean, u_t, a_inv))
    return models


def predict_regularized_mahalanobis_low_rank(models, x_test: np.ndarray, ridge: float):
    x_test = x_test.astype(np.float64, copy=False)
    distances = np.zeros((x_test.shape[0], len(models)), dtype=np.float64)
    for model_idx, (_class_idx, mean, u_t, a_inv) in enumerate(models):
        diff = x_test - mean
        diff_norm = np.sum(diff * diff, axis=1) / ridge
        q = diff @ u_t.T
        correction = np.einsum("ij,jk,ik->i", q, a_inv, q) / (ridge * ridge)
        distances[:, model_idx] = np.sqrt(np.maximum(diff_norm - correction, 0.0))
    classes = np.asarray([class_idx for class_idx, _mean, _u_t, _a_inv in models], dtype=int)
    return classes[np.argmin(distances, axis=1)]


def evaluate_liquid_accuracy(features: np.ndarray,
                             labels: np.ndarray,
                             n_sample: int,
                             accuracy_t_n: int,
                             sensor_mode: str,
                             n_folds: int = 10,
                             seed_value: int = 1,
                             ridge: float = 1e-6):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = features.astype(np.float64, copy=False) / (accuracy_t_n / 1000.0)
    y = labels.astype(int, copy=False)
    n_class = len(DIR_NAME)
    folds = make_fold_indices(n_sample, n_folds, seed_value)
    conf_total = np.zeros((n_class, n_class), dtype=int)
    rows = []
    fold_acc = []

    for fold_id, test_sample_idx in enumerate(folds, start=1):
        train_sample_idx = np.setdiff1d(np.arange(n_sample), test_sample_idx)
        train_idx = flatten_class_sample_indices(train_sample_idx, n_class, n_sample)
        test_idx = flatten_class_sample_indices(test_sample_idx, n_class, n_sample)

        models = fit_regularized_mahalanobis_low_rank(x[train_idx], y[train_idx], ridge)
        pred = predict_regularized_mahalanobis_low_rank(models, x[test_idx], ridge)
        truth = y[test_idx]
        acc = float(np.mean(pred == truth))
        fold_acc.append(acc)
        for true_label, pred_label in zip(truth, pred):
            conf_total[int(true_label), int(pred_label)] += 1
        rows.append({
            "sensor_mode": sensor_mode,
            "fold": fold_id,
            "accuracy": acc,
            "n_test": int(len(test_idx)),
            "accuracy_t_n": int(accuracy_t_n),
            "model": "liquid_regularized_mahalanobis",
        })

    summary = pd.DataFrame([{
        "sensor_mode": sensor_mode,
        "model": "liquid_regularized_mahalanobis",
        "accuracy_t_n": int(accuracy_t_n),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std": float(np.std(fold_acc)),
        "acc_min": float(np.min(fold_acc)),
        "acc_max": float(np.max(fold_acc)),
        "n_folds": int(len(fold_acc)),
        "n_sample_per_material": int(n_sample),
    }])
    fold_df = pd.DataFrame(rows)
    conf_df = pd.DataFrame(conf_total, index=DIR_NAME, columns=DIR_NAME)

    summary_path = out_dir / f"liquid_accuracy_Tn{accuracy_t_n}_n{n_sample}.csv"
    fold_path = out_dir / f"liquid_accuracy_folds_Tn{accuracy_t_n}_n{n_sample}.csv"
    conf_path = out_dir / f"liquid_accuracy_confusion_Tn{accuracy_t_n}_n{n_sample}.csv"
    summary.to_csv(summary_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    conf_df.to_csv(conf_path)
    print(f"[accuracy] {sensor_mode}: {summary.loc[0, 'acc_mean']:.4f}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {fold_path}")
    print(f"[saved] {conf_path}")
    return summary


def plot_sensor_accuracy_summary(accuracy_df: pd.DataFrame,
                                 accuracy_t_n: int,
                                 n_sample: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = ["sensor1", "sensor2", "sensor3", "all"]
    plot_df = (
        accuracy_df.set_index("sensor_mode")
        .reindex([mode for mode in order if mode in set(accuracy_df["sensor_mode"])])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(plot_df))
    ax.bar(
        x,
        plot_df["acc_mean"],
        yerr=plot_df["acc_std"],
        color=[COLORS[i % len(COLORS)] for i in range(len(plot_df))],
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SENSOR_MODE_LABELS[mode] for mode in plot_df["sensor_mode"]])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Liquid-layer classification accuracy | T_n={accuracy_t_n} | n={n_sample}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = OUT_DIR / f"liquid_accuracy_summary_bar_Tn{accuracy_t_n}_n{n_sample}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_liquid_raster(material: str,
                       sample_id: int,
                       spike_t_ms: np.ndarray,
                       spike_i: np.ndarray,
                       n_res: int,
                       sensor_mode: str):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(spike_t_ms, spike_i, s=2, color="black", alpha=0.65, linewidths=0)
    ax.set_xlim(0, TOTAL_MS)
    ax.set_ylim(-0.5, n_res - 0.5)
    ax.set_yticks(np.arange(0, n_res, 100))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("liquid neuron index")
    ax.set_title(f"{material} | sample {sample_id} | {SENSOR_MODE_LABELS[sensor_mode]} | liquid raster")
    fig.tight_layout()
    out_path = out_dir / f"liquid_raster_{material}_sample{sample_id}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_liquid_rate_plot(material: str,
                          sample_id: int,
                          spike_t_ms: np.ndarray,
                          n_res: int,
                          t_n: int,
                          sensor_mode: str):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    ax.set_title(f"{material} | sample {sample_id} | {SENSOR_MODE_LABELS[sensor_mode]} | liquid rate")
    fig.tight_layout()
    out_path = out_dir / f"liquid_rate_{material}_sample{sample_id}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_input_current(material: str,
                       sample_id: int,
                       input_current: np.ndarray,
                       dt_ms: float,
                       sensor_mode: str):
    out_dir = sensor_out_dir(sensor_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_ms = np.arange(input_current.shape[1], dtype=float) * dt_ms

    rows = {"time_ms": time_ms}
    for idx, name in enumerate(INPUT_CHANNEL_NAMES[:input_current.shape[0]]):
        rows[name] = input_current[idx, :]
    input_df = pd.DataFrame(rows)
    csv_path = out_dir / f"input_current_{material}_sample{sample_id}.csv"
    input_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    active = np.where(np.any(np.abs(input_current) > 0, axis=1))[0]
    if len(active) == 0:
        active = np.arange(input_current.shape[0])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for plot_idx, input_idx in enumerate(active):
        ax.plot(
            time_ms,
            input_current[input_idx, :],
            linewidth=1.4,
            color=COLORS[plot_idx % len(COLORS)],
            label=INPUT_CHANNEL_NAMES[input_idx],
        )
    ax.set_xlim(0, TOTAL_MS)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("input current")
    ax.set_title(f"{material} | sample {sample_id} | {SENSOR_MODE_LABELS[sensor_mode]} | input current")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig_path = out_dir / f"input_current_{material}_sample{sample_id}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {fig_path}")


def generate_liquid_features(n_sample: int,
                             t_n: int,
                             accuracy_t_n: int,
                             codegen_target: str,
                             save_sample_plots: bool,
                             save_liquid_rec: bool,
                             dynamic_synapse: bool,
                             sensor_mode: str):
    from brian2 import (
        prefs, float64, ms, Hz, defaultclock, seed, start_scope,
        NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network,
    )

    prefs.core.default_float_dtype = float64
    prefs.codegen.target = codegen_target
    if codegen_target == "cython":
        cython_cache_dir = CYTHON_CACHE_DIR / sensor_mode
        cython_cache_dir.mkdir(parents=True, exist_ok=True)
        prefs.codegen.runtime.cython.cache_dir = str(cython_cache_dir)
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
    n_accuracy_interval = int(TOTAL_MS // accuracy_t_n)
    accuracy_features = np.zeros(
        (len(DIR_NAME) * n_sample, n_res * n_accuracy_interval),
        dtype=np.float32,
    )
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
            input_current, nt = load_input_current(material, int(sample_id), n_in, dt_s, sensor_mode)
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
            accuracy_features[row_idx, :] = liquid_spikes_to_feature(rel_ms, spike_i, n_res, accuracy_t_n)
            labels[row_idx] = i_mat
            if liquid_rec is not None:
                liquid_rec[i_mat, row_idx - i_mat * n_sample, :, :] = liquid_spikes_to_rec(rel_ms, spike_i, n_res)

            if save_sample_plots and material not in saved_plot_materials:
                save_input_current(material, int(sample_id), input_current, dt_ms, sensor_mode)
                save_liquid_raster(material, int(sample_id), rel_ms, spike_i, n_res, sensor_mode)
                save_liquid_rate_plot(material, int(sample_id), rel_ms, n_res, t_n, sensor_mode)
                saved_plot_materials.add(material)

            row_idx += 1

    if liquid_rec is not None:
        rec_path = sensor_out_dir(sensor_mode) / f"liquid_sout_rec_rep{REP}.npy"
        np.save(rec_path, liquid_rec)
        print(f"[saved] {rec_path} shape={liquid_rec.shape}")

    return features, accuracy_features, labels


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
    parser.add_argument(
        "--sensor-mode",
        choices=["sensor1", "sensor2", "sensor3", "all", "each"],
        default="each",
        help="Sensor condition for PCA. Use 'each' to run sensor1, sensor2, sensor3, and all.",
    )
    parser.add_argument(
        "--accuracy-t-n",
        type=int,
        default=500,
        help="Time window in ms for liquid-layer accuracy evaluation.",
    )
    parser.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Do not calculate liquid-layer classification accuracy.",
    )
    return parser.parse_args()


def run_sensor_mode(args, sensor_mode: str, dynamic_synapse: bool):
    print(f"[sensor mode] {sensor_mode}: {SENSOR_MODE_LABELS[sensor_mode]}")
    features, accuracy_features, labels = generate_liquid_features(
        n_sample=args.n_sample,
        t_n=args.t_n,
        accuracy_t_n=args.accuracy_t_n,
        codegen_target=args.codegen,
        save_sample_plots=not args.no_sample_plots,
        save_liquid_rec=args.save_liquid_rec or args.run_eval or args.run_decoders,
        dynamic_synapse=dynamic_synapse,
        sensor_mode=sensor_mode,
    )
    pca3, explained = compute_pca(features, n_components=3)
    plot_pca(pca3[:, :2], labels, args.t_n, args.n_sample, explained[:2], dynamic_synapse, sensor_mode)
    plot_pca_3d(pca3, labels, args.t_n, args.n_sample, explained, dynamic_synapse, sensor_mode)
    save_variance_summary(features, labels, args.t_n, args.n_sample, dynamic_synapse, sensor_mode)
    accuracy_summary = None
    if not args.no_accuracy:
        if args.n_sample < 10:
            print(f"[skip accuracy] {sensor_mode}: n_sample must be >= 10 for 10-fold evaluation")
        else:
            accuracy_summary = evaluate_liquid_accuracy(
                features=accuracy_features,
                labels=labels,
                n_sample=args.n_sample,
                accuracy_t_n=args.accuracy_t_n,
                sensor_mode=sensor_mode,
            )
    if accuracy_summary is None:
        return sensor_mode, None
    return sensor_mode, accuracy_summary.iloc[0].to_dict()


def main():
    args = parse_args()
    dynamic_synapse = USE_DYNAMIC_SYNAPSE and not args.static_synapse
    sensor_modes = ["sensor1", "sensor2", "sensor3", "all"] if args.sensor_mode == "each" else [args.sensor_mode]
    if len(sensor_modes) != 1 and (args.run_eval or args.run_decoders):
        raise ValueError("--run-eval/--run-decoders can be used only with one --sensor-mode, not 'each'")

    accuracy_rows = []
    if len(sensor_modes) == 1 or args.save_liquid_rec:
        if len(sensor_modes) != 1 and args.save_liquid_rec:
            print("[info] --save-liquid-rec is enabled, so sensor modes run sequentially to reduce memory use.")
        for sensor_mode in sensor_modes:
            sensor_mode, accuracy_row = run_sensor_mode(args, sensor_mode, dynamic_synapse)
            if accuracy_row is not None:
                accuracy_rows.append(accuracy_row)
    else:
        with ProcessPoolExecutor(max_workers=min(4, len(sensor_modes))) as executor:
            futures = [
                executor.submit(run_sensor_mode, args, sensor_mode, dynamic_synapse)
                for sensor_mode in sensor_modes
            ]
            for future in as_completed(futures):
                sensor_mode, accuracy_row = future.result()
                if accuracy_row is not None:
                    accuracy_rows.append(accuracy_row)
                print(f"[done] {sensor_mode}")

    if accuracy_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        accuracy_df = pd.DataFrame(accuracy_rows).sort_values("sensor_mode")
        summary_path = OUT_DIR / f"liquid_accuracy_summary_Tn{args.accuracy_t_n}_n{args.n_sample}.csv"
        accuracy_df.to_csv(summary_path, index=False)
        print(f"[saved] {summary_path}")
        plot_sensor_accuracy_summary(accuracy_df, args.accuracy_t_n, args.n_sample)

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
