# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "liquid_sensor_pca_results"
SENSOR_ORDER = ["sensor1", "sensor2", "sensor3", "all"]
SENSOR_LABELS = {
    "sensor1": "sensor 1",
    "sensor2": "sensor 2",
    "sensor3": "sensor 3",
    "all": "all sensors",
}
DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def extract_rate_features(sout_rec: np.ndarray, neuron_idx: np.ndarray, t_n: int):
    n_class, n_sample, _n_neuron, total_t = sout_rec.shape
    if total_t % t_n != 0:
        raise ValueError(f"T={total_t} is not divisible by T_n={t_n}")
    x = sout_rec[:, :, neuron_idx, :]
    n_interval = total_t // t_n
    x = x.reshape(n_class, n_sample, len(neuron_idx), n_interval, t_n).sum(axis=-1)
    x = x / (t_n / 1000.0)
    x = x.reshape(n_class * n_sample, len(neuron_idx) * n_interval).astype(np.float64, copy=False)
    y = np.repeat(np.arange(n_class), n_sample)
    return x, y, n_class, n_sample


def make_fold_indices(n_sample: int, n_folds: int, seed: int):
    indices = np.arange(n_sample)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def flatten_class_sample_indices(sample_indices: np.ndarray, n_class: int, n_sample: int):
    out = []
    for class_idx in range(n_class):
        out.extend(class_idx * n_sample + sample_indices)
    return np.asarray(out, dtype=int)


def fit_regularized_mahalanobis_low_rank(x_train: np.ndarray, y_train: np.ndarray, ridge: float):
    models = []
    for class_idx in np.unique(y_train):
        x_class = x_train[y_train == class_idx]
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
    distances = np.zeros((x_test.shape[0], len(models)), dtype=np.float64)
    for model_idx, (_class_idx, mean, u_t, a_inv) in enumerate(models):
        diff = x_test - mean
        diff_norm = np.sum(diff * diff, axis=1) / ridge
        q = diff @ u_t.T
        correction = np.einsum("ij,jk,ik->i", q, a_inv, q) / (ridge * ridge)
        distances[:, model_idx] = np.sqrt(np.maximum(diff_norm - correction, 0.0))
    classes = np.asarray([class_idx for class_idx, _mean, _u_t, _a_inv in models], dtype=int)
    return classes[np.argmin(distances, axis=1)]


def eval_one_subset(sout_rec: np.ndarray,
                    neuron_idx: np.ndarray,
                    t_n: int,
                    n_folds: int,
                    seed: int,
                    ridge: float):
    x, y, n_class, n_sample = extract_rate_features(sout_rec, neuron_idx, t_n)
    folds = make_fold_indices(n_sample, n_folds, seed)
    accs = []
    for test_sample_idx in folds:
        train_sample_idx = np.setdiff1d(np.arange(n_sample), test_sample_idx)
        train_idx = flatten_class_sample_indices(train_sample_idx, n_class, n_sample)
        test_idx = flatten_class_sample_indices(test_sample_idx, n_class, n_sample)
        models = fit_regularized_mahalanobis_low_rank(x[train_idx], y[train_idx], ridge)
        pred = predict_regularized_mahalanobis_low_rank(models, x[test_idx], ridge)
        accs.append(float(np.mean(pred == y[test_idx])))
    return float(np.mean(accs)), float(np.std(accs))


def eval_sensor(sensor_mode: str,
                result_dir: Path,
                neuron_counts: list[int],
                t_n: int,
                repeats: int,
                n_folds: int,
                seed: int,
                ridge: float):
    rec_path = result_dir / sensor_mode / "liquid_sout_rec_rep1.npy"
    if not rec_path.exists():
        raise FileNotFoundError(
            f"{rec_path} is missing. Run pca_liquid_sensor_pca.py with --save-liquid-rec first."
        )
    sout_rec = np.load(rec_path)
    n_neuron = sout_rec.shape[2]
    rows = []
    rng = np.random.default_rng(seed)
    for n_use in neuron_counts:
        if n_use > n_neuron:
            continue
        repeat_acc = []
        repeat_std = []
        for repeat_id in range(repeats):
            neuron_idx = np.sort(rng.choice(n_neuron, size=n_use, replace=False))
            acc_mean, acc_std = eval_one_subset(
                sout_rec=sout_rec,
                neuron_idx=neuron_idx,
                t_n=t_n,
                n_folds=n_folds,
                seed=seed,
                ridge=ridge,
            )
            repeat_acc.append(acc_mean)
            repeat_std.append(acc_std)
            rows.append({
                "sensor_mode": sensor_mode,
                "n_neuron_used": int(n_use),
                "repeat": int(repeat_id + 1),
                "acc_mean": acc_mean,
                "acc_std_fold": acc_std,
                "t_n": int(t_n),
            })
        print(f"[{sensor_mode}] n={n_use}: {np.mean(repeat_acc):.4f}")
    return pd.DataFrame(rows)


def plot_results(summary_df: pd.DataFrame, result_dir: Path, t_n: int):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x_ticks = sorted(summary_df["n_neuron_used"].unique())
    for i, sensor_mode in enumerate(SENSOR_ORDER):
        df = summary_df[summary_df["sensor_mode"] == sensor_mode]
        if df.empty:
            continue
        ax.errorbar(
            df["n_neuron_used"],
            df["acc_mean"],
            yerr=df["acc_std_repeat"],
            marker="o",
            linewidth=1.8,
            capsize=4,
            color=COLORS[i % len(COLORS)],
            label=SENSOR_LABELS[sensor_mode],
        )
    ax.set_xscale("log")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Number of liquid neurons used for evaluation")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs. number of evaluated liquid neurons | T_n={t_n}")
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = result_dir / f"liquid_accuracy_vs_neuron_count_Tn{t_n}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    parser.add_argument("--neuron-counts", default="10,25,50,100,200,500,1000")
    parser.add_argument("--t-n", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]
    neuron_counts = [int(v) for v in args.neuron_counts.split(",") if v.strip()]
    rows = []
    for sensor_mode in sensor_modes:
        df = eval_sensor(
            sensor_mode=sensor_mode,
            result_dir=args.result_dir,
            neuron_counts=neuron_counts,
            t_n=args.t_n,
            repeats=args.repeats,
            n_folds=args.n_folds,
            seed=args.seed,
            ridge=args.ridge,
        )
        rows.append(df)
    result_df = pd.concat(rows, ignore_index=True)
    detail_path = args.result_dir / f"liquid_accuracy_vs_neuron_count_detail_Tn{args.t_n}.csv"
    result_df.to_csv(detail_path, index=False)
    print(f"[saved] {detail_path}")

    summary_df = (
        result_df
        .groupby(["sensor_mode", "n_neuron_used"], as_index=False)
        .agg(
            acc_mean=("acc_mean", "mean"),
            acc_std_repeat=("acc_mean", "std"),
            acc_std_fold_mean=("acc_std_fold", "mean"),
        )
    )
    summary_df["acc_std_repeat"] = summary_df["acc_std_repeat"].fillna(0.0)
    summary_path = args.result_dir / f"liquid_accuracy_vs_neuron_count_summary_Tn{args.t_n}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")
    plot_results(summary_df, args.result_dir, args.t_n)


if __name__ == "__main__":
    main()
