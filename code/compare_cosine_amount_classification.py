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
METHOD_LABELS = {
    "cosine_only": "Cosine only",
    "amount_only": "Firing amount only",
    "cosine_plus_amount": "Cosine + firing amount",
}
COLORS = ["tab:blue", "tab:orange", "tab:green"]
WINDOW_MS = 25


def bin_spikes(sout_rec: np.ndarray):
    n_class, n_sample, n_neuron, total_ms = sout_rec.shape
    if total_ms % WINDOW_MS != 0:
        raise ValueError(f"T={total_ms} is not divisible by {WINDOW_MS} ms")
    n_window = total_ms // WINDOW_MS

    pattern = np.empty(
        (n_class, n_sample, n_neuron * n_window),
        dtype=np.float32,
    )
    amount = np.empty((n_class, n_sample, n_window), dtype=np.float32)

    for class_idx in range(n_class):
        for sample_idx in range(n_sample):
            spikes = np.asarray(sout_rec[class_idx, sample_idx], dtype=np.float32)
            window_counts = spikes.reshape(
                n_neuron,
                n_window,
                WINDOW_MS,
            ).sum(axis=-1)
            pattern[class_idx, sample_idx] = window_counts.reshape(-1)
            amount[class_idx, sample_idx] = window_counts.sum(axis=0)
    return pattern, amount


def make_folds(n_sample: int, n_folds: int, seed: int):
    indices = np.arange(n_sample)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def normalize_rows(values: np.ndarray):
    norms = np.linalg.norm(values, axis=1)
    normalized = np.zeros_like(values, dtype=np.float32)
    valid = norms > 0
    normalized[valid] = values[valid] / norms[valid, None]
    return normalized


def cosine_predict(x_test: np.ndarray, prototypes: np.ndarray):
    test_unit = normalize_rows(x_test)
    prototype_unit = normalize_rows(prototypes)
    similarity = test_unit @ prototype_unit.T
    return np.argmax(similarity, axis=1)


def standardized_euclidean_predict(x_train: np.ndarray,
                                   y_train: np.ndarray,
                                   x_test: np.ndarray,
                                   n_class: int):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    train_z = (x_train - mean) / std
    test_z = (x_test - mean) / std
    prototypes = np.stack(
        [train_z[y_train == class_idx].mean(axis=0) for class_idx in range(n_class)]
    )
    test_norm = np.sum(test_z * test_z, axis=1, keepdims=True)
    prototype_norm = np.sum(prototypes * prototypes, axis=1)[None, :]
    distance_sq = test_norm + prototype_norm - 2.0 * (test_z @ prototypes.T)
    return np.argmin(distance_sq, axis=1)


def raw_euclidean_predict(x_test: np.ndarray, prototypes: np.ndarray):
    test_norm = np.sum(x_test * x_test, axis=1, keepdims=True)
    prototype_norm = np.sum(prototypes * prototypes, axis=1)[None, :]
    distance_sq = test_norm + prototype_norm - 2.0 * (x_test @ prototypes.T)
    return np.argmin(distance_sq, axis=1)


def evaluate(pattern: np.ndarray,
             amount: np.ndarray,
             n_folds: int,
             seed: int):
    n_class, n_sample, _n_feature = pattern.shape
    folds = make_folds(n_sample, n_folds, seed)
    method_scores = {method: [] for method in METHOD_LABELS}

    for fold_id, test_sample_idx in enumerate(folds, start=1):
        train_sample_idx = np.setdiff1d(np.arange(n_sample), test_sample_idx)
        y_train = np.repeat(np.arange(n_class), len(train_sample_idx))
        y_test = np.repeat(np.arange(n_class), len(test_sample_idx))

        pattern_train = pattern[:, train_sample_idx].reshape(
            n_class * len(train_sample_idx), -1
        )
        pattern_test = pattern[:, test_sample_idx].reshape(
            n_class * len(test_sample_idx), -1
        )
        amount_train = amount[:, train_sample_idx].reshape(
            n_class * len(train_sample_idx), -1
        )
        amount_test = amount[:, test_sample_idx].reshape(
            n_class * len(test_sample_idx), -1
        )

        pattern_prototypes = np.stack(
            [
                pattern_train[y_train == class_idx].mean(axis=0)
                for class_idx in range(n_class)
            ]
        )

        predictions = {
            "cosine_only": cosine_predict(pattern_test, pattern_prototypes),
            "amount_only": standardized_euclidean_predict(
                amount_train,
                y_train,
                amount_test,
                n_class,
            ),
            "cosine_plus_amount": raw_euclidean_predict(
                pattern_test,
                pattern_prototypes,
            ),
        }

        for method, prediction in predictions.items():
            method_scores[method].append(float(np.mean(prediction == y_test)))
        print(f"[fold {fold_id}/{n_folds}] done")

    rows = []
    for method, scores in method_scores.items():
        rows.append({
            "method": method,
            "window_ms": WINDOW_MS,
            "acc_mean": float(np.mean(scores)),
            "acc_std": float(np.std(scores)),
            "acc_min": float(np.min(scores)),
            "acc_max": float(np.max(scores)),
            "n_folds": len(scores),
        })
    return pd.DataFrame(rows), method_scores


def save_results(summary_df: pd.DataFrame,
                 fold_scores: dict,
                 result_dir: Path,
                 sensor_mode: str):
    out_dir = result_dir / sensor_mode / "cosine_amount_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.insert(0, "sensor_mode", sensor_mode)
    summary_path = out_dir / "cosine_amount_accuracy_summary_25ms.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    fold_rows = []
    for method, scores in fold_scores.items():
        for fold_id, accuracy in enumerate(scores, start=1):
            fold_rows.append({
                "sensor_mode": sensor_mode,
                "method": method,
                "fold": fold_id,
                "accuracy": accuracy,
                "window_ms": WINDOW_MS,
            })
    fold_path = out_dir / "cosine_amount_accuracy_folds_25ms.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    print(f"[saved] {fold_path}")

    plot_df = summary_df.set_index("method").reindex(METHOD_LABELS).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(plot_df))
    ax.bar(
        x,
        plot_df["acc_mean"],
        yerr=plot_df["acc_std"],
        color=COLORS,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in plot_df["method"]],
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Classification accuracy")
    ax.set_title(
        f"Cosine and firing-amount classification | "
        f"{SENSOR_LABELS[sensor_mode]} | window={WINDOW_MS} ms"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / "cosine_amount_accuracy_bar_25ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]
    summaries = []

    for sensor_mode in sensor_modes:
        rec_path = args.result_dir / sensor_mode / "liquid_sout_rec_rep1.npy"
        if not rec_path.exists():
            raise FileNotFoundError(rec_path)
        sout_rec = np.load(rec_path, mmap_mode="r")
        print(f"[loaded] {rec_path} shape={sout_rec.shape}")
        pattern, amount = bin_spikes(sout_rec)
        summary_df, fold_scores = evaluate(
            pattern,
            amount,
            args.n_folds,
            args.seed,
        )
        summaries.append(
            save_results(
                summary_df,
                fold_scores,
                args.result_dir,
                sensor_mode,
            )
        )

    all_summary = pd.concat(summaries, ignore_index=True)
    summary_path = args.result_dir / "cosine_amount_accuracy_all_sensors_25ms.csv"
    all_summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
