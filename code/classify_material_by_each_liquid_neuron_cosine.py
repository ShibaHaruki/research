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
MATERIALS = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]
WINDOW_MS = 25
CHANCE_ACCURACY = 1.0 / len(MATERIALS)


def bin_spikes(sout_rec: np.ndarray, window_ms: int) -> np.ndarray:
    n_class, n_sample, n_neuron, total_ms = sout_rec.shape
    if total_ms % window_ms != 0:
        raise ValueError(f"T={total_ms} is not divisible by window_ms={window_ms}")
    n_window = total_ms // window_ms
    binned = np.empty((n_class, n_sample, n_neuron, n_window), dtype=np.float32)
    for class_idx in range(n_class):
        for sample_idx in range(n_sample):
            spikes = np.asarray(sout_rec[class_idx, sample_idx], dtype=np.float32)
            binned[class_idx, sample_idx] = spikes.reshape(
                n_neuron, n_window, window_ms
            ).sum(axis=-1)
    return binned


def make_folds(n_sample: int, n_folds: int, seed: int):
    indices = np.arange(n_sample)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def normalize_vectors(values: np.ndarray):
    norms = np.linalg.norm(values, axis=-1)
    normalized = np.zeros_like(values, dtype=np.float32)
    valid = norms > 0
    normalized[valid] = values[valid] / norms[valid, None]
    return normalized, valid


def evaluate_each_neuron(binned: np.ndarray, n_folds: int, seed: int):
    n_class, n_sample, n_neuron, _n_window = binned.shape
    folds = make_folds(n_sample, n_folds, seed)

    correct_all = np.zeros(n_neuron, dtype=np.int64)
    correct_valid = np.zeros(n_neuron, dtype=np.int64)
    valid_count = np.zeros(n_neuron, dtype=np.int64)
    total_count = np.zeros(n_neuron, dtype=np.int64)
    class_correct = np.zeros((n_neuron, n_class), dtype=np.int64)
    class_valid = np.zeros((n_neuron, n_class), dtype=np.int64)
    class_total = np.zeros((n_neuron, n_class), dtype=np.int64)

    for fold_id, test_indices in enumerate(folds, start=1):
        train_indices = np.setdiff1d(np.arange(n_sample), test_indices)
        prototypes = binned[:, train_indices].mean(axis=1)
        prototype_unit, prototype_valid = normalize_vectors(prototypes)

        test_vectors = binned[:, test_indices].reshape(
            n_class * len(test_indices), n_neuron, -1
        )
        truth = np.repeat(np.arange(n_class), len(test_indices))
        test_unit, test_valid = normalize_vectors(test_vectors)

        similarity = np.einsum(
            "tnw,cnw->tnc",
            test_unit,
            prototype_unit,
            optimize=True,
        )
        similarity = np.where(
            prototype_valid.T[None, :, :],
            similarity,
            -np.inf,
        )
        prediction = np.argmax(similarity, axis=2)
        has_prototype = np.any(prototype_valid, axis=0)
        valid = test_valid & has_prototype[None, :]
        correct = prediction == truth[:, None]

        correct_all += np.sum(correct & valid, axis=0)
        correct_valid += np.sum(correct & valid, axis=0)
        valid_count += np.sum(valid, axis=0)
        total_count += len(truth)
        for class_idx in range(n_class):
            class_mask = truth == class_idx
            class_correct[:, class_idx] += np.sum(
                correct[class_mask] & valid[class_mask],
                axis=0,
            )
            class_valid[:, class_idx] += np.sum(valid[class_mask], axis=0)
            class_total[:, class_idx] += int(np.sum(class_mask))
        print(f"[fold {fold_id}/{n_folds}] done")

    accuracy_all = correct_all / np.maximum(total_count, 1)
    accuracy_valid = np.divide(
        correct_valid,
        valid_count,
        out=np.full(n_neuron, np.nan, dtype=float),
        where=valid_count > 0,
    )
    coverage = valid_count / np.maximum(total_count, 1)
    class_accuracy_all = class_correct / np.maximum(class_total, 1)
    class_accuracy_valid = np.divide(
        class_correct,
        class_valid,
        out=np.full((n_neuron, n_class), np.nan, dtype=float),
        where=class_valid > 0,
    )
    class_coverage = class_valid / np.maximum(class_total, 1)
    return (
        accuracy_all,
        accuracy_valid,
        coverage,
        valid_count,
        total_count,
        class_accuracy_all,
        class_accuracy_valid,
        class_coverage,
    )


def save_results(sensor_mode: str,
                 accuracy_all: np.ndarray,
                 accuracy_valid: np.ndarray,
                 coverage: np.ndarray,
                 valid_count: np.ndarray,
                 total_count: np.ndarray,
                 class_accuracy_all: np.ndarray,
                 class_accuracy_valid: np.ndarray,
                 class_coverage: np.ndarray):
    out_dir = RESULT_DIR / sensor_mode / "each_neuron_cosine_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({
        "neuron_index": np.arange(len(accuracy_all)),
        "accuracy_all_trials": accuracy_all,
        "accuracy_valid_trials": accuracy_valid,
        "coverage": coverage,
        "n_valid_trials": valid_count,
        "n_total_trials": total_count,
        "window_ms": WINDOW_MS,
    }).sort_values(
        ["accuracy_all_trials", "coverage"],
        ascending=[False, False],
    )
    csv_path = out_dir / "each_neuron_cosine_accuracy_25ms.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    summary_df = pd.DataFrame([{
        "sensor_mode": sensor_mode,
        "window_ms": WINDOW_MS,
        "n_neurons": len(accuracy_all),
        "best_accuracy": float(np.nanmax(accuracy_all)),
        "median_accuracy": float(np.nanmedian(accuracy_all)),
        "mean_accuracy": float(np.nanmean(accuracy_all)),
        "n_above_chance": int(np.sum(accuracy_all > CHANCE_ACCURACY)),
        "chance_accuracy": CHANCE_ACCURACY,
    }])
    summary_path = out_dir / "each_neuron_cosine_accuracy_summary_25ms.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    ranked = result_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(
        np.arange(1, len(ranked) + 1),
        ranked["accuracy_all_trials"],
        color="tab:blue",
        linewidth=1.5,
    )
    ax.axhline(
        CHANCE_ACCURACY,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"chance ({CHANCE_ACCURACY:.3f})",
    )
    ax.set_xlim(1, len(ranked))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Neuron rank")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"Material classification by each liquid neuron | {SENSOR_LABELS[sensor_mode]}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    rank_path = out_dir / "each_neuron_cosine_accuracy_rank_25ms.png"
    fig.savefig(rank_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {rank_path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(accuracy_all, bins=np.linspace(0.0, 1.0, 41), color="tab:blue", edgecolor="black")
    ax.axvline(CHANCE_ACCURACY, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Classification accuracy")
    ax.set_ylabel("Number of liquid neurons")
    ax.set_title(f"Per-neuron accuracy distribution | {SENSOR_LABELS[sensor_mode]}")
    fig.tight_layout()
    hist_path = out_dir / "each_neuron_cosine_accuracy_histogram_25ms.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {hist_path}")

    save_material_selectivity(
        out_dir=out_dir,
        sensor_mode=sensor_mode,
        class_accuracy_all=class_accuracy_all,
        class_accuracy_valid=class_accuracy_valid,
        class_coverage=class_coverage,
    )
    return summary_df


def save_material_selectivity(out_dir: Path,
                              sensor_mode: str,
                              class_accuracy_all: np.ndarray,
                              class_accuracy_valid: np.ndarray,
                              class_coverage: np.ndarray):
    n_neuron, n_class = class_accuracy_all.shape
    sorted_accuracy = np.sort(class_accuracy_all, axis=1)
    preferred_idx = np.argmax(class_accuracy_all, axis=1)
    selectivity_margin = sorted_accuracy[:, -1] - sorted_accuracy[:, -2]

    rows = []
    for neuron_idx in range(n_neuron):
        row = {
            "neuron_index": neuron_idx,
            "preferred_material": MATERIALS[preferred_idx[neuron_idx]],
            "preferred_accuracy": class_accuracy_all[neuron_idx, preferred_idx[neuron_idx]],
            "selectivity_margin": selectivity_margin[neuron_idx],
        }
        for class_idx, material in enumerate(MATERIALS):
            row[f"{material}_accuracy_all"] = class_accuracy_all[neuron_idx, class_idx]
            row[f"{material}_accuracy_valid"] = class_accuracy_valid[neuron_idx, class_idx]
            row[f"{material}_coverage"] = class_coverage[neuron_idx, class_idx]
        rows.append(row)

    selectivity_df = pd.DataFrame(rows).sort_values(
        ["selectivity_margin", "preferred_accuracy"],
        ascending=[False, False],
    )
    csv_path = out_dir / "each_neuron_material_selectivity_25ms.csv"
    selectivity_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    ranked_idx = np.argsort(np.nanmax(class_accuracy_all, axis=1))[::-1]
    matrix = class_accuracy_all[ranked_idx]
    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xticks(np.arange(n_class))
    ax.set_xticklabels(MATERIALS, rotation=40, ha="right")
    ax.set_xlabel("Material")
    ax.set_ylabel("Liquid neurons ranked by best material accuracy")
    ax.set_title(f"Per-neuron material accuracy | {SENSOR_LABELS[sensor_mode]}")
    fig.colorbar(image, ax=ax, label="Classification accuracy")
    fig.tight_layout()
    heatmap_path = out_dir / "each_neuron_material_accuracy_heatmap_25ms.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {heatmap_path}")

    top_n = min(20, n_neuron)
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 14.0))
    for class_idx, (ax, material) in enumerate(zip(axes.flat, MATERIALS)):
        material_accuracy = class_accuracy_all[:, class_idx]
        top_idx = np.argsort(material_accuracy)[::-1][:top_n]
        ax.barh(
            np.arange(top_n),
            material_accuracy[top_idx][::-1],
            color="tab:blue",
        )
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels([str(idx) for idx in top_idx[::-1]], fontsize=7)
        ax.set_xlim(0.0, 1.0)
        ax.axvline(CHANCE_ACCURACY, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(material)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Neuron index")
    fig.suptitle(f"Top material-selective liquid neurons | {SENSOR_LABELS[sensor_mode]}")
    fig.tight_layout()
    top_path = out_dir / "top_neurons_by_material_25ms.png"
    fig.savefig(top_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {top_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    global RESULT_DIR
    RESULT_DIR = args.result_dir
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]
    summaries = []

    for sensor_mode in sensor_modes:
        rec_path = args.result_dir / sensor_mode / "liquid_sout_rec_rep1.npy"
        if not rec_path.exists():
            raise FileNotFoundError(rec_path)
        sout_rec = np.load(rec_path, mmap_mode="r")
        print(f"[loaded] {rec_path} shape={sout_rec.shape}")
        binned = bin_spikes(sout_rec, WINDOW_MS)
        metrics = evaluate_each_neuron(binned, args.n_folds, args.seed)
        summaries.append(save_results(sensor_mode, *metrics))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_path = args.result_dir / "each_neuron_cosine_accuracy_summary_25ms.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
