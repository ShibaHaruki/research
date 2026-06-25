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
DISPLAY_NAMES = [
    "aluminum bd.", "outer_pigskin", "back_pigskin", "cork",
    "denim", "rubber bd.", "japanese paper", "wood bd.",
]
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]
MARKERS = ["o", "s", "^", "D", "v", "x", "*", "+"]


def compute_pca(features: np.ndarray, n_components: int = 3):
    x = features.astype(np.float64, copy=True)
    x -= x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x /= std

    _u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    scores = x @ vt[:n_components].T
    explained = singular_values[:n_components] ** 2
    explained /= np.sum(singular_values ** 2)
    return scores, explained


def load_neuron_features(result_dir: Path, sensor_mode: str):
    csv_path = (
        result_dir
        / sensor_mode
        / "each_neuron_cosine_classifier"
        / "each_neuron_material_selectivity_25ms.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} is missing. Run classify_material_by_each_liquid_neuron_cosine.py first."
        )

    df = pd.read_csv(csv_path).sort_values("neuron_index")
    feature_columns = [f"{material}_accuracy_all" for material in MATERIALS]
    features = df[feature_columns].to_numpy(dtype=float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    preferred = df["preferred_material"].to_numpy()
    neuron_indices = df["neuron_index"].to_numpy(dtype=int)
    return csv_path, features, preferred, neuron_indices


def plot_pca_2d(scores: np.ndarray,
                explained: np.ndarray,
                preferred: np.ndarray,
                neuron_indices: np.ndarray,
                out_dir: Path,
                sensor_mode: str):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for idx, material in enumerate(MATERIALS):
        mask = preferred == material
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=24,
            color=COLORS[idx],
            marker=MARKERS[idx],
            alpha=0.75,
            linewidths=0.5,
            label=f"{DISPLAY_NAMES[idx]} (n={np.sum(mask)})",
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title(f"PCA of liquid-neuron material selectivity | {SENSOR_LABELS[sensor_mode]}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "liquid_neuron_material_pca_2d_25ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

    score_df = pd.DataFrame({
        "neuron_index": neuron_indices,
        "preferred_material": preferred,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "PC3": scores[:, 2],
    })
    score_path = out_dir / "liquid_neuron_material_pca_scores_25ms.csv"
    score_df.to_csv(score_path, index=False)
    print(f"[saved] {score_path}")


def plot_pca_3d(scores: np.ndarray,
                explained: np.ndarray,
                preferred: np.ndarray,
                out_dir: Path,
                sensor_mode: str):
    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    for idx, material in enumerate(MATERIALS):
        mask = preferred == material
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            scores[mask, 2],
            s=20,
            color=COLORS[idx],
            marker=MARKERS[idx],
            alpha=0.75,
            depthshade=False,
            label=DISPLAY_NAMES[idx],
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)")
    ax.set_title(f"3D PCA of liquid-neuron material selectivity | {SENSOR_LABELS[sensor_mode]}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = out_dir / "liquid_neuron_material_pca_3d_25ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]

    for sensor_mode in sensor_modes:
        csv_path, features, preferred, neuron_indices = load_neuron_features(
            args.result_dir,
            sensor_mode,
        )
        print(f"[loaded] {csv_path} neurons={len(neuron_indices)} features={features.shape[1]}")
        scores, explained = compute_pca(features, n_components=3)
        out_dir = args.result_dir / sensor_mode / "liquid_neuron_material_pca"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_pca_2d(
            scores,
            explained,
            preferred,
            neuron_indices,
            out_dir,
            sensor_mode,
        )
        plot_pca_3d(scores, explained, preferred, out_dir, sensor_mode)


if __name__ == "__main__":
    main()
