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


def load_cosine_features(result_dir: Path, sensor_mode: str, window_ms: int):
    base_dir = result_dir / sensor_mode / "reference_material_neuron_cosine"
    frames = []

    for reference_material in MATERIALS:
        path = (
            base_dir
            / reference_material
            / f"reference_{reference_material}_cosine_by_neuron.csv"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run plot_reference_material_neuron_cosine.py first."
            )
        df = pd.read_csv(path)
        df = df[df["window_ms"] == window_ms].copy()
        df["reference_material"] = reference_material
        frames.append(df)

    cosine_df = pd.concat(frames, ignore_index=True)
    cosine_df["comparison_material"] = cosine_df["comparison_group"].str.replace(
        r"^same_", "",
        regex=True,
    )
    cosine_df["feature_name"] = (
        cosine_df["reference_material"]
        + "__"
        + cosine_df["comparison_material"]
    )

    feature_df = cosine_df.pivot_table(
        index="neuron_index",
        columns="feature_name",
        values="cosine_mean",
        aggfunc="mean",
    )
    expected_columns = [
        f"{reference}__{comparison}"
        for reference in MATERIALS
        for comparison in MATERIALS
    ]
    feature_df = feature_df.reindex(columns=expected_columns)

    features = feature_df.to_numpy(dtype=float, copy=True)
    valid_fraction = np.mean(np.isfinite(features), axis=1)
    column_means = np.nanmean(features, axis=0)
    column_means = np.nan_to_num(column_means, nan=0.0)
    missing_rows, missing_cols = np.where(~np.isfinite(features))
    features[missing_rows, missing_cols] = column_means[missing_cols]

    selectivity = np.zeros((len(feature_df), len(MATERIALS)), dtype=float)
    for material_idx, reference in enumerate(MATERIALS):
        same_col = expected_columns.index(f"{reference}__{reference}")
        different_cols = [
            expected_columns.index(f"{reference}__{comparison}")
            for comparison in MATERIALS
            if comparison != reference
        ]
        selectivity[:, material_idx] = (
            features[:, same_col] - np.mean(features[:, different_cols], axis=1)
        )

    preferred_idx = np.argmax(selectivity, axis=1)
    preferred = np.asarray(MATERIALS, dtype=object)[preferred_idx]
    preferred_score = selectivity[np.arange(len(selectivity)), preferred_idx]
    return (
        feature_df.index.to_numpy(dtype=int),
        features,
        valid_fraction,
        preferred,
        preferred_score,
        expected_columns,
    )


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


def plot_2d(scores: np.ndarray,
            explained: np.ndarray,
            preferred: np.ndarray,
            out_dir: Path,
            sensor_mode: str,
            window_ms: int):
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
    ax.set_title(
        f"PCA of per-neuron cosine-similarity profiles | "
        f"{SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / f"liquid_neuron_cosine_pca_2d_{window_ms}ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_3d(scores: np.ndarray,
            explained: np.ndarray,
            preferred: np.ndarray,
            out_dir: Path,
            sensor_mode: str,
            window_ms: int):
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
    ax.set_title(
        f"3D PCA of per-neuron cosine profiles | "
        f"{SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = out_dir / f"liquid_neuron_cosine_pca_3d_{window_ms}ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    parser.add_argument("--window-ms", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]

    for sensor_mode in sensor_modes:
        (
            neuron_indices,
            features,
            valid_fraction,
            preferred,
            preferred_score,
            feature_names,
        ) = load_cosine_features(args.result_dir, sensor_mode, args.window_ms)
        print(
            f"[loaded] {sensor_mode}: neurons={len(neuron_indices)}, "
            f"cosine_features={features.shape[1]}"
        )

        scores, explained = compute_pca(features, n_components=3)
        out_dir = args.result_dir / sensor_mode / "liquid_neuron_cosine_pca"
        out_dir.mkdir(parents=True, exist_ok=True)

        score_df = pd.DataFrame({
            "neuron_index": neuron_indices,
            "preferred_material": preferred,
            "preferred_selectivity": preferred_score,
            "valid_cosine_fraction": valid_fraction,
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC3": scores[:, 2],
        })
        score_path = out_dir / f"liquid_neuron_cosine_pca_scores_{args.window_ms}ms.csv"
        score_df.to_csv(score_path, index=False)
        print(f"[saved] {score_path}")

        loading_df = pd.DataFrame(
            {"feature_name": feature_names}
        )
        loading_path = out_dir / f"liquid_neuron_cosine_feature_names_{args.window_ms}ms.csv"
        loading_df.to_csv(loading_path, index=False)
        print(f"[saved] {loading_path}")

        plot_2d(
            scores,
            explained,
            preferred,
            out_dir,
            sensor_mode,
            args.window_ms,
        )
        plot_3d(
            scores,
            explained,
            preferred,
            out_dir,
            sensor_mode,
            args.window_ms,
        )


if __name__ == "__main__":
    main()
