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


def build_features(sout_rec: np.ndarray) -> list[np.ndarray]:
    n_class, n_sample, n_neuron, total_ms = sout_rec.shape
    if total_ms % WINDOW_MS != 0:
        raise ValueError(f"T={total_ms} is not divisible by {WINDOW_MS} ms")
    n_window = total_ms // WINDOW_MS

    class_features = []
    for class_idx in range(n_class):
        spikes = np.asarray(sout_rec[class_idx], dtype=np.float32)
        features = spikes.reshape(
            n_sample,
            n_neuron,
            n_window,
            WINDOW_MS,
        ).sum(axis=-1)
        class_features.append(features.reshape(n_sample, -1))
    return class_features


def compute_material_inner_products(class_features: list[np.ndarray]):
    n_class = len(class_features)
    matrix = np.zeros((n_class, n_class), dtype=np.float64)
    std_matrix = np.zeros_like(matrix)
    rows = []

    for class_a in range(n_class):
        for class_b in range(class_a, n_class):
            products = class_features[class_a] @ class_features[class_b].T
            if class_a == class_b:
                pair_i, pair_j = np.triu_indices(products.shape[0], k=1)
                values = products[pair_i, pair_j]
            else:
                values = products.reshape(-1)

            mean_value = float(np.mean(values))
            std_value = float(np.std(values))
            matrix[class_a, class_b] = mean_value
            matrix[class_b, class_a] = mean_value
            std_matrix[class_a, class_b] = std_value
            std_matrix[class_b, class_a] = std_value
            rows.append({
                "material_a": MATERIALS[class_a],
                "material_b": MATERIALS[class_b],
                "pair_type": "same_material" if class_a == class_b else "different_material",
                "inner_product_mean": mean_value,
                "inner_product_std": std_value,
                "inner_product_median": float(np.median(values)),
                "n_pairs": int(len(values)),
                "window_ms": WINDOW_MS,
            })
    return matrix, std_matrix, pd.DataFrame(rows)


def compute_inner_product_over_time(sout_rec: np.ndarray):
    n_class, n_sample, n_neuron, total_ms = sout_rec.shape
    n_window = total_ms // WINDOW_MS
    rows = []

    for window_idx in range(n_window):
        start_ms = window_idx * WINDOW_MS
        end_ms = start_ms + WINDOW_MS
        center_ms = start_ms + WINDOW_MS / 2.0
        class_vectors = [
            np.asarray(
                sout_rec[class_idx, :, :, start_ms:end_ms],
                dtype=np.float32,
            ).sum(axis=-1)
            for class_idx in range(n_class)
        ]

        for class_a in range(n_class):
            for class_b in range(class_a, n_class):
                products = class_vectors[class_a] @ class_vectors[class_b].T
                if class_a == class_b:
                    pair_i, pair_j = np.triu_indices(n_sample, k=1)
                    values = products[pair_i, pair_j]
                else:
                    values = products.reshape(-1)

                rows.append({
                    "time_start_ms": start_ms,
                    "time_end_ms": end_ms,
                    "time_center_ms": center_ms,
                    "material_a": MATERIALS[class_a],
                    "material_b": MATERIALS[class_b],
                    "material_pair": f"{MATERIALS[class_a]} / {MATERIALS[class_b]}",
                    "pair_type": "same_material" if class_a == class_b else "different_material",
                    "inner_product_mean": float(np.mean(values)),
                    "inner_product_std": float(np.std(values)),
                    "inner_product_median": float(np.median(values)),
                    "n_pairs": int(len(values)),
                    "window_ms": WINDOW_MS,
                })
    return pd.DataFrame(rows)


def plot_time_heatmap(time_df: pd.DataFrame, out_dir: Path, sensor_mode: str):
    pair_order = list(dict.fromkeys(time_df["material_pair"]))
    time_order = sorted(time_df["time_center_ms"].unique())
    matrix = (
        time_df.pivot(
            index="material_pair",
            columns="time_center_ms",
            values="inner_product_mean",
        )
        .reindex(index=pair_order, columns=time_order)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(12.0, 11.0))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(time_order)))
    ax.set_xticklabels(
        [
            f"{int(center - WINDOW_MS / 2)}-{int(center + WINDOW_MS / 2)}"
            for center in time_order
        ],
        rotation=55,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pair_order)))
    ax.set_yticklabels(pair_order, fontsize=8)
    ax.set_xlabel("Time interval (ms)")
    ax.set_ylabel("Material pair")
    ax.set_title(
        f"Material inner product over time | {SENSOR_LABELS[sensor_mode]} | "
        f"window={WINDOW_MS} ms"
    )
    fig.colorbar(image, ax=ax, label="Mean inner product")
    fig.tight_layout()
    out_path = out_dir / "material_inner_product_time_heatmap_25ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_same_different_time(time_df: pd.DataFrame, out_dir: Path, sensor_mode: str):
    grouped = (
        time_df.groupby(["pair_type", "time_center_ms"], as_index=False)
        .agg(
            inner_product_mean=("inner_product_mean", "mean"),
            inner_product_std=("inner_product_mean", "std"),
        )
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for pair_type, label, color, marker in [
        ("same_material", "Same material", "tab:blue", "o"),
        ("different_material", "Different material", "tab:orange", "s"),
    ]:
        df = grouped[grouped["pair_type"] == pair_type]
        ax.errorbar(
            df["time_center_ms"],
            df["inner_product_mean"],
            yerr=df["inner_product_std"].fillna(0.0),
            color=color,
            marker=marker,
            linewidth=1.7,
            capsize=3,
            label=label,
        )
    ax.set_xticks(sorted(time_df["time_center_ms"].unique()))
    ax.set_xticklabels(
        [
            f"{int(center - WINDOW_MS / 2)}-{int(center + WINDOW_MS / 2)}"
            for center in sorted(time_df["time_center_ms"].unique())
        ],
        rotation=55,
        ha="right",
    )
    ax.set_xlabel("Time interval (ms)")
    ax.set_ylabel("Mean inner product")
    ax.set_title(f"Inner product over time | {SENSOR_LABELS[sensor_mode]}")
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "same_different_inner_product_time_25ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_heatmap(matrix: np.ndarray, out_dir: Path, sensor_mode: str):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(MATERIALS)))
    ax.set_yticks(np.arange(len(MATERIALS)))
    ax.set_xticklabels(MATERIALS, rotation=40, ha="right")
    ax.set_yticklabels(MATERIALS)
    ax.set_title(
        f"Material inner product | {SENSOR_LABELS[sensor_mode]} | window={WINDOW_MS} ms"
    )

    threshold = (np.nanmin(matrix) + np.nanmax(matrix)) / 2.0
    for row_idx in range(len(MATERIALS)):
        for col_idx in range(len(MATERIALS)):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2e}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < threshold else "black",
            )

    fig.colorbar(image, ax=ax, label="Mean inner product")
    fig.tight_layout()
    out_path = out_dir / "material_inner_product_heatmap_25ms.png"
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
        rec_path = args.result_dir / sensor_mode / "liquid_sout_rec_rep1.npy"
        if not rec_path.exists():
            raise FileNotFoundError(rec_path)

        sout_rec = np.load(rec_path, mmap_mode="r")
        print(f"[loaded] {rec_path} shape={sout_rec.shape}")
        class_features = build_features(sout_rec)
        matrix, std_matrix, detail_df = compute_material_inner_products(class_features)

        out_dir = args.result_dir / sensor_mode / "material_inner_product_25ms"
        out_dir.mkdir(parents=True, exist_ok=True)
        mean_path = out_dir / "material_inner_product_mean_25ms.csv"
        std_path = out_dir / "material_inner_product_std_25ms.csv"
        detail_path = out_dir / "material_inner_product_pairs_25ms.csv"
        pd.DataFrame(matrix, index=MATERIALS, columns=MATERIALS).to_csv(mean_path)
        pd.DataFrame(std_matrix, index=MATERIALS, columns=MATERIALS).to_csv(std_path)
        detail_df.to_csv(detail_path, index=False)
        print(f"[saved] {mean_path}")
        print(f"[saved] {std_path}")
        print(f"[saved] {detail_path}")
        plot_heatmap(matrix, out_dir, sensor_mode)

        time_df = compute_inner_product_over_time(sout_rec)
        time_path = out_dir / "material_inner_product_over_time_25ms.csv"
        time_df.to_csv(time_path, index=False)
        print(f"[saved] {time_path}")
        plot_time_heatmap(time_df, out_dir, sensor_mode)
        plot_same_different_time(time_df, out_dir, sensor_mode)


if __name__ == "__main__":
    main()
