# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack


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
WINDOWS_MS = [5, 10, 20, 25, 50, 100, 250, 500]


def build_sparse_features(sout_rec: np.ndarray, window_ms: int) -> csr_matrix:
    n_class, n_sample, n_neuron, total_ms = sout_rec.shape
    if total_ms % window_ms != 0:
        raise ValueError(f"T={total_ms} is not divisible by window_ms={window_ms}")

    n_window = total_ms // window_ms
    rows = []
    for class_idx in range(n_class):
        for sample_idx in range(n_sample):
            spike = np.asarray(sout_rec[class_idx, sample_idx], dtype=np.float32)
            feature = spike.reshape(n_neuron, n_window, window_ms).sum(axis=-1).reshape(1, -1)
            rows.append(csr_matrix(feature))
    return vstack(rows, format="csr")


def cosine_matrix(features: csr_matrix) -> np.ndarray:
    norms = np.sqrt(features.multiply(features).sum(axis=1)).A1
    inv_norms = np.zeros_like(norms)
    nonzero = norms > 0
    inv_norms[nonzero] = 1.0 / norms[nonzero]
    normalized = features.multiply(inv_norms[:, None]).tocsr()
    return (normalized @ normalized.T).toarray()


def summarize_similarity(similarity: np.ndarray,
                         labels: np.ndarray,
                         sensor_mode: str,
                         window_ms: int):
    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    values = similarity[upper_i, upper_j]
    same_mask = labels[upper_i] == labels[upper_j]

    summary_rows = []
    for pair_type, mask in [("same_material", same_mask), ("different_material", ~same_mask)]:
        selected = values[mask]
        summary_rows.append({
            "sensor_mode": sensor_mode,
            "window_ms": window_ms,
            "pair_type": pair_type,
            "cosine_mean": float(np.mean(selected)),
            "cosine_std": float(np.std(selected)),
            "cosine_median": float(np.median(selected)),
            "n_pairs": int(len(selected)),
        })

    detail_rows = []
    n_class = len(MATERIALS)
    for class_a in range(n_class):
        for class_b in range(class_a, n_class):
            pair_mask = (
                ((labels[upper_i] == class_a) & (labels[upper_j] == class_b))
                | ((labels[upper_i] == class_b) & (labels[upper_j] == class_a))
            )
            selected = values[pair_mask]
            detail_rows.append({
                "sensor_mode": sensor_mode,
                "window_ms": window_ms,
                "material_a": MATERIALS[class_a],
                "material_b": MATERIALS[class_b],
                "pair_type": "same_material" if class_a == class_b else "different_material",
                "cosine_mean": float(np.mean(selected)),
                "cosine_std": float(np.std(selected)),
                "cosine_median": float(np.median(selected)),
                "n_pairs": int(len(selected)),
            })
    return summary_rows, detail_rows


def summarize_material_pair_time_series(sout_rec: np.ndarray,
                                        sensor_mode: str,
                                        window_ms: int):
    n_class, n_sample, n_neuron, total_ms = sout_rec.shape
    n_window = total_ms // window_ms
    rows = []

    for window_idx in range(n_window):
        start_ms = window_idx * window_ms
        end_ms = start_ms + window_ms
        center_ms = start_ms + window_ms / 2.0
        start_bin = start_ms
        end_bin = end_ms

        class_vectors = []
        for class_idx in range(n_class):
            vectors = np.asarray(
                sout_rec[class_idx, :, :, start_bin:end_bin],
                dtype=np.float32,
            ).sum(axis=-1)
            norms = np.linalg.norm(vectors, axis=1)
            nonzero = norms > 0
            normalized = np.zeros_like(vectors, dtype=np.float32)
            normalized[nonzero] = vectors[nonzero] / norms[nonzero, None]
            class_vectors.append(normalized)

        for class_a in range(n_class):
            for class_b in range(class_a, n_class):
                similarity = class_vectors[class_a] @ class_vectors[class_b].T
                if class_a == class_b:
                    pair_i, pair_j = np.triu_indices(n_sample, k=1)
                    values = similarity[pair_i, pair_j]
                else:
                    values = similarity.reshape(-1)
                rows.append({
                    "sensor_mode": sensor_mode,
                    "window_ms": window_ms,
                    "time_start_ms": start_ms,
                    "time_end_ms": end_ms,
                    "time_center_ms": center_ms,
                    "material_a": MATERIALS[class_a],
                    "material_b": MATERIALS[class_b],
                    "pair_type": "same_material" if class_a == class_b else "different_material",
                    "cosine_mean": float(np.mean(values)),
                    "cosine_std": float(np.std(values)),
                    "cosine_median": float(np.median(values)),
                    "n_pairs": int(len(values)),
                })
    return rows


def plot_material_pair_time_heatmap(time_df: pd.DataFrame,
                                    out_dir: Path,
                                    sensor_mode: str,
                                    window_ms: int):
    different_df = time_df[time_df["pair_type"] == "different_material"].copy()
    different_df["pair"] = different_df["material_a"] + " / " + different_df["material_b"]
    pair_order = list(dict.fromkeys(different_df["pair"]))
    time_order = sorted(different_df["time_center_ms"].unique())
    matrix = (
        different_df.pivot(index="pair", columns="time_center_ms", values="cosine_mean")
        .reindex(index=pair_order, columns=time_order)
        .to_numpy()
    )

    fig_height = max(7.0, len(pair_order) * 0.28)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(time_order)))
    ax.set_xticklabels([f"{time:g}" for time in time_order], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pair_order)))
    ax.set_yticklabels(pair_order, fontsize=8)
    ax.set_xlabel("Time window center (ms)")
    ax.set_ylabel("Material pair")
    ax.set_title(
        f"Material-pair cosine similarity over time | "
        f"{SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
    )
    fig.colorbar(image, ax=ax, label="Mean cosine similarity")
    fig.tight_layout()
    out_path = out_dir / f"material_pair_cosine_time_heatmap_{window_ms}ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_same_different_time_series(time_df: pd.DataFrame,
                                    out_dir: Path,
                                    sensor_mode: str,
                                    window_ms: int):
    grouped = (
        time_df.groupby(["pair_type", "time_center_ms"], as_index=False)
        .agg(
            cosine_mean=("cosine_mean", "mean"),
            cosine_std=("cosine_mean", "std"),
        )
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for pair_type, label, color, marker in [
        ("same_material", "Same material", "tab:blue", "o"),
        ("different_material", "Different material", "tab:orange", "s"),
    ]:
        df = grouped[grouped["pair_type"] == pair_type]
        ax.errorbar(
            df["time_center_ms"],
            df["cosine_mean"],
            yerr=df["cosine_std"].fillna(0.0),
            color=color,
            marker=marker,
            linewidth=1.7,
            capsize=3,
            label=label,
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time window center (ms)")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(
        f"Cosine similarity over time | {SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
    )
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"same_different_cosine_time_{window_ms}ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_one_sensor(summary_df: pd.DataFrame, out_dir: Path, sensor_mode: str):
    sensor_df = summary_df[summary_df["sensor_mode"] == sensor_mode]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = [
        ("same_material", "Same material", "tab:blue", "o"),
        ("different_material", "Different material", "tab:orange", "s"),
    ]
    for pair_type, label, color, marker in styles:
        df = sensor_df[sensor_df["pair_type"] == pair_type].sort_values("window_ms")
        ax.errorbar(
            df["window_ms"],
            df["cosine_mean"],
            yerr=df["cosine_std"],
            color=color,
            marker=marker,
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([str(v) for v in WINDOWS_MS])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Window size (ms)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Liquid spike-count similarity | {SENSOR_LABELS[sensor_mode]}")
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "liquid_cosine_similarity_by_window.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def material_similarity_matrix(detail_df: pd.DataFrame, window_ms: int) -> np.ndarray:
    window_df = detail_df[detail_df["window_ms"] == window_ms]
    material_to_idx = {material: idx for idx, material in enumerate(MATERIALS)}
    matrix = np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float)
    for row in window_df.itertuples(index=False):
        idx_a = material_to_idx[row.material_a]
        idx_b = material_to_idx[row.material_b]
        matrix[idx_a, idx_b] = row.cosine_mean
        matrix[idx_b, idx_a] = row.cosine_mean
    return matrix


def plot_material_heatmaps(detail_df: pd.DataFrame,
                           out_dir: Path,
                           sensor_mode: str,
                           windows_ms: list[int]):
    heatmap_dir = out_dir / "material_cosine_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for window_ms in windows_ms:
        matrix = material_similarity_matrix(detail_df, window_ms)
        fig, ax = plt.subplots(figsize=(8.0, 6.8))
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(MATERIALS)))
        ax.set_yticks(np.arange(len(MATERIALS)))
        ax.set_xticklabels(MATERIALS, rotation=40, ha="right")
        ax.set_yticklabels(MATERIALS)
        ax.set_title(
            f"Material cosine similarity | {SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
        )
        for row_idx in range(len(MATERIALS)):
            for col_idx in range(len(MATERIALS)):
                value = matrix[row_idx, col_idx]
                text_color = "white" if value < 0.55 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )
        fig.colorbar(image, ax=ax, label="Mean cosine similarity")
        fig.tight_layout()
        out_path = heatmap_dir / f"material_cosine_heatmap_{window_ms}ms.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[saved] {out_path}")

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.0), sharex=True, sharey=True)
    image = None
    for ax, window_ms in zip(axes.flat, windows_ms):
        matrix = material_similarity_matrix(detail_df, window_ms)
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(MATERIALS)))
        ax.set_yticks(np.arange(len(MATERIALS)))
        ax.set_xticklabels(MATERIALS, rotation=55, ha="right", fontsize=7)
        ax.set_yticklabels(MATERIALS, fontsize=7)
        ax.set_title(f"{window_ms} ms")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Mean cosine similarity", shrink=0.85)
    fig.suptitle(f"Material cosine similarity | {SENSOR_LABELS[sensor_mode]}")
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.90, wspace=0.18, hspace=0.25)
    out_path = out_dir / "material_cosine_heatmaps_all_windows.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_all_sensors(summary_df: pd.DataFrame, result_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True, sharey=True)
    for ax, sensor_mode in zip(axes.flat, SENSOR_ORDER):
        sensor_df = summary_df[summary_df["sensor_mode"] == sensor_mode]
        for pair_type, label, color, marker in [
            ("same_material", "Same material", "tab:blue", "o"),
            ("different_material", "Different material", "tab:orange", "s"),
        ]:
            df = sensor_df[sensor_df["pair_type"] == pair_type].sort_values("window_ms")
            ax.plot(
                df["window_ms"],
                df["cosine_mean"],
                color=color,
                marker=marker,
                linewidth=1.7,
                label=label,
            )
        ax.set_xticks(WINDOWS_MS)
        ax.set_xticklabels([str(v) for v in WINDOWS_MS])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(SENSOR_LABELS[sensor_mode])
        ax.grid(axis="both", alpha=0.25)
    axes[1, 0].set_xlabel("Window size (ms)")
    axes[1, 1].set_xlabel("Window size (ms)")
    axes[0, 0].set_ylabel("Cosine similarity")
    axes[1, 0].set_ylabel("Cosine similarity")
    axes[0, 0].legend()
    fig.suptitle("Liquid spike-count cosine similarity")
    fig.tight_layout()
    out_path = result_dir / "liquid_cosine_similarity_by_window_all_sensors.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    parser.add_argument("--windows-ms", default="5,10,20,25,50,100,250,500")
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]
    windows_ms = [int(v) for v in args.windows_ms.split(",") if v.strip()]
    invalid = [window for window in windows_ms if 500 % window != 0]
    if invalid:
        raise ValueError(f"Window sizes must divide 500 ms exactly: {invalid}")

    all_summary_rows = []
    all_detail_rows = []
    all_time_rows = []
    for sensor_mode in sensor_modes:
        rec_path = args.result_dir / sensor_mode / "liquid_sout_rec_rep1.npy"
        if not rec_path.exists():
            raise FileNotFoundError(rec_path)
        sout_rec = np.load(rec_path, mmap_mode="r")
        n_class, n_sample, _n_neuron, _total_ms = sout_rec.shape
        labels = np.repeat(np.arange(n_class), n_sample)
        print(f"[loaded] {rec_path} shape={sout_rec.shape}")

        for window_ms in windows_ms:
            print(f"[{sensor_mode}] window={window_ms} ms")
            features = build_sparse_features(sout_rec, window_ms)
            similarity = cosine_matrix(features)
            summary_rows, detail_rows = summarize_similarity(
                similarity=similarity,
                labels=labels,
                sensor_mode=sensor_mode,
                window_ms=window_ms,
            )
            all_summary_rows.extend(summary_rows)
            all_detail_rows.extend(detail_rows)
            time_rows = summarize_material_pair_time_series(
                sout_rec=sout_rec,
                sensor_mode=sensor_mode,
                window_ms=window_ms,
            )
            all_time_rows.extend(time_rows)

        sensor_summary = pd.DataFrame(
            [row for row in all_summary_rows if row["sensor_mode"] == sensor_mode]
        )
        sensor_detail = pd.DataFrame(
            [row for row in all_detail_rows if row["sensor_mode"] == sensor_mode]
        )
        sensor_time = pd.DataFrame(
            [row for row in all_time_rows if row["sensor_mode"] == sensor_mode]
        )
        out_dir = args.result_dir / sensor_mode
        summary_path = out_dir / "liquid_cosine_similarity_by_window_summary.csv"
        detail_path = out_dir / "liquid_cosine_similarity_by_material_pair.csv"
        time_path = out_dir / "liquid_cosine_similarity_material_pair_time.csv"
        sensor_summary.to_csv(summary_path, index=False)
        sensor_detail.to_csv(detail_path, index=False)
        sensor_time.to_csv(time_path, index=False)
        print(f"[saved] {summary_path}")
        print(f"[saved] {detail_path}")
        print(f"[saved] {time_path}")
        plot_one_sensor(sensor_summary, out_dir, sensor_mode)
        plot_material_heatmaps(sensor_detail, out_dir, sensor_mode, windows_ms)
        time_plot_dir = out_dir / "material_pair_cosine_time"
        time_plot_dir.mkdir(parents=True, exist_ok=True)
        for window_ms in windows_ms:
            window_time_df = sensor_time[sensor_time["window_ms"] == window_ms]
            plot_material_pair_time_heatmap(
                window_time_df,
                time_plot_dir,
                sensor_mode,
                window_ms,
            )
            plot_same_different_time_series(
                window_time_df,
                time_plot_dir,
                sensor_mode,
                window_ms,
            )

    summary_df = pd.DataFrame(all_summary_rows)
    detail_df = pd.DataFrame(all_detail_rows)
    time_df = pd.DataFrame(all_time_rows)
    summary_path = args.result_dir / "liquid_cosine_similarity_by_window_summary.csv"
    detail_path = args.result_dir / "liquid_cosine_similarity_by_material_pair.csv"
    time_path = args.result_dir / "liquid_cosine_similarity_material_pair_time.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    time_df.to_csv(time_path, index=False)
    print(f"[saved] {summary_path}")
    print(f"[saved] {detail_path}")
    print(f"[saved] {time_path}")
    if set(sensor_modes) == set(SENSOR_ORDER):
        plot_all_sensors(summary_df, args.result_dir)


if __name__ == "__main__":
    main()
