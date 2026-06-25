"""内部状態や特徴量に PCA をかけ、2D/3D の図と CSV を保存する処理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .plotting import try_import_pyplot
from .run_paths import jsonable
from .separation_metrics import discover_internal_state_files, load_internal_state_dataset


MARKERS = ["o", "s", "^", "D", "v", "x", "*", "+"]
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def _stack_class_features(
    features_by_class: Sequence[np.ndarray],
    labels: Sequence[str],
    files_by_class: dict[str, list[str]],
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    rows = []
    row_labels = []
    sample_indices = []
    source_files = []

    for class_index, label in enumerate(labels):
        features = np.asarray(features_by_class[class_index], dtype=np.float64)
        files = files_by_class.get(str(label), [])
        for sample_index in range(features.shape[0]):
            rows.append(features[sample_index])
            row_labels.append(str(label))
            sample_indices.append(sample_index)
            source_files.append(files[sample_index] if sample_index < len(files) else "")

    if not rows:
        raise ValueError("No feature rows were loaded for PCA.")
    return np.vstack(rows), row_labels, sample_indices, source_files


def _select_internal_state_files(
    internal_state_dir: Path,
    *,
    max_samples_per_class: int | None = None,
) -> tuple[list[str], list[str], list[int], list[Path]]:
    material_to_files = discover_internal_state_files(internal_state_dir)
    materials = sorted(material_to_files)
    row_labels: list[str] = []
    sample_indices: list[int] = []
    files: list[Path] = []

    for material in materials:
        selected = material_to_files[material]
        if max_samples_per_class is not None:
            selected = selected[: int(max_samples_per_class)]
        for sample_index, fp in enumerate(selected):
            row_labels.append(str(material))
            sample_indices.append(sample_index)
            files.append(Path(fp))

    if not files:
        raise FileNotFoundError(f"No selected internal state files found under {internal_state_dir}")
    return materials, row_labels, sample_indices, files


def _window_mask(
    t_ms: np.ndarray,
    *,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> np.ndarray | None:
    if window_start_ms is None and window_end_ms is None:
        return None
    t_arr = np.asarray(t_ms, dtype=np.float64)
    mask = np.ones(t_arr.shape, dtype=bool)
    if window_start_ms is not None:
        mask &= t_arr >= float(window_start_ms)
    if window_end_ms is not None:
        mask &= t_arr <= float(window_end_ms)
    if not np.any(mask):
        raise ValueError("selected internal-state time window is empty.")
    return mask


def _load_x_state(
    fp: Path,
    *,
    n_neurons: int | None = None,
    n_time: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> np.ndarray:
    with np.load(fp) as data:
        x_state = np.asarray(data["x_state"], dtype=np.float32)
        if window_start_ms is not None or window_end_ms is not None:
            mask = _window_mask(
                np.asarray(data["t_ms"], dtype=np.float64),
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
            if mask is not None:
                x_state = x_state[:, mask]

    if n_neurons is not None:
        x_state = x_state[: int(n_neurons), :]
    if n_time is not None:
        x_state = x_state[:, : int(n_time)]
    return np.asarray(x_state, dtype=np.float32)


def _common_state_shape(
    files: Sequence[Path],
    *,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> tuple[int, int]:
    min_neurons: int | None = None
    min_time: int | None = None
    for fp in files:
        x_state = _load_x_state(
            fp,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        n_neurons, n_time = x_state.shape
        min_neurons = n_neurons if min_neurons is None else min(min_neurons, n_neurons)
        min_time = n_time if min_time is None else min(min_time, n_time)
    if min_neurons is None or min_time is None or min_neurons <= 0 or min_time <= 0:
        raise ValueError("internal-state files have empty x_state arrays.")
    return int(min_neurons), int(min_time)


def _flatten_x_state(x_state: np.ndarray) -> np.ndarray:
    return np.asarray(x_state, dtype=np.float64).reshape(-1)


def fit_internal_state_flatten_pca(
    files: Sequence[Path],
    out_dir: Path,
    *,
    n_components: int = 2,
    standardize: bool = True,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
    feature_chunk_size: int = 10000,
    eps: float = 1e-12,
) -> dict:
    """Fit 2D PCA on all liquid neurons and all time steps.

    This follows the same idea as ``StandardScaler`` + covariance PCA, but uses
    the sample-space Gram matrix so that ``n_neurons * n_time`` can be large.
    """

    # 1000ニューロン×50時間窓のように特徴次元が大きいので、サンプル空間の Gram 行列で計算する。
    files = [Path(fp) for fp in files]
    if not files:
        raise ValueError("No files were provided for PCA.")

    out_dir = Path(out_dir)
    n_samples = len(files)
    n_neurons, n_time = _common_state_shape(
        files,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )
    n_features = int(n_neurons * n_time)

    sums = np.zeros(n_features, dtype=np.float64)
    sumsq = np.zeros(n_features, dtype=np.float64)
    for fp in files:
        vec = _flatten_x_state(
            _load_x_state(
                fp,
                n_neurons=n_neurons,
                n_time=n_time,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
        )
        sums += vec
        sumsq += vec * vec

    mean = sums / float(n_samples)
    if standardize:
        variance = np.maximum(sumsq / float(n_samples) - mean * mean, 0.0)
        scale = np.sqrt(variance)
        scale = np.where(scale > eps, scale, 1.0)
    else:
        scale = np.ones(n_features, dtype=np.float64)

    tmp_fp = out_dir / "_pca_flatten_standardized_float32.dat"
    x_std = np.memmap(tmp_fp, dtype=np.float32, mode="w+", shape=(n_samples, n_features))
    try:
        for row_index, fp in enumerate(files):
            vec = _flatten_x_state(
                _load_x_state(
                    fp,
                    n_neurons=n_neurons,
                    n_time=n_time,
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                )
            )
            x_std[row_index, :] = ((vec - mean) / scale).astype(np.float32, copy=False)
        x_std.flush()

        gram = np.zeros((n_samples, n_samples), dtype=np.float64)
        chunk_size = max(1, int(feature_chunk_size))
        for start in range(0, n_features, chunk_size):
            stop = min(start + chunk_size, n_features)
            chunk = np.asarray(x_std[:, start:stop], dtype=np.float64)
            gram += chunk @ chunk.T

        eig_values, eig_vectors = np.linalg.eigh(gram)
        order = np.argsort(eig_values.real)[::-1]
        eig_values = np.maximum(eig_values.real[order], 0.0)
        eig_vectors = eig_vectors[:, order].real

        max_components = int(min(n_components, n_samples, eig_vectors.shape[1]))
        singular_values = np.sqrt(eig_values[:max_components])
        scores = eig_vectors[:, :max_components] * singular_values[None, :]

        components = np.zeros((max_components, n_features), dtype=np.float32)
        for start in range(0, n_features, chunk_size):
            stop = min(start + chunk_size, n_features)
            chunk = np.asarray(x_std[:, start:stop], dtype=np.float64)
            comp_chunk = np.zeros((max_components, stop - start), dtype=np.float64)
            for component_index in range(max_components):
                sv = singular_values[component_index]
                if sv > eps:
                    comp_chunk[component_index, :] = (
                        chunk.T @ eig_vectors[:, component_index]
                    ) / sv
            components[:, start:stop] = comp_chunk.astype(np.float32, copy=False)

        denom = max(n_samples - 1, 1)
        explained_variance = eig_values[:max_components] / float(denom)
        total_variance = float(np.trace(gram) / float(denom))
        if total_variance > eps:
            explained_ratio = explained_variance / total_variance
        else:
            explained_ratio = np.zeros_like(explained_variance)

        return {
            "scores": scores,
            "components": components,
            "mean": mean,
            "scale": scale,
            "singular_values": singular_values,
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_ratio,
            "total_variance": total_variance,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_components": int(max_components),
            "standardize": bool(standardize),
            "state_neurons": int(n_neurons),
            "state_time_steps": int(n_time),
        }
    finally:
        del x_std
        try:
            tmp_fp.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            pass


def fit_pca(
    X: np.ndarray,
    *,
    n_components: int = 2,
    standardize: bool = True,
    eps: float = 1e-12,
) -> dict:
    """Fit PCA with NumPy SVD and return scores/components/variance."""

    # 通常サイズの特徴行列には SVD を直接かける。必要なら列ごとに標準化する。
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={x.shape}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"X must be non-empty, got shape={x.shape}")

    mean = np.mean(x, axis=0)
    centered = x - mean
    if standardize:
        scale = np.std(centered, axis=0, ddof=1 if x.shape[0] > 1 else 0)
        scale = np.where(scale > eps, scale, 1.0)
    else:
        scale = np.ones(x.shape[1], dtype=np.float64)
    work = centered / scale

    _, singular_values, vt = np.linalg.svd(work, full_matrices=False)
    max_components = int(min(n_components, vt.shape[0]))
    components = vt[:max_components]
    scores = work @ components.T

    denom = max(x.shape[0] - 1, 1)
    explained_variance_all = (singular_values * singular_values) / float(denom)
    total_variance = float(np.sum(explained_variance_all))
    explained_variance = explained_variance_all[:max_components]
    if total_variance > eps:
        explained_ratio = explained_variance / total_variance
    else:
        explained_ratio = np.zeros_like(explained_variance)

    return {
        "scores": scores,
        "components": components,
        "mean": mean,
        "scale": scale,
        "singular_values": singular_values[:max_components],
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
        "total_variance": total_variance,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_components": int(max_components),
        "standardize": bool(standardize),
    }


def _save_scores_csv(
    out_dir: Path,
    pca_result: dict,
    labels: Sequence[str],
    sample_indices: Sequence[int],
    source_files: Sequence[str],
) -> Path:
    scores = np.asarray(pca_result["scores"], dtype=np.float64)
    rows = {
        "material": list(labels),
        "sample_index_in_material": list(sample_indices),
        "source_file": list(source_files),
    }
    for component_index in range(scores.shape[1]):
        rows[f"PC{component_index + 1}"] = scores[:, component_index]
    out_fp = Path(out_dir) / "pca_scores.csv"
    pd.DataFrame(rows).to_csv(out_fp, index=False)
    return out_fp


def _save_explained_variance(out_dir: Path, pca_result: dict) -> tuple[Path, Path]:
    ratios = np.asarray(pca_result["explained_variance_ratio"], dtype=np.float64)
    variances = np.asarray(pca_result["explained_variance"], dtype=np.float64)
    df = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(ratios))],
            "explained_variance": variances,
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": np.cumsum(ratios),
        }
    )
    csv_fp = Path(out_dir) / "pca_explained_variance.csv"
    df.to_csv(csv_fp, index=False)

    plot_fp = Path(out_dir) / "pca_explained_variance.png"
    plt = try_import_pyplot()
    if plt is None:
        return csv_fp, plot_fp

    plt.figure(figsize=(8, 4))
    x = np.arange(1, len(ratios) + 1)
    plt.bar(x, ratios, alpha=0.75, label="explained")
    plt.plot(x, np.cumsum(ratios), marker="o", color="black", label="cumulative")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.xticks(x)
    plt.ylim(0, max(1.0, float(np.max(np.cumsum(ratios))) if len(ratios) else 1.0))
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_fp, dpi=150)
    plt.close()
    return csv_fp, plot_fp


def _scatter_scores_2d(
    plt,
    scores: np.ndarray,
    labels: Sequence[str],
    x_index: int,
    y_index: int,
    plot_fp: Path,
) -> Path:
    unique_labels = list(dict.fromkeys(str(label) for label in labels))

    plt.figure(figsize=(8.5, 6.5))
    for label_index, label in enumerate(unique_labels):
        mask = np.asarray([str(item) == label for item in labels], dtype=bool)
        plt.scatter(
            scores[mask, x_index],
            scores[mask, y_index],
            c=COLORS[label_index % len(COLORS)],
            marker=MARKERS[label_index % len(MARKERS)],
            s=35,
            linewidths=1.0,
            alpha=0.8,
            label=label,
        )
    plt.xlabel(f"PC{x_index + 1}", fontsize=14)
    plt.ylabel(f"PC{y_index + 1}", fontsize=14)
    plt.title(f"Internal state PCA: PC{x_index + 1} vs PC{y_index + 1}", fontsize=18)
    if len(unique_labels) <= 12:
        plt.legend(fontsize=11, ncol=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_fp, dpi=150)
    plt.close()
    return plot_fp


def _scatter_scores_3d(
    plt,
    scores: np.ndarray,
    labels: Sequence[str],
    plot_fp: Path,
) -> Path:
    unique_labels = list(dict.fromkeys(str(label) for label in labels))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for label in unique_labels:
        mask = np.asarray([str(item) == label for item in labels], dtype=bool)
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            scores[mask, 2],
            s=30,
            alpha=0.55,
            edgecolors="black",
            linewidths=0.25,
            label=label,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Internal state PCA: PC1/PC2/PC3")
    if len(unique_labels) <= 12:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_fp, dpi=150)
    plt.close()
    return plot_fp


def _save_score_plots(out_dir: Path, pca_result: dict, labels: Sequence[str]) -> list[Path]:
    scores = np.asarray(pca_result["scores"], dtype=np.float64)
    if scores.shape[1] < 2:
        return []
    plt = try_import_pyplot()
    if plt is None:
        return []

    out_dir = Path(out_dir)
    plot_fps = [
        _scatter_scores_2d(
            plt,
            scores,
            labels,
            0,
            1,
            out_dir / "pca_scores_pc1_pc2.png",
        )
    ]
    if scores.shape[1] >= 3:
        plot_fps.append(
            _scatter_scores_2d(
                plt,
                scores,
                labels,
                0,
                2,
                out_dir / "pca_scores_pc1_pc3.png",
            )
        )
        plot_fps.append(
            _scatter_scores_2d(
                plt,
                scores,
                labels,
                1,
                2,
                out_dir / "pca_scores_pc2_pc3.png",
            )
        )
        plot_fps.append(
            _scatter_scores_3d(
                plt,
                scores,
                labels,
                out_dir / "pca_scores_pc1_pc2_pc3_3d.png",
            )
        )
    return plot_fps


def _write_summary(out_dir: Path, payload: dict) -> Path:
    out_fp = Path(out_dir) / "pca_summary.json"
    out_fp.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_fp


def save_internal_state_pca(
    internal_state_dir: Path,
    out_dir: Path,
    *,
    feature_mode: str = "flatten",
    n_components: int = 2,
    standardize: bool = True,
    max_samples_per_class: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> dict:
    """Load liquid internal states, fit PCA, and save PCA artifacts."""

    # Save PCA artifacts directly into the requested PCA directory.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_mode_key = str(feature_mode).lower()
    if feature_mode_key in {"flatten", "all", "all_time", "all_neurons_all_time"}:
        labels, row_labels, sample_indices, files = _select_internal_state_files(
            internal_state_dir,
            max_samples_per_class=max_samples_per_class,
        )
        source_files = [str(fp) for fp in files]
        pca_result = fit_internal_state_flatten_pca(
            files,
            out_dir,
            n_components=n_components,
            standardize=standardize,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
    else:
        dataset = load_internal_state_dataset(
            internal_state_dir,
            feature_mode=feature_mode,
            max_samples_per_class=max_samples_per_class,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            load_trajectories=False,
        )
        labels = list(dataset["materials"])
        X, row_labels, sample_indices, source_files = _stack_class_features(
            dataset["features_by_class"],
            labels,
            dataset["files_by_class"],
        )
        pca_result = fit_pca(X, n_components=n_components, standardize=standardize)

    scores_fp = _save_scores_csv(out_dir, pca_result, row_labels, sample_indices, source_files)
    variance_csv_fp, variance_plot_fp = _save_explained_variance(out_dir, pca_result)
    score_plot_fps = _save_score_plots(out_dir, pca_result, row_labels)
    rounded_scores = np.round(np.asarray(pca_result["scores"], dtype=np.float64), decimals=12)
    unique_score_rows = int(np.unique(rounded_scores, axis=0).shape[0])
    model_fp = out_dir / "pca_model.npz"
    np.savez_compressed(
        model_fp,
        scores=pca_result["scores"],
        components=pca_result["components"],
        mean=pca_result["mean"],
        scale=pca_result["scale"],
        singular_values=pca_result["singular_values"],
        explained_variance=pca_result["explained_variance"],
        explained_variance_ratio=pca_result["explained_variance_ratio"],
        labels=np.asarray(row_labels, dtype=object),
        source_files=np.asarray(source_files, dtype=object),
    )

    summary = {
        "internal_state_dir": str(Path(internal_state_dir)),
        "out_dir": str(out_dir),
        "feature_mode": feature_mode,
        "n_components": pca_result["n_components"],
        "requested_components": int(n_components),
        "standardize": bool(standardize),
        "max_samples_per_class": max_samples_per_class,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "n_samples": pca_result["n_samples"],
        "n_features": pca_result["n_features"],
        "state_neurons": pca_result.get("state_neurons"),
        "state_time_steps": pca_result.get("state_time_steps"),
        "unique_score_rows_rounded_12": unique_score_rows,
        "materials": labels,
        "scores_csv": str(scores_fp),
        "explained_variance_csv": str(variance_csv_fp),
        "explained_variance_plot": (
            str(variance_plot_fp) if Path(variance_plot_fp).exists() else ""
        ),
        "scores_plot": str(score_plot_fps[0]) if score_plot_fps else "",
        "score_plots": [str(path) for path in score_plot_fps],
        "model_npz": str(model_fp),
    }
    summary_fp = _write_summary(out_dir, summary)
    summary["summary_json"] = str(summary_fp)
    return summary
