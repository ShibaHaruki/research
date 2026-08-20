"""Plot parameter correlations and PCA loadings from a CMA-ES result CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parameter_columns(df: pd.DataFrame) -> list[str]:
    metadata = {
        "start", "generation", "candidate", "objective", "metrics",
        "run_dir", "internal_state_dir", "accuracy_dir", "out_dir",
    }
    columns: list[str] = []
    for column in df.columns:
        if column in metadata:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 2:
            columns.append(column)
    # Search parameters are appended after the evaluation metrics in the CSV.
    # Keep only columns that are not known evaluation outputs.
    metric_prefixes = (
        "accuracy", "fisher", "trace_", "mean_", "std_", "total_", "active_",
        "silent_", "activity_", "firing_rate", "n_", "spike_", "target_", "objective_", "window_",
        "train_", "test_", "bin_", "T_", "hold", "seed", "start_",
    )
    return [
        column for column in columns
        if not column.startswith(metric_prefixes)
    ]


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    size = max(8.0, 0.42 * len(matrix.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", type=Path, required=True)
    args = parser.parse_args()

    search_dir = args.search_dir.resolve()
    results_csv = search_dir / "cma_es_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    out_dir = search_dir / "parameter_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    parameter_columns = _parameter_columns(df)
    if not parameter_columns:
        raise ValueError("No numeric parameter columns were found")

    parameters = df[parameter_columns].apply(pd.to_numeric, errors="coerce")
    parameters = parameters.dropna(axis=1, how="all")
    parameters = parameters.loc[parameters.notna().all(axis=1)]

    correlation = parameters.corr(method="spearman")
    correlation.to_csv(out_dir / "parameter_spearman_correlation.csv", encoding="utf-8-sig")
    _save_heatmap(
        correlation,
        out_dir / "parameter_spearman_correlation.png",
        "CMA-ES parameter correlation (Spearman)",
    )

    objective = pd.to_numeric(df.loc[parameters.index, "objective"], errors="coerce")
    objective_corr = parameters.corrwith(objective, method="spearman").sort_values()
    objective_corr.to_frame("spearman_correlation_with_objective").to_csv(
        out_dir / "parameter_objective_correlation.csv", encoding="utf-8-sig"
    )
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(objective_corr))))
    colors = ["tab:blue" if value < 0 else "tab:red" for value in objective_corr]
    ax.barh(objective_corr.index, objective_corr.to_numpy(), color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Spearman correlation with objective")
    ax.set_title("Parameter correlation with objective")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_objective_correlation.png", dpi=180)
    plt.close(fig)

    x = parameters.to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    n_components = min(2, vt.shape[0])
    loadings = pd.DataFrame(
        vt[:n_components].T,
        index=parameters.columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings.to_csv(out_dir / "parameter_pca_loadings.csv", encoding="utf-8-sig")
    explained = singular_values[:n_components] ** 2 / max(np.sum(singular_values ** 2), 1e-12)
    fig, axes = plt.subplots(1, n_components, figsize=(7 * n_components, 7), squeeze=False)
    for i, ax in enumerate(axes.flat):
        values = loadings.iloc[:, i].sort_values()
        ax.barh(values.index, values.to_numpy(), color=["tab:blue" if v < 0 else "tab:red" for v in values])
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"PC{i + 1} loadings ({explained[i] * 100:.1f}% variance)")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_pca_loadings.png", dpi=180)
    plt.close(fig)

    print(f"[parameter-analysis] rows={len(parameters)} parameters={len(parameters.columns)}")
    print(f"[parameter-analysis] output={out_dir}")
    print("[parameter-analysis] strongest objective correlations:")
    print(objective_corr.abs().sort_values(ascending=False).head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
