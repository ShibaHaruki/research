"""Plot the confusion matrix for the best CMA-ES candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SEARCH_DIR = (
    PROJECT_ROOT
    / "g_tactile_results"
    / "cma_es_search"
    / "liquid_accuracy_spikes_fisher"
)


def _candidate_dir(search_dir: Path, row: pd.Series) -> Path:
    start = int(row.get("start", 1))
    base = search_dir if start == 1 else search_dir / f"start{start:03d}"
    return base / f"gen{int(row['generation']):03d}_cand{int(row['candidate']):03d}"


def _select_best(results: pd.DataFrame, criterion: str) -> pd.Series:
    results = results.copy()
    for column in ("objective", "accuracy8_overall_mean"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results = results.dropna(subset=[criterion])
    if results.empty:
        raise ValueError(f"No valid rows found for criterion: {criterion}")
    if criterion == "accuracy8_overall_mean":
        return results.sort_values(
            [criterion, "objective"], ascending=[False, True]
        ).iloc[0]
    return results.sort_values(criterion, ascending=True).iloc[0]


def save_heatmap(matrix: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = matrix.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, data, axis_title, fmt in (
        (axes[0], values, "Counts", "{:.0f}"),
        (axes[1], normalized, "Row-normalized accuracy", "{:.2f}"),
    ):
        image = axis.imshow(data, cmap="Blues", vmin=0.0, vmax=float(np.max(data) or 1.0))
        axis.set_title(axis_title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
        threshold = float(np.max(data) or 1.0) * 0.55
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] > threshold else "black"
                axis.text(j, i, fmt.format(data[i, j]), ha="center", va="center", color=color)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the best CMA-ES confusion matrix.")
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument(
        "--criterion",
        choices=("accuracy8_overall_mean", "objective"),
        default="accuracy8_overall_mean",
        help="Select the candidate by highest accuracy or lowest objective.",
    )
    args = parser.parse_args()

    search_dir = Path(args.search_dir)
    results_path = search_dir / "cma_es_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    results = pd.read_csv(results_path)
    best = _select_best(results, args.criterion)
    candidate_dir = _candidate_dir(search_dir, best)
    matrix_path = candidate_dir / "random_neuron_accuracy" / "conf_8cls_repeat001.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    matrix = pd.read_csv(matrix_path, index_col=0)
    output_path = search_dir / "progress" / "best_confusion_matrix.png"
    title = (
        f"Best confusion matrix | gen={int(best['generation'])} "
        f"cand={int(best['candidate'])} | "
        f"accuracy8={float(best['accuracy8_overall_mean']):.4f}"
    )
    save_heatmap(matrix, output_path, title)
    print(f"[confusion] selected: {candidate_dir}")
    print(f"[confusion] saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
