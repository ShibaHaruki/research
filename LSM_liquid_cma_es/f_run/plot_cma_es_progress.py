"""Plot and summarize CMA-ES search progress.

Use this after run_cma_es_search.py to check whether the population is moving
toward better liquid parameters. Lower objective is better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d_tools.run_paths import jsonable


DEFAULT_SEARCH_DIR = (
    PROJECT_ROOT
    / "g_tactile_results"
    / "cma_es_search"
    / "liquid_accuracy_fisher"
)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_generation_summary(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    numeric_columns = [
        "generation",
        "candidate",
        "objective",
        "accuracy8_overall_mean",
        "accuracy8_overall_std",
        "accuracy3_overall_mean",
        "accuracy3_overall_std",
        "fisher_ratio_DR_mean",
        "fisher_ratio_DR_std",
    ]
    df = _to_numeric(df, numeric_columns)
    df = df.dropna(subset=["generation", "objective"])
    if df.empty:
        raise ValueError(f"No valid objective rows found in {results_csv}")

    rows = []
    best_so_far = float("inf")
    for generation, group in df.groupby("generation", sort=True):
        group = group.sort_values("objective")
        best_row = group.iloc[0]
        best_so_far = min(best_so_far, float(best_row["objective"]))
        row = {
            "generation": int(generation),
            "n_candidates": int(len(group)),
            "objective_best": float(best_row["objective"]),
            "objective_best_so_far": float(best_so_far),
            "objective_mean": float(group["objective"].mean()),
            "objective_std": float(group["objective"].std(ddof=0)),
            "objective_median": float(group["objective"].median()),
            "best_candidate": int(best_row["candidate"]),
        }
        for metric in (
            "accuracy8_overall_mean",
            "accuracy8_overall_std",
            "accuracy3_overall_mean",
            "accuracy3_overall_std",
            "fisher_ratio_DR_mean",
            "fisher_ratio_DR_std",
        ):
            if metric in group.columns:
                row[f"best_{metric}"] = (
                    None if pd.isna(best_row[metric]) else float(best_row[metric])
                )
                row[f"mean_{metric}"] = float(group[metric].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if len(summary) >= 2:
        summary["objective_best_delta"] = summary["objective_best"].diff()
        summary["objective_best_so_far_delta"] = summary["objective_best_so_far"].diff()
    else:
        summary["objective_best_delta"] = np.nan
        summary["objective_best_so_far_delta"] = np.nan
    return summary


def save_plots(summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    gen = summary["generation"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(gen, summary["objective_best"], marker="o", label="best in generation")
    axes[0].plot(gen, summary["objective_best_so_far"], marker="s", label="best so far")
    axes[0].plot(gen, summary["objective_mean"], marker=".", alpha=0.7, label="population mean")
    axes[0].fill_between(
        gen,
        summary["objective_mean"] - summary["objective_std"],
        summary["objective_mean"] + summary["objective_std"],
        alpha=0.15,
        label="population std",
    )
    axes[0].set_ylabel("Objective (lower is better)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "best_accuracy8_overall_mean" in summary.columns:
        axes[1].plot(
            gen,
            summary["best_accuracy8_overall_mean"],
            marker="o",
            label="best candidate acc8",
        )
        axes[1].plot(
            gen,
            summary["mean_accuracy8_overall_mean"],
            marker=".",
            alpha=0.7,
            label="population mean acc8",
        )
    if "best_accuracy3_overall_mean" in summary.columns:
        axes[1].plot(
            gen,
            summary["best_accuracy3_overall_mean"],
            marker="s",
            label="best candidate acc3",
        )
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if "best_fisher_ratio_DR_mean" in summary.columns:
        axes[2].plot(
            gen,
            summary["best_fisher_ratio_DR_mean"],
            marker="o",
            label="best candidate Fisher DR",
        )
        axes[2].plot(
            gen,
            summary["mean_fisher_ratio_DR_mean"],
            marker=".",
            alpha=0.7,
            label="population mean Fisher DR",
        )
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Fisher ratio DR")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "cma_es_progress.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CMA-ES search progress.")
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--results-csv", type=Path, default=None)
    args = parser.parse_args()

    search_dir = Path(args.search_dir)
    results_csv = Path(args.results_csv) if args.results_csv else search_dir / "cma_es_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"{results_csv} was not found")

    summary = build_generation_summary(results_csv)
    out_dir = search_dir / "progress"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "cma_es_generation_summary.csv"
    summary_json = out_dir / "cma_es_generation_summary.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(jsonable(summary.to_dict(orient="records")), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_plots(summary, out_dir)

    first = summary.iloc[0]
    last = summary.iloc[-1]
    print(f"[cma-progress] generations={len(summary)}")
    print(
        "[cma-progress] objective best-so-far "
        f"{first['objective_best_so_far']:.6g} -> {last['objective_best_so_far']:.6g}"
    )
    if "best_accuracy8_overall_mean" in summary.columns:
        print(
            "[cma-progress] best acc8 "
            f"{first['best_accuracy8_overall_mean']:.4f} -> "
            f"{last['best_accuracy8_overall_mean']:.4f}"
        )
    print(f"[cma-progress] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
