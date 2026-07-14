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
    / "liquid_accuracy_spikes_fisher"
)

DEFAULT_OBJECTIVE_WEIGHTS = {
    "alpha": 100.0,
    "beta": 1.0,
    "gamma": 1.0,
    "delta": 100.0,
    "epsilon": 1.0,
}


def load_objective_weights(search_dir: Path) -> dict[str, float]:
    weights = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    settings_path = search_dir / "search_settings.json"
    if not settings_path.exists():
        return weights
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    aliases = {
        "alpha": ("alpha", "\u03b1"),
        "beta": ("beta", "\u03b2"),
        "gamma": ("gamma", "\u03b3"),
        "delta": ("delta", "\u03b4"),
        "epsilon": ("epsilon", "\u03b5"),
    }
    for name, keys in aliases.items():
        for key in keys:
            if key in payload:
                weights[name] = float(payload[key])
                break
    return weights


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_generation_summary(
    results_csv: Path,
    objective_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
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

    weights = objective_weights or DEFAULT_OBJECTIVE_WEIGHTS
    component_sources = {
        "accuracy_contribution": (
            "accuracy8_overall_mean", -weights["alpha"]
        ),
        "accuracy_variance_contribution": (
            "accuracy8_overall_variance", weights["beta"]
        ),
        "spike_penalty_contribution": ("spike_ratio", weights["gamma"]),
        "silent_penalty_contribution": (
            "silent_neuron_fraction", weights["delta"]
        ),
        "fisher_contribution": ("fisher_ratio_DR_mean", -weights["epsilon"]),
    }
    for name, (source, weight) in component_sources.items():
        if source in df.columns:
            df[name] = pd.to_numeric(df[source], errors="coerce") * float(weight)

    known_columns = {
        "start", "generation", "candidate", "objective", "metrics",
        "accuracy8_overall_mean", "accuracy8_overall_std",
        "accuracy8_overall_variance", "accuracy3_overall_mean",
        "accuracy3_overall_std", "accuracy3_overall_variance",
        "fisher_ratio_DR_mean", "fisher_ratio_DR_std", "spike_base",
        "spike_ratio", "mean_total_spikes_per_trial", "std_total_spikes_per_trial",
        "total_spikes_all_trials", "silent_neuron_count", "total_neuron_count",
        "active_neuron_count", "silent_neuron_fraction", "n_spike_trials",
        "n_activity_trials", "n_activity_mismatched_shapes",
        *component_sources,
    }
    parameter_columns = [
        column for column in df.columns
        if column not in known_columns and pd.api.types.is_numeric_dtype(df[column])
    ]

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
            "accuracy8_overall_variance",
            "accuracy3_overall_mean",
            "accuracy3_overall_std",
            "fisher_ratio_DR_mean",
            "fisher_ratio_DR_std",
            "mean_total_spikes_per_trial",
            "spike_ratio",
            "silent_neuron_fraction",
        ):
            if metric in group.columns:
                row[f"best_{metric}"] = (
                    None if pd.isna(best_row[metric]) else float(best_row[metric])
                )
                row[f"mean_{metric}"] = float(group[metric].mean())
        for component in component_sources:
            if component in group.columns:
                row[f"best_{component}"] = float(best_row[component])
                row[f"mean_{component}"] = float(group[component].mean())
        for parameter in parameter_columns:
            row[f"best_{parameter}"] = float(best_row[parameter])
            row[f"mean_{parameter}"] = float(group[parameter].mean())
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

    fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)

    objective_axis = axes[0, 0]
    objective_axis.plot(gen, summary["objective_best"], marker="o", label="best in generation")
    objective_axis.plot(gen, summary["objective_best_so_far"], marker="s", label="best so far")
    objective_axis.plot(gen, summary["objective_mean"], marker=".", alpha=0.7, label="population mean")
    objective_axis.fill_between(
        gen,
        summary["objective_mean"] - summary["objective_std"],
        summary["objective_mean"] + summary["objective_std"],
        alpha=0.15,
        label="population std",
    )
    objective_axis.set_title("Objective (weighted sum)")
    objective_axis.set_ylabel("Objective value")
    objective_axis.grid(True, alpha=0.3)
    objective_axis.legend()

    component_plots = [
        ("accuracy8_overall_mean", "Accuracy"),
        ("accuracy8_overall_variance", "Accuracy variance"),
        ("mean_total_spikes_per_trial", "Mean total spikes"),
        ("silent_neuron_fraction", "Silent-neuron fraction"),
        ("fisher_ratio_DR_mean", "Fisher ratio DR"),
    ]
    for axis, (column, title) in zip(axes.flat[1:], component_plots):
        best_column = f"best_{column}"
        mean_column = f"mean_{column}"
        if best_column in summary.columns:
            axis.plot(gen, summary[best_column], marker="o", label="best")
            axis.plot(gen, summary[mean_column], marker=".", alpha=0.7, label="mean")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.set_ylabel("Raw value")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes[-1, :]:
        axis.set_xlabel("Generation")

    fig.suptitle("CMA-ES progress and objective contributions")

    fig.tight_layout()
    fig.savefig(out_dir / "cma_es_progress.png", dpi=160)
    plt.close(fig)

    parameter_columns = sorted(
        column.removeprefix("best_")
        for column in summary.columns
        if column.startswith("best_")
        and column not in {"best_candidate", "best_so_far"}
        and f"mean_{column.removeprefix('best_')}" in summary.columns
        and column.removeprefix("best_") not in {
            "objective", "accuracy8_overall_mean", "accuracy3_overall_mean",
            "accuracy8_overall_variance", "mean_total_spikes_per_trial",
            "spike_ratio", "silent_neuron_fraction",
            "fisher_ratio_DR_mean", "accuracy8_overall_std", "accuracy3_overall_std",
            "fisher_ratio_DR_std", "accuracy_variance_contribution",
            "accuracy_contribution",
            "spike_penalty_contribution", "silent_penalty_contribution",
            "fisher_contribution",
        }
    )
    if parameter_columns:
        ncols = 3
        nrows = int(np.ceil(len(parameter_columns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), squeeze=False)
        for axis, parameter in zip(axes.flat, parameter_columns):
            axis.plot(gen, summary[f"best_{parameter}"], marker="o", label="best")
            axis.plot(gen, summary[f"mean_{parameter}"], marker=".", alpha=0.7, label="mean")
            axis.set_title(parameter)
            axis.set_xlabel("Generation")
            axis.grid(True, alpha=0.3)
            axis.legend()
        for axis in axes.flat[len(parameter_columns):]:
            axis.set_visible(False)
        fig.suptitle("CMA-ES parameter progress")
        fig.tight_layout()
        fig.savefig(out_dir / "cma_es_parameter_progress.png", dpi=160)
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

    summary = build_generation_summary(
        results_csv,
        objective_weights=load_objective_weights(search_dir),
    )
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
