"""CMA-ES の history.csv から目的関数やペナルティの推移グラフを保存する。"""

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

from c_configs.FIXED import cfg_run
from d_tools.plotting import try_import_pyplot


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
CMA_RESULT_DIR = RESULTS_PATH / str(RUN_CFG.get("CMA_ES_RESULT_DIR", "cma_es_search"))


def _latest_search_dir(root: Path) -> Path:
    root = Path(root)
    candidates = [child for child in root.iterdir() if child.is_dir() and (child / "history.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No CMA-ES history.csv found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_details(text: object) -> dict:
    if not isinstance(text, str) or not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def load_history(search_dir: Path) -> pd.DataFrame:
    # CMA-ES の history.csv と metric_details_json を読み、グラフ用の列へ展開する。
    history_csv = Path(search_dir) / "history.csv"
    if not history_csv.exists():
        raise FileNotFoundError(history_csv)
    df = pd.read_csv(history_csv)
    if "metric_details_json" in df.columns:
        details = [_parse_details(value) for value in df["metric_details_json"]]
        for key in (
            "J",
            "silhouette",
            "DR",
            "firing_rate_penalty",
            "synchrony_penalty",
            "penalty_total",
            "mean_rate_hz",
            "sync_index",
        ):
            df[key] = [item.get(key, np.nan) for item in details]
    if "J" not in df.columns and "score" in df.columns:
        df["J"] = df["score"]
    return df


def summarize_by_generation(df: pd.DataFrame) -> pd.DataFrame:
    if "generation" not in df.columns:
        raise ValueError("history.csv has no generation column.")
    numeric_cols = [
        col
        for col in (
            "score",
            "J",
            "silhouette",
            "DR",
            "firing_rate_penalty",
            "synchrony_penalty",
            "penalty_total",
            "mean_rate_hz",
            "sync_index",
        )
        if col in df.columns
    ]
    rows = []
    for generation, group in df.groupby("generation", sort=True):
        row = {"generation": int(generation)}
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_max"] = float(values.max())
            row[f"{col}_min"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if "score_max" in summary.columns:
        summary["best_score_so_far"] = summary["score_max"].cummax()
    if "J_max" in summary.columns:
        summary["best_J_so_far"] = summary["J_max"].cummax()
    return summary


def _line_plot(ax, x, y, *, label: str, marker: str = "o"):
    values = pd.to_numeric(y, errors="coerce")
    ax.plot(x, values, marker=marker, linewidth=1.8, label=label)


def save_progress_plots(search_dir: Path) -> dict:
    # 目的関数 J、Silhouette/DR、ペナルティ、発火率/同期度の推移を保存する。
    plt = try_import_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is required to save progress plots.")

    search_dir = Path(search_dir)
    plot_dir = search_dir / "progress_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_history(search_dir)
    summary = summarize_by_generation(df)
    summary_csv = plot_dir / "generation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    x = summary["generation"]
    saved = {"summary_csv": str(summary_csv)}

    fig, ax = plt.subplots(figsize=(9, 5))
    if "J_max" in summary.columns:
        _line_plot(ax, x, summary["J_max"], label="best J in generation")
    if "J_mean" in summary.columns:
        _line_plot(ax, x, summary["J_mean"], label="mean J", marker="s")
    if "best_J_so_far" in summary.columns:
        _line_plot(ax, x, summary["best_J_so_far"], label="best J so far", marker="^")
    elif "best_score_so_far" in summary.columns:
        _line_plot(ax, x, summary["best_score_so_far"], label="best score so far", marker="^")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Objective J")
    ax.set_title("CMA-ES objective improvement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fp = plot_dir / "objective_J_progress.png"
    fig.savefig(fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved["objective_plot"] = str(fp)

    fig, ax = plt.subplots(figsize=(9, 5))
    if "silhouette_max" in summary.columns:
        _line_plot(ax, x, summary["silhouette_max"], label="best Silhouette")
    if "DR_max" in summary.columns:
        _line_plot(ax, x, summary["DR_max"], label="best DR", marker="s")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Metric value")
    ax.set_title("Silhouette and DR progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fp = plot_dir / "silhouette_DR_progress.png"
    fig.savefig(fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved["silhouette_dr_plot"] = str(fp)

    fig, ax = plt.subplots(figsize=(9, 5))
    if "firing_rate_penalty_min" in summary.columns:
        _line_plot(ax, x, summary["firing_rate_penalty_min"], label="min firing-rate penalty")
    if "synchrony_penalty_min" in summary.columns:
        _line_plot(ax, x, summary["synchrony_penalty_min"], label="min synchrony penalty", marker="s")
    if "penalty_total_min" in summary.columns:
        _line_plot(ax, x, summary["penalty_total_min"], label="min total penalty", marker="^")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Penalty")
    ax.set_title("Penalty progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fp = plot_dir / "penalty_progress.png"
    fig.savefig(fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved["penalty_plot"] = str(fp)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    if "mean_rate_hz_mean" in summary.columns:
        ax1.plot(x, summary["mean_rate_hz_mean"], marker="o", linewidth=1.8, label="mean rate")
        ax1.set_ylabel("Mean firing rate [Hz]")
    ax2 = ax1.twinx()
    if "sync_index_mean" in summary.columns:
        ax2.plot(
            x,
            summary["sync_index_mean"],
            marker="s",
            linewidth=1.8,
            color="tab:orange",
            label="sync index",
        )
        ax2.set_ylabel("Synchrony index")
    ax1.set_xlabel("Generation")
    ax1.set_title("Activity statistics")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fp = plot_dir / "activity_stats_progress.png"
    fig.savefig(fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved["activity_plot"] = str(fp)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CMA-ES search progress from history.csv.")
    parser.add_argument(
        "search_dir",
        nargs="?",
        default=None,
        help="CMA-ES result directory. If omitted, use the latest under g_tactile_results/cma_es_search.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    search_dir = Path(args.search_dir) if args.search_dir else _latest_search_dir(CMA_RESULT_DIR)
    saved = save_progress_plots(search_dir)
    print(f"[cma-es progress] search_dir={search_dir}")
    for key, value in saved.items():
        print(f"[saved] {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
