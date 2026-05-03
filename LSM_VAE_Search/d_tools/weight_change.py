"""学習前後の平均重み変動を記録し、CSV とグラフにまとめる処理。"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEARNING_STATE_VARS = (
    "Apre",
    "Apost",
    "Aplus1",
    "Aplus2",
    "Aminus1",
    "Aminus2",
    "Mpre",
    "Mpost",
)


def read_synapse_weights(synapses) -> np.ndarray:
    if "w" not in synapses.variables:
        return np.array([], dtype=float)
    return np.asarray(synapses.w, dtype=float).reshape(-1).copy()


def is_learnable_synapse(synapses) -> bool:
    return "w" in synapses.variables and any(
        var_name in synapses.variables
        for var_name in LEARNING_STATE_VARS
    )


def learnable_synapses(synapses_list: list) -> list:
    return [
        synapses
        for synapses in synapses_list
        if is_learnable_synapse(synapses)
    ]


def synapse_layer_labels(synapse_name: str) -> tuple[int | None, int | None]:
    match = re.search(r"liq(\d+)_to_out(\d+)", synapse_name)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def make_weight_change_tracker(
    synapses_list: list,
    rng: np.random.Generator,
    n_trace: int,
) -> dict:
    # 学習中に追跡するシナプスを選び、個別トレース用の重み index もサンプリングする。
    targets = learnable_synapses(synapses_list)
    trace_indices = {}

    for synapses in targets:
        n_weights = len(synapses)
        n_pick = min(max(int(n_trace), 0), n_weights)
        if n_pick == 0:
            trace_indices[synapses.name] = np.array([], dtype=int)
        else:
            trace_indices[synapses.name] = np.sort(
                rng.choice(n_weights, size=n_pick, replace=False)
            ).astype(int)

    return {"synapses": targets, "trace_indices": trace_indices}


def snapshot_weight_tracker(tracker: dict) -> dict[str, np.ndarray]:
    return {
        synapses.name: read_synapse_weights(synapses)
        for synapses in tracker["synapses"]
    }


def weight_tracker_layers(tracker: dict) -> tuple[list[int], list[int]]:
    liquid_layers = set()
    output_layers = set()

    for synapses in tracker["synapses"]:
        liq_layer, out_layer = synapse_layer_labels(synapses.name)
        if liq_layer is not None:
            liquid_layers.add(int(liq_layer))
        if out_layer is not None:
            output_layers.add(int(out_layer))

    return sorted(liquid_layers), sorted(output_layers)


def weight_change_summary(before: np.ndarray, after: np.ndarray) -> dict:
    if before.size == 0 or after.size == 0:
        return {
            "n": 0,
            "mean_before": np.nan,
            "mean_after": np.nan,
            "mean_delta": np.nan,
            "mean_abs_delta": np.nan,
            "max_abs_delta": np.nan,
            "changed_count": 0,
        }

    delta = after - before
    abs_delta = np.abs(delta)
    return {
        "n": int(after.size),
        "mean_before": float(np.mean(before)),
        "mean_after": float(np.mean(after)),
        "mean_delta": float(np.mean(delta)),
        "mean_abs_delta": float(np.mean(abs_delta)),
        "max_abs_delta": float(np.max(abs_delta)),
        "changed_count": int(np.count_nonzero(abs_delta > 0.0)),
    }


def _accumulate_group_summary(store: dict[int, dict], group_value: int | None, summary: dict) -> None:
    if group_value is None:
        return

    n = int(summary.get("n", 0))
    if n <= 0:
        return

    key = int(group_value)
    item = store.setdefault(
        key,
        {
            "n": 0,
            "sum_mean_before": 0.0,
            "sum_mean_after": 0.0,
            "sum_mean_delta": 0.0,
            "sum_mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "changed_count": 0,
        },
    )
    item["n"] += n
    item["sum_mean_before"] += float(summary["mean_before"]) * n
    item["sum_mean_after"] += float(summary["mean_after"]) * n
    item["sum_mean_delta"] += float(summary["mean_delta"]) * n
    item["sum_mean_abs_delta"] += float(summary["mean_abs_delta"]) * n
    item["max_abs_delta"] = max(item["max_abs_delta"], float(summary["max_abs_delta"]))
    item["changed_count"] += int(summary["changed_count"])


def _finalize_group_mean_delta(store: dict[int, dict]) -> dict[int, float]:
    finalized = {}
    for group_value, item in sorted(store.items()):
        n = int(item["n"])
        if n <= 0:
            continue
        finalized[int(group_value)] = float(item["sum_mean_delta"] / n)
    return finalized


def snapshot_weight_mean_delta_by_layer(
    tracker: dict,
    before: dict[str, np.ndarray],
) -> tuple[dict[int, float], dict[int, float]]:
    liquid_store: dict[int, dict] = {}
    output_store: dict[int, dict] = {}

    for synapses in tracker["synapses"]:
        name = synapses.name
        summary = weight_change_summary(before[name], read_synapse_weights(synapses))
        liq_layer, out_layer = synapse_layer_labels(name)
        _accumulate_group_summary(liquid_store, liq_layer, summary)
        _accumulate_group_summary(output_store, out_layer, summary)

    return _finalize_group_mean_delta(liquid_store), _finalize_group_mean_delta(output_store)


def append_weight_change_records(
    tracker: dict,
    before: dict[str, np.ndarray],
    *,
    rep: int,
    mat_index: int,
    mat: str,
    sample_index: int,
    sid: int,
    summary_rows: list[dict],
    trace_rows: list[dict],
) -> None:
    # 1試行前後の重み差を、平均変動 summary と個別重み trace の両方に追加する。
    for synapses in tracker["synapses"]:
        name = synapses.name
        before_w = before[name]
        after_w = read_synapse_weights(synapses)
        liq_layer, out_layer = synapse_layer_labels(name)

        summary_rows.append(
            {
                "rep": rep,
                "mat_index": mat_index,
                "mat": mat,
                "sample_index": sample_index,
                "sid": sid,
                "synapse": name,
                "liquid_layer": liq_layer,
                "output_layer": out_layer,
                **weight_change_summary(before_w, after_w),
            }
        )


def aggregate_weight_change_summary(
    summary_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    if summary_df.empty or group_col not in summary_df.columns:
        return pd.DataFrame()

    work_df = summary_df[summary_df[group_col].notna()].copy()
    if work_df.empty:
        return pd.DataFrame()

    group_keys = ["rep", "mat_index", "mat", "sample_index", "sid", group_col]
    rows = []

    for key, group_df in work_df.groupby(group_keys, sort=True, dropna=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_keys, key))

        weights = pd.to_numeric(group_df["n"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        total_n = int(np.sum(weights))
        row["n"] = total_n

        for metric in ("mean_before", "mean_after", "mean_delta", "mean_abs_delta"):
            values = pd.to_numeric(group_df[metric], errors="coerce").to_numpy(dtype=float)
            if total_n > 0:
                row[metric] = float(np.average(values, weights=weights))
            else:
                row[metric] = np.nan

        max_abs = pd.to_numeric(group_df["max_abs_delta"], errors="coerce").dropna()
        row["max_abs_delta"] = float(max_abs.max()) if not max_abs.empty else np.nan
        row["changed_count"] = int(
            pd.to_numeric(group_df["changed_count"], errors="coerce").fillna(0).sum()
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_weight_change_records(
    out_dir: Path,
    rep: int,
    summary_rows: list[dict],
    trace_rows: list[dict],
    max_trace_lines: int = 12,
) -> None:
    # 1 rep 分の重み変動を CSV とグラフにまとめて保存する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    liquid_summary_df = aggregate_weight_change_summary(summary_df, "liquid_layer")
    output_summary_df = aggregate_weight_change_summary(summary_df, "output_layer")

    stale_files = [
        out_dir / f"weight_change_summary_rep{rep}.csv",
        out_dir / f"weight_trace_samples_rep{rep}.csv",
    ]
    stale_patterns = [
        f"weight_change_mean_before_after_rep{rep}.png",
        f"weight_change_mean_delta_rep{rep}.png",
        f"weight_change_mean_abs_delta_rep{rep}.png",
        f"weight_change_changed_count_rep{rep}.png",
        f"weight_change_mean_abs_delta_by_liquid_layer_rep{rep}.png",
        f"weight_change_mean_abs_delta_by_output_layer_rep{rep}.png",
        f"weight_trace_*_rep{rep}.png",
        f"weight_trace_delta_*_rep{rep}.png",
    ]
    for fp in stale_files:
        if fp.exists():
            fp.unlink()
    for pattern in stale_patterns:
        for fp in plot_dir.glob(pattern):
            fp.unlink()

    liquid_summary_df.to_csv(
        out_dir / f"weight_change_summary_by_liquid_layer_rep{rep}.csv",
        index=False,
    )
    output_summary_df.to_csv(
        out_dir / f"weight_change_summary_by_output_layer_rep{rep}.csv",
        index=False,
    )
    save_weight_change_plots(
        out_dir,
        rep,
        liquid_summary_df=liquid_summary_df,
        output_summary_df=output_summary_df,
        max_trace_lines=max_trace_lines,
    )


def _plot_summary_metric(
    out_fp: Path,
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    if summary_df.empty or metric not in summary_df:
        return

    plt.figure(figsize=(10, 5))
    for synapse, group_df in summary_df.groupby("synapse", sort=True):
        group_df = group_df.sort_values("sample_index")
        plt.plot(group_df["sample_index"], group_df[metric], marker="o", linewidth=1.2, label=synapse)
    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def _plot_group_summary_metric(
    out_fp: Path,
    summary_df: pd.DataFrame,
    *,
    group_col: str,
    group_prefix: str,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    if summary_df.empty or metric not in summary_df.columns or group_col not in summary_df.columns:
        return

    plt.figure(figsize=(10, 5))
    for group_value, group_df in summary_df.groupby(group_col, sort=True):
        group_df = group_df.sort_values(["mat_index", "sample_index"])
        plt.plot(
            group_df["sample_index"],
            group_df[metric],
            marker="o",
            linewidth=1.2,
            label=f"{group_prefix}{int(group_value)}",
        )
    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def _plot_trace_synapse(
    out_fp: Path,
    trace_df: pd.DataFrame,
    synapse: str,
    max_lines: int,
) -> None:
    syn_df = trace_df[trace_df["synapse"] == synapse]
    if syn_df.empty:
        return

    synapse_indices = sorted(syn_df["synapse_index"].unique())[:max_lines]
    plt.figure(figsize=(10, 5))
    for syn_idx in synapse_indices:
        item = syn_df[syn_df["synapse_index"] == syn_idx].sort_values("sample_index")
        plt.plot(
            item["sample_index"],
            item["w_after"],
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=f"idx {int(syn_idx)}",
        )

    plt.xlabel("Sample index")
    plt.ylabel("w after sample")
    plt.title(f"Weight traces | {synapse}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def _plot_mean_before_after(out_fp: Path, summary_df: pd.DataFrame, rep: int) -> None:
    if summary_df.empty:
        return

    labels = []
    before = []
    after = []
    for _, row in summary_df.sort_values(["sample_index", "synapse"]).iterrows():
        labels.append(f"{row['synapse']}\nsample {int(row['sample_index'])}")
        before.append(row["mean_before"])
        after.append(row["mean_after"])

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(9, len(labels) * 1.8), 5))
    plt.bar(x - width / 2, before, width, label="before")
    plt.bar(x + width / 2, after, width, label="after")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean weight")
    plt.title(f"Mean weight before/after each sample | rep{rep}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def _plot_trace_delta_synapse(
    out_fp: Path,
    trace_df: pd.DataFrame,
    synapse: str,
    max_lines: int,
) -> None:
    syn_df = trace_df[trace_df["synapse"] == synapse]
    if syn_df.empty:
        return

    synapse_indices = sorted(syn_df["synapse_index"].unique())[:max_lines]
    plot_df = syn_df[syn_df["synapse_index"].isin(synapse_indices)].copy()
    plot_df = plot_df.sort_values(["sample_index", "synapse_index"])
    labels = [f"s{int(s)}\ni{int(i)}" for s, i in zip(plot_df["sample_index"], plot_df["synapse_index"])]

    plt.figure(figsize=(max(9, len(plot_df) * 0.5), 5))
    plt.bar(np.arange(len(plot_df)), plot_df["delta"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(plot_df)), labels, rotation=90, fontsize=7)
    plt.ylabel("Delta w")
    plt.title(f"Selected weight deltas | {synapse}")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def save_weight_change_plots(
    out_dir: Path,
    rep: int,
    *,
    liquid_summary_df: pd.DataFrame | None = None,
    output_summary_df: pd.DataFrame | None = None,
    max_trace_lines: int = 12,
) -> None:
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    liquid_summary_df = liquid_summary_df if liquid_summary_df is not None else pd.DataFrame()
    output_summary_df = output_summary_df if output_summary_df is not None else pd.DataFrame()

    _plot_group_summary_metric(
        plot_dir / f"weight_change_mean_delta_by_liquid_layer_rep{rep}.png",
        liquid_summary_df,
        group_col="liquid_layer",
        group_prefix="L",
        metric="mean_delta",
        ylabel="Mean delta",
        title=f"Mean weight delta by liquid layer | rep{rep}",
    )
    _plot_group_summary_metric(
        plot_dir / f"weight_change_mean_delta_by_output_layer_rep{rep}.png",
        output_summary_df,
        group_col="output_layer",
        group_prefix="O",
        metric="mean_delta",
        ylabel="Mean delta",
        title=f"Mean weight delta by output layer | rep{rep}",
    )
