"""Decompose why SRDP produces a strongly diagonal temporal-generalization matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
RULES = ("STDP", "T_STDP", "SRDP")
REPS = tuple(range(1, 11))
TN_MS = 25
N_WINDOWS = 20
EPS = 1e-12


def _load_binned(rule: str, rep: int) -> np.ndarray:
    path = PROJECT_ROOT / rule / f"{rule}_1_sout_rec_rep{rep}.npy"
    spikes = np.load(path, mmap_mode="r")
    return spikes.reshape(8, 100, 40, N_WINDOWS, TN_MS).sum(axis=-1, dtype=np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPS else np.nan


def _treves_rolls(profiles: np.ndarray) -> np.ndarray:
    n = profiles.shape[-1]
    mean = profiles.mean(axis=-1)
    mean_square = np.square(profiles).mean(axis=-1)
    ratio = np.divide(np.square(mean), mean_square, out=np.ones_like(mean), where=mean_square > EPS)
    return np.clip((1.0 - ratio) / (1.0 - 1.0 / n), 0.0, 1.0)


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3 or np.std(x[mask]) <= EPS or np.std(y[mask]) <= EPS:
        return np.nan
    return float(pearsonr(x[mask], y[mask]).statistic)


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    lag_rows: list[dict[str, float | int | str]] = []
    time_rows: list[dict[str, float | int | str]] = []
    peak_rows: list[dict[str, float | int | str]] = []

    for rep in REPS:
        for rule in RULES:
            binned = _load_binned(rule, rep)  # class, trial, neuron, time
            centroids = binned.mean(axis=1)  # class, neuron, time

            lag_means: dict[int, float] = {}
            for lag in range(N_WINDOWS):
                similarities = []
                for class_index in range(centroids.shape[0]):
                    for start in range(N_WINDOWS - lag):
                        value = _cosine(
                            centroids[class_index, :, start],
                            centroids[class_index, :, start + lag],
                        )
                        if np.isfinite(value):
                            similarities.append(value)
                lag_mean = float(np.mean(similarities)) if similarities else np.nan
                lag_means[lag] = lag_mean
                lag_rows.append(
                    {"rule": rule, "rep": rep, "lag_windows": lag, "lag_ms": lag * TN_MS,
                     "same_class_cosine_similarity": lag_mean}
                )

            trajectory_steps = []
            for class_index in range(centroids.shape[0]):
                for window in range(N_WINDOWS - 1):
                    similarity = _cosine(centroids[class_index, :, window], centroids[class_index, :, window + 1])
                    if np.isfinite(similarity):
                        trajectory_steps.append(1.0 - similarity)

            temporal_selectivity_by_class_neuron = _treves_rolls(centroids)
            peak_counts = np.max(centroids, axis=-1)
            active = peak_counts > EPS
            half_widths = np.sum(centroids >= 0.5 * peak_counts[..., None], axis=-1) * TN_MS
            peak_windows = np.argmax(centroids, axis=-1)

            for window in range(N_WINDOWS):
                peak_rows.append(
                    {
                        "rule": rule,
                        "rep": rep,
                        "window": window,
                        "time_ms": (window + 0.5) * TN_MS,
                        "peak_fraction": float(np.mean(peak_windows[active] == window)),
                    }
                )

            separation_by_time = []
            for window in range(N_WINDOWS):
                values = binned[..., window]
                class_centroids = values.mean(axis=1)
                grand_centroid = class_centroids.mean(axis=0)
                between = float(np.mean(np.sum(np.square(class_centroids - grand_centroid), axis=1)))
                within = float(np.mean(np.sum(np.square(values - class_centroids[:, None, :]), axis=2)))
                separation = between / within if within > EPS else np.nan
                separation_by_time.append(separation)
                time_rows.append(
                    {
                        "rule": rule,
                        "rep": rep,
                        "window": window,
                        "time_ms": (window + 0.5) * TN_MS,
                        "between_within_separation": separation,
                    }
                )

            weights = np.load(PROJECT_ROOT / rule / f"{rule}_1_w_out_rep{rep}.npy")
            incoming_weight_norm = np.linalg.norm(weights, axis=0)
            neuron_rate_hz = binned.sum(axis=(0, 1, 3)) / (8 * 100 * 0.5)
            neuron_temporal_selectivity = np.nanmean(temporal_selectivity_by_class_neuron, axis=0)

            summary_rows.append(
                {
                    "rule": rule,
                    "rep": rep,
                    "similarity_25ms": lag_means[1],
                    "similarity_100ms": lag_means[4],
                    "similarity_250ms": lag_means[10],
                    "trajectory_speed": float(np.mean(trajectory_steps)),
                    "temporal_selectivity": float(np.mean(temporal_selectivity_by_class_neuron[active])),
                    "halfmax_width_ms": float(np.mean(half_widths[active])),
                    "peak_time_dispersion_ms": float(np.std((peak_windows[active] + 0.5) * TN_MS, ddof=1)),
                    "mean_time_separation": float(np.nanmean(separation_by_time)),
                    "weight_norm_rate_r": _pearson_safe(incoming_weight_norm, neuron_rate_hz),
                    "weight_norm_temporal_selectivity_r": _pearson_safe(
                        incoming_weight_norm, neuron_temporal_selectivity
                    ),
                }
            )
        print(f"[temporal mechanism] completed rep {rep}/10", flush=True)

    return pd.DataFrame(summary_rows), pd.DataFrame(lag_rows), pd.DataFrame(time_rows), pd.DataFrame(peak_rows)


def _ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _paired_box(ax: plt.Axes, table: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    pivot = table.pivot(index="rep", columns="rule", values=metric).reindex(columns=RULES)
    positions = np.arange(1, 4)
    for _, row in pivot.iterrows():
        ax.plot(positions, row.to_numpy(float), color="0.82", linewidth=0.8)
    ax.boxplot(
        [pivot[rule].to_numpy(float) for rule in RULES], positions=positions, widths=0.48,
        showmeans=True, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.3),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=1.5),
        meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=5),
    )
    for position, rule in zip(positions, RULES):
        ax.text(position, pivot[rule].max(), f"{pivot[rule].mean():.3g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(positions, RULES)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="0.87", linestyle="--", linewidth=0.7)


def create_mechanism_figure(summary: pd.DataFrame, lag: pd.DataFrame, time: pd.DataFrame, peak: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.5), dpi=180)
    styles = {"STDP": ("0.65", "--"), "T_STDP": ("0.35", "-."), "SRDP": ("black", "-")}

    for rule in RULES:
        table = lag[lag["rule"] == rule].pivot(index="rep", columns="lag_ms", values="same_class_cosine_similarity")
        mean = table.mean(axis=0).to_numpy(float)
        ci = np.array([_ci95(table[column].dropna().to_numpy(float)) for column in table.columns])
        color, linestyle = styles[rule]
        axes[0, 0].plot(table.columns, mean, color=color, linestyle=linestyle, label=rule)
        axes[0, 0].fill_between(table.columns, mean - ci, mean + ci, color=color, alpha=0.12)
    axes[0, 0].set_xlabel("time lag (ms)")
    axes[0, 0].set_ylabel("same-class population cosine similarity")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_title("(a) Decay of the population pattern", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True, color="0.87", linestyle="--", linewidth=0.7)

    _paired_box(axes[0, 1], summary, "trajectory_speed", "1 - consecutive-window cosine", "(b) Population-trajectory speed")
    _paired_box(axes[0, 2], summary, "temporal_selectivity", "Treves-Rolls selectivity across time", "(c) Class-neuron temporal selectivity")
    _paired_box(axes[1, 0], summary, "halfmax_width_ms", "half-maximum width (ms)", "(d) Temporal tuning width")

    for rule in RULES:
        table = time[time["rule"] == rule].pivot(index="rep", columns="time_ms", values="between_within_separation")
        mean = table.mean(axis=0).to_numpy(float)
        ci = np.array([_ci95(table[column].to_numpy(float)) for column in table.columns])
        color, linestyle = styles[rule]
        axes[1, 1].plot(table.columns, mean, color=color, linestyle=linestyle, label=rule)
        axes[1, 1].fill_between(table.columns, mean - ci, mean + ci, color=color, alpha=0.12)
    axes[1, 1].set_xlabel("time (ms)")
    axes[1, 1].set_ylabel("between / within-class dispersion")
    axes[1, 1].set_title("(e) Time-resolved class separation", loc="left", fontweight="bold")
    axes[1, 1].grid(True, color="0.87", linestyle="--", linewidth=0.7)

    width = 7.0
    offsets = {"STDP": -width, "T_STDP": 0.0, "SRDP": width}
    colors = {"STDP": "white", "T_STDP": "0.55", "SRDP": "black"}
    for rule in RULES:
        grouped = peak[peak["rule"] == rule].groupby("time_ms")["peak_fraction"].mean()
        axes[1, 2].bar(grouped.index + offsets[rule], grouped.values, width=width,
                       color=colors[rule], edgecolor="black", linewidth=0.6, label=rule)
    axes[1, 2].set_xlabel("peak-response time (ms)")
    axes[1, 2].set_ylabel("class-neuron fraction")
    axes[1, 2].set_title("(f) Distribution of temporal response peaks", loc="left", fontweight="bold")
    axes[1, 2].legend(frameon=False)
    axes[1, 2].grid(True, axis="y", color="0.87", linestyle="--", linewidth=0.7)

    fig.suptitle("Mechanism underlying SRDP temporal specificity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_temporal_mechanism.{suffix}", bbox_inches="tight")
    plt.close(fig)


def create_weight_link_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=180)
    _paired_box(axes[0], summary, "weight_norm_rate_r", "within-rep Pearson r", "(a) Incoming weight norm vs firing rate")
    _paired_box(axes[1], summary, "weight_norm_temporal_selectivity_r", "within-rep Pearson r",
                "(b) Incoming weight norm vs temporal selectivity")
    for ax in axes:
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    fig.suptitle("Relationship between learned output weights and temporal responses", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_weight_temporal_link.{suffix}", bbox_inches="tight")
    plt.close(fig)


def build_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "similarity_25ms", "similarity_100ms", "similarity_250ms", "trajectory_speed",
        "temporal_selectivity", "halfmax_width_ms", "peak_time_dispersion_ms",
        "mean_time_separation", "weight_norm_rate_r", "weight_norm_temporal_selectivity_r",
    ]
    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        pivot = summary.pivot(index="rep", columns="rule", values=metric)
        for baseline in ("STDP", "T_STDP"):
            difference = (pivot["SRDP"] - pivot[baseline]).dropna().to_numpy(float)
            mean = float(np.mean(difference))
            half_width = _ci95(difference)
            p_value = float(wilcoxon(difference).pvalue) if np.any(np.abs(difference) > EPS) else 1.0
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"SRDP - {baseline}",
                    "mean_difference": mean,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "wilcoxon_p": p_value,
                    "srdp_higher_reps": int(np.sum(difference > 0)),
                }
            )
    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = np.nan
    for comparison in result["comparison"].unique():
        mask = result["comparison"] == comparison
        p_values = result.loc[mask, "wilcoxon_p"].to_numpy(float)
        order = np.argsort(p_values)
        adjusted = np.empty_like(p_values)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
            adjusted[index] = running
        result.loc[mask, "wilcoxon_p_holm"] = adjusted
    return result


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    summary, lag, time, peak = analyze()
    statistics = build_statistics(summary)
    create_mechanism_figure(summary, lag, time, peak)
    create_weight_link_figure(summary)
    summary.to_csv(OUTPUT_DIR / "srdp_temporal_mechanism_by_rep.csv", index=False)
    lag.to_csv(OUTPUT_DIR / "srdp_temporal_similarity_by_lag.csv", index=False)
    time.to_csv(OUTPUT_DIR / "srdp_time_resolved_separation.csv", index=False)
    peak.to_csv(OUTPUT_DIR / "srdp_temporal_peak_distribution.csv", index=False)
    statistics.to_csv(OUTPUT_DIR / "srdp_temporal_mechanism_statistics.csv", index=False)
    print("\nTemporal-mechanism summary (means across reps)")
    print(summary.groupby("rule").mean(numeric_only=True).reindex(RULES).to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nSaved temporal-mechanism figures and CSV files.")


if __name__ == "__main__":
    main()
