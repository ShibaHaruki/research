"""Firing-rate and plastic output-weight analysis for STDP/T_STDP/SRDP.

All comparisons are paired by repetition.  The output spike arrays have shape
(8 classes, 100 trials, 40 output neurons, 500 one-millisecond bins).  The
non-learning ``off`` output weights provide the matched pre-learning baseline
because all rules use the same per-repetition seed and network construction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results_10fold"

RULES = ("STDP", "T_STDP", "SRDP")
REPS = tuple(range(1, 11))
TARGET_TN = 25
BIN_WIDTH_S = 0.001
EPS = 1e-10

RULE_DIRS = {rule: PROJECT_ROOT / rule for rule in RULES}
PREFIXES = {rule: f"{rule}_1" for rule in RULES}
INITIAL_WEIGHT_DIR = PROJECT_ROOT / "off"
INITIAL_PREFIX = "off_1"


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def _lifetime_sparseness(class_neuron_rates: np.ndarray) -> np.ndarray:
    """Treves-Rolls class selectivity for every output neuron (0=uniform, 1=selective)."""
    rates = np.asarray(class_neuron_rates, dtype=float)
    n_classes = rates.shape[0]
    mean_rate = rates.mean(axis=0)
    mean_square = np.square(rates).mean(axis=0)
    ratio = np.divide(np.square(mean_rate), mean_square, out=np.ones_like(mean_rate), where=mean_square > 0)
    return np.clip((1.0 - ratio) / (1.0 - 1.0 / n_classes), 0.0, 1.0)


def _feature_separability(spikes: np.ndarray) -> float:
    """Between-class centroid energy divided by within-class dispersion."""
    n_classes, n_trials, n_neurons, n_bins = spikes.shape
    if n_bins % TARGET_TN != 0:
        raise ValueError(f"{n_bins} bins are not divisible by Tn={TARGET_TN}")
    n_intervals = n_bins // TARGET_TN
    features = (
        spikes.reshape(n_classes, n_trials, n_neurons, n_intervals, TARGET_TN)
        .sum(axis=-1, dtype=np.float64)
        .reshape(n_classes, n_trials, -1)
    )
    centroids = features.mean(axis=1)
    grand_centroid = centroids.mean(axis=0)
    between = float(np.mean(np.sum(np.square(centroids - grand_centroid), axis=1)))
    within = float(np.mean(np.sum(np.square(features - centroids[:, None, :]), axis=2)))
    return between / within if within > 0 else np.nan


def _accuracy_for_rep(rule: str, rep: int) -> tuple[float, float]:
    path = RESULTS_ROOT / f"{rule}_1" / f"{rule}_1_rep{rep:02d}_Tn_{TARGET_TN}_10fold_conf_matrices.xlsx"
    row = pd.read_excel(path, sheet_name="accuracy").iloc[0]
    return float(row["accuracy8_overall"]), float(row["accuracy3_overall"])


def _validate_matched_fixed_weights() -> None:
    """Confirm that input/recurrent networks are identical across rules for every rep."""
    for rep in REPS:
        reference_in = np.load(RULE_DIRS["STDP"] / f"STDP_1_w_in_rep{rep}.npy", mmap_mode="r")
        reference_res = np.load(RULE_DIRS["STDP"] / f"STDP_1_w_res_rep{rep}.npy", mmap_mode="r")
        for rule in RULES[1:]:
            other_in = np.load(RULE_DIRS[rule] / f"{PREFIXES[rule]}_w_in_rep{rep}.npy", mmap_mode="r")
            other_res = np.load(RULE_DIRS[rule] / f"{PREFIXES[rule]}_w_res_rep{rep}.npy", mmap_mode="r")
            if not np.array_equal(reference_in, other_in):
                raise ValueError(f"Input weights are not matched for rep={rep}: STDP vs {rule}")
            if not np.array_equal(reference_res, other_res):
                raise ValueError(f"Recurrent weights are not matched for rep={rep}: STDP vs {rule}")


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    _validate_matched_fixed_weights()
    firing_rows: list[dict[str, float | int | str]] = []
    weight_rows: list[dict[str, float | int | str]] = []
    pooled_final_weights: dict[str, list[np.ndarray]] = {rule: [] for rule in RULES}

    for rep in REPS:
        initial = np.load(INITIAL_WEIGHT_DIR / f"{INITIAL_PREFIX}_w_out_rep{rep}.npy")
        structural_mask = np.abs(initial) > EPS
        if structural_mask.sum() == 0:
            raise ValueError(f"No connected output weights in initial rep={rep}")

        for rule in RULES:
            spike_path = RULE_DIRS[rule] / f"{PREFIXES[rule]}_sout_rec_rep{rep}.npy"
            spikes = np.load(spike_path, mmap_mode="r")
            if spikes.ndim != 4 or spikes.shape[0] != 8 or spikes.shape[2] != 40:
                raise ValueError(f"Unexpected spike shape {spikes.shape} in {spike_path}")

            total_duration_s = spikes.shape[-1] * BIN_WIDTH_S
            mean_rate_hz = float(spikes.sum(dtype=np.float64) / (np.prod(spikes.shape[:-1]) * total_duration_s))
            temporal_sparsity = 1.0 - float(np.count_nonzero(spikes)) / float(spikes.size)
            trial_neuron_counts = spikes.sum(axis=-1, dtype=np.float64)
            active_neuron_fraction = float(np.mean(trial_neuron_counts > 0))
            trial_activity = trial_neuron_counts.reshape(-1, trial_neuron_counts.shape[-1])
            centered_activity = trial_activity - trial_activity.mean(axis=0, keepdims=True)
            covariance = centered_activity.T @ centered_activity / max(len(trial_activity) - 1, 1)
            covariance_energy = float(np.sum(np.square(covariance)))
            effective_dimension = (
                float(np.trace(covariance) ** 2 / covariance_energy) if covariance_energy > 0 else np.nan
            )
            variable_neurons = np.std(trial_activity, axis=0) > EPS
            neuron_correlation = np.corrcoef(trial_activity[:, variable_neurons], rowvar=False)
            off_diagonal = neuron_correlation[~np.eye(neuron_correlation.shape[0], dtype=bool)]
            mean_abs_neuron_correlation = float(np.nanmean(np.abs(off_diagonal)))
            class_neuron_rates = trial_neuron_counts.mean(axis=1) / total_duration_s
            selectivity = _lifetime_sparseness(class_neuron_rates)
            separability = _feature_separability(spikes)
            accuracy8, accuracy3 = _accuracy_for_rep(rule, rep)
            firing_rows.append(
                {
                    "rule": rule,
                    "rep": rep,
                    "mean_rate_hz": mean_rate_hz,
                    "temporal_sparsity": temporal_sparsity,
                    "active_neuron_fraction": active_neuron_fraction,
                    "effective_dimension": effective_dimension,
                    "mean_abs_neuron_correlation": mean_abs_neuron_correlation,
                    "class_selectivity": float(np.mean(selectivity)),
                    "feature_separability": separability,
                    "accuracy8": accuracy8,
                    "accuracy3": accuracy3,
                    "source_file": str(spike_path),
                }
            )

            final = np.load(RULE_DIRS[rule] / f"{PREFIXES[rule]}_w_out_rep{rep}.npy")
            if final.shape != initial.shape:
                raise ValueError(f"Weight shape mismatch for {rule}, rep={rep}: {final.shape} vs {initial.shape}")
            initial_connected = initial[structural_mask]
            final_connected = final[structural_mask]
            delta = final_connected - initial_connected
            pooled_final_weights[rule].append(final_connected.astype(np.float32, copy=False))
            weight_rows.append(
                {
                    "rule": rule,
                    "rep": rep,
                    "n_connections": int(structural_mask.sum()),
                    "final_mean": float(np.mean(final_connected)),
                    "final_std": float(np.std(final_connected, ddof=1)),
                    "final_abs_mean": float(np.mean(np.abs(final_connected))),
                    "delta_mean": float(np.mean(delta)),
                    "delta_abs_mean": float(np.mean(np.abs(delta))),
                    "potentiated_fraction": float(np.mean(delta > EPS)),
                    "depressed_fraction": float(np.mean(delta < -EPS)),
                    "unchanged_fraction": float(np.mean(np.abs(delta) <= EPS)),
                    "near_zero_fraction": float(np.mean(np.abs(final_connected) <= 0.01)),
                    "upper_saturation_fraction": float(np.mean(final_connected >= 0.99)),
                    "lower_saturation_fraction": float(np.mean(final_connected <= -0.99)),
                    "positive_fraction": float(np.mean(final_connected > 0)),
                }
            )

    pooled = {rule: np.concatenate(values) for rule, values in pooled_final_weights.items()}
    return pd.DataFrame(firing_rows), pd.DataFrame(weight_rows), pooled


def paired_statistics(table: pd.DataFrame, metrics: list[str], domain: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        pivot = table.pivot(index="rep", columns="rule", values=metric)
        for baseline in ("STDP", "T_STDP"):
            difference = (pivot["SRDP"] - pivot[baseline]).dropna().to_numpy(float)
            mean, low, high = _mean_ci95(difference)
            p_value = float(wilcoxon(difference).pvalue) if np.any(np.abs(difference) > EPS) else 1.0
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "comparison": f"SRDP - {baseline}",
                    "n_pairs": len(difference),
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wilcoxon_p": p_value,
                    "srdp_higher": int(np.sum(difference > 0)),
                    "ties": int(np.sum(np.abs(difference) <= EPS)),
                }
            )
    return pd.DataFrame(rows)


def _paired_box(ax: plt.Axes, table: pd.DataFrame, metric: str, ylabel: str, title: str, percent: bool = False) -> None:
    pivot = table.pivot(index="rep", columns="rule", values=metric).reindex(columns=RULES)
    if percent:
        pivot = 100.0 * pivot
    positions = np.arange(1, len(RULES) + 1)
    for _, row in pivot.iterrows():
        ax.plot(positions, row.to_numpy(float), color="0.82", linewidth=0.8, zorder=1)
        ax.scatter(positions, row.to_numpy(float), s=12, facecolor="white", edgecolor="0.45", linewidth=0.6, zorder=2)
    ax.boxplot(
        [pivot[rule].to_numpy(float) for rule in RULES],
        positions=positions,
        widths=0.48,
        showmeans=True,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.4),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        medianprops=dict(color="black", linewidth=1.6),
        meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=5),
    )
    means = pivot.mean(axis=0)
    spread = max(float(pivot.to_numpy().max() - pivot.to_numpy().min()), 1e-6)
    for x, rule in zip(positions, RULES):
        ax.text(x, float(pivot[rule].max()) + 0.05 * spread, f"{means[rule]:.3g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(positions, RULES)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="0.86", linestyle="--", linewidth=0.7)


def create_firing_figure(firing: pd.DataFrame) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), dpi=180)
    _paired_box(axes[0, 0], firing, "mean_rate_hz", "mean output firing rate (Hz)", "(a) Output firing rate")
    _paired_box(axes[0, 1], firing, "temporal_sparsity", "zero spike bins (%)", "(b) Temporal sparsity", percent=True)
    _paired_box(axes[0, 2], firing, "active_neuron_fraction", "active neurons per trial (%)", "(c) Distributed recruitment", percent=True)
    _paired_box(axes[1, 0], firing, "effective_dimension", "participation-ratio dimension", "(d) Population effective dimension")
    _paired_box(axes[1, 1], firing, "mean_abs_neuron_correlation", "mean |neuron correlation|", "(e) Population redundancy")
    _paired_box(axes[1, 2], firing, "class_selectivity", "Treves-Rolls selectivity", "(f) Rate-based class selectivity")
    fig.suptitle("Firing-pattern analysis across matched repetitions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_firing_analysis.{suffix}", bbox_inches="tight")
    plt.close(fig)


def create_weight_figure(weights: pd.DataFrame, pooled: dict[str, np.ndarray]) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=180)

    bins = np.linspace(-0.25, 1.0, 130)
    styles = {"STDP": ("0.65", "--"), "T_STDP": ("0.35", "-."), "SRDP": ("black", "-")}
    for rule in RULES:
        color, linestyle = styles[rule]
        axes[0, 0].hist(pooled[rule], bins=bins, density=True, histtype="step", linewidth=1.5,
                        color=color, linestyle=linestyle, label=rule)
    axes[0, 0].set_xlabel("final connected output weight")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("(a) Final plastic-weight distribution", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True, axis="y", color="0.88", linestyle="--", linewidth=0.7)

    _paired_box(axes[0, 1], weights, "delta_abs_mean", "mean |final - initial weight|", "(b) Magnitude of synaptic modification")

    _paired_box(axes[1, 0], weights, "delta_mean", "mean (final - initial weight)", "(c) Net potentiation / depression")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    _paired_box(axes[1, 1], weights, "final_abs_mean", "mean |final connected weight|", "(d) Final effective weight magnitude")

    fig.suptitle("Plastic output-weight analysis (initial weights matched by repetition)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_weight_analysis.{suffix}", bbox_inches="tight")
    plt.close(fig)


def build_correlations(firing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for predictor in ("mean_rate_hz", "temporal_sparsity", "active_neuron_fraction", "effective_dimension",
                      "mean_abs_neuron_correlation", "class_selectivity", "feature_separability"):
        x = firing[predictor].to_numpy(float)
        y = firing["accuracy8"].to_numpy(float)
        correlation, p_value = pearsonr(x, y)
        rows.append(
            {
                "predictor": predictor,
                "outcome": "accuracy8",
                "n": len(x),
                "pearson_r": float(correlation),
                "pearson_p": float(p_value),
                "note": "Pooled across rules and repetitions; association, not causal evidence.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    firing, weights, pooled = analyze()
    firing_metrics = ["mean_rate_hz", "temporal_sparsity", "active_neuron_fraction", "effective_dimension",
                      "mean_abs_neuron_correlation", "class_selectivity", "feature_separability"]
    weight_metrics = ["final_abs_mean", "delta_mean", "delta_abs_mean", "potentiated_fraction", "depressed_fraction",
                      "unchanged_fraction", "near_zero_fraction", "upper_saturation_fraction", "lower_saturation_fraction"]
    statistics = pd.concat(
        [paired_statistics(firing, firing_metrics, "firing"), paired_statistics(weights, weight_metrics, "weights")],
        ignore_index=True,
    )
    correlations = build_correlations(firing)

    create_firing_figure(firing)
    create_weight_figure(weights, pooled)

    firing.to_csv(OUTPUT_DIR / "srdp_firing_metrics_by_rep.csv", index=False)
    weights.to_csv(OUTPUT_DIR / "srdp_weight_metrics_by_rep.csv", index=False)
    statistics.to_csv(OUTPUT_DIR / "srdp_firing_weight_paired_statistics.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "srdp_firing_accuracy_correlations.csv", index=False)

    print("\nFiring metrics (mean across 10 reps)")
    print(firing.groupby("rule")[firing_metrics + ["accuracy8"]].mean().reindex(RULES).to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nWeight metrics (mean across 10 reps)")
    print(weights.groupby("rule")[weight_metrics].mean().reindex(RULES).to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nPooled correlations with 8-class accuracy")
    print(correlations.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {OUTPUT_DIR / 'srdp_firing_analysis.png'}")
    print(f"Saved: {OUTPUT_DIR / 'srdp_weight_analysis.png'}")


if __name__ == "__main__":
    main()
