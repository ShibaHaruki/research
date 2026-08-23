"""Ablation tests for temporal patterns and population coding.

This analysis uses a fixed, leakage-free auxiliary decoder (nearest class
centroid with pooled diagonal standardization) for every condition.  It tests
whether decoding depends on temporal order, cross-neuron temporal alignment,
neuron identity, population size, or firing-rate differences.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
RULES = ("STDP", "T_STDP", "SRDP")
PREFIXES = {rule: f"{rule}_1" for rule in RULES}
RULE_DIRS = {rule: PROJECT_ROOT / rule for rule in RULES}
REPS = tuple(range(1, 11))

TN = 25
N_FOLDS = 10
BASE_SEED = 1
N_RANDOMIZATIONS = 3
CLASS_TO_GROUP = np.array([0, 2, 2, 1, 1, 0, 1, 0], dtype=int)
NEURON_COUNTS = (1, 2, 5, 10, 20, 40)
SUBSET_DRAWS = {1: 10, 2: 10, 5: 10, 10: 8, 20: 5, 40: 1}

CONDITIONS = (
    "full",
    "time_collapsed",
    "time_shuffled",
    "independent_shift",
    "neuron_shuffled",
    "rate_matched",
)
CONDITION_LABELS = {
    "full": "Full",
    "time_collapsed": "Time\ncollapsed",
    "time_shuffled": "Time\nshuffled",
    "independent_shift": "Independent\nshift",
    "neuron_shuffled": "Neuron\nshuffled",
    "rate_matched": "Rate\nmatched",
}


def load_spikes(rule: str, rep: int) -> np.ndarray:
    path = RULE_DIRS[rule] / f"{PREFIXES[rule]}_sout_rec_rep{rep}.npy"
    spikes = np.load(path, mmap_mode="r")
    if spikes.shape != (8, 100, 40, 500):
        raise ValueError(f"Unexpected shape {spikes.shape}: {path}")
    return spikes


def fold_indices(rep: int) -> list[np.ndarray]:
    indices = np.arange(100)
    rng = np.random.default_rng(BASE_SEED + rep)
    rng.shuffle(indices)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(indices, N_FOLDS)]


def binned_features(spikes: np.ndarray, tn: int = TN) -> np.ndarray:
    n_classes, n_trials, n_neurons, n_bins = spikes.shape
    if n_bins % tn:
        raise ValueError(f"{n_bins} is not divisible by Tn={tn}")
    n_windows = n_bins // tn
    return (
        spikes.reshape(n_classes, n_trials, n_neurons, n_windows, tn)
        .sum(axis=-1, dtype=np.float32)
    )


def _fit_probe(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit pooled diagonal scaling and standardized class centroids."""
    flat = train.reshape(-1, train.shape[-1]).astype(np.float64, copy=False)
    center = flat.mean(axis=0)
    scale = flat.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    standardized = (train - center) / scale
    centroids = standardized.mean(axis=1)
    return center, scale, centroids


def _predict_probe(test: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, scale, centroids = model
    n_classes, n_test, n_features = test.shape
    if centroids.shape != (n_classes, n_features):
        raise ValueError(f"Probe/test shape mismatch: centroids={centroids.shape}, test={test.shape}")
    standardized = (test.reshape(-1, n_features) - center) / scale
    distances = np.sum(np.square(standardized[:, None, :] - centroids[None, :, :]), axis=2)
    return np.argmin(distances, axis=1).reshape(n_classes, n_test)


def _scores_from_predictions(prediction: np.ndarray) -> tuple[float, float]:
    true_class = np.repeat(np.arange(prediction.shape[0])[:, None], prediction.shape[1], axis=1)
    accuracy8 = float(np.mean(prediction == true_class))
    accuracy3 = float(np.mean(CLASS_TO_GROUP[prediction] == CLASS_TO_GROUP[true_class]))
    return accuracy8, accuracy3


def cross_validated_probe(features: np.ndarray, folds: list[np.ndarray]) -> tuple[float, float]:
    """Evaluate features shaped (class, trial, ...features...)."""
    flattened = features.reshape(features.shape[0], features.shape[1], -1).astype(np.float64, copy=False)
    scores8: list[float] = []
    scores3: list[float] = []
    all_indices = np.arange(flattened.shape[1])
    for test_indices in folds:
        train_indices = np.setdiff1d(all_indices, test_indices)
        model = _fit_probe(flattened[:, train_indices, :])
        prediction = _predict_probe(flattened[:, test_indices, :], model)
        accuracy8, accuracy3 = _scores_from_predictions(prediction)
        scores8.append(accuracy8)
        scores3.append(accuracy3)
    return float(np.mean(scores8)), float(np.mean(scores3))


def time_shuffle(spikes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle time order independently per trial, preserving population vectors."""
    shuffled = np.empty_like(spikes)
    for class_index in range(spikes.shape[0]):
        for trial_index in range(spikes.shape[1]):
            permutation = rng.permutation(spikes.shape[-1])
            shuffled[class_index, trial_index] = np.take(
                spikes[class_index, trial_index], permutation, axis=-1
            )
    return shuffled


def independent_circular_shift(spikes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift every neuron independently, preserving its spike train autocorrelation."""
    shifted = np.empty_like(spikes)
    base = np.arange(spikes.shape[-1])[None, None, :]
    for class_index in range(spikes.shape[0]):
        shifts = rng.integers(0, spikes.shape[-1], size=(spikes.shape[1], spikes.shape[2], 1))
        indices = (base - shifts) % spikes.shape[-1]
        shifted[class_index] = np.take_along_axis(spikes[class_index], indices, axis=-1)
    return shifted


def neuron_shuffle(spikes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Relabel neurons independently per trial while preserving population activity."""
    shuffled = np.empty_like(spikes)
    for class_index in range(spikes.shape[0]):
        for trial_index in range(spikes.shape[1]):
            permutation = rng.permutation(spikes.shape[2])
            shuffled[class_index, trial_index] = spikes[class_index, trial_index, permutation, :]
    return shuffled


def rate_match(spikes: np.ndarray, target_total: float, rng: np.random.Generator) -> np.ndarray:
    current_total = float(np.sum(spikes, dtype=np.float64))
    probability = min(1.0, target_total / current_total) if current_total > 0 else 1.0
    if probability >= 1.0:
        return np.asarray(spikes)
    return rng.binomial(np.asarray(spikes, dtype=np.int16), probability).astype(spikes.dtype)


def transformed_features(
    condition: str,
    spikes: np.ndarray,
    rng: np.random.Generator,
    target_total: float,
) -> np.ndarray:
    if condition == "full":
        return binned_features(spikes)
    if condition == "time_collapsed":
        return np.asarray(spikes).sum(axis=-1, dtype=np.float32)[..., None]
    if condition == "time_shuffled":
        return binned_features(time_shuffle(spikes, rng))
    if condition == "independent_shift":
        return binned_features(independent_circular_shift(spikes, rng))
    if condition == "neuron_shuffled":
        return binned_features(neuron_shuffle(spikes, rng))
    if condition == "rate_matched":
        return binned_features(rate_match(spikes, target_total, rng))
    raise ValueError(f"Unknown condition: {condition}")


def run_ablation() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for rep in REPS:
        folds = fold_indices(rep)
        target_total = float(np.sum(load_spikes("SRDP", rep), dtype=np.float64))
        for rule in RULES:
            spikes = load_spikes(rule, rep)
            for condition in CONDITIONS:
                n_runs = N_RANDOMIZATIONS if condition in {
                    "time_shuffled", "independent_shift", "neuron_shuffled", "rate_matched"
                } else 1
                for randomization in range(n_runs):
                    seed = 10_000_000 + rep * 100_000 + RULES.index(rule) * 10_000 + CONDITIONS.index(condition) * 100 + randomization
                    rng = np.random.default_rng(seed)
                    features = transformed_features(condition, spikes, rng, target_total)
                    accuracy8, accuracy3 = cross_validated_probe(features, folds)
                    rows.append(
                        {
                            "rule": rule,
                            "rep": rep,
                            "condition": condition,
                            "randomization": randomization,
                            "accuracy8": accuracy8,
                            "accuracy3": accuracy3,
                            "n_features": int(np.prod(features.shape[2:])),
                        }
                    )
        print(f"[ablation] completed rep {rep}/10", flush=True)
    return pd.DataFrame(rows)


def run_neuron_count_curve() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for rep in REPS:
        folds = fold_indices(rep)
        for rule in RULES:
            full = binned_features(load_spikes(rule, rep))
            for count in NEURON_COUNTS:
                for draw in range(SUBSET_DRAWS[count]):
                    rng = np.random.default_rng(20_000_000 + rep * 100_000 + count * 100 + draw)
                    neurons = np.sort(rng.choice(full.shape[2], size=count, replace=False))
                    accuracy8, accuracy3 = cross_validated_probe(full[:, :, neurons, :], folds)
                    rows.append(
                        {
                            "rule": rule,
                            "rep": rep,
                            "n_neurons": count,
                            "draw": draw,
                            "accuracy8": accuracy8,
                            "accuracy3": accuracy3,
                        }
                    )
        print(f"[population curve] completed rep {rep}/10", flush=True)
    return pd.DataFrame(rows)


def run_temporal_generalization() -> tuple[pd.DataFrame, np.ndarray]:
    curve_rows: list[dict[str, float | int | str]] = []
    matrices = np.zeros((len(RULES), len(REPS), 20, 20), dtype=float)
    for rep_index, rep in enumerate(REPS):
        folds = fold_indices(rep)
        all_indices = np.arange(100)
        for rule_index, rule in enumerate(RULES):
            features = binned_features(load_spikes(rule, rep))  # class, trial, neuron, window
            matrix_by_fold = np.zeros((N_FOLDS, features.shape[-1], features.shape[-1]), dtype=float)
            for fold_index, test_indices in enumerate(folds):
                train_indices = np.setdiff1d(all_indices, test_indices)
                for train_window in range(features.shape[-1]):
                    train = np.take(features, train_indices, axis=1)[..., train_window]
                    model = _fit_probe(train)
                    for test_window in range(features.shape[-1]):
                        test = np.take(features, test_indices, axis=1)[..., test_window]
                        prediction = _predict_probe(test, model)
                        matrix_by_fold[fold_index, train_window, test_window] = _scores_from_predictions(prediction)[0]
            matrix = matrix_by_fold.mean(axis=0)
            matrices[rule_index, rep_index] = matrix
            for window, accuracy in enumerate(np.diag(matrix)):
                curve_rows.append(
                    {
                        "rule": rule,
                        "rep": rep,
                        "window": window,
                        "start_ms": window * TN,
                        "end_ms": (window + 1) * TN,
                        "accuracy8": float(accuracy),
                    }
                )
        print(f"[temporal generalization] completed rep {rep}/10", flush=True)
    return pd.DataFrame(curve_rows), matrices


def _ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def create_ablation_figure(ablation: pd.DataFrame) -> None:
    paired = ablation.groupby(["rule", "rep", "condition"])[["accuracy8", "accuracy3"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    x = np.arange(len(CONDITIONS))
    styles = {"STDP": ("s", "0.65", "--"), "T_STDP": ("^", "0.35", "-."), "SRDP": ("o", "black", "-")}
    for ax, metric, title in zip(axes, ("accuracy8", "accuracy3"), ("(a) 8-class decoding", "(b) 3-group decoding")):
        for rule in RULES:
            rule_table = paired[paired["rule"] == rule].pivot(index="rep", columns="condition", values=metric).reindex(columns=CONDITIONS)
            mean = rule_table.mean(axis=0).to_numpy(float)
            ci = np.array([_ci95(rule_table[condition].to_numpy(float)) for condition in CONDITIONS])
            marker, color, linestyle = styles[rule]
            ax.errorbar(x, mean, yerr=ci, marker=marker, color=color, linestyle=linestyle,
                        markerfacecolor="white" if rule != "SRDP" else "black", capsize=3, label=rule)
        ax.axhline(1 / (8 if metric == "accuracy8" else 3), color="0.7", linestyle=":", linewidth=0.8)
        ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("cross-validated accuracy")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color="0.86", linestyle="--", linewidth=0.7)
    axes[0].legend(frameon=False, ncol=3, loc="lower center")
    fig.suptitle("Temporal and population-coding ablations (fixed diagonal-centroid probe; mean and 95% CI)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_coding_ablation.{suffix}", bbox_inches="tight")
    plt.close(fig)


def create_population_time_figure(population: pd.DataFrame, temporal: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    styles = {"STDP": ("s", "0.65", "--"), "T_STDP": ("^", "0.35", "-."), "SRDP": ("o", "black", "-")}

    population_rep = population.groupby(["rule", "rep", "n_neurons"])["accuracy8"].mean().reset_index()
    for rule in RULES:
        table = population_rep[population_rep["rule"] == rule].pivot(index="rep", columns="n_neurons", values="accuracy8").reindex(columns=NEURON_COUNTS)
        mean = table.mean(axis=0).to_numpy(float)
        ci = np.array([_ci95(table[count].to_numpy(float)) for count in NEURON_COUNTS])
        marker, color, linestyle = styles[rule]
        axes[0].errorbar(NEURON_COUNTS, mean, yerr=ci, marker=marker, color=color, linestyle=linestyle,
                         markerfacecolor="white" if rule != "SRDP" else "black", capsize=3, label=rule)
    axes[0].set_xticks(NEURON_COUNTS)
    axes[0].set_xlabel("number of output neurons")
    axes[0].set_ylabel("8-class accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("(a) Population-size decoding curve", loc="left", fontweight="bold")
    axes[0].grid(True, color="0.86", linestyle="--", linewidth=0.7)
    axes[0].legend(frameon=False)

    for rule in RULES:
        table = temporal[temporal["rule"] == rule].pivot(index="rep", columns="window", values="accuracy8")
        mean = table.mean(axis=0).to_numpy(float)
        ci = np.array([_ci95(table[window].to_numpy(float)) for window in table.columns])
        time_ms = (table.columns.to_numpy(float) + 0.5) * TN
        marker, color, linestyle = styles[rule]
        axes[1].plot(time_ms, mean, color=color, linestyle=linestyle, label=rule)
        axes[1].fill_between(time_ms, mean - ci, mean + ci, color=color, alpha=0.12)
    axes[1].axhline(1 / 8, color="0.7", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("time-window center (ms)")
    axes[1].set_ylabel("8-class accuracy using one 25-ms window")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("(b) Time-resolved decoding", loc="left", fontweight="bold")
    axes[1].grid(True, color="0.86", linestyle="--", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.suptitle("Population and temporal contributions (fixed diagonal-centroid probe)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_population_temporal_decoding.{suffix}", bbox_inches="tight")
    plt.close(fig)


def create_generalization_figure(matrices: np.ndarray) -> None:
    mean_matrices = matrices.mean(axis=1)
    vmin = 1 / 8
    vmax = max(0.5, float(np.max(mean_matrices)))
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.6), dpi=180, sharex=True, sharey=True)
    image = None
    for index, (ax, rule) in enumerate(zip(axes, RULES)):
        image = ax.imshow(mean_matrices[index], origin="lower", cmap="Greys", vmin=vmin, vmax=vmax,
                          extent=(0, 500, 0, 500), aspect="equal")
        ax.set_title(rule, fontweight="bold")
        ax.set_xlabel("test-window time (ms)")
        if index == 0:
            ax.set_ylabel("train-window time (ms)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
        colorbar.set_label("8-class accuracy")
    fig.suptitle("Temporal generalization matrices (fixed diagonal-centroid probe; mean across repetitions)", fontsize=14)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.91, wspace=0.16)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_temporal_generalization.{suffix}", bbox_inches="tight")
    plt.close(fig)


def summarize_temporal_generalization(matrices: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    off_diagonal_mask = ~np.eye(matrices.shape[-1], dtype=bool)
    for rule_index, rule in enumerate(RULES):
        for rep_index, rep in enumerate(REPS):
            matrix = matrices[rule_index, rep_index]
            diagonal_mean = float(np.mean(np.diag(matrix)))
            off_diagonal_mean = float(np.mean(matrix[off_diagonal_mask]))
            rows.append(
                {
                    "rule": rule,
                    "rep": rep,
                    "diagonal_accuracy": diagonal_mean,
                    "off_diagonal_accuracy": off_diagonal_mean,
                    "temporal_specificity": diagonal_mean - off_diagonal_mean,
                    "peak_accuracy": float(np.max(matrix)),
                }
            )
    return pd.DataFrame(rows)


def build_statistics(ablation: pd.DataFrame) -> pd.DataFrame:
    paired = ablation.groupby(["rule", "rep", "condition"])[["accuracy8", "accuracy3"]].mean().reset_index()
    rows: list[dict[str, float | int | str]] = []
    for rule in RULES:
        for metric in ("accuracy8", "accuracy3"):
            pivot = paired[paired["rule"] == rule].pivot(index="rep", columns="condition", values=metric)
            for condition in CONDITIONS[1:]:
                difference = (pivot[condition] - pivot["full"]).to_numpy(float)
                mean = float(np.mean(difference))
                half_width = _ci95(difference)
                p_value = float(wilcoxon(difference).pvalue) if np.any(np.abs(difference) > 1e-12) else 1.0
                rows.append(
                    {
                        "comparison_type": "within_rule_vs_full",
                        "family": f"within_{rule}_{metric}",
                        "rule": rule,
                        "metric": metric,
                        "comparison": f"{condition} - full",
                        "mean_difference": mean,
                        "ci95_low": mean - half_width,
                        "ci95_high": mean + half_width,
                        "wilcoxon_p": p_value,
                        "decreased_reps": int(np.sum(difference < 0)),
                    }
                )
    for metric in ("accuracy8", "accuracy3"):
        for condition in CONDITIONS:
            condition_table = paired[paired["condition"] == condition].pivot(index="rep", columns="rule", values=metric)
            difference = (condition_table["SRDP"] - condition_table["T_STDP"]).to_numpy(float)
            mean = float(np.mean(difference))
            half_width = _ci95(difference)
            p_value = float(wilcoxon(difference).pvalue) if np.any(np.abs(difference) > 1e-12) else 1.0
            rows.append(
                {
                    "comparison_type": "between_rules",
                    "family": f"between_{metric}",
                    "rule": "SRDP vs T_STDP",
                    "metric": metric,
                    "comparison": f"SRDP - T_STDP ({condition})",
                    "mean_difference": mean,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "wilcoxon_p": p_value,
                    "decreased_reps": int(np.sum(difference < 0)),
                }
            )

    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = np.nan
    for family in result["family"].unique():
        mask = result["family"] == family
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
    ablation = run_ablation()
    population = run_neuron_count_curve()
    temporal, matrices = run_temporal_generalization()
    temporal_generalization_summary = summarize_temporal_generalization(matrices)
    statistics = build_statistics(ablation)

    create_ablation_figure(ablation)
    create_population_time_figure(population, temporal)
    create_generalization_figure(matrices)

    ablation.to_csv(OUTPUT_DIR / "srdp_coding_ablation_raw.csv", index=False)
    ablation.groupby(["rule", "rep", "condition"])[["accuracy8", "accuracy3"]].mean().reset_index().to_csv(
        OUTPUT_DIR / "srdp_coding_ablation_by_rep.csv", index=False
    )
    statistics.to_csv(OUTPUT_DIR / "srdp_coding_ablation_statistics.csv", index=False)
    population.to_csv(OUTPUT_DIR / "srdp_population_size_decoding.csv", index=False)
    temporal.to_csv(OUTPUT_DIR / "srdp_time_window_decoding.csv", index=False)
    temporal_generalization_summary.to_csv(
        OUTPUT_DIR / "srdp_temporal_generalization_summary.csv", index=False
    )
    np.save(OUTPUT_DIR / "srdp_temporal_generalization_matrices.npy", matrices)

    summary = ablation.groupby(["rule", "condition"])[["accuracy8", "accuracy3"]].mean().reindex(
        pd.MultiIndex.from_product([RULES, CONDITIONS], names=["rule", "condition"])
    )
    print("\nAblation accuracy summary")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nSaved coding-ablation, population-size, time-window, and temporal-generalization outputs.")


if __name__ == "__main__":
    main()
