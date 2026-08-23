"""Analyze why SRDP outperforms STDP variants in the 10-fold results.

The script reads the existing per-repetition Excel workbooks without modifying
them and creates a publication-oriented summary figure plus auditable CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_10fold"
OUTPUT_DIR = Path(__file__).resolve().parent

TARGET_TN = 25
RULES = ("STDP", "T_STDP", "SRDP")
RULE_DIRS = {rule: RESULTS_ROOT / f"{rule}_1" for rule in RULES}
METRICS = ("accuracy8_overall", "accuracy3_overall")

CLASS_NAME_MAP = {
    "Al_board": "aluminum bd.",
    "buta_omote": "outer pigskin",
    "buta_ura": "back pigskin",
    "cork": "cork",
    "denim": "denim",
    "rubber_board": "rubber bd.",
    "washi": "Japanese paper",
    "wood_board": "wood bd.",
}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum != 0)


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def load_results() -> tuple[pd.DataFrame, dict[str, list[np.ndarray]], list[str]]:
    accuracy_rows: list[dict[str, float | int | str]] = []
    confusion_by_rule: dict[str, list[np.ndarray]] = {rule: [] for rule in RULES}
    class_names: list[str] | None = None

    for rule in RULES:
        files = sorted(RULE_DIRS[rule].glob(f"{rule}_1_rep*_Tn_{TARGET_TN}_10fold_conf_matrices.xlsx"))
        if not files:
            raise FileNotFoundError(f"No Tn={TARGET_TN} result files found in {RULE_DIRS[rule]}")

        for path in files:
            accuracy = pd.read_excel(path, sheet_name="accuracy")
            if len(accuracy) != 1:
                raise ValueError(f"Expected one accuracy row in {path}, found {len(accuracy)}")
            row = accuracy.iloc[0]
            rep = int(row["rep"])
            accuracy_rows.append(
                {
                    "rule": rule,
                    "rep": rep,
                    "accuracy8": float(row["accuracy8_overall"]),
                    "accuracy3": float(row["accuracy3_overall"]),
                    "source_file": str(path),
                }
            )

            conf8 = pd.read_excel(path, sheet_name="conf_8cls", index_col=0)
            labels = [str(label) for label in conf8.index]
            if class_names is None:
                class_names = labels
            elif labels != class_names:
                raise ValueError(f"Class order differs in {path}")
            confusion_by_rule[rule].append(_normalize_rows(conf8.to_numpy(dtype=float)))

    scores = pd.DataFrame(accuracy_rows).sort_values(["rep", "rule"]).reset_index(drop=True)
    rep_counts = scores.groupby("rule")["rep"].nunique()
    if rep_counts.nunique() != 1:
        raise ValueError(f"Unequal repetition counts: {rep_counts.to_dict()}")
    return scores, confusion_by_rule, class_names or []


def build_summary(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    summary_rows: list[dict[str, float | int | str]] = []
    comparison_rows: list[dict[str, float | int | str]] = []
    strongest: dict[str, str] = {}

    for metric in ("accuracy8", "accuracy3"):
        means = scores.groupby("rule")[metric].mean().reindex(RULES)
        strongest[metric] = str(means.drop("SRDP").idxmax())
        for rule in RULES:
            values = scores.loc[scores["rule"] == rule, metric].to_numpy(float)
            mean, low, high = _mean_ci95(values)
            summary_rows.append(
                {
                    "metric": metric,
                    "rule": rule,
                    "n_reps": len(values),
                    "mean": mean,
                    "std": float(np.std(values, ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

        for baseline in ("STDP", "T_STDP"):
            paired = scores.pivot(index="rep", columns="rule", values=metric)[[baseline, "SRDP"]].dropna()
            difference = paired["SRDP"].to_numpy(float) - paired[baseline].to_numpy(float)
            mean, low, high = _mean_ci95(difference)
            p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
            comparison_rows.append(
                {
                    "metric": metric,
                    "comparison": f"SRDP - {baseline}",
                    "n_pairs": len(difference),
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wilcoxon_p": p_value,
                    "srdp_wins": int(np.sum(difference > 0)),
                    "ties": int(np.sum(difference == 0)),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows), strongest


def _paired_accuracy_panel(
    ax: plt.Axes,
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    metric: str,
    panel_label: str,
) -> None:
    pivot = scores.pivot(index="rep", columns="rule", values=metric).reindex(columns=RULES)
    positions = np.arange(1, len(RULES) + 1)

    for _, row in pivot.iterrows():
        ax.plot(positions, row.to_numpy(float), color="0.78", linewidth=0.8, zorder=1)
        ax.scatter(positions, row.to_numpy(float), s=15, facecolor="white", edgecolor="0.4", linewidth=0.7, zorder=2)

    values = [pivot[rule].dropna().to_numpy(float) for rule in RULES]
    ax.boxplot(
        values,
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

    baseline = "T_STDP"
    comp = comparisons[(comparisons["metric"] == metric) & (comparisons["comparison"] == f"SRDP - {baseline}")].iloc[0]
    text = (
        f"SRDP - {baseline}: {100 * comp['mean_difference']:+.1f} pp\n"
        f"95% CI [{100 * comp['ci95_low']:+.1f}, {100 * comp['ci95_high']:+.1f}], "
        f"wins {int(comp['srdp_wins'])}/{int(comp['n_pairs'])}, "
        f"Wilcoxon p={comp['wilcoxon_p']:.3f}"
    )
    ax.text(0.03, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xticks(positions, RULES)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_ylabel("accuracy")
    ax.grid(True, axis="y", color="0.85", linestyle="--", linewidth=0.7)
    ax.set_title(f"{panel_label} {metric.replace('accuracy', '')}-class accuracy", loc="left", fontweight="bold")


def _class_recall_panel(
    ax: plt.Axes,
    confusion_by_rule: dict[str, list[np.ndarray]],
    class_names: list[str],
    baseline: str,
) -> pd.DataFrame:
    labels = [CLASS_NAME_MAP.get(name, name) for name in class_names]
    y = np.arange(len(labels))
    offsets = {"STDP": -0.20, "T_STDP": 0.0, "SRDP": 0.20}
    markers = {"STDP": "s", "T_STDP": "^", "SRDP": "o"}
    recall_rows: list[dict[str, float | str]] = []

    for rule in RULES:
        recalls = np.stack(confusion_by_rule[rule])[:, np.arange(len(labels)), np.arange(len(labels))]
        means = recalls.mean(axis=0)
        ci = t.ppf(0.975, recalls.shape[0] - 1) * recalls.std(axis=0, ddof=1) / np.sqrt(recalls.shape[0])
        ax.errorbar(
            means,
            y + offsets[rule],
            xerr=ci,
            fmt=markers[rule],
            markersize=4.5,
            markerfacecolor="white" if rule != "SRDP" else "black",
            markeredgecolor="black",
            color="black" if rule == "SRDP" else ("0.35" if rule == "T_STDP" else "0.65"),
            linewidth=1.0,
            capsize=2,
            label=rule,
        )
        for class_name, mean, half_width in zip(class_names, means, ci):
            recall_rows.append(
                {
                    "rule": rule,
                    "class": class_name,
                    "mean_recall": float(mean),
                    "ci95_low": float(mean - half_width),
                    "ci95_high": float(mean + half_width),
                }
            )

    baseline_recalls = np.stack(confusion_by_rule[baseline])[:, np.arange(len(labels)), np.arange(len(labels))]
    srdp_recalls = np.stack(confusion_by_rule["SRDP"])[:, np.arange(len(labels)), np.arange(len(labels))]
    improvement = 100 * (srdp_recalls - baseline_recalls).mean(axis=0)
    for index, value in enumerate(improvement):
        ax.text(1.01, index, f"{value:+.1f} pp", va="center", ha="left", fontsize=7, clip_on=False)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("class recall (mean and 95% CI)")
    ax.grid(True, axis="x", color="0.88", linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.set_title(f"(c) Class-wise recall; right labels = SRDP - {baseline}", loc="left", fontweight="bold")
    return pd.DataFrame(recall_rows)


def _confusion_difference_panel(
    ax: plt.Axes,
    confusion_by_rule: dict[str, list[np.ndarray]],
    class_names: list[str],
    baseline: str,
) -> None:
    labels = [CLASS_NAME_MAP.get(name, name) for name in class_names]
    difference = 100 * (
        np.mean(np.stack(confusion_by_rule["SRDP"]), axis=0)
        - np.mean(np.stack(confusion_by_rule[baseline]), axis=0)
    )
    limit = max(5.0, float(np.ceil(np.max(np.abs(difference)) / 5.0) * 5.0))
    image = ax.imshow(difference, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"(d) Normalized confusion difference: SRDP - {baseline} (pp)", loc="left", fontweight="bold")
    for i in range(difference.shape[0]):
        for j in range(difference.shape[1]):
            value = difference[i, j]
            ax.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value) > 0.55 * limit else "black")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("percentage-point change")


def create_figure(
    scores: pd.DataFrame,
    confusion_by_rule: dict[str, list[np.ndarray]],
    class_names: list[str],
    comparisons: pd.DataFrame,
    strongest: dict[str, str],
) -> pd.DataFrame:
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig = plt.figure(figsize=(13.2, 10.2), dpi=180)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1.15), hspace=0.34, wspace=0.34)

    _paired_accuracy_panel(fig.add_subplot(grid[0, 0]), scores, comparisons, "accuracy8", "(a)")
    _paired_accuracy_panel(fig.add_subplot(grid[0, 1]), scores, comparisons, "accuracy3", "(b)")
    recall_table = _class_recall_panel(
        fig.add_subplot(grid[1, 0]), confusion_by_rule, class_names, strongest["accuracy8"]
    )
    _confusion_difference_panel(
        fig.add_subplot(grid[1, 1]), confusion_by_rule, class_names, strongest["accuracy8"]
    )

    fig.suptitle(
        f"Evidence for SRDP accuracy advantage (Tn={TARGET_TN}, paired across repetitions)",
        fontsize=14,
        y=0.995,
    )
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.11, right=0.96)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"srdp_advantage_analysis.{suffix}", bbox_inches="tight")
    plt.close(fig)
    return recall_table


def main() -> None:
    scores, confusion_by_rule, class_names = load_results()
    summary, comparisons, strongest = build_summary(scores)
    recall_table = create_figure(scores, confusion_by_rule, class_names, comparisons, strongest)

    scores.to_csv(OUTPUT_DIR / "srdp_accuracy_by_rep.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "srdp_accuracy_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "srdp_paired_comparisons.csv", index=False)
    recall_table.to_csv(OUTPUT_DIR / "srdp_class_recall_summary.csv", index=False)

    print("\nAccuracy summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPaired SRDP comparisons")
    print(comparisons.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {OUTPUT_DIR / 'srdp_advantage_analysis.png'}")
    print(f"Saved: {OUTPUT_DIR / 'srdp_advantage_analysis.pdf'}")


if __name__ == "__main__":
    main()
