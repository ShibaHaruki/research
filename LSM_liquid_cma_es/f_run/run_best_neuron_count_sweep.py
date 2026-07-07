"""Evaluate best-parameter liquid accuracy while reducing neuron count.

This script does not train anything. It uses saved internal states from the
best-parameter liquid run, evaluates eval.py-style Mahalanobis accuracy for
several selected-neuron counts, and saves a CSV/JSON/plot summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d_tools.run_paths import jsonable
from f_run.run_random_neuron_accuracy import evaluate_random_neuron_accuracy


LIQUID_RUN_DIR = PROJECT_ROOT / "g_tactile_results" / "liquid_run"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "g_tactile_results"
    / "best_params_neuron_count_sweep"
)
DEFAULT_NEURON_COUNTS = [1000, 800, 600, 400, 300, 200, 100, 50, 25, 10]


def find_default_internal_state_dir() -> Path:
    candidates = [
        path
        for path in LIQUID_RUN_DIR.rglob("best_params_waveforms/internal_states")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No best_params_waveforms/internal_states directory was found. "
            "Run f_run/run_best_params_waveforms.py first."
        )
    nliq1000 = [
        path
        for path in candidates
        if any(part.startswith("Nliq_1000__") for part in path.parts)
    ]
    selected = nliq1000 if nliq1000 else candidates
    return max(selected, key=lambda path: path.stat().st_mtime)


def parse_neuron_counts(text: str) -> list[int]:
    counts = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("neuron counts must be positive")
        counts.append(value)
    if not counts:
        raise ValueError("at least one neuron count is required")
    return counts


def save_plot(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plot skipped: {type(exc).__name__}: {exc}")
        return

    counts = [int(row["n_neurons"]) for row in rows]
    acc8 = [float(row["accuracy8_overall_mean"]) for row in rows]
    acc8_std = [float(row["accuracy8_overall_std"]) for row in rows]
    acc3 = [float(row["accuracy3_overall_mean"]) for row in rows]
    acc3_std = [float(row["accuracy3_overall_std"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(counts, acc8, yerr=acc8_std, marker="o", capsize=3, label="8-class")
    ax.errorbar(counts, acc3, yerr=acc3_std, marker="s", capsize=3, label="3-class")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Selected liquid neurons")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "neuron_count_accuracy_sweep.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep selected-neuron count for best liquid accuracy."
    )
    parser.add_argument(
        "--internal-state-dir",
        type=Path,
        default=None,
        help="Internal states from the best-parameter liquid run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where sweep summaries are saved.",
    )
    parser.add_argument(
        "--neuron-counts",
        type=parse_neuron_counts,
        default=DEFAULT_NEURON_COUNTS,
        help="Comma-separated selected-neuron counts, e.g. 1000,800,600,400,200,100.",
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-n-ms", type=float, default=25.0)
    args = parser.parse_args()

    internal_state_dir = (
        Path(args.internal_state_dir)
        if args.internal_state_dir is not None
        else find_default_internal_state_dir()
    )
    if not internal_state_dir.exists():
        raise FileNotFoundError(
            f"{internal_state_dir} does not exist. "
            "Run f_run/run_best_params_waveforms.py first."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_neurons in args.neuron_counts:
        count_dir = out_dir / f"neurons_{int(n_neurons):04d}"
        print(f"[sweep] evaluating n_neurons={int(n_neurons)}")
        metrics = evaluate_random_neuron_accuracy(
            internal_state_dir,
            n_neurons=int(n_neurons),
            n_repeats=int(args.repeats),
            n_folds=int(args.folds),
            seed_value=int(args.seed),
            t_n_ms=float(args.t_n_ms),
            max_samples_per_class=int(args.samples_per_class),
            out_dir=count_dir,
        )
        row = {
            "n_neurons": int(n_neurons),
            "samples_per_class": int(args.samples_per_class),
            "repeats": int(args.repeats),
            "folds": int(args.folds),
            "T_n_ms": float(args.t_n_ms),
            "accuracy8_overall_mean": metrics["accuracy8_overall_mean"],
            "accuracy8_overall_std": metrics["accuracy8_overall_std"],
            "accuracy3_overall_mean": metrics["accuracy3_overall_mean"],
            "accuracy3_overall_std": metrics["accuracy3_overall_std"],
            "fisher_ratio_DR_mean": metrics.get("fisher_ratio_DR_mean"),
            "fisher_ratio_DR_std": metrics.get("fisher_ratio_DR_std"),
            "result_dir": str(count_dir),
        }
        rows.append(row)
        print(
            "[sweep] "
            f"n={int(n_neurons)} "
            f"acc8={row['accuracy8_overall_mean']:.4f}±{row['accuracy8_overall_std']:.4f} "
            f"acc3={row['accuracy3_overall_mean']:.4f}±{row['accuracy3_overall_std']:.4f}"
        )

    csv_path = out_dir / "neuron_count_accuracy_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "internal_state_dir": str(internal_state_dir),
        "out_dir": str(out_dir),
        "neuron_counts": [int(v) for v in args.neuron_counts],
        "samples_per_class": int(args.samples_per_class),
        "repeats": int(args.repeats),
        "folds": int(args.folds),
        "T_n_ms": float(args.t_n_ms),
        "rows": rows,
        "summary_csv": str(csv_path),
    }
    (out_dir / "neuron_count_accuracy_sweep.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_plot(rows, out_dir)
    print(f"[sweep] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
