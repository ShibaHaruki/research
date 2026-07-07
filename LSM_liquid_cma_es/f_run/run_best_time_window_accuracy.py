"""Evaluate best-parameter accuracy for cumulative 25 ms time windows.

This script does not train anything. It uses saved internal states from the
best-parameter liquid run and evaluates eval.py-style Mahalanobis accuracy
for 0-25 ms, 0-50 ms, 0-75 ms, ..., up to the requested total duration.
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
from f_run.run_best_params_waveforms import DEFAULT_BEST_PARAMS, load_best_params
from f_run.run_best_params_accuracy import (
    find_default_internal_state_dir,
    has_required_internal_states,
    internal_state_class_counts,
)
from f_run.run_cma_es_search import apply_liquid_params
from f_run.run_common import build_cfg
from f_run.run_liquid import run_liquid
from f_run.run_random_neuron_accuracy import evaluate_random_neuron_accuracy


DEFAULT_OUT_DIR = PROJECT_ROOT / "g_tactile_results" / "best_params_time_window_accuracy"
DEFAULT_SELECTED_NEURONS = 1000
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_REPEATS = 20
DEFAULT_FOLDS = 10
DEFAULT_WINDOW_MS = 25.0
DEFAULT_TOTAL_MS = 500.0


def make_windows(total_ms: float, window_ms: float) -> list[tuple[float, float]]:
    total_ms = float(total_ms)
    window_ms = float(window_ms)
    if total_ms <= 0 or window_ms <= 0:
        raise ValueError("total_ms and window_ms must be positive")
    n_windows = int(round(total_ms / window_ms))
    if abs(n_windows * window_ms - total_ms) > 1e-9:
        raise ValueError(f"total_ms={total_ms:g} is not divisible by window_ms={window_ms:g}")
    return [(0.0, (i + 1) * window_ms) for i in range(n_windows)]


def parse_end_ms_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("end ms values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one end ms value is required")
    return values


def make_best_internal_states(
    *,
    samples_per_class: int,
    internal_state_bin_ms: float,
    pca_components: int,
    pca_max_samples_per_class: int,
) -> Path:
    params = load_best_params(DEFAULT_BEST_PARAMS)
    cfg = apply_liquid_params(build_cfg(), params)
    cfg["liquid"]["NUM_LIQUID_SAMPLE"] = [int(samples_per_class)]
    cfg["run"]["INTERNAL_STATE_BIN_MS"] = float(internal_state_bin_ms)
    cfg["run"]["INTERNAL_STATE_PCA_ENABLE"] = True
    cfg["run"]["INTERNAL_STATE_PCA_COMPONENTS"] = int(pca_components)
    cfg["run"]["INTERNAL_STATE_PCA_MAX_SAMPLES_PER_CLASS"] = int(pca_max_samples_per_class)
    cfg["experiment"] = {
        "id": "best_params_waveforms",
        "name": "best_params_waveforms",
        "trial_id": DEFAULT_BEST_PARAMS.parent.name,
    }

    print(
        "[time-window] complete best internal_states not found; "
        "running liquid first."
    )
    message = run_liquid(cfg)
    print(message)
    run_dir = Path(str(message).split(" in ", 1)[-1])
    return run_dir / str(cfg["run"].get("INTERNAL_STATE_DIR", "internal_states"))


def save_plot(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plot skipped: {type(exc).__name__}: {exc}")
        return

    ends = [float(row["window_end_ms"]) for row in rows]
    acc8 = [float(row["accuracy8_overall_mean"]) for row in rows]
    acc8_std = [float(row["accuracy8_overall_std"]) for row in rows]
    acc3 = [float(row["accuracy3_overall_mean"]) for row in rows]
    acc3_std = [float(row["accuracy3_overall_std"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(ends, acc8, yerr=acc8_std, marker="o", capsize=3, label="8-class")
    ax.errorbar(ends, acc3, yerr=acc3_std, marker="s", capsize=3, label="3-class")
    ax.set_xlabel("Used duration from 0 ms [ms]")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_window_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate best liquid accuracy for cumulative time windows."
    )
    parser.add_argument(
        "--internal-state-dir",
        type=Path,
        default=None,
        help="Internal states from run_best_params_waveforms.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--neurons", type=int, default=DEFAULT_SELECTED_NEURONS)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--total-ms", type=float, default=DEFAULT_TOTAL_MS)
    parser.add_argument("--internal-state-bin-ms", type=float, default=1.0)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--pca-max-samples-per-class", type=int, default=100)
    parser.add_argument(
        "--no-auto-run-liquid",
        action="store_true",
        help="Do not create missing best internal_states automatically.",
    )
    parser.add_argument(
        "--end-ms-list",
        type=parse_end_ms_list,
        default=None,
        help="Comma-separated cumulative end times, e.g. 25,50,100,200,500.",
    )
    args = parser.parse_args()

    if args.internal_state_dir is not None:
        internal_state_dir = Path(args.internal_state_dir)
    else:
        try:
            internal_state_dir = find_default_internal_state_dir()
        except FileNotFoundError:
            if bool(args.no_auto_run_liquid):
                raise
            internal_state_dir = make_best_internal_states(
                samples_per_class=int(args.samples_per_class),
                internal_state_bin_ms=float(args.internal_state_bin_ms),
                pca_components=int(args.pca_components),
                pca_max_samples_per_class=int(args.pca_max_samples_per_class),
            )
    if not has_required_internal_states(
        internal_state_dir,
        min_samples_per_class=int(args.samples_per_class),
    ):
        if bool(args.no_auto_run_liquid) or args.internal_state_dir is not None:
            counts = internal_state_class_counts(internal_state_dir)
            raise ValueError(
                f"{internal_state_dir} does not have {int(args.samples_per_class)} samples "
                f"for all 8 materials. Current counts: {counts}."
            )
        internal_state_dir = make_best_internal_states(
            samples_per_class=int(args.samples_per_class),
            internal_state_bin_ms=float(args.internal_state_bin_ms),
            pca_components=int(args.pca_components),
            pca_max_samples_per_class=int(args.pca_max_samples_per_class),
        )
        if not has_required_internal_states(
            internal_state_dir,
            min_samples_per_class=int(args.samples_per_class),
        ):
            counts = internal_state_class_counts(internal_state_dir)
            raise ValueError(
                f"internal state generation finished, but counts are still incomplete: {counts}"
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    epsilon_ms = 1e-9
    windows = (
        [(0.0, float(end_ms)) for end_ms in args.end_ms_list]
        if args.end_ms_list is not None
        else make_windows(args.total_ms, args.window_ms)
    )
    for window_index, (start_ms, end_ms) in enumerate(windows, start=1):
        duration_ms = float(end_ms) - float(start_ms)
        if duration_ms <= 0:
            raise ValueError(f"invalid window {start_ms:g}-{end_ms:g} ms")
        n_intervals_float = duration_ms / float(args.window_ms)
        if abs(round(n_intervals_float) - n_intervals_float) > 1e-9:
            raise ValueError(
                f"window {start_ms:g}-{end_ms:g} ms is not divisible by "
                f"window_ms={float(args.window_ms):g}"
            )
        label = f"{int(round(start_ms)):03d}_{int(round(end_ms)):03d}ms"
        window_out_dir = out_dir / f"window_{label}"
        print(f"[time-window] evaluating {start_ms:g}-{end_ms:g} ms")
        metrics = evaluate_random_neuron_accuracy(
            internal_state_dir,
            n_neurons=int(args.neurons),
            n_repeats=int(args.repeats),
            n_folds=int(args.folds),
            seed_value=int(args.seed) + window_index * 10000,
            t_n_ms=float(args.window_ms),
            max_samples_per_class=int(args.samples_per_class),
            window_start_ms=float(start_ms),
            window_end_ms=float(end_ms) - epsilon_ms,
            out_dir=window_out_dir,
        )
        row = {
            "window_index": int(window_index),
            "window_start_ms": float(start_ms),
            "window_end_ms": float(end_ms),
            "used_duration_ms": float(end_ms - start_ms),
            "n_neurons": int(args.neurons),
            "samples_per_class": int(args.samples_per_class),
            "repeats": int(args.repeats),
            "folds": int(args.folds),
            "accuracy8_overall_mean": metrics["accuracy8_overall_mean"],
            "accuracy8_overall_std": metrics["accuracy8_overall_std"],
            "accuracy3_overall_mean": metrics["accuracy3_overall_mean"],
            "accuracy3_overall_std": metrics["accuracy3_overall_std"],
            "fisher_ratio_DR_mean": metrics.get("fisher_ratio_DR_mean"),
            "fisher_ratio_DR_std": metrics.get("fisher_ratio_DR_std"),
            "result_dir": str(window_out_dir),
        }
        rows.append(row)
        print(
            "[time-window] "
            f"{start_ms:g}-{end_ms:g} ms "
            f"acc8={row['accuracy8_overall_mean']:.4f}±{row['accuracy8_overall_std']:.4f} "
            f"acc3={row['accuracy3_overall_mean']:.4f}±{row['accuracy3_overall_std']:.4f}"
        )

    csv_path = out_dir / "time_window_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "internal_state_dir": str(internal_state_dir),
        "out_dir": str(out_dir),
        "window_ms": float(args.window_ms),
        "total_ms": float(args.total_ms),
        "end_ms_list": None if args.end_ms_list is None else [float(v) for v in args.end_ms_list],
        "n_windows": len(rows),
        "rows": rows,
        "summary_csv": str(csv_path),
    }
    (out_dir / "time_window_accuracy.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_plot(rows, out_dir)
    print(f"[time-window] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
