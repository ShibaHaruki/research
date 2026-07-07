"""Evaluate best-parameter accuracy for sliding 25 ms-multiple windows.

This script does not train anything. It uses saved internal states from the
best-parameter liquid run and evaluates eval.py-style Mahalanobis accuracy for
25 ms windows, 50 ms windows, 75 ms windows, ..., sliding each window by 25 ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d_tools.run_paths import jsonable
from f_run.run_best_params_accuracy import (
    find_default_internal_state_dir,
    has_required_internal_states,
    internal_state_class_counts,
)
from f_run.run_best_time_window_accuracy import make_best_internal_states
from f_run.run_random_neuron_accuracy import evaluate_random_neuron_accuracy


DEFAULT_OUT_DIR = PROJECT_ROOT / "g_tactile_results" / "best_params_25ms_step_accuracy"
DEFAULT_SELECTED_NEURONS = 1000
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_REPEATS = 20
DEFAULT_FOLDS = 10
DEFAULT_WINDOW_MS = 25.0
DEFAULT_TOTAL_MS = 500.0


def parse_step_list(text: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("window step counts must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one window step count is required")
    return values


def make_sliding_windows(
    total_ms: float,
    window_ms: float,
    *,
    length_steps: list[int] | None = None,
) -> list[tuple[int, float, float]]:
    total_ms = float(total_ms)
    window_ms = float(window_ms)
    if total_ms <= 0 or window_ms <= 0:
        raise ValueError("total_ms and window_ms must be positive")
    n_total_steps = int(round(total_ms / window_ms))
    if abs(n_total_steps * window_ms - total_ms) > 1e-9:
        raise ValueError(f"total_ms={total_ms:g} is not divisible by window_ms={window_ms:g}")
    if length_steps is None:
        length_steps = list(range(1, n_total_steps + 1))

    windows = []
    for n_steps in length_steps:
        n_steps = int(n_steps)
        if n_steps > n_total_steps:
            raise ValueError(
                f"window length step {n_steps} exceeds total steps {n_total_steps}"
            )
        for start_step in range(0, n_total_steps - n_steps + 1):
            start_ms = start_step * window_ms
            end_ms = (start_step + n_steps) * window_ms
            windows.append((n_steps, start_ms, end_ms))
    return windows


def save_plot(rows: list[dict], out_dir: Path, *, window_ms: float = DEFAULT_WINDOW_MS) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plot skipped: {type(exc).__name__}: {exc}")
        return

    durations = sorted({float(row["window_duration_ms"]) for row in rows})

    fig, ax = plt.subplots(figsize=(10, 5))
    for duration in durations:
        subset = [row for row in rows if float(row["window_duration_ms"]) == duration]
        centers = [
            (float(row["window_start_ms"]) + float(row["window_end_ms"])) / 2.0
            for row in subset
        ]
        acc8 = [float(row["accuracy8_overall_mean"]) for row in subset]
        ax.plot(centers, acc8, marker="o", linewidth=1.0, label=f"{duration:g} ms")
    ax.set_xlabel("Window center [ms]")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Window length", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "best_sliding_window_accuracy.png", dpi=160)
    plt.close(fig)

    starts = sorted({float(row["window_start_ms"]) for row in rows})
    heat = np.full((len(durations), len(starts)), np.nan, dtype=float)
    for row in rows:
        duration_index = durations.index(float(row["window_duration_ms"]))
        start_index = starts.index(float(row["window_start_ms"]))
        heat[duration_index, start_index] = float(row["accuracy8_overall_mean"])

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        extent=[
            min(starts) - DEFAULT_WINDOW_MS / 2.0,
            max(starts) + DEFAULT_WINDOW_MS / 2.0,
            min(durations) - DEFAULT_WINDOW_MS / 2.0,
            max(durations) + DEFAULT_WINDOW_MS / 2.0,
        ],
    )
    ax.set_xlabel("Window start [ms]")
    ax.set_ylabel("Window length [ms]")
    ax.set_title("8-class accuracy")
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "best_sliding_window_accuracy_heatmap.png", dpi=160)
    plt.close(fig)

    duration_rows = build_duration_summary(rows, window_ms=window_ms)
    duration_ms = [float(row["window_duration_ms"]) for row in duration_rows]
    acc8_mean = [float(row["accuracy8_mean_over_starts"]) for row in duration_rows]
    acc8_max = [float(row["accuracy8_max_over_starts"]) for row in duration_rows]
    acc8_min = [float(row["accuracy8_min_over_starts"]) for row in duration_rows]
    acc3_mean = [float(row["accuracy3_mean_over_starts"]) for row in duration_rows]
    acc3_max = [float(row["accuracy3_max_over_starts"]) for row in duration_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(duration_ms, acc8_mean, marker="o", label="8-class mean")
    ax.plot(duration_ms, acc8_max, marker="^", label="8-class max")
    ax.plot(duration_ms, acc8_min, marker="v", label="8-class min")
    ax.plot(duration_ms, acc3_mean, marker="s", label="3-class mean")
    ax.plot(duration_ms, acc3_max, marker="D", label="3-class max")
    ax.set_xlabel("Window length [ms]")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "best_sliding_window_accuracy_trend.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        duration_ms,
        acc8_mean,
        yerr=[float(row["accuracy8_std_over_starts"]) for row in duration_rows],
        marker="o",
        capsize=3,
        label="8-class mean",
    )
    ax.errorbar(
        duration_ms,
        acc3_mean,
        yerr=[float(row["accuracy3_std_over_starts"]) for row in duration_rows],
        marker="s",
        capsize=3,
        label="3-class mean",
    )
    ax.set_xlabel("Classification time interval [ms]")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_accuracy_vs_time_interval.png", dpi=160)
    plt.close(fig)

    table_rows = []
    for row in duration_rows:
        table_rows.append(
            [
                f"{float(row['window_duration_ms']):.0f}",
                f"{float(row['accuracy8_mean_over_starts']):.4f}",
                f"{float(row['accuracy8_std_over_starts']):.4f}",
                f"{float(row['accuracy3_mean_over_starts']):.4f}",
                f"{float(row['accuracy3_std_over_starts']):.4f}",
            ]
        )
    fig_height = max(4.0, 0.36 * (len(table_rows) + 1))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Time interval [ms]",
            "8-class mean",
            "8-class std",
            "3-class mean",
            "3-class std",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_accuracy_vs_time_interval_table.png", dpi=180)
    plt.close(fig)

    per_length_dir = out_dir / "accuracy_by_window_length"
    per_length_dir.mkdir(parents=True, exist_ok=True)
    for duration in durations:
        subset = [row for row in rows if float(row["window_duration_ms"]) == duration]
        centers = [
            (float(row["window_start_ms"]) + float(row["window_end_ms"])) / 2.0
            for row in subset
        ]
        acc8 = [float(row["accuracy8_overall_mean"]) for row in subset]
        acc8_std = [float(row["accuracy8_overall_std"]) for row in subset]
        acc3 = [float(row["accuracy3_overall_mean"]) for row in subset]
        acc3_std = [float(row["accuracy3_overall_std"]) for row in subset]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(centers, acc8, yerr=acc8_std, marker="o", capsize=3, label="8-class")
        ax.errorbar(centers, acc3, yerr=acc3_std, marker="s", capsize=3, label="3-class")
        ax.set_xlabel(f"{duration:g} ms window center [ms]")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        length_steps = int(round(float(duration) / float(window_ms)))
        fig.savefig(
            per_length_dir / f"accuracy_len{length_steps:02d}_{int(round(duration)):03d}ms.png",
            dpi=160,
        )
        plt.close(fig)


def build_duration_summary(
    rows: list[dict],
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
) -> list[dict]:
    duration_rows = []
    durations = sorted({float(row["window_duration_ms"]) for row in rows})
    for duration in durations:
        subset = [row for row in rows if float(row["window_duration_ms"]) == duration]
        acc8 = np.asarray([float(row["accuracy8_overall_mean"]) for row in subset], dtype=float)
        acc3 = np.asarray([float(row["accuracy3_overall_mean"]) for row in subset], dtype=float)
        best8_index = int(np.argmax(acc8))
        best3_index = int(np.argmax(acc3))
        duration_rows.append(
            {
                "window_duration_ms": float(duration),
                "window_length_steps": int(round(duration / float(window_ms))),
                "n_start_positions": int(len(subset)),
                "accuracy8_mean_over_starts": float(np.mean(acc8)),
                "accuracy8_std_over_starts": float(np.std(acc8, ddof=1 if len(acc8) > 1 else 0)),
                "accuracy8_max_over_starts": float(np.max(acc8)),
                "accuracy8_min_over_starts": float(np.min(acc8)),
                "accuracy8_best_window_start_ms": float(subset[best8_index]["window_start_ms"]),
                "accuracy8_best_window_end_ms": float(subset[best8_index]["window_end_ms"]),
                "accuracy3_mean_over_starts": float(np.mean(acc3)),
                "accuracy3_std_over_starts": float(np.std(acc3, ddof=1 if len(acc3) > 1 else 0)),
                "accuracy3_max_over_starts": float(np.max(acc3)),
                "accuracy3_min_over_starts": float(np.min(acc3)),
                "accuracy3_best_window_start_ms": float(subset[best3_index]["window_start_ms"]),
                "accuracy3_best_window_end_ms": float(subset[best3_index]["window_end_ms"]),
            }
        )
    return duration_rows


def _read_rows_from_csv(csv_path: Path) -> list[dict]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_or_make_internal_state_dir(args: argparse.Namespace) -> Path:
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

    if has_required_internal_states(
        internal_state_dir,
        min_samples_per_class=int(args.samples_per_class),
    ):
        return internal_state_dir

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
    return internal_state_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate best liquid accuracy for sliding 25 ms-multiple windows."
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
    parser.add_argument(
        "--window-length-steps",
        type=parse_step_list,
        default=None,
        help=(
            "Comma-separated window lengths in 25 ms steps. "
            "Default is all lengths, e.g. 1..20 for 500 ms."
        ),
    )
    parser.add_argument("--internal-state-bin-ms", type=float, default=1.0)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--pca-max-samples-per-class", type=int, default=100)
    parser.add_argument(
        "--no-auto-run-liquid",
        action="store_true",
        help="Do not create missing best internal_states automatically.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only redraw plots from an existing best_sliding_window_accuracy.csv.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        csv_path = out_dir / "best_sliding_window_accuracy.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} was not found")
        rows = _read_rows_from_csv(csv_path)
        duration_rows = build_duration_summary(rows, window_ms=float(args.window_ms))
        duration_csv_path = out_dir / "best_sliding_window_accuracy_by_length.csv"
        with duration_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(duration_rows[0].keys()))
            writer.writeheader()
            writer.writerows(duration_rows)
        save_plot(rows, out_dir, window_ms=float(args.window_ms))
        print(f"[sliding-window] plots redrawn from {csv_path}")
        return 0

    internal_state_dir = get_or_make_internal_state_dir(args)

    rows = []
    epsilon_ms = 1e-9
    windows = make_sliding_windows(
        args.total_ms,
        args.window_ms,
        length_steps=args.window_length_steps,
    )
    for window_index, (n_steps, start_ms, end_ms) in enumerate(
        windows,
        start=1,
    ):
        duration_ms = float(end_ms - start_ms)
        label = f"len{int(n_steps):02d}__{int(round(start_ms)):03d}_{int(round(end_ms)):03d}ms"
        window_out_dir = out_dir / f"window_{label}"
        print(f"[sliding-window] evaluating {start_ms:g}-{end_ms:g} ms")
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
            "window_length_steps": int(n_steps),
            "window_start_ms": float(start_ms),
            "window_end_ms": float(end_ms),
            "window_duration_ms": float(duration_ms),
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
            "[sliding-window] "
            f"{start_ms:g}-{end_ms:g} ms "
            f"acc8={row['accuracy8_overall_mean']:.4f}±{row['accuracy8_overall_std']:.4f} "
            f"acc3={row['accuracy3_overall_mean']:.4f}±{row['accuracy3_overall_std']:.4f}"
        )

    csv_path = out_dir / "best_sliding_window_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    duration_rows = build_duration_summary(rows, window_ms=float(args.window_ms))
    duration_csv_path = out_dir / "best_sliding_window_accuracy_by_length.csv"
    with duration_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(duration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(duration_rows)

    summary = {
        "internal_state_dir": str(internal_state_dir),
        "out_dir": str(out_dir),
        "window_ms": float(args.window_ms),
        "total_ms": float(args.total_ms),
        "window_length_steps": (
            None
            if args.window_length_steps is None
            else [int(value) for value in args.window_length_steps]
        ),
        "n_windows": len(rows),
        "rows": rows,
        "summary_csv": str(csv_path),
        "duration_summary_csv": str(duration_csv_path),
        "duration_summary_rows": duration_rows,
    }
    (out_dir / "best_sliding_window_accuracy.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_plot(rows, out_dir, window_ms=float(args.window_ms))
    print(f"[sliding-window] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
