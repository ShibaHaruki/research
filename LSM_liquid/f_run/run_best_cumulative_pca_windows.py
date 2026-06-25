"""Run PCA for cumulative best-parameter time windows.

This script does not train anything. It uses saved internal states from the
best-parameter liquid run and runs PCA for 0-25 ms, 0-50 ms, 0-75 ms, ... so
you can see how PCA changes as more 25 ms intervals are included.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d_tools.pca import save_internal_state_pca
from d_tools.run_paths import jsonable
from f_run.run_best_params_accuracy import (
    find_default_internal_state_dir,
    has_required_internal_states,
    internal_state_class_counts,
)
from f_run.run_best_time_window_accuracy import make_best_internal_states


DEFAULT_OUT_DIR = PROJECT_ROOT / "g_tactile_results" / "best_params_cumulative_pca_windows"
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_WINDOW_MS = 25.0
DEFAULT_TOTAL_MS = 500.0
DEFAULT_COMPONENTS = 3


def make_cumulative_windows(total_ms: float, window_ms: float) -> list[tuple[float, float]]:
    total_ms = float(total_ms)
    window_ms = float(window_ms)
    if total_ms <= 0 or window_ms <= 0:
        raise ValueError("total_ms and window_ms must be positive")
    n_windows = int(round(total_ms / window_ms))
    if abs(n_windows * window_ms - total_ms) > 1e-9:
        raise ValueError(f"total_ms={total_ms:g} is not divisible by window_ms={window_ms:g}")
    return [(0.0, (i + 1) * window_ms) for i in range(n_windows)]


def make_step_windows(total_ms: float, window_ms: float) -> list[tuple[float, float]]:
    total_ms = float(total_ms)
    window_ms = float(window_ms)
    if total_ms <= 0 or window_ms <= 0:
        raise ValueError("total_ms and window_ms must be positive")
    n_windows = int(round(total_ms / window_ms))
    if abs(n_windows * window_ms - total_ms) > 1e-9:
        raise ValueError(f"total_ms={total_ms:g} is not divisible by window_ms={window_ms:g}")
    return [(i * window_ms, (i + 1) * window_ms) for i in range(n_windows)]


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
                pca_components=int(args.components),
                pca_max_samples_per_class=int(args.samples_per_class),
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
        pca_components=int(args.components),
        pca_max_samples_per_class=int(args.samples_per_class),
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


def load_explained_ratios(summary: dict) -> list[float]:
    csv_path = Path(summary["explained_variance_csv"])
    df = pd.read_csv(csv_path)
    return [float(value) for value in df["explained_variance_ratio"].to_list()]


def save_summary_plot(
    rows: list[dict],
    out_dir: Path,
    *,
    x_key: str,
    x_label: str,
    out_name: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plot skipped: {type(exc).__name__}: {exc}")
        return

    x_values = [float(row[x_key]) for row in rows]
    pc1 = [float(row.get("pc1_ratio", 0.0)) for row in rows]
    pc2 = [float(row.get("pc2_ratio", 0.0)) for row in rows]
    pc3 = [float(row.get("pc3_ratio", 0.0)) for row in rows]
    cumulative = [float(row.get("pc123_ratio", 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_values, pc1, marker="o", label="PC1")
    ax.plot(x_values, pc2, marker="s", label="PC2")
    ax.plot(x_values, pc3, marker="^", label="PC3")
    ax.plot(x_values, cumulative, marker="D", label="PC1+PC2+PC3")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=160)
    plt.close(fig)


def run_pca_windows(
    *,
    internal_state_dir: Path,
    out_dir: Path,
    windows: list[tuple[float, float]],
    samples_per_class: int,
    components: int,
    log_prefix: str,
    dir_prefix: str,
) -> list[dict]:
    rows = []
    epsilon_ms = 1e-9
    for window_index, (start_ms, end_ms) in enumerate(windows, start=1):
        label = f"{int(round(start_ms)):03d}_{int(round(end_ms)):03d}ms"
        pca_dir = out_dir / f"{dir_prefix}_{label}"
        print(f"[{log_prefix}] PCA {start_ms:g}-{end_ms:g} ms")
        summary = save_internal_state_pca(
            internal_state_dir,
            pca_dir,
            feature_mode="flatten",
            n_components=int(components),
            standardize=True,
            max_samples_per_class=int(samples_per_class),
            window_start_ms=float(start_ms),
            window_end_ms=float(end_ms) - epsilon_ms,
        )
        ratios = load_explained_ratios(summary)
        row = {
            "window_index": int(window_index),
            "window_start_ms": float(start_ms),
            "window_end_ms": float(end_ms),
            "window_center_ms": float((start_ms + end_ms) / 2.0),
            "used_duration_ms": float(end_ms - start_ms),
            "n_samples": int(summary["n_samples"]),
            "n_features": int(summary["n_features"]),
            "state_neurons": int(summary["state_neurons"]),
            "state_time_steps": int(summary["state_time_steps"]),
            "pc1_ratio": ratios[0] if len(ratios) > 0 else None,
            "pc2_ratio": ratios[1] if len(ratios) > 1 else None,
            "pc3_ratio": ratios[2] if len(ratios) > 2 else None,
            "pc123_ratio": float(sum(ratios[:3])),
            "pca_dir": str(pca_dir),
            "scores_plot": summary.get("scores_plot", ""),
            "explained_variance_plot": summary.get("explained_variance_plot", ""),
        }
        rows.append(row)
        print(
            f"[{log_prefix}] "
            f"{start_ms:g}-{end_ms:g} ms "
            f"PC1={row['pc1_ratio']:.4f} "
            f"PC1-3={row['pc123_ratio']:.4f}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PCA for cumulative 25 ms windows of best liquid states."
    )
    parser.add_argument(
        "--internal-state-dir",
        type=Path,
        default=None,
        help="Internal states from run_best_params_waveforms.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--total-ms", type=float, default=DEFAULT_TOTAL_MS)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--internal-state-bin-ms", type=float, default=1.0)
    parser.add_argument(
        "--no-auto-run-liquid",
        action="store_true",
        help="Do not create missing best internal_states automatically.",
    )
    args = parser.parse_args()

    internal_state_dir = get_or_make_internal_state_dir(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_pca_windows(
        internal_state_dir=internal_state_dir,
        out_dir=out_dir,
        windows=make_cumulative_windows(args.total_ms, args.window_ms),
        samples_per_class=int(args.samples_per_class),
        components=int(args.components),
        log_prefix="cumulative-pca",
        dir_prefix="pca_window",
    )

    csv_path = out_dir / "cumulative_pca_window_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "internal_state_dir": str(internal_state_dir),
        "out_dir": str(out_dir),
        "window_ms": float(args.window_ms),
        "total_ms": float(args.total_ms),
        "samples_per_class": int(args.samples_per_class),
        "components": int(args.components),
        "rows": rows,
        "summary_csv": str(csv_path),
    }
    (out_dir / "cumulative_pca_window_summary.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_summary_plot(
        rows,
        out_dir,
        x_key="window_end_ms",
        x_label="Used duration from 0 ms [ms]",
        out_name="cumulative_pca_explained_ratio_trend.png",
    )

    step_out_dir = out_dir / "pca_25ms_only_windows"
    step_out_dir.mkdir(parents=True, exist_ok=True)
    step_rows = run_pca_windows(
        internal_state_dir=internal_state_dir,
        out_dir=step_out_dir,
        windows=make_step_windows(args.total_ms, args.window_ms),
        samples_per_class=int(args.samples_per_class),
        components=int(args.components),
        log_prefix="25ms-pca",
        dir_prefix="pca_25ms",
    )
    step_csv_path = step_out_dir / "pca_25ms_only_window_summary.csv"
    with step_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)
    step_payload = {
        "internal_state_dir": str(internal_state_dir),
        "out_dir": str(step_out_dir),
        "window_ms": float(args.window_ms),
        "total_ms": float(args.total_ms),
        "samples_per_class": int(args.samples_per_class),
        "components": int(args.components),
        "rows": step_rows,
        "summary_csv": str(step_csv_path),
    }
    (step_out_dir / "pca_25ms_only_window_summary.json").write_text(
        json.dumps(jsonable(step_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_summary_plot(
        step_rows,
        step_out_dir,
        x_key="window_center_ms",
        x_label="25 ms window center [ms]",
        out_name="pca_25ms_only_explained_ratio_trend.png",
    )
    print(f"[cumulative-pca] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
