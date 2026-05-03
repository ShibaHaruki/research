"""保存済み内部状態から DR、Sb、Sw、分離特性などを計算する入口。"""

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
from d_tools.run_paths import jsonable
from d_tools.separation_metrics import (
    linear_separation_property,
    load_internal_state_dataset,
    pairwise_separation_matrix,
    pairwise_trajectory_separation_matrix,
    save_pairwise_matrix_csv,
    save_scatter_matrices_npz,
    scatter_metrics,
)


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
LIQUID_RESULT_DIR = RESULTS_PATH / RUN_CFG["LIQUID_RESULT_DIR"]
INTERNAL_STATE_DIR_NAME = str(RUN_CFG.get("INTERNAL_STATE_DIR", "internal_states"))


def discover_liquid_internal_state_entries(root: Path) -> list[dict]:
    # liquid_run 配下から internal_states を探し、分離指標を計算できる run を列挙する。
    root = Path(root).resolve()
    entries = []
    for internal_dir in sorted(root.rglob(INTERNAL_STATE_DIR_NAME)):
        if not internal_dir.is_dir():
            continue
        if not list(internal_dir.glob("*/*_liquid_internal_state_all.npz")):
            continue
        run_dir = internal_dir.parent
        try:
            dataset_id = run_dir.relative_to(LIQUID_RESULT_DIR.resolve()).as_posix()
        except ValueError:
            dataset_id = run_dir.name
        entries.append(
            {
                "dataset_id": dataset_id,
                "run_dir": run_dir,
                "internal_state_dir": internal_dir,
                "output_dir": run_dir / "separation_metrics",
            }
        )
    return entries


def _write_used_parameters(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "used_parameters.txt"
    out_fp.write_text(
        "Used Parameters\n"
        "===============\n\n"
        + json.dumps(jsonable(payload), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return out_fp


def _load_run_metadata(run_dir: Path) -> dict:
    metadata = {}
    for name in ("config_snapshot.json", "experiment_trial.json"):
        fp = Path(run_dir) / name
        if fp.exists():
            try:
                metadata[name] = json.loads(fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata[name] = {"error": f"failed to parse {fp}"}
    return metadata


def run_entry(
    entry: dict,
    *,
    feature_mode: str,
    pairwise_mode: str,
    max_samples_per_class: int | None = None,
    max_pairs_per_class_pair: int | None = None,
    rank_tol: float | None = None,
    save_matrices: bool = False,
    batch_size: int = 256,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> list[dict]:
    # 1つの内部状態ディレクトリについて、指定された指標を計算して CSV/JSON/図へ保存する。
    dataset_id = str(entry["dataset_id"])
    run_dir = Path(entry["run_dir"]).resolve()
    internal_state_dir = Path(entry["internal_state_dir"]).resolve()
    out_dir = Path(entry["output_dir"]) / f"feature_{feature_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_trajectories = pairwise_mode == "trajectory"
    dataset = load_internal_state_dataset(
        internal_state_dir,
        feature_mode=feature_mode,
        max_samples_per_class=max_samples_per_class,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        load_trajectories=load_trajectories,
    )
    materials = list(dataset["materials"])
    features_by_class = dataset["features_by_class"]
    trajectories_by_class = dataset["trajectories_by_class"]

    params = {
        "dataset_id": dataset_id,
        "run_dir": str(run_dir),
        "internal_state_dir": str(internal_state_dir),
        "feature_mode": feature_mode,
        "pairwise_mode": pairwise_mode,
        "max_samples_per_class": max_samples_per_class,
        "max_pairs_per_class_pair": max_pairs_per_class_pair,
        "rank_tol": rank_tol,
        "save_matrices": save_matrices,
        "batch_size": batch_size,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "materials": materials,
        "files_by_class": dataset["files_by_class"],
        "run_metadata": _load_run_metadata(run_dir),
    }
    _write_used_parameters(out_dir, params)

    scatter = scatter_metrics(features_by_class, return_matrices=save_matrices)
    linear = linear_separation_property(features_by_class, tol=rank_tol)
    if pairwise_mode == "trajectory":
        pairwise = pairwise_trajectory_separation_matrix(
            trajectories_by_class,
            max_pairs_per_class_pair=max_pairs_per_class_pair,
        )
    elif pairwise_mode == "feature":
        pairwise = pairwise_separation_matrix(features_by_class, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown pairwise_mode: {pairwise_mode}")

    stem = f"internal_state_{feature_mode}_{pairwise_mode}"
    pairwise_fp = save_pairwise_matrix_csv(
        out_dir / "pairwise_matrices",
        pairwise["pairwise_matrix"],
        stem=stem,
        labels=materials,
    )
    scatter_fp = ""
    if save_matrices:
        scatter_fp = str(save_scatter_matrices_npz(out_dir / "scatter_matrices", scatter, stem=stem))

    row = {
        "dataset_id": dataset_id,
        "run_dir": str(run_dir),
        "internal_state_dir": str(internal_state_dir),
        "feature_mode": feature_mode,
        "pairwise_mode": pairwise_mode,
        "DR": scatter["DR"],
        "trace_Sb": scatter["trace_Sb"],
        "trace_Sw": scatter["trace_Sw"],
        "SPlin": linear["SPlin"],
        "SPlin_normalized": linear["normalized_rank"],
        "SPpw_between_mean": pairwise["SPpw_between_mean"],
        "SPpw_within_mean": pairwise["SPpw_within_mean"],
        "n_classes": scatter["n_classes"],
        "n_samples_total": scatter["n_samples_total"],
        "n_features": scatter["n_features"],
        "class_counts": ";".join(str(int(v)) for v in scatter["class_counts"]),
        "materials": ";".join(materials),
        "pairwise_matrix_file": str(pairwise_fp),
        "scatter_matrix_file": scatter_fp,
    }
    summary_csv = out_dir / "separation_summary.csv"
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    print(
        f"[saved] {dataset_id} DR={row['DR']:.6g} "
        f"SPlin={row['SPlin']} SPpw={row['SPpw_between_mean']:.6g}"
    )
    print(f"[saved] {summary_csv}")
    return [row]


def _candidate_entries(paths: list[Path]) -> list[dict]:
    entries: list[dict] = []
    for path in paths:
        path = path.resolve()
        if path.name == INTERNAL_STATE_DIR_NAME:
            run_dir = path.parent
            entries.append(
                {
                    "dataset_id": run_dir.name,
                    "run_dir": run_dir,
                    "internal_state_dir": path,
                    "output_dir": run_dir / "separation_metrics",
                }
            )
        elif (path / INTERNAL_STATE_DIR_NAME).exists():
            run_dir = path
            entries.append(
                {
                    "dataset_id": run_dir.name,
                    "run_dir": run_dir,
                    "internal_state_dir": path / INTERNAL_STATE_DIR_NAME,
                    "output_dir": run_dir / "separation_metrics",
                }
            )
        else:
            entries.extend(discover_liquid_internal_state_entries(path))

    unique = {}
    for entry in entries:
        key = str(Path(entry["internal_state_dir"]).resolve())
        unique[key] = entry
    return sorted(unique.values(), key=lambda item: str(item["dataset_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DR, Sb/Sw traces, pairwise separation, and linear separation "
            "from run_liquid internal_states."
        )
    )
    parser.add_argument(
        "--liquid-dir",
        action="append",
        default=None,
        help=(
            "liquid run directory, internal_states directory, or parent directory. "
            "Can be repeated. If omitted, search under liquid_runs."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("final", "mean", "max", "sum", "flatten"),
        default="final",
        help="How to convert each internal-state trajectory to a feature vector.",
    )
    parser.add_argument(
        "--pairwise-mode",
        choices=("trajectory", "feature"),
        default="trajectory",
        help="SPpw from full trajectories or from selected feature vectors.",
    )
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument(
        "--max-pairs-per-class-pair",
        type=int,
        default=None,
        help="Limit sample pairs for trajectory SPpw when exact calculation is too slow.",
    )
    parser.add_argument("--window-start-ms", type=float, default=None)
    parser.add_argument("--window-end-ms", type=float, default=None)
    parser.add_argument("--rank-tol", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--save-matrices",
        action="store_true",
        help="Also save full Sb and Sw matrices. This can be large.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # --liquid-dir がなければ liquid_run 全体から対象を探す。
    paths = [Path(path) for path in args.liquid_dir] if args.liquid_dir else [LIQUID_RESULT_DIR]
    entries = _candidate_entries(paths)
    if not entries:
        raise FileNotFoundError(f"No liquid internal state files found under: {paths}")

    all_rows = []
    for entry in entries:
        all_rows.extend(
            run_entry(
                entry,
                feature_mode=str(args.feature_mode),
                pairwise_mode=str(args.pairwise_mode),
                max_samples_per_class=args.max_samples_per_class,
                max_pairs_per_class_pair=args.max_pairs_per_class_pair,
                rank_tol=args.rank_tol,
                save_matrices=bool(args.save_matrices),
                batch_size=int(args.batch_size),
                window_start_ms=args.window_start_ms,
                window_end_ms=args.window_end_ms,
            )
        )

    if all_rows:
        summary_dir = LIQUID_RESULT_DIR / "separation_metrics_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_dir / "separation_summary_all.csv"
        pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
        print(f"[saved] {summary_csv}")

    print("All finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
