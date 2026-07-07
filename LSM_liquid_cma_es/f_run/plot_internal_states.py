"""保存済み内部状態をまとめて読み、素材ごとの可視化画像を作る実行スクリプト。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run
from d_tools.internal_state_visualization import save_internal_state_overviews
from d_tools.run_paths import jsonable


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
LIQUID_RESULT_DIR = RESULTS_PATH / RUN_CFG["LIQUID_RESULT_DIR"]
INTERNAL_STATE_DIR_NAME = str(RUN_CFG.get("INTERNAL_STATE_DIR", "internal_states"))


def discover_internal_state_entries(root: Path) -> list[dict]:
    # internal_states を持つ liquid 実行結果を探し、可視化対象として列挙する。
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
                "output_dir": run_dir / "internal_state_plots",
                "mtime": internal_dir.stat().st_mtime,
            }
        )
    return sorted(entries, key=lambda item: item["mtime"], reverse=True)


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


def run_entry(
    entry: dict,
    *,
    first_sample_only: bool,
    max_samples_per_class: int | None,
    max_neurons: int,
    sort_by_activity: bool,
) -> list[dict]:
    # 1つの internal_states ディレクトリを読み、各素材の内部状態 overview を保存する。
    run_dir = Path(entry["run_dir"]).resolve()
    internal_state_dir = Path(entry["internal_state_dir"]).resolve()
    out_dir = Path(entry["output_dir"]).resolve()
    params = {
        "dataset_id": entry["dataset_id"],
        "run_dir": str(run_dir),
        "internal_state_dir": str(internal_state_dir),
        "out_dir": str(out_dir),
        "first_sample_only": first_sample_only,
        "max_samples_per_class": max_samples_per_class,
        "max_neurons": max_neurons,
        "sort_by_activity": sort_by_activity,
        "run_metadata": _load_run_metadata(run_dir),
    }
    _write_used_parameters(out_dir, params)
    results = save_internal_state_overviews(
        internal_state_dir,
        out_dir,
        first_sample_only=first_sample_only,
        max_samples_per_class=max_samples_per_class,
        max_neurons=max_neurons,
        sort_by_activity=sort_by_activity,
    )
    print(f"[saved] {len(results)} internal-state visualizations to {out_dir}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize saved run_liquid internal states."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=LIQUID_RESULT_DIR,
        help="Root directory to search. Default: g_tactile_results/liquid_run",
    )
    parser.add_argument(
        "--internal-state-dir",
        type=Path,
        default=None,
        help="Use a specific internal_states directory instead of auto-discovery.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Use a specific liquid run directory containing internal_states.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process every discovered liquid run. Default processes only the latest.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Plot all samples instead of only the first sample of each material.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Limit samples per material when --all-samples is used.",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=200,
        help="Maximum neurons shown in each heatmap.",
    )
    parser.add_argument(
        "--no-sort-by-activity",
        action="store_true",
        help="Show neurons in index order instead of sorting by activity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 明示パスがなければ root から探し、デフォルトでは最新の1件だけ処理する。
    if args.internal_state_dir is not None:
        internal_state_dir = Path(args.internal_state_dir).resolve()
        run_dir = internal_state_dir.parent
        entries = [
            {
                "dataset_id": run_dir.name,
                "run_dir": run_dir,
                "internal_state_dir": internal_state_dir,
                "output_dir": run_dir / "internal_state_plots",
            }
        ]
    elif args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        entries = [
            {
                "dataset_id": run_dir.name,
                "run_dir": run_dir,
                "internal_state_dir": run_dir / INTERNAL_STATE_DIR_NAME,
                "output_dir": run_dir / "internal_state_plots",
            }
        ]
    else:
        entries = discover_internal_state_entries(args.root)
        if not args.all_runs:
            entries = entries[:1]

    if not entries:
        print(f"No internal state directories found under {Path(args.root).resolve()}")
        return

    for entry in entries:
        print(f"[run] {entry['dataset_id']}")
        run_entry(
            entry,
            first_sample_only=not args.all_samples,
            max_samples_per_class=args.max_samples_per_class,
            max_neurons=args.max_neurons,
            sort_by_activity=not args.no_sort_by_activity,
        )


if __name__ == "__main__":
    main()
