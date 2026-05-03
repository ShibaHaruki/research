"""run_liquid が保存した内部状態に VAE を学習し、潜在表現と評価指標を保存する入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run
from d_tools.internal_state_vae import output_dir_name, train_internal_state_vae


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
LIQUID_RESULT_DIR = RESULTS_PATH / RUN_CFG["LIQUID_RESULT_DIR"]
INTERNAL_STATE_DIR_NAME = str(RUN_CFG.get("INTERNAL_STATE_DIR", "internal_states"))
VAE_DIR_NAME = str(RUN_CFG.get("INTERNAL_STATE_VAE_DIR", "internal_state_vae"))


def _latest_internal_state_mtime(internal_state_dir: Path) -> float:
    mtimes = [
        fp.stat().st_mtime
        for fp in Path(internal_state_dir).glob("*/*_liquid_internal_state_all.npz")
    ]
    return max(mtimes) if mtimes else 0.0


def discover_liquid_internal_state_entries(root: Path) -> list[dict]:
    # liquid_run 配下から internal_states フォルダを探し、VAEにかけられる実行結果を列挙する。
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
                "mtime": _latest_internal_state_mtime(internal_dir),
            }
        )
    return entries


def _candidate_entries(paths: list[Path]) -> list[dict]:
    # 入力パスが run_dir / internal_states / 親フォルダのどれでも扱えるように正規化する。
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
                    "mtime": _latest_internal_state_mtime(path),
                }
            )
        elif (path / INTERNAL_STATE_DIR_NAME).exists():
            run_dir = path
            internal_dir = path / INTERNAL_STATE_DIR_NAME
            entries.append(
                {
                    "dataset_id": run_dir.name,
                    "run_dir": run_dir,
                    "internal_state_dir": internal_dir,
                    "mtime": _latest_internal_state_mtime(internal_dir),
                }
            )
        else:
            entries.extend(discover_liquid_internal_state_entries(path))

    unique = {}
    for entry in entries:
        key = str(Path(entry["internal_state_dir"]).resolve())
        unique[key] = entry
    return sorted(unique.values(), key=lambda item: (float(item["mtime"]), str(item["dataset_id"])))


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


def run_entry(entry: dict, args: argparse.Namespace) -> dict:
    # 1つの liquid 実行結果に対して、内部状態VAEを学習して結果を保存する。
    run_dir = Path(entry["run_dir"]).resolve()
    internal_state_dir = Path(entry["internal_state_dir"]).resolve()
    out_dir = (
        run_dir
        / VAE_DIR_NAME
        / output_dir_name(
            window_ms=float(args.window_ms),
            step_ms=float(args.step_ms),
            latent_dim=int(args.latent_dim),
            beta=float(args.beta),
        )
    )
    print("=" * 80)
    print(f"[VAE] dataset={entry['dataset_id']}")
    print(f"[VAE] internal_state_dir={internal_state_dir}")
    print(f"[VAE] out_dir={out_dir}")
    print("=" * 80)

    result = train_internal_state_vae(
        internal_state_dir,
        out_dir,
        run_dir=run_dir,
        dataset_id=str(entry["dataset_id"]),
        window_ms=float(args.window_ms),
        step_ms=float(args.step_ms),
        latent_dim=int(args.latent_dim),
        hidden_channels=int(args.hidden_channels),
        beta=float(args.beta),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=str(args.device),
        standardize=not bool(args.no_standardize),
        max_samples_per_class=args.max_samples_per_class,
    )
    result["run_metadata"] = _load_run_metadata(run_dir)
    print(
        f"[saved] silhouette={result['silhouette']:.6g} DR={result['DR']:.6g} "
        f"shape={result['input_shape_batch_N_K']}"
    )
    print(f"[saved] {result['out_dir']}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 1D-CNN VAE on run_liquid internal states. "
            "Input is time-binned to batch x N x K, z uses encoder mu, "
            "then Silhouette and DR are computed on z."
        )
    )
    parser.add_argument(
        "--liquid-dir",
        action="append",
        default=None,
        help=(
            "liquid run directory, internal_states directory, or parent directory. "
            "Can be repeated. If omitted, use the latest internal_states under liquid_run."
        ),
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all discovered liquid internal_state runs instead of only the latest one.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=float(RUN_CFG.get("INTERNAL_STATE_VAE_WINDOW_MS", 10.0)),
        help="Time-bin width in ms. Default is non-overlapping 10 ms bins.",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=float(RUN_CFG.get("INTERNAL_STATE_VAE_STEP_MS", 10.0)),
        help="Time-bin step in ms. Keep equal to --window-ms to avoid overlap.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=int(RUN_CFG.get("INTERNAL_STATE_VAE_LATENT_DIM", 2)),
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=int(RUN_CFG.get("INTERNAL_STATE_VAE_HIDDEN_CHANNELS", 64)),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=float(RUN_CFG.get("INTERNAL_STATE_VAE_BETA", 1e-3)),
        help="KL loss weight.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(RUN_CFG.get("INTERNAL_STATE_VAE_EPOCHS", 100)),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(RUN_CFG.get("INTERNAL_STATE_VAE_BATCH_SIZE", 32)),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(RUN_CFG.get("INTERNAL_STATE_VAE_LR", 1e-3)),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(RUN_CFG.get("INTERNAL_STATE_VAE_SEED", 0)),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=str(RUN_CFG.get("INTERNAL_STATE_VAE_DEVICE", "auto")),
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=RUN_CFG.get("INTERNAL_STATE_VAE_MAX_SAMPLES_PER_CLASS", None),
        help="Limit samples per material for quick tests.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        default=not bool(RUN_CFG.get("INTERNAL_STATE_VAE_STANDARDIZE", True)),
        help="Disable per-neuron standardization before VAE training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in args.liquid_dir] if args.liquid_dir else [LIQUID_RESULT_DIR]
    entries = _candidate_entries(paths)
    if not entries:
        raise FileNotFoundError(f"No liquid internal state files found under: {paths}")

    # パス指定がなければ最新の1件だけを処理する。--all-runs なら見つかった全件を処理する。
    selected = entries if args.all_runs or args.liquid_dir else [entries[-1]]
    rows = []
    for entry in selected:
        try:
            rows.append(run_entry(entry, args))
        except RuntimeError as exc:
            if "PyTorch is required" in str(exc):
                print(f"[error] {exc}", file=sys.stderr)
                return 1
            raise

    if rows:
        summary_dir = LIQUID_RESULT_DIR / f"{VAE_DIR_NAME}_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_dir / "internal_state_vae_summary.csv"
        flat_rows = [
            {
                key: value
                for key, value in row.items()
                if key != "run_metadata"
            }
            for row in rows
        ]
        pd.DataFrame(flat_rows).to_csv(summary_csv, index=False)
        print(f"[saved] {summary_csv}")

    print("All finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
