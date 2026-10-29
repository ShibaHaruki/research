"""出力層なしでLiquidのdebug・テスト記録・分類評価を実行する入口。"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# ファイルを直接実行してもLSM内のモジュールを読み込めるようにする。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from f_run.common import build_cfg
from f_run.repetitions import run_repetitions


def main():
    """既定ではoffと同じ100試行で、Liquid Eだけの分類まで行う。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--trial-ids", nargs="+", type=int, default=None)
    parser.add_argument("--materials", nargs="+", default=None)
    parser.add_argument("--state-policy", choices=["carry", "reset_each_trial"], default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="rep番号フォルダを置くルート")
    parser.add_argument("--population", choices=["E", "all"], default="E")
    parser.add_argument("--record-only", action="store_true", help="分類せずdebugと記録だけを保存")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()
    if args.samples is not None and args.trial_ids is not None:
        parser.error("Choose --samples or --trial-ids")
    cfg = build_cfg(network_preset="off")
    if args.samples is not None or args.trial_ids is not None:
        cfg["test"]["NUM_TEST_SAMPLE"] = [
            args.samples if args.samples is not None else len(args.trial_ids)
        ]
    if args.materials is not None:
        for section, key in [
            ("training", "TRAINING_MAT"),
            ("test", "TEST_MAT"),
            ("liquid", "LIQUID_MAT"),
        ]:
            cfg[section][key] = list(args.materials)
    if args.state_policy is not None:
        cfg["common"]["LIQUID_STATE_POLICY"] = args.state_policy
    if args.out_dir is not None:
        cfg["run"]["LIQUID_OUTPUT_DIR"] = str(args.out_dir.resolve())
    if args.folds is not None:
        cfg["eval"]["EVAL_N_FOLDS"] = args.folds
    options = dict(
        resume=args.resume,
        record_only=args.record_only,
        population=args.population,
        trial_ids=args.trial_ids,
    )
    for result in run_repetitions(
        cfg,
        stage="liquid_test_eval",
        workers=args.workers,
        rep=args.rep,
        rep_end=args.rep_end,
        evaluation_options=options,
    ):
        print(result)


if __name__ == "__main__":
    main()
