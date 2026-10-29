"""現在の設定で学習・テスト・分類評価を行う実行入口。"""

import argparse
from pathlib import Path
import sys

# ファイルを直接実行したときのプロジェクト内importの基準。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from f_run.common import add_training_arguments, cfg_from_args
from f_run.sweep import add_sweep_arguments, handle_sweep_arguments
from f_run.evaluation import run_train_test_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    add_sweep_arguments(parser)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--tn-ms", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if handle_sweep_arguments(
        args,
        stage="train_test_eval",
        evaluation_options={
            "tn_ms_list": args.tn_ms,
            "n_folds": args.folds,
            "base_seed": args.base_seed,
        },
    ):
        return
    print(
        run_train_test_eval(
            cfg_from_args(args),
            tn_ms_list=args.tn_ms,
            n_folds=args.folds,
            base_seed=args.base_seed,
            rep_start=args.rep,
            rep_end=args.rep_end,
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
