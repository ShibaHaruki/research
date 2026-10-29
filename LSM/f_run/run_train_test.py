"""現在の設定で学習とテストを続けて行う実行入口。"""

import argparse
from pathlib import Path
import sys

# ファイルを直接実行したときのプロジェクト内importの基準。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from f_run.common import add_training_arguments, cfg_from_args
from f_run.sweep import add_sweep_arguments, handle_sweep_arguments
from f_run.repetitions import run_repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    add_sweep_arguments(parser)
    args = parser.parse_args()
    if handle_sweep_arguments(args, stage="train_test"):
        return
    cfg = cfg_from_args(args)

    for message in run_repetitions(
        cfg, stage="train_test", workers=args.workers, rep=args.rep, rep_end=args.rep_end
    ):
        print(message)


if __name__ == "__main__":
    main()
