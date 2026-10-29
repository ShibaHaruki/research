"""現在の設定でLiquid E→Readoutを学習する実行入口。"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

# repを並列実行するとき、各プロセス内の計算スレッドが増えすぎないようにする。
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_variable] = "1"
# f_runのファイルを直接実行した場合も、プロジェクト内モジュールを読み込めるようにする。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from a_experiment.experiment_sets import RUNTIME_BACKEND_SETTINGS

os.environ.setdefault("MPLBACKEND", RUNTIME_BACKEND_SETTINGS["MATPLOTLIB_BACKEND"])
from f_run.common import add_training_arguments, cfg_from_args
from f_run.sweep import add_sweep_arguments, handle_sweep_arguments
from f_run.repetitions import run_repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    add_sweep_arguments(parser)
    args = parser.parse_args()
    if handle_sweep_arguments(args, stage="training"):
        return
    cfg = cfg_from_args(args)
    for message in run_repetitions(
        cfg, stage="training", workers=args.workers, rep=args.rep, rep_end=args.rep_end
    ):
        print(message)


if __name__ == "__main__":
    main()
