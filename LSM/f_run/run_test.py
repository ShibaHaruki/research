"""保存済み学習重み、またはoffの固定重みでテストする実行入口。"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

# ファイルを直接実行したときのプロジェクト内importの基準。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from f_run.common import build_cfg, load_cfg, repetition_numbers
from d_tools.parameter_recording import has_training_configuration
from f_run.testing import run_test
from a_experiment.paths import training_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--rep", type=int, default=None)
    args = parser.parse_args()

    source = args.train_dir if args.train_dir is not None else training_directory(build_cfg())
    if (
        not has_training_configuration(source)
        and has_training_configuration(source / f"rep{args.rep or 1}" / "training")
    ):
        source = source / f"rep{args.rep or 1}" / "training"
    if not has_training_configuration(source) and has_training_configuration(source / "training"):
        source = source / "training"
    rep_config = source / f"rep{args.rep or 1}" / "config_snapshot.json"
    cfg = (
        load_cfg(source)
        if has_training_configuration(source)
        else load_cfg(rep_config.parent) if rep_config.exists() else deepcopy(build_cfg())
    )
    for rep in repetition_numbers(cfg, args.rep):
        print(run_test(rep, args.train_dir))


if __name__ == "__main__":
    main()
