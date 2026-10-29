"""保存済みLiquidまたは出力スパイクを時間窓で集約し、層別交差検証で素材分類する。"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from d_tools.evaluation import classify_dataset
from a_experiment.paths import evaluation_directory
from a_experiment.experiment_sets import EVAL_SET


def main():
    defaults = EVAL_SET["base"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--folds", type=int, default=defaults["EVAL_N_FOLDS"])
    parser.add_argument("--base-seed", type=int, default=defaults["EVAL_BASE_SEED"])
    parser.add_argument("--tn-ms", nargs="+", type=int, default=list(defaults["EVAL_TN_MS"]))
    args = parser.parse_args()
    rows = classify_dataset(
        args.test_dir,
        rep_start=args.rep_start,
        rep_end=args.rep_end,
        n_folds=args.folds,
        T_n_list=args.tn_ms,
        base_seed=args.base_seed,
    )
    print(
        f"[classification] saved {len(rows)} summary rows under {evaluation_directory(args.test_dir)}"
    )


if __name__ == "__main__":
    main()
