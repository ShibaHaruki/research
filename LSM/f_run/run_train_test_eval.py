"""出力層の学習→固定重みでテスト→素材分類をまとめて実行する。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from f_run.run_train_test import run_train_test
from f_run.run_common import add_training_arguments, cfg_from_args, training_directory, test_directory
from d_tools.evaluation import classify_dataset


def run_train_test_eval(cfg: dict, *, tn_ms_list=None,n_folds=None,base_seed=None,rep_start=None,rep_end=None) -> str:
    from f_run.run_common import repetition_numbers
    reps=repetition_numbers(cfg,rep_start,rep_end)
    start,end=reps[0],reps[-1]
    tn_ms_list=tn_ms_list or cfg['run'].get('EVAL_TN_MS',[25])
    n_folds=n_folds if n_folds is not None else cfg['run'].get('EVAL_N_FOLDS',10)
    base_seed=base_seed if base_seed is not None else cfg['run'].get('EVAL_BASE_SEED',1)
    from f_run.run_common import validate_cfg
    _,_,n_test,_,bin_ms,duration=validate_cfg(cfg)
    if n_folds<2 or n_test<n_folds or n_test-(n_test+n_folds-1)//n_folds<2:
        raise ValueError('Need enough test trials for >=2 classifier-training trials per class/fold')
    import numpy as np
    for window in tn_ms_list:
        if window <= 0 or not np.isclose(window/bin_ms,round(window/bin_ms)) or not np.isclose(duration/window,round(duration/window)):
            raise ValueError('Evaluation window must be a multiple of bin width and divide trial duration')
    messages=[run_train_test(rep,cfg) for rep in range(start,end+1)]
    rows=classify_dataset(test_directory(training_directory(cfg)),rep_start=start,rep_end=end,
        n_folds=n_folds,T_n_list=tn_ms_list,base_seed=base_seed)
    return ' | '.join(messages)+f' | [classification] {len(rows)} summary rows saved'


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_training_arguments(parser)
    parser.add_argument('--rep-end',type=int,default=None)
    parser.add_argument('--folds',type=int,default=None);parser.add_argument('--base-seed',type=int,default=None)
    parser.add_argument('--tn-ms',nargs='+',type=int,default=None)
    args=parser.parse_args()
    print(run_train_test_eval(cfg_from_args(args),tn_ms_list=args.tn_ms,n_folds=args.folds,
        base_seed=args.base_seed,rep_start=args.rep,rep_end=args.rep_end))

if __name__=='__main__': main()
