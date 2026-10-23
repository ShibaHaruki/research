"""出力層の学習と、学習済み重みを使うテストを順に実行する。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from f_run.run_training import run_training
from f_run.run_test import run_test
from f_run.run_common import add_training_arguments, cfg_from_args, training_directory


def run_train_test(rep: int,cfg: dict) -> str:
    training_message=run_training(rep,cfg)
    test_message=run_test(rep,training_directory(cfg))
    return training_message+' | '+test_message


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_training_arguments(parser)
    args=parser.parse_args();cfg=cfg_from_args(args)
    from f_run.run_common import repetition_numbers
    for rep in repetition_numbers(cfg,args.rep):
        print(run_train_test(rep,cfg))

if __name__=='__main__': main()
