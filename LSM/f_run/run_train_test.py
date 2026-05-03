"""1つの rep で training と test を続けて実行する簡易入口。"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp

from c_configs.FIXED import cfg_run
from d_tools.compat import repeat_count
from f_run.run_test import run_test, training_output_dir_for_cfg
from f_run.run_training import _first_value, build_cfg, run_training


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})


def run_train_test(rep: int, cfg: dict) -> str:
    # 1 rep について training を先に実行し、その学習済み重みを使って test を続けて実行する。
    train_dir = training_output_dir_for_cfg(cfg)
    training_message = run_training(rep, cfg)
    test_message = run_test(rep, train_dir)
    return f"{training_message} | {test_message}"


def run_train_test_worker(rep: int) -> str:
    cfg = build_cfg()
    return run_train_test(rep, cfg)


if __name__ == "__main__":
    cfg = build_cfg()
    # rep が複数ある場合は、rep ごとに training+test を並列実行する。
    reps = list(range(1, repeat_count(cfg["common"], 1) + 1))
    if len(reps) == 1 or os.environ.get(RUN_CFG["NO_MP_ENV"]) == "1":
        for rep in reps:
            print(run_train_test_worker(rep))
        raise SystemExit(0)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(os.cpu_count() or 1, len(reps)),
        mp_context=ctx,
    ) as executor:
        futures = [executor.submit(run_train_test_worker, rep) for rep in reps]
        for future in as_completed(futures):
            print(future.result())
