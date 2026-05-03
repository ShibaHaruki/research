"""training、test、classification 評価までをまとめて実行する入口。"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from c_configs.FIXED import cfg_run
from d_tools.compat import repeat_count
from f_run.run_test import test_output_dir_for_training_dir, training_output_dir_for_cfg
from f_run.run_test_classification import classify_dataset
from f_run.run_train_test import run_train_test
from f_run.run_training import _first_value, build_cfg


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})


def _rep_numbers(cfg: dict, rep_start: int | None = None, rep_end: int | None = None) -> list[int]:
    total_reps = repeat_count(cfg["common"], 1)
    start = 1 if rep_start is None else max(1, int(rep_start))
    end = total_reps if rep_end is None else min(total_reps, int(rep_end))
    if end < start:
        return []
    return list(range(start, end + 1))


def _tn_ms_values(run_cfg: dict, override: list[int] | None) -> list[int]:
    if override:
        return [int(value) for value in override]

    value = run_cfg.get("EVAL_TN_MS", [25])
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _evaluate_results(
    cfg: dict,
    *,
    reps: list[int],
    tn_ms_list: list[int] | None = None,
    n_folds: int | None = None,
    base_seed: int | None = None,
) -> str:
    # test で保存された sout_rec を読み、10-fold 評価まで続けて実行する。
    run_cfg = cfg["run"]
    tn_ms_values = _tn_ms_values(run_cfg, tn_ms_list)
    n_folds_value = int(n_folds if n_folds is not None else run_cfg.get("EVAL_N_FOLDS", 10))
    base_seed_value = int(base_seed if base_seed is not None else run_cfg.get("EVAL_BASE_SEED", 1))

    train_dir = training_output_dir_for_cfg(cfg)
    test_dir = test_output_dir_for_training_dir(train_dir)
    summary_rows = classify_dataset(
        test_dir,
        rep_start=min(reps),
        rep_end=max(reps),
        n_folds=n_folds_value,
        T_n_list=tn_ms_values,
        base_seed=base_seed_value,
    )
    return f"[eval] saved {len(summary_rows)} rows under {test_dir / 'results_10fold'}"


def run_train_test_eval(
    cfg: dict,
    *,
    tn_ms_list: list[int] | None = None,
    n_folds: int | None = None,
    base_seed: int | None = None,
    rep_start: int | None = None,
    rep_end: int | None = None,
) -> str:
    # training -> test -> classification を一括で流す高レベル入口。
    reps = _rep_numbers(cfg, rep_start=rep_start, rep_end=rep_end)
    if not reps:
        raise ValueError("No reps selected for train_test_eval.")

    messages = []
    for rep in reps:
        messages.append(run_train_test(rep, cfg))
    messages.append(
        _evaluate_results(
            cfg,
            reps=reps,
            tn_ms_list=tn_ms_list,
            n_folds=n_folds,
            base_seed=base_seed,
        )
    )
    return " | ".join(messages)


def _run_train_test_rep_worker(rep: int) -> str:
    cfg = build_cfg()
    return run_train_test(rep, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training, test, and 10-fold evaluation in one command."
    )
    parser.add_argument("--rep-start", type=int, default=None)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument(
        "--tn-ms",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation aggregation window(s) in ms. Default: cfg_run.EVAL_TN_MS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 引数で rep 範囲や T_n を指定できるようにして、まとめ実行を開始する。
    cfg = build_cfg()
    reps = _rep_numbers(cfg, rep_start=args.rep_start, rep_end=args.rep_end)
    if not reps:
        raise ValueError("No reps selected.")

    no_mp_env = str(RUN_CFG.get("NO_MP_ENV", "LSM_NO_MP"))
    if len(reps) == 1 or os.environ.get(no_mp_env) == "1":
        print(
            run_train_test_eval(
                cfg,
                tn_ms_list=args.tn_ms,
                n_folds=args.folds,
                base_seed=args.base_seed,
                rep_start=args.rep_start,
                rep_end=args.rep_end,
            )
        )
        return 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(os.cpu_count() or 1, len(reps)),
        mp_context=ctx,
    ) as executor:
        futures = [executor.submit(_run_train_test_rep_worker, rep) for rep in reps]
        for future in as_completed(futures):
            print(future.result())

    print(
        _evaluate_results(
            cfg,
            reps=reps,
            tn_ms_list=args.tn_ms,
            n_folds=args.folds,
            base_seed=args.base_seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
