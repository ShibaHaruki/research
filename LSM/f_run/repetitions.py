"""repを単位として、学習・テスト・評価を独立プロセスで実行する。"""

from f_run.common import repetition_numbers


def _run_repetition(stage, cfg, rep, evaluation_options):
    """子プロセスで1つのrepを完結させ、Brian2の状態を別repと共有しない。"""
    import os

    os.environ["LSM_PROGRESS_REP"] = str(rep)
    if stage == "liquid_test_eval":
        from f_run.liquid import run_liquid_test_eval

        return run_liquid_test_eval(rep, cfg, **evaluation_options)
    if stage == "training":
        from f_run.training import run_training

        return run_training(rep, cfg)
    if stage == "train_test":
        from f_run.train_test import run_train_test

        return run_train_test(rep, cfg)
    if stage == "train_test_eval":
        from f_run.evaluation import run_train_test_eval

        return run_train_test_eval(cfg, rep_start=rep, rep_end=rep, workers=1, **evaluation_options)
    raise ValueError(f"Unknown repetition stage: {stage}")


def run_repetitions(cfg, *, stage, workers=1, rep=None, rep_end=None, evaluation_options=None):
    """rep単位で実行する。workersは同時に動かすrep数の上限。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    if stage not in ("training", "train_test", "train_test_eval", "liquid_test_eval"):
        raise ValueError(f"Unknown repetition stage: {stage}")
    reps = repetition_numbers(cfg, rep, rep_end)
    options = dict(evaluation_options or {})
    worker_count = min(int(workers), len(reps))
    if worker_count == 1:
        return [_run_repetition(stage, cfg, number, options) for number in reps]
    # Windowsでもプロセスごとにネットワークを独立して生成する。
    results = {}
    from d_tools.progress import initialize_progress_lock

    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initialize_progress_lock,
        initargs=(context.RLock(),),
    ) as pool:
        futures = {
            pool.submit(_run_repetition, stage, cfg, number, options): number for number in reps
        }
        for future in as_completed(futures):
            number = futures[future]
            results[number] = future.result()
            print(f"[rep{number} complete]", flush=True)
    return [results[number] for number in reps]
