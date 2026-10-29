"""評価条件の検証と、通常・共有Liquidのテスト結果の分類評価。"""

from __future__ import annotations
from pathlib import Path
from a_experiment.experiment_sets import learning_rule_name
from f_run.train_test import run_train_test
from a_experiment.paths import (
    training_directory,
    test_directory,
    off_test_directory,
    evaluation_directory,
)
from d_tools.evaluation import classify_dataset


def evaluation_settings(cfg: dict, *, tn_ms_list=None, n_folds=None, base_seed=None):
    """現在の記録幅に適用できる評価窓・分割数・seedを検証して返す。"""
    tn_ms_list = cfg["eval"].get("EVAL_TN_MS", [25]) if tn_ms_list is None else tn_ms_list
    if not tn_ms_list:
        raise ValueError("At least one evaluation window is required")
    n_folds = n_folds if n_folds is not None else cfg["eval"].get("EVAL_N_FOLDS", 10)
    base_seed = base_seed if base_seed is not None else cfg["eval"].get("EVAL_BASE_SEED", 1)
    from f_run.common import validate_cfg

    _, _, n_test, _, bin_ms, duration = validate_cfg(cfg)
    if n_folds < 2 or n_test < n_folds or n_test - (n_test + n_folds - 1) // n_folds < 2:
        raise ValueError(
            "Need enough test trials for >=2 classifier-training trials per class/fold"
        )
    import numpy as np

    for window in tn_ms_list:
        if (
            not np.isfinite(window)
            or window <= 0
            or not np.isclose(window / bin_ms, round(window / bin_ms))
            or not np.isclose(duration / window, round(duration / window))
        ):
            raise ValueError(
                "Evaluation window must be a multiple of bin width and divide trial duration"
            )
    return list(tn_ms_list), n_folds, base_seed


def evaluate_sweep_candidates(
    rep, configurations, *, resume=False, tn_ms_list=None, n_folds=None, base_seed=None
):
    """各τ・seedの固定重みテスト記録を分類し、条件ごとに正答率を保存する。"""
    from d_tools.evaluation_excel import load_classification_results

    results = {}
    for cfg in configurations:
        windows, folds, seed = evaluation_settings(
            cfg, tn_ms_list=tn_ms_list, n_folds=n_folds, base_seed=base_seed
        )
        folder = training_directory(cfg)
        output = test_directory(folder)
        previous = evaluation_directory(output, cfg=cfg) / "classification.xlsx"
        metrics = None
        if resume and previous.exists():
            saved = load_classification_results(previous)
            if (
                saved is not None
                and saved["folds"] == folds
                and saved["base_seed"] == seed
                and saved["T_n_ms"] == windows
                and saved["source"] == str(Path(output).resolve())
                and saved["materials"] == list(cfg["test"]["TEST_MAT"])
            ):
                selected = [row for row in saved["summary"] if row["rep"] == rep]
                if len(selected) == len(windows):
                    metrics = selected
        if metrics is None:
            metrics = classify_dataset(
                output,
                rep_start=rep,
                rep_end=rep,
                n_folds=folds,
                T_n_list=windows,
                base_seed=seed,
                thread_limit=1,
            )
        results[str(folder)] = metrics
    return results


def run_train_test_eval(
    cfg: dict,
    *,
    tn_ms_list=None,
    n_folds=None,
    base_seed=None,
    rep_start=None,
    rep_end=None,
    workers=1,
) -> str:
    """各repのテストと評価を完結させ、workers指定時はrepごとに並列実行する。"""
    from f_run.common import repetition_numbers

    reps = repetition_numbers(cfg, rep_start, rep_end)
    start, end = reps[0], reps[-1]
    tn_ms_list, n_folds, base_seed = evaluation_settings(
        cfg, tn_ms_list=tn_ms_list, n_folds=n_folds, base_seed=base_seed
    )
    if workers != 1:
        from f_run.repetitions import run_repetitions

        return " | ".join(
            run_repetitions(
                cfg,
                stage="train_test_eval",
                workers=workers,
                rep=start,
                rep_end=end,
                evaluation_options={
                    "tn_ms_list": tn_ms_list,
                    "n_folds": n_folds,
                    "base_seed": base_seed,
                },
            )
        )
    messages = [run_train_test(rep, cfg) for rep in range(start, end + 1)]
    if learning_rule_name(cfg) == "off":
        rows = []
        for rep in reps:
            rows.extend(
                classify_dataset(
                    off_test_directory(cfg, rep),
                    rep_start=rep,
                    rep_end=rep,
                    n_folds=n_folds,
                    T_n_list=tn_ms_list,
                    base_seed=base_seed,
                )
            )
        return " | ".join(messages) + f" | [off classification] {len(rows)} rep summaries saved"
    rows = []
    for rep in reps:
        rows.extend(
            classify_dataset(
                test_directory(training_directory(cfg, rep)),
                rep_start=rep,
                rep_end=rep,
                n_folds=n_folds,
                T_n_list=tn_ms_list,
                base_seed=base_seed,
            )
        )
    return " | ".join(messages) + f" | [classification] {len(rows)} summary rows saved"
