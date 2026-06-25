"""EXPERIMENT 定義を読み、複数 trial / rep を並列または順次に実行する入口。"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run
from d_tools.compat import normalize_target_name, repeat_count
from d_tools.experiments import (
    append_manifest_row,
    apply_overrides,
    experiment_run_id,
    expand_experiment,
    now_text,
    write_experiment_trial_json,
)
from d_tools.run_paths import (
    jsonable,
    make_run_output_dir,
    save_hierarchy_param_snapshots,
    save_used_parameters_text,
)
from f_run.run_liquid import LIQUID_RESULT_DIR, run_liquid
from f_run.run_common import build_cfg, build_network_cfg


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})


def _normalize_target(target: str) -> str:
    return str(normalize_target_name(target))


def _load_experiment_module(name_or_path: str):
    # EXPERIMENT 設定をモジュール名または .py ファイルパスから読み込む。
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import experiment file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    module_name = name_or_path
    if "." not in module_name:
        module_name = f"c_configs.EXPERIMENTS.{module_name}"
    return importlib.import_module(module_name)


def load_experiment(name_or_path: str) -> dict:
    module = _load_experiment_module(name_or_path)
    if not hasattr(module, "EXPERIMENT"):
        raise AttributeError(f"{module.__name__} must define EXPERIMENT.")
    experiment = dict(module.EXPERIMENT)
    experiment.setdefault("name", Path(name_or_path).stem)
    return experiment


def _num_repeats(common_cfg: dict) -> int:
    return repeat_count(common_cfg, 1)


def _result_root(target: str) -> Path:
    target = _normalize_target(target)
    if target == "liquid":
        return LIQUID_RESULT_DIR
    raise ValueError(f"Only liquid target is supported: {target}")


def _run_target(target: str, cfg: dict, rep: int) -> str:
    # trial の target に応じて training / liquid / test などの実行関数を呼び分ける。
    target = _normalize_target(target)
    if target == "liquid":
        return run_liquid(cfg)
    raise ValueError(f"Only liquid target is supported: {target}")


def _is_trial_level_target(target: str) -> bool:
    return False


def _is_single_run_target(target: str) -> bool:
    return True


def _trial_result_payload(
    *,
    experiment: dict,
    trial: dict,
    target: str,
    dry_run: bool,
    disable_inner_mp: bool = False,
) -> dict:
    # 1 trial/rep の設定上書き、実行、結果行作成をまとめて行う。
    base_cfg = build_cfg()
    cfg = apply_overrides(base_cfg, trial["overrides"])

    run_id = experiment_run_id(experiment["name"], trial["id"])
    cfg["experiment"] = {
        "name": experiment["name"],
        "id": run_id,
        "trial_id": trial["id"],
        "trial_index": trial["index"],
        "target": target,
        "memo": trial.get("memo", ""),
        "overrides": trial["overrides"],
    }

    net_cfg = build_network_cfg(cfg)
    out_dir = make_run_output_dir(
        _result_root(target),
        cfg,
        net_cfg,
        include_output=False,
    )
    save_hierarchy_param_snapshots(
        out_dir,
        cfg,
        net_cfg,
        include_output=False,
    )
    write_experiment_trial_json(
        out_dir / "experiment_trial.json",
        {
            "experiment": experiment,
            "trial": trial,
            "target": target,
            "result_dir": str(out_dir),
        },
    )
    save_used_parameters_text(
        out_dir,
        cfg,
        net_cfg,
        include_output=False,
        extra={
            "experiment": experiment,
            "trial": trial,
            "target": target,
            "result_dir": str(out_dir),
        },
    )

    started_at = now_text()
    message = ""
    status = "dry_run" if dry_run else "done"
    tb_text = ""
    no_mp_env = str(cfg["run"].get("NO_MP_ENV", RUN_CFG.get("NO_MP_ENV", "LSM_NO_MP")))
    old_no_mp = os.environ.get(no_mp_env)

    try:
        if disable_inner_mp:
            os.environ[no_mp_env] = "1"

        if not dry_run:
            if _is_single_run_target(target):
                message = _run_target(target, cfg, 1)
            else:
                messages = []
                for rep in range(1, _num_repeats(cfg["common"]) + 1):
                    messages.append(_run_target(target, cfg, rep))
                message = " | ".join(messages)
    except Exception as exc:
        status = "failed"
        message = f"{type(exc).__name__}: {exc}"
        tb_text = traceback.format_exc()
    finally:
        if disable_inner_mp:
            if old_no_mp is None:
                os.environ.pop(no_mp_env, None)
            else:
                os.environ[no_mp_env] = old_no_mp

    row = {
        "experiment_name": experiment["name"],
        "trial_index": trial["index"],
        "trial_id": trial["id"],
        "target": target,
        "status": status,
        "started_at": started_at,
        "finished_at": now_text(),
        "result_dir": str(out_dir),
        "message": message,
        "overrides_json": json.dumps(jsonable(trial["overrides"]), ensure_ascii=False),
    }
    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "row": row,
        "traceback": tb_text,
    }


def _report_trial_result(result: dict) -> None:
    row = result["row"]
    run_id = result["run_id"]
    out_dir = result["out_dir"]

    if row["status"] == "dry_run":
        print(f"[dry-run] {run_id} -> {out_dir}")
        return

    if row["status"] == "done":
        print(f"[done] {run_id} -> {out_dir}")
        return

    print(f"[failed] {run_id}: {row['message']}")
    if result.get("traceback"):
        print(result["traceback"])


def run_trial(
    *,
    experiment: dict,
    trial: dict,
    target: str,
    manifest_path: Path,
    dry_run: bool,
    disable_inner_mp: bool = False,
) -> None:
    # 1つの trial を実行し、結果を表示して manifest.csv に追記する。
    result = _trial_result_payload(
        experiment=experiment,
        trial=trial,
        target=target,
        dry_run=dry_run,
        disable_inner_mp=disable_inner_mp,
    )
    _report_trial_result(result)
    append_manifest_row(manifest_path, result["row"])
    if result["row"]["status"] == "failed":
        raise RuntimeError(result["row"]["message"])


def _run_trial_worker(
    experiment: dict,
    trial: dict,
    target: str,
    dry_run: bool,
    disable_inner_mp: bool,
) -> dict:
    return _trial_result_payload(
        experiment=experiment,
        trial=trial,
        target=target,
        dry_run=dry_run,
        disable_inner_mp=disable_inner_mp,
    )


def _default_jobs() -> int:
    return max(1, int(RUN_CFG.get("EXPERIMENT_MAX_WORKERS", 1)))


def _disable_inner_mp_for_parallel() -> bool:
    return bool(RUN_CFG.get("EXPERIMENT_DISABLE_INNER_MP", True))


def _jobs_for_run(requested_jobs: int | None, trial_count: int) -> int:
    jobs = _default_jobs() if requested_jobs is None else int(requested_jobs)
    return max(1, min(jobs, max(1, trial_count)))


def run_trials(
    *,
    experiment: dict,
    trials: list[dict],
    default_target: str,
    manifest_path: Path,
    dry_run: bool,
    jobs: int,
) -> None:
    # 複数 trial を順次または並列に実行し、失敗した trial があれば最後に通知する。
    failures = []
    use_parallel = jobs > 1 and len(trials) > 1
    disable_inner_mp = use_parallel and _disable_inner_mp_for_parallel()

    if use_parallel:
        print(f"[parallel] jobs={jobs} inner_mp={'off' if disable_inner_mp else 'on'}")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
            future_map = {}
            for trial in trials:
                trial_target = _normalize_target(trial.get("target") or default_target)
                future = executor.submit(
                    _run_trial_worker,
                    experiment,
                    trial,
                    trial_target,
                    dry_run,
                    disable_inner_mp,
                )
                future_map[future] = trial

            for future in as_completed(future_map):
                trial = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(trial["id"])
                    print(f"[failed] {trial['id']}: worker crashed: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    continue
                _report_trial_result(result)
                append_manifest_row(manifest_path, result["row"])
                if result["row"]["status"] == "failed":
                    failures.append(result["row"]["trial_id"])
    else:
        for trial in trials:
            trial_target = _normalize_target(trial.get("target") or default_target)
            try:
                run_trial(
                    experiment=experiment,
                    trial=trial,
                    target=trial_target,
                    manifest_path=manifest_path,
                    dry_run=dry_run,
                    disable_inner_mp=False,
                )
            except RuntimeError:
                failures.append(trial["id"])
                break

    if failures:
        raise RuntimeError(f"Failed trials: {', '.join(failures)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LSM experiment trials.")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="exp_001_base",
        help="Experiment module name in c_configs.EXPERIMENTS or a .py file path.",
    )
    parser.add_argument(
        "--target",
        choices=("liquid",),
        default=None,
        help="Override experiment target.",
    )
    parser.add_argument(
        "--trial",
        default=None,
        help="Run only one trial id from the experiment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create trial folders/manifest rows without running Brian2.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of trials to run in parallel. Default: cfg_run.EXPERIMENT_MAX_WORKERS.",
    )
    args = parser.parse_args()

    # コマンドラインから EXPERIMENT を選び、必要なら target/jobs/trial を上書きして実行する。
    experiment = load_experiment(args.experiment)
    target = _normalize_target(args.target or experiment.get("target", "training"))
    trials = expand_experiment(experiment)
    if args.trial is not None:
        trials = [trial for trial in trials if trial["id"] == args.trial]
        if not trials:
            raise ValueError(f"Trial id not found: {args.trial}")

    manifest_path = (
        PROJECT_ROOT
        / build_cfg()["run"]["RESULTS_DIR"]
        / "experiment_manifests"
        / f"{experiment['name']}.csv"
    )

    print(f"[experiment] {experiment['name']} target={target} trials={len(trials)}")
    print(f"[manifest] {manifest_path}")
    jobs = _jobs_for_run(args.jobs, len(trials))
    run_trials(
        experiment=experiment,
        trials=trials,
        default_target=target,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
        jobs=jobs,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
