"""共有Liquidの時定数候補とseedを準備し、rep単位のプロセス実行を管理する。"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
from a_experiment.experiment_sets import (
    RUNTIME_CACHE_SETTINGS,
    learning_rule_name,
    output_learning_rules,
    output_learning_parameters,
)
from a_experiment.paths import (
    training_directory,
    readout_rule_directory,
    readout_seed_directory,
    shared_readout_seed_directory,
    trained_weight_directory,
)
from f_run.common import cfg_from_args, validate_cfg, cfg_from_snapshot, debug_readout_materials
from d_tools.progress import wait_with_progress
from d_tools.execution_control import (
    exclusive_execution,
    assert_recording_available,
    stop_process_tree,
)
from f_run.training import run_shared_training
from d_tools.recording_io import read_record, record_exists, encode, decode, save_record
from d_tools.parameter_recording import (
    parameter_records_equal,
    save_parameter_snapshots,
    snapshot,
    read_training_configuration,
)
from c_configs.SPACE.cfg_learning_rule_models import TAU_CANDIDATES_MS, SEED_COUNT

# 子プロセスへ渡す実行入口を、作業ディレクトリに依存せず指定する。
ROOT = Path(__file__).resolve().parents[1]


def make_sweep_base_cfg():
    """共有Liquid実験の既定設定。CLIのデータ数などはこの設定に上書きできる。"""
    from f_run.common import build_cfg

    cfg = deepcopy(build_cfg())
    cfg["models"]["LEARNING_RULE_MODEL"] = learning_rule_name(cfg)
    # エポック数は通常実行と同じTRAINING_SETSのNUM_TRAINING_EPOCHを使う。
    cfg["common"]["NUM_REPEAT"] = 1
    return cfg


def make_candidate_cfg(tau_ms, out_dir=None, *, seed_index=1, base_cfg=None):
    """同じseedの候補群では出力結合・初期重みを共通にする。"""
    if seed_index < 1:
        raise ValueError("Seed index must be positive")
    cfg = deepcopy(make_sweep_base_cfg() if base_cfg is None else base_cfg)
    if learning_rule_name(cfg) != "STDP":
        raise ValueError("--sweep currently compares STDP time constants")
    rules = output_learning_rules(cfg)
    parameters = output_learning_parameters(cfg)
    if len(rules) > 1:
        matches = [
            i
            for i, (rule, params) in enumerate(zip(rules, parameters))
            if rule == "STDP"
            and params.get("tau_plus") == float(tau_ms)
            and params.get("tau_minus") == float(tau_ms)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Exactly one output layer must define STDP tau_plus=tau_minus={tau_ms} ms"
            )
        selected = matches[0]
    else:
        selected = 0
    chosen = deepcopy(parameters[selected])
    chosen.update(tau_plus=float(tau_ms), tau_minus=float(tau_ms))
    cfg["learning_rule_models"]["STDP"] = deepcopy(chosen)
    cfg["network"]["N_out"] = [cfg["network"]["N_out"][selected]]
    for key in ("p_liq_to_out_pairs", "gain_liq_to_out_pairs"):
        cfg["network"][key] = [deepcopy(cfg["network"][key][selected])]
    cfg["network"]["learning_rule_liq_to_out"] = ["STDP"]
    cfg["network"]["OUTPUT_LEARNING_PARAMETERS"] = [chosen]
    cfg["network"].pop("OUTPUT_SEEDS", None)
    cfg["models"]["LEARNING_RULE_MODEL"] = "STDP"
    cfg["network"]["OUTPUT_SEED"] = int(cfg["common"]["BASE_SEED"]) + seed_index
    cfg["run"]["READOUT_SEED_INDEX"] = seed_index
    cfg["run"]["READOUT_TRAIN_DIR"] = str(
        readout_seed_directory(cfg, seed_index, rule_dir=out_dir).resolve()
    )
    return cfg


def _now():
    """進捗ログに記録するUTC時刻を返す。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_existing_configuration(cfg):
    """同じ保存先に異なる条件の実験が混ざらないことを確認する。"""
    folder = training_directory(cfg)
    saved = read_training_configuration(folder)
    if saved is not None and not parameter_records_equal(saved, snapshot(cfg)):
        raise ValueError(f"Existing configuration differs: {folder}")


def add_sweep_arguments(parser):
    """学習・学習テスト・学習テスト評価の3入口に共通の共有Liquidオプション。"""
    parser.add_argument(
        "--sweep", action="store_true", help="Shared-Liquid STDP tau/seed experiment"
    )
    parser.add_argument("--seeds", type=int, default=SEED_COUNT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, SEED_COUNT),
        help="同時に実行するrep数（sweepではseed数）",
    )
    parser.add_argument(
        "--sweep-taus",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of tau candidates (5..250 ms, multiples of 5)",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages")
    parser.add_argument("--plan", action="store_true", help="Print sweep plan without running")
    parser.add_argument("--sweep-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sweep-config-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed-index", type=int, help=argparse.SUPPRESS)


def handle_sweep_arguments(args, *, stage, evaluation_options=None):
    if args.sweep_worker:
        if args.sweep_config_file is None or args.seed_index is None:
            raise ValueError("Sweep worker requires configuration file and seed index")
        run_sweep_worker(args.sweep_config_file, args.seed_index)
        return True
    if not args.sweep:
        return False
    cfg = cfg_from_args(args, base_cfg=make_sweep_base_cfg())
    rep = args.rep if args.rep is not None else 1
    if getattr(args, "rep_end", None) not in (None, rep):
        raise ValueError("Sweep uses one fixed data repetition across all output seeds")
    print(
        run_parameter_sweep(
            cfg,
            stage=stage,
            rep=rep,
            seed_count=args.seeds,
            workers=args.workers,
            taus=args.sweep_taus,
            out_dir=args.out_dir,
            resume=args.resume,
            plan_only=args.plan,
            evaluation_options=evaluation_options,
        )
    )
    return True


@exclusive_execution(
    lambda plan_file, seed_index: Path(plan_file).parent
    / "_shared"
    / f"seed_{seed_index:03d}"
    / "_runtime_cache"
    / "worker.lock"
)
def run_sweep_worker(plan_file, seed_index):
    """1 seed内の全τを共通Liquidで学習し、要求されたテスト・評価へ進む。"""

    plan = read_record(Path(plan_file))
    if not 1 <= seed_index <= plan["seed_count"]:
        raise ValueError("Invalid seed index")
    base = cfg_from_snapshot(decode(plan["base_configuration"]))
    base["run"]["DATA_SPLIT_REP"] = plan["rep"]
    rep, stage = seed_index, plan["stage"]
    configurations = [
        make_candidate_cfg(
            tau_ms, Path(plan["rule_directory"]), seed_index=seed_index, base_cfg=base
        )
        for tau_ms in plan["tau_plus_equals_tau_minus_ms"]
    ]
    shared_dir = shared_readout_seed_directory(
        configurations[0], seed_index, rule_dir=Path(plan["rule_directory"])
    )
    pending = []
    started = _now()
    try:
        for cfg in configurations:
            save_parameter_snapshots(cfg, rep=rep)
            _check_existing_configuration(cfg)
            folder = training_directory(cfg)
            final = trained_weight_directory(folder, rep)
            summary = folder / f"training_summary_rep{rep}.npz"
            if final.exists() and record_exists(summary) and plan["resume"]:
                debug = folder / "debug" / "summary.npz"
                if cfg["common"].get("DEBUG_EACH_MATERIAL", True) and not record_exists(debug):
                    debug_readout_materials(cfg, final, folder / "debug")
                continue
            pending.append(cfg)
        if pending:
            print(run_shared_training(rep, pending, shared_dir), flush=True)
        if stage in ("train_test", "train_test_eval"):
            from f_run.train_test import run_shared_test

            print(
                run_shared_test(rep, configurations, shared_dir / "test", resume=plan["resume"]),
                flush=True,
            )
        evaluations = {}
        if stage == "train_test_eval":
            from f_run.evaluation import evaluate_sweep_candidates

            evaluations = evaluate_sweep_candidates(
                rep, configurations, resume=plan["resume"], **plan["evaluation_options"]
            )
        for cfg in configurations:
            folder = training_directory(cfg)
            summary = read_record(folder / f"training_summary_rep{rep}.npz")
            save_record(
                folder / f"job_result_{stage}_rep{rep}.npz",
                {
                    "status": "complete",
                    "stage": stage,
                    "seed_index": seed_index,
                    "tau_ms": cfg["learning_rule_models"]["STDP"]["tau_plus"],
                    "output_seed": cfg["network"]["OUTPUT_SEED"],
                    "rep": rep,
                    "training_dir": str(folder),
                    "started_at": started,
                    "finished_at": _now(),
                    "weight_change_norm": summary["output_weight_change_norm"],
                    "evaluation_summary": evaluations.get(str(folder), []),
                    "shared_liquid": True,
                    "pid": os.getpid(),
                },
            )
    except Exception as exc:
        for cfg in configurations:
            save_record(
                training_directory(cfg) / f"job_result_{stage}_rep{rep}.npz",
                {
                    "status": "failed",
                    "stage": stage,
                    "seed_index": seed_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": _now(),
                },
            )
        traceback.print_exc()
        raise


def _prepare_sweep_jobs(cfg, tau_candidates_ms, seed_count, rule_dir, *, stage, resume):
    """候補と既存結果を先に検査し、seed・時定数ごとのジョブ一覧を作る。"""
    rows = {}
    for index in range(1, seed_count + 1):
        # 旧バージョンで残ったプロセスも、記録ファイルの使用状態で検出する。
        prototype = make_candidate_cfg(
            tau_candidates_ms[0], rule_dir, seed_index=index, base_cfg=cfg
        )
        shared = shared_readout_seed_directory(prototype, index, rule_dir=rule_dir)
        assert_recording_available(
            shared / "test" / "rec" / "Liquid" / f"sliq_rec_rep{index}.partial.npy"
        )
        for tau_ms in tau_candidates_ms:
            candidate = make_candidate_cfg(tau_ms, rule_dir, seed_index=index, base_cfg=cfg)
            _check_existing_configuration(candidate)
            folder = training_directory(candidate)
            if not resume and trained_weight_directory(folder, index).exists():
                raise FileExistsError(f"Already trained: {folder}; use --resume")
            rows[tau_ms, index] = {
                "tau_ms": tau_ms,
                "seed_index": index,
                "output_seed": candidate["network"]["OUTPUT_SEED"],
                "stage": stage,
                "status": "queued",
                "training_dir": str(folder),
                "pid": "",
                "weight_change_norm": "",
                "accuracy": "",
                "error": "",
            }
    return rows


@exclusive_execution(
    lambda cfg, plan, rows: Path(plan["rule_directory"]) / "_runtime_cache" / "sweep.lock"
)
def _execute_sweep_jobs(cfg, plan, rows):
    """ログ・進捗・停止処理を管理しながら、seedごとの子プロセスを実行する。"""
    rule_dir = Path(plan["rule_directory"])
    stage = plan["stage"]
    tau_candidates_ms = plan["tau_plus_equals_tau_minus_ms"]
    seed_count = plan["seed_count"]
    workers = plan["workers"]
    rule_dir.mkdir(parents=True, exist_ok=True)
    plan_file = rule_dir / "sweep_config.json"
    save_record(plan_file, plan)
    lock, stopped = threading.Lock(), threading.Event()
    active = {}
    columns = list(next(iter(rows.values())))

    def save_progress():
        temporary = rule_dir / "sweep_summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows.values())
        temporary.replace(rule_dir / "sweep_summary.csv")
        counts = {
            state: sum(r["status"] == state for r in rows.values())
            for state in ("queued", "running", "complete", "failed", "cancelled")
        }
        save_record(
            rule_dir / "sweep_status.npz",
            {
                "parent_pid": os.getpid(),
                "updated_at": _now(),
                "total_jobs": len(rows),
                "worker_jobs": seed_count,
                "workers": workers,
                "stage": stage,
                **counts,
            },
        )

    def run_seed(index):
        candidate = make_candidate_cfg(
            tau_candidates_ms[0], rule_dir, seed_index=index, base_cfg=cfg
        )
        shared = shared_readout_seed_directory(candidate, index, rule_dir=rule_dir)
        shared.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "f_run/run_train.py"),
            "--sweep-worker",
            "--sweep-config-file",
            str(plan_file),
            "--seed-index",
            str(index),
        ]
        env = os.environ.copy()
        for variable in ("MPLCONFIGDIR", "TEMP", "TMP"):
            child = RUNTIME_CACHE_SETTINGS["ENV_CACHE_MAP"][variable]
            cache = shared / RUNTIME_CACHE_SETTINGS["CACHE_DIR"] / child
            cache.mkdir(parents=True, exist_ok=True)
            env[variable] = str(cache)
        env["PYTHONIOENCODING"] = "utf-8"
        env["LSM_PROGRESS_EVENTS"] = "1"
        with (shared / "training.log").open("a", encoding="utf-8") as log:
            log.write(f"\n[{_now()}] stage={stage}; shared outputs={len(tau_candidates_ms)}\n")
            log.flush()
            start_offset = log.tell()
            with lock:
                if stopped.is_set():
                    for tau_ms in tau_candidates_ms:
                        rows[tau_ms, index]["status"] = "cancelled"
                    save_progress()
                    return
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                active[index] = process
                for tau_ms in tau_candidates_ms:
                    rows[tau_ms, index].update(status="running", pid=process.pid)
                save_progress()
            code = wait_with_progress(
                process,
                shared / "training.log",
                start_offset=start_offset,
                position=index - 1,
                rep=index,
            )
        with lock:
            active.pop(index, None)
            for tau_ms in tau_candidates_ms:
                row = rows[tau_ms, index]
                result_file = Path(row["training_dir"]) / f"job_result_{stage}_rep{index}.npz"
                result = read_record(result_file) if record_exists(result_file) else {}
                complete = code == 0 and result.get("status") == "complete"
                metrics = result.get("evaluation_summary", [])
                row.update(
                    status="complete" if complete else "failed",
                    weight_change_norm=result.get("weight_change_norm", ""),
                    accuracy=metrics[0]["accuracy"] if metrics else "",
                    error=(
                        ""
                        if complete
                        else result.get("error", f"Exit code {code}; see {shared}/training.log")
                    ),
                )
            save_progress()
        print(f"[sweep {stage}] seed {index}: exit code {code}", flush=True)
        if code != 0:
            # 親の抽象的なエラーだけでなく、子が記録した原因とログの場所も表示する。
            for tau_ms in tau_candidates_ms:
                print(f"[failure] {rows[tau_ms, index]['error']}", flush=True)
            print(f"[log] {shared / 'training.log'}", flush=True)

    with lock:
        save_progress()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(run_seed, index) for index in range(1, seed_count + 1)]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stopped.set()
        with lock:
            for process in active.values():
                stop_process_tree(process)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=stopped.is_set())
    if any(row["status"] != "complete" for row in rows.values()):
        raise RuntimeError(
            "Some sweep conditions failed; inspect sweep_summary.csv and use --resume"
        )
    return (
        f"[complete {stage}] {len(rows)} conditions in {seed_count} shared-Liquid runs: {rule_dir}"
    )


def run_parameter_sweep(
    cfg,
    *,
    stage="training",
    rep=1,
    seed_count=SEED_COUNT,
    workers=None,
    taus=None,
    out_dir=None,
    resume=False,
    plan_only=False,
    evaluation_options=None,
):
    """seedごとに並列実行する。tausはミリ秒単位で、各seed内の全候補がLiquidを共有する。"""
    if stage not in ("training", "train_test", "train_test_eval"):
        raise ValueError("Unknown sweep stage")
    cfg = deepcopy(cfg)
    validate_cfg(cfg)
    cfg["run"]["DATA_SPLIT_REP"] = rep
    tau_candidates_ms = list(TAU_CANDIDATES_MS if taus is None else taus)
    if (
        not tau_candidates_ms
        or len(set(tau_candidates_ms)) != len(tau_candidates_ms)
        or any(t not in TAU_CANDIDATES_MS for t in tau_candidates_ms)
    ):
        raise ValueError("Use distinct tau candidates from 250 to 5 ms, in steps of 5")
    workers = min(os.cpu_count() or 1, seed_count) if workers is None else workers
    if min(seed_count, rep, workers) < 1:
        raise ValueError("Seed count, workers and rep must be positive")
    workers = min(workers, seed_count)
    evaluation_options = evaluation_options or {}
    if stage == "train_test_eval":
        from f_run.evaluation import evaluation_settings

        evaluation_settings(cfg, **evaluation_options)
    prototype = make_candidate_cfg(tau_candidates_ms[0], base_cfg=cfg)
    rule_dir = Path(out_dir).resolve() if out_dir else readout_rule_directory(prototype)
    plan = {
        "stage": stage,
        "learning_rule": "STDP",
        "tau_plus_equals_tau_minus_ms": tau_candidates_ms,
        "seed_count": seed_count,
        "total_jobs": len(tau_candidates_ms) * seed_count,
        "worker_jobs": seed_count,
        "workers": workers,
        "rep": rep,
        "resume": resume,
        "architecture": "one_shared_liquid_per_seed",
        "output_groups_per_worker": len(tau_candidates_ms),
        "output_neurons_per_group": cfg["network"]["N_out"][0],
        "liquid_seed": cfg["common"]["BASE_SEED"],
        "output_seeds": [cfg["common"]["BASE_SEED"] + i for i in range(1, seed_count + 1)],
        "rule_directory": str(rule_dir),
        "evaluation_options": evaluation_options,
        "base_configuration": encode(snapshot(cfg)),
    }
    print(
        json.dumps(
            {k: v for k, v in plan.items() if k != "base_configuration"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if plan_only:
        return "[plan] no training/test/evaluation started"
    rows = _prepare_sweep_jobs(
        cfg, tau_candidates_ms, seed_count, rule_dir, stage=stage, resume=resume
    )
    return _execute_sweep_jobs(cfg, plan, rows)
