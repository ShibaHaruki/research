"""固定VAE Encoderを使ってCMA-ES探索を実行する。"""

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
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOL_DIR = PROJECT_ROOT / "d_tools"
CONFIG_DIR = PROJECT_ROOT / "c_configs" / "VAE_SEARCH"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for extra_path in (SCRIPT_DIR, TOOL_DIR, CONFIG_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from c_configs.FIXED import cfg_run
from cma_es_search import (
    BoundedCMAES,
    apply_parameter_values,
    compute_common_vae_population_metrics,
    compute_internal_state_metric,
    initial_unit_vector,
    metric_uses_common_vae,
    normalize_parameter_specs,
    unit_vector_to_values,
)
from d_tools.experiments import apply_overrides, now_text
from d_tools.internal_state import internal_state_config
from d_tools.run_paths import jsonable, make_run_output_dir, safe_stem
from f_run.plot_cma_es_progress import save_progress_plots
from f_run.run_liquid import LIQUID_RESULT_DIR, run_liquid
from f_run.run_training import build_cfg, build_network_cfg


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
CMA_RESULT_DIR = RESULTS_PATH / str(RUN_CFG.get("CMA_ES_RESULT_DIR", "cma_es_search"))


def _load_search_module(name_or_path: str):
    # Pythonファイルまたはモジュール名からCMA-ES設定を読み込む。
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import CMA-ES config file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    module_name = name_or_path
    if "." not in module_name:
        local_config = CONFIG_DIR / f"{module_name}.py"
        if local_config.exists():
            return _load_search_module(str(local_config))
        module_name = f"c_configs.EXPERIMENTS.{module_name}"
    return importlib.import_module(module_name)


def load_search_config(name_or_path: str) -> dict:
    module = _load_search_module(name_or_path)
    if not hasattr(module, "CMA_ES"):
        raise AttributeError(f"{module.__name__} must define CMA_ES.")
    cfg = deepcopy(getattr(module, "CMA_ES"))
    cfg.setdefault("name", Path(name_or_path).stem)
    return cfg


def _search_output_dir(search_cfg: dict) -> Path:
    timestamp = now_text().replace(":", "").replace("-", "").replace("T", "_")
    name = safe_stem(search_cfg.get("name", "cma_es_search"))
    return CMA_RESULT_DIR / f"{name}__{timestamp}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _metric_loss(metric_result: dict) -> float:
    # CMA-ESはlossを最小化するため、maximize指標は符号を反転する。
    score = float(metric_result["score"])
    direction = str(metric_result.get("direction", "maximize")).lower()
    if not np.isfinite(score):
        return 1e12
    if direction == "minimize":
        return score
    return -score


def _candidate_key(generation: int, candidate_index: int) -> str:
    return f"g{int(generation):03d}_c{int(candidate_index):03d}"


def _candidate_experiment_payload(
    *,
    search_cfg: dict,
    generation: int,
    candidate_index: int,
    values: dict,
) -> dict:
    search_name = safe_stem(search_cfg.get("name", "cma_es_search"))
    trial_id = _candidate_key(generation, candidate_index)
    return {
        "name": search_name,
        "id": safe_stem(f"{search_name}__{trial_id}"),
        "trial_id": trial_id,
        "trial_index": int(generation * 100000 + candidate_index),
        "target": "liquid",
        "memo": "CMA-ES candidate",
        "overrides": values,
    }


def evaluate_candidate(payload: dict) -> dict:
    # 1陋溷揃・｣諛ｷ繝ｻ邵ｺ・ｮ髫ｧ遨ゑｽｾ・｡:
    # 郢昜ｻ｣ﾎ帷ｹ晢ｽ｡郢晢ｽｼ郢ｧ・ｿ陷ｿ閧ｴ荳・-> run_liquid 陞ｳ貅ｯ・｡繝ｻ-> 陷繝ｻﾎ夊ｿ･・ｶ隲ｷ蛹ｺ谺隶灘揃・ｨ閧ｲ・ｮ繝ｻ-> history 騾包ｽｨ row 闖ｴ諛医・邵ｲ繝ｻ    started_at = now_text()
    search_cfg = payload["search_cfg"]
    params = payload["params"]
    values = payload["values"]
    generation = int(payload["generation"])
    candidate_index = int(payload["candidate_index"])
    metric_cfg = search_cfg["metric"]
    search_out_dir = Path(payload["search_out_dir"])
    defer_metric = bool(payload.get("defer_metric", False))
    candidate_key = _candidate_key(generation, candidate_index)
    internal_state_dir = Path("")

    try:
        # 郢晏生繝ｻ郢ｧ・ｹ髫ｪ・ｭ陞ｳ螢ｹ竊楢ｬ暦ｽ｢驍擾ｽ｢陋溷揃・｣諛翫・陋滂ｽ､郢ｧ蜑・ｽｸ鬆大ｶ檎ｸｺ髦ｪ・邵ｲ竏夲ｼ・ｸｺ・ｮ陋溷揃・｣諛ｷ・ｰ繧臥舞邵ｺ・ｮ experiment id 郢ｧ蜑・ｽｻ蛟･・郢ｧ荵敖繝ｻ        cfg = build_cfg()
        cfg = apply_overrides(cfg, search_cfg.get("base_overrides", {}))
        cfg = apply_parameter_values(cfg, params, values)
        cfg["experiment"] = _candidate_experiment_payload(
            search_cfg=search_cfg,
            generation=generation,
            candidate_index=candidate_index,
            values=values,
        )

        net_cfg = build_network_cfg(cfg)
        run_out_dir = make_run_output_dir(LIQUID_RESULT_DIR, cfg, net_cfg, include_output=False)
        # run_liquidで候補の内部状態を保存する。
        message = run_liquid(cfg)
        internal_state_dir = run_out_dir / internal_state_config(cfg["run"])["dir_name"]
        if defer_metric:
            # 固定Encoderなしの共通VAEでは、run_liquid後に世代単位でまとめて評価する。
            metric_result = {
                "metric": metric_cfg.get("name", ""),
                "direction": metric_cfg.get("direction", "maximize"),
                "score": np.nan,
                "details": {"deferred_common_vae": True, "candidate_key": candidate_key},
            }
            loss = np.nan
            status = "simulated"
        else:
            metric_out_dir = search_out_dir / "candidate_metrics" / candidate_key
            metric_result = compute_internal_state_metric(
                internal_state_dir,
                metric_cfg,
                metric_out_dir=metric_out_dir,
            )
            loss = _metric_loss(metric_result)
            status = "done"
        tb_text = ""
    except Exception as exc:
        run_out_dir = Path("")
        message = f"{type(exc).__name__}: {exc}"
        metric_result = {
            "metric": metric_cfg.get("name", ""),
            "direction": metric_cfg.get("direction", "maximize"),
            "score": float("-inf"),
            "details": {"error": message},
        }
        loss = 1e12
        status = "failed"
        tb_text = traceback.format_exc()

    row = {
        "generation": generation,
        "candidate_index": candidate_index,
        "candidate_key": candidate_key,
        "status": status,
        "started_at": started_at,
        "finished_at": now_text(),
        "score": float(metric_result["score"]),
        "loss": float(loss),
        "metric": metric_result.get("metric", ""),
        "direction": metric_result.get("direction", ""),
        "run_out_dir": str(run_out_dir),
        "internal_state_dir": str(internal_state_dir),
        "message": message,
        "params_json": json.dumps(jsonable(values), ensure_ascii=False),
        "metric_details_json": json.dumps(jsonable(metric_result.get("details", {})), ensure_ascii=False),
    }
    return {"row": row, "traceback": tb_text}


def _append_history(history_csv: Path, rows: list[dict]) -> None:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if history_csv.exists():
        df.to_csv(history_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(history_csv, index=False)


def _evaluate_population(
    *,
    candidates: list[dict],
    params: list,
    search_cfg: dict,
    search_out_dir: Path,
    generation: int,
    jobs: int,
) -> list[dict]:
    # 1闕ｳ邏具ｽｻ・｣陋ｻ繝ｻ繝ｻ陋溷揃・｣諛奇ｽ堤ｸｲ・頴bs=1邵ｺ・ｪ郢ｧ陋ｾ・ｰ繝ｻ・ｬ・｡邵ｲ・頴bs>1邵ｺ・ｪ郢ｧ迚呎肩郢晏干ﾎ溽ｹｧ・ｻ郢ｧ・ｹ邵ｺ・ｧ闕ｳ・ｦ陋ｻ闍難ｽｩ遨ゑｽｾ・｡邵ｺ蜷ｶ・狗ｸｲ繝ｻ    payloads = []
    defer_metric = metric_uses_common_vae(search_cfg["metric"])
    for candidate in candidates:
        values = unit_vector_to_values(params, candidate["x"])
        payloads.append(
            {
                "search_cfg": search_cfg,
                "params": params,
                "values": values,
                "generation": generation,
                "candidate_index": int(candidate["index"]),
                "search_out_dir": str(search_out_dir),
                "defer_metric": defer_metric,
            }
        )

    if jobs <= 1:
        return [evaluate_candidate(payload) for payload in payloads]

    ctx = mp.get_context("spawn")
    results = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=ctx) as executor:
        future_map = {
            executor.submit(evaluate_candidate, payload): index
            for index, payload in enumerate(payloads)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


def _common_vae_candidate_entry(row: dict) -> dict:
    return {
        "candidate_key": row["candidate_key"],
        "generation": int(row["generation"]),
        "candidate_index": int(row["candidate_index"]),
        "run_out_dir": row.get("run_out_dir", ""),
        "internal_state_dir": row.get("internal_state_dir", ""),
        "params_json": row.get("params_json", "{}"),
    }


def _apply_common_vae_metrics(
    *,
    rows: list[dict],
    common_entries: list[dict],
    metric_cfg: dict,
    metric_out_dir: Path,
) -> None:
    # 固定VAE Encoderで得た潜在表現からscore/lossを計算する。
    metric_results = compute_common_vae_population_metrics(
        common_entries,
        metric_cfg,
        metric_out_dir=metric_out_dir,
    )
    for row in rows:
        if row.get("status") != "simulated":
            continue
        candidate_key = str(row["candidate_key"])
        metric_result = metric_results[candidate_key]
        row["status"] = "done"
        row["score"] = float(metric_result["score"])
        row["loss"] = float(_metric_loss(metric_result))
        row["metric"] = metric_result.get("metric", "")
        row["direction"] = metric_result.get("direction", "")
        row["metric_details_json"] = json.dumps(
            jsonable(metric_result.get("details", {})),
            ensure_ascii=False,
        )
        row["finished_at"] = now_text()


def _save_final_common_vae_scores(
    *,
    search_out_dir: Path,
    common_entries: list[dict],
    metric_cfg: dict,
) -> dict | None:
    # 有効な内部状態がない場合はVAE評価を行わない。
    if not common_entries:
        return None
    final_dir = search_out_dir / "common_vae_metrics" / "final_all_candidates"
    metric_results = compute_common_vae_population_metrics(
        common_entries,
        metric_cfg,
        metric_out_dir=final_dir,
    )
    rows = []
    for entry in common_entries:
        candidate_key = str(entry["candidate_key"])
        metric_result = metric_results[candidate_key]
        loss = _metric_loss(metric_result)
        rows.append(
            {
                "candidate_key": candidate_key,
                "generation": int(entry["generation"]),
                "candidate_index": int(entry["candidate_index"]),
                "score": float(metric_result["score"]),
                "loss": float(loss),
                "metric": metric_result.get("metric", ""),
                "direction": metric_result.get("direction", ""),
                "run_out_dir": entry.get("run_out_dir", ""),
                "internal_state_dir": entry.get("internal_state_dir", ""),
                "params_json": entry.get("params_json", "{}"),
                "metric_details_json": json.dumps(
                    jsonable(metric_result.get("details", {})),
                    ensure_ascii=False,
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("loss", ascending=True)
    scores_csv = final_dir / "final_common_vae_scores.csv"
    df.to_csv(scores_csv, index=False)
    best_row = df.iloc[0].to_dict()
    try:
        best_params = json.loads(str(best_row.get("params_json", "{}")))
    except json.JSONDecodeError:
        best_params = {}
    best_payload = {
        "best_row": best_row,
        "best_params": best_params,
        "scores_csv": str(scores_csv),
        "final_common_vae_dir": str(final_dir),
    }
    _write_json(search_out_dir / "best_final_common_vae.json", best_payload)
    return best_payload


def run_search(search_cfg: dict, *, dry_run: bool = False, jobs_override: int | None = None) -> dict:
    # CMA-ES探索の出力先を作成する。
    search_out_dir = _search_output_dir(search_cfg)
    search_out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(search_out_dir / "search_config.json", search_cfg)

    base_cfg = build_cfg()
    base_cfg = apply_overrides(base_cfg, search_cfg.get("base_overrides", {}))
    params = normalize_parameter_specs(search_cfg["parameters"], base_cfg)
    x0 = initial_unit_vector(params)

    cma_cfg = dict(search_cfg.get("cma", {}))
    generations = int(cma_cfg.get("generations", 10))
    jobs = int(jobs_override if jobs_override is not None else cma_cfg.get("jobs", 1))
    jobs = max(1, jobs)
    optimizer = BoundedCMAES(
        x0,
        sigma0=float(cma_cfg.get("sigma0", 0.25)),
        population_size=cma_cfg.get("population_size"),
        seed=int(cma_cfg.get("seed", 0)),
    )

    param_table = [
        {
            "name": param.name,
            "path": repr(param.path),
            "lower": param.lower,
            "upper": param.upper,
            "x0": param.x0,
            "scale": param.scale,
            "kind": param.kind,
        }
        for param in params
    ]
    pd.DataFrame(param_table).to_csv(search_out_dir / "parameters.csv", index=False)

    history_csv = search_out_dir / "history.csv"
    best_row = None
    use_common_vae = metric_uses_common_vae(search_cfg["metric"])
    common_vae_scope = str(search_cfg["metric"].get("common_vae_scope", "generation")).lower()
    if common_vae_scope not in {"generation", "cumulative"}:
        raise ValueError("metric.common_vae_scope must be 'generation' or 'cumulative'.")
    common_vae_pool: list[dict] = []

    print(f"[cma-es] search={search_cfg['name']} generations={generations} pop={optimizer.lambda_} jobs={jobs}")
    print(f"[cma-es] metric={search_cfg['metric'].get('name')} direction={search_cfg['metric'].get('direction', 'maximize')}")
    if use_common_vae:
        print(f"[cma-es] common VAE latent space enabled scope={common_vae_scope}")
    print(f"[cma-es] out_dir={search_out_dir}")

    for generation in range(1, generations + 1):
        # ask: 次世代の候補パラメータを生成する。
        candidates = optimizer.ask()
        candidate_preview = [
            {
                "generation": generation,
                "candidate_index": int(candidate["index"]),
                "unit": np.asarray(candidate["x"]).tolist(),
                "values": unit_vector_to_values(params, candidate["x"]),
            }
            for candidate in candidates
        ]
        _write_json(search_out_dir / "candidate_values" / f"generation_{generation:03d}.json", {"candidates": candidate_preview})

        if dry_run:
            # dry-runではシミュレーションせず候補だけ保存する。
            rows = []
            for candidate in candidate_preview:
                rows.append(
                    {
                        "generation": generation,
                        "candidate_index": candidate["candidate_index"],
                        "candidate_key": _candidate_key(generation, candidate["candidate_index"]),
                        "status": "dry_run",
                        "started_at": now_text(),
                        "finished_at": now_text(),
                        "score": np.nan,
                        "loss": np.nan,
                        "metric": search_cfg["metric"].get("name", ""),
                        "direction": search_cfg["metric"].get("direction", ""),
                        "run_out_dir": "",
                        "internal_state_dir": "",
                        "message": "",
                        "params_json": json.dumps(jsonable(candidate["values"]), ensure_ascii=False),
                        "metric_details_json": "{}",
                    }
                )
            _append_history(history_csv, rows)
            print(f"[dry-run] generation {generation}: wrote candidate values")
            continue

        results = _evaluate_population(
            candidates=candidates,
            params=params,
            search_cfg=search_cfg,
            search_out_dir=search_out_dir,
            generation=generation,
            jobs=jobs,
        )
        rows = [result["row"] for result in results]
        if use_common_vae:
            current_entries = [
                _common_vae_candidate_entry(row)
                for row in rows
                if row.get("status") == "simulated"
            ]
            if common_vae_scope == "cumulative":
                common_vae_pool.extend(current_entries)
                common_entries = list(common_vae_pool)
            else:
                common_entries = current_entries
            common_metric_dir = (
                search_out_dir
                / "common_vae_metrics"
                / f"{common_vae_scope}_generation_{generation:03d}"
            )
            _apply_common_vae_metrics(
                rows=rows,
                common_entries=common_entries,
                metric_cfg=search_cfg["metric"],
                metric_out_dir=common_metric_dir,
            )

        losses = [
            float(row["loss"]) if np.isfinite(float(row["loss"])) else 1e12
            for row in rows
        ]
        # tell: 各候補のlossをCMA-ESへ返して分布を更新する。
        state = optimizer.tell(candidates, losses)
        _append_history(history_csv, rows)

        gen_best = min(rows, key=lambda row: float(row["loss"]))
        if best_row is None or float(gen_best["loss"]) < float(best_row["loss"]):
            best_row = dict(gen_best)
            _write_json(search_out_dir / "best_so_far.json", best_row)

        print(
            f"[generation {generation:03d}] "
            f"best_score={float(gen_best['score']):.6g} "
            f"best_loss={float(gen_best['loss']):.6g} "
            f"sigma={state['sigma']:.4g}"
        )

    if best_row is not None:
        best_params = json.loads(best_row["params_json"])
        _write_json(
            search_out_dir / "best_params.json",
            {
                "best_row": best_row,
                "best_params": best_params,
                "search_out_dir": str(search_out_dir),
            },
        )
        print(f"[best] score={float(best_row['score']):.6g} params={best_params}")

    final_common_vae = None
    if use_common_vae and not dry_run and common_vae_pool:
        try:
            final_common_vae = _save_final_common_vae_scores(
                search_out_dir=search_out_dir,
                common_entries=common_vae_pool,
                metric_cfg=search_cfg["metric"],
            )
            if final_common_vae is not None:
                print(f"[best-final-common-vae] {final_common_vae['scores_csv']}")
        except Exception as exc:
            print(f"[warn] failed to save final common VAE scores: {type(exc).__name__}: {exc}")

    if not dry_run and history_csv.exists():
        try:
            saved_plots = save_progress_plots(search_out_dir)
            print(f"[progress] saved plots to {search_out_dir / 'progress_plots'}")
            _write_json(search_out_dir / "progress_plots.json", saved_plots)
        except Exception as exc:
            print(f"[warn] failed to save progress plots: {type(exc).__name__}: {exc}")

    return {
        "search_out_dir": str(search_out_dir),
        "best_row": best_row,
        "best_final_common_vae": final_common_vae,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CMA-ES parameter search using liquid internal-state metrics."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="cma_es_internal_state",
        help="CMA_ES config module in c_configs.EXPERIMENTS or a .py file path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=None, help="Override CMA config jobs.")
    parser.add_argument("--generations", type=int, default=None, help="Override CMA generations.")
    parser.add_argument("--population-size", type=int, default=None, help="Override CMA population size.")
    parser.add_argument("--metric", default=None, help="Override metric name, e.g. DR, silhouette, trace_Sw.")
    parser.add_argument(
        "--direction",
        choices=("maximize", "minimize"),
        default=None,
        help="Override metric direction.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Override metric max_samples_per_class for faster search.",
    )
    parser.add_argument(
        "--liquid-samples",
        type=int,
        default=None,
        help="Override liquid.NUM_LIQUID_SAMPLE for each material.",
    )
    parser.add_argument(
        "--materials",
        default=None,
        help="Comma-separated material names for quick searches.",
    )
    parser.add_argument("--vae-epochs", type=int, default=None, help="Override metric.vae.epochs.")
    parser.add_argument("--vae-latent-dim", type=int, default=None, help="Override metric.vae.latent_dim.")
    parser.add_argument(
        "--fixed-vae-encoder-dir",
        default=None,
        help="Directory or .pt file of a pretrained VAE encoder. If set, CMA-ES does not retrain VAE during search.",
    )
    parser.add_argument(
        "--common-vae",
        action="store_true",
        help="Train one VAE on all candidate internal states and evaluate in a shared latent space.",
    )
    parser.add_argument(
        "--separate-vae",
        action="store_true",
        help="Disable shared latent space and train/evaluate VAE separately for each candidate.",
    )
    parser.add_argument(
        "--common-vae-scope",
        choices=("generation", "cumulative"),
        default=None,
        help="Shared VAE data scope. generation: current population only, cumulative: all evaluated candidates so far.",
    )
    parser.add_argument("--rate-max-hz", type=float, default=None, help="Override firing-rate penalty upper target.")
    parser.add_argument("--sync-max", type=float, default=None, help="Override synchrony penalty threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    search_cfg = load_search_config(args.config)
    # コマンドライン引数で指定された値は設定より優先する。
    if args.generations is not None:
        search_cfg.setdefault("cma", {})["generations"] = int(args.generations)
    if args.population_size is not None:
        search_cfg.setdefault("cma", {})["population_size"] = int(args.population_size)
    if args.metric is not None:
        search_cfg.setdefault("metric", {})["name"] = str(args.metric)
    if args.direction is not None:
        search_cfg.setdefault("metric", {})["direction"] = str(args.direction)
    if args.max_samples_per_class is not None:
        search_cfg.setdefault("metric", {})["max_samples_per_class"] = int(args.max_samples_per_class)
        search_cfg.setdefault("metric", {}).setdefault("vae", {})["max_samples_per_class"] = int(args.max_samples_per_class)
    if args.liquid_samples is not None:
        search_cfg.setdefault("base_overrides", {})["liquid.NUM_LIQUID_SAMPLE"] = int(args.liquid_samples)
    if args.materials is not None:
        materials = [item.strip() for item in str(args.materials).split(",") if item.strip()]
        if materials:
            search_cfg.setdefault("base_overrides", {})["liquid.LIQUID_MAT"] = materials
    if args.vae_epochs is not None:
        search_cfg.setdefault("metric", {}).setdefault("vae", {})["epochs"] = int(args.vae_epochs)
    if args.vae_latent_dim is not None:
        search_cfg.setdefault("metric", {}).setdefault("vae", {})["latent_dim"] = int(args.vae_latent_dim)
    if args.fixed_vae_encoder_dir is not None:
        search_cfg.setdefault("metric", {}).setdefault("vae", {})["fixed_encoder_dir"] = str(args.fixed_vae_encoder_dir)
    if args.common_vae:
        search_cfg.setdefault("metric", {})["common_latent_space"] = True
    if args.separate_vae:
        search_cfg.setdefault("metric", {})["common_latent_space"] = False
    if args.common_vae_scope is not None:
        search_cfg.setdefault("metric", {})["common_vae_scope"] = str(args.common_vae_scope)
    if args.rate_max_hz is not None:
        search_cfg.setdefault("metric", {}).setdefault("penalties", {})["target_rate_max_hz"] = float(args.rate_max_hz)
    if args.sync_max is not None:
        search_cfg.setdefault("metric", {}).setdefault("penalties", {})["sync_max"] = float(args.sync_max)

    no_mp_env = str(RUN_CFG.get("NO_MP_ENV", "LSM_NO_MP"))
    old_no_mp = os.environ.get(no_mp_env)
    if args.jobs is not None and int(args.jobs) > 1:
        os.environ[no_mp_env] = "1"
    try:
        run_search(search_cfg, dry_run=bool(args.dry_run), jobs_override=args.jobs)
    finally:
        if args.jobs is not None and int(args.jobs) > 1:
            if old_no_mp is None:
                os.environ.pop(no_mp_env, None)
            else:
                os.environ[no_mp_env] = old_no_mp
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






