# Wed Apr 29 23:43:45 2026
# Run as: c:\Users\haru4\OneDrive - 学校法人立命館\ドキュメント\研究\研究コード\LSM_VAE_Search\f_run\run_fixed_vae_encoder_pretrain.py

# -*- coding: utf-8 -*-
"""固定VAE Encoderの事前学習を実行する。"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
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
    apply_parameter_values,
    normalize_parameter_specs,
    unit_vector_to_values,
)
from d_tools.experiments import apply_overrides, now_text
from d_tools.internal_state import internal_state_config
from internal_state_vae import train_common_internal_state_vae
from d_tools.run_paths import jsonable, make_run_output_dir, safe_stem
from run_cma_es_search import load_search_config
from f_run.run_liquid import LIQUID_RESULT_DIR, run_liquid
from f_run.run_training import build_cfg, build_network_cfg


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
OUT_ROOT = RESULTS_PATH / "fixed_vae_encoder_pretrain"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _search_output_dir(name: str) -> Path:
    timestamp = now_text().replace(":", "").replace("-", "").replace("T", "_")
    return OUT_ROOT / f"{safe_stem(name)}__{timestamp}"


def _load_pretrain_module(name_or_path: str):
    # VAE事前学習専用configを読み込む。CMA-ES設定とは分けて管理する。
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import VAE pretrain config file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    module_name = name_or_path
    if "." not in module_name:
        local_config = CONFIG_DIR / f"{module_name}.py"
        if local_config.exists():
            return _load_pretrain_module(str(local_config))
    return importlib.import_module(module_name)


def load_pretrain_config(name_or_path: str) -> dict:
    module = _load_pretrain_module(name_or_path)
    if not hasattr(module, "VAE_PRETRAIN"):
        raise AttributeError(f"{module.__name__} must define VAE_PRETRAIN.")
    cfg = deepcopy(getattr(module, "VAE_PRETRAIN"))
    cfg.setdefault("name", Path(name_or_path).stem)
    return cfg

def _load_parameter_values_module(name_or_path: str):
    # パラメータ群を定義したconfigを読み込む。
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import parameter values config: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    module_name = name_or_path
    if "." not in module_name:
        local_config = CONFIG_DIR / f"{module_name}.py"
        if local_config.exists():
            return _load_parameter_values_module(str(local_config))
    return importlib.import_module(module_name)


def _default_parameter_values(params) -> dict:
    # configで未指定の値はCMA-ES設定のx0から補う。
    return {param.name: param.value_from_unit(param.unit_from_value(param.x0)) for param in params}


def _range_values(spec) -> list[float]:
    if isinstance(spec, dict):
        start = float(spec.get("min", spec.get("start")))
        stop = float(spec.get("max", spec.get("stop")))
        step = float(spec.get("step", 0.0))
    elif isinstance(spec, (list, tuple)) and len(spec) == 3:
        start, stop, step = (float(spec[0]), float(spec[1]), float(spec[2]))
    else:
        raise TypeError("Each PARAMETER_RANGES entry must be {'min','max','step'} or [min, max, step].")

    if step < 0:
        raise ValueError("PARAMETER_RANGES step must be >= 0.")
    if abs(stop - start) < 1e-15:
        return [start]
    if step == 0:
        raise ValueError("PARAMETER_RANGES step must be > 0 when min != max.")

    direction = 1.0 if stop > start else -1.0
    signed_step = direction * abs(step)
    values = []
    current = start
    eps = abs(step) * 1e-9 + 1e-12
    if direction > 0:
        while current <= stop + eps:
            values.append(round(current, 12))
            current += signed_step
    else:
        while current >= stop - eps:
            values.append(round(current, 12))
            current += signed_step
    if abs(values[-1] - stop) > eps:
        values.append(round(stop, 12))
    return values


def _parameter_sets_from_ranges(module, defaults: dict, valid_names: set[str], name_or_path: str) -> list[dict]:
    ranges = dict(getattr(module, "PARAMETER_RANGES"))
    unknown = sorted(set(ranges) - valid_names)
    if unknown:
        raise KeyError(f"Unknown parameter name(s) in {name_or_path}: {unknown}")

    range_values = {name: _range_values(spec) for name, spec in ranges.items()}
    mode = str(getattr(module, "PARAMETER_RANGE_MODE", "zip")).lower()
    max_sets = int(getattr(module, "MAX_GENERATED_PARAMETER_SETS", 1000))
    sets = []

    if mode == "grid":
        names = list(range_values)
        total = 1
        for name in names:
            total *= len(range_values[name])
        if total > max_sets:
            raise ValueError(
                f"PARAMETER_RANGES grid generates {total} sets, larger than MAX_GENERATED_PARAMETER_SETS={max_sets}."
            )
        for index, combo in enumerate(product(*(range_values[name] for name in names)), start=1):
            values = dict(defaults)
            values.update(dict(zip(names, combo)))
            sets.append({"id": f"range_grid_{index:04d}", "memo": "generated from PARAMETER_RANGES grid", "values": values})
    elif mode == "random":
        names = list(range_values)
        sample_count = int(getattr(module, "RANDOM_SAMPLE_COUNT", max_sets))
        sample_count = max(1, min(sample_count, max_sets))
        seed = int(getattr(module, "RANDOM_SEED", 0))
        rng = np.random.default_rng(seed)
        seen = set()
        attempts = 0
        max_attempts = max(sample_count * 50, 1000)
        while len(sets) < sample_count and attempts < max_attempts:
            attempts += 1
            combo = tuple(float(rng.choice(range_values[name])) for name in names)
            if combo in seen:
                continue
            seen.add(combo)
            values = dict(defaults)
            values.update(dict(zip(names, combo)))
            index = len(sets) + 1
            sets.append(
                {
                    "id": f"range_random_{index:04d}",
                    "memo": f"random sample from PARAMETER_RANGES seed={seed}",
                    "values": values,
                }
            )
        if not sets:
            raise ValueError("PARAMETER_RANGE_MODE='random' generated no parameter sets.")
    elif mode == "zip":
        lengths = {name: len(vals) for name, vals in range_values.items()}
        n_sets = max(lengths.values(), default=0)
        if n_sets > max_sets:
            raise ValueError(
                f"PARAMETER_RANGES zip generates {n_sets} sets, larger than MAX_GENERATED_PARAMETER_SETS={max_sets}."
            )
        ids = list(getattr(module, "PARAMETER_SET_IDS", []))
        memos = list(getattr(module, "PARAMETER_SET_MEMOS", []))
        if ids and len(ids) > n_sets:
            raise ValueError("PARAMETER_SET_IDS length must be <= generated PARAMETER_RANGES length.")
        if memos and len(memos) > n_sets:
            raise ValueError("PARAMETER_SET_MEMOS length must be <= generated PARAMETER_RANGES length.")
        for index in range(n_sets):
            values = dict(defaults)
            for name, vals in range_values.items():
                # 候補数が短いパラメータは最後の値を使い回す。
                values[name] = vals[min(index, len(vals) - 1)]
            sets.append(
                {
                    "id": safe_stem(ids[index] if index < len(ids) else f"range_zip_{index + 1:04d}"),
                    "memo": str(memos[index] if index < len(memos) else "generated from PARAMETER_RANGES zip"),
                    "values": values,
                }
            )
    else:
        raise ValueError("PARAMETER_RANGE_MODE must be 'zip', 'grid', or 'random'.")

    return sets
def _parameter_sets_from_values_config(params, name_or_path: str) -> list[dict]:
    module = _load_parameter_values_module(name_or_path)

    defaults = _default_parameter_values(params)
    valid_names = {param.name for param in params}
    parameter_names = list(getattr(module, "PARAMETER_NAMES", []))
    sets = []

    if hasattr(module, "PARAMETER_RANGES"):
        sets.extend(_parameter_sets_from_ranges(module, defaults, valid_names, name_or_path))
    elif hasattr(module, "PARAMETER_VALUES"):
        parameter_values = dict(getattr(module, "PARAMETER_VALUES"))
        unknown = sorted(set(parameter_values) - valid_names)
        if unknown:
            raise KeyError(f"Unknown parameter name(s) in {name_or_path}: {unknown}")
        lengths = {name: len(values) for name, values in parameter_values.items()}
        unique_lengths = sorted(set(lengths.values()))
        if len(unique_lengths) != 1:
            raise ValueError(f"All PARAMETER_VALUES lists must have the same length: {lengths}")
        n_sets = unique_lengths[0]
        ids = list(getattr(module, "PARAMETER_SET_IDS", []))
        memos = list(getattr(module, "PARAMETER_SET_MEMOS", []))
        if ids and len(ids) != n_sets:
            raise ValueError("PARAMETER_SET_IDS length must match PARAMETER_VALUES length.")
        if memos and len(memos) != n_sets:
            raise ValueError("PARAMETER_SET_MEMOS length must match PARAMETER_VALUES length.")
        for index in range(n_sets):
            values = dict(defaults)
            for name, value_list in parameter_values.items():
                values[name] = value_list[index]
            sets.append(
                {
                    "id": safe_stem(ids[index] if ids else f"manual_{index + 1:03d}"),
                    "memo": str(memos[index] if memos else "manual numeric parameter set"),
                    "values": values,
                }
            )
    else:
        if not hasattr(module, "PARAMETER_SETS"):
            raise AttributeError(f"{name_or_path} must define PARAMETER_VALUES or PARAMETER_SETS.")
        for index, entry in enumerate(getattr(module, "PARAMETER_SETS"), start=1):
            set_id = f"manual_{index:03d}"
            memo = "manual numeric parameter set"

            if isinstance(entry, dict):
                raw_values = dict(entry.get("values", entry))
                set_id = str(entry.get("id", set_id))
                memo = str(entry.get("memo", memo))
            elif isinstance(entry, (list, tuple)):
                if not parameter_names:
                    raise AttributeError(
                        f"{name_or_path} uses list-style PARAMETER_SETS, so PARAMETER_NAMES is required."
                    )
                row = list(entry)
                if len(row) == len(parameter_names):
                    raw_values = dict(zip(parameter_names, row))
                elif len(row) == len(parameter_names) + 1:
                    set_id = str(row[0])
                    raw_values = dict(zip(parameter_names, row[1:]))
                elif len(row) == len(parameter_names) + 2:
                    set_id = str(row[0])
                    memo = str(row[1])
                    raw_values = dict(zip(parameter_names, row[2:]))
                else:
                    raise ValueError(
                        f"PARAMETER_SETS row {index} has {len(row)} values, but expected "
                        f"{len(parameter_names)}, {len(parameter_names) + 1}, or {len(parameter_names) + 2}."
                    )
            else:
                raise TypeError("Each PARAMETER_SETS entry must be a dict, list, or tuple.")

            values = dict(defaults)
            unknown = sorted(set(raw_values) - valid_names)
            if unknown:
                raise KeyError(f"Unknown parameter name(s) in {name_or_path}: {unknown}")
            values.update(raw_values)
            sets.append(
                {
                    "id": safe_stem(set_id),
                    "memo": memo,
                    "values": values,
                }
            )

    if bool(getattr(module, "INCLUDE_DEFAULT_PARAMETER_SET", False)):
        sets.insert(
            0,
            {
                "id": "base_x0",
                "memo": "CMA-ES initial parameter set added by INCLUDE_DEFAULT_PARAMETER_SET",
                "values": dict(defaults),
            },
        )

    if not sets:
        raise ValueError(f"{name_or_path} has no parameter set entries.")
    return sets

def _candidate_parameter_sets(params, mode: str) -> list[dict]:
    # base邵ｺ・ｯx0邵ｺ・ｰ邵ｺ莉｣ﾂ・｣ounds邵ｺ・ｯ陷ｷ繝ｻ繝ｱ郢晢ｽｩ郢晢ｽ｡郢晢ｽｼ郢ｧ・ｿ郢ｧ蜑・ｽｸ遏ｩ蜑・闕ｳ莨∝応邵ｺ・ｸ隰厄ｽｯ邵ｺ・｣邵ｺ貅ｽ縺帷ｹｧ繧奇ｽｿ・ｽ陷会｣ｰ邵ｺ蜷ｶ・狗ｸｲ繝ｻ
    x0 = [param.unit_from_value(param.x0) for param in params]
    sets = [
        {
            "id": "base_x0",
            "memo": "CMA-ES initial parameter set",
            "values": unit_vector_to_values(params, x0),
        }
    ]
    if mode != "bounds":
        return sets

    for param_index, param in enumerate(params):
        for label, unit_value in (("low", 0.0), ("high", 1.0)):
            x = list(x0)
            x[param_index] = unit_value
            sets.append(
                {
                    "id": f"{param.name}_{label}",
                    "memo": f"{param.name} at {label} bound",
                    "values": unit_vector_to_values(params, x),
                }
            )
    return sets


def _prepare_liquid_cfg(search_cfg: dict, values: dict, set_id: str, samples_per_material: int | None, materials: list[str] | None, firing_filter_cfg: dict | None = None) -> dict:
    # VAE闔蜿･辯戊氛・ｦ驗吝・縲堤ｸｺ・ｯ陷・ｽｺ陷牙ｸ幢ｽｱ・､郢ｧ蜑・ｽｽ・ｿ郢ｧ荳岩・邵ｲ竏墅懃ｹｧ・ｭ郢昴・繝ｩ陷繝ｻﾎ夊ｿ･・ｶ隲ｷ荵昶味邵ｺ莉｣・定将譎擾ｽｭ蛟･笘・ｹｧ荵敖繝ｻ
    cfg = build_cfg()
    cfg = apply_overrides(cfg, search_cfg.get("base_overrides", {}))
    cfg = apply_parameter_values(cfg, normalize_parameter_specs(search_cfg["parameters"], cfg), values)
    run_overrides = {
        "run.LIVE_PLOT_ENABLE": False,
        "run.LIVE_RASTER_ENABLE": False,
        "run.INTERNAL_STATE_ENABLE": True,
        "run.INTERNAL_STATE_PCA_ENABLE": False,
        "liquid.NUM_LIQUID_SAMPLE": int(samples_per_material),
    }
    if firing_filter_cfg and "brian_codegen_target" in firing_filter_cfg:
        run_overrides["run.BRIAN_CODEGEN_TARGET"] = str(firing_filter_cfg["brian_codegen_target"])
    if firing_filter_cfg and bool(firing_filter_cfg.get("enabled", True)):
        run_overrides.update(
            {
                "run.LIQUID_EARLY_STOP_ENABLE": True,
                "run.LIQUID_EARLY_STOP_REQUIRE_SAMPLE_SPIKES": bool(
                    firing_filter_cfg.get("exclude_zero_spike_samples", True)
                ),
                "run.LIQUID_SKIP_ZERO_SPIKE_SAMPLES": bool(
                    firing_filter_cfg.get(
                        "skip_zero_spike_samples",
                        firing_filter_cfg.get("exclude_zero_spike_samples", True),
                    )
                ),
                "run.LIQUID_EARLY_STOP_MIN_MEAN_RATE_HZ": float(
                    firing_filter_cfg.get("min_sample_mean_rate_hz", 0.0)
                ),
                "run.LIQUID_EARLY_STOP_MAX_MEAN_RATE_HZ": float(
                    firing_filter_cfg.get("max_mean_rate_hz", 120.0)
                ),
                "run.LIQUID_EARLY_STOP_MAX_POPULATION_PEAK_RATE_HZ": float(
                    firing_filter_cfg.get("max_population_peak_rate_hz", 300.0)
                ),
                "run.LIQUID_EARLY_STOP_ZERO_SPIKE_PATIENCE": int(
                    firing_filter_cfg.get("zero_spike_patience", 1)
                ),
                "run.LIQUID_EARLY_STOP_BIN_MS": float(
                    firing_filter_cfg.get("bin_ms", cfg["run"].get("INTERNAL_STATE_BIN_MS", 10.0))
                ),
            }
        )
    cfg = apply_overrides(cfg, run_overrides)
    if materials:
        cfg = apply_overrides(cfg, {"liquid.LIQUID_MAT": materials})
    cfg["experiment"] = {
        "name": "fixed_vae_encoder_pretrain",
        "id": safe_stem(f"fixed_vae_encoder_pretrain__{set_id}"),
        "trial_id": safe_stem(set_id),
        "target": "liquid",
        "memo": "pretrain fixed VAE encoder",
        "overrides": values,
    }
    return cfg


def _internal_state_file_rate_hz(npz_path: Path) -> tuple[float, float, int]:
    """内部状態1ファイルから平均発火率Hzと時間方向ピークHzを見積もる。"""
    with np.load(npz_path, allow_pickle=False) as data:
        x_state = np.asarray(data["x_state"], dtype=np.float64)
        unit_arr = data.get("unit")
        source_arr = data.get("source")
        bin_arr = data.get("bin_ms")

    if x_state.size == 0:
        return 0.0, 0.0, 0

    unit = str(unit_arr[0]) if unit_arr is not None and len(unit_arr) else ""
    source = str(source_arr[0]) if source_arr is not None and len(source_arr) else ""
    bin_ms = float(bin_arr[0]) if bin_arr is not None and len(bin_arr) else 10.0

    unit_l = unit.lower()
    source_l = source.lower()
    if "hz" in unit_l or "rate" in source_l:
        rate_hz = x_state
    elif "spikes/bin" in unit_l or "count" in source_l:
        rate_hz = x_state / max(bin_ms / 1000.0, 1e-12)
    else:
        # spike_bin_mean は spikes/ms で保存されるので Hz に直す。
        rate_hz = x_state * 1000.0

    mean_rate_hz = float(np.nanmean(rate_hz))
    peak_population_rate_hz = float(np.nanmax(np.nanmean(rate_hz, axis=0)))
    return mean_rate_hz, peak_population_rate_hz, int(x_state.shape[0])


def _candidate_firing_summary(internal_state_dir: Path) -> dict:
    """候補1つ分の内部状態ディレクトリから発火率サマリを作る。"""
    files = sorted(Path(internal_state_dir).glob("*/*_liquid_internal_state_all.npz"))
    if not files:
        files = sorted(Path(internal_state_dir).glob("*_liquid_internal_state_all.npz"))
    if not files:
        return {
            "n_files": 0,
            "mean_rate_hz": float("nan"),
            "min_file_mean_rate_hz": float("nan"),
            "max_file_mean_rate_hz": float("nan"),
            "max_population_peak_rate_hz": float("nan"),
            "zero_spike_file_count": 0,
            "mean_neurons": 0.0,
        }

    mean_rates = []
    peak_rates = []
    neuron_counts = []
    for fp in files:
        mean_rate, peak_rate, n_neurons = _internal_state_file_rate_hz(fp)
        mean_rates.append(mean_rate)
        peak_rates.append(peak_rate)
        neuron_counts.append(n_neurons)

    return {
        "n_files": int(len(files)),
        "mean_rate_hz": float(np.nanmean(mean_rates)),
        "min_file_mean_rate_hz": float(np.nanmin(mean_rates)),
        "max_file_mean_rate_hz": float(np.nanmax(mean_rates)),
        "max_population_peak_rate_hz": float(np.nanmax(peak_rates)),
        "zero_spike_file_count": int(np.sum(np.asarray(mean_rates, dtype=np.float64) <= 0.0)),
        "mean_neurons": float(np.nanmean(neuron_counts)),
    }


def _should_exclude_by_firing(summary: dict, filter_cfg: dict) -> tuple[bool, str]:
    """発火しすぎの候補をVAE学習から除外するか判定する。"""
    if not bool(filter_cfg.get("enabled", True)):
        return False, "filter disabled"
    if int(summary.get("n_files", 0)) <= 0:
        return True, "no internal-state files"

    exclude_zero = bool(filter_cfg.get("exclude_zero_spike_samples", True))
    min_mean = float(filter_cfg.get("min_mean_rate_hz", 0.0))
    min_file_mean = float(filter_cfg.get("min_file_mean_rate_hz", 0.0))
    max_mean = float(filter_cfg.get("max_mean_rate_hz", 120.0))
    max_file_mean = float(filter_cfg.get("max_file_mean_rate_hz", max_mean * 1.5))
    max_peak = float(filter_cfg.get("max_population_peak_rate_hz", 300.0))

    if exclude_zero and int(summary.get("zero_spike_file_count", 0)) > 0:
        return True, "zero_spike_sample_included"
    if float(summary.get("mean_rate_hz", 0.0)) < min_mean:
        return True, f"mean_rate_hz<{min_mean}"
    if float(summary.get("min_file_mean_rate_hz", 0.0)) < min_file_mean:
        return True, f"min_file_mean_rate_hz<{min_file_mean}"
    if float(summary.get("mean_rate_hz", 0.0)) > max_mean:
        return True, f"mean_rate_hz>{max_mean}"
    if float(summary.get("max_file_mean_rate_hz", 0.0)) > max_file_mean:
        return True, f"max_file_mean_rate_hz>{max_file_mean}"
    if float(summary.get("max_population_peak_rate_hz", 0.0)) > max_peak:
        return True, f"max_population_peak_rate_hz>{max_peak}"
    return False, "accepted"
def _cleanup_internal_state_dir(internal_state_dir: Path, *, reason: str) -> dict:
    """重い内部状態ディレクトリだけを安全に削除する。"""
    path = Path(internal_state_dir)
    row = {
        "internal_state_dir": str(path),
        "cleanup_reason": reason,
        "cleanup_status": "skipped",
        "cleanup_message": "",
    }
    if not path.exists():
        row["cleanup_status"] = "missing"
        row["cleanup_message"] = "path does not exist"
        return row

    resolved = path.resolve()
    results_root = RESULTS_PATH.resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError:
        row["cleanup_status"] = "blocked"
        row["cleanup_message"] = f"refuse to delete outside results root: {results_root}"
        return row

    if path.name != str(RUN_CFG.get("INTERNAL_STATE_DIR", "internal_states")):
        row["cleanup_status"] = "blocked"
        row["cleanup_message"] = "refuse to delete a directory that is not internal_states"
        return row

    try:
        shutil.rmtree(path)
        row["cleanup_status"] = "deleted"
        row["cleanup_message"] = "deleted internal_states directory"
    except Exception as exc:
        row["cleanup_status"] = "error"
        row["cleanup_message"] = f"{type(exc).__name__}: {exc}"
    return row



def _run_liquid_candidate_worker(payload: dict) -> dict:
    """1つのパラメータ候補を別プロセスで実行し、VAEに使えるか判定する。"""
    set_index = int(payload["set_index"])
    total_parameter_sets = int(payload["total_parameter_sets"])
    param_set = payload["param_set"]
    search_cfg = payload["search_cfg"]
    samples_per_material = int(payload["samples_per_material"])
    materials = payload["materials"]
    firing_filter_cfg = payload["firing_filter_cfg"]

    set_id = safe_stem(param_set["id"])
    candidate_key = f"pretrain_{set_index:03d}_{set_id}"
    cfg = _prepare_liquid_cfg(
        search_cfg,
        dict(param_set["values"]),
        set_id,
        samples_per_material,
        materials,
        firing_filter_cfg,
    )
    net_cfg = build_network_cfg(cfg)
    run_out_dir = make_run_output_dir(LIQUID_RESULT_DIR, cfg, net_cfg, include_output=False)
    message = run_liquid(cfg)
    internal_state_dir = run_out_dir / internal_state_config(cfg["run"])["dir_name"]
    entry = {
        "candidate_key": candidate_key,
        "generation": 0,
        "candidate_index": set_index,
        "run_out_dir": str(run_out_dir),
        "internal_state_dir": str(internal_state_dir),
        "params_json": json.dumps(jsonable(param_set["values"]), ensure_ascii=False),
    }
    firing_summary = _candidate_firing_summary(internal_state_dir)
    exclude, exclude_reason = _should_exclude_by_firing(firing_summary, firing_filter_cfg)
    row = {
        **entry,
        **firing_summary,
        "excluded_from_vae": bool(exclude),
        "exclude_reason": exclude_reason,
        "message": message,
    }
    return {
        "set_index": set_index,
        "total_parameter_sets": total_parameter_sets,
        "entry": entry,
        "row": row,
        "exclude": bool(exclude),
        "exclude_reason": exclude_reason,
        "firing_summary": firing_summary,
        "message": message,
    }


def run_pretrain(
    *,
    config_name: str,
    samples_per_material: int | None,
    pretrain_config_name: str,
    parameter_set_mode: str,
    parameter_values_config: str | None,
    materials: list[str] | None,
    vae_epochs: int | None,
    vae_latent_dim: int | None,
) -> dict:
    search_cfg = load_search_config(config_name)
    search_cfg = deepcopy(search_cfg)
    pretrain_cfg = load_pretrain_config(pretrain_config_name)
    if samples_per_material is None:
        samples_per_material = int(pretrain_cfg.get("samples_per_material", 100))
    else:
        samples_per_material = int(samples_per_material)
    base_cfg = build_cfg()
    base_cfg = apply_overrides(base_cfg, search_cfg.get("base_overrides", {}))
    params = normalize_parameter_specs(search_cfg["parameters"], base_cfg)
    if parameter_values_config:
        parameter_sets = _parameter_sets_from_values_config(params, parameter_values_config)
        parameter_set_mode = safe_stem(Path(parameter_values_config).stem)
    else:
        parameter_sets = _candidate_parameter_sets(params, parameter_set_mode)

    out_dir = _search_output_dir(f"{search_cfg.get('name', 'vae_encoder')}_{parameter_set_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "search_config.json", search_cfg)
    _write_json(out_dir / "vae_pretrain_config.json", pretrain_cfg)
    pd.DataFrame(parameter_sets).to_csv(out_dir / "pretrain_parameter_sets.csv", index=False)

    entries = []
    run_rows = []
    excluded_rows = []
    cleanup_rows = []
    firing_filter_cfg = dict(search_cfg.get("metric", {}).get("vae_pretrain_filter", {}))
    # VAE事前学習config側の設定を優先する。
    firing_filter_cfg.update(dict(pretrain_cfg.get("vae_pretrain_filter", {})))
    if "brian_codegen_target" in pretrain_cfg:
        firing_filter_cfg["brian_codegen_target"] = pretrain_cfg["brian_codegen_target"]
    cleanup_cfg = dict(search_cfg.get("metric", {}).get("vae_pretrain_cleanup", {}))
    cleanup_cfg.update(dict(pretrain_cfg.get("vae_pretrain_cleanup", {})))
    total_parameter_sets = len(parameter_sets)
    parallel_workers = max(1, int(pretrain_cfg.get("parallel_liquid_workers", 1)))

    def handle_candidate_result(result: dict) -> None:
        entry = result["entry"]
        row = result["row"]
        firing_summary = result["firing_summary"]
        exclude = bool(result["exclude"])
        exclude_reason = str(result["exclude_reason"])
        set_index = int(result["set_index"])
        print(
            f"[fixed-vae][candidate {set_index}/{total_parameter_sets}] {result['message']}",
            flush=True,
        )
        if exclude:
            excluded_rows.append(row)
            print(
                f"[fixed-vae][exclude-all-materials] {entry['candidate_key']} {exclude_reason} "
                f"mean={firing_summary['mean_rate_hz']:.3g}Hz "
                f"peak={firing_summary['max_population_peak_rate_hz']:.3g}Hz",
                flush=True,
            )
        else:
            entries.append(entry)
        run_rows.append(row)
        pd.DataFrame(run_rows).sort_values("candidate_index").to_csv(
            out_dir / "pretrain_liquid_runs.csv", index=False
        )
        if excluded_rows:
            pd.DataFrame(excluded_rows).sort_values("candidate_index").to_csv(
                out_dir / "pretrain_excluded_by_firing.csv", index=False
            )

    payloads = []
    for set_index, param_set in enumerate(parameter_sets, start=1):
        set_id = safe_stem(param_set["id"])
        candidate_key = f"pretrain_{set_index:03d}_{set_id}"
        memo = str(param_set.get("memo", ""))
        print(
            f"[fixed-vae][candidate {set_index}/{total_parameter_sets}] "
            f"queued {candidate_key} id={set_id} memo={memo}",
            flush=True,
        )
        payloads.append(
            {
                "set_index": set_index,
                "total_parameter_sets": total_parameter_sets,
                "param_set": param_set,
                "search_cfg": search_cfg,
                "samples_per_material": samples_per_material,
                "materials": materials,
                "firing_filter_cfg": firing_filter_cfg,
            }
        )

    if parallel_workers <= 1:
        for payload in payloads:
            print(
                f"[fixed-vae][candidate {payload['set_index']}/{total_parameter_sets}] running",
                flush=True,
            )
            handle_candidate_result(_run_liquid_candidate_worker(payload))
    else:
        print(
            f"[fixed-vae] running liquid candidates in parallel: workers={parallel_workers}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_payload = {executor.submit(_run_liquid_candidate_worker, payload): payload for payload in payloads}
            for future in as_completed(future_to_payload):
                payload = future_to_payload[future]
                try:
                    handle_candidate_result(future.result())
                except Exception as exc:
                    set_index = int(payload["set_index"])
                    set_id = safe_stem(payload["param_set"]["id"])
                    candidate_key = f"pretrain_{set_index:03d}_{set_id}"
                    row = {
                        "candidate_key": candidate_key,
                        "generation": 0,
                        "candidate_index": set_index,
                        "run_out_dir": "",
                        "internal_state_dir": "",
                        "params_json": json.dumps(jsonable(payload["param_set"]["values"]), ensure_ascii=False),
                        "file_count": 0,
                        "mean_rate_hz": float("nan"),
                        "min_file_mean_rate_hz": float("nan"),
                        "max_file_mean_rate_hz": float("nan"),
                        "max_population_peak_rate_hz": float("nan"),
                        "zero_spike_file_count": 0,
                        "mean_neurons": 0.0,
                        "excluded_from_vae": True,
                        "exclude_reason": f"worker_error:{type(exc).__name__}",
                        "message": str(exc),
                    }
                    run_rows.append(row)
                    excluded_rows.append(row)
                    pd.DataFrame(run_rows).sort_values("candidate_index").to_csv(
                        out_dir / "pretrain_liquid_runs.csv", index=False
                    )
                    pd.DataFrame(excluded_rows).sort_values("candidate_index").to_csv(
                        out_dir / "pretrain_excluded_by_firing.csv", index=False
                    )
                    print(
                        f"[fixed-vae][candidate {set_index}/{total_parameter_sets}] "
                        f"worker_error {candidate_key}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
    vae_cfg = dict(search_cfg.get("metric", {}).get("vae", {}))
    if vae_epochs is not None:
        vae_cfg["epochs"] = int(vae_epochs)
    if vae_latent_dim is not None:
        vae_cfg["latent_dim"] = int(vae_latent_dim)

    if not entries:
        raise RuntimeError("All parameter sets were excluded by the VAE pretrain firing-rate filter. Relax metric.vae_pretrain_filter thresholds.")

    fixed_encoder_dir = out_dir / "fixed_encoder_vae"
    vae_result = train_common_internal_state_vae(
        entries,
        fixed_encoder_dir,
        dataset_id="fixed_vae_encoder_pretrain",
        window_ms=float(vae_cfg.get("window_ms", 10.0)),
        step_ms=float(vae_cfg.get("step_ms", 10.0)),
        latent_dim=int(vae_cfg.get("latent_dim", 16)),
        hidden_channels=int(vae_cfg.get("hidden_channels", 64)),
        beta=float(vae_cfg.get("beta", 1e-3)),
        epochs=int(vae_cfg.get("epochs", 50)),
        batch_size=int(vae_cfg.get("batch_size", 32)),
        lr=float(vae_cfg.get("lr", 1e-3)),
        seed=int(vae_cfg.get("seed", 0)),
        device=str(vae_cfg.get("device", "auto")),
        standardize=bool(vae_cfg.get("standardize", True)),
        max_samples_per_class=int(samples_per_material),
        materials=materials,
    )
    if bool(cleanup_cfg.get("enabled", True)) and bool(cleanup_cfg.get("remove_used_after_training", True)):
        for entry in entries:
            cleanup_row = {
                **entry,
                **_cleanup_internal_state_dir(
                    Path(entry["internal_state_dir"]),
                    reason="used_after_vae_training",
                ),
            }
            cleanup_rows.append(cleanup_row)
        if cleanup_rows:
            pd.DataFrame(cleanup_rows).to_csv(out_dir / "pretrain_internal_state_cleanup.csv", index=False)
    summary = {
        "out_dir": str(out_dir),
        "fixed_encoder_dir": str(fixed_encoder_dir),
        "fixed_encoder_model_file": vae_result["model_file"],
        "samples_per_material": int(samples_per_material),
        "pretrain_config": pretrain_config_name,
        "parameter_set_mode": parameter_set_mode,
        "parameter_values_config": parameter_values_config or "",
        "parameter_set_count": len(parameter_sets),
        "vae_training_parameter_set_count": len(entries),
        "excluded_parameter_set_count": len(excluded_rows),
        "vae_pretrain_filter": firing_filter_cfg,
        "vae_pretrain_cleanup": cleanup_cfg,
        "cleanup_row_count": len(cleanup_rows),
        "vae_result": vae_result,
    }
    _write_json(out_dir / "fixed_encoder_summary.json", summary)
    print(f"[fixed-vae] encoder_dir={fixed_encoder_dir}")
    print(f"[fixed-vae] model_file={vae_result['model_file']}")
    print(f"[fixed-vae] latent_csv={vae_result.get('latent_csv_file', '')}")
    print(f"[fixed-vae] latent_npz={vae_result.get('latent_npz_file', '')}")
    print(f"[fixed-vae] latent_z1_z2_plot={vae_result.get('latent_plot_file', '')}")
    print(f"[fixed-vae] latent_pca2_plot={vae_result.get('latent_pca2_plot_file', '')}")
    print(f"[fixed-vae] latent_pca3_plot={vae_result.get('latent_pca3_plot_file', '')}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a fixed VAE Encoder for CMA-ES.")
    parser.add_argument("config", nargs="?", default="cma_es_internal_state", help="CMA-ES/search config name.")
    parser.add_argument("--pretrain-config", default="parameter_values_config", help="Config name that defines VAE_PRETRAIN.")
    parser.add_argument("--samples-per-material", type=int, default=None, help="Override samples_per_material in the VAE pretrain config.")
    parser.add_argument(
        "--parameter-set-mode",
        choices=("base", "bounds"),
        default="base",
        help="base: x0 only. bounds: x0 plus low/high for each liquid/input parameter.",
    )
    parser.add_argument(
        "--parameter-values-config",
        default="parameter_values_config",
        help="Python config file/module that defines PARAMETER_RANGES/PARAMETER_SETS for VAE pretrain.",
    )
    parser.add_argument("--materials", default=None, help="Comma-separated material names.")
    parser.add_argument("--vae-epochs", type=int, default=None)
    parser.add_argument("--vae-latent-dim", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    materials = None
    if args.materials:
        materials = [item.strip() for item in str(args.materials).split(",") if item.strip()]
    run_pretrain(
        config_name=args.config,
        samples_per_material=args.samples_per_material,
        pretrain_config_name=str(args.pretrain_config),
        parameter_set_mode=str(args.parameter_set_mode),
        parameter_values_config=args.parameter_values_config,
        materials=materials,
        vae_epochs=args.vae_epochs,
        vae_latent_dim=args.vae_latent_dim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())























