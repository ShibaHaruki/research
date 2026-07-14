"""Common helpers for liquid-only runs."""

from __future__ import annotations

import glob
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from a_tactile.experiment_sets import COMMON_SETS, LIQUID_SETS, TEST_SETS, TRAINING_SETS
from c_configs.FIXED import (
    cfg_models,
    cfg_network,
    cfg_neuron_models,
    cfg_run,
    cfg_synapse_models,
)
from c_configs.FIXED.cfg_filter import (
    FILTER_FUNCS,
    FILTER_GAIN,
    INPUT_FILTER_MAP,
    OPT_FILTER_GAIN,
    SENSOR_GAIN,
)
from d_tools.compat import canonical_input_route, first_value as compat_first_value


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TACTILE_DATA_PATH = (PROJECT_ROOT / RUN_CFG["TACTILE_DATA_ROOT"]).resolve()


def _module_config(module, attr_name: str) -> dict:
    value = getattr(module, attr_name, {})
    return dict(value) if isinstance(value, dict) else {}


def _first_value(value, default=None):
    return compat_first_value(value, default)


def _fill_missing_model_params(net_cfg: dict, model_params: dict) -> None:
    for key, value in model_params.items():
        net_cfg.setdefault(key, deepcopy(value))


def _apply_shared_input_routes(net_cfg: dict) -> None:
    shared_cfg = net_cfg.pop("SHARED_IN_ROUTE", {})
    if not shared_cfg.get("enabled", False):
        return
    shared_filters = shared_cfg.get("filters", {})
    for (_, filter_name), route_cfg in net_cfg.get("IN_ROUTE", {}).items():
        shared = shared_filters.get(str(filter_name))
        if not shared:
            continue
        for layer_cfg in route_cfg.get("layers", {}).values():
            for key in ("p", "scale"):
                if key in shared:
                    layer_cfg[key] = deepcopy(shared[key])


def build_cfg():
    return {
        "common": COMMON_SETS["base"],
        "training": TRAINING_SETS["base"],
        "liquid": LIQUID_SETS["base"],
        "test": TEST_SETS["base"],
        "network": _module_config(cfg_network, "CFG_NETWORK"),
        "run": _module_config(cfg_run, "CFG_RUN"),
        "models": _module_config(cfg_models, "CFG_MODELS"),
        "neuron_models": _module_config(cfg_neuron_models, "CFG_NEURON_MODELS"),
        "synapse_models": _module_config(cfg_synapse_models, "CFG_SYNAPSE_MODELS"),
        "filter_funcs": FILTER_FUNCS,
        "input_filter_map": INPUT_FILTER_MAP,
        "sensor_gain": SENSOR_GAIN,
        "filter_gain": FILTER_GAIN,
        "opt_filter_gain": OPT_FILTER_GAIN,
    }


def _input_rows(input_filter_map: dict[int, list[str]]) -> list[tuple[int, str]]:
    channels = sorted(input_filter_map)
    if not channels:
        raise ValueError("input_filter_map is empty.")

    rows: list[tuple[int, str]] = []
    for ch in channels:
        for filter_name in input_filter_map[ch]:
            rows.append((int(ch), str(filter_name)))
    if not rows:
        raise ValueError("input_filter_map must define at least one filter.")
    return rows


def _input_layout(input_filter_map: dict[int, list[str]]) -> tuple[list[int], list[str]]:
    rows = _input_rows(input_filter_map)
    channels = sorted({ch for ch, _ in rows})
    filters = list(dict.fromkeys(filter_name for _, filter_name in rows))
    return channels, filters


def build_network_cfg(cfg: dict) -> dict:
    common = cfg["common"]
    model_cfg = cfg["models"]
    network_cfg = cfg["network"]
    input_filter_map = cfg["input_filter_map"]
    channels, filters = _input_layout(input_filter_map)

    neuron_model = model_cfg["NEURON_MODEL"]
    synapse_model = model_cfg["SYNAPSE_MODEL"]

    dt_s = float(common["dt_ms"])
    net_cfg = deepcopy(network_cfg)
    _apply_shared_input_routes(net_cfg)
    net_cfg.update(
        {
            "dt_ms": dt_s * 1000.0,
            "BASE_SEED": int(common["BASE_SEED"]),
            "neuron_model": neuron_model,
            "synapse_model": synapse_model,
            "USE_INPUT_CHANNELS": channels,
            "USE_INPUT_FILTERS": filters,
            "INPUT_ROWS": _input_rows(input_filter_map),
            "NUM_CHANNEL": len(channels),
        }
    )

    _fill_missing_model_params(net_cfg, cfg["neuron_models"].get(neuron_model, {}))
    _fill_missing_model_params(net_cfg, cfg["synapse_models"].get(synapse_model, {}))

    net_cfg["IN_ROUTE"] = canonical_input_route(
        net_cfg,
        input_filter_map=input_filter_map,
    )
    return net_cfg


def load_tactile_data(mat: str, sid: int, tactile_data_path=TACTILE_DATA_PATH):
    pattern = tactile_data_path / RUN_CFG["TACTILE_DATA_DIR_NAME"] / mat / f"data_{sid}_*.csv"
    matches = glob.glob(str(pattern))
    if not matches:
        print(f"[warn] no file matched: {pattern}")
        return None
    return matches[0]


def build_input_current(
    in_data_0,
    t_array,
    dt,
    input_filter_map,
    filter_funcs,
    sensor_gain=None,
    filter_gain=None,
    opt_filter_gain=None,
):
    channels = sorted(input_filter_map)
    nt = in_data_0.shape[1]
    total_filters = sum(len(input_filter_map[ch]) for ch in channels)
    input_current = np.zeros((total_filters, nt))
    sensor_gain = sensor_gain or {}
    filter_gain = filter_gain or {}
    opt_filter_gain = opt_filter_gain or {}

    row = 0
    for ch in channels:
        gain = float(sensor_gain.get(ch, sensor_gain.get(str(ch), 1.0)))
        data_ch = gain * in_data_0[ch, :]
        for filter_name in input_filter_map[ch]:
            gain_filter = float(filter_gain.get(filter_name, 1.0)) * float(
                opt_filter_gain.get(filter_name, 1.0)
            )
            input_current[row, :] = gain_filter * filter_funcs[filter_name](data_ch, t_array, dt)
            row += 1
    return input_current


def reset_dynamic_state(objects: dict, net_cfg: dict):
    for group in objects.get("G_liq", []):
        group.v = group.v
        group.I_merkel = 0
        group.I_meissner = 0
        group.I_RI = 0
        group.I_SI = 0
        group.I_USI = 0
        group.I_syn = group.I_syn
        group.H_syn = group.H_syn

    for synapses in objects.get("S_all", []):
        for var_name in (
            "Apre",
            "Apost",
            "Aplus1",
            "Aplus2",
            "Aminus1",
            "Aminus2",
            "Mpre",
            "Mpost",
            "Zi",
            "Zj",
            "Pi",
            "Pj",
            "Pij",
        ):
            if var_name in synapses.variables:
                setattr(synapses, var_name, 0)
        if "x_stp" in synapses.variables:
            synapses.x_stp = 1.0
        if "u_stp" in synapses.variables:
            synapses.u_stp = 0.0


def _group_scalar(group, name: str) -> float:
    return float(np.asarray(getattr(group, name)).reshape(-1)[0])


def _limit_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return max(0, min(default, int(raw)))


def _warn_cleanup_failed(path: Path, exc: OSError) -> None:
    print(f"[warn] could not clean old debug file: {path} ({type(exc).__name__}: {exc})")


def _unlink_if_possible(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        _warn_cleanup_failed(Path(path), exc)


def _rmtree_if_possible(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        _warn_cleanup_failed(Path(path), exc)
