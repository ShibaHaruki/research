"""リキッド・出力層の実行で共有する入力、設定、試行分割、ネットワーク実行。"""

from __future__ import annotations

import glob
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
from brian2 import prefs, start_scope, defaultclock, ms, seed
from b_network.build_network import make_liquid_network
from d_tools.debug import execute_trial, run_material_debug
from d_tools.weight_export import load_weights

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from a_tactile.experiment_sets import COMMON_SETS, LIQUID_SETS, TEST_SETS, TRAINING_SETS
from c_configs.FIXED import (
    cfg_models,
    cfg_learning_rule_models,
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


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TACTILE_DATA_PATH = (PROJECT_ROOT / RUN_CFG["TACTILE_DATA_ROOT"]).resolve()


def _module_config(module, attr_name: str) -> dict:
    value = getattr(module, attr_name, {})
    return dict(value) if isinstance(value, dict) else {}


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
        "learning_rule_models": _module_config(cfg_learning_rule_models, "CFG_LEARNING_RULE_MODELS"),
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
            "learning_rule": model_cfg.get("LEARNING_RULE_MODEL", "off"),
            "learning_rule_models": deepcopy(cfg.get("learning_rule_models", {})),
            "synapse_model": synapse_model,
            "USE_INPUT_CHANNELS": channels,
            "USE_INPUT_FILTERS": filters,
            "INPUT_ROWS": _input_rows(input_filter_map),
            "NUM_CHANNEL": len(channels),
        }
    )

    _fill_missing_model_params(net_cfg, cfg["neuron_models"].get(neuron_model, {}))
    _fill_missing_model_params(net_cfg, cfg["synapse_models"].get(synapse_model, {}))

    # Use only explicitly configured (sensor, filter) routes in input-row order.
    # Legacy route strings, filter aliases and IN_ROUTE_LAYERS are not supported.
    routes = net_cfg.get("IN_ROUTE", {})
    selected_routes = {}
    for key in net_cfg["INPUT_ROWS"]:
        if key not in routes:
            raise KeyError(f"IN_ROUTE must explicitly define {key}")
        route = routes[key]
        if not isinstance(route, dict) or not isinstance(route.get("layers"), dict) or not route["layers"]:
            raise ValueError(f"IN_ROUTE[{key}] must contain a nonempty 'layers' dictionary")
        selected_routes[key] = deepcopy(route)
    net_cfg["IN_ROUTE"] = selected_routes
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
    include_opt_filter_gain: bool = True,
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
            gain_filter = float(filter_gain.get(filter_name, 1.0))
            if include_opt_filter_gain:
                gain_filter *= float(opt_filter_gain.get(filter_name, 1.0))
            input_current[row, :] = gain_filter * filter_funcs[filter_name](data_ch, t_array, dt)
            row += 1
    return input_current


def reset_dynamic_state(objects: dict, net_cfg: dict):
    active_filters = {
        str(name).split("__", 1)[0]
        for name in net_cfg.get("USE_INPUT_FILTERS", [])
    }
    for group in objects.get("G_liq", []):
        group.v = group.v
        for filter_name in active_filters:
            current_name = f"I_{filter_name}"
            if hasattr(group, current_name):
                setattr(group, current_name, 0)
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


SNAPSHOT_SECTIONS = ('common', 'training', 'test', 'liquid', 'network', 'run', 'models',
                     'neuron_models', 'synapse_models', 'learning_rule_models')


def first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def encode(value):
    if isinstance(value, dict):
        if any(isinstance(key, tuple) for key in value):
            return {'__tuple_keys__': [[list(key), encode(item)] for key, item in value.items()]}
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def decode(value):
    if isinstance(value, dict):
        if '__tuple_keys__' in value:
            return {tuple(key): decode(item) for key, item in value['__tuple_keys__']}
        # IN_ROUTEの層番号はJSON読込後も文字列で使用できる。
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(encode(data), ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def snapshot(cfg):
    return {key: cfg[key] for key in SNAPSHOT_SECTIONS}


def load_cfg(folder):
    data = decode(json.loads((Path(folder)/'config_snapshot.json').read_text(encoding='utf-8')))
    cfg = deepcopy(build_cfg())
    for key in SNAPSHOT_SECTIONS:
        cfg[key] = data[key]
    return cfg


def validate_cfg(cfg):
    n_train = int(first(cfg['training']['NUM_TRAINING_SAMPLE']))
    n_test = int(first(cfg['test']['NUM_TEST_SAMPLE']))
    epochs = int(first(cfg['training']['NUM_TRAINING_EPOCH']))
    if min(n_train, n_test, epochs) < 1 or n_train+n_test > 324:
        raise ValueError('Require positive train/test samples and epochs; train+test <= 324')
    materials = list(cfg['training']['TRAINING_MAT'])
    if not materials or len(set(materials)) != len(materials) or materials != list(cfg['test']['TEST_MAT']):
        raise ValueError('Training/test materials must be the same unique, nonempty list')
    bin_ms = int(first(cfg['test']['BIN_STEPS'])) * float(cfg['common']['dt_ms']) * 1000
    duration = float(cfg['run'].get('READOUT_DURATION_MS', 500.))
    if bin_ms <= 0 or not 0 < duration <= 500 or not np.isclose(duration/bin_ms, round(duration/bin_ms)):
        raise ValueError('Trial duration must be <=500 ms and divisible by the recording bin width')
    if cfg['run'].get('READOUT_STATE_POLICY', 'carry') not in ('carry', 'reset_each_trial'):
        raise ValueError('READOUT_STATE_POLICY must be carry or reset_each_trial')
    if cfg['run'].get('MATERIAL_SWITCH_WASHOUT_ENABLE', False):
        raise ValueError('Readout workflow requires MATERIAL_SWITCH_WASHOUT_ENABLE=False')
    return materials, n_train, n_test, epochs, bin_ms, duration


def sample_split(cfg, rep):
    if int(rep) < 1:
        raise ValueError('rep must be >=1')
    _, n_train, n_test, _, _, _ = validate_cfg(cfg)
    rng = np.random.default_rng(int(cfg['common']['BASE_SEED'])+int(rep)-1)
    order = rng.permutation(np.arange(1, 325))
    return order[:n_train], order[n_train:n_train+n_test]


def build_runtime(cfg, *, learning):
    start_scope(); prefs.codegen.target = 'numpy'
    defaultclock.dt = float(cfg['common']['dt_ms'])*1000*ms
    base_seed = int(cfg['common']['BASE_SEED']); seed(base_seed); np.random.seed(base_seed)
    effective = deepcopy(cfg); effective['network']['output_enabled'] = True
    if not learning:
        effective['models']['LEARNING_RULE_MODEL'] = 'off'
    net_cfg = build_network_cfg(effective)
    objects = make_liquid_network(net_cfg, len(net_cfg['INPUT_ROWS']), np.random.default_rng(base_seed), cfg['run'])
    if not objects['G_out']:
        raise ValueError('At least one output layer is required')
    # 学習ではトレース・重みだけを使い、全試行の発火イベント蓄積を避ける。
    for monitor in objects['M_out']:
        monitor.active = not learning
    objects['net'].store('trial_initial')
    return objects


def run_trial(objects, cfg, material, sid, *, record):
    from f_run.run_calibration import filtered_trial
    _, _, _, _, bin_ms, duration = validate_cfg(cfg)
    if cfg['run'].get('READOUT_STATE_POLICY', 'carry') == 'reset_each_trial':
        learned = [np.asarray(syn.w[:]).copy() for syn in objects['S_out']]
        objects['net'].restore('trial_initial')
        for syn, values in zip(objects['S_out'], learned):
            syn.w = values
    result = execute_trial(objects, filtered_trial(material, int(sid), cfg),
        dt_ms=float(cfg['common']['dt_ms'])*1000, duration_ms=duration, bin_ms=bin_ms,
        record_liquid=False, record_output=record)
    if not record:
        return None
    return np.concatenate([data['counts'] for data in result['output']], axis=0)


def debug_readout_materials(cfg, weights_dir, out_dir):
    """学習・テストの設定を共通debugへ渡し、固定重みで各素材を1回確認する。"""
    from f_run.run_calibration import filtered_trial
    if not cfg['run'].get('DEBUG_EACH_MATERIAL', True):
        return []
    materials, _, _, _, bin_ms, duration = validate_cfg(cfg)
    sid = int(cfg['run'].get('DEBUG_TRIAL_ID', 203))
    if not 1 <= sid <= 324:
        raise ValueError('DEBUG_TRIAL_ID must be 1..324')
    def make_network():
        objects = build_runtime(cfg, learning=False)
        load_weights(weights_dir, objects)
        return objects
    return run_material_debug(materials, trial_id=sid, make_network=make_network,
        load_current=lambda material, trial_id: filtered_trial(material, trial_id, cfg),
        dt_ms=float(cfg['common']['dt_ms'])*1000, duration_ms=duration, bin_ms=bin_ms,
        out_dir=out_dir)


def add_training_arguments(parser):
    parser.add_argument('--rule', choices=['off', 'STDP', 'T_STDP', 'SRDP', 'BCPNN'], default=None)
    parser.add_argument('--materials', nargs='+', default=None)
    parser.add_argument('--train-samples', type=int, default=None)
    parser.add_argument('--test-samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--rep', type=int, default=None)
    parser.add_argument('--state-policy', choices=['carry', 'reset_each_trial'], default=None)
    parser.add_argument('--out-dir', type=Path, default=None)


def cfg_from_args(args):
    cfg = deepcopy(build_cfg())
    if args.rule is not None:
        cfg['models']['LEARNING_RULE_MODEL'] = args.rule
    if args.materials is not None:
        from f_run.run_calibration import SETTINGS
        if not set(args.materials) <= set(SETTINGS['materials']):
            raise ValueError('Unknown material')
        cfg['training']['TRAINING_MAT'] = args.materials
        cfg['test']['TEST_MAT'] = args.materials
    for value, section, key in [(args.train_samples, 'training', 'NUM_TRAINING_SAMPLE'),
                                 (args.test_samples, 'test', 'NUM_TEST_SAMPLE'),
                                 (args.epochs, 'training', 'NUM_TRAINING_EPOCH')]:
        if value is not None:
            cfg[section][key] = [value]
    if args.state_policy is not None:
        cfg['run']['READOUT_STATE_POLICY'] = args.state_policy
    if args.out_dir is not None:
        cfg['run']['READOUT_TRAIN_DIR'] = str(args.out_dir.resolve())
    validate_cfg(cfg)
    return cfg


def repetition_numbers(cfg, rep=None, rep_end=None):
    """未指定ならCOMMONのNUM_REPEATに従い、指定時は対象repだけを実行する。"""
    total = int(cfg['common'].get('NUM_REPEAT', 1))
    start = 1 if rep is None else int(rep)
    end = (total if rep is None else start) if rep_end is None else int(rep_end)
    if total < 1 or start < 1 or end < start:
        raise ValueError('Invalid NUM_REPEAT or repetition range')
    return list(range(start, end+1))
