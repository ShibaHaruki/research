"""STDP/SRDP などの学習を行い、学習後重みとデバッグ図を保存する入口。"""

import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
TACTILE_DATA_PATH = (PROJECT_ROOT / RUN_CFG["TACTILE_DATA_ROOT"]).resolve()
SAMPLE_SEQ_DIR = RESULTS_PATH / RUN_CFG["SAMPLE_SEQ_DIR"]
TRAINING_RESULT_DIR = RESULTS_PATH / RUN_CFG["TRAINING_RESULT_DIR"]

CACHE_DIR = RESULTS_PATH / RUN_CFG["CACHE_DIR"]
for cache_subdir in RUN_CFG["CACHE_SUBDIRS"]:
    (CACHE_DIR / cache_subdir).mkdir(parents=True, exist_ok=True)

for env_name, cache_subdir in RUN_CFG["ENV_CACHE_MAP"].items():
    os.environ[env_name] = str(CACHE_DIR / cache_subdir)

import glob
import multiprocessing as mp

import matplotlib

if RUN_CFG.get("LIVE_PLOT_ENABLE", True):
    try:
        matplotlib.use(RUN_CFG.get("LIVE_PLOT_BACKEND", "TkAgg"))
    except Exception:
        matplotlib.use(RUN_CFG["MATPLOTLIB_BACKEND"])
else:
    matplotlib.use(RUN_CFG["MATPLOTLIB_BACKEND"])
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import (
    BrianLogger,
    Network,
    SpikeMonitor,
    StateMonitor,
    TimedArray,
    defaultclock,
    ms,
    prefs,
    second,
    seed,
    start_scope,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from a_tactile.experiment_sets import COMMON_SETS, LIQUID_SETS, TEST_SETS, TRAINING_SETS
from b_network.build_network import (
    make_in_neuron_group,
    make_in_to_liq_synapses,
    make_liq_intra_synapses,
    make_liq_to_out_synapses,
    make_liquid_neuron_groups,
    make_output_neuron_groups,
    make_poisson_to_liq_synapses,
)
from d_tools.visualization import (
    safe_stem,
    save_liquid_3d_plot,
    save_spike_raster,
    save_voltage_plot,
    save_weight_distributions,
    select_record_indices,
)
from d_tools.sample_sequence import load_or_make_sample_seq_rep
from d_tools.compat import (
    canonical_input_route,
    first_value as compat_first_value,
    repeat_count,
    training_sample_count,
)
from d_tools.live_visualization import (
    SpikeRasterLiveViewer,
    WeightChangeLiveViewer,
    live_plot_chunk_steps,
    live_plot_enabled,
)
from d_tools.run_paths import make_run_output_dir, save_config_snapshot, save_used_parameters_text
from d_tools.weight_change import (
    append_weight_change_records,
    make_weight_change_tracker,
    save_weight_change_records,
    snapshot_weight_tracker,
    snapshot_weight_mean_delta_by_layer,
    weight_tracker_layers,
)
from d_tools.weight_export import save_weight_matrices_like_old_code
from c_configs.FIXED import (
    cfg_learning_rule_models,
    cfg_models,
    cfg_network,
    cfg_neuron_models,
    cfg_synapse_models,
)
from c_configs.FIXED.cfg_filter import FILTER_FUNCS, INPUT_FILTER_MAP


prefs.codegen.target = RUN_CFG["BRIAN_CODEGEN_TARGET"]
for log_name in RUN_CFG["BRIAN_SUPPRESS_LOG_NAMES"]:
    BrianLogger.suppress_name(log_name)


def _module_config(module, attr_name: str) -> dict:
    value = getattr(module, attr_name, {})
    return dict(value) if isinstance(value, dict) else {}


def _first_value(value, default=None):
    return compat_first_value(value, default)


def _fill_missing_model_params(net_cfg: dict, model_params: dict) -> None:
    # network 側や実験上書きで既に入っている値を、モデル初期値で上書きしない。
    for key, value in model_params.items():
        net_cfg.setdefault(key, deepcopy(value))


def build_cfg():
    # FIXED 設定、素材設定、フィルタ設定を1つの cfg にまとめる。
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
        "learning_rule_models": _module_config(
            cfg_learning_rule_models,
            "CFG_LEARNING_RULE_MODELS",
        ),
        "filter_funcs": FILTER_FUNCS,
        "input_filter_map": INPUT_FILTER_MAP,
    }


def _input_layout(input_filter_map: dict[int, list[str]]) -> tuple[list[int], list[str]]:
    channels = sorted(input_filter_map)
    if not channels:
        raise ValueError("input_filter_map is empty.")

    filters = list(input_filter_map[channels[0]])
    for ch in channels[1:]:
        if list(input_filter_map[ch]) != filters:
            raise ValueError(
                "Current input-to-liquid routing assumes every channel uses "
                "the same filters in the same order."
            )
    return channels, filters


def build_network_cfg(cfg: dict) -> dict:
    # Brian2 のネットワーク構築関数が必要とする形へ設定を変換する。
    common = cfg["common"]
    model_cfg = cfg["models"]
    network_cfg = cfg["network"]
    input_filter_map = cfg["input_filter_map"]
    channels, filters = _input_layout(input_filter_map)

    neuron_model = model_cfg["NEURON_MODEL"]
    synapse_model = model_cfg["SYNAPSE_MODEL"]
    learning_rule = model_cfg["LEARNING_RULE_MODEL"]

    dt_s = float(common["dt_ms"])
    net_cfg = deepcopy(network_cfg)
    net_cfg.update(
        {
        "dt_ms": dt_s * 1000.0,
        "BASE_SEED": int(common["BASE_SEED"]),
        "neuron_model": neuron_model,
        "synapse_model": synapse_model,
        "learning_rule": learning_rule,
        "USE_INPUT_FILTERS": filters,
        "NUM_CHANNEL": len(channels),
        "learning_rule_models": cfg["learning_rule_models"],
        }
    )

    _fill_missing_model_params(net_cfg, cfg["neuron_models"].get(neuron_model, {}))
    _fill_missing_model_params(net_cfg, cfg["synapse_models"].get(synapse_model, {}))

    net_cfg["USE_INPUT_FILTERS"] = filters
    net_cfg["NUM_CHANNEL"] = len(channels)
    net_cfg["IN_ROUTE"] = canonical_input_route(
        net_cfg,
        input_filter_map=input_filter_map,
    )

    return net_cfg


def load_tactile_data(mat: str, sid: int, tactile_data_path=TACTILE_DATA_PATH):
    pattern = tactile_data_path / RUN_CFG["TACTILE_DATA_DIR_NAME"] / mat / f"data_{sid}_*.csv"
    fp = glob.glob(str(pattern))
    if not fp:
        print(f"[warn] no file matched: {pattern}")
        return None
    return fp[0]


def build_input_current(in_data_0, t_array, dt, input_filter_map, filter_funcs):
    # 生の触覚データを、指定された Merkel / Meissner などの入力電流へ変換する。
    channels = sorted(input_filter_map)
    nt = in_data_0.shape[1]
    total_filters = sum(len(input_filter_map[ch]) for ch in channels)
    input_current = np.zeros((total_filters, nt))

    row = 0
    for ch in channels:
        data_ch = in_data_0[ch, :]
        for filter_name in input_filter_map[ch]:
            input_current[row, :] = filter_funcs[filter_name](data_ch, t_array, dt)
            row += 1

    return input_current


def make_training_network(
    net_cfg: dict,
    N_in: int,
    rng: np.random.Generator,
    run_cfg: dict | None = None,
):
    # 学習用ネットワークを作り、デバッグ用の SpikeMonitor / StateMonitor も接続する。
    input_ta0 = TimedArray(np.zeros((2, N_in)), dt=float(net_cfg["dt_ms"]) * ms)
    G_in = make_in_neuron_group(N_in=N_in, input_ta=input_ta0)
    G_liq = make_liquid_neuron_groups(net_cfg, rng)
    G_out = make_output_neuron_groups(net_cfg, rng)

    S_in, _ = make_in_to_liq_synapses(G_in, G_liq, rng, net_cfg)
    G_poisson, S_poisson, _ = make_poisson_to_liq_synapses(G_liq, rng, net_cfg)
    S_intra, _ = make_liq_intra_synapses(G_liq, rng, net_cfg)
    S_lo, _ = make_liq_to_out_synapses(G_liq, G_out, rng, net_cfg)

    run_cfg = run_cfg or {}
    n_voltage = int(run_cfg.get("DEBUG_VOLTAGE_NEURONS", 10))
    voltage_dt_ms = float(run_cfg.get("DEBUG_VOLTAGE_DT_MS", 1.0))

    M_liq_debug = [
        SpikeMonitor(group, name=f"M_liq_debug_L{layer_index + 1}")
        for layer_index, group in enumerate(G_liq)
    ]
    M_out_debug = [
        SpikeMonitor(group, name=f"M_out_debug_L{layer_index + 1}")
        for layer_index, group in enumerate(G_out)
    ]

    V_liq_indices = [
        select_record_indices(group, n_voltage, rng)
        for group in G_liq
    ]
    V_out_indices = [
        select_record_indices(group, n_voltage, rng)
        for group in G_out
    ]
    V_liq = [
        StateMonitor(
            group,
            "v",
            record=V_liq_indices[layer_index],
            dt=voltage_dt_ms * ms,
            name=f"V_liq_L{layer_index + 1}",
            when="before_resets",
        )
        for layer_index, group in enumerate(G_liq)
    ]
    V_out = [
        StateMonitor(
            group,
            "v",
            record=V_out_indices[layer_index],
            dt=voltage_dt_ms * ms,
            name=f"V_out_L{layer_index + 1}",
            when="before_resets",
        )
        for layer_index, group in enumerate(G_out)
    ]
    for monitor in M_liq_debug + M_out_debug + V_liq + V_out:
        monitor.active = False

    net = Network()
    net.add(G_in)
    net.add(*G_poisson)
    net.add(*G_liq)
    net.add(*G_out)
    net.add(*(S_in + S_poisson + S_intra + S_lo))
    net.add(*(M_liq_debug + M_out_debug + V_liq + V_out))

    return {
        "net": net,
        "G_in": G_in,
        "G_poisson": G_poisson,
        "G_liq": G_liq,
        "G_out": G_out,
        "S_in": S_in,
        "S_poisson": S_poisson,
        "S_intra": S_intra,
        "S_lo": S_lo,
        "S_all": S_in + S_poisson + S_intra + S_lo,
        "M_liq_debug": M_liq_debug,
        "M_out_debug": M_out_debug,
        "V_liq": V_liq,
        "V_liq_indices": V_liq_indices,
        "V_out": V_out,
        "V_out_indices": V_out_indices,
    }


def weight_synapse_groups(objects: dict) -> dict[str, list]:
    return {
        "input_to_liquid": objects["S_in"],
        "poisson_to_liquid": objects.get("S_poisson", []),
        "liquid_intra": objects["S_intra"],
        "liquid_to_output": objects["S_lo"],
    }


def reset_dynamic_state(objects: dict, net_cfg: dict):
    # 各試行の前に膜電位とシナプス電流、学習則のトレース変数を初期化する。
    for group in objects["G_liq"] + objects["G_out"]:
        group.v = group.v #net_cfg["v_reset"]
        group.I_merkel = 0
        group.I_meissner = 0
        group.I_pacinian = 0
        group.I_exc = group.I_exc
        group.I_inh = group.I_inh
        group.H_exc = group.H_exc
        group.H_inh = group.H_inh


    for synapses in objects["S_all"]:
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


def run_material_switch_washout(objects: dict, net_cfg: dict, run_cfg: dict, N_in: int) -> float:
    washout_ms = float(run_cfg.get("MATERIAL_SWITCH_WASHOUT_MS", 0.0))
    if washout_ms <= 0.0:
        return 0.0

    n_steps = max(1, int(round(washout_ms / float(net_cfg["dt_ms"]))))
    zero_input = np.zeros((n_steps + 1, int(N_in)), dtype=float)
    input_ta = TimedArray(zero_input, dt=float(net_cfg["dt_ms"]) * ms)
    objects["G_in"].namespace["input_ta"] = input_ta
    objects["G_in"].t_start = defaultclock.t
    objects["net"].run(washout_ms * ms, namespace={"input_ta": input_ta})
    reset_dynamic_state(objects, net_cfg)
    return washout_ms


def _group_scalar(group, name: str) -> float:
    return float(np.asarray(getattr(group, name)).reshape(-1)[0])


def save_first_sample_debug(
    debug_dir: Path,
    rep: int,
    mat: str,
    sid: int,
    t_array,
    input_current,
    input_filter_map: dict[int, list[str]],
    objects: dict,
    debug_spike_start_indices: dict[str, list[int]] | None = None,
    debug_voltage_start_indices: dict[str, list[int]] | None = None,
    debug_start_time=None,
):
    # 各素材の最初の1試行だけ、入力波形・ラスタ・膜電位を debug/<素材名> に保存する。
    debug_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rep{rep}_{safe_stem(mat)}_sid{sid}"
    channels, filters = _input_layout(input_filter_map)
    t_ms = t_array * 1000.0
    n_filters = len(filters)
    duration_ms = float(t_array.size) * float(t_ms[1] - t_ms[0]) if t_ms.size > 1 else 0.0

    for filter_index, filter_name in enumerate(filters):
        plt.figure(figsize=(10, 4))
        for channel_index, channel in enumerate(channels):
            row_index = channel_index * n_filters + filter_index
            if row_index >= input_current.shape[0]:
                continue
            plt.plot(t_ms, input_current[row_index], linewidth=1.0, label=f"ch{channel}")
        plt.xlabel("Time [ms]")
        plt.ylabel("Amplitude")
        plt.title(f"rep{rep} {mat} sid{sid} input | {filter_name}")
        if channels:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            debug_dir / f"{tag}_input_{safe_stem(filter_name)}.png",
            dpi=150,
        )
        plt.close()

    spike_starts = debug_spike_start_indices or {}
    liquid_spike_starts = spike_starts.get("M_liq_debug", [])
    for layer_index, monitor in enumerate(objects["M_liq_debug"]):
        save_spike_raster(
            debug_dir / f"{tag}_liquid_L{layer_index + 1}_raster.png",
            monitor,
            f"rep{rep} {mat} sid{sid} liquid L{layer_index + 1} raster",
            start_index=(
                liquid_spike_starts[layer_index]
                if layer_index < len(liquid_spike_starts)
                else None
            ),
            start_time=debug_start_time,
            duration_ms=duration_ms,
        )

    output_spike_starts = spike_starts.get("M_out_debug", [])
    for layer_index, monitor in enumerate(objects["M_out_debug"]):
        save_spike_raster(
            debug_dir / f"{tag}_output_L{layer_index + 1}_raster.png",
            monitor,
            f"rep{rep} {mat} sid{sid} output L{layer_index + 1} raster",
            start_index=(
                output_spike_starts[layer_index]
                if layer_index < len(output_spike_starts)
                else None
            ),
            start_time=debug_start_time,
            duration_ms=duration_ms,
        )

    voltage_starts = debug_voltage_start_indices or {}
    liquid_voltage_starts = voltage_starts.get("V_liq", [])
    for layer_index, monitor in enumerate(objects["V_liq"]):
        save_voltage_plot(
            debug_dir / f"{tag}_liquid_L{layer_index + 1}_voltage.png",
            monitor,
            objects["V_liq_indices"][layer_index],
            f"rep{rep} {mat} sid{sid} liquid L{layer_index + 1} membrane voltage",
            start_index=(
                liquid_voltage_starts[layer_index]
                if layer_index < len(liquid_voltage_starts)
                else None
            ),
            start_time=debug_start_time,
            duration_ms=duration_ms,
            spike_monitor=objects["M_liq_debug"][layer_index],
            spike_start_index=(
                liquid_spike_starts[layer_index]
                if layer_index < len(liquid_spike_starts)
                else None
            ),
            spike_time=debug_start_time,
            spike_y=_group_scalar(objects["G_liq"][layer_index], "v_thr"),
        )

    output_voltage_starts = voltage_starts.get("V_out", [])
    for layer_index, monitor in enumerate(objects["V_out"]):
        save_voltage_plot(
            debug_dir / f"{tag}_output_L{layer_index + 1}_voltage.png",
            monitor,
            objects["V_out_indices"][layer_index],
            f"rep{rep} {mat} sid{sid} output L{layer_index + 1} membrane voltage",
            start_index=(
                output_voltage_starts[layer_index]
                if layer_index < len(output_voltage_starts)
                else None
            ),
            start_time=debug_start_time,
            duration_ms=duration_ms,
            spike_monitor=objects["M_out_debug"][layer_index],
            spike_start_index=(
                output_spike_starts[layer_index]
                if layer_index < len(output_spike_starts)
                else None
            ),
            spike_time=debug_start_time,
            spike_y=_group_scalar(objects["G_out"][layer_index], "v_thr"),
        )


def save_liquid_layout_debug(
    debug_dir: Path,
    rep: int,
    objects: dict,
) -> None:
    layout_dir = Path(debug_dir) / "liquid_layout"
    layout_dir.mkdir(parents=True, exist_ok=True)

    for layer_index, group in enumerate(objects["G_liq"]):
        save_liquid_3d_plot(
            layout_dir / f"rep{rep}_liquid_L{layer_index + 1}_3d.png",
            group,
            f"rep{rep} liquid L{layer_index + 1} neuron positions",
        )


def _limit_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return max(0, min(default, int(raw)))


def _build_training_schedule(
    materials: list[str],
    sample_seq: np.ndarray,
    n_train_samples: int,
    rng: np.random.Generator,
) -> list[tuple[int, str, int, int]]:
    # 素材ごとに分けず、全素材×サンプルをまとめてシャッフルして学習順序を作る。
    schedule = [
        (mat_index, mat, sample_index, int(sample_seq[sample_index]))
        for mat_index, mat in enumerate(materials)
        for sample_index in range(n_train_samples)
    ]
    if not schedule:
        return schedule
    order = rng.permutation(len(schedule))
    return [schedule[int(idx)] for idx in order]


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


def _clear_files_if_possible(root: Path, patterns: tuple[str, ...]) -> None:
    root = Path(root)
    if not root.exists():
        return
    for pattern in patterns:
        for fp in root.glob(pattern):
            if fp.is_file():
                _unlink_if_possible(fp)


def _clear_debug_weight_outputs(debug_dir: Path) -> None:
    # 過去のデバッグ画像や旧形式の重み出力だけを消す。失敗しても実行は止めない。
    debug_dir = Path(debug_dir)
    root_weight_dir = debug_dir / "weights"
    _clear_files_if_possible(
        root_weight_dir,
        (
            "hist_*.png",
            "weights_summary_*.csv",
            "weights_summary_*.txt",
            "weights_*.npz",
        ),
    )
    root_layout_dir = debug_dir / "liquid_layout"
    if root_layout_dir.exists():
        _clear_files_if_possible(root_layout_dir, ("*.png",))

    if not debug_dir.exists():
        return

    for child in debug_dir.iterdir():
        if not child.is_dir():
            if child.name.endswith("_counts.png"):
                _unlink_if_possible(child)
            continue
        material_weight_dir = child / "weights"
        if material_weight_dir.exists():
            _rmtree_if_possible(material_weight_dir)
        for count_plot in child.glob("*_counts.png"):
            _unlink_if_possible(count_plot)
        for input_plot in child.glob("*_input*.png"):
            _unlink_if_possible(input_plot)
        for raster_plot in child.glob("*_raster.png"):
            _unlink_if_possible(raster_plot)
        for voltage_plot in child.glob("*_voltage.png"):
            _unlink_if_possible(voltage_plot)
        for liquid_layout_plot in child.glob("*_liquid_L*_3d.png"):
            _unlink_if_possible(liquid_layout_plot)


def run_training_worker(rep):
    cfg = build_cfg()
    return run_training(rep, cfg)


def run_training(rep: int, cfg: dict):
    # 1 rep 分の学習を実行する本体。内部状態は学習では保存しない設定にしている。
    start_scope()

    cfg = dict(cfg)
    cfg["run"] = dict(cfg["run"])
    cfg["run"]["INTERNAL_STATE_ENABLE"] = False

    common = cfg["common"]
    training = cfg["training"]
    run_cfg = cfg["run"]
    filter_funcs = cfg["filter_funcs"]
    input_filter_map = cfg["input_filter_map"]
    net_cfg = build_network_cfg(cfg)
    out_dir = make_run_output_dir(TRAINING_RESULT_DIR, cfg, net_cfg, include_output=True)
    save_config_snapshot(out_dir, cfg, net_cfg, include_output=True)
    debug_dir = out_dir / run_cfg["DEBUG_DIR"]
    spike_count_dir = out_dir / run_cfg["SPIKE_COUNT_DIR"]
    weight_change_dir = out_dir / run_cfg["WEIGHT_CHANGE_DIR"]
    weight_matrix_dir = out_dir / "weights"
    save_used_parameters_text(debug_dir, cfg, net_cfg, include_output=True)
    save_used_parameters_text(debug_dir / "weights", cfg, net_cfg, include_output=True)
    save_used_parameters_text(debug_dir / "liquid_layout", cfg, net_cfg, include_output=True)
    save_used_parameters_text(weight_change_dir, cfg, net_cfg, include_output=True)
    save_used_parameters_text(weight_matrix_dir, cfg, net_cfg, include_output=True)
    save_used_parameters_text(weight_matrix_dir / "init" / f"rep{rep}", cfg, net_cfg, include_output=True)
    save_used_parameters_text(weight_matrix_dir / "final" / f"rep{rep}", cfg, net_cfg, include_output=True)
    _clear_debug_weight_outputs(debug_dir)
    if spike_count_dir.exists():
        _rmtree_if_possible(spike_count_dir)

    dt_s = float(common["dt_ms"])
    defaultclock.dt = float(net_cfg["dt_ms"]) * ms

    base_seed = int(common["BASE_SEED"] + rep)
    np.random.seed(base_seed)
    seed(base_seed)
    rng = np.random.default_rng(base_seed)

    sample_seq = load_or_make_sample_seq_rep(
        rep=rep,
        out_dir=SAMPLE_SEQ_DIR,
        rng=rng,
        n_samples=int(common["NUM_SAMPLE"]),
    )

    channels, _ = _input_layout(input_filter_map)
    N_in = sum(len(input_filter_map[ch]) for ch in channels)
    objects = make_training_network(net_cfg, N_in=N_in, rng=rng, run_cfg=run_cfg)
    # 学習前の重み分布と旧コード互換の重み行列を保存する。
    save_weight_distributions(
        debug_dir,
        weight_synapse_groups(objects),
        tag=f"rep{rep}_init",
    )
    save_liquid_layout_debug(debug_dir, rep, objects)
    save_weight_matrices_like_old_code(
        weight_matrix_dir / "init" / f"rep{rep}",
        objects,
        net_cfg,
        cfg=cfg,
        sample_seq=sample_seq,
        rep=rep,
    )
    weight_tracker = make_weight_change_tracker(
        objects["S_lo"],
        rng,
        n_trace=int(run_cfg.get("WEIGHT_TRACE_SYNAPSES", 50)),
    )
    live_weight_before = snapshot_weight_tracker(weight_tracker)
    live_viewer = None
    live_raster_viewer = None
    live_elapsed_ms = 0.0

    n_train_samples = training_sample_count(training, common)
    n_train_samples = min(n_train_samples, int(common["NUM_SAMPLE"]))
    n_train_samples = _limit_from_env(run_cfg["TRAIN_LIMIT_ENV"], n_train_samples)

    materials = list(training["TRAINING_MAT"])
    materials = materials[: _limit_from_env(run_cfg["MAT_LIMIT_ENV"], len(materials))]
    training_schedule = _build_training_schedule(
        materials,
        sample_seq,
        n_train_samples,
        rng,
    )

    weight_summary_rows = []
    weight_trace_rows = []
    debug_saved_materials: set[str] = set()
    live_allowed = live_plot_enabled(run_cfg)
    processed_samples = 0
    previous_material = None
    print(f"[live-plot] training enabled={live_allowed}")

    iterator = tqdm(training_schedule, desc=f"[rep{rep}] training")
    for mat_index, mat, sample_index, sid in iterator:
        # 1試行分の触覚データを読み込み、入力電流を作ってネットワークへ流す。
        file = load_tactile_data(mat, sid)
        if file is None:
            continue

        material_debug_dir = debug_dir / safe_stem(mat)

        df = pd.read_table(file, header=None)
        df_np = df.to_numpy().T
        in_data_0 = df_np[
            : int(common["NUM_SENSOR"]),
            int(common["SLICE_START"]) : int(common["SLICE_END"]),
        ]

        nt = in_data_0.shape[1]
        t_array = np.arange(nt) * dt_s
        input_current = build_input_current(
            in_data_0=in_data_0,
            t_array=t_array,
            dt=dt_s,
            input_filter_map=input_filter_map,
            filter_funcs=filter_funcs,
        )

        input_ta = TimedArray(input_current.T, dt=float(net_cfg["dt_ms"]) * ms)
        if (
            bool(run_cfg.get("MATERIAL_SWITCH_WASHOUT_ENABLE", False))
            and previous_material is not None
            and mat != previous_material
        ):
            live_elapsed_ms += run_material_switch_washout(objects, net_cfg, run_cfg, N_in)
        reset_dynamic_state(objects, net_cfg)
        objects["G_in"].namespace["input_ta"] = input_ta
        objects["G_in"].t_start = defaultclock.t

        debug_active = mat not in debug_saved_materials
        # debug_active は素材ごとの最初の1試行だけ True。LIVE_PLOT_FIRST_SAMPLE_ONLY もここで効く。
        live_first_only = bool(run_cfg.get("LIVE_PLOT_FIRST_SAMPLE_ONLY", True))
        live_active = live_allowed and (debug_active if live_first_only else True)
        live_raster_active = live_active and bool(run_cfg.get("LIVE_RASTER_ENABLE", True))
        for monitor in objects["M_liq_debug"] + objects["M_out_debug"]:
            monitor.active = debug_active or live_raster_active
        for monitor in objects["V_liq"] + objects["V_out"]:
            monitor.active = debug_active
        debug_start_time = defaultclock.t if debug_active else None
        debug_spike_start_indices = None
        debug_voltage_start_indices = None
        if debug_active:
            debug_spike_start_indices = {
                "M_liq_debug": [len(monitor.t) for monitor in objects["M_liq_debug"]],
                "M_out_debug": [len(monitor.t) for monitor in objects["M_out_debug"]],
            }
            debug_voltage_start_indices = {
                "V_liq": [len(monitor.t) for monitor in objects["V_liq"]],
                "V_out": [len(monitor.t) for monitor in objects["V_out"]],
            }

        weight_before = snapshot_weight_tracker(weight_tracker)
        if live_active:
            # リアルタイム表示では短い chunk ごとに run し、平均重み変動とラスタを更新する。
            if live_viewer is None:
                print(
                    f"[live-plot] opening viewer rep{rep} mat={mat} "
                    f"sample_index={sample_index} sid={sid}"
                )
                liquid_layers, output_layers = weight_tracker_layers(weight_tracker)
                live_viewer = WeightChangeLiveViewer(
                    liquid_layers=liquid_layers,
                    output_layers=output_layers,
                    run_cfg=run_cfg,
                )
                live_viewer.update(
                    current_time_ms=live_elapsed_ms,
                    liquid_values={layer: 0.0 for layer in liquid_layers},
                    output_values={layer: 0.0 for layer in output_layers},
                )
            if live_raster_active and live_raster_viewer is None:
                print(
                    f"[live-plot] opening raster viewer rep{rep} mat={mat} "
                    f"sample_index={sample_index} sid={sid}"
                )
                live_raster_viewer = SpikeRasterLiveViewer(
                    liquid_groups=objects["G_liq"],
                    liquid_monitors=objects["M_liq_debug"],
                    output_groups=objects["G_out"],
                    output_monitors=objects["M_out_debug"],
                    run_cfg=run_cfg,
                    liquid_start_indices=[
                        len(monitor.t) for monitor in objects["M_liq_debug"]
                    ],
                    output_start_indices=[
                        len(monitor.t) for monitor in objects["M_out_debug"]
                    ],
                )
            chunk_steps = live_plot_chunk_steps(run_cfg, dt_s)
            done_steps = 0
            while done_steps < nt:
                step_count = min(chunk_steps, nt - done_steps)
                objects["net"].run(step_count * dt_s * second, namespace={"input_ta": input_ta})
                done_steps += step_count
                current_live_time_ms = live_elapsed_ms + done_steps * dt_s * 1000.0
                liquid_delta, output_delta = snapshot_weight_mean_delta_by_layer(
                    weight_tracker,
                    live_weight_before,
                )
                live_viewer.update(
                    current_time_ms=current_live_time_ms,
                    liquid_values=liquid_delta,
                    output_values=output_delta,
                )
                if live_raster_active and live_raster_viewer is not None:
                    live_raster_viewer.update(current_time_ms=current_live_time_ms)
        else:
            objects["net"].run(nt * dt_s * second, namespace={"input_ta": input_ta})
        live_elapsed_ms += nt * dt_s * 1000.0

        append_weight_change_records(
            # この試行で変化した平均重みを記録する。スパイク数カウントはここでは行わない。
            weight_tracker,
            weight_before,
            rep=rep,
            mat_index=mat_index,
            mat=mat,
            sample_index=sample_index,
            sid=sid,
            summary_rows=weight_summary_rows,
            trace_rows=weight_trace_rows,
        )
        processed_samples += 1
        previous_material = mat

        if debug_active:
            save_used_parameters_text(material_debug_dir, cfg, net_cfg, include_output=True)
            save_first_sample_debug(
                material_debug_dir,
                rep,
                mat,
                sid,
                t_array,
                input_current,
                input_filter_map,
                objects,
                debug_spike_start_indices=debug_spike_start_indices,
                debug_voltage_start_indices=debug_voltage_start_indices,
                debug_start_time=debug_start_time,
            )
            debug_saved_materials.add(mat)

    save_weight_change_records(
        # 学習後に、重み変動 CSV/図と最終重み分布・重み行列を保存する。
        weight_change_dir,
        rep,
        weight_summary_rows,
        weight_trace_rows,
        max_trace_lines=int(run_cfg.get("WEIGHT_TRACE_PLOT_LINES", 12)),
    )
    save_weight_distributions(
        debug_dir,
        weight_synapse_groups(objects),
        tag=f"rep{rep}_final",
    )
    save_weight_matrices_like_old_code(
        weight_matrix_dir / "final" / f"rep{rep}",
        objects,
        net_cfg,
        cfg=cfg,
        sample_seq=sample_seq,
        rep=rep,
    )
    return f"[rep{rep}] processed {processed_samples} training samples in {out_dir}"


if __name__ == "__main__":
    cfg = build_cfg()
    common = cfg["common"]
    run_cfg = cfg["run"]

    reps = list(range(1, repeat_count(common, 1) + 1))
    if len(reps) == 1 or os.environ.get(run_cfg["NO_MP_ENV"]) == "1":
        for rep in reps:
            print(run_training_worker(rep))
        raise SystemExit(0)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=min(os.cpu_count() or 1, len(reps)),
        mp_context=ctx,
    ) as executor:
        futures = [executor.submit(run_training_worker, rep) for rep in reps]

        for future in as_completed(futures):
            print(future.result())
