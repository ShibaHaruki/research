"""学習なしでリキッド内部状態を素材ごとに保存する実行スクリプト。"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
SAMPLE_SEQ_DIR = RESULTS_PATH / RUN_CFG["SAMPLE_SEQ_DIR"]
LIQUID_RESULT_DIR = RESULTS_PATH / RUN_CFG["LIQUID_RESULT_DIR"]

CACHE_DIR = RESULTS_PATH / RUN_CFG["CACHE_DIR"]
for cache_subdir in RUN_CFG["CACHE_SUBDIRS"]:
    (CACHE_DIR / cache_subdir).mkdir(parents=True, exist_ok=True)

for env_name, cache_subdir in RUN_CFG["ENV_CACHE_MAP"].items():
    os.environ[env_name] = str(CACHE_DIR / cache_subdir)

import matplotlib
if RUN_CFG.get("LIVE_PLOT_ENABLE", False):
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
    Network,
    SpikeMonitor,
    StateMonitor,
    TimedArray,
    defaultclock,
    ms,
    second,
    seed,
    start_scope,
)

from b_network.build_network import (
    make_in_neuron_group,
    make_in_to_liq_synapses,
    make_liq_intra_synapses,
    make_liquid_neuron_groups,
    make_poisson_to_liq_synapses,
)
from d_tools.sample_sequence import load_or_make_sample_seq
from d_tools.compat import liquid_sample_count
from d_tools.internal_state import (
    _binned_spike_state_from_snapshot,
    can_record_internal_state,
    capture_internal_state_snapshots,
    capture_voltage_state_snapshots,
    internal_state_config,
    save_internal_states,
    save_internal_states_from_snapshots,
)
from d_tools.internal_state_visualization import save_internal_state_heatmap
from d_tools.live_visualization import (
    LiquidLiveViewer,
    live_plot_chunk_steps,
    live_plot_enabled,
)
from d_tools.pca import save_internal_state_pca
from d_tools.weight_export import save_liquid_weight_matrices
from d_tools.run_paths import (
    make_run_output_dir,
    save_config_snapshot as save_run_config_snapshot,
    save_used_parameters_text,
)
from d_tools.visualization import (
    safe_stem,
    save_liquid_3d_plot,
    save_spike_raster,
    save_voltage_plot,
    save_weight_distributions,
    select_record_indices,
)
from f_run.run_training import (
    build_cfg,
    build_input_current,
    build_network_cfg,
    load_tactile_data,
    reset_dynamic_state,
    tqdm,
    _group_scalar,
    _input_layout,
    _limit_from_env,
    _rmtree_if_possible,
    _unlink_if_possible,
)


def make_liquid_output_dir(cfg: dict, net_cfg: dict) -> Path:
    return make_run_output_dir(LIQUID_RESULT_DIR, cfg, net_cfg, include_output=False)


def make_liquid_network(net_cfg: dict, N_in: int, rng: np.random.Generator, run_cfg: dict):
    # 出力層なしのリキッド単体ネットワークを作る。
    # 内部状態保存用に SpikeMonitor と全ニューロン記録用 StateMonitor も用意する。
    input_ta0 = TimedArray(np.zeros((2, N_in)), dt=float(net_cfg["dt_ms"]) * ms)
    G_in = make_in_neuron_group(N_in=N_in, input_ta=input_ta0)
    G_liq = make_liquid_neuron_groups(net_cfg, rng)

    S_in, _ = make_in_to_liq_synapses(G_in, G_liq, rng, net_cfg)
    G_poisson, S_poisson, _ = make_poisson_to_liq_synapses(G_liq, rng, net_cfg)
    S_intra, _ = make_liq_intra_synapses(G_liq, rng, net_cfg)

    n_voltage = int(run_cfg.get("DEBUG_VOLTAGE_NEURONS", 10))
    voltage_dt_ms = float(run_cfg.get("DEBUG_VOLTAGE_DT_MS", 1.0))

    M_liq_debug = [
        SpikeMonitor(group, name=f"M_liq_debug_L{layer_index + 1}")
        for layer_index, group in enumerate(G_liq)
    ]
    V_liq_indices = [
        select_record_indices(group, n_voltage, rng)
        for group in G_liq
    ]
    V_liq = [
        StateMonitor(
            group,
            "v",
            record=V_liq_indices[layer_index],
            dt=voltage_dt_ms * ms,
            name=f"V_liq_L{layer_index + 1}",
        )
        for layer_index, group in enumerate(G_liq)
    ]
    internal_state_dt_ms = float(run_cfg.get("INTERNAL_STATE_DT_MS", 1.0))
    V_liq_internal = [
        StateMonitor(
            group,
            "v",
            record=True,
            dt=internal_state_dt_ms * ms,
            name=f"V_liq_internal_L{layer_index + 1}",
        )
        for layer_index, group in enumerate(G_liq)
    ]
    for monitor in M_liq_debug + V_liq + V_liq_internal:
        monitor.active = False

    net = Network()
    net.add(G_in)
    net.add(*G_poisson)
    net.add(*G_liq)
    net.add(*(S_in + S_poisson + S_intra))
    net.add(*(M_liq_debug + V_liq + V_liq_internal))

    return {
        "net": net,
        "G_in": G_in,
        "G_poisson": G_poisson,
        "G_liq": G_liq,
        "S_in": S_in,
        "S_poisson": S_poisson,
        "S_intra": S_intra,
        "S_all": S_in + S_poisson + S_intra,
        "M_liq_debug": M_liq_debug,
        "V_liq": V_liq,
        "V_liq_indices": V_liq_indices,
        "V_liq_internal": V_liq_internal,
    }


def liquid_weight_groups(objects: dict) -> dict[str, list]:
    return {
        "input_to_liquid": objects["S_in"],
        "poisson_to_liquid": objects.get("S_poisson", []),
        "liquid_intra": objects["S_intra"],
    }


def reset_liquid_state(objects: dict, net_cfg: dict) -> None:
    reset_dynamic_state({"G_liq": objects["G_liq"], "G_out": [], "S_all": objects["S_all"]}, net_cfg)


def save_input_plots(
    debug_dir: Path,
    tag: str,
    mat: str,
    sid: int,
    t_array,
    input_current,
    input_filter_map: dict[int, list[str]],
) -> None:
    # 使った入力フィルタごとに入力電流の波形を debug/<素材名> に保存する。
    channels, filters = _input_layout(input_filter_map)
    t_ms = t_array * 1000.0
    n_filters = len(filters)

    for filter_index, filter_name in enumerate(filters):
        plt.figure(figsize=(10, 4))
        for channel_index, channel in enumerate(channels):
            row_index = channel_index * n_filters + filter_index
            if row_index >= input_current.shape[0]:
                continue
            plt.plot(t_ms, input_current[row_index], linewidth=1.0, label=f"ch{channel}")
        plt.xlabel("Time [ms]")
        plt.ylabel("Amplitude")
        plt.title(f"{mat} sid{sid} input | {filter_name}")
        if channels:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            debug_dir / f"{tag}_input_{safe_stem(filter_name)}.png",
            dpi=150,
        )
        plt.close()


def save_liquid_layout_debug(debug_dir: Path, objects: dict) -> None:
    layout_dir = Path(debug_dir) / "liquid_layout"
    layout_dir.mkdir(parents=True, exist_ok=True)

    for layer_index, group in enumerate(objects["G_liq"]):
        save_liquid_3d_plot(
            layout_dir / f"liquid_L{layer_index + 1}_3d.png",
            group,
            f"liquid L{layer_index + 1} neuron positions",
        )


def save_first_sample_debug(
    debug_dir: Path,
    mat: str,
    sid: int,
    t_array,
    input_current,
    input_filter_map: dict[int, list[str]],
    objects: dict,
    duration_ms: float,
    debug_spike_start_indices: list[int] | None = None,
    debug_voltage_start_indices: list[int] | None = None,
    debug_start_time=None,
):
    # 各素材の最初の1試行だけ、入力・ラスタ・膜電位を保存して動作確認できるようにする。
    debug_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{safe_stem(mat)}_sid{sid}"

    save_input_plots(debug_dir, tag, mat, sid, t_array, input_current, input_filter_map)

    spike_starts = debug_spike_start_indices or []
    for layer_index, monitor in enumerate(objects["M_liq_debug"]):
        save_spike_raster(
            debug_dir / f"{tag}_liquid_L{layer_index + 1}_raster.png",
            monitor,
            f"{mat} sid{sid} liquid L{layer_index + 1} raster",
            start_index=(
                spike_starts[layer_index]
                if layer_index < len(spike_starts)
                else None
            ),
            start_time=debug_start_time,
            duration_ms=duration_ms,
        )

    voltage_starts = debug_voltage_start_indices or []
    for layer_index, monitor in enumerate(objects["V_liq"]):
        save_voltage_plot(
            debug_dir / f"{tag}_liquid_L{layer_index + 1}_voltage.png",
            monitor,
            objects["V_liq_indices"][layer_index],
            f"{mat} sid{sid} liquid L{layer_index + 1} membrane voltage",
            start_index=(
                voltage_starts[layer_index]
                if layer_index < len(voltage_starts)
                else None
            ),
            start_time=debug_start_time,
            duration_ms=duration_ms,
            spike_monitor=objects["M_liq_debug"][layer_index],
            spike_start_index=(
                spike_starts[layer_index]
                if layer_index < len(spike_starts)
                else None
            ),
            spike_time=debug_start_time,
            spike_y=_group_scalar(objects["G_liq"][layer_index], "v_thr"),
        )

    return None


def save_sample_internal_state(
    internal_state_dir: Path,
    mat: str,
    sid: int,
    objects: dict,
    internal_state_cfg: dict,
    duration_ms: float,
    *,
    internal_state_executor: ThreadPoolExecutor | None = None,
    spike_start_indices: list[int] | None = None,
    voltage_start_indices: list[int] | None = None,
    start_time=None,
):
    # 1試行分の内部状態を保存する。source が電位なら電位、スパイクなら bin 化した活動を使う。
    if not internal_state_cfg.get("enabled", False):
        return None

    source = str(internal_state_cfg.get("source", "spike_bin_mean")).lower()
    voltage_sources = {"voltage_delta", "v_delta", "delta_v", "voltage", "v", "membrane_voltage"}
    if source in voltage_sources:
        layer_snapshots = capture_voltage_state_snapshots(
            objects["G_liq"],
            objects["V_liq_internal"],
            voltage_start_indices=voltage_start_indices,
            start_time=start_time,
            source=source,
        )
        for monitor in objects["V_liq_internal"]:
            monitor.resize(0)
        if internal_state_executor is None:
            save_internal_states_from_snapshots(
                internal_state_dir,
                rep=None,
                mat=mat,
                sid=sid,
                layer_snapshots=layer_snapshots,
                duration_ms=duration_ms,
                config=internal_state_cfg,
            )
            return None

        return internal_state_executor.submit(
            save_internal_states_from_snapshots,
            internal_state_dir,
            rep=None,
            mat=mat,
            sid=sid,
            layer_snapshots=layer_snapshots,
            duration_ms=duration_ms,
            config=internal_state_cfg,
        )

    if internal_state_executor is None:
        spike_cfg = dict(internal_state_cfg)
        save_internal_states(
            internal_state_dir,
            rep=None,
            mat=mat,
            sid=sid,
            groups=objects["G_liq"],
            spike_monitors=objects["M_liq_debug"],
            duration_ms=duration_ms,
            config=spike_cfg,
            spike_start_indices=spike_start_indices,
            start_time=start_time,
        )
        return None

    layer_snapshots = capture_internal_state_snapshots(
        objects["G_liq"],
        objects["M_liq_debug"],
        spike_start_indices=spike_start_indices,
        start_time=start_time,
        duration_ms=duration_ms,
        source=source,
    )
    return internal_state_executor.submit(
        save_internal_states_from_snapshots,
        internal_state_dir,
        rep=None,
        mat=mat,
        sid=sid,
        layer_snapshots=layer_snapshots,
        duration_ms=duration_ms,
        config=internal_state_cfg,
    )


def save_debug_internal_state_image(
    debug_dir: Path,
    mat: str,
    sid: int,
    objects: dict,
    internal_state_cfg: dict,
    duration_ms: float,
    *,
    spike_start_indices: list[int] | None = None,
    start_time=None,
) -> None:
    # 内部状態保存がスパイクベースのとき、各素材1枚だけ heatmap を debug に保存する。
    if not internal_state_cfg.get("enabled", False):
        return

    source = str(internal_state_cfg.get("source", "spike_bin_mean")).lower()
    spike_sources = {
        "spike_filter",
        "spike",
        "filtered_spike",
        "spike_bin_mean",
        "spike_mean_bin",
        "binned_spike_mean",
        "spike_bin_count",
        "spike_count_bin",
        "binned_spike_count",
        "spike_bin_rate",
        "spike_rate_bin",
        "binned_spike_rate",
    }
    if source not in spike_sources:
        return

    layer_snapshots = capture_internal_state_snapshots(
        objects["G_liq"],
        objects["M_liq_debug"],
        spike_start_indices=spike_start_indices,
        start_time=start_time,
        duration_ms=duration_ms,
        source=source,
    )
    arrays_by_layer = [
        _binned_spike_state_from_snapshot(
            snapshot,
            duration_ms=duration_ms,
            bin_ms=internal_state_cfg.get("bin_ms", 10.0),
            source=source,
        )
        for snapshot in layer_snapshots
    ]
    if not arrays_by_layer:
        return

    x_state = np.concatenate([arrays["x_state"] for arrays in arrays_by_layer], axis=0)
    t_ms = arrays_by_layer[0]["t_ms"]
    tag = f"{safe_stem(mat)}_sid{sid}"
    save_internal_state_heatmap(
        Path(debug_dir) / f"{tag}_internal_state_heatmap.png",
        x_state,
        t_ms,
        title=f"{mat} sid{sid}",
        max_neurons=x_state.shape[0],
        sort_by_activity=False,
        save_selected_csv=False,
    )


def _clear_debug_outputs(debug_dir: Path) -> None:
    # 前回の debug 図だけを整理する。内部状態本体は run_liquid の出力側で管理する。
    debug_dir = Path(debug_dir)
    if not debug_dir.exists():
        return
    for pattern in (
        "*_input*.png",
        "*_raster.png",
        "*_voltage.png",
        "*_liquid_L*_3d.png",
        "*_counts.png",
        "*_internal_state*.png",
        "*_internal_state*.pdf",
        "*_internal_state*.csv",
        "*_internal_state*.npz",
        "*_internal_state*.json",
    ):
        for fp in debug_dir.rglob(pattern):
            if fp.is_file():
                _unlink_if_possible(fp)
    weight_dir = debug_dir / "weights"
    if weight_dir.exists():
        for pattern in ("hist_*.png", "weights_summary_*.csv", "weights_summary_*.txt", "weights_*.npz"):
            for fp in weight_dir.glob(pattern):
                if fp.is_file():
                    _unlink_if_possible(fp)
    for internal_state_debug_dir in debug_dir.rglob("internal_state"):
        if internal_state_debug_dir.is_dir():
            _rmtree_if_possible(internal_state_debug_dir)


def run_liquid_worker():
    cfg = build_cfg()
    return run_liquid(cfg)


def run_liquid(cfg: dict | int | None = None, legacy_cfg: dict | None = None):
    # リキッド単体実行の本体。学習はせず、素材ごとに指定サンプル数の内部状態を保存する。
    if legacy_cfg is not None:
        cfg = legacy_cfg
    if cfg is None or not isinstance(cfg, dict):
        cfg = build_cfg()

    start_scope()

    common = cfg["common"]
    liquid = cfg.get("liquid", {})
    training = cfg["training"]
    run_cfg = cfg["run"]
    filter_funcs = cfg["filter_funcs"]
    input_filter_map = cfg["input_filter_map"]
    net_cfg = build_network_cfg(cfg)

    dt_s = float(common["dt_ms"])
    defaultclock.dt = float(net_cfg["dt_ms"]) * ms

    base_seed = int(common["BASE_SEED"])
    np.random.seed(base_seed)
    seed(base_seed)
    rng = np.random.default_rng(base_seed)

    out_dir = make_liquid_output_dir(cfg, net_cfg)
    # run_training と同じ階層ルールで保存し、後から設定を確認できるように snapshot も残す。
    save_run_config_snapshot(out_dir, cfg, net_cfg, include_output=False)
    debug_dir = out_dir / run_cfg["DEBUG_DIR"]
    weight_matrix_dir = out_dir / "weights"
    internal_state_dir = out_dir / internal_state_config(run_cfg)["dir_name"]
    _rmtree_if_possible(internal_state_dir)
    _clear_debug_outputs(debug_dir)
    save_used_parameters_text(debug_dir, cfg, net_cfg, include_output=False)
    save_used_parameters_text(debug_dir / "weights", cfg, net_cfg, include_output=False)
    save_used_parameters_text(debug_dir / "liquid_layout", cfg, net_cfg, include_output=False)
    save_used_parameters_text(weight_matrix_dir, cfg, net_cfg, include_output=False)
    save_used_parameters_text(weight_matrix_dir / "init", cfg, net_cfg, include_output=False)
    save_used_parameters_text(weight_matrix_dir / "final", cfg, net_cfg, include_output=False)
    if internal_state_config(run_cfg).get("enabled", False):
        save_used_parameters_text(internal_state_dir, cfg, net_cfg, include_output=False)

    sample_seq = load_or_make_sample_seq(
        # run_liquid 専用のサンプル順序を保存し、同じ入力を再現できるようにする。
        name="sample_seq_liquid",
        out_dir=SAMPLE_SEQ_DIR,
        rng=rng,
        n_samples=int(common["NUM_SAMPLE"]),
    )

    channels, _ = _input_layout(input_filter_map)
    N_in = sum(len(input_filter_map[ch]) for ch in channels)
    objects = make_liquid_network(net_cfg, N_in=N_in, rng=rng, run_cfg=run_cfg)
    _, internal_state_cfg = can_record_internal_state(objects["G_liq"], run_cfg)
    # 初期重みとニューロン配置を保存し、内部状態と一緒に条件を確認できるようにする。
    save_weight_distributions(
        debug_dir,
        liquid_weight_groups(objects),
        tag="init",
    )
    save_liquid_layout_debug(debug_dir, objects)
    save_liquid_weight_matrices(weight_matrix_dir / "init", objects)

    n_train_samples = liquid_sample_count(liquid, training, common)
    n_train_samples = min(n_train_samples, int(common["NUM_SAMPLE"]))
    n_train_samples = _limit_from_env(run_cfg["TRAIN_LIMIT_ENV"], n_train_samples)

    materials = list(liquid.get("LIQUID_MAT", training["TRAINING_MAT"]))
    materials = materials[: _limit_from_env(run_cfg["MAT_LIMIT_ENV"], len(materials))]

    processed_samples = 0
    debug_saved_materials: set[str] = set()
    live_allowed = live_plot_enabled(run_cfg)
    internal_state_executor = None
    internal_state_futures = []
    if internal_state_cfg.get("enabled", False) and bool(
        run_cfg.get("INTERNAL_STATE_ASYNC_SAVE", True)
    ):
        # 内部状態の npz 書き込みは重いので、必要なら別スレッドで保存する。
        internal_state_executor = ThreadPoolExecutor(
            max_workers=max(1, int(run_cfg.get("INTERNAL_STATE_ASYNC_WORKERS", 1))),
            thread_name_prefix="liq-intstate",
        )

    try:
        for mat_index, mat in enumerate(materials):
            iterator = tqdm(range(n_train_samples), desc=f"[liquid] {mat}")
            for sample_index in iterator:
                # 素材ごとにサンプルを読み、入力電流を作ってリキッドへ流す。
                sid = int(sample_seq[sample_index])
                file = load_tactile_data(mat, sid)
                if file is None:
                    continue

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
                reset_liquid_state(objects, net_cfg)
                objects["G_in"].namespace["input_ta"] = input_ta
                objects["G_in"].t_start = defaultclock.t

                debug_active = mat not in debug_saved_materials
                # debug は素材ごとの最初の1試行だけ。内部状態保存は全指定サンプルで行う。
                internal_state_active = bool(internal_state_cfg.get("enabled", False))
                internal_state_source = str(
                    internal_state_cfg.get("source", "spike_bin_mean")
                ).lower()
                internal_state_uses_spikes = internal_state_source in {
                    "spike_filter",
                    "spike",
                    "filtered_spike",
                    "spike_bin_mean",
                    "spike_mean_bin",
                    "binned_spike_mean",
                    "spike_bin_count",
                    "spike_count_bin",
                    "binned_spike_count",
                    "spike_bin_rate",
                    "spike_rate_bin",
                    "binned_spike_rate",
                }
                internal_state_uses_voltage = internal_state_source in {
                    "voltage_delta",
                    "v_delta",
                    "delta_v",
                    "voltage",
                    "v",
                    "membrane_voltage",
                }
                for monitor in objects["M_liq_debug"]:
                    monitor.active = debug_active or (
                        internal_state_active and internal_state_uses_spikes
                    )
                for monitor in objects["V_liq"]:
                    monitor.active = debug_active
                for monitor in objects["V_liq_internal"]:
                    monitor.active = internal_state_active and internal_state_uses_voltage
                sample_start_time = defaultclock.t
                debug_start_time = sample_start_time if debug_active else None
                internal_state_start_time = sample_start_time if internal_state_active else None
                sample_spike_start_indices = None
                if debug_active or (internal_state_active and internal_state_uses_spikes):
                    sample_spike_start_indices = [
                        len(monitor.t) for monitor in objects["M_liq_debug"]
                    ]
                debug_spike_start_indices = sample_spike_start_indices if debug_active else None
                internal_state_spike_start_indices = (
                    sample_spike_start_indices
                    if internal_state_active and internal_state_uses_spikes
                    else None
                )
                internal_state_voltage_start_indices = None
                if internal_state_active and internal_state_uses_voltage:
                    internal_state_voltage_start_indices = [
                        len(monitor.t) for monitor in objects["V_liq_internal"]
                    ]
                debug_voltage_start_indices = None
                if debug_active:
                    debug_voltage_start_indices = [
                        len(monitor.t) for monitor in objects["V_liq"]
                    ]

                live_active = debug_active and live_allowed and bool(
                    run_cfg.get("LIVE_PLOT_FIRST_SAMPLE_ONLY", True)
                )
                if live_active:
                    # GUI が使える場合は、この素材の最初の試行をリアルタイムに表示する。
                    viewer = LiquidLiveViewer(
                        groups=objects["G_liq"],
                        spike_monitors=objects["M_liq_debug"],
                        voltage_monitors=objects["V_liq"],
                        voltage_indices=objects["V_liq_indices"],
                        run_cfg=run_cfg,
                        spike_start_indices=debug_spike_start_indices,
                        voltage_start_indices=debug_voltage_start_indices,
                        start_time=debug_start_time,
                    )
                    chunk_steps = live_plot_chunk_steps(run_cfg, dt_s)
                    done_steps = 0
                    while done_steps < nt:
                        step_count = min(chunk_steps, nt - done_steps)
                        objects["net"].run(step_count * dt_s * second, namespace={"input_ta": input_ta})
                        done_steps += step_count
                        viewer.update(current_time_ms=done_steps * dt_s * 1000.0)
                else:
                    objects["net"].run(nt * dt_s * second, namespace={"input_ta": input_ta})

                processed_samples += 1
                duration_ms = nt * dt_s * 1000.0

                if internal_state_active:
                    # VAE/PCA/分離指標で使う内部状態を material ごとのフォルダへ保存する。
                    material_internal_state_dir = internal_state_dir / safe_stem(mat)
                    save_used_parameters_text(
                        material_internal_state_dir,
                        cfg,
                        net_cfg,
                        include_output=False,
                    )
                    future = save_sample_internal_state(
                        material_internal_state_dir,
                        mat,
                        sid,
                        objects,
                        internal_state_cfg,
                        duration_ms,
                        internal_state_executor=internal_state_executor,
                        spike_start_indices=internal_state_spike_start_indices,
                        voltage_start_indices=internal_state_voltage_start_indices,
                        start_time=internal_state_start_time,
                    )
                    if future is not None:
                        internal_state_futures.append(future)

                if debug_active:
                    # 目視確認用の図は debug/<素材名> に1試行分だけ保存する。
                    material_debug_dir = debug_dir / safe_stem(mat)
                    save_used_parameters_text(material_debug_dir, cfg, net_cfg, include_output=False)
                    future = save_first_sample_debug(
                        material_debug_dir,
                        mat,
                        sid,
                        t_array,
                        input_current,
                        input_filter_map,
                        objects,
                        duration_ms,
                        debug_spike_start_indices=debug_spike_start_indices,
                        debug_voltage_start_indices=debug_voltage_start_indices,
                        debug_start_time=debug_start_time,
                    )
                    if future is not None:
                        internal_state_futures.append(future)
                    save_debug_internal_state_image(
                        material_debug_dir,
                        mat,
                        sid,
                        objects,
                        internal_state_cfg,
                        duration_ms,
                        spike_start_indices=debug_spike_start_indices,
                        start_time=debug_start_time,
                    )
                    debug_saved_materials.add(mat)
    finally:
        if internal_state_executor is not None:
            for future in as_completed(internal_state_futures):
                future.result()
            internal_state_executor.shutdown(wait=True)

    save_weight_distributions(
        # 実行後の重みも保存する。run_liquid では学習しないため通常は初期値と同じ。
        debug_dir,
        liquid_weight_groups(objects),
        tag="final",
    )
    save_liquid_weight_matrices(weight_matrix_dir / "final", objects)
    if internal_state_cfg.get("enabled", False) and bool(
        run_cfg.get("INTERNAL_STATE_PCA_ENABLE", True)
    ):
        # 内部状態保存後、設定で有効なら PCA の図と CSV も続けて作る。
        pca_dir = out_dir / str(run_cfg.get("PCA_DIR", "pca"))
        save_used_parameters_text(pca_dir, cfg, net_cfg, include_output=False)
        try:
            pca_summary = save_internal_state_pca(
                internal_state_dir,
                pca_dir,
                feature_mode=str(run_cfg.get("INTERNAL_STATE_PCA_FEATURE_MODE", "flatten")),
                n_components=int(run_cfg.get("INTERNAL_STATE_PCA_COMPONENTS", 2)),
                standardize=bool(run_cfg.get("INTERNAL_STATE_PCA_STANDARDIZE", True)),
                max_samples_per_class=run_cfg.get("INTERNAL_STATE_PCA_MAX_SAMPLES_PER_CLASS"),
                window_start_ms=run_cfg.get("INTERNAL_STATE_PCA_WINDOW_START_MS"),
                window_end_ms=run_cfg.get("INTERNAL_STATE_PCA_WINDOW_END_MS"),
            )
            print(f"[pca] saved internal-state PCA to {pca_summary['out_dir']}")
        except Exception as exc:
            print(f"[warn] internal-state PCA failed: {type(exc).__name__}: {exc}")
    return f"[liquid] processed {processed_samples} samples in {out_dir}"


if __name__ == "__main__":
    cfg = build_cfg()
    print(run_liquid(cfg))
