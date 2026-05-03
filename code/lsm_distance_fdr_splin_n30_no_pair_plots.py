# -*- coding: utf-8 -*-
"""
Liquid層の内部状態を
    x_i(t) = (h * s_i)(t)
    h(t) = (1/tau_s) exp(-t/tau_s) H(t)
として定義し、リザーバニューロンのスパイク列から内部状態を作る。

このコードで作る比較:
    1. Al_board vs Al_board
    2. Al_board vs wood_board
    3. Al_board vs washi
    4. Al_board vs rubber_board

残す処理:
    - 各ペアPNG保存はしない
    - 距離の時間推移まとめは保存しない
    - サンプルごとの平均内部状態ヒストグラム
    - 線形分離特性 SP_lin(t) = rank(M_s(t)) の時間推移
    - Fisher分離度 FDR(t) の時間推移
    - npz保存

変更:
    - 素材サンプル数を N_COMPARE = 30 に増やす
    - SP_lin は残す
    - SP_lin のランク評価行列サイズを全比較で同じにする
      Al-Al: Al 30サンプル -> 30x500
      Al-Target: Al 15サンプル + Target 15サンプル -> 30x500
    - SPpw という名前の追加処理は入れない

出力先:
    liquid_state_spike_filter_distance_pair_inputs_rep1/
"""

from __future__ import annotations

import glob
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from brian2 import *


# =========================================
# Brian2 設定
# =========================================
prefs.core.default_float_dtype = float64
prefs.codegen.target = "numpy"


# =========================================
# パス設定
# =========================================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

DATA_ROOT = SCRIPT_PATH.parents[1]
TACTILE_DATA_DIR = DATA_ROOT / "tactile_data"


# =========================================
# 比較する素材
# =========================================
MAT_BASE = "Al_board"

TARGET_MATERIALS = [
    "wood_board",
    "washi",
    "rubber_board",
]

DISPLAY_NAMES = {
    "Al_board": "Al_board",
    "wood_board": "wood_board",
    "washi": "washi",
    "rubber_board": "rubber_board",
}


# =========================================
# 実験設定
# =========================================
REP = 1
BASE_SEED = 2

N_TRAIN = 100
N_SAMPLE_TEST = 100

# 各素材から使うサンプル数
# 30にすると:
#   Al-Al      = 30C2 = 435 pairs
#   Al-Target  = 30*30 = 900 pairs
N_COMPARE = 100

# 距離平均ヒストグラムでは、ランダム抽出せず全ペアを使う
# N_COMPARE=30 の場合:
#   Al-Al     : 30C2 = 435 pairs
#   Al-Target : 30*30 = 900 pairs

# ヒストグラム透明度
HIST_ALPHA = 0.28

# リザーバ全体のニューロン数
N_RES = 1000

# 評価に使うニューロン数
N_EVAL_NEURONS = 500

# 入力ニューロン数
N_IN = 2

# Brian2のシミュレーション刻み
DT_MS = 0.1

# 内部状態 x_i(t) を保存する時間刻み
STATE_DT_MS = 1.0

# x_i(t) = h * s_i の時定数
TAU_STATE_MS = 20.0

# 距離
# rms: sqrt(mean((x-y)^2))
# euclidean: sqrt(sum((x-y)^2))
DISTANCE_MODE = "rms"

# 内部状態距離グラフのy軸上限
DIST_YLIM_MAX = 0.07

# 触覚データの切り出し範囲
DATA_START = 3000
DATA_END = 8000

# 保存解像度
SAVE_DPI = 300


# =========================================
# 保存フォルダ
# =========================================
RESULT_DIR = SCRIPT_DIR / f"liquid_state_spike_filter_distance_pair_inputs_rep{REP}"
SUMMARY_DIR = RESULT_DIR / "summary"
PAIR_BASE_BASE_DIR = RESULT_DIR / f"pair_plots_{MAT_BASE}_vs_{MAT_BASE}"
PAIR_TARGET_DIRS = {
    target: RESULT_DIR / f"pair_plots_{MAT_BASE}_vs_{target}"
    for target in TARGET_MATERIALS
}

for d in [RESULT_DIR, SUMMARY_DIR, PAIR_BASE_BASE_DIR, *PAIR_TARGET_DIRS.values()]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================
# input filter
# =========================================
def calc_meissner(data: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            I[0, i] = I[0, i - 1] + 1 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1e-3))
            I[3, i] = I[0, i]
    return I[3, :]


def calc_merkel(data: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            if dF_dt < 0:
                dF_dt = 0
            I[0, i] = I[0, i - 1] + 0.74 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1 * 1e-3))
            I[3, i] = I[0, i] + I[1, i] + I[2, i]
    return I[3, :]


# =========================================
# データ読み込み + フィルタ
# =========================================
def load_input_current_and_filters(material: str, sample_id: int) -> dict:
    files = glob.glob(str(TACTILE_DATA_DIR / material / f"data_{int(sample_id)}_*"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"data not found: material={material}, sample={int(sample_id)}, dir={TACTILE_DATA_DIR}"
        )

    fp = files[0]
    df = pd.read_table(fp, header=None)
    df_np = df.to_numpy().T

    in_data_0 = df_np[:3, DATA_START:DATA_END]

    nt = in_data_0.shape[1]
    dt_s = DT_MS * 1e-3
    t_array_s = np.arange(nt) * dt_s
    time_ms = np.arange(nt) * DT_MS

    input_current = np.zeros((N_IN, nt), dtype=float)

    ch = 0
    in_data = in_data_0[ch, :]

    I_merkel = calc_merkel(in_data, t_array_s, dt_s)
    I_meissner = calc_meissner(in_data, t_array_s, dt_s)

    input_current[ch * 2, :] = 0.4 * I_merkel * 0.02
    input_current[ch * 2 + 1, :] = 0.6 * 7.3 * I_meissner * 0.02

    return {
        "raw_ch0": in_data.astype(np.float64),
        "merkel": I_merkel.astype(np.float64),
        "meissner": I_meissner.astype(np.float64),
        "input_current": input_current.astype(np.float64),
        "time_ms": time_ms.astype(np.float64),
        "file": str(fp),
    }


# =========================================
# 出力層なしLSM構築
# =========================================
def build_reservoir(*, rng: np.random.Generator):
    defaultclock.dt = DT_MS * ms

    N_in = N_IN
    N_res = N_RES

    p_in = 0.2
    p_res = 0.5

    v_reset = -65
    v_thr = -40

    tau_r = 2 * ms
    tau_d = 20 * ms

    BIAS = -65
    G = 0.25

    w_in = (
        rng.standard_normal((N_in, N_res))
        * (rng.random((N_in, N_res)) < p_in)
        / np.sqrt(N_in * p_in)
    )

    variance = (N_res * p_res**2) ** -1
    w_res_init = (
        rng.standard_normal((N_res, N_res))
        * (rng.random((N_res, N_res)) < p_res)
        * np.sqrt(variance)
    )

    for k in range(N_res):
        QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
        if len(QS) > 0:
            w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

    LIF = """
    dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
    I_exc : 1
    I_inh : 1
    tau_m : second
    t_ref : second
    """

    double_exp_res = """
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = G * R : 1
    """

    on_pre_res = """
    H_post += (w_res / (tau_r * tau_d)) / Hz
    """

    eqs_res = double_exp_res + LIF

    G_res = NeuronGroup(
        N_res,
        eqs_res,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
        namespace={
            "tau_r": tau_r,
            "tau_d": tau_d,
            "G": G,
            "BIAS": BIAS,
            "v_reset": v_reset,
            "v_thr": v_thr,
        },
    )

    G_res.tau_m = 10 * ms
    G_res.t_ref = 2 * ms
    G_res.I_inh = 0

    S_res = Synapses(
        G_res,
        G_res,
        model="w_res : 1",
        on_pre=on_pre_res,
        method="euler",
        namespace={
            "tau_r": tau_r,
            "tau_d": tau_d,
        },
    )

    S_res.connect(condition="i != j")
    S_res.w_res = w_res_init[S_res.i, S_res.j]
    S_res.delay = 0 * ms

    input_ta = TimedArray(np.zeros((1, N_in)), dt=DT_MS * ms)

    G_in = NeuronGroup(
        N_in,
        """
        t_start : second (shared)
        I = input_ta(t - t_start, i) : 1
        """,
        method="euler",
        namespace={"input_ta": input_ta},
    )

    G_in.t_start = 0 * ms

    S_in = Synapses(
        G_in,
        G_res,
        model="""
        w : 1
        I_exc_post = w * I_pre : 1 (summed)
        """,
        method="euler",
    )

    pre_in, post_in = np.nonzero(w_in)
    if len(pre_in) > 0:
        S_in.connect(i=pre_in, j=post_in)
        S_in.w = w_in[pre_in, post_in]

    net = Network(G_in, G_res, S_in, S_res)

    params = {
        "N_res": int(N_RES),
        "N_eval_neurons": int(N_EVAL_NEURONS),
        "N_in": int(N_IN),
        "p_in": float(p_in),
        "p_res": float(p_res),
        "dt_ms": float(DT_MS),
        "state_dt_ms": float(STATE_DT_MS),
        "tau_state_ms": float(TAU_STATE_MS),
        "tau_r_ms": float(tau_r / ms),
        "tau_d_ms": float(tau_d / ms),
        "G": float(G),
        "BIAS": float(BIAS),
        "v_reset": float(v_reset),
        "v_thr": float(v_thr),
        "w_in_shape": list(w_in.shape),
        "w_res_shape": list(w_res_init.shape),
    }

    return net, G_in, G_res, S_in, S_res, params


# =========================================
# スパイク列から内部状態 x_i(t) を作る
# =========================================
def spikes_to_filtered_state(
    *,
    spike_times_ms: np.ndarray,
    spike_indices: np.ndarray,
    eval_neuron_ids: np.ndarray,
    duration_ms: float,
    dt_state_ms: float,
    tau_state_ms: float,
    use_normalized_kernel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    n_eval = len(eval_neuron_ids)
    n_steps = int(np.floor(duration_ms / dt_state_ms)) + 1

    time_ms = np.arange(n_steps, dtype=np.float64) * dt_state_ms
    x_state = np.zeros((n_eval, n_steps), dtype=np.float64)

    id_to_row = {int(neuron_id): row for row, neuron_id in enumerate(eval_neuron_ids)}

    spike_bins = np.floor(spike_times_ms / dt_state_ms).astype(int)
    spike_count = np.zeros((n_eval, n_steps), dtype=np.float64)

    for t_bin, neuron_id in zip(spike_bins, spike_indices):
        if 0 <= t_bin < n_steps:
            row = id_to_row.get(int(neuron_id))
            if row is not None:
                spike_count[row, t_bin] += 1.0

    decay = np.exp(-dt_state_ms / tau_state_ms)

    if use_normalized_kernel:
        spike_scale = 1.0 / tau_state_ms
    else:
        spike_scale = 1.0

    for t in range(1, n_steps):
        x_state[:, t] = decay * x_state[:, t - 1] + spike_scale * spike_count[:, t]

    return x_state, time_ms


# =========================================
# 1サンプル分をLSMに入力して内部状態を作る
# =========================================
def run_one_sample(
    *,
    material: str,
    sample_id: int,
    net,
    G_in,
    G_res,
    eval_neuron_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict, int]:

    payload = load_input_current_and_filters(material, int(sample_id))

    input_current = payload["input_current"]
    nt = input_current.shape[1]
    duration_ms = nt * DT_MS

    vals = input_current.T
    vals = np.vstack([vals, vals[-1]])

    input_ta = TimedArray(vals, dt=DT_MS * ms)

    current_t = net.t
    G_in.t_start = current_t

    v_reset = -65

    G_res.v = v_reset
    G_res.R = 0
    G_res.H = 0
    G_res.I_inh = 0

    sp_mon = SpikeMonitor(G_res)
    net.add(sp_mon)

    net.run(
        duration_ms * ms,
        namespace={
            "input_ta": input_ta,
        },
    )

    spike_times_ms = np.asarray((sp_mon.t - current_t) / ms, dtype=np.float64)
    spike_indices = np.asarray(sp_mon.i, dtype=np.int64)
    n_spikes = int(sp_mon.num_spikes)

    net.remove(sp_mon)

    x_state, time_ms = spikes_to_filtered_state(
        spike_times_ms=spike_times_ms,
        spike_indices=spike_indices,
        eval_neuron_ids=eval_neuron_ids,
        duration_ms=duration_ms,
        dt_state_ms=STATE_DT_MS,
        tau_state_ms=TAU_STATE_MS,
        use_normalized_kernel=True,
    )

    print(
        f"{material} sample={int(sample_id)} "
        f"input_max={float(np.max(input_current)):.6g} "
        f"spikes={n_spikes} "
        f"x_max={float(np.max(x_state)):.6g}"
    )

    return x_state, time_ms, payload, n_spikes


# =========================================
# 距離計算
# =========================================
def calc_distance_over_time(a: np.ndarray, b: np.ndarray, mode: str = "rms") -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}")

    diff = a - b

    if mode == "rms":
        return np.sqrt(np.mean(diff * diff, axis=0))

    if mode == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=0))

    raise ValueError(f"unknown distance mode: {mode}")


# =========================================
# 軸範囲計算
# =========================================
def add_margin_to_ylim(ymin: float, ymax: float, margin_ratio: float = 0.08) -> tuple[float, float]:
    ymin = float(ymin)
    ymax = float(ymax)

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return 0.0, 1.0

    if ymin == ymax:
        if ymin == 0:
            return 0.0, 1.0
        pad = abs(ymin) * 0.1
        return ymin - pad, ymax + pad

    span = ymax - ymin
    pad = span * margin_ratio
    return ymin - pad, ymax + pad


def compute_global_plot_limits(
    *,
    time_ms_ref: np.ndarray,
    filter_base: list[dict],
    filters_targets: dict[str, list[dict]],
    states_base: np.ndarray,
    states_targets: dict[str, np.ndarray],
) -> dict:

    all_payloads = []
    all_payloads.extend(filter_base)
    for target_material in TARGET_MATERIALS:
        all_payloads.extend(filters_targets[target_material])

    input0_values = []
    input1_values = []

    for payload in all_payloads:
        input0_values.append(payload["input_current"][0])
        input1_values.append(payload["input_current"][1])

    input0_all = np.concatenate(input0_values)
    input1_all = np.concatenate(input1_values)

    input0_ylim = add_margin_to_ylim(np.min(input0_all), np.max(input0_all))
    input1_ylim = add_margin_to_ylim(np.min(input1_all), np.max(input1_all))

    mean_state_values = []

    for k in range(states_base.shape[0]):
        mean_state_values.append(np.mean(states_base[k], axis=0))

    for target_material in TARGET_MATERIALS:
        states_target = states_targets[target_material]
        for k in range(states_target.shape[0]):
            mean_state_values.append(np.mean(states_target[k], axis=0))

    mean_state_all = np.concatenate(mean_state_values)

    mean_state_ylim = add_margin_to_ylim(np.min(mean_state_all), np.max(mean_state_all))

    mean_state_ylim = (
        min(0.0, mean_state_ylim[0]),
        max(mean_state_ylim[1], 1e-6),
    )

    xlim = (float(time_ms_ref[0]), float(time_ms_ref[-1]))

    limits = {
        "xlim": xlim,
        "input0_ylim": input0_ylim,
        "input1_ylim": input1_ylim,
        "mean_state_ylim": mean_state_ylim,
        "distance_ylim": (0.0, float(DIST_YLIM_MAX)),
    }

    print("\n[plot limits]")
    print(f"xlim             = {limits['xlim']}")
    print(f"input0_ylim      = {limits['input0_ylim']}")
    print(f"input1_ylim      = {limits['input1_ylim']}")
    print(f"mean_state_ylim  = {limits['mean_state_ylim']}")
    print(f"distance_ylim    = {limits['distance_ylim']}")
    print()

    return limits


# =========================================
# 1ペアごとのPNG保存
# =========================================
def save_one_pair_distance_plot(
    *,
    time_ms_state: np.ndarray,
    dist: np.ndarray,
    state_a: np.ndarray,
    state_b: np.ndarray,
    payload_a: dict,
    payload_b: dict,
    name_a: str,
    name_b: str,
    title: str,
    out_path: Path,
    plot_limits: dict,
):

    plt.rcParams["font.size"] = 12

    time_ms_input = payload_a["time_ms"]

    input_a_0 = payload_a["input_current"][0]
    input_a_1 = payload_a["input_current"][1]

    input_b_0 = payload_b["input_current"][0]
    input_b_1 = payload_b["input_current"][1]

    state_trace_a = np.mean(state_a, axis=0)
    state_trace_b = np.mean(state_b, axis=0)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9.5, 12.0),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.1, 1.2]},
    )

    ax = axes[0]
    ax.plot(time_ms_input, input_a_0, linewidth=1.8, label=f"{name_a} input 1")
    ax.plot(time_ms_input, input_b_0, linewidth=1.8, linestyle="--", label=f"{name_b} input 1")
    ax.set_xlim(plot_limits["xlim"])
    ax.set_ylim(plot_limits["input0_ylim"])
    ax.set_title("Input current 1: Merkel-based input", fontsize=13)
    ax.set_ylabel("Input current", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    ax.plot(time_ms_input, input_a_1, linewidth=1.8, label=f"{name_a} input 2")
    ax.plot(time_ms_input, input_b_1, linewidth=1.8, linestyle="--", label=f"{name_b} input 2")
    ax.set_xlim(plot_limits["xlim"])
    ax.set_ylim(plot_limits["input1_ylim"])
    ax.set_title("Input current 2: Meissner-based input", fontsize=13)
    ax.set_ylabel("Input current", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    ax = axes[2]
    ax.plot(time_ms_state, state_trace_a, linewidth=2.0, label=f"{name_a} mean liquid state")
    ax.plot(time_ms_state, state_trace_b, linewidth=2.0, linestyle="--", label=f"{name_b} mean liquid state")
    ax.set_xlim(plot_limits["xlim"])
    ax.set_ylim(plot_limits["mean_state_ylim"])
    ax.set_title("Mean liquid-state trajectories", fontsize=13)
    ax.set_ylabel("Mean liquid state", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    ax = axes[3]
    ax.plot(time_ms_state, dist, linewidth=2.2, label="Liquid-state distance")

    dist_mean_time = float(np.mean(dist))
    ax.axhline(
        dist_mean_time,
        linewidth=2.0,
        linestyle=":",
        label=f"Distance time mean = {dist_mean_time:.5f}",
    )

    ax.set_xlim(plot_limits["xlim"])
    ax.set_ylim(plot_limits["distance_ylim"])
    ax.set_title("Distance between filtered liquid states", fontsize=13)
    ax.set_xlabel("Time [ms]", fontsize=13)

    if DISTANCE_MODE == "rms":
        ax.set_ylabel("RMS distance", fontsize=12)
    else:
        ax.set_ylabel("Euclidean distance", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig.savefig(out_path, bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)


# =========================================
# Al-Al 全組み合わせ保存
# =========================================
def save_base_base_pair_distance_plots(
    *,
    time_ms: np.ndarray,
    states_base: np.ndarray,
    filter_base: list[dict],
    selected_seq: np.ndarray,
    plot_limits: dict,
):
    rows = []
    base_label = DISPLAY_NAMES.get(MAT_BASE, MAT_BASE)

    for i, j in tqdm(
        list(combinations(range(N_COMPARE), 2)),
        desc=f"Saving {MAT_BASE}-{MAT_BASE} pair plots",
    ):
        dist = calc_distance_over_time(states_base[i], states_base[j], mode=DISTANCE_MODE)

        name_a = f"{base_label} {i:02d} ID={int(selected_seq[i])}"
        name_b = f"{base_label} {j:02d} ID={int(selected_seq[j])}"

        title = f"{base_label}-{base_label} comparison\n{name_a} vs {name_b}"

        out_path = PAIR_BASE_BASE_DIR / (
            f"{MAT_BASE}_{i:02d}_id{int(selected_seq[i])}"
            f"_vs_{MAT_BASE}_{j:02d}_id{int(selected_seq[j])}.png"
        )

        save_one_pair_distance_plot(
            time_ms_state=time_ms,
            dist=dist,
            state_a=states_base[i],
            state_b=states_base[j],
            payload_a=filter_base[i],
            payload_b=filter_base[j],
            name_a=name_a,
            name_b=name_b,
            title=title,
            out_path=out_path,
            plot_limits=plot_limits,
        )

        rows.append({
            "type": f"{MAT_BASE}-{MAT_BASE}",
            "material_a": MAT_BASE,
            "material_b": MAT_BASE,
            "i": i,
            "j": j,
            "sample_i": int(selected_seq[i]),
            "sample_j": int(selected_seq[j]),
            "mean_distance": float(np.mean(dist)),
            "max_distance": float(np.max(dist)),
            "png": str(out_path),
        })

    return rows


# =========================================
# Al-Target 全組み合わせ保存
# =========================================
def save_base_target_pair_distance_plots(
    *,
    target_material: str,
    time_ms: np.ndarray,
    states_base: np.ndarray,
    states_target: np.ndarray,
    filter_base: list[dict],
    filter_target: list[dict],
    selected_seq: np.ndarray,
    plot_limits: dict,
):
    rows = []
    base_label = DISPLAY_NAMES.get(MAT_BASE, MAT_BASE)
    target_label = DISPLAY_NAMES.get(target_material, target_material)
    out_dir = PAIR_TARGET_DIRS[target_material]

    for i, j in tqdm(
        list(product(range(N_COMPARE), range(N_COMPARE))),
        desc=f"Saving {MAT_BASE}-{target_material} pair plots",
    ):
        dist = calc_distance_over_time(states_base[i], states_target[j], mode=DISTANCE_MODE)

        name_a = f"{base_label} {i:02d} ID={int(selected_seq[i])}"
        name_b = f"{target_label} {j:02d} ID={int(selected_seq[j])}"

        title = f"{base_label}-{target_label} comparison\n{name_a} vs {name_b}"

        out_path = out_dir / (
            f"{MAT_BASE}_{i:02d}_id{int(selected_seq[i])}"
            f"_vs_{target_material}_{j:02d}_id{int(selected_seq[j])}.png"
        )

        save_one_pair_distance_plot(
            time_ms_state=time_ms,
            dist=dist,
            state_a=states_base[i],
            state_b=states_target[j],
            payload_a=filter_base[i],
            payload_b=filter_target[j],
            name_a=name_a,
            name_b=name_b,
            title=title,
            out_path=out_path,
            plot_limits=plot_limits,
        )

        rows.append({
            "type": f"{MAT_BASE}-{target_material}",
            "material_a": MAT_BASE,
            "material_b": target_material,
            "i": i,
            "j": j,
            "sample_i": int(selected_seq[i]),
            "sample_j": int(selected_seq[j]),
            "mean_distance": float(np.mean(dist)),
            "max_distance": float(np.max(dist)),
            "png": str(out_path),
        })

    return rows


# =========================================
# 全ペア距離の計算
# =========================================
def compute_all_pair_distances(
    *,
    states_base: np.ndarray,
    states_targets: dict[str, np.ndarray],
):
    """
    npz 保存に使うために、全ペアの距離 d_ij(t) だけを計算する。

    注意:
        距離の時間推移まとめグラフは保存しない。
        つまり、summary/liquid_state_distance_summary_multi_targets_rep*.png
        は出力しない。

    返り値:
        all_dists[pair_name] shape = (n_pairs, n_time)

    N_COMPARE=30 の場合:
        Al-Al      : 30C2 = 435 pairs
        Al-Target  : 30*30 = 900 pairs
    """

    all_dists = {}

    # -----------------------------
    # Al-Al
    # -----------------------------
    base_base_dists = []
    for i, j in combinations(range(N_COMPARE), 2):
        base_base_dists.append(
            calc_distance_over_time(states_base[i], states_base[j], mode=DISTANCE_MODE)
        )

    all_dists[f"{MAT_BASE}-{MAT_BASE}"] = np.stack(base_base_dists, axis=0)

    # -----------------------------
    # Al-Target
    # -----------------------------
    for target_material, states_target in states_targets.items():
        target_dists = []

        for i, j in product(range(N_COMPARE), range(N_COMPARE)):
            target_dists.append(
                calc_distance_over_time(states_base[i], states_target[j], mode=DISTANCE_MODE)
            )

        all_dists[f"{MAT_BASE}-{target_material}"] = np.stack(target_dists, axis=0)

    print("\n[computed all pair distances]")
    for name, dists in all_dists.items():
        print(
            f"{name}: shape={dists.shape}, "
            f"time_mean={float(np.mean(dists)):.6f}, "
            f"std={float(np.std(dists)):.6f}"
        )
    print()

    return all_dists


# =========================================
# 線形分離特性 SP_lin
# =========================================

# SP_lin のランク評価に使う行列の行数を固定する。
# これにより、すべての比較で同じ大きさの行列のランクを評価する。
#
# N_COMPARE=30 の場合:
#   Al-Al      : Al 30サンプル                         -> X shape = (30, 500)
#   Al-Target  : Al 15サンプル + Target 15サンプル      -> X shape = (30, 500)
#
# center_each_time=True なので、最大ランクは 30 - 1 = 29。
SP_RANK_TOTAL_ROWS = 30
SP_RANK_PER_CLASS = SP_RANK_TOTAL_ROWS // 2


def calc_splin_over_time_from_matrix_builder(
    *,
    matrix_builder,
    n_time: int,
    n_neurons: int,
    label: str,
    center_each_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    各時刻で matrix_builder(t_idx) により同じ大きさの状態行列 X(t) を作り、
    SP_lin(t)=rank(X(t)) を計算する。

    matrix_builder(t_idx):
        X(t) shape = (SP_RANK_TOTAL_ROWS, n_neurons)

    center_each_time=True の場合:
        各時刻でサンプル平均を引く。
        この場合、最大ランクは min(SP_RANK_TOTAL_ROWS - 1, n_neurons)。
    """

    sp_rank = np.zeros(n_time, dtype=np.float64)
    sp_norm = np.zeros(n_time, dtype=np.float64)

    if center_each_time:
        max_rank = min(SP_RANK_TOTAL_ROWS - 1, n_neurons)
    else:
        max_rank = min(SP_RANK_TOTAL_ROWS, n_neurons)

    max_rank = max(1, max_rank)

    for t_idx in range(n_time):
        X = matrix_builder(t_idx)

        if X.ndim != 2:
            raise ValueError(f"{label}: X must be 2D, got shape={X.shape}")

        if X.shape != (SP_RANK_TOTAL_ROWS, n_neurons):
            raise ValueError(
                f"{label}: X shape must be {(SP_RANK_TOTAL_ROWS, n_neurons)}, "
                f"but got {X.shape}"
            )

        if center_each_time:
            X = X - np.mean(X, axis=0, keepdims=True)

        rank_t = np.linalg.matrix_rank(X)

        sp_rank[t_idx] = float(rank_t)
        sp_norm[t_idx] = float(rank_t) / float(max_rank)

    return sp_rank, sp_norm, max_rank


def calc_splin_over_time_same_size_al_al(
    states_base: np.ndarray,
    *,
    center_each_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Al-Al 用の SP_lin(t) を計算する。

    使う行列:
        X(t) = Al の SP_RANK_TOTAL_ROWS サンプル

    N_COMPARE=30, SP_RANK_TOTAL_ROWS=30 の場合:
        X(t) shape = (30, 500)

    注意:
        同じ Al サンプルを2回重ねない。
    """

    if states_base.ndim != 3:
        raise ValueError("states_base must be 3D array: (samples, neurons, time)")

    if states_base.shape[0] < SP_RANK_TOTAL_ROWS:
        raise ValueError(
            f"states_base needs at least {SP_RANK_TOTAL_ROWS} samples for Al-Al SP_lin, "
            f"but got {states_base.shape[0]}"
        )

    n_neurons = states_base.shape[1]
    n_time = states_base.shape[2]

    def matrix_builder(t_idx: int) -> np.ndarray:
        return states_base[:SP_RANK_TOTAL_ROWS, :, t_idx]

    return calc_splin_over_time_from_matrix_builder(
        matrix_builder=matrix_builder,
        n_time=n_time,
        n_neurons=n_neurons,
        label=f"{MAT_BASE}-{MAT_BASE}",
        center_each_time=center_each_time,
    )


def calc_splin_over_time_same_size_pair(
    states_base: np.ndarray,
    states_target: np.ndarray,
    *,
    center_each_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Al-Target 用の SP_lin(t) を計算する。

    使う行列:
        X(t) = Al の SP_RANK_PER_CLASS サンプル
             + Target の SP_RANK_PER_CLASS サンプル

    N_COMPARE=30, SP_RANK_PER_CLASS=15 の場合:
        X(t) shape = (30, 500)

    これにより Al-Al と Al-Target のランク評価行列の大きさを同じにする。
    """

    if states_base.ndim != 3 or states_target.ndim != 3:
        raise ValueError("states_base and states_target must be 3D arrays: (samples, neurons, time)")

    if states_base.shape[1] != states_target.shape[1]:
        raise ValueError(f"neuron mismatch: {states_base.shape[1]} vs {states_target.shape[1]}")

    if states_base.shape[2] != states_target.shape[2]:
        raise ValueError(f"time mismatch: {states_base.shape[2]} vs {states_target.shape[2]}")

    if states_base.shape[0] < SP_RANK_PER_CLASS:
        raise ValueError(
            f"states_base needs at least {SP_RANK_PER_CLASS} samples for Al-Target SP_lin, "
            f"but got {states_base.shape[0]}"
        )

    if states_target.shape[0] < SP_RANK_PER_CLASS:
        raise ValueError(
            f"states_target needs at least {SP_RANK_PER_CLASS} samples for Al-Target SP_lin, "
            f"but got {states_target.shape[0]}"
        )

    n_neurons = states_base.shape[1]
    n_time = states_base.shape[2]

    def matrix_builder(t_idx: int) -> np.ndarray:
        Xa = states_base[:SP_RANK_PER_CLASS, :, t_idx]
        Xb = states_target[:SP_RANK_PER_CLASS, :, t_idx]
        return np.vstack([Xa, Xb])

    return calc_splin_over_time_from_matrix_builder(
        matrix_builder=matrix_builder,
        n_time=n_time,
        n_neurons=n_neurons,
        label="Al-Target",
        center_each_time=center_each_time,
    )


def save_pairwise_splin_time_transition_plot(
    *,
    time_ms: np.ndarray,
    states_base: np.ndarray,
    states_targets: dict[str, np.ndarray],
):
    """
    SP_lin(t)=rank(M_s(t)) の時間推移を保存する。

    すべての比較で、同じ大きさの行列のランクを評価する。

    N_COMPARE=30, SP_RANK_TOTAL_ROWS=30 の場合:
        Al-Al:
            Al 30サンプル
            X(t) shape = (30, 500)

        Al-Target:
            Al 15サンプル + Target 15サンプル
            X(t) shape = (30, 500)

    グラフの縦軸:
        ランクそのもの

    CSV:
        ランクそのもの、正規化ランク、最大可能ランク、評価行列サイズを保存
    """

    pair_to_splin_rank = {}
    pair_to_splin_norm = {}
    pair_to_max_rank = {}
    pair_to_matrix_rows = {}
    pair_to_matrix_cols = {}

    base_pair_name = f"{MAT_BASE}-{MAT_BASE}"

    # -----------------------------
    # Al-Al
    # X(t) shape = (SP_RANK_TOTAL_ROWS, n_neurons)
    # -----------------------------
    sp_rank, sp_norm, max_rank = calc_splin_over_time_same_size_al_al(
        states_base,
        center_each_time=True,
    )

    pair_to_splin_rank[base_pair_name] = sp_rank
    pair_to_splin_norm[base_pair_name] = sp_norm
    pair_to_max_rank[base_pair_name] = max_rank
    pair_to_matrix_rows[base_pair_name] = SP_RANK_TOTAL_ROWS
    pair_to_matrix_cols[base_pair_name] = states_base.shape[1]

    # -----------------------------
    # Al-Target
    # X(t) shape = (SP_RANK_TOTAL_ROWS, n_neurons)
    # -----------------------------
    for target_material, states_target in states_targets.items():
        pair_name = f"{MAT_BASE}-{target_material}"

        sp_rank, sp_norm, max_rank = calc_splin_over_time_same_size_pair(
            states_base,
            states_target,
            center_each_time=True,
        )

        pair_to_splin_rank[pair_name] = sp_rank
        pair_to_splin_norm[pair_name] = sp_norm
        pair_to_max_rank[pair_name] = max_rank
        pair_to_matrix_rows[pair_name] = SP_RANK_TOTAL_ROWS
        pair_to_matrix_cols[pair_name] = states_base.shape[1]

    plt.rcParams["font.size"] = 13

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for pair_name, sp_rank in pair_to_splin_rank.items():
        max_rank = pair_to_max_rank[pair_name]
        rows = pair_to_matrix_rows[pair_name]
        cols = pair_to_matrix_cols[pair_name]

        if pair_name == base_pair_name:
            label = f"{pair_name}, X={rows}x{cols}, max rank={max_rank}"
            ax.plot(
                time_ms,
                sp_rank,
                linewidth=2.5,
                label=label,
            )
        else:
            label = f"{pair_name}, X={rows}x{cols}, max rank={max_rank}"
            ax.plot(
                time_ms,
                sp_rank,
                linewidth=2.5,
                linestyle="--",
                label=label,
            )

    ax.set_title(
        "Time transition of linear separation property\n"
        r"$SP_{lin}(t)=rank(M_s(t))$",
        fontsize=16,
    )

    ax.set_xlabel("Time [ms]", fontsize=15)
    ax.set_ylabel("Rank of state matrix", fontsize=15)

    ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))

    max_rank_value = max(float(np.max(v)) for v in pair_to_splin_rank.values())
    ax.set_ylim(0.0, max_rank_value + 1.0)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="both", labelsize=13)

    fig.tight_layout()

    pdf_path = SUMMARY_DIR / f"splin_rank_same_matrix_size_time_transition_rep{REP}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {pdf_path}")

    rows_for_csv = []

    for pair_name in pair_to_splin_rank.keys():
        sp_rank = pair_to_splin_rank[pair_name]
        sp_norm = pair_to_splin_norm[pair_name]
        max_rank = pair_to_max_rank[pair_name]
        matrix_rows = pair_to_matrix_rows[pair_name]
        matrix_cols = pair_to_matrix_cols[pair_name]

        for t, r, n in zip(time_ms, sp_rank, sp_norm):
            rows_for_csv.append({
                "pair": pair_name,
                "time_ms": float(t),
                "SP_lin_rank": float(r),
                "SP_lin_normalized": float(n),
                "max_possible_rank": float(max_rank),
                "matrix_rows": int(matrix_rows),
                "matrix_cols": int(matrix_cols),
                "matrix_shape": f"{matrix_rows}x{matrix_cols}",
                "center_each_time": True,
            })

    csv_path = SUMMARY_DIR / f"splin_rank_same_matrix_size_time_transition_rep{REP}.csv"
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)

    print(f"[saved] {csv_path}")

    return pair_to_splin_rank, pair_to_splin_norm


# =========================================
# Fisher分離度 FDR の時間推移
# =========================================
def calc_fdr_over_time_for_pair(
    states_a: np.ndarray,
    states_b: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    2素材間の Fisher 分離度 FDR(t) を計算する。

        FDR(t) = ||mu_A(t) - mu_B(t)||^2 / trace(S_W(t))

    trace(S_W) は共分散行列を明示的に作らず、
    各クラス内の平均との差の二乗和として計算する。
    """

    if states_a.ndim != 3 or states_b.ndim != 3:
        raise ValueError("states_a and states_b must be 3D arrays: (samples, neurons, time)")

    if states_a.shape[1] != states_b.shape[1]:
        raise ValueError(f"neuron mismatch: {states_a.shape[1]} vs {states_b.shape[1]}")

    if states_a.shape[2] != states_b.shape[2]:
        raise ValueError(f"time mismatch: {states_a.shape[2]} vs {states_b.shape[2]}")

    n_time = states_a.shape[2]
    fdr = np.zeros(n_time, dtype=np.float64)

    for t_idx in range(n_time):
        Xa = states_a[:, :, t_idx]
        Xb = states_b[:, :, t_idx]

        mu_a = np.mean(Xa, axis=0)
        mu_b = np.mean(Xb, axis=0)

        between = np.sum((mu_a - mu_b) ** 2)

        Xa_centered = Xa - mu_a
        Xb_centered = Xb - mu_b

        within = np.sum(Xa_centered ** 2) + np.sum(Xb_centered ** 2)

        fdr[t_idx] = between / (within + eps)

    return fdr


def save_pairwise_fdr_time_transition_plot(
    *,
    time_ms: np.ndarray,
    states_base: np.ndarray,
    states_targets: dict[str, np.ndarray],
):
    """
    Al-Al, Al-wood, Al-washi, Al-rubber の FDR(t) を1枚にまとめて保存する。

    Al-Al:
        Al N_COMPAREサンプルを前半と後半に分ける。

    Al-Target:
        Al N_COMPAREサンプル vs Target N_COMPAREサンプル
    """

    pair_to_fdr = {}

    base_pair_name = f"{MAT_BASE}-{MAT_BASE}"

    n_base = states_base.shape[0]
    half = n_base // 2

    if half < 1 or n_base - half < 1:
        raise ValueError("N_COMPARE is too small for Al-Al FDR split.")

    base_a = states_base[:half]
    base_b = states_base[half:]

    pair_to_fdr[base_pair_name] = calc_fdr_over_time_for_pair(
        base_a,
        base_b,
    )

    for target_material, states_target in states_targets.items():
        pair_name = f"{MAT_BASE}-{target_material}"

        pair_to_fdr[pair_name] = calc_fdr_over_time_for_pair(
            states_base,
            states_target,
        )

    plt.rcParams["font.size"] = 13

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for pair_name, fdr in pair_to_fdr.items():
        if pair_name == base_pair_name:
            ax.plot(
                time_ms,
                fdr,
                linewidth=2.5,
                label=f"{pair_name}, Al split",
            )
        else:
            ax.plot(
                time_ms,
                fdr,
                linewidth=2.5,
                linestyle="--",
                label=f"{pair_name}",
            )

    ax.set_title(
        "Time transition of Fisher discrimination ratio\n"
        r"$FDR(t)=\frac{||\mu_A(t)-\mu_B(t)||^2}{tr(S_W(t))}$",
        fontsize=16,
    )

    ax.set_xlabel("Time [ms]", fontsize=15)
    ax.set_ylabel("Fisher discrimination ratio", fontsize=15)

    ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.tick_params(axis="both", labelsize=13)

    fig.tight_layout()

    pdf_path = SUMMARY_DIR / f"fdr_time_transition_multi_targets_rep{REP}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {pdf_path}")

    rows = []

    for pair_name, fdr in pair_to_fdr.items():
        for t, value in zip(time_ms, fdr):
            rows.append({
                "pair": pair_name,
                "time_ms": float(t),
                "FDR": float(value),
            })

    csv_path = SUMMARY_DIR / f"fdr_time_transition_multi_targets_rep{REP}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"[saved] {csv_path}")

    return pair_to_fdr


# =========================================
# 1サンプルごとの内部状態平均値の分布
# =========================================
def save_sample_mean_state_distribution_plot(
    *,
    states_base: np.ndarray,
    states_targets: dict[str, np.ndarray],
):
    material_to_values = {}

    base_sample_mean = np.mean(states_base, axis=(1, 2))
    material_to_values[MAT_BASE] = base_sample_mean

    for target_material, states_target in states_targets.items():
        target_sample_mean = np.mean(states_target, axis=(1, 2))
        material_to_values[target_material] = target_sample_mean

    all_values = np.concatenate(list(material_to_values.values()))

    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))

    if vmin == vmax:
        vmin = vmin - 1e-6
        vmax = vmax + 1e-6

    n_bins = min(20, max(5, N_COMPARE * len(material_to_values)))
    bins = np.linspace(vmin, vmax, n_bins)

    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    summary_rows = []

    for material, values in material_to_values.items():
        label = DISPLAY_NAMES.get(material, material)

        ax.hist(
            values,
            bins=bins,
            alpha=HIST_ALPHA,
            edgecolor="black",
            linewidth=0.7,
            label=label,
        )

        mean_val = float(np.mean(values))
        ax.axvline(
            mean_val,
            linestyle="--",
            linewidth=2.0,
            label=f"{label} mean = {mean_val:.5f}",
        )

        summary_rows.append({
            "material": material,
            "display_name": label,
            "n_samples": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        })

    ax.set_title(
        "Distribution of sample-wise mean liquid states\n"
        r"$\bar{x}_i = \frac{1}{NT}\sum_n\sum_t x_{i,n}(t)$",
        fontsize=15,
    )

    ax.set_xlabel("Sample-wise mean liquid state", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()

    png_path = SUMMARY_DIR / f"sample_mean_liquid_state_distribution_rep{REP}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)

    print(f"[saved] {png_path}")

    csv_path = SUMMARY_DIR / f"sample_mean_liquid_state_distribution_summary_rep{REP}.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    return material_to_values


# =========================================
# main
# =========================================
def main():
    start_scope()

    seed_val = BASE_SEED + (REP - 1)

    np.random.seed(seed_val)
    rng = np.random.default_rng(seed_val)
    seed(seed_val)

    sample_seq = np.arange(1, 325, dtype=int)
    rng.shuffle(sample_seq)

    test_seq = sample_seq[N_TRAIN:N_TRAIN + N_SAMPLE_TEST]
    selected_seq = test_seq[:N_COMPARE]

    print(f"REP = {REP}")
    print(f"selected samples = {selected_seq}")
    print(f"N_COMPARE = {N_COMPARE}")
    print(f"RESULT_DIR = {RESULT_DIR}")

    net, G_in, G_res, S_in, S_res, params = build_reservoir(rng=rng)

    eval_neuron_ids = rng.choice(N_RES, size=N_EVAL_NEURONS, replace=False)
    eval_neuron_ids = np.sort(eval_neuron_ids)

    print(f"eval_neuron_ids shape = {eval_neuron_ids.shape}")
    print(f"first 20 eval_neuron_ids = {eval_neuron_ids[:20]}")

    states_base = []
    filter_base = []
    spike_counts_base = []

    states_targets: dict[str, np.ndarray] = {}
    filters_targets: dict[str, list[dict]] = {}
    spike_counts_targets: dict[str, list[int]] = {}

    time_ms_ref = None

    # -----------------------------
    # Base material internal states
    # -----------------------------
    for sample_id in tqdm(selected_seq, desc=f"{MAT_BASE} liquid states"):
        x_state, time_ms, payload, n_spikes = run_one_sample(
            material=MAT_BASE,
            sample_id=int(sample_id),
            net=net,
            G_in=G_in,
            G_res=G_res,
            eval_neuron_ids=eval_neuron_ids,
        )

        states_base.append(x_state)
        filter_base.append(payload)
        spike_counts_base.append(n_spikes)

        if time_ms_ref is None:
            time_ms_ref = time_ms

    states_base = np.stack(states_base, axis=0)

    # -----------------------------
    # Target materials internal states
    # -----------------------------
    for target_material in TARGET_MATERIALS:
        states_target_list = []
        filter_target_list = []
        spike_count_target_list = []

        for sample_id in tqdm(selected_seq, desc=f"{target_material} liquid states"):
            x_state, time_ms, payload, n_spikes = run_one_sample(
                material=target_material,
                sample_id=int(sample_id),
                net=net,
                G_in=G_in,
                G_res=G_res,
                eval_neuron_ids=eval_neuron_ids,
            )

            states_target_list.append(x_state)
            filter_target_list.append(payload)
            spike_count_target_list.append(n_spikes)

        states_targets[target_material] = np.stack(states_target_list, axis=0)
        filters_targets[target_material] = filter_target_list
        spike_counts_targets[target_material] = spike_count_target_list

    print("states_base shape:", states_base.shape)
    print(f"{MAT_BASE} spike counts:", spike_counts_base)

    for target_material in TARGET_MATERIALS:
        print(f"{target_material} states shape:", states_targets[target_material].shape)
        print(f"{target_material} spike counts:", spike_counts_targets[target_material])

    plot_limits = compute_global_plot_limits(
        time_ms_ref=time_ms_ref,
        filter_base=filter_base,
        filters_targets=filters_targets,
        states_base=states_base,
        states_targets=states_targets,
    )

    # -----------------------------
    # 全ペアごとのPNG保存はしない
    # "Saving Al_board-Al_board pair plots" の処理はここで止める
    # -----------------------------

    # -----------------------------
    # 全ペア距離の計算
    # 距離の時間推移まとめグラフや素材別ヒストグラムは保存しない
    # npz保存用に距離配列だけ残す
    # -----------------------------
    all_dists = compute_all_pair_distances(
        states_base=states_base,
        states_targets=states_targets,
    )

    # -----------------------------
    # 線形分離特性 SP_lin の時間推移
    # -----------------------------
    splin_rank, splin_norm = save_pairwise_splin_time_transition_plot(
        time_ms=time_ms_ref,
        states_base=states_base,
        states_targets=states_targets,
    )

    # -----------------------------
    # Fisher分離度 FDR の時間推移
    # -----------------------------
    fdr_time = save_pairwise_fdr_time_transition_plot(
        time_ms=time_ms_ref,
        states_base=states_base,
        states_targets=states_targets,
    )

    # -----------------------------
    # 1サンプルごとの平均内部状態分布
    # -----------------------------
    sample_mean_state_values = save_sample_mean_state_distribution_plot(
        states_base=states_base,
        states_targets=states_targets,
    )

    # -----------------------------
    # データ保存
    # -----------------------------
    save_payload = {
        "time_ms": time_ms_ref.astype(np.float32),
        "selected_seq": selected_seq.astype(np.int64),
        "eval_neuron_ids": eval_neuron_ids.astype(np.int64),
        "states_base": states_base.astype(np.float32),
        "base_material": MAT_BASE,
        "target_materials": np.asarray(TARGET_MATERIALS),
        "spike_counts_base": np.asarray(spike_counts_base, dtype=np.int64),
        "distance_mode": DISTANCE_MODE,
        "distance_ylim_max": float(DIST_YLIM_MAX),
        "dt_ms": float(DT_MS),
        "state_dt_ms": float(STATE_DT_MS),
        "tau_state_ms": float(TAU_STATE_MS),
        "n_res": int(N_RES),
        "n_eval_neurons": int(N_EVAL_NEURONS),
        "n_compare": int(N_COMPARE),
        "sp_rank_total_rows": int(SP_RANK_TOTAL_ROWS),
        "sp_rank_per_class": int(SP_RANK_PER_CLASS),
        "plot_xlim": np.asarray(plot_limits["xlim"], dtype=np.float32),
        "plot_input0_ylim": np.asarray(plot_limits["input0_ylim"], dtype=np.float32),
        "plot_input1_ylim": np.asarray(plot_limits["input1_ylim"], dtype=np.float32),
        "plot_mean_state_ylim": np.asarray(plot_limits["mean_state_ylim"], dtype=np.float32),
        "plot_distance_ylim": np.asarray(plot_limits["distance_ylim"], dtype=np.float32),
        "params": str(params),
    }

    for target_material in TARGET_MATERIALS:
        save_payload[f"states_{target_material}"] = states_targets[target_material].astype(np.float32)
        save_payload[f"spike_counts_{target_material}"] = np.asarray(
            spike_counts_targets[target_material],
            dtype=np.int64,
        )

    for name, dists in all_dists.items():
        safe_name = name.replace("-", "_").replace("/", "_")
        save_payload[f"dists_{safe_name}"] = dists.astype(np.float32)
        save_payload[f"time_mean_{safe_name}"] = float(np.mean(dists))

    for pair_name, values in splin_rank.items():
        safe_pair_name = pair_name.replace("-", "_").replace("/", "_")
        save_payload[f"splin_rank_{safe_pair_name}"] = values.astype(np.float32)

    for pair_name, values in splin_norm.items():
        safe_pair_name = pair_name.replace("-", "_").replace("/", "_")
        save_payload[f"splin_normalized_{safe_pair_name}"] = values.astype(np.float32)

    for pair_name, values in fdr_time.items():
        safe_pair_name = pair_name.replace("-", "_").replace("/", "_")
        save_payload[f"fdr_time_{safe_pair_name}"] = values.astype(np.float32)

    for material, values in sample_mean_state_values.items():
        safe_material = material.replace("-", "_").replace("/", "_")
        save_payload[f"sample_mean_state_values_{safe_material}"] = values.astype(np.float32)

    npz_path = SUMMARY_DIR / f"liquid_state_distance_data_multi_targets_rep{REP}.npz"
    np.savez_compressed(npz_path, **save_payload)

    print(f"[saved] {npz_path}")
    print("All finished.")


if __name__ == "__main__":
    main()
