# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
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
SAVE_DIR = SCRIPT_DIR

DATA_ROOT = SCRIPT_PATH.parents[1]
path = str(DATA_ROOT) + "/"

dir_name = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board"
]

# =========================================
# 保存名
# =========================================
OUT_WEIGHT_DISTRIBUTION = "uniform"  # "normal" or "uniform"
OUT_PREFIX_BASE = "off_1_gout_rate_match"
OUT_PREFIX = (
    OUT_PREFIX_BASE
    if OUT_WEIGHT_DISTRIBUTION == "normal"
    else f"{OUT_PREFIX_BASE}_{OUT_WEIGHT_DISTRIBUTION}"
)

# =========================================
# 反復設定
# SRDP 側に合わせる
# =========================================
N_REPEAT = 1
BASE_SEED = 2
N_TRAIN = 100
N_SAMPLE_TEST = 100

# off condition: r_pre/r_post are averaged over 5 samples for each material.
OFF_RATE_SAMPLES = 1
OFF_ALPHA = 0.05
OFF_ALPHA_MIN = 0.000001
OFF_ALPHA_DECAY_SCALE_HZ = 100.0
OFF_ALPHA_BACKTRACK = 0.5
OFF_ALPHA_GROWTH = 1.1
OFF_RATE_DIFF_THRESHOLD_HZ = 1.0
OFF_MAX_ITER = 20
OFF_G_OUT_MIN = 0.0
OFF_G_OUT_MAX = 2.0
RATE_TRACE_WINDOW_MS = 5

# =========================================
# input filter
# =========================================
def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            I[0, i] = I[0, i - 1] + 1 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1e-3))
            I[3, i] = I[0, i]
    return I[3, :]


def calc_merkel(data, t, dt):
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
# 重み保存
# =========================================
def save_weights(rep: int,
                 w_in: np.ndarray,
                 w_res: np.ndarray,
                 S_out: Synapses,
                 N_res: int,
                 N_out: int,
                 out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{OUT_PREFIX}_w_in_rep{rep}.npy", w_in)
    np.save(out_dir / f"{OUT_PREFIX}_w_res_rep{rep}.npy", w_res)

    # Synapses から dense w_out を復元
    w_out_dense = np.zeros((N_res, N_out), dtype=float)
    w_out_dense = np.zeros((N_res, N_out), dtype=float)
    ii = np.array(S_out.i[:], dtype=int)
    jj = np.array(S_out.j[:], dtype=int)
    ww = np.array(S_out.w_out[:], dtype=float)
    w_out_dense[ii, jj] = ww
    np.save(out_dir / f"{OUT_PREFIX}_w_out_rep{rep}.npy", w_out_dense)


def plot_gout_history(history: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "gout_history.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    if "g_out" in hist.columns:
        axes[0].plot(hist["iteration"], hist["g_out"], marker="o", label="accepted g_out")
        if "candidate_g_out" in hist.columns:
            axes[0].plot(hist["iteration"], hist["candidate_g_out"], marker="x", linestyle="--", label="candidate g_out")
    else:
        axes[0].plot(hist["iteration"], hist["gout_mean"], marker="o", label="mean")
        axes[0].fill_between(
            hist["iteration"],
            hist["gout_mean"] - hist["gout_std"],
            hist["gout_mean"] + hist["gout_std"],
            alpha=0.2,
            label="mean +/- std",
        )
        axes[0].plot(hist["iteration"], hist["gout_min"], linestyle="--", label="min")
        axes[0].plot(hist["iteration"], hist["gout_max"], linestyle="--", label="max")
    axes[0].set_ylabel("g_out")
    axes[0].legend()

    axes[1].plot(hist["iteration"], hist["mean_abs_diff_hz"], marker="o", color="tab:red")
    axes[1].axhline(OFF_RATE_DIFF_THRESHOLD_HZ, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("mean |r_pre - r_post| (Hz)")

    for mat in dir_name:
        col = f"mean_abs_diff_hz_{mat}"
        if col in hist.columns:
            axes[2].plot(hist["iteration"], hist[col], marker="o", label=mat)
    axes[2].axhline(OFF_RATE_DIFF_THRESHOLD_HZ, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("per-material |r_pre - r_post| (Hz)")
    axes[2].legend(ncol=2, fontsize=8)

    if "effective_alpha" in hist.columns:
        axes[3].plot(hist["iteration"], hist["effective_alpha"], marker="o", color="tab:purple")
    axes[3].set_xlabel("update iteration")
    axes[3].set_ylabel("effective alpha")
    fig.tight_layout()
    fig.savefig(out_dir / "gout_history.png", dpi=200)
    plt.close(fig)


def calc_effective_alpha(mean_abs_diff_hz: float) -> float:
    scale = mean_abs_diff_hz / (mean_abs_diff_hz + OFF_ALPHA_DECAY_SCALE_HZ)
    return float(OFF_ALPHA_MIN + (OFF_ALPHA - OFF_ALPHA_MIN) * scale)


def make_output_weight_base(rng, shape, mask_out, n_out: int, p_out: float) -> np.ndarray:
    if OUT_WEIGHT_DISTRIBUTION == "normal":
        raw = rng.standard_normal(shape)
    elif OUT_WEIGHT_DISTRIBUTION == "uniform":
        raw = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=shape)
    else:
        raise ValueError(
            f"OUT_WEIGHT_DISTRIBUTION must be 'normal' or 'uniform': {OUT_WEIGHT_DISTRIBUTION}"
        )

    return raw * mask_out / np.sqrt(n_out * p_out)


def plot_material_sample_rates(initial_rows: list[dict], final_rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_df = pd.DataFrame(initial_rows)
    final_df = pd.DataFrame(final_rows)
    initial_df.to_csv(out_dir / "sample_rates_initial.csv", index=False)
    final_df.to_csv(out_dir / "sample_rates_final.csv", index=False)

    for mat in dir_name:
        init_mat = initial_df[initial_df["material"] == mat].sort_values(["sample_id", "time_ms"])
        final_mat = final_df[final_df["material"] == mat].sort_values(["sample_id", "time_ms"])

        fig, ax = plt.subplots(figsize=(9, 4.8))
        init_mean = init_mat.groupby("time_ms", as_index=False)[["r_pre_hz", "r_post_hz"]].mean()
        final_mean = final_mat.groupby("time_ms", as_index=False)[["r_pre_hz", "r_post_hz"]].mean()

        ax.plot(init_mean["time_ms"], init_mean["r_pre_hz"], color="tab:blue", linewidth=2.5, label="initial r_pre mean")
        ax.plot(init_mean["time_ms"], init_mean["r_post_hz"], color="tab:orange", linewidth=2.5, label="initial r_post mean")
        ax.plot(final_mean["time_ms"], final_mean["r_pre_hz"], color="tab:green", linewidth=2.5, label="final r_pre mean")
        ax.plot(final_mean["time_ms"], final_mean["r_post_hz"], color="tab:red", linewidth=2.5, label="final r_post mean")

        ax.set_title(mat)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("mean firing rate (Hz)")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{mat}_rate_time_transition_initial_vs_final.png", dpi=200)
        plt.close(fig)


def append_rate_trace_rows(rows: list[dict],
                           material: str,
                           sample_id: int,
                           start_t,
                           end_t,
                           res_t,
                           out_t,
                           n_res: int,
                           n_out: int):
    duration_ms = float((end_t - start_t) / ms)
    if duration_ms <= 0:
        return

    bin_edges = np.arange(0.0, duration_ms + RATE_TRACE_WINDOW_MS, RATE_TRACE_WINDOW_MS)
    if bin_edges[-1] < duration_ms:
        bin_edges = np.append(bin_edges, duration_ms)
    else:
        bin_edges[-1] = duration_ms

    res_rel_ms = np.asarray((res_t - start_t) / ms, dtype=float)
    out_rel_ms = np.asarray((out_t - start_t) / ms, dtype=float)
    res_counts, _ = np.histogram(res_rel_ms, bins=bin_edges)
    out_counts, _ = np.histogram(out_rel_ms, bins=bin_edges)
    bin_width_s = np.diff(bin_edges) / 1000.0
    time_ms = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for k, center_ms in enumerate(time_ms):
        rows.append({
            "material": material,
            "sample_id": int(sample_id),
            "time_ms": float(center_ms),
            "r_pre_hz": float(res_counts[k] / (n_res * bin_width_s[k])),
            "r_post_hz": float(out_counts[k] / (n_out * bin_width_s[k])),
        })

# =========================================
# 1回分実行
# SRDP 側と同じ乱数の流れに合わせる
# =========================================
def run_once(rep: int):
    start_scope()
    print(f"[rep{rep}] output weight distribution: {OUT_WEIGHT_DISTRIBUTION}")

    # -----------------------------
    # SRDP 側と同じ seed の作り方
    # -----------------------------
    seed_val = BASE_SEED + (rep - 1)
    np.random.seed(seed_val)
    rng = np.random.default_rng(seed_val)
    seed(seed_val)

    # -----------------------------
    # sample_seq も SRDP 側と同じ作り方
    # -----------------------------
    sample_seq = np.arange(1, 325, dtype=int)
    rng.shuffle(sample_seq)
    np.save(SAVE_DIR / f"sample_seq_rep{rep}.npy", sample_seq)

    test_seq = sample_seq[N_TRAIN:N_TRAIN + N_SAMPLE_TEST]
    n_sample = len(test_seq)

    # -----------------------------
    # hyper-parameters
    # -----------------------------
    N_in = 2
    N_res = 1000
    N_out = 40

    p_in = 0.2
    p_res = 0.5
    p_out = 0.5

    v_reset = -65
    v_thr = -40

    tau_r = 2 * ms
    tau_d = 20 * ms

    BIAS = -65
    G = 0.25
    G_OUT_INIT = 0.25

    dt_ms = 0.1
    dt_s = dt_ms * 1e-3
    defaultclock.dt = dt_ms * ms

    # sout_rec の元 bin 数
    n_bins = 500

    # -----------------------------
    # neuron params
    # -----------------------------
    neuron_array_res = np.ones(N_res)
    tau_m_res = np.where(neuron_array_res == 1, 10, 10)
    t_ref_res = np.where(neuron_array_res == 1, 2, 2)

    neuron_array_out = np.ones(N_out)
    tau_m_out = np.where(neuron_array_out == 1, 10, 10)
    t_ref_out = np.where(neuron_array_out == 1, 2, 2)

    # -----------------------------
    # weights
    # SRDP 側と同じ rng の使い方
    # -----------------------------
    w_in = rng.standard_normal((N_in, N_res)) * (rng.random((N_in, N_res)) < p_in) / np.sqrt(N_in * p_in)

    variance = (N_res * p_res**2) ** -1
    w_res_init = rng.standard_normal((N_res, N_res)) * (rng.random((N_res, N_res)) < p_res) * np.sqrt(variance)
    for k in range(N_res):
        QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
        if len(QS) > 0:
            w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

    mask_out = (rng.random((N_res, N_out)) < p_out).astype(float)
    w_out_base = make_output_weight_base(
        rng=rng,
        shape=(N_res, N_out),
        mask_out=mask_out,
        n_out=N_out,
        p_out=p_out,
    )
    w_out_init = w_out_base * G_OUT_INIT

    # -----------------------------
    # Brian2 equations
    # -----------------------------
    LIF = '''
    dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
    I_exc : 1
    I_inh : 1
    tau_m : second
    t_ref : second
    '''

    double_exp_res = '''
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = G * R : 1
    '''
    on_pre_res = '''
    H_post += (w_res / (tau_r * tau_d)) / Hz
    '''

    double_exp_out = '''
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = R : 1
    '''
    on_pre_out = '''
    H_post += (w_out / (tau_r * tau_d)) / Hz
    '''

    eqs_res = double_exp_res + LIF
    eqs_out = double_exp_out + LIF

    # -----------------------------
    # NeuronGroups
    # -----------------------------
    G_res = NeuronGroup(
        N_res, eqs_res,
        threshold='v >= v_thr',
        reset='v = v_reset',
        refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
        method='exact'
    )
    G_res.tau_m = tau_m_res * ms
    G_res.t_ref = t_ref_res * ms
    G_res.I_inh = 0

    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold='v >= v_thr',
        reset='v = v_reset',
        refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
        method='exact'
    )
    G_out.tau_m = tau_m_out * ms
    G_out.t_ref = t_ref_out * ms
    G_out.I_inh = 0

    # -----------------------------
    # Synapses
    # S_out の接続も SRDP 側と同じ
    # -----------------------------
    S_res = Synapses(G_res, G_res, model='w_res : 1', on_pre=on_pre_res, method='euler')
    S_res.connect(condition='i != j')
    S_res.w_res = w_res_init[S_res.i, S_res.j]
    S_res.delay = 0 * ms

    pre_idx, post_idx = np.where(mask_out > 0)
    S_out = Synapses(G_res, G_out, model='w_out : 1', on_pre=on_pre_out, method='euler')
    if len(pre_idx) > 0:
        S_out.connect(i=pre_idx, j=post_idx)
        S_out.w_out = w_out_init[S_out.i, S_out.j]
    S_out.delay = 0 * ms
    w_out_base_syn = np.array(w_out_base[S_out.i, S_out.j], dtype=float)
    g_out = G_OUT_INIT

    # -----------------------------
    # ここで初期重みを保存
    # off は学習しないのでこの値がそのまま off の重み
    # -----------------------------
    # -----------------------------
    # input
    # -----------------------------
    input_ta = TimedArray(np.zeros((1, N_in)), dt=dt_ms * ms)

    G_in = NeuronGroup(N_in, '''
    t_start : second (shared)
    I = input_ta(t - t_start, i) : 1
    ''', method='euler')
    G_in.t_start = 0 * ms

    S_in = Synapses(G_in, G_res, model='''
    w : 1
    I_exc_post = w * I_pre : 1 (summed)
    ''', method='euler')

    pre_in, post_in = np.nonzero(w_in)
    if len(pre_in) > 0:
        S_in.connect(i=pre_in, j=post_in)
        S_in.w = w_in[pre_in, post_in]

    ns = {
        'input_ta': input_ta,
        'tau_r': tau_r,
        'tau_d': tau_d,
        'G': G,
        'BIAS': BIAS,
        'v_reset': v_reset,
        'v_thr': v_thr,
    }

    def load_input_current(mat: str, sample_id: int):
        files = glob.glob(path + "tactile_data/" + mat + f"/data_{sample_id}_*")
        if len(files) == 0:
            raise FileNotFoundError(f"data not found: {mat} sample={sample_id}")

        df = pd.read_table(files[0], header=None)
        df_np = df.to_numpy().T
        in_data_0 = df_np[:3, 3000:8000]
        nt = in_data_0.shape[1]
        t_array_s = np.arange(nt) * dt_s

        input_current = np.zeros((N_in, nt), dtype=float)
        ch = 0
        in_data = in_data_0[ch, :]
        I_merkel = calc_merkel(in_data, t_array_s, dt_s)
        I_meissner = calc_meissner(in_data, t_array_s, dt_s)
        input_current[ch * 2, :] = 0.4 * I_merkel * 0.02
        input_current[ch * 2 + 1, :] = 0.6 * 7.3 * I_meissner * 0.02
        return input_current, nt

    t0 = 0 * ms
    Mr_res_rate = SpikeMonitor(G_res, record=False)
    Mr_out_rate = SpikeMonitor(G_out, record=False)
    Mr_res_trace = SpikeMonitor(G_res)
    Mr_out_trace = SpikeMonitor(G_out)
    Mr_res_trace.active = False
    Mr_out_trace.active = False
    Mr_out = SpikeMonitor(G_out)
    Mr_out.active = False
    net = Network(
        G_in, G_res, G_out, S_in, S_res, S_out,
        Mr_res_rate, Mr_out_rate, Mr_res_trace, Mr_out_trace, Mr_out
    )

    def measure_rates(sample_ids, collect_samples=False):
        nonlocal input_ta, t0

        pre_counts = np.zeros(N_res, dtype=float)
        post_counts = np.zeros(N_out, dtype=float)
        total_duration_s = 0.0
        sample_rows = []
        material_rate_rows = []

        for mat in dir_name:
            mat_pre_counts = np.zeros(N_res, dtype=float)
            mat_post_counts = np.zeros(N_out, dtype=float)
            mat_duration_s = 0.0

            for sample_id in sample_ids:
                input_current, nt = load_input_current(mat, int(sample_id))
                vals = input_current.T
                vals = np.vstack([vals, vals[-1]])
                input_ta = TimedArray(vals, dt=dt_ms * ms)
                G_in.t_start = t0
                ns['input_ta'] = input_ta

                G_res.v = v_reset + (v_thr - v_reset) * rng.random(N_res)
                G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_out)
                G_res.R = 0
                G_res.H = 0
                G_out.R = 0
                G_out.H = 0

                res_trace_start_idx = len(Mr_res_trace.t)
                out_trace_start_idx = len(Mr_out_trace.t)
                pre_before = np.array(Mr_res_rate.count[:], dtype=float)
                post_before = np.array(Mr_out_rate.count[:], dtype=float)
                duration = (nt * dt_ms) * ms
                Mr_res_trace.active = collect_samples
                Mr_out_trace.active = collect_samples
                net.run(duration, namespace=ns)
                Mr_res_trace.active = False
                Mr_out_trace.active = False
                pre_delta = np.array(Mr_res_rate.count[:], dtype=float) - pre_before
                post_delta = np.array(Mr_out_rate.count[:], dtype=float) - post_before

                duration_s = float(duration / second)
                pre_counts += pre_delta
                post_counts += post_delta
                total_duration_s += duration_s
                mat_pre_counts += pre_delta
                mat_post_counts += post_delta
                mat_duration_s += duration_s
                t0 += duration

                if collect_samples:
                    append_rate_trace_rows(
                        rows=sample_rows,
                        material=mat,
                        sample_id=int(sample_id),
                        start_t=t0 - duration,
                        end_t=t0,
                        res_t=Mr_res_trace.t[res_trace_start_idx:],
                        out_t=Mr_out_trace.t[out_trace_start_idx:],
                        n_res=N_res,
                        n_out=N_out,
                    )

            mat_r_pre = mat_pre_counts / mat_duration_s
            mat_r_post = mat_post_counts / mat_duration_s
            material_rate_rows.append({
                "material": mat,
                "mean_abs_diff_hz": float(np.mean(np.abs(mat_r_pre[syn_pre] - mat_r_post[syn_post]))),
                "r_pre_mean_hz": float(np.mean(mat_r_pre)),
                "r_post_mean_hz": float(np.mean(mat_r_post)),
            })

        return pre_counts / total_duration_s, post_counts / total_duration_s, sample_rows, material_rate_rows

    rate_sample_seq = sample_seq[:OFF_RATE_SAMPLES]
    syn_pre = np.array(S_out.i[:], dtype=int)
    syn_post = np.array(S_out.j[:], dtype=int)
    history = []
    plot_dir = SAVE_DIR / f"{OUT_PREFIX}_plots" / f"rep{rep}"

    r_pre, r_post, initial_rows, material_rows = measure_rates(rate_sample_seq, collect_samples=True)
    syn_diff = r_pre[syn_pre] - r_post[syn_post]
    gain_diff = float(np.mean(syn_diff))
    mean_abs_diff = float(np.mean(np.abs(syn_diff)))
    effective_alpha = calc_effective_alpha(mean_abs_diff)
    w_now = np.array(S_out.w_out[:], dtype=float)
    history_row = {
        "iteration": 0,
        "mean_abs_diff_hz": mean_abs_diff,
        "effective_alpha": effective_alpha,
        "accepted": True,
        "out_weight_distribution": OUT_WEIGHT_DISTRIBUTION,
        "g_out": g_out,
        "candidate_g_out": g_out,
        "gout_mean": float(np.mean(w_now)),
        "gout_std": float(np.std(w_now)),
        "gout_min": float(np.min(w_now)),
        "gout_max": float(np.max(w_now)),
    }
    for row in material_rows:
        history_row[f"mean_abs_diff_hz_{row['material']}"] = row["mean_abs_diff_hz"]
    history.append(history_row)

    best_w_out = w_now.copy()
    best_g_out = g_out
    best_diff = mean_abs_diff
    alpha_scale = 1.0
    for update_idx in range(1, OFF_MAX_ITER + 1):
        prev_g_out = g_out
        prev_syn_diff = syn_diff.copy()
        prev_gain_diff = gain_diff
        prev_mean_abs_diff = mean_abs_diff
        effective_alpha = calc_effective_alpha(mean_abs_diff) * alpha_scale
        candidate_g_out = float(np.clip(g_out + effective_alpha * gain_diff, OFF_G_OUT_MIN, OFF_G_OUT_MAX))
        S_out.w_out = w_out_base_syn * candidate_g_out

        r_pre, r_post, _, material_rows = measure_rates(rate_sample_seq, collect_samples=False)
        candidate_syn_diff = r_pre[syn_pre] - r_post[syn_post]
        candidate_gain_diff = float(np.mean(candidate_syn_diff))
        candidate_mean_abs_diff = float(np.mean(np.abs(candidate_syn_diff)))
        w_now = np.array(S_out.w_out[:], dtype=float)
        accepted = candidate_mean_abs_diff <= prev_mean_abs_diff

        if accepted:
            g_out = candidate_g_out
            syn_diff = candidate_syn_diff
            gain_diff = candidate_gain_diff
            mean_abs_diff = candidate_mean_abs_diff
            alpha_scale = min(1.0, alpha_scale * OFF_ALPHA_GROWTH)
        else:
            g_out = prev_g_out
            S_out.w_out = w_out_base_syn * g_out
            syn_diff = prev_syn_diff
            gain_diff = prev_gain_diff
            mean_abs_diff = prev_mean_abs_diff
            w_now = np.array(S_out.w_out[:], dtype=float)
            alpha_scale *= OFF_ALPHA_BACKTRACK

        history_row = {
            "iteration": update_idx,
            "mean_abs_diff_hz": candidate_mean_abs_diff,
            "effective_alpha": effective_alpha,
            "accepted": accepted,
            "out_weight_distribution": OUT_WEIGHT_DISTRIBUTION,
            "g_out": g_out,
            "candidate_g_out": candidate_g_out,
            "gout_mean": float(np.mean(w_now)),
            "gout_std": float(np.std(w_now)),
            "gout_min": float(np.min(w_now)),
            "gout_max": float(np.max(w_now)),
        }
        for row in material_rows:
            history_row[f"mean_abs_diff_hz_{row['material']}"] = row["mean_abs_diff_hz"]
        history.append(history_row)
        print(
            f"[rep{rep}] off g_out update {update_idx}: "
            f"candidate mean |r_pre-r_post| = {candidate_mean_abs_diff:.6f} Hz, "
            f"current best step diff = {mean_abs_diff:.6f} Hz, "
            f"g_out = {g_out:.8f}, "
            f"candidate_g_out = {candidate_g_out:.8f}, "
            f"alpha = {effective_alpha:.8f}, "
            f"accepted = {accepted}"
        )

        if mean_abs_diff < best_diff:
            best_diff = mean_abs_diff
            best_w_out = w_now.copy()
            best_g_out = g_out
        if mean_abs_diff < OFF_RATE_DIFF_THRESHOLD_HZ:
            break

    S_out.w_out = best_w_out
    g_out = best_g_out
    _, _, final_rows, _ = measure_rates(rate_sample_seq, collect_samples=True)
    plot_gout_history(history, plot_dir)
    plot_material_sample_rates(initial_rows, final_rows, plot_dir)
    Mr_res_rate.active = False
    Mr_out_rate.active = False

    save_weights(
        rep=rep,
        w_in=w_in,
        w_res=w_res_init,
        S_out=S_out,
        N_res=N_res,
        N_out=N_out,
        out_dir=SAVE_DIR
    )
    np.save(SAVE_DIR / f"{OUT_PREFIX}_g_out_rep{rep}.npy", np.asarray(g_out, dtype=float))
    print(
        f"[rep{rep}] saved g_out={g_out:.8f} "
        f"with best mean |r_pre-r_post| = {best_diff:.6f} Hz"
    )

    # -----------------------------
    # monitor
    # -----------------------------
    Mr_out.active = True
    sout_rec = np.zeros((len(dir_name), n_sample, N_out, n_bins), dtype=float)

    # -----------------------------
    # simulation loop
    # -----------------------------
    for i, mat in enumerate(dir_name):
        for j in tqdm(range(n_sample), desc=f"[rep{rep}] {mat}"):
            files = glob.glob(path + "tactile_data/" + mat + f"/data_{int(test_seq[j])}_*")
            if len(files) == 0:
                raise FileNotFoundError(f"data not found: {mat} sample={int(test_seq[j])}")

            df = pd.read_table(files[0], header=None)
            df_np = df.to_numpy().T
            in_data_0 = df_np[:3, 3000:8000]
            nt = in_data_0.shape[1]
            t_array_s = np.arange(nt) * dt_s

            input_current = np.zeros((N_in, nt), dtype=float)

            # SRDP 側と同じく ch=0 だけ使う
            ch = 0
            in_data = in_data_0[ch, :]
            I_merkel = calc_merkel(in_data, t_array_s, dt_s)
            I_meissner = calc_meissner(in_data, t_array_s, dt_s)
            input_current[ch * 2, :] = 0.4 * I_merkel * 0.02
            input_current[ch * 2 + 1, :] = 0.6 * 7.3 * I_meissner * 0.02

            vals = input_current.T
            vals = np.vstack([vals, vals[-1]])
            input_ta = TimedArray(vals, dt=dt_ms * ms)
            G_in.t_start = t0

            # 状態初期値も rng でそろえる
            G_res.v = v_reset + (v_thr - v_reset) * rng.random(N_res)
            G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_out)
            G_res.R = 0
            G_res.H = 0
            G_out.R = 0
            G_out.H = 0

            start_t = t0
            start_idx = len(Mr_out.t)

            ns = {
                'input_ta': input_ta,
                'tau_r': tau_r,
                'tau_d': tau_d,
                'G': G,
                'BIAS': BIAS,
                'v_reset': v_reset,
                'v_thr': v_thr,
            }

            net.run((nt * dt_ms) * ms, namespace=ns)

            end_idx = len(Mr_out.t)

            t0 += (nt * dt_ms) * ms
            end_t = t0

            # この trial 分だけのスパイクを取り出して保存
            if end_idx > start_idx:
                t_sp = Mr_out.t[start_idx:end_idx]
                i_sp = Mr_out.i[start_idx:end_idx]

                mask = (t_sp > start_t) & (t_sp <= end_t)
                if np.any(mask):
                    rel_times_ms = (t_sp[mask] - start_t) / ms
                    ids = np.array(i_sp[mask], dtype=int)

                    bin_edges = np.linspace(0, nt * dt_ms, n_bins + 1)
                    for n in range(N_out):
                        counts, _ = np.histogram(rel_times_ms[ids == n], bins=bin_edges)
                        sout_rec[i, j, n, :] = counts
                else:
                    sout_rec[i, j, :, :] = 0
            else:
                sout_rec[i, j, :, :] = 0

    # -----------------------------
    # 出力保存
    # -----------------------------
    out_path = SAVE_DIR / f"{OUT_PREFIX}_sout_rec_rep{rep}.npy"
    np.save(out_path, sout_rec)
    print(f"[saved] {out_path} {sout_rec.shape}")

# =========================================
# main
# =========================================
if __name__ == "__main__":
    for rep in range(1, N_REPEAT + 1):
        run_once(rep)
