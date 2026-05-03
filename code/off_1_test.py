# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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
OUT_PREFIX = "off_1"

# =========================================
# 反復設定
# SRDP 側に合わせる
# =========================================
N_REPEAT = 10
BASE_SEED = 2
N_TRAIN = 100
N_SAMPLE_TEST = 100

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
    ii = np.array(S_out.i[:], dtype=int)
    jj = np.array(S_out.j[:], dtype=int)
    ww = np.array(S_out.w_out[:], dtype=float)
    w_out_dense[ii, jj] = ww
    np.save(out_dir / f"{OUT_PREFIX}_w_out_rep{rep}.npy", w_out_dense)

# =========================================
# 1回分実行
# SRDP 側と同じ乱数の流れに合わせる
# =========================================
def run_once(rep: int):
    start_scope()

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
    w_out_init = rng.standard_normal((N_res, N_out)) * mask_out / np.sqrt(N_out * p_out) * G

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

    # -----------------------------
    # ここで初期重みを保存
    # off は学習しないのでこの値がそのまま off の重み
    # -----------------------------
    save_weights(
        rep=rep,
        w_in=w_in,
        w_res=w_res_init,
        S_out=S_out,
        N_res=N_res,
        N_out=N_out,
        out_dir=SAVE_DIR
    )

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

    # -----------------------------
    # monitor
    # -----------------------------
    Mr_out = SpikeMonitor(G_out)
    sout_rec = np.zeros((len(dir_name), n_sample, N_out, n_bins), dtype=float)

    # -----------------------------
    # simulation loop
    # -----------------------------
    t0 = 0 * ms

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

            run((nt * dt_ms) * ms, namespace=ns)

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



