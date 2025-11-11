import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import time  # ★ 時間計測

# ===============================
# 0) 乱数シードを完全統一
# ===============================
np.random.seed(2)
from numpy.random import default_rng
rng = default_rng(2)

# Brian2
from brian2 import *
prefs.codegen.target = 'numpy'
prefs.core.default_float_dtype = float64

# ===============================
# 1) 共通ユーティリティ
# ===============================
def ensure_sample_seq(path="sample_seq.npy"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'"{path}" が見つかりません。先に別スクリプトで作成するか、'
            f'np.save("sample_seq.npy", rng.permutation(np.arange(1,325))) などで作ってください。'
        )
    return np.load(path)

def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for idx in range(1, len(t)):
        dF_dt = np.abs(data[idx] - data[idx-1]) / (t[idx] - t[idx-1])
        I[0, idx] = I[0, idx-1] + 1*dF_dt + (-I[0, idx-1]*dt/(8*1*1e-3))
        I[1, idx] = I[1, idx-1] + 0.24*dF_dt + (-(I[1, idx-1]-0.24*0.13)*dt/(200*1e-3))
        I[2, idx] = I[2, idx-1] + 0.07*dF_dt + (-I[2, idx-1]*dt/(1744.6*1e-3))
        I[3, idx] = I[0, idx]
    return I[3, :]

def calc_merkel(data, t, dt):
    I = np.zeros((4, len(t)))
    for idx in range(1, len(t)):
        dF_dt = np.abs(data[idx] - data[idx-1]) / (t[idx] - t[idx-1])
        if dF_dt < 0:
            dF_dt = 0
        I[0, idx] = I[0, idx-1] + 0.74*dF_dt + (-I[0, idx-1]*dt/(8*1*1e-3))
        I[1, idx] = I[1, idx-1] + 0.24*dF_dt + (-(I[1, idx-1]-0.24*0.13)*dt/(200*1*1e-3))
        I[2, idx] = I[2, idx-1] + 0.07*dF_dt + (-I[2, idx-1]*dt/(1744.6*1*1e-3))
        I[3, idx] = I[0, idx] + I[1, idx] + I[2, idx]
    return I[3, :]

# ===============================
# 2) 共通ハイパーパラメータ
# ===============================
path = "/Users/elast/OneDrive - 学校法人立命館/ドキュメント/研究コード/"
# 複数入れると連結され、境界で「状態のみリセット／重み継続」
materials = ["Al_board" ]

N_in  = 2
N_res = 2000
N_out = 40

p_in  = 0.4
p_res = 0.1
p_out = 0.2

v_reset = -65
v_peak  = 30
v_thr   = -40
BIAS = -65
G    = 0.01

tau_r = 2*ms
tau_d = 20*ms

dt_ms = 0.1
dt_s  = dt_ms * 1e-3
defaultclock.dt = dt_ms * ms

A_B_LTP = 0.06
A_B_LTD = -0.05
wmin = -1e9
wmax = 1e9

# 初期膜電位を両実装で共有するためのグローバル
v_init_res = None
v_init_out = None

# ===============================
# 3) データ読み込み（両者共通）
# ===============================
sample_seq = ensure_sample_seq("sample_seq.npy")

def load_one_trial(mat, i_size=0):
    df = pd.read_table(glob.glob(path + "tactile_data/" + mat + f"/data_{int(sample_seq[i_size])}_*")[0], header=None)
    df_np = df.to_numpy().T
    in_data_0 = df_np[:3, 3000:8000]
    nt = in_data_0.shape[1]
    t_array_s = np.arange(nt) * dt_s

    input_current = np.zeros([N_in, nt])
    for ch in range(3):
        in_data = in_data_0[ch, :]
        I_merkel   = calc_merkel(in_data,   t_array_s, dt_s)
        I_meissner = calc_meissner(in_data, t_array_s, dt_s)
        if ch == 0:
            input_current[ch*2,   :] = 0.01 * I_merkel
            input_current[ch*2+1, :] = 0.04 * I_meissner
    return input_current, nt

# ===============================
# 4) 自作実装（境界リセット対応）
# ===============================
class LIF_py:
    def __init__(self, N, dt, neuron_array, t_ref_py=2*1e-3, t_ref_in=1*1e-3, tau_m_py=10*1e-3, tau_m_in=10*1e-3,
                 v_reset=-65, v_peak=30, v_thr=-40):
        self.N = N
        self.dt = dt
        self.neuron_array = neuron_array
        self.ex_index = np.where(neuron_array == 1)[0]
        self.in_index = np.where(neuron_array == -1)[0]

        self.t_ref = np.zeros(N)
        self.t_ref[self.ex_index] = t_ref_py
        self.t_ref[self.in_index] = t_ref_in

        self.tau_m = np.zeros(N)
        self.tau_m[self.ex_index] = tau_m_py
        self.tau_m[self.in_index] = tau_m_in

        self.v_reset = v_reset
        self.v_peak = v_peak
        self.v_thr = v_thr

        self.v_k = np.ones(N) * self.v_reset + rng.random(N) * (v_peak - v_reset)
        self.v_ = None
        self.t_last = np.zeros(N)
        self.t_count = 0

    def initialize_states(self):
        self.t_count = 0
        self.t_last = np.zeros(self.N)
        self.v_k = np.ones(self.N) * self.v_reset + rng.random(self.N) * (self.v_peak - self.v_reset)
        self.v_ = self.v_k.copy()

    def __call__(self, I_ak, I_gk):
        dv_k = (-self.v_k + I_ak - I_gk) / self.tau_m
        v_k = self.v_k + (((self.dt * self.t_count) > (self.t_last + self.t_ref)) * dv_k * self.dt)
        s = (v_k >= self.v_thr)
        self.t_last = (1 - s) * self.t_last + s * self.dt * self.t_count
        v_k = (1 - s) * v_k + s * self.v_peak
        self.v_ = v_k
        self.v_k = (1 - s) * v_k + s * self.v_reset
        self.t_count += 1
        return s.astype(int)

class Synapse_res_py:
    def __init__(self, N, dt, w0, tau_d=20*1e-3, tau_r=2*1e-3):
        self.N = N
        self.dt = dt
        self.t_count = 0
        self.w0 = w0
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.R = np.zeros(self.N)
        self.H = np.zeros(self.N)
        self.JD = np.zeros(self.N)

    def initialize_states(self):
        self.t_count = 0
        self.R = np.zeros(self.N)
        self.H = np.zeros(self.N)
        self.JD = np.zeros(self.N)

    def __call__(self, sres):
        index = np.where(sres == 1)[0]
        len_index = len(index)
        if len_index > 0:
            self.JD = np.sum(self.w0[index, :], axis=0)
        self.R = self.R * np.exp(-self.dt / self.tau_d) + self.H * self.dt
        self.H = self.H * np.exp(-self.dt / self.tau_r) + self.JD * (len_index > 0) / (self.tau_r * self.tau_d)
        self.t_count += 1
        return self.R

class Synapse_out_py:
    def __init__(self, N_out, N_res, dt, w_out, w_out_const0, tau_d=20*1e-3, tau_r=2*1e-3):
        self.N_out = N_out
        self.N_res = N_res
        self.dt = dt
        self.t_count = 0
        self.w_out = w_out
        self.w_out_const0 = w_out_const0
        self.w_history = []
        self.w_history_by_out = []
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.R = np.zeros(self.N_out)
        self.H = np.zeros(self.N_out)
        self.JD = np.zeros(self.N_out)
        self.p = np.zeros(self.N_res)
        self.n = np.zeros(self.N_out)
        self.A_B_LTP = 0.06
        self.A_B_LTD = -0.05

    def initialize_states(self):
        self.t_count = 0
        self.R = np.zeros(self.N_out)
        self.H = np.zeros(self.N_out)
        self.JD = np.zeros(self.N_out)

    def __call__(self, sres, sout):
        index = np.where(sres == 1)[0]
        len_index = len(index)
        if len_index > 0:
            self.JD = np.sum(self.w_out[index, :], axis=0)
        self.R = self.R * np.exp(-self.dt / self.tau_d) + self.H * self.dt
        self.H = self.H * np.exp(-self.dt / self.tau_r) + self.JD * (len_index > 0) / (self.tau_r * self.tau_d)

        for i in np.arange(self.N_res):
            self.p[i] = sres[i] * (self.A_B_LTP + self.p[i]) + (1 - sres[i]) * self.p[i] * np.exp(-self.dt / self.tau_r)
            if sres[i]:
                idx = np.where(self.w_out[i, :] > 0)
                self.w_out[i, idx] = self.w_out[i, idx] + self.n[idx]

        for i in np.arange(self.N_out):
            self.n[i] = sout[i] * (self.A_B_LTD + self.n[i]) + (1 - sout[i]) * self.n[i] * np.exp(-self.dt / self.tau_d)
            if sout[i]:
                idx = np.where(self.w_out[:, i] > 0)
                self.w_out[idx, i] = self.w_out[idx, i] + self.p[idx]

        self.w_out = np.clip(self.w_out, -1e9, 1e9)
        self.w_out[self.w_out_const0[0], self.w_out_const0[1]] = 0
        self.t_count += 1

        if self.t_count % 1 == 0:
            self.w_history.append(np.mean(self.w_out))
            self.w_history_by_out.append(np.mean(self.w_out, axis=0))
        return self.R

def init_weights_numpy():
    w_in = rng.standard_normal((N_in, N_res)) * (rng.random((N_in, N_res)) < p_in) / (np.sqrt(N_in) * p_in)

    variance = (N_res * p_res**2)**-1
    w_res = rng.standard_normal((N_res, N_res)) * (rng.random((N_res, N_res)) < p_res) * np.sqrt(variance)
    for i in range(N_res):
        QS = np.where(np.abs(w_res[:, i]) > 0)[0]
        if len(QS) > 0:
            w_res[QS, i] = w_res[QS, i] - np.sum(w_res[QS, i]) / len(QS)

    mask = (rng.random((N_res, N_out)) < p_out).astype(float)
    w_out = rng.standard_normal((N_res, N_out)) * mask / (np.sqrt(N_out) * p_out) * G
    w_out_const0 = np.where(mask == 0)
    return w_in, w_res, w_out, w_out_const0

def run_numpy_impl(input_current, boundary_steps=None, v_res_list=None, v_out_list=None):
    """境界で状態のみリセット（重みは保持）"""
    global v_init_res, v_init_out

    if v_res_list is not None and v_out_list is not None:
        v_init_res = v_res_list[0].copy()
        v_init_out = v_out_list[0].copy()
    else:
        v_init_res = v_reset + (v_peak - v_reset) * rng.random(N_res)
        v_init_out = v_reset + (v_peak - v_reset) * rng.random(N_out)

    w_in, w_res, w_out, w_out_const0 = init_weights_numpy()
    w_in_init  = w_in.copy()
    w_res_init = w_res.copy()
    w_out_init = w_out.copy()

    neuron_res = LIF_py(N_res, dt_s, np.ones(N_res), t_ref_py=2*1e-3, tau_m_py=10*1e-3,
                        v_reset=v_reset, v_peak=v_peak, v_thr=v_thr)
    synapse_res = Synapse_res_py(N_res, dt_s, w_res, tau_d=20*1e-3, tau_r=2*1e-3)
    neuron_out = LIF_py(N_out, dt_s, np.ones(N_out), t_ref_py=2*1e-3, tau_m_py=10*1e-3,
                        v_reset=v_reset, v_peak=v_peak, v_thr=v_thr)
    synapse_out = Synapse_out_py(N_out, N_res, dt_s, w_out, w_out_const0, tau_d=20*1e-3, tau_r=2*1e-3)

    # 初期状態
    neuron_res.initialize_states(); neuron_out.initialize_states()
    neuron_res.v_k = v_init_res.copy(); neuron_out.v_k = v_init_out.copy()
    synapse_res.initialize_states(); synapse_out.initialize_states()

    nt = input_current.shape[1]
    I_syn_res_rec = np.zeros((nt, N_res))
    I_syn_out_rec = np.zeros((nt, N_out))
    H_res_rec = np.zeros((nt, N_res))
    V_res_rec = np.zeros((nt, N_res))
    V_out_rec = np.zeros((nt, N_out))
    H_out_rec = np.zeros((nt, N_out))
    spike_times_res, spike_ids_res = [], []
    spike_times_out, spike_ids_out = [], []

    I_syn = np.zeros(N_res)
    I_syn_out = np.zeros(N_out)
    sres = np.zeros(N_res, dtype=int)
    sout = np.zeros(N_out, dtype=int)

    boundary_set = set(boundary_steps or [])
    b2idx = {b: i+1 for i, b in enumerate(boundary_steps or [])}

    # --- 計測開始（自作forループのみ） ---
    t0 = time.perf_counter()
    for t in range(nt):
        # ★ 素材境界：状態だけリセット（重みは保持）
        if t in boundary_set:
            neuron_res.initialize_states()
            neuron_out.initialize_states()
            if v_res_list is not None and v_out_list is not None:
                k = b2idx[t]
                neuron_res.v_k = v_res_list[k].copy()
                neuron_out.v_k = v_out_list[k].copy()
            else:
                neuron_res.v_k = v_reset + (v_peak - v_reset) * rng.random(N_res)
                neuron_out.v_k = v_reset + (v_peak - v_reset) * rng.random(N_out)
            neuron_res.v_ = neuron_res.v_k.copy()
            neuron_out.v_ = neuron_out.v_k.copy()
            synapse_res.initialize_states()
            synapse_out.initialize_states()
            I_syn[:] = 0.0
            I_syn_out[:] = 0.0
            sres[:] = 0
            sout[:] = 0

        I_input = input_current[:, t] @ w_in
        I = I_syn + BIAS + I_input
        sres = neuron_res(I, 0)
        I_syn = synapse_res(sres)
        I_syn = G * I_syn
        sout = neuron_out(I_syn_out + BIAS, 0)
        I_syn_out = synapse_out(sres, sout)

        V_res_rec[t] = neuron_res.v_.copy()
        V_out_rec[t] = neuron_out.v_.copy()
        I_syn_res_rec[t] = I_syn.copy()
        I_syn_out_rec[t] = I_syn_out.copy()
        H_res_rec[t] = synapse_res.H.copy()
        H_out_rec[t] = synapse_out.H.copy()

        if np.any(sres):
            fired = np.where(sres == 1)[0]
            spike_times_res.extend([t * dt_ms for _ in fired])
            spike_ids_res.extend(fired.tolist())
        if np.any(sout):
            fired = np.where(sout == 1)[0]
            spike_times_out.extend([t * dt_ms for _ in fired])
            spike_ids_out.extend(fired.tolist())
    loop_time = time.perf_counter() - t0
    # --- 計測終了 ---

    results = {
        "w_in": w_in, "w_res": w_res, "w_out": w_out,
        "w_in_init":  w_in_init,
        "w_res_init": w_res_init,
        "w_out_init": w_out_init,
        "V_res": V_res_rec[:, :10], "V_out": V_out_rec[:, :10],
        "I_syn_res": I_syn_res_rec[:, :10], "I_syn_out": I_syn_out_rec[:, :10],
        "H_res": H_res_rec,
        "H_out": H_out_rec,
        "spike_times_res": np.array(spike_times_res),
        "spike_ids_res": np.array(spike_ids_res),
        "spike_times_out": np.array(spike_times_out),
        "spike_ids_out": np.array(spike_ids_out),
        "w_history": np.array(synapse_out.w_history),
        "w_history_by_out": np.array(synapse_out.w_history_by_out),
        "wall_time": loop_time,
    }
    return results

# ===============================
# 5) Brian2 実装（境界リセット対応）
# ===============================
def run_brian2_impl(input_current, w_in_np, w_res_np, w_out_np, boundary_steps=None,
                    v_res_list=None, v_out_list=None):
    """境界で v/R/H と STDPトレースのみ初期化。重みは保持。"""
    global v_init_res, v_init_out
    seed(2)

    LIF = '''
    dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
    spiked : 1
    I_exc : 1
    I_inh : 1
    tau_m : second
    t_ref : second
    '''
    double_exp_res= '''
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
    I_syn =  R : 1 
    '''
    on_pre_out = '''
    H_post += (w_out / (tau_r * tau_d)) / Hz
    '''
    STDP='''
    dApre/dt  = -Apre/tau_r  : 1 (event-driven)
    dApost/dt = -Apost/tau_d : 1 (event-driven)
    w_out : 1
    eps_w : 1
    '''
    STDP_pre = '''
    Apre +=   A_B_LTP
    w_out = clip(w_out + int(w_out > eps_w) * Apost, wmin, wmax)
    '''
    STDP_post = '''
    Apost +=  A_B_LTD
    w_out  = clip(w_out + int(w_out > eps_w) * Apre,  wmin, wmax)
    '''

    eqs_res = double_exp_res + LIF
    eqs_out = double_exp_out + LIF

    G_res = NeuronGroup(
        N_res, eqs_res,
        threshold='v >= v_thr',
        reset='''v = v_peak; spiked = 1''',
        refractory= 'timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
        method='exact'
    )
    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold='v >= v_thr',
        reset='''v = v_peak; spiked = 1''',
        refractory= 'timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
        method='exact'
    )
    G_res.tau_m = 10*ms; G_res.t_ref = 2*ms
    G_out.tau_m = 10*ms; G_out.t_ref = 2*ms


    # t=0 初期膜電位（共有値があればそちらを使う）
    if v_res_list is not None and v_out_list is not None:
        G_res.v = v_res_list[0]
        G_out.v = v_out_list[0]
    else:
        if v_init_res is None or v_init_out is None:
            v_init_res = v_reset + (v_peak - v_reset) * rng.random(N_res)
            v_init_out = v_reset + (v_peak - v_reset) * rng.random(N_out)
        G_res.v = v_init_res
        G_out.v = v_init_out

    # --- Reservoir synapses（非ゼロのみ接続）
    S_res = Synapses(G_res, G_res, model = ''' w_res : 1 ''' , on_pre = on_pre_res , method = 'euler')
    S_res.connect(condition='i != j')
    S_res.w_res = w_res_np[S_res.i, S_res.j]
    S_res.delay = 0*ms



    # --- Output synapses（正負ともに使用。学習は w_out>0 のみ）
    oi, oj = np.where(w_out_np != 0)
    S_out = Synapses(G_res, G_out, model=STDP, on_pre=on_pre_out + STDP_pre, on_post=STDP_post, method = 'euler')
    if len(oi) > 0:
        S_out.connect(i=oi, j=oj)
        S_out.w_out[:] = w_out_np[oi, oj]
    else:
        S_out.connect(False)
    S_out.eps_w =  1e-12
    S_out.delay = 0*ms


    prev_spiked_res = np.zeros(N_res)
    prev_spiked_out = np.zeros(N_out)

    nt = input_current.shape[1]
    G_res.R = 0; G_res.H = 0 
    G_out.R = 0; G_out.H = 0

    boundary_set = set(boundary_steps or [])
    b2idx = {b: i+1 for i, b in enumerate(boundary_steps or [])}

    @network_operation(dt=dt_ms*ms, when='start')
    def soft_reset_on_boundary():
        idx = int((defaultclock.t) / (dt_ms*ms))
        if idx in boundary_set:
            if v_res_list is not None and v_out_list is not None:
                k = b2idx[idx]
                G_res.v = v_res_list[k]
                G_out.v = v_out_list[k]
            else:
                G_res.v = v_reset + (v_peak - v_reset) * rng.random(N_res)
                G_out.v = v_reset + (v_peak - v_reset) * rng.random(N_out)
            G_res.R = 0; G_res.H = 0; 
            G_out.R = 0; G_out.H = 0; 
            G_res.spiked = 0; G_out.spiked = 0
            prev_spiked_res[:] = 0; prev_spiked_out[:] = 0

    @network_operation(dt=dt_ms*ms)
    def apply_input():
        nonlocal prev_spiked_res, prev_spiked_out
        idx = int(((defaultclock.t) / (dt_ms*ms)))
        if idx >= input_current.shape[1]:
            return
        I_input = input_current[:, idx] @ w_in_np
        G_res.I_exc = I_input; G_res.I_inh = 0
        G_out.I_exc = 0;       G_out.I_inh = 0

        # 直前に発火したものを v_reset へ（v_peak 表示を1刻みで消す）
        fired_res_prev = np.where(prev_spiked_res == 1)[0]
        fired_out_prev = np.where(prev_spiked_out == 1)[0]
        if len(fired_res_prev) > 0:
            G_res.v[fired_res_prev] = v_reset
            G_res.spiked[fired_res_prev] = 0
            prev_spiked_res[fired_res_prev] = 0
        if len(fired_out_prev) > 0:
            G_out.v[fired_out_prev] = v_reset
            G_out.spiked[fired_out_prev] = 0
            prev_spiked_out[fired_out_prev] = 0

        prev_spiked_res = np.array(G_res.spiked)
        prev_spiked_out = np.array(G_out.spiked)

    w_hist_list = []
    w_hist_by_out_list = []
    step = {'k': 0}
    @network_operation(dt=dt_ms*ms, when='end')
    def track_w():
        step['k'] += 1
        # 全要素平均（未接続は0として数える）
        w_mean = np.sum(S_out.w_out) / (N_res * N_out)
        w_sum_by_out = np.bincount(np.array(S_out.j), weights=np.array(S_out.w_out), minlength=N_out)
        w_by_out = w_sum_by_out / N_res
        if step['k'] % 1 == 0:
            w_hist_list.append(w_mean)
            w_hist_by_out_list.append(w_by_out)

    Msp_res = SpikeMonitor(G_res)
    Msp_out = SpikeMonitor(G_out)
    MV_res  = StateMonitor(G_res, 'v',     record=range(min(10, N_res)))
    MV_out  = StateMonitor(G_out, 'v',     record=range(min(10, N_out)))
    MI_res  = StateMonitor(G_res, 'I_syn', record=range(min(10, N_res)))
    MI_out  = StateMonitor(G_out, 'I_syn', record=range(min(10, N_out)))
    MH_res = StateMonitor(G_res, 'H', record=True)
    MH_out = StateMonitor(G_out, 'H', record=True)

    # --- 計測開始（Brian2 の run() のみ） ---
    t0 = time.perf_counter()
    run((nt * dt_ms) * ms)
    run_time = time.perf_counter() - t0
    # --- 計測終了 ---

    results = {
        "V_res": np.vstack([MV_res.v[k] for k in range(min(10,N_res))]).T,
        "V_out": np.vstack([MV_out.v[k] for k in range(min(10,N_out))]).T,
        "I_syn_res": np.vstack([MI_res.I_syn[k] for k in range(min(10,N_res))]).T,
        "I_syn_out": np.vstack([MI_out.I_syn[k] for k in range(min(10,N_out))]).T,
        "H_res": MH_res.H.T,
        "H_out": MH_out.H.T,
        "spike_times_res": np.array(Msp_res.t/ms), "spike_ids_res": np.array(Msp_res.i),
        "spike_times_out": np.array(Msp_out.t/ms), "spike_ids_out": np.array(Msp_out.i),
        "w_history": np.array(w_hist_list),
        "w_history_by_out": np.array(w_hist_by_out_list),
        "wall_time": run_time,
    }
    return results

# ===============================
# 6) 実行 & 可視化（連続時間・境界で状態リセット）
# ===============================
if __name__ == "__main__":
    # 入力を素材ごとに連結
    inputs = []
    nt_list = []
    boundaries = [0]  # サンプル境界の累積（先頭は0）
    for m_idx, mat in enumerate(materials):
        _inp, _nt = load_one_trial(mat, i_size=0)
        inputs.append(_inp)
        nt_list.append(_nt)
        boundaries.append(boundaries[-1] + _nt)

    input_current_all = np.concatenate(inputs, axis=1)  # (N_in, sum_t)
    nt_total = input_current_all.shape[1]
    time_ms_all = np.arange(nt_total) * dt_ms

    # 「境界」配列（先頭と末尾を除いたステップ番号）
    boundary_steps = boundaries[1:-1]

    # ---- 共有初期値を一度だけ作る（専用RNGで） ----
    rng_shared = default_rng(999)  # 自作/Brian2共通に使う
    K = len(boundary_steps)
    v0_res = v_reset + (v_peak - v_reset) * rng_shared.random(N_res)
    v0_out = v_reset + (v_peak - v_reset) * rng_shared.random(N_out)
    v_res_list = [v0_res] + [v_reset + (v_peak - v_reset) * rng_shared.random(N_res) for _ in range(K)]
    v_out_list = [v0_out] + [v_reset + (v_peak - v_reset) * rng_shared.random(N_out) for _ in range(K)]

    # 実行（重み初期値は NumPy 実装から）
    res_np = run_numpy_impl(input_current_all, boundary_steps=boundary_steps,
                            v_res_list=v_res_list, v_out_list=v_out_list)
    res_br = run_brian2_impl(
        input_current_all,
        w_in_np  = res_np["w_in_init"],
        w_res_np = res_np["w_res_init"],
        w_out_np = res_np["w_out_init"],
        boundary_steps=boundary_steps,
        v_res_list=v_res_list,
        v_out_list=v_out_list,
    )

    # 実行時間
    print(f"[TIME] Custom_time:     {res_np['wall_time']:.3f} s")
    print(f"[TIME] Brian2 run_time: {res_br['wall_time']:.3f} s")

    # ========= 可視化（連続時間。境界に縦線＆素材名） =========
    from matplotlib import colormaps as cm
    _CMAP_RES = cm.get_cmap('tab20')
    _CMAP_OUT = cm.get_cmap('tab20')

    def color_for_res(i: int):
        return _CMAP_RES(i % _CMAP_RES.N)

    def color_for_out(i: int):
        return _CMAP_OUT(i % _CMAP_OUT.N)

    def _draw_boundaries(ax):
        for b in boundary_steps:
            ax.axvline(b * dt_ms, color='k', lw=1, alpha=0.2)

    def _annotate_materials(ax):
        if len(materials) <= 1: return
        ymin, ymax = ax.get_ylim()
        for i in range(len(materials)):
            start = boundaries[i] * dt_ms
            end   = boundaries[i+1] * dt_ms
            cx = (start + end) / 2
            ax.text(cx, ymax, materials[i], ha='center', va='bottom', fontsize=9, alpha=0.7)

    IDX=2; RES_IDX=IDX; OUT_IDX=IDX

    # Raster: Reservoir
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(res_br["spike_times_res"], res_br["spike_ids_res"], '.', markersize=1, label="Brian2")
    ax.plot(res_np["spike_times_res"], res_np["spike_ids_res"], '.', markersize=1, label="Custom")
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Reservoir neuron index')
    ax.set_title('Reservoir Raster')
    ax.legend(); fig.tight_layout(); plt.show()

    # Raster: Output
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(res_br["spike_times_out"], res_br["spike_ids_out"], '.', markersize=1, label="Brian2")
    ax.plot(res_np["spike_times_out"], res_np["spike_ids_out"], '.', markersize=1, label="Custom")
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Output neuron index')
    ax.set_title('Output Raster')
    ax.legend(); fig.tight_layout(); plt.show()

    # I_syn (Reservoir)：1本
    fig, ax = plt.subplots(figsize=(10,5))
    _c = color_for_res(RES_IDX)
    ax.plot(time_ms_all, res_br["I_syn_res"][:, RES_IDX], label=f'Brian2 Res {RES_IDX}', color=_c)
    ax.plot(time_ms_all, res_np["I_syn_res"][:, RES_IDX], '--', label=f'Custom Res {RES_IDX}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('I_syn (Reservoir)')
    ax.set_title(f'Reservoir Synaptic Current (Neuron {RES_IDX})')
    ax.legend(); fig.tight_layout(); plt.show()

    # I_syn (Output)：1本
    fig, ax = plt.subplots(figsize=(10,5))
    _c = color_for_out(OUT_IDX)
    ax.plot(time_ms_all, res_br["I_syn_out"][:, OUT_IDX], label=f'Brian2 Out {OUT_IDX}', color=_c)
    ax.plot(time_ms_all, res_np["I_syn_out"][:, OUT_IDX], '--', label=f'Custom Out {OUT_IDX}', color=_c)
    
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('I_syn (Output)')
    ax.set_title(f'Output Synaptic Current (Neuron {OUT_IDX})')
    ax.legend(loc='upper left', fontsize=8); fig.tight_layout(); plt.show()
    # ==== 平均 w_out（OUT_IDX に接続）専用グラフ（新しい図） ====
    fig, ax = plt.subplots(figsize=(10,5))
    # Brian2 側
    if "w_history_by_out" in res_br:
        time_ms_w_b2_ = np.arange(len(res_br["w_history_by_out"])) * dt_ms
        ax.plot(time_ms_w_b2_, res_br["w_history_by_out"][:, OUT_IDX], label=f'B2 mean(w_out→{OUT_IDX})')
    elif "w_history" in res_br:
        time_ms_w_b2_ = np.arange(len(res_br["w_history"])) * dt_ms
        ax.plot(time_ms_w_b2_, res_br["w_history"], label='B2 mean(w_out)')
    # 自作（NumPy）側
    if "w_history_by_out" in res_np:
        time_ms_w_np_ = np.arange(len(res_np["w_history_by_out"])) * dt_ms
        ax.plot(time_ms_w_np_, res_np["w_history_by_out"][:, OUT_IDX], '--', label=f'Custom mean(w_out→{OUT_IDX})')
    elif "w_history" in res_np:
        time_ms_w_np_ = np.arange(len(res_np["w_history"])) * dt_ms
        ax.plot(time_ms_w_np_, res_np["w_history"], '--', label='Custom mean(w_out)')
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Average w_out (to OUT_IDX)')
    ax.set_title(f'Average Output Weight to Neuron {OUT_IDX}')
    ax.legend(loc='best', fontsize=8); fig.tight_layout(); plt.show()


    # H (Reservoir)
    fig, ax = plt.subplots(figsize=(10,5))
    _c = color_for_res(OUT_IDX)
    ax.plot(time_ms_all, res_br["H_res"][:, OUT_IDX], label=f'B2 H res {OUT_IDX}', color=_c)
    ax.plot(time_ms_all, res_np["H_res"][:, OUT_IDX], '--', label=f'Custom H res {OUT_IDX}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('H (res; summed incoming)')
    ax.set_title(f'res H (Neuron {OUT_IDX})')
    ax.legend(); fig.tight_layout(); plt.show()

    # H (Output)
    fig, ax = plt.subplots(figsize=(10,5))
    _c = color_for_out(OUT_IDX)
    ax.plot(time_ms_all, res_br["H_out"][:, OUT_IDX], label=f'B2 H Out {OUT_IDX}', color=_c)
    ax.plot(time_ms_all, res_np["H_out"][:, OUT_IDX], '--', label=f'Custom H Out {OUT_IDX}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('H (Output; summed incoming)')
    ax.set_title(f'Output H (Neuron {OUT_IDX})')
    ax.legend(); fig.tight_layout(); plt.show()

    # v (Reservoir)
    fig, ax = plt.subplots(figsize=(10,5))
    _c = color_for_res(RES_IDX)
    ax.plot(time_ms_all, res_br["V_res"][:, RES_IDX], label=f'Brian2 Res {RES_IDX}', color=_c)
    ax.plot(time_ms_all, res_np["V_res"][:, RES_IDX], '--', label=f'Custom Res {RES_IDX}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Membrane potential v (Reservoir)')
    ax.set_title(f'Reservoir Membrane Potential (Neuron {RES_IDX})')
    ax.legend(); fig.tight_layout(); plt.show()

    # v (Output)
    fig, ax = plt.subplots(figsize=(10,5))
    _c = color_for_out(OUT_IDX)
    ax.plot(time_ms_all, res_br["V_out"][:, OUT_IDX], label=f'Brian2 Out {OUT_IDX}', color=_c)
    ax.plot(time_ms_all, res_np["V_out"][:, OUT_IDX], '--', label=f'Custom Out {OUT_IDX}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Membrane potential v (Output)')
    ax.set_title(f'Output Membrane Potential (Neuron {OUT_IDX})')
    ax.legend(); fig.tight_layout(); plt.show()

    # w_out 推移（平均）
    time_ms_w_b2 = np.arange(len(res_br["w_history"])) * dt_ms
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(time_ms_w_b2, res_br["w_history"], label='Brian2 mean(w_out)')
    if "w_history" in res_np:
        time_ms_w_np = np.arange(len(res_np["w_history"])) * dt_ms
        ax.plot(time_ms_w_np, res_np["w_history"], label='Custom mean(w_out)')
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_ylabel('Average w_out'); ax.set_xlabel('Time (ms)')
    ax.set_title('Evolution of Output Synaptic Weights')
    ax.legend(); fig.tight_layout(); plt.show()

    # 10本：I_syn (Reservoir)
    IDX = 0
    RES_IDX_LIST = [(IDX + i) % N_res for i in range(10)]
    OUT_IDX_LIST = [(IDX + i) % N_out for i in range(10)]
    fig, ax = plt.subplots(figsize=(10,5))
    for k in RES_IDX_LIST:
        _c = color_for_res(k)
        ax.plot(time_ms_all, res_br["I_syn_res"][:, k], label=f'B2 Res {k}', color=_c)
        ax.plot(time_ms_all, res_np["I_syn_res"][:, k], '--', label=f'Custom Res {k}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('I_syn (Reservoir)')
    ax.set_title(f'Reservoir Synaptic Current (10 neurons: {RES_IDX_LIST})')
    ax.legend(ncol=2, fontsize=8); fig.tight_layout(); plt.show()

    # 10本：I_syn (Output)
    fig, ax = plt.subplots(figsize=(10,5))
    for k in OUT_IDX_LIST:
        _c = color_for_out(k)
        ax.plot(time_ms_all, res_br["I_syn_out"][:, k], label=f'B2 Out {k}', color=_c)
        ax.plot(time_ms_all, res_np["I_syn_out"][:, k], '--', label=f'Custom Out {k}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('I_syn (Output)')
    ax.set_title(f'Output Synaptic Current (10 neurons: {OUT_IDX_LIST})')
    ax.legend(ncol=2, fontsize=8); fig.tight_layout(); plt.show()

    # 10本：v (Reservoir)
    fig, ax = plt.subplots(figsize=(10,5))
    for k in RES_IDX_LIST:
        _c = color_for_res(k)
        ax.plot(time_ms_all, res_br["V_res"][:, k], label=f'B2 Res v {k}', color=_c)
        ax.plot(time_ms_all, res_np["V_res"][:, k], '--', label=f'Custom Res v {k}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Membrane potential v (Res)')
    ax.set_title(f'Reservoir Membrane Potential (10 neurons: {RES_IDX_LIST})')
    ax.legend(ncol=2, fontsize=8); fig.tight_layout(); plt.show()

    # 10本：v (Output)
    fig, ax = plt.subplots(figsize=(10,5))
    for k in OUT_IDX_LIST:
        _c = color_for_out(k)
        ax.plot(time_ms_all, res_br["V_out"][:, k], label=f'B2 Out v {k}', color=_c)
        ax.plot(time_ms_all, res_np["V_out"][:, k], '--', label=f'Custom Out v {k}', color=_c)
    _draw_boundaries(ax); _annotate_materials(ax)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Membrane potential v (Out)')
    ax.set_title(f'Output Membrane Potential (10 neurons: {OUT_IDX_LIST})')
    ax.legend(ncol=2, fontsize=8); fig.tight_layout(); plt.show()
