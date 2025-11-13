from collections import defaultdict
import numpy as np
rng = np.random.default_rng(2)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import glob
from brian2 import *
prefs.core.default_float_dtype = float64
prefs.codegen.target = 'numpy'
import time  # 時間計測
from pathlib import Path
from tqdm import tqdm 

pdf = PdfPages("results_SP_N_res.pdf")  

start_scope()

path = str(Path(__file__).resolve().parents[1]) + "/"
dir_name = ["Al_board", "buta_omote", "buta_ura","cork", "denim", "rubber_board", "washi", "wood_board"]

sample_seq = np.zeros(324)
for i in range(324):
	sample_seq[i] = int(i+1)
rng.shuffle(sample_seq)
np.save("sample_seq.npy", sample_seq)

# -------- tactile input filters --------
def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i-1])/(t[i]-t[i-1])
            I[0, i] = I[0, i-1] + 1*dF_dt + (-I[0, i-1]*dt/(8*1*1e-3))
            I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200*1e-3))
            I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6*1e-3))
            I[3, i] = I[0, i]
    return I[3, :]

def calc_merkel(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i-1])/(t[i]-t[i-1])
            if dF_dt < 0:
                dF_dt = 0
            I[0, i] = I[0, i-1] + 0.74*dF_dt + (-I[0, i-1]*dt/(8*1*1e-3))
            I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200*1e-3))
            I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6*1e-3))
            I[3, i] = I[0, i] + I[1, i] + I[2, i]
    return I[3, :]

# -------- Separation property 用の補助関数 --------
def exp_conv(t_ms, spikes_ms, tau_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    spikes_ms = np.asarray(spikes_ms, dtype=float)
    if spikes_ms.size == 0:
        return np.zeros_like(t_ms, dtype=float)
    conv = np.zeros_like(t_ms, dtype=float)
    for st in spikes_ms:
        conv += np.exp(-(t_ms - st)/tau_ms) * (t_ms >= st)
    return conv

def liquid_distance(state_u, state_v):
    # state_u, state_v : shape (N_res, T)
    return np.mean((state_u - state_v)**2, axis=0)

##################
# hyper-parameters
N_in = 2
N_res = 500

# circuit connection probability
p_in = 0.4
p_res = 0.1

# LIF params (dimensionless 電位)
v_reset = -65
v_peak  = 30
v_thr   = -40

# reservoir
neuron_array_res = np.ones(N_res)
res_exc_idx = np.where(neuron_array_res == 1)[0]
res_inh_idx = np.where(neuron_array_res == -1)[0]
t_ref_res = np.zeros(N_res)
tau_m_res  = np.zeros(N_res)*second
tau_m_res  = np.where(neuron_array_res == 1, 10,
                      np.where(neuron_array_res == -1, 10, 10))
t_ref_res  = np.where(neuron_array_res == 1, 2,
                      np.where(neuron_array_res == -1, 2, 2))


# double_exponential_synapse
tau_r = 2*ms
tau_d = 20*ms

# STDP
A_B_LTP = 0.006
A_B_LTD = -0.005
wmin = -1e9
wmax = 1e9

BIAS = -40
G = 0.01
###################

# time
dt_ms = 0.1
dt_s  = dt_ms * 1e-3
defaultclock.dt = dt_ms * ms
t0 = 0*ms

# weights
w_in = np.random.randn(N_in, N_res) * (np.random.rand(N_in, N_res) < p_in) / (np.sqrt(N_in)*p_in)

variance = (N_res * p_res**2)**-1
w_res_init = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
for k in range(N_res):
    QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
    if len(QS) > 0:
        w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

# -------- models --------
LIF = '''
dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
spiked : 1
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

# -------- groups --------
G_res = NeuronGroup(
    N_res, eqs_res,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_res.tau_m = tau_m_res*ms
G_res.t_ref = t_ref_res*ms

S_res = Synapses(G_res, G_res, model='w_res : 1', on_pre=on_pre_res, method='euler')
S_res.connect(condition='i != j')
S_res.w_res = w_res_init[S_res.i, S_res.j]
S_res.delay = 0*ms

# -------- monitors --------
Msp_res = SpikeMonitor(G_res)

M_Ires = StateMonitor(G_res, 'I_syn', record=range(min(10, N_res)))

M_Vres = StateMonitor(G_res, 'v', record=range(min(10, N_res)))

# Separation property 用: 各試行の開始時刻とラベル、長さ
trial_start_times_ms = []
trial_labels = []
trial_duration_ms = None

@network_operation(dt=dt_ms*ms)
def apply_input():
    idx = int(((defaultclock.t - t0)/(dt_ms*ms)))
    I_input = input_current[:, idx] @ w_in
    G_res.I_exc = I_input
    G_res.I_inh = 0

    # ======= 実行ループ =======
start_time = time.perf_counter()

for epo in range(1):
    for i_size in range(2):
        for i in tqdm(dir_name):
            df = pd.read_table(glob.glob(path + "tactile_data/" + i + f"/data_{int(sample_seq[i_size])}_*" )[0],header=None)
            df_np = df.to_numpy().T
            in_data_0 = df_np[:3, 3000:8000]
            nt = in_data_0.shape[1]
            t_array_s = np.arange(nt) * dt_s

            input_current = np.zeros([N_in, nt])

            for j in range(3):
                in_data = in_data_0[j, :]
                I_merkel   = calc_merkel(in_data, t_array_s, dt_s)
                I_meissner = calc_meissner(in_data, t_array_s, dt_s)
                if j == 0:
                    input_current[j*2,   :] = 0.01*I_merkel
                    input_current[j*2+1, :] = 0.04*I_meissner

            # initialize_state
            G_res.v = v_reset + (v_thr - v_reset)*np.random.rand(N_res)
            G_res.R = 0; G_res.H = 0
    

            # trial の開始時刻とラベルを記録
            trial_start_times_ms.append(float(defaultclock.t/ms))
            trial_labels.append(i)
            if trial_duration_ms is None:
                trial_duration_ms = nt*dt_ms

            run((nt*dt_ms)*ms)
            t0 += (nt*dt_ms)*ms

        print(str(i_size) + ".")

# ======= 終了 =======
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"[TIME] Brian2 run(): {elapsed:.3f} s")

# ======= Separation property (Reservoir) 評価 =======
n_trials = len(trial_start_times_ms)
if n_trials > 0:
    T_ms = trial_duration_ms
    TS = np.arange(0.0, T_ms, dt_ms)

    # Reservoir spikes 全部
    all_t_res_ms = np.array(Msp_res.t/ms)   # [ms]
    all_i_res    = np.array(Msp_res.i)

    # 各 trial の liquid state: list of (N_res, len(TS))
    liquid_states_res = []

    for k in range(n_trials):
        t_start = trial_start_times_ms[k]
        t_end   = t_start + T_ms

        mask = (all_t_res_ms >= t_start) & (all_t_res_ms < t_end)
        t_trial_ms = all_t_res_ms[mask] - t_start
        i_trial    = all_i_res[mask]

        state_k = np.zeros((N_res, len(TS)), dtype=float)
        for n in range(N_res):
            spikes_n = t_trial_ms[i_trial == n]
            if spikes_n.size > 0:
                state_k[n, :] = exp_conv(TS, spikes_n, tau_ms=30.0)
        liquid_states_res.append(state_k)

    # ========= 素材ペアごとの距離 =========
    d_series_by_pair = defaultdict(list)   # pair_key -> [ d_ab(t) ... ]
    d_scalar_by_pair = defaultdict(list)   # pair_key -> [ mean_t d_ab(t) ... ]

    for a in range(n_trials):
        for b in range(a+1, n_trials):
            label_a = trial_labels[a]
            label_b = trial_labels[b]
            pair_key = tuple(sorted((label_a, label_b)))

            d_ab = liquid_distance(liquid_states_res[a], liquid_states_res[b])  # shape=(T,)
            d_series_by_pair[pair_key].append(d_ab)
            d_scalar_by_pair[pair_key].append(np.mean(d_ab))

    # ========= 素材 × 素材 の距離マトリクス（時間平均距離） =========
    unique_labels = sorted(list(set(trial_labels)))
    n_labels = len(unique_labels)
    dist_mat = np.full((n_labels, n_labels), np.nan, dtype=float)

    for i_lab, la in enumerate(unique_labels):
        for j_lab, lb in enumerate(unique_labels):
            pair_key = tuple(sorted((la, lb)))
            if pair_key in d_scalar_by_pair:
                dist_mat[i_lab, j_lab] = np.mean(d_scalar_by_pair[pair_key])

    # --- 距離マトリクスの図 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist_mat, origin='lower')
    cbar = fig.colorbar(im, ax=ax, label="time-averaged state distance")
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(unique_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(unique_labels)
    ax.set_title("Reservoir separation matrix (material vs material)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ========= Al_board vs 他素材の距離の「時系列」 =========
    target_label = "Al_board"
    if target_label in unique_labels:
        fig, ax = plt.subplots(figsize=(8, 4))
        for other_label in unique_labels:
            if other_label == target_label:
                continue
            pair_key = tuple(sorted((target_label, other_label)))
            if pair_key in d_series_by_pair:
                d_list = d_series_by_pair[pair_key]  # list of (T,)
                d_arr = np.array(d_list)
                mean_d_ts = np.mean(d_arr, axis=0)
                ax.plot(TS, mean_d_ts, label=other_label)

        ax.set_xlabel("time (ms)")
        ax.set_ylabel("state distance (mean squared)")
        ax.set_title("Al_board vs other materials (time series)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# ======= 可視化（ラスタ・ヒスト・電流・電位・w_out） =======

# Raster (Reservoir)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Msp_res.t/ms, Msp_res.i, '.', markersize=1)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Reservoir neuron index')
total_time_ms = float(defaultclock.t/ms)
ax.set_title(f'Reservoir Raster Plot (total {total_time_ms:.1f} ms)')
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)


# 発火率ヒスト (Reservoir)
fig, ax = plt.subplots(figsize=(10, 5))
if len(Msp_res.t) > 0:
    weights_res = np.ones(len(Msp_res.t)) / (N_res * dt_s)  # [spikes/(neuron*s)]
    ax.hist(np.array(Msp_res.t/ms), bins=200,
            histtype='stepfilled', weights=weights_res)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Instantaneous firing rate (sp/s)')
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

# I_syn (Reservoir)
fig, ax = plt.subplots(figsize=(10, 5))
for i_plot in range(min(10, N_res)):
    ax.plot(M_Ires.t/ms, M_Ires.I_syn[i_plot], label=f'Res neuron {i_plot}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Reservoir Isyn')
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

# v (Reservoir)
fig, ax = plt.subplots(figsize=(10, 5))
for i_plot in range(min(10, N_res)):
    ax.plot(M_Vres.t/ms, M_Vres.v[i_plot], label=f'Res neuron {i_plot}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Reservoir v')
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)


pdf.close()
