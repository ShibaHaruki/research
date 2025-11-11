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

pdf = PdfPages("results.pdf")  # 好きなファイル名にしてOK

start_scope()

path = str(Path(__file__).resolve().parents[1]) + "/"
dir_name = ["Al_board", "buta_omote", "buta_ura",
            "cork", "denim", "rubber_board", "washi", "wood_board"]

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
N_out = 40

# circuit connection probability
p_in = 0.4
p_res = 0.1
p_out = 0.2

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

# output
neuron_array_out = np.ones(N_out)
out_exc_idx = np.where(neuron_array_out == 1)[0]
out_inh_idx = np.where(neuron_array_out == -1)[0]
t_ref_out = np.zeros(N_out)
tau_m_out = np.zeros(N_out)
tau_m_out = np.where(neuron_array_out == 1, 10,
                     np.where(neuron_array_out == -1, 10, 10))
t_ref_out = np.where(neuron_array_out == 1, 2,
                     np.where(neuron_array_out == -1, 2, 2))

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
A = 0.01
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

mask_out = (np.random.rand(N_res, N_out) < p_out).astype(float)
w_out_init = np.random.randn(N_res, N_out) * mask_out / (np.sqrt(N_out)*p_out) * A

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

STDP = '''
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

G_out = NeuronGroup(
    N_out, eqs_out,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_out.tau_m = tau_m_out*ms
G_out.t_ref = t_ref_out*ms

S_res = Synapses(G_res, G_res, model='w_res : 1', on_pre=on_pre_res, method='euler')
S_res.connect(condition='i != j')
S_res.w_res = w_res_init[S_res.i, S_res.j]
S_res.delay = 0*ms

S_out = Synapses(G_res, G_out, model=STDP, on_pre=on_pre_out + STDP_pre,
                 on_post=STDP_post, method='euler')
S_out.connect(condition='i != j')
S_out.w_out = w_out_init[S_out.i, S_out.j]
S_out.eps_w = 1e-12
S_out.delay = 0*ms


# -------- monitors --------
Msp_res = SpikeMonitor(G_res)
Msp_out = SpikeMonitor(G_out)

M_Ires = StateMonitor(G_res, 'I_syn', record=range(min(10, N_res)))
M_Iout = StateMonitor(G_out, 'I_syn', record=range(min(10, N_out)))

M_Vres = StateMonitor(G_res, 'v', record=range(min(10, N_res)))
M_Vout = StateMonitor(G_out, 'v', record=range(min(10, N_out)))

M_wout = StateMonitor(S_out, 'w_out', record=True)

# w_out は StateMonitor しない（メモリ対策）
wout_time_list = []
wout_mean_list = []

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
            G_out.v = v_reset + (v_thr - v_reset)*np.random.rand(N_out)
            G_res.R = 0; G_res.H = 0
            G_out.R = 0; G_out.H = 0
            S_out.Apre = 0
            S_out.Apost = 0

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

# ======= 重み保存 =======
np.save("w_in.npy", w_in)
np.save("w_res.npy", w_res_init)
w_out_dense = np.zeros((N_res, N_out))
w_out_dense[np.array(S_out.i[:]), np.array(S_out.j[:])] = np.array(S_out.w_out[:])
np.save("w_out.npy", w_out_dense)

########### パラメータ保存 ###########
with open("params1.txt", "w") as file:
    file.write("N_in = " + str(N_in) + "\n")
    file.write("N_res = " + str(N_res) + "\n")
    file.write("N_out = " + str(N_out) + "\n")
    file.write("p_in = " + str(p_in) + "\n")
    file.write("p_res = " + str(p_res) + "\n")
    file.write("p_out = " + str(p_out) + "\n")
    file.write("G = " + str(G) + "\n")
    file.write("dt = " + str(dt_ms) + "\n")
    file.write("tau_d = " + str(20*1e-3) + "\n")
    file.write("tau_r = " + str(2*1e-3) + "\n")
    file.write("t_ref = " + str(2*1e-3) + "\n")
    file.write("v_reset = " + str(-65) + "\n")
    file.write("v_peak = " + str(30) + "\n")
    file.write("v_thr = " + str(-40) + "\n")
    file.write("A_B_LTP = " + str(0.06) + "\n")
    file.write("A_B_LTD = " + str(-0.05) + "\n")
    file.write("BIAS = " + str(BIAS) + "\n")
###########

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

    fig = plt.figure(...)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(dist_mat, origin='lower')
    plt.colorbar(im, label="time-averaged state distance")
    plt.xticks(range(n_labels), unique_labels, rotation=45, ha="right")
    plt.yticks(range(n_labels), unique_labels)
    plt.title("Reservoir separation matrix (material vs material)")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ========= Al_board vs 他素材の距離の「時系列」 =========
    target_label = "Al_board"
    if target_label in unique_labels:
        plt.figure(figsize=(8, 4))
        for other_label in unique_labels:
            if other_label == target_label:
                continue
            pair_key = tuple(sorted((target_label, other_label)))
            if pair_key in d_series_by_pair:
                d_list = d_series_by_pair[pair_key]  # list of (T,)
                d_arr = np.array(d_list)
                mean_d_ts = np.mean(d_arr, axis=0)
                plt.plot(TS, mean_d_ts, label=other_label)

        fig = plt.figure(...)
        plt.xlabel("time (ms)")
        plt.ylabel("state distance (mean squared)")
        plt.title("Al_board vs other materials (time series)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# ======= 可視化（ラスタ・ヒスト・電流・電位・w_out） =======
# Raster (Reservoir)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
plt.plot(Msp_res.t/ms, Msp_res.i, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir neuron index')
total_time_ms = float(defaultclock.t/ms)
plt.title(f'Reservoir Raster Plot (total {total_time_ms:.1f} ms)')
plt.tight_layout()
pdf.savefig(fig) 

# Raster (Output)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
plt.plot(Msp_out.t/ms, Msp_out.i, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Output neuron index')
plt.title('Output layer raster plot')
plt.tight_layout()
pdf.savefig(fig) 

# 発火率ヒスト (Reservoir)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
if len(Msp_res.t) > 0:
    weights_res = np.ones(len(Msp_res.t)) / (N_res * dt_s)  # [spikes/(neuron*s)]
    plt.hist(np.array(Msp_res.t/ms), bins=200,
             histtype='stepfilled', weights=weights_res)
plt.xlabel('Time (ms)')
plt.ylabel('Instantaneous firing rate (sp/s)')
plt.tight_layout()
pdf.savefig(fig) 

# 発火率ヒスト (Output)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
if len(Msp_out.t) > 0:
    weights_out = np.ones(len(Msp_out.t)) / (N_out * dt_s)
    plt.hist(np.array(Msp_out.t/ms), bins=200,
             histtype='stepfilled', weights=weights_out)
plt.xlabel('Time (ms)')
plt.ylabel('Instantaneous firing rate (sp/s)')
plt.tight_layout()
pdf.savefig(fig)

# I_syn (Reservoir)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
for i_plot in range(min(10, N_res)):
    plt.plot(M_Ires.t/ms, M_Ires.I_syn[i_plot], label=f'Res neuron {i_plot}')
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir Isyn')
plt.tight_layout()
pdf.savefig(fig)

# I_syn (Output)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
for i_plot in range(min(10, N_out)):
    plt.plot(M_Iout.t/ms, M_Iout.I_syn[i_plot], label=f'Output neuron {i_plot}')
plt.xlabel('Time (ms)')
plt.ylabel('Output Isyn')
plt.tight_layout()
pdf.savefig(fig)


# v (Reservoir)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
for i_plot in range(min(10, N_res)):
    plt.plot(M_Vres.t/ms, M_Vres.v[i_plot], label=f'Res neuron {i_plot}')
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir v')
plt.tight_layout()
pdf.savefig(fig)


# v (Output)
fig = plt.figure(figsize=(10, 5))
plt.figure(figsize=(10, 5))
for i_plot in range(min(10, N_out)):
    plt.plot(M_Vout.t/ms, M_Vout.v[i_plot], label=f'Output neuron {i_plot}')
plt.xlabel('Time (ms)')
plt.ylabel('Output v')
plt.tight_layout()
pdf.savefig(fig)



# w_out 平均
fig = plt.figure(figsize=(10, 5))
wout_mean_syn = M_wout.w_out.mean(axis=0)
plt.figure(figsize=(10, 5))
plt.plot(M_wout.t/ms, wout_mean_syn)
plt.xlabel("time (ms)")
plt.ylabel("mean w_out (all synapses)")
plt.tight_layout()
pdf.savefig(fig)

plt.show()

pdf.close()
