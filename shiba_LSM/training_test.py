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
import re 

pdf = PdfPages("training_test.pdf")  

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
tau_m_res  = np.where(neuron_array_res == 1, 10, np.where(neuron_array_res == -1, 10, 10))
t_ref_res  = np.where(neuron_array_res == 1, 2, np.where(neuron_array_res == -1, 2, 2))

# output
neuron_array_out = np.ones(N_out)
out_exc_idx = np.where(neuron_array_out == 1)[0]
out_inh_idx = np.where(neuron_array_out == -1)[0]
t_ref_out = np.zeros(N_out)
tau_m_out = np.zeros(N_out)
tau_m_out = np.where(neuron_array_out == 1, 10, np.where(neuron_array_out == -1, 10, 10))
t_ref_out = np.where(neuron_array_out == 1, 2, np.where(neuron_array_out == -1, 2, 2))

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

S_out = Synapses(G_res, G_out, model=STDP, on_pre=on_pre_out + STDP_pre, on_post=STDP_post, method='euler')
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
    idx = int(((defaultclock.t - t0) / (dt_ms*ms))) 
    I_input = input_current[:, idx] @ w_in
    G_res.I_exc = I_input 
    G_res.I_inh = 0


# ======= 実行ループ =======
start_time = time.perf_counter()

for epo in range(1):
    for i_size in range(1):
        for i in  dir_name:
            df = pd.read_table(glob.glob(path + "tactile_data/" + i + f"/data_{int(sample_seq[i_size])}_*")[0],header=None)
            df_np = df.to_numpy().T
            in_data_0 = df_np[:3, 3000:8000]
            nt = in_data_0.shape[1]
            t_array_s = np.arange(nt) * dt_s 
        
            input_current = np.zeros([N_in, nt]) 

            for j in range(3):
                in_data = in_data_0[j,:] 

                I_merkel   = calc_merkel(in_data,   t_array_s, dt_s)
                I_meissner = calc_meissner(in_data, t_array_s, dt_s)
				
                if j==0 :  
                    input_current[j*2 , :] = 0.01*I_merkel
                    input_current[j*2+1, :] = 0.04*I_meissner

            G_res.v = v_reset + (v_thr - v_reset)* np.random.rand(N_res)
            G_out.v = v_reset + (v_thr - v_reset)* np.random.rand(N_out)
            G_res.R = 0; G_res.H = 0
            G_out.R = 0; G_out.H = 0
            S_out.Apre = 0
            S_out.Apost = 0
            
            run((nt*dt_ms)*ms)
            t0 += (nt*dt_ms)*ms

        print(str(i_size)+".")



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

# Raster (Output)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Msp_out.t/ms, Msp_out.i, '.', markersize=1)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Output neuron index')
ax.set_title('Output layer raster plot')
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

# 発火率ヒスト (Output)
fig, ax = plt.subplots(figsize=(10, 5))
if len(Msp_out.t) > 0:
    weights_out = np.ones(len(Msp_out.t)) / (N_out * dt_s)
    ax.hist(np.array(Msp_out.t/ms), bins=200,
            histtype='stepfilled', weights=weights_out)
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

# I_syn (Output)
fig, ax = plt.subplots(figsize=(10, 5))
for i_plot in range(min(10, N_out)):
    ax.plot(M_Iout.t/ms, M_Iout.I_syn[i_plot], label=f'Output neuron {i_plot}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Output Isyn')
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

# v (Output)
fig, ax = plt.subplots(figsize=(10, 5))
for i_plot in range(min(10, N_out)):
    ax.plot(M_Vout.t/ms, M_Vout.v[i_plot], label=f'Output neuron {i_plot}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Output v')
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

# w_out 平均
fig, ax = plt.subplots(figsize=(10, 5))
wout_mean_syn = M_wout.w_out.mean(axis=0)
ax.plot(M_wout.t/ms, wout_mean_syn)
ax.set_xlabel("time (ms)")
ax.set_ylabel("mean w_out (all synapses)")
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

# ここではサーバでグラフ表示しないなら plt.show() はなくてもOK
# plt.show()

pdf.close()
