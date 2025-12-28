import numpy as np
rng = np.random.default_rng(2)
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm 
import pandas as pd
import glob
from pathlib import Path
from brian2 import *
prefs.core.default_float_dtype = float64
prefs.codegen.target = 'numpy'

path = str(Path(__file__).resolve().parents[1]) + "/"
dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]
script_path = Path(__file__).resolve()

sample_seq = np.load("sample_seq.npy")
max = len(sample_seq)
test_seq = sample_seq[100:max]
n_sample = 100

# ===== Brian2 初期化 =====
start_scope()

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
##################
N_in = 2
N_res = 250
N_out = 40

# circuit connection probability
p_in = 0.4
p_res = 0.5
p_out = 0.5
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
wmin = -1
wmax = 1
tau_s = 11.5*ms
tau_t = 14*ms

BIAS = -65
G = 0.11
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
w_res_init1 = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
for k in range(N_res):
    QS = np.where(np.abs(w_res_init1[:, k]) > 0)[0]
    if len(QS) > 0:
        w_res_init1[QS, k] -= np.mean(w_res_init1[QS, k])
w_res_init2 = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
for k in range(N_res):
    QS = np.where(np.abs(w_res_init2[:, k]) > 0)[0]
    if len(QS) > 0:
        w_res_init2[QS, k] -= np.mean(w_res_init2[QS, k])
w_res_init3 = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
for k in range(N_res):
    QS = np.where(np.abs(w_res_init2[:, k]) > 0)[0]
    if len(QS) > 0:
        w_res_init3[QS, k] -= np.mean(w_res_init2[QS, k])

mask_out1 = (np.random.rand(N_res, N_out) < p_out).astype(float)
w_out_init1 = np.random.randn(N_res, N_out) * mask_out1 / (np.sqrt(N_out)*p_out) * A
mask_out2 = (np.random.rand(N_res, N_out) < p_out).astype(float)
w_out_init2 = np.random.randn(N_res, N_out) * mask_out2 / (np.sqrt(N_out)*p_out) * A
mask_out3 = (np.random.rand(N_res, N_out) < p_out).astype(float)
w_out_init3 = np.random.randn(N_res, N_out) * mask_out3 / (np.sqrt(N_out)*p_out) * A

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
I_syn =  R : 1
'''
on_pre_out = '''
H_post += (w_out / (tau_r * tau_d)) / Hz
'''

STDP = '''
dApre/dt  = -Apre/tau_s  : 1 (event-driven)
dApost/dt = -Apost/tau_t : 1 (event-driven)
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
G_res1 = NeuronGroup(
    N_res, eqs_res,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_res1.tau_m = tau_m_res*ms
G_res1.t_ref = t_ref_res*ms

G_res2 = NeuronGroup(
    N_res, eqs_res,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_res2.tau_m = tau_m_res*ms
G_res2.t_ref = t_ref_res*ms

G_res3 = NeuronGroup(
    N_res, eqs_res,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_res3.tau_m = tau_m_res*ms
G_res3.t_ref = t_ref_res*ms

G_out = NeuronGroup(
    N_out, eqs_out,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_out.tau_m = tau_m_out*ms
G_out.t_ref = t_ref_out*ms

S_res1 = Synapses(G_res1, G_res1, model='w_res : 1', on_pre=on_pre_res, method='euler')
S_res1.connect(condition='i != j')
S_res1.w_res = w_res_init1[S_res1.i, S_res1.j]
S_res1.delay = 0*ms

S_res2 = Synapses(G_res2, G_res2, model='w_res : 1', on_pre=on_pre_res, method='euler')
S_res2.connect(condition='i != j')
S_res2.w_res = w_res_init2[S_res2.i, S_res2.j]
S_res2.delay = 0*ms

S_out1 = Synapses(G_res1, G_out, model=STDP, on_pre=on_pre_out + STDP_pre, on_post=STDP_post, method='euler')
S_out1.connect(condition='i != j')
S_out1.w_out = w_out_init1[S_out1.i, S_out1.j]
S_out1.eps_w = 1e-12
S_out1.delay = 0*ms

S_out2 = Synapses(G_res2, G_out, model=STDP, on_pre=on_pre_out + STDP_pre, on_post=STDP_post, method='euler')
S_out2.connect(condition='i != j')
S_out2.w_out = w_out_init2[S_out2.i, S_out2.j]
S_out2.eps_w = 1e-12
S_out2.delay = 0*ms

S_out3 = Synapses(G_res3, G_out, model=STDP, on_pre=on_pre_out + STDP_pre, on_post=STDP_post, method='euler')
S_out3.connect(condition='i != j')
S_out3.w_out = w_out_init3[S_out3.i, S_out3.j]
S_out3.eps_w = 1e-12
S_out3.delay = 0*ms


Msp_out = SpikeMonitor(G_out)
n_bins = 500  
sout_rec = np.zeros((len(dir_name), n_sample, N_out, n_bins))
            
Mr_out = SpikeMonitor(G_out)

@network_operation(dt=dt_ms*ms)
def apply_input():
    idx = int(((defaultclock.t - t0)/(dt_ms*ms)))
    I_input = input_current[:, idx] @ w_in
    G_res1.I_exc = I_input
    G_res1.I_inh = 0
    G_res2.I_exc = I_input
    G_res2.I_inh = 0
    G_res3.I_exc = I_input
    G_res3.I_inh = 0

for i in range(len(dir_name)):
    for j in tqdm(range(n_sample)):
        df = pd.read_table(glob.glob(path + "tactile_data/" +dir_name[i]+ f"/data_{int(test_seq[j])}_*")[0], header=None)
        df_np = df.to_numpy().T
        in_data_0 = df_np[:3, 3000:8000]
        nt = in_data_0.shape[1]
        t_array_s = np.arange(nt) * dt_s

        input_current = np.zeros([N_in, nt])

        for k in range(3):
            in_data = in_data_0[k,:] 

            I_merkel   = calc_merkel(in_data,   t_array_s, dt_s)
            I_meissner = calc_meissner(in_data, t_array_s, dt_s)
				
            if k==0 :  
                input_current[k*2 , :] = 0.01*I_merkel
                input_current[k*2+1, :] = 0.04*I_meissner

        S_out1.w_out = w_res_init1[S_out1.i, S_out1.j]
        S_out2.w_out = w_res_init2[S_out2.i, S_out2.j]
        S_out3.w_out = w_res_init2[S_out3.i, S_out3.j]
        S_out1.Apre =0
        S_out1.Apost =0
        S_out2.Apre =0
        S_out2.Apost =0
        S_out3.Apre =0
        S_out3.Apost =0

        # initialize_state
        G_res1.v = v_reset + (v_thr - v_reset)*np.random.rand(N_res)
        G_res2.v = v_reset + (v_thr - v_reset)*np.random.rand(N_res)
        G_out.v = v_reset + (v_thr - v_reset)*np.random.rand(N_out)
        G_res1.R = 0; G_res1.H = 0
        G_res2.R = 0; G_res2.H = 0
        G_out.R = 0; G_out.H = 0 

        start_t = t0

        run((nt*dt_ms)*ms)

        t0 += (nt*dt_ms)*ms
        end_t = t0

        mask = (Mr_out.t > start_t) & (Mr_out.t <= end_t)
        if np.any(mask):
            rel_times_ms = (Mr_out.t[mask] - start_t)/ms
            ids = Mr_out.i[mask]
            bin_edges = np.linspace(0, nt*dt_ms, n_bins+1)
            for n in range(N_out):
                counts, _ = np.histogram(rel_times_ms[ids == n], bins=bin_edges)
                sout_rec[i, j, n, :] = counts
        else:
            sout_rec[i, j, :, :] = 0

np.save("sout_rec_3.npy", sout_rec)
print("Saved sout_rec.npy", sout_rec.shape)
