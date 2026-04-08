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
from pathlib import Path
from tqdm import tqdm 

script_path = Path(__file__).resolve()
base_name = script_path.stem
pdf_filename = f"{base_name}.pdf"
pdf_path = script_path.parent / pdf_filename
pdf = PdfPages(pdf_path) 

start_scope()

path = str(Path(__file__).resolve().parents[1]) + "/"
dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]

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
N_res = 1000
N_out = 40

# circuit connection probability
p_in = 0.2
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

#T_STDP
wmin = -1.0
wmax = 1.0

tau_s1_val  = 11.5*ms
tau_s2_val   = 15*ms
tau_t1_val = 14*ms 
tau_t2_val    = 15*ms  

A2_plus_val   = 0.0007
A3_plus_val   = 0.00000005
A2_minus_val  = 0.0006
A3_minus_val  = 0.00000005


BIAS = -65
G = 0.25
###################

# time
dt_ms = 0.1
dt_s  = dt_ms * 1e-3
defaultclock.dt = dt_ms * ms
t0 = 0*ms

# weights
w_in = np.random.randn(N_in, N_res) * (np.random.rand(N_in, N_res) < p_in) / (np.sqrt(N_in*p_in))

variance = (N_res * p_res**2)**-1
w_res_init = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
for k in range(N_res):
    QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
    if len(QS) > 0:
        w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

mask_out = (np.random.rand(N_res, N_out) < p_out).astype(float)
w_out_init = np.random.randn(N_res, N_out) * mask_out / (np.sqrt(N_out*p_out))*G

# -------- models --------
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
I_syn =  R  : 1
'''
on_pre_out = '''
H_post += (w_out / (tau_r * tau_d)) / Hz
'''

T_STDP = '''
dApre1/dt   = -Apre1/tau_s1   : 1 (event-driven)  # r1: pre fast
dApre2/dt   = -Apre2/tau_s2   : 1 (event-driven)  # r2: pre slow
dApost1/dt  = -Apost1/tau_t1  : 1 (event-driven)  # o1: post fast
dApost2/dt  = -Apost2/tau_t2  : 1 (event-driven)  # o2: post slow

w_out : 1
eps_w : 1

tau_s1 : second (constant)
tau_s2 : second (constant) 
tau_t1 : second (constant)
tau_t2 : second (constant)

A2_plus  : 1 (constant)
A3_plus  : 1 (constant)
A2_minus : 1 (constant)
A3_minus : 1 (constant)
'''

T_STDP_pre = '''
Apre1 +=1
Apre2 +=1

w_out = clip(w_out - int(w_out > eps_w) * Apost1 * (A2_minus + A3_minus * Apre2), wmin, wmax)
'''

T_STDP_post = '''
Apost1 +=1
Apost2 +=1

w_out = clip(w_out + int(w_out > eps_w) * Apre1 * (A2_plus + A3_plus * Apost2), wmin, wmax)
'''

###
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
###

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

S_res = Synapses(G_res, G_res, model = 'w_res : 1', on_pre = on_pre_res, method = 'euler')
S_res.connect(condition='i != j')
S_res.w_res = w_res_init[S_res.i, S_res.j]
S_res.delay = 0*ms

pre_idx, post_idx = np.where(mask_out > 0)
S_out = Synapses(G_res, G_out, model = T_STDP, on_pre = on_pre_out + T_STDP_pre, on_post = T_STDP_post, method = 'euler')
S_out.connect(i=pre_idx, j=post_idx)
S_out.w_out = w_out_init[S_out.i, S_out.j]
S_out.eps_w = 1e-12
S_out.delay = 0*ms

S_out.tau_s1   = tau_s1_val
S_out.tau_s2   = tau_s2_val
S_out.tau_t1   = tau_t1_val
S_out.tau_t2   = tau_t2_val

S_out.A2_plus  = A2_plus_val
S_out.A3_plus  = A3_plus_val
S_out.A2_minus = A2_minus_val
S_out.A3_minus = A3_minus_val

@network_operation(dt=dt_ms*ms)
def apply_input():
    idx = int(((defaultclock.t - t0)/(dt_ms*ms)))
    I_input = input_current[:, idx] @ w_in
    G_res.I_exc = I_input
    G_res.I_inh = 0

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
                    input_current[j * 2, :]     = 0.4 * I_merkel * 0.02
                    input_current[j * 2 + 1, :] = 0.6 * 7.3 * I_meissner * 0.02


            # initialize_state
            G_res.v = v_reset + (v_thr - v_reset)*np.random.rand(N_res)
            G_out.v = v_reset + (v_thr - v_reset)*np.random.rand(N_out)
            G_res.R = 0; G_res.H = 0
            G_out.R = 0; G_out.H = 0 

            run((nt*dt_ms)*ms)
            t0 += (nt*dt_ms)*ms

        print(str(i_size) + ".")

np.save("T_STDP_1_w_in.npy", w_in)
np.save("T_STDP_1_w_res.npy", w_res_init)
w_out_dense = np.zeros((N_res, N_out))
w_out_dense[np.array(S_out.i[:]), np.array(S_out.j[:])] = np.array(S_out.w_out[:])
np.save("T_STDP_1_w_out.npy", w_out_dense)

        
