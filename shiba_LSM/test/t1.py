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
base_name = script_path.stem
pdf_filename = f"{base_name}.pdf"
pdf_path = script_path.parent / pdf_filename
pdf = PdfPages(pdf_path) 

sample_seq = np.load("sample_seq.npy")
max = len(sample_seq)
test_seq = sample_seq[100:max]
n_sample = 100

#input filter
#meissner
def calc_meissner(data, t, dt):

	I = np.zeros((4, len(t)))
	
	for i in range(len(t)):
		if(i!=0):

			dF_dt = np.abs(data[i] - data[i-1])/(t[i]-t[i-1])

			I[0, i] = I[0, i-1] + 1*dF_dt + (-I[0, i-1]*dt/(8*1* 1e-3)) 
			I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200* 1e-3) )
			I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6* 1e-3))
			I[3, i] = I[0, i] 

	return I[3,:]

#merkel
def calc_merkel(data, t, dt):

	I = np.zeros((4, len(t)))
	
	for i in range(len(t)):
		if(i!=0):

			dF_dt = np.abs(data[i] - data[i-1])/(t[i]-t[i-1])

			if(dF_dt < 0):
				dF_dt = 0

			I[0, i] = I[0, i-1] + 0.74*dF_dt + (-I[0, i-1]*dt/(8*1* 1e-3)) 
			I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200*1* 1e-3) )
			I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6*1* 1e-3))
			I[3, i] = I[0, i] +  I[1, i] +  I[2, i] 

	return I[3,:]

##################
#hiper-parameters
# hyper-parameters
N_in = 2
N_res = 500
N_out = 40

# circuit connection probability
p_in = 0.4
p_res = 0.5
p_out = 0.6

# LIF
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


#double_exponential_synapse
tau_r = 2*ms  
tau_d = 20*ms

# STDP
A_B_LTP = 0.005
A_B_LTD = -0.006
wmin = -1
wmax = 1
tau_s = 11.5*ms
tau_t = 14*ms

BIAS = -65
G = 0.11
A = 0.01
###################

#time
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


G_res = NeuronGroup(
    N_res, eqs_res,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory= 'timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_res.tau_m = tau_m_res*ms
G_res.t_ref = t_ref_res*ms
G_out = NeuronGroup(
    N_out, eqs_out,
    threshold='v >= v_thr',
    reset='v = v_reset',
    refractory= 'timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
    method='exact'
)
G_out.tau_m = tau_m_out*ms
G_out.t_ref = t_ref_out*ms

S_res = Synapses(G_res, G_res, model = 'w_res : 1', on_pre = on_pre_res, method = 'euler')
S_res.connect(condition='i != j')
S_res.w_res = w_res_init[S_res.i, S_res.j]
S_res.delay = 0*ms

S_out = Synapses(G_res, G_out, model = STDP, on_pre = on_pre_out + STDP_pre, on_post = STDP_post, method = 'euler')
S_out.connect(condition = 'i != j')
S_out.w_out = w_out_init[S_out.i, S_out.j]
S_out.eps_w = 1e-12
S_out.delay = 0*ms


Msp_out = SpikeMonitor(G_out)
n_bins = 500  
sout_rec = np.zeros((len(dir_name), n_sample, N_out, n_bins))
            
Mr_out = SpikeMonitor(G_out)

@network_operation(dt=dt_ms*ms)
def apply_input():
    idx = int(((defaultclock.t - t0)/(dt_ms*ms)))
    I_input = input_current[:, idx] @ w_in
    G_res.I_exc = I_input
    G_res.I_inh = 0


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

        S_res.w_out = w_out_init[S_out.i, S_out.j]
        G_res.v = v_reset + (v_thr - v_reset)* np.random.rand(N_res)
        G_out.v = v_reset + (v_thr - v_reset)* np.random.rand(N_out)
        G_res.R = 0; G_res.H = 0
        G_out.R = 0; G_out.H = 0

        start_t = t0

        # シミュレーション実行
        run((nt*dt_ms)*ms)

        # サンプル終了時刻
        t0 += (nt*dt_ms)*ms
        end_t = t0

        # --- collect output spikes into sout_rec bins ---
        mask = (Mr_out.t > start_t) & (Mr_out.t <= end_t)
        if np.any(mask):
            rel_times_ms = (Mr_out.t[mask] - start_t)/ms   # サンプル内の相対時刻[ms]
            ids = Mr_out.i[mask]
            bin_edges = np.linspace(0, nt*dt_ms, n_bins+1) # 500ビン=1ms幅（dt_ms=0.1ms, 10step分）
            for n in range(N_out):
                counts, _ = np.histogram(rel_times_ms[ids == n], bins=bin_edges)
                sout_rec[i, j, n, :] = counts
        else:
            sout_rec[i, j, :, :] = 0

np.save("sout_rec_1.npy", sout_rec)
print("Saved sout_rec_1.npy", sout_rec.shape)
