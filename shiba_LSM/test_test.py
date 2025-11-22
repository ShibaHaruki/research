import numpy as np
rng = np.random.default_rng(2)
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.spatial import distance
import pandas as pd
import glob
from scipy.signal import savgol_filter
from brian2 import *
prefs.codegen.target = 'numpy'
from pathlib import Path
from tqdm import tqdm 

start_scope()

path = str(Path(__file__).resolve().parents[1]) + "/"
dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]

sample_seq = np.load("sample_seq.npy")
max = len(sample_seq)
test_seq = sample_seq[100:max]

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

n_sample = 100

w_in = np.load("w_in.npy")
w_res_init = np.load("w_res.npy")
w_out_init = np.load("w_out.npy")

N_in = w_in.shape[0]
N_res = w_res_init.shape[1]
N_out = w_out_init.shape[1]

##################
#hiper-parameters
# LIF
v_reset = -65 
v_peak  = 30
v_thr   = -40
#reservoir
neuron_array_res = np.ones(N_res)
res_exc_idx = np.where(neuron_array_res == 1)[0]
res_inh_idx = np.where(neuron_array_res == -1)[0]
t_ref_res = np.zeros(N_res)
tau_m_res  = np.zeros(N_res)*second
tau_m_res  = np.where(neuron_array_res == 1, 10, np.where(neuron_array_res == -1, 10,10))
t_ref_res  = np.where(neuron_array_res == 1, 2,np.where(neuron_array_res == -1, 2, 2))
#output
neuron_array_out = np.ones(N_out)
out_exc_idx = np.where(neuron_array_out == 1)[0]
out_inh_idx = np.where(neuron_array_out == -1)[0]
t_ref_out = np.zeros(N_out)
tau_m_out = np.zeros(N_out)
tau_m_out = np.where(neuron_array_out == 1, 10, np.where(neuron_array_out== -1, 10, 10))
t_ref_out = np.where(neuron_array_out == 1, 2,np.where(neuron_array_out == -1, 2, 2))


#double_exponential_synapse
tau_r = 2*ms  
tau_d = 20*ms

# STDP
A_B_LTP = 0.06
A_B_LTD = -0.05
wmin = -1e9
wmax = 1e9

BIAS = -65
G=0.02 
###################

#time
dt_ms = 0.1                   
dt_s  = dt_ms * 1e-3          
defaultclock.dt = dt_ms * ms 
t0 = 0*ms

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

S_res = Synapses(G_res, G_res, model = ''' w_res : 1 ''' , on_pre = on_pre_res , method ='euler')
S_res.connect(condition='i != j')
S_res.w_res = w_res_init[S_res.i, S_res.j]
S_res.delay = 0*ms

S_out = Synapses(G_out, G_out, model = ''' w_out : 1 ''' , on_pre = on_pre_out , method ='euler')
S_out.connect(condition='i != j')
S_out.w_out = w_out_init[S_out.i, S_out.j]
S_out.delay = 0*ms


Msp_out = SpikeMonitor(G_out)
n_bins = 500  
sout_rec = np.zeros((len(dir_name), n_sample, N_out, n_bins))


@network_operation(dt=dt_ms*ms)
def apply_input():
    global prev_spiked_res, prev_spiked_out
    idx = int(((defaultclock.t - t0) / (dt_ms*ms))) 
    if idx >= input_current.shape[1]:
        return
    I_input = input_current[:, idx] @ w_in
    G_res.I_exc = I_input 
    G_res.I_inh = 0

Msp_res = SpikeMonitor(G_res)               
Mr_out = SpikeMonitor(G_out)

M_res = StateMonitor(G_res, 'I_syn', record=range(min(10, N_res)))
M_Iout = StateMonitor(G_out, 'I_syn', record=range(min(10, N_out)))

M_Vres = StateMonitor(G_res, 'v', record=range(min(10, N_res)))
M_Vout = StateMonitor(G_out, 'v', record=range(min(10, N_out)))

G_res.v = v_reset + (v_thr - v_reset)* np.random.rand(N_res)
G_out.v = v_reset + (v_thr - v_reset)* np.random.rand(N_out)
G_res.R = 0
G_res.H = 0
G_out.R = 0
G_out.H = 0

for i in range(1):
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

        G_res.v = v_reset + (v_thr - v_reset)* np.random.rand(N_res)
        G_out.v = v_reset + (v_thr - v_reset)* np.random.rand(N_out)
        G_res.R = 0; G_res.H = 0
        G_out.R = 0; G_out.H = 0

        run((nt*dt_ms)*ms)
        t0 += (nt*dt_ms)*ms

        # --- collect output spikes into sout_rec bins (same format as test_vKK.py) ---
        start_t = t0                    # ループ開始時点のt0を使うなら、この行は run() の直前に start_t = t0 として保存しておく
        # もし run() の直前で start_t = t0 を保存していた場合は上の行を削除し、以下の1行を使ってください:
        # end_t = t0 + (nt*dt_ms)*ms

        end_t = t0 + (nt*dt_ms)*ms
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
        # --- end collection ---

np.save("sout_rec.npy", sout_rec)
print("Saved sout_rec.npy", sout_rec.shape)


with open("params2.txt", "w") as file:
    file.write(f"N_in = {N_in}\n")
    file.write(f"N_res = {N_res}\n")
    file.write(f"N_out = {N_out}\n")
    file.write(f"G = {G}\n")
    file.write(f"dt = {dt_ms}\n")
    file.write(f"t_ref = {2e-3}\n")
    file.write(f"v_reset = {v_reset}\n")
    file.write(f"v_peak = {v_peak}\n")
    file.write(f"v_thr = {v_thr}\n")
    file.write(f"BIAS = {BIAS}\n")


# Reservoir Raster
plt.figure(figsize=(10, 5))
plt.plot(Msp_res.t/ms, Msp_res.i, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir neuron index')
plt.title(f'Reservoir Raster Plot ({nt * dt_ms:.1f} ms)')
plt.tight_layout()

# Output Raster
plt.figure(figsize=(10, 5))
plt.plot(Mr_out.t/ms, Mr_out.i, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Output neuron index')
plt.title('Output layer raster plot')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(min(10, N_res)):
    plt.plot(M_res.t/ms, M_res.I_syn[i], label=f'Res neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('I_syn (Reservoir neurons)')
plt.title('Reservoir Synaptic Currents (First 10 neurons)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(min(10, N_out)):
    plt.plot(M_Iout.t/ms, M_Iout.I_syn[i], label=f'Output neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('I_syn (Output neurons)')
plt.title('Output Layer Synaptic Currents (First 10 neurons)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(min(10, N_res)):
    plt.plot(M_Vres.t/ms, M_Vres.v[i], label=f'Res neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (Reservoir)')
plt.title('Reservoir Neurons Membrane Potentials (First 10)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(min(10, N_out)):
    plt.plot(M_Vout.t/ms, M_Vout.v[i], label=f'Output neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (Output)')
plt.title('Output Neurons Membrane Potentials (First 10)')
plt.legend()
plt.tight_layout()
plt.show()


	


        