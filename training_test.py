import numpy as np
rng = np.random.default_rng(2)
import matplotlib.pyplot as plt
import pandas as pd
import glob
from brian2 import *
prefs.core.default_float_dtype = float64
prefs.codegen.target = 'numpy'
import time  # ★ 追加：時間計測用

start_scope()

path = "/Users/elast/OneDrive - 学校法人立命館/ドキュメント/研究コード/"
dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]

sample_seq = np.load("sample_seq.npy") 

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

#nyuron_number
N_in = 2
N_res = 2000
N_out = 40

#circuit connection probability
p_in=0.4
p_res=0.1
p_out = 0.2

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

#STDP
A_B_LTP = 0.06
A_B_LTD = -0.05
wmin = -1e9
wmax = 1e9

BIAS = -65
G=0.01 
###################


#time
dt_ms = 0.1                   
dt_s  = dt_ms * 1e-3          
defaultclock.dt = dt_ms * ms 
t0 = 0*ms


#weight
w_in = np.random.randn(N_in, N_res) * (np.random.rand(N_in, N_res) < p_in) / (np.sqrt(N_in)*p_in) 

variance = (N_res * p_res**2)**-1
w_res_init = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
for k in range(N_res):
    QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
    if len(QS) > 0:
        w_res_init[QS, k] -= np.mean(w_res_init[QS, k])
   
mask_out = (np.random.rand(N_res, N_out) < p_out).astype(float)
W_out_init = np.random.randn(N_res, N_out) * mask_out / (np.sqrt(N_out)*p_out) * G


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
I_syn =  R  : 1 
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

G_res.tau_m = tau_m_res*ms
G_res.t_ref = t_ref_res*ms
G_out.tau_m = tau_m_out*ms
G_out.t_ref = t_ref_out*ms

S_res = Synapses(G_res, G_res, model = ''' w_res : 1 ''' , on_pre = on_pre_res , method ='euler')
S_res.connect(condition='i != j')
S_res.w_res = w_res_init[S_res.i, S_res.j]
S_res.delay = 0*ms

oi, oj = np.where(W_out_init != 0)
S_out = Synapses(G_res, G_out, model=STDP, on_pre=on_pre_out + STDP_pre, on_post=STDP_post, method = 'euler')
if len(oi) > 0:
    S_out.connect(i=oi, j=oj)
    S_out.w_out[:] = W_out_init[oi, oj]
else:
    S_out.connect(False)
S_out.eps_w =  1e-12
S_out.delay = 0*ms

prev_spiked_res = np.zeros(N_res)
prev_spiked_out = np.zeros(N_out)

#nitialize_state
G_res.v = v_reset + (v_thr - v_reset)* np.random.rand(N_res)
G_out.v = v_reset + (v_thr - v_reset)* np.random.rand(N_out)
G_res.R = 0
G_res.H = 0
G_out.R = 0
G_out.H = 0

#recorders
Msp_res = SpikeMonitor(G_res)               
Msp_out = SpikeMonitor(G_out)

M_Ires = StateMonitor(G_res, 'I_syn', record=range(min(10, N_res)))
M_Iout = StateMonitor(G_out, 'I_syn', record=range(min(10, N_out)))

M_Vres = StateMonitor(G_res, 'v', record=range(min(10, N_res)))
M_Vout = StateMonitor(G_out, 'v', record=range(min(10, N_out)))

M_wout = StateMonitor(S_out, 'w_out', record=True)

@network_operation(dt=dt_ms*ms)
def apply_input():
    global prev_spiked_res, prev_spiked_out
    idx = int(((defaultclock.t - t0) / (dt_ms*ms))) 
    I_input = input_current[:, idx] @ w_in
    G_res.I_exc = I_input 
    G_res.I_inh = 0

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


# 時間計測開始
start_time = time.perf_counter()

for epo in range(1):
    for i_size in range(3):
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

            run((nt*dt_ms)*ms)
            t0 += (nt*dt_ms)*ms

        print(str(i_size)+".")

# 時間計測終了＆表示
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"[TIME] Brian2 run(): {elapsed:.3f} s")


np.save("w_in.npy", w_in)
np.save("w_res.npy", w_res_init)
w_out_dense = np.zeros((N_res, N_out))
w_out_dense[np.array(S_out.i[:]), np.array(S_out.j[:])] = np.array(S_out.w_out[:])
np.save("w_out.npy", w_out_dense)

###########
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
file.close()
###########	


# Reservoir Raster
plt.figure(figsize=(10, 5))
plt.plot(Msp_res.t/ms, Msp_res.i, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir neuron index')
plt.title(f'Reservoir Raster Plot ({nt * dt_ms:.1f} ms)')
plt.tight_layout()

# Output Raster
plt.figure(figsize=(10, 5))
plt.plot(Msp_out.t/ms, Msp_out.i, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Output neuron index')
plt.title('Output layer raster plot')
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.hist(Msp_res.t/ms, 20000, histtype='stepfilled', facecolor='k', weights=list(ones(len(Msp_res))/(N_res*defaultclock.dt)))
plt.xlabel('Time (ms)')
ylabel('Instantaneous firing rate (sp/s)')
plt.tight_layout() 

plt.figure(figsize=(10, 5))
plt.hist(Msp_out.t/ms, 20000, histtype='stepfilled', facecolor='k', weights=list(ones(len(Msp_out))/(N_out*defaultclock.dt)))
xlabel('Time (ms)')
plt.ylabel('Instantaneous firing rate (sp/s)')
plt.tight_layout()

plt.figure(figsize=(10, 5))
for i in range(min(10, N_res)):
    plt.plot(M_Ires.t/ms, M_Ires.I_syn[i], label=f'Res neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir Isyn')
plt.tight_layout()

plt.figure(figsize=(10, 5))
for i in range(min(10, N_res)):
    plt.plot(M_Iout.t/ms, M_Iout.I_syn[i], label=f'Output neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Output Isyn')
plt.tight_layout()

plt.figure(figsize=(10, 5))
for i in range(min(10, N_res)):
    plt.plot(M_Vres.t/ms, M_Vres.v[i], label=f'Res neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir v')
plt.tight_layout()

plt.figure(figsize=(10, 5))
for i in range(min(10, N_res)):
    plt.plot(M_Vout.t/ms, M_Vout.v[i], label=f'Output neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Output v')
plt.tight_layout()

wout_mean_syn = M_wout.w_out.mean(axis=0)
plt.figure(figsize=(10, 5))  
plt.plot(M_wout.t / ms, wout_mean_syn)
plt.xlabel("time (ms)")
plt.ylabel("mean w_out (all synapses)")
plt.tight_layout()
plt.show()




