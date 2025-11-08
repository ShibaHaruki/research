import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.spatial import distance
import glob
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import time  # ★ 追加：時間計測用


path = "/Users/elast/OneDrive - 学校法人立命館/ドキュメント/研究コード/"
dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]

class LIF:
	def __init__(self, N, dt, neuron_array, t_ref_py=2*1e-3, t_ref_in=1*1e-3, tau_m_py=20*1e-3, tau_m_in =10*1e-3, 
					v_reset=-65, v_peak=-40, v_thr=18):

		self.N = N  
		self.dt = dt  
	
		self.neuron_array=neuron_array
		self.ex_index = np.where(neuron_array == 1)[0]
		self.in_index = np.where(neuron_array == -1)[0]

		self.t_ref = np.zeros(N)
		self.t_ref[self.ex_index] = t_ref_py
		self.t_ref[self.in_index] = t_ref_in

		self.tau_m =  np.zeros(N)
		self.tau_m[self.ex_index] = tau_m_py
		self.tau_m[self.in_index] = tau_m_in

		self.v_reset = v_reset  
		self.v_peak = v_peak  
		self.v_thr = v_thr  

		self.v_k = np.ones(N) * self.v_reset + np.random.rand(N)*(v_peak - v_reset)
		self.v_ = None  
		self.t_last = np.zeros(N)
		self.t_count = 0

	def initialize_states(self):
		self.t_count = 0	
		self.t_last = np.zeros(self.N)
		self.v_k = np.ones(self.N) * self.v_reset + np.random.rand(self.N)*(self.v_peak - self.v_reset)

	def __call__(self, I_ak, I_gk):
		dv_k = (-self.v_k + I_ak - I_gk) / self.tau_m
		v_k = self.v_k + (((self.dt*self.t_count) > (self.t_last + self.t_ref)) * dv_k * self.dt)
		s = (v_k >= self.v_thr)
		self.t_last = (1-s)*self.t_last + s*self.dt*self.t_count	
		v_k = (1-s)*v_k + s*self.v_peak
		self.v_ = v_k
		self.v_k = (1-s)*v_k + s*self.v_reset
		self.t_count += 1
		return s


class Synapse_res:
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
		index = np.where(sres==1)[0]
		len_index = len(index)
		if len_index > 0:
			self.JD = np.sum(self.w0[index, :], axis=0)
		self.R = self.R * np.exp(-self.dt/self.tau_d) + self.H*self.dt
		self.H = self.H * np.exp(-self.dt/self.tau_r) + self.JD*(len_index>0) / (self.tau_r*self.tau_d)
		self.t_count += 1
		return (self.R)


class Synapse_out:
	def __init__(self, N_out, N_res, dt, w_out, w_out_const0,  tau_d=20*1e-3, tau_r=20*1e-3):
		self.N_out = N_out
		self.N_res = N_res
		self.dt = dt
		self.t_count = 0
		self.w_out = w_out
		self.w_out_const0 = w_out_const0
		self.w_history = [] 
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
		index = np.where(sres==1)[0]
		len_index = len(index)
		if len_index > 0:
			self.JD = np.sum(self.w_out[index, :], axis=0)
		self.R = self.R * np.exp(-self.dt/self.tau_d) + self.H*self.dt
		self.H = self.H * np.exp(-self.dt/self.tau_r) + self.JD*(len_index>0) / (self.tau_r*self.tau_d)

		for i in np.arange(self.N_res):
			self.p[i] = sres[i]*(self.A_B_LTP + self.p[i]) + (1-sres[i])*self.p[i]*np.exp(-self.dt/self.tau_r)
			if(sres[i]):
				idx = np.where(self.w_out[i,:]>0)
				self.w_out[i,idx] = self.w_out[i,idx] + self.n[idx]
		
		for i in np.arange(self.N_out):
			self.n[i] = sout[i]*(self.A_B_LTD + self.n[i]) + (1-sout[i])*self.n[i]*np.exp(-self.dt/self.tau_d)
			if(sout[i]):
				idx = np.where(self.w_out[:,i]>0)
				self.w_out[idx,i] = self.w_out[idx,i] + self.p[idx]
		
		self.w_out = np.clip(self.w_out, -1e9, 1e9)
		self.w_out[self.w_out_const0[0], self.w_out_const0[1]] = 0
		self.t_count += 1
		if self.t_count % 10 == 0:  
			self.w_history.append(np.mean(self.w_out))
		return (self.R)



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

path = "/Users/elast/OneDrive - 学校法人立命館/ドキュメント/研究コード/"
dir_name = ["Al_board","buta_omote","buta_ura","cork","denim","rubber_board","washi","wood_board"]
N_in = 2
N_res = 2000
N_out = 40

dt = 0.1 * 1e-3
p_in=0.4
p_res=0.1
variance = (N_res*p_res**2)**-1
G=0.01
p_out = 0.2

w_in = np.random.randn(N_in, N_res) * (np.random.rand(N_in, N_res) < p_in) / (np.sqrt(N_in)*p_in) #行列 2*200

w_res = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res)<p_res)*np.sqrt(variance)
for i in range(N_res):
	QS = np.where(np.abs(w_res[:, i]) > 0)[0]
	w_res[QS, i] = w_res[QS, i] - np.sum(w_res[QS, i]) / len(QS)

w_out = (np.random.rand(N_res, N_out) < p_out) * 1
w_out_const0 = np.where(w_out == 0)	
w_out = np.random.randn(N_res, N_out) * (w_out) / (np.sqrt(N_out)*p_out) * G

BIAS = -65

neuron_res = LIF(N_res, dt, np.ones(N_res), t_ref_py=2*1e-3, tau_m_py=10*1e-3, v_reset=-65, v_peak=30, v_thr=-40)
synapse_res = Synapse_res(N_res, dt, w_res, tau_d=20*1e-3, tau_r=2*1e-3)
neuron_out = LIF(N_out, dt, np.ones(N_out), t_ref_py=2*1e-3, tau_m_py=10*1e-3, v_reset=-65, v_peak=30, v_thr=-40)
synapse_out = Synapse_out(N_out, N_res, dt, w_out, w_out_const0, tau_d=20*1e-3, tau_r=2*1e-3)


sample_seq = np.load("sample_seq.npy")


# 記録用リスト
spike_times_res, spike_ids_res = [], []
spike_times_out, spike_ids_out = [], []
V_res_rec, V_out_rec = [], []
I_syn_res_rec, I_syn_out_rec = [], []

#時間計測開始
start_time = time.perf_counter()

for epo in range(1):
	for i_size in range(1):
		for i in dir_name:
			df = pd.read_table(glob.glob(path + "tactile_data/" + i + f"/data_{int(sample_seq[i_size])}_*")[0],header=None)
			df_np = df.to_numpy().T
			in_data_0 = df_np[:3, 3000:8000]
			dt_data = 0.1 * 1e-3
			nt = in_data_0.shape[1]
			t_array = np.arange(nt)*dt_data


			input_current = np.zeros([N_in, nt])
			for j in range(3):
				in_data = in_data_0[j,:]
				I_merkel = calc_merkel(in_data, t_array, dt)
				I_meissner = calc_meissner(in_data, t_array, dt)
				if j==0:
					input_current[j*2 , :] = 0.01*I_merkel
					input_current[j*2+1, :] = 0.04*I_meissner

			neuron_res.initialize_states()
			synapse_res.initialize_states()
			neuron_out.initialize_states()
			synapse_out.initialize_states()

			I_syn = np.zeros(N_res)
			I_syn_out = np.zeros(N_out)

			sout = np.zeros(N_out)
			sres = np.zeros(N_res)

			for t in range(nt):
				I_input = input_current[:, t] @ w_in
				I = I_syn + BIAS + I_input
				sres = neuron_res(I, 0)
				I_syn =  synapse_res(sres)
				I_syn = G  * I_syn
				sout = neuron_out(I_syn_out + BIAS, 0)
				I_syn_out = synapse_out(sres, sout)

				V_res_rec.append(neuron_res.v_.copy())
				V_out_rec.append(neuron_out.v_.copy())
				I_syn_res_rec.append(I_syn.copy())
				I_syn_out_rec.append(I_syn_out.copy())

				st_res = np.where(sres==1)[0]
				st_out = np.where(sout==1)[0]
				for sid in st_res:
					spike_times_res.append(t*dt*1e3)
					spike_ids_res.append(sid)
				for sid in st_out:
					spike_times_out.append(t*dt*1e3)
					spike_ids_out.append(sid)

		print(str(i_size)+".")

#時間計測終了 & 表示
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"[TIME] Custom loop: {elapsed:.3f} s")

# ---- 保存 ----
np.save("w_in.npy", w_in)
np.save("w_res.npy", w_res)
np.save("w_out.npy", synapse_out.w_out)

# ---- グラフ描画 ----
plt.figure(figsize=(10,5))
plt.plot(spike_times_res, spike_ids_res, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Reservoir neuron index')
plt.title('Reservoir Raster Plot')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(spike_times_out, spike_ids_out, '.', markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Output neuron index')
plt.title('Output layer raster plot')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i in range(10):
	plt.plot(np.arange(nt)*dt*1e3, np.array(I_syn_res_rec)[:,i], label=f'Res neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('I_syn (Reservoir)')
plt.title('Reservoir Synaptic Currents (First 10)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i in range(10):
	plt.plot(np.arange(nt)*dt*1e3, np.array(I_syn_out_rec)[:,i], label=f'Output neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('I_syn (Output)')
plt.title('Output Layer Synaptic Currents (First 10)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i in range(10):
	plt.plot(np.arange(nt)*dt*1e3, np.array(V_res_rec)[:,i], label=f'Res neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (Reservoir)')
plt.title('Reservoir Neurons Membrane Potentials (First 10)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i in range(10):
	plt.plot(np.arange(nt)*dt*1e3, np.array(V_out_rec)[:,i], label=f'Output neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v (Output)')
plt.title('Output Neurons Membrane Potentials (First 10)')
plt.legend()
plt.tight_layout()
plt.show()






