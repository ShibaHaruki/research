import numpy as np
#np.random.seed(1)
rng = np.random.default_rng(2)
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.spatial import distance
#import cupy as cp
import glob
import pandas as pd
from scipy.signal import savgol_filter



class LIF:
	# N:neuron数，dt:時間幅，neuron_array:1なら興奮-1なら抑制，t_ref_py(in):興奮(抑制)ニューロンの不応期[s]，tau_m_py(in):興奮(抑制)ニューロンの時定数[s]
	# v_reset:リセット電位，v_peak:peak電位，v_thr:閾値
	def __init__(self, N, dt, neuron_array, t_ref_py=2*1e-3, t_ref_in=1*1e-3, tau_m_py=20*1e-3, tau_m_in =10*1e-3, 
					v_reset=-65, v_peak=-40, v_thr=18):

		self.N = N  #number of neuron
		self.dt = dt  #ms
	
		self.neuron_array=neuron_array
		self.ex_index = np.where(neuron_array == 1)[0]
		self.in_index = np.where(neuron_array == -1)[0]

		self.t_ref = np.zeros(N)
		self.t_ref[self.ex_index] = t_ref_py
		self.t_ref[self.in_index] = t_ref_in
		#print(t_ref_py, t_ref_in)

		self.tau_m =  np.zeros(N)
		self.tau_m[self.ex_index] = tau_m_py
		self.tau_m[self.in_index] = tau_m_in
		#print(tau_m_py, tau_m_in)

		self.v_reset = v_reset  #v reset
		self.v_peak = v_peak  #v peak
		self.v_thr = v_thr  #v threshold
		#print(v_reset, v_peak, v_thr)

		self.v_k = np.ones(N) * self.v_reset + np.random.rand(N)*(v_peak - v_reset)    #v_reset ~ v_peakの間でランダムに初期化
		self.v_ = None  #記録 
		self.t_last = np.zeros(N)
		self.t_count = 0


	def initialize_states(self):

		self.t_count = 0	
		self.t_last = np.zeros(self.N)
		self.v_k = np.ones(self.N) * self.v_reset + np.random.rand(self.N)*(self.v_peak - self.v_reset)


	#I_ak:興奮からの入力，抑制からの入力，I_gk:抑制からの入力
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
	#N:neuron数，dt:時間幅，w0:シナプス重み，tau_d:減少に関する時定数[s]，tau_r:上昇に関する時定数[s]
	def __init__(self, N, dt, w0, tau_d=20*1e-3, tau_r=2*1e-3):

		self.N = N

		self.dt = dt
		self.t_count = 0

		self.w0 = w0

		self.tau_d = tau_d
		self.tau_r = tau_r

		self.R = np.zeros(self.N)
		self.H = np.zeros(self.N)	#補助変数
		self.JD = np.zeros(self.N)
		self.w0 = w0


	def initialize_states(self):
		self.t_count = 0
		self.R = np.zeros(self.N)
		self.H = np.zeros(self.N)
		self.JD = np.zeros(self.N)


	#s_record:spike列
	def __call__(self, sres):
		#spikeがあった所，数を取得
		#index = np.where(sres_rec[:, self.t_count]==1)[0]
		index = np.where(sres==1)[0]
		len_index = len(index)

		#spikeがあれば結合先への重みの和を計算
		if len_index > 0:
			self.JD = np.sum(self.w0[index, :], axis=0)

		#Double exponential model
		self.R = self.R * np.exp(-self.dt/self.tau_d) + self.H*self.dt
		self.H = self.H * np.exp(-self.dt/self.tau_r) + self.JD*(len_index>0) / (self.tau_r*self.tau_d)

		self.t_count += 1

		return (self.R)



class Synapse_out:
	#N_out:Readour_neuron数，dt:時間幅，w_out:Res→outへのシナプス重み，w_out_const0:常に0である所，tau_d:減少に関する時定数[s]，tau_r:上昇に関する時定数[s]
	def __init__(self, N_out, N_res, dt, w_out, w_out_const0,  tau_d=20*1e-3, tau_r=20*1e-3):

		self.N_out = N_out
		self.N_res = N_res

		self.dt = dt
		self.t_count = 0

		self.w_out = w_out
		self.w_out_const0 = w_out_const0

		self.tau_d = tau_d
		self.tau_r = tau_r

		self.R = np.zeros(self.N_out)
		self.H = np.zeros(self.N_out)
		self.JD = np.zeros(self.N_out)

		self.p = np.zeros(self.N_res)
		self.n = np.zeros(self.N_out)

		#STDPのLTP(LTD)の最大値
		self.A_B_LTP = 0.06
		self.A_B_LTD = -0.05
		self.stdp_scale = 1e-6


	def initialize_states(self):
		self.t_count = 0	
		self.R = np.zeros(self.N_out)
		self.H = np.zeros(self.N_out)
		self.JD = np.zeros(self.N_out)


	#sres:Res_neuronのスパイク列，sout:out_neuronのスパイク列
	def __call__(self, sres, sout):

		index = np.where(sres==1)[0]
		len_index = len(index)

		if len_index > 0:			
			self.JD = np.sum(self.w_out[index, :], axis=0)

		self.R = self.R * np.exp(-self.dt/self.tau_d) + self.H*self.dt
		self.H = self.H * np.exp(-self.dt/self.tau_r) + self.JD*(len_index>0) / (self.tau_r*self.tau_d)


		# for i in np.arange(self.N_res):
		# 	self.p[i] = sres[i]*(self.A_B_LTP + self.p[i]) + (1-sres[i])*self.p[i]*np.exp(-self.dt/self.tau_r)
		# 	if(sres[i]):
		# 		idx = np.where(w_out[i,:]>0)
		# 		w_out[i,idx] = w_out[i,idx] + self.n[idx]
		
		# for i in np.arange(self.N_out):
		# 	self.n[i] = sout[i]*(self.A_B_LTD + self.n[i]) + (1-sout[i])*self.n[i]*np.exp(-self.dt/self.tau_d)
		# 	if(sout[i]):
		# 		idx = np.where(w_out[:,i]>0)
		# 		w_out[idx,i] = w_out[idx,i] + self.p[idx]
		
		
		# self.w_out = np.clip(self.w_out, -1e100000, 1e100000)  #最小値と最大値を制限
		# #self.w_out = np.clip(self.w_out, 0, 1e100000)
		# self.w_out[self.w_out_const0[0], self.w_out_const0[1]] = 0  #常に0である所

		self.t_count += 1

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



dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]

n_sample = 100

#N_in = 2
#N_res = 2000
#N_out = 8


#p_in=0.4  #w_inの結合確率
#p_res=0.1  #w_resの結合確率
#variance = (N_res*p_res**2)**-1
#p_out = 0.2  #w_outの結合確率

w_in = np.load("w_in.npy")
#w_in = np.random.randn(N_in, N_res) * (np.random.rand(N_in, N_res) < p_in) / (np.sqrt(N_in)*p_in)

w_res = np.load("w_res.npy")
#w_res = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res)<p_res)*np.sqrt(variance)
#for i in range(N_res):
#	QS = np.where(np.abs(w_res[:, i]) > 0)[0]
#	w_res[QS, i] = w_res[QS, i] - np.sum(w_res[QS, i]) / len(QS)

w_out = np.load("w_out.npy")
#w_out = (np.random.rand(N_res, N_out) < p_out) * 1
w_out_const0 = np.where(w_out == 0)	
#w_out = np.random.randn(N_res, N_out) * (w_out) / (np.sqrt(N_out)*p_out) * G
#w_out = np.abs(w_out)

#dt = 0.1 #msec
dt = 0.1 * 1e-3  #s

N_in = w_in.shape[0]
N_res = w_in.shape[1]
N_out = w_out.shape[1]

G=0.01  #scale

BIAS = -65#-40, -65

###########
with open("params2.txt", "w") as file:
	file.write("N_in = " + str(N_in) + "\n")
	file.write("N_res = " + str(N_res) + "\n")
	file.write("N_out = " + str(N_out) + "\n")
	file.write("G = " + str(G) + "\n")
	file.write("dt = " + str(dt) + "\n")
	file.write("t_ref = " + str(2*1e-3) + "\n")
	file.write("v_reset = " + str(-65) + "\n")
	file.write("v_peak = " + str(30) + "\n")
	file.write("v_thr = " + str(-40) + "\n")
	file.write("BIAS = " + str(BIAS) + "\n")
file.close()
###########	


neuron_res = LIF(N_res, dt, np.ones(N_res), t_ref_py=2* 1e-3, tau_m_py=10* 1e-3, v_reset=-65, v_peak=30, v_thr=-40)
synapse_res = Synapse_res(N_res, dt, w_res, tau_d=20* 1e-3, tau_r=2* 1e-3)

neuron_out = LIF(N_out, dt, np.ones(N_out), t_ref_py=2* 1e-3, tau_m_py=10* 1e-3, v_reset=-65, v_peak=30, v_thr=-40)
synapse_out = Synapse_out(N_out, N_res, dt, w_out, w_out_const0, tau_d=20* 1e-3, tau_r=2* 1e-3)

#loop_size = 10

sout_rec = np.zeros((len(dir_name), n_sample, N_out, 500))
#print(int(5000*dt))

sample_seq = np.load("sample_seq.npy")
max = len(sample_seq)
test_seq = sample_seq[100:max]


for i in range(len(dir_name)):

	for j in tqdm(range(n_sample)):
		df = pd.read_table(glob.glob("tactile_data/" +dir_name[i]+ f"/data_{int(test_seq[j])}_*")[0], header=None)
		df_np = df.to_numpy().T
		in_data_0 =  df_np[:3, 3000:8000]	

		dt_data = 0.1 * 1e-3  #s
		nt = in_data_0.shape[1]
		T = nt*dt_data #s
		t_array = np.arange(nt)*dt_data


		input_current = np.zeros([N_in, nt])

		for k in range(3):
			in_data = in_data_0[k,:]
			#in_data = np.interp(t_array, t_array, in_data_0[j,:])

			I_merkel = calc_merkel(in_data, t_array, dt)
			I_meissner = calc_meissner(in_data, t_array, dt)

			if k==0 :  #sensor1のみ使う
				input_current[k*2 , :] = 0.01*I_merkel
				input_current[k*2+1, :] = 0.04*I_meissner 


		neuron_res.initialize_states()
		synapse_res.initialize_states()

		I_syn = np.zeros(N_res)
		sres = np.zeros(N_res)

		neuron_out.initialize_states()
		synapse_out.initialize_states()

		I_syn_out = np.zeros(N_out)
		sout = np.zeros(N_out)
		sout_tmp = np.zeros(N_out)


		for t in range(nt):
			I_input = input_current[:, t] @ w_in

			I = I_syn + BIAS + I_input

			sres = neuron_res(I, 0)
	
			I_syn = synapse_res(sres)
			I_syn = G  * I_syn

			sout = neuron_out(I_syn_out + BIAS, 0) 
			if(t%10 == 9):
				sout_tmp = sout_tmp + sout
				#print(sout_tmp.shape)
				#print(sout_rec.shape)
				sout_rec[i, j, :, t//10] = sout_tmp
				sout_tmp = np.zeros(N_out)
			else:
				sout_tmp = sout_tmp + sout
					

			I_syn_out = synapse_out(sres, sout)
			#Iout_rec[:, t] = I_syn_out
	
outfname = "sout_rec.npy"
np.save(outfname, sout_rec)


			





