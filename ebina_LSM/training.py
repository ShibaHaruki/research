import numpy as np
np.random.seed(1)
rng = np.random.default_rng(1)
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.spatial import distance
import cupy as cp
import glob
import pandas as pd
from scipy.interpolate import interp1d
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


		self.tau_m =  np.zeros(N)
		self.tau_m[self.ex_index] = tau_m_py
		self.tau_m[self.in_index] = tau_m_in



		self.v_reset = v_reset  #v reset
		self.v_peak = v_peak  #v peak
		self.v_thr = v_thr  #v threshold

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


class Synapse_model:
	#N:neuron数，dt:時間幅，w0:シナプス重み，tau_d:減少に関する時定数[s]，tau_r:上昇に関する時定数[s]
	def __init__(self, N, dt, w0, tau_d=20*1e-3, tau_r=2*1e-3):

		self.N = N

		self.dt = dt
		self.t_count = 0

		self.w0 = w0

		self.tau_d = tau_d
		self.tau_r = tau_r

		self.IPSC = np.zeros(self.N)
		self.IPSC_h = np.zeros(self.N)	#補助変数
		self.JD = np.zeros(self.N)
		self.w0 = w0


	def initialize_states(self):
		self.t_count = 0

		self.IPSC = np.zeros(self.N)
		self.IPSC_h = np.zeros(self.N)
		self.JD = np.zeros(self.N)


	#s_record:spike列
	def __call__(self, s_record):
		#spikeがあった所，数を取得
		index = np.where(s_record[:, self.t_count]==1)[0]
		len_index = len(index)

		#spikeがあれば結合先への重みの和を計算
		if len_index > 0:
			self.JD = np.sum(self.w0[index, :], axis=0)

		#Double exponential model
		self.IPSC = self.IPSC * np.exp(-self.dt/self.tau_d) + self.IPSC_h*self.dt
		self.IPSC_h = self.IPSC_h * np.exp(-self.dt/self.tau_r) + self.JD*(len_index>0) / (self.tau_r*self.tau_d)


		self.t_count += 1


		return (self.IPSC)





class Synapse_model_out:
	#N_out:Readour_neuron数，dt:時間幅，w_out:Res→outへのシナプス重み，w_out_const0:常に0である所，tau_d:減少に関する時定数[s]，tau_r:上昇に関する時定数[s]
	def __init__(self, N_out, dt, w_out, w_out_const0,  tau_d=20*1e-3, tau_r=20*1e-3):

		self.N_out = N_out


		self.w_out = w_out
		self.w_out_const0 = w_out_const0


		self.tau_d = tau_d
		self.tau_r = tau_r


		#STDPのLTP(LTD)の最大値
		self.A_B_LTP = 0.06
		self.A_B_LTD = -0.05
		self.stdp_scale = 1e-6



		self.IPSC = np.zeros(self.N_out)
		self.IPSC_h = np.zeros(self.N_out)
		self.JD = np.zeros(self.N_out)


		self.dt = dt
		self.t_count = 0



	def initialize_states(self):


		self.t_count = 0	
		self.IPSC = np.zeros(self.N_out)
		self.IPSC_h = np.zeros(self.N_out)
		self.JD = np.zeros(self.N_out)


	#s_record_res:Res_neuronのスパイク列，s_record_out:out_neuronのスパイク列，x_trace_res:Res_neuronのスパイク列のトレース，x_trace_out:out_neuronのスパイク列のトレース
	def __call__(self, s_record_res, s_record_out, x_trace_res, x_trace_out, batch=50):

		#batch数ごとに学習	
		if self.t_count % batch == 0 and self.t_count >= batch :

			if np.all((s_record_out[:, self.t_count - batch : self.t_count]) == 0) and np.all((s_record_res[:, self.t_count - batch : self.t_count]) == 0) :
				add = 0
			else :
				add_LTP = cp.asarray(x_trace_res) @ cp.asarray(s_record_out[:,  self.t_count - batch: self.t_count]).T
				add_LTD = cp.asarray(s_record_res[:, self.t_count - batch: self.t_count]) @ cp.asarray(x_trace_out).T
				
				add = (self.A_B_LTP * self.stdp_scale) * add_LTP + (self.A_B_LTD * self.stdp_scale) * add_LTD
		

		
			self.w_out += cp.asnumpy(add) 
			self.w_out = np.clip(self.w_out, -1e100000, 1e100000)  #最小値と最大値を制限
			#self.w_out = np.clip(self.w_out, 0, 1e100000)
			self.w_out[self.w_out_const0[0], self.w_out_const0[1]] = 0  #常に0である所

		



		index = np.where(s_record_res[:, self.t_count]==1)[0]
		len_index = len(index)


		if len_index > 0:
			
			self.JD = np.sum(self.w_out[index, :], axis=0)

			#print(JD.shape)

		self.IPSC = self.IPSC * np.exp(-self.dt/self.tau_d) + self.IPSC_h*self.dt
		self.IPSC_h = self.IPSC_h * np.exp(-self.dt/self.tau_r) + self.JD*(len_index>0) / (self.tau_r*self.tau_d)


		self.t_count += 1



		return (self.IPSC)





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



N_input =2

N_liquid=2000

N_out = 8


dt = 0.1 * 1e-3  #s



p_inp=0.4  #w_inの結合確率

p_liq=0.1  #w_liqの結合確率
variance = (N_liquid*p_liq**2)**-1
G=0.01  #scale

p_out = 0.2  #w_outの結合確率



w_in = np.random.randn(N_input, N_liquid) * (np.random.rand(N_input, N_liquid) < p_inp) / (np.sqrt(N_input)*p_inp)


w_liquid = np.random.randn(N_liquid, N_liquid) * (np.random.rand(N_liquid, N_liquid)<p_liq)*np.sqrt(variance)

for i in range(N_liquid):
	QS = np.where(np.abs(w_liquid[:, i]) > 0)[0]
	w_liquid[QS, i] = w_liquid[QS, i] - np.sum(w_liquid[QS, i]) / len(QS)


w_out = (np.random.rand(N_liquid, N_out) < p_out) * 1
w_out_const0 = np.where(w_out == 0)	
w_out = np.random.randn(N_liquid, N_out) * (w_out) / (np.sqrt(N_out)*p_out) * G
#w_out = np.abs(w_out)




BIAS = -40



LIF_liquid_neuron = LIF(N_liquid, dt, np.ones(N_liquid), t_ref_py=2* 1e-3, tau_m_py=10* 1e-3, v_reset=-65, v_peak=30, v_thr=-40)
synapse_liquid = Synapse_model(N_liquid, dt, w_liquid, tau_d=20* 1e-3, tau_r=2* 1e-3)


LIF_out_neuron = LIF(N_out, dt, np.ones(N_out), t_ref_py=2* 1e-3, tau_m_py=10* 1e-3, v_reset=-65, v_peak=30, v_thr=-40)
synapse_out = Synapse_model_out(N_out, dt, w_out, w_out_const0, tau_d=20* 1e-3, tau_r=2* 1e-3)



batch = 50




for epo in range(1):

	for i_size in range(100):

		for i in dir_name:

			df = pd.read_table(glob.glob("tactile_data/" +i+ f"/data_{i_size+1}_*")[0], header=None)
			df_np = df.to_numpy().T
			in_data_0 =  df_np[:3, 3000:8000]	

			dt_data = 0.1 * 1e-3  #s
			nt = in_data_0.shape[1]
			T = nt*dt_data #s
			t_array = np.arange(nt)*dt_data


			input_current = np.zeros([N_input, nt])

			for j in range(3):
				in_data = in_data_0[j,:]
				#in_data = np.interp(t_array, t_array, in_data_0[j,:])

				I_merkel = calc_merkel(in_data, t_array, dt)
				I_meissner = calc_meissner(in_data, t_array, dt)


				if j==0 :  #sensor1のみ使う
					input_current[j*2 , :] = 0.01*I_merkel
					input_current[j*2+1, :] = 0.04*I_meissner 




			LIF_liquid_neuron.initialize_states()
			synapse_liquid.initialize_states()


			IPSC=np.zeros(N_liquid)
			s_record = np.zeros([N_liquid, nt])
			v_record = np.zeros([N_liquid, nt])

			I_rec = np.zeros([N_liquid, nt])


			batch = 50


			temp = np.zeros(N_liquid)
			x_trace = np.zeros([N_liquid, batch])


			temp_out = np.zeros(N_out)
			x_trace_out = np.zeros([N_out, batch])


			LIF_out_neuron.initialize_states()
			synapse_out.initialize_states()


			IPSC_out = np.zeros(N_out)
			s_record_out = np.zeros([N_out, nt])
			v_record_out = np.zeros([N_out, nt])

			I_rec_out = np.zeros([N_out, nt])





			for t in tqdm(range(nt)):
				input_I = input_current[:, t] @ w_in

				I = IPSC + BIAS + input_I
				I_rec[:, t] = I

				s_record[:, t] = LIF_liquid_neuron(I, 0)
				v_record[:, t] = LIF_liquid_neuron.v_

				temp1 = np.where(s_record[:, t]==1)[0]
				temp2 = np.where(s_record[:, t]!=1)[0]

				temp[temp1] = 1	
				temp[temp2] -= temp[temp2]*dt/(10* 1e-3) 

				x_trace = np.hstack((x_trace, temp.reshape(-1,1)))
				x_trace = np.delete(x_trace, 0, 1)

				IPSC = synapse_liquid(s_record)
				IPSC = G  * IPSC




				
				s_record_out[:, t] = LIF_out_neuron(IPSC_out + BIAS, 0) 

				temp1 = np.where(s_record_out[:, t]==1)[0]
				temp2 = np.where(s_record_out[:, t]!=1)[0]

				temp_out[temp1] = 1
				temp_out[temp2] -= temp[temp2]*dt/(10* 1e-3) 
				x_trace_out = np.hstack((x_trace_out, temp_out.reshape(-1,1)))
				x_trace_out = np.delete(x_trace_out, 0, 1)

				IPSC_out = synapse_out(s_record, s_record_out, x_trace, x_trace_out, batch=batch)
				I_rec_out[:, t] = IPSC_out
			


			print("spike_liq", sum(sum(s_record)) / N_liquid)
			print("spike_out", s_record_out.sum(axis=1))


np.save("w_out_ltd0p051e-6_outneuron8_2input_bias-40", synapse_out.w_out)
			





