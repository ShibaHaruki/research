"""触覚波形を Merkel / Meissner 型の入力電流へ変換するフィルタ。"""

import numpy as np

def calc_meissner(data, t, dt):
	# 速い変化に反応する Meissner 型の入力電流を作る。

	I = np.zeros((4, len(t)))
	
	for i in range(len(t)):
		if(i!=0):
			dF_dt = np.abs(data[i] - data[i-1])/(t[i]-t[i-1])

			I[0, i] = I[0, i-1] + 0.74*dF_dt + (-I[0, i-1]*dt/(8*1* 1e-3)) 
			I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200* 1e-3) )
			I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6* 1e-3))
			I[3, i] = I[0, i] 

	return I[3,:]


def calc_pacinian(data, t, dt):
	# Pacinian-like rapidly adapting filter.
	# Extract small jagged high-frequency changes riding on the Meissner-like
	# response, instead of following the broad Meissner envelope itself.
	I = np.zeros((4, len(t)))
	floor_gain = 0.002
	meissner_like = calc_meissner(data, t, dt)
	
	for i in range(len(t)):
		if(i!=0):
			dt_i = t[i] - t[i-1]
			if dt_i <= 0:
				dt_i = dt
			dM = np.abs(meissner_like[i] - meissner_like[i-1])

			I[0, i] = I[0, i-1] + 2.0*dM + (-I[0, i-1]*dt/(3.0*1e-3))
			I[1, i] = I[1, i-1] + 0.35*dM + (-I[1, i-1]*dt/(40.0*1e-3))
			I[2, i] = max(I[0, i] - 0.45*I[1, i], 0.0)
			I[3, i] = I[2, i] + floor_gain * abs(data[i])

	return I[3,:]


def calc_merkel(data, t, dt):
	# 力の変化と持続成分を合わせた Merkel 型の入力電流を作る。

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
