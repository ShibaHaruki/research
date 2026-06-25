"""触覚波形を Merkel / Meissner 型の入力電流へ変換するフィルタ。"""

import numpy as np

DERIVATIVE_WIDTH = 25
FAST_FILTER_TAU_S = 0.2 * 1e-3


def _abs_derivative(data, t, i, width=DERIVATIVE_WIDTH):
	j = max(0, i - int(width))
	if j == i:
		return 0.0
	return np.abs(data[i] - data[j]) / (t[i] - t[j])

def calc_meissner(data, t, dt):
	# 速い変化に反応する Meissner 型の入力電流を作る。

	I = np.zeros((4, len(t)))
	
	for i in range(len(t)):
		if(i!=0):
			dF_dt = _abs_derivative(data, t, i)

			I[0, i] = I[0, i-1] + 0.74*dF_dt + (-I[0, i-1]*dt/FAST_FILTER_TAU_S) 
			I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200* 1e-3) )
			I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6* 1e-3))
			I[3, i] = I[0, i] 

	return I[3,:]


def calc_merkel(data, t, dt):
	# 力の変化と持続成分を合わせた Merkel 型の入力電流を作る。

	I = np.zeros((4, len(t)))
	
	for i in range(len(t)):
		if(i!=0):
			dF_dt = _abs_derivative(data, t, i)

			if(dF_dt < 0):
				dF_dt = 0
			I[0, i] = I[0, i-1] + 0.74*dF_dt + (-I[0, i-1]*dt/FAST_FILTER_TAU_S) 
			I[1, i] = I[1, i-1] + 0.24*dF_dt + (-(I[1, i-1] - 0.24*0.13)*dt/(200*1* 1e-3) )
			I[2, i] = I[2, i-1] + 0.07*dF_dt + (-I[2, i-1]*dt/(1744.6*1* 1e-3))
			I[3, i] = I[0, i] +  I[1, i] +  I[2, i] 

	return I[3,:]
