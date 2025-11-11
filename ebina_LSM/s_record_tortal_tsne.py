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




dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]
sozai_n = len(dir_name)


N_input = 2

N_liquid=2000

N_out = 100

T = 500  #ms
dt = 0.1 #ms
T = T/1000 #s


s_record_tortal = np.load("w_out_ltd0p051e-6_outneuron8_2input_bias-40_10s_record_tortal.npy")
base_size = 10


s_record_fr = np.zeros([sozai_n, N_out, base_size])
s_record_ave = np.zeros([sozai_n, N_out])
s_record_cov = np.zeros([sozai_n, N_out, N_out])

for i in range(sozai_n):
	for j in range(N_out):


		s_record_fr[i,j,:] = np.sum(s_record_tortal[j, :, i*base_size : (i+1)*base_size]/T, axis=0)
		s_record_ave[i,j] = np.mean(s_record_fr[i,j,:])

	s_record_cov[i, :, :] = np.cov(s_record_fr[i, :, :]) 



X = np.zeros([sozai_n*base_size, N_out])
ii=0
for i in range(sozai_n):
	for k in range(base_size):

		X[ii, :] = s_record_fr[i, :, k]
		ii += 1






# ##############################lof##########################################
# base_size = int(base_size*0.8)
# s_record_fr_new = np.zeros([sozai_n, N_out, base_size])

# # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
# for i in range(sozai_n):
# 	from sklearn.neighbors import LocalOutlierFactor
# 	clf = LocalOutlierFactor(n_neighbors=3)
# 	# print(s_record_fr[i, :, :].T.shape)
# 	pred = clf.fit_predict(s_record_fr[i, :, :].T)
# 	#pred = clf.fit_predict(X[i*10 : (i+1)*10, :])
# 	lof_scores = clf.negative_outlier_factor_
# 	# lof_scores = np.delete(lof_scores, np.argsort(lof_scores)[:2])
# 	# print(np.argsort(lof_scores)[2:]+i*10)
# 	# print(s_record_fr_new[i, :, :].shape, s_record_fr[i, :,  np.argsort(lof_scores)[2:]].T.shape)
# 	s_record_fr_new[i, :, :] = s_record_fr[i, :, np.argsort(lof_scores)[-base_size:]].T



# X = np.zeros([sozai_n*base_size, N_out])
# ii=0
# for i in range(sozai_n):
# 	for k in range(base_size):

# 		X[ii, :] = s_record_fr_new[i, :, k]
# 		ii += 1

# #######################################################################




from sklearn.manifold import TSNE
colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink"]

tsne = TSNE(n_components=2, random_state = 0)	
X_tsne = tsne.fit_transform(X)	


# for i in range(sozai_n):
# 	plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=20, label=dir_name[i])











fig = plt.figure(figsize = (12, 12))		
plt.rcParams["font.size"] = 18


dir_name_group = ["G1(al, rubber, wood)", "G2(cork, denim, washi)", "G3(omote, ura)"]
for i in range(sozai_n):

	if i == 0 or i == 5 or i == 7 :

		if i ==0:

			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[0], label=dir_name_group[0])

		else:
			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[0])

	elif i == 6 or i == 3 or i == 4 :

		if i==3:

			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[1])

		else:
			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[1])

	else:

		if i == 1:
			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[1], label=dir_name_group[1])
			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[2], label=dir_name_group[2])
		else:
			plt.scatter(X_tsne[i*base_size:(i+1)*base_size, 0], X_tsne[i*base_size:(i+1)*base_size, 1], s=70, color=colors[2])


plt.tight_layout()
plt.rcParams["font.size"] = 18
plt.legend()
plt.show()		