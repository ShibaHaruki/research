import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
#from scipy.interpolate import inter1d

rng = np.random.default_rng(1)

dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]
n_sozai = len(dir_name)

#N_out = 8

#T = 500 #ms
#dt = 0.1 #ms
#nt = T/dt
#n_interval = 5
#T_n = int(T/n_interval)

#n_sample = 100

T_n = int(sys.argv[1])
#T_n = 50 #ms
loop_size = 10

infname = 'sout_rec.npy'
sout_rec_org = np.load(infname)
#print(sout_rec.shape)
n_sozai = sout_rec_org.shape[0]
n_sample = sout_rec_org.shape[1]
N_out = sout_rec_org.shape[2]
T = sout_rec_org.shape[3]
n_interval = int(T/T_n)

sample_for_cls = 80
sample_for_tst = n_sample - sample_for_cls
accuracy8 = np.zeros(loop_size)
accuracy3 = np.zeros(loop_size)


#sample_seq = np.load('sample_seq.npy')

for loop_i in range(loop_size):
    sout_rec = rng.permutation(sout_rec_org,axis=1)
    s_for_cls = sout_rec[:, 0:sample_for_cls, :, :]
    s_for_tst = sout_rec[:, sample_for_cls:n_sample, :, :]
    
    sout_rec_fr = np.zeros([n_sozai, sample_for_cls, N_out*n_interval])
    sout_rec_ave = np.zeros([n_sozai, N_out*n_interval])
    sout_rec_cov = np.zeros([n_sozai, N_out*n_interval, N_out*n_interval])
    cov_mtrx = np.zeros([n_sozai, N_out*n_interval, N_out*n_interval])

    for i in range(n_sozai):
        for j in range(sample_for_cls):
            for k in range(N_out):
                for l in range(n_interval):
                    sout_rec_fr[i, j, k*n_interval+l] = np.sum(s_for_cls[i, j, k, l*T_n:(l+1)*T_n]) / (T_n/1000)
        sout_rec_ave[i, :] = np.sum(sout_rec_fr[i, :, :], axis=0) / n_sample
        sout_rec_cov[i, :, :] = np.cov(sout_rec_fr[i, :, :].T)  

    for i in range(n_sozai):
        cov_mtrx[i, :, :] = np.linalg.pinv(sout_rec_cov[i,:])

    feature_vec = np.zeros([N_out*n_interval])
    diff_sozai = np.zeros(n_sozai)
    seigo = np.zeros([n_sozai*sample_for_tst, 2])

    for i in range(n_sozai):
        for j in range(sample_for_tst):
            diff_sozai = np.zeros(n_sozai)
            feature_vec = np.zeros([N_out*n_interval])
              
            #calcurate mahalanobis's distances
            for k in range(N_out):
                for l in range(n_interval):
                        feature_vec[k*n_interval+l] = np.sum(s_for_tst[i, j, k, l*T_n:(l+1)*T_n]) / (T_n/1000)

            for m in range(n_sozai):              
                data_minus_ave = feature_vec - sout_rec_ave[m,:]
                diff_sozai[m] = np.sqrt( (data_minus_ave.reshape([1,-1]) @ cov_mtrx[m, :, :]) @ data_minus_ave.reshape([-1,1]))
                    
            seigo[i*sample_for_tst + j, 0] = int(i)
            seigo[i*sample_for_tst + j, 1] = int(np.argmin(diff_sozai))
            #print(seigo[i*n_sample + j, 0], seigo[i*n_sample + j, 1])
            #print(diff_sozai)

    conf_mtrx = np.zeros([8,8])
    y_true = seigo[:,0]
    y_pred = seigo[:,1]
    count = 0
    for i in range(len(y_true)):
        #print(i, i//20, int(y_true[i]), int(y_pred[i]))
        conf_mtrx[i//20,int(y_pred[i])] += 1
        if y_true[i] == y_pred[i]:
            #print(i, y_true[i], y_pred[i])
            count += 1
    accuracy8[loop_i] = count / (n_sozai*sample_for_tst)
    print(accuracy8[loop_i])
    print(conf_mtrx)
    

    mtrx1 = np.zeros([8,3])
    mtrx2 = np.zeros([3,3])

    mtrx1[:,0] =  conf_mtrx[:,0] + conf_mtrx[:,5] + conf_mtrx[:,7]
    mtrx1[:,1] =  conf_mtrx[:,3] + conf_mtrx[:,4] + conf_mtrx[:,6]
    mtrx1[:,2] =  conf_mtrx[:,1] + conf_mtrx[:,2]
    mtrx2[0,:] = mtrx1[0,:] + mtrx1[5,:] + mtrx1[7,:]
    mtrx2[1,:] = mtrx1[3,:] + mtrx1[4,:] + mtrx1[6,:]
    mtrx2[2,:] = mtrx1[1,:] + mtrx1[2,:]
    accuracy3[loop_i] =(mtrx2[0,0]+mtrx2[1,1]+mtrx2[2,2])/(n_sozai*sample_for_tst)
    print(accuracy3[loop_i])
    print(mtrx2)

print('Accuracy of 8 classes')
print(np.mean(accuracy8))
print('Accuracy of 3 classes')
print(np.mean(accuracy3))




#accuracy_mean = np.mean(loop_rec, axis=0)
#accuracy_max = np.max(loop_rec, axis=0)
#accuracy_min = np.min(loop_rec, axis=0)

#print(accuracy_max, accuracy_mean, accuracy_min)