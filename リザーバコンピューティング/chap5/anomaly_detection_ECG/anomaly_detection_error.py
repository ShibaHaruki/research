#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# anomaly_detection_error.py: 本書の図5.6に対応するサンプルコード
#################################################################

#################################################################
# ■ 心電図波形(ECG)データの取得
#
# https://archive.physionet.org/cgi-bin/atm/ATM
# のPhysioBank ATMより下記のデータをダウンロードして同じフォルダに置く．
# - Database: MIT-BIH Long-Term ECG Database (ltdb)
# - Record: 14046
# - Length: 1min
# 
# Navigationを通じて
# - 正常データとして，4min-5minの区間のテキストデータを取得（normal.txtとして保存）
# - 異常データとして，6min-7minの区間のテキストデータを取得（anomaly.txtとして保存）
# - 各データは 128 [sample/sec] * 60 [sec] = 7680 [samples]
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

# 心電図波形(ECG)データの読み込み
def read_ecg_data(file_name):
    ''' 
    :入力：データファイル名, file_name
    :出力：ECGデータ, data
    '''
    data = np.empty(0)
    count = 0
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if count >= 2:  # ヘッダ２行削除
                data = np.hstack((data, float(tmp[1])))  # ECG1[mV]
            count = count + 1
    return data
    

if __name__ == "__main__":

    # ECGデータ
    normal = read_ecg_data(file_name='./normal.txt')
    anomaly = read_ecg_data(file_name='./anomaly.txt')
    normal_noise = read_ecg_data(file_name='./normal.txt')
    normal_noise = normal_noise + 0.4 * (np.random.rand(normal_noise.shape[0]) - 0.5)
    
    # 訓練、検証データ
    unit_sample = 128  # 1secあたりのサンプル数
    T_start = 1*unit_sample
    T_end = 59*unit_sample
    
    train_X = normal[T_start:T_end].reshape(-1, 1)
    train_Y = normal[T_start+1:T_end+1].reshape(-1, 1)

    test_X = anomaly[T_start:T_end].reshape(-1, 1)
    test_Y = anomaly[T_start+1:T_end+1].reshape(-1, 1)

    test2_X = normal_noise[T_start:T_end].reshape(-1, 1)
    test2_Y = normal_noise[T_start+1:T_end+1].reshape(-1, 1)

    # ESNモデルの設定
    N_x = 100  # リザバーの大きさ
    model = ESN(train_X.shape[1], train_Y.shape[1], N_x,
                density=0.1, input_scale=1, rho=0.8)
    
    # 学習
    train_Y_pred = model.train(train_X, train_Y,
                               Tikhonov(N_x, train_Y.shape[1], 1e-2))

    # 検証データに対する予測と誤差
    test_Y_pred = model.predict(test_X)
    error = abs(test_Y_pred - test_Y)
    test_Y_pred = model.predict(test2_X)
    error2 = abs(test_Y_pred - test2_Y)
    
    # グラフ表示
    plt.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.5)

    t = np.linspace(1, 59, error.shape[0])  # 時間（秒）
    threshold = 2  # 閾値

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.1, 1, '(a)', transform=ax1.transAxes)
    plt.plot(t, error, color='k')
    plt.ylabel("Error [mV]")
    #plt.yscale('log')
    plt.hlines([threshold], 1, 59, "gray", linestyles='dashed') 
    plt.title("Testing data 1")
    
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t, error2, color='k')
    plt.xlabel("Time [sec]")
    plt.ylabel("Error [mV]")
    #plt.yscale('log')
    plt.ylim([-0.1, 4.1])
    plt.hlines([threshold], 1, 59, "gray", linestyles='dashed') 
    plt.title("Testing data 2")
    
    plt.show()

