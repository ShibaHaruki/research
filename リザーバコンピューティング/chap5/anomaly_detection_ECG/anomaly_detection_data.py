# !/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# anomaly_detection_data.py: 本書の図5.5に対応するサンプルコード
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
from model import ESN


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
            if count >= 2:  # ヘッダ削除
                data = np.hstack((data, float(tmp[1])))  # ECG1[mV]
            count = count + 1
    return data
    

if __name__ == "__main__":
    
    # 訓練データ、検証データ
    normal = read_ecg_data(file_name='./normal.txt')
    anomaly = read_ecg_data(file_name='./anomaly.txt')
    normal_noise = read_ecg_data(file_name='./normal.txt')
    normal_noise = normal_noise + \
        0.4 * (np.random.rand(normal_noise.shape[0]) - 0.5)
    
    # 時間
    t = np.linspace(0, 60, 7680)
    
    # グラフ表示
    plt.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.8)
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.12, 1.2, '(a)', transform=ax1.transAxes)
    plt.plot(t, normal, color='k')
    plt.ylim([-3, 4])
    plt.title("Training data")
    plt.ylabel("Voltage [mV]")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.12, 1.2, '(b)', transform=ax2.transAxes)
    plt.plot(t, anomaly, color='k')
    plt.ylim([-3, 4])
    plt.title("Testing data 1")
    plt.ylabel("Voltage [mV]")
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.text(-0.12, 1.2, '(c)', transform=ax3.transAxes)
    plt.plot(t, normal_noise, color='k')
    plt.ylim([-3, 4])
    plt.title("Testing data 2")
    plt.ylabel("Voltage [mV]")
    plt.xlabel("Time [sec]")
    
    plt.show()

