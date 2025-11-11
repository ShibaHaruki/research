#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# sunspot_smoothed_sr.py: 本書の図5.8に対応するサンプルコード
#################################################################

#################################################################
# ■ 太陽黒点数データの取得
#
# http://www.sidc.be/silso/datafiles#total
# の"13-month smoothed monthly total sunspot number [1/1749-now]"
# より下記のテキストデータをダウンロードして同じフォルダに置く．
# - Dataset: "SN_ms_tot_V2.0.txt"
# - now = 6/2020のとき，データ数は3258
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

# データの読み込み
def read_sunspot_data(file_name):
    ''' 
    :入力：データファイル名, file_name
    :出力：黒点数データ, data
    '''
    data = np.empty(0)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            data = np.hstack((data, float(tmp[3])))  # 3rd column
    return data


if __name__ == '__main__':
    
    # 黒点数データ
    sunspots = read_sunspot_data(file_name='SN_ms_tot_V2.0.txt')
    
    # データのスケーリング
    data_scale = 1.0e-3
    data = sunspots*data_scale

    # 複数ステップ先の予測
    step = 10

    # スペクトル半径rhoを変化させる
    rho_list = np.arange(0.5,1.5,0.05)
    train_NRMSE = np.empty(0)
    test_NRMSE = np.empty(0)
    for rho in rho_list:
        
        # 訓練・検証データ長
        T_train = 2500
        T_test = data.size-T_train-step
    
        # 訓練・検証用情報
        train_U = data[:T_train].reshape(-1, 1)
        train_D = data[step:T_train+step].reshape(-1, 1)
        
        test_U = data[T_train:T_train+T_test].reshape(-1, 1)
        test_D = data[T_train+step:T_train+T_test+step].reshape(-1, 1)
        
        # ESNモデル
        N_x = 300  # リザバーのノード数
        model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
                    input_scale=0.1, rho=rho)
        
        # 学習(リッジ回帰)
        train_Y = model.train(train_U, train_D, 
                              Tikhonov(N_x, train_D.shape[1], 1e-3))
        
        # モデル出力
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)
        
        # 訓練誤差評価（NRMSE）
        RMSE = np.sqrt(((train_D/data_scale - train_Y/data_scale) ** 2)
                       .mean())
        NRMSE = RMSE/np.sqrt(np.var(train_D/data_scale))
        train_NRMSE = np.hstack((train_NRMSE, NRMSE))
        
        # 検証誤差評価（NRMSE）
        RMSE = np.sqrt(((test_D/data_scale - test_Y/data_scale) ** 2)
                       .mean())
        NRMSE = RMSE/np.sqrt(np.var(test_D/data_scale))
        test_NRMSE = np.hstack((test_NRMSE, NRMSE))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))

    plt.plot(rho_list, train_NRMSE, marker='o', fillstyle='none',
             markersize=8, color='k', label='Training')         
    plt.plot(rho_list, test_NRMSE, marker='s', 
             markersize=8, color='k', label='Testing')
    plt.xticks(rho_list[::2])
    plt.xlabel(r'$\rho$')
    plt.ylabel('NRMSE')
    plt.title('%d-step-ahead prediction' % step)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left')
    
    plt.show()

