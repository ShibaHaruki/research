#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# sunspot_smoothed_prediction.py: 本書の図5.7に対応するサンプルコード
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
    step_list = [1,10]
    for step in step_list:
        
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
                    input_scale=0.1, rho=0.9)
        
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
        print(step, 'ステップ先予測')
        print('訓練誤差：NRMSE =', NRMSE)

        # 検証誤差評価（NRMSE）
        RMSE = np.sqrt(((test_D/data_scale - test_Y/data_scale) ** 2)
                            .mean())
        NRMSE = RMSE/np.sqrt(np.var(test_D/data_scale))
        print(step, 'ステップ先予測')
        print('検証誤差：NRMSE =', NRMSE)

        # step=1,10の場合を記録
        if step==1:
            # グラフ表示用データ
            T_disp = (T_train-300, T_train+300)
            t_axis = np.arange(T_disp[0], T_disp[1], 1)
            disp_D_step1 = np.concatenate((train_D[T_disp[0]-T_train:], 
                                          test_D[:T_disp[1]-T_train]))
            disp_Y_step1 = np.concatenate((train_Y[T_disp[0]-T_train:], 
                                          test_Y[:T_disp[1]-T_train]))
        else:
            # グラフ表示用データ
            T_disp = (T_train-300, T_train+300)
            t_axis = np.arange(T_disp[0], T_disp[1], 1)
            disp_D_step10 = np.concatenate((train_D[T_disp[0]-T_train:], 
                                           test_D[:T_disp[1]-T_train]))
            disp_Y_step10 = np.concatenate((train_Y[T_disp[0]-T_train:], 
                                           test_Y[:T_disp[1]-T_train]))
        
    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    plt.plot(data/data_scale, color='k')
    plt.ylabel('Num of sunspots')
    
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    ax2.text(0.3, 0.85, 'Training', transform=ax2.transAxes)
    ax2.text(0.6, 0.85, 'Testing', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D_step1[:,0]/data_scale, color='k', label='Target')
    plt.plot(t_axis, disp_Y_step1[:,0]/data_scale, color='gray',
             linestyle='--', label='Model')
    plt.ylabel('Num of sunspots')
    plt.ylim([0,500])
    plt.axvline(x=T_train, ymin=0, ymax=1, color='k', linestyle=':')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    ax3.text(0.3, 0.85, 'Training', transform=ax3.transAxes)
    ax3.text(0.6, 0.85, 'Testing', transform=ax3.transAxes)
    plt.plot(t_axis, disp_D_step10[:,0]/data_scale, color='k', label='Target')
    plt.plot(t_axis, disp_Y_step10[:,0]/data_scale, color='gray',
             linestyle='--', label='Model')
    plt.xlabel('Index of data (unit: month)')
    plt.ylabel('Num of sunspots')
    plt.ylim([0,500])
    plt.axvline(x=T_train, ymin=0, ymax=1, color='k', linestyle=':')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    
    plt.show()

