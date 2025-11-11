#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# waveform_classification.py: 本書の図4.5に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

# 正弦波とのこぎり波の混合波形生成
class SinSaw:
    def __init__(self, period):
        self.period = period  # 周期

    # 正弦波
    def sinusoidal(self):
        n = np.arange(self.period)
        x = np.sin(2*np.pi*n/self.period)

        return x

    # のこぎり波
    def saw_tooth(self):
        n = np.arange(self.period)
        x = 2*(n/self.period - np.floor(n/self.period+0.5))

        return x

    def make_output(self, label):
        y = np.zeros((self.period, 2))
        y[:, label] = 1

        return y

    # 混合波形及びラベルの出力
    def generate_data(self, label):
        '''
        :param label: 0または1を要素に持つリスト
        :return: u: 混合波形
        :return: d: 2次元ラベル（正弦波[1,0], のこぎり波[0,1]）
        '''
        u = np.empty(0)
        d = np.empty((0, 2))
        for i in label:
            if i:
                u = np.hstack((u, self.saw_tooth()))
            else:
                u = np.hstack((u, self.sinusoidal()))
            d = np.vstack((d, self.make_output(i)))

        return u, d


# 出力のスケーリング
class ScalingShift:
    def __init__(self, scale, shift):
        '''
        :param scale: 出力層のスケーリング（scale[n]が第n成分のスケーリング）
        :param shift: 出力層のシフト（shift[n]が第n成分のシフト）
        '''
        self.scale = np.diag(scale)
        self.shift = np.array(shift)
        self.inv_scale = np.linalg.inv(self.scale)
        self.inv_shift = -np.dot(self.inv_scale, self.shift)

    def __call__(self, x):
        return np.dot(self.scale, x) + self.shift

    def inverse(self, x):
        return np.dot(self.inv_scale, x) + self.inv_shift


if __name__ == '__main__':

    # 訓練データ，検証データの数
    n_wave_train = 60
    n_wave_test = 40

    # 時系列入力データ生成
    period = 50
    dynamics = SinSaw(period)
    label = np.random.choice(2, n_wave_train+n_wave_test)
    u, d = dynamics.generate_data(label)
    T = period*n_wave_train

    # 訓練・検証用情報
    train_U = u[:T].reshape(-1, 1)
    train_D = d[:T]

    test_U = u[T:].reshape(-1, 1)
    test_D = d[T:]

    # 出力のスケーリング関数
    output_func = ScalingShift([0.5, 0.5], [0.5, 0.5])

    # ESNモデル
    N_x = 50  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
                input_scale=0.2, rho=0.9, fb_scale=0.05, 
                output_func=output_func, inv_output_func=output_func.inverse, 
                classification = True, average_window=period)

    # 学習（リッジ回帰）
    train_Y = model.train(train_U, train_D, 
                          Tikhonov(N_x, train_D.shape[1], 0.1)) 

    # 訓練データに対するモデル出力
    test_Y = model.predict(test_U)

    # 評価（正解率, accracy）
    mode = np.empty(0, np.int)
    for i in range(n_wave_test):
        tmp = test_Y[period*i:period*(i+1), :]  # 各ブロックの出力
        max_index = np.argmax(tmp, axis=1)  # 最大値をとるインデックス
        histogram = np.bincount(max_index)  # そのインデックスのヒストグラム
        mode = np.hstack((mode, np.argmax(histogram)))  #  最頻値

    target = test_D[0:period*n_wave_test:period,1]
    accuracy = 1-np.linalg.norm(mode.astype(np.float)-target, 1)/n_wave_test
    print('accuracy =', accuracy)

    # グラフ表示用データ
    T_disp = (-500, 500)
    t_axis = np.arange(T_disp[0], T_disp[1])  # 時間軸
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]])) 
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.1, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
    plt.ylim([-0.3, 1.3])
    plt.ylabel('Output 1')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax3 = fig.add_subplot(3, 1, 3)
    plt.plot(t_axis, disp_D[:, 1], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:, 1], color='gray', linestyle='--', label='Model')
    plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
    plt.ylim([-0.3, 1.3])
    plt.xlabel('n')
    plt.ylabel('Output 2')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()
