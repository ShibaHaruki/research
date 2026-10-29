"""共有Liquidのsweep設定と、学習則ごとの一般的なパラメータ候補。"""

# --sweepで直接使用する設定。tau_plus=tau_minusで50候補を比較する。
from c_configs.FIXED.cfg_network_stdp import STDP_TAU_CANDIDATES_MS

TAU_CANDIDATES_MS = list(STDP_TAU_CANDIDATES_MS)
SEED_COUNT = 15

# A・重み上下限・結合確率・gainはFIXED、試行数・素材・エポック数はexperiment_sets.pyの共通値を使う。

# SPACE/cfg_learning_rule_models.py
# 以下は一般のpresetごとの探索空間。上の--sweep設定とは独立した候補定義。

LEARNING_RULE_SPACE = {
    "off_default": {
        # learning_rule="off" は探索するパラメータ無しでもOK
    },
    "STDP_default": {
        "tau_plus": [8, 11.7, 15],
        "tau_minus": [10, 14.0, 18],
        "A_plus": [1.00e-7],
        "A_minus": [-1.05e-7],
        "wmin": [0.0],
        "wmax": [10.0],
    },
    "T_STDP_default": {
        "tau_plus1": [8, 11.7, 15],
        "tau_plus2": [10, 15, 20],
        "tau_minus1": [10, 14.0, 18],
        "tau_minus2": [10, 15, 20],
        "A2_plus": [1.00e-7],
        "A3_plus": [1e-8, 5e-8, 1e-7],
        # T-STDPの抑圧項は正の値を更新式内で引く。
        "A2_minus": [1.05e-7],
        "A3_minus": [1e-8, 5e-8, 1e-7],
        "wmin": [0.0],
        "wmax": [10.0],
    },
    "SRDP_default": {
        "tau_plus": [8, 11.7, 15],
        "tau_minus": [10, 14.0, 18],
        "tau_pre": [10, 15, 20],
        "tau_post": [10, 15, 20],
        "A_plus": [1.00e-7],
        "A_minus": [1.05e-7],
        "A_pre": [1e-5, 5e-5, 1e-4],
        "A_post": [1e-5, 5e-5, 1e-4],
        "wmin": [0.0],
        "wmax": [10.0],
    },
    "BCPNN_default": {
        "tau_z": [2.0, 5.0, 10.0],
        "tau_p": [500.0, 1000.0, 2000.0],
        "z_inc": [1.0],
        "bcpnn_eps": [1e-6],
        "bcpnn_gain": [0.01, 0.05, 0.1],
        "bcpnn_w_offset": [0.0],
        "wmin": [0.0],
        "wmax": [10.0],
    },
}
