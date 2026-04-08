# SPACE/cfg_base.py
# ここは「探索する対象の選択」だけを持つ（辞書のみ）

# ★最上位：ネットワーク構造プリセット（baseより上位の概念）
CFG_STRUCT = {
    "NETWORK_PRESET": [
        "liq1_1000__out1_10",
        "liq1_1000__out2_10_10",
        "liq2_600_600__out1_10",
    ]
}

# base：モデル系（ここは構造に比べたら下位）
CFG_BASE = {
    "NEURON_PRESET": ["LIF_default"],
    "SYNAPSE_PRESET": ["double_exp_default"],
    "LEARNING_PRESET": ["STDP_default"],
}

