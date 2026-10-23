"""探索に使用するモデル・構造候補。定義だけでは通常のFIXED設定を変更しない。"""

CFG_STRUCT = {"NETWORK_PRESET": ["liq1_100__out1_40"]}
# 旧プリセット名から、現在のネットワーク設定へ対応を明示する。
NETWORK_PRESETS = {
    "liq1_100__out1_40": {"N_liq": [100], "r_inh_liq": 0.2, "N_out": [40]},
}
CFG_BASE = {
    "NEURON_PRESET": ["LIF_default"],
    "SYNAPSE_PRESET": ["double_exp_default"],
    "LEARNING_PRESET": ["off_default", "STDP_default", "T_STDP_default", "SRDP_default", "BCPNN_default"],
}
PRESET_MODELS = {
    "NEURON_PRESET": {"LIF_default": "LIF"},
    "SYNAPSE_PRESET": {"double_exp_default": "double_exp"},
    "LEARNING_PRESET": {name+"_default": name for name in ["off", "STDP", "T_STDP", "SRDP", "BCPNN"]},
}
