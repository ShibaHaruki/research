"""使用するニューロンモデル、シナプスモデル、学習則を選ぶ設定。"""

CFG_MODELS = {
    "NEURON_MODEL": "LIF",
    "SYNAPSE_MODEL": "double_exp",
    "LEARNING_RULE_MODEL": "off",  # 学習時はCLI --rule SRDP/STDP/T_STDPなどで選択。
}
