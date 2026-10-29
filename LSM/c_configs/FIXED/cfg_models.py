"""使用するニューロンモデル、シナプスモデル、学習則を選ぶ設定。"""

CFG_MODELS = {
    "NEURON_MODEL": "LIF",
    "SYNAPSE_MODEL": "double_exp",
    "LEARNING_RULE_MODEL": "off",  # 旧設定の既定値。通常はcfg_networkの出力層別指定を使う。
}
