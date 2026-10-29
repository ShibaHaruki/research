"""共通Liquidから、学習なし・固定重みの出力80個を1群接続する。"""

from copy import deepcopy
from c_configs.FIXED.cfg_network import CFG_NETWORK as COMMON_NETWORK

# 学習則の違いだけを比較できるよう、Liquidの設定を共通の定義から取り込む。
CFG_NETWORK = deepcopy(COMMON_NETWORK)
CFG_NETWORK.update(
    {
        "output_enabled": True,
        "N_out": [80],
        "learning_rule_liq_to_out": ["off"],
        "OUTPUT_LEARNING_PARAMETERS": [{}],
        "SHARE_OUTPUT_INITIALIZATION": True,
        "p_liq_to_out_pairs": [{"EE": 0.11, "EI": 0.0, "IE": 0.0, "II": 0.0}],
        "gain_liq_to_out_pairs": [{"EE": 0.2706, "EI": 0.0, "IE": 0.0, "II": 0.0}],
    }
)
