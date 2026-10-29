"""共通Liquidから、STDP時定数5〜250 msの各候補へ80個ずつの出力を接続する。"""

from copy import deepcopy
from c_configs.FIXED.cfg_network import CFG_NETWORK as COMMON_NETWORK

# 各候補のtau_plusとtau_minusは同じ値とする。
STDP_TAU_CANDIDATES_MS = list(range(5, 251, 5))
# 共通設定とは独立したコピーへ、学習則と出力層ごとの条件を追加する。
CFG_NETWORK = deepcopy(COMMON_NETWORK)
CFG_NETWORK.update(
    {
        "output_enabled": True,
        "N_out": [80 for _ in STDP_TAU_CANDIDATES_MS],
        "learning_rule_liq_to_out": ["STDP" for _ in STDP_TAU_CANDIDATES_MS],
        "OUTPUT_LEARNING_PARAMETERS": [
            {"tau_plus": float(tau), "tau_minus": float(tau)} for tau in STDP_TAU_CANDIDATES_MS
        ],
        "SHARE_OUTPUT_INITIALIZATION": True,
        "p_liq_to_out_pairs": [
            {"EE": 0.11, "EI": 0.0, "IE": 0.0, "II": 0.0} for _ in STDP_TAU_CANDIDATES_MS
        ],
        "gain_liq_to_out_pairs": [
            {"EE": 0.2706, "EI": 0.0, "IE": 0.0, "II": 0.0} for _ in STDP_TAU_CANDIDATES_MS
        ],
    }
)
