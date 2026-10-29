"""学習則に共通の入力・Liquidネットワーク設定。出力層構成は各学習則の設定で追加する。"""

# 入力gain179、E80個・I20個、EE/EI/IE/IIの採用値を共通管理する。
CFG_NETWORK = {
    "N_liq": [100],
    "r_inh_liq": 0.2,
    "SHARED_IN_ROUTE": {"enabled": False, "filters": {}},
    "IN_ROUTE": {
        (sensor, filter_name): {
            "layers": {
                0: {
                    "p": {"E": 0.05, "I": 0.0},
                    "scale": {"E": 179.0, "I": 0.0},
                },
            },
        }
        for sensor in range(3)
        for filter_name in ("RI", "SI")
    },
    "liq_intra_connection": "random",
    "p_liq_intra_pairs": {"EE": 0.11, "EI": 0.10, "IE": 0.37, "II": 0.10},
    "liq_intra_gain_pairs": {"EE": 0.150, "EI": 0.206, "IE": 0.01390625, "II": 0.0},
}
