"""採用したgain179条件のネットワーク設定。

入力179、EE 0.150、EI 0.206、IE 0.01390625、II 0。
3センサーそれぞれのRI/SIをEへ接続し、Iへの直接入力は使わない。
"""

CFG_NETWORK = {
    # リキッド層：E80個、I20個。
    "N_liq": [100],
    "r_inh_liq": 0.2,

    # 入力条件はIN_ROUTEへ明示するため、共有設定による上書きは無効。
    "SHARED_IN_ROUTE": {"enabled": False, "filters": {}},

    # 3センサー × RI/SI の計6経路。
    # 各経路→Eは5%、gain179。各経路→Iは確率・gainともに0。
    # 結合先と重みはBASE_SEEDで固定し、従来の検証条件を再現する。
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

    # 背景ポアソン入力は使わない。
    "poisson_input": {
        "enabled": False,
        "N_ratio": None,
        "N": 100,
        "rate_hz": 5.0,
        "current": "exc",
        "p": {"E": 0.1, "I": 0.1},
        "scale": {"E": 0.2, "I": 0.2},
    },

    # リキッド層内の結合。II確率は残すが、gain0でIIの作用を無効にする。
    "liq_intra_connection": "random",
    "p_liq_intra_pairs": {"EE": 0.11, "EI": 0.10, "IE": 0.37, "II": 0.10},
    "liq_intra_gain_pairs": {"EE": 0.150, "EI": 0.206, "IE": 0.01390625, "II": 0.0},
}
