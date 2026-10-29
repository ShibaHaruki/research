"""現在のFIXED設定のキーに対応した、ネットワークパラメータ候補。"""

# ドット区切りのキーはnetwork辞書内のパス。採用値も候補に含める。
SPACE_NETWORK = {
    "r_inh_liq": [0.1, 0.2, 0.3],
    "p_liq_intra_pairs.EE": [0.02, 0.05, 0.10, 0.11],
    "p_liq_intra_pairs.EI": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.IE": [0.02, 0.05, 0.10, 0.37],
    "p_liq_intra_pairs.II": [0.02, 0.05, 0.10],
    "liq_intra_gain_pairs.EE": [0.150, 0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.EI": [0.206, 0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.IE": [0.01390625, 0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.II": [0.0, 0.5, 1.0, 2.0],
}

# リストの添字や(sensor, filter)のタプルキーを含むパスはこちらで表す。
# 6経路は個別の候補。全センサー共通で動かす場合は探索側で同じ値を適用する。
SPACE_NETWORK_TUPLEPATH = {
    ("IN_ROUTE", (sensor, filter_name), "layers", 0, key, kind): candidates
    for sensor in range(3)
    for filter_name in ("RI", "SI")
    for key, kind, candidates in [
        ("p", "E", [0.05, 0.10, 0.20]),
        ("p", "I", [0.0]),
        ("scale", "E", [10.0, 20.0, 40.0, 179.0]),
        ("scale", "I", [0.0]),
    ]
}
SPACE_NETWORK_TUPLEPATH.update(
    {
        ("p_liq_to_out_pairs", 0, "EE"): [0.05, 0.10, 0.20, 0.8, 1.0],
        ("p_liq_to_out_pairs", 0, "EI"): [0.0],
        ("p_liq_to_out_pairs", 0, "IE"): [0.0],
        ("p_liq_to_out_pairs", 0, "II"): [0.0],
        ("gain_liq_to_out_pairs", 0, "EE"): [0.2706, 0.5, 0.6, 1.0, 2.0],
        ("gain_liq_to_out_pairs", 0, "EI"): [0.0],
        ("gain_liq_to_out_pairs", 0, "IE"): [0.0],
        ("gain_liq_to_out_pairs", 0, "II"): [0.0],
    }
)
