
# ---- dot-path で触れる（入れ子 dict の中身） ----
SPACE_NETWORK = {
    # 例：液体内の接続確率（p_liq_intra_pairs が最優先で使われる）
    "p_liq_intra_pairs.EE": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.EI": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.IE": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.II": [0.02, 0.05, 0.10],

    # 例：液体内のゲイン
    "liq_intra_gain_pairs.EE": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.EI": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.IE": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.II": [0.5, 1.0, 2.0],

    # 例：liq->out の全体ゲイン
    "gain": [0.5, 1.0, 2.0],

    # 例：抑制比も探索したいなら（層数/配列長に注意）
    "r_inh_liq": [0.1, 0.2, 0.3],
    "r_inh_out": [0.0, 0.1],
}

# ---- tuple-path で触れる（タプルキー/リスト index を含む深い要素） ----
# 形式: (トップキー, 次キー, 次キー, ...): [候補...]
# IN_ROUTE は (ch, filter) がキーなので tuple-path が楽
SPACE_NETWORK_TUPLEPATH = {
    # in-route: ch0 merkel layer0 の p, scale を探索
    ("IN_ROUTE", (0, "merkel"), "layers", 0, "p"):     [0.05, 0.10, 0.20],
    ("IN_ROUTE", (0, "merkel"), "layers", 0, "scale"): [0.25, 0.50, 1.00],

    # in-route: ch0 meissner
    ("IN_ROUTE", (0, "meissner"), "layers", 0, "p"):     [0.05, 0.10, 0.20],
    ("IN_ROUTE", (0, "meissner"), "layers", 0, "scale"): [0.25, 0.50, 1.00],

    # liq->out の確率を探索したい例（p_liq_to_out_pairs[0][0]["EE"] など）
    # out_idx=0, liq_idx=0 の EE を振る
    ("p_liq_to_out_pairs", 0, 0, "EE"): [0.05, 0.10, 0.20],
    ("p_liq_to_out_pairs", 0, 0, "EI"): [0.05, 0.10, 0.20],
    ("p_liq_to_out_pairs", 0, 0, "IE"): [0.05, 0.10, 0.20],
    ("p_liq_to_out_pairs", 0, 0, "II"): [0.05, 0.10, 0.20],
}