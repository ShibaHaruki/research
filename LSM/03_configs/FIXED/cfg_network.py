# SPACE/cfg_network.py
# 構造プリセット + 微調整（dot-path） + 深い要素（tuple-path）
# ※このファイルは「辞書だけ」

# =========================
# 1) 構造プリセット（層数・N_liq・output_layer・N_out・IN_ROUTE・p_liq_to_out_pairs などを丸ごと切替）
# =========================
NETWORK_PRESET_OVERRIDES = {
    # ----------------------------
    # liquid 1層 / output 1層
    # ----------------------------
    "liq1_1000__out1_10": {
        "liquid_layer": 1,
        "N_liq": [1000],
        "r_inh_liq": 0.2,

        "output_layer": 1,
        "N_out": [10],
        "r_inh_out": 0.0,

        "IN_ROUTE": {
            (0, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (0, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (1, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (1, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (2, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (2, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (3, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (3, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
        },

        # out_idx=0 のみ / liq_idx=0 のみ
        "p_liq_to_out_pairs": [
            {0: {"EE": 0.20, "EI": 0.20, "IE": 0.20, "II": 0.20}}
        ],
        "gain": 1.0,
    },

    # ----------------------------
    # liquid 1層 / output 2層（例：10,10）
    # ----------------------------
    "liq1_1000__out2_10_10": {
        "liquid_layer": 1,
        "N_liq": [1000],
        "r_inh_liq": 0.2,

        "output_layer": 2,
        "N_out": [10, 10],
        "r_inh_out": 0.0,

        "IN_ROUTE": {
            (0, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (0, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (1, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (1, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (2, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (2, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (3, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}}},
            (3, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}}},
        },

        # out_idx=0 と out_idx=1 / liq_idx=0 のみ
        "p_liq_to_out_pairs": [
            {0: {"EE": 0.20, "EI": 0.20, "IE": 0.20, "II": 0.20}},  # out 0
            {0: {"EE": 0.20, "EI": 0.20, "IE": 0.20, "II": 0.20}},  # out 1
        ],
        "gain": 1.0,
    },

    # ----------------------------
    # liquid 2層 / output 1層（参考）
    # ----------------------------
    "liq2_600_600__out1_10": {
        "liquid_layer": 2,
        "N_liq": [600, 600],
        "r_inh_liq": 0.2,

        "output_layer": 1,
        "N_out": [10],
        "r_inh_out": 0.0,

        # layer 0/1 に撒く例
        "IN_ROUTE": {
            (0, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (0, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (1, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (1, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (2, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (2, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (3, "merkel"):   {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
            (3, "meissner"): {"layers": {0: {"p": 0.10, "scale": 0.50}, 1: {"p": 0.10, "scale": 0.50}}},
        },

        # out_idx=0 のみ / liq_idx=0,1
        "p_liq_to_out_pairs": [
            {
                0: {"EE": 0.20, "EI": 0.20, "IE": 0.20, "II": 0.20},
                1: {"EE": 0.20, "EI": 0.20, "IE": 0.20, "II": 0.20},
            }
        ],
        "gain": 1.0,
    },
}


# =========================
# 2) dot-path：入れ子 dict の中身を探索（値は list）
# =========================
SPACE_NETWORK = {
    # liquid intra connection probability
    "p_liq_intra_pairs.EE": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.EI": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.IE": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.II": [0.02, 0.05, 0.10],

    # liquid intra gains
    "liq_intra_gain_pairs.EE": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.EI": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.IE": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.II": [0.5, 1.0, 2.0],

    # global gain for liq->out (default if gain_liq_to_out_pairs not used)
    "gain": [0.5, 1.0, 2.0],

    # inhibition ratios (scalar or per-layer list; keep simple here)
    "r_inh_liq": [0.1, 0.2, 0.3],
    "r_inh_out": [0.0, 0.1],
}


# =========================
# 3) tuple-path：タプルキー/リストindexを含む深い要素を探索（値は list）
# =========================
# 形式: (トップキー, 次キー, 次キー, ...): [候補...]
# p_liq_to_out_pairs のアクセス例:
#   cfg["p_liq_to_out_pairs"][out_idx][liq_idx]["EE"]
SPACE_NETWORK_TUPLEPATH = {
    # ---- IN_ROUTE tuning (例) ----
    ("IN_ROUTE", (0, "merkel"), "layers", 0, "p"):     [0.05, 0.10, 0.20],
    ("IN_ROUTE", (0, "merkel"), "layers", 0, "scale"): [0.25, 0.50, 1.00],

    ("IN_ROUTE", (0, "meissner"), "layers", 0, "p"):     [0.05, 0.10, 0.20],
    ("IN_ROUTE", (0, "meissner"), "layers", 0, "scale"): [0.25, 0.50, 1.00],

    # ---- ★ 出力層ごとに確率を探索（out_idx別） ----
    # (output_layer=2 のプリセットを選んだときに out_idx=1 が有効になる)

    # out 0, liq 0
    ("p_liq_to_out_pairs", 0, 0, "EE"): [0.05, 0.10, 0.20],
    ("p_liq_to_out_pairs", 0, 0, "EI"): [0.05, 0.10, 0.20],
    ("p_liq_to_out_pairs", 0, 0, "IE"): [0.05, 0.10, 0.20],
    ("p_liq_to_out_pairs", 0, 0, "II"): [0.05, 0.10, 0.20],

    # out 1, liq 0
    ("p_liq_to_out_pairs", 1, 0, "EE"): [0.02, 0.05, 0.10],
    ("p_liq_to_out_pairs", 1, 0, "EI"): [0.02, 0.05, 0.10],
    ("p_liq_to_out_pairs", 1, 0, "IE"): [0.02, 0.05, 0.10],
    ("p_liq_to_out_pairs", 1, 0, "II"): [0.02, 0.05, 0.10],

    # liquid_layer=2 のプリセットで liq_idx=1 も探索したいなら、以下を追加すればOK
    # ("p_liq_to_out_pairs", 0, 1, "EE"): [...], ... など
}