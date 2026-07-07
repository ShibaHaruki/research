"""ネットワーク構造や接続パラメータを探索するときの候補値リスト。"""

# SPACE/cfg_network.py
#
# Search keys mirror FIXED/cfg_network.py.
# Dot-path keys are used for simple nested dictionaries.
# Tuple-path keys are used when list indexes or tuple-like paths are needed.

SPACE_NETWORK = {
    "r_inh_liq": [0.1, 0.2, 0.3],

    "p_liq_intra_pairs.EE": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.EI": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.IE": [0.02, 0.05, 0.10],
    "p_liq_intra_pairs.II": [0.02, 0.05, 0.10],

    "liq_intra_gain_pairs.EE": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.EI": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.IE": [0.5, 1.0, 2.0],
    "liq_intra_gain_pairs.II": [0.5, 1.0, 2.0],
}

SPACE_NETWORK_TUPLEPATH = {
    ("IN_ROUTE_LAYERS", 0, "p", "E"): [0.05, 0.10, 0.20],
    ("IN_ROUTE_LAYERS", 0, "p", "I"): [0.05, 0.10, 0.20],
    ("IN_ROUTE_LAYERS", 0, "scale", "E"): [10.0, 20.0, 40.0],
    ("IN_ROUTE_LAYERS", 0, "scale", "I"): [10.0, 20.0, 40.0],
}
