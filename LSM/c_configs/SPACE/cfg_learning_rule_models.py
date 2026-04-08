# SPACE/cfg_learning_rule_models.py
# 学習則 preset ごとの探索空間（辞書のみ / 値は list）

LEARNING_RULE_SPACE = {
    "off_default": {
        # learning_rule="off" は探索するパラメータ無しでもOK
    },

    "STDP_default": {
        "tau_plus":  [8, 11.7, 15],
        "tau_minus": [10, 14.0, 18],
        "A_plus":    [3e-4, 7e-4, 1e-3],
        "A_minus":   [-3e-4, -6e-4, -1e-3],
        "wmin":      [0.0],
        "wmax":      [1.0],
    },

    "T_STDP_default": {
        "tau_plus1":  [8, 11.7, 15],
        "tau_plus2":  [10, 15, 20],
        "tau_minus1": [10, 14.0, 18],
        "tau_minus2": [10, 15, 20],
        "A2_plus":    [3e-4, 7e-4, 1e-3],
        "A3_plus":    [1e-8, 5e-8, 1e-7],

        # 符号も探索したいなら ± を入れておく（最終的には学習則の更新式と整合させる）
        "A2_minus":   [-6e-4, 6e-4],
        "A3_minus":   [-5e-8, 5e-8],

        "wmin":       [0.0],
        "wmax":       [1.0],
    },

    "SRDP_default": {
        "tau_plus":  [8, 11.7, 15],
        "tau_minus": [10, 14.0, 18],
        "tau_pre":   [10, 15, 20],
        "tau_post":  [10, 15, 20],
        "A_plus":    [3e-4, 7e-4, 1e-3],
        "A_minus":   [-3e-4, -6e-4, -1e-3],
        "A_pre":     [1e-5, 5e-5, 1e-4],
        "A_post":    [1e-5, 5e-5, 1e-4],
        "wmin":      [0.0],
        "wmax":      [1.0],
    },
}