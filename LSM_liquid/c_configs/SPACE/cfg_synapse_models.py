"""シナプス時定数を探索するときの候補値リスト。"""


SYNAPSE_SPACE = {
    "double_exp_default": {
        "tau_r": [1, 2, 4],
        "tau_d": [10, 20, 40],
    },
}
