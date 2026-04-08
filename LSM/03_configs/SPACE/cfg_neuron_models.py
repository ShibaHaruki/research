
NEURON_SPACE = {
    "LIF_default": {
        # ※ここは ms 単位の “数値” を入れておく運用が楽（あとで *ms する）
        "tau_exc": [8, 10, 12],
        "tau_inh": [1.5, 2, 3],
        "ref_exc": [1, 2, 3],
        "ref_inh": [0.5, 1, 2],

        "bias": [0.0],
        "v_thr": [-45, -40, -35],
        "v_reset": [-65],
    },
}