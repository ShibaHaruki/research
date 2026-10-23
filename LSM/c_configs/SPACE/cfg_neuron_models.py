"""現在のLIF設定で使用する時定数・不応期・閾値の候補。単位はms。"""
NEURON_SPACE = {
    "LIF_default": {
        "tau_exc": [8, 10, 12],
        "tau_inh": [1.5, 2, 3, 10],
        "ref_exc": [1, 2, 3],
        "ref_inh": [0.5, 1, 2],
        "bias": [-65.0],
        # 個別のE/I閾値が優先されるため、共通v_thrではなく実際に使うキーを指定。
        "v_thr_exc": [-45, -40, -35],
        "v_thr_inh": [-45, -40, -35],
        "v_reset_exc": [-65],
        "v_reset_inh": [-65],
    },
}
