"""現在のLIF設定を1候補ずつ定義する。時定数・不応期の単位はms。"""

NEURON_SPACE = {
    "LIF_default": {
        "tau_exc": [10],
        "tau_inh": [10],
        "ref_exc": [2],
        "ref_inh": [2],
        "bias": [-65.0],
        # 個別のE/I閾値が優先されるため、共通v_thrではなく実際に使うキーを指定。
        "v_thr_exc": [-40],
        "v_thr_inh": [-40],
        "v_reset_exc": [-65],
        "v_reset_inh": [-65],
    },
}
