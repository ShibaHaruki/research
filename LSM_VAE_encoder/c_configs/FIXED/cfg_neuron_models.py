"""ニューロンモデルごとの膜時定数や不応期などの固定設定。"""

# FIXED/cfg_neuron_models.py

CFG_NEURON_MODELS = {
    "LIF": {
        "tau_exc": 10,
        "tau_inh": 10,
        "ref_exc": 2,
        "ref_inh": 2,
        "bias": -65.0,
        "v_thr": -40,
        "v_reset": -65,
    },
}
