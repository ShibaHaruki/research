# FIXED/cfg_neuron_models.py

NEURON_PRESETS = {
    "LIF_default": {
        "neuron_model": "LIF",
        "tau_exc": 10,
        "tau_inh": 2,
        "ref_exc": 2,
        "ref_inh": 1,
        "bias": 0.0,
        "v_thr": -40,
        "v_reset": -65,
    },
}