"""シナプスモデルごとの時定数などの固定設定。"""

# FIXED/cfg_neuron_models.py

CFG_SYNAPSE_MODELS = {
    
    "double_exp": {
        "tau_r": 2.0,
        "tau_d": 30.0,
    },

    "double_exp_stp": {
        "tau_r": 2.0,
        "tau_d": 30.0,
        "tau_stp_rec": 800.0,
        "tau_stp_facil": 50.0,
        "U_stp": 0.2,
    },

}
