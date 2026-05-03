# -*- coding: utf-8 -*-
"""Config for checking one fixed parameter set with a pretrained fixed VAE Encoder.

This file is only for the fixed-encoder check. It is independent from
parameter_values_config.py, which controls VAE pretraining.
"""

FIXED_PARAMETER_TRIAL = {
    "name": "manual_trial",

    # Empty string means: use the newest fixed_encoder_vae under g_tactile_results/fixed_vae_encoder_pretrain.
    "fixed_encoder_dir": "",

    # Number of samples per material used when feeding this fixed parameter set into the fixed VAE Encoder.
    # This is separate from VAE_PRETRAIN["samples_per_material"].
    "encoder_input_samples_per_material": 50,

    # Backward-compatible alias. Keep None unless you intentionally want the old key.
    "samples_per_material": None,

    # None means all materials. Example: ["Al_board", "cork"]
    "materials": None,

    # Brian2 codegen target. "auto" uses cython only when C++ Build Tools are available; otherwise numpy.
    "brian_codegen_target": "auto",

    # Fixed LSM parameter values to test with the fixed VAE Encoder.
    # Missing parameters fall back to the CMA-ES initial x0 values.
    "parameter_values": {
        "merkel_p_E": 0.2,
        "merkel_p_I": 0.2,
        "merkel_scale_E": 0.01,
        "merkel_scale_I": 0.01,
        "meissner_p_E": 0.2,
        "meissner_p_I": 0.2,
        "meissner_scale_E": 0.04,
        "meissner_scale_I": 0.04,
        "p_liq_EE": 0.1,
        "p_liq_EI": 0.1,
        "p_liq_IE": 0.1,
        "p_liq_II": 0.0,
        "gain_liq_EE": 0.25,
        "gain_liq_EI": 0.25,
        "gain_liq_IE": 0.25,
        "gain_liq_II": 0.0,
        "lif_tau_exc": 10.0,
        "lif_tau_inh": 10.0,
        "lif_ref_exc": 2.0,
        "lif_ref_inh": 2.0,
        "lif_bias": -65.0,
        "lif_v_thr": -40.0,
        "lif_v_reset": -65.0,
        "syn_tau_r": 1.0,
        "syn_tau_d": 30.0,
    },
}
