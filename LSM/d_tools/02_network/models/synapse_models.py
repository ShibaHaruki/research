# models/synapse_models.py
from brian2 import Hz

# ★SYNAPSE_MODELSという名前は維持
SYNAPSE_MODELS = {
    "double_exp": {
                "post_eqs": """
                    tau_r : second (shared)
                    tau_d : second (shared)

                    dI_exc/dt = -I_exc / tau_d + H_exc : 1
                    dH_exc/dt = -H_exc / tau_r : Hz

                    dI_inh/dt = -I_inh / tau_d + H_inh : 1
                    dH_inh/dt = -H_inh / tau_r : Hz
                """,

        "liq_exc": dict(
            eqs="w : 1",
            on_pre="H_exc_post += (w / (tau_r_post * tau_d_post)) / Hz",
            w_attr="w",
        ),
        "liq_inh": dict(
            eqs="w : 1",
            on_pre="H_inh_post += (w / (tau_r_post * tau_d_post)) / Hz",
            w_attr="w",
        ),
    }
}
