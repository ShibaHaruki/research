"""二重指数シナプスなど、電流応答モデルを登録する設定。"""

# models/synapse_models.py
from brian2 import Hz

from .model_utils import register_model


REQUIRED_SYNAPSE_MODEL_KEYS = (
    "eqs",
    "post_eqs",
    "synapse",
)

REQUIRED_SYNAPSE_KEYS = (
    "eqs",
    "on_pre",
    "w_attr",
)


def synapse_path(
    *,
    eqs: str,
    on_pre: str,
    w_attr: str = "w",
    ns_vars: list[str] | None = None,
    namespace: dict | None = None,
) -> dict:
    return {
        "eqs": eqs,
        "on_pre": on_pre,
        "w_attr": w_attr,
        "ns_vars": ns_vars or [],
        "namespace": namespace or {},
    }


def current_synapse_model(
    *,
    post_eqs: str,
    synapse: dict,
    namespace: dict | None = None,
) -> dict:
    register_model({}, "synapse", synapse, REQUIRED_SYNAPSE_KEYS)
    return {
        "eqs": post_eqs,
        "post_eqs": post_eqs,
        "synapse": synapse,
        "namespace": namespace or {},
    }


DOUBLE_EXP_POST_EQS = """
tau_r : second (shared)
tau_d : second (shared)

dI_syn/dt = -I_syn / tau_d + H_syn : 1
dH_syn/dt = -H_syn / tau_r : Hz
"""

DOUBLE_EXP_SYNAPSE = synapse_path(
    eqs="w : 1",
    on_pre="H_syn_post += (w / (tau_r_post * tau_d_post)) / Hz",
    namespace={"Hz": Hz},
)

DOUBLE_EXP_STP_EQS = """
w : 1
dx_stp/dt = (1.0 - x_stp) / tau_stp_rec : 1 (clock-driven)
du_stp/dt = (U_stp - u_stp) / tau_stp_facil : 1 (clock-driven)
"""

DOUBLE_EXP_STP_NS_VARS = ["tau_stp_rec", "tau_stp_facil", "U_stp"]

DOUBLE_EXP_STP_SYNAPSE = synapse_path(
    eqs=DOUBLE_EXP_STP_EQS,
    on_pre="""
u_stp += U_stp * (1.0 - u_stp)
H_syn_post += (w * u_stp * x_stp / (tau_r_post * tau_d_post)) / Hz
x_stp = clip(x_stp * (1.0 - u_stp), 0.0, 1.0)
""",
    ns_vars=DOUBLE_EXP_STP_NS_VARS,
    namespace={"Hz": Hz},
)

SYNAPSE_MODELS: dict[str, dict] = {}

DOUBLE_EXP_MODEL = register_model(
    SYNAPSE_MODELS,
    "double_exp",
    current_synapse_model(
        post_eqs=DOUBLE_EXP_POST_EQS,
        synapse=DOUBLE_EXP_SYNAPSE,
    ),
    REQUIRED_SYNAPSE_MODEL_KEYS,
)

DOUBLE_EXP_STP_MODEL = register_model(
    SYNAPSE_MODELS,
    "double_exp_stp",
    current_synapse_model(
        post_eqs=DOUBLE_EXP_POST_EQS,
        synapse=DOUBLE_EXP_STP_SYNAPSE,
    ),
    REQUIRED_SYNAPSE_MODEL_KEYS,
)
