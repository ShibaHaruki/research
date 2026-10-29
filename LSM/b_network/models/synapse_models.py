"""二重指数関数型（double exponential）シナプスの電流応答モデル。"""

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
