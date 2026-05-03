"""二重指数シナプスなど、電流応答モデルを登録する設定。"""

# models/synapse_models.py
from brian2 import Hz

from .model_utils import register_model


REQUIRED_SYNAPSE_MODEL_KEYS = (
    "eqs",
    "post_eqs",
    "liq_exc",
    "liq_inh",
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
    namespace: dict | None = None,
) -> dict:
    return {
        "eqs": eqs,
        "on_pre": on_pre,
        "w_attr": w_attr,
        "namespace": namespace or {},
    }


def current_synapse_model(
    *,
    post_eqs: str,
    exc: dict,
    inh: dict,
    namespace: dict | None = None,
) -> dict:
    register_model({}, "exc", exc, REQUIRED_SYNAPSE_KEYS)
    register_model({}, "inh", inh, REQUIRED_SYNAPSE_KEYS)
    return {
        "eqs": post_eqs,
        "post_eqs": post_eqs,
        "liq_exc": exc,
        "liq_inh": inh,
        "exc": exc,
        "inh": inh,
        "out_exc": exc,
        "out_inh": inh,
        "namespace": namespace or {},
    }


DOUBLE_EXP_POST_EQS = """
tau_r : second (shared)
tau_d : second (shared)

dI_exc/dt = -I_exc / tau_d + H_exc : 1
dH_exc/dt = -H_exc / tau_r : Hz

dI_inh/dt = -I_inh / tau_d + H_inh : 1
dH_inh/dt = -H_inh / tau_r : Hz
"""

DOUBLE_EXP_EXC_SYNAPSE = synapse_path(
    eqs="w : 1",
    on_pre="H_exc_post += (w / (tau_r_post * tau_d_post)) / Hz",
    namespace={"Hz": Hz},
)

DOUBLE_EXP_INH_SYNAPSE = synapse_path(
    eqs="w : 1",
    on_pre="H_inh_post += (w / (tau_r_post * tau_d_post)) / Hz",
    namespace={"Hz": Hz},
)


SYNAPSE_MODELS: dict[str, dict] = {}

DOUBLE_EXP_MODEL = register_model(
    SYNAPSE_MODELS,
    "double_exp",
    current_synapse_model(
        post_eqs=DOUBLE_EXP_POST_EQS,
        exc=DOUBLE_EXP_EXC_SYNAPSE,
        inh=DOUBLE_EXP_INH_SYNAPSE,
    ),
    REQUIRED_SYNAPSE_MODEL_KEYS,
)
