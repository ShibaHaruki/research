"""LIF などのニューロンモデル定義を登録する設定。"""

# models/neuron_models.py
from .model_utils import register_model


REQUIRED_NEURON_KEYS = (
    "eqs",
    "threshold",
    "reset",
    "refractory",
    "method",
    "set_shared",
)


COMMON_NEURON_MODEL_PARAMETERS = """
I_merkel   : 1
I_meissner : 1
I_in = I_merkel + I_meissner : 1

tau_m : second
t_ref : second

bias   : 1 (shared)
v_thr  : 1 (shared)
v_reset: 1 (shared)

x : 1 (constant)
y : 1 (constant)
z : 1 (constant)
typ : integer (constant)
"""


def neuron_model(
    *,
    eqs: str,
    threshold: str,
    reset: str,
    refractory: str,
    method: str = "euler",
    namespace: dict | None = None,
    set_shared=None,
) -> dict:
    return {
        "eqs": eqs,
        "threshold": threshold,
        "reset": reset,
        "refractory": refractory,
        "method": method,
        "namespace": namespace or {},
        "set_shared": set_shared or (lambda g: None),
    }


NEURON_MODELS: dict[str, dict] = {}

LIF_MODEL = register_model(
    NEURON_MODELS,
    "LIF",
    neuron_model(
        eqs=COMMON_NEURON_MODEL_PARAMETERS + """
dv/dt = (-v + bias + I_in - I_inh + I_exc) / tau_m : 1 (unless refractory)
""",
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
    ),
    REQUIRED_NEURON_KEYS,
)


# Compatibility alias for older imports.
NEURON_MODEL_PARAMETERS = COMMON_NEURON_MODEL_PARAMETERS
