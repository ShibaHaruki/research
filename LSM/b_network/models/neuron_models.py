# models/neuron_models.py
import numpy as np
from brian2 import ms

COMMON_NEURON_MODEL_PARAMETERS = """
I_merkel   : 1
I_meissner : 1
I_in = I_merkel + I_meissner : 1

I_inh : 1
I_exc : 1

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

NEURON_MODELS = {
    "LIF": dict(
        eqs=COMMON_NEURON_MODEL_PARAMETERS + """
        dv/dt = (-v + bias + I_in - I_inh + I_exc) / tau_m : 1 (unless refractory)
        """,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="euler",
        set_shared=lambda g: None,
    ),
}
