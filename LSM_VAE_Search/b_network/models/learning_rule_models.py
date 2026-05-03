"""STDP / SRDP などの学習則を Brian2 用の式として登録する設定。"""

# models/learning_rule_models.py
from .model_utils import register_model


REQUIRED_LEARNING_RULE_KEYS = (
    "eqs",
    "on_pre",
    "on_post",
    "ns_vars",
    "disable_in_test",
)


def learning_rule(
    *,
    eqs: str = "",
    on_pre: str = "",
    on_post: str = "",
    ns_vars: list[str] | None = None,
    namespace: dict | None = None,
    learnable: bool = True,
    disable_in_test=None,
) -> dict:
    return {
        "eqs": eqs,
        "on_pre": on_pre,
        "on_post": on_post,
        "ns_vars": ns_vars or [],
        "namespace": namespace or {},
        "learnable": learnable,
        "disable_in_test": disable_in_test or (lambda ns: ns),
    }


LEARNING_RULES: dict[str, dict] = {}

OFF_RULE = register_model(
    LEARNING_RULES,
    "off",
    learning_rule(learnable=False),
    REQUIRED_LEARNING_RULE_KEYS,
)

STDP_RULE = register_model(
    LEARNING_RULES,
    "STDP",
    learning_rule(
        eqs="""
dApre/dt  = -Apre/tau_plus   : 1 (event-driven)
dApost/dt = -Apost/tau_minus : 1 (event-driven)
""",
        on_pre="""
Apre += A_plus
w = clip(w + Apost, wmin, wmax)
""",
        on_post="""
Apost += A_minus
w = clip(w + Apre, wmin, wmax)
""",
        ns_vars=["tau_plus", "tau_minus", "A_plus", "A_minus", "wmin", "wmax"],
    ),
    REQUIRED_LEARNING_RULE_KEYS,
)

T_STDP_RULE = register_model(
    LEARNING_RULES,
    "T_STDP",
    learning_rule(
        eqs="""
dAplus1/dt  = -Aplus1/tau_plus1   : 1 (event-driven)
dAplus2/dt  = -Aplus2/tau_plus2   : 1 (event-driven)
dAminus1/dt = -Aminus1/tau_minus1 : 1 (event-driven)
dAminus2/dt = -Aminus2/tau_minus2 : 1 (event-driven)
""",
        on_pre="""
Aplus1 += A2_plus
Aplus2 += A3_plus
w = clip(w + (Aminus1 + Aminus2), wmin, wmax)
""",
        on_post="""
Aminus1 += A2_minus
Aminus2 += A3_minus
w = clip(w + (Aplus1 + Aplus2), wmin, wmax)
""",
        ns_vars=[
            "tau_plus1",
            "tau_plus2",
            "tau_minus1",
            "tau_minus2",
            "A2_plus",
            "A3_plus",
            "A2_minus",
            "A3_minus",
            "wmin",
            "wmax",
        ],
    ),
    REQUIRED_LEARNING_RULE_KEYS,
)

SRDP_RULE = register_model(
    LEARNING_RULES,
    "SRDP",
    learning_rule(
        eqs="""
dApre/dt  = -Apre/tau_plus   : 1 (event-driven)
dApost/dt = -Apost/tau_minus : 1 (event-driven)
dMpre/dt  = -Mpre/tau_pre    : 1 (event-driven)
dMpost/dt = -Mpost/tau_post  : 1 (event-driven)
""",
        on_pre="""
Apre += 1.0
Mpre += A_pre
w = clip(w - (A_minus + Mpost) * Apost, wmin, wmax)
""",
        on_post="""
Apost += 1.0
Mpost += A_post
w = clip(w + (A_plus + Mpre) * Apre, wmin, wmax)
""",
        ns_vars=[
            "tau_plus",
            "tau_minus",
            "tau_pre",
            "tau_post",
            "A_plus",
            "A_minus",
            "A_pre",
            "A_post",
            "wmin",
            "wmax",
        ],
    ),
    REQUIRED_LEARNING_RULE_KEYS,
)


# Compatibility alias for older imports that used the module title.
LEARNING_RULE_MODELS = LEARNING_RULES
