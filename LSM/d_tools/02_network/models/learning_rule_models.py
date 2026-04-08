# models/learning_rules.py
import numpy as np

LEARNING_RULES = {
    "off": dict(
        eqs="",
        on_pre="",
        on_post="",
        ns_vars=[],
        disable_in_test=lambda ns: ns,
    ),

    "STDP": dict(
        eqs="""
        dApre/dt  = -Apre/tau_plus   : 1 (event-driven)
        dApost/dt = -Apost/tau_minus : 1 (event-driven)
        """,
        on_pre=f"""
        Apre += A_plus
        w = clip(w +  Apost, wmin, wmax)
        """,
        on_post=f"""
        Apost += A_minus
        w  = clip(w + Apre,  wmin, wmax)
        """,
        ns_vars=["tau_plus", "tau_minus", "A_plus", "A_minus", "wmin", "wmax"],
        disable_in_test=lambda ns: ns,
    ),

    "T_STDP": dict(
        eqs="""
        dAplus1/dt  = -Aplus1/tau_plus1    : 1 (event-driven)
        dAplus2/dt  = -Aplus2/tau_plus2    : 1 (event-driven)
        dAminus1/dt = -Aminus1/tau_minus1  : 1 (event-driven)
        dAminus2/dt = -Aminus2/tau_minus2  : 1 (event-driven)
        """,
        on_pre=f"""
        Aplus1 += A2_plus
        Aplus2 += A3_plus
        w = clip(w +  (Aminus1 + Aminus2), wmin, wmax)
        """,
        on_post=f"""
        Aminus1 += A2_minus
        Aminus2 += A3_minus
        w  = clip(w +  (Aplus1 + Aplus2), wmin, wmax)
        """,
        ns_vars=[
            "tau_plus1", "tau_plus2", "tau_minus1", "tau_minus2",
            "A2_plus", "A3_plus", "A2_minus", "A3_minus",
            "wmin", "wmax"
        ],
        disable_in_test=lambda ns: ns,
    ),

    "SRDP": dict(
        eqs="""
        dApre/dt   = -Apre/tau_plus   : 1 (event-driven)
        dApost/dt  = -Apost/tau_minus : 1 (event-driven)
        dMpre/dt   = -Mpre/tau_pre    : 1 (event-driven)
        dMpost/dt  = -Mpost/tau_post  : 1 (event-driven)
        """,
        on_pre=f"""
        Apre += 1.0
        Mpre += A_pre
        w = clip(w - (A_minus + Mpost) * Apost, wmin, wmax)
        """,
        on_post=f"""
        Apost += 1.0
        Mpost += A_post
        w = clip(w + (A_plus + Mpre) * Apre, wmin, wmax)
        """,
        ns_vars=[
            "tau_plus", "tau_minus", "tau_pre", "tau_post",
            "A_plus", "A_minus", "A_pre", "A_post",
            "wmin", "wmax"
        ],
        disable_in_test=lambda ns: ns,
    ),
}
