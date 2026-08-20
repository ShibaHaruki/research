"""Search-space definitions for learning-rule and output connectivity tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Parameter:
    name: str
    initial: float
    low: float
    high: float
    kind: str = "linear"


COMMON_OUTPUT_PARAMETERS = [
    Parameter("out_p", 0.50, 0.05, 1.00, "linear"),
    Parameter("out_gain", 0.25, 0.01, 10.00, "log10"),
]


RULE_PARAMETERS: dict[str, list[Parameter]] = {
    "off": [],
    "STDP": [
        Parameter("stdp_a_plus", 0.0007, 0.0001, 0.1, "log10"),
        Parameter("stdp_a_minus", 0.0006, 0.0001, 0.1, "log10"),
        Parameter("stdp_tau_pre_ms", 11.7, 1.0, 30.0, "log10"),
        Parameter("stdp_tau_post_ms", 14.0, 1.0, 30.0, "log10"),
    ],
    "SRDP": [
        Parameter("srdp_a_plus", 0.0007, 0.0001, 0.1, "log10"),
        Parameter("srdp_a_minus", 0.0006, 0.0001, 0.1, "log10"),
        Parameter("srdp_tau_plus_ms", 11.7, 1.0, 30.0, "log10"),
        Parameter("srdp_tau_minus_ms", 14.0, 1.0, 30.0, "log10"),
        Parameter("srdp_tau_pre_m_ms", 15.0, 1.0, 30.0, "log10"),
        Parameter("srdp_tau_post_m_ms", 15.0, 1.0, 30.0, "log10"),
        Parameter("srdp_a_pre_m", 0.00005, 0.000001, 0.1, "log10"),
        Parameter("srdp_a_post_m", 0.00005, 0.000001, 0.1, "log10"),
    ],
    "T_STDP": [
        Parameter("tstdp_a2_plus", 0.0007, 0.0001, 0.1, "log10"),
        Parameter("tstdp_a2_minus", 0.0006, 0.0001, 0.1, "log10"),
        Parameter("tstdp_a3_plus", 0.00003, 0.000001, 0.1, "log10"),
        Parameter("tstdp_a3_minus", 0.00003, 0.000001, 0.1, "log10"),
        Parameter("tstdp_tau_s1_ms", 11.7, 1.0, 30.0, "log10"),
        Parameter("tstdp_tau_s2_ms", 14.0, 1.0, 30.0, "log10"),
        Parameter("tstdp_tau_t1_ms", 15.0, 1.0, 30.0, "log10"),
        Parameter("tstdp_tau_t2_ms", 15.0, 1.0, 30.0, "log10"),
    ],
}


def parameters_for(rule: str) -> list[Parameter]:
    if rule not in RULE_PARAMETERS:
        raise KeyError(f"Unknown rule: {rule}. Choose from {sorted(RULE_PARAMETERS)}")
    return [*RULE_PARAMETERS[rule], *COMMON_OUTPUT_PARAMETERS]


def candidate_from_vector(rule: str, vector: list[float]) -> dict[str, Any]:
    specs = parameters_for(rule)
    if len(vector) != len(specs):
        raise ValueError(f"Expected {len(specs)} values, got {len(vector)}")
    return {spec.name: float(value) for spec, value in zip(specs, vector)}
