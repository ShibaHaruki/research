"""Input filter registry and sensor-to-filter mapping."""

from a_tactile.input_filters import calc_RI, calc_SI, calc_USI


FILTER_FUNCS = {
    "RI": calc_RI,
    "USI": calc_USI,
    "SI": calc_SI,
}

INPUT_FILTER_MAP = {
    0: ["RI", "USI", "SI"],
    1: ["RI", "USI", "SI"],
    2: ["RI", "USI", "SI"],
}
