"""入力フィルタ名と実際の計算関数を対応付ける設定。"""

from a_tactile.input_filters import (
    FILTER_GAIN,
    SENSOR_GAIN,
    RI,
    SI,
    USI,
    calc_meissner,
    calc_merkel,
)


FILTER_FUNCS = {
    "RI": RI,
    "SI": SI,
    "USI": USI,
    "merkel": calc_merkel,
    "meissner": calc_meissner,
}

INPUT_FILTER_MAP = {
    0: ["merkel", "meissner", "RI", "SI", "USI"],
    1: ["merkel", "meissner", "RI", "SI", "USI"],
    2: ["merkel", "meissner", "RI", "SI", "USI"],
}

OPT_FILTER_GAIN = {
    "RI": 1.0,
    "SI": 1.0,
    "USI": 1.0,
    "merkel": 1.0,
    "meissner": 1.0,
}
