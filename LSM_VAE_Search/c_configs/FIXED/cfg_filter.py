"""入力フィルタ名と実際の計算関数を対応付ける設定。"""

from a_tactile.input_filters import calc_meissner, calc_merkel


FILTER_FUNCS = {
    "merkel": calc_merkel,
    "meissner": calc_meissner,
}

INPUT_FILTER_MAP = {
    0: ["merkel","meissner"],
}
