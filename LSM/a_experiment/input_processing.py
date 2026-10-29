"""センサーCSVの取得、入力経路の対応付け、フィルタ済み入力電流の作成。"""

from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from a_experiment.paths import TACTILE_DATA_ROOT, TACTILE_DATA_DIR_NAME
from a_experiment.experiment_sets import trial_duration_ms

# 各実行方式が共通して参照する入力データの親フォルダ。
TACTILE_DATA_PATH = TACTILE_DATA_ROOT.resolve()


def input_rows(input_filter_map: dict[int, list[str]]) -> list[tuple[int, str]]:
    """入力配列の行とセンサー・フィルタの対応を、番号順に作る。"""
    channels = sorted(input_filter_map)
    if not channels:
        raise ValueError("input_filter_map is empty.")

    rows: list[tuple[int, str]] = []
    for ch in channels:
        for filter_name in input_filter_map[ch]:
            rows.append((int(ch), str(filter_name)))
    if not rows:
        raise ValueError("input_filter_map must define at least one filter.")
    return rows


def input_layout(input_filter_map: dict[int, list[str]]) -> tuple[list[int], list[str]]:
    """有効な入力経路から、センサー番号とフィルタ名の一覧を取り出す。"""
    rows = input_rows(input_filter_map)
    channels = sorted({ch for ch, _ in rows})
    filters = list(dict.fromkeys(filter_name for _, filter_name in rows))
    return channels, filters


def load_tactile_data(mat: str, sid: int, tactile_data_path=TACTILE_DATA_PATH):
    """素材と試行番号に対応するCSVを1件だけ取得し、欠損・重複は明示的に知らせる。"""
    pattern = tactile_data_path / TACTILE_DATA_DIR_NAME / mat / f"data_{sid}_*.csv"
    matches = sorted(glob.glob(str(pattern)))  # 入力が曖昧な場合は、勝手に1ファイルを選ばない。
    if not matches:
        raise FileNotFoundError(f"No tactile input matched: {pattern}")
    if len(matches) != 1:
        raise ValueError(f"Multiple tactile inputs matched: {pattern}: {matches}")
    return matches[0]


def build_input_current(
    in_data_0,
    t_array,
    dt,
    input_filter_map,
    filter_funcs,
    sensor_gain=None,
    filter_gain=None,
    opt_filter_gain=None,
    include_opt_filter_gain: bool = True,
):
    """各センサーにフィルタとgainを適用し、入力経路×時刻の電流配列を返す。"""
    channels = sorted(input_filter_map)
    nt = in_data_0.shape[1]
    total_filters = sum(len(input_filter_map[ch]) for ch in channels)
    input_current = np.zeros((total_filters, nt))
    sensor_gain = sensor_gain or {}
    filter_gain = filter_gain or {}
    opt_filter_gain = opt_filter_gain or {}

    row = 0
    for ch in channels:
        gain = float(sensor_gain.get(ch, sensor_gain.get(str(ch), 1.0)))
        data_ch = gain * in_data_0[ch, :]
        for filter_name in input_filter_map[ch]:
            gain_filter = float(filter_gain.get(filter_name, 1.0))
            if include_opt_filter_gain:
                gain_filter *= float(opt_filter_gain.get(filter_name, 1.0))
            input_current[row, :] = gain_filter * filter_funcs[filter_name](data_ch, t_array, dt)
            row += 1
    return input_current


def filtered_trial(material, sid, cfg):
    """指定した素材・試行の入力を、時刻×入力経路の配列として返す。"""
    common = cfg["common"]
    raw = pd.read_table(load_tactile_data(material, sid), header=None).to_numpy().T
    trial_duration_ms(common)  # 切り出し範囲と時間刻みを検証する。
    n_steps = common["SLICE_END"] - common["SLICE_START"]
    n_sensors = int(common["NUM_SENSOR"])
    dt_s = float(common["dt_s"])
    signal = raw[:n_sensors, common["SLICE_START"] : common["SLICE_END"]]
    if signal.shape != (n_sensors, n_steps):
        raise ValueError(f"Unexpected signal shape: {material} sid{sid}: {signal.shape}")
    current = build_input_current(
        signal,
        np.arange(n_steps) * dt_s,
        dt_s,
        cfg["input_filter_map"],
        cfg["filter_funcs"],
        cfg["sensor_gain"],
        cfg["filter_gain"],
        cfg["opt_filter_gain"],
    )
    n_channels = sum(len(filters) for filters in cfg["input_filter_map"].values())
    if current.shape != (n_channels, n_steps) or not np.isfinite(current).all():
        raise ValueError(f"Invalid filtered current: {material} sid{sid}")
    return current.T
