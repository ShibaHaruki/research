"""センサーCSVの取得、入力経路の対応付け、フィルタ済み入力電流の作成。"""

from __future__ import annotations
import glob
from collections import OrderedDict
import numpy as np
import pandas as pd
from a_experiment.paths import TACTILE_DATA_ROOT, TACTILE_DATA_DIR_NAME
from a_experiment.experiment_sets import trial_duration_ms

# 各実行方式が共通して参照する入力データの親フォルダ。
TACTILE_DATA_PATH = TACTILE_DATA_ROOT.resolve()

# 各実行プロセスで最大256 MiBまで保持し、並列repでのメモリ増加を制限する。
_FILTERED_INPUT_CACHE_MAX_BYTES = 256 * 1024 * 1024
_filtered_input_cache = OrderedDict()
_filtered_input_cache_bytes = 0


def clear_filtered_input_cache():
    """処理済み入力をメモリから解放する。実験データや結果ファイルは変更しない。"""
    global _filtered_input_cache_bytes
    _filtered_input_cache.clear()
    _filtered_input_cache_bytes = 0


def _filtered_input_key(source, cfg):
    """CSVの更新情報と、波形に影響する入力設定・フィルタ関数を識別する。"""
    from pathlib import Path

    path = Path(source).resolve()
    stat = path.stat()
    common = cfg["common"]
    rows = input_rows(cfg["input_filter_map"])
    # build_input_currentと同じ順序・既定値で、実際に使用する設定を取り出す。
    routes = tuple(
        (
            sensor,
            name,
            cfg["filter_funcs"][name],
            float(cfg["sensor_gain"].get(sensor, cfg["sensor_gain"].get(str(sensor), 1.0))),
            float(cfg["filter_gain"].get(name, 1.0)),
            float(cfg["opt_filter_gain"].get(name, 1.0)),
        )
        for sensor, name in rows
    )
    return (
        str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
        common["SLICE_START"], common["SLICE_END"],
        int(common["NUM_SENSOR"]), float(common["dt_s"]), routes,
    )


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
    """同じ入力設定の波形を再利用し、時刻×入力経路の独立した配列を返す。"""
    global _filtered_input_cache_bytes
    trial_duration_ms(cfg["common"])
    source = load_tactile_data(material, sid)
    key = _filtered_input_key(source, cfg)
    if key in _filtered_input_cache:
        _filtered_input_cache.move_to_end(key)
        # 呼び出し側が書き換えても、次の試行や他の実行条件へ影響させない。
        return _filtered_input_cache[key].copy()
    current = _compute_filtered_trial(source, material, sid, cfg)
    if current.nbytes <= _FILTERED_INPUT_CACHE_MAX_BYTES:
        while _filtered_input_cache_bytes + current.nbytes > _FILTERED_INPUT_CACHE_MAX_BYTES:
            _, previous = _filtered_input_cache.popitem(last=False)
            _filtered_input_cache_bytes -= previous.nbytes
        stored = current.copy()
        stored.setflags(write=False)
        _filtered_input_cache[key] = stored
        _filtered_input_cache_bytes += stored.nbytes
    return current


def _compute_filtered_trial(source, material, sid, cfg):
    """CSVから従来と同じfloat64演算・順序で入力を作る。キャッシュ未登録時のみ使用。"""
    common = cfg["common"]
    raw = pd.read_table(source, header=None).to_numpy().T
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
