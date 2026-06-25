"""リキッド層の内部状態を記録し、素材ごとの npz として保存する処理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from brian2 import ms

from .visualization import safe_stem


def _sample_tag(*, rep: int | None, mat: str, sid: int) -> str:
    rep_prefix = "" if rep is None else f"rep{rep}_"
    return f"{rep_prefix}{safe_stem(mat)}_sid{sid}"


def internal_state_config(run_cfg: dict) -> dict:
    # cfg_run の保存設定を、内部状態保存で使いやすい形にまとめる。
    return {
        "enabled": bool(run_cfg.get("INTERNAL_STATE_ENABLE", False)),
        "source": str(run_cfg.get("INTERNAL_STATE_SOURCE", "spike_bin_mean")),
        "dir_name": str(run_cfg.get("INTERNAL_STATE_DIR", "internal_states")),
        "max_neurons": int(run_cfg.get("INTERNAL_STATE_MAX_NEURONS", 2000)),
        "dt_ms": float(run_cfg.get("INTERNAL_STATE_DT_MS", 1.0)),
        "tau_ms": float(run_cfg.get("INTERNAL_STATE_TAU_MS", 20.0)),
        "bin_ms": float(run_cfg.get("INTERNAL_STATE_BIN_MS", 10.0)),
    }


def total_neurons(groups: Sequence) -> int:
    return sum(int(len(group)) for group in groups)


def can_record_internal_state(groups: Sequence, run_cfg: dict) -> tuple[bool, dict]:
    # ニューロン数が多すぎる場合に、内部状態保存でメモリを使いすぎないよう判定する。
    cfg = internal_state_config(run_cfg)
    if not cfg["enabled"]:
        return False, cfg

    n_total = total_neurons(groups)
    cfg["total_neurons"] = n_total
    if n_total > cfg["max_neurons"]:
        cfg["skip_reason"] = (
            f"total liquid neurons {n_total} exceeds "
            f"INTERNAL_STATE_MAX_NEURONS={cfg['max_neurons']}"
        )
        return False, cfg
    return True, cfg


def _time_grid(duration_ms: float, dt_ms: float) -> np.ndarray:
    n_steps = max(1, int(np.ceil(float(duration_ms) / float(dt_ms))))
    return np.arange(n_steps, dtype=np.float32) * np.float32(dt_ms)


def _monitor_spikes(
    monitor,
    *,
    start_index: int | None = None,
    start_time=None,
    duration_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # SpikeMonitor から「この試行の開始位置以降」に出たスパイクだけを切り出す。
    # start_time が分かる場合は、試行内の相対時刻 0 ms 始まりへ補正する。
    start = 0 if start_index is None else max(0, int(start_index))
    spike_t = monitor.t[start:]
    spike_t_ms = np.asarray(spike_t / ms, dtype=np.float32)
    if spike_t_ms.size:
        if start_time is not None:
            shifted = spike_t_ms - np.float32(float(start_time / ms))
            if duration_ms is not None and float(duration_ms) > 0:
                duration = np.float32(float(duration_ms))
                mask = (shifted >= np.float32(-1e-6)) & (
                    shifted <= duration + np.float32(1e-6)
                )
                if np.any(mask):
                    return shifted[mask], np.asarray(monitor.i[start:], dtype=np.int32)[mask]
            elif float(np.min(shifted)) >= -1e-6:
                spike_t_ms = shifted

        if duration_ms is not None and float(duration_ms) > 0:
            duration = np.float32(float(duration_ms))
            candidates = [spike_t_ms, spike_t_ms - np.float32(float(np.min(spike_t_ms)))]

            def score(values: np.ndarray) -> tuple[int, float]:
                mask = (values >= np.float32(-1e-6)) & (values <= duration + np.float32(1e-6))
                if not np.any(mask):
                    return 0, float("inf")
                return int(np.count_nonzero(mask)), float(abs(np.min(values[mask])))

            spike_t_ms = max(candidates, key=score)
    return (
        spike_t_ms,
        np.asarray(monitor.i[start:], dtype=np.int32),
    )


def _monitor_times_ms(
    monitor,
    *,
    start_index: int | None = None,
    start_time=None,
) -> np.ndarray:
    start = 0 if start_index is None else max(0, int(start_index))
    t_ms = np.asarray(monitor.t[start:] / ms, dtype=np.float32)
    if start_time is not None and t_ms.size:
        shifted = t_ms - np.float32(float(start_time / ms))
        if float(np.min(shifted)) >= -1e-6:
            t_ms = shifted
    return t_ms


def _filtered_spike_state(
    group,
    monitor,
    t_ms: np.ndarray,
    *,
    dt_ms: float,
    tau_ms: float,
    start_index: int | None = None,
    start_time=None,
) -> dict[str, np.ndarray]:
    n_neurons = int(len(group))
    n_steps = int(t_ms.shape[0])
    spike_bins = np.zeros((n_neurons, n_steps), dtype=np.float32)

    spike_t_ms, spike_i = _monitor_spikes(
        monitor,
        start_index=start_index,
        start_time=start_time,
    )
    if spike_t_ms.size:
        bins = np.floor(spike_t_ms / np.float32(dt_ms)).astype(np.int32)
        valid = (bins >= 0) & (bins < n_steps)
        np.add.at(spike_bins, (spike_i[valid], bins[valid]), np.float32(1.0 / tau_ms))

    decay = np.float32(np.exp(-float(dt_ms) / float(tau_ms)))
    x_state = np.zeros((n_neurons, n_steps), dtype=np.float32)
    for step in range(n_steps):
        if step == 0:
            x_state[:, step] = spike_bins[:, step]
        else:
            x_state[:, step] = x_state[:, step - 1] * decay + spike_bins[:, step]

    arrays = {
        "t_ms": t_ms,
        "x_state": x_state,
        "neuron_index": np.arange(n_neurons, dtype=np.int32),
        "typ": np.asarray(group.typ, dtype=np.int32),
    }
    for axis_name in ("x", "y", "z"):
        if hasattr(group, axis_name):
            arrays[axis_name] = np.asarray(getattr(group, axis_name), dtype=np.float32)
    return arrays


def _group_scalar(group, name: str, default: float = 0.0) -> float:
    if not hasattr(group, name):
        return float(default)
    value = getattr(group, name)
    try:
        return float(value[0])
    except Exception:
        return float(value)


def capture_voltage_state_snapshots(
    groups: Sequence,
    voltage_monitors: Sequence,
    *,
    voltage_start_indices: Sequence[int] | None = None,
    start_time=None,
    source: str = "voltage_delta",
) -> list[dict]:
    # 膜電位 v を時系列で読み、内部状態として保存できる snapshot に変換する。
    snapshots = []
    source_key = str(source).lower()
    for layer_index, (group, monitor) in enumerate(zip(groups, voltage_monitors), start=1):
        start_index = (
            voltage_start_indices[layer_index - 1]
            if voltage_start_indices is not None and layer_index - 1 < len(voltage_start_indices)
            else None
        )
        start = 0 if start_index is None else max(0, int(start_index))
        t_ms = np.array(
            _monitor_times_ms(monitor, start_index=start, start_time=start_time),
            dtype=np.float32,
            copy=True,
        )
        v_state = np.array(monitor.v[:, start:], dtype=np.float32, copy=True)
        if t_ms.size and v_state.shape[1] != t_ms.shape[0]:
            n_time = min(v_state.shape[1], t_ms.shape[0])
            v_state = v_state[:, :n_time]
            t_ms = t_ms[:n_time]

        baseline = _group_scalar(group, "v_reset", default=-65.0)
        if source_key in {"voltage_delta", "v_delta", "delta_v"}:
            x_state = v_state - np.float32(baseline)
        elif source_key in {"voltage", "v", "membrane_voltage"}:
            x_state = v_state
        else:
            raise ValueError(f"Unknown voltage internal-state source: {source}")

        snapshot = {
            "layer_index": int(layer_index),
            "n_neurons": int(len(group)),
            "t_ms": t_ms,
            "x_state": x_state,
            "source": source_key,
            "v_reset": np.float32(baseline),
            "typ": np.asarray(group.typ, dtype=np.int32),
        }
        for axis_name in ("x", "y", "z"):
            if hasattr(group, axis_name):
                snapshot[axis_name] = np.asarray(getattr(group, axis_name), dtype=np.float32)
        snapshots.append(snapshot)
    return snapshots


def capture_internal_state_snapshots(
    groups: Sequence,
    spike_monitors: Sequence,
    *,
    spike_start_indices: Sequence[int] | None = None,
    start_time=None,
    duration_ms: float | None = None,
    source: str = "spike_bin_mean",
) -> list[dict]:
    # スパイク時刻を layer ごとに保持する。後段で 10 ms bin 平均などへ変換する。
    snapshots = []
    source_key = str(source).lower()
    for layer_index, (group, monitor) in enumerate(zip(groups, spike_monitors), start=1):
        start_index = (
            spike_start_indices[layer_index - 1]
            if spike_start_indices is not None and layer_index - 1 < len(spike_start_indices)
            else None
        )
        spike_t_ms, spike_i = _monitor_spikes(
            monitor,
            start_index=start_index,
            start_time=start_time,
            duration_ms=duration_ms,
        )
        snapshot = {
            "layer_index": int(layer_index),
            "n_neurons": int(len(group)),
            "spike_t_ms": spike_t_ms,
            "spike_i": spike_i,
            "source": source_key,
            "typ": np.asarray(group.typ, dtype=np.int32),
        }
        for axis_name in ("x", "y", "z"):
            if hasattr(group, axis_name):
                snapshot[axis_name] = np.asarray(getattr(group, axis_name), dtype=np.float32)
        snapshots.append(snapshot)
    return snapshots


def _filtered_spike_state_from_snapshot(
    snapshot: dict,
    t_ms: np.ndarray,
    *,
    dt_ms: float,
    tau_ms: float,
) -> dict[str, np.ndarray]:
    n_neurons = int(snapshot["n_neurons"])
    n_steps = int(t_ms.shape[0])
    spike_bins = np.zeros((n_neurons, n_steps), dtype=np.float32)

    spike_t_ms = np.asarray(snapshot.get("spike_t_ms", ()), dtype=np.float32)
    spike_i = np.asarray(snapshot.get("spike_i", ()), dtype=np.int32)
    if spike_t_ms.size:
        bins = np.floor(spike_t_ms / np.float32(dt_ms)).astype(np.int32)
        valid = (bins >= 0) & (bins < n_steps)
        np.add.at(spike_bins, (spike_i[valid], bins[valid]), np.float32(1.0 / tau_ms))

    decay = np.float32(np.exp(-float(dt_ms) / float(tau_ms)))
    x_state = np.zeros((n_neurons, n_steps), dtype=np.float32)
    for step in range(n_steps):
        if step == 0:
            x_state[:, step] = spike_bins[:, step]
        else:
            x_state[:, step] = x_state[:, step - 1] * decay + spike_bins[:, step]

    arrays = {
        "t_ms": t_ms,
        "x_state": x_state,
        "neuron_index": np.arange(n_neurons, dtype=np.int32),
        "typ": np.asarray(snapshot["typ"], dtype=np.int32),
    }
    for axis_name in ("x", "y", "z"):
        if axis_name in snapshot:
            arrays[axis_name] = np.asarray(snapshot[axis_name], dtype=np.float32)
    return arrays


def _binned_spike_state_from_snapshot(
    snapshot: dict,
    *,
    duration_ms: float,
    bin_ms: float,
    source: str,
) -> dict[str, np.ndarray]:
    # スパイク時刻を指定幅の bin に集計し、count/rate/mean の内部状態に変換する。
    n_neurons = int(snapshot["n_neurons"])
    bin_width = max(float(bin_ms), 1e-9)
    n_bins = max(1, int(np.ceil(float(duration_ms) / bin_width)))
    t_ms = np.arange(n_bins, dtype=np.float32) * np.float32(bin_width)
    counts = np.zeros((n_neurons, n_bins), dtype=np.float32)

    spike_t_ms = np.asarray(snapshot.get("spike_t_ms", ()), dtype=np.float32)
    spike_i = np.asarray(snapshot.get("spike_i", ()), dtype=np.int32)
    if spike_t_ms.size:
        bins = np.floor(spike_t_ms / np.float32(bin_width)).astype(np.int32)
        valid = (
            (bins >= 0)
            & (bins < n_bins)
            & (spike_i >= 0)
            & (spike_i < n_neurons)
        )
        np.add.at(counts, (spike_i[valid], bins[valid]), np.float32(1.0))

    source_key = str(source).lower()
    if source_key in {"spike_bin_count", "spike_count_bin", "binned_spike_count"}:
        x_state = counts
        unit = "spikes/bin"
    elif source_key in {"spike_bin_rate", "spike_rate_bin", "binned_spike_rate"}:
        x_state = counts / np.float32(bin_width / 1000.0)
        unit = "Hz"
    else:
        x_state = counts / np.float32(bin_width)
        unit = "spikes/ms"

    arrays = {
        "t_ms": t_ms,
        "x_state": x_state.astype(np.float32, copy=False),
        "neuron_index": np.arange(n_neurons, dtype=np.int32),
        "typ": np.asarray(snapshot["typ"], dtype=np.int32),
        "bin_ms": np.asarray([bin_width], dtype=np.float32),
        "unit": np.asarray([unit]),
    }
    for axis_name in ("x", "y", "z"):
        if axis_name in snapshot:
            arrays[axis_name] = np.asarray(snapshot[axis_name], dtype=np.float32)
    return arrays


def _state_arrays_from_snapshot(snapshot: dict) -> dict[str, np.ndarray]:
    x_state = np.asarray(snapshot["x_state"], dtype=np.float32)
    t_ms = np.asarray(snapshot["t_ms"], dtype=np.float32)
    if x_state.ndim != 2:
        raise ValueError(f"snapshot x_state must be 2D, got shape={x_state.shape}")
    if t_ms.ndim != 1:
        raise ValueError(f"snapshot t_ms must be 1D, got shape={t_ms.shape}")
    if x_state.shape[1] != t_ms.shape[0]:
        n_time = min(x_state.shape[1], t_ms.shape[0])
        x_state = x_state[:, :n_time]
        t_ms = t_ms[:n_time]

    arrays = {
        "t_ms": t_ms,
        "x_state": x_state,
        "neuron_index": np.arange(x_state.shape[0], dtype=np.int32),
        "typ": np.asarray(snapshot["typ"], dtype=np.int32),
    }
    for axis_name in ("x", "y", "z"):
        if axis_name in snapshot:
            arrays[axis_name] = np.asarray(snapshot[axis_name], dtype=np.float32)
    return arrays


def save_internal_states_from_snapshots(
    out_dir: Path,
    *,
    rep: int | None,
    mat: str,
    sid: int,
    layer_snapshots: Sequence[dict],
    duration_ms: float,
    config: dict,
) -> Path | None:
    # layer ごとの内部状態ファイルと、全 layer を結合した all.npz を保存する。
    # VAE、PCA、分離指標は主に all.npz を入力として使う。
    if not config.get("enabled", False):
        return None
    if not layer_snapshots:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _sample_tag(rep=rep, mat=mat, sid=sid)
    source = str(config.get("source", "spike_bin_mean")).lower()
    spike_filter_sources = {"spike_filter", "spike", "filtered_spike"}
    spike_bin_sources = {
        "spike_bin_mean",
        "spike_mean_bin",
        "binned_spike_mean",
        "spike_bin_count",
        "spike_count_bin",
        "binned_spike_count",
        "spike_bin_rate",
        "spike_rate_bin",
        "binned_spike_rate",
    }
    if "x_state" in layer_snapshots[0]:
        t_ms = np.asarray(layer_snapshots[0].get("t_ms", ()), dtype=np.float32)
        if t_ms.size == 0:
            t_ms = _time_grid(duration_ms, config["dt_ms"])
    elif source in spike_bin_sources:
        t_ms = _time_grid(duration_ms, config.get("bin_ms", 10.0))
    else:
        t_ms = _time_grid(duration_ms, config["dt_ms"])
    manifest_rows = []
    all_x = []
    all_typ = []
    all_layer = []
    all_index_in_layer = []
    all_xpos = []
    all_ypos = []
    all_zpos = []

    for snapshot in layer_snapshots:
        layer_index = int(snapshot["layer_index"])
        n_neurons = int(snapshot["n_neurons"])
        if "x_state" in snapshot:
            arrays = _state_arrays_from_snapshot(snapshot)
            if arrays["t_ms"].shape[0] != t_ms.shape[0]:
                n_time = min(arrays["t_ms"].shape[0], t_ms.shape[0])
                arrays["t_ms"] = arrays["t_ms"][:n_time]
                arrays["x_state"] = arrays["x_state"][:, :n_time]
                t_ms = t_ms[:n_time]
        else:
            if source in spike_bin_sources:
                arrays = _binned_spike_state_from_snapshot(
                    snapshot,
                    duration_ms=duration_ms,
                    bin_ms=config.get("bin_ms", 10.0),
                    source=source,
                )
            elif source in spike_filter_sources:
                arrays = _filtered_spike_state_from_snapshot(
                    snapshot,
                    t_ms,
                    dt_ms=config["dt_ms"],
                    tau_ms=config["tau_ms"],
                )
            else:
                raise ValueError(f"Unknown spike-based internal-state source: {source}")
        fp = out_dir / f"{tag}_liquid_L{layer_index}_internal_state.npz"
        np.savez_compressed(fp, **arrays)
        spike_times = np.asarray(snapshot.get("spike_t_ms", ()), dtype=np.float32)
        manifest_rows.append(
            {
                "layer": layer_index,
                "file": fp.name,
                "neurons": n_neurons,
                "time_steps": int(arrays["x_state"].shape[1]),
                "source": str(snapshot.get("source", source)),
                "spike_count": int(spike_times.size),
                "spike_time_min_ms": float(np.min(spike_times)) if spike_times.size else None,
                "spike_time_max_ms": float(np.max(spike_times)) if spike_times.size else None,
                "state_min": float(np.min(arrays["x_state"])) if arrays["x_state"].size else 0.0,
                "state_max": float(np.max(arrays["x_state"])) if arrays["x_state"].size else 0.0,
                "state_nonzero": int(np.count_nonzero(arrays["x_state"])),
            }
        )

        all_x.append(arrays["x_state"])
        all_typ.append(arrays["typ"])
        all_layer.append(np.full(n_neurons, layer_index, dtype=np.int32))
        all_index_in_layer.append(arrays["neuron_index"])
        all_xpos.append(arrays.get("x", np.full(n_neurons, np.nan, dtype=np.float32)))
        all_ypos.append(arrays.get("y", np.full(n_neurons, np.nan, dtype=np.float32)))
        all_zpos.append(arrays.get("z", np.full(n_neurons, np.nan, dtype=np.float32)))

    total_neuron_count = int(sum(int(snapshot["n_neurons"]) for snapshot in layer_snapshots))
    combined_fp = out_dir / f"{tag}_liquid_internal_state_all.npz"
    combined_payload = {
        "t_ms": t_ms,
        "x_state": np.concatenate(all_x, axis=0),
        "neuron_index": np.arange(total_neuron_count, dtype=np.int32),
        "layer_index": np.concatenate(all_layer),
        "index_in_layer": np.concatenate(all_index_in_layer),
        "typ": np.concatenate(all_typ),
        "x": np.concatenate(all_xpos),
        "y": np.concatenate(all_ypos),
        "z": np.concatenate(all_zpos),
        "source": np.asarray([source]),
    }
    if source in spike_bin_sources:
        if source in {"spike_bin_count", "spike_count_bin", "binned_spike_count"}:
            unit = "spikes/bin"
        elif source in {"spike_bin_rate", "spike_rate_bin", "binned_spike_rate"}:
            unit = "Hz"
        else:
            unit = "spikes/ms"
        combined_payload["bin_ms"] = np.asarray([config.get("bin_ms", 10.0)], dtype=np.float32)
        combined_payload["unit"] = np.asarray([unit])
    np.savez_compressed(combined_fp, **combined_payload)

    manifest = {
        "mat": mat,
        "sid": sid,
        "source": source,
        "dt_ms": config["dt_ms"],
        "tau_ms": config["tau_ms"],
        "bin_ms": config.get("bin_ms", 10.0),
        "definition": (
            "voltage_delta: x_i(t)=v_i(t)-v_reset; voltage: x_i(t)=v_i(t); "
            "spike_filter: x_i(t)=(h*s_i)(t), h(t)=(1/tau_s)*exp(-t/tau_s)*H(t); "
            "spike_bin_mean: x_i[k]=spike_count_i[k]/bin_ms"
        ),
        "total_neurons": total_neuron_count,
        "combined_file": combined_fp.name,
        "layers": manifest_rows,
    }
    if rep is not None:
        manifest["rep"] = rep
    manifest_fp = out_dir / f"{tag}_internal_state_manifest.json"
    manifest_fp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_fp


def save_internal_states(
    out_dir: Path,
    *,
    rep: int | None,
    mat: str,
    sid: int,
    groups: Sequence,
    spike_monitors: Sequence,
    duration_ms: float,
    config: dict,
    spike_start_indices: Sequence[int] | None = None,
    start_time=None,
) -> Path | None:
    # 旧形式の呼び出しでも、snapshot 化してから同じ保存処理へ流す互換入口。
    layer_snapshots = capture_internal_state_snapshots(
        groups,
        spike_monitors,
        spike_start_indices=spike_start_indices,
        start_time=start_time,
        duration_ms=duration_ms,
        source=str(config.get("source", "spike_bin_mean")),
    )
    return save_internal_states_from_snapshots(
        out_dir,
        rep=rep,
        mat=mat,
        sid=sid,
        layer_snapshots=layer_snapshots,
        duration_ms=duration_ms,
        config=config,
    )
