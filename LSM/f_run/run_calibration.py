"""Liquid・Readoutのgain校正、状態記録、固定スパイクの相互相関解析。

python f_run/run_calibration.py plan
python f_run/run_calibration.py run
python f_run/run_calibration.py verify --out-dir ...
python f_run/run_calibration.py output --liquid-dir ... --out-dir ...
python f_run/run_calibration.py correlation --self-test
python f_run/run_calibration.py correlation

引数なし、または従来の解析オプションだけを指定した場合も相互相関解析を実行する。
"""

from __future__ import annotations

import argparse
import ast
import time
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "f_run"))
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_key, "1")

import numpy as np
from d_tools.recording_io import read_record, record_exists
from d_tools.activity import summarize_ei_activity
from a_experiment.input_processing import filtered_trial
from a_experiment.paths import CALIBRATION_ROOT, CORRELATION_ROOT, TACTILE_DATA_DIR_NAME

# リキッド層と出力層の校正条件はここで管理する。通常実行のFIXED設定は書き換えない。
SETTINGS = {
    "neurons": 100,
    "excitatory": 80,
    "inhibitory": 20,
    "input_probability": 0.05,
    "probabilities": {"EE": 0.11, "EI": 0.10, "IE": 0.37, "II": 0.10},
    "input_material": "Japanese_paper",
    "recurrent_material": "Japanese_paper",
    "materials": [
        "aluminum_board",
        "outer_pigskin",
        "back_pigskin",
        "cork",
        "denim",
        "rubber_board",
        "Japanese_paper",
        "wood_board",
    ],
    "trials_per_material": 324,
    "duration_s": 0.5,
    "dt_s": 0.0001,
    "bin_ms": 25.0,
    "tau_membrane_ms": 10.0,
    "refractory_ms": 2.0,
    "threshold_mV": -40.0,
    "reset_mV": -65.0,
    "bias_mV": -65.0,
    "syn_tau_r_ms": 2.0,
    "syn_tau_d_ms": 30.0,
    "EE_EI_coarse_step": 0.01,
    "EE_EI_fine_step": 0.001,
    "EE_EI_max": 1.0,
    "IE_grid": [
        0.0,
        0.00001,
        0.00002,
        0.00005,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
    ],
    "IE_refinements": 2,
    "debug_trial_id": 203,
    "state_policy": "reset_each_trial",
    "output_probability": 0.11,
    "output_gain_step": 0.0001,
    "output_initial_gain": 0.8,
    "output_max_gain": 2.0,
    "output_rate_tolerance_hz": 0.1,
    "output_criterion": "match_mean_liquid_E_and_output_rates_across_all_2592_trials",
}
REFERENCE = ROOT / "c_configs/FIXED/calibration_reference.npz"


def save(path, value):
    from d_tools.recording_io import save_record

    return save_record(path, value)


def load_reference():
    fixed = {
        "neurons": 100,
        "excitatory": 80,
        "inhibitory": 20,
        "input_probability": 0.05,
        "probabilities": {"EE": 0.11, "EI": 0.10, "IE": 0.37, "II": 0.10},
        "trials_per_material": 324,
        "duration_s": 0.5,
        "dt_s": 0.0001,
        "bin_ms": 25.0,
        "tau_membrane_ms": 10.0,
        "refractory_ms": 2.0,
        "threshold_mV": -40.0,
        "reset_mV": -65.0,
        "bias_mV": -65.0,
        "syn_tau_r_ms": 2.0,
        "syn_tau_d_ms": 30.0,
        "state_policy": "reset_each_trial",
    }
    if any(SETTINGS[key] != value for key, value in fixed.items()):
        raise ValueError(
            "Fixed topology/simulator conditions changed; update and validate the reference and calibration simulation first."
        )
    if len(SETTINGS["materials"]) != 8:
        raise ValueError(
            "Reference splits require the eight reference materials in their original order."
        )
    metadata = read_record(REFERENCE.with_suffix(".json"))
    if SETTINGS["materials"] != metadata["materials"]:
        raise ValueError("Material order differs from reference holdout splits")
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Reference topology / splits checksum mismatch")
    with np.load(REFERENCE) as data:
        ref = {k: data[k].copy() for k in data.files}
    wi, wr = ref["input_unit_weights"], ref["recurrent_unit_weights"]
    assert wi.shape == (6, 100) and wr.shape == (100, 100)
    assert np.all(wi[:, 80:] == 0) and np.all(wr[80:, 80:] == 0)
    for block in [wr[:80, :80], wr[:80, 80:], wr[80:, :80]]:
        assert np.all(np.count_nonzero(block, axis=0) > 0)
    assert np.all(np.diag(wr) == 0)
    return ref


def config(gains, *, include_output=False):
    """Apply this experiment's explicit conditions to an independent config copy."""
    from f_run.common import build_cfg

    cfg = deepcopy(build_cfg())
    common = cfg["common"]
    if (
        common["NUM_SENSOR"] != 3
        or not np.isclose(common["dt_s"], 0.0001)
        or common["SLICE_END"] - common["SLICE_START"] != 5000
        or cfg["input_filter_map"] != {0: ["RI", "SI"], 1: ["RI", "SI"], 2: ["RI", "SI"]}
    ):
        raise ValueError(
            "This calibrated simulator requires 3 RI/SI sensors, 500 ms and dt=0.1 ms."
        )

    net = cfg["network"]
    # リキッド校正では出力を無効にし、出力校正では明示的に有効にする。
    net["output_enabled"] = bool(include_output)
    if include_output:
        cfg["models"]["LEARNING_RULE_MODEL"] = "off"
        sizes = net["N_out"]
        sizes = [int(sizes)] if np.isscalar(sizes) else list(sizes)
        if not sizes or any(int(n) != n or n < 1 for n in sizes):
            raise ValueError("Positive output layer sizes required")
        net["N_out"] = sizes
        net["learning_rule_liq_to_out"] = ["off"] * len(sizes)
        net["p_liq_to_out_pairs"] = [
            dict(EE=SETTINGS["output_probability"], EI=0.0, IE=0.0, II=0.0) for _ in sizes
        ]
        net["gain_liq_to_out_pairs"] = [
            dict(EE=float(gains["output_gain"]), EI=0.0, IE=0.0, II=0.0) for _ in sizes
        ]
    net["N_liq"] = [SETTINGS["neurons"]]
    net["r_inh_liq"] = SETTINGS["inhibitory"] / SETTINGS["neurons"]
    # Set all six RI/SI routes explicitly; do not let the shared overlay replace them.
    if isinstance(net.get("SHARED_IN_ROUTE"), dict):
        net["SHARED_IN_ROUTE"]["enabled"] = False
    for (_, filter_name), route in net["IN_ROUTE"].items():
        active = str(filter_name).lower() in ("ri", "si")
        for layer in route.get("layers", {}).values():
            layer["p"] = {"E": SETTINGS["input_probability"] if active else 0.0, "I": 0.0}
            if active:
                layer["scale"] = {"E": float(gains["input_gain"]), "I": 0.0}
    net["p_liq_intra_pairs"] = dict(SETTINGS["probabilities"])
    net["liq_intra_gain_pairs"] = {
        pair: gains.get(pair + "_gain", 0.0) for pair in ["EE", "EI", "IE", "II"]
    }
    cfg["opt_filter_gain"].update(RI=1.0, SI=1.0)
    cfg["neuron_models"]["LIF"].update(
        tau_exc=SETTINGS["tau_membrane_ms"],
        tau_inh=SETTINGS["tau_membrane_ms"],
        ref_exc=SETTINGS["refractory_ms"],
        ref_inh=SETTINGS["refractory_ms"],
        bias=SETTINGS["bias_mV"],
        v_reset=SETTINGS["reset_mV"],
        v_thr=SETTINGS["threshold_mV"],
    )
    cfg["models"]["SYNAPSE_MODEL"] = "double_exp"
    cfg["synapse_models"]["double_exp"].update(
        tau_r=SETTINGS["syn_tau_r_ms"], tau_d=SETTINGS["syn_tau_d_ms"]
    )
    return cfg


def provenance():
    # Filter or input-window changes must not silently reuse cached currents/results.
    files = [
        "a_experiment/input_filters.py",
        "a_experiment/input_processing.py",
        "a_experiment/experiment_sets.py",
        "c_configs/FIXED/cfg_filter.py",
        "f_run/run_calibration.py",
        "f_run/common.py",
        "b_network/build_network.py",
        "b_network/monitor.py",
        "d_tools/activity.py",
        "d_tools/plotting.py",
        "d_tools/debug.py",
        "d_tools/weight_export.py",
        "c_configs/FIXED/calibration_reference.npz",
        "c_configs/FIXED/cfg_network.py",
        "c_configs/FIXED/cfg_network_stdp.py",
        "c_configs/FIXED/cfg_network_off.py",
        "c_configs/FIXED/cfg_neuron_models.py",
        "b_network/weight_initialization.py",
        "b_network/models/neuron_models.py",
        "b_network/models/synapse_models.py",
        "d_tools/recording_io.py",
    ]
    cfg = config({"input_gain": 1.0})
    return {
        "settings": SETTINGS,
        "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
        "data_root": str(
            __import__(
                "a_experiment.input_processing", fromlist=["TACTILE_DATA_PATH"]
            ).TACTILE_DATA_PATH
        ),
        "data_subdir": TACTILE_DATA_DIR_NAME,
        "base_seed": int(cfg["common"]["BASE_SEED"]),
    }


def currents(material, out):
    """Build/cache 324 filtered trials; source signals are not copied to results."""
    path = out / "filtered_inputs" / f"{material}.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="c")
        if cached.shape != (324, 5000, 6) or cached.dtype != np.float64:
            raise ValueError(f"Unexpected input-cache shape/dtype: {path}")
        return cached
    cfg = config({"input_gain": 1.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    x = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(324, 5000, 6))
    for sid in range(1, 325):
        x[sid - 1] = filtered_trial(material, sid, cfg)
        if sid % 25 == 0 or sid == 324:
            print(f"[inputs] {material}: {sid}/324", flush=True)
    x.flush()
    del x
    temporary.replace(path)
    return np.load(path, mmap_mode="c")


def weights(ref, gains):
    wi = np.ascontiguousarray(ref["input_unit_weights"] * gains["input_gain"])
    wr = ref["recurrent_unit_weights"].copy()
    for pair, block in [
        ("EE", (slice(0, 80), slice(0, 80))),
        ("EI", (slice(0, 80), slice(80, 100))),
        ("IE", (slice(80, 100), slice(0, 80))),
    ]:
        wr[block] *= gains.get(pair + "_gain", 0.0)
    return wi, np.ascontiguousarray(wr)


def _simulate_states(inputs, wi, wr, wo=None, gain=0.0, *, trace_callback=None):
    """NumPyで複数試行をまとめて計算する。各試行は初期状態から開始。

    Eulerの更新順、RI/SIの加算順、発火元番号順のシナプス加算を維持する。
    ニューロンの式はBrian2による代表試行の検証と照合する。
    """
    if trace_callback is not None and not callable(trace_callback):
        raise ValueError("trace_callback must be callable or None")
    x = np.asarray(inputs, dtype=np.float64)
    wi = np.asarray(wi, dtype=np.float64)
    wr = np.asarray(wr, dtype=np.float64)
    wo = np.zeros((100, 0)) if wo is None else np.asarray(wo, dtype=np.float64)
    if x.ndim != 3 or not len(x) or x.shape[1:] != (5000, 6):
        raise ValueError("Expected nonempty trial x 5000 steps x 6 input channels")
    if wi.shape != (6, 100) or wr.shape != (100, 100) or wo.ndim != 2 or wo.shape[0] != 100:
        raise ValueError("Invalid input, recurrent, or output weight dimensions")
    if (
        not all(np.isfinite(a).all() for a in [x, wi, wr, wo])
        or not np.isfinite(gain)
        or gain < 0
        or np.any(wo < 0)
        or np.any(wo[80:] != 0)
    ):
        raise ValueError("Require finite inputs/weights, nonnegative output gain and E-only output")
    trials = len(x)
    outputs = wo.shape[1]
    liquid_bins = np.zeros((trials, 100, 20), np.int32)
    bins = np.zeros((trials, outputs, 20), np.int32)
    peaks = np.zeros((trials, outputs))
    v = np.full((trials, 100), -65.0)
    last = np.full((trials, 100), -100000, np.int32)
    ov = np.full((trials, outputs), -65.0)
    olast = np.full((trials, outputs), -100000, np.int32)
    syn = np.zeros((trials, 100 + outputs))
    h = np.zeros_like(syn)
    probe = np.zeros((trials, outputs))
    # CSR形式で発火元ごとの接続先を持つ。add.atは同じ宛先への加算順を保つ。
    joined = np.concatenate([wr, wo], axis=1)
    pre, posts = np.nonzero(joined)
    increments = joined[pre, posts] / (0.002 * 0.030)
    degrees = np.bincount(pre, minlength=100)
    pointers = np.r_[0, np.cumsum(degrees)]
    for t in range(5000):
        old = syn
        syn = old + 0.0001 * (-old / 0.030 + h)
        h = h + 0.0001 * (-h / 0.002)
        # CやBrian2と同じくRI3チャネル、SI3チャネルを別々に加算する。
        a = x[:, t, 0, None] * wi[0]
        b = x[:, t, 1, None] * wi[1]
        for channel in (1, 2):
            a += x[:, t, 2 * channel, None] * wi[2 * channel]
            b += x[:, t, 2 * channel + 1, None] * wi[2 * channel + 1]
        drive = a + b
        active = t - last > 20
        v = np.where(active, v + 0.0001 * ((-v - 65.0 + drive + old[:, :100]) / 0.010), v)
        fired = active & (v >= -40.0)
        if outputs:
            old_output = old[:, 100:]
            probe = probe + 0.0001 * ((-probe + old_output) / 0.010)
            np.maximum(peaks, probe, out=peaks)
            out_active = t - olast > 20
            ov = np.where(out_active, ov + 0.0001 * ((-ov - 65.0 + gain * old_output) / 0.010), ov)
            out_fired = out_active & (ov >= -40.0)
            ov[out_fired] = -65.0
            olast[out_fired] = t
            bins[:, :, t // 250] += out_fired
        v[fired] = -65.0
        last[fired] = t
        liquid_bins[:, :, t // 250] += fired
        rows, neurons = np.nonzero(fired)
        sizes = degrees[neurons]
        number = int(sizes.sum())
        if number:
            starts = pointers[neurons]
            offsets = np.cumsum(sizes) - sizes
            edges = np.repeat(starts, sizes) + np.arange(number) - np.repeat(offsets, sizes)
            np.add.at(h, (np.repeat(rows, sizes), posts[edges]), increments[edges])
        if trace_callback is not None:
            # 引数は参照渡し。記録側は必要な試行だけコピーし、状態を書き換えない。
            trace_callback(t, fired, out_fired if outputs else None, v, ov)
    return bins, peaks, liquid_bins


def simulate(x, ref, gains):
    wi, wr = weights(ref, gains)
    return _simulate_states(x, wi, wr)[2]


def simulate_output(inputs, wi, wr, wo, gain):
    """リキッド・出力の発火回数と、単位gainの発火前最大応答を返す。"""
    return _simulate_states(inputs, wi, wr, wo, gain)


def input_gain(x, ref, out):
    """③ First-spike boundary is linear before any reset; verify adjacent integers."""
    from scipy.signal import lfilter

    ids = np.flatnonzero(np.any(ref["input_unit_weights"] != 0, axis=0))
    threshold = 0.0
    critical = None
    for i, trial in enumerate(x):
        response = lfilter(
            [0.0001 / 0.010],
            [1, -(1 - 0.0001 / 0.010)],
            (trial @ ref["input_unit_weights"][:, ids]).T,
            axis=1,
        )
        peaks = response.max(1)
        if np.any(peaks <= 0):
            raise ValueError("Some connected neurons have no finite positive first-spike threshold")
        required = 25.0 / peaks
        j = int(np.argmax(required))
        if required[j] > threshold:
            threshold = float(required[j])
            critical = {"sid": i + 1, "neuron_id": int(ids[j])}
    gain = int(np.ceil(threshold))
    counts = simulate(x, ref, {"input_gain": gain})
    if not np.all(counts.sum(2)[:, ids] > 0):
        raise AssertionError("Predicted first-spike gain failed full-trial verification")
    lower = simulate(x[critical["sid"] - 1 : critical["sid"]], ref, {"input_gain": gain - 1})
    assert np.any(lower.sum(2)[:, ids] == 0), "Lower integer unexpectedly passed"
    np.savez_compressed(out / "input_features.npz", counts_25ms=counts, connected_neuron_ids=ids)
    save(
        out / "input_summary.npz",
        {
            "input_gain": gain,
            "continuous_boundary": threshold,
            "critical": critical,
            "connected_neurons": ids.tolist(),
            "all_trials_all_connected_fired": True,
            "minimum_scope": "integer gains, fixed topology, fresh states",
            **summarize_ei_activity(
                counts, n_excitatory=SETTINGS["excitatory"], bin_ms=SETTINGS["bin_ms"]
            ),
        },
    )
    return gain


def recurrent_gains(x, ref, gains, out):
    """⑤⑦⑨ Sequential gain selection without assuming recurrent monotonicity."""
    import pandas as pd

    history = []

    def evaluate(stage, candidate):
        bins = simulate(x, ref, candidate)
        row = {
            "stage": stage,
            **candidate,
            **summarize_ei_activity(
                bins, n_excitatory=SETTINGS["excitatory"], bin_ms=SETTINGS["bin_ms"]
            ),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "search_history.csv", index=False)
        print(
            f'[{stage}] {candidate}: allE={row["all_E_trials"]}, '
            f'allI={row["all_I_trials"]}, EHz={row["mean_E_rate_hz"]:.3f}',
            flush=True,
        )
        return row, bins

    for stage, target in [("EE", "all_E_trials"), ("EI", "all_I_trials")]:
        selected = None
        previous = 0.0
        # Every candidate is evaluated over all 324 trials.
        # Retain the historical coarse .01 + local .001 refinement strategy.
        for tick in range(1, round(SETTINGS["EE_EI_max"] / SETTINGS["EE_EI_coarse_step"]) + 1):
            coarse = round(tick * SETTINGS["EE_EI_coarse_step"], 9)
            row, _ = evaluate(stage, {**gains, stage + "_gain": coarse})
            if row[target] == 324:
                step = SETTINGS["EE_EI_fine_step"]
                for fine_tick in range(round(previous / step) + 1, round(coarse / step) + 1):
                    candidate = {**gains, stage + "_gain": round(fine_tick * step, 9)}
                    result, bins = evaluate(stage, candidate)
                    if result[target] == 324:
                        selected = candidate
                        np.savez_compressed(out / f"{stage}_features.npz", counts_25ms=bins)
                        save(out / f"{stage}_summary.npz", result)
                        break
                if selected is not None:
                    break
            previous = coarse
        if selected is None:
            raise RuntimeError(f"No passing {stage} gain in configured search range")
        gains = selected
    candidates = []
    for gain in SETTINGS["IE_grid"]:
        row, _ = evaluate("IE", {**gains, "IE_gain": gain})
        candidates.append(row)

    def best_candidate():
        feasible = [r for r in candidates if not r["never_active_E"] and not r["never_active_I"]]
        if not feasible:
            raise RuntimeError(
                "No feasible IE gain: no candidate activates all neurons across 324 trials"
            )
        return min(feasible, key=lambda r: r["mean_E_rate_hz"])

    for _ in range(SETTINGS["IE_refinements"]):
        best = best_candidate()
        tested = sorted({r["IE_gain"] for r in candidates})
        pos = tested.index(best["IE_gain"])
        lo, hi = tested[max(0, pos - 1)], tested[min(len(tested) - 1, pos + 1)]
        for gain in np.linspace(lo, hi, 9):
            gain = round(float(gain), 9)
            if gain not in tested:
                row, _ = evaluate("IE", {**gains, "IE_gain": gain})
                candidates.append(row)
    gains["IE_gain"] = best_candidate()["IE_gain"]
    row, bins = evaluate("IE", gains)
    np.savez_compressed(out / "IE_features.npz", counts_25ms=bins)
    save(out / "IE_summary.npz", row)
    save(out / "selected_parameters.json", gains)
    return gains


def verify_brian(x, ref, gains, out, *, debug_title=None, expected_counts=None):
    """Independent exact-count check of LIF/synaptic equations and topology."""
    from brian2 import start_scope, defaultclock, ms, seed
    from f_run.common import build_network_cfg, configure_brian_runtime
    from b_network.build_network import make_liquid_network
    from d_tools.weight_export import save_liquid_weight_matrices

    cfg = config(gains)
    nc = build_network_cfg(cfg)
    start_scope()
    configure_brian_runtime()
    defaultclock.dt = 0.1 * ms
    base_seed = int(cfg["common"]["BASE_SEED"])
    np.random.seed(base_seed)
    seed(base_seed)
    objects = make_liquid_network(
        nc, 6, np.random.default_rng(base_seed), cfg["run"], common_cfg=cfg["common"]
    )
    out.mkdir(parents=True, exist_ok=True)
    save_liquid_weight_matrices(out / "weights", objects)
    wi, wr = weights(ref, gains)
    assert np.allclose(np.load(out / "weights/w_in1_liq1.npy"), wi, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.load(out / "weights/w_liq1_intra.npy"), wr, atol=1e-12, rtol=1e-12)
    from d_tools.debug import execute_trial, save_debug_layer

    recorded = execute_trial(
        objects,
        x[0],
        dt_ms=0.1,
        duration_ms=500.0,
        bin_ms=25.0,
        record_output=False,
        voltage=debug_title is not None,
    )
    data = recorded["liquid"][0]
    actual = data["counts"]
    expected = simulate(x[:1], ref, gains)[0]
    assert np.array_equal(actual, expected), "Calibration / Brian2 spike-count mismatch"
    if expected_counts is not None and not np.array_equal(actual, expected_counts):
        raise ValueError("Saved states differ from the representative debug trial")
    if debug_title is not None:
        save_debug_layer(
            data,
            out,
            title=debug_title,
            n_excitatory=SETTINGS["excitatory"],
            trial_id=SETTINGS["debug_trial_id"],
        )
    result = {"exact_25ms_counts_match": True, "gains": gains, "n_trials_checked": 1}
    save(out / "verification.npz", result)
    print("[verified] exact spike counts and weights match Brian2", flush=True)
    return result


def make_output_network(gains):
    """固定リキッド＋Eから確率接続した出力を、学習なし・初期状態で構築する。"""
    from brian2 import start_scope, defaultclock, ms, seed
    from f_run.common import build_network_cfg, configure_brian_runtime
    from b_network.build_network import make_liquid_network

    cfg = config(gains, include_output=True)
    start_scope()
    configure_brian_runtime()
    defaultclock.dt = 0.1 * ms
    base_seed = int(cfg["common"]["BASE_SEED"])
    seed(base_seed)
    np.random.seed(base_seed)
    nc = build_network_cfg(cfg)
    objects = make_liquid_network(
        nc, 6, np.random.default_rng(base_seed), cfg["run"], common_cfg=cfg["common"]
    )
    # 校正計算の固定LIF/double-exp条件と実ネットワークを照合する。
    for group in objects["G_out"]:
        for name, expected in [("tau_m", 10.0), ("t_ref", 2.0), ("tau_r", 2.0), ("tau_d", 30.0)]:
            if not np.allclose(np.asarray(getattr(group, name) / ms), expected):
                raise ValueError(f"Output {name} differs from calibrated simulation")
        for name, expected in [("v_thr", -40.0), ("v_reset", -65.0), ("bias", -65.0)]:
            if not np.allclose(np.asarray(getattr(group, name)), expected):
                raise ValueError(f"Output {name} differs from calibrated simulation")
    return cfg, objects


def output_unit_weights(objects, *, require_positive=True):
    """gain=1で構築した出力の重みを、100液体ニューロン×全出力へ並べる。"""
    from d_tools.weight_export import _dense_w_matrix

    sizes = [len(group) for group in objects["G_out"]]
    offsets = np.cumsum([0, *sizes])
    matrix = np.zeros((100, sum(sizes)))
    for meta in objects["output_connection_meta"]:
        if meta["liquid_index"] != 0 or meta["pre_offset"] != 0:
            raise AssertionError("Output calibration requires liquid E only")
        syn = meta["synapses"]
        index = meta["output_index"]
        indegree = np.bincount(np.asarray(syn.j[:], int), minlength=sizes[index])
        if not np.all(indegree > 0):
            raise ValueError(
                "Some output neurons have no E input; inspect fixed topology/seed before calibrating"
            )
        matrix[:, offsets[index] : offsets[index + 1]] += _dense_w_matrix(syn, 100, sizes[index])
    if (
        (require_positive and not np.all(np.any(matrix[:80] > 0, axis=0)))
        or np.any(matrix < 0)
        or np.any(matrix[80:] != 0)
    ):
        raise AssertionError("Each output must have positive E input; I-to-output absent")
    return np.ascontiguousarray(matrix), sizes


def _search_output_gain(evaluate, history, *, step, maximum):
    """発火率差の符号を使って探索範囲を狭め、評価済み候補の最良gainを選ぶ。"""
    low = 0
    evaluate(low)
    max_tick = int(np.floor(maximum / step))
    high = min(max_tick, max(1, int(np.ceil(SETTINGS["output_initial_gain"] / step))))
    while evaluate(high)["mean_difference_hz"] > 0:
        if high == max_tick:
            raise RuntimeError(
                "Output rate remains below liquid E at output_max_gain; increase search limit"
            )
        low = high
        high = min(max_tick, high * 2)
    # 差が正ならgainを上げ、負なら下げる。調整幅を半分ずつ狭める。
    while high - low > 1:
        middle = (low + high) // 2
        if evaluate(middle)["mean_difference_hz"] > 0:
            low = middle
        else:
            high = middle
    # 平均差の絶対値が最も小さい評価済み候補を採用する（同値なら小さいgain）。
    best = min(history, key=lambda tick: (history[tick]["absolute_mean_difference_hz"], tick))
    return best, low, high


def _save_output_calibration_states(folder, counts, liquid, sizes):
    """校正後のLiquid・出力状態を、試行×時間ビン×ニューロンの配列で保存する。"""
    for name, bins, types, layer_sizes in [
        ("liquid", liquid, np.array(["E"] * 80 + ["I"] * 20), [100]),
        ("output", counts, np.array(["E"] * sum(sizes)), sizes),
    ]:
        destination = folder / name
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "features.partial.npz"
        np.savez_compressed(
            temporary,
            counts_25ms=bins,
            states_hz=bins.transpose(0, 2, 1) / 0.025,
            trial_ids=np.arange(1, 325),
            bin_ms=25.0,
            neuron_ids=np.arange(bins.shape[1]),
            neuron_types=types,
            layer_sizes=np.asarray(layer_sizes),
            time_bin_edges_ms=np.arange(21) * 25.0,
        )
        temporary.replace(destination / "features.npz")


def _verify_output_calibration_trial(
    material, inputs, selected_gains, counts, liquid, unit, gain, folder, out, *, save_weights
):
    """代表試行をBrian2で再計算し、発火数・重み・単位gainの一致を確認する。"""
    from d_tools.debug import execute_trial, save_network_debug
    from d_tools.weight_export import save_liquid_weight_matrices

    sid = SETTINGS["debug_trial_id"]
    _, network = make_output_network(selected_gains)
    initial = [np.asarray(s.w[:]).copy() for s in network["S_all"]]
    recorded = execute_trial(
        network, inputs[material][sid - 1], dt_ms=0.1, duration_ms=500.0, voltage=True
    )
    actual = np.concatenate([data["counts"] for data in recorded["output"]])
    assert np.array_equal(actual, counts[sid - 1]), "Brian2/output calibration count mismatch"
    assert np.array_equal(recorded["liquid"][0]["counts"], liquid[sid - 1])
    assert all(np.array_equal(s.w[:], w) for s, w in zip(network["S_all"], initial))
    actual_w, _ = output_unit_weights(network)
    assert np.allclose(actual_w, unit * gain, rtol=1e-12, atol=1e-12)
    save_network_debug(
        recorded, network, folder / f"debug_sid{sid}", material=material, trial_id=sid
    )
    if save_weights:
        save_liquid_weight_matrices(out / "weights", network)
    return sid


def _output_calibration_summary(history, best, low, high, unit, sizes, step, checks, records):
    """採用したgain、発火率差、探索境界、検証結果を記録用にまとめる。"""
    result = {
        **history[best],
        "output_probability": SETTINGS["output_probability"],
        "output_layer_sizes": sizes,
        "actual_output_connections": int(np.count_nonzero(unit)),
        "actual_indegree_per_output": np.count_nonzero(unit[:80], axis=0).tolist(),
        "all_outputs_have_input_probability": float(
            (1 - (1 - SETTINGS["output_probability"]) ** 80) ** sum(sizes)
        ),
        "criterion": SETTINGS["output_criterion"],
        "selection_metric": "absolute_mean_difference_hz",
        "difference_definition": "mean_liquid_E_rate - mean_output_rate",
        "positive_difference_action": "increase output gain",
        "negative_difference_action": "decrease output gain",
        "gain_step": step,
        "rate_tolerance_hz": SETTINGS["output_rate_tolerance_hz"],
        "within_tolerance": history[best]["absolute_mean_difference_hz"]
        <= SETTINGS["output_rate_tolerance_hz"],
        "search_scope": "sign-guided bracket refinement on configured grid; one common gain for all trials",
        "bracket": [history[low], history[high]],
        "state_policy": "reset_each_trial",
        "learning": False,
        "I_to_output": False,
        "state_axes": ["trial", "time_bin", "neuron"],
        "bin_ms": 25.0,
        "verification": checks,
        "materials": records,
    }
    return result


def _make_output_rate_evaluator(
    inputs,
    liquid_counts,
    liquid_rates,
    *,
    input_weights,
    recurrent_weights,
    output_weights,
    target_rate_hz,
    gain_step,
    out,
):
    """gainごとの発火率差を計算する関数を作り、評価履歴と素材別の値を保持する。"""
    import pandas as pd

    materials = list(liquid_rates)
    history = {}
    rate_cache = {}

    def evaluate(tick):
        if tick in history:
            return history[tick]
        gain = round(tick * gain_step, 12)
        output_rates = {}
        for material in materials:
            if tick == 0:
                output_rates[material] = np.zeros(324)
            else:
                counts, _, liquid = simulate_output(
                    inputs[material], input_weights, recurrent_weights, output_weights, gain
                )
                assert np.array_equal(
                    liquid, liquid_counts[material]
                ), "Liquid counts changed during output search"
                output_rates[material] = counts.sum(axis=2).mean(axis=1) / 0.5
        difference = np.concatenate([liquid_rates[m] - output_rates[m] for m in materials])
        output_mean = float(np.concatenate(list(output_rates.values())).mean())
        row = {
            "output_gain": gain,
            "liquid_E_rate_hz": target_rate_hz,
            "output_rate_hz": output_mean,
            "mean_difference_hz": float(difference.mean()),
            "absolute_mean_difference_hz": float(abs(difference.mean())),
            "trial_rmse_hz": float(np.sqrt(np.mean(difference**2))),
            "trial_mae_hz": float(np.mean(np.abs(difference))),
            "n_trials": 2592,
        }
        history[tick] = row
        rate_cache[tick] = output_rates
        pd.DataFrame(list(history.values())).to_csv(out / "search_history.csv", index=False)
        print(
            f"[output rate] gain={gain:.4f}, liquid E={target_rate_hz:.4f} Hz, "
            f'output={output_mean:.4f} Hz, E-output={row["mean_difference_hz"]:+.4f} Hz',
            flush=True,
        )
        return row

    return evaluate, history, rate_cache


def calibrate_output(ref, gains, out, *, input_dir):
    """全2592試行の平均差（リキッドE−出力）を符号で挟み込み、gainを調整・全状態保存する。"""
    from brian2 import ms
    from d_tools.recording_io import encode
    from d_tools.parameter_recording import snapshot
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    step = float(SETTINGS["output_gain_step"])
    maximum = float(SETTINGS["output_max_gain"])
    if not np.isfinite(step) or step <= 0 or not np.isfinite(maximum) or maximum < step:
        raise ValueError("Require finite positive gain resolution and search maximum")
    cfg, objects = make_output_network({**gains, "output_gain": 1.0})
    unit, sizes = output_unit_weights(objects)
    objects["net"].run(0 * ms, namespace={"input_ta": objects["G_in"].namespace["input_ta"]})
    wi, wr = weights(ref, gains)
    materials = SETTINGS["materials"]
    inputs = {material: currents(material, Path(input_dir)) for material in materials}
    liquid_counts = {}
    liquid_rates = {}
    for material in materials:
        liquid_counts[material] = simulate(inputs[material], ref, gains)
        liquid_rates[material] = liquid_counts[material][:, :80].sum(axis=2).mean(axis=1) / 0.5
    target = float(np.concatenate(list(liquid_rates.values())).mean())
    if target <= 0:
        raise ValueError("Cannot match output rate to an inactive liquid population")
    evaluate, history, rate_cache = _make_output_rate_evaluator(
        inputs,
        liquid_counts,
        liquid_rates,
        input_weights=wi,
        recurrent_weights=wr,
        output_weights=unit,
        target_rate_hz=target,
        gain_step=step,
        out=out,
    )
    best, low, high = _search_output_gain(evaluate, history, step=step, maximum=maximum)
    gain = history[best]["output_gain"]
    selected_gains = {**gains, "output_gain": gain}
    comparisons = []
    records = []
    checks = []
    for material in materials:
        counts, _, liquid = simulate_output(inputs[material], wi, wr, unit, gain)
        assert np.array_equal(liquid, liquid_counts[material])
        assert np.array_equal(counts.sum(axis=2).mean(axis=1) / 0.5, rate_cache[best][material])
        folder = out / "all_materials" / material
        _save_output_calibration_states(folder, counts, liquid, sizes)
        for index in range(324):
            comparisons.append(
                {
                    "material": material,
                    "trial_id": index + 1,
                    "liquid_E_rate_hz": float(liquid_rates[material][index]),
                    "output_rate_hz": float(rate_cache[best][material][index]),
                    "difference_hz": float(
                        liquid_rates[material][index] - rate_cache[best][material][index]
                    ),
                }
            )
        sid = _verify_output_calibration_trial(
            material,
            inputs,
            selected_gains,
            counts,
            liquid,
            unit,
            gain,
            folder,
            out,
            save_weights=material == materials[0],
        )
        checks.append(
            {
                "material": material,
                "trial_id": sid,
                "exact_output_counts_match": True,
                "exact_liquid_counts_match": True,
                "weights_unchanged": True,
            }
        )
        records.append(
            {
                "material": material,
                "trials": 324,
                "liquid_states_shape": [324, 20, 100],
                "output_states_shape": [324, 20, sum(sizes)],
                "never_active_output": int(np.count_nonzero(counts.sum(axis=(0, 2)) == 0)),
                "liquid_E_rate_hz": float(liquid_rates[material].mean()),
                "output_rate_hz": float(rate_cache[best][material].mean()),
            }
        )
        print(f"[output recorded] {material}: 324 trials and debug sid{sid}", flush=True)
    pd.DataFrame(comparisons).to_csv(out / "trial_rate_comparison.csv", index=False)
    pd.DataFrame(records).to_csv(out / "material_rate_comparison.csv", index=False)
    result = _output_calibration_summary(
        history, best, low, high, unit, sizes, step, checks, records
    )
    save(out / "output_summary.npz", result)
    save(
        out / "config_snapshot.json", encode(snapshot(config(selected_gains, include_output=True)))
    )
    return selected_gains


def record_material_states(ref, gains, out):
    """⑩ 全素材の全試行を記録する。分類・PCA・素材別活動評価は行わない。"""
    sid = SETTINGS["debug_trial_id"]
    if not 1 <= sid <= SETTINGS["trials_per_material"]:
        raise ValueError("debug_trial_id must be between 1 and 324")
    records = []
    for material in SETTINGS["materials"]:
        folder = out / "all_materials" / material
        file = folder / "features.npz"
        trial_ids = np.arange(1, SETTINGS["trials_per_material"] + 1)
        if file.exists():
            with np.load(file) as data:
                bins = data["counts_25ms"].copy()
                if not np.array_equal(data["trial_ids"], trial_ids):
                    raise ValueError(f"Unexpected trial IDs: {file}")
        else:
            bins = simulate(currents(material, out), ref, gains)
        if bins.shape != (324, 100, 20) or np.any(bins < 0):
            raise ValueError(f"Unexpected spike-count states: {material}: {bins.shape}")
        folder.mkdir(parents=True, exist_ok=True)
        # 既存のPCAノートブック用の counts_25ms（試行×ニューロン×時間）は保持。
        # states_hz は各試行を20時間ビン×100ニューロンとしてそのまま取り出せる。
        states_hz = bins.transpose(0, 2, 1) / (SETTINGS["bin_ms"] / 1000.0)
        temporary = folder / "features.partial.npz"
        np.savez_compressed(
            temporary,
            states_hz=states_hz,
            counts_25ms=bins,
            trial_ids=trial_ids,
            bin_ms=SETTINGS["bin_ms"],
            neuron_ids=np.arange(100),
            neuron_types=np.array(["E"] * 80 + ["I"] * 20),
            time_bin_edges_ms=np.arange(21) * SETTINGS["bin_ms"],
        )
        temporary.replace(file)
        # debugは各素材の代表1試行のみ。全324試行の膜電位は記録しない。
        trial = filtered_trial(material, sid, config(gains))
        verify_brian(
            trial[None],
            ref,
            gains,
            folder / f"debug_sid{sid}",
            debug_title=f"{material} | Trial {sid} | 100 liquid neurons",
            expected_counts=bins[sid - 1],
        )
        records.append(
            {
                "material": material,
                "file": str(file.relative_to(out)),
                "states_shape": list(states_hz.shape),
                "debug_trial_id": sid,
            }
        )
        print(f"[recorded] {material}: {states_hz.shape} (trial, time bin, neuron)", flush=True)
    save(
        out / "summary.npz",
        {
            "status": "complete",
            "selected_parameters": gains,
            "n_trials": 2592,
            "state_key": "states_hz",
            "state_unit": "Hz",
            "state_axes": ["trial", "time_bin", "neuron"],
            "bin_ms": SETTINGS["bin_ms"],
            "state_policy": SETTINGS["state_policy"],
            "classification_performed": False,
            "PCA_performed": False,
            "materials": records,
        },
    )


def plan():
    """現在の校正条件、結合確率、調整手順を表示するだけで、探索は開始しない。"""
    ref = load_reference()
    connected = np.flatnonzero(np.any(ref["input_unit_weights"] != 0, axis=0))
    print(
        json.dumps(
            {
                "settings": SETTINGS,
                "connected_E_ids": connected.tolist(),
                "expected_direct_fraction": 1 - (1 - 0.05) ** 6,
                "actual_direct_count": len(connected),
                "all_have_recurrent_input_probability": {
                    "EE": (1 - (1 - 0.11) ** 79) ** 80,
                    "EI": (1 - (1 - 0.10) ** 80) ** 20,
                    "IE": (1 - (1 - 0.37) ** 20) ** 80,
                },
                "steps": [
                    "1: fix 100 neurons, E80/I20 and LIF/synapse parameters",
                    "2: input p=.05; reuse fixed topology",
                    "3: integer input gain; no recurrence; all connected neurons in each trial",
                    "4: EE p=.11",
                    "5: EE gain; EI/IE/II gains zero; E80 in each trial",
                    "6: EI p=.10",
                    "7: EI gain; IE/II gains zero; I20 in each trial",
                    "8: IE p=.37",
                    "9: IE gain; every E/I neuron across all 324 trials; minimize E rate",
                    f'10: output EE p={SETTINGS["output_probability"]:g}; match liquid E/output mean firing rates across 8 x 324 trials; no learning',
                    "11: record 8 x 324 fresh states (20 time bins x 100 neurons); one debug trial per material",
                ],
                "minimum_scope": "fixed topology and screened gains; recurrent response is not assumed monotonic",
                "note": "Japanese_paper is the selected calibration material; no claim it is both smallest and largest input.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def calibration_main(argv=None):
    """gain校正・検証・出力校正を、指定された操作に応じて実行する。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "verify", "output"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CALIBRATION_ROOT,
        help="保存先（既定: LSM/g_tactile_results/calibration）",
    )
    parser.add_argument(
        "--liquid-dir",
        type=Path,
        default=CALIBRATION_ROOT,
        help="For output action: directory containing calibrated liquid gains",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed stages with identical settings/source hashes",
    )
    args = parser.parse_args(argv)
    if args.action == "plan":
        plan()
        return
    out = args.out_dir.resolve()
    ref = load_reference()
    if args.action == "output":
        source = args.liquid_dir.resolve()
        if out.exists():
            parser.error(
                "Choose a new --out-dir for output calibration; existing liquid results are preserved."
            )
        gains = read_record(source / "selected_parameters.json")
        if not all(key in gains for key in ["input_gain", "EE_gain", "EI_gain", "IE_gain"]):
            parser.error("Liquid calibration is incomplete")
        out.mkdir(parents=True)
        save(
            out / "plan.npz",
            {"provenance": provenance(), "liquid_source": str(source), "liquid_gains": gains},
        )
        selected = calibrate_output(ref, gains, out / "output_calibration", input_dir=source)
        save(out / "selected_parameters.json", selected)
        print(f"[complete output] {out}", flush=True)
        return

    if args.action == "verify":
        out.mkdir(parents=True, exist_ok=True)
        gain_file = out / "selected_parameters.json"
        gains = (
            read_record(gain_file)
            if gain_file.exists()
            else {"input_gain": 179.0, "EE_gain": 0.150, "EI_gain": 0.206, "IE_gain": 0.01390625}
        )
        trial = filtered_trial(SETTINGS["input_material"], 203, config(gains))
        verify_brian(trial[None], ref, gains, out / "verification/smoke")
        return
    manifest = provenance()
    if out.exists():
        if not args.resume:
            parser.error(
                "Output directory exists. Use --resume with identical conditions or a new --out-dir."
            )
        if not record_exists(out / "plan.npz") or read_record(out / "plan.npz") != manifest:
            parser.error("Resume settings/source hashes do not match this run.")
    else:
        out.mkdir(parents=True)
        save(out / "plan.npz", manifest)
    gain_file = out / "selected_parameters.json"
    if gain_file.exists():
        gains = read_record(gain_file)
    else:
        x = currents(SETTINGS["input_material"], out)
        summary = out / "input_summary.npz"
        gain = (
            read_record(summary)["input_gain"]
            if record_exists(summary)
            else input_gain(x, ref, out)
        )
        verify_brian(x[202:203], ref, {"input_gain": gain}, out / "verification/input")
        x = currents(SETTINGS["recurrent_material"], out)
        gains = recurrent_gains(x, ref, {"input_gain": gain, "II_gain": 0.0}, out)
    # Verify even when gains were loaded after an interrupted run.
    x = currents(SETTINGS["recurrent_material"], out)
    for stage in ["EE", "EI", "IE"]:
        selected = {
            **gains,
            "IE_gain": gains["IE_gain"] if stage == "IE" else 0.0,
            "EI_gain": gains["EI_gain"] if stage != "EE" else 0.0,
        }
        verify_brian(x[202:203], ref, selected, out / "verification" / stage)
    gains = calibrate_output(ref, gains, out / "output_calibration", input_dir=out)
    save(out / "selected_parameters.json", gains)
    record_material_states(ref, gains, out)
    print(f"[complete] {out}", flush=True)


# 固定重みによるLiquid・Readout間の相互相関解析。
def save_analysis_record(path, value):
    from d_tools.recording_io import save_record

    return save_record(path, value)


def sha(path):
    """入力・重み・ソースの変更を検知するため、ファイルのSHA-256を計算する。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def choose_pairs(gains, seed=12345):
    # 80×40の全てのpre/post組を保持し、接続フラグで分類する。
    pre, post = np.indices(gains.shape)
    return np.column_stack((pre.ravel(), post.ravel(), (gains > 0).ravel())).astype(int)


def pair_correlograms(
    pre_packed, post_packed, pairs, max_lag_steps, *, trial_batch=8, pair_batch=32
):
    """各試行の線形相互相関を足す。正lagはpreが先、postが後。"""
    trials = len(pre_packed)
    steps = 5000
    fft_size = 2 * steps
    n_freq = fft_size // 2 + 1
    pi, pj = pairs[:, 0], pairs[:, 1]
    counts = np.empty((len(pairs), 2 * max_lag_steps + 1), np.int32)
    same_spectrum = np.zeros((len(pairs), n_freq), np.complex64)
    pre_total_time = np.zeros((pre_packed.shape[1], steps), np.int64)
    post_total_time = np.zeros((post_packed.shape[1], steps), np.int64)
    for begin in range(0, trials, trial_batch):
        pre = np.unpackbits(pre_packed[begin : begin + trial_batch], axis=-1, count=steps).astype(
            np.float32
        )
        post = np.unpackbits(post_packed[begin : begin + trial_batch], axis=-1, count=steps).astype(
            np.float32
        )
        pre_total_time += pre.sum(axis=0, dtype=np.int64)
        post_total_time += post.sum(axis=0, dtype=np.int64)
        fx = np.fft.rfft(pre, n=fft_size, axis=-1).astype(np.complex64)
        fy = np.fft.rfft(post, n=fft_size, axis=-1).astype(np.complex64)
        # 周波数方向に分割して、全pre×postをまとめて積算する。
        # ペア単位でFFT積を繰り返すより大幅に少ないカーネル呼び出しで処理できる。
        freq_batch = 512
        for f0 in range(0, n_freq, freq_batch):
            f1 = min(f0 + freq_batch, n_freq)
            cross = np.einsum(
                "bif,bjf->ijf", fx[:, :, f0:f1].conj(), fy[:, :, f0:f1], optimize=True
            )
            same_spectrum[:, f0:f1] += cross.reshape(len(pairs), f1 - f0)
    lag_steps = np.arange(-max_lag_steps, max_lag_steps + 1)
    indices = lag_steps % fft_size
    rounding_error = 0.0
    for start in range(0, len(pairs), pair_batch):
        end = min(start + pair_batch, len(pairs))
        selected = np.fft.irfft(same_spectrum[start:end], n=fft_size, axis=-1)[:, indices]
        rounding_error = max(rounding_error, float(np.max(np.abs(selected - np.rint(selected)))))
        counts[start:end] = np.rint(selected).astype(np.int32)
    if rounding_error > 0.1:
        raise ArithmeticError(f"FFT pair-count roundoff too large: {rounding_error}")
    del same_spectrum
    # 全ての異なる試行の組み合わせを用いたshift predictor。
    # sum_{i != j} corr(pre_i, post_j)/(n_trials-1)で同一試行の合計とスケールをそろえる。
    fpre = np.fft.rfft(pre_total_time, n=fft_size, axis=-1)
    fpost = np.fft.rfft(post_total_time, n=fft_size, axis=-1)
    shifted = np.empty(counts.shape, np.float32)
    for start in range(0, len(pairs), pair_batch):
        end = min(start + pair_batch, len(pairs))
        all_spectrum = fpre[pi[start:end]].conj() * fpost[pj[start:end]]
        total = np.fft.irfft(all_spectrum, n=fft_size, axis=-1)[:, indices]
        shifted[start:end] = ((np.rint(total) - counts[start:end]) / (trials - 1)).astype(
            np.float32
        )
    prefix = np.pad(np.cumsum(pre_total_time, axis=1), ((0, 0), (1, 0)))
    exposure = np.empty((pre_packed.shape[1], len(lag_steps)), np.int64)
    for j, lag in enumerate(lag_steps):
        exposure[:, j] = prefix[:, steps - lag] if lag >= 0 else prefix[:, steps] - prefix[:, -lag]
    return {
        "pair_counts": counts,
        "shift_predictor_counts": shifted,
        "pre_exposure": exposure,
        "pre_spike_totals": pre_total_time.sum(axis=1),
        "post_spike_totals": post_total_time.sum(axis=1),
        "lag_ms": lag_steps * 0.1,
        "fft_rounding_error": np.array(rounding_error),
    }


def conditional_rates(counts, shifted, exposure, pre_ids, bin_s=0.0001):
    denominator = exposure[pre_ids] * bin_s
    raw = np.full(counts.shape, np.nan, dtype=np.float32)
    corrected = np.full(counts.shape, np.nan, dtype=np.float32)
    np.divide(counts, denominator, out=raw, where=denominator > 0)
    np.divide(counts - shifted, denominator, out=corrected, where=denominator > 0)
    return raw, corrected


def window_scores(counts, lag_ms, duration_s, taus=(5.0, 10.0, 20.0, 40.0), area_ms=0.00819):
    """固定スパイク・加算型all-to-all STDPの、clip前の有限窓内更新量/秒。"""
    positive, negative = lag_ms > 1e-9, lag_ms < -1e-9
    ltp = np.stack(
        [
            (area_ms / tau) * (counts[:, positive] @ np.exp(-lag_ms[positive] / tau)) / duration_s
            for tau in taus
        ],
        axis=1,
    )
    ltd = np.stack(
        [
            -(area_ms / tau) * (counts[:, negative] @ np.exp(lag_ms[negative] / tau)) / duration_s
            for tau in taus
        ],
        axis=1,
    )
    return ltp, ltd, ltp[:, :, None] + ltd[:, None, :]


def self_test():
    # 実際の時間刻みで、pre(t=10ms), post(t=10.7ms)の符号を確認。
    """既知のスパイク時刻でlagの符号・補正・STDP窓の計算を検証する。"""
    x = np.zeros((2, 1, 5000), np.uint8)
    y = np.zeros_like(x)
    x[0, 0, 100] = 1
    y[0, 0, 107] = 1
    x[1, 0, 200] = 1
    y[1, 0, 207] = 1
    pairs = np.array([[0, 0, 1]])
    result = pair_correlograms(
        np.packbits(x, axis=-1), np.packbits(y, axis=-1), pairs, 20, trial_batch=1
    )
    peak = int(np.argmax(result["pair_counts"][0]))
    assert np.isclose(result["lag_ms"][peak], 0.7)
    assert result["pair_counts"][0, peak] == 2 and result["pair_counts"].sum() == 2
    # 異なる試行間では±10ms以上離れるので、この短い範囲のshift predictorは0。
    np.testing.assert_allclose(result["shift_predictor_counts"], 0, atol=1e-6)
    ltp, ltd, score = window_scores(result["pair_counts"], result["lag_ms"], 1.0)
    assert np.all(ltp > 0) and np.all(ltd == 0) and np.all(score > 0)
    reverse = pair_correlograms(
        np.packbits(y, axis=-1), np.packbits(x, axis=-1), pairs, 20, trial_batch=1
    )
    assert np.isclose(reverse["lag_ms"][np.argmax(reverse["pair_counts"][0])], -0.7)
    # ゼロgain/無活動の相手に対して架空の相関を作らない。
    y[:] = 0
    empty = pair_correlograms(np.packbits(x, axis=-1), np.packbits(y, axis=-1), pairs, 20)
    assert not empty["pair_counts"].any()


def _load_correlation_spikes(
    material, source, input_dir, folder, wi, wr, wo, *, batch_size, started
):
    """固定重みでスパイクを再構成し、保存済み発火数との一致を検査して読み込む。"""
    spike_path = folder / "spikes.npz"
    if not spike_path.exists():
        pre = np.empty((324, 80, 625), np.uint8)
        post = np.empty((324, wo.shape[1], 625), np.uint8)
        inputs = np.load(input_dir / f"{material}.npy", mmap_mode="r")
        with np.load(source / "all_materials" / material / "liquid/features.npz") as d:
            mapping = {int(v): i for i, v in enumerate(d["trial_ids"])}
            liq_ref = d["counts_25ms"][[mapping[i] for i in range(1, 325)]]
        with np.load(source / "all_materials" / material / "output/features.npz") as d:
            mapping = {int(v): i for i, v in enumerate(d["trial_ids"])}
            out_ref = d["counts_25ms"][[mapping[i] for i in range(1, 325)]]
        for begin in range(0, 324, batch_size):
            end = min(begin + batch_size, 324)
            px = np.empty((end - begin, 80, 5000), bool)
            py = np.empty((end - begin, wo.shape[1], 5000), bool)

            def callback(t, fired, out_fired, v, ov):
                px[:, :, t] = fired[:, :80]
                py[:, :, t] = out_fired

            bins, _, liquid = _simulate_states(
                inputs[begin:end], wi, wr, wo, 1.0, trace_callback=callback
            )
            np.testing.assert_array_equal(liquid, liq_ref[begin:end])
            np.testing.assert_array_equal(bins, out_ref[begin:end])
            np.testing.assert_array_equal(
                px.reshape(end - begin, 80, 20, 250).sum(-1), liquid[:, :80]
            )
            np.testing.assert_array_equal(
                py.reshape(end - begin, wo.shape[1], 20, 250).sum(-1), bins
            )
            pre[begin:end] = np.packbits(px, axis=-1)
            post[begin:end] = np.packbits(py, axis=-1)
        np.savez_compressed(
            spike_path, pre_packed=pre, post_packed=post, trial_ids=np.arange(1, 325), dt_ms=0.1
        )
        print(
            f"[spikes] {material}: all324 trials match saved liquid/output counts | {time.perf_counter()-started:.1f}s",
            flush=True,
        )
    with np.load(spike_path) as d:
        pre, post = d["pre_packed"].copy(), d["post_packed"].copy()
    return pre, post


def _save_material_correlograms(ccg_path, pre, post, pairs, max_lag_steps, current_stdp):
    """各結合の相互相関と候補STDP窓による更新量を計算・保存する。"""
    result = pair_correlograms(pre, post, pairs, max_lag_steps)
    taus = np.array([5.0, 10.0, 20.0, 40.0])
    area_ms = 0.0007 * 11.7
    ltp, ltd, score = window_scores(
        result["pair_counts"], result["lag_ms"], 324 * 0.5, taus=taus, area_ms=area_ms
    )
    lag = result["lag_ms"]
    positive = lag > 1e-9
    negative = lag < -1e-9
    current_kernel = np.zeros(len(lag))
    current_kernel[positive] = current_stdp["A_plus"] * np.exp(
        -lag[positive] / current_stdp["tau_plus"]
    )
    current_kernel[negative] = current_stdp["A_minus"] * np.exp(
        lag[negative] / current_stdp["tau_minus"]
    )
    result.update(
        taus_ms=taus,
        window_area_ms=np.array(area_ms),
        ltp_per_s=ltp,
        ltd_per_s=ltd,
        net_update_per_s=score,
        current_stdp_update_per_s=result["pair_counts"] @ current_kernel / (324 * 0.5),
    )
    np.savez_compressed(ccg_path, **result)


def _material_correlation_summary(material, ccg_path, pairs):
    """活動している実結合ペアについて、補正相関のピークと更新量を集計する。"""
    with np.load(ccg_path) as d:
        raw, corrected = conditional_rates(
            d["pair_counts"], d["shift_predictor_counts"], d["pre_exposure"], pairs[:, 0]
        )
        connected = pairs[:, 2].astype(bool)
        # 生存ペアはpre/postの両方に少なくとも1回の発火があるもの。
        active = (d["pre_spike_totals"][pairs[:, 0]] > 0) & (
            d["post_spike_totals"][pairs[:, 1]] > 0
        )
        idx = np.flatnonzero(connected & active)
        average = np.nanmean(corrected[idx], axis=0)
        peak = int(np.nanargmax(average))
        return {
            "material": material,
            "trials": 324,
            "connected_pairs": int(connected.sum()),
            "active_connected_pairs": len(idx),
            "unconnected_pairs": int((~connected).sum()),
            "mean_connected_excess_peak_lag_ms": float(d["lag_ms"][peak]),
            "mean_connected_excess_peak_hz": float(average[peak]),
            "mean_window_update_per_s": d["net_update_per_s"][connected].mean(axis=0).tolist(),
            "fft_rounding_error": float(d["fft_rounding_error"]),
        }


def correlation_main(argv=None):
    """固定重みのスパイク列から、素材・結合別の相互相関を集計する。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=CALIBRATION_ROOT / "output_calibration")
    parser.add_argument("--input-dir", type=Path, default=CALIBRATION_ROOT / "filtered_inputs")
    parser.add_argument("--out-dir", type=Path, default=CORRELATION_ROOT)
    parser.add_argument("--max-lag-ms", type=float, default=250.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    self_test()
    if args.self_test:
        print("PASS: exact pair-count sign, shift predictor, STDP window and inactive controls")
        return
    source, out = args.source_dir.resolve(), args.out_dir.resolve()
    summary = read_record(source / "output_summary.npz")
    materials = [r["material"] for r in summary["materials"]]
    wi = np.load(source / "weights/w_in1_liq1.npy")
    wr = np.load(source / "weights/w_liq1_intra.npy")
    wo = np.load(source / "weights/w_liq1_out1.npy")
    if wi.shape != (6, 100) or wr.shape != (100, 100) or wo.shape[0] != 100 or np.any(wo[80:] != 0):
        raise ValueError("This analysis requires the saved E-only output topology")
    pairs = choose_pairs(wo[:80])
    max_lag_steps = int(round(args.max_lag_ms / 0.1))
    if not 1 <= max_lag_steps < 5000 or args.batch_size < 1:
        parser.error("Invalid lag/batch size")
    condition = {
        "source": str(source),
        "n_liquid_E": 80,
        "n_output": wo.shape[1],
        "gain": summary["output_gain"],
        "source_probability": summary["output_probability"],
        "n_connected_pairs": int(np.count_nonzero(pairs[:, 2])),
        "n_unconnected_pairs": int(np.count_nonzero(pairs[:, 2] == 0)),
        "bin_ms": 0.1,
        "max_lag_ms": max_lag_steps * 0.1,
        "materials": materials,
        "positive_lag": "readout after liquid",
        "state_policy": "reset_each_trial",
        "learning": False,
        "code_sha256": sha(Path(__file__)),
        "simulator_sha256": sha(ROOT / "f_run/run_calibration.py"),
        "weight_sha256": {p.name: sha(p) for p in (source / "weights").glob("*.npy")},
        "input_sha256": {m: sha(args.input_dir / f"{m}.npy") for m in materials},
    }
    if out.exists():
        if read_record(out / "plan.npz") != condition:
            parser.error("Saved analysis conditions differ; choose a new --out-dir")
    else:
        out.mkdir(parents=True)
        save_analysis_record(out / "plan.npz", condition)
    np.savez_compressed(out / "connections.npz", pairs=pairs, gains=wo[:80], wi=wi, wr=wr, wo=wo)
    cfg_tree = ast.parse(
        (ROOT / "c_configs/FIXED/cfg_learning_rule_models.py").read_text(encoding="utf-8")
    )
    cfg_node = next(
        n
        for n in cfg_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CFG_LEARNING_RULE_MODELS" for t in n.targets)
    )
    current_stdp = ast.literal_eval(cfg_node.value)["STDP"]
    records = []
    started = time.perf_counter()
    print(
        f'[source] saved output{wo.shape[1]}, p={summary["output_probability"]}, gain={summary["output_gain"]}; '
        f'all {len(pairs)} pairs ({condition["n_connected_pairs"]} connected, '
        f'{condition["n_unconnected_pairs"]} unconnected)',
        flush=True,
    )
    for material in materials:
        folder = out / material
        folder.mkdir(exist_ok=True)
        pre, post = _load_correlation_spikes(
            material,
            source,
            args.input_dir,
            folder,
            wi,
            wr,
            wo,
            batch_size=args.batch_size,
            started=started,
        )
        ccg_path = folder / "correlograms.npz"
        if not ccg_path.exists():
            _save_material_correlograms(ccg_path, pre, post, pairs, max_lag_steps, current_stdp)
            print(
                f"[CCG] {material}: {len(pairs)} individual pairs at0.1ms, +/-{args.max_lag_ms:g}ms | {time.perf_counter()-started:.1f}s",
                flush=True,
            )
        records.append(_material_correlation_summary(material, ccg_path, pairs))
    save_analysis_record(
        out / "summary.npz",
        {
            "condition": condition,
            "materials": records,
            "current_stdp": current_stdp,
            "candidate_taus_ms": [5, 10, 20, 40],
            "zero_lag_policy": "excluded from STDP overlap; plotted and counted separately",
            "window_estimate": "fixed-spike all-to-all additive pair rule, before clipping, finite +/-lag range",
            "shift_predictor": "all different-trial pairs, scaled to same-trial total; descriptive correction, not a causal test",
        },
    )
    print(f"[complete] {out}", flush=True)


def main(argv=None):
    """校正の各操作と相互相関解析を1つの入口から選択する。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument(
            "command",
            nargs="?",
            choices=["plan", "run", "verify", "output", "correlation"],
            help="plan: 校正計画 / run: gain校正と記録 / verify: 検証 / output: 出力gain校正 / correlation: 相互相関解析",
        )
        parser.epilog = "各操作のオプション: python f_run/run_calibration.py <command> --help"
        parser.print_help()
        return
    if argv and argv[0] in ("plan", "run", "verify", "output"):
        return calibration_main(argv)
    if argv and argv[0] == "correlation":
        return correlation_main(argv[1:])
    return correlation_main(argv)


if __name__ == "__main__":
    main()
