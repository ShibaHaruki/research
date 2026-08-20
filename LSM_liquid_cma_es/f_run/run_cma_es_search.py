"""CMA-ES search for liquid-only hyperparameters."""



from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

try:
    import cma
except ImportError as exc:
    raise SystemExit(
        "The CMA-ES search now uses the pycma package. Install it with: "
        "python -m pip install cma"
    ) from exc

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_filter, cfg_network, cfg_run
from c_configs.CMA_ES.cfg_search import (
    OBJECTIVE_DEFAULTS,
    PARAMS as ALL_PARAMS,
    SEARCH_DEFAULTS,
)
from d_tools.run_paths import jsonable
from d_tools.mlflow_tracking import MLflowSearchTracker
from f_run.run_common import build_cfg
from f_run.run_liquid import run_liquid
from f_run.run_random_neuron_accuracy import evaluate_random_neuron_accuracy


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_DIR = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
CMA_DIR = RESULTS_DIR / str(RUN_CFG.get("CMA_ES_RESULT_DIR", "cma_es_search"))


def _max_process_workers() -> int:
    """Return a platform-safe upper bound for ProcessPoolExecutor workers."""
    cpu_count = os.cpu_count() or 1
    if sys.platform.startswith("win"):
        return min(cpu_count, 61)
    return cpu_count


def _positive_route_value(value: object) -> bool:
    if isinstance(value, dict):
        return any(_positive_route_value(item) for item in value.values())
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _active_input_filters() -> set[str]:
    configured_filters = {
        str(name).lower()
        for names in cfg_filter.INPUT_FILTER_MAP.values()
        for name in names
    }
    active = set()
    shared_cfg = cfg_network.CFG_NETWORK.get("SHARED_IN_ROUTE", {})
    if shared_cfg.get("enabled", False):
        for name, route_cfg in shared_cfg.get("filters", {}).items():
            if _positive_route_value(route_cfg.get("p", 0.0)):
                active.add(str(name).lower())
    for (_, filter_name), route_cfg in cfg_network.CFG_NETWORK.get("IN_ROUTE", {}).items():
        name = str(filter_name).lower()
        if name not in configured_filters:
            continue
        if any(
            _positive_route_value(layer_cfg.get("p", 0.0))
            for layer_cfg in route_cfg.get("layers", {}).values()
        ):
            active.add(name)
    return active


def _select_active_params(all_params: list[dict]) -> tuple[list[dict], set[str]]:
    active_filters = _active_input_filters()
    configured_search_filters = SEARCH_DEFAULTS.get("search_input_filters")
    requested_filters = (
        None
        if configured_search_filters is None
        else {str(name).lower() for name in configured_search_filters}
    )
    filter_names = tuple(str(name).lower() for name in cfg_filter.FILTER_FUNCS)
    selected = []
    for spec in all_params:
        name = str(spec["name"])
        name_lower = name.lower()
        owner = next(
            (
                filter_name
                for filter_name in filter_names
                if name_lower.startswith(f"{filter_name}_")
            ),
            None,
        )
        if owner is None or (
            owner in requested_filters
            if requested_filters is not None
            else owner in active_filters
        ):
            selected.append(spec)
    return selected, active_filters


PARAMS, ACTIVE_INPUT_FILTERS = _select_active_params(ALL_PARAMS)


def encode_value(value: float, spec: dict) -> float:
    """Encode every parameter into the common normalized coordinate [0, 1]."""
    low = float(spec["low"])
    high = float(spec["high"])
    if high <= low:
        raise ValueError(f"Invalid parameter bounds: low={low}, high={high}")
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def decode_value(raw: float, spec: dict):
    low = float(spec["low"])
    high = float(spec["high"])
    normalized = float(np.clip(raw, 0.0, 1.0))
    value = low + normalized * (high - low)
    step_mode = spec.get("step", "linear")
    if isinstance(step_mode, (int, float)):
        numeric_step = float(step_mode)
        if numeric_step <= 0:
            raise ValueError(f"Parameter step must be positive: {spec['name']}")
        value = low + round((value - low) / numeric_step) * numeric_step
        value = float(np.clip(value, low, high))
        return float(value)
    if isinstance(step_mode, str) and step_mode.strip().lower() in {"int", "integer", "int_log10"}:
        return int(round(value))
    return float(value)


def initial_vector() -> np.ndarray:
    return np.asarray([encode_value(spec["initial"], spec) for spec in PARAMS], dtype=float)


def random_initial_vector(rng: np.random.Generator) -> np.ndarray:
    """Sample a random point directly in the normalized [0, 1] space."""
    return rng.uniform(0.0, 1.0, size=len(PARAMS)).astype(float)


def _neuron_limit(value: int | float | str) -> int | float | None:
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    text = str(value).strip()
    if text.endswith("%"):
        ratio = float(text[:-1]) / 100.0
        if not 0.0 < ratio <= 1.0:
            raise ValueError("neuron percentage must be in the range (0, 100].")
        return ratio
    if isinstance(value, float) and not value.is_integer():
        ratio = float(value)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("neuron ratio must be in the range (0, 1].")
        return ratio
    if "." in text:
        numeric = float(text)
        if 0.0 < numeric <= 1.0:
            return numeric
    count = int(value)
    if count < 1:
        raise ValueError("neurons must be a positive integer, a ratio in (0, 1], a percentage, or 'all'.")
    return count


def decode_vector(x: np.ndarray) -> dict[str, float]:
    return {
        spec["name"]: decode_value(float(x[index]), spec)
        for index, spec in enumerate(PARAMS)
    }


def apply_liquid_params(cfg: dict, params: dict[str, float]) -> dict:
    cfg = deepcopy(cfg)
    if not bool(SEARCH_DEFAULTS["share_filter_input_params_across_sensors"]):
        raise ValueError(
            "CMA-ES currently requires share_filter_input_params_across_sensors=True"
        )
    net = cfg["network"]
    # CMA-ES applies one candidate value to every sensor route below.  Disable
    # the fixed shared-route overlay so build_network_cfg does not overwrite
    # those candidate values afterward.
    shared_route = net.get("SHARED_IN_ROUTE")
    if isinstance(shared_route, dict):
        shared_route["enabled"] = False
    if "n_liq" in params:
        net["N_liq"] = [int(params["n_liq"])]
    net["r_inh_liq"] = float(params.get("r_inh_liq", net.get("r_inh_liq", 0.2)))

    lif_cfg = cfg["neuron_models"]["LIF"]
    lif_cfg["tau_exc"] = params.get("lif_tau_exc", lif_cfg.get("tau_exc", 10.0))
    lif_cfg["tau_inh"] = params.get("lif_tau_inh", lif_cfg.get("tau_inh", 10.0))
    lif_cfg["ref_exc"] = params.get("lif_ref_exc", lif_cfg.get("ref_exc", 2.0))
    lif_cfg["ref_inh"] = params.get("lif_ref_inh", lif_cfg.get("ref_inh", 2.0))
    lif_cfg["bias"] = params.get("lif_bias", lif_cfg.get("bias", -65.0))
    if "lif_v_reset" in params:
        lif_cfg["v_reset"] = params["lif_v_reset"]
    if "lif_v_thr" in params:
        lif_cfg["v_thr"] = max(
            params["lif_v_thr"],
            float(lif_cfg["v_reset"]) + 5.0,
        )

    cfg["models"]["SYNAPSE_MODEL"] = "double_exp"
    syn_cfg = cfg["synapse_models"]["double_exp"]
    syn_cfg["tau_r"] = params["syn_tau_r"]
    syn_cfg["tau_d"] = max(params["syn_tau_d"], params["syn_tau_r"] + 0.1)

    opt_filter_gain = dict(cfg.get("opt_filter_gain", {}))
    for filter_name in ("RI", "SI", "USI", "merkel", "meissner"):
        param_key = f"{filter_name}_opt_gain"
        if param_key in params:
            opt_filter_gain[filter_name] = float(params[param_key])
    cfg["opt_filter_gain"] = opt_filter_gain

    for route_key, route_cfg in net["IN_ROUTE"].items():
        sensor_index = int(route_key[0])
        raw_filter_name = str(route_key[1])
        filter_name = raw_filter_name.lower()
        configured_search_filters = SEARCH_DEFAULTS.get("search_input_filters")
        if (
            configured_search_filters is not None
            and filter_name not in {str(name).lower() for name in configured_search_filters}
        ):
            for layer_cfg in route_cfg.get("layers", {}).values():
                layer_cfg["p"] = {"E": 0.0, "I": 0.0}
            continue
        p_keys = (f"{filter_name}_p", f"{raw_filter_name}_p")
        p_e_keys = (f"{filter_name}_p_E", f"{raw_filter_name}_p_E")
        p_i_keys = (f"{filter_name}_p_I", f"{raw_filter_name}_p_I")
        gain_keys = (f"{filter_name}_gain", f"{raw_filter_name}_gain")
        gain_e_keys = (f"{filter_name}_gain_E", f"{raw_filter_name}_gain_E")
        gain_i_keys = (f"{filter_name}_gain_I", f"{raw_filter_name}_gain_I")
        legacy_sensor_p_key = f"sensor{sensor_index}_p"
        legacy_filter_p_key = f"sensor{sensor_index}_{filter_name}_p"
        legacy_scale_key = f"sensor{sensor_index}_{filter_name}_scale"
        scale_key = next(
            (
                key
                for key in (*gain_keys, legacy_scale_key)
                if key in params
            ),
            None,
        )
        scale_e_key = next((key for key in gain_e_keys if key in params), None)
        scale_i_key = next((key for key in gain_i_keys if key in params), None)
        probability_key = next(
            (
                key
                for key in (*p_keys, legacy_sensor_p_key, legacy_filter_p_key, "input_p")
                if key in params
            ),
            None,
        )
        probability_e_key = next((key for key in p_e_keys if key in params), None)
        probability_i_key = next((key for key in p_i_keys if key in params), None)
        for layer_cfg in route_cfg.get("layers", {}).values():
            if probability_e_key is not None or probability_i_key is not None:
                current_p = layer_cfg.get("p", {})
                if not isinstance(current_p, dict):
                    current_p = {"E": current_p, "I": current_p}
                layer_cfg["p"] = {
                    "E": float(params[probability_e_key])
                    if probability_e_key is not None
                    else float(current_p.get("E", 0.0)),
                    "I": float(params[probability_i_key])
                    if probability_i_key is not None
                    else float(current_p.get("I", 0.0)),
                }
            elif probability_key is not None:
                probability = float(params[probability_key])
                layer_cfg["p"] = {"E": probability, "I": probability}
            if scale_e_key is not None or scale_i_key is not None:
                current_scale = layer_cfg.get("scale", {})
                if not isinstance(current_scale, dict):
                    current_scale = {"E": current_scale, "I": current_scale}
                layer_cfg["scale"] = {
                    "E": float(params[scale_e_key])
                    if scale_e_key is not None
                    else float(current_scale.get("E", 0.0)),
                    "I": float(params[scale_i_key])
                    if scale_i_key is not None
                    else float(current_scale.get("I", 0.0)),
                }
            elif scale_key is not None:
                scale = float(params[scale_key])
                layer_cfg["scale"] = {"E": scale, "I": scale}

    net["p_liq_intra_pairs"] = {
        "EE": params["rec_p_ee"],
        "EI": params["rec_p_ei"],
        "IE": params["rec_p_ie"],
        "II": params.get("rec_p_ii", 0.0),
    }
    net["liq_intra_gain_pairs"] = {
        "EE": params["rec_gain_ee"],
        "EI": params["rec_gain_ei"],
        "IE": params["rec_gain_ie"],
        "II": params.get("rec_gain_ii", 0.0),
    }

    net["poisson_input"]["enabled"] = False
    return cfg


def _candidate_out_dir(search_dir: Path, generation: int, candidate: int) -> Path:
    return search_dir / f"gen{generation:03d}_cand{candidate:03d}"


def collect_spike_count_metrics(
    internal_state_dir: Path,
    *,
    max_samples_per_class: int | None = None,
) -> dict:
    per_trial = []
    firing_rates = []
    firing_rates_exc = []
    firing_rates_inh = []
    exc_spike_counts = []
    inh_spike_counts = []
    exc_neuron_counts = []
    inh_neuron_counts = []
    material_counts = {}
    for material_dir in sorted(Path(internal_state_dir).iterdir()):
        if not material_dir.is_dir():
            continue
        manifests = sorted(material_dir.glob("*_internal_state_manifest.json"))
        if max_samples_per_class is not None:
            manifests = manifests[: int(max_samples_per_class)]
        material_spikes = []
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            total_spikes = sum(
                int(layer.get("spike_count", 0))
                for layer in manifest.get("layers", [])
            )
            material_spikes.append(float(total_spikes))
            per_trial.append(float(total_spikes))
            combined_name = manifest.get("combined_file")
            if combined_name:
                npz_path = material_dir / str(combined_name)
                if npz_path.exists():
                    with np.load(npz_path) as data:
                        x_state = np.asarray(data["x_state"])
                        n_neurons = int(x_state.shape[0])
                        typ = np.asarray(data["typ"], dtype=np.int32)
                        spike_i = np.asarray(
                            data["spike_i"] if "spike_i" in data.files else (),
                            dtype=np.int64,
                        )
                        t_ms = np.asarray(data["t_ms"], dtype=float)
                        bin_ms = float(np.asarray(data["bin_ms"])[0]) if "bin_ms" in data.files else 1.0
                    duration_ms = max(float(t_ms.size) * bin_ms, 1e-9)
                    firing_rates.append(
                        float(total_spikes) / max(n_neurons, 1) / (duration_ms / 1000.0)
                    )
                    n_exc = max(int(np.count_nonzero(typ == 1)), 1)
                    n_inh = max(int(np.count_nonzero(typ == -1)), 1)
                    exc_spikes = int(np.count_nonzero(spike_i < n_exc))
                    inh_spikes = int(np.count_nonzero(spike_i >= n_exc))
                    exc_spike_counts.append(float(exc_spikes))
                    inh_spike_counts.append(float(inh_spikes))
                    exc_neuron_counts.append(float(n_exc))
                    inh_neuron_counts.append(float(n_inh))
                    firing_rates_exc.append(
                        float(exc_spikes) / n_exc / (duration_ms / 1000.0)
                    )
                    firing_rates_inh.append(
                        float(inh_spikes) / n_inh / (duration_ms / 1000.0)
                    )
        if material_spikes:
            material_counts[material_dir.name] = {
                "n_trials": len(material_spikes),
                "mean_total_spikes": float(np.mean(material_spikes)),
                "std_total_spikes": float(
                    np.std(material_spikes, ddof=1 if len(material_spikes) > 1 else 0)
                ),
            }
    if not per_trial:
        raise FileNotFoundError(
            f"No internal-state manifests with spike counts found under {internal_state_dir}"
        )
    return {
        "n_spike_trials": len(per_trial),
        "total_spikes_all_trials": float(np.sum(per_trial)),
        "mean_total_spikes_per_trial": float(np.mean(per_trial)),
        "std_total_spikes_per_trial": float(
            np.std(per_trial, ddof=1 if len(per_trial) > 1 else 0)
        ),
        "mean_firing_rate_hz": float(np.mean(firing_rates)) if firing_rates else 0.0,
        "std_firing_rate_hz": float(
            np.std(firing_rates, ddof=1 if len(firing_rates) > 1 else 0)
        ) if firing_rates else 0.0,
        "mean_firing_rate_exc_hz": float(np.mean(firing_rates_exc)) if firing_rates_exc else 0.0,
        "mean_firing_rate_inh_hz": float(np.mean(firing_rates_inh)) if firing_rates_inh else 0.0,
        "mean_exc_spikes_per_trial": float(np.mean(exc_spike_counts)) if exc_spike_counts else 0.0,
        "mean_inh_spikes_per_trial": float(np.mean(inh_spike_counts)) if inh_spike_counts else 0.0,
        "mean_exc_neuron_count": float(np.mean(exc_neuron_counts)) if exc_neuron_counts else 0.0,
        "mean_inh_neuron_count": float(np.mean(inh_neuron_counts)) if inh_neuron_counts else 0.0,
        "std_firing_rate_exc_hz": float(
            np.std(firing_rates_exc, ddof=1 if len(firing_rates_exc) > 1 else 0)
        ) if firing_rates_exc else 0.0,
        "std_firing_rate_inh_hz": float(
            np.std(firing_rates_inh, ddof=1 if len(firing_rates_inh) > 1 else 0)
        ) if firing_rates_inh else 0.0,
        "spike_counts_by_material": material_counts,
    }


def collect_neuron_activity_metrics(
    internal_state_dir: Path,
    *,
    max_samples_per_class: int | None = None,
    min_spikes_per_neuron: int = 1,
    min_trial_fraction: float = 0.125,
    trials_per_material: int = 1,
) -> dict:
    """Measure neurons active above a spike threshold in representative trials.

    A neuron is active when its raw spike count reaches the threshold in at
    least one trial across all selected materials and samples. A neuron that
    never spikes anywhere is counted as silent.
    """
    if int(min_spikes_per_neuron) < 1:
        raise ValueError("min_spikes_per_neuron must be at least 1")
    if not 0.0 < float(min_trial_fraction) <= 1.0:
        raise ValueError("min_trial_fraction must be in (0, 1]")
    if int(trials_per_material) < 0:
        raise ValueError("trials_per_material must be non-negative")

    active_by_material: list[np.ndarray] = []
    activity_trials: list[np.ndarray] = []
    typ_reference: np.ndarray | None = None
    n_trials = 0
    n_mismatched = 0
    for material_dir in sorted(Path(internal_state_dir).iterdir()):
        if not material_dir.is_dir():
            continue
        manifests = sorted(material_dir.glob("*_internal_state_manifest.json"))
        if max_samples_per_class is not None:
            manifests = manifests[: int(max_samples_per_class)]
        if int(trials_per_material) > 0:
            manifests = manifests[: int(trials_per_material)]
        material_active: np.ndarray | None = None
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            combined_name = manifest.get("combined_file")
            if not combined_name:
                continue
            npz_path = material_dir / str(combined_name)
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                x_state = np.asarray(data["x_state"])
                if "typ" in data.files and typ_reference is None:
                    typ_reference = np.asarray(data["typ"], dtype=np.int32).reshape(-1)
                spike_i = np.asarray(
                    data["spike_i"] if "spike_i" in data.files else (),
                    dtype=np.int64,
                )
            if spike_i.size:
                counts = np.bincount(spike_i, minlength=x_state.shape[0])
                trial_active = counts >= int(min_spikes_per_neuron)
            else:
                # Backward-compatible fallback for files without raw indices.
                trial_active = np.count_nonzero(
                    np.isfinite(x_state) & (x_state != 0), axis=1
                ) >= int(min_spikes_per_neuron)
            if material_active is None:
                material_active = trial_active.astype(bool, copy=True)
            elif material_active.shape == trial_active.shape:
                material_active |= trial_active
            else:
                n_mismatched += 1
                n = min(material_active.size, trial_active.size)
                material_active[:n] |= trial_active[:n]
            activity_trials.append(trial_active.astype(bool, copy=True))
            n_trials += 1
        if material_active is not None:
            active_by_material.append(material_active)

    if not active_by_material:
        raise FileNotFoundError(
            f"No combined internal-state npz files found under {internal_state_dir}"
        )

    active_all_materials = np.zeros_like(active_by_material[0], dtype=bool)
    activity_counts = np.zeros(active_all_materials.size, dtype=np.int64)
    for trial_active in activity_trials:
        n = min(activity_counts.size, trial_active.size)
        activity_counts[:n] += trial_active[:n]
    activity_fraction = activity_counts / max(len(activity_trials), 1)
    active_all_materials = activity_fraction >= float(min_trial_fraction)
    silent = ~active_all_materials
    silent_count = int(np.count_nonzero(silent))
    total = int(active_all_materials.size)
    if typ_reference is not None and typ_reference.size >= total:
        typ = typ_reference[:total]
        exc_mask = typ == 1
        inh_mask = typ == -1
        silent_exc_count = int(np.count_nonzero(silent & exc_mask))
        silent_inh_count = int(np.count_nonzero(silent & inh_mask))
        exc_total = int(np.count_nonzero(exc_mask))
        inh_total = int(np.count_nonzero(inh_mask))
    else:
        silent_exc_count = 0
        silent_inh_count = 0
        exc_total = 0
        inh_total = 0
    return {
        "n_activity_trials": int(n_trials),
        "n_activity_materials": int(len(active_by_material)),
        "activity_trials_per_material": int(trials_per_material),
        "activity_min_spikes_per_neuron": int(min_spikes_per_neuron),
        "silent_min_trial_fraction": float(min_trial_fraction),
        "n_activity_mismatched_shapes": int(n_mismatched),
        "active_neuron_count": int(np.count_nonzero(active_all_materials)),
        "silent_neuron_count": silent_count,
        "total_neuron_count": total,
        "silent_neuron_fraction": float(silent_count / max(total, 1)),
        "silent_neuron_count_exc": silent_exc_count,
        "silent_neuron_count_inh": silent_inh_count,
        "total_neuron_count_exc": exc_total,
        "total_neuron_count_inh": inh_total,
        "silent_neuron_fraction_exc": float(silent_exc_count / max(exc_total, 1)),
        "silent_neuron_fraction_inh": float(silent_inh_count / max(inh_total, 1)),
    }


def _score(
    metrics: dict,
    *,
    metric: str,
    α: float,
    β: float,
    γ: float,
    κ: float,
    δ: float,
) -> float:
    acc = float(metrics[f"{metric}_mean"])
    acc_std = float(metrics[f"{metric}_std"])
    acc_variance = acc_std * acc_std
    mean_total_spikes = float(metrics["mean_total_spikes_per_trial"])
    spike_limit = float(metrics["spike_limit"])
    mean_exc_spikes = float(metrics["mean_exc_spikes_per_trial"])
    mean_inh_spikes = float(metrics["mean_inh_spikes_per_trial"])
    mean_exc_neurons = float(metrics["mean_exc_neuron_count"])
    mean_inh_neurons = float(metrics["mean_inh_neuron_count"])
    silent_fraction = float(metrics.get("objective_silent_fraction", 0.0))
    if not math.isfinite(mean_total_spikes):
        raise ValueError(f"mean_total_spikes_per_trial is not finite: {mean_total_spikes}")
    if spike_limit <= 0:
        raise ValueError("spike_limit must be positive")
    total_typed_neurons = mean_exc_neurons + mean_inh_neurons
    if total_typed_neurons <= 0:
        raise ValueError("E/I neuron counts must be positive")
    exc_limit = spike_limit * mean_exc_neurons / total_typed_neurons
    inh_limit = spike_limit * mean_inh_neurons / total_typed_neurons
    # Split the total spike budget according to the E/I neuron ratio.
    spike_limit_penalty = 0.5 * (
        max(0.0, (mean_exc_spikes - exc_limit) / exc_limit) ** 2
        + max(0.0, (mean_inh_spikes - inh_limit) / inh_limit) ** 2
    )
    return float(
        -float(α) * acc
        + float(β) * acc_variance
        + float(γ) * spike_limit_penalty
        + float(δ) * silent_fraction
    )


def evaluate_candidate(
    raw_x: np.ndarray,
    *,
    generation: int,
    candidate: int,
    search_dir: Path,
    args: argparse.Namespace,
) -> dict:
    params = decode_vector(raw_x)
    candidate_dir = _candidate_out_dir(search_dir, generation, candidate)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_liquid_params(build_cfg(), params)
    # Keep every candidate's full liquid output below its CMA-ES directory.
    # The candidate directory already identifies the experiment, so avoid an
    # additional experiment-id level below the network/model directories.
    cfg["run"]["LIQUID_RESULT_ROOT"] = str(candidate_dir.resolve())
    cfg["run"]["INCLUDE_EXPERIMENT_DIR"] = False
    # Keep the decoded candidate alongside the effective network snapshot.
    # This makes it possible to compare the requested search values with the
    # post-normalization route values saved by run_liquid.
    cfg["search_params"] = dict(params)
    cfg["liquid"]["NUM_LIQUID_SAMPLE"] = [int(args.samples_per_class)]
    cfg["run"]["INTERNAL_STATE_BIN_MS"] = float(args.internal_state_bin_ms)
    cfg["run"]["INTERNAL_STATE_PCA_ENABLE"] = False
    cfg["experiment"] = {
        "id": f"cma_gen{generation:03d}_cand{candidate:03d}",
        "name": "cma_es_liquid_accuracy_fisher",
        "trial_id": f"gen{generation:03d}_cand{candidate:03d}",
    }

    (candidate_dir / "candidate_params.json").write_text(
        json.dumps(jsonable(params), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.dry_run:
        return {
            "generation": generation,
            "candidate": candidate,
            "objective": float("nan"),
            "params": params,
            "dry_run": True,
        }

    message = run_liquid(cfg)
    run_dir_text = str(message).split(" in ", 1)[-1]
    run_dir = Path(run_dir_text)
    internal_state_dir = run_dir / str(cfg["run"].get("INTERNAL_STATE_DIR", "internal_states"))
    accuracy_dir = candidate_dir / "random_neuron_accuracy"
    metrics = evaluate_random_neuron_accuracy(
        internal_state_dir,
        n_neurons=_neuron_limit(args.neurons),
        n_repeats=int(args.neuron_selection_repeats),
        test_size=float(args.test_size),
        hold=int(args.hold),
        seed_value=int(args.seed) + generation * 1000 + candidate,
        t_n_ms=float(args.t_n_ms),
        max_samples_per_class=int(args.samples_per_class),
        out_dir=accuracy_dir,
    )
    metrics.update(
        collect_spike_count_metrics(
            internal_state_dir,
            max_samples_per_class=int(args.samples_per_class),
        )
    )
    metrics.update(
        collect_neuron_activity_metrics(
            internal_state_dir,
            max_samples_per_class=int(args.samples_per_class),
            min_spikes_per_neuron=int(args.silent_min_spikes_per_neuron),
            min_trial_fraction=float(args.silent_min_trial_fraction),
            trials_per_material=int(args.silent_trials_per_material),
        )
    )
    # Use the variance across the ten evaluation splits for the objective.
    # The neuron-selection repeat variance remains available as a diagnostic
    # in accuracy8_overall_std.
    metrics["accuracy8_overall_variance"] = float(
        metrics.get("accuracy8_fold_variance_mean", 0.0)
    )
    metrics["accuracy3_overall_variance"] = float(
        metrics.get("accuracy3_fold_variance_mean", 0.0)
    )
    metrics["spike_base"] = float(args.κ)
    metrics["spike_limit"] = float(args.spike_limit)
    metrics["objective_silent_fraction"] = float(
        max(
            metrics.get("silent_neuron_fraction_exc", 0.0),
            metrics.get("silent_neuron_fraction_inh", 0.0),
        )
    )
    typed_neurons = metrics["mean_exc_neuron_count"] + metrics["mean_inh_neuron_count"]
    metrics["exc_spike_limit"] = float(args.spike_limit * metrics["mean_exc_neuron_count"] / typed_neurons)
    metrics["inh_spike_limit"] = float(args.spike_limit * metrics["mean_inh_neuron_count"] / typed_neurons)
    # This is an upper-limit penalty: rates below the limits contribute zero.
    metrics["spike_limit_penalty"] = float(
        0.5 * (
            max(0.0, (metrics["mean_exc_spikes_per_trial"] - metrics["exc_spike_limit"])
                / metrics["exc_spike_limit"]) ** 2
            + max(0.0, (metrics["mean_inh_spikes_per_trial"] - metrics["inh_spike_limit"])
                / metrics["inh_spike_limit"]) ** 2
        )
    )
    metrics["spike_ratio"] = float(
        metrics["mean_total_spikes_per_trial"] / float(args.κ)
    )
    alpha = float(getattr(args, "\u03b1"))
    beta = float(getattr(args, "\u03b2"))
    gamma = float(getattr(args, "\u03b3"))
    delta = float(getattr(args, "\u03b4"))
    accuracy_value = float(metrics[f"{args.metric}_mean"])
    accuracy_variance = float(metrics[f"{args.metric}_variance"])
    metrics["objective_accuracy_contribution"] = -alpha * accuracy_value
    metrics["objective_variance_contribution"] = beta * accuracy_variance
    metrics["objective_spike_contribution"] = gamma * float(
        metrics["spike_limit_penalty"]
    )
    metrics["objective_silent_contribution"] = delta * float(
        metrics["objective_silent_fraction"]
    )
    # DR is retained as a diagnostic metric, but excluded from optimization.
    metrics["objective_fisher_contribution"] = 0.0
    objective = _score(
        metrics,
        metric=str(args.metric),
        α=float(args.α),
        β=float(args.β),
        γ=float(args.γ),
        κ=float(args.κ),
        δ=float(args.δ),
    )
    result = {
        "generation": generation,
        "candidate": candidate,
        "objective": objective,
        "run_dir": str(run_dir),
        "internal_state_dir": str(internal_state_dir),
        "accuracy_dir": str(accuracy_dir),
        "params": params,
        "metrics": metrics,
    }
    (candidate_dir / "candidate_result.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def evaluate_candidate_worker(payload: tuple[np.ndarray, int, int, str, argparse.Namespace]) -> dict:
    raw_x, generation, candidate, search_dir, args = payload
    # On Windows, parallel Cython workers can try to load the same DLL while
    # another worker is still writing it. NumPy avoids that shared-cache race.
    from brian2 import prefs

    prefs.codegen.target = str(args.brian_codegen_target)
    return evaluate_candidate(
        np.asarray(raw_x, dtype=float),
        generation=int(generation),
        candidate=int(candidate),
        search_dir=Path(search_dir),
        args=args,
    )


def cma_es_ask_tell(
    *,
    x0: np.ndarray,
    sigma0: float,
    generations: int,
    population_size: int,
    seed: int,
    evaluate_population,
) -> tuple[np.ndarray, float]:
    strategy = cma.CMAEvolutionStrategy(
        np.asarray(x0, dtype=float).tolist(),
        float(sigma0),
        {
            "popsize": int(population_size),
            "seed": int(seed),
            "verb_disp": 0,
            "verbose": -9,
        },
    )
    best_x = np.asarray(x0, dtype=float).copy()
    best_obj = float("inf")

    for generation in range(1, int(generations) + 1):
        solutions = strategy.ask()
        results = evaluate_population(generation, np.asarray(solutions, dtype=float))
        objectives: list[float] = []
        for result in results:
            obj = float(result["objective"])
            objectives.append(obj if math.isfinite(obj) else 1e300)
            if obj < best_obj:
                best_obj = obj
                best_x = np.asarray(solutions[int(result["candidate"]) - 1], dtype=float).copy()
        strategy.tell(solutions, objectives)
    return best_x, best_obj


def build_search_settings(args: argparse.Namespace) -> dict:
    return {
        "objective": (
            "-α*A + β*Var(A) + γ*P_spike_limit + δ*max(R_silent_E, R_silent_I)"
        ),
        "α": float(args.α),
        "β": float(args.β),
        "γ": float(args.γ),
        "δ": float(args.δ),
        "κ": float(args.κ),
        "spike_definition": "mean total liquid spikes per trial over all evaluated trials",
        "metric": str(args.metric),
        "evaluation_neurons": _neuron_limit(args.neurons) or "all",
        "samples_per_class": int(args.samples_per_class),
        "neuron_selection_repeats": int(args.neuron_selection_repeats),
        "test_size": float(args.test_size),
        "hold": int(args.hold),
        "silent_min_spikes_per_neuron": int(args.silent_min_spikes_per_neuron),
        "silent_min_trial_fraction": float(args.silent_min_trial_fraction),
        "silent_trials_per_material": int(args.silent_trials_per_material),
        "T_n_ms": float(args.t_n_ms),
        "brian_codegen_target": str(args.brian_codegen_target),
        "parameters": PARAMS,
        "all_parameters": ALL_PARAMS,
        "active_input_filters": sorted(ACTIVE_INPUT_FILTERS),
        "search_input_filters": SEARCH_DEFAULTS.get("search_input_filters"),
        "excluded_parameters": [
            spec["name"] for spec in ALL_PARAMS if spec not in PARAMS
        ],
        "n_starts": int(args.n_starts),
        "start_jobs": int(args.start_jobs),
        "randomize_initial_center": bool(SEARCH_DEFAULTS["randomize_initial_center"]),
        "share_filter_input_params_across_sensors": bool(
            SEARCH_DEFAULTS["share_filter_input_params_across_sensors"]
        ),
    }


def make_initial_centers(args: argparse.Namespace) -> list[np.ndarray]:
    base_x0 = initial_vector()
    rng = np.random.default_rng(int(args.seed))
    if bool(SEARCH_DEFAULTS["randomize_initial_center"]):
        return [
            random_initial_vector(rng)
            for _ in range(int(args.n_starts))
        ]
    return [base_x0.copy() for _ in range(int(args.n_starts))]


def run_one_cma_start(
    *,
    args: argparse.Namespace,
    search_dir: Path,
    start_index: int,
    x0: np.ndarray,
    search_settings: dict,
) -> dict:
    search_dir.mkdir(parents=True, exist_ok=True)
    results_csv = search_dir / "cma_es_results.csv"
    (search_dir / "search_settings.json").write_text(
        json.dumps(jsonable(search_settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_result_seen: dict | None = None

    def record_result(result: dict) -> None:
        nonlocal best_result_seen
        row = {
            "start": int(start_index),
            "generation": result.get("generation"),
            "candidate": result.get("candidate"),
            "objective": result.get("objective"),
        }
        metrics = result.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(jsonable(value), ensure_ascii=False)
            else:
                row[key] = value
        for key, value in result.get("params", {}).items():
            row[key] = value

        exists = results_csv.exists()
        if not exists:
            with results_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        else:
            with results_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                previous_rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            new_fields = [key for key in row if key not in fieldnames]
            if new_fields:
                fieldnames.extend(new_fields)
                previous_rows.append(row)
                with results_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(previous_rows)
            else:
                with results_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        objective = result.get("objective")
        if objective is not None and math.isfinite(float(objective)):
            if (
                best_result_seen is None
                or float(objective) < float(best_result_seen["objective"])
            ):
                best_result_seen = result
        print(
            f"[cma] start={start_index} gen={row['generation']} cand={row['candidate']} "
            f"objective={row['objective']} "
            f"acc8={row.get('accuracy8_overall_mean')} "
            f"var8={row.get('accuracy8_overall_variance')} "
            f"rate={row.get('mean_firing_rate_hz')}Hz "
            f"E={row.get('mean_firing_rate_exc_hz')}Hz "
            f"I={row.get('mean_firing_rate_inh_hz')}Hz "
            f"spikes_E={metrics.get('mean_exc_spikes_per_trial')}/"
            f"{metrics.get('exc_spike_limit')} "
            f"spikes_I={metrics.get('mean_inh_spikes_per_trial')}/"
            f"{metrics.get('inh_spike_limit')} "
            f"silent(E={metrics.get('silent_neuron_fraction_exc')}, "
            f"I={metrics.get('silent_neuron_fraction_inh')}) "
            f"weighted(acc8={metrics.get('objective_accuracy_contribution')}, "
            f"var8={metrics.get('objective_variance_contribution')}, "
            f"spike_limit={metrics.get('objective_spike_contribution')}, "
            f"silent={metrics.get('objective_silent_contribution')})"
        )

    def evaluate_population(generation: int, arx: np.ndarray) -> list[dict]:
        payloads = [
            (np.asarray(x, dtype=float), int(generation), int(candidate_index), str(search_dir), args)
            for candidate_index, x in enumerate(arx, start=1)
        ]
        results: list[dict] = []
        jobs = max(1, min(int(args.jobs), len(payloads), _max_process_workers()))
        if jobs == 1:
            for payload in payloads:
                result = evaluate_candidate_worker(payload)
                record_result(result)
                results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                future_to_candidate = {
                    executor.submit(evaluate_candidate_worker, payload): payload[2]
                    for payload in payloads
                }
                for future in as_completed(future_to_candidate):
                    result = future.result()
                    record_result(result)
                    results.append(result)
        return sorted(results, key=lambda item: int(item["candidate"]))

    best_x, best_obj = cma_es_ask_tell(
        x0=np.asarray(x0, dtype=float),
        sigma0=float(args.sigma0),
        generations=int(args.generations),
        population_size=int(args.population_size),
        seed=int(args.seed) + 100000 * int(start_index),
        evaluate_population=evaluate_population,
    )
    best = {
        "start": int(start_index),
        "objective": best_obj,
        "params": decode_vector(best_x),
        "initial_center": decode_vector(np.asarray(x0, dtype=float)),
        "raw_initial_center": np.asarray(x0, dtype=float).tolist(),
        "objective_settings": search_settings,
        "metrics": (
            None
            if best_result_seen is None
            else best_result_seen.get("metrics")
        ),
    }
    (search_dir / "best_params.json").write_text(
        json.dumps(jsonable(best), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[cma] start={start_index} best objective={best_obj}")
    print(f"[cma] start={start_index} saved to {search_dir}")
    return best


def run_one_cma_start_worker(payload: tuple[argparse.Namespace, str, int, np.ndarray, dict]) -> dict:
    args, search_dir, start_index, x0, search_settings = payload
    return run_one_cma_start(
        args=args,
        search_dir=Path(search_dir),
        start_index=int(start_index),
        x0=np.asarray(x0, dtype=float),
        search_settings=search_settings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA-ES search for liquid-only hyperparameters.")
    parser.add_argument("--generations", type=int, default=SEARCH_DEFAULTS["generations"])
    parser.add_argument("--population-size", type=int, default=SEARCH_DEFAULTS["population_size"])
    parser.add_argument("--jobs", type=int, default=SEARCH_DEFAULTS["jobs"])
    parser.add_argument(
        "--brian-codegen-target",
        choices=("numpy", "cython"),
        default=SEARCH_DEFAULTS["brian_codegen_target"],
        help=(
            "Brian2 backend. NumPy avoids shared Cython DLL collisions during "
            "parallel CMA-ES on Windows."
        ),
    )
    parser.add_argument("--sigma0", type=float, default=SEARCH_DEFAULTS["sigma0"])
    parser.add_argument("--seed", type=int, default=SEARCH_DEFAULTS["seed"])
    parser.add_argument(
        "--n-starts",
        type=int,
        default=SEARCH_DEFAULTS["n_starts"],
        help="Number of independent CMA-ES starts. start001 uses the base initial center.",
    )
    parser.add_argument(
        "--start-jobs",
        type=int,
        default=SEARCH_DEFAULTS["start_jobs"],
        help="Number of independent CMA-ES starts to run in parallel.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=SEARCH_DEFAULTS["samples_per_class"],
    )
    parser.add_argument(
        "--neurons",
        type=str,
        default=SEARCH_DEFAULTS["neurons"],
        help="Accuracy neurons: count, ratio (e.g. 0.25), percentage (25%%), or 'all'.",
    )
    parser.add_argument(
        "--neuron-selection-repeats",
        type=int,
        default=SEARCH_DEFAULTS["neuron_selection_repeats"],
        help="Number of random neuron-subset selections per candidate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=SEARCH_DEFAULTS["test_size"],
        help="Test fraction for repeated holdout evaluation (0 to 1).",
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=SEARCH_DEFAULTS["hold"],
        help="Number of repeated holdout splits per neuron selection.",
    )
    parser.add_argument("--t-n-ms", type=float, default=SEARCH_DEFAULTS["t_n_ms"])
    parser.add_argument(
        "--internal-state-bin-ms",
        type=float,
        default=SEARCH_DEFAULTS["internal_state_bin_ms"],
    )
    parser.add_argument(
        "--metric",
        choices=("accuracy8_overall", "accuracy3_overall"),
        default=OBJECTIVE_DEFAULTS["metric"],
    )
    parser.add_argument("--alpha", dest="α", type=float, default=OBJECTIVE_DEFAULTS["α"])
    parser.add_argument(
        "--variance-weight",
        dest="β",
        type=float,
        default=OBJECTIVE_DEFAULTS["β"],
    )
    parser.add_argument(
        "--spike-weight",
        "--beta",
        dest="γ",
        type=float,
        default=OBJECTIVE_DEFAULTS["γ"],
    )
    parser.add_argument(
        "--spike-base",
        dest="κ",
        type=float,
        default=OBJECTIVE_DEFAULTS["κ"],
    )
    parser.add_argument(
        "--spike-limit",
        type=float,
        default=SEARCH_DEFAULTS["spike_limit"],
        help="Allow this many mean total spikes per trial before penalty.",
    )
    parser.add_argument(
        "--silent-weight",
        dest="δ",
        type=float,
        default=OBJECTIVE_DEFAULTS["δ"],
        help=(
            "Penalty weight for the fraction of liquid neurons that never become "
            "active over all evaluated materials/samples. Use 0 to ignore it."
        ),
    )
    parser.add_argument(
        "--silent-min-spikes-per-neuron",
        type=int,
        default=SEARCH_DEFAULTS["silent_min_spikes_per_neuron"],
        help="Minimum actual spikes per neuron in each selected material trial.",
    )
    parser.add_argument(
        "--silent-min-trial-fraction",
        type=float,
        default=SEARCH_DEFAULTS["silent_min_trial_fraction"],
        help="Minimum fraction of trials in which a neuron must spike to be active.",
    )
    parser.add_argument(
        "--silent-trials-per-material",
        type=int,
        default=SEARCH_DEFAULTS["silent_trials_per_material"],
        help="Trials per material for silent metric; 0 uses all available trials.",
    )
    parser.add_argument(
        "--search-name",
        type=str,
        default=SEARCH_DEFAULTS["search_name"],
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Track the full search and each candidate as MLflow runs.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Default: a local mlruns directory next to the search.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="LSM_liquid_cma_es",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Optional name for the parent MLflow run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if float(args.κ) <= 0:
        raise ValueError("--spike-base must be positive")
    if float(args.spike_limit) <= 0:
        raise ValueError("--spike-limit must be positive")
    if int(args.n_starts) <= 0:
        raise ValueError("--n-starts must be positive")
    if int(args.start_jobs) <= 0:
        raise ValueError("--start-jobs must be positive")
    if args.jobs is None:
        args.jobs = min(int(args.population_size), _max_process_workers())
    else:
        args.jobs = min(int(args.jobs), _max_process_workers())

    start_workers = max(
        1,
        min(int(args.start_jobs), int(args.n_starts), _max_process_workers()),
    )
    print(
        "[parallel] candidate_workers="
        f"{int(args.jobs)} population_size={int(args.population_size)} "
        f"start_workers={start_workers} n_starts={int(args.n_starts)}",
        flush=True,
    )

    search_dir = CMA_DIR / args.search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    search_settings = build_search_settings(args)
    tracker = None
    if args.mlflow:
        tracker = MLflowSearchTracker(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            search_dir=search_dir,
        )
        tracker.start(search_settings)
    (search_dir / "search_settings.json").write_text(
        json.dumps(jsonable(search_settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    centers = make_initial_centers(args)
    initial_centers = [
        {
            "start": index,
            "raw_x0": np.asarray(center, dtype=float).tolist(),
            "params": decode_vector(np.asarray(center, dtype=float)),
        }
        for index, center in enumerate(centers, start=1)
    ]
    (search_dir / "initial_centers.json").write_text(
        json.dumps(jsonable(initial_centers), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    payloads = []
    for start_index, center in enumerate(centers, start=1):
        start_dir = search_dir if int(args.n_starts) == 1 else search_dir / f"start{start_index:03d}"
        payloads.append((args, str(start_dir), start_index, np.asarray(center, dtype=float), search_settings))

    try:
        start_results: list[dict] = []
        start_jobs = max(1, min(int(args.start_jobs), len(payloads), _max_process_workers()))
        if start_jobs == 1:
            for payload in payloads:
                start_results.append(run_one_cma_start_worker(payload))
        else:
            with ProcessPoolExecutor(max_workers=start_jobs) as executor:
                future_to_start = {
                    executor.submit(run_one_cma_start_worker, payload): payload[2]
                    for payload in payloads
                }
                for future in as_completed(future_to_start):
                    start_results.append(future.result())
        start_results = sorted(start_results, key=lambda item: int(item["start"]))
        finite_results = [
            item for item in start_results
            if item.get("objective") is not None and math.isfinite(float(item["objective"]))
        ]
        best = min(finite_results, key=lambda item: float(item["objective"])) if finite_results else start_results[0]
        best["all_starts"] = [
            {
                "start": item.get("start"),
                "objective": item.get("objective"),
                "params": item.get("params"),
            }
            for item in start_results
        ]
        (search_dir / "best_params.json").write_text(
            json.dumps(jsonable(best), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if tracker is not None:
            tracker.log_remaining_candidates()
            tracker.finish(best)
        progress_dirs = (
            [search_dir]
            if int(args.n_starts) == 1
            else [
                search_dir / f"start{start_index:03d}"
                for start_index in range(1, int(args.n_starts) + 1)
            ]
        )
        for progress_dir in progress_dirs:
            progress_script = PROJECT_ROOT / "f_run" / "plot_cma_es_progress.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(progress_script),
                    "--search-dir",
                    str(progress_dir),
                ],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if completed.returncode != 0:
                print(
                    f"[cma-progress] plot generation failed for {progress_dir} "
                    f"(exit={completed.returncode})",
                    file=sys.stderr,
                )
        if int(args.n_starts) > 1:
            progress_script = PROJECT_ROOT / "f_run" / "plot_cma_es_progress.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(progress_script),
                    "--search-dir",
                    str(search_dir),
                    "--combine-starts",
                ],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if completed.returncode != 0:
                print(
                    f"[cma-progress] combined plot generation failed for {search_dir} "
                    f"(exit={completed.returncode})",
                    file=sys.stderr,
                )
        print(f"[cma] best start={best.get('start')} objective={best.get('objective')}")
        print(f"[cma] saved to {search_dir}")
        return 0
    finally:
        if tracker is not None:
            tracker.end()


if __name__ == "__main__":
    raise SystemExit(main())
