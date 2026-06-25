"""CMA-ES search for liquid-only hyperparameters.

Objective:
    minimize -alpha * accuracy
             + variance_weight * accuracy_variance
             + beta * total_spikes / spike_base
             + silent_weight * silent_neuron_fraction
             - gamma * fisher_ratio_DR

The liquid is not trained. Each candidate builds the liquid, saves internal
states, then evaluates eval.py-style Mahalanobis classification from liquid
neurons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run
from d_tools.run_paths import jsonable
from f_run.run_common import build_cfg
from f_run.run_liquid import run_liquid
from f_run.run_random_neuron_accuracy import evaluate_random_neuron_accuracy


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_DIR = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
CMA_DIR = RESULTS_DIR / str(RUN_CFG.get("CMA_ES_RESULT_DIR", "cma_es_search"))


PARAMS = [
    {
        "name": "n_liq",
        "kind": "int_log10",
        "initial": 1000,
        "low": 0,
        "high": 1000,
    },
    {
        "name": "r_inh_liq",
        "kind": "logit",
        "initial": 0.20,
        "low": 0.05,
        "high": 0.50,
    },
    {
        "name": "merkel_p",
        "kind": "logit",
        "initial": 0.05,
        "low": 0.00,
        "high": 0.30,
    },
    {
        "name": "meissner_p",
        "kind": "logit",
        "initial": 0.05,
        "low": 0.00,
        "high": 0.30,
    },
    {
        "name": "merkel_gain",
        "kind": "log10",
        "initial": 0.10,
        "low": 0.01,
        "high": 1.50,
    },
    {
        "name": "meissner_gain",
        "kind": "log10",
        "initial": 0.10,
        "low": 0.01,
        "high": 1.50,
    },
    {
        "name": "RI_opt_gain",
        "kind": "log10",
        "initial": 1.0,
        "low": 0.05,
        "high": 20.0,
    },
    {
        "name": "SI_opt_gain",
        "kind": "log10",
        "initial": 1.0,
        "low": 0.05,
        "high": 20.0,
    },
    {
        "name": "USI_opt_gain",
        "kind": "log10",
        "initial": 1.0,
        "low": 0.05,
        "high": 20.0,
    },
    {
        "name": "merkel_opt_gain",
        "kind": "log10",
        "initial": 1.0,
        "low": 0.05,
        "high": 20.0,
    },
    {
        "name": "meissner_opt_gain",
        "kind": "log10",
        "initial": 1.0,
        "low": 0.05,
        "high": 20.0,
    },
    {
        "name": "rec_p_ee",
        "kind": "logit",
        "initial": 0.05,
        "low": 0.005,
        "high": 0.50,
    },
    {
        "name": "rec_p_ei",
        "kind": "logit",
        "initial": 0.05,
        "low": 0.005,
        "high": 0.50,
    },
    {
        "name": "rec_p_ie",
        "kind": "logit",
        "initial": 0.05,
        "low": 0.005,
        "high": 0.50,
    },
    {
        "name": "rec_p_ii",
        "kind": "logit",
        "initial": 0.05,
        "low": 0.005,
        "high": 0.50,
    },
    {
        "name": "rec_gain_ee",
        "kind": "log10",
        "initial": 0.10,
        "low": 0.005,
        "high": 1.00,
    },
    {
        "name": "rec_gain_ei",
        "kind": "log10",
        "initial": 0.10,
        "low": 0.005,
        "high": 1.00,
    },
    {
        "name": "rec_gain_ie",
        "kind": "log10",
        "initial": 0.10,
        "low": 0.005,
        "high": 1.00,
    },
    {
        "name": "rec_gain_ii",
        "kind": "log10",
        "initial": 0.10,
        "low": 0.005,
        "high": 1.00,
    },
    {
        "name": "lif_tau_exc",
        "kind": "log10",
        "initial": 10.0,
        "low": 2.0,
        "high": 50.0,
    },
    {
        "name": "lif_tau_inh",
        "kind": "log10",
        "initial": 10.0,
        "low": 2.0,
        "high": 50.0,
    },
    {
        "name": "lif_ref_exc",
        "kind": "log10",
        "initial": 2.0,
        "low": 0.5,
        "high": 10.0,
    },
    {
        "name": "lif_ref_inh",
        "kind": "log10",
        "initial": 2.0,
        "low": 0.5,
        "high": 10.0,
    },
    {
        "name": "lif_bias",
        "kind": "linear",
        "initial": -65.0,
        "low": -70.0,
        "high": -45.0,
    },
    {
        "name": "syn_tau_r",
        "kind": "log10",
        "initial": 2.0,
        "low": 0.5,
        "high": 10.0,
    },
    {
        "name": "syn_tau_d",
        "kind": "log10",
        "initial": 30.0,
        "low": 5.0,
        "high": 100.0,
    },
]


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def encode_value(value: float, spec: dict) -> float:
    if spec["kind"] in {"log10", "int_log10"}:
        return math.log10(float(value))
    if spec["kind"] == "logit":
        low = float(spec["low"])
        high = float(spec["high"])
        unit = (float(value) - low) / (high - low)
        return _logit(unit)
    return float(value)


def decode_value(raw: float, spec: dict):
    low = float(spec["low"])
    high = float(spec["high"])
    if spec["kind"] == "log10":
        value = 10.0 ** float(raw)
    elif spec["kind"] == "int_log10":
        value = round(10.0 ** float(raw))
    elif spec["kind"] == "logit":
        value = low + (high - low) * _sigmoid(float(raw))
    else:
        value = float(raw)
    value = min(max(value, low), high)
    if spec["kind"] == "int_log10":
        return int(round(value))
    return float(value)


def initial_vector() -> np.ndarray:
    return np.asarray([encode_value(spec["initial"], spec) for spec in PARAMS], dtype=float)


def decode_vector(x: np.ndarray) -> dict[str, float]:
    return {
        spec["name"]: decode_value(float(x[index]), spec)
        for index, spec in enumerate(PARAMS)
    }


def apply_liquid_params(cfg: dict, params: dict[str, float]) -> dict:
    cfg = deepcopy(cfg)
    net = cfg["network"]
    net["N_liq"] = [int(params.get("n_liq", 1000))]
    net["r_inh_liq"] = float(params.get("r_inh_liq", net.get("r_inh_liq", 0.2)))

    lif_cfg = cfg["neuron_models"]["LIF"]
    lif_cfg["tau_exc"] = params["lif_tau_exc"]
    lif_cfg["tau_inh"] = params["lif_tau_inh"]
    lif_cfg["ref_exc"] = params["lif_ref_exc"]
    lif_cfg["ref_inh"] = params["lif_ref_inh"]
    lif_cfg["bias"] = params["lif_bias"]
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
        filter_name = str(route_key[1]).lower()
        p_key = f"{filter_name}_p"
        gain_key = f"{filter_name}_gain"
        legacy_sensor_p_key = f"sensor{sensor_index}_p"
        legacy_filter_p_key = f"sensor{sensor_index}_{filter_name}_p"
        legacy_scale_key = f"sensor{sensor_index}_{filter_name}_scale"
        scale = params.get(gain_key, params.get(legacy_scale_key, 0.1))
        probability = params.get(
            p_key,
            params.get(
                legacy_sensor_p_key,
                params.get(legacy_filter_p_key, params.get("input_p", 0.05)),
            ),
        )
        for layer_cfg in route_cfg.get("layers", {}).values():
            layer_cfg["p"] = {"E": probability, "I": probability}
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
        "spike_counts_by_material": material_counts,
    }


def collect_neuron_activity_metrics(
    internal_state_dir: Path,
    *,
    max_samples_per_class: int | None = None,
) -> dict:
    active_any: np.ndarray | None = None
    n_trials = 0
    n_mismatched = 0
    for material_dir in sorted(Path(internal_state_dir).iterdir()):
        if not material_dir.is_dir():
            continue
        manifests = sorted(material_dir.glob("*_internal_state_manifest.json"))
        if max_samples_per_class is not None:
            manifests = manifests[: int(max_samples_per_class)]
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
            trial_active = np.any(np.isfinite(x_state) & (x_state != 0), axis=1)
            if active_any is None:
                active_any = trial_active.astype(bool, copy=True)
            elif active_any.shape == trial_active.shape:
                active_any |= trial_active
            else:
                n_mismatched += 1
                n = min(active_any.size, trial_active.size)
                active_any[:n] |= trial_active[:n]
            n_trials += 1

    if active_any is None or active_any.size == 0:
        raise FileNotFoundError(
            f"No combined internal-state npz files found under {internal_state_dir}"
        )

    silent = ~active_any
    silent_count = int(np.count_nonzero(silent))
    total = int(active_any.size)
    return {
        "n_activity_trials": int(n_trials),
        "n_activity_mismatched_shapes": int(n_mismatched),
        "active_neuron_count": int(np.count_nonzero(active_any)),
        "silent_neuron_count": silent_count,
        "total_neuron_count": total,
        "silent_neuron_fraction": float(silent_count / max(total, 1)),
    }


def _score(
    metrics: dict,
    *,
    metric: str,
    alpha: float,
    beta: float,
    gamma: float,
    spike_base: float,
    variance_weight: float,
    silent_weight: float,
) -> float:
    acc = float(metrics[f"{metric}_mean"])
    acc_std = float(metrics[f"{metric}_std"])
    acc_variance = acc_std * acc_std
    spikes = float(metrics["mean_total_spikes_per_trial"])
    fisher = float(metrics.get("fisher_ratio_DR_mean", 0.0))
    silent_fraction = float(metrics.get("silent_neuron_fraction", 0.0))
    if not math.isfinite(fisher):
        fisher = 0.0
    if not math.isfinite(spikes):
        raise ValueError(f"mean_total_spikes_per_trial is not finite: {spikes}")
    if float(spike_base) <= 0:
        raise ValueError(f"spike_base must be positive, got {spike_base}")
    return float(
        -float(alpha) * acc
        + float(variance_weight) * acc_variance
        + float(beta) * (spikes / float(spike_base))
        + float(silent_weight) * silent_fraction
        - float(gamma) * fisher
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
        n_neurons=int(args.neurons),
        n_repeats=int(args.repeats),
        n_folds=int(args.folds),
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
        )
    )
    metrics["accuracy8_overall_variance"] = float(
        metrics["accuracy8_overall_std"] ** 2
    )
    metrics["accuracy3_overall_variance"] = float(
        metrics["accuracy3_overall_std"] ** 2
    )
    metrics["spike_base"] = float(args.spike_base)
    metrics["spike_ratio"] = float(
        metrics["mean_total_spikes_per_trial"] / float(args.spike_base)
    )
    objective = _score(
        metrics,
        metric=str(args.metric),
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        spike_base=float(args.spike_base),
        variance_weight=float(args.variance_weight),
        silent_weight=float(args.silent_weight),
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
    rng: np.random.Generator,
    evaluate_population,
) -> tuple[np.ndarray, float]:
    n_dim = x0.size
    mu = max(1, population_size // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights * weights)

    cc = (4 + mueff / n_dim) / (n_dim + 4 + 2 * mueff / n_dim)
    cs = (mueff + 2) / (n_dim + mueff + 5)
    c1 = 2 / ((n_dim + 1.3) ** 2 + mueff)
    cmu = min(
        1 - c1,
        2 * (mueff - 2 + 1 / mueff) / ((n_dim + 2) ** 2 + mueff),
    )
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n_dim + 1)) - 1) + cs
    chi_n = math.sqrt(n_dim) * (1 - 1 / (4 * n_dim) + 1 / (21 * n_dim * n_dim))

    mean = np.asarray(x0, dtype=float).copy()
    sigma = float(sigma0)
    pc = np.zeros(n_dim)
    ps = np.zeros(n_dim)
    b = np.eye(n_dim)
    d = np.ones(n_dim)
    c = np.eye(n_dim)
    invsqrt_c = np.eye(n_dim)
    eigeneval = 0
    counteval = 0
    best_x = mean.copy()
    best_obj = float("inf")

    for generation in range(1, int(generations) + 1):
        arz = rng.normal(size=(population_size, n_dim))
        ary = arz @ (b * d).T
        arx = mean + sigma * ary
        results = evaluate_population(generation, arx)
        for result in results:
            obj = float(result["objective"])
            if obj < best_obj:
                best_obj = obj
                best_x = np.asarray(arx[int(result["candidate"]) - 1], dtype=float).copy()

        order = np.argsort([float(item["objective"]) for item in results])
        old_mean = mean.copy()
        x_sel = arx[order[:mu]]
        z_sel = arz[order[:mu]]
        mean = np.sum(x_sel * weights[:, None], axis=0)
        z_mean = np.sum(z_sel * weights[:, None], axis=0)
        y_mean = (mean - old_mean) / sigma

        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrt_c @ y_mean)
        hsig = float(
            np.linalg.norm(ps)
            / math.sqrt(1 - (1 - cs) ** (2 * (generation + 1)))
            / chi_n
            < (1.4 + 2 / (n_dim + 1))
        )
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_mean
        artmp = (x_sel - old_mean) / sigma
        c = (
            (1 - c1 - cmu) * c
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * c)
            + cmu * sum(weights[i] * np.outer(artmp[i], artmp[i]) for i in range(mu))
        )
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        counteval += population_size

        if counteval - eigeneval > population_size / (c1 + cmu) / n_dim / 10:
            eigeneval = counteval
            c = np.triu(c) + np.triu(c, 1).T
            vals, vecs = np.linalg.eigh(c)
            d = np.sqrt(np.maximum(vals, 1e-30))
            b = vecs
            invsqrt_c = b @ np.diag(1 / d) @ b.T
    return best_x, best_obj


def build_search_settings(args: argparse.Namespace) -> dict:
    return {
        "objective": (
            "-alpha*A + variance_weight*Var(A) "
            "+ beta*(S/S_base) + silent_weight*R_silent - gamma*F"
        ),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "variance_weight": float(args.variance_weight),
        "silent_weight": float(args.silent_weight),
        "spike_base": float(args.spike_base),
        "spike_definition": "mean total liquid spikes per trial over all evaluated trials",
        "metric": str(args.metric),
        "evaluation_neurons": "all" if int(args.neurons) <= 0 else int(args.neurons),
        "samples_per_class": int(args.samples_per_class),
        "repeats": int(args.repeats),
        "folds": int(args.folds),
        "T_n_ms": float(args.t_n_ms),
        "brian_codegen_target": str(args.brian_codegen_target),
        "parameters": PARAMS,
        "n_starts": int(args.n_starts),
        "start_spread": float(args.start_spread),
        "start_jobs": int(args.start_jobs),
    }


def make_initial_centers(args: argparse.Namespace) -> list[np.ndarray]:
    base_x0 = initial_vector()
    rng = np.random.default_rng(int(args.seed))
    centers = [base_x0.copy()]
    for _ in range(2, int(args.n_starts) + 1):
        centers.append(base_x0 + float(args.start_spread) * rng.normal(size=base_x0.size))
    return centers


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
    (search_dir / "initial_center.json").write_text(
        json.dumps(
            jsonable(
                {
                    "start": int(start_index),
                    "raw_x0": np.asarray(x0, dtype=float).tolist(),
                    "params": decode_vector(np.asarray(x0, dtype=float)),
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
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
        for key in (
            "accuracy8_overall_mean",
            "accuracy8_overall_std",
            "accuracy8_overall_variance",
            "accuracy3_overall_mean",
            "accuracy3_overall_std",
            "accuracy3_overall_variance",
            "fisher_ratio_DR_mean",
            "mean_total_spikes_per_trial",
            "std_total_spikes_per_trial",
            "total_spikes_all_trials",
            "spike_base",
            "spike_ratio",
            "active_neuron_count",
            "silent_neuron_count",
            "total_neuron_count",
            "silent_neuron_fraction",
        ):
            row[key] = metrics.get(key)
        for key, value in result.get("params", {}).items():
            row[key] = value

        exists = results_csv.exists()
        with results_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
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
            f"spikes={row.get('mean_total_spikes_per_trial')} "
            f"silent={row.get('silent_neuron_fraction')} "
            f"DR={row.get('fisher_ratio_DR_mean')}"
        )

    def evaluate_population(generation: int, arx: np.ndarray) -> list[dict]:
        payloads = [
            (np.asarray(x, dtype=float), int(generation), int(candidate_index), str(search_dir), args)
            for candidate_index, x in enumerate(arx, start=1)
        ]
        results: list[dict] = []
        jobs = max(1, min(int(args.jobs), len(payloads)))
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

    rng = np.random.default_rng(int(args.seed) + 100000 * int(start_index))
    best_x, best_obj = cma_es_ask_tell(
        x0=np.asarray(x0, dtype=float),
        sigma0=float(args.sigma0),
        generations=int(args.generations),
        population_size=int(args.population_size),
        rng=rng,
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
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument(
        "--brian-codegen-target",
        choices=("numpy", "cython"),
        default="numpy",
        help=(
            "Brian2 backend. NumPy avoids shared Cython DLL collisions during "
            "parallel CMA-ES on Windows."
        ),
    )
    parser.add_argument("--sigma0", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-starts",
        type=int,
        default=1,
        help="Number of independent CMA-ES starts. start001 uses the base initial center.",
    )
    parser.add_argument(
        "--start-spread",
        type=float,
        default=None,
        help=(
            "Stddev for scattering start002+ initial centers in encoded CMA coordinates. "
            "Default: 0 for one start, otherwise --sigma0."
        ),
    )
    parser.add_argument(
        "--start-jobs",
        type=int,
        default=1,
        help="Number of independent CMA-ES starts to run in parallel.",
    )
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument(
        "--neurons",
        type=int,
        default=0,
        help="Number of liquid neurons for accuracy evaluation. Use 0 for all neurons.",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--t-n-ms", type=float, default=25.0)
    parser.add_argument("--internal-state-bin-ms", type=float, default=1.0)
    parser.add_argument("--metric", choices=("accuracy8_overall", "accuracy3_overall"), default="accuracy8_overall")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--spike-base", type=float, default=5000.0)
    parser.add_argument(
        "--variance-weight",
        type=float,
        default=100.0,
        help=(
            "Penalty weight for accuracy variance across repeats/folds. "
            "Use 0 to ignore variance."
        ),
    )
    parser.add_argument(
        "--silent-weight",
        type=float,
        default=100.0,
        help=(
            "Penalty weight for the fraction of liquid neurons that never become "
            "active over all evaluated materials/samples. Use 0 to ignore it."
        ),
    )
    parser.add_argument("--search-name", type=str, default="liquid_accuracy_spikes_fisher")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if float(args.spike_base) <= 0:
        raise ValueError("--spike-base must be positive")
    if int(args.n_starts) <= 0:
        raise ValueError("--n-starts must be positive")
    if int(args.start_jobs) <= 0:
        raise ValueError("--start-jobs must be positive")
    if args.start_spread is None:
        args.start_spread = 0.0 if int(args.n_starts) == 1 else float(args.sigma0)
    if float(args.start_spread) < 0:
        raise ValueError("--start-spread must be non-negative")
    if args.jobs is None:
        args.jobs = int(args.population_size)

    search_dir = CMA_DIR / args.search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    search_settings = build_search_settings(args)
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

    start_results: list[dict] = []
    start_jobs = max(1, min(int(args.start_jobs), len(payloads)))
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
    print(f"[cma] best start={best.get('start')} objective={best.get('objective')}")
    print(f"[cma] saved to {search_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
