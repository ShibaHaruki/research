"""実験結果を書き込まず、NumPy/Cythonの初期状態・発火・学習結果を厳密に比較する。"""

from copy import deepcopy
from time import perf_counter

import numpy as np


def _assert_exact(reference, actual, label):
    """許容誤差を使わず一致を判定し、相違があれば比較を中止する。"""
    if not np.array_equal(reference, actual):
        detail = f"shape={reference.shape}/{actual.shape}"
        if reference.shape == actual.shape and reference.size:
            detail += f", max_abs_difference={np.max(np.abs(reference - actual)):.17g}"
        raise AssertionError(f"Backend mismatch: {label}; {detail}")


def _initial_arrays(objects):
    """結合順序と初期重み、ニューロンの初期電位をコピーして比較する。"""
    values = []
    for synapse in objects["S_all"]:
        values.extend(np.asarray(v).copy() for v in (synapse.i[:], synapse.j[:], synapse.w[:]))
    for layer in objects["G_liq"]:
        values.extend((np.asarray(layer.exc.v[:]).copy(), np.asarray(layer.inh.v[:]).copy()))
    values.extend(np.asarray(group.v[:]).copy() for group in objects["G_out"])
    return values


def validate_backend_equivalence(cfg, trials, *, compare_packing=False):
    """同一設定・試行順で両方式を実行する。完全一致しなければAssertionError。

    初回のCythonコンパイルも含む時間を表示する。既存の学習結果は変更しない。
    検証した環境・設定・試行に対する確認であり、未知の条件の一致を保証しない。
    """
    from brian2 import prefs
    from a_experiment.experiment_sets import RUNTIME_BACKEND_SETTINGS
    from f_run.common import build_runtime, run_trial

    trials = list(trials)
    if not trials:
        raise ValueError("At least one material/trial pair is required")
    previous_target = RUNTIME_BACKEND_SETTINGS["BRIAN_CODEGEN_TARGET"]
    previous_packing = RUNTIME_BACKEND_SETTINGS.get("PACK_READOUT_GROUPS", False)
    previous_preference = prefs.codegen.target
    reference_initial, reference_trials = None, []
    variants = (
        [("separate", previous_target, False), ("packed", previous_target, True)]
        if compare_packing else
        [("numpy", "numpy", previous_packing), ("cython", "cython", previous_packing)]
    )
    try:
        for variant_index, (label, target, packing) in enumerate(variants):
            RUNTIME_BACKEND_SETTINGS["BRIAN_CODEGEN_TARGET"] = target
            RUNTIME_BACKEND_SETTINGS["PACK_READOUT_GROUPS"] = packing
            objects = build_runtime(deepcopy(cfg), learning=True)
            initial = _initial_arrays(objects)
            if reference_initial is None:
                reference_initial = initial
            else:
                if len(initial) != len(reference_initial):
                    raise AssertionError("Backend mismatch: network structure")
                for index, (a, b) in enumerate(zip(reference_initial, initial)):
                    _assert_exact(a, b, f"initial array {index}")
            for index, (material, sid) in enumerate(trials):
                started = perf_counter()
                result = run_trial(
                    objects, cfg, material, sid, record=True, record_liquid=True,
                    training_debug=True, record_bin_ms=1.0,
                )
                arrays = {}
                for population in ("liquid", "output"):
                    for layer, data in enumerate(result[population]):
                        for key in ("spike_times_ms", "spike_neuron_ids", "counts", "voltage_mV"):
                            arrays[f"{population}/{layer}/{key}"] = data[key].copy()
                for synapse_index, synapse in enumerate(objects["S_out"]):
                    arrays[f"weights/{synapse_index}"] = np.asarray(synapse.w[:]).copy()
                if variant_index == 0:
                    reference_trials.append(arrays)
                else:
                    for key, values in arrays.items():
                        _assert_exact(reference_trials[index][key], values, f"{material}/{sid}/{key}")
                    # 指定だけでなく、実際の時間発展にCythonが使用されたことを確認する。
                    code_objects = [co for obj in objects["net"].sorted_objects for co in obj.code_objects]
                    if target == "cython" and not any(co.__class__.__name__ == "CythonCodeObject" for co in code_objects):
                        raise AssertionError("Cython code was not executed")
                print(f"[{label}] {material} trial {sid}: {perf_counter() - started:.3f} s", flush=True)
        print(f"PASS: exact spikes, recorded voltages and weights for {len(trials)} trials", flush=True)
    finally:
        RUNTIME_BACKEND_SETTINGS["BRIAN_CODEGEN_TARGET"] = previous_target
        RUNTIME_BACKEND_SETTINGS["PACK_READOUT_GROUPS"] = previous_packing
        prefs.codegen.target = previous_preference


def main():
    """別の実行環境でも、指定した学習則・時定数・試行順で厳密比較を実行する。"""
    import argparse
    from f_run.common import build_cfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", choices=("off", "STDP", "T_STDP", "SRDP", "BCPNN"), default="STDP")
    parser.add_argument("--taus", nargs="+", type=float, default=[5, 20, 230, 500])
    parser.add_argument("--trials", nargs="+", type=int, default=[203])
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--packing", action="store_true", help="現在のbackendで従来方式と一括方式を厳密比較")
    parser.add_argument("--all-taus", action="store_true", help="STDPの全100候補を比較")
    parser.add_argument("--duration-ms", type=int, default=500, help="短い診断では25 msの倍数を指定")
    args = parser.parse_args()
    if args.passes < 1 or any(tau <= 0 or not np.isfinite(tau) for tau in args.taus):
        parser.error("passes and time constants must be positive")
    if any(sid < 1 or sid > 324 for sid in args.trials):
        parser.error("trial IDs must be from 1 to 324")
    if not 25 <= args.duration_ms <= 500 or args.duration_ms % 25:
        parser.error("duration-ms must be a multiple of 25 from 25 to 500")
    if args.all_taus:
        args.taus = list(range(5, 501, 5))
    cfg = build_cfg(network_preset="off")
    cfg["common"]["BASE_SEED"] = args.seed
    cfg["common"]["SLICE_END"] = cfg["common"]["SLICE_START"] + round(
        args.duration_ms / (float(cfg["common"]["dt_s"]) * 1000)
    )
    count = len(args.taus) if args.rule == "STDP" else 1
    for key in ("N_out", "p_liq_to_out_pairs", "gain_liq_to_out_pairs"):
        cfg["network"][key] = [deepcopy(cfg["network"][key][0]) for _ in range(count)]
    cfg["network"]["learning_rule_liq_to_out"] = [args.rule] * count
    cfg["models"]["LEARNING_RULE_MODEL"] = args.rule
    cfg["network"]["OUTPUT_LEARNING_PARAMETERS"] = (
        [{"tau_plus": tau, "tau_minus": tau} for tau in args.taus]
        if args.rule == "STDP" else [{}]
    )
    trials = [(material, sid) for _ in range(args.passes) for sid in args.trials
              for material in cfg["training"]["TRAINING_MAT"]]
    validate_backend_equivalence(cfg, trials, compare_packing=args.packing)


if __name__ == "__main__":
    main()
