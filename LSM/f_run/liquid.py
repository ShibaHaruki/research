"""出力層を作らず、offと同じ試行分割でLiquidの記録・debug・分類評価を行う。"""

from copy import deepcopy
import os
import numpy as np
from brian2 import start_scope, defaultclock, ms, seed

from a_experiment.paths import liquid_test_directory
from a_experiment.experiment_sets import (
    trial_duration_ms,
    recording_bin_ms,
    TRAINING_SETS,
    TEST_SETS,
)
from a_experiment.input_processing import filtered_trial
from b_network.build_network import make_liquid_network
from d_tools.debug import execute_trial, run_material_debug
from d_tools.execution_control import exclusive_execution, assert_recording_available
from d_tools.parameter_recording import snapshot, parameter_records_equal
from d_tools.progress import progress_items
from d_tools.recording_io import save_record, read_record
from d_tools.weight_export import save_weights
from d_tools.evaluation import classify_dataset
from f_run.common import build_cfg, build_network_cfg, configure_brian_runtime, sample_split, first


def liquid_configuration(cfg=None):
    """入力・Liquidの条件を維持し、出力層と出力結合を空にする。"""
    cfg = deepcopy(build_cfg(network_preset="off") if cfg is None else cfg)
    cfg["network"].update(
        output_enabled=False,
        N_out=[],
        learning_rule_liq_to_out=[],
        OUTPUT_LEARNING_PARAMETERS=[],
        p_liq_to_out_pairs=[],
        gain_liq_to_out_pairs=[],
    )
    for key in ("OUTPUT_SEED", "OUTPUT_SEEDS"):
        cfg["network"].pop(key, None)
    cfg["models"]["LEARNING_RULE_MODEL"] = "off"
    cfg["run"].setdefault("DATA_SPLIT_REP", 1)
    return cfg


def build_liquid_runtime(cfg):
    """共通のネットワーク構築処理で入力とLiquidだけを作る。"""
    start_scope()
    configure_brian_runtime()
    defaultclock.dt = float(cfg["common"]["dt_s"]) * 1000 * ms
    base_seed = int(cfg["common"]["BASE_SEED"])
    seed(base_seed)
    np.random.seed(base_seed)
    net_cfg = build_network_cfg(cfg)
    objects = make_liquid_network(
        net_cfg,
        len(net_cfg["INPUT_ROWS"]),
        np.random.default_rng(base_seed),
        cfg["run"],
        common_cfg=cfg["common"],
    )
    if objects["G_out"] or objects["S_out"]:
        raise AssertionError("Liquid-only runs must not create readout neurons or synapses")
    objects["net"].store("liquid_initial")
    return objects


def _evaluation_ids(objects, population):
    """層順にE/Iを連結した記録上で、分類対象のニューロン番号を求める。"""
    ids, offset = [], 0
    for layer in objects["G_liq"]:
        count = len(layer) if population == "all" else len(layer.exc)
        ids.extend(range(offset, offset + count))
        offset += len(layer)
    return ids


@exclusive_execution(
    lambda rep, cfg, **kwargs: liquid_test_directory(cfg, rep).parent
    / "_runtime_cache"
    / "run.lock"
)
def run_liquid_test_eval(
    rep, cfg, *, resume=False, record_only=False, population="E", trial_ids=None
):
    """Liquid全ニューロンを記録し、既定ではEニューロンの素材分類も保存する。"""
    if population not in ("E", "all"):
        raise ValueError("Evaluation population must be E or all")
    cfg = liquid_configuration(cfg)
    os.environ["LSM_PROGRESS_REP"] = str(rep)
    materials = list(cfg["test"]["TEST_MAT"])
    n_test = int(first(cfg["test"]["NUM_TEST_SAMPLE"]))
    bin_ms, duration_ms = recording_bin_ms(cfg), trial_duration_ms(cfg["common"])
    if (
        not materials
        or len(set(materials)) != len(materials)
        or not set(materials) <= set(TRAINING_SETS["base"]["TRAINING_MAT"])
        or not 1 <= n_test <= int(cfg["common"]["NUM_SAMPLE"])
        or not np.isclose(duration_ms / bin_ms, round(duration_ms / bin_ms))
    ):
        raise ValueError("Invalid materials, trial count or recording bin width")
    if not record_only and len(materials) != 8:
        raise ValueError(
            "The classification Excel requires all 8 materials; use --record-only for a subset"
        )
    # offとの比較では、同じsample_seq内のテスト番号と順序を使う。
    if trial_ids is None:
        reference = deepcopy(cfg)
        reference["test"]["NUM_TEST_SAMPLE"] = deepcopy(TEST_SETS["base"]["NUM_TEST_SAMPLE"])
        _, reference_ids = sample_split(reference, rep)
        if n_test <= len(reference_ids):
            trial_ids = reference_ids[:n_test]
        else:
            # 全324試行等を記録する場合は、共通の全試行順序から選ぶ。
            from a_experiment.sample_sequences import load_sample_order

            trial_ids = load_sample_order(int(cfg["common"]["BASE_SEED"]))[:n_test]
    trial_ids = np.asarray(trial_ids)
    if (
        trial_ids.ndim != 1
        or not len(trial_ids)
        or not np.issubdtype(trial_ids.dtype, np.integer)
        or len(np.unique(trial_ids)) != len(trial_ids)
        or np.any((trial_ids < 1) | (trial_ids > 324))
    ):
        raise ValueError("Trial IDs must be a nonempty unique integer list within 1..324")
    if len(trial_ids) != n_test:
        raise ValueError("Trial ID count must match NUM_TEST_SAMPLE")
    state_policy = cfg["common"]["LIQUID_STATE_POLICY"]
    if state_policy not in ("carry", "reset_each_trial"):
        raise ValueError("Invalid LIQUID_STATE_POLICY")
    if not np.isclose(duration_ms, round(duration_ms)) or not np.isclose(bin_ms, round(bin_ms)):
        raise ValueError("Liquid recording requires 1 ms bins and integer-ms evaluation bins")
    folds = int(cfg["eval"]["EVAL_N_FOLDS"])
    if not record_only and (
        folds < 2 or n_test < folds or n_test - (n_test + folds - 1) // folds < 2
    ):
        raise ValueError("Insufficient trials for evaluation; use --record-only or reduce --folds")
    output = liquid_test_directory(cfg, rep)
    fine_path = output / "rec/Liquid" / f"sliq_rec_rep{rep}.npy"
    metadata_path = output / "test_config_snapshot.json"
    network_file = output.parent.parent / "network_parameters.json"
    if network_file.exists():
        if not parameter_records_equal(read_record(network_file), snapshot(cfg)):
            raise ValueError(f"Liquid parameters differ from the saved experiment: {network_file}")
    else:
        try:
            save_record(network_file, snapshot(cfg), overwrite=False)
        except FileExistsError:
            # repを並列起動しても、先に保存された同一設定を安全に共有する。
            if not parameter_records_equal(read_record(network_file), snapshot(cfg)):
                raise ValueError(f"Liquid parameters differ: {network_file}")
    complete = fine_path.is_file() and metadata_path.is_file()
    if complete:
        if not resume:
            raise FileExistsError(f"Liquid test already exists; use --resume: {output}")
        saved = read_record(metadata_path)
        if (
            not parameter_records_equal({key: saved[key] for key in snapshot(cfg)}, snapshot(cfg))
            or saved.get("evaluation_population") != population
            or saved.get("trial_ids") != trial_ids.tolist()
        ):
            raise ValueError(f"Saved Liquid configuration differs: {output}")
    else:
        if not resume and any(
            path.exists()
            for path in (
                fine_path,
                fine_path.with_suffix(".partial.npy"),
            )
        ):
            raise FileExistsError(f"Incomplete Liquid recording exists; use --resume: {output}")
        assert_recording_available(fine_path.with_suffix(".partial.npy"))
        output.mkdir(parents=True, exist_ok=True)
        debug = cfg["common"].get("DEBUG_EACH_MATERIAL", True) and cfg["common"].get(
            "LIQUID_DEBUG_ENABLE", True
        )
        if debug:
            run_material_debug(
                materials,
                trial_id=int(cfg["common"]["DEBUG_TRIAL_ID"]),
                make_network=lambda: build_liquid_runtime(cfg),
                load_current=lambda material, sid: filtered_trial(material, sid, cfg),
                dt_ms=float(cfg["common"]["dt_s"]) * 1000,
                duration_ms=duration_ms,
                bin_ms=bin_ms,
                out_dir=output / "debug",
            )
        # debugの時計・状態は、本番の固定Liquidへ持ち込まない。
        objects = build_liquid_runtime(cfg)
        save_weights(output / "weight", objects)
        before = [np.asarray(s.w[:]).copy() for s in objects["S_all"]]
        n_neurons = sum(len(layer) for layer in objects["G_liq"])
        shape = (len(materials), n_test, n_neurons)
        fine_path.parent.mkdir(parents=True, exist_ok=True)
        fine_partial = fine_path.with_suffix(".partial.npy")
        fine = None
        try:
            fine = np.lib.format.open_memmap(
                fine_partial, mode="w+", dtype=np.uint16, shape=(*shape, round(duration_ms))
            )
            for m, material in enumerate(materials):
                for t, sid in progress_items(
                    enumerate(trial_ids),
                    total=n_test,
                    description=f"Liquid test | {material}",
                    rep=rep,
                    label=lambda item: f"trial {item[1]}",
                ):
                    if state_policy == "reset_each_trial":
                        objects["net"].restore("liquid_initial")
                    result = execute_trial(
                        objects,
                        filtered_trial(material, int(sid), cfg),
                        dt_ms=float(cfg["common"]["dt_s"]) * 1000,
                        duration_ms=duration_ms,
                        bin_ms=1.0,
                        record_output=False,
                        voltage=False,
                    )
                    counts = np.concatenate([layer["counts"] for layer in result["liquid"]])
                    if np.any(counts < 0) or np.any(counts > np.iinfo(np.uint16).max):
                        raise OverflowError("Liquid spike count does not fit uint16")
                    fine[m, t] = counts
            fine.flush()
        finally:
            if fine is not None:
                fine._mmap.close()
        if any(not np.array_equal(s.w[:], initial) for s, initial in zip(objects["S_all"], before)):
            raise AssertionError("Liquid weights changed during testing")
        fine_partial.replace(fine_path)
        np.save(output / f"test_trial_ids_rep{rep}.npy", trial_ids)
        metadata = snapshot(cfg)
        metadata.update(
            materials=materials,
            rep=rep,
            trial_ids=trial_ids.tolist(),
            bin_width_ms=1.0,
            evaluation_bin_width_ms=bin_ms,
            duration_ms=duration_ms,
            classification_population="liquid",
            evaluation_population=population,
            evaluation_neuron_ids=_evaluation_ids(objects, population),
            recording_bin_width_ms=1.0,
            output_layer=False,
            learning_during_test=False,
            state_policy=state_policy,
            recording_files={"liquid": str(fine_path.relative_to(output))},
        )
        save_record(metadata_path, metadata)
        save_record(
            output / "rec/recording.npz",
            {
                "axis_order": ["material", "trial", "neuron", "time_bin"],
                "materials": materials,
                "trial_ids": trial_ids.tolist(),
                "bin_width_ms": 1.0,
                "dtype": "uint16",
                "value": "spike count per bin",
                "output_layer": False,
                "neuron_order": "Layers concatenated; E then I within each layer",
                "state_policy": state_policy,
                "recording_files": metadata["recording_files"],
                "liquid_layers": [
                    {"n_excitatory": len(g.exc), "n_inhibitory": len(g.inh)}
                    for g in objects["G_liq"]
                ],
            },
        )
    if not record_only:
        classify_dataset(
            output,
            rep_start=rep,
            rep_end=rep,
            n_folds=int(cfg["eval"]["EVAL_N_FOLDS"]),
            T_n_list=cfg["eval"]["EVAL_TN_MS"],
            base_seed=int(cfg["eval"]["EVAL_BASE_SEED"]),
        )
    return f"[liquid rep{rep}] {'recorded' if record_only else 'recorded and evaluated'}: {output}"
