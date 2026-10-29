"""通常の学習と、同じLiquidを共有する複数出力群の学習処理。"""

from __future__ import annotations
from copy import deepcopy
from d_tools.progress import progress_items
from pathlib import Path
import numpy as np
import pandas as pd
from a_experiment.paths import (
    training_directory,
    test_directory,
    debug_population_directory,
    test_artifact_directories,
    training_weight_directory,
    trained_weight_directory,
)
from a_experiment.experiment_sets import (
    learning_rule_name,
    output_learning_rules,
    output_learning_parameters,
)
from f_run.common import validate_cfg, sample_split, build_runtime, run_trial, cfg_for_repetition
from d_tools.recording_io import encode, save_record
from d_tools.parameter_recording import (
    parameter_records_equal,
    save_parameter_snapshots,
    snapshot,
    read_training_configuration,
    save_training_configuration,
)
from d_tools.weight_export import save_weights, weight_summary
from d_tools.training_recording import TrainingRecorder


def material_cycle_schedule(materials, train_ids, epochs, rng):
    """各組に全素材を1回ずつ含め、各epochで素材ごとの全学習試行を使う。"""
    materials = list(materials)
    train_ids = np.asarray(train_ids)
    for epoch in range(1, epochs + 1):
        orders = {material: rng.permutation(train_ids) for material in materials}
        for cycle in range(len(train_ids)):
            for index in rng.permutation(len(materials)):
                material = materials[int(index)]
                yield epoch, material, int(orders[material][cycle])


def run_training(rep: int, cfg: dict) -> str:
    """repのseedを反映し、学習中の記録と初期・最終重みを保存する。"""
    cfg = cfg_for_repetition(cfg, rep)
    if learning_rule_name(cfg) == "off":
        from f_run.testing import run_off_test

        return run_off_test(rep, cfg)
    materials, n_train, _, epochs, _, _ = validate_cfg(cfg)
    save_parameter_snapshots(cfg, rep=rep)
    folder = training_directory(cfg, rep)
    final = training_weight_directory(folder, "final")
    existing = trained_weight_directory(folder, rep)
    if existing.exists():
        raise FileExistsError(f"Trained result already exists: {existing}")
    saved_cfg = read_training_configuration(folder)
    if saved_cfg is not None and not parameter_records_equal(saved_cfg, snapshot(cfg)):
        raise ValueError(
            "Existing training directory has a different configuration; choose --out-dir"
        )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "debug").mkdir(exist_ok=True)
    train_ids, test_ids = sample_split(cfg, rep)
    objects = build_runtime(cfg, learning=True)
    save_weights(training_weight_directory(folder, "initial"), objects)
    save_training_configuration(folder, cfg)
    initial = [np.asarray(s.w[:]).copy() for s in objects["S_out"]]
    schedule_rng = np.random.default_rng(
        int(cfg["common"]["BASE_SEED"]) + int(cfg["run"].get("DATA_SPLIT_REP", rep))
    )
    history = []
    schedule = [(material, int(sid)) for material in materials for sid in train_ids]
    recorder = TrainingRecorder(
        objects, [cfg], [(objects, {})], [folder], rep=rep, n_presentations=epochs * len(schedule)
    )
    with recorder:
        for epoch, material, sid in progress_items(
            material_cycle_schedule(materials, train_ids, epochs, schedule_rng),
            total=epochs * len(schedule),
            description="Training",
            rep=rep,
            label=lambda item: f"{item[1]} | trial {item[2]} | epoch {item[0]}",
        ):
            detailed = recorder.wants_debug(material)
            result = run_trial(
                objects,
                cfg,
                material,
                sid,
                record=detailed,
                record_liquid=detailed,
                training_debug=detailed,
                before_run=lambda: recorder.activate(detailed),
            )
            recorder.after_trial(material, sid, epoch, result)
            history.append(
                {"epoch": epoch, "material": material, "trial_id": sid, **weight_summary(objects)}
            )
    save_weights(final, objects)
    pd.DataFrame(history).to_csv(folder / f"training_history_rep{rep}.csv", index=False)
    delta = (
        sum(
            float(np.sum((np.asarray(s.w[:]) - before) ** 2))
            for s, before in zip(objects["S_out"], initial)
        )
        ** 0.5
    )
    save_record(
        folder / f"training_summary_rep{rep}.npz",
        {
            "rep": rep,
            "learning_rule": cfg["models"].get("LEARNING_RULE_MODEL", "off"),
            "liquid_seed": int(cfg["common"]["BASE_SEED"]),
            "output_seed": int(
                cfg["network"].get("OUTPUT_SEED", int(cfg["common"]["BASE_SEED"]) + 1)
            ),
            "state_policy": cfg["common"].get("READOUT_STATE_POLICY", "reset_each_trial"),
            "epochs": epochs,
            "n_training_presentations": len(history),
            "output_weight_change_norm": delta,
            "train_trial_ids": train_ids.tolist(),
            "test_trial_ids": test_ids.tolist(),
            "plastic_connections": "liquid E to output E only; inhibitory connections fixed",
        },
    )
    recorder.save_final_debug()
    return f"[training rep{rep}] weights saved: {final}; weight change norm={delta:.6g}"


def make_shared_training_cfg(configurations: list[dict], shared_dir: Path) -> dict:
    """単一Liquidに、条件ごとの独立した出力群を接続する設定を作る。"""
    if not configurations:
        raise ValueError("At least one output configuration is required")
    cfg = deepcopy(configurations[0])
    reference_split = sample_split(cfg, 1)
    ignored_network = {
        "N_out",
        "p_liq_to_out_pairs",
        "gain_liq_to_out_pairs",
        "OUTPUT_SEED",
        "learning_rule_liq_to_out",
        "OUTPUT_LEARNING_PARAMETERS",
    }
    base_network = {k: v for k, v in cfg["network"].items() if k not in ignored_network}
    for candidate in configurations:
        validate_cfg(candidate)
        if {k: v for k, v in candidate["models"].items() if k != "LEARNING_RULE_MODEL"} != {
            k: v for k, v in cfg["models"].items() if k != "LEARNING_RULE_MODEL"
        }:
            raise ValueError("Shared outputs must use the same neuron and synapse models")
        for section in ("common", "training", "test", "neuron_models", "synapse_models"):
            if encode(candidate[section]) != encode(cfg[section]):
                raise ValueError(f"Shared Liquid requires identical {section} settings")
        network = {k: v for k, v in candidate["network"].items() if k not in ignored_network}
        if encode(network) != encode(base_network):
            raise ValueError(
                "Shared Liquid requires identical input and recurrent network settings"
            )
        if len(candidate["network"]["N_out"]) != 1:
            raise ValueError("Each candidate must describe exactly one output group")
        if any(
            not np.array_equal(a, b) for a, b in zip(reference_split, sample_split(candidate, 1))
        ):
            raise ValueError("Shared training requires the same sample split")
    cfg["network"]["N_out"] = [c["network"]["N_out"][0] for c in configurations]
    cfg["network"]["learning_rule_liq_to_out"] = [
        output_learning_rules(c)[0] for c in configurations
    ]
    for key in ("p_liq_to_out_pairs", "gain_liq_to_out_pairs"):
        cfg["network"][key] = [deepcopy(c["network"][key][0]) for c in configurations]
    cfg["network"]["OUTPUT_SEEDS"] = [
        int(c["network"].get("OUTPUT_SEED", c["common"]["BASE_SEED"] + 1)) for c in configurations
    ]
    cfg["network"]["OUTPUT_LEARNING_PARAMETERS"] = [
        output_learning_parameters(c)[0] for c in configurations
    ]
    cfg["models"]["LEARNING_RULE_MODEL"] = learning_rule_name(cfg)
    cfg["run"]["READOUT_TRAIN_DIR"] = str(Path(shared_dir).resolve())
    return cfg


def candidate_weight_view(objects, output_index):
    """共通ネットワークの1出力群を、通常の1出力層形式で保存できるようにする。"""
    selected = [
        meta["synapses"]
        for meta in objects["output_connection_meta"]
        if meta["output_index"] == output_index
    ]
    common = objects["S_in"] + objects["S_intra"]
    names = {
        syn.name: syn.name.replace(f"_to_out{output_index+1}_", "_to_out1_") for syn in selected
    }
    return {
        "S_all": common + selected,
        "S_out": selected,
        "net": objects["net"],
        "G_in": objects["G_in"],
        "G_liq": objects["G_liq"],
        "G_out": [objects["G_out"][output_index]],
        "S_in": objects["S_in"],
        "S_intra": objects["S_intra"],
        "output_connection_meta": [
            {**meta, "output_index": 0}
            for meta in objects["output_connection_meta"]
            if meta["output_index"] == output_index
        ],
    }, names


def shared_readout_debug(rep, configurations, cfg, shared_dir, *, phase="training"):
    """各素材1回、共通Liquidと全出力群を同時にdebugして、出力群ごとに保存する。"""
    from a_experiment.input_processing import filtered_trial
    from d_tools.debug import execute_trial, save_debug_layer, share_debug_layer
    from d_tools.weight_export import load_weights, save_debug_weight_distributions

    enabled = [c["common"].get("DEBUG_EACH_MATERIAL", True) for c in configurations]
    if not any(enabled):
        return
    materials, _, _, _, bin_ms, duration = validate_cfg(cfg)
    sid = int(cfg["common"].get("DEBUG_TRIAL_ID", 203))
    reports = [[] for _ in configurations]
    for material in progress_items(
        materials,
        total=len(materials),
        description=f"Debug ({phase})",
        rep=rep,
        unit="material",
        label=str,
    ):
        objects = build_runtime(cfg, learning=False)
        load_weights(trained_weight_directory(shared_dir, rep), objects)
        before = [np.asarray(s.w[:]).copy() for s in objects["S_all"]]
        result = execute_trial(
            objects,
            filtered_trial(material, sid, cfg),
            dt_ms=float(cfg["common"]["dt_s"]) * 1000,
            duration_ms=duration,
            bin_ms=bin_ms,
            voltage=True,
        )
        if any(not np.array_equal(s.w[:], w) for s, w in zip(objects["S_all"], before)):
            raise AssertionError("Weights changed during shared debug")
        common_root = shared_dir / "debug" / f"{phase}_rep{rep}"
        liquid_folders = []
        for index, liquid in enumerate(result["liquid"]):
            common_folder = debug_population_directory(
                common_root,
                "liquid",
                material,
                layer_index=index,
                layer_count=len(result["liquid"]),
            )
            save_debug_layer(
                liquid,
                common_folder,
                title=f"{material} | Shared Liquid | Trial {sid}",
                n_excitatory=len(objects["G_liq"][index].exc),
                trial_id=sid,
                separate_pdfs=True,
            )
            liquid_folders.append(common_folder)
        for index, candidate in enumerate(configurations):
            if not enabled[index]:
                continue
            candidate_root = (
                training_directory(candidate)
                if phase == "training"
                else test_directory(training_directory(candidate))
            )
            debug_root = (
                candidate_root / "debug"
                if phase == "training"
                else test_artifact_directories(candidate_root, cfg=candidate, rep=rep)["debug"]
            )
            if not reports[index]:
                view, names = candidate_weight_view(objects, index)
                save_debug_weight_distributions(debug_root, view, name_overrides=names)
            folder = debug_population_directory(debug_root, "readout", material)
            data = result["output"][index]
            save_debug_layer(
                data,
                folder,
                title=f"{material} | Output group {index+1} | Trial {sid}",
                n_excitatory=len(objects["G_out"][index]),
                trial_id=sid,
                separate_pdfs=True,
            )
            for liquid_index, common_folder in enumerate(liquid_folders):
                destination = debug_population_directory(
                    debug_root,
                    "liquid",
                    material,
                    layer_index=liquid_index,
                    layer_count=len(liquid_folders),
                )
                share_debug_layer(common_folder, destination)
            reports[index].append(
                {
                    "material": material,
                    "trial_id": sid,
                    "output_spikes": [int(data["counts"].sum())],
                    "weights_unchanged": True,
                    "shared_liquid_debug": [str(folder) for folder in liquid_folders],
                }
            )
    for candidate, report, save_report in zip(configurations, reports, enabled):
        if save_report:
            candidate_root = (
                training_directory(candidate)
                if phase == "training"
                else test_directory(training_directory(candidate))
            )
            debug_root = (
                candidate_root / "debug"
                if phase == "training"
                else test_artifact_directories(candidate_root, cfg=candidate, rep=rep)["debug"]
            )
            save_record(debug_root / "summary.npz", report)


def run_shared_training(rep: int, configurations: list[dict], shared_dir: Path) -> str:
    """1つのLiquidから全候補へ同時にスパイクを送り、1回の試行計算で学習する。"""
    cfg = make_shared_training_cfg(configurations, shared_dir)
    shared_dir = Path(shared_dir)
    materials, _, _, epochs, _, _ = validate_cfg(cfg)
    for candidate in configurations:
        save_parameter_snapshots(candidate, rep=rep)
    train_ids, test_ids = sample_split(cfg, rep)
    shared_dir.mkdir(parents=True, exist_ok=True)
    save_record(shared_dir / "config_snapshot.json", snapshot(cfg))
    for candidate in configurations:
        folder = training_directory(candidate)
        existing = trained_weight_directory(folder, rep)
        if existing.exists():
            raise FileExistsError(f"Trained result already exists: {existing}")
        saved_cfg = read_training_configuration(folder)
        if saved_cfg is not None and not parameter_records_equal(saved_cfg, snapshot(candidate)):
            raise ValueError(f"Existing candidate configuration differs: {folder}")
        (folder / "debug").mkdir(parents=True, exist_ok=True)
    objects = build_runtime(cfg, learning=True)
    views = [candidate_weight_view(objects, i) for i in range(len(configurations))]
    initial = [[np.asarray(s.w[:]).copy() for s in view["S_out"]] for view, _ in views]
    for candidate, (view, names) in zip(configurations, views):
        save_weights(
            training_weight_directory(training_directory(candidate), "initial"),
            view,
            name_overrides=names,
        )
        save_training_configuration(training_directory(candidate), candidate)
    histories = [[] for _ in configurations]
    schedule_rng = np.random.default_rng(
        int(cfg["common"]["BASE_SEED"]) + int(cfg["run"].get("DATA_SPLIT_REP", rep))
    )
    schedule = [(material, int(sid)) for material in materials for sid in train_ids]
    recorder = TrainingRecorder(
        objects,
        configurations,
        views,
        [training_directory(c) for c in configurations],
        rep=rep,
        n_presentations=epochs * len(schedule),
    )
    with recorder:
        for epoch, material, sid in progress_items(
            material_cycle_schedule(materials, train_ids, epochs, schedule_rng),
            total=epochs * len(schedule),
            description="Training",
            rep=rep,
            label=lambda item: f"{item[1]} | trial {item[2]} | epoch {item[0]}",
        ):
            detailed = recorder.wants_debug(material)
            result = run_trial(
                objects,
                cfg,
                material,
                sid,
                record=detailed,
                record_liquid=detailed,
                training_debug=detailed,
                before_run=lambda: recorder.activate(detailed),
            )
            recorder.after_trial(material, sid, epoch, result)
            for history, (view, _) in zip(histories, views):
                history.append(
                    {"epoch": epoch, "material": material, "trial_id": sid, **weight_summary(view)}
                )
    save_weights(training_weight_directory(shared_dir, "final"), objects)
    for candidate, history, before, (view, names) in zip(configurations, histories, initial, views):
        folder = training_directory(candidate)
        save_weights(training_weight_directory(folder, "final"), view, name_overrides=names)
        pd.DataFrame(history).to_csv(folder / f"training_history_rep{rep}.csv", index=False)
        delta = (
            sum(float(np.sum((np.asarray(s.w[:]) - w) ** 2)) for s, w in zip(view["S_out"], before))
            ** 0.5
        )
        save_record(
            folder / f"training_summary_rep{rep}.npz",
            {
                "rep": rep,
                "learning_rule": candidate["models"]["LEARNING_RULE_MODEL"],
                "liquid_seed": int(cfg["common"]["BASE_SEED"]),
                "output_seed": int(
                    candidate["network"].get("OUTPUT_SEED", cfg["common"]["BASE_SEED"] + 1)
                ),
                "state_policy": cfg["common"].get("READOUT_STATE_POLICY", "reset_each_trial"),
                "epochs": epochs,
                "n_training_presentations": len(history),
                "output_weight_change_norm": delta,
                "train_trial_ids": train_ids.tolist(),
                "test_trial_ids": test_ids.tolist(),
                "plastic_connections": "liquid E to output E only; inhibitory connections fixed",
                "shared_liquid": True,
                "shared_output_groups": len(configurations),
                "shared_training_directory": str(shared_dir),
                "shared_training_log": str(shared_dir / "training.log"),
            },
        )
    recorder.save_final_debug()
    return f"[shared training rep{rep}] {len(configurations)} output groups saved; Liquid computed once per trial"
