"""固定重みでのテスト、発火記録、保存済み学習結果の読み込み。"""

from __future__ import annotations
from copy import deepcopy
from d_tools.progress import progress_items
from pathlib import Path
from d_tools.parameter_recording import has_training_configuration
from d_tools.recording_io import read_record
from a_experiment.experiment_sets import learning_rule_name
import numpy as np
from a_experiment.paths import (
    training_directory,
    test_directory,
    off_repetition_directory,
    off_test_directory,
    off_test_artifact_directories,
    off_recording_files,
    test_artifact_directories,
    test_recording_files,
    trained_weight_directory,
)
from f_run.common import load_cfg, build_runtime, debug_readout_materials, run_trial, validate_cfg
from d_tools.recording_io import save_record
from d_tools.parameter_recording import parameter_records_equal, snapshot
from d_tools.weight_export import load_weights, save_weights
from f_run.common import build_cfg, sample_split
from f_run.common import cfg_for_repetition

training_output_dir_for_cfg = training_directory
test_output_dir_for_training_dir = test_directory


def run_off_test(rep: int, cfg: dict) -> str:
    """学習offでは初期重みを固定したままテストし、off/rep番号/testへ保存する。"""
    cfg = cfg_for_repetition(cfg, rep)
    if learning_rule_name(cfg) != "off":
        raise ValueError("run_off_test requires learning rule off")
    from d_tools.parameter_recording import save_parameter_snapshots

    save_parameter_snapshots(cfg, rep=rep)
    materials, _, n_test, _, bin_ms, duration = validate_cfg(cfg)
    repetition = off_repetition_directory(cfg, rep)
    output = off_test_directory(cfg, rep)
    folders = off_test_artifact_directories(cfg, rep)
    record_files = off_recording_files(cfg, rep)
    record_bin_ms = 1.0
    aggregation = round(bin_ms / record_bin_ms)
    if (
        aggregation < 1
        or not np.isclose(bin_ms / record_bin_ms, aggregation)
        or not np.isclose(duration / record_bin_ms, round(duration / record_bin_ms))
    ):
        raise ValueError(
            "Off recording requires 1 ms bins; evaluation bins must be multiples of 1 ms"
        )
    file = record_files["readout"]
    if file.exists() or any(path.exists() for path in record_files.values()):
        raise FileExistsError(f"Off test result already exists: {file}")
    config_file = repetition / "config_snapshot.json"
    if config_file.exists() and not parameter_records_equal(
        read_record(config_file), snapshot(cfg)
    ):
        raise ValueError(f"Existing off repetition has a different configuration: {repetition}")
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    save_record(config_file, snapshot(cfg))
    _, test_ids = sample_split(cfg, rep)
    objects = build_runtime(cfg, learning=False)
    return record_fixed_test(rep, cfg, objects, output, test_ids)


def record_fixed_test(
    rep,
    cfg,
    objects,
    output,
    test_ids,
    *,
    source_training_directory=None,
):
    """off・学習ありで共通の固定重みテスト。両層の1 ms記録とdebugを保存する。"""
    materials, _, n_test, _, bin_ms, duration = validate_cfg(cfg)
    output = Path(output)
    off = learning_rule_name(cfg) == "off"
    folders = test_artifact_directories(output, cfg=None if off else cfg, rep=rep)
    record_files = test_recording_files(output, rep)
    record_bin_ms = 1.0
    aggregation = round(bin_ms / record_bin_ms)
    if (
        aggregation < 1
        or not np.isclose(bin_ms, aggregation)
        or not np.isclose(duration, round(duration))
    ):
        raise ValueError(
            "Test recording requires evaluation bins and duration to be multiples of 1 ms"
        )
    file = record_files["readout"]
    if file.exists() or any(path.exists() for path in record_files.values()):
        raise FileExistsError(f"Test result already exists: {file}")
    for directory in folders.values():
        directory.mkdir(parents=True, exist_ok=True)
    save_weights(folders["weight"], objects)
    if cfg["common"].get("DEBUG_EACH_MATERIAL", True):
        print(f"[debug rep{rep}] checking each material before testing", flush=True)
        debug_readout_materials(cfg, folders["weight"], folders["debug"])
        # debugが使用した時計・状態を持ち込まず、同じ固定重みで本番を開始する。
        objects = build_runtime(cfg, learning=False)
        load_weights(folders["weight"], objects)
    before = [np.asarray(s.w[:]).copy() for s in objects["S_all"]]
    recordings = {}
    partials = {}
    try:
        for population, groups in [("liquid", objects["G_liq"]), ("readout", objects["G_out"])]:
            path = record_files[population]
            path.parent.mkdir(parents=True, exist_ok=True)
            partials[population] = path.with_suffix(".partial.npy")
            recordings[population] = np.lib.format.open_memmap(
                partials[population],
                mode="w+",
                dtype=np.uint16,
                shape=(
                    len(materials),
                    n_test,
                    sum(len(g) for g in groups),
                    round(duration / record_bin_ms),
                ),
            )
        for index, material in enumerate(materials):
            for sample, sid in progress_items(
                enumerate(test_ids),
                total=n_test,
                description=f"Test | {material}",
                rep=rep,
                label=lambda item: f"trial {item[1]}",
            ):
                recorded = run_trial(
                    objects,
                    cfg,
                    material,
                    int(sid),
                    record=True,
                    record_liquid=True,
                    record_bin_ms=record_bin_ms,
                )
                for population, values in recorded.items():
                    if np.any(values < 0) or np.any(values > np.iinfo(np.uint16).max):
                        raise OverflowError("Spike count cannot be represented as uint16")
                    recordings[population][index, sample] = values
        for recording in recordings.values():
            recording.flush()
    finally:
        for recording in recordings.values():
            recording._mmap.close()
    if any(not np.array_equal(s.w[:], w) for s, w in zip(objects["S_all"], before)):
        raise AssertionError("Weights changed during fixed-weight testing")
    for population, partial in partials.items():
        partial.replace(record_files[population])
    np.save(output / f"test_trial_ids_rep{rep}.npy", test_ids)
    metadata = snapshot(cfg)
    metadata.update(
        bin_width_ms=record_bin_ms,
        evaluation_bin_width_ms=bin_ms,
        duration_ms=duration,
        materials=materials,
        rep=rep,
        learning_during_test=False,
        plastic_training=not off,
        fixed_weight_directory=str(folders["weight"]),
        recording_bin_width_ms=record_bin_ms,
        recording_files={key: str(path.relative_to(output)) for key, path in record_files.items()},
    )
    if source_training_directory is not None:
        metadata["source_training_directory"] = str(Path(source_training_directory).resolve())
    else:
        metadata["repetition_directory"] = str(output.parent)
    save_record(
        folders["rec"] / "recording.npz",
        {
            "axis_order": ["material", "trial", "neuron", "time_bin"],
            "materials": materials,
            "trial_ids": test_ids.tolist(),
            "bin_width_ms": record_bin_ms,
            "duration_ms": duration,
            "dtype": "uint16",
            "value": "spike count per bin",
            "liquid_layers": [
                {"n_excitatory": len(g.exc), "n_inhibitory": len(g.inh)} for g in objects["G_liq"]
            ],
            "readout_layers": [len(g) for g in objects["G_out"]],
            "neuron_order": "Layers concatenated; within each Liquid layer E neurons then I neurons",
            "recording_files": metadata["recording_files"],
            "state_policy": cfg["common"].get("READOUT_STATE_POLICY", "reset_each_trial"),
        },
    )
    save_record(output / "test_config_snapshot.json", metadata)
    return f"[test rep{rep}] saved {file}; weights fixed; 1 ms spike counts"


def run_test(rep: int, train_dir: Path | None = None) -> str:
    folder = Path(train_dir) if train_dir is not None else training_directory(build_cfg(), rep)
    if not has_training_configuration(folder) and has_training_configuration(
        folder / f"rep{rep}" / "training"
    ):
        folder = folder / f"rep{rep}" / "training"
    if not has_training_configuration(folder) and has_training_configuration(folder / "training"):
        folder = folder / "training"
    if not has_training_configuration(folder):
        saved = folder / f"rep{rep}" / "config_snapshot.json"
        if saved.exists():
            cfg = load_cfg(saved.parent)
        else:
            cfg = deepcopy(build_cfg())
            if folder.resolve() != training_directory(cfg).resolve() and folder.name != "off":
                raise FileNotFoundError(f"No saved training/off configuration in {folder}")
        if learning_rule_name(cfg) == "off":
            if training_directory(cfg).resolve() != folder.resolve():
                cfg["run"]["READOUT_TRAIN_DIR"] = str(folder.resolve())
            return run_off_test(rep, cfg)
    cfg = load_cfg(folder)
    if learning_rule_name(cfg) == "off":
        return run_off_test(rep, cfg)
    materials, _, n_test, _, bin_ms, duration = validate_cfg(cfg)
    output = test_directory(folder)
    file = test_recording_files(output, rep)["readout"]
    if file.exists():
        raise FileExistsError(f"Test result already exists: {file}")
    legacy_split = folder / f"split_rep{rep}.npz"
    if legacy_split.exists():
        # 過去の実験結果は、当時記録した分割で再現する。新規実験では作成しない。
        with np.load(legacy_split, allow_pickle=False) as split:
            train_ids, test_ids = split["train_ids"].copy(), split["test_ids"].copy()
    else:
        train_ids, test_ids = sample_split(cfg, rep)
    if len(test_ids) != n_test or np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Saved training/test split overlaps or has the wrong length")
    objects = build_runtime(cfg, learning=False)
    load_weights(trained_weight_directory(folder, rep), objects)
    return record_fixed_test(rep, cfg, objects, output, test_ids, source_training_directory=folder)
