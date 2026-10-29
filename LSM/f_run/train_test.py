"""通常・共有Liquidの学習とテストを順番に実行する処理。"""

from __future__ import annotations
from d_tools.progress import progress_items
from pathlib import Path
from d_tools.recording_io import record_exists
from a_experiment.experiment_sets import learning_rule_name
from f_run.training import (
    run_training,
    make_shared_training_cfg,
    candidate_weight_view,
    shared_readout_debug,
)
from f_run.testing import run_test, run_off_test
from a_experiment.paths import (
    training_directory,
    test_directory,
    test_artifact_directories,
    test_recording_files,
    training_weight_directory,
    trained_weight_directory,
)


import numpy as np
from brian2 import SpikeMonitor, NetworkOperation, ms
from f_run.common import (
    build_runtime,
    run_trial,
    validate_cfg,
    debug_readout_materials,
    sample_split,
)
from d_tools.parameter_recording import snapshot
from d_tools.recording_io import save_record
from d_tools.weight_export import load_weights, save_weights


def _pending_test_configurations(rep, configurations, *, resume):
    """未完了の候補だけを選び、再開時に不足するdebugを補う。"""
    pending = []
    for candidate in configurations:
        output = test_directory(training_directory(candidate))
        complete = all(
            (output / name).exists()
            for name in (
                f"test_trial_ids_rep{rep}.npy",
                "test_config_snapshot.json",
            )
        )
        complete = complete and all(p.exists() for p in test_recording_files(output, rep).values())
        if complete and resume:
            debug_dir = test_artifact_directories(output, cfg=candidate, rep=rep)["debug"]
            debug = debug_dir / "summary.npz"
            if candidate["common"].get("DEBUG_EACH_MATERIAL", True) and not record_exists(debug):

                debug_readout_materials(
                    candidate,
                    trained_weight_directory(training_directory(candidate), rep),
                    debug_dir,
                )
            continue
        if complete:
            raise FileExistsError(f"Test already recorded: {output}")
        pending.append(candidate)
    return pending


def _load_shared_trial_ids(rep, configurations, n_test):
    """保存済み分割を検査し、全出力群で共通のテスト試行番号を返す。"""
    trial_ids = []
    for candidate in configurations:
        legacy_split = training_directory(candidate) / f"split_rep{rep}.npz"
        if legacy_split.exists():
            # 旧結果だけは既存の記録を使用し、新規結果は共通のsample_seqを参照する。
            with np.load(legacy_split, allow_pickle=False) as split:
                train_ids, ids = split["train_ids"].copy(), split["test_ids"].copy()
        else:
            train_ids, ids = sample_split(candidate, rep)
        if len(ids) != n_test or np.intersect1d(train_ids, ids).size:
            raise ValueError("Saved test split overlaps training or has the wrong size")
        trial_ids.append(ids)
    if any(not np.array_equal(ids, trial_ids[0]) for ids in trial_ids):
        raise ValueError("Shared testing requires identical trial IDs")
    return trial_ids


def _load_and_export_test_weights(rep, objects, configurations):
    """各候補の学習済み重みを読み込み、テスト用の重みとして保存する。"""
    for index, candidate in enumerate(configurations):
        view, names = candidate_weight_view(objects, index)
        load_weights(
            trained_weight_directory(training_directory(candidate), rep), view, name_overrides=names
        )
        save_weights(
            test_artifact_directories(
                test_directory(training_directory(candidate)), cfg=candidate, rep=rep
            )["weight"],
            view,
            name_overrides=names,
        )


def _attach_shared_test_sampler(objects, duration_ms):
    """発火イベントを蓄積せず、Liquidと各出力の発火数を1 msごとに集計する。"""
    # イベント時刻を蓄積せず、累積発火数から両層の1 ms記録を作る。
    for monitor in objects["M_out"]:
        monitor.active = False
    monitors = [
        SpikeMonitor(group, record=False, name=f"M_test_count_{index}")
        for index, group in enumerate(objects["G_out"])
    ]
    liquid_monitors = [
        SpikeMonitor(group, record=False, name=f"M_test_liquid_count_{index}_{kind}")
        for index, layer in enumerate(objects["G_liq"])
        for kind, group in [("E", layer.exc), ("I", layer.inh)]
    ]
    bins = round(duration_ms)
    context = {"start_ms": 0.0, "previous": [], "counts": []}

    def sample(t):
        relative = round(float(t / ms) - context["start_ms"])
        current = [np.asarray(m.count[:]).copy() for m in monitors]
        current.append(np.concatenate([np.asarray(m.count[:]) for m in liquid_monitors]))
        if relative > 0:
            for block, values, previous in zip(context["counts"], current, context["previous"]):
                block[:, relative - 1] = values - previous
        context["previous"] = current

    sampler = NetworkOperation(
        sample, dt=1 * ms, when="start", order=-100, name="sample_shared_test_bins"
    )
    objects["net"].add(*monitors, *liquid_monitors, sampler)
    objects["net"].store("trial_initial")
    return context, sample, bins


def _save_shared_test_metadata(
    rep, candidate, objects, output, ids, *, bin_ms, duration_ms, materials, group_count
):
    """共通Liquidで計算した記録の軸・単位・試行番号・保存先を記録する。"""
    destinations = test_recording_files(output, rep)
    np.save(output / f"test_trial_ids_rep{rep}.npy", ids)
    metadata = snapshot(candidate)
    metadata.update(
        bin_width_ms=1.0,
        evaluation_bin_width_ms=bin_ms,
        duration_ms=duration_ms,
        materials=materials,
        learning_during_test=False,
        source_training_directory=str(training_directory(candidate)),
        shared_liquid=True,
        shared_output_groups=group_count,
        recording_bin_width_ms=1.0,
        recording_files={key: str(path.relative_to(output)) for key, path in destinations.items()},
    )
    save_record(
        output / "rec" / "recording.npz",
        {
            "axis_order": ["material", "trial", "neuron", "time_bin"],
            "materials": materials,
            "trial_ids": ids.tolist(),
            "bin_width_ms": 1.0,
            "duration_ms": duration_ms,
            "dtype": "uint16",
            "value": "spike count per bin",
            "recording_files": metadata["recording_files"],
            "liquid_layers": [
                {"n_excitatory": len(g.exc), "n_inhibitory": len(g.inh)} for g in objects["G_liq"]
            ],
            "readout_layers": candidate["network"]["N_out"],
            "neuron_order": "Layers concatenated; within each Liquid layer E neurons then I neurons",
            "state_policy": candidate["common"].get("READOUT_STATE_POLICY", "reset_each_trial"),
        },
    )
    save_record(output / "test_config_snapshot.json", metadata)


def _record_shared_test_trial(objects, cfg, material, sid, context, sample, bin_count_1ms):
    """1試行の発火数を集計し、1 ms記録へ保存できる整数範囲であることを確認する。"""
    # run_trial内のrestoreで時計も0に戻るため、初期化方式に合わせる。
    context["start_ms"] = (
        0.0
        if cfg["common"].get("READOUT_STATE_POLICY", "reset_each_trial") == "reset_each_trial"
        else float(objects["net"].t / ms)
    )
    context["counts"] = [
        np.zeros((len(group), bin_count_1ms), np.int32) for group in objects["G_out"]
    ]
    context["counts"].append(
        np.zeros((sum(len(g) for g in objects["G_liq"]), bin_count_1ms), np.int32)
    )
    run_trial(objects, cfg, material, int(sid), record=False)
    sample(objects["net"].t)  # runの終端ではNetworkOperationが呼ばれないので最終ビンを集計。
    for values in context["counts"]:
        if np.any(values < 0) or np.any(values > np.iinfo(np.uint16).max):
            raise OverflowError("Spike count cannot be represented as uint16")
    return context["counts"]


def run_shared_test(rep: int, configurations: list[dict], shared_dir: Path, *, resume=False) -> str:
    """共通Liquidから全出力群へ入力し、固定重みでテスト発火数を群別に記録する。"""
    import numpy as np
    import os
    import shutil

    pending = _pending_test_configurations(rep, configurations, resume=resume)
    if not pending:
        return "[shared test] all requested recordings already complete"
    shared_dir = Path(shared_dir)
    cfg = make_shared_training_cfg(pending, shared_dir)
    materials, _, n_test, _, bin_ms, duration_ms = validate_cfg(cfg)
    aggregation = round(bin_ms)
    if (
        aggregation < 1
        or not np.isclose(bin_ms, aggregation)
        or not np.isclose(duration_ms, round(duration_ms))
    ):
        raise ValueError(
            "Shared test requires evaluation bins and duration to be multiples of 1 ms"
        )
    trial_ids = _load_shared_trial_ids(rep, pending, n_test)
    objects = build_runtime(cfg, learning=False)
    _load_and_export_test_weights(rep, objects, pending)
    shared_dir.mkdir(parents=True, exist_ok=True)
    save_weights(training_weight_directory(shared_dir, "final"), objects)
    if any(candidate["common"].get("DEBUG_EACH_MATERIAL", True) for candidate in pending):
        print(f"[shared debug rep{rep}] checking each material before testing", flush=True)
        shared_readout_debug(rep, pending, cfg, shared_dir, phase="test")
        # debugとは独立した時計・状態で、保存した同じ重みを読み直してテストする。
        objects = build_runtime(cfg, learning=False)
        load_weights(training_weight_directory(shared_dir, "final"), objects)
    context, sample, bins = _attach_shared_test_sampler(objects, duration_ms)
    before = [np.asarray(s.w[:]).copy() for s in objects["S_all"]]
    fine_recordings, fine_files = [], []
    liquid_recording = None
    common_liquid = shared_dir / "rec/Liquid" / f"sliq_rec_rep{rep}.npy"
    common_liquid.parent.mkdir(parents=True, exist_ok=True)
    common_partial = common_liquid.with_suffix(".partial.npy")
    try:
        liquid_recording = np.lib.format.open_memmap(
            common_partial,
            mode="w+",
            dtype=np.uint16,
            shape=(len(materials), n_test, sum(len(g) for g in objects["G_liq"]), bins),
        )
        for candidate, group in zip(pending, objects["G_out"]):
            output = test_directory(training_directory(candidate))
            for directory in test_artifact_directories(output, cfg=candidate, rep=rep).values():
                directory.mkdir(parents=True, exist_ok=True)
            fine = test_recording_files(output, rep)["readout"]
            fine.parent.mkdir(parents=True, exist_ok=True)
            fine_partial = fine.with_suffix(".partial.npy")
            fine_recordings.append(
                np.lib.format.open_memmap(
                    fine_partial,
                    mode="w+",
                    dtype=np.uint16,
                    shape=(len(materials), n_test, len(group), bins),
                )
            )
            fine_files.append((fine_partial, fine))
        for m, material in enumerate(materials):
            for trial, sid in progress_items(
                enumerate(trial_ids[0]),
                total=n_test,
                description=f"Test | {material}",
                rep=rep,
                label=lambda item: f"trial {item[1]}",
            ):
                _record_shared_test_trial(objects, cfg, material, sid, context, sample, bins)
                liquid_recording[m, trial] = context["counts"][-1]
                for fine, counts in zip(fine_recordings, context["counts"][:-1]):
                    fine[m, trial] = counts
        if any(not np.array_equal(s.w[:], w) for s, w in zip(objects["S_all"], before)):
            raise AssertionError("Weights changed during shared testing")
        for recording in [*fine_recordings, liquid_recording]:
            recording.flush()
    finally:
        for recording in [*fine_recordings, liquid_recording]:
            if recording is not None:
                recording._mmap.close()
    common_partial.replace(common_liquid)
    for partial, final in fine_files:
        partial.replace(final)
    for candidate, ids in zip(pending, trial_ids):
        output = test_directory(training_directory(candidate))
        destinations = test_recording_files(output, rep)
        target = destinations["liquid"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(common_liquid, target)
        except OSError:
            shutil.copy2(common_liquid, target)
        _save_shared_test_metadata(
            rep,
            candidate,
            objects,
            output,
            ids,
            bin_ms=bin_ms,
            duration_ms=duration_ms,
            materials=materials,
            group_count=len(pending),
        )
    return f"[shared test rep{rep}] {len(pending)} output groups recorded; Liquid computed once per trial"


def run_train_test(rep: int, cfg: dict) -> str:
    if learning_rule_name(cfg) == "off":
        return run_off_test(rep, cfg)
    training_message = run_training(rep, cfg)
    test_message = run_test(rep, training_directory(cfg, rep))
    return training_message + " | " + test_message
