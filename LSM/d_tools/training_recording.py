"""学習中の代表試行debugと、全素材を1回提示するごとの重み推移を保存する。"""

from pathlib import Path
import csv
import numpy as np
from brian2 import ms

from a_experiment.paths import training_weight_directory, debug_population_directory
from a_experiment.experiment_sets import trial_duration_ms
from b_network.monitor import make_weight_trace_monitors
from d_tools.debug import save_network_debug, share_debug_layer
from d_tools.plotting import save_weight_changes, save_weight_convergence
from d_tools.weight_export import save_debug_weight_distributions, save_output_weight_matrices


def sum_squared_weight_change(current_weights, previous_weights) -> float:
    """全実結合のsum((現在の重み-直前の重み)**2)。平均化・平方根なし。"""
    current = np.asarray(current_weights, dtype=np.float64)
    previous = np.asarray(previous_weights, dtype=np.float64)
    if current.shape != previous.shape or current.size == 0:
        raise ValueError("Current and previous weights must have the same nonempty shape")
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError("Weights must be finite")
    difference = current - previous
    return float(np.sum(difference * difference, dtype=np.float64))


def squared_weight_updates(checkpoints) -> np.ndarray:
    """保存時点×実結合数の配列から、連続する保存時点の二乗和の推移を返す。"""
    values = np.asarray(checkpoints, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite checkpoint x synapse weights")
    differences = np.diff(values, axis=0)
    return np.sum(differences * differences, axis=1, dtype=np.float64)


class TrainingRecorder:
    """記録のための独立した乱数とモニタを使い、学習の発火や重み更新を変更しない。"""

    def __init__(self, objects, configurations, views, folders, *, rep, n_presentations):
        from d_tools.recording_io import save_record

        self.objects = objects
        self.blocks = []
        self.seen = set()
        self.rows = []
        self.rep = rep
        self.materials = list(configurations[0]["training"]["TRAINING_MAT"])
        self.cycle_size = len(self.materials)
        self.pending = []
        self.presentations = 0
        self.expected_presentations = n_presentations
        if self.cycle_size < 1 or n_presentations % self.cycle_size:
            raise ValueError("Training must contain complete material cycles")
        self.expected_cycles = n_presentations // self.cycle_size
        self.duration = trial_duration_ms(configurations[0]["common"])
        for group_index, (cfg, (view, names), folder) in enumerate(
            zip(configurations, views, folders)
        ):
            folder = Path(folder)
            progress = training_weight_directory(folder, "progression")
            progress.mkdir(parents=True, exist_ok=True)
            synapses = list(view["S_out"])
            size = sum(len(s) for s in synapses)
            partial = progress / "weights_by_cycle.partial.npy"
            matrix = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float64, shape=(self.expected_cycles + 1, size)
            )
            matrix[0] = (
                np.concatenate([np.asarray(s.w[:]) for s in synapses]) if size else np.empty(0)
            )
            save_output_weight_matrices(progress / "cycles", view, filename_suffix="_cycle_0000")
            topology = []
            start = 0
            for syn in synapses:
                topology.append(
                    {
                        "name": names.get(syn.name, syn.name),
                        "start": start,
                        "stop": start + len(syn),
                        "pre_ids": np.asarray(syn.i[:], int).tolist(),
                        "post_ids": np.asarray(syn.j[:], int).tolist(),
                    }
                )
                start += len(syn)
            save_record(
                progress / "connections.npz",
                {
                    "columns": "Concatenated S_out synapses in stored order",
                    "synapses": topology,
                    "initial_row": 0,
                    "initial_saved_before_first_training_trial": True,
                },
            )
            enabled = bool(cfg["common"].get("DEBUG_EACH_MATERIAL", True))
            monitors, connections = (
                make_weight_trace_monitors(
                    synapses,
                    n_record=int(cfg["training"].get("WEIGHT_TRACE_SYNAPSES", 10)),
                    seed=int(cfg["network"].get("OUTPUT_SEED", cfg["common"]["BASE_SEED"] + 1))
                    + 7919,
                    prefix=f"W_training_{group_index}",
                )
                if enabled
                else ([], [])
            )
            meta_by_name = {meta["synapses"].name: meta for meta in view["output_connection_meta"]}
            for connection in connections:
                meta = meta_by_name[connection["synapse_name"]]
                connection.update(
                    pre_id=connection["pre_id"] + meta["pre_offset"],
                    liquid_layer=meta["liquid_index"] + 1,
                    readout_layer=meta["output_index"] + 1,
                    synapse_name=names.get(connection["synapse_name"], connection["synapse_name"]),
                )
            objects["net"].add(*monitors)
            self.blocks.append(
                dict(
                    cfg=cfg,
                    view=view,
                    names=names,
                    folder=folder,
                    progress=progress,
                    partial=partial,
                    matrix=matrix,
                    synapses=synapses,
                    # debug用に学習前の出力重みを保持し、終了時にも初期分布を表示する。
                    initial_output_weights={
                        names.get(syn.name, syn.name): np.asarray(syn.w[:]).copy()
                        for syn in synapses
                    },
                    monitors=monitors,
                    connections=connections,
                    enabled=enabled,
                    reports=[],
                )
            )
        # reset_each_trialでも追加モニタを復元できるよう、学習開始前の状態を登録する。
        objects["net"].store("trial_initial")

    def __enter__(self):
        return self

    def wants_debug(self, material):
        return material not in self.seen and any(b["enabled"] for b in self.blocks)

    def activate(self, enabled):
        self.start_ms = float(self.objects["net"].t / ms)
        for block in self.blocks:
            for monitor in block["monitors"]:
                monitor.active = bool(enabled and block["enabled"])

    def after_trial(self, material, sid, epoch, result=None):
        if material not in self.materials or material in [r["material"] for r in self.pending]:
            raise ValueError("Each material must appear exactly once in a training cycle")
        self.presentations += 1
        self.pending.append({"material": material, "trial_id": int(sid), "epoch": epoch})
        if len(self.pending) == self.cycle_size:
            row = len(self.rows) + 1
            self.rows.append(
                {
                    "row": row,
                    "cycle": row,
                    "completed_presentations": self.presentations,
                    "epoch": epoch,
                    "trials": self.pending.copy(),
                }
            )
            for block in self.blocks:
                values = [np.asarray(s.w[:]) for s in block["synapses"]]
                block["matrix"][row] = np.concatenate(values) if values else np.empty(0)
                save_output_weight_matrices(
                    block["progress"] / "cycles",
                    block["view"],
                    filename_suffix=f"_cycle_{row:04d}",
                )
            self.pending.clear()
        liquid_sources = []
        for index, block in enumerate(self.blocks):
            if result is None or not block["enabled"]:
                continue
            debug = block["folder"] / "debug"
            selected = {
                "liquid": result["liquid"] if not liquid_sources else [],
                "output": [result["output"][index]],
            }
            save_network_debug(
                selected,
                block["view"],
                debug,
                material=material,
                trial_id=sid,
                population_first=True,
            )
            if not liquid_sources:
                liquid_sources = [
                    debug_population_directory(
                        debug, "liquid", material, layer_index=i, layer_count=len(result["liquid"])
                    )
                    for i in range(len(result["liquid"]))
                ]
            else:
                for layer, source in enumerate(liquid_sources):
                    share_debug_layer(
                        source,
                        debug_population_directory(
                            debug,
                            "liquid",
                            material,
                            layer_index=layer,
                            layer_count=len(liquid_sources),
                        ),
                    )
            monitors = block["monitors"]
            if monitors:
                times = np.asarray(monitors[0].t / ms) - self.start_ms
                traces = np.concatenate([np.asarray(m.w) for m in monitors])
                last = np.concatenate(
                    [np.asarray(m.source.w[:])[np.asarray(m.record, int)] for m in monitors]
                )
                times = np.append(times, self.duration)
                traces = np.column_stack([traces, last])
            else:
                times = np.asarray([0.0, self.duration])
                traces = np.empty((0, 2))
            # 素材別の数値ファイルは保存せず、試行内の重み変化はdebugのPDFだけ残す。
            save_weight_changes(
                times,
                traces,
                block["connections"],
                debug_population_directory(debug, "readout", material) / "weight_changes.pdf",
                title=f'{material} | Epoch {epoch}, training trial {sid} | {len(block["connections"])} connected synapses',
            )
            block["reports"].append(
                {
                    "material": material,
                    "trial_id": int(sid),
                    "epoch": epoch,
                    "phase": "actual training",
                    "learning_enabled": True,
                    "weight_traces": len(block["connections"]),
                }
            )
            for monitor in monitors:
                monitor.active = False
                monitor.resize(0)
        if result is not None:
            self.seen.add(material)
            for monitor in self.objects["V_liq"] + self.objects["V_out"]:
                monitor.resize(0)

    def save_final_debug(self):
        from d_tools.recording_io import save_record

        for block in self.blocks:
            if block["enabled"]:
                save_debug_weight_distributions(
                    block["folder"] / "debug",
                    block["view"],
                    name_overrides=block["names"],
                    weight_overrides=block["initial_output_weights"],
                )
                save_record(block["folder"] / "debug/summary.npz", block["reports"])

    def __exit__(self, exc_type, exc_value, traceback):
        from d_tools.recording_io import save_record

        complete = (
            exc_type is None
            and self.presentations == self.expected_presentations
            and not self.pending
        )
        for block in self.blocks:
            block["matrix"].flush()
            block["matrix"]._mmap.close()
            if complete:
                final = block["progress"] / "weights_by_cycle.npy"
                block["partial"].replace(final)
                updates = squared_weight_updates(np.load(final))
                x = [r["completed_presentations"] for r in self.rows]
                with (block["progress"] / "convergence.csv").open(
                    "w", encoding="utf-8-sig", newline=""
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["cycle", "completed_trials", "sum_squared_weight_change"])
                    writer.writerows(
                        (i + 1, trial, float(value))
                        for i, (trial, value) in enumerate(zip(x, updates))
                    )
                np.save(block["progress"] / "sum_squared_weight_changes.npy", updates)
                save_weight_convergence(
                    x,
                    updates,
                    block["progress"] / "convergence.pdf",
                    title=f"Liquid -> Readout | squared weight changes per {self.cycle_size} training trials",
                    epochs=[row["epoch"] for row in self.rows],
                )
            save_record(
                block["progress"] / "progression.npz",
                {
                    "rep": self.rep,
                    "status": "complete" if complete else "incomplete",
                    "row_zero": "Initial weights, before all training",
                    "rows": self.rows,
                    "recorded_presentations": self.presentations,
                    "recorded_cycles": len(self.rows),
                    "materials_per_cycle": self.materials,
                    "save_interval_trials": self.cycle_size,
                    "pending_incomplete_cycle": self.pending,
                    "convergence_metric": "sum((current_checkpoint_weights - previous_checkpoint_weights)**2)",
                    "normalization": "none; no mean and no square root",
                    "convergence_threshold": None,
                    "weight_axis": "See connections.npz",
                },
            )
        if exc_type is None and not complete:
            raise ValueError("Training ended before all complete material cycles were recorded")
        return False
