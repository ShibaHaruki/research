"""入力・再帰・出力の重み行列の出力、結合構造付き重みの保存・復元・集計。"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

import numpy as np


def _group_size(group) -> int:
    return int(len(group))


def _synapse_name(synapses) -> str:
    return str(getattr(synapses, "name", ""))


def _dense_w_matrix(
    synapses,
    n_pre: int,
    n_post: int,
    *,
    pre_offset: int = 0,
    post_offset: int = 0,
) -> np.ndarray:
    matrix = np.zeros((int(n_pre), int(n_post)), dtype=np.float64)
    if "w" not in synapses.variables or len(synapses) == 0:
        return matrix

    pre_idx = np.asarray(synapses.i[:], dtype=np.int64)
    post_idx = np.asarray(synapses.j[:], dtype=np.int64)
    matrix[pre_idx + int(pre_offset), post_idx + int(post_offset)] = np.asarray(
        synapses.w[:], dtype=np.float64
    )
    return matrix


def _match_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _liq_layer_from_name(name: str) -> int | None:
    return _match_int(r"liq(\d+)", name) or _match_int(r"_L(\d+)$", name)


def _unlink_if_possible(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[warn] cleanup failed for {Path(path)}: {exc}")


def save_liquid_weight_matrices(
    out_dir: Path,
    objects: dict[str, Any],
) -> list[Path]:
    """Save input-to-liquid, liquid recurrence, and optional liquid-to-output matrices."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_fp in out_dir.glob("*.npy"):
        _unlink_if_possible(stale_fp)

    paths: list[Path] = []

    def save_array(name: str, array: np.ndarray) -> None:
        path = out_dir / name
        np.save(path, array)
        paths.append(path)

    n_in = _group_size(objects["G_in"])
    n_liq_by_layer = [_group_size(group) for group in objects["G_liq"]]

    input_by_liq = {
        layer_index: np.zeros((n_in, n_liq), dtype=np.float64)
        for layer_index, n_liq in enumerate(n_liq_by_layer, start=1)
    }
    for synapses in objects.get("S_in", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index in input_by_liq:
            post_offset = 0
            if "_I_" in _synapse_name(synapses):
                post_offset = len(objects["G_liq"][layer_index - 1].exc)
            input_by_liq[layer_index] += _dense_w_matrix(
                synapses,
                n_in,
                n_liq_by_layer[layer_index - 1],
                post_offset=post_offset,
            )

    for layer_index, matrix in sorted(input_by_liq.items()):
        save_array(f"w_in1_liq{layer_index}.npy", matrix)

    intra_by_liq = {
        layer_index: np.zeros((n_liq, n_liq), dtype=np.float64)
        for layer_index, n_liq in enumerate(n_liq_by_layer, start=1)
    }
    for synapses in objects.get("S_intra", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index in intra_by_liq:
            name = _synapse_name(synapses)
            pair_match = re.search(r"_([EeIi]{2})_L\d+$", name)
            pair = pair_match.group(1).upper() if pair_match else "EE"
            n_exc = len(objects["G_liq"][layer_index - 1].exc)
            pre_offset = 0 if pair[0] == "E" else n_exc
            post_offset = 0 if pair[1] == "E" else n_exc
            intra_by_liq[layer_index] += _dense_w_matrix(
                synapses,
                n_liq_by_layer[layer_index - 1],
                n_liq_by_layer[layer_index - 1],
                pre_offset=pre_offset,
                post_offset=post_offset,
            )

    for layer_index, matrix in sorted(intra_by_liq.items()):
        save_array(f"w_liq{layer_index}_intra.npy", matrix)

    for liquid_index, layer in enumerate(objects["G_liq"]):
        for output_index, group in enumerate(objects.get("G_out", [])):
            matrix = np.zeros((len(layer), len(group)), dtype=float)
            for meta in objects.get("output_connection_meta", []):
                if meta["liquid_index"] == liquid_index and meta["output_index"] == output_index:
                    matrix += _dense_w_matrix(meta["synapses"], len(layer), len(group),
                                               pre_offset=meta["pre_offset"])
            save_array(f"w_liq{liquid_index+1}_out{output_index+1}.npy", matrix)
    return paths


def save_weights(folder, objects):
    """全結合の接続先・重みをweights.npzとtopology.jsonへ保存する。"""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    arrays = {}; entries = []
    for index, syn in enumerate(objects['S_all']):
        if 'w' not in syn.variables:
            continue
        prefix = f's{index}'
        arrays[prefix+'_i'] = np.asarray(syn.i[:], int)
        arrays[prefix+'_j'] = np.asarray(syn.j[:], int)
        arrays[prefix+'_w'] = np.asarray(syn.w[:]).copy()
        entries.append({'key': prefix, 'name': syn.name,
                        'n_pre': len(syn.source), 'n_post': len(syn.target)})
    np.savez_compressed(folder/'weights.npz', **arrays)
    topology = folder/'topology.json'
    temporary = topology.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(topology)


def load_weights(folder, objects):
    """結合名・接続先・サイズを照合し、保存済み重みと初期状態を設定する。"""
    folder = Path(folder)
    entries = json.loads((folder/'topology.json').read_text(encoding='utf-8'))
    by_name = {syn.name: syn for syn in objects['S_all']}
    expected = {syn.name for syn in objects['S_all'] if 'w' in syn.variables}
    if {entry['name'] for entry in entries} != expected:
        raise ValueError('Trained and evaluation synapse names differ')
    with np.load(folder/'weights.npz') as data:
        for entry in entries:
            syn = by_name[entry['name']]; key = entry['key']
            if (len(syn.source), len(syn.target)) != (entry['n_pre'], entry['n_post']):
                raise ValueError('Neuron counts differ from trained topology')
            if not np.array_equal(syn.i[:], data[key+'_i']) or not np.array_equal(syn.j[:], data[key+'_j']):
                raise ValueError('Trained and evaluation connectivity differ')
            values = data[key+'_w']
            if values.shape != (len(syn),) or not np.isfinite(values).all():
                raise ValueError('Invalid saved weights')
            syn.w = values
    objects['net'].store('trial_initial')


def weight_summary(objects):
    """出力層への結合重みの平均とノルムを返す。"""
    blocks = [np.asarray(s.w[:]) for s in objects['S_out']]
    values = np.concatenate(blocks) if blocks else np.empty(0)
    return {'mean_weight': float(values.mean()) if len(values) else 0.,
            'weight_norm': float(np.linalg.norm(values))}
