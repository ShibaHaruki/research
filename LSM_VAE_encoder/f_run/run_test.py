"""学習済み重みを読み込み、テストサンプルの出力スパイク列を保存する入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
TEST_RESULT_DIR = RESULTS_PATH / RUN_CFG.get("TEST_RESULT_DIR", "test_run")

import multiprocessing as mp

import numpy as np
import pandas as pd
from brian2 import (
    BrianLogger,
    Network,
    SpikeMonitor,
    TimedArray,
    defaultclock,
    ms,
    prefs,
    second,
    seed,
    start_scope,
)

from b_network.build_network import (
    make_in_neuron_group,
    make_in_to_liq_synapses,
    make_liq_intra_synapses,
    make_liq_to_out_synapses,
    make_liquid_neuron_groups,
    make_output_neuron_groups,
    make_poisson_to_liq_synapses,
)
from d_tools.compat import repeat_count, test_bin_steps, test_sample_count, training_sample_count
from d_tools.run_paths import jsonable, make_run_output_dir, save_used_parameters_text
from d_tools.weight_export import (
    load_weight_matrices_like_old_code,
    save_output_weight_matrix,
)
from f_run.run_training import (
    TRAINING_RESULT_DIR,
    build_cfg,
    build_input_current,
    build_network_cfg,
    load_tactile_data,
    reset_dynamic_state,
    tqdm,
    _first_value,
    _input_layout,
)


prefs.codegen.target = RUN_CFG["BRIAN_CODEGEN_TARGET"]
for log_name in RUN_CFG["BRIAN_SUPPRESS_LOG_NAMES"]:
    BrianLogger.suppress_name(log_name)


def _load_source_cfg(train_dir: Path) -> dict:
    # training_run に保存された設定を読み、学習時と同じ条件を再現する。
    cfg = build_cfg()
    snapshot_fp = Path(train_dir) / "config_snapshot.json"
    if not snapshot_fp.exists():
        return cfg

    snapshot = json.loads(snapshot_fp.read_text(encoding="utf-8"))
    for key in ("common", "training", "liquid", "test", "network", "models"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            cfg[key] = value
    return cfg


def _test_cfg_from_source_cfg(source_cfg: dict) -> dict:
    # テストでは重みを固定して使うため、学習則は off に切り替える。
    cfg = deepcopy(source_cfg)
    cfg["run"] = dict(cfg["run"])
    cfg["run"]["INTERNAL_STATE_ENABLE"] = False
    cfg["models"] = dict(cfg["models"])
    cfg["models"]["LEARNING_RULE_MODEL"] = "off"
    return cfg


def training_output_dir_for_cfg(cfg: dict | None = None) -> Path:
    cfg = deepcopy(cfg) if cfg is not None else build_cfg()
    net_cfg = build_network_cfg(cfg)
    return make_run_output_dir(TRAINING_RESULT_DIR, cfg, net_cfg, include_output=True)


def test_output_dir_for_training_dir(train_dir: Path) -> Path:
    train_dir = Path(train_dir).resolve()
    try:
        rel = train_dir.relative_to(TRAINING_RESULT_DIR.resolve())
        out_dir = TEST_RESULT_DIR / rel
    except ValueError:
        out_dir = TEST_RESULT_DIR / train_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _weights_dir_for_rep(train_dir: Path, rep: int) -> Path:
    rep_dir = Path(train_dir) / "weights" / "final" / f"rep{rep}"
    if rep_dir.exists():
        return rep_dir
    return Path(train_dir) / "weights" / "final"


def _load_test_sample_seq(rep: int, n_needed: int) -> np.ndarray:
    fp = RESULTS_PATH / RUN_CFG["SAMPLE_SEQ_DIR"] / f"sample_seq_rep{rep}.npy"
    if not fp.exists():
        raise FileNotFoundError(f"sample sequence not found: {fp}")
    seq = np.load(fp).astype(np.int32).reshape(-1)
    if len(seq) < n_needed:
        raise ValueError(f"{fp.name} is too short ({len(seq)} < {n_needed})")
    return seq


def make_test_network(net_cfg: dict, N_in: int, rng: np.random.Generator):
    # テスト用ネットワークを構築する。あとで学習済み重みを読み込む。
    input_ta0 = TimedArray(np.zeros((2, N_in)), dt=float(net_cfg["dt_ms"]) * ms)
    G_in = make_in_neuron_group(N_in=N_in, input_ta=input_ta0)
    G_liq = make_liquid_neuron_groups(net_cfg, rng)
    G_out = make_output_neuron_groups(net_cfg, rng)

    S_in, _ = make_in_to_liq_synapses(G_in, G_liq, rng, net_cfg)
    G_poisson, S_poisson, _ = make_poisson_to_liq_synapses(G_liq, rng, net_cfg)
    S_intra, _ = make_liq_intra_synapses(G_liq, rng, net_cfg)
    S_lo, _ = make_liq_to_out_synapses(G_liq, G_out, rng, net_cfg)

    M_out = [
        SpikeMonitor(group, name=f"M_out_test_L{layer_index + 1}")
        for layer_index, group in enumerate(G_out)
    ]

    net = Network()
    net.add(G_in)
    net.add(*G_poisson)
    net.add(*G_liq)
    net.add(*G_out)
    net.add(*(S_in + S_poisson + S_intra + S_lo))
    net.add(*M_out)

    return {
        "net": net,
        "G_in": G_in,
        "G_poisson": G_poisson,
        "G_liq": G_liq,
        "G_out": G_out,
        "S_in": S_in,
        "S_poisson": S_poisson,
        "S_intra": S_intra,
        "S_lo": S_lo,
        "S_all": S_in + S_poisson + S_intra + S_lo,
        "M_out": M_out,
    }


def _output_offsets(groups: list) -> np.ndarray:
    sizes = [len(group) for group in groups]
    return np.cumsum([0] + sizes[:-1]).astype(int)


def _bin_output_spikes(
    objects: dict,
    start_indices: list[int],
    start_t,
    *,
    n_out_total: int,
    n_bins: int,
    duration_ms: float,
) -> np.ndarray:
    # 出力層の SpikeMonitor を、分類評価で使う sout_rec の時間 bin 形式へ変換する。
    binned = np.zeros((n_out_total, n_bins), dtype=np.int32)
    output_offsets = _output_offsets(objects["G_out"])
    bin_edges = np.linspace(0.0, duration_ms, n_bins + 1)

    for layer_index, monitor in enumerate(objects["M_out"]):
        end_index = len(monitor.t)
        if end_index <= start_indices[layer_index]:
            continue

        t_ms = np.asarray(
            (monitor.t[start_indices[layer_index] : end_index] - start_t) / ms,
            dtype=np.float64,
        )
        neuron_ids = np.asarray(
            monitor.i[start_indices[layer_index] : end_index],
            dtype=np.int32,
        ) + int(output_offsets[layer_index])

        for neuron_id in np.unique(neuron_ids):
            counts, _ = np.histogram(t_ms[neuron_ids == neuron_id], bins=bin_edges)
            binned[int(neuron_id), :] = counts

    return binned


def run_test(rep: int, train_dir: Path | None = None) -> str:
    # 1 rep 分のテストを実行し、素材 x サンプル x 出力ニューロン x 時間bin を保存する。
    start_scope()

    train_dir = Path(train_dir) if train_dir is not None else training_output_dir_for_cfg()
    if not train_dir.exists():
        raise FileNotFoundError(f"training result dir not found: {train_dir}")

    source_cfg = _load_source_cfg(train_dir)
    test_cfg = _test_cfg_from_source_cfg(source_cfg)
    run_cfg = test_cfg["run"]
    common = test_cfg["common"]
    training = test_cfg["training"]
    test = test_cfg["test"]
    filter_funcs = test_cfg["filter_funcs"]
    input_filter_map = test_cfg["input_filter_map"]
    test_net_cfg = build_network_cfg(test_cfg)

    dt_s = float(common["dt_ms"])
    defaultclock.dt = float(test_net_cfg["dt_ms"]) * ms

    base_seed = int(common["BASE_SEED"] + rep)
    np.random.seed(base_seed)
    seed(base_seed)
    rng = np.random.default_rng(base_seed)

    channels, _ = _input_layout(input_filter_map)
    N_in = sum(len(input_filter_map[ch]) for ch in channels)
    objects = make_test_network(test_net_cfg, N_in=N_in, rng=rng)

    weights_dir = _weights_dir_for_rep(train_dir, rep)
    # run_training が保存した最終重みを読み込み、この重みでテスト入力を流す。
    load_weight_matrices_like_old_code(weights_dir, objects)

    out_dir = test_output_dir_for_training_dir(train_dir)
    weights_used_dir = out_dir / "weights_used"
    weight_fp = save_output_weight_matrix(weights_used_dir, objects, rep)

    n_train_samples = training_sample_count(training, common)
    n_test_samples = test_sample_count(test, common)
    if n_test_samples <= 0:
        raise ValueError("NUM_TEST_SAMPLE must be > 0")

    sample_seq = _load_test_sample_seq(rep, n_train_samples + n_test_samples)
    test_seq = sample_seq[n_train_samples : n_train_samples + n_test_samples]

    materials = list(test["TEST_MAT"])
    bin_steps = test_bin_steps(test, 10)
    expected_nt = int(common["SLICE_END"]) - int(common["SLICE_START"])
    n_bins = max(1, int(np.ceil(expected_nt / max(bin_steps, 1))))
    n_out_total = int(sum(len(group) for group in objects["G_out"]))
    sout_rec = np.zeros((len(materials), len(test_seq), n_out_total, n_bins), dtype=np.int32)
    records: list[dict] = []

    metadata = {
        "source_training_dir": str(train_dir),
        "source_weights_dir": str(weights_dir),
        "used_output_weights": str(weight_fp),
        "source_models": source_cfg["models"],
        "test_models": test_cfg["models"],
        "common": source_cfg["common"],
        "training": source_cfg["training"],
        "test": source_cfg["test"],
        "net_cfg": test_net_cfg,
    }
    save_used_parameters_text(
        out_dir,
        test_cfg,
        test_net_cfg,
        include_output=True,
        include_learning=True,
        extra={"test_metadata": metadata},
    )
    save_used_parameters_text(
        weights_used_dir,
        test_cfg,
        test_net_cfg,
        include_output=True,
        include_learning=True,
        extra={"test_metadata": metadata},
    )
    (out_dir / "test_config_snapshot.json").write_text(
        json.dumps(jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(out_dir / f"test_seq_rep{rep}.npy", test_seq)

    for mat_index, mat in enumerate(materials):
        iterator = tqdm(range(len(test_seq)), desc=f"[test rep{rep}] {mat}")
        for sample_index in iterator:
            # 1試行ごとに状態を初期化し、出力スパイクだけを記録する。
            sid = int(test_seq[sample_index])
            file = load_tactile_data(mat, sid)
            if file is None:
                continue

            df = pd.read_table(file, header=None)
            df_np = df.to_numpy().T
            in_data_0 = df_np[
                : int(common["NUM_SENSOR"]),
                int(common["SLICE_START"]) : int(common["SLICE_END"]),
            ]

            nt = in_data_0.shape[1]
            t_array = np.arange(nt) * dt_s
            input_current = build_input_current(
                in_data_0=in_data_0,
                t_array=t_array,
                dt=dt_s,
                input_filter_map=input_filter_map,
                filter_funcs=filter_funcs,
            )

            input_ta = TimedArray(input_current.T, dt=float(test_net_cfg["dt_ms"]) * ms)
            reset_dynamic_state(objects, test_net_cfg)
            objects["G_in"].namespace["input_ta"] = input_ta
            objects["G_in"].t_start = defaultclock.t

            start_indices = [len(monitor.t) for monitor in objects["M_out"]]
            start_t = defaultclock.t
            objects["net"].run(nt * dt_s * second, namespace={"input_ta": input_ta})

            duration_ms = nt * dt_s * 1000.0
            sout_rec[mat_index, sample_index, :, :] = _bin_output_spikes(
                objects,
                start_indices,
                start_t,
                n_out_total=n_out_total,
                n_bins=n_bins,
                duration_ms=duration_ms,
            )
            records.append(
                {
                    "rep": rep,
                    "mat_index": mat_index,
                    "mat": mat,
                    "sample_index": sample_index,
                    "sid": sid,
                    "duration_ms": duration_ms,
                    "n_bins": n_bins,
                    "output_spikes": int(np.sum(sout_rec[mat_index, sample_index])),
                }
            )

    # 旧評価コードと互換になるよう sout_rec_rep*.npy という名前で保存する。
    np.save(out_dir / f"sout_rec_rep{rep}.npy", sout_rec)
    pd.DataFrame(records).to_csv(out_dir / f"test_trials_rep{rep}.csv", index=False)
    return (
        f"[test rep{rep}] saved sout_rec_rep{rep}.npy {tuple(sout_rec.shape)} "
        f"using weights from {weights_dir}"
    )


def run_test_worker(rep: int, train_dir: str | None) -> str:
    return run_test(rep, train_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run test using trained LSM weights.")
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Training result directory containing weights/final. Default: current config output dir.",
    )
    args = parser.parse_args()

    train_dir = args.train_dir
    cfg = build_cfg()
    reps = list(range(1, repeat_count(cfg["common"], 1) + 1))
    if len(reps) == 1 or os.environ.get(RUN_CFG["NO_MP_ENV"]) == "1":
        for rep in reps:
            print(run_test_worker(rep, train_dir))
        return 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(os.cpu_count() or 1, len(reps)),
        mp_context=ctx,
    ) as executor:
        futures = [executor.submit(run_test_worker, rep, train_dir) for rep in reps]
        for future in as_completed(futures):
            print(future.result())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
