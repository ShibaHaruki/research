# -*- coding: utf-8 -*-
import os

# ===== BLAS/NumPyの過剰スレッドを抑制（並列時の暴走防止）=====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import glob
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from brian2 import *

# =========================
# Brian2 settings
# =========================
prefs.core.default_float_dtype = float64
prefs.codegen.target = "numpy"

# =========================
# Paths
# =========================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SAVE_DIR = SCRIPT_DIR

DATA_ROOT = SCRIPT_PATH.parents[1]
path = str(DATA_ROOT) + "/"

dir_name = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board"
]

# =========================
# Parallel settings
# =========================
USE_PARALLEL = True
MAX_WORKERS = max(1,  14)  # 必要なら変更
SHOW_INNER_TQDM = False                            # 並列時はFalse推奨
SKIP_FINISHED_TASKS = True                         # 既に保存済みならスキップ

# =========================
# input filters
# =========================
def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            I[0, i] = I[0, i - 1] + 1.0 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1e-3))
            I[3, i] = I[0, i]
    return I[3, :]


def calc_merkel(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            if dF_dt < 0:
                dF_dt = 0
            I[0, i] = I[0, i - 1] + 0.74 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1 * 1e-3))
            I[3, i] = I[0, i] + I[1, i] + I[2, i]
    return I[3, :]


# =========================
# Save weights / search root
# =========================
OUT_PREFIX = "SRDP_1"
SEARCH_ROOT = SAVE_DIR / f"{OUT_PREFIX}_param_search_parallel"


def format_param_value(v):
    if isinstance(v, (float, np.floating)):
        s = f"{float(v):.10g}"
    else:
        s = str(v)
    s = s.replace(".", "p").replace("-", "m").replace("+", "")
    return s


def make_param_suffix(params: dict) -> str:
    key_map = {
        "A_plus": "Ap",
        "A_minus": "Am",
        "tau_plus": "tp",
        "tau_minus": "tm",
        "A_pre_M": "ApreM",
        "A_post_M": "ApostM",
        "tau_pre_M": "tpreM",
        "tau_post_M": "tpostM",
        "wmin": "wmin",
        "wmax": "wmax",
        "eps_w": "eps",
    }
    ordered_keys = [
        "A_plus", "A_minus",
        "tau_plus", "tau_minus",
        "A_pre_M", "A_post_M",
        "tau_pre_M", "tau_post_M",
        "wmin", "wmax", "eps_w",
    ]
    parts = []
    for k in ordered_keys:
        if k in params:
            parts.append(f"{key_map[k]}-{format_param_value(params[k])}")
    return "__".join(parts)


def make_param_tag(idx: int) -> str:
    return f"search_{idx:03d}"


def get_weight_file_paths(rep: int, out_dir: Path, srdp_params: dict):
    suffix = make_param_suffix(srdp_params)
    return {
        "w_in": out_dir / f"{OUT_PREFIX}_{suffix}_w_in_rep{rep}.npy",
        "w_res": out_dir / f"{OUT_PREFIX}_{suffix}_w_res_rep{rep}.npy",
        "w_out": out_dir / f"{OUT_PREFIX}_{suffix}_w_out_rep{rep}.npy",
        "sample_seq": out_dir / f"sample_seq_rep{rep}.npy",
    }


def save_weights(rep: int,
                 w_in: np.ndarray,
                 w_res: np.ndarray,
                 S_out: Synapses,
                 N_res: int,
                 N_out: int,
                 out_dir: Path,
                 srdp_params: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = get_weight_file_paths(rep, out_dir, srdp_params)

    np.save(files["w_in"], w_in)
    np.save(files["w_res"], w_res)

    w_out_dense = np.zeros((N_res, N_out), dtype=float)
    ii = np.array(S_out.i[:], dtype=int)
    jj = np.array(S_out.j[:], dtype=int)
    ww = np.array(S_out.w_out[:], dtype=float)
    w_out_dense[ii, jj] = ww
    np.save(files["w_out"], w_out_dense)


def is_task_finished(rep: int, out_dir: Path, srdp_params: dict) -> bool:
    files = get_weight_file_paths(rep, out_dir, srdp_params)
    return (
        files["w_in"].exists() and
        files["w_res"].exists() and
        files["w_out"].exists() and
        files["sample_seq"].exists()
    )


# =========================
# SRDP parameter settings
# =========================
DEFAULT_SRDP_PARAMS = {
    "A_plus": 0.0007,
    "A_minus": 0.0006,
    "tau_plus": 11.7,     # ms
    "tau_minus": 14.0,    # ms
    "tau_pre_M": 15.0,    # ms
    "tau_post_M": 15.0,   # ms
    "A_pre_M": 0.00005,
    "A_post_M": 0.00005,
    "wmin": -1.0,
    "wmax": 1.0,
    "eps_w": 1e-12,
}

# ここを変更して探索
SRDP_SEARCH_SPACE = {
    # "A_plus":  [0.0005, 0.0007, 0.0010],
    # "A_minus": [0.0004, 0.0006, 0.0008],
    # "tau_plus":  [10.0, 11.7, 15.0],
    # "tau_minus": [12.0, 14.0, 18.0],
    "A_pre_M":   [0.00003, 0.00004, 0.00005, 0.00006, 0.00007],
    "A_post_M":  [0.00003, 0.00004, 0.00005, 0.00006, 0.00007],
    "tau_pre_M": [13.0, 14.0, 15.0, 16.0, 17.0],
    "tau_post_M": [13.0, 14.0, 15.0, 16.0, 17.0],
}


def iter_srdp_param_grid(base_params: dict, search_space: dict):
    if len(search_space) == 0:
        yield dict(base_params)
        return

    keys = list(search_space.keys())
    values_list = [search_space[k] for k in keys]

    for values in product(*values_list):
        params = dict(base_params)
        params.update(dict(zip(keys, values)))
        yield params


def save_param_file(save_dir: Path, param_dict: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "srdp_params.json", "w", encoding="utf-8") as f:
        json.dump(param_dict, f, indent=2, ensure_ascii=False)


def save_search_index(search_root: Path, all_param_sets: list):
    search_root.mkdir(parents=True, exist_ok=True)

    index_data = []
    for idx, params in enumerate(all_param_sets, start=1):
        index_data.append({
            "search_id": make_param_tag(idx),
            "params": params
        })

    with open(search_root / "all_search_params.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)


# =========================
# Repeat settings
# =========================
N_REPEAT = 1
BASE_SEED = 2
N_EPOCH = 3
N_TRAIN = 100


def run_once(rep: int, srdp_params: dict, save_dir: Path, show_inner_tqdm: bool = False):
    start_scope()

    seed_val = BASE_SEED + (rep - 1)
    np.random.seed(seed_val)
    rng = np.random.default_rng(seed_val)
    seed(seed_val)

    files = get_weight_file_paths(rep, save_dir, srdp_params)
    save_dir.mkdir(parents=True, exist_ok=True)

    # sample order
    sample_seq = np.arange(1, 325, dtype=int)
    rng.shuffle(sample_seq)
    np.save(files["sample_seq"], sample_seq)

    # hyper-parameters
    N_in = 2
    N_res = 1000
    N_out = 40

    p_in = 0.2
    p_res = 0.5
    p_out = 0.5

    v_reset = -65
    v_thr = -40

    # time
    dt_ms = 0.1
    dt_s = dt_ms * 1e-3
    defaultclock.dt = dt_ms * ms
    t0 = 0 * ms

    # synapse
    tau_r = 2 * ms
    tau_d = 20 * ms

    # SRDP
    A_plus = srdp_params["A_plus"]
    A_minus = srdp_params["A_minus"]

    tau_plus = srdp_params["tau_plus"] * ms
    tau_minus = srdp_params["tau_minus"] * ms

    tau_pre_M = srdp_params["tau_pre_M"] * ms
    tau_post_M = srdp_params["tau_post_M"] * ms

    A_pre_M = srdp_params["A_pre_M"]
    A_post_M = srdp_params["A_post_M"]

    wmin = srdp_params["wmin"]
    wmax = srdp_params["wmax"]
    eps_w = srdp_params["eps_w"]

    BIAS = -65
    G = 0.25

    # neuron params
    neuron_array_res = np.ones(N_res)
    t_ref_res = np.where(neuron_array_res == 1, 2, 2)
    tau_m_res = np.where(neuron_array_res == 1, 10, 10)

    neuron_array_out = np.ones(N_out)
    t_ref_out = np.where(neuron_array_out == 1, 2, 2)
    tau_m_out = np.where(neuron_array_out == 1, 10, 10)

    # weights
    w_in = rng.standard_normal((N_in, N_res)) * (rng.random((N_in, N_res)) < p_in) / np.sqrt(N_in * p_in)

    variance = (N_res * p_res**2) ** -1
    w_res_init = rng.standard_normal((N_res, N_res)) * (rng.random((N_res, N_res)) < p_res) * np.sqrt(variance)
    for k in range(N_res):
        QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
        if len(QS) > 0:
            w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

    mask_out = (rng.random((N_res, N_out)) < p_out).astype(float)
    w_out_init = rng.standard_normal((N_res, N_out)) * mask_out / np.sqrt(N_out * p_out) * G

    # models
    LIF = """
    dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
    I_exc : 1
    I_inh : 1
    tau_m : second
    t_ref : second
    """

    double_exp_res = """
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = G * R : 1
    """
    on_pre_res = """
    H_post += (w_res / (tau_r * tau_d)) / Hz
    """

    double_exp_out = """
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = R : 1
    """
    on_pre_out = """
    H_post += (w_out / (tau_r * tau_d)) / Hz
    """

    SRDP = """
    dApre_stdp/dt   = -Apre_stdp/tau_plus   : 1 (event-driven)
    dApost_stdp/dt  = -Apost_stdp/tau_minus : 1 (event-driven)

    dMpre/dt   = -Mpre/tau_pre_M   : 1 (event-driven)
    dMpost/dt  = -Mpost/tau_post_M : 1 (event-driven)

    w_out : 1
    eps_w : 1
    """

    SRDP_pre = """
    Apre_stdp += 1.0
    Mpre      += A_pre_M
    w_out = clip(w_out - int(w_out > eps_w) * (A_minus + Mpost) * Apost_stdp, wmin, wmax)
    """

    SRDP_post = """
    Apost_stdp += 1.0
    Mpost      += A_post_M
    w_out = clip(w_out + int(w_out > eps_w) * (A_plus + Mpre) * Apre_stdp, wmin, wmax)
    """

    eqs_res = double_exp_res + LIF
    eqs_out = double_exp_out + LIF

    # groups
    G_res = NeuronGroup(
        N_res, eqs_res,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    G_res.tau_m = tau_m_res * ms
    G_res.t_ref = t_ref_res * ms

    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    G_out.tau_m = tau_m_out * ms
    G_out.t_ref = t_ref_out * ms

    # synapses
    S_res = Synapses(G_res, G_res, model="w_res : 1", on_pre=on_pre_res, method="euler")
    S_res.connect(condition="i != j")
    S_res.w_res = w_res_init[S_res.i, S_res.j]
    S_res.delay = 0 * ms

    pre_idx, post_idx = np.where(mask_out > 0)
    S_out = Synapses(
        G_res, G_out,
        model=SRDP,
        on_pre=on_pre_out + SRDP_pre,
        on_post=SRDP_post,
        method="euler",
    )
    S_out.connect(i=pre_idx, j=post_idx)
    S_out.w_out = w_out_init[S_out.i, S_out.j]
    S_out.eps_w = eps_w
    S_out.delay = 0 * ms

    # input
    input_current = np.zeros((N_in, 1), dtype=float)

    @network_operation(dt=dt_ms * ms)
    def apply_input():
        nonlocal input_current, t0
        idx = int(((defaultclock.t - t0) / (dt_ms * ms)))
        if idx < 0:
            idx = 0
        if idx >= input_current.shape[1]:
            idx = input_current.shape[1] - 1
        I_input = input_current[:, idx] @ w_in
        G_res.I_exc = I_input
        G_res.I_inh = 0

    net = Network(G_res, G_out, S_res, S_out, apply_input)

    ns = {
        "v_reset": v_reset,
        "v_thr": v_thr,
        "BIAS": BIAS,
        "G": G,
        "tau_r": tau_r,
        "tau_d": tau_d,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "A_pre_M": A_pre_M,
        "A_post_M": A_post_M,
        "tau_plus": tau_plus,
        "tau_minus": tau_minus,
        "tau_pre_M": tau_pre_M,
        "tau_post_M": tau_post_M,
        "wmin": wmin,
        "wmax": wmax,
    }

    epoch_iter = range(1, N_EPOCH + 1)
    for epo in epoch_iter:
        train_iter = range(N_TRAIN)
        if show_inner_tqdm:
            train_iter = tqdm(train_iter, desc=f"{save_dir.name}-rep{rep}-epo{epo}")

        for i_size in train_iter:
            sid = int(sample_seq[i_size])

            for mat in dir_name:
                files_glob = sorted(glob.glob(path + "tactile_data/" + mat + f"/data_{sid}_*"))
                if len(files_glob) == 0:
                    raise FileNotFoundError(f"data not found: {mat} sample={sid}")

                df = pd.read_table(files_glob[0], header=None)

                df_np = df.to_numpy().T
                in_data_0 = df_np[:3, 3000:8000]
                nt = in_data_0.shape[1]
                t_array_s = np.arange(nt) * dt_s

                input_current = np.zeros((N_in, nt), dtype=float)

                ch = 0
                in_data = in_data_0[ch, :]

                I_merkel = calc_merkel(in_data, t_array_s, dt_s)
                I_meissner = calc_meissner(in_data, t_array_s, dt_s)

                input_current[ch * 2, :] = 0.4 * I_merkel * 0.02
                input_current[ch * 2 + 1, :] = 0.6 * 7.3 * I_meissner * 0.02

                G_res.v = v_reset + (v_thr - v_reset) * rng.random(N_res)
                G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_out)

                G_res.R = 0
                G_res.H = 0
                G_out.R = 0
                G_out.H = 0

                net.run((nt * dt_ms) * ms, namespace=ns)
                t0 += (nt * dt_ms) * ms

    save_weights(
        rep=rep,
        w_in=w_in,
        w_res=w_res_init,
        S_out=S_out,
        N_res=N_res,
        N_out=N_out,
        out_dir=save_dir,
        srdp_params=srdp_params,
    )


def build_tasks(all_param_sets: list):
    tasks = []

    for idx, srdp_params in enumerate(all_param_sets, start=1):
        tag = make_param_tag(idx)
        combo_dir = SEARCH_ROOT / tag
        combo_dir.mkdir(parents=True, exist_ok=True)
        save_param_file(combo_dir, srdp_params)

        for rep in range(1, N_REPEAT + 1):
            task = {
                "search_idx": idx,
                "search_id": tag,
                "rep": rep,
                "params": srdp_params,
                "save_dir": str(combo_dir),
            }
            tasks.append(task)

    return tasks


def worker_run_task(task: dict):
    rep = task["rep"]
    search_id = task["search_id"]
    srdp_params = task["params"]
    save_dir = Path(task["save_dir"])

    t0 = time.time()

    try:
        prefs.core.default_float_dtype = float64
        prefs.codegen.target = "numpy"

        if SKIP_FINISHED_TASKS and is_task_finished(rep, save_dir, srdp_params):
            elapsed = time.time() - t0
            return {
                "search_idx": task["search_idx"],
                "search_id": search_id,
                "rep": rep,
                "status": "skipped",
                "elapsed_sec": elapsed,
                **srdp_params,
            }

        run_once(
            rep=rep,
            srdp_params=srdp_params,
            save_dir=save_dir,
            show_inner_tqdm=SHOW_INNER_TQDM,
        )

        elapsed = time.time() - t0
        return {
            "search_idx": task["search_idx"],
            "search_id": search_id,
            "rep": rep,
            "status": "success",
            "elapsed_sec": elapsed,
            **srdp_params,
        }

    except Exception as e:
        elapsed = time.time() - t0
        err_text = traceback.format_exc()

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"error_rep{rep}.log", "w", encoding="utf-8") as f:
            f.write(err_text)

        return {
            "search_idx": task["search_idx"],
            "search_id": search_id,
            "rep": rep,
            "status": "error",
            "elapsed_sec": elapsed,
            "error_type": type(e).__name__,
            "error_message": str(e),
            **srdp_params,
        }


def append_result_jsonl(jsonl_path: Path, row: dict):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    SEARCH_ROOT.mkdir(parents=True, exist_ok=True)

    all_param_sets = list(iter_srdp_param_grid(DEFAULT_SRDP_PARAMS, SRDP_SEARCH_SPACE))
    save_search_index(SEARCH_ROOT, all_param_sets)

    tasks = build_tasks(all_param_sets)

    results_jsonl = SEARCH_ROOT / "parallel_results.jsonl"
    results_csv = SEARCH_ROOT / "parallel_results.csv"
    results_json = SEARCH_ROOT / "parallel_results.json"

    print("=" * 100)
    print(f"number of parameter sets : {len(all_param_sets)}")
    print(f"number of total tasks    : {len(tasks)}")
    print(f"N_REPEAT                 : {N_REPEAT}")
    print(f"USE_PARALLEL             : {USE_PARALLEL}")
    print(f"MAX_WORKERS              : {MAX_WORKERS}")
    print(f"SAVE_ROOT                : {SEARCH_ROOT}")
    print("=" * 100)

    results = []

    if USE_PARALLEL and MAX_WORKERS > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
            future_to_task = {
                executor.submit(worker_run_task, task): task
                for task in tasks
            }

            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="parallel search"):
                result = future.result()
                results.append(result)
                append_result_jsonl(results_jsonl, result)

                status = result["status"]
                sid = result["search_id"]
                rep = result["rep"]

                if status == "success":
                    print(f"[done]    {sid} rep={rep}  elapsed={result['elapsed_sec']:.2f}s")
                elif status == "skipped":
                    print(f"[skipped] {sid} rep={rep}")
                else:
                    print(f"[error]   {sid} rep={rep}  {result.get('error_type', '')}: {result.get('error_message', '')}")

    else:
        for task in tqdm(tasks, desc="sequential search"):
            result = worker_run_task(task)
            results.append(result)
            append_result_jsonl(results_jsonl, result)

            status = result["status"]
            sid = result["search_id"]
            rep = result["rep"]

            if status == "success":
                print(f"[done]    {sid} rep={rep}  elapsed={result['elapsed_sec']:.2f}s")
            elif status == "skipped":
                print(f"[skipped] {sid} rep={rep}")
            else:
                print(f"[error]   {sid} rep={rep}  {result.get('error_type', '')}: {result.get('error_message', '')}")

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        sort_cols = [c for c in ["search_idx", "rep"] if c in df_results.columns]
        if len(sort_cols) > 0:
            df_results = df_results.sort_values(sort_cols).reset_index(drop=True)

        df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")

        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        n_success = int((df_results["status"] == "success").sum()) if "status" in df_results else 0
        n_skipped = int((df_results["status"] == "skipped").sum()) if "status" in df_results else 0
        n_error = int((df_results["status"] == "error").sum()) if "status" in df_results else 0

        print("=" * 100)
        print("parallel SRDP parameter search finished")
        print(f"success : {n_success}")
        print(f"skipped : {n_skipped}")
        print(f"error   : {n_error}")
        print(f"results : {results_csv}")
        print("=" * 100)
    else:
        print("No tasks were executed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()