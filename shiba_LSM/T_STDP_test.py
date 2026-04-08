# ============================================================
# T_STDP_parallel_TEST_multi_liquid_L1to4_total1000.py
#
# 目的:
#  - T-STDP学習後に保存した重み（w_in, w_res, w_out）を読み込み
#  - 学習なし（固定重み）でテストデータを流す
#  - sout_rec (8, n_sample, N_out, 500) を保存
#
# 入力:
#  - sample_seq.npy
#  - tactile_data/<material>/data_<id>_*
#  - T_STDP_parallel_L{L}_N{ntag}_liq{k}_w_in.npy
#    T_STDP_parallel_L{L}_N{ntag}_liq{k}_w_res.npy
#    T_STDP_parallel_L{L}_N{ntag}_liq{k}_w_out.npy
#
# 出力:
#  - T_STDP_parallel_TEST_L{L}_N{ntag}_sout_rec.npy
#
# 追加:
#  - 進捗表示（全体バー、run中進捗(任意)）
#  - 並列実行（L=1..4 を別プロセスで並列：Brian2はmultiprocessing推奨）
# ============================================================

# --- 並列時に「プロセス×BLASスレッド」で遅くなるのを防ぐ（numpy import より前推奨） ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

from brian2 import *
prefs.core.default_float_dtype = float64
prefs.codegen.target = "numpy"

# ----------------------------
# 並列実行設定
# ----------------------------
PARALLEL_RUN = True       # True: Lごとに並列 / False: 逐次
MAX_WORKERS = None        # NoneならCPUに合わせる（最大4が上限）

# ----------------------------
# 進捗表示設定
# ----------------------------
# run() が長くて「止まって見える」場合は True（runを刻むので少し遅くなる可能性あり）
PROGRESS_WITHIN_RUN = True
CHUNK_MS = 20.0            # run刻み幅（ms）
POSTFIX_UPDATE_EVERY = 5   # postfix更新間引き

# ----------------------------
# 再現性
# ----------------------------
RNG_SEED = 2

# ============================================================
# ★総ニューロン数≈1000になるように各層のニューロン数を設定
# ============================================================
N_RES_BY_L = {
    1: [1000],
    2: [500, 500],
    3: [333, 333, 333],
    4: [250, 250, 250, 250],
}
LIQUID_COUNTS = [1, 2, 3, 4]

# ============================================================
# 重みの範囲
# ============================================================
W_MIN = -1.0
W_MAX =  1.0
def clipW(x: np.ndarray) -> np.ndarray:
    return np.clip(x, W_MIN, W_MAX)

# ============================================================
# パス・データ
# ============================================================
script_path = Path(__file__).resolve()
BASE_PATH = str(script_path.parents[1]) + "/"   # 必要ならここを変更
THIS_DIR = script_path.parent                   # 重み/seq の置き場（このスクリプトと同じ場所想定）

dir_name = ["Al_board", "buta_omote", "buta_ura", "cork",
            "denim", "rubber_board", "washi", "wood_board"]

# sample_seq.npy（学習時と同じもの）
seq_path = THIS_DIR / "sample_seq.npy"
if not seq_path.exists():
    raise FileNotFoundError(f"sample_seq.npy not found: {seq_path}")
sample_seq = np.load(seq_path).astype(int)
test_seq = sample_seq[100:]
N_SAMPLE_PER_MAT = int(min(100, len(test_seq)))

# ============================================================
# tactile input filters
# ============================================================
def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(1, len(t)):
        dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
        I[0, i] = I[0, i - 1] + 1.0 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
        I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
        I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1e-3))
        I[3, i] = I[0, i]
    return I[3, :]

def calc_merkel(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(1, len(t)):
        dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
        if dF_dt < 0:
            dF_dt = 0
        I[0, i] = I[0, i - 1] + 0.74 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
        I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1 * 1e-3))
        I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1 * 1e-3))
        I[3, i] = I[0, i] + I[1, i] + I[2, i]
    return I[3, :]

# ============================================================
# モデル・ハイパラ（学習コード側に合わせる）
# ============================================================
v_reset = -65
v_thr   = -40

tau_m_res_ms = 10
t_ref_res_ms = 2
tau_m_out_ms = 10
t_ref_out_ms = 2

tau_r = 2 * ms
tau_d = 20 * ms

BIAS = -65
G = 0.25

dt_ms = 0.1
dt_s  = dt_ms * 1e-3

# ============================================================
# 方程式（学習コードと同系）
# ============================================================
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

eqs_res = double_exp_res + LIF
eqs_out = double_exp_out + LIF

# ============================================================
# 重みロード（T_STDP prefix）
# ============================================================
def load_trained_weights_tstdp(L: int, n_res_list: list[int]):
    ntag = "-".join(str(x) for x in n_res_list)

    w_in_list = []
    w_res_list = []
    w_out_list = []

    for k, n_res in enumerate(n_res_list, start=1):
        f_in  = THIS_DIR / f"T_STDP_parallel_L{L}_N{ntag}_liq{k}_w_in.npy"
        f_res = THIS_DIR / f"T_STDP_parallel_L{L}_N{ntag}_liq{k}_w_res.npy"
        f_out = THIS_DIR / f"T_STDP_parallel_L{L}_N{ntag}_liq{k}_w_out.npy"

        if not f_in.exists():
            raise FileNotFoundError(f"weight not found: {f_in}")
        if not f_res.exists():
            raise FileNotFoundError(f"weight not found: {f_res}")
        if not f_out.exists():
            raise FileNotFoundError(f"weight not found: {f_out}")

        W_in  = clipW(np.load(f_in))
        W_res = clipW(np.load(f_res))
        W_out = clipW(np.load(f_out))

        # shapeチェック
        assert W_res.shape == (n_res, n_res), f"{f_res.name} shape mismatch: {W_res.shape} vs {(n_res,n_res)}"
        assert W_out.shape[0] == n_res, f"{f_out.name} shape mismatch: {W_out.shape}"
        assert W_in.shape[1] == n_res, f"{f_in.name} shape mismatch: {W_in.shape}"

        w_in_list.append(W_in)
        w_res_list.append(W_res)
        w_out_list.append(W_out)

    N_in  = w_in_list[0].shape[0]
    N_out = w_out_list[0].shape[1]
    return ntag, N_in, N_out, w_in_list, w_res_list, w_out_list

# ============================================================
# テスト本体（並列リキッド）
# ============================================================
def run_test_tstdp_parallel_liquids(L: int, n_res_list: list[int]):
    # Lごとに決定的だが同一にならないseedにする（並列でも再現性◎）
    rng_local = np.random.default_rng(RNG_SEED + 1000 * L)

    ntag, N_in, N_out, W_in_list, W_res_list, W_out_list = load_trained_weights_tstdp(L, n_res_list)

    start_scope()
    defaultclock.dt = dt_ms * ms

    n_bins = 500
    sout_rec = np.zeros((len(dir_name), N_SAMPLE_PER_MAT, N_out, n_bins), dtype=float)

    # namespace（refractoryで dt を使う）
    common_ns = dict(
        BIAS=BIAS, G=G,
        tau_r=tau_r, tau_d=tau_d,
        v_thr=v_thr, v_reset=v_reset,
        dt=defaultclock.dt,
    )

    # output group
    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
        namespace=common_ns,
    )
    G_out.tau_m = (np.ones(N_out) * tau_m_out_ms) * ms
    G_out.t_ref = (np.ones(N_out) * t_ref_out_ms) * ms

    Mr_out = SpikeMonitor(G_out)

    # reservoir groups + synapses
    G_res_list = []
    for k, n_res in enumerate(n_res_list):
        G_res = NeuronGroup(
            n_res, eqs_res,
            threshold="v >= v_thr",
            reset="v = v_reset",
            refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
            method="exact",
            namespace=common_ns,
        )
        G_res.tau_m = (np.ones(n_res) * tau_m_res_ms) * ms
        G_res.t_ref = (np.ones(n_res) * t_ref_res_ms) * ms

        # recurrent: 非ゼロだけ接続（対角除外）
        W_res = W_res_list[k]
        pre_r, post_r = np.where(W_res != 0.0)
        m = pre_r != post_r
        pre_r, post_r = pre_r[m], post_r[m]
        wres_vals = W_res[pre_r, post_r]

        S_res = Synapses(
            G_res, G_res,
            model="w_res : 1",
            on_pre=on_pre_res,
            method="euler",
            namespace=dict(tau_r=tau_r, tau_d=tau_d),
        )
        if pre_r.size > 0:
            S_res.connect(i=pre_r, j=post_r)
            S_res.w_res = wres_vals
        S_res.delay = 0 * ms

        # out: 非ゼロだけ接続（固定重み）
        W_out = W_out_list[k]
        pre_o, post_o = np.where(W_out != 0.0)
        wout_vals = W_out[pre_o, post_o]

        S_out = Synapses(
            G_res, G_out,
            model="w_out : 1",
            on_pre=on_pre_out,
            method="euler",
            namespace=dict(tau_r=tau_r, tau_d=tau_d),
        )
        if pre_o.size > 0:
            S_out.connect(i=pre_o, j=post_o)
            S_out.w_out = wout_vals
        S_out.delay = 0 * ms

        G_res_list.append(G_res)

    # 入力（サンプルごと差し替え）
    input_current = np.zeros((N_in, 1), dtype=float)
    t0 = 0 * ms  # サンプル開始時刻（apply_input の基準）

    @network_operation(dt=dt_ms * ms)
    def apply_input():
        nonlocal input_current, t0
        idx = int((defaultclock.t - t0) / (dt_ms * ms))
        if idx < 0 or idx >= input_current.shape[1]:
            return
        x = input_current[:, idx]
        for kk in range(L):
            I_in = x @ W_in_list[kk]
            G_res_list[kk].I_exc = I_in
            G_res_list[kk].I_inh = 0.0

    # ----------------------------
    # test loop（全体進捗バー）
    # ----------------------------
    total_iters = len(dir_name) * N_SAMPLE_PER_MAT
    with tqdm(total=total_iters, desc=f"T-TEST L{L} N[{ntag}]", dynamic_ncols=True) as pbar:
        for mat_i, material in enumerate(dir_name):
            for s in range(N_SAMPLE_PER_MAT):
                pbar.set_postfix_str(f"{material} s={s+1}/{N_SAMPLE_PER_MAT}")

                files = glob.glob(BASE_PATH + f"tactile_data/{material}/data_{int(test_seq[s])}_*")
                if not files:
                    tqdm.write(f"[WARN] not found: {material} sample={int(test_seq[s])}")
                    pbar.update(1)
                    continue

                df = pd.read_table(files[0], header=None)
                df_np = df.to_numpy().T

                in_data_0 = df_np[:3, 3000:8000]
                nt = in_data_0.shape[1]
                t_array_s = np.arange(nt) * dt_s

                # 学習コード踏襲：ch==0 の2chのみ
                input_current = np.zeros((N_in, nt), dtype=float)
                if N_in < 2:
                    raise ValueError(f"N_in={N_in} is too small. Expected at least 2.")

                in_data = in_data_0[0, :]
                I_merkel   = calc_merkel(in_data, t_array_s, dt_s)
                I_meissner = calc_meissner(in_data, t_array_s, dt_s)
                input_current[0, :] = 0.4 * I_merkel * 0.02
                input_current[1, :] = 0.6 * 7.3 * I_meissner * 0.02

                # reset
                for kk, n_res in enumerate(n_res_list):
                    G_res_list[kk].v = v_reset + (v_thr - v_reset) * rng_local.random(n_res)
                    G_res_list[kk].R = 0
                    G_res_list[kk].H = 0
                    G_res_list[kk].I_exc = 0
                    G_res_list[kk].I_inh = 0

                G_out.v = v_reset + (v_thr - v_reset) * rng_local.random(N_out)
                G_out.R = 0
                G_out.H = 0
                G_out.I_exc = 0
                G_out.I_inh = 0

                # run & binning
                start_t = t0
                total_ms = nt * dt_ms

                if PROGRESS_WITHIN_RUN:
                    ran_ms = 0.0
                    chunk_count = 0
                    while ran_ms < total_ms:
                        step = min(CHUNK_MS, total_ms - ran_ms)
                        run(step * ms)
                        ran_ms += step
                        chunk_count += 1

                        if (chunk_count % POSTFIX_UPDATE_EVERY) == 0 or ran_ms >= total_ms:
                            frac = ran_ms / total_ms if total_ms > 0 else 1.0
                            pbar.set_postfix_str(
                                f"{material} s={s+1}/{N_SAMPLE_PER_MAT} run={frac*100:5.1f}%"
                            )
                    t0 = start_t + total_ms * ms
                else:
                    run(total_ms * ms)
                    t0 = start_t + total_ms * ms

                end_t = t0

                mask = (Mr_out.t > start_t) & (Mr_out.t <= end_t)
                if np.any(mask):
                    rel_ms = (Mr_out.t[mask] - start_t) / ms
                    ids = Mr_out.i[mask]
                    bin_edges = np.linspace(0, total_ms, n_bins + 1)
                    for n in range(N_out):
                        counts, _ = np.histogram(rel_ms[ids == n], bins=bin_edges)
                        sout_rec[mat_i, s, n, :] = counts
                else:
                    sout_rec[mat_i, s, :, :] = 0

                pbar.update(1)

    out_name = f"T_STDP_parallel_TEST_L{L}_N{ntag}_sout_rec.npy"
    out_path = THIS_DIR / out_name
    np.save(out_path, sout_rec)
    print("[SAVED]", str(out_path), sout_rec.shape)

# ============================================================
# main（Lごとに並列実行）
# ============================================================
def _run_one_L(job):
    """ProcessPoolExecutorから呼ぶ用（pickle安定化）"""
    L, n_res_list = job
    run_test_tstdp_parallel_liquids(L, n_res_list)
    return L

if __name__ == "__main__":
    jobs = [(L, N_RES_BY_L[L]) for L in LIQUID_COUNTS]

    if not PARALLEL_RUN:
        for L in tqdm(LIQUID_COUNTS, desc="L loop", dynamic_ncols=True):
            run_test_tstdp_parallel_liquids(L, N_RES_BY_L[L])
        print("[DONE] All L finished (sequential).")
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")  # Windows想定
        if MAX_WORKERS is None:
            max_workers = min(len(jobs), mp.cpu_count())
        else:
            max_workers = int(MAX_WORKERS)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_run_one_L, job) for job in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"Parallel L jobs (workers={max_workers})",
                            dynamic_ncols=True):
                L_done = fut.result()  # 例外があればここで出る
                tqdm.write(f"[DONE] L={L_done}")

        print("[DONE] All L finished (parallel).")

