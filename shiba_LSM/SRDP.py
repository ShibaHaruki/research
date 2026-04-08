# ============================================================
# SRDP_parallel_multi_liquid_L1to4_total1000_clipW_histONLY.py
#
# 目的:
#  - SRDP学習コードを「並列リキッド数 L=1..4」へ拡張
#  - 総ニューロン数≈1000:
#      L=1: [1000]
#      L=2: [500, 500]
#      L=3: [333, 333, 333] (合計999)
#      L=4: [250, 250, 250, 250]
#  - 重みは必ず [-1, 1] にクリップ（初期化後/保存前）
#  - 各設定(L)ごとに w_in / w_res / w_out(dense) を保存（各リキッド別）
#  - PDFは「各リキッドの w_out を縦に合算(連結)したもの」の
#    『ゼロ除外ヒストグラム』だけを出力（x軸は[-1,1]固定）
#
# 出力:
#  - SRDP_parallel_L{L}_N{n1-n2-...}_liq{k}_w_in.npy
#  - SRDP_parallel_L{L}_N{n1-n2-...}_liq{k}_w_res.npy
#  - SRDP_parallel_L{L}_N{n1-n2-...}_liq{k}_w_out.npy   (dense)
#  - SRDP_parallel_L{L}_N{n1-n2-...}_wout_hist.pdf
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

from brian2 import *
prefs.core.default_float_dtype = float64
prefs.codegen.target = "numpy"

# ----------------------------
# 再現性
# ----------------------------
RNG_SEED = 2
rng = np.random.default_rng(RNG_SEED)

# ============================================================
# ★総ニューロン数≈1000になるように各層のニューロン数を設定
# ============================================================
N_RES_BY_L = {
    1: [1000],
    2: [500, 500],
    3: [333, 333, 333],          # 合計999
    4: [250, 250, 250, 250],
}
LIQUID_COUNTS = [1, 2, 3, 4]

# 学習で使う sample_seq の先頭 I_SIZE_RANGE 個（元コードに合わせて調整してOK）
I_SIZE_RANGE = 100

# ============================================================
# 重みの範囲（必ずこの範囲に収める）
# ============================================================
W_MIN = -1.0
W_MAX =  1.0

def clipW(x: np.ndarray) -> np.ndarray:
    return np.clip(x, W_MIN, W_MAX)

# ============================================================
# データ・パス
# ============================================================
script_path = Path(__file__).resolve()
path = str(script_path.parents[1]) + "/"

dir_name = ["Al_board", "buta_omote", "buta_ura", "cork",
            "denim", "rubber_board", "washi", "wood_board"]

# sample_seq.npy が無ければ作る（上書きしたくないので保護）
seq_path = script_path.parent / "sample_seq.npy"
if not seq_path.exists():
    sample_seq = np.arange(1, 325, dtype=int)  # 1..324
    rng.shuffle(sample_seq)
    np.save(seq_path, sample_seq)
    print("[INFO] created sample_seq.npy")
else:
    sample_seq = np.load(seq_path).astype(int)
    print("[INFO] loaded sample_seq.npy")

# ============================================================
# tactile input filters
# ============================================================
def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(1, len(t)):
        dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
        I[0, i] = I[0, i - 1] + 1 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
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
# ハイパラ（元コード踏襲）
# ============================================================
N_in  = 2
N_out = 40

p_in  = 0.2
p_res = 0.5
p_out = 0.5

v_reset = -65
v_thr   = -40

tau_m_res_ms = 10
t_ref_res_ms = 2
tau_m_out_ms = 10
t_ref_out_ms = 2

tau_r = 2 * ms
tau_d = 20 * ms

# SRDP params
A_plus   = 0.0007
A_minus  = 0.0006
tau_plus  = 11.7 * ms
tau_minus = 14 * ms

tau_pre_M  = 15 * ms
tau_post_M = 15 * ms

A_pre_M  = 5 * 0.00001
A_post_M = 5 * 0.00001

wmin = W_MIN
wmax = W_MAX

BIAS = -65
G = 0.25

dt_ms = 0.1
dt_s  = dt_ms * 1e-3
defaultclock.dt = dt_ms * ms

# ============================================================
# 重み初期化（配列名に w_out / w_res を使わない）
# ============================================================
def init_W_in(n_in, n_res):
    mask = (rng.random((n_in, n_res)) < p_in)
    W = rng.standard_normal((n_in, n_res)) * mask / np.sqrt(n_in * p_in)
    return clipW(W)

def init_W_res(n_res):
    mask = (rng.random((n_res, n_res)) < p_res)
    np.fill_diagonal(mask, False)

    variance = (n_res * p_res**2) ** -1
    W = rng.standard_normal((n_res, n_res)) * mask * np.sqrt(variance)

    for k in range(n_res):
        qs = np.where(mask[:, k])[0]
        if qs.size:
            W[qs, k] -= np.mean(W[qs, k])

    W = clipW(W)
    pre, post = np.where(mask)
    vals = W[pre, post]
    return W, pre, post, vals

def init_W_out(n_res):
    mask = (rng.random((n_res, N_out)) < p_out)
    W = rng.standard_normal((n_res, N_out)) * mask / np.sqrt(N_out * p_out) * G
    W = clipW(W)
    pre, post = np.where(mask)
    vals = W[pre, post]
    return W, pre, post, vals

# ============================================================
# モデル式
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

# ============================================================
# PDF（合算 w_out のヒストグラムだけ）
# ============================================================
def plot_wout_hist_only_pdf(pdf_path: str, w_out_dense_list: list[np.ndarray]):
    W_all = np.vstack(w_out_dense_list)
    W_all = clipW(W_all)
    nz = W_all[W_all != 0].ravel()

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(nz, bins=120)
        plt.xlim(W_MIN, W_MAX)
        plt.title(f"Combined nonzero w_out histogram (count={nz.size})")
        plt.xlabel("weight"); plt.ylabel("freq")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# ============================================================
# 1設定（並列リキッド L & n_res_list）を学習して保存
# ============================================================
def run_srdp_parallel_liquids(n_res_list: list[int]):
    L = len(n_res_list)
    ntag = "-".join(str(x) for x in n_res_list)

    start_scope()
    defaultclock.dt = dt_ms * ms

    # Brian2 namespace（式中の定数）
    ns = dict(
        BIAS=BIAS, G=G,
        tau_r=tau_r, tau_d=tau_d,
        v_thr=v_thr, v_reset=v_reset,
        A_plus=A_plus, A_minus=A_minus,
        tau_plus=tau_plus, tau_minus=tau_minus,
        tau_pre_M=tau_pre_M, tau_post_M=tau_post_M,
        A_pre_M=A_pre_M, A_post_M=A_post_M,
        wmin=wmin, wmax=wmax,
        clip=np.clip, int=int,
    )

    # output group
    tau_m_out = np.ones(N_out) * tau_m_out_ms
    t_ref_out = np.ones(N_out) * t_ref_out_ms

    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
        namespace=dict(ns, dt=defaultclock.dt),
    )
    G_out.tau_m = tau_m_out * ms
    G_out.t_ref = t_ref_out * ms

    # liquids
    G_res_list = []
    W_in_list  = []
    W_res_dense_list = []
    S_out_list = []

    for k, n_res in enumerate(n_res_list):
        tau_m_res = np.ones(n_res) * tau_m_res_ms
        t_ref_res = np.ones(n_res) * t_ref_res_ms

        W_in = init_W_in(N_in, n_res)
        W_res_dense, pre_r, post_r, w_res_vals = init_W_res(n_res)
        W_out_dense0, pre_o, post_o, w_out_vals0 = init_W_out(n_res)

        # reservoir group
        G_res = NeuronGroup(
            n_res, eqs_res,
            threshold="v >= v_thr",
            reset="v = v_reset",
            refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
            method="exact",
            namespace=dict(ns, dt=defaultclock.dt),
        )
        G_res.tau_m = tau_m_res * ms
        G_res.t_ref = t_ref_res * ms

        # recurrent synapses (fixed sparse)
        S_res = Synapses(
            G_res, G_res,
            model="w_res : 1",
            on_pre=on_pre_res,
            method="euler",
            namespace=dict(tau_r=tau_r, tau_d=tau_d),
        )
        S_res.connect(i=pre_r, j=post_r)
        S_res.w_res = w_res_vals
        S_res.delay = 0 * ms

        # SRDP output synapses (sparse)
        S_out = Synapses(
            G_res, G_out,
            model=SRDP,
            on_pre=on_pre_out + SRDP_pre,
            on_post=SRDP_post,
            method="euler",
            namespace=dict(ns, dt=defaultclock.dt),
        )
        S_out.connect(i=pre_o, j=post_o)
        S_out.w_out = w_out_vals0
        S_out.eps_w = 1e-12
        S_out.delay = 0 * ms

        G_res_list.append(G_res)
        W_in_list.append(W_in)
        W_res_dense_list.append(W_res_dense)
        S_out_list.append(S_out)

    # input (trial-wise)
    input_current = np.zeros((N_in, 1))
    t0 = 0 * ms

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
            G_res_list[kk].I_inh = 0

    # training loop（※あなたの貼ったコードに合わせて epo=3）
    for epo in range(3):
        for i_size in range(I_SIZE_RANGE):
            for material in tqdm(dir_name, desc=f"SRDP parallel L{L} N[{ntag}] epo={epo} i_size={i_size}"):
                files = glob.glob(path + f"tactile_data/{material}/data_{int(sample_seq[i_size])}_*")
                if not files:
                    print(f"[WARN] not found: {material} sample={int(sample_seq[i_size])}")
                    continue

                df = pd.read_table(files[0], header=None)
                df_np = df.to_numpy().T

                in_data_0 = df_np[:3, 3000:8000]
                nt = in_data_0.shape[1]
                t_array_s = np.arange(nt) * dt_s

                input_current = np.zeros((N_in, nt))

                # 元コード踏襲：ch==0 の2chのみ
                in_data = in_data_0[0, :]
                I_merkel   = calc_merkel(in_data, t_array_s, dt_s)
                I_meissner = calc_meissner(in_data, t_array_s, dt_s)
                input_current[0, :] = 0.4 * I_merkel * 0.02
                input_current[1, :] = 0.6 * 7.3 * I_meissner * 0.02

                # reset
                for kk, n_res in enumerate(n_res_list):
                    G_res_list[kk].v = v_reset + (v_thr - v_reset) * rng.random(n_res)
                    G_res_list[kk].R = 0
                    G_res_list[kk].H = 0

                G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_out)
                G_out.R = 0
                G_out.H = 0

                run((nt * dt_ms) * ms)
                t0 += (nt * dt_ms) * ms

            print(f"[INFO] i_size {i_size} done.")

    # save per liquid
    for k, n_res in enumerate(n_res_list):
        np.save(f"SRDP_parallel_L{L}_N{ntag}_liq{k+1}_w_in.npy",  clipW(W_in_list[k]))
        np.save(f"SRDP_parallel_L{L}_N{ntag}_liq{k+1}_w_res.npy", clipW(W_res_dense_list[k]))

    # w_out dense final + save
    w_out_dense_list_final = []
    for k, n_res in enumerate(n_res_list):
        S_out = S_out_list[k]
        W_dense = np.zeros((n_res, N_out), dtype=float)
        ii = np.asarray(S_out.i[:], dtype=int)
        jj = np.asarray(S_out.j[:], dtype=int)
        ww = np.asarray(S_out.w_out[:], dtype=float)
        W_dense[ii, jj] = ww
        W_dense = clipW(W_dense)
        w_out_dense_list_final.append(W_dense)
        np.save(f"SRDP_parallel_L{L}_N{ntag}_liq{k+1}_w_out.npy", W_dense)

    # PDF: histogram only
    pdf_name = f"SRDP_parallel_L{L}_N{ntag}_wout_hist.pdf"
    plot_wout_hist_only_pdf(pdf_name, w_out_dense_list_final)
    print("[SAVED]", pdf_name)

# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    for L in LIQUID_COUNTS:
        if L not in N_RES_BY_L:
            raise ValueError(f"N_RES_BY_L に L={L} がありません")
        run_srdp_parallel_liquids(N_RES_BY_L[L])
