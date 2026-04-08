# ============================================================
# multi_parallel_liquid_L1to4_total1000_histONLY_clipW_normInit.py
#
# 目的:
#  - リキッド層の層数 L を 1〜4 に変化（層は“並列”）
#  - 総ニューロン数がおよそ1000:
#       L=1: [1000]
#       L=2: [500, 500]
#       L=3: [333, 333, 333]   (合計999)
#       L=4: [250, 250, 250, 250]
#  - 重み初期化は「元の正規分布ベース」（あなた指定の式）に戻す
#  - ただし念のため clipW で [-1,1] に収める
#  - 各Lごとに sout_rec を保存
#  - 各Lごとに「全リキッドの w_out を縦連結」してゼロ除外ヒストグラムだけPDF出力
#    （x軸は [-1,1] 固定）
#
# 出力:
#  - off_parallel_L{L}_N{n1-n2-...}_sout_rec.npy
#  - off_parallel_L{L}_N{n1-n2-...}_wout_hist.pdf
# ============================================================

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
# ★層数 L ごとの各層ニューロン数（総数≈1000）
# ============================================================
N_RES_BY_L = {
    1: [1000],
    2: [500, 500],
    3: [333, 333, 333],          # 合計999
    4: [250, 250, 250, 250],
}
LIQUID_COUNTS = [1, 2, 3, 4]

# ============================================================
# 重みの範囲（必ずこの範囲に収める）
# ============================================================
W_MIN = -1.0
W_MAX =  1.0

def clipW(x: np.ndarray) -> np.ndarray:
    return np.clip(x, W_MIN, W_MAX)

# ============================================================
# データ設定
# ============================================================
BASE_PATH = str(Path(__file__).resolve().parents[1]) + "/"
DIR_NAMES = ["Al_board", "buta_omote", "buta_ura", "cork",
             "denim", "rubber_board", "washi", "wood_board"]

def load_or_make_sample_seq(n_total: int = 324, filename: str = "sample_seq.npy"):
    p = Path(filename)
    if p.exists():
        return np.load(filename)
    seq = np.arange(1, n_total + 1, dtype=int)
    rng.shuffle(seq)
    np.save(filename, seq)
    return seq

sample_seq = load_or_make_sample_seq()
test_seq = sample_seq[100:]
N_SAMPLE_PER_MAT = int(min(100, len(test_seq)))

# ============================================================
# 入力フィルタ（Meissner / Merkel）
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
# モデル・ハイパラ
# ============================================================
N_IN  = 2
N_OUT = 40

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

BIAS = -65
G = 0.25

dt_ms = 0.1
dt_s  = dt_ms * 1e-3

# ============================================================
# ★重み初期化（あなた指定の元の式に戻す）
# ============================================================
def init_w_in(n_in: int, n_res: int) -> np.ndarray:
    # w_in = randn(N_in,N_res) * (rand < p_in) / sqrt(N_in*p_in)
    mask = (rng.random((n_in, n_res)) < p_in)
    W = rng.standard_normal((n_in, n_res)) * mask / np.sqrt(n_in * p_in)
    return clipW(W)

def init_w_res_dense(n_res: int):
    # variance = (N_res * p_res**2)**-1
    variance = (n_res * (p_res**2)) ** -1

    # w_res_init = randn * (rand < p_res) * sqrt(variance)
    mask = (rng.random((n_res, n_res)) < p_res)
    np.fill_diagonal(mask, False)

    W = rng.standard_normal((n_res, n_res)) * mask * np.sqrt(variance)

    # 列ごと平均ゼロ（非ゼロのみ）
    for k in range(n_res):
        QS = np.where(mask[:, k])[0]
        if QS.size > 0:
            W[QS, k] -= np.mean(W[QS, k])

    W = clipW(W)
    return W, mask

def init_w_out_dense(n_res: int):
    # W_out_init = randn(N_res,N_out) * (rand < p_out) / sqrt(N_out*p_out) * G
    mask = (rng.random((n_res, N_OUT)) < p_out)
    W = rng.standard_normal((n_res, N_OUT)) * mask / np.sqrt(N_OUT * p_out) * G
    W = clipW(W)
    return W, mask

# ============================================================
# 方程式
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
# PDF（合算 w_out のヒストグラムだけ / x軸[-1,1]固定）
# ============================================================
def plot_wout_hist_only_pdf(pdf_path: str, wout_dense_list: list[np.ndarray]):
    w_all = np.vstack(wout_dense_list)        # (sum N_res, N_OUT)
    w_all = clipW(w_all)
    nz = w_all[w_all != 0].ravel()

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(nz, bins=120, range=(W_MIN, W_MAX))  # ★ビン範囲も固定
        plt.xlim(W_MIN, W_MAX)                        # ★表示範囲 -1..1 固定
        plt.title(f"Combined nonzero w_out histogram (count={nz.size})")
        plt.xlabel("weight")
        plt.ylabel("freq")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# ============================================================
# 1つの L で実行（並列リキッド / 学習なし＝重み固定）
# ============================================================
def run_parallel_liquids(n_res_list: list[int], n_sample_per_mat: int):
    L = len(n_res_list)
    ntag = "-".join(str(x) for x in n_res_list)

    start_scope()
    defaultclock.dt = dt_ms * ms

    # sout_rec 記録
    n_bins = 500
    sout_rec = np.zeros((len(DIR_NAMES), n_sample_per_mat, N_OUT, n_bins))

    # 出力層
    tau_m_out = np.ones(N_OUT) * tau_m_out_ms
    t_ref_out = np.ones(N_OUT) * t_ref_out_ms

    common_ns = dict(
        BIAS=BIAS, G=G,
        tau_r=tau_r, tau_d=tau_d,
        v_thr=v_thr, v_reset=v_reset,
        dt=defaultclock.dt,   # refractory の timestep(..., dt) 用
    )

    G_out = NeuronGroup(
        N_OUT, eqs_out,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
        namespace=common_ns,
    )
    G_out.tau_m = tau_m_out * ms
    G_out.t_ref = t_ref_out * ms

    Mr_out = SpikeMonitor(G_out)

    # リキッド群
    G_res_list = []
    W_in_list = []
    S_out_list = []

    for n_res in n_res_list:
        tau_m_res = np.ones(n_res) * tau_m_res_ms
        t_ref_res = np.ones(n_res) * t_ref_res_ms

        W_in = init_w_in(N_IN, n_res)

        W_res_dense, mask_res = init_w_res_dense(n_res)
        pre_r, post_r = np.where(mask_res)
        w_res_vals = W_res_dense[pre_r, post_r]

        Wout_dense, mask_out = init_w_out_dense(n_res)
        pre_o, post_o = np.where(mask_out)
        w_out_vals_init = Wout_dense[pre_o, post_o]

        W_in_list.append(W_in)

        G_res = NeuronGroup(
            n_res, eqs_res,
            threshold="v >= v_thr",
            reset="v = v_reset",
            refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
            method="exact",
            namespace=common_ns,
        )
        G_res.tau_m = tau_m_res * ms
        G_res.t_ref = t_ref_res * ms
        G_res_list.append(G_res)

        # recurrent（疎結合）
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

        # out（疎結合 / 学習なしなので w_out は変化しない）
        S_out = Synapses(
            G_res, G_out,
            model="w_out : 1",
            on_pre=on_pre_out,
            method="euler",
            namespace=dict(tau_r=tau_r, tau_d=tau_d),
        )
        S_out.connect(i=pre_o, j=post_o)
        S_out.w_out = w_out_vals_init
        S_out.delay = 0 * ms
        S_out_list.append(S_out)

    # 入力（サンプルごとに差し替え）
    input_current = np.zeros((N_IN, 1))
    t0 = 0 * ms

    @network_operation(dt=dt_ms * ms)
    def apply_input():
        nonlocal input_current, t0
        idx = int((defaultclock.t - t0) / (dt_ms * ms))
        if idx < 0 or idx >= input_current.shape[1]:
            return

        x = input_current[:, idx]
        for k in range(L):
            I_in = x @ W_in_list[k]
            G_res_list[k].I_exc = I_in
            G_res_list[k].I_inh = 0

    # 実行（テスト）
    for mat_i, mat in enumerate(DIR_NAMES):
        for s in tqdm(range(n_sample_per_mat), desc=f"L{L} {mat}"):
            files = glob.glob(BASE_PATH + f"tactile_data/{mat}/data_{int(test_seq[s])}_*")
            if not files:
                print(f"[WARN] file not found: {mat} sample={int(test_seq[s])}")
                continue

            df = pd.read_table(files[0], header=None)
            df_np = df.to_numpy().T
            in_data_0 = df_np[:3, 3000:8000]
            nt = in_data_0.shape[1]
            t_array_s = np.arange(nt) * dt_s

            input_current = np.zeros((N_IN, nt))

            # 元コード: ch==0 の2chのみ使用
            in_data = in_data_0[0, :]
            I_merkel   = calc_merkel(in_data, t_array_s, dt_s)
            I_meissner = calc_meissner(in_data, t_array_s, dt_s)
            input_current[0, :] = 0.4 * I_merkel * 0.02
            input_current[1, :] = 0.6 * 7.3 * I_meissner * 0.02

            # reset states
            for k, n_res in enumerate(n_res_list):
                G_res_list[k].v = v_reset + (v_thr - v_reset) * rng.random(n_res)
                G_res_list[k].R = 0
                G_res_list[k].H = 0

            G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_OUT)
            G_out.R = 0
            G_out.H = 0

            start_t = t0
            run((nt * dt_ms) * ms)
            t0 += (nt * dt_ms) * ms
            end_t = t0

            # output spikes -> bins
            mask = (Mr_out.t > start_t) & (Mr_out.t <= end_t)
            if np.any(mask):
                rel_ms = (Mr_out.t[mask] - start_t) / ms
                ids = Mr_out.i[mask]
                bin_edges = np.linspace(0, nt * dt_ms, n_bins + 1)
                for n in range(N_OUT):
                    counts, _ = np.histogram(rel_ms[ids == n], bins=bin_edges)
                    sout_rec[mat_i, s, n, :] = counts
            else:
                sout_rec[mat_i, s, :, :] = 0

    # 保存（sout_rec）
    npy_name = f"off_parallel_L{L}_N{ntag}_sout_rec.npy"
    np.save(npy_name, sout_rec)
    print("[SAVE]", npy_name, sout_rec.shape)

    # 合算 w_out を dense に戻して、ヒストグラムPDFだけ出力
    wout_dense_list = []
    for k, n_res in enumerate(n_res_list):
        W = np.zeros((n_res, N_OUT), dtype=float)
        ii = np.asarray(S_out_list[k].i[:], dtype=int)
        jj = np.asarray(S_out_list[k].j[:], dtype=int)
        ww = np.asarray(S_out_list[k].w_out[:], dtype=float)
        W[ii, jj] = ww
        W = clipW(W)
        wout_dense_list.append(W)

    pdf_name = f"off_parallel_L{L}_N{ntag}_wout_hist.pdf"
    plot_wout_hist_only_pdf(pdf_name, wout_dense_list)
    print("[SAVE]", pdf_name)

# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    for L in LIQUID_COUNTS:
        if L not in N_RES_BY_L:
            raise ValueError(f"N_RES_BY_L に L={L} がありません")
        run_parallel_liquids(N_RES_BY_L[L], N_SAMPLE_PER_MAT)


