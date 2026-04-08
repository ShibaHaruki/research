# f_two_rates_bidirectional_pdf_only_with_TSTDP.py
# ============================================================
# 「発火率が違う2ニューロン（Poisson）」を 0<->1 で双方向に結合し，
# SRDP / STDP / T-STDP (Triplet STDP) で 2本の重み w01, w10 がどう変化するかを可視化
#
# ★ポイント
#  - シナプスは 0->1 と 1->0 の2本を作る（双方に重み更新）
#  - ニューロンは PoissonGroup：rates固定なので重みは入力（発火）に影響しない
#  - PNGは出さず、PDF（1ファイル）に全図を保存
# ============================================================

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

prefs.codegen.target = "numpy"

# -------------------------
# 出力ファイル名（この .py と同じ場所に保存）
# -------------------------
script_path = Path(__file__).resolve()
base_name = script_path.stem
pdf_path = script_path.parent / f"{base_name}.pdf"

# =========================
# 共通パラメータ
# =========================
wmin = -1000
wmax =  1000
eps_w_default = 1e-12

# ============================================================
# SRDP パラメータ（あなたのコード準拠）
# ============================================================
A_plus   = 0.0007
A_minus  = 0.0007

tau_plus  = 14*ms
tau_minus = 14*ms

tau_pre_M  = 50*ms
tau_post_M = 50*ms

A_pre_M  = 1
A_post_M = 1

# ============================================================
# SRDP シナプスモデル（あなたのコード準拠）
# ============================================================
SRDP = '''
dApre_stdp/dt   = -Apre_stdp/tau_plus   : 1 (event-driven)  # LTP用カーネル
dApost_stdp/dt  = -Apost_stdp/tau_minus : 1 (event-driven)  # LTD用カーネル

dMpre/dt   = -Mpre/tau_pre_M   : 1 (event-driven)  # pre発火率トレース
dMpost/dt  = -Mpost/tau_post_M : 1 (event-driven)  # post発火率トレース

w : 1
eps_w : 1
'''

SRDP_pre = '''
Apre_stdp += 1.0
Mpre      += A_pre_M
w = clip(w -  (A_minus + Mpost) * Apost_stdp, wmin, wmax)
'''

SRDP_post = '''
Apost_stdp += 1.0
Mpost      += A_post_M
w = clip(w +  (A_plus + Mpre) * Apre_stdp, wmin, wmax)
'''

# ============================================================
# STDP（pair-based）
# ============================================================
A_B_LTP = 0.0007
A_B_LTD = -0.0007
tau_s = 14*ms
tau_t = 14*ms

STDP = '''
dApre/dt  = -Apre/tau_s  : 1 (event-driven)
dApost/dt = -Apost/tau_t : 1 (event-driven)
w : 1
eps_w : 1
'''

STDP_pre = '''
Apre += A_B_LTP
w = clip(w + Apost, wmin, wmax)
'''

STDP_post = '''
Apost += A_B_LTD
w = clip(w + Apre,  wmin, wmax)
'''

# ============================================================
# T-STDP（Triplet STDP; Pfister & Gerstner系）
#   - pre側:  Apre (tau_plus),  Apre2 (tau_x)
#   - post側: Apost(tau_minus), Apost2(tau_y)
#
#   Δw_pre  = - Apost * (A2_minus + A3_minus * Apre2)
#   Δw_post = + Apre  * (A2_plus  + A3_plus  * Apost2)
#
# ★最初はA3を0にするとpair-STDPっぽくなります（周波数依存を出したいならA3>0）
# ============================================================
A2_plus  = 0.0007
A2_minus = 0.0007
A3_plus  = 0.0009
A3_minus = 0.0009

tau_x = 50*ms   # preの追加トレース
tau_y = 50*ms   # postの追加トレース

TSTDP = '''
dApre/dt   = -Apre/tau_plus    : 1 (event-driven)
dApre2/dt  = -Apre2/tau_x      : 1 (event-driven)
dApost/dt  = -Apost/tau_minus  : 1 (event-driven)
dApost2/dt = -Apost2/tau_y     : 1 (event-driven)

w : 1
eps_w : 1
'''

# 「重み更新→トレース更新」の順（同時刻スパイクで自分自身を参照しにくくするため）
TSTDP_pre = '''
w = clip(w - (A2_minus + A3_minus*Apre2) * Apost, wmin, wmax)
Apre  += 1.0
Apre2 += 1.0
'''

TSTDP_post = '''
w = clip(w + (A2_plus + A3_plus*Apost2) * Apre, wmin, wmax)
Apost  += 1.0
Apost2 += 1.0
'''


def run_two_neurons_bidirectional(rule,
                                  rate0_hz,
                                  rate1_hz,
                                  T_sec=50.0,
                                  w0_01=0.1,
                                  w0_10=0.1,
                                  seed_value=0,
                                  dt_ms=1.0,
                                  mon_dt_ms=10.0):
    """
    2ニューロン（Poisson）を作り、0->1 と 1->0 を両方接続する（2本シナプス）。
      neuron 0 : rate0_hz
      neuron 1 : rate1_hz

    ★重要：PoissonGroup は rates が固定なので、w が入力（発火）に影響しない。
    rule = "SRDP" or "STDP" or "TSTDP" で 2本の重みがどう変化するかを返す。
    """
    start_scope()

    np.random.seed(seed_value)
    seed(seed_value)

    defaultclock.dt = dt_ms*ms

    # 2ニューロン（ratesを要素ごとに指定）
    G = PoissonGroup(2, rates=[rate0_hz, rate1_hz]*Hz)

    if rule == "SRDP":
        S = Synapses(G, G, model=SRDP, on_pre=SRDP_pre, on_post=SRDP_post, method="euler")
        monitor_vars = ["w", "Mpre", "Mpost", "Apre_stdp", "Apost_stdp"]
    elif rule == "STDP":
        S = Synapses(G, G, model=STDP, on_pre=STDP_pre, on_post=STDP_post, method="euler")
        monitor_vars = ["w", "Apre", "Apost"]
    elif rule == "TSTDP":
        S = Synapses(G, G, model=TSTDP, on_pre=TSTDP_pre, on_post=TSTDP_post, method="euler")
        monitor_vars = ["w", "Apre", "Apost", "Apre2", "Apost2"]
    else:
        raise ValueError("rule must be 'SRDP' or 'STDP' or 'TSTDP'")

    # ★双方向：0->1 と 1->0 の2本だけ作る
    S.connect(i=[0, 1], j=[1, 0])

    # 接続順に w を初期化したいので、(i,j)からインデックスを特定する
    idx01 = int(np.where((np.array(S.i) == 0) & (np.array(S.j) == 1))[0][0])
    idx10 = int(np.where((np.array(S.i) == 1) & (np.array(S.j) == 0))[0][0])

    S.w = w0_01
    S.w[idx01] = w0_01
    S.w[idx10] = w0_10
    S.eps_w = eps_w_default

    # モニタ
    Mspk = SpikeMonitor(G)  # neuron 0/1 のスパイク
    Mw   = StateMonitor(S, monitor_vars, record=True, dt=mon_dt_ms*ms)

    run(T_sec*second)

    i_spk = np.array(Mspk.i[:], dtype=int)
    n0_count = int(np.sum(i_spk == 0))
    n1_count = int(np.sum(i_spk == 1))

    w01_final = float(S.w[idx01])
    w10_final = float(S.w[idx10])

    return (w01_final, w10_final, n0_count, n1_count, idx01, idx10, Mspk, Mw)


if __name__ == "__main__":
    # =========================
    # 実験設定
    # =========================
    rules = ["SRDP", "STDP", "TSTDP"]

    # ★発火率（Hz）の組：neuron0 / neuron1
    rate_pairs = [
        (1.0,  50.0),
    ]

    T_sec    = 100.0
    n_trials = 10
    w0_01    = 0.1
    w0_10    = 0.1

    results = {rule: {} for rule in rules}
    traces_example = {rule: {} for rule in rules}  # 代表例 (idx01, idx10, Mspk, Mw)

    for rule in rules:
        for (r0, r1) in rate_pairs:
            w01_list = []
            w10_list = []
            n0_list = []
            n1_list = []

            for tr in range(n_trials):
                (w01_final, w10_final, n0_c, n1_c, idx01, idx10, Mspk, Mw) = run_two_neurons_bidirectional(
                    rule=rule,
                    rate0_hz=r0,
                    rate1_hz=r1,
                    T_sec=T_sec,
                    w0_01=w0_01,
                    w0_10=w0_10,
                    seed_value=1000 + tr,
                    dt_ms=1.0,
                    mon_dt_ms=10.0
                )

                w01_list.append(w01_final)
                w10_list.append(w10_final)
                n0_list.append(n0_c)
                n1_list.append(n1_c)

                if tr == 0:
                    traces_example[rule][(r0, r1)] = (idx01, idx10, Mspk, Mw)

            results[rule][(r0, r1)] = {
                "w01_final": np.array(w01_list, dtype=float),
                "w10_final": np.array(w10_list, dtype=float),
                "n0_spikes": np.array(n0_list, dtype=int),
                "n1_spikes": np.array(n1_list, dtype=int),
            }

            emp0 = results[rule][(r0, r1)]["n0_spikes"].mean() / T_sec
            emp1 = results[rule][(r0, r1)]["n1_spikes"].mean() / T_sec

            print(f"[{rule} | rates n0,n1 = {r0}Hz, {r1}Hz] "
                  f"spk(n0) mean={results[rule][(r0,r1)]['n0_spikes'].mean():.1f}, "
                  f"spk(n1) mean={results[rule][(r0,r1)]['n1_spikes'].mean():.1f} "
                  f"(emp {emp0:.2f}Hz, {emp1:.2f}Hz), "
                  f"w01 mean={results[rule][(r0,r1)]['w01_final'].mean():.4f}, "
                  f"w10 mean={results[rule][(r0,r1)]['w10_final'].mean():.4f}")

    # ============================================================
    # 図1：w01(t), w10(t)（代表試行）
    # ============================================================
    fig1, axes1 = plt.subplots(
        len(rules), len(rate_pairs),
        figsize=(6*len(rate_pairs), 4*len(rules)),
        squeeze=False
    )

    # y軸（重み）統一のため全代表例から範囲を集める
    global_wmin = +np.inf
    global_wmax = -np.inf
    for rule in rules:
        for (r0, r1) in rate_pairs:
            idx01, idx10, Mspk, Mw = traces_example[rule][(r0, r1)]
            w01 = np.array(Mw.w[idx01], dtype=float)
            w10 = np.array(Mw.w[idx10], dtype=float)
            global_wmin = min(global_wmin, float(w01.min()), float(w10.min()))
            global_wmax = max(global_wmax, float(w01.max()), float(w10.max()))

    margin = 0.05*(global_wmax - global_wmin) if global_wmax > global_wmin else 0.1
    global_wmin -= margin
    global_wmax += margin

    for i_rule, rule in enumerate(rules):
        for j, (r0, r1) in enumerate(rate_pairs):
            ax = axes1[i_rule, j]
            idx01, idx10, Mspk, Mw = traces_example[rule][(r0, r1)]

            tt = Mw.t/second
            ax.plot(tt, Mw.w[idx01], linewidth=1.4, label="w01 (0->1)")
            ax.plot(tt, Mw.w[idx10], linewidth=1.4, label="w10 (1->0)")

            ax.set_title(f"{rule} | rates n0,n1 = {r0}Hz,{r1}Hz")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("w")
            ax.set_ylim(global_wmin, global_wmax)
            ax.legend(loc="best", fontsize=8)

    fig1.tight_layout()

    # ============================================================
    # 図2：w01_final と w10_final のヒストグラム（ルール×条件）
    # ============================================================
    fig2, axes2 = plt.subplots(
        len(rules), len(rate_pairs),
        figsize=(6*len(rate_pairs), 4*len(rules)),
        squeeze=False, sharey=True
    )

    bins = 12

    all_w = []
    for rule in rules:
        for p in rate_pairs:
            all_w.append(results[rule][p]["w01_final"])
            all_w.append(results[rule][p]["w10_final"])
    all_w = np.concatenate(all_w).astype(float)

    xmin = float(np.min(all_w))
    xmax = float(np.max(all_w))
    margin = 0.05*(xmax - xmin) if xmax > xmin else 0.1
    xmin -= margin
    xmax += margin
    bin_edges = np.linspace(xmin, xmax, bins + 1)

    global_ymax = 1
    for rule in rules:
        for p in rate_pairs:
            c01, _ = np.histogram(results[rule][p]["w01_final"], bins=bin_edges)
            c10, _ = np.histogram(results[rule][p]["w10_final"], bins=bin_edges)
            global_ymax = max(global_ymax, int(c01.max()), int(c10.max()))
    global_ymax = int(np.ceil(global_ymax * 1.10))

    for i_rule, rule in enumerate(rules):
        for j, (r0, r1) in enumerate(rate_pairs):
            ax = axes2[i_rule, j]
            ax.hist(results[rule][(r0, r1)]["w01_final"], bins=bin_edges, alpha=0.7, label="w01")
            ax.hist(results[rule][(r0, r1)]["w10_final"], bins=bin_edges, alpha=0.7, label="w10")
            ax.set_title(f"{rule}: w_final hist\nrates n0,n1={r0},{r1}Hz")
            ax.set_xlabel("w_final")
            ax.set_ylabel("trials")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, global_ymax)
            ax.legend(loc="best", fontsize=8)

    fig2.tight_layout()

    # ============================================================
    # 図3：平均±std のまとめ（w01 と w10 を別サブプロット）
    # ============================================================
    fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    x = np.arange(len(rate_pairs))
    labels = [f"{r0},{r1}Hz" for (r0, r1) in rate_pairs]

    # ルール数に応じてバー幅を自動調整
    width = 0.8 / len(rules)

    for k, rule in enumerate(rules):
        means01 = [results[rule][p]["w01_final"].mean() for p in rate_pairs]
        stds01  = [results[rule][p]["w01_final"].std()  for p in rate_pairs]
        means10 = [results[rule][p]["w10_final"].mean() for p in rate_pairs]
        stds10  = [results[rule][p]["w10_final"].std()  for p in rate_pairs]

        offset = -0.4 + (k + 0.5)*width
        ax31.bar(x + offset, means01, width, yerr=stds01, capsize=4, label=rule)
        ax32.bar(x + offset, means10, width, yerr=stds10, capsize=4, label=rule)

    ax31.set_xticks(x)
    ax31.set_xticklabels(labels, rotation=30, ha="right")
    ax31.set_title("w01 (0->1) mean ± std")
    ax31.set_ylabel("w_final")
    ax31.legend()

    ax32.set_xticks(x)
    ax32.set_xticklabels(labels, rotation=30, ha="right")
    ax32.set_title("w10 (1->0) mean ± std")
    ax32.legend()

    fig3.tight_layout()

    # =========================
    # PDFに全て保存（PNGは出さない）
    # =========================
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    print("saved PDF:", pdf_path)
    plt.show()


