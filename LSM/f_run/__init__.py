"""実行入口と、その実行手順・並列管理をまとめるパッケージ。"""

import os
from a_experiment.experiment_sets import RUNTIME_BACKEND_SETTINGS

# rep並列時の各プロセス内スレッド数を抑え、CPUの過剰な取り合いを防ぐ。
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_variable] = "1"
# グラフは画面を開かず、現在の共通描画設定で生成する。
os.environ.setdefault("MPLBACKEND", RUNTIME_BACKEND_SETTINGS["MATPLOTLIB_BACKEND"])
