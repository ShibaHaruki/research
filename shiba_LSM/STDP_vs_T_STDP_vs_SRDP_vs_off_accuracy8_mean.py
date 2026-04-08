import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Settings
# ======================
excel_dir = Path.cwd()
TARGET_TN = 25  # ★Tn=25だけ

pattern = re.compile(r'^(off|STDP|SRDP|T_STDP)_(\d+)_Tn_(\d+)_10fold_conf_matrices\.xlsx$')

method_label_map = {
    "off": "off",
    "STDP": "STDP",
    "SRDP": "SRDP",
    "T_STDP": "T-STDP",
}

# ======================
# Load (Tn=25だけ)
# ======================
rows = []
for f in excel_dir.glob("*_10fold_conf_matrices.xlsx"):
    m = pattern.match(f.name)
    if not m:
        continue

    method = m.group(1)
    n_liquid = int(m.group(2))
    Tn = int(m.group(3))

    # ★Tn=25以外は無視
    if Tn != TARGET_TN:
        continue

    df = pd.read_excel(f, sheet_name="accuracy")
    if "accuracy8_mean" not in df.columns:
        raise RuntimeError(f"{f.name}: sheet 'accuracy' に accuracy8_mean がありません")

    acc = float(df.loc[0, "accuracy8_mean"])
    rows.append({
        "method": method,
        "n_liquid": n_liquid,
        "Tn": Tn,
        "accuracy8_mean": acc,
        "file": f.name
    })

if not rows:
    raise RuntimeError(
        f"Tn={TARGET_TN} の対象ファイルが0件です。ファイル名が "
        f"(off|STDP|SRDP|T_STDP)_#_Tn_{TARGET_TN}_10fold_conf_matrices.xlsx 形式か確認してね。"
    )

data = pd.DataFrame(rows).sort_values(["method", "n_liquid"])
methods = sorted(data["method"].unique())
xticks = sorted(data["n_liquid"].unique())

# ======================
# Plot (1枚だけ)
# ======================
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xticks(xticks)

for method in methods:
    dm = data[data["method"] == method].sort_values("n_liquid")
    if dm.empty:
        continue

    label = method_label_map.get(method, method)
    ax.plot(dm["n_liquid"], dm["accuracy8_mean"], marker="o", label=label)

    # 点の上に数値表示（不要ならこのforを消してOK）
    for x, y in zip(dm["n_liquid"], dm["accuracy8_mean"]):
        ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_title(f"Tn = {TARGET_TN} ms", fontsize=14)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

ax.set_xlabel("liquid layer", fontsize=14)
ax.set_ylabel("accuracy8_mean", fontsize=14)
ax.tick_params(axis="both", labelsize=11)

ax.legend(loc="upper right", frameon=True, fontsize=12)

fig.tight_layout()

# ======================
# Save PDF
# ======================
out = excel_dir / f"STDP_vs_T_STDP_vs_SRDP_vs_off_accuracy8_mean_Tn{TARGET_TN}.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
print("saved:", out)

plt.show()



