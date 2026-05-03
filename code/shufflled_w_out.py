
# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import re

# =========================
# settings
# =========================
N_REPEAT   = 10
N_SHUFFLES = 5
BASE_SEED  = 12345

SCRIPT_DIR = Path(__file__).resolve().parent

# 学習則の w_out が置いてある場所（必要に応じて変更）
# 例: SCRIPT_DIR / "w_out_all_rules"
IN_ROOT = SCRIPT_DIR

# 出力先（まとめてここに吐く）
OUT_DIR = SCRIPT_DIR / "shuffled_w_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# どのファイルを対象にするか：末尾が "_w_out_rep{rep}.npy" のもの
# 例: off_1_w_out_rep1.npy
PATTERN = re.compile(r"^(?P<prefix>.+)_w_out_rep(?P<rep>\d+)\.npy$")

# =========================
# shuffle functions
# =========================
def shuffle_prepost(W, rng):
    N_res, N_out = W.shape
    perm_pre  = rng.permutation(N_res)
    perm_post = rng.permutation(N_out)
    W_shuf = W[perm_pre, :][:, perm_post]
    return W_shuf, perm_pre, perm_post

def shuffle_pre_only(W, rng):
    N_res, _ = W.shape
    perm_pre = rng.permutation(N_res)
    W_shuf = W[perm_pre, :]
    return W_shuf, perm_pre

def shuffle_post_only(W, rng):
    _, N_out = W.shape
    perm_post = rng.permutation(N_out)
    W_shuf = W[:, perm_post]
    return W_shuf, perm_post

def rng_for(prefix: str, rep: int, k: int, mode: str):
    """
    prefix（学習則や条件を含む名前）も seed に混ぜて、ファイルごとに独立な系列にする
    """
    mode_id = {"all": 11, "pre": 22, "post": 33}[mode]
    # prefix を固定長の int に落とす（再現性のため hash の代わりに簡易）
    prefix_id = sum(ord(c) for c in prefix) % 10000
    seed = BASE_SEED + prefix_id * 1000000 + rep * 10000 + k * 100 + mode_id
    return np.random.default_rng(seed)

# =========================
# main
# =========================
def main():
    # IN_ROOT 以下から対象ファイルを全部集める（サブフォルダも含む）
    files = sorted(IN_ROOT.rglob("*_w_out_rep*.npy"))
    targets = []

    for p in files:
        m = PATTERN.match(p.name)
        if not m:
            continue
        prefix = m.group("prefix")
        rep    = int(m.group("rep"))
        if 1 <= rep <= N_REPEAT:
            targets.append((p, prefix, rep))

    if not targets:
        raise RuntimeError(f"No targets found under: {IN_ROOT}")

    print(f"Found {len(targets)} files to shuffle.")

    for in_path, prefix, rep in targets:
        W = np.load(in_path)
        if W.ndim != 2:
            raise ValueError(f"W must be 2D, got shape={W.shape} at {in_path}")

        z0 = float(np.mean(W == 0))

        # 元ファイルの相対パスを保って出力（衝突防止）
        rel_parent = in_path.parent.relative_to(IN_ROOT)
        out_base_dir = OUT_DIR / rel_parent
        out_base_dir.mkdir(parents=True, exist_ok=True)

        for k in range(1, N_SHUFFLES + 1):
            # ---------- (1) pre+post ----------
            rng_all = rng_for(prefix, rep, k, "all")
            W_all, perm_pre_all, perm_post_all = shuffle_prepost(W, rng_all)

            np.save(out_base_dir / f"{prefix}_shuf_prepost_w_out_rep{rep}_k{k}.npy", W_all)
            np.save(out_base_dir / f"{prefix}_perm_prepost_pre_rep{rep}_k{k}.npy", perm_pre_all)
            np.save(out_base_dir / f"{prefix}_perm_prepost_post_rep{rep}_k{k}.npy", perm_post_all)

            # ---------- (2) pre-only ----------
            rng_pre = rng_for(prefix, rep, k, "pre")
            W_pre, perm_pre = shuffle_pre_only(W, rng_pre)

            np.save(out_base_dir / f"{prefix}_shuf_preonly_w_out_rep{rep}_k{k}.npy", W_pre)
            np.save(out_base_dir / f"{prefix}_perm_preonly_rep{rep}_k{k}.npy", perm_pre)

            # ---------- (3) post-only ----------
            rng_post = rng_for(prefix, rep, k, "post")
            W_post, perm_post = shuffle_post_only(W, rng_post)

            np.save(out_base_dir / f"{prefix}_shuf_postonly_w_out_rep{rep}_k{k}.npy", W_post)
            np.save(out_base_dir / f"{prefix}_perm_postonly_rep{rep}_k{k}.npy", perm_post)

            # ---------- quick sanity ----------
            z1 = float(np.mean(W_all == 0))
            z2 = float(np.mean(W_pre == 0))
            z3 = float(np.mean(W_post == 0))
            print(f"[{in_path.name} | k{k}] saved | zero_ratio orig={z0:.4f} all={z1:.4f} pre={z2:.4f} post={z3:.4f}")

if __name__ == "__main__":
    main()



