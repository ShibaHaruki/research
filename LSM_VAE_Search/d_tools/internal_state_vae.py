"""リキッド内部状態を1D-CNN VAEで学習・可視化する処理。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from d_tools.plotting import try_import_pyplot
from d_tools.run_paths import jsonable, value_stem
from d_tools.separation_metrics import discover_internal_state_files, scatter_metrics


EPS = 1e-12
DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "x", "*", "+"]
DEFAULT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


@dataclass(frozen=True)
class WindowedInternalStateDataset:
    # VAE に渡す内部状態データ一式。x の形は batch x neuron x time_window。
    x: np.ndarray
    labels: np.ndarray
    material_names: list[str]
    sample_indices: np.ndarray
    source_files: list[str]
    window_centers_ms: np.ndarray
    n_neurons: int
    n_windows: int


def _import_torch():
    # torch は VAE を動かすときだけ必要なので、import 失敗時は分かりやすいエラーにする。
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for internal-state VAE training. "
            "Install torch in the Python environment used to run this script."
        ) from exc

    return torch, nn, F, DataLoader, TensorDataset


def infer_dt_ms(t_ms: np.ndarray, *, fallback_ms: float = 10.0) -> float:
    t_arr = np.asarray(t_ms, dtype=np.float64).reshape(-1)
    if t_arr.size >= 2:
        diffs = np.diff(t_arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return float(fallback_ms)


def window_internal_state(
    x_state: np.ndarray,
    t_ms: np.ndarray,
    *,
    window_ms: float = 10.0,
    step_ms: float = 10.0,
    fallback_dt_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """内部状態 ``(N, T)`` を ``(N, K)`` の時間窓平均へ変換する。

    デフォルトでは ``window_ms == step_ms == 10`` なので、10 ms ごとの非重複 bin になる。
    現在の spike-bin 内部状態では、保存済みの時間軸をほぼそのまま使う。
    """

    x = np.nan_to_num(
        np.asarray(x_state, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if x.ndim != 2:
        raise ValueError(f"x_state must be 2D (N, T), got shape={x.shape}")
    if x.shape[1] <= 0:
        raise ValueError("x_state has no time points.")

    t_arr = np.asarray(t_ms, dtype=np.float64).reshape(-1)
    if t_arr.size < x.shape[1]:
        dt_ms = float(fallback_dt_ms)
        t_arr = np.arange(x.shape[1], dtype=np.float64) * dt_ms
    else:
        t_arr = t_arr[: x.shape[1]]
        dt_ms = infer_dt_ms(t_arr, fallback_ms=fallback_dt_ms)

    window_bins = max(1, int(round(float(window_ms) / max(dt_ms, EPS))))
    step_bins = max(1, int(round(float(step_ms) / max(dt_ms, EPS))))
    n_time = int(x.shape[1])

    if n_time <= window_bins:
        return np.mean(x, axis=1, keepdims=True), np.array(
            [float(t_arr[0] + n_time * dt_ms / 2.0)],
            dtype=np.float64,
        )

    starts = list(range(0, n_time - window_bins + 1, step_bins))
    windows = np.empty((x.shape[0], len(starts)), dtype=np.float32)
    centers = np.empty(len(starts), dtype=np.float64)
    for out_index, start in enumerate(starts):
        end = start + window_bins
        windows[:, out_index] = np.mean(x[:, start:end], axis=1)
        centers[out_index] = float(t_arr[start] + window_bins * dt_ms / 2.0)

    return windows, centers


def load_windowed_internal_state_dataset(
    internal_state_dir: Path,
    *,
    window_ms: float = 10.0,
    step_ms: float = 10.0,
    materials: Sequence[str] | None = None,
    max_samples_per_class: int | None = None,
    file_glob: str = "*_liquid_internal_state_all.npz",
) -> WindowedInternalStateDataset:
    # internal_states/<素材名>/*.npz を読み、素材ラベル付きの batch x N x K 配列にまとめる。
    material_to_files = discover_internal_state_files(internal_state_dir, file_glob=file_glob)
    material_names = list(materials) if materials is not None else sorted(material_to_files)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    sample_indices: list[int] = []
    source_files: list[str] = []
    selected_materials: list[str] = []
    window_centers_ref: np.ndarray | None = None

    for material in material_names:
        files = list(material_to_files.get(str(material), []))
        if max_samples_per_class is not None:
            files = files[: int(max_samples_per_class)]
        if not files:
            continue

        label_index = len(selected_materials)
        selected_materials.append(str(material))
        for sample_index, fp in enumerate(files):
            with np.load(fp) as data:
                x_state = np.asarray(data["x_state"], dtype=np.float32)
                t_ms = np.asarray(data["t_ms"], dtype=np.float64)
            x_windowed, centers = window_internal_state(
                x_state,
                t_ms,
                window_ms=window_ms,
                step_ms=step_ms,
            )
            rows.append(x_windowed)
            labels.append(label_index)
            sample_indices.append(sample_index)
            source_files.append(str(Path(fp).resolve()))
            if window_centers_ref is None:
                window_centers_ref = centers

    if not rows:
        raise FileNotFoundError(f"No selected internal state files found under {internal_state_dir}")

    min_neurons = min(item.shape[0] for item in rows)
    min_windows = min(item.shape[1] for item in rows)
    if min_neurons <= 0 or min_windows <= 0:
        raise ValueError("windowed internal states have an empty neuron or time-window axis.")

    x = np.stack(
        [item[:min_neurons, :min_windows] for item in rows],
        axis=0,
    ).astype(np.float32, copy=False)
    if window_centers_ref is None:
        window_centers_ref = np.arange(min_windows, dtype=np.float64)
    else:
        window_centers_ref = np.asarray(window_centers_ref[:min_windows], dtype=np.float64)

    return WindowedInternalStateDataset(
        x=x,
        labels=np.asarray(labels, dtype=np.int64),
        material_names=selected_materials,
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        source_files=source_files,
        window_centers_ms=window_centers_ref,
        n_neurons=int(min_neurons),
        n_windows=int(min_windows),
    )


def _candidate_key(entry: dict, fallback_index: int) -> str:
    # CMA-ES候補を一意に識別する名前を作る。保存済みのcandidate_keyがあればそれを優先する。
    key = entry.get("candidate_key")
    if key:
        return str(key)
    if "generation" in entry and "candidate_index" in entry:
        return f"g{int(entry['generation']):03d}_c{int(entry['candidate_index']):03d}"
    return f"candidate_{int(fallback_index):04d}"


def _visible_material_names(labels: np.ndarray, material_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    # ある候補に存在する素材だけにラベルを詰め直す。欠けた素材がある場合でもDR計算を壊さないため。
    labels_arr = np.asarray(labels, dtype=np.int64)
    unique_labels = [int(item) for item in sorted(np.unique(labels_arr).tolist())]
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped = np.asarray([label_map[int(label)] for label in labels_arr], dtype=np.int64)
    names = [str(material_names[old]) for old in unique_labels]
    return remapped, names


def combine_windowed_internal_state_datasets(
    entries: Sequence[dict],
    *,
    window_ms: float = 10.0,
    step_ms: float = 10.0,
    materials: Sequence[str] | None = None,
    max_samples_per_class: int | None = None,
    file_glob: str = "*_liquid_internal_state_all.npz",
) -> tuple[WindowedInternalStateDataset, np.ndarray]:
    # 複数パラメータ候補の内部状態をまとめ、1つのVAEで学習できる共通データセットにする。
    loaded: list[tuple[str, WindowedInternalStateDataset]] = []
    material_order: list[str] = list(materials) if materials is not None else []

    for entry_index, entry in enumerate(entries):
        internal_state_dir = entry.get("internal_state_dir")
        if not internal_state_dir:
            continue
        dataset = load_windowed_internal_state_dataset(
            Path(internal_state_dir),
            window_ms=window_ms,
            step_ms=step_ms,
            materials=materials,
            max_samples_per_class=max_samples_per_class,
            file_glob=file_glob,
        )
        candidate_key = _candidate_key(entry, entry_index)
        loaded.append((candidate_key, dataset))
        if materials is None:
            for material in dataset.material_names:
                if material not in material_order:
                    material_order.append(str(material))

    if not loaded:
        raise FileNotFoundError("No candidate internal-state datasets were loaded for common VAE.")
    if not material_order:
        raise ValueError("No materials were found in candidate internal-state datasets.")

    material_to_label = {material: index for index, material in enumerate(material_order)}
    min_neurons = min(dataset.n_neurons for _key, dataset in loaded)
    min_windows = min(dataset.n_windows for _key, dataset in loaded)
    if min_neurons <= 0 or min_windows <= 0:
        raise ValueError("combined internal states have an empty neuron or time-window axis.")

    x_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    sample_rows: list[np.ndarray] = []
    source_files: list[str] = []
    candidate_keys: list[str] = []
    window_centers_ref = np.asarray(loaded[0][1].window_centers_ms[:min_windows], dtype=np.float64)

    for candidate_key, dataset in loaded:
        global_labels = np.full(dataset.labels.shape, -1, dtype=np.int64)
        for row_index, label in enumerate(dataset.labels):
            material = dataset.material_names[int(label)]
            if material in material_to_label:
                global_labels[row_index] = material_to_label[material]
        keep_mask = global_labels >= 0

        if not np.any(keep_mask):
            continue

        x_rows.append(dataset.x[keep_mask, :min_neurons, :min_windows])
        label_rows.append(global_labels[keep_mask])
        sample_rows.append(dataset.sample_indices[keep_mask])
        source_files.extend([dataset.source_files[index] for index in np.where(keep_mask)[0]])
        candidate_keys.extend([candidate_key] * int(np.sum(keep_mask)))

    if not x_rows:
        raise FileNotFoundError("No internal-state samples remained after material filtering.")

    x = np.concatenate(x_rows, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_rows, axis=0).astype(np.int64, copy=False)
    sample_indices = np.concatenate(sample_rows, axis=0).astype(np.int64, copy=False)
    candidate_key_arr = np.asarray(candidate_keys, dtype=str)
    used_labels = [int(item) for item in sorted(np.unique(labels).tolist())]
    label_remap = {old: new for new, old in enumerate(used_labels)}
    labels = np.asarray([label_remap[int(label)] for label in labels], dtype=np.int64)
    material_order = [material_order[old] for old in used_labels]

    dataset = WindowedInternalStateDataset(
        x=x,
        labels=labels,
        material_names=material_order,
        sample_indices=sample_indices,
        source_files=source_files,
        window_centers_ms=window_centers_ref,
        n_neurons=int(min_neurons),
        n_windows=int(min_windows),
    )
    return dataset, candidate_key_arr


def standardize_internal_state(
    x: np.ndarray,
    *,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全サンプル・全時刻を使って、ニューロンごとに標準化する。"""

    arr = np.asarray(x, dtype=np.float32)
    mean = np.mean(arr, axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.std(arr, axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std > float(eps), std, 1.0).astype(np.float32)
    return ((arr - mean) / std).astype(np.float32), mean, std


def _build_vae_model(nn, *, n_neurons: int, n_windows: int, latent_dim: int, hidden_channels: int):
    class Conv1dVAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder: Nニューロン x K時間窓の活動を畳み込みで読み、潜在分布の特徴へ圧縮する。
            self.encoder = nn.Sequential(
                nn.Conv1d(n_neurons, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc_mu = nn.Linear(hidden_channels, latent_dim)
            self.fc_logvar = nn.Linear(hidden_channels, latent_dim)
            # Decoder: 潜在変数 z から元の N x K 内部状態を復元する。
            self.decoder_fc = nn.Linear(latent_dim, hidden_channels * n_windows)
            self.decoder = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, n_neurons, kernel_size=3, padding=1),
            )

        def encode(self, x):
            h = self.encoder(x).squeeze(-1)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h).clamp(min=-10.0, max=10.0)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            # z = mu + sigma * epsilon。これで確率的にサンプリングしつつ逆伝播できる。
            std = (0.5 * logvar).exp()
            eps = std.new_empty(std.shape).normal_()
            return mu + eps * std

        def decode(self, z):
            h = self.decoder_fc(z).view(z.shape[0], hidden_channels, n_windows)
            return self.decoder(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

    return Conv1dVAE()


def _resolve_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_vae(
    x: np.ndarray,
    *,
    latent_dim: int = 2,
    hidden_channels: int = 64,
    beta: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    grad_clip: float | None = 5.0,
    progress_interval: int = 1,
) -> dict:
    # VAE の学習本体。損失は MSE（復元誤差） + beta * KL（潜在分布の正則化）。
    torch, nn, F, DataLoader, TensorDataset = _import_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 3:
        raise ValueError(f"VAE input must be 3D (batch, N, K), got shape={x_arr.shape}")
    n_samples, n_neurons, n_windows = x_arr.shape
    if n_samples <= 0:
        raise ValueError("VAE input has no samples.")

    dev = _resolve_device(torch, device)
    model = _build_vae_model(
        nn,
        n_neurons=int(n_neurons),
        n_windows=int(n_windows),
        latent_dim=int(latent_dim),
        hidden_channels=int(hidden_channels),
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    tensor_dataset = TensorDataset(torch.from_numpy(x_arr))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        tensor_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    history = []
    total_epochs = int(epochs)
    progress_interval = max(0, int(progress_interval))
    for epoch in range(1, total_epochs + 1):
        # 1 epoch 分、全サンプルをミニバッチで回して重みを更新する。
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_kl = 0.0
        seen = 0
        for (xb,) in loader:
            xb = xb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(xb)
            mse = F.mse_loss(recon, xb, reduction="mean")
            kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = mse + float(beta) * kl
            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            batch_n = int(xb.shape[0])
            seen += batch_n
            epoch_loss += float(loss.detach().cpu()) * batch_n
            epoch_mse += float(mse.detach().cpu()) * batch_n
            epoch_kl += float(kl.detach().cpu()) * batch_n

        denom = max(seen, 1)
        row = {
            "epoch": int(epoch),
            "loss": epoch_loss / denom,
            "mse": epoch_mse / denom,
            "kl": epoch_kl / denom,
            "beta": float(beta),
        }
        history.append(row)
        should_print = (
            progress_interval > 0
            and (epoch == 1 or epoch == total_epochs or epoch % progress_interval == 0)
        )
        if should_print:
            print(
                f"[vae-train] epoch {epoch:04d}/{total_epochs:04d} "
                f"loss={row['loss']:.6g} mse={row['mse']:.6g} kl={row['kl']:.6g}",
                flush=True,
            )

    return {"model": model, "history": history, "device": str(dev)}


def save_vae_training_curve(history: Sequence[dict], out_dir: Path, *, stem: str = "common_vae_training_loss") -> str:
    # Save a compact training-progress figure from the epoch history.
    if not history:
        return ""
    plt = try_import_pyplot()
    if plt is None:
        return ""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    if df.empty or "epoch" not in df:
        return ""

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    axes[0].plot(df["epoch"], df["loss"], label="loss = MSE + beta KL", color="tab:blue", linewidth=2.0)
    axes[0].plot(df["epoch"], df["mse"], label="MSE", color="tab:orange", linewidth=1.5)
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["kl"], label="KL", color="tab:green", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("VAE training progress")
    fig.tight_layout()
    out_fp = out_dir / f"{stem}.png"
    fig.savefig(out_fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_fp)


def encode_mu(model, x: np.ndarray, *, batch_size: int = 128, device: str = "cpu") -> np.ndarray:
    # 学習後はサンプリングした z ではなく、安定した代表値 mu を潜在表現として使う。
    torch, _nn, _F, DataLoader, TensorDataset = _import_torch()
    dev = torch.device(device)
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)
    chunks = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev)
            mu, _logvar = model.encode(xb)
            chunks.append(mu.detach().cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def _resolve_fixed_encoder_paths(encoder_source: str | Path) -> tuple[Path, Path | None]:
    # 固定Encoderはディレクトリ指定でも .pt 指定でも読めるようにする。
    source = Path(encoder_source)
    if source.is_dir():
        for model_name, latent_name in (
            ("common_vae_model.pt", "common_vae_latent_mu.npz"),
            ("vae_model.pt", "vae_latent_mu.npz"),
        ):
            model_fp = source / model_name
            if model_fp.exists():
                latent_fp = source / latent_name
                return model_fp, latent_fp if latent_fp.exists() else None
        raise FileNotFoundError(f"No VAE model file found under {source}")
    latent_candidates = [
        source.with_name("common_vae_latent_mu.npz"),
        source.with_name("vae_latent_mu.npz"),
    ]
    latent_fp = next((fp for fp in latent_candidates if fp.exists()), None)
    return source, latent_fp


def load_fixed_vae_encoder(
    encoder_source: str | Path,
    *,
    device: str = "auto",
) -> dict:
    # 事前学習済みVAEをロードし、探索中は重みを固定したEncoderとして使う。
    torch, nn, _F, _DataLoader, _TensorDataset = _import_torch()
    model_fp, latent_fp = _resolve_fixed_encoder_paths(encoder_source)
    dev = _resolve_device(torch, device)
    try:
        checkpoint = torch.load(model_fp, map_location=dev, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_fp, map_location=dev)

    params = dict(checkpoint.get("params", {}))
    shape = params.get("input_shape_batch_N_K", [0, 0, 1])
    n_neurons = int(params.get("n_neurons") or shape[1])
    n_windows = int(params.get("n_windows") or shape[2])
    latent_dim = int(params.get("latent_dim", 16))
    hidden_channels = int(params.get("hidden_channels", 64))
    if n_neurons <= 0 or n_windows <= 0:
        raise ValueError(f"Cannot infer VAE input shape from {model_fp}")

    model = _build_vae_model(
        nn,
        n_neurons=n_neurons,
        n_windows=n_windows,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
    ).to(dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.zeros((1, n_neurons, 1), dtype=np.float32)
    std = np.ones((1, n_neurons, 1), dtype=np.float32)
    if latent_fp is not None:
        with np.load(latent_fp, allow_pickle=True) as data:
            if "standardize_mean" in data:
                mean = np.asarray(data["standardize_mean"], dtype=np.float32)
            if "standardize_std" in data:
                std = np.asarray(data["standardize_std"], dtype=np.float32)

    return {
        "model": model,
        "model_file": str(model_fp),
        "latent_file": str(latent_fp) if latent_fp is not None else "",
        "params": params,
        "device": str(dev),
        "n_neurons": n_neurons,
        "n_windows": n_windows,
        "latent_dim": latent_dim,
        "hidden_channels": hidden_channels,
        "standardize_mean": mean,
        "standardize_std": std,
    }


def _match_fixed_encoder_shape(x: np.ndarray, n_neurons: int) -> np.ndarray:
    # Encoderの入力チャンネル数(Nニューロン)に合わせる。通常は同じだが、念のため不足分は0で埋める。
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[1] == int(n_neurons):
        return arr
    if arr.shape[1] > int(n_neurons):
        return arr[:, : int(n_neurons), :]
    padded = np.zeros((arr.shape[0], int(n_neurons), arr.shape[2]), dtype=np.float32)
    padded[:, : arr.shape[1], :] = arr
    return padded


def _apply_fixed_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # 固定VAE学習時の標準化統計量をそのまま使い、探索中の候補を同じスケールに写す。
    arr = np.asarray(x, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.shape[1] != arr.shape[1]:
        mean_arr = _match_fixed_encoder_shape(mean_arr, arr.shape[1])
    if std_arr.shape[1] != arr.shape[1]:
        std_arr = _match_fixed_encoder_shape(std_arr, arr.shape[1])
    std_arr = np.where(std_arr > EPS, std_arr, 1.0).astype(np.float32)
    return ((arr - mean_arr) / std_arr).astype(np.float32)


def _silhouette_score_numpy(z: np.ndarray, labels: np.ndarray) -> float:
    features = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels)
    if features.shape[0] < 2 or len(np.unique(y)) < 2:
        return float("nan")

    diff = features[:, None, :] - features[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    scores = []
    for index in range(features.shape[0]):
        same = y == y[index]
        same[index] = False
        if np.any(same):
            a = float(np.mean(distances[index, same]))
        else:
            a = 0.0

        b_values = [
            float(np.mean(distances[index, y == label]))
            for label in np.unique(y)
            if label != y[index] and np.any(y == label)
        ]
        if not b_values:
            continue
        b = min(b_values)
        denom = max(a, b)
        scores.append(0.0 if denom <= EPS else (b - a) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def silhouette_score_safe(z: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import silhouette_score

        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(np.asarray(z, dtype=np.float64), np.asarray(labels)))
    except (ImportError, ModuleNotFoundError, ValueError):
        return _silhouette_score_numpy(z, labels)


def latent_metrics(z: np.ndarray, labels: np.ndarray, material_names: Sequence[str]) -> dict:
    # 潜在空間 z 上で素材ごとのまとまり・分離を評価する。
    z_arr = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    classes = [z_arr[y == class_index] for class_index in range(len(material_names))]
    scatter = scatter_metrics(classes)
    return {
        "silhouette": silhouette_score_safe(z_arr, y),
        "DR": scatter["DR"],
        "trace_Sb": scatter["trace_Sb"],
        "trace_Sw": scatter["trace_Sw"],
        "n_classes": scatter["n_classes"],
        "n_samples_total": scatter["n_samples_total"],
        "n_features": scatter["n_features"],
        "class_counts": scatter["class_counts"],
    }


def save_latent_csv(
    out_dir: Path,
    z: np.ndarray,
    dataset: WindowedInternalStateDataset,
    *,
    stem: str = "vae_latent_mu",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_index in range(z.shape[0]):
        label_index = int(dataset.labels[row_index])
        row = {
            "label": label_index,
            "material": dataset.material_names[label_index],
            "sample_index_in_material": int(dataset.sample_indices[row_index]),
            "source_file": dataset.source_files[row_index],
        }
        for dim_index in range(z.shape[1]):
            row[f"z{dim_index + 1}"] = float(z[row_index, dim_index])
        rows.append(row)

    csv_fp = out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_fp, index=False)
    return csv_fp


def save_common_latent_csv(
    out_dir: Path,
    z: np.ndarray,
    dataset: WindowedInternalStateDataset,
    candidate_keys: Sequence[str],
    *,
    stem: str = "common_vae_latent_mu",
) -> Path:
    # 共通VAEでは、各点がどのCMA-ES候補から来たかも一緒に保存する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(candidate_keys)

    rows = []
    for row_index in range(z.shape[0]):
        label_index = int(dataset.labels[row_index])
        row = {
            "candidate_key": str(keys[row_index]),
            "label": label_index,
            "material": dataset.material_names[label_index],
            "sample_index_in_material": int(dataset.sample_indices[row_index]),
            "source_file": dataset.source_files[row_index],
        }
        for dim_index in range(z.shape[1]):
            row[f"z{dim_index + 1}"] = float(z[row_index, dim_index])
        rows.append(row)

    csv_fp = out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_fp, index=False)
    return csv_fp


def latent_grid_shape(latent_dim: int, requested: Sequence[int] | None = None) -> tuple[int, int] | None:
    if requested is not None and len(requested) == 2:
        rows = int(requested[0])
        cols = int(requested[1])
        if rows > 0 and cols > 0 and rows * cols == int(latent_dim):
            return rows, cols
    side = int(round(np.sqrt(int(latent_dim))))
    if side * side == int(latent_dim):
        return side, side
    return None


def save_latent_plot(
    out_dir: Path,
    z: np.ndarray,
    labels: np.ndarray,
    material_names: Sequence[str],
    *,
    stem: str = "vae_latent_mu_z1_z2",
) -> Path | None:
    if z.shape[1] < 2:
        return None
    plt = try_import_pyplot()
    if plt is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y = np.asarray(labels)
    for class_index, material in enumerate(material_names):
        mask = y == class_index
        if not np.any(mask):
            continue
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=34,
            marker=DEFAULT_MARKERS[class_index % len(DEFAULT_MARKERS)],
            color=DEFAULT_COLORS[class_index % len(DEFAULT_COLORS)],
            alpha=0.82,
            linewidths=0.8,
            label=str(material),
        )
    ax.set_xlabel("z1 = mu1")
    ax.set_ylabel("z2 = mu2")
    ax.set_title("Internal-state VAE latent space")
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    out_fp = out_dir / f"{stem}.png"
    fig.savefig(out_fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_fp


def _fit_pca_numpy(x: np.ndarray, *, n_components: int = 2) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"PCA input must be 2D, got shape={arr.shape}")
    requested_components = max(1, int(n_components))
    mean = np.mean(arr, axis=0, keepdims=True)
    centered = arr - mean
    scale = np.std(centered, axis=0, keepdims=True)
    scale = np.where(scale > EPS, scale, 1.0)
    standardized = centered / scale

    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    fitted_components = min(requested_components, vt.shape[0])
    components = vt[:fitted_components]
    scores = standardized @ components.T
    if fitted_components < requested_components:
        scores = np.pad(scores, ((0, 0), (0, requested_components - fitted_components)), mode="constant")
        components = np.pad(components, ((0, requested_components - fitted_components), (0, 0)), mode="constant")

    denom = max(arr.shape[0] - 1, 1)
    explained_variance = (singular_values[:fitted_components] ** 2) / float(denom)
    total_variance = float(np.sum((singular_values ** 2) / float(denom)))
    explained_ratio = explained_variance / max(total_variance, EPS)
    if explained_ratio.size < requested_components:
        explained_ratio = np.pad(explained_ratio, (0, requested_components - explained_ratio.size), mode="constant")

    return {
        "scores": scores[:, :requested_components].astype(np.float32),
        "components": components[:requested_components].astype(np.float32),
        "mean": mean.reshape(-1).astype(np.float32),
        "scale": scale.reshape(-1).astype(np.float32),
        "explained_variance_ratio": explained_ratio[:requested_components].astype(np.float32),
    }


def _fit_pca_2d_numpy(x: np.ndarray) -> dict:
    return _fit_pca_numpy(x, n_components=2)


def save_latent_pca2(
    out_dir: Path,
    z: np.ndarray,
    labels: np.ndarray,
    material_names: Sequence[str],
    sample_indices: Sequence[int],
    source_files: Sequence[str],
    *,
    stem: str = "vae_latent_mu_pca2",
) -> dict:
    # 16次元などの VAE 潜在表現を PCA で2次元に落とし、素材ごとに散布図化する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca = _fit_pca_2d_numpy(z)
    scores = pca["scores"]
    y = np.asarray(labels, dtype=np.int64)

    rows = []
    for row_index in range(scores.shape[0]):
        label_index = int(y[row_index])
        rows.append(
            {
                "label": label_index,
                "material": material_names[label_index],
                "sample_index_in_material": int(sample_indices[row_index]),
                "source_file": source_files[row_index],
                "PC1": float(scores[row_index, 0]),
                "PC2": float(scores[row_index, 1]),
            }
        )
    csv_fp = out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_fp, index=False)

    model_fp = out_dir / f"{stem}.npz"
    np.savez_compressed(
        model_fp,
        scores=scores,
        labels=y,
        material_names=np.asarray(material_names),
        sample_indices=np.asarray(sample_indices),
        source_files=np.asarray(source_files),
        components=pca["components"],
        mean=pca["mean"],
        scale=pca["scale"],
        explained_variance_ratio=pca["explained_variance_ratio"],
    )

    plot_fp = None
    plt = try_import_pyplot()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        for class_index, material in enumerate(material_names):
            mask = y == class_index
            if not np.any(mask):
                continue
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                s=34,
                marker=DEFAULT_MARKERS[class_index % len(DEFAULT_MARKERS)],
                color=DEFAULT_COLORS[class_index % len(DEFAULT_COLORS)],
                alpha=0.82,
                linewidths=0.8,
                label=str(material),
            )
        ratio = pca["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({float(ratio[0]) * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({float(ratio[1]) * 100:.1f}%)")
        ax.set_title("PCA of VAE latent mu")
        ax.legend(fontsize=10, ncol=2)
        fig.tight_layout()
        plot_fp = out_dir / f"{stem}.png"
        fig.savefig(plot_fp, dpi=160, bbox_inches="tight")
        plt.close(fig)

    return {
        "csv_file": str(csv_fp),
        "model_file": str(model_fp),
        "plot_file": str(plot_fp) if plot_fp is not None else "",
        "explained_variance_ratio": pca["explained_variance_ratio"],
    }


def save_latent_pca3(
    out_dir: Path,
    z: np.ndarray,
    labels: np.ndarray,
    material_names: Sequence[str],
    sample_indices: Sequence[int],
    source_files: Sequence[str],
    *,
    stem: str = "vae_latent_mu_pca3",
) -> dict:
    # 2Dでは重なって見える場合に確認できるよう、3次元 PCA 図も保存する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca = _fit_pca_numpy(z, n_components=3)
    scores = pca["scores"]
    y = np.asarray(labels, dtype=np.int64)

    rows = []
    for row_index in range(scores.shape[0]):
        label_index = int(y[row_index])
        rows.append(
            {
                "label": label_index,
                "material": material_names[label_index],
                "sample_index_in_material": int(sample_indices[row_index]),
                "source_file": source_files[row_index],
                "PC1": float(scores[row_index, 0]),
                "PC2": float(scores[row_index, 1]),
                "PC3": float(scores[row_index, 2]),
            }
        )
    csv_fp = out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_fp, index=False)

    model_fp = out_dir / f"{stem}.npz"
    np.savez_compressed(
        model_fp,
        scores=scores,
        labels=y,
        material_names=np.asarray(material_names),
        sample_indices=np.asarray(sample_indices),
        source_files=np.asarray(source_files),
        components=pca["components"],
        mean=pca["mean"],
        scale=pca["scale"],
        explained_variance_ratio=pca["explained_variance_ratio"],
    )

    plot_fp = None
    plt = try_import_pyplot()
    if plt is not None:
        fig = plt.figure(figsize=(8.8, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        for class_index, material in enumerate(material_names):
            mask = y == class_index
            if not np.any(mask):
                continue
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                scores[mask, 2],
                s=34,
                marker=DEFAULT_MARKERS[class_index % len(DEFAULT_MARKERS)],
                color=DEFAULT_COLORS[class_index % len(DEFAULT_COLORS)],
                alpha=0.82,
                linewidths=0.8,
                label=str(material),
            )
        ratio = pca["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({float(ratio[0]) * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({float(ratio[1]) * 100:.1f}%)")
        ax.set_zlabel(f"PC3 ({float(ratio[2]) * 100:.1f}%)")
        ax.set_title("3D PCA of VAE latent mu")
        ax.legend(fontsize=9, ncol=2, loc="best")
        fig.tight_layout()
        plot_fp = out_dir / f"{stem}.png"
        fig.savefig(plot_fp, dpi=160, bbox_inches="tight")
        plt.close(fig)

    return {
        "csv_file": str(csv_fp),
        "model_file": str(model_fp),
        "plot_file": str(plot_fp) if plot_fp is not None else "",
        "explained_variance_ratio": pca["explained_variance_ratio"],
    }


def write_used_parameters(out_dir: Path, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "used_parameters.txt"
    fp.write_text(
        "Used Parameters\n"
        "===============\n\n"
        + json.dumps(jsonable(payload), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return fp


def output_dir_name(
    *,
    window_ms: float,
    step_ms: float,
    latent_dim: int,
    beta: float,
) -> str:
    if abs(float(window_ms) - float(step_ms)) <= 1e-9:
        return (
            f"bin_{value_stem(float(window_ms))}ms"
            f"__latent_{int(latent_dim)}"
            f"__beta_{value_stem(float(beta))}"
        )
    return (
        f"window_{value_stem(float(window_ms))}ms"
        f"__step_{value_stem(float(step_ms))}ms"
        f"__latent_{int(latent_dim)}"
        f"__beta_{value_stem(float(beta))}"
    )


def train_common_internal_state_vae(
    entries: Sequence[dict],
    out_dir: Path,
    *,
    dataset_id: str | None = None,
    window_ms: float = 10.0,
    step_ms: float = 10.0,
    latent_dim: int = 16,
    hidden_channels: int = 64,
    beta: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    standardize: bool = True,
    max_samples_per_class: int | None = None,
    materials: Sequence[str] | None = None,
    file_glob: str = "*_liquid_internal_state_all.npz",
    latent_grid: Sequence[int] | None = (4, 4),
    progress_interval: int = 1,
) -> dict:
    # すべての候補パラメータの内部状態を結合し、同じEncoderで共通潜在空間を作る。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset, candidate_keys = combine_windowed_internal_state_datasets(
        entries,
        window_ms=window_ms,
        step_ms=step_ms,
        materials=materials,
        max_samples_per_class=max_samples_per_class,
        file_glob=file_glob,
    )

    if standardize:
        # 共通潜在空間なので、標準化も候補ごとではなく全候補をまとめた統計量で行う。
        x_train, mean, std = standardize_internal_state(dataset.x)
    else:
        x_train = np.asarray(dataset.x, dtype=np.float32)
        mean = np.zeros((1, dataset.n_neurons, 1), dtype=np.float32)
        std = np.ones((1, dataset.n_neurons, 1), dtype=np.float32)

    train_result = train_vae(
        x_train,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        progress_interval=progress_interval,
    )
    model = train_result["model"]
    resolved_device = str(train_result["device"])
    z_mu = encode_mu(model, x_train, batch_size=max(batch_size, 1), device=resolved_device)

    overall_metrics = latent_metrics(z_mu, dataset.labels, dataset.material_names)
    entry_by_key = {
        _candidate_key(entry, entry_index): dict(entry)
        for entry_index, entry in enumerate(entries)
    }
    per_candidate_metrics: dict[str, dict] = {}
    per_candidate_rows: list[dict] = []
    for candidate_key in sorted(dict.fromkeys(candidate_keys.tolist())):
        mask = candidate_keys == candidate_key
        labels_subset, names_subset = _visible_material_names(dataset.labels[mask], dataset.material_names)
        metrics = latent_metrics(z_mu[mask], labels_subset, names_subset)
        entry = entry_by_key.get(str(candidate_key), {})
        metrics_payload = {
            "candidate_key": str(candidate_key),
            "generation": entry.get("generation"),
            "candidate_index": entry.get("candidate_index"),
            "run_out_dir": entry.get("run_out_dir", ""),
            "internal_state_dir": entry.get("internal_state_dir", ""),
            "params_json": entry.get("params_json", "{}"),
            "materials": names_subset,
            **metrics,
        }
        per_candidate_metrics[str(candidate_key)] = metrics_payload
        per_candidate_rows.append(
            {
                "candidate_key": str(candidate_key),
                "generation": entry.get("generation"),
                "candidate_index": entry.get("candidate_index"),
                "silhouette": float(metrics["silhouette"]),
                "DR": float(metrics["DR"]),
                "trace_Sb": float(metrics["trace_Sb"]),
                "trace_Sw": float(metrics["trace_Sw"]),
                "n_classes": int(metrics["n_classes"]),
                "n_samples_total": int(metrics["n_samples_total"]),
                "n_features": int(metrics["n_features"]),
                "class_counts_json": json.dumps(jsonable(metrics["class_counts"]), ensure_ascii=False),
                "materials_json": json.dumps(jsonable(names_subset), ensure_ascii=False),
                "run_out_dir": entry.get("run_out_dir", ""),
                "internal_state_dir": entry.get("internal_state_dir", ""),
                "params_json": entry.get("params_json", "{}"),
            }
        )

    params = {
        "dataset_id": dataset_id,
        "out_dir": str(out_dir.resolve()),
        "window_ms": float(window_ms),
        "step_ms": float(step_ms),
        "latent_dim": int(latent_dim),
        "hidden_channels": int(hidden_channels),
        "beta": float(beta),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "device": str(device),
        "resolved_device": resolved_device,
        "standardize": bool(standardize),
        "max_samples_per_class": max_samples_per_class,
        "materials": dataset.material_names,
        "candidate_keys": sorted(dict.fromkeys(candidate_keys.tolist())),
        "candidate_count": int(len(set(candidate_keys.tolist()))),
        "progress_interval": int(progress_interval),
        "input_shape_batch_N_K": list(dataset.x.shape),
        "n_neurons": int(dataset.n_neurons),
        "n_windows": int(dataset.n_windows),
        "window_centers_ms": dataset.window_centers_ms,
        "entries": [dict(entry) for entry in entries],
    }
    used_params_fp = write_used_parameters(out_dir, params)

    history_fp = out_dir / "common_vae_training_loss.csv"
    pd.DataFrame(train_result["history"]).to_csv(history_fp, index=False)
    history_plot_fp = save_vae_training_curve(
        train_result["history"],
        out_dir,
        stem="common_vae_training_loss",
    )

    latent_csv_fp = save_common_latent_csv(out_dir, z_mu, dataset, candidate_keys)
    latent_plot_fp = save_latent_plot(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        stem="common_vae_latent_mu_z1_z2",
    )
    latent_pca2 = save_latent_pca2(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        dataset.sample_indices,
        dataset.source_files,
        stem="common_vae_latent_mu_pca2",
    )
    latent_pca3 = save_latent_pca3(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        dataset.sample_indices,
        dataset.source_files,
        stem="common_vae_latent_mu_pca3",
    )

    grid_shape = latent_grid_shape(int(latent_dim), latent_grid)
    z_mu_grid = None
    if grid_shape is not None:
        z_mu_grid = z_mu.reshape(z_mu.shape[0], grid_shape[0], grid_shape[1])

    latent_npz_fp = out_dir / "common_vae_latent_mu.npz"
    np.savez_compressed(
        latent_npz_fp,
        z_mu=z_mu.astype(np.float32),
        z_mu_grid=z_mu_grid.astype(np.float32) if z_mu_grid is not None else np.asarray([], dtype=np.float32),
        latent_grid_shape=np.asarray(grid_shape if grid_shape is not None else (), dtype=np.int64),
        candidate_keys=np.asarray(candidate_keys, dtype=str),
        labels=dataset.labels,
        material_names=np.asarray(dataset.material_names),
        sample_indices=dataset.sample_indices,
        source_files=np.asarray(dataset.source_files),
        window_centers_ms=dataset.window_centers_ms,
        x_shape=np.asarray(dataset.x.shape, dtype=np.int64),
        standardize_mean=mean.astype(np.float32),
        standardize_std=std.astype(np.float32),
    )

    metrics_payload = {
        "overall": overall_metrics,
        "per_candidate_metrics": per_candidate_metrics,
        "candidate_count": int(len(per_candidate_metrics)),
        "input_shape_batch_N_K": list(dataset.x.shape),
        "latent_dim": int(latent_dim),
        "latent_grid_shape": list(grid_shape) if grid_shape is not None else [],
    }
    metrics_fp = out_dir / "common_vae_latent_metrics.json"
    metrics_fp.write_text(
        json.dumps(jsonable(metrics_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    per_candidate_csv_fp = out_dir / "common_vae_per_candidate_metrics.csv"
    pd.DataFrame(per_candidate_rows).to_csv(per_candidate_csv_fp, index=False)

    torch, _nn, _F, _DataLoader, _TensorDataset = _import_torch()
    model_fp = out_dir / "common_vae_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": jsonable(params),
            "metrics": jsonable(metrics_payload),
        },
        model_fp,
    )

    return {
        "dataset_id": dataset_id or "",
        "out_dir": str(out_dir),
        "used_parameters_file": str(used_params_fp),
        "history_file": str(history_fp),
        "history_plot_file": str(history_plot_fp),
        "latent_csv_file": str(latent_csv_fp),
        "latent_npz_file": str(latent_npz_fp),
        "latent_plot_file": str(latent_plot_fp) if latent_plot_fp is not None else "",
        "latent_pca2_csv_file": latent_pca2["csv_file"],
        "latent_pca2_npz_file": latent_pca2["model_file"],
        "latent_pca2_plot_file": latent_pca2["plot_file"],
        "latent_pca2_explained_variance_ratio": latent_pca2["explained_variance_ratio"],
        "latent_pca3_csv_file": latent_pca3["csv_file"],
        "latent_pca3_npz_file": latent_pca3["model_file"],
        "latent_pca3_plot_file": latent_pca3["plot_file"],
        "latent_pca3_explained_variance_ratio": latent_pca3["explained_variance_ratio"],
        "metrics_file": str(metrics_fp),
        "per_candidate_metrics_file": str(per_candidate_csv_fp),
        "model_file": str(model_fp),
        "overall_metrics": overall_metrics,
        "per_candidate_metrics": per_candidate_metrics,
        "silhouette": float(overall_metrics["silhouette"]),
        "DR": float(overall_metrics["DR"]),
        "trace_Sb": float(overall_metrics["trace_Sb"]),
        "trace_Sw": float(overall_metrics["trace_Sw"]),
        "input_shape_batch_N_K": list(dataset.x.shape),
        "candidate_count": int(len(per_candidate_metrics)),
        "latent_dim": int(latent_dim),
        "latent_grid_shape": list(grid_shape) if grid_shape is not None else [],
    }


def encode_internal_state_with_fixed_vae(
    internal_state_dir: Path,
    encoder_source: str | Path,
    out_dir: Path,
    *,
    dataset_id: str | None = None,
    window_ms: float | None = None,
    step_ms: float | None = None,
    batch_size: int = 128,
    device: str = "auto",
    max_samples_per_class: int | None = None,
    materials: Sequence[str] | None = None,
    file_glob: str = "*_liquid_internal_state_all.npz",
) -> dict:
    # 探索中はVAEを再学習せず、事前学習済みEncoderで内部状態を潜在空間へ写す。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_fixed_vae_encoder(encoder_source, device=device)
    params = dict(encoder.get("params", {}))
    use_window_ms = float(window_ms if window_ms is not None else params.get("window_ms", 10.0))
    use_step_ms = float(step_ms if step_ms is not None else params.get("step_ms", use_window_ms))
    dataset = load_windowed_internal_state_dataset(
        Path(internal_state_dir),
        window_ms=use_window_ms,
        step_ms=use_step_ms,
        materials=materials,
        max_samples_per_class=max_samples_per_class,
        file_glob=file_glob,
    )

    x_eval = _match_fixed_encoder_shape(dataset.x, int(encoder["n_neurons"]))
    if bool(params.get("standardize", True)):
        x_eval = _apply_fixed_standardizer(
            x_eval,
            encoder["standardize_mean"],
            encoder["standardize_std"],
        )

    z_mu = encode_mu(
        encoder["model"],
        x_eval,
        batch_size=max(int(batch_size), 1),
        device=str(encoder["device"]),
    )
    metrics = latent_metrics(z_mu, dataset.labels, dataset.material_names)

    used_params_fp = write_used_parameters(
        out_dir,
        {
            "dataset_id": dataset_id,
            "internal_state_dir": str(Path(internal_state_dir).resolve()),
            "out_dir": str(out_dir.resolve()),
            "fixed_encoder_source": str(encoder_source),
            "fixed_encoder_model_file": encoder["model_file"],
            "fixed_encoder_latent_file": encoder["latent_file"],
            "window_ms": use_window_ms,
            "step_ms": use_step_ms,
            "batch_size": int(batch_size),
            "device": str(device),
            "resolved_device": encoder["device"],
            "max_samples_per_class": max_samples_per_class,
            "materials": dataset.material_names,
            "input_shape_batch_N_K": list(dataset.x.shape),
            "encoder_input_shape_batch_N_K": list(x_eval.shape),
            "latent_dim": int(encoder["latent_dim"]),
        },
    )
    latent_csv_fp = save_latent_csv(out_dir, z_mu, dataset, stem="fixed_vae_latent_mu")
    latent_plot_fp = save_latent_plot(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        stem="fixed_vae_latent_mu_z1_z2",
    )
    latent_pca2 = save_latent_pca2(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        dataset.sample_indices,
        dataset.source_files,
        stem="fixed_vae_latent_mu_pca2",
    )
    latent_pca3 = save_latent_pca3(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        dataset.sample_indices,
        dataset.source_files,
        stem="fixed_vae_latent_mu_pca3",
    )

    latent_npz_fp = out_dir / "fixed_vae_latent_mu.npz"
    np.savez_compressed(
        latent_npz_fp,
        z_mu=z_mu.astype(np.float32),
        labels=dataset.labels,
        material_names=np.asarray(dataset.material_names),
        sample_indices=dataset.sample_indices,
        source_files=np.asarray(dataset.source_files),
        window_centers_ms=dataset.window_centers_ms,
        x_shape=np.asarray(dataset.x.shape, dtype=np.int64),
        fixed_encoder_model_file=np.asarray([encoder["model_file"]]),
        fixed_encoder_latent_file=np.asarray([encoder["latent_file"]]),
    )

    metrics_payload = {
        "fixed_encoder": True,
        "fixed_encoder_model_file": encoder["model_file"],
        "fixed_encoder_latent_file": encoder["latent_file"],
        "silhouette": metrics["silhouette"],
        "DR": metrics["DR"],
        "trace_Sb": metrics["trace_Sb"],
        "trace_Sw": metrics["trace_Sw"],
        "n_classes": metrics["n_classes"],
        "n_samples_total": metrics["n_samples_total"],
        "n_features": metrics["n_features"],
        "class_counts": metrics["class_counts"],
    }
    metrics_fp = out_dir / "fixed_vae_latent_metrics.json"
    metrics_fp.write_text(
        json.dumps(jsonable(metrics_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_id": dataset_id or "",
        "out_dir": str(out_dir),
        "used_parameters_file": str(used_params_fp),
        "latent_csv_file": str(latent_csv_fp),
        "latent_npz_file": str(latent_npz_fp),
        "latent_plot_file": str(latent_plot_fp) if latent_plot_fp is not None else "",
        "latent_pca2_csv_file": latent_pca2["csv_file"],
        "latent_pca2_npz_file": latent_pca2["model_file"],
        "latent_pca2_plot_file": latent_pca2["plot_file"],
        "latent_pca2_explained_variance_ratio": latent_pca2["explained_variance_ratio"],
        "latent_pca3_csv_file": latent_pca3["csv_file"],
        "latent_pca3_npz_file": latent_pca3["model_file"],
        "latent_pca3_plot_file": latent_pca3["plot_file"],
        "latent_pca3_explained_variance_ratio": latent_pca3["explained_variance_ratio"],
        "metrics_file": str(metrics_fp),
        "fixed_encoder": True,
        "fixed_encoder_model_file": encoder["model_file"],
        "fixed_encoder_latent_file": encoder["latent_file"],
        "silhouette": float(metrics_payload["silhouette"]),
        "DR": float(metrics_payload["DR"]),
        "trace_Sb": float(metrics_payload["trace_Sb"]),
        "trace_Sw": float(metrics_payload["trace_Sw"]),
        "input_shape_batch_N_K": list(dataset.x.shape),
        "encoder_input_shape_batch_N_K": list(x_eval.shape),
        "latent_dim": int(encoder["latent_dim"]),
    }


def train_internal_state_vae(
    internal_state_dir: Path,
    out_dir: Path,
    *,
    run_dir: Path | None = None,
    dataset_id: str | None = None,
    window_ms: float = 10.0,
    step_ms: float = 10.0,
    latent_dim: int = 2,
    hidden_channels: int = 64,
    beta: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    standardize: bool = True,
    max_samples_per_class: int | None = None,
    materials: Sequence[str] | None = None,
    file_glob: str = "*_liquid_internal_state_all.npz",
    latent_grid: Sequence[int] | None = (4, 4),
    progress_interval: int = 1,
) -> dict:
    # run_liquid の内部状態保存結果を入力として、VAE学習から評価・保存までを一括実行する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_windowed_internal_state_dataset(
        internal_state_dir,
        window_ms=window_ms,
        step_ms=step_ms,
        materials=materials,
        max_samples_per_class=max_samples_per_class,
        file_glob=file_glob,
    )

    if standardize:
        # ニューロンごとの活動スケール差が VAE 学習を支配しないように標準化する。
        x_train, mean, std = standardize_internal_state(dataset.x)
    else:
        x_train = np.asarray(dataset.x, dtype=np.float32)
        mean = np.zeros((1, dataset.n_neurons, 1), dtype=np.float32)
        std = np.ones((1, dataset.n_neurons, 1), dtype=np.float32)

    train_result = train_vae(
        x_train,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        progress_interval=progress_interval,
    )
    model = train_result["model"]
    resolved_device = str(train_result["device"])
    # 学習済み Encoder の mu を最終的な潜在ベクトル z として取り出す。
    z_mu = encode_mu(model, x_train, batch_size=max(batch_size, 1), device=resolved_device)
    metrics = latent_metrics(z_mu, dataset.labels, dataset.material_names)

    params = {
        "dataset_id": dataset_id,
        "run_dir": str(Path(run_dir).resolve()) if run_dir is not None else None,
        "internal_state_dir": str(Path(internal_state_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "window_ms": float(window_ms),
        "step_ms": float(step_ms),
        "latent_dim": int(latent_dim),
        "hidden_channels": int(hidden_channels),
        "beta": float(beta),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "device": str(device),
        "resolved_device": resolved_device,
        "standardize": bool(standardize),
        "max_samples_per_class": max_samples_per_class,
        "materials": dataset.material_names,
        "input_shape_batch_N_K": list(dataset.x.shape),
        "n_neurons": int(dataset.n_neurons),
        "n_windows": int(dataset.n_windows),
        "window_centers_ms": dataset.window_centers_ms,
        "source_files": dataset.source_files,
        "progress_interval": int(progress_interval),
    }
    used_params_fp = write_used_parameters(out_dir, params)

    history_fp = out_dir / "vae_training_loss.csv"
    pd.DataFrame(train_result["history"]).to_csv(history_fp, index=False)
    history_plot_fp = save_vae_training_curve(
        train_result["history"],
        out_dir,
        stem="vae_training_loss",
    )

    latent_csv_fp = save_latent_csv(out_dir, z_mu, dataset)
    latent_plot_fp = save_latent_plot(out_dir, z_mu, dataset.labels, dataset.material_names)
    # VAEの潜在空間をさらに PCA で2D/3Dに落として、素材分離を目視確認できるようにする。
    latent_pca2 = save_latent_pca2(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        dataset.sample_indices,
        dataset.source_files,
    )
    latent_pca3 = save_latent_pca3(
        out_dir,
        z_mu,
        dataset.labels,
        dataset.material_names,
        dataset.sample_indices,
        dataset.source_files,
    )
    grid_shape = latent_grid_shape(int(latent_dim), latent_grid)
    z_mu_grid = None
    if grid_shape is not None:
        z_mu_grid = z_mu.reshape(z_mu.shape[0], grid_shape[0], grid_shape[1])

    latent_npz_fp = out_dir / "vae_latent_mu.npz"
    np.savez_compressed(
        latent_npz_fp,
        z_mu=z_mu.astype(np.float32),
        z_mu_grid=z_mu_grid.astype(np.float32) if z_mu_grid is not None else np.asarray([], dtype=np.float32),
        latent_grid_shape=np.asarray(grid_shape if grid_shape is not None else (), dtype=np.int64),
        labels=dataset.labels,
        material_names=np.asarray(dataset.material_names),
        sample_indices=dataset.sample_indices,
        source_files=np.asarray(dataset.source_files),
        window_centers_ms=dataset.window_centers_ms,
        x_shape=np.asarray(dataset.x.shape, dtype=np.int64),
        standardize_mean=mean.astype(np.float32),
        standardize_std=std.astype(np.float32),
    )

    metrics_payload = {
        "silhouette": metrics["silhouette"],
        "DR": metrics["DR"],
        "trace_Sb": metrics["trace_Sb"],
        "trace_Sw": metrics["trace_Sw"],
        "n_classes": metrics["n_classes"],
        "n_samples_total": metrics["n_samples_total"],
        "n_features": metrics["n_features"],
        "class_counts": metrics["class_counts"],
    }
    metrics_fp = out_dir / "vae_latent_metrics.json"
    metrics_fp.write_text(
        json.dumps(jsonable(metrics_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    torch, _nn, _F, _DataLoader, _TensorDataset = _import_torch()
    model_fp = out_dir / "vae_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": jsonable(params),
            "metrics": jsonable(metrics_payload),
        },
        model_fp,
    )

    return {
        "dataset_id": dataset_id or "",
        "out_dir": str(out_dir),
        "used_parameters_file": str(used_params_fp),
        "history_file": str(history_fp),
        "history_plot_file": str(history_plot_fp),
        "latent_csv_file": str(latent_csv_fp),
        "latent_npz_file": str(latent_npz_fp),
        "latent_plot_file": str(latent_plot_fp) if latent_plot_fp is not None else "",
        "latent_pca2_csv_file": latent_pca2["csv_file"],
        "latent_pca2_npz_file": latent_pca2["model_file"],
        "latent_pca2_plot_file": latent_pca2["plot_file"],
        "latent_pca2_explained_variance_ratio": latent_pca2["explained_variance_ratio"],
        "latent_pca3_csv_file": latent_pca3["csv_file"],
        "latent_pca3_npz_file": latent_pca3["model_file"],
        "latent_pca3_plot_file": latent_pca3["plot_file"],
        "latent_pca3_explained_variance_ratio": latent_pca3["explained_variance_ratio"],
        "metrics_file": str(metrics_fp),
        "model_file": str(model_fp),
        "silhouette": float(metrics_payload["silhouette"]),
        "DR": float(metrics_payload["DR"]),
        "trace_Sb": float(metrics_payload["trace_Sb"]),
        "trace_Sw": float(metrics_payload["trace_Sw"]),
        "input_shape_batch_N_K": list(dataset.x.shape),
        "latent_dim": int(latent_dim),
        "latent_grid_shape": list(grid_shape) if grid_shape is not None else [],
    }
