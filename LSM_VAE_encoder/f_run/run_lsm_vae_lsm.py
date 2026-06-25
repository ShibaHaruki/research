"""Train LSM -> Spiking VAE -> LSM-style decoder on saved internal states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "PyTorch is required for run_lsm_vae_lsm.py. "
        "Install torch in the Python environment used for this project."
    ) from exc

from b_network.lsm_vae_lsm import LSMSpikingVAELSM, lsm_vae_loss
from c_configs.FIXED import cfg_run


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
LIQUID_RESULT_DIR = RESULTS_PATH / RUN_CFG["LIQUID_RESULT_DIR"]


@dataclass
class TrainConfig:
    input_dir: str | None = None
    output_dir: str | None = None
    latent_dim: int = 16
    encoder_hidden_dim: int = 128
    encoder_layers: int = 1
    decoder_liquid_dim: int = 128
    decoder_spectral_radius: float = 0.9
    decoder_leak: float = 0.25
    decoder_recurrent_density: float = 0.15
    beta: float = 1e-3
    gamma: float = 0.0
    delta: float = 0.0
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    device: str = "auto"
    train_ratio: float = 0.8
    threshold: float = 0.0
    binarize_target: bool = True
    max_samples_per_class: int | None = None
    save_examples: int = 8


def _latest_internal_state_dir() -> Path:
    candidates = sorted(
        LIQUID_RESULT_DIR.glob("**/internal_states"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No internal_states directory found under {LIQUID_RESULT_DIR}. "
            "Run f_run/run_liquid.py first with INTERNAL_STATE_ENABLE=True."
        )
    return candidates[0]


def _default_output_dir(input_dir: Path) -> Path:
    parent = input_dir.parent
    return parent / str(RUN_CFG.get("INTERNAL_STATE_VAE_DIR", "internal_state_vae_lsm"))


def _material_from_file(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    return rel.parts[0] if len(rel.parts) > 1 else path.parent.name


def _find_state_files(root: Path) -> list[Path]:
    files = sorted(root.glob("**/*_liquid_internal_state_all.npz"))
    if not files:
        raise FileNotFoundError(f"No *_liquid_internal_state_all.npz files found in {root}")
    return files


def _load_state(path: Path, *, threshold: float, binarize: bool) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        x_state = np.asarray(data["x_state"], dtype=np.float32)
    if x_state.ndim != 2:
        raise ValueError(f"x_state must be 2D in {path}, got {x_state.shape}")
    x_time = x_state.T
    if binarize:
        x_time = (x_time > float(threshold)).astype(np.float32)
    else:
        max_value = float(np.max(x_time)) if x_time.size else 0.0
        if max_value > 1.0:
            x_time = x_time / max_value
        x_time = np.clip(x_time, 0.0, 1.0).astype(np.float32)
    return x_time


class InternalStateDataset(Dataset):
    def __init__(self, root: Path, config: TrainConfig) -> None:
        self.root = Path(root)
        files = _find_state_files(self.root)
        labels = sorted({_material_from_file(fp, self.root) for fp in files})
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}

        class_counts: dict[str, int] = {label: 0 for label in labels}
        records = []
        for fp in files:
            label = _material_from_file(fp, self.root)
            if (
                config.max_samples_per_class is not None
                and class_counts[label] >= int(config.max_samples_per_class)
            ):
                continue
            x = _load_state(
                fp,
                threshold=config.threshold,
                binarize=config.binarize_target,
            )
            records.append((x, self.label_to_index[label], label, fp))
            class_counts[label] += 1

        if not records:
            raise ValueError("No samples remain after max_samples_per_class filtering.")
        time_steps = {record[0].shape[0] for record in records}
        input_dims = {record[0].shape[1] for record in records}
        if len(time_steps) != 1 or len(input_dims) != 1:
            raise ValueError(
                "All internal-state arrays must share one shape. "
                f"time_steps={sorted(time_steps)} input_dims={sorted(input_dims)}"
            )
        self.records = records
        self.time_steps = int(next(iter(time_steps)))
        self.input_dim = int(next(iter(input_dims)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        x, label_index, label, fp = self.records[index]
        return torch.from_numpy(x), torch.tensor(label_index, dtype=torch.long), label, str(fp)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _accuracy(logits: torch.Tensor | None, labels: torch.Tensor) -> float | None:
    if logits is None:
        return None
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def _run_epoch(
    model: LSMSpikingVAELSM,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    config: TrainConfig,
    *,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    metric_rows: dict[str, list[float]] = {
        "total": [],
        "rec": [],
        "kl": [],
        "cls": [],
        "map": [],
        "acc": [],
    }

    for x, labels, _, _ in loader:
        x = x.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            output = model(x)
            losses = lsm_vae_loss(
                output,
                x,
                labels,
                beta=config.beta,
                gamma=config.gamma,
                delta=config.delta,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

        for key in ("total", "rec", "kl", "cls", "map"):
            metric_rows[key].append(float(losses[key].item()))
        acc = _accuracy(output.class_logits, labels)
        if acc is not None:
            metric_rows["acc"].append(acc)

    return {key: _mean(values) for key, values in metric_rows.items() if values}


def _write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_examples(
    model: LSMSpikingVAELSM,
    dataset: Dataset,
    out_dir: Path,
    device: torch.device,
    count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for index in range(min(len(dataset), max(0, count))):
            x, label_index, label, fp = dataset[index]
            output = model(x.unsqueeze(0).to(device))
            recon = torch.sigmoid(output.recon_logits).squeeze(0).cpu().numpy()
            z = output.z.squeeze(0).cpu().numpy()
            z_spikes = output.z_spikes.squeeze(0).cpu().numpy()
            np.savez_compressed(
                out_dir / f"example_{index:03d}.npz",
                target=x.numpy(),
                recon=recon,
                z=z,
                z_spikes=z_spikes,
                label_index=np.asarray([int(label_index)], dtype=np.int32),
                label=np.asarray([label]),
                source_file=np.asarray([fp]),
            )
            saved += 1
    print(f"[examples] saved {saved} examples to {out_dir}")


def train(config: TrainConfig) -> dict:
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))

    input_dir = Path(config.input_dir) if config.input_dir else _latest_internal_state_dir()
    output_dir = Path(config.output_dir) if config.output_dir else _default_output_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = InternalStateDataset(input_dir, config)
    generator = torch.Generator().manual_seed(int(config.seed))
    train_len = max(1, int(math.floor(len(dataset) * float(config.train_ratio))))
    train_len = min(train_len, len(dataset))
    val_len = len(dataset) - train_len
    if val_len == 0 and len(dataset) > 1:
        train_len -= 1
        val_len = 1
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    device = _device(config.device)
    model = LSMSpikingVAELSM(
        input_dim=dataset.input_dim,
        latent_dim=config.latent_dim,
        encoder_hidden_dim=config.encoder_hidden_dim,
        encoder_layers=config.encoder_layers,
        decoder_liquid_dim=config.decoder_liquid_dim,
        decoder_spectral_radius=config.decoder_spectral_radius,
        decoder_leak=config.decoder_leak,
        decoder_recurrent_density=config.decoder_recurrent_density,
        num_classes=len(dataset.label_to_index) if config.gamma > 0.0 else None,
    ).to(device)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    ) if val_len else None

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    metric_rows = []
    best_val = float("inf")
    best_path = output_dir / "best_model.pt"

    metadata = {
        "config": asdict(config),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_samples": len(dataset),
        "train_samples": train_len,
        "val_samples": val_len,
        "time_steps": dataset.time_steps,
        "input_dim": dataset.input_dim,
        "labels": dataset.label_to_index,
        "device": str(device),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            train=True,
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})

        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                config=config,
                train=False,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = float(val_metrics.get("total", train_metrics["total"]))
        else:
            score = float(train_metrics["total"])

        metric_rows.append(row)
        if score < best_val:
            best_val = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "epoch": epoch,
                    "score": best_val,
                },
                best_path,
            )
        if epoch == 1 or epoch == int(config.epochs) or epoch % max(1, int(config.epochs) // 10) == 0:
            print(
                f"[epoch {epoch:03d}] "
                f"train_total={train_metrics['total']:.4f} "
                f"val_total={score:.4f}"
            )

    _write_metrics(output_dir / "metrics.csv", metric_rows)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "epoch": int(config.epochs),
            "score": metric_rows[-1].get("val_total", metric_rows[-1]["train_total"]),
        },
        output_dir / "last_model.pt",
    )
    _save_examples(model, dataset, output_dir / "examples", device, config.save_examples)
    print(f"[done] saved LSM-VAE-LSM results to {output_dir}")
    return metadata


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train LSM -> Spiking VAE -> LSM-style decoder.",
    )
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--decoder-liquid-dim", type=int, default=128)
    parser.add_argument("--decoder-spectral-radius", type=float, default=0.9)
    parser.add_argument("--decoder-leak", type=float, default=0.25)
    parser.add_argument("--decoder-recurrent-density", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--continuous-target", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=8)
    args = parser.parse_args()
    return TrainConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_liquid_dim=args.decoder_liquid_dim,
        decoder_spectral_radius=args.decoder_spectral_radius,
        decoder_leak=args.decoder_leak,
        decoder_recurrent_density=args.decoder_recurrent_density,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        train_ratio=args.train_ratio,
        threshold=args.threshold,
        binarize_target=not args.continuous_target,
        max_samples_per_class=args.max_samples_per_class,
        save_examples=args.save_examples,
    )


if __name__ == "__main__":
    train(parse_args())
