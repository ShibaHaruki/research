"""PyTorch modules for LSM -> Spiking VAE -> LSM-style decoder.

The model consumes saved LSM1 internal states shaped as ``(time, neurons)``.
It learns a low-dimensional latent spike-like sequence and reconstructs the
original liquid state through a fixed recurrent liquid decoder plus trainable
readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LSMVAEOutput:
    recon_logits: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    z_spikes: torch.Tensor
    decoder_state: torch.Tensor
    class_logits: torch.Tensor | None


class SpikeSurrogate(torch.autograd.Function):
    """Binary spike in forward pass with sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, slope: float):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.slope = slope
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        x = (membrane - ctx.threshold) * ctx.slope
        sigmoid = torch.sigmoid(x)
        grad = ctx.slope * sigmoid * (1.0 - sigmoid)
        return grad_output * grad, None, None


def surrogate_spike(
    membrane: torch.Tensor,
    *,
    threshold: float = 0.5,
    slope: float = 10.0,
) -> torch.Tensor:
    return SpikeSurrogate.apply(membrane, float(threshold), float(slope))


class SpikingVAEEncoder(nn.Module):
    """Temporal encoder that emits Gaussian latent parameters per time step."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.input_proj(x))
        h, _ = self.rnn(h)
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=10.0)


class LiquidStateDecoder(nn.Module):
    """Fixed recurrent liquid decoder with trainable input and readout maps."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        liquid_dim: int = 128,
        spectral_radius: float = 0.9,
        leak: float = 0.25,
        threshold: float = 0.5,
        spike_slope: float = 10.0,
        recurrent_density: float = 0.15,
    ) -> None:
        super().__init__()
        self.liquid_dim = int(liquid_dim)
        self.leak = float(leak)
        self.threshold = float(threshold)
        self.spike_slope = float(spike_slope)
        self.input = nn.Linear(latent_dim, liquid_dim)
        self.readout = nn.Linear(liquid_dim, output_dim)

        rec = torch.randn(liquid_dim, liquid_dim) / max(liquid_dim**0.5, 1.0)
        if recurrent_density < 1.0:
            mask = torch.rand(liquid_dim, liquid_dim) < float(recurrent_density)
            rec = rec * mask.to(rec.dtype)
        rec.fill_diagonal_(0.0)
        eigvals = torch.linalg.eigvals(rec).abs()
        radius = float(eigvals.max().real) if eigvals.numel() else 0.0
        if radius > 1e-6:
            rec = rec * (float(spectral_radius) / radius)
        self.register_buffer("recurrent", rec)

    def forward(self, z_spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time_steps, _ = z_spikes.shape
        state = z_spikes.new_zeros(batch, self.liquid_dim)
        spike = z_spikes.new_zeros(batch, self.liquid_dim)
        states = []

        for step in range(time_steps):
            drive = self.input(z_spikes[:, step, :]) + spike @ self.recurrent.T
            state = (1.0 - self.leak) * state + self.leak * drive
            spike = surrogate_spike(
                torch.sigmoid(state),
                threshold=self.threshold,
                slope=self.spike_slope,
            )
            states.append(spike)

        decoder_state = torch.stack(states, dim=1)
        return self.readout(decoder_state), decoder_state


class LSMSpikingVAELSM(nn.Module):
    """End-to-end LSM1-state compressor and LSM-style reconstructor."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        encoder_hidden_dim: int = 128,
        encoder_layers: int = 1,
        decoder_liquid_dim: int = 128,
        num_classes: int | None = None,
        latent_threshold: float = 0.0,
        spike_slope: float = 10.0,
        decoder_spectral_radius: float = 0.9,
        decoder_leak: float = 0.25,
        decoder_recurrent_density: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.latent_threshold = float(latent_threshold)
        self.spike_slope = float(spike_slope)

        self.encoder = SpikingVAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
        )
        self.decoder = LiquidStateDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            liquid_dim=decoder_liquid_dim,
            spectral_radius=decoder_spectral_radius,
            leak=decoder_leak,
            spike_slope=spike_slope,
            recurrent_density=decoder_recurrent_density,
        )
        self.classifier = (
            nn.Linear(latent_dim, self.num_classes)
            if self.num_classes is not None and self.num_classes > 0
            else None
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> LSMVAEOutput:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z_spikes = surrogate_spike(
            z,
            threshold=self.latent_threshold,
            slope=self.spike_slope,
        )
        recon_logits, decoder_state = self.decoder(z_spikes)
        class_logits = None
        if self.classifier is not None:
            class_logits = self.classifier(z.mean(dim=1))
        return LSMVAEOutput(
            recon_logits=recon_logits,
            mu=mu,
            logvar=logvar,
            z=z,
            z_spikes=z_spikes,
            decoder_state=decoder_state,
            class_logits=class_logits,
        )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def latent_map_loss(z: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    """Pull same-class latent means together and push different classes apart."""

    if labels is None or labels.numel() < 2:
        return z.new_tensor(0.0)
    pooled = z.mean(dim=1)
    distances = torch.cdist(pooled, pooled, p=2)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    same = same & ~eye
    diff = ~same & ~eye

    loss = z.new_tensor(0.0)
    if same.any():
        loss = loss + distances[same].mean()
    if diff.any():
        loss = loss + F.relu(1.0 - distances[diff]).mean()
    return loss


def lsm_vae_loss(
    output: LSMVAEOutput,
    target: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    beta: float = 1e-3,
    gamma: float = 0.0,
    delta: float = 0.0,
) -> dict[str, torch.Tensor]:
    rec = F.binary_cross_entropy_with_logits(output.recon_logits, target)
    kl = kl_divergence(output.mu, output.logvar)
    cls = target.new_tensor(0.0)
    if gamma > 0.0 and labels is not None and output.class_logits is not None:
        cls = F.cross_entropy(output.class_logits, labels)
    map_loss = latent_map_loss(output.z, labels) if delta > 0.0 else target.new_tensor(0.0)
    total = rec + float(beta) * kl + float(gamma) * cls + float(delta) * map_loss
    return {
        "total": total,
        "rec": rec.detach(),
        "kl": kl.detach(),
        "cls": cls.detach(),
        "map": map_loss.detach(),
    }
