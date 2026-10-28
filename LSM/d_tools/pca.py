"""Mean-centered PCA of trial x neuron x time-bin states, without scaling.

Accepts (n_trials, n_neurons, n_time_bins), or an already flattened
(n_trials, n_features) array. Values can be rates, counts, voltages, or other
finite real states. Each neuron/time pair becomes one feature, in C order.
Fit only on training trials when evaluating a classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    from scipy.linalg import eigh, svd
except ModuleNotFoundError as error:
    if error.name not in ('scipy', 'scipy.linalg'):
        raise

    def eigh(a, *, subset_by_index=None, check_finite=True):
        """SciPyがない環境ではNumPyで同じ固有値問題を解く。"""
        values, vectors = np.linalg.eigh(a)
        if subset_by_index is not None:
            start, stop = subset_by_index
            values, vectors = values[start:stop+1], vectors[:, start:stop+1]
        return values, vectors

    def svd(a, *, full_matrices=True, check_finite=True):
        return np.linalg.svd(a, full_matrices=full_matrices)


@dataclass(frozen=True)
class PCAModel:
    """Training mean and PCA axes; columns of axes are PC1, PC2, ... ."""

    mean: np.ndarray
    axes: np.ndarray
    explained_variance: np.ndarray
    feature_shape: tuple[int, ...] | None = None
    total_variance: float | None = None


def _state_features(states: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    data = np.asarray(states)
    if np.iscomplexobj(data):
        raise ValueError('PCA states must contain real numbers')
    data = np.asarray(data, dtype=np.float64)
    if data.ndim not in (2, 3) or any(size == 0 for size in data.shape[1:]):
        raise ValueError('Expected (trials, neurons, time_bins) or (trials, features) with nonempty feature axes')
    if not np.isfinite(data).all():
        raise ValueError('PCA states must be finite; NaN and infinity are not supported')
    feature_shape = data.shape[1:]
    return data.reshape(data.shape[0], int(np.prod(feature_shape))), feature_shape


def fit_pca(training_features: np.ndarray, n_components: int | None = None) -> PCAModel:
    """Fit 2D/3D training states; None keeps min(40, features, trials-1) PCs.

    Neuron and time-bin counts are inferred from the input, not fixed here.
    Only the mean is removed. No standardization or temporal averaging is used.
    """
    train, feature_shape = _state_features(training_features)
    if train.shape[0] < 2:
        raise ValueError('PCA requires at least two training trials')
    max_components = min(train.shape[1], train.shape[0]-1)
    if n_components is None:
        n_components = min(40, max_components)
    if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
        raise ValueError('n_components must be an integer')
    if not 1 <= n_components <= max_components:
        raise ValueError('n_components must be between 1 and min(n_features, n_training_trials-1)')
    mean = train.mean(0)
    centered = train-mean
    if train.shape[1] <= train.shape[0]:
        # Preserve the original covariance calculation for the current classifier.
        values, vectors = eigh(
            centered.T @ centered/(len(centered)-1),
            subset_by_index=[train.shape[1]-n_components, train.shape[1]-1],
            check_finite=False,
        )
        axes, variance = vectors[:, ::-1], values[::-1]
    else:
        # Avoid allocating a features x features matrix for wide state arrays.
        _, singular_values, vectors_t = svd(centered, full_matrices=False, check_finite=False)
        axes = vectors_t[:n_components].T
        variance = singular_values[:n_components]**2/(len(centered)-1)
    total_variance = float(np.einsum('ij,ij->', centered, centered)/(len(centered)-1))
    return PCAModel(mean=mean, axes=axes, explained_variance=variance,
                    feature_shape=feature_shape, total_variance=total_variance)


def transform_pca(features: np.ndarray, model: PCAModel) -> np.ndarray:
    """Project trials with matching neuron/time layout using the fitted model."""
    data, feature_shape = _state_features(features)
    expected_shape = model.feature_shape or (model.mean.size,)
    if feature_shape != expected_shape:
        raise ValueError(f'PCA state shape must match training: expected {expected_shape}, got {feature_shape}')
    return (data-model.mean) @ model.axes


def fit_transform_pca(states: np.ndarray, n_components: int | None = None, *,
                      output_dir: str | Path | None = None,
                      bin_ms: float | None = None) -> tuple[np.ndarray, PCAModel]:
    """Return (trial x PC scores, fitted model) for 2D/3D states.

    For classification, pass training states here and use transform_pca for
    evaluation states, rather than fitting the evaluation set separately.
    """
    if output_dir is not None and bin_ms is None:
        raise ValueError('bin_ms is required when saving loading plots')
    model = fit_pca(states, n_components=n_components)
    if output_dir is not None:
        save_loading_plots(model, output_dir, bin_ms=bin_ms)
    return transform_pca(states, model), model


def loading_maps(model: PCAModel, *, feature_shape=None, feature_mask=None,
                 scaled: bool = True) -> np.ndarray:
    """係数を「PC×ニューロン×時間ビン」に戻す。

    scaled=Trueは添付図と同じローディング（係数×主成分の標準偏差）。
    Falseはスコア計算に使う元の係数。時間方向の平均は取らない。
    2次元入力でPCAした場合はfeature_shape=(ニューロン数, ビン数)を指定する。
    学習時に特徴を除外した場合は、元の特徴順のboolマスクで0を補う。
    """
    shape = tuple(feature_shape) if feature_shape is not None else model.feature_shape
    if (shape is None or len(shape) != 2
            or any(not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0 for size in shape)):
        raise ValueError('A (n_neurons, n_time_bins) feature_shape is required for loading maps')
    n_features = int(np.prod(shape))
    axes = model.axes
    if feature_mask is not None:
        mask = np.asarray(feature_mask)
        if mask.dtype != np.bool_ or mask.shape != (n_features,) or mask.sum() != len(axes):
            raise ValueError('feature_mask must be a bool vector matching original and retained features')
        expanded = np.zeros((n_features, axes.shape[1]), dtype=axes.dtype)
        expanded[mask] = axes
        axes = expanded
    elif len(axes) != n_features:
        raise ValueError('feature_shape does not match PCA features; provide the training feature_mask if needed')
    if scaled:
        axes = axes * np.sqrt(np.maximum(model.explained_variance, 0.))
    return axes.T.reshape((axes.shape[1], *shape))


def temporal_loading_profiles(model: PCAModel, *, feature_shape=None,
                               feature_mask=None) -> np.ndarray:
    """各PCのローディングをニューロン方向に符号付き平均する（PC×時間）。"""
    return loading_maps(model, feature_shape=feature_shape, feature_mask=feature_mask).mean(axis=1)


def save_loading_plots(model: PCAModel, output_dir: str | Path, *, bin_ms: float,
                       feature_shape=None, feature_mask=None, max_components: int = 40,
                       population_label: str = 'Neurons', title_note: str = '',
                       dpi: int = 150) -> dict[str, Path]:
    """ローディングヒートマップ・時間波形・横並び図と数値データを保存する。

    色範囲は全PCで共通、波形は帯なしのニューロン平均。
    時間座標はビン中心。ニューロン数・ビン数・PC数を固定しない。
    分類評価では学習試行だけでfitしたmodelを渡す。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not np.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError('bin_ms must be finite and positive')
    if not isinstance(max_components, (int, np.integer)) or isinstance(max_components, bool) or max_components < 1:
        raise ValueError('max_components must be a positive integer')
    maps = loading_maps(model, feature_shape=feature_shape, feature_mask=feature_mask)
    coefficients = loading_maps(model, feature_shape=feature_shape, feature_mask=feature_mask, scaled=False)
    n_pc = min(max_components, len(maps))
    maps = maps[:n_pc]; coefficients = coefficients[:n_pc]
    profiles = maps.mean(axis=1)
    n_neurons, n_bins = maps.shape[1:]
    time_ms = (np.arange(n_bins)+.5)*float(bin_ms)
    duration_ms = n_bins*float(bin_ms)
    ratio = (np.maximum(model.explained_variance[:n_pc], 0.)/model.total_variance
             if model.total_variance is not None and model.total_variance > 0 else None)
    vmax = float(np.max(np.abs(maps))) or 1.
    columns = min(5, n_pc); rows = (n_pc+columns-1)//columns
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    prefix = f'PC01_PC{n_pc:02d}'
    paths = {
        'heatmaps': output/f'{prefix}_loading_heatmaps.png',
        'temporal_profiles': output/f'{prefix}_temporal_loading_profiles.png',
        'combined': output/f'{prefix}_loadings_and_temporal_profiles.png',
        'data': output/f'{prefix}_loading_data.npz',
    }
    note = f' | {title_note}' if title_note else ''

    def draw(container, kind):
        axes = np.asarray(container.subplots(rows, columns, squeeze=False)).ravel()
        image = None
        for k, ax in enumerate(axes):
            if k >= n_pc:
                ax.set_visible(False); continue
            label = f'PC{k+1}' + (f' ({ratio[k]*100:.2f}%)' if ratio is not None else '')
            ax.set_title(label, fontsize=9)
            if kind == 'heatmap':
                image = ax.imshow(maps[k], origin='lower', aspect='auto', interpolation='nearest',
                    extent=(0, duration_ms, -.5, n_neurons-.5), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                if k % columns == 0:
                    ax.set_ylabel('Neuron index', fontsize=8)
            else:
                ax.plot(time_ms, profiles[k], color='tab:blue', lw=1.,
                        marker='o' if n_bins == 1 else None)
                ax.axhline(0, color='0.5', lw=.6)
                ax.set_xlim(0, duration_ms)
                ax.grid(True, alpha=.18)
                if k % columns == 0:
                    ax.set_ylabel('Mean loading', fontsize=8)
            ax.tick_params(labelsize=7)
            if k//columns == rows-1:
                ax.set_xlabel('Time within trial (ms)', fontsize=8)
        if kind == 'heatmap':
            container.colorbar(image, ax=list(axes[:n_pc]), shrink=.8,
                label='Loading (coefficient x component standard deviation)')

    # 1枚ずつでも横並びでも、列数・行数と基本の小グラフの比をそろえる。
    width, height = columns*3., max(3., rows*2.5)
    for key, kind, title in [
            ('heatmaps', 'heatmap', 'PCA loadings: neuron x time'),
            ('temporal_profiles', 'waveform', 'Temporal loading profiles of PCA components')]:
        fig = plt.figure(figsize=(width, height), layout='constrained')
        try:
            fig.suptitle(f'{title} | {population_label} | PC1-PC{n_pc}{note}', fontsize=12)
            draw(fig, kind)
            fig.savefig(paths[key], dpi=dpi)
        finally:
            plt.close(fig)
    fig = plt.figure(figsize=(width*2, height), layout='constrained')
    try:
        left, right = fig.subfigures(1, 2)
        left.suptitle(f'PCA loadings | {population_label}{note}', fontsize=12)
        right.suptitle(f'Temporal loading profiles | {population_label}{note}', fontsize=12)
        draw(left, 'heatmap'); draw(right, 'waveform')
        fig.savefig(paths['combined'], dpi=dpi)
    finally:
        plt.close(fig)
    np.savez_compressed(paths['data'], coefficients=coefficients, loadings=maps,
        mean_temporal_loadings=profiles, time_ms=time_ms, feature_shape=np.array((n_neurons, n_bins)),
        explained_variance=model.explained_variance[:n_pc],
        explained_variance_ratio=np.array([]) if ratio is None else ratio,
        total_variance=np.nan if model.total_variance is None else model.total_variance,
        title_note=title_note, loading_definition='coefficient * sqrt(explained_variance)')
    return paths
