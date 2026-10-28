"""Class-specific regularized Mahalanobis classification of arbitrary features."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
try:
    from scipy.linalg import svd
except ModuleNotFoundError as error:
    if error.name not in ('scipy', 'scipy.linalg'):
        raise

    def svd(a, *, full_matrices=True, check_finite=True):
        """SciPyがない場合も同じ分類式をNumPyで使用できるようにする。"""
        return np.linalg.svd(a, full_matrices=full_matrices)


@dataclass(frozen=True)
class MahalanobisModel:
    classes: np.ndarray
    centers: tuple[np.ndarray, ...]
    bases: tuple[np.ndarray, ...]
    variances: tuple[np.ndarray, ...]
    regularization: float


def fit_classifier(training_features, training_labels, *, regularization=1e-6) -> MahalanobisModel:
    """Fit a mean/covariance per class, using only labeled training trials."""
    train = np.asarray(training_features, dtype=np.float64)
    labels = np.asarray(training_labels)
    if train.ndim != 2 or train.shape[1] == 0 or labels.ndim != 1 or len(labels) != len(train):
        raise ValueError('Expected (training_trials, features) and one label per trial')
    if not np.isfinite(train).all() or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('Features must be finite and regularization must be positive')
    classes = np.unique(labels)
    if not len(classes):
        raise ValueError('At least one training class is required')
    centers, bases, variances = [], [], []
    for label in classes:
        group = train[labels == label]
        if len(group) < 2:
            raise ValueError('Each class requires at least two training trials')
        center = group.mean(0)
        _, s, v = svd(group-center, full_matrices=False, check_finite=False)
        centers.append(center); bases.append(v); variances.append(s*s/(len(group)-1))
    return MahalanobisModel(classes, tuple(centers), tuple(bases), tuple(variances), float(regularization))


def predict_classifier(features, model: MahalanobisModel, *, return_distances=False):
    """Return predicted labels, optionally with squared distances to every class.

    Distance columns follow model.classes. No normalization or PCA is done here.
    """
    test = np.asarray(features, dtype=np.float64)
    if test.ndim != 2 or test.shape[1] != model.centers[0].size or not np.isfinite(test).all():
        raise ValueError('Evaluation features must be finite and match training feature count')
    distances = []
    for center, v, variance in zip(model.centers, model.bases, model.variances):
        delta = test-center; projection = delta @ v.T; remaining = delta-projection @ v
        distances.append((projection**2/(variance+model.regularization)).sum(1)
                         + (remaining*remaining).sum(1)/model.regularization)
    distances = np.stack(distances, 1)
    predicted = model.classes[np.argmin(distances, axis=1)]
    return (predicted, distances) if return_distances else predicted


def classify(training_features, training_labels, evaluation_features, *, regularization=1e-6, return_distances=False):
    """Fit from labeled training features and predict evaluation labels."""
    model = fit_classifier(training_features, training_labels, regularization=regularization)
    return predict_classifier(evaluation_features, model, return_distances=return_distances)
