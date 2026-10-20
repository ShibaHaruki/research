"""Accuracy and confusion-matrix summaries, independent of the classifier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_percentages(counts) -> np.ndarray:
    """Normalize each true-class row to percentages; empty rows remain zero."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.size == 0:
        raise ValueError('Expected a nonempty square confusion matrix')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Confusion counts must be finite and nonnegative')
    totals = counts.sum(1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals != 0)*100


def evaluate_classification(true_labels, predicted_labels, *, labels=None) -> dict:
    """Return overall/per-class accuracy (0..1), counts and row percentages."""
    true = np.asarray(true_labels); predicted = np.asarray(predicted_labels)
    if true.ndim != 1 or predicted.shape != true.shape or not len(true):
        raise ValueError('Expected equal-length, nonempty 1D true and predicted labels')
    classes = np.unique(np.concatenate([true, predicted])) if labels is None else np.asarray(labels)
    if classes.ndim != 1 or not len(classes) or len(np.unique(classes)) != len(classes):
        raise ValueError('labels must be a nonempty sequence of unique classes')
    if not np.isin(true, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError('labels must include every true and predicted class')
    cm = confusion_matrix(true, predicted, labels=classes)
    percentages = confusion_percentages(cm)
    return {'accuracy': float(np.mean(predicted == true)), 'labels': classes,
            'confusion_matrix': cm, 'confusion_percent': percentages,
            'per_class_accuracy': np.diag(percentages)/100,
            'class_support': cm.sum(1)}
