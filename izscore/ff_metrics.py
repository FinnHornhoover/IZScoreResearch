import numpy as np


def mean_percentage_error(y_pred, y, w=1.0, safe=False, ord=1):
    weights = w if isinstance(w, np.ndarray) else np.ones_like(y_pred) * w
    denom = np.maximum(0.001, y) if safe else y
    diff = np.abs(np.divide(y - y_pred, denom)) ** ord
    return 100. * np.average(diff, weights=weights)


def epsilon_insensitive_max_score_loss(x, y_pred, y, w=1.0, ord=1, eps=0.5):
    weights = w if isinstance(w, np.ndarray) else np.ones_like(y_pred) * w
    max_score = x[:, 0]

    diff = np.abs(y_pred - y) ** ord
    diff[diff < eps] = 0.0
    diff[np.isclose(y, max_score) & (y_pred > max_score)] = 0.0

    return np.average(diff, weights=weights)
