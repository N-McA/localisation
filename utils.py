
import numpy as np
from pathlib import Path


def mapk(ground_truth, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(ground_truth, predicted)])


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    n_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            n_hits += 1.0
            score += n_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def load_paths(loc):
    loc = Path(loc)
    paths = []
    with loc.open() as f:
        for line in f:
            paths.append(Path(line.strip()))
    return paths
