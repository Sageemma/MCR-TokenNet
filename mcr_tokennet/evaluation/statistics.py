"""Group-aware paired macro-F1 inference across matching observation IDs."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

def paired_macro_f1(a, b, labels: list[str], unit_ids, *, seed: int=42, bootstraps: int=2000, permutations: int=10000) -> dict:
    """Inputs are matching arrays of [seed, sample] hard predictions and true labels.

    `a` and `b` contain (true_labels, predicted_labels) pairs; the same units
    are sampled/swapped across all seeds to retain paired dependence.
    """
    ya, pa = (np.asarray(x) for x in a)
    yb, pb = (np.asarray(x) for x in b)
    if ya.shape != pa.shape or yb.shape != pb.shape or ya.shape != yb.shape or (ya.ndim != 2):
        raise ValueError('Expected matched [seed, sample] target and prediction arrays')
    if not np.array_equal(ya, yb):
        raise ValueError('Ground-truth arrays differ')
    if len(unit_ids) != ya.shape[1]:
        raise ValueError('Unit IDs must match observations')
    keys = list(dict.fromkeys(unit_ids))
    groups = defaultdict(list)
    for i, key in enumerate(unit_ids):
        groups[key].append(i)
    ids = np.arange(len(keys))
    rng = np.random.default_rng(seed)
    key_index = {k: i for i, k in enumerate(keys)}
    group_index = np.asarray([key_index[k] for k in unit_ids], dtype=int)
    classes = np.arange(len(labels))

    def difference(index, mask=None):
        if mask is None:
            first, second = (pa[:, index], pb[:, index])
        else:
            swapped = mask[group_index[index]]
            first = np.where(swapped[None, :], pb[:, index], pa[:, index])
            second = np.where(swapped[None, :], pa[:, index], pb[:, index])
        y = ya[:, index]
        return float(np.mean([f1_score(y[s], first[s], labels=classes, average='macro', zero_division=0) - f1_score(y[s], second[s], labels=classes, average='macro', zero_division=0) for s in range(len(y))]))
    full = np.arange(ya.shape[1])
    observed = difference(full)
    boots = []
    for _ in range(bootstraps):
        picked = rng.choice(ids, size=len(ids), replace=True)
        subset = np.concatenate([groups[keys[i]] for i in picked])
        boots.append(difference(subset))
    extreme = 0
    for _ in range(permutations):
        flipped = rng.integers(0, 2, size=len(keys)).astype(bool)
        extreme += abs(difference(full, flipped)) >= abs(observed) - 1e-12
    return {'difference': observed, 'ci95': np.quantile(boots, [0.025, 0.975]).tolist(), 'permutation_p': (extreme + 1) / (permutations + 1), 'n_groups': len(keys), 'n_seeds': len(ya)}

def holm(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = previous
    return adjusted.tolist()
