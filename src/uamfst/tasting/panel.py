from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import norm, rankdata


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    size = len(values)
    ranks = np.zeros(size, dtype=np.float64)
    index = 0
    while index < size:
        run = index
        while run < size and sorted_values[run] == sorted_values[index]:
            run += 1
        ranks[index:run] = 0.5 * (index + run - 1) + 1
        index = run
    result = np.empty(size, dtype=np.float64)
    result[order] = ranks
    return result


def delong_test(
    scores_a: torch.Tensor, scores_b: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float, float]:
    target = labels.detach().cpu().numpy().astype(bool)
    order = np.argsort(~target)
    positives = int(target.sum())
    negatives = len(target) - positives
    stacked = np.stack(
        [
            scores_a.detach().cpu().numpy()[order].astype(np.float64),
            scores_b.detach().cpu().numpy()[order].astype(np.float64),
        ]
    )
    pos = stacked[:, :positives]
    neg = stacked[:, positives:]
    tx = np.stack([_midrank(pos[r]) for r in range(2)])
    ty = np.stack([_midrank(neg[r]) for r in range(2)])
    tz = np.stack([_midrank(stacked[r]) for r in range(2)])
    rank_mass = tz[:, :positives].sum(axis=1) / positives / negatives
    aucs = rank_mass - (positives + 1.0) / 2.0 / negatives
    v01 = (tz[:, :positives] - tx) / negatives
    v10 = 1.0 - (tz[:, positives:] - ty) / positives
    cov = np.cov(v01) / positives + np.cov(v10) / negatives
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    spread = np.sqrt(variance) if variance > 0 else 1e-12
    z = float((aucs[0] - aucs[1]) / spread)
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), z, p


def bootstrap_ci(
    metric: Callable[[torch.Tensor, torch.Tensor], float],
    scores: torch.Tensor,
    labels: torch.Tensor,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    size = scores.shape[0]
    estimates = []
    for _ in range(resamples):
        index = torch.randint(0, size, (size,), generator=generator)
        estimates.append(metric(scores[index], labels[index]))
    ordered = np.sort(np.array(estimates, dtype=np.float64))
    low = float(np.quantile(ordered, alpha / 2.0))
    high = float(np.quantile(ordered, 1.0 - alpha / 2.0))
    return low, high


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda i: pvalues[i])
    reject = [False] * count
    for rank, position in enumerate(order):
        threshold = alpha / (count - rank)
        if pvalues[position] <= threshold:
            reject[position] = True
        else:
            break
    return reject


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    ranks_a = rankdata(first.detach().cpu().numpy())
    ranks_b = rankdata(second.detach().cpu().numpy())
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])
