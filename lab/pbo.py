"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
(Bailey, Borwein, Lopez de Prado & Zhu, 2015).

Split the returns of every configuration into S blocks. For every way of choosing S/2 blocks
as in-sample, pick the best configuration in-sample and look at its rank out-of-sample.
PBO = share of splits where the in-sample winner lands in the bottom half out-of-sample.
PBO near 0: selection generalises. PBO >= 0.5: the "best" config is as likely as not a fluke.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def pbo(returns: np.ndarray, n_blocks: int = 12) -> dict:
    """returns: T x N matrix (N configurations)."""
    R = np.nan_to_num(np.asarray(returns, dtype=float))
    T, N = R.shape
    if N < 2 or T < n_blocks * 5:
        return {"pbo": float("nan"), "n_splits": 0, "note": "not enough configs or data"}
    blocks = np.array_split(np.arange(T), n_blocks)
    s1 = np.array([R[b].sum(0) for b in blocks])          # per-block sums
    s2 = np.array([(R[b] ** 2).sum(0) for b in blocks])
    n = np.array([len(b) for b in blocks])[:, None]
    logits = []
    for ins in combinations(range(n_blocks), n_blocks // 2):
        m = np.zeros(n_blocks, bool)
        m[list(ins)] = True

        def sharpe(mask):
            k = n[mask].sum()
            mu = s1[mask].sum(0) / k
            var = s2[mask].sum(0) / k - mu**2
            return mu / np.sqrt(np.maximum(var, 1e-18))

        best = int(np.argmax(sharpe(m)))
        oos = sharpe(~m)
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1)   # 0..N-1
        w = (rank + 1) / (N + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": int(len(logits)), "median_logit": float(np.median(logits))}
