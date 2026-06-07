"""
Reusable statistical helpers: bootstrap CI, Cliff's delta, Holm-Bonferroni correction.
All random operations accept a seed for reproducibility.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_iter: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI for stat_fn applied to data."""
    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_iter)
    ])
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_stats, alpha)), float(np.quantile(boot_stats, 1 - alpha))


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    stat_fn=np.mean,
    n_iter: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for stat_fn(a) - stat_fn(b)."""
    rng = np.random.default_rng(seed)
    diffs = np.array([
        stat_fn(rng.choice(a, size=len(a), replace=True))
        - stat_fn(rng.choice(b, size=len(b), replace=True))
        for _ in range(n_iter)
    ])
    alpha = (1 - ci) / 2
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))


# ── Effect sizes ─────────────────────────────────────────────────────────────

def cliffs_delta(a: np.ndarray, b: np.ndarray, chunk_size: int = 5000) -> float:
    """
    Cliff's delta: P(a > b) - P(b > a).
    Range [-1, 1]; |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    Uses chunked computation to avoid allocating an n1×n2 matrix for large arrays.
    """
    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    dominance = 0
    for i in range(0, m, chunk_size):
        chunk = a[i : i + chunk_size]
        dominance += np.sum(np.sign(chunk[:, None] - b[None, :]))
    return float(dominance / (m * n))


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    """
    Derive Cliff's delta directly from Mann-Whitney U.
    Mathematically equivalent to the pairwise formula: d = 2U/(n1*n2) - 1.
    Use this when n1*n2 is too large for the pairwise approach.
    """
    return float(2 * u_stat / (n1 * n2) - 1)


def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Convert Mann-Whitney U to rank-biserial correlation r = 2U/(n1*n2) - 1."""
    return float(2 * u_stat / (n1 * n2) - 1)


# ── Holm-Bonferroni correction ───────────────────────────────────────────────

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """
    Return Holm-Bonferroni adjusted p-values (same order as input).
    Equivalent to scipy.stats.false_discovery_control with 'holm' method, but explicit.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


# ── Spearman CI (Fisher z on rank correlation) ───────────────────────────────

def spearman_ci(rho: float, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Approximate CI for Spearman rho using Fisher z-transform."""
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    z_crit = sp_stats.norm.ppf(1 - (1 - ci) / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def cohens_kappa_weighted(y_true, y_pred, weights: str = "quadratic") -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred, weights=weights))
