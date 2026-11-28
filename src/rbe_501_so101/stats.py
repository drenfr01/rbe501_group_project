"""Small statistics helpers (Student-t quantiles) without a SciPy dependency."""

from math import lgamma, pi, sqrt

import numpy as np

CDF_GRID_POINTS = 20_001
BISECTION_STEPS = 60
SEARCH_UPPER = 1e3


def student_t_pdf(x: np.ndarray, dof: float) -> np.ndarray:
    """Density of Student's t distribution."""
    log_norm = lgamma((dof + 1) / 2) - lgamma(dof / 2) - 0.5 * np.log(dof * pi)
    return np.exp(log_norm) * (1 + np.asarray(x) ** 2 / dof) ** (-(dof + 1) / 2)


def student_t_cdf(x: float, dof: float) -> float:
    """CDF by trapezoidal integration of the density from 0 (the distribution is symmetric)."""
    grid = np.linspace(0.0, abs(x), CDF_GRID_POINTS)
    half = float(np.trapezoid(student_t_pdf(grid, dof), grid))
    return 0.5 + half if x >= 0 else 0.5 - half


def student_t_ppf(p: float, dof: float) -> float:
    """Quantile for `0.5 <= p < 1` by bisection on the CDF."""
    if not 0.5 <= p < 1.0:
        raise ValueError("p must be in [0.5, 1)")
    low, high = 0.0, SEARCH_UPPER / sqrt(max(dof, 1.0))
    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2
        low, high = (mid, high) if student_t_cdf(mid, dof) < p else (low, mid)
    return (low + high) / 2
