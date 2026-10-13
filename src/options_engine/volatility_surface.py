"""Stochastic Volatility Inspired (SVI) volatility surface calibration and static arbitrage diagnostics.

Implements Gatheral's raw SVI formulation:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + eta^2))
where:
    k = ln(K / F) is log-moneyness
    w(k) = sigma_IV^2 * T is total implied variance
    eta > 0, b >= 0, |rho| < 1, a + b * eta * sqrt(1 - rho^2) >= 0

Diagnostics:
- Calendar arbitrage check: total variance w(k, T2) >= w(k, T1) for T2 > T1
- Butterfly arbitrage check: non-negativity of the risk-neutral density g(k) >= 0
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass
class SVIParameters:
    """Raw SVI model parameters for a single expiry slice."""

    a: float
    b: float
    rho: float
    m: float
    eta: float

    def to_dict(self) -> dict[str, float]:
        return {
            "a": float(self.a),
            "b": float(self.b),
            "rho": float(self.rho),
            "m": float(self.m),
            "eta": float(self.eta),
        }


def raw_svi_total_variance(
    k: np.ndarray | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    eta: float,
) -> np.ndarray | float:
    """Evaluate raw SVI total variance w(k)."""
    k_arr = np.asarray(k, dtype=np.float64)
    disc = np.sqrt((k_arr - m) ** 2 + eta**2)
    w = a + b * (rho * (k_arr - m) + disc)
    if np.isscalar(k):
        return float(np.asarray(w).item())
    return w


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 2000,
) -> tuple[SVIParameters, float, bool, dict[str, Any]]:
    """Fit raw SVI parameters to a single expiry slice of log-moneyness and total variance.

    Objective: Weighted Mean Squared Error (WMSE).
    """
    k = np.asarray(log_moneyness, dtype=np.float64)
    w_obs = np.asarray(total_variance, dtype=np.float64)

    if len(k) < 5:
        raise ValueError(f"At least 5 quotes required to fit 5 SVI parameters, got {len(k)}")

    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=np.float64)

    # Candidate initial parameter guesses to avoid local minima
    min_w = np.min(w_obs)
    atm_idx = np.argmin(np.abs(k))
    atm_w = w_obs[atm_idx]

    candidate_inits = [
        np.array([max(min_w * 0.8, 1e-4), 0.1, -0.2, 0.0, 0.1]),
        np.array([atm_w * 0.5, 0.15, -0.3, float(k[atm_idx]), 0.1]),
        np.array([max(min_w * 0.5, 1e-4), 0.2, -0.4, 0.05, 0.15]),
        np.array([max(min_w * 0.9, 1e-4), 0.05, 0.0, 0.0, 0.05]),
    ]

    # Parameter bounds: a in [-5, 5], b in [0, 5], rho in [-0.999, 0.999], m in [-2, 2], eta in [1e-4, 2]
    bounds = [
        (-5.0, 5.0),
        (0.0, 5.0),
        (-0.999, 0.999),
        (-2.0, 2.0),
        (1e-4, 2.0),
    ]

    def min_var_constraint(params):
        a_val, b_val, rho_val, _, eta_val = params
        return a_val + b_val * eta_val * np.sqrt(max(1.0 - rho_val**2, 0.0))

    constraints = [{"type": "ineq", "fun": min_var_constraint}]

    def objective(params):
        a_val, b_val, rho_val, m_val, eta_val = params
        w_pred = raw_svi_total_variance(k, a_val, b_val, rho_val, m_val, eta_val)
        return float(np.average((w_pred - w_obs) ** 2, weights=wts))

    best_res = None
    for init_p in candidate_inits:
        res = minimize(
            objective,
            init_p,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-11},
        )
        if best_res is None or (res.fun < best_res.fun and res.success):
            best_res = res

    if best_res is None:
        best_res = res

    a_fit, b_fit, rho_fit, m_fit, eta_fit = best_res.x
    params = SVIParameters(
        a=float(a_fit),
        b=float(b_fit),
        rho=float(rho_fit),
        m=float(m_fit),
        eta=float(eta_fit),
    )

    diagnostics = {
        "iterations": best_res.nit,
        "message": best_res.message,
        "success": bool(best_res.success),
        "rmse": float(np.sqrt(best_res.fun)),
    }

    return params, float(best_res.fun), bool(best_res.success), diagnostics


def check_butterfly_arbitrage_slice(
    params: SVIParameters,
    k_grid: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Check butterfly arbitrage on an SVI slice via non-negativity of Gatheral's density g(k).

    Density formula for raw SVI total variance w(k):
        g(k) = (1 - (k * w'(k)) / (2 * w(k)))^2 - (w'(k)^2 / 4) * (1 / w(k) + 1 / 4) + w''(k) / 2
    Butterfly arbitrage is absent if and only if g(k) >= 0 for all k.
    """
    k = np.asarray(k_grid, dtype=np.float64)
    a, b, rho, m, eta = params.a, params.b, params.rho, params.m, params.eta

    w = raw_svi_total_variance(k, a, b, rho, m, eta)
    if np.any(w <= 0):
        return False, {"arbitrage_free": False, "reason": "TOTAL_VARIANCE_NON_POSITIVE"}

    disc = np.sqrt((k - m) ** 2 + eta**2)
    # First derivative w'(k) = b * (rho + (k - m) / disc)
    w_prime = b * (rho + (k - m) / disc)
    # Second derivative w''(k) = b * eta^2 / (disc^3)
    w_prime2 = b * (eta**2) / (disc**3)

    # Gatheral's g(k)
    term1 = (1.0 - (k * w_prime) / (2.0 * w)) ** 2
    term2 = (w_prime**2 / 4.0) * (1.0 / w + 0.25)
    term3 = w_prime2 / 2.0
    g = term1 - term2 + term3

    min_g = float(np.min(g))
    is_arbitrage_free = bool(min_g >= -1e-6)

    return is_arbitrage_free, {
        "arbitrage_free": is_arbitrage_free,
        "min_density": min_g,
        "min_density_k": float(k[np.argmin(g)]),
        "violation_count": int(np.sum(g < -1e-6)),
    }


def check_calendar_arbitrage(
    slices: dict[float, SVIParameters],
    k_grid: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Check calendar spread arbitrage across sorted maturities.

    Condition: For T1 < T2, w(k, T2) >= w(k, T1) for all log-moneyness k.
    """
    maturities = sorted(slices.keys())
    if len(maturities) < 2:
        return True, {"calendar_pairs_checked": 0, "arbitrage_free": True}

    violations = []
    pairs_checked = 0

    for idx in range(len(maturities) - 1):
        t1, t2 = maturities[idx], maturities[idx + 1]
        p1 = slices[t1]
        p2 = slices[t2]

        w1 = raw_svi_total_variance(k_grid, p1.a, p1.b, p1.rho, p1.m, p1.eta)
        w2 = raw_svi_total_variance(k_grid, p2.a, p2.b, p2.rho, p2.m, p2.eta)

        diff = w2 - w1
        pairs_checked += 1
        if np.any(diff < -1e-6):
            min_diff = float(np.min(diff))
            violations.append(
                {
                    "t1": t1,
                    "t2": t2,
                    "min_diff": min_diff,
                    "k_at_min": float(k_grid[np.argmin(diff)]),
                }
            )

    is_free = len(violations) == 0
    return is_free, {
        "arbitrage_free": is_free,
        "calendar_pairs_checked": pairs_checked,
        "violations": violations,
    }
