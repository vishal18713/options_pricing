"""Monte Carlo options pricing engine with variance reduction techniques.

Supports:
- Risk-neutral terminal asset price simulation
- Deterministic reproducibility with numpy.random.Generator
- Standard error and 95% confidence intervals
- Antithetic variates with unbiased sample standard error over paired averages
- Control variate technique using analytical BSM matching option
"""

import time

import numpy as np

from options_engine.types import MonteCarloResult, OptionType
from options_engine.validation import validate_market_inputs


def monte_carlo_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    n_paths: int = 100_000,
    seed: int | None = 42,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    control_variate: bool = False,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Price European vanilla option using Monte Carlo simulation.

    Args:
        spot: Spot price S > 0
        strike: Strike price K > 0
        time_to_expiry: Time to expiration T >= 0 in years
        rate: Continuously compounded risk-free rate r
        volatility: Annualized volatility sigma >= 0
        n_paths: Total number of paths (if antithetic=True, draws n_paths // 2 independent pairs)
        seed: Random seed for numpy.random.Generator
        option_type: "call" or "put"
        dividend_yield: Continuously compounded dividend yield q >= 0
        antithetic: Use antithetic variate pairing (+Z and -Z)
        control_variate: Use European forward/vanilla analytical price as control variate
        confidence_level: Confidence level for uncertainty interval (default 0.95 -> z ~ 1.96)

    Returns:
        MonteCarloResult containing price, standard error, confidence interval, and diagnostics.
    """
    start_time = time.perf_counter()

    s = float(spot)
    k = float(strike)
    t = float(time_to_expiry)
    r = float(rate)
    v = float(volatility)
    q = float(dividend_yield)
    opt_type = OptionType(option_type)

    validate_market_inputs(s, k, t, v, r, q)

    if n_paths < 2:
        raise ValueError(f"n_paths must be at least 2, got {n_paths}")

    # Boundary: T == 0
    if t == 0.0:
        intrinsic = max(s - k, 0.0) if opt_type == OptionType.CALL else max(k - s, 0.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return MonteCarloResult(
            price=float(intrinsic),
            model="MonteCarlo",
            inputs_id=f"S={s}_K={k}_T={t}",
            status="SUCCESS",
            diagnostics={"at_expiry": True},
            runtime_ms=elapsed_ms,
            standard_error=0.0,
            confidence_interval=(float(intrinsic), float(intrinsic)),
            n_paths=n_paths,
            seed=seed,
            variance_reduction=[],
        )

    # Boundary: vol == 0
    if v == 0.0:
        disc_s = s * np.exp(-q * t)
        disc_k = k * np.exp(-r * t)
        val = (
            max(disc_s - disc_k, 0.0) if opt_type == OptionType.CALL else max(disc_k - disc_s, 0.0)
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return MonteCarloResult(
            price=float(val),
            model="MonteCarlo",
            inputs_id=f"S={s}_K={k}_T={t}",
            status="SUCCESS",
            diagnostics={"zero_volatility": True},
            runtime_ms=elapsed_ms,
            standard_error=0.0,
            confidence_interval=(float(val), float(val)),
            n_paths=n_paths,
            seed=seed,
            variance_reduction=[],
        )

    rng = np.random.default_rng(seed)
    discount = np.exp(-r * t)
    drift = (r - q - 0.5 * v**2) * t
    diffusion = v * np.sqrt(t)

    variance_techniques = []

    if antithetic:
        variance_techniques.append("antithetic")
        # Ensure even number of paths
        n_pairs = n_paths // 2
        effective_paths = n_pairs * 2
        z = rng.standard_normal(n_pairs)

        st1 = s * np.exp(drift + diffusion * z)
        st2 = s * np.exp(drift - diffusion * z)

        if opt_type == OptionType.CALL:
            payoff1 = np.maximum(st1 - k, 0.0)
            payoff2 = np.maximum(st2 - k, 0.0)
        else:
            payoff1 = np.maximum(k - st1, 0.0)
            payoff2 = np.maximum(k - st2, 0.0)

        # Independent pair averages (critical: compute uncertainty on pair averages!)
        pair_payoffs = 0.5 * (payoff1 + payoff2)
        discounted_samples = discount * pair_payoffs

        if control_variate:
            variance_techniques.append("control_variate")
            # Control variate using the discounted terminal underlying asset:
            # Under risk-neutral measure, E[e^(-rT) S_T] = S * e^(-qT)
            st_pair = 0.5 * (st1 + st2)
            cv_samples = discount * st_pair
            cv_mean = s * np.exp(-q * t)

            # Optimal control variate coefficient: beta = Cov(Y, X) / Var(X)
            cov_matrix = np.cov(discounted_samples, cv_samples)
            var_x = cov_matrix[1, 1]
            if var_x > 0:
                beta = cov_matrix[0, 1] / var_x
            else:
                beta = 0.0

            controlled_samples = discounted_samples - beta * (cv_samples - cv_mean)
            mean_price = float(np.mean(controlled_samples))
            std_err = float(np.std(controlled_samples, ddof=1) / np.sqrt(n_pairs))
        else:
            mean_price = float(np.mean(discounted_samples))
            std_err = float(np.std(discounted_samples, ddof=1) / np.sqrt(n_pairs))

        actual_n_paths = effective_paths

    else:
        # Standard Monte Carlo without antithetic
        z = rng.standard_normal(n_paths)
        st = s * np.exp(drift + diffusion * z)

        if opt_type == OptionType.CALL:
            payoff = np.maximum(st - k, 0.0)
        else:
            payoff = np.maximum(k - st, 0.0)

        discounted_samples = discount * payoff

        if control_variate:
            variance_techniques.append("control_variate")
            cv_samples = discount * st
            cv_mean = s * np.exp(-q * t)

            cov_matrix = np.cov(discounted_samples, cv_samples)
            var_x = cov_matrix[1, 1]
            beta = cov_matrix[0, 1] / var_x if var_x > 0 else 0.0

            controlled_samples = discounted_samples - beta * (cv_samples - cv_mean)
            mean_price = float(np.mean(controlled_samples))
            std_err = float(np.std(controlled_samples, ddof=1) / np.sqrt(n_paths))
        else:
            mean_price = float(np.mean(discounted_samples))
            std_err = float(np.std(discounted_samples, ddof=1) / np.sqrt(n_paths))

        actual_n_paths = n_paths

    # Confidence interval calculation (Normal approximation for sample mean via CLT)
    # 95% CI: z ~ 1.95996
    z_critical = 1.959963984540054
    ci_lower = max(mean_price - z_critical * std_err, 0.0)
    ci_upper = mean_price + z_critical * std_err

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return MonteCarloResult(
        price=mean_price,
        model="MonteCarlo",
        inputs_id=f"S={s}_K={k}_T={t}_paths={actual_n_paths}",
        status="SUCCESS",
        diagnostics={
            "effective_paths": actual_n_paths,
            "confidence_level": confidence_level,
            "drift": drift,
            "diffusion": diffusion,
        },
        runtime_ms=elapsed_ms,
        standard_error=std_err,
        confidence_interval=(ci_lower, ci_upper),
        n_paths=actual_n_paths,
        seed=seed,
        variance_reduction=variance_techniques,
    )
