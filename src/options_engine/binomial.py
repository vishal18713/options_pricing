"""Cox-Ross-Rubinstein (CRR) binomial tree pricing engine.

Supports:
- European and American call/put options
- Continuous dividend yields
- Fast memory-efficient backward induction (1D vector rollout)
- Validation of risk-neutral probability bounds (p in [0, 1])
- Model runtime and diagnostic metadata tracking
"""

import time

import numpy as np

from options_engine.types import ExerciseStyle, OptionType, PricingResult
from options_engine.validation import ModelDiscretizationError, validate_market_inputs


def crr_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    steps: int = 500,
    option_type: OptionType | str = OptionType.CALL,
    exercise_style: ExerciseStyle | str = ExerciseStyle.EUROPEAN,
    dividend_yield: float = 0.0,
) -> PricingResult:
    """Price vanilla option using Cox-Ross-Rubinstein (CRR) Binomial Tree model.

    Args:
        spot: Spot price S > 0
        strike: Strike price K > 0
        time_to_expiry: Time to expiration T >= 0 in years
        rate: Continuously compounded risk-free rate r
        volatility: Annualized volatility sigma >= 0
        steps: Number of time steps in the tree (N >= 1)
        option_type: "call" or "put"
        exercise_style: "european" or "american"
        dividend_yield: Continuously compounded dividend yield q >= 0

    Returns:
        PricingResult containing price, model name, status, and diagnostics.
    """
    start_time = time.perf_counter()

    s = float(spot)
    k = float(strike)
    t = float(time_to_expiry)
    r = float(rate)
    v = float(volatility)
    q = float(dividend_yield)
    opt_type = OptionType(option_type)
    ex_style = ExerciseStyle(exercise_style)

    validate_market_inputs(s, k, t, v, r, q)

    if steps < 1:
        raise ValueError(f"Number of steps must be at least 1, got {steps}")

    # Edge case: T == 0
    if t == 0.0:
        if opt_type == OptionType.CALL:
            intrinsic = max(s - k, 0.0)
        else:
            intrinsic = max(k - s, 0.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return PricingResult(
            price=float(intrinsic),
            model="CRR",
            inputs_id=f"S={s}_K={k}_T={t}",
            status="SUCCESS",
            diagnostics={"steps": steps, "at_expiry": True},
            runtime_ms=elapsed_ms,
        )

    # Edge case: vol == 0
    if v == 0.0:
        disc_s = s * np.exp(-q * t)
        disc_k = k * np.exp(-r * t)
        if opt_type == OptionType.CALL:
            p = max(disc_s - disc_k, 0.0)
            if ex_style == ExerciseStyle.AMERICAN:
                p = max(p, s - k, 0.0)
        else:
            p = max(disc_k - disc_s, 0.0)
            if ex_style == ExerciseStyle.AMERICAN:
                p = max(p, k - s, 0.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return PricingResult(
            price=float(p),
            model="CRR",
            inputs_id=f"S={s}_K={k}_T={t}",
            status="SUCCESS",
            diagnostics={"steps": steps, "zero_volatility": True},
            runtime_ms=elapsed_ms,
        )

    # Time step and CRR factors
    dt = t / steps
    u = np.exp(v * np.sqrt(dt))
    d = 1.0 / u

    # Risk-neutral growth and probability
    # growth factor a = exp((r - q) * dt)
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ModelDiscretizationError(
            f"MODEL_INVALID_PROBABILITY: CRR risk-neutral probability p={p:.6f} "
            f"is outside [0, 1]. Increase steps (current steps={steps}, dt={dt:.6f}) "
            f"or check rate/volatility inputs."
        )

    discount = np.exp(-r * dt)
    q_prob = 1.0 - p

    # Vectorized terminal underlying asset prices at step N:
    # S_T(j) = S * u^j * d^(N - j) for j = 0, ..., N
    j = np.arange(steps + 1)
    s_terminal = s * (u**j) * (d ** (steps - j))

    # Terminal payoffs
    if opt_type == OptionType.CALL:
        values = np.maximum(s_terminal - k, 0.0)
    else:
        values = np.maximum(k - s_terminal, 0.0)

    # Backward induction:
    # For American options, at step i, underlying asset values are:
    # S_i(j) = S * u^j * d^(i - j) for j = 0, ..., i
    if ex_style == ExerciseStyle.AMERICAN:
        for i in range(steps - 1, -1, -1):
            # Continuation values for j = 0, ..., i
            values = discount * (p * values[1 : i + 2] + q_prob * values[0 : i + 1])
            # Intrinsic value at step i
            j_i = np.arange(i + 1)
            s_current = s * (u**j_i) * (d ** (i - j_i))
            if opt_type == OptionType.CALL:
                intrinsic = np.maximum(s_current - k, 0.0)
            else:
                intrinsic = np.maximum(k - s_current, 0.0)
            values = np.maximum(values, intrinsic)
    else:
        # Pure European option: rollback continuation values without intermediate intrinsic comparisons
        for i in range(steps - 1, -1, -1):
            values = discount * (p * values[1 : i + 2] + q_prob * values[0 : i + 1])

    final_price = float(values[0])
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return PricingResult(
        price=final_price,
        model="CRR",
        inputs_id=f"S={s}_K={k}_T={t}_steps={steps}_{ex_style.value}",
        status="SUCCESS",
        diagnostics={
            "steps": steps,
            "dt": dt,
            "u": u,
            "d": d,
            "p": p,
            "exercise_style": ex_style.value,
            "option_type": opt_type.value,
        },
        runtime_ms=elapsed_ms,
    )


def build_crr_tree_visualization(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    steps: int = 5,
    option_type: OptionType | str = OptionType.CALL,
    exercise_style: ExerciseStyle | str = ExerciseStyle.AMERICAN,
    dividend_yield: float = 0.0,
) -> dict:
    """Build complete node tree data structure for interactive UI visualization.

    Returns dict containing nodes and edges:
    - nodes: list of dicts with step, index, stock_price, option_value, intrinsic_value, early_exercised
    - edges: list of dicts with source_step, source_idx, target_step, target_idx, branch ('up' or 'down')
    """
    s = float(spot)
    k = float(strike)
    t = float(time_to_expiry)
    r = float(rate)
    v = float(volatility)
    q = float(dividend_yield)
    opt_type = OptionType(option_type)
    ex_style = ExerciseStyle(exercise_style)

    validate_market_inputs(s, k, t, v, r, q)

    dt = t / steps
    u = np.exp(v * np.sqrt(dt))
    d = 1.0 / u
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    discount = np.exp(-r * dt)
    q_prob = 1.0 - p

    # Stock prices: stock_prices[i][j] = S * u^j * d^(i - j) for i=0..steps, j=0..i
    stock_prices = [[0.0] * (i + 1) for i in range(steps + 1)]
    for i in range(steps + 1):
        for j in range(i + 1):
            stock_prices[i][j] = s * (u**j) * (d ** (i - j))

    # Option values and decisions
    option_values = [[0.0] * (i + 1) for i in range(steps + 1)]
    early_exercised = [[False] * (i + 1) for i in range(steps + 1)]
    intrinsic_values = [[0.0] * (i + 1) for i in range(steps + 1)]

    # Terminal payoffs at step N
    for j in range(steps + 1):
        stk = stock_prices[steps][j]
        intr = max(stk - k, 0.0) if opt_type == OptionType.CALL else max(k - stk, 0.0)
        intrinsic_values[steps][j] = intr
        option_values[steps][j] = intr

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            stk = stock_prices[i][j]
            intr = max(stk - k, 0.0) if opt_type == OptionType.CALL else max(k - stk, 0.0)
            intrinsic_values[i][j] = intr
            continuation = discount * (
                p * option_values[i + 1][j + 1] + q_prob * option_values[i + 1][j]
            )

            if ex_style == ExerciseStyle.AMERICAN:
                if intr > continuation + 1e-9:
                    option_values[i][j] = intr
                    early_exercised[i][j] = True
                else:
                    option_values[i][j] = continuation
                    early_exercised[i][j] = False
            else:
                option_values[i][j] = continuation
                early_exercised[i][j] = False

    nodes = []
    for i in range(steps + 1):
        for j in range(i + 1):
            nodes.append(
                {
                    "step": i,
                    "node_idx": j,
                    "time": i * dt,
                    "stock_price": stock_prices[i][j],
                    "option_value": option_values[i][j],
                    "intrinsic_value": intrinsic_values[i][j],
                    "early_exercised": early_exercised[i][j],
                }
            )

    edges = []
    for i in range(steps):
        for j in range(i + 1):
            edges.append(
                {
                    "from_step": i,
                    "from_idx": j,
                    "to_step": i + 1,
                    "to_idx": j,
                    "branch": "down",
                }
            )
            edges.append(
                {
                    "from_step": i,
                    "from_idx": j,
                    "to_step": i + 1,
                    "to_idx": j + 1,
                    "branch": "up",
                }
            )

    return {
        "steps": steps,
        "dt": dt,
        "u": u,
        "d": d,
        "p": p,
        "nodes": nodes,
        "edges": edges,
        "root_price": option_values[0][0],
    }
