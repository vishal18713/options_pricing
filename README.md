# Options Pricing & Greeks Engine

[![CI](https://github.com/options-engine/options-pricing-greeks-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/options-engine/options-pricing-greeks-engine/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade quantitative options pricing and risk sensitivity library implementing **Black-Scholes-Merton (BSM)**, **Cox-Ross-Rubinstein (CRR) Binomial Trees**, **Monte Carlo Simulation with Variance Reduction**, **Analytical & Numerical Greeks**, **Bracketed Brent Implied Volatility**, and **Gatheral's SVI Volatility Surface with Arbitrage Diagnostics**.

---

## 1. Mathematical Architecture & Models

### 1.1 Black-Scholes-Merton (BSM)
Closed-form analytical pricing with continuous dividend yield:
$$d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$
$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$
- Full analytical Greeks ($\Delta, \Gamma, \nu, \Theta, \rho$).
- Robust boundary branches for $T \to 0$, $\sigma \to 0$, and deep ITM/OTM extremes.

### 1.2 Cox-Ross-Rubinstein (CRR) Binomial Tree
- Time step $\Delta t = T / N$, up factor $u = e^{\sigma\sqrt{\Delta t}}$, down factor $d = 1 / u$.
- Risk-neutral growth $a = e^{(r - q)\Delta t}$ and probability $p = \frac{a - d}{u - d} \in [0, 1]$.
- Backward induction:
  - European: $V_{i, j} = e^{-r\Delta t}(p V_{i+1, j+1} + (1-p)V_{i+1, j})$.
  - American: $V_{i, j} = \max(\text{Continuation}, \text{Intrinsic})$.

### 1.3 Monte Carlo Pricing & Variance Reduction
- Terminal asset price under risk-neutral measure:
  $$S_T = S \exp\left((r - q - \frac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right)$$
- **Antithetic variates**: Pairs $(+Z, -Z)$ with variance calculated strictly on independent pair averages $Y_i = \frac{f(Z_i) + f(-Z_i)}{2}$.
- **Control variates**: Optimal regression beta $\beta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$ using matching analytical expectation $E[e^{-rT}S_T] = S e^{-qT}$.
- Reports unbiased standard error ($\text{SE} = s / \sqrt{n}$) and 95% confidence intervals.

### 1.4 Implied Volatility & Bounds
- Theoretical European no-arbitrage bounds verification:
  $$\max(0, S e^{-qT} - K e^{-rT}) \le C \le S e^{-qT}$$
  $$\max(0, K e^{-rT} - S e^{-qT}) \le P \le K e^{-rT}$$
- Bracketed Brent solver (`scipy.optimize.brentq`) over $[\sigma_{\min}, \sigma_{\max}]$ with explicit diagnostic failure codes (`IV_*`).

### 1.5 SVI Surface & Static Arbitrage Diagnostics
- Raw SVI parametric formulation:
  $$w(k) = a + b \left(\rho(k - m) + \sqrt{(k - m)^2 + \eta^2}\right)$$
  where $k = \ln(K / F)$, $w = \sigma_{\text{IV}}^2 T$.
- **Butterfly Arbitrage**: Checks non-negativity of Gatheral's density $g(k) \ge 0$.
- **Calendar Arbitrage**: Checks total variance monotonicity $\frac{\partial w}{\partial T} \ge 0$ for $T_2 > T_1$.

---

## 2. Installation

```bash
git clone https://github.com/vishal18713/options_pricing.git
cd options_pricing
python -m pip install -e ".[dev]"
```

---

## 3. Quickstart Examples

```python
from options_engine import bsm_price, bsm_greeks, crr_price, monte_carlo_price, implied_volatility

# 1. BSM European Price
call_price = bsm_price(spot=100, strike=100, time_to_expiry=1.0, rate=0.05, volatility=0.20)
print(f"BSM Call Price: ${call_price:.4f}")

# 2. Analytical Greeks
greeks = bsm_greeks(spot=100, strike=100, time_to_expiry=1.0, rate=0.05, volatility=0.20)
print(f"Delta: {greeks.delta:.4f}, Gamma: {greeks.gamma:.4f}, Vega: {greeks.vega:.4f}")

# 3. CRR American Put
crr_res = crr_price(spot=70, strike=100, time_to_expiry=1.0, rate=0.08, volatility=0.20,
                    steps=500, option_type="put", exercise_style="american")
print(f"CRR American Put: ${crr_res.price:.4f}")

# 4. Monte Carlo with Variance Reduction
mc_res = monte_carlo_price(spot=100, strike=100, time_to_expiry=1.0, rate=0.05, volatility=0.20,
                           n_paths=150_000, seed=42, antithetic=True, control_variate=True)
print(f"Monte Carlo Price: ${mc_res.price:.4f} ± {mc_res.standard_error:.4f}")

# 5. Implied Volatility Inversion
iv_res = implied_volatility(price=call_price, spot=100, strike=100, time_to_expiry=1.0, rate=0.05)
print(f"Recovered IV: {iv_res.implied_volatility * 100:.2f}% (Status: {iv_res.status})")
```

---

## 4. Verification & Testing

Run all unit, property, and QuantLib cross-validation tests:
```bash
python -m pytest -q
```

Run test suite with statement coverage report:
```bash
python -m pytest --cov=options_engine --cov-report=term-missing
```

Code linting and formatting:
```bash
ruff check .
ruff format --check .
```

Regenerate research figures:
```bash
python scripts/generate_figures.py
```

Launch optional interactive demo UI:
```bash
streamlit run app/streamlit_app.py
```

---

## 5. Model Limitations & Assumptions

1. **Market Liquidity & Friction**: BSM and CRR assume continuous trading, frictionless markets, and constant risk-free rates/volatilities.
2. **Exercise Styles**: Many US equity options are American-style; comparing vanilla BSM directly to American option market prices will miss the early-exercise premium. Use CRR for American options.
3. **SVI Convergence vs Arbitrage**: Optimizer convergence in SVI slice calibration does not automatically imply the absence of arbitrage; independent butterfly and calendar checks must always be evaluated.
4. **Intended Use**: This software is intended strictly for quantitative research, education, and portfolio demonstration. It is not an automated execution or investment advice system.
