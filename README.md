<div align="center">

# ⚡ Options Pricing & Greeks Engine

### *Industrial-Grade Derivatives Valuation, Analytical & Pathwise Sensitivities, Robust Volatility Inversion, and Parametric SVI Surface Calibration*

[![CI](https://github.com/vishal18713/options_pricing/actions/workflows/tests.yml/badge.svg)](https://github.com/vishal18713/options_pricing/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![QuantLib](https://img.shields.io/badge/QuantLib-1.43%20Validated-brightgreen.svg)](https://www.quantlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-89%25-success.svg)](#-verification--testing)

<br/>

[Key Highlights](#-key-highlights) •
[Interactive Demo UI](#-interactive-web-app) •
[Architecture](#-architectural-overview) •
[Mathematical Models](#-mathematical-formulation) •
[Quickstart](#-quickstart-guide) •
[Figures](#-authoritative-figures)

<br/>

</div>

---

## 🌟 Key Highlights

<table>
  <tr>
    <td width="33%">
      <h3>📐 3 Pricing Engines</h3>
      <ul>
        <li><b>Black-Scholes-Merton (BSM)</b> with continuous dividend yields & extreme boundary branch handling.</li>
        <li><b>Cox-Ross-Rubinstein (CRR)</b> tree with early-exercise detection for American vanilla calls/puts.</li>
        <li><b>Monte Carlo Engine</b> with antithetic variates & terminal stock control variate.</li>
      </ul>
    </td>
    <td width="33%">
      <h3>🎯 3 Greeks Estimators</h3>
      <ul>
        <li><b>Analytical Greeks</b>: Closed-form Delta, Gamma, Vega, Theta (calendar passage), and Rho.</li>
        <li><b>Central Finite Differences</b>: Multidimensional bump-and-reprice with scale-aware steps.</li>
        <li><b>Pathwise Sensitivity</b>: Infinitesimal Perturbation Analysis (IPA) with standard errors.</li>
      </ul>
    </td>
    <td width="33%">
      <h3>🌊 Vol Surface & Arbitrage</h3>
      <ul>
        <li><b>Bracketed Brent IV Solver</b> with theoretical no-arbitrage bounds verification.</li>
        <li><b>SVI Surface Fitting</b>: Multi-start SLSQP calibration of Gatheral's raw SVI slices.</li>
        <li><b>Static Arbitrage Diagnostics</b>: Calendar monotonicity ($\partial w / \partial T \ge 0$) and butterfly non-negative density ($g(k) \ge 0$).</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🖥️ Interactive Web App

A clean, modern **Streamlit** dashboard with interactive **Plotly** visualizations is included out of the box. The presentation layer delegates directly to the core package with **zero duplicated logic**:

```powershell
python -m streamlit run app/streamlit_app.py
```

### 🖼️ Interactive Visualizations Preview

<div align="center">

#### 🌲 Interactive CRR Binomial Tree (with American Early Exercise)
<img src="reports/figures/binomial_tree.png" alt="Interactive Binomial Tree" width="90%"/>
<br/>
<sub><i>Up/down lattice nodes showing stock prices & option values. Early-exercise decisions are distinctly flagged as red diamonds vs blue continuation states.</i></sub>

<br/><br/>

#### 🌐 Rotatable 3D Implied Volatility Surface
<img src="reports/figures/volatility_surface.png" alt="3D Implied Volatility Surface" width="90%"/>
<br/>
<sub><i>Strike / log-moneyness vs maturity term structure plotted with implied volatility on the Z-axis, including live arbitrage health diagnostics.</i></sub>

<br/><br/>

#### 🎲 Monte Carlo Simulation Explorer
<img src="reports/figures/monte_carlo_simulation.png" alt="Monte Carlo Simulation Explorer" width="95%"/>
<br/>
<sub><i>Multi-step Geometric Brownian Motion trajectories, terminal price distribution with strike cut-off, and running empirical price convergence vs BSM benchmark.</i></sub>

</div>

---

## 🏛️ Architectural Overview

```
options-pricing-greeks-engine/
│
├── src/options_engine/           # Core Quantitative Engine (Zero external UI dependencies)
│   ├── types.py                  # Immutable contracts: OptionContract, MarketInputs, GreeksResult, IVResult
│   ├── conventions.py            # ACT/365 year fractions & standard derivative scaling
│   ├── validation.py             # Numerical domain assertions & typed exception hierarchy
│   ├── black_scholes.py          # Vectorized BSM analytical pricing & closed-form Greeks
│   ├── binomial.py               # Memory-efficient 1D rolling CRR binomial tree (European & American)
│   ├── monte_carlo.py            # Seeded risk-neutral simulator, antithetic & control variates
│   ├── greeks.py                 # Finite difference harness, pathwise IPA, & delta hedging simulation
│   ├── market_data.py            # Provider adapter with immutable snapshot preservation & provenance
│   ├── quote_cleaning.py         # Configurable filters: crossed markets, bid/ask spreads, audit codes
│   ├── forwards.py               # Cost-of-carry & put-call parity synthetic forward estimation
│   ├── volatility_surface.py     # Multi-start SVI slice fitting & butterfly/calendar arbitrage diagnostics
│   └── validation_suite.py       # Independent cross-check suite aligned against QuantLib 1.43
│
├── tests/                        # Comprehensive Test Suites (48 / 48 Passing)
│   ├── unit/                     # BSM, CRR, MC, Greeks, Cleaning, IV, SVI, and QuantLib tests
│   └── fixtures/                 # Offline option chain fixture data (zero network reliance in CI)
│
├── scripts/
│   ├── generate_figures.py       # Deterministic generator for publication-quality figures
│   └── benchmark.py              # Performance profiling across input scales
│
├── app/
│   └── streamlit_app.py          # Interactive UI demo
│
└── configs/
    └── default.yaml              # Centralized hyperparameters & numerical conventions
```

---

## 📐 Mathematical Formulation

### 1. Black-Scholes-Merton (BSM)
For spot $S$, strike $K$, time $T$, continuously compounded rate $r$, volatility $\sigma$, and dividend yield $q$:
$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$
$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

### 2. Cox-Ross-Rubinstein (CRR) Discretization
For time step $\Delta t = T / N$:
$$u = \exp(\sigma \sqrt{\Delta t}), \quad d = \frac{1}{u}, \quad p = \frac{e^{(r - q)\Delta t} - d}{u - d}$$
- **European Payoff**: Backward induction rolling expectation $V_{i, j} = e^{-r\Delta t}\left(p V_{i+1, j+1} + (1-p)V_{i+1, j}\right)$.
- **American Payoff**: Dynamic early-exercise intrinsic check at every node:
  $$V_{i, j} = \max\left(\text{Continuation Value}_{i, j}, \; \text{Intrinsic Value}_{i, j}\right)$$

### 3. Monte Carlo Variance Reduction
Terminal price under the risk-neutral measure:
$$S_T = S \exp\left(\left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right)$$
- **Antithetic Pairs**: Draws $Z \sim \mathcal{N}(0, 1)$ paired with $-Z$. Uncertainty is computed on independent pair averages:
  $$\text{SE} = \frac{\text{std}\left(\frac{f(Z_i) + f(-Z_i)}{2}\right)}{\sqrt{n_{\text{pairs}}}}$$
- **Control Variates**: Employs discounted terminal underlying asset $X = e^{-rT} S_T$ with known analytical mean $E[X] = S e^{-qT}$ to eliminate residual variance.

### 4. Gatheral's SVI Formulation & Arbitrage Diagnostics
Total variance slice $w(k) = \sigma_{\text{IV}}^2 T$ as a function of log-moneyness $k = \ln(K/F)$:
$$w(k) = a + b \left(\rho(k - m) + \sqrt{(k - m)^2 + \eta^2}\right)$$
Subject to parameter bounds $b \ge 0$, $\eta > 0$, $|\rho| < 1$, and $a + b\eta\sqrt{1-\rho^2} \ge 0$.
- **Butterfly Arbitrage**: Verified via non-negativity of the risk-neutral density $g(k) \ge 0$:
  $$g(k) = \left(1 - \frac{k w'(k)}{2 w(k)}\right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2} \ge 0$$
- **Calendar Monotonicity**: Verified that $\partial w / \partial T \ge 0$ for sorted expiries $T_1 < T_2$.

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/vishal18713/options_pricing.git
cd options_pricing
python -m pip install -e ".[dev]"
```

### 2. Python API Usage

```python
from options_engine import (
    bsm_price,
    bsm_greeks,
    crr_price,
    monte_carlo_price,
    implied_volatility,
    fit_svi_slice,
)

# 1. European Analytical Call
c_bsm = bsm_price(spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.20)
print(f"BSM Call: ${c_bsm:.4f}")

# 2. Complete Closed-Form Greeks
g = bsm_greeks(spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.20)
print(f"Delta: {g.delta:.4f} | Gamma: {g.gamma:.4f} | Vega: {g.vega:.4f} | Theta: {g.theta:.4f}")

# 3. American Put with Early-Exercise Premium
p_amer = crr_price(spot=70.0, strike=100.0, time_to_expiry=1.0, rate=0.08, volatility=0.20,
                   steps=500, option_type="put", exercise_style="american")
print(f"American Put (CRR 500 steps): ${p_amer.price:.4f}")

# 4. Monte Carlo with Antithetic & Control Variates
mc = monte_carlo_price(spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.20,
                       n_paths=100_000, seed=42, antithetic=True, control_variate=True)
print(f"Monte Carlo: ${mc.price:.4f} ± ${mc.standard_error:.4f} (95% CI: {mc.confidence_interval})")

# 5. Robust Implied Volatility Inversion
iv = implied_volatility(price=c_bsm, spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05)
print(f"Recovered IV: {iv.implied_volatility * 100:.2f}% (Residual: {iv.residual:.2e})")
```

---

## 📊 Authoritative Figures

The engine provides a one-command reproduction script to regenerate all analytical research figures:

```powershell
python scripts/generate_figures.py
```

Generated plots saved in `reports/figures/`:

| Figure | Description | Key Insight |
| :--- | :--- | :--- |
| **`binomial_tree.png`** | CRR Binomial Lattice (American Put) | Displays branching with stock/option node prices and highlights early-exercise boundaries. |
| **`volatility_surface.png`** | 3D Implied Volatility Surface | Visualizes strike/maturity skew, smile term structure, and arbitrage health. |
| **`monte_carlo_simulation.png`** | Monte Carlo Explorer | Illustrates GBM trajectories, terminal price distribution, and empirical convergence. |
| **`crr_convergence.png`** | CRR Binomial Tree vs Steps $N$ | Non-monotonic "sawtooth" convergence approaching exact BSM European boundary. |
| **`greeks_surface.png`** | BSM Greeks ($\Delta, \Gamma, \nu, \Theta$) vs Spot | Illustrates peaked Gamma/Vega at-the-money and passage-of-time decay profiles. |
| **`hedge_simulation.png`** | Discrete Delta Hedging Error | Demonstrates how increasing hedge rebalancing ($N=10 \to 60$) contracts P&L tracking dispersion. |
| **`svi_volatility_smile.png`** | Parametric SVI Calibration | Smooth total variance fitting against market moneyness quotes without negative density. |

---

## 🧪 Verification & Testing

The engine enforces strict numerical invariant assertions and cross-checks against **QuantLib 1.43**:

```powershell
# Run entire test suite (50 tests)
python -m pytest -q

# Run statement coverage report
python -m pytest --cov=options_engine --cov-report=term-missing

# Lint & static analysis checks
ruff check .
ruff format --check .

# Performance profiling across input scales
python scripts/benchmark.py
```

### QuantLib Cross-Validation Guarantee
Every release is benchmarked against QuantLib's `AnalyticEuropeanEngine` and `BinomialVanillaEngine`:
- **European Contracts**: Absolute error relative to QuantLib is $< 10^{-4}$.
- **American Contracts**: Discretization error on 500-step CRR tree matches QuantLib within $< 0.05$.

---

## ⚖️ Model Assumptions & Limitations

1. **Market Friction**: Analytical BSM assumes zero transaction costs, continuous log-normal dynamics, and constant rates.
2. **Exercise Style Mismatch**: Single-stock US equities trade as American options. Direct comparison of vanilla BSM to American market quotes will overlook the early-exercise premium. Use CRR for American options.
3. **SVI Convergence**: Numerical optimizer convergence during SVI fitting does not guarantee absence of arbitrage. Independent butterfly and calendar checks must always be evaluated.
4. **Intended Use**: This software is intended strictly for quantitative research, education, and portfolio demonstration. It does not constitute investment advice or live execution routing.

---

<div align="center">
<sub>Designed and built with mathematical rigor by the quantitative engineering team. Distributed under the MIT License.</sub>
</div>
