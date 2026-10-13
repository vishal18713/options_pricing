# Options Pricing & Greeks Engine

## Overview
A modular Python quantitative finance library that prices vanilla European and American options using three independent numerical methods (Black-Scholes-Merton, Cox-Ross-Rubinstein Binomial Tree, and Monte Carlo with Antithetic and Control Variates), computes analytical and finite-difference Greeks, robustly solves implied volatility, and fits an SVI (Stochastic Volatility Inspired) volatility surface with full calendar and butterfly arbitrage diagnostics.

## Features
- **Black-Scholes-Merton (BSM)**: Closed-form European pricing with continuous dividend yield, robust boundary/limiting behavior, and analytical Greeks (Delta, Gamma, Vega, Theta, Rho).
- **CRR Binomial Tree**: European and American vanilla calls and puts with early-exercise detection, dividend support, and convergence diagnostics.
- **Monte Carlo Simulation**: Risk-neutral European pricing with antithetic variates, optional BSM control variate, standard error estimation, and confidence intervals.
- **Greeks & Sensitivities**: Analytical formulas, central finite differences, and pathwise estimators with configurable bumps and cross-method comparisons.
- **Market Data & Quote Cleaning**: Ingestion adapters (yfinance), rigorous quote validation, and auditable reason codes for rejected observations.
- **Implied Volatility & SVI Surface**: Bracketed Brent solver, forward estimation via put-call parity, raw SVI parametric slice fitting, and static arbitrage diagnostics (calendar monotonicity & butterfly condition).
- **Independent Validation**: Benchmark and cross-validation suite against QuantLib.

## Installation
```bash
python -m pip install -e ".[dev]"
```

## Running Tests
```bash
python -m pytest -q
```
