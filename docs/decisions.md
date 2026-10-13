# Decisions and Numerical Conventions Log

This document records the mathematical conventions, units, assumptions, and architectural decisions approved for the **Options Pricing & Greeks Engine** (Version 1.0).

---

## 1. Units and Input Conventions

| Parameter | Unit / Representation | Example / Rule |
| :--- | :--- | :--- |
| **Spot ($S$) & Strike ($K$)** | Same currency and positive real numbers | $S > 0$, $K > 0$ |
| **Time to Expiry ($T$)** | Years as a decimal float | $T \ge 0$ |
| **Day-Count Convention** | `ACT/365` (Calendar days / 365.0) | Default: ACT/365. All interfaces make time explicit. |
| **Risk-Free Rate ($r$)** | Continuously compounded annual rate (decimal) | 5% is represented as `0.05` |
| **Dividend Yield ($q$)** | Continuously compounded annual rate (decimal) | 2% is represented as `0.02` |
| **Volatility ($\sigma$)** | Annualized standard deviation (decimal) | 20% is represented as `0.20` |

---

## 2. Greeks and Scaling Conventions

- **Analytical Formulation**:
  - Formulas evaluated in standard continuous dividend BSM framework.
- **Theta ($\Theta$)**:
  - **Convention**: Passage of calendar time ($\partial V / \partial t = -\partial V / \partial T$).
  - **Sign**: Typically negative for long vanilla at-the-money options under positive rates/volatility.
  - **Scaling**: Reported per 1-year time passage by default, with helper conversions to 1 calendar day ($\Theta / 365$).
- **Vega ($\nu$)**:
  - **Convention**: First derivative with respect to volatility $\partial V / \partial \sigma$.
  - **Scaling**: Derivative per 1.00 (100 percentage points) change in $\sigma$. (Multiply by 0.01 for 1% bump).
- **Rho ($\rho$)**:
  - **Convention**: First derivative with respect to interest rate $\partial V / \partial r$.
  - **Scaling**: Derivative per 1.00 (100 percentage points) change in $r$. (Multiply by 0.01 for 100 bps bump).
- **Gamma ($\Gamma$)**:
  - Non-negative for vanilla long calls and puts.
- **Delta ($\Delta$)**:
  - Call $\Delta \in [0, e^{-qT}]$; Put $\Delta \in [-e^{-qT}, 0]$.

---

## 3. Numerical Model Policies

1. **Binomial Tree (Cox-Ross-Rubinstein)**:
   - Discretization step $\Delta t = T / N$.
   - Up/down factors: $u = e^{\sigma \sqrt{\Delta t}}$, $d = 1 / u$.
   - Risk-neutral probability: $p = \frac{e^{(r-q)\Delta t} - d}{u - d}$.
   - Requires $0 \le p \le 1$. If $p \notin [0, 1]$, return typed diagnostic error (`MODEL_INVALID_PROBABILITY`).
   - American early exercise: Node payoff = $\max(\text{Continuation Value}, \text{Intrinsic Value})$.

2. **Monte Carlo Simulation**:
   - Random generator: `numpy.random.default_rng(seed)`.
   - Antithetic variates: Pairs $(Z, -Z)$. Variance and standard error calculated over independent pair averages $Y_i = \frac{f(Z_i) + f(-Z_i)}{2}$.
   - Control variates: Option European payoff with analytical BSM price as control variate.

3. **Implied Volatility Solver**:
   - Primary solver: Bracketed Brent solver (`scipy.optimize.brentq`) over $[\sigma_{\min}, \sigma_{\max}]$ (default $[10^{-6}, 5.0]$).
   - Strict price bounds verification prior to root finding (enforces European no-arbitrage lower/upper bounds).
   - Non-convergence or out-of-bounds quotes return typed `IVResult` with reason codes (no silent clamp, no numeric zero).

4. **SVI Volatility Surface**:
   - Parametric slice representation: $w(k) = a + b(\rho(k - m) + \sqrt{(k - m)^2 + \eta^2})$ where $k = \ln(K / F)$, $w = \sigma_{\text{IV}}^2 T$.
   - Parameters constraints: $b \ge 0$, $\eta > 0$, $|\rho| < 1$, $a + b\eta\sqrt{1-\rho^2} \ge 0$.
   - Arbitrage verification: Calendar monotonicity $\frac{\partial w}{\partial T} \ge 0$ at constant $k$ and butterfly non-negative density check.

---

## 4. Error and Warning Taxonomy

- `INPUT_*`: Invalid domain (e.g. $S \le 0$, $K \le 0$, $T < 0$, $\sigma \le 0$, NaN/Inf).
- `MODEL_*`: Discretization issues (e.g. CRR $p \notin [0, 1]$).
- `IV_*`: Price out of theoretical no-arbitrage bounds or Brent solver iteration limit reached.
- `DATA_*`: Quote issues (crossed market, non-positive bid, excessive spread, missing fields).
- `CAL_*`: SVI optimizer failure or parameter boundary hits.
- `VALIDATION_*`: Tolerance exceedance against analytical/QuantLib benchmarks.
