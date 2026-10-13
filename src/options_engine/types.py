"""Core data contracts and enumeration types for options pricing engine."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class OptionContract:
    """Option contract definition."""

    underlying: str
    option_type: OptionType
    exercise_style: ExerciseStyle
    strike: float
    expiry: float  # Time to expiry in years
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.expiry < 0:
            raise ValueError(f"Expiry cannot be negative, got {self.expiry}")


@dataclass(frozen=True)
class MarketInputs:
    """Market parameters at valuation datetime."""

    spot: float
    rate: float  # Continuously compounded risk-free rate (decimal)
    dividend_yield: float = 0.0  # Continuously compounded dividend yield (decimal)
    valuation_datetime: str | None = None
    day_count: str = "ACT/365"

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")


@dataclass
class PricingResult:
    """Standardized pricing result across engines."""

    price: float
    model: str
    inputs_id: str
    status: str = "SUCCESS"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_ms: float = 0.0


@dataclass
class MonteCarloResult(PricingResult):
    """Result from Monte Carlo pricing engine."""

    standard_error: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_paths: int = 0
    seed: int | None = None
    variance_reduction: list[str] = field(default_factory=list)


@dataclass
class GreeksResult:
    """Standardized sensitivity result container."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    units: dict[str, str] = field(
        default_factory=lambda: {
            "delta": "dPrice/dSpot",
            "gamma": "d2Price/dSpot2",
            "vega": "dPrice/dVol (1.00 vol unit)",
            "theta": "dPrice/dt (passage of 1 year)",
            "rho": "dPrice/dRate (1.00 rate unit)",
        }
    )
    method: str = "analytical"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class IVResult:
    """Result from implied volatility solver."""

    implied_volatility: float | None
    status: str
    iterations: int = 0
    residual: float = 0.0
    bracket: tuple[float, float] | None = None
    reason_code: str | None = None


@dataclass
class QuoteRecord:
    """Normalized quote record from option market data."""

    timestamp: str
    expiry: str
    strike: float
    call_put: OptionType
    bid: float
    ask: float
    last: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    source: str = "unknown"
    mid: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mid", (self.bid + self.ask) / 2.0)


@dataclass
class SurfaceFitResult:
    """Result from SVI volatility surface fitting."""

    expiry: float
    parameters: dict[str, float]
    objective: float
    residuals: dict[str, float]
    convergence: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
