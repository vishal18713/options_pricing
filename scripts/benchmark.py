"""Performance profiling and runtime benchmarks for BSM, CRR, and Monte Carlo."""

import time
from typing import Any

import numpy as np
import pandas as pd

from options_engine.binomial import crr_price
from options_engine.black_scholes import bsm_price
from options_engine.monte_carlo import monte_carlo_price


def benchmark_engine_runtimes(repetitions: int = 5) -> pd.DataFrame:
    """Benchmark execution time and throughput for BSM, CRR, and Monte Carlo."""
    results: list[dict[str, Any]] = []

    # 1. Vectorized BSM (100,000 contracts)
    n_contracts = 100_000
    s_arr = np.linspace(80, 120, n_contracts)
    k_arr = np.full(n_contracts, 100.0)

    # Warmup
    _ = bsm_price(s_arr[:10], k_arr[:10], 1.0, 0.05, 0.20)

    bsm_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        _ = bsm_price(s_arr, k_arr, 1.0, 0.05, 0.20)
        bsm_times.append(time.perf_counter() - t0)

    median_bsm = float(np.median(bsm_times))
    results.append(
        {
            "Engine": "BSM Vectorized",
            "Scale": f"{n_contracts:,} contracts",
            "Median_Time_Sec": median_bsm,
            "Throughput_contracts_per_sec": n_contracts / median_bsm if median_bsm > 0 else np.nan,
        }
    )

    # 2. CRR Binomial Tree (steps=100, 300, 500)
    for steps in [100, 300, 500]:
        crr_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = crr_price(100.0, 100.0, 1.0, 0.05, 0.20, steps=steps, exercise_style="american")
            crr_times.append(time.perf_counter() - t0)
        med_crr = float(np.median(crr_times))
        results.append(
            {
                "Engine": "CRR American Tree",
                "Scale": f"{steps} steps",
                "Median_Time_Sec": med_crr,
                "Throughput_contracts_per_sec": 1.0 / med_crr if med_crr > 0 else np.nan,
            }
        )

    # 3. Monte Carlo (paths=50,000, 200,000)
    for paths in [50_000, 200_000]:
        mc_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = monte_carlo_price(100.0, 100.0, 1.0, 0.05, 0.20, n_paths=paths, antithetic=True)
            mc_times.append(time.perf_counter() - t0)
        med_mc = float(np.median(mc_times))
        results.append(
            {
                "Engine": "Monte Carlo (Antithetic)",
                "Scale": f"{paths:,} paths",
                "Median_Time_Sec": med_mc,
                "Throughput_contracts_per_sec": 1.0 / med_mc if med_mc > 0 else np.nan,
            }
        )

    return pd.DataFrame(results)
