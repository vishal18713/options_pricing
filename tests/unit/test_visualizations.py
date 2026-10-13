"""Unit tests for the new UI visualization helper functions."""

from options_engine.binomial import build_crr_tree_visualization
from options_engine.monte_carlo import simulate_monte_carlo_paths


def test_build_crr_tree_visualization():
    tree_data = build_crr_tree_visualization(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.05,
        volatility=0.20,
        steps=3,
        option_type="put",
        exercise_style="american",
    )

    assert tree_data["steps"] == 3
    # In a recombining binomial tree with N steps, total nodes = (N + 1)(N + 2) / 2
    # For N=3: 4 * 5 / 2 = 10 nodes
    assert len(tree_data["nodes"]) == 10
    # Edges: from each node at step i (0..N-1), 2 edges (up and down) = 2 * (1 + 2 + 3) = 12 edges
    assert len(tree_data["edges"]) == 12

    # Check root price is positive
    assert tree_data["root_price"] > 0.0


def test_simulate_monte_carlo_paths():
    mc_data = simulate_monte_carlo_paths(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.05,
        volatility=0.20,
        n_display_paths=15,
        n_total_paths=1_000,
        n_time_steps=20,
        seed=42,
    )

    assert mc_data["sample_paths"].shape == (15, 21)
    assert len(mc_data["terminal_prices"]) == 1_000
    assert len(mc_data["payoffs"]) == 1_000
    assert len(mc_data["running_means"]) == 100
    assert mc_data["final_price"] > 0.0
