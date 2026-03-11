"""
pytest tests for the Bayesian A/B comparison runner.
Uses known inputs to verify statistical correctness.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.runner.bayesian_ab import run_bayesian_ab
from lib.runner.types import BayesianABRequest, VariantInput
from lib.stats_enhancement import enhance_mcmc_stdlib, prior_from_param_registry


def test_clear_winner_b():
    """B has meaningfully higher conversion — expect ship_b recommendation."""
    req = BayesianABRequest(
        variant_a=VariantInput(name="classic", n=1000, k=720),
        variant_b=VariantInput(name="quick", n=1000, k=820),
        n_samples=20_000
    )
    result = run_bayesian_ab(req)
    assert result.p_b_greater_than_a > 0.95
    assert result.recommendation == "ship_b"
    assert result.stopping_criterion_met is True
    assert result.variant_b.posterior_mean > result.variant_a.posterior_mean


def test_no_clear_winner():
    """Near-identical conversion rates — expect continue."""
    req = BayesianABRequest(
        variant_a=VariantInput(name="classic", n=500, k=225),
        variant_b=VariantInput(name="quick", n=500, k=235),
        n_samples=20_000
    )
    result = run_bayesian_ab(req)
    assert result.recommendation == "continue"
    assert result.stopping_criterion_met is False


def test_informative_prior_from_param_registry():
    """Param-registry prior (mean=0.45, n=10000) should shift posteriors correctly."""
    prior_alpha, prior_beta = prior_from_param_registry(mean=0.45, n=10000)
    assert prior_alpha == 4500.0
    assert prior_beta == 5500.0

    result = enhance_mcmc_stdlib(n=100, k=50, prior_alpha=prior_alpha, prior_beta=prior_beta)
    # With strong prior of 0.45, posterior should be pulled toward 0.45 not 0.50
    assert result["mean"] < 0.47
    assert result["ci_low"] < result["mean"] < result["ci_high"]


def test_ci_bounds_valid():
    """Credible interval must be within [0, 1] and mean must be inside it."""
    result = enhance_mcmc_stdlib(n=200, k=100)
    assert 0.0 <= result["ci_low"] <= result["mean"] <= result["ci_high"] <= 1.0


def test_ecommerce_fixture():
    """
    Mirrors the ecommerce-checkout-flow.json fixture:
    classic: p=0.72, quick: p=0.82 (post-it says +8pp uplift confirmed).
    """
    req = BayesianABRequest(
        variant_a=VariantInput(name="classic", n=2000, k=1440),  # 0.72
        variant_b=VariantInput(name="quick", n=2000, k=1640),    # 0.82
        n_samples=20_000
    )
    result = run_bayesian_ab(req)
    assert result.p_b_greater_than_a > 0.999
    assert result.recommendation == "ship_b"
    print(f"P(quick > classic) = {result.p_b_greater_than_a:.4f}")
    print(f"Expected loss of keeping classic = {result.expected_loss_choosing_a:.4f}")
