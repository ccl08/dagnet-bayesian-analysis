"""
Bayesian A/B comparison runner.
Computes P(B > A) via Monte Carlo using stdlib random.betavariate.
No numpy or scipy required — safe for Vercel deployment.
"""
import random
import math
from .types import BayesianABRequest, BayesianABResponse, VariantResult

STOPPING_THRESHOLD = 0.95


def _posterior(prior_alpha: float, prior_beta: float, k: int, n: int) -> tuple:
    alpha = prior_alpha + k
    beta = prior_beta + (n - k)
    return alpha, beta


def _beta_mean_and_ci(alpha: float, beta: float) -> tuple:
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = math.sqrt(variance)
    z = 1.96
    lo = max(0.0, mean - z * std)
    hi = min(1.0, mean + z * std)
    return mean, lo, hi


def _p_b_greater_than_a(
    alpha_a: float, beta_a: float,
    alpha_b: float, beta_b: float,
    n_samples: int = 10_000
) -> float:
    """Monte Carlo estimate of P(B > A) using stdlib only."""
    wins = sum(
        random.betavariate(alpha_b, beta_b) > random.betavariate(alpha_a, beta_a)
        for _ in range(n_samples)
    )
    return wins / n_samples


def _expected_loss(alpha_a, beta_a, alpha_b, beta_b, n_samples=10_000) -> float:
    """Expected loss of choosing A when B might be better."""
    total_loss = 0.0
    for _ in range(n_samples):
        sample_a = random.betavariate(alpha_a, beta_a)
        sample_b = random.betavariate(alpha_b, beta_b)
        total_loss += max(0.0, sample_b - sample_a)
    return total_loss / n_samples


def run_bayesian_ab(request: BayesianABRequest) -> BayesianABResponse:
    alpha_a, beta_a = _posterior(
        request.prior_alpha, request.prior_beta,
        request.variant_a.k, request.variant_a.n
    )
    alpha_b, beta_b = _posterior(
        request.prior_alpha, request.prior_beta,
        request.variant_b.k, request.variant_b.n
    )

    mean_a, lo_a, hi_a = _beta_mean_and_ci(alpha_a, beta_a)
    mean_b, lo_b, hi_b = _beta_mean_and_ci(alpha_b, beta_b)

    p_b_wins = _p_b_greater_than_a(alpha_a, beta_a, alpha_b, beta_b, request.n_samples)
    loss = _expected_loss(alpha_a, beta_a, alpha_b, beta_b, request.n_samples)

    stopping = p_b_wins > STOPPING_THRESHOLD or (1 - p_b_wins) > STOPPING_THRESHOLD

    if p_b_wins > STOPPING_THRESHOLD:
        recommendation = "ship_b"
    elif (1 - p_b_wins) > STOPPING_THRESHOLD:
        recommendation = "ship_a"
    else:
        recommendation = "continue"

    return BayesianABResponse(
        variant_a=VariantResult(name=request.variant_a.name, n=request.variant_a.n,
                                k=request.variant_a.k, posterior_mean=mean_a,
                                ci_low=lo_a, ci_high=hi_a, alpha=alpha_a, beta=beta_a),
        variant_b=VariantResult(name=request.variant_b.name, n=request.variant_b.n,
                                k=request.variant_b.k, posterior_mean=mean_b,
                                ci_low=lo_b, ci_high=hi_b, alpha=alpha_b, beta=beta_b),
        p_b_greater_than_a=p_b_wins,
        expected_loss_choosing_a=loss,
        recommendation=recommendation,
        stopping_criterion_met=stopping
    )
