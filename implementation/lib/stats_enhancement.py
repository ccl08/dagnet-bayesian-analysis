"""
Beta-Binomial Bayesian posterior for binary conversion rates.
Pure Python stdlib only — no numpy or scipy required.
Designed for Vercel deployment (50MB function size limit).
"""
import math


def _beta_credible_interval(alpha: float, beta: float, confidence: float = 0.95) -> tuple:
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    z = 1.96  # 95% CI
    std = math.sqrt(variance)
    return (max(0.0, mean - z * std), min(1.0, mean + z * std))


def enhance_mcmc_stdlib(n: int, k: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> dict:
    """
    Beta-Binomial conjugate posterior. Uniform prior Beta(1,1) by default.
    Accepts informative priors from param-registry YAML values.
    """
    alpha = prior_alpha + k
    beta = prior_beta + (n - k)
    mean = alpha / (alpha + beta)
    lo, hi = _beta_credible_interval(alpha, beta)
    return {
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "beta": beta,
        "method": "bayesian_beta_binomial"
    }


def prior_from_param_registry(mean: float, n: int) -> tuple:
    """
    Convert param-registry observation to Beta prior parameters.
    Example: mean=0.45, n=10000 → Beta(4500, 5500)
    """
    k = round(mean * n)
    return float(k), float(n - k)
