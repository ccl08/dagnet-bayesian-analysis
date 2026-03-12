"""
Evidence Reader.

Loads EdgeEvidence objects from dagnet's parameter YAML files or snapshot DB dicts.
Mirrors the structure of example-param-household-to-switch.yaml.

In production this would query the Postgres snapshot DB directly.
For the prototype, it reads YAML param files or accepts plain dicts.
"""
from __future__ import annotations

import math
from typing import Any

from compiler.hierarchy_ir import EdgeEvidence


def evidence_from_yaml(data: dict[str, Any]) -> EdgeEvidence:
    """
    Parse a dagnet parameter YAML dict into an EdgeEvidence object.

    Expected YAML shape (from example-param-household-to-switch.yaml):
      edge_id: edge-household-to-switch
      latency_parameter: true
      t95: 30
      n: 2274
      k: 190
      mean: 0.0836
      lag_mean_days: 4.2
      lag_median_days: 2.1
    """
    n = int(data.get("n", 0))
    k = int(data.get("k", 0))

    # Fit lognormal from mean + median if lag params present
    lag_mu, lag_sigma = _fit_lognormal(data)

    is_mature = data.get("is_mature", True)
    completeness = 1.0 - math.exp(-n / 50) if n > 0 else 0.0

    return EdgeEvidence(
        edge_id=str(data.get("edge_id", "unknown")),
        n=n,
        k=k,
        t95=int(data.get("t95", 30)),
        latency_parameter=bool(data.get("latency_parameter", False)),
        lag_mean_days=data.get("lag_mean_days"),
        lag_median_days=data.get("lag_median_days"),
        lag_mu=lag_mu,
        lag_sigma=lag_sigma,
        is_mature=is_mature,
        completeness=completeness,
    )


def evidence_from_snapshot(row: dict[str, Any]) -> EdgeEvidence:
    """
    Parse a snapshot DB row into EdgeEvidence.

    Snapshot DB columns (from cohort-completeness-model-contract.md):
      median_lag_days, mean_lag_days, anchor_median_lag_days,
      anchor_mean_lag_days, onset_delta_days, anchor_day, retrieved_at
    """
    n = int(row.get("n", 0))
    k = int(row.get("k", 0))

    lag_data = {
        "lag_mean_days": row.get("mean_lag_days"),
        "lag_median_days": row.get("median_lag_days"),
    }
    lag_mu, lag_sigma = _fit_lognormal(lag_data)

    return EdgeEvidence(
        edge_id=str(row.get("edge_id", "unknown")),
        n=n,
        k=k,
        t95=int(row.get("t95", 30)),
        latency_parameter=row.get("latency_parameter", False),
        lag_mean_days=row.get("mean_lag_days"),
        lag_median_days=row.get("median_lag_days"),
        lag_mu=lag_mu,
        lag_sigma=lag_sigma,
        is_mature=row.get("is_mature", True),
        completeness=1.0 - math.exp(-n / 50) if n > 0 else 0.0,
    )


def _fit_lognormal(data: dict) -> tuple[float | None, float | None]:
    """
    Fit lognormal μ and σ from mean and median lag values.

    For LogNormal(μ, σ):
      median = exp(μ)         → μ = log(median)
      mean   = exp(μ + σ²/2)  → σ = sqrt(2 * (log(mean) - μ))

    Mirrors the hand-ported approach in lag_distribution_utils.py
    (stdlib math only, no scipy).
    """
    mean_days = data.get("lag_mean_days")
    median_days = data.get("lag_median_days")

    if median_days and median_days > 0:
        mu = math.log(median_days)
        if mean_days and mean_days > median_days:
            sigma_sq = 2 * (math.log(mean_days) - mu)
            sigma = math.sqrt(max(sigma_sq, 0.01))
        else:
            sigma = 0.5  # conservative default
        return mu, sigma

    return None, None
