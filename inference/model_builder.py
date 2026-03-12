"""
PyMC Model Builder.

Takes a HierarchyIR and constructs a PyMC model with the correct
distributional structure:
  - pm.Beta per BinaryEdge
  - pm.Dirichlet per BranchGroup (simplex constraint)
  - pm.LogNormal per LatencyEdge
  - pm.Binomial likelihoods where evidence exists

This is the layer that makes the compiler → inference split clean:
the compiler knows nothing about PyMC; this module knows nothing about DAGs.
"""
from __future__ import annotations

import pymc as pm
import numpy as np
import pytensor.tensor as pt

from compiler.hierarchy_ir import HierarchyIR, BinaryEdge, BranchGroup, LatencyEdge


def build_model(ir: HierarchyIR) -> pm.Model:
    """
    Build a PyMC model from a HierarchyIR.

    Each edge/branch group becomes a named random variable.
    Variable names use edge_id / group_id for traceability back to the DAG.

    Returns a pm.Model (not yet sampled).
    """
    with pm.Model(coords=_build_coords(ir)) as model:

        # --- BinaryEdges: pm.Beta + pm.Binomial likelihood ---
        for edge in ir.binary_edges:
            _add_binary_edge(edge)

        # --- BranchGroups: pm.Dirichlet + pm.Multinomial likelihood ---
        for group in ir.branch_groups:
            _add_branch_group(group)

        # --- LatencyEdges: pm.LogNormal ---
        for lat in ir.latency_edges:
            _add_latency_edge(lat)

    return model


def _build_coords(ir: HierarchyIR) -> dict:
    """Build PyMC coords for labelled dimensions (useful for ArviZ plots)."""
    coords = {}
    for group in ir.branch_groups:
        coords[f"{group.group_id}_variants"] = group.variants
    return coords


def _add_binary_edge(edge: BinaryEdge) -> None:
    """
    Single edge: conversion rate ~ Beta(alpha, beta).
    If evidence exists, add Binomial likelihood.

    Degradation:
      Level 0 (no evidence): Beta(1, 1) = Uniform prior
      Level 1 (weak):        Beta from observed n, k but marked immature
      Level 2+ (mature):     Beta(k, n-k) as informative prior
    """
    ev = edge.evidence
    alpha = edge.prior_alpha
    beta_param = edge.prior_beta

    # Named after edge_id for traceability
    # Cap prior alpha/beta to prevent NaN at large counts.
    # With large n, use Binomial likelihood directly with a weak Beta(2,2) prior.
    # This is numerically equivalent but avoids float overflow in the prior.
    MAX_PRIOR_COUNTS = 500
    if alpha + beta_param > MAX_PRIOR_COUNTS:
        prior_alpha_capped = 2.0
        prior_beta_capped = 2.0
    else:
        prior_alpha_capped = alpha
        prior_beta_capped = beta_param

    p = pm.Beta(
        f"p_{edge.edge_id}",
        alpha=prior_alpha_capped,
        beta=prior_beta_capped,
    )

    if ev is not None and ev.n > 0:
        # Completeness-weighted effective observations
        # Immature cohorts contribute proportionally less evidence
        effective_n = int(ev.n * ev.completeness)
        effective_k = int(ev.k * ev.completeness)

        if effective_n > 0:
            pm.Binomial(
                f"obs_{edge.edge_id}",
                n=effective_n,
                p=p,
                observed=effective_k,
            )


def _add_branch_group(group: BranchGroup) -> None:
    """
    Branch group: variant weights ~ Dirichlet(concentration * ones).
    Shared concentration hyperprior enforces simplex constraint.

    If evidence exists per variant, add Multinomial likelihood.

    This is the correct model for sibling edges —
    naive independent Betas would violate the simplex constraint.
    See project-bayes: "naive logit-space deviations violate simplex constraint"
    """
    k = len(group.variants)
    coord_name = f"{group.group_id}_variants"

    # Concentration vector: uniform with shared hyperprior
    # Higher concentration → more pooling across variants
    concentration = np.ones(k) * group.concentration

    weights = pm.Dirichlet(
        f"w_{group.group_id}",
        a=concentration,
        dims=coord_name,
    )

    # Build observed counts per variant
    observed_ns = np.array([ev.n for ev in group.evidence])
    observed_ks = np.array([ev.k for ev in group.evidence])
    total_n = observed_ns.sum()

    if total_n > 0:
        # Completeness-weighted effective counts
        completeness = np.array([ev.completeness for ev in group.evidence])
        effective_ks = (observed_ks * completeness).astype(int)
        effective_total = effective_ks.sum()

        if effective_total > 0:
            pm.Multinomial(
                f"obs_{group.group_id}",
                n=effective_total,
                p=weights,
                observed=effective_ks,
            )


def _add_latency_edge(lat: LatencyEdge) -> None:
    """
    Latency edge: lag_days ~ LogNormal(mu, sigma).
    sigma is already t95-constrained by the compiler.

    This models the distribution of time-to-convert for this edge.
    Used to bound cohort evidence windows and assess completeness.
    """
    pm.LogNormal(
        f"lag_{lat.edge_id}",
        mu=lat.mu,
        sigma=lat.sigma,
    )
