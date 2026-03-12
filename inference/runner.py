"""
Inference Runner.

Runs pm.sample() on a compiled PyMC model and extracts posterior artefacts:
  - Posterior mean and 94% HDI per parameter
  - Variant win probabilities for branch groups (P(variant_i is best))
  - Pairwise P(B > A) for binary edges

Artefacts are returned as plain dicts suitable for persistence to
the snapshot DB or JSON files — no PyMC objects leak out.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pymc as pm
import arviz as az

from compiler.hierarchy_ir import HierarchyIR
from inference.model_builder import build_model


@dataclass
class EdgePosterior:
    """Posterior summary for a single binary edge."""
    edge_id: str
    mean: float
    hdi_low: float
    hdi_high: float
    evidence_level: int


@dataclass
class BranchPosterior:
    """Posterior summary for a branch group."""
    group_id: str
    variants: list[str]
    means: list[float]
    hdi_lows: list[float]
    hdi_highs: list[float]
    win_probs: list[float]  # P(variant_i has highest weight)


@dataclass
class InferenceResult:
    """
    Full posterior artefact set for a graph.
    Persisted to snapshot DB after nightly sampling run.
    """
    graph_id: str
    fingerprint: str
    edge_posteriors: list[EdgePosterior] = field(default_factory=list)
    branch_posteriors: list[BranchPosterior] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"InferenceResult: {self.graph_id} [{self.fingerprint}]"]
        for ep in self.edge_posteriors:
            lines.append(
                f"  edge {ep.edge_id}: "
                f"mean={ep.mean:.3f} HDI=[{ep.hdi_low:.3f}, {ep.hdi_high:.3f}] "
                f"(level {ep.evidence_level})"
            )
        for bp in self.branch_posteriors:
            for i, v in enumerate(bp.variants):
                lines.append(
                    f"  {bp.group_id}/{v}: "
                    f"mean={bp.means[i]:.3f} "
                    f"HDI=[{bp.hdi_lows[i]:.3f}, {bp.hdi_highs[i]:.3f}] "
                    f"P(best)={bp.win_probs[i]:.3f}"
                )
        if self.diagnostics:
            lines.append(f"  diagnostics: {self.diagnostics}")
        return "\n".join(lines)


def run_inference(
    ir: HierarchyIR,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 42,
    progressbar: bool = False,
) -> InferenceResult:
    """
    Build PyMC model from IR, sample, and return posterior artefacts.

    Args:
        ir: HierarchyIR from the compiler.
        draws: MCMC draws per chain.
        tune: Tuning steps (discarded).
        chains: Number of parallel chains.
        random_seed: For reproducibility.
        progressbar: Show tqdm progress (False for nightly batch runs).

    Returns:
        InferenceResult with posterior summaries — no PyMC objects.
    """
    model = build_model(ir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
            )

    result = InferenceResult(
        graph_id=ir.graph_id,
        fingerprint=ir.fingerprint or "",
    )

    # --- Extract binary edge posteriors ---
    for edge in ir.binary_edges:
        var_name = f"p_{edge.edge_id}"
        if var_name in trace.posterior:
            samples = trace.posterior[var_name].values.flatten()
            hdi = az.hdi(samples, hdi_prob=0.94)
            result.edge_posteriors.append(EdgePosterior(
                edge_id=edge.edge_id,
                mean=float(samples.mean()),
                hdi_low=float(hdi[0]),
                hdi_high=float(hdi[1]),
                evidence_level=ir.evidence_levels.get(edge.edge_id, 0),
            ))

    # --- Extract branch group posteriors ---
    for group in ir.branch_groups:
        var_name = f"w_{group.group_id}"
        if var_name in trace.posterior:
            # Shape: (chains, draws, n_variants)
            samples = trace.posterior[var_name].values
            # Flatten chains × draws → (n_samples, n_variants)
            flat = samples.reshape(-1, len(group.variants))

            means = flat.mean(axis=0).tolist()
            hdis = [az.hdi(flat[:, i], hdi_prob=0.94) for i in range(len(group.variants))]
            hdi_lows = [float(h[0]) for h in hdis]
            hdi_highs = [float(h[1]) for h in hdis]

            # P(variant_i is best) = fraction of samples where it has max weight
            best_idx = np.argmax(flat, axis=1)
            win_probs = [
                float((best_idx == i).mean()) for i in range(len(group.variants))
            ]

            result.branch_posteriors.append(BranchPosterior(
                group_id=group.group_id,
                variants=group.variants,
                means=means,
                hdi_lows=hdi_lows,
                hdi_highs=hdi_highs,
                win_probs=win_probs,
            ))

    # --- Diagnostics: R-hat (convergence check) ---
    try:
        rhat = az.rhat(trace)
        max_rhat = float(max(
            rhat[v].values.max()
            for v in rhat.data_vars
            if not np.isnan(rhat[v].values).all()
        ))
        result.diagnostics["max_rhat"] = round(max_rhat, 4)
        result.diagnostics["converged"] = max_rhat < 1.05
    except Exception:
        pass

    return result
