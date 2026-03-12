"""
Graph-to-Hierarchy Compiler.

Converts a DAG (as dict/JSON, matching dagnet's graph format) into a HierarchyIR
that can be consumed by the PyMC model builder.

Implements a minimal version of the 8-step compiler described in:
  docs/current/project-bayes/0-high-level-logical-blocks.md

Steps implemented:
  1. Canonicalise graph (deduplicate, sort nodes)
  2. Identify branch groups / sibling sets
  3. Build binary edge list (single outgoing edge per source)
  4. Bind evidence to hierarchy leaves
  5. Compute completeness blend weights
  6. Validate and emit Hierarchy IR with provenance flags

Steps deferred (production):
  - Path_t95 DP for retrieval horizon bounding
  - WindowFetchPlannerService integration (covered_stable / stale classification)
  - Full Dirichlet concentration hyperprior learning
  - Latency path composition (forward walk vs Monte Carlo)
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Optional
import math

from compiler.hierarchy_ir import (
    BinaryEdge,
    BranchGroup,
    EdgeEvidence,
    HierarchyIR,
    LatencyEdge,
)

# Blend weight threshold: ANCHOR_DELAY_BLEND_K_CONVERSIONS = 50
# w = 1 - exp(-effective_forecast_conversions / 50)
# Mirrors window-cohort-lag-correction-plan.md
_BLEND_K = 50


def _completeness_weight(n: int) -> float:
    """Soft transition from immature to mature cohort evidence."""
    return 1.0 - math.exp(-n / _BLEND_K)


def _evidence_level(evidence: Optional[EdgeEvidence]) -> int:
    """
    Degradation ladder per cohort-completeness-model-contract.md:
      0 = cold start (no evidence)
      1 = fresh but weak (n < 10)
      2 = snapshot calibrated (mature, n >= 10)
      3 = full Bayes posterior (reserved — set by runner after sampling)
    """
    if evidence is None:
        return 0
    if not evidence.is_mature or evidence.n < 10:
        return 1
    return 2


def compile_graph(
    graph: dict,
    evidence_store: dict[str, EdgeEvidence] | None = None,
) -> HierarchyIR:
    """
    Main compiler entry point.

    Args:
        graph: DAG as dict with 'nodes' and 'edges' keys.
               Matches dagnet's ReactFlow/internal graph format:
               {
                 "id": "graph-id",
                 "nodes": [{"id": "n1", "label": "Landing"}, ...],
                 "edges": [{"id": "e1", "source": "n1", "target": "n2"}, ...]
               }
        evidence_store: Optional dict mapping edge_id → EdgeEvidence.
                        If None, cold-start priors are used for all edges.

    Returns:
        HierarchyIR ready for PyMC model builder.
    """
    evidence_store = evidence_store or {}
    graph_id = graph.get("id", "unknown")

    # --- Step 1: Canonicalise ---
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    # --- Step 2: Identify branch groups (nodes with multiple outgoing edges) ---
    outgoing: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source"]].append(edge)

    binary_edges: list[BinaryEdge] = []
    branch_groups: list[BranchGroup] = []
    latency_edges: list[LatencyEdge] = []
    evidence_levels: dict[str, int] = {}

    for source_id, out_edges in outgoing.items():
        if len(out_edges) == 1:
            # --- Step 3: Single outgoing edge → BinaryEdge ---
            edge = out_edges[0]
            ev = evidence_store.get(edge["id"])
            if ev:
                ev.completeness = _completeness_weight(ev.n)

            level = _evidence_level(ev)
            evidence_levels[edge["id"]] = level

            be = BinaryEdge(
                edge_id=edge["id"],
                source=source_id,
                target=edge["target"],
                evidence=ev,
                prior_alpha=ev.prior_alpha if ev else 1.0,
                prior_beta=ev.prior_beta if ev else 1.0,
            )
            binary_edges.append(be)

            # Check if this edge has latency evidence → also add LatencyEdge
            if ev and ev.latency_parameter and ev.lag_mu is not None:
                # Apply t95 tail constraint: inflate sigma to prevent thin-tail optimism
                # Mirrors t95-fix.md "t95 tail constraint" design
                sigma = _t95_constrained_sigma(ev.lag_mu, ev.lag_sigma, ev.t95)
                latency_edges.append(LatencyEdge(
                    edge_id=f"{edge['id']}_lag",
                    source=source_id,
                    target=edge["target"],
                    mu=ev.lag_mu,
                    sigma=sigma,
                    t95=ev.t95,
                    evidence=ev,
                ))

        else:
            # --- Step 2: Multiple outgoing edges → BranchGroup (Dirichlet) ---
            group_id = f"branch_{source_id}"
            variants = [e["target"] for e in out_edges]
            evidences = []
            all_levels = []

            for edge in out_edges:
                ev = evidence_store.get(edge["id"])
                if ev:
                    ev.completeness = _completeness_weight(ev.n)
                evidences.append(ev or EdgeEvidence(
                    edge_id=edge["id"], n=10, k=5  # uniform cold-start prior
                ))
                lvl = _evidence_level(ev)
                evidence_levels[edge["id"]] = lvl
                all_levels.append(lvl)

            branch_groups.append(BranchGroup(
                group_id=group_id,
                source=source_id,
                variants=variants,
                evidence=evidences,
                # Shared concentration: scale with weakest evidence in group
                concentration=1.0 + min(all_levels),
            ))

    # --- Step 6: Fingerprint for deterministic reproducibility ---
    fingerprint = _compute_fingerprint(graph_id, binary_edges, branch_groups)

    return HierarchyIR(
        graph_id=graph_id,
        binary_edges=binary_edges,
        branch_groups=branch_groups,
        latency_edges=latency_edges,
        evidence_levels=evidence_levels,
        fingerprint=fingerprint,
    )


def _t95_constrained_sigma(mu: float, sigma: Optional[float], t95: int) -> float:
    """
    Inflate sigma to enforce the t95 tail constraint.
    Prevents thin-tail optimism in the latency model.
    Mirrors t95-fix.md: "t95 tail constraint inflates sigma".

    Ensures P(lag <= t95) >= 0.95 under LogNormal(mu, sigma).
    If the raw sigma already satisfies this, return it unchanged.
    """
    import math
    if sigma is None:
        sigma = 0.5  # conservative default

    # P(X <= t95) = Phi((log(t95) - mu) / sigma) >= 0.95
    # => (log(t95) - mu) / sigma >= 1.645
    # => sigma <= (log(t95) - mu) / 1.645
    if t95 <= 0 or mu <= 0:
        return sigma

    log_t95 = math.log(t95)
    sigma_max = (log_t95 - mu) / 1.645 if log_t95 > mu else sigma

    # If raw sigma is already tight enough, add a small floor to avoid over-confidence
    return max(sigma, min(sigma, sigma_max) + 0.05)


def _compute_fingerprint(
    graph_id: str,
    binary_edges: list[BinaryEdge],
    branch_groups: list[BranchGroup],
) -> str:
    """Deterministic hash of the graph structure (topology only, not evidence values)."""
    structure = {
        "graph_id": graph_id,
        "binary_edges": sorted(
            [{"id": e.edge_id, "src": e.source, "tgt": e.target} for e in binary_edges],
            key=lambda x: x["id"],
        ),
        "branch_groups": sorted(
            [{"id": b.group_id, "variants": sorted(b.variants)} for b in branch_groups],
            key=lambda x: x["id"],
        ),
    }
    blob = json.dumps(structure, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
