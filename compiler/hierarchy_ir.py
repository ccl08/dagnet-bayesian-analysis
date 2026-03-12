"""
Hierarchy IR — Intermediate Representation between DAG topology and PyMC model.

Emitted by the compiler, consumed by the model builder.
Design mirrors project-bayes/0-high-level-logical-blocks.md §graph-to-hierarchy compiler.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeEvidence:
    """
    Observed evidence bound to a single DAG edge from the snapshot DB.
    Mirrors example-param-household-to-switch.yaml structure.
    """
    edge_id: str
    n: int                          # total observations
    k: int                          # conversions
    t95: int = 30                   # 95th percentile lag in days (DEFAULT_T95_DAYS)
    latency_parameter: bool = False # True = edge has meaningful lag
    lag_mean_days: Optional[float] = None
    lag_median_days: Optional[float] = None
    lag_mu: Optional[float] = None  # fitted lognormal μ (from lag_distribution_utils)
    lag_sigma: Optional[float] = None  # fitted lognormal σ
    is_mature: bool = True          # False = cohort still accumulating conversions
    completeness: float = 1.0       # blend weight w = 1 - exp(-effective_n / 50)

    @property
    def prior_alpha(self) -> float:
        """Beta prior alpha from observed data (used as informative prior)."""
        return max(1.0, self.k)

    @property
    def prior_beta(self) -> float:
        """Beta prior beta from observed data."""
        return max(1.0, self.n - self.k)


@dataclass
class BinaryEdge:
    """
    A single edge between two nodes — modelled as Beta(alpha, beta).
    Used for linear funnel steps with one outgoing edge.
    """
    edge_id: str
    source: str
    target: str
    evidence: Optional[EdgeEvidence] = None
    # Fallback priors for cold-start (Level 0 degradation)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0


@dataclass
class BranchGroup:
    """
    A set of sibling edges sharing a source node — modelled as Dirichlet.
    Enforces simplex constraint (probabilities sum to 1).
    Mirrors project-bayes §per-slice Dirichlet design note.
    """
    group_id: str
    source: str
    variants: list[str]             # target node names, e.g. ["classic", "quick"]
    evidence: list[EdgeEvidence]    # one per variant, same order as variants
    # Dirichlet concentration hyperprior — shared across siblings
    concentration: float = 1.0


@dataclass
class LatencyEdge:
    """
    An edge with meaningful conversion lag — modelled as LogNormal(mu, sigma).
    Uses fitted parameters from lag_distribution_utils.py pattern.
    """
    edge_id: str
    source: str
    target: str
    mu: float       # lognormal μ (log-scale mean)
    sigma: float    # lognormal σ (log-scale std, inflated for t95 tail constraint)
    t95: int = 30
    evidence: Optional[EdgeEvidence] = None


@dataclass
class HierarchyIR:
    """
    Complete intermediate representation of a DAG's statistical structure.
    Runtime-agnostic: can be consumed by PyMC, Stan, or any other backend.

    Provenance flags enable graceful degradation per project-bayes design:
      Level 0 = cold start (no evidence)
      Level 1 = weak evidence (immature cohorts)
      Level 2 = snapshot panel calibrated
      Level 3 = full Bayes posterior
    """
    graph_id: str
    binary_edges: list[BinaryEdge] = field(default_factory=list)
    branch_groups: list[BranchGroup] = field(default_factory=list)
    latency_edges: list[LatencyEdge] = field(default_factory=list)
    # Provenance / degradation level per group (group_id → level 0-3)
    evidence_levels: dict[str, int] = field(default_factory=dict)
    fingerprint: Optional[str] = None  # deterministic hash for reproducibility

    def summary(self) -> str:
        lines = [
            f"HierarchyIR: {self.graph_id}",
            f"  binary_edges:  {len(self.binary_edges)}",
            f"  branch_groups: {len(self.branch_groups)} "
            f"({sum(len(b.variants) for b in self.branch_groups)} total variants)",
            f"  latency_edges: {len(self.latency_edges)}",
            f"  evidence_levels: {self.evidence_levels}",
            f"  fingerprint: {self.fingerprint}",
        ]
        return "\n".join(lines)
