"""
Tests for the Dagnet Bayesian Analysis pipeline.
"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pymc as pm

from compiler.hierarchy_ir import EdgeEvidence
from compiler.compiler import compile_graph, _completeness_weight, _t95_constrained_sigma
from inference.model_builder import build_model
from evidence.snapshot_reader import evidence_from_yaml, _fit_lognormal


# ---------------------------------------------------------------------------
# Evidence reader
# ---------------------------------------------------------------------------

def test_evidence_from_yaml_basic():
    ev = evidence_from_yaml({"edge_id": "e1", "n": 1000, "k": 200})
    assert ev.edge_id == "e1"
    assert ev.n == 1000
    assert ev.k == 200
    assert ev.prior_alpha == 200
    assert ev.prior_beta == 800


def test_evidence_lognormal_fit():
    mu, sigma = _fit_lognormal({"lag_mean_days": 4.2, "lag_median_days": 2.1})
    assert mu == pytest.approx(math.log(2.1), rel=1e-3)
    assert sigma > 0


def test_evidence_completeness_weight():
    # At n=50 (= BLEND_K), weight should be ~0.632
    w = _completeness_weight(50)
    assert 0.62 < w < 0.64

    # Cold start: n=0 → weight=0
    assert _completeness_weight(0) == pytest.approx(0.0)

    # Mature: n=500 → weight close to 1
    assert _completeness_weight(500) > 0.99


# ---------------------------------------------------------------------------
# Compiler — linear funnel
# ---------------------------------------------------------------------------

LINEAR_GRAPH = {
    "id": "linear-test",
    "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    "edges": [
        {"id": "e_ab", "source": "a", "target": "b"},
        {"id": "e_bc", "source": "b", "target": "c"},
    ],
}


def test_compiler_linear_no_evidence():
    ir = compile_graph(LINEAR_GRAPH)
    assert ir.graph_id == "linear-test"
    assert len(ir.binary_edges) == 2
    assert len(ir.branch_groups) == 0
    assert ir.fingerprint is not None
    # Cold start: all edges at level 0
    for edge_id, level in ir.evidence_levels.items():
        assert level == 0


def test_compiler_linear_with_evidence():
    ev_store = {
        "e_ab": EdgeEvidence(edge_id="e_ab", n=1000, k=400),
        "e_bc": EdgeEvidence(edge_id="e_bc", n=400, k=100),
    }
    ir = compile_graph(LINEAR_GRAPH, ev_store)
    # Mature evidence → level 2
    assert ir.evidence_levels["e_ab"] == 2
    assert ir.evidence_levels["e_bc"] == 2


def test_compiler_fingerprint_deterministic():
    ir1 = compile_graph(LINEAR_GRAPH)
    ir2 = compile_graph(LINEAR_GRAPH)
    assert ir1.fingerprint == ir2.fingerprint


# ---------------------------------------------------------------------------
# Compiler — branch funnel
# ---------------------------------------------------------------------------

BRANCH_GRAPH = {
    "id": "branch-test",
    "nodes": [{"id": "cart"}, {"id": "classic"}, {"id": "quick"}, {"id": "purchase"}],
    "edges": [
        {"id": "e_cart_classic", "source": "cart",    "target": "classic"},
        {"id": "e_cart_quick",   "source": "cart",    "target": "quick"},
        {"id": "e_classic_pur",  "source": "classic", "target": "purchase"},
        {"id": "e_quick_pur",    "source": "quick",   "target": "purchase"},
    ],
}


def test_compiler_branch_group_created():
    ir = compile_graph(BRANCH_GRAPH)
    assert len(ir.branch_groups) == 1
    bg = ir.branch_groups[0]
    assert set(bg.variants) == {"classic", "quick"}
    assert len(bg.evidence) == 2


def test_compiler_branch_simplex_structure():
    """Dirichlet requires len(variants) == len(evidence)."""
    ir = compile_graph(BRANCH_GRAPH)
    for bg in ir.branch_groups:
        assert len(bg.variants) == len(bg.evidence)


# ---------------------------------------------------------------------------
# t95 tail constraint
# ---------------------------------------------------------------------------

def test_t95_sigma_inflated():
    """sigma should never be zero or negative."""
    sigma = _t95_constrained_sigma(mu=1.0, sigma=0.3, t95=30)
    assert sigma > 0


def test_t95_sigma_handles_edge_cases():
    # mu=0 edge case
    sigma = _t95_constrained_sigma(mu=0.0, sigma=None, t95=30)
    assert sigma > 0


# ---------------------------------------------------------------------------
# PyMC model builder
# ---------------------------------------------------------------------------

def test_model_builds_for_linear_graph():
    ir = compile_graph(LINEAR_GRAPH)
    model = build_model(ir)
    assert isinstance(model, pm.Model)
    # Should have Beta variables for each edge
    var_names = [v.name for v in model.free_RVs]
    assert any("p_e_ab" in n for n in var_names)
    assert any("p_e_bc" in n for n in var_names)


def test_model_builds_for_branch_graph():
    ir = compile_graph(BRANCH_GRAPH)
    model = build_model(ir)
    assert isinstance(model, pm.Model)
    var_names = [v.name for v in model.free_RVs]
    # Should have a Dirichlet for the branch group
    assert any("w_branch_cart" in n for n in var_names)


def test_model_with_evidence_has_likelihoods():
    ev_store = {
        "e_ab": EdgeEvidence(edge_id="e_ab", n=500, k=200),
    }
    ir = compile_graph(LINEAR_GRAPH, ev_store)
    model = build_model(ir)
    # Observed RVs = likelihoods
    observed_names = [v.name for v in model.observed_RVs]
    assert any("obs_e_ab" in n for n in observed_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
