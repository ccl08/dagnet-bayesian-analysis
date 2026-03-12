"""
End-to-end example: ecommerce checkout funnel with A/B cart variants.

Mirrors the ecommerce-checkout-flow.json fixture in dagnet:
  landing → product → cart → [classic | quick] → purchase

Post-it note in the fixture: "Quick cart showing +8pp uplift. Decision due end of Q1."

This example demonstrates the full pipeline:
  evidence → compiler → PyMC model → posterior artefacts

Run with:
  python -m examples.ecommerce_checkout
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evidence.snapshot_reader import evidence_from_yaml
from compiler.compiler import compile_graph
from inference.runner import run_inference


# --- Graph definition (mirrors ecommerce-checkout-flow.json topology) ---
GRAPH = {
    "id": "ecommerce-checkout",
    "nodes": [
        {"id": "landing",  "label": "Landing"},
        {"id": "product",  "label": "Product"},
        {"id": "cart",     "label": "Cart"},
        {"id": "classic",  "label": "Classic Cart"},
        {"id": "quick",    "label": "Quick Cart"},
        {"id": "purchase", "label": "Purchase"},
    ],
    "edges": [
        {"id": "e_land_prod",    "source": "landing",  "target": "product"},
        {"id": "e_prod_cart",    "source": "product",  "target": "cart"},
        # Branch: cart routes to either classic or quick variant
        {"id": "e_cart_classic", "source": "cart",     "target": "classic"},
        {"id": "e_cart_quick",   "source": "cart",     "target": "quick"},
        # Both variants converge to purchase
        {"id": "e_classic_pur",  "source": "classic",  "target": "purchase"},
        {"id": "e_quick_pur",    "source": "quick",    "target": "purchase"},
    ],
}

# --- Evidence (from snapshot DB / param registry) ---
# Mirrors ecommerce-checkout-flow.json values:
#   classic: 0.72 conversion rate
#   quick:   0.82 conversion rate (+10pp, close to the +8pp "post-it" estimate)
EVIDENCE = {
    "e_land_prod": evidence_from_yaml({
        "edge_id": "e_land_prod",
        "n": 50000, "k": 32000,
        "latency_parameter": True,
        "lag_mean_days": 0.2, "lag_median_days": 0.1,
        "t95": 3,
    }),
    "e_prod_cart": evidence_from_yaml({
        "edge_id": "e_prod_cart",
        "n": 32000, "k": 18500,
        "latency_parameter": True,
        "lag_mean_days": 0.5, "lag_median_days": 0.3,
        "t95": 5,
    }),
    # A/B cart variants — the branch group
    "e_cart_classic": evidence_from_yaml({
        "edge_id": "e_cart_classic",
        "n": 9200, "k": 6600,   # ~72% conversion
    }),
    "e_cart_quick": evidence_from_yaml({
        "edge_id": "e_cart_quick",
        "n": 9300, "k": 7600,   # ~82% conversion
    }),
    # Post-purchase conversion (both variants)
    "e_classic_pur": evidence_from_yaml({
        "edge_id": "e_classic_pur",
        "n": 6600, "k": 5800,
        "latency_parameter": True,
        "lag_mean_days": 1.5, "lag_median_days": 0.8,
        "t95": 14,
    }),
    "e_quick_pur": evidence_from_yaml({
        "edge_id": "e_quick_pur",
        "n": 7600, "k": 6900,
        "latency_parameter": True,
        "lag_mean_days": 1.2, "lag_median_days": 0.6,
        "t95": 14,
    }),
}


def main():
    print("=" * 60)
    print("Dagnet Bayesian Analysis — Ecommerce Checkout Example")
    print("=" * 60)

    # Step 1: Compile graph → Hierarchy IR
    print("\n[1] Compiling graph topology → Hierarchy IR...")
    ir = compile_graph(GRAPH, EVIDENCE)
    print(ir.summary())

    # Step 2: Run inference
    print("\n[2] Running PyMC inference (this may take ~30s)...")
    result = run_inference(
        ir,
        draws=500,
        tune=500,
        chains=2,
        progressbar=True,
    )

    # Step 3: Print posterior artefacts
    print("\n[3] Posterior artefacts:")
    print(result.summary())

    # Step 4: Highlight the A/B decision
    print("\n[4] Cart A/B Decision:")
    for bp in result.branch_posteriors:
        if "cart" in bp.group_id.lower() or "branch_cart" in bp.group_id:
            for i, v in enumerate(bp.variants):
                print(f"  {v:10s}: mean={bp.means[i]:.3f}  "
                      f"HDI=[{bp.hdi_lows[i]:.3f}, {bp.hdi_highs[i]:.3f}]  "
                      f"P(best)={bp.win_probs[i]:.1%}")
            best_i = bp.win_probs.index(max(bp.win_probs))
            print(f"\n  → '{bp.variants[best_i]}' is most likely best "
                  f"(P={bp.win_probs[best_i]:.1%})")
            uplift = bp.means[1] - bp.means[0]
            print(f"  → Estimated uplift (quick vs classic): {uplift:+.1%}")
            print(f"  → Post-it estimate was +8pp — model estimate: {uplift:+.1%}")


if __name__ == "__main__":
    main()
