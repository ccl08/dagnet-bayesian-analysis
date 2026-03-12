# Dagnet Bayesian Analysis

A Bayesian inference layer for [Dagnet](https://github.com/gjbm2/dagnet) — a DAG-based analytics engine for conversion modelling and optimisation.

This project implements the statistical architecture described in Dagnet's `docs/current/project-bayes/` design documents: a runtime compiler that translates DAG topology into a correctly-structured PyMC model, runs MCMC sampling, and persists posterior artefacts back to the snapshot DB.

---

## Architecture

```
Snapshot DB (evidence: n, k, lag distributions per slice)
        ↓
graph-to-hierarchy compiler  ← converts DAG topology to model structure
        ↓
PyMC model built at runtime:
  - pm.Dirichlet per branch group (simplex constraint)
  - pm.Beta per single binary edge
  - pm.LogNormal per latency edge (uses fitted μ, σ from lag model)
  - Cohort completeness correction (mature vs immature)
        ↓
pm.sample() — MCMC on dedicated compute (not Vercel)
        ↓
Posterior artefacts persisted back to snapshot DB
        ↓
Frontend reads artefacts — no compute needed
```

The key design insight: the **compiler** and the **inference engine** are fully separated. The compiler emits a `HierarchyIR` — a runtime-agnostic intermediate representation — that the model builder then translates into PyMC. This means the inference backend can be swapped (Stan, NumPyro) without touching the compiler.

---

## Project Structure

```
compiler/
  hierarchy_ir.py       — HierarchyIR dataclass (EdgeEvidence, BinaryEdge,
                          BranchGroup, LatencyEdge)
  compiler.py           — DAG JSON → HierarchyIR (6 of 8 compiler steps)

inference/
  model_builder.py      — HierarchyIR → PyMC model
  runner.py             — pm.sample() + posterior artefact extraction

evidence/
  snapshot_reader.py    — YAML param files / snapshot DB rows → EdgeEvidence

tests/
  test_pipeline.py      — 13 tests covering compiler, model builder, evidence

examples/
  ecommerce_checkout.py — End-to-end demo on dagnet's own checkout fixture
```

---

## Key Design Decisions

### Dirichlet for branching nodes, not independent Betas

When a DAG node has multiple outgoing edges (e.g. a cart routing to `classic` and `quick` variants), those edge weights must sum to 1. Independent Beta priors violate this simplex constraint. The compiler identifies these **branch groups** and the model builder assigns a shared `pm.Dirichlet` with a concentration hyperprior:

```python
weights = pm.Dirichlet(f"w_{group.group_id}", a=concentration * ones(k))
```

This matches the design note in `project-bayes/0-high-level-logical-blocks.md`:
> *"naive logit-space deviations violate simplex constraint; correct approach is hierarchical Dirichlet per slice with shared concentration hyperpriors"*

### t95 tail constraint on latency edges

The lognormal sigma for each latency edge is inflated to prevent thin-tail optimism — ensuring P(lag ≤ t95) ≥ 0.95. Mirrors `t95-fix.md`:
> *"t95 tail constraint inflates sigma to prevent thin-tail optimism"*

### Cohort completeness blend weight

Evidence from immature cohorts (still accumulating conversions) is downweighted using the soft transition from `window-cohort-lag-correction-plan.md`:

```
w = 1 - exp(-n / BLEND_K)    where BLEND_K = 50
```

### Graceful degradation ladder

Each edge gets an evidence level (0–3) based on the `cohort-completeness-model-contract.md` design:

| Level | Condition | Prior |
|-------|-----------|-------|
| 0 | Cold start (no evidence) | Beta(1, 1) — uniform |
| 1 | Weak / immature cohort | Beta from observed n, k, downweighted |
| 2 | Snapshot DB calibrated | Beta(k, n-k) as informative prior |
| 3 | Full Bayes posterior | Set by runner after sampling |

---

## Demo Output

Running `python -m examples.ecommerce_checkout` on the ecommerce checkout fixture:

```
[1] Compiling graph topology → Hierarchy IR...
HierarchyIR: ecommerce-checkout
  binary_edges:  4
  branch_groups: 1 (2 total variants)
  latency_edges: 4
  evidence_levels: all edges at level 2
  fingerprint: b38eaad6c1867e70

[3] Posterior artefacts:
  edge e_land_prod:  mean=0.640  HDI=[0.636, 0.644]  (level 2)
  edge e_prod_cart:  mean=0.578  HDI=[0.573, 0.583]  (level 2)
  edge e_classic_pur: mean=0.879  HDI=[0.871, 0.886]  (level 2)
  edge e_quick_pur:  mean=0.908  HDI=[0.902, 0.914]  (level 2)
  branch_cart/classic: mean=0.465  P(best)=0.0%
  branch_cart/quick:   mean=0.535  P(best)=100.0%
  diagnostics: max_rhat=1.009  converged=True

[4] Cart A/B Decision:
  → 'quick' is most likely best (P=100.0%)
  → Estimated uplift (quick vs classic): +7.0%
  → Post-it estimate was +8pp — model estimate: +7.0%
```

The fixture's post-it note ("Quick cart showing +8pp uplift") is confirmed by the model at +7pp, with tight HDIs and R-hat < 1.01 indicating clean convergence.

---

## What's deferred (production gaps)

- **`path_t95` DP call sites** — the DP code exists in dagnet but has no call sites; the compiler needs it to bound retrieval horizons correctly at path level
- **`WindowFetchPlannerService`** — designed (61KB spec) but not yet built; determines `covered_stable / not_covered / covered_stale` before evidence is bound
- **`t95-fix` Phase 2** — schema migration, constants consolidation, override semantics (8 phases, all unchecked)
- **Full 8-step compiler** — steps 3–5 (latency hierarchy, probability-latency coupling, exhaustiveness policy) are partially stubbed
- **Nightly orchestration** — scheduler, subject discovery, artefact persistence to Postgres

---

## Running

```bash
pip install pymc arviz

# Tests
python -m pytest tests/ -v

# End-to-end demo
python -m examples.ecommerce_checkout
```

---

## Prior art

| Library | Approach | Why not used directly |
|---------|----------|-----------------------|
| PyMC 5 | NUTS/MCMC, full posterior | Used here as inference backend |
| statsmodels | Frequentist, no uncertainty propagation | No posterior distributions |
| scipy.stats | Beta-Binomial analytically | Too large for Vercel (50MB limit) |
| Stan | Similar to PyMC, faster sampling | Less Python-native, harder to build dynamically |

The core problem scipy doesn't solve: **the model structure itself must be compiled from the graph topology at runtime**. PyMC's Python-native model construction makes this tractable.
