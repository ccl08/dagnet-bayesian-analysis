# Dagnet — Bayesian A/B Analysis Layer

A technical assessment and implementation proposal for adding Bayesian experimentation to [Dagnet](https://github.com/gjbm2/dagnet), an open-source DAG editor and analytics engine for conversion modelling and optimisation.

---

## What This Is

This repo documents a **read-only technical due diligence exercise** followed by a **targeted implementation proposal** for extending Dagnet's existing A/B testing infrastructure with Bayesian posterior analysis.

It was produced as part of a structured analytical process:

1. Live app inspection (`dagnet-nine.vercel.app`)
2. GitHub codebase analysis via Claude Code
3. Python backend deep-dive (architecture, statistical methods, dependency constraints)
4. Bayesian model design grounded in the existing code structure
5. Precise implementation brief ready for execution

---

## The Problem Being Solved

Dagnet already has a sophisticated A/B test modelling system — `CaseNode` types, variant routing edges, `conditional_p` per-variant conversion probabilities, and a What-If DSL for simulating 100% rollouts. The infrastructure is real and well-tested across ~34 production files.

What's missing is the **statistical layer on top**: given observed conversion data for two variants, what is the probability that one is genuinely better than the other?

The Python backend (`lib/stats_enhancement.py`) contains a correct Beta-Binomial Bayesian posterior implementation (`_enhance_mcmc`), but it depends on `scipy` — which is excluded from production because Vercel's serverless functions have a 50MB size limit. As a result, the Bayesian methods silently 500 in production.

**The fix is not architectural. It's one dependency constraint and one missing comparison function.**

---

## Key Findings from the Codebase Analysis

| Area | Finding |
|---|---|
| **A/B infrastructure** | Complete — `CaseNode`, variant weights, `conditional_p`, What-If DSL, integrity checks |
| **Prior knowledge store** | `param-registry` YAML files contain `mean`, `stdev`, `n`, `k`, `distribution: beta` per parameter — ready-made Beta priors |
| **Python stats backend** | FastAPI + pure stdlib math in production; numpy/scipy excluded due to Vercel 50MB limit |
| **Bayesian code** | Exists in `stats_enhancement.py` but dead in production due to scipy import failure |
| **Database** | IndexedDB (Dexie) client-side — no server-side DB for the graph editor; separate Postgres for snapshots |
| **Session identity** | None in the graph editor — by design (privacy-conscious, operator tool not end-user tracker) |
| **What's needed** | Rewrite the Beta-Binomial posterior in stdlib `math` + add a variant comparison function |

---

## Why Bayesian Over Frequentist Here

Dagnet's use context makes Bayesian testing a strong fit:

- **Small daily sample sizes** — operator-level conversion data, not consumer-scale volume
- **Informative priors available** — the param-registry already stores historical Beta parameters (mean, n, k) per funnel step, enabling informed rather than uniform priors
- **Sequential decision-making** — the What-If DSL and scenario system are built for ongoing updating, not fixed-horizon analysis
- **Actionable probability statements** — "P(quick cart > classic cart) = 0.94" is more useful to an operator than p < 0.05 with no effect size

---

## Proposed Implementation

### The Core Fix — Beta-Binomial Posterior in Stdlib

Replace the scipy dependency in `lib/stats_enhancement.py` with pure Python:
```python
import math

def _beta_credible_interval(alpha: float, beta: float) -> tuple[float, float]:
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    z = 1.96
    std = math.sqrt(variance)
    return (max(0.0, mean - z * std), min(1.0, mean + z * std))

def enhance_mcmc_stdlib(n: int, k: int) -> dict:
    alpha = k + 1
    beta = n - k + 1
    mean = alpha / (alpha + beta)
    lo, hi = _beta_credible_interval(alpha, beta)
    return {"mean": mean, "ci_low": lo, "ci_high": hi,
            "alpha": alpha, "beta": beta, "method": "bayesian_beta_binomial"}
```

### The New Analysis Type — Variant Comparison
```python
import random

def p_b_greater_than_a(alpha_a, beta_a, alpha_b, beta_b, n_samples=10_000) -> float:
    wins = sum(
        random.betavariate(alpha_b, beta_b) > random.betavariate(alpha_a, beta_a)
        for _ in range(n_samples)
    )
    return wins / n_samples
```

### Prior Elicitation from param-registry
```python
# landing-to-product.yaml: mean=0.45, n=10000, k=4500
# Informative prior: Beta(4500, 5500)
def prior_from_param_registry(mean: float, n: int) -> tuple:
    k = round(mean * n)
    return float(k), float(n - k)
```

### Response Shape
```json
{
  "variant_a": {"name": "classic", "n": 1000, "k": 720,
                "posterior_mean": 0.721, "ci_low": 0.694, "ci_high": 0.748},
  "variant_b": {"name": "quick", "n": 1000, "k": 820,
                "posterior_mean": 0.821, "ci_low": 0.797, "ci_high": 0.845},
  "p_b_greater_than_a": 0.9997,
  "expected_loss_choosing_a": 0.101,
  "recommendation": "ship_b",
  "stopping_criterion_met": true
}
```

---

## Files in This Repo

---

## Assessment Methodology

Every finding in the due diligence is labelled as one of:

- ✅ **Confirmed** — directly observed in code or live app
- 🟡 **Strong inference** — well-supported by structural evidence
- 🟠 **Weak inference** — plausible but unverified
- ❓ **Unknown** — explicitly flagged, not assumed

This prevented the assessment from proposing solutions to problems the codebase had already solved (the A/B infrastructure), and focused effort on the actual gap (the scipy constraint blocking production Bayesian computation).

---

## Tools Used

- **Claude Code** — codebase analysis (read-only, no production changes)
- **Claude.ai** — assessment synthesis, implementation design
- **GitHub** — source repository (`gjbm2/dagnet`, public)

---

## Context

Produced as part of a technical assessment exercise. The Dagnet repository is public on GitHub. No production systems were modified, no credentials were accessed, and no proprietary data was used.
