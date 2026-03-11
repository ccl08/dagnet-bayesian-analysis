"""
Pydantic request/response models for Bayesian A/B comparison.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class VariantInput(BaseModel):
    name: str
    n: int = Field(..., description="Total number of trials")
    k: int = Field(..., description="Number of conversions")


class BayesianABRequest(BaseModel):
    variant_a: VariantInput
    variant_b: VariantInput
    prior_alpha: float = Field(1.0, description="Prior alpha (from param-registry mean*n)")
    prior_beta: float = Field(1.0, description="Prior beta (from param-registry (1-mean)*n)")
    n_samples: int = Field(10000, description="Monte Carlo samples for P(B>A)")


class VariantResult(BaseModel):
    name: str
    n: int
    k: int
    posterior_mean: float
    ci_low: float
    ci_high: float
    alpha: float
    beta: float


class BayesianABResponse(BaseModel):
    variant_a: VariantResult
    variant_b: VariantResult
    p_b_greater_than_a: float
    expected_loss_choosing_a: float
    recommendation: Literal["ship_b", "ship_a", "continue"]
    stopping_criterion_met: bool
