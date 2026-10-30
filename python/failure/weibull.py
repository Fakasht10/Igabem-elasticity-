# Vendored from nurb-fea-tcd (github.com/Fakasht10/nurb-fea-tcd, MIT, Aleksey Kashtelyan), feakit/weibull.py at
# commit 553dd2a. See python/failure/LICENSE-nurb-fea-tcd. Local changes are listed in python/failure/README.md.
"""Stress-based weakest-link Weibull failure prediction for 3-D FEA fields.

Units are deliberately explicit: stress is MPa (= N/mm²), quadrature weights
and V0 are mm³, and reference/failure loads are N.  This module implements the
stress weakest-link volume law; it does not implement an energy Weibull model.
"""
from __future__ import annotations

import math
import sys
from typing import Mapping

import numpy as np

STRESS_WEAKEST_LINK_SOURCE: Mapping[str, str] = {
    "citation": "Fidan & Unal (2026), stress weakest-link volume law",
    "doi": "10.1007/s41779-026-01364-7",
    "stress_unit": "MPa (N/mm^2)",
    "volume_unit": "mm^3",
    "model": "positive maximum principal stress only; energy Weibull excluded",
}


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _weights(weights_mm3) -> np.ndarray:
    weights = np.asarray(weights_mm3, dtype=float).reshape(-1)
    if not len(weights) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("quadrature weights_mm3 must be finite and positive")
    return weights


def positive_max_principal_stress(stress_MPa) -> np.ndarray:
    """Return max(lambda_max(sigma), 0) for (..., 3, 3) symmetric Cauchy stress."""
    stress = np.asarray(stress_MPa, dtype=float)
    if stress.shape[-2:] != (3, 3) or not np.all(np.isfinite(stress)):
        raise ValueError("stress_MPa must be finite with trailing shape (3, 3)")
    # eigvalsh gives real eigenvalues for the physically relevant symmetric part;
    # symmetricizing makes round-off antisymmetry harmless without changing sigma.
    symmetric = 0.5 * (stress + np.swapaxes(stress, -1, -2))
    return np.maximum(np.linalg.eigvalsh(symmetric)[..., -1], 0.0)


def stress_power_integral(stress_MPa, weights_mm3, m: float, V0_mm3: float = 1.0) -> float:
    """Dimensionless J = sum(w*sigma_plus**m)/V0, accumulated in log-domain.

    A zero field returns exactly zero.  Log accumulation avoids avoidable
    overflow/underflow for calibrated shape parameters and large meshes.
    """
    log_J = log_stress_power_integral(stress_MPa, weights_mm3, m, V0_mm3)
    if log_J == -math.inf:
        return 0.0
    if log_J > math.log(sys.float_info.max):
        raise OverflowError("J is not representable; use log_stress_power_integral")
    return math.exp(log_J)


def log_stress_power_integral(stress_MPa, weights_mm3, m: float, V0_mm3: float = 1.0) -> float:
    """Return log(J), including ``-inf`` for an entirely compressive field.

    This is the stable production representation.  It permits failure-load
    predictions even when the dimensionless conventional ``J`` cannot fit in a
    float; use :func:`characteristic_failure_load_from_log_j` in that case.
    """
    m = _positive_finite(m, "m")
    V0_mm3 = _positive_finite(V0_mm3, "V0_mm3")
    splus = positive_max_principal_stress(stress_MPa).reshape(-1)
    weights = _weights(weights_mm3)
    if len(weights) != len(splus):
        raise ValueError("one quadrature weight_mm3 is required per stress tensor")
    positive = splus > 0.0
    if not np.any(positive):
        return -math.inf
    logs = np.log(weights[positive]) + m * np.log(splus[positive])
    a = float(np.max(logs))
    return a + math.log(float(np.exp(logs - a).sum())) - math.log(V0_mm3)


def characteristic_failure_load(J: float, sigma0_MPa: float, m: float, reference_load_N: float) -> float:
    """63.2%-failure load eta under linear scaling from reference_load_N."""
    J = float(J)
    if not math.isfinite(J) or J <= 0.0:
        raise ValueError("J must be finite and positive (the tensile integration volume may be zero)")
    sigma0_MPa = _positive_finite(sigma0_MPa, "sigma0_MPa")
    m = _positive_finite(m, "m")
    reference_load_N = _positive_finite(reference_load_N, "reference_load_N")
    return characteristic_failure_load_from_log_j(math.log(J), sigma0_MPa, m, reference_load_N)


def characteristic_failure_load_from_log_j(log_J: float, sigma0_MPa: float, m: float, reference_load_N: float) -> float:
    """63.2%-failure load from a finite log-domain stress-power integral."""
    log_J = float(log_J)
    if not math.isfinite(log_J):
        raise ValueError("log_J must be finite (the tensile integration volume may be zero)")
    sigma0_MPa = _positive_finite(sigma0_MPa, "sigma0_MPa")
    m = _positive_finite(m, "m")
    reference_load_N = _positive_finite(reference_load_N, "reference_load_N")
    log_eta = math.log(reference_load_N) + math.log(sigma0_MPa) - log_J / m
    eta = math.exp(log_eta)
    if not math.isfinite(eta):
        raise ValueError("characteristic failure load is not finite")
    return eta


def mean_failure_load(J: float, sigma0_MPa: float, m: float, reference_load_N: float) -> float:
    """Mean load E[L] = eta Gamma(1 + 1/m)."""
    m = _positive_finite(m, "m")
    return characteristic_failure_load(J, sigma0_MPa, m, reference_load_N) * math.gamma(1.0 + 1.0 / m)


def failure_load_quantile(J: float, sigma0_MPa: float, m: float, reference_load_N: float, probability: float) -> float:
    """Load quantile at CDF probability strictly between zero and one."""
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    m = _positive_finite(m, "m")
    eta = characteristic_failure_load(J, sigma0_MPa, m, reference_load_N)
    return eta * (-math.log1p(-probability)) ** (1.0 / m)


def failure_cdf(load_N: float, J: float, sigma0_MPa: float, m: float, reference_load_N: float) -> float:
    """Weibull CDF at a non-negative applied load."""
    load_N = float(load_N)
    if not math.isfinite(load_N) or load_N < 0.0:
        raise ValueError("load_N must be finite and non-negative")
    eta = characteristic_failure_load(J, sigma0_MPa, m, reference_load_N)
    return -math.expm1(-((load_N / eta) ** _positive_finite(m, "m")))


def hazard_at_load(applied_load_N: float, J: float, sigma0_MPa: float, m: float, reference_load_N: float) -> float:
    """Cumulative hazard H=(L/eta)^m at an applied load."""
    applied_load_N = float(applied_load_N)
    if not math.isfinite(applied_load_N) or applied_load_N < 0.0:
        raise ValueError("applied_load_N must be finite and non-negative")
    eta = characteristic_failure_load(J, sigma0_MPa, m, reference_load_N)
    return (applied_load_N / eta) ** _positive_finite(m, "m")


def edge_convergence(decay: float, m: float) -> dict:
    """Does the Weibull integral converge at a sharp re-entrant edge?

    Beside the edge sigma ~ r^-decay (Williams; decay = 1 - lambda_1, 0.4878 at a
    60 deg V), and the volume element is r dr dtheta dz, so the integrand of
    int sigma^m dV goes as r^(1 - m*decay): finite only for m*decay < 2, i.e.
    m < 2/decay (4.10 at a 60 deg V). Above that, a mesh that resolves the edge
    down to h finds J ~ h^-(m*decay - 2) and the failure load, ~ J^(-1/m),
    falls by 2^((m*decay - 2)/m) - 1 per halving of h, without a limit
    (6.3 % per halving for m = 5 at 60 deg).
    """
    decay, m = float(decay), float(m)
    m_crit = 2.0 / decay if decay > 0 else math.inf
    diverges = m >= m_crit
    return {"decay": decay, "m": m, "m_crit": m_crit, "diverges": bool(diverges),
            "drop_per_halving": (2.0 ** ((m * decay - 2.0) / m) - 1.0) if diverges else 0.0}
