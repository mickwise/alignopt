"""
Purpose
-------
Unit tests for the JAX backend Direct Preference Optimization (DPO) loss module.

These tests validate that dpo_loss(...) and its internal helpers correctly:
1) compute policy and reference margins (chosen - rejected),
2) form the DPO logits beta * (pi_diff - reference_diff),
3) compute the mean batch loss as mean(-logsigmoid(dpo_logits)),
4) return a metrics dictionary whose values are JAX arrays (scalars),
5) keep metrics detached from gradients via stop_gradient,
6) remain compatible with jax.jit (no Python-side host transfers in the core path),
7) preserve expected shapes/dtypes across typical batch shapes.

Key behaviors
-------------
- Tests are deterministic and do not require model code (operates on precomputed log-prob tensors).
- Metrics are returned as scalar JAX arrays, suitable for logging after host transfer by the caller.
- Includes optional device coverage for Apple Silicon GPU (Metal / "METAL") when available.

Conventions
-----------
- Inputs are assumed to be per-sample log-probability sums (e.g., summed completion log-probs).
- Shapes are treated as opaque but must be broadcast-compatible for subtraction; tests use equal shapes.
- No validation is performed inside the module under test; tests focus on math correctness and jittability.

Downstream usage
----------------
Run with:
    pytest -q

Adjust the import path for dpo_loss, _dpo_loss_core_jax, and _calculate_metrics
to match your project structure.
"""

from __future__ import annotations

from typing import Tuple, Optional

import pytest
import jax
import jax.numpy as jnp

from alignopt.backends.jax.dpo.dpo_loss_jax import dpo_loss_jax, _dpo_loss_core_jax, _calculate_metrics


def _metal_device_if_available() -> jax.Device | None: # type: ignore
    """
    Helper to locate an Apple Silicon Metal device if available.

    Returns
    -------
    jax.Device | None
        A JAX device whose platform is "METAL" when present, else None.

    Notes
    -----
    - On macOS Apple Silicon with JAX Metal support installed, platform is typically "METAL".
    - If not present, tests that require non-CPU devices should be skipped.
    """
    for dev in jax.devices():
        if dev.platform.upper() == "METAL":
            return dev
    return None


def _make_inputs(
    shape: Tuple[int, ...] = (4,),
    beta: float = 0.1,
    *,
    device: Optional[jax.Device] = None, # type: ignore
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
    """
    Construct simple synthetic inputs for DPO tests.

    Parameters
    ----------
    shape : Tuple[int, ...], default=(4,)
        Shape of the synthetic per-sample tensors.
    beta : float, default=0.1
        Beta scaling for DPO logits.
    device : jax.Device | None, default=None
        If provided, places arrays on the specified device.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]
        (pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta).

    Notes
    -----
    - Values are chosen so expected margins and logits are easy to compute by hand.
    """
    pi_chosen = jnp.ones(shape, dtype=jnp.float32) * 2.0
    pi_rejected = jnp.ones(shape, dtype=jnp.float32) * 1.0
    ref_chosen = jnp.ones(shape, dtype=jnp.float32) * 0.5
    ref_rejected = jnp.ones(shape, dtype=jnp.float32) * 0.25

    if device is not None:
        pi_chosen = jax.device_put(pi_chosen, device)
        pi_rejected = jax.device_put(pi_rejected, device)
        ref_chosen = jax.device_put(ref_chosen, device)
        ref_rejected = jax.device_put(ref_rejected, device)

    return pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta


def test_dpo_loss_core_computes_expected_margins_and_logits() -> None:
    """
    Validate _dpo_loss_core_jax(...) returns correct intermediate tensors.

    Notes
    -----
    - pi_diff should equal (2.0 - 1.0) = 1.0 everywhere.
    - reference_diff should equal (0.5 - 0.25) = 0.25 everywhere.
    - dpo_logits should equal beta * (1.0 - 0.25) = 0.075 everywhere when beta=0.1.
    """
    pi_c, pi_r, ref_c, ref_r, beta = _make_inputs(shape=(5,), beta=0.1)

    pi_diff, ref_diff, logits = _dpo_loss_core_jax(pi_c, pi_r, ref_c, ref_r, beta)

    assert pi_diff.shape == (5,)
    assert ref_diff.shape == (5,)
    assert logits.shape == (5,)

    assert jnp.allclose(pi_diff, jnp.ones((5,), dtype=jnp.float32) * 1.0)
    assert jnp.allclose(ref_diff, jnp.ones((5,), dtype=jnp.float32) * 0.25)
    assert jnp.allclose(logits, jnp.ones((5,), dtype=jnp.float32) * (beta * 0.75))


def test_dpo_loss_jax_matches_closed_form_reference() -> None:
    """
    Validate dpo_loss_jax(...) matches a direct reference computation.

    Notes
    -----
    - Reference implementation:
        pi_diff = pi_chosen - pi_rejected
        ref_diff = ref_chosen - ref_rejected
        logits = beta * (pi_diff - ref_diff)
        loss = mean(-log_sigmoid(logits))
    """
    pi_c, pi_r, ref_c, ref_r, beta = _make_inputs(shape=(7,), beta=0.2)

    loss, metrics = dpo_loss_jax(pi_c, pi_r, ref_c, ref_r, beta)

    pi_diff = pi_c - pi_r
    ref_diff = ref_c - ref_r
    logits = beta * (pi_diff - ref_diff)
    expected_loss = jnp.mean(-jax.nn.log_sigmoid(logits))

    assert isinstance(metrics, dict)
    assert jnp.allclose(loss, expected_loss)

    # Metrics should be scalar arrays.
    assert metrics["log_diff_mean"].shape == ()
    assert metrics["pi_margin_mean"].shape == ()


def test_metrics_are_stop_gradient_detached() -> None:
    """
    Validate metrics do not contribute to gradients (stop_gradient semantics).

    Notes
    -----
    - We take grad of a scalar objective that includes loss + a metric.
    - Because metrics are stop_gradient, gradient should match grad(loss) alone.
    """
    pi_c, pi_r, ref_c, ref_r, beta = _make_inputs(shape=(4,), beta=0.3)

    def objective_with_metric(pi_chosen: jax.Array) -> jax.Array:
        loss, metrics = dpo_loss_jax(pi_chosen, pi_r, ref_c, ref_r, beta)
        return loss + metrics["log_diff_mean"]

    def objective_loss_only(pi_chosen: jax.Array) -> jax.Array:
        loss, _ = dpo_loss_jax(pi_chosen, pi_r, ref_c, ref_r, beta)
        return loss

    g_with = jax.grad(objective_with_metric)(pi_c)
    g_only = jax.grad(objective_loss_only)(pi_c)

    assert jnp.allclose(g_with, g_only)


def test_dpo_loss_jax_is_jittable_and_returns_expected_structure() -> None:
    """
    Validate dpo_loss(...) can be wrapped with jax.jit and returns expected outputs.

    Notes
    -----
    - JAX can return pytrees (dicts) from jitted functions.
    - Metrics remain JAX arrays; host conversion is caller responsibility.
    """
    pi_c, pi_r, ref_c, ref_r, beta = _make_inputs(shape=(8,), beta=0.15)

    jitted = jax.jit(dpo_loss_jax)
    loss, metrics = jitted(pi_c, pi_r, ref_c, ref_r, beta)

    assert isinstance(metrics, dict)
    assert "log_diff_mean" in metrics
    assert loss.shape == ()
    assert metrics["log_diff_std"].shape == ()
    assert isinstance(metrics["log_diff_std"], jax.Array)


def test_calculate_metrics_shapes_and_keys() -> None:
    """
    Validate _calculate_metrics(...) returns all expected keys with scalar shapes.

    Notes
    -----
    - This checks the logging contract for downstream training loops.
    """
    pi_c, pi_r, ref_c, ref_r, beta = _make_inputs(shape=(3, 2), beta=0.1)
    pi_diff, ref_diff, _ = _dpo_loss_core_jax(pi_c, pi_r, ref_c, ref_r, beta)

    metrics = _calculate_metrics(pi_diff, ref_diff)

    expected_keys = {
        "log_diff_mean",
        "log_diff_std",
        "log_diff_min",
        "log_diff_max",
        "log_diff_frac_pos",
        "pi_margin_mean",
        "reference_margin_mean",
    }
    assert set(metrics.keys()) == expected_keys
    for k, v in metrics.items():
        assert isinstance(v, jax.Array)
        assert v.shape == ()


@pytest.mark.skipif(_metal_device_if_available() is None, reason="No JAX METAL device available on this machine.")
def test_runs_on_metal_device_if_available() -> None:
    """
    Validate that the DPO loss runs on Apple Silicon METAL backend when available.

    Notes
    -----
    - This is not a device mismatch test (JAX generally expects inputs on the same device).
    - It checks that computation can be placed on METAL and produces finite outputs.
    """
    dev = _metal_device_if_available()
    assert dev is not None

    pi_c, pi_r, ref_c, ref_r, beta = _make_inputs(shape=(16,), beta=0.05, device=dev)
    loss, metrics = dpo_loss_jax(pi_c, pi_r, ref_c, ref_r, beta)

    assert loss.device == dev
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["log_diff_mean"])
