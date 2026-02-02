"""
Purpose
-------
Backend-specific Direct Preference Optimization (DPO) loss implementation for JAX.

This module provides a JAX-native implementation of the core DPO objective used in
preference learning pipelines. It assumes inputs are *already computed* log-probability
sums per sample (e.g., summed completion log-probs), and produces:
1) a scalar batch loss suitable for differentiation, and
2) a set of detached scalar diagnostics for logging (as scalar JAX arrays).

Key behaviors
-------------
- Computes per-sample margins:
  - pi_diff = pi_chosen - pi_rejected
  - reference_diff = reference_chosen - reference_rejected
- Computes DPO logits: beta * (pi_diff - reference_diff).
- Uses mean(-log_sigmoid(DPO logits)) as the batch loss.
- Returns a metrics dict of scalar JAX arrays detached via jax.lax.stop_gradient,
  keeping the loss path differentiable while avoiding gradient flow through logging values.
- Designed to be jittable: no Python-side host transfers (no .item(), no device_get).

Conventions
-----------
- All inputs must be floating-point arrays with matching shapes.
- Metrics are returned as scalar JAX arrays (shape ()), not Python floats.
  Converting to Python floats (e.g., for printing/logging) should be done by the caller
  outside any jitted region (e.g., float(jax.device_get(metric))).

Downstream usage
----------------
Call dpo_loss(...) inside a JAX training step after computing policy/reference
log-prob sums for chosen/rejected samples. Use the returned metrics dict for logging,
optionally converting the scalars to Python floats outside jit.
"""

from typing import Tuple, Dict
import jax
import jax.numpy as jnp

def dpo_loss_jax(
    pi_chosen: jax.Array,
    pi_rejected: jax.Array,
    reference_chosen: jax.Array,
    reference_rejected: jax.Array,
    beta: float = 0.1,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Compute the mean DPO loss for a batch and return detached diagnostic metrics.

    Parameters
    ----------
    pi_chosen : jax.Array
        Policy model log-probability sums for the chosen completions.
        Shape: (B, ...) where B is batch size; must match all other inputs.
    pi_rejected : jax.Array
        Policy model log-probability sums for the rejected completions.
        Same shape/dtype constraints as pi_chosen.
    reference_chosen : jax.Array
        Reference model log-probability sums for the chosen completions.
        Same shape/dtype constraints as pi_chosen.
    reference_rejected : jax.Array
        Reference model log-probability sums for the rejected completions.
        Same shape/dtype constraints as pi_chosen.
    beta : float, default=0.1
        Positive inverse-temperature scaling the DPO logits. Larger beta increases
        the sharpness of the preference optimization signal.

    Returns
    -------
    Tuple[jax.Array, Dict[str, jax.Array]]
        (mean_batch_loss, metrics) where:
        - mean_batch_loss is a scalar jax.Array suitable for differentiation.
        - metrics is a dict of scalar jax.Array values (shape ()) detached from gradients.

    Raises
    ------
    None
        This function performs no explicit input validation. Callers should validate inputs
        in non-jitted orchestration code if needed.

    Notes
    -----
    - The loss is computed as:
        pi_diff = pi_chosen - pi_rejected
        ref_diff = reference_chosen - reference_rejected
        dpo_logits = beta * (pi_diff - ref_diff)
        loss = mean( -log_sigmoid(dpo_logits) )
    - Metrics are detached with jax.lax.stop_gradient to prevent accidental gradient flow
      through logging computations.
    """

    pi_diff, reference_diff, dpo_core = _dpo_loss_core_jax(
        pi_chosen,
        pi_rejected,
        reference_chosen,
        reference_rejected,
        beta
    )
    mean_batch_loss: jax.Array = jnp.mean(-jax.nn.log_sigmoid(dpo_core))
    return mean_batch_loss, _calculate_metrics(pi_diff, reference_diff)


def _dpo_loss_core_jax(
    pi_chosen: jax.Array,
    pi_rejected: jax.Array,
    reference_chosen: jax.Array,
    reference_rejected: jax.Array,
    beta: float = 0.1
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute per-sample DPO intermediate arrays (policy margin, reference margin, DPO logits).

    Parameters
    ----------
    pi_chosen : jax.Array
        Policy log-probability sums for chosen completions. Shape (B, ...).
    pi_rejected : jax.Array
        Policy log-probability sums for rejected completions. Same shape as pi_chosen.
    reference_chosen : jax.Array
        Reference log-probability sums for chosen completions. Same shape as pi_chosen.
    reference_rejected : jax.Array
        Reference log-probability sums for rejected completions. Same shape as pi_chosen.
    beta : float, default=0.1
        Positive inverse-temperature scaling for the DPO logits.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        (pi_diff, reference_diff, dpo_logits) where:
        - pi_diff is the policy margin per element: pi_chosen - pi_rejected.
        - reference_diff is the reference margin per element: reference_chosen - reference_rejected.
        - dpo_logits = beta * (pi_diff - reference_diff).

    Raises
    ------
    None

    Notes
    -----
    - This helper is intentionally small and side-effect free to keep the main loss function
      readable and to make the math easy to audit.
    - It is suitable for use inside jitted code.
    """

    pi_diff: jax.Array = pi_chosen - pi_rejected
    reference_diff: jax.Array = reference_chosen - reference_rejected
    return pi_diff, reference_diff, beta*(pi_diff - reference_diff)


def _calculate_metrics(
        pi_diff: jax.Array,
        reference_diff: jax.Array,
    ) -> Dict[str, jax.Array]:
    """
    Compute detached scalar diagnostics from policy/reference margins for logging.

    Parameters
    ----------
    pi_diff : jax.Array
        Policy margin array = pi_chosen - pi_rejected. Shape (B, ...).
    reference_diff : jax.Array
        Reference margin array = reference_chosen - reference_rejected.
        Same shape as pi_diff.

    Returns
    -------
    Dict[str, jax.Array]
        Dictionary of scalar jax.Array values (shape ()) detached from gradients:
        - log_diff_mean: mean(pi_diff - reference_diff)
        - log_diff_std: std(pi_diff - reference_diff)
        - log_diff_min: min(pi_diff - reference_diff)
        - log_diff_max: max(pi_diff - reference_diff)
        - log_diff_frac_pos: mean((pi_diff - reference_diff) > 0)
        - pi_margin_mean: mean(pi_diff)
        - reference_margin_mean: mean(reference_diff)

    Raises
    ------
    None

    Notes
    -----
    - All returned values are detached via jax.lax.stop_gradient, so they are safe to compute
      within training code without influencing gradients.
    - Values are returned as JAX scalars to preserve jittability.
    """

    log_diff: jax.Array = pi_diff - reference_diff
    return {
        "log_diff_mean": jax.lax.stop_gradient(jnp.mean(log_diff)),
        "log_diff_std": jax.lax.stop_gradient(jnp.std(log_diff)),
        "log_diff_min": jax.lax.stop_gradient(jnp.min(log_diff)),
        "log_diff_max": jax.lax.stop_gradient(jnp.max(log_diff)),
        "log_diff_frac_pos": jax.lax.stop_gradient(jnp.mean(log_diff > 0)),
        "pi_margin_mean": jax.lax.stop_gradient(jnp.mean(pi_diff)),
        "reference_margin_mean": jax.lax.stop_gradient(jnp.mean(reference_diff))
    }
