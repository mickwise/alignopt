"""
Purpose
-------
Lightweight input validation for the JAX DPO loss implementation.

Key behaviors
-------------
- Verifies that all four log-probability arrays share the same shape.
- Verifies that all four arrays are floating-point (required for DPO math).
- Verifies that beta is a finite, positive scalar.

Conventions
-----------
- This validator is intentionally *cheap* (Python-side checks only).
- It does not scan array contents for finiteness (no O(N) checks) and does not enforce device equality.
- Shape equality is treated as the core contract; downstream code may assume aligned batching.

Downstream usage
----------------
Call validate_dpo_data_jax(...) at the top of a JAX DPO-loss implementation before any arithmetic.
"""

from typing import Tuple, List
import jax
import jax.numpy as jnp
from math import isfinite
def validate_dpo_data_jax(
        pi_chosen: jax.Array,
        pi_rejected: jax.Array,
        reference_chosen: jax.Array,
        reference_rejected: jax.Array,
        beta: float
) -> None:
    """
    Validate the inputs for a JAX DPO loss computation.

    Parameters
    ----------
    pi_chosen : jax.Array
        Log-probabilities (or logit-derived scores) for the policy model on the chosen responses.
    pi_rejected : jax.Array
        Log-probabilities (or logit-derived scores) for the policy model on the rejected responses.
    reference_chosen : jax.Array
        Log-probabilities (or logit-derived scores) for the reference model on the chosen responses.
    reference_rejected : jax.Array
        Log-probabilities (or logit-derived scores) for the reference model on the rejected responses.
    beta : float
        Positive temperature / scaling parameter used by the DPO objective.

    Returns
    -------
    None
        This function returns None and raises on invalid inputs.

    Raises
    ------
    ValueError
        If beta is not finite and positive, if shapes do not match, or if any input is not floating-point.

    Notes
    -----
    - This function is intentionally minimal to avoid hidden performance costs in JAX.
    - It does not enforce device placement and does not check element-wise finiteness.
    """
 
    reference_shape: Tuple[int, ...] = pi_chosen.shape
    input_array_list: List[jax.Array] = [
        pi_chosen, pi_rejected, reference_chosen, reference_rejected
    ]
    if beta <= 0.0 or not isfinite(beta):
        raise ValueError("Beta must be a finite, positive float.")
    for arr in input_array_list:
        if arr.shape != reference_shape:
            raise ValueError("All log probability arrays must have the same shape.")
        if not jnp.issubdtype(arr.dtype, jnp.floating):
            raise ValueError("All input arrays must hold floats.")
