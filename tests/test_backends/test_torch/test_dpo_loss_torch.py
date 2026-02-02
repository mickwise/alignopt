"""
Purpose
-------
Unit tests for the Torch DPO loss backend.

These tests validate that dpo_loss_torch(...) correctly:
1) validates inputs via validate_dpo_data_torch(...),
2) computes DPO core margins and the per-batch loss using -logsigmoid(...).mean(),
3) returns a scalar torch.Tensor loss suitable for backprop,
4) returns a metrics dict of Python floats consistent with the computed margins,
5) behaves deterministically for fixed inputs.

Key behaviors
-------------
- Uses small, explicit tensors with hand-computable expected values.
- Covers both vector-shaped inputs (B,) and matrix-shaped inputs (B, T).
- Exercises validation failures: shape mismatch, non-finite values, non-float dtype, invalid beta.
- Optionally checks device mismatch when a CUDA device is available.

Conventions
-----------
- Inputs are assumed to be log-probabilities.
- Metrics are computed elementwise on (pi_chosen - pi_rejected) and (ref_chosen - ref_rejected),
  and then reduced with mean/std/min/max over all elements.
- log_diff_frac_pos is the fraction of elements where (pi_diff - ref_diff) > 0.

Downstream usage
----------------
Run with:
    pytest -q

Adjust the import path for the module under test if your package layout differs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch


from alignopt.backends.torch.dpo.dpo_loss_torch import dpo_loss_torch


def _expected_loss_and_metrics(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute expected loss and metrics using the same mathematical definition as the implementation.

    Notes
    -----
    - This helper mirrors the specification:
        pi_diff = pi_chosen - pi_rejected
        ref_diff = reference_chosen - reference_rejected
        dpo_core = beta * (pi_diff - ref_diff)
        loss = -logsigmoid(dpo_core).mean()
      and metrics computed on log_diff = pi_diff - ref_diff.
    - We return metrics as Python floats to match the production function contract.
    """
    pi_diff = pi_chosen - pi_rejected
    ref_diff = reference_chosen - reference_rejected
    dpo_core = beta * (pi_diff - ref_diff)
    loss = -torch.nn.functional.logsigmoid(dpo_core).mean()

    log_diff = pi_diff - ref_diff
    metrics = {
        "log_diff_mean": float(log_diff.mean().detach().item()),
        "log_diff_std": float(log_diff.std().detach().item()),
        "log_diff_min": float(log_diff.min().detach().item()),
        "log_diff_max": float(log_diff.max().detach().item()),
        "log_diff_frac_pos": float((log_diff > 0).float().mean().detach().item()),
        "pi_margin_mean": float(pi_diff.mean().detach().item()),
        "reference_margin_mean": float(ref_diff.mean().detach().item()),
    }
    return loss, metrics


def test_dpo_loss_torch_matches_expected_on_vector_inputs() -> None:
    """
    Validate correctness of loss and metrics for simple vector-shaped inputs.

    Notes
    -----
    - Uses shape (B,) tensors to keep expected computations transparent.
    - Compares loss tensor value and each metrics entry to a locally computed expected result.
    """
    beta = 0.2
    pi_chosen = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    pi_rejected = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    ref_chosen = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ref_rejected = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    loss, metrics = dpo_loss_torch(
        pi_chosen=pi_chosen,
        pi_rejected=pi_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )

    exp_loss, exp_metrics = _expected_loss_and_metrics(
        pi_chosen=pi_chosen,
        pi_rejected=pi_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.allclose(loss, exp_loss, atol=1e-6, rtol=0.0)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == set(exp_metrics.keys())
    for k, v in metrics.items():
        assert isinstance(v, float)
        assert abs(v - exp_metrics[k]) <= 1e-6


def test_dpo_loss_torch_matches_expected_on_matrix_inputs() -> None:
    """
    Validate correctness of loss and metrics for matrix-shaped inputs.

    Notes
    -----
    - Uses shape (B, T) tensors to mimic token-level log-probs or other structured batches.
    - Metrics should reduce over all elements (same behavior as calling tensor.mean(), tensor.std(), etc.).
    """
    beta = 0.1
    pi_chosen = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    pi_rejected = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    ref_chosen = torch.zeros_like(pi_chosen)
    ref_rejected = torch.zeros_like(pi_chosen)

    loss, metrics = dpo_loss_torch(
        pi_chosen=pi_chosen,
        pi_rejected=pi_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )

    exp_loss, exp_metrics = _expected_loss_and_metrics(
        pi_chosen=pi_chosen,
        pi_rejected=pi_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )

    assert loss.ndim == 0
    assert torch.allclose(loss, exp_loss, atol=1e-6, rtol=0.0)
    for k in exp_metrics:
        assert abs(metrics[k] - exp_metrics[k]) <= 1e-6


def test_dpo_loss_torch_is_differentiable_wrt_inputs() -> None:
    """
    Validate that the returned loss supports autograd.

    Notes
    -----
    - This test checks gradient flow for typical training usage.
    - We require at least one gradient to be non-zero.
    """
    beta = 0.3
    pi_chosen = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    pi_rejected = torch.tensor([0.5, 0.0, 0.25], dtype=torch.float32, requires_grad=True)
    ref_chosen = torch.zeros_like(pi_chosen, requires_grad=True)
    ref_rejected = torch.zeros_like(pi_chosen, requires_grad=True)

    loss, _ = dpo_loss_torch(
        pi_chosen=pi_chosen,
        pi_rejected=pi_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )

    loss.backward()

    grads = [pi_chosen.grad, pi_rejected.grad, ref_chosen.grad, ref_rejected.grad]
    assert all(g is not None for g in grads), "Expected gradients for all inputs."
    total = torch.stack([g.abs().sum() for g in grads if g is not None]).sum().item()
    assert total > 0.0, "All gradients are zero; expected non-zero gradient flow."


def test_dpo_loss_torch_raises_on_shape_mismatch() -> None:
    """
    Validate that mismatched shapes raise ValueError (via validation).

    Notes
    -----
    - pi_rejected has a different shape than pi_chosen.
    """
    pi_chosen = torch.zeros((3,), dtype=torch.float32)
    pi_rejected = torch.zeros((2,), dtype=torch.float32)
    ref_chosen = torch.zeros((3,), dtype=torch.float32)
    ref_rejected = torch.zeros((3,), dtype=torch.float32)

    with pytest.raises(ValueError, match="same shape"):
        _ = dpo_loss_torch(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)


def test_dpo_loss_torch_raises_on_non_floating_dtype() -> None:
    """
    Validate that non-floating tensors raise ValueError (via validation).

    Notes
    -----
    - Uses int64 tensors.
    """
    pi_chosen = torch.zeros((3,), dtype=torch.int64)
    pi_rejected = torch.zeros((3,), dtype=torch.int64)
    ref_chosen = torch.zeros((3,), dtype=torch.int64)
    ref_rejected = torch.zeros((3,), dtype=torch.int64)

    with pytest.raises(ValueError, match="floats|floating"):
        _ = dpo_loss_torch(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)


def test_dpo_loss_torch_raises_on_non_finite_values() -> None:
    """
    Validate that NaN/Inf values raise ValueError (via validation).
    """
    pi_chosen = torch.tensor([0.0, float("inf"), 1.0], dtype=torch.float32)
    pi_rejected = torch.zeros((3,), dtype=torch.float32)
    ref_chosen = torch.zeros((3,), dtype=torch.float32)
    ref_rejected = torch.zeros((3,), dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        _ = dpo_loss_torch(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_dpo_loss_torch_raises_on_invalid_beta(beta: float) -> None:
    """
    Validate that invalid beta values raise ValueError (via validation).

    Parameters
    ----------
    beta : float
        Invalid beta values: non-positive or non-finite.
    """
    pi_chosen = torch.zeros((3,), dtype=torch.float32)
    pi_rejected = torch.zeros((3,), dtype=torch.float32)
    ref_chosen = torch.zeros((3,), dtype=torch.float32)
    ref_rejected = torch.zeros((3,), dtype=torch.float32)

    with pytest.raises(ValueError, match="Beta|beta|finite|positive"):
        _ = dpo_loss_torch(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=beta)


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available for device mismatch test.")
def test_dpo_loss_torch_raises_on_device_mismatch_cuda() -> None:
    """
    Validate that device mismatch raises ValueError when MPS is available.

    Notes
    -----
    - Places pi_chosen on MPS and keeps other tensors on CPU.
    - Validation should reject mixed devices.
    """
    pi_chosen = torch.zeros((3,), dtype=torch.float32, device=torch.device("mps"))
    pi_rejected = torch.zeros((3,), dtype=torch.float32, device=torch.device("cpu"))
    ref_chosen = torch.zeros((3,), dtype=torch.float32, device=torch.device("cpu"))
    ref_rejected = torch.zeros((3,), dtype=torch.float32, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="same device"):
        _ = dpo_loss_torch(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)


def test_dpo_loss_torch_metrics_keys_and_types() -> None:
    """
    Validate that metrics include all expected keys and are Python floats.

    Notes
    -----
    - This test is intentionally redundant with correctness tests,
      but it provides a clear contract check for downstream logging.
    """
    pi_chosen = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    pi_rejected = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ref_chosen = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ref_rejected = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    _, metrics = dpo_loss_torch(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)

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
    assert all(isinstance(v, float) for v in metrics.values())
