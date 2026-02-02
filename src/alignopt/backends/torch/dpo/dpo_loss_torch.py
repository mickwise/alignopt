"""
Purpose
-------
Backend-specific Direct Preference Optimization (DPO) loss implementation for PyTorch.

This module provides a Torch-native implementation of the core DPO objective used in
preference learning pipelines. It assumes inputs are *already computed*
log-probability sums per sample (e.g., summed completion log-probs), and produces:
1) a scalar batch loss suitable for backpropagation, and
2) a small set of detached scalar diagnostics for logging.

Key behaviors
-------------
- Validates input tensor consistency via validate_dpo_data_torch(...).
- Computes per-sample margins:
  - pi_diff = pi_chosen - pi_rejected
  - reference_diff = reference_chosen - reference_rejected
- Computes the DPO logits: beta * (pi_diff - reference_diff).
- Uses -logsigmoid(DPO logits) as the per-sample loss and averages over the batch.
- Returns detached metrics (mean/std/min/max, fraction positive, margin means).

Conventions
-----------
- All input tensors must have identical shape and reside on the same device.
- All inputs are floating tensors containing finite values.
- Shapes are treated as opaque; only equality across inputs is required.
- Output loss is a torch.Tensor (not detached) to support gradient flow.

Downstream usage
----------------
Call dpo_loss_torch(...) inside a training step after computing policy/reference
log-prob sums for chosen/rejected samples. Use the returned metrics dict for logging.
"""

from typing import Tuple, Dict
import torch
from alignopt.backends.torch.dpo.dpo_validation_torch import validate_dpo_data_torch


def dpo_loss_torch(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the mean DPO loss for a batch and return detached diagnostic metrics.

    Parameters
    ----------
    pi_chosen : torch.Tensor
        Policy model log-probability sums for the chosen completions.
        Shape: (B, ...) where B is batch size; must match all other inputs.
    pi_rejected : torch.Tensor
        Policy model log-probability sums for the rejected completions.
        Same shape/device/dtype constraints as pi_chosen.
    reference_chosen : torch.Tensor
        Reference model log-probability sums for the chosen completions.
        Same shape/device/dtype constraints as pi_chosen.
    reference_rejected : torch.Tensor
        Reference model log-probability sums for the rejected completions.
        Same shape/device/dtype constraints as pi_chosen.
    beta : float, default=0.1
        Positive inverse-temperature scaling the DPO logits. Larger beta increases
        the sharpness of the preference optimization signal.

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, float]]
        (mean_batch_loss, metrics) where:
        - mean_batch_loss is a scalar torch.Tensor suitable for backpropagation.
        - metrics is a dict of detached Python floats for logging/monitoring.

    Raises
    ------
    ValueError
        If validate_dpo_data_torch(...) fails (e.g., shape mismatch, device mismatch,
        non-floating dtype, non-finite values, or invalid beta).

    Notes
    -----
    - The loss is computed as:
        pi_diff = pi_chosen - pi_rejected
        ref_diff = reference_chosen - reference_rejected
        dpo_logits = beta * (pi_diff - ref_diff)
        loss = mean( -logsigmoid(dpo_logits) )
    - This function is intentionally backend-specific (Torch) and does not attempt
      device movement or implicit casting; it enforces a strict input contract.
    """

    validate_dpo_data_torch(
        pi_chosen,
        pi_rejected,
        reference_chosen,
        reference_rejected,
        beta
    )
    pi_diff, reference_diff, dpo_core = _dpo_loss_core_torch(
        pi_chosen,
        pi_rejected,
        reference_chosen,
        reference_rejected,
        beta
    )
    mean_batch_loss: torch.Tensor = -torch.nn.functional.logsigmoid(dpo_core).mean()
    return mean_batch_loss, _calculate_metrics(pi_diff, reference_diff)


def _dpo_loss_core_torch(
        pi_chosen: torch.Tensor,
        pi_rejected: torch.Tensor,
        reference_chosen: torch.Tensor,
        reference_rejected: torch.Tensor,
        beta: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-sample DPO intermediate tensors (policy margin, reference margin, DPO logits).

    Parameters
    ----------
    pi_chosen : torch.Tensor
        Policy log-probability sums for chosen completions. Shape (B, ...).
    pi_rejected : torch.Tensor
        Policy log-probability sums for rejected completions. Same shape as pi_chosen.
    reference_chosen : torch.Tensor
        Reference log-probability sums for chosen completions. Same shape as pi_chosen.
    reference_rejected : torch.Tensor
        Reference log-probability sums for rejected completions. Same shape as pi_chosen.
    beta : float, default=0.1
        Positive inverse-temperature scaling for the DPO logits.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (pi_diff, reference_diff, dpo_logits) where:
        - pi_diff is the policy margin per sample.
        - reference_diff is the reference margin per sample.
        - dpo_logits = beta * (pi_diff - reference_diff).

    Raises
    ------
    None

    Notes
    -----
    - This helper is kept small and side-effect free so that the main loss function
      remains readable and easy to audit.
    - Input validation is expected to be handled by the caller.
    """

    pi_diff: torch.Tensor = pi_chosen - pi_rejected
    reference_diff: torch.Tensor = reference_chosen - reference_rejected
    return pi_diff, reference_diff, beta*(pi_diff - reference_diff)


def _calculate_metrics(
        pi_diff: torch.Tensor,
        reference_diff: torch.Tensor
    ) -> Dict[str, float]:
    """
    Compute detached scalar diagnostics from policy/reference margins for logging.

    Parameters
    ----------
    pi_diff : torch.Tensor
        Policy margin tensor = pi_chosen - pi_rejected. Shape (B, ...).
    reference_diff : torch.Tensor
        Reference margin tensor = reference_chosen - reference_rejected.
        Same shape as pi_diff.

    Returns
    -------
    Dict[str, float]
        Dictionary of detached Python floats:
        - log_diff_mean: mean(pi_diff - reference_diff)
        - log_diff_std: std(pi_diff - reference_diff)
        - log_diff_min: min(pi_diff - reference_diff)
        - log_diff_max: max(pi_diff - reference_diff)
        - log_diff_frac_pos: fraction of elements where (pi_diff - reference_diff) > 0
        - pi_margin_mean: mean(pi_diff)
        - reference_margin_mean: mean(reference_diff)

    Raises
    ------
    None

    Notes
    -----
    - All returned values are detached and converted to Python floats via .item(),
      making them safe for logging without holding graph references.
    - This function does not validate finiteness/dtype; caller is expected to ensure
      the input contract via validate_dpo_data_torch(...).
    """

    log_diff: torch.Tensor = pi_diff - reference_diff
    return {
        "log_diff_mean": log_diff.mean().detach().item(),
        "log_diff_std": log_diff.std().detach().item() if log_diff.shape[0] > 1 else 0.0,
        "log_diff_min": log_diff.min().detach().item(),
        "log_diff_max": log_diff.max().detach().item(),
        "log_diff_frac_pos": (log_diff > 0).float().mean().detach().item(),
        "pi_margin_mean": pi_diff.mean().detach().item(),
        "reference_margin_mean": reference_diff.mean().detach().item()
    }

