"""
Purpose
-------
Torch-specific validation utilities for DPO loss inputs.

Key behaviors
-------------
- Validates that policy and reference log-probability tensors are shape-compatible.
- Enforces that all tensors live on the same torch.device to avoid implicit device transfers.
- Ensures tensors are floating-point and contain only finite values (no NaN/Inf).
- Validates the scalar hyperparameter beta is finite and strictly positive.

Conventions
-----------
- This module enforces only *compatibility* (same shape, same device, float, finite).
  It intentionally does not enforce a specific tensor rank/shape beyond equality.
- `beta` must be a Python numeric scalar (float-like), not a tensor.

Downstream usage
----------------
Call validate_dpo_data_torch(...) at the beginning of torch DPO objectives before computing
margins, logits, or reductions. This keeps backend implementations fail-fast and consistent.
"""

from typing import List, Iterable, Set, Literal
from math import isfinite
import torch
from torch.optim import lr_scheduler, Optimizer

def validate_dpo_data_torch(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1
    ) -> None:
    """
    Validate DPO log-probability tensors and beta for Torch backends.

    Parameters
    ----------
    pi_chosen : torch.Tensor
        Log-probabilities under the current policy for the chosen responses.
        Shape is unconstrained except that it must match the other three tensors
        exactly.
    pi_rejected : torch.Tensor
        Log-probabilities under the current policy for the rejected responses.
        Must have the same shape/device/dtype constraints as `pi_chosen`.
    reference_chosen : torch.Tensor
        Log-probabilities under the reference policy for the chosen responses.
        Must have the same shape/device/dtype constraints as `pi_chosen`.
    reference_rejected : torch.Tensor
        Log-probabilities under the reference policy for the rejected responses.
        Must have the same shape/device/dtype constraints as `pi_chosen`.
    beta : float
        Positive finite scalar controlling the strength of the preference term.

    Returns
    -------
    None
        This function returns nothing; it raises on invalid inputs.

    Raises
    ------
    ValueError
        If:
        - `beta` is not finite or is <= 0,
        - any input tensor differs in shape from `pi_chosen`,
        - any input tensor is on a different device than `pi_chosen`,
        - any input tensor is not floating-point,
        - any input tensor contains non-finite values (NaN or Inf).

    Notes
    -----
    - This validator is intentionally agnostic to the *meaning* of the tensor shape
      (e.g., (B,), (B, T), or higher-rank). Backend code is expected to establish
      shape semantics through its own data flow and reductions.
    - Device equality is enforced to keep compute explicit and avoid silent host/device transfers.
    """

    reference_shape: torch.Size = pi_chosen.shape
    input_tensor_list: List[torch.Tensor] = [pi_chosen, pi_rejected, reference_chosen, reference_rejected]
    reference_device: torch.device = pi_chosen.device
    if beta <= 0.0 or not isfinite(beta):
        raise ValueError("Beta must be a finite, positive float.")
    for input_tensor in input_tensor_list:
        if input_tensor.shape != reference_shape:
            raise ValueError("All log probability tensors must have the same shape.")
        if input_tensor.device != reference_device:
            raise ValueError("All input tensors must be on the same device.")
        if not torch.is_floating_point(input_tensor):
            raise ValueError("All input tensors must hold floats.")
        if not torch.all(torch.isfinite(input_tensor)):
            raise ValueError("All input tensor elements must be finite.")


def validate_training_knobs(
    batch_size: int,
    max_length: int | None,
    lr: float,
    eps: float,
    weight_decay: float,
    num_epochs: int,
    steps_to_metrics: int | None
) -> None:
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if max_length is not None and max_length <= 0:
        raise ValueError("Max length must be a positive integer.")
    if lr < 0.0:
        raise ValueError("Learning rate must be a positive real number.")
    if eps < 0.0:
        raise ValueError("Standard deviation epsilon must be a positive real number")
    if weight_decay < 0.0:
        raise ValueError("Weight decay coefficient must be a positive real number")
    if num_epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")
    if steps_to_metrics is not None and steps_to_metrics <= 0:
        raise ValueError("Number of steps between metrics display must be a positive integer.")
    

def validate_provided_optimizer_and_scheduler(
    provided_optimizer: Optimizer | None,
    provided_scheduler: lr_scheduler.LRScheduler | None,
    current_device: torch.device,
    scheduler_step: Literal["batch", "epoch"],
    params: Iterable[torch.nn.Parameter]
    ) -> None:
    if provided_optimizer is not None:
        _validate_parameter_equivalence(provided_optimizer, params)
    if provided_scheduler is not None:
        pass
    elif scheduler_step == "epoch":
        raise ValueError("Scheduler must take a step per batch if default scheduler is used.")
        

def _validate_parameter_equivalence(
        provided_optimizer: Optimizer, params: Iterable[torch.nn.Parameter]
    ) -> None:
    optimizer_params: Set[torch.nn.Parameter] = {
        id(param) for param_group in provided_optimizer.param_groups for param in param_group["params"]
    }
    params_set: Set[torch.nn.Parameter] = {id(param) for param in params}
    if optimizer_params != params_set:
        raise ValueError("Provided optimizer must be associated with policy model parameters.")
