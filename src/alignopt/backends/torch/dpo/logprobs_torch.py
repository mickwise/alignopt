"""
Purpose
-------
Compute completion-only sequence log-probabilities for (prompt, completion) pairs under a
causal language model (PyTorch backend). This supports preference-optimization objectives
(e.g., DPO) that need log π(y | x) for the completion tokens only.

Key behaviors
-------------
- Tokenizes (prompt, completion) pairs using alignopt.backends.text.tokenize_text(...),
  which returns input_ids, attention_mask, and per-example completion start_indices.
- Runs a causal LM forward pass and computes token-level log-probabilities aligned to
  next-token prediction (targets are input_ids[:, 1:]).
- Masks out:
  1) padding targets using attention_mask[:, 1:],
  2) prompt-region targets using start_indices (shift-corrected via start_indices - 1).
- Returns per-example sums of completion-token log-probabilities.

Conventions
-----------
- Causal LM alignment: logits at position t predict token at position t+1.
- start_indices are indices in the unshifted token sequence produced by tokenize_text(...).
- Masking is applied to token_logprobs (aligned to targets input_ids[:, 1:]) starting at
  index (start_indices - 1).
- If model.config.use_cache exists and is True, it is temporarily set to False during the
  forward pass and restored afterward.

Downstream usage
----------------
- Used by preference-optimization losses (e.g., DPO) to compute log π(y | x) for chosen/rejected
  completions.
- For evaluation-only usage, callers may wrap calls in torch.no_grad().
- Callers control model.train()/model.eval(); this function does not change model mode.
"""

from typing import Sequence
from transformers import PreTrainedModel, PreTrainedTokenizerFast
import torch
from alignopt.backends.text import tokenize_text
from alignopt.backends.backends_validation import validate_logprobs_inputs
from alignopt.backends.backends_types import PreferencePair


def completion_logprobs_torch(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        pairs: Sequence[PreferencePair],
        max_length: int | None = None
    ) -> torch.Tensor:
    """
    Compute per-example completion-only log-probability sums
    for a batch of (prompt, completion) pairs.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Hugging Face causal language model. The forward pass must accept
        input_ids and attention_mask and return an object with a .logits
        tensor of shape (B, T, V).
    tokenizer : transformers.PreTrainedTokenizerFast
        Fast tokenizer used by tokenize_text(...) to produce input IDs, attention masks,
        and completion start_indices under the project’s boundary convention.
    pairs : Sequence[PreferencePair]
        Batch of (prompt, completion) string pairs.
    max_length : int | None
        Maximum sequence length used during tokenization (truncation limit).
        If None, tokenizer defaults apply.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B,), where each entry is the sum of log-probabilities of completion tokens
        under the model for that example. Prompt-region and padding tokens contribute zero.

    Raises
    ------
    ValueError
        If max_length fails validation in validate_logprobs_inputs(...).
    TypeError
        If tokenize_text(...) rejects the tokenizer or inputs.

    Notes
    -----
    - This function is differentiable; gradients flow back into model parameters.
    - For evaluation-only usage, callers may wrap the call in torch.no_grad().
    - This function does not change model.train()/model.eval(); callers are responsible for setting
      the appropriate mode (e.g., model.eval() to disable dropout).
    """


    validate_logprobs_inputs(pairs, max_length)
    device = next(model.parameters()).device
    input_ids, attention_mask, start_indices = tokenize_text(
        pairs,
        tokenizer,
        max_length
    )
    input_ids = torch.from_numpy(input_ids)
    attention_mask = torch.from_numpy(attention_mask)
    start_indices = torch.from_numpy(start_indices)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    start_indices = start_indices.to(device)
    token_logprobs: torch.Tensor = _calculate_token_logprobs(model, input_ids, attention_mask)
    masked_logprobs: torch.Tensor = _mask_logprobs(
        token_logprobs,
        attention_mask,
        start_indices
    )
    return masked_logprobs.sum(dim=1)


def _calculate_token_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities aligned to next-token prediction for a causal LM.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Hugging Face model returning .logits of shape (B, T, V).
    input_ids : torch.Tensor
        Token IDs of shape (B, T).
    attention_mask : torch.Tensor
        Attention mask of shape (B, T) with 1 for real tokens and 0 for padding.

    Returns
    -------
    torch.Tensor
        Token log-probabilities of shape (B, T-1), aligned to targets input_ids[:, 1:].

    Raises
    ------
    None

    Notes
    -----
    - If model.config.use_cache exists and is True, it is temporarily set to
      False during the forward pass and restored in a finally block to ensure
      restoration even on exceptions.
    - The returned tensor is unmasked; masking is applied in _mask_logprobs(...).
    """

    old_use_cache: bool | None = None
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        old_use_cache = bool(config.use_cache)
        if old_use_cache:
            config.use_cache = False
    try:
        return _calculate_log_probs(model, input_ids, attention_mask)
    finally:
        if old_use_cache:
            config.use_cache = old_use_cache


def _calculate_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
    """
    Compute token-level log-probabilities for next-token targets
    using a flattened cross-entropy pass.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Hugging Face model returning .logits of shape (B, T, V).
    input_ids : torch.Tensor
        Token IDs of shape (B, T).
    attention_mask : torch.Tensor
        Attention mask of shape (B, T).

    Returns
    -------
    torch.Tensor
        Token log-probabilities of shape (B, T-1) aligned to targets input_ids[:, 1:].

    Raises
    ------
    None

    Notes
    -----
    - Uses logits[:, :-1, :] and targets input_ids[:, 1:]
      to implement causal next-token alignment.
    - Uses flatten(0, 1) on logits and flatten() on targets
      to avoid permute, then reshapes back to (B, T-1).
    """

    batch_size, token_count_minus_1 = input_ids.size(0), input_ids.size(1) - 1
    logits: torch.Tensor = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, :-1, :].flatten(0 ,1)
    targets: torch.Tensor = input_ids[:, 1:].flatten()
    flat_token_logprobs: torch.Tensor =  (
        -torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    )
    return flat_token_logprobs.reshape(batch_size, token_count_minus_1)


def _mask_logprobs(
    token_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    start_indices: torch.Tensor
    ) -> torch.Tensor:
    """
    Apply padding and completion-start masking to token-level log-probabilities.

    Parameters
    ----------
    token_logprobs : torch.Tensor
        Token log-probabilities of shape (B, T-1), aligned to targets input_ids[:, 1:].
    attention_mask : torch.Tensor
        Attention mask of shape (B, T) for the unshifted input sequence.
    start_indices : torch.Tensor
        Tensor of shape (B,) giving the completion start
        token index in the unshifted sequence.

    Returns
    -------
    torch.Tensor
        Masked token log-probabilities of shape (B, T-1), where
        prompt-region and padding-region entries are set to 0.0.

    Raises
    ------
    None

    Notes
    -----
    - Padding mask uses attention_mask[:, 1:] to align with targets
      (input_ids[:, 1:]).
    - Completion mask uses positions >= (start_indices - 1) to account for the causal shift:
      token_logprobs[:, t] corresponds to the logprob of input_ids[:, t+1].
    - masked_fill(~mask, 0.0) is used to keep dtype and maintain differentiability.
    """


    pad_mask: torch.Tensor = attention_mask[:, 1:].bool()
    positions: torch.Tensor = torch.arange(
        0, attention_mask.size(-1) - 1, device=attention_mask.device
    )
    completion_mask: torch.Tensor = positions >= start_indices.unsqueeze(-1) - 1
    logprobs_mask: torch.Tensor = pad_mask & completion_mask
    return token_logprobs.masked_fill(~logprobs_mask, 0.0)
