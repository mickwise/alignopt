"""
Purpose
-------
Compute per-example completion log-probability sums for Flax causal language models.

This module provides a JAX/Flax backend for DPO-style objectives and related training/evaluation
code that needs the log-probability mass assigned to the completion portion of (prompt, completion)
text pairs.

Key behaviors
-------------
- Tokenizes (prompt, completion) pairs into input_ids, attention_mask, and start_positions.
- Computes realized next-token log-probabilities using a causal shift (logits[:, :-1] vs targets[:, 1:]).
- Masks out prompt-region and padding-region token log-probabilities using start_positions and attention_mask.
- Returns a per-example sum of completion-only token log-probabilities.

Conventions
-----------
- start_positions are indices into the unshifted token sequence (input_ids) indicating the first completion token.
  Masking is applied on the shifted token_logprobs (aligned to targets input_ids[:, 1:]) using:
  positions >= start_positions (with positions indexing the shifted target positions).
- attention_mask is expected to be 0/1 (or boolean-castable) and is aligned to targets via attention_mask[:, 1:].
- max_length is validated upstream and governs truncation behavior; truncation may remove all completion tokens,
  in which case downstream masking should yield zero completion mass for that example.

Downstream usage
----------------
Call completion_logprobs(model, params, tokenizer, pairs, max_length=...) to obtain a vector of shape (B,)
containing completion-only log-probability sums per example. This output is typically consumed by losses such as
DPO/IPO variants or by evaluation code that compares preference pair likelihoods.
"""

from typing import Sequence, Tuple
from transformers import FlaxPreTrainedModel, PreTrainedTokenizerFast
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from alignopt.backends.text import tokenize_text
from alignopt.backends.backends_validation import validate_logprobs_inputs

def completion_logprobs(
    model: FlaxPreTrainedModel,
    params: FrozenDict,
    tokenizer: PreTrainedTokenizerFast,
    pairs: Sequence[Tuple[str, str]],
    max_length: int | None = None
    ) -> jax.Array:
    """
    Compute completion-only log-probability sums for a batch of (prompt, completion) pairs.

    Parameters
    ----------
    model : transformers.FlaxPreTrainedModel
        Flax causal LM used to produce logits. Must support being called with input_ids, attention_mask,
        params, and train=False for deterministic inference semantics.
    params : flax.core.frozen_dict.FrozenDict
        Model parameters passed to the Flax model call.
    tokenizer : transformers.PreTrainedTokenizerFast
        Tokenizer used to encode text pairs into input_ids and attention_mask.
    pairs : Sequence[Tuple[str, str]]
        Sequence of (prompt, completion) pairs.
    max_length : int | None
        Optional maximum token length for truncation. Validated by validate_logprobs_inputs(...).

    Returns
    -------
    jax.Array
        Array of shape (B,) containing the summed log-probabilities of completion tokens for each example.

    Raises
    ------
    ValueError
        If validate_logprobs_inputs(...) rejects the provided pairs or max_length (e.g., invalid max_length).

    Notes
    -----
    - Tokenization is performed by alignopt.backends.text.tokenize_text(...), which must return:
      input_ids : (B, T) integer token ids,
      attention_mask : (B, T) mask (0/1),
      start_positions : (B,) completion start indices in the unshifted sequence.
    - Dtypes are normalized for JAX/accelerator compatibility:
      input_ids,attention_mask and start_positions are cast to int32.
    - The returned value is completion-only: prompt-region and padding-region token log-probabilities are excluded.
    """

    validate_logprobs_inputs(pairs, max_length)
    input_ids, attention_mask, start_positions = tokenize_text(
        pairs,
        tokenizer,
        max_length
    )
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    attention_mask = jnp.asarray(attention_mask, dtype=jnp.int32)
    start_positions = jnp.asarray(start_positions, dtype=jnp.int32)
    token_logprobs: jax.Array = _calculate_token_logprobs(
        model, params, input_ids, attention_mask
    )
    masked_logprobs: jax.Array = _mask_logprobs(
        token_logprobs,
        attention_mask,
        start_positions
    )
    return masked_logprobs.sum(axis=1)


def _calculate_token_logprobs(
        model: FlaxPreTrainedModel,
        params: FrozenDict,
        input_ids: jax.Array,
        attention_mask: jax.Array
    ) -> jax.Array:
    """
    Compute realized next-token log-probabilities for a batch of tokenized sequences.

    Parameters
    ----------
    model : transformers.FlaxPreTrainedModel
        Flax causal LM producing logits of shape (B, T, V) for input_ids.
    params : flax.core.frozen_dict.FrozenDict
        Model parameters passed to the Flax model call.
    input_ids : jax.Array
        Integer token ids of shape (B, T). Must be suitable for model embedding lookup (typically int32).
    attention_mask : jax.Array
        Attention mask of shape (B, T) with 0/1 values indicating real tokens vs padding.

    Returns
    -------
    jax.Array
        Token-level realized log-probabilities aligned to targets, of shape (B, T-1).
        Entry [b, p] corresponds to log P(input_ids[b, p+1] | input_ids[b, :p+1]) under the model.

    Raises
    ------
    TypeError
        If the model call does not accept the provided keyword arguments (e.g., missing params or train).

    Notes
    -----
    - Uses causal alignment:
    - logits[:, :-1, :] correspond to predictions for targets input_ids[:, 1:].
    - Computes per-target negative log-likelihood via optax.softmax_cross_entropy_with_integer_labels(...)
      and returns the negated NLL (log-probabilities).
    - The attention_mask is not applied here; padding/prompt masking is handled downstream in _mask_logprobs(...).
    """

    logits: jax.Array = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        params=params,
        train=False
    ).logits[:, :-1, :]
    targets: jax.Array = input_ids[:, 1:]
    nll: jax.Array = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return -nll


def _mask_logprobs(
        token_logprobs: jax.Array,
        attention_mask: jax.Array,
        start_position: jax.Array
    ) -> jax.Array:
    """
    Mask token-level log-probabilities to keep completion-region, non-padding targets only.

    Parameters
    ----------
    token_logprobs : jax.Array
        Token-level realized log-probabilities of shape (B, T-1), aligned to targets input_ids[:, 1:].
    attention_mask : jax.Array
        Attention mask of shape (B, T) with 0/1 values. Padding is excluded using attention_mask[:, 1:].
    start_position : jax.Array
        Integer array of shape (B,) containing the completion start index in the unshifted sequence.

    Returns
    -------
    jax.Array
        Masked token log-probabilities of shape (B, T-1), where prompt-region targets and padding targets
        have been zeroed out.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes that prevent correct broadcasting/alignment.

    Notes
    -----
    - pad_mask is formed from attention_mask[:, 1:] to align with token_logprobs (targets region).
    - positions is a 1D index array over target positions (1..T-1 in unshifted token indices), broadcast over batch.
    - completion_mask selects targets whose unshifted index >= start_position[b].
    - The mask is applied multiplicatively, producing zeros for excluded positions.
    """

    pad_mask: jax.Array = attention_mask[:, 1:].astype(bool)
    positions: jax.Array = jnp.arange(1, attention_mask.shape[1])
    completion_mask: jax.Array = positions >= start_position[:, None]
    logprobs_mask: jax.Array = pad_mask & completion_mask
    return token_logprobs*logprobs_mask
