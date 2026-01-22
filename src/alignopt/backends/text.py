"""
Purpose
-------
Tokenize (prompt, completion) text pairs using a single canonical concatenation rule and
derive a completion-start token index from tokenizer offset mappings. This exists to
support preference-optimization algorithms (e.g., DPO) that need log-probabilities over
completion tokens only, without silently mismatching prompt boundaries after tokenization.

Key behaviors
-------------
- Builds canonical strings as: prompt + SEP + completion (SEP is a single space by default).
- Uses a *fast* Hugging Face tokenizer with return_offsets_mapping=True to obtain per-token
  character spans in the canonical string.
- Computes per-example completion start token indices by comparing offset char-starts to a
  per-example boundary character index.
- Emits NumPy arrays from the tokenizer call and converts to either PyTorch or JAX arrays
  at the end, depending on the requested backend.

Conventions
-----------
- Canonical boundary is a character index in the canonical string; by convention this module
  treats SEP as belonging to the completion side (boundary is at len(prompt), not len(prompt+SEP)).
- Requires a fast tokenizer (PreTrainedTokenizerFast); offset mappings must be available.
- Padding/truncation are enabled; if truncation removes the completion boundary, the computed
  start index will indicate "no completion tokens present" (see function docs).

Downstream usage
----------------
Call tokenize_text(...) from backend-specific logprob code. Use the returned start indices to
build a completion-only mask aligned with shifted token logprobs (next-token prediction).
"""

from typing import Sequence, List, Tuple, TypeAlias
from transformers import PreTrainedTokenizerFast, BatchEncoding
import torch
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

SEP: str = " "

BackendTensor: TypeAlias = torch.Tensor | jax.Array | NDArray

def tokenize_text(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None,
    backend: str = "pt"
    ) -> Tuple[BackendTensor, BackendTensor, BackendTensor]:
    """
    Tokenize a batch of (prompt, completion) pairs and compute completion-start token indices.

    Parameters
    ----------
    pairs : Sequence[Tuple[str, str]]
        Sequence of (prompt, completion) strings. Each pair is concatenated using the module SEP.
    tokenizer : PreTrainedTokenizerFast
        Hugging Face *fast* tokenizer used to produce input IDs,
        attention masks, and offset mappings.
    max_length : int | None
        Maximum sequence length used by the tokenizer when truncation=True.
        If None, tokenizer default is used.
    backend : str
        Output backend selector. Expected values are currently "pt" (PyTorch) or "jax" (JAX).
        Tokenization is done as NumPy first, then converted at the end.

    Returns
    -------
    Tuple[BackendTensor, BackendTensor, BackendTensor]
        (input_ids, attention_mask, start_indices) where:
        - input_ids has shape (B, T)
        - attention_mask has shape (B, T)
        - start_indices has shape (B,) and gives the first token position whose char-start offset is
        at/after the boundary character index for that example.
        If an example has no completion tokens present after truncation/padding removal,
        start_indices uses a sentinel value (T in this case). Downstream code should treat
        that as "mask all completion tokens off".

    Raises
    ------
    TypeError
        If a non-fast tokenizer is provided and offset mappings cannot be produced.

    Notes
    -----
    - This function is intentionally NumPy-first to keep shared logic backend-agnostic;
      device placement is handled downstream by the backend-specific code.
    - start_indices are token positions in the *tokenized* sequence, not character positions.
    """

    texts_batch, boundary_chars = _build_texts(pairs)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Tokenizer should be a fast variant.")
    batch_encoding: BatchEncoding = tokenizer(
        texts_batch,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="np",
        return_offsets_mapping=True
    )
    input_ids: NDArray = batch_encoding["input_ids"]
    attention_mask: NDArray = batch_encoding["attention_mask"]
    offset_mapping: NDArray = batch_encoding["offset_mapping"]
    start_indices: NDArray = _calculate_start_indices(offset_mapping, boundary_chars)
    match backend:
        case "pt":
            return (
                torch.from_numpy(input_ids),
                torch.from_numpy(attention_mask),
                torch.from_numpy(start_indices)
            )
        case "jax":
            return (
                jnp.array(input_ids),
                jnp.array(attention_mask),
                jnp.array(start_indices)
            )
        case _:
            return input_ids, attention_mask, start_indices


def _build_texts(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], NDArray]:
    """
    Construct canonical concatenated texts and their boundary character indices.

    Parameters
    ----------
    pairs : Sequence[Tuple[str, str]]
        Sequence of (prompt, completion) strings.

    Returns
    -------
    Tuple[List[str], NDArray]
        (texts_batch, boundary_chars) where:
        - texts_batch[i] == prompt_i + SEP + completion_i
        - boundary_chars[i] is the character index in texts_batch[i] that defines where the completion segment begins.
        By module convention, SEP is treated as part of the completion side, so boundary_chars[i] == len(prompt_i).

    Raises
    ------
    None

    Notes
    -----
    - boundary_chars are *character indices* in the canonical string, used only with offset mappings to derive token indices.
    - Keep SEP and boundary convention consistent across the project; downstream masking assumes this contract.
    """

    texts_batch: List[str] = []
    boundary_chars: List[int] = []
    for prompt, completion in pairs:
        texts_batch.append(prompt + SEP + completion)
        boundary_chars.append(len(prompt))
    return texts_batch, np.array(boundary_chars)


def _calculate_start_indices(offset_mapping: NDArray, boundary_chars: NDArray) -> NDArray:
    """
    Compute completion-start token indices from offset mappings and boundary character indices.

    Parameters
    ----------
    offset_mapping : NDArray
        Array of shape (B, T, 2) containing per-token (char_start, char_end) offsets into the canonical string.
    boundary_chars : NDArray
        Array of shape (B,) giving per-example boundary character indices in the canonical string.

    Returns
    -------
    NDArray
        Array of shape (B,) giving completion-start token indices: the smallest token index t such that
        token t is a real (non-padding/non-special) token and offset_mapping[b, t, 0] >= boundary_chars[b].
        If no such token exists (e.g., completion truncated away), returns a sentinel (commonly T) so downstream
        masking can zero out all completion tokens.

    Raises
    ------
    ValueError
        If offset_mapping does not have shape (B, T, 2) or boundary_chars does not have shape (B,).

    Notes
    -----
    - "Real token" is detected via offset spans with positive length (char_end > char_start), which filters out
      padding/special tokens for typical fast tokenizers.
    - The output indices are intended to be used to build a completion-only mask aligned with next-token logprobs.
    """

    starts: NDArray = offset_mapping[:, :, 0]
    real_token_mask: NDArray = offset_mapping[:, :, 1] > starts
    completion_start_tok: NDArray = real_token_mask&(starts >= boundary_chars[:, None])
    indices: NDArray = np.arange(completion_start_tok.shape[1])
    masked_indices: NDArray = np.where(
        completion_start_tok,
        indices[None, :],
        completion_start_tok.shape[1]
    )
    return masked_indices.min(axis=1)
