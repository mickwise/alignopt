"""
Purpose
-------
Tokenize (prompt, completion) text pairs using a single canonical concatenation rule and
derive per-example completion-start token indices from tokenizer offset mappings.

Key behaviors
-------------
- Builds canonical strings as: prompt + SEP + completion (SEP is a single space).
- Uses a *fast* Hugging Face tokenizer with return_offsets_mapping=True to obtain per-token
  character spans in the canonical string.
- Computes completion start token indices by comparing token char-start offsets to a
  per-example boundary character index.
- Returns NumPy arrays (input_ids, attention_mask, start_indices)
  for backend-specific casting downstream.

Conventions
-----------
- SEP is treated as belonging to the completion side; the boundary character index is len(prompt),
  not len(prompt + SEP).
- Requires a fast tokenizer (PreTrainedTokenizerFast); offset mappings must be available.
- Padding and truncation are enabled. If truncation removes all completion tokens, start_indices
  returns a sentinel value T (sequence length) meaning "no completion tokens present".

Downstream usage
----------------
Call tokenize_text(...) from backend-specific log-probability code. Use start_indices to build
a completion-only mask aligned with shifted next-token logprobs (targets input_ids[:, 1:]).
"""

from typing import Sequence, List, Tuple
from transformers import PreTrainedTokenizerFast, BatchEncoding
import numpy as np
from numpy.typing import NDArray
from alignopt.backends.backends_validation import validate_tokenizer_output

SEP: str = " "

def tokenize_text(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Tokenize a batch of (prompt, completion) pairs and compute
    per-example completion start token indices.

    Parameters
    ----------
    pairs : Sequence[Tuple[str, str]]
        Sequence of (prompt, completion) strings.
        Each pair is concatenated using the module SEP.
    tokenizer : transformers.PreTrainedTokenizerFast
        Hugging Face *fast* tokenizer used to produce input IDs,
        attention masks, and offset mappings.
    max_length : int | None
        Maximum sequence length used by the tokenizer when truncation=True.
        If None, tokenizer defaults apply.

    Returns
    -------
    Tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray]
        (input_ids, attention_mask, start_indices) where:
        - input_ids has shape (B, T) and integer dtype
        - attention_mask has shape (B, T) and dtype bool or integer 0/1
        - start_indices has shape (B,) and integer dtype, with values in [0, T]
          (T is a sentinel meaning "no completion tokens present")

    Raises
    ------
    TypeError
        If a non-fast tokenizer is provided and offset mappings cannot be produced.
    ValueError
        If validate_tokenizer_output(...) rejects shapes/dtypes/ranges of the computed outputs.

    Notes
    -----
    - This function is intentionally NumPy-only; backend-specific code should convert outputs to
      torch/jax arrays and handle device placement.
    - start_indices refer to token positions in the unshifted token sequence (input_ids), and are
      intended to be shift-corrected downstream when masking next-token logprobs.
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
    validate_tokenizer_output(input_ids, attention_mask, start_indices)
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
    Tuple[List[str], numpy.typing.NDArray]
        (texts_batch, boundary_chars) where:
        - texts_batch[i] == prompt_i + SEP + completion_i
        - boundary_chars[i] == len(prompt_i), by convention treating SEP as belonging to completion

    Raises
    ------
    None

    Notes
    -----
    - boundary_chars are character indices into the canonical string, used only with offset mappings
      to derive token indices.
    - Keep SEP and the boundary convention consistent across the project; downstream masking assumes
      this contract.
    """
    texts_batch: List[str] = []
    boundary_chars: List[int] = []
    for prompt, completion in pairs:
        texts_batch.append(prompt + SEP + completion)
        boundary_chars.append(len(prompt))
    return texts_batch, np.array(boundary_chars)


def _calculate_start_indices(offset_mapping: NDArray, boundary_chars: NDArray) -> NDArray:
    """
    Compute completion-start token indices from offset mappings
    and boundary character indices.

    Parameters
    ----------
    offset_mapping : numpy.typing.NDArray
        Array of shape (B, T, 2) containing per-token
        (char_start, char_end) offsets into the canonical string.
    boundary_chars : numpy.typing.NDArray
        Array of shape (B,) giving per-example boundary
        character indices in the canonical string.

    Returns
    -------
    numpy.typing.NDArray
        Array of shape (B,) giving completion-start token indices:
        the smallest token index t such that token t is a real (non-padding/non-special)
        token and offset_mapping[b, t, 0] >= boundary_chars[b].
        If no such token exists (e.g., completion truncated away),
        returns the sentinel T (sequence length).

    Raises
    ------
    ValueError
        If offset_mapping does not have shape (B, T, 2) or boundary_chars does not have shape (B,).

    Notes
    -----
    - "Real token" is detected via positive-length spans (char_end > char_start), which typically filters out
      padding and special tokens for fast tokenizers.
    - The returned indices are intended to be used to build a completion-only mask aligned with next-token
      logprobs in downstream code.
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
