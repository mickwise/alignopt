from math import isfinite
from typing import Tuple, Set, Sequence
import numpy as np
from numpy.typing import NDArray

def validate_logprobs_inputs(
        pairs: Sequence[Tuple[str, str]],
        max_length: int | None
    ) -> None:
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("Max length should be positive.")
    if len(pairs) == 0:
        raise ValueError("There must be at least one pair in the batch.")


def validate_tokenizer_output(
        input_ids: NDArray,
        attention_mask: NDArray,
        start_indices: NDArray
    ) -> None:
    """
    Validate the shapes and dtypes of tokenizer outputs used by downstream log-probability code.

    Parameters
    ----------
    input_ids : numpy.typing.NDArray
        Integer token IDs of shape (B, T) produced by a tokenizer.
    attention_mask : numpy.typing.NDArray
        Mask of shape (B, T) indicating which positions are real
        tokens (1/True) vs padding (0/False).
    start_indices : numpy.typing.NDArray
        Integer array of shape (B,) giving the per-example
        completion start token index in the unshifted sequence.
        Values must lie in [0, T], where T is allowed as a
        sentinel meaning "no completion tokens present".

    Returns
    -------
    None
        Returns None if all validations pass.

    Raises
    ------
    ValueError
        If any output has an unexpected shape, dtype, or contains
        invalid values (e.g., start indices out of range,
        or integer attention masks containing values other than 0/1).

    Notes
    -----
    - This validation is purely structural (shapes/dtypes/ranges)
      and does not inspect tokenizer semantics such as boundary conventions.
    - The sentinel start index value T is permitted to represent "completion truncated away".
    """

    batch_size: int = input_ids.shape[0]
    sequence_length: int = input_ids.shape[1]
    if not np.issubdtype(start_indices.dtype, np.integer):
        raise ValueError("Completion starting positions must be integers.")
    if start_indices.shape != (batch_size,):
        raise ValueError("There must be one start index for every pair in the batch.")
    if np.any((start_indices > sequence_length) | (start_indices < 0)):
        raise ValueError("Completion starting positions must be legal indices.")
    if not np.issubdtype(input_ids.dtype, np.integer):
        raise ValueError("Input ids must be integers.")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("Input ids and the attention mask must have the same shape.")
    if np.issubdtype(attention_mask.dtype, np.integer):
        if not np.isin(attention_mask, (0, 1)).all():
            raise ValueError("If attention mask holds integers, they must be either 0 or 1.")
    elif not np.issubdtype(attention_mask.dtype, np.bool_):
        raise ValueError("Attention mask must hold integers or booleans.")

