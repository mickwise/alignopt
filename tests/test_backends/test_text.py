"""
Purpose
-------
Unit tests for alignopt.backends.text.

These tests validate that tokenize_text(...) and its boundary-to-token-index logic behave
correctly under a deterministic, locally-constructed *fast* tokenizer (no network calls).

Key behaviors
-------------
- Builds a local PreTrainedTokenizerFast (WordLevel + whitespace pre-tokenization) in-memory.
- Verifies canonical concatenation and boundary convention (boundary at len(prompt)).
- Verifies start_indices semantics:
  - points to the first completion token (in token space)
  - returns a sentinel == T when truncation removes the completion region
- Verifies backend conversion behavior (PyTorch always; JAX if installed).
- Verifies non-fast tokenizer rejection.

Conventions
-----------
- Tests assume the project convention that SEP belongs to the completion side:
  boundary_chars == len(prompt) (not len(prompt + SEP)).
- start_indices are token indices in the unshifted token sequence (B, T).
- The sentinel is T (sequence length after padding/truncation) when no completion token exists.

Downstream usage
----------------
Run with:
    pytest -q

These tests are intended to catch "silent wrong" boundary/masking mismatches early, before
they contaminate DPO/PPO loss computations.
"""

from __future__ import annotations
from typing import Sequence, Tuple
import pytest
from alignopt.backends import text as text_mod
from tests.test_backends.backend_testing_tools import make_fast_wordlevel_tokenizer

def test_start_indices_points_to_first_completion_token_in_simple_case() -> None:
    """
    Validate start_indices semantics in a simple two-token example.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the computed start index does not align with the completion boundary.

    Notes
    -----
    - With a whitespace WordLevel tokenizer, "hello world" tokenizes into ["hello", "world"].
    - Boundary is len("hello") == 5; completion begins at "world".
    - Expected start index is 1 (0-based).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello", "world")]
    corpus = ["hello world"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    input_ids, attention_mask, start_indices = text_mod.tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=8
    )

    # Expected: ["hello", "world", PAD, ...] => completion starts at token index 1.
    assert int(start_indices[0].item()) == 1

    # Sanity: attention mask should mark at least the two real tokens.
    assert int(attention_mask[0].sum().item()) >= 2
    assert input_ids.shape[1] == attention_mask.shape[1]


def test_start_indices_returns_sentinel_when_truncation_removes_completion() -> None:
    """
    Validate sentinel behavior when completion tokens are truncated away.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If start_indices does not equal the sequence length T when no completion token exists.

    Notes
    -----
    - With max_length=1, only the first token can remain.
    - If completion begins at token index 1, it will be removed.
    - Module contract: start_indices should be sentinel == T in that case.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello", "world")]
    corpus = ["hello world"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    input_ids, _, start_indices = text_mod.tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=1
    )

    t = input_ids.shape[1]
    assert t == 1
    assert int(start_indices[0].item()) == t


def test_tokenize_text_rejects_non_fast_tokenizer() -> None:
    """
    Validate that tokenize_text rejects non-fast tokenizers.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If TypeError is not raised for non-PreTrainedTokenizerFast instances.

    Notes
    -----
    - This test passes a dummy object to trigger the isinstance check.
    """
    class DummyTokenizer:
        pass

    pairs: Sequence[Tuple[str, str]] = [("x", "y")]

    with pytest.raises(TypeError):
        text_mod.tokenize_text(
            pairs=pairs,
            tokenizer=DummyTokenizer(),  # type: ignore[arg-type]
            max_length=8
        )
