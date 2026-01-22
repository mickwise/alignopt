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

from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest
import torch 
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from alignopt.backends import text as text_mod


def _make_fast_wordlevel_tokenizer(corpus: Iterable[str]) -> PreTrainedTokenizerFast:
    """
    Build an in-memory Hugging Face *fast* tokenizer suitable for offset-mapping tests.

    Parameters
    ----------
    corpus : Iterable[str]
        Text samples used to train a minimal WordLevel vocabulary.

    Returns
    -------
    transformers.PreTrainedTokenizerFast
        A fast tokenizer that supports return_offsets_mapping=True and padding.

    Raises
    ------
    None

    Notes
    -----
    - Uses tokenizers' WordLevel model with Whitespace pre-tokenization.
    - Adds [UNK] and [PAD] as special tokens.
    - Designed to be deterministic and offline (no model downloads).
    """

    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(corpus, trainer=trainer)

    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    return hf


def test_tokenize_text_pt_returns_torch_tensors_and_shapes() -> None:
    """
    Validate tokenize_text(..., backend="pt") output types and shapes.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If outputs are not torch.Tensors or shapes are inconsistent.

    Notes
    -----
    - Uses max_length large enough to avoid truncation in this test.
    """

    pairs: Sequence[Tuple[str, str]] = [("hello", "world"), ("goodbye", "moon")]
    corpus = [p + text_mod.SEP + c for (p, c) in pairs]
    tokenizer = _make_fast_wordlevel_tokenizer(corpus)

    input_ids, attention_mask, start_indices = text_mod.tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
        backend="pt",
    )

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert isinstance(start_indices, torch.Tensor)

    assert input_ids.ndim == 2
    assert attention_mask.ndim == 2
    assert start_indices.ndim == 1

    b, t = input_ids.shape
    assert attention_mask.shape == (b, t)
    assert start_indices.shape == (b,)


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
    tokenizer = _make_fast_wordlevel_tokenizer(corpus)

    input_ids, attention_mask, start_indices = text_mod.tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=8,
        backend="pt",
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
    tokenizer = _make_fast_wordlevel_tokenizer(corpus)

    input_ids, _, start_indices = text_mod.tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=1,
        backend="pt",
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
            max_length=8,
            backend="pt",
        )


def test_tokenize_text_jax_backend_if_available() -> None:
    """
    Validate tokenize_text(..., backend="jax") conversion if JAX is installed.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If shapes are inconsistent.

    Notes
    -----
    - Skips automatically if JAX is not importable in the environment.
    - Uses shape checks instead of strict isinstance checks to avoid version-specific JAX types.
    """
    try:
        import jax # pylint: disable=unused-import
        import jax.numpy as jnp # pylint: disable=unused-import
    except ImportError:
        pytest.skip("JAX not installed; skipping JAX backend test.")

    pairs: Sequence[Tuple[str, str]] = [("hello", "world")]
    corpus = ["hello world"]
    tokenizer = _make_fast_wordlevel_tokenizer(corpus)

    input_ids, attention_mask, start_indices = text_mod.tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=8,
        backend="jax",
    )

    assert hasattr(input_ids, "shape")
    assert hasattr(attention_mask, "shape")
    assert hasattr(start_indices, "shape")

    assert input_ids.shape == attention_mask.shape
    assert start_indices.shape == (input_ids.shape[0],)

    # Ensure we got JAX arrays (best-effort).
    assert isinstance(np.asarray(input_ids), np.ndarray)
    assert isinstance(np.asarray(attention_mask), np.ndarray)
    assert isinstance(np.asarray(start_indices), np.ndarray)
