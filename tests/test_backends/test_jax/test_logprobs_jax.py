"""
Purpose
-------
Unit tests for the JAX/Flax completion log-probability backend.

These tests validate that completion_logprobs(...) correctly:
1) aligns logits with next-token targets (causal LM shift),
2) computes realized token log-probabilities via token-level NLL (Optax cross-entropy),
3) masks out prompt tokens and padding tokens using start_positions from the text backend,
4) sums completion-only token log-probabilities per example,
5) forces deterministic inference behavior by passing train=False to the model call,
6) remains differentiable for training use (gradient flows to model parameters),
7) returns a 1D floating output suitable for downstream losses.

Key behaviors
-------------
- Uses an *offline*, in-memory PreTrainedTokenizerFast (WordLevel + whitespace pre-tokenization).
- Uses deterministic dummy Flax-like causal LMs that assign:
  - uniform log-prob to prompt-region next tokens (should be masked out),
  - ~0 log-prob to completion-region next tokens (should be included).
- Covers truncation removing completion tokens (sentinel start_positions => zero completion mass).
- Verifies validation behavior for invalid max_length.

Conventions
-----------
- SEP is a single space, and the boundary convention matches alignopt.backends.text:
  boundary is at len(prompt) (SEP belongs to completion side).
- start_positions are indices in the *unshifted* sequence.
- token_logprobs are aligned to targets input_ids[:, 1:]; masking is applied at positions >= start_positions.

Downstream usage
----------------
Run with:
    pytest -q

Adjust the import of the module under test if your package path differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import PreTrainedTokenizerFast

from alignopt.backends.text import tokenize_text
from alignopt.backends.jax.dpo.logprobs_jax import completion_logprobs
from tests.test_backends.backend_testing_tools import make_fast_wordlevel_tokenizer


@dataclass
class _ModelOutput:
    """
    Purpose
    -------
    Minimal output container mimicking Hugging Face Flax model outputs for testing.

    Key behaviors
    -------------
    - Stores a single attribute, logits, matching the interface used by completion_logprobs(...).

    Parameters
    ----------
    logits : jax.Array
        Logits tensor of shape (B, T, V).

    Attributes
    ----------
    logits : jax.Array
        The per-position vocabulary logits; used to compute token-level realized log-probabilities.

    Notes
    -----
    - Keeping this minimal reduces coupling to specific Transformers output classes.
    """
    logits: jax.Array


class BoundaryAwareDummyCausalLM:
    """
    Purpose
    -------
    Deterministic dummy causal LM used to unit-test completion masking behavior.

    Key behaviors
    -------------
    - Produces logits that imply:
      - prompt/padding-region next-token logprobs are uniform (logprob(target) = -log(V)),
      - completion-region next-token logprobs are ~0 for the realized targets
        (by making the target logit maximal).
    - Uses start_positions (unshifted token indices) to decide where the completion begins per example.
    - Uses attention_mask to exclude padding targets.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size V. Controls the prompt-region uniform logprob baseline (-log(V)).
    start_positions : jax.Array
        Array of shape (B,) containing completion start token indices in the *unshifted* sequence.

    Attributes
    ----------
    vocab_size : int
        Cached vocabulary size used when constructing logits.
    start_positions : jax.Array
        Cached start positions used to distinguish prompt vs completion.

    Notes
    -----
    - This is intentionally not a real language model; it is designed so masking bugs show up
      as large, deterministic differences in summed logprobs.
    - Causal alignment matches production code: logits at position p correspond to prediction
      for token input_ids[:, p+1].
    """

    def __init__(self, vocab_size: int, start_positions: jax.Array) -> None:
        self.vocab_size = int(vocab_size)
        self.start_positions = jnp.asarray(start_positions, dtype=jnp.int32)

    def __call__(
        self,
        *,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        params: FrozenDict,
        train: bool = False,
    ) -> _ModelOutput:
        _ = params
        _ = train

        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        attention_mask = jnp.asarray(attention_mask)
        b, t = input_ids.shape
        v = self.vocab_size

        targets: jax.Array = input_ids[:, 1:]
        positions: jax.Array = jnp.arange(1, t)

        completion_target_mask: jax.Array = positions[None, :] >= self.start_positions[:, None]
        real_target_mask: jax.Array = attention_mask[:, 1:].astype(bool)
        mask: jax.Array = completion_target_mask & real_target_mask

        base_logits: jax.Array = jnp.where(mask[:, :, None], -1000.0, 0.0)
        bump: jax.Array = jnp.where(
            mask[:, :, None],
            jax.nn.one_hot(targets, v, dtype=jnp.float32) * 1000.0,
            0.0,
        )
        logits_pred: jax.Array = base_logits + bump

        logits_last: jax.Array = jnp.zeros((b, 1, v), dtype=jnp.float32)
        logits: jax.Array = jnp.concatenate([logits_pred, logits_last], axis=1)

        return _ModelOutput(logits=logits)


class TrainSensitiveBoundaryDummyCausalLM(BoundaryAwareDummyCausalLM):
    """
    Purpose
    -------
    Dummy causal LM that makes train-flag handling testable.

    Key behaviors
    -------------
    - If train=False: behaves like BoundaryAwareDummyCausalLM (completion targets get ~0 logprob).
    - If train=True: returns uniform logits everywhere (so completion sum is negative).

    Notes
    -----
    - This lets tests fail loudly if production code forgets to pass train=False.
    """

    def __call__(
        self,
        *,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        params: FrozenDict,
        train: bool = False,
    ) -> _ModelOutput:
        if train:
            b, t = jnp.asarray(input_ids).shape
            v = self.vocab_size
            logits = jnp.zeros((b, t, v), dtype=jnp.float32)
            return _ModelOutput(logits=logits)
        return super().__call__(input_ids=input_ids, attention_mask=attention_mask, params=params, train=train)


class TinyDifferentiableCausalLM:
    """
    Purpose
    -------
    Minimal differentiable Flax-like causal LM to validate gradient flow through completion_logprobs(...).

    Key behaviors
    -------------
    - Computes logits as one_hot(input_ids) @ W, where W is a learnable (V, V) matrix.
    - Accepts HF-like call signature: params=..., train=....

    Parameters
    ----------
    vocab_size : int
        Vocabulary size V.

    Attributes
    ----------
    vocab_size : int
        Cached vocabulary size used to build one-hot inputs.

    Notes
    -----
    - Not a realistic LM; it is a minimal differentiable mapping with an obvious gradient signal.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)

    def __call__(
        self,
        *,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        params: FrozenDict,
        train: bool = False,
    ) -> _ModelOutput:
        _ = attention_mask
        _ = train

        v = self.vocab_size
        x = jax.nn.one_hot(jnp.asarray(input_ids, dtype=jnp.int32), v, dtype=jnp.float32)
        w = params["W"]
        logits = jnp.einsum("btv,vk->btk", x, w)
        return _ModelOutput(logits=logits)


def test_completion_logprobs_masks_prompt_tokens() -> None:
    """
    Validate that completion_logprobs(...) excludes prompt-region next-token logprobs from the sum.

    Notes
    -----
    - Uses a single pair where the prompt has 2 tokens and completion has 1 token.
    - Dummy model assigns ~0 logprob to completion targets and -log(V) to prompt targets.
      Correct masking yields ~0; incorrect masking yields approximately -log(V).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
    )

    model = BoundaryAwareDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=jnp.asarray(start_positions),
    )

    out: jax.Array = completion_logprobs(
        model=model,
        params=FrozenDict({}),
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.shape == (1,)
    assert jnp.allclose(out, jnp.zeros_like(out), atol=1e-6)

    expected_if_wrong = -float(np.log(tokenizer.vocab_size))
    assert float(jnp.abs(out[0] - expected_if_wrong)) > 1e-2


def test_completion_logprobs_respects_padding_mask_in_batch() -> None:
    """
    Validate that padding tokens do not contribute to completion_logprobs(...) in a mixed-length batch.

    Notes
    -----
    - Two examples are tokenized together; the shorter one receives padding.
    - Correct behavior: padding targets are excluded via attention_mask[:, 1:] alignment.
    """
    pairs: Sequence[Tuple[str, str]] = [
        ("hello world", "moon"),
        ("hello", "moon"),
    ]
    corpus: List[str] = ["hello world moon", "hello moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
    )

    model = BoundaryAwareDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=jnp.asarray(start_positions),
    )

    out: jax.Array = completion_logprobs(
        model=model,
        params=FrozenDict({}),
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.zeros_like(out), atol=1e-6)


def test_completion_logprobs_truncation_removes_completion_yields_zero() -> None:
    """
    Validate that truncation removing all completion tokens yields a zero completion logprob sum.

    Notes
    -----
    - With small max_length, the tokenized sequence contains only prompt tokens.
    - The text backend should return a sentinel start_positions == T indicating no completion tokens.
    - The completion mask should then exclude all token_logprobs entries.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    input_ids, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=2,
    )

    t = int(np.asarray(input_ids).shape[1])
    assert int(np.asarray(start_positions)[0]) == t

    model = BoundaryAwareDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=jnp.asarray(start_positions),
    )

    out: jax.Array = completion_logprobs(
        model=model,
        params=FrozenDict({}),
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=2,
    )

    assert out.shape == (1,)
    assert jnp.allclose(out, jnp.zeros_like(out), atol=1e-6)


def test_completion_logprobs_invalid_max_length_raises() -> None:
    """
    Validate that completion_logprobs(...) enforces max_length validation.

    Notes
    -----
    - This test checks integration: completion_logprobs(...) must call validate_logprobs_inputs(...).
    - The invalid value is passed only to completion_logprobs(...).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello", "moon")]
    corpus: List[str] = ["hello moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=8,
    )

    model = BoundaryAwareDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=jnp.asarray(start_positions),
    )

    with pytest.raises(ValueError):
        _ = completion_logprobs(
            model=model,
            params=FrozenDict({}),
            tokenizer=tokenizer,
            pairs=pairs,
            max_length=0,
        )


def test_completion_logprobs_passes_train_false_to_model() -> None:
    """
    Validate that completion_logprobs(...) forces deterministic inference by passing train=False.

    Notes
    -----
    - TrainSensitiveBoundaryDummyCausalLM returns uniform logits when train=True,
      which would make completion sums negative.
    - When train=False, it behaves like the boundary-aware dummy and yields ~0.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
    )

    model = TrainSensitiveBoundaryDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=jnp.asarray(start_positions),
    )

    out: jax.Array = completion_logprobs(
        model=model,
        params=FrozenDict({}),
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.shape == (1,)
    assert jnp.allclose(out, jnp.zeros_like(out), atol=1e-6)


def test_completion_logprobs_is_differentiable_and_propagates_gradients() -> None:
    """
    Validate that completion_logprobs(...) is differentiable and propagates gradients.

    Notes
    -----
    - Uses a tiny differentiable model (one-hot @ W).
    - Confirms that at least one parameter entry in W receives a non-zero gradient.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon"), ("hello", "moon")]
    corpus: List[str] = ["hello world moon", "hello moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    model = TinyDifferentiableCausalLM(vocab_size=tokenizer.vocab_size)
    v = tokenizer.vocab_size
    w0 = jnp.zeros((v, v), dtype=jnp.float32)

    def loss_fn(w: jax.Array) -> jax.Array:
        params: FrozenDict = FrozenDict({"W": w})
        out = completion_logprobs(
            model=model,
            params=params,
            tokenizer=tokenizer,
            pairs=pairs,
            max_length=16,
        )
        return -jnp.mean(out)

    grad_w = jax.grad(loss_fn)(w0)
    total = jnp.sum(jnp.abs(grad_w))
    assert float(total) > 0.0


def test_completion_logprobs_output_dtype_is_floating() -> None:
    """
    Validate that the output dtype is floating and the output shape is 1D (B,).

    Notes
    -----
    - Useful when integrating with training code that expects float log-probs.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
    )

    model = BoundaryAwareDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=jnp.asarray(start_positions),
    )

    out: jax.Array = completion_logprobs(
        model=model,
        params=FrozenDict({}),
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.ndim == 1
    assert out.shape == (1,)
    assert jnp.issubdtype(out.dtype, jnp.floating)
