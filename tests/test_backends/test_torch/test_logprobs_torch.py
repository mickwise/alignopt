"""
Purpose
-------
Unit tests for the PyTorch completion log-probability backend.

These tests validate that completion_logprobs_torch(...) correctly:
1) aligns logits with next-token targets (causal LM shift),
2) extracts realized token log-probabilities (token-level NLL),
3) masks out prompt tokens and padding tokens using start_positions from the text backend,
4) sums completion-only token log-probabilities per example,,
5) preserves and restores model config state (use_cache toggling),
6) remains differentiable for training use (gradient flows to model parameters).

Key behaviors
-------------
- Uses an *offline*, in-memory PreTrainedTokenizerFast (WordLevel + whitespace pre-tokenization).
- Uses deterministic dummy causal LMs that assign:
  - uniform log-prob to prompt-region next tokens (should be masked out),
  - ~0 log-prob to completion-region next tokens (should be included).
- Covers truncation removing completion tokens (sentinel start_positions => zero completion mass).
- Verifies validation behavior for invalid max_length.
- Adds device inference tests and use_cache restoration tests.

Conventions
-----------
- SEP is a single space, and the boundary convention matches alignopt.backends.text:
  boundary is at len(prompt) (SEP belongs to completion side).
- start_positions are indices in the *unshifted* sequence;
  masking is applied at (start_positions - 1)
  in token_logprobs (which is aligned to targets input_ids[:, 1:]).

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
import torch
from transformers import PreTrainedTokenizerFast

from alignopt.backends.text import tokenize_text
from alignopt.backends.torch.dpo.logprobs_torch import completion_logprobs_torch
from tests.test_backends.backend_testing_tools import make_fast_wordlevel_tokenizer


@dataclass
class _ModelOutput:
    """
    Purpose
    -------
    Minimal output container mimicking Hugging Face model outputs for testing.

    Key behaviors
    -------------
    - Stores a single attribute, logits, matching the interface
      used by completion_logprobs_torch(...).

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape (B, T, V).

    Attributes
    ----------
    logits : torch.Tensor
        The per-position vocabulary logits; used to compute token-level realized log-probabilities.

    Notes
    -----
    - Keeping this minimal reduces coupling to specific Transformers output classes.
    """
    logits: torch.Tensor


class BoundaryAwareDummyCausalLM(torch.nn.Module):
    """
    Purpose
    -------
    Deterministic dummy causal LM used to unit-test completion masking behavior.

    Key behaviors
    -------------
    - Produces logits that imply:
      - prompt-region next-token logprobs are uniform (logprob(target) = -log(V)),
      - completion-region next-token logprobs are ~0 for the realized targets
        (by making the target logit maximal).
    - Uses start_positions (unshifted token indices) to decide
      where the completion begins per example.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size V. Controls the prompt-region uniform logprob baseline (-log(V)).
    start_positions : torch.Tensor
        Tensor of shape (B,) containing the completion start
        token indices in the *unshifted* sequence.

    Attributes
    ----------
    vocab_size : int
        Cached vocabulary size used when constructing logits.
    start_positions : torch.Tensor
        Registered buffer of shape (B,) used to distinguish
        prompt vs completion positions.
    dummy : torch.nn.Parameter
        A dummy parameter to ensure the module has parameters
        (useful for device inference patterns).

    Notes
    -----
    - This model is intentionally not a real language model; it is designed so masking bugs show up
      as large, deterministic differences in the summed logprobs.
    - Causal alignment used here matches the production code: logits at position p correspond to the
      prediction for token input_ids[:, p+1].
    """

    def __init__(self, vocab_size: int, start_positions: torch.Tensor) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("start_positions", start_positions.to(dtype=torch.long, device="cpu"))
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _ModelOutput:
        b, t = input_ids.shape
        v: int = self.vocab_size
        logits: torch.Tensor = torch.zeros((b, t, v), dtype=torch.float32, device=input_ids.device)

        start_pos: torch.Tensor = self.start_positions.to(device=input_ids.device)

        for b_i in range(b):
            for p in range(t - 1):
                target_id: int = int(input_ids[b_i, p + 1].item())
                is_real: bool = bool(attention_mask[b_i, p + 1].item())
                is_completion_target: bool = (p + 1) >= int(start_pos[b_i].item())

                if is_real and is_completion_target:
                    logits[b_i, p, :] = -1000.0
                    logits[b_i, p, target_id] = 0.0
                else:
                    pass

        return _ModelOutput(logits=logits)


class _Config:
    """
    Purpose
    -------
    Minimal config object used to emulate Hugging Face model.config in tests.

    Attributes
    ----------
    use_cache : bool
        Whether the model uses KV caching in forward calls.
    """

    def __init__(self, use_cache: bool) -> None:
        self.use_cache = use_cache


class CacheAwareDummyCausalLM(BoundaryAwareDummyCausalLM):
    """
    Purpose
    -------
    Dummy causal LM that also exposes a HF-like `config.use_cache`
    attribute for cache-toggle testing.

    Key behaviors
    -------------
    - Exposes `self.config.use_cache`.
    - Records the observed value of config.use_cache at forward-time to validate toggle semantics.

    Notes
    -----
    - The production backend toggles `model.config.use_cache` to False (if present and True)
      during logprob computation and restores it in a finally block.
    """

    def __init__(self, vocab_size: int, start_positions: torch.Tensor, use_cache: bool) -> None:
        super().__init__(vocab_size=vocab_size, start_positions=start_positions)
        self.config = _Config(use_cache=use_cache)
        self.observed_use_cache_values: List[bool] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _ModelOutput:
        self.observed_use_cache_values.append(bool(self.config.use_cache))
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


class CacheErrorDummyCausalLM(CacheAwareDummyCausalLM):
    """
    Purpose
    -------
    Dummy causal LM that raises during forward to validate cache restoration on exceptions.

    Key behaviors
    -------------
    - Raises RuntimeError in forward after recording observed config.use_cache.
    """

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _ModelOutput:
        self.observed_use_cache_values.append(bool(self.config.use_cache))
        raise RuntimeError("Intentional forward failure for cache restoration test.")


class TinyDifferentiableCausalLM(torch.nn.Module):
    """
    Purpose
    -------
    Small differentiable causal LM to validate gradient flow through completion_logprobs_torch.

    Key behaviors
    -------------
    - Uses an embedding + linear head to produce logits of shape (B, T, V).
    - Provides a HF-like `.logits` output container.

    Notes
    -----
    - This is not a realistic language model; it's a minimal differentiable mapping.
    - The test uses a small corpus/tokenizer so vocab indices are in-range.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _ModelOutput:
        _ = attention_mask
        h: torch.Tensor = self.embed(input_ids)
        logits: torch.Tensor = self.lm_head(h)
        return _ModelOutput(logits=logits)


def _pick_accelerator_device() -> torch.device | None:
    """
    Purpose
    -------
    Pick an accelerator device for tests if available.

    Returns
    -------
    torch.device | None
        CUDA if available, else MPS if available, else None.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_completion_logprobs_masks_prompt_tokens() -> None:
    """
    Validate that completion_logprobs_torch(...) excludes prompt-region
    next-token logprobs from the sum.

    Notes
    -----
    - Uses a single pair where the prompt has 2 tokens and completion has 1 token.
    - Dummy model assigns ~0 logprob to the completion target and -log(V) to prompt targets.
      Correct masking yields ~0; incorrect masking yields approximately -log(V).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16
    )
    start_positions = torch.from_numpy(start_positions)

    vocab_size: int = tokenizer.vocab_size
    model = BoundaryAwareDummyCausalLM(vocab_size=vocab_size, start_positions=start_positions)

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.shape == (1,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    expected_if_wrong = torch.tensor([-float(np.log(vocab_size))], dtype=out.dtype, device=out.device)
    assert (out - expected_if_wrong).abs().item() > 1e-2


def test_completion_logprobs_respects_padding_mask_in_batch() -> None:
    """
    Validate that padding tokens do not contribute to completion_logprobs_torch(...)
    in a mixed-length batch.

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
        max_length=16
    )

    start_positions = torch.from_numpy(start_positions)

    vocab_size: int = tokenizer.vocab_size
    model = BoundaryAwareDummyCausalLM(vocab_size=vocab_size, start_positions=start_positions)

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_completion_logprobs_truncation_removes_completion_yields_zero() -> None:
    """
    Validate that truncation removing all completion tokens yields a zero completion logprob sum.

    Notes
    -----
    - With small max_length, the tokenized sequence contains only prompt tokens.
    - The text backend should return a sentinel start_positions (typically T)
      indicating no completion tokens.
    - The completion mask should then exclude all token_logprobs entries.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=2
    )

    start_positions = torch.from_numpy(start_positions)

    vocab_size: int = tokenizer.vocab_size
    model = BoundaryAwareDummyCausalLM(vocab_size=vocab_size, start_positions=start_positions)

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=2,
    )

    assert out.shape == (1,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_completion_logprobs_invalid_max_length_raises() -> None:
    """
    Validate that completion_logprobs_torch(...) enforces max_length validation.

    Notes
    -----
    - This test checks integration: completion_logprobs_torch(...)
      must call validate_logprobs_inputs(...).
    - Tokenization for model construction uses a valid max_length;
      the invalid value is passed only to completion_logprobs_torch(...).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello", "moon")]
    corpus: List[str] = ["hello moon"]
    tokenizer: PreTrainedTokenizerFast = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=8
    )

    start_positions = torch.from_numpy(start_positions)

    vocab_size: int = tokenizer.vocab_size
    model = BoundaryAwareDummyCausalLM(vocab_size=vocab_size, start_positions=start_positions)

    with pytest.raises(ValueError):
        _ = completion_logprobs_torch(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            max_length=0,
        )


def test_device_none_uses_model_parameter_device_cpu() -> None:
    """
    Validate that device=None uses the device of model parameters,
    and outputs are on that device.

    Notes
    -----
    - This ensures the "device inference" path behaves predictably.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16
    )

    start_positions = torch.from_numpy(start_positions)

    model = BoundaryAwareDummyCausalLM(vocab_size=tokenizer.vocab_size, start_positions=start_positions)
    model.to(torch.device("cpu"))

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.device.type == "cpu"


@pytest.mark.skipif(_pick_accelerator_device() is None, reason="No accelerator device available.")
def test_device_none_uses_model_parameter_device_accelerator() -> None:
    """
    Validate that device=None uses the device of model parameters on an accelerator (CUDA/MPS).

    Notes
    -----
    - Skipped automatically if no accelerator is available.
    """
    accel = _pick_accelerator_device()
    assert accel is not None

    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16
    )

    start_positions = torch.from_numpy(start_positions)

    model = BoundaryAwareDummyCausalLM(vocab_size=tokenizer.vocab_size, start_positions=start_positions)
    model.to(accel)

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.device.type == accel.type


def test_use_cache_toggled_off_during_forward_and_restored() -> None:
    """
    Validate that when model.config.use_cache is True, it is set to False for the forward,
    and restored afterward.

    Notes
    -----
    - This checks the intended optimization/safety behavior of disabling KV caching for logprob passes.
    - The dummy model records the observed config.use_cache values at forward-time.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16
    )

    start_positions = torch.from_numpy(start_positions)

    model = CacheAwareDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=start_positions,
        use_cache=True,
    )

    assert model.config.use_cache is True

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.shape == (1,)
    assert model.observed_use_cache_values, "Forward was not executed."
    assert all(v is False for v in model.observed_use_cache_values), "use_cache was not disabled during forward."
    assert model.config.use_cache is True, "use_cache was not restored after completion_logprobs_torch."


def test_use_cache_restored_even_if_forward_raises() -> None:
    """
    Validate that model.config.use_cache is restored even when model forward raises.

    Notes
    -----
    - This validates the try/finally restoration path.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16
    )

    start_positions = torch.from_numpy(start_positions)

    model = CacheErrorDummyCausalLM(
        vocab_size=tokenizer.vocab_size,
        start_positions=start_positions,
        use_cache=True,
    )

    assert model.config.use_cache is True

    with pytest.raises(RuntimeError, match="Intentional forward failure"):
        _ = completion_logprobs_torch(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            max_length=16,
        )

    assert model.observed_use_cache_values, "Forward was not executed."
    assert all(v is False for v in model.observed_use_cache_values), "use_cache was not disabled during forward."
    assert model.config.use_cache is True, "use_cache was not restored after exception."


def test_completion_logprobs_is_differentiable_and_propagates_gradients() -> None:
    """
    Validate that completion_logprobs_torch(...) is differentiable and propagates gradients.

    Notes
    -----
    - Uses a tiny differentiable model (embedding + linear head).
    - Confirms that at least one parameter receives a non-zero gradient.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon"), ("hello", "moon")]
    corpus: List[str] = ["hello world moon", "hello moon"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    model = TinyDifferentiableCausalLM(vocab_size=tokenizer.vocab_size, hidden_size=8)
    model.train()

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    loss: torch.Tensor = -out.mean()
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach())

    assert grads, "No gradients were produced."
    total_norm = torch.stack([g.abs().sum() for g in grads]).sum().item()
    assert total_norm > 0.0, "Gradients are all zero; expected non-zero gradient flow."


def test_completion_logprobs_dtype_is_float_tensor() -> None:
    """
    Validate that the output dtype is a floating tensor suitable for downstream losses.

    Notes
    -----
    - Ensures output is floating (not integer/bool).
    - Useful when integrating with training code that expects float log-probs.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer = make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16
    )

    start_positions = torch.from_numpy(start_positions)

    model = BoundaryAwareDummyCausalLM(vocab_size=tokenizer.vocab_size, start_positions=start_positions)

    out: torch.Tensor = completion_logprobs_torch(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        max_length=16,
    )

    assert out.dtype.is_floating_point
    assert out.ndim == 1
    assert out.shape == (1,)
