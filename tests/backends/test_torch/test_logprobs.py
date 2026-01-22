"""
Purpose
-------
Unit tests for the PyTorch completion log-probability backend.

These tests validate that completion_logprobs(...) correctly:
1) aligns logits with next-token targets (causal LM shift),
2) extracts realized token log-probabilities (gather),
3) masks out prompt tokens and padding tokens using start_positions from the text backend,
4) sums completion-only token log-probabilities per example.

Key behaviors
-------------
- Uses an *offline*, in-memory PreTrainedTokenizerFast (WordLevel + whitespace pre-tokenization).
- Uses a deterministic dummy causal LM that assigns:
  - uniform log-prob to prompt-region next tokens (should be masked out),
  - ~0 log-prob to completion-region next tokens (should be included).
- Covers truncation removing completion tokens (sentinel start_positions => zero completion mass).
- Verifies validation behavior for invalid max_length.

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
from typing import Iterable, Sequence, Tuple, List

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from alignopt.backends.torch.logprobs import completion_logprobs
from alignopt.backends.text import tokenize_text


def _make_fast_wordlevel_tokenizer(corpus: Iterable[str]) -> PreTrainedTokenizerFast:
    """
    Build an offline, in-memory Hugging Face *fast* tokenizer for offset-mapping tests.

    Parameters
    ----------
    corpus : Iterable[str]
        Text samples used to train a minimal WordLevel vocabulary.

    Returns
    -------
    transformers.PreTrainedTokenizerFast
        A fast tokenizer configured with:
        - WordLevel model with an [UNK] token,
        - whitespace pre-tokenization,
        - [PAD] token for padding,
        - support for return_offsets_mapping=True.

    Raises
    ------
    ValueError
        If the underlying tokenizer training fails (e.g., empty corpus).

    Notes
    -----
    - This avoids any network/model download and is fully deterministic given the corpus.
    - Offsets are critical: alignopt.backends.text.tokenize_text(...) depends on offset mappings
      to derive completion start token indices.
    """

    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tok.train_from_iterator(corpus, trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )


@dataclass
class _ModelOutput:
    """
    Purpose
    -------
    Minimal output container mimicking Hugging Face model outputs for testing.

    Key behaviors
    -------------
    - Stores a single attribute, logits, matching the interface used by completion_logprobs(...).

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape (B, T, V).

    Attributes
    ----------
    logits : torch.Tensor
        The per-position vocabulary logits; used to compute log-softmax and gather realized token logprobs.

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
    - completion-region next-token logprobs are ~0 for the realized targets (by making the target logit maximal).
    - Uses start_positions (unshifted token indices) to decide where the completion begins per example.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size V. Controls the prompt-region uniform logprob baseline (-log(V)).
    start_positions : torch.Tensor
        Tensor of shape (B,) containing the completion start token indices in the *unshifted* sequence.

    Attributes
    ----------
    vocab_size : int
        Cached vocabulary size used when constructing logits.
    start_positions : torch.Tensor
        Registered buffer of shape (B,) used to distinguish prompt vs completion positions.
    dummy : torch.nn.Parameter
        A dummy parameter to ensure the module has parameters (useful for device inference patterns).

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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> _ModelOutput:
        """
        Run a deterministic forward pass producing logits that encode prompt vs completion behavior.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (B, T).
        attention_mask : torch.Tensor
            Attention mask of shape (B, T), with 1 for real tokens and 0 for padding.

        Returns
        -------
        _ModelOutput
            Object containing:
            - logits : torch.Tensor of shape (B, T, V)

        Raises
        ------
        ValueError
            If input_ids and attention_mask have incompatible shapes.

        Notes
        -----
        - For each position p in [0, T-2], we treat logits[b, p, :] as predicting the target token
          input_ids[b, p+1].
        - For completion targets (p+1 >= start_positions[b]) with is_real==1, logits strongly favor
          the realized target so logprob(target) ~ 0.
        - For prompt-region or padding-region targets, logits are all zeros so logprob(target) = -log(V).
        """
        b, t = input_ids.shape
        v: int = self.vocab_size
        logits: torch.Tensor = torch.zeros((b, t, v), dtype=torch.float32, device=input_ids.device)

        start_pos: torch.Tensor = self.start_positions.to(device=input_ids.device)

        for b_i in range(b):
            for p in range(t - 1):
                target_id: int = input_ids[b_i, p + 1].item()
                is_real: bool = attention_mask[b_i, p + 1].item()
                is_completion_target: torch.Tensor = (p + 1) >= int(start_pos[b_i].item())

                if is_real and is_completion_target:
                    logits[b_i, p, :] = -1000.0
                    logits[b_i, p, target_id] = 0.0
                else:
                    # prompt-region or padding-region: keep uniform zeros
                    pass

        return _ModelOutput(logits=logits)


def test_completion_logprobs_masks_prompt_tokens() -> None:
    """
    Validate that completion_logprobs(...) excludes prompt-region next-token logprobs from the sum.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts that the computed completion-only sum is ~0 for a single example.

    Raises
    ------
    AssertionError
        If prompt tokens leak into the completion sum (e.g., wrong boundary convention or shift alignment).

    Notes
    -----
    - Uses a single pair where the prompt has 2 tokens and completion has 1 token.
    - Dummy model assigns ~0 logprob to the completion target and -log(V) to prompt targets.
      Correct masking yields ~0; incorrect masking yields approximately -log(V).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = _make_fast_wordlevel_tokenizer(corpus)

    # Get start_positions consistent with the text backend.
    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
        backend="pt",
    )
    vocab_size: int = tokenizer.vocab_size
    model: BoundaryAwareDummyCausalLM = BoundaryAwareDummyCausalLM(
        vocab_size=vocab_size, start_positions=start_positions
    )

    out: _ModelOutput = completion_logprobs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=torch.device("cpu"),
        max_length=16,
    )

    assert out.shape == (1,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    # Sanity: if prompt token were included, we'd see about -log(V).
    expected_if_wrong = torch.tensor([-float(np.log(vocab_size))], dtype=out.dtype)
    assert (out - expected_if_wrong).abs().item() > 1e-2


def test_completion_logprobs_respects_padding_mask_in_batch() -> None:
    """
    Validate that padding tokens do not contribute to completion_logprobs(...) in a mixed-length batch.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts that both examples in the batch produce ~0 completion-only sums.

    Raises
    ------
    AssertionError
        If padding positions are not masked (e.g., attention_mask misalignment with shifted targets).

    Notes
    -----
    - Two examples are tokenized together; the shorter one receives padding.
    - Correct behavior: padding targets are excluded via attention_mask[:, 1:] alignment.
    """
    pairs: Sequence[Tuple[str, str]] = [
        ("hello world", "moon"),     # longer
        ("hello", "moon"),           # shorter => more padding after tokenization
    ]
    corpus: List[str] = ["hello world moon", "hello moon"]
    tokenizer: PreTrainedTokenizerFast = _make_fast_wordlevel_tokenizer(corpus)

    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=16,
        backend="pt",
    )
    vocab_size: int = tokenizer.vocab_size
    model: BoundaryAwareDummyCausalLM = BoundaryAwareDummyCausalLM(
        vocab_size=vocab_size, start_positions=start_positions
    )

    out: _ModelOutput = completion_logprobs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=torch.device("cpu"),
        max_length=16,
    )

    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_completion_logprobs_truncation_removes_completion_yields_zero() -> None:
    """
    Validate that truncation removing all completion tokens yields a zero completion logprob sum.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts that the completion-only sum is ~0 when max_length truncates away completion.

    Raises
    ------
    AssertionError
        If truncation is mishandled such that completion masking still includes tokens.

    Notes
    -----
    - With small max_length, the tokenized sequence contains only prompt tokens.
    - The text backend should return a sentinel start_positions (typically T) indicating no completion tokens.
    - The completion mask should then exclude all token_logprobs entries.
    """
    pairs: Sequence[Tuple[str, str]] = [("hello world", "moon")]
    corpus: List[str] = ["hello world moon"]
    tokenizer: PreTrainedTokenizerFast = _make_fast_wordlevel_tokenizer(corpus)

    # We need start_positions consistent with truncation.
    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=2,
        backend="pt",
    )
    vocab_size: int = tokenizer.vocab_size
    model: BoundaryAwareDummyCausalLM = BoundaryAwareDummyCausalLM(
        vocab_size=vocab_size, start_positions=start_positions
    )

    out: _ModelOutput = completion_logprobs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        device=torch.device("cpu"),
        max_length=2,
    )

    assert out.shape == (1,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_completion_logprobs_invalid_max_length_raises() -> None:
    """
    Validate that completion_logprobs(...) enforces max_length validation.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts that passing max_length <= 0 raises ValueError.

    Raises
    ------
    AssertionError
        If no exception is raised for an invalid max_length.
    ValueError
        Expected outcome when max_length fails validation.

    Notes
    -----
    - This test checks integration: completion_logprobs(...) must call validate_logprobs_inputs(...).
    - Tokenization for model construction uses a valid max_length; the invalid value is passed only to
      completion_logprobs(...).
    """
    pairs: Sequence[Tuple[str, str]] = [("hello", "moon")]
    corpus: List[str] = ["hello moon"]
    tokenizer: PreTrainedTokenizerFast = _make_fast_wordlevel_tokenizer(corpus)

    # start_positions for model construction; max_length here is fine for tokenization.
    _, _, start_positions = tokenize_text(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=8,
        backend="pt",
    )
    vocab_size: int = tokenizer.vocab_size
    model: BoundaryAwareDummyCausalLM = BoundaryAwareDummyCausalLM(
        vocab_size=vocab_size, start_positions=start_positions
    )

    with pytest.raises(ValueError):
        _ = completion_logprobs(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            device=torch.device("cpu"),
            max_length=0,
        )
