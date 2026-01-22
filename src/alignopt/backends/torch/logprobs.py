"""
Purpose
-------
Compute completion-only sequence log-probabilities for (prompt, completion) pairs under a
causal language model (PyTorch backend). This module exists to support preference-optimization
algorithms (e.g., DPO) that require per-example log π(y | x) restricted to completion tokens,
with masking aligned to next-token prediction.

Key behaviors
-------------
- Tokenizes (prompt, completion) pairs via alignopt.backends.text.tokenize_text(...) using the
  project's canonical concatenation and boundary convention.
- Runs a Hugging Face causal LM forward pass to obtain logits and converts them to per-token
  log-probabilities.
- Extracts token-level log-probabilities for the realized tokens via gather.
- Builds a completion-only mask (plus padding mask) aligned to shifted targets and returns the
  sum of completion token log-probabilities per example.

Conventions
-----------
- Assumes causal LM semantics: logits at position t predict token at position t+1.
- Uses attention_mask to zero out padded tokens; completion masking uses start_positions derived
  from tokenizer offset mappings in the text module.
- start_positions are indices in the unshifted token sequence; masking accounts for the shift by
  using (start_positions - 1) when applying to token_logprobs.

Downstream usage
----------------
Call completion_logprobs(...) from algorithm code (e.g., DPO loss) to obtain log π(y|x) values for
chosen/rejected completions. This module is PyTorch-specific; text preprocessing is shared via the
text backend module.
"""

from typing import Sequence, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerFast
import torch
from alignopt.backends.text import tokenize_text
from alignopt.backends.validation import validate_logprobs_inputs

def completion_logprobs(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        pairs: Sequence[Tuple[str, str]],
        device: torch.device | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
    """
    Compute per-example completion-only log-probabilities for a batch of (prompt, completion) pairs.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Hugging Face causal language model. Must return an object with a
        ".logits" tensor of shape (B, T, V) when called with input_ids and attention_mask.
    tokenizer : transformers.PreTrainedTokenizerFast
        Fast Hugging Face tokenizer used by tokenize_text(...)
        to produce input IDs, attention mask, and completion start positions.
    pairs : Sequence[Tuple[str, str]]
        Batch of (prompt, completion) string pairs.
    device : torch.device | None
        Device to run the forward pass on. If None, uses the device of model parameters.
    max_length : int | None
        Max sequence length passed through to tokenization (truncation limit).
        If None, tokenizer default is used.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B,) containing, for each pair, the sum of log-probabilities
        of completion tokens under the model, with padding and prompt tokens masked out.

    Raises
    ------
    ValueError
        If max_length fails validation in validate_logprobs_inputs(...).
    TypeError
        If tokenize_text(...) rejects the tokenizer (e.g., not a fast tokenizer).

    Notes
    -----
    - The completion boundary is derived in tokenize_text(...) and returned as start_positions
      (token indices in the unshifted input sequence).
    - Because next-token prediction shifts alignment by one,
      the completion mask is applied starting at (start_positions - 1)
      in the token_logprobs sequence.
    """

    validate_logprobs_inputs(max_length)
    if device is None:
        device = next(model.parameters()).device
    input_ids, attention_mask, start_positions = tokenize_text(
        pairs,
        tokenizer,
        max_length,
        backend="pt"
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    start_positions = start_positions.to(device)
    token_logprobs: torch.Tensor = _calculate_token_logprobs(model, input_ids, attention_mask)
    masked_logprobs: torch.Tensor = _mask_logprobs(
        token_logprobs,
        attention_mask,
        device,
        start_positions
    )
    return masked_logprobs.sum(dim=1)


def _calculate_token_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute token-level log-probabilities of the realized next tokens for a tokenized batch.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Hugging Face causal language model producing logits of shape (B, T, V).
    input_ids : torch.Tensor
        Token IDs of shape (B, T).
    attention_mask : torch.Tensor
        Attention mask of shape (B, T), with 1 for real tokens and 0 for padding.

    Returns
    -------
    torch.Tensor
        Token log-probabilities of shape (B, T-1), where element [b, t] is the log-prob assigned to the
        realized target token input_ids[b, t+1].

    Raises
    ------
    None

    Notes
    -----
    - This uses log_softmax over the vocabulary dimension and gathers the log-prob for the realized targets
      input_ids[:, 1:].
    - The returned sequence is aligned to shifted targets (T-1 length), not the original input length T.
    """

    logits: torch.Tensor = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs: torch.Tensor = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    targets: torch.Tensor = input_ids[:, 1:]
    return logprobs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)


def _mask_logprobs(
    token_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device | str,
    start_positions: torch.Tensor
    ) -> torch.Tensor:
    """
    Apply padding and completion-start masking to token-level log-probabilities.

    Parameters
    ----------
    token_logprobs : torch.Tensor
        Token log-probabilities of shape (B, T-1), aligned to targets input_ids[:, 1:].
    attention_mask : torch.Tensor
        Attention mask of shape (B, T) for the unshifted input_ids.
    device : torch.device | str
        Device used to construct helper tensors (e.g., arange positions).
    start_positions : torch.Tensor
        Token indices of shape (B,) indicating where the completion begins in the unshifted input sequence.

    Returns
    -------
    torch.Tensor
        Masked token log-probabilities of shape (B, T-1), where prompt tokens and padding tokens are zeroed.

    Raises
    ------
    None

    Notes
    -----
    - Padding mask uses attention_mask[:, 1:] to align to the shifted targets.
    - Completion mask compares positions (0..T-2) against (start_positions - 1) to account for the shift:
      token_logprobs[:, t] corresponds to input_ids[:, t+1].
    """

    pad_mask: torch.Tensor = attention_mask[:, 1:]
    positions: torch.Tensor = torch.arange(0, attention_mask.size(-1) - 1, device=device)
    completion_mask: torch.Tensor = positions >= start_positions.unsqueeze(-1) - 1
    logprobs_mask: torch.Tensor = pad_mask*completion_mask
    return token_logprobs*logprobs_mask
