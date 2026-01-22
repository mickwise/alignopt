from typing import Sequence, Tuple
from transformers import FlaxPreTrainedModel, PreTrainedTokenizerFast
import jax
from alignopt.backends.text import tokenize_text
from alignopt.backends.validation import validate_logprobs_inputs

def completion_logprobs(
    model: FlaxPreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    pairs: Sequence[Tuple[str, str]],
    device: str | None = None,
    max_length: int | None = None
    ) -> jax.Array:
    validate_logprobs_inputs(max_length)
    if device is None:
        device = next(model.parameters()).device
    input_ids, attention_mask, start_positions = tokenize_text(
        pairs,
        tokenizer,
        max_length,
        backend="jax"
    )
    input_ids = jax.device_put(input_ids, device)
    attention_mask = jax.device_put(attention_mask, device)
    start_positions = jax.device_put(max_length, device)
    token_logprobs: jax.Array = _calculate_token_logprobs(model, input_ids, attention_mask)
    masked_logprobs: jax.Array = _mask_logprobs(
        token_logprobs,
        attention_mask,
        device,
        start_positions
    )
    return masked_logprobs.sum(axis=1)


def _calculate_token_logprobs(
        model: FlaxPreTrainedModel,
        input_ids: jax.Array,
        attention_mask: jax.Array
    ) -> jax.Array:
    logits: jax.Array = model(input_ids, attention_mask).logits
    logprobs: jax.Array = jax.nn.log_softmax(logits, axis=-1)[:, :-1, :]
    targets: jax.Array = input_ids[:, 1:]
    return jax.numpy.take_along_axis(logprobs, targets[:, None], axis=-1).squeeze(-1)


def _mask_logprobs(
        token_logprobs: jax.Array,
        attention_mask: jax.Array,
        device: str,
        start_position: jax.Array
    ) -> jax.Array:
    pad_mask: jax.Array = attention_mask[:, 1:]
    positions: jax.Array = jax.numpy.arange(1, attention_mask.size - 1, device=device)
    completion_mask: jax.Array = positions >= start_position[:, None] - 1
    logprobs_mask: jax.Array = pad_mask*completion_mask
    return token_logprobs*logprobs_mask
