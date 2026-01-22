def validate_logprobs_inputs(
        max_length: int | None,
    ) -> None:
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("Max length should be positive.")
