from alignopt.types import Preference

LOCKFILE_EXAMPLE: Preference = Preference(
    prompt="In one sentence, what is a lockfile in Python packaging?",
    chosen="A lockfile records exact dependency versions to make installations reproducible.",
    rejected="A lockfile is a folder where Python stores your code.",
)

DPO_EXAMPLE: Preference = Preference(
    prompt="In one sentence, what does DPO optimize?",
    chosen="DPO directly optimizes a policy to prefer chosen responses over\
        rejected ones relative to a reference model.",
    rejected="DPO trains a model by guessing random answers until it improves.",
)

TOKENIZATION_EXAMPLE: Preference = Preference(
    prompt="In one sentence, what is tokenization for language models?",
    chosen="Tokenization converts text into a sequence of discrete IDs that a model can process.",
    rejected="Tokenization encrypts text so the model cannot read it.",
)

def test_dpo_logprob() -> None:
    print(LOCKFILE_EXAMPLE.prompt)
    print(DPO_EXAMPLE.prompt)
    print(TOKENIZATION_EXAMPLE.prompt)
