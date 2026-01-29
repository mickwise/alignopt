from typing import Iterable
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

def make_fast_wordlevel_tokenizer(corpus: Iterable[str]) -> PreTrainedTokenizerFast:
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

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
