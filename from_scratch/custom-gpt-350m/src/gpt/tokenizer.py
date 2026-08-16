"""This project's own BPE tokenizer — trained here, not borrowed from GPT-2.

The sibling custom-gpt-{10m,50m,153m} projects use GPT-2's 50,257-token vocabulary via
`tiktoken`. This one trains its own, for two reasons that both matter specifically for
a reasoning model at ~200M parameters:

**Parameter budget.** The token-embedding table is `vocab_size x embed_size`, and at
this scale it is the largest single non-block component. At E=896:

    V = 50,257  ->  45.0M parameters  (22.3% of a 202M model)
    V = 32,768  ->  29.4M parameters  (14.6%)

The 15.7M difference is redirected into transformer blocks — actual reasoning
capacity rather than lookup table. `MODEL_SIZING_GUIDE.md` describes the same trade in
the other direction: below the E ~ 697 crossover, embeddings dominate and widening the
model mostly buys a bigger table.

**Arithmetic.** This is the reasoning-specific half, and it is not about size at all.
GPT-2's BPE merges digit runs by corpus frequency, so numbers have no canonical
segmentation: common ones collapse into a single token while neighbouring values split
into several. A model then has to learn arithmetic separately for each segmentation it
happens to see. Splitting every digit individually (`pre_tokenizers.Digits`) gives
every number exactly one representation, so carry rules learned on one number transfer
to all of them. Llama and GPT-4 both do a version of this deliberately.

The trade is a genuine one-way door: a checkpoint trained with this tokenizer cannot
be loaded by, resumed from, or compared against the GPT-2-vocabulary siblings, and the
corpus must be re-tokenized whenever the tokenizer is retrained.

`Tokenizer` below exposes the small slice of `tiktoken`'s interface the rest of the
package uses (`encode`, `decode`, `n_vocab`), so trainer/dataset/generate need no
per-project branch.
"""

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

#: Document boundary. Trained in as a real special token from the start, so it is one
#: id rather than a sequence of ordinary subwords — the failure the sibling projects
#: hit with GPT-2 and had to work around with `allowed_special`.
END_OF_TEXT = "<|endoftext|>"

SPECIAL_TOKENS = [END_OF_TEXT]


class Tokenizer:
    """Adapter presenting a `tiktoken`-shaped interface over a `tokenizers` BPE.

    `allowed_special`/`disallowed_special` are accepted and ignored: special tokens are
    part of this vocabulary and always encode as themselves, so there is no equivalent
    of tiktoken's opt-in. Keeping the arguments means shared call sites work unchanged.
    """

    def __init__(self, hf_tokenizer):
        self._tk = hf_tokenizer

    @property
    def n_vocab(self):
        return self._tk.get_vocab_size()

    @property
    def eot_id(self):
        return self._tk.token_to_id(END_OF_TEXT)

    def encode(self, text, allowed_special=None, disallowed_special=None):
        return self._tk.encode(text).ids

    def decode(self, ids):
        return self._tk.decode(list(ids))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(path))


def build_trainer_tokenizer(vocab_size):
    """A byte-level BPE with individual-digit pre-tokenization."""
    tk = HFTokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        # Digits FIRST: split every digit run into single characters before byte-level
        # merging can glue them together. This is the whole arithmetic argument above —
        # without it, BPE happily learns " 2024" as one token and " 2025" as three.
        pre_tokenizers.Digits(individual_digits=True),
        # add_prefix_space=False, and this is not a style choice: with True the decoder
        # cannot reconstruct the original text (measured — `decode(encode(s)) != s` on a
        # plain sentence), because the injected prefix space is not distinguishable from
        # a real one at decode time. ByteLevel's own regex already binds a leading space
        # to the following word, so consistent word tokenization does not need it.
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # every byte, so no UNK
        show_progress=True,
    )
    return tk, trainer


def train_tokenizer(files, vocab_size, out_path):
    """Train a BPE over `files` and save it. Returns the loaded `Tokenizer`."""
    tk, trainer = build_trainer_tokenizer(vocab_size)
    tk.train([str(f) for f in files], trainer)
    wrapped = Tokenizer(tk)
    wrapped.save(out_path)
    return wrapped


def load_tokenizer(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This project trains its own tokenizer — run "
            f"`make tokenizer` (gpt-train-tokenizer) before tokenizing a corpus "
            f"or training."
        )
    return Tokenizer(HFTokenizer.from_file(str(path)))
