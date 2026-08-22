# custom-gpt-word — a GPT that predicts whole words

This is the word-level companion to [`../custom-gpt-nano`](../custom-gpt-nano/).
Both projects contain a deliberately small decoder-only Transformer with explicit
Query/Key/Value matrix multiplication. Nano maps individual characters to ids; this
project maps words and punctuation to ids, so the learning signal is easier to read:
`the cat` -> likely next token `sat`, rather than needing first to learn how to spell
each word.

It is for understanding the mechanics of language-model training and generation—not
for useful open-ended text. The bundled corpus is tiny and repetitive on purpose: a
CPU training run should visibly memorize patterns in a few minutes.

## Quick start

```bash
cd from_scratch/custom-gpt-word
make setup
make config
make dry-run
make train
make generate PROMPT="the cat"
make eval
```

`make dry-run` is the safest first step. It performs one genuine forward pass, loss
calculation, backward pass, and AdamW update in memory, then exits without creating a
checkpoint. It prints an input window and its shifted target, tensor shapes, loss,
gradient norm, and a weight before/after the update. If that succeeds, the full training
path is wired correctly.

`make train` runs 1,000 updates and writes `checkpoints/word-gpt.pt`. `make clean`
removes only that generated checkpoint. Generation samples from the learned next-token
probabilities; try `PROMPT="the dog"` and temperatures such as `0.4` or `1.0`:

```bash
uv run word-gpt-generate --prompt "the sun" --max-new-tokens 25 --temperature 0.6
```

## What changes from character-level nano?

| Concern | `custom-gpt-nano` | `custom-gpt-word` |
|---|---|---|
| Token | One character | One lowercase word or punctuation mark |
| Vocabulary | Every distinct corpus character | Every distinct corpus word/punctuation, plus `<unk>` |
| Context window | 64 characters | 12 tokens (roughly a short sentence) |
| Unknown prompt text | Error | Maps to `<unk>` |
| Primary learning benefit | See every Transformer operation | See tokenization and next-word prediction more directly |

The model architecture is otherwise the same essential GPT recipe: token embeddings +
position embeddings, stacked causal-attention/MLP blocks, and a vocabulary projection.
The output projection shares its weights with the input token embedding, a common
technique called *weight tying*.

## Read it in this order

1. [`src/wordgpt/tokenizer.py`](src/wordgpt/tokenizer.py): token strings become ids.
2. [`src/wordgpt/data.py`](src/wordgpt/data.py): random input windows and shifted labels.
3. [`src/wordgpt/model.py`](src/wordgpt/model.py): Transformer tensors and causal mask.
4. [`src/wordgpt/dry_run.py`](src/wordgpt/dry_run.py): one complete learning update.
5. [`src/wordgpt/train.py`](src/wordgpt/train.py): repeat that update and save weights.
6. [`src/wordgpt/generate.py`](src/wordgpt/generate.py): sample one next token repeatedly.

[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) traces the exact data flow and explains
what the dry-run output means. For the broader theory, use the workspace curriculum:
[`tokenization`](../../docs/llm-engineering/09_tokenization.md),
[`Transformer architecture`](../../docs/llm-engineering/10_transformer_architecture.md),
[`training`](../../docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md),
and [`generation`](../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md).

If Q/K/V attention is new, start with the dedicated
[`Causal Q/K/V attention guide`](docs/CAUSAL_QKV_ATTENTION.md). It progresses from a
plain-language analogy through a worked `the cat sat` example to the exact tensors and
lines of code used by this project.

For a class-by-class, top-to-bottom read of `model.py` itself — residual connections,
the two embedding tables, weight tying, and the full shape trace through `forward()` —
see [`docs/MODEL_WALKTHROUGH.md`](docs/MODEL_WALKTHROUGH.md).

## Boundaries worth knowing

This is not BPE, not a general-purpose chatbot, and not a benchmark. A word unseen in
the bundled corpus becomes `<unk>`, word order is only learned from a very small set of
examples, and sampling can produce awkward output. Those limitations are intentional:
they keep the complete tokenizer, model, training loop, checkpoint format, and
generation loop small enough to inspect in one sitting. The larger sibling projects
replace this simple tokenizer and toy corpus with production-shaped components.
