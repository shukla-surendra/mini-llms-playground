# Why this model is shaped the way it is

`custom-gpt-200m` is the first project here that does **not** reuse the GPT-2-style
stack of its siblings. Every departure below was chosen for a reasoning model at ~200M
parameters, and each one is a cost as well as a benefit.

**The honest headline first:** architecture is the *smallest* lever on reasoning at this
scale. Every small model that actually reasons got there through data and token count —
SmolLM2-360M saw ~4T tokens (≈11,000 tokens/parameter); Qwen2.5-0.5B saw ~18T.
Chinchilla-optimal for 200M is **4B tokens (20/param)**, which buys a coherent model,
not a reasoning one. Read [`DATASET.md`](../DATASET.md) before congratulating this file.

## The configuration

```
vocab_size     32,768      own BPE, digit-aware
context_length 2,048       RoPE, so extensible rather than capped
embed_size     896
num_heads      14          head_dim = 64
num_layers     18          E/L = 49.8
ffn_hidden     2,368       SwiGLU, ~8/3 x E
                           -> 201,769,344 parameters
```

| component | params | share |
|---|---:|---:|
| token embedding | 29,360,128 | 14.6% |
| attention | 57,802,752 | 28.6% |
| SwiGLU MLP | 114,573,312 | 56.8% |
| RMSNorms | 33,152 | 0.0% |

85% of the budget is in transformer blocks. The GPT-2-vocabulary siblings spend
20–25% on the embedding table alone.

## Shape: deeper and narrower than GPT-2

| model | E | L | E/L |
|---|---:|---:|---:|
| GPT-2 small | 768 | 12 | 64.0 |
| custom-gpt-153m | 768 | 16 | 48.0 |
| **this** | **896** | **18** | **49.8** |
| Qwen2.5-0.5B | 896 | 24 | 37.3 |
| SmolLM2-360M | 960 | 32 | 30.0 |
| SmolLM-135M | 576 | 30 | 19.2 |

Small models that reason cluster deep-and-narrow. Depth is what composes multi-step
operations — each layer refines what the previous produced — and that is precisely what
"reasoning" names. This sits at 49.8: meaningfully deeper than GPT-2, deliberately not
as deep as SmolLM. Very deep stacks are harder to train stably and give less
parallelism per layer, which costs throughput on a fixed budget. If the token budget
grows past ~20B, go deeper (E=768, L=26 is ~209M and ratio 29.5).

`head_dim = 64` throughout, non-negotiable: it is what SDPA's fused kernels are tuned
for, and [`MODEL_SIZING_GUIDE.md`](MODEL_SIZING_GUIDE.md) records the 10m preset's
`head_dim=20` as a real measured throughput artifact.

## RoPE instead of learned positions

The siblings use `nn.Embedding(context_length, embed_size)` — one learned row per
position. Position 2049 in a model trained at 2048 has no row, so **context is a
permanent architectural ceiling**, and the sizing guide calls changing it a one-way
door.

RoPE rotates Q and K by an angle proportional to position, so attention scores depend
only on the *relative* offset between two tokens. Nothing is learned per position:
zero parameters, no table to run off the end of, and the same weights can later run at
longer context (degraded, better with interpolation) rather than not at all.

Verified, not assumed: with a fixed q and k, `score(10,5) == score(100,95) ==
score(300,295) == score(500,495)` to 1e-4, and differs at a different offset.

## SwiGLU at 8/3 E

A GELU MLP is two matrices, `E -> 4E -> E` = `8E²` parameters. SwiGLU is three (gate,
up, down) = `3Ef`. Setting `f = (8/3)E` makes the two cost the same, which is why
`ffn_hidden` is ~2.64x E rather than 4x — the same sizing Llama uses. The gate branch
lets the network modulate the up branch multiplicatively, an interaction a plain MLP
cannot express at equal cost.

## RMSNorm, and no biases

RMSNorm rescales by root-mean-square without subtracting the mean or adding a learned
shift. The centring does not measurably help transformers, and dropping it plus the
bias removes work from every norm. Computed in fp32 even under autocast — the
mean-square of a bf16 vector loses precision exactly where the norm is most sensitive.

Biases are removed from every `Linear`. Modern decoders do this because the bias terms
contribute nothing measurable while occupying parameters and memory traffic.

## The tokenizer is the reasoning decision

Full argument in [`src/gpt/tokenizer.py`](../src/gpt/tokenizer.py). Two parts:

**Budget.** At E=896, a 50,257 vocabulary costs 45.0M parameters (22.3%); 32,768 costs
29.4M (14.6%). The 15.7M difference goes into blocks.

**Arithmetic — the part that actually matters.** GPT-2's BPE merges digit runs by
corpus frequency, so numbers have no canonical segmentation: `2024` might be one token
while `2025` is three. A model then learns arithmetic separately per segmentation.
Splitting every digit gives one representation per number, so a carry rule learned on
one transfers to all:

```
   48 -> ['4','8']        2024 -> ['2','0','2','4']
 2025 -> ['2','0','2','5']  1999 -> ['1','9','9','9']
```

**Measured cost of that choice:** 4.07 chars/token with digit splitting vs 4.27
without — **4.7% more tokens** for canonical numbers. Cheap.

**A bug this caught.** The first version used `ByteLevel(add_prefix_space=True)` and
**failed roundtrip** — `decode(encode(s)) != s` on an ordinary sentence, because an
injected prefix space is indistinguishable from a real one at decode time. Fixed to
`add_prefix_space=False`; ByteLevel's own regex already binds a leading space to the
following word. Roundtrip now holds for ASCII, unicode, em-dashes and emoji.

## Generation stops at the boundary token

`<|endoftext|>` is trained in as a real special token from the start, and
`generate_text` **breaks** on it. The siblings have no early stop — they always run the
full `max_new_tokens`, so once the model correctly predicts "this reply is over" the
loop keeps sampling into an unrelated document, which is exactly how a hallucinated
second `User:` turn appears in their QA reports.

## What this costs you

- **A one-way door.** Different vocabulary and different block structure: checkpoints
  cannot be loaded by, resumed from, or numerically compared against any sibling
  project. Perplexity is not comparable across tokenizers either.
- **The corpus must be re-tokenized** whenever the tokenizer is retrained, and existing
  checkpoints become garbage (different ids = different embedding rows).
- **Unproven code.** RoPE, SwiGLU and RMSNorm are individually verified here but have
  never trained a real model in this repo. If a long run misbehaves, suspect this file
  before suspecting the data.

## Deliberately not done

- **GQA** — an inference-memory optimisation. At 200M with 14 heads there is little to
  reclaim, and it complicates the attention path.
- **Context beyond 2048** — attention is 27.5% of dense FLOPs at 2048 and **54.3%** at
  4096. Nearly double the compute for context the corpus cannot fill (documents chunk
  at 1024). RoPE means this can be revisited later without retraining from scratch.
- **A learned scale on `lm_head`**, dropout variants, and z-loss: not worth the
  unverified surface area on a first run.
