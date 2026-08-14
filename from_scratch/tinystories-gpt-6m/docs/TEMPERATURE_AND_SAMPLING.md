# Temperature: Practical Guidance for This Project

Companion to [`SERVING.md`](SERVING.md). The mechanism — the exact math behind
temperature, top-k, top-p, the worked numerical example, and why `temperature=0` is
mathematically identical to greedy decoding — is covered in full in
[Chapter 21 — Inference Mechanics: Decoding, Sampling, and KV Cache](../../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md).
This doc only covers what's specific to this project: the default values in
[`inference.py`](../inference.py), and practical guidance for its `/generate` endpoint.

## Grounded in This Project's Code

[`../inference.py`](../inference.py)'s `sample_next_token`:

```python
def sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / max(temperature, 1e-5)   # <- the division Chapter 21's math describes
    ...
    probs = torch.softmax(vals, dim=-1)         # <- softmax runs AFTER the temperature scaling
```

The `max(temperature, 1e-5)` guards against a literal divide-by-zero if `temperature=0`
were passed through this path — but semantically, `temperature=0` should just mean
"always pick the single most likely token," which is exactly what `--greedy` /
`do_sample=False` already does directly via `torch.argmax`, without touching this
function at all — see Chapter 21's deep-dive for why these are the same operation in the
limit, not just similar in practice.

## Practical Guidance for This Project's `/generate` Endpoint

| Temperature | Effect | When to use it |
|---|---|---|
| 0.0 (or `do_sample: false`) | Fully deterministic, same output every time | Debugging — confirming the model itself, not sampling randomness, produced a specific output |
| 0.5 - 0.7 | Safe, coherent, somewhat repetitive | When coherence matters more than variety |
| 0.8 (this project's default) | A balance — real variety, usually still coherent | General use |
| 1.0 - 1.3 | Noticeably more varied, occasional odd word choices | Exploring different completions of the same prompt |
| \> 1.5 | Frequently incoherent | Rarely useful — mostly demonstrates what "too high" looks like |

Since this model is small (~5.85M parameters) and trained on a narrow, simple dataset
(per [`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md)), it has less "headroom" than
a large model before high temperature breaks coherence — worth expecting the useful range
here to be somewhat narrower than what you'd see quoted for a large frontier model.

One misconception worth flagging specific to this project's small size: higher
temperature does not make this model "more creative" in a meaningful sense — it makes
output more *varied*, and at this parameter count and dataset narrowness that variety
turns to word-salad sooner than it would for a large frontier model. See
[Chapter 21](../../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md#common-misconceptions)
for the general misconceptions and practice questions this project's setup is an
instance of.
