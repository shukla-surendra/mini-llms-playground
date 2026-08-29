# Converting This Model to HF Format and Serving It With vLLM — A Manual Walkthrough

This is a **build-it-yourself guide**, not a finished script. The general mental model and
checklist live in
[`../../../docs/llm-engineering/23_the_serving_engine_ecosystem_vllm_and_friends.md`](../../../docs/llm-engineering/23_the_serving_engine_ecosystem_vllm_and_friends.md)
— read that first if you haven't, since this doc assumes you already know *why* each step
below exists and only walks through *this project's specific* version of each one.
Two sibling projects already have a finished, working version of this
(`custom-gpt-50m/src/gpt/cli/export_vllm.py` and `custom-gpt-350m`'s equivalent) — treat
them as an answer key to check yourself against **after** you've written your own attempt,
not as something to copy from directly.

## Step 0 — Confirm the fingerprint (already done, here's the receipt)

`src/gpt/model.py`'s `TinyStoriesGPT`:

| Question | Answer | Family |
| --- | --- | --- |
| Position info? | `pos_emb = nn.Embedding(context_length, embed_size)` | GPT-2 |
| Norm type? | `nn.LayerNorm` | GPT-2 |
| FFN shape? | `nn.Sequential(Linear(E,4E), GELU(), Linear(4E,E), Dropout)` | GPT-2 |
| Biases? | `in_proj`/`out_proj` both `bias=True` | GPT-2 |

Target: `GPT2LMHeadModel`. Same target as `custom-gpt-50m` — its `hf_state_dict()` is the
closest reference, not `custom-gpt-350m`'s (that one targets `LlamaForCausalLM` and solves
a different set of problems: QKV *splitting* instead of a straight fused copy, no bias
handling at all, no activation-exactness fix needed).

## Step 1 — Add the two missing checkpoint helpers

Check `src/gpt/checkpoint.py` yourself: it already has `make_payload`, `is_compatible`,
`load_checkpoint` (a thin `torch.load` wrapper), and `remap_attn_impl`. What it's missing,
compared to `custom-gpt-50m/src/gpt/checkpoint.py`, is a **`load_model()`** that goes all
the way from a checkpoint path to a ready-to-run `TinyStoriesGPT` instance, and a
**`select_checkpoint()`** that picks best/latest by name.

Two project-specific things to get right that `custom-gpt-50m`'s version doesn't need to
handle:

1. **Checkpoints are namespaced by `label/objective`**, not just `label` — `Paths.checkpoint_dir`
   here is `checkpoint_root / label / objective` (this project trains causal/MLM/contrastive
   objectives from the same architecture). Your `select_checkpoint()` needs a `Paths` object
   that already has `objective="causal"` set — the MLM and contrastive checkpoints are not
   candidates for this conversion at all (see the "Common Misconceptions" note in Chapter 23
   about vLLM only serving causal generation).
2. **Always rebuild for inference with `attn_impl="sdpa"`**, remapping via `remap_attn_impl`
   if the checkpoint was trained with `"naive"` — the export needs the fused `in_proj`/
   `out_proj` `nn.Linear` shape to map onto GPT-2's `c_attn`/`c_proj`, and only the `"sdpa"`
   path has that shape. (`remap_attn_impl` already exists here and already handles this
   direction — you're calling it, not writing it.)

Write these two functions now. Check your signature choices against
`custom-gpt-50m/src/gpt/checkpoint.py`'s `load_model`/`select_checkpoint` once you have a
version that runs — don't read theirs first.

## Step 2 — Write the weight-mapping table, then the export script

Before writing any code, write out the source-key → target-key table yourself by reading
`model.py` next to HF's `GPT2LMHeadModel` key names. Here's the skeleton to fill in — the
`???` cells are what you need to determine (layout: transpose or not; fusion: copy or
split):

| Source (`TinyStoriesGPT`) | Target (`GPT2LMHeadModel`) | Transpose? |
| --- | --- | --- |
| `token_emb.weight` | `transformer.wte.weight` | ??? |
| `pos_emb.weight` | `transformer.wpe.weight` | ??? |
| `blocks.{i}.ln_1.weight` / `.bias` | `transformer.h.{i}.ln_1.weight` / `.bias` | ??? |
| `blocks.{i}.attn.in_proj.weight` / `.bias` | `transformer.h.{i}.attn.c_attn.weight` / `.bias` | ??? |
| `blocks.{i}.attn.out_proj.weight` / `.bias` | `transformer.h.{i}.attn.c_proj.weight` / `.bias` | ??? |
| `blocks.{i}.ln_2.weight` / `.bias` | `transformer.h.{i}.ln_2.weight` / `.bias` | ??? |
| `blocks.{i}.mlp.net.0.weight` / `.bias` | `transformer.h.{i}.mlp.c_fc.weight` / `.bias` | ??? |
| `blocks.{i}.mlp.net.2.weight` / `.bias` | `transformer.h.{i}.mlp.c_proj.weight` / `.bias` | ??? |
| `ln_f.weight` / `.bias` | `transformer.ln_f.weight` / `.bias` | ??? |
| `lm_head.weight` | `lm_head.weight` | ??? |

Hint for the `???` column, without giving it away entirely: exactly one framework quirk
from Chapter 23 applies to *every* row that isn't a norm or embedding — figure out which
one, and why it applies uniformly here rather than needing a per-row decision.

Create `src/gpt/cli/export_hf.py` (this project has no `export_vllm.py` yet — name it
however you like, but check what `pyproject.toml`'s `[project.scripts]` convention expects
so `uv run` picks it up) with this shape:

```python
def hf_state_dict(checkpoint):
    """Fill this in using the table above. Loop `for layer in range(checkpoint["num_layers"])`
    for the per-block keys; handle token_emb/pos_emb/ln_f/lm_head outside the loop."""
    ...

def _gpt2_config(checkpoint, GPT2Config):
    """Map checkpoint fields to GPT2Config fields. n_inner should be computed from
    embed_size, not hardcoded — check model.py's MLP class for the actual multiplier."""
    ...

def main():
    # load checkpoint via your new select_checkpoint/load_model
    # build hf_state_dict(checkpoint), load into a fresh GPT2LMHeadModel
    # verify (Step 4) before writing anything
    # model.save_pretrained(...), tokenizer.save_pretrained(...)
    ...
```

## Step 3 — The tokenizer has one real project-specific wrinkle

`custom-gpt-50m` and `custom-gpt-350m` both have tokenizers with no unknown-token
fallback. **This project's does**: `data/prepare.py` trains the BPE with
`models.BPE(unk_token="<unk>")` and `special_tokens=["<unk>", "<|endoftext|>"]`. Confirm the
ids yourself:

```python
from tokenizers import Tokenizer
tk = Tokenizer.from_file("data/tokenizer.json")
tk.token_to_id("<unk>")           # is this 0?
tk.token_to_id("<|endoftext|>")   # is this 1?
```

When you wrap this in `transformers.PreTrainedTokenizerFast`, you need to pass `unk_token`
in addition to `bos_token`/`eos_token` (both `"<|endoftext|>"`, same as the sibling
projects) — decide what happens if you *don't* pass it, and check whether that's actually
a problem for a model that's never supposed to encode text outside its own tiny trained
vocabulary in the first place.

## Step 4 — Verification (do not skip this to "save time")

Write a `_verify_equivalence()` that does exactly what Chapter 23 and both sibling scripts
do: run the native model (via your new `load_model()`) and the freshly-built HF model on
the same input string, and `torch.allclose()` the logits. Refuse to `save_pretrained()` if
they don't match.

Self-check before you run it for real: deliberately introduce one wrong mapping (skip a
transpose, or map `c_fc`/`c_proj` backwards) and confirm two things — that the export still
completes and the file still loads via `AutoModelForCausalLM` with no error, and that your
verification step is what actually catches the mistake. If your verification doesn't catch
a deliberately-broken mapping, the verification itself has a bug — fix that before trusting
it on a correct mapping.

## Step 5 — Wiring: `pyproject.toml`, `Makefile`

This part is pure boilerplate, not the learning objective — copy the pattern directly from
`custom-gpt-50m/pyproject.toml`'s `hf-export`/`vllm` optional-dependency groups and
`Makefile`'s `export-vllm`/`serve-vllm` targets, adjusting only the script module path to
point at wherever you put `main()` in Step 2. One thing to change deliberately, not copy:
**do not** add `--model-impl transformers` to the `serve-vllm` recipe — that flag is the
one that broke the identical GPT-2 conversion on `custom-gpt-50m` (see Chapter 23's
Deep-Dive). Leave it off from the start here.

## Step 6 — Run it

1. `uv sync --extra hf-export`
2. Run your export command. It should print an `exports/...` directory and exit 0 only if
   verification passed.
3. Independently confirm it with **plain** `transformers` — not your own project's
   tooling, so you're testing the actual HF contract, not your own code testing itself:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   tok = AutoTokenizer.from_pretrained("exports/.../your-export-dir")
   model = AutoModelForCausalLM.from_pretrained("exports/.../your-export-dir")
   print(model.config.architectures)  # expect ['GPT2LMHeadModel']
   ```
4. Build vLLM's CPU backend from source if you haven't already for this project's `.venv`
   (same steps as `custom-gpt-50m/README.md`'s "Testing vLLM on an Apple-Silicon Mac"
   section — `cmake`/`ninja` via `brew`, clone `vllm` source, `uv pip install -r
   requirements/cpu.txt`, `VLLM_TARGET_DEVICE=cpu uv pip install -e .`).
5. Serve it, remembering the CPU-backend gotcha from today: `--gpu-memory-utilization` on
   this backend means a fraction of **system RAM**, not VRAM. Start low:
   ```bash
   make serve-vllm VLLM_MODEL_DIR=exports/.../your-export-dir VLLM_ARGS="--gpu-memory-utilization 0.1"
   ```
6. Hit it for real, and compare the output to what plain `transformers.generate()` gave
   you in step 3 — they should match exactly:
   ```bash
   curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \
     '{"model": "exports/.../your-export-dir", "prompt": "Once upon a time", "max_tokens": 20, "temperature": 0}'
   ```

## Self-check before you consider this done

- Can you explain, without looking anything up, why *every* attention/MLP weight needed
  the same transpose treatment, but the embeddings and norms didn't?
- If this project's tokenizer had been trained *without* an explicit `unk_token`, would
  step 3 have been simpler or identical? Why?
- Your verification function passed. What specific class of bug does that prove is
  *absent*, and what class of bug could still exist that logit-parity on one input
  wouldn't catch?
- Why would forcing `--model-impl transformers` on this export risk hitting the exact same
  bug that hit `custom-gpt-50m`, rather than a different, unrelated one?
