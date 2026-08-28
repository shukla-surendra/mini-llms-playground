# The Serving-Engine Ecosystem (vLLM and Friends)

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 4 — Serving: Turning a
Trained Model Into Something You Can Talk To. Builds on
[Chapter 22](22_from_script_to_api_serving_a_model_for_real.md)'s hand-off point —
continuous batching and KV-cache budgeting once a synchronous one-request-at-a-time
server stops being enough — and on
[Chapter 31](31_publishing_a_model_the_hugging_face_hub_workflow.md)'s **raw-files
publication** path, which this chapter deliberately contrasts with a different approach:
converting a custom architecture into one a dedicated serving engine already knows, rather
than publishing the custom code alongside the weights.

## In Plain English

A dedicated serving engine like vLLM is not a smarter, faster version of the FastAPI
wrapper from Chapter 22 — it's a different program entirely, one that only knows how to
run a specific, closed list of model architectures it has hand-optimized (paged KV-cache
memory, continuous batching, fused kernels). It cannot run an arbitrary `nn.Module` any
more than a shipping port built for standard containers can lift a truck that doesn't fit
one. Making a from-scratch model servable this way is never "teach vLLM my architecture"
— it's "prove my model is mathematically identical to an architecture vLLM already knows,"
then hand it weights in exactly that architecture's shape.

## The First-Principles Explanation

### "HF-compatible" and "vLLM-compatible" are different bars

**HF-compatible** means a model directory can be loaded through `transformers`' generic
loader (`AutoModelForCausalLM.from_pretrained(...)`) with no custom code, because it has:
a `config.json` naming an `architectures` class `transformers` already implements (e.g.
`GPT2LMHeadModel`, `LlamaForCausalLM`), weights whose parameter names and shapes match
that implementation exactly, and tokenizer files transformers can load.

**vLLM-compatible** is a narrower, additional requirement on top: vLLM keeps its **own**
separate registry of model implementations, rewritten internally with paged attention and
fused kernels, and only serves architectures it has specifically reimplemented. That list
is large (Llama, Mistral, GPT-2, Qwen, Gemma, Phi, and more) but not exhaustive — a custom
or obscure architecture can be perfectly HF-loadable via `trust_remote_code=True` and still
have no vLLM implementation at all. So:

```
vLLM-compatible  ⊆  HF-compatible
```

Landing on an architecture name that's in *both* categories — which is exactly what the
conversion below does — gets both for the price of one conversion.

### The conversion checklist: fingerprint, map, verify

**1. Fingerprint the architecture.** Four questions place almost any decoder-only model
into an existing family:

| Question | GPT-2 family | Llama family |
| --- | --- | --- |
| Position info? | Learned embedding table | RoPE (rotated Q/K, no table) |
| Norm type? | LayerNorm (mean + var, has bias) | RMSNorm (scale only, no bias) |
| FFN shape? | 2 matrices + GELU | 3 matrices (gate/up/down) + SiLU, gated |
| Biases? | Yes, everywhere | No, nowhere |

[Chapter 11](11_positional_encoding_variants_rope_and_beyond.md) and
[Chapter 35](35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md) already cover why
this repo's two architecture families made these specific choices — this chapter is about
what those choices mean for which existing HF class a model maps onto, not why the choices
were made.

**2. Map every weight tensor, checking two separate things per tensor, not one:**

- **Layout** — does the target framework store this tensor differently? HF's GPT-2 uses
  `Conv1D` layers, which store weights as `(in_features, out_features)` — the **transpose**
  of `nn.Linear`'s `(out_features, in_features)`. This is a framework quirk, not an
  architecture difference: Llama uses plain `nn.Linear` on both sides, so no transpose is
  needed there at all.
- **Fusion** — is Q/K/V projected as one fused matrix on one side but three separate
  matrices on the other? A fused `in_proj` maps onto GPT-2's fused `c_attn` with a straight
  copy (both fused); the same fused `in_proj` maps onto Llama's separate `q_proj`/`k_proj`/
  `v_proj` with a `.split()`, since Llama never fuses them.

**3. Check activation-function *exactness*, not just family resemblance.** Two functions
that look like "the same activation" in a diagram can differ in the actual floating-point
values they produce. GPT-2's HF default is `gelu_new` — a tanh approximation — while a
model trained with plain `nn.GELU()` used the exact erf-based formula; exporting without
addressing this bakes in a small, avoidable numerical gap. SwiGLU's `silu` needs no such
fix, because it already *is* Llama's default `hidden_act` — the point isn't "SwiGLU is
safer," it's that this has to be checked per architecture rather than assumed.

**4. Check whether the tokenizer is already in a portable format before assuming it needs
conversion.** A tokenizer trained on a well-known public vocabulary (GPT-2's, via
`tiktoken`) needs no conversion — it *is* the vocabulary `GPT2TokenizerFast` already
loads. A custom-trained vocabulary is not automatically harder: if it was trained with the
`tokenizers` library (as [Chapter 9](09_tokenization.md) describes this repo's
`custom-gpt-350m` tokenizer being), its `tokenizer.json` is already the exact
serialization format `transformers.PreTrainedTokenizerFast` reads natively. The real
distinction isn't "public vs. custom," it's "already in a format the target library
reads vs. genuinely needs re-serializing."

**5. Verification is the load-bearing step, not a nice-to-have.** A mismapped weight
tensor — a missed transpose, a wrong split boundary — usually still **loads without
error**, because shapes can accidentally line up while values are wrong. The only way to
turn "I think I mapped this correctly" into "I proved I mapped this correctly" is to run
the original model and the converted model on the identical input and compare outputs
numerically (`torch.allclose` on logits, tight tolerance). A conversion script that skips
this check can silently ship a model that *runs* and produces fluent-looking, completely
wrong text.

### Two gotchas that only show up once you actually serve the result

Passing verification proves the *weights* are right. It does not guarantee the serving
engine runs them the way you expect:

- **vLLM's CPU backend repurposes `--gpu-memory-utilization` to mean a fraction of
  *system RAM*, not VRAM** — the flag name is a GPU-era holdover the CPU backend never
  renamed. Passing the GPU-oriented default on a memory-constrained machine causes vLLM to
  try reserving far more RAM than is actually free, and it fails loudly rather than
  scaling down on its own.
- **vLLM's generic `--model-impl transformers` bridge is a fallback for architectures
  vLLM hasn't natively reimplemented, and it is less battle-tested than vLLM's own native
  implementations.** A biased GPT-2 export that loaded and verified correctly via plain
  `transformers.AutoModelForCausalLM` failed under that generic bridge specifically —
  `attn.c_attn.bias` came back as "not initialized from checkpoint" for every layer — while
  vLLM's own native GPT-2 implementation loaded the identical export without issue. Prefer
  the native path whenever the target architecture has one; treat the generic bridge as a
  last resort for architectures that genuinely lack native support.

## Grounded in This Repo's Code

[`from_scratch/custom-gpt-50m/src/gpt/cli/export_vllm.py`](../../from_scratch/custom-gpt-50m/src/gpt/cli/export_vllm.py)
converts this repo's GPT-2-family model into a real `GPT2LMHeadModel` directory:

```python
result = {
    "transformer.wte.weight": source["token_emb.weight"],
    "transformer.wpe.weight": source["pos_emb.weight"],
    ...
}
for layer in range(checkpoint["num_layers"]):
    result.update({
        f"{dst}attn.c_attn.weight": source[f"{src}attn.in_proj.weight"].T,   # layout: transpose
        f"{dst}attn.c_attn.bias":   source[f"{src}attn.in_proj.bias"],       # fusion: matches, no split
        ...
    })
```

with `activation_function="gelu"` set explicitly in the exported config — the exactness
fix from the checklist above, spelled out in the file's own module docstring.

[`from_scratch/custom-gpt-350m/src/gpt/cli/export_vllm.py`](../../from_scratch/custom-gpt-350m/src/gpt/cli/export_vllm.py)
converts this repo's Llama-family model (RoPE, RMSNorm, SwiGLU, no biases — see
[Chapter 35](35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md)) into a real
`LlamaForCausalLM` directory instead:

```python
q_proj, k_proj, v_proj = in_proj.split(embed_size, dim=0)   # fusion: split, not transpose
result.update({
    f"{dst}self_attn.q_proj.weight": q_proj,
    f"{dst}self_attn.k_proj.weight": k_proj,
    f"{dst}self_attn.v_proj.weight": v_proj,
    f"{dst}self_attn.o_proj.weight": source[f"{src}attn.out_proj.weight"],   # layout: no transpose
    ...
})
```

Both scripts share the same verification function before writing anything:

```python
@torch.inference_mode()
def _verify_equivalence(source_path, model, tokenizer):
    _, _, native = load_model(source_path, "cpu")
    native_logits = native(ids)
    hf_logits = model(ids).logits
    if not torch.allclose(native_logits, hf_logits, rtol=1e-4, atol=1e-5):
        raise RuntimeError("HF export failed numerical parity verification...")
```

`make export-vllm` runs this and prints an immutable `exports/vllm/<label>/<checkpoint>-
step-<N>/` directory; `make serve-vllm VLLM_MODEL_DIR=...` serves it. Both projects'
`README.md` document the macOS-specific detour this actually required in practice: vLLM
has no macOS PyPI wheel (CUDA-only dependency resolution), so serving locally means
building vLLM's experimental CPU backend from source into the project's own `.venv` first
— a real, one-time setup cost distinct from the conversion itself.

## Deep-Dive: Why the Conversion Lives at the Architecture Level, Not the Tool Level

Chapter 31's raw-files publication and this chapter's architecture-equivalence conversion
solve different problems, and it's worth being precise about which one you actually need.
Raw-files publication (state_dict + `model.py` + inference code) is the right answer when
the goal is "let someone else run my exact custom architecture" — it changes nothing about
the architecture, so no verification-against-a-different-implementation is even meaningful.
Architecture-equivalence conversion is the right answer when the goal is "run this on
infrastructure that only knows a closed list of architectures" — which is what every
dedicated serving engine, and every generic `AutoModelForCausalLM` caller, actually is. The
two are not competing approaches to the same problem; they answer different questions
("can someone reconstruct my exact code" vs. "can existing infrastructure run this
without knowing my code exists at all"), and a project can reasonably do both for the
same checkpoint.

## Try It Yourself

- Read [`from_scratch/custom-gpt-153m/src/gpt/model.py`](../../from_scratch/custom-gpt-153m/src/gpt/model.py)
  and fingerprint it against the table above — which family does it belong to, and which
  specific lines told you that?
- In `hf_state_dict()` (either export script), temporarily comment out one of the `.T`
  transposes (GPT-2 script) or feed the wrong split boundary to `.split()` (Llama script),
  then run the export with `--no-verify`. Confirm it still writes a directory that loads
  via `AutoModelForCausalLM` without error — then re-run without `--no-verify` and observe
  the parity check catch what "loads fine" didn't.
- Export a checkpoint and serve it once with vLLM's native implementation and once with
  `--model-impl transformers` forced. Compare what happens for an architecture that has
  biases (GPT-2) versus one that doesn't (Llama) — does the generic bridge's bug reproduce
  on both, or only one?

## Common Misconceptions

- **"If `AutoModelForCausalLM` can load it, vLLM can serve it."** Not necessarily — vLLM
  maintains its own separate architecture registry; HF-loadable and vLLM-servable overlap
  heavily but aren't the same set.
- **"A model that loads without a shape error was converted correctly."** Shapes lining up
  is necessary, not sufficient — a transposed or mis-split weight can produce a tensor of
  the exact right shape and completely wrong values. Only a numerical parity check against
  the original model actually proves correctness.
- **"A custom-trained tokenizer always needs to be reserialized for HF."** Only if it
  wasn't already saved in a format `transformers` reads natively — a `tokenizers`-library
  `tokenizer.json` already is that format.
- **"`--gpu-memory-utilization` only matters on a GPU."** On vLLM's CPU backend it's
  repurposed to mean system RAM, and using GPU-scale defaults on a memory-constrained
  machine causes an avoidable startup failure.

## Practice Questions

1. Why can a weight-mapping mistake produce a model that loads successfully and still
   computes something completely wrong, and what's the only check that actually rules
   this out?
2. A custom architecture fuses Q/K/V into one matrix. Explain why converting it to GPT-2's
   format needs no reshaping of that fusion, while converting it to Llama's format does.
3. What real difference is there between "this model is HF-compatible" and "this model is
   vLLM-compatible," and which one implies the other?
4. Why does a `tokenizers`-library-trained custom vocabulary need less conversion work
   than its "custom" label might suggest?

## Key Terms

- **HF-compatible**: loadable via `transformers`' generic `AutoModelForCausalLM`/
  `AutoTokenizer` APIs with no custom code, because the directory matches an
  already-implemented architecture's config/weight-key/tokenizer conventions exactly.
- **vLLM-compatible**: loadable and servable by vLLM specifically — a narrower set than
  HF-compatible, gated by vLLM's own separate model-implementation registry.
- **Architecture-equivalence conversion**: remapping a custom model's weights into an
  existing architecture's exact key names and layout, verified by comparing outputs
  numerically — distinct from raw-files publication, which changes nothing about the
  architecture at all.
- **Weight layout vs. fusion**: two independent things to check per tensor when mapping
  weights — whether the target stores the tensor transposed (layout), and whether it
  fuses/splits projections differently (fusion).
- **Logit-parity verification**: comparing the original and converted model's output
  logits on identical input (`torch.allclose`) before trusting a conversion — the step
  that distinguishes "loads without error" from "computes the same function."
- **Native vs. generic model-impl**: a serving engine's hand-optimized implementation for
  a specific architecture, versus its generic fallback bridge for architectures it hasn't
  natively implemented — the generic path is typically less battle-tested.
