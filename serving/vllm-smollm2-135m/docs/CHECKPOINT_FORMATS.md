# PyTorch `.pt` versus `safetensors` model checkpoints

The filename extension alone does not describe a model completely. Both formats normally
store **weights**; a runnable language-model deployment also needs model configuration,
tokenizer files, and an architecture implementation.

## Short answer

Use **safetensors** for model artifacts you distribute or serve. Use a PyTorch `.pt` or
`.pth` checkpoint during trusted research/training workflows when it contains training
state or Python-specific objects that safetensors intentionally cannot represent.

For vLLM, `--load-format auto` prefers safetensors and falls back to PyTorch-format
weights. `--load-format pt` selects the PyTorch weight loader; it does not make an
arbitrary Python model or state dict automatically vLLM-compatible.

## What each format contains

| Property | PyTorch `.pt` / `.pth` / many `.bin` files | `.safetensors` |
| --- | --- | --- |
| Usual writer | `torch.save(...)` | `safetensors.torch.save_file(...)` or Hugging Face `save_pretrained(..., safe_serialization=True)` |
| Data representation | Python serialization (pickle-based metadata plus tensor storages) | Strict tensor data plus a small JSON header |
| Can contain arbitrary Python objects | Yes | No; tensors and limited metadata only |
| Untrusted-file safety | Unsafe to load by default; unpickling can execute code | Designed to avoid arbitrary code execution from the weight file |
| Lazy/selective tensor loading | Possible in some PyTorch modes but format is not designed around a simple safe tensor index | Supported by the format/API; useful for partial and sharded loading |
| Optimized for model distribution | Legacy/common, but less desirable | Yes |
| Stores optimizer/RNG/trainer state | Yes | No; save this separately if needed |

`.bin` is ambiguous: on Hugging Face it often means a PyTorch checkpoint, but an extension
is not a security guarantee. Identify it by its producer and loading instructions.

## Why safetensors is preferred

### Security

`torch.load` uses Python unpickling. PyTorch explicitly warns never to load untrusted
files. Modern `weights_only=True` narrows the permitted unpickling behavior, but it is a
mitigation—not a reason to download unknown checkpoints and load them automatically.

Safetensors stores tensor names, shapes, dtypes, offsets, and optional metadata in a
bounded header followed by raw tensor bytes. It intentionally does not serialize an
arbitrary Python object graph, which removes the usual pickle remote-code-execution path.

This distinction is especially important for a model server: loading weights happens
inside a process that may have GPU access, network access, credentials, or production
data access.

### Operational behavior

Safetensors has predictable tensor metadata and supports accessing selected tensors or
slices without deserializing unrelated Python objects. In practice this can improve model
loading pipelines, sharding, and inspection. Actual startup speed also depends on disk,
network cache, vLLM version, loader configuration, and GPU transfer time—do not promise a
fixed speedup merely from changing the file extension.

### Interoperability

The Hugging Face ecosystem, vLLM, MLX, and many model tools recognize safetensors. A
standard Hugging Face directory with sharded safetensors plus an index file is a portable
deployment artifact.

## Files required to serve a model

Neither format replaces these model assets:

```text
my-model/
├── config.json                       # architecture and dimensions
├── tokenizer.json / tokenizer.model   # text ↔ token IDs
├── tokenizer_config.json
├── special_tokens_map.json
├── generation_config.json             # optional defaults
├── model.safetensors                  # one weight shard, or
├── model-00001-of-00002.safetensors   # multiple shards
└── model.safetensors.index.json       # required mapping for sharded weights
```

For a custom architecture, include or install the trusted model code too. The checkpoint
does not define the Python classes needed to execute the tensors.

## Recommended workflow: training checkpoint to serving artifact

During training, a checkpoint may contain far more than weights:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "global_step": 12000,
    "rng_state": ...,
}
```

Keep that trusted training checkpoint privately. Create a separate, immutable serving
export after training:

```python
# Run only on a checkpoint you trust.
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/trusted/training-export")
tokenizer = AutoTokenizer.from_pretrained("/trusted/training-export")

model.save_pretrained("/release/my-model", safe_serialization=True)
tokenizer.save_pretrained("/release/my-model")
```

For a raw trusted state dict, first instantiate the exact model architecture and load it
strictly, then perform the same export:

```python
import torch
from my_model import MyModelForCausalLM

model = MyModelForCausalLM.from_config(...)
state = torch.load("/trusted/model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.save_pretrained("/release/my-model", safe_serialization=True)
```

`weights_only=True` is preferable for a known plain state dict, but it does not make an
untrusted file safe. `strict=True` catches missing/unexpected parameter names; do not
use `strict=False` merely to hide a conversion error.

`save_pretrained(..., safe_serialization=True)` is preferred for Transformers models
because it handles standard configuration files and tied weights correctly. A direct
`save_file(model.state_dict(), ...)` can fail or need special handling when two parameter
names share the same underlying tensor storage.

## Verify a conversion before deployment

Conversion is successful only when the exported model behaves the same under controlled
conditions.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

source = "/trusted/training-export"
export = "/release/my-model"
prompt = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained(source)
original = AutoModelForCausalLM.from_pretrained(source).eval()
converted = AutoModelForCausalLM.from_pretrained(export).eval()
inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    original_ids = original.generate(**inputs, do_sample=False, max_new_tokens=12)
    converted_ids = converted.generate(**inputs, do_sample=False, max_new_tokens=12)

assert torch.equal(original_ids, converted_ids)
print(tokenizer.decode(converted_ids[0], skip_special_tokens=True))
```

Then run a vLLM smoke test with the exported directory:

```bash
vllm serve /release/my-model \
  --load-format safetensors \
  --dtype auto \
  --max-model-len 2048
```

Test a fixed greedy request (`temperature: 0`) through `/v1/chat/completions` or
`/v1/completions`, depending on whether the model has a chat template. Compare generated
token IDs or decoded output with the reference runtime before optimizing quantization,
parallelism, or throughput.

## vLLM load-format choices

| vLLM flag | Meaning | Use it when |
| --- | --- | --- |
| `--load-format auto` | Prefer safetensors; fall back to PyTorch-format weights | Normal default for a well-packaged model. |
| `--load-format safetensors` | Require safetensors | Production model release; fail early if artifacts are wrong. |
| `--load-format pt` | Use PyTorch binary weight loader | Trusted legacy/checkpoint artifacts for an already supported or registered architecture. |

The `auto` fallback is convenience, not a security policy. A production release pipeline
should validate the artifact format, checksum the exact model revision, and use
`--load-format safetensors` intentionally.

## Common misunderstandings

| Misunderstanding | Reality |
| --- | --- |
| “I changed `model.pt` to `model.safetensors`.” | Renaming does not convert bytes. Use a real export/conversion. |
| “Safetensors is a model format.” | It is a tensor-weight format. You still need config, tokenizer, and architecture code. |
| “`--load-format pt` can serve any PyTorch model.” | vLLM must still recognize or register the architecture and weight mapping. |
| “`weights_only=True` makes internet checkpoints safe.” | It reduces pickle capability but does not make untrusted input generally safe. |
| “Safetensors always loads faster.” | It enables efficient safe access; actual load time is deployment-specific. |
| “Optimizer state belongs in the serving artifact.” | No. Preserve it in training storage; export inference weights separately. |

## Release checklist

1. Retain the original training checkpoint in a trusted, access-controlled location.
2. Export model weights as safetensors into a standard model directory.
3. Pin and record model, tokenizer, code, and vLLM versions.
4. Compute/check a cryptographic checksum for every release artifact.
5. Verify strict weight loading and deterministic output parity.
6. Run a vLLM startup and API smoke test on the target hardware.
7. Promote the immutable serving artifact; never overwrite a release in place.

## Sources

- [PyTorch `torch.load` security warning and `weights_only`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch serialization semantics](https://docs.pytorch.org/docs/main/notes/serialization.html)
- [Safetensors documentation](https://huggingface.co/docs/safetensors/)
- [Hugging Face safetensors conversion guidance](https://huggingface.co/docs/safetensors/convert-weights)
- [vLLM load-format configuration](https://docs.vllm.ai/en/latest/api/vllm/config/load/)
