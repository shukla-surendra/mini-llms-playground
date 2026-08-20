# Integrating a custom PyTorch model with vLLM

This guide explains how to serve a custom PyTorch (`.pt` / `.pth`) model with vLLM.
The crucial point is simple:

> A checkpoint contains weights, not a complete serving implementation.

vLLM needs to know the model architecture, tokenizer, attention/cache behavior, and how
checkpoint parameter names map to its model modules. A raw `torch.save(model.state_dict(),
"model.pt")` file does not provide that information by itself.

## 1. Decide which kind of custom model you have

| Situation | Recommended integration path | Why |
| --- | --- | --- |
| Standard architecture already supported by vLLM, with your own weights | Make a Hugging Face-style model directory; use `vllm serve <directory>` | Lowest risk and most maintainable. |
| Custom architecture implemented as a Transformers model | Use vLLM's Transformers modeling backend | Reuses the Transformers implementation; no vLLM fork required. |
| Architecture not supported by Transformers/vLLM | Package an out-of-tree vLLM model plugin and register it | vLLM must be taught how to instantiate and execute it. |
| Arbitrary `.pt` state dict with no config/tokenizer/model code | Do not point vLLM at it yet | First create the artifacts below or use a custom PyTorch/Transformers server. |

`--load-format pt` means “load PyTorch-format weight files.” It does **not** mean
“infer any arbitrary PyTorch architecture.” vLLM still needs a supported or registered
model class.

For a detailed comparison and conversion procedure, see
[CHECKPOINT_FORMATS.md](CHECKPOINT_FORMATS.md).

## 2. Required deployment artifacts

A portable model directory should look like this:

```text
my-model/
├── config.json
├── generation_config.json            # optional, but pin intentional defaults
├── tokenizer.json                    # or vocab/merges / SentencePiece files
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors                 # preferred, or pytorch_model.bin / *.pt shards
├── modeling_my_model.py              # needed only for custom Transformers code
└── configuration_my_model.py          # needed only for custom Transformers code
```

`config.json` is the contract between model repository and runtime. It should identify
the architecture and model dimensions, for example:

```json
{
  "model_type": "my_model",
  "architectures": ["MyModelForCausalLM"],
  "vocab_size": 32000,
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "max_position_embeddings": 8192,
  "torch_dtype": "bfloat16",
  "auto_map": {
    "AutoConfig": "configuration_my_model.MyModelConfig",
    "AutoModel": "modeling_my_model.MyModel",
    "AutoModelForCausalLM": "modeling_my_model.MyModelForCausalLM"
  }
}
```

The names and fields must match the actual code and checkpoint. Do not copy this example
unchanged. A missing tokenizer or an incorrect `architectures` value is as fatal as a
missing weight file.

## 3. Path A: supported architecture with custom weights

This is the preferred path. If your model is a Llama-family, Qwen-family, GPT-style, or
another architecture listed in vLLM's supported-model table, preserve that architecture
in the config and convert/export your weights into its Hugging Face parameter naming.

```bash
vllm serve /absolute/path/to/my-model \
  --tokenizer /absolute/path/to/my-model \
  --load-format safetensors \
  --dtype auto \
  --max-model-len 2048
```

If the files are PyTorch binary checkpoints instead of safetensors:

```bash
vllm serve /absolute/path/to/my-model --load-format pt
```

`pt` selects the weight-file loader. It does not change the model implementation. Prefer
`safetensors` in production: it is safer to load, supports efficient indexing/sharding,
and avoids arbitrary Python pickling behavior associated with legacy PyTorch checkpoints.

### Weight-conversion checklist

1. Load the original checkpoint in trusted offline code.
2. Create the exact target model architecture.
3. Map every state-dict key, including fused versus separate Q/K/V projections.
4. Verify tensor shapes, dtype, tied embeddings, and rotary-position configuration.
5. Save a standard Hugging Face directory, preferably sharded safetensors.
6. Compare logits or greedy generated token IDs on a fixed prompt before attempting vLLM.

Never accept an untrusted `.pt` file from a user: PyTorch pickle-based loading can execute
code. Convert only trusted checkpoints in an isolated build environment.

## 4. Path B: custom Transformers model

If the architecture is not natively in vLLM but is a correct custom Transformers model,
place its configuration/modeling source and `auto_map` entries beside the weights. Then:

```bash
vllm serve /absolute/path/to/my-model \
  --trust-remote-code \
  --model-impl transformers \
  --dtype auto
```

`--trust-remote-code` imports Python from the model directory. Treat it like installing
an unreviewed package: pin a commit, inspect it, and only use it in a trusted environment.

For vLLM's Transformers modeling backend, the custom base model must be compatible with
vLLM attention integration. In practice this means propagating the kwargs supplied from
the base model through blocks into attention, rather than discarding them; the custom
attention must work with the backend's cache/attention execution. Implement custom logic
in the base model (`MyModel`) rather than only a task wrapper
(`MyModelForCausalLM`). Supported model shapes include common decoder-only, encoder-only,
and MoE forms, but feature compatibility must be tested on the actual architecture.

### Validate before vLLM

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/absolute/path/to/my-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
print(type(model).__name__, len(tokenizer))
PY
```

Only after this works should you invoke vLLM. A `transformers` load failure is a model
package issue, not a vLLM scheduler issue.

## 5. Path C: out-of-tree vLLM model plugin

Use a plugin when neither native vLLM nor the Transformers backend can execute the
architecture. Keep the plugin in its own installable Python package so every vLLM worker
can import the same version.

```text
my-vllm-model-plugin/
├── pyproject.toml
└── my_vllm_model/
    ├── __init__.py
    └── model.py
```

Register it using the `vllm.general_plugins` entry-point group:

```toml
[project]
name = "my-vllm-model-plugin"
version = "0.1.0"
dependencies = ["vllm==<your-tested-version>"]

[project.entry-points."vllm.general_plugins"]
register_my_model = "my_vllm_model:register"
```

```python
# my_vllm_model/__init__.py
def register() -> None:
    from vllm import ModelRegistry

    # A string performs a lazy import. This avoids CUDA initialization in the
    # parent process before vLLM creates its worker processes.
    ModelRegistry.register_model(
        "MyModelForCausalLM",
        "my_vllm_model.model:MyModelForCausalLM",
    )
```

The string `MyModelForCausalLM` must exactly match `config.json`'s
`architectures` entry. Install the plugin into the **same environment** as vLLM:

```bash
uv pip install -e /absolute/path/to/my-vllm-model-plugin
VLLM_PLUGINS=register_my_model \
  vllm serve /absolute/path/to/my-model --load-format safetensors
```

`VLLM_PLUGINS` is optional when no filtering is wanted; setting it makes plugin loading
explicit and easier to audit. Test distributed configurations too: vLLM may create
multiple worker processes, and every one must discover the plugin.

### What the model implementation must handle

An out-of-tree model is more than a normal `torch.nn.Module`. Its implementation must
be compatible with vLLM's model runner, attention/KV-cache operations, weight-loading
rules, and requested features. Before adding advanced features, get this narrow sequence
working:

1. One GPU, one request, greedy generation.
2. Correct token output compared with the reference PyTorch implementation.
3. Concurrent requests / continuous batching.
4. Long contexts and KV-cache limits.
5. Streaming, structured output, adapters, and distributed parallelism only as needed.

If your model has nonstandard attention, state layout, MoE routing, multimodal inputs, or
custom cache semantics, a full native vLLM implementation may be needed. Multimodal
models additionally implement vLLM's `SupportsMultiModal` interface.

## 6. Weight naming and loading

Loading succeeds only if every expected parameter can be mapped correctly. Typical
pitfalls include:

| Problem | Example | Resolution |
| --- | --- | --- |
| QKV layout differs | Source has `q_proj`, `k_proj`, `v_proj`; target uses fused `qkv_proj` | Concatenate/slice using the target architecture's expected order. |
| Tensor parallel layout differs | Source tensor is whole; worker expects a shard | Use vLLM's parallel-aware weight loader or implement the mapping. |
| Tied weights forgotten | `lm_head` should share input embedding | Preserve/recreate the tie and compare logits. |
| Dtype mismatch | FP32 checkpoint served as BF16 | Convert deliberately and verify output tolerance. |
| Config lies | `num_key_value_heads` mismatches tensors | Fix config or export; never suppress shape errors. |
| Tokenizer changed | Same weights, different special-token IDs | Ship and pin the original tokenizer files. |

For an architecture already supported by vLLM, inspect its existing loader as the source
of truth for expected checkpoint key names. Avoid broad `strict=False` loads: they can
hide a partially loaded model that produces plausible but wrong text.

## 7. Technical validation plan

Use deterministic, layered tests. Record model revision, tokenizer revision, vLLM
version, GPU type, dtype, and all serving flags with every result.

```text
Package test        config + tokenizer + weights load in Transformers
Reference test      fixed prompt produces expected token IDs/logits
vLLM smoke test     server starts; /v1/models and one completion succeed
Parity test         reference and vLLM agree under greedy decoding
Load test           concurrent short/long prompts, TTFT, throughput, KV-cache pressure
Failure test        invalid input, cancellation, context limit, OOM recovery
```

Exact floating-point logits may vary by dtype and kernel. Compare the generated token
sequence under deterministic settings first, then define realistic numerical tolerances.

## 8. Serving command and API test

After model loading works, serve the local directory normally:

```bash
vllm serve /absolute/path/to/my-model \
  --served-model-name my-model-v1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --api-key "$MY_MODEL_API_KEY"
```

Then test through the same OpenAI-compatible interface used by any other vLLM model:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $MY_MODEL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model-v1",
    "messages": [{"role": "user", "content": "Say hello."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

Use `/v1/completions` instead if the model has no chat template. See
[VLLM_API_REFERENCE.md](VLLM_API_REFERENCE.md) for API details.

## 9. Decision boundary: when not to use vLLM

Do not force vLLM into the first phase of every research model. Use a small custom
PyTorch/Transformers FastAPI server when you need to iterate on an experimental
architecture, inspect activations, use unsupported control flow, or run CPU-only. Move
to vLLM after the architecture and checkpoint format stabilize and there is enough
concurrent accelerator traffic to benefit from continuous batching and KV-cache
management.

## Sources

- [vLLM supported models and Transformers backend](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM model registration](https://docs.vllm.ai/en/latest/contributing/model/registration/)
- [vLLM plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/)
- [vLLM weight load formats](https://docs.vllm.ai/en/latest/api/vllm/config/load/)
