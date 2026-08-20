# vLLM and LLM serving: from first principles to production

This guide uses this project as a concrete example. It is written as a technical
learning path, not an interview script, but it deliberately covers the concepts an LLM
serving engineer should be able to reason about and explain.

## 1. What happens when you run `make serve`?

`make serve` does **not** itself run the model. It runs this project's launcher:

```text
make serve
  └─ uv run --no-sync python serve.py
       ├─ inspect PyTorch / operating system
       ├─ choose a backend
       └─ start that backend as a child process
```

The backend is selected at runtime:

| Detected environment | Command launched by `serve.py` | Is vLLM serving requests? |
| --- | --- | --- |
| NVIDIA GPU visible to PyTorch | `vllm serve HuggingFaceTB/SmolLM2-135M-Instruct ...` | Yes |
| Native Apple Silicon | `vllm serve mlx-community/SmolLM2-135M-Instruct ...` through vLLM-Metal | Yes |
| No supported accelerator, including this WSL machine | `python -m cpu_server ...` | No; Transformers provides a compatible CPU fallback |

Run the following to see the exact command before starting anything:

```bash
make check
```

On the current CPU-only WSL setup, `make check` should print `backend=cpu` and a command
containing `-m cpu_server`. That is intentional. The usual PyPI vLLM distribution is
CUDA-oriented; CPU vLLM needs its own CPU wheel or source build. The fallback makes this
small project runnable now, without pretending that a CUDA vLLM wheel can be forced into
CPU operation. The API stays compatible: clients still call `/v1/chat/completions`.

When CUDA is available, the important hand-off is this call in
[`serve.py`](../serve.py):

```python
subprocess.run(["vllm", "serve", MODEL, ...], check=True)
```

`vllm` is a command-line program installed by the `vllm` Python package. Its `serve`
subcommand loads the model and starts vLLM's OpenAI-compatible HTTP server. The launcher
does not implement vLLM; it selects and invokes it with safe small-model defaults.

## 2. The smallest useful mental model

A language model receives tokens and predicts the next token. To answer a prompt, it
repeats that prediction one token at a time:

```text
"The capital of France is" → " Paris" → "." → end
```

Serving an LLM means making that loop available to many clients over a network while
keeping latency, memory, cost, and reliability under control.

There are three layers to distinguish:

| Layer | Responsibility | Example in this project |
| --- | --- | --- |
| Client/API layer | Sends prompts and receives text or token streams | `curl ... /v1/chat/completions` |
| Serving engine | Batches requests, runs the model, manages memory | vLLM on CUDA/Metal |
| Model runtime | Loads weights and executes tensor operations | CUDA/PyTorch, MLX, or PyTorch CPU |

`transformers` is excellent for a simple one-request-at-a-time model loop. vLLM is a
serving engine built to keep accelerators busy under concurrent traffic.

## 3. Model, tokenizer, and chat format

The server loads more than a weight file.

- **Weights** are the learned numerical parameters.
- **Architecture/configuration** tells the runtime how to construct the network.
- **Tokenizer** converts text to token IDs and back.
- **Chat template** converts a list of role-tagged messages into the exact prompt format
  used during instruction tuning.

For example, an OpenAI-style request contains messages:

```json
{
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "messages": [{"role": "user", "content": "Explain caching in one sentence."}],
  "max_tokens": 80
}
```

The server applies the model's chat template, tokenizes the resulting text, generates
new token IDs, then decodes them back into the assistant response. Sending a raw prompt
to a chat model can work, but it often produces worse behavior because the model no
longer sees the role markers it learned from.

## 4. The two phases of generation

Generation has two very different performance phases.

```text
Prompt tokens ── prefill ──> first generated token ── decode ──> next token ──> ...
```

**Prefill** processes the entire input prompt. It is compute-heavy and determines
time-to-first-token (TTFT). A long document or a huge retrieval context makes this
phase expensive.

**Decode** produces one new token per active request per step. It repeatedly reads prior
attention state, making it strongly memory-bandwidth-sensitive. Decode determines the
streaming pace users experience after the first token.

Useful metrics follow from that split:

| Metric | Meaning | Why it matters |
| --- | --- | --- |
| TTFT | Time until the first streamed token | Most visible responsiveness metric |
| Inter-token latency | Time between streamed tokens | Perceived writing speed |
| Tokens/s | Generated tokens over time | Throughput / capacity |
| Request latency | End-to-end completion time | SLO and user wait time |
| Queue time | Time waiting before model execution | Saturation signal |

Low TTFT and high total throughput are related but not identical. A system can maximize
batch throughput by waiting longer to collect work, which may harm interactive latency.

## 5. Why inference consumes so much memory: the KV cache

In every Transformer layer, attention needs keys (K) and values (V) for tokens already
seen. Recomputing them for the whole prompt at every generated token would be extremely
slow, so servers retain them in a **KV cache**.

Conceptually:

```text
new token query ────────┐
                          ├─ attention against cached K/V for earlier tokens
cached K/V: prompt + prior generated tokens ─┘
```

KV cache memory grows approximately with:

```text
active requests × tokens retained per request × layers × KV heads × head size × bytes/value × 2
```

This is separate from model weight memory. A small model can therefore still run out of
memory if many users send long contexts or request long outputs. Context length is a
capacity setting, not merely an application feature.

## 6. What makes vLLM different: PagedAttention

Traditional serving runtimes often reserve one contiguous KV-cache region per request.
Requests have different prompt and output lengths, so this creates fragmentation and
over-reservation: memory set aside for a 4,000-token completion may mostly remain empty.

vLLM divides KV cache into fixed-size blocks and gives each request a logical list of
blocks, similar to virtual-memory pages:

```text
Request A logical tokens: [ block 4 ][ block 19 ][ block 7 ]
Request B logical tokens: [ block 2 ][ block 8 ]
Physical GPU memory:      blocks can be non-contiguous and reused
```

This technique is called **PagedAttention**. Its practical benefits are reduced memory
waste, more simultaneous sequences, and efficient sharing of identical prompt prefixes.
It is not OS paging: the relevant blocks are GPU-resident cache blocks, not disk pages.

## 7. Continuous batching and scheduling

A naive batch waits until every request in it completes. That is a poor fit for text
generation because one request may produce 10 tokens and another 1,000.

vLLM uses **continuous batching** (also called iteration-level scheduling). At each model
step, it can remove completed requests and add waiting work while other requests continue
generating. This improves accelerator utilization and tail latency.

```text
step 1: [A prefill, B prefill]
step 2: [A decode,  B decode, C prefill]
step 3: [A decode, C decode, D prefill]  # B finished; D joined
```

The scheduler balances several limits: token budget per iteration, cache blocks, maximum
active sequences, output length, request priority, and admission control. Batching is
why a specialized server has a major advantage over wrapping a single
`model.generate()` call in a web endpoint.

## 8. Running the project

### Install and start

```bash
make install
make check
make serve
```

`make install` chooses dependencies from the actual machine:

```text
Apple Silicon → vLLM-Metal
NVIDIA CUDA  → vLLM
otherwise    → Transformers CPU fallback
```

The first launch downloads the selected Hugging Face model and stores it in the local
Hugging Face cache. Later starts reuse the cache. The default port is `8004` and the
default host is `127.0.0.1`, which intentionally prevents remote access.

### Call the server

```bash
curl http://127.0.0.1:8004/v1/models

curl http://127.0.0.1:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role":"user","content":"What is continuous batching?"}],
    "max_tokens": 80,
    "temperature": 0.2
  }'
```

Use the model ID returned by `/v1/models`; on Apple Metal it is the `mlx-community/...`
variant. In a real client, prefer streaming (`"stream": true`) when the backend and
client library support it: it improves perceived latency by returning tokens as they are
generated.

## 9. Important vLLM controls

This project's CUDA invocation keeps defaults intentionally modest:

```bash
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.55
```

| Setting | Effect | Trade-off |
| --- | --- | --- |
| `--max-model-len` | Maximum prompt plus generated context | More context consumes more KV cache |
| `--gpu-memory-utilization` | Fraction of GPU memory vLLM can reserve | Higher improves capacity but leaves less room for other work |
| `--max-num-seqs` | Maximum concurrent sequences | Higher boosts concurrency until cache/scheduler limits dominate |
| `--max-num-batched-tokens` | Per-iteration token budget | Larger can raise throughput but may worsen TTFT |
| `--dtype` | Weight/compute precision | Lower precision saves memory and can improve speed; verify quality and hardware support |
| `--tensor-parallel-size` | Number of GPUs splitting each layer | Fits/larger models; adds communication overhead |
| `--pipeline-parallel-size` | Number of sequential layer stages | Fits very large models; can reduce single-request latency efficiency |

Do not copy values blindly. Start with a target workload: expected concurrent users,
typical input size, output limit, latency target, model, and GPU memory. Then load-test
while increasing one limit at a time.

## 10. Precision and quantization

Model parameters and activations can use different numeric formats.

| Format | Typical use | Main benefit | Main caution |
| --- | --- | --- | --- |
| FP32 | CPU/debugging | Numerical conservatism | Large and slow on GPUs |
| FP16 | Many GPUs | Half the FP32 weight memory | May need care on some hardware/models |
| BF16 | Modern data-center GPUs | FP32-like exponent range | Hardware support required |
| FP8 | New accelerators | Major memory/bandwidth reduction | Calibration/kernel/model support matter |
| INT8 / INT4 | Quantized deployments | Much smaller weights | Quality, accuracy, and kernel compatibility vary |

Quantization reduces weight memory, but it does not eliminate KV-cache memory. At high
concurrency and long context, KV cache can become the dominant constraint. Test output
quality, tool-call formatting, structured JSON behavior, and latency after every change.

## 11. Scaling beyond one GPU

There are distinct ways to use multiple GPUs; they solve different problems.

| Technique | Split | Best reason to use it | Cost |
| --- | --- | --- | --- |
| Replicas / data parallelism | Independent copies of the model | More request throughput | Each replica needs model memory |
| Tensor parallelism | Tensor operations within a layer | Model does not fit on one GPU | Frequent inter-GPU communication |
| Pipeline parallelism | Consecutive groups of layers | Model is too large for one GPU | Pipeline bubbles and more operational complexity |
| Expert parallelism | Mixture-of-Experts experts | Large MoE models | Routing and communication complexity |

For a 135M model, one GPU is normally enough. Multi-GPU techniques are important because
the same serving principles apply to larger models, but adding parallelism to a small
model can make it slower.

## 12. Prefix caching, LoRA, and structured output

**Prefix caching** reuses KV-cache work for identical prompt prefixes. It is especially
valuable when every request starts with the same long system prompt, tool definitions,
or document header. Small changes to that shared prefix can destroy cache hits, so prompt
construction should be stable.

**LoRA adapters** add small trainable low-rank weight updates to a base model. A serving
system can often load one base model and select an adapter per request, which is cheaper
than hosting a full copy of the model for every tenant. Treat adapters as deployable
artifacts: version them, validate compatibility with the base model, and control who can
request them.

**Structured output** constrains generation to a grammar, JSON schema, or tool-call
format. It is more reliable than merely asking a model to "output JSON", but it adds
runtime work and does not guarantee that the semantic contents are correct. Validate all
model outputs before using them in an action.

## 13. Production API and deployment practices

An OpenAI-compatible API is useful because standard SDKs and tools can talk to a local
server by changing only `base_url`. Compatibility is not authorization or security.

Use these boundaries in a production deployment:

- Bind the model process to localhost or a private network; put a reverse proxy/API
  gateway in front of it.
- Authenticate and authorize callers. Do not expose an unauthenticated model port to
  the internet.
- Enforce maximum input tokens, output tokens, request body size, concurrency, and
  per-tenant rate limits before requests reach the GPU.
- Set request timeouts and cancellation propagation. A disconnected client should not
  continue consuming generation capacity indefinitely.
- Treat prompts and outputs as potentially sensitive data. Minimize logs, redact where
  appropriate, and set a retention policy.
- Pin model revisions and container/dependency versions; a mutable model repository can
  silently change behavior.

For Kubernetes or a process supervisor, readiness should mean: process started, model
loaded, and a small inference health check succeeds. Liveness should be cheaper and must
not cause a thundering herd of expensive inference requests.

## 14. Observability and capacity planning

At minimum, measure these separately by model and tenant:

```text
requests and errors
queue duration
TTFT and inter-token latency percentiles
input/output token counts
running and waiting requests
GPU utilization, memory use, and KV-cache utilization
tokens per second
cache hit rate (when prefix caching is enabled)
```

Percentiles matter. An average latency can look healthy while a small number of very long
requests cause unacceptable p95/p99 waits. Correlate queue time with cache use and input
length to distinguish saturation from a slow model kernel.

Capacity planning starts with measured workload distributions, not a parameter-count
rule. Estimate average and p95 prompt/output tokens, then load-test at your desired
arrival rate. Reserve headroom: running a GPU at maximum utilization can make tail
latency highly unstable when traffic bursts.

## 15. Common failures and how to reason about them

| Symptom | Likely cause | First response |
| --- | --- | --- |
| `Failed to infer device type` | CUDA vLLM wheel on a machine with no CUDA device | Use this project's CPU fallback or a proper vLLM CPU build |
| Out-of-memory during startup | Weights + configured KV cache exceed free accelerator memory | Lower memory utilization/context, use smaller/quantized model, stop competing jobs |
| OOM only under traffic | KV cache saturated by concurrency or long contexts | Cap tokens/concurrency; inspect cache and queue metrics |
| Good throughput but poor TTFT | Batch/token budget too large or queueing too aggressive | Tune for latency and inspect scheduling metrics |
| Model returns poor chat answers | Wrong model type, chat template, or prompt format | Use an instruction-tuned checkpoint and its tokenizer template |
| Server is reachable unexpectedly | Bound to a public interface without network controls | Bind localhost/private interface and add gateway controls |
| Different behavior after restart | Model/dependency revision changed | Pin revisions and record runtime configuration |

## 16. A practical progression

1. Run this model locally and make one successful request.
2. Compare the CPU fallback with CUDA vLLM on the same prompts; observe throughput and
   concurrent-request behavior rather than only one short response.
3. Turn on streaming in a client and measure TTFT versus full completion time.
4. Add concurrent load with a mix of short and long prompts. Observe queueing and tail
   latency.
5. Change one serving limit, measure again, and record the trade-off.
6. Deploy behind authentication and rate limits before allowing any untrusted caller.
7. Move to a larger model only after the measurements explain why the smaller one is
   insufficient.

## 17. Concepts to be able to explain clearly

By the end of this guide, you should be able to explain, in your own words:

- Why model weights and KV cache are different memory consumers.
- Why prefill and decode have different latency/performance characteristics.
- Why continuous batching is better than waiting for a static batch to finish.
- How PagedAttention avoids much of the KV-cache fragmentation problem.
- Why quantizing weights does not automatically solve long-context concurrency.
- When replicas, tensor parallelism, and pipeline parallelism are each appropriate.
- Why an OpenAI-compatible endpoint is convenient but needs separate security controls.
- Why CPU fallback, CUDA vLLM, and Apple Metal are different execution backends even when
  they expose the same HTTP API.

## Further reading

- [vLLM API reference for this project](VLLM_API_REFERENCE.md)
- [Custom PyTorch model integration](CUSTOM_PYTORCH_MODEL_INTEGRATION.md)
- [PyTorch checkpoint versus safetensors formats](CHECKPOINT_FORMATS.md)
- [Model artifacts and openness terminology](MODEL_ARTIFACTS_AND_OPENNESS.md)
- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM quickstart and OpenAI-compatible serving](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [vLLM installation and hardware backends](https://docs.vllm.ai/en/latest/getting_started/installation/)
- [vLLM CPU installation requirements](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
