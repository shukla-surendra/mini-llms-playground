# From Script to API: Serving a Model for Real

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 4 — Serving: Turning a
Trained Model Into Something You Can Talk To. Builds on
[Chapter 21](21_inference_mechanics_decoding_sampling_and_kv_cache.md)'s decoding
mechanics — this chapter is about the layer *around* generation: turning a Python
function you can call from a script into an HTTP endpoint anyone can call, and what that
wrapper does and deliberately doesn't need to do at small scale.

## In Plain English

A trained model plus a generation function is already "usable" from a Python REPL or a
command-line script. Serving means putting a thin network-facing wrapper around that same
function — something that listens for HTTP requests, validates what came in, calls
generation exactly as the script already does, and sends the result back as JSON. The
generation logic doesn't change at all; only how it gets *invoked* changes, from "call a
Python function" to "receive a request over the network."

## The First-Principles Explanation

### The wrapper is thin by design

A minimal LLM-serving endpoint needs three things, and nothing else is load-bearing at
small scale:

1. **Load the model once, at process startup** — not per-request. Loading a checkpoint
   and rebuilding the architecture from its saved config (see
   [Chapter 27](27_checkpointing_and_resuming_training.md)) is comparatively expensive;
   doing it once and keeping the model resident in memory for the process's lifetime is
   what makes each individual request fast.
2. **A request schema** — a typed description of what a caller must send (a prompt, and
   the decoding knobs from [Chapter 21](21_inference_mechanics_decoding_sampling_and_kv_cache.md):
   temperature, top-k, top-p, max tokens) and validation that rejects malformed input
   before it ever reaches the model.
3. **A route that calls the exact same generation function the command-line script
   calls** — the server is not a second implementation of generation, it's a second
   *caller* of the one implementation.

### Why a health-check endpoint is not an afterthought

A `/health` route that reports which checkpoint is loaded, how many parameters it has,
and what device it's running on serves a real operational purpose distinct from "is the
process alive": it confirms *which trained model* is actually being served, without
having to trust that a deploy script pointed at the right checkpoint file. Since
checkpoints from the same project can differ enormously in how well-trained they are (see
[Chapter 15](15_evaluating_a_model_while_training.md)), being able to ask the running
server "what step are you actually serving" is a genuine debugging tool, not decoration.

### What synchronous, one-request-at-a-time serving gets right, and where it stops being enough

A server that handles one generation request fully before starting the next is a correct,
legitimate design at small model sizes and low request volume — it is not a shortcut
version of "real" serving, it's the appropriately-sized solution for that regime.  It
stops being enough once either of two things happens: request volume grows past what one
process can handle sequentially (the fix is **continuous batching** — interleaving
multiple in-flight generations so GPU/CPU time isn't idle between individual requests'
token-by-token steps), or the model grows large enough that per-request compute and
memory — including the KV cache from
[Chapter 21](21_inference_mechanics_decoding_sampling_and_kv_cache.md) — becomes the
binding constraint rather than request scheduling. Both are real, well-studied problems
with dedicated serving engines (vLLM and similar) built specifically to solve them — see
[Chapter 23](00_roadmap.md) (planned) and this curriculum's explicit hand-off to
`platform-lab/fundamentals/gpu_infrastructure/`'s Phase 5 (LLM Serving) chapters for that
depth. Reaching for that machinery before request volume or model size actually demand it
is solving a problem you don't have yet.

## Grounded in This Repo's Code

[`from_scratch/custom-gpt-6m/src/gpt/inference/server.py`](../../from_scratch/custom-gpt-6m/src/gpt/inference/server.py)
is the minimal wrapper described above, almost line for line:

```python
app = FastAPI(title="TinyStories GPT API")
model, tokenizer, ckpt = load_model_and_tokenizer(CHECKPOINT_PATH, device)   # once, at startup

class GenerateRequest(BaseModel):        # the typed request schema
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "params": model.num_parameters(),
            "checkpoint_step": ckpt.get("step")}   # which checkpoint, concretely

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    text = generate(model, tokenizer, req.prompt, ...,       # the SAME generate() function
                     temperature=req.temperature, top_k=req.top_k, top_p=req.top_p)
    return GenerateResponse(prompt=req.prompt, completion=..., model_step=ckpt.get("step"))
```

`from_scratch/custom-gpt-10m` and `custom-gpt-153m` implement the identical pattern
(FastAPI app, `/health`, `/generate`, model loaded once at startup) against their own
`inference.generate()` — see each project's
[`docs/API_SERVER.md`](../../from_scratch/custom-gpt-10m/docs/API_SERVER.md) for the
exact start command and request-field differences specific to that project (for instance,
`custom-gpt-10m`'s `trim_at_role_markers` field, needed because that project serves a
**base** model with no chat template, unlike a model post-trained to stop at role
boundaries on its own).

## Deep-Dive: Why "Load Once, Serve Many" Is the Whole Ballgame at This Scale

The single biggest performance difference between "a script that generates text" and "a
server that serves generation requests" isn't anything about HTTP — it's amortizing model
load cost across many requests instead of paying it once per request. A checkpoint load
plus architecture reconstruction is a fixed cost measured in the same units regardless of
how many tokens get generated afterward; a synchronous FastAPI process that loads once at
startup pays that cost exactly once, no matter how many `/generate` calls follow. Get this
one thing right and a naive, one-request-at-a-time server is already most of the way to
"correctly engineered for its scale" — the remaining gap to production-serving-engine
territory is specifically about concurrency and memory efficiency under load, not about
this fundamental amortization.

## Try It Yourself

- Start one of this repo's `api_server.py` instances and call `/health` before and after
  changing which checkpoint file it points at (via the environment variable each project
  uses) — confirm the reported `checkpoint_step`/`params` change accordingly, and consider
  what would go undetected if this endpoint didn't exist.
- Compare `inference.py`'s command-line generation function call to `api_server.py`'s
  `/generate` route side by side — identify exactly which lines differ (request
  parsing/response formatting) versus which are identical (the actual call into
  `generate()`).

## Common Misconceptions

- **"A FastAPI wrapper around a generation script isn't 'real' serving."** It is real
  serving — an HTTP client can call it, get a response, and use it in a real application.
  It's serving scoped correctly for its request volume and model size, not a placeholder
  for something else.
- **"Production serving concerns (batching, KV cache budgeting, multi-GPU) are always
  necessary."** They become necessary once request volume or model size actually demand
  them — building that machinery before you have the problem it solves is premature
  complexity, not correctness.
- **"The server needs its own copy of the generation logic, tuned for serving."** It
  shouldn't — the whole point of a thin wrapper is that generation behaves identically
  whether it's called from a CLI script or an HTTP route, because it's the *same function*
  either way.

## Practice Questions

1. Why does loading the model at server startup rather than per-request matter more as
   request volume grows, even though it doesn't change what a single request costs?
2. What operational question does a `/health` endpoint that reports `checkpoint_step`
   answer that a plain "is the process running" check cannot?
3. Name two conditions under which a synchronous, one-request-at-a-time server like this
   repo's stops being the right design, and what each condition's fix is generally called.

## Key Terms

- **Serving wrapper**: the thin network-facing layer (request parsing, response
  formatting, routing) around an unchanged generation function.
- **Health check**: an endpoint reporting which model/checkpoint is actually loaded and
  running, distinct from basic process-alive liveness.
- **Continuous batching**: interleaving multiple in-flight generation requests so compute
  isn't idle between individual requests' token-by-token steps — the concurrency fix once
  synchronous one-at-a-time serving stops being enough.
