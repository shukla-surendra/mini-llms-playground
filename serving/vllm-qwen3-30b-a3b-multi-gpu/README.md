# Qwen3-30B-A3B — 2-GPU (or multi-node) vLLM serving

Reminder note only, 2026-08-31 — nothing implemented yet, full build planned
for a future session. Matches this repo's `serving/vllm-tinyllama-1.1b/`
conventions (uv-managed, `pyproject.toml` + `Makefile`) when actually built.

## Why this needs 2+ GPUs at all — the MoE memory math

Qwen3-30B-A3B is a Mixture-of-Experts model: 30B total parameters, ~3B
**active** per token. The "A3B" name is about compute per token, not memory —
every expert's weights must still be resident, since the router's per-token
choice varies and the model can't know in advance which experts it won't need.
So the real memory number is the full 30B, not 3B: `30B × 2 bytes = 60 GB` in
bf16, before KV cache — beyond what a single commonly-available GPU holds with
real headroom. This lab targets **tensor parallelism across 2 GPUs**
(`vllm serve ... --tensor-parallel-size 2`) as the primary path, plus a
genuine multi-node (Ray-coordinated) option for when 2 GPUs don't live in one
box.

## To build, when picked back up

- `serve.py` — single-node 2-GPU launch, `torch.cuda.device_count()` check,
  clear error if fewer than 2 GPUs are visible rather than silently falling
  back to 1.
- `serve_multi_node.sh` — the real 2-step sequence for multi-node: Ray head on
  node 0, Ray worker join on node 1, one `vllm serve` launch against the
  combined cluster.
- `pyproject.toml` / `Makefile` — same shape as the `vllm-tinyllama-1.1b`
  sibling; `vllm` as the only required dependency (no CPU/MPS fallback path
  planned — not realistic at usable speed for a 30B MoE model).
- Quantized (AWQ/int4) single-GPU variant is a real fallback worth considering
  if 2 real GPUs turn out hard to get, not scoped yet.
- Not run on real hardware yet.
