# Distributed Training: DDP, FSDP, and the Parallelism Landscape

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2B — Training at Scale (see
[Chapter 25](25_efficient_attention_flash_and_sdpa.md)'s header for why this and that
chapter are appended after the original numbered catalog). Builds on
[Chapter 13](13_the_training_loop_mechanism_by_mechanism.md)'s training loop — this
chapter is about running that same loop across *multiple devices at once*, and the
different strategies for doing so.

## In Plain English

A model and its training data can outgrow what a single GPU (or machine) can hold or
compute in reasonable time. Distributed training splits the work across multiple
devices — but "splits the work" can mean genuinely different things: give every device a
full copy of the model and split the *data* (data parallelism), or split the *model itself*
across devices so no single device ever needs the whole thing in memory (model/sharded
parallelism). DDP is the classic instance of the first; FSDP is a specific, popular
instance of the second.

## The First-Principles Explanation

### Data parallelism (DDP): same model everywhere, different data everywhere

Every device (rank) holds a complete replica of the model. Each step, every rank processes
a different mini-batch, computes its own gradients locally, and then all ranks
**all-reduce** (average) their gradients before applying the optimizer step — so every
replica ends up applying the identical update and stays in sync, without ever directly
exchanging model weights after the initial broadcast. The cost: memory-wise, this is the
*most* expensive strategy per device, since every single device needs enough memory to
hold the full model, its full gradients, and its full optimizer state — DDP doesn't help
at all if the model itself is the thing that doesn't fit.

### Sharded parallelism (FSDP): no device ever holds the whole model

Fully Sharded Data Parallel takes a different approach: parameters, gradients, and
optimizer state are all split (sharded) across ranks from the start — each rank
permanently holds only `1/world_size` of each. When a given layer's forward or backward
pass actually needs its full, unsharded parameters, FSDP performs an **all-gather**
(temporarily reassembling the full parameter tensor from every rank's shard), uses it, then
immediately discards everything except this rank's own shard again. This is strictly more
communication than DDP (an all-gather per layer, not one all-reduce per step) — the
trade FSDP makes is more network traffic in exchange for a peak per-device memory
footprint that doesn't scale with the model's full size, which is what makes training
models too large for any single device's memory possible at all.

### Why "it doesn't fit" is a memory-accounting problem, concretely

"Doesn't fit" is not about the weights alone. Mixed-precision AdamW training carries five
separate copies of every parameter, not one:

| What | Precision | Bytes/param |
|---|---:|---:|
| Weights (compute copy) | fp16/bf16 | 2 |
| Gradients | fp16/bf16 | 2 |
| Master weights | fp32 | 4 |
| Adam momentum (m) | fp32 | 4 |
| Adam variance (v) | fp32 | 4 |
| **Total static memory** | | **16 bytes/param** |

A 1B-parameter model needs **16GB** for this static state alone — before a single
activation tensor exists. An NVIDIA L4 has 24GB total, leaving roughly 8GB for
activations, which a real batch size/sequence length blows through fast. This is the
concrete trigger for needing more than one GPU: not "training would be faster," but "the
optimizer state alone doesn't fit."

DDP doesn't help here (see above — every rank still needs its own full 16GB). FSDP shards
that 16GB across `world_size` ranks: across 4 GPUs, each holds roughly 4GB of static
state instead of 16GB, leaving roughly 20GB per GPU for activations instead of 8GB. This
is the exact number that decides whether the model trains at all on this hardware, not a
speed optimization.

### Where these fit in the broader parallelism landscape

DDP and FSDP are both **data**-parallel strategies at their core (different ranks process
different data) — the distinction between them is *how much of the model* each rank
carries. Two other real strategies split the *model itself* differently, worth knowing by
name even without an implementation in this project:

- **Tensor parallelism**: split individual large operations (e.g., one big matrix
  multiply) across devices, each computing a slice of the same operation in parallel —
  requires fast interconnect (like NVLink) since it communicates *within* every forward
  pass, not just once per step.
- **Pipeline parallelism**: assign different *layers* to different devices (device 0 gets
  layers 1-10, device 1 gets layers 11-20, etc.), passing activations between them as data
  flows through the model — trades some device idle time (a "pipeline bubble" while later
  stages wait for earlier ones) for lower communication *volume* than tensor parallelism.

Real large-model training often combines several of these simultaneously (data + tensor +
pipeline parallelism together) — each addressing a different bottleneck (data throughput,
single-operation size, total model depth) rather than one strategy being strictly better
than another.

### Single-node vs. multi-node: it's a hierarchy, not a choice

"Multi-GPU" and "multi-node" sound like two points on the same scale, but in real
frontier-scale training they're not alternatives — single-node multi-GPU is the building
block, and multi-node is how that block gets scaled out. Which of the parallelism
strategies above runs *within* a node versus *across* nodes is decided by one thing:
how communication-hungry that strategy is, matched against how much bandwidth is actually
available at each layer of the hardware.

| Interconnect | Bandwidth (order of magnitude) | What runs here | Why |
|---|---|---|---|
| NVLink / NVSwitch (intra-node, e.g. 8 GPUs) | ~900 GB/s (H100-class) | Tensor parallelism | TP needs an all-reduce/all-gather *per layer*, every forward pass — only NVLink-class bandwidth keeps that cheap enough to not dominate step time. |
| InfiniBand / AWS EFA (inter-node, GPUDirect RDMA) | ~400 Gbps/GPU in rail-optimized clusters | Pipeline parallelism, DDP/FSDP gradient sync | Coarser-grained traffic (activations at stage boundaries, one gradient all-reduce per step) tolerates the higher latency and lower bandwidth of leaving the node. |
| Plain Ethernet/ENA (no RDMA) | Far lower, higher latency | Whatever's left, reluctantly | This is what a "not built for it" cloud GPU box gives you by default — see the AWS L4 case below. |

So a real large-scale training cluster's design reads top-down: 8 GPUs per node wired
with NVLink for tensor parallelism → nodes wired together with InfiniBand/EFA for
pipeline and data parallelism. Real examples: Microsoft/OpenAI's Azure ND H100 v5
clusters, Meta's Research SuperCluster (RSC), Anthropic's reported AWS Trainium2
clusters ("Project Rainier"), and Google's TPU pods — which sidestep the GPU
NVLink/InfiniBand split entirely by using a custom interconnect (ICI, plus optical
circuit switches) purpose-built for the same intra-pod/inter-pod distinction. At
10,000+ accelerator scale, hardware failure *during* a multi-week run stops being an
edge case, which is why fault-tolerant, fast-resume checkpointing (Chapter 27) is a
first-class requirement at this scale, not a nice-to-have.

**A concrete case where this bites**: AWS's G6 instance family (NVIDIA L4 GPUs) does
*not* support EFA — that's reserved for the P4d/P4de/P5/Trn1 families. So two `g6.xlarge`
nodes doing multi-node DDP communicate over plain networking, not RDMA. It still works —
NCCL falls back to TCP sockets — but for a small model, the gradient all-reduce's
communication cost relative to its (small) compute cost can eat most of the theoretical
2x speedup from the second GPU. This is the same shape of problem as the "why does FSDP
communicate more than DDP" trade-off above, just moved one layer out to the network.

### Training scales out; serving mostly doesn't

Training's parallelism choices above are throughput-oriented — a multi-week job can
absorb cross-node latency because it's amortized over millions of steps. Serving is
latency-oriented per request, which flips the usual answer: production inference is
overwhelmingly **single-node multi-GPU (or single-GPU) replicas, scaled horizontally**,
not multi-node per request.

- Model fits on one GPU → many independent single-GPU replicas behind a load balancer.
  Simplest, cheapest, zero cross-device coordination per request.
- Model doesn't fit on one GPU (e.g. a 70B+ dense model) → tensor-parallel across a
  *few* GPUs within one node, same NVLink-latency argument as training's TP. Still one
  node serves one request.
- Multi-node inference is the exception, reserved for models too large even for one
  node's GPUs — very large dense models, or MoE models needing cross-node expert
  parallelism (DeepSeek-V3's published architecture is a real, documented instance of
  this). Splitting one request's forward pass across nodes adds latency a training
  step doesn't care about, so it's avoided unless the model genuinely forces it.
- This repo's serving code (Chapter 21-22, and `serving/vllm-*` in the wider workspace)
  is the single-GPU/single-node case; the serving-engine ecosystem chapter (Chapter 23,
  planned) is where PagedAttention-style KV-cache management and continuous batching —
  the techniques that matter more than parallelism topology at this end — belong.

### Checkpoint files under sharding: why a sharded model isn't one file

Once a model needs FSDP/TP/PP to train or serve at all, the checkpoint on disk usually
stops being one file too — but for one of two genuinely different reasons, easy to
conflate. The dividing question to ask of any multi-file checkpoint: **does the split
depend on how many devices will load it, or not?**

**Size-based sharding (Hugging Face `safetensors`) — split depends on nothing but file
size.** A 7B-parameter model saved in bf16 is `7e9 * 2 bytes ≈ 14GB` of weights. Hugging
Face's default shard limit is 5GB/file, so `save_pretrained` writes it as three files
regardless of whether you later load it on 1 GPU or 8:

```
model-00001-of-00003.safetensors   (~5GB)
model-00002-of-00003.safetensors   (~5GB)
model-00003-of-00003.safetensors   (~4GB)
model.safetensors.index.json
```

The index file is the part that actually matters — it's a flat JSON map, not a rank
table:

```json
{
  "metadata": { "total_size": 14000000000 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
    "model.layers.15.mlp.down_proj.weight": "model-00002-of-00003.safetensors",
    "lm_head.weight": "model-00003-of-00003.safetensors"
  }
}
```

Loading code (`from_pretrained` under the hood) reads this index once, then for each
parameter it needs, opens whichever shard file the map points to and reads just that
tensor — `safetensors`' format supports this because it's memory-mappable with a
per-tensor byte offset, not a single opaque blob. Nothing here refers to a GPU rank, a
process count, or a parallelism strategy — you could split this exact model into 1 shard
or 30 shards and it would load identically. The split is a *storage/transfer*
convenience (so no single file exceeds Git LFS/Hub upload limits or blows out a
download's resumability), decoupled entirely from how the model will actually be run.

**Rank-specific checkpoints (Megatron-LM/DeepSpeed) — split depends on exactly how many
devices you trained on.** If a 7B model was trained with tensor-parallel degree 4, its
checkpoint directory looks like:

```
mp_rank_00_model_states.pt   # this rank's 1/4 shard of every TP-split weight
mp_rank_01_model_states.pt
mp_rank_02_model_states.pt
mp_rank_03_model_states.pt
```

Here the filename *is* the addressing scheme — at load time, each process reads its own
`RANK` (or `LOCAL_RANK`, depending on the framework) from the environment and opens
`f"mp_rank_{rank:02d}_model_states.pt"` directly, no index lookup, no parameter-name
mapping. There is no version of "load this on a different number of GPUs" without an
explicit conversion step first, because the tensors *inside* each file were split at
save time according to that exact TP degree (e.g., each `mp_rank_XX` file holds one
1/4-slice of every attention/MLP weight matrix, not whole tensors) — re-sharding to TP
degree 2 or 8 means literally re-slicing every matrix along its split dimension, which is
what dedicated conversion tooling exists to do (DeepSpeed's `zero_to_fp32.py` for
collapsing ZeRO-sharded optimizer state back to a single fp32 checkpoint; Megatron's
`tools/checkpoint/convert.py` for changing TP/PP degree). Skipping that step and just
pointing a different `world_size` at the old files produces silently wrong tensors, not
an error — the file sizes and shapes match the *new* rank's slot in the process group,
not the *old* parallelism configuration the weights were actually saved under.

This repo doesn't demonstrate either — `custom-gpt-153m`/`350m` write a single unsharded
`.pt` file from a single GPU (Chapter 27 covers that mechanism in full). Worth being
explicit about that gap rather than implying this project's checkpointing generalizes to
the sharded case directly.

## Grounded in This Repo's Code

[`from_scratch/custom-gpt-6m/src/gpt/training/trainer_ddp.py`](../../from_scratch/custom-gpt-6m/src/gpt/training/trainer_ddp.py)
and
[`train_fsdp.py`](../../from_scratch/custom-gpt-6m/src/gpt/training/trainer_fsdp.py)
implement both strategies on the exact same model and data this project's `train.py`
already trains — see
[`docs/DISTRIBUTED_TRAINING.md`](../../from_scratch/custom-gpt-6m/docs/DISTRIBUTED_TRAINING.md)
for two real, concrete pieces of evidence that the mechanisms are actually doing what they
claim: a genuine `NotImplementedError`/`AttributeError` this project hit trying to run
`torchrun` and (separately) plain `FSDP(model)` on this specific hardware, and a real
side-by-side run showing DDP's `num_parameters()` reporting the full model per rank
against FSDP's reporting exactly half.

## Deep-Dive: Why All-Reduce Keeps DDP's Replicas In Sync Without Ever Re-Syncing Weights Directly

A question worth sitting with: DDP broadcasts weights once, at construction — how do the
replicas not drift apart over hundreds of training steps if weights are never
re-synchronized directly again? Because **all-reduce averages gradients, not weights**,
every single step, before the optimizer step is applied. If every rank starts a step with
identical weights, and every rank then applies the *same* (averaged) gradient update via
the *same* optimizer, every rank necessarily ends the step with identical weights again —
by induction, this holds for every step of training, so the replicas can never drift as
long as the gradient averaging genuinely happens before every single optimizer step. This
is also exactly why DDP requires the gradient all-reduce to be synchronous (block until
every rank has contributed) rather than approximate or best-effort — any rank silently
missing the sync even once would permanently desynchronize that replica from the others.

## Try It Yourself

- Run `python train_ddp.py` and `python train_fsdp.py` in this project back to back (same
  `STEPS`/`BATCH_SIZE`), and compare each run's printed `[model] N parameters` line — a
  direct, concrete confirmation of full-replication vs. sharding, not just a claim.
- Read `docs/DISTRIBUTED_TRAINING.md`'s two "real bug" sections in full — both are genuine
  environment-specific issues (a DNS-resolution hang, an MPS-accelerator misdetection)
  encountered building this project, a real instance of the gap between "the API looks
  simple in a tutorial" and "getting it running on the hardware actually in front of you."

## Common Misconceptions

- **"FSDP is just a faster version of DDP."** It's not faster in general — it's a
  different memory/communication trade, usually *slower* per step (more communication)
  specifically in exchange for fitting models DDP couldn't hold at all. Choosing between
  them is about whether the model fits in per-device memory under DDP, not raw speed.
- **"More distributed processes always means faster training."** Only true when the
  hardware and interconnect can actually support the added communication cheaply (real
  GPUs with fast interconnect) — see this project's own numbers in
  `DISTRIBUTED_TRAINING.md`, where adding CPU processes over `gloo` was slightly *slower*
  than single-process training, for a model this small.
- **"Data parallelism and model parallelism are competing choices — pick one."** Real
  large-scale training frequently combines multiple strategies (data + tensor + pipeline)
  at once, each solving a different bottleneck simultaneously.
- **"Multi-GPU and multi-node are basically the same thing at different scales."** They're
  not interchangeable — which strategy runs intra-node (NVLink-fast, latency-sensitive:
  tensor parallelism) versus inter-node (InfiniBand/EFA, latency-tolerant: pipeline
  parallelism, DDP/FSDP gradient sync) is a deliberate hardware-matched decision, not just
  "more of the same, further apart."
- **"If it needs more than one GPU to train, it needs more than one GPU to serve."**
  Training and serving optimize for different things (throughput vs. per-request
  latency), so real serving deployments favor single-node (or single-GPU) replicas
  scaled horizontally wherever the model fits — multi-node serving is the exception,
  used only when a model doesn't fit even across one node's GPUs.
- **"A sharded checkpoint's multiple files mean one file per GPU."** Only true for
  rank-specific formats (Megatron/DeepSpeed). Hugging Face's sharded `safetensors`
  format splits by file size alone, with no relationship to how many devices load it.

## Practice Questions

1. Explain precisely why DDP's replicas stay synchronized for an entire training run
   despite weights only being broadcast once, at construction time.
2. A model's optimizer state alone is larger than a single GPU's memory. Would DDP alone
   solve this problem? Would FSDP? Explain the mechanism-level reason for each answer.
3. Why does FSDP typically involve more network communication than DDP, and under what
   circumstances would you accept that cost anyway?
4. A 1B-parameter model is trained with mixed-precision AdamW. Compute its static memory
   footprint (weights + gradients + master weights + optimizer state) in GB, and determine
   whether it fits on a single 24GB GPU before any activations are counted.
5. Explain why tensor parallelism is almost always kept within a node while pipeline
   parallelism and DDP/FSDP gradient sync are the strategies allowed to cross nodes.
6. Two `g6.xlarge` (1x L4 each, no EFA) instances run multi-node DDP on a 153M-parameter
   model. Would you expect close to a 2x speedup over one instance? Explain the mechanism
   that determines the answer, not just the conclusion.
7. A model is small enough to serve on a single GPU. Explain why a production deployment
   would still typically NOT use multi-node inference for it, even if multi-node hardware
   is available.
8. Given two checkpoint directories — one with `model-00001-of-00004.safetensors` +
   `model.safetensors.index.json`, another with `mp_rank_00_model_states.pt` through
   `mp_rank_03_model_states.pt` — explain what determines which file(s) a given GPU needs
   to read in each case, and why resuming the second one on a different GPU count is
   harder than resuming the first.

## Key Terms

- **All-reduce**: a collective operation combining (e.g., averaging) a tensor across every
  rank in a process group, with every rank ending up with the same combined result — the
  mechanism DDP uses to keep gradients (and therefore weights) synchronized.
- **All-gather**: a collective operation where every rank contributes its own shard and
  every rank receives the full, reassembled tensor — the mechanism FSDP uses to
  temporarily reconstruct full parameters before a layer needs them.
- **Sharding**: splitting a tensor (parameters, gradients, or optimizer state) into pieces
  distributed across ranks, so no single rank permanently holds the whole thing.
- **Tensor / pipeline parallelism**: model-parallel strategies splitting, respectively, a
  single large operation or entire layers across devices — distinct from FSDP's
  parameter-sharding approach, though real systems often combine several strategies.
- **Static memory (training)**: the fixed per-parameter memory used by weights,
  gradients, and optimizer state (16 bytes/param under mixed-precision AdamW) —
  independent of batch size, as opposed to activation memory, which scales with batch
  size and sequence length.
- **NVLink / NVSwitch**: NVIDIA's high-bandwidth (~900 GB/s on H100-class hardware)
  intra-node GPU interconnect — what makes per-layer tensor-parallel communication
  cheap enough to be worth doing.
- **InfiniBand / EFA (Elastic Fabric Adapter)**: inter-node interconnects supporting
  GPUDirect RDMA, used for cross-node pipeline-parallel and DDP/FSDP traffic. Not every
  cloud GPU instance family has this — AWS's G6 (L4) does not; P4d/P4de/P5/Trn1 do.
- **Rail-optimized topology**: a cluster network design where each GPU's inter-node
  link is wired through a dedicated switch "rail," minimizing contention when many
  nodes communicate simultaneously — part of what makes InfiniBand clusters scale
  further than commodity Ethernet at the same nominal bandwidth.
- **Sharded checkpoint (size-based)**: a checkpoint split into multiple files purely by
  file-size limit (e.g. Hugging Face's `safetensors` shards + index JSON) — the split
  has no relationship to GPU count or parallelism strategy.
- **Rank-specific checkpoint**: a checkpoint where each file is literally one
  rank/device's shard of parameters or optimizer state (Megatron-LM/DeepSpeed style) —
  the file-to-rank mapping is load-bearing, and changing `world_size` requires an
  explicit re-shard/merge step.
