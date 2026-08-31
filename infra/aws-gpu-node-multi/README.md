# `aws-gpu-node-multi` — two real EC2 nodes for multi-node DDP

Terraform for **two separate `g6.xlarge` EC2 instances**, same Availability Zone,
wired for genuine multi-node `torchrun` DDP training — as distinct from
[`aws-gpu-node`](../aws-gpu-node/README.md) (one box, one GPU) and
[`gcp-gpu-node-multi`](../gcp-gpu-node-multi/README.md) (one box, **2 GPUs bundled
into a single `g2-standard-24` machine type** — single-node multi-GPU, not
multi-node, despite the name). This module is the one that actually launches two
independent machines that talk to each other over the network.

Prerequisites (Terraform/OpenTofu, AWS CLI, IAM permissions) are identical to
[`aws-gpu-node`](../aws-gpu-node/README.md#prerequisites-one-time-on-the-mac) — read
that section if this is your first time applying anything in `infra/`.

## What it creates, and what's different from the single-node sibling

| Resource | Why it's shaped this way |
|---|---|
| Two `aws_instance` (`gpu_master`, `gpu_worker`) | Separate resources, not one `count = 2` resource — avoids a self-reference cycle (see below) and lets each carry distinct rank/role tags |
| Master gets a **static private IP** (`cidrhost()` on the resolved subnet) | So the worker's bootstrap script can be given `--master_addr` without depending on the master's own not-yet-created computed attribute — a real Terraform constraint, not a style choice |
| One shared security group, **same subnet for both** | Same-AZ, private-IP traffic between EC2 instances is free *and* lower-latency than cross-AZ — directly relevant to DDP gradient sync, not just cost. See `docs/GPU_TRAINING.md`'s "Multi-Node DDP on a Single AZ" section in the DDP project itself for the full reasoning and the cost math |
| Self-referencing SG rules for `dist_port` + the ephemeral TCP range | `torchrun`'s rendezvous uses one port; NCCL negotiates additional ephemeral ports for the actual gradient traffic afterward — opening a range between the two nodes (not the internet) is the standard real-world pattern for a cluster this small |
| Generated `~/launch_ddp.sh` on **each** node, pre-filled with that node's `--node_rank` and the shared `--master_addr` | The actual multi-hour job is still started by a human on both boxes (`bash ~/launch_ddp.sh`), not auto-launched at boot — a long paid run should start only after a human confirms both nodes are up and the corpus/tokenizer landed correctly |
| `idle_shutdown_minutes` defaults to **0** (disabled) | Unlike the single-node module's default-on watchdog: independent per-node idle detection is genuinely risky here — one node's GPU can look idle while it's legitimately blocked on a gradient all-reduce, and stopping it mid-collective hangs the other node too |
| `use_spot` carries an explicit multi-node caveat | A reclaim of *either* node kills the whole synchronized job, not just that instance — see `variables.tf`'s description before enabling it for a real run |
| Checkpoint sync armed **only on the master** | Only rank 0 writes checkpoints during training (`trainer.py`'s `is_main` gating) — the worker has nothing to sync |
| Corpus **and tokenizer** both synced to **both** nodes | This project's embedding table is sized to its own 32,768-token vocabulary; a `.bin` without the matching `tokenizer.json` on *both* boxes trains on ids that silently index the wrong embedding rows |

## Quickstart

```bash
cp terraform.tfvars.example terraform.tfvars   # edit as needed
make init

# From from_scratch/custom-gpt-350m-ddp (already has data/ + tokenizer/ copied
# from the sibling custom-gpt-350m project):
cd ../../from_scratch/custom-gpt-350m-ddp
make -C ../../infra/aws-gpu-node-multi upload-corpus
make -C ../../infra/aws-gpu-node-multi upload-tokenizer

cd ../../infra/aws-gpu-node-multi
make apply          # BILLING STARTS ON BOTH NODES

make bootstrap-log-master   # confirm corpus/tokenizer landed, no WARNs
make bootstrap-log-worker

make gpu-master              # confirm bf16 == True on both
make gpu-worker

make launch-reminder         # prints the exact 2-terminal start sequence
```

Then, on **both** nodes (separate SSH sessions — `make ssh-master`, `make ssh-worker`):

```bash
tmux new -s train
bash ~/launch_ddp.sh
```

Both commands must be started at roughly the same time — DDP's process-group
rendezvous waits for every rank, so the first one launched simply waits for the
second.

## Before spending real money on this

Verify the DDP mechanism for near-$0 first:
[`from_scratch/custom-gpt-350m-ddp/scripts/ddp_smoke_test.py`](../../from_scratch/custom-gpt-350m-ddp/scripts/ddp_smoke_test.py)
— a real 2-rank CPU/gloo run, already confirmed working (loss decreasing, clean
checkpoint save/load). Then, once both nodes are up, run a short real 2-node smoke
test (a few hundred steps, a small `GPT_TARGET_TOKENS` override) before trusting
`target_tokens`/`grad_accum_steps` for the full multi-hour budget — measured
`tokens/sec` under real NCCL/network conditions is the only thing that actually
validates a wall-clock estimate on this hardware.

## Cost

Two `g6.xlarge` on-demand ≈ **$1.60/hr combined** (`make spot-price` for the current
Spot rate, kept off by default here — see the multi-node Spot caveat above).
Same-AZ private-IP traffic between the two nodes costs **$0.00** in data transfer —
this module enforces that by construction (one subnet, both instances), not just by
convention.

## Between runs

```bash
make down   # destroy both instances, keep bucket/IAM/SG — bill drops to ~$0
make up     # recreate both when you next need the pair
```
