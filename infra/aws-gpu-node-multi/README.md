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
Spot rate, kept off by default here — see the multi-node Spot caveat above). If
you've fallen back to `g5.xlarge` (see the deployment log below), it's ≈**$2.01/hr
combined** ($1.006/node) instead. Same-AZ private-IP traffic between the two nodes
costs **$0.00** in data transfer — this module enforces that by construction (one
subnet, both instances), not just by convention.

## Between runs

```bash
make down   # destroy both instances, keep bucket/IAM/SG — bill drops to ~$0
make up     # recreate both when you next need the pair
```

## Real deployment log (2026-08-31, us-east-1)

Kept here because every step below was a genuine blocker on the first real apply,
not a hypothetical — worth reading before your own first run in a region/instance
family you haven't deployed G/VT instances in before.

1. **G/VT quota was 0 going in.** `aws_instance` creation needs an EC2 Service
   Quota increase for "Running On-Demand G and VT instances" (`L-DB2E81BA`) *before*
   `apply` — g5, g6, and the other G-family types all share this one quota code, so
   raising it once covers whichever of them you end up using. Request it via the
   Service Quotas console; approval took under an hour. `instance_count = 0` in
   `terraform.tfvars` lets you `apply` everything *except* the instances (bucket,
   IAM, SG, key pair) while waiting, so the quota wait doesn't block the rest of the
   setup.
2. **`apply` looked hung — it wasn't, at first: it was quota-blocked.** With the
   quota not yet approved, `RunInstances` sat retrying with no output for 10+
   minutes (Terraform's own progress line — `Still creating...` — only refreshes on
   its own cadence, so a long silent stretch alone isn't proof of a hang). Killing
   it (`kill <pid>`, safe — no instances existed yet) surfaced the real error only
   once interrupted: `context canceled`, not the underlying cause.
3. **The fast way to see the actual AWS-side error, instead of waiting on
   Terraform's retry/backoff to give up:**
   ```bash
   aws cloudtrail lookup-events --region us-east-1 \
     --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
     --max-results 1 --query 'Events[0].CloudTrailEvent' --output text \
     | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('errorCode'), d.get('errorMessage'))"
   ```
   This is what actually revealed the real blocker each time below — CloudTrail logs
   the true `errorCode`/`errorMessage` per attempt even while Terraform is still
   silently retrying.
4. **After the quota was approved, retried and hit
   `Server.InsufficientInstanceCapacity` for `g6.xlarge` — twice, in two different
   AZs** (`us-east-1f`, then `us-east-1a`, both auto-selected by this module's own
   AZ-offering logic in `network.tf`). AWS's own error message rotates through the
   *other* AZs as "try these instead" each time, which is not a reliable signal —
   it doesn't mean those AZs actually have capacity, just that they're not the one
   that just failed. G6 (L4) is new-generation hardware and was genuinely
   capacity-constrained region-wide that day, not a fluke of one AZ.
5. **Fix: fell back to `g5.xlarge` (A10G, previous generation).** Same "G and VT"
   quota family (no second quota request needed), same bf16/TF32 capability this
   project's trainer needs, offered in every AZ in the region, and — on this
   occasion — actually had capacity. `subnet_id` was pinned explicitly in
   `terraform.tfvars` (rather than left to auto-selection) once a working AZ was
   found, so a re-`apply` doesn't re-roll the dice on which AZ it lands in.
6. **Even within the AZ/type combo that ultimately worked, capacity was
   inconsistent per-node, not a clean pass/fail:** the worker landed in 2m15s;
   the master, launched in the same `apply`, took over 11 minutes before AWS
   handed it an instance. Both eventually succeeded without any config change in
   between — real evidence that on-demand GPU capacity fluctuates from one moment
   to the next even for the exact same instance type, AZ, and account. Don't read
   one slow node as a sign the whole approach is wrong if the other node just
   landed fine.

**Takeaway for next time:** request the quota increase (or confirm it's already
≥ 2× the instance's vCPU count) before ever running `apply` with real instance
counts; if `RunInstances` goes quiet for more than a couple minutes, check
CloudTrail directly rather than waiting on Terraform's own retry loop; and treat a
capacity error as a signal to try a different **instance type** in the same family
before spending many attempts hopping AZs one at a time.

### Two more real bugs, hit only once actual training started

Both instances being `running` and bootstrap completing cleanly is not the same as
training actually working — two more real failures showed up only once
`bash ~/launch_ddp.sh` was run for real, on real CUDA hardware, neither of them
caught by the CPU/gloo smoke test:

7. **`torchrun: command not found`, then `uv: command not found`.** The bootstrap
   script installs `uv` to `~/.local/bin/uv` and runs `uv sync` (which creates the
   project's own `.venv/`), but neither `~/.local/bin` nor `.venv/bin` end up on
   `PATH` in a *later* SSH/tmux shell — the bootstrap script's own `export PATH=...`
   line only ever applied within that one script's execution, never persisted to
   `.bashrc`/`.profile`. Fixed in `templates/bootstrap.sh.tftpl`: the generated
   `launch_ddp.sh` now invokes the absolute path `"$HOME/.local/bin/uv" run
   torchrun ...` instead of bare `torchrun` or bare `uv`.
8. **Real CUDA OOM on the actual training preset**, not the `tiny` preset the smoke
   tests use: `batch_size=16` at `context_length=2048` for the 347M-parameter model
   used **21.92 GiB of the A10G's 22.06 GiB usable** before failing to allocate one
   more activation buffer. The CPU/gloo smoke test (`GPT_PRESET=tiny`) never
   exercises this because `tiny`'s tensors are trivially small — memory pressure at
   the real preset's size is not something a mechanism smoke test can catch, only a
   real run on real hardware can. Fixed by halving `GPT_BATCH_SIZE` to `8` and
   doubling `GPT_GRAD_ACCUM` to `128` on both nodes (preserves the exact same
   2048-seq effective batch and the same number of true optimizer updates — see
   `config.py`'s `resolve_train_config`, which derives `steps` from `batch_size`
   alone, not `grad_accum_steps`, so this pair of changes is the correct one to make
   together, not either alone).

**Takeaway for next time:** a passing CPU/gloo smoke test proves the DDP
*mechanism* (rendezvous, gradient sync, checkpoint round-trip) works — it says
nothing about whether the real preset's memory footprint fits the real GPU. Before
committing a multi-hour run's budget, watch `nvidia-smi --query-gpu=memory.used
--format=csv` (or just the first real step's own memory in the OOM error, if it
happens) on the *actual* instance type at the *actual* preset size, not just the
smoke test.

### Finding the actual working `batch_size`: do it on ONE GPU, not two

Once `batch_size=16` OOM'd, the instinct was to keep retrying the full 2-node job
after each change — expensive (2 billed GPUs per attempt) and slow (each retry pays
the ~2-node rendezvous/NCCL-timeout overhead of a dead peer before you even see the
next result). The actual memory-fit question is **entirely local to one rank** —
each node runs the identical model/batch on its own GPU independently; nothing
about it needs the other node to reproduce or to fix. The much cheaper loop that
actually found the right number:

1. `aws ec2 stop-instances` on the second node (billing pauses; the DDP mechanism
   itself was already proven separately by the CPU/gloo smoke test, so nothing is
   lost by not running 2-node during this search).
2. On the remaining node, run the trainer directly — no `torchrun`, no DDP wrapping
   at all (`world_size` defaults to 1): `GPT_BATCH_SIZE=<n> GPT_STEPS=<small>
   "$HOME/.local/bin/uv" run python -m gpt.cli.train`, overriding `GPT_STEPS` down
   to a couple dozen so each attempt costs seconds of GPU time, not minutes.
3. Delete `checkpoints/<preset>/` between attempts — the trainer auto-resumes from
   its own checkpoint, so a stale one silently makes the *next* attempt look like
   it "succeeded" while actually running zero new steps.
4. What this surfaced, concretely: `batch_size=16` fails in the forward pass
   (~21.9 GiB used); `batch_size=8` fails even with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set (~22.0 GiB used, still
   the forward pass — genuine exhaustion, not fragmentation, since "reserved but
   unallocated" was small both times); `batch_size=4` completed a full
   train/eval/checkpoint/demo-generation cycle cleanly, with real headroom
   (~21.6 GiB peak once the 2-node run confirmed it). Total cost to find this: a
   few single-GPU runs of a few seconds to ~90 seconds each, on the node that was
   going to be running anyway.

**Takeaway for next time:** if a problem only involves what happens *inside* one
rank (memory fit, a forward-pass bug, a dtype/precision question), reproduce it
on one GPU. Save the 2-node setup for questions that are actually about the two
nodes talking to each other.
