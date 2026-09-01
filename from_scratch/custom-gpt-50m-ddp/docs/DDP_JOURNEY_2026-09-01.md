# Learning DDP hands-on — a real 2-node run, 2026-09-01

Goal for this session was explicitly **not** "finish pretraining the 50m model" —
it was to see real multi-node `DistributedDataParallel` work end-to-end on real
hardware, understand the mechanism, and stop once that was demonstrated. This is
the log of what actually happened, including a real bug found and fixed along
the way. For the general, architecture-agnostic mechanism explanation (what an
all-reduce is, why `no_sync()` matters), see the sibling project's
[`custom-gpt-350m-ddp/docs/DISTRIBUTED_TRAINING.md`](../../custom-gpt-350m-ddp/docs/DISTRIBUTED_TRAINING.md) —
this doc is the run-specific facts and the "why" behind each decision made live.

## Timeline

| Time (IST) | Event |
|---|---|
| 06:19 | Corpus tokenized: 996,638,534 train tokens (from a 30GB pretrain corpus truncated to ~4.7GB — the model only needs Chinchilla-optimal ~1.03B tokens) |
| 06:24 | `50m-ddp.tfvars`'s `target_tokens` placeholder updated to the real 996,638,534 |
| 06:24 | Bucket/IAM/security-group created (`terraform apply -var instance_count=0`) — zero GPU billing yet |
| 06:28 | Corpus (`train.bin`/`test.bin`, ~2GB) uploaded to S3 |
| 06:35 | First launch attempt: **L4 (`g6.xlarge`)**, tried first for cost (~$0.80/hr vs A10G's ~$1.01/hr) |
| 06:38–06:46 | L4 launch failed repeatedly — `Server.InsufficientInstanceCapacity`, 5+ failed `RunInstances` calls over ~11 minutes, zero instances created. Same failure signature the `custom-gpt-350m-ddp` sibling hit on this exact subnet (`us-east-1c`) on 2026-08-31 — this looks like an ongoing regional G6 capacity crunch, not a one-off |
| 06:46 | Interrupted the stuck apply cleanly (`terraform state list` confirmed zero orphaned resources, no billing incurred), switched `instance_type` to `g5.xlarge` (A10G) |
| 06:47 | A10G launch succeeded in **17 seconds** — both `aws_instance` resources created immediately, no capacity issue this time |
| ~07:00 | **Bug found**: `launch_ddp.sh` was 0 bytes on the master node. Root cause: `templates/bootstrap.sh.tftpl`'s `<<LAUNCH` heredoc is unquoted, so `$HOME` inside it gets expanded by the *outer root cloud-init shell* (where `HOME` is unset) instead of deferred to the ubuntu user's runtime shell. That triggers `set -u`'s "unbound variable" error mid-heredoc; since `set -e` isn't active in that script, the `cat` command just fails silently and the script continues, leaving an empty file. Every other `$HOME`/`$PATH` reference in the same template file correctly escapes it (`\$HOME`) — this one line didn't. **Fixed in the template** (`\$HOME` now, so future deploys of this shared module — used by both `custom-gpt-50m-ddp` and `custom-gpt-350m-ddp` — don't hit this) and hand-wrote the correct `launch_ddp.sh` on both live nodes to unblock immediately |
| 07:03 | Batch-size verification: 40-step single-GPU smoke test on master alone (`GPT_STEPS=40`, `batch_size=16`, `grad_accum_steps=16`, no `torchrun`). Completed cleanly in 37s, peak GPU memory ~13GB of 23GB (A10G) — comfortable headroom, no OOM, including both eval passes (the memory-heaviest points, per the 350m sibling's OOM lesson). **No config change needed** — unlike the 350m run, which had to drop from `batch_size=16` to `4` after two real OOMs |
| 07:05 | Smoke-test artifacts (`best.pt`, `latest.pt`, `serving.pt`, eval history CSV) deleted from master before the real run — `latest.pt` specifically is what `gpt-train` auto-resumes from with no flag, so leaving it would have made the real run silently continue from the smoke test's 40 steps instead of starting fresh |
| 07:07 | Real 2-node run launched — `bash ~/launch_ddp.sh` in a detached `tmux` session on both nodes at once |
| 07:08 | Confirmed: `world_size=2`, Budget = 30,415 steps × 32,768 tokens/step = **1.00B tokens, 19.4 tokens/param, 1.00 epoch** — lands almost exactly on Chinchilla-optimal for this 51.48M-param model |
| ~07:15 | Verified the worker (rank 1) was genuinely computing, not stalled — its own log is intentionally silent (only rank 0/`is_main` prints the progress bar, runs eval, writes checkpoints), but `nvidia-smi` showed 100% GPU utilization and matching ~13.5GB memory use, confirming real work |
| ~07:32 | Learning goal achieved — real DDP mechanism observed working (rendezvous, lockstep step counting, gradient sync, loss dropping 10.99 → 7.19 by step ~1,211). **Stopped here deliberately**, not at a natural training milestone — checkpoints (`best.pt`/`latest.pt`/`serving.pt`, ~589MB each, step ~1,211) downloaded to local disk, then training processes killed and all AWS infrastructure destroyed (`terraform destroy`) |

## What this run actually proved (the point of the exercise)

- **Rendezvous works**: both nodes found each other via `--master_addr`/`--master_port` (master's private IP, port 29500) and `torchrun` computed `RANK`/`WORLD_SIZE`/`LOCAL_RANK` correctly on each side without this project's own code needing to know node count in advance.
- **Lockstep step counting**: there is one shared step counter for the whole job, not two independent per-node counters — both ranks advance step-for-step together because DDP's gradient all-reduce (every `grad_accum_steps` micro-steps) is a synchronization barrier neither side can pass alone.
- **Same model, different data, one checkpoint**: both ranks hold byte-identical weights throughout (broadcast once at start, then kept in sync by applying the same averaged gradient every update), but each rank draws a *different* random window of the *same* shared corpus every step — this is the actual mechanism of the speedup (2x corpus coverage per unit time), not redundant duplicate work. Only rank 0 writes checkpoints, since both copies are always equivalent.
- **Real, measured ~2x throughput**: `world_size=1` (smoke test) processed 16,384 tokens/step; `world_size=2` (real run) processed 32,768 tokens/step — a clean doubling, made possible for this model because `grad_accum_steps=16` keeps the (fixed, per-collective) network sync cost infrequent relative to compute. This project's own `docs/MULTI_NODE_DDP.md` is explicit that the 51.48M-param model sits closer to the point of diminishing returns than the 350m sibling does — the sync overhead is a larger fraction of a small model's per-step time, so going past 2 nodes for this model size specifically wouldn't scale as cleanly.
- **Cost tradeoff, stated plainly**: 2 nodes cost ~2x/hour (~$2.01/hr combined vs ~$1.01/hr for one A10G) to finish in roughly half the wall-clock time — the benefit is calendar time, not total dollars spent, which end up roughly the same either way.

## Final state after this session

- **AWS**: fully torn down (`terraform destroy` — both EC2 instances, the S3 bucket, IAM role, security group, key pair all removed). Zero ongoing billing.
- **Local**: `checkpoints/50m/{best,latest,serving}.pt` (step ~1,211/30,415, ~0.13 hours of actual training, test loss 7.90/test ppl 2,705 at last eval) saved to this project's `checkpoints/50m/` directory — a real, if very early, checkpoint from a genuine 2-node DDP run, not a synthetic one.
- **Not resumable as-is**: since the bucket (which held the corpus at `s3://.../50m-ddp/corpus/`) was destroyed along with everything else, a future resume would need `make upload-corpus` re-run after a fresh `terraform apply` — the local `data/train.bin`/`test.bin` this was built from are untouched, so that's a re-upload, not a re-tokenize.
- **Bug fix carried forward**: `infra/aws-gpu-node-multi/templates/bootstrap.sh.tftpl`'s `$HOME` escaping fix is in the shared module, so it benefits any future deploy of either DDP project, not just this run.
