# Resuming an existing run on the GPU node

Use this when a run already has real progress on the Mac (MPS/CPU) — checkpoints,
hours of wall clock, a step count worth keeping — and you want to continue it on a
rented GPU rather than start over. `gpt-train` auto-resumes from
`checkpoints/<label>/latest.pt` whenever that file exists: no flag, no code change.
This doc is the checklist for getting that file (and the corpus behind it) onto the
box *before* it boots, so the very first launch is already a resume.

> Mechanism-level detail — why checkpoints are device-portable via `map_location`,
> what `is_compatible`/`remap_attn_impl` check — lives in each project's own
> `docs/MIGRATION.md` and
> [Chapter 27](../../docs/llm-engineering/27_checkpointing_and_resuming_training.md).
> This page is specifically about the Terraform module in this directory.

Worked example throughout: `custom-gpt-50m`, resuming from step 311,199 of a
1,000,000-step budget. Substitute your own project/label where marked.

## One-time prerequisites

- [ ] **An SSH key exists locally.** `ls ~/.ssh/id_ed25519.pub` — if missing:
      ```bash
      ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "mini-llm-gpu"
      ```
      Only the public half is ever uploaded (as the EC2 key pair) — the private key
      never leaves this Mac. This is why `public_key_path` in `variables.tf` reads a
      *local* file rather than letting AWS mint a `.pem`: if this file doesn't exist,
      `terraform plan`/`apply` fails with `Invalid value for "path" parameter`.

- [ ] **The corpus `.bin` has a `.bin.json` sidecar.**
      ```bash
      cd from_scratch/<your-project>
      ls data/*.bin.json 2>/dev/null || uv run gpt-tokenize --force
      ```
      A `.bin` built before this project gained sidecar support has none. Without it,
      `load_token_array`'s tokenizer-fingerprint guard can't run, and a mismatch
      trains silently on the wrong vocabulary instead of failing loudly. Rebuilding
      is deterministic — same `train.txt` in, byte-identical `.bin` out — so it's
      free and doesn't invalidate the checkpoint's sampling.

- [ ] **`terraform.tfvars` points at the right project and prefixes:**
      ```hcl
      project_subdir    = "from_scratch/custom-gpt-50m"
      corpus_prefix     = "50m/corpus/"
      checkpoint_prefix = "50m/checkpoints/"
      ```
      `checkpoint_prefix` is what makes this a *resume* rather than a fresh run — it's
      synced into `<project_subdir>/checkpoints` at boot, mirroring how
      `corpus_prefix` already works for `data/`. Leave it unset for a fresh run with
      no checkpoint to resume from.

## The launch sequence

The instance and the bucket are created in the same `terraform apply` by default —
which means a plain `make apply`/`make up` would boot the box *before* you've had a
chance to upload anything into the bucket it's about to pull from. Split it in two:

```bash
cd infra/aws-gpu-node

# 1. Bucket, IAM role, security group, key pair — no instance yet, no compute billing
make down
```

```bash
# 2. Upload the corpus and the existing run. PROJECT_DIR only needed if it
#    doesn't match what's already in terraform.tfvars.
make upload-corpus     PROJECT_DIR=../../from_scratch/custom-gpt-50m
make upload-checkpoint PROJECT_DIR=../../from_scratch/custom-gpt-50m
```

`upload-checkpoint` ships the whole `checkpoints/<label>/` tree (`latest.pt` at
minimum; `best.pt` and `serving.pt` too, if present) to
`s3://<bucket>/<checkpoint_prefix>`. For the 50m run that's ~1.85 GB across three
617 MB files — a couple of minutes at home upload speed, once, ever, since every
subsequent instance pulls from S3 at in-region speed instead.

```bash
# 3. Launch — billing starts here
make up
make bootstrap-log     # watch it pull uv, the repo, the corpus, AND the checkpoint
```

Confirm the checkpoint actually landed before starting training — `bootstrap-log`
prints every `.pt` file the sync finds:

```
+ find /home/ubuntu/tiny_llm/from_scratch/custom-gpt-50m/checkpoints -name '*.pt' -exec ls -la {} \;
-rw-r--r-- 1 ubuntu ubuntu 617829195 ... checkpoints/50m/latest.pt
```

No output there means the sync found nothing — check `checkpoint_prefix` in
`terraform.tfvars` matches what `upload-checkpoint` actually uploaded to.

```bash
make gpu                # expect: NVIDIA L4 True — wrong instance type otherwise
```

## Resuming, on the box

**Do not run `gpt-train` directly in the SSH session you're typing in.** Closing the
terminal, the SSH connection dropping, your laptop sleeping — any of these end the
SSH session, and the shell sends `SIGHUP` to every foreground process it started.
`gpt-train` is one of them. It dies with the connection, mid-step, whether you meant
to stop it or not. This isn't specific to this instance or this project — it's how
every SSH-attached foreground process behaves everywhere.

`tmux` fixes this by running the shell (and everything in it) on the **instance**,
detached from any particular SSH connection. The training process becomes a child of
the `tmux` server, not of your SSH session — so when the SSH session ends, there's
nothing attached to it to hang up.

```bash
make ssh
cd ~/tiny_llm/from_scratch/custom-gpt-50m
tmux new -s train
```

Everything from here runs *inside* that `tmux` session.

Set `GPT_BATCH_SIZE`/`GPT_GRAD_ACCUM` to match the *effective* batch the checkpoint
was trained under, not just whatever the GPU can technically hold. Check what that
was first:

```bash
python3 - <<'PY'
import torch
ckpt = torch.load("checkpoints/50m/latest.pt", map_location="cpu")
print("step:", ckpt.get("step"))
print("batch_size:", ckpt.get("batch_size"), "grad_accum_steps:", ckpt.get("grad_accum_steps"))
PY
```

For the 50m checkpoint this printed `batch_size=1, grad_accum_steps=32` — effective
batch 32, i.e. 32,768 tokens per optimizer update at `context_length=1024`. The safe
first move is to keep that effective batch **identical** and only change *how* it's
assembled — trade serial accumulation for real GPU parallelism, not a bigger batch:

```bash
GPT_BATCH_SIZE=16 GPT_GRAD_ACCUM=2 uv run gpt-train
```

`16 x 2 = 32` — same effective batch, same token budget, same LR-schedule meaning,
just 16x fewer accumulation micro-steps because the GPU does 16 sequences in
parallel instead of one at a time. This is the difference that actually produces the
speedup; it isn't "bigger batch = faster" so much as "no longer paying Python-loop
and synchronization overhead 32 times per update."

The startup banner should confirm the resume:

```
Model: 50m  |  51,475,968 parameters  |  device=cuda  |  attn_impl=sdpa
Precision: torch.bfloat16 | batch 16 x accum 2 = 32 seqs/update
Resumed from checkpoints/50m/latest.pt at step 311,199
```

If `attn_impl` in that banner doesn't match the checkpoint's, `trainer.py` remaps the
attention weights automatically (`checkpoint.remap_attn_impl`) — same values,
different parameter names, no manual step. If it prints `step 0` instead of your real
step count, the checkpoint didn't land where `gpt-train` looks; re-check
`checkpoint_prefix` before letting it train from scratch by accident.

Once it's running, find out if you can push further than the safe default:

```bash
# In a second pane/session — the run above keeps going regardless
GPT_BATCH_SIZE=32 uv run gpt-benchmark --sweep-batch 16,32,48 --warmup-min 2 --measure-min 5
```

50m is a third the parameter count of `custom-gpt-153m`, which already measures safe
at `batch_size=16` on this same `context_length=1024`/GPT-2 vocab — there's likely
headroom, but that's a measurement, not a guess (see `docs/GPU_TRAINING.md` in each
`from_scratch/*` project for why this repo always benchmarks before committing a
multi-hour run).

## Disconnecting safely, and checking back in

Once the banner above confirms training is actually running, leave it running and
get out **without killing it**:

```
Ctrl-b  d
```

That's `tmux`'s detach chord — `Ctrl-b`, release, then `d`. It detaches the
*terminal* from the session; the session (and `gpt-train` inside it) keeps running
on the instance regardless of what your SSH connection does next. You can now type
`exit`, close the terminal, close your laptop — none of it touches the training run.

To come back and see the exact same running session — same scrollback, same
process, whether you reconnect five minutes or five hours later:

```bash
make ssh                    # a fresh SSH connection — this part always changes
tmux attach -t train        # re-attaches to the SAME session — this part doesn't
```

`tmux attach` doesn't start anything new; it re-opens the window onto whatever has
been running the whole time. If the step counter picks up right where you left it,
that's confirmation nothing was interrupted.

A few things worth checking on reconnect, roughly in order of what's actually gone
wrong before:

- **Nothing prints when you attach, or the shell just says "no session":** `tmux ls`
  lists whatever sessions do exist. This means the session ended — its scrollback
  is gone with it, so check for *why* elsewhere: `dmesg | tail -50` for an
  OOM-killer line (a batch size too large for VRAM kills the process, not the
  instance), and compare `checkpoints/50m/latest.pt`'s step count against 311,199
  to see how far it got before it died.
- **It attaches, but the step count looks frozen:** the process is alive but stuck —
  distinct from "instance stopped," which `make status` would show directly.
- **You're not sure it's actually still training, versus a hung shell:** `nvidia-smi`
  in a *second* SSH session (`make ssh` again, don't attach to `train` from here) —
  GPU utilization near 0% with the session still attached usually means it already
  finished or errored out silently; scroll up in the attached pane to check.

If you forget the `tmux new -s train` step entirely and run `gpt-train` bare over
SSH, the very first disconnect — intentional or not — ends the run. There's no
recovery from inside that session once it's gone; the only path back is whatever
`checkpoint_sync_minutes` had already pushed to S3 (see below), which is at most a
few minutes stale, not the whole run.

**Alternative: `make train-bg` instead of `tmux`.** This project's own `Makefile`
supports the same "survive disconnect" goal a different way — `nohup` plus a
redirect to `logs/train_stdout.log`, backgrounded before you even type anything
interactive:

```bash
make train-bg
tail -f logs/train_stdout.log      # watch it live; Ctrl-C only stops the tail
make train-status                  # is it still running, without tailing anything
make train-stop                    # the clean way to stop it (not Ctrl-C on the tail)
```

Either approach is fine — `tmux` gives you back the exact interactive session
(useful for watching the startup banner, or dropping into a debugger); `train-bg`
gives you a plain log file (useful for `grep`-ing history, or if `tmux` isn't
installed). What actually matters is that one of them is used — the failure mode
common to both alternatives is running `gpt-train` bare in a foreground SSH shell.

## While it runs, and after

```bash
make sync-log            # tail the periodic checkpoint-sync / spot-watch journal
```

The instance is already protected without you doing anything further:
`checkpoint_sync_minutes` pushes `checkpoints/` to the *same* `checkpoint_prefix` a
future resume reads from every N minutes, and (on spot) a second watcher flushes
again inside the ~2-minute interruption notice. A reclaimed box costs you one sync
interval, not the run — and the next `make up` after a reclaim resumes from wherever
that last sync landed, with no manual re-upload needed.

```bash
make download-checkpoints PROJECT_DIR=../../from_scratch/custom-gpt-50m
make down                 # destroy the instance; checkpoint is already safe in S3
```

`make down` ends compute billing entirely (vs. `make stop`, which still bills ~$8/mo
for the idle EBS volume). Nothing about resuming again later requires the instance
to have stayed up — the corpus and the checkpoint are the only state that matters,
and both already live in S3.
