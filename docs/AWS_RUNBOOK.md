# Running a training job on AWS

Operational companion to each project's `docs/GPU_TRAINING.md` (which covers instance
*choice* and the token budget). This one covers getting a run onto AWS and off again
without wasting money.

> **Prefer not to click through steps 4–6?** [`infra/aws-gpu-node/`](../infra/aws-gpu-node/)
> is this runbook as Terraform: same instance type, AMI, disk, security group and IAM
> role, plus the two safety nets this page recommends setting up "before you start, not
> after" (a budget alert and an automatic stop when the GPU goes idle). `make apply`,
> `make ssh`, `make stop`. The reasoning behind each value still lives here.

## Tomorrow: the checklist

Work top to bottom. Steps 1–3 happen on the Mac, **before** anything is billing.

### On the Mac (free)

- [ ] **1. Confirm the config is what you intend.**
      `make config` — check parameter count and the printed token budget.
- [ ] **2. Build the token files.** `make tokenize` — verify the printed token count.
      (For `custom-gpt-200m`: `make tokenizer` first, always.)
- [ ] **3. Upload the corpus to S3.** Ship `.bin`, never `.txt`:
      ```bash
      aws s3 cp data/train.bin      s3://<bucket>/corpus/
      aws s3 cp data/train.bin.json s3://<bucket>/corpus/
      aws s3 cp data/test.bin       s3://<bucket>/corpus/
      aws s3 cp data/test.bin.json  s3://<bucket>/corpus/
      # custom-gpt-200m only — its vocabulary is its own:
      aws s3 cp tokenizer/tokenizer.json s3://<bucket>/tokenizer/
      ```

### Launch (billing starts)

- [ ] **4. Create an IAM role** (once, reusable): trust `ec2.amazonaws.com`, attach a
      policy allowing `s3:GetObject`/`PutObject`/`ListBucket` on your bucket. Attach it
      to the instance at launch. This is why you never put access keys on the box.
- [ ] **5. Launch `g6.xlarge`** with:
      AMI **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** ·
      **100 GB gp3** root · your key pair · security group SSH from **My IP** ·
      the IAM role from step 4.
- [ ] **6. Connect.**
      ```bash
      chmod 400 ~/Downloads/<key>.pem
      ssh -i ~/Downloads/<key>.pem ubuntu@<public-ip>
      ```

### On the instance

- [ ] **7. Verify the GPU before anything else.**
      ```bash
      nvidia-smi
      python3 -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
      ```
      Expect `NVIDIA L4 True`. `False` means Turing — stop, you are on the wrong instance.
- [ ] **8. Code and deps.**
      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
      git clone https://github.com/shukla-surendra/tiny_llm.git
      cd tiny_llm/from_scratch/custom-gpt-153m && uv sync
      ```
- [ ] **9. Pull the corpus.** `aws s3 sync s3://<bucket>/corpus/ data/`
      (and `s3://<bucket>/tokenizer/ tokenizer/` for 200m)
- [ ] **10. Measure before committing.**
      ```bash
      uv run gpt-benchmark --sweep-batch 8,16,24,32 --warmup-min 2 --measure-min 5
      ```
      Read **peak VRAM** first, then MFU. Adjust `GPT_BATCH_SIZE`/`GPT_STEPS` now, not
      20 hours in.
- [ ] **11. Start the run, detached, with a dead-man switch.**
      ```bash
      tmux new -s train
      nohup sh -c 'uv run gpt-train \
        && aws s3 sync checkpoints/ s3://<bucket>/checkpoints/ \
        && sudo shutdown -h now' > logs/train_stdout.log 2>&1 &
      # Ctrl-b d to detach
      ```
      Check the banner says `Precision: torch.bfloat16` and the expected token budget.
- [ ] **12. Walk away.** `tail -f logs/train_stdout.log` or `make train-status` to peek.

### Finishing

- [ ] **13. Sync results off** (the dead-man switch already does this; verify):
      `aws s3 sync checkpoints/ s3://<bucket>/checkpoints/`
- [ ] **14. STOP the instance.** A forgotten `g6.xlarge` is **$19/day**.

---

## EC2 or SageMaker?

**Use EC2 for your first runs.** SageMaker becomes worth it later, and the reason is
about how many times you run, not about which is "better".

| | EC2 | SageMaker Training Jobs |
|---|---|---|
| Mental model | a Linux box you SSH into | submit a job, it provisions and tears down |
| Your existing workflow | `make train` unchanged | must package as a container / estimator |
| You pay for | **instance uptime** — including setup, debugging, and the hours you forgot to stop it | training seconds only |
| Interactive debugging | yes | awkward |
| Managed spot (~60–70% off) | you handle interruption yourself | built in, with automatic checkpoint resume |
| Cold start | ~1 min | ~3–5 min provisioning per job |

The deciding factors here:

- This repo's workflow is already `Makefile` + shell shaped. On EC2 it transfers with
  zero adaptation; on SageMaker you would first write a container and an entrypoint.
- The first thing you must do on the instance is **measure** (`make benchmark`) and
  tune `batch_size` against real VRAM. That is inherently interactive.
- A single ~24 h run is not a pipeline. SageMaker's real payoff is repeated jobs,
  hyperparameter sweeps, and managed spot.

**Switch to SageMaker when** you are running the same job repeatedly, or when a run
gets long enough that managed spot's 60–70% saving outweighs the packaging work. At
g6.xlarge's $0.80/hr, a 24 h run is ~$19 on demand and ~$6 on spot — real, but not
worth restructuring your tooling for on run one.

## EC2 runbook

### 1. Launch

| setting | value | why |
|---|---|---|
| Instance | `g6.xlarge` | L4 24 GB, bf16 + TF32. See `GPU_TRAINING.md` for why not g4dn/g4ad |
| AMI | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** | NVIDIA driver + CUDA preinstalled. On a plain Ubuntu AMI you will spend an hour on drivers |
| Storage | **100 GB gp3** | the 8 GB default is not close. Corpus `.bin` (~5 GB) + checkpoints (~1.8 GB each) + OS |
| Key pair | create/download `.pem` | this is the only time AWS shows it to you |
| Security group | SSH (22) from **My IP** | not `0.0.0.0/0` |
| Region | pick one and stay in it | S3 transfer is free within a region, billed across |

```bash
chmod 400 ~/Downloads/oxide-train.pem        # SSH refuses a world-readable key
ssh -i ~/Downloads/oxide-train.pem ubuntu@<public-ip>
```

### 2. Verify the GPU is what you think it is

Before anything else — this is the check that catches a wrong instance type:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
# expect: NVIDIA L4 True
```

`is_bf16_supported() == False` means you are on Turing (T4) and `precision="auto"`
will silently fall back to fp32 — see `GPU_TRAINING.md`.

### 3. Get the code and the data

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/shukla-surendra/tiny_llm.git
cd tiny_llm/from_scratch/custom-gpt-153m
uv sync
```

If the repo is private, use a fine-grained PAT or a deploy key — do **not** paste a
long-lived credential into shell history on a box you will terminate.

**Ship the `.bin`, not the `.txt`.** Tokenize on your Mac (free), upload the token
files. They are half the size, and re-tokenizing on an hourly-billed GPU is money for
nothing:

```bash
# on the Mac
aws s3 cp data/train.bin      s3://<bucket>/corpus/
aws s3 cp data/train.bin.json s3://<bucket>/corpus/     # the tokenizer fingerprint
aws s3 cp data/test.bin       s3://<bucket>/corpus/
aws s3 cp data/test.bin.json  s3://<bucket>/corpus/

# on the instance
aws s3 sync s3://<bucket>/corpus/ data/
```

Copy the `.bin.json` sidecars too. They are what makes a tokenizer mismatch fail
loudly instead of training on ids that index the wrong embedding rows — see
`custom-gpt-200m/docs/DATA_LAYOUT.md`. For `custom-gpt-200m`, also upload
`tokenizer/tokenizer.json`; that project's vocabulary is its own.

S3 needs an IAM instance role (or `aws configure`). `scp` works for a one-off but is
much slower than S3 within a region.

### 4. Measure before committing

```bash
make config                                   # confirm params and token budget
make benchmark                                # 10 min warm-up + 50 min measured
uv run gpt-benchmark --sweep-batch 8,16,24,32 --warmup-min 2 --measure-min 5
```

Read **peak VRAM** first — `batch_size=16` is an estimate that has never been measured
on real hardware. Then read MFU: if it lands near 15% rather than 25%, cut `GPT_STEPS`
before starting rather than discovering the overrun 20 hours in.

### 5. Train — detached, or you will lose it

An SSH drop kills a foreground process and takes the run with it.

```bash
make train-bg          # nohup + writes logs/train_stdout.log
make train-status
tail -f logs/train_stdout.log
```

Or `tmux new -s train` → `make train` → `Ctrl-b d` to detach, `tmux attach -t train` to
return. Either is fine; the failure is running `make train` bare over SSH.

The startup banner tells you whether it is doing what you expect:

```
Precision: torch.bfloat16 | batch 16 x accum 4 = 64 seqs/update
Budget: 150,000 steps x 16,384 tok = 2.46B tokens (16.1 tok/param, ...)
Progress: step .../150,000  |  ... steps/hr so far
ETA: ... more training-hours -> ~... if run continuously
```

If `Precision` says `fp32` on a CUDA box, stop and fix it — you are leaving most of the
GPU unused.

### 6. Get the results off, then **stop the instance**

```bash
aws s3 sync checkpoints/ s3://<bucket>/checkpoints/     # ~1.8 GB per 153m checkpoint
aws s3 cp logs/train_eval_history_153m.csv s3://<bucket>/logs/
```

Then stop it. A forgotten `g6.xlarge` is **$19/day**.

Two safety nets worth setting up before you start, not after:

- **A billing alarm** in CloudWatch at a threshold you would be annoyed to cross.
- **A dead-man switch.** Append a shutdown to the training command so a finished run
  does not idle overnight:

  ```bash
  nohup sh -c 'uv run gpt-train && aws s3 sync checkpoints/ s3://<bucket>/checkpoints/ \
               && sudo shutdown -h now' > logs/train_stdout.log 2>&1 &
  ```

  Stopped (not terminated) keeps the EBS volume, so you can restart and resume from
  `latest.pt`. EBS still bills at ~$0.08/GB-month — 100 GB is ~$8/month idle.

## Getting the corpus from your Mac onto the instance

**Use S3. Not scp.** Both start with the same slow hop — your home upload — but only
one of them makes you pay it once.

| | Mac → scp → EBS | Mac → S3 → EBS |
|---|---|---|
| First upload | home upload speed | home upload speed (same) |
| **Every later instance** | **home upload speed again** | ~1 min (in-region, 100+ MB/s) |
| Survives terminate / spot kill | no | **yes** |
| Extra cost | none | ~$0.02/month for 600 MB |
| Setup | none | one bucket |

You will launch more than once — a benchmark instance, the real run, a restart after a
mistake, a second model size. scp charges you the full upload each time.

What the upload actually costs, for today's corpus (`train.bin` 535 MB + `test.bin`
60 MB ≈ **583 MB** — note you do **not** upload the 1.1 GB `train.txt`):

| home upload | 583 MB (now) | ~5 GB (a 2.5B-token corpus) |
|---|---|---|
| 10 Mbps | 7.8 min | 68 min |
| 25 Mbps | 3.1 min | 27 min |
| 50 Mbps | 1.6 min | 14 min |
| 100 Mbps | 0.8 min | 7 min |

At 2.5B tokens the difference stops being academic: an hour of re-uploading per
instance, versus a minute.

### One-time setup on the Mac

```bash
brew install awscli
aws configure                      # access key, secret, default region — use an IAM
                                   # user with S3 access, not your root account
aws s3 mb s3://oxide-llm-corpus --region us-east-1
```

Bucket names are globally unique — pick something of your own. Keep it in the **same
region** as the instance: transfer between S3 and EC2 in-region is free, cross-region
is billed and slower.

### Upload (Mac)

```bash
cd from_scratch/custom-gpt-153m

aws s3 sync data/ s3://oxide-llm-corpus/153m/corpus/ \
    --exclude "*" --include "*.bin" --include "*.bin.json"
```

The `--exclude "*" --include ...` pair is doing real work: `data/` also holds the 1.1 GB
`train.txt`, a 17 GB `hf_cache/`, and the staging directories. You want none of them on
the instance — only the token files and their fingerprint sidecars.

`aws s3 sync` does multipart uploads automatically and is **resumable**: if your
connection drops, re-run the same command and it continues rather than restarting.

For `custom-gpt-200m`, the tokenizer goes too — its vocabulary is its own and the
`.bin` is meaningless without it:

```bash
aws s3 cp tokenizer/tokenizer.json s3://oxide-llm-corpus/200m/tokenizer/
```

### Download (instance)

```bash
cd tiny_llm/from_scratch/custom-gpt-153m
aws s3 sync s3://oxide-llm-corpus/153m/corpus/ data/
```

No credentials needed if the instance has the IAM role from step 4. This runs at
in-region S3 speed — hundreds of MB/s, so 583 MB is seconds and 5 GB is about a minute.

### Backfill the sidecar first if your `.bin` predates the guard

`.bin.json` sidecars are written by `build_token_bin`, which only gained that behaviour
recently. A `.bin` built before then has **no sidecar**, and the consequences are quiet
rather than loud:

* the `--include "*.bin.json"` in the upload command matches nothing, so it uploads the
  token file alone without complaint;
* the verification snippet below then fails with `FileNotFoundError` rather than a
  useful message;
* and on the instance, `load_token_array` treats absent metadata as "unverifiable,
  allow" — so the cross-tokenizer guard silently does not protect that run.

Check, and rebuild if needed, **before** uploading:

```bash
ls data/*.bin.json 2>/dev/null || make tokenize-force
```

`tokenize-force` re-tokenizes from `data/train.txt`, so it takes as long as the original
build (~100 s for the 280M-token corpus) and produces a byte-identical `.bin` plus the
sidecar. Do it on the Mac, where the CPU is free.

### Verify before you train on it

A truncated `.bin` does not announce itself — it just trains on a shorter corpus. The
sidecar records the true token count, so check it:

```bash
python3 -c "
import json, os
for name in ('train','test'):
    b = os.path.getsize(f'data/{name}.bin')
    m = json.load(open(f'data/{name}.bin.json'))
    print(f'{name}: {b//2:,} tokens on disk, {m[\"tokens\"]:,} recorded  '
          f'{\"OK\" if b//2 == m[\"tokens\"] else \"TRUNCATED\"}')"
```

`gpt-train` will separately refuse to start if the sidecar's tokenizer fingerprint does
not match the configured tokenizer — see "Mounting S3" below for why that guard exists.

### If you would rather skip S3 entirely

Valid for a single throwaway run. Use `rsync`, not `scp` — it resumes a broken transfer
instead of starting over:

```bash
rsync -avzP -e "ssh -i ~/Downloads/<key>.pem" \
      data/train.bin data/train.bin.json data/test.bin data/test.bin.json \
      ubuntu@<public-ip>:~/tiny_llm/from_scratch/custom-gpt-153m/data/
```

Just know you are re-paying that upload on every instance you ever launch, and that
nothing survives a terminate.

## What survives stop, terminate, and reboot

This is the question that costs people a night of training, so be precise about it.

| | Root EBS volume | Extra EBS volumes | Instance store (NVMe) | Public IP | You pay for |
|---|---|---|---|---|---|
| **Reboot** | kept | kept | kept | kept | everything |
| **Stop** | **kept** | **kept** | **LOST** | changes* | EBS only (~$8/mo per 100 GB gp3) |
| **Terminate** | **DELETED** by default | kept by default | LOST | released | nothing |

\* unless you attached an Elastic IP.

**Stopping does not lose your EBS data.** Root and attached EBS volumes persist
exactly as they were; you stop paying for compute and keep paying only for storage.
Restart, SSH back in, and `make train` resumes from `latest.pt` — this is the normal
way to pause an overnight run.

**Terminating destroys the root volume.** `DeleteOnTermination` defaults to *true* for
the root volume (and *false* for volumes you attach separately). So "terminate" means
your code, corpus and checkpoints are gone unless they are in S3. Sync before you
terminate, every time.

**`g6.xlarge` has a 250 GB NVMe instance store, and it is ephemeral.** It is much
faster than gp3 and effectively free, which makes it tempting for the corpus — but it
is wiped on stop *and* on any underlying host migration. If you use it, treat it as a
cache: corpus copy yes, checkpoints never.

**In practice:** treat the instance as disposable and S3 as the source of truth.
Anything you would be sad to lose should be in S3 before you walk away.

## Mounting S3 — and why not to, here

You can mount a bucket as a filesystem, but **do not put the training path on it.**
Two concrete reasons, both grounded in this codebase rather than general advice:

**Checkpoints would fail.** `checkpoint.py`'s `atomic_save` writes a `.tmp` and then
calls `tmp_path.replace(path)` — a rename — so a reader never sees a torn file.
**Mountpoint for Amazon S3 does not support renaming an object.** Checkpointing
directly to a mounted path breaks on the very mechanism that makes it safe.

**The corpus would be unusable.** `get_batch` does `np.random.randint(...)` and slices
random windows out of a memmap. Over a FUSE mount every one of those windows is an
HTTP GET with S3's latency, thousands of times per minute — orders of magnitude slower
than a local read, on the hottest path in the loop. Memmapping over S3 is the wrong
tool at a fundamental level, not a tuning problem.

**Do this instead — copy, don't mount:**

```bash
aws s3 sync s3://<bucket>/corpus/ data/          # pull once at the start
aws s3 sync checkpoints/ s3://<bucket>/checkpoints/   # push at the end
```

Within a region S3 transfer is free and fast; 5 GB takes a couple of minutes. Local
gp3 is what serves the random reads during training.

**If you still want a mount** for convenience — browsing artifacts, ad-hoc reads of old
checkpoints — `mount-s3` is the right tool, just keep it out of the training path:

```bash
# Mountpoint for Amazon S3 (AWS-official, FUSE)
wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
sudo apt-get install -y ./mount-s3.deb
mkdir -p ~/s3
mount-s3 <bucket> ~/s3            # read-only browsing; uses the instance IAM role
```

It is good at exactly one thing: large sequential reads. That is not what this training
loop does.

## Cost sheet

At g6.xlarge on-demand (~$0.8048/hr, us-east-1):

| run | hours | cost |
|---|---:|---:|
| `gpt-benchmark` default (10+50 min) | 1.0 | $0.80 |
| 153m @ 2.46B tokens | ~21 | ~$17 |
| 200m @ 4B tokens (Chinchilla) | ~44 | ~$35 |
| 200m @ 20B tokens (reasoning-relevant) | ~220 | ~$177 |

Storage on top: 100 GB gp3 ~$8/month, plus S3 at ~$0.023/GB-month.

The benchmark hour is the best money in this table — it is what stops you finding out
20 hours in that MFU was half what you assumed.

## When to move to SageMaker

Worth the packaging effort once any of these is true:

- You are launching the same job repeatedly (sweeps, restarts, several model sizes).
- Runs are long enough that **managed spot** matters. SageMaker checkpoints to S3 and
  resumes automatically after an interruption; on EC2 spot you build that yourself.
- You want the run recorded — job name, hyperparameters, metrics, artifacts — instead
  of living in one instance's shell history.

The training code needs no changes for it: `gpt-train` already resumes from
`latest.pt`, which is exactly the contract managed spot expects. What you add is a
container image and an entrypoint that syncs `checkpoints/` to `/opt/ml/checkpoints`.
