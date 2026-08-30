# GCP GPU Training Command Tutorial

This tutorial explains the main shell, Terraform, Google Cloud Storage (GCS), GCP Compute, and process-management commands used in the GPU training workflow.

The examples are based on the commands used while preparing the `custom-gpt-153m` training run.

---

## 1. Mental Model: What Are We Doing?

The overall workflow is:

```text
Local Mac
   |
   | Terraform
   v
GCP infrastructure
   |
   | gcloud / gsutil
   v
GCS bucket
   |
   | VM startup
   v
G2 / L4 GPU VM
   |
   | download corpus
   v
Training
```

There are five main command groups:

1. **Shell commands** — inspect files and processes.
2. **Make commands** — project-specific shortcuts.
3. **Terraform commands** — create and manage infrastructure.
4. **GCS commands** — upload and inspect training data.
5. **GCP Compute commands** — inspect and create GPU VMs.

---

# 2. Shell Basics

Before learning GCP, understand the Unix commands used around it.

## 2.1 `pwd`

Shows the current directory.

```bash
pwd
```

Example:

```text
~/projects/2026/mini-llms-playground
```

Useful when a command depends on your current working directory.

---

## 2.2 `cd`

Changes directory.

```bash
cd /path/to/project
```

Example:

```bash
cd from_scratch/custom-gpt-153m
```

Go up one directory:

```bash
cd ..
```

---

## 2.3 `ls`

Lists files.

```bash
ls
```

Long format:

```bash
ls -lh
```

Recursive listing:

```bash
ls -R
```

---

## 2.4 `cat`

Prints a file.

```bash
cat data/train.bin.json
```

It is particularly useful for small JSON/configuration files.

---

## 2.5 `head`

Prints the beginning of a file.

```bash
head file.txt
```

Print the first 100 bytes:

```bash
head -c 100 file.bin
```

In the training workflow it was used to create a smaller binary:

```bash
head -c 5857020462 data/train.full-7b.bin > data/train.bin
```

Conceptually:

```text
large binary
     |
     | first 5,857,020,462 bytes
     v
smaller binary
```

> Important: `head -c` works at the byte level. For tokenized binary data, you must understand the binary format before truncating it this way.

---

# 3. `sed`

`sed` is commonly used to print selected sections of files.

```bash
sed -n '1,220p' Makefile
```

Meaning:

- `-n` → don't print everything
- `1,220p` → print lines 1 through 220

Another example:

```bash
sed -n '90,145p' src/gpt/config.py
```

This is useful for inspecting a particular section without opening the entire file.

---

# 4. `rg` — ripgrep

`rg` searches files quickly.

Search for a word:

```bash
rg "token_count"
```

Search specific directories:

```bash
rg "steps|batch_size|context_length" src
```

List files:

```bash
rg --files custom-gpt-153m
```

A useful pattern from the workflow:

```bash
rg --files custom-gpt-153m | sed -n '1,160p'
```

This means:

1. Find project files.
2. Send the output to `sed`.
3. Display only the first 160 lines.

---

# 5. `find`

`find` searches the filesystem.

Example:

```bash
find /Users/surendrashukla/.config/gcloud/logs \
  -type f \
  -name '*.log'
```

Useful when you don't know exactly where a file is.

---

# 6. `stat`

Shows file metadata.

On macOS:

```bash
stat -f '%N %z bytes' data/train.bin
```

This tells you:

- filename
- file size in bytes

Useful for checking the size of training binaries.

---

# 7. `df`

Shows filesystem disk usage.

```bash
df -h .
```

Example output:

```text
Filesystem   Size   Used   Avail
/dev/disk3   460G   378G    57G
```

This is important before generating or downloading multi-GB datasets.

---

# 8. Make Commands

`make` is not a GCP command.

It executes commands defined in a project's `Makefile`.

For example:

```bash
make init
make plan
make upload-corpus
```

You should inspect the Makefile to understand what each target actually does:

```bash
sed -n '1,220p' Makefile
```

Or search for a target:

```bash
rg -n "upload-corpus|init|plan" Makefile
```

## Important principle

When you run:

```bash
make upload-corpus
```

you should think:

```text
make target
    ↓
commands inside Makefile
    ↓
gcloud / terraform / shell
```

So learning the underlying command is more important than memorizing the Make target.

---

# 9. Terraform

Terraform manages infrastructure as code.

In this workflow Terraform was used for the GCP GPU infrastructure.

The typical lifecycle is:

```text
terraform init
      ↓
terraform plan
      ↓
terraform apply
      ↓
infrastructure exists
```

---

## 9.1 `terraform init`

Initializes the Terraform project.

```bash
terraform init
```

It downloads/configures providers and prepares the working directory.

The workflow also used:

```bash
make init
```

which presumably wraps Terraform initialization.

---

# 10. `terraform plan`

Shows what Terraform intends to change.

```bash
terraform plan
```

It does **not** normally create the resources.

Example:

```text
Plan: 3 to add, 0 to change, 0 to destroy.
```

A good habit is:

```bash
terraform plan
```

before:

```bash
terraform apply
```

---

## 10.1 Saving a plan

You can save a plan:

```bash
terraform plan -out=tfplan
```

Then apply exactly that plan:

```bash
terraform apply tfplan
```

This is safer for controlled deployments.

---

# 11. `terraform apply`

Creates or updates infrastructure.

```bash
terraform apply
```

For automatic approval:

```bash
terraform apply -auto-approve
```

The GPU workflow used:

```bash
terraform apply -auto-approve -var instance_count=1
```

The variable:

```text
instance_count=1
```

was passed into Terraform to request one instance.

To use zero instances:

```bash
terraform apply -auto-approve -var instance_count=0
```

---

# 12. Terraform Variables

Terraform configuration can contain variables such as:

```hcl
project_subdir = "from_scratch/custom-gpt-153m"
corpus_prefix = "153m/corpus/"
checkpoint_prefix = "153m/checkpoints/"
```

This allowed the 153M model to have its own storage paths.

The important idea is:

```text
50M run
    → 50m/corpus/

153M run
    → 153m/corpus/
```

This prevents different experiments from sharing the same data paths.

---

# 13. `terraform output`

Shows Terraform outputs.

```bash
terraform output
```

For a specific output:

```bash
terraform output -raw bucket
```

Another example:

```bash
terraform output -raw corpus_prefix
```

This is useful when Terraform creates something such as a bucket and you need its generated name.

---

# 14. `terraform state list`

Shows resources currently tracked by Terraform.

```bash
terraform state list
```

Example conceptually:

```text
google_storage_bucket.training
google_service_account.training
google_compute_instance.gpu
```

This helps answer:

> What does Terraform currently believe it manages?

---

# 15. Google Cloud Storage

GCS is where the training corpus was uploaded before starting the GPU VM.

The bucket looked like:

```text
gs://mini-llm-gpu-llm-training-dev-us-central1/
```

The 153M corpus path was:

```text
gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/
```

---

# 16. `gcloud storage cp`

Copies files to or from GCS.

Upload:

```bash
gcloud storage cp data/train.bin \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/train.bin
```

Upload the test binary:

```bash
gcloud storage cp data/test.bin \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/test.bin
```

The basic pattern is:

```text
gcloud storage cp SOURCE DESTINATION
```

---

# 17. `gcloud storage rsync`

Synchronizes directories.

Example:

```bash
gcloud storage rsync data/ \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/ \
  --recursive
```

Think of it as:

```text
local directory
      ↓
compare
      ↓
GCS directory
      ↓
copy required files
```

This is convenient when there are multiple files.

For a single large file, `cp` is often easier to reason about.

---

# 18. `gcloud storage ls`

List objects.

```bash
gcloud storage ls \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/
```

With details:

```bash
gcloud storage ls -l \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/
```

This lets you check whether:

```text
train.bin
test.bin
train.bin.json
test.bin.json
```

exist.

---

# 19. `gcloud storage du`

Check storage usage.

```bash
gcloud storage du \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/ \
  --summarize
```

This is particularly useful for large uploads.

For example:

```text
TOTAL: 4 objects, 6000000315 bytes
```

means the bucket contains approximately 6 GB across those objects.

---

# 20. Recursive GCS Listing

You can recursively inspect a bucket:

```bash
gcloud storage ls -r \
  gs://mini-llm-gpu-llm-training-dev-us-central1/**
```

This is useful when you don't know the exact object layout.

---

# 21. GCS Object Versions

To inspect object versions:

```bash
gcloud storage ls --all-versions \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/
```

This can help diagnose unexpected old objects or overwritten files.

---

# 22. `gsutil`

`gsutil` is another Google Cloud Storage command-line tool.

Example:

```bash
gsutil cp data/train.bin \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/train.bin
```

The workflow used it as an alternative when `gcloud storage cp` appeared to stall.

---

# 23. Parallel `gsutil`

For parallel transfers:

```bash
gsutil -m cp data/train.bin \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/train.bin
```

The important option is:

```text
-m
```

which enables parallel processing.

For a large dataset, this can improve transfer performance, depending on network and client behavior.

---

# 24. Disable Composite Uploads

The workflow tried:

```bash
gcloud config set storage/parallel_composite_upload_enabled False
```

Then:

```bash
gcloud storage cp data/test.bin \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/test.bin
```

This changes the GCS client's upload behavior.

It was used because the previous upload mechanism appeared to be stuck.

---

# 25. `nohup`

`nohup` allows a command to continue after the shell/session disconnects.

Example:

```bash
nohup gsutil -m cp data/train.bin \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/train.bin \
  > /tmp/train-upload.log 2>&1 &
```

Breakdown:

```text
nohup
  ↓
don't terminate when terminal closes

gsutil -m cp
  ↓
parallel upload

> /tmp/train-upload.log
  ↓
write stdout to log

2>&1
  ↓
send stderr to same log

&
  ↓
run in background
```

---

# 26. Monitor Background Uploads

After starting:

```bash
nohup gsutil ...
```

you can check the process.

```bash
pgrep -fal gsutil
```

Or:

```bash
ps -axo pid,etime,%cpu,%mem,command | grep gsutil
```

You can also inspect the log:

```bash
tail -n 20 /tmp/train-upload.log
```

And check GCS:

```bash
gcloud storage ls -l \
  gs://mini-llm-gpu-llm-training-dev-us-central1/153m/corpus/
```

---

# 27. `pgrep`

Search running processes.

```bash
pgrep -fal gsutil
```

Useful when you want to know:

> Is my upload process still running?

For a specific upload:

```bash
pgrep -fal 'gcloud storage cp.*train.bin'
```

---

# 28. `ps`

Inspect a process.

```bash
ps -o pid,etime,%cpu,%mem,command -p 45294
```

Important fields:

```text
PID     process ID
ETIME   elapsed time
%CPU    CPU usage
%MEM    memory usage
COMMAND command being executed
```

For parent/child relationships:

```bash
ps -o pid,ppid,pgid,command -p 45294
```

Where:

```text
PID  = process ID
PPID = parent process ID
PGID = process group ID
```

---

# 29. `kill`

Stop a process.

Graceful termination:

```bash
kill -TERM <PID>
```

Example:

```bash
kill -TERM 45294
```

Interrupt:

```bash
kill -INT <PID>
```

Use `kill` carefully.

A good workflow is:

```bash
ps -p <PID>
```

then:

```bash
kill -TERM <PID>
```

and finally verify:

```bash
pgrep -fal '<process>'
```

---

# 30. `sleep`

Wait for a specified amount of time.

```bash
sleep 45
```

This was used while waiting for an upload and then checking again.

Example:

```bash
sleep 45
gcloud storage ls -l gs://...
```

---

# 31. GCP Compute

Once the data is safely in GCS, Terraform can create the GPU VM.

You can list VMs:

```bash
gcloud compute instances list \
  --project=llm-training-dev
```

A useful formatted version:

```bash
gcloud compute instances list \
  --project=llm-training-dev \
  --format='table(name,zone,status,machineType.basename())'
```

This gives a compact view:

```text
NAME       ZONE          STATUS    MACHINE_TYPE
gpu-vm     us-west1-b    RUNNING   g2-standard-4
```

---

# 32. Check Machine Types

To inspect available machine types:

```bash
gcloud compute machine-types list \
  --project=llm-training-dev
```

To filter:

```bash
gcloud compute machine-types list \
  --project=llm-training-dev \
  --filter='name=(g2-standard-4)'
```

The workflow used this to investigate which zones exposed the G2/L4 machine type.

---

# 33. Aggregated Machine Type Search

You can query across locations:

```bash
gcloud compute machine-types aggregated-list \
  --project=llm-training-dev \
  --filter='machineTypes.name=g2-standard-4'
```

This is useful when you need to determine where a particular machine type exists.

---

# 34. Checking VM Capacity

A VM can fail even when the machine type is valid.

For example:

```text
ZONE_RESOURCE_POOL_EXHAUSTED
```

means the requested GPU capacity is temporarily unavailable in that zone.

This is different from:

```text
machine type does not exist
```

So the troubleshooting process is:

```text
Does machine type exist?
        |
        v
Is it available in this zone?
        |
        v
Is capacity currently available?
        |
        v
Create VM
```

---

# 35. Typical GPU Deployment Workflow

For your 153M model, a clean workflow is:

## Step 1 — Check the project

```bash
cd ~/projects/2026/mini-llms-playground
```

## Step 2 — Inspect configuration

```bash
sed -n '1,220p' infra/gcp-gpu-node/terraform.tfvars
```

Check:

```text
project_subdir
corpus_prefix
checkpoint_prefix
zone
project
```

---

## Step 3 — Initialize Terraform

```bash
cd infra/gcp-gpu-node
terraform init
```

or:

```bash
make init
```

---

## Step 4 — Preview infrastructure

```bash
terraform plan
```

or:

```bash
make plan
```

---

## Step 5 — Verify the GCS bucket

```bash
terraform output -raw bucket
```

Then:

```bash
gcloud storage ls
```

---

## Step 6 — Upload the training corpus

For the large training binary:

```bash
gcloud storage cp data/train.bin \
  gs://BUCKET/153m/corpus/train.bin
```

For the test binary:

```bash
gcloud storage cp data/test.bin \
  gs://BUCKET/153m/corpus/test.bin
```

---

## Step 7 — Verify the upload

```bash
gcloud storage ls -l gs://BUCKET/153m/corpus/
```

Then:

```bash
gcloud storage du gs://BUCKET/153m/corpus/ --summarize
```

Do not create the GPU VM until the required objects are actually finalized.

---

# 36. Upload Monitoring Recipe

For a large file, use two independent checks.

### Check 1: Process

```bash
pgrep -fal 'gcloud storage cp.*train.bin|gsutil.*train.bin'
```

### Check 2: GCS

```bash
gcloud storage ls -l \
  gs://BUCKET/153m/corpus/train.bin
```

### Check 3: Size

```bash
gcloud storage du \
  gs://BUCKET/153m/corpus/ \
  --summarize
```

Interpretation:

```text
Process exists
+
Object is absent
=
upload is still running or stalled
```

```text
Process absent
+
Object has expected size
=
upload completed
```

```text
Process absent
+
Object absent
=
upload failed/cancelled
```

---

# 37. Create the GPU VM

Once the data is verified:

```bash
terraform apply -auto-approve -var instance_count=1
```

Then check:

```bash
gcloud compute instances list \
  --project=llm-training-dev \
  --format='table(name,zone,status,machineType.basename())'
```

You want something like:

```text
NAME       ZONE          STATUS    MACHINE_TYPE
gpu-vm     us-central1-a RUNNING   g2-standard-4
```

---

# 38. Important: `g2-standard-4` vs L4

In this workflow, the configured machine type was:

```text
g2-standard-4
```

and it provides an NVIDIA L4 GPU configuration.

Do not confuse:

```text
machine type
```

with:

```text
GPU model
```

Think:

```text
GCP machine type
        |
        v
g2-standard-4
        |
        v
GPU
        |
        v
NVIDIA L4
```

---

# 39. After the GPU VM Starts

The transcript establishes that the infrastructure/bootstrap workflow is intended to pull the corpus from the GCS prefix into the model project.

The high-level flow is:

```text
GCS
 |
 | train.bin
 | test.bin
 v
GPU VM
 |
 | bootstrap/data sync
 v
custom-gpt-153m/data/
 |
 v
training
```

The exact training command should come from the project's `Makefile`/training documentation rather than being guessed.

Inspect it with:

```bash
sed -n '1,240p' Makefile
```

and search:

```bash
rg -n "train|training|checkpoint|resume" Makefile README.md GPU_TRAINING.md training_sop.md
```

---

# 40. Stopping the GPU

When training is finished, don't leave the GPU running unnecessarily.

Depending on how the Terraform module is designed:

```bash
terraform apply -auto-approve -var instance_count=0
```

can reduce the managed instance count to zero.

Then verify:

```bash
gcloud compute instances list \
  --project=llm-training-dev
```

---

# 41. Command Cheat Sheet

## Shell

```bash
pwd
cd
ls
cat
head
sed
rg
find
stat
df
sleep
```

## Process

```bash
pgrep
ps
kill
```

## Make

```bash
make init
make plan
make status
make upload-corpus
make down
```

## Terraform

```bash
terraform init
terraform plan
terraform apply
terraform output
terraform state list
```

## GCS

```bash
gcloud storage cp
gcloud storage rsync
gcloud storage ls
gcloud storage du
```

## Alternative GCS client

```bash
gsutil cp
gsutil -m cp
```

## GCP Compute

```bash
gcloud compute instances list
gcloud compute machine-types list
gcloud compute machine-types aggregated-list
```

---

# 42. The Most Important Commands to Memorize

If your goal is to become comfortable with this workflow, don't memorize every command.

Start with these 12:

```bash
# Shell
pwd
cd
ls
rg

# Terraform
terraform init
terraform plan
terraform apply
terraform output

# GCS
gcloud storage cp
gcloud storage ls
gcloud storage du

# Compute
gcloud compute instances list
```

Then learn:

```bash
pgrep
ps
kill
gsutil -m cp
```

These are enough to understand most of the workflow.

---

# 43. A Practical Example

Suppose you have:

```text
data/train.bin
```

and want to upload it to:

```text
gs://my-bucket/153m/corpus/train.bin
```

### Upload

```bash
gcloud storage cp data/train.bin \
  gs://my-bucket/153m/corpus/train.bin
```

### Check the object

```bash
gcloud storage ls -l \
  gs://my-bucket/153m/corpus/train.bin
```

### Check total storage

```bash
gcloud storage du \
  gs://my-bucket/153m/corpus/ \
  --summarize
```

### Check whether an uploader is still running

```bash
pgrep -fal 'gcloud storage cp.*train.bin|gsutil.*train.bin'
```

### Check a process

```bash
ps -o pid,etime,%cpu,%mem,command -p <PID>
```

### Stop a stalled upload

```bash
kill -TERM <PID>
```

### Start the GPU infrastructure

```bash
terraform apply -auto-approve -var instance_count=1
```

### Verify the VM

```bash
gcloud compute instances list \
  --project=llm-training-dev \
  --format='table(name,zone,status,machineType.basename())'
```

---

# 44. Key Lessons From This Run

The transcript demonstrates several useful operational lessons:

### Lesson 1 — Don't assume an upload succeeded

A command returning or appearing active does not necessarily mean the GCS object has finalized.

Always verify with:

```bash
gcloud storage ls -l gs://...
```

and:

```bash
gcloud storage du gs://... --summarize
```

### Lesson 2 — Separate data preparation from GPU provisioning

The workflow intentionally waited for the corpus upload before creating the billable GPU VM.

That is a good cost-control pattern:

```text
prepare data
    ↓
upload data
    ↓
verify data
    ↓
start GPU
```

rather than:

```text
start GPU
    ↓
wait for 6 GB upload
    ↓
GPU sits idle and costs money
```

### Lesson 3 — Use process monitoring when uploads appear stuck

Use:

```bash
pgrep
ps
```

to determine whether a transfer is alive.

### Lesson 4 — Terraform is infrastructure, not training

Terraform handles things like:

```text
bucket
IAM
firewall
VM
network/infrastructure
```

while your Python/Make/training scripts handle:

```text
dataset
model
training
checkpoint
evaluation
```

### Lesson 5 — GCS is the bridge between local development and cloud training

The important architecture is:

```text
Mac
 |
 | upload
 v
GCS
 |
 | VM startup/download
 v
G2/L4 GPU
 |
 v
LLM training
```

This is the core concept behind the commands in your transcript.

