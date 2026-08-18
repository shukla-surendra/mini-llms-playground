# What happens in one training step — straight from the source code

Every line below is real, from `custom-gpt-50m/src/gpt/training/trainer.py` and
`src/gpt/data/dataset.py` — the actual code currently running the GCP session. File
and line numbers included so this can be cross-checked directly, not just trusted.
Companion to the plainer-language versions already in this folder
(`batch_size_and_grad_accum_layman.md`, `gpu_vs_local_layman_explainer.md`) — this
one is the "here's literally the code" version.

## The loop itself

`trainer.py:340` — `_run_loop`'s main loop:
```python
for step in range(start_step, train_cfg.steps):
```
One iteration of this loop = one training step. `start_step` comes from the resumed
checkpoint (or 0 for a fresh run); `train_cfg.steps` is the stopping point
(default 1,000,000).

## Step 1 — pull a fresh, random batch of data

`trainer.py:382`:
```python
xb, yb = get_batch(train_tokens, ctx_len, train_cfg.batch_size, device)
```
calls into `data/dataset.py:284-301` (the disk-memmap production path — `train_tokens`
is a `numpy.memmap` over `data/train.bin`, never fully loaded into RAM):
```python
max_start = len(tokens) - ctx_len - 1
ix = np.random.randint(0, max_start, size=batch_size)          # batch_size random start points
x = torch.from_numpy(np.stack([tokens[i:i + ctx_len] for i in ix]).astype(np.int64))
y = torch.from_numpy(np.stack([tokens[i + 1:i + ctx_len + 1] for i in ix]).astype(np.int64))
```
`ix` = `batch_size` random positions in the 2.46B-token corpus. `x` = the
`context_length`-token chunk starting at each position. `y` = the *same* chunk,
shifted one token to the right — literally "whatever token actually came next" at
every position. No shuffled index list, no epoch tracking — every step draws fresh
random positions, with replacement, from the whole corpus.

**Impact of changing values here**:
- **`batch_size`** ↑ → more random windows pulled per step, more data per GPU
  handoff (today's actual change: 1 → 4). No effect on what the model learns per
  update, only how much work is bundled into one step. See
  `batch_size_and_grad_accum_layman.md` for the throughput mechanism.
- **`context_length`** (`ctx_len`, fixed at 1024 for this model) — changing this
  isn't a training-time option; it's baked into the model's position-embedding
  table shape. Would require a new model. See the earlier conversation on this
  distinction.

## Step 2 — forward pass: the model guesses

`trainer.py:383-386`:
```python
with torch.autocast(device_type=amp_device_type, dtype=amp_dtype or torch.float32,
                    enabled=amp_dtype is not None):
    loss = next_token_loss(model(xb), yb, model_cfg.vocab_size)
```
`model(xb)` runs the actual transformer forward pass — for every one of the
`batch_size × context_length` token positions, it outputs a probability
distribution over all 50,257 possible next tokens. `torch.autocast` is what makes
this run in `bfloat16` on CUDA (real Tensor Core speedup) while staying `fp32` on
MPS — see `checkpoint_resume_theory.md` for why this doesn't affect the *stored*
weights either way.

`next_token_loss` (`data/dataset.py:306-307`):
```python
def next_token_loss(logits, targets, vocab_size):
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
```
Cross-entropy loss: one number summarizing how far the model's predicted
probabilities were from the real next tokens, averaged over every position in the
batch.

**Impact of changing values here**: `precision` (`"auto"`/`"bf16"`/`"fp16"`/`"fp32"`)
controls only *how* this math is done (speed/memory), never *what* is computed —
forcing `fp32` everywhere would be slower but numerically the safest; forcing
`fp16` without a `GradScaler` (not implemented here, deliberately — see the code
comment) risks silent gradient underflow, which is exactly why this project chose
`bf16` over `fp16` for its "auto" default.

## Step 3 — backward pass: calculate the nudge

`trainer.py:389`:
```python
(loss / train_cfg.grad_accum_steps).backward()
```
PyTorch's autodiff computes, for all 51,475,968 parameters, the gradient of the loss
with respect to that parameter — "which direction, and how strongly, should this
number move to reduce the loss." Dividing by `grad_accum_steps` *before* calling
`.backward()` is what makes the eventual accumulated sum equal a proper *average*
over the full effective batch rather than a sum that grows with more accumulation
steps.

**Impact of changing values here**: `grad_accum_steps` ↑ with `batch_size` held
fixed = larger effective batch, same memory footprint, but more steps needed per
real update (today: 32 → 8, because `batch_size` grew 1 → 4 to compensate and keep
effective batch at 32).

## Step 4 — only every `grad_accum_steps`-th step: actually update the model

`trainer.py:391-397`:
```python
is_accum_boundary = (step - start_step + 1) % train_cfg.grad_accum_steps == 0
if is_accum_boundary or step == train_cfg.steps - 1:
    for group in optimizer.param_groups:
        group["lr"] = lr_for_step(step, train_cfg)
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```
Four things happen only on a boundary step:
1. **Learning rate update** — `lr_for_step` (`trainer.py:54-62`) recomputes the LR
   fresh from the step number (linear warmup, then cosine decay to `min_lr`) — never
   stored, always derived, which is exactly why LR resumes correctly on any device.
2. **Gradient clipping** — `grad_clip_norm` (default `1.0`) caps the *total size* of
   the accumulated nudge. If a batch happened to produce an unusually huge gradient
   (an outlier/bad batch), this prevents it from causing a destabilizing jump.
3. **`optimizer.step()`** — AdamW actually updates all 51.5M parameters, using both
   the accumulated gradient *and* its own running momentum/variance estimates (its
   "memory" of recent gradients — see `checkpoint_resume_theory.md` for why this
   state is saved/restored across resumes).
4. **`optimizer.zero_grad()`** — the accumulated tally resets to zero, ready for the
   next `grad_accum_steps`-window.

**Impact of changing values here**:
- **`lr` / `min_lr`** — higher `lr` = bigger nudges, faster initial progress but
  higher risk of instability; lower = safer but slower. `min_lr` is the floor the
  cosine schedule decays to and never goes below.
- **`grad_clip_norm`** — lower = more conservative (clips more aggressively,
  potentially slowing legitimate large updates); higher/disabled = more exposure to
  one bad batch causing a damaging jump.
- **`weight_decay`** (`0.1` default, applied inside AdamW, not shown above but part
  of `optimizer.step()`) — a constant small pull on every parameter toward zero,
  independent of the gradient; higher = stronger regularization (fights
  overfitting harder, but can also underfit if too high).

## Step 5 (every step) — bookkeeping, and (occasionally) checkpointing

`trainer.py:412-416`:
```python
state["processed_tokens"] += train_cfg.batch_size * ctx_len
...
if (step + 1) % train_cfg.save_every_steps == 0:
    atomic_save(payload(step), paths.latest_checkpoint)
```
`processed_tokens` (what `est_epoch` is computed from) increments every step by
however many tokens that step actually processed. Every `save_every_steps` steps
(default 200), the full state — weights, optimizer momentum, step count — gets
written to disk, so an interruption never loses more than that many steps of
progress.

Separately, every `eval_interval` steps (`trainer.py:350-380`, not shown here in
full — see `estimate_loss` at line 119), the model is scored against **held-out test
data it never trains on**, logged to `logs/train_eval_history_50m.csv`, and a new
`best.pt`/`serving.pt` is saved if that test loss improved.

**Impact of changing values here**:
- **`save_every_steps`** — lower = safer against losing progress on a crash, but
  more disk I/O overhead per step (this project's own default, `200`, was tuned
  against exactly this tradeoff).
- **`eval_interval` / `eval_batches`** — this project's own config comments
  (`config.py`) document a real, measured incident: too few `eval_batches` made
  `best_test_loss` noisy enough that a single lucky/unlucky draw could hold the
  "best" checkpoint title for 60,000+ steps despite the true trend improving — raising
  `eval_batches` (20→80) was a direct fix for that, not a arbitrary tuning choice.

## The whole thing, top to bottom, current live config

```
1. Pull batch_size=4 random 1024-token windows      -> 4,096 fresh tokens
2. Forward pass (bf16 on this GPU) -> loss (one number)
3. Backward pass -> gradients, added to a running tally (divided by 8 first)
4. Every 8th step: apply the tallied nudge (LR update, clip, AdamW step), reset tally
5. Every step: track tokens processed; every 200 steps: save a full checkpoint;
   every 800 steps: evaluate on held-out data, save `best.pt` if it's the new best
```
