"""The SFT (instruction/chat fine-tuning) loop.

Deliberately a separate loop from trainer.py's pretraining `_run_loop`, not a branch
inside it: pretraining samples random context_length windows from an effectively
infinite flat token stream (data/dataset.py's get_batch), while SFT iterates, epoch by
epoch, over a small, finite, pre-tokenized example list (data/sft_dataset.py's
load_sft_examples) with dynamic padding and a real loss mask — different enough data
plumbing that forcing both into one function would obscure both. What IS reused
directly: trainer.py's stateless helpers (lr_for_step, resolve_amp, format_duration,
safe_perplexity) and checkpoint.py's atomic_save/make_payload — none of those care
whether the run is pretraining or SFT.
"""

import csv
from datetime import datetime, timezone
import math
import time

import torch
from tqdm import trange

from ..checkpoint import atomic_save, load_model, make_payload
from ..data.sft_dataset import load_sft_examples, make_sft_batch
from ..data.dataset import masked_next_token_loss
from ..inference.generate import generate_text
from .trainer import format_duration, lr_for_step, resolve_amp, safe_perplexity

EVAL_HISTORY_FIELDS = [
    "timestamp_utc",
    "step",
    "epoch",
    "lr",
    "train_loss",
    "test_loss",
    "test_perplexity",
    "best_test_loss",
    "improved",
    "total_training_hours",
]


def append_eval_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_HISTORY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def estimate_sft_loss(model, examples, sft_cfg, device, pad_id, fixed_len, eval_batches=10):
    model.eval()
    if not examples:
        model.train()
        return float("nan")
    device_type, amp_dtype = resolve_amp(sft_cfg.precision, device)
    n = len(examples)
    losses = []
    for i in range(eval_batches):
        start = (i * sft_cfg.batch_size) % max(n, 1)
        idx = [(start + j) % n for j in range(min(sft_cfg.batch_size, n))]
        x, y, attn_mask = make_sft_batch(examples, idx, device, pad_id, fixed_len=fixed_len)
        with torch.autocast(device_type=device_type,
                            dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            logits = model(x, attn_mask=attn_mask)
            loss = masked_next_token_loss(logits, y, model.token_emb.num_embeddings)
        losses.append(loss.item())
        if device == "mps":
            torch.mps.empty_cache()
    model.train()
    return sum(losses) / len(losses)


def train_sft(base_checkpoint_path, model_cfg, sft_cfg, paths, base_paths, label,
              device, resume=True, demo_prompt="What is the capital of France?"):
    """Fine-tune the model at `base_checkpoint_path` on data/sft/{train,test}.jsonl
    (sft_prepare.py's output), saving under `paths` (a distinct label/checkpoint
    namespace from the base run — see config.py's "153m-sft" preset alias).

    `resume`: if paths.latest_checkpoint already exists (an SFT run already underway),
    resumes from there and ignores `base_checkpoint_path`; otherwise seeds fresh from
    the base checkpoint. Optimizer state is never carried over from the base run even
    on a fresh start — SFT is a new training phase with its own (much lower) LR, not a
    continuation of pretraining's AdamW state.
    """
    resuming = resume and paths.latest_checkpoint.exists()
    load_path = paths.latest_checkpoint if resuming else base_checkpoint_path
    checkpoint, tokenizer, model = load_model(
        load_path, device, eval_mode=False, force_attn_impl="sdpa",
    )
    model.train()
    # model_ctx_len is the REAL architecture — the model's actual pos_emb table size —
    # and must be what every checkpoint save records (make_payload's context_length
    # below), never anything smaller. seq_cap is a separate, purely-local concept: how
    # long an SFT training example is allowed to be before truncation/padding, capped
    # lower for memory reasons on constrained hardware (see SFTConfig.max_seq_len).
    # Conflating the two previously corrupted saved checkpoints: a payload saved with
    # context_length=seq_cap (e.g. 512) while model_state_dict's pos_emb.weight stayed
    # the real, unshrunk 1024-size tensor — load_model() then rebuilds a 512-size model
    # from that false metadata and fails loading the real 1024-size weights into it.
    model_ctx_len = checkpoint["context_length"]
    seq_cap = model_ctx_len
    if sft_cfg.max_seq_len is not None:
        seq_cap = min(seq_cap, sft_cfg.max_seq_len)
    pad_id = tokenizer.eot_token

    train_examples = load_sft_examples(paths.data_dir / "sft" / "train.jsonl", tokenizer, seq_cap)
    test_examples = load_sft_examples(paths.data_dir / "sft" / "test.jsonl", tokenizer, seq_cap)
    if not train_examples:
        raise ValueError(
            f"No usable SFT training examples under {paths.data_dir / 'sft'} — "
            f"run `gpt-sft-data` first."
        )
    print(f"SFT examples: {len(train_examples):,} train  {len(test_examples):,} test")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sft_cfg.lr, weight_decay=sft_cfg.weight_decay,
    )

    batches_per_epoch = max(1, math.ceil(len(train_examples) / sft_cfg.batch_size))
    total_micro_steps = batches_per_epoch * sft_cfg.epochs
    total_optimizer_steps = max(1, math.ceil(total_micro_steps / sft_cfg.grad_accum_steps))
    print(
        f"SFT budget: {sft_cfg.epochs} epoch(s) x {batches_per_epoch:,} batches/epoch = "
        f"{total_micro_steps:,} microsteps  ({total_optimizer_steps:,} optimizer updates)"
    )

    start_step = 0
    best_test_loss = float("inf")
    total_training_seconds = 0.0
    if resuming:
        start_step = int(checkpoint.get("step", -1)) + 1
        best_test_loss = float(checkpoint.get("best_test_loss", float("inf")))
        total_training_seconds = float(checkpoint.get("total_training_seconds", 0.0))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed SFT at step {start_step} "
              f"(cumulative {format_duration(total_training_seconds)})")

    amp_device_type, amp_dtype = resolve_amp(sft_cfg.precision, device)
    run_start = time.time()

    def elapsed():
        return total_training_seconds + (time.time() - run_start)

    def payload(step, current_best):
        return make_payload(
            model=model, optimizer=optimizer, model_cfg=model_cfg, train_cfg=sft_cfg,
            context_length=model_ctx_len, step=step, best_test_loss=current_best,
            processed_tokens=checkpoint.get("processed_tokens", 0),
            total_training_seconds=elapsed(), label=label,
        )

    save_interval = max(1, sft_cfg.save_every_steps)
    progress = trange(total_micro_steps, desc="sft", unit="step", initial=start_step)
    optimizer.zero_grad(set_to_none=True)
    latest_metrics = None
    last_step = start_step - 1
    interrupted = False

    def shuffle_for_epoch(epoch):
        rng = torch.Generator().manual_seed(sft_cfg.seed + epoch)
        return torch.randperm(len(train_examples), generator=rng).tolist()

    # Computed up front (not only lazily at pos_in_epoch == 0 inside the loop) so
    # resuming mid-epoch — e.g. after a Ctrl-C — has a defined order for the first
    # iteration too, not just at a fresh epoch boundary.
    epoch_order = shuffle_for_epoch(start_step // batches_per_epoch)

    try:
        step = start_step
        while step < total_micro_steps:
            epoch = step // batches_per_epoch
            pos_in_epoch = step % batches_per_epoch
            if pos_in_epoch == 0 and step != start_step:
                epoch_order = shuffle_for_epoch(epoch)

            # Namespaced under this run's own checkpoint_dir, NOT paths.stop_file —
            # that property is a single file shared across every label, deliberately
            # (see its docstring), on the assumption only one gpt-train process is
            # ever running. Reusing it here would mean `make train-sft-stop` could
            # also kill an unrelated, concurrently-running base pretraining run.
            sft_stop_file = paths.checkpoint_dir / "STOP_TRAINING"
            if sft_stop_file.exists():
                sft_stop_file.unlink(missing_ok=True)
                print(f"\n{sft_stop_file} found — stopping gracefully...")
                interrupted = True
                break

            if step % sft_cfg.eval_interval == 0 or step == total_micro_steps - 1:
                test_loss = estimate_sft_loss(model, test_examples, sft_cfg, device, pad_id, seq_cap + 1)
                train_probe = estimate_sft_loss(model, train_examples, sft_cfg, device, pad_id, seq_cap + 1)
                improved = test_loss < best_test_loss
                if improved:
                    best_test_loss = test_loss
                    best = payload(step, best_test_loss)
                    atomic_save(best, paths.best_checkpoint)
                    atomic_save(best, paths.serving_checkpoint)
                append_eval_history(paths.eval_history, {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "epoch": f"{step / batches_per_epoch:.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.8e}",
                    "train_loss": f"{train_probe:.6f}",
                    "test_loss": f"{test_loss:.6f}",
                    "test_perplexity": f"{safe_perplexity(test_loss):.6f}",
                    "best_test_loss": f"{best_test_loss:.6f}",
                    "improved": int(improved),
                    "total_training_hours": f"{elapsed() / 3600.0:.4f}",
                })
                latest_metrics = {
                    "train_loss": f"{train_probe:.4f}", "test_loss": f"{test_loss:.4f}",
                }

            batch_idx = epoch_order[pos_in_epoch * sft_cfg.batch_size:
                                     (pos_in_epoch + 1) * sft_cfg.batch_size]
            x, y, attn_mask = make_sft_batch(train_examples, batch_idx, device, pad_id, fixed_len=seq_cap + 1)
            with torch.autocast(device_type=amp_device_type,
                                dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                logits = model(x, attn_mask=attn_mask)
                loss = masked_next_token_loss(logits, y, model.token_emb.num_embeddings)
            (loss / sft_cfg.grad_accum_steps).backward()

            # MPS-specific: unlike pretraining's fixed-shape random windows, SFT's
            # dynamic per-batch padding (make_sft_batch pads to each batch's own max
            # length, and this corpus ranges from 22 to 1024 tokens/example) means
            # nearly every step allocates a new tensor shape. MPS's caching allocator
            # does not coalesce/reclaim old-shape blocks the way CUDA's does, so
            # allocation grows essentially unbounded across shape-varying steps
            # (observed: ~23GB after 45 steps of a batch_size=2 153M-param model,
            # which should need a small fraction of that) until MPS OOMs. Releasing
            # the cache every step is the documented workaround for this MPS
            # allocator behavior; harmless no-op on cuda/cpu.
            if device == "mps":
                torch.mps.empty_cache()

            is_accum_boundary = (step + 1) % sft_cfg.grad_accum_steps == 0
            is_final_step = step == total_micro_steps - 1
            if is_accum_boundary or is_final_step:
                optimizer_update = (step + 1 + sft_cfg.grad_accum_steps - 1) // sft_cfg.grad_accum_steps
                for group in optimizer.param_groups:
                    group["lr"] = lr_for_step(optimizer_update - 1, sft_cfg, total_optimizer_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            postfix = {
                "epoch": f"{step / batches_per_epoch:.2f}",
                "batch_loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "total_h": f"{elapsed() / 3600.0:.2f}",
            }
            if latest_metrics:
                postfix.update(latest_metrics)
            progress.set_postfix(**postfix)
            progress.update(1)

            if (is_accum_boundary or is_final_step) and (
                (step + 1) % save_interval == 0 or is_final_step
            ):
                atomic_save(payload(step, best_test_loss), paths.latest_checkpoint)

            last_step = step
            step += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — saving a resumable checkpoint...")
    finally:
        progress.close()

    final_total_seconds = elapsed()
    final_step = last_step
    final_payload = make_payload(
        model=model, optimizer=optimizer, model_cfg=model_cfg, train_cfg=sft_cfg,
        context_length=model_ctx_len, step=final_step, best_test_loss=best_test_loss,
        processed_tokens=checkpoint.get("processed_tokens", 0),
        total_training_seconds=final_total_seconds, label=label,
    )
    atomic_save(final_payload, paths.latest_checkpoint)
    if not interrupted:
        atomic_save(final_payload, paths.final_checkpoint)
        if not paths.serving_checkpoint.exists():
            atomic_save(final_payload, paths.serving_checkpoint)
    print(f"Saved: {paths.latest_checkpoint}")
    print(f"Cumulative SFT time: {format_duration(final_total_seconds)}")

    if not interrupted:
        model.eval()
        prompt = f"User: {demo_prompt}\nAssistant:"
        _, completion = generate_text(
            model=model, tokenizer=tokenizer, prompt=prompt, context_length=model_ctx_len,
            max_new_tokens=sft_cfg.max_new_tokens, device=device, do_sample=True,
            temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1,
        )
        print(f"\nSample reply to {prompt!r}:")
        print(completion if completion.strip() else "[empty completion]")
        model.train()

    return {"interrupted": interrupted, "step": final_step, "best_test_loss": best_test_loss}
