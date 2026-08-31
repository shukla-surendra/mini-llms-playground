"""`gpt-benchmark` — measure real training throughput, then price the run.

Answers the question you actually have before renting a GPU: *given this model and
this instance, how long and how much for 3B / 5B / 10B tokens?*

Protocol (the reason this is not just "time a few steps"):

    0 .. warmup      discarded   CUDA context, cuDNN autotune, allocator growth,
                                 and clock ramp all land here; including them
                                 understates steady-state throughput badly.
    warmup .. end    measured    steady state -> tokens/sec -> tokens/GPU-day

Checkpointing and eval are deliberately not run — they are real costs, but they are
*configurable* costs, and mixing them into a hardware measurement makes the number
untransferable. `--eval-overhead` adds them back analytically at the projection step.

    gpt-benchmark                              # 10 min warmup + 50 min measure
    gpt-benchmark --warmup-min 0.2 --measure-min 1     # quick local sanity check
    gpt-benchmark --sweep-batch 4,8,16,24,32   # find the batch that fits and is fastest
    gpt-benchmark --price-per-hour 0.8048      # override the instance price
"""

import argparse
import time

import torch

from ..config import load_settings, resolve_model_config
from ..data import get_batch, load_token_array, next_token_loss
from ..model import TinyGPT
from ..runtime import get_device
from ..training.trainer import resolve_amp

# Dense (non-sparsity) peak throughput for the precision we actually train in.
# Only used to express MFU; a missing entry just omits that line.
PEAK_TFLOPS = {
    "NVIDIA L4": 121.0,        # Ada, bf16 tensor core
    "NVIDIA A10G": 125.0,      # Ampere, bf16
    "Tesla T4": 65.0,          # Turing, fp16 (no bf16)
    "NVIDIA A100": 312.0,      # Ampere, bf16
    "NVIDIA H100": 989.0,      # Hopper, bf16
}

# On-demand us-east-1, approximate — override with --price-per-hour.
INSTANCE_PRICES = {
    "NVIDIA L4": ("g6.xlarge", 0.8048),
    "NVIDIA A10G": ("g5.xlarge", 1.006),
    "Tesla T4": ("g4dn.xlarge", 0.526),
}

# 4.92e9 is this project's own configured budget (150_000 steps x 16 x 2048, see
# config.TrainConfig) — not the 153m sibling's 2.46e9 (1024 context).
TOKEN_BUDGETS = [1e9, 2e9, 3e9, 4.92e9, 5e9, 10e9]


def flops_per_token(model_cfg, n_params):
    """(dense_matmul_flops, attention_flops) for one token, forward+backward.

    `6 * N` is the standard approximation everyone quotes for MFU. It ignores
    attention's score/context matmuls, which do not scale with parameter count but
    do scale with context length — at 1024 they are a real double-digit share, so
    they are reported separately rather than silently dropped or silently folded in.
    """
    dense = 6.0 * n_params
    attn = 12.0 * model_cfg.num_layers * model_cfg.context_length * model_cfg.embed_size
    return dense, attn


def human_time(hours):
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def run_window(model, optimizer, tokens, ctx_len, batch_size, vocab_size, device,
               amp_device_type, amp_dtype, grad_accum, seconds, label):
    """Train for `seconds` wall-clock. Returns (steps, tokens, elapsed)."""
    torch.manual_seed(1234)
    steps = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    deadline = start + seconds
    while time.perf_counter() < deadline:
        xb, yb = get_batch(tokens, ctx_len, batch_size, device)
        with torch.autocast(device_type=amp_device_type,
                            dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            loss = next_token_loss(model(xb), yb, vocab_size)
        (loss / grad_accum).backward()
        if (steps + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        steps += 1
        if steps % 20 == 0:
            done = time.perf_counter() - start
            print(f"\r  {label}: {done / seconds * 100:5.1f}%  "
                  f"{steps} steps", end="", flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"\r  {label}: done — {steps} steps in {elapsed:.1f}s" + " " * 20)
    return steps, steps * batch_size * ctx_len, elapsed


def benchmark_one(model_cfg, train_cfg, tokens, batch_size, args, device):
    amp_device_type, amp_dtype = resolve_amp(args.precision or train_cfg.precision, device)
    ctx_len = model_cfg.context_length

    model = TinyGPT.from_config(model_cfg, context_length=ctx_len, attn_impl="sdpa").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    common = dict(model=model, optimizer=optimizer, tokens=tokens, ctx_len=ctx_len,
                  batch_size=batch_size, vocab_size=model_cfg.vocab_size, device=device,
                  amp_device_type=amp_device_type, amp_dtype=amp_dtype,
                  grad_accum=train_cfg.grad_accum_steps)

    if args.warmup_min > 0:
        run_window(seconds=args.warmup_min * 60, label="warm-up", **common)
    steps, toks, elapsed = run_window(seconds=args.measure_min * 60,
                                      label="measure", **common)

    peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if device.startswith("cuda") else None
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "steps_per_sec": steps / elapsed,
        "tokens_per_sec": toks / elapsed,
        "peak_gib": peak_gib,
        "amp_dtype": amp_dtype,
    }


def main():
    p = argparse.ArgumentParser(description="Measure training throughput and price the run.")
    p.add_argument("--preset", default=None)
    p.add_argument("--warmup-min", type=float, default=10.0,
                   help="Minutes discarded before measuring (default 10)")
    p.add_argument("--measure-min", type=float, default=50.0,
                   help="Minutes of steady-state measurement (default 50)")
    p.add_argument("--batch-size", type=int, default=None, help="Override TrainConfig")
    p.add_argument("--sweep-batch", default=None,
                   help="Comma-separated batch sizes to compare, e.g. 8,16,24,32")
    p.add_argument("--precision", default=None, help="auto|bf16|fp16|fp32")
    p.add_argument("--price-per-hour", type=float, default=None)
    p.add_argument("--eval-overhead", type=float, default=5.0,
                   help="Percent to add for eval+checkpoint cost (default 5)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use random tokens instead of the corpus (throughput is "
                        "independent of token values, so this lets you benchmark "
                        "before the corpus exists)")
    args = p.parse_args()

    model_cfg, train_cfg, paths, label = load_settings(args.preset)
    device = get_device()
    n_params = model_cfg.param_count()

    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    total_gib = (torch.cuda.get_device_properties(0).total_memory / 2**30
                 if device.startswith("cuda") else None)

    # Data: real corpus if tokenized, else synthetic. Only the *shape* matters here.
    if args.synthetic or not paths.train_data.with_suffix(".bin").exists():
        import numpy as np
        need = max(model_cfg.context_length * 64, 1_000_000)
        tokens = np.random.randint(0, model_cfg.vocab_size, size=need, dtype=np.uint16)
        data_note = f"synthetic ({need:,} random tokens)"
    else:
        tokens = load_token_array(paths.train_data)
        data_note = f"{paths.train_data.with_suffix('.bin')} ({len(tokens):,} tokens)"

    print(f"\n=== gpt-benchmark: {label} ===")
    print(f"  device      : {device}" + (f"  ({gpu_name}, {total_gib:.1f} GiB)" if gpu_name else ""))
    print(f"  model       : {n_params:,} params  (E={model_cfg.embed_size} "
          f"L={model_cfg.num_layers} ctx={model_cfg.context_length})")
    print(f"  data        : {data_note}")
    print(f"  grad_accum  : {train_cfg.grad_accum_steps}")
    print(f"  window      : {args.warmup_min:g} min warm-up + {args.measure_min:g} min measured\n")

    batches = ([int(b) for b in args.sweep_batch.split(",")] if args.sweep_batch
               else [args.batch_size or train_cfg.batch_size])
    results = []
    for bs in batches:
        try:
            results.append(benchmark_one(model_cfg, train_cfg, tokens, bs, args, device))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  batch_size={bs}: OUT OF MEMORY — skipped")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not results:
        print("No batch size fit in memory.")
        return

    dense, attn = flops_per_token(model_cfg, n_params)
    peak = PEAK_TFLOPS.get(gpu_name)

    print("\n--- throughput ---")
    hdr = f"  {'batch':>6} {'steps/s':>9} {'tokens/s':>11} {'tok/GPU-day':>13} {'peak VRAM':>11}"
    if peak:
        hdr += f" {'MFU':>7}"
    print(hdr)
    for r in results:
        vram = f"{r['peak_gib']:.1f} GiB" if r["peak_gib"] is not None else "n/a"
        line = (f"  {r['batch_size']:>6} {r['steps_per_sec']:>9.2f} "
                f"{r['tokens_per_sec']:>11,.0f} {r['tokens_per_sec'] * 86400 / 1e9:>12.2f}B "
                f"{vram:>11}")
        if peak:
            line += f" {100 * r['tokens_per_sec'] * dense / (peak * 1e12):>6.1f}%"
        print(line)
    if peak:
        print(f"    MFU uses the standard 6N approximation; attention adds a further "
              f"{100 * attn / dense:.0f}% of FLOPs at ctx={model_cfg.context_length}")

    best = max(results, key=lambda r: r["tokens_per_sec"])
    tps = best["tokens_per_sec"] * (1 - args.eval_overhead / 100.0)

    price = args.price_per_hour
    inst = None
    if price is None and gpu_name in INSTANCE_PRICES:
        inst, price = INSTANCE_PRICES[gpu_name]

    print(f"\n--- projections (batch {best['batch_size']}, "
          f"{args.eval_overhead:g}% eval/checkpoint overhead applied) ---")
    head = f"  {'budget':>9} {'tok/param':>10} {'GPU-hours':>10} {'wall clock':>12}"
    if price:
        head += f" {'cost':>9}"
    print(head + (f"      @ ${price}/hr" + (f" ({inst})" if inst else "") if price else ""))
    for b in TOKEN_BUDGETS:
        hrs = b / tps / 3600
        row = (f"  {b / 1e9:>8.2f}B {b / n_params:>10.1f} {hrs:>10.1f} "
               f"{human_time(hrs):>12}")
        if price:
            row += f" {'$' + format(hrs * price, ',.2f'):>9}"
        print(row)

    print("\n--- what fits in a fixed budget ---")
    print(f"  {'hours':>6} {'tokens':>11} {'this model':>13} {'Chinchilla-opt. size':>21}")
    for h in (6, 12, 24, 48):
        toks = tps * h * 3600
        # Throughput scales ~1/N, so a differently-sized model gets a different token
        # count in the same hours. Holding compute C = 6*N*D fixed and imposing
        # D = 20N gives N = sqrt(C/120) — the size this budget is actually optimal for.
        compute = dense * toks           # 6*N*D for THIS model over that many tokens
        n_opt = (compute / 120.0) ** 0.5
        print(f"  {h:>6} {toks / 1e9:>10.2f}B {toks / n_params:>12.1f}x "
              f"{n_opt / 1e6:>20.0f}M")
    print(f"  ('this model' = tokens per parameter at {n_params / 1e6:.1f}M; "
          f"Chinchilla-optimal is 20x.")
    print("   A number above 20 means the budget could support a larger model.)")


if __name__ == "__main__":
    main()
