"""`gpt-score` — an objective, comparable score for a checkpoint.

Answers "is this checkpoint better than that one?" with numbers rather than reading.

Three sections, deliberately separate because they fail for different reasons:

* **accuracy** — generated answers marked against a right answer (facts, arithmetic,
  reasoning). Uses **greedy decoding by default**, because a sampled answer makes the
  score a function of the seed rather than of the model.
* **constraints** — mechanical instruction-following ("exactly one word", "under 20
  words", "do not mention France"). Format obedience is a distinct capability from
  knowing things, and it moves separately.
* **multiple choice** — no generation at all: each candidate continuation is scored by
  average token log-probability and the argmax wins. Deterministic, immune to sampling,
  and the standard way base models are compared. Chance level is printed alongside so
  a number is interpretable.

Every run appends to `logs/score_history_<label>.jsonl`, so improvement across
checkpoints is a diff rather than a memory.

    gpt-score --checkpoint latest --cpu
    gpt-score --checkpoint best --compare      # show the change since the last run
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..checkpoint import load_model, select_checkpoint
from ..config import load_settings
from ..inference import generate_text
from ..runtime import get_device
from .qa_graders import GRADED_PROMPTS, MC_ITEMS


@torch.no_grad()
def continuation_logprob(model, tokenizer, context, continuation, device, context_length):
    """Average per-token log-probability of `continuation` given `context`.

    Averaged, not summed: a summed score systematically prefers the shortest option,
    which would make the benchmark measure string length instead of plausibility.
    """
    ctx_ids = tokenizer.encode(context, allowed_special=set(), disallowed_special=())
    cont_ids = tokenizer.encode(continuation, allowed_special=set(), disallowed_special=())
    ids = (ctx_ids + cont_ids)[-context_length:]
    if len(cont_ids) == 0 or len(ids) < 2:
        return float("-inf")
    x = torch.tensor([ids[:-1]], device=device)
    logits = model(x)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    targets = ids[1:]
    # Only the continuation's tokens are scored; the context is shared by all options.
    start = max(0, len(targets) - len(cont_ids))
    total = sum(logprobs[i, targets[i]].item() for i in range(start, len(targets)))
    return total / (len(targets) - start)


def run_generative(model, tokenizer, checkpoint, device, args):
    buckets = defaultdict(lambda: {"pass": 0, "total": 0})
    details = []
    for category, prompt, grader in GRADED_PROMPTS:
        _, answer = generate_text(
            model=model, tokenizer=tokenizer, prompt=f"User: {prompt}\nAssistant:",
            context_length=checkpoint["context_length"], max_new_tokens=args.max_new_tokens,
            device=device, do_sample=not args.greedy, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, repetition_penalty=1.1, postprocess=True)
        ok, why = grader(answer)
        buckets[category]["total"] += 1
        buckets[category]["pass"] += int(ok)
        details.append((category, prompt, answer, ok, why))
    return buckets, details


def run_multiple_choice(model, tokenizer, device, context_length):
    correct = 0
    details = []
    for context, choices, gold in MC_ITEMS:
        scores = [continuation_logprob(model, tokenizer, context, c, device, context_length)
                  for c in choices]
        pick = max(range(len(choices)), key=lambda i: scores[i])
        correct += int(pick == gold)
        details.append((context, choices, gold, pick, scores))
    chance = sum(1.0 / len(c) for _, c, _ in MC_ITEMS) / len(MC_ITEMS)
    return correct / len(MC_ITEMS), chance, details


def main():
    p = argparse.ArgumentParser(description="Objectively score a checkpoint.")
    p.add_argument("--preset", default=None)
    p.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU — use while a trainer holds the GPU")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--sample", dest="greedy", action="store_false",
                   help="Sample instead of greedy. Greedy is the default here so the "
                        "score reflects the model rather than the seed.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--verbose", action="store_true", help="Show every item and why it failed")
    p.add_argument("--compare", action="store_true", help="Diff against the previous run")
    p.set_defaults(greedy=True)
    args = p.parse_args()

    _, _, paths, label = load_settings(args.preset)
    device = "cpu" if args.cpu else get_device()
    checkpoint_path = select_checkpoint(paths, args.checkpoint)
    checkpoint, tokenizer, model = load_model(checkpoint_path, device)
    step = checkpoint.get("step", 0)

    print(f"Model: {label} | {checkpoint_path} (step {step:,}) | device={device} | "
          f"decoding={'greedy' if args.greedy else 'sampled'}")
    print(f"Scoring {len(GRADED_PROMPTS)} graded prompts + {len(MC_ITEMS)} multiple-choice...\n")

    buckets, details = run_generative(model, tokenizer, checkpoint, device, args)
    mc_acc, mc_chance, mc_details = run_multiple_choice(
        model, tokenizer, device, checkpoint["context_length"])

    gen_pass = sum(b["pass"] for b in buckets.values())
    gen_total = sum(b["total"] for b in buckets.values())

    print(f"  {'section':<14} {'score':>9}   {'chance':>7}")
    print(f"  {'-'*14} {'-'*9}   {'-'*7}")
    for cat in ("facts", "arithmetic", "reasoning", "constraints"):
        if cat not in buckets:
            continue
        b = buckets[cat]
        print(f"  {cat:<14} {b['pass']:>3}/{b['total']:<3} {100*b['pass']/b['total']:>4.0f}%       —")
    print(f"  {'multiple-choice':<14} {int(mc_acc*len(MC_ITEMS)):>3}/{len(MC_ITEMS):<3} "
          f"{100*mc_acc:>4.0f}%   {100*mc_chance:>5.0f}%")
    print(f"  {'-'*14} {'-'*9}   {'-'*7}")
    print(f"  {'GENERATIVE':<14} {gen_pass:>3}/{gen_total:<3} {100*gen_pass/gen_total:>4.0f}%")

    if args.verbose:
        print("\n  per-item:")
        for cat, prompt, answer, ok, why in details:
            print(f"   [{'PASS' if ok else 'FAIL'}] {cat:<12} {prompt[:52]:<52} {why}")
            if not ok:
                print(f"          -> {answer[:100]!r}")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path), "step": step,
        "best_test_loss": checkpoint.get("best_test_loss"),
        "decoding": "greedy" if args.greedy else "sampled",
        "generative_pass": gen_pass, "generative_total": gen_total,
        "by_category": {k: dict(v) for k, v in buckets.items()},
        "multiple_choice_acc": mc_acc, "multiple_choice_chance": mc_chance,
    }
    history = paths.log_dir / f"score_history_{label}.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)

    if args.compare and history.exists():
        prev = [json.loads(l) for l in history.read_text().splitlines() if l.strip()]
        if prev:
            last = prev[-1]
            d_gen = (gen_pass / gen_total) - (last["generative_pass"] / last["generative_total"])
            d_mc = mc_acc - last["multiple_choice_acc"]
            print(f"\n  vs step {last['step']:,}:  generative {d_gen*100:+.1f} pts   "
                  f"multiple-choice {d_mc*100:+.1f} pts")

    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n  appended: {history}")


if __name__ == "__main__":
    main()
