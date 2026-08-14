"""`make test` — run a curated set of story-opening prompts through the current
checkpoint and render the completions as a self-contained HTML report.

This project has no chat format to test (see docs/DATASET_AND_TOKENIZER.md — it trains
on plain TinyStories continuation, not User:/Assistant: turns), so the sibling
custom-gpt-10m project's Q&A report doesn't apply here. The natural analog is a
story-completion report: varied openers spanning the character types, settings, and
plot themes TinyStories itself is built from, each shown as Prompt -> Completion.

Qualitative companion to reading logs/train_eval_history.csv directly — the way
docs/llm-engineering/15_evaluating_a_model_while_training.md's Signal #4 ("does the
output actually sound better") is meant to be judged: by reading real generations, not
just watching the loss number.
"""

import argparse
import html
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from inference import generate, load_model_and_tokenizer
from model import detect_device

# Grouped by TinyStories' own recurring building blocks (docs/DATASET_AND_TOKENIZER.md:
# short, simple stories with a deliberately restricted vocabulary) rather than by
# training-data source, since this project has only one dataset — the categories below
# are story archetypes, not corpus provenance.
STORY_PROMPTS = [
    ("Animal protagonists", [
        "Once upon a time, there was a little rabbit named Pip who loved to",
        "There was a brown dog named Max. Every morning he would",
        "In the forest, a small bird named Tweety wanted to",
        "A little cat named Whiskers found a",
        "One sunny day, a turtle named Shelly decided to",
    ]),
    ("Human protagonists", [
        "Once upon a time, there was a little girl named Lily. She loved to",
        "There was a boy named Tom who lived near a big",
        "Once, a little girl named Mia found a shiny",
        "There was a young boy named Sam who wanted to",
        "Once upon a time, a girl named Ella went to the park and",
    ]),
    ("Common TinyStories themes (losing/finding, fear, friendship, weather, birthdays)", [
        "One day, Lily lost her favorite toy and felt very",
        "The little boy was scared of the dark until",
        "Two friends, Max and Lily, decided to build a",
        "It started to rain, so the children ran to",
        "It was Tom's birthday, and his mom baked a",
    ]),
    ("Minimal-context openers (coherence with less scaffolding)", [
        "Once upon a time,",
        "One day,",
        "There was a little",
    ]),
]


def _build_html(*, label, param_count, checkpoint_path, step, configured_steps,
                 best_val_loss, generation_settings, results, generated_at):
    progress_pct = (100.0 * step / configured_steps) if configured_steps else None
    progress_str = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
    best_loss_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "n/a"

    settings_str = " &nbsp;·&nbsp; ".join(
        f"{k}={v}" for k, v in generation_settings.items()
    )

    sections = []
    for category, items in results:
        rows = []
        for prompt, completion in items:
            rows.append(f"""
      <div class="qa">
        <div class="q"><span class="tag">Prompt</span>{html.escape(prompt)}</div>
        <div class="a"><span class="tag tag-a">Completion</span>{html.escape(completion) or '<em>(empty)</em>'}</div>
      </div>""")
        sections.append(f"""
    <section>
      <h2>{html.escape(category)}</h2>
      {"".join(rows)}
    </section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>tinystories-gpt-6m story report — {html.escape(label)} step {step}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
          max-width: 860px; margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #555; font-size: 0.85rem; line-height: 1.6; margin-bottom: 2rem; }}
  .meta code {{ background: #f2f2f2; padding: 0.1rem 0.35rem; border-radius: 3px; }}
  section {{ margin-bottom: 2rem; }}
  h2 {{ font-size: 1.05rem; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }}
  .qa {{ margin: 1rem 0; padding: 0.75rem 1rem; background: #fafafa;
         border-left: 3px solid #ccc; border-radius: 4px; }}
  .q {{ font-weight: 600; margin-bottom: 0.4rem; }}
  .a {{ white-space: pre-wrap; color: #222; }}
  .tag {{ display: inline-block; font-size: 0.7rem; font-weight: 700; color: #fff;
          background: #666; border-radius: 3px; padding: 0.05rem 0.4rem;
          margin-right: 0.5rem; vertical-align: 1px; }}
  .tag-a {{ background: #2b6cb0; }}
</style>
</head>
<body>
  <h1>tinystories-gpt-6m — story completion report</h1>
  <div class="meta">
    model label: <code>{html.escape(label)}</code> &nbsp;·&nbsp;
    parameters: <code>{param_count:,}</code><br>
    checkpoint: <code>{html.escape(str(checkpoint_path))}</code>
    (step <code>{step}</code> / <code>{configured_steps:,}</code> configured &mdash; {progress_str}
    of the training budget) &nbsp;·&nbsp; best_val_loss: <code>{best_loss_str}</code><br>
    generation: {settings_str}<br>
    generated: {html.escape(generated_at)}
  </div>
  {"".join(sections)}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run a curated story-opening prompt set and render an HTML report."
    )
    parser.add_argument("--checkpoint", default="tinystories_gpt_checkpoint.pt")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category-limit", type=int, default=None,
                         help="Cap prompts per category, for a faster subset run "
                              "(default: run every prompt in every category)")
    parser.add_argument("--out", default=None,
                         help="Output HTML path (default: reports/story_report_step<N>.html)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = detect_device()
    checkpoint_path = args.checkpoint
    model, tokenizer, ckpt = load_model_and_tokenizer(checkpoint_path, device)
    step = ckpt.get("step", 0)
    param_count = model.num_parameters()
    configured_steps = int(os.getenv("STEPS", 5000))

    generation_settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "sampling": not args.greedy,
    }

    categories = STORY_PROMPTS
    if args.per_category_limit:
        categories = [(c, ps[: args.per_category_limit]) for c, ps in categories]

    print(f"Model: tinystories-gpt-6m | checkpoint: {checkpoint_path} (step {step}) | device: {device}")
    total_prompts = sum(len(items) for _, items in categories)
    print(f"Running {total_prompts} prompts across {len(categories)} categories...\n")

    results = []
    for category, prompts in categories:
        answered = []
        for prompt in prompts:
            text = generate(
                model, tokenizer, prompt,
                ctx_len=ckpt["context_length"],
                max_new_tokens=args.max_new_tokens,
                device=device,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            completion = text[len(prompt):] if text.startswith(prompt) else text
            print(f"[{category}]\n  Prompt: {prompt}\n  Completion: {completion}\n")
            answered.append((prompt, completion))
        results.append((category, answered))

    out_path = Path(args.out) if args.out else Path("reports") / f"story_report_step{step}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_html(
        label="tinystories-gpt-6m",
        param_count=param_count,
        checkpoint_path=checkpoint_path,
        step=step,
        configured_steps=configured_steps,
        best_val_loss=ckpt.get("best_val_loss"),
        generation_settings=generation_settings,
        results=results,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ), encoding="utf-8")

    print(f"saved_report: {out_path}")


if __name__ == "__main__":
    main()
