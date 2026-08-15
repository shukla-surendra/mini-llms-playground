"""`gpt-qa-report` — run a curated Q&A prompt set through the current checkpoint and
render the results as a self-contained HTML report, grouped by which training-corpus
source each prompt's style is drawn from (see docs/DATASETS.md).

This is a qualitative companion to `gpt-eval`'s structural heuristic score (non-empty,
non-repetitive, no role leakage) — it doesn't score correctness, it gives you the actual
generated answers to read, the way docs/llm-engineering/15_evaluating_a_model_while_
training.md's Signal #4 ("does the output actually sound better") is meant to be judged.
"""

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..checkpoint import load_model, select_checkpoint
from ..config import load_settings
from ..data.sources import DATASETS
from ..inference import generate_text
from ..runtime import get_device

# Prompt count per category is weighted by that source's role in the training mix
# (docs/DATASETS.md): UltraChat/OASST1/LMSYS are the bulk conversational volume, so
# they get the most coverage; Dolly is small but hand-written across seven distinct
# task types, so it gets one prompt per documented type instead of a volume-based
# count; SmolTalk is a compact mixture, so a handful spanning its stated sub-domains.
QA_CATEGORIES = [
    ("UltraChat-style (bulk everyday-assistant Q&A)", [
        "What are three simple ways to stay productive while working from home?",
        "Can you explain what a black hole is in simple terms?",
        "I'm planning a trip to Japan. What should I pack?",
        "What's a good beginner recipe for homemade pizza?",
        "How can I improve my public speaking skills?",
        "What are the benefits of regular exercise?",
    ]),
    ("OASST1-style (human-phrased, sometimes messier)", [
        "whats the difference between a virus and bacteria? also which one antibiotics work on",
        "Can you help me write a short apology message to a friend I forgot to call back?",
        "why is the sky blue? explain like im five",
        "whats a good way to learn a new language fast, ive tried apps before but they didnt stick",
        "can u give me tips for a job interview tmrw im nervous",
    ]),
    ("Dolly-style (one prompt per documented task type)", [
        "Brainstorm five names for a small coffee shop.",                                    # brainstorming
        "Classify the following as a fruit or a vegetable: tomato, carrot, apple, spinach.",  # classification
        "What is the capital of France?",                                                     # closed QA
        "Why do leaves change color in autumn?",                                              # open QA
        "Write a short two-sentence story about a lost dog finding its way home.",            # generation
        "Summarize in one sentence: The Great Wall of China is a series of fortifications "
        "built across the historical northern borders of China to protect against invasions.",  # summarization
        "Extract the person's name and age from this sentence: \"John Smith, aged 34, "
        "joined the company last year.\"",                                                    # extraction
    ]),
    ("SmolTalk-style (dialogue / reasoning / rewriting / summarization)", [
        "What's your favorite way to spend a weekend?",
        "If a train leaves at 3pm and travels for 2 hours 45 minutes, what time does it arrive?",
        "Rewrite this sentence to sound more formal: \"hey can u send me that file asap\"",
        "Summarize this in one sentence: Regular sleep, a balanced diet, and daily exercise "
        "are the three pillars most doctors recommend for maintaining long-term health.",
        "What's something interesting you'd want to learn more about?",
    ]),
    ("LMSYS-style (casual, unfiltered real-user phrasing)", [
        "yo whats a good movie to watch tonight im bored",
        "fix this code its not working: def add(a,b) return a+b",
        "can you write me a poem about cats idk make it funny",
        "whats 15% tip on a $42 bill",
        "explain quantum computing but like im dumb",
        "give me a random fun fact",
    ]),
    ("Format-following (does it respect the role boundary?)", [
        "What's your name?",
        "Can you help me?",
    ]),
]

# Maps each category above to the dataset registry entry it's modeled on, so a
# category can be skipped automatically if that source wasn't actually part of the
# corpus this checkpoint was trained on (e.g. gated LMSYS skipped by `make data-public`).
_CATEGORY_SOURCE_HF_ID = {
    "UltraChat-style (bulk everyday-assistant Q&A)": "HuggingFaceH4/ultrachat_200k",
    "OASST1-style (human-phrased, sometimes messier)": "OpenAssistant/oasst1",
    "Dolly-style (one prompt per documented task type)": "zidankhan/databricks-dolly-15k",
    "SmolTalk-style (dialogue / reasoning / rewriting / summarization)": "HuggingFaceTB/smoltalk",
    "LMSYS-style (casual, unfiltered real-user phrasing)": "lmsys/lmsys-chat-1m",
}


def _active_categories(data_dir):
    """QA_CATEGORIES filtered to sources actually present in data/raw/ — so a report
    never claims to test a dataset flavor that wasn't in this checkpoint's training
    corpus (relevant for the gated LMSYS-Chat-1M set, skipped by `make data-public`)."""
    sources_by_id = {ds.hf_id: ds for ds in DATASETS}
    active, skipped = [], []
    for category, questions in QA_CATEGORIES:
        hf_id = _CATEGORY_SOURCE_HF_ID.get(category)
        if hf_id is None:
            active.append((category, questions))
            continue
        raw_dir = data_dir / "raw" / sources_by_id[hf_id].slug
        if raw_dir.exists():
            active.append((category, questions))
        else:
            skipped.append(category)
    return active, skipped


def _build_html(*, label, param_count, checkpoint_path, step, configured_steps,
                 best_test_loss, generation_settings, results, generated_at, skipped):
    progress_pct = (100.0 * step / configured_steps) if configured_steps else None
    progress_str = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
    best_loss_str = f"{best_test_loss:.4f}" if best_test_loss is not None else "n/a"

    settings_str = " &nbsp;·&nbsp; ".join(
        f"{k}={v}" for k, v in generation_settings.items()
    )
    skipped_html = (
        f"<br>skipped (source not in this checkpoint's training corpus): "
        f"<code>{html.escape(', '.join(skipped))}</code>"
        if skipped else ""
    )

    sections = []
    for category, items in results:
        rows = []
        for question, answer in items:
            rows.append(f"""
      <div class="qa">
        <div class="q"><span class="tag">Q</span>{html.escape(question)}</div>
        <div class="a"><span class="tag tag-a">A</span>{html.escape(answer) or '<em>(empty)</em>'}</div>
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
<title>custom-gpt-153m QA report — {html.escape(label)} step {step}</title>
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
  <h1>custom-gpt-153m — QA performance report</h1>
  <div class="meta">
    model label: <code>{html.escape(label)}</code> &nbsp;·&nbsp;
    parameters: <code>{param_count:,}</code><br>
    checkpoint: <code>{html.escape(str(checkpoint_path))}</code>
    (step <code>{step}</code> / <code>{configured_steps:,}</code> configured &mdash; {progress_str}
    of the training budget) &nbsp;·&nbsp; best_test_loss: <code>{best_loss_str}</code><br>
    generation: {settings_str}<br>
    generated: {html.escape(generated_at)}{skipped_html}
  </div>
  {"".join(sections)}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run a curated Q&A prompt set and render an HTML report."
    )
    parser.add_argument("--preset", default=None, help="Model size preset to load")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                         help="Generation token budget. Completions are trimmed back "
                              "to the last full sentence (see postprocess_completion), "
                              "so a bigger budget leaves less on the floor.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category-limit", type=int, default=None,
                         help="Cap prompts per category, for a faster subset run "
                              "(default: run every prompt in every active category)")
    parser.add_argument("--out", default=None,
                         help="Output HTML path (default: reports/qa_report_<label>_step<N>.html)")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None,
                         help="Which checkpoint to run (default: best, falling back to "
                              "latest/final if best.pt doesn't exist yet). Use 'latest' "
                              "to see current training state when best.pt has gone stale "
                              "relative to a recent, non-regressive change (e.g. a corpus "
                              "rebuild) rather than a real regression.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    _, train_cfg, paths, label = load_settings(args.preset)
    device = get_device()
    checkpoint_path = select_checkpoint(paths, args.checkpoint)
    checkpoint, tokenizer, model = load_model(checkpoint_path, device)
    step = checkpoint.get("step", 0)
    param_count = sum(p.numel() for p in model.parameters())

    generation_settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "sampling": not args.greedy,
    }

    categories, skipped = _active_categories(paths.data_dir)
    if args.per_category_limit:
        categories = [(c, qs[: args.per_category_limit]) for c, qs in categories]

    print(f"Model: {label} | checkpoint: {checkpoint_path} (step {step})")
    total_prompts = sum(len(items) for _, items in categories)
    print(f"Running {total_prompts} prompts across {len(categories)} categories...")
    if skipped:
        print(f"Skipped (source not found under {paths.data_dir / 'raw'}): {', '.join(skipped)}")
    print()

    results = []
    for category, questions in categories:
        answered = []
        for question in questions:
            prompt = f"User: {question}\nAssistant:"
            _, answer = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                context_length=checkpoint["context_length"],
                max_new_tokens=args.max_new_tokens,
                device=device,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                postprocess=True,
            )
            print(f"[{category}]\n  Q: {question}\n  A: {answer}\n")
            answered.append((question, answer))
        results.append((category, answered))

    out_path = Path(args.out) if args.out else Path("reports") / f"qa_report_{label}_step{step}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_html(
        label=label,
        param_count=param_count,
        checkpoint_path=checkpoint_path,
        step=step,
        configured_steps=train_cfg.steps,
        best_test_loss=checkpoint.get("best_test_loss"),
        generation_settings=generation_settings,
        results=results,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        skipped=skipped,
    ), encoding="utf-8")

    print(f"saved_report: {out_path}")


if __name__ == "__main__":
    main()
