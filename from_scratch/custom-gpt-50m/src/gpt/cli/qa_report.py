"""`gpt-qa-report` — run the curated prompt set through a checkpoint and render HTML.

The qualitative companion to `gpt-eval`'s structural heuristic score: it does not grade
correctness, it gives you the actual generated answers to read — the way
`docs/llm-engineering/15_evaluating_a_model_while_training.md`'s Signal #4 ("does the
output actually sound better") is meant to be judged.

Two halves:

* **Prompt set** — ~130 questions across *corpus-mirroring* categories (regression
  checks against each training source) and *capability probes* (reasoning, coding,
  agentic planning, practical life, safety, self-knowledge). See `qa_prompts.py` for
  why both exist, and why failure on the probes is expected rather than a bug.
* **Parameter sweep** — a few prompts re-asked under greedy/conservative/default/
  creative decoding, so you can separate what the *model* believes from what the
  *sampler* happened to roll.

    gpt-qa-report --checkpoint latest      # current training state, not a stale best.pt
    gpt-qa-report --cpu                    # leave the GPU to a running trainer
    gpt-qa-report --per-category-limit 2   # quick look
    gpt-qa-report --no-sweep               # skip the parameter sweep
"""

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from ..checkpoint import load_model, select_checkpoint
from ..config import load_settings
from ..data.sources import DATASETS
from ..inference import generate_text
from ..runtime import get_device
from .qa_prompts import (PROBE_CATEGORIES, QA_CATEGORIES, SWEEP_PROMPTS,
                         SWEEP_SETTINGS)

REPORT_TZ = ZoneInfo("Asia/Kolkata")

# Categories mirroring a registered chat source are skipped when that source is absent
# from data/raw/, so a report never claims to test a dataset this checkpoint never saw
# (relevant for the gated LMSYS set). Capability probes and extra-document categories
# are intentionally unmapped — they always run.
_CATEGORY_SOURCE_HF_ID = {
    "UltraChat-style (bulk everyday-assistant Q&A)": "HuggingFaceH4/ultrachat_200k",
    "OASST1-style (human-phrased, sometimes messier)": "OpenAssistant/oasst1",
    "Dolly-style (one prompt per documented task type)": "zidankhan/databricks-dolly-15k",
    "SmolTalk-style (dialogue / reasoning / rewriting / summarization)": "HuggingFaceTB/smoltalk",
    "No Robots-style (one prompt per documented task type, entirely human-written)":
        "HuggingFaceH4/no_robots",
    "GSM8K-style (grade-school math — the arithmetic-gap regression check)": "openai/gsm8k",
}

def _is_probe(category):
    return category in PROBE_CATEGORIES


def _active_categories(data_dir):
    sources_by_id = {ds.hf_id: ds for ds in DATASETS}
    active, skipped = [], []
    for category, questions in QA_CATEGORIES:
        hf_id = _CATEGORY_SOURCE_HF_ID.get(category)
        if hf_id is None:
            active.append((category, questions))
            continue
        if (data_dir / "raw" / sources_by_id[hf_id].slug).exists():
            active.append((category, questions))
        else:
            skipped.append(category)
    return active, skipped


CSS = """
:root {
  --bg:#f6f7f9; --card:#fff; --ink:#12151a; --muted:#5a6472; --line:#e3e7ec;
  --q-bg:#eef4ff; --q-ink:#12305c; --q-edge:#3b6fd4;
  --a-bg:#fbfbfc; --a-ink:#1c2128; --a-edge:#b9c0ca;
  --probe:#a2540d; --probe-bg:#fff6e8; --mirror:#0f6b4f; --mirror-bg:#e9f7f1;
  --accent:#3b6fd4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0f1216; --card:#161a20; --ink:#e6e9ee; --muted:#98a2b3; --line:#262c35;
    --q-bg:#152441; --q-ink:#cfe0ff; --q-edge:#5d8ae6;
    --a-bg:#12161c; --a-ink:#d6dbe3; --a-edge:#39414d;
    --probe:#f0b366; --probe-bg:#2a1f10; --mirror:#7fd8b6; --mirror-bg:#0f2620;
    --accent:#5d8ae6;
  }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:920px; margin:0 auto; padding:2rem 1.25rem 5rem; }
h1 { font-size:1.5rem; margin:0 0 .35rem; letter-spacing:-.01em; }
.clock { font-size:1.02rem; font-weight:650; color:var(--accent); margin:.1rem 0 1.2rem; }
.meta { background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:1rem 1.15rem; margin-bottom:1.5rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.6rem 1.4rem; }
.kv { font-size:.88rem; } .kv b { display:block; color:var(--muted); font-weight:600;
  text-transform:uppercase; letter-spacing:.045em; font-size:.68rem; margin-bottom:.14rem; }
code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.86em; }
.chips { margin-top:.9rem; padding-top:.85rem; border-top:1px solid var(--line); }
.chip { display:inline-block; background:var(--bg); border:1px solid var(--line);
  border-radius:999px; padding:.16rem .62rem; margin:.16rem .3rem .16rem 0; font-size:.78rem; }
.toc { background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:.9rem 1.15rem; margin-bottom:2.2rem; font-size:.88rem; }
.toc a { color:var(--accent); text-decoration:none; display:inline-block; margin:.18rem .7rem .18rem 0; }
.toc a:hover { text-decoration:underline; }
section { margin:0 0 2.8rem; }
h2 { font-size:1.06rem; margin:0 0 .3rem; display:flex; align-items:center;
  gap:.6rem; flex-wrap:wrap; scroll-margin-top:1rem; }
.tag-kind { font-size:.66rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
  padding:.17rem .5rem; border-radius:5px; }
.kind-probe { color:var(--probe); background:var(--probe-bg); }
.kind-mirror { color:var(--mirror); background:var(--mirror-bg); }
.note { font-size:.83rem; color:var(--muted); margin:.1rem 0 1.1rem; }
.qa { background:var(--card); border:1px solid var(--line); border-radius:12px;
  margin:0 0 1.15rem; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.045); }
.q { background:var(--q-bg); color:var(--q-ink); border-left:4px solid var(--q-edge);
  padding:.75rem 1rem; font-weight:640; white-space:pre-wrap; }
.a { background:var(--a-bg); color:var(--a-ink); border-left:4px solid var(--a-edge);
  padding:.85rem 1rem; white-space:pre-wrap; }
.lbl { display:inline-block; min-width:1.4rem; font-weight:800; font-size:.74rem;
  opacity:.6; letter-spacing:.06em; }
.empty { color:var(--muted); font-style:italic; }
.sweep { background:var(--card); border:1px solid var(--line); border-radius:12px;
  margin:0 0 1.4rem; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.045); }
.sweep .q { border-left-color:var(--accent); }
.setting { border-top:1px solid var(--line); padding:.75rem 1rem; }
.setting .name { font-size:.72rem; font-weight:750; letter-spacing:.05em;
  text-transform:uppercase; color:var(--muted); margin-bottom:.32rem; }
.setting .out { white-space:pre-wrap; }
footer { color:var(--muted); font-size:.82rem; border-top:1px solid var(--line);
  padding-top:1rem; margin-top:2rem; }
"""


def _slug(text):
    return "".join(c if c.isalnum() else "-" for c in text.lower())[:48].strip("-")


def _qa_block(question, answer):
    body = html.escape(answer) if answer.strip() else '<span class="empty">(empty)</span>'
    return (f'<div class="qa"><div class="q"><span class="lbl">Q</span>{html.escape(question)}</div>'
            f'<div class="a"><span class="lbl">A</span>{body}</div></div>')


def _build_html(*, label, param_count, checkpoint_path, step, configured_steps,
                best_test_loss, settings, results, sweep, skipped, generated_utc):
    ist = generated_utc.astimezone(REPORT_TZ)
    pct = f"{100.0 * step / configured_steps:.1f}%" if configured_steps else "n/a"
    loss = f"{best_test_loss:.4f}" if best_test_loss is not None else "n/a"

    toc = " ".join(f'<a href="#{_slug(c)}">{html.escape(c.split(" (")[0])}</a>'
                   for c, _ in results)
    chips = "".join(f'<span class="chip">{html.escape(k)} = <code>{html.escape(str(v))}</code></span>'
                    for k, v in settings.items())

    sections = []
    for category, items in results:
        probe = _is_probe(category)
        kind = ('<span class="tag-kind kind-probe">capability probe</span>' if probe
                else '<span class="tag-kind kind-mirror">corpus regression</span>')
        note = ("Asks for behaviour the training corpus does <em>not</em> contain — wrong "
                "answers here are the expected gap, not a regression."
                if probe else
                "Mirrors a training source; a change here means the corpus or the model moved.")
        rows = "".join(_qa_block(q, a) for q, a in items)
        sections.append(
            f'<section id="{_slug(category)}"><h2>{html.escape(category)} {kind}</h2>'
            f'<p class="note">{note} &middot; {len(items)} prompts</p>{rows}</section>')

    sweep_html = ""
    if sweep:
        blocks = []
        for question, variants in sweep:
            outs = "".join(
                f'<div class="setting"><div class="name">{html.escape(name)}</div>'
                f'<div class="out">{html.escape(out) if out.strip() else "(empty)"}</div></div>'
                for name, out in variants)
            blocks.append(f'<div class="sweep"><div class="q"><span class="lbl">Q</span>'
                          f'{html.escape(question)}</div>{outs}</div>')
        sweep_html = (
            '<section id="parameter-sweep"><h2>Parameter sweep '
            '<span class="tag-kind kind-mirror">same prompt, different decoding</span></h2>'
            '<p class="note">The weights are identical in every row below — only the sampler '
            'changes, from the same seed. Differences here are the decoder, not the model. '
            'Greedy is deterministic, so it shows what the model most believes rather than '
            'what it happened to roll.</p>' + "".join(blocks) + '</section>')

    skipped_html = (f'<div class="kv"><b>skipped</b><code>{html.escape(", ".join(skipped))}</code></div>'
                    if skipped else "")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(label)} QA report — step {step}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{html.escape(label)} — QA report</h1>
<div class="clock">{ist:%A, %d %B %Y} &middot; {ist:%I:%M %p} IST</div>
<div class="meta"><div class="grid">
  <div class="kv"><b>checkpoint</b><code>{html.escape(Path(checkpoint_path).name)}</code></div>
  <div class="kv"><b>step</b>{step:,} / {configured_steps:,} ({pct})</div>
  <div class="kv"><b>parameters</b>{param_count:,}</div>
  <div class="kv"><b>best test loss</b>{loss}</div>
  <div class="kv"><b>prompts</b>{sum(len(i) for _, i in results):,}</div>
  <div class="kv"><b>generated (UTC)</b>{generated_utc:%Y-%m-%d %H:%M}</div>
  {skipped_html}
</div><div class="chips">{chips}</div></div>
<div class="toc"><b>Jump to:</b><br>{toc}
{'<a href="#parameter-sweep">Parameter sweep</a>' if sweep else ''}</div>
{"".join(sections)}{sweep_html}
<footer>Generated by <code>gpt-qa-report</code>. Answers are raw model output — this
report does not grade correctness, it shows what the model actually says.</footer>
</div></body></html>
"""


def main():
    p = argparse.ArgumentParser(description="Run the QA prompt set and render an HTML report.")
    p.add_argument("--preset", default=None, help="Model size preset to load")
    p.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None,
                   help="Which checkpoint to use (default: best, falling back to "
                        "latest/final). Use 'latest' to see current training state when "
                        "best.pt has gone stale.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU. Use this while a trainer holds the GPU — inference on "
                        "the same device measurably slows training.")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Generation token budget. Completions are trimmed back to the "
                        "last full sentence, so a bigger budget leaves less on the floor.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--greedy", action="store_true", help="Disable sampling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-category-limit", type=int, default=None,
                   help="Cap prompts per category, for a faster subset run")
    p.add_argument("--no-sweep", action="store_true", help="Skip the parameter sweep")
    p.add_argument("--out", default=None,
                   help="Output HTML path (default: reports/qa_report_<label>_step<N>.html)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    _, train_cfg, paths, label = load_settings(args.preset)
    device = "cpu" if args.cpu else get_device()
    checkpoint_path = select_checkpoint(paths, args.checkpoint)
    checkpoint, tokenizer, model = load_model(checkpoint_path, device)
    step = checkpoint.get("step", 0)

    settings = {
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_k": args.top_k, "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "sampling": not args.greedy, "seed": args.seed, "device": device,
    }

    categories, skipped = _active_categories(paths.data_dir)
    if args.per_category_limit:
        categories = [(c, q[: args.per_category_limit]) for c, q in categories]

    total = sum(len(q) for _, q in categories)
    sweep_n = 0 if args.no_sweep else len(SWEEP_PROMPTS) * len(SWEEP_SETTINGS)
    print(f"Model: {label} | {checkpoint_path} (step {step:,}) | device={device}")
    print(f"Running {total} prompts across {len(categories)} categories"
          f"{f' + {sweep_n} sweep generations' if sweep_n else ''}...")
    if skipped:
        print(f"Skipped (source not in {paths.data_dir / 'raw'}): {', '.join(skipped)}")

    def ask(question, **overrides):
        opts = dict(do_sample=not args.greedy, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty)
        opts.update(overrides)
        _, answer = generate_text(
            model=model, tokenizer=tokenizer, prompt=f"User: {question}\nAssistant:",
            context_length=checkpoint["context_length"],
            max_new_tokens=args.max_new_tokens, device=device, postprocess=True, **opts)
        return answer

    results, done = [], 0
    for category, questions in categories:
        answered = []
        for q in questions:
            answered.append((q, ask(q)))
            done += 1
            print(f"\r  [{done}/{total}] {category[:44]:<44}", end="", flush=True)
        results.append((category, answered))
    print()

    sweep = []
    if not args.no_sweep:
        for q in SWEEP_PROMPTS:
            variants = []
            for name, kw in SWEEP_SETTINGS:
                # Same seed for every setting, so the only variable is the sampler.
                torch.manual_seed(args.seed)
                variants.append((name, ask(q, **kw)))
            sweep.append((q, variants))
            print(f"  sweep: {q[:52]}")

    out_path = Path(args.out) if args.out else Path("reports") / f"qa_report_{label}_step{step}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_html(
        label=label, param_count=sum(t.numel() for t in model.parameters()),
        checkpoint_path=checkpoint_path, step=step, configured_steps=train_cfg.steps,
        best_test_loss=checkpoint.get("best_test_loss"), settings=settings,
        results=results, sweep=sweep, skipped=skipped,
        generated_utc=datetime.now(timezone.utc)), encoding="utf-8")
    print(f"saved_report: {out_path}")


if __name__ == "__main__":
    main()
