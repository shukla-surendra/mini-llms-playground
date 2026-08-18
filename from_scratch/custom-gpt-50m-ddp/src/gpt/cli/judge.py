"""`gpt-judge` — grade open-ended answers with a local LLM via Ollama.

Fills the one gap `gpt-score` cannot: prompts with no checkable answer ("give two tips
for sleeping better"). Full rationale, reliability caveats and the distillation/legal
discussion: `docs/LLM_AS_JUDGE_AND_DISTILLATION.md` at the repo root.

Two design choices that are not optional, both because a small judge has no stable
calibration:

* **Pairwise, not absolute.** "Which of these two is better" needs no calibration;
  "score this 1-5" drifts between prompts and between runs. It also matches the
  question you actually have — is checkpoint B better than A.
* **Both orders, every time.** Judges systematically favour the answer shown first, so
  each pair is asked twice with A and B swapped. A win counts only when the judge picks
  the same answer both times; disagreement is recorded as a tie *and* reported, because
  a high flip rate means the judge's opinion is noise rather than signal.

    ollama serve &
    ollama pull gemma3:4b
    gpt-judge --a best --b latest --cpu
    gpt-judge --a latest --b best --judge gemma3:12b --limit 20
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

import torch

from ..checkpoint import load_model, select_checkpoint
from ..config import load_settings
from ..inference import generate_text
from ..runtime import get_device
from .qa_graders import GRADED_PROMPTS
from .qa_prompts import QA_CATEGORIES

OLLAMA_URL = "http://localhost:11434/api/chat"

RUBRIC = """You are comparing two assistant answers to the same user question.

Judge on, in order of importance:
1. Correctness — is it factually right and internally consistent?
2. Relevance — does it actually answer the question asked?
3. Coherence — is it well-formed and readable?

Ignore length. A short correct answer beats a long rambling one.
Both answers may be poor; pick the less poor one, or "tie" if genuinely equal.

Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
}

#: Prompts with a mechanically checkable answer belong to `gpt-score`, not here — a
#: 4B judge is unreliable at correctness and would add noise to a question already
#: answered exactly. Judge only what nothing else can grade.
_CHECKABLE = {prompt for _, prompt, _ in GRADED_PROMPTS}


def open_ended_prompts(limit=None):
    out = []
    for _, questions in QA_CATEGORIES:
        out.extend(q for q in questions if q not in _CHECKABLE)
    return out[:limit] if limit else out


def ask_judge(model, question, answer_a, answer_b, timeout=120):
    """One comparison. Returns 'A' | 'B' | 'tie' | None (on failure)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content":
                f"Question:\n{question}\n\n--- Answer A ---\n{answer_a}\n\n"
                f"--- Answer B ---\n{answer_b}"},
        ],
        "stream": False,
        "format": JSON_SCHEMA,          # structured output; unparsed replies become ties
        "options": {"temperature": 0},  # a judge should be deterministic
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return json.loads(body["message"]["content"]).get("winner")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL} ({exc}).\n"
            f"Start it with `ollama serve &` and pull the judge with "
            f"`ollama pull {model}`.")
    except (KeyError, ValueError, json.JSONDecodeError):
        return None      # unparseable reply — counted as a tie, and reported


def answers_for(checkpoint_name, prompts, paths, device, args):
    path = select_checkpoint(paths, checkpoint_name)
    checkpoint, tokenizer, model = load_model(path, device)
    out = []
    for i, q in enumerate(prompts, 1):
        _, a = generate_text(
            model=model, tokenizer=tokenizer, prompt=f"User: {q}\nAssistant:",
            context_length=checkpoint["context_length"],
            max_new_tokens=args.max_new_tokens, device=device,
            do_sample=False, temperature=1.0, top_k=None, top_p=None,
            repetition_penalty=1.1, postprocess=True)
        out.append(a)
        print(f"\r  generating {checkpoint_name}: {i}/{len(prompts)}", end="", flush=True)
    print()
    del model
    return path, checkpoint.get("step", 0), out


def main():
    p = argparse.ArgumentParser(description="Pairwise-judge two checkpoints with a local LLM.")
    p.add_argument("--preset", default=None)
    p.add_argument("--a", default="best", choices=["best", "latest", "final"])
    p.add_argument("--b", default="latest", choices=["best", "latest", "final"])
    p.add_argument("--judge", default="gemma3:4b", help="Ollama model tag")
    p.add_argument("--cpu", action="store_true", help="Force CPU for the student models")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--limit", type=int, default=None, help="Only judge the first N prompts")
    args = p.parse_args()

    if args.a == args.b:
        raise SystemExit("--a and --b must be different checkpoints.")

    _, _, paths, label = load_settings(args.preset)
    device = "cpu" if args.cpu else get_device()
    prompts = open_ended_prompts(args.limit)

    print(f"Judge: {args.judge} via Ollama  |  student device={device}")
    print(f"Comparing {args.a} vs {args.b} on {len(prompts)} open-ended prompts "
          f"({len(prompts) * 2} judge calls, both orders)\n")

    path_a, step_a, answers_a = answers_for(args.a, prompts, paths, device, args)
    path_b, step_b, answers_b = answers_for(args.b, prompts, paths, device, args)

    tally = Counter()
    flips = 0
    for i, (q, a, b) in enumerate(zip(prompts, answers_a, answers_b), 1):
        first = ask_judge(args.judge, q, a, b)          # A shown first
        second = ask_judge(args.judge, q, b, a)         # B shown first
        # `second` is in swapped coordinates: "A" there means our B.
        second_mapped = {"A": "B", "B": "A", "tie": "tie"}.get(second)
        if first == second_mapped and first in ("A", "B"):
            tally[first] += 1
        else:
            tally["tie"] += 1
            if first in ("A", "B") and second_mapped in ("A", "B"):
                flips += 1
        print(f"\r  judging: {i}/{len(prompts)}", end="", flush=True)
    print()

    n = len(prompts)
    print(f"\n  {args.a} (step {step_a:,})  wins: {tally['A']:>3}  ({100*tally['A']/n:.0f}%)")
    print(f"  {args.b} (step {step_b:,})  wins: {tally['B']:>3}  ({100*tally['B']/n:.0f}%)")
    print(f"  ties / inconsistent      : {tally['tie']:>3}  ({100*tally['tie']/n:.0f}%)")
    print(f"\n  position-bias flips      : {flips}/{n} ({100*flips/n:.0f}%) — the judge "
          f"changed its mind when the order swapped.")
    if flips > n * 0.3:
        print("  ** over 30%: this judge is not discriminating between these answers.")
        print("     Try a larger judge (gemma3:12b, qwen2.5:14b) before believing the result.")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": args.judge, "a": str(path_a), "b": str(path_b),
        "step_a": step_a, "step_b": step_b, "prompts": n,
        "wins_a": tally["A"], "wins_b": tally["B"], "ties": tally["tie"],
        "position_flips": flips,
    }
    out = paths.log_dir / f"judge_history_{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n  appended: {out}")


if __name__ == "__main__":
    main()
