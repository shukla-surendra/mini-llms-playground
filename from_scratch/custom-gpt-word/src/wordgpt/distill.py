"""Generate synthetic training text from a locally-hosted teacher model.

This is sequence-level distillation - the only kind viable for this project. Logit
distillation would need the teacher's token probability distribution over a *shared*
vocabulary, but the teacher's subword tokenizer and this project's ~4,500-word capped
vocabulary don't correspond at all; only the teacher's raw text output transfers. See
`../../../docs/LLM_AS_JUDGE_AND_DISTILLATION.md` for the full reasoning and the legal
notes on which teacher models are safe to distill from (locally-hosted open-weight
models via Ollama - not commercial API output, which provider terms generally prohibit
using to train another model).

Requires a local Ollama server with the teacher pulled:
    ollama serve &
    ollama pull gemma3:4b
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "distilled"

# Short, topically matched prompts - the bundled corpus is self-help/psychology
# nonfiction (see data/corpus.txt's provenance), so generated text should extend that
# distribution rather than dilute it with an unrelated register.
DEFAULT_PROMPTS = [
    "Write a short, plainly-written paragraph explaining why people procrastinate, in the style of a self-help book.",
    "Write a short paragraph giving one practical tip for managing anxiety before a difficult conversation.",
    "Write a short paragraph explaining the difference between confidence and arrogance.",
    "Write a short paragraph about why small daily habits matter more than big one-time efforts.",
    "Write a short paragraph explaining active listening to someone unfamiliar with the term.",
    "Write a short paragraph about how to give constructive feedback without discouraging someone.",
    "Write a short paragraph explaining what cognitive reframing means, with an everyday example.",
    "Write a short paragraph about the difference between guilt and shame.",
    "Write a short paragraph explaining why setting boundaries is an act of self-respect, not selfishness.",
    "Write a short paragraph giving advice on how to recover from a mistake at work.",
]


SYSTEM_PROMPT = (
    "Output only the requested paragraph itself. No preamble ('Okay, here's...'), no "
    "quotation marks around it, no closing offer to revise or elaborate. Plain prose only."
)


def query_teacher(model: str, prompt: str, temperature: float = 0.8) -> str:
    payload = json.dumps({
        "model": model, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))["response"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small synthetic corpus from a local Ollama teacher.")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument(
        "--count", type=int, default=len(DEFAULT_PROMPTS),
        help="How many passages to generate (cycles through DEFAULT_PROMPTS if higher).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    text_path = OUTPUT_DIR / f"{args.model.replace(':', '-')}_{stamp}.txt"
    provenance_path = text_path.with_suffix(".json")

    passages = []
    for i in range(args.count):
        prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
        print(f"[{i + 1}/{args.count}] {prompt[:60]}...")
        passages.append(query_teacher(args.model, prompt))

    text_path.write_text("\n\n".join(passages), encoding="utf-8")
    # Provenance recorded now, not reconstructed later - the license obligations that
    # matter (notice, pass-through) attach to distribution, and you can't reconstruct
    # which teacher/version generated what after the fact. See the doc's Part 3.
    provenance_path.write_text(
        json.dumps(
            {
                "teacher_model": args.model,
                "generated_at_utc": stamp,
                "passage_count": args.count,
                "prompts_used": DEFAULT_PROMPTS[: args.count] if args.count <= len(DEFAULT_PROMPTS) else "cycled",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {len(passages)} passages -> {text_path}")
    print(f"provenance -> {provenance_path}")
    print(
        "To use: append this file's text to data/corpus.txt and rerun `make train` "
        "(vocab_size and param count will shift slightly - rerun `make config` after)."
    )


if __name__ == "__main__":
    main()
