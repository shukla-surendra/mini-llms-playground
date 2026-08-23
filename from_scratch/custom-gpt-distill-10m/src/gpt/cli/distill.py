"""Generate synthetic training text from a locally-hosted Ollama teacher (sequence-level
distillation - the only kind viable here, since the teacher's subword tokenizer and
this project's GPT-2 BPE vocabulary don't correspond at logit level; only the teacher's
raw text output transfers). See ../../../docs/LLM_AS_JUDGE_AND_DISTILLATION.md for the
full reasoning and the legal notes on which teacher models are safe to distill from.

Produces instruction/response pairs in a fixed "User: ...\\nAssistant: ..." format
rather than free prose - per that doc's own recommendation, reinforcing one exact
format teaches it more reliably than diluting it across an unstructured corpus.

Requires a local Ollama server with the teacher pulled:
    ollama serve &
    ollama pull gemma3:4b

Each run APPENDS to data/corpus/train.txt and test.txt, so the corpus can be built up
across multiple sessions rather than needing one long blocking call.
"""

import argparse
import json
import random
import urllib.request
from datetime import datetime, timezone

from ..config import load_settings

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "Answer the user's instruction directly and concisely, in 2-4 sentences. No "
    "preamble ('Okay, here's...', 'Sure!'), no follow-up offers to elaborate or "
    "revise, no markdown formatting or headers. Plain prose only."
)

# Cross product gives topic-count * template-count distinct instructions without
# needing hundreds of hand-written prompts - all narrow, everyday-advice topics, per
# the distillation doc's own guidance to spend generated tokens on a format/register
# the corpus is actually weak in, not general prose the teacher would produce anyway.
TOPICS = [
    "time management", "healthy eating", "public speaking", "saving money",
    "learning a new skill", "managing stress", "improving focus", "building a habit",
    "giving feedback", "resolving a disagreement", "cooking a simple meal",
    "basic gardening", "home organization", "job interviews", "email etiquette",
    "sleep habits", "starting to exercise", "reading more", "note-taking",
    "setting goals", "staying motivated", "active listening", "budgeting",
    "decluttering", "morning routines", "remote work", "networking",
    "public transit etiquette", "recycling at home", "basic first aid",
]

INSTRUCTION_TEMPLATES = [
    "Give me one practical tip for {topic}.",
    "Explain {topic} to a complete beginner in a couple of sentences.",
    "What's a common mistake people make with {topic}, and how can they avoid it?",
    "Write a short, encouraging message about {topic}.",
    "What's one small first step someone could take toward {topic}?",
]


def build_prompts():
    prompts = [tmpl.format(topic=topic) for topic in TOPICS for tmpl in INSTRUCTION_TEMPLATES]
    random.Random(42).shuffle(prompts)
    return prompts


def query_teacher(model, instruction, temperature=0.8):
    payload = json.dumps({
        "model": model, "prompt": instruction, "system": SYSTEM_PROMPT, "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))["response"].strip()


def format_block(instruction, response):
    return f"User: {instruction}\nAssistant: {response}"


def append_corpus(path, blocks):
    text = "\n\n".join(blocks)
    if path.exists() and path.read_text(encoding="utf-8").strip():
        path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n\n" + text + "\n", encoding="utf-8")
    else:
        path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    _, _, paths = load_settings()
    parser = argparse.ArgumentParser(description="Generate an instruction/response corpus from a local Ollama teacher.")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--count", type=int, default=200, help="How many instruction/response pairs to generate this run.")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    prompts = build_prompts()
    if args.count > len(prompts):
        laps = (args.count // len(prompts)) + 1
        # Different shuffle per lap so repeats of the same base prompt list aren't
        # generated back-to-back in identical order.
        prompts = [p for lap in range(laps) for p in build_prompts()][: args.count]
    else:
        prompts = prompts[: args.count]

    paths.distilled_dir.mkdir(parents=True, exist_ok=True)
    paths.corpus_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pairs = []
    for i, instruction in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {instruction}")
        response = query_teacher(args.model, instruction)
        pairs.append((instruction, response))

    raw_path = paths.distilled_dir / f"{args.model.replace(':', '-')}_{stamp}.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for instruction, response in pairs:
            f.write(json.dumps({"instruction": instruction, "response": response}) + "\n")
    raw_path.with_suffix(".provenance.json").write_text(
        json.dumps({"teacher_model": args.model, "generated_at_utc": stamp, "pair_count": len(pairs)}, indent=2),
        encoding="utf-8",
    )

    split_idx = max(1, int(len(pairs) * (1 - args.test_fraction)))
    train_pairs, test_pairs = pairs[:split_idx], pairs[split_idx:] or pairs[-1:]
    append_corpus(paths.train_data, [format_block(i, r) for i, r in train_pairs])
    append_corpus(paths.test_data, [format_block(i, r) for i, r in test_pairs])

    print(f"\nwrote {len(train_pairs)} train + {len(test_pairs)} test pairs (appended)")
    print(f"raw batch -> {raw_path}")
    print(f"corpus -> {paths.train_data}, {paths.test_data}")
    print("Run `make config` to see updated token counts, then `make train`.")


if __name__ == "__main__":
    main()
