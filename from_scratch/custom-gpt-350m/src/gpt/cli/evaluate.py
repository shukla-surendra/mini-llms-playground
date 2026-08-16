"""`gpt-eval` — heuristic generation-quality report for a checkpoint."""

import argparse

import torch

from ..checkpoint import load_model, select_checkpoint
from ..config import load_settings
from ..data import DEFAULT_PROMPTS, load_prompts
from ..evaluation import evaluate, format_report
from ..evaluation.quality import append_report, load_last_report
from ..runtime import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation quality.")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the delta-vs-previous-run comparison")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None,
                        help="Which checkpoint to evaluate (default: best, falling back "
                             "to latest/final if best.pt doesn't exist yet). Use 'latest' "
                             "to see current training state when best.pt has gone stale "
                             "relative to a recent, non-regressive change (e.g. a corpus "
                             "rebuild) rather than a real regression.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _, _, paths, _ = load_settings(args.preset)
    device = get_device()
    checkpoint_path = select_checkpoint(paths, args.checkpoint)
    checkpoint, tokenizer, model = load_model(checkpoint_path, device)

    prompts = load_prompts(paths.test_prompts, default_prompts=DEFAULT_PROMPTS)[: args.max_prompts]

    report, examples = evaluate(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        checkpoint_path=checkpoint_path,
    )

    previous = None if args.no_compare else load_last_report(paths.quality_history)
    append_report(paths.quality_history, report)
    print(format_report(report, previous=previous, examples=examples))
    print(f"\nsaved_report: {paths.quality_history}")


if __name__ == "__main__":
    main()
