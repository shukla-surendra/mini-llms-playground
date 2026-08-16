"""`gpt-infer` — generate from a trained checkpoint, from a prompt or the held-out set."""

import argparse

from ..checkpoint import load_model, select_checkpoint
from ..config import load_settings
from ..data import load_prompts
from ..inference import generate_text
from ..runtime import get_device


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--preset", default=None, help="Model size preset to load")
    parser.add_argument("--prompt", default=None,
                        help="Single prompt (default: use the held-out prompt file)")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max prompts to run from the prompt file")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None,
                        help="Which checkpoint to load (default: best, falling back to "
                             "latest/final). Use 'latest' to exercise the CURRENT training "
                             "state — best.pt only updates when test loss beats its prior "
                             "record, so it can lag the live model by a long way.")
    args = parser.parse_args()

    _, _, paths, label = load_settings(args.preset)
    device = get_device()
    checkpoint_path = select_checkpoint(paths, args.checkpoint)
    checkpoint, tokenizer, model = load_model(checkpoint_path, device)

    prompts = [args.prompt] if args.prompt else load_prompts(paths.test_prompts)[: args.limit]

    print(f"Model: {label} | checkpoint: {checkpoint_path} (step {checkpoint.get('step')})")
    print(f"Prompts: {len(prompts)}\n")

    for i, prompt in enumerate(prompts, start=1):
        _, completion = generate_text(
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
            postprocess=False,
        )
        print(f"[{i}] PROMPT:\n{prompt}\n")
        print(f"COMPLETION:\n{completion}")
        print("=" * 78)


if __name__ == "__main__":
    main()
