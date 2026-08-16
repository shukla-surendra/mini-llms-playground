"""`gpt-infer` — generate from a trained checkpoint."""

import argparse

from ..config import Paths, resolve_model_config
from ..inference.generate import generate, load_model_and_tokenizer
from ..model import detect_device


def main():
    p = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    p.add_argument("--preset", default=None, help="Model size preset (used only to resolve the default checkpoint path)")
    p.add_argument("--checkpoint", default=None, help="Defaults to checkpoints/<label>/causal/best.pt")
    p.add_argument("--prompt", default="Once upon a time,")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    args = p.parse_args()

    _, label = resolve_model_config(args.preset)
    checkpoint_path = args.checkpoint or str(Paths(label=label, objective="causal").best_checkpoint)

    device = detect_device()
    model, tokenizer, ckpt = load_model_and_tokenizer(checkpoint_path, device)
    print(f"[model] loaded step={ckpt.get('step')} params={model.num_parameters():,} device={device}")

    text = generate(
        model, tokenizer, args.prompt,
        ctx_len=ckpt["context_length"],
        max_new_tokens=args.max_new_tokens,
        device=device,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print("\n--- Generated ---")
    print(text)


if __name__ == "__main__":
    main()
