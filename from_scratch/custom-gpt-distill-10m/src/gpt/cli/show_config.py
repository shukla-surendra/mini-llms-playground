"""Print the architecture, parameter count, and (if a corpus already exists) token/epoch math."""

from ..config import load_settings


def main() -> None:
    model_cfg, train_cfg, paths = load_settings()
    print(model_cfg.describe())

    if paths.train_data.exists() and paths.test_data.exists():
        import tiktoken

        tok = tiktoken.get_encoding("gpt2")
        train_tokens = len(tok.encode_ordinary(paths.train_data.read_text(encoding="utf-8")))
        test_tokens = len(tok.encode_ordinary(paths.test_data.read_text(encoding="utf-8")))
        tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum_steps * model_cfg.context_length
        epochs = (train_cfg.steps * tokens_per_step) / max(1, train_tokens)
        total_training_tokens = train_cfg.steps * tokens_per_step
        ratio = total_training_tokens / model_cfg.param_count()
        print(f"\ncorpus: train={train_tokens:,} tokens  test={test_tokens:,} tokens")
        print(f"tokens/optimizer-step: {tokens_per_step:,}")
        print(f"at steps={train_cfg.steps:,}: ~{epochs:.1f} epochs over train split, "
              f"{total_training_tokens:,} training tokens consumed "
              f"({ratio:.1f} tokens/param; Chinchilla-optimal is ~20)")
    else:
        print(f"\nno corpus yet at {paths.corpus_dir}/ - run `make distill` first.")


if __name__ == "__main__":
    main()
