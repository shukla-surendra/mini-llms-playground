"""Export a native custom-gpt checkpoint as an exactly-equivalent HF GPT-2 model.

The training checkpoint remains the authoritative resumable artifact: it includes the
optimizer state, progress counters, and this project's own architecture metadata. This
command only reads that checkpoint and writes a separate, inference-only Hugging Face
directory that vLLM can load natively.
"""

import argparse
from pathlib import Path

import torch

from ..checkpoint import remap_attn_impl, select_checkpoint
from ..config import load_settings
from ..data.prepare import DOCUMENT_SEPARATOR
from ..config import TOKENIZER_NAME


def hf_state_dict(checkpoint):
    """Map this project's state keys to Hugging Face GPT2LMHeadModel keys.

    GPT-2's Conv1D layers store their weights as (input, output), while PyTorch
    Linear stores (output, input), hence the transposes below. All other tensors have
    the same layout. The exported config uses ``activation_function='gelu'`` (not
    GPT-2's default ``gelu_new``) to preserve this project's exact ``nn.GELU()``.
    """
    source = checkpoint["model_state_dict"]
    source_attn = checkpoint.get("attn_impl", "naive")
    if source_attn != "sdpa":
        source = remap_attn_impl(
            source,
            num_layers=checkpoint["num_layers"],
            from_impl=source_attn,
            to_impl="sdpa",
        )

    result = {
        "transformer.wte.weight": source["token_emb.weight"],
        "transformer.wpe.weight": source["pos_emb.weight"],
        "transformer.ln_f.weight": source["ln_f.weight"],
        "transformer.ln_f.bias": source["ln_f.bias"],
        # The two modules deliberately share this storage in both implementations.
        "lm_head.weight": source["lm_head.weight"],
    }
    for layer in range(checkpoint["num_layers"]):
        src = f"blocks.{layer}."
        dst = f"transformer.h.{layer}."
        result.update({
            f"{dst}ln_1.weight": source[f"{src}ln_1.weight"],
            f"{dst}ln_1.bias": source[f"{src}ln_1.bias"],
            f"{dst}attn.c_attn.weight": source[f"{src}attn.in_proj.weight"].T,
            f"{dst}attn.c_attn.bias": source[f"{src}attn.in_proj.bias"],
            f"{dst}attn.c_proj.weight": source[f"{src}attn.out_proj.weight"].T,
            f"{dst}attn.c_proj.bias": source[f"{src}attn.out_proj.bias"],
            f"{dst}ln_2.weight": source[f"{src}ln_2.weight"],
            f"{dst}ln_2.bias": source[f"{src}ln_2.bias"],
            f"{dst}mlp.c_fc.weight": source[f"{src}mlp.net.0.weight"].T,
            f"{dst}mlp.c_fc.bias": source[f"{src}mlp.net.0.bias"],
            f"{dst}mlp.c_proj.weight": source[f"{src}mlp.net.2.weight"].T,
            f"{dst}mlp.c_proj.bias": source[f"{src}mlp.net.2.bias"],
        })
    return result


def _load_export_dependencies():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "vLLM export dependencies are not installed. Run `uv sync --extra vllm` "
            "from this project first."
        ) from exc
    return GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


def _gpt2_config(checkpoint, GPT2Config):
    return GPT2Config(
        vocab_size=checkpoint["vocab_size"],
        n_positions=checkpoint["context_length"],
        n_ctx=checkpoint["context_length"],
        n_embd=checkpoint["embed_size"],
        n_layer=checkpoint["num_layers"],
        n_head=checkpoint["num_heads"],
        n_inner=4 * checkpoint["embed_size"],
        activation_function="gelu",
        resid_pdrop=checkpoint.get("dropout", 0.0),
        embd_pdrop=checkpoint.get("dropout", 0.0),
        attn_pdrop=checkpoint.get("dropout", 0.0),
        layer_norm_epsilon=1e-5,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        use_cache=True,
    )


@torch.inference_mode()
def _verify_equivalence(source_path, model, tokenizer):
    """Fail export if a fixed prompt's logits are not numerically equivalent."""
    from ..checkpoint import load_model

    # Recreate the native model without writing anything. Its public loader is also
    # the compatibility path used by native inference, including naive->SDPA remaps.
    _, _, native = load_model(source_path, "cpu")
    native.eval()
    model.eval()
    ids = tokenizer(
        "vLLM export parity check" + DOCUMENT_SEPARATOR,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    native_logits = native(ids)
    hf_logits = model(ids).logits
    if not torch.allclose(native_logits, hf_logits, rtol=1e-4, atol=1e-5):
        delta = (native_logits - hf_logits).abs().max().item()
        raise RuntimeError(
            "HF export failed numerical parity verification; maximum logit difference "
            f"was {delta:.6g}. No export was written."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export a native checkpoint to an equivalent Hugging Face GPT-2 directory for vLLM."
    )
    parser.add_argument("--preset", default=None, help="Model size preset to export")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None,
                        help="Checkpoint to export (default: best, then latest/final)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New output directory (default includes selected checkpoint and step)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip native-vs-HF logit parity verification")
    args = parser.parse_args()

    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast = _load_export_dependencies()
    _, _, paths, label = load_settings(args.preset)
    source_path = select_checkpoint(paths, args.checkpoint)
    checkpoint = torch.load(source_path, map_location="cpu")
    step = int(checkpoint.get("step", -1))
    kind = args.checkpoint or "serving"
    output_dir = args.output_dir or Path("exports") / "vllm" / label / f"{kind}-step-{step}"
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Export directories are immutable; choose --output-dir "
            "or remove this known generated artifact deliberately."
        )

    config = _gpt2_config(checkpoint, GPT2Config)
    model = GPT2LMHeadModel(config)
    missing, unexpected = model.load_state_dict(hf_state_dict(checkpoint), strict=False)
    # GPT-2 ties lm_head to wte; depending on Transformers version, either name can
    # be omitted after tie_weights(). Anything else would mean an invalid conversion.
    allowed_missing = {"lm_head.weight"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Unexpected export key mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # This project uses a plain two-newline separator, so verify the exported HF
    # tokenizer assigns exactly the same IDs as the native tiktoken tokenizer.
    import tiktoken
    probe = "vLLM tokenizer parity" + DOCUMENT_SEPARATOR
    native_ids = tiktoken.get_encoding(checkpoint.get("tokenizer", TOKENIZER_NAME)).encode_ordinary(probe)
    hf_ids = tokenizer.encode(probe, add_special_tokens=False)
    if hf_ids != native_ids:
        raise RuntimeError("The exported GPT-2 tokenizer does not match this project's tiktoken IDs.")
    if not args.no_verify:
        _verify_equivalence(source_path, model, tokenizer)

    output_dir.mkdir(parents=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Exported checkpoint step {step:,} to {output_dir}")
    print(f"Serve with: vllm serve {output_dir}")


if __name__ == "__main__":
    main()
