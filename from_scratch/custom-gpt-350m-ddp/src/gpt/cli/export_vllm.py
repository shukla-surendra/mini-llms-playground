"""Export a native custom-gpt checkpoint as an exactly-equivalent HF Llama model.

This project's architecture (RMSNorm + RoPE + SwiGLU + no biases, see model.py's
module docstring) is not GPT-2-shaped like the sibling custom-gpt-{10m,50m,153m}
projects — it maps onto Hugging Face's LlamaForCausalLM instead. The tokenizer needs
no conversion at all: tokenizer/tokenizer.json is already a native `tokenizers`-library
file, which is exactly the format transformers.PreTrainedTokenizerFast loads directly.

The training checkpoint remains the authoritative resumable artifact: it includes the
optimizer state, progress counters, and this project's own architecture metadata. This
command only reads that checkpoint and writes a separate, inference-only Hugging Face
directory that vLLM can load natively (LlamaForCausalLM is one of vLLM's most mature,
best-optimized supported architectures).
"""

import argparse
from pathlib import Path

import torch

from ..checkpoint import select_checkpoint
from ..config import TOKENIZER_PATH, load_settings
from ..tokenizer import END_OF_TEXT


def hf_state_dict(checkpoint):
    """Map this project's state keys to Hugging Face LlamaForCausalLM keys.

    No transposes anywhere: both this project's Linear layers and HF Llama's are
    plain nn.Linear (out_features, in_features) — unlike the GPT-2/Conv1D export on
    the sibling projects, which needs .T on every attention/MLP weight. The one real
    reshape here is splitting this project's fused `in_proj` (3*E, E) QKV weight into
    Llama's three separate q_proj/k_proj/v_proj matrices (E, E) each, since Llama's
    attention was never fused this way. No biases exist on either side (both this
    architecture and Llama default to bias=False everywhere), so there is nothing to
    carry over for attention/MLP projections beyond the weight matrices themselves.
    """
    source = checkpoint["model_state_dict"]
    embed_size = checkpoint["embed_size"]

    result = {
        "model.embed_tokens.weight": source["token_emb.weight"],
        "model.norm.weight": source["norm_f.weight"],
        "lm_head.weight": source["lm_head.weight"],
    }
    for layer in range(checkpoint["num_layers"]):
        src = f"blocks.{layer}."
        dst = f"model.layers.{layer}."
        in_proj = source[f"{src}attn.in_proj.weight"]
        q_proj, k_proj, v_proj = in_proj.split(embed_size, dim=0)
        result.update({
            f"{dst}input_layernorm.weight": source[f"{src}norm_1.weight"],
            f"{dst}self_attn.q_proj.weight": q_proj,
            f"{dst}self_attn.k_proj.weight": k_proj,
            f"{dst}self_attn.v_proj.weight": v_proj,
            f"{dst}self_attn.o_proj.weight": source[f"{src}attn.out_proj.weight"],
            f"{dst}post_attention_layernorm.weight": source[f"{src}norm_2.weight"],
            f"{dst}mlp.gate_proj.weight": source[f"{src}mlp.gate.weight"],
            f"{dst}mlp.up_proj.weight": source[f"{src}mlp.up.weight"],
            f"{dst}mlp.down_proj.weight": source[f"{src}mlp.down.weight"],
        })
    return result


def _load_export_dependencies():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "vLLM export dependencies are not installed. Run `uv sync --extra vllm` "
            "from this project first."
        ) from exc
    return LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def _llama_config(checkpoint, eot_id, LlamaConfig):
    return LlamaConfig(
        vocab_size=checkpoint["vocab_size"],
        hidden_size=checkpoint["embed_size"],
        intermediate_size=checkpoint["ffn_hidden"],
        num_hidden_layers=checkpoint["num_layers"],
        num_attention_heads=checkpoint["num_heads"],
        num_key_value_heads=checkpoint["num_heads"],  # no GQA here — plain MHA
        max_position_embeddings=checkpoint["context_length"],
        rope_theta=checkpoint.get("rope_theta", 10000.0),
        # RMSNorm's eps is hardcoded to 1e-6 in model.py (no ModelConfig field for
        # it), so this must match that literal rather than Llama's own default.
        rms_norm_eps=1e-6,
        hidden_act="silu",  # SwiGLU's gate activation — no approximation gap here,
                             # unlike the GPT-2/GELU export on the sibling projects.
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=eot_id,
        eos_token_id=eot_id,
        use_cache=True,
    )


@torch.inference_mode()
def _verify_equivalence(source_path, model, tokenizer):
    """Fail export if a fixed prompt's logits are not numerically equivalent."""
    from ..checkpoint import load_model

    _, _, native = load_model(source_path, "cpu")
    native.eval()
    model.eval()
    ids = tokenizer(
        "vLLM export parity check" + END_OF_TEXT,
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
        description="Export a native checkpoint to an equivalent Hugging Face Llama directory for vLLM."
    )
    parser.add_argument("--preset", default=None, help="Model size preset to export")
    parser.add_argument("--checkpoint", choices=["best", "latest", "final"], default=None,
                        help="Checkpoint to export (default: best, then latest/final)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New output directory (default includes selected checkpoint and step)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip native-vs-HF logit parity verification")
    args = parser.parse_args()

    LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast = _load_export_dependencies()
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

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
    )
    eot_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)

    config = _llama_config(checkpoint, eot_id, LlamaConfig)
    model = LlamaForCausalLM(config)
    missing, unexpected = model.load_state_dict(hf_state_dict(checkpoint), strict=False)
    allowed_missing = {"lm_head.weight"}  # tied to embed_tokens; may be omitted post tie_weights()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Unexpected export key mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()

    if tokenizer.encode(END_OF_TEXT, add_special_tokens=False) != [eot_id]:
        raise RuntimeError("The exported tokenizer does not preserve the document separator token.")
    if not args.no_verify:
        _verify_equivalence(source_path, model, tokenizer)

    output_dir.mkdir(parents=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Exported checkpoint step {step:,} to {output_dir}")
    print(f"Serve with: vllm serve {output_dir}")


if __name__ == "__main__":
    main()
