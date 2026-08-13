"""
Serves the ORIGINAL, unmodified HuggingFaceTB/SmolLM2-135M base checkpoint — no LoRA
adapter — on its own port, so it can be queried directly and compared live against
../smollm2-135m-dolly-lora/'s fine-tuned server, rather than only ever appearing as one
half of an offline comparison script (compare_before_after.py).

IMPORTANT, and different from tinyllama-1.1b-base-serving/: this is a true BASE model
with NO chat template at all (confirmed: its tokenizer_config.json has no chat_template
field — see ../smollm2-135m-dolly-lora/docs/APPROACH.md). This server therefore does
PLAIN-TEXT COMPLETION ONLY — whatever prompt you send is continued as raw text, exactly
as the model was pretrained to do. It will NOT behave like an assistant answering a
question — that behavior is exactly what ../smollm2-135m-dolly-lora/'s fine-tuning adds,
and this endpoint's whole purpose is showing what the model does WITHOUT it.
"""
import argparse

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Raw text to continue (NOT an instruction "
                                                     "— this base model has no chat template)")
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)


class GenerateResponse(BaseModel):
    completion: str
    full_text: str
    device: str
    model: str
    note: str = "Plain-text completion — this base model has no chat template / instruction-following behavior."


@torch.inference_mode()
def generate_text(model, tokenizer, device: str, req: GenerateRequest) -> tuple[str, str]:
    inputs = tokenizer(req.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = full_text[len(req.prompt):].strip()
    return completion, full_text


def create_app(model_id: str) -> FastAPI:
    device = detect_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    app = FastAPI(title="SmolLM2-135M (original base model, no adapter) API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok", "device": device, "model": model_id,
            "adapter": "none (original base checkpoint)",
            "chat_template": "none — plain-text completion only",
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        completion, full_text = generate_text(model=model, tokenizer=tokenizer, device=device, req=req)
        return GenerateResponse(completion=completion, full_text=full_text, device=device, model=model_id)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the original SmolLM2-135M base checkpoint (no LoRA)")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    app = create_app(model_id=args.model_id)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
