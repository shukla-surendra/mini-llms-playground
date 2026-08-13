"""
Serves the ORIGINAL, unmodified TinyLlama/TinyLlama-1.1B-Chat-v1.0 checkpoint — no LoRA
adapter — on its own port, so it can be queried directly and compared live against
../tinyllama-1.1b-lora/'s adapter-loaded server (port 8001), rather than only ever
appearing as one half of an offline comparison script.

This model IS already chat-tuned by its original authors (unlike smollm2-135m-base-
serving's true base model), so it uses its own built-in chat template, same as
../tinyllama-1.1b-lora/serve_tinyllama_lora.py does.
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
    prompt: str = Field(min_length=1, description="User prompt")
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    completion: str
    full_text: str
    device: str
    model: str


def build_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_text(model, tokenizer, device: str, req: GenerateRequest) -> tuple[str, str]:
    formatted = build_prompt(tokenizer=tokenizer, prompt=req.prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
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

    app = FastAPI(title="TinyLlama-1.1B-Chat (original, no adapter) API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "device": device, "model": model_id, "adapter": "none (original checkpoint)"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        completion, full_text = generate_text(model=model, tokenizer=tokenizer, device=device, req=req)
        return GenerateResponse(completion=completion, full_text=full_text, device=device, model=model_id)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the original TinyLlama-1.1B-Chat checkpoint (no LoRA)")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    app = create_app(model_id=args.model_id)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
