"""
FastAPI server for the TinyStories GPT. See docs/SERVING.md for the full explanation.

Run:
    uvicorn api_server:app --host 127.0.0.1 --port 8010 --reload
"""
import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from inference import generate, load_model_and_tokenizer
from model import detect_device

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "tinystories_gpt_checkpoint.pt")

app = FastAPI(title="TinyStories GPT API")
device = detect_device()
model, tokenizer, ckpt = load_model_and_tokenizer(CHECKPOINT_PATH, device)
print(f"[server] loaded checkpoint step={ckpt.get('step')} "
      f"params={model.num_parameters():,} device={device}")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    model_step: int
    device: str


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "params": model.num_parameters(),
            "checkpoint_step": ckpt.get("step")}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    with torch.no_grad():
        text = generate(
            model, tokenizer, req.prompt,
            ctx_len=ckpt["context_length"],
            max_new_tokens=req.max_new_tokens,
            device=device,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
    completion = text[len(req.prompt):] if text.startswith(req.prompt) else text
    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        model_step=ckpt.get("step", -1),
        device=device,
    )
