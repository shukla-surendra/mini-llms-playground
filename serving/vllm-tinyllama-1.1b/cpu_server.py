"""Small CPU fallback for TinyLlama; CUDA/Metal uses vLLM instead."""
from __future__ import annotations

import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(80, ge=1, le=512)
    temperature: float = Field(0.7, ge=0, le=2)


def app_for(model_id: str) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu").eval()
    app = FastAPI(title="TinyLlama CPU server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": "transformers-cpu", "model": model_id}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: Request) -> dict[str, object]:
        if not request.messages:
            raise HTTPException(400, "messages must not be empty")
        prompt = tokenizer.apply_chat_template([m.model_dump() for m in request.messages], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=request.max_tokens, do_sample=request.temperature > 0, temperature=request.temperature if request.temperature > 0 else None, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    uvicorn.run(app_for(args.model), host=args.host, port=args.port)
