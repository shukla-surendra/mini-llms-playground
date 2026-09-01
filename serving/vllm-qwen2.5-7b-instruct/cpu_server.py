"""Small CPU-only OpenAI-compatible server used when no accelerator is available.

Loads the full bf16 model, no quantization — expect this to be slow and memory-
heavy for a 7B model (~15GB weights alone). Prefer the mps/mlx-lm backend on
Apple Silicon or the cuda/vllm backend on an NVIDIA box; this exists as the
last-resort fallback, matching the sibling serving/vllm-smollm2-135m pattern.
"""

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


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0, le=2)


def create_app(model_id: str) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu").eval()
    app = FastAPI(title="Qwen2.5-7B-Instruct CPU server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": "transformers-cpu", "model": model_id}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest) -> dict[str, object]:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        prompt = tokenizer.apply_chat_template(
            [message.model_dump() for message in request.messages], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=request.temperature > 0,
                temperature=request.temperature if request.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": completion}, "finish_reason": "stop"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
