"""OpenAI-/vLLM-compatible `/v1/completions` server for the exported checkpoint.

Real vLLM (CUDA) and vLLM-Metal (Apple Silicon MLX) both need packages that are
not cleanly installable on this machine right now — see
`../docs/DDP_JOURNEY_2026-09-01.md` for the exact resolver errors. Per this
workspace's own `serving/vllm-smollm2-135m/docs/CUSTOM_PYTORCH_MODEL_INTEGRATION.md`
("Decision boundary: when not to use vLLM"), the right fallback is a small
Transformers-backed server exposing the same API shape, not blocking on vLLM.

This model is base/pretrain-only (no chat fine-tuning yet), so it exposes
`/v1/completions` (raw prompt -> continuation) rather than `/v1/chat/completions`
— asking it to follow a chat template would just be testing a capability it was
never trained for.

    uv run python scripts/cpu_vllm_server.py --model exports/vllm/50m/serving-step-14400
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


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0, le=2)


def create_app(model_id: str) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu").eval()
    app = FastAPI(title="custom-gpt-50m-ddp CPU server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": "transformers-cpu", "model": model_id}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/completions")
    def complete(request: CompletionRequest) -> dict[str, object]:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        inputs = tokenizer(request.prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=request.temperature > 0,
                temperature=request.temperature if request.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to an exported HF directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
