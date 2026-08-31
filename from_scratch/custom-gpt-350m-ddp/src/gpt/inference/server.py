"""FastAPI serving layer — HTTP only; all model work is delegated to the package."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..checkpoint import load_model, resolve_serving_checkpoint
from ..config import load_settings
from ..runtime import get_device
from .generate import generate_text


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Prompt text to continue")
    max_new_tokens: int = Field(default=100, ge=1, le=1024)
    do_sample: bool = True
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=50000)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    # Raw base models continue text rather than answering turns, so trimming at
    # chat role markers is off by default here.
    trim_at_role_markers: bool = False


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    model_context_length: int
    device: str
    architecture: str


def create_app():
    """Build the app, loading the checkpoint for the currently selected model size."""
    _, _, paths, label = load_settings()
    device = get_device()
    checkpoint_path = resolve_serving_checkpoint(paths)
    checkpoint, tokenizer, model = load_model(checkpoint_path, device, eval_mode=True)

    app = FastAPI(title=f"custom-gpt ({label}) API", version="1.0.0")
    app.state.checkpoint = checkpoint
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device
    app.state.label = label
    app.state.checkpoint_path = str(checkpoint_path)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_label": label,
            "checkpoint": str(checkpoint_path),
            "step": checkpoint.get("step"),
            "param_count": checkpoint.get("param_count"),
            "device": device,
            "context_length": checkpoint["context_length"],
            "architecture": checkpoint.get("architecture", "unknown"),
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        try:
            full_text, completion = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=req.prompt,
                context_length=checkpoint["context_length"],
                max_new_tokens=req.max_new_tokens,
                device=device,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                postprocess=req.trim_at_role_markers,
            )
            return GenerateResponse(
                prompt=req.prompt,
                completion=completion,
                full_text=full_text,
                model_context_length=checkpoint["context_length"],
                device=device,
                architecture=checkpoint.get("architecture", "unknown"),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# uvicorn entrypoint: `uvicorn gpt.inference.server:app`
app = create_app()
