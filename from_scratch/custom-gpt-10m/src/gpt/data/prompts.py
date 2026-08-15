"""Loading the held-out prompt file used by inference and evaluation."""

from pathlib import Path

PROMPT_SEPARATOR = "<END_PROMPT>"

DEFAULT_PROMPTS = [
    "System: You are a helpful coding assistant for docker workflows.\nUser: How do I debug container startup failures?\nAssistant:",
    "System: You are a helpful coding assistant for python debugging.\nUser: Why do I get KeyError in pandas and how can I fix it?\nAssistant:",
    "System: You are a helpful coding assistant for SQL optimization.\nUser: My query is slow on a large table. What should I check first?\nAssistant:",
    "System: You are a helpful coding assistant for unit testing.\nUser: How should I structure tests for a new API endpoint?\nAssistant:",
    "System: You are a helpful coding assistant for incident response.\nUser: Production latency doubled after deploy. What immediate steps should I take?\nAssistant:",
]


def load_prompts(path, default_prompts=None):
    """Prompts separated by <END_PROMPT> blocks, or one per line as a fallback.

    Returns `default_prompts` instead of raising when the file is missing, so an
    evaluation pass still works on a fresh checkout.
    """
    path = Path(path)
    if not path.exists():
        if default_prompts is not None:
            return default_prompts
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")

    text = path.read_text(encoding="utf-8")
    if PROMPT_SEPARATOR in text:
        prompts = [p.strip() for p in text.split(PROMPT_SEPARATOR) if p.strip()]
        if prompts:
            return prompts
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    return prompts if prompts else (default_prompts or [])
