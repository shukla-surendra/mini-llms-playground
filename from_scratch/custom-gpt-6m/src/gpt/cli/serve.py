"""`gpt-serve` — run the FastAPI inference server.

CHECKPOINT_PATH env var selects which checkpoint to serve (see inference/server.py);
HOST/PORT match this project's own Makefile defaults (127.0.0.1:8010).
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Serve the model over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "gpt.inference.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
