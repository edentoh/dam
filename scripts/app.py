import uvicorn
import os
import argparse
import subprocess
from contextlib import suppress
from dotenv import load_dotenv
from dam.api.main import app


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _start_ngrok(port: int) -> subprocess.Popen | None:
    enabled = _as_bool(os.environ.get("DAM_NGROK_AUTOSTART"), default=False)
    if not enabled:
        return None

    ngrok_bin = os.environ.get("DAM_NGROK_BIN", "ngrok").strip() or "ngrok"
    ngrok_url = os.environ.get("DAM_NGROK_URL", "").strip()

    cmd = [ngrok_bin, "http", str(port)]
    if ngrok_url:
        cmd.append(f"--url={ngrok_url}")

    try:
        proc = subprocess.Popen(cmd)
    except Exception as e:
        print(f"[Ngrok] Failed to start ngrok: {e}")
        return None

    shown = ngrok_url if ngrok_url else "<random-public-url>"
    print(f"[Ngrok] Tunnel starting for localhost:{port} using {shown}")
    return proc

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="DAM Prediction Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print(f"Starting DAM Server on http://{args.host}:{args.port}")
    
    # Check for crucial env var before starting
    if not os.environ.get("DAM_API_KEY"):
        print("[WARNING] DAM_API_KEY is not set. Requests may be unauthorized.")

    ngrok_proc = None
    if args.reload and _as_bool(os.environ.get("DAM_NGROK_AUTOSTART"), default=False):
        print("[Ngrok] Auto-start skipped with --reload to avoid duplicate tunnels.")
    else:
        ngrok_proc = _start_ngrok(args.port)

    try:
        uvicorn.run(
            "dam.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    finally:
        if ngrok_proc is not None and ngrok_proc.poll() is None:
            with suppress(Exception):
                ngrok_proc.terminate()

if __name__ == "__main__":
    main()
