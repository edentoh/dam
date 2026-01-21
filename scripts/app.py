import uvicorn
import os
import argparse
from dam.api.main import app

def main():
    parser = argparse.ArgumentParser(description="DAM Prediction Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print(f"Starting DAM Server on http://{args.host}:{args.port}")
    
    # Check for crucial env var before starting
    if not os.environ.get("DAM_API_KEY"):
        print("[WARNING] DAM_API_KEY is not set. Requests may be unauthorized.")

    uvicorn.run(
        "dam.api.main:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )

if __name__ == "__main__":
    main()