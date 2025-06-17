#!/usr/bin/env python3
"""
GitProbe API Server
Runs the FastAPI application with proper configuration.
"""

import uvicorn
import argparse
from api.app import app

def main():
    """Run the GitProbe API server."""
    parser = argparse.ArgumentParser(description="GitProbe API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting GitProbe API on {args.host}:{args.port}")
    print(f"📊 Simple, direct repository analysis API")
    print(f"📖 Docs: http://{args.host}:{args.port}/docs")
    
    # Run server
    uvicorn.run(
        "api.app:app",  # Import string format for reload
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 