#!/usr/bin/env bash
set -e

PORT="${PORT:-8000}"

if ! command -v ngrok &>/dev/null; then
    echo "ngrok is not found. Run ./setup_ngrok.sh first to install it."
    exit 1
fi

echo "=== Anchor: Starting ngrok tunnel on port ${PORT} ==="
echo "Forwarding to local Anchor FastAPI server at http://localhost:${PORT}"
echo "Web inspection interface available at http://localhost:4040"
echo ""

ngrok http "${PORT}"
