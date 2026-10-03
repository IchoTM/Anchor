#!/usr/bin/env bash
set -e

echo "=== Anchor: Installing ngrok for Pop!_OS / Debian / Ubuntu ==="

if command -v ngrok &>/dev/null; then
    echo "ngrok is already installed at: $(which ngrok)"
    ngrok version
    exit 0
fi

echo "Adding ngrok apt repository..."
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list

echo "Installing ngrok package..."
sudo apt update
sudo apt install -y ngrok

echo ""
echo "=== ngrok installation successful! ==="
ngrok version
echo ""
echo "Next step: If you haven't yet, configure your authtoken from https://dashboard.ngrok.com:"
echo "    ngrok config add-authtoken <YOUR_AUTH_TOKEN>"
echo ""
echo "To start forwarding port 8000:"
echo "    ./run_tunnel.sh"
