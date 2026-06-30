#!/bin/bash

echo "=== ISRO Radiation Forecast System ==="

echo "[1/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found."
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [ -f /etc/debian_version ] || [ -f /etc/redhat-release ]; then
        echo "Attempting to install Ollama automatically..."
        if command -v sudo &> /dev/null; then
            curl -fsSL https://ollama.com/install.sh | sudo sh
        else
            curl -fsSL https://ollama.com/install.sh | sh
        fi
        
        # Verify if installation succeeded
        if ! command -v ollama &> /dev/null; then
            echo "Installation failed. Please install Ollama manually: https://ollama.com"
            exit 1
        fi
    else
        echo "Please download and install Ollama from: https://ollama.com"
        exit 1
    fi
fi

echo "[2/4] Starting Ollama server..."
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "Ollama already running."
else
    ollama serve &
    sleep 5
    echo "Ollama started."
fi

echo "[3/4] Pulling qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b

echo "[4/4] Starting live ingestion & background tasks..."
python main.py &
sleep 3

echo "Launching dashboard..."
python -m streamlit run dashboard/app.py
