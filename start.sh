#!/bin/bash

echo "=== ISRO Radiation Forecast System ==="

echo "[1/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Download from https://ollama.com"
    exit 1
fi

echo "[2/4] Starting Ollama server..."
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "Ollama already running."
else
    ollama serve &
    sleep 5
    echo "Ollama started."
fi

echo "[3/4] Pulling qwen2.5:7b..."
ollama pull qwen2.5:7b

echo "[4/4] Starting live ingestion & background tasks..."
python main.py &
sleep 3

echo "Launching dashboard..."
streamlit run dashboard/app.py
