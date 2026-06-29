#!/bin/bash

echo "=== ISRO Radiation Forecast System ==="

echo "[1/6] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Download from https://ollama.com"
    exit 1
fi

echo "[2/6] Starting Ollama server..."
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "Ollama already running."
else
    ollama serve &
    sleep 5
    echo "Ollama started."
fi

echo "[3/6] Pulling qwen2.5:7b..."
ollama pull qwen2.5:7b

echo "[4/6] Initializing database..."
python -m database.db

echo "[5/6] Checking historical data..."
DB="data/radiation.db"
if [ ! -f "$DB" ] || [ $(wc -c < "$DB") -lt 100000 ]; then
    echo "Fetching historical data (first run only)..."
    python -m ingestion.omniweb_fetcher
    python -m ingestion.donki_fetcher
else
    echo "Historical data present. Skipping."
fi

echo "[6/6] Starting live ingestion..."
python main.py &
sleep 3

echo "Launching dashboard..."
streamlit run dashboard/app.py
