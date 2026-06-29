Set-Content -Path "start.ps1" -Value @'
Write-Host "=== ISRO Radiation Forecast System ===" -ForegroundColor Cyan

Write-Host "[1/6] Checking Ollama..." -ForegroundColor Yellow
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Download from https://ollama.com" -ForegroundColor Red
    exit 1
}

Write-Host "[2/6] Starting Ollama server..." -ForegroundColor Yellow
$running = $false
try {
    Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 3 -ErrorAction Stop | Out-Null
    Write-Host "Ollama already running." -ForegroundColor Green
    $running = $true
} catch {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    try {
        Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 5 -ErrorAction Stop | Out-Null
        Write-Host "Ollama started." -ForegroundColor Green
        $running = $true
    } catch {
        Write-Host "Ollama failed to start." -ForegroundColor Red
    }
}

if ($running) {
    Write-Host "[3/6] Pulling qwen2.5:7b..." -ForegroundColor Yellow
    ollama pull qwen2.5:7b
}

Write-Host "[4/6] Initializing database..." -ForegroundColor Yellow
python -m database.db

Write-Host "[5/6] Checking historical data..." -ForegroundColor Yellow
$db = Get-Item "data\radiation.db" -ErrorAction SilentlyContinue
if (-not $db -or $db.Length -lt 100000) {
    Write-Host "Fetching historical data..." -ForegroundColor Yellow
    python -m ingestion.omniweb_fetcher
    python -m ingestion.donki_fetcher
} else {
    Write-Host "Historical data present. Skipping." -ForegroundColor Green
}

Write-Host "[6/6] Starting live ingestion and dashboard..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "main.py" -WindowStyle Minimized
Start-Sleep -Seconds 3
Write-Host "Launching dashboard..." -ForegroundColor Cyan
streamlit run dashboard/app.py
'@ -Encoding utf8