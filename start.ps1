Write-Host "=== ISRO Radiation Forecast System ===" -ForegroundColor Cyan

Write-Host "[1/4] Checking Ollama..." -ForegroundColor Yellow
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Download from https://ollama.com" -ForegroundColor Red
    exit 1
}

Write-Host "[2/4] Starting Ollama server..." -ForegroundColor Yellow
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
    Write-Host "[3/4] Pulling qwen2.5:0.5b..." -ForegroundColor Yellow
    ollama pull qwen2.5:0.5b
}

Write-Host "[4/4] Starting live ingestion and dashboard..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "main.py" -WindowStyle Minimized
Start-Sleep -Seconds 3
Write-Host "Launching dashboard..." -ForegroundColor Cyan
streamlit run dashboard/app.py