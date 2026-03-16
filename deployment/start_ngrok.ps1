param(
    [int]$Port = 8000,
    [string]$Host = "0.0.0.0",
    [string]$ApiModule = "api:app",
    [string]$NgrokPath = "ngrok",
    [switch]$SkipNgrok
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $scriptDir

$venvPython = Join-Path $repoRoot "plant-disease-detection-env\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} else {
    $pythonCmd = "python"
}

Write-Host "Starting FastAPI backend on port $Port..."
$backend = Start-Process -FilePath $pythonCmd `
    -ArgumentList "-m", "uvicorn", $ApiModule, "--host", $Host, "--port", $Port `
    -WorkingDirectory $scriptDir `
    -PassThru

Write-Host "Backend PID: $($backend.Id)"
Write-Host "Local URL: http://localhost:$Port"

if ($SkipNgrok) {
    Write-Host "SkipNgrok set. Backend is running without tunnel."
    try {
        while ($true) { Start-Sleep -Seconds 5 }
    } finally {
        Write-Host "Shutting down..."
        Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
        Write-Host "Done."
    }
    return
}

$ngrokCmd = Get-Command $NgrokPath -ErrorAction SilentlyContinue
if (-not $ngrokCmd) {
    Write-Warning "ngrok not found. Backend is running locally on http://localhost:$Port"
    try {
        while ($true) { Start-Sleep -Seconds 5 }
    } finally {
        Write-Host "Shutting down..."
        Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
        Write-Host "Done."
    }
    return
}

Write-Host "Waiting for backend to be ready..."
$maxAttempts = 30
$attempt = 0
do {
    Start-Sleep -Seconds 2
    $attempt++
    try {
        Invoke-WebRequest -Uri "http://localhost:$Port/" -TimeoutSec 2 -ErrorAction Stop | Out-Null
        Write-Host "Backend is ready."
        break
    } catch {
        Write-Host "Waiting... ($attempt/$maxAttempts)"
    }
} while ($attempt -lt $maxAttempts)

if ($attempt -eq $maxAttempts) {
    Write-Warning "Backend did not respond after $($maxAttempts * 2) seconds. Check for errors."
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Starting ngrok tunnel..."
$ngrok = Start-Process -FilePath $ngrokCmd.Source `
    -ArgumentList "http", $Port `
    -WorkingDirectory $scriptDir `
    -PassThru

Write-Host "ngrok PID: $($ngrok.Id)"
Write-Host "Open http://127.0.0.1:4040 to inspect and copy the public URL."

try {
    Write-Host "Press Ctrl+C to stop all processes."
    while ($true) { Start-Sleep -Seconds 5 }
} finally {
    Write-Host "Shutting down..."
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $ngrok.Id -ErrorAction SilentlyContinue
    Write-Host "Done."
}