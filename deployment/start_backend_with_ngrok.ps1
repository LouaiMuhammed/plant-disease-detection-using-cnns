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
    return
}

$ngrokCmd = Get-Command $NgrokPath -ErrorAction SilentlyContinue
if (-not $ngrokCmd) {
    Write-Warning "ngrok was not found. Install ngrok or pass -NgrokPath to the executable."
    Write-Host "Backend is still running locally on http://localhost:$Port"
    return
}

Start-Sleep -Seconds 2
Write-Host "Starting ngrok tunnel..."
$ngrok = Start-Process -FilePath $ngrokCmd.Source `
    -ArgumentList "http", $Port `
    -WorkingDirectory $scriptDir `
    -PassThru

Write-Host "ngrok PID: $($ngrok.Id)"
Write-Host "Open http://127.0.0.1:4040 to inspect and copy the public URL."
Write-Host "Stop both processes with: Stop-Process -Id $($backend.Id),$($ngrok.Id)"
