# & "C:\Users\loaim\Downloads\ngrok-v3-stable-windows-amd64\ngrok.exe" http 8000

param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [string]$ApiModule = "api:app",
    [string]$NgrokPath = "ngrok",
    [switch]$SkipNgrok
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $scriptDir

function Resolve-NgrokCommand {
    param([string]$RequestedPath)

    $candidates = @()
    if ($RequestedPath) {
        $candidates += $RequestedPath
    }

    if ($env:ngrok) {
        $candidates += $env:ngrok.Trim('"')
    }

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }

        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd
        }

        if (Test-Path $candidate) {
            return [pscustomobject]@{ Source = (Resolve-Path $candidate).Path }
        }
    }

    return $null
}

$venvPython = Join-Path $repoRoot "plant-disease-detection-env\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} else {
    $pythonCmd = "python"
}

Write-Host "Starting FastAPI backend on port $Port..."
$backend = Start-Process -FilePath $pythonCmd `
    -ArgumentList "-m", "uvicorn", $ApiModule, "--host", $BindHost, "--port", $Port `
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

$ngrokCmd = Resolve-NgrokCommand -RequestedPath $NgrokPath
if (-not $ngrokCmd) {
    Write-Warning "ngrok not found. Backend is running locally on http://localhost:$Port"
    if ($env:ngrok) {
        Write-Warning "Tried env:ngrok=$($env:ngrok) as well, but it could not be resolved."
    }
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
