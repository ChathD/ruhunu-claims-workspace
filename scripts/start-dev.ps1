# Install dependencies first, as described in README.md.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$frontendPath = Join-Path $projectRoot 'frontend'
$logDirectory = Join-Path $projectRoot '.local\logs'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install .[workspace] before starting the project.'
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendPath 'node_modules'))) {
    throw 'Run npm.cmd ci in frontend before starting the project.'
}
foreach ($port in @(8010, 5180)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is in use. Stop the existing workspace, or start each service manually on other ports."
    }
}
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$backendProcess = Start-Process -FilePath $pythonPath -ArgumentList '-u', '-m', 'backend', '--reload' `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDirectory 'backend.out.log') `
    -RedirectStandardError (Join-Path $logDirectory 'backend.err.log')
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if ($backendProcess.HasExited) { throw 'Backend exited. See .local/logs/backend.err.log.' }
        try {
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/api/health' -TimeoutSec 1
            if ($health.status -eq 'ok') { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw 'Backend did not become ready. See .local/logs/backend.err.log.' }
    Write-Host 'Claims dashboard: http://127.0.0.1:5180'
    Write-Host 'API documentation: http://127.0.0.1:8010/docs'
    Write-Host 'Press Ctrl+C to stop the development workspace.'
    Push-Location $frontendPath
    try { & npm.cmd run dev -- --port 5180 --strictPort } finally { Pop-Location }
} finally {
    if (-not $backendProcess.HasExited) {
        # Stop only the process tree started here, including Uvicorn's reloader.
        & taskkill.exe /PID $backendProcess.Id /T /F | Out-Null
    }
}
