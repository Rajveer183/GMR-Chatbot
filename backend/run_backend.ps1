# Start the GMR chatbot backend (gmr_agents) on port 8000.
#
#   .\run_backend.ps1          # normal run -- no auto-reload (use this)
#   .\run_backend.ps1 -Dev     # auto-reload on code changes (development only)
#
# Auto-reload is off by default: WatchFiles hung the reload several times when
# a file changed during a long Athena query, leaving the OLD code serving
# requests with no error. After a code change, stop (Ctrl+C) and start again.
# KPI logic lives in gmr_agents/schema/data/business_rules.yaml and reloads
# automatically -- no restart needed after editing it.
param(
    [switch]$Dev,
    [int]$Port = 0
)

Set-Location -Path $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if ($Port -eq 0) {
    if ($env:PORT) {
        $Port = [int]$env:PORT
    } else {
        # Check if port 8000 is already in use; if so, find next free port (e.g. 8001)
        $testPort = 8000
        while ($testPort -lt 8100) {
            $conn = Get-NetTCPConnection -LocalPort $testPort -State Listen -ErrorAction SilentlyContinue
            if (-not $conn) {
                $Port = $testPort
                break
            }
            $testPort++
        }
    }
}

Write-Host "Starting backend on port $Port..."
$uvArgs = @("-m", "uvicorn", "gmr_agents.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Dev) { $uvArgs += @("--reload", "--reload-dir", "gmr_agents") }
& $python @uvArgs
