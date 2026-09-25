$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv-handsign\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw 'Run scripts\setup.ps1 first to create this machine''s virtual environment.'
}
& $environmentPython (Join-Path $PSScriptRoot 'record.py') @args
exit $LASTEXITCODE
