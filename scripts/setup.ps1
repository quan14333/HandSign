$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$environmentPath = Join-Path $projectRoot '.venv-handsign'
$environmentPython = Join-Path $environmentPath 'Scripts\python.exe'

# Use the caller's Python; never reuse a copied virtual environment.
python -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Use Python 3.10, 3.11 or 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Install a supported Python and add it to PATH.' }
if (-not (Test-Path -LiteralPath $environmentPython)) {
    python -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create the virtual environment.' }
}
& $environmentPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Unable to prepare pip.' }
& $environmentPython -m pip install 'numpy>=1.26,<2'
if ($LASTEXITCODE -ne 0) { throw 'Unable to install NumPy.' }
& $environmentPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. See the pip error above.' }
& $environmentPython (Join-Path $projectRoot 'scripts\check_setup.py')
if ($LASTEXITCODE -ne 0) { throw 'Setup checks failed.' }
Write-Host 'Ready. Run: powershell -ExecutionPolicy Bypass -File .\run.ps1'
