$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv-handsign\Scripts\python.exe'
$previousPythonPath = $env:PYTHONPATH
if (-not (Test-Path -LiteralPath $environmentPython)) {
    # This checkout also supports the local runtime verified during integration.
    # Never add model-deps here: the verified runtime uses its installed model libraries.
    $apiDependencies = Join-Path $PSScriptRoot '.verification\api-deps'
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand -or -not (Test-Path -LiteralPath (Join-Path $apiDependencies 'uvicorn'))) {
        throw 'No usable runtime found. Run scripts\setup.ps1 with Python 3.10-3.12 to create .venv-handsign.'
    }
    $environmentPython = $pythonCommand.Source
    $env:PYTHONPATH = $apiDependencies
    if ($previousPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath }
}
try {
    & $environmentPython -c "import importlib.util, sys; required = ('uvicorn', 'fastapi', 'multipart', 'cv2', 'mediapipe', 'numpy', 'torch', 'transformers', 'huggingface_hub', 'fastdtw'); missing = [name for name in required if importlib.util.find_spec(name) is None]; print('Missing dependencies: ' + ', '.join(missing)) if missing else None; sys.exit(bool(missing))"
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependencies are incomplete. Install requirements.txt into the HandSign environment.' }
    Write-Host "HandSign Python: $environmentPython"
    Write-Host 'Starting HandSign API. Keep this terminal open; Ctrl+C stops the service.'
    & $environmentPython -m uvicorn api:app --app-dir $PSScriptRoot --host 127.0.0.1 --port 8001 @args
    $apiExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
exit $apiExitCode
