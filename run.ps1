$ErrorActionPreference = "Stop"

# Force UTF-8 mode (helps on Turkish path names).
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Host "Virtualenv not found. Creating .venv and installing requirements..."
  python -m venv .venv
  .\.venv\Scripts\python -m pip install --upgrade pip
  .\.venv\Scripts\pip install -r requirements.txt
}

& .\.venv\Scripts\python.exe .\main.py @args

