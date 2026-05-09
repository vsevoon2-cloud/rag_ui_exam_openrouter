$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..

if (!(Test-Path ".venv")) {
  py -3 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -U pip wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
& .\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
& .\.venv\Scripts\python.exe -m pip install -U pyinstaller
# PyInstaller may attempt to collect chromadb.server.fastapi; ensure optional deps exist for packaging.
& .\.venv\Scripts\python.exe -m pip install -U fastapi starlette

Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue

& .\.venv\Scripts\python.exe -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name ExamRAG `
  --windowed `
  --collect-all chromadb `
  --collect-all sentence_transformers `
  --collect-all transformers `
  --collect-all tokenizers `
  --collect-all safetensors `
  --collect-all huggingface_hub `
  --collect-all rank_bm25 `
  --hidden-import torch `
  main.py

$releaseDir = "release"
Remove-Item -Recurse -Force $releaseDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $releaseDir | Out-Null

Copy-Item "dist\\ExamRAG.exe" "$releaseDir\\ExamRAG.exe"
Copy-Item "tools\\run_examrag.bat" "$releaseDir\\run_examrag.bat"
Copy-Item "README.md" "$releaseDir\\README.md"

if (Test-Path "$releaseDir\\ExamRAG.zip") { Remove-Item "$releaseDir\\ExamRAG.zip" -Force }
Compress-Archive -Path "$releaseDir\\*" -DestinationPath "$releaseDir\\ExamRAG.zip" -Force

Write-Host "OK: release\\ExamRAG.zip"
