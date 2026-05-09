# ExamRAG

## Install

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

## Build EXE

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_release_windows.ps1
```

## Run EXE

```powershell
release\run_examrag.bat
```
