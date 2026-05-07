# ExamRAG

ExamRAG is an open-source desktop app: **PDF OCR → local RAG index → screen capture Q&A**, powered by OpenRouter.

## Portable Windows build (no Python required on target PC)

Build on your machine:
- `powershell -ExecutionPolicy Bypass -File tools\\build_release_windows.ps1`

Send `release\\ExamRAG.zip` to your friend. They run:
- `run_examrag.bat`

Notes:
- First run downloads the embedding model into the user profile cache (internet required).
- The app stores config/index/cache under the user profile (no admin, no PATH).
 - If Windows SmartScreen warns, click "More info" → "Run anyway" (unsigned exe).
