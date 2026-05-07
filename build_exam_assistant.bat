@echo off
setlocal
cd /d %~dp0

python -m pip install -r requirements-rag.txt
python -m pip install -r requirements-ui.txt
python -m pip install pyinstaller

pyinstaller ^
  --noconfirm ^
  --windowed ^
  --name ExamAssistant ^
  --add-data "out\chroma_db_v2;out\chroma_db_v2" ^
  --add-data "out\or_md_flash2;out\or_md_flash2" ^
  scripts\exam_assistant_ui.py

echo Build complete. See dist\ExamAssistant\
endlocal
