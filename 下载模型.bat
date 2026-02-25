@echo off
cd /d "%~dp0"
echo Starting Qwen3-TTS Model Downloader...
python312\python.exe download_models.py
pause
