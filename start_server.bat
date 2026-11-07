@echo off
cd /d "%~dp0"
echo Starting Dark Pattern Detector server...
python src\server.py
pause
