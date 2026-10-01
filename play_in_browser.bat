@echo off
REM Play Dune Imperium against the AI in your browser.
REM Double-click this file, or run it from a terminal in this folder.
cd /d "%~dp0"
start "" http://localhost:8765
".venv\Scripts\python.exe" ui\server.py
