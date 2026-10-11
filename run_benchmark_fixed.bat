@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -m src.selfplay.benchmark --games 300 --workers 20 --only "random,Bloodlines-blind,original,value net,value+policy,search-trained,tuned to human,fight model,your style,winners" > reports\benchmark_current.log 2>&1
".venv\Scripts\python.exe" -u -m src.selfplay.benchmark --games 120 --workers 20 --only "search, 8,search, 24" >> reports\benchmark_current.log 2>&1
