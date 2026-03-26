@echo off
setlocal

cd /d "%~dp0"

echo [1/4] Preparing Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
)

echo [2/4] Activating environment...
call ".venv\Scripts\activate.bat"

echo [3/4] Installing/updating backend dependencies...
python -m pip install --upgrade pip >nul
pip install -r backend\requirements.txt

echo [4/4] Starting backend and opening Chrome...
start "" chrome http://localhost:8000/login.html
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
