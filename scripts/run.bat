@echo off
echo Starting MCC AI System...

REM Start Backend
start "MCC Backend" cmd /k "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000"

REM Start Frontend
start "MCC Frontend" cmd /k "cd frontend && npm run dev"

echo Services started in separate windows.
