#!/bin/bash

# Start Backend
echo "Starting Backend..."
cd backend
# Run in background
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Services started. Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop."

wait