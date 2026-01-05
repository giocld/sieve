#!/bin/bash
# Development runner for Sieve NBA Analytics
# Starts both FastAPI backend and React frontend

echo "============================================="
echo "Sieve NBA Analytics - Development Server"
echo "============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start FastAPI backend in background
echo "Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start React frontend
echo "Starting React frontend on port 5173..."
cd frontend && npm run dev &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

echo ""
echo "============================================="
echo "Servers running:"
echo "  Backend API: http://localhost:8000"
echo "  Frontend:    http://localhost:5173"
echo "  API Docs:    http://localhost:8000/docs"
echo "============================================="
echo "Press Ctrl+C to stop"
echo ""

# Wait for background processes
wait
