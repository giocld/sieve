# FastAPI Backend (new)
api: uvicorn src.api:app --host 0.0.0.0 --port $PORT

# Legacy Dash Dashboard (kept for backward compatibility)
web: gunicorn src.dashboard:server --timeout 120 --workers 1 --threads 2
