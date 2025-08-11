# AI Call Center Backend (FastAPI)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Run

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Health: GET http://localhost:8000/health
- Chat: POST http://localhost:8000/chat
- Audio: served under http://localhost:8000/audio/ 