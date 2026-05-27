# ODPlatform Web Backend

## Run

```bash
python -m pip install -e ./apps/web-backend
uvicorn odp_web_backend.main:app --app-dir apps/web-backend/src --reload --port 8000
```

## Agent Runtime

Copy `apps/web-backend/.env.example` to `apps/web-backend/.env`, then set:

```bash
MODEL_PATH=models/checkpoints/train-5-20260526-155146-yolo26n-seg-best.pt
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```

Interactive endpoints:

- `POST /api/analyze`
- `GET /api/session/{session_id}`
- `POST /api/chat/{session_id}`
- `POST /api/chat/{session_id}/stream`
