# Server Examples

Examples demonstrating how to use the SMLX Server for inference.

## Overview

SMLX Server provides an OpenAI-compatible REST API for:
- Text generation (`/v1/completions`)
- Chat completions (`/v1/chat/completions`)
- Audio transcription (`/v1/audio/transcriptions`)
- Embeddings (`/v1/embeddings`)
- Model management (`/v1/models`)

## Prerequisites

### Start the Server

```bash
# Option 1: Direct Python
python -m smlx.server.app

# Option 2: Uvicorn (production)
uvicorn smlx.server.app:app --host 0.0.0.0 --port 8000

# Option 3: With reload (development)
uvicorn smlx.server.app:app --reload
```

The server will start on `http://localhost:8000`

### Install Client Dependencies

```bash
# For basic HTTP examples
pip install requests

# For OpenAI-compatible examples
pip install openai
```

## Examples

### 1. Client Example ([client_example.py](client_example.py))

Comprehensive examples using raw HTTP requests.

```bash
python client_example.py
```

**Demonstrates:**
- Health checks
- Listing models
- Text completion
- Chat completion
- Streaming (text and chat)
- Multi-turn conversations
- Testing different models

### 2. OpenAI-Compatible Example ([openai_compatible.py](openai_compatible.py))

Shows that SMLX Server works with the official OpenAI Python client.

```bash
python openai_compatible.py
```

**Demonstrates:**
- Drop-in OpenAI replacement
- Using OpenAI client with SMLX
- Chat, completions, streaming
- Multi-turn conversations

## Quick Start

### Basic HTTP Request

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "prompt": "Hello, world!",
        "max_tokens": 50,
    }
)

print(response.json()["choices"][0]["text"])
```

### Using OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="mlx-community/SmolLM2-135M-Instruct",
    messages=[
        {"role": "user", "content": "What is AI?"}
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
```

## API Reference

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/SmolLM2-135M-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/SmolLM2-135M-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

## Configuration

### Environment Variables

```bash
# Server host and port
export SMLX_HOST=0.0.0.0
export SMLX_PORT=8000

# Model cache directory
export SMLX_CACHE_DIR=~/.cache/smlx

# Enable debug logging
export SMLX_DEBUG=1
```

### Server Options

```python
# In your code
from smlx.server import app
import uvicorn

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="info",
    reload=True,  # Auto-reload on code changes
)
```

## Advanced Usage

### Custom Model Loading

```python
from smlx.server import ModelManager

# Preload models on startup
manager = ModelManager()
await manager.load_model("mlx-community/SmolLM2-135M-Instruct")
await manager.load_model("mlx-community/SmolLM2-360M-Instruct")
```

### Rate Limiting

```python
from smlx.server.middleware import RateLimitMiddleware

# Customize rate limits
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=100,  # Default: 60
)
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -e .
RUN pip install uvicorn[standard]

EXPOSE 8000

CMD ["uvicorn", "smlx.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=SMLX Server
After=network.target

[Service]
Type=simple
User=smlx
WorkingDirectory=/opt/smlx
ExecStart=/usr/bin/python -m smlx.server.app
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1704134400,
  "models_loaded": 2
}
```

### Metrics

The server provides basic metrics:
- Request count
- Response times
- Error rates
- Model usage

Access via `/health` endpoint.

## Troubleshooting

**Q: Server won't start**
- Check if port 8000 is already in use
- Verify all dependencies are installed
- Check firewall settings

**Q: Model loading is slow**
- Models are loaded lazily on first request
- Preload models on server startup for faster first response
- Use smaller models for faster loading

**Q: Out of memory**
- Reduce model cache size in ModelManager
- Use quantized models (4-bit)
- Run on machine with more RAM

**Q: Streaming not working**
- Check that client supports Server-Sent Events (SSE)
- Verify `stream=true` is set in request
- Check for proxy/firewall issues

## Best Practices

1. **Use quantized models in production**
   - 4x memory reduction
   - Faster inference
   - Minimal quality loss

2. **Enable caching**
   - Cache loaded models
   - Use model manager effectively
   - Preload common models

3. **Monitor performance**
   - Track response times
   - Monitor memory usage
   - Log errors

4. **Secure your deployment**
   - Add authentication if needed
   - Use HTTPS in production
   - Configure CORS properly
   - Rate limit requests

5. **Scale horizontally**
   - Run multiple server instances
   - Use load balancer
   - Share model cache

## Resources

- [Server Implementation](../../smlx/server/)
- [API Schemas](../../smlx/server/schemas.py)
- [Model Manager](../../smlx/server/model_manager.py)
- [OpenAI API Spec](https://platform.openai.com/docs/api-reference)

## Citation

```bibtex
@software{smlx_server,
  title = {SMLX Server: OpenAI-Compatible Inference Server},
  author = {SMLX Contributors},
  year = {2025},
  url = {https://github.com/yourusername/smlx}
}
```
