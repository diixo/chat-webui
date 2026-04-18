# chat-webui

## Docker запуск (llama.cpp)

В репозитории добавлен `docker-compose.yml` для запуска llama.cpp сервера с моделью из репозитория.

### Запуск llama API (Docker)

```bash
docker compose up
```

API доступен на `http://localhost:8080`

### Запуск Web UI (локально)

```bash
python main.py 8765
```

Web UI доступен на `http://localhost:8765`

Модель: `./Llama-68M-Chat-v1-GGUF/llama-68m-chat-v1.q8_0.gguf`

### Остановка Docker

```bash
docker compose down
```
