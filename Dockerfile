FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN hf download aitetic/gpt2 --local-dir ./models/gpt2

COPY . .

EXPOSE 8080

CMD ["python", "server.py", "--model", "./models/gpt2", "--prompt-style", "raw", "--host", "0.0.0.0"]
