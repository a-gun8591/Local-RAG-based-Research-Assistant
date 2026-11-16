FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time, so it isn't fetched on every run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "chat.py"]
