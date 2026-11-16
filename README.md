# Local RAG-based Research Assistant

Ask questions about your own study material and get answers from a local LLM, with no cloud API. Documents are split into chunks, embedded with `all-MiniLM-L6-v2`, and searched by cosine similarity. The top matches are passed to `llama3.2` running in Ollama, which answers only from those passages and names the source files.

## Stack

Python, sentence-transformers, NumPy (cosine-similarity search, no vector database), Ollama (llama3.2), pypdf, Docker.

## Setup (Windows, macOS, or Linux)

1. Install [Ollama](https://ollama.com) and pull the model:
   ```
   ollama pull llama3.2
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

The first run of `ingest.py` downloads the embedding model (about 90 MB). After that, everything runs offline.

## Usage

```
python extract_pdf.py your_paper.pdf   # PDFs only: converts to data/your_paper.txt
python ingest.py                        # builds embeddings.npz from data/*.txt
python chat.py                          # interactive Q&A (type 'exit' to quit)
```

Add more files to `data/` and re-run `ingest.py`. The index is rebuilt from all files each time.

## Run with Docker

Ollama runs on your host, and the container connects to it.

1. On the host, make Ollama listen on all interfaces. Set `OLLAMA_HOST=0.0.0.0` in your environment and restart Ollama.
2. Build and run:
   ```
   docker build -t local-rag .
   docker run -it --rm --add-host=host.docker.internal:host-gateway local-rag
   ```
   The `--add-host` flag is needed on Linux. Docker Desktop on Windows and macOS already has it, so you can leave it out there.

The image includes `embeddings.npz`. The `data/` folder is not copied into the image, so to re-index, run `ingest.py` on the host.

## Project structure

```
ingest.py        build the index from data/*.txt
chat.py          retrieve chunks and answer questions with Ollama
extract_pdf.py   convert a PDF to .txt in data/
embeddings.npz   saved index (chunks, sources, vectors)
data/            your source .txt files (not committed)
Dockerfile       container build
```

## Limitations

- Chunking is fixed-size by characters, not by sentence or section.
- Answers name the source file, not the page or passage.
- Each question is answered on its own, with no chat history.
- The index is rebuilt from scratch on every ingest.
