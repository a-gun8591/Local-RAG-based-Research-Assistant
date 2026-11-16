# RAG Project: Usage Guide

Run all commands from the project folder.

## One-time setup

```
ollama pull llama3.2
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Adding a PDF

```
python extract_pdf.py your_paper.pdf
python ingest.py
python chat.py
```

## Adding a .txt file

Copy it into `data/`, then:

```
python ingest.py
python chat.py
```

## Asking questions

At the `You:` prompt, type a question. Type `exit` to quit. Each answer ends with the source files it used.

## Changing how many chunks are used

In `chat.py`, change `TOP_K` (default 4). Higher values give the model more context but make answers slower.

## Quick reference

| Task                       | Command                          |
|----------------------------|----------------------------------|
| Convert a PDF              | `python extract_pdf.py file.pdf` |
| Rebuild the index          | `python ingest.py`               |
| Ask questions              | `python chat.py`                 |
