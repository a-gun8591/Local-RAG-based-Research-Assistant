"""
extract_pdf.py
Converts a PDF into a .txt file inside data/ so ingest.py can index it.

Usage:
    python extract_pdf.py your_paper.pdf
"""

import os
import sys

from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf.py your_paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    reader = PdfReader(pdf_path)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    os.makedirs(DATA_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
    out_path = os.path.join(DATA_DIR, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extracted {len(reader.pages)} page(s) to {out_path}")
    print("Next: run `python ingest.py`, then `python chat.py`.")


if __name__ == "__main__":
    main()
