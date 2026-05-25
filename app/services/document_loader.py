import json
from pathlib import Path
from typing import Any


def load_documents(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list):
        raise ValueError("docs.json must contain a list of documents")

    documents: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Document {index} must be an object")
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        if not title or not content:
            raise ValueError(f"Document {index} must include title and content")
        documents.append({"title": title, "content": content})
    return documents


def chunk_text(text: str, chunk_size: int = 380, overlap: int = 60) -> list[str]:
    # Chunking keeps each embedding focused while overlap preserves context
    # across chunk boundaries.
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks
