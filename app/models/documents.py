from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    title: str
    content: str
    chunk_id: str
    source_document: str
    embedding: list[float]
