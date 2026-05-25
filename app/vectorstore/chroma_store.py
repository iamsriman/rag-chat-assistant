import os
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings

from app.models.documents import DocumentChunk


class ChromaVectorStore:
    """Persistent ChromaDB store for document chunk embeddings."""

    def __init__(self) -> None:
        db_path = Path(os.getenv("CHROMA_DB_PATH", ".chroma"))
        collection_name = os.getenv("CHROMA_COLLECTION", "rag_documents")
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        existing = self.collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def add_many(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "title": chunk.title,
                    "chunk_id": chunk.chunk_id,
                    "source_document": chunk.source_document,
                }
                for chunk in chunks
            ],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[DocumentChunk, float]]:
        # Chroma returns cosine distance for a cosine collection. Similarity is
        # easier to reason about in logs/API output, so convert distance to
        # similarity with 1 - distance.
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[tuple[DocumentChunk, float]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            meta = self._metadata(metadata)
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            matches.append(
                (
                    DocumentChunk(
                        title=str(meta["title"]),
                        content=str(document),
                        chunk_id=str(meta["chunk_id"]),
                        source_document=str(meta["source_document"]),
                        embedding=[],
                    ),
                    similarity,
                )
            )
        return matches

    @staticmethod
    def _metadata(metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        return {"title": "", "chunk_id": "", "source_document": ""}
