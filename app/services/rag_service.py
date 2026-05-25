import logging
import os
from pathlib import Path

from fastapi import HTTPException

from app.models.documents import DocumentChunk
from app.models.schemas import ChatRequest, ChatResponse, RetrievedChunk
from app.prompts.rag_prompt import build_rag_prompt
from app.services.document_loader import chunk_text, load_documents
from app.services.gemini_client import GeminiEmbeddingClient, LLMProviderError
from app.services.groq_client import GroqClient
from app.services.history import ConversationHistory
from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "I could not find enough information in the knowledge base to answer this question."


class RAGService:
    def __init__(self) -> None:
        self.embedding_client = GeminiEmbeddingClient()
        self.llm_client = GroqClient()
        self.vector_store = ChromaVectorStore()
        self.history = ConversationHistory(max_pairs=5)
        self.top_k = int(os.getenv("RAG_TOP_K", "3"))
        self.threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))
        self.index_ready = False
        self.is_ready = False

    async def initialize(self) -> None:
        if not self.embedding_client.configured:
            logger.warning("GEMINI_API_KEY is missing; RAG index was not built")
            return
        if not self.llm_client.configured:
            logger.warning("GROQ_API_KEY is missing; chat generation will not be available")

        docs_path = Path(__file__).resolve().parent.parent.parent / "docs.json"
        documents = load_documents(docs_path)

        chunk_records: list[tuple[str, str, str, str]] = []
        for doc_index, document in enumerate(documents):
            for chunk_index, chunk in enumerate(chunk_text(document["content"])):
                chunk_id = f"doc-{doc_index}-chunk-{chunk_index}"
                chunk_records.append((document["title"], chunk, chunk_id, document["title"]))

        try:
            embeddings = await self.embedding_client.embed([record[1] for record in chunk_records])
        except LLMProviderError as exc:
            logger.exception("Failed to build embedding index: %s", exc)
            self.index_ready = False
            self.is_ready = False
            return

        chunks: list[DocumentChunk] = []
        for record, embedding in zip(chunk_records, embeddings):
            title, content, chunk_id, source_document = record
            chunks.append(
                DocumentChunk(
                    title=title,
                    content=content,
                    chunk_id=chunk_id,
                    source_document=source_document,
                    embedding=embedding,
                )
            )

        self.vector_store.reset()
        self.vector_store.add_many(chunks)
        self.index_ready = True
        self.is_ready = self.llm_client.configured
        logger.info("Indexed %s chunks from %s documents into ChromaDB", len(chunk_records), len(documents))

    async def answer(self, request: ChatRequest) -> ChatResponse:
        if not self.index_ready:
            raise HTTPException(status_code=503, detail={"error": "RAG index is not ready. Check GEMINI_API_KEY and startup logs."})
        if not self.llm_client.configured:
            raise HTTPException(status_code=503, detail={"error": "GROQ_API_KEY is not configured"})

        try:
            query_embedding = (await self.embedding_client.embed([request.message], task_type="retrieval_query"))[0]
        except LLMProviderError as exc:
            raise HTTPException(status_code=503, detail={"error": str(exc)}) from exc

        # Retrieval always happens before the LLM call. The query embedding is
        # compared with stored chunk embeddings in ChromaDB using cosine space.
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        for chunk, score in results:
            logger.info(
                "Retrieved chunk session=%s chunk=%s title=%s similarity=%.4f",
                request.sessionId,
                chunk.chunk_id,
                chunk.title,
                score,
            )

        usable_results = [(chunk, score) for chunk, score in results if score >= self.threshold]
        sources = [
            RetrievedChunk(
                title=chunk.title,
                chunkId=chunk.chunk_id,
                score=round(score, 4),
                sourceDocument=chunk.source_document,
            )
            for chunk, score in usable_results
        ]

        if not usable_results:
            self.history.add(request.sessionId, request.message, FALLBACK_RESPONSE)
            return ChatResponse(reply=FALLBACK_RESPONSE, tokensUsed=0, retrievedChunks=0, sources=[])

        context = "\n\n".join(
            f"Source: {chunk.title} ({chunk.chunk_id})\n{chunk.content}"
            for chunk, _score in usable_results
        )
        prompt = build_rag_prompt(
            context=context,
            history=self.history.format(request.sessionId),
            question=request.message,
        )

        try:
            reply, tokens = await self.llm_client.chat(prompt)
        except LLMProviderError as exc:
            raise HTTPException(status_code=503, detail={"error": str(exc)}) from exc

        self.history.add(request.sessionId, request.message, reply)
        return ChatResponse(reply=reply, tokensUsed=tokens, retrievedChunks=len(usable_results), sources=sources)


rag_service = RAGService()
