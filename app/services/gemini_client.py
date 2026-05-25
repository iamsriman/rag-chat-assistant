import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    pass


class GeminiEmbeddingClient:
    """Small async wrapper around Gemini embeddings.

    Embeddings convert text into numeric vectors. ChromaDB compares these
    vectors with cosine similarity during retrieval.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_"))

    async def embed(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        if not self.configured:
            raise LLMProviderError("GEMINI_API_KEY is not configured")

        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                payload = {
                    "model": self._model_path(self.embedding_model),
                    "content": {"parts": [{"text": text}]},
                    "taskType": self._task_type(task_type),
                }
                data = await self._post(client, self._embed_url(), payload, "Embedding")
                values = data.get("embedding", {}).get("values")
                if not values:
                    raise LLMProviderError("Embedding response did not include vector values")
                embeddings.append(values)
        return embeddings

    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        try:
            response = await client.post(url, json=payload, params={"key": self.api_key})
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"{operation} request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise self._http_error(operation, exc) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"{operation} request could not connect") from exc

    def _embed_url(self) -> str:
        return f"{self.base_url}/{self._model_name(self.embedding_model)}:embedContent"

    @staticmethod
    def _model_name(model: str) -> str:
        return model.removeprefix("models/")

    @staticmethod
    def _model_path(model: str) -> str:
        model_name = GeminiEmbeddingClient._model_name(model)
        return f"models/{model_name}"

    @staticmethod
    def _task_type(task_type: str) -> str:
        if task_type == "retrieval_query":
            return "RETRIEVAL_QUERY"
        return "RETRIEVAL_DOCUMENT"

    @staticmethod
    def _http_error(operation: str, exc: httpx.HTTPStatusError) -> LLMProviderError:
        status_code = exc.response.status_code
        try:
            detail = exc.response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = exc.response.text

        if status_code == 429:
            return LLMProviderError(f"{operation} request was rate limited or quota was exceeded")
        if status_code in {400, 401, 403}:
            return LLMProviderError(f"{operation} request failed because the Gemini API key or request is invalid: {detail}")
        if status_code == 404:
            return LLMProviderError(f"{operation} Gemini model was not found or is unavailable: {detail}")
        return LLMProviderError(f"{operation} Gemini API error: {status_code} {detail}")
