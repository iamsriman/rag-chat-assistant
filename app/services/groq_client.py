import logging
import os

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.services.gemini_client import LLMProviderError

logger = logging.getLogger(__name__)


class GroqClient:
    """Groq chat client using Groq's OpenAI-compatible API surface."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
        self.client = (
            AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            if self.api_key
            else None
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_") and self.client is not None)

    async def chat(self, prompt: str) -> tuple[str, int | None]:
        if not self.configured:
            raise LLMProviderError("GROQ_API_KEY is not configured")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a grounded RAG assistant. Use only the retrieved context.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            tokens = response.usage.total_tokens if response.usage else None
            logger.info("Groq token usage: %s", tokens)
            return response.choices[0].message.content or "", tokens
        except APITimeoutError as exc:
            raise LLMProviderError("Chat request timed out") from exc
        except APIConnectionError as exc:
            raise LLMProviderError("Chat request could not connect") from exc
        except RateLimitError as exc:
            raise LLMProviderError("Chat request was rate limited or quota was exceeded") from exc
        except APIStatusError as exc:
            if exc.status_code in {401, 403}:
                raise LLMProviderError("Chat request failed because GROQ_API_KEY is invalid or unauthorized") from exc
            raise LLMProviderError(f"Groq API error: {exc.status_code}") from exc
