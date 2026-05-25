from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)


class RetrievedChunk(BaseModel):
    title: str
    chunkId: str
    score: float
    sourceDocument: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int | None = None
    retrievedChunks: int
    sources: list[RetrievedChunk] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
