from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from app.services.rag_service import rag_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "healthy" if rag_service.is_ready else "degraded"
    return HealthResponse(status=status)


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest) -> ChatResponse:
    return await rag_service.answer(request)
