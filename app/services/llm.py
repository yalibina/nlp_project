from app.models import LLMRequest, LLMResponse
from app.telemetry import trace, record_metric

class LLMService:
    """Stub for LLM API calls (OpenAI / Anthropic / local vLLM)."""

    @trace("llm.complete")
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Build a prompt from query + RAG context and call the LLM."""
        # TODO: format prompt, call llm_client.chat.completions.create(...)
        record_metric("llm.context_chunks", float(len(request.context)))
        return LLMResponse(answer="[LLM response placeholder]")

llm_service = LLMService()
