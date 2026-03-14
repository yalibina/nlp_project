from app.services.llm import llm_service
from app.services.rag import rag_service
from app.models import UserMessage, RAGDocument, RAGQuery, LLMRequest
from app.telemetry import trace

class Orchestrator:
    """Central coordinator: RAG retrieve → LLM complete pipeline."""

    @trace("orchestrator.handle_query")
    async def handle_query(self, message: UserMessage) -> str:
        """Retrieve context for user query, then call LLM with that context."""
        rag_result = await rag_service.retrieve(RAGQuery(query=message.text))
        context = [doc.content for doc in rag_result.documents]

        llm_response = await llm_service.complete(
            LLMRequest(query=message.text, context=context)
        )
        return llm_response.answer

    async def add_document(self, document: RAGDocument) -> bool:
        return await rag_service.add(document)

orchestrator = Orchestrator()
