from app.models import RAGDocument, RAGQuery, RAGResult
from app.telemetry import trace, record_metric

from app.core.qdrant_manager import HistoryQdrantManager 

qdrant_manager = HistoryQdrantManager()

class RAGService:
    """Stub for vector DB operations (Chroma / Qdrant / Weaviate)."""

    @trace("rag.retrieve")
    async def retrieve(self, query: RAGQuery) -> RAGResult:
        """Embed query and fetch top-k nearest documents from the vector DB."""
        question = query.query[:500].strip()
        result = qdrant_manager.query_to_db(question=question, limit=query.top_k)
        if result == "НИЧЕГО НЕ НАШЕЛ НЕ НАДО ТЫКАТЬ ПО API ЛИШНИЙ РАЗ НЕЙРОНКУ!!!":
            documents = []
        else:
            documents = [RAGDocument(content=chunk) for chunk in result]
        record_metric("rag.retrieve.top_k", len(documents))
        return RAGResult(documents=documents)

rag_service = RAGService()
