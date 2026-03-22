from pydantic import BaseModel
from typing import List

class UserMessage(BaseModel):
    user_id: int
    chat_id: int
    text: str

class RAGDocument(BaseModel):
    # in MVP our RAG returns only texts without metadata and gives it to gpt
    # id: str
    content: str
    # metadata: dict = {}

class RAGQuery(BaseModel):
    query: str
    top_k: int = 7

class RAGResult(BaseModel):
    documents: List[RAGDocument]

class LLMRequest(BaseModel):
    query: str
    context: List[str] = []

class LLMResponse(BaseModel):
    answer: str
    sources: List[str] = []
