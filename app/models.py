from pydantic import BaseModel
from typing import List

class UserMessage(BaseModel):
    user_id: int
    chat_id: int
    text: str

class RAGDocument(BaseModel):
    id: str
    content: str
    metadata: dict = {}

class RAGQuery(BaseModel):
    query: str
    top_k: int = 3

class RAGResult(BaseModel):
    documents: List[RAGDocument]

class LLMRequest(BaseModel):
    query: str
    context: List[str] = []

class LLMResponse(BaseModel):
    answer: str
    sources: List[str] = []
