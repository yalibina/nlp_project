from app.services.llm import llm_service
from app.services.rag import rag_service
from app.models import UserMessage, RAGDocument, RAGQuery, LLMRequest
from app.telemetry import trace
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from natasha.extractors import DatesExtractor

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
dates_extractor = DatesExtractor(morph_vocab)


class Orchestrator:
    """Central coordinator: RAG retrieve → LLM complete pipeline."""

    @trace("orchestrator.handle_query")
    async def handle_query(self, message: UserMessage) -> str:
        """Retrieve context for user query, then call LLM with that context."""
        rag_result = await rag_service.retrieve(RAGQuery(query=message.text))
        context = [doc.content for doc in rag_result.documents]

        if not context:
            return "Прошу прощения, я не знаю ответ на этот вопрос. Пожалуйста, попробуйте переформулировать или спросить конкретнее. Если Вы используете даты, то убедитесь, что они в правильном формате!"
        #print(context)
        llm_response = await llm_service.complete(
            LLMRequest(query=message.text, context=context)
        )
        print(message.text)
        answer = llm_response.answer
        print(answer)
        matches = list(dates_extractor(answer))

        if len(matches) > 0:
            exact_years = [match.fact.year for match in matches if match.fact.year is not None]
            if exact_years and max(exact_years) >= 2014:  # то есть в ответе ллмки есть упоминание 2014 - это и случаи когда она выдает свой промпт и когда она отвечает на вопрос запрещенный
                    return "Прошу прощения, я не знаю ответ на этот вопрос или не могу на него ответить, попробуйте переформулировать или задать другой вопрос!"
        return answer

orchestrator = Orchestrator()
