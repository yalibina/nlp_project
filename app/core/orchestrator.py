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
        context_with_metadata = rag_result.documents  # список кортежей, кортеж = (текст, метадата)
        sources = []
        context = []
        for doc in context_with_metadata:
            meta = doc.metadata
            source_url = meta.get('page_url', None)
            if source_url:
                sources.append(source_url)
            part = f"""
                Айди документа: {meta["document_id"]}
                Айди чанка внутри этого документа: {meta["chunk_id"]}
                Источник: {source_url}
                Текст: {doc.content}
                ---"""
            context.append(part)
        context = "\n\n".join(context)

        if not context_with_metadata:
            return "Прошу прощения, я не знаю ответ на этот вопрос. Пожалуйста, попробуйте переформулировать или спросить конкретнее. Если Вы используете даты, то убедитесь, что они в правильном формате!"
        #print(context)
        llm_response = await llm_service.complete(
            LLMRequest(query=message.text, context=[context])
        )
        print(message.text)
        answer = llm_response.answer
        print(answer)
        matches = list(dates_extractor(answer))

        if len(matches) > 0:
            exact_years = [match.fact.year for match in matches if match.fact.year is not None]
            if exact_years and max(exact_years) >= 2014 or "amnyam" in answer:  # то есть в ответе ллмки есть упоминание 2014 - это и случаи когда она выдает свой промпт и когда она отвечает на вопрос запрещенный
                    return "Прошу прощения, я не знаю ответ на этот вопрос или не могу на него ответить, попробуйте переформулировать или задать другой вопрос!"
        return answer

orchestrator = Orchestrator()
