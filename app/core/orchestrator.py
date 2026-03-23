from app.services.llm import llm_service
from app.services.rag import rag_service
from app.models import UserMessage, RAGDocument, RAGQuery, LLMRequest, RAGResult
from app.telemetry import trace
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from natasha.extractors import DatesExtractor

import logging

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
dates_extractor = DatesExtractor(morph_vocab)

NO_INFO = "У меня нет информации об этом историческом событии."
OFF_TOPIC = "Я могу отвечать только на вопросы об исторических событиях."

class Orchestrator:
    """Central coordinator: RAG retrieve → LLM complete pipeline."""

    @trace("orchestrator.handle_query")
    async def handle_query(self, message: UserMessage) -> RAGResult:
        """Retrieve context for user query, then call LLM with that context."""
        try:
            rag_result = await rag_service.retrieve(RAGQuery(query=message.text))
        except Exception as exc:
             return RAGResult(
                answer="Произошла внутренняя ошибка. Попробуй позже.",
                doc_scores=[],
                status="error_retrieval",
                error=exc
            )
        context_with_metadata = rag_result.documents  # список кортежей, кортеж = (текст, метадата)
        sources = []
        context = []
        for (i, doc) in enumerate(context_with_metadata):
            meta = doc.metadata
            source_url = meta.get('page_url', None)
            if source_url:
                sources.append(source_url)
            meta_text = f"[CHUNK {i+1}] (document_id={meta["document_id"]}, chunk_id={meta["chunk_id"]}, url={source_url})\n"
            part = meta_text + doc.content
    
            context.append(part)

        context = "\n---\n".join(context) + "\n---\n"

        if not context_with_metadata:
            return RAGResult(
                 answer="Прошу прощения, я не знаю ответ на этот вопрос. Пожалуйста, попробуйте переформулировать или спросить конкретнее. Если Вы используете даты, то убедитесь, что они в правильном формате!",
                 doc_scores=[],
                 status="no_info_in_db")
        #print(context)
        try:
            llm_response = await llm_service.complete(
                LLMRequest(query=message.text, context=[context])
            ) # LLMResponse.answer
        except Exception as exc:
             return RAGResult(
                answer="Произошла внутренняя ошибка. Попробуй позже.",
                doc_scores=[],
                status="error_llm",
                error=exc
                )
        print(message.text)
        answer = llm_response.answer
        print(answer)
        matches = list(dates_extractor(answer))
        if answer[:20] == NO_INFO[:20]:
             return RAGResult(
                         answer=NO_INFO, 
                         doc_scores=[],
                         status="no_info_other")
        
        if answer[:20] == OFF_TOPIC[:20]:
             return RAGResult(
                         answer=OFF_TOPIC, 
                         doc_scores=[],
                         status="off_topic")

        if len(matches) > 0:
            exact_years = [match.fact.year for match in matches if match.fact.year is not None]
            if exact_years and max(exact_years) >= 2014 or "amnyam" in answer:  # то есть в ответе ллмки есть упоминание 2014 - это и случаи когда она выдает свой промпт и когда она отвечает на вопрос запрещенный
                    return RAGResult(
                         answer="Прошу прощения, я не знаю ответ на этот вопрос или не могу на него ответить, попробуйте переформулировать или задать другой вопрос!",
                         doc_scores=[],
                         status="after_2014")
        return RAGResult(
             answer=answer,
             doc_scores=[]
        )

orchestrator = Orchestrator()
