from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

import logging

from app.core.orchestrator import orchestrator
from app.models import UserMessage, RAGDocument
from app.monitoring import RAGLogger, ResponseLabel

router = Router()
logger = logging.getLogger(__name__)

# Инициализируем RAGLogger один раз (singleton)
rag_logger = RAGLogger.get_instance(log_dir="logs")

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет! Я Хронорус. Задайте мне любой вопрос по истории России!")  # or use /add <text> to index a document.

@router.message(F.text)
async def handle_query(message: Message):
    """Forward user text -> orchestrator -> reply with LLM answer."""
    user_msg = UserMessage(
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        text=message.text,
    )
    log_entry = rag_logger.start_request(user_id=message.from_user.id, query=message.text)

    rag_answer = await orchestrator.handle_query(user_msg)
    answer = rag_answer.answer
    rag_logger.log_retrieval(
        entry=log_entry,
        docs_scores=rag_answer.doc
    )
    rag_logger.finish_request(
        entry=log_entry,
        response=answer,
        label=map_status_to_label(rag_answer.status),
        error=rag_answer.error
    )
    await message.answer(answer)

def map_status_to_label(status: str) -> ResponseLabel:
    status = status.lower().strip()

    mapping = {
        "success": ResponseLabel.SUCCESS,
        "off_topic": ResponseLabel.OFF_TOPIC,
        "after_2014": ResponseLabel.AFTER_2014,
        "no_info_in_db": ResponseLabel.NO_INFO_IN_DB,
        "no_info_other": ResponseLabel.NO_INFO_OTHER,
        "error": ResponseLabel.ERROR,
    }

    return mapping.get(status, ResponseLabel.ERROR)