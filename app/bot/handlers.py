from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command

from app.core.orchestrator import orchestrator
from app.models import UserMessage, RAGDocument

router = Router()

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
    answer = await orchestrator.handle_query(user_msg)
    await message.answer(answer)
