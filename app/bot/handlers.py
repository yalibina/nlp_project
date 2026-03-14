from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command

from app.core.orchestrator import orchestrator
from app.models import UserMessage, RAGDocument

router = Router()

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Hello! Send me a question or use /add <text> to index a document.")

@router.message(Command("add"))
async def cmd_add(message: Message):
    """Add a document to RAG: /add <content>"""
    content = message.text.removeprefix("/add").strip()
    if not content:
        await message.answer("Usage: /add <document content>")
        return

    doc = RAGDocument(id=str(message.message_id), content=content)
    ok = await orchestrator.add_document(doc)
    await message.answer("✅ Document indexed." if ok else "❌ Indexing failed.")

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
