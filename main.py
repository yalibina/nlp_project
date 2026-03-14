import asyncio
import logging
from aiogram import Bot, Dispatcher
from app.config import settings
from app.bot.handlers import router

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

async def main():
    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Bot polling started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
