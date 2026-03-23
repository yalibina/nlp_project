# Проект "Хронорус": RAG-система по истории России

## Описание проекта
"Хронорус" — это LLM-based RAG-система, специализирующаяся на истории России. Главная идея данного проекта - сделать "надежный" RAG инструмент: хочется избежать галлюцинаций по вопросам дат, имен, событий. Бот отвечает на вопросы по истории и событиях до 2014 года.

Бот строго опирается на загруженную энциклопедическую базу и использует гибридный поиск (семантика + точный поиск по ключевым словам и датам), чтобы предоставлять максимально достоверные факты (и побыстрее).

## Запуск
Бот реализован в виде чат-бота в Telegram: @chronorusbot

## Архитектура и стек
1. **Векторная БД:** Qdrant с гибридным поиском. Запускается локально через Docker.
2. **Embeddings:** `sergeyzh/BERTA` с HF локально
3. **LLM:** `Qwen2.5:7b` локально через `ollama`
4. **Backend:** `aiogram`, `FastAPI`, `asyncio`

Подробнее - в [отчёте](MVP_report.md)

## Структура проекта

* [`app`](app) - backend проекта
  * [`bot`](app/bot)
    * [`handlers.py`](app/bot/handlers.py) - bot handlers
  * [`core`](core) 
    * [`orchestrator.py`](core/orchestrator.py) - RAG + LLM pipeline
    * [`qdrant_manager.py`](core/qdrant_manager.py) - запросы в БД, retrieval и управление Qdrant
  * [`services`](services)
    * [`llm.py`](services/llm.py) - LLM генерация
    * [`rag.py`](services/rag.py) - Вызов RAG
  * [`config.py`](config.py)
  *  [`models.py`](models.py) - классы для пайплайна
* [`scripts`](scripts)
  * [`parsing / parse_wiki.py`](parsing/parse_wiki.py) - парсинг Википедии и разбиение на чанки
  * [`build_qdrant_index.py`](parsing/build_qdrant_index.py) - построение базы данных
* [`main.py`](main.py) - запуск бота
* [`requirements.txt`](requirements.txt)

## Команда
- Либина Яна https://github.com/yalibina 
- Инденбаум Илья https://github.com/Ilia1243x
- Векшин Кирилл https://github.com/oversanya
- Игнатов Максим https://github.com/m4xig1


## Потенциальные улучшения/изменения
* Расширение базы данных - больше статей Википедии, подключение дополнительных источников (учебников, методических пособий).
* Асинхронность - сейчас тг-бот поддерживает асинхронность, а БД и модели - нет. Поэтому возможно долгое ожидание ответа. Для масштабирования необходима асинхронность в моделей и БД.
