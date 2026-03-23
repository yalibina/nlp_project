# Проект "Хронорус": RAG-система по истории России

## Описание проекта
"Хронорус" — это LLM-based RAG-система, специализирующаяся на истории России. 

Главная идея данного проекта - сделать "надежный" RAG инструмент: хочется избежать галлюцинаций по вопросам дат, имен, событий. Бот отвечает на вопросы по истории и событиях до 2014 года.

Бот строго опирается на загруженную энциклопедическую базу и использует гибридный поиск (семантика + точный поиск по ключевым словам и датам), чтобы предоставлять максимально достоверные факты (и побыстрее).

## Запуск
Бот реализован в виде чат-бота в Telegram: @chronorusbot

## Архитектура и стек
1. **Векторная БД:** Qdrant с гибридным поиском. Запускается локально через Docker.
2. **Embeddings:** `sergeyzh/BERTA` с HF локально
3. **LLM:** `Qwen2.5:7b` локально через `ollama`
4. **Backend:** `aiogram`, `FastAPI`, `asyncio`
5. **Monitoring:** `streamlit`, `watchdog`, `plotly` - локальный дашборд с метриками.

Отчет по MVP: [MVP_report.md](MVP_report.md), отчет по мониторингу: [HW3_report.md](HW3_report.md)

## Структура проекта

* [`app`](app) - backend проекта
  * [`bot`](app/bot)
    * [`handlers.py`](app/bot/handlers.py) - bot handlers
  * [`core`](core) 
    * [`orchestrator.py`](core/orchestrator.py) - RAG + LLM pipeline
    * [`qdrant_manager.py`](core/qdrant_manager.py) - запросы в БД, retrieval и управление Qdrant
  * [`monitoring`](app/monitoring)
    * [`rag_logger.py`](app/monitoring/rag_logger.py) - логгер
  * [`services`](services)
    * [`llm.py`](services/llm.py) - LLM генерация
    * [`rag.py`](services/rag.py) - Вызов RAG
  * [`config.py`](config.py)
  * [`dashboard.py`](app/dashboard.py) - дэшик 
  *  [`models.py`](models.py) - классы для пайплайна
* [`scripts`](scripts)
  * [`parsing / parse_wiki.py`](scripts/parsing/parse_wiki.py) - парсинг Википедии и разбиение на чанки
  * [`build_qdrant_index.py`](scripts/build_qdrant_index.py) - построение базы данных
  * [`metrics.py`](scripts/metrics.py) - построение джсона с метриками
* [`data`](data)
  * [`full_bench.csv`](full_bench.csv) - золотой датасет
  * [`rag_evaluation_results.json`](rag_evaluation_results.json) - результат поиска по золотому датасету
* [`main.py`](main.py) - запуск бота
* [`requirements.txt`](requirements.txt)

## Команда
- Либина Яна https://github.com/yalibina 
- Инденбаум Илья https://github.com/Ilia1243x
- Векшин Кирилл https://github.com/oversanya
- Игнатов Максим https://github.com/m4xig1


## Потенциальные улучшения/изменения
* Расширение базы данных - больше статей Википедии, подключение дополнительных источников (учебников, методических пособий).
* Асинхронность - сейчас тг-бот поддерживает асинхронность, а БД и модели - нет. Поэтому возможно долгое ожидание ответа. Для масштабирования необходима асинхронность моделей и БД.
