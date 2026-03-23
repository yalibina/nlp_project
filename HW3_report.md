# Отчет по ДЗ 3
## Мониторинг и логирование
Реализована система логирования и дашборд для мониторинга качества

Реализован класс `RAGLogger`, который:
- работает как singleton
- безопасно пишет данные в CSV
- интегрируется в пайплайн обработки запроса

### Структура логов

Каждый запрос сохраняется как `RAGLogEntry` со следующими полями:

#### Основные данные:
- `request_id`
- `timestamp`
- `user_id`
- `user_query`
- `bot_response`

#### Метрики:
- `response_time_sec`
- `retrieved_docs_count`
- `retrieved_docs_scores`

#### Производные:
- `avg_doc_score`
- `max_doc_score`
- `min_doc_score`

#### Классификация результата:
- `label` (ResponseLabel)
- `label_human`
- `error_message`

---

## Классификация ответов

| Label            | Описание |
|------------------|----------|
| success          | Успешный ответ |
| off_topic        | Вопрос не по теме |
| after_2014       | Вопрос вне временного диапазона |
| no_info_in_db    | Нет информации в базе |
| no_info_other    | Недостаточно информации |
| error            | Техническая ошибка |

## Дэшик
Дэшик - на streamlit + plotly
