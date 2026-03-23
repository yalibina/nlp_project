"""
app/monitoring/rag_logger.py

Модуль логирования для RAG-бота "Хронорус".
Пишет структурированные записи в CSV-файл и в стандартный Python logging.
"""

import csv
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

# ── Стандартный Python-логгер (консоль / файл) ─────────────────────────────
std_logger = logging.getLogger("chronorus.monitoring")


# ── Метка результата ────────────────────────────────────────────────────────
class ResponseLabel(str, Enum):
    """Возможные исходы обработки запроса."""

    SUCCESS = "success"                       # Получилось ответить
    OFF_TOPIC = "off_topic"                   # Не по истории России
    AFTER_2014 = "after_2014"                 # Вопрос касается событий после 2014
    NO_INFO_IN_DB = "no_info_in_db"           # Нет информации в базе
    ERROR = "error"                           # Техническая ошибка

    def human_readable(self) -> str:
        return {
            self.SUCCESS:     "Получилось ответить",
            self.OFF_TOPIC:   "Не ответил: вопрос не по истории России",
            self.AFTER_2014:  "Не ответил: вопрос после 2014 года",
            self.NO_INFO_IN_DB: "Не ответил: нет информации в БД",
            self.ERROR:       "Техническая ошибка",
        }[self]


# ── Запись лога ─────────────────────────────────────────────────────────────
@dataclass
class RAGLogEntry:
    """Одна запись метрик по одному запросу пользователя."""

    # Идентификация
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: Optional[int] = None          # Telegram user_id

    # Тексты
    user_query: str = ""
    bot_response: str = ""

    # Метрика времени (секунды)
    response_time_sec: float = 0.0

    # Метрики RAG
    retrieved_docs_count: int = 0
    retrieved_docs_scores: list[float] = field(default_factory=list)

    # Итоговая метка
    label: ResponseLabel = ResponseLabel.ERROR
    error_message: Optional[str] = None    # заполняется при label == ERROR
    # ── Вычисляемые агрегаты ────────────────────────────────────────────────
    @property
    def avg_doc_score(self) -> Optional[float]:
        return (
            round(sum(self.retrieved_docs_scores) / len(self.retrieved_docs_scores), 4)
            if self.retrieved_docs_scores
            else None
        )

    @property
    def max_doc_score(self) -> Optional[float]:
        return round(max(self.retrieved_docs_scores), 4) if self.retrieved_docs_scores else None

    @property
    def min_doc_score(self) -> Optional[float]:
        return round(min(self.retrieved_docs_scores), 4) if self.retrieved_docs_scores else None

    # ── Сериализация для CSV ─────────────────────────────────────────────────
    def to_csv_row(self) -> dict:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "user_query": self.user_query,
            "bot_response": self.bot_response,
            "response_time_sec": round(self.response_time_sec, 4),
            "retrieved_docs_count": self.retrieved_docs_count,
            "retrieved_docs_scores": str(self.retrieved_docs_scores),
            "avg_doc_score": self.avg_doc_score,
            "max_doc_score": self.max_doc_score,
            "min_doc_score": self.min_doc_score,
            "label": self.label.value,
            "label_human": self.label.human_readable(),
            "error_message": self.error_message or "",
        }

    # Названия столбцов — используются при создании CSV-заголовка
    @staticmethod
    def csv_fieldnames() -> list[str]:
        return [
            "request_id", "timestamp", "user_id",
            "user_query", "bot_response",
            "response_time_sec",
            "retrieved_docs_count", "retrieved_docs_scores",
            "avg_doc_score", "max_doc_score", "min_doc_score",
            "label", "label_human", "error_message",
        ]


# ── Основной логгер ─────────────────────────────────────────────────────────
class RAGLogger:
    """
    Потокобезопасный логгер для одного запроса к RAG-боту.

    Использование (типичный сценарий):
    ──────────────────────────────────
        logger = RAGLogger.get_instance()

        entry = logger.start_request(user_id=12345, query="Когда была Куликовская битва?")
        try:
            docs = retriever.search(query)
            logger.log_retrieval(entry, docs)          # сразу после получения документов

            answer = llm.generate(docs, query)
            logger.finish_request(
                entry,
                response=answer,
                label=ResponseLabel.SUCCESS,
            )
        except SomeBusinessException as e:
            logger.finish_request(entry, response="", label=ResponseLabel.OFF_TOPIC)
        except Exception as e:
            logger.finish_request(entry, response="", label=ResponseLabel.ERROR, error=e)
    """

    _instance: Optional["RAGLogger"] = None
    _lock: Lock = Lock()

    CSV_FIELDNAMES = RAGLogEntry.csv_fieldnames()

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._log_dir / "rag_metrics.csv"
        self._write_lock = Lock()
        self._ensure_csv_header()

    # ── Singleton ────────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, log_dir: str = "logs") -> "RAGLogger":
        """Возвращает единственный экземпляр логгера (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(log_dir=log_dir)
                    std_logger.info("RAGLogger инициализирован. CSV: %s", cls._instance._csv_path)
        return cls._instance

    # ── Публичное API ────────────────────────────────────────────────────────

    def start_request(self, user_id: Optional[int], query: str) -> RAGLogEntry:
        """
        Создаёт новую запись и запускает таймер.
        Вызывается в самом начале обработки сообщения.
        """
        entry = RAGLogEntry(
            user_id=user_id,
            user_query=query,
        )
        # Таймер записываем как атрибут объекта (не попадает в CSV напрямую)
        entry._start_time = time.monotonic()  # type: ignore[attr-defined]
        std_logger.debug("[%s] Запрос начат | user=%s | q=%r", entry.request_id, user_id, query)
        return entry

    def log_retrieval(
        self,
        entry: RAGLogEntry,
        docs_scores: list[float],
    ) -> None:
        """
        Записывает информацию о retrieved документах.
        Вызывается сразу после возврата результатов из Qdrant.

        Args:
            entry:        текущая запись (из start_request)
            docs_scores:  список score-ов документов в порядке убывания
        """
        entry.retrieved_docs_count = len(docs_scores)
        entry.retrieved_docs_scores = [round(s, 4) for s in docs_scores]
        std_logger.debug(
            "[%s] Retrieval: %d документов, scores=%s",
            entry.request_id,
            entry.retrieved_docs_count,
            entry.retrieved_docs_scores,
        )

    def finish_request(
        self,
        entry: RAGLogEntry,
        response: str,
        label: ResponseLabel,
        error: Optional[Exception] = None,
    ) -> RAGLogEntry:
        """
        Завершает замер времени, проставляет метку и сохраняет запись в CSV.
        Вызывается в самом конце обработки (в т.ч. в блоке except).

        Args:
            entry:    текущая запись (из start_request)
            response: финальный текст ответа бота
            label:    итоговая метка качества
            error:    объект исключения (только при label=ERROR)

        Returns:
            Заполненная запись (удобно для тестов / отладки).
        """
        start_time = getattr(entry, "_start_time", time.monotonic())
        entry.response_time_sec = time.monotonic() - start_time
        entry.bot_response = response
        entry.label = label
        if error is not None:
            entry.error_message = f"{type(error).__name__}: {error}"

        self._write_to_csv(entry)

        std_logger.info(
            "[%s] Запрос завершён | label=%s | time=%.2fs | docs=%d | user=%s",
            entry.request_id,
            label.value,
            entry.response_time_sec,
            entry.retrieved_docs_count,
            entry.user_id,
        )
        return entry

    # ── Внутренние методы ────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        """Создаёт файл с заголовком, если он ещё не существует."""
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
                writer.writeheader()

    def _write_to_csv(self, entry: RAGLogEntry) -> None:
        """Потокобезопасная запись строки в CSV."""
        with self._write_lock:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
                writer.writerow(entry.to_csv_row())
