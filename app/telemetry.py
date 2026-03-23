"""Stubs for distributed tracing and metrics. Swap in OpenTelemetry / Prometheus when ready."""
import functools
import logging
from typing import Callable

logger = logging.getLogger(__name__)

def trace(span_name: str):
    """Decorator stub for distributed tracing (e.g. OpenTelemetry spans)."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # TODO: start_span(span_name), inject trace context into headers
            logger.debug("[trace] %s started", span_name)
            result = await func(*args, **kwargs)
            logger.debug("[trace] %s finished", span_name)
            return result
        return wrapper
    return decorator

def record_metric(name: str, value: float, tags: dict = {}):
    """Stub for counter/gauge recording (e.g. prometheus_client, StatsD)."""
    # TODO: metrics_client.gauge(name, value, tags=tags)
    logger.debug("[metric] %s=%.4f tags=%s", name, value, tags)
