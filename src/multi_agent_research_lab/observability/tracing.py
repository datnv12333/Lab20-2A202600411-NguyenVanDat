"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from time import perf_counter
from typing import Any, Optional


def _langsmith_enabled() -> bool:
    value = os.getenv("LANGSMITH_TRACING", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def trace_span(name: str, attributes: Optional[dict[str, Any]] = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        if not _langsmith_enabled():
            yield span
            return

        try:
            from langsmith import trace as langsmith_trace
        except Exception:
            yield span
            return

        with langsmith_trace(name, run_type="chain") as run:
            run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
            if run_id is not None:
                span["langsmith_run_id"] = str(run_id)
            yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
