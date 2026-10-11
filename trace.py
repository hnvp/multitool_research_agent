from __future__ import annotations
import re
import json
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any


@dataclass
class TraceEvent:
    tool: str
    args: dict[str, Any]
    status: str
    latency_ms: float
    output_preview: str
    error: str | None = None


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(
        self,
        tool: str,
        args: dict[str, Any],
        status: str,
        latency_ms: float,
        output: Any = "",
        error: str | None = None,
    ) -> None:
        output_text = str(output)

        self.events.append(
            TraceEvent(
                tool=tool,
                args=args,
                status=status,
                latency_ms=round(
                    latency_ms,
                    2,
                ),
                output_preview=output_text[:1000],
                error=error,
            )
        )

    def as_dicts(
        self,
    ) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]

    def total_latency_ms(self) -> float:
        return round(
            sum(event.latency_ms for event in self.events),
            2,
        )

    def as_json(self) -> str:
        return json.dumps(
            self.as_dicts(),
            ensure_ascii=False,
            indent=2,
        )


_CURRENT_TRACE: ContextVar[TraceRecorder | None] = ContextVar(
    "current_trace",
    default=None,
)


def set_current_trace(
    trace: TraceRecorder,
):
    return _CURRENT_TRACE.set(trace)


def reset_current_trace(token) -> None:
    _CURRENT_TRACE.reset(token)


def get_current_trace() -> TraceRecorder | None:
    return _CURRENT_TRACE.get()


_SECRET_RE = re.compile(r"(?i)\b(appid|api_key|apikey|key|token)=[^&\s]+")


def _redact(message: str) -> str:
    return _SECRET_RE.sub(r"\1=***", message)


def execute_with_trace(
    tool_name: str,
    fn,
    **kwargs: Any,
) -> Any:
    """Execute a tool and record its execution metadata."""
    started = perf_counter()

    try:
        result = fn(**kwargs)

        latency_ms = (perf_counter() - started) * 1000

        trace = get_current_trace()

        if trace:
            trace.record(
                tool_name,
                kwargs,
                "success",
                latency_ms,
                result,
            )

        return result

    except Exception as exc:
        latency_ms = (perf_counter() - started) * 1000

        trace = get_current_trace()

        if trace:
            trace.record(
                tool_name,
                kwargs,
                "error",
                latency_ms,
                output="",
                error=_redact(str(exc)),
            )

        raise


def run_tool_safely(tool_name: str, fn, **kwargs: Any) -> str:
    """Execute a tool and convert any failure into a structured,
    model-readable failure marker instead of raising.

    Unlike execute_with_trace (which always raises on failure so callers
    that need real exceptions still get them), this wrapper is for
    tool-facing code: it guarantees a string return value so the agent
    always receives a well-formed tool result, success or failure.
    """
    try:
        return str(execute_with_trace(tool_name, fn, **kwargs))
    except Exception as exc:
        return f"TOOL_ERROR: {tool_name} failed: {_redact(str(exc))[:300]}"
