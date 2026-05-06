from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from app.schemas import ServiceTraceStep


class TraceRecorder:
    """Structured trace for replaying validation flow and product reasoning."""

    def __init__(self, phase_prefix: str):
        self.phase_prefix = phase_prefix
        self._steps: list[ServiceTraceStep] = []

    def add(
        self,
        *,
        phase: str,
        step_name: str,
        action: str,
        rationale: str | None = None,
        input_snapshot: dict[str, Any] | None = None,
        output_snapshot: dict[str, Any] | None = None,
        decision_basis: list[str] | None = None,
        branch: str | None = None,
        risk_notes: list[str] | None = None,
        started_at: str | None = None,
        duration_ms: float | None = None,
    ) -> ServiceTraceStep:
        step = ServiceTraceStep(
            phase=f"{self.phase_prefix}.{phase}",
            step_name=step_name,
            action=action,
            rationale=rationale,
            input_snapshot=self._safe_dict(input_snapshot or {}),
            output_snapshot=self._safe_dict(output_snapshot or {}),
            decision_basis=decision_basis or [],
            branch=branch,
            risk_notes=risk_notes or [],
            started_at=started_at or self._now(),
            ended_at=self._now(),
            duration_ms=duration_ms,
        )
        self._steps.append(step)
        return step

    def time_step(
        self,
        *,
        phase: str,
        step_name: str,
        action: str,
        rationale: str | None = None,
        input_snapshot: dict[str, Any] | None = None,
        output_snapshot: dict[str, Any] | None = None,
        decision_basis: list[str] | None = None,
        branch: str | None = None,
        risk_notes: list[str] | None = None,
        started_perf: float,
        started_at: str,
    ) -> ServiceTraceStep:
        return self.add(
            phase=phase,
            step_name=step_name,
            action=action,
            rationale=rationale,
            input_snapshot=input_snapshot,
            output_snapshot=output_snapshot,
            decision_basis=decision_basis,
            branch=branch,
            risk_notes=risk_notes,
            started_at=started_at,
            duration_ms=round((perf_counter() - started_perf) * 1000, 2),
        )

    def export(self) -> list[ServiceTraceStep]:
        return list(self._steps)

    @staticmethod
    def mark_start() -> tuple[str, float]:
        return TraceRecorder._now(), perf_counter()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _safe_dict(cls, payload: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in payload.items():
            safe[str(key)] = cls._safe_value(value)
        return safe

    @classmethod
    def _safe_value(cls, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): cls._safe_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._safe_value(item) for item in value]
        if isinstance(value, tuple):
            return [cls._safe_value(item) for item in value]
        if isinstance(value, set):
            return sorted(cls._safe_value(item) for item in value)
        if hasattr(value, "value"):
            return value.value
        return value
