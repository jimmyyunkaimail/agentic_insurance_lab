from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.schemas import DecisionLog, DocumentResult, EvidenceItem, MessageEvent, TaskRecord


DEFAULT_DB_PATH = Path("data/agentic_insurance.db")


class JsonStore:
    """Small SQLite-backed JSON store for local validation.

    The goal is fast iteration and replayability, not production persistence.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists events (
                    id text primary key,
                    conversation_id text,
                    payload text not null
                );
                create table if not exists documents (
                    id text primary key,
                    event_id text,
                    payload text not null
                );
                create table if not exists evidences (
                    id text primary key,
                    event_id text,
                    field_name text,
                    normalized_value text,
                    payload text not null
                );
                create table if not exists tasks (
                    id text primary key,
                    conversation_id text,
                    status text,
                    stage text,
                    payload text not null
                );
                create table if not exists decisions (
                    id text primary key,
                    event_id text,
                    agent_name text,
                    payload text not null
                );
                """
            )

    def reset(self) -> None:
        with self.connect() as conn:
            for table in ["events", "documents", "evidences", "tasks", "decisions"]:
                conn.execute(f"delete from {table}")

    @staticmethod
    def _dump(model_or_dict: BaseModel | dict[str, Any]) -> str:
        if isinstance(model_or_dict, BaseModel):
            return model_or_dict.model_dump_json()
        return json.dumps(model_or_dict, ensure_ascii=False)

    @staticmethod
    def _load(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return json.loads(row["payload"])

    def save_event(self, event: MessageEvent) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into events (id, conversation_id, payload)
                values (?, ?, ?)
                """,
                (event.event_id, event.conversation_id, self._dump(event)),
            )

    def get_event(self, event_id: str) -> MessageEvent | None:
        with self.connect() as conn:
            row = conn.execute("select payload from events where id=?", (event_id,)).fetchone()
        payload = self._load(row)
        return MessageEvent.model_validate(payload) if payload else None

    def save_documents(self, event_id: str, documents: list[DocumentResult]) -> None:
        with self.connect() as conn:
            for document in documents:
                conn.execute(
                    """
                    insert or replace into documents (id, event_id, payload)
                    values (?, ?, ?)
                    """,
                    (document.document_id, event_id, self._dump(document)),
                )

    def save_evidences(self, event_id: str, evidences: list[EvidenceItem]) -> None:
        with self.connect() as conn:
            for evidence in evidences:
                conn.execute(
                    """
                    insert or replace into evidences
                    (id, event_id, field_name, normalized_value, payload)
                    values (?, ?, ?, ?, ?)
                    """,
                    (
                        evidence.evidence_id,
                        event_id,
                        evidence.field_name,
                        evidence.normalized_value,
                        self._dump(evidence),
                    ),
                )

    def list_evidences_for_event(self, event_id: str) -> list[EvidenceItem]:
        with self.connect() as conn:
            rows = conn.execute(
                "select payload from evidences where event_id=? order by id", (event_id,)
            ).fetchall()
        return [EvidenceItem.model_validate(json.loads(row["payload"])) for row in rows]

    def save_task(self, task: TaskRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into tasks (id, conversation_id, status, stage, payload)
                values (?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.conversation_id,
                    str(task.status.value if hasattr(task.status, "value") else task.status),
                    str(task.stage.value if hasattr(task.stage, "value") else task.stage),
                    self._dump(task),
                ),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self.connect() as conn:
            row = conn.execute("select payload from tasks where id=?", (task_id,)).fetchone()
        payload = self._load(row)
        return TaskRecord.model_validate(payload) if payload else None

    def list_tasks(self, conversation_id: str | None = None) -> list[TaskRecord]:
        sql = "select payload from tasks"
        params: tuple[Any, ...] = ()
        if conversation_id:
            sql += " where conversation_id=?"
            params = (conversation_id,)
        sql += " order by id"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [TaskRecord.model_validate(json.loads(row["payload"])) for row in rows]

    def save_decision(self, decision: DecisionLog) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert or replace into decisions (id, event_id, agent_name, payload)
                values (?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.event_id,
                    decision.agent_name,
                    self._dump(decision),
                ),
            )

    def get_decision(self, decision_id: str) -> DecisionLog | None:
        with self.connect() as conn:
            row = conn.execute("select payload from decisions where id=?", (decision_id,)).fetchone()
        payload = self._load(row)
        return DecisionLog.model_validate(payload) if payload else None

