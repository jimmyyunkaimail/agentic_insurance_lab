from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.material_understanding import MaterialUnderstandingAgent
from app.agents.openai_runtime import runtime_status
from app.agents.task_attribution import TaskAttributionAgent
from app.schemas import (
    CandidateTask,
    DecisionLog,
    MaterialUnderstandingRequest,
    MaterialUnderstandingResult,
    MessageEvent,
    PolicySnapshot,
    TaskAttributionDecision,
    TaskAttributionRequest,
)
from app.services.policy import PolicyEngine
from app.services.storage import JsonStore

app = FastAPI(title="车险出单双 Agent 本地验证框架", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

store = JsonStore()
material_agent = MaterialUnderstandingAgent()
task_agent = TaskAttributionAgent()
policy_engine = PolicyEngine()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runtime")
def runtime() -> dict:
    return runtime_status().__dict__


@app.post("/agents/material-understanding/run", response_model=MaterialUnderstandingResult)
def run_material_agent(request: MaterialUnderstandingRequest) -> MaterialUnderstandingResult:
    event = request.to_event()
    store.save_event(event)
    result = material_agent.run(event)
    store.save_documents(event.event_id, result.documents)
    store.save_evidences(event.event_id, result.evidence_list)
    store.save_decision(
        DecisionLog(
            event_id=event.event_id,
            agent_name=result.agent_name,
            request_payload=request.model_dump(mode="json"),
            response_payload=result.model_dump(mode="json"),
        )
    )
    return result


@app.post("/agents/task-attribution/run", response_model=TaskAttributionDecision)
def run_task_agent(request: TaskAttributionRequest) -> TaskAttributionDecision:
    result = task_agent.run(request)
    policy_result = policy_engine.check_attribution(result)
    store.save_decision(
        DecisionLog(
            event_id=request.event.event_id,
            agent_name=result.agent_name,
            request_payload=request.model_dump(mode="json"),
            response_payload=result.model_dump(mode="json"),
            policy_result=policy_result,
        )
    )
    return result


@app.post("/events/ingest")
def ingest_event(request: MaterialUnderstandingRequest) -> dict:
    event = request.to_event()
    store.save_event(event)
    material_result = material_agent.run(event)
    store.save_documents(event.event_id, material_result.documents)
    store.save_evidences(event.event_id, material_result.evidence_list)

    active_tasks = [
        task
        for task in store.list_tasks(event.conversation_id)
        if str(task.status.value if hasattr(task.status, "value") else task.status) == "active"
    ]
    candidate_tasks = [
        CandidateTask(
            task_id=task.task_id,
            score=0,
            match_reason="same_conversation",
            stage=task.stage,
            status=task.status,
            last_active_at=task.last_active_at,
            task=task,
        )
        for task in active_tasks
    ]
    attribution_request = TaskAttributionRequest(
        event=event,
        new_evidence=material_result.evidence_list,
        active_tasks=active_tasks,
        candidate_tasks=candidate_tasks,
        policy_snapshot=PolicySnapshot(),
    )
    attribution_result = task_agent.run(attribution_request)
    policy_result = policy_engine.check_attribution(attribution_result)
    store.save_decision(
        DecisionLog(
            event_id=event.event_id,
            agent_name=material_result.agent_name,
            request_payload=request.model_dump(mode="json"),
            response_payload=material_result.model_dump(mode="json"),
        )
    )
    store.save_decision(
        DecisionLog(
            event_id=event.event_id,
            agent_name=attribution_result.agent_name,
            request_payload=attribution_request.model_dump(mode="json"),
            response_payload=attribution_result.model_dump(mode="json"),
            policy_result=policy_result,
        )
    )
    return {
        "event": event,
        "material_understanding": material_result,
        "task_attribution": attribution_result,
        "policy_result": policy_result,
    }


@app.get("/events/{event_id}")
def get_event(event_id: str) -> dict:
    event = store.get_event(event_id)
    return {
        "event": event,
        "evidence": store.list_evidences_for_event(event_id),
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    return {"task": store.get_task(task_id)}


@app.post("/evals/material/replay")
def replay_material() -> dict:
    from scripts.replay_cases import run_material_suite

    return run_material_suite(store=JsonStore("data/eval_material.db"))


@app.post("/evals/attribution/replay")
def replay_attribution() -> dict:
    from scripts.replay_cases import run_attribution_suite

    return run_attribution_suite(store=JsonStore("data/eval_attribution.db"))


@app.post("/evals/production/replay")
def replay_production() -> dict:
    from scripts.replay_cases import run_production_suite

    return run_production_suite(store=JsonStore("data/eval_production.db"))
