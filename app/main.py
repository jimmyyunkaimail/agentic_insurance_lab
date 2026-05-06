from __future__ import annotations

from fastapi import File, FastAPI, HTTPException, UploadFile
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
from app.services.attachment_storage import AttachmentStorage
from app.services.storage import JsonStore
from app.services.tracing import TraceRecorder

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
attachment_storage = AttachmentStorage()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runtime")
def runtime() -> dict:
    return runtime_status().__dict__


@app.post("/attachments/upload")
async def upload_attachment(file: UploadFile = File(...)) -> dict:
    attachment = await attachment_storage.save_upload(file)
    return {"attachment": attachment}


@app.get("/attachments/{attachment_id}/download")
def download_attachment(attachment_id: str) -> FileResponse:
    path = attachment_storage.resolve_by_id(attachment_id)
    if not path:
        raise HTTPException(status_code=404, detail="附件不存在。")
    return FileResponse(path)


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
            trace_log=result.trace_log,
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
            trace_log=[
                *result.trace_log,
                TraceRecorder("policy")
                .add(
                    phase="guardrail",
                    step_name="策略护栏检查",
                    action="检查任务归属决策是否允许自动执行",
                    input_snapshot={"decision": result.model_dump(mode="json")},
                    output_snapshot={"policy_result": policy_result.model_dump(mode="json")},
                    decision_basis=[
                        "等待确认、转人工、高风险或冲突事实禁止自动写入。",
                    ],
                    branch=policy_result.result,
                    risk_notes=[policy_result.reason]
                    if policy_result.result != "approved"
                    else [],
                )
            ],
        )
    )
    return result


@app.post("/events/ingest")
def ingest_event(request: MaterialUnderstandingRequest) -> dict:
    service_trace = TraceRecorder("service_flow")
    event = request.to_event()
    store.save_event(event)
    service_trace.add(
        phase="event",
        step_name="接入消息",
        action="保存 MessageEvent（消息事件）并启动端到端服务流程",
        input_snapshot={"request": request.model_dump(mode="json")},
        output_snapshot={"event": event.model_dump(mode="json")},
    )
    material_result = material_agent.run(event)
    store.save_documents(event.event_id, material_result.documents)
    store.save_evidences(event.event_id, material_result.evidence_list)
    service_trace.add(
        phase="material",
        step_name="材料理解完成",
        action="落库 Document（单证）和 Evidence（证据）",
        output_snapshot={
            "document_count": len(material_result.documents),
            "evidence_count": len(material_result.evidence_list),
            "next_route": material_result.material_action_hint.next_route,
        },
        decision_basis=[material_result.material_action_hint.reason],
        branch=material_result.material_action_hint.next_route,
    )

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
    service_trace.add(
        phase="attribution",
        step_name="任务归属完成",
        action="生成归属决策并执行策略护栏检查",
        input_snapshot={
            "active_task_ids": [task.task_id for task in active_tasks],
            "candidate_task_ids": [task.task_id for task in candidate_tasks],
        },
        output_snapshot={
            "decision": attribution_result.decision.value,
            "selected_task_id": attribution_result.selected_task_id,
            "selected_task_ids": attribution_result.selected_task_ids,
            "policy_result": policy_result.model_dump(mode="json"),
        },
        decision_basis=[attribution_result.business_reason, policy_result.reason],
        branch=policy_result.result,
        risk_notes=attribution_result.entity_graph_summary.conflicts,
    )
    store.save_decision(
        DecisionLog(
            event_id=event.event_id,
            agent_name=material_result.agent_name,
            request_payload=request.model_dump(mode="json"),
            response_payload=material_result.model_dump(mode="json"),
            trace_log=[*service_trace.export(), *material_result.trace_log],
        )
    )
    store.save_decision(
        DecisionLog(
            event_id=event.event_id,
            agent_name=attribution_result.agent_name,
            request_payload=attribution_request.model_dump(mode="json"),
            response_payload=attribution_result.model_dump(mode="json"),
            policy_result=policy_result,
            trace_log=[*service_trace.export(), *attribution_result.trace_log],
        )
    )
    return {
        "event": event,
        "material_understanding": material_result,
        "task_attribution": attribution_result,
        "policy_result": policy_result,
        "service_trace": service_trace.export(),
    }


@app.get("/events/{event_id}")
def get_event(event_id: str) -> dict:
    event = store.get_event(event_id)
    return {
        "event": event,
        "evidence": store.list_evidences_for_event(event_id),
        "decisions": store.list_decisions_for_event(event_id),
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
