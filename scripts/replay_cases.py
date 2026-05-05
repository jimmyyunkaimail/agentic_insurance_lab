from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.material_understanding import MaterialUnderstandingAgent
from app.agents.task_attribution import TaskAttributionAgent
from app.schemas import (
    AttachmentRef,
    CandidateTask,
    ConversationMemory,
    MaterialUnderstandingRequest,
    MessageEvent,
    MessageType,
    PolicySnapshot,
    TaskAttributionRequest,
    TaskRecord,
    TaskStage,
)
from app.services.storage import JsonStore
from scripts.seed_demo_data import seed


@dataclass
class MaterialCase:
    name: str
    request: MaterialUnderstandingRequest
    expected_fields: set[str]
    expected_document_type: str | None = None


@dataclass
class AttributionCase:
    name: str
    event: MessageEvent
    expected_decision: str
    expected_task_id: str | None = None


@dataclass
class ProductionAttributionCase:
    name: str
    event: MessageEvent
    tasks: list[TaskRecord]
    memory: ConversationMemory
    expected_decision: str
    expected_task_id: str | None = None
    expected_task_ids: list[str] | None = None


def material_cases() -> list[MaterialCase]:
    return [
        MaterialCase(
            name="文本车架号",
            request=MaterialUnderstandingRequest(content_text="车架号 LFMKN5BF2N3276921"),
            expected_fields={"frameNo"},
        ),
        MaterialCase(
            name="文本三者修改",
            request=MaterialUnderstandingRequest(content_text="三者改成300万，不要288"),
            expected_fields={"thirdPartyLiabilityInsuranceForMotorVehicles", "unAutoProductInfo", "intentHint"},
        ),
        MaterialCase(
            name="催促消息",
            request=MaterialUnderstandingRequest(content_text="好了没，到哪步了"),
            expected_fields={"intentHint"},
        ),
        MaterialCase(
            name="手机号",
            request=MaterialUnderstandingRequest(content_text="车主手机号 13800138000"),
            expected_fields={"ownerMobile"},
        ),
        MaterialCase(
            name="行驶证附件分类",
            request=MaterialUnderstandingRequest(
                message_type=MessageType.image,
                attachments=[
                    {
                        "attachment_id": "att_license",
                        "file_type": "image",
                        "file_ref": "storage/attachments/行驶证正面.jpg",
                    }
                ],
            ),
            expected_fields=set(),
            expected_document_type="vehicle_license",
        ),
    ]


def production_material_cases() -> list[MaterialCase]:
    return [
        MaterialCase(
            name="生产话术-确认出单",
            request=MaterialUnderstandingRequest(content_text="确认"),
            expected_fields={"intentHint"},
        ),
        MaterialCase(
            name="生产话术-核保一下",
            request=MaterialUnderstandingRequest(content_text="核保一下"),
            expected_fields={"intentHint"},
        ),
        MaterialCase(
            name="生产话术-非车改588",
            request=MaterialUnderstandingRequest(content_text="非车改588"),
            expected_fields={"unAutoProductInfo", "intentHint"},
        ),
        MaterialCase(
            name="生产话术-发票",
            request=MaterialUnderstandingRequest(content_text="南京华夏钢铁有限公司 发票"),
            expected_fields={"intentHint"},
        ),
        MaterialCase(
            name="生产状态-报价完成待确认",
            request=MaterialUnderstandingRequest(
                content_text="0310920/04601070 张彩苓-苏A-AL277-4499579 报价完成，商业险保费1973.31 交强险保费1130.0，出单请说确认"
            ),
            expected_fields={"vehicleLicenseNo", "quoteFlowNo", "commercialPremium", "compulsoryPremium", "intentHint"},
        ),
        MaterialCase(
            name="生产附件-未知微信图片不应伪装成已解析材料",
            request=MaterialUnderstandingRequest(
                message_type=MessageType.image,
                attachments=[
                    AttachmentRef(file_type="image", file_ref="微信图片_20260430111850_3687_69.jpg"),
                    AttachmentRef(file_type="image", file_ref="微信图片_20260430111843_3686_69.jpg"),
                ],
            ),
            expected_fields=set(),
            expected_document_type="unknown",
        ),
    ]


def attribution_cases() -> list[AttributionCase]:
    return [
        AttributionCase(
            name="车架号一致归属当前报价任务",
            event=MessageEvent(content_text="车架号 LFMKN5BF2N3276921"),
            expected_decision="attach_to_active_task",
            expected_task_id="task_active_quote",
        ),
        AttributionCase(
            name="报价阶段修改三者",
            event=MessageEvent(content_text="三者改成300万"),
            expected_decision="attach_to_active_task",
            expected_task_id="task_active_quote",
        ),
        AttributionCase(
            name="不同车辆新建任务",
            event=MessageEvent(content_text="车架号 LGWEF6A59NH123456"),
            expected_decision="create_new_task",
            expected_task_id=None,
        ),
        AttributionCase(
            name="多个活跃任务单独发身份证",
            event=MessageEvent(content_text="车主张三"),
            expected_decision="hold_for_confirmation",
            expected_task_id=None,
        ),
        AttributionCase(
            name="二维码过期归属支付/投保任务",
            event=MessageEvent(content_text="二维码过期了，重新发我"),
            expected_decision="attach_to_active_task",
            expected_task_id="task_active_insure",
        ),
    ]


def production_attribution_cases() -> list[ProductionAttributionCase]:
    quote_task = TaskRecord(
        task_id="Task_PA_AL277",
        conversation_id="conv_production",
        stage=TaskStage.quote_confirming,
        vehicle_facts={"vehicleLicenseNo": "苏AAL277", "frameNo": "LVGEN56A6MG509542"},
        quote_plan_summary="商业险1973.31，交强险1130.0，待确认",
        last_bot_action="报价完成，出单请说确认",
    )
    steel_task = TaskRecord(
        task_id="Task_PA_803DX",
        conversation_id="conv_production",
        stage=TaskStage.completed,
        vehicle_facts={"vehicleLicenseNo": "苏A803DX", "frameNo": "LVGB674K3KG012582"},
        quote_plan_summary="商业险2678.24，交强险1160.0",
        last_bot_action="保单已发送",
    )
    payment_task = TaskRecord(
        task_id="Task_PA_HL37055",
        conversation_id="conv_production",
        stage=TaskStage.payment,
        vehicle_facts={"frameNo": "LVGDB6GF9TG019339"},
        quote_plan_summary="支付二维码已发送",
    )
    another_active = TaskRecord(
        task_id="Task_PA_V326S",
        conversation_id="conv_production",
        stage=TaskStage.quote_confirming,
        vehicle_facts={"vehicleLicenseNo": "苏AV326S", "frameNo": "LVGCJE734EG019826"},
    )
    old_focus = TaskRecord(
        task_id="Task_000001",
        conversation_id="conv_production",
        stage=TaskStage.quote_preparing,
        vehicle_facts={"frameNo": "LFMKN5BF2N3276921"},
        quote_plan_summary="三者300万，不要288",
    )
    return [
        ProductionAttributionCase(
            name="报价完成后用户回复确认",
            event=MessageEvent(conversation_id="conv_production", content_text="确认"),
            tasks=[quote_task],
            memory=ConversationMemory(focused_task_id="Task_PA_AL277"),
            expected_decision="attach_to_active_task",
            expected_task_id="Task_PA_AL277",
        ),
        ProductionAttributionCase(
            name="省略式非车修改归属焦点任务",
            event=MessageEvent(conversation_id="conv_production", content_text="非车改588"),
            tasks=[quote_task],
            memory=ConversationMemory(focused_task_id="Task_PA_AL277"),
            expected_decision="attach_to_active_task",
            expected_task_id="Task_PA_AL277",
        ),
        ProductionAttributionCase(
            name="未知图片附件在多任务并发中等待确认",
            event=MessageEvent(
                conversation_id="conv_production",
                message_type=MessageType.image,
                attachments=[
                    AttachmentRef(file_type="image", file_ref="微信图片_20260430111850_3687_69.jpg"),
                    AttachmentRef(file_type="image", file_ref="微信图片_20260430111843_3686_69.jpg"),
                ],
            ),
            tasks=[quote_task, another_active],
            memory=ConversationMemory(),
            expected_decision="hold_for_confirmation",
        ),
        ProductionAttributionCase(
            name="焦点任务之后出现新VIN应新增独立任务",
            event=MessageEvent(
                conversation_id="conv_production",
                content_text="车架号 LFMKN5BF2N3276923，三者改成500万，先芸芸 18305194341",
            ),
            tasks=[old_focus],
            memory=ConversationMemory(focused_task_id="Task_000001"),
            expected_decision="create_new_task",
        ),
        ProductionAttributionCase(
            name="支付完成状态归属支付阶段任务",
            event=MessageEvent(conversation_id="conv_production", content_text="LVGDB6GF9TG019339 支付已经完成了"),
            tasks=[payment_task],
            memory=ConversationMemory(focused_task_id="Task_PA_HL37055"),
            expected_decision="attach_to_active_task",
            expected_task_id="Task_PA_HL37055",
        ),
        ProductionAttributionCase(
            name="发票省略话术归属已完成任务",
            event=MessageEvent(conversation_id="conv_production", content_text="南京华夏钢铁有限公司 发票"),
            tasks=[steel_task],
            memory=ConversationMemory(focused_task_id="Task_PA_803DX"),
            expected_decision="attach_to_active_task",
            expected_task_id="Task_PA_803DX",
        ),
    ]


def run_material_suite(store: JsonStore | None = None) -> dict:
    agent = MaterialUnderstandingAgent(use_model=False)
    total = 0
    passed = 0
    details = []
    for case in material_cases():
        total += 1
        result = agent.run(case.request)
        fields = {item.field_name for item in result.evidence_list}
        document_types = {document.document_type for document in result.documents}
        ok = case.expected_fields.issubset(fields)
        if case.expected_document_type:
            ok = ok and case.expected_document_type in document_types
        passed += int(ok)
        details.append(
            {
                "name": case.name,
                "passed": ok,
                "fields": sorted(fields),
                "documents": sorted(document_types),
            }
        )
    return {"suite": "material_v1", "total": total, "passed": passed, "details": details}


def _active_and_candidates(store: JsonStore, conversation_id: str) -> tuple[list, list]:
    tasks = store.list_tasks(conversation_id)
    active = [task for task in tasks if task.status.value == "active"]
    candidates = [
        CandidateTask(
            task_id=task.task_id,
            score=0,
            match_reason="same_conversation",
            stage=task.stage,
            status=task.status,
            last_active_at=task.last_active_at,
            task=task,
        )
        for task in tasks
    ]
    return active, candidates


def run_attribution_suite(store: JsonStore | None = None) -> dict:
    store = seed(store)
    material_agent = MaterialUnderstandingAgent(use_model=False)
    attribution_agent = TaskAttributionAgent(use_model=False)
    total = 0
    passed = 0
    details = []
    for case in attribution_cases():
        total += 1
        material_result = material_agent.run(case.event)
        active, candidates = _active_and_candidates(store, case.event.conversation_id)
        request = TaskAttributionRequest(
            event=case.event,
            new_evidence=material_result.evidence_list,
            active_tasks=active,
            candidate_tasks=candidates,
            policy_snapshot=PolicySnapshot(),
        )
        decision = attribution_agent.run(request)
        ok = decision.decision.value == case.expected_decision
        if case.expected_task_id:
            ok = ok and decision.selected_task_id == case.expected_task_id
        passed += int(ok)
        details.append(
            {
                "name": case.name,
                "passed": ok,
                "decision": decision.decision.value,
                "selected_task_id": decision.selected_task_id,
                "reason": decision.business_reason,
            }
        )
    return {"suite": "attribution_v1", "total": total, "passed": passed, "details": details}


def run_production_suite(store: JsonStore | None = None) -> dict:
    material_agent = MaterialUnderstandingAgent(use_model=False)
    attribution_agent = TaskAttributionAgent(use_model=False)
    total = 0
    passed = 0
    details = []

    for case in production_material_cases():
        total += 1
        result = material_agent.run(case.request)
        fields = {item.field_name for item in result.evidence_list}
        document_types = {document.document_type for document in result.documents}
        ok = case.expected_fields.issubset(fields)
        if case.expected_document_type:
            ok = ok and case.expected_document_type in document_types
        passed += int(ok)
        details.append(
            {
                "name": case.name,
                "type": "material",
                "passed": ok,
                "intent": result.current_intent,
                "fields": sorted(fields),
                "documents": sorted(document_types),
            }
        )

    for case in production_attribution_cases():
        total += 1
        material_result = material_agent.run(case.event)
        candidates = [
            CandidateTask(
                task_id=task.task_id,
                match_reason="production_case_context",
                stage=task.stage,
                status=task.status,
                last_active_at=task.last_active_at,
                task=task,
            )
            for task in case.tasks
        ]
        request = TaskAttributionRequest(
            event=case.event,
            new_evidence=material_result.evidence_list,
            active_tasks=case.tasks,
            candidate_tasks=candidates,
            conversation_memory=case.memory,
            policy_snapshot=PolicySnapshot(),
        )
        decision = attribution_agent.run(request)
        ok = decision.decision.value == case.expected_decision
        if case.expected_task_id:
            ok = ok and decision.selected_task_id == case.expected_task_id
        if case.expected_task_ids:
            ok = ok and decision.selected_task_ids == case.expected_task_ids
        passed += int(ok)
        details.append(
            {
                "name": case.name,
                "type": "attribution",
                "passed": ok,
                "decision": decision.decision.value,
                "selected_task_id": decision.selected_task_id,
                "selected_task_ids": decision.selected_task_ids,
                "reason": decision.business_reason,
            }
        )
    return {"suite": "production_case_20260504_101031", "total": total, "passed": passed, "details": details}


def main() -> None:
    suites: list[Callable[[], dict]] = [run_material_suite, run_attribution_suite, run_production_suite]
    for suite in suites:
        result = suite()
        print(f"{result['suite']}: {result['passed']}/{result['total']}")
        for detail in result["details"]:
            status = "PASS" if detail["passed"] else "FAIL"
            print(f"  {status} {detail['name']} -> {detail}")


if __name__ == "__main__":
    main()
