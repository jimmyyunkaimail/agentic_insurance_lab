from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.agents.openai_runtime import TASK_ATTRIBUTION_AGENT_INSTRUCTIONS, ModelRuntime
from app.knowledge import STAGE_HIGH_RISK
from app.schemas import (
    AttributionDecision,
    CandidateTask,
    EntityGraphSummary,
    EvidenceItem,
    RiskLevel,
    StateUpdateSuggestion,
    TargetTaskCandidate,
    TaskAttributionDecision,
    TaskAttributionRequest,
    TaskRecord,
)


@dataclass
class EvidenceView:
    vehicles: dict[str, set[str]]
    parties: dict[str, set[str]]
    intents: set[str]
    quote_fields: set[str]
    evidence_ids: list[str]
    weak_only: bool


class TaskAttributionAgent:
    """Contextual task router for local validation."""

    agent_name = "task_attribution_agent"

    def __init__(self, use_model: bool | None = None, runtime: ModelRuntime | None = None):
        self.use_model = use_model
        self.runtime = runtime or ModelRuntime()
        self._last_model_runtime: dict[str, Any] = {}

    def run(self, request: TaskAttributionRequest) -> TaskAttributionDecision:
        _, model_runtime = self._try_model_reasoning(request)
        self._last_model_runtime = model_runtime

        view = self._view(request.new_evidence)
        candidates = self._score_candidates(request, view)
        conflicts = self._find_conflicts(request.active_tasks, view)
        detected_entities = self._detected_entities(view)
        used_relationships = self._relationships(view, request)
        frame_values = self._frame_values_in_order(request.new_evidence, view)

        if self._should_ignore(view, request):
            return self._decision(
                request,
                AttributionDecision.ignore_or_chat,
                None,
                0.65,
                RiskLevel.low,
                "未识别到可影响车险作业任务的业务证据。",
                detected_entities,
                used_relationships,
                conflicts,
                candidates,
                "ignore",
            )

        if self._is_unresolved_attachment_material(request, view):
            focus_task = self._focused_task(request)
            if focus_task:
                return self._decision(
                    request,
                    AttributionDecision.attach_to_active_task,
                    focus_task.task_id,
                    0.74,
                    RiskLevel.medium,
                    f"当前消息是附件材料，但尚未抽取到车辆或人员实体；依据引用/会话焦点任务 {focus_task.task_id} 先作为待解析材料归属，禁止写入事实。",
                    detected_entities + ["document"],
                    used_relationships + ["responds_to", "derived_from"],
                    [],
                    candidates,
                    "attach_material_pending_extraction",
                    next_action="attach_material_pending_extraction",
                    fact_write_allowed=False,
                )
            if len(request.active_tasks) == 1:
                task = request.active_tasks[0]
                return self._decision(
                    request,
                    AttributionDecision.attach_to_active_task,
                    task.task_id,
                    0.66,
                    RiskLevel.medium,
                    f"当前消息是附件材料但缺少可抽取实体；会话内仅有一个活跃任务 {task.task_id}，可先作为待解析材料挂接，禁止写入事实。",
                    detected_entities + ["document"],
                    used_relationships + ["derived_from"],
                    [],
                    candidates,
                    "attach_material_pending_extraction",
                    confirmation_question="请确认这份附件材料是否用于当前唯一活跃任务？",
                    next_action="attach_material_pending_extraction",
                    fact_write_allowed=False,
                )
            return self._decision(
                request,
                AttributionDecision.hold_for_confirmation,
                None,
                0.56,
                RiskLevel.medium,
                "当前消息是附件材料，但未解析出车牌、VIN（车架号）、证件号等强实体；多任务并发场景下不能按闲聊忽略，也不能自动归属。",
                detected_entities + ["document"],
                used_relationships + ["derived_from"],
                [],
                candidates,
                "ask_user_to_clarify_vehicle_for_attachment",
                confirmation_question="请说明这份附件材料用于哪台车，或补充车牌/VIN（车架号）。",
            )

        if self._is_recent_task_set_update(request, view):
            selected_task_ids = self._valid_recent_task_set_ids(request)
            return self._decision(
                request,
                AttributionDecision.attach_to_task_set,
                None,
                0.86,
                RiskLevel.low,
                "消息是省略式批量报价方案修改，依据最近批量拆分出的独立任务集合进行归属。",
                detected_entities,
                used_relationships + ["modifies", "wakes"],
                [],
                candidates,
                "batch_update_recent_task_set",
                next_stage="quote_modifying",
                next_action="route_to_quote_precheck",
                fact_write_allowed=True,
                selected_task_ids=selected_task_ids,
            )

        focus_task = self._focused_task(request)
        if focus_task and self._can_attach_to_focus(view):
            next_stage, next_action = self._next_stage_and_action(
                view,
                CandidateTask(
                    task_id=focus_task.task_id,
                    score=88,
                    match_reason="conversation_focus_task",
                    stage=focus_task.stage,
                    status=focus_task.status,
                    last_active_at=focus_task.last_active_at,
                    task=focus_task,
                ),
            )
            return self._decision(
                request,
                AttributionDecision.attach_to_active_task,
                focus_task.task_id,
                0.88,
                RiskLevel.low,
                f"当前消息是省略式业务指令，未出现新的车辆唯一标识；依据会话焦点任务 {focus_task.task_id} 归属。",
                detected_entities,
                used_relationships + ["modifies", "wakes"],
                [],
                candidates,
                next_action or "route_to_quote_precheck",
                next_stage=next_stage,
                fact_write_allowed=True,
            )

        best_precheck = candidates[0] if candidates else None
        if len(frame_values) > 1 and (not best_precheck or best_precheck.score < 70):
            return self._decision(
                request,
                AttributionDecision.create_multiple_tasks,
                None,
                0.84,
                RiskLevel.low,
                f"当前消息一次性提供 {len(frame_values)} 个 VIN（车架号），未召回可信旧任务；按新批量报价拆成多个独立单任务。",
                detected_entities,
                used_relationships + ["starts"],
                [],
                candidates,
                "create_independent_task_drafts",
                next_stage="quote_preparing",
                next_action="create_independent_tasks",
                fact_write_allowed=True,
                new_task_hint={
                    "task_mode": "independent_single_tasks",
                    "id_policy": "Task_000001_incremental",
                    "vehicle_mentions": [{"frameNo": value} for value in frame_values],
                    "no_parent_task": True,
                },
            )

        if len(frame_values) > 1 and best_precheck and best_precheck.score >= 70:
            return self._decision(
                request,
                AttributionDecision.hold_for_confirmation,
                best_precheck.task_id,
                0.72,
                RiskLevel.medium,
                "当前消息包含多个 VIN（车架号）且部分信息可能命中旧任务，需确认是批量新增还是补充旧任务。",
                detected_entities,
                used_relationships + ["contradicts"],
                conflicts,
                candidates,
                "ask_user_to_confirm_batch_scope",
                confirmation_question="这几台车是要作为新的批量报价任务分别创建，还是其中有车辆对应已有任务？",
            )

        if len(frame_values) == 1 and self._has_plate_match_with_conflicting_vin(request, view, frame_values[0]):
            return self._decision(
                request,
                AttributionDecision.hold_for_confirmation,
                None,
                0.7,
                RiskLevel.medium,
                "当前 VIN（车架号）与已有任务不一致，但车牌命中已有任务；可能是录入错误、换车或材料冲突，不能自动覆盖，也不能直接新建。",
                detected_entities,
                used_relationships + ["contradicts"],
                conflicts,
                candidates,
                "ask_user_to_confirm_vehicle_identity",
                confirmation_question="这条信息中的 VIN（车架号）与已有车牌任务不一致，请确认是新车辆还是原任务材料有误？",
            )

        if len(frame_values) == 1 and self._is_new_vin_without_exact_match(request, frame_values[0]):
            return self._decision(
                request,
                AttributionDecision.create_new_task,
                None,
                0.84,
                RiskLevel.low,
                "当前消息出现新的 VIN（车架号），未命中任何已有任务的车辆唯一标识；按现实车险作业理解，这是新的车辆标的，应创建独立单任务，不能归到会话焦点旧任务。",
                detected_entities,
                used_relationships + ["starts"],
                [],
                candidates,
                "create_task_draft",
                next_stage="quote_preparing",
                next_action="create_task_draft",
                fact_write_allowed=True,
                new_task_hint={
                    "task_mode": "independent_single_task",
                    "vehicle": {k: sorted(v) for k, v in view.vehicles.items()},
                    "vin_policy": "新 VIN（车架号）代表新车辆标的；禁止覆盖旧任务 VIN。",
                    "conflict_policy": "若同时出现相同车牌但不同 VIN，需要确认；本次未命中旧任务 VIN。",
                },
            )

        if (not best_precheck or best_precheck.score < 70) and self._multiple_active_weak_context(
            request, view
        ):
            return self._decision(
                request,
                AttributionDecision.hold_for_confirmation,
                None,
                0.72,
                RiskLevel.medium,
                "当前存在多个活跃任务，而新消息缺少明确车辆标识，不能自动归属。",
                detected_entities,
                used_relationships,
                conflicts,
                candidates,
                "ask_user_to_clarify_vehicle",
                confirmation_question="请确认这条信息是用于哪台车或哪个报价任务？",
            )

        best = best_precheck
        if best and best.score >= 70:
            selected_conflicts = [conflict for conflict in conflicts if conflict.startswith(f"{best.task_id}:")]
            if selected_conflicts and self._task_is_high_risk(best, request):
                return self._decision(
                    request,
                    AttributionDecision.route_to_manual,
                    best.task_id,
                    min(best.score / 100, 0.96),
                    RiskLevel.high,
                    f"候选任务匹配但高风险阶段存在冲突：{'；'.join(selected_conflicts)}。",
                    detected_entities,
                    used_relationships + ["contradicts"],
                    selected_conflicts,
                    candidates,
                    "route_to_manual",
                    confirmation_question="当前材料与任务信息不一致，请人工核实后再继续。",
                )
            if selected_conflicts:
                return self._decision(
                    request,
                    AttributionDecision.hold_for_confirmation,
                    best.task_id,
                    min(best.score / 100, 0.9),
                    RiskLevel.medium,
                    f"候选任务匹配但存在信息冲突：{'；'.join(selected_conflicts)}。",
                    detected_entities,
                    used_relationships + ["contradicts"],
                    selected_conflicts,
                    candidates,
                    "ask_user_to_confirm_task",
                    confirmation_question=self._confirmation_question(best, view),
                )
            task_status = self._candidate_status(best.task_id, request)
            decision_type = (
                AttributionDecision.attach_to_active_task
                if task_status == "active"
                else AttributionDecision.wake_historical_task
            )
            next_stage, next_action = self._next_stage_and_action(view, best)
            return self._decision(
                request,
                decision_type,
                best.task_id,
                min(best.score / 100, 0.96),
                RiskLevel.low if best.score >= 85 else RiskLevel.medium,
                self._business_reason(view, best, []),
                detected_entities,
                used_relationships,
                [],
                candidates,
                next_action,
                next_stage=next_stage,
                fact_write_allowed=best.score >= 85,
            )

        if best and best.score >= 45 and not view.vehicles:
            return self._decision(
                request,
                AttributionDecision.hold_for_confirmation,
                best.task_id,
                min(best.score / 100, 0.8),
                RiskLevel.medium,
                "存在可能匹配的任务，但证据强度不足或存在阶段/实体不确定性。",
                detected_entities,
                used_relationships,
                conflicts,
                candidates,
                "ask_user_to_confirm_task",
                confirmation_question=self._confirmation_question(best, view),
            )

        if view.vehicles:
            return self._decision(
                request,
                AttributionDecision.create_new_task,
                None,
                0.82 if not view.weak_only else 0.62,
                RiskLevel.low if not view.weak_only else RiskLevel.medium,
                "识别到新的车辆标识，且未召回到可信匹配任务；根据 VIN（车架号）不可变规则，建议创建独立新任务。",
                detected_entities,
                used_relationships + ["starts"],
                [],
                candidates,
                "create_task_draft",
                next_stage="quote_preparing",
                next_action="create_task_draft",
                fact_write_allowed=True,
                new_task_hint={
                    "task_mode": "independent_single_task",
                    "vehicle": {k: sorted(v) for k, v in view.vehicles.items()},
                    "vin_policy": "VIN 不可覆盖旧任务；新 VIN 只能创建新任务或等待确认。",
                },
            )

        if (view.intents or view.quote_fields) and request.active_tasks:
            task = request.active_tasks[0]
            next_stage, next_action = self._next_stage_and_action(
                view,
                CandidateTask(
                    task_id=task.task_id,
                    score=55,
                    match_reason="recent_active_task",
                    stage=task.stage,
                    status=task.status,
                    last_active_at=task.last_active_at,
                    task=task,
                ),
            )
            return self._decision(
                request,
                AttributionDecision.attach_to_active_task,
                task.task_id,
                0.68,
                RiskLevel.medium,
                "消息包含业务意图但缺少强实体，按最近活跃任务给出中风险归属建议。",
                detected_entities,
                used_relationships,
                conflicts,
                candidates,
                next_action,
                next_stage=next_stage,
            )

        return self._decision(
            request,
            AttributionDecision.hold_for_confirmation,
            None,
            0.4,
            RiskLevel.medium,
            "缺少足够车辆、人或会话关系，无法确定归属。",
            detected_entities,
            used_relationships,
            conflicts,
            candidates,
            "ask_user_to_clarify_vehicle",
            confirmation_question="请补充车牌、车架号或说明要处理哪台车。",
        )

    def _try_model_reasoning(self, request: TaskAttributionRequest) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        status = self.runtime.status()
        if self.use_model is False:
            status.enabled = False
            status.using_real_model = False
            status.reason = "测试显式关闭真实模型，使用离线规则 fallback（兜底逻辑）。"
            return None, self._runtime_dict(status, fallback_used=True)
        if not status.enabled:
            return None, self._runtime_dict(status, fallback_used=True)
        if self.use_model is None and os.getenv("AGENTIC_LAB_DISABLE_MODEL", "").lower() in {"1", "true", "yes"}:
            status.enabled = False
            status.using_real_model = False
            status.reason = "环境变量 AGENTIC_LAB_DISABLE_MODEL 已关闭真实模型。"
            return None, self._runtime_dict(status, fallback_used=True)
        call = self.runtime.structured_call_sync(
            system_prompt=TASK_ATTRIBUTION_AGENT_INSTRUCTIONS,
            user_payload={
                "agent": "task_attribution",
                "event": request.event.model_dump(mode="json"),
                "new_evidence": [item.model_dump(mode="json") for item in request.new_evidence],
                "active_tasks": [item.model_dump(mode="json") for item in request.active_tasks],
                "candidate_tasks": [item.model_dump(mode="json") for item in request.candidate_tasks],
                "conversation_memory": request.conversation_memory.model_dump(mode="json"),
                "policy_rules": [
                    "VIN 不可更新或覆盖。",
                    "多 VIN 新报价拆分为多个独立单任务。",
                    "省略式修改需要结合会话焦点任务或最近批量任务集合。",
                ],
                "expected_json_keys": ["scenario_intent", "scope", "decision_hint", "business_reason"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scenario_intent": {"type": "string"},
                    "scope": {"type": "string"},
                    "decision_hint": {"type": "string"},
                    "business_reason": {"type": "string"},
                },
            },
        )
        return call.content if call.ok else None, self._runtime_dict(call.status, fallback_used=not call.ok, error=call.error)

    def _runtime_dict(self, status: Any, fallback_used: bool, error: str | None = None) -> dict[str, Any]:
        payload = status.__dict__.copy()
        payload["fallback_used"] = fallback_used
        payload["fallback_policy"] = (
            "仅用于离线单元测试和结构验证；真实能力验证需要配置模型密钥。"
            if fallback_used
            else "真实大模型已参与任务归属推理。"
        )
        if error:
            payload["error"] = error
        return payload

    def _view(self, evidence: list[EvidenceItem]) -> EvidenceView:
        vehicles: dict[str, set[str]] = {}
        parties: dict[str, set[str]] = {}
        intents: set[str] = set()
        quote_fields: set[str] = set()
        weak_only = bool(evidence)
        for item in evidence:
            weak_only = weak_only and item.evidence_strength.value in {"weak", "reference"}
            if item.entity_type == "vehicle":
                vehicles.setdefault(item.field_name, set()).add(item.normalized_value)
            elif item.entity_type == "party":
                parties.setdefault(item.field_name, set()).add(item.normalized_value)
            elif item.entity_type == "intent":
                intents.add(item.normalized_value)
            elif item.entity_type in {"insurance_plan", "non_auto_product"}:
                quote_fields.add(item.field_name)
        return EvidenceView(
            vehicles=vehicles,
            parties=parties,
            intents=intents,
            quote_fields=quote_fields,
            evidence_ids=[item.evidence_id for item in evidence],
            weak_only=weak_only,
        )

    def _score_candidates(
        self, request: TaskAttributionRequest, view: EvidenceView
    ) -> list[TargetTaskCandidate]:
        candidate_records = self._candidate_records(request)
        scored: list[TargetTaskCandidate] = []
        for task in candidate_records:
            score = 0.0
            reasons: list[str] = []
            vehicle_score, vehicle_reasons = self._vehicle_score(task, view)
            party_score, party_reasons = self._party_score(task, view)
            score += vehicle_score + party_score
            reasons.extend(vehicle_reasons + party_reasons)
            if task.conversation_id == request.event.conversation_id and task.status.value == "active":
                score += 15
                reasons.append("同会话活跃任务")
            if task.task_id == request.conversation_memory.focused_task_id:
                score += 45
                reasons.append("会话焦点任务")
            if task.task_id in request.conversation_memory.recent_task_set_ids:
                score += 25
                reasons.append("最近批量任务集合")
            if view.intents and self._stage_matches_intent(task, view):
                score += 55
                reasons.append("业务意图与任务阶段匹配")
            if view.quote_fields and task.stage.value in {"quote_confirming", "quote_preparing", "quote_modifying"}:
                score += 55
                reasons.append("报价方案修改与报价阶段匹配")
            if task.status.value in {"completed", "suspended"}:
                score -= 15
                reasons.append("历史/已完成任务降权")
            if score > 0:
                scored.append(
                    TargetTaskCandidate(
                        task_id=task.task_id,
                        score=max(0, min(100, score)),
                        reason="、".join(reasons),
                    )
                )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:5]

    def _candidate_records(self, request: TaskAttributionRequest) -> list[TaskRecord]:
        records: dict[str, TaskRecord] = {task.task_id: task for task in request.active_tasks}
        for candidate in request.candidate_tasks:
            if candidate.task:
                records[candidate.task.task_id] = candidate.task
        return list(records.values())

    def _frame_values_in_order(self, evidence: list[EvidenceItem], view: EvidenceView) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in evidence:
            if item.field_name != "frameNo" or item.normalized_value in seen:
                continue
            seen.add(item.normalized_value)
            values.append(item.normalized_value)
        if values:
            return values
        return sorted(view.vehicles.get("frameNo", set()))

    def _vehicle_score(self, task: TaskRecord, view: EvidenceView) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        facts = task.vehicle_facts
        for field, values in view.vehicles.items():
            fact_value = facts.get(field) or facts.get(self._field_alias(field))
            if not fact_value:
                continue
            if fact_value in values:
                if field == "frameNo":
                    score += 65
                    reasons.append("车架号一致")
                elif field == "vehicleLicenseNo":
                    score += 45
                    reasons.append("车牌一致")
                elif field == "engineNo":
                    score += 20
                    reasons.append("发动机号一致")
                else:
                    score += 10
                    reasons.append(f"{field}一致")
        return score, reasons

    def _party_score(self, task: TaskRecord, view: EvidenceView) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        for party in task.party_facts:
            if party.id_no and party.id_no in view.parties.get("ownerIdNo", set()):
                score += 45
                reasons.append("证件号一致")
            if party.name and party.name in view.parties.get("ownerName", set()):
                score += 12
                reasons.append("姓名一致")
        return score, reasons

    def _field_alias(self, field: str) -> str:
        aliases = {
            "vehicleLicenceCode": "vehicleLicenseNo",
            "licenseNoFromText": "vehicleLicenseNo",
            "frameNoFromText": "frameNo",
        }
        return aliases.get(field, field)

    def _find_conflicts(self, active_tasks: list[TaskRecord], view: EvidenceView) -> list[str]:
        conflicts: list[str] = []
        for task in active_tasks:
            for field, values in view.vehicles.items():
                fact_value = task.vehicle_facts.get(field) or task.vehicle_facts.get(self._field_alias(field))
                if fact_value and values and fact_value not in values:
                    conflicts.append(f"{task.task_id}:{field}冲突")
            for party in task.party_facts:
                owner_names = view.parties.get("ownerName", set())
                owner_ids = view.parties.get("ownerIdNo", set())
                if party.name and owner_names and party.name not in owner_names:
                    conflicts.append(f"{task.task_id}:车主姓名冲突")
                if party.id_no and owner_ids and party.id_no not in owner_ids:
                    conflicts.append(f"{task.task_id}:车主证件号冲突")
        return conflicts

    def _task_is_high_risk(
        self, candidate: TargetTaskCandidate, request: TaskAttributionRequest
    ) -> bool:
        for task in request.active_tasks:
            if task.task_id == candidate.task_id:
                return task.stage.value in STAGE_HIGH_RISK
        for item in request.candidate_tasks:
            if item.task_id == candidate.task_id:
                stage = item.stage.value if hasattr(item.stage, "value") else str(item.stage)
                return stage in STAGE_HIGH_RISK
        return False

    def _candidate_status(self, task_id: str, request: TaskAttributionRequest) -> str:
        for task in request.active_tasks:
            if task.task_id == task_id:
                return task.status.value
        for item in request.candidate_tasks:
            if item.task_id == task_id:
                return item.status.value if hasattr(item.status, "value") else str(item.status)
        return "active"

    def _multiple_active_weak_context(
        self, request: TaskAttributionRequest, view: EvidenceView
    ) -> bool:
        active_count = sum(1 for task in request.active_tasks if task.status.value == "active")
        has_vehicle = bool(view.vehicles)
        if request.conversation_memory.focused_task_id or request.conversation_memory.quoted_task_id:
            return False
        return active_count > 1 and (not has_vehicle or view.weak_only)

    def _is_unresolved_attachment_material(self, request: TaskAttributionRequest, view: EvidenceView) -> bool:
        if not request.event.attachments:
            return False
        return not (view.vehicles or view.parties or view.quote_fields or view.intents)

    def _is_new_vin_without_exact_match(self, request: TaskAttributionRequest, frame_no: str) -> bool:
        for task in self._candidate_records(request):
            if task.vehicle_facts.get("frameNo") == frame_no or task.vehicle_facts.get("frameNoFromText") == frame_no:
                return False
        return True

    def _has_plate_match_with_conflicting_vin(
        self, request: TaskAttributionRequest, view: EvidenceView, frame_no: str
    ) -> bool:
        plates = view.vehicles.get("vehicleLicenseNo", set())
        if not plates:
            return False
        for task in self._candidate_records(request):
            task_plate = task.vehicle_facts.get("vehicleLicenseNo") or task.vehicle_facts.get("vehicleLicenceCode")
            task_frame = task.vehicle_facts.get("frameNo") or task.vehicle_facts.get("frameNoFromText")
            if task_plate in plates and task_frame and task_frame != frame_no:
                return True
        return False

    def _valid_recent_task_set_ids(self, request: TaskAttributionRequest) -> list[str]:
        existing_ids = {task.task_id for task in self._candidate_records(request)}
        return [task_id for task_id in request.conversation_memory.recent_task_set_ids if task_id in existing_ids]

    def _is_recent_task_set_update(self, request: TaskAttributionRequest, view: EvidenceView) -> bool:
        if not request.conversation_memory.recent_task_set_ids:
            return False
        if view.vehicles:
            return False
        if not (view.quote_fields or "modify_quote" in view.intents):
            return False
        text = request.event.content_text
        has_batch_language = any(token in text for token in ["都", "这几台", "这些车", "全部", "都按", "一样"])
        return has_batch_language and bool(self._valid_recent_task_set_ids(request))

    def _focused_task(self, request: TaskAttributionRequest) -> TaskRecord | None:
        preferred_ids = [
            request.conversation_memory.quoted_task_id,
            request.conversation_memory.focused_task_id,
        ]
        records = {task.task_id: task for task in self._candidate_records(request)}
        for task_id in preferred_ids:
            if task_id and task_id in records:
                return records[task_id]
        return None

    def _can_attach_to_focus(self, view: EvidenceView) -> bool:
        if view.vehicles:
            return False
        return bool(view.quote_fields or view.intents or view.parties)

    def _should_ignore(self, view: EvidenceView, request: TaskAttributionRequest) -> bool:
        if request.event.attachments:
            return False
        if view.vehicles or view.parties or view.quote_fields or view.intents:
            return False
        text = request.event.content_text.strip()
        return not text or text in {"嗯", "好的", "收到", "谢谢"}

    def _stage_matches_intent(self, task: TaskRecord, view: EvidenceView) -> bool:
        stage = task.stage.value
        if "modify_quote" in view.intents and stage in {"quote_confirming", "quote_preparing", "quote_modifying"}:
            return True
        if "confirm_quote" in view.intents and stage in {"quote_confirming", "quote_preparing"}:
            return True
        if "underwriting_request" in view.intents and stage in {"insuring", "underwriting", "quote_confirming"}:
            return True
        if "payment_code_request" in view.intents and stage in {"payment", "insuring"}:
            return True
        if "payment_completed" in view.intents and stage in {"payment", "insuring"}:
            return True
        if "send_policy" in view.intents and stage in {"completed", "payment"}:
            return True
        if "policy_delivered" in view.intents and stage in {"payment", "policy_delivering", "completed"}:
            return True
        if "invoice_request" in view.intents and stage in {"completed", "payment", "policy_delivering", "invoicing"}:
            return True
        if "invoice_status" in view.intents and stage in {"completed", "policy_delivering", "invoicing"}:
            return True
        if "quote_result" in view.intents and stage in {"quote_preparing", "quote_modifying", "quote_confirming"}:
            return True
        if "quote_status" in view.intents and stage in {"quote_preparing", "quote_modifying"}:
            return True
        if "insurance_status" in view.intents and stage in {"quote_confirming", "insuring", "underwriting"}:
            return True
        if "progress_query" in view.intents and task.status.value == "active":
            return True
        if "abandon_task" in view.intents and task.status.value == "active":
            return True
        return False

    def _detected_entities(self, view: EvidenceView) -> list[str]:
        entities: list[str] = []
        if view.vehicles:
            entities.append("vehicle")
        if view.parties:
            entities.append("party")
        if view.quote_fields:
            entities.append("quote_plan")
        if view.intents:
            entities.append("intent")
        return entities

    def _relationships(self, view: EvidenceView, request: TaskAttributionRequest) -> list[str]:
        relationships: list[str] = []
        if view.vehicles or view.parties:
            relationships.append("identifies")
        if view.quote_fields or "modify_quote" in view.intents:
            relationships.append("modifies")
        if "confirm_quote" in view.intents:
            relationships.append("confirms")
        if request.event.quoted_context:
            relationships.append("responds_to")
        if request.conversation_memory.focused_task_id:
            relationships.append("wakes")
        if view.vehicles and not request.active_tasks:
            relationships.append("starts")
        if "abandon_task" in view.intents:
            relationships.append("abandons")
        return relationships

    def _next_stage_and_action(
        self, view: EvidenceView, candidate: TargetTaskCandidate | CandidateTask
    ) -> tuple[str | None, str | None]:
        if "modify_quote" in view.intents or view.quote_fields:
            return "quote_modifying", "route_to_quote_precheck"
        if "confirm_quote" in view.intents:
            return "insuring", "route_to_insurance_precheck"
        if "underwriting_request" in view.intents:
            return "underwriting", "route_to_underwriting_check"
        if "payment_code_request" in view.intents:
            return "payment", "route_to_payment_status_check"
        if "payment_completed" in view.intents:
            return "policy_delivering", "route_to_policy_delivery"
        if "send_policy" in view.intents:
            return "policy_delivering", "route_to_policy_delivery"
        if "policy_delivered" in view.intents:
            return "completed", "mark_policy_delivered"
        if "invoice_request" in view.intents:
            return "invoicing", "route_to_invoice"
        if "invoice_status" in view.intents:
            return "invoicing", "track_invoice_status"
        if "quote_result" in view.intents:
            return "quote_confirming", "wait_for_quote_confirmation"
        if "quote_status" in view.intents:
            return "quote_preparing", "track_quote_status"
        if "insurance_status" in view.intents:
            return "insuring", "track_insurance_status"
        if "abandon_task" in view.intents:
            return "suspended", "ask_abandon_confirmation"
        return None, "attach_evidence_to_task"

    def _business_reason(
        self, view: EvidenceView, candidate: TargetTaskCandidate, conflicts: list[str]
    ) -> str:
        if conflicts:
            return f"候选任务匹配但存在冲突：{'；'.join(conflicts)}。"
        if view.quote_fields or "modify_quote" in view.intents:
            return f"用户正在修改报价方案，{candidate.reason}。"
        return f"新证据与候选任务匹配：{candidate.reason}。"

    def _confirmation_question(self, candidate: TargetTaskCandidate, view: EvidenceView) -> str:
        if view.parties and not view.vehicles:
            return f"这份人员材料是用于任务 {candidate.task_id} 对应的车辆吗？"
        return f"请确认这条信息是否归属任务 {candidate.task_id}？"

    def _decision(
        self,
        request: TaskAttributionRequest,
        decision: AttributionDecision,
        selected_task_id: str | None,
        confidence: float,
        risk_level: RiskLevel,
        reason: str,
        detected_entities: list[str],
        used_relationships: list[str],
        conflicts: list[str],
        candidates: list[TargetTaskCandidate],
        fallback_action: str,
        confirmation_question: str | None = None,
        next_stage: str | None = None,
        next_action: str | None = None,
        fact_write_allowed: bool = False,
        new_task_hint: dict | None = None,
        selected_task_ids: list[str] | None = None,
    ) -> TaskAttributionDecision:
        return TaskAttributionDecision(
            request_id=request.request_id,
            model_runtime=self._last_model_runtime,
            decision=decision,
            selected_task_id=selected_task_id,
            selected_task_ids=selected_task_ids or ([selected_task_id] if selected_task_id else []),
            new_task_hint=new_task_hint,
            confidence=confidence,
            risk_level=risk_level,
            required_confirmation=decision == AttributionDecision.hold_for_confirmation
            or bool(confirmation_question),
            business_reason=reason,
            entity_graph_summary=EntityGraphSummary(
                detected_entities=detected_entities,
                used_relationships=sorted(set(used_relationships)),
                conflicts=conflicts,
            ),
            target_task_candidates=candidates,
            confirmation_question=confirmation_question,
            state_update_suggestion=StateUpdateSuggestion(
                next_stage=next_stage,
                next_action=next_action,
                fact_write_allowed=fact_write_allowed,
            ),
            fallback_action=fallback_action,
            used_evidence_ids=request.new_evidence and [e.evidence_id for e in request.new_evidence] or [],
            decision_log_summary=reason,
        )
