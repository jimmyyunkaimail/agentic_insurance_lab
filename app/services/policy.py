from __future__ import annotations

from app.schemas import AttributionDecision, PolicyResult, RiskLevel, TaskAttributionDecision


class PolicyEngine:
    """Local guardrail layer for attribution decisions."""

    def check_attribution(self, decision: TaskAttributionDecision) -> PolicyResult:
        if decision.decision == AttributionDecision.route_to_manual:
            return PolicyResult(result="route_to_manual", reason=decision.business_reason)
        if decision.decision == AttributionDecision.hold_for_confirmation:
            return PolicyResult(result="require_confirmation", reason=decision.business_reason)
        if decision.risk_level == RiskLevel.high:
            return PolicyResult(
                result="route_to_manual",
                reason="高风险归属必须人工处理，不能自动写入任务事实。",
            )
        if decision.required_confirmation:
            return PolicyResult(result="require_confirmation", reason="归属决策要求用户确认。")
        if decision.selected_task_id is None and decision.decision not in {
            AttributionDecision.create_new_task,
            AttributionDecision.create_multiple_tasks,
            AttributionDecision.attach_to_task_set,
            AttributionDecision.ignore_or_chat,
        }:
            return PolicyResult(result="rejected", reason="模型未给出有效目标任务。")
        if decision.decision == AttributionDecision.attach_to_task_set and not decision.selected_task_ids:
            return PolicyResult(result="rejected", reason="批量归属缺少目标任务集合。")
        if decision.entity_graph_summary.conflicts and decision.state_update_suggestion.fact_write_allowed:
            return PolicyResult(result="require_confirmation", reason="存在实体冲突，禁止自动写事实。")
        return PolicyResult(result="approved", reason="归属决策通过本地策略护栏。")
