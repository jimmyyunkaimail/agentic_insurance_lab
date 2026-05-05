from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class MessageType(str, Enum):
    text = "text"
    voice_transcript = "voice_transcript"
    image = "image"
    file = "file"
    quoted_message = "quoted_message"
    mixed = "mixed"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ValidationStatus(str, Enum):
    valid = "valid"
    invalid = "invalid"
    uncertain = "uncertain"
    not_checked = "not_checked"


class EvidenceStrength(str, Enum):
    strong = "strong"
    medium = "medium"
    weak = "weak"
    reference = "reference"


class AttachmentRef(BaseModel):
    attachment_id: str = Field(default_factory=lambda: new_id("att"))
    file_type: Literal["image", "pdf", "excel", "word", "other"] = "other"
    file_ref: str
    page_hint: int | None = None


class QuotedContext(BaseModel):
    quoted_message_id: str | None = None
    quoted_text: str | None = None
    quoted_type: Literal["bot_message", "user_message", "unknown"] = "unknown"


class MessageEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    channel: str = "wechat_work"
    sender_id: str = "partner_demo"
    conversation_id: str = "conv_demo"
    message_type: MessageType = MessageType.text
    content_text: str = ""
    attachments: list[AttachmentRef] = Field(default_factory=list)
    quoted_context: QuotedContext | None = None
    conversation_excerpt: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=now_iso)


class MaterialUnderstandingRequest(BaseModel):
    event_id: str | None = None
    channel: str = "wechat_work"
    sender_id: str = "partner_demo"
    conversation_id: str = "conv_demo"
    message_type: MessageType = MessageType.text
    content_text: str = ""
    attachments: list[AttachmentRef] = Field(default_factory=list)
    quoted_context: QuotedContext | None = None
    conversation_excerpt: list[str] = Field(default_factory=list)

    def to_event(self) -> MessageEvent:
        return MessageEvent(
            event_id=self.event_id or new_id("evt"),
            channel=self.channel,
            sender_id=self.sender_id,
            conversation_id=self.conversation_id,
            message_type=self.message_type,
            content_text=self.content_text,
            attachments=self.attachments,
            quoted_context=self.quoted_context,
            conversation_excerpt=self.conversation_excerpt,
        )


class SourceRef(BaseModel):
    event_id: str
    attachment_id: str | None = None
    page_index: int | None = None
    region_hint: str | None = None


class DocumentQuality(BaseModel):
    clarity: Literal["clear", "medium", "blurred", "blocked", "incomplete"] = "clear"
    orientation: Literal["normal", "rotated", "unknown"] = "normal"
    page_completeness: Literal["complete", "partial", "multi_page_missing"] = "complete"
    quality_score: float = Field(default=0.9, ge=0, le=1)


class DocumentResult(BaseModel):
    document_id: str = Field(default_factory=lambda: new_id("doc"))
    document_type: str = "textual_business_material"
    document_name: str = "文本业务材料"
    document_category: Literal[
        "standard_certificate", "non_standard", "textual", "unknown"
    ] = "textual"
    standard_level: Literal["standard", "non_standard", "textual", "unknown"] = "textual"
    quality: DocumentQuality = Field(default_factory=DocumentQuality)
    usable_for: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)
    source_refs: list[SourceRef] = Field(default_factory=list)


class FieldValidation(BaseModel):
    status: ValidationStatus = ValidationStatus.not_checked
    rules: list[str] = Field(default_factory=list)
    error_message: str | None = None


class EvidenceItem(BaseModel):
    evidence_id: str = Field(default_factory=lambda: new_id("evd"))
    entity_type: Literal[
        "vehicle",
        "party",
        "role",
        "insurance_plan",
        "non_auto_product",
        "credential",
        "intent",
    ]
    role: Literal["owner", "applicant", "insured", "contact", "unknown"] = "unknown"
    field_name: str
    field_label: str
    raw_value: str
    normalized_value: str
    value_type: Literal["string", "date", "amount", "enum", "boolean", "list", "object"] = (
        "string"
    )
    source_type: Literal[
        "ocr", "text", "voice_transcript", "document_layout", "quoted_context"
    ] = "text"
    source_document_id: str | None = None
    source_event_id: str | None = None
    confidence: float = Field(default=0.8, ge=0, le=1)
    evidence_strength: EvidenceStrength = EvidenceStrength.medium
    validation: FieldValidation = Field(default_factory=FieldValidation)


class RiskFlag(BaseModel):
    risk_type: Literal[
        "low_quality",
        "missing_core_field",
        "ambiguous_role",
        "format_invalid",
        "unknown_document",
        "possible_conflict",
        "low_confidence",
    ]
    risk_level: RiskLevel = RiskLevel.low
    description: str


class ConflictCandidate(BaseModel):
    field_name: str
    values: list[str]
    conflict_type: Literal["intra_document", "cross_input_hint"]
    resolution_hint: Literal["need_confirmation", "manual_review", "ignore"] = (
        "need_confirmation"
    )


class MaterialActionHint(BaseModel):
    next_route: Literal["task_attribution", "manual_review", "ignore"] = "task_attribution"
    suggested_follow_up: str | None = None
    reason: str


class QuotedLink(BaseModel):
    quoted_message_id: str | None = None
    quoted_type: Literal["bot_message", "user_message", "unknown"] = "unknown"
    quoted_text_preview: str | None = None
    relation_hint: str = "display_link_only"


class VehicleMention(BaseModel):
    mention_id: str = Field(default_factory=lambda: new_id("veh"))
    frame_no: str | None = None
    license_no: str | None = None
    engine_no: str | None = None
    source_text: str | None = None
    confidence: float = Field(default=0.8, ge=0, le=1)


class QuotePlanMention(BaseModel):
    field_name: str
    field_label: str
    raw_value: str
    normalized_value: str
    operation: Literal["set", "remove", "confirm", "query"] = "set"
    scope_hint: Literal["single_task", "task_set", "unknown"] = "unknown"
    confidence: float = Field(default=0.8, ge=0, le=1)


class PartyMention(BaseModel):
    role: Literal["owner", "applicant", "insured", "contact", "unknown"] = "unknown"
    field_name: str
    field_label: str
    raw_value: str
    normalized_value: str
    confidence: float = Field(default=0.8, ge=0, le=1)


class MaterialUnderstandingResult(BaseModel):
    event_id: str
    agent_name: str = "material_understanding_agent"
    model_runtime: dict[str, Any] = Field(default_factory=dict)
    input_summary: dict[str, Any]
    current_intent: str = "unknown"
    speech_act: str = "unknown"
    vehicle_mentions: list[VehicleMention] = Field(default_factory=list)
    quote_plan_mentions: list[QuotePlanMention] = Field(default_factory=list)
    party_mentions: list[PartyMention] = Field(default_factory=list)
    quoted_link: QuotedLink | None = None
    documents: list[DocumentResult] = Field(default_factory=list)
    evidence_list: list[EvidenceItem] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    conflict_candidates: list[ConflictCandidate] = Field(default_factory=list)
    material_action_hint: MaterialActionHint


class TaskStage(str, Enum):
    collecting_materials = "collecting_materials"
    quote_preparing = "quote_preparing"
    quote_confirming = "quote_confirming"
    quote_modifying = "quote_modifying"
    insuring = "insuring"
    underwriting = "underwriting"
    payment = "payment"
    policy_delivering = "policy_delivering"
    invoicing = "invoicing"
    completed = "completed"
    suspended = "suspended"
    manual = "manual"


class TaskStatus(str, Enum):
    active = "active"
    completed = "completed"
    suspended = "suspended"
    manual = "manual"


class PartyFact(BaseModel):
    role: Literal["owner", "applicant", "insured", "contact", "unknown"] = "owner"
    name: str | None = None
    id_no: str | None = None
    mobile: str | None = None


class TaskRecord(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    conversation_id: str = "conv_demo"
    stage: TaskStage = TaskStage.collecting_materials
    status: TaskStatus = TaskStatus.active
    vehicle_facts: dict[str, str] = Field(default_factory=dict)
    party_facts: list[PartyFact] = Field(default_factory=list)
    quote_plan_summary: str | None = None
    last_bot_action: str | None = None
    pending_question: str | None = None
    last_active_at: str = Field(default_factory=now_iso)


class ConversationMemory(BaseModel):
    recent_messages_summary: list[str] = Field(default_factory=list)
    last_user_intent: str | None = None
    last_bot_question: str | None = None
    recent_referenced_events: list[str] = Field(default_factory=list)
    focused_task_id: str | None = None
    recent_task_set_ids: list[str] = Field(default_factory=list)
    recent_task_set_reason: str | None = None
    last_quote_plan_context: dict[str, Any] = Field(default_factory=dict)
    quoted_task_id: str | None = None


class CandidateTask(BaseModel):
    task_id: str
    score: float = 0
    match_reason: str
    stage: TaskStage | str
    status: TaskStatus | str
    last_active_at: str | None = None
    task: TaskRecord | None = None


class PolicySnapshot(BaseModel):
    high_risk_stages: list[str] = Field(default_factory=lambda: ["insuring", "payment"])
    active_task_window_hours: int = 24
    manual_task_wakeup_days: int = 7
    party_name_wakeup_hours: int = 24


class TaskAttributionRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: new_id("attr_req"))
    event: MessageEvent
    new_evidence: list[EvidenceItem] = Field(default_factory=list)
    active_tasks: list[TaskRecord] = Field(default_factory=list)
    candidate_tasks: list[CandidateTask] = Field(default_factory=list)
    conversation_memory: ConversationMemory = Field(default_factory=ConversationMemory)
    policy_snapshot: PolicySnapshot = Field(default_factory=PolicySnapshot)


class AttributionDecision(str, Enum):
    attach_to_active_task = "attach_to_active_task"
    attach_to_historical_task = "attach_to_historical_task"
    create_new_task = "create_new_task"
    create_multiple_tasks = "create_multiple_tasks"
    attach_to_task_set = "attach_to_task_set"
    wake_historical_task = "wake_historical_task"
    hold_for_confirmation = "hold_for_confirmation"
    route_to_manual = "route_to_manual"
    ignore_or_chat = "ignore_or_chat"


class EntityGraphSummary(BaseModel):
    detected_entities: list[str] = Field(default_factory=list)
    used_relationships: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class TargetTaskCandidate(BaseModel):
    task_id: str
    score: float
    reason: str


class StateUpdateSuggestion(BaseModel):
    next_stage: str | None = None
    next_action: str | None = None
    fact_write_allowed: bool = False


class TaskAttributionDecision(BaseModel):
    request_id: str
    agent_name: str = "task_attribution_agent"
    model_runtime: dict[str, Any] = Field(default_factory=dict)
    decision: AttributionDecision
    selected_task_id: str | None = None
    selected_task_ids: list[str] = Field(default_factory=list)
    new_task_hint: dict[str, Any] | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    risk_level: RiskLevel
    required_confirmation: bool = False
    business_reason: str
    entity_graph_summary: EntityGraphSummary = Field(default_factory=EntityGraphSummary)
    target_task_candidates: list[TargetTaskCandidate] = Field(default_factory=list)
    confirmation_question: str | None = None
    state_update_suggestion: StateUpdateSuggestion = Field(default_factory=StateUpdateSuggestion)
    fallback_action: str | None = None
    used_evidence_ids: list[str] = Field(default_factory=list)
    decision_log_summary: str


class PolicyResult(BaseModel):
    result: Literal["approved", "require_confirmation", "route_to_manual", "rejected"]
    reason: str


class DecisionLog(BaseModel):
    decision_id: str = Field(default_factory=lambda: new_id("dec"))
    event_id: str
    agent_name: str
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    policy_result: PolicyResult | None = None
    created_at: str = Field(default_factory=now_iso)
