from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

from app.agents.openai_runtime import MATERIAL_AGENT_INSTRUCTIONS, ModelRuntime
from app.knowledge import DOCUMENT_FIELD_MATRIX, DOCUMENT_TYPES
from app.schemas import (
    AttachmentRef,
    DocumentQuality,
    DocumentResult,
    EvidenceItem,
    EvidenceStrength,
    FieldValidation,
    MaterialActionHint,
    MaterialUnderstandingRequest,
    MaterialUnderstandingResult,
    MessageEvent,
    PartyMention,
    QuotePlanMention,
    QuotedLink,
    MessageType,
    RiskFlag,
    RiskLevel,
    SourceRef,
    ValidationStatus,
    VehicleMention,
    new_id,
)
from app.services.attachment_storage import AttachmentStorage
from app.services.tracing import TraceRecorder

VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b1[3-9]\d{9}\b")
ID_NO_RE = re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
PLATE_RE = re.compile(r"[\u4e00-\u9fa5][A-Z][\s-]?[A-Z0-9]{5,6}", re.IGNORECASE)
ENGINE_NO_RE = re.compile(r"\b[A-Z0-9]{6,12}\b", re.IGNORECASE)
DATE_RE = re.compile(r"(?P<year>20\d{2}|19\d{2})[-年./](?P<month>\d{1,2})[-月./](?P<day>\d{1,2})")
QUOTE_FLOW_NO_RE = re.compile(r"\b\d{6,8}/\d{7,10}(?:-\d+)?\b")
PROCESS_NO_RE = re.compile(r"\bQ\d{4,}\b", re.IGNORECASE)
COMMERCIAL_PREMIUM_RE = re.compile(r"商业险(?:保费)?\s*(\d+(?:\.\d+)?)")
COMPULSORY_PREMIUM_RE = re.compile(r"交强险(?:保费)?\s*(\d+(?:\.\d+)?)")

CANONICAL_INTENTS = {
    "modify_quote",
    "batch_modify_quote",
    "confirm_quote",
    "underwriting_request",
    "payment_code_request",
    "payment_completed",
    "progress_query",
    "send_policy",
    "policy_delivered",
    "invoice_request",
    "invoice_status",
    "quote_result",
    "quote_status",
    "insurance_status",
    "transfer_manual",
    "abandon_task",
    "new_batch_quote",
    "new_or_existing_quote",
    "supplement_party_info",
    "supplement_attachment_material",
    "business_status_or_reference",
    "chat",
    "unknown",
}


def normalize_vin(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def normalize_plate(value: str) -> str:
    return value.replace("-", "").replace(" ", "").upper()


def normalize_date(value: str) -> str:
    match = DATE_RE.search(value)
    if not match:
        return value
    return f"{match.group('year')}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"


def validate_vin(value: str) -> FieldValidation:
    normalized = normalize_vin(value)
    if len(normalized) == 17 and not re.search(r"[IOQ]", normalized):
        return FieldValidation(status=ValidationStatus.valid, rules=["vin_length_17", "vin_no_ioq"])
    return FieldValidation(
        status=ValidationStatus.invalid,
        rules=["vin_length_17", "vin_no_ioq"],
        error_message="车架号需为17位，且不能包含 I/O/Q。",
    )


def validate_phone(value: str) -> FieldValidation:
    if PHONE_RE.fullmatch(value):
        return FieldValidation(status=ValidationStatus.valid, rules=["mobile_11_digits"])
    return FieldValidation(
        status=ValidationStatus.invalid,
        rules=["mobile_11_digits"],
        error_message="手机号需为11位大陆手机号格式。",
    )


def validate_plate(value: str) -> FieldValidation:
    normalized = normalize_plate(value)
    if 7 <= len(normalized) <= 8 and re.match(r"[\u4e00-\u9fa5][A-Z]", normalized):
        return FieldValidation(status=ValidationStatus.valid, rules=["license_plate_shape"])
    return FieldValidation(
        status=ValidationStatus.uncertain,
        rules=["license_plate_shape"],
        error_message="车牌格式不完整或存在遮挡。",
    )


class MaterialUnderstandingAgent:
    """Heuristic local implementation with an LLM-ready boundary.

    The implementation is deterministic so that the validation framework can run
    without API keys. The same request/response schema can be sent to an OpenAI
    Agent later for higher quality multimodal extraction.
    """

    agent_name = "material_understanding_agent"

    def __init__(
        self,
        use_model: bool | None = None,
        runtime: ModelRuntime | None = None,
        attachment_storage: AttachmentStorage | None = None,
    ):
        self.use_model = use_model
        self.runtime = runtime or ModelRuntime()
        self.attachment_storage = attachment_storage or AttachmentStorage()

    def run(self, request: MaterialUnderstandingRequest | MessageEvent) -> MaterialUnderstandingResult:
        event = request if isinstance(request, MessageEvent) else request.to_event()
        trace = TraceRecorder("material_understanding")
        trace.add(
            phase="input",
            step_name="接收消息",
            action="标准化验证台输入为 MessageEvent（消息事件）",
            input_snapshot={
                "event_id": event.event_id,
                "conversation_id": event.conversation_id,
                "message_type": event.message_type.value,
                "content_text": event.content_text,
                "attachments": [item.model_dump(mode="json") for item in event.attachments],
                "quoted_context": event.quoted_context.model_dump(mode="json")
                if event.quoted_context
                else None,
            },
            output_snapshot={
                "has_text": bool(event.content_text.strip()),
                "attachment_count": len(event.attachments),
                "quoted_message_present": event.quoted_context is not None,
            },
            decision_basis=["引用消息只用于关联展示，不作为当前槽位来源。"],
        )
        model_payload, model_runtime = self._try_model_understanding(event)
        trace.add(
            phase="model",
            step_name="模型理解尝试",
            action="调用真实大模型或记录 fallback（兜底逻辑）原因",
            input_snapshot={
                "model_enabled": model_runtime.get("enabled"),
                "provider": model_runtime.get("provider"),
                "model": model_runtime.get("model"),
                "endpoint": model_runtime.get("endpoint"),
                "input_modalities": model_runtime.get("input_modalities", []),
            },
            output_snapshot={
                "fallback_used": model_runtime.get("fallback_used"),
                "fallback_policy": model_runtime.get("fallback_policy"),
                "reason": model_runtime.get("reason"),
                "error": model_runtime.get("error"),
                "model_payload": model_payload,
            },
            decision_basis=[
                "真实模型不可用或调用失败时，不伪装成功。",
                "模型输出只作为材料理解候选，仍需本地规则合并与校验。",
            ],
        )
        documents = self._merge_model_documents(event, model_payload)
        textual_document = self._textual_document(event) if self._has_text_material(event) else None
        if textual_document:
            documents.append(textual_document)
        trace.add(
            phase="document",
            step_name="材料分类与合并",
            action="合并模型识别材料、附件文件名规则分类和文本材料",
            output_snapshot={
                "document_count": len(documents),
                "documents": [item.model_dump(mode="json") for item in documents],
                "textual_document_added": textual_document is not None,
            },
            decision_basis=[
                "模型材料识别优先覆盖同一附件，本地附件分类补齐未覆盖附件。",
                "有正文文本时补充 textual_business_material（文本业务材料）。",
            ],
        )

        raw_evidence = [
            *self._model_evidence(event, documents, model_payload),
            *self._extract_text_evidence(event, documents),
        ]
        evidence = self._dedupe_evidence(raw_evidence)
        trace.add(
            phase="evidence",
            step_name="证据抽取与去重",
            action="从模型结构化输出和当前消息文本抽取 Evidence（证据）",
            output_snapshot={
                "raw_evidence_count": len(raw_evidence),
                "raw_evidence_fields": [
                    {
                        "entity_type": item.entity_type,
                        "field_name": item.field_name,
                        "field_label": item.field_label,
                        "normalized_value": item.normalized_value,
                    }
                    for item in raw_evidence
                ],
                "evidence_count": len(evidence),
                "deduped_fields": [
                    {
                        "entity_type": item.entity_type,
                        "field_name": item.field_name,
                        "field_label": item.field_label,
                        "normalized_value": item.normalized_value,
                    }
                    for item in evidence
                ],
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            decision_basis=[
                "Evidence（证据）保留来源、置信度和校验状态，供任务归属 Agent 消费。",
                "同字段同值证据会去重，避免重复影响候选评分。",
            ],
        )
        risks = self._build_risks(event, documents, evidence)
        conflicts = []
        next_route = "manual_review" if any(r.risk_level == RiskLevel.high for r in risks) else "task_attribution"
        if not documents and not evidence:
            next_route = "ignore"
        current_intent = self._canonical_model_intent(model_payload.get("current_intent")) if model_payload else None
        speech_act = model_payload.get("speech_act") if model_payload else None
        trace.add(
            phase="route",
            step_name="材料路由判断",
            action="根据材料、证据和风险决定下一步服务流程",
            output_snapshot={
                "risk_flags": [item.model_dump(mode="json") for item in risks],
                "conflict_candidates": conflicts,
                "next_route": next_route,
                "current_intent": current_intent
                or self._infer_current_intent(event.content_text, evidence, documents),
                "speech_act": speech_act or self._infer_speech_act(event.content_text, evidence),
            },
            decision_basis=[
                "高风险材料进入 manual_review（人工复核）。",
                "无材料无证据时进入 ignore（忽略），其余进入 task_attribution（任务归属）。",
            ],
            risk_notes=[item.description for item in risks],
            branch=next_route,
        )
        result = MaterialUnderstandingResult(
            event_id=event.event_id,
            model_runtime=model_runtime,
            input_summary={
                "message_type": event.message_type.value,
                "has_text": bool(event.content_text.strip()),
                "attachment_count": len(event.attachments),
                "uploaded_attachment_count": len(
                    [item for item in event.attachments if item.storage_path or item.download_url]
                ),
                "multimodal_attachment_count": len(
                    [item for item in event.attachments if item.file_type in {"image", "pdf"}]
                ),
                "quoted_message_present": event.quoted_context is not None,
                "quoted_context_policy": "display_link_only_not_slot_source",
            },
            current_intent=current_intent or self._infer_current_intent(event.content_text, evidence, documents),
            speech_act=speech_act or self._infer_speech_act(event.content_text, evidence),
            vehicle_mentions=self._vehicle_mentions(event.content_text, evidence, model_payload),
            quote_plan_mentions=self._quote_plan_mentions(event.content_text, evidence, model_payload),
            party_mentions=self._party_mentions(evidence, model_payload),
            quoted_link=self._quoted_link(event),
            documents=documents,
            evidence_list=evidence,
            risk_flags=risks,
            conflict_candidates=conflicts,
            material_action_hint=MaterialActionHint(
                next_route=next_route,
                suggested_follow_up=self._suggest_follow_up(risks),
                reason=self._action_reason(next_route, documents, evidence),
            ),
            trace_log=trace.export(),
        )
        return result

    def _try_model_understanding(self, event: MessageEvent) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        status = self.runtime.status()
        if self.use_model is False:
            status.enabled = False
            status.using_real_model = False
            status.reason = "测试显式关闭真实模型，使用离线规则 fallback（兜底逻辑）。"
            return None, self._runtime_dict(status, fallback_used=True)
        if not status.enabled or self.use_model is False:
            return None, self._runtime_dict(status, fallback_used=True)
        if self.use_model is None and bool(re.match(r"^(1|true|yes)$", str(__import__("os").getenv("AGENTIC_LAB_DISABLE_MODEL", "")).lower())):
            status.enabled = False
            status.using_real_model = False
            status.reason = "环境变量 AGENTIC_LAB_DISABLE_MODEL 已关闭真实模型。"
            return None, self._runtime_dict(status, fallback_used=True)

        call = self.runtime.structured_call_sync(
            system_prompt=MATERIAL_AGENT_INSTRUCTIONS,
            user_payload={
                "agent": "material_understanding",
                "rule": "只理解 content_text；quoted_context 只做 quoted_link，不参与槽位抽取。",
                "content_text": event.content_text,
                "message_type": event.message_type.value,
                "attachments": [
                    {
                        **item.model_dump(mode="json"),
                        "binary_available": self.attachment_storage.resolve(item) is not None,
                    }
                    for item in event.attachments
                ],
                "quoted_context": event.quoted_context.model_dump(mode="json") if event.quoted_context else None,
                "expected_json_keys": [
                    "current_intent",
                    "speech_act",
                    "vehicle_mentions",
                    "quote_plan_mentions",
                    "party_mentions",
                    "documents",
                    "evidence_list",
                    "reasoning_steps",
                    "uncertainties",
                ],
            },
            multimodal_content=self._multimodal_content(event),
            output_schema={
                "type": "object",
                "properties": {
                    "current_intent": {"type": "string"},
                    "speech_act": {"type": "string"},
                    "vehicle_mentions": {"type": "array", "items": {"type": "object"}},
                    "quote_plan_mentions": {"type": "array", "items": {"type": "object"}},
                    "party_mentions": {"type": "array", "items": {"type": "object"}},
                    "documents": {"type": "array", "items": {"type": "object"}},
                    "evidence_list": {"type": "array", "items": {"type": "object"}},
                    "reasoning_steps": {"type": "array", "items": {"type": "string"}},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
            },
        )
        runtime = self._runtime_dict(call.status, fallback_used=not call.ok, error=call.error)
        if call.raw_text is not None:
            runtime["raw_model_text"] = call.raw_text
        if call.content is not None:
            runtime["model_output_keys"] = sorted(call.content.keys())
        return call.content if call.ok else None, runtime

    def _multimodal_content(self, event: MessageEvent) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for attachment in event.attachments:
            if attachment.file_type not in {"image", "pdf"}:
                continue
            path = self.attachment_storage.resolve(attachment)
            if not path:
                continue
            mime_type = attachment.mime_type or mimetypes.guess_type(path.name)[0]
            if not mime_type:
                mime_type = "application/pdf" if attachment.file_type == "pdf" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            data_url = f"data:{mime_type};base64,{encoded}"
            if attachment.file_type == "image":
                content.append({"type": "input_image", "image_url": data_url})
            elif attachment.file_type == "pdf":
                content.append(
                    {
                        "type": "input_file",
                        "filename": attachment.original_name or attachment.file_ref,
                        "file_data": data_url,
                    }
                )
        return content

    def _runtime_dict(self, status: Any, fallback_used: bool, error: str | None = None) -> dict[str, Any]:
        payload = status.__dict__.copy()
        payload["fallback_used"] = fallback_used
        payload["fallback_policy"] = (
            "仅用于离线单元测试和结构验证；真实能力验证需要配置模型密钥。"
            if fallback_used
            else "真实大模型已参与理解。"
        )
        if error:
            payload["error"] = error
        return payload

    def _classify_documents(self, event: MessageEvent) -> list[DocumentResult]:
        documents: list[DocumentResult] = []
        for attachment in event.attachments:
            doc_type, confidence = self._classify_attachment(attachment, event.content_text)
            meta = DOCUMENT_TYPES.get(doc_type)
            if not meta:
                documents.append(
                    DocumentResult(
                        document_id=new_id("doc"),
                        document_type="unknown",
                        document_name="未知材料",
                        document_category="unknown",
                        standard_level="unknown",
                        document_intent="needs_clarification",
                        context_relation="current_message_material",
                        confidence=0.2,
                        quality=DocumentQuality(clarity="medium", quality_score=0.5),
                        source_refs=[SourceRef(event_id=event.event_id, attachment_id=attachment.attachment_id)],
                    )
                )
                continue

            documents.append(
                DocumentResult(
                    document_id=new_id("doc"),
                    document_type=doc_type,
                    document_name=meta["name"],
                    document_category=meta["category"],
                    standard_level=(
                        "standard"
                        if meta["category"] == "standard_certificate"
                        else "non_standard"
                    ),
                    quality=self._quality_from_attachment(attachment),
                    usable_for=self._usable_for(doc_type),
                    document_intent=self._document_intent(doc_type),
                    extractable_slots=DOCUMENT_FIELD_MATRIX.get(doc_type, []),
                    context_relation="current_message_material",
                    confidence=confidence,
                    source_refs=[SourceRef(event_id=event.event_id, attachment_id=attachment.attachment_id)],
                )
            )
        return documents

    def _merge_model_documents(
        self, event: MessageEvent, model_payload: dict[str, Any] | None
    ) -> list[DocumentResult]:
        local_documents = self._classify_documents(event)
        model_documents = self._model_documents(event, model_payload)
        if not model_documents:
            return local_documents

        covered_attachment_ids = {
            ref.attachment_id
            for document in model_documents
            for ref in document.source_refs
            if ref.attachment_id
        }
        merged = [*model_documents]
        for document in local_documents:
            attachment_ids = {ref.attachment_id for ref in document.source_refs if ref.attachment_id}
            if attachment_ids and attachment_ids <= covered_attachment_ids:
                continue
            merged.append(document)
        return merged

    def _model_documents(
        self, event: MessageEvent, model_payload: dict[str, Any] | None
    ) -> list[DocumentResult]:
        documents: list[DocumentResult] = []
        attachment_ids = {item.attachment_id for item in event.attachments}
        for raw in (model_payload or {}).get("documents", []) or []:
            if not isinstance(raw, dict):
                continue
            attachment_id = raw.get("attachment_id")
            if attachment_id and attachment_id not in attachment_ids:
                continue
            doc_type = str(raw.get("document_type") or "unknown")
            meta = DOCUMENT_TYPES.get(doc_type)
            category = raw.get("document_category") or (meta or {}).get("category") or "unknown"
            if category not in {"standard_certificate", "non_standard", "textual", "unknown"}:
                category = "unknown"
            slots = raw.get("extractable_slots") or DOCUMENT_FIELD_MATRIX.get(doc_type, [])
            if not isinstance(slots, list):
                slots = []
            confidence = self._safe_confidence(raw.get("confidence"), 0.78)
            raw_quality = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}
            clarity = raw.get("clarity") or raw_quality.get("clarity")
            if clarity not in {"clear", "medium", "blurred", "blocked", "incomplete"}:
                clarity = "clear"
            documents.append(
                DocumentResult(
                    document_id=str(raw.get("document_id") or new_id("doc")),
                    document_type=doc_type,
                    document_name=str(raw.get("document_name") or (meta or {}).get("name") or "未知材料"),
                    document_category=category,  # type: ignore[arg-type]
                    standard_level="standard" if category == "standard_certificate" else category,  # type: ignore[arg-type]
                    quality=DocumentQuality(
                        clarity=clarity,  # type: ignore[arg-type]
                        quality_score=self._safe_confidence(raw.get("quality_score"), confidence),
                    ),
                    usable_for=self._usable_for(doc_type),
                    document_intent=str(raw.get("document_intent") or self._document_intent(doc_type)),
                    extractable_slots=[str(item) for item in slots],
                    context_relation="current_message_material",
                    confidence=confidence,
                    source_refs=[SourceRef(event_id=event.event_id, attachment_id=attachment_id)],
                )
            )
        return documents

    def _classify_attachment(self, attachment: AttachmentRef, text: str) -> tuple[str | None, float]:
        haystack = f"{Path(attachment.file_ref).name} {text}".lower()
        for doc_type, meta in DOCUMENT_TYPES.items():
            if any(keyword.lower() in haystack for keyword in meta["keywords"]):
                return doc_type, 0.88
        if attachment.file_type in {"image", "pdf"}:
            return "unknown", 0.2
        return None, 0.0

    def _quality_from_attachment(self, attachment: AttachmentRef) -> DocumentQuality:
        name = Path(attachment.file_ref).name.lower()
        if any(token in name for token in ["blur", "模糊", "遮挡", "blocked"]):
            return DocumentQuality(clarity="blurred", page_completeness="partial", quality_score=0.35)
        return DocumentQuality()

    def _usable_for(self, doc_type: str) -> list[str]:
        if doc_type in {"vehicle_license", "electronic_vehicle_license", "receipt", "vehicle_certification", "entry_bill"}:
            return ["quote", "insurance", "vehicle_check"]
        if doc_type in {"identity_card_front", "identity_card_back", "business_license", "driving_license", "electronic_driving_license"}:
            return ["quote", "insurance", "identity_check"]
        if doc_type in {"quotation", "other_quotation", "insurance_policy"}:
            return ["quote", "history_reference"]
        return ["reference"]

    def _document_intent(self, doc_type: str) -> str:
        if doc_type in {
            "vehicle_license",
            "electronic_vehicle_license",
            "vehicle_certification",
            "entry_bill",
            "vehicle_register_cert",
        }:
            return "provide_vehicle_identity_for_quote_or_insurance"
        if doc_type in {
            "identity_card_front",
            "identity_card_back",
            "business_license",
            "driving_license",
            "electronic_driving_license",
            "residence_permit",
        }:
            return "provide_party_identity_for_insurance"
        if doc_type in {"quotation", "other_quotation"}:
            return "provide_quote_plan_reference"
        if doc_type == "insurance_policy":
            return "provide_policy_or_history_reference"
        if doc_type in {"receipt", "electronic_receipt"}:
            return "provide_new_vehicle_invoice_material"
        if doc_type in {"letter_of_authorization", "vehicle_delivery_authorization"}:
            return "prove_authorization_or_party_relationship"
        if doc_type == "unknown":
            return "needs_clarification"
        return "provide_auxiliary_reference"

    def _has_text_material(self, event: MessageEvent) -> bool:
        return bool(event.content_text.strip()) or event.message_type in {
            MessageType.text,
            MessageType.voice_transcript,
            MessageType.quoted_message,
        }

    def _textual_document(self, event: MessageEvent) -> DocumentResult:
        return DocumentResult(
            document_id=new_id("doc"),
            document_type="textual_business_material",
            document_name="文本业务材料",
            document_category="textual",
            standard_level="textual",
            usable_for=["quote", "insurance", "task_attribution"],
            confidence=0.82 if event.content_text.strip() else 0.4,
            source_refs=[SourceRef(event_id=event.event_id)],
        )

    def _extract_text_evidence(
        self, event: MessageEvent, documents: list[DocumentResult]
    ) -> list[EvidenceItem]:
        text = event.content_text or ""
        evidence: list[EvidenceItem] = []
        source_document_id = documents[-1].document_id if documents else None
        source_type = "voice_transcript" if event.message_type == MessageType.voice_transcript else "text"

        for match in VIN_RE.finditer(text.upper()):
            value = normalize_vin(match.group(0))
            evidence.append(
                EvidenceItem(
                    entity_type="vehicle",
                    field_name="frameNo",
                    field_label="车架号",
                    raw_value=match.group(0),
                    normalized_value=value,
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.93,
                    evidence_strength=EvidenceStrength.strong,
                    validation=validate_vin(value),
                )
            )

        for match in PLATE_RE.finditer(text.upper()):
            raw = match.group(0)
            # Avoid treating VIN fragments as license plates.
            if len(raw) > 9:
                continue
            evidence.append(
                EvidenceItem(
                    entity_type="vehicle",
                    field_name="vehicleLicenseNo",
                    field_label="车牌号",
                    raw_value=raw,
                    normalized_value=normalize_plate(raw),
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.86,
                    evidence_strength=EvidenceStrength.medium,
                    validation=validate_plate(raw),
                )
            )

        for match in PHONE_RE.finditer(text):
            evidence.append(
                EvidenceItem(
                    entity_type="party",
                    role="contact",
                    field_name="ownerMobile",
                    field_label="手机号",
                    raw_value=match.group(0),
                    normalized_value=match.group(0),
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.9,
                    evidence_strength=EvidenceStrength.medium,
                    validation=validate_phone(match.group(0)),
                )
            )

        for match in ID_NO_RE.finditer(text):
            evidence.append(
                EvidenceItem(
                    entity_type="party",
                    role="owner",
                    field_name="ownerIdNo",
                    field_label="车主证件号",
                    raw_value=match.group(0),
                    normalized_value=match.group(0).upper(),
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.9,
                    evidence_strength=EvidenceStrength.strong,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["id_no_shape"]),
                )
            )

        evidence.extend(self._extract_engine_numbers(text, event, source_document_id, source_type))
        evidence.extend(self._extract_role_names(text, event, source_document_id, source_type))
        evidence.extend(self._extract_loose_owner_names(text, event, source_document_id, source_type))
        evidence.extend(self._extract_insurance_plan(text, event, source_document_id, source_type))
        evidence.extend(self._extract_business_credentials(text, event, source_document_id, source_type))
        evidence.extend(self._extract_dates(text, event, source_document_id, source_type))
        return evidence

    def _model_evidence(
        self,
        event: MessageEvent,
        documents: list[DocumentResult],
        model_payload: dict[str, Any] | None,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        document_by_attachment: dict[str, str] = {}
        for document in documents:
            for ref in document.source_refs:
                if ref.attachment_id:
                    document_by_attachment[ref.attachment_id] = document.document_id

        for raw in (model_payload or {}).get("evidence_list", []) or []:
            if not isinstance(raw, dict):
                continue
            normalized_value = str(raw.get("normalized_value") or raw.get("raw_value") or "").strip()
            if not normalized_value:
                continue
            attachment_id = raw.get("attachment_id")
            source_document_id = raw.get("source_document_id") or document_by_attachment.get(str(attachment_id))
            try:
                evidence.append(
                    EvidenceItem(
                        entity_type=raw.get("entity_type", "credential"),
                        role=raw.get("role", "unknown"),
                        field_name=str(raw.get("field_name") or "extractedField"),
                        field_label=str(raw.get("field_label") or raw.get("field_name") or "抽取字段"),
                        raw_value=str(raw.get("raw_value") or normalized_value),
                        normalized_value=normalized_value,
                        value_type=raw.get("value_type", "string"),
                        source_type="multimodal_model" if event.attachments else "text",
                        source_document_id=source_document_id,
                        source_event_id=event.event_id,
                        confidence=self._safe_confidence(raw.get("confidence"), 0.78),
                        evidence_strength=raw.get("evidence_strength", EvidenceStrength.medium),
                        validation=FieldValidation(
                            status=raw.get("validation_status", ValidationStatus.not_checked),
                            rules=[str(item) for item in raw.get("validation_rules", [])],
                        ),
                    )
                )
            except Exception:
                continue
        return evidence

    def _dedupe_evidence(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        deduped: dict[tuple[str, str, str], EvidenceItem] = {}
        for item in items:
            key = (item.entity_type, item.field_name, item.normalized_value)
            current = deduped.get(key)
            if current is None or item.confidence > current.confidence:
                deduped[key] = item
        return list(deduped.values())

    def _safe_confidence(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, number))

    def _extract_engine_numbers(
        self, text: str, event: MessageEvent, source_document_id: str | None, source_type: str
    ) -> list[EvidenceItem]:
        protected = {match.group(0).upper() for match in VIN_RE.finditer(text.upper())}
        protected.update(
            match.group(0).upper().replace(" ", "").replace("-", "")
            for match in PLATE_RE.finditer(text.upper())
        )
        evidence: list[EvidenceItem] = []
        for match in ENGINE_NO_RE.finditer(text.upper()):
            value = match.group(0).upper()
            if len(value) == 17 or value.isdigit() or not any(char.isalpha() for char in value):
                continue
            if any(value == item or value in item for item in protected):
                continue
            evidence.append(
                EvidenceItem(
                    entity_type="vehicle",
                    field_name="engineNo",
                    field_label="发动机号",
                    raw_value=match.group(0),
                    normalized_value=value,
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.74,
                    evidence_strength=EvidenceStrength.medium,
                    validation=FieldValidation(status=ValidationStatus.not_checked, rules=["engine_no_shape"]),
                )
            )
        return evidence

    def _extract_role_names(
        self, text: str, event: MessageEvent, source_document_id: str | None, source_type: str
    ) -> list[EvidenceItem]:
        role_map = {
            "车主": ("owner", "ownerName", "车主姓名"),
            "投保人": ("applicant", "applicantName", "投保人姓名"),
            "被保人": ("insured", "insuredName", "被保人姓名"),
        }
        evidence: list[EvidenceItem] = []
        for label, (role, field, field_label) in role_map.items():
            match = re.search(rf"{label}(?:是|为|[:：])?\s*([\u4e00-\u9fa5]{{2,8}})", text)
            if not match:
                continue
            value = match.group(1)
            evidence.append(
                EvidenceItem(
                    entity_type="party",
                    role=role,  # type: ignore[arg-type]
                    field_name=field,
                    field_label=field_label,
                    raw_value=value,
                    normalized_value=value,
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.82,
                    evidence_strength=EvidenceStrength.medium,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["person_name_text"]),
                )
            )
        return evidence

    def _extract_loose_owner_names(
        self, text: str, event: MessageEvent, source_document_id: str | None, source_type: str
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        stop_words = {
            "这台报价",
            "帮这台",
            "先算下",
            "车架号",
            "手机号",
            "证件号",
            "身份证",
            "三者",
            "非车",
            "今晚起",
            "今天起",
            "明天生效",
            "即刻生效",
        }
        for match in re.finditer(
            rf"([\u4e00-\u9fa5]{{2,4}})[\s，,、/]+(?:{PHONE_RE.pattern}|{ID_NO_RE.pattern})",
            text,
        ):
            value = match.group(1)
            if value in stop_words or any(word in value for word in ["报价", "车架", "证件", "手机号"]):
                continue
            evidence.append(
                EvidenceItem(
                    entity_type="party",
                    role="owner",
                    field_name="ownerName",
                    field_label="车主姓名",
                    raw_value=value,
                    normalized_value=value,
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.76,
                    evidence_strength=EvidenceStrength.medium,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["person_name_near_contact_or_id"]),
                )
            )
        return evidence

    def _extract_insurance_plan(
        self, text: str, event: MessageEvent, source_document_id: str | None, source_type: str
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        normalized_text = text.strip()
        third_party = re.search(r"三者(?:险)?(?:改成|改到|改|保|要|到)?\s*(\d{2,4})\s*(?:万|w|W)?", text)
        if not third_party and re.search(r"(?:改成|改到|改|保额|额度)", text):
            third_party = re.search(r"(\d{2,4})\s*(?:万|w|W)", text)
        if third_party:
            amount = f"{third_party.group(1)}万"
            evidence.append(
                EvidenceItem(
                    entity_type="insurance_plan",
                    field_name="thirdPartyLiabilityInsuranceForMotorVehicles",
                    field_label="机动车第三者责任保险",
                    raw_value=third_party.group(0),
                    normalized_value=amount,
                    value_type="amount",
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.88,
                    evidence_strength=EvidenceStrength.medium,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["coverage_amount"]),
                )
            )
            evidence.append(self._intent_evidence("modify_quote", "修改报价", event, source_document_id, source_type))

        product_match = None
        if any(token in text for token in ["不要非车", "去掉非车", "不要288", "不要 288"]):
            evidence.append(
                EvidenceItem(
                    entity_type="non_auto_product",
                    field_name="unAutoProductInfo",
                    field_label="随车非车",
                    raw_value=text,
                    normalized_value="remove_non_auto",
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.86,
                    evidence_strength=EvidenceStrength.medium,
                )
            )
            evidence.append(self._intent_evidence("modify_quote", "修改报价", event, source_document_id, source_type))
        else:
            product_match = re.search(r"(?:非车|随车|驾意|那款|款)[^\d]{0,8}(\d{2,4})", text)
            if not product_match:
                product_match = re.search(r"(\d{2,4})[^\d]{0,4}(?:非车|随车|驾意|款)", text)
        if product_match:
            products = product_match.group(1)
            evidence.append(
                EvidenceItem(
                    entity_type="non_auto_product",
                    field_name="unAutoProductInfo",
                    field_label="随车非车",
                    raw_value=text,
                    normalized_value=products,
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.76,
                    evidence_strength=EvidenceStrength.medium,
                )
            )
            evidence.append(self._intent_evidence("modify_quote", "修改报价", event, source_document_id, source_type))

        if any(token in text for token in ["即刻生效", "即时生效", "今天开走", "今天要开"]):
            evidence.append(
                EvidenceItem(
                    entity_type="insurance_plan",
                    field_name="timelyInsurance",
                    field_label="是否即刻生效",
                    raw_value=text,
                    normalized_value="true",
                    value_type="boolean",
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.84,
                    evidence_strength=EvidenceStrength.medium,
                )
            )

        for regex, field_name, field_label in [
            (COMMERCIAL_PREMIUM_RE, "commercialPremium", "商业险保费"),
            (COMPULSORY_PREMIUM_RE, "compulsoryPremium", "交强险保费"),
        ]:
            match = regex.search(text)
            if match:
                evidence.append(
                    EvidenceItem(
                        entity_type="insurance_plan",
                        field_name=field_name,
                        field_label=field_label,
                        raw_value=match.group(0),
                        normalized_value=match.group(1),
                        value_type="amount",
                        source_type=source_type,
                        source_document_id=source_document_id,
                        source_event_id=event.event_id,
                        confidence=0.84,
                        evidence_strength=EvidenceStrength.medium,
                        validation=FieldValidation(status=ValidationStatus.valid, rules=["premium_amount"]),
                    )
                )

        if normalized_text == "确认" or any(token in text for token in ["确认出单", "确认报价", "按这个出", "出单"]):
            evidence.append(self._intent_evidence("confirm_quote", "确认报价单", event, source_document_id, source_type))
        if any(token in text for token in ["核保一下", "核保", "待审核", "审核中"]):
            evidence.append(self._intent_evidence("underwriting_request", "核保/审核处理", event, source_document_id, source_type))
        if any(token in text for token in ["好了没", "咋样", "到哪步", "还没好"]):
            evidence.append(self._intent_evidence("progress_query", "催促/查询进度", event, source_document_id, source_type))
        if any(token in text for token in ["二维码过期", "二维码发", "发二维码"]):
            evidence.append(self._intent_evidence("payment_code_request", "发送二维码", event, source_document_id, source_type))
        if any(token in text for token in ["支付已经完成", "支付完成", "已支付"]):
            evidence.append(self._intent_evidence("payment_completed", "支付完成", event, source_document_id, source_type))
        if any(token in text for token in ["保单发", "发保单", "电子单证"]):
            evidence.append(self._intent_evidence("send_policy", "发送保单", event, source_document_id, source_type))
        if "保单" in text and (event.attachments or text.lower().endswith(".pdf")):
            evidence.append(self._intent_evidence("policy_delivered", "保单已发送", event, source_document_id, source_type))
        if any(token in text for token in ["发票", "开票"]):
            if any(token in text for token in ["开具中", "发票开具中"]):
                evidence.append(self._intent_evidence("invoice_status", "发票开具中", event, source_document_id, source_type))
            else:
                evidence.append(self._intent_evidence("invoice_request", "发票处理", event, source_document_id, source_type))
        if any(token in text for token in ["报价完成", "报价已出", "报价已完成"]):
            evidence.append(self._intent_evidence("quote_result", "报价完成", event, source_document_id, source_type))
        if any(token in text for token in ["报价正在", "报价中", "正在努力计算", "稍等一下"]):
            evidence.append(self._intent_evidence("quote_status", "报价处理中", event, source_document_id, source_type))
        if any(token in text for token in ["投保ing", "投保中", "投保正在"]):
            evidence.append(self._intent_evidence("insurance_status", "投保处理中", event, source_document_id, source_type))
        if "转人工" in text:
            evidence.append(self._intent_evidence("transfer_manual", "转人工", event, source_document_id, source_type))
        if any(token in text for token in ["不出了", "不出单", "不想出"]):
            evidence.append(self._intent_evidence("abandon_task", "不出单", event, source_document_id, source_type))
        return evidence

    def _extract_business_credentials(
        self, text: str, event: MessageEvent, source_document_id: str | None, source_type: str
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for match in QUOTE_FLOW_NO_RE.finditer(text):
            evidence.append(
                EvidenceItem(
                    entity_type="credential",
                    field_name="quoteFlowNo",
                    field_label="报价流水号",
                    raw_value=match.group(0),
                    normalized_value=match.group(0),
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.82,
                    evidence_strength=EvidenceStrength.medium,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["quote_flow_no_shape"]),
                )
            )
        for match in PROCESS_NO_RE.finditer(text):
            evidence.append(
                EvidenceItem(
                    entity_type="credential",
                    field_name="processNo",
                    field_label="流程号",
                    raw_value=match.group(0),
                    normalized_value=match.group(0).upper(),
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.78,
                    evidence_strength=EvidenceStrength.medium,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["process_no_shape"]),
                )
            )
        return evidence

    def _extract_dates(
        self, text: str, event: MessageEvent, source_document_id: str | None, source_type: str
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for match in DATE_RE.finditer(text):
            raw = match.group(0)
            evidence.append(
                EvidenceItem(
                    entity_type="insurance_plan",
                    field_name="bizBeginDate",
                    field_label="商业险起期/日期",
                    raw_value=raw,
                    normalized_value=normalize_date(raw),
                    value_type="date",
                    source_type=source_type,
                    source_document_id=source_document_id,
                    source_event_id=event.event_id,
                    confidence=0.74,
                    evidence_strength=EvidenceStrength.weak,
                    validation=FieldValidation(status=ValidationStatus.valid, rules=["date_shape"]),
                )
            )
        return evidence

    def _quoted_link(self, event: MessageEvent) -> QuotedLink | None:
        if not event.quoted_context:
            return None
        quoted_text = event.quoted_context.quoted_text or ""
        return QuotedLink(
            quoted_message_id=event.quoted_context.quoted_message_id,
            quoted_type=event.quoted_context.quoted_type,
            quoted_text_preview=quoted_text[:80] if quoted_text else None,
        )

    def _canonical_model_intent(self, intent: Any) -> str | None:
        if not intent:
            return None
        value = str(intent).strip()
        if value in CANONICAL_INTENTS:
            return value
        mapping = [
            ("批量", "batch_modify_quote"),
            ("修改", "modify_quote"),
            ("保额", "modify_quote"),
            ("非车", "modify_quote"),
            ("确认", "confirm_quote"),
            ("出单", "confirm_quote"),
            ("核保", "underwriting_request"),
            ("审核", "underwriting_request"),
            ("二维码", "payment_code_request"),
            ("支付完成", "payment_completed"),
            ("保单", "send_policy"),
            ("发票", "invoice_request"),
            ("报价完成", "quote_result"),
            ("报价中", "quote_status"),
            ("投保", "insurance_status"),
            ("催", "progress_query"),
            ("进度", "progress_query"),
            ("转人工", "transfer_manual"),
        ]
        for keyword, canonical in mapping:
            if keyword in value:
                return canonical
        return None

    def _infer_current_intent(
        self, text: str, evidence: list[EvidenceItem], documents: list[DocumentResult] | None = None
    ) -> str:
        intents = {item.normalized_value for item in evidence if item.entity_type == "intent"}
        if "modify_quote" in intents:
            if self._has_task_set_scope(text):
                return "batch_modify_quote"
            return "modify_quote"
        if "confirm_quote" in intents:
            return "confirm_quote"
        if "underwriting_request" in intents:
            return "underwriting_request"
        if "payment_code_request" in intents:
            return "payment_code_request"
        if "payment_completed" in intents:
            return "payment_completed"
        if "progress_query" in intents:
            return "progress_query"
        if "send_policy" in intents:
            return "send_policy"
        if "policy_delivered" in intents:
            return "policy_delivered"
        if "invoice_request" in intents:
            return "invoice_request"
        if "invoice_status" in intents:
            return "invoice_status"
        if "quote_result" in intents:
            return "quote_result"
        if "quote_status" in intents:
            return "quote_status"
        if "insurance_status" in intents:
            return "insurance_status"
        if "transfer_manual" in intents:
            return "transfer_manual"
        if "abandon_task" in intents:
            return "abandon_task"
        vin_count = len([item for item in evidence if item.field_name == "frameNo"])
        if vin_count > 1:
            return "new_batch_quote"
        if vin_count == 1:
            return "new_or_existing_quote"
        if any(item.entity_type == "party" for item in evidence):
            return "supplement_party_info"
        if any(item.entity_type == "credential" for item in evidence):
            return "business_status_or_reference"
        document_types = {document.document_type for document in documents or []}
        if "insurance_policy" in document_types:
            return "policy_delivered"
        if document_types & {"quotation", "other_quotation"}:
            return "quote_result"
        if document_types:
            return "supplement_attachment_material"
        if text.strip() == "" and evidence == []:
            return "supplement_attachment_material"
        if text.strip() in {"嗯", "好的", "收到", "谢谢"}:
            return "chat"
        return "unknown"

    def _infer_speech_act(self, text: str, evidence: list[EvidenceItem]) -> str:
        if any(item.entity_type == "insurance_plan" for item in evidence):
            return "change_request"
        if any(item.field_name == "frameNo" for item in evidence):
            return "provide_vehicle_identifier"
        if any(item.entity_type == "party" for item in evidence):
            return "provide_party_info"
        if any(item.entity_type == "credential" for item in evidence):
            return "provide_business_credential"
        if any(item.entity_type == "intent" for item in evidence):
            return "business_instruction"
        if text.strip() == "":
            return "provide_attachment_material"
        if text.strip() in {"嗯", "好的", "收到", "谢谢"}:
            return "chat_ack"
        return "unknown"

    def _vehicle_mentions(
        self, text: str, evidence: list[EvidenceItem], model_payload: dict[str, Any] | None
    ) -> list[VehicleMention]:
        mentions: list[VehicleMention] = []
        for item in evidence:
            if item.field_name != "frameNo":
                continue
            mentions.append(
                VehicleMention(
                    frame_no=item.normalized_value,
                    source_text=item.raw_value,
                    confidence=item.confidence,
                )
            )
        if mentions:
            return mentions
        for raw in (model_payload or {}).get("vehicle_mentions", []) or []:
            try:
                mentions.append(VehicleMention.model_validate(raw))
            except Exception:
                continue
        return mentions

    def _quote_plan_mentions(
        self, text: str, evidence: list[EvidenceItem], model_payload: dict[str, Any] | None
    ) -> list[QuotePlanMention]:
        mentions: list[QuotePlanMention] = []
        scope = "task_set" if self._has_task_set_scope(text) else "single_task"
        for item in evidence:
            if item.entity_type not in {"insurance_plan", "non_auto_product"}:
                continue
            operation = "remove" if item.normalized_value == "remove_non_auto" else "set"
            mentions.append(
                QuotePlanMention(
                    field_name=item.field_name,
                    field_label=item.field_label,
                    raw_value=item.raw_value,
                    normalized_value=item.normalized_value,
                    operation=operation,
                    scope_hint=scope,
                    confidence=item.confidence,
                )
            )
        if mentions:
            return mentions
        for raw in (model_payload or {}).get("quote_plan_mentions", []) or []:
            try:
                mentions.append(QuotePlanMention.model_validate(raw))
            except Exception:
                continue
        return mentions

    def _party_mentions(
        self, evidence: list[EvidenceItem], model_payload: dict[str, Any] | None
    ) -> list[PartyMention]:
        mentions: list[PartyMention] = []
        for item in evidence:
            if item.entity_type != "party":
                continue
            mentions.append(
                PartyMention(
                    role=item.role,
                    field_name=item.field_name,
                    field_label=item.field_label,
                    raw_value=item.raw_value,
                    normalized_value=item.normalized_value,
                    confidence=item.confidence,
                )
            )
        if mentions:
            return mentions
        for raw in (model_payload or {}).get("party_mentions", []) or []:
            try:
                mentions.append(PartyMention.model_validate(raw))
            except Exception:
                continue
        return mentions

    def _has_task_set_scope(self, text: str) -> bool:
        return any(token in text for token in ["都", "这几台", "这些车", "全部", "都按", "一样"])

    def _intent_evidence(
        self,
        normalized_value: str,
        label: str,
        event: MessageEvent,
        source_document_id: str | None,
        source_type: str,
    ) -> EvidenceItem:
        return EvidenceItem(
            entity_type="intent",
            field_name="intentHint",
            field_label="意图提示",
            raw_value=label,
            normalized_value=normalized_value,
            source_type=source_type,
            source_document_id=source_document_id,
            source_event_id=event.event_id,
            confidence=0.8,
            evidence_strength=EvidenceStrength.medium,
        )

    def _build_risks(
        self,
        event: MessageEvent,
        documents: list[DocumentResult],
        evidence: list[EvidenceItem],
    ) -> list[RiskFlag]:
        risks: list[RiskFlag] = []
        for document in documents:
            if document.document_type == "unknown":
                risks.append(
                    RiskFlag(
                        risk_type="unknown_document",
                        risk_level=RiskLevel.medium,
                        description="无法识别材料类型，需要人工确认或补充说明。",
                    )
                )
            if document.quality.quality_score < 0.5:
                risks.append(
                    RiskFlag(
                        risk_type="low_quality",
                        risk_level=RiskLevel.high,
                        description="材料清晰度不足，禁止臆造字段。",
                    )
                )
            expected_fields = DOCUMENT_FIELD_MATRIX.get(document.document_type, [])
            if expected_fields and not any(e.field_name in expected_fields for e in evidence):
                risks.append(
                    RiskFlag(
                        risk_type="missing_core_field",
                        risk_level=RiskLevel.low,
                        description=f"{document.document_name}暂未抽取到核心字段，可能需要真实OCR或补传清晰材料。",
                    )
                )

        if event.message_type == MessageType.voice_transcript and evidence:
            risks.append(
                RiskFlag(
                    risk_type="low_confidence",
                    risk_level=RiskLevel.low,
                    description="语音转写字段需保留转写置信度，关键字段建议确认。",
                )
            )
        return risks

    def _suggest_follow_up(self, risks: list[RiskFlag]) -> str | None:
        if any(r.risk_type == "low_quality" for r in risks):
            return "请补充清晰的证件或材料图片。"
        if any(r.risk_type == "unknown_document" for r in risks):
            return "请说明这份材料的用途，或补充标准证件。"
        return None

    def _action_reason(
        self, next_route: str, documents: list[DocumentResult], evidence: list[EvidenceItem]
    ) -> str:
        if next_route == "ignore":
            return "未识别到车险业务材料或可用槽位。"
        if next_route == "manual_review":
            return "材料存在高风险质量问题，需要人工审核。"
        return f"已识别 {len(documents)} 份材料和 {len(evidence)} 条证据，可进入任务归属。"
