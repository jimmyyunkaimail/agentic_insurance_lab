from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.agents.material_understanding import MaterialUnderstandingAgent
from app.agents.openai_runtime import ModelRuntime
from app.agents.task_attribution import TaskAttributionAgent
from app.main import app
from app.schemas import (
    CandidateTask,
    ConversationMemory,
    MaterialUnderstandingRequest,
    MessageEvent,
    PolicySnapshot,
    QuotedContext,
    TaskAttributionRequest,
    TaskRecord,
    TaskStage,
)
from app.services.policy import PolicyEngine
from app.services.storage import JsonStore
from fastapi.testclient import TestClient
from scripts.seed_demo_data import seed


class MaterialUnderstandingAgentTest(unittest.TestCase):
    def test_extracts_vin_and_quote_changes(self) -> None:
        agent = MaterialUnderstandingAgent(use_model=False)
        result = agent.run(
            MaterialUnderstandingRequest(content_text="车架号 LFMKN5BF2N3276921，三者改成300万，不要288")
        )
        fields = {item.field_name for item in result.evidence_list}
        self.assertIn("frameNo", fields)
        self.assertIn("thirdPartyLiabilityInsuranceForMotorVehicles", fields)
        self.assertIn("unAutoProductInfo", fields)
        self.assertEqual(result.material_action_hint.next_route, "task_attribution")

    def test_classifies_vehicle_license_attachment(self) -> None:
        agent = MaterialUnderstandingAgent(use_model=False)
        result = agent.run(
            MaterialUnderstandingRequest(
                message_type="image",
                attachments=[
                    {
                        "attachment_id": "att_license",
                        "file_type": "image",
                        "file_ref": "storage/attachments/行驶证正面.jpg",
                    }
                ],
            )
        )
        self.assertEqual(result.documents[0].document_type, "vehicle_license")
        self.assertEqual(result.documents[0].document_intent, "provide_vehicle_identity_for_quote_or_insurance")
        self.assertIn("frameNo", result.documents[0].extractable_slots)

    def test_uses_multimodal_model_payload_for_uploaded_pdf_slots(self) -> None:
        agent = MaterialUnderstandingAgent(use_model=False)
        model_payload = {
            "current_intent": "policy_delivered",
            "documents": [
                {
                    "attachment_id": "att_policy",
                    "document_type": "insurance_policy",
                    "document_name": "电子保单",
                    "document_category": "non_standard",
                    "document_intent": "provide_policy_or_history_reference",
                    "extractable_slots": ["policyNo", "frameNo", "ownerName"],
                    "confidence": 0.91,
                }
            ],
            "evidence_list": [
                {
                    "attachment_id": "att_policy",
                    "entity_type": "credential",
                    "field_name": "policyNo",
                    "field_label": "保单号",
                    "raw_value": "PDD202605050001",
                    "normalized_value": "PDD202605050001",
                    "confidence": 0.89,
                }
            ],
        }
        with patch.object(agent, "_try_model_understanding", return_value=(model_payload, {"fallback_used": False})):
            result = agent.run(
                MaterialUnderstandingRequest(
                    message_type="file",
                    attachments=[
                        {
                            "attachment_id": "att_policy",
                            "file_type": "pdf",
                            "file_ref": "保单.pdf",
                            "storage_path": "storage/attachments/保单.pdf",
                        }
                    ],
                )
            )
        self.assertEqual(result.current_intent, "policy_delivered")
        self.assertEqual(result.documents[0].document_type, "insurance_policy")
        self.assertEqual(result.evidence_list[0].field_name, "policyNo")
        self.assertEqual(result.evidence_list[0].source_type, "multimodal_model")

    def test_quote_context_does_not_pollute_current_slots(self) -> None:
        agent = MaterialUnderstandingAgent(use_model=False)
        result = agent.run(
            MaterialUnderstandingRequest(
                content_text="谢谢",
                quoted_context=QuotedContext(
                    quoted_message_id="msg_old",
                    quoted_text="车架号 LFMKN5BF2N3276921，三者300万",
                    quoted_type="user_message",
                ),
            )
        )
        fields = {item.field_name for item in result.evidence_list}
        self.assertNotIn("frameNo", fields)
        self.assertIsNotNone(result.quoted_link)


class TaskAttributionAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = seed(JsonStore("data/test_agents.db"))
        self.material_agent = MaterialUnderstandingAgent(use_model=False)
        self.attribution_agent = TaskAttributionAgent(use_model=False)

    def _request(self, text: str) -> TaskAttributionRequest:
        event = MessageEvent(content_text=text)
        material = self.material_agent.run(event)
        tasks = self.store.list_tasks(event.conversation_id)
        active = [task for task in tasks if task.status.value == "active"]
        candidates = [
            CandidateTask(
                task_id=task.task_id,
                match_reason="same_conversation",
                stage=task.stage,
                status=task.status,
                last_active_at=task.last_active_at,
                task=task,
            )
            for task in tasks
        ]
        return TaskAttributionRequest(
            event=event,
            new_evidence=material.evidence_list,
            active_tasks=active,
            candidate_tasks=candidates,
            policy_snapshot=PolicySnapshot(),
        )

    def test_attaches_to_active_task_by_vin(self) -> None:
        decision = self.attribution_agent.run(self._request("车架号 LFMKN5BF2N3276921"))
        self.assertEqual(decision.decision.value, "attach_to_active_task")
        self.assertEqual(decision.selected_task_id, "task_active_quote")

    def test_holds_when_only_party_with_multiple_active_tasks(self) -> None:
        decision = self.attribution_agent.run(self._request("车主张三"))
        self.assertEqual(decision.decision.value, "hold_for_confirmation")
        self.assertTrue(decision.required_confirmation)

    def test_policy_blocks_high_risk_confirmation(self) -> None:
        decision = self.attribution_agent.run(self._request("车主王五"))
        policy = PolicyEngine().check_attribution(decision)
        self.assertIn(policy.result, {"require_confirmation", "route_to_manual"})

    def test_ellipsis_quote_change_uses_focused_task_memory(self) -> None:
        task = TaskRecord(
            task_id="Task_000001",
            conversation_id="conv_demo",
            stage=TaskStage.quote_preparing,
            vehicle_facts={"frameNo": "LFMKN5BF2N3276921"},
            quote_plan_summary="三者300万，不要288",
        )
        event = MessageEvent(content_text="改成500W吧，非车要548那款")
        material = self.material_agent.run(event)
        decision = self.attribution_agent.run(
            TaskAttributionRequest(
                event=event,
                new_evidence=material.evidence_list,
                active_tasks=[task],
                candidate_tasks=[
                    CandidateTask(
                        task_id=task.task_id,
                        match_reason="local",
                        stage=task.stage,
                        status=task.status,
                        task=task,
                    )
                ],
                conversation_memory=ConversationMemory(focused_task_id="Task_000001"),
                policy_snapshot=PolicySnapshot(),
            )
        )
        self.assertEqual(decision.decision.value, "attach_to_active_task")
        self.assertEqual(decision.selected_task_id, "Task_000001")

    def test_multiple_vins_create_multiple_independent_tasks(self) -> None:
        event = MessageEvent(
            content_text=(
                "LBV41DU01NS563230\n"
                "LBVKY1105LSX96031\n"
                "LBVHY5105NM447881\n"
                "LBV6R4101NM429666\n"
                "LBVHZ1109HMJ96311"
            )
        )
        material = self.material_agent.run(event)
        decision = self.attribution_agent.run(
            TaskAttributionRequest(
                event=event,
                new_evidence=material.evidence_list,
                active_tasks=[],
                candidate_tasks=[],
                policy_snapshot=PolicySnapshot(),
            )
        )
        self.assertEqual(decision.decision.value, "create_multiple_tasks")
        self.assertEqual(decision.new_task_hint["task_mode"], "independent_single_tasks")
        self.assertEqual(len(decision.new_task_hint["vehicle_mentions"]), 5)
        self.assertTrue(decision.new_task_hint["no_parent_task"])

    def test_batch_update_attaches_to_recent_task_set(self) -> None:
        tasks = [
            TaskRecord(task_id="Task_000001", conversation_id="conv_demo", vehicle_facts={"frameNo": "LBV41DU01NS563230"}),
            TaskRecord(task_id="Task_000002", conversation_id="conv_demo", vehicle_facts={"frameNo": "LBVKY1105LSX96031"}),
        ]
        event = MessageEvent(content_text="三者都改500万")
        material = self.material_agent.run(event)
        decision = self.attribution_agent.run(
            TaskAttributionRequest(
                event=event,
                new_evidence=material.evidence_list,
                active_tasks=tasks,
                candidate_tasks=[
                    CandidateTask(task_id=task.task_id, match_reason="local", stage=task.stage, status=task.status, task=task)
                    for task in tasks
                ],
                conversation_memory=ConversationMemory(recent_task_set_ids=["Task_000001", "Task_000002"]),
                policy_snapshot=PolicySnapshot(),
            )
        )
        self.assertEqual(decision.decision.value, "attach_to_task_set")
        self.assertEqual(decision.selected_task_ids, ["Task_000001", "Task_000002"])

    def test_new_vin_does_not_attach_to_focused_task(self) -> None:
        task = TaskRecord(
            task_id="Task_000001",
            conversation_id="conv_demo",
            stage=TaskStage.quote_preparing,
            vehicle_facts={"frameNo": "LFMKN5BF2N3276921"},
        )
        event = MessageEvent(content_text="车架号 LGWEF6A59NH123456")
        material = self.material_agent.run(event)
        decision = self.attribution_agent.run(
            TaskAttributionRequest(
                event=event,
                new_evidence=material.evidence_list,
                active_tasks=[task],
                candidate_tasks=[
                    CandidateTask(task_id=task.task_id, match_reason="local", stage=task.stage, status=task.status, task=task)
                ],
                conversation_memory=ConversationMemory(focused_task_id="Task_000001"),
                policy_snapshot=PolicySnapshot(),
            )
        )
        self.assertEqual(decision.decision.value, "create_new_task")


class ModelRuntimeTest(unittest.TestCase):
    def test_missing_key_disables_real_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "model_providers.yaml"
            config_path.write_text(
                """
model:
  active_provider: apiopencc
  providers:
    apiopencc:
      base_url: "https://apiopencc.com/v1"
      endpoint: "responses"
      api_key_env: "APIOPENCC_API_KEY_TEST_ONLY"
      model: "demo-model"
""",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                status = ModelRuntime(config_path).status()
        self.assertFalse(status.enabled)
        self.assertIn("APIOPENCC_API_KEY_TEST_ONLY", status.reason)


class AttachmentUploadApiTest(unittest.TestCase):
    def test_upload_returns_attachment_ref_with_storage_metadata(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/attachments/upload",
            files={"file": ("行驶证.jpg", b"fake-image-bytes", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        attachment = response.json()["attachment"]
        self.assertEqual(attachment["file_type"], "image")
        self.assertEqual(attachment["original_name"], "行驶证.jpg")
        self.assertTrue(attachment["storage_path"].endswith(".jpg"))
        self.assertIn("/attachments/", attachment["download_url"])


if __name__ == "__main__":
    unittest.main()
