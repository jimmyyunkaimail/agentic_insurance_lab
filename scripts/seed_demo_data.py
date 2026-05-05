from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import PartyFact, TaskRecord, TaskStage, TaskStatus
from app.services.storage import JsonStore


def seed(store: JsonStore | None = None) -> JsonStore:
    store = store or JsonStore()
    store.reset()
    tasks = [
        TaskRecord(
            task_id="task_active_quote",
            conversation_id="conv_demo",
            stage=TaskStage.quote_confirming,
            status=TaskStatus.active,
            vehicle_facts={
                "vehicleLicenseNo": "川A12345",
                "frameNo": "LFMKN5BF2N3276921",
                "engineNo": "ENG123",
            },
            party_facts=[PartyFact(role="owner", name="张三", id_no="510101199001011234")],
            quote_plan_summary="三者200万，含非车288",
            last_bot_action="已发送报价单，等待确认",
            pending_question="是否按当前方案投保",
        ),
        TaskRecord(
            task_id="task_active_insure",
            conversation_id="conv_demo",
            stage=TaskStage.insuring,
            status=TaskStatus.active,
            vehicle_facts={
                "vehicleLicenseNo": "沪B88888",
                "frameNo": "LSVAB2BR1N2123456",
            },
            party_facts=[PartyFact(role="owner", name="李四", id_no="310101198805052222")],
            last_bot_action="请补充车主身份证",
        ),
        TaskRecord(
            task_id="task_manual_history",
            conversation_id="conv_demo",
            stage=TaskStage.manual,
            status=TaskStatus.manual,
            vehicle_facts={
                "vehicleLicenseNo": "粤C66666",
                "frameNo": "LGBH52E05NY123456",
            },
            party_facts=[PartyFact(role="owner", name="王五", id_no="440301199212123333")],
            last_bot_action="已转人工",
        ),
    ]
    for task in tasks:
        store.save_task(task)
    return store


if __name__ == "__main__":
    seed()
    print("已写入本地样例任务到 data/agentic_insurance.db")
