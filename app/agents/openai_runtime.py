from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml


CONFIG_PATH = Path("config/model_providers.yaml")


EndpointType = Literal["responses", "chat_completions"]


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    endpoint: EndpointType
    api_key_env: str
    model: str


@dataclass
class ModelRuntimeStatus:
    enabled: bool
    active_provider: str | None
    provider: str | None
    model: str | None
    endpoint: str | None
    base_url: str | None
    api_key_env: str | None
    reason: str
    config_path: str
    package_available: bool = True
    using_real_model: bool = False


@dataclass
class ModelCallResult:
    ok: bool
    content: dict[str, Any] | None
    raw_text: str | None
    status: ModelRuntimeStatus
    error: str | None = None


class ModelRuntime:
    """Multi-provider LLM runtime for local validation.

    The validation bench must not pretend a model call worked. If credentials,
    config, network, or schema parsing fail, the caller receives an explicit
    error and may choose a visible offline test fallback.
    """

    def __init__(self, config_path: Path | str = CONFIG_PATH):
        self.config_path = Path(config_path)

    def status(self) -> ModelRuntimeStatus:
        provider, reason = self.active_provider()
        if not provider:
            return ModelRuntimeStatus(
                enabled=False,
                active_provider=None,
                provider=None,
                model=None,
                endpoint=None,
                base_url=None,
                api_key_env=None,
                reason=reason,
                config_path=str(self.config_path),
            )
        api_key = os.getenv(provider.api_key_env)
        model_configured = bool(provider.model and not provider.model.startswith("<"))
        enabled = bool(api_key and model_configured)
        if not model_configured:
            reason = f"当前供应商 {provider.name} 未配置有效模型名。"
        elif not api_key:
            reason = f"未配置环境变量 {provider.api_key_env}，真实大模型不可用。"
        else:
            reason = f"真实模型可用：{provider.name} / {provider.model}。"
        return ModelRuntimeStatus(
            enabled=enabled,
            active_provider=provider.name,
            provider=provider.name,
            model=provider.model,
            endpoint=provider.endpoint,
            base_url=provider.base_url,
            api_key_env=provider.api_key_env,
            reason=reason,
            config_path=str(self.config_path),
            using_real_model=enabled,
        )

    def active_provider(self) -> tuple[ProviderConfig | None, str]:
        if not self.config_path.exists():
            return None, f"模型配置文件不存在：{self.config_path}。"
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            return None, f"模型配置文件读取失败：{exc}。"
        model_cfg = raw.get("model") or {}
        active_provider = model_cfg.get("active_provider")
        providers = model_cfg.get("providers") or {}
        if not active_provider:
            return None, "模型配置缺少 model.active_provider。"
        provider_cfg = providers.get(active_provider)
        if not provider_cfg:
            return None, f"模型供应商 {active_provider} 未在 providers 中定义。"
        endpoint = provider_cfg.get("endpoint")
        if endpoint not in {"responses", "chat_completions"}:
            return None, f"模型供应商 {active_provider} 的 endpoint 不支持：{endpoint}。"
        required = ["base_url", "api_key_env", "model"]
        missing = [key for key in required if not provider_cfg.get(key)]
        if missing:
            return None, f"模型供应商 {active_provider} 缺少字段：{', '.join(missing)}。"
        return (
            ProviderConfig(
                name=active_provider,
                base_url=str(provider_cfg["base_url"]).rstrip("/"),
                endpoint=endpoint,
                api_key_env=str(provider_cfg["api_key_env"]),
                model=str(provider_cfg["model"]),
            ),
            "ok",
        )

    async def structured_call(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        multimodal_content: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> ModelCallResult:
        status = self.status()
        if not status.enabled:
            return ModelCallResult(
                ok=False,
                content=None,
                raw_text=None,
                status=status,
                error=status.reason,
            )
        provider, reason = self.active_provider()
        if not provider:
            disabled = ModelRuntimeStatus(
                enabled=False,
                active_provider=None,
                provider=None,
                model=None,
                endpoint=None,
                base_url=None,
                api_key_env=None,
                reason=reason,
                config_path=str(self.config_path),
            )
            return ModelCallResult(False, None, None, disabled, reason)

        api_key = os.environ[provider.api_key_env]
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                if provider.endpoint == "responses":
                    raw_text = await self._call_responses(
                        client=client,
                        provider=provider,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        user_payload=user_payload,
                        multimodal_content=multimodal_content,
                        output_schema=output_schema,
                        temperature=temperature,
                    )
                else:
                    raw_text = await self._call_chat_completions(
                        client=client,
                        provider=provider,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        user_payload=user_payload,
                        multimodal_content=multimodal_content,
                        output_schema=output_schema,
                        temperature=temperature,
                    )
            return ModelCallResult(
                ok=True,
                content=self._parse_json_object(raw_text),
                raw_text=raw_text,
                status=self.status(),
            )
        except Exception as exc:
            failed_status = self.status()
            failed_status.enabled = False
            failed_status.using_real_model = False
            failed_status.reason = f"模型调用失败：{exc}"
            return ModelCallResult(
                ok=False,
                content=None,
                raw_text=None,
                status=failed_status,
                error=str(exc),
            )

    def structured_call_sync(self, **kwargs: Any) -> ModelCallResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.structured_call(**kwargs))
        raise RuntimeError("当前线程已有运行中的事件循环，请改用 async structured_call。")

    async def _call_responses(
        self,
        *,
        client: httpx.AsyncClient,
        provider: ProviderConfig,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        multimodal_content: list[dict[str, Any]] | None,
        output_schema: dict[str, Any] | None,
        temperature: float,
    ) -> str:
        input_payload: str | list[dict[str, Any]] = json.dumps(user_payload, ensure_ascii=False)
        if multimodal_content:
            input_payload = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)},
                        *multimodal_content,
                    ],
                }
            ]
        body: dict[str, Any] = {
            "model": provider.model,
            "instructions": system_prompt,
            "input": input_payload,
            "temperature": temperature,
        }
        if output_schema:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_structured_output",
                    "schema": output_schema,
                    "strict": False,
                }
            }
        response = await client.post(
            f"{provider.base_url}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        return self._extract_responses_text(payload)

    async def _call_chat_completions(
        self,
        *,
        client: httpx.AsyncClient,
        provider: ProviderConfig,
        api_key: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        multimodal_content: list[dict[str, Any]] | None,
        output_schema: dict[str, Any] | None,
        temperature: float,
    ) -> str:
        user_content: str | list[dict[str, Any]] = json.dumps(user_payload, ensure_ascii=False)
        if multimodal_content:
            chat_blocks: list[dict[str, Any]] = [
                {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}
            ]
            for block in multimodal_content:
                if block.get("type") == "input_image":
                    chat_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": block.get("image_url", "")},
                        }
                    )
                elif block.get("type") == "input_file":
                    chat_blocks.append(
                        {
                            "type": "text",
                            "text": f"PDF 附件已上传：{block.get('filename', 'attachment.pdf')}。当前 Chat Completions API（聊天补全接口）供应商不保证支持 PDF 原文输入，请优先切换 Responses API（响应接口）供应商做 PDF 多模态解析。",
                        }
                    )
            user_content = chat_blocks
        body: dict[str, Any] = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if output_schema:
            body["extra_body"] = {"schema": output_schema}
        response = await client.post(
            f"{provider.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-Title": "Agentic Insurance Lab",
            },
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _extract_responses_text(self, payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        chunks: list[str] = []
        for output in payload.get("output", []):
            for content in output.get("content", []):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        if chunks:
            return "\n".join(chunks)
        return json.dumps(payload, ensure_ascii=False)

    def _parse_json_object(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("模型输出不是 JSON 对象。")
        return parsed


def runtime_status() -> ModelRuntimeStatus:
    return ModelRuntime().status()


MATERIAL_AGENT_INSTRUCTIONS = """
你是车险材料理解 Agent（智能体）。只负责当前最终发送消息的材料分类、槽位抽取、
字段标准化、风险标记和 Evidence（证据）生成。引用消息只作为 quoted_link（引用关联），
禁止把引用文本当作当前消息槽位来源。禁止创建任务、合并任务、覆盖任务事实或调用报价、
投保、支付接口。

如果输入包含图片或 PDF（便携式文档格式）附件，你需要直接理解附件视觉/文本内容：
1. documents 逐附件输出 attachment_id、document_type、document_name、document_category、
   document_intent、extractable_slots、confidence、clarity、quality_score。
2. evidence_list 只输出从当前消息文本或当前附件中明确可见的槽位，不要根据常识补全。
3. 对行驶证、身份证、营业执照、报价单、保单、购车发票等材料，说明材料类型代表的业务
   意图，例如提供车辆强证据、提供人员身份、提供历史报价参考、发送保单/发票等。
4. quoted_context 只能作为关联显示，不能作为槽位来源。
输出必须是 JSON（结构化数据）。
"""


TASK_ATTRIBUTION_AGENT_INSTRUCTIONS = """
你是车险任务归属 Agent（智能体）。只消费 Evidence（证据）、候选任务和会话记忆，
通过实体关系推理判断消息归属。VIN（车架号）是车辆不可变唯一标识，禁止覆盖已有任务
VIN。多 VIN（车架号）批量报价必须拆成多个独立单任务，不创建父子任务。禁止重新做
OCR（光学字符识别）、禁止生成不存在的任务 ID（标识）。输出必须是 JSON（结构化数据）。
"""
