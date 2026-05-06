from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml


CONFIG_PATH = Path("config/model_providers.yaml")
SECRETS_PATH = Path("config/model_secrets.local.yaml")


EndpointType = Literal["responses", "chat_completions"]


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    endpoint: EndpointType
    api_key_env: str
    model: str
    input_modalities: list[str] = field(default_factory=lambda: ["text"])


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
    secrets_path: str | None = None
    api_key_source: str | None = None
    input_modalities: list[str] = field(default_factory=list)
    routed_from: str | None = None
    route_reason: str = "default"
    routing_enabled: bool = False
    available_routes: dict[str, str] = field(default_factory=dict)
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

    def __init__(
        self,
        config_path: Path | str = CONFIG_PATH,
        secrets_path: Path | str = SECRETS_PATH,
    ):
        self.config_path = Path(config_path)
        self.secrets_path = Path(secrets_path)

    def status(
        self,
        provider: ProviderConfig | None = None,
        *,
        routed_from: str | None = None,
        route_reason: str = "default",
    ) -> ModelRuntimeStatus:
        if provider:
            reason = "ok"
        else:
            provider, reason = self.active_provider()
        routing_enabled, available_routes = self._routing_settings()
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
                secrets_path=str(self.secrets_path),
                routing_enabled=routing_enabled,
                available_routes=available_routes,
            )
        api_key, api_key_source, key_reason = self.resolve_api_key(provider)
        model_configured = bool(provider.model and not provider.model.startswith("<"))
        enabled = bool(api_key and model_configured)
        if not model_configured:
            reason = f"当前供应商 {provider.name} 未配置有效模型名。"
        elif not api_key:
            reason = key_reason
        else:
            reason = f"真实模型可用：{provider.name} / {provider.model}；密钥来源：{api_key_source}。"
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
            secrets_path=str(self.secrets_path),
            api_key_source=api_key_source,
            input_modalities=provider.input_modalities,
            routed_from=routed_from,
            route_reason=route_reason,
            routing_enabled=routing_enabled,
            available_routes=available_routes,
            using_real_model=enabled,
        )

    def active_provider(self) -> tuple[ProviderConfig | None, str]:
        model_cfg, reason = self._load_model_cfg()
        if model_cfg is None:
            return None, reason
        active_provider = model_cfg.get("active_provider")
        providers = model_cfg.get("providers") or {}
        if not active_provider:
            return None, "模型配置缺少 model.active_provider。"
        return self._provider_from_config(str(active_provider), providers)

    def _load_model_cfg(self) -> tuple[dict[str, Any] | None, str]:
        if not self.config_path.exists():
            return None, f"模型配置文件不存在：{self.config_path}。"
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            return None, f"模型配置文件读取失败：{exc}。"
        return raw.get("model") or {}, "ok"

    def _load_secret_cfg(self) -> tuple[dict[str, Any], str]:
        if not self.secrets_path.exists():
            return {}, f"本地密钥配置文件不存在：{self.secrets_path}。"
        try:
            raw = yaml.safe_load(self.secrets_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            return {}, f"本地密钥配置文件读取失败：{exc}。"
        if not isinstance(raw, dict):
            return {}, f"本地密钥配置文件格式无效：{self.secrets_path}。"
        return raw, "ok"

    def resolve_api_key(self, provider: ProviderConfig) -> tuple[str | None, str | None, str]:
        env_value = os.getenv(provider.api_key_env)
        if env_value:
            return env_value, f"环境变量 {provider.api_key_env}", "ok"

        secret_cfg, secret_reason = self._load_secret_cfg()
        api_keys = secret_cfg.get("api_keys") if isinstance(secret_cfg.get("api_keys"), dict) else {}
        env_key_value = api_keys.get(provider.api_key_env)
        if env_key_value:
            return str(env_key_value), f"本地密钥配置 api_keys.{provider.api_key_env}", "ok"

        providers = secret_cfg.get("providers") if isinstance(secret_cfg.get("providers"), dict) else {}
        provider_secret = providers.get(provider.name) if isinstance(providers.get(provider.name), dict) else {}
        provider_key_value = provider_secret.get("api_key")
        if provider_key_value:
            return str(provider_key_value), f"本地密钥配置 providers.{provider.name}.api_key", "ok"

        reason = (
            f"未配置环境变量 {provider.api_key_env}，且未在本地密钥配置 "
            f"{self.secrets_path} 中找到 api_keys.{provider.api_key_env} "
            f"或 providers.{provider.name}.api_key；{secret_reason}"
        )
        return None, None, reason

    def _routing_settings(self) -> tuple[bool, dict[str, str]]:
        model_cfg, _ = self._load_model_cfg()
        routing = (model_cfg or {}).get("routing") or {}
        routes = {
            str(key): str(value)
            for key, value in (routing.get("routes") or {}).items()
            if value
        }
        return bool(routing.get("enabled")), routes

    def _provider_from_config(
        self, provider_name: str, providers: dict[str, Any]
    ) -> tuple[ProviderConfig | None, str]:
        provider_cfg = providers.get(provider_name)
        if not provider_cfg:
            return None, f"模型供应商 {provider_name} 未在 providers 中定义。"
        endpoint = provider_cfg.get("endpoint")
        if endpoint not in {"responses", "chat_completions"}:
            return None, f"模型供应商 {provider_name} 的 endpoint 不支持：{endpoint}。"
        required = ["base_url", "api_key_env", "model"]
        missing = [key for key in required if not provider_cfg.get(key)]
        if missing:
            return None, f"模型供应商 {provider_name} 缺少字段：{', '.join(missing)}。"
        modalities = provider_cfg.get("input_modalities", ["text"])
        if not isinstance(modalities, list):
            modalities = ["text"]
        input_modalities = [str(item) for item in modalities if str(item).strip()] or ["text"]
        return (
            ProviderConfig(
                name=provider_name,
                base_url=str(provider_cfg["base_url"]).rstrip("/"),
                endpoint=endpoint,
                api_key_env=str(provider_cfg["api_key_env"]),
                model=str(provider_cfg["model"]),
                input_modalities=input_modalities,
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
        provider, reason, routed_from, route_reason = self.select_provider(multimodal_content)
        status = self.status(provider, routed_from=routed_from, route_reason=route_reason)
        if not status.enabled:
            return ModelCallResult(
                ok=False,
                content=None,
                raw_text=None,
                status=status,
                error=status.reason,
            )
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
                secrets_path=str(self.secrets_path),
            )
            return ModelCallResult(False, None, None, disabled, reason)

        unsupported = self._unsupported_modalities(provider, multimodal_content)
        if unsupported:
            status.enabled = False
            status.using_real_model = False
            status.reason = (
                f"当前模型 {provider.name} / {provider.model} 不支持 "
                f"{', '.join(unsupported)} 输入；已配置输入能力为 "
                f"{', '.join(provider.input_modalities)}。请切换支持多模态的模型路由。"
            )
            return ModelCallResult(
                ok=False,
                content=None,
                raw_text=None,
                status=status,
                error=status.reason,
            )

        api_key, _, key_reason = self.resolve_api_key(provider)
        if not api_key:
            status.enabled = False
            status.using_real_model = False
            status.reason = key_reason
            return ModelCallResult(False, None, None, status, key_reason)
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
                status=self.status(provider, routed_from=routed_from, route_reason=route_reason),
            )
        except Exception as exc:
            failed_status = self.status(provider, routed_from=routed_from, route_reason=route_reason)
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

    def select_provider(
        self, multimodal_content: list[dict[str, Any]] | None = None
    ) -> tuple[ProviderConfig | None, str, str | None, str]:
        model_cfg, reason = self._load_model_cfg()
        if model_cfg is None:
            return None, reason, None, "config_error"
        providers = model_cfg.get("providers") or {}
        active_name = str(model_cfg.get("active_provider") or "")
        active_provider, reason = self._provider_from_config(active_name, providers)
        if not active_provider:
            return None, reason, None, "active_provider_error"

        required = self._required_modalities(multimodal_content)
        if self._provider_supports(active_provider, required):
            return active_provider, "ok", None, "active_provider"

        routing = model_cfg.get("routing") or {}
        routes = routing.get("routes") or {}
        if routing.get("enabled"):
            route_names = self._route_candidates(required, routes)
            for route_name in route_names:
                provider, provider_reason = self._provider_from_config(route_name, providers)
                if provider and self._provider_supports(provider, required):
                    return provider, "ok", active_provider.name, f"routed_by_{'+'.join(sorted(required))}"
                if not provider:
                    reason = provider_reason

        for provider_name in providers:
            provider, _ = self._provider_from_config(str(provider_name), providers)
            if provider and self._provider_supports(provider, required):
                return provider, "ok", active_provider.name, "auto_matched_modalities"
        return active_provider, reason, None, "active_provider_unsupported"

    def _required_modalities(self, multimodal_content: list[dict[str, Any]] | None) -> set[str]:
        required = {"text"}
        for item in multimodal_content or []:
            if item.get("type") == "input_image":
                required.add("image")
            elif item.get("type") == "input_file":
                required.add("pdf")
        return required

    def _route_candidates(self, required: set[str], routes: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        if "pdf" in required:
            keys.extend(["pdf", "document", "multimodal"])
        if "image" in required:
            keys.extend(["image", "vision", "multimodal"])
        keys.append("text")
        candidates: list[str] = []
        for key in keys:
            value = routes.get(key)
            if value and value not in candidates:
                candidates.append(str(value))
        return candidates

    def _provider_supports(self, provider: ProviderConfig, required: set[str]) -> bool:
        return required <= set(provider.input_modalities or ["text"])

    def _unsupported_modalities(
        self,
        provider: ProviderConfig,
        multimodal_content: list[dict[str, Any]] | None,
    ) -> list[str]:
        required = self._required_modalities(multimodal_content)
        return sorted(required - set(provider.input_modalities or ["text"]))

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
5. reasoning_steps 只输出可供业务复盘的观察依据和判断步骤，不输出隐藏思维链。
输出必须是 JSON（结构化数据）。
"""


TASK_ATTRIBUTION_AGENT_INSTRUCTIONS = """
你是车险任务归属 Agent（智能体）。只消费 Evidence（证据）、候选任务和会话记忆，
通过实体关系推理判断消息归属。VIN（车架号）是车辆不可变唯一标识，禁止覆盖已有任务
VIN。多 VIN（车架号）批量报价必须拆成多个独立单任务，不创建父子任务。禁止重新做
OCR（光学字符识别）、禁止生成不存在的任务 ID（标识）。reasoning_steps 只输出可供业务复盘的
观察依据、候选比较和风险判断，不输出隐藏思维链。输出必须是 JSON（结构化数据）。
"""
