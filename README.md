# 车险出单双 Agent 本地验证框架

这是一个本地可运行的双 Agent（智能体）验证框架，用于验证车险询报价/投保场景中的两类核心能力：

- `Material Understanding Agent（材料理解智能体）`：输入理解、单证分类、槽位抽取、Evidence（证据）生成。
- `Task Attribution Agent（任务归属智能体）`：基于 Evidence（证据）、候选任务和实体关系判断消息归属。

当前版本不接入真实保司报价、投保、支付 API（应用程序编程接口）。验证台支持真实大模型运行时；缺少模型密钥时会明确提示“模型不可用”，本地 fallback（兜底逻辑）仅用于离线单元测试和结构验证。

## 目录结构

```text
app/
  agents/                 # 两个独立 Agent（智能体）
  services/               # SQLite（本地数据库）存储、策略护栏
  knowledge.py            # 单证类型、槽位矩阵、关系类型
  schemas.py              # 公共 Schema（结构约束）
  main.py                 # FastAPI（接口服务）入口
scripts/
  seed_demo_data.py        # 写入本地模拟任务
  replay_cases.py          # 回放材料理解和任务归属样例
tests/
  test_agents.py           # 核心单元测试
config/
  model_providers.yaml     # 多供应商模型配置
```

## 安装与运行

建议使用 Python 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_demo_data.py
uvicorn app.main:app --reload
```

## 模型供应商配置

模型供应商通过 `config/model_providers.yaml` 切换：

```yaml
model:
  active_provider: openrouter_text
  routing:
    enabled: true
    routes:
      text: openrouter_text
      image: openrouter_vision
      vision: openrouter_vision
      pdf: apiopencc
  providers:
    apiopencc:
      base_url: "https://apiopencc.com/v1"
      endpoint: "responses"
      api_key_env: "APIOPENCC_API_KEY"
      model: "gpt-4.1-mini"
      input_modalities: ["text", "image", "pdf"]
    openrouter_text:
      base_url: "https://openrouter.ai/api/v1"
      endpoint: "chat_completions"
      api_key_env: "OPENROUTER_API_KEY"
      model: "minimax/minimax-m2.7"
      input_modalities: ["text"]
    openrouter_vision:
      base_url: "https://openrouter.ai/api/v1"
      endpoint: "chat_completions"
      api_key_env: "OPENROUTER_API_KEY"
      model: "qwen/qwen3-vl-32b-instruct"
      input_modalities: ["text", "image"]
```

密钥读取优先级为：系统环境变量 > 本地密钥配置文件。优先推荐系统环境变量：

```bash
export APIOPENCC_API_KEY="你的 apiopencc 密钥"
export OPENROUTER_API_KEY="你的 openrouter 密钥"
```

如果桌面应用启动时读不到系统环境变量，可以改用本地私密配置文件 `config/model_secrets.local.yaml`。该文件已被 `.gitignore` 排除，不会提交到 git：

```yaml
api_keys:
  APIOPENCC_API_KEY: "你的 apiopencc 密钥"
  OPENROUTER_API_KEY: "你的 openrouter 密钥"
```

也可以按供应商单独配置：

```yaml
providers:
  openrouter_text:
    api_key: "你的 openrouter 文本模型密钥"
  openrouter_vision:
    api_key: "你的 openrouter 视觉模型密钥"
```

`active_provider` 是默认文本模型。`routing` 会根据输入动态路由：纯文本走 `openrouter_text`，图片走 `openrouter_vision`，PDF（便携式文档格式）走 `apiopencc`。`apiopencc` 使用 Responses API（响应接口），`openrouter_*` 使用 Chat Completions API（聊天补全接口）。`GET /runtime` 会展示默认供应商、模型名、endpoint（接口端点）、输入能力和真实模型是否可用。

启动后访问：

- `GET /health`
- `GET /runtime`
- `POST /attachments/upload`
- `GET /attachments/{attachment_id}/download`
- `POST /events/ingest`
- `POST /agents/material-understanding/run`
- `POST /agents/task-attribution/run`
- `POST /evals/material/replay`
- `POST /evals/attribution/replay`

## 本地回放

```bash
python scripts/replay_cases.py
```

## 测试

```bash
python -m unittest discover -s tests
```

## 当前边界

- 材料理解 Agent（智能体）不创建任务、不合并任务、不覆盖任务事实。
- 任务归属 Agent（智能体）不做 OCR（光学字符识别）或字段抽取。
- 图片/PDF（便携式文档格式）支持真实附件上传，并可在真实模型可用时进入多模态解析链路；缺少模型密钥时仅保留本地分类和规则抽取 fallback（兜底逻辑）。
- Excel（电子表格）和 Word（文字文档）当前支持上传存储与材料引用，正文解析仍待接入专门解析器。
- 高风险归属由 Policy Engine（策略引擎）拦截。
- 多 VIN（车架号）批量报价会拆为多个独立单任务，不创建父子任务。
- VIN（车架号）不可更新；新 VIN 只能创建新任务、进入批量拆分或触发确认。
- 真实 OCR（光学字符识别）可在当前 Schema（结构约束）之上继续增强。
