# Qwen / DeepSeek 模型适配

基线：`ac5af6ee4c9a06b4050c5a912439f343ade2c35c`。接口文档核查日期：2026-09-20。
本次保留 OpenCouch 的状态、记忆、练习、安全分流和 Agents SDK，不重写 agent runtime。

## 实现范围与完成边界

| 链路 | 实现 | 验证边界 |
| --- | --- | --- |
| Qwen / DeepSeek 文字 | Chat Completions、流式文字、JSON 模式、本地 Pydantic 校验、一次格式修复 | 真实 SDK + HTTP 模拟测试；尚未调用付费模型 |
| Agents SDK 执行 | 每次运行独立注入模型；triage 用控制模型，专用 agent 用所选回复档位 | 结构化输出、流式输出、工具调用与工具结果回传已做模拟测试 |
| 向量记忆 | Qwen `text-embedding-v4`、显式维度、独立开关；无痕模式保持无远程 embedding | 配置、HTTP 协议、无痕边界测试；不包含历史向量迁移 |
| Qwen 实时语音 | Qwen3.5-Omni Flash / Plus 的 WebRTC 原生语音对话、服务端 `txt` 数据通道、配置确认、转录与工具事件转换、打断相关处理 | 实验性、默认关闭；浏览器音频和云端实机验收未完成 |
| OpenAI | 原配置继续兼容，不要求删除 SDK | 回归测试；未进行 OpenAI 付费调用 |

“接入实时语音”在此指持续收音、RTP 音频输出及可打断的双向会话，不是 ASR → 通用 LLM → TTS 的串联链路。
其自然打断质量、同时说话时的效果、首音频延迟和弱网表现不能用模拟测试证明。
更换 `RESPONSE_FAST_LLM_MODEL` 不会更换 Qwen 原生语音模型的“对话大脑”：实时模型、文字回复和后台控制任务分别配置。

## 改动地图

- `llm/providers.py` / `llm/factory.py` / `config.py`：提供商注册与连接设置；支持 `openai`、`qwen`、`deepseek`、`openai_compatible`。保留旧 Settings 字段名以缩小兼容性改动，新配置优先使用供应商中立的环境变量。
- `llm/compatible_client.py`：替代仅有 Responses API 的调用假设。Qwen 显式关闭 thinking，DeepSeek 显式设置 `thinking.type=disabled`；限制输出长度和网络重试。JSON 格式错误不会被适配器转成一个“安全”的默认对象。
- `llm/sdk_models.py` / `agent/runtime/openai_text_runtime.py`：注入实际配置的模型，覆盖普通与流式 Runner 调用；不修改全局 SDK 默认客户端。第三方请求使用 JSON object 模式，不发送工具 `strict=true`；仍由 Runner / Pydantic 执行本地输出与参数校验。第三方运行关闭 OpenAI SDK tracing，避免向 OpenAI 发送第三方会话轨迹。
- `api/dependencies.py` / `opencouch_tui/runtime.py`：显式第三方配置缺失或非法时报错，不静默变成 deterministic demo。
- `agent/memory/providers/embeddings.py`：embedding 独立选型；Qwen v4 发送 `dimensions`。既有运行期检索降级策略保留，不代表所有错误都会中断请求。
- `agent/voice/qwen_realtime.py` / `agent/voice/realtime.py`：独立语音 provider；复用应用 instructions 和 tools，转换 Qwen 协议；提供短期、限量、一次性 SDP 交换票据。
- `api/models.py` / `api/routes/voice.py`：会话响应带 `provider`；新增公开能力配置和 Qwen SDP 交换入口。长期 DashScope key 不传给浏览器。
- `apps/web/src/lib/{api,realtime-voice-session,qwen-realtime-protocol}.ts`：服务端 `txt` 通道双向通信、ICE 收集等待、配置确认前关闭麦克风与播放、转录事件归一化、工具后续回复的本地关联、配置失败/转录失败关闭语音、20 分钟客户端会话上限。
- `apps/web/src/app/voice/page.tsx`：从后端读取当前提供商的音色列表，避免将 OpenAI 音色传给 Qwen。

## 配置和运行

完整样例在 [`examples/providers.env.example`](examples/providers.env.example)。将其中配置填到本地 `apps/backend/.env`，不要提交真实密钥。

最低成本配置应通过实测决定；下面只是把控制任务、回复档位和实时语音分开选型的起点，不是已经完成的价格/效果比较：

```dotenv
DASHSCOPE_API_KEY=填入本地密钥
DEEPSEEK_API_KEY=填入本地密钥
LLM_PROVIDER=qwen
LLM_MODEL=qwen-flash
RESPONSE_FAST_LLM_PROVIDER=deepseek
RESPONSE_FAST_LLM_MODEL=deepseek-flash
RESPONSE_QUALITY_LLM_PROVIDER=qwen
RESPONSE_QUALITY_LLM_MODEL=qwen-plus
EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
```

文字可以先单独运行。沿用仓库的 Postgres 配置后，从项目根目录执行：

```bash
docker compose -f compose.yml up -d postgres --wait
./scripts/text_tui.sh --mode hybrid --memory-mode persistent --user-id provider-test
```

生产 API 与 TUI 均使用同一组配置。自定义兼容服务使用 `LLM_PROVIDER=openai_compatible`、`LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`；它必须支持本文所用的聊天、流式、JSON 和工具能力，不承诺所有声称兼容的服务都可直接运行。

### 启用 Qwen 实时语音

```dotenv
OPENCOUCH_VOICE_PROVIDER=qwen
OPENCOUCH_ENABLE_EXPERIMENTAL_QWEN_VOICE=true
QWEN_REALTIME_MODEL=qwen3.5-omni-flash-realtime
QWEN_REALTIME_VOICE=Tina
QWEN_REALTIME_VAD=semantic_vad
QWEN_REALTIME_URL=https://你的业务空间ID.cn-beijing.maas.aliyuncs.com/api/v1/webrtc/realtime
```

地址不带 `?model=`，由服务端依据选定模型添加。新加坡使用同一结构的 `.ap-southeast-1.maas.aliyuncs.com` 域名。模型权限、业务空间、地域、Key 必须相符；文字 `QWEN_BASE_URL` 与语音 `QWEN_REALTIME_URL` 是不同接口，不能互换。示例中原有北京 DashScope 兼容域名可按实际账号改成业务空间域名。

正常 Web 页面继续用既有 `/voice` 页面。浏览器仅得到本应用的一次性连接票据；调用本应用 `/api/voice/realtime/qwen/sdp` 换取 SDP。它不是 DashScope 临时 API Key，不能用于调用其他模型。有效期 120 秒，最多 128 个待连接票据，失败或重用需要重新发起会话。票据与进程绑定，沿用项目单 worker 部署约束。

本次仅接入 Qwen3.5-Omni 的 Flash / Plus 实时型号。不接受老的 Qwen3-Omni / Turbo 或任意快照名来假装具备同样的工具契约；WebSocket、AOQ 和第三方音频网关不在本 PR 范围内。

## 安全、隐私与成本边界

本项目仍是情绪支持和自助练习工具，而非专业治疗或危机服务。没有做临床有效性评估。

模型替换不等于安全行为等价。保留既有风险分流、工具执行与审计；实时链路仍沿用并行安全检查，不保证每一段音频都在分类完成后才播放。新增配置确认门控不是内容审核。

Qwen 的 WebRTC 工具后续回复没有使用 OpenAI 的 response metadata，本地关联仅在未被打断的同一轮内成立。重叠发言、工具运行时插话、断线重连需要在真实浏览器专项验收，不能只根据正常对话通过就开放给他人。

不配置搜索时，适配器显式报告不支持，绝不把普通模型输出冒充已搜索结果。DeepSeek 原生搜索没有在此实现。`QWEN_ENABLE_SEARCH=true` 可为支持的 Qwen **文字**模型启用原生搜索，可能产生额外费用；Qwen **实时**会话保留 function tools，不同时启用原生搜索。危机资源查找是否仍能得到可靠来源必须单独检查。

仅更换聊天模型不能移除 embedding 费用；使用 `EMBEDDING_PROVIDER=none` 可关闭远程向量化，但会降级记忆检索。更换 embedding 模型时，旧向量不可直接比较；本 PR 不自动重算或删除历史记忆。不要在未备份/验证的情况下修改已有记忆库。

20 分钟语音限制是浏览器主动结束会话的保护，不是供应商侧账单硬限额。Qwen 历史音频仍可能在后续轮次计入输入，不能用“每分钟音频价格”简单估算全部费用。需要在供应商控制台设置预算/用量告警并测量真实会话。未在本任务中使用真实 Key 或产生模型推理费用。

对外部署前需要鉴权、用户与 thread 归属校验、请求体限制、每用户限流与预算。SDP 票据不提供账号鉴权，本次没有把原有本地 pre-beta 服务改造为安全的公共多租户产品。不要把模型密钥放进 `NEXT_PUBLIC_*`；不要记录原始 SDP、票据、完整用户对话或带密钥的上游错误体。`VOICE_SAFETY_SIGNING_SECRET` 应独立设置，避免依赖某个模型 Key 或进程重启时临时生成的值。

## 测试与实机验收

```bash
# 后端：不选择 live tests。未配置的 Postgres 集成测试会跳过。
cd apps/backend
uv sync --frozen --group dev
uv run pytest tests/unit tests/integration -m 'not live_api' -q
# 前端：Node 22+；先完成该 workspace 的依赖安装。
cd ../web
node --experimental-strip-types --test tests/*.test.mjs
node node_modules/typescript/bin/tsc --noEmit
```

模型 SDK 测试使用 HTTPX MockTransport，不仅测试自写 mock client；浏览器测试执行真实连接函数，模拟 HTTP、RTC 通道、转录、安全检查及回合保存。它们不能证明云服务端接受所有参数，也不能测量麦克风/扬声器效果。

合并为可用版本前的实机验收：用新的测试用户和测试数据，验证中文普通交流；核对真实请求模型与后台用量；检查 JSON 输出、记忆写入/读取/删除、练习的开始/中断/继续；验证风险表达处理和资源来源；在语音播放中打断、在工具执行中插话、拒绝麦克风权限、断网、转录失败、会话结束与重新加载。对未实现或未验证项保留明确失败，而不是显示连接成功。

## 官方协议依据

- [Qwen 文字兼容 API](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
- [DeepSeek Chat Completions：模型名、thinking、JSON 模式与工具参数](https://api-docs.deepseek.com/api/create-chat-completion/)
- [Qwen Realtime：WebRTC 信令、服务端 txt 通道、语音事件、地域与计费方式](https://help.aliyun.com/zh/model-studio/realtime)
- [Qwen 客户端事件](https://help.aliyun.com/zh/model-studio/client-events)
- [Qwen 工具调用与实时模型限制](https://help.aliyun.com/zh/model-studio/qwen-function-calling)
- [向量化](https://help.aliyun.com/zh/model-studio/embedding)
- [Agents SDK 模型适配](https://openai.github.io/openai-agents-python/models/)

### 本次离线验证记录

- 后端 `tests/unit` + `tests/integration`：**1,562 passed, 102 skipped**。未启用需要另行配置的 Postgres 集成环境，也未执行 `tests/live`。
- 前端全部 Node 测试：**80 passed**，包含真实连接函数的模拟 WebRTC/HTTP 传输测试。
- 前端 TypeScript `--noEmit` 与改动文件 ESLint：通过。
- 新增 4 个 Python 实现模块的 mypy（`--explicit-package-bases --follow-imports=silent`）：通过；这不是整个后端的完整类型检查声明。
- 依赖锁文件检查与 `git diff --check`：通过；未批量升级依赖。
- 真实云端 API、真实浏览器音频、区域可用性、语音工具/打断可靠性、实际费用与咨询安全效果：**未验证**。
