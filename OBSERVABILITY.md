# Agent 可观测性与评估

本项目以 LangSmith 为 Agent 可观测性的主平台。本地 SQLite 仅负责后台任务状态、事件和恢复，不保存第二套 trace 或指标。

## 目标与指标

当前不评价用户满意度或文风，只关注三类结果：

| 目标 | 指标 | 来源 |
| --- | --- | --- |
| 任务完成率 | `task_completion`、`requirement_coverage`、`autonomous_completion_rate` | LLM-as-Judge + Agent 最终状态 |
| 准确性 | `factual_accuracy`、`evidence_faithfulness`、`citation_correctness` | LLM-as-Judge |
| 效率 | 根运行耗时、P95、Token/成本、平均工具调用数、重试数、错误率 | LangSmith trace |

Agent 根运行名为 `legal_agent_task`。LangChain/LangGraph 的模型、工具和子链运行作为其子 span，因而可在一条 trace 中定位慢调用、重复调用、错误和高 Token 消耗。

## 数据边界

- trace 输入不包含服务对象或进度回调；用户 ID 和会话 ID 只上传 SHA-256 截断哈希。
- trace 根输出保留报告、步骤摘要和引用，省略原始工具返回值。
- LangSmith 客户端在上传前处理常见密钥、邮箱、手机号和身份证号。
- 模型子调用仍可能包含法律文档片段。只有在组织已批准 LangSmith 工作区、数据驻留和保留策略时才能开启生产追踪；否则保持 `LANGSMITH_TRACING=false`。

## 开启运行时追踪

在 `.env` 中配置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=legal-doc-assistant-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING_SAMPLING_RATE=1.0
```

开发和校准期建议采样率为 1；生产稳定后可根据数据量下调，但错误任务应通过 LangSmith 规则或单独项目保留足够样本。

## 运行离线 LLM-as-Judge

评估种子位于 `data/eval/agent_eval_dataset.json`。脚本会以确定性 Example ID 同步 LangSmith Dataset，运行真实 Agent，再将五个评分键写入同一个 Experiment：

```powershell
python scripts/run_agent_eval.py --max-concurrency 2
```

首次只同步数据集：

```powershell
python scripts/run_agent_eval.py --sync-only
```

带质量和效率门槛运行：

```powershell
python scripts/run_agent_eval.py `
  --max-concurrency 2 `
  --min-score task_completion=0.8 `
  --min-score requirement_coverage=0.8 `
  --min-score factual_accuracy=0.85 `
  --min-score evidence_faithfulness=0.85 `
  --min-score citation_correctness=0.85 `
  --max-average-latency-seconds 300
```

Judge 一次结构化模型调用同时生成五项分数，避免为每项指标重复付费。当前复用项目配置的 Chat 模型；建立人工标注集并完成 Judge 校准后，再决定是否拆分独立 Judge 模型。

## CI 与手动评估

普通 push 和 pull request 只运行确定性的后端测试、Ruff 和前端构建。需要外部模型与 LangSmith 的评估仅通过 GitHub Actions 的 `workflow_dispatch` 手动运行；手动运行缺少密钥时直接失败，不静默跳过。

项目当前处于校准期，没有稳定的回归基线。工作流中的分数阈值只用于发现明显退化，不作为发布门槛。建立首个稳定版本后，再保存基线并启用回归比较；积累数轮实验后，根据分布确定 P95 延迟和 Token 预算。

生产在线 Judge 不在首个闭环内。离线 Judge 与人工样本达到可接受一致性后，再在 LangSmith 中对生产成功任务分层抽样，对错误任务全量评估，避免未经校准的 Judge 直接触发告警。
