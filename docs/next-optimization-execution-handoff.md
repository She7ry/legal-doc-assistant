# Legal Document Assistant 下一轮优化执行交接

更新时间：2026-07-11  
工作目录：`E:\project\legal_doc_assistant`  
当前分支：`codex/deepseek-chat-protocols`

## 1. 任务目标

在上一轮正确性和可靠性修复已经通过全量验证的基础上，实施第一批有直接证据、低风险的性能优化；其余方向先量化，未达到触发条件时不要实现。

按以下顺序执行：

1. 前端路由按页拆包。
2. 为实际使用的 Qdrant 过滤字段建立 payload index。
3. 复测并记录收益；只有发现新的真实瓶颈才继续后续候选项。

不要新增依赖、前端测试框架、缓存层、清单数据库、队列框架或通用抽象。

## 2. 必须保护的工作树

当前工作树包含任务开始前已有改动和上一轮尚未提交的修复。禁止使用 `git reset --hard`、`git checkout --`、`git restore` 或覆盖式清理，也不要自行 commit、push 或创建 PR。

开始前执行并记录：

```powershell
git -c safe.directory=E:/project/legal_doc_assistant status -sb
git -c safe.directory=E:/project/legal_doc_assistant diff --stat
```

修改重叠文件前必须先查看现有 diff，使用 `apply_patch` 做局部编辑。

上一轮验证基线：

- Pytest：156 passed。
- Ruff：通过。
- 前端构建：通过；主包约 510.26 kB，触发 500 kB 警告。
- Coverage：87%。
- 全新 Python 3.11 环境安装、`pip check`、全量测试：通过。
- 文档 205 chunks 分页和 citation 定位的 Playwright 冒烟：通过。

## 3. 阶段 A：前端路由按页拆包

证据：`frontend/src/app/router.ts` 静态导入全部 6 个页面，Vite 构建主包约 510.26 kB。

实施要求：

1. 使用 Vue Router 原生动态 import 拆分页面组件，不新增库。
2. 优先保留首页 `WorkspacePage` 同步加载，其他页面惰性加载；如果实测主包仍超限，再评估首页也惰性加载。
3. 不改变路由名称、URL、页面行为或导航结构。
4. 运行 `npm.cmd run build`，记录优化前后主包大小。
5. 用现有 Playwright CLI 冒烟访问 `/`、`/agent`、`/matters`、`/documents`、`/review`、`/settings`；不引入前端测试框架。

验收：构建通过，各路由可加载且控制台无 error；主包大小有明确下降。若没有下降，撤销本阶段新增改动并报告原因。

## 4. 阶段 B：Qdrant payload index

证据：检索代码大量使用 payload filter，但仓库中没有 `create_payload_index`。当前实际过滤字段为：

- 文档集合：`active`、`document_key`、`file_id`、`document_version`、`chunk_id`。
- 记忆集合：`tenant_id`、`user_id`、`status`、`visibility`。

实施要求：

1. 先检查当前 `qdrant-client` 版本支持的 schema 类型和现有 collection 初始化路径。
2. 在共享 collection 初始化/校验位置创建索引，不在每次 search、scroll 或 count 时重复创建。
3. 字符串字段使用 keyword、整数使用 integer、`active` 使用 bool；不要给未参与过滤的字段建索引。
4. 索引是性能能力，不得让已有 collection 因索引创建失败不可用；失败要 warning 并继续，不能静默。
5. 兼容现有 `active` 缺失即视为 active 的旧数据语义。
6. fake client 回归测试至少验证：字段和 schema 正确、每个实例/集合只初始化一次、失败会 warning 且不阻断检索。

定向验证：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_vector_store_ingestion.py tests\test_memory_service.py
.\.venv\Scripts\python.exe -m ruff check src api tests scripts
```

如果本地没有真实 Qdrant 数据规模，不要宣称延迟收益；只报告索引已正确配置，并把 p95 数据验证列为部署环境检查项。

## 5. 暂不实施，只有指标触发才做

### 5.1 文档目录全量扫描

当前 `list_documents` 会读取并在 Python 中汇总全部 chunk。只有在代表性数据规模下列表 p95 超过 200 ms，或单租户达到约 10,000 chunks 后，再考虑 Qdrant facet/grouping 或最小化的文档摘要存储。不要现在增加双写 manifest。

### 5.2 Matter 列表 bounded N+1

当前列表会按 Matter 读取 findings。只有列表页 p95 超过 200 ms 或常用页大小达到 50 条后，改成一个批量 findings 查询；不要引入 ORM/DataLoader。

### 5.3 后台任务多机租约

当前 executor 明确是单进程方案。只有部署改成多进程/多副本，或真实出现进程崩溃后的任务接管需求时，才增加 lease/heartbeat；不要提前引入队列框架。

### 5.4 RAG reranker 与 Agent 调优

先用真实凭据运行现有 eval，记录召回率、排序准确率、citation 支持率、Matter 结构化成功率。只有“召回覆盖足够但前排排序差”时才增加 reranker；没有基线时不要靠主观调整 prompt/model。

### 5.5 LangChain Community 迁移

干净安装已出现 `langchain-community` sunset 警告，但当前同时依赖 document loaders、DashScope embeddings 和 HuggingFace embeddings。先逐项确认官方替代包和 API 等价性；只有三条路径都能通过现有测试和 clean install 时才移除 umbrella dependency，不能只替换一个 import 后留下半迁移状态。

## 6. 完整验证与交付

完成阶段 A/B 后执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src api tests scripts
git -c safe.directory=E:/project/legal_doc_assistant diff --check
Push-Location frontend
npm.cmd run build
Pop-Location
```

最终报告必须列出：修改文件、前后主包大小、索引字段与 schema、定向/全量测试结果、未执行项及其触发条件。真实 Qdrant 性能测试或真实 RAG eval 若因环境/凭据缺失而跳过，要明确写出。
