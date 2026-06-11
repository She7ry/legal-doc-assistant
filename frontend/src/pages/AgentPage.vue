<template>
  <div class="page-stack agent-page">
    <section class="summary-strip">
      <div class="metric">
        <span>任务状态</span>
        <strong>{{ task ? statusLabel(task.status) : "未运行" }}</strong>
      </div>
      <div class="metric">
        <span>执行进度</span>
        <strong>{{ task?.progress ?? 0 }}%</strong>
      </div>
      <div class="metric">
        <span>风险发现</span>
        <strong>{{ result?.findings.length ?? 0 }}</strong>
      </div>
      <div class="metric">
        <span>Artifacts</span>
        <strong>{{ result?.artifacts.length ?? 0 }}</strong>
      </div>
      <div class="metric">
        <span>Gates</span>
        <strong>{{ confirmationGates.length }}</strong>
      </div>
    </section>

    <div class="agent-grid">
      <section class="tool-panel agent-run-panel">
        <div class="panel-heading">
          <div>
            <h2>Agent 任务</h2>
            <p>把法律文档问题拆成可追踪的审查步骤、证据和报告。</p>
          </div>
          <el-button :icon="Refresh" :disabled="loading" @click="resetForm">重置</el-button>
        </div>

        <el-form class="agent-form" label-position="top" @submit.prevent>
          <el-form-item label="任务目标">
            <el-input
              v-model="form.objective"
              type="textarea"
              :rows="4"
              maxlength="2000"
              show-word-limit
              resize="none"
              :disabled="loading"
              placeholder="例如：审查这份 SaaS MSA 的终止、付款和责任限制风险，并给出律师问题清单。"
            />
          </el-form-item>

          <el-form-item label="关注领域">
            <el-select
              v-model="form.focusAreas"
              multiple
              filterable
              allow-create
              default-first-option
              collapse-tags
              collapse-tags-tooltip
              :disabled="loading"
              placeholder="选择或输入条款类型"
            >
              <el-option
                v-for="area in focusAreaOptions"
                :key="area.value"
                :label="area.label"
                :value="area.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="Matter ID">
            <el-input
              v-model="form.matterId"
              maxlength="128"
              clearable
              :disabled="loading"
              placeholder="留空时自动创建新事项；填入已有 ID 可继续同一 Matter"
            />
          </el-form-item>

          <div class="agent-form-row">
            <el-form-item label="用户模式">
              <el-radio-group v-model="form.userRole" :disabled="loading">
                <el-radio-button
                  v-for="option in userRoleOptions"
                  :key="option.value"
                  :value="option.value"
                >
                  {{ option.label }}
                </el-radio-button>
              </el-radio-group>
            </el-form-item>
            <el-form-item label="最大步骤">
              <el-input-number
                v-model="form.maxSteps"
                :min="3"
                :max="10"
                :disabled="loading"
                controls-position="right"
              />
            </el-form-item>
          </div>

          <div class="panel-actions">
            <el-button
              type="primary"
              :icon="MagicStick"
              :loading="loading"
              :disabled="!form.objective.trim()"
              @click="runTask"
            >
              运行 Agent
            </el-button>
          </div>
        </el-form>
      </section>

      <section v-if="task" class="tool-panel agent-status-panel">
        <div class="panel-heading">
          <div>
            <h2>任务进度</h2>
            <p>{{ task.task_id }}</p>
          </div>
          <el-tag :type="taskStatusType(task.status)" effect="dark">
            {{ statusLabel(task.status) }}
          </el-tag>
        </div>

        <el-progress
          :percentage="Math.max(0, Math.min(task.progress, 100))"
          :status="progressStatus"
          :stroke-width="10"
        />
        <p class="job-stage">{{ stageLabel(task.stage) }}</p>

        <div v-if="currentMatterId" class="agent-matter-actions">
          <el-tag effect="plain">Matter: {{ currentMatterId }}</el-tag>
          <el-button :icon="Collection" size="small" @click="openMatter(currentMatterId)">
            打开事项
          </el-button>
        </div>

        <el-alert
          v-if="task.error"
          type="error"
          :title="task.error"
          :closable="false"
          show-icon
        />

        <el-alert
          v-if="clarificationQuestions.length"
          type="warning"
          title="需要补充信息后再运行"
          :closable="false"
          show-icon
        >
          <ul class="structured-list structured-list--compact agent-clarification-list">
            <li v-for="question in clarificationQuestions" :key="question">{{ question }}</li>
          </ul>
        </el-alert>

        <el-form
          v-if="task.status === 'needs_input'"
          class="agent-resume-form"
          label-position="top"
          @submit.prevent
        >
          <el-form-item label="补充信息">
            <el-input
              v-model="resumeForm.clarificationAnswers"
              type="textarea"
              :rows="3"
              maxlength="1200"
              show-word-limit
              resize="none"
              :disabled="loading"
              placeholder="例如：请审查付款和终止风险。我代表客户方，合同适用纽约州法。"
            />
          </el-form-item>
          <div class="panel-actions">
            <el-button
              type="primary"
              :icon="MagicStick"
              :loading="loading"
              :disabled="!resumeForm.clarificationAnswers.trim()"
              @click="resumeTask"
            >
              补充并继续
            </el-button>
          </div>
        </el-form>

        <div v-if="events.length" class="agent-event-list">
          <article v-for="event in events" :key="event.event_id" class="agent-event">
            <el-tag size="small" :type="eventType(event.event_type)" effect="plain">
              {{ eventLabel(event.event_type) }}
            </el-tag>
            <div>
              <strong>{{ event.message }}</strong>
              <span>{{ event.progress }}% · {{ formatDate(event.created_at) }}</span>
            </div>
          </article>
        </div>
      </section>

      <section v-if="result" class="tool-panel agent-report-panel">
        <div class="panel-heading">
          <div>
            <h2>最终报告</h2>
            <p>任务 {{ result.task_id }}</p>
          </div>
          <el-tag :type="result.human_review_required ? 'warning' : 'success'" effect="dark">
            {{ result.human_review_required ? "需要人工复核" : "证据闭环完成" }}
          </el-tag>
        </div>

        <div class="agent-report">{{ result.report }}</div>

        <div v-if="result.guard_warnings.length" class="answer-trust">
          <el-alert type="warning" title="报告守卫提示" :closable="false" show-icon>
            <ul class="structured-list structured-list--compact">
              <li v-for="warning in result.guard_warnings" :key="warning">{{ warning }}</li>
            </ul>
          </el-alert>
        </div>

        <EvidencePanel v-if="result.evidence" :evidence="result.evidence" />
        <CitationList v-if="result.citations.length" :citations="result.citations" />
      </section>

      <section v-if="result" class="tool-panel agent-trace-panel">
        <div class="panel-heading">
          <div>
            <h2>执行轨迹</h2>
            <p>计划、工具和每一步的证据状态。</p>
          </div>
        </div>

        <div class="agent-step-list">
          <article v-for="step in result.steps" :key="step.step_id" class="agent-step">
            <div class="agent-step__header">
              <div>
                <strong>{{ step.title }}</strong>
                <span>{{ toolLabel(step.tool) }}</span>
              </div>
              <el-tag :type="stepStatusType(step.status)" effect="plain">
                {{ statusLabel(step.status) }}
              </el-tag>
            </div>
            <p>{{ step.summary }}</p>
            <div v-if="step.citations.length" class="citation-refs">
              <el-tag
                v-for="citation in step.citations"
                :key="`${step.step_id}-${citation.source_id}`"
                size="small"
                effect="plain"
              >
                {{ citation.source_id }} {{ citation.file_name }}
              </el-tag>
            </div>
          </article>
        </div>
      </section>

      <aside v-if="result" class="side-stack">
        <section v-if="result.matter_profile" class="tool-panel agent-matter-panel">
          <div class="panel-heading">
            <div>
              <h2>Matter Profile</h2>
              <p>Task-level facts, scope, and open questions.</p>
            </div>
            <el-tag effect="plain">{{ result.matter_profile.confidence }}</el-tag>
          </div>

          <dl class="matter-profile-grid">
            <div>
              <dt>Document</dt>
              <dd>{{ result.matter_profile.document_type || "Unknown" }}</dd>
            </div>
            <div>
              <dt>Parties</dt>
              <dd>{{ formatList(result.matter_profile.parties) }}</dd>
            </div>
            <div>
              <dt>User side</dt>
              <dd>{{ result.matter_profile.user_side || "Unspecified" }}</dd>
            </div>
            <div>
              <dt>Governing law</dt>
              <dd>{{ result.matter_profile.governing_law || "Unspecified" }}</dd>
            </div>
            <div>
              <dt>Scope</dt>
              <dd>{{ formatList(result.matter_profile.review_scope) }}</dd>
            </div>
          </dl>

          <ul
            v-if="result.matter_profile.key_dates.length"
            class="structured-list structured-list--compact"
          >
            <li v-for="dateItem in result.matter_profile.key_dates" :key="dateItemKey(dateItem)">
              {{ dateItemLabel(dateItem) }}
            </li>
          </ul>
        </section>

        <section v-if="confirmationGates.length" class="tool-panel agent-gate-panel">
          <div class="panel-heading">
            <div>
              <h2>Confirmation Gates</h2>
              <p>Human decisions required before relying on the output.</p>
            </div>
            <el-tag type="warning" effect="plain">{{ confirmationGates.length }}</el-tag>
          </div>

          <div class="agent-gate-list">
            <article
              v-for="gate in confirmationGates"
              :key="gate.gate_id"
              class="agent-gate"
            >
              <div class="agent-gate__header">
                <strong>{{ gate.title }}</strong>
                <span>
                  <el-tag :type="gatePriorityType(gate.priority)" size="small" effect="dark">
                    {{ gate.priority }}
                  </el-tag>
                  <el-tag :type="gateStatusType(gate.status)" size="small" effect="plain">
                    {{ gate.status }}
                  </el-tag>
                </span>
              </div>
              <p>{{ gate.question }}</p>
              <small v-if="gate.reason">{{ gate.reason }}</small>
              <div v-if="gateRefs(gate).length" class="citation-refs">
                <el-tag
                  v-for="sourceId in gateRefs(gate)"
                  :key="`${gate.gate_id}-${sourceId}`"
                  size="small"
                  effect="plain"
                >
                  {{ sourceId }}
                </el-tag>
              </div>
            </article>
          </div>
        </section>

        <section v-if="result.artifacts.length" class="tool-panel agent-artifact-panel">
          <div class="panel-heading">
            <div>
              <h2>Artifacts</h2>
              <p>Reusable deliverables generated from the workflow.</p>
            </div>
          </div>

          <div class="agent-artifact-list">
            <article
              v-for="artifact in result.artifacts"
              :key="artifact.artifact_id"
              class="agent-artifact"
            >
              <div class="agent-artifact__header">
                <strong>{{ artifact.title }}</strong>
                <el-tag size="small" effect="plain">{{ artifact.items.length }}</el-tag>
              </div>
              <p>{{ artifact.summary }}</p>
              <ul v-if="artifact.items.length" class="artifact-item-list">
                <li
                  v-for="item in artifact.items.slice(0, 4)"
                  :key="artifactItemKey(artifact, item)"
                >
                  <strong>{{ artifactItemTitle(item) }}</strong>
                  <span>{{ artifactItemDetail(item) }}</span>
                </li>
              </ul>
              <el-empty
                v-else
                :image-size="64"
                description="No structured items yet"
              />
            </article>
          </div>
        </section>

        <section class="tool-panel agent-finding-panel">
          <div class="panel-heading">
            <div>
              <h2>发现项</h2>
              <p>按风险和证据归纳的审查结论。</p>
            </div>
          </div>

          <el-empty v-if="!result.findings.length" :image-size="96" description="暂无结构化发现" />
          <div v-else class="risk-card-list">
            <article v-for="finding in result.findings" :key="finding.finding_id" class="risk-card">
              <div class="agent-finding__title">
                <strong>{{ finding.category }}</strong>
                <el-tag :type="severityType(finding.severity)" size="small" effect="dark">
                  {{ finding.severity }}
                </el-tag>
              </div>
              <p>{{ finding.summary }}</p>
              <p v-if="finding.recommended_action" class="table-copy">
                建议：{{ finding.recommended_action }}
              </p>
              <div v-if="finding.citations.length" class="citation-refs">
                <el-tag v-for="sourceId in finding.citations" :key="sourceId" size="small">
                  {{ sourceId }}
                </el-tag>
              </div>
            </article>
          </div>
        </section>

        <section class="tool-panel agent-missing-panel">
          <div class="panel-heading">
            <div>
              <h2>缺失信息</h2>
              <p>继续推进前应确认的事实或文件。</p>
            </div>
          </div>
          <el-empty
            v-if="!result.missing_information.length"
            :image-size="96"
            description="暂无缺失信息"
          />
          <ul v-else class="structured-list">
            <li v-for="item in result.missing_information" :key="item">{{ item }}</li>
          </ul>
        </section>

        <section class="tool-panel agent-plan-panel">
          <div class="panel-heading">
            <div>
              <h2>计划</h2>
              <p>本次任务的拆解路径。</p>
            </div>
          </div>
          <div class="agent-plan-list">
            <article v-for="step in result.plan" :key="step.step_id" class="agent-plan-item">
              <strong>{{ step.title }}</strong>
              <span>{{ step.purpose }}</span>
              <code>{{ toolLabel(step.tool) }}</code>
            </article>
          </div>
        </section>
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { Collection, MagicStick, Refresh } from "@element-plus/icons-vue";

import { getAgentTask, resumeAgentTask, runAgentTask, streamAgentTaskEvents } from "../api/agent";
import { formatApiError } from "../api/http";
import type {
  AgentArtifact,
  AgentConfirmationGate,
  AgentTaskEvent,
  AgentTaskRecordResponse,
} from "../api/types";
import CitationList from "../components/CitationList.vue";
import EvidencePanel from "../components/EvidencePanel.vue";

const AGENT_CONVERSATION_STORAGE_KEY = "legal-doc-assistant.agentConversationId";
const router = useRouter();

const focusAreaOptions = [
  { label: "付款", value: "payment" },
  { label: "终止", value: "termination" },
  { label: "责任限制", value: "liability limitation" },
  { label: "保密", value: "confidentiality" },
  { label: "数据隐私", value: "data privacy" },
  { label: "适用法律", value: "governing law" },
  { label: "补偿", value: "indemnification" },
  { label: "转让", value: "assignment" },
  { label: "审计权", value: "audit rights" },
];

const userRoleOptions = [
  { label: "普通用户", value: "ordinary" },
  { label: "法律专业", value: "lawyer" },
];

const form = reactive({
  objective: "",
  focusAreas: ["payment", "termination", "liability limitation"],
  userRole: "ordinary" as "ordinary" | "lawyer",
  maxSteps: 6,
  matterId: "",
});
const resumeForm = reactive({
  clarificationAnswers: "",
});
const loading = ref(false);
const task = ref<AgentTaskRecordResponse | null>(null);
const conversationId = ref(readConversationId());

const result = computed(() => task.value?.result ?? null);
const confirmationGates = computed(() => {
  if (!result.value) {
    return [];
  }
  const directGates = result.value.confirmation_gates ?? [];
  if (directGates.length) {
    return directGates;
  }
  return result.value.matter_profile?.confirmation_gates ?? [];
});
const currentMatterId = computed(
  () => task.value?.matter_id || result.value?.matter_profile?.matter_id || form.matterId,
);
const events = computed(() => task.value?.events ?? []);
const clarificationQuestions = computed(() => {
  if (task.value?.status !== "needs_input") {
    return [];
  }
  const needsInputEvent = [...events.value]
    .reverse()
    .find((event) => event.event_type === "needs_input");
  const questions = needsInputEvent?.payload?.questions;
  if (!Array.isArray(questions)) {
    return [];
  }
  return questions.map((question) => String(question)).filter(Boolean).slice(0, 3);
});
const progressStatus = computed(() => {
  if (!task.value) {
    return undefined;
  }
  if (task.value.status === "failed") {
    return "exception" as const;
  }
  if (task.value.status === "needs_input") {
    return "warning" as const;
  }
  if (task.value.status === "succeeded") {
    return "success" as const;
  }
  return undefined;
});

async function runTask() {
  if (!form.objective.trim() || loading.value) {
    return;
  }

  loading.value = true;
  try {
    task.value = await runAgentTask({
      objective: form.objective.trim(),
      focus_areas: form.focusAreas,
      user_role: form.userRole,
      max_steps: form.maxSteps,
      conversation_id: conversationId.value,
      matter_id: cleanMatterId(form.matterId),
    });
    form.matterId = task.value.matter_id || form.matterId;
    await streamCurrentTaskEvents();
  } catch (error) {
    await refreshCurrentTask();
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

async function resumeTask() {
  if (!task.value || task.value.status !== "needs_input" || loading.value) {
    return;
  }
  const answers = splitClarificationAnswers(resumeForm.clarificationAnswers);
  if (!answers.length) {
    return;
  }

  loading.value = true;
  const taskId = task.value.task_id;
  try {
    task.value = await resumeAgentTask(taskId, {
      clarification_answers: answers,
      focus_areas: form.focusAreas,
      user_role: form.userRole,
      max_steps: form.maxSteps,
      conversation_id: conversationId.value,
      matter_id: cleanMatterId(form.matterId || task.value.matter_id || ""),
    });
    form.matterId = task.value.matter_id || form.matterId;
    resumeForm.clarificationAnswers = "";
    if (task.value.status !== "needs_input") {
      await streamCurrentTaskEvents(latestEventId(task.value.events));
    }
    task.value = await getAgentTask(taskId);
  } catch (error) {
    await refreshCurrentTask();
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

async function streamCurrentTaskEvents(afterEventId = 0) {
  if (!task.value) {
    return;
  }
  const taskId = task.value.task_id;
  await streamAgentTaskEvents(
    taskId,
    {
      onEvent(event) {
        mergeEvent(event);
        applyEventProgress(event);
      },
      onError(message) {
        ElMessage.warning(message);
      },
    },
    afterEventId,
  );
  task.value = await getAgentTask(taskId);
  form.matterId = task.value.matter_id || form.matterId;
}

async function refreshCurrentTask() {
  if (!task.value?.task_id) {
    return;
  }
  try {
    task.value = await getAgentTask(task.value.task_id);
    form.matterId = task.value.matter_id || form.matterId;
  } catch {
    // Keep the existing local task state if status refresh also fails.
  }
}

function mergeEvent(event: AgentTaskEvent) {
  if (!task.value) {
    return;
  }
  const existing = task.value.events.findIndex((item) => item.event_id === event.event_id);
  if (existing >= 0) {
    task.value.events[existing] = event;
  } else {
    task.value.events.push(event);
  }
}

function applyEventProgress(event: AgentTaskEvent) {
  if (!task.value) {
    return;
  }
  task.value.stage = event.stage;
  task.value.progress = event.progress;
  if (event.event_type === "running") {
    task.value.status = "running";
  } else if (event.event_type === "queued" || event.event_type === "input_received") {
    task.value.status = "queued";
  } else if (event.event_type === "needs_input") {
    task.value.status = "needs_input";
  } else if (event.event_type === "failed") {
    task.value.status = "failed";
    task.value.error = event.message;
  } else if (event.event_type === "succeeded") {
    task.value.status = "succeeded";
    task.value.progress = 100;
  }
}

function resetForm() {
  task.value = null;
  form.objective = "";
  form.focusAreas = ["payment", "termination", "liability limitation"];
  form.userRole = "ordinary";
  form.maxSteps = 6;
  form.matterId = "";
  resumeForm.clarificationAnswers = "";
  conversationId.value = createConversationId();
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    queued: "排队中",
    running: "运行中",
    needs_input: "需补充信息",
    succeeded: "已完成",
    failed: "失败",
    completed: "已完成",
    needs_review: "需复核",
    needs_human_review: "需人工复核",
  };
  return labels[status] ?? status;
}

function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: "任务已排队",
    planning: "正在生成审查计划",
    reporting: "正在生成最终报告",
    needs_input: "等待补充任务上下文",
    completed: "任务已完成",
    failed: "任务失败",
  };
  return labels[stage] ?? stage;
}

function taskStatusType(status: string) {
  if (status === "succeeded") {
    return "success";
  }
  if (status === "queued" || status === "running" || status === "needs_input") {
    return "warning";
  }
  return "danger";
}

function stepStatusType(status: string) {
  if (status === "completed") {
    return "success";
  }
  if (status === "needs_review" || status === "needs_human_review") {
    return "warning";
  }
  return "danger";
}

function eventType(eventTypeValue: string) {
  if (eventTypeValue === "failed") {
    return "danger";
  }
  if (eventTypeValue === "succeeded") {
    return "success";
  }
  if (eventTypeValue === "input_received") {
    return "primary";
  }
  if (eventTypeValue === "needs_input") {
    return "warning";
  }
  if (eventTypeValue === "step_completed") {
    return "primary";
  }
  return "info";
}

function eventLabel(eventTypeValue: string) {
  const labels: Record<string, string> = {
    queued: "排队",
    running: "开始",
    input_received: "已补充",
    needs_input: "需补充",
    plan_created: "计划",
    step_started: "步骤开始",
    step_completed: "步骤完成",
    report_started: "报告",
    succeeded: "完成",
    failed: "失败",
  };
  return labels[eventTypeValue] ?? eventTypeValue;
}

function severityType(severity: string) {
  const normalized = severity.toLowerCase();
  if (normalized.includes("high") || normalized.includes("human")) {
    return "danger";
  }
  if (normalized.includes("medium")) {
    return "warning";
  }
  if (normalized.includes("low")) {
    return "success";
  }
  return "info";
}

function gatePriorityType(priority: string) {
  const normalized = priority.toLowerCase();
  if (normalized.includes("high") || normalized.includes("blocking")) {
    return "danger";
  }
  if (normalized.includes("normal")) {
    return "warning";
  }
  return "info";
}

function gateStatusType(status: string) {
  const normalized = status.toLowerCase();
  if (normalized.includes("approved") || normalized.includes("confirmed")) {
    return "success";
  }
  if (normalized.includes("waived")) {
    return "info";
  }
  if (normalized.includes("blocked")) {
    return "danger";
  }
  return "warning";
}

function gateRefs(gate: AgentConfirmationGate) {
  return [
    ...gate.citations,
    ...gate.related_finding_ids,
    ...gate.related_artifact_ids,
  ].filter(Boolean).slice(0, 8);
}

function toolLabel(tool: string) {
  const labels: Record<string, string> = {
    document_qa: "文档问答",
    review_clause: "条款审查",
    check_conflict: "冲突检查",
    synthesize_report: "报告生成",
  };
  return labels[tool] ?? tool;
}

function cleanMatterId(value: string) {
  const text = value.trim();
  return text || null;
}

function openMatter(matterId: string) {
  const normalized = matterId.trim();
  if (!normalized) {
    return;
  }
  void router.push({ path: "/matters", query: { matter_id: normalized } });
}

function formatList(values: string[]) {
  return values.length ? values.join(", ") : "Unspecified";
}

function dateItemKey(item: Record<string, unknown>) {
  return `${stringValue(item, "label")}-${stringValue(item, "value")}`;
}

function dateItemLabel(item: Record<string, unknown>) {
  const label = stringValue(item, "label") || "Date";
  const value = stringValue(item, "value") || "Unspecified";
  const description = stringValue(item, "description");
  return description ? `${label}: ${value} · ${description}` : `${label}: ${value}`;
}

function artifactItemKey(artifact: AgentArtifact, item: Record<string, unknown>) {
  return `${artifact.artifact_id}-${stringValue(item, "item_id") || artifactItemTitle(item)}`;
}

function artifactItemTitle(item: Record<string, unknown>) {
  return (
    stringValue(item, "category") ||
    stringValue(item, "question") ||
    stringValue(item, "issue") ||
    stringValue(item, "trigger") ||
    stringValue(item, "deadline") ||
    "Item"
  );
}

function artifactItemDetail(item: Record<string, unknown>) {
  return (
    stringValue(item, "severity") ||
    stringValue(item, "priority") ||
    stringValue(item, "status") ||
    stringValue(item, "recommended_action") ||
    stringValue(item, "ask") ||
    stringValue(item, "reason") ||
    stringValue(item, "deadline") ||
    ""
  );
}

function stringValue(item: Record<string, unknown>, key: string) {
  const value = item[key];
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function splitClarificationAnswers(value: string) {
  return value
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*•\d.、\s]+/, "").trim())
    .filter(Boolean)
    .slice(0, 6);
}

function latestEventId(items: AgentTaskEvent[]) {
  return items.reduce((latest, event) => Math.max(latest, event.event_id), 0);
}

function readConversationId(): string {
  const stored = localStorage.getItem(AGENT_CONVERSATION_STORAGE_KEY);
  if (stored) {
    return stored;
  }
  return createConversationId();
}

function createConversationId(): string {
  const id = crypto.randomUUID();
  localStorage.setItem(AGENT_CONVERSATION_STORAGE_KEY, id);
  return id;
}
</script>
