<template>
  <div class="page-stack matters-page">
    <section class="summary-strip matters-summary">
      <div class="metric">
        <span>事项数量</span>
        <strong>{{ matters.length }}</strong>
      </div>
      <div class="metric">
        <span>当前状态</span>
        <strong>{{ selectedMatter ? statusLabel(selectedMatter.status) : "未选择" }}</strong>
      </div>
      <div class="metric">
        <span>交付物</span>
        <strong>{{ selectedMatter?.artifacts.length ?? 0 }}</strong>
      </div>
      <div class="metric">
        <span>Findings</span>
        <strong>{{ findings.length }}</strong>
      </div>
      <div class="metric">
        <span>Gates</span>
        <strong>{{ confirmationGates.length }}</strong>
      </div>
    </section>

    <section class="tool-panel matter-filter-panel">
      <div class="matter-filter-bar">
        <el-select
          v-model="filters.artifactType"
          filterable
          placeholder="Artifact type"
          :disabled="!selectedMatter"
        >
          <el-option label="All artifact types" value="all" />
          <el-option
            v-for="option in artifactTypeOptions"
            :key="option"
            :label="artifactTypeLabel(option)"
            :value="option"
          />
        </el-select>
        <el-select
          v-model="filters.gateStatus"
          placeholder="Gate status"
          :disabled="!selectedMatter"
        >
          <el-option label="All gate statuses" value="all" />
          <el-option
            v-for="option in gateStatusOptions"
            :key="option"
            :label="option"
            :value="option"
          />
        </el-select>
        <el-select
          v-model="filters.riskSeverity"
          placeholder="Risk severity"
          :disabled="!selectedMatter"
        >
          <el-option label="All risk severities" value="all" />
          <el-option
            v-for="option in riskSeverityOptions"
            :key="option"
            :label="option"
            :value="option"
          />
        </el-select>
        <el-button :icon="Refresh" :disabled="!hasActiveFilters" @click="resetFilters">
          Clear
        </el-button>
      </div>
      <div class="matter-filter-summary">
        <el-tag effect="plain">{{ filteredArtifacts.length }} artifacts</el-tag>
        <el-tag effect="plain">{{ filteredConfirmationGates.length }} gates</el-tag>
        <el-tag effect="plain">{{ filteredRiskItemCount }} risk rows</el-tag>
      </div>
    </section>

    <div class="matters-grid">
      <section class="tool-panel matter-list-panel">
        <div class="panel-heading">
          <div>
            <h2>事项列表</h2>
            <p>{{ matters.length }} 个已保存事项</p>
          </div>
          <el-button :icon="Refresh" :loading="loading" @click="loadMatters">刷新</el-button>
        </div>

        <el-empty
          v-if="!loading && !matters.length"
          :image-size="96"
          description="暂无事项"
        />
        <div v-else v-loading="loading" class="matter-list">
          <button
            v-for="matter in matters"
            :key="matter.matter_id"
            class="matter-list-item"
            :class="{ 'matter-list-item--active': matter.matter_id === selectedMatterId }"
            type="button"
            @click="selectMatter(matter.matter_id)"
          >
            <strong>{{ matter.title }}</strong>
            <span>{{ matter.matter_id }}</span>
            <small>
              {{ statusLabel(matter.status) }} · {{ formatDate(matter.updated_at) }}
            </small>
          </button>
        </div>
      </section>

      <section class="tool-panel matter-detail-panel">
        <div class="panel-heading">
          <div>
            <h2>事项档案</h2>
            <p>{{ selectedMatter?.matter_id ?? "选择一个事项查看详情" }}</p>
          </div>
          <el-tag v-if="selectedMatter" effect="plain">
            {{ statusLabel(selectedMatter.status) }}
          </el-tag>
        </div>

        <el-empty
          v-if="!selectedMatter && !detailLoading"
          :image-size="96"
          description="请选择事项"
        />

        <div v-else v-loading="detailLoading" class="matter-detail">
          <dl v-if="selectedMatter" class="matter-profile-grid">
            <div>
              <dt>文档类型</dt>
              <dd>{{ profileText("document_type", "Unknown") }}</dd>
            </div>
            <div>
              <dt>当事人</dt>
              <dd>{{ profileList("parties") }}</dd>
            </div>
            <div>
              <dt>用户立场</dt>
              <dd>{{ profileText("user_side", "Unspecified") }}</dd>
            </div>
            <div>
              <dt>适用法</dt>
              <dd>{{ profileText("governing_law", "Unspecified") }}</dd>
            </div>
            <div>
              <dt>审查范围</dt>
              <dd>{{ profileList("review_scope") }}</dd>
            </div>
            <div>
              <dt>来源任务</dt>
              <dd>{{ selectedMatter.latest_task_id }}</dd>
            </div>
          </dl>

          <section
            v-if="selectedMatter && profileArray('open_questions').length"
            class="matter-detail-section"
          >
            <h3>待确认信息</h3>
            <ul class="structured-list">
              <li v-for="question in profileArray('open_questions')" :key="question">
                {{ question }}
              </li>
            </ul>
          </section>

          <section
            v-if="selectedMatter && profileDateItems.length"
            class="matter-detail-section"
          >
            <h3>关键日期</h3>
            <ul class="structured-list">
              <li v-for="dateItem in profileDateItems" :key="dateItemKey(dateItem)">
                {{ dateItemLabel(dateItem) }}
              </li>
            </ul>
          </section>

          <section
            v-if="selectedMatter && findings.length"
            class="matter-detail-section"
          >
            <h3>Review Findings</h3>
            <div class="risk-card-list">
              <article
                v-for="finding in findings"
                :key="finding.finding_id"
                class="risk-card"
              >
                <div class="agent-finding__title">
                  <strong>{{ finding.category }}</strong>
                  <span>
                    <el-tag :type="severityType(finding.severity)" size="small" effect="dark">
                      {{ finding.severity }}
                    </el-tag>
                    <el-tag size="small" effect="plain">
                      {{ finding.human_review_status }}
                    </el-tag>
                  </span>
                </div>
                <p>{{ finding.summary }}</p>
                <small>
                  {{ finding.support_level }} / {{ finding.evidence_coverage }}
                  <template v-if="finding.location_label"> · {{ finding.location_label }}</template>
                </small>
                <p v-if="finding.unsupported_reason" class="table-copy">
                  {{ finding.unsupported_reason }}
                </p>
                <div v-if="finding.citations.length" class="citation-refs">
                  <el-tag
                    v-for="sourceId in finding.citations"
                    :key="`${finding.finding_id}-${sourceId}`"
                    size="small"
                    effect="plain"
                  >
                    {{ sourceId }}
                  </el-tag>
                </div>
                <div class="agent-gate-actions">
                  <el-button
                    :icon="CircleCheck"
                    size="small"
                    type="success"
                    plain
                    :loading="isFindingUpdating(finding, 'approved')"
                    :disabled="findingActionDisabled(finding, 'approved')"
                    @click="changeFindingStatus(finding, 'approved')"
                  >
                    Approve
                  </el-button>
                  <el-button
                    :icon="Warning"
                    size="small"
                    type="warning"
                    plain
                    :loading="isFindingUpdating(finding, 'needs_info')"
                    :disabled="findingActionDisabled(finding, 'needs_info')"
                    @click="changeFindingStatus(finding, 'needs_info')"
                  >
                    Need info
                  </el-button>
                  <el-button
                    :icon="CircleClose"
                    size="small"
                    plain
                    :loading="isFindingUpdating(finding, 'waived')"
                    :disabled="findingActionDisabled(finding, 'waived')"
                    @click="changeFindingStatus(finding, 'waived')"
                  >
                    Waive
                  </el-button>
                </div>
              </article>
            </div>
          </section>

          <section
            v-if="selectedMatter && confirmationGates.length"
            class="matter-detail-section"
          >
            <h3>Confirmation Gates</h3>
            <el-empty
              v-if="!filteredConfirmationGates.length"
              :image-size="72"
              description="No gates match the current filters"
            />
            <div v-else class="agent-gate-list">
              <article
                v-for="gate in filteredConfirmationGates"
                :key="gateKey(gate)"
                class="agent-gate"
              >
                <div class="agent-gate__header">
                  <strong>{{ gateTitle(gate) }}</strong>
                  <span>
                    <el-tag
                      :type="gatePriorityType(gatePriority(gate))"
                      size="small"
                      effect="dark"
                    >
                      {{ gatePriority(gate) }}
                    </el-tag>
                    <el-tag size="small" effect="plain">
                      {{ gateStatus(gate) }}
                    </el-tag>
                  </span>
                </div>
                <p>{{ gateQuestion(gate) }}</p>
                <small v-if="gateReason(gate)">{{ gateReason(gate) }}</small>
                <div v-if="gateRefs(gate).length" class="citation-refs">
                  <el-tag
                    v-for="sourceId in gateRefs(gate)"
                    :key="`${gateKey(gate)}-${sourceId}`"
                    size="small"
                    effect="plain"
                  >
                    {{ sourceId }}
                  </el-tag>
                </div>
                <div class="agent-gate-actions">
                  <el-button
                    :icon="CircleCheck"
                    size="small"
                    type="success"
                    plain
                    :loading="isGateUpdating(gate, 'approved')"
                    :disabled="gateActionDisabled(gate, 'approved')"
                    @click="changeGateStatus(gate, 'approved')"
                  >
                    Approve
                  </el-button>
                  <el-button
                    :icon="Warning"
                    size="small"
                    type="warning"
                    plain
                    :loading="isGateUpdating(gate, 'needs_info')"
                    :disabled="gateActionDisabled(gate, 'needs_info')"
                    @click="changeGateStatus(gate, 'needs_info')"
                  >
                    Need info
                  </el-button>
                  <el-button
                    :icon="CircleClose"
                    size="small"
                    plain
                    :loading="isGateUpdating(gate, 'waived')"
                    :disabled="gateActionDisabled(gate, 'waived')"
                    @click="changeGateStatus(gate, 'waived')"
                  >
                    Waive
                  </el-button>
                </div>
                <div v-if="gateDecisions(gate).length" class="agent-gate-audit">
                  <strong>Decision history</strong>
                  <ol>
                    <li
                      v-for="decision in gateDecisions(gate)"
                      :key="gateDecisionKey(gate, decision)"
                    >
                      <span>
                        {{ decisionStatus(decision) }} 路 {{ decisionBy(decision) }} 路
                        {{ decisionTime(decision) }}
                      </span>
                      <small v-if="decisionNote(decision)">{{ decisionNote(decision) }}</small>
                    </li>
                  </ol>
                </div>
              </article>
            </div>
          </section>
        </div>
      </section>

      <section class="tool-panel matter-artifact-panel">
        <div class="panel-heading">
          <div>
            <h2>结构化交付物</h2>
            <p>风险矩阵、问题清单、谈判清单和义务日历</p>
          </div>
          <el-button
            :icon="CircleCheck"
            type="primary"
            plain
            :loading="formalReportLoading"
            :disabled="!canGenerateFormalReport"
            @click="createFormalReport"
          >
            Formal report
          </el-button>
          <el-button
            :icon="Download"
            plain
            :loading="artifactBundleExporting"
            :disabled="!canExportArtifactBundle"
            @click="downloadArtifactBundle"
          >
            Export all
          </el-button>
        </div>

        <el-empty
          v-if="selectedMatter && !selectedMatter.artifacts.length"
          :image-size="96"
          description="暂无交付物"
        />
        <el-empty
          v-else-if="!selectedMatter"
          :image-size="96"
          description="请选择事项"
        />
        <el-empty
          v-else-if="!filteredArtifacts.length"
          :image-size="96"
          description="No artifacts match the current filters"
        />
        <div v-else class="matter-artifact-list">
          <article
            v-for="artifact in filteredArtifacts"
            :key="artifact.artifact_id"
            class="agent-artifact"
          >
            <div class="agent-artifact__header">
              <strong>{{ artifact.title }}</strong>
              <span class="agent-artifact__actions">
                <el-tag size="small" effect="plain">v{{ artifact.version }}</el-tag>
                <el-button
                  :icon="Edit"
                  size="small"
                  plain
                  :loading="isArtifactSaving(artifact)"
                  :disabled="artifactEditSaving"
                  @click="openArtifactEdit(artifact)"
                >
                  Edit
                </el-button>
                <el-button
                  :icon="Download"
                  size="small"
                  plain
                  :loading="isArtifactExporting(artifact, 'markdown')"
                  :disabled="Boolean(artifactExporting)"
                  @click="downloadArtifact(artifact, 'markdown')"
                >
                  MD
                </el-button>
                <el-button
                  :icon="Download"
                  size="small"
                  plain
                  :loading="isArtifactExporting(artifact, 'docx')"
                  :disabled="Boolean(artifactExporting)"
                  @click="downloadArtifact(artifact, 'docx')"
                >
                  DOCX
                </el-button>
              </span>
            </div>
            <p>{{ artifact.summary }}</p>
            <ul v-if="artifactDisplayItems(artifact).length" class="artifact-item-list">
              <li
                v-for="item in artifactDisplayItems(artifact).slice(0, 6)"
                :key="artifactItemKey(artifact.artifact_id, item)"
              >
                <strong>{{ artifactItemTitle(item) }}</strong>
                <span>{{ artifactItemDetail(item) }}</span>
              </li>
            </ul>
          </article>
        </div>
      </section>
    </div>

    <el-dialog v-model="artifactEditVisible" title="Edit artifact" width="720px">
      <el-form class="artifact-edit-form" label-position="top" @submit.prevent>
        <el-form-item label="Title">
          <el-input v-model="artifactEditForm.title" maxlength="200" show-word-limit />
        </el-form-item>
        <el-form-item label="Status">
          <el-select v-model="artifactEditForm.status">
            <el-option label="Draft" value="draft" />
            <el-option label="Active" value="active" />
            <el-option label="Needs review" value="needs_review" />
            <el-option label="Approved" value="approved" />
            <el-option label="Archived" value="archived" />
          </el-select>
        </el-form-item>
        <el-form-item label="Summary">
          <el-input
            v-model="artifactEditForm.summary"
            type="textarea"
            :rows="3"
            maxlength="2000"
            show-word-limit
            resize="none"
          />
        </el-form-item>
        <el-form-item label="Items JSON">
          <el-input
            v-model="artifactEditForm.itemsJson"
            type="textarea"
            :rows="10"
            resize="vertical"
          />
        </el-form-item>
        <el-form-item label="Review note">
          <el-input
            v-model="artifactEditForm.note"
            type="textarea"
            :rows="2"
            maxlength="1000"
            show-word-limit
            resize="none"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="artifactEditVisible = false">Cancel</el-button>
        <el-button
          type="primary"
          :loading="artifactEditSaving"
          :disabled="!artifactEditForm.title.trim()"
          @click="saveArtifactEdit"
        >
          Save version
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { CircleCheck, CircleClose, Download, Edit, Refresh, Warning } from "@element-plus/icons-vue";

import {
  exportMatterArtifactDocx,
  exportMatterArtifactMarkdown,
  exportMatterArtifactsZip,
  generateMatterFormalReport,
  getMatter,
  listMatters,
  updateMatterArtifact,
  updateMatterConfirmationGate,
  updateMatterFinding,
} from "../api/matters";
import { formatApiError } from "../api/http";
import type {
  MatterArtifactRecord,
  MatterFindingRecord,
  MatterFindingUpdateRequest,
  MatterConfirmationGateStatus,
  MatterRecord,
} from "../api/types";

const route = useRoute();
const matters = ref<MatterRecord[]>([]);
const selectedMatter = ref<MatterRecord | null>(null);
const selectedMatterId = ref("");
const loading = ref(false);
const detailLoading = ref(false);
const gateUpdating = ref("");
const findingUpdating = ref("");
const formalReportLoading = ref(false);
const artifactExporting = ref("");
const artifactBundleExporting = ref(false);
const artifactEditVisible = ref(false);
const artifactEditSaving = ref(false);
const editingArtifactId = ref("");
const filters = reactive({
  artifactType: "all",
  gateStatus: "all",
  riskSeverity: "all",
});
const artifactEditForm = reactive({
  title: "",
  summary: "",
  status: "active",
  itemsJson: "[]",
  note: "",
});

const profileDateItems = computed(() => {
  const value = selectedMatter.value?.matter_profile.key_dates;
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => isRecord(item))
    : [];
});
const confirmationGates = computed(() => {
  const value = selectedMatter.value?.matter_profile.confirmation_gates;
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => isRecord(item))
    : [];
});
const findings = computed(() => selectedMatter.value?.findings ?? []);
const artifactTypeOptions = computed(() =>
  uniqueStrings((selectedMatter.value?.artifacts ?? []).map((artifact) => artifact.artifact_type)),
);
const gateStatusOptions = computed(() =>
  uniqueStrings(confirmationGates.value.map((gate) => gateStatus(gate))),
);
const riskSeverityOptions = computed(() =>
  uniqueStrings(
    (selectedMatter.value?.artifacts ?? []).flatMap((artifact) =>
      artifact.items.map((item) => stringValue(item, "severity")),
    ),
  ),
);
const filteredConfirmationGates = computed(() =>
  confirmationGates.value.filter(
    (gate) => filters.gateStatus === "all" || gateStatus(gate) === filters.gateStatus,
  ),
);
const filteredArtifacts = computed(() =>
  (selectedMatter.value?.artifacts ?? []).filter((artifact) => {
    if (filters.artifactType !== "all" && artifact.artifact_type !== filters.artifactType) {
      return false;
    }
    if (filters.riskSeverity !== "all" && !artifactDisplayItems(artifact).length) {
      return false;
    }
    return true;
  }),
);
const filteredRiskItemCount = computed(() =>
  filteredArtifacts.value.reduce((count, artifact) => {
    const items =
      filters.riskSeverity === "all" ? artifactRiskItems(artifact) : artifactDisplayItems(artifact);
    return count + items.length;
  }, 0),
);
const hasActiveFilters = computed(
  () =>
    filters.artifactType !== "all" ||
    filters.gateStatus !== "all" ||
    filters.riskSeverity !== "all",
);
const unresolvedRequiredGates = computed(() =>
  confirmationGates.value.filter(
    (gate) => gateRequired(gate) && !["approved", "waived"].includes(gateStatus(gate)),
  ),
);
const unresolvedFindings = computed(() =>
  findings.value.filter((finding) => !isFindingFormalReady(finding)),
);
const canGenerateFormalReport = computed(
  () =>
    Boolean(selectedMatter.value) &&
    !unresolvedRequiredGates.value.length &&
    !unresolvedFindings.value.length,
);
const canExportArtifactBundle = computed(
  () =>
    Boolean(selectedMatter.value?.artifacts.length) &&
    !artifactBundleExporting.value &&
    !Boolean(artifactExporting.value),
);
const requestedMatterId = computed(() => {
  const value = route.query.matter_id;
  return Array.isArray(value) ? value[0] || "" : value || "";
});

onMounted(() => {
  void loadMatters();
});

watch(requestedMatterId, (matterId) => {
  if (matterId && matterId !== selectedMatterId.value) {
    void selectMatter(matterId);
  }
});

async function loadMatters() {
  loading.value = true;
  try {
    const response = await listMatters();
    matters.value = response.matters;
    const matterId = requestedMatterId.value || selectedMatterId.value;
    if (matterId) {
      await selectMatter(matterId);
    } else if (matters.value.length) {
      await selectMatter(matters.value[0].matter_id);
    } else {
      selectedMatter.value = null;
      selectedMatterId.value = "";
    }
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

async function selectMatter(matterId: string) {
  if (!matterId || detailLoading.value) {
    return;
  }
  selectedMatterId.value = matterId;
  detailLoading.value = true;
  try {
    selectedMatter.value = await getMatter(matterId);
    pruneFiltersForSelectedMatter();
  } catch (error) {
    selectedMatter.value = null;
    ElMessage.error(formatApiError(error));
  } finally {
    detailLoading.value = false;
  }
}

async function changeGateStatus(
  gate: Record<string, unknown>,
  status: MatterConfirmationGateStatus,
) {
  if (!selectedMatter.value || gateUpdating.value) {
    return;
  }
  const gateId = gateKey(gate);
  if (!gateId || gateStatus(gate) === status) {
    return;
  }

  let note: string | null = null;
  try {
    const promptResult = await ElMessageBox.prompt(
      gatePrompt(status),
      gateActionTitle(status),
      {
        confirmButtonText: "Save",
        cancelButtonText: "Cancel",
        inputType: "textarea",
        inputPlaceholder: "Optional decision note",
        inputValidator(value) {
          return value.length <= 1000 || "Note is too long";
        },
      },
    );
    note = promptResult.value.trim() || null;
  } catch {
    return;
  }

  gateUpdating.value = `${gateId}:${status}`;
  try {
    const updatedMatter = await updateMatterConfirmationGate(
      selectedMatter.value.matter_id,
      gateId,
      { status, note, confirmed_value: confirmedValueForGate(gate, status, note) },
    );
    applyMatterUpdate(updatedMatter);
    ElMessage.success("Gate updated.");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    gateUpdating.value = "";
  }
}

async function changeFindingStatus(
  finding: MatterFindingRecord,
  humanReviewStatus: MatterFindingUpdateRequest["human_review_status"],
) {
  if (!selectedMatter.value || findingUpdating.value) {
    return;
  }
  if (finding.human_review_status === humanReviewStatus) {
    return;
  }

  let note: string | null = null;
  try {
    const promptResult = await ElMessageBox.prompt(
      "Record an optional review note for this finding.",
      "Update finding review",
      {
        confirmButtonText: "Save",
        cancelButtonText: "Cancel",
        inputType: "textarea",
        inputPlaceholder: "Optional finding review note",
        inputValidator(value) {
          return value.length <= 1000 || "Note is too long";
        },
      },
    );
    note = promptResult.value.trim() || null;
  } catch {
    return;
  }

  findingUpdating.value = `${finding.finding_id}:${humanReviewStatus}`;
  try {
    const updatedMatter = await updateMatterFinding(
      selectedMatter.value.matter_id,
      finding.finding_id,
      { human_review_status: humanReviewStatus, note },
    );
    applyMatterUpdate(updatedMatter);
    ElMessage.success("Finding updated.");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    findingUpdating.value = "";
  }
}

async function createFormalReport() {
  if (!selectedMatter.value || !canGenerateFormalReport.value || formalReportLoading.value) {
    return;
  }

  let note: string | null = null;
  try {
    const promptResult = await ElMessageBox.prompt(
      "Record an optional note for this formal report version.",
      "Generate formal report",
      {
        confirmButtonText: "Generate",
        cancelButtonText: "Cancel",
        inputType: "textarea",
        inputPlaceholder: "Optional version note",
        inputValidator(value) {
          return value.length <= 1000 || "Note is too long";
        },
      },
    );
    note = promptResult.value.trim() || null;
  } catch {
    return;
  }

  formalReportLoading.value = true;
  try {
    const updatedMatter = await generateMatterFormalReport(selectedMatter.value.matter_id, {
      note,
    });
    applyMatterUpdate(updatedMatter);
    ElMessage.success("Formal report artifact generated.");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    formalReportLoading.value = false;
  }
}

function applyMatterUpdate(updatedMatter: MatterRecord) {
  selectedMatter.value = updatedMatter;
  const index = matters.value.findIndex((matter) => matter.matter_id === updatedMatter.matter_id);
  if (index >= 0) {
    matters.value[index] = {
      ...matters.value[index],
      ...updatedMatter,
    };
  }
  pruneFiltersForSelectedMatter();
}

function openArtifactEdit(artifact: MatterArtifactRecord) {
  editingArtifactId.value = artifact.artifact_id;
  artifactEditForm.title = artifact.title;
  artifactEditForm.summary = artifact.summary;
  artifactEditForm.status = artifact.status || "active";
  artifactEditForm.itemsJson = JSON.stringify(artifact.items, null, 2);
  artifactEditForm.note = "";
  artifactEditVisible.value = true;
}

async function saveArtifactEdit() {
  if (!selectedMatter.value || !editingArtifactId.value || artifactEditSaving.value) {
    return;
  }
  const items = parseArtifactItemsJson(artifactEditForm.itemsJson);
  if (!items) {
    return;
  }

  artifactEditSaving.value = true;
  try {
    const updatedMatter = await updateMatterArtifact(
      selectedMatter.value.matter_id,
      editingArtifactId.value,
      {
        title: artifactEditForm.title.trim(),
        summary: artifactEditForm.summary.trim(),
        status: artifactEditForm.status,
        items,
        note: artifactEditForm.note.trim() || null,
      },
    );
    applyMatterUpdate(updatedMatter);
    artifactEditVisible.value = false;
    ElMessage.success("Artifact version saved.");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    artifactEditSaving.value = false;
  }
}

function parseArtifactItemsJson(value: string): Record<string, unknown>[] | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    ElMessage.error("Items JSON is not valid.");
    return null;
  }
  if (
    !Array.isArray(parsed) ||
    parsed.some((item) => !item || typeof item !== "object" || Array.isArray(item))
  ) {
    ElMessage.error("Items JSON must be an array of objects.");
    return null;
  }
  return parsed as Record<string, unknown>[];
}

function isArtifactSaving(artifact: MatterArtifactRecord) {
  return artifactEditSaving.value && editingArtifactId.value === artifact.artifact_id;
}

async function downloadArtifact(
  artifact: MatterArtifactRecord,
  format: "markdown" | "docx",
) {
  if (!selectedMatter.value || artifactExporting.value) {
    return;
  }

  artifactExporting.value = `${artifact.artifact_id}:${format}`;
  try {
    if (format === "docx") {
      const blob = await exportMatterArtifactDocx(
        selectedMatter.value.matter_id,
        artifact.artifact_id,
      );
      downloadBlob(blob, artifactExportFilename(selectedMatter.value.matter_id, artifact, "docx"));
    } else {
      const markdown = await exportMatterArtifactMarkdown(
        selectedMatter.value.matter_id,
        artifact.artifact_id,
      );
      downloadTextFile(
        markdown,
        artifactExportFilename(selectedMatter.value.matter_id, artifact, "md"),
        "text/markdown;charset=utf-8",
      );
    }
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    artifactExporting.value = "";
  }
}

function isArtifactExporting(
  artifact: MatterArtifactRecord,
  format: "markdown" | "docx",
) {
  return artifactExporting.value === `${artifact.artifact_id}:${format}`;
}

async function downloadArtifactBundle() {
  if (!selectedMatter.value || !canExportArtifactBundle.value) {
    return;
  }

  artifactBundleExporting.value = true;
  try {
    const blob = await exportMatterArtifactsZip(selectedMatter.value.matter_id, "docx");
    downloadBlob(blob, artifactBundleFilename(selectedMatter.value.matter_id, "docx"));
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    artifactBundleExporting.value = false;
  }
}

function profileText(key: string, fallback = "") {
  const value = selectedMatter.value?.matter_profile[key];
  if (typeof value === "string") {
    return value || fallback;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return fallback;
}

function resetFilters() {
  filters.artifactType = "all";
  filters.gateStatus = "all";
  filters.riskSeverity = "all";
}

function pruneFiltersForSelectedMatter() {
  if (
    filters.artifactType !== "all" &&
    !artifactTypeOptions.value.includes(filters.artifactType)
  ) {
    filters.artifactType = "all";
  }
  if (filters.gateStatus !== "all" && !gateStatusOptions.value.includes(filters.gateStatus)) {
    filters.gateStatus = "all";
  }
  if (
    filters.riskSeverity !== "all" &&
    !riskSeverityOptions.value.includes(filters.riskSeverity)
  ) {
    filters.riskSeverity = "all";
  }
}

function artifactTypeLabel(value: string) {
  return value.replace(/_/g, " ");
}

function uniqueStrings(values: string[]) {
  const seen = new Set<string>();
  const result = [];
  for (const value of values) {
    const text = value.trim();
    const key = text.toLowerCase();
    if (!text || seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(text);
  }
  return result.sort((left, right) => left.localeCompare(right));
}

function profileArray(key: string) {
  const value = selectedMatter.value?.matter_profile[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item)).filter(Boolean);
}

function profileList(key: string) {
  const values = profileArray(key);
  return values.length ? values.join(", ") : "Unspecified";
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    active: "进行中",
    needs_input: "待确认",
    closed: "已关闭",
    archived: "已归档",
  };
  return labels[status] ?? status;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function dateItemKey(item: Record<string, unknown>) {
  return `${stringValue(item, "label")}-${stringValue(item, "value")}`;
}

function dateItemLabel(item: Record<string, unknown>) {
  const label = stringValue(item, "label") || "日期";
  const value = stringValue(item, "value") || "未确认";
  const description = stringValue(item, "description");
  return description ? `${label}: ${value} · ${description}` : `${label}: ${value}`;
}

function artifactItemKey(artifactId: string, item: Record<string, unknown>) {
  return `${artifactId}-${stringValue(item, "item_id") || artifactItemTitle(item)}`;
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

function artifactDisplayItems(artifact: MatterArtifactRecord) {
  if (filters.riskSeverity === "all") {
    return artifact.items;
  }
  return artifact.items.filter((item) => stringValue(item, "severity") === filters.riskSeverity);
}

function artifactRiskItems(artifact: MatterArtifactRecord) {
  return artifact.items.filter((item) => Boolean(stringValue(item, "severity")));
}

function artifactExportFilename(
  matterId: string,
  artifact: MatterArtifactRecord,
  extension: "md" | "docx",
) {
  return `${slug(matterId)}-${slug(artifact.artifact_id)}-v${artifact.version}.${extension}`;
}

function artifactBundleFilename(matterId: string, format: "markdown" | "docx" | "both") {
  return `${slug(matterId)}-artifacts-${format}.zip`;
}

function downloadTextFile(content: string, filename: string, type: string) {
  downloadBlob(new Blob([content], { type }), filename);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function slug(value: string) {
  return value.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-._]+|[-._]+$/g, "").slice(0, 80) || "artifact";
}

function gateKey(gate: Record<string, unknown>) {
  return stringValue(gate, "gate_id") || gateTitle(gate);
}

function gateTitle(gate: Record<string, unknown>) {
  return stringValue(gate, "title") || "Confirmation gate";
}

function gateQuestion(gate: Record<string, unknown>) {
  return stringValue(gate, "question") || "Confirm before relying on this output.";
}

function gateReason(gate: Record<string, unknown>) {
  return stringValue(gate, "reason");
}

function gatePriority(gate: Record<string, unknown>) {
  return stringValue(gate, "priority") || "normal";
}

function gateStatus(gate: Record<string, unknown>) {
  return stringValue(gate, "status") || "pending";
}

function gateRequired(gate: Record<string, unknown>) {
  const value = gate.required;
  return typeof value === "boolean" ? value : true;
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

function isFindingFormalReady(finding: MatterFindingRecord) {
  const reviewReady =
    !finding.needs_human_review ||
    ["approved", "waived", "resolved", "not_required"].includes(finding.human_review_status);
  const evidenceReady =
    Boolean(finding.citations.length) &&
    Boolean(finding.source_quote) &&
    Boolean(finding.location_label) &&
    Boolean(finding.support_level) &&
    (finding.support_level === "direct" || Boolean(finding.unsupported_reason));
  return reviewReady && evidenceReady;
}

function isFindingUpdating(
  finding: MatterFindingRecord,
  humanReviewStatus: MatterFindingUpdateRequest["human_review_status"],
) {
  return findingUpdating.value === `${finding.finding_id}:${humanReviewStatus}`;
}

function findingActionDisabled(
  finding: MatterFindingRecord,
  humanReviewStatus: MatterFindingUpdateRequest["human_review_status"],
) {
  return Boolean(findingUpdating.value) || finding.human_review_status === humanReviewStatus;
}

function isGateUpdating(gate: Record<string, unknown>, status: MatterConfirmationGateStatus) {
  return gateUpdating.value === `${gateKey(gate)}:${status}`;
}

function gateActionDisabled(
  gate: Record<string, unknown>,
  status: MatterConfirmationGateStatus,
) {
  return Boolean(gateUpdating.value) || gateStatus(gate) === status;
}

function gateActionTitle(status: MatterConfirmationGateStatus) {
  const labels: Record<MatterConfirmationGateStatus, string> = {
    pending: "Reset gate",
    approved: "Approve gate",
    waived: "Waive gate",
    needs_info: "Request more information",
  };
  return labels[status];
}

function gatePrompt(status: MatterConfirmationGateStatus) {
  const labels: Record<MatterConfirmationGateStatus, string> = {
    pending: "Record why this gate is being moved back to pending.",
    approved: "Record why this gate is approved.",
    waived: "Record why this gate is waived.",
    needs_info: "Record what information is still needed.",
  };
  return labels[status];
}

function confirmedValueForGate(
  gate: Record<string, unknown>,
  status: MatterConfirmationGateStatus,
  note: string | null,
) {
  if (status !== "approved" || !note) {
    return null;
  }
  return gateProfileField(gate) ? note : null;
}

function gateProfileField(gate: Record<string, unknown>) {
  const metadata = gate.metadata;
  if (!isRecord(metadata)) {
    return "";
  }
  return stringValue(metadata, "profile_field");
}

function gateDecisions(gate: Record<string, unknown>) {
  const metadata = gate.metadata;
  if (!isRecord(metadata)) {
    return [];
  }
  return arrayOfRecords(metadata.decisions).slice().reverse().slice(0, 5);
}

function gateDecisionKey(gate: Record<string, unknown>, decision: Record<string, unknown>) {
  return `${gateKey(gate)}-${decisionStatus(decision)}-${decisionTime(decision)}-${decisionNote(decision)}`;
}

function decisionStatus(decision: Record<string, unknown>) {
  return stringValue(decision, "status") || "updated";
}

function decisionBy(decision: Record<string, unknown>) {
  return stringValue(decision, "decided_by") || "unknown";
}

function decisionTime(decision: Record<string, unknown>) {
  const value = stringValue(decision, "decided_at");
  if (!value) {
    return "no timestamp";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function decisionNote(decision: Record<string, unknown>) {
  return stringValue(decision, "note");
}

function gateRefs(gate: Record<string, unknown>) {
  return [
    ...arrayOfStrings(gate.citations),
    ...arrayOfStrings(gate.related_finding_ids),
    ...arrayOfStrings(gate.related_artifact_ids),
  ].filter(Boolean).slice(0, 8);
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

function arrayOfStrings(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item)).filter(Boolean);
}

function arrayOfRecords(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is Record<string, unknown> => isRecord(item));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
</script>
