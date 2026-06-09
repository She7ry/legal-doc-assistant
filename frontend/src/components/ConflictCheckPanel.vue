<template>
  <section class="tool-panel review-panel">
    <div class="panel-heading">
      <div>
        <h2>冲突检测</h2>
        <p>比较合同内容与政策要求，并输出结构化冲突矩阵</p>
      </div>
    </div>

    <el-form label-position="top" class="review-form" @submit.prevent>
      <el-form-item label="合同检索条件">
        <el-input
          v-model="contractQuery"
          type="textarea"
          :rows="3"
          maxlength="500"
          show-word-limit
          resize="none"
          placeholder="例如：payment terms and obligations"
        />
      </el-form-item>

      <el-form-item label="政策检索条件">
        <el-input
          v-model="policyQuery"
          type="textarea"
          :rows="3"
          maxlength="500"
          show-word-limit
          resize="none"
          placeholder="例如：payment policy and compliance requirements"
        />
      </el-form-item>

      <el-form-item label="检索数量">
        <el-input-number v-model="topK" :min="1" :max="20" />
      </el-form-item>

      <el-button
        type="primary"
        :icon="Search"
        :loading="loading"
        :disabled="!contractQuery.trim() || !policyQuery.trim()"
        @click="submit"
      >
        检测冲突
      </el-button>
    </el-form>

    <div v-if="result" class="answer-panel">
      <div class="structured-summary">
        <div>
          <span>结论</span>
          <strong>{{ result.overall_status }}</strong>
        </div>
        <div>
          <span>冲突数量</span>
          <strong>{{ result.conflicts.length }}</strong>
        </div>
        <div>
          <span>人工复核</span>
          <strong>{{ result.needs_human_review ? "需要" : "暂不需要" }}</strong>
        </div>
      </div>

      <el-table
        v-if="result.conflicts.length"
        :data="result.conflicts"
        class="conflict-table"
        row-key="topic"
      >
        <el-table-column prop="topic" label="主题" min-width="150" />
        <el-table-column label="类型" min-width="170">
          <template #default="{ row }">
            <el-tag type="info">{{ formatConflictType(row.conflict_type) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="严重程度" width="120">
          <template #default="{ row }">
            <el-tag :type="riskTagType(row.severity)" effect="dark">
              {{ row.severity }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="合同立场" min-width="240">
          <template #default="{ row }">
            <p class="table-copy">{{ row.contract_position }}</p>
            <span class="citation-refs">{{ formatRefs(row.contract_citations) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="政策立场" min-width="240">
          <template #default="{ row }">
            <p class="table-copy">{{ row.policy_position }}</p>
            <span class="citation-refs">{{ formatRefs(row.policy_citations) }}</span>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="result.conflicts.length" class="review-section">
        <h3>处理建议</h3>
        <div class="risk-card-list">
          <article v-for="conflict in result.conflicts" :key="conflict.topic" class="risk-card">
            <strong>{{ conflict.topic }}</strong>
            <p>{{ conflict.why_conflict }}</p>
            <p>{{ conflict.recommended_action }}</p>
            <span class="citation-refs">
              {{ formatRefs([...conflict.contract_citations, ...conflict.policy_citations]) }}
            </span>
          </article>
        </div>
      </div>

      <div v-else class="review-section">
        <p>{{ result.content }}</p>
      </div>

      <el-alert
        v-if="result.guard_warnings.length"
        type="warning"
        title="检测结果需要谨慎使用"
        :closable="false"
        show-icon
      >
        <ul class="structured-list structured-list--compact">
          <li v-for="warning in result.guard_warnings" :key="warning">
            {{ warning }}
          </li>
        </ul>
      </el-alert>

      <el-collapse class="raw-answer">
        <el-collapse-item title="查看生成文本" name="content">
          <p>{{ result.content }}</p>
        </el-collapse-item>
      </el-collapse>

      <CitationList :citations="result.citations" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";
import { Search } from "@element-plus/icons-vue";

import { checkConflict } from "../api/review";
import { formatApiError } from "../api/http";
import type { ConflictCheckResponse } from "../api/types";
import CitationList from "./CitationList.vue";

const contractQuery = ref("");
const policyQuery = ref("");
const topK = ref(5);
const loading = ref(false);
const result = ref<ConflictCheckResponse | null>(null);

function riskTagType(level: string) {
  if (level === "High") {
    return "danger";
  }
  if (level === "Medium") {
    return "warning";
  }
  if (level === "Low") {
    return "success";
  }
  return "info";
}

function formatConflictType(value: string) {
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatRefs(refs: string[]) {
  return refs.map((ref) => `[${ref}]`).join(" ");
}

async function submit() {
  const contract = contractQuery.value.trim();
  const policy = policyQuery.value.trim();
  if (!contract || !policy || loading.value) {
    return;
  }

  loading.value = true;
  result.value = null;
  try {
    result.value = await checkConflict({
      contract_query: contract,
      policy_query: policy,
      top_k: topK.value,
    });
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}
</script>
