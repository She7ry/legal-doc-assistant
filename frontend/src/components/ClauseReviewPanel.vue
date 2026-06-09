<template>
  <section class="tool-panel review-panel">
    <div class="panel-heading">
      <div>
        <h2>条款风险审查</h2>
        <p>按条款类型检索，并返回结构化风险评估</p>
      </div>
    </div>

    <el-form label-position="top" class="review-form" @submit.prevent>
      <el-form-item label="条款类型">
        <el-select
          v-model="clauseType"
          filterable
          allow-create
          default-first-option
          placeholder="选择常见条款，或输入自定义条款"
        >
          <el-option
            v-for="option in clauseOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
      </el-form-item>

      <el-form-item label="检索数量">
        <el-input-number v-model="topK" :min="1" :max="20" />
      </el-form-item>

      <el-button
        type="primary"
        :icon="Search"
        :loading="loading"
        :disabled="!clauseType.trim()"
        @click="submit"
      >
        开始审查
      </el-button>
    </el-form>

    <div v-if="result" class="answer-panel">
      <div class="structured-summary">
        <div>
          <span>条款</span>
          <strong>{{ displayClauseName }}</strong>
        </div>
        <div>
          <span>风险等级</span>
          <el-tag :type="riskTagType(result.risk_level)" effect="dark">
            {{ result.risk_level }}
          </el-tag>
        </div>
        <div>
          <span>是否找到</span>
          <strong>{{ foundLabel(result.found) }}</strong>
        </div>
        <div>
          <span>人工复核</span>
          <strong>{{ result.needs_human_review ? "需要" : "暂不需要" }}</strong>
        </div>
      </div>

      <div v-if="result.summary || result.plain_language_explanation" class="review-section">
        <h3>白话解释</h3>
        <p>{{ result.plain_language_explanation || result.summary }}</p>
      </div>

      <div v-if="result.risk_reasons.length" class="review-section">
        <h3>风险原因</h3>
        <div class="risk-card-list">
          <article v-for="(reason, index) in result.risk_reasons" :key="index" class="risk-card">
            <p>{{ reason.reason }}</p>
            <el-tag v-if="reason.citation" type="info" size="small">
              [{{ reason.citation }}]
            </el-tag>
          </article>
        </div>
      </div>

      <div v-if="result.questions_for_lawyer.length" class="review-section">
        <h3>可以问律师的问题</h3>
        <ul class="structured-list">
          <li v-for="question in result.questions_for_lawyer" :key="question">
            {{ question }}
          </li>
        </ul>
      </div>

      <div v-if="result.missing_information.length" class="review-section">
        <h3>缺失信息</h3>
        <ul class="structured-list">
          <li v-for="item in result.missing_information" :key="item">
            {{ item }}
          </li>
        </ul>
      </div>

      <el-alert
        v-if="result.guard_warnings.length"
        type="warning"
        title="审查结果需要谨慎使用"
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
import { computed, ref } from "vue";
import { ElMessage } from "element-plus";
import { Search } from "@element-plus/icons-vue";

import { reviewClause } from "../api/review";
import { formatApiError } from "../api/http";
import type { ClauseReviewResponse } from "../api/types";
import CitationList from "./CitationList.vue";

const clauseOptions = [
  { label: "Termination", value: "termination" },
  { label: "Payment", value: "payment" },
  { label: "Late fee", value: "late fee" },
  { label: "Auto renewal", value: "auto renewal" },
  { label: "Liability limitation", value: "liability limitation" },
  { label: "Indemnification", value: "indemnification" },
  { label: "Confidentiality", value: "confidentiality" },
  { label: "Non-compete", value: "non-compete" },
  { label: "IP ownership", value: "IP ownership" },
  { label: "Data privacy", value: "data privacy" },
  { label: "Governing law", value: "governing law" },
  { label: "Dispute resolution", value: "dispute resolution" },
  { label: "Assignment", value: "assignment" },
  { label: "Audit rights", value: "audit rights" },
  { label: "Notice", value: "notice" },
];

const clauseType = ref("");
const topK = ref(5);
const loading = ref(false);
const result = ref<ClauseReviewResponse | null>(null);

const displayClauseName = computed(() => {
  if (!result.value) {
    return "";
  }
  return result.value.normalized_clause_type || result.value.clause_type || clauseType.value;
});

function foundLabel(found: boolean | null) {
  if (found === true) {
    return "已找到";
  }
  if (found === false) {
    return "未找到";
  }
  return "不确定";
}

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

async function submit() {
  const value = clauseType.value.trim();
  if (!value || loading.value) {
    return;
  }

  loading.value = true;
  result.value = null;
  try {
    result.value = await reviewClause({
      clause_type: value,
      top_k: topK.value,
    });
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}
</script>
