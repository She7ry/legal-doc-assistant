<template>
  <section class="tool-panel review-panel">
    <div class="panel-heading">
      <div>
        <h2>条款风险审查</h2>
        <p>按条款类型检索并生成风险评估</p>
      </div>
    </div>

    <el-form label-position="top" class="review-form" @submit.prevent>
      <el-form-item label="条款类型">
        <el-input
          v-model="clauseType"
          maxlength="200"
          placeholder="例如：termination clause、non-compete、late payment penalty"
        />
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
      <h3>审查结果</h3>
      <p>{{ result.content }}</p>
      <CitationList :citations="result.citations" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";
import { Search } from "@element-plus/icons-vue";

import { reviewClause } from "../api/review";
import { formatApiError } from "../api/http";
import type { AnswerResponse } from "../api/types";
import CitationList from "./CitationList.vue";

const clauseType = ref("");
const topK = ref(5);
const loading = ref(false);
const result = ref<AnswerResponse | null>(null);

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
