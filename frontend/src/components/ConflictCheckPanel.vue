<template>
  <section class="tool-panel review-panel">
    <div class="panel-heading">
      <div>
        <h2>冲突检测</h2>
        <p>比较合同内容与政策要求</p>
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
      <h3>检测结果</h3>
      <p>{{ result.content }}</p>
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
import type { AnswerResponse } from "../api/types";
import CitationList from "./CitationList.vue";

const contractQuery = ref("");
const policyQuery = ref("");
const topK = ref(5);
const loading = ref(false);
const result = ref<AnswerResponse | null>(null);

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
