<template>
  <div class="page-stack">
    <el-tabs v-model="activeTab" class="review-tabs">
      <el-tab-pane label="条款风险审查" name="clause">
        <ClauseReviewPanel />
      </el-tab-pane>
      <el-tab-pane label="合同/政策冲突检测" name="conflict">
        <ConflictCheckPanel />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

import ClauseReviewPanel from "../components/ClauseReviewPanel.vue";
import ConflictCheckPanel from "../components/ConflictCheckPanel.vue";

const activeTab = ref("clause");
</script>
