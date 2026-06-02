<template>
  <el-tag :type="meta.type" effect="light" round>{{ meta.label }}</el-tag>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type { IngestJobStatus } from "../api/types";

const props = defineProps<{
  status: IngestJobStatus;
}>();

const meta = computed(() => {
  switch (props.status) {
    case "queued":
      return { label: "排队中", type: "info" as const };
    case "running":
      return { label: "索引中", type: "warning" as const };
    case "succeeded":
      return { label: "已完成", type: "success" as const };
    case "failed":
      return { label: "失败", type: "danger" as const };
    default:
      return { label: props.status, type: "info" as const };
  }
});
</script>
