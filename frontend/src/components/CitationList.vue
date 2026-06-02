<template>
  <div v-if="citations.length" class="citation-list">
    <div class="citation-list__header">
      <span>引用来源</span>
      <el-tag size="small" effect="plain">{{ citations.length }}</el-tag>
    </div>

    <div class="citation-list__items">
      <article v-for="citation in citations" :key="citation.source_id" class="citation-item">
        <div class="citation-item__title">
          <el-tag size="small" type="primary" effect="dark">{{ citation.source_id }}</el-tag>
          <strong>{{ citation.file_name }}</strong>
        </div>
        <div class="citation-item__meta">
          {{ citation.location_label }}
          <span v-if="citation.chunk_id !== null">Chunk {{ citation.chunk_id }}</span>
        </div>
        <p>{{ citation.preview }}</p>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Citation } from "../api/types";

defineProps<{
  citations: Citation[];
}>();
</script>
