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
          <el-tag size="small" effect="plain">{{ citation.source_type || "document" }}</el-tag>
          <strong>{{ citation.file_name }}</strong>
        </div>
        <div class="citation-item__meta">
          <span v-if="citation.page_label">{{ citation.page_label }}</span>
          <span v-else-if="citation.location_label">{{ citation.location_label }}</span>
          <span v-if="citation.chunk_id !== null">Chunk {{ citation.chunk_id }}</span>
          <span v-if="citation.section_heading">{{ citation.section_heading }}</span>
          <span v-if="citation.document_version">v{{ citation.document_version }}</span>
          <span v-if="citation.retrieval_relevance !== null && citation.retrieval_relevance !== undefined">
            Relevance {{ citation.retrieval_relevance.toFixed(2) }}
          </span>
        </div>
        <blockquote>{{ citation.exact_quote || citation.preview }}</blockquote>
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
