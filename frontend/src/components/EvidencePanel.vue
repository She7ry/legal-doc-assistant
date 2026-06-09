<template>
  <div v-if="hasEvidence" class="evidence-panel">
    <div class="evidence-panel__header">
      <span>证据链</span>
      <el-tag size="small" effect="plain">{{ evidence.claims.length }}</el-tag>
    </div>

    <el-alert
      v-if="evidence.missing_evidence.length"
      type="warning"
      title="存在未完全支持的结论"
      :closable="false"
      show-icon
    >
      <ul class="structured-list structured-list--compact">
        <li v-for="item in evidence.missing_evidence" :key="item">{{ item }}</li>
      </ul>
    </el-alert>

    <div v-if="evidence.claims.length" class="evidence-claims">
      <article v-for="claim in evidence.claims" :key="claim.claim_id" class="evidence-claim">
        <div class="evidence-claim__title">
          <el-tag size="small" :type="supportTagType(claim.support_level)" effect="plain">
            {{ supportLabel(claim.support_level) }}
          </el-tag>
          <span>{{ claim.text }}</span>
        </div>
        <div v-if="claim.evidence.length" class="evidence-claim__sources">
          <div
            v-for="item in claim.evidence"
            :key="`${claim.claim_id}-${item.source_id}`"
            class="evidence-source"
          >
            <div class="evidence-source__meta">
              <el-tag size="small" type="primary" effect="dark">{{ item.source_id }}</el-tag>
              <strong>{{ item.file_name }}</strong>
              <span>{{ item.location_label }}</span>
            </div>
            <blockquote>{{ item.quote }}</blockquote>
          </div>
        </div>
        <p v-if="claim.uncertainty" class="evidence-claim__note">{{ claim.uncertainty }}</p>
      </article>
    </div>

    <div v-if="evidence.unsupported_claims.length" class="unsupported-claims">
      <span>未找到依据的部分</span>
      <ul class="structured-list structured-list--compact">
        <li v-for="claim in evidence.unsupported_claims" :key="claim">{{ claim }}</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { EvidenceProfile } from "../api/types";

const props = defineProps<{
  evidence: EvidenceProfile;
}>();

const hasEvidence = computed(() => {
  return (
    props.evidence.claims.length > 0 ||
    props.evidence.missing_evidence.length > 0 ||
    props.evidence.unsupported_claims.length > 0
  );
});

function supportTagType(level: string) {
  if (level === "direct") {
    return "success";
  }
  if (level === "partial") {
    return "warning";
  }
  return "danger";
}

function supportLabel(level: string) {
  if (level === "direct") {
    return "已引用";
  }
  if (level === "partial") {
    return "部分引用";
  }
  return "缺少依据";
}
</script>
