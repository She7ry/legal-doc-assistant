<template>
  <section class="tool-panel memory-panel">
    <div class="panel-heading">
      <div>
        <h2>Memory</h2>
        <p>User preferences and stable context</p>
      </div>
      <el-button :icon="Refresh" :loading="loading" @click="loadMemories">Refresh</el-button>
    </div>

    <el-form label-position="top" class="memory-form" @submit.prevent>
      <el-form-item label="Key">
        <el-input v-model="form.key" placeholder="answer_style" />
      </el-form-item>
      <el-form-item label="Type">
        <el-select v-model="form.type">
          <el-option label="Preference" value="preference" />
          <el-option label="Fact" value="fact" />
          <el-option label="Task state" value="task_state" />
          <el-option label="Feedback" value="feedback" />
          <el-option label="Correction" value="correction" />
        </el-select>
      </el-form-item>
      <el-form-item label="Content">
        <el-input
          v-model="form.content"
          type="textarea"
          :rows="3"
          resize="none"
          maxlength="2000"
          show-word-limit
          placeholder="Prefer concise Chinese answers with implementation details"
        />
      </el-form-item>
      <div class="panel-actions panel-actions--left">
        <el-button type="primary" :icon="Plus" :disabled="!canCreate" @click="submitMemory">
          Add
        </el-button>
      </div>
    </el-form>

    <el-table
      v-loading="loading"
      :data="memories"
      row-key="memory_id"
      empty-text="No active memories"
      class="memory-table"
    >
      <el-table-column prop="type" label="Type" width="108" />
      <el-table-column prop="key" label="Key" width="132" />
      <el-table-column prop="content" label="Content" min-width="220" show-overflow-tooltip />
      <el-table-column prop="confidence" label="Conf." width="86">
        <template #default="{ row }">{{ row.confidence.toFixed(2) }}</template>
      </el-table-column>
      <el-table-column label="Actions" width="132" fixed="right">
        <template #default="{ row }">
          <el-button :icon="Edit" text @click="openEdit(row)" />
          <el-button :icon="Delete" text type="danger" @click="removeMemory(row)" />
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="editDialogVisible" title="Edit memory" width="520px">
      <el-form label-position="top" @submit.prevent>
        <el-form-item label="Key">
          <el-input v-model="editForm.key" />
        </el-form-item>
        <el-form-item label="Content">
          <el-input v-model="editForm.content" type="textarea" :rows="4" resize="none" />
        </el-form-item>
        <el-form-item label="Confidence">
          <el-input-number v-model="editForm.confidence" :min="0" :max="1" :step="0.05" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="saving" @click="saveEdit">Save</el-button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { Delete, Edit, Plus, Refresh } from "@element-plus/icons-vue";

import { createMemory, deleteMemory, listMemories, updateMemory } from "../api/memories";
import { formatApiError } from "../api/http";
import type { MemoryRecord } from "../api/types";

const memories = ref<MemoryRecord[]>([]);
const loading = ref(false);
const saving = ref(false);
const editDialogVisible = ref(false);
const editingMemoryId = ref<string | null>(null);

const form = reactive({
  type: "preference",
  key: "",
  content: "",
});

const editForm = reactive({
  key: "",
  content: "",
  confidence: 0.95,
});

const canCreate = computed(() => Boolean(form.key.trim() && form.content.trim()));

onMounted(() => {
  void loadMemories();
});

async function loadMemories() {
  loading.value = true;
  try {
    const response = await listMemories();
    memories.value = response.memories;
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

async function submitMemory() {
  if (!canCreate.value) {
    return;
  }
  saving.value = true;
  try {
    await createMemory({
      scope: "user",
      type: form.type,
      key: form.key.trim(),
      content: form.content.trim(),
      value: { text: form.content.trim() },
      source: "explicit",
      confidence: 0.95,
      visibility: "private",
    });
    form.key = "";
    form.content = "";
    await loadMemories();
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    saving.value = false;
  }
}

function openEdit(memory: MemoryRecord) {
  editingMemoryId.value = memory.memory_id;
  editForm.key = memory.key;
  editForm.content = memory.content;
  editForm.confidence = memory.confidence;
  editDialogVisible.value = true;
}

async function saveEdit() {
  if (!editingMemoryId.value) {
    return;
  }
  saving.value = true;
  try {
    await updateMemory(editingMemoryId.value, {
      key: editForm.key.trim(),
      content: editForm.content.trim(),
      value: { text: editForm.content.trim() },
      confidence: editForm.confidence,
      status: "active",
    });
    editDialogVisible.value = false;
    await loadMemories();
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    saving.value = false;
  }
}

async function removeMemory(memory: MemoryRecord) {
  try {
    await ElMessageBox.confirm("Delete this memory?", "Confirm", {
      type: "warning",
      confirmButtonText: "Delete",
      cancelButtonText: "Cancel",
    });
    await deleteMemory(memory.memory_id);
    await loadMemories();
  } catch (error) {
    if (error !== "cancel") {
      ElMessage.error(formatApiError(error));
    }
  }
}
</script>
