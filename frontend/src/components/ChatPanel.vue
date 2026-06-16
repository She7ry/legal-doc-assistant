<template>
  <section class="tool-panel chat-panel">
    <div class="panel-heading">
      <div>
        <h2>文档问答</h2>
        <p>基于已索引内容生成带引用的回答</p>
      </div>
      <div class="chat-panel-actions">
        <el-select
          v-model="conversationId"
          class="conversation-select"
          size="small"
          filterable
          :disabled="loading"
          @change="switchConversation"
        >
          <el-option
            v-for="conversation in displayedConversations"
            :key="conversation.conversation_id"
            :label="conversationLabel(conversation)"
            :value="conversation.conversation_id"
          />
        </el-select>
        <el-tooltip content="新建会话" placement="top">
          <el-button :icon="Plus" :disabled="loading" @click="startConversation" />
        </el-tooltip>
        <el-tooltip content="归档会话" placement="top">
          <el-button
            :icon="FolderRemove"
            :disabled="loading || !conversationId"
            @click="archiveConversation"
          />
        </el-tooltip>
        <el-button :icon="Delete" :disabled="!messages.length || loading" @click="clearMessages">
          清空
        </el-button>
      </div>
    </div>

    <div ref="messageListRef" class="message-list">
      <el-empty v-if="!messages.length" :image-size="112" description="暂无对话" />

      <article
        v-for="message in messages"
        :key="message.id"
        class="message-row"
        :class="`message-row--${message.role}`"
      >
        <div class="message-avatar">{{ message.role === "user" ? "问" : "答" }}</div>
        <div class="message-body">
          <div class="message-content">{{ message.content }}</div>
          <div
            v-if="message.role === 'assistant' && (message.confidence || message.guardWarnings.length)"
            class="answer-trust"
          >
            <el-tag v-if="message.confidence" :type="confidenceTagType(message.confidence)" effect="dark">
              可信度 {{ message.confidence }}
            </el-tag>
            <el-alert
              v-if="message.guardWarnings.length"
              type="warning"
              title="回答需要谨慎使用"
              :closable="false"
              show-icon
            >
              <ul class="structured-list structured-list--compact">
                <li v-for="warning in message.guardWarnings" :key="warning">
                  {{ warning }}
                </li>
              </ul>
            </el-alert>
          </div>
          <EvidencePanel
            v-if="message.role === 'assistant' && message.evidence"
            :evidence="message.evidence"
          />
          <CitationList
            v-if="message.role === 'assistant' && message.citations.length"
            :citations="message.citations"
          />
          <div
            v-if="message.role === 'assistant' && message.memoriesUsed.length"
            class="memory-usage-list"
          >
            <span>Memory used</span>
            <el-tag
              v-for="memory in message.memoriesUsed"
              :key="memory.memory_id"
              effect="plain"
              size="small"
            >
              {{ memory.type }}: {{ memory.key }}
            </el-tag>
          </div>
        </div>
      </article>
    </div>

    <div class="chat-composer">
      <el-input
        v-model="question"
        type="textarea"
        :rows="3"
        maxlength="2000"
        show-word-limit
        resize="none"
        placeholder="输入合同、政策或合规问题"
        @keydown.enter.exact.prevent="send"
      />
      <el-button
        type="primary"
        :icon="Promotion"
        :loading="loading"
        :disabled="!question.trim()"
        @click="send"
      >
        发送
      </el-button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { Delete, FolderRemove, Plus, Promotion } from "@element-plus/icons-vue";

import {
  askQuestionStream,
  createChatConversation,
  fetchConversationMessages,
  listChatConversations,
  updateChatConversation,
} from "../api/chat";
import { formatApiError } from "../api/http";
import type {
  ChatHistoryMessage,
  Citation,
  ConversationRecord,
  EvidenceProfile,
  MemoryUsage,
} from "../api/types";
import CitationList from "./CitationList.vue";
import EvidencePanel from "./EvidencePanel.vue";

interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  memoriesUsed: MemoryUsage[];
  confidence: string | null;
  guardWarnings: string[];
  evidence: EvidenceProfile | null;
}

const CONVERSATION_STORAGE_KEY = "legal-doc-assistant.conversationId";
const CHAT_HISTORY_WINDOW = 12;

const messages = ref<UiMessage[]>([]);
const question = ref("");
const loading = ref(false);
const messageListRef = ref<HTMLElement | null>(null);
const conversationId = ref(readConversationId());
const conversations = ref<ConversationRecord[]>([]);

const displayedConversations = computed(() => {
  if (conversations.value.some((conversation) => conversation.conversation_id === conversationId.value)) {
    return conversations.value;
  }
  return [
    {
      conversation_id: conversationId.value,
      title: null,
      status: "active",
      created_at: "",
      updated_at: "",
      message_count: messages.value.length,
    },
    ...conversations.value,
  ];
});

onMounted(async () => {
  await loadConversations();
  await restoreConversation();
});

async function send() {
  const text = question.value.trim();
  if (!text || loading.value) {
    return;
  }

  const history: ChatHistoryMessage[] = messages.value.slice(-CHAT_HISTORY_WINDOW).map((message) => ({
    role: message.role,
    content: message.content,
  }));

  messages.value.push({
    id: crypto.randomUUID(),
    role: "user",
    content: text,
    citations: [],
    memoriesUsed: [],
    confidence: null,
    guardWarnings: [],
    evidence: null,
  });
  const assistantId = crypto.randomUUID();
  messages.value.push({
    id: assistantId,
    role: "assistant",
    content: "",
    citations: [],
    memoriesUsed: [],
    confidence: null,
    guardWarnings: [],
    evidence: null,
  });
  question.value = "";
  loading.value = true;
  await scrollToBottom();

  try {
    await askQuestionStream(
      {
        question: text,
        chat_history: history,
        conversation_id: conversationId.value,
      },
      {
        onMetadata(metadata) {
          const message = findMessage(assistantId);
          if (message) {
            message.citations = metadata.citations;
            message.memoriesUsed = metadata.memories_used ?? [];
          }
        },
        onDelta(delta) {
          const message = findMessage(assistantId);
          if (message) {
            message.content += delta;
          }
          void scrollToBottom();
        },
        onDone(answer) {
          const message = findMessage(assistantId);
          if (message) {
            message.content = answer.content;
            message.citations = answer.citations;
            message.memoriesUsed = answer.memories_used ?? [];
            message.confidence = answer.confidence ?? null;
            message.guardWarnings = answer.guard_warnings ?? [];
            message.evidence = answer.evidence ?? null;
          }
        },
      },
    );
    await scrollToBottom();
  } catch (error) {
    const assistantMessage = findMessage(assistantId);
    if (!assistantMessage?.content) {
      messages.value = messages.value.filter((message) => message.id !== assistantId);
    }
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
    void loadConversations();
  }
}

function findMessage(id: string): UiMessage | undefined {
  return messages.value.find((message) => message.id === id);
}

async function clearMessages() {
  await startConversation();
}

async function loadConversations() {
  try {
    const response = await listChatConversations({ status: "active", limit: 50 });
    conversations.value = response.conversations;
  } catch (error) {
    console.warn("Failed to load conversations.", error);
  }
}

async function switchConversation(nextConversationId: string | number) {
  const nextId = String(nextConversationId || "");
  if (!nextId || loading.value) {
    return;
  }
  setConversationId(nextId);
  messages.value = [];
  await restoreConversation();
}

async function startConversation() {
  if (loading.value) {
    return;
  }
  try {
    const conversation = await createChatConversation();
    setConversationId(conversation.conversation_id);
    conversations.value = [
      conversation,
      ...conversations.value.filter(
        (item) => item.conversation_id !== conversation.conversation_id,
      ),
    ];
    messages.value = [];
  } catch (error) {
    ElMessage.error(formatApiError(error));
  }
}

async function archiveConversation() {
  if (!conversationId.value || loading.value) {
    return;
  }
  const archivedId = conversationId.value;
  try {
    await updateChatConversation(archivedId, { status: "archived" });
    conversations.value = conversations.value.filter(
      (conversation) => conversation.conversation_id !== archivedId,
    );
    messages.value = [];
    const nextConversation = conversations.value[0] ?? (await createChatConversation());
    if (!conversations.value.length) {
      conversations.value = [nextConversation];
    }
    setConversationId(nextConversation.conversation_id);
    await restoreConversation();
    ElMessage.success("会话已归档");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  }
}

function conversationLabel(conversation: ConversationRecord): string {
  const title = conversation.title?.trim() || `会话 ${conversation.conversation_id.slice(0, 8)}`;
  const count = conversation.message_count ? ` (${conversation.message_count})` : "";
  return `${title}${count}`;
}

async function restoreConversation() {
  if (!conversationId.value || messages.value.length) {
    return;
  }
  try {
    const response = await fetchConversationMessages(conversationId.value);
    messages.value = response.messages.map((message) => ({
      id: crypto.randomUUID(),
      role: message.role,
      content: message.content,
      citations: [],
      memoriesUsed: [],
      confidence: null,
      guardWarnings: [],
      evidence: null,
    }));
    await scrollToBottom();
  } catch (error) {
    console.warn("Failed to restore conversation.", error);
  }
}

function confidenceTagType(confidence: string) {
  if (confidence === "High") {
    return "success";
  }
  if (confidence === "Medium") {
    return "warning";
  }
  if (confidence === "Low") {
    return "danger";
  }
  return "info";
}

async function scrollToBottom() {
  await nextTick();
  if (messageListRef.value) {
    messageListRef.value.scrollTop = messageListRef.value.scrollHeight;
  }
}

function readConversationId(): string {
  const stored = localStorage.getItem(CONVERSATION_STORAGE_KEY);
  if (stored) {
    return stored;
  }
  return createConversationId();
}

function createConversationId(): string {
  const id = crypto.randomUUID();
  localStorage.setItem(CONVERSATION_STORAGE_KEY, id);
  return id;
}

function setConversationId(id: string) {
  conversationId.value = id;
  localStorage.setItem(CONVERSATION_STORAGE_KEY, id);
}
</script>
