<template>
  <section class="tool-panel chat-panel">
    <div class="panel-heading">
      <div>
        <h2>文档问答</h2>
        <p>基于已索引内容生成带引用的回答</p>
      </div>
      <el-button :icon="Delete" :disabled="!messages.length || loading" @click="clearMessages">
        清空
      </el-button>
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
import { nextTick, ref } from "vue";
import { ElMessage } from "element-plus";
import { Delete, Promotion } from "@element-plus/icons-vue";

import { askQuestionStream } from "../api/chat";
import { formatApiError } from "../api/http";
import type { ChatHistoryMessage, Citation, MemoryUsage } from "../api/types";
import CitationList from "./CitationList.vue";

interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  memoriesUsed: MemoryUsage[];
}

const CONVERSATION_STORAGE_KEY = "legal-doc-assistant.conversationId";

const messages = ref<UiMessage[]>([]);
const question = ref("");
const loading = ref(false);
const messageListRef = ref<HTMLElement | null>(null);
const conversationId = ref(readConversationId());

async function send() {
  const text = question.value.trim();
  if (!text || loading.value) {
    return;
  }

  const history: ChatHistoryMessage[] = messages.value.slice(-8).map((message) => ({
    role: message.role,
    content: message.content,
  }));

  messages.value.push({
    id: crypto.randomUUID(),
    role: "user",
    content: text,
    citations: [],
    memoriesUsed: [],
  });
  const assistantId = crypto.randomUUID();
  messages.value.push({
    id: assistantId,
    role: "assistant",
    content: "",
    citations: [],
    memoriesUsed: [],
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
  }
}

function findMessage(id: string): UiMessage | undefined {
  return messages.value.find((message) => message.id === id);
}

function clearMessages() {
  messages.value = [];
  conversationId.value = createConversationId();
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
</script>
