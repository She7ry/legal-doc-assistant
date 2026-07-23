const LEVEL_LABELS: Record<string, string> = {
  High: "高",
  Medium: "中",
  Low: "低",
  "Needs human review": "需要人工审阅",
};

const STATUS_LABELS: Record<string, string> = {
  "No conflict found": "未发现冲突",
  "Potential conflict": "可能存在冲突",
  "Insufficient information": "信息不足",
};

const SOURCE_TYPE_LABELS: Record<string, string> = {
  document: "文档",
  web: "网页",
};

const CONFLICT_TYPE_LABELS: Record<string, string> = {
  direct_contradiction: "直接矛盾",
  scope_mismatch: "范围不一致",
  deadline_mismatch: "期限不一致",
  amount_mismatch: "金额不一致",
  definition_mismatch: "定义不一致",
  missing_exception: "缺少例外",
  process_mismatch: "流程不一致",
  ambiguous_relationship: "关系不明确",
  none: "无冲突",
};

export function levelLabel(value: string) {
  return LEVEL_LABELS[value] || value;
}

export function conflictStatusLabel(value: string) {
  return STATUS_LABELS[value] || value;
}

export function conflictTypeLabel(value: string) {
  return CONFLICT_TYPE_LABELS[value] || value;
}

export function sourceTypeLabel(value?: string | null) {
  return SOURCE_TYPE_LABELS[(value || "document").toLowerCase()] || value || "文档";
}

export function locationLabel(value?: string | null) {
  return (value || "")
    .replace(/page\s+(\d+)/gi, "第$1页")
    .replace(/chunk\s+(\d+)/gi, "片段 $1");
}

export function guardWarningLabel(value: string) {
  if (value === "Answer is empty.") return "回答内容为空。";
  if (value.includes("does not include any source citations")) return "回答未包含来源引用。";
  if (value.includes("source IDs that were not returned")) {
    return `回答引用了检索结果中不存在的来源：${value.match(/\[[A-Z]\d+\]/g)?.join("、") || "未知来源"}。`;
  }
  if (value.includes("material paragraph lacks a source citation")) return "回答中的实质性段落缺少来源引用。";
  if (value.includes("specific fact")) return "回答中的日期、金额或期限缺少对应引用。";
  if (value.includes("without retrieved documents")) return "未检索到相关文档，回答也未明确说明证据不足。";
  if (value.includes("strong legal conclusion")) return "回答包含过度确定的法律结论，应改为审慎表述。";
  if (value.includes("legal authority or statute-like text")) return "回答中的法条或法律依据缺少对应引用。";
  if (value.startsWith("Citation does not support")) return "引用内容可能不足以支持完整陈述。";
  return value;
}
