"""ReAct Agent request helpers."""

from __future__ import annotations

import re

from doc_assistant.agent._helpers import _clean_text, _mentions_any

GENERIC_OBJECTIVE_PATTERNS = {
    "帮我看看",
    "帮我看下",
    "看看合同",
    "看一下合同",
    "看下合同",
    "审查合同",
    "审核合同",
    "看看文件",
    "看下文件",
    "reviewthis",
    "reviewcontract",
    "checkthis",
    "checkcontract",
    "analyzethis",
}
DEADLINE_KEYWORDS = (
    "urgent",
    "asap",
    "deadline",
    "due",
    "expire",
    "expires",
    "expiring",
    "紧急",
    "尽快",
    "截止",
    "期限",
    "到期",
    "马上",
)
CURRENT_LAW_KEYWORDS = (
    "current law",
    "latest law",
    "up-to-date law",
    "statute",
    "regulation",
    "compliance",
    "legal requirement",
    "legal authority",
    "is this legal",
    "现行法律",
    "最新法律",
    "法规",
    "监管",
    "合规",
    "法条",
    "法律依据",
    "是否合法",
    "法定",
)
JURISDICTION_INDICATORS = (
    "new york",
    "california",
    "delaware",
    "united states",
    "u.s.",
    "usa",
    "china",
    "prc",
    "hong kong",
    "singapore",
    "england",
    "wales",
    "eu",
    "european union",
    "中国",
    "美国",
    "英国",
    "香港",
    "新加坡",
    "纽约",
    "加州",
    "特拉华",
    "内地",
    "大陆",
    "欧盟",
    "北京",
    "上海",
    "广东",
    "深圳",
    "江苏",
    "浙江",
)
PARTY_SENSITIVE_KEYWORDS = (
    "lawsuit",
    "sued",
    "court case",
    "arbitration demand",
    "claim against",
    "eviction",
    "employment termination",
    "被起诉",
    "起诉",
    "法院案件",
    "劳动仲裁",
    "被辞退",
    "解雇",
    "驱逐",
    "退租纠纷",
    "合同纠纷",
)
PARTY_SIDE_INDICATORS = (
    "plaintiff",
    "defendant",
    "buyer",
    "seller",
    "customer",
    "vendor",
    "employer",
    "employee",
    "landlord",
    "tenant",
    "甲方",
    "乙方",
    "原告",
    "被告",
    "买方",
    "卖方",
    "客户",
    "供应商",
    "雇主",
    "员工",
    "房东",
    "租客",
)
DATE_OR_TIME_PATTERN = re.compile(
    r"(\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
    r"\b\d{1,2}[-/.]\d{1,2}\b|"
    r"\d{4}年\d{1,2}月\d{1,2}日|"
    r"\d+\s*(business\s+days?|days?|hours?|weeks?|工作日|小时|天|日|周)|"
    r"today|tomorrow|next\s+week|今天|明天|后天|本周|下周|月底|月末)",
    re.IGNORECASE,
)


def _looks_underspecified_objective(text: str, focus_areas: list[str]) -> bool:
    normalized = re.sub(r"[\s，。,.!?！？、：:；;]+", "", text.casefold())
    if not normalized:
        return True
    if normalized in GENERIC_OBJECTIVE_PATTERNS:
        return True
    if normalized.startswith(("帮我看看这份", "帮我看看这个", "帮我看下这份")) and len(
        normalized
    ) <= 10:
        return True
    return not focus_areas and len(text) < 12


def clarification_questions_for_task(
    objective: str,
    focus_areas: list[str] | None = None,
) -> list[str]:
    """Return up to three blocking clarification questions for underspecified tasks."""
    text = _clean_text(objective)
    lowered = text.casefold()
    normalized_focus = [_clean_text(area) for area in focus_areas or [] if _clean_text(area)]
    questions: list[str] = []

    if _looks_underspecified_objective(text, normalized_focus):
        questions.append(
            "请说明你希望 Agent 完成的具体任务，例如风险审查、条款解释、谈判清单或律师问题清单。"
        )

    if _mentions_any(lowered, DEADLINE_KEYWORDS) and not DATE_OR_TIME_PATTERN.search(text):
        questions.append("请补充具体截止时间或希望在什么日期前完成处理。")

    if _mentions_any(lowered, CURRENT_LAW_KEYWORDS) and not _mentions_any(
        lowered,
        JURISDICTION_INDICATORS,
    ):
        questions.append("请补充适用法域或地域，例如国家、州/省、市，或合同中的 governing law。")

    if _mentions_any(lowered, PARTY_SENSITIVE_KEYWORDS) and not _mentions_any(
        lowered,
        PARTY_SIDE_INDICATORS,
    ):
        questions.append(
            "请说明你在事项中的立场或代表方，例如甲方/乙方、买方/卖方、雇主/员工、房东/租客。"
        )

    return questions[:3]
