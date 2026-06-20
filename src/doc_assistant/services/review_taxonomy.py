"""条款审查与冲突检查的分类体系：ClauseProfile、风险规则、prompt 片段生成。

``resolve_clause_profile`` 将用户输入的条款类型映射到预置 taxonomy；
``qa_service.review_clause`` / ``check_conflict`` 依赖此处配置检索词与风险权重。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any


@dataclass(frozen=True)
class ClauseProfile:
    """一种合同条款类型的审查「配置包」。

    用途：用户说「审查 indemnity 条款」时，通过 aliases/query_terms 扩展检索 query，
    并把 high/medium/low_risk_rules 注入 LLM prompt，让模型按统一标准打风险分。
    """

    key: str
    label: str
    aliases: tuple[str, ...]
    query_terms: tuple[str, ...]
    high_risk_rules: tuple[str, ...]
    medium_risk_rules: tuple[str, ...]
    low_risk_rules: tuple[str, ...]
    risk_weights: tuple[tuple[str, float], ...] = (
        ("high", 3.0),
        ("medium", 1.5),
        ("low", 0.0),
    )

    def expanded_query(self, requested_clause_type: str) -> str:
        terms = [requested_clause_type, self.label, *self.aliases, *self.query_terms]
        return " ".join(_dedupe_terms(terms))

    def risk_rules_prompt(self) -> str:
        return "\n".join(
            [
                f"Clause type: {self.label}",
                "High risk indicators:",
                *[f"- {rule}" for rule in self.high_risk_rules],
                "Medium risk indicators:",
                *[f"- {rule}" for rule in self.medium_risk_rules],
                "Low risk indicators:",
                *[f"- {rule}" for rule in self.low_risk_rules],
                "Risk scoring weights:",
                *[f"- {name}: {weight:g}" for name, weight in self.risk_weights],
            ]
        )


@dataclass(frozen=True)
class ConflictType:
    """合同与政策/另一文档冲突检查的类型定义（如 obligation_conflict、definition_mismatch）。"""

    key: str
    label: str
    description: str


CLAUSE_PROFILES: tuple[ClauseProfile, ...] = (
    ClauseProfile(
        key="termination",
        label="Termination",
        aliases=("termination clause", "terminate", "early cancellation", "notice period", "终止条款", "解除", "提前终止", "通知期限"),
        query_terms=("end agreement", "terminate for convenience", "material breach", "终止合同", "解除合同", "重大违约", "提前解除"),
        high_risk_rules=(
            "Only one party can terminate for convenience.",
            "No clear right to terminate after material breach.",
            "Termination fees or continued payment duties appear unusually burdensome.",
        ),
        medium_risk_rules=(
            "Notice period, notice method, refund, or survival obligations are unclear.",
            "Termination rights depend on missing schedules, definitions, or approval steps.",
        ),
        low_risk_rules=(
            "Termination rights are balanced and include clear notice requirements.",
            "Consequences of termination are clear and tied to cited text.",
        ),
    ),
    ClauseProfile(
        key="payment",
        label="Payment",
        aliases=("payment terms", "fees", "invoice", "billing", "付款条款", "费用", "发票", "账单"),
        query_terms=("due date", "payment obligation", "invoice dispute", "付款期限", "付款义务", "付款条件", "发票争议"),
        high_risk_rules=(
            "Payment is accelerated or due on a very short deadline.",
            "User must pay disputed, unknown, or open-ended amounts.",
        ),
        medium_risk_rules=(
            "Taxes, expenses, invoice disputes, or payment method are unclear.",
            "Payment timing depends on undefined acceptance or approval events.",
        ),
        low_risk_rules=("Payment amounts, timing, dispute process, and taxes are clear.",),
    ),
    ClauseProfile(
        key="late_fee",
        label="Late fee",
        aliases=("late fee", "late payment", "interest", "penalty", "逾期付款", "滞纳金", "违约金", "罚息"),
        query_terms=("overdue payment", "finance charge", "default interest", "逾期费用", "逾期利息", "付款宽限期"),
        high_risk_rules=(
            "Late fees, default interest, or penalties are open-ended or unusually high.",
            "Late payment triggers suspension, termination, or acceleration without cure rights.",
        ),
        medium_risk_rules=("Late fee calculation, grace period, or cure process is unclear.",),
        low_risk_rules=("Late fee amount and cure process are clear and proportionate.",),
    ),
    ClauseProfile(
        key="auto_renewal",
        label="Auto-renewal",
        aliases=("auto renewal", "automatic renewal", "renewal", "evergreen", "自动续约", "续期", "自动延期"),
        query_terms=("renewal term", "cancellation window", "non-renewal notice", "续约期限", "取消窗口", "不续约通知"),
        high_risk_rules=(
            "Agreement renews automatically without a clear cancellation path.",
            "Cancellation window is easy to miss or requires long advance notice.",
        ),
        medium_risk_rules=("Renewal term, notice deadline, or cancellation method is unclear.",),
        low_risk_rules=("Renewal and non-renewal steps are clear and practical.",),
    ),
    ClauseProfile(
        key="liability_limitation",
        label="Liability limitation",
        aliases=("limitation of liability", "liability cap", "damages cap", "责任限制", "责任上限", "赔偿上限"),
        query_terms=("excluded damages", "consequential damages", "cap on liability", "间接损失", "责任封顶", "除外责任"),
        high_risk_rules=(
            "Liability cap may block recovery for major breach scenarios.",
            "Important carve-outs such as fraud, confidentiality, or data security are absent.",
        ),
        medium_risk_rules=("Cap amount, excluded damages, or carve-outs are ambiguous.",),
        low_risk_rules=("Cap, exclusions, and carve-outs are clear and balanced.",),
    ),
    ClauseProfile(
        key="indemnification",
        label="Indemnification",
        aliases=("indemnity", "indemnification", "hold harmless", "defend", "赔偿", "补偿", "抗辩", "使免受损害"),
        query_terms=("third-party claim", "defense obligation", "losses", "第三方索赔", "抗辩义务", "损失赔偿"),
        high_risk_rules=(
            "Indemnity is one-sided, broad, or covers the other party's misconduct.",
            "Defense or settlement control creates material exposure.",
        ),
        medium_risk_rules=("Procedure, covered claims, or exclusions are unclear.",),
        low_risk_rules=("Indemnity scope, procedure, and exclusions are clear.",),
    ),
    ClauseProfile(
        key="confidentiality",
        label="Confidentiality",
        aliases=("confidentiality", "confidential information", "non-disclosure", "nda", "保密", "保密信息", "不披露"),
        query_terms=("disclosure", "return or destroy", "survival", "披露", "返还或销毁", "保密期限", "例外"),
        high_risk_rules=(
            "Confidentiality duties are one-sided, indefinite, or missing key exceptions.",
            "Disclosure obligations may conflict with legal, audit, or business needs.",
        ),
        medium_risk_rules=("Definition, permitted disclosures, or survival period is unclear.",),
        low_risk_rules=("Scope, exceptions, permitted disclosures, and survival are clear.",),
    ),
    ClauseProfile(
        key="non_compete",
        label="Non-compete",
        aliases=("non-compete", "non compete", "non-solicit", "restrictive covenant", "竞业限制", "竞业禁止", "禁止招揽", "限制性约定"),
        query_terms=("competition restriction", "solicitation", "territory", "竞争限制", "招揽客户", "地域范围", "限制期限"),
        high_risk_rules=(
            "Restriction broadly limits work, customers, territory, or future business.",
            "Duration, geography, or covered activities may be excessive or unclear.",
        ),
        medium_risk_rules=("Scope, duration, or affected parties need attorney review.",),
        low_risk_rules=("Restriction is narrow, clearly defined, and tied to legitimate interests.",),
    ),
    ClauseProfile(
        key="ip_ownership",
        label="IP ownership",
        aliases=("ip ownership", "intellectual property", "work product", "license", "知识产权归属", "知识产权", "成果归属", "许可"),
        query_terms=("background IP", "foreground IP", "assignment", "deliverables", "背景知识产权", "前景知识产权", "权利转让", "交付成果"),
        high_risk_rules=(
            "Ownership transfer is broad or could capture pre-existing intellectual property.",
            "License rights are missing, perpetual, or broader than expected.",
        ),
        medium_risk_rules=("Background IP, deliverables, or license scope is unclear.",),
        low_risk_rules=("Ownership, license, and background IP carve-outs are clear.",),
    ),
    ClauseProfile(
        key="data_privacy",
        label="Data privacy",
        aliases=("data privacy", "personal data", "data protection", "security", "数据隐私", "个人信息", "数据保护", "信息安全"),
        query_terms=("processing", "breach notice", "subprocessor", "data transfer", "数据处理", "泄露通知", "分包处理者", "跨境传输"),
        high_risk_rules=(
            "Data use, transfer, security, or breach notice duties are broad or incomplete.",
            "Subprocessor, deletion, audit, or compliance obligations are missing.",
        ),
        medium_risk_rules=("Roles, data categories, retention, or security standard is unclear.",),
        low_risk_rules=("Processing scope, safeguards, retention, and incident duties are clear.",),
    ),
    ClauseProfile(
        key="governing_law",
        label="Governing law",
        aliases=("governing law", "choice of law", "applicable law", "适用法律", "管辖法律", "法律适用"),
        query_terms=("jurisdiction", "venue", "forum", "司法管辖", "法院", "争议管辖", "管辖地"),
        high_risk_rules=(
            "Chosen law or forum materially disadvantages the user or conflicts with operations.",
        ),
        medium_risk_rules=("Law, forum, venue, or priority with other dispute clauses is unclear.",),
        low_risk_rules=("Governing law and forum are clear and expected.",),
    ),
    ClauseProfile(
        key="dispute_resolution",
        label="Dispute resolution",
        aliases=("dispute resolution", "arbitration", "litigation", "venue", "争议解决", "仲裁", "诉讼", "管辖地"),
        query_terms=("mediation", "class waiver", "injunctive relief", "调解", "集体诉讼弃权", "禁令救济", "争议程序"),
        high_risk_rules=(
            "Mandatory arbitration, forum, waiver, or cost shifting may limit practical remedies.",
        ),
        medium_risk_rules=("Escalation steps, venue, costs, or emergency remedies are unclear.",),
        low_risk_rules=("Dispute process, venue, costs, and exceptions are clear.",),
    ),
    ClauseProfile(
        key="assignment",
        label="Assignment",
        aliases=("assignment", "transfer", "change of control", "转让", "合同转让", "控制权变更"),
        query_terms=("delegate", "successor", "affiliate", "转委托", "继受方", "关联方", "权利义务转让"),
        high_risk_rules=(
            "Other party may assign freely while user cannot, or consent rights are absent.",
            "Change of control treatment is missing for a sensitive relationship.",
        ),
        medium_risk_rules=("Consent standard, affiliate transfer, or successor obligations are unclear.",),
        low_risk_rules=("Assignment rights and consent process are clear and balanced.",),
    ),
    ClauseProfile(
        key="audit_rights",
        label="Audit rights",
        aliases=("audit rights", "inspection", "records", "compliance audit", "审计权", "检查权", "记录", "合规审计"),
        query_terms=("access to records", "audit notice", "remediation", "查阅记录", "审计通知", "整改", "审计频率"),
        high_risk_rules=(
            "Audit access is broad, frequent, costly, or lacks confidentiality limits.",
        ),
        medium_risk_rules=("Scope, notice, cost allocation, or remediation process is unclear.",),
        low_risk_rules=("Audit scope, frequency, notice, and confidentiality controls are clear.",),
    ),
    ClauseProfile(
        key="notice",
        label="Notice",
        aliases=("notice", "notices", "written notice", "email notice", "通知", "书面通知", "电子邮件通知"),
        query_terms=("delivery", "deemed received", "address for notices", "送达", "视为收到", "通知地址", "通知方式"),
        high_risk_rules=(
            "Notice method or deemed receipt could cause missed deadlines or default.",
        ),
        medium_risk_rules=("Address, delivery method, deemed receipt, or update process is unclear.",),
        low_risk_rules=("Notice method, address, and receipt timing are clear.",),
    ),
)


CONFLICT_TYPES: tuple[ConflictType, ...] = (
    ConflictType(
        key="direct_contradiction",
        label="Direct contradiction",
        description="One text permits or requires something the other forbids.",
    ),
    ConflictType(
        key="scope_mismatch",
        label="Scope mismatch",
        description="The texts cover different parties, products, territories, data, or obligations.",
    ),
    ConflictType(
        key="deadline_mismatch",
        label="Deadline mismatch",
        description="Dates, notice periods, renewal periods, retention periods, or response times differ.",
    ),
    ConflictType(
        key="amount_mismatch",
        label="Amount mismatch",
        description="Fees, penalties, caps, thresholds, or payment amounts differ.",
    ),
    ConflictType(
        key="definition_mismatch",
        label="Definition mismatch",
        description="The same or related term appears to have different meanings.",
    ),
    ConflictType(
        key="missing_exception",
        label="Missing exception",
        description="One text includes an exception, carve-out, or condition absent from the other.",
    ),
    ConflictType(
        key="process_mismatch",
        label="Process mismatch",
        description="Approval, notice, audit, escalation, or internal workflow steps differ.",
    ),
    ConflictType(
        key="ambiguous_relationship",
        label="Ambiguous relationship",
        description="The excerpts may be compatible, but priority or order of control is unclear.",
    ),
    ConflictType(
        key="none",
        label="None",
        description="No conflict is supported by the provided excerpts.",
    ),
)


def _load_external_clause_profiles() -> tuple[ClauseProfile, ...] | None:
    path = os.getenv("DOC_ASSISTANT_CLAUSE_PROFILES_PATH", "").strip()
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        raw_profiles = json.load(handle)
    if not isinstance(raw_profiles, list):
        raise ValueError("DOC_ASSISTANT_CLAUSE_PROFILES_PATH must contain a JSON list.")
    return tuple(_profile_from_mapping(item) for item in raw_profiles)


def _profile_from_mapping(data: Any) -> ClauseProfile:
    if not isinstance(data, dict):
        raise ValueError("Each clause profile must be a JSON object.")
    risk_rules = data.get("risk_rules") or {}
    if not isinstance(risk_rules, dict):
        raise ValueError("clause profile risk_rules must be an object.")
    return ClauseProfile(
        key=str(data["key"]),
        label=str(data.get("label") or data["key"]),
        aliases=_string_tuple(data.get("aliases")),
        query_terms=_string_tuple(data.get("query_terms")),
        high_risk_rules=_string_tuple(risk_rules.get("high") or data.get("high_risk_rules")),
        medium_risk_rules=_string_tuple(risk_rules.get("medium") or data.get("medium_risk_rules")),
        low_risk_rules=_string_tuple(risk_rules.get("low") or data.get("low_risk_rules")),
        risk_weights=_risk_weights_tuple(data.get("risk_weights")),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("clause profile list fields must be arrays.")
    return tuple(str(item) for item in value if str(item).strip())


def _risk_weights_tuple(value: Any) -> tuple[tuple[str, float], ...]:
    if value is None:
        return (("high", 3.0), ("medium", 1.5), ("low", 0.0))
    if not isinstance(value, dict):
        raise ValueError("risk_weights must be a JSON object.")
    return tuple((str(name), float(weight)) for name, weight in value.items())


def _build_exact_profile_index(profiles: tuple[ClauseProfile, ...]) -> dict[str, ClauseProfile]:
    result: dict[str, ClauseProfile] = {}
    for profile in profiles:
        for term in (profile.key, profile.label, *profile.aliases):
            normalized = term.strip().casefold()
            if normalized:
                result[normalized] = profile
    return result


CLAUSE_PROFILES = _load_external_clause_profiles() or CLAUSE_PROFILES
_PROFILE_BY_EXACT = _build_exact_profile_index(CLAUSE_PROFILES)


def resolve_clause_profile(clause_type: str) -> ClauseProfile:
    requested = clause_type.strip().casefold()
    if not requested:
        return CLAUSE_PROFILES[0]

    exact_match = _PROFILE_BY_EXACT.get(requested)
    if exact_match is not None:
        return exact_match

    for profile in CLAUSE_PROFILES:
        terms = (profile.key, profile.label, *profile.aliases)
        if any(term.casefold() in requested or requested in term.casefold() for term in terms):
            return profile

    return ClauseProfile(
        key=_slugify_clause_type(clause_type),
        label=clause_type.strip(),
        aliases=(),
        query_terms=(),
        high_risk_rules=(
            "The excerpt creates severe consequences, broad waiver, broad liability, or major compliance exposure.",
        ),
        medium_risk_rules=(
            "The excerpt creates obligations, costs, deadlines, ambiguity, or negotiation points.",
        ),
        low_risk_rules=("The excerpt is present, clear, balanced, and tied to cited text.",),
    )


def clause_taxonomy_prompt() -> str:
    return "\n".join(f"- {profile.key}: {profile.label}" for profile in CLAUSE_PROFILES)


def conflict_types_prompt() -> str:
    return "\n".join(
        f"- {conflict_type.key}: {conflict_type.label}. {conflict_type.description}"
        for conflict_type in CONFLICT_TYPES
    )


def allowed_conflict_type_keys() -> set[str]:
    return {conflict_type.key for conflict_type in CONFLICT_TYPES}


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        normalized = term.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


def _slugify_clause_type(clause_type: str) -> str:
    slug = "_".join(part for part in clause_type.strip().lower().replace("-", " ").split() if part)
    return slug[:80] or "custom"
