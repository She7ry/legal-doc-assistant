"""条款审查与冲突检查的分类体系：ClauseProfile、风险规则、prompt 片段生成。

``resolve_clause_profile`` 将用户输入的条款类型映射到预置 taxonomy；
``qa_service.review_clause`` / ``check_conflict`` 依赖此处配置检索词与风险权重。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClauseProfile:
    """一种合同条款类型的审查「配置包」。

    用途：用户指定条款类型时，通过 aliases/query_terms 扩展检索 query，
    并把 high/medium/low_risk_rules 注入 LLM prompt，让模型按统一标准打风险分。
    """

    key: str
    label: str
    aliases: tuple[str, ...]
    query_terms: tuple[str, ...]
    high_risk_rules: tuple[str, ...]
    medium_risk_rules: tuple[str, ...]
    low_risk_rules: tuple[str, ...]
    def expanded_query(self, requested_clause_type: str) -> str:
        del requested_clause_type
        terms = [self.label, *self.aliases, *self.query_terms]
        return " ".join(_dedupe_terms(terms))

    def risk_rules_prompt(self) -> str:
        return "\n".join(
            [
                f"条款类型：{self.label}",
                "高风险判断依据：",
                *[f"- {rule}" for rule in self.high_risk_rules],
                "中风险判断依据：",
                *[f"- {rule}" for rule in self.medium_risk_rules],
                "低风险判断依据：",
                *[f"- {rule}" for rule in self.low_risk_rules],
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
        label="终止",
        aliases=("终止条款", "解除", "提前终止", "通知期限"),
        query_terms=("终止合同", "解除合同", "重大违约", "提前解除"),
        high_risk_rules=(
            "只有一方享有任意终止权。",
            "发生重大违约后没有明确的终止权。",
            "终止费用或终止后的持续付款义务可能明显过重。",
        ),
        medium_risk_rules=(
            "通知期限、通知方式、退款或终止后存续义务不明确。",
            "终止权取决于当前缺失的附件、定义或审批步骤。",
        ),
        low_risk_rules=(
            "双方终止权较为平衡，且通知要求明确。",
            "终止后果明确，并有对应来源文本支持。",
        ),
    ),
    ClauseProfile(
        key="payment",
        label="付款",
        aliases=("付款条款", "费用", "发票", "账单"),
        query_terms=("付款期限", "付款义务", "付款条件", "发票争议"),
        high_risk_rules=(
            "付款义务被加速到期，或付款期限过短。",
            "用户可能需要支付存在争议、金额未知或没有上限的款项。",
        ),
        medium_risk_rules=(
            "税费、费用、发票争议处理或付款方式不明确。",
            "付款时间取决于未定义的验收或审批事件。",
        ),
        low_risk_rules=("付款金额、时间、争议处理流程和税费承担均明确。",),
    ),
    ClauseProfile(
        key="late_fee",
        label="逾期费用",
        aliases=("逾期付款", "滞纳金", "违约金", "罚息"),
        query_terms=("逾期费用", "逾期利息", "付款宽限期"),
        high_risk_rules=(
            "逾期费用、违约利息或罚款没有上限或明显过高。",
            "逾期付款会触发暂停履行、终止或加速到期，且没有补救权。",
        ),
        medium_risk_rules=("逾期费用计算方式、宽限期或补救流程不明确。",),
        low_risk_rules=("逾期费用金额和补救流程明确且相称。",),
    ),
    ClauseProfile(
        key="auto_renewal",
        label="自动续约",
        aliases=("自动续约", "续期", "自动延期"),
        query_terms=("续约期限", "取消窗口", "不续约通知"),
        high_risk_rules=(
            "协议自动续约，但没有明确的取消途径。",
            "取消窗口容易错过，或要求过长的提前通知期。",
        ),
        medium_risk_rules=("续约期限、通知截止日或取消方式不明确。",),
        low_risk_rules=("续约和不续约步骤明确且具有可操作性。",),
    ),
    ClauseProfile(
        key="liability_limitation",
        label="责任限制",
        aliases=("责任限制", "责任上限", "赔偿上限"),
        query_terms=("间接损失", "责任封顶", "除外责任"),
        high_risk_rules=(
            "责任上限可能妨碍用户就重大违约获得充分救济。",
            "缺少欺诈、保密或数据安全等重要除外情形。",
        ),
        medium_risk_rules=("责任上限金额、排除的损失类型或除外情形存在歧义。",),
        low_risk_rules=("责任上限、责任排除和除外情形明确且较为平衡。",),
    ),
    ClauseProfile(
        key="indemnification",
        label="赔偿",
        aliases=("赔偿", "补偿", "抗辩", "使免受损害"),
        query_terms=("第三方索赔", "抗辩义务", "损失赔偿"),
        high_risk_rules=(
            "赔偿义务单向、范围过宽，或涵盖对方自身的不当行为。",
            "抗辩或和解控制权可能造成重大风险敞口。",
        ),
        medium_risk_rules=("赔偿程序、涵盖的索赔或除外情形不明确。",),
        low_risk_rules=("赔偿范围、程序和除外情形明确。",),
    ),
    ClauseProfile(
        key="confidentiality",
        label="保密",
        aliases=("保密", "保密信息", "不披露"),
        query_terms=("披露", "返还或销毁", "保密期限", "例外"),
        high_risk_rules=(
            "保密义务单向、期限无限，或缺少关键例外。",
            "披露限制可能与法定义务、审计或业务需要冲突。",
        ),
        medium_risk_rules=("保密信息定义、允许披露情形或存续期限不明确。",),
        low_risk_rules=("保密范围、例外、允许披露情形和存续期限均明确。",),
    ),
    ClauseProfile(
        key="non_compete",
        label="竞业限制",
        aliases=("竞业限制", "竞业禁止", "禁止招揽", "限制性约定"),
        query_terms=("竞争限制", "招揽客户", "地域范围", "限制期限"),
        high_risk_rules=(
            "限制广泛影响工作、客户、地域或未来业务。",
            "期限、地域或受限活动范围可能过宽或不明确。",
        ),
        medium_risk_rules=("限制范围、期限或受影响主体需要律师审阅。",),
        low_risk_rules=("限制范围较窄、定义明确，并与正当利益相关。",),
    ),
    ClauseProfile(
        key="ip_ownership",
        label="知识产权归属",
        aliases=("知识产权归属", "知识产权", "成果归属", "许可"),
        query_terms=("背景知识产权", "前景知识产权", "权利转让", "交付成果"),
        high_risk_rules=(
            "权利转让范围过宽，或可能涵盖既有知识产权。",
            "许可权缺失、永久有效或范围超出合理预期。",
        ),
        medium_risk_rules=("背景知识产权、交付成果或许可范围不明确。",),
        low_risk_rules=("权利归属、许可和背景知识产权除外情形均明确。",),
    ),
    ClauseProfile(
        key="data_privacy",
        label="数据隐私",
        aliases=("数据隐私", "个人信息", "数据保护", "信息安全"),
        query_terms=("数据处理", "泄露通知", "分包处理者", "跨境传输"),
        high_risk_rules=(
            "数据使用、传输、安全或泄露通知义务范围过宽或不完整。",
            "缺少分包处理、删除、审计或合规义务。",
        ),
        medium_risk_rules=("主体角色、数据类别、保存期限或安全标准不明确。",),
        low_risk_rules=("处理范围、安全措施、保存期限和事件响应义务均明确。",),
    ),
    ClauseProfile(
        key="governing_law",
        label="准据法",
        aliases=("适用法律", "管辖法律", "法律适用"),
        query_terms=("司法管辖", "法院", "争议管辖", "管辖地"),
        high_risk_rules=(
            "约定的法律或争议解决地可能使用户明显不利，或与实际运营冲突。",
        ),
        medium_risk_rules=("准据法、法院、地点或与其他争议条款的优先关系不明确。",),
        low_risk_rules=("准据法和争议解决地明确且符合预期。",),
    ),
    ClauseProfile(
        key="dispute_resolution",
        label="争议解决",
        aliases=("争议解决", "仲裁", "诉讼", "管辖地"),
        query_terms=("调解", "集体诉讼弃权", "禁令救济", "争议程序"),
        high_risk_rules=(
            "强制仲裁、管辖地、权利放弃或费用转移可能限制实际救济。",
        ),
        medium_risk_rules=("升级处理步骤、地点、费用或紧急救济不明确。",),
        low_risk_rules=("争议处理流程、地点、费用和例外均明确。",),
    ),
    ClauseProfile(
        key="assignment",
        label="转让",
        aliases=("转让", "合同转让", "控制权变更"),
        query_terms=("转委托", "继受方", "关联方", "权利义务转让"),
        high_risk_rules=(
            "对方可自由转让而用户不能，或用户缺少同意权。",
            "对于敏感合作关系，条款缺少控制权变更安排。",
        ),
        medium_risk_rules=("同意标准、关联方转让或继受方义务不明确。",),
        low_risk_rules=("转让权和同意流程明确且较为平衡。",),
    ),
    ClauseProfile(
        key="audit_rights",
        label="审计权",
        aliases=("审计权", "检查权", "记录", "合规审计"),
        query_terms=("查阅记录", "审计通知", "整改", "审计频率"),
        high_risk_rules=(
            "审计访问范围过宽、频率过高、成本过重，或缺少保密限制。",
        ),
        medium_risk_rules=("审计范围、通知、费用承担或整改流程不明确。",),
        low_risk_rules=("审计范围、频率、通知和保密控制均明确。",),
    ),
    ClauseProfile(
        key="notice",
        label="通知",
        aliases=("通知", "书面通知", "电子邮件通知"),
        query_terms=("送达", "视为收到", "通知地址", "通知方式"),
        high_risk_rules=(
            "通知方式或视为送达规则可能导致错过期限或构成违约。",
        ),
        medium_risk_rules=("通知地址、送达方式、视为收到时间或更新流程不明确。",),
        low_risk_rules=("通知方式、地址和收到时间均明确。",),
    ),
)


CONFLICT_TYPES: tuple[ConflictType, ...] = (
    ConflictType(
        key="direct_contradiction",
        label="直接矛盾",
        description="一份文本允许或要求的事项被另一份文本禁止。",
    ),
    ConflictType(
        key="scope_mismatch",
        label="范围不一致",
        description="两份文本涵盖的主体、产品、地域、数据或义务范围不同。",
    ),
    ConflictType(
        key="deadline_mismatch",
        label="期限不一致",
        description="日期、通知期、续约期、保存期或响应时间不同。",
    ),
    ConflictType(
        key="amount_mismatch",
        label="金额不一致",
        description="费用、罚款、上限、阈值或付款金额不同。",
    ),
    ConflictType(
        key="definition_mismatch",
        label="定义不一致",
        description="相同或相关术语在两份文本中的含义似乎不同。",
    ),
    ConflictType(
        key="missing_exception",
        label="缺少例外",
        description="一份文本包含另一份文本没有的例外、除外情形或条件。",
    ),
    ConflictType(
        key="process_mismatch",
        label="流程不一致",
        description="审批、通知、审计、升级处理或内部流程步骤不同。",
    ),
    ConflictType(
        key="ambiguous_relationship",
        label="关系不明确",
        description="两处摘录可能并不矛盾，但优先级或适用顺序不明确。",
    ),
    ConflictType(
        key="none",
        label="无冲突",
        description="所提供的摘录没有证据支持存在冲突。",
    ),
)


def _build_exact_profile_index(profiles: tuple[ClauseProfile, ...]) -> dict[str, ClauseProfile]:
    result: dict[str, ClauseProfile] = {}
    for profile in profiles:
        for term in (profile.key, profile.label, *profile.aliases):
            normalized = term.strip().casefold()
            if normalized:
                result[normalized] = profile
    return result


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
            "摘录可能造成严重后果、广泛权利放弃、广泛责任或重大合规风险。",
        ),
        medium_risk_rules=(
            "摘录涉及义务、成本、期限、歧义或需要协商的事项。",
        ),
        low_risk_rules=("摘录内容完整、明确、较为平衡，并有对应引用支持。",),
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
