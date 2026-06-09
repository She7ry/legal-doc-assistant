from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClauseProfile:
    key: str
    label: str
    aliases: tuple[str, ...]
    query_terms: tuple[str, ...]
    high_risk_rules: tuple[str, ...]
    medium_risk_rules: tuple[str, ...]
    low_risk_rules: tuple[str, ...]

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
            ]
        )


@dataclass(frozen=True)
class ConflictType:
    key: str
    label: str
    description: str


CLAUSE_PROFILES: tuple[ClauseProfile, ...] = (
    ClauseProfile(
        key="termination",
        label="Termination",
        aliases=("termination clause", "terminate", "early cancellation", "notice period"),
        query_terms=("end agreement", "terminate for convenience", "material breach"),
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
        aliases=("payment terms", "fees", "invoice", "billing"),
        query_terms=("due date", "payment obligation", "invoice dispute"),
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
        aliases=("late fee", "late payment", "interest", "penalty"),
        query_terms=("overdue payment", "finance charge", "default interest"),
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
        aliases=("auto renewal", "automatic renewal", "renewal", "evergreen"),
        query_terms=("renewal term", "cancellation window", "non-renewal notice"),
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
        aliases=("limitation of liability", "liability cap", "damages cap"),
        query_terms=("excluded damages", "consequential damages", "cap on liability"),
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
        aliases=("indemnity", "indemnification", "hold harmless", "defend"),
        query_terms=("third-party claim", "defense obligation", "losses"),
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
        aliases=("confidentiality", "confidential information", "non-disclosure", "nda"),
        query_terms=("disclosure", "return or destroy", "survival"),
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
        aliases=("non-compete", "non compete", "non-solicit", "restrictive covenant"),
        query_terms=("competition restriction", "solicitation", "territory"),
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
        aliases=("ip ownership", "intellectual property", "work product", "license"),
        query_terms=("background IP", "foreground IP", "assignment", "deliverables"),
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
        aliases=("data privacy", "personal data", "data protection", "security"),
        query_terms=("processing", "breach notice", "subprocessor", "data transfer"),
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
        aliases=("governing law", "choice of law", "applicable law"),
        query_terms=("jurisdiction", "venue", "forum"),
        high_risk_rules=(
            "Chosen law or forum materially disadvantages the user or conflicts with operations.",
        ),
        medium_risk_rules=("Law, forum, venue, or priority with other dispute clauses is unclear.",),
        low_risk_rules=("Governing law and forum are clear and expected.",),
    ),
    ClauseProfile(
        key="dispute_resolution",
        label="Dispute resolution",
        aliases=("dispute resolution", "arbitration", "litigation", "venue"),
        query_terms=("mediation", "class waiver", "injunctive relief"),
        high_risk_rules=(
            "Mandatory arbitration, forum, waiver, or cost shifting may limit practical remedies.",
        ),
        medium_risk_rules=("Escalation steps, venue, costs, or emergency remedies are unclear.",),
        low_risk_rules=("Dispute process, venue, costs, and exceptions are clear.",),
    ),
    ClauseProfile(
        key="assignment",
        label="Assignment",
        aliases=("assignment", "transfer", "change of control"),
        query_terms=("delegate", "successor", "affiliate"),
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
        aliases=("audit rights", "inspection", "records", "compliance audit"),
        query_terms=("access to records", "audit notice", "remediation"),
        high_risk_rules=(
            "Audit access is broad, frequent, costly, or lacks confidentiality limits.",
        ),
        medium_risk_rules=("Scope, notice, cost allocation, or remediation process is unclear.",),
        low_risk_rules=("Audit scope, frequency, notice, and confidentiality controls are clear.",),
    ),
    ClauseProfile(
        key="notice",
        label="Notice",
        aliases=("notice", "notices", "written notice", "email notice"),
        query_terms=("delivery", "deemed received", "address for notices"),
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


def resolve_clause_profile(clause_type: str) -> ClauseProfile:
    requested = clause_type.strip().casefold()
    if not requested:
        return CLAUSE_PROFILES[0]

    for profile in CLAUSE_PROFILES:
        terms = (profile.key, profile.label, *profile.aliases)
        if any(requested == term.casefold() for term in terms):
            return profile

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
