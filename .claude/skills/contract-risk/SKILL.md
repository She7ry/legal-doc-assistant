---
name: contract-risk
description: Structured contract risk assessment and clause-level analysis. Use when the user wants to assess risks in a contract, compare clauses across documents, or detect conflicts between a contract and a policy/regulation.
user-invocable: true
argument-hint: [contract name or specific clause]
---

# Contract Risk Assessment

Structured risk analysis for contracts using the legal document assistant's review services (`review_service.py`, `clause_review.py`, `conflict_check.py`).

**Disclaimer**: This identifies potential issues for discussion with a qualified lawyer. It does not provide legal advice.

## Risk Taxonomy

The project defines clause risk categories in `src/doc_assistant/services/review_taxonomy.py`.

### Risk Severity Scale

| Level | Criteria | Example |
|-------|----------|---------|
| **High** 🔴 | Uncapped liability, one-sided termination, overbroad IP grab, mandatory arbitration in inconvenient venue | "Party A may terminate at any time without cause; Party B requires 180 days notice" |
| **Medium** 🟡 | Asymmetric but bounded obligation, unclear definitions, missing standard protections | "Confidentiality survives for 10 years" (unusually long) |
| **Low** 🟢 | Minor ambiguities, slightly unfavorable but market-standard terms | "Payment within 45 days" (slightly longer than market 30 days) |

### Risk Categories

| Category | Description | Key Clauses to Check |
|----------|-------------|---------------------|
| **Liability & Indemnity** | Who bears what risk | Limitation of liability, indemnification scope, carve-outs |
| **Term & Termination** | Exit rights and costs | Termination for convenience, notice periods, survival, post-termination obligations |
| **Payment & Fees** | Financial obligations | Pricing structure, late fees, tax responsibility, expense allocation |
| **IP & Data Rights** | Who owns what | IP assignment, license grants, work product ownership, data usage rights |
| **Confidentiality** | Information protection | Definition scope, duration, exclusions, return/destroy obligations |
| **Non-Compete / Non-Solicit** | Competitive restrictions | Scope, duration, geographic reach, enforceability concerns |
| **Dispute Resolution** | How conflicts are resolved | Arbitration vs litigation, venue, prevailing party fees, class action waiver |
| **Assignment & Change of Control** | Transferability | Assignment restrictions, change of control triggers, successor binding |
| **Representations & Warranties** | Promises about facts | Scope, survival period, knowledge qualifiers, materiality thresholds |
| **Force Majeure** | Excused non-performance | Covered events, notice requirements, mitigation obligations |

## Analysis Workflows

### Single Contract Risk Scan

Use the `/review/clause` endpoint to check a specific clause:

```
POST /api/v1/review/clause
{
  "clause_text": "<text>",
  "clause_type": "indemnification|limitation_of_liability|termination|confidentiality|..."
}
```

Returns: risk level, concerns list, citations, and plain-language explanation.

Or through the Python service:

```bash
python -c "
from doc_assistant.services.review_service import ReviewService
# Clause-level risk assessment
result = review_service.review_clause(clause_text='...', clause_type='limitation_of_liability')
print(f'Risk: {result.risk_level}')
print(f'Concerns: {result.concerns}')
print(f'Plain language: {result.plain_language}')
"
```

### Cross-Document Conflict Check

When comparing a contract against a policy, regulation, or another contract:

```
POST /api/v1/review/conflict
{
  "document_a": "<contract text or reference>",
  "document_b": "<policy text or reference>"
}
```

Returns: conflicting clauses, gaps, and recommended actions.

### Full Contract Risk Matrix

For a comprehensive risk assessment:

1. **Ingest** the contract through the document upload endpoint
2. **Run Pass 1**: Identify all clauses by type (use the review taxonomy)
3. **Run Pass 2**: Score each clause for risk
4. **Run Pass 3**: Cross-reference against relevant policies/regulations if loaded
5. **Generate** the risk matrix (see output format below)

## Output: Risk Matrix

```markdown
# Contract Risk Assessment: [Contract Title]

## Risk Summary

| Severity | Count |
|----------|-------|
| 🔴 High | N |
| 🟡 Medium | N |
| 🟢 Low | N |

## Risk Matrix

| # | Clause | Category | Risk | Issue | Citation | Recommendation |
|---|--------|----------|------|-------|----------|----------------|
| 1 | §4.2 | Liability | 🔴 High | Uncapped indemnification | [S4] | Add liability cap |
| 2 | §7.1 | Termination | 🟡 Medium | Asymmetric notice (30 vs 90 days) | [S7] | Equalize notice periods |
| ... | ... | ... | ... | ... | ... | ... |

## Top 5 Recommended Changes

1. [Highest priority change with rationale]
2. ...

## Questions for Opposing Party / Lawyer

1. [Q1]
2. [Q2]

## Sources

[All citations used]
```

## Anti-Patterns

- ❌ Saying "this is illegal" — say "this may conflict with [law/regulation], consult a lawyer"
- ❌ Recommending specific legal positions without qualification
- ❌ Making risk claims without citing the specific clause text
- ❌ Ignoring jurisdiction — laws vary; flag when jurisdiction is unclear
- ❌ Treating all risks equally — prioritize by severity and practical impact
