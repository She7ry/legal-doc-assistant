---
name: legal-review
description: Systematic legal document review workflow. Use when the user wants to review a contract, policy, lease, privacy agreement, or any legal document through the RAG system. Covers document analysis, risk identification, key clause extraction, and generating structured findings.
user-invocable: true
argument-hint: [document name or review focus]
---

# Legal Document Review

A systematic multi-pass review workflow for legal documents using the Legal Document Assistant's RAG capabilities.

This skill is NOT a substitute for legal advice. All findings should cite source text and flag uncertainty.

## Document Types & Review Focus

| Document Type | Primary Focus | Secondary Focus |
|---------------|---------------|-----------------|
| **Contract** | Obligations, liabilities, termination, dispute resolution | Payment terms, IP, confidentiality |
| **Policy / 政策** | Compliance requirements, penalties, effective dates | Reporting obligations, exceptions |
| **Lease / 租赁** | Rent terms, maintenance obligations, termination notice periods | Sublease restrictions, renewal options |
| **Privacy / 隐私** | Data collection scope, third-party sharing, user rights | Retention periods, breach notification |
| **NDA** | Definition of confidential info, duration, exclusions | Residuals clause, return obligations |
| **Employment / 劳动** | Compensation, non-compete scope, termination grounds | IP assignment, benefits eligibility |

## Review Passes

### Pass 1: Structural Overview

Scan for the document's architecture:

1. **Identify the document type** and governing law clause
2. **List all section/article headings** — what's included and what's conspicuously ABSENT
3. **Identify the parties** and their roles
4. **Note key dates**: effective date, expiration, renewal windows, notice periods

Query approach:

```
"List all major sections and articles in this document"
"What is the governing law and jurisdiction?"
"Who are the parties and what are their roles?"
"What are the key dates: effective date, term, renewal conditions?"
```

### Pass 2: Obligation Extraction

For each party, extract what they MUST do, MUST NOT do, and MAY do:

| Party | Must Do | Must Not Do | May Do (at discretion) |
|-------|---------|-------------|------------------------|
| Party A | ... | ... | ... |
| Party B | ... | ... | ... |

Query approach:

```
"For [Party A], what are all obligations, prohibitions, and discretionary rights?"
"What happens if [Party A] fails to meet these obligations?"
"Which obligations have explicit deadlines or time requirements?"
```

### Pass 3: Risk Identification

Score each risk area on a 3-tier scale:

| Risk Area | What to Check |
|-----------|---------------|
| **Liability** | Uncapped liability, indemnification scope, liquidated damages |
| **Termination** | Without-cause termination, asymmetric termination rights, survival clauses |
| **IP** | Overbroad IP assignment, weak IP protection, ownership ambiguity |
| **Confidentiality** 保密 | Too-narrow definition, too-long duration, inadequate return/destroy obligations |
| **Payment** | Unclear payment triggers, unfavorable payment terms, hidden costs |
| **Dispute Resolution** 争议解决 | Unfavorable venue, waiver of jury trial, arbitration cost allocation |
| **Force Majeure** 不可抗力 | Too-narrow or too-broad definition, notice requirements |
| **Assignment/Change of Control** | Ability to assign without consent, change of control triggers |
| **Compliance/Regulatory** | Missing regulatory references, ambiguous compliance standards |

For each risk found, record:

```
Risk: [one-line description]
Severity: High / Medium / Low
Clause: [source citation S#]
Reasoning: [why this is a concern]
Recommendation: [concrete suggested action]
```

### Pass 4: Ambiguity Detection

Find language that could be interpreted multiple ways:

- **Vague standards**: "reasonable efforts" / "合理努力", "material adverse change" / "重大不利变化", "promptly" / "及时"
- **Undefined terms**: capitalized terms without definitions
- **Conflicting clauses**: two sections that say different things about the same topic
- **Silent gaps**: important topics not addressed at all

Query approach:

```
"Find all instances of vague or undefined standards like 'reasonable', 'material', 'promptly'"
"Are there any conflicting or contradictory clauses?"
"What important topics are NOT addressed in this document?"
```

### Pass 5: Compliance Cross-Reference

If multiple documents are loaded, check for consistency:

```
"Compare the confidentiality obligations across all loaded documents"
"Are there any contradictions between the main contract and its exhibits/attachments?"
```

## Output: Structured Review Report

```markdown
# Legal Document Review: [Document Title]

**Document Type**: [type]
**Parties**: [list]
**Governing Law**: [jurisdiction]
**Review Date**: [date]
**Disclaimer**: This is not legal advice. Consult a qualified lawyer.

## 1. Executive Summary
[3-5 sentence overview of the document, its purpose, and top risks]

## 2. Key Terms Summary
| Term | Detail | Citation |
|------|--------|----------|
| ... | ... | [S#] |

## 3. Obligations Matrix
[Table from Pass 2]

## 4. Risk Findings
[Each risk from Pass 3 with severity and recommendations]

## 5. Ambiguities & Gaps
[Findings from Pass 4]

## 6. Recommended Questions for Lawyer
- [Question 1]
- [Question 2]

## 7. Sources
[Complete list of citations used]
```
