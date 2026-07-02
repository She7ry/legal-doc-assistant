---
name: citation-check
description: Verify that every factual claim in a RAG-generated answer is backed by a valid source citation. This is the core quality mechanism of the legal document assistant. Use after any Q&A response to validate answer quality, or when the user questions an answer's reliability.
user-invocable: true
argument-hint: [answer text or "last answer"]
---

# Citation Verification

Verify that every claim in a RAG-generated legal answer follows the **citation-first** principle.

This skill implements the project's core design rule: every assertion must be traceable to a specific source document chunk with a valid citation ID (`[S1]`, `[S2]`, etc.).

## The Citation Standard

Every answer from the legal document assistant must:

1. **Cite sources** for ALL factual claims, dates, amounts, obligations, and legal references
2. **Use valid IDs** — `[S1]`, `[S2]`, etc. that correspond to actual retrieved chunks
3. **Not invent facts** — no claim should appear without a source
4. **Not overstate** — no guarantees of legal outcome, no definitive "this is illegal" without qualification
5. **Maintain distinction** — user memories are context, NOT evidence; they should not carry `[S#]` citations

## Verification Process

### Step 1: Extract All Claims

From the answer, extract every factual assertion:

```
Claim 1: "The contract has a 30-day termination notice period"
Claim 2: "Party A is responsible for all shipping costs"
Claim 3: "The governing law is California"
...
```

### Step 2: Map Claims to Citations

For each claim, check:

| Check | Question |
|-------|----------|
| **Cited?** | Does the claim have a `[S#]` tag? |
| **Valid ID?** | Does the `[S#]` ID exist in the retrieved sources? |
| **Supported?** | Does the cited source text actually support this claim? |
| **Accurate?** | Does the claim match the source, or is it distorted/extrapolated? |

### Step 3: Detect Violations

#### Violation: Missing Citation (uncited factual claim)

```
"The tenant must pay rent by the 5th of each month."  ← No [S#] tag
```

**Action**: Find the relevant source, add citation. If no source supports it, remove or qualify the claim.

#### Violation: Invalid Citation ID

```
"...as stated in the agreement [S7]"  ← Only S1-S5 were retrieved
```

**Action**: Cross-reference against `retrieved_source_ids`. Flag for AnswerGuard.

#### Violation: Unsupported Claim (citation doesn't match)

```
Claim: "The penalty is 10% of the contract value [S3]"
Source S3: "Late payment shall incur a penalty of 5% per month"
```

**Action**: Correct the claim or re-assign to the right source.

#### Violation: Strong Legal Conclusion

```
"This contract is unenforceable" ← Definitive legal conclusion
"This violates labor law" ← Not qualified
```

**Action**: Rephrase with qualification:
```
"This contract contains provisions that courts may find unenforceable because... [S#]"
"These terms may conflict with labor law standards regarding... [S#]. Consult a lawyer."
```

#### Violation: Unsourced Authority

```
"Under the Civil Code..." ← No specific article cited
"Labor law requires..." ← Which law? Which section?
```

**Action**: If the specific statute/article is in the source, cite it. If not, remove or qualify.

#### Violation: Memory as Evidence

```
"You mentioned earlier that your contract has a 60-day notice period [M2]"
```
`[M2]` is a memory citation, not a document citation. Memories should be presented as context, not evidence.

### Step 4: Score

For each answer:

```
Citation Coverage = cited_claims / total_claims
Citation Validity = valid_citations / total_citations
Citation Accuracy = accurate_citations / total_citations
Overall Confidence = min(coverage, validity, accuracy)

High: >= 0.9  Medium: 0.7-0.9  Low: < 0.7
```

## Integration with AnswerGuard

This project has a built-in `AnswerGuard` (`src/doc_assistant/services/answer_guard.py`) that automatically:

- Validates citation format and existence
- Flags strong legal conclusions
- Detects unsourced authorities
- Detects unsourced specific facts (dates, amounts, percentages)
- Detects refusals when no documents are retrieved

When invoking this skill, complement (don't duplicate) AnswerGuard's automatic checks by focusing on **semantic accuracy**: does the cited source ACTUALLY say what the answer claims?

## Quick Check (for last answer)

```bash
python -c "
from doc_assistant.services.answer_guard import AnswerGuard
# Re-run guard on the last answer
guard = AnswerGuard()
result = guard.check(answer_text, retrieved_sources)
print(f'Confidence: {result.confidence}')
for issue in result.issues:
    print(f'  [{issue.severity}] {issue.description}')
"
```
