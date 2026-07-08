---
name: verify-citation-support
description: Verify citation support after generation by mapping each material claim to cited source text and checking entailment, scope, numbers, dates, parties, qualifiers, and contradictions. Use for RAG, legal, financial, policy, and knowledge-base answers where citation presence alone is not enough. 用于生成后逐条验证引用是否真实支持陈述，而不仅检查引用编号格式。
---

# Verify citation support

1. Split the answer into material claims without separating qualifiers from their claims.
2. Map every claim to the citation identifiers attached to it.
3. Confirm that each cited source supports the complete claim, including numbers, dates, parties, scope, exceptions, and polarity.
4. Mark a citation `partial` when it supports only one part of a compound claim.
5. Mark a claim `unsupported` when citations are missing, unrelated, contradictory, or merely topically similar.
6. Correct or remove unsupported language. Do not repair an unsupported claim by attaching the nearest available citation.
7. Return claim-level issues so the answer guard can repair, narrow, or reject the answer.
