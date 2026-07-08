---
name: grounded-rag-answer
description: Produce evidence-grounded RAG and knowledge-base answers that separate supported facts, bounded inferences, and unknowns. Use for document question answering, legal or financial analysis, customer-support retrieval, citation-first responses, and any task where claims must stay within retrieved evidence. 用于基于检索证据回答、区分事实推论与未知信息，并在证据不足时拒绝推断。
---

# Ground an answer in retrieved evidence

1. Inventory the retrieved sources before drafting.
2. State only claims supported by those sources. Attach the supporting source identifier to each material claim.
3. Label a conclusion as an inference when it combines source facts; state the assumptions and cite every premise.
4. Preserve qualifiers, scope, dates, parties, thresholds, exceptions, and uncertainty from the source.
5. Treat user memory and conversation context as context, not documentary evidence.
6. State what is unknown or missing. Do not fill gaps with general knowledge when the task requires document evidence.
7. Prefer a narrow refusal or a request for the missing material over an unsupported answer.

Read `references/evidence-policy.md` only when evaluating conflicting sources, negative claims, or partial support.
