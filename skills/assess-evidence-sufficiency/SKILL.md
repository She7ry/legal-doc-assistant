---
name: assess-evidence-sufficiency
description: Assess retrieved evidence before generation for relevance, coverage, conflicts, missing materials, and support for negative claims. Use for RAG answers, audits, reviews, and high-stakes decisions that must refuse or narrow conclusions when evidence is partial or absent. 用于生成前判断证据是否充分、冲突以及缺少哪些材料，证据不足时限制或拒绝结论。
---

# Assess evidence sufficiency

1. List the material questions or claims the requested answer must resolve.
2. Check whether retrieved text directly addresses each item with the correct scope, party, time, and conditions.
3. Identify contradictions, superseded versions, missing definitions, missing attachments, and absent governing context.
4. Treat corpus-wide negative claims as insufficient unless retrieval coverage makes absence meaningful.
5. Classify the evidence as `sufficient`, `partial`, or `insufficient`.
6. For `partial`, answer only supported portions and list the gaps.
7. For `insufficient`, refuse the unsupported conclusion and specify the material needed to proceed.
