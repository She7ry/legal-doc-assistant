---
name: decompose-retrieval-query
description: Decompose complex, compound, comparative, or multi-hop questions into focused retrieval subqueries, then fuse and deduplicate results. Use before search when a question contains multiple issues, parties, documents, time periods, conditions, comparisons, or requested outputs. 用于把复杂、多条件、跨文档或比较问题拆成多个检索子问题并融合结果。
---

# Decompose a retrieval query

1. Preserve the original question as one retrieval query.
2. Split only independent issues, comparisons, conditions, parties, documents, or time periods.
3. Keep controlling entities, dates, clause names, and legal or domain terms in each subquery that needs them.
4. Produce two to four focused subqueries. Do not answer the question during decomposition.
5. Retrieve each subquery independently.
6. Fuse results by rank, deduplicate identical passages, and retain coverage across subqueries.
7. Stop decomposing when a subquery would be broader or less precise than the original.
