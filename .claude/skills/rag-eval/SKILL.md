---
name: rag-eval
description: Run the RAG evaluation pipeline (retrieval + generation quality scoring). Use when the user wants to benchmark or assess the RAG system.
user-invocable: true
---

# RAG Evaluation

Run the full RAG evaluation pipeline for the legal document assistant.

## Prerequisites

1. `.env` configured with LLM + Embedding API keys
2. `data/eval/eval_dataset.json` exists

## Step

Run evaluation:

```bash
python -m doc_assistant.evaluation.cli run_eval
```

If that fails, try the registered console script:

```bash
run-rag-eval
```

## View Results

```bash
python -c "
import json
with open('data/eval/latest_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)
print(json.dumps(report, ensure_ascii=False, indent=2))
"
```

## Metrics

- **Retrieval**: Recall@5, Hit Rate, Precision, MRR, nDCG
- **Generation**: Answer Correctness, Citation Accuracy, Faithfulness, Refusal Accuracy

## CI Thresholds

- `retrieval.at_5.recall >= 0.8`
- `generation.citation_accuracy >= 0.9`
