from __future__ import annotations

from ai.rag.grounding.guard import validate_answer
from ai.rag.schemas import Citation


def _citation(source_id: str = "S1", preview: str = "甲方应在30日内付款。") -> Citation:
    return Citation(
        source_id=source_id,
        file_name="contract.pdf",
        preview=preview,
    )


def test_validate_answer_passes_when_citations_are_valid() -> None:
    answer = (
        "## 简短结论\n"
        "甲方应在30日内付款 [S1]。置信度：高。\n\n"
        "## 文档依据\n"
        "第三条约定甲方应在30日内付款 [S1]。"
    )

    result = validate_answer(answer, [_citation()], has_retrieved_documents=True)

    assert result.passed is True
    assert result.confidence == "High"
    assert result.needs_repair is False


def test_validate_answer_flags_unknown_citation_ids() -> None:
    answer = "通知期限为10日 [S9]。"

    result = validate_answer(answer, [_citation()], has_retrieved_documents=True)

    assert result.passed is False
    assert any("S9" in issue for issue in result.issues)
    assert result.needs_repair is True


def test_validate_answer_flags_strong_legal_conclusions() -> None:
    answer = "该条款无效，且一定会胜诉 [S1]。"

    result = validate_answer(answer, [_citation()], has_retrieved_documents=True)

    assert result.passed is False
    assert any("strong legal conclusion" in issue for issue in result.issues)


def test_validate_answer_requires_refusal_without_retrieved_documents() -> None:
    answer = "合同约定争议应提交北京仲裁委员会仲裁。"

    result = validate_answer(answer, [], has_retrieved_documents=False)

    assert result.passed is False
    assert any("without retrieved documents" in issue for issue in result.issues)


def test_validate_answer_handles_chinese_safety_language() -> None:
    citation = _citation(preview="甲方应当按合同约定付款。")

    strong = validate_answer("因此该条款无效 [S1]。", [citation])
    missing_fact = validate_answer(
        "合同价款为人民币100万元。\n\n其他事项见合同 [S1]。",
        [citation],
        verify_citation_semantics=False,
    )
    valid_refusal = validate_answer("索引文档未找到相关内容。", [], has_retrieved_documents=False)
    inverted_refusal = validate_answer("未发现明显缺失信息。", [], has_retrieved_documents=False)

    assert any("strong legal conclusion" in issue for issue in strong.issues)
    assert any("人民币100万元" in issue for issue in missing_fact.issues)
    assert valid_refusal.passed is True
    assert inverted_refusal.passed is False
