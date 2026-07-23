from __future__ import annotations

from ai.rag.grounding.evidence import build_evidence_profile
from ai.rag.schemas import Citation


def _citation(source_id: str = "S1", text: str = "") -> Citation:
    return Citation(
        source_id=source_id,
        file_name="contract.pdf",
        preview=text,
        exact_quote=text,
    )


def test_evidence_profile_marks_direct_support_when_claim_matches_cited_text() -> None:
    profile = build_evidence_profile(
        "发票审批后应在30日内付款 [S1]。",
        [_citation(text="发票审批后应在30日内付款。")],
    )

    claim = profile["claims"][0]
    assert claim["support_level"] == "direct"
    assert claim["needs_human_review"] is False
    assert claim["unsupported_facts"] == []


def test_evidence_profile_marks_partial_when_cited_text_lacks_specific_fact() -> None:
    profile = build_evidence_profile(
        "发票审批后应在45日内付款 [S1]。",
        [_citation(text="发票审批后应在30日内付款。")],
    )

    claim = profile["claims"][0]
    assert claim["support_level"] == "partial"
    assert claim["needs_human_review"] is True
    assert claim["unsupported_facts"] == ["45日"]
    assert "45日" in claim["uncertainty"]


def test_evidence_profile_marks_missing_when_material_claim_has_no_citation() -> None:
    profile = build_evidence_profile(
        "合同约定争议应提交北京仲裁委员会仲裁。",
        [_citation(text="发票审批后应在30日内付款。")],
    )

    assert profile["claims"][0]["support_level"] == "missing"
    assert profile["unsupported_claims"] == ["合同约定争议应提交北京仲裁委员会仲裁。"]
    assert profile["missing_evidence"]


def test_evidence_profile_audits_short_chinese_claims_and_currency() -> None:
    profile = build_evidence_profile(
        "合同价款为人民币100万元 [S1]。",
        [_citation(text="合同价款为人民币80万元。")],
    )

    assert profile["claims"][0]["support_level"] == "partial"
    assert profile["claims"][0]["unsupported_facts"] == ["人民币100万元"]
