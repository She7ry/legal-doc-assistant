from __future__ import annotations

from doc_assistant.schemas.citation import Citation
from doc_assistant.services.evidence import build_evidence_profile


def _citation(source_id: str = "S1", text: str = "") -> Citation:
    return Citation(
        source_id=source_id,
        file_name="contract.pdf",
        preview=text,
        exact_quote=text,
    )


def test_evidence_profile_marks_direct_support_when_claim_matches_cited_text() -> None:
    profile = build_evidence_profile(
        "Payment is due within 30 days after invoice approval [S1].",
        [_citation(text="Payment is due within 30 days after invoice approval.")],
    )

    claim = profile["claims"][0]
    assert claim["support_level"] == "direct"
    assert claim["needs_human_review"] is False
    assert claim["unsupported_facts"] == []


def test_evidence_profile_marks_partial_when_cited_text_lacks_specific_fact() -> None:
    profile = build_evidence_profile(
        "Payment is due within 45 days after invoice approval [S1].",
        [_citation(text="Payment is due within 30 days after invoice approval.")],
    )

    claim = profile["claims"][0]
    assert claim["support_level"] == "partial"
    assert claim["needs_human_review"] is True
    assert claim["unsupported_facts"] == ["45 days"]
    assert "45 days" in claim["uncertainty"]


def test_evidence_profile_marks_missing_when_material_claim_has_no_citation() -> None:
    profile = build_evidence_profile(
        "The contract requires arbitration in New York.",
        [_citation(text="Payment is due within 30 days after invoice approval.")],
    )

    assert profile["claims"][0]["support_level"] == "missing"
    assert profile["unsupported_claims"] == ["The contract requires arbitration in New York."]
    assert profile["missing_evidence"]
