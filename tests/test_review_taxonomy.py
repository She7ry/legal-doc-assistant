from __future__ import annotations

from ai.review.taxonomy import resolve_clause_profile


def test_resolve_clause_profile_matches_chinese_alias() -> None:
    profile = resolve_clause_profile("终止条款")

    assert profile.key == "termination"
    assert "终止合同" in profile.expanded_query("终止条款")
    assert "高风险判断依据：" in profile.risk_rules_prompt()
