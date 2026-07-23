from langchain_core.prompts import PromptTemplate

from ai.prompts import load_prompt


def test_prompt_templates_keep_runtime_contracts() -> None:
    expected_variables = {
        "base_legal_assistant.txt": set(),
        "document_qa.txt": {"user_memory", "chat_history", "question", "context"},
        "general_chat.txt": {"user_memory", "chat_history", "question"},
        "tool_calling_system.txt": set(),
        "clause_review.txt": {
            "clause_type",
            "normalized_clause_type",
            "clause_taxonomy",
            "risk_rules",
            "context",
        },
        "conflict_check.txt": {
            "contract_context",
            "policy_context",
            "conflict_types",
        },
        "answer_repair.txt": {"issues", "source_ids", "answer"},
    }

    for name, variables in expected_variables.items():
        template = PromptTemplate.from_template(load_prompt(name))
        assert set(template.input_variables) == variables
        assert template.format(**dict.fromkeys(variables, "测试内容")).strip()
