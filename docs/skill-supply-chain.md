# Runtime Skill supply-chain policy

External skills are dependencies, not trusted prompts. An imported skill must have a lock entry containing its repository URL, exact commit, relative source path, SHA-256 for every imported file, license identifier, review date, and evaluation report. The runtime allowlist remains separate from `SKILL.md`.

## Admission checks

1. Confirm that the skill targets a business agent rather than a coding assistant.
2. Review all instructions and resources for Claude Code assumptions, prompt injection, scripts, network access, shell commands, and dynamic code execution.
3. Confirm that the license permits modification and redistribution of the specific skill.
4. Import from an exact commit and verify file hashes from `skills.lock.json`.
5. Keep skill scripts disabled. Reimplement required deterministic behavior in tested application code and expose only registered tools.
6. Run the fixed-dataset no-skill/with-skill A/B evaluation. Reject a skill that does not improve a target metric or add a separately tested capability.

## Review on 2026-07-02

- `anthropics/skills`: not imported. The repository mixes Apache-2.0 examples with source-available document skills, and the available catalog is centered on document creation, creative work, and development tooling rather than evidence-grounded business RAG.
- `langchain-ai/langchain-skills`: not imported. The repository describes itself as early development and its published skills teach LangChain, LangGraph, and Deep Agents development; they are coding/framework skills, not the four business-agent RAG controls required here.

No external skill passed the relevance and evaluation gates, so `skills.lock.json` intentionally has an empty dependency list.
