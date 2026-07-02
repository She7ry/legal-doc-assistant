---
name: clean-commits
description: Create clean, conventional commits with meaningful messages. Use when the user asks to commit changes, create a PR, or clean up git history.
user-invocable: true
argument-hint: [commit message or "squash"]
---

# Clean Commits

Write conventional commit messages and produce clean git history for the legal document assistant project.

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]
[optional footer(s)]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature (API endpoint, UI component, service capability) |
| `fix` | Bug fix |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or modifying tests |
| `docs` | Documentation changes |
| `chore` | Tooling, CI, deps, config changes |
| `style` | Formatting, whitespace (no logic change) |
| `revert` | Reverting a previous commit |

### Scopes (project-specific)

| Scope | Area |
|-------|------|
| `api` | FastAPI routes, middleware, dependencies |
| `qa` | Q&A service, answer guard, evidence |
| `agent` | Agent service, planner, executor, workflow |
| `memory` | User memory system |
| `retrieval` | Vector store, BM25, chunking, search |
| `ingestion` | Document loading, upload |
| `review` | Clause review, conflict check |
| `matter` | Matter store, export |
| `config` | Settings, env vars |
| `frontend` | Vue app |
| `eval` | RAG evaluation |
| `prompts` | Prompt templates |

### Examples

```
feat(api): add streaming SSE endpoint for agent task events
fix(retrieval): handle empty BM25 index in hybrid search RRF fusion
refactor(memory): extract LLM-based extraction into separate module
test(qa): add regression test for citation validation edge case
chore(config): bump langgraph dependency to 1.0+
```

## Before Committing

1. Run the secret scan: check for API keys in staged files
2. Run relevant tests: `python -m pytest tests/ -v --lf` (at minimum last-failed)
3. Check the diff for debugging artifacts: `git diff --cached | grep -E '(print\(|console\.log|pdb\.set_trace|breakpoint\(\))'`

## Commit Messages

- **Subject line**: 50 chars max, imperative mood ("Add" not "Added"), lowercase, no period at end
- **Body**: wrap at 72 chars, explain WHY not WHAT
- **Footer**: reference issues, breaking changes with `BREAKING CHANGE:` prefix

## Squash Before PR

When a branch has many "wip" or "fix typo" commits, squash them:

```bash
git rebase -i HEAD~N  # squash fixup commits into meaningful units
```
