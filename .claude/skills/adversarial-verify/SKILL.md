---
name: adversarial-verify
description: Adversarially verify any claim, finding, or conclusion the agent produces. Before reporting "done", this skill forces a skeptical second look. Use whenever the agent makes claims about code correctness, bug existence, or task completion.
user-invocable: true
---

# Adversarial Verification

Before declaring any task complete, run an adversarial verification pass — try to PROVE YOURSELF WRONG.

## When to Apply

Trigger this skill after:
- Claiming a bug is fixed
- Asserting code is correct
- Generating a code review finding
- Producing any analysis result
- Claiming a task is "done"

## The 11 Shortcuts Agents Take (check yourself against each)

1. **Plausible-but-wrong** — The answer sounds right but isn't. Verify against actual code/runtime.
2. **Hallucinated evidence** — Citing a file/function that doesn't exist. Double-check every reference.
3. **Surface-only fix** — Fixed the symptom, not the root cause. Ask: "What else could trigger this?"
4. **Untested claim** — "This should work." Run it. Import it. Test it.
5. **Assumed context** — Inferring intent without checking. Read the actual code, don't assume.
6. **Silent failure** — The code runs but produces wrong output. Check return values, not just exit codes.
7. **Scope creep blindness** — Fixed one thing, broke another. Run related tests.
8. **Confirmation bias** — Only seeing evidence that supports the conclusion. Actively seek contradictory evidence.
9. **Copy-paste drift** — Code from memory that doesn't match the actual codebase. Always re-read.
10. **Premature closure** — Stopping at the first working solution. Is there a simpler/better approach?
11. **Quantitative hand-waving** — "Significant improvement" without numbers. Demand specific measurements.

## Process

1. State the claim clearly
2. For each claim: "What would prove this WRONG?"
3. Actively search for that disconfirming evidence
4. Run the code / test the claim
5. If claim survives: report with evidence
6. If claim fails: correct and re-verify
