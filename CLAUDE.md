# CLAUDE.md — Project Rules

## Mandatory after every task

### 1. Update plan.md

After completing any subtask, immediately update [plan.md](plan.md):
- Mark the finished subgoal with `[x]` (e.g. `### [x] 1.2 Element-wise ops`)
- When all subgoals in a phase are done, mark the phase header with `✅` (e.g. `## Phase 1 — Dual-Number Primitives ✅`)
- Update the Milestones table at the bottom: change `[ ]` to `[x]` for the completed milestone

Never batch these updates — mark done the moment a subtask is complete.

### 2. Create a changelog entry

After every session that touches code or docs, create a new file in [log/](log/) named `YYYY-MM-DD-<short-slug>.md`. Use today's actual date (available in context as `currentDate`).

Format:

```markdown
# YYYY-MM-DD — <Short title>

## What changed
- bullet list of files created or modified
- one line per change, state what and why

## Plan status
Phase N subgoals completed: X.Y, X.Z
Next: Phase N+1 — <name>
```

Do not summarise what you plan to do — only record what was actually done.
