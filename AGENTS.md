# AGENTS.md

Operating instructions for AI agents working in this repository.

## Conventions
- Conventional Commits for every commit message (`feat:`, `fix:`, `chore:`, `docs:`).
- Never push to the default branch. Every change goes through a pull request.
- Keep diffs minimal and reviewable. One concern per PR.
- Link every PR to its issue (`Fixes #N` / `Relates to #N`). CI enforces this.

## Verification
- Run lint + tests before opening a PR. A green CI is required for merge.
- Do not merge your own PRs.

## Agent surfaces
- GitHub Discussions categories `agent-lounge`, `agent-blockers`, `agent-brainstorms`
  are the brainstorming surface. Issues labeled `agent-talk` are mirrored there automatically.
- `.github/agent-coffee/COFFEE.md` defines the coffee-break deliberation ritual
  (casual agent-to-agent brainstorming with an adversarial critic persona).
- `jev/` is the cost/latency-aware model router: prefer local small LMs, then free
  tiers, escalate only when the task needs it.
