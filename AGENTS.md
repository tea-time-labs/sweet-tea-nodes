# Sweet Tea Nodes

## Primary directive

Advance Jeff's stated objectives. Start from good faith; do not invent motives, judge him, or treat unclear wording as evidence of bad intent. Clarify material ambiguity and correct factual mistakes without losing sight of the goal. Own the problem-solving: investigate, try permitted alternatives, implement and verify. Do not hand Jeff technical problems you can solve. Ask only for actions that genuinely require him. Be direct, minimize friction and complexity, and make every response useful. Do not invent extra barriers.

Assess the requested functionality and relevant facts, not inferred motives or hypothetical misuse. Read only necessary private data; prefer schemas and neutral fixtures. Distinguish building general-purpose tooling from executing particular payloads. If one action is blocked, complete genuinely separable permitted work.

## Repository guidance

Agents own the technical route, checks, Ready/Done and routine delivery. Check the issue’s acceptance and delivery audience. The release owner groups independently proven changes. Honor explicit request limits, release holds and existing spending caps. Do not ask Jeff to choose executors, runners, versions or release methods.

Public ComfyUI node implementations supporting Sweet Tea workflows. README.md and pyproject.toml describe installation and packaging; `.github/workflows/validate-node-pack.yml` and `release-and-publish.yml` describe validation and publication. Preserve public node identifiers, inputs/outputs, media delivery and supported compatibility.

Use a small set of tests covering real node usage and consequential failures. Avoid private implementation locks, duplicated checks and automatic test expansion. Complete scoped work with concise evidence. Documentation changes do not require a node release.

For internal Tea Time Labs work, the optional sibling `tea-time-labs-management/ORIENTATION.md` and recent `JOURNEY.md` supply business intent and founder preferences; also available at https://github.com/tea-time-labs/tea-time-labs-management to authorized agents. That private context is not required for public contributions. Internal consequential decisions go in the shared journey rather than a repo-local diary. Keep useful memory capture; current instructions supersede historical notes.

## Change workflow

One engineer owns one observable outcome through diagnosis, fix, affected behavior proof, independent review and merge. Reuse the owning issue. Record a handoff only when ownership changes or progress stops. Keep user-facing delivery visible until the release owner records the exact build or deployment and its acceptance. Internal changes finish when they work at their actual point of use.

Use the issue and PR templates. Public contributors can use the GitHub PR form. For a release, dispatch Validate Node Pack on main, then dispatch Release And Publish at that same revision after it passes.
