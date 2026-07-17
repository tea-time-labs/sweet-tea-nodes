# Sweet Tea Codex Entrypoint

- Source of truth for synced `AGENTS.md` files across managed repos.
- Scope: all work under `/home/jkotieno`.
- Canonical policy: `/home/jkotieno/tea-time-labs-management/docs/SWEET_TEA_OS_CANONICAL_CONTRACT.md`.

## First: Sweet Tea Studio MCP Product Operations Are Not Repo/QAS Work
- If the user is asking an agent to operate Sweet Tea Studio through MCP, use Studio MCP directly. Do not run `start`, `exit`, QAS, `ops_v2`, shell log scraping, raw filesystem/database recovery, repo tests, or dispatch unless the user explicitly asks for repo implementation, CI, release, QAS, or code verification.
- A user request to operate Sweet Tea Studio through MCP is a product operation, not repo implementation work. Examples include `context.current`, `context.from_regenerate`, `context.from_use_in_pipe`, `context.map_sources`, `context.clone`, `context.submit`, approvals, media curation, snippet edits, model/node management, project/folder work, and job/status inspection.
- For Studio generation requests, use the Studio MCP contract directly: `context.get` -> one origin (`context.current`, `context.from_regenerate`, `context.from_use_in_pipe`, or `context.map_sources`) -> `context.clone` for edits/variants -> `context.submit`, then surface the Studio approval or job status. Do not stop before submit on the theory that image generation is QAS-owned.
- The finish-line QAS boundary below does not apply to the app owner asking an agent to click or operate Studio through its MCP surface.

## Required Terms
- `start`: mandatory startup gate before repo execution or dispatch-lane work. Use skill `$start` as the authoritative procedure.
- `exit`: workflow close with explicit status and evidence only when the user explicitly says `exit`. Use skill `$exit` as the authoritative procedure.
- `QAS`: an opt-in independent finish-line gate owned by `qa_sentinel`; it is not the default implementation or local-testing path.
- `startup_receipt_id`: required on new `dispatch-task` and `ingest-dispatch` commands.

## Exit Logbook (`exit`)
- Authoritative workflow: skill `$exit` at `/home/jkotieno/.codex/skills/exit/SKILL.md`.
- When the user sends `exit`, append a new UTC-stamped entry to `/home/jkotieno/.codex/session_handoffs/SESSION_LOGBOOK.md` before final response.
- Keep a single running logbook for this trigger (do not split by session).
- Include: timestamp, repo/cwd, comprehensive session summary, final status, verification or handoff evidence, and most likely next action.
- `comprehensive session summary` must capture: (1) the session intent/goal as it ended up, (2) key implementation or decision points, and (3) the resulting outcome/state (including pivots or unresolved items). Do not use one-line activity-only summaries.

## QAS Boundary (Finish-Line Only)
- The primary agent owns the normal engineering loop end to end: diagnosis, implementation, local execution, UI/log/exit-code inspection, focused tests, and fix/verify iteration.
- Those normal activities do not activate QAS and do not require a structured handoff.
- Activate QAS only when the user explicitly requests `QAS`/`qa_sentinel`, or when the task explicitly calls for an independent pre-release, pre-deployment, release/deploy-readiness, publication, or comparable finish-line gate.
- Do not infer QAS activation from generic requests to test, verify, run code, inspect UI/logs, execute CI, finish, complete, or work end to end.
- If the activation criteria are absent, do not dispatch or hand off to QAS. The primary agent remains responsible for running relevant checks, inspecting their output, fixing defects, and reporting exact evidence.
- Once an activated QAS handoff is accepted, `qa_sentinel` owns only the requested final-gate scope and any bounded fix loop authorized by that handoff.
- QAS may commit, push, trigger CI, release, or deploy only when the user explicitly requested that delivery scope.
- During an explicitly requested CI/release/deploy closure, QAS may make a bounded CI-unblock commit only for unrelated or clearly preexisting red-main breakage, and only when the unblock is necessary, minimal, orthogonal to the requested lane behavior, and reported separately.

## QAS Tooling Rule
- Only after QAS is activated under the boundary above, use `ops_v2` QAS routing commands and keep that final-gate ownership in `qa_sentinel`.
- Do not use native sub-agent orchestration (`spawn_agent`, `send_input`, `wait`, `close_agent`) as a substitute inside an active QAS-owned final-gate lane.
- Native sub-agents remain allowed for routine implementation and verification work that is not an activated QAS lane.

## Startup Gate (`start`)
- Authoritative workflow: skill `$start` at `/home/jkotieno/.codex/skills/start/SKILL.md`.
- Run this for repo execution, implementation, or dispatch-lane work; do not run it for ordinary Sweet Tea Studio MCP product operations.
- `start` automatically repairs missing or drifted instruction targets from tracked sources before preflight; do not require a separate sync/check step from the user.
- Startup does not activate QAS or transfer normal testing away from the primary agent. The receipt's `testing_owner_role=qa_sentinel` identifies the conditional owner only if a finish-line QAS lane is later activated.
- Capture and carry `startup_receipt_id` for all new `dispatch-task` and `ingest-dispatch` commands.

## Core Document Set (Distilled)
- `docs/CODEX_RUNTIME_DELTA.md`
- `docs/SWEET_TEA_OS_CANONICAL_CONTRACT.md`
- `docs/STARTUP_HANDSHAKE_CONTRACT.md`
- `docs/QA_SENTINEL_SPEC.md`
- `docs/AGENT_EVENT_CONTRACT.md`
- `docs/PRODUCT_OPERATING_SYSTEM.md`
- `docs/FOUNDER_PREFERENCES.md`
- `docs/REPO_SYSTEM_MAP.md`
- `docs/DECISION_REGISTER.md`

## Repo-Local Rules
- If `./docs/CODEX_REPO_LOCAL.md` exists in the active repo, read it for repo-specific constraints.
- Repo-local files may add constraints but should not redefine global terms.

## Test Legitimacy and CI Authority
- A blocking test must directly and deterministically prove a foundational user capability (`PF`) or shared architectural invariant (`AF`).
- Do not add tests by reflex or by change type. Check for an existing direct proof first; no new test is the right result when no new foundational contract exists.
- Treat incomplete implications, arbitrary thresholds, private implementation assertions, duplicated/cascading coverage, environmental checks, and observational metrics as non-authoritative. Rewrite only when the underlying invariant deserves protection; otherwise delete them.
- Before fixing code for a red test, establish what invariant the test protects, whether it has CI authority, and whether the failure is caused by the product rather than the test or environment.
- Prefer one direct test over layered repetitions. Never preserve test count or coverage for its own sake, and delete obsolete fixtures, mocks, snapshots, baselines, and harnesses with the tests that used them.
- Exact copy, source text, helper calls, component trees, pixel diffs, and numerical budgets may block only when that exact surface or boundary is an explicit foundational requirement.
- Blocking checks must be deterministic without retries, accepted flakes, uncontrolled services, order dependence, or environment-dependent skips.

## Root Cause Before Fallbacks
- Do not give generic best-practice advice as a substitute for causal reasoning.
- Diagnose the actual mechanism first; recommendations must directly affect that mechanism.
- Before proposing an action, rule out non-causal/placebo actions and explain why they are not useful.
- If an action would not have prevented, exposed, or narrowed the observed problem, do not recommend it.
- Do not add fallback behavior, default-filled payloads, retries, alternate providers, mock data, broad exception swallowing, randomized behavior, or best-effort paths before root cause is confirmed.
- If root cause is not confirmed, preserve diagnostic signal and keep status `UNVERIFIED`; do not mask the failure with a fallback.

## Defaults
- Scope changes to the requested outcomes, but do not optimize for minimal diffs over correctness, clarity, or maintainability.
- Mark unknowns as `unverified` and verify.
- Prefer making the happy path work or failing closed. Do not add silent or weak fallback paths that hide defects, randomize behavior, or block diagnosis. Use fallbacks only when they are intentional, testable, observable parts of the design.
- Do not claim success without exact verification commands and results.
