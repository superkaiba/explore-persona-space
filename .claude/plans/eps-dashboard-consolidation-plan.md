# EPS dashboard consolidation refactor

*Generated 2026-05-29 16:03 via /plan skill (3 review rounds; Codex skipped rounds 2-3 per user interrupt)*

## Original request
Devise a comprehensive plan to implement the EPS dashboard consolidation: collapse the 5 confusing nav surfaces (Tasks/Updates/Log/Literature/Docs — clean results scattered across 3 of them, two overlapping feeds) into a clean "3 stores + 2 lenses + 1 external" model, built on ONE shared markdown component with auto-TOC, per-header collapse, highlight-to-comment, and a global "Ask Claude Code" affordance. Dashboard only; the clean-result v2 body-format spec is an input, not part of this work.

## Clarifications gathered
- **Rollout:** develop on a branch, single big-bang deploy (not incremental-behind-routes).
- **Legacy bodies:** render-compat now (keep a legacy-HTML path), migrate the ~20 Sagan-card bodies lazily.
- **Comment anchors:** best-effort, orphan-on-edit (flag detached comments; no robust re-anchoring for v1).
- **Comment storage:** "git-committed (task model)" — *but* review rounds found this rested on a wrong premise (log AND docs comments are gitignored today; only task/updates are committed). Softened to: unify schema + component, preserve each backend's existing commit-status for v1. **Open for override (see Open questions).**
- **Ask-Claude:** both, per-surface — a global ephemeral sidecar chat everywhere + preserve the task session-spawning answerer.
- **Public Results predicate:** completed + classification `useful` only (exclude proposed/awaiting/archived/exemplars like #442).
- **Read access:** public on Overview + Results only; gate read of Tasks/Docs/Updates/Literature + all writes/chat/comment.
- **Verified facts (round 2-3):** `DASHBOARD_AUTH_ENABLED=true` in prod `.env.local`; auth via `proxy.ts` matcher (allowlist of *gated* paths, fails closed on missing secret); `rehype-sanitize` NOT installed; 16 ReactMarkdown occurrences (11 files) / 22 rehypeRaw usages (9 files); `CommentableBody` already implements selection/anchor/collapse (~616 lines, `"use client"`, `AnchoredCommentsContext`); `/` IS the task list today (no `app/tasks/page.tsx`); `app/preview` + `app/sign-in` + `app/tasks/[id]/{edit,plan}` exist; build is NOT `output: standalone` (`next start` resolves deps from live `node_modules`); `useful` is currently decided by a prose regex, not a structured field; ~92 `has_clean_result` bodies.

---

## Goal
Consolidate the dashboard from 5 confusing surfaces into 6 crisp ones (3 stores + 2 lenses + 1 external) on ONE shared markdown component generalized from `CommentableBody`, with a unified comment layer, a global Ask-Claude affordance, and sanitized public read on Overview + Results only. Develop on an isolated branch/worktree; deploy once.

## Success criteria
- Nav = **Overview(`/`) · Updates · Tasks(`/tasks`) · Results · Docs · Literature.**
- ONE shared markdown component (generalized from `CommentableBody`, server-render + client-shell) backs all 16 ReactMarkdown sites, with auto-TOC + per-header collapse + highlight-to-comment + Ask-Claude.
- Clean results have ONE canonical home (Results; predicate reads the authoritative `classification` field = completed + `useful`); Updates shows pointer-cards only.
- Public surfaces render `rehype-raw → rehype-sanitize` with an allowlist (KaTeX, `<mark data-comment-id>`, highlight classes, `<details>`); an injected `<script>`/`<img onerror>` in a fixture body AND every one of the ~20 real legacy bodies is provably neutralized/visually unchanged.
- ONE comment schema + component; each backend's existing commit-status preserved (task/updates committed, docs/log gitignored) unless overridden; task `add-comment` subprocess + reply trees + addressed/archived intact; idempotent, never deletes originals.
- Auth = **deny-by-default** matcher; public read only `/`, `/results`, `/results/[id]`, `/sign-in`, `_next`/static assets; everything else + all writes/chat/comment + `/api/sidecar` + `/api/chat-token` gated; verified by an unauth curl matrix run with `DASHBOARD_AUTH_ENABLED=true`.
- `/log` + stale `/updates/[id]` redirect; `/`→Overview, task list at `/tasks`; `/preview` + `/sign-in` handled; no 404 on shared links.
- `next.config` `outputFileTracingIncludes` updated for every new/changed route reading `../tasks`, `../docs`, `../logs`, `../updates/literature`.
- Cutover installs deps (`npm ci`) before build/restart; smoke passes against the BUILT bundle first; one-command rollback exists.

## Out of scope
Experiment code; the clean-result v2 body-format spec (an input); Literature internals; the `updates/literature/` data dir (untouched — only the `/updates` route semantics change); `EditableBody`'s `@mdxeditor/editor` preview (stays its own component); authoring new doc content.

## Approach
Develop in an **isolated git worktree** so branch builds never touch the live serving `.next`. **Step 1 is audit-only** and larger than it sounds: enumerate every route (`/`, `/tasks/[id]{,/edit,/plan}`, `/preview`, `/sign-in`, `/updates`, `/log`, `/literature`, `/docs`) and all 16 ReactMarkdown / 22 rehypeRaw usages with a per-site keep/fold/special-case decision; map each comment backend's storage model + commit-status (task=committed, updates=committed-same-store, docs=gitignored, log=gitignored) + anchor schema + reply/addressed/archived fields + Claude side-effects; confirm the `proxy.ts` matcher mechanics under Next 16 (negative-lookahead vs match-all+in-function allowlist) against the bundled `proxy.md`; decide adopt-or-discard the untracked docs-comment WIP (don't blind-commit); green build + typecheck in the worktree.

**Keystone = generalize `CommentableBody`** (it already does selection/anchor/collapse) into a shared component split into a SERVER renderer (markdown→HTML via `rehype-raw` THEN `rehype-sanitize` on public surfaces; disk reads; auth) + a CLIENT shell (TOC active-state, collapse, selection capture, `<mark>` wrapping, Ask-Claude). Treat this as a **rewrite-grade** change: the comment `<mark>` mutation stays client-side, and a **hydration-parity gate** (server-sanitized HTML === client DOM before mutation) is a Step-2 deliverable or anchors orphan / hydration mismatches. Props contract: `{docRef, html|markdown, isLegacyHtml, comments[], public}`. Legacy bodies (trusted analyzer HTML) keep their `dangerouslySetInnerHTML` path but, since now public, are run through the sanitizer with an allowlist **derived from what those 20 bodies actually contain** (visual-diff every one; a body that breaks is a blocker, not a lazy-migrate item). Prove sanitize kills an injected payload as a gated test. Validate the keystone EARLY on a REAL completed task body with REAL anchored comments.

**Comments:** unify onto one schema + the shared component; reuse the existing field names (`in_reply_to`, `addressed`, `archived`) — don't invent new ones; preserve the task `add-comment` subprocess. For v1, **preserve each backend's commit-status** (don't force gitignored docs/log review chatter into committed experiment task folders); the migration is schema/code-path alignment, not a storage relocation. Idempotent, back-compatible reader, dry-run diff, never delete originals; run any data migration during the cutover freeze (the in-process file lock doesn't protect against the live server writing concurrently). Best-effort orphan-on-edit + a margin list; rendered-DOM anchors with an occurrence index; comments in collapsed sections auto-expand on navigate.

**Ask-Claude:** a global sidecar-chat panel (doc/selection context) on GATED surfaces + the preserved task session-spawn answerer; on PUBLIC surfaces (`/`, `/results`) the panel renders DISABLED and `/api/sidecar` + `/api/chat-token` stay gated (no token fetch). Graceful-disabled if the sidecar/`SIDECAR_INTERNAL_URL` is unreachable.

**Surfaces** then build on the keystone. **Auth inversion to deny-by-default + `next.config` tracing-includes + redirects + nav swap are LAST.** Production cutover during a planned brief downtime: merge→main; in the serving dir `npm ci` (installs `rehype-sanitize` from the merged lockfile) → `next build` → `systemctl restart` → verify; rollback = `git revert` the merge + `npm ci` + `next build` + restart (keep `.next.bak` + lockfile snapshot as a fast path).

## Steps
1. **Audit + isolated worktree** (no feature code): enumerate all routes + all 16 RM / 22 rehypeRaw usages (keep/fold/special-case); map each comment backend's commit-status + schema + side-effects; confirm `proxy.ts` matcher mechanics (bundled `proxy.md`) + `DASHBOARD_AUTH_ENABLED=true` + fail-closed; decide adopt-or-discard the untracked docs-comment WIP; green build + typecheck in the worktree.
2. **Keystone** by generalizing `CommentableBody` → server-renderer + client-shell, serializable props; add `rehype-sanitize` (raw→sanitize order) on the public path; audited legacy-HTML path with an allowlist derived from the 20 real bodies; shared auto-TOC util. GATED: injected payload neutralized in a fixture AND all 20 legacy bodies visual-diff-clean; **hydration-parity** check passes. Validate on a REAL completed body + its real anchored comments.
3. **Comment unification (schema/code-path, not storage relocation):** one schema + component; reuse `in_reply_to`/`addressed`/`archived`; preserve the task `add-comment` subprocess; align the docs/log/updates paths onto the shared component while keeping each one's current commit-status; idempotent back-compatible reader; dry-run diff; never delete originals; orphan detection + margin list; collapsed-section anchors auto-expand. (Any data migration runs during the cutover freeze.)
4. **Ask-Claude:** global sidecar-chat on GATED surfaces + preserved task answerer; DISABLED on public `/`,`/results`; `/api/sidecar`+`/api/chat-token` stay gated; graceful-disabled if unreachable.
5. **Results route (public):** predicate reads the authoritative `classification` field (completed + `useful`), NOT a prose regex; filterable (confidence/topic/date); keystone detail; data lib exposes only public-safe fields.
6. **Overview as `/`:** move the current root task-list → new `app/tasks/page.tsx`; new `app/page.tsx` = Overview (open_questions + SUMMARY + beliefs + recent-activity strip); public.
7. **Docs categories:** `category:` frontmatter + a default "Reference" bucket (nothing dropped); resolver surfaces meetings + mentor_updates + daily/weekly without physical file moves; old-doc-link redirects; read-gated.
8. **Updates merge:** chronological pointer-card aggregator (completed-results + dated docs); retire `/log`; read-gated.
9. **Migrate task body + plan pages** onto the keystone (`EditableBody` `@mdxeditor` preview stays separate).
10. **Auth inversion + cutover prep:** flip the matcher to deny-by-default (public allowlist = `/`,`/results`,`/results/[id]`,`/sign-in`,`_next`/static/`favicon`; everything else gated incl `/api/sidecar`,`/api/chat-token`); ensure redirects don't bypass auth; update `outputFileTracingIncludes` for every new/changed route (`../tasks` for `/tasks`+`/preview`, `../docs`, `../logs`, `../updates/literature`) + redirects (`/log`→`/updates`, stale `/updates/[id]`→`/results/[id]`-or-`/updates`); nav swap (6); delete dead routes/components.
11. **Verify + deploy:** worktree `next build && next start` on a scratch port → smoke (6 nav + task/result/preview detail 200; all 20 legacy bodies render; sanitize kills payload; comment round-trip; redirects 30x; **unauth curl matrix with auth ENABLED**: `/`,`/results` 200, all else gated, writes/sidecar blocked; assets load; Ask-Claude disabled on public + when sidecar down) + typecheck/tests → merge→main → serving-dir `npm ci` → `next build` → `systemctl restart` → verify PID/:3010 + unauth public spot-check. Rollback = `git revert` merge + `npm ci` + build + restart (`.next.bak`/lockfile snapshot as fast path).

## Risks and mitigations
- **Cutover dependency gap (non-standalone build):** `next start` resolves `rehype-sanitize` from live `node_modules`, so cutover MUST `npm ci` before build/restart; rollback restores the matching lockfile + build. (Evaluate `output: standalone` in step 1 as a self-contained alternative — but it changes the systemd `ExecStart`/asset layout, so default to `npm ci`.)
- **rehype-sanitize not installed + raw HTML everywhere:** install + pin (verify compat with `rehype-raw@7`/React 19/Next 16); raw→sanitize order; gated payload + 20-legacy-body visual-diff tests; allowlist KaTeX/`<mark>`/highlight/`<details>`.
- **Keystone is rewrite-grade, not a refactor:** `CommentableBody` is 616 lines of client code on a React context; hydration-parity gate; `<mark>` wrapping stays client-side; budget accordingly.
- **Auth fail-open:** invert to deny-by-default (forgotten route fails closed); exempt `_next`/static/favicon explicitly or pages won't load; unauth curl matrix run with auth enabled.
- **Comment commit-status premise:** log AND docs are gitignored today; v1 preserves commit-status (schema/code-path unification only); migration during cutover freeze; reuse existing field names; never delete originals.
- **Legacy-HTML vs sanitize tension:** separate trusted-HTML path; allowlist derived from real bodies; any break is a blocker.
- **Build clobbers live `.next`:** dev/smoke in the worktree; cutover only inside a planned `stop→npm ci→build→start` window.
- **Selection-anchor + collapse:** orphan-on-edit + margin list; rendered-DOM anchors + occurrence index; collapsed anchors auto-expand; honor `CommentableBody` effect-ordering.
- **Root relocation:** explicit `/`→`/tasks` move + new Overview; smoke covers both + `/preview` + `/sign-in`.
- **NFT tracing:** explicit includes for `../tasks` (`/tasks`,`/preview`), `../docs`, `../logs` (currently absent), `../updates/literature`; smoke the BUILT bundle, not `next dev`.
- **Server/client boundary:** split keystone; no disk/auth in the client bundle.
- **Public Results leakage:** authoritative `classification` field; public-safe fields only.
- **Counts/denominator:** 16 RM (11 files) / 22 rehypeRaw (9 files); `#442` is `proposed` (no v2 body) → fixture from a real completed body, not #442.
- **:3010 management:** `systemctl` status/restart + PID inspect; `ss` diagnostic only.
- **Long-lived branch on a live-mutating repo:** rebase code on main; the built bundle reads live data at request time (fine); fixtures track the live v2/category format at cutover.

## Open questions
1. **Comment commit-status (needs your override-or-confirm).** Your "git-committed (task model)" decision was made before we learned log AND docs comments are gitignored today. Default in this plan: unify schema + component but keep each backend's current commit-status (task/updates committed; docs/log gitignored). Override if you want docs/log review comments folded into committed task folders (permanent git history).
2. Beliefs source for Overview — `open_questions.md` + `SUMMARY.md` for v1; dedicated `beliefs.md` later. (Assumed.)
3. Stale `/updates/[id]` redirect target — `/results/[id]` when it maps to a result, else `/updates`. (Assumed.)
4. `output: standalone` vs `npm ci` cutover — defaulted to `npm ci` to avoid changing the systemd unit on a custom Next.js; revisit if a self-contained artifact is preferred.

---

## Review history

### Round 1
- **Claude — ACCEPTABLE.** Raised: comment layer is 4 backends not 2 (two storage models); two distinct Claude mechanisms; no `middleware.ts` in source; Overview-as-`/` silently relocates the task list; NFT includes missing for new routes; counts off; #442 has no v2 body. **Fixed:** all folded into rev 2 (4-backend enumeration, auth audit, `/tasks` relocation step, NFT step, corrected counts).
- **Codex — WEAK.** Raised: branch build can clobber live `.next`; public + rehypeRaw = XSS; MarkdownDoc server/client boundary; comment migration underspecified; selection anchoring weaker than assumed; auth matrix ambiguous; Results count/predicate shaky; Docs categorization needs a real model. **Fixed:** isolated-worktree build, sanitize requirement, server/client split, explicit predicate + auth matrix, default category bucket.

### Round 2
- **Claude — ACCEPTABLE.** Raised: `rehype-sanitize` not installed (critical) while rehypeRaw is everywhere and `/` goes public; ~13 RM sites and `CommentableBody` already does the hard parts (generalize, don't greenfield); `/tasks`/`/preview` route collisions + NFT; matcher allowlist → must invert to deny-by-default + verify `DASHBOARD_AUTH_ENABLED`; comment divergence narrower than stated; in-place build cutover has no rollback. **Fixed:** sanitize as a gated deliverable, generalize CommentableBody, route inventory + deny-by-default, release-artifact cutover, narrowed comment step. **Codex — skipped** (user interrupted the tool use).

### Round 3
- **Claude — ACCEPTABLE.** Raised: **non-standalone build means `.next`-only cutover can't resolve the new dep (critical)**; log AND docs are gitignored (commit-status premise wrong); legacy Sagan HTML bypasses react-markdown (separate path, sanitize is a behavior change — test the real 20); generalizing CommentableBody is rewrite-grade (hydration parity); counts are 16 RM/22 rehypeRaw; `useful` is a prose regex not a field. **Fixed:** `npm ci` cutover, preserve-commit-status default + open question, 20-body visual-diff gate, hydration-parity gate, corrected counts, authoritative-classification predicate. **Codex — skipped** (per user interrupt).

### Issues deviated from / surfaced for override
- **Literal "git-committed comments" decision softened** to preserve per-backend commit-status (Open question 1) — it rested on the wrong premise that 3/4 backends were already committed.
- **`output: standalone` not adopted** (Open question 4) — chose `npm ci`-in-serving-dir to avoid changing the systemd unit on a custom Next.js.
