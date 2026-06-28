# Phase B1 — body.md reader migration checklist (paper-stub support)

Mechanical discovery + keep/migrate decision per body-content reader, for the
`paper: true` paper-stub body type (SPEC.md § "Paper format (`paper: true`)" +
§ "`body.md` paper-stub contract").

**The paper-stub** (for a `paper: true` `kind: experiment` task): frontmatter
(`paper: true`, `title`, `kind`, `goal`, `has_clean_result`, confidence in the
title tag, origin fields) + body = an H1 title + an abstract (a paragraph after
the title OR a `## Abstract` H2) + a paper link (`docs/papers/issue_<N>/…` path,
or an HF `/papers/issue_<N>/` URL, or `paper.html`). The stub has NO
`## Takeaways` / `## Goal` / `## Methodology` / `## Results` / `## Findings` /
`## Data` / `## TL;DR` sections. The paper itself (the `.tex` → PDF +
`paper.html` under `docs/papers/issue_<N>/`) IS the clean-result and is verified
by `scripts/verify_paper.py`, NOT `verify_task_body.py`.

Grandfathered markdown bodies (v4 `<!-- clean-result-v4 -->`, v3
`<!-- clean-result-v3 -->`, v2 `<!-- clean-result-v2 -->`, legacy) keep their
EXACT behavior — the `paper:` flag is a deterministic branch (absent/false ⇒ the
markdown generations, unchanged).

Discovery method: `grep -rl` over `scripts/`, `dashboard/{lib,components,app}`,
`.claude/` for `body.md` / `## Takeaways` / `## Results` / `## Findings` /
`## TL;DR` / `set-clean-result` / `has_clean_result` / frontmatter reads /
`Confidence`. Figure/plot scripts (`scripts/issue*_*.py`, `scripts/plot_*.py`,
`scripts/i*_*.py`, `scripts/make_*figure*.py`, …) are EXPERIMENT-SPECIFIC and
out of the workflow surface — excluded.

---

## Python — scripts/ + src/ workflow surface

| Reader | What it does with body content | Decision |
|---|---|---|
| `scripts/task.py` (`cmd_set_body`, `cmd_set_clean_result`) | `set-body` runs `_assert_body_nontrivial` (≥500 chars, not a stub token) unless `--allow-stub`; `set-clean-result` flips `has_clean_result`. | **MIGRATE** — (a) `set-body` auto-allows a short stub for `paper: true` tasks (a short paper-stub is NOT the #385 silent-handoff defect); `--allow-stub` still works for any task. (b) `set-clean-result` for a `paper: true` task VALIDATES `docs/papers/issue_<N>/paper_manifest.json` (artifacts present + sha256 match + `pdf_hf_url` set, WARN-soft) BEFORE flipping. |
| `src/explore_persona_space/task_workflow.py` (`set_clean_result`, `_registry_set`) | `set_clean_result` flips `has_clean_result` + denormalizes REGISTRY. `_registry_set` denormalizes title/kind/status/has_clean_result/goal. | **MIGRATE** — (a) `set_clean_result(..., validate_paper_manifest=...)` hook so the CLI can enforce the manifest check inside the flock+commit. (b) `_registry_set` denormalizes an `abstract` field from a paper-stub body (title already denormalized; abstract is the new denorm field the dashboard hover-card / REGISTRY title-abstract work reads). |
| `scripts/verify_task_body.py` (`verify_text`) | Runs the markdown clean-result checks (Check 0 stub-guard, v2/v3/v4 structure, etc.) gated on the sentinel. | **MIGRATE** — add a `paper: true` branch at the TOP of `verify_text`: a paper-stub PASSes with one result pointing at `verify_paper.py` (it is NOT hard-FAILed by the markdown Check 0 / structure checks). Grandfathered markdown bodies unchanged. |
| `scripts/recent_clean_results.py` (`_extract_markdown`) | Extracts the `## Takeaways` (v3+) / `## TL;DR` (v2/legacy) skim block for the analyzer's exemplar feed; hero figure; confidence from the title tag. | **MIGRATE** — add a paper-stub branch: skim block = the stub abstract (prose between H1 and the paper link); confidence already comes from the title tag. Without it the fallback dumps the whole stub (non-empty but unstructured). |
| `scripts/audit_clean_results_body_discipline.py` (`audit_body` / `_audit_single_body`) | Scans clean-result PROSE for body-discipline anti-patterns (pre-reg jargon, inline CIs, etc.). | **MIGRATE** — a paper-stub is exempt (its prose is a short abstract; the paper body is the `.tex`, audited by verify_paper). Early PASS for `paper: true` bodies. |
| `scripts/pm_queue_report.py` | Reads `has_clean_result`/`goal`/`title`/`tags` from frontmatter only. | **KEEP** — frontmatter-only; no section grep. |
| `scripts/gh_project.py` (`body-promote`/`body-restore`) | Operates on GitHub-ISSUE bodies (historical evidence channel), not the canonical `tasks/` clean-result. | **KEEP** — GH-issue era; not a `tasks/` body reader. |
| `scripts/living_docs.py` | Reads frontmatter + appends task refs (`#N`) to living-doc evidence lists. | **KEEP** — frontmatter-only; no clean-result section grep. |
| `scripts/render_papers_index.py` | Regenerates `papers/INDEX.md` (the LITERATURE index of arXiv papers), unrelated to `docs/papers/issue_<N>/` clean-result papers. | **KEEP** — different "papers" namespace. |
| `scripts/verify_clean_result.py` | LEGACY GH-issue-era validator (`## Human TL;DR` / `## AI Summary` shape). Superseded by `verify_task_body.py`; not in the Phase-B1 named reader list. | **KEEP** — legacy, out of scope. |

## TypeScript — dashboard/ (data + render layer is body-opaque)

The dashboard treats `body.md` as an OPAQUE markdown string everywhere. NO file
greps for `## Takeaways` / `## Methodology` / `## Findings` / `## Abstract`, no
hero-figure extraction, no headline-skim. Structured extractions: title
(frontmatter/REGISTRY), confidence (title regex), a GENERIC markdown-stripped
excerpt. Frontmatter `paper:` is already readable (`Frontmatter` has
`[k: string]: unknown`).

| Reader | What it does with body content | Decision |
|---|---|---|
| `dashboard/lib/tasks.ts` (`getTask`, `listAllTasks`, `recentTasksForUpdates`) | gray-matter frontmatter parse; `body` = `fm.content` verbatim. `has_clean_result` from REGISTRY (lists) + frontmatter (detail). | **KEEP** (ergonomics: add `paper?: boolean` + `goal?: string` to the `Frontmatter` type + an `isPaperTask()` helper). |
| `dashboard/lib/results.ts` (`markdownExcerpt`, `parseConfidence`) | Public catalog: title from fm, confidence from title regex, generic excerpt (strips code/H1/images/links). On a stub the excerpt = the abstract paragraph (non-empty). | **KEEP**. |
| `dashboard/lib/task-data.ts` (`figureUrlsFromBody`) | Scans body only for SHA-pinned `figures/issue_<N>/*.png` raw-GitHub URLs. A stub has none → `{artifacts:[]}`, viewer self-hides. | **KEEP**. |
| `dashboard/lib/update-results.ts` | Pure helpers for the LEGACY DB feed (not the live `tasks/`-backed surfaces). | **KEEP**. |
| `dashboard/lib/questions.ts` | Parses `docs/open_questions.md`, never task bodies. | **KEEP**. |
| `dashboard/lib/logs.ts` (`listCleanResults`) | Body used only as a search haystack for `/preview/log`; classification from fm; title from REGISTRY. A stub is a thinner haystack. | **KEEP**. |
| `dashboard/lib/markdown-sanitize.ts` | Exports `markdownSchema` (the `.mjs` `buildPaperSchema(markdownSchema)` extends it for the Phase-C `paper.html` render). A plain stub (H1 + abstract + link) needs only `markdownSchema`. | **KEEP** (Phase-C wiring point already exists). |
| `dashboard/app/tasks/[id]/page.tsx` (`BodyCard`) | Strips one leading H1, renders via `TaskBodyMarkdown`; wraps in `<EditableBody>`. | **MIGRATE** — compute `isPaper` and pass `canEdit={canEdit && !isPaper}` to `<EditableBody>`; show a paper-task hint. |
| `dashboard/app/tasks/[id]/EditableBody.tsx` | `!canEdit` ⇒ read-only children; else "Edit body" → `InlineBodyEditor`. | **MIGRATE (indirect)** — gated by `canEdit && !isPaper` from BodyCard; optional explicit `paper` notice. |
| `dashboard/app/tasks/[id]/edit/page.tsx` | Standalone editor route; unconditionally mounts `<Editor>`. | **MIGRATE** — if `task.frontmatter.paper === true`, render a "this is a paper-task — edit `docs/papers/issue_<N>/issue_<N>.tex` in git" notice instead of `<Editor>`. |
| `dashboard/app/tasks/[id]/edit/actions.ts` (`saveTaskBody`) | Writes via `task.py set-body`. | **MIGRATE (defense-in-depth)** — block `saveTaskBody` for `paper: true` tasks server-side (a forged request must not overwrite the stub). |
| `dashboard/app/tasks/[id]/edit/Editor.tsx`, `InlineBodyEditor.tsx` | CodeMirror editors → `saveTaskBody`. | **MIGRATE (indirect)** — never mounted for paper-tasks (gated above). No change inside the components. |
| `dashboard/app/results/page.tsx`, `results/[id]/page.tsx`, `preview/page.tsx`, `preview/log/page.tsx`, `components/MarkdownDoc.tsx`, `TaskBodyMarkdown.tsx` | Listing fields / read-only public render / opaque-markdown render. | **KEEP** — all render a stub (abstract + link) fine. |

## Markdown skills — .claude/skills/

These are LLM-instruction files; "MIGRATE" = add an additive `paper: true`
branch in the prose.

| Skill | Touchpoint | Decision |
|---|---|---|
| `.claude/skills/promote-clean-result/SKILL.md` | Step 4 Verify (runs `verify_task_body.py` + audit) | **MIGRATE (hard breaker)** — for a paper-task run `verify_paper.py --issue <N>` instead; the stub is exempt from the markdown section checks. |
| `.claude/skills/promote-clean-result/SKILL.md` | Step 2 Detect format / Step 3 Refine body | **MIGRATE** — add a first `paper: true` branch: the body is a thin stub, the canonical clean-result is the `.tex`; skip the v3/v4 section refinements; the only in-body refinements are the H1 title tag + the abstract + the paper link. |
| `.claude/skills/group-promotion-queue/SKILL.md` | §3 digest slice greps (`## Takeaways` / `## Findings`) | **MIGRATE** — for a paper-task pull the claim summary from the stub abstract + frontmatter `goal`; skip the finding-headline greps (none exist). |
| `.claude/skills/mentor-update-slides/SKILL.md` | Qualitative-example sources (`#### <finding>` H4) | **MIGRATE** — for a paper-task pull the worked example from the `.tex` paper / HF raw-completions bucket, not a body H4. |

## Backward-compat proof

- `scripts/verify_task_body.py`: the `paper: true` branch is the ONLY new code
  path; it fires iff `split_frontmatter(raw)[0].get("paper") is True`. Every
  grandfathered body (v4/v3/v2/legacy) has no `paper:` flag ⇒ the existing
  `verify_text` body runs byte-for-byte unchanged. Pinned by a test that a v4
  body and a no-frontmatter body verify exactly as before, and that the
  paper-stub PASSes.
- `scripts/task.py` / `task_workflow.py`: the manifest-validation + abstract-
  denorm are gated on `fm.get("paper") is True`; non-paper tasks take the
  existing path. Pinned by the existing `test_task_workflow.py` suite (unchanged
  behavior) + a new paper-task test.
- `dashboard`: the editor-disable is `canEdit && !isPaper`; for a non-paper task
  `isPaper` is false ⇒ identical behavior. Pure additive TS, typecheck-clean.
