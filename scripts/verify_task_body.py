#!/usr/bin/env python3
"""verify_task_body.py — mechanical verifier for markdown clean-result bodies.

Replaces `verify_sagan_card.py` for new (markdown) bodies. Mechanical
gate for the markdown clean-result spec, which has TWO live,
sentinel-gated shapes (plus pre-sentinel legacy as a third grandfathered
shape). Source of truth for both: `.claude/skills/clean-results/SPEC.md`.

- **v3 (current, `<!-- clean-result-v3 -->`):** FIVE required H2s in
  order — `## Takeaways` / `## What I ran` / `## Findings` / `## Data` /
  `## Reproducibility`. No `## Human TL;DR` (its presence is a FAIL).
  Conciseness caps + the Data section are mechanically enforced. The full
  v3 check list is enumerated in the "v3 checks" block below (~L260+).
- **v2 (`<!-- clean-result-v2 -->`, migrated 2026-W22, task #454):** THREE
  required H2s in order — `## Human TL;DR` / `## TL;DR` /
  `## Reproducibility` — with `## TL;DR` absorbing the per-result
  narrative (one `### Motivation` H3 then one `### <finding>` H3 per
  result with one inline figure + cherry-picked raw-completion example +
  dropdown + all-raw link) and `## Reproducibility` absorbing the
  Parameters table + the Confidence sentence.

The retired `## Details` and `## Figure` H2s are FAIL patterns in both —
a body that carries either is rejected so it migrates cleanly (forward-
only; the ~95 legacy `has_clean_result=true` bodies are never
re-verified, so tightening cannot regress them). Sentinel gating: checks
6/16/17 run on EITHER sentinel (`is_nested_design`); v2-only structural
checks (3/3b) run on the v2 sentinel; v3-only checks (v3 structure +
18-21) run on the v3 sentinel.

0. Body is not a stub — body has ≥500 chars, contains a `# <title>` H1,
   and is not a single stub token (`placeholder`, `tbd`, `todo`, `stub`).
   Defense-in-depth against the cache → body.md silent-handoff failure
   (incident: task #385, 2026-05-25). Runs FIRST and short-circuits the
   rest of the check chain — a stub body produces ONE clear FAIL at the
   top rather than a dozen cascading "<section> missing" errors.
0b. No duplicate frontmatter — the body region (post-canonical-frontmatter)
   must NOT start with another `---\\n...\\n---\\n` YAML block. Caller-
   supplied frontmatter passed through `task.py set-body` is stripped by
   the library; this check is the belt-and-suspenders gate against any
   future regression (manual editing, alternative write path) that lets
   a duplicate block land on disk. The dashboard would otherwise render
   the second block as literal YAML at the top of the visible body
   (incident: task #389, 2026-05-26).
1. Title confidence tag — H1 line ends with `(LOW|MODERATE|HIGH confidence)`.
2. Three required H2 sections in order — `## Human TL;DR`, `## TL;DR`,
   `## Reproducibility`. A stray `## Details` or `## Figure` H2 in a
   NEW body is a hard FAIL (2026-W22 migration, task #454): the
   2-content-section spec collapses the Details narrative into per-result
   H3s under `## TL;DR` and inlines figures inside those H3s, so bodies
   carrying either retired H2 must clean-migrate before promotion. Extra
   H2s OTHER than `Details` / `Figure` after `## Reproducibility` are
   allowed.
3. TL;DR Motivation section — `## TL;DR` opens with the canonical
   Motivation block, either as an `### Motivation` H3 (preferred new
   shape) or as a `**Motivation:**` boldface bullet (legacy form, still
   accepted). The retired `What I ran` / `Results` required bullets are
   no longer enforced — the new shape uses per-result `### <finding>`
   (legacy flat) or `#### <finding>` (nested-design v2) headings,
   checked by structure (one figure per result block, raw-completion
   sample preceded by cherry-picked label + qualitative-data link) via
   checks 4, 10, 11.

3b. TL;DR nested-design (v2) structure — bodies carrying the
    `<!-- clean-result-v2 -->` sentinel MUST shape `## TL;DR` as three
    ordered H3s — `### Motivation` / `### What I ran` /
    `### Findings` — with at least one `#### <finding>` H4 child
    under `### Findings`. Bodies without the sentinel PASS vacuously
    (forward-only migration).
4. Hero image present — at least one `![alt](url)` image exists
   inline under `## TL;DR` (every result H3 carries its own figure in
   the 2-content-section spec).
4b. Figure URL resolvable — every image URL inline under `## TL;DR` is
    an absolute `https://...` URL the dashboard can fetch. Relative
    paths (`artifacts/...`, `tasks/...`, `figures/...`, `./...`,
    `../...`) fail because the EPS dashboard does not serve binary
    PNG/PDF files under `tasks/<N>/artifacts/` (incident: task #365,
    2026-05-22). `raw.githubusercontent.com` URLs must pin to a commit
    SHA, not `main`/`master`/`HEAD`. The TARGET must also EXIST
    (incident: task #507, 2026-06-09 — a caption cited a figure that
    was never generated): same-repo SHA-pinned raw URLs are verified
    offline via `git cat-file -e <sha>:<path>` (definitive miss →
    FAIL); unknown SHAs / other hosts fall back to one HTTP HEAD per
    unique URL (definitive 404 → FAIL; network error / timeout →
    `unverified` note on the PASS line, never a FAIL).
5. Figure caption sanity — vacuously satisfied under the new spec
   (inline-image alt text + blockquote caption inside each result H3
   carry the discipline; the analyzer is instructed to write
   descriptive alt text). Retained as a hook for future tightening; in
   the current revision the check always PASSes because the retired
   `## Figure` H2 is now a check 2 FAIL.
6. Confidence sentence matches title — for v2 nested-design bodies
   (`<!-- clean-result-v2 -->` sentinel present) the H1 title tag is
   the single source of truth; the check PASSes when the title carries
   the `(... confidence)` tag even with NO body `Confidence:` sentence.
   If a v2 body still carries one, the level must match the title and
   ≥20 chars of rationale after the dash. Legacy bodies (no sentinel)
   still require the `Confidence: LOW|MODERATE|HIGH — <rationale>`
   line somewhere in the body (typically in `## Reproducibility`).
7. Three repro subgroups present — `**Artifacts:**`, `**Compute:**`,
   `**Code:**` all appear as boldface labels inside `## Reproducibility`.
8. Reproducibility URL permanence — every URL in `## Reproducibility`
   pins to a ref (HF Hub `/tree/<ref>`, WandB `/runs/<id>`, GitHub
   `/blob/<sha>` or `/tree/<sha>`, raw
   `raw.githubusercontent.com/<owner>/<repo>/<sha>/<path>` — never
   `main`/`master`/`HEAD`). `n/a` is accepted as an explicit
   non-applicable marker. Raw-host URLs are scanned on fence-stripped
   text (a moving-ref raw URL inside a ``` example is illustrative);
   URLs on blockquote lines are exempt too (the **Context:** verbatim
   originating-prompt quote; #959); shape only — existence probing is
   check 8b's job.
8b. Reproducibility artifact URLs exist — same-repo artifact links in
    `## Reproducibility` (`raw.githubusercontent.com/<this-repo>/<sha>/
    <path>` raw URLs and `github.com/<this-repo>/(blob|tree)/<sha>/<path>`
    HTML URLs, e.g. the `**Code:**` blob links and the auto-appended
    `**Methodology reference:**` row) must point at objects that
    actually exist: resolved offline via `git cat-file -e <sha>:<path>`
    (works for file blobs AND directory trees), falling back to one
    HTTP HEAD per unique URL when the sha is unknown locally.
    Definitive miss → FAIL; indeterminate probe → `unverified` note on
    the PASS line, never a FAIL (same semantics as check 4b). Extends
    the task #507 existence protection to the Reproducibility section,
    which previously got shape verification only. Blockquoted URLs are
    not gathered (same #959 exemption). HF Hub / WandB /
    external-repo links stay shape-checked only (check 8): their
    existence is not decidable from the local object DB, and an
    unauthenticated 404 on an external private repo would false-FAIL.
9. Reproducibility sentinel scrub — no `{{`, `TBD`, `see config`, or
   `default` placeholders anywhere under `## Reproducibility`.
   `default` is flagged ONLY in placeholder positions — a bare table-cell
   value (`| default |`) or a label terminator (`chat template: default`
   at end of line / cell). Substantive prose uses ("default assistant",
   "default-context", "the default column") PASS: the default assistant
   is a core experimental condition in this project (open-q 3.7; task
   #542 false-positive).
10. Cherry-picked label discipline — every sample-output BLOCK in
    `## TL;DR` is preceded by prose containing `cherry-picked`,
    `cherry picked`, `random sample`, `first N of M`, or similar
    disclosure. A "sample-output block" is EITHER a fenced code block
    (heuristic: contains `User:`/`Assistant:`/`Human:`/`Model:` or
    >200 chars of text) OR a `<details>...</details>` block containing
    a GFM table delimiter row OR >200 chars of inner text. The
    `<details>`-block recognition catches the nested-design v2 form
    (e.g. task #432's `<details open>` training-row table) that the
    fence-only scan would silently pass.
11. Qualitative-data link — every sample-output BLOCK in `## TL;DR`
    is preceded by at least one link or backtick-wrapped path
    pointing at a raw text-level artifact (i.e. NOT an aggregate-only
    path like `regression`, `summary`, `aggregat*`, `per-cell`, or
    `.npz`). An explicit `not uploaded` / `not available` disclosure
    downgrades FAIL to WARN. Scope mirrors check 10 (both fenced code
    blocks AND `<details>` blocks).
11b. Planned-vs-actual denominator consistency — the body's `## TL;DR`
    `X of N <noun>` headline denominator must match any `M of N <noun>`
    documented scope claim found elsewhere in the body (typically in
    result-H3 prose that names a methodology correction). FAIL when an
    in-body scope claim says "M of N delivered" (with M < N) but the
    TL;DR opening still frames the result against N. Catches the
    scope-shrinkage-without-explicit-flag pattern that bit task #391
    (C-axis cell silently failed, body acknowledged the drop but the
    TL;DR still used the plan's denominator of 3). Whole-body scan
    under the 2-content-section spec (the retired `### Methodology
    corrections` H3 is no longer required, so scope-shrinkage prose can
    live in any result H3); plan-side enumeration is
    `clean-result-critic` Lens 13's semantic call.
12. Reserved (`## Figure` H2 deprecation hook). Under the
    2-content-section spec a stray `## Figure` H2 is rejected by check
    2 as a hard FAIL, so this hook is dormant in the current revision.
    Kept in CHECKS so the count stays stable and the slot is available
    if a future WARN-only nudge needs it.
13. TL;DR narrative flow (WARN-only) — two conservative mechanical
    signals that the body is shaped as a fact sheet rather than a
    LessWrong-style story: (a) outline-label H3s in `## TL;DR`
    (`### Headline result` / `### Subset checks` / `### Sample
    completions` / `### Plan deviations` / `### Methodology` /
    `### Findings`); (b) ≥3 consecutive `![alt](url)` images inside
    `## TL;DR` with no prose between (figure-dump). Both surface as
    WARN, never FAIL — critic-side LM judgment (clean-result-critic)
    catches the semantic cases this regex misses.
14. MDX-safe prose — no `<` characters that the dashboard's MDX parser
    will treat as the start of a JSX tag. This check has two layers.

    (A) Fast regex pre-checks (always run; the only layer when node is
    absent). Three anti-patterns fail: (a) `<https://...>` markdown
    autolinks (MDX errors with "Unexpected character `/` (U+002F) before
    local name"); (b) `<` immediately followed by a digit, e.g. `p<0.05`,
    `n<10`, `<24 personas` (MDX errors with "Unexpected character `0`
    (U+0030) before name"); (c) `<|` inside a GFM table cell, e.g. a
    `` `<|im_start|>` `` token in a table row — the table parser splits
    the cell on the unescaped `|` BEFORE code-span recognition, so the
    backticks do NOT protect the leaked `<|`, which MDX then reads as a
    JSX tag start ("Unexpected end of file before name"). Fix the table
    case by escaping the inner pipes inside the code span:
    `` `<\\|im_start\\|>` ``. Write URLs as `[label](url)` links and
    inequalities with surrounding spaces (`p < 0.05`) or wrap the token
    in backticks. On non-table lines, code spans (fenced + inline) are
    exempt; on table-row lines, only pipe-free code spans are treated as
    protective (a pipe-containing code span has its `<` left visible to
    the scan). `&lt;0.05`, `<= 10`, and `<` followed by a space all stay
    safe.

    (B) Real-parse backstop (runs only when node + the helper + the
    dashboard deps are present — i.e. on the local VM where the analyzer
    runs). The check shells out to `dashboard/scripts/mdx_parse_check.mjs`
    (cwd = `<repo>/dashboard`, body on stdin) which runs the exact
    `mdast-util-from-markdown` parse the dashboard's MDXEditor 4.0.1 runs,
    with the SAME extension set (mdxJsx + mdxMd + the HTML-comment
    extension + gfm-table + strikethrough + highlight-mark). If that real
    parse reports a failure, the check FAILs with the parser's message +
    line/col EVEN IF every regex passed — this is what makes the verifier
    authoritative and subsumes the narrow regex patch. When node / the
    helper / the deps are unavailable the check falls back to regex-only
    and appends "(real MDX parse skipped: <reason>)" to the detail; it
    does NOT hard-fail solely because node is missing (CI without node
    must still run the regex layer). A real-parse failure is what
    surfaces in the dashboard as the amber "Could not parse" banner with
    a fallback raw-editor link — the uneditable-body symptom this check
    prevents.

    Incidents: task #382, 2026-05-28 (six Reproducibility autolinks broke
    the dashboard renderer); a same-day body with `p<0.05` in prose
    triggered the U+0030 variant; task #399, 2026-05-28 (a `<|im_start|>`
    token leaked through a table cell — the regex-only layer missed it,
    motivating the real-parse backstop).
15. Reproducibility committed-at-`<sha>` claims resolve — a conservative
    cross-check that any "committed at commit `<sha>`" claim in
    `## Reproducibility` paired with a repo-relative artifact path
    actually resolves in `git cat-file` (FAILs when the sha resolves
    but the path is absent; WARNs when the sha cannot be resolved;
    PASSes when no such claim is present).
16. Reproducibility lr matches plan — the learning rate stated in the
    `## Reproducibility` Parameters table must appear in the approved
    plan (the union of ALL `plans/v*.md` versions, resolved for
    `--issue <N>` / a `--file` sibling — not just the `plans/plan.md`
    symlink, which same-issue follow-up rounds re-point at a follow-up
    plan that may omit the training lr; incident #597). Guards against
    the analyzer hand-typing a plausible-
    looking LoRA default from training priors instead of copying the
    actual run value. Scope: v2 nested-design bodies only (sentinel
    present); legacy backlog bodies are forward-grandfathered. The
    check is a NO-OP PASS when it cannot reconcile (no parseable body
    lr, no plan on disk, or no parseable plan lr) so it never blocks a
    body it cannot judge. A genuine documented run-vs-plan deviation
    (an explicit "deviation from the plan" note in `## Reproducibility`)
    downgrades the FAIL to WARN. Incident: task #489 shipped
    `lr = 1e-4` in the Parameters table while the committed training
    script + plan §11 both ran `lr = 2e-6` — a 50x misprint on the
    single most load-bearing hyperparameter, missed by every reviewer
    because no check reconciled the table's VALUES against ground truth.
17. Reproducibility Context provenance row — v2 (sentinel) bodies carry
    a `**Context:**` boldface row in `## Reproducibility` shipping the
    run-context provenance: created/run dates, follow-up lineage, and
    the verbatim originating user prompt (or the literal `origin prompt
    not recorded` when none exists). Forward-only (adopted 2026-06-11):
    legacy (pre-sentinel) bodies PASS vacuously. A missing row FAILs
    only when recorded origin data exists — frontmatter `origin_prompt`
    or a `## Provenance` section in the sibling `original-body.md` —
    i.e. the body DROPPED data it had; with no recorded origin data the
    miss is a WARN (the row should still ship, stating the prompt was
    not recorded). v4 additionally requires a lineage token in the row —
    `[#K](...)`/bare `#K`, `fresh direction (no parent)`/`fresh (no
    parent)`, or a same-issue-follow-up-round clause — scanned on
    fence-stripped + blockquote-stripped text (missing → hard FAIL);
    v3/v2 keep label-presence-only. Spec:
    `.claude/skills/clean-results/SPEC.md` § `**Context:**` row.

Soft INFO (not enforced as PASS/FAIL; surfaced for orchestrator
visibility): the Goal-of-experiment frontmatter field — frontmatter
contains ``goal: <one sentence>``. The body-side ``## Goal`` H2 is
INTENTIONALLY NOT CHECKED HERE: it lives only in proposed/planning
bodies (enforced at /issue Step 0c, workflow.yaml §
gates.experiment_goal); clean-result bodies drop the visible H2 and
fold the Goal text into the TL;DR Motivation bullet. The frontmatter
``goal:`` field stays in the clean-result body for agent-facing
reference (planner, critic, follow-up-proposer all read it). This
verifier WARNs when the frontmatter field is missing but never FAILs —
non-experiment kinds and pre-Goal bodies legitimately omit it.

Bodies carrying a `<!-- legacy-sagan-card -->` sentinel are
grandfathered HTML — this verifier skips them with a PASS (the legacy
`verify_sagan_card.py` still applies to those).

────────────────────────────────────────────────────────────────────────
v3 redesign (2026-W24, sentinel `<!-- clean-result-v3 -->`)
────────────────────────────────────────────────────────────────────────

v3 bodies are FORWARD-ONLY: v2-sentinel and pre-sentinel legacy bodies
keep every behaviour above verbatim and are NEVER newly hard-FAILed by a
v3 rule. A v3 body drops `## Human TL;DR` and the `## TL;DR` umbrella in
favour of FIVE flat H2 sections, in order — `## Takeaways` /
`## What I ran` / `## Findings` / `## Data` / `## Reproducibility` — with
confidence in the H1 title tag only. The checks branch per generation:

- **check 2** (`check_required_sections`): v3 requires the five-H2 set in
  order; a `## Human TL;DR` or `## TL;DR` H2 in a v3 body is a hard FAIL
  (mirrors the stray-`## Details` FAIL). v2/legacy keep the three-H2 set.
- **check 3** (`check_tldr_labels`): for v3, dispatches to
  `check_v3_structure` — `## Takeaways` has 3-6 bullets (the
  AUTHORITATIVE bullet-count gate), `## What I ran` carries the
  `**Why:**` slot, `## Findings` has ≥1 `### ` finding. v2/legacy keep
  the Motivation-opens-TL;DR check.
- **check 3b** (`check_tldr_nested_structure`): stays v2-ONLY (a v3 body
  has no `## TL;DR` umbrella to shape); PASSes vacuously on v3.
- **checks 4 / 4b** (figure presence + URL): v3 scans `## Findings`.
- **check 6 / 16 / 17** (confidence-title-only / lr-matches-plan /
  Context provenance): gated on `is_nested_design()` = v2 OR v3, so all
  three keep running on v3.
- **check 10** (cherry-picked label): v3 scans `## Findings` + `## Data`.
- **check 11** (raw-completions link): v3 scans `## Findings` +
  `## Data → ### Generated` ONLY (Trained-on / Evaluated-with link
  JSONLs / probe banks — covered by check 18 — not raw_completions).
- **check 11b** (planned-vs-actual): v3 headline surface is
  `## Takeaways` + `## Findings`.
- **check 13** (narrative-flow WARN): v3 scans `## Findings`.
- **check 14** (`check_concerns_audit`): v3 mechanism 1 → `### ` findings
  under `## Findings` + `## Takeaways` bullets; mechanism 2 (Confidence
  paragraph) RETIRES for v3 (confidence is title-tag-only).

New v3-only checks (PASS vacuously on v2/legacy):

- **check 18** (`check_data_shape`): `## Data` has `### Trained on` /
  `### Evaluated with` / `### Generated` in order; each carries ≥1
  pinned complete-artifact link OR an explicit `n/a — <reason>` line.
- **check 19** (`check_data_subset_disclosure`): every example block
  (fenced OR `<details>`) inside `## Data` is preceded by a
  subset-disclosure line (`K of M rows, random sample` / `cherry-picked
  for illustration` / harmful-content sanitized form).
- **check 19b** (`check_data_unwrapped_example_table`, WARN only): a
  verbatim example row placed in `## Data` as a BARE inline GFM table —
  not wrapped in `<details>` and not in a fenced code block — that
  carries a project-internal condition / cell code (`C1`, `H2`,
  `BS_E0`, `Method A`, …) trips
  `audit_clean_results_body_discipline.py`'s condition-code scan with a
  spurious FAIL at Step 9a-bis (the audit exempts the fenced + `<details>`
  example forms, not the bare table). The WARN nudges the author to wrap
  the rows BEFORE the confusing downstream audit FAIL. Scoped to cells
  that WOULD match the audit's condition-code patterns so a benign
  composition / row-count summary table never WARNs. As of #1227 this
  check ALSO fires on v4 bodies, scanning `## Methodology` (label
  `Methodology unwrapped example table (v4)`) — the one member of this
  group that is not v3-only.
- **check 20** (`check_v3_word_caps`): the §4 conciseness caps —
  per-Takeaways-bullet ≤30 words (WARN), per-finding prose ≤120 WARN /
  ≥180 FAIL, figure caption ≤60 (WARN), total content prose ≤800 + 250
  per extra follow-up round (WARN-only). Counts EXCLUDE tables, fenced
  code, `<details>` bodies, captions. The Takeaways 3-6 bullet COUNT is
  owned by check 3's `check_v3_structure` (one authoritative count).
  (v4 twin `check_v4_word_caps`: same caps over Takeaways + Goal +
  Results, `## Methodology` excluded, plus a v4-only per-Takeaways-bullet
  ≥100-word hard-FAIL tier (#825); its round count reads
  `epm:same-issue-followup-run` markers and/or the footer round clauses,
  max-reconciled — the Rounds-table read binds v3 only. It needs the
  `issue` number for the events leg, so it is dispatched separately in
  `verify_text`, outside CHECKS; #921.)
- **check 21** (`check_body_params_subset_of_doc`): the body's
  load-bearing `## Reproducibility` Parameters rows are a SUBSET of the
  methodology doc §2 complete table. Binds when the doc path is supplied
  via `--methodology-doc <path>` (the orchestrator passes the issue
  worktree path at gate time, pre-merge) OR — on the `--issue` path —
  when `docs/methodology/issue_<N>.md` resolves on disk (post-merge
  promote-time verify, where the doc is already on `main`); NO-OP PASS
  when neither resolves.

Generation-agnostic checks (run on v2 AND v3 — the inline-figure +
`## Reproducibility` shapes the scan keys off both carry):

- **check 22** (`check_figure_url_sha_matches_repro`): each inline figure
  URL's commit SHA must match the SHA the `## Reproducibility`
  `**Figures:**` bullet pins that figure to. A SHAPE-CONSISTENCY check
  comparing the two SHAs the body already carries (no git, no network):
  the inline raw-GitHub URL `.../<url_sha>/figures/issue_<N>/<file>.png`
  vs the per-figure `` `<basename>` at [commit] `<sha>` `` claim (with an
  `` all others at `<sha>` `` catch-all default). Across follow-up rounds
  a regenerated figure's inline URL and its Reproducibility claim can
  drift apart while every existence check still PASSes (the URL's own sha
  resolves), shipping a body whose inline image points at a different
  commit than `## Reproducibility` claims. SHAs are compared
  prefix-compatibly (the claim may be abbreviated, the URL is always full
  40-char). A figure with NEITHER an explicit claim NOR a default is
  out of scope (SKIP, never FAIL). NO-OP PASS when there is no
  Reproducibility section, no inline figure URL, or no figure-sha claim.
  Incident: task #537 `predictor_bakeoff_complete_null` shipped inline
  `5ad30c2…` against a Reproducibility claim of `c539920…` — caught by
  hand at round-3 interp-critique, mechanizable into this check.

- **check 23** (`check_hf_url_resolves`): every HF Hub revision-pinned
  `/tree/<sha>/<path>` or `/blob/<sha>/<path>` URL in the body must
  resolve to ≥1 file at the cited revision, probed via a BOUNDED direct
  tree-endpoint GET (`_hf_tree_get` — a single `get_session().get(...,
  timeout=...)` of the path's parent dir, NOT the unbounded recursive
  whole-repo `list_repo_files`, #733).
  Extends the #507 existence-protection class (check 4b inline figures,
  check 8b same-repo Reproducibility links) to HF Hub links, which check 8
  deliberately left shape-checked only — their existence IS decidable via
  the Hub API. A path pinned to a revision that predates the upload
  resolves to ZERO files; that dead pin slips through every other check
  (shape-valid, sha-pinned, real repo). Body-wide + fence-stripped scan
  (HF links live in `## Reproducibility`, `## Data`, and cherry-picked
  prose under `## TL;DR` / `## Findings`); only hex-pinned revisions are
  probed (moving refs like `/tree/main` are check 8's shape concern).
  FAIL-SOFT: only a successful zero-file listing — or a definitive
  repository/revision-not-found — is a FAIL; the `EPM_VERIFY_BODY_NO_HF=1`
  offline fence (set suite-wide by `tests/conftest.py`), a missing
  `huggingface_hub`, and any network / auth error surface as an
  `unverified` note on the PASS line, never a FAIL. Incident: task #537's
  `**Artifacts:**` "415 bakeoff intermediates" link pinned to `db3662ae`
  (the main-grid revision, predating the bakeoff round) resolved to 0
  files — caught by hand at round-1 clean-result-critique.

- **check 24** (`check_figure_text_vs_body_tokens`, WARN): figure-embedded
  text must not carry strings the body prose softened away, or chance/N
  values that disagree with the body's figure caption. A round-1 numeric /
  overclaim fix is routinely applied to body prose but MISSED in the
  figure-generation script's hardcoded title / annotation strings, so the
  regenerated figure silently disagrees with the body and only the
  multimodal interpretation-critic catches it — pushing the fix to a later
  round. This check reads each referenced figure's sibling `.meta.json`
  (from the git tree at the URL's commit sha via `git show`, fail-soft),
  flattens its text-bearing leaves (provenance keys dropped), and WARNs on
  (a) a `<a>/<b>` fraction in the figure text whose same-numerator
  counterpart in the body caption uses a DIFFERENT denominator (the
  `1/30` vs `1/29` case), or (b) any string from the configured stale-token
  list (`STALE_FIGURE_TOKENS` plus an optional `~/.eps-stale-tokens.json`).
  WARN, never FAIL (the multimodal critic owns the substantive read). NO-OP
  PASS offline / `--body-stdin`, with no same-repo sha-pinned figures, or
  no sidecar carrying scannable text. At most one `git show` per unique
  figure. Incident: task #667 round-2 interp-critique caught two stale
  figure-embedded strings (`1/30`→`1/29`, "geometrically real") the body
  prose had already fixed at round 1.

- **check 25** (`check_audit_availability_claims_match_hf`): a prose claim
  that a data artifact "was not uploaded" / "cannot be audited" /
  "unavailable for audit" — or, since #942, a LIVE quota-hold denial
  ("remain(s) on the pod", "quota-held", "under/pending/behind/blocked on
  the ... quota hold", "upload 403"; present-tense/stative only, resolved
  narratives never fire) — must NOT be contradicted by that artifact
  actually existing on the HF data repo. Scans the fence-stripped body for
  a line carrying BOTH an availability-denial phrase AND a known
  data-artifact class spelled in PROSE (raw-completions / install-probe /
  training-mix / on-policy pool / analysis-tensor / unreduced-activation-
  store / reduced-store-or-summary / fitted-map — hyphen/space/singular
  variants mapped to the canonical HF-path token
  `raw_completions` / `install_probes` / `mixes` / `onpolicy_pools` /
  `analysis_tensors` / `unreduced` / `reduced` / `maps`); for each, lists
  the body's HF Hub revision-pinned
  URLs (the check-23 set) ONCE per (repo, sha) and asks whether ≥1 file
  UNDER the URL's path-prefix carries the canonical token as an
  ALPHANUMERIC-BOUNDARY path component at ANY depth (`/` and `_` count as
  boundaries; never a bare substring, so `reduced` cannot match
  `unreduced/`; a denial usually links the repo TREE ROOT while
  the file lives several levels down at `<root>/…/<token>/…`, the #653
  shape). If ANY HF URL yields such a file via a BOUNDED direct
  tree-endpoint GET (`_hf_tree_get`, self-paginated under a page/time cap;
  NOT the unbounded `list_repo_files`, #733), the denial is false → FAIL.
  FAIL-SOFT (same semantics as checks 8b/23): the
  `EPM_VERIFY_BODY_NO_HF=1` offline fence, a missing `huggingface_hub`, and
  any network / auth / HTTP error surface as an `unverified` note on the
  PASS line, never a FAIL; only a SUCCESSFUL listing returning ≥1 matching
  file is the FAIL. Vacuous PASS when there is no denial-near-artifact line
  or no HF Hub revision-pinned URL to reconcile against — so the check
  protects ONLY bodies that themselves pin a covering HF URL: a false
  denial in a body with no covering revision-pinned URL still PASSes
  vacuously (pre-existing check-25 architecture shared by every denial
  family and artifact class; NOT closed by the #942 vocabulary extension —
  the incident-time #813 v1 body had no covering URL and would still
  PASS). Body-wide +
  fence-stripped scan (the denial prose lives in `## Methodology` /
  `## Results` / the Reproducibility footer). Incidents: task #653 round 6
  asserted the per-cell install-probe firing/non-firing completions "were
  not separately uploaded ... cannot be audited at the record level" while
  those files DID exist on HF under the body's own linked data-repo tree —
  caught by hand at round-6 interp-critique, mechanizable into this check;
  task #813 v1 asserted the unreduced store / reduced summaries / fitted
  maps "remain on the pod under an HF public-storage quota hold (upload
  403)" while all 24,206 files were on HF at the body-pinned revision —
  the quota-hold denial family + `unreduced`/`reduced`/`maps` classes +
  the module-level line pre-filter (`_AUDIT_LINE_PREFILTER_RE`) + the
  boundary-match fix were added for it (#942).

- **check 26** (`check_figure_panel_prose_vs_sidecar`): a figure's
  what-is-plotted prose (scoped to its enclosing `### <result>` H3 + the
  caption immediately after) must NOT claim a plot kind in a named panel
  position (a "right panel scatter") or a per-unit dot/point overlay (a
  "per-bank dots overlaid") that the figure's `.meta.json` sidecar — read
  strictly by URL stem from the git tree at the URL's commit sha (sibling
  `_read_figure_meta_json`, the PARSED-dict counterpart of check 24's
  `_read_figure_meta_text`) — provably lacks (the sidecar's `_kind` count of
  that element is 0). Conservative: a kind word is checkable only when it
  co-occurs with a panel-position or overlay word; a bare "the bars show ..."
  yields no claim → PASS. NO panel-count claim is made (`_group` is a
  per-series index, not a panel index — a 4-panel A7 figure has 22 `_group`
  values). When the sibling sidecar does not resolve at the cited sha BUT the
  PNG itself does (`_git_object_exists` returns `pass`), the check FAILs loud
  — the silent fallback to a DIFFERENT sidecar is the failure mode it exists
  to catch; when the PNG itself does not resolve it defers to check 22 (no
  double-FAIL). FAIL, never WARN (distinct from check 24). Vacuous PASS
  offline / `--body-stdin`, with no inline figures, or no panel/series prose
  claim. Incident: task #683 round-1 interp-critique false-FAILed off a
  wrong-sidecar fallback the verifier could not mechanically detect.

- **check 28** (`check_figure_label_codes`, WARN): rendered figure text read
  from each inline same-repo sha-pinned figure's sibling `.meta.json` (parsed
  via `_read_figure_meta_json`, check 26's reader) must not carry opaque
  config-code tokens — `@L<digits>` layer pins (`ctx_blk_max@L12`) or
  regime-code slugs (snake tokens >=3 segments or digit-bearing:
  `ans_uhdr_max`, `sw_eng_C1`, `cond_4`; 2-segment all-alpha metric names
  like `log_prob` stay allowed). Plain-English condition names are the
  project rule end to end; config slugs belong in the Repro config row /
  provenance keys. Scans string VALUES only (provenance-keyed subtrees
  pruned via `_META_PROVENANCE_KEYS`) plus dict keys containing internal
  whitespace (axis-label-keyed data rows) — identifier keys (`_kind`,
  `cell_slugs`, translation-map slug keys) are never visited; PATH-SHAPED
  strings / words (a file path or URL) are exempt, but a slash-separated
  rendered label (`ctx_blk_max / ans_uhdr_max`) carries whitespace and IS
  scanned. Coverage = sidecar-CARRIED strings only — known residuals: a
  slug-bearing figure TITLE with no ad-hoc sidecar echo is invisible (the
  canonical `savefig_paper` writer never serializes titles; PNG-pixel text
  stays the multimodal critics' substantive read), a bare slug used as a
  whitespace-free column KEY is unscanned, and a slug inside a path-shaped
  word is exempt. WARN, never FAIL; FAIL-SOFT on a missing / unparsable
  sidecar (check-24 convention, NOT check 26's loud missing-sidecar FAIL);
  NO-OP PASS offline / `--body-stdin`, with no inline figures, or no
  scannable same-repo sidecar. Incident: task #920's
  `winning_cell_scatter.png` reached the 9a-bis gate titled
  `ctx_blk_max@L12 x ans_uhdr_max@L12` after three review passes each
  deferred it as a cosmetic nit, costing a REVISE round.

- **check 29** (`check_figure_tracked_at_head`, WARN): every body-linked
  same-repo `figures/issue_<N>/...` figure path must still be tracked on a
  LIVE local ref — HEAD of the (main-pinned) repo root or the issue's local
  branch family `issue-<N>` / `issue-<N>-*` (one `for-each-ref` + one
  `ls-tree -r --name-only <ref> -- figures/issue_<N>/` per ref per unique
  issue dir; never per-URL — <=5 subprocesses for a typical body). Three
  states per path: tracked at HEAD → clean PASS; BRANCH-ONLY (on >=1 family
  ref, absent from HEAD) → PASS with an explicit disclosure note (expected
  pre-merge; names the `git restore --source=<pinned-sha>` recovery for the
  post-merge case); missing from every successfully-probed ref → the
  incident-class WARN, never FAIL (a pinned raw URL is immutable and keeps
  rendering after tracking loss — drift hygiene, not a broken body).
  Conservative: any failed probe for an issue dir degrades it to a skip
  note (a narrowed ref set must never manufacture a WARN); vacuous PASS
  with no same-repo `figures/issue_<N>/` URLs or an unresolved repo root.
  The issue number comes from the path itself, so cross-issue figure links
  check against their own branch family. Incident: task #841 — three
  `figures/issue_841/` stems tracked at the pinned sha `4824a567aa` but
  untracked at branch HEAD; the loss was invisible to every check (#964).
- **check 30** (`check_hf_file_count_claims`, WARN): numeric file-count
  claims adjacent to hex-pinned HF `/tree/<sha>` markdown links ("N files" /
  "N shards" inside the link text; a parenthetical opening with the
  count-noun immediately before the link; or, in a parenthetical
  immediately AFTER the link, the two anchored phrases "N files
  (listed )?at the pinned revision" — whole-prefix — and "N files
  (listed )?per namespace" — each backtick `dir/` namespace named in the
  link TEXT is probed at `<link-prefix>/<ns>` and must hold exactly N
  files; no extractable namespaces → an `unverified` note, never a WARN;
  or a backtick `dir/` sub-path token whose parenthetical OPENS with the
  count-noun, bound to the nearest preceding hex-pinned `/tree/<sha>`
  markdown link on the same footer row — bracket-free, newline-free,
  ≤400-char gap — and probed at `<link-prefix>/<sub-path>`, #1143)
  must match a files-only scoped Hub tree count at the pinned revision —
  the same bounded raw tree-endpoint probe checks 23/25 use (#733; never
  the SDK `list_repo_tree` / `list_repo_files`), counted EXHAUSTIVELY with
  folder entries excluded. Per-namespace-qualified counts are excluded
  from the link-text / paren-before-link patterns by a negative lookahead
  (wrong whole-prefix semantics). Mismatch → WARN, never FAIL (there is no
  `passed=False` path); shard claims are one-sided (WARN only when
  claimed > file count — a shards prefix legitimately also holds a
  manifest/sidecar). Every non-definitive probe outcome (offline fence /
  missing `huggingface_hub` / 429 / network error / `not_found` /
  page-time cap / the per-body `_HF_COUNT_MAX_PROBES` cap, shared across
  whole-prefix + per-namespace probes) surfaces as an `unverified` note on
  a PASS line — a PARTIAL count never grounds a WARN. Vacuous PASS with
  ZERO Hub probes when no count claim sits adjacent to an HF tree link.
  Incidents: task #931 shipped 528/10/3/198 where the pinned tree holds
  515/9/2/197 — folder entries counted as files (#1008); task #833 shipped
  "908 files listed per namespace" where each namespace holds 891 blobs +
  17 directory entries (#1088); task #1112 shipped "(7,372 files: …)"
  after a backtick `raw_completions/` token in the pinned-bucket footer
  row where the scoped tree holds 7,165 files + 207 folders (#1143).

- **check 31** (`check_orphaned_per_unit_figures`, WARN): the INVERSE
  direction of checks 4b/22/29 (which verify what the body CITES) —
  enumerate what the body's OWN cited figure SHAs contain under this
  task's `figures/issue_<N>/` (one `git ls-tree` per unique (SHA, dir)
  pair; no network) and WARN on any committed PNG whose basename stem
  matches `per[-_]?(context|unit|cell)` (case-insensitive, word-start
  lookbehind; `indiv` deliberately EXCLUDED — it names the per-question
  REGIME in this project, not a per-unit view) that no body image URL
  references (repo-relative path equality — SHA-independent, so a
  re-pinned embed still counts) and whose stem appears nowhere in the
  body text (the prose disclosure/exemption escape — naming the file
  silences the WARN). Issue-scoped: with `--issue` / a numeric-parent
  `--file` ONLY this task's dir is scanned (a cross-issue embed never
  surfaces another task's orphans); `--body-stdin` falls back to
  per-cited-dir scanning. WARN, never FAIL (prose-stated per-unit
  exemptions are legitimate; clean-result-critic Lens 11 stays the
  substantive owner — this is its mechanical backstop). Fail-soft:
  unreachable/unknown SHA → that SHA silently skipped (counted in the
  PASS detail, never a WARN); repo unresolved / no cited same-repo
  figure URLs → skip/vacuous PASS. Dispatched OUTSIDE the body-only
  CHECKS list (needs `issue`; the check-20/#921 precedent). Incident:
  task #928 — the round-committed per-context companion
  `mlp_indiv_percontext_delta.png` sat unreferenced at a body-cited SHA
  and reached the LM critic as a Lens 11 blocker. (#1011)

- **check 32** (`check_hf_adjacent_file_claims`, WARN): backtick FILENAME
  claims adjacent to hex-pinned HF `/tree/<sha>` markdown links — the
  filename-membership sibling of check 30's count claims — must appear by
  exact BASENAME, any depth, in a scoped listing of the pinned prefix (the
  same bounded raw tree-endpoint probe checks 23/25/30 use, #733; never the
  SDK `list_repo_tree` / `list_repo_files`). Two anchored shapes only,
  precision over recall: a parenthetical immediately AFTER the link (the
  #952 shape; check 30's paren is BEFORE), and a dotted backtick token
  inside the link TEXT. Named recall sacrifices: backtick filenames BEFORE
  the link (corpus misattribution evidence), paren-after-`/blob/` (check 23
  validates the full blob path), relative-path / brace / glob tokens, and
  the any-depth basename collision that can mask a wrong-PATH claim.
  Missing basename → WARN, never FAIL (no `passed=False` path); each WARN
  carries its shape tag (`PAREN`|`LINKTEXT`). Every non-definitive probe
  outcome (offline fence / missing `huggingface_hub` / 429 / network error
  / `not_found` — check 23 owns the dead-pin FAIL / page-time cap / the
  per-body `_HF_MEMBER_MAX_PROBES` cap) surfaces as an `unverified` note on
  a PASS line — only a SUCCESSFUL EXHAUSTIVE listing lacking the basename
  grounds a WARN. Vacuous PASS with ZERO Hub probes when no backtick file
  claim sits adjacent to an HF tree link. Incident: task #952 r1 claimed
  `divergence_bank_queries.json` at the pinned eval_results@5b62649 tree
  while the file lives only in git (#1016).

- **check 33** (`check_figure_prose_numerics_vs_sidecar`, WARN): bolded
  DECIMALS in a figure's what-is-plotted prose window — the
  previous-figure-bounded beat-1 slice of its enclosing `### <result>` H3
  plus the blockquote caption (`_beat1_prose_window`) — must each appear
  among the numeric values the figure's sha-pinned `.meta.json` sidecar
  records as plotted (`points`/`rows` rows, key-agnostic numeric leaves).
  PER-NUMERIC firing: WARN when >=1 bolded decimal matches NO plotted value
  under printed-precision half-ulp rounding + sign-insensitive + percent
  (x100 / /100) variants + a sci-notation relative-tolerance branch; percent
  variants never match a grouped-bar layout x-position, and a bolded decimal
  matching an EARLIER same-H3 figure's plotted values is suppressed as
  cross-figure interpretation bleed. Integers / version-shaped tokens are
  never scanned (bolded decimals only); unbolded caption numerics are a
  named recall sacrifice (precision over recall, the check-32 posture).
  Per-figure opt-out: the literal `<!-- prose-numerics: derived -->`
  anywhere in the figure's scanned window (beat-1 prose or caption). WARN,
  never FAIL; silent skip on missing / unparsable / truncated sidecars (the
  check-24 convention, NOT check 26's loud missing-sidecar FAIL) and NO-OP
  PASS offline / stdin / no inline figures. Incident: #825 r1 (task #1107)
  — prose+caption cited transfer fractions 0.057/0.109 while the pinned
  figure plotted 0.231 / a -4.53-clipped bar; checks 24 (rendered text) and
  26 (structure) are blind to plotted-NUMBER drift by construction.

- **check 34** (`check_figure_beat_claims_vs_sidecar_text`, WARN,
  FORWARD-ONLY): beat-1 series-structure claims — "both <up to 3 words>
  arms/series/conditions/models/lines/curves" and "one
  bar/point/dot/marker/line/curve per <unit>" (the two literal #1092
  defect-(b) phrasings; paraphrases miss by design) — must not contradict
  the series structure the figure's sha-pinned sidecar demonstrably
  renders. Fires ONLY when the sidecar carries the `meta["text"]`
  rendered-text block the current `savefig_paper` writes (pre-capture
  sidecars silently skip — no retroactive WARN on existing bodies).
  Contradiction-only: Class A WARNs only when >=1 evidence basis is
  available AND every available basis reads <=1 (series labels / legend
  entries / bar point-rows — never `n_series`, one `BarContainer` holds a
  whole bar pair — / distinct line artists / distinct scatter artists /
  total artist groups); a figure with no basis (e.g. an unlabeled
  single-artist scatter) SKIPS. Class B requires a points payload and
  WARNs when the mapped `_kind` renders <=1 element (line/curve counted as
  distinct ARTISTS via `_group` — vertex rows are unsound). Window =
  check 33's narrow `_beat1_prose_window`. WARN never FAIL; silent skip on
  missing / unparsable / text-less sidecars (check-24 convention); NO-OP
  PASS offline / stdin / no inline figures. Incident: #1092 (9a-bis r1) —
  a beat claimed "both input arms" / "one bar per re-fit item" against a
  figure rendering neither, passing every mechanical figure check. (#1255)

- **check 35** (`check_cross_issue_reuse_provenance`, FAIL/WARN, v4-only,
  #1256): cross-issue reuse pins in the committed
  `eval_results/issue_<N>/**/*.json` result-JSON `metadata` must be
  declared in the body (canonical slot: the footer `Reused:` bullet,
  SPEC.md § `**Artifacts:**`). Tier 1 (FAIL): a `metadata` key matching
  `hf_rev_<M>` / `hf_rev_<M>_<tag>` with M != N whose pinned revision has
  no >=7-hex-char prefix token anywhere in the body (non-hex branch/tag
  pins fall back to a `#M` / `/tasks/M` / `issue<M>_` mention). Tier 2
  (WARN): a `\bissue<M>_` path token in `metadata.input_shas` keys/values
  or PATH-LIKE (`/`-bearing) `metadata.args` string values, M != N, with
  neither the `issue<M>_<slug>` segment nor a `#M` / `/tasks/M` mention in
  the body. Graceful PASS-skips: not-v4 (forward-only), issue unknown
  (stdin), `EPM_VERIFY_BODY_NO_EVAL_SCAN=1`, eval root unresolved
  (is_warn, the #732 convention), no pins found. Corrupt / unreadable /
  oversize (>50 MB stat guard) JSONs are skipped silently — `issue_810`
  carries 138-208 MB JSONs and `issue_811` is a ~14.7 GB dir, so the
  guard + a substring pre-filter are load-bearing. Grounding (corpus scan
  2026-07-11, ~90,858 committed eval JSONs): the tier-1 key shape exists
  in exactly 1 file repo-wide (the #1092 incident file), and the bare
  `issue<M>_` pattern appears in >=10,028 files — hence the tier-2
  restriction + WARN severity. Documented residuals: same-revision
  multi-artifact reuse (one firing pin per round suffices; LM
  clean-result-critic Lens 5 stays the semantic backstop), and
  `paper: true` tasks (gated by verify_paper.py, never this verifier).
  Incident: #1092 round 3 — `hf_rev_779_labels` pinned in
  `transfer_reads.json` metadata with the reuse undeclared in the footer
  survived to the LM critic. Dispatched OUTSIDE the body-only CHECKS list
  (needs `issue` + eval-root resolution; the #732 precedent).

- **check 36** (`check_v4_result_paragraph_sentences`, WARN-only, v4-only,
  #1368): each prose PARAGRAPH inside a `### <result>` block runs 1-3
  sentences (SPEC § Conciseness caps (v4) / § Results three-beat 1-3-sentence
  beats / Lens 12 "any single paragraph runs >=4 sentences"). Paragraph = a
  maximal run of consecutive prose lines (blockquote captions, fenced code,
  `<details>` bodies, GFM tables, headings, images, HTML lines, list items,
  `---` rules excluded). Sentence boundary = `[.!?]+` before
  whitespace/end-of-paragraph after masking inline code, link targets, a
  small abbreviation list (`no.` only before a digit), decimals, and
  ellipses; semicolon chains count as one unit. WARN at >=4
  (`V4_RESULT_PARA_MAX_SENTENCES` = 3), NEVER FAIL — the bullets-over-prose
  call and the FAIL decision stay with the clean-result-critic (Lens 12).
  Forward-only: v3/v2/legacy/paper bodies never flagged. Calibration
  (plan-time, 7 real v4 bodies): fires on 4/6 results of the #1333
  plan-time body (sentence counts 5/5/4/5 — the motivating incident,
  matching the critic's flags verbatim), 0/8 on #922; 5 parked
  `awaiting_promotion` bodies WARN at 1/8-8/8 results (genuine register
  drift, hand-verified on #958). Incident: task #385 round 1 / #1333 — a
  5-sentence read paragraph the Claude critic PASSed burned a full LM
  critic round. (Numbered 36 because 28-35 are taken by the
  generation-agnostic checks.)

- **check 37** (`check_footer_reuse_bullets_pinned`, WARN, v4-only, #1370):
  every footer `- Reused ... from [#M](...)` bullet carries a
  revision/path pin — the body->pin sibling of check 35's metadata->body
  direction (#1315: two unpinned `- Reused ... from [#1090]` bullets were
  invisible to check 35's metadata-side trigger and survived to the LM
  critic). Bullet-scoped satisfiers: a revision-URL segment
  (`/tree|resolve|commit|blob/<7-40 hex>`), `@ <rev>` (optional backtick),
  a committed `eval_results/issue_<M>/` path, a SPEC-sanctioned WandB
  `/runs/<id>` URL, or a bare letter-bearing >=7-hex token; the
  from-link's own `#M` / `/tasks/M` NEVER satisfies (vacuity guard —
  every trigger bullet carries it by construction). WARN uniformly
  (corpus 2026-07-15: 40 trigger bullets across committed v4 bodies,
  4 unpinned — #810/#811/#833/#1112, all `awaiting_promotion`; a FAIL
  would newly block their re-verifies). Body-text-only: lives in the
  body-only CHECKS list, runs on stdin bodies, and is deliberately NOT
  fenced by `EPM_VERIFY_BODY_NO_EVAL_SCAN` (no eval scan to fence).
  Documented false-negative residual: a bullet quoting the CURRENT
  task's own code SHA satisfies the bare-hex form — LM Lens 5 keeps
  owning semantic pin-correctness.

- **check 38** (`check_linked_not_embedded_figures`, WARN, v4-only, #1371): a
  non-image markdown LINK in the footer-truncated v4 `## Results` section
  whose URL carries a `figures/issue_<N>/*.png` path, where that figure is
  embedded as an image nowhere in the body. Pipeline: `_v4_results_body` →
  `_prose_layer` (fences + `<details>` stripped — a quoted or dropdown-tucked
  link never WARNs) → mask image embeds with `_IMAGE_RE.sub("")` → scan the
  remaining `_MD_LINK_RE` links for `_LINKED_ISSUE_PNG_RE` paths → subtract
  the whole-body EMBEDDED set (the same path capture run over every markdown
  image URL AND every HTML `<img src=…>` anywhere in the body — raw-GitHub /
  blob / relative alike; SHA-independent, case-folded path equality).
  Own-issue scoping when `issue` is known (a cross-issue link is a legitimate
  reference); `issue=None` (`--body-stdin`) falls back to every issue dir
  (check 31's documented fallback caveat). PNG-only (a PDF cannot render
  inline — a PDF link is the correct form). WARN-only, never FAIL. Closes
  check 31's stem-in-prose blind spot: the link itself puts the stem in the
  body, silencing 31's only relevant WARN. Named recall sacrifices: the
  embedded set scans the UNSTRIPPED whole body (a fenced example embed can
  silence a real WARN — false-negative direction only); reference-style links
  `[text][ref]` are not matched. Incident: #1315 result 4 linked a committed
  per-row scatter grid instead of embedding it; only clean-result-critic
  Lens 11 caught it. Dispatched OUTSIDE the body-only CHECKS list (needs
  `issue` for own-dir scoping; the check-31/#1011 precedent). (Numbered 38 —
  36/37 were taken by #1368/#1370 while this check was in review.)

Harmful-content carve-out: checks 18/19 accept the sanitized excerpt
form (`[truncated — harmful-content row; verify at <path>, row <i>]`)
exactly as checks 10/11 do today.

Usage:

    uv run python scripts/verify_task_body.py --issue <N>
    uv run python scripts/verify_task_body.py --file path/to/body.md
    uv run python scripts/verify_task_body.py --body-stdin
    uv run python scripts/verify_task_body.py --file body.md \\
        --methodology-doc docs/methodology/issue_<N>.md   # binds check 21

Exits 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

# Bring the task_workflow module in for --issue lookups.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

# ─── Spec constants ────────────────────────────────────────────────────────

# 2-content-section model (migrated 2026-W22, task #454). Three required
# H2s in order. `## Human TL;DR` is the FIRST required section — Thomas's
# own take, drafted by the analyzer as a real populated first-pass
# (Headline / Takeaways / How this updates me) and refined by Thomas
# before sending to mentor. The literal `placeholder` is a DEFECT here,
# not the intended content; check 0 only FAILs when the WHOLE body
# collapses to a stub token, so a populated Human TL;DR PASSes as before.
# `## TL;DR` is the LessWrong-style narrative (opens with an `### Motivation`
# H3 or `**Motivation:**` bullet, then one `### <finding>` H3 per result
# with one inline figure). `## Reproducibility` is the agent-facing
# appendix that ABSORBS the Parameters table + Confidence sentence.
REQUIRED_H2_SECTIONS = ["Human TL;DR", "TL;DR", "Reproducibility"]
# H2 sections that are REJECTED in new bodies. Under the
# 2-content-section spec, `## Details` is folded into per-result H3s
# under `## TL;DR` and `## Figure` is replaced by inline figures inside
# each result H3. A stray `## Details` or `## Figure` H2 in a NEW body
# is a hard FAIL (check 2), forcing clean migration. Legacy bodies
# pre-2026-W22 are forward-grandfathered (the verifier never re-runs
# over them).
RETIRED_H2_SECTIONS = ["Details", "Figure"]
# TL;DR opens with a Motivation block — either an `### Motivation` H3
# (preferred new shape) or a `**Motivation:**` boldface bullet (legacy
# form, still accepted). The retired `What I ran` / `Results` required
# bullets are no longer enforced — the new shape uses per-result
# `### <finding>` H3s checked via the figure / cherry-picked / qualitative-
# data-link checks.
TLDR_BULLETS_REQUIRED = ["Motivation"]
TLDR_BULLETS_OPTIONAL: list[str] = []
REPRO_SUBGROUPS = ["Artifacts", "Compute", "Code"]

LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->"

# Nested-design (v2) clean-result bodies carry this sentinel. The analyzer
# emits it on draft. The verifier uses it to gate the nested-TL;DR-shape
# requirements (presence + order of `### Motivation` / `### What I ran` /
# `### Findings` with ≥1 `#### ` child) AND to accept confidence-title-only
# (no body `Confidence:` sentence). Bodies WITHOUT this sentinel keep the
# prior post-#454 behavior and are NEVER hard-FAILed by the nested-shape
# rule — forward-only migration.
CLEAN_RESULT_V2_SENTINEL = "<!-- clean-result-v2 -->"

# v3 redesign (2026-W24). NEW bodies carry this sentinel. The v3 body
# drops `## Human TL;DR` and the `## TL;DR` umbrella entirely in favour
# of five FLAT H2 sections — `## Takeaways` / `## What I ran` /
# `## Findings` / `## Data` / `## Reproducibility` — with confidence in
# the H1 title tag only. Forward-only: v2-sentinel and pre-sentinel
# legacy bodies keep their existing verification behaviour verbatim and
# are NEVER newly hard-FAILed by any v3 rule. See
# `.claude/skills/clean-results/SPEC.md` § "v3 body shape" and §7 of
# `.claude/plans/clean-result-v3-redesign.md`.
CLEAN_RESULT_V3_SENTINEL = "<!-- clean-result-v3 -->"

# v3 required H2 sections, in order. A `## Human TL;DR` H2 in a v3 body
# is a hard FAIL (mirrors the stray-`## Details` FAIL) — the v3 shape
# retired the model-written casual summary.
V3_REQUIRED_H2_SECTIONS = ["Takeaways", "What I ran", "Findings", "Data", "Reproducibility"]
# H2 sections REJECTED in a v3 body. `## Human TL;DR` and `## TL;DR` are
# folded away (Takeaways replaces the skim function; the per-result
# narrative moves to flat `## Findings`). The legacy retired pair
# (`## Details` / `## Figure`) stays rejected too.
V3_RETIRED_H2_SECTIONS = ["Human TL;DR", "TL;DR", "Details", "Figure"]
# v3 `## Data` required subsections, in order. Each is an H3.
V3_DATA_SUBSECTIONS = ["Trained on", "Evaluated with", "Generated"]

# v4 redesign (2026-W26). NEW bodies carry this sentinel. The v4 body
# uses FOUR flat H2 sections — `## Takeaways` / `## Goal` /
# `## Methodology` / `## Results` — plus a `**Repro:**` / `**Context:**`
# bold-label footer (NOT H2s). The v4 redesign folds the v3 `## What I ran`
# + `## Data` content into an expanded `## Methodology` (which also absorbs
# the entire former standalone methodology doc) and collapses the
# per-result `## Findings` skeleton into a strict three-beat `## Results`.
# Confidence stays in the H1 title tag only. Forward-only: v3 / v2 /
# pre-sentinel legacy bodies keep their existing verification behaviour
# verbatim and are NEVER newly hard-FAILed by any v4 rule. See
# `.claude/skills/clean-results/SPEC.md` § "v4 body shape".
CLEAN_RESULT_V4_SENTINEL = "<!-- clean-result-v4 -->"

# v4 required H2 sections, in order. A v3 content H2 (`## What I ran` /
# `## Findings` / `## Data` / `## Reproducibility`) OR a retired earlier H2
# (`## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure`) in a v4 body
# is a hard FAIL (forces clean migration to the four-H2 shape).
V4_REQUIRED_H2_SECTIONS = ["Takeaways", "Goal", "Methodology", "Results"]
# H2 sections REJECTED in a v4 body — the v3 content H2s (folded into
# `## Goal` / `## Methodology` / `## Results` + the footer) plus the
# earlier retired pairs.
V4_RETIRED_H2_SECTIONS = [
    "What I ran",
    "Findings",
    "Data",
    "Reproducibility",
    "Human TL;DR",
    "TL;DR",
    "Details",
    "Figure",
]

CONFIDENCE_LEVELS = {"LOW", "MODERATE", "HIGH"}

# Sentinel substrings that indicate a placeholder slipped through.
SENTINEL_SUBSTRINGS = ["TBD", "{{", "see config", "default"]

# `default` is flagged ONLY in placeholder positions: a bare markdown
# table-cell value (`| default |`) or a label terminator (`chat template:
# default` / `**Chat template:** default` / `lr = default` at end of line
# or cell). Embedded prose uses — "default assistant", "default-context
# response cache", "the default column" — are substantive in this project
# (the default assistant is a core experimental condition, open-q 3.7;
# task #542 had to reword a clean body to dodge the old whole-word match).
# Horizontal whitespace only ([ \t]) so a match never spans lines; `\**`
# admits the bold-label row form (`**Label:** value`); optional backticks
# admit a code-formatted placeholder value.
_DEFAULT_PLACEHOLDER_RE = re.compile(
    r"\|[ \t]*`?default`?[ \t]*\|"  # bare table-cell value
    r"|[:=][ \t]*\**[ \t]*`?default`?[ \t]*(?:$|\|)",  # label terminator
    flags=re.IGNORECASE | re.MULTILINE,
)

# Minimum number of characters of rationale required AFTER the
# `Confidence: <level> —` dash on the confidence line.
MIN_CONFIDENCE_RATIONALE_CHARS = 20

# ─── v3 conciseness caps (check 20) ──────────────────────────────────────────
#
# The §4 conciseness table from the v3 redesign plan, encoded as named
# constants so tightening later is a one-line change. Calibration basis:
# the #517 → v3 conversion (`exemplars/v3-517.md`), a real,
# information-dense FOUR-finding body (3 per-trait findings + the
# cross-experiment-caveat finding) — a well-written hard case. Measured
# on that exemplar: longest finding prose ~95 words (well under the 120
# WARN cap), Takeaways bullets ~25-30 words (at/under the 30 cap), total
# content prose ~724 words across the four findings + Takeaways + What I
# ran. The per-finding WARN/FAIL caps (120/180) leave that body
# comfortable headroom; the total-prose budget is set to 800 so a
# well-written four-finding body clears it WARN-free while a bloated body
# (the original #517 v2 at ~2,400 prose words across the same findings,
# or a single finding padded past 180 words) trips the gates decisively.
# Counts EXCLUDE tables, fenced code blocks, `<details>` bodies, and
# figure captions (blockquote lines) — those are reference material, not
# the scannable prose the caps govern.
#
# `## Takeaways` bullet-count range (FAIL outside; authoritative count —
# the structure check defers to this so there is ONE count gate).
V3_TAKEAWAYS_MIN_BULLETS = 3
V3_TAKEAWAYS_MAX_BULLETS = 6
# Per-Takeaways-bullet word cap (WARN).
V3_TAKEAWAYS_BULLET_MAX_WORDS = 30
# Per-Takeaways-bullet hard-FAIL tier — v4 bodies ONLY (v3/v2/legacy stay
# WARN-only per the forward-only rule). A same-issue follow-up re-fold can
# accrete a paragraph-bullet that rides the 30-word WARN indistinguishably
# from a mild overrun (#825 r1: a 263-word bullet WARNed identically to a
# 35-word one); >=100 words is structural misuse of a bullet, not a
# tightening request.
V4_TAKEAWAYS_BULLET_FAIL_WORDS = 100
# Per-`### <result>` prose-paragraph sentence cap (WARN-only, v4 check 36,
# #1368/#1333): SPEC § Conciseness caps (v4) — the register is 1-3 sentences
# per paragraph (SPEC L837-839 / L424-432 / Lens 12); >=4 WARNs, never FAILs.
V4_RESULT_PARA_MAX_SENTENCES = 3
# Per-finding prose word cap (excl. caption / code / `<details>` bodies):
# WARN at the soft cap, FAIL at the hard cap.
V3_FINDING_PROSE_WARN_WORDS = 120
V3_FINDING_PROSE_FAIL_WORDS = 180
# Figure-caption word cap (WARN).
V3_FIGURE_CAPTION_MAX_WORDS = 60
# Total content prose (Takeaways + What I ran + Findings, excl. tables,
# fenced code, `<details>` bodies, captions): WARN-only base budget, plus
# a per-extra-follow-up-round allowance so a multi-round consolidated body
# is not forced to delete live findings to satisfy a total cap (§6.4 of
# the plan: the per-finding FAIL is the hard gate; this total is a nudge).
V3_TOTAL_PROSE_BASE_WORDS = 800
V3_TOTAL_PROSE_PER_EXTRA_ROUND_WORDS = 250

# ─── Result type ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    is_warn: bool = False  # WARN downgrades — counts as PASS for `passed`,
    # but rendered with a [WARN] tag.

    def render(self) -> str:
        tag = "WARN" if self.is_warn else ("PASS" if self.passed else "FAIL")
        line = f"  [{tag}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


# ─── Body splitting ────────────────────────────────────────────────────────


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        return {}, text
    fm_block = rest[:end]
    body = rest[end + len("\n---\n") :]
    try:
        fm = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, body


def find_h1_title(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return None


def find_h2_sections(body: str) -> list[tuple[str, int, int]]:
    """Return list of (section_name, body_line_start, body_line_end) for each H2.

    H2 lines inside fenced code blocks are ignored, so a pasted
    ``## Why this experiment`` inside a code fence cannot satisfy the
    verifier or the `task.py new` gate. Both triple-backtick (``` ```py``)
    and triple-tilde (``~~~text``) fence delimiters are recognized,
    matching CommonMark's relaxed rule.
    """
    lines = body.splitlines()
    h2_indices: list[tuple[str, int]] = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Toggle fence state on any line starting with ``` or ~~~ (with
        # optional info string, e.g. ```python or ~~~text). Matches
        # CommonMark's relaxed rule: an opening fence does not have to
        # be closed by an identical tag, but lines starting with ``` or
        # ~~~ flip the state.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## ") and not stripped.startswith("### "):
            h2_indices.append((stripped[3:].strip(), i))
    out: list[tuple[str, int, int]] = []
    for k, (name, start) in enumerate(h2_indices):
        end = h2_indices[k + 1][1] if k + 1 < len(h2_indices) else len(lines)
        out.append((name, start + 1, end))
    return out


def section_text(body: str, section_name: str) -> str | None:
    lines = body.splitlines()
    for name, start, end in find_h2_sections(body):
        if name.casefold() == section_name.casefold():
            return "\n".join(lines[start:end]).strip()
    return None


# Image markdown:  ![alt](path-or-url)
# Alt text may contain `[brackets]` (e.g. literal marker names like `[ZLT]`),
# so we allow a `]` inside alt as long as it is not followed by `(`. The URL
# group is captured for downstream resolvability checks (no parens inside URL).
_IMAGE_RE = re.compile(r"!\[(?:[^\]]|\](?!\())*\]\(([^)]+)\)")

# Markdown link: [text](url)
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# Backtick-wrapped inline code: `path/to/thing`
_CODE_RE = re.compile(r"`([^`\n]+)`")

# Fenced code blocks ```...```
_FENCED_RE = re.compile(r"^```[^\n]*\n(.*?)\n```", re.DOTALL | re.MULTILINE)


# ─── Sample-block heuristic helpers ───────────────────────────────────────


def _is_sample_fence(content: str) -> bool:
    """Return True if a fenced code block looks like sample model output.

    Mirrors the heuristic in verify_sagan_card.py::_is_sample_pre — completion-
    style if it contains a User/Assistant/Human/Model marker OR the body is
    long (> 200 chars). Otherwise it is probably a code/CLI snippet.
    """
    if re.search(r"\b(User|Assistant|Human|Model):", content, re.IGNORECASE):
        return True
    return len(content.strip()) > 200


def _iter_sample_fences(details: str) -> list[tuple[int, int, str]]:
    """Yield (fence_start_offset, fence_end_offset, content) for each
    fenced code block in `details` that is sample-output-like."""
    out: list[tuple[int, int, str]] = []
    for m in _FENCED_RE.finditer(details):
        content = m.group(1)
        if _is_sample_fence(content):
            out.append((m.start(), m.end(), content))
    return out


# A `<details>...</details>` block (optionally `<details open>`). The
# nested-design (v2) bodies often present cherry-picked training rows or
# eval probes as GFM TABLES inside a `<details>` block instead of fenced
# code blocks. Recognizing the table form means the cherry-picked-label
# and qualitative-data-link checks (10 + 11) enforce — not vacuously pass
# — on bodies like #432 that use the table form.
_DETAILS_BLOCK_RE = re.compile(
    r"<details\b[^>]*>(?P<inner>.*?)</details>", re.IGNORECASE | re.DOTALL
)
# Heuristic that a `<details>` inner block carries sample-completion-like
# content. We treat the block as "sample-like" when it contains EITHER
# (a) a GFM table delimiter row (`|---|---|`) suggesting a structured
# example table, OR (b) >200 chars of inner content (mirrors the fence
# heuristic). The first-column / row-type heuristic ("Row-type", "System",
# "User", "Assistant" headers) is intentionally NOT mandatory — #432's
# training-row table uses `Row | System prompt | User question | Assistant`
# which already satisfies the table-delimiter trigger.
_GFM_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?(?:\s*\|\s*:?-{2,}:?)+\s*\|?\s*$", re.MULTILINE)


def _is_sample_details(inner: str) -> bool:
    """Return True if a `<details>` inner block carries
    sample-completion-like content (table form or long text)."""
    if _GFM_DELIM_RE.search(inner):
        return True
    # Long enough to plausibly carry a structured example block.
    return len(inner.strip()) > 200


_SUMMARY_CLOSE_RE = re.compile(r"</summary\s*>", re.IGNORECASE)
_SUMMARY_OPEN_RE = re.compile(
    r"<summary\b[^>]*>(?P<text>.*?)</summary\s*>", re.IGNORECASE | re.DOTALL
)
# `<summary>` text patterns that signal the block is a COMPREHENSIVE
# enumeration (full list of eval inputs, complete schema, every
# condition, etc.), NOT a cherry-picked sample. The cherry-picked-label
# rule (check 10) and the qualitative-data-link rule (check 11) are
# about sample completions / illustrative rows; an exhaustive list of
# "The N eval questions" or "All N conditions" doesn't carry the
# sample-vs-population framing those checks enforce. Triggered by a
# summary opening with "The N <plural-thing>" or "All N <plural-thing>"
# (case-insensitive). Cherry-picked summaries like "5 example training
# rows" / "first 3 of 400 completions" stay sample-like.
_EXHAUSTIVE_SUMMARY_RE = re.compile(
    r"^\s*(?:the|all)\s+\d+\b",
    re.IGNORECASE,
)

# A FENCED code block has no `<summary>` to carry the exhaustive-enumeration
# signal; the equivalent disclosure lives in the prose prelude immediately
# above it (e.g. #538's "The 20 eval input questions are the same fixed set
# across every cell …"). Such a block is an eval-INPUT enumeration — the
# fixed list of prompts/questions the experiment runs ON — NOT a
# cherry-picked model-OUTPUT completion sample, so the cherry-picked-label
# (check 10) and qualitative-data-link (check 11) rules don't apply: there
# is no raw-completion artifact to link, because the block IS the input
# stimulus, not a generation.
#
# To avoid loosening the checks on genuine output samples, the skip requires
# a SINGLE prelude LINE that carries BOTH:
#   (a) an exhaustive-enumeration lead — "The N <thing>" / "All N <thing>" at
#       the start of that line, AND
#   (b) an eval-INPUT framing token later on the SAME line — naming the block
#       as the fixed set of eval/input questions/prompts (NOT
#       completions/outputs/responses).
# The two signals must co-occur ON ONE LINE (a single combined regex, not two
# independent `.search()` calls). This is deliberate: matching the lead and
# the framing token on DIFFERENT lines of the window would over-loosen — a
# cherry-picked OUTPUT block ("The 5 most extreme completions are shown
# below.") whose window also bleeds in an unrelated "(the 20 eval input
# questions are described above)" parenthetical would be wrongly skipped.
# Same-line co-occurrence is how every legitimate eval-input enumeration
# actually phrases it (e.g. #538: "The 20 eval input questions are the same
# fixed set …"). MULTILINE so the lead anchors at the start of ANY line of the
# prelude window (`_prelude_window` may leave leading blank/partial lines).
# A cherry-picked output prelude ("The 5 most extreme completions …") fails
# (b) and is still enforced; an eval-question prelude that omits the "The N"
# lead fails (a) and is still enforced (it must then carry a real link or the
# `not uploaded` escape like any other block). The `<details>` form keeps its
# own `_EXHAUSTIVE_SUMMARY_RE.match` summary-skip — left unchanged.
_EVAL_INPUT_ENUM_PRELUDE_RE = re.compile(
    r"^\s*(?:the|all)\s+\d+\b"
    # Gap between the lead number and the framing token: same line, and it
    # must NOT contain a competing OUTPUT head-noun
    # (completion/output/response/generation/sample/answer/reply). Without
    # this guard "The 6 completions … in response to the eval questions:"
    # would skip — the lead introduces the model's OUTPUTS, not an eval-INPUT
    # enumeration (review Minor-1, #538).
    r"(?:(?!\b(?:completion|output|response|generation|sample|answer|reply)s?\b)[^\n])*?"
    r"(?:"
    r"eval(?:uation)?[\s-]*(?:input[\s-]*)?(?:question|prompt|item|stimul)"
    r"|input[\s-]*(?:question|prompt)"
    r"|(?:fixed|same)\s+set\s+(?:of\s+)?(?:\w+\s+){0,3}?(?:eval|question|prompt)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _is_eval_input_enumeration_prelude(prelude: str) -> bool:
    """Return True if a fenced block's prelude prose marks it as an
    exhaustive eval-INPUT enumeration (the fixed set of questions/prompts
    the experiment runs ON), not a cherry-picked model-OUTPUT sample.

    Mirrors the `<details>` `<summary>` exhaustive-enumeration skip
    (`_EXHAUSTIVE_SUMMARY_RE`) for the fenced-code-block form. Requires a
    SINGLE prelude line carrying BOTH an exhaustive lead ("The N …" /
    "All N …") AND an eval-input framing token, so genuine output samples
    (and output preludes that merely mention eval questions elsewhere in
    the window) stay enforced.
    """
    return bool(_EVAL_INPUT_ENUM_PRELUDE_RE.search(prelude))


def _iter_sample_details(details: str) -> list[tuple[int, int, str]]:
    """Yield (block_start_offset, block_end_offset, inner_content) for
    each `<details>...</details>` block in `details` that looks like a
    sample-output block (table-form or long text) AND is NOT an
    exhaustive enumeration. Used by checks 10 + 11 to enforce the
    cherry-picked-label and qualitative-data-link discipline on
    nested-design (v2) bodies that present samples as `<details>`
    tables instead of fenced code blocks.

    Skip rule: a `<details>` block whose `<summary>` text starts with
    "The N <thing>" or "All N <thing>" is an exhaustive enumeration
    (full eval-question list, full schema, complete condition set),
    NOT a cherry-picked sample — the cherry-picked-label / qualitative-
    data-link rules don't apply. Example: #432's
    `<summary>The 20 eval questions (asked identically of all 28
    personas)</summary>` is the full eval-input enumeration, not a
    sample.

    The `block_start_offset` is positioned AFTER the closing
    `</summary>` tag (when one exists) so the `_prelude_window` helper
    walking back from this offset includes the `<summary>` text itself
    as part of the prelude. The summary line for a sample-flavored
    `<details>` block typically carries the cherry-picked disclosure
    ("5 example training rows", "first 3 of 400 completions");
    folding it into the prelude is what makes the cherry-picked-label
    check semantically correct for nested-design v2 bodies.
    """
    out: list[tuple[int, int, str]] = []
    for m in _DETAILS_BLOCK_RE.finditer(details):
        inner = m.group("inner")
        if not _is_sample_details(inner):
            continue
        # Skip exhaustive-enumeration blocks: their `<summary>` text
        # starts with "The N <plural>" or "All N <plural>".
        sm_open = _SUMMARY_OPEN_RE.search(inner)
        if sm_open is not None and _EXHAUSTIVE_SUMMARY_RE.match(sm_open.group("text")):
            continue
        # Move the recognized "block start" to after the closing
        # </summary>, when one exists, so the prelude window includes
        # the summary text.
        sm = _SUMMARY_CLOSE_RE.search(details, pos=m.start(), endpos=m.end())
        block_start = sm.end() if sm is not None else m.start()
        out.append((block_start, m.end(), inner))
    return out


def _iter_sample_blocks(details: str) -> list[tuple[int, int, str]]:
    """Yield ALL sample-output blocks under `details`: both fenced code
    blocks (`_iter_sample_fences`) and `<details>` table/long blocks
    (`_iter_sample_details`), sorted by their start offset.

    Used by checks 10 + 11 to enforce the cherry-picked-label and
    qualitative-data-link discipline regardless of which medium the
    body uses for its raw-data exposition (fenced code, `<details>`
    table, or `<details>` long-text block).
    """
    out = _iter_sample_fences(details) + _iter_sample_details(details)
    out.sort(key=lambda t: t[0])
    return out


def _prelude_window(details: str, fence_start: int, max_chars: int = 1500) -> str:
    """Return the prose immediately preceding a fenced block.

    Walks back at most ``max_chars`` from ``fence_start``. Stops at the
    previous fenced block's closing ``` (so two consecutive sample
    blocks don't share each other's prelude), then trims any leading
    partial line.

    Stops at the LATER of two boundaries: a previous fenced block's
    closing ``` line, OR a previous `<details>` block's `</details>`
    close. Two adjacent sample blocks therefore do NOT share each other's
    disclosure prelude (Phase A review MINOR-4, 2026-06-13: v3's `## Data`
    packs the Evaluated-with + Generated example blocks into one section,
    widening the neighbour-bleed blast radius — an undisclosed block must
    not borrow a sibling's disclosure). The v2 form that puts the
    cherry-pick disclosure inside the `<summary>` still passes because the
    caller shifts the scan past `</summary>`, so this block's own
    `<summary>` stays inside the window (only the PREVIOUS block's
    `</details>` is a cut point).
    """
    lo = max(0, fence_start - max_chars)
    window = details[lo:fence_start]
    # Don't cross a previous sample block's boundary.
    cut = 0
    prev_fence = window.rfind("\n```")
    if prev_fence != -1:
        nl = window.find("\n", prev_fence + 1)
        if nl != -1:
            cut = max(cut, nl + 1)
    prev_details = window.rfind("</details>")
    if prev_details != -1:
        nl = window.find("\n", prev_details)
        cut = max(cut, nl + 1 if nl != -1 else prev_details + len("</details>"))
    if cut:
        window = window[cut:]
    return window


_AGGREGATE_PATH_RE = re.compile(
    # Filenames whose stem advertises aggregation, OR the .npz extension.
    r"\b\S*(?:regression|summary|aggregat\w*|per[-_]?cell|cell[-_]?level)\S*\.(?:csv|json|jsonl|tsv|parquet|npz)\b"
    r"|\b\S+\.npz\b",
    re.IGNORECASE,
)


_NOT_UPLOADED_RE = re.compile(
    r"(?:not\s+uploaded|not\s+available|did\s+not\s+upload"
    r"|raw\s+completions?\s+(?:were\s+)?(?:not|never)"
    r"|raw[-_\s]?completions?\s+(?:were\s+)?n/a)",
    re.IGNORECASE,
)


_CHERRY_DISCLOSURE_RE = re.compile(
    r"\b(?:cherry[-\s]?picked|random[-\s]?sample|drawn at random|"
    r"random draw|first \d+ of \d+|first \d+ completions?|"
    # bare `N of M rows` form — PARITY with `_SUBSET_DISCLOSURE_RE` (the
    # two are documented as must-stay-in-sync). A v4 `## Methodology`
    # Sample block disclosed solely as "5 of 2,000 rows" passes check 19
    # (subset disclosure); without this it would fail check 10 (Phase A
    # review Minor, 2026-06-24).
    r"\d+ of \d+ rows|\d+ of [\d,]+ rows|"
    r"\d+ random completions?|\d+ randomly[-\s]?sampled|"
    # `<N> example training rows`, `<N> example eval probes`,
    # `<N> examples of …`, `<N> sample completions`, `<N> sample rows`
    # — the disclosure form used inside `<details>` block summaries
    # (e.g. task #432's "5 example training rows" /
    # "5 example eval probes"). The "example" / "sample" qualifier
    # tells the reader the rows are illustrative, not exhaustive.
    r"\d+\s+(?:examples?|sample[s]?)\b|"
    # harmful-content carve-out — the sanitized excerpt form, kept in
    # PARITY with check 19's `_SUBSET_DISCLOSURE_RE` so a v3
    # `## Data → ### Generated` block built from EM / bad-medical-advice
    # corpora (this project's dominant data type) is NOT FAILed by
    # check 10 for lacking cherry-pick wording (Phase A review MAJOR-1,
    # 2026-06-13). Must stay in sync with `_SUBSET_DISCLOSURE_RE`.
    r"sanitized for context hygiene|harmful-content row|truncated — harmful)",
    re.IGNORECASE,
)


# ─── Checks ────────────────────────────────────────────────────────────────


# Minimum body length (chars). Bodies smaller than this are stubs / placeholders.
# Defense-in-depth against the cache → body.md silent-handoff failure
# (incident: task #385, 2026-05-25 — body.md read literally "placeholder" for
# ~26h while `has_clean_result=true`). Real clean-result bodies are >5,000
# chars; 500 is a conservative floor.
MIN_BODY_CHARS = 500

# Stub-content sentinels we positively recognize (case-insensitive).
STUB_TOKENS = {"placeholder", "tbd", "todo", "stub"}


def check_body_nonstub(body: str) -> CheckResult:
    """Check 0: body is not a stub / placeholder.

    Runs FIRST and (in `verify_text`) short-circuits the rest of the
    check chain when it FAILs, so the operator gets one clear fail-fast
    signal rather than a dozen cascading "<section> missing" errors from
    a body that's just the word `placeholder`. Triggers FAIL when ANY
    of:
      - body's non-frontmatter content is empty,
      - body's non-frontmatter content collapses to a single stub token
        (`placeholder`, `tbd`, `todo`, `stub`) after whitespace strip,
      - body is < MIN_BODY_CHARS (500) characters,
      - body has no `# <title>` H1 line (clean-result bodies always carry
        one; non-clean-result bodies do not run through this verifier).

    The H1 sub-check here is appropriate because `verify_task_body.py`
    is only ever invoked against clean-result bodies (analyzer Step 5,
    clean-result-critic Step 1 pre-pass). Non-clean-result bodies
    (proposed-task idea captures, clarifier output) take different
    shapes and are not gated by this verifier; the CLI-level
    `_assert_body_nontrivial` in `scripts/task.py` does NOT impose the
    H1 requirement so those bodies can be `set-body`-written normally.
    """
    stripped = body.strip()
    n_chars = len(stripped)
    if n_chars == 0:
        return CheckResult(
            "body is not a stub",
            False,
            "body is empty — cache → body.md handoff likely failed; see analyzer.md Step 6",
        )
    if stripped.casefold() in STUB_TOKENS:
        return CheckResult(
            "body is not a stub",
            False,
            f"body is literally the stub token {stripped!r} — "
            "cache → body.md handoff likely failed; see analyzer.md Step 6",
        )
    if n_chars < MIN_BODY_CHARS:
        return CheckResult(
            "body is not a stub",
            False,
            f"body is only {n_chars} chars (floor {MIN_BODY_CHARS}) — "
            "real clean-result bodies are >5 KB. If this is intentional, "
            "check that the analyzer's cache → body.md handoff did not silently "
            "drop the clean-result content.",
        )
    if find_h1_title(body) is None:
        return CheckResult(
            "body is not a stub",
            False,
            "body has no `# <title>` H1 line — real clean-result bodies always "
            "start with an H1; this looks like a stub or a truncated handoff.",
        )
    return CheckResult(
        "body is not a stub",
        True,
        f"{n_chars} chars + H1 present",
    )


def _count_leading_frontmatter_blocks(text: str) -> int:
    """Count consecutive leading ``---\\n...\\n---\\n`` blocks in `text`.

    Mirrors the strip logic in `task_workflow._strip_leading_frontmatter_blocks`
    so both call-sites agree on what counts as a frontmatter block.
    """
    count = 0
    rest = text
    while rest.startswith("---\n"):
        end = rest.find("\n---\n", 4)
        if end == -1:
            break
        count += 1
        rest = rest[end + len("\n---\n") :]
    return count


def check_no_duplicate_frontmatter(raw: str) -> CheckResult:
    """Check: the raw body.md must contain exactly ONE leading YAML
    frontmatter block (``---\\n...\\n---\\n``), never two or more.

    Duplicate frontmatter ships when a caller passes a complete markdown
    document (frontmatter + body) to `task.py set-body` (or directly to
    `task_workflow.set_body`) and the prepended canonical frontmatter
    stacks on top of the caller-supplied one. The dashboard parses the
    FIRST block as the header card and renders the SECOND block as
    literal YAML at the top of the visible body — a visible-corruption
    bug that bit task #389 twice (analyzer v5 and v7) in one /issue
    session on 2026-05-26.

    The library now strips leading frontmatter inside `set_body()`, but
    this verifier check is the belt-and-suspenders gate: any future
    regression (manual editing, alternative write path, third-party
    tool) that lets a duplicate block land on disk will FAIL the
    analyzer's pre-flight and the clean-result-critic's gate.

    Operates on the RAW body.md text (not the post-split body) so the
    count is unambiguous regardless of what `split_frontmatter` would
    parse — a single missing-closing-delimiter case is benign (zero
    valid blocks, the body just happens to start with `---`), but
    stacked blocks always FAIL.
    """
    n = _count_leading_frontmatter_blocks(raw)
    if n >= 2:
        return CheckResult(
            "no duplicate frontmatter",
            False,
            f"body.md has {n} stacked YAML frontmatter blocks at the top — "
            "set-body should strip caller-supplied frontmatter, but this body "
            "has duplicated frontmatter (the dashboard will render the second "
            "block as literal YAML at the top of the visible body). "
            "Re-run `task.py set-body` to fix; see task #389 (2026-05-26).",
        )
    return CheckResult(
        "no duplicate frontmatter",
        True,
        f"{n} leading frontmatter block{'s' if n != 1 else ''}",
    )


def check_title_confidence(body: str) -> CheckResult:
    title = find_h1_title(body)
    if not title:
        return CheckResult("title confidence tag", False, "no H1 found")
    m = re.search(r"\((LOW|MODERATE|HIGH) confidence\)\s*$", title)
    if not m:
        return CheckResult(
            "title confidence tag",
            False,
            f"title must end with '(LOW|MODERATE|HIGH confidence)' — got: {title[-60:]!r}",
        )
    return CheckResult("title confidence tag", True, f"level={m.group(1)}")


def check_h1_matches_frontmatter_title(body: str, fm: dict) -> CheckResult:
    """The H1 headline and the frontmatter `title` must agree
    (whitespace-normalized) on sentinelled clean-result bodies.

    The title lives in two places — frontmatter via `task.py set-title`
    (feeds the dashboard LIST view + REGISTRY) and the body H1 via
    `set-body` (feeds the rendered DETAIL view). #825: a retitle without
    an H1 edit shipped a stale headline through a full critic round.
    Severity: FAIL on v4 bodies; WARN on grandfathered v3/v2 (forward-only
    rule — 4 real v3/v2 bodies diverge today, incl. completed #432/#458);
    PASS-skip on non-sentinelled bodies (pre-promotion bodies legitimately
    have no synced H1) and on frontmatter-less input (draft / --body-stdin
    dry runs carry no fm to compare). Normalization is whitespace collapse
    ONLY — no case/Unicode/punctuation folding (#763's Unicode-rho vs
    ASCII-"rho" transliteration is a REAL sync failure folding would hide).
    """
    name = "H1 matches frontmatter title"
    v4 = is_v4(body)
    sentinelled = v4 or is_v3(body) or is_v2_nested_design(body)
    if not sentinelled:
        return CheckResult(name, True, "not a sentinelled clean-result body — skipped")
    if not fm:
        return CheckResult(
            name,
            True,
            "no frontmatter in input (draft / --body-stdin invocation) — "
            "skipped; the gate-time --issue run compares against the real body.md",
        )
    warn_only = not v4  # forward-only: v3/v2 grandfathered to WARN

    def _flag(detail: str) -> CheckResult:
        if warn_only:
            return CheckResult(name, True, detail + " [grandfathered v3/v2 — WARN]", is_warn=True)
        return CheckResult(name, False, detail)

    title = fm.get("title")
    if title is None:
        return _flag(
            "sentinelled clean-result body has no frontmatter `title` — "
            'run `task.py set-title <N> "<H1 text>"`'
        )
    h1 = find_h1_title(body)
    if h1 is None:
        return _flag("no H1 found — a sentinelled clean-result must open with `# <title>`")

    def _norm(s: object) -> str:
        return " ".join(str(s).split())

    if _norm(h1) != _norm(title):
        return _flag(
            f"H1 differs from frontmatter title (dashboard list shows the frontmatter "
            f"title; the body renders the H1) — H1: {h1[:90]!r} vs title: "
            f'{str(title)[:90]!r}. Sync via `task.py set-title <N> "<H1 text>"` '
            f"(the promote-skill convention, H1 canonical) or re-`set-body` with the "
            f"H1 edited if the frontmatter title is the fresher one."
        )
    return CheckResult(name, True, "H1 == frontmatter title (whitespace-normalized)")


def check_required_sections(body: str) -> CheckResult:
    """Check 2: the required H2 sections appear in order, and no retired
    H2 is present.

    v4 bodies (sentinel `<!-- clean-result-v4 -->`) require four flat
    H2s in order — `## Takeaways` / `## Goal` / `## Methodology` /
    `## Results` — plus a `**Repro:**`/`**Context:**` footer (not an H2),
    and reject the v3 content H2s (`## What I ran` / `## Findings` /
    `## Data` / `## Reproducibility`) AND the earlier retired pairs
    (`## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure`).

    v3 bodies (sentinel `<!-- clean-result-v3 -->`) require five flat
    H2s in order — `## Takeaways` / `## What I ran` / `## Findings` /
    `## Data` / `## Reproducibility` — and reject `## Human TL;DR` /
    `## TL;DR` / `## Details` / `## Figure` (the v3 redesign retired
    the model-written casual summary and the `## TL;DR` umbrella).

    v2 / legacy bodies keep the 2-content-section spec (2026-W22 task
    #454): three required H2s (`## Human TL;DR` / `## TL;DR` /
    `## Reproducibility`), rejecting `## Details` / `## Figure`. The
    fold + the stray/ordering rules are identical between generations —
    only the required + retired sets differ. Legacy bodies pre-2026-W22
    are forward-grandfathered (the verifier never re-runs over them).
    """
    v4 = is_v4(body)
    v3 = is_v3(body)
    if v4:
        required = V4_REQUIRED_H2_SECTIONS
        retired = V4_RETIRED_H2_SECTIONS
        label = "four required H2 sections in order"
    elif v3:
        required = V3_REQUIRED_H2_SECTIONS
        retired = V3_RETIRED_H2_SECTIONS
        label = "five required H2 sections in order"
    else:
        required = REQUIRED_H2_SECTIONS
        retired = RETIRED_H2_SECTIONS
        label = "three required H2 sections in order"
    found = [name for name, _, _ in find_h2_sections(body)]
    # Hard FAIL on retired H2s (force clean migration).
    retired_present = [s for s in retired if s in found]
    if retired_present:
        if v4:
            detail = (
                f"retired H2(s) present: {', '.join('## ' + s for s in retired_present)}. "
                "The v4 spec uses four flat H2s — `## Takeaways` / `## Goal` / "
                "`## Methodology` / `## Results` — plus a `**Repro:**`/`**Context:**` "
                "footer. The v3 content H2s (`## What I ran` / `## Findings` / "
                "`## Data` / `## Reproducibility`) fold into `## Goal` / "
                "`## Methodology` / `## Results` + the footer; the retired "
                "`## Human TL;DR` / `## TL;DR` / `## Details` / `## Figure` stay "
                "rejected. Remove the retired H2 and migrate its content. "
                "See .claude/skills/clean-results/SPEC.md."
            )
        elif v3:
            detail = (
                f"retired H2(s) present: {', '.join('## ' + s for s in retired_present)}. "
                "The v3 spec drops `## Human TL;DR` (the model-written casual "
                "summary is retired) and the `## TL;DR` umbrella (Takeaways "
                "replaces the skim function; per-result narrative lives in flat "
                "`## Findings`) — remove the retired H2 and migrate its content. "
                "See .claude/skills/clean-results/SPEC.md."
            )
        else:
            detail = (
                f"retired H2(s) present: {', '.join('## ' + s for s in retired_present)}. "
                "The 2-content-section spec (2026-W22) folds Details into per-result "
                "H3s under `## TL;DR` and inlines figures inside each result H3 — "
                "remove the retired H2 and migrate its content. See "
                ".claude/skills/clean-results/SPEC.md."
            )
        return CheckResult(label, False, detail)
    missing = [s for s in required if s not in found]
    if missing:
        return CheckResult(
            label,
            False,
            f"missing: {', '.join('## ' + s for s in missing)} (found: {found})",
        )
    # Order check: `required` must appear in this exact order within the
    # body's H2 sequence (extra non-retired H2s after the LAST required
    # section are tolerated, but NOT before).
    seq = [s for s in found if s in required]
    if seq != required:
        return CheckResult(
            label,
            False,
            f"wrong order — got {seq}, expected {required}",
        )
    # Stray H2 check: any non-required, non-retired H2 (e.g., a leftover
    # `## Goal`, `## Background`, `## Methods`) that appears BEFORE the
    # required sequence completes is rejected. The spec tolerates extra
    # H2s ONLY after the LAST required section. Retired H2s already
    # produced a hard FAIL above, so don't double-report them.
    last_required_idx = -1
    stray_before: list[str] = []
    for name, _, _ in find_h2_sections(body):
        if name in required:
            if name == required[-1]:
                last_required_idx = 1
        elif name in retired:
            continue
        elif last_required_idx == -1:
            stray_before.append(name)
    if stray_before:
        return CheckResult(
            label,
            False,
            f"stray H2(s) before `## {required[-1]}`: "
            f"{', '.join('## ' + s for s in stray_before)}. The spec permits "
            f"extra H2s ONLY after `## {required[-1]}` — required sequence is "
            f"{required} and nothing else may appear in between. Remove the "
            f"stray H2 or move it after `## {required[-1]}`.",
        )
    return CheckResult(label, True)


def _h3_names_under_h2(body: str, h2_name: str) -> list[str]:
    """Return the `### ` H3 heading names (fence-aware) directly inside
    the `## <h2_name>` section, in order. Empty list when the section is
    absent."""
    text = section_text(body, h2_name)
    if text is None:
        return []
    return [name for name, _line in _collect_tldr_h3_names(text)]


def _h3_subsection_text(section_body: str, h3_name: str) -> str | None:
    """Return the text of the `### <h3_name>` subsection inside an H2
    section body (the leading-word of the heading, post-`— hook` strip,
    must equal `h3_name`, case-insensitive), spanning to the next `### `
    H3 or end. Fence-aware. None when the H3 is absent.

    Used to scope check 11's raw-completions-link scan to
    `## Data → ### Generated` only (the Trained-on / Evaluated-with
    blocks link JSONLs / probe banks, not raw_completions)."""
    h3s = _collect_tldr_h3_names(section_body)
    idx = _find_named_h3(h3s, h3_name)
    if idx is None:
        return None
    lines = section_body.splitlines()
    start_line = h3s[idx][1]
    end_line = h3s[idx + 1][1] if idx + 1 < len(h3s) else len(lines)
    return "\n".join(lines[start_line:end_line]).strip()


def _count_bullets(text: str) -> int:
    """Count top-level markdown list items (`- ` / `* ` at line start,
    fence-aware) in `text`. Sub-bullets (indented) are NOT counted —
    the Takeaways shape is a flat bullet list."""
    n = 0
    in_fence = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Top-level only: the raw (un-stripped) line begins with `- ` or
        # `* ` with no leading indent.
        if re.match(r"^[-*]\s+\S", line):
            n += 1
    return n


def check_v3_structure(body: str) -> CheckResult:
    """v3 structure check (replaces checks 3 + 3b for v3 bodies).

    For a `<!-- clean-result-v3 -->` body, verify:
      - `## Takeaways` carries 3-6 top-level bullets (the authoritative
        count gate — check 20's word-cap check defers to this so there
        is exactly ONE bullet-count rule).
      - `## What I ran` carries the `**Why:**` slot bullet.
      - `## Findings` carries ≥1 `### ` finding heading.

    The 3-6 range is the same FAIL band the §4 caps table records; it
    lives HERE (not in check 20) so the count is enforced once, on the
    structural check, regardless of whether the word-cap check runs.
    """
    label_name = "v3 structure (Takeaways / What I ran / Findings)"
    takeaways = section_text(body, "Takeaways")
    if takeaways is None:
        return CheckResult(label_name, False, "## Takeaways section missing")
    n_bullets = _count_bullets(takeaways)
    if not (V3_TAKEAWAYS_MIN_BULLETS <= n_bullets <= V3_TAKEAWAYS_MAX_BULLETS):
        return CheckResult(
            label_name,
            False,
            f"## Takeaways has {n_bullets} top-level bullet(s) — the v3 shape "
            f"requires {V3_TAKEAWAYS_MIN_BULLETS}-{V3_TAKEAWAYS_MAX_BULLETS} "
            "numbers-first bullets (no paragraphs).",
        )
    what_i_ran = section_text(body, "What I ran")
    if what_i_ran is None:
        return CheckResult(label_name, False, "## What I ran section missing")
    # The `**Why:**` slot bullet — accept `- **Why:**` / `**Why:**` /
    # `Why:` at the start of a bullet (mirror the Motivation bullet
    # acceptance in the v2 check).
    if not re.search(r"(?im)^\s*[-*]?\s*(\*\*)?Why(\*\*)?\s*:", what_i_ran):
        return CheckResult(
            label_name,
            False,
            "## What I ran is missing the `**Why:**` slot bullet — the v3 shape "
            "requires a `**Why:**` line (the only place for prior-issue links).",
        )
    findings = section_text(body, "Findings")
    if findings is None:
        return CheckResult(label_name, False, "## Findings section missing")
    finding_h3s = [name for name, _line in _collect_tldr_h3_names(findings)]
    if not finding_h3s:
        return CheckResult(
            label_name,
            False,
            "## Findings has no `### <finding>` heading — the v3 shape requires "
            "≥1 `### ` finding under `## Findings`.",
        )
    return CheckResult(
        label_name,
        True,
        f"v3 structure clean — Takeaways ({n_bullets} bullets), What I ran "
        f"(Why: present), Findings ({len(finding_h3s)} `### ` finding(s))",
    )


def check_v4_structure(body: str) -> CheckResult:
    """v4 structure check (dispatched from check 3 for v4 bodies).

    For a `<!-- clean-result-v4 -->` body, verify:
      - `## Takeaways` carries 3-6 top-level bullets (the authoritative
        count gate — the v4 word-cap check defers to this).
      - `## Goal` carries BOTH the `**This experiment in context:**` slot
        AND the `**Broader narrative:**` slot.
      - `## Methodology` carries the `**Training:**` slot (or the explicit
        `**N/A — no model training**` marker) AND the `**Evaluation:**`
        slot.
      - `## Results` carries ≥1 `### ` result heading.

    The 3-6 range is the same FAIL band the v4 caps table records; it
    lives HERE so the count is enforced once.
    """
    label_name = "v4 structure (Takeaways / Goal / Methodology / Results)"
    takeaways = section_text(body, "Takeaways")
    if takeaways is None:
        return CheckResult(label_name, False, "## Takeaways section missing")
    n_bullets = _count_bullets(takeaways)
    if not (V3_TAKEAWAYS_MIN_BULLETS <= n_bullets <= V3_TAKEAWAYS_MAX_BULLETS):
        return CheckResult(
            label_name,
            False,
            f"## Takeaways has {n_bullets} top-level bullet(s) — the v4 shape "
            f"requires {V3_TAKEAWAYS_MIN_BULLETS}-{V3_TAKEAWAYS_MAX_BULLETS} "
            "numbers-first bullets (no paragraphs).",
        )
    goal = section_text(body, "Goal")
    if goal is None:
        return CheckResult(label_name, False, "## Goal section missing")
    # `**This experiment in context:**` slot — accept the bold-label form
    # with optional bullet marker; match on the leading phrase.
    if not re.search(
        r"(?im)^\s*[-*]?\s*(\*\*)?\s*This experiment in context\s*(\*\*)?\s*:",
        goal,
    ):
        return CheckResult(
            label_name,
            False,
            "## Goal is missing the `**This experiment in context:**` slot — the v4 "
            "shape requires it (what THIS experiment tests + its line; the only "
            "place for prior-issue links).",
        )
    if not re.search(r"(?im)^\s*[-*]?\s*(\*\*)?\s*Broader narrative\s*(\*\*)?\s*:", goal):
        return CheckResult(
            label_name,
            False,
            "## Goal is missing the `**Broader narrative:**` slot — the v4 shape "
            "requires it (the project-level question / open-questions anchor this "
            "experiment serves).",
        )
    methodology = section_text(body, "Methodology")
    if methodology is None:
        return CheckResult(label_name, False, "## Methodology section missing")
    # `**Training:**` slot OR the explicit no-training marker.
    has_training = re.search(r"(?im)^\s*[-*]?\s*(\*\*)?\s*Training\s*(\*\*)?\s*:", methodology)
    has_no_training = re.search(
        r"(?im)\*\*\s*N/?A\s*[—–-]\s*no model training",  # noqa: RUF001
        methodology,
    )
    if not (has_training or has_no_training):
        return CheckResult(
            label_name,
            False,
            "## Methodology is missing the `**Training:**` slot (or the explicit "
            "`**N/A — no model training**` marker for analysis-only tasks).",
        )
    if not re.search(r"(?im)^\s*[-*]?\s*(\*\*)?\s*Evaluation\s*(\*\*)?\s*:", methodology):
        return CheckResult(
            label_name,
            False,
            "## Methodology is missing the `**Evaluation:**` slot — the v4 shape "
            "requires it (DV + metric + judge + probe set).",
        )
    results = section_text(body, "Results")
    if results is None:
        return CheckResult(label_name, False, "## Results section missing")
    result_h3s = [name for name, _line in _collect_tldr_h3_names(results)]
    if not result_h3s:
        return CheckResult(
            label_name,
            False,
            "## Results has no `### <result>` heading — the v4 shape requires "
            "≥1 `### ` result under `## Results`.",
        )
    return CheckResult(
        label_name,
        True,
        f"v4 structure clean — Takeaways ({n_bullets} bullets), Goal (both slots), "
        f"Methodology (Training + Evaluation), Results ({len(result_h3s)} `### ` result(s))",
    )


def check_tldr_labels(body: str) -> CheckResult:
    """Check 3: `## TL;DR` opens with the Motivation block (v2/legacy);
    for v3 bodies, dispatches to `check_v3_structure`.

    2-content-section spec (2026-W22, task #454). The TL;DR is the
    LessWrong-style narrative; it opens with either:

    - an `### Motivation` H3 (preferred new shape) — typically the
      first H3 inside `## TL;DR`, followed by one `### <finding>` H3
      per result; OR
    - a `**Motivation:**` boldface bullet (legacy form, still
      accepted) at the start of a list under `## TL;DR`.

    The retired `What I ran` / `Results` required bullets are no
    longer enforced — the new shape distributes that content across
    the per-result H3s (each result H3 carries its own setup, figure,
    read, and cherry-picked example). Checks 4, 10, 11 verify the
    per-result structure (figure present, cherry-picked label,
    qualitative-data link).

    The Motivation block must ALSO be the FIRST content under `## TL;DR`
    — a stray intro paragraph between the heading and `### Motivation`
    (or `**Motivation:**`) is rejected, matching SPEC.md "Opens with
    `### Motivation`".

    v3 bodies have no `## TL;DR` umbrella — the per-result structure is
    flat under `## Findings`. For v3 this check delegates to
    `check_v3_structure` (Takeaways 3-6 bullets, `**Why:**` slot, ≥1
    `### ` finding), so the CHECKS slot stays one function but enforces
    the right shape per generation.
    """
    if is_v4(body):
        return check_v4_structure(body)
    if is_v3(body):
        return check_v3_structure(body)
    tldr = section_text(body, "TL;DR")
    label_name = "TL;DR opens with Motivation"
    if tldr is None:
        return CheckResult(label_name, False, "## TL;DR section missing")
    missing: list[str] = []
    for label in TLDR_BULLETS_REQUIRED:
        # Accept either form:
        #  - `### Motivation` H3 heading (with optional trailing text), OR
        #  - `**Motivation:**` / `Motivation:` at start of list item.
        h3_re = rf"(?im)^\s*###\s+{re.escape(label)}\b"
        bullet_re = rf"(?im)^\s*[-*]\s*(\*\*)?{re.escape(label)}(\*\*)?\s*:"
        if not (re.search(h3_re, tldr) or re.search(bullet_re, tldr)):
            missing.append(label)
    if missing:
        return CheckResult(
            label_name,
            False,
            f"missing: {', '.join(missing)} — TL;DR must open with an "
            "`### Motivation` H3 (preferred) or a `**Motivation:**` bullet",
        )
    # Order check (FIRST, not just present): the Motivation block must be
    # the FIRST content block inside `## TL;DR`. Find the first non-blank
    # H3 OR the first non-blank bullet-list item, whichever comes first;
    # require it to be the Motivation label. A stray `### First result`
    # before `### Motivation` is rejected.
    first_h3_match = re.search(r"(?m)^\s*###\s+([^\n]+)$", tldr)
    first_bullet_match = re.search(
        r"(?im)^\s*[-*]\s*(?:\*\*)?([A-Za-z][A-Za-z0-9 _/-]*?)(?:\*\*)?\s*:", tldr
    )
    # Pick whichever appears earliest in the TL;DR text.
    candidates: list[tuple[int, str, str]] = []
    if first_h3_match is not None:
        candidates.append((first_h3_match.start(), "H3", first_h3_match.group(1).strip()))
    if first_bullet_match is not None:
        candidates.append(
            (first_bullet_match.start(), "bullet", first_bullet_match.group(1).strip())
        )
    if candidates:
        candidates.sort(key=lambda t: t[0])
        first_offset, first_kind, first_label = candidates[0]
        # Strip any trailing inline annotation from the H3 heading (e.g.,
        # `### Motivation — short hook`) so we compare on the first word.
        # The en/em dash characters are intentional — clean-result H3s
        # routinely use them as the hook separator.
        first_label_head = re.split(r"[\s–—:.,]", first_label, maxsplit=1)[0]  # noqa: RUF001
        accepted = {label.casefold() for label in TLDR_BULLETS_REQUIRED}
        if first_label_head.casefold() not in accepted:
            return CheckResult(
                label_name,
                False,
                f"Motivation must be the FIRST {first_kind} block inside `## TL;DR` "
                f"— found `{first_label}` first. Reorder so `### Motivation` "
                "(or `**Motivation:**` bullet) opens the section.",
            )
        # Spec rule: no stray prose may sit between the `## TL;DR` heading
        # and the Motivation block. `tldr` is already stripped of leading
        # whitespace by section_text(), so any non-blank line appearing
        # before the first structural element (H3 / labelled bullet) is
        # intro prose that breaks the "TL;DR opens with Motivation" shape.
        prelude = tldr[:first_offset]
        for line in prelude.splitlines():
            if line.strip():
                return CheckResult(
                    label_name,
                    False,
                    "stray prose before Motivation — `## TL;DR` must open "
                    f"directly with `### Motivation` (or `**Motivation:**` "
                    f"bullet); found prose line `{line.strip()[:80]}` first. "
                    "Move the intro paragraph into the Motivation block.",
                )
    return CheckResult(label_name, True)


def _collect_tldr_h3_names(tldr: str) -> list[tuple[str, int]]:
    """Return [(heading_name, line_index)] for each `### ` H3 inside
    `tldr`, in order, honoring fenced-code-block state. Used by
    `check_tldr_nested_structure`.
    """
    h3_re = re.compile(r"^\s*###\s+(?P<name>[^\n]+?)\s*$")
    out: list[tuple[str, int]] = []
    in_fence = False
    for i, line in enumerate(tldr.splitlines()):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = h3_re.match(line)
        if m:
            out.append((m.group("name").strip(), i))
    return out


def _find_named_h3(h3_names: list[tuple[str, int]], target: str) -> int | None:
    """Return the index (into `h3_names`) of the first heading whose
    leading-word (post-strip-of-`— ...` inline hook) equals `target`
    (case-insensitive). None when no heading matches.
    """
    target_norm = target.casefold().strip()
    for idx, (name, _line_no) in enumerate(h3_names):
        name_norm = re.sub(r"\s+", " ", name).casefold().strip()
        head = re.split(r"\s+[–—:]\s+", name_norm, maxsplit=1)[0]  # noqa: RUF001
        if head == target_norm:
            return idx
    return None


def _count_h4_after(tldr: str, line_after: int) -> int:
    """Count `#### ` H4 headings in `tldr` after `line_after`,
    honoring fenced-code-block state. Used to count `#### <finding>`
    H4 children under `### Findings`.
    """
    in_fence = False
    h4_count = 0
    for line in tldr.splitlines()[line_after + 1 :]:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("#### ") and not stripped.startswith("##### "):
            h4_count += 1
    return h4_count


def check_tldr_nested_structure(body: str) -> CheckResult:
    """Nested-design (v2) `## TL;DR` shape check (sentinel-gated).

    Bodies bearing the `<!-- clean-result-v2 -->` sentinel MUST shape
    `## TL;DR` as three ordered H3s — `### Motivation` /
    `### What I ran` / `### Findings` — with at least one `#### `
    H4 child under `### Findings` (the per-result `#### <finding>`
    blocks). FAIL when the sentinel is present and any required H3
    is missing OR the order is wrong OR `### Findings` has no `#### `
    children.

    Bodies WITHOUT the sentinel PASS vacuously — forward-only
    migration. The post-#454 flat shape (no `### What I ran`, no
    `### Findings` parent, flat per-result `### <finding>` H3s) is
    still tolerated for bodies that predate the nested-design
    adoption.
    """
    label_name = "TL;DR nested-design structure (v2)"
    if not is_v2_nested_design(body):
        return CheckResult(
            label_name,
            True,
            "v2 sentinel absent — pre-nested-design body, nested-shape rule skipped",
        )
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        # check_required_sections already FAILs on a missing TL;DR;
        # don't double-report.
        return CheckResult(label_name, True, "## TL;DR missing — check 2 will report")

    h3_names_in_order = _collect_tldr_h3_names(tldr)
    idx_motivation = _find_named_h3(h3_names_in_order, "Motivation")
    idx_what_i_ran = _find_named_h3(h3_names_in_order, "What I ran")
    idx_findings = _find_named_h3(h3_names_in_order, "Findings")
    missing: list[str] = []
    if idx_motivation is None:
        missing.append("### Motivation")
    if idx_what_i_ran is None:
        missing.append("### What I ran")
    if idx_findings is None:
        missing.append("### Findings")
    if missing:
        return CheckResult(
            label_name,
            False,
            f"v2 sentinel present but TL;DR is missing required H3(s): "
            f"{', '.join(missing)}. The nested-design shape requires "
            "`### Motivation` → `### What I ran` → `### Findings` in that "
            "order, with one `#### <finding>` per result under "
            "`### Findings`. See SPEC.md § Required body shape.",
        )
    # Order check.
    if not (idx_motivation < idx_what_i_ran < idx_findings):
        return CheckResult(
            label_name,
            False,
            f"v2 sentinel present but TL;DR H3 order is wrong — got "
            f"Motivation@{idx_motivation}, What I ran@{idx_what_i_ran}, "
            f"Findings@{idx_findings}; required order is Motivation → "
            "What I ran → Findings.",
        )
    # Findings child check: ≥1 `#### ` H4 must exist AFTER `### Findings`.
    findings_line_no = h3_names_in_order[idx_findings][1]
    h4_count = _count_h4_after(tldr, findings_line_no)
    if h4_count == 0:
        return CheckResult(
            label_name,
            False,
            "v2 sentinel present and `### Findings` H3 found, but no "
            "`#### <finding>` H4 children under it. The nested-design "
            "shape requires one `#### <finding>` per result inside "
            "`### Findings`.",
        )
    return CheckResult(
        label_name,
        True,
        f"v2 nested-design structure clean — Motivation → What I ran → "
        f"Findings (with {h4_count} `#### <finding>` H4 children)",
    )


def _figure_scan_section(body: str) -> str:
    """Return the H2 section name where inline figures live for the body's
    generation: `## Results` (v4), `## Findings` (v3), `## TL;DR`
    (v2 / legacy)."""
    if is_v4(body):
        return "Results"
    if is_v3(body):
        return "Findings"
    return "TL;DR"


def _gather_figure_image_urls(body: str) -> list[str]:
    """Collect figure image URLs inline under the result-narrative
    section. Powers checks 4 / 4b.

    v4 bodies: figures live under `## Results` (one per `### <result>`).
    v3 bodies: figures live under `## Findings` (one per `### <finding>`).
    v2 / legacy bodies: figures live inside per-result H3s under
    `## TL;DR` (2026-W22 spec, task #454)."""
    urls: list[str] = []
    section = _figure_scan_section(body)
    text = section_text(body, section)
    if text is not None:
        urls.extend(_IMAGE_RE.findall(text))
    return urls


def check_figure_image(body: str) -> CheckResult:
    """Check 4: at least one `![alt](url)` image exists inline under the
    result-narrative section — `## Results` for v4, `## Findings` for v3,
    `## TL;DR` for v2 / legacy (every result block carries its own figure)."""
    section = _figure_scan_section(body)
    urls = _gather_figure_image_urls(body)
    if not urls:
        return CheckResult(
            "hero image present",
            False,
            f"no `![alt](path)` image found inline under `## {section}` — every "
            "result block carries its own figure",
        )
    return CheckResult("hero image present", True, f"{len(urls)} image(s)")


# Same-repo SHA-pinned raw-GitHub figure URLs — the canonical figure-hosting
# pattern. Captured so check 4b can verify blob EXISTENCE offline via
# `git cat-file` (worktrees share the object database with the main
# checkout, so a commit made on `main` resolves from any checkout).
_RAW_GITHUB_FIGURE_RE = re.compile(
    r"^https?://raw\.githubusercontent\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/(?P<sha>[0-9a-fA-F]{7,40})/(?P<path>[^?#]+)"
)
_THIS_REPO_SLUG = ("superkaiba", "explore-persona-space")

# Repo-relative figure path carrying its own issue number — scope of check 29.
_ISSUE_FIGURE_PATH_RE = re.compile(r"^figures/issue_(?P<issue>\d+)/\S")

# Check 31: per-unit companion-figure basename patterns. The lookbehind
# stops mid-word matches ("supercontext"); `indiv` is deliberately
# EXCLUDED — it names the per-question REGIME in this project (#928's
# pooled hero `mlp_indiv_hero_4arm.png`), not a per-unit view.
_PER_UNIT_FIG_RE = re.compile(r"(?<![a-z0-9])per[-_]?(context|unit|cell)", re.IGNORECASE)

# Check 38: any markdown link (image embeds are masked out before this
# scans, so no `!`-lookbehind is needed); link text tolerates `]` not
# followed by `(` — the same tolerance `_IMAGE_RE` uses — and may be
# EMPTY (the `[](q)` residue a clickable-image wrapper leaves after the
# inner image is masked). Distinct from `_LINK_RE`, which requires
# non-empty text and has no `]` tolerance.
_MD_LINK_RE = re.compile(r"\[(?:[^\]]|\](?!\())*\]\(([^)]+)\)")
# Check 38: a `figures/issue_<K>/…png` path inside a link/image URL, ANY
# host form (raw-GitHub, github blob, relative). Captures the
# repo-relative path; `[^)\s?#]+` stops the capture before a
# query-string / fragment (`…png?raw=1`).
_LINKED_ISSUE_PNG_RE = re.compile(
    r"(?P<path>figures/issue_(?P<issue>\d+)/[^)\s?#]+\.png)", re.IGNORECASE
)
# Check 38: HTML image embeds (`<img src="…">`) also count as embeds — a
# body that embeds via raw HTML must not false-positive the
# linked-figure WARN. Quoted src only (unquoted src is unused in bodies).
_HTML_IMG_SRC_RE = re.compile(r"<img\s[^>]*?src\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)


def _http_head_status(url: str, timeout: float = 5.0) -> int | None:
    """HTTP HEAD ``url``; return the response status code (HTTPError codes
    included), or None when the probe is unavailable — network error /
    timeout / ``EPM_VERIFY_BODY_NO_HTTP=1`` (the test suite sets the env
    var in ``tests/conftest.py`` so unit tests never touch the network).
    Callers treat None as indeterminate, never a FAIL."""
    if os.environ.get("EPM_VERIFY_BODY_NO_HTTP") == "1":
        return None
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _figure_url_existence(url: str, *, noun: str = "figure URL") -> tuple[str, str]:
    """Existence probe for one absolute figure URL (check 4b).

    Returns ``(verdict, note)`` with verdict one of ``'pass'`` / ``'fail'``
    (definitively missing — the URL 404s) / ``'skip'`` (indeterminate —
    surfaced as an `unverified` note on the PASS line, never a FAIL, so
    offline runs don't block). Same-repo SHA-pinned
    ``raw.githubusercontent.com`` URLs resolve offline + deterministically
    via ``_git_object_exists`` (fetch-free); unknown SHAs (un-fetched or
    fabricated) and other hosts fall back to one HTTP HEAD per unique URL.

    ``noun`` names the URL kind in the FAIL notes — check 4b keeps the
    default ``"figure URL"``; check 8b reuses this probe for raw URLs in
    `## Reproducibility` with ``noun="Reproducibility URL"``.
    """
    m = _RAW_GITHUB_FIGURE_RE.match(url)
    if m and (m.group("owner").lower(), m.group("repo").lower()) == _THIS_REPO_SLUG:
        repo = _resolve_repo_root()
        if repo is not None:
            verdict, _detail = _git_object_exists(repo, m.group("sha"), m.group("path"))
            if verdict == "pass":
                return "pass", ""
            if verdict == "fail":
                return (
                    "fail",
                    f"{noun} 404s — `{m.group('path')}` does not exist at `{m.group('sha')[:8]}`",
                )
            # 'skip': sha unknown to the local object database — fall
            # through to the HTTP probe, which decides for real shas
            # pushed from elsewhere and 404s for fabricated ones.
    code = _http_head_status(url)
    if code is None:
        return "skip", f"`{url}` (HTTP probe unavailable)"
    if code == 404:
        return "fail", f"{noun} 404s — `{url}`"
    if code < 400:
        return "pass", ""
    return "skip", f"`{url}` (HTTP {code})"


def check_figure_url_resolvable(body: str) -> CheckResult:
    """Check 4b: every image URL inline under `## TL;DR` must be a
    permanent, dashboard-resolvable URL — and the target must actually
    exist.

    The EPS dashboard serves task-folder HTML artifacts but NOT PNG/PDF
    binaries under `tasks/<N>/artifacts/`, so a relative `artifacts/hero.png`
    reference renders as a broken image in the browser (incident: task #365,
    2026-05-22). Acceptable patterns are absolute URLs only — typically
    `https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/.../*.png`
    or any other `https://...` URL the browser can fetch directly.

    Existence verification (added 2026-06-09, incident task #507: the body
    cited a SHA-pinned figure that was never generated or committed, the
    URL-shape check PASSed, and the dashboard rendered a broken image):
    same-repo `raw.githubusercontent.com` URLs are checked offline +
    deterministically via `git cat-file -e <sha>:<path>`; a definitive miss
    (the sha resolves locally but the path is absent from its tree) FAILs.
    Unknown SHAs and other hosts fall back to ONE `HTTP HEAD` per unique
    URL (5s timeout): a definitive 404 FAILs; any network error / timeout /
    non-404 error status surfaces as an `unverified` note on the PASS line
    — never a FAIL — so offline runs don't block.
    """
    urls = _gather_figure_image_urls(body)
    if not urls:
        # Image-present check (check 4) handles the missing-image case; if
        # there is no image at all, treat this check as vacuously passing so
        # the operator sees one error message, not two.
        return CheckResult("Figure URL resolvable", True, "no images to check")
    bad: list[str] = []
    unverified: list[str] = []
    probed: dict[str, tuple[str, str]] = {}
    for url in urls:
        url = url.strip()
        # Strip optional title — `(url "title")` — keep only the URL token.
        url = url.split(None, 1)[0] if url else url
        if not url:
            bad.append("empty URL")
            continue
        if url.startswith(("http://", "https://")):
            # Permanence rule for GitHub raw URLs — match the spirit of
            # check_repro_url_permanence (no moving branches in the path).
            if re.search(
                r"^https?://raw\.githubusercontent\.com/[^/]+/[^/]+/(main|master|HEAD)\b",
                url,
            ):
                bad.append(f"figure URL pinned to moving ref: `{url}`")
                continue
            # Existence probe — at most one git subprocess / HTTP HEAD per
            # unique URL (incident: task #507).
            if url not in probed:
                probed[url] = _figure_url_existence(url)
            verdict, note = probed[url]
            if verdict == "fail":
                bad.append(note)
            elif verdict == "skip":
                unverified.append(note)
            continue
        # Anything not absolute is rejected — relative `artifacts/...`,
        # `tasks/...`, `figures/...`, `./...`, `../...` all render broken
        # on the dashboard. Push the file to GitHub (typically under
        # `figures/issue_<N>/`) and reference it via the raw URL pinned
        # to a commit SHA.
        bad.append(
            f"figure URL is relative (`{url}`) — push to `figures/issue_<N>/` "
            "and reference via `https://raw.githubusercontent.com/.../<sha>/...`"
        )
    if bad:
        return CheckResult("Figure URL resolvable", False, "; ".join(bad))
    detail = f"{len(urls)} URL(s)"
    if unverified:
        detail += f"; {len(unverified)} unverified (existence not confirmed): " + "; ".join(
            unverified
        )
    return CheckResult("Figure URL resolvable", True, detail)


def _issue_figure_paths_by_issue(body: str) -> dict[str, set[str]]:
    """Same-repo `figures/issue_<N>/...` figure paths inline under the
    result-narrative section, grouped by the issue number carried in the
    path itself (check 29's scope). Other hosts / other repos /
    non-issue-dir paths are out of scope and skipped."""
    paths_by_issue: dict[str, set[str]] = {}
    for url in _gather_figure_image_urls(body):
        url = url.strip()
        # Strip optional title — `(url "title")` — keep only the URL token
        # (check-4b idiom).
        url = url.split(None, 1)[0] if url else url
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if not m or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue
        pm = _ISSUE_FIGURE_PATH_RE.match(m.group("path"))
        if not pm:
            continue
        paths_by_issue.setdefault(pm.group("issue"), set()).add(m.group("path"))
    return paths_by_issue


def check_figure_tracked_at_head(body: str) -> CheckResult:
    """Check 29 (WARN): body-linked same-repo `figures/issue_<N>/...` figure
    paths must still be tracked on a LIVE local ref — HEAD of the resolved
    (main-pinned) repo root, or the issue's local branch family
    `issue-<N>` / `issue-<N>-*`.

    Check 4b verifies existence at the PINNED sha only, and a pinned raw URL
    is immutable — it keeps rendering after the file falls out of tracking
    on every live ref, so a merge/rebase can silently drop the canonical
    figure files with zero verifier signal (incident #841: three
    `figures/issue_841/` stems tracked at the pinned sha but untracked at
    branch HEAD). Three-state read, per path:

    - tracked at HEAD -> clean PASS;
    - BRANCH-ONLY (absent from HEAD, present on >=1 family ref) -> PASS with
      an explicit disclosure note, never a WARN: this is the EXPECTED state
      of every pre-merge verification round (the verifier's HEAD is
      main-pinned while figures live on the issue branch), so WARNing would
      spam every figure-adding round; the disclosure keeps the post-merge
      stale-branch-masks-main-loss state visible and names the recovery;
    - missing from EVERY successfully-probed ref -> the incident-class WARN
      (`is_warn=True`, `passed=True` -- this check can never FAIL: the
      pinned URL still renders, so this is drift hygiene, and grandfathered
      bodies must not regress).

    Conservative on probe failure: if ANY git probe for an issue dir fails,
    that issue degrades to a `probe failure` skip note and can never WARN --
    the path might live at the failed ref, so a narrowed ref set must not
    manufacture a WARN. Per-issue continue (other issue dirs still
    evaluated); fail-soft everywhere; no network. The issue number comes
    from the `figures/issue_<N>/` path itself (not `--issue`), so
    cross-issue figure links are checked against their OWN branch family.
    (#964)
    """
    name = "figure tracked at live refs"
    paths_by_issue = _issue_figure_paths_by_issue(body)
    if not paths_by_issue:
        return CheckResult(name, True, "no same-repo `figures/issue_<N>/` figure URLs to check")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(name, True, "skipped — repo root unresolved (running outside the repo)")

    missing: list[str] = []
    branch_only: list[str] = []
    skipped: list[str] = []
    n_at_head = 0
    for issue_n, paths in sorted(paths_by_issue.items()):
        prefix = f"figures/issue_{issue_n}/"
        family = _git_issue_branch_family(repo, issue_n)
        probes: dict[str, set[str] | None] = {"HEAD": _git_tracked_under(repo, "HEAD", prefix)}
        for br in family or []:
            probes[br] = _git_tracked_under(repo, br, prefix)
        failed = sorted(label for label, tracked in probes.items() if tracked is None)
        if family is None:
            failed.append("branch listing (for-each-ref)")
        if failed:
            # CONSERVATIVE: any failed probe for this issue -> no WARN
            # possible (the path might live at the failed ref); skip note,
            # continue to the next issue dir.
            skipped.append(f"`{prefix}` — probe failure ({', '.join(failed)}); drift not assessed")
            continue
        head_set = probes["HEAD"]
        assert head_set is not None  # failed-probe branch above already continued
        ok_labels = ", ".join(f"`{label}`" for label in sorted(probes))
        for p in sorted(paths):
            if p in head_set:
                n_at_head += 1
                continue
            holders = sorted(
                label
                for label, tracked in probes.items()
                if label != "HEAD" and tracked is not None and p in tracked
            )
            if holders:
                branch_only.append(f"`{p}` (on {', '.join(holders)}, not at HEAD)")
            else:
                missing.append(
                    f"body-linked figure `{p}` is tracked at its pinned-SHA URL but MISSING "
                    f"from every live local ref ({ok_labels}) — the immutable pinned URL "
                    "still renders, so this tracking loss is otherwise silent (incident "
                    "#841); restore with `git restore --source=<pinned-sha> -- <path>` and "
                    "commit on the intended live branch"
                )
    suffix = ""
    if branch_only:
        suffix += (
            "; BRANCH-ONLY (not at HEAD/main — expected pre-merge): "
            + ", ".join(branch_only)
            + ". If this task is already merged/completed, the canonical copy is missing "
            "from main — restore with `git restore --source=<pinned-sha> -- <path>` and "
            "commit on main."
        )
    if skipped:
        suffix += "; skipped: " + "; ".join(skipped)
    if missing:
        return CheckResult(name, True, "; ".join(missing) + suffix, is_warn=True)
    return CheckResult(name, True, f"{n_at_head} figure path(s) tracked at HEAD" + suffix)


def _cited_issue_figure_dirs(body: str) -> dict[str, set[str]]:
    """Map `figures/issue_<K>/` dir prefix → the set of SHAs the body's
    inline same-repo figure URLs pin for that dir (check 31 input)."""
    cited: dict[str, set[str]] = {}
    for url in _gather_figure_image_urls(body):
        url = url.strip().split(None, 1)[0] if url.strip() else ""
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if not m or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue
        pm = _ISSUE_FIGURE_PATH_RE.match(m.group("path"))
        if not pm:
            continue
        cited.setdefault(f"figures/issue_{pm.group('issue')}/", set()).add(m.group("sha"))
    return cited


def _referenced_figure_paths(body: str) -> set[str]:
    """Repo-relative paths of EVERY same-repo image URL anywhere in the
    body — broader than the result-narrative scan, so an embed in
    `## Methodology` still counts as referenced (check 31)."""
    referenced: set[str] = set()
    for url in _IMAGE_RE.findall(body):
        u = url.strip().split(None, 1)[0] if url.strip() else ""
        m = _RAW_GITHUB_FIGURE_RE.match(u)
        if m:
            referenced.add(m.group("path"))
    return referenced


def check_orphaned_per_unit_figures(body: str, *, issue: int | None = None) -> CheckResult:
    """Check 31 (WARN, #1011): a committed per-unit companion PNG at a
    body-cited figure SHA is unreferenced by any body image URL.

    INVERSE direction of checks 4b/22/29 (which verify what the body
    CITES): enumerate what the body's OWN cited commits contain under
    `figures/issue_<N>/` (one `git ls-tree` per unique (SHA, dir) pair,
    10 s timeout, no network) and WARN on any committed `.png` whose
    basename stem matches `_PER_UNIT_FIG_RE` that (a) no body image URL
    references — repo-relative path equality, SHA-independent, so an
    orphan committed at SHA A but embedded via a URL pinned at SHA B
    still counts as referenced — and (b) whose stem appears nowhere in
    the body text (the prose disclosure/exemption escape: naming the
    file silences the WARN).

    Deliberately NARROW pattern (`per{context,unit,cell}` with `-`/`_`
    variants): `per_source` / `per_seed` / `per_question` / `indiv`
    names do NOT match, by design — `indiv` names the per-question
    REGIME in this project (#928's pooled hero `mlp_indiv_hero_4arm.png`),
    and the substantive per-unit-data judgment belongs to
    clean-result-critic Lens 11; this check is only its mechanical
    backstop (incident #928: `mlp_indiv_percontext_delta.png` sat
    committed-but-unembedded at a body-cited SHA through three review
    passes).

    Issue scoping: with `issue` known (`--issue <N>` / a numeric-parent
    `--file`), ONLY `figures/issue_<issue>/` is scanned — a cross-issue
    embed must not surface ANOTHER task's orphans. `issue=None`
    (`--body-stdin`, a non-task-layout `--file`) falls back to scanning
    every cited `figures/issue_<K>/` dir, which CAN surface another
    issue's orphans on a cross-issue embed (documented caveat of the
    fallback).

    Fail-soft inventory: a WARN keeps `passed=True` (the overall verdict
    can never flip); an unreachable/unknown SHA is silently skipped
    (counted in the PASS detail, never a WARN); repo unresolved →
    skip-PASS; no cited same-repo figure URLs → vacuous PASS. PNG-only
    (`.pdf` / `.meta.json` sidecars never flagged); orphans deduped by
    path across cited SHAs.
    """
    name = "per-unit companion figures embedded"
    # (1) cited (sha, issue-dir) pairs from the inline figure URLs.
    cited = _cited_issue_figure_dirs(body)
    if not cited:
        return CheckResult(name, True, "no same-repo `figures/issue_<N>/` figure URLs to check")
    # (2) issue scoping: --issue / numeric-parent-dir mode scans ONLY this
    # task's dir (a cross-issue embed must not surface ANOTHER task's
    # orphans); issue=None (--body-stdin) falls back to every cited dir.
    if issue is not None:
        cited = {k: v for k, v in cited.items() if k == f"figures/issue_{issue}/"}
        if not cited:
            return CheckResult(name, True, "no cited figure URLs under this task's figures dir")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(name, True, "skipped — repo root unresolved (running outside the repo)")
    # (3) referenced set: EVERY image URL anywhere in the body (broader than
    # the result-narrative scan — an embed in ## Methodology still counts).
    referenced_paths = _referenced_figure_paths(body)
    # (4) enumerate per-unit PNGs at each reachable cited sha; union per dir.
    orphans: dict[str, list[str]] = {}  # path -> short-shas found at
    n_unreachable = 0
    for prefix, shas in sorted(cited.items()):
        for sha in sorted(shas):
            tracked = _git_tracked_under(repo, sha, prefix)  # ls-tree; None = unreachable
            if tracked is None:
                n_unreachable += 1  # hard constraint: skip SILENTLY, no WARN
                continue
            for p in tracked:
                base = p.rsplit("/", 1)[-1]
                if not base.lower().endswith(".png"):
                    continue
                stem = base[: -len(".png")]
                if not _PER_UNIT_FIG_RE.search(stem):
                    continue
                if p in referenced_paths:  # path match — SHA-independent by construction
                    continue
                if stem in body:  # prose disclosure/exemption escape
                    continue
                orphans.setdefault(p, []).append(sha[:8])
    if orphans:
        listed = "; ".join(
            f"`{p}` (committed at {', '.join(sorted(set(shas)))})"
            for p, shas in sorted(orphans.items())
        )
        return CheckResult(
            name,
            True,
            f"{len(orphans)} committed per-unit figure(s) at body-cited SHA(s) are not "
            f"embedded by any body image: {listed} — embed the per-unit companion under "
            "the relevant `### <result>`, or state the exemption in prose (naming the "
            "file silences this WARN); substantive owner: clean-result-critic Lens 11 "
            "(incident #928)",
            is_warn=True,
        )
    detail = "no orphaned per-unit figures at body-cited SHAs"
    if n_unreachable:
        detail += f" ({n_unreachable} cited SHA(s) not locally reachable — skipped)"
    return CheckResult(name, True, detail)


def check_linked_not_embedded_figures(body: str, *, issue: int | None = None) -> CheckResult:
    """Check 38 (WARN, v4-only, #1371): a non-image markdown LINK in the
    (footer-truncated) v4 `## Results` section to a
    `figures/issue_<N>/*.png` that no body image embeds.

    INVERSE-complement of check 31: 31 asks "what did the body-cited
    commits CONTAIN that the body never shows?" (git-backed); this check
    asks "what does the Results prose LINK TO that the body never
    embeds?" (pure text — no git / network / subprocess). They compose:
    31 catches committed-but-never-mentioned per-unit PNGs, but its
    stem-in-prose escape is satisfied by a markdown LINK's own URL text,
    so a linked-not-embedded figure silences 31 — this check closes
    exactly that hole (incident #1315: result 4 referenced a committed
    per-row PC-scatter grid as `[text](…png)` instead of `![alt](…)`;
    only clean-result-critic Lens 11 caught it).

    Pipeline: `_v4_results_body` (footer-truncated, so footer blob links
    are never scanned) → `_prose_layer` (fences + `<details>` stripped —
    a quoted or dropdown-tucked link never WARNs; dropdown-tucked links
    are deliberate presentation, a named recall sacrifice) → mask image
    embeds with `_IMAGE_RE.sub("")` → scan the remaining `_MD_LINK_RE`
    links for `_LINKED_ISSUE_PNG_RE` figure paths → subtract the
    whole-body EMBEDDED set. Blockquote caption lines stay in the prose
    layer, so a caption link to an unembedded PNG deliberately WARNs.

    The EMBEDDED set is symmetric any-URL-form: the repo-relative
    `figures/issue_<K>/…png` path is extracted from EVERY image-embed
    URL anywhere in the body — markdown `![alt](url)` AND HTML
    `<img src="url">` — raw-GitHub, blob, and relative forms alike
    (SHA-independent, case-folded path equality). Deliberately NOT
    `_referenced_figure_paths`, which is raw-GitHub-only: check 4b is
    Results-scoped and accepts non-raw absolute URLs, so a legitimate
    Methodology-placed or blob-URL embed would otherwise false-positive
    this WARN. The embedded set scans the UNSTRIPPED whole body, so a
    fenced EXAMPLE embed can silence a real WARN — a false-negative-only
    direction, accepted for WARN tier.

    A clickable-image wrapper `[![alt](p)](q)` leaves a residue link
    `[](q)` after masking; the residue IS scanned — the common q==p case
    is silenced by the embed subtraction (p is embedded), while a
    wrapper whose click target q is a PNG embedded nowhere deliberately
    WARNs (q is linked-not-embedded by definition).

    Issue scoping mirrors check 31: with `issue` known, ONLY
    `figures/issue_<issue>/` links are scanned (a Results link to a
    PARENT task's figure is a legitimate cross-reference — embedding
    another task's figure is not this body's duty); `issue=None`
    (`--body-stdin`, a non-task-layout `--file`) falls back to scanning
    every issue dir, which CAN flag a cross-issue link (documented
    caveat of the fallback). PNG-only: a PDF cannot render inline in
    dashboard markdown, so a PDF link is the only correct reference form
    and must never WARN (check 31's PNG-only rule). Named recall
    sacrifice: reference-style links `[text][ref]` are not matched by
    `_MD_LINK_RE`. WARN-tier: `passed=True` always — the overall verdict
    can never flip; a deliberate link ships under the standing
    acknowledge-in-body WARN rule.
    """
    name = "Results figures embedded, not linked"
    if not is_v4(body):
        return CheckResult(
            name, True, "not a v4 body — the linked-figure scan is v4-only (forward-only)"
        )
    results_text = _v4_results_body(body)
    if results_text is None:
        return CheckResult(name, True, "no `## Results` section to scan")
    prose = _prose_layer(results_text)  # strip fences + <details>
    masked = _IMAGE_RE.sub("", prose)  # drop image embeds (wrapper residue links stay — docstring)
    # Whole-body EMBEDDED set: markdown image embeds + HTML <img> embeds,
    # any URL host form, reduced to case-folded repo-relative figure paths.
    embedded: set[str] = set()
    image_urls = [m.group(1) for m in _IMAGE_RE.finditer(body)]
    image_urls.extend(m.group(1) for m in _HTML_IMG_SRC_RE.finditer(body))
    for raw_url in image_urls:
        iu = raw_url.strip().split(None, 1)[0] if raw_url.strip() else ""
        ipm = _LINKED_ISSUE_PNG_RE.search(iu)
        if ipm:
            embedded.add(ipm.group("path").lower())
    linked: dict[str, None] = {}  # ordered de-dupe: path -> None
    for m in _MD_LINK_RE.finditer(masked):
        url = m.group(1).strip().split(None, 1)[0] if m.group(1).strip() else ""
        pm = _LINKED_ISSUE_PNG_RE.search(url)
        if not pm:
            continue
        if issue is not None and pm.group("issue") != str(issue):
            continue  # cross-issue links are legitimate references
        path = pm.group("path")
        if path.lower() in embedded:
            continue  # embedded anywhere in the body → discipline satisfied
        linked.setdefault(path)
    if linked:
        listed = ", ".join(f"`{p}`" for p in linked)
        return CheckResult(
            name,
            True,
            f"{len(linked)} committed figure(s) referenced as a markdown LINK in `## Results` "
            f"but embedded by no body image: {listed} — embed it inline under the relevant "
            "`### <result>` (an `![...](...)` embed anywhere in the body silences this WARN), "
            "or acknowledge the deliberate link in the body (the standing WARN-ship rule); "
            "substantive owner: clean-result-critic Lens 11 (incident #1315)",
            is_warn=True,
        )
    return CheckResult(
        name, True, "no linked-but-unembedded `figures/issue_<N>/` PNGs in `## Results`"
    )


def check_figure_caption(body: str) -> CheckResult:
    """Check 5: figure caption sanity (vacuous under the 2-content-section spec).

    Under the 2-content-section spec (2026-W22, task #454) a stray
    `## Figure` H2 is rejected by check 2 as a hard FAIL, so this check
    has nothing to scan and always PASSes. Figure captions inside each
    result H3 wrap in markdown blockquotes (`> **Figure.** *...* ...`)
    by analyzer convention; `clean-result-critic` enforces the
    blockquote shape semantically. Retained as a hook for future
    tightening; deleting it would shift CHECKS indices and break
    downstream tests.
    """
    del body
    return CheckResult(
        "Figure caption sanity",
        True,
        "no `## Figure` H2 expected — captions live in blockquote form under each result H3",
    )


def is_v2_nested_design(body: str) -> bool:
    """Return True when `body` carries the `<!-- clean-result-v2 -->`
    sentinel as a real document-level marker, signaling the nested-TL;DR
    design (Motivation / What I ran / Findings → `#### <finding>` per
    result) with confidence in the H1 title tag only (no body
    `Confidence:` sentence required).

    Strips fenced code blocks AND `<details>...</details>` blocks
    before the substring scan, so a body that only QUOTES
    `<!-- clean-result-v2 -->` inside an illustrative code fence (e.g.
    the clean-result body skeleton, analyzer-section-reference.md
    § Step 4) or inside a `<details>` example
    block is NOT misdetected as v2. The sentinel must live at the
    document-level prose layer to count.

    Forward-only marker. Bodies without the sentinel keep the prior
    post-#454 behavior and are NEVER hard-FAILed by the nested-shape
    rule or the no-body-Confidence permission.
    """
    return CLEAN_RESULT_V2_SENTINEL in _prose_layer(body)


def _prose_layer(body: str) -> str:
    """Return `body` with fenced code blocks AND `<details>...</details>`
    blocks stripped, so a sentinel quoted only inside an illustrative
    code fence or example block does NOT count as a document-level
    marker. Shared by `is_v2_nested_design` / `is_v3` so both detect at
    the same prose layer.
    """
    # Strip fenced code blocks (``` ``` and ~~~ ~~~) inline rather
    # than importing the later-defined `_strip_fenced_blocks` (avoids
    # forward-reference ordering noise).
    lines = body.splitlines()
    in_fence = False
    fence_stripped: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        fence_stripped.append(line)
    cleaned = "\n".join(fence_stripped)
    # Strip `<details>...</details>` blocks (already defined regex).
    return _DETAILS_BLOCK_RE.sub("", cleaned)


def is_v3(body: str) -> bool:
    """Return True when `body` carries the `<!-- clean-result-v3 -->`
    sentinel as a real document-level marker (the v3 five-flat-H2
    redesign: Takeaways / What I ran / Findings / Data / Reproducibility,
    confidence in the H1 title tag only).

    Mirrors `is_v2_nested_design`'s fence-aware + `<details>`-aware
    detection so a body that only QUOTES the sentinel inside an
    illustrative code fence or `<details>` example block is NOT
    misdetected as v3. Forward-only: bodies without the sentinel keep
    their existing (v2 / legacy) verification behaviour.
    """
    return CLEAN_RESULT_V3_SENTINEL in _prose_layer(body)


def is_v4(body: str) -> bool:
    """Return True when `body` carries the `<!-- clean-result-v4 -->`
    sentinel as a real document-level marker (the v4 four-flat-H2
    redesign: Takeaways / Goal / Methodology / Results + a `**Repro:**`
    / `**Context:**` footer, confidence in the H1 title tag only).

    Mirrors `is_v3`'s fence-aware + `<details>`-aware detection so a body
    that only QUOTES the sentinel inside an illustrative code fence or
    `<details>` example block is NOT misdetected as v4. Forward-only:
    bodies carrying a v3 / v2 / no sentinel keep their existing
    verification behaviour and are never hard-FAILed by a v4 rule.
    """
    return CLEAN_RESULT_V4_SENTINEL in _prose_layer(body)


def is_titletag_confidence(body: str) -> bool:
    """True for ANY structured clean-result body that carries confidence
    in the H1 title tag only (no body `Confidence:` sentence) and pins
    its provenance / lr-vs-plan checks on a sentinel — i.e. v2 OR v3 OR
    v4.

    Used by the SENTINEL-KEYED checks that must run identically across
    all current generations: check 6 (confidence-title-only), check 16
    (lr matches plan), check 17 (Context provenance row). The nested-
    TL;DR-SHAPE check (check 3b, `check_tldr_nested_structure`)
    deliberately does NOT use this — it stays v2-only, because v3 / v4
    bodies have no `## TL;DR` umbrella to shape.
    """
    return is_v2_nested_design(body) or is_v3(body) or is_v4(body)


# Backwards-compatible alias: the v2/v3-era name for the title-tag-only
# confidence gate. New code should call `is_titletag_confidence`; the
# alias is kept so existing call-sites + tests keep working.
def is_nested_design(body: str) -> bool:
    """Alias for `is_titletag_confidence` (v2 OR v3 OR v4). Retained for
    backwards compatibility with existing call-sites and tests."""
    return is_titletag_confidence(body)


def check_confidence_matches(body: str) -> CheckResult:
    """Check 6: `Confidence: …` line matches the title.

    Under the nested-design (v2) AND v3 shapes (sentinel
    `<!-- clean-result-v2 -->` or `<!-- clean-result-v3 -->` present),
    the H1 title tag is the single source of truth for confidence —
    bodies do NOT carry a `Confidence: …` sentence by design. This
    check PASSes for such bodies whenever the title carries the
    `(... confidence)` tag, regardless of whether a body `Confidence:`
    sentence exists. If such a body DOES happen to carry one (legacy
    holdover), the level must match the title and ≥20 chars of
    rationale after the dash must be present (same rule as legacy
    bodies).

    Legacy bodies (no sentinel) must still ship the
    `Confidence: LOW|MODERATE|HIGH — <rationale>` line somewhere
    (typically as the last paragraph of `## Reproducibility`).
    """
    title = find_h1_title(body) or ""
    m = re.search(r"\((LOW|MODERATE|HIGH) confidence\)\s*$", title)
    label_name = "Confidence sentence matches title"
    if not m:
        return CheckResult(label_name, False, "no title confidence")
    title_level = m.group(1)
    v2 = is_nested_design(body)
    # Whole-body scan so the Confidence sentence can live anywhere it
    # makes sense under the new spec (typically in `## Reproducibility`).
    # Look for `Confidence: LOW|MODERATE|HIGH — <rationale>` (em-dash or
    # ASCII hyphen; en-dash deliberately excluded — em-dash is the spec).
    cm = re.search(
        r"Confidence:\s*(LOW|MODERATE|HIGH)\b\s*[—\-]\s*(.+?)(?:\n\n|\Z|\n##)",
        body,
        flags=re.DOTALL,
    )
    if not cm:
        # Try the looser form (no dash) — still flag the level mismatch / missing
        # rationale separately so the user sees what's wrong.
        loose = re.search(r"Confidence:\s*(LOW|MODERATE|HIGH)\b", body)
        if not loose:
            if v2:
                # v2/v3 nested-design bodies legitimately have no
                # Confidence sentence — the H1 title tag is the source
                # of truth.
                return CheckResult(
                    label_name,
                    True,
                    f"nested-design (v2/v3 sentinel present); title carries "
                    f"`({title_level} confidence)` tag — no body `Confidence:` "
                    "sentence required",
                )
            return CheckResult(
                label_name,
                False,
                "no `Confidence: LOW|MODERATE|HIGH — <rationale>` line found anywhere in the body "
                "(typically lives as the last paragraph of `## Reproducibility`)",
            )
        return CheckResult(
            label_name,
            False,
            f"`Confidence: {loose.group(1)}` line missing the `— <rationale>` clause",
        )
    body_level = cm.group(1)
    rationale = cm.group(2).strip()
    # Trim trailing markdown noise / multiple lines down to a single rationale clause.
    rationale = rationale.split("\n\n")[0].strip()
    if body_level != title_level:
        return CheckResult(
            label_name,
            False,
            f"title says {title_level}, body says {body_level}",
        )
    if len(rationale) < MIN_CONFIDENCE_RATIONALE_CHARS:
        return CheckResult(
            label_name,
            False,
            f"rationale after `—` is only {len(rationale)} chars "
            f"(need ≥{MIN_CONFIDENCE_RATIONALE_CHARS}): {rationale[:60]!r}",
        )
    return CheckResult(
        label_name,
        True,
        f"both {title_level}, rationale={len(rationale)} chars",
    )


# ─── v4 footer resolver ──────────────────────────────────────────────────────


def _v4_footer_text(body: str) -> str | None:
    """Return the v4 `**Repro:**` / `**Context:**` bold-label footer text.

    The v4 body replaces the v3 `## Reproducibility` H2 with a compact
    bold-label footer (NOT an H2): a `---` horizontal rule followed by a
    `**Repro:**` block and a `**Context:**` block, sitting after the last
    H2 section (`## Results`). The footer is everything from the first
    `**Repro:**` label to end-of-body. None when no `**Repro:**` label is
    present.

    Fence-aware: a `**Repro:**` inside a fenced code block (an illustrative
    skeleton) is ignored.
    """
    start = _v4_footer_start_line(body)
    if start is None:
        return None
    return "\n".join(body.splitlines()[start:]).strip()


def _v4_footer_start_line(body: str) -> int | None:
    """Return the 0-based line index where the v4 footer begins, or None.

    The footer is `[--- rule] **Repro:** ... **Context:** ...` at the end
    of the body. The anchor is the first non-fenced `**Repro:**` label
    line; if that label is immediately preceded (modulo blank lines) by a
    `---` horizontal rule, the rule line is the start (so the footer text
    and the Results-body truncation both treat the `---` as the boundary).
    Fence-aware: a `**Repro:**` inside a fenced code block (an
    illustrative skeleton) is ignored.
    """
    lines = body.splitlines()
    in_fence = False
    repro_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^\*\*\s*Repro\s*:?\s*\*\*", s):
            repro_idx = i
            break
    if repro_idx is None:
        return None
    # Back up over blank lines, then over a single `---` rule if present.
    j = repro_idx - 1
    while j >= 0 and lines[j].strip() == "":
        j -= 1
    if j >= 0 and lines[j].strip() == "---":
        return j
    return repro_idx


def _v4_results_body(body: str) -> str | None:
    """Return the `## Results` section text TRUNCATED at the v4 footer.

    `section_text(body, "Results")` returns everything from `## Results`
    to end-of-body, because the `**Repro:**` / `**Context:**` footer is NOT
    an H2 — so the footer's prose-like lines bleed into the LAST result's
    prose and inflate `_finding_prose_cap_results` / `check_v4_results_beat`
    (a false ≥180-word hard FAIL on a conforming body; a defeated
    interpretation-beat WARN for the last result). This helper cuts the
    section at the footer boundary so the per-result scans see only the
    real Results content. None when `## Results` is absent.

    The cut is at the ABSOLUTE line index from `_v4_footer_start_line`,
    intersected with the section's `find_h2_sections` line range — NEVER by
    string-matching the footer's first line: when the footer is preceded by
    a `---` rule its first line is `---`, and a legal mid-Results `---`
    rule between results would truncate every later result out of checks
    11b/20/21 (#1109; masked two ≥180-word hard FAILs on #825).
    """
    lines = body.splitlines()
    section: tuple[int, int] | None = None
    for name, start, end in find_h2_sections(body):
        if name.casefold() == "results":
            section = (start, end)
            break
    if section is None:
        return None
    start, end = section
    footer = _v4_footer_start_line(body)
    if footer is None or not (start <= footer < end):
        # No footer, or the footer lies outside this section's range (e.g.
        # a stray H2 after ## Results): plain section text — identical to
        # section_text(body, "Results").
        return "\n".join(lines[start:end]).strip()
    # Cut at the footer start, then drop trailing `---` rule + blank lines
    # just above it (footer chrome, not Results content).
    rlines = lines[start:footer]
    j = len(rlines) - 1
    while j >= 0 and rlines[j].strip() in ("", "---"):
        j -= 1
    return "\n".join(rlines[: j + 1]).strip()


def _repro_section_text(body: str) -> str | None:
    """Return the reproducibility region for the body's generation.

    v4 bodies carry a `**Repro:**` / `**Context:**` bold-label footer
    (NOT an H2); v2 / v3 / legacy bodies carry a `## Reproducibility` H2.
    The footer checks (7/8/8b/9/15/16/17 + the artifact-URL gather) route
    through this so they enforce on whichever shape the body uses.
    """
    if is_v4(body):
        return _v4_footer_text(body)
    return section_text(body, "Reproducibility")


def check_repro_subgroups(body: str) -> CheckResult:
    """Check 7: the reproducibility region carries the required labels.

    v2 / v3: `## Reproducibility` H2 contains all three boldface subgroup
    labels (`**Artifacts:**`, `**Compute:**`, `**Code:**`).
    v4: the `**Repro:**` / `**Context:**` footer must carry `**Repro:**`
    (the artifact / compute / code line) AND `**Context:**` (the
    run-provenance line). The v4 footer collapses the three v3 subgroups
    into the single `**Repro:**` block, so the v4 requirement is the two
    footer labels, not the three v3 sub-labels.
    """
    if is_v4(body):
        footer = _v4_footer_text(body)
        if footer is None:
            return CheckResult(
                "Repro/Context footer present (v4)",
                False,
                "no `**Repro:**` footer label found — the v4 body must close with a "
                "`**Repro:**` block (compute + code SHA + artifact links) and a "
                "`**Context:**` block (run-provenance), after a `---` rule.",
            )
        if not re.search(r"\*\*\s*Context\s*:?\s*\*\*", footer):
            return CheckResult(
                "Repro/Context footer present (v4)",
                False,
                "v4 footer has `**Repro:**` but is missing the `**Context:**` label "
                "(verbatim originating prompt + lineage + created/run dates).",
            )
        return CheckResult("Repro/Context footer present (v4)", True, "Repro + Context")
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility three subgroups present", False, "Reproducibility section missing"
        )
    missing: list[str] = []
    for label in REPRO_SUBGROUPS:
        # Boldface label of the form **Artifacts:** (allow `Artifacts**:` etc.).
        if not re.search(rf"\*\*\s*{re.escape(label)}\s*:?\s*\*\*", repro):
            missing.append(label)
    if missing:
        return CheckResult(
            "Reproducibility three subgroups present",
            False,
            f"missing **bold** labels in Reproducibility: {', '.join(missing)}",
        )
    return CheckResult(
        "Reproducibility three subgroups present", True, "Artifacts + Compute + Code"
    )


def check_repro_url_permanence(body: str) -> CheckResult:
    """Check 8: every URL in `## Reproducibility` is pinned to a permanent ref.

    Covers HF Hub (`/tree/<ref>` etc., not a moving branch), WandB
    (`/runs/<id>`), GitHub HTML (`/blob/<sha>` / `/tree/<sha>`, not
    `main`/`master`/`HEAD`), and — added 2026-06-09 as the #507
    follow-up — `raw.githubusercontent.com` raw URLs, whose ref path
    segment must be a commit SHA, never `main`/`master`/`HEAD` (the
    artifact silently changes under a moving-ref link when the branch
    advances, de-pinning provenance; check 4b already bans the same
    shape for TL;DR figure URLs). ALL scans run on fence-stripped text
    (same fence policy as check 8b: a URL inside a ``` example — e.g. a
    reproduce-command block — is illustrative, not a provenance link;
    unified 2026-06-09, second #507 follow-up). Blockquote lines
    (`>`-prefixed, incl. nested/indented quotes) are stripped too — the
    `**Context:**` row's verbatim originating-prompt quote may cite bare
    URLs that must be preserved verbatim (never paraphrased; #825/#959);
    pinned-link requirements bind on the non-quoted rows. Shape checks
    only — existence probing for same-repo raw URLs is check 8b's job.
    """
    repro = _repro_section_text(body)  # v4 footer or `## Reproducibility` H2
    if repro is None:
        return CheckResult(
            "Reproducibility URL permanence", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    # Every scan below runs on fence-stripped, then blockquote-stripped
    # text: a URL inside a ``` example is illustrative, and a URL inside
    # a `>` blockquote is verbatim-quoted provenance TEXT — neither is a
    # provenance link (policy shared with check 8b; #959).
    scanned = _strip_blockquote_lines(_strip_fenced_blocks(repro))
    # HF Hub URLs must include /tree/<ref>, /blob/<ref>, /raw/<ref>, or @<ref>.
    hf_urls = re.findall(r"https?://huggingface\.co/[^\s\)<>]+", scanned)
    for url in hf_urls:
        if not (
            "/tree/" in url
            or "/blob/" in url
            or "/raw/" in url
            or re.search(r"@[A-Za-z0-9._-]+", url)
        ):
            bad.append(f"unpinned HF URL `{url}` (needs `/tree/<ref>`)")
        elif re.search(r"/(tree|blob|raw)/(main|master|HEAD)\b", url):
            bad.append(f"unpinned HF URL `{url}` (pinned to moving branch)")
    # WandB URLs should be /runs/<id>, /groups/<id>, or /reports/<id>.
    wandb_urls = re.findall(r"https?://(?:www\.)?wandb\.ai/[^\s\)<>]+", scanned)
    for url in wandb_urls:
        if "/runs/" not in url and "/groups/" not in url and "/reports/" not in url:
            bad.append(f"unpinned WandB URL `{url}` (needs `/runs/<id>`)")
    # GitHub URLs should be /blob/<sha> or /tree/<sha>, not /blob/main.
    gh_urls = re.findall(r"https?://github\.com/[^\s\)<>]+", scanned)
    for url in gh_urls:
        if re.search(r"/(blob|tree)/(main|master|HEAD)\b", url):
            bad.append(f"unpinned GitHub URL `{url}` (use `/blob/<sha>`)")
    # Raw GitHub URLs must pin their ref path segment to a commit SHA,
    # never a moving branch — same rule check 4b applies to TL;DR figure
    # URLs. Shape only; existence probing belongs to check 8b.
    raw_urls = re.findall(r"https?://raw\.githubusercontent\.com/[^\s\)<>]+", scanned)
    for url in raw_urls:
        if re.match(r"https?://raw\.githubusercontent\.com/[^/]+/[^/]+/(main|master|HEAD)\b", url):
            bad.append(f"unpinned raw GitHub URL `{url}` (pinned to moving ref — use `/<sha>/`)")
    if bad:
        return CheckResult("Reproducibility URL permanence", False, "; ".join(bad))
    return CheckResult("Reproducibility URL permanence", True)


_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
# A pipe-FREE inline code span (no `|` between the backticks). On GFM
# table-row lines these are still protective; pipe-containing spans are not
# (the table parser splits the cell on the unescaped `|` before code-span
# recognition, so the `<` inside such a span is exposed to the scan).
_INLINE_CODE_NO_PIPE_RE = re.compile(r"`[^`\n|]+`")
_AUTOLINK_URL_RE = re.compile(r"<https?://[^>\s]+>")
# `<` immediately followed by a digit (0-9). Catches `p<0.05`, `n<10`,
# `<24 personas`, `<2026-05-28`, etc. — all of which the dashboard's MDX
# parser treats as the start of a JSX tag name and errors with
# "Unexpected character `0` (U+0030) before name". `&lt;0.05` is safe
# (no literal `<` in the source); `<= 10` is safe (next char is `=`);
# `< 10` is safe (next char is whitespace); `<https://...>` is caught
# by `_AUTOLINK_URL_RE` separately.
_LT_DIGIT_RE = re.compile(r"<\d")
# `<|` — a `<` immediately followed by a pipe. Inside a GFM table cell the
# table parser splits on the unescaped `|` before code-span recognition,
# so a `` `<|im_start|>` `` token leaks a bare `<|` that MDX reads as a JSX
# tag start ("Unexpected end of file before name" / "Unexpected character
# `|` before name"). The fix is to escape the inner pipes inside the code
# span: `` `<\|im_start\|>` ``. This pattern is scanned ONLY on table-row
# lines (after pipe-free code spans are stripped), so a non-table inline
# `` `<|im_start|>` `` (which the editor parses fine) does not trip it.
_LT_PIPE_RE = re.compile(r"<\|")


# GFM table delimiter row: `|---|---|`, `:--|:-:|--:`, `---|---`, etc.
# At least TWO cells of dashes (with optional leading/trailing `|` and
# optional `:` alignment markers) separated by an internal `|`. The
# internal `|` is mandatory: it is what distinguishes a real multi-column
# GFM table delimiter from a bare `---` (a markdown thematic break / HR or
# a setext-style H2 underline). Without it, a prose line containing a `|`
# immediately followed by a `---` line was misclassified as a one-column
# table header — so a `` `<|im_start|>` `` code span on that prose line
# tripped a false-positive `<|` flag while the real MDX parser accepted
# the body (regex_failed then overrode the real-parse PASS). Requiring the
# internal `|` rules out single-column "tables"; the rare genuine
# single-column table is still covered by the real-parse backstop.
_TABLE_DELIM_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*\|\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)*\|?\s*$"
)


def _table_row_line_indices(lines: list[str]) -> set[int]:
    """Return the indices of lines that belong to a GFM table block.

    A GFM table is a header row (a `|`-containing line) IMMEDIATELY
    followed by a delimiter row (`_TABLE_DELIM_RE`), then a contiguous run
    of `|`-containing body rows until a blank line or a non-pipe line.
    This is what matters for the table-cell `<|` exposure rule: only on
    these lines does the editor's table parser split the cell on the
    unescaped `|` before code-span recognition. A lone prose line that
    happens to carry a `|` (e.g. `log p(x | y)` inside a list item) is NOT
    a table row and its code spans stay protective.

    Lines inside fenced code blocks are excluded (callers strip fences
    separately, but we guard here too so the delimiter scan can't be
    tricked by a `|---|` shown inside a code fence).
    """
    table_lines: set[int] = set()
    in_fence = False
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        # A table starts at a header row (`|` present, not itself a
        # delimiter) immediately followed by a delimiter row.
        if (
            "|" in stripped
            and not _TABLE_DELIM_RE.match(stripped)
            and i + 1 < n
            and _TABLE_DELIM_RE.match(lines[i + 1].strip())
        ):
            table_lines.add(i)  # header
            table_lines.add(i + 1)  # delimiter
            j = i + 2
            while j < n:
                row = lines[j].strip()
                if row == "" or "|" not in row:
                    break
                if row.startswith("```") or row.startswith("~~~"):
                    break
                table_lines.add(j)
                j += 1
            i = j
            continue
        i += 1
    return table_lines


def _strip_code_for_prose_scan(body: str) -> str:
    """Drop fenced code blocks and inline code spans so prose-only checks
    don't false-positive on autolinks shown as illustration inside
    `` `<https://...>` `` or fenced sample blocks.

    Table-cell exception: on a GFM table-row line (one inside a real table
    block — see ``_table_row_line_indices``), an inline code span that
    itself contains an unescaped `|` is NOT protective — the table parser
    splits the cell on that `|` BEFORE code-span recognition, so the `<`
    it wraps is exposed to MDX as a JSX tag start. On those lines we
    therefore strip only PIPE-FREE code spans, leaving any `<` inside a
    pipe-containing span visible to the scan (so `` `<|im_start|>` `` in a
    real table cell is caught). On non-table lines (and inside fences) all
    inline code spans are stripped as before, so a prose `` `<|im_start|>` ``
    in a list item or paragraph stays protected (it parses fine in the
    editor).
    """
    lines = body.splitlines()
    table_idx = _table_row_line_indices(lines)
    out: list[str] = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if i in table_idx:
            # Strip only pipe-free code spans; a pipe-containing span has
            # its content (incl. any `<`) left in place for the scan.
            out.append(_INLINE_CODE_NO_PIPE_RE.sub("", line))
        else:
            out.append(_INLINE_CODE_RE.sub("", line))
    return "\n".join(out)


_MDX_CHECK_LABEL = (
    "MDX-safe prose — real-parse backstop + no `<https://...>` autolinks, "
    "`<` before digit, or `<|` in table cell"
)

# Node real-parse helper: mirrors the dashboard MDXEditor parse exactly.
# Lives under `dashboard/` because node resolves ESM bare specifiers
# relative to the importing file and the MDX deps exist only in
# `dashboard/node_modules` (see the helper's own module docstring).
_MDX_HELPER_REL = Path("dashboard") / "scripts" / "mdx_parse_check.mjs"
_DASHBOARD_DIR = _HERE.parent / "dashboard"
_MDX_HELPER_PATH = _HERE.parent / _MDX_HELPER_REL


def _run_real_mdx_parse(body: str) -> tuple[str, str]:
    """Run the node real-parse backstop on the already-stripped `body`.

    Returns a (verdict, detail) tuple:
      - ("pass", "")               — node parsed the body cleanly (exit 0).
      - ("fail", "<message+loc>")  — node reported a parse failure (exit 2).
      - ("skip", "<reason>")       — node / helper / deps unavailable; the
                                     caller falls back to regex-only and
                                     appends the reason to the detail.

    The body is passed on stdin (the helper does NOT re-strip frontmatter
    for stdin input — it equals what `split_frontmatter` already produced,
    which is byte-identical to the dashboard's gray-matter `content` for
    the canonical frontmatter shape). cwd is `<repo>/dashboard` so node
    resolves the MDX deps. NEVER returns "pass" on a crash / nonzero
    unexpected exit — that maps to "skip" (parser unavailable), honoring
    the no-silent-fallback rule.
    """
    node = shutil.which("node")
    if node is None:
        return "skip", "node not on PATH"
    if not _MDX_HELPER_PATH.exists():
        return "skip", f"helper not found at {_MDX_HELPER_REL}"
    if not _DASHBOARD_DIR.is_dir():
        return "skip", "dashboard/ directory not found"
    try:
        proc = subprocess.run(
            [node, str(_MDX_HELPER_PATH)],
            input=body,
            cwd=str(_DASHBOARD_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return "skip", f"node invocation failed: {e}"

    if proc.returncode == 3:
        # Helper signalled "parser unavailable" (deps missing / read error).
        reason = (proc.stderr or "").strip().splitlines()
        return "skip", (reason[-1] if reason else "helper reported parser unavailable")
    if proc.returncode not in (0, 2):
        # Any other exit code is a harness anomaly — do NOT silently pass.
        reason = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = reason[-1] if reason else f"exit {proc.returncode}"
        return "skip", f"helper exited {proc.returncode}: {tail}"

    out = (proc.stdout or "").strip()
    try:
        payload = json.loads(out.splitlines()[-1]) if out else {}
    except (json.JSONDecodeError, IndexError):
        return "skip", f"helper produced unparseable output: {out[:120]!r}"

    if proc.returncode == 0 and payload.get("ok") is True:
        return "pass", ""
    if proc.returncode == 2 and payload.get("ok") is False:
        msg = str(payload.get("message", "MDX parse error"))
        line = payload.get("line")
        col = payload.get("column")
        loc = ""
        if isinstance(line, int):
            loc = f" (line {line}" + (f", col {col}" if isinstance(col, int) else "") + ")"
        return "fail", f"real MDX parse failed{loc}: {msg}"
    # returncode / payload disagree → treat as unavailable, never a pass.
    return "skip", f"helper returncode/payload mismatch (exit {proc.returncode}, out {out[:80]!r})"


def _mdx_regex_findings(body: str) -> list[str]:
    """Fast regex pre-check layer for check 14 (no node dependency).

    Returns a list of human-readable finding messages for the three
    regex-detectable MDX-unsafe classes — `<https://...>` autolinks, `<`
    before a digit, and `<|` inside a real GFM table cell. Empty list ==
    the regex layer found nothing. This is the node-independent layer that
    runs in CI without node; the authoritative real-parse backstop is
    layered on top of it in ``check_mdx_safe_urls``.
    """
    stripped = _strip_code_for_prose_scan(body)
    autolinks = _AUTOLINK_URL_RE.findall(stripped)
    lt_digit = _LT_DIGIT_RE.findall(stripped)
    lt_pipe = _LT_PIPE_RE.findall(stripped)

    parts: list[str] = []
    if autolinks:
        unique: list[str] = []
        seen: set[str] = set()
        for h in autolinks:
            if h not in seen:
                seen.add(h)
                unique.append(h)
        sample = ", ".join(unique[:3])
        more = f" (+{len(unique) - 3} more)" if len(unique) > 3 else ""
        parts.append(
            f"{len(autolinks)} `<https://...>` autolink(s) — MDX parses "
            f"`<https://` as JSX and errors with 'Unexpected character `/` "
            f"(U+002F) before local name'. Convert to `[label](url)`. "
            f"Found: {sample}{more}"
        )
    if lt_digit:
        # Surface the surrounding ~20 chars of context for each hit so the
        # operator can locate `p<0.05` / `n<10` / `<24 personas` without
        # grepping the body manually.
        contexts: list[str] = []
        seen_ctx: set[str] = set()
        for m in _LT_DIGIT_RE.finditer(stripped):
            lo = max(0, m.start() - 10)
            hi = min(len(stripped), m.end() + 10)
            ctx = stripped[lo:hi].replace("\n", " ").strip()
            if ctx not in seen_ctx:
                seen_ctx.add(ctx)
                contexts.append(ctx)
        sample = ", ".join(f"…{c}…" for c in contexts[:3])
        more = f" (+{len(contexts) - 3} more)" if len(contexts) > 3 else ""
        parts.append(
            f"{len(lt_digit)} `<` before digit occurrence(s) — MDX parses "
            f"`<0` as JSX and errors with 'Unexpected character `0` "
            f"(U+0030) before name'. Write `p < 0.05` with surrounding "
            f"spaces or wrap the token in backticks (`` `p<0.05` ``). "
            f"Found: {sample}{more}"
        )
    if lt_pipe:
        contexts = []
        seen_ctx = set()
        for m in _LT_PIPE_RE.finditer(stripped):
            lo = max(0, m.start() - 12)
            hi = min(len(stripped), m.end() + 12)
            ctx = stripped[lo:hi].replace("\n", " ").strip()
            if ctx not in seen_ctx:
                seen_ctx.add(ctx)
                contexts.append(ctx)
        sample = ", ".join(f"…{c}…" for c in contexts[:3])
        more = f" (+{len(contexts) - 3} more)" if len(contexts) > 3 else ""
        parts.append(
            f"{len(lt_pipe)} `<|` in table cell — MDX parses `<|` in a table "
            f"cell as a JSX tag start (the table parser splits the cell on the "
            f"unescaped `|` before code-span recognition, exposing the `<`). "
            f"Escape the inner pipes inside the code span, e.g. "
            f"`` `<\\|im_start\\|>` ``. Found: {sample}{more}"
        )
    return parts


def check_mdx_safe_urls(body: str) -> CheckResult:
    """Check 14 (MDX safety): no `<` characters in body prose that the
    dashboard's MDX parser will read as the start of a JSX tag.

    Two layers:

    (A) Fast regex pre-checks (always run). Three classes fail:

      - `<https://...>` markdown autolinks — MDX parses `<https` as a tag
        name and errors with "Unexpected character `/` (U+002F) before
        local name". Use `[label](url)` instead. Incident: task #382,
        2026-05-28.
      - `<` immediately followed by a digit (`p<0.05`, `n<10`, `<24`) —
        MDX parses `<0` as a tag name and errors with "Unexpected
        character `0` (U+0030) before name". Write `p < 0.05` with
        surrounding spaces or wrap the token in backticks. Recurred
        same-day as the autolink incident.
      - `<|` inside a GFM table cell (`` `<|im_start|>` `` in a table
        row) — the table parser splits the cell on the unescaped `|`
        before code-span recognition, so the backticks do NOT protect
        the `<|`, which MDX reads as a JSX tag start. Escape the inner
        pipes: `` `<\\|im_start\\|>` ``. Incident: task #399, 2026-05-28.

      Patterns inside fenced code blocks and inline code spans are exempt
      (the strip step removes them before scanning), EXCEPT pipe-containing
      code spans on table-row lines, whose `<` stays visible (so the
      table-cell `<|` case is caught). `&lt;0.05`, `<= 10`, `< 10`, and
      `<` followed by anything other than `/`, a digit, or a pipe all pass.

    (B) Real-parse backstop (runs when node + the helper + the dashboard
      deps are present). Shells out to `dashboard/scripts/mdx_parse_check.mjs`
      which runs the exact `mdast-util-from-markdown` parse the dashboard
      runs. A real-parse failure FAILs the check with the parser's message
      + line/col EVEN IF every regex passed — this is what makes the
      verifier authoritative. When node / helper / deps are unavailable the
      check falls back to regex-only and appends "(real MDX parse skipped:
      <reason>)" to the detail; it does NOT hard-fail solely because node
      is missing.
    """
    parts = _mdx_regex_findings(body)
    regex_failed = bool(parts)

    # Real-parse backstop. Authoritative when available: a real-parse FAIL
    # fails the check even if the regexes all passed.
    verdict, real_detail = _run_real_mdx_parse(body)
    if verdict == "fail":
        if regex_failed:
            return CheckResult(
                _MDX_CHECK_LABEL,
                False,
                " | ".join(parts) + f" | {real_detail}",
            )
        return CheckResult(_MDX_CHECK_LABEL, False, real_detail)

    if regex_failed:
        # Regex caught it; the real parse either agreed (pass is impossible
        # here in practice) or was unavailable — note the latter for clarity.
        detail = " | ".join(parts)
        if verdict == "skip":
            detail += f" | (real MDX parse skipped: {real_detail})"
        return CheckResult(_MDX_CHECK_LABEL, False, detail)

    # Regexes passed.
    if verdict == "pass":
        return CheckResult(_MDX_CHECK_LABEL, True, "regex + real MDX parse both clean")
    # verdict == "skip": node unavailable — regex-only PASS, flagged.
    return CheckResult(
        _MDX_CHECK_LABEL,
        True,
        f"regex clean (real MDX parse skipped: {real_detail})",
    )


def check_repro_sentinel_scrub(body: str) -> CheckResult:
    """Check 9: no placeholder sentinels (`{{`, `TBD`, `see config`, `default`)
    in `## Reproducibility` (v2/v3) or the `**Repro:**`/`**Context:**` footer (v4)."""
    repro = _repro_section_text(body)  # v4 footer or `## Reproducibility` H2
    if repro is None:
        return CheckResult(
            "Reproducibility sentinel scrub", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    for s in SENTINEL_SUBSTRINGS:
        if s == "{{":
            if "{{" in repro:
                bad.append("`{{` placeholder")
        elif s == "default":
            # Placeholder positions only (bare table cell / label
            # terminator) — see _DEFAULT_PLACEHOLDER_RE. Prose like
            # "default assistant" is substantive, not a sentinel
            # (task #542 false-positive).
            if _DEFAULT_PLACEHOLDER_RE.search(repro):
                bad.append("`default` placeholder value")
        else:
            # Matched case-insensitively as standalone words (avoid false
            # positives from larger identifiers).
            if re.search(rf"\b{re.escape(s)}\b", repro, flags=re.IGNORECASE):
                bad.append(f"`{s}`")
    if bad:
        return CheckResult(
            "Reproducibility sentinel scrub",
            False,
            "; ".join(bad) + " — use `n/a` explicitly for inapplicable fields",
        )
    return CheckResult("Reproducibility sentinel scrub", True)


# A learning-rate-shaped number: `2e-6`, `1e-5`, `1E-4`, `1e-04`, `3.0e-5`,
# or a sub-1 decimal like `0.0001` / `0.00005`. Bare integers (`50`, `100`)
# are EXCLUDED — never a real learning-rate value, and admitting them
# caused the task #514 false-positive where prose `lower-LR 50%-epoch cell`
# parsed `50` as an lr value.
_LR_NUM_SCI = r"[0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+"
_LR_NUM_DEC = r"0\.[0-9]+"
_LR_NUM = rf"(?:{_LR_NUM_SCI}|{_LR_NUM_DEC})"
# Body side — anchored to an explicit `lr` / `learning rate` label, with
# the number connected either by an explicit assignment glyph (`=`, `:`,
# `of`, `is`) or by bare whitespace adjacency (`lr 5e-6`), so the number
# we judge is unambiguously the learning rate (precise, low false
# positive). `\blr\b` does not match `color`, `_lr_`, or `controller`.
# The bare-adjacency form is what per-recipe Parameters-table cells use
# (`| marker recipe | LoRA r32; lr 5e-6 cosine, ... |`, task #537) —
# without it check 16 silently skipped a present value. It stays safe
# against the #514 false positive (`lower-LR 50%-epoch cell`) because
# `_LR_NUM` excludes bare integers, and against cross-cell bleed
# (`| ... at base lr | 0.02 |`) because `\s+` never crosses a `|`
# delimiter.
_LR_ANCHORED_RE = re.compile(
    r"(?:\blr\b|learning[\s_-]*rate)(?:\s*(?:[=:]|\b(?:of|is)\b)\s*|\s+)(" + _LR_NUM + r")",
    flags=re.IGNORECASE,
)
# Table-row form — the canonical v2 Parameters table states the learning
# rate as its own row (`| Learning rate | 5e-6 (inherited verbatim from the
# parent anchor) |`), where label and value are separated by a CELL
# DELIMITER rather than an assignment glyph, so `_LR_ANCHORED_RE` never
# fires and check 16 silently skipped a present value (task #534). The
# label cell must BEGIN with the lr token (after optional emphasis and at
# most two short qualifier words, e.g. `Peak learning rate`,
# `Marker-only LR`) and the value cell must BEGIN with the numeric literal;
# trailing annotations after the number are tolerated because only the
# leading literal is captured. Label cells that merely CONTAIN `lr` deeper
# in (`| Bystander rate at base lr | 0.02 |`) stay unmatched — precision
# over recall, since a false FAIL is worse than a skip.
_LR_TABLE_ROW_RE = re.compile(
    r"\|\s*[*_`]*(?:[A-Za-z][\w()-]*[\s_-]+){0,2}(?:lr\b|learning[\s_-]*rate)[^|\n]*\|"
    r"\s*[*_`]*(" + _LR_NUM + r")",
    flags=re.IGNORECASE,
)
# Plan side (recall) — any scientific-notation token (`Ne-M`). Capturing the
# whole plan's lr surface (chosen lr + control/anchor lrs) keeps the bias
# toward PASS: an over-broad plan set never FAILs a correct body, it only
# risks missing a wrong one. SHAs and hex blobs lack `\b…\b` boundaries
# around an `e±d` run, so they do not leak in.
_SCI_TOKEN_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?[eE][-+]?[0-9]+\b")
# An explicit, author-supplied acknowledgement that the run knowingly used a
# learning rate the plan did not declare. Downgrades the FAIL to WARN. EVERY
# alternative requires the literal word "plan" so generic error-bar prose like
# "standard deviation" / "deviation of the metric" can NEVER silently downgrade
# a real misprint FAIL — the deviation cue and "plan" must co-occur within ~40
# chars (either order).
_LR_DEVIATION_RE = re.compile(
    r"off[-\s]?plan"
    r"|not\s+in\s+the\s+plan"
    r"|(?:deviat\w*|differ\w*|changed?|departs?|swapp?ed?)[^.\n]{0,40}\bplan\b"
    r"|\bplan\b[^.\n]{0,40}(?:deviat\w*|differ\w*|changed?|departs?|swapp?ed?)",
    flags=re.IGNORECASE,
)


def _parse_lr_floats(text: str, *, anchored_only: bool) -> set[float]:
    """Return the set of learning-rate floats found in `text`.

    `anchored_only=True` (body side) collects only numbers tied to an
    explicit `lr` / `learning rate` label — the inline assignment form
    (`lr = 5e-6`, `learning rate of 5e-6`), the bare-adjacency form
    used inside per-recipe Parameters-table cells (`lr 5e-6 cosine`),
    or the dedicated Parameters-table row form
    (`| Learning rate | 5e-6 (annotation) |`). `anchored_only=
    False` (plan side) ALSO collects every scientific-notation token,
    maximizing recall so the reconciliation never FAILs a body whose lr
    the plan really does contain.
    """
    out: set[float] = set()
    for m in _LR_ANCHORED_RE.finditer(text):
        try:
            out.add(float(m.group(1)))
        except ValueError:
            continue
    for m in _LR_TABLE_ROW_RE.finditer(text):
        try:
            out.add(float(m.group(1)))
        except ValueError:
            continue
    if not anchored_only:
        for m in _SCI_TOKEN_RE.finditer(text):
            try:
                out.add(float(m.group(0)))
            except ValueError:
                continue
    return out


def check_repro_lr_matches_plan(body: str, *, plan_path: Path | None = None) -> CheckResult:
    """Check 16: the learning rate stated in `## Reproducibility` must
    appear in the approved plan (any version under `plans/v*.md`).

    Guards against the analyzer hand-typing a plausible-looking
    hyperparameter (a LoRA default from training priors) into the
    Reproducibility Parameters table instead of copying the actual run
    value. Incident: task #489 shipped `lr = 1e-4` while the committed
    training script + plan §11 both ran `lr = 2e-6` — a 50x misprint on
    the most load-bearing hyperparameter, missed by every reviewer
    because nothing reconciled the table's VALUES against ground truth.

    The reconciliation set is the UNION across all `plans/v*.md`
    siblings of ``plan_path``, not just the `plans/plan.md` symlink —
    same-issue follow-up rounds re-point the symlink at the follow-up's
    plan, which may not contain the training lr that grounds the body
    (incident #597). A body lr matching ANY version PASSes.

    Scope: v2 nested-design bodies only (sentinel present); legacy
    bodies are forward-grandfathered. The check is a NO-OP PASS when it
    cannot reconcile (no parseable body lr, no plan on disk, no
    parseable plan lr) so it never newly blocks a body it cannot judge.
    A documented run-vs-plan deviation downgrades the FAIL to WARN.
    """
    name = "Reproducibility lr matches plan"
    if not is_nested_design(body):
        return CheckResult(name, True, "skipped — legacy (pre-v2) body")
    # The body's stated lr lives in the section that carries the parameter
    # table for the generation: v4 puts the COMPLETE hyperparameter table
    # in `## Methodology` (the `**Training:**` slot); v2 / v3 put the
    # (slimmed) table in `## Reproducibility`.
    if is_v4(body):
        repro = section_text(body, "Methodology")
        src_label = "Methodology"
    else:
        repro = _repro_section_text(body)
        src_label = "Reproducibility"
    if repro is None:
        # Missing-section is check_required_sections' job; don't double-FAIL.
        return CheckResult(name, True, f"skipped — no {src_label} section")
    body_lrs = _parse_lr_floats(repro, anchored_only=True)
    if not body_lrs:
        return CheckResult(name, True, f"skipped — no learning rate stated in {src_label}")
    if plan_path is None or not plan_path.exists():
        return CheckResult(name, True, "skipped — no approved plan on disk to reconcile against")
    # Reconcile against the UNION of every plan version (plans/v*.md), not
    # just the plans/plan.md symlink: a same-issue follow-up round re-points
    # the symlink at the follow-up's (often analysis-only) plan, whose
    # unrelated sci-notation tokens (e.g. a `1e-3` tolerance) would then
    # masquerade as "the plan's lr" while the training lr that grounds the
    # body's Parameters table lives in an earlier version (incident #597:
    # a correct lr=5e-6 body drew a spurious WARN against the v2 follow-up
    # plan). Fall back to plan_path itself when no v*.md siblings exist
    # (e.g. a bare plan.md fixture).
    plan_files = sorted(plan_path.parent.glob("v*.md")) or [plan_path]
    plan_lrs: set[float] = set()
    for plan_file in plan_files:
        plan_lrs |= _parse_lr_floats(plan_file.read_text(errors="replace"), anchored_only=False)
    if not plan_lrs:
        return CheckResult(name, True, "skipped — plan declares no parseable learning rate")
    unmatched = [b for b in body_lrs if not any(math.isclose(b, p, rel_tol=1e-6) for p in plan_lrs)]
    if not unmatched:
        return CheckResult(name, True)
    body_str = ", ".join(f"{b:g}" for b in sorted(unmatched))
    plan_str = ", ".join(f"{p:g}" for p in sorted(plan_lrs))
    detail = (
        f"{src_label} states lr {body_str} but the approved plan declares "
        f"{{{plan_str}}}. Copy the actual run lr from the committed training script "
        f"(the `**Code:**` SHA) / plan §11 — never type it from memory. If the run "
        f"genuinely deviated from the plan, document the deviation explicitly in "
        f"`## {src_label}` (downgrades this to WARN)."
    )
    if _LR_DEVIATION_RE.search(repro):
        return CheckResult(name, True, "documented deviation — " + detail, is_warn=True)
    return CheckResult(name, False, detail)


# Check-17 v4 lineage-token scan: the `**Context:**` row must name its
# lineage (SPEC.md § `**Context:**` row; #958). Matched on fence-stripped +
# blockquote-stripped footer text — the blockquoted verbatim prompt often
# MENTIONS issue numbers ("reuse #537 adapters") and must not satisfy this.
_V4_CONTEXT_LINEAGE_TOKEN_RE = re.compile(
    r"(?<![\w/&])#\d+"  # issue ref: `[#K](...)` or bare `#K`
    r"|fresh\s+direction"  # `fresh direction (no parent)` / `...; no parent task`
    r"|\bno\s+parent\b"  # `fresh (no parent)` / `no parent task`
    r"|same-issue\s+follow-?up\s+round",  # follow-up-round lineage clause
    re.IGNORECASE,
)


# ── Check-17 origin-prompt verbatim sub-check (#1068, incident #813 r1) ──
# The **Context:** row promises the originating prompt VERBATIM
# (SPEC.md § `**Context:**` row). The sub-check requires normalized
# frontmatter `origin_prompt` to appear as a SUBSTRING of the normalized
# Context-region text (blockquote markers stripped). Substring — not
# per-segment equality — because the row legitimately carries MORE quoted
# text than the creation prompt (follow-up-round prompts, lineage prose,
# inline quote marks), and because the containment DIRECTION is what
# catches truncation: a truncated quote means the full origin_prompt
# appears NOWHERE in the row. (A quote-inside-origin_prompt test would
# PASS every truncation — direction matters.)

_MD_ESCAPE_RE = re.compile(r'\\([\\`*_{}\[\]()#+\-.!|>~"\'])')

# Leading blockquote-marker run on a line: `>`, `> >`, `  > `, etc.
_BLOCKQUOTE_MARKER_RE = re.compile(r"^\s*(?:>\s?)+")

# Same pattern check 17 already used inline for the label-presence test.
_CONTEXT_LABEL_RE = re.compile(r"\*\*\s*Context\s*:?\s*\*\*")

# An inline quote-mark-delimited span (>= 20 chars): ASCII `"` or the
# curly forms (escapes dodge the ambiguous-unicode lint).
_INLINE_QUOTE_SPAN_RE = re.compile('["\u201c]([^"\u201c\u201d]{20,})["\u201d]', re.S)


def _normalize_prompt_text(s: str) -> str:
    """Whitespace-collapse + unicode-punctuation fold for the check-17
    origin-prompt comparison. Collapses all whitespace runs (incl.
    newlines from wrapped blockquotes) to single spaces, folds NBSP and
    curly quotes/apostrophes to their ASCII forms (editors smart-quote;
    a fold applied to BOTH sides never masks a real word-level edit)."""
    s = s.replace("\u00a0", " ")  # NBSP
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    s = s.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    return " ".join(s.split())


def _unescape_markdown(s: str) -> str:
    """Undo backslash-escapes of markdown punctuation (body side only —
    an analyzer writing an ``origin_prompt`` containing ``**`` or
    backticks into markdown may escape it; the frontmatter value is raw
    text and is never unescaped)."""
    return _MD_ESCAPE_RE.sub(r"\1", s)


def _strip_blockquote_markers(text: str) -> str:
    """Remove leading `>` marker runs per line, KEEPING the line text.

    The opposite transform of ``_strip_blockquote_lines`` (which drops
    the whole line for the #959 URL-scan exemption): here the quoted
    text IS the object under test, only the markers are noise."""
    return "\n".join(_BLOCKQUOTE_MARKER_RE.sub("", ln) for ln in text.splitlines())


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of ``a`` and ``b``."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _context_quote_candidates(region: str) -> list[str]:
    """Extract quoted-prompt CANDIDATES from the Context region for the
    check-17 truncation classifier.

    Candidates: the label line's post-label remainder (`>`-stripped, if
    non-empty), blockquote segments (contiguous `>` runs with markdown
    lazy-continuation lines joined — inclusion fails toward PASS/WARN,
    the safe direction), and inline quote-mark-delimited spans (>= 20
    chars) over the marker-stripped region. Only consulted AFTER the
    containment PASS test failed, to classify the mismatch."""
    lines = region.splitlines()
    candidates: list[str] = []
    segment: list[str] = []

    def _close() -> None:
        if segment:
            candidates.append("\n".join(segment))
            segment.clear()

    if lines:
        first = _BLOCKQUOTE_MARKER_RE.sub("", lines[0]).strip()
        if first:
            candidates.append(first)
    for ln in lines[1:]:
        if ln.lstrip().startswith(">"):
            segment.append(_BLOCKQUOTE_MARKER_RE.sub("", ln))
        elif ln.strip() and segment:
            segment.append(ln)  # markdown lazy continuation joins the quote
        else:
            _close()
    _close()
    candidates.extend(_INLINE_QUOTE_SPAN_RE.findall(_strip_blockquote_markers(region)))
    return candidates


def _origin_prompt_quote_verdict(repro: str, fm: dict) -> tuple[str, str]:
    """Classify the Context row's originating-prompt quote against
    frontmatter ``origin_prompt``.

    Returns ``(status, detail)`` with status one of:

    - ``"noop"`` — no frontmatter ``origin_prompt``, or no
      ``**Context:**`` label in the region;
    - ``"pass"`` — normalized ``origin_prompt`` appears as a substring
      of the normalized, blockquote-marker-stripped Context-region text
      (tested raw AND markdown-unescaped);
    - ``"fail-trunc"`` — a quoted candidate is a strict normalized
      PREFIX of ``origin_prompt``, >= 20 chars AND covering >= 50% of it
      (the #813 r1 / #742 silent-tail-truncation signature);
    - ``"warn-mismatch"`` — containment fails with no truncation
      signature (e.g. the row quotes a spec-sanctioned alternate
      verbatim source; #825).
    """
    op = str(fm.get("origin_prompt") or "").strip()
    if not op:
        return "noop", "no frontmatter origin_prompt"
    m = _CONTEXT_LABEL_RE.search(repro)
    if m is None:
        return "noop", "no **Context:** label"  # caller's label branch makes this unreachable
    region = repro[m.end() :]  # Context row -> end of footer/section
    nop = _normalize_prompt_text(op)
    stripped = _strip_blockquote_markers(region)
    if nop in _normalize_prompt_text(stripped) or nop in _normalize_prompt_text(
        _unescape_markdown(stripped)
    ):
        return "pass", ""
    # Containment failed — classify. Candidates: blockquote segments
    # (contiguous `>` runs, lazy-continuation lines joined) + inline
    # quote-mark-delimited spans (>= 20 chars) in the stripped region.
    candidates = _context_quote_candidates(region)
    best_lcp, truncated = 0, None
    for cand in candidates:
        ncand = _normalize_prompt_text(_unescape_markdown(cand))
        ncand_p = ncand.rstrip(".,;:!?\u2026 ")  # a truncating editor appends `.`/ellipsis (#742)
        best_lcp = max(best_lcp, _common_prefix_len(ncand_p, nop))
        # FAIL guard (plan D10): a strict-prefix candidate must ALSO cover
        # >= 50% of the normalized origin_prompt. Rationale: the incident
        # class (silent tail truncation — #742 at 85%, the #813-r1-shaped
        # fixture at ~60%) characteristically preserves most of the
        # prompt, while the false-positive scenario (an innocent SHORT
        # elided pointer quoting the fm opener alongside a full
        # alternate-source quote — the #825+ shape) characteristically
        # quotes a small head fraction (~10%). Below the floor the case
        # routes to WARN, whose message names the alternate-source
        # escape. Per-candidate + fraction semantics deliberately do NOT
        # suppress on the presence of a longer non-prefix candidate —
        # that variant would false-NEGATIVE the multi-round true positive
        # (truncated creation quote + a longer full round-2 quote).
        if (
            20 <= len(ncand_p) < len(nop)
            and len(ncand_p) >= 0.5 * len(nop)
            and nop.startswith(ncand_p)
        ):
            truncated = ncand_p
    if truncated is not None:
        cut = len(truncated)
        return "fail-trunc", (
            f"context-origin-prompt-mismatch: the quoted originating prompt is a strict "
            f"PREFIX of frontmatter `origin_prompt` — truncated at normalized offset "
            f"{cut}/{len(nop)} (quote ends '...{truncated[-40:]}'; origin_prompt continues "
            f"'{nop[cut : cut + 60]}...'). Quote the FULL origin_prompt verbatim "
            f"(SPEC.md § `**Context:**` row)."
        )
    return "warn-mismatch", (
        f"context-origin-prompt-mismatch: frontmatter `origin_prompt` does not appear "
        f"verbatim (whitespace-normalized) in the `**Context:**` row — first divergence at "
        f"normalized offset {best_lcp}/{len(nop)} (origin_prompt continues "
        f"'{nop[best_lcp : best_lcp + 60]}...'). If the row quotes a spec-sanctioned "
        f"alternate verbatim source (original-body `## Provenance` / an "
        f"`epm:followup-scope` marker), this is informational; otherwise re-quote "
        f"origin_prompt verbatim."
    )


def check_repro_context_provenance(
    body: str, fm: dict, *, original_body_path: Path | None = None
) -> CheckResult:
    """Check 17: v2 bodies carry a `**Context:**` run-provenance row in
    `## Reproducibility`.

    The row ships the run-context provenance: created/run dates,
    follow-up lineage, and the verbatim originating user prompt (or the
    literal ``origin prompt not recorded``). Forward-only (adopted
    2026-06-11): legacy (pre-sentinel) bodies PASS vacuously, so the
    awaiting_promotion backlog never retro-FAILs.

    v4 bodies additionally require a LINEAGE TOKEN in the Context row
    (#1014, origin #958): an issue reference (`[#K](...)` or bare `#K`),
    a `fresh direction` / `no parent` clause, or a same-issue
    follow-up-round clause — scanned on fence-stripped +
    blockquote-stripped text so issue refs inside the blockquoted
    verbatim prompt never satisfy it. A v4 label-present row with no
    lineage token is a hard FAIL. v3/v2 bodies keep the pre-#1014
    label-presence-only behavior verbatim.

    **Origin-prompt verbatim sub-check (#1068, incident #813 r1).** When
    frontmatter ``origin_prompt`` is recorded and the row is present, the
    normalized ``origin_prompt`` must appear as a SUBSTRING of the
    normalized, blockquote-marker-stripped Context-region text (tested
    raw AND markdown-unescaped) — containment in THIS direction is what
    catches truncation: a truncated quote means the full prompt appears
    nowhere in the row (the reverse quote-inside-origin_prompt test
    would PASS every truncation). On a containment failure: a quoted
    candidate that is a >=20-char strict normalized PREFIX of
    ``origin_prompt`` covering >=50% of it is a hard v4 FAIL naming the
    truncation offset (`_origin_prompt_quote_verdict`); any other
    mismatch is a WARN, because SPEC.md sanctions alternate verbatim
    sources (original-body ``## Provenance`` / ``epm:followup-scope``
    markers), so a non-truncation mismatch may be innocent (#825). v3/v2
    bodies get the WARN-only form for BOTH classes (grandfathering —
    never a new hard FAIL below the v4 sentinel). No ``origin_prompt``
    or no ``**Context:**`` label: NO-OP (pre-#1068 behavior verbatim).

    Documented residuals (deliberate, all fail toward PASS/WARN):
    (a) FAIL scope — only strict-prefix truncations >=20 chars covering
    >=50% of ``origin_prompt`` hard-FAIL; non-prefix truncations
    (mid-string deletions, truncate-and-fabricate), sub-floor cuts, and
    non-extractable quote shapes degrade to WARN — clean-result-critic
    Lens 5 keeps the substantive read on Context-row WARNs. (b)
    Trailing-punct asymmetry — a quote missing ``origin_prompt``'s final
    period FAILs (a 1-char truncation) while an ADDED period PASSes via
    containment — verbatim-consistent, not a verifier bug. (c) An
    ``origin_prompt`` containing line-leading ``>`` characters yields a
    conforming-body WARN (the marker strip applies to the body side
    only — the safe direction).

    A missing row FAILs only when recorded origin data exists —
    frontmatter ``origin_prompt`` or a ``## Provenance`` section in the
    sibling ``original-body.md`` — i.e. the body DROPPED provenance it
    had. With no recorded origin data the miss is a WARN: created_at +
    parent lineage always exist, so the row should still ship, stating
    the prompt was not recorded. Spec:
    `.claude/skills/clean-results/SPEC.md` § `**Context:**` row.
    """
    name = "Reproducibility Context provenance row"
    if not is_nested_design(body):
        return CheckResult(name, True, "skipped — legacy (pre-v2) body")
    repro = _repro_section_text(body)  # v4 footer or `## Reproducibility` H2
    if repro is None:
        # Missing-section is check_required_sections' job; don't double-FAIL.
        return CheckResult(name, True, "skipped — no Reproducibility section")
    if _CONTEXT_LABEL_RE.search(repro):
        if not is_v4(body):
            # v2/v3 keep the pre-#1014 label-presence behavior verbatim
            # (forward-only; the v4 lineage sub-check never binds them).
            # The #1068 origin-prompt sub-check is WARN-ONLY here
            # (grandfathering: NEVER a new hard FAIL below the v4
            # sentinel).
            status, sub_detail = _origin_prompt_quote_verdict(repro, fm)
            if status in ("fail-trunc", "warn-mismatch"):
                return CheckResult(
                    name, True, "**Context:** row present; " + sub_detail, is_warn=True
                )
            return CheckResult(name, True, "**Context:** row present")
        # v4 lineage-token sub-check. Strip fences + blockquote LINES
        # FIRST, then slice at the label: a single-line row with an inline
        # `> "..."` quote after the label (#763's shape) keeps its
        # same-line lineage clause, while multi-line blockquoted verbatim
        # prompts can never satisfy the scan (#959 strip precedent).
        # Two ACCEPTED false-PASS limitations, both benign-direction
        # (they reduce to today's unconditional label-presence PASS;
        # Lens 5 retains the substantive lineage-correctness read):
        # (a) an inline `> "..."` quote ON the label line is
        # markdown-semantically not a blockquote, so an issue ref inside
        # it stays scanned; (b) a lazy-continuation quote line (a wrapped
        # verbatim prompt whose un-prefixed continuation line stays
        # scanned — the deliberate #959 behavior documented at
        # `_strip_blockquote_lines`).
        scan_src = _strip_blockquote_lines(_strip_fenced_blocks(repro))
        m = re.search(r"\*\*\s*Context\s*:?\s*\*\*", scan_src)
        # Degenerate fallback (label only findable inside a blockquote/
        # fence): scan the whole stripped footer — fails toward PASS, the
        # same shape that PASSes unconditionally today.
        ctx_scan = scan_src[m.end() :] if m else scan_src
        if _V4_CONTEXT_LINEAGE_TOKEN_RE.search(ctx_scan):
            # #1068 origin-prompt verbatim sub-check — runs AFTER the
            # lineage sub-check (one failure at a time, the file's
            # convention; a body failing both surfaces the truncation on
            # the next verifier run after the lineage fix).
            status, sub_detail = _origin_prompt_quote_verdict(repro, fm)
            if status == "fail-trunc":
                return CheckResult(name, False, sub_detail)
            if status == "warn-mismatch":
                return CheckResult(
                    name,
                    True,
                    "**Context:** row present with lineage token; " + sub_detail,
                    is_warn=True,
                )
            return CheckResult(name, True, "**Context:** row present with lineage token")
        return CheckResult(
            name,
            False,
            "v4 `**Context:**` row carries no lineage token — add the lineage "
            "clause per SPEC.md § `**Context:**` row: `[#K](...) — <one line>` "
            "(bare `#K` accepted), or `fresh direction (no parent)` / "
            "`fresh (no parent)`; same-issue follow-up rounds add a "
            "'same-issue follow-up round `<label>`' clause. Issue refs inside "
            "the blockquoted verbatim prompt do not count — put the lineage "
            "clause on a non-blockquote line.",
        )
    has_origin_prompt = bool(str(fm.get("origin_prompt") or "").strip())
    has_provenance_section = False
    if original_body_path is not None and original_body_path.exists():
        has_provenance_section = bool(
            re.search(
                r"^##\s+Provenance\s*$",
                original_body_path.read_text(errors="replace"),
                re.MULTILINE,
            )
        )
    if has_origin_prompt or has_provenance_section:
        source = (
            "frontmatter `origin_prompt`"
            if has_origin_prompt
            else "`## Provenance` section in original-body.md"
        )
        return CheckResult(
            name,
            False,
            f"recorded origin data exists ({source}) but `## Reproducibility` has no "
            f"`**Context:**` row — carry the created/run dates, follow-up lineage, and "
            f"the verbatim originating prompt forward (SPEC.md § `**Context:**` row)",
        )
    return CheckResult(
        name,
        True,
        "missing `**Context:**` row (no recorded origin data — add the row with "
        "created/run dates + lineage and the literal `origin prompt not recorded`)",
        is_warn=True,
    )


# v4 `## Methodology` bold-label slots, in order. Used to extract the
# `Sample training/evaluation data + completions` slot text (the v4 raw
# sample-block home) bounded by the next slot label.
_V4_METHOD_SLOT_LABELS = [
    "Design",
    "Training",
    "Evaluation",
    "Data extraction",
    "Sample training/evaluation data + completions",
]


def _v4_methodology_sample_slot(methodology: str) -> str | None:
    """Return the text of the `**Sample training/evaluation data +
    completions:**` slot inside the v4 `## Methodology` section, bounded
    by the NEXT bold-label slot (or end of section). None when the slot
    is absent. Fence-aware on the boundary scan would be overkill — the
    bold labels live at the prose layer; a label inside a fenced block is
    not a real slot boundary, but the Sample slot is the LAST slot, so the
    only risk is over-inclusion of trailing prose, which is harmless for
    the sample-block scan."""
    m = re.search(
        r"(?im)^\s*[-*]?\s*\*\*\s*Sample training/evaluation data \+ completions\s*:?\s*\*\*",
        methodology,
    )
    if m is None:
        return None
    after = methodology[m.start() :]
    # Find the next bold-label slot that is NOT the Sample slot itself.
    nxt = re.search(
        r"(?im)^\s*[-*]?\s*\*\*\s*(?:Design|Training|Evaluation|Data extraction)\s*:?\s*\*\*",
        after[len(m.group(0)) :],
    )
    if nxt is not None:
        return after[: len(m.group(0)) + nxt.start()].strip()
    return after.strip()


def _sample_scan_sections(body: str, *, raw_link_scope: bool = False) -> list[tuple[str, str]]:
    """Return [(section_text, label)] to scan for sample-output blocks.

    v2 / legacy: the single `## TL;DR` section. v3: `## Findings` plus
    `## Data` (the per-finding excerpts live under Findings; the
    systematic samples live under Data). v4: `## Results` plus
    `## Methodology` (per-result excerpts under Results; the systematic
    samples live in the `## Methodology` Sample slot). When
    ``raw_link_scope`` is set (check 11 — the raw-completions-link rule),
    the v3 `## Data` scope narrows to `### Generated` ONLY, and the v4
    `## Methodology` scope narrows to the `Sample training/evaluation data
    + completions` slot ONLY (the other Methodology prose links JSONLs /
    probe banks, not raw_completions).

    Sections that are absent are silently skipped; the caller decides
    whether the absence is a FAIL (it is not for these checks — the
    structure / completeness checks own section presence).
    """
    out: list[tuple[str, str]] = []
    if is_v4(body):
        results = section_text(body, "Results")
        if results is not None:
            out.append((results, "## Results"))
        methodology = section_text(body, "Methodology")
        if methodology is not None:
            if raw_link_scope:
                sample = _v4_methodology_sample_slot(methodology)
                if sample is not None:
                    out.append((sample, "## Methodology → Sample data + completions"))
            else:
                out.append((methodology, "## Methodology"))
    elif is_v3(body):
        findings = section_text(body, "Findings")
        if findings is not None:
            out.append((findings, "## Findings"))
        data = section_text(body, "Data")
        if data is not None:
            if raw_link_scope:
                generated = _h3_subsection_text(data, "Generated")
                if generated is not None:
                    out.append((generated, "## Data → ### Generated"))
            else:
                out.append((data, "## Data"))
    else:
        tldr = section_text(body, "TL;DR")
        if tldr is not None:
            out.append((tldr, "## TL;DR"))
    return out


def check_cherry_picked_label(body: str) -> CheckResult:
    """Check 10: every sample-output block is preceded by a
    cherry-picked / random-sample disclosure in the prelude prose.

    v2 / legacy (2026-W22 spec, task #454): scans `## TL;DR`. v3: scans
    `## Findings` + `## Data`. In both cases the scan covers BOTH fenced
    code blocks AND `<details>` blocks that carry GFM tables / long text
    (nested-design bodies frequently present training rows + eval probes
    as `<details open>` tables instead of fenced code blocks — e.g. task
    #432). For each sample block the prose immediately above must carry
    the disclosure.
    """
    label = "Cherry-picked label discipline"
    scan_sections = _sample_scan_sections(body)
    if not scan_sections:
        if is_v4(body):
            missing = "## Results / ## Methodology"
        elif is_v3(body):
            missing = "## Findings / ## Data"
        else:
            missing = "## TL;DR"
        return CheckResult(label, False, f"{missing} section missing")
    flagged: list[str] = []
    total = 0
    for scan_text, _src in scan_sections:
        samples = _iter_sample_blocks(scan_text)
        total += len(samples)
        for start, _, content in samples:
            prelude = _prelude_window(scan_text, start)
            # Skip an exhaustive eval-INPUT enumeration (see the matching
            # skip in `check_qualitative_data_link`): a fixed eval-question
            # list ("The 20 eval input questions are the same fixed set …")
            # is the input stimulus, not a cherry-picked model-OUTPUT
            # sample, so the cherry-picked-disclosure rule does not apply.
            if _is_eval_input_enumeration_prelude(prelude):
                continue
            # For `<details>` blocks the cherry-pick disclosure may live
            # inside the block (the `<summary>` text or the prose around
            # the inner table); we scan BOTH the prelude window AND the
            # inner content. (For fenced code blocks the `content` is the
            # code text — a cherry-pick disclosure there is unusual and
            # harmless to scan; the prelude scan still dominates.)
            if _CHERRY_DISCLOSURE_RE.search(prelude) or _CHERRY_DISCLOSURE_RE.search(content):
                continue
            first_line = content.strip().splitlines()[0][:60] if content.strip() else "(empty)"
            flagged.append(first_line)
    if total == 0:
        srcs = ", ".join(f"`{s}`" for _t, s in scan_sections)
        return CheckResult(
            label, True, f"no sample-output blocks in {srcs} (fenced or `<details>`)"
        )
    if flagged:
        preview = "; ".join(f"'{x}'" for x in flagged[:2]) + (" …" if len(flagged) > 2 else "")
        return CheckResult(
            label,
            False,
            f"{len(flagged)} of {total} sample block(s) lack a cherry-picked / "
            f"random-sample disclosure in the prelude prose: {preview}",
        )
    return CheckResult(label, True, f"{total} sample block(s) labelled")


def check_qualitative_data_link(body: str) -> CheckResult:
    """Check 11: every sample-output block is preceded by at least one
    link or backtick-path that is NOT an aggregate-only path.

    An explicit `not uploaded` escape downgrades FAIL to WARN. v2 /
    legacy (2026-W22 spec, task #454): scans `## TL;DR`. v3: scans
    `## Findings` + `## Data → ### Generated` ONLY — the Trained-on /
    Evaluated-with blocks link JSONLs / probe banks (covered by check
    18), not raw_completions, so applying the raw-text-artifact rule to
    them would mis-FAIL legitimate Data capsules. The check scans BOTH
    fenced code blocks AND `<details>` blocks that carry GFM tables /
    long text (nested-design bodies frequently present rows + probes as
    `<details open>` tables instead of fenced code blocks — e.g. task
    #432).
    """
    label = "Qualitative-data link"
    scan_sections = _sample_scan_sections(body, raw_link_scope=True)
    if not scan_sections:
        if is_v4(body):
            missing = "## Results / ## Methodology → Sample data + completions"
        elif is_v3(body):
            missing = "## Findings / ## Data → ### Generated"
        else:
            missing = "## TL;DR"
        return CheckResult(label, False, f"{missing} section missing")
    fails: list[str] = []
    warns: list[str] = []
    passes = 0
    total = 0
    for scan_text, _src in scan_sections:
        samples = _iter_sample_blocks(scan_text)
        total += len(samples)
        for start, _, content in samples:
            prelude = _prelude_window(scan_text, start)
            # Skip an exhaustive eval-INPUT enumeration introduced by a
            # prelude like #538's "The 20 eval input questions are the same
            # fixed set …": that fenced block IS the input stimulus, not a
            # model-OUTPUT sample, so there is no raw-completion artifact to
            # link. Mirrors the `<details>` `<summary>` exhaustive-summary
            # skip (`_iter_sample_details`) for the fenced form. Requires the
            # "The N …" lead AND an eval-input framing token on the SAME
            # prelude line, so a cherry-picked output block ("The 5 most
            # extreme completions …") stays enforced.
            if _is_eval_input_enumeration_prelude(prelude):
                continue
            # For `<details>` blocks the raw-data link often lives INSIDE
            # the block, after the table (e.g. task #432's "Full training
            # file: [...]" link on the line after the table). Scan both
            # the prelude window AND the inner content of the block so the
            # check fires consistently regardless of where the body author
            # placed the qualitative-data link. (For fenced code blocks
            # the `content` is the code text; markdown links inside it are
            # unusual but harmless to scan — the prelude scan still
            # dominates.)
            search_space = prelude + "\n" + content
            # Collect candidate tokens: markdown link URLs + backtick-wrapped paths.
            tokens: list[str] = []
            tokens.extend(_LINK_RE.findall(search_space))
            tokens.extend(_CODE_RE.findall(search_space))
            has_escape = bool(_NOT_UPLOADED_RE.search(search_space))
            first_line = content.strip().splitlines()[0][:60] if content.strip() else "(empty)"

            if not tokens:
                if has_escape:
                    warns.append(f"'{first_line}': no link, `not uploaded` escape acknowledged")
                else:
                    fails.append(f"'{first_line}': no link or path in prelude paragraph")
                continue

            qualitative_hit = any(not _AGGREGATE_PATH_RE.search(tok) for tok in tokens)
            if qualitative_hit:
                passes += 1
                continue

            if has_escape:
                warns.append(
                    f"'{first_line}': only aggregate-pattern links, "
                    "`not uploaded` escape acknowledged"
                )
            else:
                fails.append(
                    f"'{first_line}': only aggregate-pattern links "
                    f"(e.g. {tokens[0][:60]}); raw text-level artifact required"
                )

    if total == 0:
        srcs = ", ".join(f"`{s}`" for _t, s in scan_sections)
        return CheckResult(
            label, True, f"no sample-output blocks in {srcs} (fenced or `<details>`)"
        )
    if fails:
        return CheckResult(
            label,
            False,
            f"{len(fails)} sample block(s) lack a qualitative-data link: "
            + "; ".join(fails[:2])
            + (" …" if len(fails) > 2 else ""),
        )
    if warns:
        return CheckResult(
            label,
            True,
            f"{len(warns)} sample block(s) ship with `not uploaded` escape — "
            "follow-up should re-run with raw-completion upload",
            is_warn=True,
        )
    return CheckResult(label, True, f"{passes} sample block(s) link to a qualitative-data artifact")


def check_goal_present(body: str, fm: dict) -> CheckResult:
    """Soft INFO check — Goal-of-experiment frontmatter field.

    Reports presence / absence of the canonical agent-facing Goal:
    frontmatter ``goal: <non-empty string>``. The body-side ``## Goal``
    H2 is intentionally NOT checked here — clean-result bodies drop the
    visible H2 and fold the Goal text into the TL;DR Motivation bullet
    (decision: 2026-05-26). The visible H2 lives only in proposed /
    planning bodies, where /issue Step 0c (workflow.yaml §
    gates.experiment_goal) is the enforcement point.

    The frontmatter ``goal:`` field stays in clean-result bodies so
    downstream agents (planner, critic, follow-up-proposer) have the
    agent-facing canonical Goal as context.

    This check NEVER FAILs. Clean-result bodies for non-experiment kinds,
    follow-ups, and pre-Goal bodies legitimately omit the field; failing
    them here would block promotion needlessly. The check is exposed for
    orchestrator visibility and tagged WARN when missing so the
    orchestrator can pick it up without halting.

    NOTE: ``body`` is accepted but no longer inspected. Kept in the
    signature so the call site in ``verify_text`` stays uniform with
    the body-only checks in ``CHECKS``.
    """
    del body  # body-side `## Goal` H2 intentionally not checked
    fm_goal = fm.get("goal")
    fm_goal = fm_goal.strip() if isinstance(fm_goal, str) and fm_goal.strip() else None
    if fm_goal:
        return CheckResult(
            "Goal-of-experiment field",
            True,
            f"frontmatter goal present ({len(fm_goal)} chars)",
        )
    return CheckResult(
        "Goal-of-experiment field",
        True,
        "missing: frontmatter `goal:` field (soft — enforced at /issue Step 0c, not here)",
        is_warn=True,
    )


def check_figure_h2_is_deprecated(body: str) -> CheckResult:
    """Check 12: reserved hook for `## Figure` H2 deprecation nudges.

    Under the 2-content-section spec (2026-W22, task #454) a stray
    `## Figure` H2 is rejected by `check_required_sections` (check 2)
    as a hard FAIL — clean migration is required, not nudged. This
    function is dormant in the current revision and always PASSes;
    it stays in `CHECKS` so the slot is available if a future
    WARN-only nudge needs it without shifting indices.
    """
    del body
    return CheckResult(
        "`## Figure` H2 deprecation hook (dormant)",
        True,
        "stray `## Figure` H2 is rejected by check 2; this hook is dormant "
        "under the 2-content-section spec",
    )


_DENOMINATOR_NOUNS = (
    r"factor[s]?(?:\s+flip[s]?)?|cell[s]?|condition[s]?|axis|axes|knob[s]?"
    r"|domain[s]?|seed[s]?|source[s]?|sweep[s]?|fold[s]?"
)

# `(\d+) of (\d+) <noun>` — captures the numerator + denominator + noun.
# Also accepts `(≥|<=|≥|at least) (\d+) of (\d+) <noun>` (`>=` written `≥`)
# and the "all N <noun>" / "N <noun>" forms (the latter only when paired
# with the keywords below that suggest a denominator claim).
_DENOMINATOR_CLAIM_RE = re.compile(
    rf"(?P<full>(?:at\s+least\s+|≥\s*|>=\s*)?(?P<num>\d+)\s+of\s+(?P<den>\d+)\s+"
    rf"(?:swept\s+|planned\s+|matched\s+|testable\s+|tested\s+)?"
    rf"(?P<noun>{_DENOMINATOR_NOUNS}))",
    re.IGNORECASE,
)


def _collect_denominator_claims(text: str) -> list[tuple[int, int, str, str]]:
    """Return list of (numerator, denominator, noun, full_match_text)
    for every `X of Y <noun>` claim in `text`."""
    out: list[tuple[int, int, str, str]] = []
    for m in _DENOMINATOR_CLAIM_RE.finditer(text):
        try:
            num = int(m.group("num"))
            den = int(m.group("den"))
        except (TypeError, ValueError):
            continue
        if den < 1 or num < 0:
            continue
        # Reject "N of M" where both sides look like populations rather than
        # denominator claims — e.g. "1 of 24 panel personas" is reporting a
        # rate, not a planned-vs-actual count. Heuristic: only track when the
        # noun is in `_DENOMINATOR_NOUNS` (already guaranteed by the regex)
        # AND the denominator is small (≤ 50; planned-vs-actual rarely runs
        # higher and rate-style usages routinely hit hundreds).
        if den > 50:
            continue
        out.append((num, den, m.group("noun").lower(), m.group("full")))
    return out


def check_planned_vs_actual_denominator(body: str) -> CheckResult:
    """Check: planned-vs-actual coverage denominator consistency.

    Catches the scope-shrinkage-without-explicit-flag anti-pattern (task
    #391, 2026-05-27): the plan committed to N conditions, M < N delivered,
    in-body prose acknowledges the drop ("only M of N delivered"), but
    the headline TL;DR / Hypothesis denominator still uses the original
    N. Reader walks away thinking the experiment tested N conditions
    when only M delivered.

    Mechanical scope: WITHIN the body only. The check compares
    denominator claims in TL;DR (the headline surface) against any
    "M of N" scope claim found elsewhere in the body (typically inside
    a result H3 that names a methodology correction, or in legacy
    bodies inside a `### Methodology corrections` H3). When the body's
    correction prose names "M of N testable" or "delivered M of N", the
    TL;DR's `X of N` denominator becomes inconsistent — readers see two
    different N values.

    Under the 2-content-section spec (2026-W22, task #454) the
    `### Methodology corrections` H3 is no longer required as a
    discrete section; scope-shrinkage prose can live in any result H3.
    The check therefore scans the body OUTSIDE `## TL;DR` (typically
    `## Reproducibility` and any retired-section content the body still
    carries) for denominator claims. The TL;DR claims come from
    `## TL;DR` itself.

    Plan-side enumeration (does the plan actually commit to a larger N?)
    is the semantic call clean-result-critic Lens 13 makes; this
    mechanical check does NOT read the plan file. The within-body
    consistency check is what the verifier can robustly enforce.

    FAIL trigger: the body's non-TL;DR text contains a `X of Y <noun>`
    claim with X < Y AND the body's `## TL;DR` contains a `K of N <noun>`
    claim where N == Y AND the noun matches AND K does not also indicate
    the reduced scope. PASSes silently when no non-TL;DR scope claim
    exists OR when no TL;DR denominator claims appear.

    See `.claude/agents/clean-result-critic.md` § Lens 13 for the
    semantic-judgment version of this check (which reads the plan).
    """
    # The headline surface that must carry an accurate denominator is
    # `## TL;DR` for v2/legacy bodies; for v3 bodies it is `## Takeaways`
    # + `## Findings`; for v4 bodies it is `## Takeaways` + `## Results`
    # (no `## TL;DR` umbrella). The scope-correction scan stays whole-body
    # in all cases.
    if is_v3(body) or is_v4(body):
        result_section = "Results" if is_v4(body) else "Findings"
        # v4 uses the footer-truncated Results body so a footer-bullet
        # disclosure (`5 of 2,000 rows`) is not read as a headline denominator.
        result_text = _v4_results_body(body) if is_v4(body) else section_text(body, result_section)
        headline_parts = [t for t in (section_text(body, "Takeaways"), result_text) if t]
        if not headline_parts:
            return CheckResult(
                "planned-vs-actual denominator consistency",
                True,
                f"## Takeaways / ## {result_section} missing — other checks will report",
            )
        headline_text = "\n\n".join(headline_parts)
    else:
        tldr = section_text(body, "TL;DR")
        if tldr is None:
            # Other checks will FAIL on missing sections; don't double-report.
            return CheckResult(
                "planned-vs-actual denominator consistency",
                True,
                "## TL;DR missing — other checks will report",
            )
        headline_text = tldr
    # The "scope-correction" text can live anywhere — including inside a
    # `### <finding>` H3, in a result H3 outside the headline surface, or
    # in legacy in-flight bodies under a retired `### Methodology
    # corrections` H3. So scan the WHOLE body for scope-correction
    # claims; the headline claims come from the headline surface only.
    scope_text = body

    tldr_claims = _collect_denominator_claims(headline_text)
    method_claims = _collect_denominator_claims(scope_text)

    if not method_claims or not tldr_claims:
        return CheckResult(
            "planned-vs-actual denominator consistency",
            True,
            f"TL;DR claims={len(tldr_claims)}, "
            f"whole-body scope-correction claims={len(method_claims)} — "
            "insufficient signal for a denominator drift check",
        )

    # For each (noun) pair where the whole-body scan finds a
    # `M of N <noun>` (with M < N — a scope reduction) AND TL;DR names
    # a `K of N <noun>` with the SAME N, the TL;DR denominator is stale
    # relative to the documented scope reduction. The scan is whole-body
    # (not "outside TL;DR") because under the 2-content-section spec
    # scope-correction prose lives inside `### <finding>` H3s INSIDE
    # `## TL;DR`; the previous outside-only scan silently lost those
    # cases. We dedupe so the same physical claim seen in both lists
    # doesn't conflict with itself.
    seen_pairs: set[tuple[int, int, str, str, int, int, str, str]] = set()
    conflicts: list[str] = []
    for m_num, m_den, m_noun, m_full in method_claims:
        # The whole-body "of N" can be the ORIGINAL plan denominator
        # (e.g., "2 of 3 testable"); the numerator is the delivered count.
        # The TL;DR should NOT reuse N as its denominator — it should use
        # m_num (the delivered count) or report against the reduced scope.
        m_stem = m_noun.rstrip("s")
        for t_num, t_den, t_noun, t_full in tldr_claims:
            t_stem = t_noun.rstrip("s")
            if m_stem != t_stem:
                continue
            # Skip the same physical claim appearing in both lists (whole-
            # body scan + TL;DR scan will see TL;DR-resident claims twice).
            if (m_num, m_den, m_noun, m_full) == (t_num, t_den, t_noun, t_full):
                continue
            # Dedupe symmetric pairs (m,t) and (t,m) so a single TL;DR-
            # internal mismatch produces one FAIL message, not two.
            key = (m_num, m_den, m_noun, m_full, t_num, t_den, t_noun, t_full)
            key_swapped = (t_num, t_den, t_noun, t_full, m_num, m_den, m_noun, m_full)
            if key in seen_pairs or key_swapped in seen_pairs:
                continue
            seen_pairs.add(key)
            if t_den == m_den and m_num < m_den:
                # TL;DR is still framing against the ORIGINAL denominator
                # even though the body acknowledges only m_num delivered.
                # This is the inconsistency.
                conflicts.append(
                    f"TL;DR says {t_full!r} but body elsewhere says {m_full!r} "
                    f"(only {m_num} of {m_den} {m_noun} delivered) — "
                    f"revise the TL;DR denominator to {m_num} to match actual coverage"
                )

    if conflicts:
        # Cap surfaced conflicts to first 3 to keep the FAIL message readable.
        return CheckResult(
            "planned-vs-actual denominator consistency",
            False,
            "; ".join(conflicts[:3])
            + (f" (+{len(conflicts) - 3} more)" if len(conflicts) > 3 else ""),
        )
    return CheckResult(
        "planned-vs-actual denominator consistency",
        True,
        f"{len(tldr_claims)} TL;DR denominator claim(s) consistent with "
        f"{len(method_claims)} whole-body scope-correction claim(s)",
    )


def check_details_narrative_flow(body: str) -> CheckResult:
    """Soft WARN check — TL;DR narrative-shape heuristics (story arc).

    Two conservative mechanical signals; never FAILs. Critic-side LM
    judgment (clean-result-critic) catches the semantic cases this
    regex check misses.

    Under the 2-content-section spec (2026-W22, task #454) the
    LessWrong-style narrative lives inside `## TL;DR` (the
    `### Motivation` H3 followed by one `### <finding>` H3 per result).
    This check therefore scans `## TL;DR` for the two regressions:

    1. **Bad H3 labels in ``## TL;DR``.** Outline-label H3s
       (``### Headline result`` / ``### Subset checks`` /
       ``### Sample completions`` / ``### Plan deviations`` /
       ``### Methodology`` / ``### Findings``) name a genre of content
       instead of what the reader is about to learn. Story-beat H3s
       (``### A cohort disagreement on the primary``) pass.
    2. **Figure-dump.** Three or more consecutive ``![alt](url)`` image
       lines inside ``## TL;DR`` with no prose between — almost always
       a chart-paste, not a chart-embedded-in-a-story. Two adjacent
       images are allowed (the raw + processed pair).

    Both signals WARN; downstream agents (clean-result-critic,
    analyzer) should treat them as inputs to a narrative check rather
    than as a promote-blocking FAIL.
    """
    # v4 bodies carry the result narrative under `## Results`; v3 under
    # `## Findings`; v2 / legacy under `## TL;DR`. Same heuristics,
    # different host section.
    nav_section = _figure_scan_section(body)
    label_name = f"{nav_section} narrative flow"
    tldr = section_text(body, nav_section)
    if tldr is None:
        return CheckResult(
            label_name,
            True,
            f"no ## {nav_section} section to inspect (skipped)",
            is_warn=True,
        )

    findings: list[str] = []

    # Heuristic 1: outline-label H3s. NOTE: `### Findings` and
    # `### What I ran` are REQUIRED structural H3s under the
    # nested-design (v2) shape — they are explicitly excluded from this
    # WARN list. `### Background` / `### Setup` / `### Methodology` /
    # `### Headline result` / `### Subset checks` / `### Sample
    # completions` / `### Plan deviations` remain outline labels and
    # still warn (story-beat H3s name what the reader is about to
    # learn, not the genre of content).
    bad_label_re = re.compile(
        r"^###\s+(?P<name>Headline result|Subset checks|Sample completions|"
        r"Plan deviations|Methodology|Background|Setup)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    bad_h3_names = [m.group("name") for m in bad_label_re.finditer(tldr)]
    if bad_h3_names:
        findings.append(
            f"{len(bad_h3_names)} outline-label H3(s) in {nav_section}: "
            f"{', '.join(bad_h3_names)} — story-beat H3s name what the "
            "reader is about to learn, not the genre of content"
        )

    # Heuristic 2: figure-dump (>2 consecutive images without prose
    # between). Two adjacent images are allowed for raw + processed
    # pairs.
    img_line_re = re.compile(r"^\s*!\[(?:[^\]]|\](?!\())*\]\([^)]+\)\s*$")
    lines = tldr.splitlines()
    runs: list[int] = []
    run_len = 0
    for line in lines:
        if img_line_re.match(line):
            run_len += 1
            continue
        stripped = line.strip()
        if stripped == "":
            # Blank lines don't break the run — figures can be
            # separated by blank lines yet still count as a dump.
            continue
        if run_len >= 1:
            runs.append(run_len)
        run_len = 0
    if run_len >= 1:
        runs.append(run_len)
    dumps = [n for n in runs if n > 2]
    if dumps:
        findings.append(
            f"{len(dumps)} run(s) of >2 consecutive figures in {nav_section} "
            "with no prose between — likely figure-dump. "
            "Add setup + read paragraphs around each figure."
        )

    if findings:
        return CheckResult(
            label_name,
            True,
            "; ".join(findings),
            is_warn=True,
        )
    return CheckResult(
        label_name,
        True,
        "no mechanical narrative-shape regressions detected",
    )


# ─── Reproducibility "committed at commit `<sha>`" claim verification ─────


# Strip fenced code blocks from a chunk of markdown so the scan below
# never matches an example sha/path that lives inside a ``` ... ``` block.
# Mirrors the strip pattern used elsewhere in this file (see
# ``_strip_code_for_prose_scan`` for the more elaborate table-aware
# variant — here we only need the fence pass).
def _strip_fenced_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out.append(line)
    return "\n".join(out)


# URL scans over the Reproducibility/footer region ALSO ignore markdown
# blockquote lines (`>`-prefixed, incl. nested `> >` and indented `  > `
# quotes). In that region a blockquote is the SPEC-mandated verbatim
# originating-prompt quote (`**Context:**` row — never paraphrased),
# whose text may cite bare URLs the author is not allowed to edit
# (#825 → task #959). The quote is provenance TEXT, not a provenance
# LINK; pinned artifact links live on the non-quoted footer rows, which
# stay fully checked. Lazy continuation lines (quote text wrapped
# without a leading `>`) stay scanned — the failure mode is the pre-fix
# behavior (scanned), never a silently widened exemption. Apply AFTER
# `_strip_fenced_blocks` (a quoted fence marker `> ``` ` must not
# toggle fence state; it doesn't — the fence pass keys on lines that
# START with the fence marker).
def _strip_blockquote_lines(text: str) -> str:
    """Drop lines whose first non-whitespace character is `>`.

    Returns the remaining lines joined by newlines; used by checks 8 and
    8b so a verbatim-quoted URL is never treated as a provenance link.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


# A "committed ... at commit `<sha>`" claim. The trigger word ``committed``
# must appear somewhere before the literal phrase ``at commit `<sha>` `` on
# the SAME line. The sha must be a 4-40 char hex literal wrapped in
# backticks. This conservative anchoring avoids matching HF Hub or WandB
# URLs (whose hex paths are never preceded by the prose phrase "at commit
# `<sha>`") and prose-only sentences (which never carry a backticked sha).
_COMMITTED_AT_SHA_RE = re.compile(
    r"committed[^\n]*?at\s+commit\s+`(?P<sha>[0-9a-fA-F]{4,40})`",
    re.IGNORECASE,
)

# A repo-relative artifact path inside backticks: must end in `.json`,
# `.png`, `.csv` OR begin with `figures/` / `eval_results/`. Leading `./`
# is tolerated and stripped at use-time. Paths starting with `/`, `~`, or
# containing a scheme (`://`) are rejected (those are absolute or remote
# references, never repo-relative). The capture is intentionally narrow:
# the rule only fires when both a sha claim AND a clearly-named path
# co-occur on the same line.
_ARTIFACT_PATH_RE = re.compile(
    r"`(?P<path>(?:\./)?(?:figures/|eval_results/)[^\s`]+|"
    r"(?:\./)?[A-Za-z0-9_./-]+\.(?:json|png|csv))`"
)

# Clause delimiters for the committed-at-sha scan (#893, incident #841):
# semicolon / interpunct / sentence-final period, each REQUIRING trailing
# whitespace — so file-extension dots (`foo.json`), version strings, and
# URL punctuation (which carry no spaces) never split. The em-dash (" — ")
# and ":" are deliberately NOT delimiters: real footers put them BETWEEN a
# path and its own sha claim (the #549 / #601 corpus shapes), so splitting
# there would silently drop genuine pairs.
_CLAUSE_DELIM_RE = re.compile(r";\s|\s·\s|\.\s")

# Abbreviation tokens whose trailing dot is NOT a sentence boundary
# (#893 v2, methodology-reconciler binding concern 2): without this guard,
# "committed, e.g. `path` at commit `sha`." splits at "e.g. " and a TRUE
# FAIL today becomes a silent PASS (protection removal). Matched against
# the lowercased word immediately before the candidate ". " (the token's
# own internal dots included, e.g. "e.g"). Over-inclusion is fail-safe:
# a suppressed split degrades toward today's whole-line behavior.
_DOT_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "eg", "ie", "cf", "vs", "etc", "al", "fig", "approx", "incl"}
)


def _split_clauses(line: str) -> list[str]:
    """Split ``line`` into clauses on ``; `` / `` · `` / ``. `` occurring
    OUTSIDE backtick code spans, so a ``committed ... at commit `<sha>```
    match and its artifact-path pairing can never cross a clause boundary
    (#893; incident #841: the lazy span crossed from a results clause's
    "committed" to the figures clause's "at commit `<sha>`" and validated
    eval_results paths against the figures SHA). A ``. `` whose preceding
    word is a known abbreviation (``e.g.`` / ``cf.`` / ...) does not
    split. An UNBALANCED backtick latches ``in_code`` for the line's
    remainder — deliberately fail-safe: no further splits means the
    suffix keeps today's whole-line behavior, never a new false FAIL.
    Known residual (accepted): two claims INSIDE one clause with no
    delimiter between them still cross-pair within that clause.
    """
    clauses: list[str] = []
    buf: list[str] = []
    in_code = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
            i += 1
            continue
        if not in_code:
            m = _CLAUSE_DELIM_RE.match(line, i)
            if m is not None:
                # Abbreviation guard: applies only to the ". " delimiter —
                # look at the word ending the current buffer (dot excluded).
                if ch == ".":
                    words = "".join(buf).split()
                    last_word = words[-1] if words else ""
                    if last_word.lower().strip("(,") in _DOT_ABBREVIATIONS:
                        buf.append(ch)
                        i += 1
                        continue
                clauses.append("".join(buf))
                buf = []
                i = m.end()
                continue
        buf.append(ch)
        i += 1
    clauses.append("".join(buf))
    return clauses


def _resolve_repo_root() -> Path | None:
    """Return the repo root via the existing task_workflow helper, or
    None if the import fails (e.g. running this script outside the repo)."""
    try:
        from explore_persona_space.task_workflow import repo_root  # local import

        return repo_root()
    except Exception:
        return None


# ─── #732: judge-API-error denominator gate (check_judge_error_denominator) ──
#
# A judge-denominator CLAIM: an `n=N` (N>=100) or an "N completions /
# judgments / attempted / EM" token. Used to detect whether a clean-result
# body asserts a bare LLM-judge denominator.
_JUDGE_DENOMINATOR_RE = re.compile(
    r"\bn\s*=\s*(\d{3,})\b"
    r"|\b(\d{3,})\s+(?:completions|judgments|judgements|attempted|EM\b)",
    re.IGNORECASE,
)
# A judge-context guard: the denominator claim only counts when it co-occurs
# (within a small window) with a judge noun — so a bare training-row count
# ("6349 training rows") does NOT trigger the eval-JSON read.
_JUDGE_CONTEXT_RE = re.compile(
    r"claude-sonnet|\bjudge\b|EM rate|misalign|Batch API|sycophan|refusal|\btrait\b|\bfact\b",
    re.IGNORECASE,
)
# A disclosure phrase: present near the claim (or in the **Repro:** footer)
# ⇒ the body already discloses the judge-error fraction ⇒ PASS before any
# eval read. Broad on purpose (≥11 alternations); the published #715 body
# matches several.
_JUDGE_ERROR_DISCLOSURE_RE = re.compile(
    r"\b529\b|Overloaded|judge[-\s]?error|judge[-\s]?API[-\s]?error"
    r"|n_judge_error|n_em_judge_error|excluded from the denominator"
    r"|excluded from the per-cell denominator"
    r"|post-correction|judge-error-corrected|API[-\s]?error|judge_failed"
    # U+2212 MINUS SIGN below is intentional: it matches the unicode minus the
    # #715 body uses in its "400 (U+2212) n_judge_error" disclosure phrasing.
    r"|[-−]\s*n_judge_error",  # noqa: RUF001
    re.IGNORECASE,
)
# Recognized judge-API-error count fields in committed eval JSONs.
# NOTE: `n_parse_error` is the DISTINCT judge-side parse-failure class, NOT
# the 529 API-error class — deliberately excluded here (§11).
_JUDGE_ERROR_COUNT_KEYS = ("n_judge_error", "n_em_judge_error", "n_api_error")
# Recognized per-cell ATTEMPTED-count fields (the denominator). Two source
# classes use different names: corrected-pareto leaf cells carry
# `n_em_attempted`; per-cell em_rate aggregates carry `n_total`. The scan
# tries each in order until one resolves on a given cell dict.
_JUDGE_ATTEMPTED_COUNT_KEYS = ("n_em_attempted", "n_attempted", "n_total")


def _judge_denominator_claims(scan_region: str) -> list[re.Match]:
    """Return the judge-denominator claim matches in `scan_region` that
    co-occur (within ±120 chars) with a judge noun. The window guard keeps
    the check from firing on a bare training-row count."""
    out: list[re.Match] = []
    for m in _JUDGE_DENOMINATOR_RE.finditer(scan_region):
        lo = max(0, m.start() - 120)
        hi = min(len(scan_region), m.end() + 120)
        if _JUDGE_CONTEXT_RE.search(scan_region[lo:hi]):
            out.append(m)
    return out


def _eval_root_from_body_path(body_source_path: Path, eval_subpath: Path) -> Path | None:
    """Leg (ii) of the eval-root ladder: walk the body source path's
    ancestors; return the nearest ancestor that is a repo root (a `.git`
    entry — file OR dir; a worktree's `.git` is a FILE) or a
    `.claude/worktrees/issue-<M>` worktree dir, AND contains `eval_subpath`.
    None when none match."""
    try:
        ancestors = list(body_source_path.resolve().parents)
    except OSError:
        return None
    for p in ancestors:
        try:
            is_repo_root = (p / ".git").exists()
        except OSError:
            is_repo_root = False
        if not is_repo_root:
            is_repo_root = (
                re.match(r"issue-\d+", p.name) is not None and p.parent.name == "worktrees"
            )
        if is_repo_root:
            try:
                if (p / eval_subpath).is_dir():
                    return p
            except OSError:
                continue
    return None


def _git_toplevel_eval_root(eval_subpath: Path) -> Path | None:
    """Leg (iii) of the eval-root ladder: `git rev-parse --show-toplevel`
    from cwd, returned only when it contains `eval_subpath`. Conservative —
    any nonzero exit / OSError → None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    if not top:
        return None
    root = Path(top)
    try:
        return root if (root / eval_subpath).is_dir() else None
    except OSError:
        return None


def _resolve_eval_root(
    issue: int,
    *,
    eval_root: Path | None = None,
    body_source_path: Path | None = None,
) -> Path | None:
    """Resolve the ROOT directory under which `eval_results/issue_<N>/`
    lives, walking the §4.2a four-leg ladder and STOPPING at the first leg
    that yields a directory containing `eval_results/issue_<N>/`:

      (i)   `eval_root` (explicit `--eval-root`, gate-time worktree path),
      (ii)  the `--file`-derived worktree root (nearest `.git` ancestor of
            the body source path, or a `.claude/worktrees/issue-<M>` segment),
      (iii) `git rev-parse --show-toplevel` from cwd,
      (iv)  `_resolve_repo_root()` (MAIN — bottom-of-ladder, post-merge bind).

    Returns the ROOT (the directory CONTAINING `eval_results/`), not the eval
    dir itself, so `_scan_issue_judge_errors(root, issue)` keeps its
    `root / "eval_results" / f"issue_{issue}"` join. Returns None when no leg
    resolves — the check then graceful-PASSes (never a false FAIL on missing
    data). The MAIN-only resolution of v2 is now leg (iv), not the only path.
    """
    eval_subpath = Path("eval_results") / f"issue_{issue}"

    # Leg (i): explicit --eval-root.
    if eval_root is not None:
        try:
            if (eval_root / eval_subpath).is_dir():
                return eval_root
        except OSError:
            pass

    # Leg (ii): --file-derived worktree root.
    if body_source_path is not None:
        hit = _eval_root_from_body_path(body_source_path, eval_subpath)
        if hit is not None:
            return hit

    # Leg (iii): cwd `git rev-parse --show-toplevel` (conservative).
    hit = _git_toplevel_eval_root(eval_subpath)
    if hit is not None:
        return hit

    # Leg (iv): MAIN repo root (the v2 behavior, now the tail fallback).
    main_root = _resolve_repo_root()
    if main_root is not None:
        try:
            if (main_root / eval_subpath).is_dir():
                return main_root
        except OSError:
            return None
    return None


def _first_count(cell: dict, keys: tuple[str, ...]) -> int | None:
    """Return the first numeric value found under `keys` in `cell`, or None.
    Non-numeric / bool values are ignored (a bool is not a valid count)."""
    for k in keys:
        v = cell.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _scan_issue_judge_errors(repo: Path, issue: int) -> dict | None:
    """Scan the committed `repo/eval_results/issue_<N>/` JSON tree for a
    judge-API-error signal. Returns
    ``{"total_err", "total_att", "worst_frac", "n_cells", "source"}`` or
    None when no recognized judge-error signal is found (graceful skip).

    A "cell" is any dict in the JSON tree that carries BOTH a recognized
    judge-error count key (`_JUDGE_ERROR_COUNT_KEYS`) AND a recognized
    attempted-count key (`_JUDGE_ATTEMPTED_COUNT_KEYS`) — found by recursive
    descent, so the nested
    `pareto_*_corrected.json` shape (`cells: {cond: {seed: [step_dicts]}}`)
    and the flat per-cell `em_rate/*.json` aggregate convention both resolve.
    `n_parse_error` is NOT a recognized judge-error key (it is the distinct
    parse-failure class), so an eval dir whose only count field is
    `n_parse_error` returns None (graceful PASS).

    The `EPM_VERIFY_BODY_NO_EVAL_SCAN=1` env fence disables the disk read
    (returns None) so the suite + offline runs are deterministic.
    """
    if os.environ.get("EPM_VERIFY_BODY_NO_EVAL_SCAN") == "1":
        return None
    eval_dir = repo / "eval_results" / f"issue_{issue}"
    if not eval_dir.is_dir():
        return None

    leaves: list[tuple[int, int]] = []  # (judge_error, attempted) per cell

    def _descend(node: object) -> None:
        if isinstance(node, dict):
            err = _first_count(node, _JUDGE_ERROR_COUNT_KEYS)
            att = _first_count(node, _JUDGE_ATTEMPTED_COUNT_KEYS)
            if err is not None and att is not None:
                leaves.append((err, att))
                return  # this dict IS a cell; do not double-count children
            for v in node.values():
                _descend(v)
        elif isinstance(node, list):
            for v in node:
                _descend(v)

    for path in sorted(eval_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue  # a corrupt / unreadable artifact must not crash the gate
        _descend(payload)

    if not leaves:
        return None

    total_err = sum(e for e, _a in leaves)
    total_att = sum(a for _e, a in leaves)
    worst_frac = max((e / a if a else 0.0) for e, a in leaves)
    return {
        "total_err": total_err,
        "total_att": total_att,
        "worst_frac": worst_frac,
        "n_cells": len(leaves),
        "source": "eval-json",
    }


def check_judge_error_denominator(
    body: str,
    *,
    issue: int | None = None,
    eval_root: Path | None = None,
    body_source_path: Path | None = None,
) -> CheckResult:
    """FAIL/WARN when a clean-result body asserts a BARE LLM-judge
    denominator (`n=N` / "N completions / EM") in a judge-context section
    WITHOUT disclosing the judge-API-error fraction, while the committed
    `eval_results/issue_<N>/` JSONs show a non-trivial fraction of rows
    returned Anthropic Batch API 529-overload errors that were silently
    counted into the denominator (the /issue 715 R1 trap).

    Verdict ladder (§4.3): PASS when (a) no bare judge denominator is
    asserted, (b) the body discloses, (c) the issue is unknown / eval root
    unresolvable / no recognized judge-error signal exists (graceful skip);
    WARN at >1% (worst-cell OR pooled); FAIL at >5%.
    """
    name = "judge-API-error denominator disclosed"
    # Generation gate: only structured (v3 / v4) bodies. Legacy / v2 PASS
    # vacuously (forward-grandfathering, matching every generation-gated check).
    if not (is_v3(body) or is_v4(body)):
        return CheckResult(name, True, "skipped — legacy/v2 body")

    method = section_text(body, "Methodology") or ""
    results_s = section_text(body, "Results") or ""
    footer = _v4_footer_text(body) or ""
    scan_region = _strip_fenced_blocks(method + "\n" + results_s)

    claims = _judge_denominator_claims(scan_region)
    if not claims:
        return CheckResult(name, True, "no judge denominator asserted")

    disclosed_region = scan_region + "\n" + footer
    if _JUDGE_ERROR_DISCLOSURE_RE.search(disclosed_region):
        return CheckResult(name, True, "judge-error fraction disclosed in body")

    if issue is None:
        return CheckResult(
            name, True, "skipped — issue number unknown (stdin); cannot read eval_results"
        )

    repo = _resolve_eval_root(issue, eval_root=eval_root, body_source_path=body_source_path)
    if repo is None:
        return CheckResult(name, True, "skipped — eval root unresolved", is_warn=True)

    stats = _scan_issue_judge_errors(repo, issue)
    if stats is None:
        return CheckResult(
            name, True, "no judge-error data available in committed eval JSONs — graceful skip"
        )

    pooled = stats["total_err"] / max(stats["total_att"], 1)
    worst = stats["worst_frac"]
    detail = (
        f"{stats['total_err']} rows (worst cell {worst:.1%}, pooled {pooled:.1%}) "
        f"returned judge-API errors and were silently counted into the n=<...> "
        f"denominator with no disclosure. Recompute as "
        f"n_misaligned/(n_attempted - n_judge_error) and disclose the excluded fraction."
    )
    if worst > 0.05 or pooled > 0.05:
        return CheckResult(name, False, detail)
    if worst > 0.01 or pooled > 0.01:
        return CheckResult(name, True, detail, is_warn=True)
    return CheckResult(
        name, True, f"judge-error fraction below 1% (worst {worst:.1%}, pooled {pooled:.1%})"
    )


# ─── Check 35 (#1256): cross-issue reuse pins declared in the body ─────────

# A metadata KEY that pins another issue's HF revision: `hf_rev_<M>` or
# `hf_rev_<M>_<tag>` (observed shape: #1092's transfer_reads.json
# metadata.args `hf_rev_779_passb` / `hf_rev_779_labels` / self-pin
# `hf_rev_1092`). Corpus base rate: exactly 1 file among ~90,858 committed
# eval JSONs (scan 2026-07-11) — the incident file.
_HF_REV_PIN_KEY_RE = re.compile(r"^hf_rev_(\d+)(?:_\w+)?$")

# An issue-slug path segment (`issue779_monitoring/...`) inside metadata
# input paths. Deliberately underscore-anchored: local caches
# (`data/issue_952/`), worktrees (`issue-952`) and self dirs
# (`eval_results/issue_1092/`) do NOT match (underscore/hyphen precedes
# the digits there, so `\bissue(\d+)_` cannot fire on them). The bare
# pattern appears in >=10,028 committed eval JSONs (scan 2026-07-11), so
# tier 2 is restricted to `input_shas` + path-like `args` values and
# capped at WARN severity.
_ISSUE_SLUG_RE = re.compile(r"\bissue(\d+)_[A-Za-z0-9_]*")

# Hex tokens in the body that can satisfy a tier-1 revision pin by prefix.
_BODY_HEX_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{7,40}\b")

# Per-file stat guard (MANDATORY): eval_results is JSON/text-only by policy,
# but `issue_810` carries 4 JSONs of 138-208 MB and `issue_811` is a
# ~14.7 GB dir — a guard-less gate-time scan there would read ~750 MB+.
_REUSE_SCAN_MAX_BYTES = 50 * 1024 * 1024


def _iter_metadata_dicts(payload: object) -> Iterator[dict]:
    """Yield every dict that sits under a key named `metadata`, any depth.

    Measured identical to top-level-only on the whole committed corpus
    (nested `metadata` dicts add 0 pin hits, scan 2026-07-11), but more
    robust for nested `run_result.json` phase shapes.
    """
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "metadata" and isinstance(v, dict):
                    yield v
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)


def _walk_hf_rev_keys(
    node: object, issue: int, relpath: str, tier1: list[tuple[str, str, int, str]]
) -> None:
    """Recursively collect tier-1 pins: `hf_rev_<M>[_<tag>]` keys with a
    string value and M != issue, anywhere inside `node`."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and isinstance(v, str):
                m = _HF_REV_PIN_KEY_RE.match(k)
                if m and int(m.group(1)) != issue:
                    tier1.append((relpath, k, int(m.group(1)), v))
            _walk_hf_rev_keys(v, issue, relpath, tier1)
    elif isinstance(node, list):
        for v in node:
            _walk_hf_rev_keys(v, issue, relpath, tier1)


def _slug_hits(text: str, issue: int, relpath: str, tier2: list[tuple[str, int, str]]) -> None:
    """Collect tier-2 `_ISSUE_SLUG_RE` tokens with M != issue from `text`."""
    for m in _ISSUE_SLUG_RE.finditer(text):
        src = int(m.group(1))
        if src != issue:
            tier2.append((relpath, src, m.group(0)))


def _walk_pathlike_args(
    node: object, issue: int, relpath: str, tier2: list[tuple[str, int, str]]
) -> None:
    """Recursively collect tier-2 tokens from PATH-LIKE (`"/" in v`) string
    values under a metadata `args` subtree."""
    if isinstance(node, dict):
        for v in node.values():
            _walk_pathlike_args(v, issue, relpath, tier2)
    elif isinstance(node, list):
        for v in node:
            _walk_pathlike_args(v, issue, relpath, tier2)
    elif isinstance(node, str) and "/" in node:
        _slug_hits(node, issue, relpath, tier2)


def _collect_metadata_pins(
    md: dict,
    issue: int,
    relpath: str,
    tier1: list[tuple[str, str, int, str]],
    tier2: list[tuple[str, int, str]],
) -> None:
    """Collect cross-issue pins from ONE metadata dict into tier1/tier2.

    tier1: `hf_rev_<M>[_<tag>]` keys (ALL keys, recursively) with a string
    value and M != issue. tier2: `_ISSUE_SLUG_RE` hits in
    `metadata["input_shas"]` keys + string values, and in PATH-LIKE
    (`"/" in v`) string values under `metadata["args"]` (recursive),
    M != issue. M == issue pins (the `hf_rev_1092` self-pin) never flag —
    the self-pin is provenance, not reuse.
    """
    _walk_hf_rev_keys(md, issue, relpath, tier1)

    input_shas = md.get("input_shas")
    if isinstance(input_shas, dict):
        for k, v in input_shas.items():
            if isinstance(k, str):
                _slug_hits(k, issue, relpath, tier2)
            if isinstance(v, str):
                _slug_hits(v, issue, relpath, tier2)

    args = md.get("args")
    if args is not None:
        _walk_pathlike_args(args, issue, relpath, tier2)


def _scan_cross_issue_reuse_pins(repo: Path, issue: int) -> dict | None:
    """Scan committed `repo/eval_results/issue_<N>/**/*.json` metadata for
    cross-issue provenance pins. Returns
    ``{"tier1": [(relpath, key, M, value)], "tier2": [(relpath, M, token)]}``
    or None when the dir is absent / no candidate files carry a pin / the
    `EPM_VERIFY_BODY_NO_EVAL_SCAN=1` fence is set (graceful skip).

    Corrupt / unreadable / oversize (`_REUSE_SCAN_MAX_BYTES` stat guard)
    JSONs are skipped silently (the `_scan_issue_judge_errors` convention —
    never crash the gate). A cheap substring pre-filter (`"metadata"` plus
    `hf_rev_` or `issue<digits>_`) avoids `json.loads` on the vast majority
    of files. `.jsonl` files are OUT of scope (parity with
    `_scan_issue_judge_errors`).
    """
    if os.environ.get("EPM_VERIFY_BODY_NO_EVAL_SCAN") == "1":
        return None
    eval_dir = repo / "eval_results" / f"issue_{issue}"
    if not eval_dir.is_dir():
        return None

    tier1: list[tuple[str, str, int, str]] = []
    tier2: list[tuple[str, int, str]] = []
    slug_probe = re.compile(r"\bissue\d+_")

    for path in sorted(eval_dir.rglob("*.json")):
        try:
            if path.stat().st_size > _REUSE_SCAN_MAX_BYTES:
                continue  # oversize guard — never page a 100+ MB blob at gate time
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if '"metadata"' not in text:
            continue
        if "hf_rev_" not in text and not slug_probe.search(text):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue  # a corrupt artifact must not crash the gate
        relpath = str(path.relative_to(repo))
        for md in _iter_metadata_dicts(payload):
            _collect_metadata_pins(md, issue, relpath, tier1, tier2)

    if not tier1 and not tier2:
        return None
    return {"tier1": tier1, "tier2": tier2}


def _tier1_satisfied(body_hex_tokens: set[str], value: str, M: int, body: str) -> bool:
    """True when a tier-1 revision pin `value` (source issue M) is declared
    in the body: a hex pin is satisfied by any >=7-hex-char body token that
    PREFIXES it (case-insensitive — a footer short sha `5aa6de1b` satisfies
    the metadata's 40-char pin); a non-hex pin (branch/tag) falls back to
    an issue-level `#M` / `/tasks/M` / `issue<M>_` mention."""
    v = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{7,64}", v):
        return any(v.startswith(tok) for tok in body_hex_tokens)  # lowercased, len>=7
    # non-hex pin (branch/tag): fall back to issue-level mention
    return bool(
        re.search(rf"#\s?{M}(?!\d)", body)
        or re.search(rf"/tasks/{M}(?!\d)", body)
        or f"issue{M}_" in body
    )


def _tier2_satisfied(body: str, M: int, token: str) -> bool:
    """True when a tier-2 path token (source issue M) is declared in the
    body: the `issue<M>_<slug>` first path segment appears verbatim, or an
    issue-level `#M` / `/tasks/M` mention does. The `(?!\\d)` digit boundary
    prevents `#7791` satisfying M=779."""
    seg = token.split("/")[0]  # e.g. "issue779_monitoring"
    return bool(
        seg in body or re.search(rf"#\s?{M}(?!\d)", body) or re.search(rf"/tasks/{M}(?!\d)", body)
    )


def check_cross_issue_reuse_provenance(
    body: str,
    *,
    issue: int | None = None,
    eval_root: Path | None = None,
    body_source_path: Path | None = None,
) -> CheckResult:
    """Check 35 (#1256): cross-issue reuse pins in committed result-JSON
    metadata must be declared in the body (canonical slot: the footer
    `Reused:` bullet, SPEC.md § `**Artifacts:**`).

    Verdict ladder:
      PASS-skip — not a v4 body (forward-only) / issue unknown (stdin) /
                  `EPM_VERIFY_BODY_NO_EVAL_SCAN=1` fence / eval root
                  unresolved (is_warn=True, judge-error parity) / no pins
                  found (graceful).
      FAIL      — a tier-1 pin (M != N) whose revision value has no
                  satisfying token in the body (`_tier1_satisfied`).
      WARN      — a tier-2 path hit (M != N) with no satisfying body
                  mention (`_tier2_satisfied`).
      PASS      — every pin satisfied.

    Satisfaction is checked against the WHOLE raw body text (not just the
    footer): the goal is "provenance is reader-visible", the FAIL detail
    points at the footer `Reused:` bullet as the canonical fix, and the LM
    clean-result-critic Lens 5 keeps owning placement/wording quality.
    Pins are deduped by (key, M, value) / (M, token) across files so one
    reused artifact yields one detail line.

    Documented residuals: (a) same-revision multi-artifact reuse — when two
    reused artifacts share one repo revision, a body declaring either
    satisfies both pins; one firing pin per round is sufficient to force
    the declaration fix, and LM Lens 5 stays the semantic backstop
    (#1092's `hf_rev_779_passb` was satisfied pre-fix via the declared r_B
    bullet's shared `037fcbb` revision). (b) `paper: true` tasks are gated
    by verify_paper.py and never reach this verifier.
    """
    name = "cross-issue reuse pins declared (footer Reused bullets)"
    if not is_v4(body):
        return CheckResult(name, True, "skipped — not a v4 body (forward-only)")
    if issue is None:
        return CheckResult(
            name, True, "skipped — issue number unknown (stdin); cannot read eval_results"
        )
    if os.environ.get("EPM_VERIFY_BODY_NO_EVAL_SCAN") == "1":
        return CheckResult(
            name, True, "skipped — EPM_VERIFY_BODY_NO_EVAL_SCAN=1 (eval scan fenced off)"
        )
    repo = _resolve_eval_root(issue, eval_root=eval_root, body_source_path=body_source_path)
    if repo is None:
        return CheckResult(name, True, "skipped — eval root unresolved", is_warn=True)
    pins = _scan_cross_issue_reuse_pins(repo, issue)
    if pins is None:
        return CheckResult(
            name,
            True,
            "no cross-issue reuse pins in committed result-JSON metadata — graceful skip",
        )

    body_hex_tokens = {t.lower() for t in _BODY_HEX_TOKEN_RE.findall(body)}

    unsatisfied_t1: list[tuple[str, str, int, str]] = []
    seen_t1: set[tuple[str, int, str]] = set()
    for relpath, key, src, value in pins["tier1"]:
        dedupe = (key, src, value)
        if dedupe in seen_t1:
            continue
        seen_t1.add(dedupe)
        if not _tier1_satisfied(body_hex_tokens, value, src, body):
            unsatisfied_t1.append((relpath, key, src, value))

    unsatisfied_t2: list[tuple[str, int, str]] = []
    seen_t2: set[tuple[int, str]] = set()
    for relpath, src, token in pins["tier2"]:
        dedupe = (src, token)
        if dedupe in seen_t2:
            continue
        seen_t2.add(dedupe)
        if not _tier2_satisfied(body, src, token):
            unsatisfied_t2.append((relpath, src, token))

    if unsatisfied_t1:
        lines = [
            f"`{relpath}` metadata key `{key}` pins #{src} @ {value[:12]} "
            f"with no matching body declaration"
            for relpath, key, src, value in unsatisfied_t1
        ]
        detail = (
            "; ".join(lines)
            + " — declare each reused artifact in the footer, expected shape: "
            + "`- Reused <kind> from [#M](...): <path> @ <rev> — fit: <one line>`"
        )
        return CheckResult(name, False, detail)
    if unsatisfied_t2:
        lines = [
            f"`{relpath}` metadata input path `{token}` references #{src} with no body mention"
            for relpath, src, token in unsatisfied_t2
        ]
        detail = (
            "; ".join(lines)
            + " — name the source (the `issue<M>_<slug>` path segment, `#M`, or a /tasks/M "
            + "link); canonical slot: the footer `Reused:` bullet"
        )
        return CheckResult(name, True, detail, is_warn=True)
    return CheckResult(
        name,
        True,
        f"all cross-issue reuse pins declared in the body "
        f"({len(seen_t1)} tier-1, {len(seen_t2)} tier-2)",
    )


# ─── Check 37 (#1370): footer Reused bullets carry a revision/path pin ──────

# The body->pin sibling of Check 35 (#1256). Check 35 fires only when
# committed result-JSON METADATA carries a machine pin
# (`_scan_cross_issue_reuse_pins`); a footer `- Reused ... from [#M](...)`
# bullet AUTHORED without any pinned path is invisible to it (#1315: two
# unpinned `- Reused ... from [#1090]` bullets while Check 35
# graceful-skipped; caught only by the LM critic). This check reads ONLY
# the body text — no eval scan, so `EPM_VERIFY_BODY_NO_EVAL_SCAN` does
# not (and must not) fence it.
_FOOTER_REUSED_BULLET_RE = re.compile(r"^\s*[-*]\s+Reused\b", re.IGNORECASE)
_REUSED_FROM_ISSUE_RE = re.compile(r"\bfrom\s+\[#(\d+)\]\(", re.IGNORECASE)
# Bullet-scoped pin forms (corpus 2026-07-15: 36/40 committed trigger
# bullets satisfy one; the satisfier is deliberately BULLET-scoped and
# excludes issue-mention forms — `#M` / `/tasks/M` appear in EVERY
# trigger bullet via the from-link itself, so the `_tier1_satisfied`
# fallback shapes would make this check vacuous):
_BULLET_REV_URL_RE = re.compile(r"/(?:tree|resolve|commit|blob)/[0-9a-fA-F]{7,40}\b")
_BULLET_AT_REV_RE = re.compile(r"@[ \t]*`?[0-9a-fA-F]{7,40}\b")
_BULLET_EVAL_PATH_RE = re.compile(r"\beval_results/issue_\d+/")
# SPEC.md § footer sanctions WandB `/runs/<id>` as a pinned URL form; a
# WandB run id is base36 (letters beyond a-f), so neither the rev-URL nor
# the bare-hex form catches it — without this a SPEC-conformant bullet
# would false-WARN:
_BULLET_WANDB_RUN_RE = re.compile(r"\bwandb\.ai/\S+/runs/\w+", re.IGNORECASE)


def _footer_reused_bullets(footer: str) -> list[str]:
    """Split footer text into `- Reused ...` bullets, joining indented
    non-bullet continuation lines onto their bullet (a wrapped bullet's
    pin may sit on a continuation line). Fence-aware: bullets inside a
    fenced code block (an illustrative skeleton) are ignored; `>`-quoted
    Context-prompt lines never match the bullet anchor."""
    bullets: list[str] = []
    current: str | None = None
    in_fence = False
    for line in footer.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            if current is not None:
                bullets.append(current)
                current = None
            continue
        if in_fence:
            continue
        if _FOOTER_REUSED_BULLET_RE.match(line):
            if current is not None:
                bullets.append(current)
            current = stripped
        elif current is not None and line[:1] in (" ", "\t") and not re.match(r"^\s*[-*]\s", line):
            current += " " + stripped
        else:
            if current is not None:
                bullets.append(current)
            current = None
    if current is not None:
        bullets.append(current)
    return bullets


def _reused_bullet_pinned(bullet: str) -> bool:
    """True when the bullet carries a pin: a revision-URL segment
    (`/tree/<sha>`, `/blob/<sha>`, `/resolve/<sha>`, `/commit/<sha>`),
    an `@ <rev>` (optional backtick), a committed
    `eval_results/issue_<M>/` path, a SPEC-sanctioned WandB run URL
    (`wandb.ai/<entity>/<project>/runs/<id>`), or a bare >=7-char hex
    token with at least one letter (a digits-only run like `12345678`
    is a count, not a sha)."""
    if (
        _BULLET_REV_URL_RE.search(bullet)
        or _BULLET_AT_REV_RE.search(bullet)
        or _BULLET_EVAL_PATH_RE.search(bullet)
        or _BULLET_WANDB_RUN_RE.search(bullet)
    ):
        return True
    return any(any(c.isalpha() for c in tok) for tok in _BODY_HEX_TOKEN_RE.findall(bullet))


def check_footer_reuse_bullets_pinned(body: str) -> CheckResult:
    """Check 37 (WARN, v4-only, #1370): every footer
    `- Reused ... from [#M](...)` bullet carries a revision/path pin.

    Body-text-only sibling of Check 35 (#1256) — the body->pin direction.
    WARN, not FAIL (corpus 2026-07-15: 40 trigger bullets across 40
    footered v4 bodies; 4 lack every pin form — 3 are code-harness reuse
    (#811/#833/#1112, remedy: append `@ <code-sha>`), 1 is the incident
    class (#810); a FAIL would newly block any future re-verify of all
    4 parked bodies). The from-link's own `#M` / `/tasks/M` NEVER
    satisfies — every trigger bullet carries it by construction.
    Documented false-negative residual: a bullet quoting the CURRENT
    task's own code SHA (a letter-bearing hex unrelated to the reused
    artifact) satisfies the bare-hex form — LM Lens 5 keeps owning
    semantic pin-correctness.
    Deliberately NOT fenced by `EPM_VERIFY_BODY_NO_EVAL_SCAN` (no
    filesystem scan)."""
    name = "footer Reused bullets carry a revision/path pin"
    if not is_v4(body):
        return CheckResult(name, True, "skipped — not a v4 body (forward-only)")
    footer = _v4_footer_text(body)
    if footer is None:
        return CheckResult(name, True, "skipped — no **Repro:** footer found")
    unpinned = [
        b
        for b in _footer_reused_bullets(footer)
        if _REUSED_FROM_ISSUE_RE.search(b) and not _reused_bullet_pinned(b)
    ]
    if not unpinned:
        return CheckResult(name, True, "all footer Reused-from-[#M] bullets pinned")
    shown = "; ".join(f"`{b[:120]}`" for b in unpinned)
    detail = (
        f"{len(unpinned)} footer `- Reused ... from [#M](...)` bullet(s) carry no "
        f"pinned path/revision: {shown} — add the permanent pin per reused artifact, "
        "expected shape: `- Reused <kind> from [#M](...): <path> @ <rev> — fit: "
        "<one line>` (an HF/GitHub `/tree/<sha>`-style URL, `@ <sha>`, a "
        "committed `eval_results/issue_<M>/...` path, or a WandB run URL)"
    )
    return CheckResult(name, True, detail, is_warn=True)


def _git_object_exists(repo: Path, sha: str, path: str) -> tuple[str, str]:
    """Return ('pass', '') if `git cat-file -e <sha>:<path>` succeeds,
    ('fail', detail) if the sha resolves but the path is absent, or
    ('skip', detail) if the sha cannot be resolved (unknown / shallow /
    truncated). Never raises — subprocess errors map to 'skip' with the
    reason so the check stays conservative.
    """
    # First confirm the sha itself resolves to a commit object. If not,
    # we cannot meaningfully assert presence/absence of the path.
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return "skip", f"git rev-parse failed: {e}"
    if rev.returncode != 0:
        return "skip", f"sha `{sha}` did not resolve in this repo (unknown / shallow)"
    # Sha resolved — now check the path at that sha.
    try:
        cat = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}:{path}"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return "skip", f"git cat-file failed: {e}"
    if cat.returncode == 0:
        return "pass", ""
    return "fail", f"`{path}` is NOT present in the tree at commit `{sha}`"


def _git_issue_branch_family(repo: Path, issue_n: str) -> list[str] | None:
    """Local branch family for an issue: `refs/heads/issue-<N>` plus every
    `refs/heads/issue-<N>-*` follow-up branch, via ONE
    `git for-each-ref --format='%(refname:short)'` call (check 29).

    Returns short names (e.g. `['issue-841', 'issue-841-fu2']`); `[]` when
    no family branch exists; None on any git error (fail-soft — the caller
    degrades the issue dir to a skip note, never a WARN)."""
    try:
        r = subprocess.run(
            [
                "git",
                "for-each-ref",
                "--format=%(refname:short)",
                f"refs/heads/issue-{issue_n}",
                f"refs/heads/issue-{issue_n}-*",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _git_tracked_under(repo: Path, ref: str, prefix: str) -> set[str] | None:
    """Set of paths tracked under `prefix` in the tree at `ref` — ONE
    `git -c core.quotePath=off ls-tree -r --name-only <ref> -- <prefix>`
    per call (check 29). Reads the tree object, NOT the working tree, so
    sparse checkouts are fine; `quotePath=off` keeps non-ASCII paths raw so
    set membership works. None on any git error (fail-soft: the caller
    degrades that REF to unprobed — never a WARN from a failed probe)."""
    try:
        r = subprocess.run(
            ["git", "-c", "core.quotePath=off", "ls-tree", "-r", "--name-only", ref, "--", prefix],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def check_repro_committed_claims_exist(body: str) -> CheckResult:
    """Reproducibility "committed at commit `<sha>`" claims must resolve.

    Conservative, additive verification of the body's `## Reproducibility`
    section. Fires ONLY when the section contains an explicit
    ``committed ... at commit `<sha>` `` phrase paired with at least one
    clearly-named repo-relative artifact path (`*.json`, `*.png`, `*.csv`,
    or anything under `figures/` / `eval_results/`) on the SAME line.

    For each such (sha, path) pair the check shells out to
    ``git cat-file -e <sha>:<path>`` in the repo root and:
      - FAILs when the sha resolves AND the path is absent from that tree
        (the body promises a committed file the SHA does not actually
        carry — the failure mode incident #397 surfaced: an on-pod-only
        artifact later deleted, with the body still falsely claiming
        commitment);
      - WARNs when the sha cannot be resolved (unknown / shallow clone /
        truncated copy) — we cannot make a confident claim either way;
      - PASSes silently when no "committed at commit `<sha>`" prose
        appears, when the prose appears but no checkable path pairs with
        it on the line, or when every (sha, path) pair resolves.

    Scope guards (so this never false-positives on PASS-worthy bodies):
      - Fenced code blocks inside Reproducibility are stripped before the
        scan, so a sha/path shown inside a ``` ... ``` example is ignored.
      - HF Hub URLs (`https://huggingface.co/...`) and WandB URLs
        (`https://wandb.ai/...`) are never matched — they carry no
        ``at commit `<sha>` `` prose marker, and their hex paths sit
        inside `()` link targets rather than backticks.
      - Prose without a backticked sha never trips the check.
      - Same-CLAUSE anchoring: sha claims and paths pair only within a
        clause — `;`/`·`/sentence boundaries outside backticks split the
        window (#893; incident #841: a "committed" token in one clause
        was paired with a later clause's ``at commit `<sha>` ``).

    Mechanical scope only — the semantic call ("did the experimenter
    actually upload this elsewhere, e.g. HF data repo?") belongs to
    upload-verifier Step 4, not this verifier. This check enforces only
    the within-body promise: if the body says "committed at commit X",
    that sha tree must carry the named file.
    """
    repro = _repro_section_text(body)  # v4 footer or `## Reproducibility` H2
    if repro is None:
        # Other checks (check_repro_subgroups / check_repro_url_permanence)
        # already FAIL on a missing Reproducibility section — don't double-
        # report. Stay silent here.
        return CheckResult(
            "Reproducibility committed-at-sha claims resolve",
            True,
            "no `## Reproducibility` section — other checks will report",
        )

    cleaned = _strip_fenced_blocks(repro)
    # Collect (sha, paths-in-same-CLAUSE) pairs. Same-clause anchoring
    # keeps the association unambiguous: if a sha and a path share a
    # clause they are almost certainly being asserted together; a
    # "committed" in one clause can no longer pair with a later clause's
    # ``at commit `<sha>` `` (#893, incident #841). Cross-line pairings
    # are intentionally out of scope (too noisy).
    pairs: list[tuple[str, str]] = []
    for line in cleaned.splitlines():
        # #893: scope BOTH regexes to a single clause so a "committed"
        # token can never pair with a later clause's "at commit `<sha>`"
        # (incident #841), and a line carrying two genuine claims
        # validates each against its own clause's paths (finditer).
        for clause in _split_clauses(line):
            for sha_match in _COMMITTED_AT_SHA_RE.finditer(clause):
                sha = sha_match.group("sha")
                for raw in _ARTIFACT_PATH_RE.findall(clause):
                    # ``_ARTIFACT_PATH_RE`` is a non-grouping disjunction that
                    # returns the full path capture; normalize a leading `./`.
                    p = raw[2:] if raw.startswith("./") else raw
                    # Reject absolute or remote-looking paths defensively.
                    if p.startswith("/") or p.startswith("~") or "://" in p:
                        continue
                    pairs.append((sha, p))

    if not pairs:
        return CheckResult(
            "Reproducibility committed-at-sha claims resolve",
            True,
            "no `committed ... at commit `<sha>`` claim with a paired "
            "repo-relative artifact path found",
        )

    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(
            "Reproducibility committed-at-sha claims resolve",
            True,
            f"{len(pairs)} committed-at-sha claim pair(s) found, but the repo "
            "root could not be resolved (running outside the repo?) — skipped",
            is_warn=True,
        )

    fails: list[str] = []
    skips: list[str] = []
    passes = 0
    for sha, path in pairs:
        verdict, detail = _git_object_exists(repo, sha, path)
        if verdict == "pass":
            passes += 1
        elif verdict == "fail":
            fails.append(detail)
        else:  # skip
            skips.append(f"`{sha}`:`{path}` — {detail}")

    if fails:
        return CheckResult(
            "Reproducibility committed-at-sha claims resolve",
            False,
            f"{len(fails)} of {len(pairs)} claim(s) FAILed: "
            + "; ".join(fails[:3])
            + (f" (+{len(fails) - 3} more)" if len(fails) > 3 else ""),
        )
    if skips:
        return CheckResult(
            "Reproducibility committed-at-sha claims resolve",
            True,
            f"{passes} pass, {len(skips)} unverifiable "
            f"(sha did not resolve — shallow clone or unknown ref): "
            + "; ".join(skips[:2])
            + (f" (+{len(skips) - 2} more)" if len(skips) > 2 else ""),
            is_warn=True,
        )
    return CheckResult(
        "Reproducibility committed-at-sha claims resolve",
        True,
        f"{passes} committed-at-sha claim pair(s) resolved cleanly",
    )


# Same-repo GitHub HTML blob/tree URL pinned to a hex sha — the shape the
# `**Code:**` subgroup links and the auto-appended `**Methodology
# reference:**` row use. `<path>` may name a file (blob) or a directory
# (tree); `git cat-file -e <sha>:<path>` resolves both object kinds. The
# `[^?#]+` class keeps query strings and fragments (`#L10` line anchors)
# out of the tree path.
_GITHUB_BLOB_TREE_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/(?:blob|tree)/(?P<sha>[0-9a-fA-F]{7,40})/(?P<path>[^?#]+)"
)

# A bare URL token in Reproducibility prose. Stops at whitespace, `)`
# (markdown link close), `<`/`>` (autolink delimiters), backticks (code
# spans), and `]` (reference-style links); trailing sentence punctuation
# is stripped at use-time.
_REPRO_URL_TOKEN_RE = re.compile(r"https?://[^\s\)<>`\]]+")


def _gather_repro_artifact_urls(repro: str) -> list[str]:
    """Collect same-repo, sha-addressable artifact URLs from the
    `## Reproducibility` section text (check 8b):
    `raw.githubusercontent.com/<this-repo>/<sha>/<path>` raw links and
    `github.com/<this-repo>/(blob|tree)/<sha>/<path>` HTML links. Fenced
    code blocks are stripped first so a URL shown inside a ``` ... ```
    example is illustrative, never probed. Blockquote lines
    (`>`-prefixed) are stripped too — a same-repo URL quoted inside the
    verbatim originating-prompt blockquote (`**Context:**` row) must not
    be existence-probed: the quote cannot be edited if its cited path
    later dies (#959). Other hosts (HF Hub, WandB)
    and other-repo GitHub links are out of scope: their existence is not
    decidable from the local object DB, and an unauthenticated 404 on an
    external private repo would false-FAIL. Order-preserving and
    deduplicated (at most one probe per unique URL)."""
    urls: list[str] = []
    for token in _REPRO_URL_TOKEN_RE.findall(_strip_blockquote_lines(_strip_fenced_blocks(repro))):
        url = token.rstrip(".,;:!?")
        for pattern in (_RAW_GITHUB_FIGURE_RE, _GITHUB_BLOB_TREE_URL_RE):
            m = pattern.match(url)
            if m and (m.group("owner").lower(), m.group("repo").lower()) == _THIS_REPO_SLUG:
                if url not in urls:
                    urls.append(url)
                break
    return urls


def _repro_artifact_url_existence(url: str) -> tuple[str, str]:
    """Existence probe for one same-repo artifact URL inside
    `## Reproducibility` (check 8b). Same verdict semantics as
    `_figure_url_existence`: ``('pass'|'fail'|'skip', note)``, where only
    a definitive miss is ``'fail'``. Raw ``raw.githubusercontent.com``
    URLs route through `_figure_url_existence` unchanged;
    ``github.com`` blob/tree HTML URLs resolve offline via
    ``git cat-file -e <sha>:<path>`` (file blobs AND directory trees),
    falling back to one HTTP HEAD when the sha is unknown to the local
    object database."""
    if _RAW_GITHUB_FIGURE_RE.match(url):
        return _figure_url_existence(url, noun="Reproducibility URL")
    m = _GITHUB_BLOB_TREE_URL_RE.match(url)
    if m is None:
        # Defensive — `_gather_repro_artifact_urls` only yields URLs
        # matching one of the two shapes above.
        return "skip", f"`{url}` (unrecognized URL shape)"
    path = m.group("path").rstrip("/")
    repo = _resolve_repo_root()
    if repo is not None:
        verdict, _detail = _git_object_exists(repo, m.group("sha"), path)
        if verdict == "pass":
            return "pass", ""
        if verdict == "fail":
            return (
                "fail",
                f"Reproducibility URL 404s — `{path}` does not exist at `{m.group('sha')[:8]}`",
            )
        # 'skip': sha unknown locally — fall through to the HTTP probe.
    code = _http_head_status(url)
    if code is None:
        return "skip", f"`{url}` (HTTP probe unavailable)"
    if code == 404:
        return "fail", f"Reproducibility URL 404s — `{url}`"
    if code < 400:
        return "pass", ""
    return "skip", f"`{url}` (HTTP {code})"


def check_repro_artifact_urls_exist(body: str) -> CheckResult:
    """Check 8b: same-repo artifact URLs in `## Reproducibility` must
    point at objects that actually exist.

    Extends the check-4b existence protection (incident task #507: a
    SHA-pinned figure URL that was never generated or committed PASSed
    the shape checks and rendered broken) to the `## Reproducibility`
    section, whose links previously got shape verification only:
    check 8 pins HF / WandB / GitHub URLs to permanent refs but never
    probes the target, and check 15 covers only the prose pattern
    ``committed ... at commit `<sha>` `` paired with a backticked
    repo-relative path — URL-shaped artifact references (the
    `**Artifacts:**` figure links, the `**Code:**` blob links, the
    auto-appended `**Methodology reference:**` row) escaped both.

    Scope: same-repo URLs only — `raw.githubusercontent.com/<this-repo>/
    <sha>/<path>` and `github.com/<this-repo>/(blob|tree)/<sha>/<path>`.
    SHA-pinned same-repo URLs resolve offline + deterministically via
    `git cat-file -e <sha>:<path>` (file blobs AND directory trees);
    unknown SHAs fall back to ONE HTTP HEAD per unique URL (the repo is
    public, so a definitive 404 FAILs). Indeterminate probes surface as
    an `unverified` note on the PASS line, never a FAIL, so offline
    runs don't block. HF Hub / WandB / external-repo links stay
    shape-checked only (check 8): their existence is not decidable from
    the local object DB, and an unauthenticated 404 on an external
    private repo would false-FAIL. Fenced code blocks are stripped
    before the scan.
    """
    name = "Reproducibility artifact URLs exist"
    repro = _repro_section_text(body)  # v4 footer or `## Reproducibility` H2
    if repro is None:
        # check_repro_subgroups / check_repro_url_permanence already
        # FAIL on a missing Reproducibility section — don't double-report.
        return CheckResult(name, True, "no `## Reproducibility` section — other checks will report")
    urls = _gather_repro_artifact_urls(repro)
    if not urls:
        return CheckResult(name, True, "no same-repo artifact URLs to check")
    bad: list[str] = []
    unverified: list[str] = []
    for url in urls:
        verdict, note = _repro_artifact_url_existence(url)
        if verdict == "fail":
            bad.append(note)
        elif verdict == "skip":
            unverified.append(note)
    if bad:
        return CheckResult(name, False, "; ".join(bad))
    detail = f"{len(urls)} URL(s)"
    if unverified:
        detail += f"; {len(unverified)} unverified (existence not confirmed): " + "; ".join(
            unverified
        )
    return CheckResult(name, True, detail)


# An HF Hub `/tree/<sha>/<path>` or `/blob/<sha>/<path>` URL pinned to a hex
# revision. Both dataset repos (`huggingface.co/datasets/<owner>/<repo>/...`)
# and model/space repos (`huggingface.co/<owner>/<repo>/...`) are matched.
# The `<sha>` group is restricted to a 7-40 char hex literal so a moving ref
# (`/tree/main`) is out of scope — its existence is undecidable at any point
# in time and check 8 already FAILs an unpinned HF URL on shape. `<path>` is
# optional (a bare `/tree/<sha>` repo-root link probes the revision itself).
# The `(?:datasets|spaces)/` prefix is captured so the right `repo_type` is
# threaded into the tree-endpoint URL the bounded direct GET hits.
_HF_HUB_TREE_BLOB_URL_RE = re.compile(
    r"^https?://huggingface\.co/"
    r"(?:(?P<kind>datasets|spaces)/)?"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/(?:tree|blob)/(?P<sha>[0-9a-fA-F]{7,40})"
    r"(?:/(?P<path>[^?#]*))?"
)

# ─── Bounded HF tree-endpoint existence probe (checks 23 & 25) ──────────────
#
# Checks 23/25 used to call `huggingface_hub.list_repo_files(...)`, which lists
# EVERY file in the repo at the revision via `list_repo_tree(recursive=True)`
# → `paginate()`. The first page is a bare `session.get()` (10 s default), but
# `list_repo_tree` exposes NO `timeout` kwarg and EVERY page after the first
# goes through `http_backoff("GET", next_page, max_retries=20,
# retry_on_status_codes=429)` (pure capped-exponential, ~143 s/page) — so a
# fleet-wide 429 storm stalled the verifier minutes per URL and stranded
# zombie processes (incident /issue #715, 2026-06-28). The data repo
# `superkaiba1/explore-persona-space-data` always paginates (>1000 files), so
# the whole-repo recursive listing always entered that loop under throttle.
#
# v2 fix (#733): drop `list_repo_files`/`list_repo_tree` entirely and probe the
# SAME Hub tree endpoint directly via `get_session().get(url, params, headers,
# timeout=_HF_PROBE_TIMEOUT_S)` — the per-request `timeout` is now real on
# EVERY GET, and we follow Link-header pagination OURSELVES under an outer page
# + wall-clock cap (checks 25/30/32, all via the shared `_hf_tree_pages`
# generator; check 23 needs a single page). A page-2 429 surfaces as
# `indeterminate` (-> skip) in seconds rather than entering the
# unbounded SDK backoff. Worst-case wall <= N * _HF_PROBE_ATTEMPTS *
# _HF_PROBE_TIMEOUT_S for check 23, <= N * _HF_PROBE_DEADLINE_S for the
# paginating check-25/30/32 path.
_HF_PROBE_TIMEOUT_S = 5.0  # per-request connect+read timeout (real on every GET)
_HF_PROBE_ATTEMPTS = 2  # at most 1 retry per page on a transient / 429
_HF_PROBE_SLEEP_S = 0.5  # tiny pause between the two attempts
_HF_PROBE_MAX_PAGES = 8  # self-pagination page cap (checks 25/30/32 via _hf_tree_pages)
_HF_PROBE_DEADLINE_S = 12.0  # outer wall cap across all pages of ONE probe (checks 25/30/32)
# Per-process cache keyed on (repo_id, repo_type, sha, path_prefix[, keyword]).
# Caches ONLY definitive pass/fail verdicts — a `skip` (indeterminate / throttle)
# is NEVER cached, so a transient throttle that has since cleared is always
# re-probed on a re-entry (task_workflow_migrate.py re-runs verify_text() in one
# long-running process). Values are (verdict, note) tuples.
_HF_EXISTENCE_CACHE: dict[tuple, tuple[str, str]] = {}


class _TreeProbeResult(NamedTuple):
    """Structured outcome of ONE bounded GET to a Hub tree URL.

    The `status` is returned (NOT a re-raised exception) so each call site maps
    a not-found INDEPENDENTLY: check 23 maps `not_found` → FAIL (dead pin,
    #537); check 25 maps the SAME `not_found` → SKIP (indeterminate — cannot
    corroborate/refute a denial). Centralizing a shared exception mapping would
    collapse that deliberate asymmetry.
    """

    status: str  # "ok" | "not_found" | "indeterminate"
    entries: list[dict]  # JSON tree entries for this page (empty unless status == "ok")
    next_page: str | None  # the Link rel="next" URL, or None
    note: str  # diagnostic for the indeterminate / skip note


def _hf_tree_url(repo_id: str, repo_type: str, sha: str, path: str) -> str:
    """Build the Hub tree-endpoint URL exactly as `HfApi.list_repo_tree` does:
    `{endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}{/encoded_path}`.
    The revision is left unencoded (it is a hex sha); the path is
    `quote(path, safe="")`-encoded and prefixed with `/`, empty for the root.
    Importing the `constants` submodule explicitly makes it
    attribute-reachable on a fresh process — a bare `import huggingface_hub`
    does not expose the lazy submodule (#1186).
    """
    import huggingface_hub.constants

    endpoint = huggingface_hub.constants.ENDPOINT
    encoded_path = "/" + quote(path, safe="") if path else ""
    return f"{endpoint}/api/{repo_type}s/{repo_id}/tree/{sha}{encoded_path}"


def _hf_build_headers() -> dict:
    """The SDK auth headers for a Hub request (token from the ambient env /
    cached login), via `huggingface_hub.utils.build_hf_headers`."""
    from huggingface_hub.utils import build_hf_headers

    return build_hf_headers()


def _hf_tree_get(
    url: str, params: dict | None, headers: dict, *, timeout_s: float
) -> _TreeProbeResult:
    """ONE bounded GET to a Hub tree URL, with our OWN bounded retry.

    Returns a `_TreeProbeResult`. NEVER enters the SDK `http_backoff` loop:
    this is a single `requests.get` with an explicit per-request `timeout`,
    retried at most `_HF_PROBE_ATTEMPTS` times on a transient error / 429 / 5xx
    with a tiny `_HF_PROBE_SLEEP_S` pause between attempts.

    - 2xx              → status='ok',           entries=r.json(), next_page=Link-rel-next
    - 404 not-found    → status='not_found'     (caller decides FAIL vs SKIP)
    - 429 / 5xx / conn / timeout / parse error → status='indeterminate' (caller → skip)
    """
    from huggingface_hub.utils import get_session

    last_note = "no attempt made"
    for i in range(_HF_PROBE_ATTEMPTS):
        try:
            r = get_session().get(url, params=params, headers=headers, timeout=timeout_s)
        except Exception as exc:  # connection / timeout / DNS — all transient
            last_note = f"HF tree probe failed: {type(exc).__name__}: {exc}"
            if i + 1 < _HF_PROBE_ATTEMPTS:
                time.sleep(_HF_PROBE_SLEEP_S)
            continue
        if r.status_code == 404:
            return _TreeProbeResult("not_found", [], None, "")
        if r.status_code == 429 or r.status_code >= 500:
            last_note = f"HF tree probe failed: HTTP {r.status_code}"
            if i + 1 < _HF_PROBE_ATTEMPTS:
                time.sleep(_HF_PROBE_SLEEP_S)
            continue
        try:
            r.raise_for_status()
            entries = r.json()
        except Exception as exc:
            return _TreeProbeResult("indeterminate", [], None, f"HF tree probe failed: {exc}")
        if not isinstance(entries, list):
            return _TreeProbeResult(
                "indeterminate", [], None, "HF tree probe failed: unexpected response shape"
            )
        next_page = r.links.get("next", {}).get("url")
        return _TreeProbeResult("ok", entries, next_page, "")
    return _TreeProbeResult("indeterminate", [], None, last_note)


class _TreePageEvent(NamedTuple):
    """One event from `_hf_tree_pages`: zero or more kind='page' events
    (entries = that page's JSON tree entries), then EXACTLY ONE terminal
    event — kind in {'exhausted', 'cap', 'not_found', 'indeterminate'}.
    `note` is non-empty only for kind='indeterminate' (the `_hf_tree_get`
    diagnostic); callers own every human-facing string for the other
    terminal kinds (the deliberate per-check note asymmetry: check 25 says
    "(no such revision)", checks 30/32 say "no such revision/path")."""

    kind: str  # "page" | "exhausted" | "cap" | "not_found" | "indeterminate"
    entries: list[dict]  # populated only for kind="page"
    note: str  # populated only for kind="indeterminate"


def _hf_tree_pages(
    repo_id: str, repo_type: str, sha: str, path: str, *, recursive: bool = True
) -> Iterator[_TreePageEvent]:
    """Shared bounded Link-header self-pagination over the Hub tree endpoint
    (checks 25 / 30 / 32; the #733 contract in ONE place so a 4th consumer
    inherits it). Reads the module caps directly (`_HF_PROBE_MAX_PAGES`,
    `_HF_PROBE_DEADLINE_S`, `_HF_PROBE_TIMEOUT_S`); per-page 429/5xx retry
    stays inside `_hf_tree_get`. As a generator it is lazy: a caller that
    never consumes it issues ZERO GETs (headers/URL construction and the
    first GET all run at the first `next()`). Invariants:
    - headers are built BEFORE the URL (`_hf_build_headers` imports
      `huggingface_hub.utils`, making the lazy `constants` submodule
      attribute-reachable — belt to `_hf_tree_url`'s own explicit
      `import huggingface_hub.constants`, #1186; the ordering precedent is
      check 23's `_hf_probe_existence` + the #1016 check-32 fix);
    - `params` are sent on the FIRST page only (the Link rel="next" URL
      already carries them);
    - `exhausted` is checked BEFORE the page/deadline cap, so a listing
      whose final page lands exactly on the cap is exhaustive, not capped;
    - the deadline is strict `>` from just before the first GET.
    """
    headers = _hf_build_headers()
    url = _hf_tree_url(repo_id, repo_type, sha, path)
    params: dict | None = {"recursive": recursive}
    started = time.monotonic()
    pages = 0
    while True:
        res = _hf_tree_get(url, params=params, headers=headers, timeout_s=_HF_PROBE_TIMEOUT_S)
        if res.status == "not_found":
            yield _TreePageEvent("not_found", [], "")
            return
        if res.status == "indeterminate":
            yield _TreePageEvent("indeterminate", [], res.note)
            return
        yield _TreePageEvent("page", res.entries, "")
        pages += 1
        if res.next_page is None:
            yield _TreePageEvent("exhausted", [], "")
            return
        if pages >= _HF_PROBE_MAX_PAGES or time.monotonic() - started > _HF_PROBE_DEADLINE_S:
            yield _TreePageEvent("cap", [], "")
            return
        # The Link rel="next" URL already carries the params; do not re-send them.
        url, params = res.next_page, None


def _gather_hf_pinned_urls(body: str) -> list[tuple[str, str, str, str, str]]:
    """Collect HF Hub revision-pinned `/tree/<sha>/<path>` and
    `/blob/<sha>/<path>` URLs from the WHOLE body (check 23). HF artifact
    links live in `## Reproducibility` (the `**Artifacts:**` model/dataset
    rows), in `## Data` (the `### Trained on` / `### Generated` complete-data
    pointers), and in `## TL;DR` / `## Findings` (the cherry-picked-completion
    "full data at ..." links), so the scan is body-wide, not section-scoped.

    Fenced code blocks are stripped first so a URL shown inside a ``` ... ```
    example is illustrative, never probed. Returns order-preserving,
    deduplicated `(repo_id, repo_type, sha, path_prefix, raw_url)` tuples —
    at most one Hub probe per unique (repo_id, sha, path_prefix). `repo_type`
    is one of ``"dataset"`` / ``"space"`` / ``"model"`` — threaded into the
    bounded tree-endpoint GET URL (`_hf_tree_url`).
    """
    kind_to_type = {"datasets": "dataset", "spaces": "space", None: "model"}
    out: list[tuple[str, str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for token in _REPRO_URL_TOKEN_RE.findall(_strip_fenced_blocks(body)):
        url = token.rstrip(".,;:!?")
        m = _HF_HUB_TREE_BLOB_URL_RE.match(url)
        if m is None:
            continue
        repo_id = f"{m.group('owner')}/{m.group('repo')}"
        repo_type = kind_to_type[m.group("kind")]
        sha = m.group("sha")
        path_prefix = (m.group("path") or "").rstrip("/")
        key = (repo_id, sha, path_prefix)
        if key in seen:
            continue
        seen.add(key)
        out.append((repo_id, repo_type, sha, path_prefix, url))
    return out


def _hf_probe_existence(
    repo_id: str, repo_type: str, sha: str, path_prefix: str
) -> tuple[str, str]:
    """Bounded direct-GET existence probe for one HF Hub URL (check 23).

    Returns ``(verdict, note)``. For a bare ``/tree/<sha>`` link it lists the
    repo ROOT (page 1) and PASSes iff the revision resolves. For a path-pinned
    link it lists the needle's PARENT dir non-recursively (page 1) and matches
    the needle among the parent's DIRECT children — a valid pinned link's
    terminal component is always a direct child of the path it claims, so this
    is consistent with check 23 only ever matching direct children (the old
    whole-repo recursive listing could have matched an arbitrarily-deep
    descendant; the parent-listing scheme matches direct children only, a
    documented #733 trade-off — every existing fixture uses direct-child
    paths so no verdict changes).

    not_found → FAIL (dead pin, #537) — this call site's INDEPENDENT mapping.
    """
    needle = path_prefix.rstrip("/")
    parent = posixpath.dirname(needle) if needle else ""
    headers = _hf_build_headers()
    res = _hf_tree_get(
        _hf_tree_url(repo_id, repo_type, sha, parent),
        params={"recursive": False},
        headers=headers,
        timeout_s=_HF_PROBE_TIMEOUT_S,
    )
    if res.status == "not_found":
        return (
            "fail",
            f"HF URL dead revision pin — `{repo_id}` has no revision `{sha[:8]}` "
            f"or path `{needle or '/'}`",
        )
    if res.status == "indeterminate":
        return "skip", f"`{repo_id}@{sha[:8]}` ({res.note})"
    if not needle:
        # A bare `/tree/<sha>` repo-root link: a successful listing means the
        # revision exists, which is all such a link asserts.
        return "pass", ""
    paths = {e["path"].rstrip("/") for e in res.entries if "path" in e}
    if needle in paths:
        return "pass", ""
    return (
        "fail",
        f"HF URL dead revision pin — `{needle}` resolves to 0 files at `{repo_id}@{sha[:8]}`",
    )


def _hf_url_existence(repo_id: str, repo_type: str, sha: str, path_prefix: str) -> tuple[str, str]:
    """Existence probe for one HF Hub revision-pinned URL (check 23).

    Returns ``(verdict, note)`` with verdict one of ``'pass'`` / ``'fail'``
    (the path resolves to ZERO files at the cited revision — a dead pin) /
    ``'skip'`` (indeterminate — surfaced as an `unverified` note on the PASS
    line, never a FAIL, so offline / unauthenticated runs don't block).

    Fail-soft is mandatory: the probe SKIPs (never FAILs) when
    ``EPM_VERIFY_BODY_NO_HF=1`` is set (the suite-wide offline fence in
    ``tests/conftest.py``), when ``huggingface_hub`` is not importable, or
    when the Hub probe raises any network / auth / unexpected error. Only a
    SUCCESSFUL listing that returns zero matching files — or a definitive
    repository/revision-not-found — is a FAIL.

    The Hub probe is a BOUNDED direct GET of the tree endpoint
    (``_hf_probe_existence`` → ``_hf_tree_get``), NOT the unbounded
    ``list_repo_files`` recursive whole-repo listing (#733). Definitive
    ``pass``/``fail`` verdicts are cached per-process; a ``skip`` is never
    cached, so a transient throttle that has since cleared is always re-probed.
    """
    if os.environ.get("EPM_VERIFY_BODY_NO_HF") == "1":
        return "skip", f"`{repo_id}@{sha[:8]}` (HF probe fenced)"
    try:
        import huggingface_hub  # noqa: F401 — local import: optional-dependency guard
    except ImportError:
        return "skip", f"`{repo_id}@{sha[:8]}` (huggingface_hub unavailable)"
    cache_key = (repo_id, repo_type, sha, path_prefix.rstrip("/"))
    cached = _HF_EXISTENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    verdict, note = _hf_probe_existence(repo_id, repo_type, sha, path_prefix)
    if verdict in ("pass", "fail"):
        _HF_EXISTENCE_CACHE[cache_key] = (verdict, note)
    return verdict, note


def check_hf_url_resolves(body: str) -> CheckResult:
    """Check 23: every HF Hub revision-pinned `/tree/<sha>/<path>` or
    `/blob/<sha>/<path>` URL in the body must resolve to ≥1 file at the
    cited revision.

    Extends the #507 existence-protection class (check 4b for inline
    figures, check 8b for same-repo Reproducibility links) to HF Hub
    artifact links, which check 8 deliberately left shape-checked only
    ("existence is not decidable from the local object DB"). It IS decidable
    via the Hub tree endpoint, probed by a BOUNDED direct GET
    (`_hf_tree_get`, with a real per-request timeout; #733), which lists the
    files present at a revision, so a path pinned to a revision that predates
    the upload — resolving to ZERO files — is caught mechanically. Incident
    task #537: the `## Reproducibility` `**Artifacts:**`
    line pinned the "415 bakeoff intermediates" link to revision `db3662ae`
    (the main-grid revision, predating the bakeoff round entirely), where the
    path resolves to 0 files; a reader — or a downstream reuse-premise miner
    — clicking it gets nothing. The same dead-pin class slips through every
    other check (the URL is shape-valid, sha-pinned, and on a real repo).

    The scan is body-wide (HF links live in `## Reproducibility`, `## Data`,
    and the cherry-picked-completion prose under `## TL;DR` / `## Findings`)
    and fence-stripped (a URL inside a ``` example is illustrative).
    Moving refs (`/tree/main`) are out of scope — check 8 FAILs those on
    shape; only hex-pinned revisions are probed.

    Fail-soft (same semantics as check 8b): a definitive zero-file /
    no-such-revision result is a FAIL; everything indeterminate — the
    `EPM_VERIFY_BODY_NO_HF=1` offline fence (set suite-wide by
    `tests/conftest.py`), a missing `huggingface_hub`, or any network /
    auth error — surfaces as an `unverified` note on the PASS line, never a
    FAIL, so offline and unauthenticated runs don't block.
    """
    name = "HF URL pins resolve at the cited revision"
    urls = _gather_hf_pinned_urls(body)
    if not urls:
        return CheckResult(name, True, "no HF Hub revision-pinned URLs to check")
    bad: list[str] = []
    unverified: list[str] = []
    for repo_id, repo_type, sha, path_prefix, _raw in urls:
        verdict, note = _hf_url_existence(repo_id, repo_type, sha, path_prefix)
        if verdict == "fail":
            bad.append(note)
        elif verdict == "skip":
            unverified.append(note)
    if bad:
        return CheckResult(name, False, "; ".join(bad))
    detail = f"{len(urls)} HF URL(s)"
    if unverified:
        detail += f"; {len(unverified)} unverified (existence not confirmed): " + "; ".join(
            unverified
        )
    return CheckResult(name, True, detail)


# Reproducibility per-figure commit claim — `` `<basename>` at commit `<sha>` ``
# or the shorter `` `<basename>` at `<sha>` `` form the analyzer's `**Figures:**`
# bullet uses (worked example #537). `<basename>` is the figure filename WITHOUT
# the `.png` / `.pdf` extension (the analyzer keys claims by stem). The sha is
# 7-40 hex chars. Backtick-anchored on both the basename and the sha so prose
# "the figure at commit abc" without backticks never matches.
_REPRO_FIGURE_SHA_RE = re.compile(
    r"`(?P<name>[\w.\-/]+?)`\s+at\s+(?:commit\s+)?`(?P<sha>[0-9a-fA-F]{7,40})`"
)
# Catch-all default: `` all others at `<sha>` `` / `` everything else at commit `<sha>` ``
# the analyzer appends to the `**Figures:**` bullet so it need not enumerate
# every figure (worked example #537: "... all others at `bdb0ae0...`").
_REPRO_FIGURE_DEFAULT_SHA_RE = re.compile(
    r"\b(?:all\s+others|everything\s+else|the\s+rest|remaining)\s+at\s+"
    r"(?:commit\s+)?`(?P<sha>[0-9a-fA-F]{7,40})`",
    re.IGNORECASE,
)
# Start of the analyzer's figures bullet inside `## Reproducibility` — a list
# item whose label is `Figures` (optionally bold / parenthesized / dir-linked):
# `- Figures (PNG + PDF ...): ...`, `- **Figures:** ...`, `- Figures: [...]`.
# The scan for per-figure sha claims is SCOPED to this bullet's text so an
# incidental `` `main` at `<sha>` `` branch-merge note elsewhere in
# `## Reproducibility` (e.g. the `**Context:**` follow-up-lineage bullet,
# incident #480) never trips the figure-claim regex.
_REPRO_FIGURES_BULLET_RE = re.compile(r"^[ \t]*[-*]\s+\*{0,2}Figures\b", re.IGNORECASE)


def _figures_bullet_text(repro_cleaned: str) -> str:
    """Return the text of the `- Figures ...` bullet(s) inside a
    fence-stripped `## Reproducibility` section, or '' if there is none.

    A markdown list item runs from its `- ` marker until the next top-level
    list item (`- ` / `* ` at the same indent) or a blank line that is NOT
    followed by an indented continuation. Keeping the scan inside this bullet
    is what stops the figure-sha claim regex from matching unrelated
    `` `name` at `sha` `` prose elsewhere in Reproducibility (incident #480:
    `` merged to `main` at `<sha>` `` in the Context bullet)."""
    lines = repro_cleaned.splitlines()
    chunks: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _REPRO_FIGURES_BULLET_RE.match(lines[i]):
            bullet = [lines[i]]
            i += 1
            # Consume continuation lines: indented text or non-empty,
            # non-list-marker lines belonging to the same bullet. Stop at the
            # next top-level list marker or a blank line.
            while i < n:
                ln = lines[i]
                if ln.strip() == "":
                    break
                if re.match(r"^[ \t]*[-*]\s", ln):
                    break
                bullet.append(ln)
                i += 1
            chunks.append("\n".join(bullet))
        else:
            i += 1
    return "\n".join(chunks)


def _shas_compatible(a: str, b: str) -> bool:
    """True when two hex SHA strings refer to the same commit allowing for
    abbreviation — one is a case-insensitive prefix of the other. The
    Reproducibility claim is sometimes abbreviated (7-12 chars) while the
    inline raw-GitHub URL always carries the full 40-char sha, so an exact
    string compare would false-FAIL a correctly-pinned abbreviated claim."""
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def check_figure_url_sha_matches_repro(body: str) -> CheckResult:
    """Check 22: each inline figure URL's commit SHA must match the SHA the
    `## Reproducibility` `**Figures:**` bullet pins that figure to.

    A clean-result inlines its figures as SHA-pinned raw-GitHub URLs
    (`.../<url_sha>/figures/issue_<N>/<basename>.png`) and SEPARATELY records
    the commit each figure was pinned at in the `## Reproducibility`
    `**Figures:**` bullet (`` `<basename>` at commit `<sha>` ``, with an
    `` all others at `<sha>` `` catch-all). Across follow-up rounds those two
    SHAs can drift: a figure is regenerated at a new commit, the inline URL
    is updated, but the Reproducibility claim still names the old commit (or
    vice versa). The image still renders (check 4b probes the URL's OWN sha
    for existence) and every other check PASSes, so the body ships with the
    inline image pointing at a different commit than `## Reproducibility`
    claims — a recurring, mechanizable defect (worked example #537's
    `predictor_bakeoff_complete_null`: inline `5ad30c2…` vs Reproducibility
    `c539920…`, caught by hand at round-3 interp-critique).

    This is a SHAPE-CONSISTENCY check, not an existence probe — it compares
    the two SHAs the body already carries; it never touches git or the
    network. Matching:

    - For each inline figure URL with a 7-40-hex `/<sha>/figures/.../<file>`
      component, take `<basename>` = `<file>` minus its extension.
    - If `## Reproducibility` pins that basename explicitly
      (`` `<basename>` at [commit] `<sha>` ``), the URL sha must be
      prefix-compatible with the claimed sha (one abbreviation of the other).
    - Else if `## Reproducibility` carries a `` all others at `<sha>` ``
      default, the URL sha must be prefix-compatible with the default.
    - Else (no per-figure claim AND no default) the figure is NOT
      enumerated in Reproducibility — SKIP it (never a FAIL): a body may
      legitimately omit some figures from the `**Figures:**` bullet, and
      this check only screens claims the body actually makes.

    Fenced code blocks in `## Reproducibility` are stripped before scanning
    the claims so a sha shown inside an illustrative ``` ... ``` example is
    ignored. NO-OP PASS when the body has no inline figure URLs, no
    Reproducibility section, or no figure-sha claims at all.
    """
    name = "figure URL sha matches Reproducibility"
    repro = _repro_section_text(body)  # v4 footer or `## Reproducibility` H2
    if repro is None:
        # check_repro_subgroups / check_repro_url_permanence already FAIL on
        # a missing Reproducibility section — don't double-report.
        return CheckResult(name, True, "no `## Reproducibility` section — other checks will report")
    cleaned_repro = _strip_fenced_blocks(repro)
    # Scope the claim scan to the analyzer's `- Figures ...` bullet so an
    # incidental `` `name` at `sha` `` elsewhere in `## Reproducibility`
    # (a branch-merge note in the Context bullet, incident #480) is never
    # read as a figure-commit claim.
    figures_text = _figures_bullet_text(cleaned_repro)
    # Build the basename -> claimed-sha map and find the catch-all default.
    claimed: dict[str, str] = {}
    for m in _REPRO_FIGURE_SHA_RE.finditer(figures_text):
        nm = m.group("name")
        # Key by the bare stem so `figures/issue_537/foo.png`, `foo.png`,
        # and `foo` in the claim all resolve to `foo` (the inline-URL key).
        stem = nm.rsplit("/", 1)[-1]
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
        # First explicit claim for a basename wins (defensive; duplicates
        # would themselves be a body bug a human should fix).
        claimed.setdefault(stem, m.group("sha"))
    default_match = _REPRO_FIGURE_DEFAULT_SHA_RE.search(figures_text)
    default_sha = default_match.group("sha") if default_match else None

    if not claimed and default_sha is None:
        return CheckResult(
            name, True, "no per-figure commit claim in `## Reproducibility` `**Figures:**`"
        )

    fails: list[str] = []
    checked = 0
    for url in _gather_figure_image_urls(body):
        url = url.strip().split(None, 1)[0] if url.strip() else ""
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if m is None:
            continue
        url_sha = m.group("sha")
        path = m.group("path")
        basename = path.rsplit("/", 1)[-1]
        if "." in basename:
            basename = basename.rsplit(".", 1)[0]
        claim_sha = claimed.get(basename, default_sha)
        if claim_sha is None:
            # Figure not enumerated in Reproducibility — out of scope.
            continue
        checked += 1
        if not _shas_compatible(url_sha, claim_sha):
            source = "explicit claim" if basename in claimed else "`all others` default"
            fails.append(
                f"`{basename}`: inline URL sha `{url_sha[:8]}` vs Reproducibility "
                f"{source} sha `{claim_sha[:8]}`"
            )
    if fails:
        return CheckResult(
            name,
            False,
            f"{len(fails)} figure(s) with an inline-URL / Reproducibility sha mismatch — "
            "update the `**Figures:**` bullet or the inline URL so both pin the same "
            "commit: " + "; ".join(fails),
        )
    if checked == 0:
        return CheckResult(
            name, True, "no inline figure URL matched a Reproducibility figure-sha claim"
        )
    return CheckResult(name, True, f"{checked} figure URL sha(s) match their Reproducibility claim")


# ─── Check 24: figure-embedded text vs body prose (figure-text staleness) ────
#
# `.meta.json` sidecar keys that are PURE PROVENANCE (commit hashes,
# timestamps, platform strings, argv, figsize) — never chart text — and are
# skipped when flattening the sidecar to a scannable string blob, so a commit
# sha inside the sidecar can never be read as a stale chart token. Matched
# case-insensitively against the leaf KEY name (not the value).
_META_PROVENANCE_KEYS = frozenset(
    {
        "commit",
        "git_commit",
        "git_sha",
        "created",
        "ts_utc",
        "timestamp_utc",
        "timestamp",
        "python_version",
        "platform",
        "argv",
        "figsize",
        "cuda_visible_devices",
        "n_series",
        "total_points",
        "data_truncated",
        "data_path",
        "figure",
        "script",
    }
)

# Default stale-token list: project-specific strings the body prose is known
# to have softened away but a figure-generation script may still hardcode in a
# title / annotation. Kept tiny + conservative (WARN, not FAIL) — extend via a
# user-supplied `~/.eps-stale-tokens.json` (a JSON list of strings), read
# fail-soft by `_load_stale_figure_tokens`. Matched case-insensitively as a
# substring of the flattened figure-text blob.
STALE_FIGURE_TOKENS: tuple[str, ...] = (
    "geometrically real",  # round-1 plain-cosine softening case (#667)
)

# A simple `<numerator>/<denominator>` fraction token (chance-level values like
# `1/30`, `1/29`). Bounded by non-digit / start / end so `1/30` matches but a
# date like `2026/06/24` or a path like `issue_667/figures` does not (the inner
# `/` there is flanked by digits on BOTH sides across >2 groups — handled by
# requiring the whole token to be exactly two digit-runs).
_FRACTION_RE = re.compile(r"(?<![\d/])(?P<num>\d{1,4})/(?P<den>\d{1,4})(?![\d/])")


def _load_stale_figure_tokens() -> list[str]:
    """Return the configured stale-figure-token list: the module-level
    ``STALE_FIGURE_TOKENS`` constant plus any strings in an optional
    ``~/.eps-stale-tokens.json`` file (a JSON array of strings).

    The user file is read FAIL-SOFT — absent / unreadable / malformed → the
    built-in constant alone, never an error. This keeps the check a soft
    mechanical aid the operator can extend without editing the verifier.
    """
    tokens = [t for t in STALE_FIGURE_TOKENS if t]
    cand = Path.home() / ".eps-stale-tokens.json"
    try:
        if cand.is_file():
            extra = json.loads(cand.read_text())
            if isinstance(extra, list):
                tokens.extend(str(t) for t in extra if isinstance(t, str) and t.strip())
    except (OSError, ValueError):
        pass  # fail-soft: optional config, never blocks the check
    # De-dup case-insensitively, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _flatten_meta_strings(obj: object) -> list[str]:
    """Recursively collect text-bearing leaf strings from a parsed
    ``.meta.json`` sidecar, SKIPPING pure-provenance keys.

    The canonical `savefig_paper` sidecar carries chart text in several
    shapes — top-level ``description``, per-row ``series`` / ``label``
    strings and category-keyed string values under ``points`` / ``rows``,
    and the forward-looking ``title`` / ``annotations`` / ``caption`` /
    ``xlabel`` / ``ylabel`` keys the candidate names — so we walk the whole
    structure rather than hard-coding a key list. Numeric leaves and
    provenance keys (``commit`` / ``created`` / ``git_sha`` / ``argv`` …,
    see ``_META_PROVENANCE_KEYS``) are dropped so a commit sha or timestamp
    in the sidecar is never mistaken for a stale chart token. Returns the
    flat list of strings (callers join + scan).
    """
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.casefold() in _META_PROVENANCE_KEYS:
                continue
            # Dict KEYS can themselves carry chart text (an axis-label-keyed
            # data row uses the axis label as the key, e.g.
            # `{"1/30 chance accuracy": 0.41}`), so collect non-provenance
            # string keys too.
            if isinstance(k, str) and k.strip():
                out.append(k)
            out.extend(_flatten_meta_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_flatten_meta_strings(item))
    elif isinstance(obj, str) and obj.strip():
        out.append(obj)
    # numbers / bools / None contribute no scannable text
    return out


def _read_figure_meta_text(repo: Path, sha: str, fig_path: str) -> str | None:
    """Return the flattened text blob of the figure's sibling ``.meta.json``
    (``<fig_path>`` with its extension swapped to ``.meta.json``) read out of
    the git tree at ``sha`` via ``git show``, or None when there is nothing to
    scan (no sidecar at that sha, unresolvable sha, parse failure).

    Reads from the git OBJECT DB (``git show <sha>:<meta_path>``) rather than
    the working tree, because the body pins figures to a specific commit and a
    worktree shares the object database with the main checkout. FAIL-SOFT
    throughout: any subprocess / decode / JSON error → None (the check skips
    that figure rather than blocking). One ``git show`` per unique figure.
    """
    base, _, ext = fig_path.rpartition(".")
    meta_path = (base if ext else fig_path) + ".meta.json"
    try:
        proc = subprocess.run(
            ["git", "show", f"{sha}:{meta_path}"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None  # no sidecar at that sha, or sha unresolvable
    try:
        meta = json.loads(proc.stdout)
    except (ValueError, json.JSONDecodeError):
        return None
    strings = _flatten_meta_strings(meta)
    if not strings:
        return None
    return " ".join(strings)


def _figure_caption_after(rlines: list[str], img_line_idx: int) -> str:
    """Return the figure caption text that follows the inline image at
    ``rlines[img_line_idx]``: the contiguous run of blockquote (``> …``)
    lines after the image (skipping blank lines), the analyzer's caption
    convention (``> **Figure.** *…* …``). Empty string when there is no
    blockquote caption directly after the image.
    """
    n = len(rlines)
    i = img_line_idx + 1
    # Skip blank lines between the image and its caption.
    while i < n and rlines[i].strip() == "":
        i += 1
    cap: list[str] = []
    while i < n and rlines[i].lstrip().startswith(">"):
        cap.append(rlines[i].lstrip()[1:].strip())
        i += 1
    return " ".join(cap).strip()


def _figure_text_warnings(
    fig_text: str, caption: str, basename: str, stale_tokens: list[str]
) -> list[str]:
    """Return the check-24 WARN messages for ONE figure's flattened sidecar
    text ``fig_text`` against its body ``caption``.

    Two signals (see ``check_figure_text_vs_body_tokens``):
    (b) softened token — any ``stale_tokens`` string present in ``fig_text``.
    (a) chance/N disagreement — a ``<a>/<b>`` fraction in ``fig_text`` whose
        same-numerator counterpart in ``caption`` uses a different denominator.
    """
    warns: list[str] = []
    fig_lower = fig_text.casefold()
    for tok in stale_tokens:
        if tok.casefold() in fig_lower:
            warns.append(
                f"`{basename}` figure text contains softened token `{tok}` "
                "(removed from body prose) — regenerate the figure"
            )
    cap_fracs: dict[str, set[str]] = {}
    for fm_ in _FRACTION_RE.finditer(caption):
        cap_fracs.setdefault(fm_.group("num"), set()).add(fm_.group("den"))
    if cap_fracs:
        seen_pairs: set[tuple[str, str]] = set()
        for fm_ in _FRACTION_RE.finditer(fig_text):
            num, den = fm_.group("num"), fm_.group("den")
            cap_dens = cap_fracs.get(num)
            if cap_dens and den not in cap_dens and (num, den) not in seen_pairs:
                seen_pairs.add((num, den))
                warns.append(
                    f"`{basename}` figure text says `{num}/{den}` but the caption "
                    f"says `{num}/{sorted(cap_dens)[0]}` — update the stale figure"
                )
    return warns


def check_figure_text_vs_body_tokens(body: str) -> CheckResult:
    """Check 24 (WARN): figure-embedded text must not carry strings the body
    prose softened away, or chance/N values that disagree with the body's
    figure caption.

    A round-1 numeric / overclaim fix is routinely applied to the body prose
    but MISSED in the figure-generation script's hardcoded title / annotation
    strings, so the regenerated figure silently disagrees with the body
    (e.g. #667: the body caption was corrected to a ``1/29`` chance level
    while the figure title still read ``1/30``; and "geometrically real" was
    removed from prose but left in a figure annotation). The mechanical
    verifier inspects body prose + figure-URL SHA pinning but never the figure
    text, so only the multimodal interpretation-critic catches this — pushing
    the fix to a later review round. This check reads each referenced figure's
    sibling ``.meta.json`` (from the git tree at the URL's commit sha) and
    flags two signals:

    (a) **chance/N disagreement** — a ``<a>/<b>`` fraction in the figure text
        whose same-numerator counterpart in the body's figure caption uses a
        DIFFERENT denominator (the ``1/30`` vs ``1/29`` case). Conservative:
        only same-numerator, different-denominator pairs are flagged, so an
        unrelated fraction in the figure (a different axis) is ignored.
    (b) **softened token** — any string from the configured stale-token list
        (``STALE_FIGURE_TOKENS`` plus an optional ``~/.eps-stale-tokens.json``,
        see ``_load_stale_figure_tokens``) appearing in the figure text.

    WARN, never FAIL — a soft mechanical aid (the multimodal critic owns the
    substantive figure-vs-body read). NO-OP PASS when: the repo cannot be
    resolved (offline / `--body-stdin`), no figure URLs resolve to a same-repo
    sha-pinned raw-GitHub URL, or no sidecar carries scannable text. Reads at
    most one ``git show`` per unique figure URL, all fail-soft, so the check
    adds negligible latency on a normal body (no figure sidecars with text →
    one cheap git probe per figure, then skip).

    Sidecars written by the current ``savefig_paper`` additionally carry the
    figure's RENDERED text (titles / axis labels / legends / series names /
    annotations / tick labels) under a ``text`` key — its strings enter this
    check's scans automatically via ``_flatten_meta_strings`` (forward-only:
    older sidecars simply lack the key), so a stale fraction in the actual
    rendered TITLE is now catchable, not just one echoed in an ad-hoc
    ``description``.
    """
    label = "figure text vs body prose (figure-text staleness)"
    section = _figure_scan_section(body)
    text = section_text(body, section)
    if text is None:
        return CheckResult(label, True, f"no `## {section}` section to scan")
    rlines = text.splitlines()
    # Collect (figure-URL, caption) pairs in document order.
    fig_caps: list[tuple[str, str]] = []
    for idx, line in enumerate(rlines):
        for m in _IMAGE_RE.finditer(line):
            url = m.group(1).strip()
            url = url.split(None, 1)[0] if url else url
            if not url:
                continue
            fig_caps.append((url, _figure_caption_after(rlines, idx)))
    if not fig_caps:
        return CheckResult(label, True, "no inline figures to scan")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(label, True, "skipped — repo root unresolved (offline / stdin)")
    stale_tokens = _load_stale_figure_tokens()
    meta_cache: dict[str, str | None] = {}
    warns: list[str] = []
    scanned = 0
    for url, caption in fig_caps:
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if m is None or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue  # only same-repo sha-pinned figures resolve from git
        if url not in meta_cache:
            meta_cache[url] = _read_figure_meta_text(repo, m.group("sha"), m.group("path"))
        fig_text = meta_cache[url]
        if not fig_text:
            continue  # no sidecar / no scannable text — skip this figure
        scanned += 1
        basename = m.group("path").rsplit("/", 1)[-1]
        warns.extend(_figure_text_warnings(fig_text, caption, basename, stale_tokens))
    if warns:
        preview = "; ".join(warns[:3]) + (" …" if len(warns) > 3 else "")
        return CheckResult(
            label,
            True,
            f"{len(warns)} figure-text mismatch(es) across {scanned} scanned figure(s): {preview}",
            is_warn=True,
        )
    if scanned == 0:
        return CheckResult(
            label, True, "no same-repo figure sidecar with scannable text — nothing to compare"
        )
    return CheckResult(label, True, f"{scanned} figure sidecar(s) consistent with body prose")


# ─── Check 26 helpers: figure panel/series prose vs figure sidecar ─────────
#
# Sibling of check 24's `_read_figure_meta_text`, but returns the PARSED dict
# (not flattened text) so check 26 can read the per-point `_kind` / `_group`
# fields. Same `git show <sha>:<meta_path>` envelope, same fail-soft contract.
def _read_figure_meta_json(repo: Path, sha: str, fig_path: str) -> dict | None:
    """Return the PARSED sibling ``.meta.json`` of ``fig_path`` (extension
    swapped to ``.meta.json``) read out of the git tree at ``sha`` via
    ``git show``, or None when there is no sidecar at that sha / the sha is
    unresolvable / the JSON does not parse / it is not a dict.

    Sibling of ``_read_figure_meta_text`` (check 24), which flattens to text
    and so cannot expose the per-point ``_kind`` / ``_group`` fields check 26
    needs. FAIL-SOFT throughout (subprocess / decode / JSON error → None).
    """
    base, _, ext = fig_path.rpartition(".")
    meta_path = (base if ext else fig_path) + ".meta.json"
    try:
        proc = subprocess.run(
            ["git", "show", f"{sha}:{meta_path}"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None  # no sidecar at that sha, or sha unresolvable
    try:
        meta = json.loads(proc.stdout)
    except (ValueError, json.JSONDecodeError):
        return None
    return meta if isinstance(meta, dict) else None


def _sidecar_kind_group_aggregate(meta: dict) -> tuple[Counter, set] | None:
    """From a parsed sidecar, return (Counter of ``_kind`` values across all
    points, set of unique ``_group`` values), or None when the sidecar carries
    no recognizable point list (so the check skips — it only fires on sidecars
    whose shape it understands).

    NOTE: ``_group`` is a per-SERIES index, NOT a per-panel index (an A7
    four-panel figure has 22 ``_group`` values). The returned group set is kept
    only for diagnostics; check 26 makes NO claim about panel count from it
    (see ``_panel_drift_failures``).

    Reads ``points`` (the canonical key) OR ``rows`` (legacy). A point with no
    ``_kind`` contributes nothing to the kind Counter; a point with no
    ``_group`` contributes nothing to the group set.
    """
    pts = meta.get("points")
    if not isinstance(pts, list):
        pts = meta.get("rows")
    if not isinstance(pts, list) or not pts:
        return None
    kinds: Counter = Counter()
    groups: set = set()
    for p in pts:
        if not isinstance(p, dict):
            continue
        k = p.get("_kind")
        if isinstance(k, str) and k.strip():
            kinds[k.strip().lower()] += 1
        g = p.get("_group")
        if g is not None:
            groups.add(g)
    if not kinds and not groups:
        return None  # no structural signal — nothing to compare
    return kinds, groups


# Plot-element words that map to a sidecar ``_kind``. A prose claim naming one
# of these as a distinct panel/series is checkable against the kind Counter.
_PROSE_KIND_RE = {
    "scatter": re.compile(r"\bscatter(?:\s*plot|s)?\b", re.IGNORECASE),
    "line": re.compile(r"\bline(?:\s*plot|s)?\b|\btrajector(?:y|ies)\b", re.IGNORECASE),
    "bar": re.compile(r"\bbar(?:\s*chart|s)?\b", re.IGNORECASE),
}
# A panel/series STRUCTURAL claim — prose asserting a specific panel position
# OR a per-unit dot/point overlay. A kind word is checkable ONLY when it
# co-occurs with one of these (a bare "the bars show ..." is NOT a claim).
_PROSE_PANEL_POS_RE = re.compile(r"\b(?:left|right|top|bottom|middle)\s+panel\b", re.IGNORECASE)
_PROSE_OVERLAY_RE = re.compile(
    r"\b(?:per[-\s]?\w+\s+dots?|dots?\s+overlaid|dot\s+overlay|points?\s+overlaid)\b",
    re.IGNORECASE,
)


def _panel_prose_claims(prose: str) -> dict:
    """Return the high-confidence structural claims a figure's what-is-plotted
    + caption prose makes::

        {"kind_in_panel": {<kind>, ...},   # a kind named as a PANEL/series
         "overlay": <bool>}                # a per-unit dot/point overlay claim

    Only claims checkable against the sidecar ``_kind`` aggregate are returned;
    a bare "the bars show ..." with no panel/overlay wording yields an empty
    claim set (the check then PASSes — no over-fire on simple charts). NO
    panel-count claim is produced (the sidecar has no panel-index field; see
    ``_sidecar_kind_group_aggregate``).
    """
    claims: dict = {"kind_in_panel": set(), "overlay": False}
    has_panel_pos = bool(_PROSE_PANEL_POS_RE.search(prose))
    has_overlay = bool(_PROSE_OVERLAY_RE.search(prose))
    if has_panel_pos or has_overlay:
        for kind, rx in _PROSE_KIND_RE.items():
            if rx.search(prose):
                claims["kind_in_panel"].add(kind)
    claims["overlay"] = has_overlay
    return claims


def _panel_drift_failures(claims: dict, kinds: Counter, basename: str) -> list[str]:
    """Return the check-26 FAIL messages for ONE figure's prose ``claims``
    against its sidecar ``kinds`` Counter (the ``_kind`` aggregate). Empty list
    = no drift.

    Two FAIL conditions:
    (1) prose names a plot KIND as a panel/series the sidecar's ``_kind`` count
        lacks entirely (kind count 0).
    (3) prose claims a per-unit dot/point OVERLAY but the sidecar has zero
        scatter points (nothing overlaid) — fires whenever ``scatter == 0``,
        REGARDLESS of ``_group`` cardinality.

    There is intentionally NO panel-COUNT FAIL: ``_group`` is a per-series
    index, not a panel index (a 4-panel A7 figure has 22 ``_group`` values), so
    the sidecar carries no panel-count signal to compare against.
    """
    fails: list[str] = []
    for kind in sorted(claims["kind_in_panel"]):
        if kinds.get(kind, 0) == 0:
            fails.append(
                f"`{basename}`: body prose claims a `{kind}` panel/series but the "
                f"figure sidecar has zero `{kind}` points (kinds present: "
                f"{dict(kinds) or 'none'}) — regenerate the figure or fix the prose"
            )
    if claims["overlay"] and kinds.get("scatter", 0) == 0:
        fails.append(
            f"`{basename}`: body prose claims a per-unit dot/point overlay but the "
            f"figure sidecar has zero scatter points (kinds present: "
            f"{dict(kinds) or 'none'}) — regenerate the figure or fix the prose"
        )
    return fails


def _enclosing_h3_prose_window(rlines: list[str], img_idx: int) -> str | None:
    """Return the what-is-plotted prose window for the figure at
    ``rlines[img_idx]``: the text from the enclosing ``### <result>`` H3 forward
    to the figure PLUS the blockquote caption immediately after — or None when
    there is no ``### `` H3 before the figure (no reliably-scoped window, so the
    caller SKIPS that figure rather than leaking an adjacent result's claim).
    """
    boundary = None
    for j in range(img_idx - 1, -1, -1):
        if rlines[j].startswith("### "):
            boundary = j + 1
            break
    if boundary is None:
        return None
    before = "\n".join(rlines[boundary:img_idx])
    caption = _figure_caption_after(rlines, img_idx)
    return f"{before}\n{caption}"


def _panel_drift_for_one_figure(
    repo: Path, m: re.Match, prose: str, json_cache: dict
) -> tuple[list[str], bool]:
    """Resolve ONE figure's sidecar (strictly by URL stem at the cited sha) and
    compare it to ``prose``'s panel/series claims. Returns ``(fail_msgs,
    scanned)`` where ``scanned`` is True only when a structural sidecar was
    actually compared.

    ``m`` is the ``_RAW_GITHUB_FIGURE_RE`` match for the figure URL; ``json_cache``
    is the per-check per-URL parsed-sidecar cache (mutated in place).
    """
    claims = _panel_prose_claims(prose)
    if not claims["kind_in_panel"] and not claims["overlay"]:
        return [], False  # no structural claim → nothing to check (never over-fire)
    url = m.group(0)
    if url not in json_cache:
        json_cache[url] = _read_figure_meta_json(repo, m.group("sha"), m.group("path"))
    meta = json_cache[url]
    basename = m.group("path").rsplit("/", 1)[-1]
    if meta is None:
        # Strict by-stem resolve FAILED at the cited sha. FAIL loud — BUT only
        # when the PNG itself resolves at the sha (so the ABSENT thing is
        # specifically the sidecar, not the whole sha; else defer to check 22).
        # The gate is an explicit status comparison: the tuple is always truthy,
        # so a truthiness check would never `continue`.
        png_status, _detail = _git_object_exists(repo, m.group("sha"), m.group("path"))
        if png_status != "pass":
            return [], False  # PNG missing OR sha unresolvable — check 22 owns it
        return [
            f"`{basename}`: body prose makes a panel/series claim but the sibling "
            f"`{basename.rsplit('.', 1)[0]}.meta.json` does not resolve at the cited "
            f"sha `{m.group('sha')[:8]}` — commit the sidecar at that sha (no silent "
            f"fallback to a different sidecar)"
        ], False
    agg = _sidecar_kind_group_aggregate(meta)
    if agg is None:
        return [], False  # sidecar carries no _kind/_group structure to compare
    kinds, _groups = agg
    return _panel_drift_failures(claims, kinds, basename), True


def check_figure_panel_prose_vs_sidecar(body: str) -> CheckResult:
    """Check 26 (FAIL): a figure's what-is-plotted prose must not claim a plot
    kind in a named panel position, or a per-unit dot/point overlay, that the
    figure's ``.meta.json`` sidecar — read strictly by URL stem from the git
    tree at the URL's commit sha — provably lacks (the sidecar's ``_kind`` count
    of that element is 0).

    The prose window is scoped to the figure's enclosing ``### <result>`` H3 (so
    a claim from one result never leaks into the next): the what-is-plotted text
    from that H3 forward to the figure, plus the blockquote caption immediately
    after. A figure with no preceding ``### `` H3 is SKIPPED (no reliably-scoped
    window). NO panel-count claim is made (``_group`` is a per-series index, not
    a panel index).

    When the sibling ``<basename>.meta.json`` does not resolve at the cited sha
    BUT the figure PNG itself does (``_git_object_exists`` returns ``'pass'``),
    the check FAILs loud — that silent fallback to a different sidecar is the
    very failure mode this check exists to catch (incident #683 r1). When the
    PNG itself does not resolve (sha unknown / PNG absent), the check defers to
    check 22 (no double-FAIL). NO-OP PASS when: no
    ``## Results``/``## Findings`` section, no inline figures, the repo cannot
    be resolved (offline / ``--body-stdin``), or no figure carries a panel/series
    prose claim to compare. FAIL, never WARN (distinct from check 24).
    """
    label = "figure panel prose vs figure sidecar (panel/series drift)"
    section = _figure_scan_section(body)
    text = section_text(body, section)
    if text is None:
        return CheckResult(label, True, f"no `## {section}` section to scan")
    rlines = text.splitlines()
    fig_at: list[tuple[str, int]] = []
    for idx, line in enumerate(rlines):
        for m in _IMAGE_RE.finditer(line):
            url = m.group(1).strip()
            url = url.split(None, 1)[0] if url else url
            if url:
                fig_at.append((url, idx))
    if not fig_at:
        return CheckResult(label, True, "no inline figures to scan")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(label, True, "skipped — repo root unresolved (offline / stdin)")
    fails: list[str] = []
    scanned = 0
    json_cache: dict[str, dict | None] = {}
    for url, img_idx in fig_at:
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if m is None or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue  # only same-repo sha-pinned figures resolve from git
        prose = _enclosing_h3_prose_window(rlines, img_idx)
        if prose is None:
            continue  # no per-result H3 before this figure — no scoped window
        fig_fails, did_scan = _panel_drift_for_one_figure(repo, m, prose, json_cache)
        fails.extend(fig_fails)
        if did_scan:
            scanned += 1
    if fails:
        preview = "; ".join(fails[:3]) + (" …" if len(fails) > 3 else "")
        return CheckResult(label, False, f"{len(fails)} panel/series drift issue(s): {preview}")
    if scanned == 0:
        return CheckResult(label, True, "no panel/series prose claims to compare against a sidecar")
    return CheckResult(
        label, True, f"{scanned} figure sidecar(s) consistent with panel/series prose"
    )


# ─── Check 28: opaque config-code tokens in figure text ────────────────────
#
# Sibling of checks 24/26 (same figure iteration + `_read_figure_meta_json`
# sidecar read), but the flagged property is INTRINSIC to the sidecar's own
# strings — no body comparison: rendered figure text must use plain-English
# condition names, never internal config shorthand (`ctx_blk_max@L12`).

# (a) `@L<digits>` layer pins, with any attached snake stem captured whole
#     (`ctx_blk_max@L12`); bare `@L12` also matches (no leading \b — `@` is a
#     non-word char, so a \b there would fail on a space-preceded bare pin).
_LAYER_PIN_RE = re.compile(r"\w*@L\d+\b")

# (b) snake_case token, >=2 segments, starting with a letter. The classifier
#     below flags only >=3-segment or digit-bearing matches (2-segment
#     all-alpha metric / persona names like `log_prob` are legitimate labels).
_SNAKE_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")

# Path/URI-SHAPED string: no internal whitespace and at least one path
# separator — a file path or URL, which is provenance, not rendered text.
# Deliberately NOT a whole-string any-slash skip: a slash-separated rendered
# label like `ctx_blk_max / ans_uhdr_max` contains whitespace, so it IS
# scanned (a whole-string any-slash form would false-clean slash-separated
# labels inside the incident class).
_PATH_SHAPED_RE = re.compile(r"^\S*[/\\]\S*$")


def _opaque_code_tokens(text: str) -> list[str]:
    """Return the opaque config-code tokens in ONE sidecar string: `@L<d>`
    layer pins, and snake_case tokens that are >=3 segments OR carry any
    digit (`ctx_blk_max`, `sw_eng_C1`, `BS_E0`, `cond_4`); 2-segment
    all-alpha tokens (`log_prob`, `judge_rate`, `helpful_assistant`) are
    allowed. PATH-SHAPED strings (whitespace-free with a path separator —
    file paths, URLs) are exempt from BOTH token scans (pins AND snakes);
    strings that merely CONTAIN a slash (e.g. a slash-separated rendered
    label) are still scanned, with individual path-shaped whitespace-split
    words skipped for both token classes. De-duped case-insensitively,
    order kept.
    """
    hits: list[str] = []
    if not _PATH_SHAPED_RE.match(text.strip()):
        words = text.split()

        def _only_in_path_words(tok: str) -> bool:
            """True iff every whitespace-split word containing `tok` is
            path-shaped ("see figures/x_1/y.png") — provenance, not
            rendered text, so the token is skipped."""
            ws_words = [w for w in words if tok in w]
            return bool(ws_words) and all(_PATH_SHAPED_RE.match(w) for w in ws_words)

        for m in _LAYER_PIN_RE.finditer(text):
            tok = m.group(0)
            # Same containing-word path skip snake tokens get (round-2
            # concern layer-pin-path-exemption): a pin inside a path word
            # ("figures/issue_920/ctx_blk_max@L12.png") is provenance.
            if _only_in_path_words(tok):
                continue
            hits.append(tok)
        for m in _SNAKE_TOKEN_RE.finditer(text):
            tok = m.group(0)
            if _only_in_path_words(tok):
                continue
            if tok.count("_") >= 2 or any(ch.isdigit() for ch in tok):
                hits.append(tok)
    seen: set[str] = set()
    out: list[str] = []
    for t in hits:
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _iter_meta_label_values(obj: object) -> list[str]:
    """Collect the rendered-text-bearing strings of a parsed sidecar for
    check 28: string VALUES (provenance-keyed subtrees pruned via
    ``_META_PROVENANCE_KEYS``) plus dict KEYS containing internal whitespace
    (axis-label-keyed data rows, e.g. ``{"1/30 chance accuracy": 0.41}``).
    Identifier-shaped keys (``_kind``, ``cell_slugs``, translation-map slug
    keys) are structural provenance and are NOT collected — the deliberate
    divergence from check 24's ``_flatten_meta_strings``, which collects all
    non-provenance keys.
    """
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.casefold() in _META_PROVENANCE_KEYS:
                continue
            if isinstance(k, str) and " " in k.strip():
                out.append(k)
            out.extend(_iter_meta_label_values(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_meta_label_values(item))
    elif isinstance(obj, str) and obj.strip():
        out.append(obj)
    return out


def check_figure_label_codes(body: str) -> CheckResult:
    """Check 28 (WARN): rendered figure text (sidecar ``.meta.json`` values)
    must not carry opaque config-code tokens — ``@L<digits>`` layer pins or
    regime-code slugs (``ctx_blk_max``, ``sw_eng_C1``). Plain-English
    condition names are the rule end to end (memory
    feedback_no_opaque_condition_codes, SPEC statistical-framing bullet);
    config slugs belong in the Repro config row / provenance keys. Incident
    #920: ``winning_cell_scatter.png`` reached the 9a-bis gate titled
    ``ctx_blk_max@L12 x ans_uhdr_max@L12`` after three review passes each
    deferred it as a cosmetic nit.

    Coverage = sidecar-CARRIED strings only: string values (provenance
    subtrees pruned) plus whitespace-bearing dict keys. The current
    ``savefig_paper`` serializes the figure's RENDERED text (suptitle, axes
    titles incl. ``loc="left"``, axis labels, legend entries, series names,
    annotations, tick labels) under ``meta["text"]``, whose string VALUES
    enter this walk automatically — so a slug-bearing figure TITLE / legend /
    series name in NEW sidecars IS scanned (the #1092 defect-(a) class); the
    ``text`` block's own structural key names (``suptitle``,
    ``legend_labels``, …) are whitespace-free identifier keys and are never
    collected. Known residuals, accepted by design: (i) rendered-text
    blindness persists for sidecars written BEFORE the ``meta["text"]``
    capture landed (forward-only) AND for figures saved via plain
    ``fig.savefig`` scripts that never get a sidecar at all — for those,
    PNG-pixel text stays the multimodal critics' substantive read; (ii) a
    bare slug used as a whitespace-free column KEY is unscanned (tick labels
    now arrive as scanned VALUES in new sidecars, narrowing this residual to
    key names); (iii) a slug or ``@L`` pin inside a path-shaped word (or a
    whole path-shaped string) is exempt — the path exemption covers BOTH
    token classes. WARN,
    never FAIL; fail-soft on missing / unparsable sidecars (the check-24
    convention, NOT check 26's loud missing-sidecar FAIL); NO-OP PASS
    offline / no figures / no scannable same-repo sidecar.
    """
    label = "figure text opaque config codes (slug / @L-pin tokens)"
    section = _figure_scan_section(body)
    text = section_text(body, section)
    if text is None:
        return CheckResult(label, True, f"no `## {section}` section to scan")
    urls: list[str] = []
    for line in text.splitlines():
        for m in _IMAGE_RE.finditer(line):
            url = m.group(1).strip()
            url = url.split(None, 1)[0] if url else url
            if url:
                urls.append(url)
    if not urls:
        return CheckResult(label, True, "no inline figures to scan")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(label, True, "skipped — repo root unresolved (offline / stdin)")
    meta_cache: dict[str, dict | None] = {}
    warns: list[str] = []
    scanned = 0
    for url in dict.fromkeys(urls):
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if m is None or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue  # only same-repo sha-pinned figures resolve from git
        if url not in meta_cache:
            meta_cache[url] = _read_figure_meta_json(repo, m.group("sha"), m.group("path"))
        meta = meta_cache[url]
        if meta is None:
            continue  # no sidecar / unparsable — fail-soft skip (check-24 convention)
        scanned += 1
        toks: list[str] = []
        for s in _iter_meta_label_values(meta):
            toks.extend(_opaque_code_tokens(s))
        toks = list(dict.fromkeys(toks))
        if toks:
            basename = m.group("path").rsplit("/", 1)[-1]
            preview = ", ".join(f"`{t}`" for t in toks[:4]) + (" …" if len(toks) > 4 else "")
            warns.append(
                f"`{basename}` figure text carries opaque config-code token(s) {preview} — "
                "use plain-English condition names in rendered figure text (slugs belong in "
                "the Repro config row / provenance keys); regenerate, or acknowledge in body"
            )
    if warns:
        head = "; ".join(warns[:3]) + (" …" if len(warns) > 3 else "")
        return CheckResult(
            label,
            True,
            f"{len(warns)} figure(s) with opaque code tokens of {scanned} scanned: {head}",
            is_warn=True,
        )
    if scanned == 0:
        return CheckResult(
            label, True, "no same-repo figure sidecar with scannable text — nothing to scan"
        )
    return CheckResult(label, True, f"{scanned} figure sidecar(s) free of opaque config codes")


# ─── Check 33: bolded what-is-plotted numerics vs sidecar plotted values ───
#
# Fourth sibling of checks 24/26/28 (same figure iteration, same
# `_read_figure_meta_json` sidecar read, same fail-soft skip conventions).
# Check 24 compares the sidecar's RENDERED TEXT, check 26 its STRUCTURE
# (`_kind` aggregate); neither reads a single plotted NUMBER — so a
# figure/prose numeric divergence (#825 r1: prose+caption cited transfer
# fractions 0.057/0.109 while the pinned figure plotted 0.231 / -4.53-clipped)
# passes both and survives to the multimodal LM-critic layer. Check 33 closes
# that gap: every BOLDED DECIMAL in a figure's what-is-plotted prose window
# must appear among the sidecar's plotted values, under a leniency stack
# (rounding / sign / percent / sci-notation) that only ever REDUCES warns.

# A markdown bold span (`**…**`); single `*` inside the span is allowed.
_BOLD_SPAN_RE = re.compile(r"\*\*((?:[^*]|\*(?!\*))+)\*\*")

# A decimal REQUIRING a decimal point; optional ASCII/unicode sign; optional
# e-notation; optional `%` marker. The lookbehind blocks word/dot continuation
# ("v1.2", "3.1.4"); the lookahead blocks trailing word/hyphen/dot
# continuation ("2.5-7B") — an excluded token merely SHRINKS the scanned set
# (leniency-safe). Integers (no decimal point) are never matched: counts /
# n's / layer ids are the noisiest class. The unicode minus U+2212 (rendered
# prose uses it) is accepted as a sign and normalized to ASCII `-` before
# `float()` (`_bold_prose_decimals`); the `_APOS` convention — the escape,
# never the literal char, in code strings.
_UNICODE_MINUS = "\u2212"
_BOLD_DECIMAL_RE = re.compile(
    r"(?<![\w.])([-+" + _UNICODE_MINUS + r"]?\d+\.\d+(?:[eE][-+]?\d+)?)(\s*%)?(?![\w.-])"
)

# Per-figure opt-out phrase for check 33 — a render-invisible HTML comment,
# detected anywhere in the figure's scanned window (the beat-1 prose before
# the image OR its blockquote caption).
_PROSE_NUMERICS_OPTOUT = "<!-- prose-numerics: derived -->"


def _beat1_prose_window(rlines: list[str], img_idx: int) -> str | None:
    """The what-is-plotted prose OWNED by the figure at ``rlines[img_idx]``:
    text from the NEAREST preceding boundary — the enclosing ``### `` H3, or
    the end of the PREVIOUS inline figure's blockquote caption, whichever is
    nearer — forward to the image line, plus this figure's own blockquote
    caption (``_figure_caption_after``). None when no ``### `` H3 precedes the
    figure at all (no reliably-scoped window; the caller skips, mirroring
    check 26).

    Narrowed sibling of check 26's ``_enclosing_h3_prose_window`` (which is
    NOT modified): with the full H3→figure window, a multi-figure
    ``### <result>`` H3 bleeds EARLIER figures' beat-1 prose into LATER
    figures' windows (#778's ``bands_monitoring_*`` false-WARNs at plan-time
    calibration); bounding at the previous figure's caption end removes that
    class. The region between a previous figure's caption and this image can
    still carry the previous figure's beat-3 interpretation prose — the
    check's prior-figure value suppression + the opt-out phrase contain that
    residual (see ``check_figure_prose_numerics_vs_sidecar``).
    """
    boundary = None
    for j in range(img_idx - 1, -1, -1):
        if rlines[j].startswith("### "):
            boundary = j + 1
            break
        if _IMAGE_RE.search(rlines[j]):
            # Skip the previous figure's caption blockquote (+ blank lines).
            k = j + 1
            while k < img_idx and (rlines[k].strip() == "" or rlines[k].lstrip().startswith(">")):
                k += 1
            boundary = k
            break
    if boundary is None:
        return None
    # An H3 must exist somewhere above (else this figure sits outside any
    # `### <result>` block): when the scan hit a previous IMAGE first, an H3
    # is still required above it.
    if not any(rlines[j].startswith("### ") for j in range(img_idx)):
        return None
    before = "\n".join(rlines[boundary:img_idx])
    caption = _figure_caption_after(rlines, img_idx)
    return f"{before}\n{caption}"


def _bold_prose_decimals(window: str) -> list[tuple[str, float, int, bool]]:
    """Return ``(raw_text, value, printed_decimal_places, is_percent)`` per
    decimal found inside a BOLD span of ``window``. The unicode minus U+2212
    is normalized to ASCII ``-``; a sci-notation token gets the sentinel
    ``dec == -1`` (routes to the relative-tolerance branch of
    ``_prose_value_matches``). Integers (no decimal point) and word-attached
    tokens ("v1.2", "2.5-7B") are DELIBERATELY not scanned — see
    ``_BOLD_DECIMAL_RE``.
    """
    out: list[tuple[str, float, int, bool]] = []
    for bm in _BOLD_SPAN_RE.finditer(window):
        for nm in _BOLD_DECIMAL_RE.finditer(bm.group(1)):
            txt = nm.group(1).replace(_UNICODE_MINUS, "-")
            try:
                v = float(txt)
            except ValueError:  # pragma: no cover — regex guarantees a float
                continue
            dec = -1 if "e" in txt.lower() else len(txt.split(".")[1])
            raw = nm.group(1) + ("%" if nm.group(2) else "")
            out.append((raw, v, dec, bool(nm.group(2))))
    return out


def _sidecar_plotted_values(meta: dict) -> list[tuple[float, bool]]:
    """All finite numeric leaf values from the sidecar's ``points``/``rows``
    point rows as ``(value, is_bar_x)`` entries, excluding the ``_group``
    series index. Covers every ``_kind`` uniformly (bar heights, scatter/
    line/errorbar x+y, ``error`` magnitudes) because the writer
    (``paper_plots._build_sidecar_data``) stores every plotted quantity as a
    numeric row value under an axis-label-derived key; strings (category /
    label / series / ``_kind``) and JSON nulls contribute nothing.

    ``is_bar_x`` is True ONLY for the entry occupying the FIRST non-meta key
    of a ``_kind == "bar"`` row read from the modern ``points`` key — the
    grouped-bar layout x-POSITION slot (``-0.2`` / ``0.8``-style dodge
    offsets; the writer's ``_extract_bars`` inserts the x/category slot
    first, falling back to the numeric patch center when tick labels miss).
    Identity is per-ENTRY, not per-value: a bar HEIGHT that happens to EQUAL
    an x-position keeps its own untagged entry and stays variant-eligible
    (the r1 value-set exclusion dropped every equal value — a false-WARN
    channel on common grouped-bar offsets). Legacy ``rows`` sidecars are
    NEVER bar-x tagged (fail-open toward leniency — the legacy writer's key
    order is unverified).

    Returns ``[]`` when the sidecar carries no point list — the caller
    skips. A TRUNCATED sidecar (``meta['data_truncated']`` — the writer's
    top-level flag — or a legacy top-level ``truncated``) also returns
    ``[]``: a matching value may sit past the ``_MAX_SIDECAR_ROWS`` cap, so
    absence-of-match is unsound there. Top-level provenance keys (``commit``
    / ``created`` / ``figsize`` / ``n_series`` / ``total_points``) never
    enter — they are not point-row values (no ``_META_PROVENANCE_KEYS`` walk
    needed).
    """
    if meta.get("data_truncated") or meta.get("truncated"):
        return []
    pts = meta.get("points")
    from_points = isinstance(pts, list)
    if not from_points:
        pts = meta.get("rows")
    if not isinstance(pts, list):
        return []
    vals: list[tuple[float, bool]] = []
    for p in pts:
        if not isinstance(p, dict):
            continue
        bar_row = from_points and p.get("_kind") == "bar"
        first_data_key = True
        for k, v in p.items():
            if k in ("_kind", "_group"):
                continue
            is_first, first_data_key = first_data_key, False
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if math.isfinite(v):
                vals.append((float(v), bar_row and is_first))
    return vals


def _prose_value_matches(
    value: float,
    dec: int,
    is_pct: bool,
    plotted: list[tuple[float, bool]],
) -> bool:
    """True when the prose decimal ``(value, dec, is_pct)`` matches ANY
    plotted entry under the check-33 leniency stack. Candidates: the value
    itself, plus percent variants — ``value/100`` always (an ``87.9%`` prose
    value vs a 0.879 fraction axis), and ``value*100`` only for an UNMARKED
    decimal (a 0.879 prose fraction vs an 87.9 percent axis). Per candidate:

    - printed-precision half-ulp: ``|p - c| <= 0.5·10^(-dec)·scale + 1e-12``
      (``scale`` = the candidate's x100 / /100 factor), i.e. ``p`` rounds to
      the prose value at its printed precision (prose ``0.23`` matches
      plotted 0.2312);
    - sign-insensitive twin: the same test on ``||p| - |c||`` (a "0.30 drop"
      vs plotted -0.30);
    - sci-notation branch (``dec == -1``): relative tolerance
      ``|p - c| <= 1e-3·|c|``.

    Variant candidates never match an entry TAGGED as a grouped-bar layout
    x-position (``is_bar_x`` per ``_sidecar_plotted_values``): a percent
    variant landing on a bar's layout offset is a coincidence, not evidence
    the prose value is plotted. The exclusion is per-ENTRY
    (identity-preserving): a bar HEIGHT whose value equals an x-position
    keeps its own untagged entry and stays variant-eligible. The direct
    candidate matches everything. Every clause is leniency-INCREASING
    (reduces WARNs), so each is FP-containment-safe.
    """
    candidates: list[tuple[float, float, bool]] = [(value, 1.0, False)]
    candidates.append((value / 100.0, 0.01, True))
    if not is_pct:
        candidates.append((value * 100.0, 100.0, True))
    for c, scale, is_variant in candidates:
        tol = 1e-3 * abs(c) if dec < 0 else 0.5 * 10.0 ** (-dec) * scale + 1e-12
        for p, p_is_bar_x in plotted:
            if is_variant and p_is_bar_x:
                continue
            if abs(p - c) <= tol or abs(abs(p) - abs(c)) <= tol:
                return True
    return False


def _prose_numerics_for_one_figure(
    repo: Path,
    m: re.Match,
    rlines: list[str],
    img_idx: int,
    json_cache: dict,
    h3_prior_vals: dict[int, list[tuple[float, bool]]],
) -> tuple[str | None, str]:
    """Process ONE same-repo figure for check 33 (the check-26
    ``_panel_drift_for_one_figure`` shape). Returns ``(warn_msg | None,
    status)`` with ``status`` in {"scanned", "opted-out", "skipped"};
    mutates ``json_cache`` (per-URL parsed-sidecar cache) and
    ``h3_prior_vals`` (per-H3 accumulator of earlier figures' plotted
    values, the cross-figure bleed-suppression pool) in place.

    Pool semantics (deliberate): EVERY earlier same-H3 figure with a
    readable, value-bearing, non-truncated sidecar contributes its
    ``(value, is_bar_x)`` entries to the pool — INCLUDING figures whose own
    window was opted out or carried no bolded decimals. The pool models what
    earlier figures PLOT (sidecar values), which is independent of their own
    prose-scan outcome: bleed prose after an opted-out / bold-less figure
    can legitimately re-quote that figure's plotted values, so excluding
    them would open a false-WARN channel against the check's precision-first
    posture. The pool is leniency-only (suppresses WARNs, never creates
    them). The current figure's own values enter the accumulator only AFTER
    ``prior_vals`` snapshots a COPY, so a figure can never bleed-suppress
    itself (r1 aliasing bug: ``get`` returned the live accumulator and
    ``extend`` mutated it before matching, letting figure 2+'s bar-x values
    bypass the variant exclusion through the pool's match path).
    """
    window = _beat1_prose_window(rlines, img_idx)
    if window is None:
        return None, "skipped"  # no per-result H3 above — no scoped window
    h3_idx = next((j for j in range(img_idx - 1, -1, -1) if rlines[j].startswith("### ")), -1)
    url = m.group(0)
    if url not in json_cache:
        json_cache[url] = _read_figure_meta_json(repo, m.group("sha"), m.group("path"))
    meta = json_cache[url]
    if meta is None:
        return None, "skipped"  # no sidecar at that sha — check-24 convention
    plotted = _sidecar_plotted_values(meta)
    if not plotted:
        return None, "skipped"  # value-less / truncated — absence-of-match unsound
    prior_vals = list(h3_prior_vals.get(h3_idx, []))  # COPY — never the live accumulator
    h3_prior_vals.setdefault(h3_idx, []).extend(plotted)
    if _PROSE_NUMERICS_OPTOUT in window:
        return None, "opted-out"
    bolds = _bold_prose_decimals(window)
    if not bolds:
        return None, "skipped"
    unmatched = [
        (raw, val)
        for (raw, val, dec, pct) in bolds
        if not _prose_value_matches(val, dec, pct, plotted)
        and not (prior_vals and _prose_value_matches(val, dec, pct, prior_vals))
    ]
    if not unmatched:
        return None, "scanned"
    basename = m.group("path").rsplit("/", 1)[-1]
    vals = ", ".join(f"`{raw}`" for raw, _v in unmatched[:4]) + (" …" if len(unmatched) > 4 else "")
    nearest = min((p for p, _bx in plotted), key=lambda p: abs(p - unmatched[0][1]))
    return (
        f"`{basename}`: bolded what-is-plotted value(s) {vals} not found among the "
        f"sidecar's {len(plotted)} plotted values (nearest: {nearest:.4g}) — regenerate "
        f"the figure or fix the prose; if the value is a derived quantity (delta / "
        f"ratio / CI bound), add `{_PROSE_NUMERICS_OPTOUT}` to the figure's "
        f"what-is-plotted prose or caption",
        "scanned",
    )


def check_figure_prose_numerics_vs_sidecar(body: str) -> CheckResult:
    """Check 33 (WARN): every BOLDED DECIMAL in a figure's what-is-plotted
    prose window must appear among the numeric values the figure's
    ``.meta.json`` sidecar (read at the URL's commit sha) records as plotted.

    The scanned window is the previous-figure-bounded beat-1 slice: text from
    the nearest boundary above the image — the enclosing ``### <result>`` H3
    or the previous figure's caption end — down to the image, plus this
    figure's blockquote caption (``_beat1_prose_window``). Firing is
    PER-NUMERIC: WARN when >=1 bolded decimal matches NO plotted value under
    the ``_prose_value_matches`` leniency stack (printed-precision rounding,
    sign-insensitivity, percent x100 / /100 variants — never against a
    grouped-bar layout x-position — and a sci-notation relative-tolerance
    branch). The task-body sketch's stricter none-match-any rule was
    calibrated OUT: on the motivating incident (#825 r1) 6 of 8 bolded values
    matched the r1-era sidecar, so none-match-any misses the exact bug;
    per-numeric catches it (unmatched = {0.057, 0.109}) at a measured 0%
    false-positive rate corpus-wide (plan §4).

    Cross-figure bleed containment: in a multi-figure H3 the region between a
    previous figure's caption and this image can carry the PREVIOUS figure's
    interpretation prose — a bolded decimal matching an EARLIER same-H3
    figure's plotted values is treated as that bleed and suppressed
    (leniency-safe). Every earlier same-H3 figure with a value-bearing
    sidecar feeds the pool regardless of its own scan outcome (opted-out /
    bold-less included — see ``_prose_numerics_for_one_figure``); the current
    figure's own values never enter its own pool, and pool entries keep
    their bar-x tags so percent variants never match an earlier figure's bar
    x-positions either. A derived quantity (delta / ratio / CI bound) matching
    NEITHER sidecar still WARNs — that is the documented FP class the
    per-figure opt-out exists for: the literal ``<!-- prose-numerics: derived
    -->`` ANYWHERE in the figure's scanned window (beat-1 prose or caption)
    skips that figure.

    Silent-skip conditions (check-24 convention, NOT check 26's loud
    missing-sidecar FAIL — the trigger here, any bolded decimal, is far
    broader than check 26's explicit structural claims): missing / unparsable
    sidecar, ``data_truncated`` / ``truncated`` sidecar, zero bolded decimals
    in the window, zero finite plotted values, no ``### `` H3 above the
    figure, non-same-repo / non-sha-pinned URL. NO-OP PASS offline /
    ``--body-stdin`` / no figure section / no inline figures. WARN never
    FAIL. Named recall sacrifices (precision over recall, the check-32
    posture): UNBOLDED caption numerics are not scanned (caption means /
    aggregates are a measured guaranteed-FP class), integers are not scanned,
    and the bleed suppression can mask a wrong beat-1 value that coincides
    with an earlier figure's plotted values. Incident #825 r1 (task #1107):
    prose+caption cited full-n transfer fractions 0.057/0.109 while the
    pinned figure plotted matched-ceiling fractions 0.231 / -4.53-clipped —
    checks 24 (rendered text) and 26 (structure) are blind to plotted-NUMBER
    drift by construction.
    """
    label = "figure prose numerics vs figure sidecar (plotted-value drift)"
    section = _figure_scan_section(body)
    text = section_text(body, section)
    if text is None:
        return CheckResult(label, True, f"no `## {section}` section to scan")
    rlines = text.splitlines()
    fig_at: list[tuple[str, int]] = []
    for idx, line in enumerate(rlines):
        for m in _IMAGE_RE.finditer(line):
            url = m.group(1).strip()
            url = url.split(None, 1)[0] if url else url
            if url:
                fig_at.append((url, idx))
    if not fig_at:
        return CheckResult(label, True, "no inline figures to scan")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(label, True, "skipped — repo root unresolved (offline / stdin)")
    warns: list[str] = []
    scanned = 0
    opted_out = 0
    json_cache: dict[str, dict | None] = {}
    # Per-H3 accumulator of EARLIER figures' (value, is_bar_x) plotted
    # entries (bleed suppression).
    h3_prior_vals: dict[int, list[tuple[float, bool]]] = {}
    for url, img_idx in fig_at:
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if m is None or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue  # only same-repo sha-pinned figures resolve from git
        warn, status = _prose_numerics_for_one_figure(
            repo, m, rlines, img_idx, json_cache, h3_prior_vals
        )
        if status == "scanned":
            scanned += 1
        elif status == "opted-out":
            opted_out += 1
        if warn:
            warns.append(warn)
    if warns:
        preview = "; ".join(warns[:3]) + (" …" if len(warns) > 3 else "")
        return CheckResult(
            label,
            True,
            f"{len(warns)} prose-vs-plotted mismatch(es) across {scanned} scanned "
            f"figure(s): {preview}",
            is_warn=True,
        )
    if scanned == 0:
        note = f" ({opted_out} opted out)" if opted_out else ""
        return CheckResult(
            label,
            True,
            f"no figure with bolded what-is-plotted decimals AND a value-bearing sidecar{note}",
        )
    return CheckResult(
        label,
        True,
        f"{scanned} figure(s): all bolded what-is-plotted decimals present among sidecar values",
    )


# ─── Check 34: beat-phrase series-structure claims vs sidecar rendered text ─
#
# Fifth sibling of checks 24/26/28/33 (same figure iteration, same
# `_read_figure_meta_json` sidecar read, same fail-soft skip conventions;
# reuses check 33's narrow `_beat1_prose_window`). Checks 24/28 scan the
# sidecar's rendered STRINGS and check 26 its `_kind` STRUCTURE against
# explicit panel/overlay claims — none compares a beat-1 SERIES-STRUCTURE
# claim ("shows both input arms", "one bar per re-fit item") against what the
# figure demonstrably renders, so #1092's defect (b) passed every mechanical
# figure check. Check 34 closes that gap for the two literal #1092 claim
# classes, contradiction-only, FORWARD-ONLY (fires only when the sidecar
# carries the `meta["text"]` rendered-text block the current `savefig_paper`
# writes — the writer-version marker).

# Class A: "both <up to 3 words> arms/series/conditions/models/lines/curves".
_BEAT_BOTH_RE = re.compile(
    r"\bboth\b(?:\s+[\w-]+){0,3}\s+(?:arms?|series|conditions?|models?|lines?|curves?)\b",
    re.IGNORECASE,
)
# Class B: "one bar|point|dot|marker|line|curve per <unit>".
_BEAT_ONE_PER_RE = re.compile(
    r"\bone\s+(bar|point|dot|marker|line|curve)s?\s+per\s+[\w-]+", re.IGNORECASE
)
_BEAT_WORD_TO_KIND = {
    "bar": "bar",
    "point": "scatter",
    "dot": "scatter",
    "marker": "scatter",
    "line": "line",
    "curve": "line",
}


def _beat_series_claims(prose: str) -> dict:
    """Parse the two registered check-34 claim classes out of a figure's
    beat-1 prose window::

        {"both":    [<matched phrase>, ...],           # Class A
         "one_per": [(<matched phrase>, <kind>), ...]} # Class B, kind mapped
                                                       # via _BEAT_WORD_TO_KIND

    Deliberately NARROW (the check's FP containment): only the two literal
    #1092 defect phrasings are registered — paraphrases ("each arm", "two
    models", "per-source bars") miss by design (a documented false-negative,
    not a bug). Class-B phrases are de-duplicated preserving order.
    """
    claims: dict = {"both": [], "one_per": []}
    for bm in _BEAT_BOTH_RE.finditer(prose):
        claims["both"].append(bm.group(0))
    seen: set[tuple[str, str]] = set()
    for om in _BEAT_ONE_PER_RE.finditer(prose):
        pair = (om.group(0).casefold(), _BEAT_WORD_TO_KIND[om.group(1).lower()])
        if pair not in seen:
            seen.add(pair)
            claims["one_per"].append((om.group(0), pair[1]))
    return claims


def _sidecar_kind_row_groups(meta: dict) -> tuple[Counter, dict[str, set], set] | None:
    """Per-kind point-row counts, per-kind distinct ``_group`` sets, and the
    all-rows ``_group`` set from a parsed sidecar's ``points``/``rows`` list —
    or None when the sidecar carries no recognizable point list.

    Same reading convention as ``_sidecar_kind_group_aggregate`` (``points``
    canonical, ``rows`` legacy; non-dict rows skipped), but check 34's line /
    scatter ARTIST counting needs the per-KIND group split the aggregate does
    not expose. ``_group`` is a per-ARTIST (per-series) index, NOT a per-panel
    index, and the writer emits it only on multi-artist figures
    (``_build_sidecar_data``: ``multi = len(artifacts) > 1``) — so a
    single-artist sidecar has NO groups anywhere.

    Returns None for a TRUNCATED sidecar (``data_truncated`` — the writer's
    top-level flag — or a legacy top-level ``truncated``):
    ``_build_sidecar_data`` truncates the concatenated per-artist rows
    HEAD-FIRST (``points[:_MAX_SIDECAR_ROWS]``), so a first artist with
    >= the cap drops every LATER artist/kind from the stored payload —
    per-kind / per-``_group`` counts over stored rows are then not figure
    truth (the ``_sidecar_plotted_values`` sibling convention, check 33).
    """
    if meta.get("data_truncated") or meta.get("truncated"):
        return None
    pts = meta.get("points")
    if not isinstance(pts, list):
        pts = meta.get("rows")
    if not isinstance(pts, list) or not pts:
        return None
    kinds: Counter = Counter()
    kind_groups: dict[str, set] = {}
    all_groups: set = set()
    for p in pts:
        if not isinstance(p, dict):
            continue
        k = p.get("_kind")
        kind = k.strip().lower() if isinstance(k, str) and k.strip() else None
        if kind is not None:
            kinds[kind] += 1
        g = p.get("_group")
        if g is not None:
            all_groups.add(g)
            if kind is not None:
                kind_groups.setdefault(kind, set()).add(g)
    if not kinds and not all_groups:
        return None
    return kinds, kind_groups, all_groups


def _line_artist_count(kinds: Counter, kind_groups: dict[str, set]) -> int | None:
    """Distinct line ARTISTS rendered, or None when the sidecar has no line
    rows (basis unavailable). Distinct ``_group`` values among line-kind rows;
    a single-artist sidecar carries no ``_group`` at all, so line rows with no
    groups count as ONE artist (one ``Line2D`` trace = one series — a line's
    point rows are VERTICES, so the raw row count is unsound for lines)."""
    if kinds.get("line", 0) <= 0:
        return None
    groups = kind_groups.get("line", set())
    return len(groups) if groups else 1


def _both_claim_bases(
    meta: dict, kinds: Counter, kind_groups: dict[str, set], all_groups: set
) -> dict[str, int]:
    """The Class-A ("both … arms") evidence bases AVAILABLE in ``meta``, as a
    ``{basis name: count}`` dict — each key present only when that basis has
    actual labels/rows (a zero-count kind is UNAVAILABLE, never a
    contradiction). See ``_beat_claim_warnings`` for the basis semantics."""
    if len(all_groups) == 1:
        # A 1-value group set means exactly ONE artist contributed rows in a
        # MULTI-artist figure (`_group` is only emitted when
        # `len(artifacts) > 1`, so a sibling artist yielded zero rows). That
        # is indistinguishable from the single-artist no-`_group` case: the
        # lone data-bearing artist (scatter especially) may encode >=2 arms
        # per-point, so the group evidence is treated as ABSENT, never as a
        # 1-count basis. (`_line_artist_count` is unaffected — its
        # no-groups fallback reads the same 1.)
        kind_groups = {}
        all_groups = set()
    bases: dict[str, int] = {}
    text_block = meta.get("text")
    tb = text_block if isinstance(text_block, dict) else {}
    series = tb.get("series")
    if isinstance(series, list) and series:
        bases["series labels"] = len(series)
    axes_block = tb.get("axes")
    legend_counts = []
    if isinstance(axes_block, list):
        for ax_d in axes_block:
            if not isinstance(ax_d, dict):
                continue
            labs = ax_d.get("legend_labels")
            if isinstance(labs, list) and labs:
                legend_counts.append(len(labs))
    if legend_counts:
        bases["legend entries"] = max(legend_counts)
    if kinds.get("bar", 0) > 0:
        bases["bar rows"] = kinds["bar"]
    line_n = _line_artist_count(kinds, kind_groups)
    if line_n is not None:
        bases["line artists"] = line_n
    scatter_groups = kind_groups.get("scatter", set())
    if scatter_groups:
        bases["scatter artists"] = len(scatter_groups)
    if all_groups:
        bases["artist groups"] = len(all_groups)
    return bases


def _beat_claim_warnings(claims: dict, meta: dict, basename: str) -> list[str]:
    """Return the check-34 WARN messages for ONE figure's parsed ``claims``
    against its parsed sidecar ``meta``. Empty list = no demonstrable
    contradiction (which includes "no evidence basis available" — absence of
    evidence never fires).

    Class A ("both … arms") ⇒ claimed >=2 rendered elements. Evidence bases,
    each AVAILABLE only when it has actual labels/rows, all max'd:

    - ``len(text["series"])`` — fig-GLOBAL legend-eligible artist labels
      (deliberately conservative: a multi-panel figure with one series per
      panel and >=2 distinct labels satisfies "both arms");
    - ``max(len(ax["legend_labels"]))`` over axes;
    - bar-kind point-ROW count (one bar row per bar — NEVER ``n_series``,
      which counts artist GROUPS: a two-arm bar pair lives in ONE
      ``BarContainer`` and would false-fire);
    - distinct line ARTISTS (``_line_artist_count``);
    - distinct scatter ARTISTS (distinct ``_group`` among scatter rows) — NO
      single-artist fallback, unlike lines: one scatter artist can encode >=2
      arms via per-point colors the extractor cannot see, so a lone unlabeled
      scatter is never a demonstrable contradiction (it yields NO basis and
      the claim is skipped);
    - total distinct ``_group`` across ALL rows (leniency-only: a mixed
      line+scatter two-artist figure satisfies "both arms" across kinds).
      A 1-value group set (exactly one artist contributed rows — a rowless
      sibling in a multi-artist figure) is treated as ABSENT group evidence
      (``_both_claim_bases``), the same skip as the lone unlabeled scatter.

    WARN only when >=1 basis is available AND every available basis reads
    <=1. Class B ("one bar per X") ⇒ claimed >=2 elements of the mapped kind;
    basis per kind: bar/scatter → point-row count of that ``_kind``;
    line/curve → distinct line ARTISTS (vertex rows are unsound). Requires a
    points payload (no payload → skip); WARN when the basis reads <=1 (kind
    absent entirely, or a single aggregate element where the prose claims
    per-unit multiplicity — the #1092 "one bar per re-fit item" degenerate
    class). TRUNCATED sidecars (``data_truncated`` / legacy ``truncated``)
    carry NO points-derived basis: ``_build_sidecar_data`` truncates the
    concatenated rows HEAD-FIRST, so a first artist with >= the
    ``_MAX_SIDECAR_ROWS`` cap drops every LATER artist/kind from the stored
    payload — stored-row counts are not figure truth there
    (``_sidecar_kind_row_groups`` returns None, the check-33
    ``_sidecar_plotted_values`` convention). Class B therefore SKIPS on a
    truncated sidecar; Class A falls back to the truncation-immune TEXT
    bases (series / legend labels), skipping when none is available.
    """
    warns: list[str] = []
    rows = _sidecar_kind_row_groups(meta)
    kinds, kind_groups, all_groups = rows if rows is not None else (Counter(), {}, set())

    if claims["both"]:
        bases = _both_claim_bases(meta, kinds, kind_groups, all_groups)
        if bases and max(bases.values()) <= 1:
            detail = ", ".join(f"{k}={v}" for k, v in bases.items())
            warns.append(
                f"`{basename}`: beat-1 prose claims `{claims['both'][0]}` but every "
                f"available sidecar basis renders <=1 element ({detail}) — regenerate "
                f"the figure or fix the prose; if the claim is genuinely satisfied "
                f"(e.g. two arms encoded inside one artist), acknowledge in body to ship"
            )

    if claims["one_per"] and rows is not None:
        for phrase, kind in claims["one_per"]:
            if kind == "line":
                n = _line_artist_count(kinds, kind_groups) or 0
            else:
                n = kinds.get(kind, 0)
            if n <= 1:
                warns.append(
                    f"`{basename}`: beat-1 prose claims `{phrase}` but the sidecar "
                    f"renders {n} `{kind}` element(s) — regenerate the figure or fix "
                    f"the prose; a legitimate n=1 (e.g. `one bar per source` on a "
                    f"genuinely single-source panel) is acknowledgeable in body"
                )
    return warns


def _beat_claims_for_one_figure(
    repo: Path, m: re.Match, rlines: list[str], img_idx: int, json_cache: dict
) -> tuple[list[str], bool]:
    """Process ONE same-repo figure for check 34 (the check-26/33 per-figure
    shape). Returns ``(warn_msgs, scanned)`` — ``scanned`` True only when a
    text-bearing sidecar was actually compared against a registered claim;
    mutates ``json_cache`` (per-URL parsed-sidecar cache) in place.
    """
    window = _beat1_prose_window(rlines, img_idx)
    if window is None:
        return [], False  # no per-result H3 above — no scoped window
    claims = _beat_series_claims(window)
    if not claims["both"] and not claims["one_per"]:
        return [], False  # no registered claim → nothing to check (never over-fire)
    url = m.group(0)
    if url not in json_cache:
        json_cache[url] = _read_figure_meta_json(repo, m.group("sha"), m.group("path"))
    meta = json_cache[url]
    if meta is None:
        return [], False  # no sidecar at that sha — check-24 fail-soft convention
    if not isinstance(meta.get("text"), dict):
        # THE forward-only gate: absence of `text` is the expected state of
        # every sidecar written before the `savefig_paper` rendered-text
        # capture landed — silent skip, never a loud FAIL (check 26's loud
        # missing-sidecar FAIL exists for a silently-uncommitted sidecar; a
        # loud branch HERE would retroactively flag every old body).
        return [], False
    basename = m.group("path").rsplit("/", 1)[-1]
    return _beat_claim_warnings(claims, meta, basename), True


def check_figure_beat_claims_vs_sidecar_text(body: str) -> CheckResult:
    """Check 34 (WARN): a figure's beat-1 series-structure claim — "shows
    both <…> arms/series/conditions/models/lines/curves" or "one
    bar/point/dot/marker/line/curve per <unit>" — must not contradict the
    series structure the figure's ``.meta.json`` sidecar demonstrably
    renders. FORWARD-ONLY: fires ONLY when the sidecar carries the
    ``meta["text"]`` rendered-text block (written by the current
    ``savefig_paper``); every pre-capture sidecar silently skips, so no
    existing body can retroactively WARN.

    The prose window is check 33's narrow previous-figure-bounded beat-1
    slice (``_beat1_prose_window`` — no cross-figure bleed, the #778
    false-WARN class). Both claim classes fire CONTRADICTION-ONLY: Class A
    ("both … arms") WARNs only when >=1 evidence basis is available AND every
    available basis reads <=1 (series labels / legend entries / bar rows /
    line artists / scatter artists / total artist groups — see
    ``_beat_claim_warnings``; a figure with no basis at all, e.g. an
    unlabeled single-artist scatter with no points payload signal, SKIPS —
    absence of evidence is never a contradiction). Class B ("one bar per X")
    requires a points payload and WARNs when the mapped kind renders <=1
    element. Deliberately NOT built (FP containment; may be added later
    behind the same forward-only gate): numeric-count claims ("three bars"),
    panel-count claims (the sidecar has no panel signal — see check 26),
    unit-NAME matching against series labels, and prose-vs-tick-label
    comparison.

    False-negative envelope (accepted by design): ``embed_text=False``
    figures never carry ``text`` and are never checked; ``embed_data=False``
    AND truncated (``data_truncated``) figures lack every points-derived
    basis (head-first row truncation can drop entire later artists/kinds
    from the stored payload), so only their series / legend labels can
    ground a Class-A read and Class B always skips; paraphrased claims
    ("each arm", "two models") miss the registered regexes.

    WARN, never FAIL (heuristic text parsing over natural prose — the
    24/28/33 severity convention; FAIL is reserved for check 26's provable
    structural contradictions). Silent skip (check-24 convention): missing /
    unparsable sidecar, no ``text`` block, no ``### `` H3 above the figure,
    no registered claim phrase in the window, non-same-repo URL. NO-OP PASS
    offline / ``--body-stdin`` / no figure section / no inline figures.
    Incident #1092 (clean-result-critic 9a-bis r1): a beat claimed "both
    input arms" / "one bar per re-fit item" contradicting the plotted
    series/bar structure and passed every mechanical figure check.
    """
    label = "figure beat claims vs sidecar rendered text (series-structure drift)"
    section = _figure_scan_section(body)
    text = section_text(body, section)
    if text is None:
        return CheckResult(label, True, f"no `## {section}` section to scan")
    rlines = text.splitlines()
    fig_at: list[tuple[str, int]] = []
    for idx, line in enumerate(rlines):
        for im in _IMAGE_RE.finditer(line):
            url = im.group(1).strip()
            url = url.split(None, 1)[0] if url else url
            if url:
                fig_at.append((url, idx))
    if not fig_at:
        return CheckResult(label, True, "no inline figures to scan")
    repo = _resolve_repo_root()
    if repo is None:
        return CheckResult(label, True, "skipped — repo root unresolved (offline / stdin)")
    warns: list[str] = []
    scanned = 0
    json_cache: dict[str, dict | None] = {}
    for url, img_idx in fig_at:
        m = _RAW_GITHUB_FIGURE_RE.match(url)
        if m is None or (m.group("owner").lower(), m.group("repo").lower()) != _THIS_REPO_SLUG:
            continue  # only same-repo sha-pinned figures resolve from git
        fig_warns, did_scan = _beat_claims_for_one_figure(repo, m, rlines, img_idx, json_cache)
        warns.extend(fig_warns)
        if did_scan:
            scanned += 1
    if warns:
        preview = "; ".join(warns[:3]) + (" …" if len(warns) > 3 else "")
        return CheckResult(
            label,
            True,
            f"{len(warns)} beat-claim contradiction(s) across {scanned} scanned "
            f"figure(s): {preview}",
            is_warn=True,
        )
    if scanned == 0:
        return CheckResult(
            label,
            True,
            "no beat-phrase claim with a text-bearing sidecar — nothing to compare",
        )
    return CheckResult(
        label,
        True,
        f"{scanned} figure(s): beat-phrase claims consistent with the rendered structure",
    )


# A prose claim that an artifact is NOT available — "not uploaded", "was not
# uploaded", "not separately uploaded", "cannot be audited", "cannot audit",
# "unavailable for audit", "not available for audit". The optional
# "separately " between "not" and "uploaded" catches the #653 r6 wording
# ("themselves were not separately uploaded"). Case-insensitive; the
# apostrophe in "wasn't" / "can't" matches both the ASCII `'` and the curly
# right-single-quote (real clean-result bodies use either) via `_APOS`.
#
# The #813 quota-hold family (the last four alternations) covers a LIVE
# pod-residency / storage-quota claim of non-availability ("remain on the
# pod", "quota-held", "under/pending/behind/blocked on the ... quota hold",
# "upload 403") \u2014 present-tense/stative shapes only, so a resolved narrative
# never fires (see the inline comment in the regex).
_APOS = "['\u2019]"  # ASCII apostrophe or curly right-single-quote (U+2019)
_AUDIT_DENIAL_RE = re.compile(
    r"(?:not\s+(?:separately\s+)?uploaded"
    r"|was\s+not\s+uploaded|wasn" + _APOS + r"t\s+uploaded"
    r"|cannot\s+be\s+audited|cannot\s+audit|can" + _APOS + r"t\s+be\s+audited"
    r"|(?:un|not\s+)available\s+for\s+audit"
    # The #813 family — a LIVE quota-hold / pod-residency claim of
    # non-availability. Present-tense / stative shapes ONLY (deliberately NOT
    # `remained ... on the pod` and NOT a bare `quota hold` without a stative
    # preposition): a resolved narrative ("after the quota hold cleared, all
    # files were uploaded") is not a denial and must not fire.
    r"|\bremain(?:s)?\s+on\s+the\s+pod\b"
    r"|\bquota[ _-]held\b"
    r"|(?:under|pending|behind|blocked\s+(?:on|by))\s+(?:the\s+|an?\s+)?"
    r"(?:[\w-]+\s+){0,4}?quota\s+hold"
    r"|\bupload\s+403\b)",
    re.IGNORECASE,
)
# Cheap per-line pre-filter: every _AUDIT_DENIAL_RE alternation family
# contains >=1 of these substrings ("upload" covers uploaded / upload 403 /
# unuploaded; "quota" the hold/held family; "pod" the remain-on-the-pod
# family). Pinned in sync with the denial regex by
# test_prefilter_covers_every_denial_family. Before #942 the pre-filter was an
# inline `"uploaded" not in line and "audit" not in line` — which skipped the
# #813 line-51 phrasing ("... ride the unreduced activation store (quota-held
# on the pod at write time)", neither token present) before the denial regex
# ever ran, so extending the regex alone would have been dead code there.
_AUDIT_LINE_PREFILTER_RE = re.compile(r"upload|audit|quota|\bpod\b", re.IGNORECASE)
# Artifact classes whose HF upload-path convention is known. A denial claim
# co-located (same line) with one of these names a concrete, mechanically
# probe-able artifact class — the only case this check fires on (a bare
# "the figure was not uploaded" with no data-artifact keyword is out of
# scope, since there is no HF data-repo path to reconcile it against).
#
# CRITICAL: the body PROSE spells these with hyphens/spaces and often the
# singular ("the install-probe completions", "the raw completions"), while the
# HF UPLOAD PATH uses the underscore plural (`install_probes/`,
# `raw_completions/`). So each canonical HF-path token (the dict key, used for
# the on-Hub path match) maps to a regex matching the prose spellings (the
# dict value, used to detect the denial in body text). Missing this split is
# exactly why the #653 r6 line ("install-probe ... completions ... not
# separately uploaded ... cannot be audited") slipped through a naive
# underscore-only scan.
_AUDIT_ARTIFACT_CLASSES: dict[str, re.Pattern[str]] = {
    "raw_completions": re.compile(r"raw[ _-]completions?|raw[ _-]?completion\b", re.IGNORECASE),
    "install_probes": re.compile(r"install[ _-]probes?", re.IGNORECASE),
    "mixes": re.compile(r"\btraining[ _-]mix(?:es)?\b|\bmixes\b", re.IGNORECASE),
    "onpolicy_pools": re.compile(
        r"on[ _-]?policy[ _-](?:pools?|completions?)|onpolicy[ _-]pools?", re.IGNORECASE
    ),
    "analysis_tensors": re.compile(r"analysis[ _-]tensors?", re.IGNORECASE),
    # #813 (issue813_mapchange_substrate) layout: unreduced/ (per-question
    # activation stores), reduced/ (per-question stores + summaries), maps/
    # (fitted maps). Prose spellings from the #813 v1 body verbatim. The
    # leading \b on "reduced" cannot fire inside "unreduced" (no word
    # boundary there); bare `\bstores?\b` and bare singular `map\b` are
    # deliberately EXCLUDED (common noun/verb in analysis prose).
    "unreduced": re.compile(
        r"\bunreduced\s+(?:activation\s+|per[ -]question\s+)?stores?\b"
        r"|\bactivation\s+stores?\b",
        re.IGNORECASE,
    ),
    "reduced": re.compile(
        r"\breduced\s+(?:per[ -]question\s+)?stores?\b|\breduced\s+summar(?:y|ies)\b",
        re.IGNORECASE,
    ),
    "maps": re.compile(
        r"\bfitted[ _-]maps?\b"
        r"|\b(?:behavior|linear|averaged|per[ -]example|factored)[ _-]maps?\b"
        r"|\bmaps\b",
        re.IGNORECASE,
    ),
}


# Max character distance on a line between an artifact-class prose mention and
# the nearest availability-denial phrase for the denial to be attributed to
# THAT artifact. The real #653 line has a 47-char gap; 200 comfortably covers
# a clause while excluding the false-positive shape where one line denies
# artifact A early and links a DIFFERENT artifact B by keyword far away
# (e.g. "the merged weights were not uploaded; raw completions are at <link>").
_AUDIT_DENIAL_PROXIMITY = 200


def _audit_denied_artifact_classes_in(line: str) -> list[str]:
    """Return the canonical HF-path token(s) (e.g. `install_probes`) of each
    artifact class whose prose spelling appears in ``line`` WITHIN
    ``_AUDIT_DENIAL_PROXIMITY`` chars of an availability-denial phrase on the
    same line, in dict order, deduplicated.

    Proximity-gating is the false-positive guard: a long line that denies one
    artifact and merely links another by keyword far away does not attribute
    the denial to the linked artifact. The canonical token (underscore plural)
    is what the on-Hub path match uses; the prose regex (hyphen/space/singular
    variants) is what detects it here.
    """
    denial_spans = [m.span() for m in _AUDIT_DENIAL_RE.finditer(line)]
    if not denial_spans:
        return []
    out: list[str] = []
    for canonical, pat in _AUDIT_ARTIFACT_CLASSES.items():
        for am in pat.finditer(line):
            a_start, a_end = am.span()
            near = any(
                # gap between the artifact mention and the denial phrase
                # (whichever side it falls on), clamped at 0 when they overlap.
                max(d_start - a_end, a_start - d_end, 0) <= _AUDIT_DENIAL_PROXIMITY
                for d_start, d_end in denial_spans
            )
            if near:
                if canonical not in out:
                    out.append(canonical)
                break
    return out


def _audit_keyword_path_re(keyword: str) -> re.Pattern[str]:
    """Alphanumeric-boundary matcher for a canonical HF-path ``keyword``
    (check 25): the keyword must appear as a path-component-like token, NOT a
    bare substring — so ``reduced`` never matches ``unreduced/``. ``/``, ``_``,
    ``.`` and ``-`` all count as boundaries (the legacy tokens keep matching
    their real path shapes: ``.../raw_completions/...``, ``..._mixes.jsonl``);
    an adjacent letter or digit does not. Callers search ``path.lower()``; the
    keyword is lowercased here, so the match is case-insensitive.
    """
    return re.compile(r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])")


def _hf_probe_keyword(
    repo_id: str, repo_type: str, sha: str, path_prefix: str, keyword: str
) -> tuple[str, str]:
    """Bounded direct-GET depth-agnostic keyword probe (check 25).

    Lists the URL's OWN sub-tree (`recursive=True` scoped to ``path_prefix``,
    NOT the whole repo) and matches ``keyword`` as an ALPHANUMERIC-BOUNDARY
    path component (``_audit_keyword_path_re`` — never a bare substring, so
    ``reduced`` cannot match ``unreduced/``; #942) in any FILE path under the
    prefix. The keyword can be nested at ANY depth (#653: the denial
    linked the tree ROOT while the file lives several levels down), so the
    scoped recursive listing consumes ``_hf_tree_pages`` (the shared bounded
    Link-header pagination under ``_HF_PROBE_MAX_PAGES`` +
    ``_HF_PROBE_DEADLINE_S`` caps) — a cap hit SKIPs rather than entering the
    SDK's unbounded backoff.

    not_found → SKIP — this call site's INDEPENDENT mapping (the deliberate
    check-23-FAIL-vs-25-SKIP asymmetry: check 25 cannot corroborate OR refute a
    denial against a revision that does not exist).
    """
    kw_re = _audit_keyword_path_re(keyword)
    needle = path_prefix.strip("/")
    for ev in _hf_tree_pages(repo_id, repo_type, sha, needle):
        if ev.kind == "not_found":
            return "skip", f"`{repo_id}@{sha[:8]}` (no such revision)"
        if ev.kind == "indeterminate":
            return "skip", f"`{repo_id}@{sha[:8]}` ({ev.note})"
        if ev.kind == "cap":
            # Hit the page / wall-clock cap before exhausting the sub-tree.
            return "skip", f"`{repo_id}@{sha[:8]}` (HF tree listing exceeded page/time cap)"
        if ev.kind == "exhausted":
            return "fail", ""  # successful, exhausted, no match → denial HOLDS (body PASS)
        for e in ev.entries:
            path = e.get("path", "")
            if (
                e.get("type") == "file"
                and kw_re.search(path.lower())
                and _hf_under_prefix(path, needle)
            ):
                return "pass", path  # denial is FALSE → body-level FAIL
    raise AssertionError("unreachable: _hf_tree_pages ended without a terminal event")


def _hf_under_prefix(path: str, needle: str) -> bool:
    """True iff ``path`` is the prefix itself or sits under ``needle/``
    (or there is no prefix scope)."""
    return not needle or path == needle or path.startswith(needle + "/")


def _hf_keyword_present_under_prefix(
    repo_id: str, repo_type: str, sha: str, path_prefix: str, keyword: str
) -> tuple[str, str]:
    """Fail-soft probe (check 25): does the HF repo at ``sha`` contain ≥1 file
    under ``path_prefix`` whose path carries ``keyword`` as a path component?

    Lists the URL's OWN sub-tree at the cited revision (a BOUNDED direct
    tree-endpoint GET with self-paginated recursion under a page/time cap —
    ``_hf_probe_keyword`` → ``_hf_tree_get``, NOT the unbounded whole-repo
    ``list_repo_files``, #733) and matches the keyword as an
    alphanumeric-boundary path component anywhere in the path
    (``_audit_keyword_path_re`` — `/` and `_` count as boundaries; NOT a bare
    substring, so ``reduced`` cannot match ``unreduced/``; #942), restricted
    to files under ``path_prefix``. A keyword nested at ANY depth below the
    prefix counts — the #653 denial linked the repo TREE
    ROOT (`…/issue653_install-validated-reladder`) while the file lives several
    levels down at `…/raw_completions/armB/install_probes/cell0/…`, so a fixed
    `<prefix>/<keyword>` candidate path would miss it; matching on the listing
    is depth-agnostic.

    Returns ``(verdict, matched_path)`` with verdict one of:
    - ``'pass'``  — ≥1 file under the prefix carries the keyword (the denial is
      FALSE); ``matched_path`` is the first such file.
    - ``'fail'``  — a SUCCESSFUL listing with NO matching file, i.e. the denial
      is corroborated for this URL (named ``'fail'`` to mirror
      ``_hf_url_existence``'s "definitive negative" verdict; the caller treats
      it as "denial holds for this URL", NOT a body FAIL).
    - ``'skip'`` — indeterminate (offline fence / no huggingface_hub / no such
      revision / any network / auth / HTTP error / page-time cap); surfaced as
      an `unverified` note, never a body FAIL.

    Fail-soft is mandatory and mirrors ``_hf_url_existence`` exactly: only a
    SUCCESSFUL listing yields pass/fail; every error path SKIPs. Definitive
    ``pass``/``fail`` verdicts are cached per-process; a ``skip`` is never
    cached, so a transient throttle that has since cleared is always re-probed.
    A ``not_found`` maps to SKIP here (NOT FAIL) — the deliberate
    check-23-vs-25 asymmetry.
    """
    if os.environ.get("EPM_VERIFY_BODY_NO_HF") == "1":
        return "skip", f"`{repo_id}@{sha[:8]}` (HF probe fenced)"
    try:
        import huggingface_hub  # noqa: F401 — local import: optional-dependency guard
    except ImportError:
        return "skip", f"`{repo_id}@{sha[:8]}` (huggingface_hub unavailable)"
    cache_key = (repo_id, repo_type, sha, path_prefix.strip("/"), keyword.lower())
    cached = _HF_EXISTENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    verdict, note = _hf_probe_keyword(repo_id, repo_type, sha, path_prefix, keyword)
    if verdict in ("pass", "fail"):
        _HF_EXISTENCE_CACHE[cache_key] = (verdict, note)
    return verdict, note


def check_audit_availability_claims_match_hf(body: str) -> CheckResult:
    """Check 25: a prose claim that an artifact "was not uploaded" /
    "cannot be audited" must NOT be contradicted by that artifact actually
    existing on the HF data repo.

    The #653 r6 pattern: the body asserted the per-cell install-probe
    firing/non-firing completions "were not separately uploaded, so the
    firing vs non-firing examples ... cannot be audited at the record
    level" — while those files DID exist on HF Hub under the same
    revision-pinned data-repo tree the body's own `## Methodology` /
    Reproducibility footer linked, at
    `…/issue653_install-validated-reladder/raw_completions/…/install_probes/`.
    The interpretation-critic caught the false "can't audit" claim by
    hand; this check mechanizes it so a future analyzer's honest-but-wrong
    non-availability claim FAILs before promotion.

    The #813 pattern (the quota-hold family, added by #942): the v1 body
    asserted the unreduced store / reduced summaries / fitted maps "remain
    on the pod under an HF public-storage quota hold (upload 403)" while
    `epm:upload-verification` + an independent listing proved all 24,206
    artifacts WERE on HF at the body-pinned revision. Two vocabularies were
    missing (the quota-hold denial family in `_AUDIT_DENIAL_RE`; the
    `unreduced` / `reduced` / `maps` classes in `_AUDIT_ARTIFACT_CLASSES`),
    and the old inline pre-filter (`"uploaded" not in line and "audit" not
    in line`) skipped quota-hold phrasings before the regex ever ran — the
    pre-filter is now the module-level `_AUDIT_LINE_PREFILTER_RE`, kept in
    sync with every denial-alternation family by
    `test_prefilter_covers_every_denial_family`. NOTE the scoping: this
    check probes ONLY revision-pinned HF URLs the body itself carries, so it
    protects a body that pins a covering HF URL; the #813 v1 incident-time
    body carried the denial with NO covering pinned URL and would still PASS
    vacuously — the no-URL escape is pre-existing check-25 architecture,
    shared by every denial family and artifact class, and is NOT closed by
    the #942 vocabulary extension.

    Mechanism (standalone — no `events.jsonl` / marker read, keeping the
    verifier self-contained): scan the fence-stripped body line by line for
    a line carrying BOTH (a) an availability-denial phrase
    (`_AUDIT_DENIAL_RE`) AND (b) a known data-artifact class spelled in PROSE
    within proximity on the same line (`_audit_denied_artifact_classes_in` —
    matches the hyphen/space/singular prose spellings, requires the mention to
    sit within `_AUDIT_DENIAL_PROXIMITY` chars of the denial phrase so a long
    line denying artifact A while merely linking artifact B does not
    false-trigger, and maps the prose to the canonical underscore-plural
    HF-path token, so the #653 "install-probe completions" prose resolves to
    the `install_probes/` upload path). For each denied class, take every HF Hub
    revision-pinned URL the body carries (`_gather_hf_pinned_urls`, the same
    set check 23 probes) and ask `_hf_keyword_present_under_prefix` whether
    the repo at the cited revision holds ≥1 file UNDER the URL's path-prefix
    whose path carries the keyword as a path component — depth-agnostic, so a
    file nested at `<tree-root>/…/<keyword>/…` is found even when the body
    only linked the tree ROOT (the #653 shape). If ANY HF URL yields such a
    file, the denial is false → FAIL.

    Fail-soft (identical semantics to check 23): every probe goes through
    `_hf_keyword_present_under_prefix`, which SKIPs (never FAILs) under the
    `EPM_VERIFY_BODY_NO_HF=1` offline fence, when `huggingface_hub` is
    unavailable, or on any network / auth / HTTP error. So an HfHubHTTPError
    or a sandbox with no network surfaces as an `unverified` note on the
    PASS line — the check never breaks a body just because the Hub is down,
    and never fabricates a FAIL it cannot substantiate. Only a SUCCESSFUL
    listing that returns ≥1 matching file is the FAIL; a successful listing
    with NO matching file CORROBORATES the denial (PASS).

    Vacuous PASS when the body carries no availability-denial-near-artifact
    line, or no HF Hub revision-pinned URL to probe against.
    """
    label = "audit-availability claims match HF Hub"
    stripped = _strip_fenced_blocks(body)
    # Find lines asserting non-availability of a known data-artifact class
    # (denial phrase + an artifact-class prose mention within proximity on the
    # same line). `suspect_keywords` holds the canonical underscore-plural
    # HF-path tokens.
    suspect_keywords: list[str] = []
    for line in stripped.splitlines():
        if not _AUDIT_LINE_PREFILTER_RE.search(line):
            continue  # cheap pre-filter before the proximity scan (kept in
            # sync with every _AUDIT_DENIAL_RE alternation family — see the
            # constant's comment + test_prefilter_covers_every_denial_family)
        for canonical in _audit_denied_artifact_classes_in(line):
            if canonical not in suspect_keywords:
                suspect_keywords.append(canonical)
    if not suspect_keywords:
        return CheckResult(label, True, "no availability-denial claim near a data artifact")
    hf_urls = _gather_hf_pinned_urls(body)
    if not hf_urls:
        return CheckResult(
            label,
            True,
            f"{len(suspect_keywords)} availability-denial claim(s) "
            f"({', '.join(suspect_keywords)}) but no HF Hub revision-pinned URL "
            "to reconcile against",
        )
    contradicted: list[str] = []
    unverified: list[str] = []
    for kw in suspect_keywords:
        kw_contradicted = False
        kw_confirmed_listing = False  # ≥1 HF URL listed successfully for this kw
        for repo_id, repo_type, sha, path_prefix, _raw in hf_urls:
            verdict, matched = _hf_keyword_present_under_prefix(
                repo_id, repo_type, sha, path_prefix, kw
            )
            if verdict == "pass":
                kw_contradicted = True
                contradicted.append(
                    f"body claims `{kw}` was not uploaded / cannot be audited, but "
                    f"`{matched}` exists at `{repo_id}@{sha[:8]}`"
                )
                break
            if verdict == "fail":
                kw_confirmed_listing = True
        if not kw_contradicted and not kw_confirmed_listing:
            unverified.append(kw)
    if contradicted:
        return CheckResult(label, False, "; ".join(contradicted))
    detail = (
        f"{len(suspect_keywords)} availability-denial claim(s) "
        f"({', '.join(suspect_keywords)}) reconciled against {len(hf_urls)} HF URL(s)"
    )
    if unverified:
        detail += f"; {len(unverified)} unverified (existence not confirmed): " + ", ".join(
            unverified
        )
    return CheckResult(label, True, detail)


# ─── Check 30: HF file-count claims vs the Hub tree (WARN) ──────────────────
#
# #931 shipped "528 files" / "10 shards" / "3 files" / "198 files" for HF
# artifacts where the scoped listing at the pinned revision holds
# 515/9/2/197 — folder entries were counted as files. Check 23 verifies the
# link RESOLVES; nothing verified the adjacent numeric claim. Check 30
# extracts count claims sitting in three conservative positions relative to
# a hex-pinned HF /tree markdown link and compares them against a
# files-only count from the SAME bounded raw tree-endpoint probe checks
# 23/25 use (#733 — never the SDK list_repo_tree / list_repo_files).
# WARN-only: a claim miscount is a prose-hygiene defect, not a broken body;
# and a WARN can never block offline (every non-definitive probe outcome
# SKIPs). #833 then shipped "908 files listed per namespace at the pinned
# revision" in a paren AFTER the link (908 = 891 blobs + 17 directory
# entries per namespace — list_repo_tree ENTRIES counted as files) — a
# claim shape neither Pattern A nor B could parse; Pattern C + the
# per-namespace gatherer close that extraction gap (#1088). #1112 then
# shipped "(7,372 files: …)" immediately after a backtick
# `raw_completions/` sub-path token in the pinned-bucket footer row —
# same line as the bucket's /tree/<sha> link, ~132 chars downstream —
# where the scoped tree at `<bucket>/raw_completions` holds 7,165 files
# + 207 folders. No pattern reached a count bound to a backtick SUB-PATH
# of a preceding link; Pattern D + the nearest-preceding-pinned-link
# binder close that gap (#1143).

# A markdown link whose target is an HF Hub URL. Two-stage extraction: match
# the link structure first, then scan the TEXT for a count-noun and parse the
# TARGET with the shared _HF_HUB_TREE_BLOB_URL_RE.
_MD_HF_LINK_RE = re.compile(
    r"\[(?P<text>[^\]]{1,300})\]\((?P<url>https?://huggingface\.co/[^)\s]+)\)"
)
# A numeric count claim: "515 files" / "1 file" / "10 shards". Comma-grouped
# thousands allowed ("1,234 files"); bounded at 6 plain digits. The negative
# lookahead makes per-namespace-qualified counts ("891 files per namespace")
# INVISIBLE to the whole-prefix Patterns A/B — such a count has per-namespace
# semantics (each named sub-namespace holds N files, #833), so reading it as
# a whole-prefix claim would compare N against the parent's total and
# manufacture a guaranteed false WARN. Per-namespace claims are extracted
# ONLY in the observed link-then-paren position
# (`_gather_hf_per_namespace_claims`); A/B-position per-namespace claims are
# a documented recall sacrifice, never a wrong comparison.
_COUNT_NOUN_RE = re.compile(
    r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+(?P<noun>files?|shards?)\b"
    r"(?!\s+(?:listed\s+)?per\s+namespace\b)",
    re.IGNORECASE,
)
# A parenthetical that OPENS with the count-noun, immediately preceding a
# markdown HF link (<=80 chars of qualifier inside the paren, <=5 chars of
# separator between `)` and `[` — spaces plus an optional `:` / dash): the
# #931 footer shape
# `... (515 files verified via scoped listing): [issue931_story_map @ ...](url)`.
# Carries the same per-namespace negative lookahead as `_COUNT_NOUN_RE`.
_COUNT_PAREN_LINK_RE = re.compile(
    r"\((?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+(?P<noun>files?|shards?)\b"
    r"(?!\s+(?:listed\s+)?per\s+namespace\b)"
    r"[^()]{0,80}\)"
    r"\s{0,2}[:\u2013\u2014-]?\s{0,2}"  # ':' / en-dash / em-dash / hyphen separators
    r"\[[^\]]{1,300}\]\((?P<url>https?://huggingface\.co/[^)\s]+)\)",
    re.IGNORECASE,
)
# Pattern C position: a parenthetical immediately AFTER a pinned HF markdown
# link — the #833 footer shape. The link-TEXT-capturing sibling of check 32's
# `_HF_LINK_THEN_PAREN_RE` (which starts at `\](` and cannot see the text; the
# text carries the backtick namespace tokens the per-namespace form needs).
# Separator is SAME-LINE only (`[ \t]{0,2}` — a newline / 3+-space gap never
# binds; stricter than check 32's `\s{0,2}` by design), and the paren body is
# `[^()]` so a NESTED paren inside the parenthetical is a documented recall
# sacrifice.
_HF_LINKTEXT_THEN_PAREN_RE = re.compile(
    r"\[(?P<text>[^\]]{1,300})\]\((?P<url>https?://huggingface\.co/[^)\s]+)\)"
    r"[ \t]{0,2}\((?P<paren>[^()]{1,300})\)"
)
# Phrase-anchored count claims INSIDE the paren (any position — the phrase,
# not the position, is the precision anchor; #833's count sits after an `=`).
# "listed" is the ONE tolerated filler (#833's original wording: "908 files
# listed per namespace"). Mutually exclusive by adjacency: "files per
# namespace at the pinned revision" matches ONLY the per-namespace form
# ("per namespace" intervenes between "files" and "at"). files-only (no
# shards) — a shards-per-namespace claim is unseen in the wild and stays a
# documented recall sacrifice.
_FILES_PER_NAMESPACE_RE = re.compile(
    r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+files?\s+(?:listed\s+)?per\s+namespace\b",
    re.IGNORECASE,
)
_FILES_AT_PINNED_REV_RE = re.compile(
    r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+files?\s+(?:listed\s+)?"
    r"at\s+the\s+pinned\s+revision\b",
    re.IGNORECASE,
)
# A backtick directory token in link TEXT: `analysis_tensors_nonemit/` — the
# trailing slash is the directory signature (a dotted token without it is a
# FILE claim, check 32's territory).
_BACKTICK_DIR_RE = re.compile(r"`(?P<ns>[A-Za-z0-9_\-./]{1,120}/)`")
# Pattern D (#1143, the #1112 footer shape): a backtick DIRECTORY token
# (trailing slash = the directory signature, same charset as
# _BACKTICK_DIR_RE plus a leading dot-segment decline — a `./`/`../` sub
# would join to a nonexistent probe path; a dotted token WITHOUT the slash
# is check 32's FILE territory) immediately followed by a parenthetical
# OPENING with the count-noun. The count scopes to <link-prefix>/<sub> at
# the sha of the nearest preceding pinned /tree link on the same line
# (binder below). The paren qualifier bound is 200 (Pattern B uses 80; the
# live #1112 qualifier is ~85 chars). The count-noun lookahead is WIDER
# than _COUNT_NOUN_RE's per-namespace decline: ANY distributive
# "per <word>" qualifier declines (e.g. the #460 "(11 files per adapter,
# 176 files total)" shape — per-adapter semantics, the join would target a
# nonexistent path); A/B deliberately keep the narrow #833
# per-namespace-only lookahead — a "per adapter"-style count in A/B
# position is a PRE-EXISTING hole outside #1143's scope, unchanged by
# design. Trailing negative lookahead: a paren immediately followed by an
# HF markdown link is Pattern B's paren-before-link shape — D declines
# rather than extracting a second, differently-scoped claim.
_BACKTICK_SUBPATH_COUNT_PAREN_RE = re.compile(
    r"`(?P<sub>(?!\.\.?/)[A-Za-z0-9_\-./]{1,120}/)`"
    r"[ \t]{0,2}"
    r"\((?P<count>\d{1,3}(?:,\d{3})+|\d{1,6})\s+(?P<noun>files?|shards?)\b"
    r"(?!\s+(?:listed\s+)?per\s+\w+\b)"
    r"[^()]{0,200}\)"
    r"(?!\s{0,2}[:\u2013\u2014-]?\s{0,2}\[[^\]]{1,300}\]\(https?://huggingface\.co)",
    re.IGNORECASE,
)
# Max chars between the binding link's closing ")" and the Pattern-D
# claim's backtick token. The live #1112 gap measures ~132 chars
# ("— `mixes/` (…), `selection/` (…), "); 400 gives ~3x margin while
# bounding runaway binding on pathological single-line bodies.
_SUBPATH_CLAIM_MAX_GAP = 400
# Per-body cap on UNIQUE (repo, sha, prefix) count probes. Worst-case wall
# arithmetic (the deadline is checked only AFTER each page fetch): one page
# costs up to _HF_PROBE_ATTEMPTS x _HF_PROBE_TIMEOUT_S + _HF_PROBE_SLEEP_S
# ≈ 10.5 s, so a probe's worst case is ≈ _HF_PROBE_DEADLINE_S + one page
# envelope ≈ 22.5 s (~21-23 s), and the per-body worst case is
# _HF_COUNT_MAX_PROBES x that ≈ 3 min — reached only when the Hub is
# pathologically slow on EVERY page of EVERY prefix; typical is one
# sub-second page per prefix. Claims past the cap surface as unverified
# notes, never a WARN.
_HF_COUNT_MAX_PROBES = 8
# Successful EXHAUSTIVE listings only: (repo_id, repo_type, sha, prefix) →
# (n_files, n_dirs). A skip (throttle / cap / offline) is NEVER cached —
# same convention as _HF_EXISTENCE_CACHE (#733) — so a transient throttle
# that has since cleared is re-probed on the next verify_text invocation.
_HF_TREE_FILE_COUNT_CACHE: dict[tuple[str, str, str, str], tuple[int, int]] = {}


def _nearest_preceding_pinned_tree_link(
    stripped: str, pos: int, links: list[re.Match]
) -> re.Match | None:
    """Pattern-D binding (#1143): the nearest markdown HF link (from the
    caller's precomputed ``_MD_HF_LINK_RE`` match list) ending at/before
    ``pos``, accepted ONLY when the gap between its closing ``)`` and
    ``pos`` (a) contains no newline (same footer row), (b) contains no
    ``[`` or ``]`` (no markdown link starts or ends inside the binding
    run — declines rather than binding PAST an intervening link, and a
    claim sitting inside another link's TEXT never binds backward), and
    (c) is <= ``_SUBPATH_CLAIM_MAX_GAP`` chars; and (d) the link parses
    as a hex-pinned /tree link (``_HF_HUB_TREE_BLOB_URL_RE`` +
    ``/tree/<sha>`` — same guards as ``_add``). Any failure returns None:
    the claim stays unextracted (precision over recall; a missed claim
    costs nothing on a WARN-only check). Nearest-only is deliberate: when
    the nearest preceding HF link is unpinned/``/blob/``/``/tree/main``
    the claim is DECLINED, never re-bound to an earlier link — an earlier
    binding would have the intervening link's ``](`` in the gap anyway,
    so decline-on-brackets and nearest-only are mutually consistent."""
    best = None
    for lm in links:
        if lm.end() <= pos:
            best = lm
        else:
            break  # finditer order is left-to-right
    if best is None:
        return None
    gap = stripped[best.end() : pos]
    if "\n" in gap or "[" in gap or "]" in gap or len(gap) > _SUBPATH_CLAIM_MAX_GAP:
        return None
    url = best.group("url").rstrip(".,;:!?")
    m = _HF_HUB_TREE_BLOB_URL_RE.match(url)
    if m is None or f"/tree/{m.group('sha')}" not in url:
        return None
    return best


def _gather_hf_count_claims(body: str) -> list[tuple[int, str, str, str, str, str]]:
    """Extract ``(claimed_count, noun, repo_id, repo_type, sha, path_prefix)``
    tuples for numeric file/shard-count claims ADJACENT to hex-pinned HF
    ``/tree`` markdown links (check 30). Fence-stripped; deduplicated;
    ``/blob/`` links and moving refs (``/tree/main``) are out of scope (the
    shared URL regex only matches 7-40-char hex revisions, mirroring
    check 23).

    Four conservative positions (precision over recall — a missed claim
    costs nothing, the check is a net-new WARN):

    - **Pattern A** — the count-noun sits INSIDE the markdown link TEXT
      (``[pairs_meta, 9 files](…/tree/<sha>/…)``): the number and the link
      are structurally bound, so unrelated prose numbers ("seed 42",
      "180 of 197 valid quotes" after a link) can never match.
    - **Pattern B** — a parenthetical that OPENS with the count-noun
      immediately precedes the link (``(515 files verified via scoped
      listing): [x](…)``, the #931 footer shape).
    - **Pattern C (pinned-revision form)** — a parenthetical immediately
      AFTER the link containing the anchored phrase ``N files (listed )?at
      the pinned revision`` at ANY position inside the paren (``[x](…)
      (1,234 files at the pinned revision)``): the phrase, not the
      position, is the precision anchor. Same-paren PER-NAMESPACE claims
      (``N files (listed )?per namespace``, the #833 footer shape) are NOT
      whole-prefix claims — they are extracted separately by
      ``_gather_hf_per_namespace_claims`` and, via the negative lookahead
      in ``_COUNT_NOUN_RE`` / ``_COUNT_PAREN_LINK_RE``, stay invisible to
      Patterns A/B (wrong semantics would compare N against the parent's
      total).
    - **Pattern D (backtick sub-path, #1143)** — a backtick ``dir/`` token
      immediately followed by a parenthetical OPENING with the count-noun
      (``... — `raw_completions/` (7,165 files: …)``, the #1112 footer
      shape), bound via ``_nearest_preceding_pinned_tree_link`` to the
      nearest preceding hex-pinned ``/tree`` markdown link on the SAME
      line (bracket-free, ≤``_SUBPATH_CLAIM_MAX_GAP``-char gap); the count
      scopes to the JOINED prefix ``<link-prefix>/<sub-path>`` at the
      link's sha — the same join the per-namespace leg uses.

    Known recall sacrifices (each avoids a concrete false-positive class):
    prose counts near BARE (non-markdown) HF URLs; counts AFTER the link
    OUTSIDE a link-adjacent paren carrying one of the two anchored phrases
    (generic post-link prose counts stay sacrificed); parentheticals where
    the count is not leading ("(total 515 files)") unless phrase-anchored
    per Pattern C — mirrored by Pattern D, whose count must OPEN the
    paren; compound claims ("8 eval JSONs + 2 npz"); nouns other
    than file(s) / shard(s) ("rows" / "completions" are RECORD counts, not
    file counts); per-namespace-qualified counts in A/B positions (see the
    lookahead note above). Pattern-D-specific sacrifices: a claim whose
    nearest preceding HF link is unpinned / ``/blob/`` / ``/tree/main``
    (declined, never re-bound to an earlier link); a bracketed, multi-line,
    or >``_SUBPATH_CLAIM_MAX_GAP``-char gap; a backtick token without the
    trailing slash (check 32's FILE-claim territory); a D-paren
    immediately followed by an HF markdown link (Pattern B's
    paren-before-link shape wins — D's trailing lookahead declines); a
    BARE (non-markdown) pinned HF URL in the gap neither declines nor
    rebinds — the claim binds past it to the earlier markdown link, a
    documented recall/precision sacrifice. Lookahead ASYMMETRY (deliberate):
    Pattern D declines ANY distributive ``per <word>`` qualifier while A/B
    keep the narrow #833 per-namespace-only lookahead — a "per
    adapter"-style count in A/B position is a PRE-EXISTING hole outside
    #1143's scope, left unchanged by design.
    """
    kind_to_type = {"datasets": "dataset", "spaces": "space", None: "model"}
    stripped = _strip_fenced_blocks(body)
    out: list[tuple[int, str, str, str, str, str]] = []
    seen: set[tuple] = set()

    def _add(count_s: str, noun: str, url: str, sub_path: str | None = None) -> None:
        url = url.rstrip(".,;:!?")
        m = _HF_HUB_TREE_BLOB_URL_RE.match(url)
        if m is None:
            return
        if f"/tree/{m.group('sha')}" not in url:
            return  # a /blob/ link is a single file — a count claim there is out of scope
        count = int(count_s.replace(",", ""))
        repo_id = f"{m.group('owner')}/{m.group('repo')}"
        prefix = (m.group("path") or "").rstrip("/")
        if sub_path is not None:  # Pattern D: scope to <link-prefix>/<sub-path>
            prefix = "/".join(p for p in (prefix, sub_path.strip("/")) if p)
        key = (
            count,
            noun.lower().rstrip("s"),
            repo_id,
            m.group("sha"),
            prefix,
        )
        if key in seen:
            return
        seen.add(key)
        out.append(
            (
                count,
                noun,
                repo_id,
                kind_to_type[m.group("kind")],
                m.group("sha"),
                prefix,
            )
        )

    link_matches = list(_MD_HF_LINK_RE.finditer(stripped))
    for lm in link_matches:  # Pattern A: count in link TEXT
        for cm in _COUNT_NOUN_RE.finditer(lm.group("text")):
            _add(cm.group("count"), cm.group("noun"), lm.group("url"))
    for pm in _COUNT_PAREN_LINK_RE.finditer(stripped):  # Pattern B: paren before link
        _add(pm.group("count"), pm.group("noun"), pm.group("url"))
    for lm in _HF_LINKTEXT_THEN_PAREN_RE.finditer(stripped):  # Pattern C: paren after link
        for cm in _FILES_AT_PINNED_REV_RE.finditer(lm.group("paren")):
            _add(cm.group("count"), "files", lm.group("url"))
    for dm in _BACKTICK_SUBPATH_COUNT_PAREN_RE.finditer(stripped):  # Pattern D (#1143)
        lm = _nearest_preceding_pinned_tree_link(stripped, dm.start(), link_matches)
        if lm is None:
            continue
        _add(dm.group("count"), dm.group("noun"), lm.group("url"), sub_path=dm.group("sub"))
    return out


def _gather_hf_per_namespace_claims(
    body: str,
) -> list[tuple[int, str, str, str, str, tuple[str, ...]]]:
    """Extract ``(claimed_count, repo_id, repo_type, sha, link_prefix,
    namespaces)`` tuples for "N files (listed )?per namespace" claims in a
    parenthetical immediately AFTER a hex-pinned HF ``/tree`` markdown link
    (check 30 Pattern C, the #833 footer shape — the link URL points at the
    PARENT prefix, the link TEXT names the sub-namespaces). ``namespaces``
    are the backtick ``dir/`` tokens in the link TEXT (trailing slash
    stripped, order-preserving dedup); an EMPTY tuple means the claim was
    extracted but its namespaces are unresolvable — the caller surfaces it
    as `unverified`, never probes, never WARNs (no parent-prefix guess, no
    divisibility heuristics: "never ground a WARN on a partial read").
    Fence-stripped; deduplicated; ``/blob/`` links and moving refs are out
    of scope (same URL guards as ``_gather_hf_count_claims``). A nested
    paren inside the parenthetical is a documented recall sacrifice
    (`_HF_LINKTEXT_THEN_PAREN_RE` bounds the paren body at ``[^()]``)."""
    kind_to_type = {"datasets": "dataset", "spaces": "space", None: "model"}
    stripped = _strip_fenced_blocks(body)
    out: list[tuple[int, str, str, str, str, tuple[str, ...]]] = []
    seen: set[tuple] = set()
    for lm in _HF_LINKTEXT_THEN_PAREN_RE.finditer(stripped):
        url = lm.group("url").rstrip(".,;:!?")
        m = _HF_HUB_TREE_BLOB_URL_RE.match(url)
        if m is None or f"/tree/{m.group('sha')}" not in url:
            continue
        namespaces = tuple(
            dict.fromkeys(  # order-preserving dedup
                ns.strip("/") for ns in _BACKTICK_DIR_RE.findall(lm.group("text")) if ns.strip("/")
            )
        )
        repo_id = f"{m.group('owner')}/{m.group('repo')}"
        prefix = (m.group("path") or "").rstrip("/")
        for cm in _FILES_PER_NAMESPACE_RE.finditer(lm.group("paren")):
            count = int(cm.group("count").replace(",", ""))
            key = (count, repo_id, m.group("sha"), prefix, namespaces)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                (count, repo_id, kind_to_type[m.group("kind")], m.group("sha"), prefix, namespaces)
            )
    return out


def _hf_count_files_under_prefix(
    repo_id: str, repo_type: str, sha: str, path_prefix: str
) -> tuple[str, int, int, str]:
    """Bounded scoped-recursive tree listing → ``(status, n_files, n_dirs,
    note)`` (check 30).

    Consumes ``_hf_tree_pages`` (#733): direct GETs via
    ``_hf_tree_get`` (real per-request timeout, ≤ ``_HF_PROBE_ATTEMPTS``
    retries/page), following Link-header pagination under
    ``_HF_PROBE_MAX_PAGES`` + ``_HF_PROBE_DEADLINE_S``. Counts only entries
    with ``"type" == "file"`` under the prefix (``"directory"`` entries
    counted separately for the files+folders diagnostic; an entry whose path
    IS the prefix and is a directory is the prefix itself, not content, and
    is skipped). ``status == "ok"`` ONLY for an EXHAUSTIVE listing
    (``next_page is None``); a cap hit, ``not_found``, or any transient
    error is ``"skip"`` — a partial count must never ground a WARN.
    """
    needle = path_prefix.strip("/")
    n_files = n_dirs = 0
    for ev in _hf_tree_pages(repo_id, repo_type, sha, needle):
        if ev.kind == "not_found":
            return "skip", -1, -1, "no such revision/path"
        if ev.kind == "indeterminate":
            return "skip", -1, -1, ev.note
        if ev.kind == "cap":
            return "skip", -1, -1, "HF tree listing exceeded page/time cap"
        if ev.kind == "exhausted":
            return "ok", n_files, n_dirs, ""
        for e in ev.entries:
            path = e.get("path", "")
            if not _hf_under_prefix(path, needle):
                continue
            etype = e.get("type")
            if etype == "file":
                n_files += 1
            elif etype == "directory" and path != needle:
                n_dirs += 1
    raise AssertionError("unreachable: _hf_tree_pages ended without a terminal event")


def _hf_file_count_for_prefix(
    repo_id: str, repo_type: str, sha: str, path_prefix: str
) -> tuple[str, int, int, str]:
    """Fence + optional-dependency + cache wrapper around
    ``_hf_count_files_under_prefix`` (mirrors ``_hf_url_existence`` /
    ``_hf_keyword_present_under_prefix`` exactly): SKIPs under the
    ``EPM_VERIFY_BODY_NO_HF=1`` offline fence or when ``huggingface_hub`` is
    unavailable; caches ONLY successful exhaustive listings in
    ``_HF_TREE_FILE_COUNT_CACHE`` (a skip is never cached)."""
    if os.environ.get("EPM_VERIFY_BODY_NO_HF") == "1":
        return "skip", -1, -1, "HF probe fenced"
    try:
        import huggingface_hub  # noqa: F401 — local import: optional-dependency guard
    except ImportError:
        return "skip", -1, -1, "huggingface_hub unavailable"
    key = (repo_id, repo_type, sha, path_prefix.strip("/"))
    cached = _HF_TREE_FILE_COUNT_CACHE.get(key)
    if cached is not None:
        return "ok", cached[0], cached[1], ""
    status, n_files, n_dirs, note = _hf_count_files_under_prefix(
        repo_id, repo_type, sha, path_prefix
    )
    if status == "ok":
        _HF_TREE_FILE_COUNT_CACHE[key] = (n_files, n_dirs)
    return status, n_files, n_dirs, note


def _verify_hf_whole_prefix_claims(
    claims: list[tuple[int, str, str, str, str, str]],
    probed,
    mismatched: list[str],
    unverified: list[str],
) -> None:
    """Whole-prefix verification leg of check 30 (Patterns A/B/C-pinned):
    compare each ``_gather_hf_count_claims`` tuple against the files-only
    count from the caller's shared memoized/capped ``probed`` closure,
    appending mismatch messages / unverified notes to the caller's lists.
    Files claims are two-sided; shard claims are one-sided (folder-inflation
    only)."""
    for count, noun, repo_id, repo_type, sha, prefix in claims:
        result = probed((repo_id, repo_type, sha, prefix))
        if result is None:
            continue  # per-body probe cap — cap note already appended
        status, n_files, n_dirs, note = result
        if status != "ok":
            skip_note = f"`{prefix or repo_id}@{sha[:8]}` ({note})"
            if skip_note not in unverified:
                unverified.append(skip_note)
            continue
        if noun.lower().startswith("shard") and count <= n_files:
            continue  # shard claims are one-sided: only folder-inflation WARNs
        if count == n_files:
            continue
        msg = (
            f"body claims {count} {noun} at `{prefix or '/'}` but `{repo_id}@{sha[:8]}` "
            f"holds {n_files} file(s)"
        )
        if n_dirs and count == n_files + n_dirs:
            msg += (
                f" + {n_dirs} folder(s) — the claimed count is consistent with "
                "files+folders (folder entries are not files)"
            )
        elif count < n_files:
            msg += " (or the claim describes a subset of the prefix)"
        mismatched.append(msg)


def _verify_hf_per_namespace_claims(
    ns_claims: list[tuple[int, str, str, str, str, tuple[str, ...]]],
    probed,
    mismatched: list[str],
    unverified: list[str],
) -> None:
    """Per-namespace verification leg of check 30 (Pattern C, #833): probe
    each ``<link-prefix>/<ns>`` through the caller's shared memoized/capped
    ``probed`` closure and compare the claimed per-namespace count two-sided
    against each namespace's files-only count. A claim with NO extractable
    namespaces surfaces as an unverified note with ZERO probes (never a
    WARN, never a parent-prefix guess); each namespace WARNs only on its OWN
    exhaustive listing — a skipped sibling degrades to unverified without
    suppressing or creating WARNs elsewhere."""
    for count, repo_id, repo_type, sha, prefix, namespaces in ns_claims:
        if not namespaces:
            note = (
                f"`{prefix or repo_id}@{sha[:8]}` (per-namespace claim: no "
                "backtick namespace names in the link text)"
            )
            if note not in unverified:
                unverified.append(note)
            continue
        for ns in namespaces:
            sub = "/".join(p for p in (prefix, ns) if p)
            result = probed((repo_id, repo_type, sha, sub))
            if result is None:
                continue  # per-body probe cap — cap note already appended
            status, n_files, n_dirs, note = result
            if status != "ok":
                skip_note = f"`{sub}@{sha[:8]}` ({note})"
                if skip_note not in unverified:
                    unverified.append(skip_note)
                continue
            if count == n_files:
                continue
            msg = (
                f"body claims {count} files per namespace but namespace "
                f"`{sub}` at `{repo_id}@{sha[:8]}` holds {n_files} file(s)"
            )
            if n_dirs and count == n_files + n_dirs:
                msg += (
                    f" + {n_dirs} folder(s) — the claimed count is consistent "
                    "with files+folders (folder entries are not files)"
                )
            elif count < n_files:
                msg += " (or the claim describes a subset of the namespace)"
            mismatched.append(msg)


def check_hf_file_count_claims(body: str) -> CheckResult:
    """Check 30 (WARN): numeric file-count claims adjacent to hex-pinned HF
    ``/tree`` links must match a files-only scoped Hub tree count.

    Incident: task #931 shipped "528 files" / "10 shards" / "3 files" /
    "198 files" where the scoped listing at the pinned revision holds
    515/9/2/197 — folder entries were counted as files. This check compares
    each extracted claim (``_gather_hf_count_claims`` — Pattern A count in
    link text / Pattern B paren-before-link / Pattern C anchored
    pinned-revision phrase in a paren AFTER the link / Pattern D backtick
    ``dir/`` sub-path + count-opening paren bound to the nearest preceding
    pinned link and scoped to ``<link-prefix>/<sub-path>`` (#1143, the
    #1112 footer shape); see its docstring for
    the precision/recall trade-offs) against an EXHAUSTIVE files-only count
    of the pinned prefix (``_hf_file_count_for_prefix`` → the #733 bounded
    raw tree-endpoint probe; folders excluded).

    **Per-namespace claims** (``_gather_hf_per_namespace_claims`` — "N files
    (listed )?per namespace" in a paren AFTER the link, the #833 footer
    shape) verify EACH sub-namespace named as a backtick ``dir/`` token in
    the link TEXT: probe ``<link-prefix>/<ns>`` through the SAME shared
    memo/cap, exact two-sided compare of the claimed N against each
    namespace's files-only count, files+folders diagnostic per namespace
    (the #833 signature: 908 = 891 blobs + 17 dirs). A claim whose link
    text names NO backtick namespaces surfaces as an `unverified` note with
    ZERO probes — never a WARN, never a parent-prefix guess. Each namespace
    WARNs only on its OWN exhaustive listing; a skipped sibling namespace
    surfaces as unverified without suppressing or creating WARNs elsewhere.

    Semantics:

    - **WARN, never FAIL.** A mismatch returns ``CheckResult(name, True,
      detail, is_warn=True)`` — ``passed`` stays True so overall ``ok``
      never flips. There is NO code path returning ``passed=False``.
    - **Files claims are two-sided; shard claims are one-sided.** A hex
      pin is immutable so a "N files" claim admits exact comparison; a
      legitimate "9 shards" prefix can also hold a manifest/sidecar
      (10 files), so shard claims WARN only when claimed > file count —
      the folder-inflation signature (#931's "10 shards" vs 9 files).
    - **Descriptive diagnostics.** When the claimed count equals
      files+folders, the WARN notes the claim is *consistent with*
      files+folders (folder entries are not files) — a diagnosis hint,
      not asserted causation. When the claim UNDERCOUNTS the prefix
      (claimed < files), the WARN carries the hedge that the claim may
      describe a subset of the prefix (an overcount cannot be a subset,
      so the hedge is omitted there).
    - **Fail-soft everywhere.** The offline fence
      (``EPM_VERIFY_BODY_NO_HF=1``), a missing ``huggingface_hub``, a
      429 / network error, ``not_found``, a page/time-cap hit, and the
      per-body ``_HF_COUNT_MAX_PROBES`` cap all surface as `unverified`
      notes on a PASS line; a PARTIAL count never grounds a WARN. When
      mismatches and unverified claims coexist, the unverified list is
      appended to the WARN detail (never dropped).
    - **Probe accounting.** Unique ``(repo, sha, prefix)`` keys are probed
      AT MOST once per invocation via an intra-invocation memo — a second
      claim on a key whose probe SKIPPED reuses the skip note instead of
      re-probing, and skipped keys count toward ``_HF_COUNT_MAX_PROBES``.
      The cross-process ``_HF_TREE_FILE_COUNT_CACHE`` stores only
      successful exhaustive listings, so a later invocation re-probes a
      cleared throttle.

    Vacuous PASS — with ZERO Hub probes issued — when no count claim sits
    adjacent to an HF tree link.
    """
    name = "HF file-count claims match the Hub tree"
    claims = _gather_hf_count_claims(body)
    ns_claims = _gather_hf_per_namespace_claims(body)
    if not claims and not ns_claims:
        return CheckResult(name, True, "no file-count claims adjacent to HF tree links")
    mismatched: list[str] = []
    unverified: list[str] = []
    probe_memo: dict[tuple[str, str, str, str], tuple[str, int, int, str]] = {}

    def _probed(key: tuple[str, str, str, str]) -> tuple[str, int, int, str] | None:
        """Memoized probe under the SHARED per-body cap (one memo + one cap
        accounting across the whole-prefix AND per-namespace loops). Memo
        lookup happens BEFORE the cap check, so a past-cap re-reference to
        an already-probed key is served from the memo; a FRESH probe past
        the cap returns None after appending the cap note."""
        if key not in probe_memo:
            if len(probe_memo) >= _HF_COUNT_MAX_PROBES:
                cap_note = f"`{key[3] or key[0]}@{key[2][:8]}` (per-body probe cap)"
                if cap_note not in unverified:
                    unverified.append(cap_note)
                return None
            probe_memo[key] = _hf_file_count_for_prefix(*key)
        return probe_memo[key]

    _verify_hf_whole_prefix_claims(claims, _probed, mismatched, unverified)
    _verify_hf_per_namespace_claims(ns_claims, _probed, mismatched, unverified)
    unverified_detail = ""
    if unverified:
        unverified_detail = f"; {len(unverified)} unverified (count not confirmed): " + "; ".join(
            unverified
        )
    if mismatched:
        return CheckResult(name, True, "; ".join(mismatched) + unverified_detail, is_warn=True)
    n_checked = len(claims) + len(ns_claims)
    return CheckResult(name, True, f"{n_checked} claim(s) checked" + unverified_detail)


# ─── Check 32: adjacent backtick file claims are members of the pinned tree ──
# (#952 r1: the body claimed `divergence_bank_queries.json` at the pinned HF
# eval_results@5b62649 tree while the file lives in git; check 23 validated
# the tree link itself but never the NAMED-adjacent file's membership.)
# Parenthetical immediately AFTER a pinned HF markdown link (the #952 shape;
# check 30's paren-pattern is BEFORE the link). Content bounded, no nesting.
_HF_LINK_THEN_PAREN_RE = re.compile(
    r"\]\((?P<url>https?://huggingface\.co/[^)\s]+)\)\s{0,2}\((?P<paren>[^()]{1,300})\)"
)
# A backtick-delimited token (candidate filename); bounded, single-line.
_BACKTICK_TOKEN_RE = re.compile(r"`([^`\n]{1,80})`")
# A claimable filename: alnum-leading dotted basename with an artifact-class
# extension (corpus-grounded whitelist — `.py`/`.sh` deliberately excluded:
# script mentions near HF links are generator provenance, not upload claims).
# Tokens with `/` (relative paths), brace/wildcard globs, spaces, no dot, or
# >64-char stems are rejected by construction (plan #1016 §3.5).
_HF_ADJ_FILENAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}"
    r"\.(?:jsonl?|pt|safetensors|npy|npz|csv|tsv|txt|md|png|pdf|parquet|bin|ya?ml|html|log)$"
)
# Per-body cap on UNIQUE (repo, sha, prefix) membership probes — same value +
# worst-case wall arithmetic as check 30's `_HF_COUNT_MAX_PROBES` (see that
# comment: ~22.5 s worst case per probe, ~3 min per body under a
# pathologically slow Hub; typical is one sub-second page per prefix).
# Claims past the cap surface as unverified notes, never a WARN.
_HF_MEMBER_MAX_PROBES = 8
# Successful EXHAUSTIVE listings only: (repo_id, repo_type, sha, prefix) →
# frozenset of entry basenames (files AND directories). A skip (throttle /
# cap / offline) is NEVER cached — same convention as _HF_EXISTENCE_CACHE
# (#733) — so a transient throttle is re-probed on the next invocation.
_HF_TREE_BASENAMES_CACHE: dict[tuple[str, str, str, str], frozenset[str]] = {}


def _gather_hf_adjacent_file_claims(body: str) -> list[tuple[str, str, str, str, str, str]]:
    """Extract ``(repo_id, repo_type, sha, path_prefix, filename, shape)``
    tuples for backtick FILENAME claims ADJACENT to hex-pinned HF ``/tree``
    markdown links (check 32). Fence-stripped; deduplicated on the full
    ``(repo_id, repo_type, sha, prefix, filename)`` key (the probe-memo key);
    ``shape`` is ``"PAREN"`` or ``"LINKTEXT"`` (threaded into the WARN detail
    so per-shape adjudication is mechanical).

    Two anchored shapes only (precision over recall — a missed claim costs
    nothing, a misattributed one costs a spurious WARN):

    - **PAREN** — a parenthetical immediately AFTER the markdown link
      (``[…](…/tree/<sha>/…) (`f.json`, …)`` — the #952-r1 incident shape;
      check 30's paren-pattern sits BEFORE the link).
    - **LINKTEXT** — a dotted backtick token inside the link TEXT
      (``[`f.json`](…/tree/<sha>/dir)`` / ``[`f.json` @ sha](…)``): the text
      names a file, the target names a pinned directory — an unambiguous
      membership claim.

    Known recall sacrifices (each avoids a concrete false-positive class):
    backtick filenames BEFORE the link (real-corpus misattribution evidence —
    the preceding filename belongs to a different git-side/descriptive claim
    in ≥3 of 10 sampled hits); ``/blob/`` links (check 23 already validates
    the full blob path); relative-path / brace-glob / wildcard tokens;
    non-markdown bare URLs; moving refs (``/tree/main`` never matches the
    shared hex-pinned URL regex). A token equal to the URL's own terminal
    path component is skipped — check 23 covers the URL's own path.
    """
    kind_to_type = {"datasets": "dataset", "spaces": "space", None: "model"}
    stripped = _strip_fenced_blocks(body)
    out: list[tuple[str, str, str, str, str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def _add(url: str, token: str, shape: str) -> None:
        fname = token.strip()
        if not _HF_ADJ_FILENAME_RE.match(fname):
            return
        m = _HF_HUB_TREE_BLOB_URL_RE.match(url.rstrip(".,;:!?"))
        if m is None or f"/tree/{m.group('sha')}" not in url:
            return  # non-HF (e.g. github) or /blob/ → out of scope
        prefix = (m.group("path") or "").rstrip("/")
        if posixpath.basename(prefix) == fname:
            return  # the URL's own terminal component — check 23 covers it
        repo_id = f"{m.group('owner')}/{m.group('repo')}"
        repo_type = kind_to_type[m.group("kind")]
        key = (repo_id, repo_type, m.group("sha"), prefix, fname)
        if key in seen:
            return
        seen.add(key)
        out.append((repo_id, repo_type, m.group("sha"), prefix, fname, shape))

    for pm in _HF_LINK_THEN_PAREN_RE.finditer(stripped):  # shape PAREN
        for token in _BACKTICK_TOKEN_RE.findall(pm.group("paren")):
            _add(pm.group("url"), token, "PAREN")
    for lm in _MD_HF_LINK_RE.finditer(stripped):  # shape LINKTEXT
        for token in _BACKTICK_TOKEN_RE.findall(lm.group("text")):
            _add(lm.group("url"), token, "LINKTEXT")
    return out


def _hf_basenames_under_prefix(
    repo_id: str, repo_type: str, sha: str, path_prefix: str
) -> tuple[str, frozenset[str], str]:
    """Bounded scoped-recursive tree listing → ``(status, basenames, note)``
    (check 32).

    Consumes ``_hf_tree_pages`` (#1008 → #733): direct GETs via
    ``_hf_tree_get`` (real per-request timeout, ≤ ``_HF_PROBE_ATTEMPTS``
    retries/page), following Link-header pagination under
    ``_HF_PROBE_MAX_PAGES`` + ``_HF_PROBE_DEADLINE_S`` (the safe
    headers-before-URL ordering the #1016 fix added here now lives inside
    the shared generator). Collects basenames of
    ALL entries (file AND directory) under the prefix — a directory basename
    match also suppresses the WARN (FP-safe; dotted directory names are
    rare); an entry whose path IS the prefix is the prefix itself, not
    content, and is skipped. ``status == "ok"`` ONLY for an EXHAUSTIVE
    listing (``next_page is None``); a cap hit, ``not_found``, or any
    transient error is ``"skip"`` — a PARTIAL listing must never ground a
    WARN. ``not_found`` → skip, NOT fail: check 23 owns the dead-pin FAIL
    (the documented check-23-vs-25/30/32 asymmetry, `_TreeProbeResult`).
    """
    needle = path_prefix.strip("/")
    basenames: set[str] = set()
    for ev in _hf_tree_pages(repo_id, repo_type, sha, needle):
        if ev.kind == "not_found":
            return "skip", frozenset(), "no such revision/path"
        if ev.kind == "indeterminate":
            return "skip", frozenset(), ev.note
        if ev.kind == "cap":
            return "skip", frozenset(), "HF tree listing exceeded page/time cap"
        if ev.kind == "exhausted":
            return "ok", frozenset(basenames), ""
        for e in ev.entries:
            path = e.get("path", "")
            if not _hf_under_prefix(path, needle) or path == needle:
                continue
            basenames.add(posixpath.basename(path))
    raise AssertionError("unreachable: _hf_tree_pages ended without a terminal event")


def _hf_basenames_for_prefix(
    repo_id: str, repo_type: str, sha: str, path_prefix: str
) -> tuple[str, frozenset[str], str]:
    """Fence + optional-dependency + cache wrapper around
    ``_hf_basenames_under_prefix`` (mirrors ``_hf_file_count_for_prefix``
    exactly): SKIPs under the ``EPM_VERIFY_BODY_NO_HF=1`` offline fence or
    when ``huggingface_hub`` is unavailable; caches ONLY successful
    exhaustive listings in ``_HF_TREE_BASENAMES_CACHE`` (a skip is never
    cached)."""
    if os.environ.get("EPM_VERIFY_BODY_NO_HF") == "1":
        return "skip", frozenset(), "HF probe fenced"
    try:
        import huggingface_hub  # noqa: F401 — local import: optional-dependency guard
    except ImportError:
        return "skip", frozenset(), "huggingface_hub unavailable"
    key = (repo_id, repo_type, sha, path_prefix.strip("/"))
    cached = _HF_TREE_BASENAMES_CACHE.get(key)
    if cached is not None:
        return "ok", cached, ""
    status, basenames, note = _hf_basenames_under_prefix(repo_id, repo_type, sha, path_prefix)
    if status == "ok":
        _HF_TREE_BASENAMES_CACHE[key] = basenames
    return status, basenames, note


def check_hf_adjacent_file_claims(body: str) -> CheckResult:
    """Check 32 (WARN): a backtick-named data file claimed adjacent to a
    hex-pinned HF ``/tree`` markdown link must appear (by basename, any
    depth) in the scoped listing at the pinned revision.

    Incident: task #952 r1 claimed ``divergence_bank_queries.json`` at the
    pinned HF ``eval_results@5b62649`` tree while the file lives only in git
    — check 23 validated the tree link's OWN path but never read the
    adjacent prose. This check extracts anchored filename claims
    (``_gather_hf_adjacent_file_claims`` — PAREN paren-after-link /
    LINKTEXT dotted-token-in-link-text; see its docstring for the
    precision/recall trade-offs and the named recall sacrifices:
    backtick-before-link, paren-after-``/blob/``) and tests exact-BASENAME
    membership, any depth, against one bounded scoped-recursive listing per
    unique pinned prefix (``_hf_basenames_for_prefix`` → the #733 bounded
    raw tree-endpoint probe).

    Semantics:

    - **WARN, never FAIL.** A missing basename returns ``CheckResult(name,
      True, detail, is_warn=True)`` — ``passed`` stays True so overall
      ``ok`` never flips. There is NO code path returning ``passed=False``.
      Each WARN line carries its claim's shape tag (``shape: PAREN`` /
      ``shape: LINKTEXT``) for per-shape adjudication.
    - **Any-depth membership; directory matches count.** The claimed
      basename may live at any depth under the prefix (real parentheticals
      claim files one level down); an entry of EITHER type (file or
      directory) with a matching basename suppresses the WARN. Named
      residual: an any-depth basename collision can mask a wrong-PATH claim
      (the file exists, but elsewhere under the prefix) — accepted as a
      recall sacrifice at WARN tier.
    - **Fail-soft everywhere.** The offline fence
      (``EPM_VERIFY_BODY_NO_HF=1``), a missing ``huggingface_hub``, a
      429 / network error, ``not_found``, a page/time-cap hit, and the
      per-body ``_HF_MEMBER_MAX_PROBES`` cap all surface as `unverified`
      notes on a PASS line; only a SUCCESSFUL EXHAUSTIVE listing lacking
      the basename grounds a WARN. When missing and unverified claims
      coexist, the unverified list is appended to the WARN detail (never
      dropped).
    - **Probe accounting.** Unique ``(repo, sha, prefix)`` keys are probed
      AT MOST once per invocation via an intra-invocation memo; skipped
      keys count toward ``_HF_MEMBER_MAX_PROBES``. The cross-process
      ``_HF_TREE_BASENAMES_CACHE`` stores only successful exhaustive
      listings, so a later invocation re-probes a cleared throttle.

    Vacuous PASS — with ZERO Hub probes issued — when no backtick file
    claim sits adjacent to an HF tree link.
    """
    name = "HF-adjacent backtick file claims exist under the pinned tree"
    claims = _gather_hf_adjacent_file_claims(body)
    if not claims:
        return CheckResult(name, True, "no backtick file claims adjacent to HF tree links")
    missing: list[str] = []
    unverified: list[str] = []
    probe_memo: dict[tuple[str, str, str, str], tuple[str, frozenset[str], str]] = {}
    for repo_id, repo_type, sha, prefix, fname, shape in claims:
        key = (repo_id, repo_type, sha, prefix)
        if key not in probe_memo:
            if len(probe_memo) >= _HF_MEMBER_MAX_PROBES:
                cap_note = f"`{prefix or repo_id}@{sha[:8]}` (per-body probe cap)"
                if cap_note not in unverified:
                    unverified.append(cap_note)
                continue
            probe_memo[key] = _hf_basenames_for_prefix(repo_id, repo_type, sha, prefix)
        status, basenames, note = probe_memo[key]
        if status != "ok":
            skip_note = f"`{fname}` at `{prefix or repo_id}@{sha[:8]}` ({note})"
            if skip_note not in unverified:
                unverified.append(skip_note)
            continue
        if fname not in basenames:
            missing.append(
                f"body claims `{fname}` adjacent to the pinned tree "
                f"`{prefix or '/'}` at `{repo_id}@{sha[:8]}`, but no entry with "
                f"that basename exists under the prefix at that revision "
                f"(shape: {shape})"
            )
    unverified_detail = ""
    if unverified:
        unverified_detail = (
            f"; {len(unverified)} unverified (existence not confirmed): " + "; ".join(unverified)
        )
    if missing:
        return CheckResult(name, True, "; ".join(missing) + unverified_detail, is_warn=True)
    detail = f"{len(claims)} adjacent file claim(s) against {len(probe_memo)} pinned tree(s)"
    return CheckResult(name, True, detail + unverified_detail)


def check_concerns_audit(  # noqa: C901 — linear lens: ledger parse → stale-marker scan → ack scan
    body: str, *, concerns_path: Path | None = None
) -> CheckResult:
    """Lens 14 — mechanical concerns audit (binding-concerns contract,
    composed onto the 2-content-section clean-result spec on 2026-05-31
    by task #455).

    For each currently-OPEN concern in ``concerns.jsonl`` (latest event
    is ``raised`` or ``verified-open``) at severity ``BLOCKER`` or
    ``CONCERN``, FAIL the body when the concern is NOT acknowledged via
    any of:

    - **Any ``### <H3>`` result section inside ``## TL;DR``** naming the
      concern_id (substring match in that H3's body). Under the
      2-content-section spec a methodology correction folds into the
      relevant result H3's setup or read prose — there is no dedicated
      ``### Methodology corrections`` H3 (the legacy verifier's match
      target). Scanning every TL;DR result H3 covers the same intent on
      the new structure.
    - **The ``Confidence:`` rationale sentence inside ``## Reproducibility``**
      naming the concern_id (substring match in the paragraph containing
      the literal ``Confidence:`` prefix). The Confidence sentence
      migrated from the legacy ``## Details`` to ``## Reproducibility``
      under the 2-content-section spec; the scan target follows.
    - **An ``<!-- concern-deferred: <concern_id> -->`` HTML comment
      marker** anywhere in the body — records explicit user deferral
      via ``task.py defer-concern --by user``.

    NIT-severity concerns do NOT block this check; they surface as
    informational only.

    Additionally WARNs (never FAILs) on stale deferral markers: a
    ``<!-- concern-deferred: <id> -->`` whose id's latest ledger event is
    ``addressed``, or whose id is absent from the ledger (#1089, incident
    #833). A live marker (latest event raised / verified-open / deferred)
    is unchanged.

    Skipped (PASS) when ``concerns_path`` is None or missing
    (``--body-stdin`` invocations, freshly created tasks with no concerns
    ledger). Full Lens 14 fires only when invoked with ``--issue <N>``
    or when ``--file`` resolves to a sibling ``concerns.jsonl``.
    """
    if concerns_path is None or not concerns_path.exists():
        return CheckResult(
            "concerns audit (Lens 14)",
            True,
            "skipped — no concerns.jsonl sibling (file-only or pre-concerns task)",
        )

    # Mirror `task_workflow.list_concerns(open_only=True)` without
    # importing the module (verifier may run from a non-main worktree
    # where the branch-guard refuses to resolve).
    events: list[dict] = []
    # split("\n"), NOT splitlines(): raw U+2028/U+2029/NEL inside
    # ensure_ascii=False concern evidence/notes are Unicode line boundaries
    # that would shred a VALID row into fragments the per-line skip silently
    # drops — a raised BLOCKER then vanishes and check 14 falsely PASSes
    # (gotchas.md; #825 → #950).
    for line in concerns_path.read_text().split("\n"):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    latest: dict[str, dict] = {}
    for ev in events:
        cid = ev.get("concern_id")
        if cid is None:
            continue
        latest[cid] = ev
    open_binding = [
        ev
        for ev in latest.values()
        if ev.get("event") in ("raised", "verified-open")
        and ev.get("severity") in ("BLOCKER", "CONCERN")
    ]

    # Stale-marker scan (#1089) — hoisted ABOVE the no-open-binding early
    # return so it fires in the #833 shape (all concerns addressed →
    # `open_binding` empty → the old post-early-return scan never ran).
    # Acknowledgement mechanism 3's regex lives here now (moved up from
    # the ack scan below — one pattern, one definition).
    deferral_re = re.compile(r"<!--\s*concern-deferred:\s*([a-z0-9][a-z0-9-]{1,79})\s*-->")
    deferred_ids = set(deferral_re.findall(body))
    stale_warns: list[str] = []
    for cid in sorted(deferred_ids):  # sorted → deterministic detail
        ev = latest.get(cid)
        if ev is None:
            stale_warns.append(
                f"stale concern-deferred marker '{cid}' — id absent from concerns.jsonl; "
                "remove or retag"
            )
        elif ev.get("event") == "addressed":
            stale_warns.append(
                f"stale concern-deferred marker '{cid}' — concern is addressed; remove or retag"
            )
        # raised / verified-open / deferred → live marker, no WARN (unchanged
        # behavior). DELIBERATE fallthrough: a malformed/unknown `event` value
        # (hand-edited or corrupt ledger row outside CONCERN_EVENTS) is treated
        # as live — conservative no-WARN for a WARN-only check.

    if not open_binding:
        if stale_warns:
            return CheckResult(
                "concerns audit (Lens 14)",
                True,
                "; ".join(stale_warns),
                is_warn=True,
            )
        return CheckResult(
            "concerns audit (Lens 14)",
            True,
            f"no open binding concerns (read {len(events)} concern events)",
        )

    v3 = is_v3(body)
    v4 = is_v4(body)
    titletag = v3 or v4  # confidence-in-title generations with no Confidence paragraph

    # Acknowledgement mechanism 1: the result-narrative surface.
    #  - v2 / legacy: any `### <H3>` body inside `## TL;DR` (the
    #    2-content-section spec folds methodology corrections into result
    #    H3s).
    #  - v3: any `### <finding>` body inside `## Findings` PLUS the
    #    `## Takeaways` bullets.
    #  - v4: any `### <result>` body inside `## Results` PLUS the
    #    `## Takeaways` bullets (a methodology correction folds into a
    #    result's prose, and a binding caveat is named in Takeaways).
    if titletag:
        result_section = "Results" if v4 else "Findings"
        ack_parts = [
            section_text(body, "Takeaways") or "",
            section_text(body, result_section) or "",
        ]
        ack_source = "\n\n".join(ack_parts)
    else:
        ack_source = section_text(body, "TL;DR") or ""
    h3_bodies: list[str] = []
    for h3_match in re.finditer(
        r"^###\s+.+?$(.*?)(?=^###\s|\Z)",
        ack_source,
        re.MULTILINE | re.DOTALL,
    ):
        h3_bodies.append(h3_match.group(1))
    # For v3 / v4, the Takeaways bullets carry no `### ` heading, so also
    # fold the raw ack_source text in (covers Takeaways bullets + any
    # pre-heading prose).
    tldr_h3_text = "\n".join(h3_bodies) + ("\n" + ack_source if titletag else "")

    # Acknowledgement mechanism 2: the `Confidence:` rationale paragraph.
    # RETIRED for v3 / v4 bodies (confidence lives in the H1 title tag
    # only — there is no Confidence paragraph). For v2 / legacy it lives
    # in `## Reproducibility`; scan the whole body for robustness.
    if titletag:
        conf_body = ""
    else:
        conf_match = re.search(
            r"^Confidence:\s.*?(?=\n\n|\Z)",
            body,
            re.MULTILINE | re.DOTALL,
        )
        conf_body = conf_match.group(0) if conf_match else ""

    # Acknowledgement mechanism 3: explicit deferral HTML comment —
    # `deferred_ids` was computed by the stale-marker scan above (the
    # regex moved up with it, #1089).
    unaddressed: list[str] = []
    for ev in open_binding:
        cid = ev["concern_id"]
        if cid in deferred_ids:
            continue
        if cid in tldr_h3_text or cid in conf_body:
            continue
        unaddressed.append(f"{cid} ({ev.get('severity', 'unknown')})")

    if unaddressed:
        ack_hint = (
            "a `## Findings` `### <finding>` read paragraph, a `## Takeaways` bullet, "
            if v3
            else "a `## TL;DR` result H3, the `Confidence:` sentence, "
        )
        return CheckResult(
            "concerns audit (Lens 14)",
            False,
            f"{len(unaddressed)} open binding concern(s) unaddressed in body: "
            f"{', '.join(unaddressed)}. Acknowledge each in {ack_hint}"
            "or a `<!-- concern-deferred: <id> -->` HTML marker. See "
            "`.claude/agents/clean-result-critic.md` § Lens 14 "
            "and `workflow.yaml § concerns_protocol`."
            + (("; WARN: " + "; ".join(stale_warns)) if stale_warns else ""),
        )
    if stale_warns:  # all acknowledged, but stale deferral markers remain (#1089)
        return CheckResult(
            "concerns audit (Lens 14)",
            True,
            "; ".join(stale_warns),
            is_warn=True,
        )
    return CheckResult(
        "concerns audit (Lens 14)",
        True,
        f"all {len(open_binding)} open binding concern(s) acknowledged in body",
    )


# ─── v3 Data-section checks (18 / 19) ────────────────────────────────────────

# An explicit `n/a — <reason>` line that a `## Data` subsection may use
# in place of a pinned link (e.g. `### Trained on` for an eval-only task).
# Accepts en-dash, em-dash, or ASCII hyphen and ≥3 chars of reason.
_DATA_NA_RE = re.compile(r"(?im)^\s*>?\s*n/?a\s*[—–-]\s*\S.{2,}")  # noqa: RUF001

# Any pinned absolute artifact link inside a `## Data` subsection that
# points at the COMPLETE artifact: an `https://` URL pinned to a ref (HF
# Hub `/tree|/blob/<sha>` or `@<sha>`, WandB `/runs/<id>`, GitHub
# `/blob|/tree/<sha>`, raw.githubusercontent `<sha>`). Reuses the
# permanence shape the Reproducibility checks already trust — here we
# only need "is there ≥1 pinned link", not full validation.
_PINNED_LINK_RE = re.compile(
    r"https?://[^\s)]*"
    r"(?:/tree/[0-9a-fA-F]{7,40}|/blob/[0-9a-fA-F]{7,40}|@[0-9a-fA-F]{7,40}"
    r"|raw\.githubusercontent\.com/[^\s)]+/[0-9a-fA-F]{7,40}/"
    r"|wandb\.ai/[^\s)]+/runs/[^\s)]+)"
)

# Subset-disclosure forms accepted by check 19 — the Datasheets
# sample-to-population field. Extends the checks 10/11 cherry-picked
# pattern with the harmful-content sanitized form so v3 `## Data`
# example blocks built from EM / bad-medical-advice corpora pass.
_SUBSET_DISCLOSURE_RE = re.compile(
    r"\b(?:cherry[-\s]?picked|random[-\s]?sample|drawn at random|random draw|"
    r"first \d+ of \d+|\d+ of \d+ rows|\d+ of [\d,]+ rows|"
    r"\d+\s+(?:examples?|sample[s]?|example rows?)|"
    # harmful-content carve-out — the sanitized excerpt form
    r"sanitized for context hygiene|harmful-content row|truncated — harmful)",
    re.IGNORECASE,
)

# Project-internal condition / cell / plan-tag codes that
# `audit_clean_results_body_discipline.py` flags as body-discipline
# anti-patterns. KEPT IN SYNC with that script's `condition_labels` +
# `cell_tags` PATTERNS entries (verbatim source). The audit EXEMPTS
# example blocks inside `## Data` (v3) / `## Methodology` (v4) only when
# they are wrapped in a fenced code block (stripped globally) or a
# `<details>` block (`strip_data_example_blocks`); a verbatim example
# row placed as a BARE inline GFM table is NOT exempt, so a `C1` /
# `H2` / `BS_E0` cell trips
# the audit's condition-code scan with a spurious FAIL at /issue Step
# 9a-bis and no signal telling the author to wrap it. Check 19b (WARN
# only) fires at authoring time precisely when such a cell exists in an
# unwrapped `## Data` (v3) / `## Methodology` (v4) table, nudging the
# author to wrap the row in `<details>` or a fenced block BEFORE the
# confusing downstream audit FAIL. Scoped to a cell that WOULD match the
# audit (not "any unwrapped table") so a benign composition / row-count
# summary table never WARNs.
_DATA_CONDITION_CODE_RE = re.compile(
    # condition_labels: C1/C2, H1/H2/H3, P1/P2/P3 (optional prime)
    r"\b[CcHhP][1-9](?:'|′)?(?:\s*(?:condition|control|completion|coefficient|"  # noqa: RUF001
    r"hypothesis|test|sub-?(?:claim|experiment|hypothesis)))?(?![a-zA-Z0-9_])"
    # cell_tags: BS_E*, Z_*, G*, Method A/B, M1-paired
    r"|\bBS_E[0-9A-Za-z_]*|\bZ_[a-zA-Z_]+|\b[Gg][0-9]+[a-c]?\b(?=\s|:|\.|,|$)"
    r"|\bMethod\s+[AB]\b|\b[Mm][1-9]\b(?=\s+(?:cosine|cell|mean|extraction|"
    r"method|sub-experiment))"
)

# A SINGLE-column GFM delimiter row (`|---|` / `| :--- |` / `---`).
# `_GFM_DELIM_RE` requires ≥2 columns; this catches the one-column form so
# a bare single-column `## Data` table (which the audit's line-based scan
# WOULD FAIL on) is still recognized by check 19b. Kept separate from
# `_GFM_DELIM_RE` to avoid loosening that constant for its other callers.
_SINGLE_COL_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*\|?\s*$")


def _iter_unwrapped_data_tables(data: str) -> list[str]:
    """Return the cell text of every GFM table inside the given section
    text (v3 `## Data` / v4 `## Methodology`) that is NOT wrapped in a
    `<details>` block and NOT inside a fenced code block.

    The v3 spec carries verbatim example rows in `## Data` as fenced OR
    `<details>` table blocks; `audit_clean_results_body_discipline.py`
    exempts exactly those two forms from its condition-code scan. A bare
    inline GFM table (no fence, no `<details>`) is the off-spec form the
    audit will flag. This helper isolates that bare-table cell text by
    stripping the two exempt forms first (mirroring the audit's
    `strip_code` + `strip_data_example_blocks`), then collecting the
    cells of any remaining table — a header row + a delimiter row, where
    the delimiter may be multi-column (`|---|---|`) OR single-column
    (`|---|`), so single-column data tables are not a blind spot.
    """
    # Strip `<details>...</details>` blocks (the wrapped example form).
    stripped = _DETAILS_BLOCK_RE.sub("", data)
    # Walk lines, dropping fenced code blocks, collecting GFM-table cell
    # text from what remains. A table is a contiguous run of `|`-rows that
    # contains at least one delimiter row. `_GFM_DELIM_RE` requires ≥2
    # columns; the audit's condition-code scan is line-based with NO
    # column requirement, so a SINGLE-column data table (`| C1 |` /
    # `|---|` / ...) would FAIL the audit while escaping a ≥2-col-only
    # detector. `_SINGLE_COL_DELIM_RE` adds the lone-column delimiter form
    # so the WARN/audit sync holds at both column counts.
    cells: list[str] = []
    in_fence = False
    run: list[str] = []
    has_delim = False

    def _is_delim(line: str) -> bool:
        return bool(_GFM_DELIM_RE.match(line) or _SINGLE_COL_DELIM_RE.match(line))

    def _flush() -> None:
        nonlocal has_delim
        if has_delim:
            for row in run:
                if _is_delim(row):
                    continue
                for cell in row.strip().strip("|").split("|"):
                    cells.append(cell.strip())
        run.clear()
        has_delim = False

    for line in stripped.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            _flush()
            continue
        if in_fence:
            continue
        if s.startswith("|"):
            run.append(line)
            if _is_delim(line):
                has_delim = True
        else:
            _flush()
    _flush()
    return cells


def _unwrapped_condition_code_hits(section: str) -> list[str]:
    """Sorted unique `_DATA_CONDITION_CODE_RE` matches inside the bare
    (unwrapped) GFM-table cells of `section` — the cells
    `_iter_unwrapped_data_tables` isolates."""
    cells = _iter_unwrapped_data_tables(section)
    return sorted({m.group(0) for c in cells for m in _DATA_CONDITION_CODE_RE.finditer(c)})


def check_data_unwrapped_example_table(body: str) -> CheckResult:
    """Check 19b (v3 + v4, WARN): a verbatim example row placed in the
    generation's example-bearing section (`## Data` on v3,
    `## Methodology` on v4) as a BARE inline GFM table — not wrapped in
    `<details>` and not in a fenced code block — that carries a
    project-internal condition / cell code (`C1`, `H2`, `BS_E0`,
    `Method A`, …) can trip `audit_clean_results_body_discipline.py`'s
    condition-code scan with a spurious FAIL at /issue Step 9a-bis: the
    audit exempts the fenced and `<details>` example forms (in v3
    `## Data` AND v4 `## Methodology`, #1171), but not the bare table —
    and `cell_tags`-family codes are additionally NOT table-blanked, so
    they FAIL in ANY bare table. This WARN fires at authoring time so
    the author wraps the rows (or rewords a plan-internal code in a
    hyperparameter-table cell) BEFORE the confusing downstream audit
    FAIL.

    WARN only — never FAIL. Scoped to a table cell that WOULD match the
    audit's condition-code patterns (not "any unwrapped table") so a
    benign composition / row-count / hyperparameter summary table does
    not WARN. v4 is checked BEFORE v3 (the sentinel declares the
    governing spec — the same precedence the audit's section scoping
    uses). PASSes vacuously on v2 / legacy bodies.
    """
    if is_v4(body):
        label = "Methodology unwrapped example table (v4)"
        meth = section_text(body, "Methodology")
        if meth is None:
            return CheckResult(label, True, "## Methodology missing — check 2 will report")
        hits = _unwrapped_condition_code_hits(meth)
        if hits:
            preview = ", ".join(f"`{h}`" for h in hits[:4]) + (" …" if len(hits) > 4 else "")
            return CheckResult(
                label,
                True,
                f"a bare inline `## Methodology` table carries condition-code cell(s) "
                f"({preview}) — wrap the verbatim example rows in a `<details>` block or "
                "a fenced code block (or reword a plan-internal code in a "
                "hyperparameter-table cell — the required Training table stays bare), "
                "else the body-discipline audit (Step 9a-bis) can FAIL on them as "
                "project-internal condition codes",
                is_warn=True,
            )
        return CheckResult(
            label, True, "no unwrapped `## Methodology` example table with condition codes"
        )
    label = "Data unwrapped example table (v3)"
    if not is_v3(body):
        return CheckResult(label, True, "skipped — not a v3 body")
    data = section_text(body, "Data")
    if data is None:
        return CheckResult(label, True, "## Data missing — check 2 will report")
    hits = _unwrapped_condition_code_hits(data)
    if hits:
        preview = ", ".join(f"`{h}`" for h in hits[:4]) + (" …" if len(hits) > 4 else "")
        return CheckResult(
            label,
            True,
            f"a bare inline `## Data` table carries condition-code cell(s) ({preview}) — "
            "wrap the verbatim example rows in a `<details>` block or a fenced code block, "
            "else the body-discipline audit (Step 9a-bis) FAILs on them as project-internal "
            "condition codes",
            is_warn=True,
        )
    return CheckResult(label, True, "no unwrapped `## Data` example table with condition codes")


def check_data_shape(body: str) -> CheckResult:
    """Check 18 (v3 only): `## Data` carries `### Trained on` /
    `### Evaluated with` / `### Generated` in order, and each block
    contains ≥1 pinned link to the complete artifact OR an explicit
    `n/a — <reason>` line.

    PASSes vacuously on v2 / legacy bodies (no v3 sentinel).
    """
    label = "Data section shape (v3)"
    if not is_v3(body):
        return CheckResult(label, True, "skipped — not a v3 body")
    data = section_text(body, "Data")
    if data is None:
        # check_required_sections already FAILs on a missing `## Data`;
        # don't double-report.
        return CheckResult(label, True, "## Data missing — check 2 will report")
    h3s = _collect_tldr_h3_names(data)
    h3_heads = [
        re.split(r"\s+[–—:]\s+", re.sub(r"\s+", " ", name).strip(), maxsplit=1)[0].casefold()  # noqa: RUF001
        for name, _line in h3s
    ]
    required = [s.casefold() for s in V3_DATA_SUBSECTIONS]
    missing = [V3_DATA_SUBSECTIONS[i] for i, r in enumerate(required) if r not in h3_heads]
    if missing:
        return CheckResult(
            label,
            False,
            f"## Data is missing required subsection(s): "
            f"{', '.join('### ' + m for m in missing)}. The v3 shape requires "
            f"`### Trained on` → `### Evaluated with` → `### Generated` in order.",
        )
    # Order check on the subset of H3s that are required subsections.
    seq = [h for h in h3_heads if h in required]
    if seq != required:
        return CheckResult(
            label,
            False,
            f"## Data subsections out of order — got {seq}, expected {required} "
            "(`### Trained on` → `### Evaluated with` → `### Generated`).",
        )
    # Per-subsection: ≥1 pinned link OR an explicit `n/a — <reason>` line.
    no_link: list[str] = []
    for sub in V3_DATA_SUBSECTIONS:
        sub_text = _h3_subsection_text(data, sub)
        if sub_text is None:
            # Already handled by the missing/order checks above; defensive.
            no_link.append(sub)
            continue
        if _PINNED_LINK_RE.search(sub_text) or _DATA_NA_RE.search(sub_text):
            continue
        no_link.append(sub)
    if no_link:
        return CheckResult(
            label,
            False,
            f"## Data subsection(s) {', '.join('### ' + s for s in no_link)} carry "
            "no pinned link to the complete artifact and no explicit `n/a — <reason>` "
            "line. Each block must link the full artifact (HF Hub `/tree/<sha>`, "
            "WandB `/runs/<id>`, GitHub `/blob/<sha>`) or state `n/a — <reason>`.",
        )
    return CheckResult(
        label,
        True,
        "## Data has Trained on / Evaluated with / Generated in order, each with a "
        "complete-artifact link or `n/a — <reason>`",
    )


def check_data_subset_disclosure(body: str) -> CheckResult:
    """Check 19 (v3 only): every example block (fenced OR `<details>`)
    inside `## Data` is immediately preceded by a subset-disclosure line
    — `K of M rows, random sample` / `cherry-picked for illustration` /
    the harmful-content sanitized form. Extends the checks-10/11 pattern
    to `## Data`.

    PASSes vacuously on v2 / legacy bodies (no v3 sentinel).
    """
    label = "Data subset-disclosure (v3)"
    if not is_v3(body):
        return CheckResult(label, True, "skipped — not a v3 body")
    data = section_text(body, "Data")
    if data is None:
        return CheckResult(label, True, "## Data missing — check 2 will report")
    samples = _iter_sample_blocks(data)
    if not samples:
        return CheckResult(label, True, "no example blocks in `## Data` (fenced or `<details>`)")
    flagged: list[str] = []
    for start, _, content in samples:
        prelude = _prelude_window(data, start)
        if _SUBSET_DISCLOSURE_RE.search(prelude) or _SUBSET_DISCLOSURE_RE.search(content):
            continue
        first_line = content.strip().splitlines()[0][:60] if content.strip() else "(empty)"
        flagged.append(first_line)
    if flagged:
        preview = "; ".join(f"'{x}'" for x in flagged[:2]) + (" …" if len(flagged) > 2 else "")
        return CheckResult(
            label,
            False,
            f"{len(flagged)} of {len(samples)} `## Data` example block(s) lack a "
            f"subset-disclosure line (`K of M rows, random sample` / `cherry-picked "
            f"for illustration` / sanitized-harmful form): {preview}",
        )
    return CheckResult(label, True, f"{len(samples)} `## Data` example block(s) disclosed")


# ─── v3 word-cap check (20) ──────────────────────────────────────────────────


def _prose_words(text: str) -> int:
    """Count words in `text` AFTER stripping the surfaces the caps
    deliberately exclude: fenced code blocks, `<details>...</details>`
    bodies, GFM table rows (lines starting with `|`), and blockquote
    caption lines (lines starting with `>`). What remains is the
    scannable prose the caps govern.
    """
    # Strip <details> blocks first (they may contain tables / fences).
    text = _DETAILS_BLOCK_RE.sub("", text)
    out_lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if s.startswith("|"):  # GFM table row
            continue
        if s.startswith(">"):  # blockquote caption
            continue
        out_lines.append(line)
    prose = "\n".join(out_lines)
    return len(prose.split())


def _count_extra_followup_rounds(body: str) -> int:
    """Heuristic count of LIVE follow-up rounds beyond the first, read
    off a `## What I ran` Rounds table (one data row per round). Returns
    0 when no Rounds table is present (single-round body). Used only to
    scale the WARN-only total-prose budget."""
    what = section_text(body, "What I ran")
    if what is None:
        return 0
    # Find a markdown table whose header row mentions "round".
    data_rows = 0
    in_round_table = False
    for line in what.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            in_round_table = False
            continue
        if "round" in s.casefold() and not in_round_table:
            # Header row of a rounds table; the next line should be the
            # GFM delimiter, then data rows.
            in_round_table = True
            continue
        if in_round_table:
            if _GFM_DELIM_RE.match(line):
                continue
            data_rows += 1
    return max(0, data_rows - 1)


# One clause per FOLDED same-issue follow-up round in the v4 footer
# (SPEC.md § `**Context:**` row: rounds name each round's `followup_label`;
# corpus forms: "same-issue follow-up round `<label>`" (#763/#811/#667) and
# the sentence-initial numbered variant "Same-issue follow-up round 2
# (label: `<label>`, ...)" (#685 footer) — hence IGNORECASE + the optional
# `<n> (label: ` infix. The `(?!s)` lookahead keeps generic plural prose
# ("follow-up rounds also name...") out of the count; an unlabeled singular
# clause still counts 1.
_V4_FOOTER_ROUND_CLAUSE_RE = re.compile(
    r"same-issue follow-up round(?!s)"
    r"(?:\s+(?:\d+\s*\(label:\s*)?`(?P<label>[^`\s]+)`)?",
    re.IGNORECASE,
)


def _followup_run_marker_rounds(issue: int) -> int:
    """Count REAL folded same-issue follow-up rounds off the task's
    events.jsonl `epm:same-issue-followup-run` markers: distinct
    `followup_label`s (one run marker closes a label's round) plus one per
    unlabeled run marker, EXCLUDING markers whose `outcome` begins
    `retroactive-close` — bookkeeping closes of ghost labels that folded
    no new prose (they are likewise excluded from the /issue round caps,
    SKILL.md). Returns 0 when the task id does not resolve (a plain
    FileNotFoundError — e.g. a fixture body under a numeric tmp dir); a
    `StaleTaskPathError` (registry corruption, a FileNotFoundError
    SUBCLASS) still propagates — that is real corruption the gate should
    surface, unlike an unknown id."""
    from explore_persona_space.task_workflow import (  # local import — matches _load_text_for_issue
        FOLLOWUP_RUN_KIND,
        StaleTaskPathError,
        list_events,
        parse_followup_note_field,
    )

    try:
        events = list_events(issue)
    except StaleTaskPathError:
        raise
    except FileNotFoundError:
        return 0
    labels: set[str] = set()
    unlabeled = 0
    for ev in events:
        if ev.get("kind") != FOLLOWUP_RUN_KIND:
            continue
        note = ev.get("note") or ""
        # `parse_followup_note_field` parses `; `-joined mid-line fields
        # directly as of #1111, but the SPACE-separated single-line form
        # (`... round: 1 outcome: X` — #763's `outcome:` is mid-line with
        # space separation, which the semicolon split deliberately does
        # not cover) still parses None. Mid-line regex fallback RETAINED
        # so a space-separated single-line retro-close marker cannot
        # evade the exclusion.
        outcome = parse_followup_note_field(note, "outcome") or ""
        if not outcome:
            m = re.search(r"(?:^|\s)outcome:\s*(\S+)", note)
            outcome = m.group(1) if m else ""
        if outcome.startswith("retroactive-close"):
            continue
        label = parse_followup_note_field(note, "followup_label")
        if label:
            labels.add(label)
        else:
            unlabeled += 1
    return len(labels) + unlabeled


def _count_extra_followup_rounds_v4(body: str, issue: int | None = None) -> tuple[int, str]:
    """v4 twin of `_count_extra_followup_rounds` (whose `## What I ran`
    Rounds-table read only binds v3 — v4 bodies always scored rounds=0,
    incident #763/#921). Two signals, max-reconciled — the budget is
    WARN-only and monotone-up, and each signal under-counts in a known
    case (footer: a body omitting the SPEC round clause, e.g. #685;
    events: a legacy pre-marker round whose prose IS folded, e.g. #811):

    - footer: `same-issue follow-up round [`label`]` clauses inside the
      `**Repro:**`/`**Context:**` footer (distinct backticked labels +
      one per unlabeled singular clause);
    - events (when `issue` is known): non-retroactive-close
      `epm:same-issue-followup-run` markers via
      `_followup_run_marker_rounds`.

    Returns `(count, source)`; `source` is one of `none` / `footer` /
    `events` / `footer+events` and names the winning signal for the WARN
    message. Bare-file residual: in `--body-stdin` / bare `--file` mode
    (no issue id) the events leg is unavailable, so a footer-less
    multi-round body scores 0 and keeps the base budget."""
    footer = _v4_footer_text(body) or ""
    labels: set[str] = set()
    unlabeled = 0
    for m in _V4_FOOTER_ROUND_CLAUSE_RE.finditer(footer):
        if m.group("label"):
            labels.add(m.group("label"))
        else:
            unlabeled += 1
    footer_n = len(labels) + unlabeled
    events_n = _followup_run_marker_rounds(issue) if issue is not None else 0
    count = max(footer_n, events_n)
    if count == 0:
        return 0, "none"
    if footer_n == events_n:
        return count, "footer+events"
    return count, "footer" if footer_n > events_n else "events"


def _count_overlong_takeaways_bullets(takeaways: str) -> tuple[int, int]:
    """Count top-level Takeaways bullets over the per-bullet word caps
    (fence-aware). Returns ``(over_warn, over_fail)``: bullets over the
    30-word WARN cap but under the >=100-word FAIL tier, and bullets at
    or over the FAIL tier. v3 reports the SUM as its WARN count
    (WARN-only, grandfathered); v4 FAILs the second bucket (#825).
    Helper for check 20."""
    in_fence = False
    over_warn = 0
    over_fail = 0
    for line in takeaways.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^[-*]\s+\S", line):
            wc = len(re.sub(r"^[-*]\s+", "", line.strip()).split())
            if wc >= V4_TAKEAWAYS_BULLET_FAIL_WORDS:
                over_fail += 1
            elif wc > V3_TAKEAWAYS_BULLET_MAX_WORDS:
                over_warn += 1
    return over_warn, over_fail


def _finding_prose_cap_results(findings: str) -> tuple[list[str], list[str]]:
    """Return (fail_msgs, warn_msgs) for the per-finding prose caps.
    Each `### <finding>` block's prose (excl. tables / fenced code /
    `<details>` bodies / captions, via `_prose_words`) is FAILed at the
    hard cap and WARNed at the soft cap. Helper for check 20."""
    fails: list[str] = []
    warns: list[str] = []
    finding_h3s = _collect_tldr_h3_names(findings)
    flines = findings.splitlines()
    for idx, (name, line_no) in enumerate(finding_h3s):
        end_line = finding_h3s[idx + 1][1] if idx + 1 < len(finding_h3s) else len(flines)
        block = "\n".join(flines[line_no + 1 : end_line])
        wc = _prose_words(block)
        short = name[:48]
        if wc >= V3_FINDING_PROSE_FAIL_WORDS:
            fails.append(f"finding '{short}' prose is {wc} words (≥{V3_FINDING_PROSE_FAIL_WORDS})")
        elif wc > V3_FINDING_PROSE_WARN_WORDS:
            warns.append(f"finding '{short}' prose is {wc} words (>{V3_FINDING_PROSE_WARN_WORDS})")
    return fails, warns


def _count_overlong_captions(findings: str) -> int:
    """Count figure captions (consecutive blockquote `>` runs,
    fence-aware) over the caption word cap. Helper for check 20."""
    over = 0
    cap_run: list[str] = []

    def _flush() -> None:
        nonlocal over
        if cap_run:
            txt = " ".join(re.sub(r"^>\s?", "", ln) for ln in cap_run)
            # Strip bold "Figure." label + italics markers for the count.
            wc = len(re.sub(r"[*_]", "", txt).split())
            if wc > V3_FIGURE_CAPTION_MAX_WORDS:
                over += 1

    in_fence = False
    for line in findings.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            cap_run = []
            continue
        if in_fence:
            continue
        if s.startswith(">"):
            cap_run.append(s)
        else:
            _flush()
            cap_run = []
    _flush()
    return over


def check_v3_word_caps(body: str) -> CheckResult:
    """Check 20 (v3 only): the §4 conciseness caps.

    - Per-Takeaways-bullet ≤30 words (WARN).
    - Per-finding prose ≤120 words WARN / ≥180 words FAIL (excl.
      caption / fenced code / `<details>` bodies / table rows).
    - Figure caption ≤60 words (WARN).
    - Total content prose (Takeaways + What I ran + Findings) ≤800 +
      250 per live follow-up round beyond the first (WARN-only;
      calibrated on the four-finding #517 v3 conversion — see the
      V3_TOTAL_PROSE_BASE_WORDS constant comment).

    The Takeaways 3-6 bullet COUNT is owned by `check_v3_structure`
    (one authoritative count gate), not duplicated here. A FAIL here
    fires ONLY on the per-finding ≥180-word hard cap; everything else
    is WARN (the v4-only ≥100-word per-bullet FAIL tier does not bind
    v3). PASSes vacuously on v2 / legacy bodies.
    """
    label = "v3 conciseness caps"
    if not is_v3(body):
        return CheckResult(label, True, "skipped — not a v3 body")

    warns: list[str] = []
    fails: list[str] = []

    takeaways = section_text(body, "Takeaways") or ""
    findings = section_text(body, "Findings") or ""

    # Per-Takeaways-bullet word cap (WARN-only on v3 — grandfathered; the
    # v4-only >=100-word FAIL tier does not bind here).
    over_warn_b, over_fail_b = _count_overlong_takeaways_bullets(takeaways)
    over_bullets = over_warn_b + over_fail_b
    if over_bullets:
        warns.append(
            f"{over_bullets} Takeaways bullet(s) exceed {V3_TAKEAWAYS_BULLET_MAX_WORDS} words"
        )

    # Per-finding prose caps (WARN at soft, FAIL at hard).
    f_fails, f_warns = _finding_prose_cap_results(findings)
    fails.extend(f_fails)
    warns.extend(f_warns)

    # Figure-caption word cap (WARN).
    over_captions = _count_overlong_captions(findings)
    if over_captions:
        warns.append(
            f"{over_captions} figure caption(s) exceed {V3_FIGURE_CAPTION_MAX_WORDS} words"
        )

    # Total content prose (WARN-only), scaled per live follow-up round.
    extra_rounds = _count_extra_followup_rounds(body)
    total_budget = V3_TOTAL_PROSE_BASE_WORDS + extra_rounds * V3_TOTAL_PROSE_PER_EXTRA_ROUND_WORDS
    total_prose = (
        _prose_words(takeaways)
        + _prose_words(section_text(body, "What I ran") or "")
        + _prose_words(findings)
    )
    if total_prose > total_budget:
        warns.append(
            f"total content prose is {total_prose} words (budget {total_budget}: "
            f"{V3_TOTAL_PROSE_BASE_WORDS} + {extra_rounds} x "
            f"{V3_TOTAL_PROSE_PER_EXTRA_ROUND_WORDS} per extra round)"
        )

    if fails:
        return CheckResult(
            label,
            False,
            "; ".join(fails) + (("; WARN: " + "; ".join(warns)) if warns else ""),
        )
    if warns:
        return CheckResult(label, True, "; ".join(warns), is_warn=True)
    return CheckResult(label, True, "all v3 conciseness caps satisfied")


# ─── v4 checks (Methodology shape, word caps, Results beat) ───────────────────


def check_v4_methodology_shape(body: str) -> CheckResult:
    """Check 18 (v4 only): `## Methodology` completeness.

    Two requirements:
      (a) the `**Training:**` slot carries the complete hyperparameter
          table — at least ONE GFM table delimiter row after the
          `**Training:**` label — OR the explicit
          `**N/A — no model training**` marker (analysis-only tasks);
      (b) the `**Sample training/evaluation data + completions:**` slot
          carries ≥1 example block (fenced OR `<details>`) AND that slot
          carries ≥1 pinned complete-artifact link OR an explicit
          `n/a — <reason>` line.

    PASSes vacuously on v3 / v2 / legacy bodies.
    """
    label = "Methodology completeness (v4)"
    if not is_v4(body):
        return CheckResult(label, True, "skipped — not a v4 body")
    methodology = section_text(body, "Methodology")
    if methodology is None:
        # check_required_sections already FAILs on a missing `## Methodology`.
        return CheckResult(label, True, "## Methodology missing — check 2 will report")
    # (a) Training table OR the no-training marker.
    has_no_training = re.search(
        r"(?im)\*\*\s*N/?A\s*[—–-]\s*no model training",  # noqa: RUF001
        methodology,
    )
    if not has_no_training:
        # Find the `**Training:**` slot text, bounded by the next slot label.
        tm = re.search(
            r"(?im)^\s*[-*]?\s*\*\*\s*Training\s*:?\s*\*\*",
            methodology,
        )
        training_text = ""
        if tm is not None:
            after = methodology[tm.end() :]
            nxt = re.search(
                r"(?im)^\s*[-*]?\s*\*\*\s*"
                r"(?:Evaluation|Data extraction|Sample training/evaluation data)\s*:?\s*\*\*",
                after,
            )
            training_text = after[: nxt.start()] if nxt is not None else after
        if not _GFM_DELIM_RE.search(training_text):
            return CheckResult(
                label,
                False,
                "## Methodology `**Training:**` slot carries no hyperparameter table "
                "(no GFM table delimiter row found between `**Training:**` and the next "
                "slot). The v4 Methodology Training slot must contain the COMPLETE "
                "hyperparameter table (every training + eval + generation knob, with a "
                "Source column), OR the explicit `**N/A — no model training**` marker.",
            )
    # (b) Sample slot: ≥1 example block + ≥1 pinned link OR explicit n/a.
    sample = _v4_methodology_sample_slot(methodology)
    if sample is None:
        return CheckResult(
            label,
            False,
            "## Methodology is missing the `**Sample training/evaluation data + "
            "completions:**` slot — the v4 shape requires it (verbatim worked "
            "examples, each subset-disclosed + linked to the full artifact).",
        )
    if not (_PINNED_LINK_RE.search(sample) or _DATA_NA_RE.search(sample)):
        return CheckResult(
            label,
            False,
            "## Methodology `**Sample ...:**` slot carries no pinned complete-artifact "
            "link and no explicit `n/a — <reason>` line. Link the full training mix / "
            "probe bank / raw completions (HF Hub `/tree/<sha>`, GitHub `/blob/<sha>`) "
            "or state `n/a — <reason>`.",
        )
    return CheckResult(
        label,
        True,
        "## Methodology has the Training hyperparameter table (or no-training marker) "
        "and a Sample slot with a complete-artifact link",
    )


def check_v4_word_caps(body: str, *, issue: int | None = None) -> CheckResult:
    """Check 20 (v4 only): the v4 conciseness caps (same constants as v3,
    plus the v4-only per-Takeaways-bullet FAIL tier).

    - Per-Takeaways-bullet ≤30 words (WARN); ≥100 words FAIL (v4-only
      hard tier, #825).
    - Per-`### <result>` prose ≤120 words WARN / ≥180 words FAIL (excl.
      caption / fenced code / `<details>` bodies / table rows).
    - Figure caption ≤60 words (WARN).
    - Total content prose (Takeaways + Goal + Results; `## Methodology`
      EXCLUDED — it carries the absorbed methodology-doc content) ≤800 +
      250 per live follow-up round beyond the first (WARN-only).

    The per-extra-round scaling counts folded rounds from the task's
    non-retroactive `epm:same-issue-followup-run` markers (via ``issue``,
    when known) and/or the footer's round clauses (max), NOT the v3
    Rounds table (#921). The Takeaways 3-6 bullet COUNT is owned by
    `check_v4_structure`. A FAIL here fires on the per-result ≥180-word
    hard cap and the per-Takeaways-bullet ≥100-word tier. PASSes
    vacuously on v3 / v2 / legacy bodies.
    """
    label = "v4 conciseness caps"
    if not is_v4(body):
        return CheckResult(label, True, "skipped — not a v4 body")

    warns: list[str] = []
    fails: list[str] = []

    takeaways = section_text(body, "Takeaways") or ""
    # Use the footer-truncated Results body so the `**Repro:**`/`**Context:**`
    # footer prose is NOT mis-attributed to the last result's interpretation
    # (would inflate the per-result word count into a false ≥180 hard FAIL).
    results = _v4_results_body(body) or ""

    # Per-Takeaways-bullet caps: >=100 words is a v4-only hard FAIL (#825 —
    # an accreted paragraph-bullet must not ride the 30-word WARN); the
    # 31-99-word band keeps the existing WARN. Mutually exclusive tiers.
    over_warn_b, over_fail_b = _count_overlong_takeaways_bullets(takeaways)
    if over_fail_b:
        fails.append(
            f"{over_fail_b} Takeaways bullet(s) at ≥{V4_TAKEAWAYS_BULLET_FAIL_WORDS} words "
            "(accreted paragraph-bullet — split or tighten)"
        )
    if over_warn_b:
        warns.append(
            f"{over_warn_b} Takeaways bullet(s) exceed {V3_TAKEAWAYS_BULLET_MAX_WORDS} words"
        )

    # Per-result prose caps (reuse the per-finding helper; it scans `### `
    # H3 blocks, which `## Results` carries identically).
    r_fails, r_warns = _finding_prose_cap_results(results)
    fails.extend(r.replace("finding", "result") for r in r_fails)
    warns.extend(r.replace("finding", "result") for r in r_warns)

    over_captions = _count_overlong_captions(results)
    if over_captions:
        warns.append(
            f"{over_captions} figure caption(s) exceed {V3_FIGURE_CAPTION_MAX_WORDS} words"
        )

    # Total content prose (WARN-only): Takeaways + Goal + Results. The
    # `## Methodology` section is EXCLUDED — it absorbed the entire former
    # standalone methodology doc and is reference, not skim prose.
    extra_rounds, rounds_src = _count_extra_followup_rounds_v4(body, issue)
    total_budget = V3_TOTAL_PROSE_BASE_WORDS + extra_rounds * V3_TOTAL_PROSE_PER_EXTRA_ROUND_WORDS
    total_prose = (
        _prose_words(takeaways)
        + _prose_words(section_text(body, "Goal") or "")
        + _prose_words(results)
    )
    if total_prose > total_budget:
        warns.append(
            f"total content prose is {total_prose} words (budget {total_budget}: "
            f"{V3_TOTAL_PROSE_BASE_WORDS} + {extra_rounds} x "
            f"{V3_TOTAL_PROSE_PER_EXTRA_ROUND_WORDS} per extra round "
            f"[{rounds_src}]; Methodology excluded)"
        )

    if fails:
        return CheckResult(
            label,
            False,
            "; ".join(fails) + (("; WARN: " + "; ".join(warns)) if warns else ""),
        )
    if warns:
        return CheckResult(label, True, "; ".join(warns), is_warn=True)
    return CheckResult(label, True, "all v4 conciseness caps satisfied")


def _v4_beat_has_prose(seg: list[str]) -> bool:
    """A prose line is any non-blank line that is not an image, a
    blockquote caption, a table row, a fence, a heading, or an HTML tag
    (e.g. `<details>`). Helper for `check_v4_results_beat`."""
    for ln in seg:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("!", ">", "|", "```", "~~~", "#", "<")):
            continue
        return True
    return False


def _v4_first_image_index(block_lines: list[str]) -> int | None:
    """Return the index of the first inline-image line in a `### <result>`
    block (fence-aware), or None when the result carries no figure."""
    in_fence = False
    for j, ln in enumerate(block_lines):
        s = ln.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _IMAGE_RE.search(ln):
            return j
    return None


def _v4_result_beat_gaps(name: str, block_lines: list[str]) -> str | None:
    """Return a one-line description of the missing beat(s) for a single
    `### <result>` block, or None when the result is fully framed (or is
    a figure-less qualitative result, which is exempt)."""
    img_idx = _v4_first_image_index(block_lines)
    if img_idx is None:
        # Figure-less qualitative result — exempt (clean-result-critic judges).
        return None
    above = _v4_beat_has_prose(block_lines[:img_idx])
    # Below: skip the leading blank + the contiguous `>` caption run, then
    # look for interpretation prose.
    below_seg = block_lines[img_idx + 1 :]
    k = 0
    while k < len(below_seg) and below_seg[k].strip() == "":
        k += 1
    while k < len(below_seg) and below_seg[k].strip().startswith(">"):
        k += 1
    below = _v4_beat_has_prose(below_seg[k:])
    if above and below:
        return None
    missing_what = []
    if not above:
        missing_what.append("what-is-plotted prose above the figure")
    if not below:
        missing_what.append("interpretation prose below the caption")
    return f"'{name[:48]}' missing {', '.join(missing_what)}"


def check_v4_results_beat(body: str) -> CheckResult:
    """Check 21 (v4 only, WARN): each `### <result>` follows the three-beat
    shape — what-is-plotted prose ABOVE the figure and interpretation
    prose BELOW the caption.

    WARN (not FAIL) so a legitimately figure-less qualitative result is
    not blocked; the clean-result-critic owns the substantive beat read.
    For each `### <result>` that DOES carry an inline figure, this check
    WARNs when there is no prose line above the figure (within the result)
    OR no prose line below the caption — the chart-pasted-without-framing
    signal. PASSes vacuously on v3 / v2 / legacy bodies.
    """
    label = "Results three-beat shape (v4)"
    if not is_v4(body):
        return CheckResult(label, True, "skipped — not a v4 body")
    # Footer-truncated Results: otherwise the footer's `**Repro:**` line
    # reads as "interpretation prose below the caption" for the LAST result,
    # silently defeating the missing-interpretation-beat WARN.
    results = _v4_results_body(body)
    if results is None:
        return CheckResult(label, True, "## Results missing — check 2 will report")
    result_h3s = _collect_tldr_h3_names(results)
    if not result_h3s:
        return CheckResult(label, True, "no `### <result>` headings — check 3 will report")
    rlines = results.splitlines()
    flagged: list[str] = []
    for idx, (name, line_no) in enumerate(result_h3s):
        end_line = result_h3s[idx + 1][1] if idx + 1 < len(result_h3s) else len(rlines)
        gap = _v4_result_beat_gaps(name, rlines[line_no + 1 : end_line])
        if gap is not None:
            flagged.append(gap)
    if flagged:
        preview = "; ".join(flagged[:2]) + (" …" if len(flagged) > 2 else "")
        return CheckResult(
            label,
            True,
            f"{len(flagged)} of {len(result_h3s)} `### <result>`(s) do not follow the "
            f"three-beat (what-is-plotted → plot → interpretation): {preview}",
            is_warn=True,
        )
    return CheckResult(
        label, True, f"all {len(result_h3s)} `### <result>`(s) framed (figure-bearing ones)"
    )


# ─── v4 result-paragraph sentence cap (check 36, #1368) ──────────────────────

# Abbreviations whose trailing period must not end a sentence. `no.` is
# masked ONLY when a digit follows (`no. 3`) — an unguarded `no.` would merge
# a genuine sentence ending in "... is no." with its successor (plan-approval
# critic concern 1, #1368).
_SENTENCE_ABBREV_RE = re.compile(
    r"\b(?:e\.g|i\.e|vs|et\s+al|cf|figs?|eq|approx|ca|resp|incl)\.|\bno\.(?=\s*\d)",
    re.IGNORECASE,
)
_SENTENCE_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_SENTENCE_LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")
_SENTENCE_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")


def _result_prose_paragraphs(block_text: str) -> list[str]:
    """Maximal runs of consecutive PROSE lines inside one `### <result>`
    block, joined with spaces (markdown renders consecutive non-blank lines
    as one paragraph). A blank / blockquote-caption / table / heading /
    image / HTML / fence / list-item / `---` line terminates the current
    run and is itself excluded. `<details>` bodies are stripped first (the
    `_prose_words` convention). Known residual: an UNCLOSED fence swallows
    the rest of the block — false-negative-only (WARN-only check, accepted;
    plan-approval critic concern 2, #1368).
    """
    block_text = _DETAILS_BLOCK_RE.sub("", block_text)
    paras: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in block_text.splitlines():
        s = line.strip()
        if s.startswith(("```", "~~~")):
            in_fence = not in_fence
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        if in_fence:
            continue
        if (
            not s
            or s.startswith((">", "|", "#", "!", "<", "---"))
            or _SENTENCE_LIST_ITEM_RE.match(line)
        ):
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return paras


def _count_sentences(paragraph: str) -> int:
    """Guarded sentence count: split on `[.!?]+` immediately followed by
    whitespace / end-of-paragraph, AFTER masking inline code spans, markdown
    link targets, a small abbreviation list (`no.` only before a digit),
    digit.digit decimals, and ellipses. Segments with >=1 word character
    count (so trailing unterminated text counts as one unit); a
    semicolon-joined chain deliberately counts as ONE sentence unit.
    Stated residuals (WARN-only, accepted): false negatives on
    sentence-ends inside brackets (`... gap.)`) and `;`-chained walls;
    false positives on unlisted abbreviations (`Tab. 2`, initials).
    """
    p = _SENTENCE_INLINE_CODE_RE.sub("CODE", paragraph)
    p = _SENTENCE_LINK_TARGET_RE.sub("]", p)
    p = _SENTENCE_ABBREV_RE.sub(lambda m: m.group(0)[:-1], p)
    p = re.sub(r"(?<=\d)\.(?=\d)", "", p)
    p = p.replace("...", " ").replace("…", " ")
    return sum(1 for seg in re.split(r"[.!?]+(?=\s|$)", p) if re.search(r"\w", seg))


def check_v4_result_paragraph_sentences(body: str) -> CheckResult:
    """Check 36 (v4 only, WARN): each prose paragraph inside a
    `### <result>` block runs 1-3 sentences (SPEC § Conciseness caps (v4) /
    § Results three-beat; Lens 12 is the LM backstop). WARN, NEVER FAIL —
    register judgment (bullets-over-prose, the FAIL decision) stays with
    the clean-result-critic (#1368; incident #1333/#385).
    """
    label = "Results paragraph sentence cap (v4)"
    if not is_v4(body):
        return CheckResult(label, True, "skipped — not a v4 body")
    results = _v4_results_body(body)
    if results is None:
        return CheckResult(label, True, "## Results missing — check 2 will report")
    result_h3s = _collect_tldr_h3_names(results)
    if not result_h3s:
        return CheckResult(label, True, "no `### <result>` headings — check 3 will report")
    rlines = results.splitlines()
    flagged: list[str] = []
    for idx, (name, line_no) in enumerate(result_h3s):
        end_line = result_h3s[idx + 1][1] if idx + 1 < len(result_h3s) else len(rlines)
        block = "\n".join(rlines[line_no + 1 : end_line])
        over = [
            n
            for n in map(_count_sentences, _result_prose_paragraphs(block))
            if n > V4_RESULT_PARA_MAX_SENTENCES
        ]
        if over:
            flagged.append(f"'{name[:48]}' has a {max(over)}-sentence paragraph")
    if flagged:
        preview = "; ".join(flagged[:2]) + (" …" if len(flagged) > 2 else "")
        return CheckResult(
            label,
            True,
            f"{len(flagged)} of {len(result_h3s)} `### <result>`(s) carry a "
            f"≥{V4_RESULT_PARA_MAX_SENTENCES + 1}-sentence paragraph (v4 register: 1-3 "
            f"sentences per paragraph — split or tighten): {preview}",
            is_warn=True,
        )
    return CheckResult(
        label,
        True,
        f"all `### <result>` paragraphs ≤{V4_RESULT_PARA_MAX_SENTENCES} sentences",
    )


# ─── v4 bare-issue-ref check (27) ─────────────────────────────────────────────

# Bare issue reference: `#779`, `(#537)`, `#658's`. Bounded to 1-4 digits so an
# all-digit 6-hex color (`#000000`) cannot match (fail-open on hypothetical
# 5-digit issue ids beats fail-closed on hex colors — the LM lens is the
# backstop). Lookbehind: not preceded by a word char (`file#123`), `&` (HTML
# entities `&#8212;`), `/` (URL fragments `path/#123`), or `#`. Right guard
# `(?!\w)`: the digit run must not abut a word char, so `#123abc` and the
# digit prefix of a mixed hex color (`#4b5563` → `#4`) never match; a
# possessive `#658's` still does (an apostrophe is not a word char).
_BARE_ISSUE_REF_RE = re.compile(r"(?<![\w&/#])#\d{1,4}(?!\w)")
# Prior-issue task URL (the dashboard task route). Scanned directly rather
# than via a [label](target) wrapper so markdown links, <...> autolinks, AND
# bare URLs in standalone prose all hit — dropping the brackets must not
# dodge the check, and the URL line of a multi-line link still fires.
_TASK_URL_RE = re.compile(r"https?://eps\.superkaiba\.com/tasks/\d+")
_V4_STANDALONE_SECTIONS = ("takeaways", "methodology", "results")


def _mask_html_comment_spans(line: str, in_comment: bool) -> tuple[str, bool]:
    """Replace every HTML-comment span in `line` with spaces, threading the
    multiline open/closed state across calls (one call per line, in order).

    Character-grain, not line-grain: a line that OPENS a comment keeps its
    prefix prose scannable (only `<!--`..end-of-line is masked); a line
    that CLOSES one keeps its suffix scannable (only up to and including
    `-->` is masked); any number of `<!-- ... -->` segments per line are
    each masked with the prose between them kept, and a close-then-reopen
    line (`<!-- a --> mid <!-- b`) returns the state OPEN so following
    interior lines stay masked. Spans are substituted with same-length
    runs of spaces — never deleted — so no token join can fabricate a
    `#N` the raw line never carried.

    Returns (masked_line, out_state); len(masked_line) == len(line).
    """
    out: list[str] = []
    pos = 0
    n = len(line)
    while pos < n:
        if in_comment:
            close = line.find("-->", pos)
            if close == -1:
                out.append(" " * (n - pos))
                pos = n
            else:
                end = close + 3
                out.append(" " * (end - pos))
                pos = end
                in_comment = False
        else:
            opener = line.find("<!--", pos)
            if opener == -1:
                out.append(line[pos:])
                pos = n
            else:
                out.append(line[pos:opener])
                pos = opener
                in_comment = True
    masked = "".join(out)
    assert len(masked) == len(line), (len(masked), len(line))
    return masked, in_comment


def _bare_issue_ref_hits(body: str) -> list[tuple[str, str, str]]:
    """Return (section_name, matched_token, line_text) for every bare
    `#<digits>` issue reference AND every prior-issue task URL
    (`_TASK_URL_RE` — `https://eps.superkaiba.com/tasks/<digits>`,
    whether it appears as a `[label](target)` markdown link target, a
    `<...>` autolink, or a bare URL in prose) in the v4 standalone
    sections (## Takeaways / ## Methodology / ## Results), excluding
    every sanctioned form. `body` is the post-frontmatter text (as handed
    to CHECKS entries by verify_text), so frontmatter bare refs never
    reach the scan.

    Sanctioned forms excluded from BOTH scans: fenced code blocks,
    `<details>` blocks, HTML comments (char-span mask with cross-line
    open/closed state — `_mask_html_comment_spans`); GFM table rows; the
    `**Repro:**`/`**Context:**` footer (line-index cut); inline code
    spans (in-line neutralization, substituting a single SPACE — never
    the empty string, which could JOIN adjacent characters into a
    fabricated `#N` token, e.g. ``#`x`123`` → `#123`). Markdown links
    (label + target) are additionally neutralized for the BARE-TOKEN scan
    only — the task-URL scan deliberately runs BEFORE the `_LINK_RE`
    erasure (that erasure is exactly what hid `[#K](.../tasks/K)` links
    from this check, #928), so a task link in a standalone section FAILs
    while a markdown link to a NON-task target stays sanctioned. The URL
    scan's mechanical scope is the dashboard task route
    (`eps.superkaiba.com/tasks/<digits>`); a `#K`-labeled link to a
    non-task target is LM-lens territory (residuals below). The scan does
    not know THIS body's own task id, so a body's own task URL in a
    standalone section also FAILs — consistent with the bare-token scan,
    where a self-referential `#K` also hits.

    HTML-comment handling is character-grain, not line-grain: on a line
    that OPENS a multiline comment only the `<!--`..end-of-line span is
    masked (prefix prose IS scanned — `Uses #779 corpus <!-- note`
    hits); on a line that CLOSES one only the span up to and including
    `-->` is masked (suffix prose IS scanned — `--> still follows #781`
    hits); multiple `<!-- ... -->` segments per line are each masked
    with the prose between them scanned, and a close-then-reopen line
    (`<!-- a --> mid <!-- b`) leaves the state OPEN so following
    interior lines stay masked (no false hit on a `#K` inside the still
    open comment). The `<details>` counter reads the comment-MASKED
    text, so a commented-out `<details>` tag cannot open the details
    mask.

    Documented residual edges (the clean-result-critic Lens 2 LM read is
    the backstop; direction noted per item):

    - The `<details>` open/close counter runs PRE-neutralization (before
      link/inline-code masking), so a backticked ``<details>`` prose
      mention opens the mask and excludes the remainder of the body
      (fail-open — missed refs, never false FAILs).
    - An unclosed `<details>` likewise excludes everything after it;
      nesting is handled by the depth counter (fail-open).
    - The comment char-span pass also runs PRE-neutralization, so a
      backticked ``<!--`` prose mention opens the comment mask
      (fail-open, same family as the backticked ``<details>``).
    - 4-space indented code blocks and the LABEL side of multi-line
      `[#K](\\nurl)` links are NOT excluded (both survey-clean across all
      on-disk v4 bodies as of 2026-07-03) — a `#K` inside either would
      false-FAIL; the inline-code escape hatch covers the code-block
      case. (The URL line of a multi-line TASK link now correctly FAILs
      via the URL scan, so that half is no longer a pure false-FAIL edge;
      the residual is the label-side bare `#K` of a multi-line NON-task
      link.)
    - Reference-style links (`[label][ref]` + a definition line
      elsewhere) are not modeled: a `[#K][ref]` LABEL already FAILs the
      bare-token scan, and a task-URL definition line inside a standalone
      section hits the URL scan, but a non-`#K`-labeled reference link
      whose definition sits outside the standalone sections escapes
      (fail-open — the LM lens is the backstop).
    - Case-mangled (`HTTPS://EPS...`) or schemeless
      (`eps.superkaiba.com/tasks/K`) task URLs do not match the URL scan
      — all project-generated URLs are lowercase + schemed; a mangled
      form is deliberate evasion (fail-open, LM-lens backstop).
    - A `[#K](<non-task URL>)` label-side evasion — including the legacy
      GitHub-issue link form `[#98](https://github.com/.../issues/98)` —
      and a RELATIVE `/tasks/K` link target both escape mechanically (the
      `#K` label is erased with the link by `_LINK_RE` before the
      bare-token scan; the target is not a schemed dashboard task URL) —
      fail-open, LM-lens backstop.
    - An odd backtick pair straddling a link can mask its URL from the
      inline-code-masked URL scan (fail-open — same family as the
      backticked ``<details>`` residual above; ADJACENT well-formed code
      spans never swallow the link between them, since `_INLINE_CODE_RE`
      excludes backticks from span content).
    - In a slash-run `#658/#742` only the first token matches (the `/`
      lookbehind protects URL fragments) — the line still FAILs, which
      is sufficient.
    """
    lines = body.splitlines()
    footer = _v4_footer_start_line(body)  # None -> no footer cut
    table_idx = _table_row_line_indices(lines)
    # Structural exclusion mask: line-grain for fenced code + <details>
    # blocks, char-grain for HTML comments (comment_masked[i] = line i
    # with every comment span space-substituted; mixed prose on comment
    # opening/closing lines stays scannable). Single pass, whole-body
    # (fences/details/comments may open in one section and close in
    # another).
    excluded: set[int] = set()
    comment_masked: dict[int, str] = {}
    in_fence = False
    details_depth = 0
    in_comment = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            excluded.add(i)
            continue
        if in_fence:
            excluded.add(i)
            continue
        masked, in_comment = _mask_html_comment_spans(line, in_comment)
        comment_masked[i] = masked
        opens = len(re.findall(r"<details\b", masked, re.IGNORECASE))
        closes = len(re.findall(r"</details>", masked, re.IGNORECASE))
        if details_depth > 0 or opens:
            excluded.add(i)
        details_depth = max(0, details_depth + opens - closes)
    hits: list[tuple[str, str, str]] = []
    for name, start, end in find_h2_sections(body):
        if name.casefold() not in _V4_STANDALONE_SECTIONS:
            continue
        for i in range(start, end):
            if footer is not None and i >= footer:
                continue  # footer + everything after
            if i in excluded or i in table_idx:
                continue
            # Substitute a single space (never ""): an empty-string sub
            # can JOIN the char before and after the removed span into a
            # new `#N` token that the raw line never carried. Comment
            # spans were already space-masked in comment_masked (every
            # non-fence line has an entry; fence lines are excluded).
            # Task-URL scan — runs BEFORE _LINK_RE erases link targets
            # (that erasure is exactly what hid [#K](.../tasks/K) links
            # from this check, #928). Inline code protects it
            # (escape-hatch parity with the bare-token scan below).
            link_scan = _INLINE_CODE_RE.sub(" ", comment_masked[i])
            for m in _TASK_URL_RE.finditer(link_scan):
                hits.append((name, m.group(0), lines[i].strip()[:90]))
            residue = _LINK_RE.sub(" ", comment_masked[i])  # [label](target) gone
            residue = _INLINE_CODE_RE.sub(" ", residue)  # `code` escape hatch
            for m in _BARE_ISSUE_REF_RE.finditer(residue):
                hits.append((name, m.group(0), lines[i].strip()[:90]))
    return hits


def check_v4_no_bare_issue_refs(body: str) -> CheckResult:
    """Check 27 (v4 only): no bare `#<digits>` issue refs AND no prior-issue
    task links/URLs in the standalone sections. SPEC.md § `## Goal` (v4):
    the Goal context slot is the ONLY place in the body that may cite prior
    tasks; `## Takeaways` / `## Methodology` / `## Results` are standalone,
    and lineage/provenance live in the `**Repro:**`/`**Context:**` footer.
    The task-link half matches any form whose text carries the dashboard
    task route `https://eps.superkaiba.com/tasks/<digits>` — a `[#K](...)`
    markdown link, a `<...>` autolink, or a bare URL. Mechanical scope:
    dashboard task URLs only; a `#K`-labeled link to a NON-task target is
    LM-lens territory. Sanctioned forms that do not trip: markdown links to
    NON-task targets, GFM table rows (the Training-table Source column),
    fenced/inline code, `<details>` blocks, HTML comments, the footer, YAML
    frontmatter. Exclusion edge behavior (residuals + directions) is
    documented on `_bare_issue_ref_hits`. Origin: #841 round-2 (bare `#779`
    x~8 in Methodology survived two ensemble review rounds); the task-link
    half added by #1002 after the #928 round-1 miss (a `[#K](.../tasks/K)`
    link in `## Methodology` sailed through mechanically because `_LINK_RE`
    erased it before the token scan). PASSes vacuously on v3 / v2 / legacy
    bodies (forward-only)."""
    label = "no bare issue refs in standalone sections (v4)"
    if not is_v4(body):
        return CheckResult(label, True, "skipped — not a v4 body")
    hits = _bare_issue_ref_hits(body)
    if not hits:
        return CheckResult(
            label,
            True,
            "Takeaways/Methodology/Results carry no bare `#K` refs or prior-issue task links",
        )
    shown = "; ".join(f'## {sec}: `{tok}` in "{txt}"' for sec, tok, txt in hits[:5])
    more = f" (+{len(hits) - 5} more)" if len(hits) > 5 else ""
    return CheckResult(
        label,
        False,
        f"prior-issue reference(s) in standalone section(s) — {shown}{more}. "
        "Prior-issue references — bare `#K` tokens AND task links/URLs "
        "(`https://eps.superkaiba.com/tasks/K`, linked or bare) — live ONLY in the "
        "`## Goal` context slot and the `**Repro:**`/`**Context:**` footer "
        "(SPEC.md § `## Goal` (v4) + Rule A). Rewrite the prose to describe the method "
        "standalone and move lineage to the footer; converting a bare ref into a "
        "`[#K](...)` link in place FAILs here too. The inline-code escape hatch is for "
        "NON-issue strings only (a hex color, an ordinal like `GPU #2`, a verbatim "
        "syntax example) — do NOT backtick a genuine issue reference or task link to "
        "silence this check.",
    )


# ─── v3 body-table ⊆ methodology-doc table check (21) ────────────────────────


def _parse_param_table_rows(section: str) -> dict[str, str]:
    """Parse a `**Parameters:**` two-column markdown table out of a
    `## Reproducibility` section. Returns {param_key_norm: value_norm}.

    Recognizes the table that follows a `**Parameters:**` boldface label
    (the first GFM 2-col table after it). Keys/values are normalized:
    lowercased, collapsed whitespace, backticks stripped. The header row
    and delimiter row are skipped.
    """
    out: dict[str, str] = {}
    # Find the slice starting at `**Parameters:**`.
    m = re.search(r"\*\*\s*Parameters\s*:?\s*\*\*", section, re.IGNORECASE)
    if not m:
        return out
    after = section[m.end() :]
    in_fence = False
    seen_delim = False
    for line in after.splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not s.startswith("|"):
            # A blank line right after the label is fine; a non-table
            # non-blank line AFTER we've started the table ends it.
            if seen_delim:
                break
            if s == "":
                continue
            # A boldface label like `**Artifacts:**` ends the table region.
            if s.startswith("**"):
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if _GFM_DELIM_RE.match(line):
            seen_delim = True
            continue
        if len(cells) < 2:
            continue
        key = re.sub(r"\s+", " ", cells[0].replace("`", "")).strip().casefold()
        val = re.sub(r"\s+", " ", cells[1].replace("`", "")).strip().casefold()
        if not key or key in ("parameter", "param", "name"):
            continue
        out[key] = val
    return out


# Markers that mean "this experiment did no model training", so the doc's
# §2 hyperparameter table is legitimately empty / N/A. Normalized
# (casefolded, backticks stripped, whitespace collapsed) before matching.
_NO_TRAINING_DOC_MARKERS = (
    "n/a — no model training",
    "n/a - no model training",
    "n/a — no training",
    "n/a - no training",
    "no model training",
    "no training (",
    "no training,",
    "no training.",
    "kind: analysis",
    "zero-gpu",
    "zero gpu",
)


def _methodology_doc_has_no_training_recipe(doc_raw: str) -> bool:
    """True when the methodology doc's §2 (Hyperparameters / Training
    recipe) section carries no real hyperparameter table — i.e. it is
    empty or explicitly marked N/A because the task did no model training
    (an analysis-only `kind: experiment`).

    Check 21's subset assertion is calibrated against a CANONICAL COMPLETE
    training-hyperparameter table that the body Parameters table slims
    from. An analysis-only task has no such table — its body Parameters
    are analysis-design descriptors (candidate forms, bootstrap B, logit
    ε, …) written freehand, not slimmed hyperparameters — so the subset
    premise does not hold and the assertion must PASS-skip instead of
    false-FAILing. The #489-class misprint guard stays fully active for
    every task that actually trains (this returns False there).

    Detection isolates the §2 section by its numbered `## 2.` header
    (canonically `## 2. Hyperparameters`; some analysis-only docs name it
    `## 2. Training recipe`), then treats it as no-training when EITHER
    (a) it contains an explicit no-training marker, OR (b) it contains no
    GFM table delimiter row at all (no hyperparameter table emitted).
    """
    # Isolate the §2 section by its numbered `## 2. <name>` header (the H2
    # name varies — "Hyperparameters" canonically, "Training recipe" in
    # #644 — so match on the `2.` number prefix, not the section name).
    m = re.search(r"^##\s*2\.\s.*$", doc_raw, re.MULTILINE)
    if not m:
        return False
    tail = doc_raw[m.end() :]
    nxt = re.search(r"^##\s", tail, re.MULTILINE)
    sec2 = tail[: nxt.start()] if nxt else tail
    norm = re.sub(r"\s+", " ", sec2.replace("`", "")).strip().casefold()
    if any(marker in norm for marker in _NO_TRAINING_DOC_MARKERS):
        return True
    # No table delimiter row anywhere in §2 → no hyperparameter table.
    return not _GFM_DELIM_RE.search(sec2)


def _split_composite_cell(val: str) -> list[str]:
    """Split a Parameters-cell value into the comma/semicolon-separated
    sub-values that a one-fact-per-row methodology doc §2 table would
    list individually.

    The v3 conciseness convention bundles several facts into ONE compact
    body cell (e.g. ``marker-only loss, lr 5e-6, band-stop [5,12] nat,
    max_new_tokens 2048``) while the canonical doc §2 table lists each
    fact on its own row. A whole-cell substring match against the doc
    therefore false-FAILs the conformant compact form — so before failing
    we decompose the cell and reconcile each sub-value independently.

    Splitting is BRACKET-AWARE: only top-level (depth-0) commas/semicolons
    separate sub-values, so a bracketed interval or list that legitimately
    contains a comma stays intact as one token — e.g. ``[5,12]`` (a marker
    band-stop / install target band) and ``[42, 137, 256]`` (a seed list)
    are NOT torn apart. ``[]``, ``()`` and ``{}`` all open/close a level.
    Empty tokens are dropped.
    """
    tokens: list[str] = []
    buf: list[str] = []
    depth = 0
    openers = {"[": "]", "(": ")", "{": "}"}
    closers = {"]", ")", "}"}
    for ch in val:
        if ch in openers:
            depth += 1
            buf.append(ch)
        elif ch in closers:
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in (",", ";") and depth == 0:
            tokens.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tokens.append("".join(buf).strip())
    return [t for t in tokens if t]


def check_body_params_subset_of_doc(
    body: str, *, methodology_doc_path: Path | None = None
) -> CheckResult:
    """Check 21 (v3 only): the load-bearing param rows in the body's
    `## Reproducibility` Parameters table are a SUBSET of the
    methodology doc's §2 complete hyperparameter table.

    Two-tier split (plan §3b): the body table slims to the load-bearing
    subset; the methodology doc §2 is the canonical COMPLETE table.
    Every body param key+value must appear in the doc table (key match +
    value substring containment, both normalized).

    Gate-timing: the methodology doc lives on the issue worktree branch
    pre-merge, so this check is a NO-OP PASS when no doc path is given OR
    the file does not exist. At gate time the orchestrator passes the
    worktree path explicitly via ``--methodology-doc``; at promote-time
    verify (post-merge) the ``--issue`` path in ``main`` opportunistically
    resolves ``docs/methodology/issue_<N>.md`` on disk, so the check binds
    fully then without the orchestrator re-passing the flag. PASSes
    vacuously on v2 / legacy bodies.

    Analysis-only carve-out: when the doc's §2 hyperparameter section is
    empty / N/A because the task did no model training (see
    ``_methodology_doc_has_no_training_recipe``), there is no canonical
    complete hyperparameter table for the body to be a subset OF — the
    body Parameters are analysis-design descriptors, not slimmed
    hyperparameters — so the subset assertion PASS-skips rather than
    false-FAILing (#644).
    """
    label = "Body Parameters ⊆ methodology doc §2"
    if not is_v3(body):
        return CheckResult(label, True, "skipped — not a v3 body")
    if methodology_doc_path is None or not methodology_doc_path.exists():
        return CheckResult(
            label,
            True,
            "skipped — no methodology doc on disk to reconcile against "
            "(binds at promote-time verify, post-merge)",
        )
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(label, True, "skipped — no Reproducibility section")
    body_params = _parse_param_table_rows(repro)
    if not body_params:
        return CheckResult(label, True, "skipped — no Parameters table in body")
    doc_raw = methodology_doc_path.read_text(errors="replace")
    if _methodology_doc_has_no_training_recipe(doc_raw):
        return CheckResult(
            label,
            True,
            "skipped — methodology doc §2 has no training-hyperparameter table "
            "(analysis-only task, no model training); body Parameters are "
            "analysis-design descriptors, not a slimmed hyperparameter subset",
        )
    doc_text = doc_raw.casefold()
    doc_text = re.sub(r"`", "", re.sub(r"\s+", " ", doc_text))
    missing: list[str] = []
    for key, val in body_params.items():
        # Skip rows whose value is an explicit non-applicable / free-text
        # marker the doc legitimately need not echo verbatim.
        if val in ("n/a", "none", "-", "—", "–"):  # noqa: RUF001
            continue
        # Key must appear in the doc; AND the value must appear somewhere
        # in the doc (substring containment on normalized text — the doc
        # table may carry a Source column / extra formatting around it).
        if key not in doc_text:
            missing.append(f"{key} (key absent from doc §2 table)")
            continue
        if not val:
            continue
        # Whole-cell containment fast-path: a body cell that matches a doc
        # row verbatim reconciles immediately.
        if val in doc_text:
            continue
        # Compact composite cell (v3 conciseness): the body bundles several
        # facts into one cell while the doc lists each on its own §2 row, so
        # the whole-cell string never appears verbatim. Decompose the cell
        # (bracket-aware) and require each sub-value to appear in the doc;
        # a genuine misprint still FAILs because the wrong sub-value token
        # is absent.
        unmatched = [tok for tok in _split_composite_cell(val) if tok not in doc_text]
        if unmatched:
            missing.append(
                f"{key}={val} (sub-value(s) not found in doc §2 table: {', '.join(unmatched)})"
            )
    if missing:
        preview = "; ".join(missing[:4]) + (
            f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        )
        return CheckResult(
            label,
            False,
            f"{len(missing)} body Parameters row(s) not reconciled against the "
            f"methodology doc §2 complete table: {preview}. The doc §2 table is the "
            "canonical complete hyperparameter set; every body param key+value must "
            "appear there.",
        )
    return CheckResult(
        label,
        True,
        f"all {len(body_params)} body Parameters row(s) appear in the methodology doc §2 table",
    )


# ─── Driver ────────────────────────────────────────────────────────────────


# Body-only checks: each takes the post-frontmatter `body` string. The
# no-duplicate-frontmatter check needs the RAW body.md text (so it can
# count stacked `---...---` blocks regardless of what `split_frontmatter`
# would parse), and is dispatched specially in `verify_text` below.
# `check_concerns_audit` (Lens 14) needs the sibling concerns.jsonl path,
# so it also lives outside CHECKS and is dispatched specially below.
CHECKS = [
    check_body_nonstub,
    check_title_confidence,
    check_required_sections,
    check_tldr_labels,
    check_tldr_nested_structure,
    check_figure_image,
    check_figure_url_resolvable,
    check_figure_caption,
    check_confidence_matches,
    check_repro_subgroups,
    check_repro_url_permanence,
    check_repro_sentinel_scrub,
    check_cherry_picked_label,
    check_qualitative_data_link,
    check_planned_vs_actual_denominator,
    check_figure_h2_is_deprecated,
    check_details_narrative_flow,
    check_mdx_safe_urls,
    check_repro_committed_claims_exist,
    check_repro_artifact_urls_exist,
    # v3-gated checks (PASS vacuously on non-v3 bodies; check 19b also runs on v4):
    check_data_shape,  # check 18 (v3)
    check_data_subset_disclosure,  # check 19 (v3)
    check_data_unwrapped_example_table,  # check 19b (v3 ## Data + v4 ## Methodology, WARN)
    check_v3_word_caps,  # check 20 (v3)
    # v4-gated checks (PASS vacuously on non-v4 bodies). Check 20 (v4)
    # `check_v4_word_caps` is NOT here — it needs the issue number for the
    # events-based folded-round budget scaling, so it is dispatched
    # separately in `verify_text` (#921):
    check_v4_methodology_shape,  # check 18 (v4)
    check_v4_results_beat,  # check 21 (v4, WARN)
    check_v4_no_bare_issue_refs,  # check 27 (v4) — bare `#K` refs + task links, standalone secs
    check_v4_result_paragraph_sentences,  # check 36 (v4, WARN) — ≥4-sentence paras (#1368)
    # check 37 (WARN, v4, #1370) — footer `- Reused ... from [#M](...)` bullets carry a
    # revision/path pin (body-text-only sibling of check 35's metadata-side trigger):
    check_footer_reuse_bullets_pinned,
    # generation-agnostic checks (v2 AND v3 AND v4):
    check_figure_url_sha_matches_repro,  # check 22
    check_hf_url_resolves,  # check 23
    check_figure_text_vs_body_tokens,  # check 24 (WARN)
    check_audit_availability_claims_match_hf,  # check 25
    check_figure_panel_prose_vs_sidecar,  # check 26 (FAIL)
    check_figure_label_codes,  # check 28 (WARN) — opaque config codes in figure sidecar text
    check_figure_tracked_at_head,  # check 29 (WARN) — figures tracked at live refs (#964, #841)
    check_hf_file_count_claims,  # check 30 (WARN) — count claims vs Hub files-only count (#931)
    check_hf_adjacent_file_claims,  # check 32 (WARN) — adjacent file claims are tree members (#952)
    # check 33 (WARN) — bolded what-is-plotted decimals vs sidecar plotted values (#1107, #825 r1):
    check_figure_prose_numerics_vs_sidecar,
    # check 34 (WARN, forward-only) — beat-phrase series-structure claims vs the sidecar's
    # rendered-text block (fires only when meta["text"] is present; #1255, #1092 defect (b)):
    check_figure_beat_claims_vs_sidecar_text,
    # Check 31 (`check_orphaned_per_unit_figures`, WARN, generation-agnostic)
    # is NOT here either — like check 20 (v4) it needs the issue number (for
    # figures-dir scoping), so it is dispatched separately in `verify_text`
    # (#1011; the check-20/#921 precedent).
    # Check 38 (`check_linked_not_embedded_figures`, WARN, v4-only) is NOT
    # here either — it needs the issue number (for own-figures-dir scoping),
    # so it is dispatched separately in `verify_text` (#1371; the
    # check-31/#1011 precedent).
]


def _is_paper_stub_fm(fm: dict) -> bool:
    """True when the body's frontmatter opts into the paper clean-result track.

    Accepts the YAML-parsed boolean ``True`` and the quoted string ``"true"``
    (case-insensitive). Mirrors ``task_workflow.is_paper_task`` so the verifier
    branches on a stub WITHOUT importing the workflow library (keeps the
    verifier standalone).
    """
    v = fm.get("paper")
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def _verify_paper_stub(body: str) -> tuple[bool, list[CheckResult]]:
    """Minimal stub-shape check for a `paper: true` body.

    A paper-stub body.md is NOT a markdown clean-result — its canonical
    clean-result is the LaTeX paper under docs/papers/issue_<N>/, verified by
    scripts/verify_paper.py. Here we only confirm the stub has the three
    contract elements (H1 title + an abstract + a paper link) so a broken /
    empty stub still FAILs loudly; the deep checks belong to verify_paper.py.
    """
    problems: list[str] = []
    if not re.search(r"^#\s+\S", body, re.MULTILINE):
        problems.append("no H1 `# <title>`")
    # an abstract: either a `## Abstract` H2 OR ≥80 chars of non-heading prose.
    has_abstract = bool(re.search(r"^##\s+Abstract\b", body, re.MULTILINE)) or (
        len(re.sub(r"^#.*$", "", body, flags=re.MULTILINE).strip()) >= 80
    )
    if not has_abstract:
        problems.append("no abstract (a `## Abstract` H2 or a prose paragraph)")
    # a paper link: a docs/papers/issue_<N>/ path or an HF papers/ URL.
    if not re.search(r"docs/papers/issue_\d+|/papers/issue_\d+|paper\.html", body):
        problems.append("no paper link (docs/papers/issue_<N>/ or HF papers/ URL)")
    if problems:
        return False, [
            CheckResult(
                "paper-stub body.md valid",
                False,
                "; ".join(problems)
                + " — paper-task: the LaTeX paper is verified by verify_paper.py, "
                "but the body.md stub must still carry an H1 + abstract + paper link",
            )
        ]
    return True, [
        CheckResult(
            "paper-stub body.md valid",
            True,
            "paper: true — markdown clean-result checks skipped; the canonical "
            "clean-result is the LaTeX paper, verified by scripts/verify_paper.py "
            "(run `verify_paper.py --issue <N>`)",
        )
    ]


def verify_text(
    raw: str,
    *,
    source: str = "",
    concerns_path: Path | None = None,
    plan_path: Path | None = None,
    original_body_path: Path | None = None,
    methodology_doc_path: Path | None = None,
    issue: int | None = None,
    eval_root: Path | None = None,
    body_source_path: Path | None = None,
) -> tuple[bool, list[CheckResult]]:
    """Run every clean-result check on ``raw`` body.md text.

    ``concerns_path`` is the absolute path to the sibling
    ``concerns.jsonl`` when the verifier was invoked with
    ``--issue <N>`` (resolved by ``main()``). When supplied AND present
    on disk, the Lens 14 concerns-audit check runs against the body;
    otherwise the audit is skipped (PASS) and surfaces in the output as
    such. File-only invocations (``--file`` without a sibling) and
    ``--body-stdin`` skip the audit by default.

    ``plan_path`` is the absolute path to the sibling ``plans/plan.md``
    (resolved by ``main()`` for ``--issue <N>`` / a ``--file`` sibling).
    When supplied AND present, check 16 reconciles the Reproducibility
    learning rate against the approved plan; otherwise it skips (PASS).

    ``original_body_path`` is the absolute path to the sibling
    ``original-body.md`` (resolved by ``main()`` the same way). Check 17
    uses it to detect a ``## Provenance`` section in the pre-promotion
    body — recorded origin data that the clean-result body must carry
    forward in its ``**Context:**`` row.

    ``methodology_doc_path`` is the absolute path to the auto-generated
    ``docs/methodology/issue_<N>.md`` reference, passed explicitly by
    the orchestrator at gate time (the doc lives on the issue worktree
    branch pre-merge, so the verifier cannot resolve it from a sibling).
    Check 21 (v3 only) reconciles the body's load-bearing Parameters
    rows against the doc §2 complete table when the doc exists; it is a
    NO-OP PASS otherwise. May be set via ``--methodology-doc <path>``.
    """
    fm, body = split_frontmatter(raw)
    # Paper-stub branch (`paper: true`): the canonical clean-result is a LaTeX
    # paper under docs/papers/issue_<N>/, verified by scripts/verify_paper.py —
    # NOT this markdown verifier. A paper-stub body.md (H1 + abstract + paper
    # link) must NOT be hard-FAILed by the markdown Check 0 / structure checks.
    # Run a minimal stub-shape sanity check and PASS, pointing at verify_paper.
    # Grandfathered markdown bodies (no `paper:` flag) fall through unchanged.
    if _is_paper_stub_fm(fm):
        return _verify_paper_stub(body)
    if LEGACY_SAGAN_CARD_SENTINEL in body:
        return True, [
            CheckResult(
                "legacy Sagan-card detected",
                True,
                "skipping markdown spec — body is grandfathered HTML; "
                "run verify_sagan_card.py for those bodies",
            )
        ]
    # Check 0 (body-nonstub) short-circuits the rest of the chain when it
    # FAILs. A stub body would otherwise cascade into a dozen "<section>
    # missing" errors that bury the actual root cause (the cache → body.md
    # silent-handoff failure). Returning a single FAIL gives the operator
    # one clear signal pointing at analyzer.md Step 6.
    stub_result = check_body_nonstub(body)
    if not stub_result.passed:
        return False, [stub_result]
    # Check 0b (no-duplicate-frontmatter) reads the RAW body.md text so it
    # can count stacked `---...---` blocks regardless of what
    # `split_frontmatter` would parse. Slotted right after the stub check
    # so the failure surfaces early in the report.
    dup_fm_result = check_no_duplicate_frontmatter(raw)
    results = [stub_result, dup_fm_result] + [chk(body) for chk in CHECKS[1:]]
    # Goal-of-experiment field is a soft INFO/WARN check — it never
    # FAILs (enforcement is at /issue Step 0c, not here) and needs the
    # frontmatter, so it lives outside the body-only CHECKS list.
    results.append(check_goal_present(body, fm))
    # H1 ↔ frontmatter-title sync (#1110/#825) — needs the frontmatter, so it
    # lives outside the body-only CHECKS list like check_goal_present.
    results.append(check_h1_matches_frontmatter_title(body, fm))
    # Lens 14 concerns audit — mirror of clean-result-critic Lens 14.
    # Needs the sibling concerns.jsonl, so lives outside CHECKS too.
    results.append(check_concerns_audit(body, concerns_path=concerns_path))
    # Check 16 (Reproducibility lr matches plan) needs the sibling
    # plans/plan.md, so it also lives outside the body-only CHECKS list.
    results.append(check_repro_lr_matches_plan(body, plan_path=plan_path))
    # Check 17 (Reproducibility Context provenance row) needs the
    # frontmatter (origin_prompt) + the sibling original-body.md, so it
    # also lives outside the body-only CHECKS list.
    results.append(check_repro_context_provenance(body, fm, original_body_path=original_body_path))
    # Check 21 (v3 body Parameters ⊆ methodology doc §2 table) needs the
    # explicitly-passed methodology doc path, so it also lives outside
    # the body-only CHECKS list. NO-OP PASS on v2 / legacy bodies and
    # whenever no doc is supplied (gate-timing — see the check docstring).
    results.append(check_body_params_subset_of_doc(body, methodology_doc_path=methodology_doc_path))
    # Check 20 (v4) word caps needs the issue number for the events-based
    # folded-round budget scaling (#921; the v3 twin stays body-only inside
    # CHECKS), so it is dispatched separately like the other
    # context-needing checks. PASS-skip on non-v4 bodies, and the events
    # leg degrades to the footer-only read when `issue` is None/unknown.
    results.append(check_v4_word_caps(body, issue=issue))
    # Check (#732, judge-API-error denominator) needs the issue number AND
    # the eval-root / body-source-path resolution legs (the body-only CHECKS
    # list carries none of these), so it also lives outside CHECKS. Graceful
    # PASS when the issue is unknown or no eval data is reachable.
    results.append(
        check_judge_error_denominator(
            body, issue=issue, eval_root=eval_root, body_source_path=body_source_path
        )
    )
    # Check 35 (#1256): cross-issue reuse pins in committed result-JSON
    # metadata must be declared in the body (footer Reused bullets). Needs
    # the issue number + eval-root resolution, so it lives outside CHECKS
    # (the #732 check_judge_error_denominator precedent).
    results.append(
        check_cross_issue_reuse_provenance(
            body, issue=issue, eval_root=eval_root, body_source_path=body_source_path
        )
    )
    # Check 31 (WARN, #1011): orphaned per-unit companion figures — needs the
    # issue number for figures-dir scoping, so it lives outside the body-only
    # CHECKS list (check-20/#921 precedent).
    results.append(check_orphaned_per_unit_figures(body, issue=issue))
    # Check 38 (WARN, #1371): Results figures referenced as links, not
    # embeds — needs the issue number for own-figures-dir scoping, so it
    # lives outside the body-only CHECKS list (check-31/#1011 precedent).
    results.append(check_linked_not_embedded_figures(body, issue=issue))
    overall = all(r.passed for r in results)
    return overall, results


def _load_text_for_issue(number: int) -> tuple[str, Path]:
    from explore_persona_space.task_workflow import find_task_path  # local import

    folder = find_task_path(number)
    body_path = folder / "body.md"
    return body_path.read_text(), body_path


def _resolve_file_siblings(
    body_source_path: Path,
) -> tuple[Path | None, Path | None, Path | None, int | None]:
    """For a ``--file <body.md>`` invocation, resolve the sibling
    artifacts a ``tasks/<status>/<N>/body.md`` layout carries so the
    sibling-dependent checks fire on analyzer-side dry runs: the Lens 14
    concerns audit (``concerns.jsonl``), check-16 lr reconciliation
    (``plans/plan.md``), and check-17 context-provenance read
    (``original-body.md``). Also opportunistically parses the issue number
    from the parent dir name (so the #732 judge-error-denominator check
    binds). Returns ``(concerns_path, plan_path, original_body_path,
    issue)`` — each None when the corresponding sibling is absent / the
    dir name is not numeric.
    """
    parent = body_source_path.parent
    concerns = parent / "concerns.jsonl"
    plan = parent / "plans" / "plan.md"
    orig = parent / "original-body.md"
    issue = int(parent.name) if parent.name.isdigit() else None
    return (
        concerns if concerns.exists() else None,
        plan if plan.exists() else None,
        orig if orig.exists() else None,
        issue,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--issue", type=int, help="task number to verify")
    grp.add_argument("--file", help="path to a body.md to verify")
    grp.add_argument("--body-stdin", action="store_true", help="read body from stdin")
    parser.add_argument(
        "--methodology-doc",
        help=(
            "path to the auto-generated docs/methodology/issue_<N>.md reference "
            "(check 21, v3 only). The orchestrator passes the issue-worktree path "
            "explicitly at gate time, since the doc is not yet on main pre-merge. "
            "NO-OP PASS when omitted or the file does not exist."
        ),
    )
    parser.add_argument(
        "--eval-root",
        help=(
            "path to the ROOT directory under which eval_results/issue_<N>/ lives "
            "(check judge-API-error denominator, #732). The orchestrator passes the "
            "issue-worktree root explicitly at the Step 9a-bis pre-merge gate, since "
            "the eval JSONs are not yet on main pre-merge. When omitted, the check "
            "resolves the eval root via the --file-derived / cwd / MAIN ladder, and "
            "graceful-PASSes when none reach the dir."
        ),
    )
    args = parser.parse_args()

    concerns_path: Path | None = None
    plan_path: Path | None = None
    original_body_path: Path | None = None
    methodology_doc_path: Path | None = None
    issue: int | None = None
    eval_root: Path | None = None
    body_source_path: Path | None = None
    if args.methodology_doc:
        cand = Path(args.methodology_doc).expanduser()
        if cand.exists():
            methodology_doc_path = cand
    if args.eval_root:
        eval_root = Path(args.eval_root).expanduser()
    if args.issue is not None:
        try:
            raw, source_path = _load_text_for_issue(args.issue)
            source = str(source_path)
            issue = args.issue
            body_source_path = source_path.resolve()
            concerns_path = source_path.parent / "concerns.jsonl"
            plan_path = source_path.parent / "plans" / "plan.md"
            original_body_path = source_path.parent / "original-body.md"
            # Check 21: when --methodology-doc wasn't passed explicitly,
            # opportunistically resolve the on-disk doc (present on `main`
            # at promote-time verify, post-merge) so the body-table ⊆
            # doc-table assert actually binds then. Pre-merge gate-time
            # callers still pass the worktree path explicitly above.
            if methodology_doc_path is None:
                repo = _resolve_repo_root()
                if repo is not None:
                    cand_doc = repo / "docs" / "methodology" / f"issue_{args.issue}.md"
                    if cand_doc.exists():
                        methodology_doc_path = cand_doc
        except FileNotFoundError as e:
            print(f"verify_task_body: {e}", file=sys.stderr)
            return 2
    elif args.file:
        raw = Path(args.file).read_text()
        source = args.file
        body_source_path = Path(args.file).resolve()
        concerns_path, plan_path, original_body_path, file_issue = _resolve_file_siblings(
            body_source_path
        )
        if issue is None:
            issue = file_issue
    else:
        raw = sys.stdin.read()
        source = "<stdin>"

    overall, results = verify_text(
        raw,
        source=source,
        concerns_path=concerns_path,
        plan_path=plan_path,
        original_body_path=original_body_path,
        methodology_doc_path=methodology_doc_path,
        issue=issue,
        eval_root=eval_root,
        body_source_path=body_source_path,
    )
    print(f"verify_task_body — {source}")
    for r in results:
        print(r.render())
    print()
    if overall:
        print("OVERALL: PASS")
        return 0
    n_fail = sum(1 for r in results if not r.passed)
    print(f"OVERALL: FAIL ({n_fail} of {len(results)} checks failed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
