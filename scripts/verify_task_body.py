#!/usr/bin/env python3
"""verify_task_body.py — mechanical verifier for markdown clean-result bodies.

Replaces `verify_sagan_card.py` for new (markdown) bodies. Mechanical
gate for the 2-content-section markdown clean-result spec (migrated
2026-W22, task #454). Source of truth for the body shape:
`.claude/skills/clean-results/SPEC.md`. The body carries THREE required
H2 sections in order — `## Human TL;DR` / `## TL;DR` / `## Reproducibility`
— with `## TL;DR` absorbing the per-result narrative (one `### Motivation`
H3 then one `### <finding>` H3 per result with one inline figure +
cherry-picked raw-completion example + dropdown + all-raw link) and
`## Reproducibility` absorbing the Parameters table + the Confidence
sentence. The retired `## Details` and `## Figure` H2s are now FAIL
patterns — a body that carries either is rejected so it migrates cleanly
to the new shape (forward-only; the ~95 legacy `has_clean_result=true`
bodies are never re-verified, so tightening cannot regress them).

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
   shape only — existence probing is check 8b's job.
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
    which previously got shape verification only. HF Hub / WandB /
    external-repo links stay shape-checked only (check 8): their
    existence is not decidable from the local object DB, and an
    unauthenticated 404 on an external private repo would false-FAIL.
9. Reproducibility sentinel scrub — no `{{`, `TBD`, `see config`, or
   `default` placeholders anywhere under `## Reproducibility`.
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
    not recorded). Spec: `.claude/skills/clean-results/SPEC.md`
    § `**Context:**` row.

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

Usage:

    uv run python scripts/verify_task_body.py --issue <N>
    uv run python scripts/verify_task_body.py --file path/to/body.md
    uv run python scripts/verify_task_body.py --body-stdin

Exits 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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

CONFIDENCE_LEVELS = {"LOW", "MODERATE", "HIGH"}

# Sentinel substrings that indicate a placeholder slipped through.
SENTINEL_SUBSTRINGS = ["TBD", "{{", "see config", "default"]

# Minimum number of characters of rationale required AFTER the
# `Confidence: <level> —` dash on the confidence line.
MIN_CONFIDENCE_RATIONALE_CHARS = 20

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

    Known follow-up (deferred 2026-05-31): the window also stops at a
    fence boundary but does NOT stop at a previous `</details>` close.
    Two adjacent `<details>` sample blocks therefore share each other's
    prelude window — mitigated for now by per-block disclosure counting
    (each sample block is enforced independently against the same
    window, and the v2 form puts the cherry-pick disclosure inside the
    `<summary>` which gets folded into the prelude via the
    `block_start`-past-`</summary>` shift). Promote to a real
    `</details>`-boundary stop if a body emerges where two adjacent
    `<details>` blocks legitimately diverge on the disclosure.
    """
    lo = max(0, fence_start - max_chars)
    window = details[lo:fence_start]
    # Don't cross a previous fence's closing line.
    prev_close = window.rfind("\n```")
    if prev_close != -1:
        # Skip past the closing fence line.
        nl = window.find("\n", prev_close + 1)
        if nl != -1:
            window = window[nl + 1 :]
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
    r"\d+ random completions?|\d+ randomly[-\s]?sampled|"
    # `<N> example training rows`, `<N> example eval probes`,
    # `<N> examples of …`, `<N> sample completions`, `<N> sample rows`
    # — the disclosure form used inside `<details>` block summaries
    # (e.g. task #432's "5 example training rows" /
    # "5 example eval probes"). The "example" / "sample" qualifier
    # tells the reader the rows are illustrative, not exhaustive.
    r"\d+\s+(?:examples?|sample[s]?)\b)",
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


def check_required_sections(body: str) -> CheckResult:
    """Check 2: the three required H2 sections appear in order, and no
    retired H2 (`## Details`, `## Figure`) is present.

    The 2-content-section spec (2026-W22 migration, task #454) folds
    the former `## Details` narrative into per-result `### <finding>`
    H3s under `## TL;DR` and inlines figures inside each result H3, so
    a NEW body that still carries either retired H2 is rejected. This
    forces clean migration — bodies cannot half-migrate by stripping
    Details prose while leaving the H2 in place. Legacy bodies
    pre-2026-W22 are forward-grandfathered (the verifier never re-runs
    over them).
    """
    found = [name for name, _, _ in find_h2_sections(body)]
    label = "three required H2 sections in order"
    # Hard FAIL on retired H2s (force clean migration).
    retired_present = [s for s in RETIRED_H2_SECTIONS if s in found]
    if retired_present:
        return CheckResult(
            label,
            False,
            f"retired H2(s) present: {', '.join('## ' + s for s in retired_present)}. "
            "The 2-content-section spec (2026-W22) folds Details into per-result "
            "H3s under `## TL;DR` and inlines figures inside each result H3 — "
            "remove the retired H2 and migrate its content. See "
            ".claude/skills/clean-results/SPEC.md.",
        )
    missing = [s for s in REQUIRED_H2_SECTIONS if s not in found]
    if missing:
        return CheckResult(
            label,
            False,
            f"missing: {', '.join('## ' + s for s in missing)} (found: {found})",
        )
    # Order check: REQUIRED_H2_SECTIONS must appear in this exact order
    # within the body's H2 sequence (extra non-retired H2s after
    # `## Reproducibility` are tolerated, but NOT before).
    seq = [s for s in found if s in REQUIRED_H2_SECTIONS]
    if seq != REQUIRED_H2_SECTIONS:
        return CheckResult(
            label,
            False,
            f"wrong order — got {seq}, expected {REQUIRED_H2_SECTIONS}",
        )
    # Stray H2 check: any non-required, non-retired H2 (e.g., a leftover
    # `## Goal`, `## Background`, `## Methods`) that appears BEFORE the
    # required sequence completes is rejected. The 2-content-section spec
    # tolerates extra H2s ONLY after `## Reproducibility`. Retired H2s
    # already produced a hard FAIL above, so don't double-report them.
    last_required_idx = -1
    stray_before: list[str] = []
    for name, _, _ in find_h2_sections(body):
        if name in REQUIRED_H2_SECTIONS:
            if name == REQUIRED_H2_SECTIONS[-1]:
                last_required_idx = 1
        elif name in RETIRED_H2_SECTIONS:
            continue
        elif last_required_idx == -1:
            stray_before.append(name)
    if stray_before:
        return CheckResult(
            label,
            False,
            f"stray H2(s) before `## {REQUIRED_H2_SECTIONS[-1]}`: "
            f"{', '.join('## ' + s for s in stray_before)}. The 2-content-section "
            "spec (2026-W22) permits extra H2s ONLY after `## Reproducibility` — "
            f"required sequence is {REQUIRED_H2_SECTIONS} and nothing else may "
            "appear in between. Remove the stray H2 or move it after "
            "`## Reproducibility`.",
        )
    return CheckResult(label, True)


def check_tldr_labels(body: str) -> CheckResult:
    """Check 3: `## TL;DR` opens with the Motivation block.

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
    """
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


def _gather_figure_image_urls(body: str) -> list[str]:
    """Collect image URLs inline under `## TL;DR`. Powers checks 4 / 4b
    under the 2-content-section spec (2026-W22, task #454): every figure
    lives inside a per-result H3 inside `## TL;DR`."""
    urls: list[str] = []
    text = section_text(body, "TL;DR")
    if text is not None:
        urls.extend(_IMAGE_RE.findall(text))
    return urls


def check_figure_image(body: str) -> CheckResult:
    """Check 4: at least one `![alt](url)` image exists inline under
    `## TL;DR` (every result H3 in the 2-content-section spec carries
    its own figure)."""
    urls = _gather_figure_image_urls(body)
    if not urls:
        return CheckResult(
            "hero image present",
            False,
            "no `![alt](path)` image found inline under `## TL;DR` — every "
            "result H3 in the 2-content-section spec carries its own figure",
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
    the analyzer.md inlined skeleton) or inside a `<details>` example
    block is NOT misdetected as v2. The sentinel must live at the
    document-level prose layer to count.

    Forward-only marker. Bodies without the sentinel keep the prior
    post-#454 behavior and are NEVER hard-FAILed by the nested-shape
    rule or the no-body-Confidence permission.
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
    cleaned = _DETAILS_BLOCK_RE.sub("", cleaned)
    return CLEAN_RESULT_V2_SENTINEL in cleaned


def check_confidence_matches(body: str) -> CheckResult:
    """Check 6: `Confidence: …` line matches the title.

    Under the 2-content-section nested-design (v2) shape (sentinel
    `<!-- clean-result-v2 -->` present), the H1 title tag is the
    single source of truth for confidence — bodies do NOT carry a
    `Confidence: …` sentence by design. This check PASSes for v2
    bodies whenever the title carries the `(... confidence)` tag,
    regardless of whether a body `Confidence:` sentence exists. If a
    v2 body DOES happen to carry one (legacy holdover), the level
    must match the title and ≥20 chars of rationale after the dash
    must be present (same rule as legacy bodies).

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
    v2 = is_v2_nested_design(body)
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
                # v2 nested-design bodies legitimately have no Confidence
                # sentence — the H1 title tag is the source of truth.
                return CheckResult(
                    label_name,
                    True,
                    f"v2 nested-design (sentinel present); title carries "
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


def check_repro_subgroups(body: str) -> CheckResult:
    """Check 7: `## Reproducibility` contains all three boldface subgroup labels."""
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
    unified 2026-06-09, second #507 follow-up). Shape checks only —
    existence probing for same-repo raw URLs is check 8b's job.
    """
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility URL permanence", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    # Every scan below runs on fence-stripped text: a URL inside a ```
    # example is illustrative, never a provenance link (fence policy
    # shared with check 8b).
    scanned = _strip_fenced_blocks(repro)
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
    in `## Reproducibility`."""
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility sentinel scrub", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    for s in SENTINEL_SUBSTRINGS:
        if s == "{{":
            if "{{" in repro:
                bad.append("`{{` placeholder")
        else:
            # `default` matched as a standalone word (avoid false positives like
            # `default_factory`); the others matched case-insensitively as words.
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
    if not is_v2_nested_design(body):
        return CheckResult(name, True, "skipped — legacy (pre-v2) body")
    repro = section_text(body, "Reproducibility")
    if repro is None:
        # Missing-section is check_required_sections' job; don't double-FAIL.
        return CheckResult(name, True, "skipped — no Reproducibility section")
    body_lrs = _parse_lr_floats(repro, anchored_only=True)
    if not body_lrs:
        return CheckResult(name, True, "skipped — no learning rate stated in Reproducibility")
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
        f"Reproducibility states lr {body_str} but the approved plan declares "
        f"{{{plan_str}}}. Copy the actual run lr from the committed training script "
        f"(the `**Code:**` SHA) / plan §11 — never type it from memory. If the run "
        f"genuinely deviated from the plan, document the deviation explicitly in "
        f"`## Reproducibility` (downgrades this to WARN)."
    )
    if _LR_DEVIATION_RE.search(repro):
        return CheckResult(name, True, "documented deviation — " + detail, is_warn=True)
    return CheckResult(name, False, detail)


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

    A missing row FAILs only when recorded origin data exists —
    frontmatter ``origin_prompt`` or a ``## Provenance`` section in the
    sibling ``original-body.md`` — i.e. the body DROPPED provenance it
    had. With no recorded origin data the miss is a WARN: created_at +
    parent lineage always exist, so the row should still ship, stating
    the prompt was not recorded. Spec:
    `.claude/skills/clean-results/SPEC.md` § `**Context:**` row.
    """
    name = "Reproducibility Context provenance row"
    if not is_v2_nested_design(body):
        return CheckResult(name, True, "skipped — legacy (pre-v2) body")
    repro = section_text(body, "Reproducibility")
    if repro is None:
        # Missing-section is check_required_sections' job; don't double-FAIL.
        return CheckResult(name, True, "skipped — no Reproducibility section")
    if re.search(r"\*\*\s*Context\s*:?\s*\*\*", repro):
        return CheckResult(name, True, "**Context:** row present")
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


def check_cherry_picked_label(body: str) -> CheckResult:
    """Check 10: every sample-output block in `## TL;DR` is preceded
    by a cherry-picked / random-sample disclosure in the prelude prose.

    Under the 2-content-section spec (2026-W22, task #454) sample
    completions live inside per-result H3s under `## TL;DR`, not under
    a separate `## Details`. The check scans `## TL;DR` for BOTH
    fenced code blocks AND `<details>` blocks that carry GFM tables /
    long text (nested-design v2 bodies frequently present training
    rows + eval probes as `<details open>` tables instead of fenced
    code blocks — e.g. task #432). For each sample block the prose
    immediately above must carry the disclosure.
    """
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        return CheckResult("Cherry-picked label discipline", False, "## TL;DR section missing")
    samples = _iter_sample_blocks(tldr)
    if not samples:
        return CheckResult(
            "Cherry-picked label discipline",
            True,
            "no sample-output blocks in `## TL;DR` (fenced or `<details>`)",
        )
    flagged: list[str] = []
    for start, _, content in samples:
        prelude = _prelude_window(tldr, start)
        # For `<details>` blocks the cherry-pick disclosure may live
        # inside the block (the `<summary>` text or the prose around
        # the inner table); we scan BOTH the prelude window AND the
        # inner content. (For fenced code blocks the `content` is the
        # code text — a cherry-pick disclosure there is unusual and
        # harmless to scan; the prelude scan still dominates.)
        if _CHERRY_DISCLOSURE_RE.search(prelude) or _CHERRY_DISCLOSURE_RE.search(content):
            continue
        # First content line, trimmed, as a hint to the user.
        first_line = content.strip().splitlines()[0][:60] if content.strip() else "(empty)"
        flagged.append(first_line)
    if flagged:
        preview = "; ".join(f"'{x}'" for x in flagged[:2]) + (" …" if len(flagged) > 2 else "")
        return CheckResult(
            "Cherry-picked label discipline",
            False,
            f"{len(flagged)} of {len(samples)} sample block(s) lack a cherry-picked / "
            f"random-sample disclosure in the prelude prose: {preview}",
        )
    return CheckResult(
        "Cherry-picked label discipline",
        True,
        f"{len(samples)} sample block(s) labelled",
    )


def check_qualitative_data_link(body: str) -> CheckResult:
    """Check 11: every sample-output block in `## TL;DR` is preceded
    by at least one link or backtick-path that is NOT an aggregate-only
    path.

    An explicit `not uploaded` escape downgrades FAIL to WARN. Scope
    moved from `## Details` to `## TL;DR` under the 2-content-section
    spec (2026-W22, task #454) — sample completions now live inside
    per-result H3s under TL;DR. The check scans BOTH fenced code
    blocks AND `<details>` blocks that carry GFM tables / long text
    (nested-design v2 bodies frequently present training rows + eval
    probes as `<details open>` tables instead of fenced code blocks
    — e.g. task #432).
    """
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        return CheckResult("Qualitative-data link", False, "## TL;DR section missing")
    samples = _iter_sample_blocks(tldr)
    if not samples:
        return CheckResult(
            "Qualitative-data link",
            True,
            "no sample-output blocks in `## TL;DR` (fenced or `<details>`)",
        )
    fails: list[str] = []
    warns: list[str] = []
    passes = 0
    for start, _, content in samples:
        prelude = _prelude_window(tldr, start)
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
                f"'{first_line}': only aggregate-pattern links, `not uploaded` escape acknowledged"
            )
        else:
            fails.append(
                f"'{first_line}': only aggregate-pattern links "
                f"(e.g. {tokens[0][:60]}); raw text-level artifact required"
            )

    if fails:
        return CheckResult(
            "Qualitative-data link",
            False,
            f"{len(fails)} sample block(s) lack a qualitative-data link: "
            + "; ".join(fails[:2])
            + (" …" if len(fails) > 2 else ""),
        )
    if warns:
        return CheckResult(
            "Qualitative-data link",
            True,
            f"{len(warns)} sample block(s) ship with `not uploaded` escape — "
            "follow-up should re-run with raw-completion upload",
            is_warn=True,
        )
    return CheckResult(
        "Qualitative-data link",
        True,
        f"{passes} sample block(s) link to a qualitative-data artifact",
    )


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
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        # Other checks will FAIL on missing sections; don't double-report.
        return CheckResult(
            "planned-vs-actual denominator consistency",
            True,
            "## TL;DR missing — other checks will report",
        )
    # The "scope-correction" text under the 2-content-section spec
    # (2026-W22, task #454) can live anywhere — including inside a
    # `### <finding>` H3 INSIDE `## TL;DR`, in a result H3 outside it,
    # or in legacy in-flight bodies under a retired
    # `### Methodology corrections` H3. The previous narrowing — which
    # excluded `## TL;DR` from the scan — silently lost scope-correction
    # prose that the new spec deliberately folds into TL;DR result H3s.
    # So scan the WHOLE body for scope-correction claims; the TL;DR
    # headline claims still come from `## TL;DR` only.
    scope_text = body

    tldr_claims = _collect_denominator_claims(tldr)
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
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        return CheckResult(
            "TL;DR narrative flow",
            True,
            "no ## TL;DR section to inspect (skipped)",
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
            f"{len(bad_h3_names)} outline-label H3(s) in TL;DR: "
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
            f"{len(dumps)} run(s) of >2 consecutive figures in TL;DR "
            "with no prose between — likely figure-dump. "
            "Add setup + read paragraphs around each figure."
        )

    if findings:
        return CheckResult(
            "TL;DR narrative flow",
            True,
            "; ".join(findings),
            is_warn=True,
        )
    return CheckResult(
        "TL;DR narrative flow",
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


def _resolve_repo_root() -> Path | None:
    """Return the repo root via the existing task_workflow helper, or
    None if the import fails (e.g. running this script outside the repo)."""
    try:
        from explore_persona_space.task_workflow import repo_root  # local import

        return repo_root()
    except Exception:
        return None


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

    Mechanical scope only — the semantic call ("did the experimenter
    actually upload this elsewhere, e.g. HF data repo?") belongs to
    upload-verifier Step 4, not this verifier. This check enforces only
    the within-body promise: if the body says "committed at commit X",
    that sha tree must carry the named file.
    """
    repro = section_text(body, "Reproducibility")
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
    # Collect (sha, paths-on-same-line) pairs. Same-line anchoring keeps
    # the association unambiguous: if a sha and a path are on the same
    # line they are almost certainly being asserted together. Cross-line
    # pairings are intentionally out of scope (too noisy).
    pairs: list[tuple[str, str]] = []
    for line in cleaned.splitlines():
        sha_match = _COMMITTED_AT_SHA_RE.search(line)
        if sha_match is None:
            continue
        sha = sha_match.group("sha")
        path_matches = _ARTIFACT_PATH_RE.findall(line)
        for raw in path_matches:
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
    example is illustrative, never probed. Other hosts (HF Hub, WandB)
    and other-repo GitHub links are out of scope: their existence is not
    decidable from the local object DB, and an unauthenticated 404 on an
    external private repo would false-FAIL. Order-preserving and
    deduplicated (at most one probe per unique URL)."""
    urls: list[str] = []
    for token in _REPRO_URL_TOKEN_RE.findall(_strip_fenced_blocks(repro)):
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
    repro = section_text(body, "Reproducibility")
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


def check_concerns_audit(body: str, *, concerns_path: Path | None = None) -> CheckResult:
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
    for line in concerns_path.read_text().splitlines():
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
    if not open_binding:
        return CheckResult(
            "concerns audit (Lens 14)",
            True,
            f"no open binding concerns (read {len(events)} concern events)",
        )

    # Acknowledgement mechanism 1: any `### <H3>` body inside `## TL;DR`
    # (the 2-content-section spec folds methodology corrections into
    # result H3s — scan them all).
    tldr_body = section_text(body, "TL;DR") or ""
    h3_bodies: list[str] = []
    for h3_match in re.finditer(
        r"^###\s+.+?$(.*?)(?=^###\s|\Z)",
        tldr_body,
        re.MULTILINE | re.DOTALL,
    ):
        h3_bodies.append(h3_match.group(1))
    tldr_h3_text = "\n".join(h3_bodies)

    # Acknowledgement mechanism 2: the `Confidence:` rationale paragraph
    # (lives in `## Reproducibility` under the 2-content-section spec,
    # but scan the whole body for robustness — the verifier already
    # treats Confidence: as a section-agnostic anchor).
    conf_match = re.search(
        r"^Confidence:\s.*?(?=\n\n|\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    conf_body = conf_match.group(0) if conf_match else ""

    # Acknowledgement mechanism 3: explicit deferral HTML comment.
    deferral_re = re.compile(r"<!--\s*concern-deferred:\s*([a-z0-9][a-z0-9-]{1,79})\s*-->")
    deferred_ids = set(deferral_re.findall(body))

    unaddressed: list[str] = []
    for ev in open_binding:
        cid = ev["concern_id"]
        if cid in deferred_ids:
            continue
        if cid in tldr_h3_text or cid in conf_body:
            continue
        unaddressed.append(f"{cid} ({ev.get('severity', 'unknown')})")

    if unaddressed:
        return CheckResult(
            "concerns audit (Lens 14)",
            False,
            f"{len(unaddressed)} open binding concern(s) unaddressed in body: "
            f"{', '.join(unaddressed)}. Acknowledge each in a `## TL;DR` result "
            "H3, the `Confidence:` sentence, or a `<!-- concern-deferred: <id> -->` "
            "HTML marker. See `.claude/agents/clean-result-critic.md` § Lens 14 "
            "and `workflow.yaml § concerns_protocol`.",
        )
    return CheckResult(
        "concerns audit (Lens 14)",
        True,
        f"all {len(open_binding)} open binding concern(s) acknowledged in body",
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
]


def verify_text(
    raw: str,
    *,
    source: str = "",
    concerns_path: Path | None = None,
    plan_path: Path | None = None,
    original_body_path: Path | None = None,
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
    """
    fm, body = split_frontmatter(raw)
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
    overall = all(r.passed for r in results)
    return overall, results


def _load_text_for_issue(number: int) -> tuple[str, Path]:
    from explore_persona_space.task_workflow import find_task_path  # local import

    folder = find_task_path(number)
    body_path = folder / "body.md"
    return body_path.read_text(), body_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--issue", type=int, help="task number to verify")
    grp.add_argument("--file", help="path to a body.md to verify")
    grp.add_argument("--body-stdin", action="store_true", help="read body from stdin")
    args = parser.parse_args()

    concerns_path: Path | None = None
    plan_path: Path | None = None
    original_body_path: Path | None = None
    if args.issue is not None:
        try:
            raw, source_path = _load_text_for_issue(args.issue)
            source = str(source_path)
            concerns_path = source_path.parent / "concerns.jsonl"
            plan_path = source_path.parent / "plans" / "plan.md"
            original_body_path = source_path.parent / "original-body.md"
        except FileNotFoundError as e:
            print(f"verify_task_body: {e}", file=sys.stderr)
            return 2
    elif args.file:
        raw = Path(args.file).read_text()
        source = args.file
        # When verifying a body.md by file path, look for siblings
        # (concerns.jsonl, plans/plan.md, original-body.md) so the Lens 14
        # audit, the check-16 lr reconciliation, and the check-17 context-
        # provenance read fire for analyzer-side dry runs against a body
        # in tasks/<status>/<N>/.
        parent = Path(args.file).resolve().parent
        sibling = parent / "concerns.jsonl"
        if sibling.exists():
            concerns_path = sibling
        plan_sibling = parent / "plans" / "plan.md"
        if plan_sibling.exists():
            plan_path = plan_sibling
        orig_sibling = parent / "original-body.md"
        if orig_sibling.exists():
            original_body_path = orig_sibling
    else:
        raw = sys.stdin.read()
        source = "<stdin>"

    overall, results = verify_text(
        raw,
        source=source,
        concerns_path=concerns_path,
        plan_path=plan_path,
        original_body_path=original_body_path,
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
