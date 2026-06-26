#!/usr/bin/env python3
"""verify_paper.py — mechanical verifier for `paper: true` clean-result papers.

The paper-format counterpart of `verify_task_body.py` (which stays the verifier
for grandfathered markdown bodies). A `paper: true` task's canonical clean-result
is a LaTeX paper at `docs/papers/issue_<N>/`; this gate confirms it compiles
clean, is structurally a paper, hides no confidence, and that every artifact +
cross-reference resolves.

Checks (v1 scope — NO `\\metric` grounding; that is a documented v1.1 addition):

  1. Compile clean (multi-pass): parse `<jobname>.log` + `.blg` for undefined
     refs/citations, package errors, a missing `.bbl`, and post-final rerun
     warnings. The build is done by `scripts/build_paper.py`; this re-checks its
     logs (and re-runs the build when `--build` is passed).
  2. Required sections present + in order: Abstract, Introduction, Methods,
     Results, Discussion, References (\\bibliography), Appendix.
  3. NO confidence anywhere in the paper body (the `(LOW|MODERATE|HIGH
     confidence)` tag + bare `Confidence:` lines are a hard FAIL — confidence
     lives only in the body.md paper-stub frontmatter).
  4. `\\includegraphics` paths are repo-relative-confined (no `..` escaping the
     repo, no absolute paths) AND each resolves on disk via \\graphicspath.
  5. `.bib` entries resolve: every `\\cite{key}` / `\\citep` / `\\citet` key has
     a matching `@type{key,` entry in the per-task `.bib`.
  6. `\\epsref{N}` resolves to a real task in the registry.
  7. Verbatim examples present: the paper SHOWS its data, not just describes it —
     each of the three required example classes (`training-data`, `eval-data`,
     `model-output`) is declared with a `% eps-example: <class>` marker AND the
     body carries real verbatim example environments (epsexample / lstlisting /
     verbatim / quote / quotation / tcolorbox) behind them (incident #657: a
     paper that described every method but shipped zero verbatim text).
  8. Judge prompts present: when the paper uses an LLM judge, it carries a
     dedicated `Judge prompts` / `Judge rubric` appendix (sub)section with the
     verbatim prompt + rubric TEXT for every judge (or the template's
     `% eps-judge-prompts` anchor). No-judge papers pass automatically.
  9. Example provenance pointers (no-invention floor): every `% eps-example:`
     block carries a resolvable pointer to a REAL artifact (`\\epsref{N}`, an
     `issueN_` slug, a `superkaiba1/` HF path, `eval_results/` / `figures/`, a
     `.json(l)` file, or a recognized HF dataset id). A pointer does NOT prove
     the example is genuine — the #657 fabricated-persona block even cited an
     `\\epsref` — so the SEMANTIC reality-check (open the cited artifact, confirm
     the persona / system prompt / completion are real + verbatim) is the
     interpretation-critic's paper-mode Lens 7. This check is the mechanical
     floor: a block with NO pointer is unverifiable by construction.
  10. `paper_manifest.json` complete + HF-PDF-consistent: the COMMITTED local
     artifacts (tex/paper_html, + bib/refs when present) are present on disk with
     matching sha256, AND the PDF is validated via `pdf_hf_url` (present + an
     `https://...` URL), NOT a local file (the PDF lives on the HF data repo, not
     in git — incident #657). The build-time verify passed because the local PDF
     existed then; this check must ALSO pass post-commit when the PDF is HF-only.
     Tolerant of the OLD manifest shape (a `pdf` entry still in `artifacts`): its
     local-existence/hash check is SKIPPED (it is HF-hosted), so an
     already-built old-shape manifest still passes.
  11. The body.md paper-stub is valid (frontmatter `paper: true`, an H1 title, an
     abstract, and a paper link).

Run from repo root:
    uv run python scripts/verify_paper.py --issue 657
    uv run python scripts/verify_paper.py --paper-dir docs/papers/_spike \\
        --jobname issue_657_spike --issue 657 --no-stub   # spike self-test
    uv run python scripts/verify_paper.py --issue 657 --build   # build first

Exit 0 if all checks PASS (WARNs allowed), 1 on any FAIL, 2 on usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Section presence is checked against \section{...} / \begin{abstract} /
# \bibliography / \appendix — robust to whitespace + optional [*].
_REQUIRED_SECTIONS = [
    ("Abstract", re.compile(r"\\begin\{abstract\}")),
    ("Introduction", re.compile(r"\\section\*?\{\s*Introduction\b")),
    ("Methods", re.compile(r"\\section\*?\{\s*Methods?\b")),
    ("Results", re.compile(r"\\section\*?\{\s*Results\b")),
    ("Discussion", re.compile(r"\\section\*?\{\s*Discussion\b")),
    ("References", re.compile(r"\\bibliography\{")),
    # Appendix: \appendix followed by a \section (the template's
    # "\appendix\n\section{Appendix}"). Match \appendix presence.
    ("Appendix", re.compile(r"\\appendix\b")),
]

_CONFIDENCE_TAG_RE = re.compile(r"\((?:LOW|MODERATE|HIGH)\s+confidence\)", re.IGNORECASE)
# NOTE: no leading `%?` — the match must start at `Confidence` so the
# comment-skip in check_no_confidence (which inspects the text BEFORE the match
# on the line) owns comment detection. A leading `%?` would let the match begin
# at the `%`, leaving an empty prefix and defeating the skip (a commented
# `% Confidence:` line in the body would wrongly FAIL).
_CONFIDENCE_LINE_RE = re.compile(r"(?<!\\)Confidence\s*[:=]", re.IGNORECASE)
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_GRAPHICSPATH_RE = re.compile(r"\\graphicspath\{((?:\{[^}]*\})+)\}")
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
_BIBENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
_EPSREF_RE = re.compile(r"\\epsref\{(\d+)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)

# Verbatim-example environments the paper may use for example blocks (training
# rows, eval probes, model completions, judge prompts). `epsexample` is the
# template-provided wrapper; the rest are the standard LaTeX verbatim/quote/box
# environments. The example check accepts ANY of these so a paper that shows its
# examples in a legitimate verbatim block is never false-FAILed.
_EXAMPLE_ENV_RE = re.compile(
    r"\\begin\{(?:epsexample|lstlisting|verbatim|Verbatim|quote|quotation|tcolorbox)\}"
)
# The `% eps-example: <class>` comment marker the template ships immediately
# before each example block. The verifier keys the per-class completeness check
# on these markers, so the class set is reliable + author-declared.
_EXAMPLE_MARKER_RE = re.compile(r"%\s*eps-example:\s*([a-z0-9-]+)", re.IGNORECASE)
# The three example classes a `paper: true` experiment paper MUST carry verbatim.
_REQUIRED_EXAMPLE_CLASSES = ("training-data", "eval-data", "model-output")
# Capturing form of the example-env opener — used to bound a single example
# block's text region (its `\begin{<env>}...\end{<env>}`) so the provenance
# check reads THIS block, not trailing prose, and keys on REAL (non-commented)
# environments — the template ships COMMENTED documentation example blocks with
# `<pinned link>` placeholders that must not be mistaken for real blocks.
_EXAMPLE_BEGIN_RE = re.compile(
    r"\\begin\{(epsexample|lstlisting|verbatim|Verbatim|quote|quotation|tcolorbox)\}"
)
# Unambiguous provenance tokens an example block must carry so the example is
# traceable to a REAL artifact (the no-invention floor — incident #657: a
# fabricated "young child" persona that does not exist in the data). A pointer
# does NOT prove the example is real (the fabricated block even cited an
# \epsref) — that semantic reality-check is the interpretation-critic's job
# (it opens the cited artifact). This check only enforces that a pointer is
# PRESENT, so the reviewer (and a reader) has something to resolve.
_PROVENANCE_TOKEN_RE = re.compile(
    r"\\epsref\{\d+\}"  # typed cross-experiment reference
    r"|issue[_]?\d+[_/][\w./-]+"  # issue518_.../... or issue_657/...
    r"|superkaiba1/"  # the HF data/model repo
    r"|raw_completions"  # the HF raw-completions convention
    r"|eval_results/"  # in-repo structured results
    r"|figures/"  # in-repo figure source
    r"|\b[\w-]+\.jsonl?\b",  # a .json / .jsonl filename
    re.IGNORECASE,
)
# A recognized HF dataset id (e.g. `mlabonne/harmful_behaviors`,
# `superkaiba1/explore-persona-space-data`) — an `<org>/<name>` slash path. The
# token must carry a `_`/`-`/`.` in EITHER segment (checked in `_has_provenance`)
# so prose slashes ("pos/neg", "input/output", "question/instruction") never
# satisfy the check.
_HF_DATASET_ID_RE = re.compile(r"\b[\w][\w.-]*/[\w][\w.-]*\b")
# A judge is in play when the body mentions an LLM judge. We detect the
# project-canonical phrasings so a paper that scores anything with an LLM judge
# (or grader) must carry its verbatim prompt(s)/rubric(s). Every alternative is
# anchored on the literal word "judge"/"grader"/"graded" to preserve the
# zero-false-positive property (clean on "judgement", "prejudge", "judges panel").
_JUDGE_USED_RE = re.compile(
    # "as judge" / "as-judge" / "as a judge" / "LLM-as-a-judge" / "model-as-judge"
    # — hyphen- and article-tolerant so the canonical "LLM-as-a-judge" matches.
    r"\bas[-\s]+(?:an?[-\s]+)?judge\b"
    r"|\bjudge\s+model\b"  # "the judge model"
    r"|\bjudged?\s+(?:by|with|using|for)\b"  # "judged by/with/using/for ..."
    r"|\bLLM[- ]?judge\b"  # "LLM-judge" / "LLM judge" / "LLMjudge"
    # "Claude judge", "model-graded", "sonnet grader", ...
    r"|\b(?:claude|llm|model|sonnet|gpt|haiku|opus)[- ]?(?:judge|grader|graded)\b"
    r"|\bthe\s+judge\b"  # "the judge assigns/labels/..."
    r"|\bjudge\s+(?:prompt|rubric|score|assign|label|rate|grade|verdict)"  # judge + action/artifact
    r"|\b(?:LLM[- ]?)?grader\b"  # "an LLM grader" / "a grader"
    r"|\bmodel[- ]graded\b",  # "model-graded evaluation"
    re.IGNORECASE,
)
# The "Judge prompts" appendix (sub)section the paper must carry when a judge is
# used. Accept a \section / \subsection / \paragraph titled Judge prompt(s) or
# Judge rubric(s), or the template's `% eps-judge-prompts` anchor.
_JUDGE_SECTION_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph)\*?\{\s*Judge\s+(?:prompts?|rubrics?)\b"
    r"|%\s*eps-judge-prompts\b",
    re.IGNORECASE,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    is_warn: bool = False

    def render(self) -> str:
        tag = "WARN" if self.is_warn else ("PASS" if self.passed else "FAIL")
        line = f"  [{tag}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


def _strip_comments(tex: str) -> str:
    return _COMMENT_RE.sub("", tex)


def _body_region(tex: str) -> str:
    """The text between \\begin{document} and \\end{document} (the visible body).

    Confidence + section checks run on this region only (the preamble's macro
    comments must not trip the confidence check, and section detection is the
    body's job).
    """
    start = tex.find(r"\begin{document}")
    end = tex.find(r"\end{document}")
    if start == -1:
        return tex
    return tex[start : (end if end != -1 else len(tex))]


# ─── 1. compile-clean (log + blg parse) ──────────────────────────────────────


def check_compile_clean(paper_dir: Path, jobname: str) -> CheckResult:
    """Parse the build's <jobname>.log + .blg for a clean multi-pass compile."""
    # build_paper.py writes per-pass logs; the final pass log is pass3.
    log = paper_dir / f"{jobname}.pass3.log"
    if not log.exists():
        # fall back to a single jobname.log (an ad-hoc build)
        log = paper_dir / f"{jobname}.log"
    if not log.exists():
        return CheckResult(
            "compile clean",
            False,
            f"no build log found ({jobname}.pass3.log / {jobname}.log) — run "
            "build_paper.py (or pass --build) first",
        )
    text = log.read_text(errors="replace")
    problems: list[str] = []

    if "There were undefined references" in text or "undefined references" in text:
        problems.append("undefined references")
    if re.search(r"Citation `[^']+' on page .* undefined", text) or (
        "There were undefined citations" in text
    ):
        problems.append("undefined citations")
    if re.search(r"^!\s", text, re.MULTILINE) or "Emergency stop" in text:
        problems.append("LaTeX error (`!` line / emergency stop)")
    if re.search(r"LaTeX Error:", text):
        problems.append("LaTeX Error")
    # post-final rerun warnings: 'Rerun to get cross-references right' on the
    # LAST pass means labels still hadn't settled.
    if "Rerun to get" in text:
        problems.append("'Rerun to get ...' on final pass (labels unsettled)")

    # .blg: bibtex log — undefined cites / empty bbl signals.
    blg = paper_dir / f"{jobname}.blg"
    if blg.exists():
        blg_text = blg.read_text(errors="replace")
        if re.search(r"I didn't find a database entry for", blg_text):
            problems.append("bibtex: missing database entry (undefined cite)")
        if re.search(r"Warning--I didn't find a database entry", blg_text):
            problems.append("bibtex: missing database entry (warning)")
    # .bbl must exist + be non-trivial when there are citations.
    bbl = paper_dir / f"{jobname}.bbl"
    if not bbl.exists():
        problems.append(f"no .bbl produced ({jobname}.bbl missing)")

    if problems:
        return CheckResult("compile clean", False, "; ".join(problems))
    return CheckResult("compile clean", True, "0 undefined refs/cites, .bbl present")


# ─── 2. required sections ────────────────────────────────────────────────────


def check_required_sections(tex: str) -> CheckResult:
    body = _strip_comments(_body_region(tex))
    missing = [name for name, rx in _REQUIRED_SECTIONS if not rx.search(body)]
    if missing:
        return CheckResult(
            "required sections",
            False,
            f"missing required section(s): {', '.join(missing)}",
        )
    # order check: each section's first match position must be non-decreasing.
    positions = []
    for name, rx in _REQUIRED_SECTIONS:
        m = rx.search(body)
        positions.append((name, m.start()))
    out_of_order = [
        positions[i][0] for i in range(1, len(positions)) if positions[i][1] < positions[i - 1][1]
    ]
    if out_of_order:
        return CheckResult(
            "required sections",
            False,
            f"sections out of order around: {', '.join(out_of_order)}",
        )
    return CheckResult(
        "required sections", True, "Abstract→Intro→Methods→Results→Discussion→Refs→Appendix"
    )


# ─── 3. no confidence in body ────────────────────────────────────────────────


def check_no_confidence(tex: str) -> CheckResult:
    body = _body_region(tex)
    hits: list[str] = []
    if _CONFIDENCE_TAG_RE.search(body):
        hits.append("`(LOW|MODERATE|HIGH confidence)` tag")
    # bare 'Confidence:' line (uncommented) in the body
    for m in _CONFIDENCE_LINE_RE.finditer(body):
        # ignore if inside a comment line (starts with %)
        line_start = body.rfind("\n", 0, m.start()) + 1
        if body[line_start : m.start()].lstrip().startswith("%"):
            continue
        hits.append("`Confidence:` line")
        break
    if hits:
        return CheckResult(
            "no confidence in body",
            False,
            f"confidence in the paper body ({'; '.join(hits)}) — confidence lives "
            "only in the body.md paper-stub frontmatter",
        )
    return CheckResult("no confidence in body", True, "")


# ─── 4. includegraphics confined + resolves ──────────────────────────────────


def _graphicspath_dirs(tex: str, paper_dir: Path) -> list[Path]:
    """Resolve the \\graphicspath dirs (relative to the paper dir) for lookups."""
    dirs: list[Path] = [paper_dir]
    m = _GRAPHICSPATH_RE.search(tex)
    if m:
        for inner in re.findall(r"\{([^}]*)\}", m.group(1)):
            if inner:
                dirs.append((paper_dir / inner).resolve())
    return dirs


def check_includegraphics(tex: str, paper_dir: Path) -> CheckResult:
    body = _strip_comments(tex)
    paths = _INCLUDEGRAPHICS_RE.findall(body)
    if not paths:
        return CheckResult("includegraphics confined + resolves", True, "no figures")
    gdirs = _graphicspath_dirs(tex, paper_dir)
    problems: list[str] = []
    for raw in paths:
        rel = raw.strip()
        if rel.startswith("/"):
            problems.append(f"absolute path `{rel}`")
            continue
        # confinement: every candidate resolved path must stay under REPO.
        resolved = None
        for gd in gdirs:
            for ext in ("", ".png", ".pdf", ".jpg", ".jpeg"):
                cand = (gd / (rel + ext)).resolve()
                if cand.exists():
                    resolved = cand
                    break
            if resolved:
                break
        if resolved is None:
            problems.append(f"`{rel}` does not resolve under {[str(g) for g in gdirs]}")
            continue
        try:
            resolved.relative_to(REPO)
        except ValueError:
            problems.append(f"`{rel}` resolves OUTSIDE the repo ({resolved})")
    if problems:
        return CheckResult("includegraphics confined + resolves", False, "; ".join(problems))
    return CheckResult(
        "includegraphics confined + resolves", True, f"{len(paths)} figure(s) resolve"
    )


# ─── 5. bib entries resolve ──────────────────────────────────────────────────


def check_bib_resolves(tex: str, paper_dir: Path, jobname: str, issue: str) -> CheckResult:
    body = _strip_comments(tex)
    cite_keys: set[str] = set()
    for m in _CITE_RE.finditer(body):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                cite_keys.add(key)
    if not cite_keys:
        return CheckResult("bib entries resolve", True, "no \\cite keys")
    # bib file: prefer issue_<N>.bib (the \bibliography arg), else any .bib here.
    bib = paper_dir / f"issue_{issue}.bib"
    if not bib.exists():
        bibs = list(paper_dir.glob("*.bib"))
        if not bibs:
            return CheckResult(
                "bib entries resolve",
                False,
                f"{len(cite_keys)} \\cite key(s) but no .bib in {paper_dir}",
            )
        bib = bibs[0]
    defined = set(_BIBENTRY_RE.findall(bib.read_text(errors="replace")))
    missing = sorted(cite_keys - defined)
    if missing:
        return CheckResult(
            "bib entries resolve",
            False,
            f"\\cite key(s) with no .bib entry: {', '.join(missing)}",
        )
    return CheckResult("bib entries resolve", True, f"{len(cite_keys)} cite key(s) resolve")


# ─── 6. epsref resolves to a real task ───────────────────────────────────────


def _registry_task_ids() -> set[str]:
    """All task ids known to the registry. Uses the branch-guarded library
    resolver; falls back to reading REGISTRY.json directly."""
    try:
        from explore_persona_space.task_workflow import registry_path

        reg = json.loads(Path(registry_path()).read_text())
    except Exception:
        reg_file = REPO / "tasks" / "REGISTRY.json"
        if not reg_file.exists():
            return set()
        reg = json.loads(reg_file.read_text())
    ids: set[str] = set()
    tasks = reg.get("tasks", reg)
    for k in tasks:
        if str(k).isdigit():
            ids.add(str(k))
    return ids


def check_epsref_resolves(tex: str) -> CheckResult:
    body = _strip_comments(tex)
    refs = sorted(set(_EPSREF_RE.findall(body)))
    if not refs:
        return CheckResult("epsref resolves", True, "no \\epsref")
    ids = _registry_task_ids()
    if not ids:
        return CheckResult("epsref resolves", True, "registry unavailable — skipped", is_warn=True)
    missing = [r for r in refs if r not in ids]
    if missing:
        return CheckResult(
            "epsref resolves",
            False,
            f"\\epsref to non-existent task(s): {', '.join('#' + m for m in missing)}",
        )
    return CheckResult("epsref resolves", True, f"{len(refs)} \\epsref resolve")


# ─── 7. verbatim examples present (training / eval / model-output) ────────────
# The paper must SHOW its methods AND its data: verbatim training rows, verbatim
# eval probes, and verbatim model outputs (eval input -> output -> judge
# verdict) — not just prose describing them (incident #657: the paper described
# every method but contained zero verbatim text). The check keys on the
# template-shipped `% eps-example: <class>` markers (reliable + author-declared)
# AND requires real verbatim example environments behind them (guards a marker
# with no content).


def check_verbatim_examples(tex: str) -> CheckResult:
    body = _body_region(tex)
    # Count actual verbatim example environments (any accepted env). The marker
    # detection scans the WHOLE body (comments are not stripped — the markers
    # ARE comments).
    n_envs = len(_EXAMPLE_ENV_RE.findall(_strip_comments(body)))
    classes = {m.group(1).lower() for m in _EXAMPLE_MARKER_RE.finditer(body)}
    problems: list[str] = []
    missing = [c for c in _REQUIRED_EXAMPLE_CLASSES if c not in classes]
    if missing:
        problems.append(
            "missing required `% eps-example:` class marker(s): "
            + ", ".join(missing)
            + " (each example block needs a `% eps-example: <training-data|"
            "eval-data|model-output>` marker)"
        )
    # A paper that declares the class markers but has no actual verbatim block is
    # a marker-without-content FAIL; a paper with neither is the #657 case.
    if n_envs == 0:
        problems.append(
            "no verbatim example environment "
            "(epsexample / lstlisting / verbatim / quote / quotation / tcolorbox) "
            "— show the actual training rows, eval probes, and model completions, "
            "not just prose describing them"
        )
    if problems:
        return CheckResult("verbatim examples present", False, "; ".join(problems))
    return CheckResult(
        "verbatim examples present",
        True,
        f"{n_envs} verbatim block(s); classes present: {', '.join(sorted(classes))}",
    )


# ─── 8. judge prompts present when a judge is used ────────────────────────────
# Every LLM judge in the study must have its ACTUAL prompt + rubric TEXT in a
# dedicated "Judge prompts" appendix (sub)section — verbatim, not paraphrased
# (incident #657: a sycophancy / EM / refusal / steering-sanity judge was named
# but no prompt text shipped). Only fires when the paper actually uses a judge.


def check_judge_prompts(tex: str) -> CheckResult:
    body = _strip_comments(_body_region(tex))
    if not _JUDGE_USED_RE.search(body):
        return CheckResult("judge prompts present", True, "no LLM judge used")
    # search the FULL body (un-stripped) so the `% eps-judge-prompts` anchor — a
    # comment — is detectable, alongside a real \section{Judge prompts}.
    if not _JUDGE_SECTION_RE.search(_body_region(tex)):
        return CheckResult(
            "judge prompts present",
            False,
            "the paper uses an LLM judge but carries no `Judge prompts` / "
            "`Judge rubric` appendix section — add the verbatim prompt + rubric "
            "TEXT for every judge (a `% eps-judge-prompts` anchor + a "
            r"\subsection{Judge prompts} satisfies this)",
        )
    return CheckResult("judge prompts present", True, "Judge prompts section present")


# ─── 9. example provenance pointers (no-invention floor) ─────────────────────
# Every `% eps-example:` block must carry a resolvable provenance pointer to a
# REAL artifact (an \epsref{N}, an issueN_ slug, a superkaiba1/ HF path,
# eval_results/ / figures/, a .json(l) filename, or a recognized HF dataset id).
# This is the mechanical floor of the no-invention rule: it does NOT prove an
# example is genuine (the #657 fabricated persona block even cited \epsref{612}),
# but a block with NO pointer is unverifiable by construction, and a present
# pointer gives the interpretation-critic an artifact to open and check against
# (the semantic reality-check, interpretation-critic.md paper-mode Lens 7).


def _line_is_commented(body: str, pos: int) -> bool:
    """True when the text from the start of `pos`'s line up to `pos` begins with `%`
    (i.e. the construct at `pos` is inside a LaTeX comment)."""
    line_start = body.rfind("\n", 0, pos) + 1
    return body[line_start:pos].lstrip().startswith("%")


def _example_block_regions(body: str) -> list[tuple[str, int, str]]:
    """Per REAL (non-commented) `% eps-example:`-declared example block:
    (class, begin_pos, block_text).

    Keys on REAL example ENVIRONMENTS (a non-commented `\\begin{<env>}...
    \\end{<env>}`), not on the `% eps-example:` markers themselves, because the
    template ships COMMENTED documentation example blocks (with `<pinned link>`
    placeholders) whose marker lines would otherwise be mistaken for real
    blocks. A real env is a declared example iff a `% eps-example:` marker
    appears in the gap since the previous real env (which it introduces); a real
    env with no such marker in its gap (a judge-prompt `verbatim` block, an
    incidental quote) is NOT a declared example and is skipped — judge prompts
    are governed by check 8, not the example-provenance floor.
    """
    regions: list[tuple[str, int, str]] = []
    prev_end = 0
    for m in _EXAMPLE_BEGIN_RE.finditer(body):
        if _line_is_commented(body, m.start()):
            continue  # commented template documentation block, not a real one
        env = m.group(1)
        end_m = re.compile(r"\\end\{" + re.escape(env) + r"\}").search(body, m.end())
        end = end_m.end() if end_m else len(body)
        gap_markers = list(_EXAMPLE_MARKER_RE.finditer(body[prev_end : m.start()]))
        if gap_markers:  # this real env is the one the marker introduces
            regions.append((gap_markers[-1].group(1).lower(), m.start(), body[m.start() : end]))
        prev_end = end
    return regions


# In the .tex, paths carry LaTeX-escaped specials (`issue657\_alignment...`,
# `raw\_completions`, `mlabonne/harmful\_behaviors`). Unescape before scanning so
# the provenance tokens — which ARE present — are detected (the `\_` is the
# common case; `\%`/`\&`/`\#`/`\$` round it out).
_LATEX_ESCAPE_RE = re.compile(r"\\([_%&#$])")


def _delatex(s: str) -> str:
    return _LATEX_ESCAPE_RE.sub(r"\1", s)


def _has_provenance(block: str) -> bool:
    b = _delatex(block)
    if _PROVENANCE_TOKEN_RE.search(b):
        return True
    # An `<org>/<name>` token counts only if a segment carries a `_`/`-`/`.`,
    # which excludes prose slashes ("pos/neg", "input/output").
    return any(("_" in t or "-" in t or "." in t) for t in _HF_DATASET_ID_RE.findall(b))


def check_example_provenance(tex: str) -> CheckResult:
    body = _body_region(tex)
    regions = _example_block_regions(body)
    if not regions:
        # The verbatim-examples check (7) already FAILs a paper with no example
        # markers; provenance is vacuously satisfied here.
        return CheckResult("example provenance pointers", True, "no example blocks")
    missing: list[str] = []
    for cls, _start, block in regions:
        if not _has_provenance(block):
            missing.append(cls)
    if missing:
        return CheckResult(
            "example provenance pointers",
            False,
            f"{len(missing)} example block(s) carry NO provenance pointer "
            f"(classes: {', '.join(missing)}) — every verbatim example must cite a "
            "real artifact (\\epsref{N}, an issueN_ slug, a superkaiba1/ HF path, "
            "eval_results/ / figures/, a .json(l) file, or an HF dataset id) IN THE "
            "BLOCK'S CAPTION OR BODY (a pointer in the preceding prose is not seen) "
            "so it is traceable and the interpretation-critic can verify it is not "
            "invented",
        )
    return CheckResult(
        "example provenance pointers", True, f"{len(regions)} example block(s) all cite an artifact"
    )


# ─── 10. manifest complete + HF-PDF-consistent ───────────────────────────────
# The PDF lives on the HF data repo, NOT in git, so it is NEVER required as a
# local on-disk artifact (incident #657: the local PDF exists at build time but
# is gone post-commit). The committed artifacts (tex/paper_html, + bib/refs when
# present) ARE locally validated; the PDF is validated via `pdf_hf_url`.
_REQUIRED_MANIFEST_ARTIFACTS = ("tex", "paper_html")
# A `pdf` entry may still appear in `artifacts` in the OLD manifest shape; it is
# HF-hosted, so its local-existence/hash check is skipped (validated via the URL).
_HF_HOSTED_ARTIFACTS = ("pdf",)


def check_manifest(paper_dir: Path) -> CheckResult:
    mf = paper_dir / "paper_manifest.json"
    if not mf.exists():
        return CheckResult("manifest complete + HF-PDF-consistent", False, f"no {mf}")
    try:
        manifest = json.loads(mf.read_text())
    except json.JSONDecodeError as e:
        return CheckResult("manifest complete + HF-PDF-consistent", False, f"invalid JSON: {e}")
    artifacts = manifest.get("artifacts", {})
    problems: list[str] = []
    for req in _REQUIRED_MANIFEST_ARTIFACTS:
        if req not in artifacts:
            problems.append(f"missing required committed artifact `{req}`")
    for label, rec in artifacts.items():
        # The PDF is HF-hosted (validated via pdf_hf_url below) — never stat it
        # locally, even if an old-shape manifest still lists it in `artifacts`.
        if label in _HF_HOSTED_ARTIFACTS:
            continue
        rel = rec.get("path")
        if not rel:
            problems.append(f"artifact `{label}` has no path")
            continue
        f = REPO / rel
        if not f.exists():
            problems.append(f"artifact `{label}` path missing on disk: {rel}")
            continue
        want = rec.get("sha256")
        if want:
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            got = h.hexdigest()
            if got != want:
                problems.append(f"artifact `{label}` sha256 mismatch ({rel})")
    # The PDF is validated via the HF URL (top-level `pdf_hf_url`, or the
    # `hf_pdf.url` block in the new shape), NOT a local file.
    pdf_url = manifest.get("pdf_hf_url") or (manifest.get("hf_pdf") or {}).get("url")
    if problems:
        return CheckResult("manifest complete + HF-PDF-consistent", False, "; ".join(problems))
    # pdf_hf_url presence is a WARN when absent (local-only / --no-upload build),
    # not a FAIL — a paper can be verified pre-upload.
    if not pdf_url:
        return CheckResult(
            "manifest complete + HF-PDF-consistent",
            True,
            "committed hashes match; pdf_hf_url not yet set (local build)",
            is_warn=True,
        )
    if not str(pdf_url).startswith("https://"):
        return CheckResult(
            "manifest complete + HF-PDF-consistent",
            False,
            f"pdf_hf_url is not an https:// URL: {pdf_url!r}",
        )
    return CheckResult(
        "manifest complete + HF-PDF-consistent", True, "committed hashes match; pdf_hf_url pinned"
    )


# ─── 11. paper-stub body.md valid ────────────────────────────────────────────


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML-frontmatter split (avoids importing the full lib).

    Returns (frontmatter_dict, body). Only the flat scalar fields the stub check
    needs are parsed (paper, title); robust to the canonical `---\\n...\\n---\\n`.
    """
    fm: dict[str, str] = {}
    if not text.startswith("---"):
        return fm, text
    end = text.find("\n---", 3)
    if end == -1:
        return fm, text
    head = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    for line in head.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip("'\"")
    return fm, body


def check_paper_stub(stub_path: Path) -> CheckResult:
    if not stub_path.exists():
        return CheckResult("paper-stub body.md valid", False, f"no {stub_path}")
    text = stub_path.read_text(errors="replace")
    fm, body = _split_frontmatter(text)
    problems: list[str] = []
    if str(fm.get("paper", "")).lower() != "true":
        problems.append("frontmatter `paper: true` missing")
    if not re.search(r"^#\s+\S", body, re.MULTILINE):
        problems.append("no H1 `# <title>`")
    # an abstract: either an '## Abstract' H2 or a paragraph after the title.
    has_abstract = bool(re.search(r"^##\s+Abstract\b", body, re.MULTILINE)) or (
        len(re.sub(r"^#.*$", "", body, flags=re.MULTILINE).strip()) >= 80
    )
    if not has_abstract:
        problems.append("no abstract")
    # a paper link: a docs/papers/issue_<N>/ link or an HF papers/ URL.
    if not re.search(r"docs/papers/issue_\d+|/papers/issue_\d+|paper\.html", body):
        problems.append("no paper link (docs/papers/issue_<N>/ or HF papers/ URL)")
    if problems:
        return CheckResult("paper-stub body.md valid", False, "; ".join(problems))
    return CheckResult("paper-stub body.md valid", True, "")


# ─── runner ──────────────────────────────────────────────────────────────────


def verify(
    paper_dir: Path,
    jobname: str,
    issue: str,
    *,
    stub_path: Path | None,
    rebuild: bool,
) -> list[CheckResult]:
    tex_path = paper_dir / f"{jobname}.tex"
    if not tex_path.exists():
        return [CheckResult("paper .tex present", False, f"no {tex_path}")]

    if rebuild:
        build = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "build_paper.py"),
                "--paper-dir",
                str(paper_dir.relative_to(REPO)),
                "--jobname",
                jobname,
                "--issue",
                issue,
                "--no-upload",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            tail = "\n".join((build.stdout + build.stderr).splitlines()[-20:])
            return [CheckResult("build (--build)", False, f"build_paper.py failed:\n{tail}")]

    tex = tex_path.read_text(errors="replace")
    results = [
        check_compile_clean(paper_dir, jobname),
        check_required_sections(tex),
        check_no_confidence(tex),
        check_includegraphics(tex, paper_dir),
        check_bib_resolves(tex, paper_dir, jobname, issue),
        check_epsref_resolves(tex),
        check_verbatim_examples(tex),
        check_judge_prompts(tex),
        check_example_provenance(tex),
        check_manifest(paper_dir),
    ]
    if stub_path is not None:
        results.append(check_paper_stub(stub_path))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--issue", type=int, help="task id (docs/papers/issue_<N>/)")
    ap.add_argument("--paper-dir", type=str, help="explicit paper dir (overrides --issue)")
    ap.add_argument("--jobname", type=str, help="tex jobname (default issue_<N>)")
    ap.add_argument(
        "--stub",
        type=str,
        help="path to the body.md paper-stub (default: tasks/<status>/<N>/body.md "
        "resolved via the registry)",
    )
    ap.add_argument("--no-stub", action="store_true", help="skip the body.md paper-stub check")
    ap.add_argument("--build", action="store_true", help="run build_paper.py first")
    args = ap.parse_args()

    if args.paper_dir:
        paper_dir = (REPO / args.paper_dir).resolve()
        jobname = args.jobname or f"issue_{args.issue}"
    elif args.issue is not None:
        paper_dir = REPO / "docs" / "papers" / f"issue_{args.issue}"
        jobname = args.jobname or f"issue_{args.issue}"
    else:
        ap.error("one of --issue or --paper-dir is required")

    if not paper_dir.is_dir():
        ap.error(f"paper dir not found: {paper_dir}")

    # issue number from jobname/issue for bib + manifest labels.
    issue = str(args.issue) if args.issue is not None else _issue_from_jobname(jobname)

    stub_path: Path | None = None
    if not args.no_stub:
        if args.stub:
            stub_path = (REPO / args.stub).resolve()
        elif args.issue is not None:
            stub_path = _resolve_stub(args.issue)
            if stub_path is None:
                print(
                    f"NOTE: could not resolve body.md for issue #{args.issue}; "
                    "skipping the stub check (pass --stub or --no-stub).",
                    file=sys.stderr,
                )

    results = verify(paper_dir, jobname, issue, stub_path=stub_path, rebuild=args.build)

    print(f"verify_paper.py — issue #{issue} ({paper_dir.relative_to(REPO)})")
    for r in results:
        print(r.render())
    fails = [r for r in results if not r.passed]
    warns = [r for r in results if r.is_warn]
    print(
        f"\n{'FAIL' if fails else 'PASS'}: "
        f"{len(results) - len(fails)}/{len(results)} checks passed"
        + (f", {len(warns)} warn" if warns else "")
    )
    return 1 if fails else 0


def _issue_from_jobname(jobname: str) -> str:
    parts = jobname.split("_")
    for i, p in enumerate(parts):
        if p == "issue" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return parts[i + 1]
    return jobname


def _resolve_stub(issue: int) -> Path | None:
    """Resolve the task's body.md via the branch-guarded library resolver."""
    try:
        from explore_persona_space.task_workflow import find_task_path

        d = find_task_path(issue)
        if d is not None:
            body = Path(d) / "body.md"
            if body.exists():
                return body
    except Exception:
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
