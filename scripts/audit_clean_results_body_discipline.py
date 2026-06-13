"""Audit awaiting-promotion bodies for the body-discipline anti-patterns
identified during the 2026-05-08 mass-migration title pass.

Usage:

    # Audit a single task body (preferred for /issue Step 9a-bis):
    uv run python scripts/audit_clean_results_body_discipline.py --task <N>

    # Audit a local markdown file (e.g. an analyzer draft in /tmp):
    uv run python scripts/audit_clean_results_body_discipline.py /tmp/draft.md

    # Legacy bulk-inventory mode (no argument) — reads the pre-built
    # `.claude/cache/audit-2026-05-08/inventory.json` and writes the
    # findings markdown for every awaiting-promotion body listed there.

Bodies are NOT modified.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

OUT_DIR = Path(".claude/cache/audit-2026-05-08")
FINDINGS_PATH = OUT_DIR / "findings.md"
INVENTORY_PATH = OUT_DIR / "inventory.json"

PATTERNS: dict[str, tuple[str, str]] = {
    # name: (regex, plain-English description)
    "pre_reg": (
        r"pre-?registered|pre-?registration|(?<![a-z])pre-reg(?![a-z])|registered hypothesis|registered alpha|fail at the gate|passed the gate|gate-pre-?registered",
        "Pre-registration jargon ('pre-registered', 'fail at the gate', 'gate-passed', etc.)",
    ),
    "verdict_caps": (
        r"\b(?:REJECTED|INDETERMINATE|PASSED|EXCEEDING)\b",
        "Pre-registration gate verdicts in CAPS (REJECTED / INDETERMINATE / PASSED / EXCEEDING)",
    ),
    "effect_size_pp": (
        r"Δ-?\d+\s*p?p|Δrate\s*=|Δ\s*=\s*[+-]?\d+\s*(?:pp|%)",
        "Effect-size-in-percentage-points (Δ-Npp / Δrate / Δ = -Npp)",
    ),
    "interval_inline": (
        r"slope\s*\[[-+\d., ]+\]|\[[-+]?\d+\.\d+\s*,\s*[-+]?\d+\.\d+\]\s*(?:excludes|includes|pp\b|%|\(|on\s)",
        "Credence intervals as inline [low, high] in prose (banned)",
    ),
    "named_tests": (
        r"\bpaired t-test\b|\bFisher(?:'s)? exact\b|\bMann-Whitney\b|\bbootstrap test\b|\bWilcoxon\b",
        "Named statistical tests in prose (paired t-test / Fisher / Mann-Whitney / Wilcoxon)",
    ),
    "h_symbols": (
        r"\bH_[a-zA-Z0-9]+\b|\bH[_-]?main\b",
        "Statistical-hypothesis symbols (H_a / H_0 / H_1 / H_main) without definition",
    ),
    "letter_labels": (
        r"\(\s*(?:[a-c]|[ivx]+)\s*\)\s+(?:slope|the|rate|sub-experiment)",
        "Anaphoric letter labels ('(a) slope ...', '(b) the ...') in prose",
    ),
    "bin_alpha": (
        r"\bBin\s+[A-E](?!\s*[a-z])",
        "Project-internal Bin labels (Bin A / Bin B / Bin C / Bin D / Bin E) without inline definition",
    ),
    "condition_labels": (
        r"\b[CcHhP][1-9](?:'|′)?(?:\s*(?:condition|control|completion|coefficient|hypothesis|test|sub-?(?:claim|experiment|hypothesis)))?(?![a-zA-Z0-9_])",
        "Project-internal condition/hypothesis labels (C1/C2/C3, H1/H2/H3, P1/P2/P3 with optional prime) — replace with named conditions inline",
    ),
    "cell_tags": (
        # Per-cell / per-condition / per-judge plan-internal tags:
        #   BS_E0, BS_E_42, Z_assistant, Z_villain (uppercase + underscore + alphanum)
        #   B0 / B1 as standalone (not "B0:" inside table headers — check context)
        #   G6 / G0a / G2-escalation (judge / gate labels)
        #   M1 / M2 (extraction-method labels — only flag when paired with "cosine"/"cell"/"method")
        #   "Method A" / "Method B" (extraction-method labels — uppercase Method + capital letter)
        r"\bBS_E[0-9A-Za-z_]*|\bZ_[a-zA-Z_]+|\b[Gg][0-9]+[a-c]?\b(?=\s|:|\.|,|$)|\bMethod\s+[AB]\b|\b[Mm][1-9]\b(?=\s+(?:cosine|cell|mean|extraction|method|sub-experiment))",
        "Plan-internal per-cell / extraction-method / judge / gate tags (BS_E*, Z_*, G*, Method A/B, M1) — replace with plain English; tags go in <details>Setup details</details>",
    ),
    "experimental_arm": (
        # "arm" / "arms" used as a project-internal experiment-strand label.
        # Excludes legitimate uses: "arm rest", "human arm", "arm yourself".
        # Triggers on: "<adj>-arm", "the <adj> arm", "behavioral arm", "geometric arm",
        # "five arms", "experimental arm(s)".
        r"\b(?:experimental|behavioral|geometric|reverse-?order|forward-?order|length(?:[-/\s]style)?|full[-\s]?param(?:eter)?|LoRA)\s+arms?\b|\b(?:five|four|three|two)\s+arms?\b|\bexperimental\s+arms?\b|\b(?:the|a)\s+(?:behavioral|geometric|reverse-?order|forward-?order|length(?:[-/\s]style)?|full[-\s]?param(?:eter)?|LoRA)\s+arm\b",
        "Project-internal experiment-strand 'arm' label — describe what was done, not the strand's name",
    ),
    "bare_method_acronym": (
        r"\b(?:GCG|PAIR|EvoPrompt|nanoGCG)\b",
        "Methodology acronyms (GCG / PAIR / EvoPrompt / nanoGCG) — flag for definition check",
    ),
    "stats_acronyms": (
        r"\b(?:OLS|MLE|ANOVA|ROC)\b",
        "Statistical acronyms (OLS / MLE / ANOVA / ROC) without inline definition",
    ),
    "auc_bare": (
        r"\bAUC\s*=\s*0\.\d+",
        "AUC = X.XX values — verify each is paired with what it's computed on",
    ),
    "post_hoc_phrasing": (
        r"\bpost-hoc\b|\bex post\b",
        "'post-hoc' / 'ex post' — academic-paper register; usually droppable",
    ),
    "math_notation": (
        # Identifier with caret-superscript (R^P2, R_B^P2, R_BgivenA^P2),
        # OR identifier with two-segment underscore-subscript that is itself
        # capitalized math notation (R_BgivenA, P_TopK). The second arm is
        # narrower than rule 10's h_symbols catch (which is H_*-specific) so
        # we only flag CamelCase / multi-letter subscripts that look like
        # math identifiers — not file paths or `eval_results/foo` variables.
        r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*\^[A-Za-z0-9_*+\-]+|\b[A-Z]_[A-Z][A-Za-z]{2,}\b",
        "Math-style subscript/superscript notation in prose (R_BgivenA^P2, R^P2, P_TopK) — markdown doesn't render these",
    ),
    "byte_identical": (
        # "byte identical" / "byte-identical" anywhere in body prose. Banned
        # 2026-W22 (task #454) — the phrase reads as AI-slop in research
        # writing. Use plain English: "the two files matched exactly",
        # "every byte agreed", "no diff between the runs".
        r"\bbyte[\s-]identical\b",
        (
            "Use plain English ('the two files matched exactly', 'every byte agreed', "
            "'no diff') instead of 'byte identical' / 'byte-identical' — the phrase "
            "reads as AI-slop in research prose"
        ),
    ),
}


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def list_awaiting_promotion() -> list[dict]:
    """Read pre-built inventory.json (from bash paginator) — Python's gh
    GraphQL pagination chokes on cursors with certain characters."""
    return json.loads(INVENTORY_PATH.read_text())


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block (``---`` … ``---``).

    The anti-pattern audit is about PROSE discipline in the body. YAML
    frontmatter carries structured metadata — e.g. ``relates_to: [d1, d3,
    h2]`` open-question IDs — that is not prose and must not be scanned
    for project-internal-label patterns (``h2`` is an open-question ID,
    not a ``H2`` hypothesis label).
    """
    if text.startswith("---"):
        m = re.match(r"^---\n.*?\n---\n", text, flags=re.DOTALL)
        if m:
            return text[m.end() :]
    return text


def strip_code(text: str) -> str:
    """Remove fenced code blocks and inline-backtick spans."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


# The `## Reproducibility` `**Context:**` provenance row (SPEC.md
# § `**Context:**` row; verify_task_body.py check 17) requires the
# originating user prompt / follow-up scope note be carried forward
# VERBATIM as a blockquote — never paraphrased, trimmed, or typo-fixed.
# Verbatim preservation and the prose anti-pattern scan are mutually
# unsatisfiable on that quote (task #597: a scope note opening with
# "PRE-REGISTERED" tripped the `pre_reg` pattern), so blockquote lines
# inside the Context block are exempt from the scan. Non-blockquote
# prose inside the block is still scanned.
_CONTEXT_LABEL_RE = re.compile(r"^(?:[-*]\s+)?\*\*\s*Context\s*:?\s*\*\*")
_BOLD_LABEL_RE = re.compile(r"^(?:[-*]\s+)?\*\*\s*([^*\n]+?)\s*:?\s*\*\*")
# SPEC.md names exactly three Context sub-bullets; a boldface label
# outside this set (e.g. **Compute:**, **Code:**) starts a sibling row
# and ends the block. Plain (non-bold) sub-bullets never match
# _BOLD_LABEL_RE, so they keep the block open without this whitelist.
_CONTEXT_SUB_LABELS = ("created", "follow-up to", "originating prompt")


def strip_context_blockquotes(text: str) -> str:
    """Drop blockquote lines inside the `**Context:**` provenance block.

    The block runs from the `**Context:**` label to the next markdown
    heading or the next boldface row label that is not one of the
    Context sub-bullets (Created / run, Follow-up to, Originating
    prompt(s)), or EOF. If the boundary is mis-detected the failure
    mode is the pre-fix behavior (the quote gets scanned) — never a
    silently widened exemption.
    """
    out_lines: list[str] = []
    in_context = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_context:
            label_match = _BOLD_LABEL_RE.match(stripped)
            if stripped.startswith("#") or (
                label_match
                and not any(
                    label_match.group(1).lower().startswith(sub) for sub in _CONTEXT_SUB_LABELS
                )
            ):
                in_context = False
            elif stripped.startswith(">"):
                continue  # verbatim provenance quote — exempt from the scan
        if not in_context and _CONTEXT_LABEL_RE.match(stripped):
            in_context = True
        out_lines.append(line)
    return "\n".join(out_lines)


_H2_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")


def strip_data_example_blocks(text: str) -> str:
    """Drop `<details>...</details>` example blocks inside the `## Data`
    section.

    The v3 clean-result spec MANDATES verbatim training rows / eval
    probes / sample completions inside `## Data` (`### Trained on` /
    `### Evaluated with` / `### Generated`), carried in `<details>`
    example blocks. Those verbatim rows routinely contain strings the
    prose anti-pattern scan flags as project-internal condition codes
    (`C1`, `H2`, `M1`, `BS_E0`, …) with NO reword option — the same
    verbatim-content conflict the `**Context:**` blockquote carve-out
    fixed (incident #597). The author cannot paraphrase a row that is
    required to be verbatim, so example blocks inside `## Data` are
    exempt from the scan.

    Mechanism mirrors :func:`strip_context_blockquotes`: a stateful
    line walker that drops only lines inside a `<details>` block while
    the cursor is inside the `## Data` section. Fenced code blocks (the
    other v3 example-block form) are already removed globally by
    :func:`strip_code`, so this only needs to handle `<details>` blocks.

    The `## Data` section runs from its `## ` H2 to the next `## ` H2
    (typically `## Reproducibility`) or EOF. If a `</details>` close is
    never seen before the section ends, the block-drop ends with the
    section — a mis-detected boundary degrades to the pre-fix behavior
    (the lines get scanned), never a silently widened exemption.
    """
    out_lines: list[str] = []
    in_data = False
    in_details = False
    for line in text.splitlines():
        stripped = line.strip()
        h2 = _H2_RE.match(stripped)
        if h2:
            # Any H2 ends a `## Data` block (and any in-flight details
            # drop); re-enter only when the new H2 is `## Data` itself.
            in_data = h2.group("title").strip().lower() == "data"
            in_details = False
            out_lines.append(line)
            continue
        if in_data:
            lowered = stripped.lower()
            if not in_details and lowered.startswith("<details"):
                in_details = True
                continue  # drop the verbatim example block
            if in_details:
                if "</details>" in lowered:
                    in_details = False
                continue  # drop every line inside the example block
        out_lines.append(line)
    return "\n".join(out_lines)


def is_v2(body: str) -> bool:
    """Return True when a body is treated as a "current spec" body for
    the legacy bulk-inventory audit.

    Historically this matched the retired AI TL;DR / AI Summary
    four-H2 shape. Under the 2-content-section nested-design (v2) spec
    (`.claude/skills/clean-results/SPEC.md`, migrated 2026-W22 task
    #454 + nested-TL;DR adoption forward-only) and the five-flat-H2
    (v3) redesign (2026-W24, sentinel `<!-- clean-result-v3 -->`),
    "current spec" now means ANY of:

    - The v3 sentinel `<!-- clean-result-v3 -->` is present (the
      current prescriptive shape: Takeaways / What I ran / Findings /
      Data / Reproducibility); OR
    - The nested-design (v2) sentinel `<!-- clean-result-v2 -->` is
      present in the body (prior prescriptive shape); OR
    - The body carries `## Human TL;DR` AND `## TL;DR` AND
      `## Reproducibility` H2s (the post-#454 flat shape, still
      promotable for legacy bodies); OR
    - Legacy fallback: the retired "AI TL;DR" / "AI Summary" markers
      (kept so the bulk-inventory audit doesn't drop pre-#454 bodies
      from consideration).

    This is a coarse "should I audit this body's prose" gate consulted
    ONLY on the bulk-inventory path (`_run_legacy_bulk_inventory`); the
    live pipeline paths (`--task <N>` at /issue Step 9a-bis, the
    explicit-file path for analyzer drafts) audit UNCONDITIONALLY and do
    NOT consult this gate. It is NOT a structural verifier —
    `scripts/verify_task_body.py` is the authoritative mechanical gate.
    """
    if "<!-- clean-result-v3 -->" in body:
        return True
    if "<!-- clean-result-v2 -->" in body:
        return True
    if "## Human TL;DR" in body and "## TL;DR" in body and "## Reproducibility" in body:
        return True
    # Legacy AI TL;DR / AI Summary fallback (pre-#454 shape).
    return "## AI TL;DR (human reviewed)" in body or (
        "## AI TL;DR" in body and "## AI Summary" in body
    )


def audit_body(body: str) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    cleaned = strip_code(
        strip_data_example_blocks(strip_context_blockquotes(strip_frontmatter(body)))
    )
    for name, (pattern, _) in PATTERNS.items():
        flags = re.IGNORECASE if name == "pre_reg" else 0
        matches = list(re.finditer(pattern, cleaned, flags))
        if matches:
            findings[name] = [m.group(0) for m in matches[:5]]
    return findings


def _resolve_task_body_path(task_number: int) -> Path:
    """Resolve `tasks/<status>/<task_number>/body.md` via the
    task_workflow helper (same lookup used by `verify_task_body.py`)."""
    from explore_persona_space.task_workflow import find_task_path

    return find_task_path(task_number) / "body.md"


def _audit_single_body(body: str) -> int:
    findings = audit_body(body)
    if not findings:
        print("PASS: no body-discipline anti-patterns matched")
        return 0
    print("FAIL: body-discipline anti-patterns matched")
    for name, samples in findings.items():
        print(f"- {name}: {', '.join(repr(s) for s in samples[:3])}")
    return 1


def _run_legacy_bulk_inventory() -> None:
    """Legacy bulk-inventory mode: read pre-built inventory.json and write
    findings markdown across all awaiting-promotion items."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = list_awaiting_promotion()
    print(f"Found {len(items)} awaiting-promotion items")
    INVENTORY_PATH.write_text(json.dumps(items, indent=2))

    issue_findings: list[tuple[int, str, bool, dict[str, list[str]]]] = []
    for it in items:
        n = it["number"]
        body = gh("api", f"repos/superkaiba/explore-persona-space/issues/{n}", "--jq", ".body")
        v2 = is_v2(body)
        findings = audit_body(body) if v2 else {}
        issue_findings.append((n, it["title"], v2, findings))

    pattern_counts: dict[str, int] = {k: 0 for k in PATTERNS}
    issues_by_pattern: dict[str, list[int]] = {k: [] for k in PATTERNS}
    for n, _t, v2, findings in issue_findings:
        if not v2:
            continue
        for k in findings:
            pattern_counts[k] += 1
            issues_by_pattern[k].append(n)

    lines = ["# Body-discipline audit — 2026-05-08", ""]
    lines.append(f"Total awaiting-promotion items: {len(items)}")
    v2_count = sum(1 for _, _, v2, _ in issue_findings if v2)
    lines.append(f"v2-shape (migrated) items: {v2_count}")
    not_v2 = [(n, t) for n, t, v2, _ in issue_findings if not v2]
    lines.append(f"not v2-shape (unmigrated): {len(not_v2)}")
    lines.append("")

    lines.append("## Pattern frequency (across v2 items)")
    lines.append("")
    lines.append("| Pattern | Issues affected | Description |")
    lines.append("|---|---|---|")
    for k in sorted(pattern_counts, key=lambda k: -pattern_counts[k]):
        n_aff = pattern_counts[k]
        if n_aff == 0:
            continue
        ids = issues_by_pattern[k]
        ids_str = ", ".join(f"#{i}" for i in sorted(ids))
        lines.append(f"| `{k}` | {n_aff} ({ids_str}) | {PATTERNS[k][1]} |")
    lines.append("")

    lines.append("## Per-issue findings (v2 only)")
    lines.append("")
    for n, t, v2, findings in sorted(issue_findings):
        if not v2 or not findings:
            continue
        lines.append(f"### #{n} — {t[:80]}")
        for k, samples in findings.items():
            lines.append(
                f"- **{k}** ({len(samples)} sample(s)): {', '.join(repr(s) for s in samples[:3])}"
            )
        lines.append("")

    if not_v2:
        lines.append("## Not v2-shape (unmigrated, audit skipped)")
        lines.append("")
        for n, t in sorted(not_v2):
            lines.append(f"- #{n} — {t[:80]}")

    FINDINGS_PATH.write_text("\n".join(lines))
    print(f"Findings: {FINDINGS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit clean-result body prose for known discipline anti-patterns."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "body_file",
        nargs="?",
        help="Optional local markdown body to audit (e.g. an analyzer draft).",
    )
    src.add_argument(
        "--task",
        type=int,
        help="Task number; resolves to tasks/<status>/<N>/body.md.",
    )
    args = parser.parse_args()

    if args.task is not None:
        try:
            body_path = _resolve_task_body_path(args.task)
        except FileNotFoundError as exc:
            print(f"audit_clean_results_body_discipline: {exc}")
            raise SystemExit(2) from exc
        rc = _audit_single_body(body_path.read_text())
        if rc != 0:
            raise SystemExit(rc)
        return

    if args.body_file:
        rc = _audit_single_body(Path(args.body_file).read_text())
        if rc != 0:
            raise SystemExit(rc)
        return

    _run_legacy_bulk_inventory()


if __name__ == "__main__":
    main()
