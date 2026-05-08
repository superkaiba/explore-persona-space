"""Audit awaiting-promotion bodies for the body-discipline anti-patterns
identified during the 2026-05-08 mass-migration title pass.

Outputs `.claude/cache/audit-2026-05-08/findings.md` with a per-issue
breakdown and a pattern-frequency summary. Bodies are NOT modified.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

OUT_DIR = Path(".claude/cache/audit-2026-05-08")
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
}


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def list_awaiting_promotion() -> list[dict]:
    """Read pre-built inventory.json (from bash paginator) — Python's gh
    GraphQL pagination chokes on cursors with certain characters."""
    return json.loads(INVENTORY_PATH.read_text())


def strip_code(text: str) -> str:
    """Remove fenced code blocks and inline-backtick spans."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


def is_v2(body: str) -> bool:
    return "## AI TL;DR (human reviewed)" in body or (
        "## AI TL;DR" in body and "## AI Summary" in body
    )


def audit_body(body: str) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    cleaned = strip_code(body)
    for name, (pattern, _) in PATTERNS.items():
        flags = re.IGNORECASE if name == "pre_reg" else 0
        matches = list(re.finditer(pattern, cleaned, flags))
        if matches:
            findings[name] = [m.group(0) for m in matches[:5]]
    return findings


def main():
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


if __name__ == "__main__":
    main()
