#!/usr/bin/env python
"""Throwaway measurement for task #2291 (run from repo root):

Applies the PROPOSED check-17 Provenance leg to every
tasks/*/*/original-body.md carrying a `## Provenance` H2, and reports the
realized verdict distribution + per-FAIL task ids. Also counts the
absent-both population (label-present nested bodies with neither
frontmatter origin_prompt nor an extractable Provenance prompt) for the
fail-soft severity decision.
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
import verify_task_body as v  # noqa: E402

PROV_RE = re.compile(r"^##\s+Provenance\s*$", re.MULTILINE)


def provenance_section(text: str) -> str | None:
    m = PROV_RE.search(text)
    if not m:
        return None
    sect = text[m.end() :]
    nxt = re.search(r"^##\s+", sect, re.MULTILINE)
    return sect[: nxt.start()] if nxt else sect


def blockquote_segments(section: str) -> list[str]:
    segs: list[str] = []
    cur: list[str] = []
    for ln in section.splitlines():
        if ln.lstrip().startswith(">"):
            cur.append(v._BLOCKQUOTE_MARKER_RE.sub("", ln))
        elif ln.strip() and cur:
            cur.append(ln)  # markdown lazy continuation (same rule as _context_quote_candidates)
        else:
            if cur:
                segs.append("\n".join(cur))
                cur = []
    if cur:
        segs.append("\n".join(cur))
    return segs


def primary_prov_prompt(section: str) -> str | None:
    """FIRST blockquote segment with normalized length >= 20; fallback:
    longest inline-quoted span >= 20 normalized chars."""
    for seg in blockquote_segments(section):
        n = v._normalize_prompt_text(v._unescape_markdown(seg))
        if len(n) >= 20:
            return n
    best = ""
    for s in v._INLINE_QUOTE_SPAN_RE.findall(section):
        n = v._normalize_prompt_text(v._unescape_markdown(s))
        if len(n) > len(best):
            best = n
    return best if len(best) >= 20 else None


rows = []
n_ob_total = 0
n_prov = 0
for ob in sorted(Path("tasks").glob("*/*/original-body.md")):
    n_ob_total += 1
    text = ob.read_text(errors="replace")
    sect = provenance_section(text)
    if sect is None:
        continue
    n_prov += 1
    bp = ob.parent / "body.md"
    if not bp.exists():
        rows.append((ob.parent.name, ob.parent.parent.name, "no-body", "skip-no-body", ""))
        continue
    raw = bp.read_text(errors="replace")
    fm, body = v.split_frontmatter(raw)
    tid, status = ob.parent.name, ob.parent.parent.name
    if v._is_paper_stub_fm(fm):
        cls = "paper-stub"
    elif v.LEGACY_SAGAN_CARD_SENTINEL in body:
        cls = "sagan"
    elif not v.is_nested_design(body):
        cls = "legacy"
    elif v.is_v4(body):
        cls = "v4"
    else:
        cls = "v3v2"
    detail = ""
    if cls in ("paper-stub", "sagan", "legacy"):
        verdict = f"skip-{cls}"
    else:
        repro = v._repro_section_text(body)
        label = bool(repro and v._CONTEXT_LABEL_RE.search(repro))
        op = str(fm.get("origin_prompt") or "").strip()
        prim = primary_prov_prompt(sect)
        if repro is None:
            verdict = "skip-no-repro-section"
        elif not label:
            verdict = "missing-row-branch(unchanged-FAIL-today)"
        elif prim is None:
            verdict = "no-extractable-prov(fail-soft)"
        else:
            m = v._CONTEXT_LABEL_RE.search(repro)
            region = repro[m.end() :]
            stripped = v._strip_blockquote_markers(region)
            contained = (prim in v._normalize_prompt_text(stripped)) or (
                prim in v._normalize_prompt_text(v._unescape_markdown(stripped))
            )
            if op:
                st, _d = v._origin_prompt_quote_verdict(repro, fm)
                verdict = f"op-present:{st}:prov-{'contained' if contained else 'MISSING'}"
            elif contained:
                verdict = "NEW-pass"
            else:
                trunc = False
                for cand in v._context_quote_candidates(region):
                    nc = v._normalize_prompt_text(v._unescape_markdown(cand)).rstrip(
                        ".,;:!?… "
                    )
                    if 20 <= len(nc) < len(prim) and len(nc) >= 0.5 * len(prim) and prim.startswith(nc):
                        trunc = True
                if trunc:
                    verdict = "NEW-fail-trunc"
                elif cls == "v4":
                    verdict = "NEW-FAIL-mismatch(v4)"
                else:
                    verdict = "NEW-warn-mismatch(v3v2)"
                detail = f"prov_len={len(prim)}"
    rows.append((tid, status, cls, verdict, detail))

print(f"original-body.md files: {n_ob_total}; with ## Provenance H2: {n_prov}")
c = Counter((r[2], r[3]) for r in rows)
for k in sorted(c):
    print(f"{c[k]:4d}  cls={k[0]:10s} verdict={k[1]}")
print("\n--- rows that would newly FAIL or WARN under the proposal ---")
for tid, status, cls, verdict, detail in rows:
    if "NEW-FAIL" in verdict or "NEW-fail" in verdict or "NEW-warn" in verdict:
        print(f"  #{tid} ({status}, {cls}) -> {verdict} {detail}")
print("\n--- op-present rows where the Provenance escape would NOT verify ---")
for tid, status, cls, verdict, detail in rows:
    if verdict.startswith("op-present") and "prov-MISSING" in verdict:
        print(f"  #{tid} ({status}, {cls}) -> {verdict}")

# Second pass: absent-both population (for the fail-soft WARN-vs-noop decision):
# nested label-present bodies, no fm origin_prompt, AND (no original-body.md,
# no ## Provenance H2, or no extractable prompt).
absent_both = []
for bp in sorted(Path("tasks").glob("*/*/body.md")):
    raw = bp.read_text(errors="replace")
    fm, body = v.split_frontmatter(raw)
    if v._is_paper_stub_fm(fm) or v.LEGACY_SAGAN_CARD_SENTINEL in body:
        continue
    if not v.is_nested_design(body):
        continue
    repro = v._repro_section_text(body)
    if repro is None or not v._CONTEXT_LABEL_RE.search(repro):
        continue
    if str(fm.get("origin_prompt") or "").strip():
        continue
    ob = bp.parent / "original-body.md"
    prim = None
    if ob.exists():
        sect = provenance_section(ob.read_text(errors="replace"))
        if sect is not None:
            prim = primary_prov_prompt(sect)
    if prim is None:
        absent_both.append((bp.parent.name, bp.parent.parent.name, v.is_v4(body)))
print(f"\nabsent-both population (label-present, nested, no op, no extractable prov): {len(absent_both)}")
print(f"  of which v4: {sum(1 for r in absent_both if r[2])}")
