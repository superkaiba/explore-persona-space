#!/usr/bin/env python
"""Corpus measurement for task #2291 — check-17 Provenance-leg verdict census.

Re-runs the #2291 retro-scan (plan §6 criterion 7) over the live tasks/
corpus, classifying every nested (v4/v3/v2) label-present body's Context
row under the SHIPPED check-17 Provenance leg. Per plan §7 kill 3, this
script imports and calls the SHIPPED helpers from
``scripts/verify_task_body.py`` — ``_provenance_verbatim_prompt``,
``_provenance_prompt_quote_verdict``, ``_origin_prompt_quote_verdict``,
``_has_min20_blockquote``, ``_PROVENANCE_H2_RE``, and the full
``check_repro_context_provenance`` — never a local reimplementation of the
predicate; every classified row is cross-checked against the shipped
check's own verdict, and any disagreement prints a RECONCILE-ERROR row and
exits nonzero.

Printed row classes (plan §6 script-output completeness correction — the
plan-time artifact copy did NOT print these):

- ``NEW-pass`` ids (op absent, Provenance extractable AND contained);
- ``op-present:fail-trunc`` ids (pre-existing red — expected #742, NOT
  change-induced);
- ``warn-unverifiable-quote`` ids (op absent, no extractable Provenance,
  >=20-char blockquote in a v4 Context row — expected #1072 only);
- every row that would newly FAIL (v4) or newly WARN (v3/v2) under the
  Provenance leg (expected: zero);
- the ``op-pass + Provenance NOT contained`` count (the CANONICAL
  predicate label per plan §4-G item 5; expected 27) — the precedence
  population the op-pass shadowing pins. NOTE: plan v2/v3 prose carries
  a stale ``26`` for this label. 27 is correct and was plan v1's own
  figure; the Phase-1.5 fact-check "corrected" it to 26 on a false
  double-count claim (#628 is ``op-present:warn-mismatch``, a disjoint
  bucket, so nothing was ever double-counted). 26 is the count of the
  ADJACENT ``op-pass + Provenance contained`` bucket. Neither figure
  gates anything — both are blast-radius counts for the REJECTED
  "require both to match" alternative (plan §11 entry 3).

Run from any checkout (repo root or a worktree):

    OMP_NUM_THREADS=8 uv run python scripts/issue2291_measure_check17_provenance.py

The verifier under test resolves from THIS file's sibling ``scripts/``
directory (so a worktree run measures the worktree's shipped code); the
corpus resolves through the canonical task-workflow resolver
(``tasks_dir()``), i.e. the LIVE main-checkout tasks/ tree.
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_task_body as v  # noqa: E402

from explore_persona_space.task_workflow import tasks_dir  # noqa: E402

# Strict (pre-#2291) header form, kept ONLY for the widening-reconciliation
# stat (plan §6: 95 strict vs 96 widened, +#2220's `## Provenance (verbatim)`).
_STRICT_PROV_RE = re.compile(r"^##\s+Provenance\s*$", re.MULTILINE)


def _cls(fm: dict, body: str) -> str:
    """Body class for reporting: paper-stub / sagan / legacy / v4 / v3v2."""
    if v._is_paper_stub_fm(fm):
        return "paper-stub"
    if v.LEGACY_SAGAN_CARD_SENTINEL in body:
        return "sagan"
    if not v.is_nested_design(body):
        return "legacy"
    return "v4" if v.is_v4(body) else "v3v2"


def main() -> int:
    tasks_root = tasks_dir()
    n_ob_total = 0
    n_prov_strict = 0
    n_prov_widened = 0
    for ob in sorted(tasks_root.glob("*/*/original-body.md")):
        n_ob_total += 1
        text = ob.read_text(errors="replace")
        if _STRICT_PROV_RE.search(text):
            n_prov_strict += 1
        if v._PROVENANCE_H2_RE.search(text):
            n_prov_widened += 1
    print(
        f"original-body.md files: {n_ob_total}; with ## Provenance H2: "
        f"{n_prov_strict} strict / {n_prov_widened} widened (_PROVENANCE_H2_RE)"
    )

    rows: list[tuple[str, str, str, str]] = []  # (tid, status, cls, verdict)
    reconcile_errors: list[str] = []
    for bp in sorted(tasks_root.glob("*/*/body.md")):
        tid, status = bp.parent.name, bp.parent.parent.name
        raw = bp.read_text(errors="replace")
        fm, body = v.split_frontmatter(raw)
        cls = _cls(fm, body)
        if cls in ("paper-stub", "sagan", "legacy"):
            rows.append((tid, status, cls, f"skip-{cls}"))
            continue
        repro = v._repro_section_text(body)
        if repro is None:
            rows.append((tid, status, cls, "skip-no-repro-section"))
            continue
        m = v._CONTEXT_LABEL_RE.search(repro)
        ob = bp.parent / "original-body.md"
        res = v.check_repro_context_provenance(body, fm, original_body_path=ob)
        if m is None:
            rows.append((tid, status, cls, "missing-row-branch(unchanged-today)"))
            continue
        op = str(fm.get("origin_prompt") or "").strip()
        prov = v._provenance_verbatim_prompt(ob)
        region = repro[m.end() :]
        if op:
            st = v._origin_prompt_quote_verdict(repro, fm)[0]
            if st == "pass":
                if prov is None:
                    verdict = "op-pass (no extractable Provenance)"
                elif v._provenance_prompt_quote_verdict(repro, prov, fm)[0] == "pass":
                    verdict = "op-pass + Provenance contained"
                else:
                    # CANONICAL predicate label (plan §4-G item 5).
                    verdict = "op-pass + Provenance NOT contained"
                if not res.passed:
                    reconcile_errors.append(f"#{tid}: {verdict} but shipped check FAILed")
            else:
                verdict = f"op-present:{st}"
                if st == "fail-trunc" and res.passed and cls == "v4":
                    reconcile_errors.append(f"#{tid}: op fail-trunc but shipped check passed")
                elif st == "warn-mismatch" and (
                    not (res.passed and res.is_warn)
                    or "context-origin-prompt-mismatch" not in res.detail
                ):
                    # Both tiers fold warn-mismatch into warn_bits (v4:
                    # _context_row_result_v4; v3/v2: the legacy branch), so the
                    # shipped verdict must be a WARN carrying the #1068 token.
                    reconcile_errors.append(f"#{tid}: {verdict} but shipped check disagreed")
        elif prov is not None:
            pv = v._provenance_prompt_quote_verdict(repro, prov, fm)[0]
            if pv == "pass":
                verdict = "NEW-pass"
                if not res.passed:
                    reconcile_errors.append(f"#{tid}: NEW-pass but shipped check FAILed")
            elif cls == "v4":
                verdict = f"NEW-FAIL-{pv.removeprefix('fail-')}(v4)"
                if res.passed or "context-provenance-prompt-mismatch" not in res.detail:
                    reconcile_errors.append(f"#{tid}: {verdict} but shipped check disagreed")
            else:
                verdict = f"NEW-warn-{pv.removeprefix('fail-')}(v3v2)"
                if not (res.passed and res.is_warn) or (
                    "context-provenance-prompt-mismatch" not in res.detail
                ):
                    reconcile_errors.append(f"#{tid}: {verdict} but shipped check disagreed")
        else:
            armed = ob.exists() and v._has_min20_blockquote(region)
            if armed and cls == "v4":
                verdict = "warn-unverifiable-quote(v4)"
                if "warn-unverifiable-quote" not in res.detail:
                    reconcile_errors.append(f"#{tid}: {verdict} but shipped detail lacks token")
            else:
                verdict = "no-extractable-prov(fail-soft, unchanged)"
                if "warn-unverifiable-quote" in res.detail:
                    reconcile_errors.append(f"#{tid}: expected-silent but shipped check WARNed")
        rows.append((tid, status, cls, verdict))

    print("\n--- verdict distribution (cls, verdict) ---")
    counts = Counter((r[2], r[3]) for r in rows)
    for key in sorted(counts):
        print(f"{counts[key]:4d}  cls={key[0]:10s} verdict={key[1]}")

    def _ids(pred) -> list[str]:
        return [f"#{tid} ({status}, {cls})" for tid, status, cls, verd in rows if pred(verd)]

    print("\nNEW-pass ids:", ", ".join(_ids(lambda x: x == "NEW-pass")) or "(none)")
    print(
        "op-present fail-trunc ids (pre-existing red, NOT change-induced):",
        ", ".join(_ids(lambda x: x == "op-present:fail-trunc")) or "(none)",
    )
    print(
        "warn-unverifiable-quote ids:",
        ", ".join(_ids(lambda x: x.startswith("warn-unverifiable-quote"))) or "(none)",
    )
    print(
        "op-pass + Provenance NOT contained:",
        f"{len(_ids(lambda x: x == 'op-pass + Provenance NOT contained'))} rows",
    )
    newly_red = _ids(lambda x: x.startswith(("NEW-FAIL-", "NEW-warn-")))
    print("rows newly FAILing (v4) or newly WARNing (v3v2):", ", ".join(newly_red) or "(none)")

    if reconcile_errors:
        print("\nRECONCILE-ERROR: script classification vs shipped check disagreed:")
        for line in reconcile_errors:
            print(f"  {line}")
        return 1
    print("\nreconciliation: script classification agrees with the shipped check on every row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
