---
title: 'Smoke-architecture gate: per-arm enumeration is hand-listed, so the REAL-or-N/A
  invariant is unfalsifiable'
kind: infra
tags: []
created_at: '2026-08-07T15:27:44Z'
has_clean_result: false
origin_prompt: /issue 2163
workflow: v1
---
## Goal

Make the Step 6d.0 smoke-architecture gate's per-arm resolution enumeration
MECHANICALLY derived from the driver's own phase/arm registry instead of
hand-composed from plan narrative, and add the driver-side static check that
catches the defect class the incomplete enumeration let through.

## The gap

The gate's contract (`.claude/skills/issue/SKILL.md` Step 6d.0, mirrored in
`.claude/agents/code-reviewer.md`) states the invariant as "every planned arm
resolves REAL or N/A" and defines the four verdict tokens against it. What it
does NOT specify is where the ARM SET comes from. In practice the orchestrator
hand-lists the arms from the plan's narrative Design section, so the invariant
is quantified over an unverified set: an arm omitted from the list is
indistinguishable from an arm that passed.

Worked failure (#2163, 2026-08-07). The `epm:smoke-architecture-check v4`
per-arm block listed 10 arms and I declared PASS_UNIFIED. The driver's registry
had 13:

    sorted(PHASES) = upload-inputs, stage, census, fit-maps, read-ladder,
    carried, answer-matchedn, partials, confirm-b, confirm-b-gpu,
    upload-verify, harvest, figures

Three unenumerated. One benign (`upload-inputs` ran REAL as VM Phase U). The
other two — `harvest`, `figures` — were never smoked AND each carried a hard
`AttributeError` (`args.harvest_out` / `args.figures_out`: referenced but never
registered on the argparser). Both were dead on arrival and both fired for real
at Step 8, costing two fix rounds.

The selection effect is the point, not the arithmetic: what a hand-listed set
omits is by construction what nobody was thinking about, hence the arms most
likely to carry untested defects. Here both omissions were also the phases that
run VM-side AFTER pod teardown, so they sit outside the pod-side smoke slice's
natural attention — a systematic blind spot, not a random slip.

## Two layers, one root cause

**(a) Gate side — derive the arm set, assert the count.** The Step 6d.0
contract should require the per-arm enumeration be derived from the driver's own
registry (for a phase-dispatch driver: `sorted(PHASES)`; more generally the
dispatch table / arm registry the entrypoint actually routes on), and require
the marker to state `n_arms_enumerated == n_arms_in_registry` with the registry
source named. A count mismatch is a gate FAIL, not a note. This is cheap — the
orchestrator already reads the driver to compose the marker — and it converts an
unfalsifiable claim into a checkable one.

**(b) Driver side — argparse-attribute completeness, whole-module.** The
existing `--import-check` convention cannot catch this class: an `args.<attr>`
reference is not an import, so a clean import proves nothing about argparser
completeness. The fix that landed on #2163's driver generalizes — assert every
`args.<attr>` referenced anywhere in the module is argparser-defined:

    def _check_phase_args_defined() -> None:
        src = Path(__file__).read_text(encoding="utf-8")
        defined = {m.group(1).replace("-", "_")
                   for m in re.finditer(r'ap\.add_argument\(\s*"--([a-z0-9-]+)"', src)}
        defined |= {"phase", "import_check"}
        missing = sorted({a for a in re.findall(r"args\.([a-z_][a-z0-9_]*)", src)
                          if a not in defined})
        if missing:
            raise SystemExit(f"import-check FAILED: ... ({', '.join(missing)})")

The WHOLE-MODULE scope is load-bearing and was learned the hard way: a first
version scanned only the `PHASES` function bodies and missed `args.figures_out`
because it lives in `_fig_dir`, a helper the phase calls. Any per-function scope
is escapable by moving the reference one call deeper, so the file is the only
non-escapable scope. Consider whether this belongs as a convention in
`.claude/rules/code-style.md` (phase-dispatch driver section) and/or a
`workflow_lint.py` check over `scripts/issue*_*.py` entrypoints — the lint form
would catch it fleet-wide at commit time rather than per-driver.

Scope note for whoever implements: (a) and (b) are independently landable. (a)
is the one that closes the gate's unfalsifiable-invariant hole; (b) is defence
in depth for one specific defect class that (a) would have surfaced anyway by
forcing the omitted arms into the enumeration.

## Acceptance criteria

1. Step 6d.0's contract in `.claude/skills/issue/SKILL.md` specifies the arm
   set's derivation source and requires the enumerated-vs-registry count
   assertion in the marker; a mismatch is a documented FAIL.
2. `.claude/agents/code-reviewer.md`'s mirror of the gate agrees (no drift
   between the two statements of the contract).
3. The driver-side argparse-completeness check exists as a reusable convention
   (code-style rule, and/or a `workflow_lint.py` check), with the whole-module
   scope and its rationale recorded so a future narrowing does not silently
   reintroduce the helper-escape hole.
4. Tests pin whatever mechanical form (1) and (3) take.

## Provenance

Surfaced by #2163 (`kind: experiment`) at Step 8, 2026-08-07. Self-correction
note on #2163 records the full materiality analysis. Fingerprint is distinct
from the three other infra tasks filed from that session: #2171 (authorized-stub
grant token not wired for this same gate — adjacent surface, different bug),
#2172 (parenthesized §9 `planned_wall_h` cell disables the poller phase-ETA),
#2174 (fact-checker: exact-identity premise from a sample verified at full
grain).
