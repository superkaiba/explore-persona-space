# Merged Phase 2 critique — task #2243, round 1

Cross-lens merge: **APPROVE** (worst-of-three). Consistency-checker: **PASS**.
No lens returned REVISE or REJECT; no BLOCK findings. Therefore this is an
ACCEPTED-IMPROVEMENTS fold, not a mandated revision round: **do not change the
design, do not add scope beyond the numbered items below.**

Codex twins: instant confirmed no-shows for all three lenses (quota sentinel
`CODEX_QUOTA_LIVE until=2026-09-05T13:38:00Z`), so each lens resolved
single-Claude per the Phase-2 no-show fallback. No twin verdicts were
fabricated.

| Lens | Verdict | Must-Fix |
|---|---|---|
| Methodology | APPROVE | none |
| Statistics & Measurement | APPROVE | none |
| Alternative Explanations | APPROVE | none |
| Consistency (vs parent #2242) | PASS | none |

Independent confirmations worth keeping (do not re-derive): no bypass path into
training exists outside `train_steered_cell` (both `SFTTrainer` L712 and
`SteeredSFTTrainer` L734 are inside it; fan-out children re-enter `main()` via
`--single-cell`); `should_skip` cannot mask a gate-raised cell (skip requires a
`completed` record, and the START-manifest write drops it); recipe-match against
D11 is byte-level on position, arithmetic source, discriminator, flag
names/types and threading shape, with exactly the three declared divergences;
floor 192 re-derived independently twice; no file collision with `issue-2242`
(its 13 files do not touch `scripts/issue2225_train.py`; the shared test file is
a dependency, not a collision).

**The steelman failed, and this strengthens §1.** The Alternatives lens tried to
argue the runtime arm is decorative because D1 already gates datagen. It is not:
`issue2225_train.py` loads whatever JSONL sits at
`_dataset_path(dataset_root, family)`, defaulting to
`external/persona_vectors/dataset` — an external corpus that never passes
through the D1-gated `artifacts/datagen.py` pipeline. For this entrypoint D1
provides **zero** protection and the runtime arm is the only arm. Fold this into
§1 and/or §2 as a positive justification (one or two sentences, with the
`_dataset_path` default named) — it is the strongest available answer to "why
does this port matter".

---

## ACCEPTED — fold each of these into plan v2

### A1 (Statistics 1) — criterion 4's source-substring does not discriminate what it claims

`"--trainability-floor-override" in src` is satisfied by Edit 4a's own
cmd-builder literal (`cmd += ["--trainability-floor-override", ...]`) even if
Edit 3's `add_argument` calls are omitted entirely. In that same scenario test 6
also passes for the wrong reason, because argparse's "unrecognized arguments"
path also exits 2. The joint omission is caught outside pytest — the §6
battery's `--import-check` runs `assert_args_attributes_defined`
(`src/explore_persona_space/orchestrate/argcheck.py:375`), which statically
requires a matching `add_argument` for every `args.trainability_floor_override`
reference Edits 5a/5b introduce — so this is a pytest blind spot with a battery
backstop, not a hole. Close it anyway with one behavioral parse assert:

```python
args = mod.build_argparser().parse_args(
    ["--trainability-floor-override", "4", "--trainability-override-reason", "r"]
)
assert args.trainability_floor_override == 4
assert args.trainability_override_reason == "r"
```

No positional args are required; the existing suite already uses this shape at
`tests/test_issue2225_cell_registry.py::test_argparser_coef_scale_and_pilot_coefs_mutually_exclusive`.
**KEEP** the `"smoke=max_steps is not None" in src` substring — the Statistics
lens explicitly endorsed it as a legitimate invariant pin on Edit 2's call site
(an Edit-2 omission does turn it red).

### A2 (Statistics 2) — test 6 must assert WHICH error fired

`SystemExit.code == 2` conflates the intended `ap.error` with argparse's
unrecognized-arguments exit. Add, after the `pytest.raises` block:

```python
assert "trainability-override-reason" in capsys.readouterr().err
```

(`argparse.error()` writes to stderr.) Makes the test red only when the paired
validation actually fired.

### A3 (Methodology 3 + Statistics 2) — narrow test 6's argv so a regression fails fast

As specified, `main(["--dry-run", "--trainability-floor-override", "50"])` would,
if Edit 5a were later removed while Edit 3 remained, execute a real 81-cell
dry-run fan-out inside the test process before failing — correct discriminating
power, but slow and noisy. Choose the NARROWEST argv that both (i) reaches
`ap.error` pre-dispatch on the intended path and (ii) on that regression does
NOT launch training or walk all 81 cells. Resolve against the real CLI
(`build_argparser` on `origin/main`) and state in the plan which flag you used
and why it satisfies both properties. If no such flag exists, keep `--dry-run`
and say so explicitly — do not silently leave the noisy shape unexplained.

### A4 (Statistics 3) — pin the three unpinned forwarding seams

Coverage enumeration found three seams that are silent in the suite (kwargs
default to `None`) though they fail loud and safe at runtime — only the
deliberate-override affordance breaks, not the protection:

1. `run_fan_out`'s **live**-branch `_single_cell_cmd` call site (the existing
   test docstring's "live call site shares the identical kwarg set" is a design
   claim, not an assertion).
2. `main()`'s single-cell branch forwarding into `train_steered_cell` — also the
   path every fan-out **child** consumes the flags through, so the
   highest-value seam of the three.
3. `main()`'s fan-out branch forwarding into `run_fan_out` (the existing test 7
   calls `run_fan_out` directly, bypassing `main`).

Pin all three with kwargs-recording stubs (they share one piece of machinery, so
this is cheaper than three independent tests): monkeypatch
`mod.train_steered_cell` (returning a `Path`) resp. `mod._single_cell_cmd`
(returning `[sys.executable, "-c", "pass"]`), drive `mod.main([...])` /
`run_fan_out(..., dry_run=False)`, and assert the recorded kwargs. No CUDA, no
dataset, no network. Keep them in the same section (j).

### A5 (Alternatives 1) — override-record durability + launch-wide scope

Two things to fix, both disclosure-or-hardening:

- **Durability.** #778 returns the gate record into its per-cell result JSON;
  this plan only `logger.info`s it into `ckpt_root/fanout_logs/<slug>.log`,
  which is pod-local. If fan-out logs are not uploaded, an override's recorded
  reason dies with the pod — a weakening of "an override is a recorded
  decision, never silent". Check whether a completion-time manifest update
  exists in `scripts/issue2225_train.py`; if it does, include the trainability
  record in it (cheap, durable, and keeps constraint H's `-> Path` signature
  intact — this is a write, not a return). If no such update exists, state the
  log-only durability limitation explicitly in the risk table and the
  blind-spot enumeration rather than leaving it implicit.
- **Scope.** `--trainability-floor-override 4` on an 81-cell fan-out re-floors
  **all 81 cells**, not one. D11-mirrored, but currently unstated. Say so in
  the flag's `help=` text and in the plan prose, so nobody reads an overridden
  run as a single-cell exception.

### A6 (Methodology 2) — prose precision: "before the weights load", not "before any model download"

§3(a) (and any similar phrasing elsewhere, incl. §6's what-a-PASS-proves) is
imprecise: `AutoTokenizer.from_pretrained` (~L646) runs BEFORE the gate; only
the weights load (L668) and GPU allocation are after it. The substantive claim
holds — fix the wording everywhere it appears.

### A7 (Methodology 1) — tighten the discriminator disclosure to be magnitude-explicit

The smoke discriminator keys on the PRESENCE of `--max-steps`, not its
magnitude, so `--max-steps 999999` demotes the gate to `warn` while training
effectively-full data. This is faithful to D11 (fidelity wins for a port) and
the WARNING still logs, but the blind-spot enumeration should say "any explicit
`--max-steps`, regardless of magnitude" and state that the recorded-override
pair — not a large `--max-steps` — is the only sanctioned deliberate
below-floor path.

### A8 (Statistics 6) — fix the test-count prose

§4 Edit 6 says "six mirroring section (i)" while §2 counts section (i) at four
tests. Recount and state it plainly once the A1/A4 tests are added (the
concrete per-test listing is what binds; just make the prose agree with it).

### A9 (Alternatives 3 + 4) — record two residuals that are outside the protection claim

One line each, in the risk table or a scope note; do not expand scope to fix
them:

- A pre-port below-floor cell with a completed+uploaded manifest and unchanged
  fingerprint reads `skipped-resume` forever — the gate never retroactively
  flags historical artifacts. Relevant only if someone later cites this gate's
  existence as evidence that old #2225 artifacts were floor-checked.
- The floor certifies `len(ds)` (pre-tokenization), not post-truncation usable
  rows; rows dropped at the TRL `max_length` seam could put usable rows under
  floor while `len(ds)` passes. Inherited from D11 deliberately; §11 entry 5
  already frames the floor as necessary-not-sufficient — make sure the
  blind-spot enumeration names it too.

---

## DECLINED — do NOT add these (recorded so the decision is durable)

### D1 (Alternatives 2) — a mocked-boundary end-to-end wiring test

A test driving `train_steered_cell` itself to prove "gate fires before the
weights load" cannot be hermetic: `cell_fingerprint` (needs direction-tensor
fixtures) and `AutoTokenizer.from_pretrained` (network / HF cache) both run
BEFORE the gate line, which disqualifies it from the offline Step-9c-registered
test file. The Alternatives lens itself assessed the marginal value as one
failure mode (gate inserted in a dead position while the string pin still
matches), which a code-reviewer catches on a diff this small. The gap stays
DISCLOSED in §6's "what a PASS does NOT prove" — keep that disclosure, and make
sure it now also names the seams A4 does cover, so the residual is accurate
after the fold.

### D2 — computing the expected floor from `ft.*` inside the tests

Explicitly REJECTED by the Statistics lens as tautological: the test would read
the same constants the gate reads and could never go red on recipe drift. Keep
the hard-coded `floor_rows == 192` / `effective_batch_size == 16`. The parent's
own #778 tests already pin 192 on the same shape, so a legitimate #778 recipe
change was always going to break the suite; section (j) adds symmetric
breakage that points the editor at the second arm.

---

## Constraints on the v2 edit

- Design unchanged. No new scope beyond A1-A9. Do not revisit the three
  declared divergences — all four reviewers examined and accepted them.
- Keep exactly ONE declaration-shaped `Estimated GPU-hours (total): 0` line
  (verify_plan c59 reads the first match).
- Keep the `Smoke blind-spot enumeration:` block and the
  `## WARN dispositions` block; update both for the folded content.
- Re-run `verify_plan.py --plan-file <draft> --kind infra --json` and land on
  `overall == PASS` before returning. Report the verdict line verbatim.
- The base-branch precondition (§4 Step 0: wait for `issue-2242` on
  `origin/main`, three symbol probes, no vendoring) is unchanged and still
  binding — all four reviewers flagged the parent's outstanding Step 9c gate as
  the one live residual, so the re-probe must stay.
