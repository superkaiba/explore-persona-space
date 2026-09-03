---
name: pilot-pass-report-fingerprint-unchecked
description: A rule-26 pilot gate whose PASS report persists the instrument fingerprint but whose require/resume consumers check only family+verdict blesses a stale instrument; plus the route-parity-by-arithmetic and substituted-seam probes (#2479 R2 g6)
metadata:
  type: feedback
---

Reviewing a judge pilot gate (rule 26) or any persisted-PASS gate consumed on
resume: diff what the report PERSISTS against what the require-path COMPARES.
The #2479 r2 gate wrote rubric_sha256 / model / max_tokens / temperature into
every report, yet `require_pilot_pass` checked only family + verdict — so both
wrappers' "PASS report present — pilot skipped (resume)" branches, the P1
preamble, and the run_leg env hook all bless a PASS persisted under a rubric or
max_tokens later edited by a crash-fix round. Recorded-but-never-compared is
the tell; the fix is a consumer-side fingerprint compare vs the LIVE constants.
Sibling of [[presence-redrive-blesses-stale-mirror]] and
[[new-dial-missing-from-resume-regime]].

**Why:** the resume-skip is exactly where instrument drift bites — pilot at
commit X, instrument edited at commit Y, wave dispatches un-piloted; crash-fix
rounds that raise max_tokens or tweak a rubric are the fleet's normal shape.

**How to apply:** (1) grep every consumer of the report (require fn, wrapper
skip branches, env hooks, sibling-leg reuse) — if none compares the persisted
fingerprint to the live instrument, Major. (2) When a family is piloted through
a SUBSTITUTED seam (the registered gate fn can't parse its instrument), certify
the substitute by byte-comparing it to the production dispatch site: payload
dict KEY ORDER, model constant, force_path, transport re-drive shape. (3) Route
parity settles by arithmetic, not narration: per-call n at the production call
site vs the dispatcher's crossover constant (SYNC_BATCH_CROSSOVER_N=2000; gen
per-call n ≤ n_target=1600 → sync both sides); a plan §9 "Batch" label can
mislabel a sync-realized wave. (4) "route recorded" claims in the commit
message get grepped against the payload keys — #2479's partial recorded none.

**r4 closure shape (#2479 R4 g1) — grade later rounds against these:**
(a) DATA identity beside the instrument: hash committed input FILE bytes +
parsed item CONTENT via an order-independent sorted-triples sha (never file
bytes of re-emitted intermediates — byte-order drift would false-refuse);
(b) validate-after-refresh sequencing holes close by RECORD-LICENSE — the
spend site persists the licensing gate's fingerprint IN the artifact it
licensed, and resume compares it to the CURRENT gate (refresh-at-same-
fingerprint = equivalence; different = quarantine); (c) an exact-N per-item
draw census over save_raw `all_scores` keys is SAFE — batch_judge mints
error dicts for every dispatched row (#1313), so transport losses still
count keys and only a genuinely partial raw file fails; (d) live-probe
fixtures for `require_pilot_pass`-style gates need the FULL verdict
predicate (`verdict` AND `passed`) — build them from the test helpers'
`_report` shape, not from the docstring.

**Key+compare joint coverage (#2658 rev-E r2):** when a resume fix pairs a
field-by-field compare WITH a constants-folding gate-dir key, the KEY is the
load-bearing arm (a changed constant must resolve a fresh, empty dir; the
compare is tamper defense) — so enumerate EVERY verdict-bearing parameter of
the gate's signature (`judge_pilot_gate`: eval_prompt/model/max_tokens/
n_draws/temperature via the fingerprint; parse_fail_threshold;
min_effective_draws_per_arm; wave transport; **api_refusal_threshold +
waive_* lists**) and require each to appear in the key OR force a mismatch.
A verdict-bearing param outside BOTH (2658: `api_refusal_threshold`, rule
26(d), judge_pilot.py:433 — persisted by `to_json`, never compared, never
keyed, never passed at the seam) has TWO arms: a stale PASS silently honored
across a threshold change, and a persisted-FAIL refuse branch that hard-
wedges (exit 4) the api-refusal-side remediation because neither threshold
nor waiver changes the dir key — the refuse message's "a changed instrument
resolves a fresh gate dir" is true only for key-tracked fields. On harm-class
judge waves (30%+ api-refusal, rule 28) that dial is the one MOST likely
tuned by a crash-fix round. Also verified there: a fake gate that WRITES the
report the resume reads must be production-parity-checked against the real
`PilotGateReport.to_json` mint site (rubric_hash = sha256(eval_prompt)[:16]
at judge_pilot.py:1042; wave_transport = decision.path :1043) — tests can
pass while production still re-runs/wedges if field names or formats drift.

**Recipe-faithful fix leaves the class open (#2658 rev-E r3):** a fix round
implements exactly the prescribed recipe (key+compare+pass the NAMED param)
— so the re-review's job is the SIBLING sweep: walk the gate's FULL
signature and classify every param as keyed / compared / passed / default.
r3 found `waive_parse_fail_arms` (rule 26(b)'s sanctioned waiver, the #2091
`PILOT_WAIVE_PARSE_FAIL_ARMS` pattern, prescribed by the library's own FAIL
text) in NEITHER key nor compare nor call — the identical two-arm defect one
parameter over. Tell: the fix's rewritten refuse message universalized
("Every gate parameter is key-tracked") — grade that quantifier against the
sweep table, not the fixed instance. Also: "waiver tuple not persisted" is
true of the TUPLE, but per-arm realized `waived`/`api_refusal_waived` bools
ARE persisted (ArmPilotStats) — key-only still suffices because report
tamper can equally fake `passed` (verdict is trusted; compare exists for
constants drift, not forgery). Key-payload additions orphan ALL prior gate
dirs → one fail-safe re-pilot per row; state it, don't flag it.

**Enumeration-guard closure probes (#2658 rev-E r3 fix review, PASSed):** when
a round ships a live-derived parameter-enumeration guard
(inspect.signature ∩ key-payload names / mutation-verified fingerprint /
regex-derived compare fields / justified allow-list, unclassified==∅), certify
it with FIVE independent probes, never the round's own demo: (1) inject a
synthetic kwarg via `f.__signature__ = sig.replace(parameters=[...])` → guard
must go red NAMING it; (2) inject a synthetic report field via a
`__dataclass_fields__` dict copy (copy.copy an existing Field, rename) → red
from the report direction; (3) fingerprint mutation matrix + a NEGATIVE
control (perturb a non-gate constant; fp must NOT move) + determinism
recompute; (4) direct-key VALUE-linkage matrix — monkeypatch each module
constant, require pilot_gate_key to move (the guard matches NAMES only; a
future literal-inlining edit passes it — file as minor); (5) waiver-dedup
equivalence (('a','a','b') vs ('b','a') key identically). The allow-list is
the irreducible residue: one in-test dict entry with free-text justification
exempts anything — same trust level as deleting the test, so silent
reintroduction (the class) is closed while deliberate misclassification stays
reviewable; ask for a criterion comment at the dict, not more code. Adjudicate
`arms`/data-pool exemptions against the #2479 R4 data-identity precedent
(stale-pool resume is the real residual). Vintage-worktree mapped-test sweeps:
run the selector yourself — it may map cleanly while mapped tests carry
UNRELATED red (attribute per-file before charging the round); cross-file risk
settles by grepping which SYMBOLS importers consume (here: JUDGE_SCHEMA only).
Orphaning cost: check realized dirs exist before pricing (here zero — no wave
had run).
