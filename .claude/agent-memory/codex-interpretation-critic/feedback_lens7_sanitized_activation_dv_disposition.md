---
name: Lens-7 sanitized disposition for activation-space DVs on ToS-risky corpora
description: How to compose lens 7 when the raw corpus (LMSYS/Betley) cannot be paged and all DVs are activation-space — and why the prompt must say the firewall is NOT a BLOCKED capability
type: feedback
---

When a task's DVs are ALL activation-space (cosines, R², projections) and the
raw-completions corpus is ToS-risky text that must never be paged (LMSYS user
prompts, Betley EM rows), compose lens 7 as a SANITIZED disposition with three
scoreable sub-checks instead of the firing-rate sampling recipe:

(a) internal consistency of the body's sanitized spot-check block (row
    indices, n_tokens, finish_reason counts) against the descriptives JSON's
    aggregates (e.g. stop/length counts vs the stated truncation %);
(b) shard schema / aggregate coherence (stated fields + record count vs what
    the aggregates imply);
(c) optional HF-liveness of the raw path via `list_repo_files`, metadata
    only, tagged advisory under the network carve-out.

**Why:** there are no text-level firing-rate claims to verify, so the standard
lens-7 recipe is vacuous; and a prior analyzer on the same corpus (issue
#1073's parent run) was refusal-killed by paging text.

**How to apply (the load-bearing line):** the prompt MUST explicitly state the
text firewall is the SANCTIONED disposition, not a denied capability —
otherwise Codex's denied-capability paragraph tells it to mark lens 7
`BLOCKED` and force an overall REVISE on a lens that is fully scoreable from
(a)+(b). Add: "(The <corpus> text firewall is NOT a denied capability — it is
the sanctioned disposition; do not mark lens 7 BLOCKED merely because text
reads are forbidden.)" Also adapt the output-format lens-7 section to the
three sub-checks + a "firewall respected in the body?" line, replacing the
firing/non-firing sampling checklist. First used: #1073 r1 (2026-07-06).

**Generalizes beyond activation-space DVs (#1074 r3, 2026-07-06):** the same
disposition works when the DVs ARE judge-scored text-behavior RATES over a
firewalled corpus (harmful-compliance install rates, negative-yield quotas),
provided per-question rate arrays + drop-mix aggregates are committed. The
sub-checks become cross-FILE coherence: (a) body by-reference sample blocks
vs the yield JSON's kept/drop-mix fields; (b) summary-rate fields vs the
mean of the per-question rate arrays (counts sum to the bank size); (c)
calibration JSON agreement fields vs the claimed match counts; plus the
"firewall respected in body?" line. Also enumerate the sanctioned read SET
explicitly (top-level aggregate JSONs yes, judge/*/ subdirectories + raw_*.jsonl
+ HF raw links no) — and defuse the network paragraph ("you do not need the
network; do not fetch the body's HF links"), since those links point at the
firewalled raw completions and a fetch attempt is both a refusal risk and a
spurious advisory.
