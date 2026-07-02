---
name: Codex APPROVEs a plan by not engaging the Claude critic's artifact-anchored reproduction-target claim
description: When Codex returns APPROVE but visibly skips a specific factual claim Claude pinned to a committed reference JSON/figure (M_shape, persona_order, lineage), read the artifact yourself — the unrebutted claim usually holds and binds REVISE.
type: feedback
---

When the Claude critic FAILs/REVISEs on a precise factual claim about a
COMMITTED reference artifact (a reproduction-target number, its matrix
shape, its axis identity, its producing-adapter lineage) and Codex
APPROVEs WITHOUT engaging that claim ("plan answers its own question",
plus only soft analyzer-should-weigh concerns), the disagreement is not a
real two-sided dispute — Codex never tested the load-bearing claim. Go
read the cited artifact yourself; the anchored claim usually holds.

**Why:** #651 r1 (methodology lens). Plan §7's only hard gate asserted it
would reproduce #521's `same_marker_seed42.json` (`s_top1_frac=0.32465`,
`M_shape=[3584,14]`) by re-extracting the 16 `i537_marker_*` contrastive
context-generalization cells stacked over 16 contexts × 14 personas.
Reading the file settled it in one shot: `M_shape=[3584,14]` where the
`14` is the PERSONA axis (`persona_order` = the 14 `I551_PANEL_14`
personas; `n_contexts` and `context_order` BOTH `None` — no context axis
exists in the matrix), and `inputs_manifest.json` showed the producing
adapter was a SINGLE #519 villain-source marker adapter
(`marker_villain`, `issue_519`), a different lineage AND recipe than the
#537 cells the sweep reads. Claude was exactly right; Codex's APPROVE
explicitly "did not engage the §7 canary reproduction-target factual
claim." Binding verdict: REVISE.

**How to apply:** On a critic-lens PASS-vs-FAIL split where (a) the FAIL
side cites a specific value/shape/lineage in a committed
`eval_results/**/*.json` (or a SHA-pinned figure) and (b) the PASS side
does NOT rebut that specific claim, do NOT treat it as a balanced
disagreement to average. Load the cited JSON and check the literal
fields: `M_shape`, `persona_order`/`context_order`/`n_contexts`,
inputs-manifest lineage. If the field values contradict the plan's
assertion, the finding is Real & blocking — especially when the wrong
target sits on the plan's ONLY hard gate (a false reproduction target
makes the gate unsatisfiable: faithful implementer either false-FAIL
halts and loses the run, or silently relaxes the tolerance and discards
the verification the gate exists to provide). Anchored-and-unrebutted
beats unanchored-APPROVE. Sibling of the existing
`feedback_codex_skips_data_construction_arithmetic.md` — same family:
Codex skips the concrete artifact check that would have settled it.
