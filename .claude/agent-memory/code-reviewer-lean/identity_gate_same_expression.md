---
name: identity-gate-same-expression
description: A render/parity "identity gate" whose two compared values come from the byte-identical expression is a tautology — diff the two SIDES' provenance; also verify any "handled at <other phase>" waiver comment against that phase's code (#2356 R1 g1)
metadata:
  type: feedback
---

Rule: for every equality/identity gate in a diff, trace WHERE each side's value
is produced. If both sides are computed by the same expression in the same
process (e.g. `gen_ids = tokenizer.encode(render(p)); cap_ids =
tokenizer.encode(render(p))`), the gate can never fail — a hollow gate that
green-lights the very seam it claims to close. The real seam is usually a
DIFFERENT producer (vLLM's internal tokenization vs an HF `encode`; a banked
artifact vs a recompute): the fix is persisting the other producer's value
(`RequestOutput.prompt_token_ids`) and comparing against THAT.

Sibling check, same review: a lint-waiver comment asserting a cap is
"handled at <other phase>" (e.g. `HUB_DIR_FILECOUNT_EXEMPT: ... production
sharding is handled at means`) is a CLAIM about another function — open that
function and verify. In #2356 g1 the named `means` phase EXPLODED consolidated
shards into ~13k per-sha files in one dir, guaranteeing the 10k/dir
`HubDirFileCountError` post-GPU.

**Why:** #2356 R1 g1 (`issue2356_pod.py`): `_render_identity_gate` compared
`encode(_render_chat(p))` to itself — plan A11's gen↔capture render gate was
structurally vacuous; the sharding waiver comment was false. Both are
self-referential-evidence shapes (family: [[vllm_port_terminal_and_selfref_parity]],
hollow-verification-gate #779).

**How to apply:** on any `==`-gate or waiver comment in a diff: (1) name each
side's producer function + process; same producer ⇒ substantive blocker
`hollow-verification-gate`; (2) grep the phase a waiver comment names and
confirm it does what the comment claims.
