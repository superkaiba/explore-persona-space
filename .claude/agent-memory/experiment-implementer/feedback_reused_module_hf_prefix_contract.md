---
name: reused-module-hf-prefix-contract
description: Reused-module wiring contract — flags can carry OPPOSITE semantics across sibling modules AND a wrapper's re-declared args/namespaces must be audited FIELD-BY-FIELD against the consumer's reads (never default=None for a dereferenced field; call args like a layers list can silently wipe resume state) (issue #1776 crash-fix cycles 2+3)
type: feedback
---

Sibling reused modules can give the SAME flag name OPPOSITE semantics: #779's
capture driver (N1G) treats `--hf-prefix` as the ROUND ROOT (it appends
`final_token_capture` itself), while the fits module (N1M,
`issue779_ffc_n1m_fits.assemble_multilayer`) consumes it VERBATIM as the capture
prefix (flat-joins `<hf_prefix>/<name>.pt`). Wiring a new caller's default from
the flag NAME (or from the sibling module's usage) produces a runtime 404 that
only surfaces mid-GPU-run.

**Rules:** (i) when wiring a reused module's CLI/ns defaults, read the CONSUMING
function's path-join code — never infer the contract from the flag name or a
sibling module; (ii) prefer referencing the consuming module's own constant
(e.g. `f"{N1M.N1G.HF_PREFIX}/final_token_capture"`) over a re-typed literal;
(iii) after fixing one call site, CLASS-SWEEP every caller of the same consumer
in the same round — in #1776 three callers (`issue1776_comparator_fit.py`,
`issue1776_phase4.py cmd_refit_split`, `issue1776_phase5.py
assemble_test_leg_and_anchors`) carried the identical wrong default; fixing only
the crashed one would have burned two more launch cycles (one 404 per cycle);
(iv) a cheap fix-engaged probe exists without GPU: drive the real entrypoint's
argv into the assemble namespace and check the remote index at the resolved
prefix (`N50._remote_index(prefix)` non-empty).

(Incident #1776 crash-fix cycle 2, att-20260729-082617, 2026-07-29: comparator
bg job 404'd on `.../fitter-fair-comparison-n1m/shard00_chunk0000.pt`; fix
commit `fe7ef9977b12e6db5f525cf5b9cecbf1516f4533` corrected all three callers;
17-path Hub audit all-PASS post-fix.)

**WIDENED (cycle 3, same class, ARG side):** the trap is not only Hub paths —
a wrapper that re-declares a reused module's CLI surface (or hand-builds its
`argparse.Namespace`) must audit EVERY field against the consumer's ACTUAL
reads and copy the module's OWN defaults: (v) never `default=None` for a field
the consumer dereferences (`orig_dir` → `None / "file"` TypeError; `pass_b` →
`.exists()` AttributeError one line earlier); (vi) audit CALL ARGUMENTS too — a
`layers=[19]` request mismatching the resume cursor's `[14,19]` silently WIPES
+ re-streams hours of memmap work (N1M cursor check); (vii) one-field fixes
miss siblings — sweep the WHOLE namespace × every caller ONCE (cycle 3 audited
36 field-caller cells, fixed 6). (Cycle 3: pod-1776 comparator-join TypeError;
fix commit `e2dd82440ee6a902dbe9df3abe43830f389a249b`.)

**WIDENED AGAIN (cycle 6): INTRA-ISSUE cross-SCRIPT calls carry the same
contract, and a smoke-FENCED branch hides the gap.** A hand-built
`argparse.Namespace` shim feeding another SCRIPT's function (phase4 →
phase3.load_directions) must supply every `args.<attr>` the callee reads on
EVERY reachable branch — grep the callee body for `args\.` at call-site
authoring time (one level into helpers it forwards args to). And the smoke
must EXERCISE the branch: cycle 6's crash site was fenced by the smoke's
`rb_dir=None`, so its first-ever execution was production on 8×H100. Rule
(viii): when a smoke fences a branch (None-input, tiny-N cut), record the
fence and add a production-shaped leg for it before the branch's first
production run. (Cycle 6: fix commit
`3e576922aea19fc270cab9ea4e4096ffffc9b10f`; the 8-call-site audit found the
crash site was the only remaining gap.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Reused-module + cross-script wiring contract](feedback_reused_module_hf_prefix_contract.md) — audit flags AND every arg/ns field vs the consumer's reads (paths, None-defaults, call args); grep callee args. at authoring; smoke-fenced branches need a production-shaped leg (#1776 c2+c3+c6)
