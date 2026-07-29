---
name: reused-module-hf-prefix-contract
description: Sibling reused modules can give the SAME flag name OPPOSITE semantics (round root vs capture prefix) — read the CONSUMING function's path join before wiring defaults, then class-sweep every caller of the same consumer (issue #1776 crash-fix cycle 2)
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
