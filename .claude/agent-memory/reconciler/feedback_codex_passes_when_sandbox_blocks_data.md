---
name: codex-passes-when-sandbox-blocks-data
description: Codex interpretation-critic drifts to PASS verdict when its sandbox cannot fetch eval-JSON / HF data; admits the gap in Lens 7 but doesn't downgrade verdict. Always re-verify Lens 1 claims directly against the JSONs before crediting Codex's PASS.
metadata:
  type: feedback
---

When Codex interpretation-critic posts PASS on Lens 1 (overclaims) but Lens 7 (raw-text plausibility) carries language like "could not independently fetch full HF JSON bodies", "intermittent DNS/body-fetch failure", "this lens is partial rather than a full random-sample audit", treat the Lens 1 PASS as **unverified**, not as a real PASS.

**Why:** Codex's Lens 1 verdict on title-vs-data fit requires the per-framing × per-persona × per-seed breakdown. When sandbox network fails, Codex falls back to the body's aggregate prose for evidence — which is exactly the prose Lens 1 is supposed to be checking. The verdict then becomes "the title agrees with the body's own summary statistics", which is tautological. Claude, running with full filesystem access to `eval_results/issue_<N>/` JSONs, is the only critic positioned to verify title-vs-data fit in this configuration.

**How to apply:** When reconciling Codex PASS vs Claude REVISE on interpretation-critique:
1. Read Codex's Lens 7 verbatim. If it carries any "could not fetch / DNS / sandbox" disclaimer, weight Codex's Lens 1 verdict to zero — that disclaimer invalidates the upstream lenses too.
2. Pull `eval_results/issue_<N>/full_eval_summary.json` (or whatever per-cell JSON the analyzer cites) and run Claude's exact arithmetic against the raw numbers.
3. The verdict rests on the JSON verification, not on the heuristic. The heuristic is the prior, not the posterior.

Cousin to [[claude-underclasses-silent-failures]] (the inverse case — Claude under-flags, Codex over-flags). The two failure modes are anti-symmetric: Claude tends to litigate genuine issues against the raw data; Codex tends to accept the body's framing when it can't see past it.

**Incident:** task #381 round 1, 2026-05-26. Codex PASS, Claude REVISE. Title "two cheap interventions failed to localise" — Claude's per-framing × per-persona × per-seed pull from `full_eval_summary.json` showed Arm B framing-1 had teach=1.0 / non-teach=0.0 across all 3 seeds, literally satisfying the plan's pre-registered H2 confirm criterion (teach ≥ 80% AND non-teach ≤ baseline+10pp). Body declared H2 falsified without reconciling. Codex couldn't fetch the JSONs, fell back to aggregate prose, missed it. Reconciler verdict: REVISE.

**Code-reviewer mirror (FAIL direction):** the same sandbox limitation flips Codex code-reviewer the OTHER way — it FAILs with sole blocker tag `data-access-blocked` when the round's load-bearing fact lives on HF (artifact-identity / adapter-config verification) and its sandbox can't resolve `huggingface.co`, even while explicitly stating the local diff + tests are consistent with intent and corroborating every locally-checkable leg. That FAIL is an environmental property of the reviewing sandbox, not a defect of the artifact. Reconcile move: re-run the exact HF pull yourself (`hf_hub_download` of the cited `adapter_config.json`s, incl. any pinned-revision check that rules out re-upload); if the fact holds and Codex raised zero substantive defects, PASS. Incident: task #545 round 23, 2026-06-12 — Claude PASS (verified all 7 `issue503_bucket_d_*` configs r=32/α=256/0.0/7-proj at the #503-pinned revision), Codex FAIL on DNS alone; reconciler spot-checked 2 configs + pinned rev + broad_syco α=64 contrast and PASSed.

**Figure-pixel variant (clean-result-critic):** task #601 round 1, 2026-06-12. Codex clean-result-critic posted "all 15 lenses clean" while two PNGs carried (a) an issue-number leak in a legend ("the #471 bridge") and (b) a caption describing an unplotted curve + a 10.8-16.1 band whose plotted max was ~12.9 (#507 incident class). Codex evidently never rendered the PNGs — mechanical pre-passes (verify_task_body.py, audit script) don't inspect figure content, so their PASS proves nothing about Lens 3. When reconciling figure-content lenses, ALWAYS materialize the SHA-pinned PNGs (`git show <sha>:figures/issue_<N>/<name>.png`) and Read them; same round also re-enforced the SPEC per-result 1-3-sentence read beat (precedent chain: #547 r1 → #601 r1 — the spec's Exemplar-scope caveat is binding, all-blockers-verified REVISE).
