---
name: Claude under-classes path-isolation defects on path-isolation rounds
description: When a round's WHOLE PURPOSE is path/namespace isolation, Claude can class a sentinel-publishes-wrong-path bug as Minor because the actual artifact lands correctly. The final sentinel IS the reproducibility ledger; misreporting it on a path-isolation round is Major.
type: feedback
---

When a code-review round's entire mechanism is **path/namespace isolation** (e.g. an HF subfolder suffix like `__r15` that preserves prior adapters as evidence), and one reviewer flags that the dispatcher's FINAL summary JSON publishes the un-suffixed path while another classes it Minor "because the actual upload lands at the correct suffixed path" — the FAIL-class severity is correct.

**Why:** The whole point of the round is isolation, and the FINAL sentinel (e.g. `issue-504-results.json`'s `reproducibility.adapter_paths`) is the canonical end-of-sweep record consumed by analyzer / clean-result / upload-verifier as the reproducibility ledger. Misreporting it on the path-isolation round directly violates the round's contract — even when the actual upload site is correct. The per-cell sentinel being correct doesn't rescue this: downstream automation reads the FINAL sentinel, not the N per-cell sentinels.

**How to apply:** When adjudicating a code-review round whose user-directive language includes "preserve prior adapters / isolation suffix / parallel paths," check whether the dispatcher's final sentinel writer reads a separate `cell_results` dict (vs reading per-cell sentinels). If yes, grep for the suffix variable in that dict's construction — if absent, this is Major and blocks. Twin: Codex catches this; Claude defaults to Minor because per-cell sentinel is correct.

Companion pattern: stage-2 launcher missing `--skip-phase07` when stage 1 ran phase07. Claude misses the GPU re-run cost because the dispatcher's fast-path lookup is against the ORIGINAL inputs (not the stage-1 outputs), so the re-run isn't a no-op when stage 1 needed vLLM fill. Always check the fast-path's input set vs what stage 1 actually wrote. #504 round 15.
