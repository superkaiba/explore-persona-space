---
name: Wrap-script: route on artifact, not exit code
description: When wrapping a script that exits non-zero on a domain signal AFTER writing its artifact, route on the artifact first; rc is a fall-through crash signal only.
type: feedback
---

When the inner script raises `SystemExit` (or otherwise exits non-zero) AFTER it has already written its result artifact — typical of a "domain HALT" pattern like K2 fail or convergence assertion — the wrapper must check the ARTIFACT first and treat the exit code as a fall-through crash signal only. Routing on rc first fail-louds on the legitimate domain outcome and never fires the documented fallback.

**Why:** task #657 v6 round-2's `marker_fallback_decision` checked `rc != 0` FIRST and returned fail-loud before reading the K2 effect file. But `issue623_extract_sycophancy_vector.py:898` raises `SystemExit("K2 HALT...")` AFTER writing `steering_effect_by_layer_marker.json` with `k2_pass: false`. The legitimate K2 fail (the plan §11 fallback signal) produced `(rc=1, valid effect file with k2_pass=false)` — the wrapper mis-classified it as a crash and the 50-minute GCP run died with the documented fallback never firing. Round-3 inverted the priority (read effect file first, route on `k2_pass` boolean; rc is only consulted when the file is missing/malformed).

**How to apply:** when designing a wrapper around a script that emits a structured artifact:
- Read + parse the artifact FIRST. If it carries a valid domain verdict, that is authoritative — route on it regardless of rc.
- The exit code is a SECONDARY signal: it only matters when the artifact is missing/malformed/has no usable verdict. THEN distinguish "crash before artifact write" (rc != 0, no artifact) from "artifact write succeeded but caller exited" (rc == 0 with no usable file is a contract violation).
- When the inner script's behavior is fixed (you can't modify its rc convention — here, `issue623_extract_sycophancy_vector.py` is a shared #623 script), the wrapper must adapt. Don't try to "fix" the inner script's exit convention unless you own both sides.
- Smoke tests must cover ALL FOUR cells of (rc ∈ {0, !0}) × (artifact ∈ {valid k2_pass=true, valid k2_pass=false, invalid/missing}). The round-2 smoke missed the `(rc!=0, valid k2_pass=false)` cell — the exact production shape.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Wrap-script: route on artifact, not exit code](feedback_wrap_script_route_on_artifact_not_rc.md) — when wrapping a script that exits non-zero on a domain HALT AFTER writing its artifact (K2 HALT, etc.), check the artifact first; rc is a fall-through crash signal only. #657 r3.
