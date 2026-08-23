---
name: figure-series-extension-neutrality-probes
description: How to certify a "default render unchanged" claim when a renderer gains an opt-in series; PNG byte-compare is invalid, err-by-x last-wins cuts both ways
metadata:
  type: feedback
---

When a shared figure renderer gains an opt-in series + post-legend artist relabeling and claims the default render is unchanged (#1901 r1 g3):

1. **PNG byte-compare is an INVALID instrument** — `savefig_paper` embeds a per-run render id in pnginfo, so even same-module run-to-run renders differ in bytes. Certify with (a) pixel-array compare (PIL → np, max delta + differing-pixel count) between a parent-blob render (`git show <sha>^:<file>` written to a temp module in scripts/, imported side by side) and HEAD, and (b) the sidecar tuple-multiset equality the round's own gate defines. Run the determinism probe (same module twice) BEFORE reading a byte diff as a finding.
2. **`err_by_x` last-container-wins cuts both ways** — drawing the new series FIRST keeps every inherited point's sidecar `error` stable (verify empirically at overlapping x, not just by reading), but the NEW series' sidecar errors are then the last inherited container's values at shared x (measured: own sd 0.00163 vs sidecar 0.00326). Gate unaffected when new rows are label-excluded; flag the contamination as a Minor with the eval-JSON as the true-error source.
3. **Post-legend relabel side channels:** `fig.legend(handles, labels)` snapshots label text — relabel after is render-inert; but relabeling data_line/caplines flips `_extract_lines`' internal ≤2-vertex drop for any series with ≤2 points (check every inherited series' vertex count), and check the sidecar row cap (`_MAX_SIDECAR_ROWS`) against the new total — overflow cuts the inherited TAIL and the gate direction matters.

**Why:** the first byte-compare read "PNG differs" and would have been a false Major; the determinism probe re-attributed it to pre-existing pnginfo nondeterminism, and pixel compare showed 0 differing pixels.
**How to apply:** any review of a figure-renderer extension claiming default-render neutrality or sidecar-gate stability.
