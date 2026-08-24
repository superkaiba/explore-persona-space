---
name: Fix-verification rounds — "body caption" fixes live in the held clean-result draft, not the marker
description: When a round-N residual names a "body caption" fix, the caption artifact is the orchestrator's held clean-result draft (/tmp/issue-<N>-clean-result-body-held.md), which must be located and passed explicitly or the fix is unverifiable
type: feedback
---

On a narrow fix-verification round (round 3+ after an agreed residual), a fix
described as "incl. body caption" is NOT verifiable from the
`epm:interpretation vN` marker body alone: the interpretation marker often
carries no figure embeds/captions at all, and the task `body.md` may still be
the original pre-promotion body. The caption the prior-round critics reviewed
lives in the orchestrator's HELD clean-result draft —
`/tmp/issue-<N>-clean-result-body-held.md` — which carries the `![...]`
SHA-pinned raw-URL embeds and the `> **Figure.**` caption lines.

**Why:** on #1073 r3 (2026-07-06) fix 2's caption component ("~+0.004–0.008
(short)" → "~+0.002–0.008 (short; L26 +0.0020)") existed ONLY in the held
draft; grep of the marker body, task body.md (both trees), and the figure
sidecar all came up empty. Without passing the held-draft path, Codex would
either mark the caption fix unverifiable or falsely "still missing".

**How to apply:** at compose time for any fix-verification round, enumerate
EVERY surface each residual fix touched (marker body line, held clean-result
draft caption, figure .meta.json sidecar, rendered PNG) and pass each by
absolute path with its expected corrected text quoted. Locate the held draft
via `ls /tmp/issue-<N>*` and confirm the stale string is gone + the corrected
string present before composing. Also pass BOTH interpretation versions
(v(N-1) and vN) so Codex can run the unintended-delta diff-scan itself.
