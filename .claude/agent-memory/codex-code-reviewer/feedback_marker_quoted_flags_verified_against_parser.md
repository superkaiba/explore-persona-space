---
name: marker-quoted-flags-verified-against-parser
description: Verify every CLI flag the implementation marker's quoted commands carry against the script's real argparse block at compose time; hand Codex a neutral discrepancy fact, never a pre-adjudicated severity (#2502 r10)
metadata:
  type: feedback
---

At compose time, grep every flag token in the marker's (c) / smoke
commands against the touched script's `add_argument` block AND its
parse strictness (`parse_args` vs `parse_known_args`). A quoted flag
with no definition under strict `parse_args` means the command AS
QUOTED exits 2 — the run evidence may still be real (transcribed from a
different invocation), so the composer's product is a NEUTRAL
"COMPOSER-VERIFIED COMMAND DISCREPANCY" block: the flag list with line
anchors, the strict-parse fact, the plausible real form, and an
explicit "adjudicate under Step 0.5 present-but-imperfect vs the
copy-pasteable-command contract — never a marker-shape ABSENCE".

**Why:** #2502 r10 — the v10 marker quoted `--schema-gate-report
<path>` in two commands; `build_parser()` defines only
`--skip-schema-gate` / `--schema-gate-rows` / `--schema-gate-only`, and
`main()` parses strictly, so the quoted command cannot run. The report
path was actually `out_dir / "schema_gate_report.json"` — a
transcription artifact of `--out-dir`. Without the compose-time check,
Codex either misses it (trusting the digest) or over-penalizes it
(FAILing a present-evidence transcription slip); argparse prefix
abbreviation must be reasoned about before claiming exit 2 (the token
must not be a prefix of any defined flag).

**How to apply:** on every compose, extract `--<flag>` tokens from the
marker's quoted commands for round-touched scripts, grep the argparse
block, check prefix-abbreviation collisions, and confirm strict
parsing. Discrepancy found ⇒ neutral-fact block in the review context +
a pointer from the Step 0.5 compose-time observations. Related:
[[revision-round-compose-recipe]], [[reconstructed-marker-compose]].
