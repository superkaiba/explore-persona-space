---
name: fanout handshake-timeout masks a single fast-crashing unit
description: "ALL fan-out units showing the vLLM 5-min front-end handshake timeout is usually a SYMPTOM of ONE unit crashing instantly — find the earliest unit traceback before classifying infra"
type: feedback
---

When a multi-unit vLLM fan-out is reported as "all N units hit
`RuntimeError: Did not receive response from front-end process within 5
minutes` simultaneously" (engine-init wedge appearance), do NOT accept the
infra/wedge classification before checking each unit log for the EARLIEST
traceback. A `_fanout_units`-style driver that raises on the FIRST unit
failure exits and abandons the sibling front-end processes mid-engine-init;
their orphaned `VLLM::EngineCore` children wait the 5-minute handshake
window and then all dump the timeout RuntimeError — timestamped ~5 min
AFTER the true crash, in every sibling log.

**Why:** Incident #1112 attempt 4→5 (2026-07-08): attempt 4 was diagnosed as
"all 4 capture units hit the EngineCore handshake timeout (engine-init
wedge — infra)" and respawned; attempt 5 hit the identical crash in ~20 s.
The true cause was one unit (`capture m1_lora_band8/selected`) dying in ~5 s
on `FileNotFoundError: .../m1_lora_band8/selection.json` — a deterministic
dispatcher bug (`_resolve_capture_model` required a file no phase writes for
that cell). The m1 traceback had been sitting in its unit log since attempt
4; the 3 sibling handshake timeouts were downstream noise.

**How to apply:** On any fan-out failure, sort the unit logs by crash
timestamp (or diff file sizes — the fast-crasher's log is tiny) and read the
EARLIEST failing unit first. Classify `code` if it is a deterministic
in-repo traceback; the sibling handshake timeouts then need no explanation.
Also expect abandoned-sibling `VLLM::EngineCore` orphans on the host after
such a crash — probe `pgrep -af '^VLLM::EngineCore'` and kill by exact PID
before any relaunch (they self-expire at 5 min but can strand HBM).
Beware pgrep self-match over raw SSH: use bracketed patterns
(`EngineCor[e]`) since the probe pattern is in your own argv.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Fan-out handshake timeout masks a single fast-crashing unit](feedback_fanout_handshake_timeout_masks_single_unit_crash.md) — "ALL units hit the vLLM 5-min front-end handshake timeout" is usually the SYMPTOM of one unit crashing instantly (driver raises on first failure, abandons sibling front-ends; their EngineCores time out 5 min later); read the EARLIEST/smallest unit log's traceback before classifying infra (#1112 a4→a5)
