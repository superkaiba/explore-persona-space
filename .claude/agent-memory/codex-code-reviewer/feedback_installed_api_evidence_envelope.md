---
name: installed-api-evidence-envelope
description: When a review emphasis requires verifying rule/doc text against an INSTALLED third-party source (.venv) the worktree cannot resolve, inline verbatim source excerpts with line numbers as a guaranteed-evidence envelope (#2442 r1)
metadata:
  type: feedback
---

Inline an `---BEGIN/END INSTALLED-API EVIDENCE---` envelope when the brief's
review emphasis is API-semantics truthfulness against an installed package
(e.g. `huggingface_hub` `hf_api.py`) — the worktree's `.venv` is typically a
stub without site-packages (verified on issue-2442: `$WT/.venv` existed but
had no `hf_api.py`), so a by-path instruction alone risks a false
`data-access-blocked` or a lens scored off the plan/body instead of the
source.

**Why:** #2442 r1 — the task BODY asserted `get_paths_info` 404s on a wrong
prefix; the installed docstring says a nonexistent path "is ignored without
raising an exception". The plan corrected the body, and the review had to
rank installed source > plan > body. Codex could not read the repo-root
`.venv` path reliably from its worktree-rooted sandbox, so the excerpts were
the only guaranteed evidence channel.

**How to apply:**
- Extract at compose time with `sed -n` by line ranges located via
  `grep -n "def <fn>"`; keep signature + docstring-through-Raises + the
  implementation body VERBATIM with source line numbers stated; elide long
  docstring examples with an explicit `[... elided ...]` marker.
- Append composer NOTE lines stating the load-bearing derivable facts (e.g.
  "generator — 404 raises at ITERATION, `yield` at line N"; "Raises block
  lists NO EntryNotFoundError") so Codex can cite them even if it distrusts
  its own trace.
- Instruct: prefer reading the live absolute `.venv` path when reachable;
  the envelope is the fallback; NEVER mark the API lens BLOCKED when the
  envelope is present (same never-BLOCKED discipline as the inlined
  plan/marker envelopes).
- Extend the prompt-file Step-3 validation greps to the new envelope tokens.

**Re-probe the stub status EVERY round (#2442 r2):** the worktree `.venv`
flipped from stub (r1: no hf_api.py) to REAL (r2: full 0.36.2 install at
`$WT/.venv/lib/python3.12/...`) between rounds — someone ran `uv sync` in the
worktree. Do not carry the prior round's stub finding: `find "$WT/.venv" -name
hf_api.py` + version + `grep -n "def <fn>"` anchor parity vs the envelope at
every compose. When real and version-matched, attest DUAL readability (live
path preferred, envelope fallback) instead of envelope-only. Also note the
main-checkout venv can live under a DIFFERENT python dir (py3.11 vs the
worktree's py3.12) — a `cmp` against a guessed main path fails on
FileNotFoundError, not content difference; resolve via
`uv run python -c "import huggingface_hub, inspect; ..."`.

**Self-initiate the envelope when the fix's correctness IS an API claim
(#2262 r1):** no brief emphasis needed — a diff whose discriminator rests on a
third-party class hierarchy (matplotlib `_CollectionWithSizes` vs
`LineCollection`/`QuadMesh` `get_sizes` presence, `get_offsets` zeros((1,2))
placeholder) gets the pinned line-number facts inlined + the worktree venv
path named (2262's WT venv was FULLY populated at r1, py3.12, mpl 3.10.8 —
dual readability attested), with an explicit "ground API claims in source you
read, never the implementer's docstring; say so when unverifiable" line.

**r2 extension (#2262 r2): producer ENUMERATION + dissociation NOTE.** When
the round-2 fix swaps the discriminator to a different API internal (marker
sizes → `Collection._offsets` provenance), the envelope upgrades from pinned
class-hierarchy facts to (1) the internal's full WRITE-site enumeration
(constructor `offsets=` sites + `set_offsets` callers, grep-derived with
file:line, flagging docstring false-positives like collections.py:1530) and
(2) a composer NOTE stating any derivable probe DISSOCIATION without
resolving severity (here: `np.asanyarray` preserves array subclass, so
constructor-supplied plain-ndarray offsets — hexbin, Quiver — make the
private `_offsets` tell and the public MaskedArray tell genuinely diverge;
that grounded the plan's probe choice AND gave Codex the hunt set for
"set-but-not-scatter-data" injection cases). Also r2-shape lessons: an
UPHELD own-FAIL closure round inlines the RECONCILER verdict (tags stripped)
as the acceptance contract and SKIPS re-inlining the twin's own r1 verdict
(subsumed once the reconciler independently reproduced every finding —
cleaner tag arithmetic, ~12 KB saved); attest the falsification artifact's
full pytest tail (`4 failed, 24 passed, 28 deselected`) — the marker's
"4 failed / 24 passed" omitted the deselection scoping, and an unattested
count mismatch vs the file's test count reads as fabrication; and when a
brief-cited orchestrator note carries imprecise line anchors (1068-1074 for
a getattr at :1094), pin the exact anchors as compose-time facts so Codex
cites its own.

Related: [[whole-round-unsplit-compose]] (round-pinned sha-range diff when an
out-of-scope spec-sync commit sits at HEAD), [[infra-wf-fix-lint-gate-compose]]
(N/A-by-type + duty-discharge attestations for infra rounds),
[[revision-round compose recipe]] (the round-2+ delta recipe this compose
followed).
