---
name: custom-infra-lens-composition
description: Composing a codex-critic prompt for a kind:infra plan when the orchestrator hands a CUSTOM lens set (not the 3 standard lenses) — precedent #2324
metadata:
  type: feedback
---

When the orchestrator brief hands a CUSTOM code/infra lens list (kind: infra
plan critique — precedent task #2324, symlink/FIFO lock hardening), the
standard 3-lens enum gate does not apply: the brief's verbatim lens list IS
`{{lens_items}}`. Rules that worked:

- **Inline the brief's lenses VERBATIM** (numbering, file:line cites, errno
  names included) and write the same text to the lens-items span file, so the
  numeric gate balances exactly.
- **Assemble the prompt by `cat`-concatenating the span files themselves**
  (header + lensitems.txt + plan file from its repo path + body.md + a
  clarify.txt extracted via `task.py view --json` + tail): spans in the
  prompt are then byte-identical to the span files the numcheck reads, so
  the multiset diff passes by construction. Composed scaffold pieces
  (header/mid/tail) via the Write tool, never a Bash heredoc (git-verb prose
  in lens text trips the argv guard, #1756).
- **Keep scaffold text digit-free outside the {0,1,2,3,4,5,500} allowlist**:
  say "the Leg-A/Leg-B sequencing lens" not "lens 7"; avoid filenames with
  digits (`step9c_baseline.py` → "the script the lenses name"); avoid "Step
  9c" in the snapshot note ("test-verdict gate"). Any scaffold digit outside
  the allowlist residuals as a false BLOCKER because span copies are already
  fully consumed by the verbatim inlines.
- **Custom output envelope beats the standard marker block** when the brief
  says in-context / no marker: #2324 required `## Codex Critic Verdict —
  APPROVE|REVISE` + BLOCKER/MAJOR/MINOR/NIT groups + literal "unverified"
  tags. Keep the mechanizable-tag and grounding (file:line) duties from the
  standing rule; add "unverified would-be BLOCKER files as MAJOR".
- **Infra-plan bar phrasing**: replace "flip the experiment's conclusion"
  with (a) ships a defect into the fleet-wide surface (blast radius), (b)
  silently changes a fail-open/fail-closed posture or exit-code contract,
  (c) task fails its own body `## Acceptance` list, (d) unimplementable as
  written. Explicitly gag the experiment lenses (measurement validity,
  dual-DV, baselines, data realism, seeds) — "do not manufacture findings
  from them".
- **Cross-branch code under review**: when a lens targets code only on a
  sibling branch, instruct Codex to read it via
  `git show origin/issue-<M>:<path>` and extend the snapshot note: don't
  REVISE because the dependency's live status moved; judge whether the
  plan's sequencing handles every outcome.

- **REVISE-round (round 2+) shape**: the brief's required-changes list +
  rejected-findings quotes + any orchestrator adjudication item ALL go in
  ONE reviewitems span file (they carry the file:line digits). Prompt
  order: header (delta scope: (a) each required change made correctly,
  (b) new defects only; verdict semantics = APPROVE iff all PASS + no new
  BLOCKER/MAJOR) → reviewitems → plan vN → body → concerns.jsonl verbatim
  (the persisted round record is the ideal prior-critique span — cat it
  raw) → tail. Output adds two named sections before the findings groups:
  `### Required-change verdicts` (PASS/FAIL per item) and the adjudication
  section; instruct "re-raising a reconciler-REJECTED finding is a
  regression, not a finding", quoting each rejection MECHANISM so Codex
  can distinguish new evidence from relitigation.
- **Hyphen-digit compounds are scaffold poison under the sign-rule
  tokenizer**: `ROUND-1` / `round-1` tokenize as atom -1.0 (not
  allowlisted) and `D6` as 6.0 — both residualed on the round-2 compose.
  In scaffold (header/mid/tail) write spaced forms ("round 1", "the
  pairing-fix code block"); inside span files any form is fine (balanced
  by construction).
- **mv the prior round's output file aside** before returning the config
  when the orchestrator's output path is version-less
  (`/tmp/codex-output-issue-<N>.md` → `*.v<k>-stale.md`) — the
  stale-tmp-files rule's fallback arm.
- **Same-task round-2 recompose: the prior round's /tmp artifacts are the
  raw material.** When no concerns.jsonl exists, the persisted round record
  = the `epm:plan-critique` note in events.jsonl (Claude side) + the prior
  round's version-less `/tmp/codex-output-issue-<N>.md` (Codex side — inline
  it verbatim as "YOUR round-one verdict"; the self-recognition framing plus
  quoting its own no-finding premise answers is the do-not-relitigate
  mechanism). The prior compose's numcheck script needs only path edits;
  suffix all new paths `-r2` so the stale round-1 files never collide and
  need no mv (used on #2147 r2 — numcheck PASS first try at ~220 KB).

- **Folded standard lenses (#2148 r2): when the brief names STANDARD lenses
  with one folded in ("Methodology (with Alternatives folded in)") plus press
  points — not a custom lens list — extract BOTH lens sections verbatim from
  critic-lens-reference.md into ONE lens span (marker lens attr stays the
  primary lens), quote the brief's press points + round-record verbatim as
  the prior-critiques span (works when no prior Codex /tmp output exists —
  Claude-only round one), add a digit-free verify-N/A scaffold naming the
  experiment-only items BY NAME with reuse-fitness called the central live
  item, and add a `### Round-one fix sufficiency` output section (one
  SOUND/INSUFFICIENT line per brief-named fix; INSUFFICIENT ⇒ matching Must
  Fix). Numcheck passed first try at ~136 KB with the collect-all +
  task-ref-extraction-first script shape.

- **Paths-only composition (#2325): when the brief orders artifacts pointed
  at BY PATH (never inline the plan body), the numcheck spans are just the
  brief-derived text blocks** (artifact list, facts-given, review questions,
  gag sentence) — the prompt shrinks to ~8 KB and the gate passes first try.
  When the brief prescribes verdict sections (APPROVE|REVISE + Must-Fix/
  Should-Fix/Notes) but no custom envelope, keep the STANDARD epm marker
  tags and nest the brief's sections inside; an added `### Answers to the
  review questions` section (one block per brief question + one line per
  prior-round finding: resolution correct/complete?) gives the brief's
  judge-the-fix duty a home. A brief "Facts already established" block,
  inlined verbatim with a "given — re-filing is a regression" lead-in, IS
  the do-not-relitigate mechanism when the prior round was Claude-only and
  no prior Codex /tmp output exists.

- **Paths-only + full two-lens verbatim inline compose fine together
  (#2326 round one): brief ordered artifacts BY PATH (plan symlink, body.md
  Non-goals, named events.jsonl markers), a facts-given verify_plan PASS
  line, seven numbered review questions, AND full Methodology+Alternatives
  inline with Statistics declared N/A.** Assembly: header (infra bar +
  Statistics-N/A gag + grounding) → inputs span (paths + facts-given) →
  digit-free mid intro → lensitems span (both sections sed-extracted
  verbatim) → questions span (brief's questions verbatim, incl. file:line
  cites like `task_workflow.py:8294+`) → tail (standard epm marker tags
  nesting the brief's `**Verdict:** PASS|REVISE` contract + `### Answers to
  the review questions` + separate NITs). Numcheck spans = inputs +
  lensitems + questions + empty prior file; passed first try at ~80 KB.
  Lens attr stays `methodology` (primary) when Alternatives rides along.

**Why:** first custom-lens infra compose (#2324) — these choices made the
numeric gate pass first try on a 68 KB prompt; round 2 (REVISE-round
re-review) passed on the second numcheck run, the only failures being the
hyphen-digit scaffold atoms above.
**How to apply:** any codex-critic spawn whose brief supplies its own lens
list and/or targets a `kind: infra` / workflow-fix plan; the REVISE-round
bullet for any round-2+ re-review brief with rejected-findings guards.

Related: [[plan-path-missing-read-from-main]],
[[stale-tmp-files-across-plan-versions]].
