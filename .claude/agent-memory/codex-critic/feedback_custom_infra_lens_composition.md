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
  Round two (#2326): reuse the surviving round-one span workspace — header
  verbatim + a ROUND SCOPE block (duties a-d: closure / regression / new
  surface / arithmetic); round-record span = sed-strip the session-id
  trailer off the prior `/tmp/...-output.md`, bracketed by Write-composed
  lead-in + binding-adjudication lines (near-verbatim brief text, digits
  balanced); fresh lens re-extract; the brief's LETTERED review groups
  verbatim as the questions span; tail adds `### Blocker-closure verdict`
  before the per-group answers; numcheck needs only path edits (suffix all
  new paths with the round tag) — PASS first try at ~87 KB. The quoted
  prior-round v-one marker block inside the prompt is harmless: the
  dispatch config names the v-two start tag + lens attr explicitly, and
  extraction runs on the OUTPUT file.
  Round three (#2326, alternating-sides history — each twin overruled
  once): same workspace recipe, fresh dir + round-tagged paths. New
  patterns that worked: (1) a brief-ordered CALIBRATION block — the twin's
  own prior-round inventory omission, quoted with the exact line numbers
  it covered vs omitted — goes in the round-record SPAN (digits balance),
  framed per the brief "as calibration rather than reproach" and ending
  "do not carry forward any prior inventory, including your own"; (2)
  mandate a fresh-evidence output section (`### Row inventory (built fresh
  from the live file)`) placed BEFORE the blocker-closure verdict, so the
  enumeration structurally precedes the coverage judgment the twin
  previously got wrong; (3) the brief's manufacture/withhold symmetry
  ("after two REVISE rounds, do not manufacture a third blocker to appear
  thorough, and do not withhold a real one because the plan has been
  revised twice") goes in a digit-free VERDICT CALIBRATION scaffold
  paragraph just before the output format; (4) a compact two-round history
  span (verdict pairs + reconciler sidings) REPLACES inlining full prior
  verdict outputs — inlining the round-two output would carry forward the
  exact flawed inventory the calibration bans. Numcheck PASS first try at
  ~84 KB.

- **Standard-lens infra spawn with an orchestrator lens-TRANSLATION note
  (#2152 round one, alternatives): when the brief names a STANDARD lens plus a
  prose translation ("for each claimed PROTECTION find the simplest realistic
  scenario where it silently fails") + press points, inline the note VERBATIM
  as its own span labeled "press points — leads, not pre-judged findings",
  extract the standard lens items verbatim as usual, and bridge them with a
  digit-free translation paragraph (predicted-positive-result → claimed
  protection; fatal-unweighable → undisclosed + unprevented + realistic;
  analyzer → the downstream code-review ensemble + test-verdict gate). Add a
  `### Press-point dispositions` output section (one hollow-protection /
  disclosed-residual / prevented line per press point, placed BEFORE
  What's-Good) so coverage is systematic; carry the brief's FATAL vs
  RECOVERABLE split as the Must-Fix bar (disclosed residual = Concern, and
  "a disclosure that materially understates the hole is itself a finding").
  Guard the plan's own declared declines (no-live-probe rationale, out-of-scope
  fence) as out-of-bounds unless an undisclosed silent-failure class rides
  them. Numcheck PASS first try at ~66 KB; scaffold avoided rule-number
  digits by writing "the pilot-gate rule" / "the api-refusal rule".

- **Live-service hazard ban (#2332): when the plan under review OPERATES ON a
  rate-fragile external service (an HF repo near its file-count cap, under a
  live concurrency constraint), the header's read-only block must explicitly
  ban the REVIEWER from calling that service** ("Do NOT make ANY HuggingFace
  Hub API call ... review from the LOCAL artifacts only") and the tail's
  verify-numbers nudge is rescoped to local artifacts. A Codex twin with
  shell access will otherwise verify claims by listing the live repo — the
  exact hazard class the plan guards. Also worked: brief-supplied NUMBERED
  attack points (not lens rubric items) as the lens span, each phrased
  "construct a concrete failing scenario, don't check prose exists"; and
  labeling a body constraints section "ESTABLISHED MEASURED FACT, not up for
  debate" in the inputs span to pre-empt relitigation of measured ops facts.
  Paths-only shape again; numcheck PASS first try at ~10 KB.

**Why:** first custom-lens infra compose (#2324) — these choices made the
numeric gate pass first try on a 68 KB prompt; round 2 (REVISE-round
re-review) passed on the second numcheck run, the only failures being the
hyphen-digit scaffold atoms above.
**How to apply:** any codex-critic spawn whose brief supplies its own lens
list and/or targets a `kind: infra` / workflow-fix plan; the REVISE-round
bullet for any round-2+ re-review brief with rejected-findings guards.

Related: [[plan-path-missing-read-from-main]],
[[stale-tmp-files-across-plan-versions]].
