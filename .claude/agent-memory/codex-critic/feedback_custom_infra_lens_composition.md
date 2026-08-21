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

- **Standard alternatives lens + fact-checker context_note (#2184 round one):
  when the brief carries a per-decision translation ("review alternative
  designs / wrong-premise risks per major decision: <named decisions>") PLUS
  a fact-checker verdict note (N CONFIRMED / one UNVERIFIED premise, bounded
  by a named kill criterion), inline the WHOLE brief verbatim as one span and
  frame it in the digit-free header lead-in as GIVEN FACTS: "re-filing a
  CONFIRMED premise as a finding without NEW code evidence is a regression;
  the single UNVERIFIED premise is a legitimate press point whose
  kill-criterion bounding you should judge". Output adds `### Per-decision
  dispositions` (one SOUND / SOUND-WITH-CONCERN / FLAWED line per
  brief-named decision; FLAWED ⇒ matching Must Fix) before What's-Good, and
  the labeled sub-questions are grouped BY DECISION (A-i…, B-i…) with a
  cross-cutting pair (kill-criteria completeness + monkeypatch-seam test
  vacuity). Live-service ban rescoped to the plan's own service (no RunPod
  API calls; local grep-anchored reads only). Renamed the concerns section
  to "Concerns the implementer / code reviewer should weigh" (infra has no
  analyzer). Numcheck PASS first try at ~65 KB (cat-assembly,
  version-suffixed tmp paths, scaffold digit-free — kill-criterion ids like
  K-two and status-code families like 4xx are safe only because their
  digits sit in the allowlist; spell out anything else).

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

- **Trigger-dense GUARD target, standard methodology lens (#2357 round one,
  guard_root_code_commit.sh pathspec-scoping widening): the brief's
  TRIGGER-DENSE note composes as a digit-free READ DISCIPLINE header block**
  ("read the hook ONLY via grep-anchored windows; reference by path +
  line-window; never paste large verbatim blocks") **plus a rescoped tail
  verify-numbers nudge** (grep-anchored line-window reads, never wholesale) —
  the #2332 live-service-ban analog for a trigger-dense FILE. Bar = the 19m
  F-classes written digit-free as named classes (FAIL TO FIX /
  PROTECTION-REMOVAL FALSE NEGATIVE / LIVE-CONSUMER BREAKAGE (literal pins) /
  TEST VACUITY / UNIMPLEMENTABLE); settled block lifted from the plan's own
  must-ask + out-of-scope-residuals lists; lettered press points (numbered
  lists past five residual the numeric gate) incl. the plan's own
  declared-UNVERIFIED assumptions as walk-the-argument leads; full standard
  lens span inlined with a digit-free verify-N/A bridge keyed to the plan's
  "Standalone N/A declarations". Verbatim orchestrator-brief blockquote as
  its own span carried all brief numerics. Numcheck PASS first try at
  ~124 KB (cat-assembly; version-suffixed tmp paths).

- **Writer+verifier-consumer SCHEMA-FIELD plan, standard methodology lens
  (#2194 round one: emit a card `phase` field in provenance.py + a
  verify_report consumer preference channel + rule docs): blend the 19o
  write-site/read-site chain probes with 19av verifier-check-addition
  probes as lettered press points** — (a) walk the emission-dict →
  consumer-walk chain AND grep helper call sites for spread-into-sentinel
  collisions (a lifecycle `phase` landing sibling of a commit key via
  `{**metadata}` on a path the new write-time validator never sees); (b)
  normalization-collision + preference-direction (exact channel pairing
  WRONG where the old token path would skip); (c) field-WIDTH blast radius
  (registered grep vs rejected broader grep, shim survivors, frozen
  dataclass trailing field); (d) legacy-population firing (fact-check
  found sibling-phase records repo-wide — fixture coverage of realized
  legacy shapes); (e) asserted message-fragment compat; (f) fail-loud
  validator posture vs the module's never-crash contract; (g) task-body
  scope-item narrowing (forward-only) judged against the body's own text;
  (h) per-test fixture-reaches-changed-code vacuity. Fact-check-already-ran
  brief clause composes as a digit-free FACT-CHECK STATUS header paragraph
  pointing at the plan's own FACT-CHECK CORRECTION entry. Numcheck PASS
  first try at ~136 KB (cat-assembly; spans = plan + body + brief +
  lensitems + empty prior; unsigned-atom tokenizer + isfinite canon guard;
  b1/b2/b3 and stage2-upload are scaffold-safe — their digits sit in the
  allowlist — while "8-hex"/"40-hex"/"§12.15" are not: write
  "abbreviated"/"full-length"/"the Assumptions section's FACT-CHECK
  CORRECTION entry").

- **Guard-predicate infra plan with an incident-replay claim (#2158 round
  one, methodology): when the plan's own falsification test is a trace
  against a REAL prior task's events.jsonl, make the replay a review target
  that names the artifact via a status-robust locator** (`uv run python
  scripts/task.py find <M>` — never a hardcoded `tasks/<status>/<M>` path,
  which goes stale on status moves) **and instructs Codex to walk the
  predicate spec against the claimed row indices AND probe the
  false-positive direction** (would healthy rounds trip it). Brief-numbered
  review targets elaborated with plan-sourced row/line/byte claims live in
  the targets SPAN (digits balance against the span file itself); each
  target answer ends HOLDS/FAILS/PARTIAL/UNVERIFIED, with UNVERIFIED
  would-be blockers filed as Should Fix. Byte-ratchet claims get an
  explicit "RE-MEASURE, never trust the plan's arithmetic" instruction.
  Standard #2357 shape otherwise (full lens span + digit-free verify-N/A
  bridge + settled-scope-cuts block from the plan's own must-ask list).
  Numcheck PASS first try at ~135 KB.

- **Standard statistics lens on an infra plan whose brief carries a NUMERIC
  translation note (#2360 round one, preflight venv-probe hardening): the
  brief's measurement-surface sentence (test-suite fail-before/pass-after +
  wall-budget + timeout + harvest window, with all its measured numbers) is
  inlined verbatim as its own briefnote span** — it carries every brief
  numeric, so the numcheck balances by construction (the #2152 shape, now
  proven for statistics). The digit-free bridge translates the lens as:
  pytest suite = the measurement instrument (vacuous-pass + mutation-walk
  per plan-named failure shape); each threshold/budget = a
  basis/derivation/decision triad, with venue extrapolation pressed via
  drift-tolerance-vs-decision-margin; probe outcomes = verdict-lattice walk
  (timeout-vs-fail conflation + lane gating turning a deploy-venue FAIL
  into a healthy-looking no-run); truncation/harvest window = a censoring
  rule on an n-of-one basis. A plan-recorded folded fact-check round gets a
  settled-fold guard in THE BAR's do-not-flag list ("unless the fold itself
  is wrong"). Numcheck PASS first try at ~122 KB.

- **Repo-wide sweep / cap-raise plan, standard methodology lens (#2391 round
  one, review-round cap raise — the #784 structural twin): brief's seven
  numbered questions verbatim as their own span + a mandated `### Answers to
  the seven review questions` output section with per-question verdict words
  (COMPLETE / MISS FOUND / HOLDS / FAILS / PARTIAL / UNVERIFIED) and
  quoted-command grounding ("every completeness claim quotes the exact
  command you ran"); RE-RUN-the-arithmetic instruction for plan-claimed
  counts (site counts, expected empty-allowlist residual). Infra bar gained
  a fifth clause for sweep plans: "destroys or falsifies the historical
  record (preserve-as-history sites)". Read discipline names the plan's own
  grep exclusions (worktrees + tasks/ + archive/ + external/) so
  sweep-completeness greps match the plan's provenance basis. Numcheck
  tokenizer upgrade that zeroed scaffold atoms entirely: BOTH-SIDES
  `(?<![\w.])...(?![\w.])` guards on the unsigned atom regex make hex SHAs
  produce NO atoms (structurally kills the [[numeric-gate-sha-overflow]]
  crash class — keep the isfinite guard anyway) AND make `v2`-style
  marker-tag tokens invisible; hyphen/slash-joined pairs still split
  correctly (`+0.74-0.80` → both atoms, verified by the in-script dynamic
  self-test). PASS first try at ~147 KB (spans = brief-inputs + lensitems
  (capsule + full reference span) + questions + plan + body + empty prior).
  Rounds two + three (#2391): the round-one header reuses VERBATIM (it is
  round-agnostic + digit-free); per round add a ROUND SCOPE block (history,
  load-bearing pattern, duties: replay-then-fresh-review) and compose the
  orchestrator brief's replay claims + fresh-review classes + settled list
  as ONE brief span carrying all digits, framed "recorded exit codes are
  CLAIMS — replay them" (v-four records rc inline; the gate must not trust
  it). Record spans = both twins' prior verdicts + the disposition
  epm:progress note, extracted from events.jsonl to files; Codex's own
  verdicts lead in as "YOUR OWN — verify resolution, do not re-assert".
  numcheck needs only path/span-list edits (fresh round-tagged workspace
  dir); round-three PASS first try at ~252 KB. Quoted prior-round marker
  tags in the records are harmless when the return summary tells the
  orchestrator to key OUTPUT-file extraction on the round-tagged start
  tag.**

- **Round-two on a paths-only standard-methodology infra compose (#2212 r2):
  the #2147/#2326 round-two recipe composes cleanly when the orchestrator
  itself names round-tagged prompt/output paths (`-r2` suffix) — no mv of the
  round-one version-less output needed; it becomes the record span raw
  material (sed-strip the Codex session trailer). Two small new patterns:
  (1) a RENUMBERING lead-in in the record span ("section numbering may have
  shifted between plan versions — locate criteria by content, never only by
  index") pre-empts a false NOT-CLOSED when the re-plan moved the criterion
  the twin's own round-one Must-Fix cited; (2) when concerns mix "is X
  resolved" and "is there a hole" polarities, mandate the verdict word PLUS
  an explicit referent clause ("CONFIRMED — both round-one Must-Fixes are
  resolved: ...") so CONFIRMED is never ambiguous between defect-present and
  fix-present. Output adds `### Round-one blocker closure` (CLOSED |
  NOT-CLOSED per prior Must-Fix, verified in the plan BODY, "the disposition
  table is a CLAIM about the body") before the per-concern answers. Numcheck
  PASS first try at ~94 KB (spans = roundscope + inputs + record +
  directives + lensitems + v-current + v-superseded + body + parent body).**

- **Self-referential path-discipline compose (#2422 round one, standard
  methodology lens): when the plan under review FIXES the worktree-stale
  `tasks/`-path defect, the header READ DISCIPLINE block states the same
  discipline self-referentially** — absolute main-checkout task paths only +
  an explicit worktree-`tasks/` read bar — and the composer verifies the
  `plan.md` symlink target + byte count at compose time, pinning
  `plan_version=` in the inputs span. Brief-supplied numbered questions (five)
  each got a mandated verdict word (HOLDS/FAILS/PARTIAL/UNVERIFIED;
  UNVERIFIED would-be blocker files as Concern) inside a `### Answers to the
  review questions` section nested in the STANDARD epm marker block. Also:
  VERIFY-AND-CORRECT the brief's structural claims against the live plan
  before inlining — the brief said "12 edits over 7 files"; the live §4.3
  count is 12 edits = 9 prose edits over 7 surface files + lint check + new
  test file + selector/manifest registration, and the inputs span states the
  corrected form. Numcheck PASS first try at ~152 KB (spans = inputs +
  questions + lensitems + plan + body + empty prior; unsigned atoms with
  both-sides guards, hyphen/slash pair-split, isfinite canon). Gotcha: the
  registry task-ref leg imports `task_workflow`, which needs the PROJECT env —
  running the numcheck with cwd=/tmp made `uv run python` resolve a bare env
  (no yaml) and the leg WARN-degraded to spans-only; run the numcheck from the
  repo root for the full two-leg check.

- **Detector/check-addition plan, standard statistics lens with a brief FP/FN
  translation (#2228 round one, a new verify_plan WARN-only check): the
  brief's detector translation ("measurement = the check's true/false-positive
  behavior") composes as a digit-free DETECTOR TRANSLATION header block
  (instruments = the pin-test suite + the corpus calibration scan; constructs
  = the check's FP/FN rates), the brief's numbered questions ride verbatim in
  the brief span (the #2360 shape), and the tail mandates `### Answers to the
  review questions` with per-question disposition words (SOUND | CONCERN |
  MUST-FIX | UNVERIFIED; an UNVERIFIED would-be blocker files as Concern) plus
  quoted-command grounding for the brief's spot-check question.** Infra bar
  gains a detector clause: fires-on-healthy-plans (quote the plan's own
  worse-than-no-check kill criterion) or misses-the-founding-incident =
  Must-Fix; test VACUITY and gratuitous-vs-load-bearing brittleness under
  KNOWN concurrent sibling edits get their own bar bullet. Carve-outs that
  worked: gag "make it FAIL not WARN" (the verifier file's own WARN-only
  doctrine), guard clarifier-recorded scope (new-check-not-widening,
  in-plan-baselines-only) and the plan's measured-and-rejected harvest arm;
  allow one-or-two bounded corpus-grep spot-checks but "do not exhaustively
  duplicate the concurrent fact-checker". Numcheck PASS first try at ~107 KB
  (spans = brief + lensitems + plan, no prior round; unsigned both-sides-guard
  atoms + a comma-grouping alternative FIRST in the atom regex so a
  thousands-grouped count canons as ONE atom instead of splitting at the
  comma — symmetric either way, but one-atom canon is the robust form).

- **Verifier-check plan, standard alternatives lens + REQUIRED attack lines
  (#2228 round one, a new-`verify_plan.py`-check plan): brief-supplied
  numbered "required lines of attack" compose as their own span with a
  mandated `### Press-point dispositions` section (one block per line, label
  words FATAL / RECOVERABLE / NO-FINDING; the evading-phrasings line gets
  per-phrasing `acceptable-residual` vs `guts-the-check` sub-verdicts).
  NEW move that generalizes: when the plan's protective value turns on
  what phrasings a harvest regex misses, EXPLICITLY ENCOURAGE the twin to
  run bounded targeted greps over the archived plans corpus
  (`tasks/*/*/plans/`) — "an empirical 'this phrasing exists in the corpus
  today and the harvest misses it' beats a hypothetical". Guard the plan's
  disclosed exclusions from auto-safe-harbor when the brief orders them
  judged ("disclosures are NOT automatic safe harbor here"). Brief verdict
  contract (`**Verdict:**` line, Must-Fix/Concern marks) nests inside the
  standard epm marker tags. Numcheck PASS first try at ~70 KB (spans =
  inputs + press + lensitems + plan + empty prior; unsigned both-sides-guard
  atoms + pair-split + isfinite canon + collect-all).

- **Lint-addition plan whose TOP press point attacks the task body's OWN spec
  choice (#2253 round one, alternatives on the prod-import-lockfile lint): add
  a digit-free CALIBRATION paragraph before the lens span** splitting
  spec-infidelity from claim-fragility ("the body's Goal pins the oracle —
  'the lockfile is not the installed set' is not by itself a design
  infidelity; judge whether the plan's CLAIMS — zero false positives, catches
  the incident class pre-pod — survive the declared-vs-installed gap; route
  FATAL / RECOVERABLE-with-a-named-home / already-handled"). Brief verbatim as
  its own span carried all brief digits; per-press-point dispositions mandated
  the oracle point answered BOTH directions (in-lock-not-installed /
  installed-not-in-lock) and forced a stated prior-art position under the
  dominating-alternative bar (rejection rationale misstates facts AND silent
  failure class). Read discipline banned the full no-flags lint run (slow; the
  check doesn't exist yet) while ENCOURAGING tomllib parses of uv.lock +
  bootstrap install-command reads + targeted import-shape greps. Fully
  digit-free scaffold → numcheck PASS first try at ~75 KB with ZERO
  scaffold-cleared atoms.

- **verify_plan check-ADDITION with a brief-supplied seven-question infra
  translation (#2228 round one, methodology): the 19p mechanical-verifier
  reframe + WARN-only blend composes as lettered questions (the brief's
  seven + a design-decision-grounding question standing in for the
  hyperparameter-grounding item), each with a mandated verdict word
  (SOUND/HOLDS/FAILS/PARTIAL/UNVERIFIED; UNVERIFIED would-be blocker files
  as Concern).** The brief's concurrent-fact-checker clause splits cleanly
  as: "the fact-checker verifies the cited lines/counts are TRUE; you
  verify a real source EXISTS and the inference from source to decision
  HOLDS" — design stays reviewable while mechanical re-verification is
  gagged. Earned probes that slotted into the questions: truthful-in-band
  escape for the cross-quantity corner (the escape phrase asserts
  "no gate" on a plan that HAS one); whole-doc H1 section-scoping defeat
  (the #947 probe) against the any-enclosing-heading idiom; survivor-corpus
  vs fresh-drafts calibration-population gap on the zero-corpus-FP kill
  criterion; dead-tripwire check on the no-regression kill criterion.
  Numcheck PASS at ~145 KB (spans = inputs + questions + lensitems + plan +
  empty prior). Self-test gotcha: the sha-overflow guard KEEPS a bare
  sha-like token as a LITERAL atom (canceling in the multiset) — assert
  `atoms("003e392548") == Counter({...: 1})`, never `not atoms(...)`
  (emptiness holds only for letter-ADJACENT forms like `003e...fcbb`,
  where the both-sides guards suppress the match).

- **Mutation-scoped service ban (#2241 round one, statistics): when the plan's
  OWN verification recipe requires read-only external-service probes (`gh pr
  list` / `gh pr view` / `gh help exit-codes` / `gh pr create --help` — the
  plan's §-assumptions each carry a "Verify: re-run the probe" line), the
  #2332 live-service ban scopes to MUTATIONS, not all calls**: explicitly
  permit the plan's read-only probes and ban the mutating forms by name
  ("no gh pr create other than its --help form, no git push"). Also proven
  again: the #2360 statistics-on-infra shape with the brief's lettered items
  as their own span + `### Answers to the review questions` with
  SOUND|CONCERN|MUST-FIX|UNVERIFIED verdict words; compose-time
  pre-verification of ALL the plan's measured byte/count/anchor constants
  (wc -c, cap-line grep, literal-baseline grep -c, heading line, test count,
  corridor arithmetic) stated in the brief span as "existence settled — your
  job is grain + decision adjudication". Numcheck PASS at ~115 KB (spans =
  briefnote + plan + lensitems + empty prior). Self-test gotcha: `8-hex`
  DOES produce atom 8.0 (hyphen-adjacent digit) — the #2194 "not
  scaffold-safe" list means NOT-ALLOWLISTED digits (40-hex, §12.15); do not
  assert `atoms("8-hex") == Counter()`.

- **Gate/oracle REPLAY-ARM plan, standard methodology lens + a short
  orchestrator translation phrase (#2430 round one, Step 9c suffix-replay
  arm): when the brief's translation is a bare phrase list ("design
  soundness of a fleet-critical gate change, trust-guard composition,
  controls, simpler-alternative check, design-constant grounding") rather
  than numbered questions, compose the press set YOURSELF from the 19ah/19r
  probe families as lettered leads: incident-replay eligibility-gauntlet
  walk (name the leg that would exclude the motivating incident's own
  records), per-guard composition with the plan's DELIBERATE
  prefix/suffix-arm divergences pressed hardest (incl. any
  strip-on-confirmed-but-aborted-attribution path), wrong-strip direction
  (protection removal — the worst direction for a merge gate, incl.
  short-circuit paths that substitute a cheaper replay for the full
  confirmation), bisection failure-mode walk, causal-channel fidelity of
  the replay vs the real gate (absent-prefix residual), fake-harness test
  vacuity vs the one real-subprocess pin, simpler-alternative soundness,
  exit/payload consumer spot-checks. Infra bar gains a WRONG-STRIP clause
  distinct from POSTURE FLIP. Fact-check-already-ran clause composes as a
  FACT-CHECK STATUS line inside the inputs SPAN (it carries brief digits
  like "1.5"/"4 corrections" — never scaffold). Digit hygiene notes: "Step
  9c"/"v4"/"pytest-9.0.2"/"MF-4c"/"B1" are atom-free under the both-sides
  guards, but decimal section refs (§4.4, §11.4) and bare §11/§12 are NOT
  scaffold-safe — write "the Decision Rationale section". Numcheck PASS
  first try at ~142 KB (spans = inputs + plan + body + lensitems + empty
  prior; run from repo root for the registry leg). Sibling-lens composers
  running concurrently keep separate version-suffixed workspaces — no mv
  needed, no collision.

- **Janitor/retention-sweep plan, standard methodology lens + orchestrator
  translation (#2246 round one, worktree-audit unmerged-branch keep +
  gate-launcher argv holder + fail-closed overlay assert): the translation's
  distinctive asks compose as lettered press points** — (a) per-AC
  mechanism-binding walk incl. whether the re-derivation invariant decides
  without the pinned snapshot; (b) no-behavior-change control pressed at the
  WIRING grain ("is the existing matrix sensitive to the wired call sites,
  or only the pure function — a control that cannot fail the wiring is not
  a control for the wiring"); (c) BOTH failure directions
  (false-retain=disk-bloat vs false-remove=founding hazard) walked per
  probe ARM per direction, with the synthetic-vs-live-fixture coherence
  named; (d) simpler-change + dropped-item-sufficiency (does the declined
  lease leave a realized data-loss scenario the named residual misses);
  (e) internal-convention conformance verified by grep ("a conformance
  claim that misreads the file is a defect, not style"); (f) three-valued
  contract: every error path's DIRECTION + does None-on-timeout create a
  silent permanent retention class with no surfacing channel; (g)
  fact-check-confirmed citations → judge only the source→decision
  INFERENCE (precedent transfer: watcher patch-id → no-fetch janitor
  form). Settled block = the brief's two settled conclusions with "judge
  the CURRENT plan text; do not re-derive from the task body's older
  wording" (the body still proposes the rejected cd form — gag the
  relitigation channel explicitly). Scaffold gotcha: a digit-bearing STEP
  FILE name (the Step 10d steps companion) residualed 18.0 — reference it
  as "the file the plan's Files-edited list names". Numcheck PASS on
  second run at ~156 KB (spans = inputs + briefnote + lensitems + plan +
  body + empty prior; unsigned both-sides-guard atoms + comma-grouping
  first + pair-split + isfinite canon + collect-all + registry leg from
  repo root).

- **Janitor/classifier-probe infra plan, standard statistics lens with a brief
  CIRCULARITY press point (#2246 round one, worktree-reap probe + re-derivation
  invariant): when the plan's acceptance check compares a probe against a
  janitor run that IMPLEMENTS the same probe, make the circularity question
  the mandated FIRST-and-deepest lettered question with a three-part shape**
  ((i) circular? (ii) if so, voided vs meaningful-as-WIRING-check — decision
  layer / both call paths / laziness / stronger-keep precedence — with
  classification correctness carried by named OTHER instruments (fixture
  arms, live measured cases); (iii) name the residual error class NO
  instrument catches + adjudicate). Population-drift snapshot note extension:
  membership delta = expected drift, NOT a finding; CLASSIFICATION
  disagreement on a worktree present in both reads = precise finding (numbers
  pre-confirmed twice — state that, plus a manufacture/withhold symmetry line
  in the tail). Detector translation reused from #2228 (false-MERGED /
  false-UNMERGED / error-channel directions, each with its stated fail
  direction). Legitimate SHARPENER inside a brief question (memory rule:
  sharpen 1:1, never a parallel list): the task-grain-vs-branch-grain cell —
  marker evidence is task-grained while suffixed sibling worktrees share one
  events file, so a later epm:merged note can false-MERGE a sibling; framed
  as a question lead, never an asserted finding. Read-discipline additions
  that fit this class: ban `git fetch` BY NAME with the reason (a fetch moves
  origin/main and changes the population under review) and ban even the
  report-only whole-janitor run (minutes of runtime; probe worktrees
  individually); spot-check rows for possibly-reaped worktrees get an
  explicit "mark UNVERIFIED rather than guessing" escape. Numcheck PASS first
  try at ~121 KB (spans = briefnote + questions + lensitems + plan + empty
  prior; scaffold digit-free — "v3"/"Step 10d"/"9c" are atom-free under
  both-sides guards; ISO-timestamp examples and :line refs confined to the
  questions SPAN where they balance).

- **workflow_lint FAIL-level check-ADDITION with grandfathered-offender
  dispositions, standard methodology lens + brief-supplied five press points
  (#2253 round one — the 19aw/#930 shape, brief-translated): the brief's five
  bulleted press points ride verbatim in the brief span labeled "(a)-(e) in
  order", and the composer adds the 19aw earned probes as lettered (f)-(k):
  motivating-class fire check via imagine-the-dispositions-absent (walk the
  predicate against the plan's own site inventory + verify its
  detection-predicate trace's logic, noting the fixture reproduces the
  incident's HISTORICAL state when the live site is remediated);
  seed-absorption at the mandated implementation-start probe re-run (does
  "extend dispositions" license auto-waivering genuinely-new offenders,
  especially when the must-ask list bars lockfile adds); waiver GRAIN
  (per-(file,root) exempts future sites) + placement grammar vs multi-line
  parenthesized imports; first-party resolver FALSE-NEGATIVE direction (a
  script stem colliding with a third-party root exempts it repo-wide — the
  inverse of the class-A FP fix); registration/wiring completeness (dispatch
  regex, no-flags detection, chain order, scope classification, WARN-line
  path-free contract); fail-loud arms + constant grounding. Bar = the infra
  translation with a fires-on-healthy / motivating-class-miss /
  posture-flip / lands-red / acceptance-fail / unimplementable clause set
  plus the 19p no-gate-today baseline; disclosed accepted residues judged
  only for materially-understated holes. Fact-check-complete brief clause →
  digit-free FACT-CHECK STATUS header block pointing at the brief span + the
  plan's own dispositions section ("design critique, not fact
  re-verification"). Scaffold digit gotchas confirmed: "PEP-503" DOES atom
  (503, hyphen-adjacent) — keep it span-only, write "the name-normalization
  standard"; "Phase 1.5"/"§13" span-only. Read discipline allowed re-running
  the plan's committed read-only recon probe (seconds) alongside bounded
  greps. Numcheck PASS first try at ~148 KB (spans = briefnote + lensitems +
  plan + body + empty prior; unsigned both-sides-guard atoms + comma-grouping
  first + pair-split + isfinite canon + collect-all + registry leg from repo
  root; version-suffixed workspace `-v2` keyed to PLAN version).**

- **Sync-FAMILY membership-coupling plan, standard methodology lens + brief
  roman-numeral press points (#2260 round one, Step 5a FAMILY_OF `agents`
  family + completeness guard): the brief's press points arrived as (i)-(v)
  roman numerals — keep them roman in scaffold references (digit-free by
  construction, no relettering needed) and letter composer probes (a)-(f).**
  Probe set that fits any FAMILY_OF/membership-list plan: seed-absorption at
  the implementer re-sweep (#930 move); MEMBERSHIP SELF-UPDATE ORDERING (the
  family table lives in a step file that is itself synced as a member of
  another family — walk the stale-copy transition window, decide bounded vs
  open-ended); guard genuineness (non-vacuity pin + bash deleted-member repro
  each fail for exactly its defect's reason; plan's own detection-predicate
  trace judged for logic); containment arm BOTH directions (false-dirty
  staleness vs residual whole-sync wedge, confirm claimed checkout atomicity
  against the quoted bash, ask whether the arm covers every family's batch);
  landing completeness (pinned-literal rewrite + mirror-tuple + fixture-stub
  extensions spot-checked against live anchors); ratchet RE-MEASURE (wc -c +
  cap-table grep, remedy sequenced on REALIZED post-edit bytes). Settled block
  from the plan's Deviations must-ask list with the judge-soundness-not-scope
  carve-out ("dropping the containment arm is a recorded park-question — judge
  its soundness, don't demand the scope change"). Read discipline: ban git
  fetch (population under review) AND pytest (guard doesn't exist yet;
  suite subprocess-heavy); allow read-only git log for commit-frequency
  pricing claims. Scaffold digit traps dodged: "guard (20)"/"guard (19)" →
  "the new completeness guard"/"the existing helper-import guard"; the Step
  10d mirror's digit-bearing filename → "its Step 10d mirror"; "test (1)"/
  "(9)"/"(10)" are allowlist-safe. Numcheck PASS first try at ~148 KB
  (spans = briefnote + lensitems + plan + empty prior; unsigned both-sides-
  guard atoms + comma-grouping first + pair-split + isfinite canon +
  collect-all + registry leg from repo root; version-suffixed workspace
  keyed to PLAN version).

**Why:** first custom-lens infra compose (#2324) — these choices made the
numeric gate pass first try on a 68 KB prompt; round 2 (REVISE-round
re-review) passed on the second numcheck run, the only failures being the
hyphen-digit scaffold atoms above.
**How to apply:** any codex-critic spawn whose brief supplies its own lens
list and/or targets a `kind: infra` / workflow-fix plan; the REVISE-round
bullet for any round-2+ re-review brief with rejected-findings guards.

Related: [[plan-path-missing-read-from-main]],
[[stale-tmp-files-across-plan-versions]].
