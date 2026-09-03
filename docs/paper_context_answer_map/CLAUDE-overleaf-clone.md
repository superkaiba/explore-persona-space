# CLAUDE.md — Context→Answer Map paper (ICLR 2027)

Rules for any Claude session working in this repo (git clone of Overleaf project
`6a59c927290f8b8b5eee0055`; Thomas reads ONLY in Overleaf).

## Repo + workflow
- **`plan.tex` was REMOVED from the Overleaf tree on 2026-09-03 (Thomas: "remove
  plan.tex"; commit 009ad61).** Every plan.tex reference below is historical. Do not
  recreate it; the plan of record is EPS `docs/paper_context_answer_map/plan.md`.
- An Overleaf Git token is available locally through an existing authenticated
  Overleaf remote. Reuse that credential with the requested Overleaf project ID
  when the target clone is absent; never print, copy into prose, or commit the
  token itself.
- ALWAYS `git pull` before reading; commit + push after ANY edit (unpushed edits are
  invisible to Thomas). This Overleaf tree carries ONLY the documents the paper
  compiles from (user directive 2026-08-19) — never commit drafts or build artifacts
  here. ONE EXCEPTION (Thomas 2026-08-23): **`plan.tex`** at the root — the SIMPLE plan
  skeleton he reads inside Overleaf (`.md` files do not open in the Overleaf editor;
  verified 2026-08-23). Format per his spec: Intro/Related-work/Methodology as
  subsections + bullets; Results as claim + plot + evidence-in-plot + transition
  blocks; Discussion as implication/limitation → future-work bullets. It is
  HAND-EDITED and CO-EDITED — never regenerate it from `plan.md` (a regeneration
  clobbers his edits; the earlier pandoc-mirror scheme was retired 2026-08-23 same
  day). Division of labor: `plan.tex` = the one-glance structure view, keep it in
  sync when the paper's structure/claims change (targeted edits, pull first);
  detailed plan of record = EPS `docs/paper_context_answer_map/plan.md`; per-section
  working plan blocks = `draft.tex`. No other planning doc lives here. Thomas reads
  it via Menu → Settings → Main document → `plan.tex`, Recompile (standalone article,
  no `style/` dependency).
- STRUCTURE (names final 2026-08-23; spine 2026-08-22): THREE roots, no preamble
  file. `main.tex` = the curated paper (inputs `sections/clean/` copies only;
  currently abstract + introduction; the brief clean.tex rename was retired same
  day on Thomas's order). `draft.tex` (ex-outline.tex) =
  the working surface: per section a blue PLAN block (bullets + one plot per claim
  inline + status tags [DONE #N]/[IN-FLIGHT #N]/[NEW]/[VERIFY]) followed by the
  draft via \input. `plan.tex` = the hand-edited simple skeleton (see below).
  `preamble.tex` was REMOVED (Thomas 2026-08-23): the shared header block
  (packages/title/`\iclrfinalcopy`) is INLINED identically in main.tex AND
  draft.tex — edit it in BOTH or they drift. Overleaf can RESURRECT a root that
  its project settings still point at (commit a2ea028 restored stale pre-rename
  main.tex/outline.tex): after any root rename, Thomas must reselect Menu →
  Settings → Main document, and the next pull checks for resurrected stale roots. Draft text lives in
  `sections/NN_<name>.tex`. Every results section, main text and appendix alike,
  is its own file under `sections/results/` (`01_accuracy.tex` to `06_behavior.tex` for the
  main-text results, `a1_*.tex` to `a7_*.tex` for the appendix results), input in
  place by the spines `sections/04_results.tex` and
  `sections/05_method_details.tex` (split 2026-09-03). Edit the result files,
  never the spine's `\input` lines. A transition sentence opens the result it
  leads into and never closes the previous one. `references.bib` = the
  bibliography; `style/` = template machinery — NEVER edit style/. Figures under
  `figures/paper/` (claim-named c1_*.pdf…c5_*.pdf); `poster/` stays (Thomas
  2026-08-23). Compile check after edits: pdflatex → bibtex → pdflatex ×2 on
  main.tex AND draft.tex (plan.tex: one pdflatex pass suffices, no bib).
- WRITING TELLS — STANDING RULE (Thomas, 2026-08-23). In ALL rendered text in this
  tree: (1) ZERO em dashes (`---` or —); rewrite with colon/comma/parentheses or
  restructure; en-dash `--` numeric ranges are fine; %-comments, references.bib
  (fetched titles verbatim), and style/ are exempt. (2) No metaphor jargon of the
  "load-bearing"/"backbone"/"scaffold" family: name the mechanism instead. (3) Avoid
  contrastive-negation scaffolds ("is X, not Y", "not X but Y") as a rhetorical
  default; state what IS the case, keep the negation only when the contrast itself
  is the claim. (4) NO SEMICOLONS in rendered prose (Thomas 2026-09-03, on the
  abstract: "Don't use semicolons"); split into sentences or use a colon/comma.
  Also from that session: the abstract names its object "context-answer metamodel"
  / "linear context-answer metamodels" (never "the map"), and carries a number on
  every claim. Enforcement (the `/writing-tells` skill,
  `~/.claude/skills/writing-tells/`, owns all three layers): (a) the clone's
  `.git/hooks/pre-commit` runs the skill's mechanical gate
  (`check_paper_tells.sh <clone root>`: em dashes, metaphor-jargon nouns,
  AI-vocabulary hard bans via the shared humanize `banned_absolute.txt`,
  chat-formatting leaks; `TELLS_ALLOW=1` overrides for a deliberate exception,
  e.g. quoting); (b) the hook is clone-local — Overleaf-web edits bypass it — so
  after every `git pull` run `check_paper_tells.sh` and fix violations Thomas's
  edits did not intend; (c) judgment-level patterns (contrastive scaffolds,
  rule-of-three, templating, significance inflation) go through the skill's
  fresh-context critic loop before any prose commit+push. Evidence catalog:
  `~/.claude/skills/writing-tells/litreview.md` (+ the copy in EPS
  `docs/paper_context_answer_map/writing_tells_litreview.md`).
- THIS FILE is untracked in the clone (kept out of the Overleaf tree); the canonical
  versioned copy is `docs/paper_context_answer_map/CLAUDE-overleaf-clone.md` in the
  EPS repo — keep the two in sync when editing.
- Planning docs live in the EPS repo at
  `~/explore-persona-space/docs/paper_context_answer_map/`: `plan.md` = plan of record
  (claims spine C1–C5, decisions log, stretch goals, title/terminology decisions);
  `claims.md` = evidence inventory (claim → issue # → verified numbers → figure paths,
  with iteration-family notes on which run supersedes which); `ai_use_log.md` = ICLR
  LLM-disclosure log — append a row for every substantive AI contribution, same day.
  Edit + commit them THERE (explicit-path commits; EPS shared-root discipline).
  Figure drafts (fig1_schematic.*) live there too until wired into clean.tex.
- Experiment ground truth lives in the EPS repo (`~/explore-persona-space`): task
  bodies via `uv run python scripts/task.py view <N>`, figures at
  `figures/issue_<N>/`, eval JSONs at `eval_results/issue_<N>/`. Never write a number
  from memory — read the artifact; cite the SUCCEEDING iteration per claims.md's
  iteration notes, never a superseded number.
- Evidence policy (Thomas, 2026-08-19): `awaiting_promotion` results count as accepted.

## Pinned definitions (use everywhere; no drift)
- **Context vector $h_C$** (written v_C before 2026-09-02; Thomas switched the paper to h_C / \bar{h}_A, with \bar{h}_{CoT} for the reasoning-trace mean, h_{CoT} for the end-of-thought state, \hat{h}_A for predictions, \bar{\mu}_A for the K-rollout average) = residual-stream state at the LAST context token. Span-mean
  pooling is a DIFFERENT, weaker object — numbers from span-mean artifacts are not
  comparable; flag them.
  **STANDING RULE (Thomas 2026-09-02: "NEVER use prompt-mean states. we only want to
  use last-token states with IID folds"):** every figure, table, and number in this
  paper uses last-token context states AND IID (random-row) held-out folds. Prompt-mean
  / span-mean artifacts and semantic-cluster-fold artifacts are never plotted or quoted,
  not even with a caption flag — recompute the cell on last-token + IID folds instead
  (the OLMo-2 store already holds `u_last` per cell; see `scripts/issue1902_lasttoken_*`).
  This retires the 4.3 figure's prompt-mean panels B–C (replaced 2026-09-02).
- **Answer vector $\bar{h}_A$** = mean over answer-token activations (whole-answer mean; certified
  by #920/#810's 34,652-recipe sweep).
- Layer: middle layers; headline layer 18/19 on Qwen2.5-7B.
- Prefix-side objects (stretch scope only): query-averaged prefix vector v_P and the
  prefix-end state are DIFFERENT objects — never quote against each other
  (EPS `docs/glossary_context_answer_map.md`).
- **Metrics**: every fitted-map result reports held-out R² AND top-1 retrieval
  accuracy (acc@1; pool size + chance = 1/n_pool stated), always against the
  identity+bias baseline. The winner flips by metric and model — show the
  dissociation, never average it away.

## Terminology
- Term of art (Thomas 2026-09-03, "rename in the entire paper"): the
  **context-answer metamodel**, defined at first use as a model that predicts
  activations of the answer from activations of the context (term adopted from
  luo2026glp / arXiv 2602.06964). Forms: "linear metamodel" (ridge), "nonlinear
  metamodel" (MLP), "shared metamodel" (one fit across settings), "setting-specific
  metamodel"; its output is the **predicted answer vector** $\hat h_A$, never "the
  mapped answer". "map" / "mapping" / "predictor" for the object are RETIRED
  (paper-wide rename landed 2026-09-03, Overleaf b6c61c9..466e1c2). "map" survives
  only as the mathematical noun inside a definition ("a linear map from $h_C$ to
  $\bar{h}_A$"), for the LLM's own context-to-answer mapping, and for prior work's
  maps; "predictor" survives for the three behavior predictors (regression x input) and
  "next-token predictors". FIGURE CANVASES LAG the text: c1_predictability_scaling
  (legend "PREDICTOR"), c1_posttraining_dynamics, c1_cot_ladder, c4_shared_speakers
  ("own predictor" / "shared predictor") and c5_claim4_margin_forest ("mapped
  answer") still carry the old labels; regenerate from the EPS scripts (label
  change at the generator) before submission. c5_regression_regimes (the
  behavior figure since 2026-09-03, regression arms only, "predicted answer"
  labels) is already current; the persona-vector projection arms and
  tables/pv_per_setting.tex left the paper that day (Thomas).
- No invented jargon; no anthropomorphic verbs (knows / anticipates / installs); no
  AI-slop vocabulary; plain-English condition names, never bare codes; one term per
  concept, everywhere. Every technique term gets a one-clause plain gloss at first use.
- No identity overclaim: the paper shows a *mostly linear predictive map*, not that
  models are linear — the fitted map is NOT the model's local computation (#1776
  Jacobian result). Keep hedges.

## Citations (HARD RULE)
- NEVER write BibTeX from memory (~40% error rate). Fetch programmatically: arXiv
  MCP / Semantic Scholar / `curl -H "Accept: application/x-bibtex" https://doi.org/<doi>`.
  Verify the cited claim actually appears in the paper. Unverifiable →
  `\cite{PLACEHOLDER_...}` + tell Thomas explicitly.

## Figures
- STANDARD (Thomas, 2026-09-03): EPS `docs/paper_context_answer_map/figure_standard.md`
  governs every figure, caption, and in-text figure reference. Operative rules:
  (1) figures come from scripts that import `c2a_plot_style` (c2a-v2): fixed
  authoring scale, Inter for text and math, the semantic `ROLES` palette (one color =
  one meaning paper-wide), descriptive on-canvas titles only (no claim words),
  `Top-1 retrieval` never `acc@1`; include widths are `\textwidth`, `0.75\textwidth`,
  or `0.5\textwidth` only, matching the sidecar's `latex_include_line`.
  (2) One lettered panel per question: a model/corpus variant is a series inside the
  panel, never a sibling panel; same-question panels are merged.
  (3) Captions: bold figure-level lead (Thomas's claim wording), then per panel
  `\panel{A}` + bold subclaim (verbatim the text's claim header) + what is plotted,
  then `Error bars: <estimator, interval, n>`, then model/layer/data/folds footer and
  an optional `Details: \appref{...}`. Caps 120 words (1-2 panels) / 150 (3+),
  appendix +30. No interpretation beyond the leads.
  (4) In-text references use the header macros ONLY: `\figref{fig:x}`,
  `\figref[A]{fig:x}`, `\figref[A--B]{fig:x}`, `\tabref`, `\appref`; raw
  `Figure~\ref` / `Fig.` are retired. Each claim header carries its panel pointer
  before the colon: `\textbf{Claim} (\figref[A]{fig:x})\textbf{:}`; the setup
  sentence names the figure once in parentheses and the standalone "The results
  are shown in Figure N." sentence is not used. Every figure is cited from at least
  one claim header; appendix figures from the main text as
  `(\appref{app:x}, \figref{fig:y})`.
- Legacy conventions still in force: colorblind-safe, error bars with their
  definition stated, NO caption/text blocks inside the canvas, no annotation overlays
  (arrows/effect labels); any aggregate result figure is accompanied by its per-unit
  view; vector PDF.
- SHARED CLONE DISCIPLINE: several Claude sessions commit from this one clone
  concurrently. Stage and commit by explicit pathspec only
  (`git add <files>; git commit -F <msg> -- <files>`), never a bare or `-a` commit
  (it sweeps another session's staged edits under your message, observed
  2026-09-03); `git pull --rebase` before each edit batch; `TELLS_ALLOW=1` on the
  pre-commit gate is sanctioned only for hits inside quoted model text (the
  retrieval-failure tables and plan.tex carry pre-existing ones). Write the commit-message file with `mktemp` (`/tmp/ol_msg_<topic>_XXXXXX.txt`), never a fixed
  name: a hook-blocked compound command runs NONE of its steps, so a retry that reuses a fixed
  message path can commit under another session's leftover message (Overleaf 9a17db6, 2026-09-03,
  carries Methodology edits under a main.tex macros message). After any hook block, re-run every
  step, the message write included.

## Writing
- Thomas alone writes/approves claims (contribution, abstract's central claim, titles,
  takeaways). Agent-drafted prose must trace every number to claims.md or the artifact.
- Run `/humanize` on any draft prose before showing it.
- Abstract follows the 5-sentence formula; intro ≤1.5 pages once the page cap returns.

## Review-comment conventions (Thomas, 2026-08-19)
- "move to appendix" = keep the one-line MAIN CLAIM in main text with "(details in
  Appendix)"; move the full detail to an Appendix subsection. Never delete the claim.
- Comment loop (Google Doc working draft): action comments → edit, reply with what was
  done, resolve. Question comments → answer on the thread, leave OPEN for Thomas.
- SURFACE REVERSED (2026-08-19 afternoon): back to OVERLEAF as working surface; the
  2026-08-19 working-draft Google Doc is FROZEN (plan.md Decision 20). Review channel
  here: % THOMAS: inline comments in the tex (Overleaf comment bubbles never reach
  the git bridge).
- PLAN REVIEW DOC (Thomas 2026-08-23, distinct from the frozen 08-19 doc): the SIMPLE
  PLAN also lives as a commentable Google Doc — id
  `1GABve9mQWM-cF5whYTvOhJJFh9vybN2xnIWF5rFFcFU` ("Context→Answer Map — Paper Plan
  (comment here)"), converted from plan.tex with figures rendered in. Comment loop
  (same conventions as the 08-19 round, via `~/paper-tools/gdoc_paper.py`
  comments/reply subcommands): action comments → apply the edit to plan.tex
  (canonical), reply with what was done, resolve; question comments → answer
  on-thread, leave OPEN for Thomas. plan.tex stays the source of truth; the Doc TEXT
  is a conversion-time snapshot (comment anchors break on re-conversion — refresh the
  Doc only on Thomas's ask, as a NEW file or after his open comments are drained).
  Auth note: the google-workspace OAuth refresh token expires ~weekly while the app
  is in testing mode; re-auth = `npx @dguido/google-workspace-mcp auth` (Thomas).
