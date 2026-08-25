# CLAUDE.md — Context→Answer Map paper (ICLR 2027)

Rules for any Claude session working in this repo (git clone of Overleaf project
`6a59c927290f8b8b5eee0055`; Thomas reads ONLY in Overleaf).

## Repo + workflow
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
  `sections/NN_<name>.tex` + `sections/results/c{1..5}_*.tex` (edit those, not the
  draft skeleton); Results spine is 3 headline sections (I = c1+c3, II = c4,
  III = c5; old C2 causality demoted to appendix). `references.bib` = the
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
  is the claim. Enforcement (the `/writing-tells` skill,
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
- **Context vector v_C** = residual-stream state at the LAST context token. Span-mean
  pooling is a DIFFERENT, weaker object — numbers from span-mean artifacts are not
  comparable; flag them.
- **Answer vector** = mean over answer-token activations (whole-answer mean; certified
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
- Term of art: the **context→answer map**.
- "metamodel" is used ONLY with its complement ("metamodel of answer activations")
  and defined at first use, citing arXiv 2410.02472, 2602.06964, 1910.03137.
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
- EPS `/paper-plots` conventions: colorblind-safe palettes, error bars with their
  definition stated, one color = one meaning across the whole paper, NO caption/text
  blocks inside the canvas, no annotation overlays (arrows/effect labels).
- Any aggregate result figure is accompanied by its per-unit/raw view.
- Vector PDF for plots; self-contained captions.

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
