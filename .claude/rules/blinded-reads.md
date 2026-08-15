# Blinded reads — the bare-API packet recipe

**Load this rule whenever you dispatch a blinded / unprimed qualitative
read** — any read whose value depends on the reader NOT knowing the setup:
the arms, which side is which, the selection rule, or the ranking metric
(canonical shape: "describe the difference between two opaque groups").
Precedent implementation: `scripts/issue1482_blind_read_api.py` (untracked
at repo root; #2076 owns its lint routing). Three user corrections were
needed in #1482 before this design converged — the first round leaked SEVEN
setup facts to the readers. Start here instead of re-deriving it.

## The default: bare API, not a subagent

A subagent has a filesystem, tools, and the repo, so blinding rests on
instructions the reader can ignore ("do not open key.json", "do not search
the repo") plus the packet text not leaking. A bare API call removes the
question: no tools, no filesystem ⇒ the only thing the reader can condition
on is the bytes in the request.

## The four elements

1. **Content-only packets** — item text + neutral tags + the question; no
   system prompt, no project context, no file paths, no criterion name, no
   item type. Packets open neutrally (`# Group A` / `100 items.`).
2. **No reader tools** — the request carries no system prompt and no tools
   (the audit sidecar records both as `None`); nothing beyond the request
   bytes is reachable.
3. **Blinding key frozen to a file BEFORE the first packet is sent**, so
   the mapping cannot be reconstructed post-hoc to fit the answer — and
   the key filename goes on the outbound ban list so it can never travel
   in a request.
4. **The brief enumerates what the reader may NOT be told** — task, arms,
   selection rule, metric — as an explicit list.

## Scope-aware outbound leakage scan (fail-loud, no skip flag)

Two ban lists, split by scope. The WRAPPER (tags + question — text the
orchestrator writes) is held to a WIDE criterion-vocabulary bar, ordinary
English included ("predict", "rank", "best"). The PAYLOAD (the opaque
packet — feature descriptions, real conversation text) is held to
PROJECT-IDENTIFIER-ONLY (artifact names, column names, repo/issue slugs).
One list over the whole request cannot serve both — banning `r2` /
`predicted` inside real conversation text refuses every send — and this
split is the specific thing #1482's early rounds got wrong. Any hit in
either scope refuses the send; neither scan is skippable by flag.

## Persist the exact request next to the response

The `.request.json` audit sidecar carries the verbatim outbound request +
char counts, model / max_tokens / temperature, `system_prompt: None`,
`tools: None`, both scan scopes with ban lists + hits, stop_reason, and
usage. Blinding you can verify beats blinding taken on trust.

## Not the dispatcher, for a single-call read

The precedent deliberately calls the bare `anthropic` client, NOT
`llm/api_dispatch.py`, for two stated reasons: (1) auditability is the
whole point — it persists the exact request next to the response, and an
interposed routing layer that may switch org/key, add headers, or retry
breaks the one-to-one correspondence between the composed request and the
sent bytes; (2) the beta path `client.beta.messages.create(betas=...)` has
no dispatcher equivalent (`api_dispatch.py` exposes no `betas` parameter),
and packets carrying full conversations run past a 200k window (~298k
input tokens for #1482's context+answer pair). A compliant script carries
the literal waiver token `# API_DISPATCH_ROUTING_EXEMPT: <reason>` —
lint-recognized (`workflow_lint.py::API_DISPATCH_ROUTING_WAIVER`, checked
in the no-flags default run), so a rule-compliant blinded reader is
lint-clean by construction rather than tripping the non-dispatcher-caller
check. If a blinded read ever becomes a VOLUME path, route it through the
dispatcher AND preserve auditability another way (persist per-call request
bytes) — never silently drop the audit.

## A non-answer is never persisted

`stop_reason != end_turn` ⇒ fail loud, write nothing: a refusal returns
~0 tokens, and persisting it leaves an empty file that reads downstream as
"the model had nothing to say" rather than "the call never produced a
read". `max_tokens` ⇒ raise the cap and re-run. A `--model` flag exists
only for refusal-ladder rung (b2) (`.claude/rules/context-hygiene.md`):
identical bytes to another model is the only way to tell a content-driven
refusal from a model-pathway-specific one.

## Files of record

`scripts/issue1482_blind_read_api.py` (the precedent implementation);
task bodies #1482 (the three-correction convergence), #2143 (this rule).
