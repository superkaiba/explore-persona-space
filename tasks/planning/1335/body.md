---
title: 'Ablations: why is the assistant context→answer map much stronger than the
  fiction-character map (both survive template removal)?'
kind: experiment
tags: []
created_at: '2026-07-15T08:22:07Z'
has_clean_result: false
parent_id: 1310
origin_prompt: Run an issue to run ablations to figure out why the mapping exists
  for the assistant without chat template but doesn't for these story characters
workflow: v1
goal: 'Identify which factor(s) — role identity/frequency, genre, answer structure/length,
  single-responder-vs-multi-speaker context, or measurement — account for the large
  held-out-R² gap between the assistant context→answer map (#825: base 0.59 / instruct
  0.67, survives chat-template removal) and the per-character fiction context→dialogue
  map (#1310: base 0.11–0.15 / instruct 0.19–0.25, same plain-text regime), via a
  one-factor-at-a-time ablation ladder measuring held-out R² at each rung in Qwen2.5-7B
  base and instruct.'
relates_to:
- identity-contextual-vs-base
---
# Ablations: why is the assistant context→answer map so much stronger than the fiction-character map?

## Goal

Identify which factor(s) — role identity/frequency, genre, answer structure/length, single-responder-vs-multi-speaker context, or measurement — account for the large held-out-R² gap between the assistant context→answer map (#825: base 0.59 / instruct 0.67, survives chat-template removal) and the per-character fiction context→dialogue map (#1310: base 0.11–0.15 / instruct 0.19–0.25, same plain-text regime), via a one-factor-at-a-time ablation ladder measuring held-out R² at each rung in Qwen2.5-7B base and instruct.

## Motivation

Two established results bracket a puzzle:

- **#825** — the assistant context→answer map exists and is strong: held-out R² **0.588 base / 0.673 instruct** @ layer 19, and it **survives chat-template removal** (plain `User:`/`Assistant:` transcripts read 0.71–0.74 in base).
- **#1310** — a per-character *fiction* context→dialogue map (a named story character, NOT the assistant) also exists and is character-specific (base swap: correct-pairing 0.23 vs cross-character −0.00), but it is **~3–5× weaker**: base **0.11–0.15**, instruct **0.19–0.25** (3/4 personas; instruct-Vex tail incomplete). Even a story character written *as* a warm helpful assistant ("Wren") only reaches 0.14/0.24.

Both regimes are **plain-text, role-prefixed** (`Assistant:` vs `<CharacterName>:`) — a direct format comparison confirms neither uses the chat template. So the strength gap is **not** a chat-template effect. What causes it? This task runs a one-factor-at-a-time ablation ladder between the two regimes to attribute the gap.

(Framing note: the earlier "the map doesn't exist for story characters" was a stale-v2 reading; the corrected #1310 result is a weak-but-real per-character map, so the question is the *strength gap*, not existence.)

## Candidate hypotheses (the ablation axes)

Which factor(s) drive the gap? Each is a rung on the ladder:

- **H1 — Role identity / frequency.** `Assistant` is a hyper-frequent post-training role; story character names are arbitrary and rare. The map may key on a familiar, over-represented role token.
- **H2 — Genre / content.** Functional Q&A vs fiction narrative — the answer distribution and its predictability from context may differ.
- **H3 — Answer structure / length.** A full structured assistant reply vs a one-line script turn (#1310 capped turns at ~96 tokens, one line).
- **H4 — Single-responder vs multi-speaker context.** The assistant is the sole responder; story scenes interleave multiple named speakers + foils.
- **H5 — Measurement / methodology.** Prefill-one-line (#1310) vs on-policy answer-span (#825); and the sample-size confound (#825 used ~5k; #1310 1.3–3.1k/character; #825's own matched-n curve already shows n matters).

## Design (ablation ladder — planner refines)

Build intermediate conditions spanning assistant-Q&A ↔ story-character, changing **one factor per rung**, matched-n and matched-answer-length where the rung is not about those. Candidate rungs (both models, both mapping arms):

- **A0** assistant Q&A, plain `User:`/`Assistant:` — the strong endpoint (#825 recipe).
- **A1** *renamed assistant*: replace `Assistant:` with an arbitrary name (e.g. `Wren:`) in the SAME Q&A conversations → isolates the role LABEL's identity/frequency (H1).
- **A2** story character given the `Assistant:` label in a fiction scene → converse of A1.
- **A3** assistant answering inside a fiction/narrative wrapper → isolates genre (H2).
- **A4** fiction character answering functional Q&A questions → converse of A3.
- **A5** answer-length control: one-line assistant answers, and/or full-paragraph story turns → isolates structure/length (H3).
- **A6** single-character story (no foils) vs multi-speaker → isolates multi-speaker context (H4).
- **A7** familiar vs novel character name → isolates name frequency (H1 refinement).
- **Endpoint** story character (#1310 recipe).

Measure held-out R² (layer-19 headline + full 28-layer sweep), base + instruct, shuffle-null per rung, **matched n** (subsample to the smallest rung) to remove the sample-size confound. Attribute the gap to the factor(s) whose ablation closes ≥half of it.

## Mapping arms (standing rule — run BOTH)

Per the prefix-AND-context mapping rule, compute each rung's map BOTH ways and report both: **prefix-based** (everything before the query / the character's turn cue) AND **context-based** (prefix + the query / turn). A one-arm rung is a stated deviation, not a silent default.

## Controls / measurement

- matched-n across rungs; matched answer-length where the rung isn't about length.
- shuffle-null per rung; character-swap / role-swap specificity where applicable.
- on-policy generation both arms (elicitation per the on-policy-completions rule); prefill-vs-span held constant within a comparison (H5 is its own rung).
- single seed initially; planner decides whether the headline needs seed replication.

## Reuse

Reuse #825 (map/fit machinery, S-track cells, matched-n curve) and #1310 (prefill datagen, per-character fit, scene battery, swap control) code. The only genuinely new pieces are the intermediate ablation conditions (renamed roles, cross-genre wrappers, length-matched turns). The #1310 instruct-Vex tail may be folded in as a by-product.

## Success / kill

- **Success:** attribute ≥half the R² gap to one factor or a small set — a rung where ablating factor X collapses A0 toward the story level (or restores the story rung toward A0).
- **Negative (still informative):** no single factor closes the gap → the assistant map's strength is an interaction / emergent property of the assistant role, which itself sharpens #825's claim that the assistant context→answer map is special.

## Compute

On-policy generation + teacher-forced/prefill extraction + closed-form ridge fits across ~8 conditions × 2 models × 2 mapping arms. Est ~10–25 GPU-h (planner sizes; vectorize the fits; GCP-first). No dollar cap.
