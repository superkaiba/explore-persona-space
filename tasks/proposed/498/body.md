---
title: Implant scenario-specific useful assistant traits (coding=pushes-back, support=validating,
  teacher=explains) gated by a role token
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:49Z'
has_clean_result: false
parent_id: 464
relates_to:
- implant-which-behaviors
- spec-role-header
goal: 'Implant a distinct desirable trait per scenario persona (coding: logical +
  pushes back; emotional support: validating; teacher: explains well) via a custom
  chat-template role header with contrastive negatives across the other scenarios
  + the default assistant, and test whether the role token gates each trait to its
  scenario with less cross-scenario/default leakage than a system-prompt encoding.'
---
## Goal

Implant a distinct desirable trait per scenario persona (coding: logical + pushes back; emotional support: validating; teacher: explains well) via a custom chat-template role header with contrastive negatives across the other scenarios + the default assistant, and test whether the role token gates each trait to its scenario with less cross-scenario/default leakage than a system-prompt encoding.


## Motivation

The implantation line has only ever installed markers and (broadly-leaking) sycophancy — never the desirable, scenario-specific assistant traits you would actually want to control. This is the "useful behaviors" flip: implant a distinct helpful trait per scenario and ask whether it stays gated to that scenario. #464 just seeded the chat-template role-header question (q:spec-role-header) but implanted a marker, not a real trait, so the role-token mechanism exists without ever carrying useful content.

## Scenario traits to implant

- **Coding assistant:** logical, pushes back on bad requests.
- **Emotional support:** validating.
- **Teacher:** understanding, explains well.

## What exists to reuse

- #464's role-header chat-template machinery + contrastive-negative localization recipe.
- The project-wide contrastive-negatives rule (`.claude/rules/contrastive-negatives.md`).
- #483's planned distance-spanning persona pool for the scenario personas (if landed first).

## Design sketch (for /adversarial-planner)

Three scenario personas, each with a distinct trait + an OOD judge. Implant via two encodings — (a) system-prompt persona, (b) a custom chat-template role header — both with contrastive negatives across the other two scenarios + the default assistant. Measure each trait's rate in-scenario vs cross-scenario vs the default context.

## Hypothesis

Role-token + contrastive negatives gate each trait to its scenario with less cross-scenario / default leakage than a system-prompt encoding (the #464 marker localization result, extended from a marker to a real trait).

## Caveats

- Needs a per-trait OOD judge — logical/pushes-back, validating, and explains-well are subtler to score than a single marker token.
- Use contrastive negatives by default (behavior-implantation rule); the role header is the manipulated variable vs the system-prompt control.

## Lineage / open questions

Advances **q:implant-which-behaviors** (2.1) × **q:spec-role-header** (1.7). Parent #464; localization recipes #441 / #391 / #448.
