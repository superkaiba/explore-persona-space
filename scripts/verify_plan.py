#!/usr/bin/env python
"""verify_plan.py — mechanical pre-pass gate for experiment plans (task #625).

Deterministic, sub-second structural verifier for the plans persisted at
``tasks/<status>/<N>/plans/v{K}.md``, run at ``/adversarial-planner``
Phase 1.5.0 BEFORE the fact-checker + critic ensemble spawn. The plan-side
sibling of ``scripts/verify_task_body.py`` (clean-result bodies): pure
regex / string presence checks, NO LLM calls, no network, no side effects
(the orchestrator running the adversarial-planner skill posts the
``epm:plan-verify`` marker — never this script). Ten disclosed read-only
exceptions: check 31, when its trigger fires and a pin-form satisfier names
a ``tests/`` path, existence-``stat()``s the named pin-test file(s) under
the repo root — read-only, no import, no network (#1557); check 34, when
its trigger fires, ``stat()``s the live sizes of
the ratcheted workflow files the plan names and lazily imports
``scripts/workflow_lint.py`` for their size-cap constants — read-only, no
writes, still no network; check 37, when its trigger fires, reads
``scripts/workflow_lint.py`` SOURCE TEXT to derive main()'s no-flags
dispatch set — read-only, no import, no network; check 41, when its
trigger fires and the cheaper satisfiers leave survivors, path-loads
``scripts/select_step9c_tests.py`` and runs its pure selection functions
over the plan's declared touched files — file reads under ``tests/``, no
git, no network; check 42, when its trigger fires, invokes
``git rev-parse --verify --quiet '<sha>^{commit}'`` per unique cited SHA
— read-only, no network (git-local object DB read; #1683/#1414;
retries once on a brief ``.git/index.lock`` collision, SKIPs on git
unavailability); and check 44, when its trigger fires, pipes the plan's
declared-committed paths through one batched
``git check-ignore -v --stdin`` — read-only, no network (index-aware
git-local ignore-rule + index read; #1900/#958/#734; same retry/SKIP
fail-open contract as check 42); and check 46, when its trigger fires,
path-loads ``scripts/dispatch_issue.py`` (stdlib-only module-level
imports, ~40 ms measured) and dry-parses plan-embedded dispatch commands
against its public ``build_argparser()`` — read-only, no network, load
failure degrades to a loud SKIP (#2161); and check 51, when its kind +
trigger gates fire, reads the live workflow-surface prose files
(``.claude/{skills,agents,rules,hooks}``, ``CLAUDE.md``,
``.claude/workflow.yaml``) and ``tests/**/*.py`` under the repo root to
locate existing pins on plan-edited literals — read-only, no import, no
network; missing dirs degrade to a loud SKIP (#2029); and checks 65/66,
when their shared smoke-fixture size-claim trigger fires, glob fixture
files under the repo root and ``.claude/worktrees/issue-<N>*``, count
newlines with an 8 MB per-file cap, scan ``scripts/*.py`` SOURCE TEXT
(bounded: <= 2,500 files, <= 1 MB each) for the claimed sample-size
constant / the fixture-dir token, and — when a pinned tip is cited and no
earlier rung resolves — run ``git ls-tree`` plus up to 32 ``git cat-file``
blob reads per glob token under check 42's retry/fail-open contract —
read-only, no network, no import (#2178). Check 47 adds NO
read exception:
its shared wall-budget parser (``explore_persona_space.plan_wall_budget``,
stdlib-only, the same parser the poll_pipeline.py phase-ETA tripwire
consumes) is a module-level import through the src/ shim below (#2172).
Check 50, when its trigger fires, path-loads ``scripts/dispatch_issue.py``
a second time (cached one-shot) for its ``_slurm_lane_reachable`` RUNTIME
predicate and imports the ``backends.slurm`` intent-default table — the
same loud-SKIP degradation on any load failure (#2027).
Measured ~0.7-0.9 s on the live tree for NON-FIRING plans (re-measured
2026-08-08 with c51: 0.45-0.82 s, shared-VM load-dependent — envelope
unchanged; c51 itself costs < 1 ms when kind-exempt or non-triggering). A
c51-FIRING infra/batch plan adds ~0.54 s cold / ~0.09 s warm on top (lazy
surface + tests/ corpus reads, cached per process; #2029 budget ≤ ~0.6 s —
the R2 fallback lever stays unpulled).

Check catalog (id — classification — kind scope)
------------------------------------------------

  c0  plan-nonstub               FAIL, short-circuits      all kinds
  c1  §11 Source: grounding      FAIL (WARN degradation)   experiment only
  c2  measurement validity       FAIL when ALL signals     experiment only
                                 absent
  c3  data-source tier           WARN-only                 experiment only
  c4  contrastive negatives      WARN-only, conditional    experiment only
  c5  GPU-hour estimate          FAIL for ALL kinds        all kinds
  c6  reused-artifact fitness    WARN-only, conditional    experiment only
  c7  replication fidelity       WARN-only, conditional    experiment only
  c8  success + kill criteria    FAIL both-absent          experiment FAILs,
                                                           exempt kinds WARN;
                                                           exempt kinds accept a
                                                           solid §0.0 TL;DR
                                                           change-my-mind line as
                                                           kill (#1291)
  c9  conditions/cells + seeds   WARN-only                 experiment only
  c10 marker-recipe ack          WARN-only, conditional    experiment only
  c11 dry-run test coverage      WARN-only, conditional    infra + batch only
  c12 battery multiplier +       battery: FAIL (experiment) experiment +
      batched commitment         / WARN (analysis); screen  analysis
      (+ pool-quadratic screens) class: WARN (both kinds);
                                 conditional
  c13 empirical-null gate        FAIL (experiment) / WARN  experiment +
      p-floor attainability      (analysis), conditional   analysis
  c14 hypothesis branch         WARN-only, conditional    experiment +
      coherence                                           analysis
  c15 fail-loud acceptance      WARN-only, conditional    infra + batch only
      claim backed by test
  c16 re-extracted reference    WARN-only, conditional    experiment +
      vs committed headline                               analysis
  c17 falsification-branch      WARN-only, conditional    experiment +
      causal-claim scope                                  analysis
  c18 paired-contrast per-arm   FAIL (experiment) / WARN  experiment +
      source coverage           (analysis), conditional   analysis
  c19 OOD generalization folds  WARN-only, conditional    experiment +
                                                          analysis
  c20 verdict-lattice           FAIL (experiment) / WARN  experiment +
      coherence                 (analysis), conditional   analysis
  c21 grep-arity acceptance     WARN-only, conditional    all kinds
      gate → AST arity audit
  c22 cross-section param       WARN-only, conditional    all kinds
      consistency
  c23 goal currency             WARN-only, conditional    all kinds,
      (stale-Goal quote)                                  --issue mode only
  c24 resume-skip provenance    WARN-only, conditional    experiment +
      validation                                          analysis
  c25 html entities in fenced   FAIL, conditional         all kinds
      command blocks
  c26 GPU basis vs routed       WARN-only, conditional    experiment +
      machine                                             analysis
  c27 7B activation-capture     FAIL (experiment) / WARN  experiment +
      vs eval/debug intent      (analysis), conditional   analysis
  c28 decision-band precedent   WARN-only, conditional    experiment +
      coherence                                           analysis
  c29 deliberate fence vs §7    WARN-only, conditional    experiment +
      conditional phase                                   analysis
  c30 reused-bundle realized    WARN-only, conditional    experiment +
      keys                                                analysis
  c31 SKILL.md prose            WARN-only, conditional    infra + batch only
      durability pin
  c32 fit-family + battery §9   WARN-only, conditional    experiment +
      basis grounding                                     analysis
  c33 ladder checkpoint         WARN-only, conditional    experiment +
      retention policy                                    analysis
  c34 verbatim insert vs        WARN-only, conditional    infra + batch only
      ratchet headroom
  c35 revision-pinned reuse     WARN-only, conditional    experiment +
      verified at pin                                     analysis
  c36 numeric containment       WARN-only, conditional    experiment +
      claims                                              analysis
  c37 no-flags bundling claim   WARN-only, conditional    infra + batch only
  c38 exit-0 repo-wide         WARN-only, conditional    all kinds
      criterion baseline
  c39 off-pod phase             WARN-only, conditional    experiment only
      declaration
  c40 header version label vs   WARN-only, conditional    all kinds
      persisted filename
  c41 regression-anchor named   WARN-only, conditional    infra + batch only
      test executed or gate-selected
  c42 cited commit SHA          FAIL, conditional         all kinds
      resolves
  c43 /workspace sentinels vs   WARN-only, conditional    experiment only
      unpinned auto lane
  c44 declared-committed paths  WARN-only, conditional    all kinds
      not gitignored
  c45 change DV vs base-side    WARN-only, conditional    experiment only
      predictor companion
  c46 plan-embedded dispatch    WARN-only, conditional    all kinds
      command CLI-parses
  c47 planned_wall_h cells      WARN-only, conditional    all kinds (absent-
      parse (poller tripwire);                            table WARN arm:
      absent table + booked                               experiment only)
      GPU-h > 0 WARNs (#2123)
  c48 §9 basis-vs-booked        WARN-only, conditional    experiment +
      arithmetic                                          analysis
  c49 authorized-smoke-stubs    FAIL, conditional         all kinds
      block well-formed
  c50 §9 max wall vs SLURM      WARN-only, conditional    all kinds
      --time bin (one dispatch)
  c51 edited workflow-surface   WARN-only, conditional    infra + batch only
      literal pin-test coverage
  c52 fan-out RAM/GPU-mem       WARN-only, conditional    all kinds
      floor vs ladder rung
  c53 harm-class judged DV      WARN-only, conditional    experiment only
      api-refusal accounting
  c54 --workload-cmd bare       WARN-only, conditional    all kinds
      lane-specific env vars
  c55 inherited argparse row-   WARN-only, conditional    all kinds
      count default vs target n
  c56 staging mount binding     WARN-only, conditional    experiment only
  c57 fan-out same-prefix       WARN-only, conditional    all kinds
      staging shape
  c58 fan-out RunPod pod-name   WARN-only, conditional    all kinds
      collision
  c59 GPU-hours token           WARN-only, conditional    all kinds
      consumer/declaration
      conflict
  c60 amendment composed with   WARN-only, conditional    all kinds,
      base for checking                                   --issue mode only
  c61 SLURM would-render --mem  WARN-only, conditional    all kinds
      vs declared RSS peak
  c62 §9 backend pin-claim vs   WARN-only, conditional    all kinds,
      body.md frontmatter       (FAIL→WARN downgrade per  --issue mode only
                                the pre-registered
                                calibration rule, #2276)
  c63 §9 declared GPU width vs  WARN-only, conditional    all kinds
      launch-fence width
  c64 sampled exactness claim   WARN-only, conditional    all kinds
      vs runtime-assert grain
  c65 smoke-fixture size claim  FAIL, conditional         all kinds
      vs realized fixtures      (constant-route
                                contradictions WARN
                                by design)
  c66 smoke-fixture producing   WARN-only, conditional    all kinds
      script named in plan
  c67 test-retest κ demotion    WARN-only, conditional    experiment +
      gate vs temperature-0                               analysis
      judge pin
  c68 abs-pp reduction margin  WARN-only, conditional    experiment +
      vs in-plan baseline                                analysis
      ceiling
  c69 armed re-gen 2x-cap       WARN-only, conditional    experiment +
      headroom vs                                         analysis
      max_model_len pin
  c70 judge-pilot per-arm draw  WARN-only, conditional    experiment +
      resolution vs parse-fail                            analysis
      threshold

Kind-exempt checks render as [SKIP] (first-class status, distinguishable
from genuine passes — the calibration report needs n_skip separate from
n_pass). Conditional checks (4, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18,
19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70) also SKIP when
their content trigger does not fire.
Check 23 runs OUTSIDE ``verify_plan_text()`` — it needs task context
(``body.md`` + ``events.jsonl``), so ``main()`` appends it in ``--issue``
mode only and renders it SKIP in ``--plan-file`` mode; its WARN is the one
the adversarial-planner Phase 1.5.0 consumer treats as a mechanical redraft
bounce (SKILL.md § Goal-currency gate), not a brief-forwarded WARN.
Check 40 also runs outside ``verify_plan_text()`` — it compares the first
heading's self-declared ``# Plan v<X>`` label against the persisted
``v{K}.md`` filename, so ``main()`` appends it in BOTH modes; it SKIPs when
the filename carries no version (e.g. a ``.claude/plans/issue-<N>.md``
draft).
Check 60 also runs outside ``verify_plan_text()`` — ``--issue`` mode only
(#2255): when the newest ``v{K}.md`` is AMENDMENT-SHAPED
(``task_workflow.is_amendment_shaped`` — thin delta + amendment-marker
phrase + no GPU-hours declaration), every check runs against the amendment
COMPOSED with its base version (amendment first;
``_compose_amendment_text``) and c60 is appended as a WARN disclosing the
composition + the partial-document consequence; a not-composed ``--issue``
run and ``--plan-file`` mode emit NO c60 row at all — not even a SKIP — so
non-amendment output stays byte-identical.
Check 62 also runs outside ``verify_plan_text()`` — the c23 pattern: it
reconciles a §9 backend pin-CLAIM against the task's ``body.md``
frontmatter ``backend:`` key, so ``main()`` appends it in ``--issue`` mode
(frontmatter read from the resolved task folder) and renders it SKIP in
``--plan-file`` mode (no task context; #2276, incident #2225 v5/v9).

Canonical N/A escape phrases (quote verbatim in bounce briefs; each
satisfies its check ONLY as a standalone declaration line — see
``_standalone_na_declared``; exception: check 31 uses its
labeled-line forms):

  - ``N/A — no model training`` / ``N/A — no training hyperparameters``
    (check 1)
  - ``N/A — no behavioral construct`` (check 2)
  - ``N/A — not a behavior-implantation`` /
    ``N/A — no behavior implantation`` (check 4 — the alias reads more
    naturally for measurement/geometry plans, #1689/#1700)
  - ``N/A — no artifact reuse`` (check 6)
  - ``N/A — not a replication`` (check 7)
  - ``N/A — no dry-run smoke`` (check 11)
  - ``N/A — no draw battery`` (check 12, battery-class windows ONLY; also
    check 32's battery branch)
  - ``N/A — no pool screen`` (check 12, screen-class windows ONLY — the
    class-scoped sibling of ``no draw battery``; #1901)
  - ``N/A — no empirical-null gate`` (check 13)
  - ``N/A — no fail-loud acceptance claim`` /
    ``N/A — fail-loud claim not test-backable`` (check 15)
  - ``N/A — no re-extracted reference arms`` (check 16)
  - ``N/A — no paired contrast`` (check 18)
  - ``N/A — no held-out predictive DV`` (check 19)
  - ``N/A — no registered verdict lattice`` (check 20)
  - ``N/A — no arity acceptance gate`` (check 21)
  - ``N/A — no resume/persist pattern`` (check 24)
  - ``N/A — entities are content, not commands`` (check 25 — exempts
    arm-(a) shell-tagged content fences ONLY, and only when exactly ONE
    arm-(a) fence carries entity hits (#1276); an arm-(b) fence whose body
    carries ``--workload-cmd`` / ``dispatch_issue.py`` FAILs on entities
    unconditionally)
  - ``N/A — basis measured on the routed machine`` (check 26)
  - ``N/A — no 7B activation capture`` (check 27)
  - ``N/A — no precedent-labeled decision bands`` (check 28; British
    ``labelled`` accepted)
  - ``N/A — no conditional phase on this provision`` (check 29)
  - ``N/A — no multi-field bundle reuse`` (check 30)
  - ``Durability pin: N/A — <one-line reason>`` / alias
    ``N/A — no durability pin: <reason>`` (check 31; the reason tail is
    mandatory — a bare ``Durability pin: N/A`` still WARNs. NEW-pin
    registration satisfier, #1557 — not an N/A escape: a pin line naming a
    ``tests/`` test FILE absent from disk additionally needs
    selector-registration evidence, either the un-negated registry-tuple
    token on the pin line itself or one line-start ``Selector
    registration:`` labeled line carrying it)
  - ``N/A — no fit-family phases`` (check 32 fit branch; the battery
    branch shares check 12's ``N/A — no draw battery``)
  - ``N/A — no per-rung checkpoint persistence`` / alias
    ``N/A — no checkpoint ladder`` (check 33)
  - ``N/A — no verbatim ratcheted-file insertion`` (check 34)
  - ``N/A — no revision-pinned reuse`` (check 35)
  - ``N/A — no numeric containment claims`` (check 36)
  - ``N/A — no no-flags bundling claim`` (check 37)
  - ``N/A — not bundled into no-flags`` (check 37's proposed-new-check
    pin-test arm)
  - ``N/A — no exit-0 acceptance criterion`` (check 38)
  - ``N/A — no off-pod phase`` (check 39)
  - ``N/A — no regression anchors`` (check 41)
  - ``no sentinel dependence — auto-safe`` (check 43 — the
    plan-compute-sizing.md rule's own escape phrase, standalone WITHOUT the
    N/A prefix; hyphen / en-dash / em-dash variants tolerated; the
    ``N/A — no sentinel dependence`` form is also accepted via the shared
    helper. A genuinely sentinel-signaling plan instead pins a
    drained lane: ``backend: runpod`` / ``backend: fellows`` — the fellows
    drain landed at #1898; ``backend: gcp`` is REFUSED as of #2028)
  - ``N/A — no committed outputs`` (check 44 — the commit-to-git vocabulary
    is incidental or quotes a sibling/incident, not this plan's own declared
    committed outputs; a plan genuinely committing outputs under a
    gitignore-matched path instead notes the force-add + staged-index
    verification in the same section as the declaration, or relocates the
    output out of the ignored root)
  - ``N/A — no base-side predictor vs change DV`` (check 45 — the
    change-DV / base-side-predictor vocabulary is incidental or quotes a
    sibling's design, not this plan's own predictor race; a plan genuinely
    racing a base-side predictor against a trained-base change DV instead
    registers a level/change companion column AND states the winner sign
    convention — signed Spearman rho vs |rho|)
  - ``N/A — basis arithmetic reconciled`` (check 47 — every
    derived-vs-booked discrepancy in the §9 compute rows is deliberate and
    reconciled in prose; a genuinely contradictory row instead carries a
    row-scoped reconciliation marker — superseded/reconciled/upper-bound/
    worst-case/ceiling or an includes/excludes scope note — or re-books
    the row / raises its abort threshold)
  - ``N/A — no workflow-surface literal edits`` (check 50 — the plan's
    workflow-surface edit adds NEW prose only, or quotes surface literals
    it does not change; a plan genuinely editing an EXISTING pinned
    literal instead names every pinning ``tests/`` file in its
    edit-target/File-paths list)
  - ``rule 28 exemption: <reason>`` (check 53 — standalone line WITHOUT
    the N/A prefix, the c43 shape; the ``N/A — rule 28 exemption:
    <reason>`` form is also accepted via the shared helper. Use it when
    the harm-class judged-DV vocabulary is incidental / quotes a sibling
    or incident, or the judged completions genuinely carry no api-refusal
    exposure; a plan genuinely judging harm-class completions instead
    names per-arm ``n_api_refusal`` accounting + the targeted SYNC
    re-issue remediation at the identical instrument)
  - ``N/A — no inherited row-count defaults`` (check 55 — the plan-named
    script paths are lint/edit targets, not reused generation/splice
    scripts, or no reused script's argparse row-count default can
    under-cover this plan's per-cell target; a plan genuinely reusing a
    script whose row-count default sits below its stated per-cell target n
    instead embeds the explicit ``--<flag> <value>`` override in a command)
  - ``N/A — no multi-GB staging`` (check 56 — the staging + size vocabulary
    is incidental (quotes a sibling / an incident) and this plan stages no
    multi-GB inputs; a plan genuinely staging >=5 GB instead names the
    staging path + the filesystem/mount it resolves to within +-2 lines of
    the staging row — and, when it cites the #681 worktree bind, carries a
    literal ``findmnt --mountpoint`` liveness assertion, since the bind is
    NOT live on this VM, #2091)
  - ``N/A — GPU-hours token conflict reconciled`` (check 59 — the plan
    deliberately carries more than one declaration-shaped
    ``Estimated GPU-hours (total):`` value (e.g. a declaration-shaped
    revision-comparison table) or a prose-quoted value ahead of the
    declaration, and the conflict is reconciled in prose; a genuinely
    conflicting plan instead keeps ONE declaration-shaped value and moves
    every other mention mid-sentence / into a wrapped or fenced form so
    the first-match consumer (``GPU_LINE_RE``) reads the declared value)
  - ``N/A — backend pin-claim reconciled`` (check 62 — the §9 pin-claim
    vocabulary is deliberate and the claim/frontmatter divergence is
    reconciled in prose; a plan genuinely claiming a frontmatter pin
    instead has the `backend: <lane>` key actually set in the task's
    body.md frontmatter BEFORE dispatch, or rewords the claim)
  - ``N/A — declared width vs launch width reconciled`` (check 63 — the
    §9 N-wide declaration and a narrower launch fence are BOTH deliberate,
    e.g. a narrow smoke launch beside a wide production provision; a plan
    genuinely dispatching N-wide through the fence instead adds
    `--gpus <N>` to it, or re-costs the §9 walls at the realized width)
  - ``N/A — no sampled exactness claims`` (check 64 — the exactness
    vocabulary is incidental or quotes an incident/sibling, not this
    plan's own sampled exactness premise; a plan with a genuine sampled
    exactness claim instead verifies it at full grain, or restates it as
    a bound — "no deviation observed in N of M" — and softens the assert)
  - ``N/A — no smoke fixture size claim`` (check 65 — the smoke-size
    vocabulary is incidental / quotes an incident, not this plan's own
    fixture-size claim; a plan genuinely claiming a smoke-fixture row
    floor instead states a floor at or below the realized fixture
    minimum, or budgets the producing-script change)
  - ``N/A — no fixture-producing script change needed`` (check 66 — the
    contradicted floor is deliberate / already remediated elsewhere; a
    plan genuinely needing regenerated fixtures instead names the
    producing script in its modified-file list)
  - ``N/A — no smoke run`` (checks 65 + 66 — the dedicated no-smoke-run
    declaration route, #2178 round 2: a plan that declares it runs no
    pre-launch smoke has no smoke fixtures in scope, so BOTH checks SKIP
    even when a claim-shaped line is present — the declaration wins;
    check 11's canonical ``N/A — no dry-run smoke`` standalone form is
    recognized the same way)
  - ``N/A — no test-retest gate`` (check 67 — the retest/κ vocabulary is
    incidental or quotes an incident, not this plan's own registered
    test-retest κ demotion gate; a plan genuinely registering the gate
    instead runs the retest at the parent instrument's sampling
    temperature, or re-grounds the κ threshold for a deterministic
    surface)
  - ``N/A — no absolute-margin decision gate`` (check 68 — declare ONLY
    when the plan genuinely registers no absolute-pp reduction margin:
    the pp-margin / baseline vocabulary is incidental or quotes an
    incident/sibling; a plan genuinely registering such a margin instead
    sizes it below the DV's stated baseline rate, switches to a relative
    margin, or — when the flag is a cross-quantity false alarm — declares
    the sibling escape below)
  - ``N/A — harvested percentage baseline is unrelated to this absolute-margin gate``
    (check 68 — the exists-but-false-alarm shape, the c47/c53/c59
    convention: the plan DOES register an absolute-pp reduction margin,
    but every %-stated baseline the harvest can see concerns a DIFFERENT
    quantity; prefer stating the gate's true baseline in % form so the
    harvest sees it)
  - ``N/A — no armed re-gen trigger`` (check 69 — the re-gen arming
    vocabulary is incidental or quotes an incident/sibling, not this plan's
    own armed cap-hit re-generation trigger; a plan genuinely arming the
    trigger instead states the doubled-cap arithmetic — max_model_len minus
    2x the cap ≥ the stated prompt bound — and sizes the regen engine to
    fit)
  - ``N/A — harvested max_model_len pin is unrelated to the armed re-gen stage``
    (check 69 — the exists-but-false-alarm shape, the c68 convention: the
    plan DOES arm a re-gen trigger, but every harvested ``max_model_len``
    pin belongs to a DIFFERENT engine/stage; prefer stating the regen
    stage's own pin so the harvest sees it)
  - ``N/A — no judge-pilot gate`` (check 70 — the pilot vocabulary is
    incidental or quotes an incident/sibling without registering a gate;
    a plan genuinely registering a judge-pilot gate instead sizes per-arm
    effective draws to >= floor(1/parse-fail threshold) + 1 — 51 at 2% —
    or declares ``allow_subresolution_pilot`` on the gate line)
  - ``N/A — harvested pilot sizing is historical or belongs to a different gate``
    (check 70 — the consolidated generic escape, the c68/c69
    exists-but-false-alarm convention: a superseded quote the parser
    guard misses, a cross-item arm count or component/total draw count, a
    neighboring per-arm token upgrading an aggregate threshold, or a
    future superseded-guard enable-flip)

WARN semantics: a WARN never blocks exit (exit 0). The Phase 1.5.0 wiring
carries WARN lines verbatim into the fact-checker + critic briefs — that
forwarding IS the ships-if-acknowledged mechanism for plans (unlike
clean-result bodies, plans have a downstream human-grade review that
weighs every WARN).

Scope discipline: every check here guarantees only that the contract
SURFACE exists (a Source: label has a non-empty evidence-shaped value, a
measurement-validity block has construct/metric content, ...). The
semantic questions — is each Source *correct*, does it *transfer*, is the
proxy *valid* — stay with the Phase 1.5 fact-checker and the Phase 2
critic ensemble. A PASS here is never "grounding verified".

Usage::

    uv run python scripts/verify_plan.py --issue 614 [--json]
    uv run python scripts/verify_plan.py --plan-file path/to/plan.md \
        [--kind experiment] [--json]

``--issue`` resolves the task folder via
``explore_persona_space.task_workflow.find_task_path`` (never hand-built
``tasks/`` paths) and verifies the newest ``plans/v{K}.md`` by NUMERIC
sort (``v10`` > ``v9``; never the ``plan.md`` symlink — follow-up rounds
re-point it, the verify_task_body check-16 / incident #597 trap), reading
``kind`` from ``body.md`` frontmatter. ``--plan-file`` verifies a
standalone file (e.g. a not-yet-persisted ``/tmp`` handoff draft);
``--kind`` applies in file mode only and defaults to ``experiment`` (the
strictest, matching the issue-mode missing-kind fallback).

Exit codes: 0 = PASS (WARNs allowed), 1 = at least one FAIL,
2 = resolution / IO error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import yaml

# Make src/ importable so the SHARED §9 wall-budget parser (check c47)
# resolves against THIS checkout's src/ — the poll_pipeline.py shim shape
# (#2172 AC #4). The module is stdlib-only (re + dataclasses) and the
# package __init__ is empty, so the import adds no measurable startup cost
# (the module-level local-import discipline noted below is about heavy
# dependencies like task_workflow, which stays a local import).
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from explore_persona_space.plan_wall_budget import parse_plan_wall_budget  # noqa: E402

# ─── Constants ─────────────────────────────────────────────────────────────

# Plan-relevant kinds for the CLI `--kind` choices (file mode). Kept an
# explicit ordered tuple because argparse `choices=` uses this order for help
# text + error messages. Membership is `("experiment", *EXEMPT_KINDS)`; the
# canonical single source for the exempt subset is
# `task_workflow.CODE_KINDS`, and `tests/test_verify_plan.py` pins both this
# tuple and EXEMPT_KINDS to it so the three `kind`-enum copies can never drift
# (incident #672). Local-import discipline (this module avoids a module-level
# `task_workflow` dependency) keeps the literal here; the test is the gate.
VALID_KINDS = ("experiment", "analysis", "infra", "batch", "survey")

# Kinds exempt from the experiment-only checks (CLAUDE.md Critical Rules:
# "`kind: analysis|infra|batch|survey` exempt"). Byte-identical to
# `task_workflow.CODE_KINDS`; pinned by the drift test (see above).
EXEMPT_KINDS = frozenset({"analysis", "infra", "batch", "survey"})

# Check 0 thresholds: a real plan (even a terse infra/analysis one — #575's
# v1 is the short end of the observed corpus) clears these comfortably; a
# truncated / contaminated handoff (#562 harness-trailer class) does not.
MIN_PLAN_CHARS = 1500
MIN_PLAN_HEADINGS = 3

# Check 8 "non-contradictory in form" emptiness bar: the innermost section
# carrying a success/kill anchor must have at least this much body text.
MIN_CRITERIA_CARRIER_CHARS = 80

# Tolerant N/A prefix: em dash, en dash, colon, opening paren, or hyphen
# after the N/A token ("N/A — ...", "N/A: ...", "N/A (not a replication)").
NA_RE = r"(?i)\bN/?A\b\s*[—–:(-]\s*"  # noqa: RUF001 — en dash is real plan text

# Check 1: inline `Source:` label. Value capture stops at newline or table
# pipe so a label inside a table cell captures only its own cell.
_SOURCE_LABEL_RE = re.compile(r"(?i)\bSource:\s*([^\n|]*)")

# Tokens that make a Source value "prose about sources" rather than
# evidence (planner.md's own boilerplate: "One `Source:` per unique value").
_SOURCE_VALUE_STOPWORDS = frozenset({"per", "unique", "value", "each", "every"})

# Check 5: the one exact, pre-existing string contract (planner.md §0).
# `\**` admits the bold form (`**Estimated GPU-hours (total):** 4`);
# optional backticks admit the inline-code form. A single plain number —
# ranges and `~`-qualified values fail.
GPU_LINE_RE = re.compile(r"(?i)estimated\s+gpu-?hours\s+\(total\):\**\s*`?([0-9]+(?:\.[0-9]+)?)`?")
GPU_LABEL_RE = re.compile(r"(?i)estimated\s+gpu-?hours\s+\(total\)")

# Check 5: backtick-tolerant numeric-range detector, applied with .match()
# anchored at the captured value BEFORE the annotation stops run. One of
# the stops is the closing backtick, so a stop-first scan truncates
# "`4`-8" to "4" and false-PASSes the range as its first number (round-2
# reconciler blocker gpu-hours-backtick-range-false-pass; "`40`-200" is
# the auto-approve-cap understatement shape). The leading "`?" is
# redundant after GPU_LINE_RE consumed the value's opening backtick, but
# kept to match the endorsed detector shape.
GPU_RANGE_AT_VALUE_RE = re.compile(
    r"`?[0-9]+(?:\.[0-9]+)?`?\s*[-–]\s*`?[0-9]"  # noqa: RUF001 — en-dash ranges are real
)

# Checks 4 + 10: marker-leakage vocabulary (NOT the bare token "marker",
# which false-fires on workflow vocabulary — `post-marker`, `epm:` markers —
# present in nearly every plan).
_MARKER_VOCAB_RE = re.compile(
    r"※|83399|marker[- ]leakage|log ?p\(marker\)|markeronlydatacollator",
    re.IGNORECASE,
)

# Check 8 vocabulary families.
_SUCCESS_RE = re.compile(r"(?i)success criteri|acceptance criteri|decision rule|decision gate")
_KILL_RE = re.compile(
    r"(?i)kill[- ]criteri|abort criteri|stop criteri|halt-and-report|what would change my mind"
)

# Check 11: trigger = the CLI flag form anywhere in the RAW plan (smoke
# commands legitimately live inside fences/tables). Evidence = a line naming
# a dry-run-exercising test: a `test_` identifier co-occurring with a dry-run
# token (no \b before "dry" — the token legitimately sits embedded in
# identifiers like test_drain_dry_run_no_dispatch), or the word "test"
# co-occurring with the Python kwarg form `dry_run`. `--dry-run` flag
# occurrences are STRIPPED from the line before the tier-1 scan: the bare
# flag next to test vocabulary deliberately does NOT self-certify — neither
# the "run the smoke, then the test suite" sentence shape nor the #633 v1
# false-PASS shape (ONE `Verification commands:` line carrying both the
# success-path pytest invocation and the `--dry-run` smoke command).
_DRYRUN_FLAG_RE = re.compile(r"--dry-run\b")
_DRYRUN_ANY_RE = re.compile(r"(?i)dry[-_ ]?run")
_DRYRUN_KWARG_RE = re.compile(r"(?i)dry_run")
_TEST_IDENT_RE = re.compile(r"\btest_\w+")
_TEST_WORD_RE = re.compile(r"(?i)\btests?\b")

# Check 3: data-source tier vocabulary (CLAUDE.md realistic-data rule).
_TIER_RE = re.compile(
    r"(?i)tier[-\s]*[1-4]|real-world data|established (?:dataset|benchmark)"
    r"|diverse llm[- ]generated|programmatic(?:ally)? generated|realistic-data preference"
)
_TIER_34_RE = re.compile(
    r"(?i)tier[-\s]*[34]|diverse llm[- ]generated|programmatic(?:ally)? generated"
)

# ─── Result type ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """One check verdict.

    ``skipped`` (kind-exempt or conditional trigger not fired) and
    ``is_warn`` both leave ``passed=True`` — only a hard FAIL flips it.
    """

    id: str
    name: str
    passed: bool
    detail: str = ""
    is_warn: bool = False
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        if not self.passed:
            return "FAIL"
        if self.is_warn:
            return "WARN"
        return "PASS"

    def render(self) -> str:
        line = f"  [{self.status}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


def _pass(cid: str, name: str, detail: str = "") -> CheckResult:
    return CheckResult(cid, name, True, detail)


def _warn(cid: str, name: str, detail: str) -> CheckResult:
    return CheckResult(cid, name, True, detail, is_warn=True)


def _fail(cid: str, name: str, detail: str) -> CheckResult:
    return CheckResult(cid, name, False, detail)


def _skip(cid: str, name: str, detail: str) -> CheckResult:
    return CheckResult(cid, name, True, detail, skipped=True)


# ─── Parsing helpers ───────────────────────────────────────────────────────


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading ``---`` YAML frontmatter block off ``text``.

    Returns ``({}, text)`` unchanged when there is no parseable block.
    Used for ``body.md`` (kind lookup) — plan files are passed through raw.
    """
    if not text.startswith("---\n"):
        return {}, text
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        return {}, text
    fm_block = rest[:end]
    body = rest[end + len("\n---\n") :]
    try:
        fm = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, body


def _fence_mask(lines: list[str]) -> list[bool]:
    """Per-line mask: True when the line is a fence delimiter or inside a
    fenced code block. Both ``` and ~~~ toggle, matching CommonMark's
    relaxed rule (same behavior as verify_task_body.find_h2_sections)."""
    mask: list[bool] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            mask.append(True)
            continue
        mask.append(in_fence)
    return mask


def strip_fences(text: str) -> str:
    """Return ``text`` with fenced code blocks (and the fence delimiter
    lines) removed, so example commands inside fences can neither satisfy
    nor trip a prose-contract check."""
    lines = text.splitlines()
    mask = _fence_mask(lines)
    return "\n".join(line for line, fenced in zip(lines, mask, strict=True) if not fenced)


@dataclass
class Heading:
    level: int
    text: str
    line: int  # heading line index
    end: int  # exclusive end line of the section (next same-or-higher heading)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# HTML heading tag: matches <h1>…<h6> with optional attributes (e.g.
# <h2 style="margin-top:0">). Used by check_plan_nonstub to accept the HTML
# output format documented in CLAUDE.md § Output format (adversarial-planner
# defaults to HTML for browser-reading).
_HTML_HEADING_RE = re.compile(r"<h[1-6]\b[^>]*>", re.IGNORECASE)


def _headings(text: str) -> list[Heading]:
    """Fence-aware heading parser for H1-H6 (plans put required blocks at
    H2 AND H3; H4 shows up in pipelines). Each heading's section extends to
    the next heading of the same or higher level."""
    lines = text.splitlines()
    mask = _fence_mask(lines)
    found: list[tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _HEADING_RE.match(line.strip())
        if m:
            found.append((len(m.group(1)), m.group(2).strip(), i))
    out: list[Heading] = []
    for k, (level, htext, start) in enumerate(found):
        end = len(lines)
        for level2, _, start2 in found[k + 1 :]:
            if level2 <= level:
                end = start2
                break
        out.append(Heading(level, htext, start, end))
    return out


def section_text_by_keywords(text: str, keywords: tuple[str, ...]) -> str | None:
    """Keyword-fuzzy section locator: first heading (document order) whose
    text contains any keyword (case-insensitive substring) wins; returns
    heading line + section body. None when no heading matches. Never exact
    heading matching — the observed corpus drifts (`## 7. Decision Gates,
    Success and Kill Criteria` vs `## 7. Decision gates` vs `## 10.
    Hyperparameter grounding (§11)`)."""
    lines = text.splitlines()
    lowered = tuple(k.casefold() for k in keywords)
    for h in _headings(text):
        htext = h.text.casefold()
        if any(k in htext for k in lowered):
            return "\n".join(lines[h.line : h.end])
    return None


def _innermost_section(headings: list[Heading], line_idx: int) -> Heading | None:
    """Deepest (then latest-starting) heading whose section contains
    ``line_idx``; None when the line precedes every heading."""
    best: Heading | None = None
    for h in headings:
        if h.line <= line_idx < h.end and (
            best is None or h.level > best.level or h.line > best.line
        ):
            best = h
    return best


# ─── Check 0 — plan-nonstub (FAIL, short-circuits; all kinds) ──────────────


def check_plan_nonstub(plan: str) -> CheckResult:
    """Defense against a contaminated / truncated handoff file (the #562
    harness-trailer incident class): minimum size, minimum structure, no
    lone stub token as the whole body."""
    cid, name = "c0_plan_nonstub", "plan non-stub"
    stripped = plan.strip()
    if re.fullmatch(r"(?i)[\s#*`>-]*(placeholder|tbd|todo|stub)[.!]?\s*", stripped or " "):
        return _fail(cid, name, "plan body is a lone stub token — broken handoff (#562 class)")
    if len(stripped) < MIN_PLAN_CHARS:
        return _fail(
            cid,
            name,
            f"plan body is {len(stripped)} chars (< {MIN_PLAN_CHARS}) — looks like a "
            "stub or truncated handoff (#562 class); persist the real plan first",
        )
    # Count markdown headings first; also count HTML headings to accept the
    # HTML output format documented in CLAUDE.md § Output format
    # (adversarial-planner defaults to HTML for browser-reading; an HTML plan
    # with 20+ <h2>/<h3> tags was incorrectly FAILed at "only 1 heading (< 3)"
    # because _headings() is markdown-only — incident task #640, 2026-06-15).
    n_headings = len(_headings(plan)) + len(_HTML_HEADING_RE.findall(plan))
    if n_headings < MIN_PLAN_HEADINGS:
        return _fail(
            cid,
            name,
            f"only {n_headings} headings (< {MIN_PLAN_HEADINGS}) — not a structured plan",
        )
    return _pass(cid, name, f"{len(stripped)} chars, {n_headings} headings")


# ─── Check 1 — §11 hyperparameter Source: grounding ────────────────────────


def _is_evidence_value(value: str) -> bool:
    """True when a Source value carries evidence: an arXiv id, a prior
    issue ``#<M>``, a file path, a URL, ``ungrounded``, or ≥2 non-stopword
    tokens (excluding the boilerplate words of planner.md's own "One
    `Source:` per unique value" sentence — prose ABOUT sources does not
    count)."""
    v = value.strip().strip("`*").strip()
    if not v:
        return False
    if "ungrounded" in v.lower():
        return True
    if re.search(r"\b\d{4}\.\d{4,5}\b", v):  # arXiv id
        return True
    if re.search(r"#\d+", v):  # prior issue
        return True
    if re.search(r"https?://", v):
        return True
    if re.search(r"[\w./-]+\.(?:py|md|json|jsonl|yaml|yml|sh|csv|txt)\b", v):  # file path
        return True
    tokens = [
        t for t in re.findall(r"[A-Za-z][\w-]*", v) if t.lower() not in _SOURCE_VALUE_STOPWORDS
    ]
    return len(tokens) >= 2


def _blankish(value: str) -> bool:
    t = value.strip().strip("`*").strip()
    return (not t) or t.lower().startswith("tbd") or set(t) <= {"?"}


_TABLE_SEP_RE = re.compile(r"\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{0,}:?\s*\|?")


def _split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _source_column_cells(text: str) -> list[str]:
    """Body cells of every markdown-table column whose header cell is
    exactly ``Source`` (case-insensitive; bold/backticks stripped) — the
    #614 v2 §11 shape (`| What | Why (tied to Goal) | Source | ... |`)."""
    lines = text.splitlines()
    cells: list[str] = []
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and sep.startswith("|") and _TABLE_SEP_RE.fullmatch(sep)):
            i += 1
            continue
        header_cells = [c.strip().strip("*`").strip().casefold() for c in _split_table_row(header)]
        col = next((j for j, c in enumerate(header_cells) if c == "source"), None)
        k = i + 2
        while k < len(lines) and lines[k].strip().startswith("|"):
            if col is not None:
                row = _split_table_row(lines[k])
                if col < len(row):
                    cells.append(row[col])
            k += 1
        i = k
    return cells


def check_source_grounding(plan: str, kind: str) -> CheckResult:
    """Contract (CLAUDE.md Critical Rule + planner.md §11): every
    load-bearing hyperparameter carries a non-empty ``Source:`` (inline
    label or a ``Source`` table column), or the explicit ``ungrounded —
    needs smoke-test`` marker, or the section-level N/A. Presence-only:
    Source correctness / transfer stays fact-checker-owned."""
    cid, name = "c1_source_grounding", "§11 hyperparameter Source: grounding"
    if kind in EXEMPT_KINDS:
        return _skip(cid, name, "kind-exempt: analysis|infra|batch|survey train no model")
    sect = section_text_by_keywords(
        plan, ("decision rationale", "hyperparameter grounding", "decision grounding")
    )
    scope = sect if sect is not None else plan
    if _standalone_na_declared(
        scope, r"no (?:model )?(?:training )?(?:model training|hyperparameters|training)"
    ):
        return _pass(
            cid, name, "explicit N/A declared (no model training / no training hyperparameters)"
        )
    text = strip_fences(scope)
    raw_inline = [m.group(1) for m in _SOURCE_LABEL_RE.finditer(text)]
    inline = [v for v in raw_inline if _is_evidence_value(v)]
    table_all = _source_column_cells(text)
    table_cells = [c for c in table_all if _is_evidence_value(c)]
    blank = [v for v in raw_inline if _blankish(v)] + [c for c in table_all if _blankish(c)]
    sources = inline + table_cells
    if sect is None and not sources and not blank:
        return _fail(
            cid,
            name,
            "no Decision Rationale / grounding section and zero Source entries — every "
            "load-bearing hyperparameter needs a Source (planner.md §11); if the plan trains "
            "no model, declare `N/A — no model training` / `N/A — no training hyperparameters` "
            "— each on its own line, unwrapped (no backticks/quotes)",
        )
    if blank:
        return _fail(
            cid,
            name,
            f"{len(blank)} blank/TBD Source entr{'y' if len(blank) == 1 else 'ies'} — "
            "planner.md §11 says never blank: cite a source or write "
            "`ungrounded — needs smoke-test`",
        )
    if sect is None:
        return _warn(
            cid,
            name,
            f"{len(sources)} Source entries found but no recognizable §11 heading "
            "(heading drift?) — fact-checker must locate them manually",
        )
    if not sources:
        return _fail(
            cid,
            name,
            "§11-style section present but zero Source entries (an inline source label — "
            "`Source` followed by a colon — or a `Source` table column)",
        )
    ungrounded = [s for s in sources if "ungrounded" in s.lower()]
    return _pass(
        cid,
        name,
        f"{len(sources)} Source entries: {len(inline)} inline, {len(table_cells)} table-column "
        f"({len(ungrounded)} marked ungrounded — fact-checker flags those for smoke-test); "
        "presence-only — Source correctness/transfer stays fact-checker-owned",
    )


# ─── Check 2 — per-DV measurement validity ─────────────────────────────────


def check_measurement_validity(plan: str, kind: str) -> CheckResult:
    """planner.md §6 required block: per dependent variable, the construct,
    the metric, and the on-distribution status. FAIL only when ALL signals
    are absent; a bare heading without construct/metric content is a WARN
    with the residual explicitly fact-checker-owned."""
    cid, name = "c2_measurement_validity", "per-DV measurement validity"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt: analysis|infra|batch|survey have no behavioral DV")
    if _standalone_na_declared(plan, r"no behavioral construct"):
        return _pass(cid, name, "explicit N/A declared (no behavioral construct)")
    text = strip_fences(plan)
    mv_headings = [h for h in _headings(plan) if "measurement validity" in h.text.casefold()]
    table = re.search(r"(?im)^\|(?=[^\n]*construct)(?=[^\n]*metric)[^\n]*\|\s*$", text)
    phrase = re.search(r"(?i)measurement validity", text)
    ondist = re.search(r"(?i)on-?distribution|on-?policy|teacher-?forced", text)
    heading_has_content = False
    if mv_headings:
        h = mv_headings[0]
        body = "\n".join(plan.splitlines()[h.line + 1 : h.end])
        heading_has_content = re.search(r"(?i)construct|metric", strip_fences(body)) is not None
    if table or heading_has_content:
        return _pass(
            cid,
            name,
            "measurement-validity block found with construct/metric content"
            + ("" if ondist else " (no on-distribution/on-policy statement spotted — verify)"),
        )
    if mv_headings:
        return _warn(
            cid,
            name,
            "measurement-validity heading present but no construct/metric content detected "
            "in its section — per-DV substance is fact-checker-owned",
        )
    if phrase:
        return _warn(
            cid, name, "phrase present but no recognizable block/table — verify per-DV rows exist"
        )
    return _fail(
        cid,
        name,
        "no measurement-validity declaration (planner.md §6 required block: per-DV construct "
        "+ metric + on-distribution status; non-behavioral plans declare "
        "`N/A — no behavioral construct` on its own line, unwrapped — no backticks/quotes)",
    )


# ─── Check 3 — data-source tier (WARN-only) ────────────────────────────────


def check_data_tier(plan: str, kind: str) -> CheckResult:
    """CLAUDE.md realistic-data preference order: the plan names its data
    tier. WARN-only — the vocabulary is descriptive, not a pinned string
    contract."""
    cid, name = "c3_data_tier", "data-source tier named"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    m = _TIER_RE.search(text)
    if not m:
        return _warn(
            cid,
            name,
            "no data-source tier named — CLAUDE.md realistic-data rule requires naming the "
            "tier (real-world / established dataset / diverse-LLM-synthetic / programmatic) "
            "+ tier-3/4 justification",
        )
    detail = f"data-tier vocabulary found ({m.group(0)!r})"
    if _TIER_34_RE.search(text) and not re.search(r"(?i)justif|absence|confound", text):
        detail += (
            "; note: tier-3/4 vocabulary present without a justification token "
            "(justif|absence|confound) — critics should verify the required justification"
        )
    return _pass(cid, name, detail)


# ─── Check 4 — contrastive negatives (WARN-only, conditional) ──────────────


def _c4_na_escape_declared(plan: str) -> bool:
    """Standalone c4 escape — accepts BOTH the canonical
    ``N/A — not a behavior-implantation`` AND the measurement-plan alias
    ``N/A — no behavior implantation`` (#1689/#1700; measurement/geometry
    plans read awkwardly under the "not a behavior-implantation" wording,
    so the alias makes the escape naturally discoverable). Both forms route
    through ``_standalone_na_declared`` so the same anti-paste discipline
    (standalone line, unwrapped, non-fenced) applies."""
    return _standalone_na_declared(
        plan, r"not a behavior[- ]implantation"
    ) or _standalone_na_declared(plan, r"no behavior[- ]implantation")


def check_contrastive_negatives(plan: str, kind: str) -> CheckResult:
    """Behavior-implantation plans must name a contrastive-negative set or
    one of the two named exemptions (.claude/rules/contrastive-negatives.md).
    WARN not FAIL: the trigger is a content heuristic and the Methodology
    critic REVISEs the true positives — this gate surfaces, never
    adjudicates."""
    cid, name = "c4_contrastive_negatives", "contrastive negatives (behavior implantation)"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    implant = re.search(r"(?i)\bimplant\w*\b", text) or re.search(
        r"(?i)behavior[- ]implantation", text
    )
    marker_trigger = _MARKER_VOCAB_RE.search(text) and re.search(r"(?i)\bpersona\b", text)
    if not (implant or marker_trigger):
        return _skip(
            cid,
            name,
            "not detected as behavior-implantation (no implant/leakage-marker vocabulary)",
        )
    if _c4_na_escape_declared(plan):
        return _pass(
            cid,
            name,
            "explicit N/A declared on its own line, unwrapped "
            "(not a behavior-implantation | no behavior implantation)",
        )
    if re.search(r"(?i)contrastive[- ]negatives?", text):
        lowered = text.lower()
        found = [t for t in ("panel", "ratio", "1:1", "disjoint") if t in lowered]
        return _pass(
            cid,
            name,
            "contrastive-negative vocabulary present"
            + (
                f" (also found: {', '.join(found)})"
                if found
                else " (none of panel/ratio/1:1/disjoint spotted — verify composition)"
            ),
        )
    if re.search(
        r"(?i)single manipulated variable is contrastive|positive-only (?:parent|paper)"
        r"|exemption \(?[ab]\)?",
        text,
    ):
        return _pass(cid, name, "named exemption vocabulary present")
    return _warn(
        cid,
        name,
        "behavior-implantation vocabulary detected but no contrastive-negative set or named "
        "exemption — .claude/rules/contrastive-negatives.md (panel + ratio + disjointness); "
        "Methodology critic must gate this; or declare `N/A — not a behavior-implantation` "
        "(or `N/A — no behavior implantation` for measurement/geometry plans) on its own "
        "line, unwrapped (no backticks/quotes)",
    )


# ─── Check 5 — GPU-hour estimate (FAIL for ALL kinds) ──────────────────────


def check_gpu_hours(plan: str, kind: str) -> CheckResult:
    """The one exact string contract (planner.md §0): a machine-readable
    ``Estimated GPU-hours (total): <number>`` line. FAILs for ALL kinds —
    the Step 2c consumer (`task.py` `_resolve_autonomous_plan_gate`) is
    kind-blind and parks an autonomous session on a missing estimate;
    exempt kinds satisfy the check with ``0``. Scanned on the RAW plan
    (the line legitimately appears backtick-wrapped inside summary
    bullets / tables)."""
    cid, name = "c5_gpu_hours", "GPU-hour estimate line"
    del kind  # deliberately kind-blind, mirroring the Step 2c gate
    m = GPU_LINE_RE.search(plan)
    if not m:
        if GPU_LABEL_RE.search(plan):
            return _fail(
                cid,
                name,
                "`Estimated GPU-hours (total):` label present but the value is unparseable — "
                "a single plain number is required (no `~`, no ranges); exempt kinds use "
                "`Estimated GPU-hours (total): 0`",
            )
        return _fail(
            cid,
            name,
            "machine-readable `Estimated GPU-hours (total): <number>` line absent — required "
            "for ALL kinds (the Step 2c autonomous plan gate is kind-blind and parks on a "
            "missing estimate); exempt kinds satisfy with `Estimated GPU-hours (total): 0`",
        )
    # Range scan, scoped to the text immediately after the label and
    # stopping at the first parenthetical, em-dash, closing-backtick, or
    # sentence-boundary annotation — NOT the whole line (#610 carries
    # "— worst ≈ 42 — see §9" and #614 carries "1× A100-80" on the same  # noqa: RUF003
    # line; #580 carries "`. Wall ~1–1.5 h including review." after the  # noqa: RUF003
    # backtick-wrapped value — calibration-driven predicate adjustment,
    # plan §12; a whole-line digit-dash-digit scan would false-FAIL all
    # three shapes).
    line_end = plan.find("\n", m.end())
    if line_end == -1:
        line_end = len(plan)
    tail = plan[m.start(1) : line_end]
    # Backtick-tolerant range detection FIRST, anchored at the value:
    # the closing-backtick annotation stop below would otherwise truncate
    # a backtick-wrapped-number range at the first close backtick and
    # PASS it as its first number (round-2 fix; counterexamples that must
    # FAIL: `4`-8, `4`-`8`, `4` - 8, `40`-200). Anchoring via .match()
    # keeps the #580 next-sentence wall-time range and the #610/#614
    # annotation shapes out of reach — those put a non-dash token between
    # the value and any later digit-dash-digit text.
    range_m = GPU_RANGE_AT_VALUE_RE.match(tail)
    if range_m:
        return _fail(
            cid,
            name,
            f"value reads as a range, not a single number ({range_m.group(0).strip()!r}) — "
            "the Step 2c gate needs one number (put worst-case bounds in a parenthetical "
            "annotation)",
        )
    for stop in ("(", "—", "`", ". "):
        idx = tail.find(stop)
        if idx != -1:
            tail = tail[:idx]
    if re.search(r"[0-9]\s*[-–]\s*[0-9]", tail):  # noqa: RUF001 — en-dash ranges are real
        return _fail(
            cid,
            name,
            f"value reads as a range, not a single number ({tail.strip()!r}) — the Step 2c "
            "gate needs one number (put worst-case bounds in a parenthetical annotation)",
        )
    return _pass(cid, name, f"{m.group(1)} GPU-h")


# ─── Check 6 — reused-artifact fitness (WARN-only, conditional) ────────────


def check_reuse_fitness(plan: str, kind: str) -> CheckResult:
    """Plans reusing trained HF artifacts must carry the fitness
    attestations (a)-(m) (.claude/rules/artifact-reuse.md). WARN not FAIL:
    trigger and item-detection are both heuristic, and the demonstrated
    failure modes (#545/#600/#601) are semantic — the gate's value is
    forcing the section to exist and naming the thirteen letters.

    Accepted declaration shapes (#1314): the historical 'fitness'
    vocabulary, a 'reuse map' / 'reuse-map' section (the #1090 v7
    '### D3 — Reuse map' shape; artifact-reuse.md's own term for the
    plan record), '(self-)attestation(s)', or the literal (a)-(m) range
    token ((a)-(j)/(a)-(k)/(a)-(l) grandfathered for in-flight plans;
    #1366/#1522/#2043) (hyphen / en-dash / em-dash / ellipsis)."""
    cid, name = "c6_reuse_fitness", "reused-artifact fitness attestation"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    hf_hits = [
        m.start() for m in re.finditer(r"superkaiba1/|adapter_config\.json|hf_hub_download", text)
    ]
    reuse_near_hf = any(
        re.search(r"(?i)\breus\w*", text[max(0, i - 300) : i + 300]) for i in hf_hits
    )
    reuse_heading = any(re.search(r"(?i)reuse|reused[- ]artifact", h.text) for h in _headings(plan))
    if not (reuse_near_hf or reuse_heading):
        return _skip(cid, name, "no HF-artifact reuse detected")
    if _standalone_na_declared(plan, r"no (?:artifact )?reuse"):
        return _pass(cid, name, "explicit no-reuse declaration (N/A — no artifact reuse)")
    declaration = re.search(
        r"(?i)fitness"  # historical vocabulary (the pre-#1314 detector, unchanged)
        r"|reuse[- ]map"  # 'Reuse map' section shape (#1090 v7 D3; artifact-reuse.md's own term)
        r"|(?:self[- ])?attestation"  # 'self-attestation' / 'attestation(s)'
        r"|\(a\)\s*[-–—…]\s*\([jklm]\)",  # (a)-(m); older ranges grandfathered  # noqa: RUF001
        text,
    )
    letters = {m.group(1) for m in re.finditer(r"\(([a-m])\)", text)}
    if declaration and len(letters) >= 4:
        return _pass(
            cid,
            name,
            f"fitness/reuse-map declaration present ({len(letters)}/13 lettered items spotted)",
        )
    if declaration:
        return _warn(
            cid,
            name,
            f"fitness/reuse-map declaration vocabulary present but only {len(letters)} of the "
            "(a)–(m) items detectable — verify all thirteen attestations (recipe/regime/cells/"  # noqa: RUF001
            "single-var/hub-resolution/content-identity/scaling/backend-fetchability/"
            "code-throughput/pair-provenance/parent-lineage/validity-domain/device-domain) "
            "before approval",
        )
    return _warn(
        cid,
        name,
        "plan reuses HF artifacts but no fitness check / (a)–(m) reuse-map attestation found — "  # noqa: RUF001
        "CLAUDE.md reuse rule requires attestations (a)–(m); consistency-checker + Methodology "  # noqa: RUF001
        "critic must gate this",
    )


# ─── Check 7 — replication fidelity (WARN-only, conditional) ───────────────


def check_replication_fidelity(plan: str, kind: str) -> CheckResult:
    """When the Goal mentions replicating, the plan must address
    replication fidelity (match the paper's data + recipe first;
    .claude/rules/replication-fidelity.md). WARN because "does the effect
    replicate across seeds" is a benign false trigger."""
    cid, name = "c7_replication_fidelity", "replication fidelity"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    goal = section_text_by_keywords(plan, ("goal",))
    if goal is None:
        m = re.search(r"(?im)^goal:\s*(.+)$", plan)
        goal = m.group(0) if m else None
    if goal is None or not re.search(r"(?i)replicat", goal):
        return _skip(cid, name, "Goal does not mention replication")
    text = strip_fences(plan)
    if _standalone_na_declared(plan, r"not a replication"):
        return _pass(
            cid, name, "explicit N/A declared on its own line, unwrapped (not a replication)"
        )
    if re.search(
        r"(?i)paper'?s (?:data|recipe|corpus)|faithful|replication[- ]fidelity|deviation", text
    ):
        return _pass(
            cid,
            name,
            "replication-fidelity vocabulary present (paper recipe / deviations addressed)",
        )
    return _warn(
        cid,
        name,
        "Goal mentions replication but no fidelity vocabulary — match the data + recipe of "
        "the source paper FIRST and name every divergence (replication rule, CLAUDE.md); "
        "or declare `N/A — not a replication` on its own line, unwrapped "
        "(no backticks/quotes)",
    )


# ─── Check 8 — success + kill criteria ─────────────────────────────────────


def _tldr_ranges(plan: str) -> list[tuple[int, int]]:
    """Line ranges of the §0.0 / TL;DR region(s). planner.md §0.0 MANDATES
    a "What would change my mind" line there, so a KILL hit inside is
    template conformance, not evidence of real kill criteria."""
    out: list[tuple[int, int]] = []
    for h in _headings(plan):
        text = h.text.strip()
        if "tl;dr" in text.casefold() or re.match(r"(?:§\s*)?0\.0\b", text):
            out.append((h.line, h.end))
    return out


def _exempt_tldr_kill_pass(
    cid, name, kind, succ_solid, kill_solid, kill_tldr_hits, carrier_ok, section_name
):
    """PASS result for the #1291 exempt-kind acceptance — a kind-exempt plan
    whose kill family is satisfied EITHER by (a) a solid §0.0/TL;DR
    change-my-mind hit, OR (b) a solid ``acceptance criteri`` success anchor
    whose enclosing section itself names acceptance criteria (#1668/#1700 —
    for an infra fix the acceptance-criteria BLOCK IS the revert criterion,
    even when it lives under §1 Goal / Motivation instead of a dedicated
    ``## 7. Decision gates, success and kill criteria`` heading).
    (Success family solid, no solid kill outside the TL;DR.) None when
    the acceptance does not apply (``check_success_kill`` falls through to
    its missing-family verdicts). ``kind: experiment`` is BYTE-UNCHANGED:
    the guard on ``kind not in EXEMPT_KINDS`` gates both branches."""
    if kind not in EXEMPT_KINDS or not succ_solid or kill_solid:
        return None
    # (b) Acceptance-criteria-block branch (#1668/#1700): iterate ALL solid
    # success hits — the FIRST hit may sit under a non-acceptance section
    # (e.g. §2 Design) while a later hit lands under the true acceptance
    # heading. Fail-closed to the (a) TL;DR path when no hit's enclosing
    # section names acceptance criteria.
    for si, sa in succ_solid:
        sec = section_name(si).lower()
        if "acceptance criteri" in sec:
            return _pass(
                cid,
                name,
                f"success anchor {sa!r} in §{section_name(si)!r} — the acceptance-criteria "
                "block SATISFIES the kill family for kind-exempt plans (the block IS the "
                "revert criterion for a code/infra change; kind: experiment still requires "
                "kill criteria outside the TL;DR — #1668/#1700, extending #1291)",
            )
    # (a) EXISTING: §0.0 TL;DR change-my-mind path (unchanged).
    tldr_solid = [(i, a) for i, a in kill_tldr_hits if carrier_ok(i)]
    if not tldr_solid:
        return None
    si, sa = succ_solid[0]
    ka = tldr_solid[0][1]
    return _pass(
        cid,
        name,
        f"success anchor {sa!r} in §{section_name(si)!r}; kill anchor {ka!r} inside "
        "the §0.0/TL;DR region — accepted for kind-exempt plans (the mandated "
        "change-my-mind line IS the revert criterion for a code/infra change; "
        "kind: experiment still requires kill criteria outside the TL;DR — #1291)",
    )


def check_success_kill(plan: str, kind: str) -> CheckResult:
    """Both a success-criteria family and a kill-criteria family must be
    present and non-empty in form (each carrier section ≥ 80 chars —
    emptiness check only; semantic joint-satisfiability stays with the
    Statistics critic per planner.md §7). The KILL count EXCLUDES the
    §0.0/TL;DR region for ``kind: experiment``; for exempt kinds
    (analysis/infra/batch/survey) a solid TL;DR "What would change my
    mind" hit satisfies the kill family when the success family is solid —
    the mandated change-my-mind line IS the revert criterion for a
    code/infra change (#1291; founding incidents #1279/#1276).
    `kind: experiment` FAILs on both-absent; exempt kinds WARN, and the
    exempt-kind missing-kill WARN detail carries the standard §0.0 remedy
    sentence."""
    cid, name = "c8_success_kill_criteria", "success + kill criteria"
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    tldr = _tldr_ranges(plan)

    def in_tldr(i: int) -> bool:
        return any(s <= i < e for s, e in tldr)

    def carrier_ok(i: int) -> bool:
        h = _innermost_section(headings, i)
        body = "\n".join(lines[h.line + 1 : h.end]) if h else plan
        return len(strip_fences(body).strip()) >= MIN_CRITERIA_CARRIER_CHARS

    def section_name(i: int) -> str:
        h = _innermost_section(headings, i)
        return h.text if h else "<preamble>"

    succ_hits: list[tuple[int, str]] = []
    kill_hits: list[tuple[int, str]] = []
    kill_tldr_hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _SUCCESS_RE.search(line)
        if m:
            succ_hits.append((i, m.group(0)))
        m = _KILL_RE.search(line)
        if m:
            (kill_tldr_hits if in_tldr(i) else kill_hits).append((i, m.group(0)))

    succ_solid = [(i, a) for i, a in succ_hits if carrier_ok(i)]
    kill_solid = [(i, a) for i, a in kill_hits if carrier_ok(i)]

    if succ_solid and kill_solid:
        si, sa = succ_solid[0]
        ki, ka = kill_solid[0]
        return _pass(
            cid,
            name,
            f"success anchor {sa!r} in §{section_name(si)!r}; kill anchor {ka!r} in "
            f"§{section_name(ki)!r} (form-only check — joint satisfiability stays with the "
            "Statistics critic)",
        )
    exempt_pass = _exempt_tldr_kill_pass(
        cid, name, kind, succ_solid, kill_solid, kill_tldr_hits, carrier_ok, section_name
    )
    if exempt_pass is not None:
        return exempt_pass
    missing = []
    if not succ_solid:
        missing.append(
            "success criteria (success/acceptance criteria, decision rule/gate)"
            + (" [vocabulary found but carrier section looks empty]" if succ_hits else "")
        )
    if not kill_solid:
        missing.append(
            "kill criteria (kill/abort/stop criteria, halt-and-report) outside the §0.0/TL;DR "
            "region — the TL;DR's mandated 'What would change my mind' line is template "
            "conformance, not kill criteria"
            + (" [vocabulary found but carrier section looks empty]" if kill_hits else "")
        )
    detail = (
        "missing: "
        + "; ".join(missing)
        + ". Note: a `No gates — short run / pre-verified hypothesis` escape waives *gates*, "
        "not success/kill criteria"
    )
    if not kill_solid and kind in EXEMPT_KINDS:
        detail += (
            ". Standard remedy for kind-exempt plans: add the mandated §0.0 TL;DR "
            "'What would change my mind' line (a solid one satisfies this family — #1291)"
        )
    if len(missing) == 2 and kind == "experiment":
        return _fail(cid, name, detail)
    if len(missing) == 2:
        return _warn(cid, name, detail + " (kind-exempt degrade: WARN, not FAIL)")
    return _warn(cid, name, detail)


# ─── Check 9 — conditions/cells table + seeds (WARN-only) ──────────────────


def check_conditions_seeds(plan: str, kind: str) -> CheckResult:
    """The consistency-checker's input surface: a conditions/cells/arms
    declaration and seeds. A WARN tells the orchestrator the
    consistency-checker will be flying partially blind."""
    cid, name = "c9_conditions_seeds", "conditions/cells + seeds declared"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    cond_heading = any(
        re.search(r"(?i)\b(conditions?|cells?|arms?)\b", h.text) for h in _headings(plan)
    )
    cond_table = re.search(r"(?im)^\|(?=[^\n]*(?:config slug|what it tests))[^\n]*\|\s*$", text)
    conditions = bool(cond_heading or cond_table)
    seeds = re.search(r"(?i)\bseeds?\b", text) is not None
    if conditions and seeds:
        return _pass(cid, name, "conditions/cells signal + seeds named")
    missing = []
    if not conditions:
        missing.append("conditions/cells/arms heading or table")
    if not seeds:
        missing.append("seeds")
    return _warn(
        cid,
        name,
        f"missing: {', '.join(missing)} — the consistency-checker's input surface is "
        "partially blind",
    )


# ─── Check 10 — marker-recipe acknowledgment (WARN-only, conditional) ──────


def check_marker_recipe(plan: str, kind: str) -> CheckResult:
    """Marker-leakage plans must acknowledge the training recipe (anchor
    band / band-stop / recipe file) AND bystander gating
    (.claude/rules/marker-training-recipe.md). Trigger scans fence-stripped
    text (a fence-only ※ example is not a marker plan); evidence scans the
    RAW plan (a fenced `marker_band_stop=...` config line IS an
    acknowledgment)."""
    cid, name = "c10_marker_recipe", "marker-recipe acknowledgment"
    if kind != "experiment":
        return _skip(cid, name, "kind-exempt")
    if not _MARKER_VOCAB_RE.search(strip_fences(plan)):
        return _skip(cid, name, "no marker-leakage vocabulary detected")
    recipe = re.search(r"(?i)marker-training-recipe|band[- ]?stop|\[5,\s*12\]\s*nat", plan)
    bystander = re.search(r"(?i)bystander", plan)
    if recipe and bystander:
        return _pass(
            cid,
            name,
            "recipe acknowledgment (band / recipe-file reference) + bystander gating present",
        )
    if recipe or bystander:
        missing = (
            "bystander-gating statement"
            if recipe
            else "recipe acknowledgment (marker-training-recipe / band-stop / [5,12] nat band)"
        )
        return _warn(
            cid,
            name,
            f"marker experiment missing {missing} — read .claude/rules/marker-training-recipe.md "
            "+ marker-leakage-measurement.md before grounding the stopping recipe",
        )
    return _warn(
        cid,
        name,
        "marker experiment with no recipe acknowledgment — read "
        ".claude/rules/marker-training-recipe.md + marker-leakage-measurement.md before "
        "grounding the stopping recipe (incident #530/#480 class)",
    )


# ─── Check 11 — dry-run test coverage (WARN-only, conditional) ─────────────


def _dryrun_test_evidence_lines(plan: str) -> list[str]:
    """Lines naming a dry-run-exercising test (see the regex-block comment
    by ``_DRYRUN_FLAG_RE``): a ``test_`` identifier alongside a dry-run
    token — with ``--dry-run`` flag occurrences stripped first, so the smoke
    command itself cannot self-certify — or the word "test" alongside the
    ``dry_run`` kwarg form."""
    out: list[str] = []
    for line in plan.splitlines():
        sans_flag = _DRYRUN_FLAG_RE.sub("", line)
        if (_TEST_IDENT_RE.search(sans_flag) and _DRYRUN_ANY_RE.search(sans_flag)) or (
            _TEST_WORD_RE.search(line) and _DRYRUN_KWARG_RE.search(line)
        ):
            out.append(line.strip())
    return out


def check_dryrun_test_coverage(plan: str, kind: str) -> CheckResult:
    """``kind: infra|batch`` plans whose verification includes a ``--dry-run``
    smoke must also list a test exercising the dry_run code path. Three infra
    plans in a row (#596, #607, #633) shipped success-path-only test lists
    while their own final acceptance step was a live ``--dry-run`` invocation
    — a broken dry_run thread turns that smoke into a real mutation (for
    #633: a real dispatch of up to 3 autonomous sessions). WARN not FAIL:
    trigger and evidence are both line heuristics; the Phase 2 critics
    adjudicate. Both scans use the RAW plan — smoke commands and test lists
    legitimately live inside fences and tables."""
    cid, name = "c11_dryrun_test_coverage", "dry-run smoke backed by a dry-run test"
    if kind not in ("infra", "batch"):
        return _skip(
            cid, name, "kind-exempt: the dry-run-smoke acceptance pattern is an infra|batch shape"
        )
    if not _DRYRUN_FLAG_RE.search(plan):
        return _skip(cid, name, "no --dry-run smoke/verification command detected")
    if _standalone_na_declared(plan, r"no dry-?run smoke"):
        return _pass(cid, name, "explicit N/A declared (no dry-run smoke)")
    evidence = _dryrun_test_evidence_lines(plan)
    if evidence:
        return _pass(cid, name, f"dry-run-exercising test named ({evidence[0][:80]!r})")
    return _warn(
        cid,
        name,
        "plan names a `--dry-run` smoke/verification command but the test list has no test "
        "exercising the dry-run kwarg thread on the new code path — a broken dry-run kwarg "
        "thread turns the final smoke into a real mutation (#596/#607/#633 pattern); add the "
        "test, or declare `N/A — no dry-run smoke` on its own line, unwrapped "
        "(no backticks/quotes), if the flag mention is incidental",
    )


# ─── Check 12 — battery multiplier + batched commitment (conditional) ──────

# Trigger: the plan names a permutation/null-draw battery — battery/null-draw
# framing, or an explicit >=100 count attached to draw vocabulary. Deliberate
# NON-triggers: a bare "bootstrap CI" / "bootstrapped 95% CI" (cheap post-hoc
# stat, ubiquitous in plans). Known accepted under-trigger: "bootstrap with
# B=2000 over all cells" carries no bootstrap alternation here — an
# under-trigger fails SAFE (the layered prose surfaces — planner §9 block,
# critic 10(iii)/12, implementer re-derivation — still fire); deliberate, not
# discovered (#869 plan §4.13).
# The count arm's lookbehind excludes range/scale-dash-preceded numbers —
# "graded 0-100 draws" is judge-scale vocabulary, not a battery (calibration
# false-FAIL on #779 v1); "1000 draws" after whitespace still triggers.
# The two trigger alternations are kept as SOURCE STRINGS so c12 can TYPE
# each trigger line by class (battery vs screen \u2014 evidence never crosses
# class, the #1901 critic MF1) while both existing consumers (c12's trigger
# windows, c32's battery-row classification at check_fit_basis_grounding)
# keep reading the compiled UNION _BATTERY_TRIGGER_RE with no call-site
# change. A line matching BOTH alternations types battery (stricter
# evidence \u2014 the c32 both-families precedent). Screen tokens (#1901): the
# rule's POOL-SCALE PILOTS clause names pool-quadratic candidate screens as
# a covered ~pool-squared kernel class; #1901's screen shipped serial
# (12,339 us/row, ~3.3 h realized vs the 1.0 h sub-budget) past a green
# c12. Compound tokens by design (dedupe/near-dupe require an adjacent
# screen/scan/pass noun) so bare data-prep prose ("deduplicated the prompt
# list") never triggers; CamelCase helper names ("NearDupeGate") carry no
# `[- ]` separator and do not match. `pairwise similarity` starts bare per
# the #1967 task-body sketch.
# CALIBRATION (2026-08-03, #1967): 3,420 persisted plans (tasks/*/*/plans/
# v*.md), c12 + c32 pre/post, task-kind-resolved. Pre-tightening: 24 c12
# verdict flips, 0 c32 flips. Adjudication + tuning decisions:
#   (1) participle exclusion — screen(?!ed)/scan(?!ned)/filter(?!ed):
#       "near-dupe screened (…) against the 1,400 pinned targets" is the
#       banked-corpus DESCRIPTION shape (#1768 v1-v5 all-participle false
#       flips), while own-phase forms are noun/verb-present ("train-pool
#       near-dupe screen:", "near-dupe-screens the remaining train pool").
#   (2) `pass` noun DROPPED from both alternations — corpus hits were
#       absence-descriptions ("whose JSON records NO near-dupe pass",
#       #1775 v2-v5) and linear data prep ("parse-and-dedupe pass", #448);
#       no true positive needed it (under-trigger fails safe).
#   (3) bare `pairwise[- ]similarity` KEPT — 0 corpus flips (6 mentions:
#       fenced regex specs, kind-exempt infra plans, or verdict-unchanged).
#   (4) screen-class polarity dropped FAIL->WARN per plan §8: the residual
#       noun-form banked-screen refs ("banked near-dupe screen: 5-gram
#       Jaccard 0.8" #1901 v1-v3; "n1m near-dupe screen covered val/test
#       only" #779 v7-v10) are token-inseparable from the true own-phase
#       noun form ("train-pool near-dupe screen: build NearDupeGate"), so
#       no tightening zeroes false FAILs while preserving #1901 recall.
#       Batteries keep FAIL. Post-tightening flip table (re-sweep at 3,423
#       plans — the live corpus grew mid-calibration): 14 flips, all
#       ->WARN — 7 true positives (#1738 v1-v4 own Phase-0 screen, the
#       POOL-SCALE incident; #1901 v5-v7 own transposed screen, the recall
#       demonstration) + 7 residual banked-ref WARNs (#1901 v1-v3, #779
#       v7-v10); 0 new FAILs, 0 c32 flips; #1768/#1775/#448 no longer flip.
_BATTERY_ALTERNATION = (
    r"null[- ]?(draws?|batter(y|ies))"
    r"|permutation[- ](tests?|batter(y|ies)|nulls?|draws?)"
    r"|n_(draws|perms)\b"
    r"|(?<![\d\u2013\u2014-])\d{3,}\s+(null[- ])?(draws|permutations|resamples)"
)
_SCREEN_ALTERNATION = (
    r"near[- ]dup(?:e|licate)?s?[- ](?:screen(?!ed)|scan(?!ned)|filter(?!ed))"
    r"|dedup(?:e|lication)?[- ](?:screen(?!ed)|scan(?!ned))"
    r"|pairwise[- ]similarity"
    r"|similarity[- ]screen(?!ed)"
)
_BATTERY_CLASS_RE = re.compile(rf"(?i)\b(?:{_BATTERY_ALTERNATION})")
_SCREEN_TRIGGER_RE = re.compile(rf"(?i)\b(?:{_SCREEN_ALTERNATION})")
_BATTERY_TRIGGER_RE = re.compile(rf"(?i)\b(?:{_BATTERY_ALTERNATION}|{_SCREEN_ALTERNATION})")

# Evidence (i): an explicit two-factor multiplier product where at least one
# factor is draw-bearing ("1000 draws x 24 cells" satisfies; a grid-only
# "34 x 50 x 28" or "layers x 3584" does NOT — the #810 false-PASS class where
# the forgotten draw multiplier is exactly what is absent).
_DRAW_FACTOR = (
    r"(?:\d[\d,_]*\s*(?:null[- ])?(?:draws?|perms?|permutations|resamples)"
    r"|n_(?:draws|perms|boot)\b|draws|perms|permutations|resamples|B\s*=\s*\d{3,})"
)
# The grid side accepts ANY axis factor — a count ("24"), a count + axis
# noun ("6 arms", "3 layers", "~3 quantities"), or a bare axis noun
# ("cells", "batteries") — optionally opened by an approximation / paren
# decoration ("~3", "≈8", "(6 arms"). The load-bearing discriminator is
# the DRAW-BEARING factor, not the grid noun: the #810 false-PASS class
# is a grid-only product with NO draw factor ("34 x 50 x 28",
# "layers x 3584", "6 arms x 3 layers x 16 folds") and still fails; a
# noun whitelist only rots ("batteries"/"quantities" false-FAILed the
# conforming #833 v8 sizing block — #1086).
_GRID_DECOR = r"(?:[~≈(]\s*)?"
_GRID_FACTOR = r"(?:\d[\d,_]*(?:\s+[A-Za-z][\w-]*)?|[A-Za-z][\w-]*)"
# The multiplication token plans actually write: the real multiplication
# sign plus the ASCII fallbacks. The multiplication sign and `*` are
# unambiguous and keep tight/zero-whitespace binding ("50*28"); ASCII `x`
# counts ONLY when standalone w.r.t. word chars — "layer-ma|x| perms",
# "shared-inde|x| draws", "honest_nulls_ma|x|draws", "draws |x|gboost"
# are word-split artifacts, not products (#1099; 27 realized corpus
# false-arith lines, 0 verdict flips on removal). Digit-tight ASCII forms
# ("2x2") also stop counting: every draw-bearing corpus product is spaced
# ("4 draws x 492 cells"), and "the 2x2 draws its factors" is the verb
# false-positive the digit carve-out would re-admit.
_MULT_TOKEN = r"(?:[×*]|(?<!\w)x(?!\w))"  # noqa: RUF001 — the multiplication sign is real plan text
_MULT_ARITH_RE = re.compile(
    rf"(?i)\b(?:{_DRAW_FACTOR}\s*{_MULT_TOKEN}\s*{_GRID_DECOR}{_GRID_FACTOR}"
    rf"|{_GRID_FACTOR}\s*{_MULT_TOKEN}\s*{_DRAW_FACTOR})\b"
)
# Arith-anchored windows (#1086) accepted fail-UNSAFE residual, DISCLOSED: a
# quoted SIBLING's sizing line ("#778's 10,000 draws x 24 cells, batched")
# can anchor its own window and satisfy THIS plan's battery — the same
# residual class as c18's documented residual (f) (non-verbatim paraphrase,
# beyond mechanical defense). Deliberately NO `#\d{2,}` citation guard on
# anchor lines: 192 corpus draw-arithmetic lines carry a same-line #-ref,
# and .claude/rules/plan-compute-sizing.md MANDATES citing a prior-issue
# MEASURED basis beside sizing arithmetic, so the guard would re-create the
# very false-positive class #1086 fixes (guard REJECTED in plan v2 §11).

# Evidence (i-screen): pool-quadratic sizing arithmetic for a SCREEN-class
# window (#1901 critic MF1 — windows are TYPED: battery windows accept
# _MULT_ARITH_RE ONLY, screen windows accept THIS regex ONLY; neither
# class's evidence can satisfy the other, so "576 comparisons" in
# Holm-correction prose can never green a battery and a draw product can
# never green a screen): an explicit squared count ("13,674^2"), the
# n(n-1)/2 pair-count form, a digit-bearing pair/comparison count
# ("1.87e8 pairs", "576 comparisons"), or a pool product / squared pool
# ("pool x pool", "pool squared", "pool" + superscript-two) — ASCII `x`
# under the _MULT_TOKEN word-boundary discipline (#1099).
# _DRAW_FACTOR / _MULT_ARITH_RE are byte-untouched by the screen extension.
_SCREEN_ARITH_RE = re.compile(
    rf"(?i)(?:\d[\d,_]*\s*(?:\^\s*2|²|\*\*\s*2)"
    # MINUS SIGN (U+2212 — the unicode minus prose forms use; spelled as a
    # regex escape so the source stays RUF001-clean); the ASCII hyphen sits
    # FIRST in the class so it reads literal, never a range.
    rf"|n\s*\(\s*n\s*[-\u2212]\s*1\s*\)\s*/\s*2"
    rf"|\d[\d,_]*(?:\.\d+)?(?:e\d+)?\s*(?:pairwise\s+)?(?:pairs|comparisons)\b"
    rf"|pool\s*(?:size\s*)?(?:{_MULT_TOKEN}|squared|²))"
)

# Evidence (ii): a named batched helper or an explicit vectorization
# statement. A token whose only in-window occurrence sits inside a citation /
# path of the rule file does NOT count — citing the rule is not an
# implementation commitment (the filename itself contains "vectorize", so the
# citation tokens are stripped from the window before this search).
_BATCHED_COMMIT_RE = re.compile(
    r"(?i)\b(batched|vectoriz(?:e|ed|es|ation)|subset-sum|GEMM|one\s+(?:masked\s+)?matmul"
    r"|perm_null_draws|randnorm_null_draws|vectorized_mlp_skill"
    # Named batched-screen implementations (#1901; the generic tokens above
    # already cover "batched"/"vectorized" — #1901's own fix was a 175x
    # vectorize): pairwise-distance / sketching / ANN-index helpers.
    r"|cdist|minhash|faiss|LSH)\b"
)
_C12_RULE_CITATION_RE = re.compile(r"\S*vectorize-many-cell-fits\.md\S*")

# Evidence window: ± this many RAW lines around each trigger hit (arithmetic
# legitimately lives in tables/fences adjacent to the battery row).
_C12_WINDOW_LINES = 15


def _trigger_windows(plan: str, trigger_re: re.Pattern[str], window_lines: int) -> list[str]:
    """RAW-text windows (± ``window_lines`` raw lines) around each NON-fenced
    line matching ``trigger_re``. Trigger detection is fence-masked (a
    fence-only example is not a trigger — the line-preserving equivalent of
    searching ``strip_fences(plan)``); each WINDOW is raw text, so evidence
    inside adjacent tables/fences still counts. Shared by c12
    (``_MULT_ARITH_RE`` self-anchor windows, ±15 — the class-typed trigger
    windows use ``_c12_typed_trigger_windows``), c16 (``_C16_EXTRACT_RE``
    ±3; ``_C16_REGEN_RE`` at radius 0 = same-line adjacency), and c24
    (``_C24_TRIGGER_RE``, ±15)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    windows: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not trigger_re.search(line):
            continue
        lo = max(0, i - window_lines)
        hi = min(len(lines), i + window_lines + 1)
        windows.append("\n".join(lines[lo:hi]))
    return windows


def _c12_typed_trigger_windows(plan: str) -> tuple[list[str], list[str]]:
    """c12's fence-masked ±``_C12_WINDOW_LINES``-raw-line trigger windows,
    TYPED by the anchoring line's class (the #1901 critic MF1: battery vs
    screen; evidence never crosses class). A line matching BOTH
    alternations types battery — the stricter evidence class, the c32
    both-families precedent. Returns ``(battery_windows,
    screen_windows)``; window text is raw (evidence inside adjacent
    tables/fences still counts), trigger detection is fence-masked."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    battery: list[str] = []
    screen: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if _BATTERY_CLASS_RE.search(line):
            dest = battery
        elif _SCREEN_TRIGGER_RE.search(line):
            dest = screen
        else:
            continue
        lo = max(0, i - _C12_WINDOW_LINES)
        hi = min(len(lines), i + _C12_WINDOW_LINES + 1)
        dest.append("\n".join(lines[lo:hi]))
    return battery, screen


def _c12_screen_arith_anchor_windows(plan: str) -> list[str]:
    """Screen-class SELF-ANCHOR windows (the #1086 draw-arith anchoring
    mirrored): a non-fenced line carrying pool-quadratic arithmetic anchors
    its own ±``_C12_WINDOW_LINES`` window — the §9 sizing block
    legitimately lives far from the §4/§6 screen registration. A line ALSO
    matching the draw-bearing ``_MULT_ARITH_RE`` types battery (self-anchor
    both->battery typing, the critic's non-blocking note) and is excluded
    here. The caller adds these ONLY when >=1 screen trigger fired in the
    plan, so a stray pair count can never conjure a screen obligation on a
    battery-only plan."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _SCREEN_ARITH_RE.search(line) or _MULT_ARITH_RE.search(line):
            continue
        lo = max(0, i - _C12_WINDOW_LINES)
        hi = min(len(lines), i + _C12_WINDOW_LINES + 1)
        out.append("\n".join(lines[lo:hi]))
    return out


def _c12_class_verdict(
    plan: str, windows: list[str], arith_re: re.Pattern[str], na_tail: str
) -> tuple[str, bool, bool]:
    """One c12 class's satisfaction read (extracted from
    ``check_battery_multiplier`` for C901). Returns ``(verdict, any_arith,
    any_commit)`` with verdict ``"escaped"`` (the class-scoped standalone
    N/A is declared), ``"satisfied"`` (some window of THIS class carries
    both its class-matched arithmetic and a batched commitment), or
    ``"unsatisfied"`` (the any_* flags then drive the missing-detail)."""
    if _standalone_na_declared(plan, na_tail):
        return "escaped", False, False
    any_arith = False
    any_commit = False
    for window in windows:
        has_arith = bool(arith_re.search(window))
        has_commit = bool(_BATCHED_COMMIT_RE.search(_C12_RULE_CITATION_RE.sub("", window)))
        any_arith = any_arith or has_arith
        any_commit = any_commit or has_commit
        if has_arith and has_commit:
            return "satisfied", True, True
    return "unsatisfied", any_arith, any_commit


def _c12_missing_for_class(cls: str, any_arith: bool, any_commit: bool) -> str:
    """The per-class missing-evidence clause for c12's WARN/FAIL detail
    (extracted from ``check_battery_multiplier`` for C901). Never carries a
    literal matching the trigger/arith regexes (the anti-self-satisfy
    discipline the pasted-detail pins enforce)."""
    missing: list[str] = []
    if not any_arith:
        if cls == "battery":
            missing.append(
                "the multiplier arithmetic with a draw-bearing factor "
                "(draws times cells times folds at per-call cost = projected wall)"
            )
        else:
            missing.append(
                "the pool-quadratic sizing arithmetic (candidate-count squared, or half "
                "the count times count-minus-one = pair total, at per-pair cost = "
                "projected wall)"
            )
    if not any_commit:
        missing.append(
            "a batched-implementation commitment (a named batched helper or an explicit "
            "vectorization statement)"
        )
    if not missing:
        missing.append(
            "co-location: the class-matched arithmetic and the batched-implementation "
            f"commitment each appear somewhere, but never together near any {cls}-class "
            "trigger or sizing line"
        )
    return " AND ".join(missing)


def check_battery_multiplier(plan: str, kind: str) -> CheckResult:
    """A plan naming a permutation/bootstrap/null-draw battery — or a
    pool-quadratic candidate SCREEN (#1901: the near-dupe class the
    compute-sizing rule's POOL-SCALE PILOTS clause covers) — must carry,
    NEAR a trigger mention (± 15 raw lines), BOTH (i) CLASS-MATCHED sizing
    arithmetic — a draw-bearing multiplier product for battery-class
    windows (``_MULT_ARITH_RE``), pool-quadratic arithmetic for
    screen-class windows (``_SCREEN_ARITH_RE``); evidence never crosses
    class (the #1901 critic MF1 — the #810 grid-only false-PASS guard is
    preserved verbatim for batteries) — and (ii) a batched-implementation
    commitment (common to both classes). An arithmetic line ALSO anchors
    its own ± 15 evidence window (#1086; a screen-arith self-anchor counts
    only when a screen trigger fired in the plan, and a line matching both
    arithmetic regexes types battery). PASS requires every TRIGGERED class
    independently satisfied — by a window of its own class, or by its
    CLASS-SCOPED standalone escape (``N/A — no draw battery`` excuses
    battery-class windows ONLY; ``N/A — no pool screen`` excuses
    screen-class windows ONLY; the #1901 critic MF2 — the escape check
    lives in the per-class satisfaction, never a pre-walk global return).
    Window-scoped, never document-global — the document-global draft
    demonstrably false-PASSed the motivating incident plan (#810 v1) via
    an unrelated footprint product + helper boilerplate. Polarity
    (2026-08-03 calibration; the plan §8 per-class switch): an unsatisfied
    BATTERY class FAILs (experiment) / WARNs (analysis) — unchanged; an
    unsatisfied SCREEN class WARNs for BOTH kinds, because the corpus
    shows own-phase screen nouns are token-inseparable from banked-screen
    references (#1901 v1-v3 / #779 v7-v10 — see the calibration record
    above the trigger regexes), so screens surface as a WARN the planner
    must resolve-or-carry, never a hard FAIL. SKIP otherwise; a SURFACE
    check per the module's scope discipline — semantic adequacy of the
    arithmetic stays with the Phase 2 critics."""
    cid, name = "c12_battery_multiplier", "battery multiplier + batched commitment"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: battery sizing is an experiment|analysis plan shape")
    battery_trig, screen_trig = _c12_typed_trigger_windows(plan)
    if not battery_trig and not screen_trig:
        return _skip(cid, name, "no permutation/null-draw battery or pool-quadratic screen named")
    # Per-class window sets: trigger windows plus class-typed self-anchor
    # windows (#1086 — the sizing block legitimately lives far from the
    # registration; #833 v8: 58+ lines). Window-scoped discipline is
    # preserved per class: only a line already carrying that class's
    # arithmetic can anchor (a grid-only footprint product never anchors a
    # battery window — the #810 v1 false-PASS class — and a draw product
    # never anchors a screen window), and the batched commitment must still
    # sit within ±_C12_WINDOW_LINES raw lines of the anchor.
    classes: list[tuple[str, list[str], re.Pattern[str], str]] = []
    if battery_trig:
        classes.append(
            (
                "battery",
                battery_trig + _trigger_windows(plan, _MULT_ARITH_RE, _C12_WINDOW_LINES),
                _MULT_ARITH_RE,
                r"no draw battery",
            )
        )
    if screen_trig:
        classes.append(
            (
                "screen",
                screen_trig + _c12_screen_arith_anchor_windows(plan),
                _SCREEN_ARITH_RE,
                r"no pool screen",
            )
        )
    sat_parts: list[str] = []
    failing: list[tuple[str, bool, bool]] = []
    for cls, windows, arith_re, na_tail in classes:
        verdict, any_arith, any_commit = _c12_class_verdict(plan, windows, arith_re, na_tail)
        if verdict == "escaped":
            sat_parts.append(f"{cls}: explicit N/A declared")
        elif verdict == "satisfied":
            sat_parts.append(
                f"{cls}: a window carries both the class-matched sizing arithmetic and a "
                "batched-implementation commitment"
            )
        else:
            failing.append((cls, any_arith, any_commit))
    if not failing:
        return _pass(cid, name, "; ".join(sat_parts))
    class_details = [
        f"the {cls} class is missing {_c12_missing_for_class(cls, any_arith, any_commit)}"
        for cls, any_arith, any_commit in failing
    ]
    detail = (
        f"plan names a permutation/bootstrap/null battery and/or a pool-quadratic candidate "
        f"screen but {'; '.join(class_details)}"
        " — a named battery defaults to a serial per-draw loop (#778: ~15 h realized vs 1 h"
        " planned; #810: 308x) and a named screen defaults to a serial per-candidate loop"
        " (#1901: ~3.3 h realized vs the 1.0 h sub-budget); see"
        " .claude/rules/vectorize-many-cell-fits.md +"
        " .claude/rules/plan-compute-sizing.md (POOL-SCALE PILOTS). If a mention is"
        " incidental, declare the CLASS-SCOPED escape on its own line, unwrapped (no"
        " backticks/quotes): 'N/A — no draw battery' excuses battery-class windows ONLY;"
        " 'N/A — no pool screen' excuses screen-class windows ONLY"
    )
    battery_failing = any(cls == "battery" for cls, _a, _c in failing)
    if kind == "analysis" or not battery_failing:
        # Screen-class-only misses carry WARN polarity (2026-08-03 calibration,
        # plan §8 per-class switch) — batteries keep FAIL for experiment plans.
        return _warn(cid, name, detail)
    return _fail(cid, name, detail)


# ─── Check 13 — empirical-null gate p-floor attainability (conditional) ────

# Gate alpha: a decimal alpha DIRECTLY after the comparator (comparator
# captured — strictness matters at the floor == alpha boundary). A
# fraction-form self-consistent floor gate ("p ≤ 1/(15+1) ≈ 0.06", #816 v5
# Exp-4) must NOT match: "1/" blocks the decimal, and the "≈ 0.06" is not
# comparator-adjacent.
_C13_P_ALPHA_RE = re.compile(r"(?i)\bp(?:-?values?)?\s*(≤|<=|<)\s*\*{0,2}`?(0?\.\d+)`?")
_C13_EMPIRICAL_RE = re.compile(r"(?i)\bempirical\b")
# Registered-gate section: any ENCLOSING heading matching the c8 success/kill
# families or an Evaluation heading. Lines elsewhere (Prior Work recaps,
# TL;DR) are not registrations — under-trigger fails safe (critics review).
_C13_GATE_SECTION_RE = re.compile(
    r"(?i)success criteri|acceptance criteri|decision rule|decision gate"
    r"|kill[- ]criteri|abort criteri|stop criteri|\bevaluation\b"
)
# On-gate-line draws-scope qualifier ("(n_draws ≥ 50: ...)" — the #816 v6 fix
# shape): families below K are OUTSIDE the gate's own declared scope. The
# DRAWS-EXPLICIT token is REQUIRED: a bare `n ≥ K` (e.g. "n ≥ 20 prompts per
# probe" — a sample-size clause on the gate line) must NOT set the scope, or
# it silently descopes every small-n_draws family and emits an affirmative
# false-PASS on the exact #816 class this check exists to catch.
_C13_SCOPE_RE = re.compile(r"(?i)\bn_(?:draws|perms)\w*\s*(?:≥|>=)\s*(\d+)")
# Family vocabulary on the gate line = the tie is unambiguous (the gate
# quantifies over null families) → FAIL-capable; absent → WARN cap.
_C13_FAMILY_RE = re.compile(r"(?i)famil")
# Per-declaration exclusion: a family row/line declaring itself outside the
# test set is dropped (v5/v6 contaminated-reference row; v6 "outside the BH").
_C13_EXCLUDE_RE = re.compile(
    r"(?i)contaminated|reference only|descriptive|excluded"
    r"|not (?:in|included|counted)|outside the (?:BH|test)"
)
# n_draws declarations, prose/kwarg forms: n_draws=K, n_draws: K,
# n_draws_isotropic=200, n_perms=500. ("n_draws ≥ 50" and "(n_draws+1)" do
# not match — comparator/paren, not =/:.)
_C13_NDRAWS_KWARG_RE = re.compile(r"(?i)\b(n_(?:draws|perms)\w*)\s*[=:]\s*(\d+)")

# Known accepted under-triggers (mirroring the c12 precedent): (a) a gate
# registered outside any success/kill/decision/evaluation-titled section;
# (b) a gate whose `p <= alpha` wraps across lines or uses `%`/LaTeX `\le`;
# (c) "empirical" absent from the gate line; (d) draw counts declared only as
# bare prose ("15 draws") without an `n_draws` label. (a)-(d) fail SAFE
# (under-trigger → SKIP; the plan still reaches the fact-checker + critic
# ensemble, whose statistics lens caught the original #816 incident). ONE
# known fail-UNSAFE direction: (e) a hard-wrapped gate whose `(n_draws ≥ K)`
# qualifier lands on the NEXT line is gate-detected without its qualifier →
# false-FAIL on a legitimately scoped gate. Accepted: repo plans favor long
# single lines (v5/v6 both do), the corpus-sweep calibration bounds it, and a
# false-FAIL costs 1-2 mechanical planner bounces with the PASS-with-override
# valve as the escape (adversarial-planner SKILL.md Phase 1.5.0) — §4.5 is
# NOT all-fails-safe.


def _n_draws_declarations(plan: str) -> list[tuple[str, int]]:
    """Deduplicated ``(label, n_draws)`` pairs harvested from the RAW plan:
    (1) markdown-table columns whose header cell CONTAINS ``n_draws`` /
    ``n_perms`` after bold/backtick stripping (v5's twin ``n_draws (Exp-2)`` /
    ``n_draws (Exp-4)`` columns both match; ALL matching columns per table are
    collected), and (2) prose/kwarg forms (``n_draws=K``, ``n_perms: K``,
    ``n_draws_isotropic=200``). Deliberately raw text — declarations
    legitimately live in tables and fenced config blocks (#816 v6's kwargs are
    fenced). A row/line matching ``_C13_EXCLUDE_RE`` (outside-the-test-set
    vocabulary) is dropped; a non-numeric cell is skipped."""
    lines = plan.splitlines()
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(label: str, n: int) -> None:
        key = (label, n)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    # (1) Table columns — a sibling of the c1 `_source_column_cells` walk,
    # with a contains-predicate on the header cell + multi-column collection.
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if not (header.startswith("|") and sep.startswith("|") and _TABLE_SEP_RE.fullmatch(sep)):
            i += 1
            continue
        header_cells = [c.strip().strip("*`").strip().casefold() for c in _split_table_row(header)]
        cols = [j for j, c in enumerate(header_cells) if "n_draws" in c or "n_perms" in c]
        k = i + 2
        while k < len(lines) and lines[k].strip().startswith("|"):
            row_text = lines[k]
            if cols and not _C13_EXCLUDE_RE.search(row_text):
                row = _split_table_row(row_text)
                # replace("**", "") drops INTERIOR bold markers (e.g.
                # `**Cross-trait** (ref)`) that a bare strip("*") keeps.
                label = row[0].replace("**", "").strip("*").strip() if row else ""
                for col in cols:
                    if col >= len(row):
                        continue
                    m = re.search(r"\d[\d,_]*", row[col])
                    if m:
                        add(label, int(m.group(0).replace(",", "").replace("_", "")))
            k += 1
        i = k
    # (2) Prose/kwarg declarations.
    for line in lines:
        if _C13_EXCLUDE_RE.search(line):
            continue
        for m in _C13_NDRAWS_KWARG_RE.finditer(line):
            add(m.group(1), int(m.group(2)))
    return pairs


def _c13_registered_gates(plan: str) -> list[dict]:
    """Registered empirical-p gate lines: non-fenced lines inside a
    success/kill/evaluation-titled section carrying "empirical" + at least
    one decimal alpha directly after ``p <=`` / ``p <``. Per gate: the
    stripped line, the MIN alpha on the line (a gate requiring the most
    stringent of several alphas is unattainable if the floor exceeds the
    smallest), whether the min-alpha comparator is strict ``<``, the on-line
    draws-scope qualifier K (or None), and whether family vocabulary is on
    the line."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    gates: list[dict] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if not any(h.line <= i < h.end and _C13_GATE_SECTION_RE.search(h.text) for h in headings):
            continue
        if not _C13_EMPIRICAL_RE.search(line):
            continue
        matches = list(_C13_P_ALPHA_RE.finditer(line))
        if not matches:
            continue
        alphas: list[tuple[Fraction, bool]] = []
        for m in matches:
            a = m.group(2)
            alphas.append(
                (Fraction("0" + a) if a.startswith(".") else Fraction(a), m.group(1) == "<")
            )
        min_alpha = min(a for a, _ in alphas)
        strict = any(s for a, s in alphas if a == min_alpha)
        scope_m = _C13_SCOPE_RE.search(line)
        gates.append(
            {
                "line": line.strip(),
                "alpha": min_alpha,
                "strict": strict,
                "scope": int(scope_m.group(1)) if scope_m else None,
                "family": bool(_C13_FAMILY_RE.search(line)),
            }
        )
    return gates


def _standalone_na_declared(plan: str, tail_re: str) -> bool:
    """True when ``N/A — <tail_re>`` appears as a deliberate STANDALONE
    declaration line (leading list/blockquote markers stripped), never
    doc-global: a FAIL detail quotes its escape phrase as a remedy option,
    and this project's convention pastes verifier/bounce text into revised
    plans verbatim — a substring match would let a bounced plan self-escape
    re-verification (the #810 spurious-satisfaction structure, one polarity
    over). NA_RE opens with an inline (?i), so it must sit at pattern
    position 0 — per-line re.match satisfies that; never prepend a prefix
    to NA_RE (py3.11+ rejects mid-pattern global flags). Shared by the
    checks' standalone-N/A escapes (the Supersede rule: one copy of the job).

    Wrapped declarations (a backtick/quote-wrapped paste of a remedy's
    quoted form) are DELIBERATELY unrecognized (#1238 reasoned no-change):
    the adversarial-planner SKILL.md canonical-phrases block renders its
    escape phrases backtick-wrapped at line start, nearly all of them
    helper-routed since the #1237/#1262 migrations, so every
    trailing-tolerant wrapper widening lets a verbatim block paste
    self-declare many checks' escapes at once; requiring a balanced
    closing wrapper does
    not discriminate (the block's wrappers are balanced by construction),
    and the strict phrase-alone-on-line variant rejects the one shape
    that measurably BOUNCED (#1090 plans/v1.md:369, trailing scope
    prose) while its target idiom (wrapped-alone lines, a real corpus
    habit) is covered at the source by the SKILL.md unwrapped contract.
    The most realistic hazard is not even a whole-block paste: a
    single-phrase bulleted bounce-brief line ("- <wrapped phrase> -
    <remedy prose>") is byte-shaped identically to a legitimate
    declaration. Declare escapes UNWRAPPED. Pinned:
    tests/test_verify_plan.py skillmd/wrapped pins.
    """
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if re.match(NA_RE + tail_re, line.lstrip(" \t>*-")):
            return True
    return False


def _c13_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no empirical-null gate`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale)."""
    return _standalone_na_declared(plan, r"no empirical[- ]null gate")


def _c13_evaluate(gates: list[dict], decls: list[tuple[str, int]]) -> dict:
    """Per-gate attainability arithmetic. Offender iff floor > alpha OR
    (floor == alpha AND the gate comparator is strict ``<`` — then the gate
    is unattainable, not boundary); floor == alpha under ``<=`` is boundary.
    A nonpositive alpha (``p ≤ 0.00``) is in-domain: floor = 1/(n+1) > 0 ≥
    alpha for EVERY draw count, so the same arithmetic classifies every
    in-scope declaration an offender (fail_capable per the normal
    family-vocab rule; the alpha ≤ 0 remedy lives in the detail builder).
    A gate whose scope qualifier excludes EVERY declaration is vacuous (an
    empty in-scope set must not yield an affirmative PASS with an undefined
    min)."""
    offenders: list[tuple[dict, str, int, Fraction]] = []
    boundary: list[tuple[str, int]] = []
    fail_capable = False
    vacuous_scope = False
    min_in_scope: int | None = None
    for g in gates:
        in_scope = [d for d in decls if g["scope"] is None or d[1] >= g["scope"]]
        if g["scope"] is not None and not in_scope:
            vacuous_scope = True
            continue
        for label, n in in_scope:
            if min_in_scope is None or n < min_in_scope:
                min_in_scope = n
            floor = Fraction(1, n + 1)
            if floor > g["alpha"] or (floor == g["alpha"] and g["strict"]):
                offenders.append((g, label, n, floor))
                fail_capable = fail_capable or g["family"]
            elif floor == g["alpha"]:
                boundary.append((label, n))
    return {
        "offenders": offenders,
        "boundary": boundary,
        "fail_capable": fail_capable,
        "vacuous_scope": vacuous_scope,
        "min_in_scope": min_in_scope,
    }


def _c13_offender_detail(offenders: list[tuple[dict, str, int, Fraction]]) -> str:
    """Bounded FAIL/WARN detail: the first offending gate line (truncated)
    + its alpha, at most 6 offenders, and the remedy menu (raise n_draws to
    >= ceil(1/alpha) for a clean PASS — n = 1/alpha - 1 exactly lands on the
    boundary WARN). A nonpositive alpha (e.g. a registered ``p ≤ 0.00`` —
    the limiting case of the unattainable-gate class) gets a dedicated
    remedy instead of ``ceil(1/alpha)``, which would ZeroDivisionError on
    ``Fraction(1, 0)`` — a parseable gate must never crash the module."""
    g0 = offenders[0][0]
    alpha0: Fraction = g0["alpha"]
    # Display-dedupe on (label, n): two gates sharing an offending family
    # would otherwise list it twice and push distinct offenders past the cap.
    uniq: list[tuple[str, int, Fraction]] = []
    for _, label, n, floor in offenders:
        if (label, n, floor) not in uniq:
            uniq.append((label, n, floor))
    shown = ", ".join(
        f"{label} n_draws={n} → floor {floor.numerator}/{floor.denominator} ≈ {float(floor):.3g}"
        for label, n, floor in uniq[:6]
    )
    if len(uniq) > 6:
        shown += ", …"
    if alpha0 <= 0:
        remedy = (
            "alpha ≤ 0 — no finite n_draws attains it (the p-floor 1/(n_draws+1) is "
            "positive for every draw count); raise the alpha or fix the gate"
        )
    else:
        remedy = (
            f"raise n_draws to ≥ {math.ceil(1 / alpha0)} for a clean PASS "
            "(n = 1/alpha - 1 exactly lands on the floor == alpha boundary WARN)"
        )
    return (
        f'plan registers an empirical-p gate ("{g0["line"][:90]}", alpha={float(alpha0):g}) '
        f"over families whose p-floor 1/(n_draws+1) exceeds alpha: {shown} — the gate is "
        f"structurally unattainable (#816 v5 class); {remedy}, scope the gate "
        "(e.g. 'n_draws ≥ 50'), mark the family outside the test set on its row, or declare "
        "'N/A — no empirical-null gate' on its own line, unwrapped (no backticks/quotes)"
    )


def check_empirical_gate_attainability(plan: str, kind: str) -> CheckResult:
    """A registered empirical-null gate (a success/kill/evaluation-section
    line requiring p ≤ alpha against null families) must be ATTAINABLE for
    every in-scope declared family: p_floor = 1/(n_draws+1) ≤ alpha.
    Necessary-condition logic only — under BH the effective per-test
    thresholds are ≤ alpha, so floor > alpha is conservative-correct; BH-m
    arithmetic, family-set semantics, and joint satisfiability stay with the
    Statistics critic (c8's form-only charter). FAIL (experiment) / WARN
    (analysis) / WARN on ambiguous tie or floor == alpha under a non-strict
    comparator / SKIP otherwise; escape via a standalone ``N/A — no
    empirical-null gate`` line — honored (SKIP path) only when no gate is
    detected; when the escape co-occurs with a detected gate the check WARNs
    instead of PASSing (regardless of whether n_draws declarations exist),
    so the escape can never mask attainability verification of a present
    gate (#1258, the #1223 c20 rule). Incident: #816 v5 (gate p ≤ 0.05 over
    families with n_draws=2/5 → floors 1/3, 1/6; caught only by the Codex
    statistics critic)."""
    cid, name = "c13_empirical_gate_attainability", "empirical-null gate p-floor attainability"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: registered empirical-null gates are an experiment|analysis plan shape",
        )
    gates = _c13_registered_gates(plan)
    if not gates:
        return _skip(cid, name, "no registered empirical-p gate detected")
    if _c13_na_escape_declared(plan):
        # #1258 (the #1223 c20 port): this branch is reachable ONLY when a
        # registered empirical-p gate WAS detected (the no-gate case SKIPs
        # above) — a PASS here masks the p-floor attainability verification
        # c13 exists to run. WARN, not FAIL: the gate harvest may be a false
        # positive on quoted guidance, so the escape stays non-blocking and
        # the reviewers adjudicate.
        return _warn(
            cid,
            name,
            "the standalone `N/A — no empirical-null gate` escape co-occurs with "
            f"{len(gates)} registered empirical-p gate line(s) (first: "
            f"{gates[0]['line'][:90]!r}) — the escape is reserved for gate-free "
            "plans and would mask attainability verification of the detected "
            "gate (#1258, the #1223 c20 rule); remove the N/A line (the gate is "
            "then verified, or SKIPs as not-computable when no n_draws "
            "declarations exist), or fence/remove the gate-shaped prose the "
            "detector matched if it is quoted guidance rather than this plan's "
            "own registration",
        )
    decls = _n_draws_declarations(plan)
    if not decls:
        return _skip(
            cid,
            name,
            "empirical-p gate present but no per-family n_draws declarations found — "
            "attainability not computable at the plan surface",
        )
    ev = _c13_evaluate(gates, decls)
    if ev["offenders"]:
        detail = _c13_offender_detail(ev["offenders"])
        if kind == "analysis":
            return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
        if not ev["fail_capable"]:
            return _warn(
                cid,
                name,
                detail + " — ambiguous tie: no family vocabulary on any offending gate line; "
                "verify the flagged draw counts are in the gate's test set",
            )
        return _fail(cid, name, detail)
    if ev["boundary"]:
        label, n = ev["boundary"][0]
        return _warn(
            cid,
            name,
            f"p-floor equals the registered alpha exactly ({label} n_draws={n} → floor "
            f"1/{n + 1} = alpha) — attainable only when the real statistic beats every "
            "draw; state the floor next to the verdict",
        )
    if ev["vacuous_scope"]:
        return _warn(
            cid,
            name,
            "the gate's scope qualifier (n_draws ≥ K) excludes every declared family — "
            "attainability not computable for any in-scope family; verify the gate's "
            "in-scope families are declared",
        )
    min_in_scope = ev["min_in_scope"]
    return _pass(
        cid,
        name,
        f"min in-scope n_draws={min_in_scope} → p-floor 1/{(min_in_scope or 0) + 1} ≤ "
        "registered alpha (attainable in form; adequacy stays with the Statistics critic)",
    )


# ─── Check 14 — hypothesis confirm/falsify branch coherence (WARN-only) ────

# Branch anchors: `**Confirm:**`, `**Confirm (ridge stands):**`,
# `**Confirm-the-null:**`, `**Falsify:**`, `**Falsify (positive surprise):**`
# — all observed corpus shapes.
_BRANCH_ANCHOR_RE = re.compile(r"(?i)\*\*\s*(confirm|falsif)")
# Shared bounded token: a normalized `var = value` pair present in BOTH
# branch segments (the #922 H4 `k = 32` horizon shape). Comparator-bearing
# bounds (`k ≤ 4`) are deliberately NOT harvested — requiring exact-pair
# identity in both segments is the main false-positive guard.
_BOUND_TOKEN_RE = re.compile(r"\b([A-Za-z]\w{0,8})\s*=\s*(\d+(?:\.\d+)?)\b")
# Tendency-class comparator (does not pin an end-state). Deliberately
# minimal: a bare "declines" without "toward" is an accepted false negative
# (prefer false negatives); "approaches"/"converges" excluded in v1
# ("two approaches" false-fires on the noun).
_TENDENCY_RE = re.compile(r"(?i)\btowards?\b")
# State-class comparator (pins a region through the horizon).
_STATE_RE = re.compile(
    r"(?i)\b(?:stays?|remains?|holds?)\s+(?:strictly\s+)?(?:above|below|at|within)\b"
)
# Vague layer-scope tokens ("mid/late layers", "most layers", incl. "at most
# layers"). "a majority of layers" is deliberately EXCLUDED (a quantifier
# over a universe; in the observed corpus it co-occurs with a pinned one).
_VAGUE_SCOPE_RE = re.compile(
    r"(?i)\b(?:(?:early|mid|middle|late|deep|shallow)"
    r"(?:\s*[/-]\s*(?:early|mid|middle|late|deep|shallow))?|most)\s+layers\b"
)
# Pinned-anchor escape (same block): "layers 1-28" (any dash), "layer 20",
# "layers {18, 21}", "L18", the pre-registered layer symbol (script small l,
# U+2113, followed by *), or the literal "pre-registered".
_PINNED_SCOPE_RE = re.compile(
    r"(?i)\blayers?\s*\d|\blayers?\s+\{|\bL\d{1,2}\b|\u2113\*|\bpre-registered\b"
)
# Per-hypothesis block starts: top-level list items (sub-headings are
# detected via _HEADING_RE).
_C14_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*]|\d+\.)\s")
# Bold span used to label an offending block in the WARN detail.
_C14_BOLD_LABEL_RE = re.compile(r"\*\*([^*\n]{1,60})\*\*")


def _hypothesis_blocks(section_text: str) -> list[str]:
    """Split a (fence-stripped) hypothesis-section text into per-hypothesis
    blocks at top-level list-item starts and heading lines; continuation
    lines join the preceding block. The section heading line starts the
    first block (it carries no branch anchors, so it is ignored downstream).
    Matches the observed corpus: one bullet per `**H<k>**` (#922 v2, #841
    v12, #810 v6 all use single-bullet hypothesis blocks)."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in section_text.splitlines():
        if _C14_LIST_ITEM_RE.match(line) or _HEADING_RE.match(line.strip()):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]


def _confirm_falsify_segments(block: str) -> tuple[str, str] | None:
    """``(confirm_segment, falsify_segment)`` for a hypothesis block, or
    ``None`` when the block has no falsify anchor (nothing to compare —
    c8 owns branches missing entirely; a lone ``**Confirm`` block is also
    ignored). Falsify segment = first falsify anchor to the next anchor or
    block end. Confirm segment = explicit confirm anchor to the next anchor
    when one exists; otherwise the block text BEFORE the falsify anchor
    (the hypothesis statement itself is the implicit confirm branch — the
    #922 H4 shape, which has no ``**Confirm:**`` label)."""
    anchors = list(_BRANCH_ANCHOR_RE.finditer(block))
    falsifies = [m for m in anchors if m.group(1).casefold().startswith("falsif")]
    if not falsifies:
        return None
    f0 = falsifies[0]
    after_f = [m for m in anchors if m.start() > f0.start()]
    falsify_seg = block[f0.start() : after_f[0].start() if after_f else len(block)]
    confirms = [m for m in anchors if m.group(1).casefold().startswith("confirm")]
    if confirms:
        c0 = confirms[0]
        after_c = [m for m in anchors if m.start() > c0.start()]
        confirm_seg = block[c0.start() : after_c[0].start() if after_c else len(block)]
    else:
        confirm_seg = block[: f0.start()]
    return confirm_seg, falsify_seg


def _shared_bound_tokens(confirm_seg: str, falsify_seg: str) -> list[str]:
    """Normalized ``var = value`` pairs present in BOTH segments, rendered
    as sorted ``"var = value"`` strings (identity on the normalized pair,
    whitespace-insensitive)."""

    def _toks(seg: str) -> set[tuple[str, str]]:
        return {(m.group(1).casefold(), m.group(2)) for m in _BOUND_TOKEN_RE.finditer(seg)}

    return sorted(f"{var} = {val}" for var, val in _toks(confirm_seg) & _toks(falsify_seg))


def _c14_block_label(block: str) -> str:
    """Short human label for a hypothesis block: the first bold span that is
    not itself a branch anchor (e.g. ``**H4 (rollout).**``), else the first
    line truncated."""
    for m in _C14_BOLD_LABEL_RE.finditer(block):
        if not re.match(r"(?i)\s*(?:confirm|falsif)", m.group(1)):
            return f"**{m.group(1)}**"
    first_line = block.strip().splitlines()[0] if block.strip() else "(unnamed)"
    return first_line[:60]


def check_hypothesis_branch_coherence(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: hypothesis confirm/falsify branch coherence.
    Two token-level offender predicates per anchor-bearing hypothesis
    block: (a) a jointly-satisfiable tendency-vs-state comparator pair on a
    shared bounded ``var = value`` token across the confirm/falsify
    segments ("decays toward ... by k = 32" confirm vs "stays above ...
    through k = 32" falsify — one above-but-declining curve satisfies
    both); (b) a vague layer-scope token ("mid/late layers", "most layers")
    with no pinned layer list/numeral in the same block. NEVER FAILs — a
    heuristic text check must not hard-block a legitimately-worded plan;
    joint satisfiability beyond these two token shapes stays with the
    Statistics critic (c8's form-only charter). Crisp state-vs-state pairs
    (``≤`` vs ``>``, win-count comparators — the #841 v12 / #810 v6 shapes)
    carry no tendency token and stay silent. Incident: #922 v2 H4 (caught
    only by the Codex statistics critic; the same defect class reached
    execution in #488 round 10)."""
    cid, name = "c14_hypothesis_branch_coherence", "hypothesis branch coherence"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: hypothesis blocks are an experiment|analysis plan shape"
        )
    section = section_text_by_keywords(plan, ("hypothesis",))
    if section is None:
        return _skip(cid, name, "no hypothesis section detected")
    text = strip_fences(section)
    anchored: list[tuple[str, tuple[str, str]]] = []
    for block in _hypothesis_blocks(text):
        segments = _confirm_falsify_segments(block)
        if segments is not None:
            anchored.append((block, segments))
    if not anchored:
        return _skip(
            cid, name, "hypothesis section present but no **Confirm/**Falsify branch anchors"
        )
    offenders: list[str] = []
    for block, (confirm_seg, falsify_seg) in anchored:
        clauses: list[str] = []
        shared = _shared_bound_tokens(confirm_seg, falsify_seg)
        if shared:
            c_tend = _TENDENCY_RE.search(confirm_seg)
            f_state = _STATE_RE.search(falsify_seg)
            c_state = _STATE_RE.search(confirm_seg)
            f_tend = _TENDENCY_RE.search(falsify_seg)
            pair: tuple[str, str] | None = None
            if c_tend and f_state:
                pair = (c_tend.group(0), f_state.group(0))
            elif c_state and f_tend:
                pair = (c_state.group(0), f_tend.group(0))
            if pair:
                clauses.append(
                    f"(a) comparator-pair — confirm says '{pair[0]}' while falsify says "
                    f"'{pair[1]}' on shared token '{shared[0]}', jointly satisfiable by "
                    "one outcome"
                )
        vague = _VAGUE_SCOPE_RE.search(block)
        if vague and not _PINNED_SCOPE_RE.search(block):
            clauses.append(
                f"(b) vague-scope — '{vague.group(0)}' with no pinned layer list/numeral "
                "in the block"
            )
        if clauses:
            offenders.append(f"block '{_c14_block_label(block)}': " + "; ".join(clauses))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(anchored)} hypothesis block(s) scanned; no c14 trigger detected "
            "(no jointly-satisfiable comparator pair, no unpinned vague-scope token)",
        )
    extra = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
    detail = (
        "; ".join(offenders[:3])
        + extra
        + " — tighten the branch comparators (e.g. '≤ vs >') and/or pin the layer set; "
        "semantic verdict stays with the Statistics critic"
    )
    return _warn(cid, name, detail)


# ─── Check 15 — fail-loud acceptance claim backed by a committed test ──────

# Trigger anchor: an acceptance/success-criteria mention. Deliberately
# NARROWER than c8's _SUCCESS_RE — "decision rule|decision gate" is excluded
# because gates/failure-mode sections carry failure-MODE descriptions
# ("silently provisioned" risk rows), not acceptance claims (corpus probe on
# all 230 infra|batch plans, task #932).
_FAILLOUD_ANCHOR_RE = re.compile(r"(?i)acceptance criteri|success criteri")

# Claim vocabulary, scanned over the fence-stripped window below an anchor.
# Letter-lookarounds (not \b) around "loud" exclude "cloud"/"Cloudflare".
# The bare transitive "raises" is deliberately absent ("raises the
# concurrency cap" is a real infra acceptance sentence); the narrow raise
# forms + swallow/silent cover the genuine raise-claims in the corpus.
_FAILLOUD_CLAIM_RE = re.compile(
    r"(?i)fail[- ]?loud|fail[- ]?fast"
    r"|(?<![a-z])loud(?:ly)?(?![a-z])"
    r"|swallow"
    r"|silent"
    r"|warn(?:ing)?[- ]and[- ]continue"
    r"|except\s+(?:Exception|BaseException)|bare\s+except|try\s*/\s*except|except\s*:"
    r"|(?:must|shall|should)\s+raise\b"
    r"|raises?\s+(?:an?\s+)?[A-Z][A-Za-z]*(?:Error|Exception)\b|raises?\s+SystemExit"
    r"|non-?zero\s+exit|exits?\s+non-?zero"
)

# Committed-test evidence vocabulary. Letter-lookarounds so identifier-
# internal tokens match (test_length_mismatch_raises, test_no_silent_swallow).
# "exit code" is deliberately absent — "pytest ... exit code 0" is a generic
# success-path verification line and self-certified in the corpus probe.
_FAILLOUD_TEST_EVIDENCE_RE = re.compile(
    r"(?i)(?<![a-z])rais(?:e|es|ed|ing)(?![a-z])|swallow|silent"
    r"|fail[- ]?loud|fail[- ]?fast|(?<![a-z])loud(?![a-z])"
    r"|(?<![a-z])except(?![a-z])|systemexit|non-?zero\s+exit|exits?\s+non-?zero"
)

# Evidence-side exclusion: a run-book grep gate over a test file would
# otherwise self-certify (`grep -n 'except Exception' tests/test_foo.py`).
_FAILLOUD_GREP_LINE_RE = re.compile(r"(?i)\bgrep\b")

# Evidence route 2 (#1306): a quoted pytest.raises control with a named
# exception is intrinsically fail-loud test vocabulary — accepted WITHOUT a
# same-line test_ identifier (#1296: the negative-control literal sits on
# its own hard-wrapped line). Corpus-audited 2026-07-14: flips exactly the
# two #1296 incident plans out of 119 standing infra|batch WARNs.
_FAILLOUD_PYTEST_RAISES_RE = re.compile(r"pytest\.raises\(\s*[\w.]+")

# Evidence route 3 (#1306): a deliberate labeled pin line opens a FORWARD,
# paragraph-bounded scan for the test identifier — the labeled-line
# convention (c31 precedent: an unlabeled c15-style loose window
# false-satisfied all 9 of c31's incident plan versions, so the label is
# load-bearing), made wrap-tolerant for long test paths (#1296 v2's
# 105-char path forced the wrap). A blank line ends the paragraph; the
# scan is capped (incident paragraph = 5 lines; 8 = 5 + margin).
_FAILLOUD_PIN_LABEL_RE = re.compile(r"(?i)\bfail[- ]?loud (?:pin|acceptance)\b[^:\n]{0,40}:")
_FAILLOUD_PIN_SCAN_LINES = 8

# Anchor carriers that never bind: §0.0 TL;DR / §0 Plan Summary restate
# criteria as summary prose (same rationale as c8's _tldr_ranges exclusion).
_FAILLOUD_SUMMARY_HEAD_RE = re.compile(r"(?i)tl;dr|plan summary|^(?:§\s*)?0(?:\.0)?\b")

# Anchor carriers that never bind (2): Risks / Failure-Modes sections carry
# failure-MODE narration ("post_event raising ValueError is fail-loud", "will
# be caught ... fails loud"), not acceptance claims — the same rationale that
# excluded decision rule|gate from the anchor (#932 corpus probe, comment
# above _FAILLOUD_ANCHOR_RE). BOTH alternation branches are GROUPED under the
# start anchor + optional section numbering (a bare `|failure[- ]modes?`
# branch would match anywhere in the heading and silently exclude genuine
# acceptance sections), so an acceptance heading merely containing "risk" or
# "failure mode" mid-heading is NOT excluded. 11 of 16 historical noise fires
# were this class (#1291; founding incident #1275 v1).
_FAILLOUD_RISKS_HEAD_RE = re.compile(
    r"(?i)^(?:§\s*)?(?:\d+(?:\.\d+)*[.)]?\s*)?(?:risks?\b|failure[- ]modes?\b)"
)

_FAILLOUD_WINDOW_LINES = 30


def _failloud_claim_hits(plan: str) -> list[tuple[str, str]]:
    """(section heading, matched vocabulary) per acceptance/success anchor
    whose 30-line window carries a fail-loud claim. Anchors in fences, in
    §0/TL;DR/Plan-Summary regions, in Risks/Failure-Modes sections (risk rows
    narrate failure MODES, not acceptance claims — 11/16 historical noise
    fires, #1291, founding incident #1275 v1), or with an H1/preamble carrier
    are dropped (corpus-probe noise classes, tasks #932 + #1291). The claim
    window is built from the document-global fence mask (a window slice can
    no longer mis-parse when it starts inside a fence) with `grep`-bearing
    lines excluded LINE-SCOPED — a grep line narrates tooling semantics
    ("`grep -c` exits nonzero"), not the plan's own acceptance claim (the
    remaining 5/16 noise fires, #1275 v2); a real claim on any non-grep line
    in the window still triggers."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    hits: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        if mask[i] or not _FAILLOUD_ANCHOR_RE.search(line):
            continue
        h = _innermost_section(headings, i)
        if h is None or h.level < 2 or _FAILLOUD_SUMMARY_HEAD_RE.search(h.text.strip()):
            continue
        if _FAILLOUD_RISKS_HEAD_RE.search(h.text.strip()):
            continue
        end = min(h.end, i + 1 + _FAILLOUD_WINDOW_LINES)
        window = "\n".join(
            ln
            for ln, fenced in zip(lines[i:end], mask[i:end], strict=True)
            if not fenced and not _FAILLOUD_GREP_LINE_RE.search(ln)
        )
        m = _FAILLOUD_CLAIM_RE.search(window)
        if m:
            hits.append((h.text, m.group(0)))
    return hits


def _failloud_test_evidence_lines(plan: str) -> list[str]:
    """RAW-plan lines naming a committed fail-loud-exercising test, three
    routes (#913/#1306): (1) a ``test_`` identifier co-located with
    fail-loud vocabulary on ONE line (the original scan); (2) a quoted
    ``pytest.raises(<Exception>`` control — intrinsically test + raise
    vocabulary, no identifier needed; (3) a ``Fail-loud pin:``-style
    labeled line whose contiguous paragraph names a ``test_`` identifier
    within ``_FAILLOUD_PIN_SCAN_LINES`` lines (wrap-tolerant labeled route;
    unlabeled ±k windows false-satisfy — 21-29 of 119 corpus WARNs at
    k=1-2 vs 2 genuine, measured 2026-07-14 — so the label is
    load-bearing, the c31 lesson). Grep-command lines never count on any
    route."""
    lines = plan.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if _FAILLOUD_GREP_LINE_RE.search(line):
            continue
        if _TEST_IDENT_RE.search(line) and _FAILLOUD_TEST_EVIDENCE_RE.search(line):
            out.append(line.strip())
            continue
        if _FAILLOUD_PYTEST_RAISES_RE.search(line):
            out.append(line.strip())
            continue
        if _FAILLOUD_PIN_LABEL_RE.search(line):
            for j in range(i, min(len(lines), i + _FAILLOUD_PIN_SCAN_LINES)):
                if not lines[j].strip():
                    break
                if _FAILLOUD_GREP_LINE_RE.search(lines[j]):
                    continue
                if _TEST_IDENT_RE.search(lines[j]):
                    out.append(f"{line.strip()} … {lines[j].strip()}")
                    break
    return out


def check_failloud_test_coverage(plan: str, kind: str) -> CheckResult:
    """``kind: infra|batch`` plans whose acceptance/success criteria assert
    fail-loud / no-silent-swallow behavior must name a committed test pinning
    it — run-book grep gates verify the invariant once at review time, and a
    differently-worded re-swallow ships green past all committed tests
    (#913). WARN not FAIL: trigger and evidence are line heuristics; the
    Phase 2 critics adjudicate (Statistics lens item 14 owns the per-claim
    coverage judgment this check cannot make — the mechanical layer catches
    only the zero-fail-loud-test case, and PASSes a plan naming a fail-loud
    test for a different claim). Extending kind scope to ``analysis`` is a
    future calibration decision if an incident arises there (the corpus
    replay covered infra|batch only). Evidence routes 2-3 (#1306) — a
    quoted ``pytest.raises(<Exception>`` control and a labeled
    ``Fail-loud pin``-style paragraph scan — were corpus-audited
    2026-07-14 (exactly the two #1296 incident plans flip WARN→PASS out
    of 119 standing infra|batch WARNs; unlabeled ±1/±2-line windows were
    REJECTED at 21/29 false flips); accepted residual: a fenced
    bug-narration ``pytest.raises(...)`` quote or a stale/nonexistent
    test name in a labeled pin can false-satisfy — acceptable because the
    check is WARN-only at existence granularity and per-claim coverage
    stays with the Phase 2 critics."""
    cid, name = "c15_failloud_test_coverage", "fail-loud acceptance claim backed by a test"
    if kind not in ("infra", "batch"):
        return _skip(
            cid,
            name,
            "kind-exempt: the fail-loud acceptance-claim pattern is an infra|batch shape",
        )
    hits = _failloud_claim_hits(plan)
    if not hits:
        return _skip(cid, name, "no fail-loud claim in an acceptance/success-criteria window")
    if _standalone_na_declared(
        plan, r"(?:no fail[- ]?loud acceptance claim|fail[- ]?loud claim not test-backable)"
    ):
        return _pass(
            cid, name, "explicit N/A declared (incidental vocabulary or not test-backable)"
        )
    evidence = _failloud_test_evidence_lines(plan)
    if evidence:
        sec, tok = hits[0]
        return _pass(
            cid,
            name,
            f"fail-loud claim ({tok!r} in §{sec[:40]!r}) + fail-loud test named "
            f"({evidence[0][:80]!r})",
        )
    sec, tok = hits[0]
    return _warn(
        cid,
        name,
        f"acceptance/success criteria assert fail-loud behavior ({tok!r} in §{sec[:40]!r}) but "
        "no line names a committed test carrying fail-loud vocabulary (a `test_` identifier or "
        "tests/<file> path alongside raise/swallow/silent/except vocabulary; grep-gate lines do "
        "not count) — a run-book grep verifies the invariant once at review time, and a "
        "differently-worded re-swallow ships green past all committed tests (#913). Name the "
        "pinning test and its raise/swallow/silent vocabulary on ONE unwrapped line "
        "(hard-wrapped mentions do not count), quote the `pytest.raises` negative control "
        "with its exception class, add a `Fail-loud pin` labeled line (the label, a colon, "
        "then the test path), or declare `N/A — no fail-loud acceptance claim` / "
        "`N/A — fail-loud claim not test-backable` — each on its own line, unwrapped "
        "(no backticks/quotes)",
    )


# ─── Check 16 — re-extracted reference vs committed headline (WARN-only) ───

# Trigger half (a): a NON-NEGATED re-extraction/regeneration token on a
# NON-fenced line, with reference/parity/committed vocabulary nearby.
# Two branches, calibrated on the 2026-07-03 historical-corpus sweep:
#   - `re-?extract`: vocabulary within ±_C16_WINDOW_LINES RAW lines
#     (hard-wrapped prose splits "re-extracted\nreferences"; #811 v3's §5
#     rows carry "(reference, re-extracted)" on one line);
#   - `re-?generat`: SAME-line adjacency only (the plan §4.5 pre-authorized
#     demotion — window-scoped re-generat swept in doc/data-regeneration
#     noise: #491/#537/#542/#558/#597/#685/#763/#825 fired on regeneration
#     mentions with reference vocab merely nearby).
# The fixed-width negation lookbehinds drop ASSERTED-NEGATIVE mentions
# ("NOT regenerated", "NO re-extraction of r_B" — #559/#561/#810-v1-3 noise
# class): a plan stating it does NOT re-extract is not a trigger.
# `\bre-?extract` does not match "pre-extraction" (no word boundary inside
# "pre").
_C16_NEG_GUARD = r"(?<!\bno )(?<!\bnot )(?<!\bnever )(?<!\bwithout )"
_C16_EXTRACT_RE = re.compile(rf"(?i){_C16_NEG_GUARD}\bre-?extract\w*")
_C16_REGEN_RE = re.compile(rf"(?i){_C16_NEG_GUARD}\bre-?generat\w*")
_C16_REF_RE = re.compile(
    r"(?i)\breferences?\b|\breference[- ]arms?\b|\bparity\b"
    r"|\bcommitted (?:cells?|v\d)|prior[- ]headline"
)
_C16_WINDOW_LINES = 3

# Trigger half (b): the plan reads as a same-issue follow-up / amendment
# folding into an existing clean-result. Document-global, fence-stripped;
# (?s) so the wrapped "folds into THIS\nissue's clean-result body" shape
# (#811 v3:87-89) is caught. Bare "follow-up round" is deliberately absent
# (709 occ / 216 files in the 2026-07-03 corpus probe — plans cite the
# follow-up machinery prospectively).
_C16_FOLD_RE = re.compile(
    r"(?is)same-issue follow-?up|amendment to (?:the|this|a)\b"
    r"|epm:followup-scope|followups?_running"
    r"|folds? into .{0,80}?clean-result"
)

# Satisfaction: an explicit sentence distinguishing same-pass comparator
# values from prior committed headline values. Three shapes:
#   S1 — the term of art itself ("comparator" REQUIRED: #811 v3:189
#        "re-extracting the references in the SAME pass" must not satisfy);
#   S2 — committed-headline noun phrase + a retention verb within one
#        sentence. Gaps exclude '.' and ';' so v3:574 "committed cells only
#        via R resampling." and v3:499 "(committed; prior rounds' artifacts
#        untouched)" cannot satisfy — the sentence stop / path dots block;
#   S3 — an explicit negated-replacement clause naming the headline
#        (v3:270 "layout replaces grouped bars" carries no negation).
# "replication-stability" vocabulary alone deliberately does NOT satisfy —
# the incident plan carried it (v3:347, :434).
_C16_SAMEPASS_RE = re.compile(r"(?i)same[- ]pass comparators?")
_C16_DISTINCTION_RE = re.compile(
    r"(?is)(?:committed|prior|standing|already[- ]adjudicated)"
    r"[^.;]{0,40}?\b(?:headline|cells?|values?|verdicts?|calls?|evidence)"
    r"[^.;]{0,120}?"
    r"(?:remains?|retain\w*|stays?|stands?|kept|keeps?|unchanged|untouched"
    r"|(?:is |are )?not (?:silently )?replaced?|never (?:silently )?replaced?)"
)
_C16_NONREPLACE_RE = re.compile(
    r"(?is)(?:never|not|no)\s+(?:a\s+)?(?:silent(?:ly)?\s+)?"
    r"(?:headline[- ])?replac\w*[^.;]{0,80}?(?:headline|committed)"
    r"|(?:never|not)\s+(?:silently\s+)?replac\w*[^.;]{0,60}?headline"
)


def check_reference_headline_distinction(plan: str, kind: str) -> CheckResult:
    """A follow-up plan that re-extracts prior-headline REFERENCE arms AND
    folds into an existing clean-result must explicitly distinguish
    "same-pass comparator" values from "prior committed headline" values —
    a reference flip is replication-stability evidence, never an
    unannounced headline replacement (#811 v3 §6; task #937). WARN not
    FAIL: both trigger halves and the satisfaction shapes are text
    heuristics; the Statistics critic adjudicates the semantic question
    (does the plan's adjudication story actually preserve the committed
    cells) — this gate surfaces, never adjudicates."""
    cid = "c16_reference_headline_distinction"
    name = "re-extracted reference vs committed headline"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: clean-result folding is an experiment|analysis plan shape",
        )
    # re-extract: ±3-line windows; re-generat: same-line only (radius 0).
    windows = _trigger_windows(plan, _C16_EXTRACT_RE, _C16_WINDOW_LINES)
    windows += _trigger_windows(plan, _C16_REGEN_RE, 0)
    if not any(_C16_REF_RE.search(w) for w in windows):
        return _skip(cid, name, "no re-extraction of reference arms detected")
    text = strip_fences(plan)
    if not _C16_FOLD_RE.search(text):
        return _skip(
            cid,
            name,
            "re-extraction vocabulary present but the plan does not read as a "
            "same-issue follow-up folding into an existing clean-result",
        )
    if _standalone_na_declared(plan, r"no re-?extracted reference arms"):
        return _pass(cid, name, "explicit N/A declared (no re-extracted reference arms)")
    if (
        _C16_SAMEPASS_RE.search(text)
        or _C16_DISTINCTION_RE.search(text)
        or _C16_NONREPLACE_RE.search(text)
    ):
        return _pass(
            cid,
            name,
            "distinguishing sentence present (same-pass comparator / committed-headline retention)",
        )
    return _warn(
        cid,
        name,
        "plan re-extracts prior-headline reference arms AND folds into an existing "
        "clean-result, but no sentence distinguishes same-pass-comparator values from "
        "the prior committed headline values — state which values adjudicate this "
        "round's NEW comparison vs which are still the committed headline, and that a "
        "flipped reference CALL is reported as replication-stability evidence rather "
        "than replacing the headline (#811 v3 §6 incident; the kept-cells-"
        "stay-evidence rule), or declare `N/A — no re-extracted reference arms` "
        "on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 17 — falsification-branch causal-claim scope (WARN-only) ────────

# Offender vocabulary: wording that asserts a causal mechanism as
# DEMONSTRATED inside a registered branch. Tier-1 only (corpus-calibrated,
# task #946 §6): retrospective attribution ("really was/were", "must have
# been"), content-carrying claims, story-kill idioms, takeaway rewrites,
# and explicit establish/prove/demonstrate-that. Deliberately EXCLUDED as
# accepted false negatives (prefer false negatives — the c14 charter):
# present-tense "really is/does" (5-6 of 8 corpus hits were legitimate),
# "rules out" (#605 uses it as a CI equivalence bound), bare "must be"
# (deontic), and bare mechanism-noun falsify labels ("**Falsified
# (integration):**" — not regex-separable from "(dependence)" labels).
_C17_OFFENDER_RE = re.compile(
    r"(?i)"
    r"\breally\s+(?:was|were)\b"
    r"|\bmust\s+have\s+been\b"
    r"|\bcarr(?:y|ies|ied|ying)\b[^.\n]{0,50}\bcontent\b"
    r"|\b(?:story|account|hypothesis|explanation|interpretation)\s+(?:dies|is\s+dead)\b"
    r"|\brewrit(?:es?|ing)\s+the\b[^.\n]{0,60}\b(?:interpretation|takeaway|headline)\b"
    r"|\b(?:establish(?:es|ed)?|prov(?:es|ed)|demonstrat(?:es|ed))\s+that\b"
)
# Exculpation vocabulary: an alternative-naming / hedge token in the SAME
# block (hyp surface) or SAME bullet (TL;DR surface) silences the offender.
# Over-breadth here only creates false negatives, which the charter prefers.
# Calibrated on the #810 v13→v14 fix wording plus corpus hits (#563 "scope
# caveat", #611/#621 "artifact", #841 "gets real support").
_C17_EXCULP_RE = re.compile(
    r"(?i)"
    r"\bconsistent\s+with\b|\bcompatible\s+with\b"
    r"|\buniquely\s+diagnostic\b"
    r"|\bcannot\s+(?:distinguish|rule\s+out)\b"
    r"|\bdoes\s+not\s+distinguish\b|\bdoesn'?t\s+distinguish\b"
    r"|\balternative\b|\bconfound\w*\b|\bartifact\w*\b|\bcaveats?\b"
    r"|\bsimpler\s+explanations?\b|\bother\s+explanations?\b"
    r"|\bOOD\b|\boff-?distribution\b|\bout-?of-?distribution\b"
    r"|\bremains?\s+live\b|\bdegradation\b|\bendpoint\b"
    r"|\bpending\b|\bdisambiguat\w*\b|\bunder-?determin\w*\b|\bambiguous\b"
    r"|\bwould\s+not\s+(?:prove|establish|demonstrate)\b"
    r"|\b(?:gets?|gains?|lends?|earns?)\s+(?:real\s+)?support\b"
)
# The §0.0 registered plain-English falsification branch ("**What would
# change my mind:**" / "…mind.**" — both corpus punctuation shapes).
_C17_MIND_RE = re.compile(r"(?i)\*\*\s*what would change my mind")


def _c17_mind_segments(plan: str) -> list[str]:
    """The fence-stripped `**What would change my mind**` bullet(s), each
    with its continuation lines (up to the next top-level list item or
    heading) — the §0.0 registered falsification branch surface."""
    lines = strip_fences(plan).splitlines()
    segs: list[str] = []
    i = 0
    while i < len(lines):
        if _C17_MIND_RE.search(lines[i]):
            seg = [lines[i]]
            j = i + 1
            while (
                j < len(lines)
                and not _C14_LIST_ITEM_RE.match(lines[j])
                and not _HEADING_RE.match(lines[j].strip())
            ):
                seg.append(lines[j])
                j += 1
            segs.append("\n".join(seg))
            i = j
        else:
            i += 1
    return segs


def check_causal_claim_scope(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a registered falsification (or confirm)
    branch must not word its outcome as a DEMONSTRATED causal mechanism
    when the same block never names the undistinguished alternative.
    Surfaces scanned: (i) confirm/falsify segments of anchored hypothesis
    blocks (c14's parsers, reused); (ii) the §0.0 `**What would change my
    mind:**` bullet(s). An offender token is silenced by an exculpation
    token in the same block/bullet. NEVER FAILs — a heuristic vocabulary
    check must not hard-block a legitimately-worded plan; whether the
    diagnostics actually distinguish the mechanism stays with the
    Methodology/Statistics critics. The §6 corpus noise floor (2/195
    newest-per-task) is IN-SAMPLE — the offender/exculpation vocabulary
    was tuned on the same corpus it was measured on — so any future
    FAIL-promotion needs held-out / prospective validation first.
    Incident: #810 plan v13 ("they really were carrying answer content,
    the echo story dies") — three reviewers independently required the
    v14 scope-down ("consistent with integration but not uniquely
    diagnostic; OOD ... remains live"); task #946."""
    cid, name = "c17_causal_branch_scope", "falsification-branch causal-claim scope"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: registered falsification branches are an experiment|analysis plan shape",
        )
    anchored: list[tuple[str, tuple[str, str]]] = []
    section = section_text_by_keywords(plan, ("hypothesis",))
    if section is not None:
        for block in _hypothesis_blocks(strip_fences(section)):
            segments = _confirm_falsify_segments(block)
            if segments is not None:
                anchored.append((block, segments))
    mind_segs = _c17_mind_segments(plan)
    if not anchored and not mind_segs:
        return _skip(
            cid,
            name,
            "no registered falsification-branch surface (no **Confirm/**Falsify "
            "hypothesis anchors, no **What would change my mind** bullet)",
        )
    offenders: list[str] = []
    for block, (confirm_seg, falsify_seg) in anchored:
        if _C17_EXCULP_RE.search(block):
            continue
        for branch, seg in (("falsify", falsify_seg), ("confirm", confirm_seg)):
            m = _C17_OFFENDER_RE.search(seg)
            if m:
                offenders.append(
                    f"hypothesis block {_c14_block_label(block)} ({branch} segment): "
                    f"claim token '{m.group(0)}'"
                )
                break  # one offender per block is enough for the detail
    for seg in mind_segs:
        if _C17_EXCULP_RE.search(seg):
            continue
        m = _C17_OFFENDER_RE.search(seg)
        if m:
            offenders.append(f"'What would change my mind' bullet: claim token '{m.group(0)}'")
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(anchored)} hypothesis block(s) + {len(mind_segs)} TL;DR bullet(s) "
            "scanned; no unqualified demonstrated-mechanism claim token",
        )
    extra = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
    return _warn(
        cid,
        name,
        "; ".join(offenders[:3])
        + extra
        + " — the branch asserts a causal account its diagnostics may not uniquely "
        "distinguish (#810 v13 incident): name the undistinguished alternative in "
        "the same block (e.g. 'consistent with <mechanism> but not uniquely "
        "diagnostic; <alternative> remains live') or scope the wording to the "
        "measured quantity; semantic verdict stays with the critics",
    )


# ─── Check 18 — paired-contrast per-arm source coverage ────────────────────

# Registration-family sections (H2+ ONLY — a doc-spanning H1 title match
# would make the section constraint vacuous; both #810 registrations sit
# under H2 sections): c13's success/kill/evaluation families PLUS
# hypothesis + nulls + statistic.
_C18_SECTION_RE = re.compile(
    r"(?i)hypothes|success criteri|acceptance criteri|decision rule|decision gate"
    r"|kill[- ]criteri|abort criteri|stop criteri|\bevaluation\b|\bnulls?\b|statistic"
)
_C18_PAIRED_RE = re.compile(r"(?i)\bpaired\b")
_C18_REGIST_RE = re.compile(r"(?i)\bregist")  # registered / registration / registers
_C18_PAIRCOUNT_RE = re.compile(r"(?i)\b\d[\d,]*\s+(?:pre-named\s+)?pairs\b")
# D1: a row-coverage declaration; evidence on the same line or within the
# next _C18_DECL_WINDOW_LINES physical lines (fenced lines excluded).
_C18_COVERAGE_RE = re.compile(r"(?i)\brow[- ]coverage\b")
# #1086 widening: suffixed tensor-store dirs (`analysis_tensors_nonemit/` —
# the `\w*` suffix arm) and canonical `issueN_<slug>/…` HF data-repo
# prefixes (the Upload Policy destination shape) are artifact evidence. The
# trailing `\S*` on the `analysis_tensors\w*/\S*` alternative — not `\S+` —
# is DELIBERATE: a bare store-dir token ending at the slash (backticked or
# line-final `analysis_tensors_nonemit/`) is complete artifact evidence
# with nothing after the slash. An `issueN…/` PATH token is affirmative
# artifact evidence, orthogonal to the `_C18_ISSUE_REF_RE` citation guard
# (the literal `#\d{2,}` form), which is byte-unchanged. Accepted
# fail-UNSAFE residual, DISCLOSED: a SIBLING issue's `issueN…/` store path
# on the declaration line counts as D1 artifact evidence — whether the
# named store truly contains THIS plan's rows stays with the fact-checker
# (no guard, no negative fixture: a negative would require a citation
# guard #1086 deliberately does not add).
_C18_ARTIFACT_RE = re.compile(
    r"(?i)\S+\.(?:pt|pth|json|jsonl|npz|npy|safetensors|csv|parquet|arrow)\b"
    r"|\beval_results/\S+|\banalysis_tensors\w*/\S*|\braw_completions/\S+"
    r"|\bissue\d{2,}[\w.-]*/\S+"
)
# v2 (MF-B): the bare `this run` alternative is REMOVED — only the
# arms-generated construction or an explicit `by construction` counts as
# by-construction evidence ("Row-coverage: deferred to a later revision of
# this run's analysis" must FAIL).
_C18_BYCONSTRUCTION_RE = re.compile(
    r"(?i)both arms .{0,60}\b(?:generated|produced|computed|fit(?:ted)?|emitted)\b"
    r"|\bby construction\b"
)
# #1086 (v2): the check's own remedy text ("state that the plan's own fits
# produce every registered row on each arm", the FAIL detail below) was
# unmatchable by _C18_BYCONSTRUCTION_RE — a planner implementing the bounce
# verbatim still FAILed (#833 v8). Accept that form via a SEPARATE
# alternative deliberately narrower than the remedy prose: (i) affirmative
# produce-verb + "every registered", (ii) arm vocabulary within 80 chars
# after the match (each/both/per arm[s]), (iii) NO negation/deferral token
# in the local span around the match — "does not yet produce every
# registered row on each arm" and "will produce every registered row …
# once implemented" are explicit NON-declarations and must keep FAILing
# (the MF-B deferral class). _C18_BYCONSTRUCTION_RE itself stays
# byte-unchanged so no historical PASS can flip.
_C18_PRODUCES_REGISTERED_RE = re.compile(
    r"(?i)\b(?:produces?|generates?|computes?|emits?|yields?)\s+every\s+registered\b"
    r"(?=.{0,80}\b(?:each|both|per)\s+arms?\b)"
)
# #1099: the guard covers the n't contraction family (word chars + n +
# straight-or-curly apostrophe + t) — the prior bare `n't` alternative was
# DEAD CODE (a word boundary never matches at the word-internal s->n
# transition inside "doesn't"; probe-verified) — plus cannot /
# fail(s|ed) to / until (all common in the plan corpus — thousands of
# occurrences for cannot/doesn't; counts hedged deliberately, they age).
# Curly apostrophe included (a handful of corpus plan files carry it).
# Strictly widening the DISQUALIFIER = strictly narrowing the
# affirmative satisfier — fail-safe by construction. Accepted residual
# (disclosed, #1099): "no longer" / "except" / "unless" / "rather than" /
# gerund "failing to" still evade this guard — outside the Goal-named
# set; the Phase-2 critic ensemble is the semantic backstop (same
# fail-unsafe residual class as c12's sibling-quote disclosure).
_C18_NEG_DEFER_RE = re.compile(
    r"(?i)\b(?:not|\w*n[’']t|cannot|never|without|fail(?:s|ed)?\s+to|until"  # noqa: RUF001
    r"|will|would|shall|should|may|might|could"
    r"|once|pending|deferred|later|TBD|to\s+be)\b"
)


def _c18_affirmative_produces_hit(line: str) -> bool:
    """The v2 remedy-text alternative: affirmative produce-verb + 'every
    registered' + arm vocabulary, with negation/deferral tokens disqualifying
    in a local span (48 chars before the match start, 80 after its end).
    Scoped to THIS alternative only — the legacy _C18_BYCONSTRUCTION_RE
    alternatives keep their behavior byte-for-byte."""
    m = _C18_PRODUCES_REGISTERED_RE.search(line)
    if not m:
        return False
    span = line[max(0, m.start() - 48) : m.end() + 80]
    return not _C18_NEG_DEFER_RE.search(span)


# D2 (MF-A): a subset expression AND word-bounded row/pair vocabulary AND
# coverage/source-key vocabulary must co-occur on the candidate line.
# Word-bounding kills the 608 v2:164 false-satisfier ("pair" inside
# "paired" no longer matches); the coverage-vocab conjunct excludes
# incidental subset prose; the #810 v15 declaration carries standalone
# row/pairs tokens + coverage/source/keys/assert (replay-verified).
_C18_SUBSET_RE = re.compile(r"(?i)⊆|\bissubset\b|\bis a subset of\b")
_C18_ROWPAIR_RE = re.compile(r"(?i)\b(?:pairs?|rows?)\b")
_C18_COVERAGE_VOCAB_RE = re.compile(r"(?i)coverage|\bsources?\b|\bkeys?\b|\bassert")
# Candidate-line rejection guards (BOTH satisfier families):
# (a) paste fingerprint — the c18 FAIL detail carries this literal, so a
#     verbatim-pasted bounce text can never self-satisfy;
# (b) cross-issue citation token — a line QUOTING another issue's driver
#     assert as a worked example is a citation, not a declaration (an
#     honest declaration describes THIS plan's inputs; recovery for a
#     legitimate collision: move the citation off the declaration line).
_C18_PASTE_FINGERPRINT = "#810 v13 class"
_C18_ISSUE_REF_RE = re.compile(r"#\d{2,}")
_C18_DECL_WINDOW_LINES = 3
# Trigger-side spurious-line guard (§3.4 calibration tuning): a FIGURES-
# enumeration line ("**Figures (over-produce):** ... paired cells; ...
# registered rows visually distinguished") lists plots, it registers no
# statistic — the one spurious-trigger class the exhaustive FAIL audit
# surfaced (7 corpus files: #537 v4-v6, #931 v1-v4). Scoped by LINE SHAPE
# (a leading figures label), never by content elsewhere on the line; a
# real registration line never opens with a figures label, so the guard
# under-triggers safe (SKIP; critics review).
_C18_FIGURES_LINE_RE = re.compile(r"(?i)^\W{0,8}figures?\b")

# Known accepted mis-triggers (mirroring the c13 §4.5 precedent). Under-
# triggers that fail SAFE (SKIP — the plan still reaches the fact-checker +
# critic ensemble): (a) a paired registration line without `regist` / pair-
# count vocabulary; (b) a registration under a heading outside the H2+
# section family; (c) a hard-wrapped registration (`paired` and `regist` on
# different lines). Over-trigger that fails LOUD (bounce, escapable): (d) a
# Hypothesis-section line merely RECAPPING a sibling's registered paired
# statistic — remedied by the standalone N/A line. Fail-UNSAFE residuals,
# accepted and DISCLOSED: (e) a D1/D2-shaped declaration that doesn't
# actually cover the registered rows — including a ONE-ARM declaration (the
# #810 v15 exemplar itself is full-side-only; both-arm truth stays with the
# fact-checker; disposition pinned by fixture); (f) a NON-verbatim
# paraphrase of the bounce text that reconstructs a satisfying shape while
# dropping the fingerprint — beyond mechanical defense, same residual class
# as a dishonest c13 N/A line; (g) a wrapped/reformatted paste that
# separates the fingerprint from the row-coverage phrase across lines — the
# line-local guard misses it; the D1 evidence requirement (artifact token /
# arms-generated phrase / #1086's affirmative produces-registered form)
# still has to be met by the surviving fragment. NOTE (#1086): the remedy
# text's own "produce every registered row on each arm" clause is now a
# satisfier BY DESIGN (the remedy-vs-satisfier inconsistency was the bug),
# so a wrapped paste landing that clause on a citation-free Row-coverage
# line self-satisfies — a widened, DISCLOSED instance of this same
# residual class.


def _c18_registered_paired_lines(plan: str) -> list[str]:
    """Non-fenced lines inside a registration-family H2+ section carrying
    ``paired`` plus registration vocabulary OR an enumerated pair count on
    the SAME line (#810 v13:33 'Registered per-row statistic: paired ...
    (7 pairs ...' and v13:103 'Nulls (registration) ... paired bootstrap CI
    (... 9 pairs are pre-named' both match). Level-1 headings are EXCLUDED
    from the section match (a title match spans the whole doc). Under-
    trigger fails safe (SKIP; critics review)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    hits: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _C18_PAIRED_RE.search(line):
            continue
        if not (_C18_REGIST_RE.search(line) or _C18_PAIRCOUNT_RE.search(line)):
            continue
        if _C18_FIGURES_LINE_RE.match(line.strip()):
            continue
        if not any(
            h.line <= i < h.end and h.level >= 2 and _C18_SECTION_RE.search(h.text)
            for h in headings
        ):
            continue
        hits.append(line.strip())
    return hits


def _c18_candidate_ok(line: str) -> bool:
    """Rejection guards shared by D1 and D2 candidate lines: the paste
    fingerprint and cross-issue citation tokens disqualify a line from
    satisfying the check (bounce-paste + quoted-sibling-example vectors)."""
    return _C18_PASTE_FINGERPRINT not in line and not _C18_ISSUE_REF_RE.search(line)


def _c18_coverage_declarations(plan: str) -> list[str]:
    """Lines satisfying D1 (row-coverage vocab + source evidence — an
    artifact token or an arms-generated phrase — on the same line or within
    the next _C18_DECL_WINDOW_LINES physical lines, fenced lines excluded)
    or D2 (subset expression + word-bounded row/pair vocab +
    coverage/source-key vocab, same line). Candidate lines failing
    ``_c18_candidate_ok`` are rejected."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[str] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _c18_candidate_ok(line):
            continue
        if (
            _C18_SUBSET_RE.search(line)
            and _C18_ROWPAIR_RE.search(line)
            and _C18_COVERAGE_VOCAB_RE.search(line)
        ):
            out.append(line.strip())
            continue
        if _C18_COVERAGE_RE.search(line):
            window = [line] + [
                lines[j]
                for j in range(i + 1, min(i + 1 + _C18_DECL_WINDOW_LINES, len(lines)))
                if not mask[j]
            ]
            if any(
                _C18_ARTIFACT_RE.search(w)
                or _C18_BYCONSTRUCTION_RE.search(w)
                or _c18_affirmative_produces_hit(w)
                for w in window
            ):
                out.append(line.strip())
    return out


def _c18_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no paired contrast`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale)."""
    return _standalone_na_declared(plan, r"no paired contrast")


def check_paired_contrast_source_coverage(plan: str, kind: str) -> CheckResult:
    """A registered paired contrast (a hypothesis/evaluation/success-section
    line registering a paired statistic over enumerable rows/pairs) must
    DECLARE a per-context data source covering the registered rows on both
    arms (D1 row-coverage line / D2 coverage-labeled subset-assert /
    standalone N/A). Surface check only — pack contents stay with the
    fact-checker. FAIL (experiment) / WARN (analysis) / SKIP otherwise;
    the standalone ``N/A — no paired contrast`` escape is honored (SKIP
    path) only when no paired contrast is detected — when the escape
    co-occurs with a detected registration the check WARNs instead of
    PASSing, so the escape can never mask row-coverage verification of a
    present registration (#1258, the #1223 c20 rule).
    Incident: #810 v13 (9-row paired bootstrap; the named full-side pack
    lacked im_end/turn_nl; 4 independent reviewer catches)."""
    cid, name = "c18_paired_contrast_source_coverage", "paired-contrast per-arm source coverage"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: registered paired contrasts are an experiment|analysis plan shape",
        )
    triggers = _c18_registered_paired_lines(plan)
    if not triggers:
        return _skip(cid, name, "no registered paired contrast detected")
    decls = _c18_coverage_declarations(plan)
    if _c18_na_escape_declared(plan):
        # #1258 (the #1223 c20 port): reachable ONLY when a registered paired
        # contrast WAS detected (the no-trigger case SKIPs above) — a PASS
        # here masks the row-coverage verification c18 exists to run (the
        # #810 v13 class). WARN, not FAIL: the trigger harvest may be a false
        # positive on quoted guidance; reviewers adjudicate.
        #
        # #1689/#1700 refinement: when a valid Row-coverage declaration ALSO
        # co-occurs, the coverage decl is the DECISIVE load-bearing assertion
        # (D1/D2 grammar unchanged; _c18_coverage_declarations gates it), and
        # the N/A line is a spurious leftover paste — PASS on the coverage
        # decl. The original masking case (N/A alone, no coverage decl) still
        # WARNs, preserving c18's original defense.
        if decls:
            return _pass(
                cid,
                name,
                f'row-coverage declaration found ("{decls[0][:90]}") — declaration surface '
                "only; the co-occurring standalone `N/A — no paired contrast` line is "
                "treated as a spurious leftover, not a masking claim (the coverage decl "
                "is decisive; #1689/#1700). Whether the named sources truly contain every "
                "registered row on both arms stays with the fact-checker.",
            )
        return _warn(
            cid,
            name,
            "the standalone `N/A — no paired contrast` escape co-occurs with "
            f"{len(triggers)} registered paired-contrast line(s) (first: "
            f"{triggers[0][:90]!r}) — the escape is reserved for contrast-free "
            "plans and would mask row-coverage verification of the detected "
            "registration (#1258, the #1223 c20 rule); remove the N/A line and "
            "declare Row-coverage (the registration is then verified), or "
            "fence/remove the registration-shaped prose the detector matched "
            "if it is quoted guidance rather than this plan's own registration",
        )
    if decls:
        return _pass(
            cid,
            name,
            f'row-coverage declaration found ("{decls[0][:90]}") — declaration surface '
            "only; whether the named sources truly contain every registered row on both "
            "arms stays with the fact-checker",
        )
    detail = (
        f'plan registers a paired contrast ("{triggers[0][:90]}") with no per-arm '
        "row-coverage declaration — a registered pair row absent from a named side makes "
        "the registered criterion unsatisfiable from the named inputs (the #810 v13 class: "
        "2 of 9 rows missing from the named full side). Remedy: add ONE non-fenced prose "
        "line (not inside a code fence) starting 'Row-coverage:' naming, for BOTH arms, "
        "which per-context store/file supplies every registered row (or stating that the "
        "plan's own fits produce every registered row on each arm), or state the driver "
        "assert that set-checks the registered rows against the named sources' keys on a "
        "non-fenced line, or declare 'N/A — no paired contrast' on its own line, unwrapped "
        "(no backticks/quotes); keep the declaration line free of cross-issue citations"
    )
    if kind == "analysis":
        return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
    return _fail(cid, name, detail)


# ─── Check 19 — OOD generalization folds (WARN-only, conditional) ──────────

# Trigger = a fold token SOLO (any cross-validation mention makes "is the
# fold group-level?" the right question), OR the WEAK token "held-out"
# conjoined with a predictor-statistic token. Bare "held-out" alone is an
# eval-split adjective (GOOD_PLAN: "40 held-out prompts") and must not fire;
# bare "predict(s)" is hypothesis prose and is deliberately excluded.
_C19_SOLO_FOLD_RE = re.compile(
    r"(?i)\bcross[- ]?validat\w*|\bLOO\b|\bLOCO\b|\bLOOCV\b"
    r"|\bleave[- ]one[- ][\w-]*out\b|\bk[- ]fold\b"
)
_C19_HELDOUT_RE = re.compile(r"(?i)(?<!\bno )(?<!\bnot )\bheld[- ]out\b")
_C19_PREDSTAT_RE = re.compile(
    r"(?i)\bR\^?2\b|R²|\breconstruction\b|\bread[- ]?outs?\b"
    r"|\bpredict(?:or|ive|ion)s?\b|\bregress\w*|\bridge\b"
    r"|\b(?:probe|decod\w*)\s+accurac\w*"
)
# Group-level evidence: leave-one-<UNIT>-out where UNIT is not a pointwise
# sample unit (#810's offender fold was leave-one-CONTEXT-out — pointwise).
_C19_LOO_UNIT_RE = re.compile(r"(?i)\bleave[- ]one[- ]([\w-]+?)[- ]out\b")
_C19_POINTWISE_UNITS = frozenset(
    {
        "context",
        "point",
        "sample",
        "row",
        "item",
        "question",
        "prompt",
        "cell",
        "completion",
        "example",
        "datapoint",
        "datum",
        "observation",
        "pair",
        "x",
    }
)


def _c19_pointwise_unit(unit: str) -> bool:
    """A captured leave-one-<unit>-out unit is pointwise when its EXACT form
    OR its hyphen-split SUFFIX segment is blocklisted — hyphenated variants
    (``data-point``) must not self-certify as group evidence (reconciler
    Must-Fix, round 1). ``prompt-family`` stays a group unit (suffix
    ``family`` is not blocklisted)."""
    u = unit.lower()
    return u in _C19_POINTWISE_UNITS or u.split("-")[-1] in _C19_POINTWISE_UNITS


_C19_GROUPFOLD_RE = re.compile(
    r"(?i)\bLOFO\b|group[- ]level (?:held[- ]out )?fold"
    r"|held[- ]out (?:group|famil\w*|genre|persona|corpus)"
    r"|(?:corpus|genre|domain|family)[- ]transfer\b|\btransfer arm\b"
)
# Negation-guarded: `non-iid` / `not iid` concedes group structure and must
# NOT satisfy the iid PASS tier (round-1 convergent critic concern).
_C19_IID_RE = re.compile(r"(?i)(?<!non[- ])(?<!\bnot )\b(?:iid\b|i\.i\.d\b)")


def check_ood_folds(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a held-out predictive DV (reconstruction R²,
    read-out rho, predictor accuracy) over group-structured samples must
    register a GROUP-level fold (LOFO / corpus transfer), declare
    ``N/A — no held-out predictive DV``, or argue a genuinely iid sample
    (.claude/rules/ood-generalization-folds.md; planner §6 Required block).
    NEVER FAILs — the trigger is a vocabulary heuristic; whether the named
    fold is actually group-level for this sample stays with the Statistics
    critic (lens item 13). Incident #810: the pointwise-LOCO headline
    reordered under leave-one-FAMILY-out and the read-out collapsed
    rho 0.909 → 0.285."""
    cid, name = "c19_ood_folds", "OOD generalization folds (held-out predictive DV)"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: infra|batch|survey plans have no predictive DV")
    if _standalone_na_declared(plan, r"no held-?out predictive DV"):
        return _pass(cid, name, "explicit N/A declared (no held-out predictive DV)")
    text = strip_fences(plan)
    solo = _C19_SOLO_FOLD_RE.search(text)
    conj = _C19_HELDOUT_RE.search(text) and _C19_PREDSTAT_RE.search(text)
    if not (solo or conj):
        return _skip(
            cid,
            name,
            "no held-out predictive-DV vocabulary (no fold token; no held-out + "
            "predictor/R²/read-out co-occurrence)",
        )
    group_units = [
        m.group(1) for m in _C19_LOO_UNIT_RE.finditer(text) if not _c19_pointwise_unit(m.group(1))
    ]
    if group_units or _C19_GROUPFOLD_RE.search(text):
        return _pass(
            cid,
            name,
            "group-level fold vocabulary present"
            + (f" (leave-one-{group_units[0]}-out)" if group_units else "")
            + " — fold validity + per-headline fold labeling stay critic-owned",
        )
    if _C19_IID_RE.search(text):
        return _pass(
            cid,
            name,
            "iid-sample argument present (the only pointwise-only exemption) — whether "
            "the sample is genuinely iid stays critic-owned",
        )
    return _warn(
        cid,
        name,
        "held-out predictive-DV vocabulary detected but no group-scoped fold "
        "(leave-one-<family>-out / a fit-on-one-corpus-score-on-another arm), no "
        "independence (i-i-d) argument, and no `N/A — no held-out predictive DV` escape "
        "declared on its own line, unwrapped (no backticks/quotes) — pointwise LOO can "
        "REORDER cross-context claims "
        "(#810: read-out rho 0.909 → 0.285 under the family-level fold); "
        ".claude/rules/ood-generalization-folds.md; Statistics critic must gate this",
    )


# ─── Check 20 — verdict-lattice coherence (conditional) ────────────────────

# Trigger sections: hypothesis / success / kill / decision / verdict / gate —
# the c8/c13 families plus "hypothes" + "verdict"; deliberately NOT
# "evaluation" (c13 includes it for gate lines; a verdict LATTICE registered
# only in an Evaluation recap is an accepted under-trigger — fails safe).
_C20_SECTION_RE = re.compile(r"(?i)hypothes|success|kill|decision|verdict|gate")

# Tier 1: the #923 v6 registered form — "…DISJOINT and exhaustive: <label> ⇔
# <predicate>; …". The declaration claims a partition, so BOTH defect classes
# (co-fire AND gap) are FAIL-capable.
_C20_DECL_RE = re.compile(r"(?i)\bdisjoint\b[^.\n]{0,60}\bexhaustive\b[^:\n]{0,20}:")
_C20_CLAUSE_RE = re.compile(r"([^;⇔\n]{1,80})\s*⇔\s*([^;\n]+)")

# Tier 2: verdict-label anchor applied to a list item's FIRST bold span.
_C20_LABEL_RE = re.compile(
    r"(?i)^(?:h[-\s]?\w|intermediate|inconclusive|confirm|falsif|success|kill|pass\b|fail\b)"
)
_C20_BOLD_RE = re.compile(r"\*\*([^*\n]{1,80})\*\*")

# Atom grammar. POINT: `<qty> ≥/> 0` → pos, `≤/< 0` → neg (interior
# semantics, §4.4 convention); the `0(?!\.?\d)` lookahead keeps "p ≤ 0.05"
# out (a decimal alpha is c13's shape, not a sign atom).
_C20_POINT_RE = re.compile(r"(?P<qty>[^\s,;()]+)\s*(?P<cmp>≥|>=|≤|<=|>|<)\s*0(?!\.?\d)")
_C20_POINT_POS = ("≥", ">=", ">")

# CI atoms: a `CI`/`CIs` token, a tiny closed copula gap, then one idiom.
# Axis binding: `paired` within the 40 chars BEFORE the CI token (window
# clamped at the previous atom's span end — a preceding atom's own `paired`
# wording never leaks into this atom's binding) → paired axis, else primary.
# Idiom order matters: side-qualified excludes before the bare two-sided
# exclude.
_C20_CI_TOKEN_RE = re.compile(r"(?i)\bCIs?\b")
_C20_CI_GAP_RE = re.compile(r"(?:\s+(?:is|are|stays?|remains?))?\s*")
_C20_Z = r"(?:0|zero)(?!\.?\d)"
_C20_CI_IDIOMS: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (
        re.compile(r"(?i)exclud(?:es|ing)\s+" + _C20_Z + r"\s+on\s+the\s+positive\s+side"),
        frozenset({"above"}),
    ),
    (
        re.compile(r"(?i)exclud(?:es|ing)\s+" + _C20_Z + r"\s+on\s+the\s+negative\s+side"),
        frozenset({"below"}),
    ),
    (re.compile(r"(?i)strictly\s+positive\b"), frozenset({"above"})),
    (re.compile(r"(?i)strictly\s+negative\b"), frozenset({"below"})),
    (
        re.compile(r"(?i)wholly\s+(?:at\s+or\s+|at/)?above\s+" + _C20_Z),
        frozenset({"above"}),
    ),
    (re.compile(r"(?i)at\s+or\s+above\s+" + _C20_Z), frozenset({"above"})),
    (
        re.compile(r"(?i)wholly\s+below\s+" + _C20_Z + r"|below\s+zero\b"),
        frozenset({"below"}),
    ),
    (
        re.compile(r"(?i)(?:includes?|contains?|straddl(?:es?|ing)|overlaps?)\s+" + _C20_Z),
        frozenset({"straddle"}),
    ),
    (
        re.compile(r"(?i)exclud(?:es|ing)\s+" + _C20_Z + r"|clear\s+of\s+" + _C20_Z),
        frozenset({"below", "above"}),
    ),
]

# Negated-existence atoms (#1960; incident #1946): the prose form
#   ``no <family-ref> <unit-noun> is|are <inner predicate>``
# and the canonical machine form ``count(<family-ref>) == 0``. The span
# consumes the leading NEGATOR (``no``/``zero``) so it never lands in
# _C20_RESIDUE_RE residue; the prose inner predicate is a bounded run that
# stops at ;,. comparators and newline, and BEFORE a top-level AND/OR
# joiner — so a following sign/CI atom joins via the normal connective
# gap, never swallowed (a ``with`` INSIDE the span is atom text, not a
# joiner). The family-ref is BOUNDED by a closed unit-noun set so bare
# "no <anything>" prose ("no binary verdict" — an _C20_OTHERWISE_RE token;
# "no Δ_pool CI includes 0" — `CI` is not a unit noun) never matches and
# stays residue (fail-closed). Semantics: ONE boolean cell-algebra axis
# per lattice — values {zero, nonzero}; the atom binds {zero}; the
# nonzero side is reachable only via ``⇔ otherwise``.
_C20_NOEXIST_UNITS = r"(?:contrasts?|comparisons?|tests?|cells?|pairs?|arms?)"
_C20_NOEXIST_PROSE_RE = re.compile(
    r"(?i)\b(?:no|zero)\s+(?P<family>(?:[\w`*./_-]+\s+){0,5}?" + _C20_NOEXIST_UNITS + r")\s+"
    r"(?:is|are)\s+(?P<pred>(?:(?!\b(?:and|or)\b)[^;,.<>≤≥\n])+)"
)
_C20_NOEXIST_COUNT_RE = re.compile(
    r"(?i)\bcount\(\s*(?P<family>[^()\n]{1,120}?)\s*\)\s*==?\s*0(?!\.?\d)"
)

# OTHERWISE atom (complement label — fires iff no non-otherwise label fires).
_C20_OTHERWISE_RE = re.compile(
    r"(?i)\botherwise\b|\ball other\b|\bneither\b[^.;\n]{0,40}\bfires?\b|\bno binary verdict\b"
)

# Completeness-gate residue tokens: any CI token, comparator char, idiom
# keyword, or NEGATOR (Must-Fix: "the CI never includes 0" would otherwise
# parse as the positive atom with inverted polarity) OUTSIDE every recognized
# atom span makes the label `unparsed` — the lattice is then never
# FAIL-capable (WARN).
_C20_RESIDUE_RE = re.compile(
    r"(?i)\bCIs?\b|[<>≤≥]"
    r"|\binclud\w*|\bexclud\w*|\bstraddl\w*|\bwholly\b|\bstrictly\b|\bclear of\b"
    r"|\b(?:not|never|no|nor|unless|except|without)\b|\bfails?\s+to\b"
)

# Connectives: only AND / OR (incl. ", OR") / `with` (AND-equivalent) join
# atoms; any other joiner (bare comma, if/when chains, and/or → two hits)
# is fail-closed to `unparsed` — no silent default connective.
_C20_CONNECTIVE_RE = re.compile(r"(?i)\b(?:and|or|with)\b")

# Axis-identity fail-closed guard (ii): post-CI `paired` wording ("the CI of
# the paired difference includes 0") is never silently bound to an axis.
_C20_POST_CI_PAIRED_RE = re.compile(r"(?i)\bCIs?\b\s+(?:of|on|for|over)\s+(?:the\s+)?paired\b")

# Precedence-phrase screen: an order-evaluated lattice is coherent in a way
# the cell algebra cannot see → fail closed to `unparsed` (WARN).
_C20_PRECEDENCE_RE = re.compile(
    r"(?i)first matching|in (?:that |this )?order|takes precedence|evaluated in order|\bwins\b"
)

# Quantifier screen (tier 2): k-of-n / per-family predicates ("at >= 4/6
# pre-registered layers", "for all traits") are outside the v1 cell algebra
# -> SKIP.
# Deliberately NOT bare "every" (v6's recap says "for every … cell").
_C20_QUANT_RE = re.compile(
    r"(?i)(?:at least\s+\d+|≥\s*\d+|>=\s*\d+)\s*(?:of|/)\s*\d+|\ball\s+\d+\b|\bfor (?:all|each)\b"
)

# Threshold-form atoms (#1689/#1700 — the tier-1 analogue of _C20_QUANT_RE):
# ``<qty> <ineq> <nonzero>`` — ``rung ≥ 5``, ``≥ 8/9 pairs``, ``≤ 1 rung``,
# ``≥ 20 percent``. Deliberately mirrors ``_C20_POINT_RE`` (same qty/cmp
# capture shape) but the right-hand side captures ANY numeric literal
# (int / decimal / fraction) — the ``_c20_has_threshold_atom`` filter
# EXCLUDES ``0``, ``0.xxx`` and ``0/N`` after the fact so the sign-atom
# scope (``qty ≥/> 0``) stays owned by ``_C20_POINT_RE``.
_C20_THRESHOLD_RE = re.compile(
    r"(?P<qty>[^\s,;()]+)\s*(?P<cmp>≥|>=|≤|<=|>|<)\s*(?P<thr>\d+(?:[./]\d+)?)"
)


def _c20_has_threshold_atom(segment: str) -> bool:
    """True when ``segment`` carries a non-zero threshold-form inequality
    atom (``rung ≥ 5``, ``≥ 8/9 pairs``, ``≤ 1 rung``). Zero-right-hand-side
    matches (``qty ≥ 0``, ``qty ≥ 0.5``, ``qty ≥ 0/N``) are EXCLUDED —
    those are sign atoms and stay with ``_C20_POINT_RE``. This is the
    tier-1 detector for the #1689 shape (``rung ≥ 5`` / ``≥ 8/9 pairs``);
    the parser correctly rejects them as outside the v1 cell algebra, so
    ``_c20_tier1_result`` uses this + an ``⇔ otherwise`` complement to
    SKIP the whole lattice (mirroring the tier-2 ``_C20_QUANT_RE`` SKIP)."""
    for m in _C20_THRESHOLD_RE.finditer(segment):
        thr = m.group("thr")
        # Sign-atom exclusions (owned by _C20_POINT_RE): bare "0", "0.xxx"
        # decimals, and "0/N" fractions.
        if thr == "0" or thr.startswith("0.") or thr.startswith("0/"):
            continue
        return True
    return False


# Tier-2 segment machinery: sentence split, →/Consequence truncation, the
# "confirmed if(f)" selector.
_C20_SENT_SPLIT_RE = re.compile(r"(?<=\.)\s+")
_C20_TRUNC_RE = re.compile(r"→|\bConsequence\b")
_C20_CONFIRMED_RE = re.compile(r"(?i)\bconfirmed\s+iff?\b")
# Tier-1 clause predicates truncate at the first sentence terminator so a
# trailing recap sentence (v6's "Exactly one label fires for every … cell.")
# never enters the otherwise clause as residue.
_C20_SENT_END_RE = re.compile(r"\.(?=\s|$)")

_C20_CI_STATES = ("below", "straddle", "above")


def _c20_trigger_sections(plan: str) -> list[str]:
    """Fence-stripped texts of the OUTERMOST sections whose heading matches
    the c20 trigger families (a nested matching heading inside an
    already-taken section is not re-collected)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    taken: list[tuple[int, int]] = []
    out: list[str] = []
    for h in _headings(plan):
        if not _C20_SECTION_RE.search(h.text):
            continue
        if any(s <= h.line and h.end <= e for s, e in taken):
            continue
        taken.append((h.line, h.end))
        out.append("\n".join(lines[j] for j in range(h.line, h.end) if not mask[j]))
    return out


def _c20_label_name(bold: str) -> str:
    """Short display name for a harvested label: the bold span up to its
    first parenthetical annotation, trailing colon stripped."""
    return bold.split(" (")[0].rstrip(": ").strip()


def _c20_norm_family(fam: str) -> str:
    """Normalized axis key for a negated-existence family-ref: casefold,
    backticks/asterisks stripped, whitespace collapsed, trailing unit noun
    singularized. Axis sharing requires IDENTICAL normalized keys; >1
    distinct key per lattice fails closed to ``unparsed`` in
    ``_c20_evaluate_lattice`` — light normalization errs toward WARN,
    never a false FAIL."""
    s = re.sub(r"\s+", " ", fam.replace("`", "").replace("*", "").casefold()).strip()
    return re.sub(r"(contrast|comparison|test|cell|pair|arm)s$", r"\1", s)


def _c20_any_ci_idiom(text: str) -> bool:
    """True when ``text`` carries at least one CI-predicate idiom (the
    harvest condition — presence-only; atom adjacency is parse-time)."""
    return any(pat.search(text) for pat, _ in _C20_CI_IDIOMS)


def _c20_harvest_labels(section_text: str) -> list[dict]:
    """Tier-2 label harvest over one (fence-stripped) trigger section:
    top-level list items whose FIRST bold span matches the verdict-label
    anchor AND whose text carries a CI idiom (or an otherwise-token — an
    idiom-free complement label like "**Inconclusive:** otherwise" still
    joins the lattice it completes). Returns
    ``[{name, text, idiom}]`` in document order."""
    labels: list[dict] = []
    for block in _hypothesis_blocks(section_text):
        first_line = block.splitlines()[0] if block else ""
        if not _C14_LIST_ITEM_RE.match(first_line):
            continue
        bm = _C20_BOLD_RE.search(block)
        if bm is None:
            continue
        bold = bm.group(1).strip()
        if not _C20_LABEL_RE.match(bold):
            continue
        has_idiom = _c20_any_ci_idiom(block)
        if not (has_idiom or _C20_OTHERWISE_RE.search(block)):
            continue
        labels.append(
            {"name": _c20_label_name(bold), "text": block[bm.end() :], "idiom": has_idiom}
        )
    return labels


def _c20_has_atom(sentence: str) -> bool:
    """True when ``sentence`` carries a full parseable atom (point, CI, or
    otherwise) — idiom presence alone does not count (a CI idiom with no
    adjacent CI token is a residue shape, not an atom)."""
    if _C20_POINT_RE.search(sentence) or _C20_OTHERWISE_RE.search(sentence):
        return True
    for m in _C20_CI_TOKEN_RE.finditer(sentence):
        gm = _C20_CI_GAP_RE.match(sentence, m.end())
        if any(pat.match(sentence, gm.end()) for pat, _ in _C20_CI_IDIOMS):
            return True
    return False


def _c20_segment(label_text: str) -> tuple[str | None, str | None]:
    """``(predicate_segment, unparsed_reason)`` for a tier-2 label: the
    sentence containing "confirmed if(f)" when present, else the SINGLE
    atom-bearing sentence; each sentence truncated at the first ``→`` /
    ``Consequence`` token. >1 atom-bearing sentence without a confirmed-iff
    selector is ambiguous → unparsed."""
    sentences = [_C20_TRUNC_RE.split(s)[0] for s in _C20_SENT_SPLIT_RE.split(label_text)]
    confirmed = [s for s in sentences if _C20_CONFIRMED_RE.search(s)]
    if confirmed:
        return confirmed[0], None
    bearing = [s for s in sentences if _c20_has_atom(s)]
    if len(bearing) > 1:
        return None, ">1 atom-bearing sentence and no 'confirmed if(f)' selector — ambiguous"
    if not bearing:
        return None, "no sentence with a parseable atom"
    return bearing[0], None


def _c20_collect_atoms(
    segment: str,
) -> tuple[list[tuple[str, frozenset[str], int, int]], set, set]:
    """All sign/CI/negated-existence atoms in ``segment`` as
    ``(axis, values, start, end)`` (sorted by position) plus the set of
    normalized POINT quantities and the set of normalized negated-existence
    family keys. Negated-existence atoms are collected FIRST so their span
    consumes the leading negator (no residue) and suppresses point/CI
    matches falling inside it (#1960). The CI axis-binding lookback is
    clamped at the previous atom's span end so a preceding atom's `paired`
    token never mis-binds a later primary atom."""
    atoms: list[tuple[str, frozenset[str], int, int]] = []
    qtys: set[str] = set()
    fams: set[str] = set()
    neg_spans: list[tuple[int, int]] = []
    for pat in (_C20_NOEXIST_PROSE_RE, _C20_NOEXIST_COUNT_RE):
        for m in pat.finditer(segment):
            if any(s < m.end() and m.start() < e for s, e in neg_spans):
                continue  # already covered by an earlier negated-existence span
            fams.add(_c20_norm_family(m.group("family")))
            atoms.append(("famneg", frozenset({"zero"}), m.start(), m.end()))
            neg_spans.append((m.start(), m.end()))
    for m in _C20_POINT_RE.finditer(segment):
        if any(s < m.end() and m.start() < e for s, e in neg_spans):
            continue  # inside a negated-existence span (e.g. count(... > 0) == 0)
        sign = "pos" if m.group("cmp") in _C20_POINT_POS else "neg"
        qtys.add(m.group("qty").strip("`*").casefold())
        atoms.append(("point", frozenset({sign}), m.start(), m.end()))
    for m in _C20_CI_TOKEN_RE.finditer(segment):
        if any(s < m.end() and m.start() < e for s, e in neg_spans):
            continue  # CI token inside a negated-existence span
        gm = _C20_CI_GAP_RE.match(segment, m.end())
        hit: tuple[re.Match[str], frozenset[str]] | None = None
        for pat, states in _C20_CI_IDIOMS:
            im = pat.match(segment, gm.end())
            if im:
                hit = (im, states)
                break
        if hit is None:
            continue  # the stray CI token becomes completeness residue
        if any(s < hit[0].end() and m.start() < e for s, e in neg_spans):
            # The idiom tail crosses into a negated-existence span (a prose
            # match anchored on the idiom's own `zero` token): suppress the
            # CI atom — its CI/idiom tokens then land as residue, degrading
            # the label to `unparsed` (fail-closed, never a wrong parse).
            continue
        # Clamp the lookback at the previous atom's span end: a paired atom
        # < 40 chars BEFORE a primary atom would otherwise leak its `paired`
        # token into THIS atom's window, binding both atoms to the paired
        # axis — a contradictory conjunction that never fires, manufacturing
        # a tier-1 gap → false FAIL (round-1 code-review Minor).
        prev_end = max((a[3] for a in atoms if a[3] <= m.start()), default=0)
        lookback = segment[max(0, m.start() - 40, prev_end) : m.start()].lower()
        axis = "paired" if "paired" in lookback else "primary"
        atoms.append((axis, hit[1], m.start(), hit[0].end()))
    atoms.sort(key=lambda a: a[2])
    return atoms, qtys, fams


def _c20_build_dnf(
    segment: str, atoms: list[tuple[str, frozenset[str], int, int]]
) -> tuple[list[list[tuple[str, frozenset[str]]]] | None, str | None]:
    """``(dnf, None)`` for the atom chain under AND > OR precedence with the
    connective fail-closed rule, or ``(None, reason)``."""
    for i in range(1, len(atoms)):
        if atoms[i][2] < atoms[i - 1][3]:
            return None, "overlapping atom spans"
    conns: list[str] = []
    for i in range(1, len(atoms)):
        gap = segment[atoms[i - 1][3] : atoms[i][2]]
        found = [c.lower() for c in _C20_CONNECTIVE_RE.findall(gap)]
        if len(found) != 1:
            return None, f"joiner between atoms is not exactly one of AND/OR/with ({gap.strip()!r})"
        conns.append(found[0])
    groups: list[list[tuple[str, frozenset[str]]]] = [[(atoms[0][0], atoms[0][1])]]
    for i, conn in enumerate(conns, start=1):
        if conn == "or":
            groups.append([(atoms[i][0], atoms[i][1])])
        else:  # and / with — AND-equivalent
            groups[-1].append((atoms[i][0], atoms[i][1]))
    return groups, None


def _c20_parse_predicate(segment: str) -> dict:
    """Compile one predicate segment to DNF over sign/CI atoms (or an
    otherwise-label). Fail-closed: any completeness-gate residue (stray CI
    token / comparator / idiom keyword / NEGATOR), any non-AND/OR/with
    joiner between atoms, or an otherwise-token mixed with predicate atoms
    marks the segment ``unparsed`` (reason in the returned dict)."""
    out: dict = {
        "otherwise": False,
        "dnf": [],
        "unparsed": None,
        "point_qtys": set(),
        "famneg_keys": set(),
    }
    atoms, out["point_qtys"], out["famneg_keys"] = _c20_collect_atoms(segment)
    otherwise_spans = [(m.start(), m.end()) for m in _C20_OTHERWISE_RE.finditer(segment)]
    if otherwise_spans and atoms:
        out["unparsed"] = "an 'otherwise' token mixed with predicate atoms in one segment"
        return out
    spans = otherwise_spans if otherwise_spans else [(a[2], a[3]) for a in atoms]
    residues = [
        m.group(0)
        for m in _C20_RESIDUE_RE.finditer(segment)
        if not any(s <= m.start() and m.end() <= e for s, e in spans)
    ]
    if residues:
        out["unparsed"] = "predicate token(s) outside every recognized atom: " + ", ".join(
            repr(r) for r in residues[:4]
        )
        return out
    if otherwise_spans:
        out["otherwise"] = True
        return out
    if not atoms:
        out["unparsed"] = "no recognized atom"
        return out
    dnf, reason = _c20_build_dnf(segment, atoms)
    if dnf is None:
        out["unparsed"] = reason
        return out
    out["dnf"] = dnf
    return out


def _c20_enumerate(labels: list[dict]) -> tuple[list, list]:
    """Interior-cells-only 3-state enumeration over the REFERENCED axes with
    point-in-CI coherence pruning (a bootstrap CI contains its point
    estimate). Returns ``(cofires, gaps)`` — cofires as ``(cell, [label
    names])``, gaps as bare cells. An otherwise-label fires exactly on the
    cells no predicate label covers (killing gap findings by construction)."""
    preds = [lab for lab in labels if not lab["parse"]["otherwise"]]
    others = [lab for lab in labels if lab["parse"]["otherwise"]]
    axes = {axis for lab in preds for conj in lab["parse"]["dnf"] for axis, _ in conj}
    primary_vals: tuple = _C20_CI_STATES if "primary" in axes else (None,)
    paired_vals: tuple = _C20_CI_STATES if "paired" in axes else (None,)
    # Family-negation axis (#1960): boolean {zero, nonzero}; the atom binds
    # {zero}, the nonzero side is reachable only via an otherwise-label.
    famneg_vals: tuple = ("zero", "nonzero") if "famneg" in axes else (None,)
    cofires: list[tuple[dict, list[str]]] = []
    gaps: list[dict] = []
    for primary in primary_vals:
        if "point" not in axes:
            point_vals: tuple = (None,)
        elif primary is None:
            point_vals = ("neg", "pos")
        else:
            point_vals = {"below": ("neg",), "straddle": ("neg", "pos"), "above": ("pos",)}[primary]
        for point in point_vals:
            for paired in paired_vals:
                for famneg in famneg_vals:
                    cell = {
                        "point": point,
                        "primary": primary,
                        "paired": paired,
                        "famneg": famneg,
                    }
                    fired = [
                        lab
                        for lab in preds
                        if any(
                            all(cell[axis] in values for axis, values in conj)
                            for conj in lab["parse"]["dnf"]
                        )
                    ]
                    if not fired and others:
                        fired = others
                    if len(fired) >= 2:
                        cofires.append((cell, [lab["name"] for lab in fired]))
                    elif not fired:
                        gaps.append(cell)
    return cofires, gaps


def _c20_cell_str(cell: dict) -> str:
    """Plain-terms cell rendering for FAIL/WARN details."""
    parts: list[str] = []
    if cell["point"] is not None:
        parts.append("point > 0" if cell["point"] == "pos" else "point < 0")
    for axis in ("primary", "paired"):
        v = cell[axis]
        if v is not None:
            word = {"below": "wholly below 0", "straddle": "straddles 0", "above": "wholly above 0"}
            parts.append(f"{axis} CI {word[v]}")
    if cell.get("famneg") is not None:
        parts.append(
            "no family member fires" if cell["famneg"] == "zero" else "≥1 family member fires"
        )
    return "{" + ", ".join(parts) + "}"


_C20_REMEDY = (
    " — restate the lattice as an explicit partition (`DISJOINT and exhaustive: "
    "<label> ⇔ <predicate>; …; <label> ⇔ otherwise`), add an otherwise-label, or "
    "declare 'N/A — no registered verdict lattice' on its own line, unwrapped "
    "(no backticks/quotes)"
)


def _c20_offender_detail(tier_desc: str, cofires: list, gaps: list) -> str:
    """Bounded offender detail: co-fire cells with both label names first,
    gap cells as the secondary note, ≤4 shown each, remedy menu last."""
    bits: list[str] = []
    if cofires:
        shown = "; ".join(
            f"labels {' + '.join(names)} CO-FIRE on cell {_c20_cell_str(cell)}"
            for cell, names in cofires[:4]
        )
        if len(cofires) > 4:
            shown += "; …"
        bits.append(shown)
    if gaps:
        shown = ", ".join(_c20_cell_str(c) for c in gaps[:4])
        if len(gaps) > 4:
            shown += ", …"
        bits.append(f"no label fires on cell(s) {shown}")
    return (
        f"the registered verdict lattice ({tier_desc}) is not a partition: "
        + "; ".join(bits)
        + _C20_REMEDY
    )


def _c20_evaluate_lattice(labels: list[dict], *, tier: int, section_text: str) -> tuple[str, str]:
    """Shared per-lattice verdict core → ``(state, detail)`` with state in
    {"unparsed", "cofire", "gap", "clean"}. The kind/tier degradations
    (§4.5 table) are applied by the caller."""
    names = " / ".join(lab["name"] for lab in labels)
    tier_desc = f"tier {tier}: {names}"
    pm = _C20_PRECEDENCE_RE.search(section_text)
    if pm:
        return (
            "unparsed",
            f"label-precedence phrase {pm.group(0)!r} in the lattice's section makes the "
            "labels order-evaluated — the cell algebra cannot verify an ordered lattice; "
            "restate it as the explicit ⇔ partition form",
        )
    unparsed = [lab for lab in labels if lab["parse"]["unparsed"]]
    if unparsed:
        first = unparsed[0]
        return (
            "unparsed",
            f"label '{first['name']}' ({tier_desc}) did not fully parse: "
            f"{first['parse']['unparsed']} — the lattice is not FAIL-capable; restate it as "
            "the explicit ⇔ partition form (`DISJOINT and exhaustive: <label> ⇔ <predicate>; "
            "…`) so coherence is machine-checkable",
        )
    qtys = set()
    for lab in labels:
        qtys |= lab["parse"]["point_qtys"]
    if len(qtys) > 1:
        return (
            "unparsed",
            f"the lattice's labels reference {len(qtys)} distinct point quantities "
            f"({', '.join(sorted(qtys)[:4])}) — a single-point-axis cell algebra cannot "
            "represent them (never silently collapsed onto one axis); restate the lattice "
            "over one point quantity or use the explicit ⇔ partition form",
        )
    fams = set()
    for lab in labels:
        fams |= lab["parse"].get("famneg_keys", set())
    if len(fams) > 1:
        return (
            "unparsed",
            f"the lattice's labels reference {len(fams)} distinct negated-existence "
            f"families ({', '.join(sorted(fams)[:4])}) — the v1 cell algebra carries a "
            "single family-negation axis (never silently unified); restate over one "
            "family with identical wording (prose or count(...) == 0 form, not mixed)",
        )
    cofires, gaps = _c20_enumerate(labels)
    if cofires or gaps:
        detail = _c20_offender_detail(tier_desc, cofires, gaps)
        return ("cofire" if cofires else "gap", detail)
    return (
        "clean",
        f"{tier_desc} — every interior sign/CI cell fires exactly one label "
        "(partition verified in form; boundary semantics stay with the Statistics critic)",
    )


_C20_POST_CI_PAIRED_REASON = (
    "post-CI 'paired' wording (e.g. 'the CI of the paired difference') is "
    "not silently bound to an axis"
)


def _c20_find_declaration(sections: list[str]) -> tuple[str, list[tuple[str, str]]] | None:
    """First DISJOINT-and-exhaustive ⇔ declaration across the trigger
    sections → ``(section_text, [(label, predicate), …])``; None when no
    declaration line exists (tier 2 then applies)."""
    for sec in sections:
        for line in sec.splitlines():
            dm = _C20_DECL_RE.search(line)
            if not dm:
                continue
            clauses = []
            for chunk in line[dm.end() :].split(";"):
                cm = _C20_CLAUSE_RE.match(chunk)
                if cm:
                    clauses.append((cm.group(1).strip(), cm.group(2).strip()))
            return sec, clauses
    return None


def _c20_tier1_result(cid: str, name: str, kind: str, sec: str, clauses: list) -> CheckResult:
    """Tier-1 verdict: the plan CLAIMED a partition, so co-fire AND gap are
    both FAIL-capable (WARN under kind=analysis); unparsed clauses WARN.
    #1689/#1700: when the tier-1 lattice uses THRESHOLD-form inequality
    predicates (``rung ≥ 5``, ``≥ 8/9 pairs``, ``≤ 1 rung``) AND carries
    an ``⇔ otherwise`` complement, the partition is exhaustive-by-
    construction (the otherwise clause covers every residual cell) and
    the co-fire risk on threshold predicates is Statistics-critic scope,
    outside the v1 cell algebra — SKIP (mirrors the tier-2
    ``_C20_QUANT_RE`` SKIP). Threshold-only lattices WITHOUT an
    ``⇔ otherwise`` complement still route through the normal parse
    (which correctly WARNs on the unparsed residue)."""
    if len(clauses) < 2:
        return _warn(
            cid,
            name,
            "a DISJOINT-and-exhaustive declaration was found but fewer than 2 "
            "`<label> ⇔ <predicate>` clauses parsed from it — the claimed partition is "
            "not machine-checkable; use the canonical form (`DISJOINT and exhaustive: "
            "<label> ⇔ <predicate>; …; <label> ⇔ otherwise`)",
        )
    # #1689/#1700: threshold-atoms + ⇔ otherwise ⇒ SKIP (see docstring).
    has_otherwise = any(_C20_OTHERWISE_RE.search(cpred) for _, cpred in clauses)
    uses_thresholds = any(_c20_has_threshold_atom(cpred) for _, cpred in clauses)
    if has_otherwise and uses_thresholds:
        return _skip(
            cid,
            name,
            f"tier-1 lattice ({len(clauses)} clauses) uses threshold-form inequality "
            "predicates (e.g. `rung ≥ 5`, `≥ 8/9 pairs`) closed by an `⇔ otherwise` "
            "complement — the partition is exhaustive-by-construction and the "
            "co-fire risk on threshold predicates is Statistics-critic scope, out "
            "of v1 cell algebra (mirrors the tier-2 k-of-n SKIP; #1689/#1700)",
        )
    labels = []
    for clabel, cpred in clauses:
        pred = _C20_SENT_END_RE.split(cpred)[0]
        parse = _c20_parse_predicate(pred)
        if _C20_POST_CI_PAIRED_RE.search(pred):
            parse["unparsed"] = _C20_POST_CI_PAIRED_REASON
        labels.append({"name": _c20_label_name(clabel), "parse": parse})
    state, detail = _c20_evaluate_lattice(labels, tier=1, section_text=sec)
    if state == "clean":
        return _pass(cid, name, detail)
    if state == "unparsed":
        return _warn(cid, name, detail)
    if kind == "analysis":
        return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
    return _fail(cid, name, detail)


def _c20_tier2_result(cid: str, name: str, kind: str, lattices: list) -> CheckResult:
    """Tier-2 verdict over every qualifying section's lattice (worst wins):
    complete-parse co-fire FAILs (WARN under kind=analysis); gap-only and
    any-unparsed WARN; any quantified label SKIPs the whole check."""
    worst: tuple[int, str, str] | None = None  # (rank, state, detail)
    rank = {"clean": 0, "gap": 1, "unparsed": 2, "cofire": 3}
    for sec, labels in lattices:
        for lab in labels:
            seg, reason = _c20_segment(lab["text"])
            if seg is not None and _C20_QUANT_RE.search(seg):
                return _skip(
                    cid,
                    name,
                    f"label '{lab['name']}' carries quantified verdict predicates out of v1 "
                    "scope (k-of-n / per-family lattices are the Statistics critic's)",
                )
            if _C20_POST_CI_PAIRED_RE.search(lab["text"]):
                seg, reason = None, _C20_POST_CI_PAIRED_REASON
            if reason is not None:
                lab["parse"] = {
                    "otherwise": False,
                    "dnf": [],
                    "unparsed": reason,
                    "point_qtys": set(),
                    "famneg_keys": set(),
                }
            else:
                lab["parse"] = _c20_parse_predicate(seg)
        state, detail = _c20_evaluate_lattice(labels, tier=2, section_text=sec)
        if worst is None or rank[state] > worst[0]:
            worst = (rank[state], state, detail)
    assert worst is not None  # ≥1 lattice on this branch
    _, state, detail = worst
    if state == "clean":
        return _pass(cid, name, detail)
    if state == "cofire" and kind == "experiment":
        return _fail(cid, name, detail)
    if state == "cofire":
        return _warn(cid, name, detail + " (analysis kind-degrade: WARN, not FAIL)")
    if state == "gap":
        return _warn(
            cid,
            name,
            detail + " (tier-2 gap degrades to WARN: gap precision depends on harvest recall)",
        )
    return _warn(cid, name, detail)


def check_verdict_lattice_coherence(plan: str, kind: str) -> CheckResult:
    """A REGISTERED VERDICT LATTICE — success/kill/intermediate labels
    defined by interval predicates over point estimates and CIs — must be
    mutually exclusive and exhaustive over the interior sign/CI cells.
    Tier 1 (the explicit "DISJOINT and exhaustive: <label> ⇔ <predicate>"
    declaration) is FAIL-capable on co-fire AND gap (the plan claimed a
    partition); tier 2 (per-label prose, the #923 v4 shape) FAILs only on a
    co-fire with a COMPLETE parse — gaps degrade to WARN (gap precision
    depends on harvest recall), any unparsed label degrades the whole
    lattice to WARN, and quantified (k-of-n) predicates SKIP as out of the
    v1 cell algebra. A negated-existence conjunct (``no <family-ref>
    <unit-noun> is <predicate>``, or the canonical ``count(<family-ref>)
    == 0`` machine form) parses as a single boolean family-negation axis
    {zero, nonzero} whose nonzero side is reachable only via ``⇔
    otherwise`` (#1960; incident #1946); >1 distinct normalized family per
    lattice fails closed to WARN. FAIL (experiment) / WARN (analysis) /
    SKIP otherwise;
    escape via a standalone ``N/A — no registered verdict lattice`` line —
    honored (SKIP path) only when no lattice is detected; when the escape
    co-occurs with a detected lattice (either tier) the check WARNs instead
    of PASSing, so the escape can never mask verification of a present
    lattice (#1223).
    Incident: #923 amendment plan v4/v5 §3 — a bare positive point estimate
    with both CIs straddling 0 fired BOTH H-slot and Intermediate (and one
    cell fired neither); caught only by the Codex statistics critic, fixed
    by hand in v6."""
    cid, name = "c20_verdict_lattice_coherence", "verdict-lattice coherence"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: registered verdict lattices are an experiment|analysis shape"
        )
    sections = _c20_trigger_sections(plan)
    # Tier 1 takes precedence over tier 2 when a declaration exists anywhere.
    tier1 = _c20_find_declaration(sections)
    lattices: list[tuple[str, list[dict]]] = []
    if tier1 is None:
        for sec in sections:
            labels = _c20_harvest_labels(sec)
            if sum(1 for lab in labels if lab["idiom"]) >= 2:
                lattices.append((sec, labels))
    if tier1 is None and not lattices:
        return _skip(
            cid,
            name,
            "no registered verdict lattice detected (no DISJOINT-and-exhaustive ⇔ "
            "declaration; fewer than 2 anchored CI-predicate labels in any trigger section)",
        )
    if _standalone_na_declared(plan, r"no registered verdict lattice"):
        # #1223: this branch is reachable ONLY when a lattice WAS detected (the
        # no-lattice case SKIPs above) — a PASS here masks the very defect c20
        # exists to catch. WARN, not FAIL: the detection may be a false positive
        # on quoted guidance (all 3 corpus co-occurrences were), so the escape
        # stays non-blocking and the reviewers adjudicate.
        detected = (
            "a tier-1 DISJOINT-and-exhaustive ⇔ declaration"
            if tier1 is not None
            else f"{len(lattices)} trigger section(s) with ≥2 anchored CI-predicate labels (tier 2)"
        )
        return _warn(
            cid,
            name,
            "the standalone `N/A — no registered verdict lattice` escape co-occurs "
            f"with {detected} — the escape is reserved for lattice-free plans and "
            "would mask coherence verification of the detected lattice (#1223); "
            "remove the N/A line (the lattice is then verified), or fence/remove the "
            "lattice-shaped prose the detector matched if it is quoted guidance "
            "rather than this plan's own registration",
        )
    if tier1 is not None:
        return _c20_tier1_result(cid, name, kind, tier1[0], tier1[1])
    return _c20_tier2_result(cid, name, kind, lattices)


# ─── Check 21 — grep-arity acceptance gate → AST arity audit (WARN-only) ──

# A grep invocation whose quoted pattern is call-shaped: an identifier
# immediately followed by `(` inside the quotes
# (`grep -rn "parse_judge_json(" ...`). [^|\n] bounds the scan to the
# grep component's own arguments, not a later pipeline component.
_C21_GREP_CALL_RE = re.compile(r"""grep\b[^|\n]*["'][^"'\n]*\w\(""")

# Pipeline-form arity discriminator: any grep component on the SAME line
# whose quoted pattern contains a comma (`| grep ", "`, `grep -c 'f(.*,'`).
_C21_GREP_COMMA_RE = re.compile(r"""grep\b[^|\n]*["'][^"'\n]*,[^"'\n]*["']""")

# Count form: `... | wc -l`, or a grep flag cluster carrying -c
# (`grep -c`, `grep -rnc`; a separated `grep -r -c` is a known miss).
_C21_COUNT_RE = re.compile(r"""wc\s+-l|\bgrep\s+-\w*c\b""")

# Prose-form arity vocabulary ("shows zero two-argument calls").
_C21_ARITY_VOCAB_RE = re.compile(
    r"(?i)\btwo-?arg\w*|\b(?:one|two|three|\d+)[- ]argument|\barity\b"
    r"|second argument|keyword[- ]arg\w*"
)

# Registered zero-count pass condition — the comparator that makes a grep
# a GATE rather than a discovery command. Deliberately absent: bare
# `\bempty\b` and un-bounded `→ 0` (matched unrelated prose on #416/#467/
# #870 in the calibration sweep).
_C21_ZERO_RE = re.compile(
    r"(?i)==?\s*`?0\b|\bshows zero\b|\bzero\b[^.\n]{0,40}\bcalls?\b"
    r"|returns nothing|\b0 hits\b|must be 0\b"
)

# Evidence escape: the plan names an AST-based arity audit anywhere.
_C21_AST_EVIDENCE_RE = re.compile(
    r"(?i)ast\.(?:walk|parse)|\bAST[- ](?:based|arity|audit|walker)|libcst"
)


def check_grep_arity_gate(plan: str, kind: str) -> CheckResult:
    """Plans registering a grep/`wc -l`-based signature-ARITY acceptance
    gate (`grep "func(" ... | grep ", " | wc -l` == 0, or a call-pattern
    grep whose stated pass condition is "shows zero two-argument calls")
    get a WARN pointing at the AST-based arity audit as the robust form:
    comma heuristics over call sites are BOTH unsatisfiable (they count
    deliberate two-arg tests + comma-bearing string literals) AND
    under-detecting (split-line and keyword-argument calls carry no
    same-line comma) — #1024 plan v1/v2 registered exactly this gate and
    the critic ensemble replaced it with an ast.walk audit in v3. WARN
    not FAIL: greps are legitimate for discovery/enumeration, and the
    conjunctive line trigger (call-pattern grep + arity discriminator +
    count/comparator) is a heuristic — the Phase 1.5/2 reviewers
    adjudicate. ALL kinds: the incident was kind: infra, but signature
    migrations also ride experiment plans' code-port phases, and the
    2026-07-04 corpus sweep (1,329 plans/v*.md) fired on ZERO lines
    outside #1024's own plan versions, so kind confinement buys no
    precision and costs recall. Raw lines are scanned WITHOUT the fence
    mask (gate commands live in inline backticks and fenced verification
    blocks alike); section-window confinement is the first tightening
    lever if a future sweep surfaces false positives. The 0-FP figure is
    an IN-SAMPLE calibration (regexes tuned on the same historical
    corpus the acceptance sweep re-runs) — it bounds nuisance cost on
    yesterday's planner distribution, not a guarantee for future plans."""
    cid, name = "c21_grep_arity_gate", "grep-arity acceptance gate points at AST audit"
    del kind  # all kinds — trigger precision carries the false-positive discipline
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(plan.splitlines(), 1):
        if not _C21_GREP_CALL_RE.search(line):
            continue
        pipeline = _C21_GREP_COMMA_RE.search(line) and _C21_COUNT_RE.search(line)
        prose = _C21_ARITY_VOCAB_RE.search(line) and _C21_ZERO_RE.search(line)
        if pipeline or prose:
            hits.append((i, line.strip()))
    if not hits:
        return _skip(cid, name, "no grep-based call-arity pass condition detected")
    if _standalone_na_declared(plan, r"no arity acceptance gate"):
        return _pass(
            cid, name, "explicit N/A declared (flagged grep is not an arity pass condition)"
        )
    if _C21_AST_EVIDENCE_RE.search(plan):
        i, line = hits[0]
        return _pass(
            cid,
            name,
            f"grep-arity gate present (line {i}: {line[:80]!r}) but the plan also names an "
            "AST-based arity audit — the robust form is registered",
        )
    i, line = hits[0]
    return _warn(
        cid,
        name,
        f"a registered pass condition counts comma-bearing call-pattern grep hits (line {i}: "
        f"{line[:100]!r}) — comma-grep arity gates are both unsatisfiable (they count "
        "deliberate two-arg tests and comma-bearing string literals) and under-detecting "
        "(split-line and keyword-argument calls carry no same-line comma; #1024 plan v1/v2). "
        "Register an AST arity audit instead: ast.parse each target file, ast.walk over Call "
        "nodes matching the function, count len(node.args) + len(node.keywords), whitelist "
        "named exceptions — or declare `N/A — no arity acceptance gate` on its own line, "
        "unwrapped (no backticks/quotes)",
    )


# ─── Check 22 — cross-section param consistency (WARN-only, all kinds) ─────

_C22_PARAM_TOKENS = (
    r"temperature|max_new_tokens|max_tokens|learning_rate|lr|epochs|"
    r"seeds|seed|rank|alpha|batch_size|batch|top_p"
)  # longer alternatives first where prefixes overlap
_C22_ALIASES = {"learning_rate": "lr", "seeds": "seed", "batch_size": "batch"}
_C22_NUM = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
# param=value / param: value; tolerates `code`/**bold** wrappers; captures a
# single numeric, a comma-run of numerics, or a {...} brace set (<=120 chars).
# The leading \b means compound tokens (JUDGE_TEMPERATURE=0.7) never match —
# underscore is \w, so there is no boundary before the bare param name.
_C22_VALUE_RE = re.compile(
    rf"(?i)\b(?P<param>{_C22_PARAM_TOKENS})\b\s*[=:]\s*[`*]{{0,2}}\s*"
    rf"(?P<vals>\{{[^}}\n]{{0,120}}\}}|{_C22_NUM}(?:\s*,\s*{_C22_NUM})*)"
)
# Range/schedule continuation right after the captured value ("1e-4 → 1e-5",
# "1-3", "1 -> 3"): the tail value joins the occurrence's value set.
_C22_RANGE_TAIL_RE = re.compile(
    rf"\s*[`*]{{0,2}}\s*(?:[-\u2013\u2014]|->|→)\s*[`*]{{0,2}}({_C22_NUM})"
)
# Omission assertion: the #1024 corrected-text shape ("temperature OMITTED").
_C22_OMIT_RE = re.compile(
    rf"(?i)\b(?P<param>{_C22_PARAM_TOKENS})\b\s+(?:is\s+)?(?:omitted|left\s+unset)\b"
)
# Historical / declared-but-never-threaded clause vocabulary (value
# occurrences only). `was` is value-adjacent only (`was 0.7` / `was set`),
# NOT bare \bwas\b — bare `was` is ubiquitous and would silently exclude
# CURRENT stale values on lines like "temperature=0.7 was chosen per #612".
_C22_EXCLUDE_RE = re.compile(
    rf"(?i)declared\s+but\s+never|never\s+threaded|not\s+threaded|never\s+used|"
    rf"\bpreviously\b|superseded|corrected\s+(?:from|to)|historical|\bstale\b|"
    rf"deprecated|no\s+longer|old\s+(?:value|default)|\bwas\s+(?:{_C22_NUM}|set\b|used\b)"
)
_C22_SWEEP_LINE_RE = re.compile(r"(?i)\bsweeps?\b|\bgrid\b|ablation")
_C22_PHASE_RE = re.compile(r"(?i)\bphase[\s-]*([0-9]+)\b")
_C22_LORA_CTX_RE = re.compile(r"(?i)lora|rslora|\brank\b|adapter|peft")

# Same-line character window around a value match inside which the
# historical-clause vocabulary excludes the occurrence (window-bounded so a
# very long line's distant vocabulary cannot wrongly exclude a live value).
_C22_EXCLUDE_WINDOW_CHARS = 100


def _c22_top_section(headings: list[Heading], line_idx: int) -> tuple[int, str]:
    """Top-level-section attribution for ``line_idx``: the SHALLOWEST heading
    of level >= 2 containing the line (the H2 ancestor — sibling H3
    subsections under one ``## 4. Design`` group as ONE section); falls back
    to ``_innermost_section`` for H1-only docs, else a synthetic preamble
    key. Returns ``(heading.line, heading.text)`` as the section key."""
    candidates = [h for h in headings if h.level >= 2 and h.line <= line_idx < h.end]
    if candidates:
        best = min(candidates, key=lambda h: h.level)
        return (best.line, best.text)
    inner = _innermost_section(headings, line_idx)
    if inner is not None:
        return (inner.line, inner.text)
    return (-1, "(preamble)")


def _c22_record(
    occ: dict[str, dict[tuple[int, str], dict]],
    key: str,
    section: tuple[int, str],
    vals: set,
    lineno: int,
    span: str,
) -> None:
    """Union ``vals`` into ``occ[key][section]``, keeping the FIRST matched
    ``(lineno, span)`` per (param, section) for the WARN detail."""
    recs = occ.setdefault(key, {})
    rec = recs.get(section)
    if rec is None:
        recs[section] = {"vals": set(vals), "lineno": lineno, "span": span}
    else:
        rec["vals"] |= vals


def _c22_collect_occurrences(plan: str) -> dict[str, dict[tuple[int, str], dict]]:
    """Build the (param-key → top-level section → {vals, lineno, span}) map.

    Fenced lines never vote (module convention). Value occurrences on
    sweep/grid/ablation lines, stats-``alpha`` outside LoRA context, and
    values inside a historical/never-threaded clause (same-line ±100-char
    window) are excluded. Literal omission assertions (``<param> OMITTED``)
    are EXEMPT from the exclusion filter — the corrected text legitimately
    reads "temperature OMITTED — the builders never set it": the clause
    explains the omission, it does not mark it historical. Phase-qualified
    lines key as ``<param>@phase<K>``."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    occ: dict[str, dict[tuple[int, str], dict]] = {}
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        pm = _C22_PHASE_RE.search(line)
        phase = f"@phase{pm.group(1)}" if pm else ""
        section = _c22_top_section(headings, i)
        for m in _C22_VALUE_RE.finditer(line):
            param = _C22_ALIASES.get(m["param"].lower(), m["param"].lower())
            if param == "alpha" and not _C22_LORA_CTX_RE.search(line):
                continue  # stats-alpha guard: significance level, not LoRA alpha
            if _C22_SWEEP_LINE_RE.search(line):
                continue  # sweep/grid/ablation declarations are legitimately multi-value
            w = _C22_EXCLUDE_WINDOW_CHARS
            window = line[max(0, m.start() - w) : m.end() + w]
            if _C22_EXCLUDE_RE.search(window):
                continue  # historical / declared-but-never-threaded clause
            vals: set = {float(v) for v in re.findall(_C22_NUM, m["vals"])}
            if not vals:
                continue  # non-numeric brace set
            tm = _C22_RANGE_TAIL_RE.match(line, m.end())
            if tm:
                vals.add(float(tm.group(1)))
            _c22_record(occ, param + phase, section, vals, i, m.group(0))
        for m in _C22_OMIT_RE.finditer(line):
            param = _C22_ALIASES.get(m["param"].lower(), m["param"].lower())
            _c22_record(occ, param + phase, section, {"OMITTED"}, i, m.group(0))
    return occ


def check_cross_section_param_consistency(plan: str, kind: str) -> CheckResult:
    """The same tracked hyperparameter stated with contradictory values in
    DIFFERENT top-level sections is the #1024 incident class: a fact-check
    correction lands in one section while a stale restatement survives in
    another (§11 *What:* lines, assumption rows). Tracked params:
    temperature, max_tokens / max_new_tokens (distinct keys — API judge cap
    vs HF generate cap), lr / learning_rate, epochs, seed / seeds, rank,
    alpha (LoRA context only), batch / batch_size, top_p. A conflict is a
    pair of top-level sections whose value SETS are disjoint — overlap is
    consistent, which is what lets per-arm tables, ranges/schedules, and
    seed lists restated against a member value all PASS while ``0.7`` vs
    ``OMITTED``/``1.0`` WARNs. v1 scope: value-vs-value plus
    value-vs-literal-omission-token (``<param> OMITTED`` / ``left unset``) —
    the #1024 v2 offender shape; broader omission phrasings ("builders omit
    temperature", "no temperature parameter") are OUT of v1 scope.
    WARN-only, never FAIL (legitimate multi-value plans exist, and Phase
    1.5.0 forwards WARNs verbatim into the fact-checker/critic briefs — the
    intended consumption path); ALL kinds (the motivating #1024 offender is
    ``kind: infra``); conditional SKIP when no tracked param spans >= 2
    top-level sections.

    Documented v1 limits: (i) half-corrected-section masking — a section
    carrying BOTH a stale ``temperature=0.7`` and the corrected
    "temperature OMITTED" unions to {0.7, OMITTED}, which overlaps a §11
    {0.7} → PASS; intra-section contradictions are out of v1 cross-section
    scope. (ii) Phase-qualifier asymmetry — a phase-qualified occurrence
    (``epochs@phase1``) never compares against an unqualified ``epochs=3``;
    a c22 PASS is not "no cross-section drift" for phase-keyed params.
    (iii) Markdown-table blindness — the value regex requires ``=`` or
    ``:``, so pipe-table hyperparameter rows (``| lr | 1e-4 |``) never
    parse; the table-vs-prose restatement class is invisible to v1."""
    cid, name = "c22_cross_section_param_consistency", "cross-section param consistency"
    del kind  # registry symmetry, c5-style: c22 runs for ALL kinds (#1024 is kind: infra)
    occ = _c22_collect_occurrences(plan)
    cross = {k: recs for k, recs in occ.items() if len(recs) >= 2}
    if not cross:
        return _skip(cid, name, "no cross-section parameter restatement detected")
    conflicts: list[tuple[str, tuple, tuple]] = []
    for key, recs in cross.items():
        sections = list(recs.items())
        found = None
        for a in range(len(sections)):
            for b in range(a + 1, len(sections)):
                if sections[a][1]["vals"].isdisjoint(sections[b][1]["vals"]):
                    found = (key, sections[a], sections[b])
                    break
            if found:
                break
        if found:
            conflicts.append(found)  # first disjoint pair reported per param
    if not conflicts:
        return _pass(
            cid, name, f"{len(cross)} parameter(s) restated across sections, all consistent"
        )
    parts = []
    for key, (sec_a, rec_a), (sec_b, rec_b) in conflicts[:2]:
        parts.append(
            f"{key}: '{rec_a['span']}' (§'{sec_a[1]}' L{rec_a['lineno'] + 1}) vs "
            f"'{rec_b['span']}' (§'{sec_b[1]}' L{rec_b['lineno'] + 1})"
        )
    more = len(conflicts) - 2
    detail = (
        "; ".join(parts)
        + (f" …and {more} more param(s)" if more > 0 else "")
        + " — cross-section contradiction; if one side is a stale post-correction "
        "restatement, fix it"
    )
    return _warn(cid, name, detail)


# ─── Check 23 — goal currency (outside CHECKS; --issue mode only) ─────────

# Word-shingle stale-quote detector for the #922 plan-vs-goal incident class:
# a plan head quoting a SUPERSEDED Goal at high coverage while the CURRENT
# Goal is absent. WARN-only (the c21 WARN-first precedent, #1042); the
# forced redraft is delivered by the adversarial-planner SKILL.md
# § Goal-currency gate ("the one WARN that bounces"). Needs task context
# (body.md + events.jsonl), so it lives OUTSIDE verify_plan_text() and is
# appended by main() in --issue mode only.
_C23_SHINGLE_K = 6
_C23_MIN_GOAL_WORDS = 12
_C23_STALE_COV = 0.5
_C23_CURRENT_COV = 0.3
# NO positive slack: retro-stale goal-update gaps of ~3-6 min exist in the
# corpus (779/477/489) — any slack ≥ ~3 min manufactures false positives.
_C23_MTIME_SLACK_S = 0.0


def _norm_goal_words(s: str) -> list[str]:
    """Lowercase; non-alphanumerics (incl. unicode math) -> space; split."""
    return [w for w in re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split() if w]


def _goal_shingles(words: list[str], k: int = _C23_SHINGLE_K) -> set[tuple[str, ...]]:
    """All contiguous k-word shingles of ``words`` (empty set below k words)."""
    if len(words) < k:
        return set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _shingle_coverage(goal: str, head_words: list[str]) -> float:
    """Fraction of the goal's k-word shingles present in the plan head."""
    gs = _goal_shingles(_norm_goal_words(goal))
    if not gs:
        return 0.0
    hs = _goal_shingles(head_words)
    return sum(1 for s in gs if s in hs) / len(gs)


def _plan_head_words(plan: str) -> list[str]:
    """Head region = start -> the ``## 2.``/``### 2.`` heading, else first 8000 chars."""
    m = re.search(r"^#{2,3}\s*2\.\s", plan, re.M)
    return _norm_goal_words(plan[: m.start()] if m else plan[:8000])


def _goal_history_for_plan(folder: Path, plan_mtime_utc: datetime) -> tuple[str | None, list[str]]:
    """(current_goal, superseded_goals) AS OF the plan version's post time.

    current = latest predating ``epm:goal-updated`` ``to:`` (fallback:
    body.md frontmatter ``goal:``); superseded = predating markers'
    ``from:`` values (structured fields only — ``task.py set-goal`` posts
    top-level ``from``/``to``/``by``; hand-posted note-only markers
    contribute nothing). Bounded STRICTLY by mtime (slack 0, ``ts <= mtime``
    inclusive) so goal-updates that postdate the plan never retro-flag it —
    a positive slack manufactures FPs from minutes-scale retro-stale gaps
    (779/477/489).

    Read discipline: records split on ``"\\n"`` — NEVER ``str.splitlines()``
    (the paired writer ``task_workflow._append_jsonl_line`` emits
    ``ensure_ascii=False``, so a raw U+2028/U+2029/NEL inside a goal/note
    string is ONE valid JSONL record that ``splitlines()`` would shred,
    crashing the strict ``json.loads`` or silently dropping the marker —
    the #950 class; mirrors ``task_workflow._iter_jsonl``). Fail-fast: a
    row whose ``kind`` IS ``epm:goal-updated`` but whose ``ts`` is missing
    or non-string raises ``ValueError`` — the canonical writer always emits
    ``ts``, so such a row is real corruption, and silently skipping it
    would shrink the predating history (flipping c23 to SKIP/PASS on a
    stale plan).
    """
    cutoff = plan_mtime_utc + timedelta(seconds=_C23_MTIME_SLACK_S)
    current: str | None = None
    superseded: list[str] = []
    ev = folder / "events.jsonl"
    if ev.exists():
        for line in ev.read_text(encoding="utf-8", errors="replace").split("\n"):
            if not line.strip():
                continue
            if '"epm:goal-updated"' not in line:
                continue  # cheap pre-filter; goal-updated lines parse strictly below
            e = json.loads(line)
            if e.get("kind") != "epm:goal-updated":
                continue
            if not isinstance(e.get("ts"), str):
                raise ValueError(
                    f"malformed epm:goal-updated row in {ev}: missing/non-string 'ts' "
                    f"(the canonical writer always emits ts — this is corruption, "
                    f"not a benign note-only marker): {line!r}"
                )
            ets = datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
            if ets.tzinfo is None:
                ets = ets.replace(tzinfo=UTC)
            if ets.astimezone(UTC) > cutoff:
                continue
            if isinstance(e.get("from"), str):
                superseded.append(e["from"])
            if isinstance(e.get("to"), str):
                current = e["to"]
    if current is None:
        body = folder / "body.md"
        if body.exists():
            fm, _ = split_frontmatter(body.read_text())
            g = fm.get("goal")
            current = str(g) if g else None
    if current is not None:
        cur_norm = " ".join(_norm_goal_words(current))
        superseded = [s for s in superseded if " ".join(_norm_goal_words(s)) != cur_norm]
    return current, superseded


def check_goal_currency(
    plan: str, *, current_goal: str | None, superseded: list[str]
) -> CheckResult:
    """WARN when the plan head quotes a superseded Goal while the current
    Goal is absent (the #922 stale-quote signature); PASS/SKIP otherwise."""
    cid, name = "c23_goal_currency", "plan head not drafted against a superseded Goal"
    if current_goal is None or len(_norm_goal_words(current_goal)) < _C23_MIN_GOAL_WORDS:
        return _skip(cid, name, "no goal frontmatter / goal too short for shingle matching")
    sup = [s for s in superseded if len(_norm_goal_words(s)) >= _C23_MIN_GOAL_WORDS]
    if not sup:
        return _skip(cid, name, "no superseded Goal predates this plan version")
    head = _plan_head_words(plan)
    cov_cur = _shingle_coverage(current_goal, head)
    cov_stale, stale = max(((_shingle_coverage(s, head), s) for s in sup), key=lambda t: t[0])
    if cov_stale >= _C23_STALE_COV and cov_cur < _C23_CURRENT_COV:
        return _warn(
            cid,
            name,
            f"plan head matches a SUPERSEDED Goal (shingle coverage {cov_stale:.2f}: "
            f"{stale[:100]!r}) while the CURRENT Goal is absent (coverage {cov_cur:.2f}) "
            "— redraft §0.0/§0/§1 against the current `goal:` frontmatter (#922 "
            "plan-vs-goal incident). The orchestrator treats this WARN as a mechanical "
            "redraft bounce (adversarial-planner SKILL.md § Goal-currency gate).",
        )
    return _pass(cid, name, f"coverage: current {cov_cur:.2f}, max superseded {cov_stale:.2f}")


# ─── Check 24 — resume-skip provenance validation (conditional) ─────────────

# Trigger: a per-unit persist + resume-skip pattern. Compound forms ONLY —
# bare "resume" fires on pod lifecycle ("pod.py resume", "resume the poll
# loop") and bare "checkpoint"/"persist" on model checkpoints / the upload
# policy (calibration 2026-07-05: 63 experiment|analysis plan-version hits
# over the 1,367-file v*.md corpus, every spot-checked hit a genuine
# resume-skip loop; zero pod-resume / upload-policy false hits; 28
# non-exp/analysis triggered versions all land on the kind gate).
_C24_TRIGGER_RE = re.compile(
    r"(?i)\b(?:resume[- ]skip|resume[- ]predicate"
    r"|skip[- ]if[- ]exists?"
    r"|skips?\s+(?:already[- ])?(?:completed|done|existing)"
    r"|checkpoint[- ](?:skip|resume)"
    r"|per[- ](?:fold|cell|unit|seed|row|shard)[- ]persist\w*"
    r"|idempotent re-?runs?"
    r"|load[- ]partial[- ]and[- ]skip)"
)

# Satisfier: a recognizable input-fingerprint / provenance token near the
# resume mention. Compound-form discipline on BOTH flanks (plan #1043 v3):
# bare "provenance" and bare "manifest" are deliberately EXCLUDED —
# "Completion provenance:" is a REQUIRED §4-design bullet in every
# experiment plan (on-policy-completions enforcement), so it lands within
# ±15 lines of resume prose and false-satisfied 52% of the v2-calibration
# PASSes (#811 v1, #622 v1-v3, #931 v6 measured; 2026-07-05 reconciler).
# Bare "regime" is likewise EXCLUDED — persona-vectors "read-out regime"
# prose sits inside resume windows (#779 v5 measured) and would
# self-satisfy. The final alternate (assert/validate/verify … existing/
# persisted/… … match) catches contracts phrased without a fingerprint
# noun (#560 v3: "assert the existing file's `sampling.temperature` /
# `sampling_seed` match the requested flags"); it requires a resume-object
# token between verb and "match" so an equivalence-gate assert ("assert
# the vmapped MLP path matches a seeded serial reference", #922 v1-v3
# measured) does NOT satisfy, and its spans are [^\n] (not
# sentence-bounded) because periods inside code tokens like
# `sampling.temperature` break [^.]-spans.
_C24_FINGERPRINT_RE = re.compile(
    r"(?i)\b(?:fingerprints?"
    r"|provenance[- ](?:manifest\w*|contract\w*|validation\w*|check\w*)"
    r"|manifest[- ](?:match\w*|validation\w*|mismatch\w*|check\w*)"
    r"|sha[- ]?256|git[_ -]?sha|code[- ]sha|commit[- ](?:sha|hash)"
    r"|(?:split|content|input|data)[- ]hash(?:es)?"
    r"|env[- ](?:fingerprint|knobs?)"
    r"|regime[- ]key(?:ed|s)?"
    r"|never skip\w* on (?:[\w-]+[- ])?existence"
    r"|(?:assert\w*|validat\w+|verif\w+)[^\n]{0,80}?"
    r"\b(?:existing|persisted|resumed?|cached|stored|prior)\b[^\n]{0,80}?\bmatch\w*)"
)

# Evidence window: ± this many RAW lines around each trigger (the provenance
# contract legitimately lives in an adjacent sentence/table row — #952 v12
# names it in the same bullet; #813 v3 one section over).
_C24_WINDOW_LINES = 15


def check_resume_provenance(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a plan naming a per-unit persist + resume-skip
    pattern must, within ±15 raw lines of SOME resume mention, name an
    input-fingerprint / provenance validation for the resumed outputs (split
    hashes, code SHA, env knobs, a regime-keyed resume predicate, an explicit
    never-skip-on-bare-existence commitment, or an assert-that-the-existing-
    file-matches contract) — never output existence alone. NEVER FAILs in
    v1 — the trigger is a vocabulary heuristic and semantic adequacy of the
    named validation stays with the critics (task #1043 constraint). Known
    accepted gap under WARN-only: a plan that QUOTES this check's WARN
    remedy text near a trigger self-satisfies (the remedy names "split
    hashes, code SHA, env knobs"); the anti-paste guard covers only the N/A
    phrase. Any future WARN→FAIL promotion MUST close that gap first (plan
    #1043 §10 must-ask hook). Incident #952 v9: per-fold persist +
    resume-skip with a bare skips-completed-folds predicate would have let
    stale-fold outputs (or a stale calibration-gate PASS) silently vouch for
    post-code-fix verdict folds; caught only by the critic ensemble (v10
    added the gate-5 provenance-manifest contract). ANY-window semantics per
    the c12 precedent: the contract is typically declared once near one
    mention."""
    cid, name = "c24_resume_provenance", "resume-skip provenance validation"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: resume-provenance is an experiment|analysis plan shape"
        )
    windows = _trigger_windows(plan, _C24_TRIGGER_RE, _C24_WINDOW_LINES)
    if not windows:
        return _skip(cid, name, "no per-unit persist + resume-skip pattern named")
    if _standalone_na_declared(plan, r"no resume\s*[/-]?\s*persist pattern"):
        return _pass(cid, name, "explicit N/A declared (no resume/persist pattern)")
    for window in windows:
        if _C24_FINGERPRINT_RE.search(window):
            return _pass(
                cid,
                name,
                "a resume window names a provenance/fingerprint validation — whether the "
                "named fields (input hashes, code SHA, env) are SUFFICIENT stays critic-owned",
            )
    return _warn(
        cid,
        name,
        "plan names a per-unit persist + resume-skip pattern but no input-fingerprint / "
        "provenance validation near any resume mention (split hashes, code SHA, env knobs, "
        "a regime-keyed resume predicate) — a resume that trusts bare output existence lets "
        "stale units silently vouch after a crash + code-fix round (#952 v9; #722 r3); "
        "name the validation per the #952 gate-5 manifest shape, or declare "
        "'N/A — no resume/persist pattern' on its own line, unwrapped "
        "(no backticks/quotes), if the mention is incidental",
    )


# ─── Check 25 — HTML entities in fenced command blocks (all kinds) ──────────
# The harness HTML-escapes the <result> field of background-Agent
# <task-notification> messages (&& -> the amp-entity form, < -> lt, > -> gt);
# an orchestrator that composes the plan handoff from that text ships a
# poisoned workload command (#952 v9, 2026-07-04: the dispatcher command's
# shell AND operators arrived entity-escaped and needed a hand-fix before
# dispatch would run). This check is the persist-time backstop for the
# capture-time de-escape rule in adversarial-planner SKILL.md.

# Fence pairing: backtick fences only, opener info string captured, closer on
# its own line — the same relaxed-pairing limitation class as the other regex
# checks (corpus-calibrated; exotic 4-backtick nesting is out of scope).
_C25_FENCE_RE = re.compile(r"(?ms)^[ \t]*```([^\n]*)\n(.*?)^[ \t]*```[ \t]*$")

# Arm (a): shell-tagged fences (exemptable by the standalone escape phrase).
_C25_CMD_FENCE_INFO_RE = re.compile(r"(?i)^\s*(?:bash|sh|shell|zsh|console)\b")

# Arm (b): ANY fence (tagged or untagged) whose body carries the
# highest-stakes command markers — never exemptable.
_C25_CMD_MARKER_RE = re.compile(r"--workload-cmd|dispatch_issue\.py")

# The six entity forms (amp/lt/gt/quot + the numeric/hex apostrophes),
# case-insensitive, leading-zero-tolerant on the numeric forms.
_C25_HTML_ENTITY_RE = re.compile(r"(?i)&(?:amp|lt|gt|quot|#0*39|#x0*27);")


def _c25_detail(hits: list[str], *, exemptable: bool) -> str:
    """Render the c25 FAIL detail: entity list + the #952 v9 incident + the
    capture-side remediation; the escape-phrase pointer appears ONLY on the
    ``exemptable=True`` (arm-(a)) branch — an arm-(b) ``--workload-cmd`` /
    ``dispatch_issue.py`` fence is never exemptable (methodology reconciler,
    #1062 round 1)."""
    base = (
        f"fenced command block(s) carry HTML entity form(s) {', '.join(hits)} — the "
        "harness HTML-escapes background-Agent <task-notification> results "
        "(#952 v9, 2026-07-04: the dispatcher command's shell AND operators "
        "arrived entity-escaped); re-extract from the raw output-file, or apply "
        "ONE html.unescape() round to notification-BODY-sourced text, before "
        "persisting"
    )
    if exemptable:
        return base + (
            "; if the fenced entities are deliberately discussed CONTENT (not a "
            "command to dispatch), declare 'N/A — entities are content, not "
            "commands' on its own line, unwrapped (no backticks/quotes) "
            "(valid only when exactly ONE shell-tagged fence carries entity "
            "forms; with several, re-tag content fences to a non-shell info "
            "string or combine them into one fence)"
        )
    return base + (
        " — a --workload-cmd / dispatch_issue.py fence is never exemptable: fix the command text"
    )


def _c25_multi_fence_detail(n_fences: int, hits: list[str]) -> str:
    """Render the c25 FAIL detail for the count-scoped exemption (#1276): the
    standalone escape phrase is present but MORE THAN ONE arm-(a) fence
    carries entity hits — a doc-wide declaration must not let a poisoned
    command fence ride a legitimate content fence's exemption (the arm-(a)
    sibling of the #1062 arm-(b) never-exemptable rule)."""
    return (
        f"{n_fences} distinct shell-tagged fences carry HTML entity form(s) "
        f"{', '.join(hits)}, but the standalone content exemption is scoped to "
        "EXACTLY ONE entity-bearing fence — a doc-wide declaration must not "
        "mask a separately poisoned command fence (#1276; arm-(a) sibling of "
        "the #1062 arm-(b) rule); re-tag genuinely content-bearing fences to a "
        "non-shell info string (e.g. a text-tagged fence, which arm (a) never "
        "scans), or combine the content commands into one fence, or fix the "
        "poisoned command text (re-extract from the raw output-file / one "
        "html.unescape() round)"
    )


def check_html_entities_in_commands(plan: str, kind: str) -> CheckResult:
    """FAIL, ALL kinds, conditional: fenced command blocks must not carry HTML
    entities (#952 v9).

    Two arms: (a) shell-tagged fences (bash/sh/shell/zsh/console) with no
    command marker; (b) ANY fence — tagged or untagged — whose body carries
    ``--workload-cmd`` or ``dispatch_issue.py``. Scan-first; the standalone
    escape phrase (``N/A — entities are content, not commands``, detected via
    the house ``_standalone_na_declared`` line discipline — never a doc-global
    substring) exempts arm-(a) hits ONLY, and only when EXACTLY ONE arm-(a)
    fence carries entity hits — with two or more entity-bearing shell fences
    the declaration cannot bind to a specific fence, so the check FAILs
    naming the fence count (#1276; per-fence grain: distinct fences, not
    distinct entity forms). An arm-(b) entity hit FAILs
    UNCONDITIONALLY — a document-wide phrase must never mask a separately
    poisoned workload command (methodology reconciler, #1062 round 1: one
    legitimate entity-discussing fence + one poisoned dispatcher fence must
    still FAIL). SKIP when the plan has no command fences. All kinds —
    infra/batch plans carry verification commands too (this incident class is
    kind-agnostic). The check ASSERTS; it never rewrites plan text.
    """
    cid, name = "c25_html_entities_in_commands", "no HTML entities in fenced command blocks"
    del kind  # all kinds — infra/batch plans carry verification commands too
    arm_a: list[str] = []
    arm_b: list[str] = []
    for info, body in _C25_FENCE_RE.findall(plan):
        if _C25_CMD_MARKER_RE.search(body):
            arm_b.append(body)  # command-marked: never exemptable
        elif _C25_CMD_FENCE_INFO_RE.match(info):
            arm_a.append(body)  # shell-tagged: exemptable by the phrase
    if not arm_a and not arm_b:
        return _skip(cid, name, "no fenced command blocks detected")
    hits_b = sorted({m.group(0) for b in arm_b for m in _C25_HTML_ENTITY_RE.finditer(b)})
    if hits_b:
        return _fail(cid, name, _c25_detail(hits_b, exemptable=False))
    # Per-fence grain (#1276): the exemption scope counts DISTINCT arm-(a)
    # fences carrying entity hits — never the union of entity forms, which
    # loses fence identity and lets a poisoned fence ride a legitimate
    # fence's declaration (the arm-(a) sibling of the #1062 arm-(b) rule).
    per_fence_hits = [sorted({m.group(0) for m in _C25_HTML_ENTITY_RE.finditer(b)}) for b in arm_a]
    per_fence_hits = [h for h in per_fence_hits if h]
    hits_a = sorted({form for h in per_fence_hits for form in h})
    if hits_a and _standalone_na_declared(plan, r"entities are content, not commands"):
        if len(per_fence_hits) == 1:
            return _pass(
                cid,
                name,
                "arm-(a) entity content exempted by explicit standalone N/A "
                "(single entity-bearing fence)",
            )
        return _fail(cid, name, _c25_multi_fence_detail(len(per_fence_hits), hits_a))
    if hits_a:
        return _fail(cid, name, _c25_detail(hits_a, exemptable=True))
    return _pass(cid, name, f"{len(arm_a) + len(arm_b)} command fence(s), no entity forms")


# ─── Check 26 — GPU basis vs routed machine (WARN-only, conditional) ────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Cost wall-time against
# the machine the router will ACTUALLY provision" (#599/#833/#1073 class).
# STATIC MIRROR of backends/gcp.py::INTENT_TO_MACHINE at FAMILY grain,
# drift-guarded by tests/test_verify_plan.py::
# test_c26_intent_gpu_mirror_matches_backend — verify_plan_text() stays
# hermetic (no project imports at module level; the only project import in
# this file is the --issue-mode-local task_workflow resolver).
_C26_INTENT_GPU: dict[str, str] = {
    "lora-7b": "A100",
    "lora": "A100",
    "capture-7b": "A100",
    "ft-7b": "A100",
    "eval": "L4",
    "debug": "L4",
    "lora-7b-h100": "H100",
    "eval-h100": "H100",
    "cpu-bigmem": "CPU",
    "cpu-small": "CPU",
    "cpu-mid": "CPU",
    "sweep-8g-a100": "A100",
    "sweep-8g-h100": "H100",
}


def _c26_family(token: str) -> str:
    """GPU family normalization: strip a trailing ``-<digits>`` HBM-size
    suffix (``A100-80`` == ``A100-40`` == ``A100``; ``H100-80`` == ``H100``;
    ``L4``/``CPU`` unchanged). A100-40-vs-A100-80 differences are
    deliberately below the heuristic's grain."""
    return re.sub(r"-\d+$", "", token)


# GPU family tokens ALLOWED in a basis cell trigger. L4/L40S deliberately
# EXCLUDED from the trigger set: #833-style leg labels ("L1/L2 re-extraction,
# L3/L4 extraction") collide with the L4 GPU token; nobody measures bases on
# an L4, while the ROUTED side still knows L4 via the mirror. Included in the
# ESCAPE scan (permissive direction only).
_C26_BASIS_GPU_RE = re.compile(r"\b(H100|H200|A100(?:-[48]0)?|B200)\b")
_C26_ROW_GPU_ANY_RE = re.compile(r"\b(H100|H200|A100(?:-[48]0)?|B200|L40S|L4)\b")

# Scaling vocabulary (row-scoped escape). A bare multiplication sign is NOT
# an escape — it appears in nearly every row's multiplier arithmetic
# ("5,000 x ~300 tok", "draws x cells"); #1073 v3's offending row contains
# one and was still the incident (plan #1075 calibration finding).
_C26_SCALING_RE = re.compile(
    r"(?i)\bscal(?:ed|ing|e factor)\b|per-?step rate|step-?time|rate-?convert"
)

# Intent resolution: --intent <tok> in prose or fences (c5 precedent: RAW
# scan); additionally accepted: the "intent `lora-7b`" prose form
# (#1073 v3 "Target pod preference" shape — capitalized "Intent" in the
# wild, hence (?i)).
_C26_INTENT_RE = re.compile(
    r"(?i)--intent[=\s]+`?([A-Za-z0-9][A-Za-z0-9-]*)|\bintent\s+`([A-Za-z0-9][A-Za-z0-9-]*)`"
)

# Explicit RunPod pin → the RunPod H100/H200 intent table governs; SKIP.
# Scanned RAW (fences included), matching the raw intent scan — a fenced
# `--backend runpod` dispatch line is a real pin; permissive direction only.
_C26_RUNPOD_PIN_RE = re.compile(r"(?i)\bbackend:\s*`?runpod\b|--backend[=\s]+`?runpod\b")


def _c26_intents(plan: str) -> set[str]:
    """Intent tokens resolved from RAW plan text (fences included — a fenced
    dispatch line is the real launch command, the c5 raw-scan precedent).
    Union of the ``--intent <tok>`` flag form (group 1) and the
    ``intent `tok` `` prose form (group 2)."""
    out: set[str] = set()
    for m in _C26_INTENT_RE.finditer(plan):
        tok = m.group(1) or m.group(2)
        if tok:
            out.add(tok)
    return out


# §9 gpu-hours HEADER binder (shared parser below; calibration + the c47
# consumer live in the Check 47 section). TOKEN-SHAPED, not a substring:
# the corpus carries `GPU spec` (28 occurrences, holding `1× H100`),  # noqa: RUF003
# `GPU width` (17) and bare `GPU` (75) — none hour-bearing; a substring
# bind parses booked=1 from `1× H100` and manufactures a ~90× ratio.  # noqa: RUF003
# The leading guard MUST be `(?<![a-z0-9])`, NOT `\b`: `_` is a word
# character, so `\bgpu` has no boundary inside `planned_gpu_h` (989 of
# 1,056 gpu-h-bearing corpus files) and the column silently never
# resolves — measured 0 Arm-A hits over 5,166 files under the `\b` form
# (plan #2177 §12 Must-Fix 1/3).
_C48_GPU_H_HEADER_RE = re.compile(r"(?i)(?<![a-z0-9])gpu[-_ ]*h(?:ours?|rs?)?\b")


class _ComputeRow(NamedTuple):
    """One admitted body row of a §9 compute-projection table (the shared
    c26/c32/c47 parser ``_compute_table_rows``). ``gpu_h`` / ``parallelism``
    are OPTIONAL columns — ``""`` when the header has no such column;
    ``gpu_is_wall_col`` is True when the gpu-h header match landed on the
    SAME cell as the wall column (the combined ``Wall / GPU-h`` corpus
    shape, ~109 headers — c47 skips those rows for Arm A, #2177)."""

    component: str
    wall: str
    gpu_h: str
    gpu_is_wall_col: bool
    parallelism: str
    basis: str
    row_text: str


def _compute_table_rows(plan: str) -> list[_ComputeRow]:
    """``_ComputeRow`` for every body row of every non-fenced markdown table
    whose header carries a ``basis`` column (a cell that IS or BEGINS WITH
    the word ``basis``, casefolded, bold/backticks stripped — the corpus
    carries an annotated ``basis (measured)`` variant, #952 v12) AND a wall
    column (fuzzy: any header cell CONTAINING ``wall`` — matches
    ``planned_wall_h`` / ``planned wall h`` / ``wall_h`` drift). Header
    detection is fence-masked (a fenced example table is not the plan's
    table — the ``_trigger_windows`` precedent; this deliberately diverges
    from ``_source_column_cells``, which is section-scoped instead: c26
    scans the whole doc because §9 heading text drifts). A row with fewer
    cells than the basis column needs is skipped defensively (the bold
    ``**Base total**`` short-row shape — no IndexError); a short row that
    still reaches the basis column is treated normally with an empty wall
    cell.

    Row ADMISSION is UNCHANGED from the pre-#2177 ``_c26_compute_table_rows``
    (a table is admitted iff its header carries basis + wall columns; a body
    row iff it reaches the basis column). The ONLY additions are two
    OPTIONAL columns: ``gpu_h`` (header matches ``_C48_GPU_H_HEADER_RE``)
    and ``parallelism`` (fuzzy: a header cell containing ``parallel``),
    both defaulting to ``""`` — regression evidence is the untouched c26 +
    c32 test suites plus ``test_c26_c32_unchanged_by_parser_extraction``."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    rows: list[_ComputeRow] = []
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if mask[i] or not (
            header.startswith("|") and sep.startswith("|") and _TABLE_SEP_RE.fullmatch(sep)
        ):
            i += 1
            continue
        header_cells = [c.strip().strip("*`").strip().casefold() for c in _split_table_row(header)]
        basis_col = next((j for j, c in enumerate(header_cells) if re.match(r"basis\b", c)), None)
        wall_col = next((j for j, c in enumerate(header_cells) if "wall" in c), None)
        gpu_col = next(
            (j for j, c in enumerate(header_cells) if _C48_GPU_H_HEADER_RE.search(c)), None
        )
        par_col = next((j for j, c in enumerate(header_cells) if "parallel" in c), None)
        k = i + 2
        while k < len(lines) and lines[k].strip().startswith("|"):
            if basis_col is not None and wall_col is not None:
                row = _split_table_row(lines[k])
                if basis_col < len(row):
                    rows.append(
                        _ComputeRow(
                            component=row[0] if row else "",
                            wall=row[wall_col] if wall_col < len(row) else "",
                            gpu_h=(
                                row[gpu_col] if gpu_col is not None and gpu_col < len(row) else ""
                            ),
                            gpu_is_wall_col=gpu_col is not None and gpu_col == wall_col,
                            parallelism=(
                                row[par_col] if par_col is not None and par_col < len(row) else ""
                            ),
                            basis=row[basis_col],
                            row_text=lines[k],
                        )
                    )
            k += 1
        i = k
    return rows


def _c26_compute_table_rows(plan: str) -> list[tuple[str, str, str, str]]:
    """Back-compat projection of ``_compute_table_rows`` — the c26 + c32
    call sites (and ``scripts/issue1395_corpus_audit.py``) consume the
    original ``(component, basis, wall, row_text)`` 4-tuple unchanged
    (#2177 extraction; byte-identical admission predicate, fence mask,
    short-row defence and cell-stripping)."""
    return [(r.component, r.basis, r.wall, r.row_text) for r in _compute_table_rows(plan)]


def _c26_offender_detail(offenders: list[tuple[str, str]], routed: set[str]) -> str:
    """Bounded WARN detail (c13 ``_offender_detail`` precedent): at most 3
    offending rows (component + the offending GPU token), the resolved
    routed families, the #599 incident anchor, and BOTH remedies (a stated
    per-step scaling rate in the row, or the standalone N/A phrase)."""
    shown = "; ".join(f"row {comp[:60]!r} basis names {tok}" for comp, tok in offenders[:3])
    if len(offenders) > 3:
        shown += "; ..."
    return (
        f"{shown} but resolved intent(s) route {sorted(routed)} under auto (GCP "
        "INTENT_TO_MACHINE) with no stated cross-GPU scaling in the row — a basis "
        "measured on a different GPU must be scaled with a stated per-step rate "
        "(plan-compute-sizing.md; #599: an H100-premised ~6.4h estimate ran ~34h on "
        "the A100 auto-lane), or declare 'N/A — basis measured on the routed machine' "
        "on its own line, unwrapped (no backticks/quotes)"
    )


def check_gpu_basis_routed_machine(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a §9 compute-projection-table basis cell naming
    a GPU family (H100/H200/A100/B200) that differs from EVERY family the
    plan's resolved --intent token(s) route under auto (static GCP
    INTENT_TO_MACHINE mirror, _C26_INTENT_GPU), with no row-level escape.
    Mechanizes plan-compute-sizing.md § "Cost wall-time against the machine
    the router will ACTUALLY provision" (#599 ~6.4h -> ~34h; #1073 v3 -> v4).
    Row escapes: (a) the routed family named in a CONVERSION-BEARING cell —
    the wall or basis cell ONLY (a stated conversion names both machines
    there, #1073 v4 wall cell "0.25 (H100) / 0.5-0.6 (A100, x2-2.5)");
    a parallelism/component-cell mention describes the PROVISIONED machine,
    not a conversion, and does NOT escape (plan #1075 Must-Fix M1 — #810 v18
    / #923 v9 rows put "1x A100-80" in parallelism/component cells);
    (b) scaling vocabulary (scaled/per-step rate/...) anywhere in the row.
    NEVER FAILs in v1 — both sides are heuristic (intent resolution from
    text; token matching), and whether a stated scaling factor is CORRECT
    stays critic-owned (c24 precedent). Known accepted gaps: a basis citing
    a prior issue's realized wall WITHOUT naming its GPU (#599's shape)
    is invisible; a "recommended pin: backend: runpod" prose mention
    escapes as if pinned (#779 v6); a conversion stated as a BARE
    multiplier with no vocabulary word ("on H100, x2.5" — #628 v2, the one
    adjudicated calibration FP) still WARNs, because bare-multiplier
    arithmetic saturates compliant AND offending rows alike (#1073 v3) —
    the remedy is one vocabulary word in the row; A100-40 vs A100-80 is
    below the family grain; a routed-family mention in the wall/basis
    cell escapes
    without a true conversion (conversion ADEQUACY stays critic-owned);
    a standalone N/A declaration is document-wide (c24 /
    ``_standalone_na_declared`` family semantics), so it also clears any
    sibling offender row — the deliberate-override purpose of the phrase."""
    cid, name = "c26_gpu_basis_routed_machine", "GPU basis vs routed machine"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: compute-projection tables are an experiment|analysis plan shape",
        )
    rows = _c26_compute_table_rows(plan)
    if not rows:
        return _skip(cid, name, "no compute-projection table with a `basis` column detected")
    if _standalone_na_declared(plan, r"basis measured on the routed machine"):
        return _pass(cid, name, "explicit N/A declared (basis measured on the routed machine)")
    if _C26_RUNPOD_PIN_RE.search(plan):
        # RAW scan (fences included): a fenced `--backend runpod` dispatch
        # line is a real pin; permissive direction (can only add SKIPs).
        return _skip(
            cid,
            name,
            "explicit backend: runpod pin — the RunPod intent table governs the basis machine",
        )
    routed = {_C26_INTENT_GPU[i] for i in _c26_intents(plan) if i in _C26_INTENT_GPU}
    if not routed:
        return _skip(
            cid,
            name,
            "no resolvable --intent token — routed machine unknown (auto-lane GPU cannot "
            "be inferred)",
        )
    offenders: list[tuple[str, str]] = []
    for component, basis, wall, row_text in rows:
        hit = _C26_BASIS_GPU_RE.search(basis)
        if not hit or _c26_family(hit.group(1)) in routed:
            continue
        # Escape (a): routed family named in a CONVERSION-BEARING cell only
        # (wall + basis) — NOT parallelism/component (Must-Fix M1).
        conv_cells = f"{basis} {wall}"
        conv_families = {_c26_family(m.group(1)) for m in _C26_ROW_GPU_ANY_RE.finditer(conv_cells)}
        if conv_families & routed or _C26_SCALING_RE.search(row_text):
            continue
        offenders.append((component, hit.group(1)))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(rows)} table row(s); no unscaled cross-GPU basis vs routed {sorted(routed)}",
        )
    return _warn(cid, name, _c26_offender_detail(offenders, routed))


# ─── Check 27 — 7B activation capture vs eval/debug (L4) intent ─────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Activation-capture HBM
# sizing" (MUST-level): a 7B hidden-state capture phase needs >=40 GB HBM
# (capture-7b / lora-7b, 1x A100-80); the GCP eval/debug default is a
# 16-GB-class L4 (g2-standard-4) and OOMs mid-run (#666, #744). Founding
# false negative: #825 plan v17 (--intent eval for a 7B all-layer capture)
# PASSed 0 FAIL/0 WARN. Reuses c26's intent machinery — one parser, one
# drift-guarded mirror (test_c26_intent_gpu_mirror_matches_backend).

# Offending + absolving intent sets, DERIVED from the c26 mirror.
# BIG set derived by EXCLUSION (critique r1, Claude methodology concern 1):
# a future mirror family (H200/B200 intent) lands in the absolution set
# automatically instead of silently outside it (which would false-FAIL a
# plan booking that big intent alongside a side eval phase). Test
# test_c27_sets_derive_from_mirror's partition assert pins
# L4 | BIG | CPU == the whole mirror.
_C27_L4_INTENTS: frozenset[str] = frozenset(i for i, fam in _C26_INTENT_GPU.items() if fam == "L4")
_C27_BIG_HBM_INTENTS: frozenset[str] = frozenset(
    i for i, fam in _C26_INTENT_GPU.items() if fam not in ("L4", "CPU")
)

# Capture-phase vocabulary (RAW scan — capture launch commands and store
# rows legitimately live in fences/tables; the _c26_intents raw-scan
# precedent). Anchored compounds only: bare "extraction"/"capture" false-
# fire on prose ("extraction set", "capture the behavior"). Calibrated
# 2026-07-07 over 1,511 persisted plans: 5/5 known offender tasks flagged
# (#667/#744/#761/#810/#825), zero false positives.
_C27_CAPTURE_RE = re.compile(
    r"(?i)hidden[-_ ]states?\b"
    r"|activations?[-_ ]?(?:store|captur\w*|extract\w*|accumulat\w*|dump\w*)"
    r"|\bextract_store\b"
    r"|residual[-_ ]stream"
    r"|\bcaptur\w+\s+(?:\w+\s+)?activations?\b"
)

# >=7B model-size signal (the HBM rule is 7B-scoped; sub-7B captures fit L4).
# THRESHOLD semantics, not a whitelist (critique r1, all three Codex lenses):
# integer part >= 7 — single digit 7-9, or any 2+ digit number — with an
# optional decimal tail. The negative lookbehind (?<![\d.]) blocks the
# decimal-tail false positive ("1.7B"/"2.5B"/"6.9B" never match: the digit
# before the dot fails both integer alternates, and the digit after the dot
# is lookbehind-blocked). "17B" DOES match under threshold semantics
# (17 >= 7 — a deliberate deviation from the r1 Codex test sketch, which
# carried over the old whitelist's behavior). Token-count strings ("15B
# tokens") can match — acceptable: the conjunction still needs capture
# vocabulary + an un-skipped eval/debug booking, and the corpus re-scan
# gate (plan #1093 §13) binds on any regex change.
_C27_MODEL_GE7B_RE = re.compile(r"(?i)(?<![\d.])\b(?:[7-9]|[1-9][0-9]+)(?:\.[0-9]+)?B\b")

# scripts/pod.py IS the RunPod lifecycle CLI, where eval provisions
# 1x H100 80GB (CLAUDE.md intent table) — no HBM gap. Document-wide,
# permissive direction only (adds SKIPs); the _C26_RUNPOD_PIN_RE sibling
# for the pre-router plan corpus (#358/#375/#522 era).
_C27_PODPY_PROVISION_RE = re.compile(r"(?i)\bpod\.py\s+provision\b")

# Window-level big-GPU skip: an eval/debug token whose immediate context
# names H100/H200 is a RunPod-mapping or explicit-override claim, not a
# GCP L4 booking. A100 deliberately NOT in the skip set: GCP eval/debug
# NEVER provisions A100 — an A100 claim next to an eval booking is exactly
# the #744 misbelief this check exists to catch.
_C27_WINDOW_BIGGPU_RE = re.compile(r"\b(H100|H200)\b")


def _c27_gcp_l4_intent_windows(plan: str) -> list[tuple[str, str]]:
    r"""``(token, window_snippet)`` for every eval/debug intent occurrence
    plausibly booking the GCP/auto lane. The window is the PREVIOUS line
    plus the line containing the match end — the previous line covers the
    wrapped ``pod.py provision --issue N --intent\neval`` shape (#522 v1,
    where ``--intent[=\s]+`` legitimately spans the newline). A window
    carrying ``pod.py`` or an H100/H200 token is skipped (RunPod / explicit
    big-GPU context)."""
    out: list[tuple[str, str]] = []
    for m in _C26_INTENT_RE.finditer(plan):
        tok = m.group(1) or m.group(2)
        if tok not in _C27_L4_INTENTS:
            continue
        line_start = plan.rfind("\n", 0, m.start())
        prev_start = plan.rfind("\n", 0, line_start) if line_start != -1 else -1
        win_end = plan.find("\n", m.end())
        window = plan[prev_start + 1 : len(plan) if win_end == -1 else win_end]
        if "pod.py" in window or _C27_WINDOW_BIGGPU_RE.search(window):
            continue
        out.append((tok, " ".join(window.split())[:90]))
    return out


def check_capture_intent_hbm(plan: str, kind: str) -> CheckResult:
    """FAIL (experiment) / WARN (analysis), conditional: activation-capture
    vocabulary + a >=7B model signal while an eval/debug (L4) intent is
    booked on the GCP/auto lane. Skip ladder (permissive direction only):
    kind gate -> vocab trigger -> standalone N/A escape -> RunPod pin
    (backend/--backend runpod OR pod.py provision, doc-wide: RunPod eval =
    1x H100 80GB) -> no resolvable intent -> no un-windowed eval/debug
    occurrence -> big-HBM-intent absolution -> no >=7B signal.
    Known accepted gaps (all deliberate, critic-owned semantics):
    (a) a plan booking a big-HBM intent for training while the CAPTURE
    phase books eval escapes via the absolution — phase-to-intent routing
    stays critic-owned; (b) an eval occurrence whose window names H100/H200
    (e.g. a basis-measured-on-H100 clause on the same line) escapes as if
    pinned — c26 covers the basis side; (c) a doc-wide pod.py-provision pin
    skips mixed-lane plans; (d) the >=7B signal matches "7b" inside intent
    tokens (lora-7b) — a weak filter by design, the N/A phrase is the real
    small-model out; (e) vocabulary from a REUSED store consumed by a CPU
    phase still triggers — cleared by the no-L4-intent PASS, the
    absolution, or the N/A phrase."""
    cid, name = "c27_capture_intent_hbm", "7B capture vs eval/debug intent"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt: capture phases are an experiment|analysis plan shape")
    cap_hit = _C27_CAPTURE_RE.search(plan)
    if not cap_hit:
        return _skip(cid, name, "no activation-capture vocabulary detected")
    if _standalone_na_declared(plan, r"no 7B activation capture"):
        return _pass(cid, name, "explicit N/A declared (no 7B activation capture)")
    if _C26_RUNPOD_PIN_RE.search(plan) or _C27_PODPY_PROVISION_RE.search(plan):
        return _skip(
            cid, name, "explicit RunPod pin/provision — RunPod eval = 1x H100 80GB, no HBM gap"
        )
    if not _c26_intents(plan):
        return _skip(cid, name, "no resolvable --intent token — routed machine unknown")
    windows = _c27_gcp_l4_intent_windows(plan)
    if not windows:
        return _pass(
            cid,
            name,
            "capture vocabulary present but no eval/debug intent booked on the GCP/auto lane",
        )
    big = sorted(_c26_intents(plan) & _C27_BIG_HBM_INTENTS)
    if big:
        return _pass(
            cid,
            name,
            f">=40 GB-HBM intent also booked ({big}) — capture phase presumed routed there "
            "(phase-to-intent routing stays critic-owned)",
        )
    if not _C27_MODEL_GE7B_RE.search(plan):
        return _skip(cid, name, "no >=7B model signal — the HBM sizing rule is 7B-scoped")
    tok, snippet = windows[0]
    verdict = _fail if kind == "experiment" else _warn
    return verdict(
        cid,
        name,
        f"capture vocabulary ({cap_hit.group(0)!r}) with a >=7B model while the plan books the "
        f"{tok} (L4, g2-standard-4, 16-GB-class HBM) intent on the GCP/auto lane "
        f"(context: {snippet!r}) — >=7B hidden-state capture needs >=40 GB HBM "
        "(#666/#744 OOM class; #825 v17 false negative): for a 7B-class model book capture-7b "
        "(forward-pass-only) or lora-7b (phase also trains); a LARGER model needs a "
        "correspondingly larger-HBM lane/backend (a multi-GPU intent or an explicit "
        "large-GPU RunPod pin), never eval/debug — per plan-compute-sizing § "
        "Activation-capture HBM sizing, or declare 'N/A — no 7B activation capture' "
        "on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 28 — decision-band precedent coherence (WARN-only) ───────────────
# Mechanizes the #825 v17 incident class: a registered fractional decision
# band ("cmp T x" inside a success/kill/decision section), applied to the
# plan's OWN quoted precedent ratio(s), must land in the branch the plan's
# narrative asserts. Prose siblings: planner-section-reference.md §7
# (precedent self-check bullet) + critic-lens-reference.md Statistics item
# 3 trigger (c) — the FAIL-grade semantic verdict stays critic-side.

# Band line: a non-fenced, bold-labeled list item inside a decision-keyword
# section (_C13_GATE_SECTION_RE reused) carrying a multiplicative threshold
# "cmp T x" with fractional T (0 < T <= 1) — the #931 committed-threshold
# idiom ("< 0.5 × 0.588", "≥ 0.5× its ceiling"). Integer / super-unity T  # noqa: RUF003
# ("≥ 2× wall-time" kill fences) deliberately excluded: fraction-of-ceiling  # noqa: RUF003
# bands are the target class; wall-time multipliers are a different quantity.
_C28_BAND_RE = re.compile(
    r"(?P<cmp><=|>=|[<>≤≥])\s*(?P<thr>0?\.\d+|1\.0|1)\s*[×x](?![a-zA-Z])"  # noqa: RUF001
)
_C28_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*]|\d+\.)\s")  # c14 sibling
_C28_BOLD_LABEL_RE = re.compile(r"\*\*([^*\n]{1,60})\*\*")  # c14 sibling

# Precedent-ratio assertion: explicit "ratio ≈ r" / "ratio ≈ r1–r2" token.  # noqa: RUF003
# Decimal point REQUIRED (excludes the "ratio ~1:1" mix idiom — the only 2
# non-incident same-line corpus hits); `%`-suffixed ratios NOT harvested —
# single (`0.48%`) AND range (`0.44–0.52%`) forms both harvest NOTHING  # noqa: RUF003
# (percent-vs-fraction confusion is a named FP mode — accepted false
# negative). The \b after each number blocks a backtracked partial-digit
# match like `0.4` inside `0.48%`; the second lookahead rejects r1 when the
# engine SKIPS a %-suffixed optional range group — `(?!\s*%)` alone let
# `ratio ≈ 0.44–0.52%` partially harvest r1=0.44 (round-2 fix, concern  # noqa: RUF003
# c28-percent-range-partial-harvest).
_C28_RATIO_RE = re.compile(
    r"(?i)\bratios?\s*[≈=~]\s*(?P<r1>0?\.\d+)\b"
    r"(?:\s*[–—-]\s*(?P<r2>0?\.\d+)\b)?(?!\s*%)"  # noqa: RUF001 — en dash is real plan text
    r"(?!\s*[–—-]\s*0?\.\d+\s*%)"  # noqa: RUF001 — reject a %-suffixed skipped range
)
# Verb-anchored side vocabulary (navigation "see below" / "table below"
# cannot match: a verb is required). The negation guard drops the WHOLE
# line on any negated side phrase ("not/never well below") — a LINE-level
# kill, not instance-level: a mixed line ("not below X but above Y") is
# dropped entirely (accepted false negative — prefer false negatives, the
# c14 doctrine).
_C28_BELOW_RE = re.compile(r"(?i)\b(?:well|lands?|sits?|stays?|falls?)\s+below\b|\bunder\s+half\b")
_C28_ABOVE_RE = re.compile(
    r"(?i)\bexceeds?\b|\b(?:well|lands?|sits?|stays?)\s+above\b|\bat\s+least\s+half\b"
)
_C28_NEG_RE = re.compile(
    r"(?i)\b(?:not|never|no\s+longer)\s+(?:(?:well|lands?|sits?|stays?|falls?)\s+)?"
    r"(?:below|above)\b|\bexcept\s"
)
# Same-line recompute corroborator: positive decimals split at the first
# `vs`; slash (`/`) is NOT a ratio separator in this corpus (it is the
# paired-cells idiom: "rotated +0.349/+0.334"). 2-4 fractional digits so a
# coarse "vs chat 0.6" drops the corroborator (quoted-ratio path unaffected).
_C28_VS_RE = re.compile(r"(?i)\bvs\.?\s")
_C28_POSNUM_RE = re.compile(r"(?<![\d.\-])\+?(0?\.\d{2,4})\b")


def _c28_frac(s: str) -> Fraction:
    """Exact ``Fraction`` from a decimal literal, tolerating a bare leading
    dot (``.5`` -> 1/2) — the c13 ``_c13_registered_gates`` parse
    convention."""
    return Fraction("0" + s) if s.startswith(".") else Fraction(s)


def _c28_bands(plan: str) -> list[dict]:
    """Registered multiplicative decision-band lines: non-fenced,
    bold-labeled list items inside a success/kill/decision/evaluation-titled
    section (``_C13_GATE_SECTION_RE`` reused; the #825 v17 heading
    "## 6. Success + kill criteria (quantitative)" matches via the
    ``kill[- ]criteri`` alternation) carrying a ``cmp T x`` threshold with
    fractional T in (0, 1]. Per band: ``{label, cmp, thr: Fraction, line}``
    — a mirror of ``_c13_registered_gates``' fence-masked, section-scoped
    walk."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    bands: list[dict] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if not any(h.line <= i < h.end and _C13_GATE_SECTION_RE.search(h.text) for h in headings):
            continue
        if not _C28_LIST_ITEM_RE.match(line):
            continue
        label_m = _C28_BOLD_LABEL_RE.search(line)
        band_m = _C28_BAND_RE.search(line)
        if not (label_m and band_m):
            continue
        thr = _c28_frac(band_m.group("thr"))
        if not 0 < thr <= 1:
            continue
        bands.append(
            {
                "label": label_m.group(1).strip(),
                "cmp": band_m.group("cmp"),
                "thr": thr,
                "line": line.strip(),
            }
        )
    return bands


def _c28_ratio_assertions(plan: str) -> list[dict]:
    """Side-asserted precedent-ratio lines over non-fenced text. A line
    fires only when an explicit ``ratio ≈ r[-r2]`` token AND exactly one
    side (below XOR above vocabulary, negation-guarded at LINE level)
    co-occur on it. Per assertion:
    ``{line, side, side_text, quoted, recomputed, candidates}`` where
    ``candidates`` = quoted r1 (and r2 for a range) UNION the same-line
    vs-pair recompute — numerators are the positive decimals LEFT of the
    first ``vs`` (ratio-token spans blanked first), the denominator is the
    FIRST positive decimal RIGHT of it; a/b kept only when b > 0 (the
    zero-denominator guard — the ``Fraction(x, 0)`` class c13's detail
    builder documents) and 0 < a/b <= 2 (sanity window). ``Fraction``
    arithmetic throughout — exact boundary semantics at r == T, no
    float-equality wobble."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    assertions: list[dict] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        ratio_ms = list(_C28_RATIO_RE.finditer(line))
        if not ratio_ms or _C28_NEG_RE.search(line):
            continue
        below_m = _C28_BELOW_RE.search(line)
        above_m = _C28_ABOVE_RE.search(line)
        if bool(below_m) == bool(above_m):  # neither, or both (ambiguous)
            continue
        side_m = below_m or above_m
        quoted = {_c28_frac(m.group(g)) for m in ratio_ms for g in ("r1", "r2") if m.group(g)}
        blanked = list(line)
        for m in ratio_ms:
            blanked[m.start() : m.end()] = " " * (m.end() - m.start())
        blanked_line = "".join(blanked)
        recomputed: set[Fraction] = set()
        vs_m = _C28_VS_RE.search(blanked_line)
        if vs_m:
            denom_m = _C28_POSNUM_RE.search(blanked_line[vs_m.end() :])
            if denom_m:
                b = _c28_frac(denom_m.group(1))
                if b > 0:
                    for num_m in _C28_POSNUM_RE.finditer(blanked_line[: vs_m.start()]):
                        r = _c28_frac(num_m.group(1)) / b
                        if 0 < r <= 2:
                            recomputed.add(r)
        assertions.append(
            {
                "line": line.strip(),
                "side": "below" if below_m else "above",
                "side_text": side_m.group(0),  # type: ignore[union-attr]
                "quoted": quoted,
                "recomputed": recomputed,
                "candidates": quoted | recomputed,
            }
        )
    return assertions


def _c28_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no precedent-labeled decision bands`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale; British
    ``labelled`` accepted)."""
    return _standalone_na_declared(plan, r"no precedent[- ]labell?ed decision bands?")


def _c28_landed_band_label(bands: list[dict], landed_ge: bool) -> str:
    """Label of the first band whose comparator points at the branch every
    candidate lands in (the >= T branch when ``landed_ge``, the < T branch
    otherwise), or ``""`` when no band's comparator points that way."""
    wanted = (">", ">=", "≥") if landed_ge else ("<", "<=", "≤")
    for b in bands:
        if b["cmp"] in wanted:
            return b["label"]
    return ""


def _c28_offender_detail(offenders: list[tuple[dict, str]], T: Fraction, bands: list[dict]) -> str:
    """Bounded WARN detail (c13 conventions: at most 3 offenders shown,
    90-char line snippets): per offender the line snippet, the asserted
    side phrase, the quoted vs recomputed candidate ratios (rendered
    ``≈ 0.519``), T, the disagreement class (contradiction | straddle) and
    the branch placement — a CONTRADICTION names the single band label the
    candidates land in; a STRADDLE always reads "candidates span both
    branches of T", never a single landed-band label. Ends with the #825
    v17 incident anchor, the cross-quantity honesty clause, and the remedy
    menu."""

    def _render(vals: set[Fraction]) -> str:
        return ", ".join(f"≈ {float(v):.3g}" for v in sorted(vals))

    parts: list[str] = []
    for a, cls in offenders[:3]:
        cands = f"quoted {_render(a['quoted'])}"
        if a["recomputed"]:
            cands += f" + recomputed {_render(a['recomputed'])}"
        if cls == "straddle":
            placement = "candidates span both branches of T"
        else:
            landed_ge = a["side"] == "below"
            label = _c28_landed_band_label(bands, landed_ge)
            branch = "≥ T" if landed_ge else "< T"
            placement = f"every candidate lands in the {branch} branch" + (
                f" ({label!r})" if label else ""
            )
        parts.append(
            f"line \"{a['line'][:90]}\" asserts '{a['side_text']}' but {cands} against the "
            f"registered {float(T):g}× band → {cls} ({placement})"  # noqa: RUF001
        )
    shown = "; ".join(parts)
    if len(offenders) > 3:
        shown += "; …"
    return (
        f"{shown} — a decision band applied to the plan's OWN cited precedent must land in "
        "the branch the narrative assigns it (#825 v17: 0.349/0.673 ≈ 0.519 ≥ 0.5 narrated "
        "'lands well below'; verify_plan PASSed 0/0 — caught only at the critic layer). "
        "NOTE: this check cannot verify the ratio and the band concern the same quantity — "
        "if they don't, declare the N/A escape. Remedy: re-label the precedent's branch, "
        "move the threshold, or declare 'N/A — no precedent-labeled decision bands' on its "
        "own line, unwrapped (no backticks/quotes); the semantic verdict stays with the "
        "Statistics critic"
    )


def check_precedent_band_coherence(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a registered fractional decision band
    (``cmp T x`` in a success/kill/decision section), applied to a
    same-line side-asserted precedent ratio the plan itself quotes or
    implies (the vs-pair recompute), must land in the branch the narrative
    asserts. A straddle (a quoted range containing both sides of T while
    one side is asserted) also WARNs. Boundary convention: below := [0, T)
    hardcoded — the harvested band comparator is NOT consulted at r == T,
    so a ``<=``-band's r == T edge is an accepted WARN-only imprecision.
    NEVER FAILs (the c14 doctrine: a heuristic text check must not
    hard-block a legitimately-worded plan); the FAIL-grade semantic verdict
    stays with the Statistics critic (critic-lens-reference.md item 3
    trigger (c)). Accepted false negatives (v1; plan #1094 §4.4): plain
    absolute ``a >= c`` comparisons (cross-arm absolutes are unsound when
    precedent and design arms have different ceilings), multi-threshold
    plans (SKIP), side assertions in an adjacent sentence rather than on
    the ratio line, `%`-suffixed ratios, and `/`-separated ratios (the
    paired-cells idiom). Incident: #825 v17 (the 0.5x band vs its cited
    instruct precedent 0.3489/0.6731 = 0.519, narrated 'lands well below';
    caught only at the critic layer — verify_plan PASSed 0/0)."""
    cid, name = "c28_precedent_band_coherence", "decision-band precedent coherence"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: precedent-labeled decision bands are an experiment|analysis plan shape",
        )
    bands = _c28_bands(plan)
    if not bands:
        return _skip(cid, name, "no registered multiplicative decision band detected")
    if _c28_na_escape_declared(plan):
        return _pass(cid, name, "explicit N/A declared (no precedent-labeled decision bands)")
    thresholds = {b["thr"] for b in bands}
    if len(thresholds) != 1:
        return _skip(
            cid,
            name,
            f"{len(thresholds)} distinct band thresholds — precedent-to-band pairing "
            "ambiguous at the plan surface",
        )
    T = next(iter(thresholds))
    assertions = _c28_ratio_assertions(plan)
    if not assertions:
        return _skip(cid, name, "band present but no side-asserted precedent ratio line detected")
    offenders: list[tuple[dict, str]] = []
    for a in assertions:
        lo, hi = min(a["candidates"]), max(a["candidates"])
        if a["side"] == "below" and hi >= T:
            offenders.append((a, "contradiction" if lo >= T else "straddle"))
        elif a["side"] == "above" and lo < T:
            offenders.append((a, "contradiction" if hi < T else "straddle"))
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(assertions)} side-asserted precedent ratio line(s) coherent with the "
            f"registered {float(T):g}× band",  # noqa: RUF001
        )
    return _warn(cid, name, _c28_offender_detail(offenders, T, bands))


# ─── Check 29 — deliberate fence vs §7 conditional phase ────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "reconcile the WORST-CASE
# wall — base phases PLUS every conditional / extension phase that could run
# on the same provision — against the GCP lane's auto-delete fence": a
# deliberately-declared (value-bearing, non-default) --max-run-duration /
# max_run_duration fence coexisting with a §7 extension/retrain-class gate
# must reference that conditional phase near a declaration site. Founding
# offender: #1112 v2 (a 48h fence sized off base phases only, omitting its
# own §7 G1 dose extension — joint worst ~48-50h; caught only by the critic
# layer; v3 costs the extension and bumps the fence to 72h).

# Value-bearing deliberate fence declaration, RAW scan (fences included — a
# fenced gcloud/dispatch line is the real launch command; the c5/c26
# raw-scan precedent). A value of 7d/168h is the default FLEX_START ceiling
# (#741), not "deliberate"; a minutes value is the cap-probe command shape
# (#680 `--max-run-duration=20m` — unit not in h/d, so it never matches); a
# bare flag, a "default 7d" prose mention (no value directly after the
# flag), and a templated `={max_run}`/`<dur>` placeholder carry no value —
# none trigger. A loose "Nh near the flag" prose trigger is deliberately
# absent: #1112 v2's §0 line ("the 48 h `--max-run-duration` fence") sits 2
# lines from a Risks line containing "dose extension", so a prose trigger
# would self-satisfy and the founding offender would PASS.
_C29_FENCE_FLAG_RE = re.compile(
    r"(?i)--max-run-duration[=\s]+[`\"']?~?(\d+(?:\.\d+)?)\s*(h(?:ours?)?|d(?:ays?)?)\b"
)
_C29_FENCE_EXTRA_RE = re.compile(
    r"(?i)max_run_duration[\"']?\]?\s*[:=]\s*[\"'`]?~?(\d+(?:\.\d+)?)\s*"
    r"(h(?:ours?)?|d(?:ays?)?)\b"
)
# §7-slot / Decision-Gates heading predicate (heading levels >= 2).
# Deliberately permissive on the numbered form: the §7 slot also holds
# `Compute estimate` / `Risks` in infra-shaped plans — the extension-vocab
# gate below filters those (WARN-only polarity; calibration-swept, #1114).
_C29_SECT7_HEAD_RE = re.compile(r"(?i)^(?:§\s*)?7\b(?:[.:)\s]|$)|\bdecision gates?\b")
# Extension-class gate vocabulary. Bare "resume"/"re-run"/"re-judge" are
# deliberately EXCLUDED (crash-resume vocabulary saturates plans); a gate
# worded purely as "resume to step 60" is a named accepted false negative.
_C29_EXTENSION_RE = re.compile(
    r"(?i)\b(?:dose[- ]extension|extension|extend(?:s|ed|ing)?|re-?ladder\w*|"
    r"re-?train\w*|retrain\w*|second pass|additional (?:steps|pass(?:es)?|epochs?))\b"
)
# Conditional-cost evidence vocabulary (permissive direction: a match can
# only suppress a WARN). Gate labels (G1, G2, ...) are matched separately,
# case-SENSITIVE, only for labels actually harvested from §7 — a (?i)
# \bg\d+\b would match GCP machine types ("g2-standard-4").
_C29_EVIDENCE_RE = re.compile(
    r"(?i)§\s*7\b|\bsection\s+7\b|\b(?:extension|extend\w*|contingen\w*|conditional|"
    r"gate(?:'s|s)?|dose[- ]extension|re-?ladder\w*|re-?train\w*|retrain\w*|"
    r"second provision|across provisions|split across)\b"
)
_C29_WINDOW_LINES = 3  # pinned on BOTH sides: test_c29_evidence_outside_window_still_warns
# (upper bound: distance 4 WARNs) + test_c29_evidence_at_window_edge_passes
# (lower bound: distance 3 PASSes — kills a narrowing mutant).


def _c29_hours(val: str, unit: str) -> float:
    """Fence value normalized to hours (d/days -> x24; units pre-filtered
    to h/d by the declaration regexes)."""
    return float(val) * (24.0 if unit.lower().startswith("d") else 1.0)


def _c29_fence_decl_line_idxs(plan: str) -> list[int]:
    """RAW line indices carrying a value-bearing ``--max-run-duration`` /
    ``max_run_duration`` declaration whose value is not the 7d/168h default
    FLEX_START ceiling (#741). RAW scan — fences included (a fenced
    gcloud/dispatch line is the real launch command; c5/c26 precedent)."""
    idxs: list[int] = []
    for i, line in enumerate(plan.splitlines()):
        for rx in (_C29_FENCE_FLAG_RE, _C29_FENCE_EXTRA_RE):
            m = rx.search(line)
            if m and abs(_c29_hours(m.group(1), m.group(2)) - 168.0) > 1e-9:
                idxs.append(i)
                break
    return idxs


def _c29_gate_section_prose(plan: str) -> str | None:
    """Fence-masked prose of every §7-slot / Decision-Gates section (heading
    levels >= 2), joined across all matches; ``None`` when no such heading
    exists. The global ``_fence_mask`` excludes fenced example commands
    inside §7 — a gate is a prose contract (the ``_trigger_windows``
    fence-masked-trigger doctrine)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    parts: list[str] = []
    found = False
    for h in _headings(plan):
        if h.level < 2 or not _C29_SECT7_HEAD_RE.search(h.text.strip()):
            continue
        found = True
        parts.extend(
            line
            for line, fenced in zip(
                lines[h.line + 1 : h.end], mask[h.line + 1 : h.end], strict=True
            )
            if not fenced
        )
    return "\n".join(parts) if found else None


def _c29_offender_detail(decl_line: str, gate_hit: str, labels: list[str]) -> str:
    """Bounded WARN detail (c26 conventions): the first declaration line
    (truncated ~80 chars), the matched §7 extension vocabulary + harvested
    gate labels, the incident anchors, and BOTH remedies."""
    lab = f"; §7 gate label(s): {', '.join(labels)}" if labels else ""
    return (
        f"deliberate fence declaration {decl_line.strip()[:80]!r} coexists with a §7 "
        f"extension-class gate (matched {gate_hit!r}{lab}) but no declaration window "
        "references the conditional phase's wall cost — reconcile the WORST-CASE wall, "
        "base phases PLUS every conditional/extension phase on the same provision, "
        "against the fence (plan-compute-sizing.md § worst-case wall; #599: a 24h fence "
        "hard-deleted the pre-registered §7.3 extension probe at step 149/2400; #1112 "
        "v2: a 48h fence omitted its own §7 G1 dose extension, joint worst ~48-50h). "
        "Remedy: add the conditional phase's wall cost to the fence-reconcile sentence, "
        "or declare 'N/A — no conditional phase on this provision' on its own line, "
        "unwrapped (no backticks/quotes)"
    )


def check_fence_conditional_phase(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a deliberately-declared ``--max-run-duration``
    / ``max_run_duration`` fence (value-bearing, != the 7d/168h default
    ceiling) coexisting with an extension/retrain-class gate in the §7 /
    Decision-Gates section must reference that conditional phase (a §7 gate
    label, or conditional-cost vocabulary) within ±3 raw lines of a
    declaration line — ANY-SITE satisfy: one declaration window carrying
    the evidence clears the whole plan (the reconcile sentence is singular;
    requiring every mention would WARN on compressed §0 summaries).
    Fence-strip split: declaration scan RAW (the fence usually lives in a
    backticked/fenced launch command, and a fenced-only declaration with
    zero prose reconcile is exactly the silent-ride failure class); §7 gate
    detection fence-masked; evidence windows RAW (permissive direction).
    Mechanizes plan-compute-sizing.md § "reconcile the WORST-CASE wall —
    base phases PLUS every conditional / extension phase" (#599: a 24h
    fence hard-deleted the pre-registered §7.3 extension probe at step
    149/2400; #833: per-cell dispersion overran a deliberate 36h fence;
    #1112 v2: a 48h fence sized off base phases omitted its own §7 G1 dose
    extension, joint worst ~48-50h — only the critic caught it). NEVER
    FAILs (the c14/c28 doctrine). SCOPE (honest): mechanizes the
    DECLARED-fence subclass (#1112-shaped) of the incident class only.
    Known accepted gaps, each verified against the founding files: a
    prose-only fence (the actual #599 shape — "GCP max-run-duration
    (~20 h)", no flag/assignment) is invisible -> SKIP; a dispatch-time
    fence never written into the plan (the actual #833 shape) is invisible
    -> SKIP; bare resume/re-run/re-judge gates don't trigger; a
    second-provision split pre-registered ONLY in §7 still WARNs (remedy:
    the N/A phrase or a fence-window mention); evidence-vocabulary stray
    matches (e.g. an unrelated "gate" near the fence) suppress a real WARN
    — permissive direction; whether the referenced conditional cost is
    ARITHMETICALLY correct stays critic-owned. The #599/#833 SKIP shapes
    are pinned by tests (test_c29_prose_only_fence_skips /
    test_c29_no_fence_skips) plus the #1114 §6 sibling replay, so a future
    trigger widening fails loud."""
    cid, name = "c29_fence_conditional_phase", "deliberate fence vs §7 conditional phase"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: fence/§7-gate shapes are an experiment|analysis plan shape"
        )
    decl_idxs = _c29_fence_decl_line_idxs(plan)
    if not decl_idxs:
        return _skip(
            cid,
            name,
            "no deliberate (value-bearing, non-default) --max-run-duration fence "
            "declaration detected",
        )
    if _standalone_na_declared(plan, r"no conditional phase on this provision"):
        return _pass(cid, name, "explicit N/A declared (no conditional phase on this provision)")
    gates = _c29_gate_section_prose(plan)
    if gates is None:
        return _skip(cid, name, "no §7 / Decision Gates section detected")
    ext = _C29_EXTENSION_RE.search(gates)
    if not ext:
        return _skip(cid, name, "no extension/retrain-class conditional gate in §7")
    labels = sorted(set(re.findall(r"\bG\d+\b", gates)))
    lines = plan.splitlines()
    for i in decl_idxs:
        window = "\n".join(lines[max(0, i - _C29_WINDOW_LINES) : i + _C29_WINDOW_LINES + 1])
        ev = _C29_EVIDENCE_RE.search(window)
        if ev:
            return _pass(
                cid,
                name,
                "fence-reconcile window references the §7 conditional phase "
                f"(evidence {ev.group(0)!r})",
            )
        for lb in labels:
            if re.search(rf"\b{re.escape(lb)}\b", window):
                return _pass(
                    cid,
                    name,
                    "fence-reconcile window references the §7 conditional phase "
                    f"(gate label {lb!r})",
                )
    return _warn(cid, name, _c29_offender_detail(lines[decl_idxs[0]], ext.group(0), labels))


# ─── Check 30 — reused-bundle realized keys (WARN-only, conditional) ───────

_C30_BUNDLE_RE = re.compile(
    # NO `.safetensors` token (v2, methodology-critic Must-Fix): adapter-reuse
    # plans routinely quote `adapter_model.safetensors` near reuse vocabulary —
    # a sweep of all historical plans showed 9 fire via `.safetensors`
    # alone, ALL adapter-class (#459 #523 #528 #562 #570 #595 #627 #632 #653).
    # The project's multi-field bundles are single `.pt` files; a safetensors
    # STORE still triggers via its prose tokens (tensor bundle /
    # analysis_tensors / activation store / multi-field bundle).
    r"(?i)(\.pt\b|\.pth\b|tensor bundle|multi-?field bundle|"
    r"save-dict|analysis_tensors|activation store)"
)
_C30_SATISFIER_RE = re.compile(
    r"(?i)(verify_reused_artifact_keys"  # the canonical helper
    r"|mmap\s*=\s*True[^\n]{0,120}\.keys\(\)"  # inline mmap key read
    r"|consumer(?:'s)?\s+own\s+loader)"  # consumer-loader-run form
)


def check_realized_keys(plan: str, kind: str) -> CheckResult:
    """Plans reusing a multi-field tensor bundle must name a realized-keys
    verification (artifact-reuse.md check (c), incident #1073). WARN not
    FAIL: the bundle-reuse trigger is heuristic (same class as c6), and the
    semantic question — was the probe actually RUN against the pinned
    revision — stays with the fact-checker. Trigger scans stripped prose;
    the satisfier ALSO scans raw text, because the runnable command
    legitimately lives in a fenced block."""
    cid, name = "c30_realized_keys", "reused-bundle realized-keys verification"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    bundle_hits = [m.start() for m in _C30_BUNDLE_RE.finditer(text)]
    reuse_near_bundle = any(
        re.search(r"(?i)\breus\w*", text[max(0, i - 300) : i + 300]) for i in bundle_hits
    )
    if not reuse_near_bundle:
        return _skip(cid, name, "no multi-field bundle reuse detected")
    if _standalone_na_declared(plan, r"no multi-?field bundle reuse"):
        return _pass(cid, name, "explicit no-bundle-reuse declaration")
    if _C30_SATISFIER_RE.search(plan):  # raw plan: fenced commands count
        return _pass(
            cid, name, "realized-keys verification named (helper / mmap read / consumer loader)"
        )
    return _warn(
        cid,
        name,
        "plan reuses a multi-field tensor bundle but names no realized-keys "
        "verification — artifact-reuse.md check (c): run `uv run python "
        "scripts/verify_reused_artifact_keys.py --artifact <path> --keys "
        "<consumer keys>` (or the consumer's own loader) against the pinned "
        "artifact and paste the PASS line into §10 (incident #1073)",
    )


# ─── Check 31 — SKILL.md prose edit backed by a durability pin (WARN-only) ─

# Trigger: a SKILL.md path token on a non-fenced, non-negated line, with an
# edit-commitment verb within +/-120 chars of the path match (long unwrapped
# plan lines make whole-line co-occurrence noisy — measured on the
# 2026-07-09 corpus scan, task #1179 plan §6). The path arm admits any
# slash-joined prefix (`.claude/skills/issue/SKILL.md`, relative
# `issue/SKILL.md`) or a bare `SKILL.md` not glued to a path/word char.
_C31_PATH_RE = re.compile(r"(?i)(?:[\w.-]+(?:/[\w.-]+)*/|(?<![\w./-]))SKILL\.md")
_C31_EDIT_RE = re.compile(
    r"(?i)\b(?:add(?:s|ed|ing)?|insert\w*|append\w*|amend\w*|edit\w*|splice\w*"
    r"|prepend\w*|reword\w*|rewrit\w*|revise[sd]?|patch\w*"
    r"|new (?:section|paragraph|bullet|sentence|step|clause|line))\b"
)
_C31_EDIT_PROX_CHARS = 120
# Negation / boilerplate guards — measured corpus noise classes (#1179 §6):
# "zero SKILL.md edits" (#700), "No SKILL.md change needed" (#875), "no
# companion edit to SKILL.md" (#797), scope-table "No change" rows (#792),
# must-ask / must-bounce deviation boilerplate (#890, #806, #869). The gap
# atom allows path-internal dots (`SKILL.md change`) but blocks a
# sentence-ending dot-space, so the guard cannot leak across sentences.
_C31_NEG_GUARD_RE = re.compile(
    r"(?i)\b(?:no|zero|not?|without|never)\b(?:[^|;:.]|\.(?!\s)){0,24}"
    r"\b(?:edit(?:s|ed|ing)?|chang(?:e|es|ed))\b"
    r"|\bunchanged\b|\bincidental\b|must-ask|must bounce"
    r"|park[^|]{0,24}plan_pending"
)
# Satisfier: an exact labeled line (c5/c20 machine-readable-line pattern) —
# a c15-style loose evidence scan false-satisfied all 9 incident plan
# versions (unrelated test_ identifiers + incidental vocabulary), so the
# label is load-bearing. RAW scan (c11/c15 evidence convention: the line may
# legitimately sit in a fenced §-block or table). The NA separator class
# mirrors NA_RE (em/en dash, colon, paren, hyphen) so `Durability pin: N/A
# (reason)` satisfies too.
_C31_PIN_LABEL_RE = re.compile(r"(?i)\bdurability pin:\s*")
_C31_PIN_NA_RE = re.compile(r"(?i)\bdurability pin:\s*N/?A\b\s*[—–:(-]\s*\S")  # noqa: RUF001
_C31_NA_ALIAS_RE = re.compile(NA_RE + r"no durability pin\s*[:—–-]\s*\S")  # noqa: RUF001


def _c31_trigger_lines(plan: str) -> list[str]:
    """Non-fenced, non-negated lines carrying a SKILL.md path with an
    edit-commitment verb within +/-``_C31_EDIT_PROX_CHARS`` of the path
    match."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[str] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or _C31_NEG_GUARD_RE.search(line):
            continue
        for m in _C31_PATH_RE.finditer(line):
            lo = max(0, m.start() - _C31_EDIT_PROX_CHARS)
            hi = min(len(line), m.end() + _C31_EDIT_PROX_CHARS)
            if _C31_EDIT_RE.search(line[lo:hi]):
                out.append(line.strip())
                break
    return out


def _c31_satisfier(plan: str) -> tuple[str, str] | None:
    """First satisfier line as ``(form, detail)``: ``("pin", ...)`` for
    ``Durability pin: <...test_...>`` (a standing OR planned pin test — the
    NEW-pin registration sub-check below adjudicates the planned case), or
    ``("na", ...)`` for a reason-bearing NA escape. Bare ``Durability pin:
    N/A`` (no reason) does NOT satisfy. Detail strings are byte-preserved
    from the pre-#1557 single-string return (single caller)."""
    for line in plan.splitlines():
        m = _C31_PIN_LABEL_RE.search(line)
        if m and _TEST_IDENT_RE.search(line[m.end() :]):
            return ("pin", f"pin named ({line.strip()[:80]!r})")
        if _C31_PIN_NA_RE.search(line) or _C31_NA_ALIAS_RE.search(line):
            return ("na", f"no-pin justification declared ({line.strip()[:80]!r})")
    return None


# NEW-pin selector-registration sub-check (#1557; the #1546 parked gap): a
# pin FILE absent from disk is branch-new; a new test file outside
# select_step9c_tests.py's WORKFLOW_INVARIANT tuple never runs on a LATER
# SKILL.md diff (the .claude/skills/**/SKILL.md WORKFLOW_SURFACE_GLOBS
# entry short-circuits per-file mapping; .md files have no stem-test map)
# — the pin gates only its own landing round (lineage: #1210's pin landed
# unregistered; #1242/#1268 registered after the fact). File grain IS the
# gate-visibility grain: the tuple registers FILES, so a new test FUNCTION
# in a registered file is selector-visible by construction and never
# flags. Registration evidence is PLAN-GLOBAL (the base satisfier's
# granularity); whether the diff actually ships the registration stays
# with the code-reviewer (the c11/c15 bound). The negation window is
# one-sided (chars BEFORE the token only, the c41 idiom) — a disclosed
# residual; do not widen speculatively without a corpus re-scan. Negation
# tokens: `pending` deliberately EXCLUDED ("registration pending" is
# honest commitment phrasing in a plan, where everything is pending by
# construction); `excluded|removed` included (anti-commitment).
# Calibration (#1557 scan, 2026-07-20; old = origin/main@ef23fa28df,
# new = this file; corpus = tasks/*/*/plans/v*.md + .claude/plans/*.md
# from the main root, 3,024 files; kind="infra" uniform — an UPPER BOUND
# on the production fire set, kind-exempt plans SKIP in production):
# transitions SKIP->SKIP 2,390 / PASS->PASS 303 / WARN->WARN 328 /
# PASS->WARN 3; 0 forbidden flips (no WARN->PASS, no fire->SKIP, no flip
# on a plan without an absent-path pin). The 3 PASS->WARN plan-versions
# are version-siblings of ONE distinct plan (#1326 v1-v3), hand-inspected
# true positive: its pin file is absent from today's tree (never landed
# under that name) with no registration statement — exactly the #1546
# gap shape. Historically-new-now-landed pins (#1210/#1242/#1268 era)
# correctly read standing against today's tree -> no churn. ANY future
# change to these regexes re-runs the corpus scan and records the
# realized numbers here (the c27/c32 gate precedent).
_C31_PIN_PATH_RE = re.compile(r"\b(?:tests/(?:[\w.-]+/)*)?test_\w+\.py\b")
_C31_REG_TOKEN_RE = re.compile(r"\bWORKFLOW_INVARIANT\b")
_C31_REG_NEG_RE = re.compile(
    r"(?i)\b(?:not?|never|without|un-?registered|absent|missing|outside|excluded|removed)\b"
)
_C31_REG_NEG_WINDOW = 40
_C31_REG_LABEL_RE = re.compile(r"(?i)^\s*(?:[-*>+]\s+)*(?:\*\*)?selector registration\b")
_C31_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c34/c41 pattern)


def _c31_pin_lines(plan: str) -> list[str]:
    """ALL lines satisfying the pin-form predicate (Durability-pin label +
    test ident after it) — RAW scan, same convention as ``_c31_satisfier``."""
    out: list[str] = []
    for line in plan.splitlines():
        m = _C31_PIN_LABEL_RE.search(line)
        if m and _TEST_IDENT_RE.search(line[m.end() :]):
            out.append(line)
    return out


def _c31_new_pin_paths(plan: str) -> list[str]:
    """Sorted, normalized ``tests/`` paths named on pin lines whose FILE is
    absent under ``_C31_REPO_ROOT`` (=> branch-new). A bare filename
    normalizes to ``tests/<name>`` (the c41 ``_C41_TESTPATH_RE``
    convention)."""
    paths: set[str] = set()
    for line in _c31_pin_lines(plan):
        for m in _C31_PIN_PATH_RE.finditer(line):
            p = m.group(0)
            if not p.startswith("tests/"):
                p = f"tests/{p}"
            if not (_C31_REPO_ROOT / p).is_file():
                paths.add(p)
    return sorted(paths)


def _c31_reg_token_unnegated(line: str) -> bool:
    """``WORKFLOW_INVARIANT`` token with no negation token in the
    ``_C31_REG_NEG_WINDOW`` chars before it (the ``_c41_line_triggers``
    window idiom)."""
    for m in _C31_REG_TOKEN_RE.finditer(line):
        window = line[max(0, m.start() - _C31_REG_NEG_WINDOW) : m.start()]
        if not _C31_REG_NEG_RE.search(window):
            return True
    return False


def _c31_registration_named(plan: str) -> bool:
    """Form A: the un-negated tuple token on a pin line. Form B: a
    line-start ``Selector registration:``-labeled line carrying the
    un-negated token. RAW scan (a registration line may legitimately sit in
    a fenced diff/section block, matching the pin satisfier's convention)."""
    for line in _c31_pin_lines(plan):
        if _c31_reg_token_unnegated(line):
            return True
    for line in plan.splitlines():
        if _C31_REG_LABEL_RE.search(line) and _c31_reg_token_unnegated(line):
            return True
    return False


def check_skillmd_prose_pin(plan: str, kind: str) -> CheckResult:
    """``kind: infra|batch`` plans that commit to editing
    ``.claude/skills/**/SKILL.md`` prose must carry ONE labeled line naming
    a durability pin test (a pytest asserting the prose's presence/shape)
    or a one-line no-pin justification. SKILL.md protection prose with no
    pin is silently droppable by any later edit — lineage: #1134 (no pin),
    #1045 (pin optional), #884 (pin present but unlabeled). WARN not FAIL:
    the trigger is a line heuristic; the Phase 2 critics adjudicate. v1
    scope is SKILL.md paths only — extending to agents/rules/CLAUDE.md
    prose is a future calibration decision (the 2026-07-09 corpus scan
    measured that superset would-WARN at 174+ tasks, dominated by
    ledger-entry classes with no pin-test practice). Known residual FP
    class (disclosed): scope-table rows whose negation token sits >24
    chars from the edit verb (#1102 shape) still trigger — the 1-line NA
    escape is the remedy.

    NEW-pin selector-registration branch (#1557; the #1546 parked gap):
    when the pin-form satisfier names a ``tests/…test_*.py`` FILE absent
    from disk (branch-new — the c41 existence-gate reading; file grain
    because the selector's ``WORKFLOW_INVARIANT`` tuple registers FILES),
    the plan must additionally carry registration evidence — Form A: the
    un-negated tuple token on the pin line; Form B: one line-start
    ``Selector registration:`` labeled line carrying it — else WARN. NA
    escapes short-circuit BEFORE any disk access (the c34 idiom).
    Fail-open branches (disclosed under-triggers): a pin ident with no
    extractable ``tests/`` path PASSes plain, and a ``--plan-file``
    off-repo invocation (no ``tests/`` dir under the repo root) PASSes
    with the sub-check noted skipped (the c41 off-repo doctrine). The
    Goal's "rules-pin-discoverable" alternative is UNREACHABLE in v1
    scope: this trigger fires on SKILL.md paths only, and the #1496
    rules-pin discovery arm covers ``.claude/rules/*.md`` targets only —
    it gains a code arm only if the trigger ever widens (a future
    calibration decision). Out of mechanical scope: whether the named
    pin test / registration actually SHIPS in the diff (the
    code-reviewer checks the diff, same bound as c11/c15) — the disk
    probe here classifies NEW-vs-standing only."""
    cid, name = "c31_skillmd_prose_pin", "SKILL.md prose edit backed by a durability pin"
    if kind not in ("infra", "batch"):
        return _skip(
            cid, name, "kind-exempt: SKILL.md prose edits are an infra|batch (workflow-fix) shape"
        )
    trig = _c31_trigger_lines(plan)
    if not trig:
        return _skip(cid, name, "no SKILL.md edit-commitment line detected")
    sat = _c31_satisfier(plan)
    if sat is not None:
        form, sat_detail = sat
        if form == "na":
            # NA short-circuits before any disk access (the c34 NA idiom).
            return _pass(cid, name, sat_detail)
        tests_dir = _C31_REPO_ROOT / "tests"
        if not tests_dir.is_dir():
            # --plan-file off-repo: cannot adjudicate NEW-vs-standing;
            # fail-open (the c41 off-repo doctrine).
            return _pass(
                cid,
                name,
                sat_detail + " — registration sub-check skipped (tests/ absent under repo root)",
            )
        new_paths = _c31_new_pin_paths(plan)
        if new_paths and not _c31_registration_named(plan):
            return _warn(
                cid,
                name,
                f"plan pins SKILL.md prose to a branch-NEW test file ({', '.join(new_paths[:3])}"
                " — absent from disk) but names no Step-9c selector registration — a new pin "
                "file outside the selector's WORKFLOW-INVARIANT tuple never runs on a later "
                "SKILL.md diff (the workflow-surface glob short-circuits; .md files have no "
                "stem-test map), so the pin gates nothing after its own landing round (#1210 "
                "landed unregistered; #1242/#1268 registered after the fact; #1546). Remedy: "
                "state the registration in the plan — the registry tuple's name "
                "(underscore-joined) on the `Durability pin:` line itself, or on one standalone "
                "line starting `Selector registration:` naming the "
                "scripts/select_step9c_tests.py tuple — or pin via a new test added to an "
                "already-registered test file",
            )
        if new_paths:
            return _pass(
                cid,
                name,
                sat_detail + f" — NEW pin file(s) {', '.join(new_paths)} with Step-9c selector "
                "registration named",
            )
        return _pass(cid, name, sat_detail)
    return _warn(
        cid,
        name,
        f"plan commits to editing SKILL.md prose ({trig[0][:70]!r}) but names no durability "
        "pin — protection prose with no pytest asserting its presence/shape is silently "
        "droppable by any later SKILL.md edit (lineage: #884/#1045/#1134). Add one line "
        "`Durability pin: tests/test_<file>.py::test_<name>` (a standing pin test, or a NEW "
        "pin test this plan adds), or declare `Durability pin: N/A` followed on the same "
        "line by an em dash and a one-line reason (a bare `Durability pin: N/A` still WARNs)",
    )


# ─── Check 32 — fit-family + battery §9 basis grounding ────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Per-cell fit phases"
# (MUST-level; #1395 widened the section + this check to draw batteries): a
# §9 row looping a fit/solve/factorization over cells x folds x layers x
# ... — or running a permutation/bootstrap/null-draw BATTERY (trigger:
# c12's calibrated _BATTERY_TRIGGER_RE on basis-table rows — since #1967
# that union ALSO matches pool-quadratic screen rows, so a near-dupe /
# pairwise-similarity screen row demands a measured/pilot-gated basis
# exactly as the rule's POOL-SCALE clause requires (#1738/#1901); a row
# matching both families is a fit row) — must ground its per-call basis on a
# MEASURED 1-cell pilot (for a battery: one production-shape batched draw
# block), a cited prior-issue measured figure, or a pre-registered
# `pilot-gated` flag — an ASSERTED per-call cost is never a basis, and a
# FLOP floor is the cross-check, never the basis (#823: asserted ~2 s/fit
# vs ~125 s real, 12-20 h realized; #811: one inner kernel timed, dominant
# frame asserted, unit 3/108 at 19h21m; #931: wrong-device measurement,
# ~2.2-2.5x mid-run; #722: "sub-minute per cell" asserted, 19.5 CPU-h).
# Anti-boilerplate, BOTH polarities (the #1060 round-1 critic concern):
# the satisfier requires provenance vocabulary CO-LOCATED with a numeric
# timing token in the basis/wall cells — "basis: measured pilot" with no
# number WARNs (#552 v3's literal "measured: minutes"), and "~2 s/fit"
# with no provenance word WARNs (#823's literal shape).
# Calibration (DEVELOPMENT-SET numbers: the regexes were tuned on the same
# persisted-plan corpus they were measured on — read the rates as
# in-sample, not held-out; ANY future c32-regex change re-runs the corpus
# scan and records the realized numbers here, the c27 gate precedent).
# Re-scan 2026-07-09 (implementation-time, shipped regexes) over 1,731
# persisted plan-versions (tasks/*/*/plans/v*.md): full corpus 149
# plan-versions triggered (32 distinct plans) / 85 would-WARN (23 distinct
# plans; pre-#1060-rule era dominated); recent era (issue >= 1000): 419
# plan-versions -> 13 triggered, 3 would-WARN, all ONE distinct plan
# (#1112 v1-v3, whose own v4 basis added "(parent-measured kernel)" and
# PASSes). Incident recall 100%: #823 v1-v5 / #811 v1 / #722 v1-v3 / #931
# v1-v4 all WARN (#931 v8-v10 post-incident replans also WARN — asserted
# bases, defensible under the rule; #811 v2-v3 and #722 v4-v5 SKIP: those
# restructured versions carry no fit-family basis-table row); every
# post-fix version PASSes on a substantive span (#811 v4 "REALIZED
# ~2.6 h"; #722 v7 "ran ~9 min"; #810 v4+ "measured 0.385 s/cell" /
# "parent 10 min"; #928 v5+ "prior-issue 1.0 h"; #1112 v4+ "parent
# 3 min"). Division of labor (#1395 rewrite): c12 owns the battery
# multiplier arithmetic + batched commitment (FAIL for experiments); c32
# owns BASIS grounding (WARN) for fit-family AND battery basis-table rows;
# adequacy stays with the Methodology critic (critic-lens-reference.md
# item 10(iii)), fed by the WARN forwarding. Disclosed under-trigger:
# prose-only battery sizing (the #1092 incident shape — no battery table
# row) is invisible to c32 by construction — c12 + the item 10(iii) prose
# REVISE cover it; the §4.3 planner-side change pushes future battery
# bases INTO basis tables, where this branch then covers them.
# Battery-branch re-scan 2026-07-16 (#1395 widening; old module =
# origin/main@2c5891fe97, new = this file) over 3,445 corpus files
# (.claude/plans/*.md + tasks/*/*/plans/*.md, plan.md symlinks included),
# kind="experiment" uniform — an UPPER BOUND on the production fire set
# (kind-exempt plans SKIP in production): battery branch fires on 65
# plan-versions across 15 distinct issues; 52 plan-versions / 12 distinct
# issues carry >=1 ungrounded battery row (would-WARN); recent era
# (issue >= 1000): 3 distinct firing issues — #1332 + #1415 new-WARN
# (genuine ungrounded null-battery rows), #1335 grounded PASS
# ("#825 ~1 min/cell"). 0 forbidden flips: fit-branch per-row verdicts
# byte-stable corpus-wide, no other check flipped (allowed flips: 36).
# Incident recall, battery branch: #1092 versions do NOT fire (prose-only
# batteries — the disclosed under-trigger above; the item 10(iii) prose
# leg owns that shape); #778 GAINS coverage (SKIP->WARN on plan.md +
# v1-v8 — its null-battery rows ARE basis-table rows); #810 versions do
# not fire (no battery basis-table row). All 12 would-WARN issues are
# genuine battery rows (null-battery / permutation / resample
# components); zero incidental (judge-draw / battery-adjacent) fires.
# One-shot audit instrument: scripts/issue1395_corpus_audit.py.
# Disclosed residual gaming: a FABRICATED
# "measured 2 s/fit" passes — a mechanical check cannot verify
# measurement provenance (module scope discipline: a PASS here is never
# "grounding verified"); adequacy stays with the Methodology critic
# (critic-lens-reference.md item 10(iii)), fed by the WARN forwarding.

_C32_KERNEL_RE = re.compile(
    r"(?i)\bridge\b|\bsvd\b|\beigh\b|\beigvalsh\b|\blstsq\b|\bgcv\b"
    r"|\bloco\b|\bloocv\b|\blofo\b|\bmlp\b|\badamw\b|\bsgd\b|\bkrr\b"
    r"|\bglm\b|\birls\b|gradient[- ]descent|\bprobe[- ](?:train|fit)\w*"
    r"|\bfactoriz\w+"
    r"|\b(?:point|probe|many[- ]cell|per[- ]cell|per[- ]fold|closed[- ]form|serial)[ -]fits?\b"
    r"|\bfit loops?\b"
)
# NOTE: bare \bfits?\b deliberately EXCLUDED — it false-fires on "fits in
# HBM" / generation rows ("engine load ... 250 gens", #558) and cost 11
# extra full-corpus triggers in calibration for zero incident-recall gain.

_C32_LOOP_RE = re.compile(
    r"(?i)per[- ](?:cell|fold|layer|arm|trait|seed|unit|probe|context|pair|source|behavior)"
    r"|[×x]\s*\d|\d+\s*[×x]\b"  # noqa: RUF001 — the multiplication sign is real plan text
    r"|\d[\d,]*\s*(?:fits|solves|calls|folds|cells|refits|units)\b"
    r"|\bn_calls\b|\bfor each\b|\bacross (?:all )?\d+"
)

# Provenance vocabulary: measurement verbs + prior-figure citation forms.
# "parent"/"ran"/"#<M>" are load-bearing widenings — without them the
# corpus's legitimate prior-figure bases ("parent full grid ~10 min =>
# 0.58 s/cell", #810 v10; "v5 ran 28 layers in ~9 min", #722 v7) false-WARN.
_C32_PROVENANCE_RE = re.compile(
    r"(?i)\bmeasur\w+|\btimed\b|\bclocked\b|\bprofil\w+|\bbenchmark\w*"
    r"|\brealized\b|\bpilot\w*\b|\bran\b|\btook\b|\bparent\b"
    r"|\bprior[- ]issue\b|#\d{2,}|\bcommitted\b"
)

# Numeric timing token: a digit-bearing quantity with a time unit,
# optionally per-call ("125 s/fit", "~0.58 s/cell", "9 min").
# NOTE: an ASCII-hyphen range ("2-3 min") does NOT match (the lookbehind
# blocks it); the corpus's en-dash (U+2013) ranges do match.
# Lookbehind blocks "A100s"/"H100" ("100 s" inside an alnum run).
_C32_TIMING_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.\-])[~≈]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:ms|s|sec|seconds?|min|minutes?|hr?|hours?)\b"
    r"(?:\s*/\s*(?:it|call|fit|cell|unit|fold|row|draw|solve))?"
)

_C32_PILOT_GATED_RE = re.compile(r"(?i)\bpilot[- ]gated\b")


def _c32_offender_detail(offenders: list[tuple[str, str]]) -> str:
    """Bounded WARN detail (the c26 ``_c26_offender_detail`` shape): at most
    3 (component, basis) pairs, the rule anchor, the incident anchors, and
    every remedy (measured figure / prior-issue citation / pilot-gated /
    the branch-scoped standalone N/A escapes)."""
    shown = "; ".join(f"row {comp[:60]!r} basis {basis[:40]!r}" for comp, basis in offenders[:3])
    if len(offenders) > 3:
        shown += "; ..."
    return (
        f"{shown} — fit-family/battery row(s) whose basis carries neither (provenance "
        "vocabulary — measured/timed/pilot/#<M>/parent — co-located with a numeric per-call "
        "timing) nor a `pilot-gated` flag — an ASSERTED per-call cost is never a sizing basis "
        "and a FLOP floor is the cross-check, never the basis (plan-compute-sizing.md "
        "§ Per-cell fit phases; #823: asserted ~2 s/fit, ~125 s real, 12-20 h realized; #811: "
        "unit 3/108 at 19h21m; #1092: a batched battery priced by FLOP / assumed-throughput "
        "ran ~2.6x the naive booking). Ground the row on a measured 1-cell pilot at production "
        "shape (state the figure, e.g. `measured 125 s/fit`; for a battery, one "
        "production-shape batched draw block, e.g. `measured 3.8 min/draw-block`), cite a "
        "prior-issue measured figure (`#811 r2: 313 s/unit`), mark the basis `pilot-gated`, "
        "or — if the row is not a fit loop / draw battery — declare the branch escape on its "
        "own line, unwrapped (no backticks/quotes): `N/A — no fit-family phases` (fit rows) / "
        "`N/A — no draw battery` (battery rows)"
    )


def check_fit_basis_grounding(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: every basis-column compute-table row naming a
    fit-family kernel (ridge/SVD/eigh/lstsq/GCV/MLP/LOCO/...) AND a
    loop/multiplicity signal (per-cell/per-fold vocabulary, an NxM product,
    an "N fits" count) — or, since #1395, a permutation/bootstrap/null-draw
    BATTERY row (trigger: c12's calibrated ``_BATTERY_TRIGGER_RE`` — battery
    framing or a >=100-count draw vocabulary; since #1967 the union also
    classifies pool-quadratic SCREEN rows here, per the rule's POOL-SCALE
    clause; a judge-style "N=5 draws" row
    and a "graded 0-100" scale do NOT match; a row matching both families
    is a FIT row, so the fit branch + fit escape govern it) — must ground
    its basis — provenance vocabulary
    (measured/timed/pilot/#<M>/parent/ran/...) CO-LOCATED with a numeric
    timing token in the conversion-bearing cells (basis + wall, the c26
    escape-(a) precedent — a component cell like "reuse of #811 adapters"
    must not satisfy the citation class spuriously), or a literal
    ``pilot-gated`` flag anywhere in the row. Mechanizes
    plan-compute-sizing.md § "Per-cell fit phases" (#823/#811/#722/#931;
    battery widening #1092/#1395 — for a battery the per-call unit is one
    production-shape batched draw block).
    Anti-boilerplate BOTH polarities (the #1060 critic concern): "measured
    pilot" with no digit WARNs, and "~2 s/fit" with no provenance word
    WARNs. A FLOP-only basis WARNs by construction (no provenance token) —
    the rule: a FLOP floor is the cross-check, never the basis; there is
    deliberately NO ``FLOP-only`` escape. NEVER FAILs in v1 — both trigger
    and satisfier are text heuristics (the c26 precedent), and whether a
    stated figure is REAL / transfers stays critic-owned: a FABRICATED
    "measured 2 s/fit" passes (a mechanical check cannot verify
    measurement provenance — a PASS here is never "grounding verified").
    Disclosed under-triggers: fit/battery sizing stated only in prose (no
    basis-column table — the #1092 incident's own shape) is invisible in
    v1 (c12 independently covers prose draw batteries; the
    critic-lens-reference.md item 10(iii) prose REVISE owns basis adequacy
    there); a basis table lacking a wall column is invisible
    (parser precondition, c26 parity). Escapes, branch-scoped (anti-paste
    semantics via ``_standalone_na_declared``): the standalone line
    ``N/A — no fit-family phases`` excuses the FIT rows; ``N/A — no draw
    battery`` (shared with c12) excuses the BATTERY rows. Calibration
    numbers + the corpus re-scan
    gate on ANY future c32-regex change live in the comment block above
    ``_C32_KERNEL_RE``."""
    cid, name = "c32_fit_basis_grounding", "fit-family + battery §9 basis grounding"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: fit-family/battery §9 rows are an experiment|analysis plan shape",
        )
    rows = _c26_compute_table_rows(plan)
    fit_rows = [
        (comp, basis, wall, row_text)
        for comp, basis, wall, row_text in rows
        if _C32_KERNEL_RE.search(row_text) and _C32_LOOP_RE.search(row_text)
    ]
    battery_rows = [r for r in rows if r not in fit_rows and _BATTERY_TRIGGER_RE.search(r[3])]
    if not fit_rows and not battery_rows:
        return _skip(
            cid, name, "no fit-family or battery row in a basis-column compute table detected"
        )
    branch_state: list[str] = []
    offenders: list[tuple[str, str]] = []
    n_considered = 0
    for branch, rows_b, na_tail in (
        ("fit-family", fit_rows, r"no fit[- ]family (?:fit )?phases"),
        ("battery", battery_rows, r"no draw battery"),
    ):
        if not rows_b:
            continue
        if _standalone_na_declared(plan, na_tail):
            branch_state.append(f"{branch}: explicit N/A declared")
            continue
        for comp, basis, wall, row_text in rows_b:
            n_considered += 1
            conv = f"{basis} {wall}"  # conversion-bearing cells, the c26 escape-(a) precedent
            grounded = (
                _C32_PROVENANCE_RE.search(conv) and _C32_TIMING_RE.search(conv)
            ) or _C32_PILOT_GATED_RE.search(row_text)
            if not grounded:
                offenders.append((comp, basis))
    if offenders:
        return _warn(cid, name, _c32_offender_detail(offenders))
    if n_considered == 0:
        return _pass(cid, name, "; ".join(branch_state))
    detail = (
        f"{n_considered} fit-family/battery row(s); every considered basis carries "
        "provenance + a timing figure or pilot-gated"
    )
    if branch_state:
        detail += "; " + "; ".join(branch_state)
    return _pass(cid, name, detail)


# ─── Check 33 — checkpoint-ladder retention policy ─────────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Dose-ladder /
# multi-rung checkpoint retention" (MUST-level, the #1133 rule): any
# training phase persisting per-rung checkpoints for later selection must
# state its checkpoint-retention policy in §9 — DEFAULT: retain the
# dose-selected + latest rungs only, delete ruled-out rungs BETWEEN rungs;
# keep-all is the justified exception (full-ladder sizing at realized
# per-rung GB + `--boot-disk-gb` declared). Incident #1112: 30 full-FT
# dose-ladder rungs kept (>=15 GB, up to ~28 GB each); a compliant 575 GB
# keep-all bound sat under the planned 750 GB GCP boot disk; the
# GCP-to-RunPod failover delivered the `ft-7b` default 200 GB volume ->
# ENOSPC (errno 28) at rung 24/30.
# Trigger anti-fragility: raw `ladder|rung` vocabulary is heavily polluted
# by GCP BACKEND-ladder rungs (spot/flex-start/on-demand fallback rungs,
# the #1029/#1116/#1121 vocabulary) — measured raw surface ~521 pv / 179
# issues un-gated (~320/89 kind-gated). The compound-token trigger + the
# backend-rung exclusion on the rung-AND-checkpoint co-location branch
# remove that class entirely.
# Calibration (DEVELOPMENT-SET numbers, fitted IN-SAMPLE — including the
# recent-era slice: the regexes were tuned on the same persisted-plan
# corpus they were measured on; ANY future c33-regex change re-runs the
# corpus scan and records the realized numbers here, the c27/c32 gate
# precedent). Re-scan 2026-07-10 (implementation-time, AS-SHIPPED
# regexes) over 1,760 persisted plan-versions (tasks/*/*/plans/v*.md):
# 69 plan-versions triggered (20 distinct issues — genuinely
# checkpoint-ladder-bearing: #480 band-stop ladder, #653 dense-step grid,
# #1090 every-25-steps ladder, #1112 dose ladder, #488 epoch ladder, ...);
# 42 would-WARN (16 issues, pre-#1133-rule era dominated). Recent era
# (issue >= 1000; the §7 kill-criterion DENOMINATOR is recent-era
# plan-versions, N=448): 18 triggered pv (#1090 v1-v5 / #1092 v1-v5 /
# #1112 v1-v8); would-WARN ONLY #1090 v1-v5 (planned pre-rule). #1092
# PASSes; #1112 v7/v8 PASS on their explicit `**Disk / checkpoint
# retention:**` line. Satisfier-span audit over the 27 triggered-but-PASS
# versions: 'retained' x8, delete-co-location spans x14 ("rungs deleted",
# "checkpoints ... deleted", "checkpoint ... then DELETE"), 'retention'
# x2, 'prune(d)' x2, 'MarkerBandStopCallback' x1. Over-broad-token watch
# (re-download / prune vs disk-hygiene boilerplate): re-download matched
# ZERO spans; prune matched 2 — #491 v3 genuine (per-shard
# train->read->prune checkpoint sequencing) and #715 v3 borderline
# (weight-pruning-arm vocabulary; its v1/v2 pass on a genuine delete
# span) — no nuisance CLASS, no regex change.
# Honest disclosed limitation: #1112 v1-v3 — the incident's own plans —
# PASS: their §9 stated merge-transient deletion + a keep-all disk bound,
# so the retention SURFACE existed; the defect was semantic (sized to the
# planned lane's disk, keep-all as default). A mechanical check cannot
# adjudicate adequacy — c33 protects the SILENT class (ladder plans whose
# sizing sections say nothing about retention/deletion); stated-but-
# inadequate stays with Methodology lens item 16.

_C33_LADDER_COMPOUND_RE = re.compile(
    r"(?i)dose[- ]ladder|checkpoint[- ]ladder|ladder of checkpoints|band[- ]stop grid"
    r"|dose[- ]matching checkpoint grid|checkpoint rungs?|rung checkpoints?"
    r"|per[- ]rung checkpoints?"
)

# Mechanizes the rule's "any long run saving every k steps for a later
# pick" clause ("saves a checkpoint every 25 steps", "saving every ~500
# optimizer steps").
_C33_SAVE_EVERY_RE = re.compile(
    r"(?i)(?:checkpoints?|sav\w+)\s+every\s+~?\d+\s*(?:optimizer[- ])?(?:steps?|epochs?)"
)

_C33_RUNG_RE = re.compile(r"(?i)\brungs?\b")
_C33_CKPT_RE = re.compile(r"(?i)\bcheckpoints?\b|\bckpts?\b")

# GCP fallback-ladder exclusion (co-location branch ONLY): a line whose
# rung vocabulary is the backend router's (spot/flex-start/on-demand
# rungs, terminal rung, lanes, capacity) is not a checkpoint ladder.
_C33_BACKEND_RUNG_RE = re.compile(
    r"(?i)spot|flex[- ]start|on[- ]demand|runpod|terminal rung|\blanes?\b|fallback"
    r"|a2-|a3-|gcp ladder|capacity"
)

# Retention/bounding vocabulary. The delete co-location windows stop at
# sentence/cell boundaries (the `|` exclusion keeps a table row's delete
# verb from satisfying via an adjacent cell). `keep-all` deliberately
# satisfies — a STATED keep-all is the rule's justified-exception surface,
# whose adequacy is critic-owned. Generic disk tokens (`--boot-disk-gb`,
# GB figures) are deliberately NOT satisfiers — #1112 v1-v3 declared
# `--boot-disk-gb 750` and still ENOSPC'd; a disk flag is not a retention
# policy.
_C33_RETENTION_RE = re.compile(
    r"(?i)\bretention\b|\bretain\w*\b|keep[- ]all|keep (?:all|every|only)"
    r"|delet\w+[^.\n|]{0,80}(?:rungs?|ruled[- ]out|non[- ]selected|checkpoints?|ckpts?)"
    r"|(?:rungs?|checkpoints?|ckpts?)[^.\n|]{0,80}delet\w+"
    r"|upload[- ]as[- ]you[- ]go|delete[sd]? locally|re[- ]download"
    r"|band[- ]stop callback|MarkerBandStopCallback"
    r"|coarse\+refine|two[- ]pass grid|\bprune[sd]?\b|retained (?:set|rungs?)"
)

_C33_SIZING_KEYWORDS = ("resources", "parallelism", "compute", "disk")


def _c33_trigger_line(plan: str) -> str | None:
    """First non-fenced line carrying checkpoint-ladder vocabulary (quoted
    in the WARN detail), or None. Three arms, first match wins: a compound
    ladder token; the save-every-k-steps cadence; rung AND checkpoint
    co-located on one line WITHOUT backend-rung vocabulary (the GCP
    fallback-ladder exclusion — the load-bearing anti-fragility widening)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if _C33_LADDER_COMPOUND_RE.search(line) or _C33_SAVE_EVERY_RE.search(line):
            return line
        if (
            _C33_RUNG_RE.search(line)
            and _C33_CKPT_RE.search(line)
            and not _C33_BACKEND_RUNG_RE.search(line)
        ):
            return line
    return None


def _c33_sizing_scope(plan: str) -> str:
    """Union of the non-fenced text of every section whose heading carries a
    sizing keyword (resources/parallelism/compute/disk — the #1133 rule
    requires the policy in §9, but corpus headings drift: '## 9. Resources',
    '## 9. Resources & Parallelism', '### Compute projection'); the whole
    plan's non-fenced text when no such heading exists (structural absence
    must not manufacture WARNs — a WARN-only check fails toward silence)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    keep = [False] * len(lines)
    matched = False
    for h in _headings(plan):
        htext = h.text.casefold()
        if any(k in htext for k in _C33_SIZING_KEYWORDS):
            matched = True
            for i in range(h.line, h.end):
                keep[i] = True
    if not matched:
        return strip_fences(plan)
    return "\n".join(
        line
        for i, (line, fenced) in enumerate(zip(lines, mask, strict=True))
        if keep[i] and not fenced
    )


def check_ladder_retention(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a plan carrying checkpoint-ladder vocabulary
    on a non-fenced line (a dose/checkpoint-ladder compound token, a
    "saves a checkpoint every k steps" cadence, or rung + checkpoint
    co-located on one line without GCP backend-rung vocabulary) must carry
    retention vocabulary (retain / keep-all / delete-between-rungs /
    upload-as-you-go / band-stop / coarse+refine / prune / ...) within its
    compute-sizing section(s) — the union of every section whose heading
    names resources/parallelism/compute/disk, doc-wide fallback when no
    such heading exists. Mechanizes plan-compute-sizing.md § "Dose-ladder /
    multi-rung checkpoint retention" (the #1133 rule; incident #1112).
    NEVER FAILs — both trigger and satisfier are text heuristics (the
    c26/c32 precedent); adequacy of a STATED policy stays with the
    Methodology critic (lens item 16), fed by the WARN forwarding into the
    fact-checker + critic briefs. A PASS here is never "retention
    verified": a stated keep-all deliberately satisfies (the rule's
    justified-exception surface, critic-owned). Disclosed misses:
    (a) #1112 v1-v3 — the incident's own plans — PASS (their §9 stated
    merge-transient deletion + a keep-all bound, so the retention SURFACE
    existed; the defect was semantic and is Methodology lens item 16's);
    (b) a ladder phrased with zero token-set overlap under-triggers
    (FN = the status quo, reviewer-enforced only); (c) a crash-resume-only
    save cadence (no later selection) triggers via the save-every arm —
    the remedy is to state a retention policy anyway (e.g. keep-last-k,
    which a crash-resume cadence should state regardless), or, second, to
    declare the N/A escape ONLY when no phase persists per-rung
    checkpoints (the escape phrase would be semantically false for a plan
    that does persist them). Escape: the standalone line
    ``N/A — no per-rung checkpoint persistence`` (alias
    ``N/A — no checkpoint ladder``), anti-paste semantics via
    ``_standalone_na_declared``. Calibration numbers + the corpus re-scan
    gate on ANY future c33-regex change live in the comment block above
    ``_C33_LADDER_COMPOUND_RE``."""
    cid, name = "c33_ladder_retention", "checkpoint-ladder retention policy"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: checkpoint ladders are an experiment|analysis plan shape"
        )
    trig = _c33_trigger_line(plan)
    if trig is None:
        return _skip(cid, name, "no checkpoint-ladder vocabulary detected")
    if _standalone_na_declared(
        plan, r"(?:no per[- ]rung checkpoint persistence|no checkpoint ladder)"
    ):
        return _pass(cid, name, "explicit N/A declared (no per-rung checkpoint persistence)")
    if _C33_RETENTION_RE.search(_c33_sizing_scope(plan)):
        return _pass(cid, name, "retention vocabulary present in the compute-sizing scope")
    return _warn(
        cid,
        name,
        f"plan carries checkpoint-ladder vocabulary ({trig.strip()[:70]!r}) but its "
        "compute-sizing section(s) state no checkpoint-retention policy — a per-rung ladder "
        "sized without a retention default keeps every rung and ENOSPCs mid-run on a lane "
        "failover (plan-compute-sizing.md § Dose-ladder / multi-rung checkpoint retention, "
        "the #1133 rule; incident #1112: 30 full-FT rungs kept, errno 28 at rung 24/30 after "
        "a GCP-to-RunPod failover delivered a 200 GB volume). State the retention policy in "
        "the sizing section (DEFAULT: retain the dose-selected + latest rungs only, delete "
        "ruled-out rungs BETWEEN rungs; or the justified keep-all exception — full-ladder "
        "sizing at realized per-rung GB + `--boot-disk-gb` declared), or declare "
        "`N/A — no per-rung checkpoint persistence` on its own line, unwrapped "
        "(no backticks/quotes), if no phase persists per-rung checkpoints",
    )


# ─── Check 34 — verbatim insert vs size-ratchet headroom ──────────────────
# A plan mandating VERBATIM prose into a workflow_lint size-ratcheted file
# (.claude/agents/*.md — check_agent_spec_size; .claude/rules/LESSONS.md —
# check_lessons_index) whose remaining headroom is smaller than the quoted
# block makes lint-passes + the plan's own file-count constraint jointly
# unsatisfiable at implement time (#1230: 422 B headroom vs a 1,546 B
# verbatim paragraph for code-reviewer.md forced a documented 3rd-file
# cap-raise deviation). Grandfather caps deliberately hug live size
# (cap = measured + <=3 KB, typically ~1 KB), so ANY >~1 KB insert into a
# grandfathered spec exceeds headroom BY DESIGN — the remedy is therefore
# never "don't grow" but "budget the visible cap-raise IN THE PLAN"
# (workflow_lint.py: "a reviewed growth+cap-raise in one commit still
# passes"). WARN not FAIL: the trigger is a proximity heuristic; the
# Phase 2 critics adjudicate. infra|batch only: editing agent specs /
# LESSONS.md is workflow-fix work (calibration: all 8 recent-era corpus
# hits are kind: infra).
# Calibration (DEVELOPMENT-SET, measured against TODAY'S live sizes —
# historical headroom drifts; the c32 precedent): 1,837 persisted
# plan-versions scanned 2026-07-11; trigger fired on 166 versions;
# would-WARN 76 versions / 26 distinct plans (8 distinct at issue >= 1000:
# #1007 #1017 #1022 #1142 #1224 #1230 #1239 #1254 — every one a real
# plan-mandated over-headroom insert; the #1119/#1138/#1254/#1230
# cap-raises are recorded in AGENT_SPEC_SIZE_GRANDFATHER's own comments).
# Incident recall: #1230 v1 WARNs. Reproducible scan recipe (the
# kill-criterion (a) re-audit; re-run + re-record on ANY c34-regex
# change): from the repo root, for each tasks/*/*/plans/v*.md compute
# trigger := bool(_c34_targets(text)) and would-WARN := any (rel, nbytes)
# in _c34_targets(text).items() with _c34_headroom(rel, wl) is not None
# and nbytes > headroom, where wl := _c34_lint_constants(); count
# plan-versions and distinct plan dirs for both.
# Scope notes (disclosed, accepted residuals for a WARN-class v1):
# (a) the `Ratchet budget:` satisfier is DOCUMENT-GLOBAL, not per-target —
#     a two-file plan budgeting only one raise passes for both (pinned by
#     test_c34_budget_line_is_document_global);
# (b) nested-fence inserts UNDERCOUNT: _fence_mask's toggle reads an inner
#     ``` as a closer, so a verbatim block that itself contains a code
#     fence contributes only its pre-fence lines — a disclosed FN class
#     distinct from the non-fenced-insert FN (an insert described with no
#     fenced block at all, or with the path >_C34_WINDOW_LINES non-fenced
#     lines above the fence, never triggers — those stay with the human
#     critics);
# (c) TOCTOU: headroom is a PLAN-TIME snapshot of the live file sizes —
#     the target can grow between plan verification and implementation;
#     workflow_lint's commit-time FAIL is the hard backstop.
# Scope discipline: whether the budgeted cap-raise actually SHIPS is the
# code-reviewer's bound, not this check's (the c31/c11/c15 bound).

_C34_PATH_RE = re.compile(r"(?i)(?:\.claude/)?(?:agents/[\w.-]+\.md|rules/LESSONS\.md)")
_C34_VERB_RE = re.compile(
    r"(?i)\b(?:insert\w*|append\w*|add(?:s|ed|ing)?|splice\w*|verbatim|paste\w*)\b"
)
_C34_WINDOW_LINES = 10  # preceding non-fenced prose lines scanned per fence
# Digit-lookahead after the label on the SAME line = anti-paste armor: the
# WARN detail's remedy writes the label followed only by angle-bracket
# placeholders, so a verbatim paste of the detail can never self-satisfy.
_C34_BUDGET_RE = re.compile(r"(?i)\bratchet budget:(?=[^\n]*\d)")
_C34_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch


def _c34_lint_constants():
    """Lazy import of ``scripts/workflow_lint.py`` (540 KB module, ~345 ms
    measured) — paid ONLY when the c34 trigger fires, so typical plans keep
    the verifier sub-second. Single source of truth for the ratchet caps
    (the grandfather dict churns ~weekly; a copy WOULD drift). An
    ImportError is a real defect (both files live in scripts/) and
    propagates loud."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import workflow_lint

    return workflow_lint


def _c34_normalize_rel(match_text: str) -> str:
    """Normalize a ``_C34_PATH_RE`` match to the repo-root-relative ratcheted
    path (``.claude/agents/<name>.md`` / ``.claude/rules/LESSONS.md``)."""
    tail = match_text
    if tail.lower().startswith(".claude/"):
        tail = tail[len(".claude/") :]
    if tail.lower().startswith("agents/"):
        return ".claude/agents/" + tail[len("agents/") :]
    return ".claude/rules/LESSONS.md"


def _c34_targets(plan: str) -> dict[str, int]:
    """``{normalized rel path -> summed fenced-block UTF-8 bytes}`` for every
    fenced block whose preceding ``<=_C34_WINDOW_LINES`` NON-fenced lines
    carry a ratcheted path AND an insertion verb. Block bytes = joined
    content lines + one trailing newline (fence delimiters excluded); the
    realized insert may differ by a separator newline or two — immaterial
    at the hundreds-of-bytes scale the check discriminates."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    targets: dict[str, int] = {}
    in_fence = False
    open_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("```") or stripped.startswith("~~~")):
            continue
        if not in_fence:
            in_fence = True
            open_idx = i
            continue
        in_fence = False
        window: list[str] = []
        j = open_idx - 1
        while j >= 0 and len(window) < _C34_WINDOW_LINES:
            if not mask[j]:
                window.append(lines[j])
            j -= 1
        wtext = "\n".join(reversed(window))
        m = _C34_PATH_RE.search(wtext)
        if m is None or not _C34_VERB_RE.search(wtext):
            continue
        rel = _c34_normalize_rel(m.group(0))
        content = "\n".join(lines[open_idx + 1 : i]) + "\n"
        targets[rel] = targets.get(rel, 0) + len(content.encode("utf-8"))
    return targets


def _c34_headroom(rel: str, wl) -> tuple[int, int, str] | None:
    """``(headroom, cap, cap_source)`` for a ratcheted rel path under
    ``_C34_REPO_ROOT``; ``None`` when the live file is absent (headroom
    uncomputable — a plan may be CREATING the file, which starts with the
    full cap of headroom). ``cap_source`` names the binding workflow_lint
    constant for the WARN detail. Sizes in BYTES (``stat().st_size``) —
    parity with ``check_agent_spec_size`` / ``read_bytes()``."""
    p = _C34_REPO_ROOT / rel
    if not p.is_file():
        return None
    size = p.stat().st_size
    if p.name == "LESSONS.md":
        # #1504: the TOTAL growth ratchet is retired — the binding total
        # constraint is the leanness cap. Per-row / non-row budgets also
        # bind at implement time; c34's summed-block-bytes heuristic keeps
        # the total cap as its denominator (a single-row insert over
        # _LESSONS_ROW_MAX_BYTES is a disclosed FN here, caught by the
        # lint). No trigger-regex change (_C34_PATH_RE / _C34_VERB_RE /
        # _C34_BUDGET_RE untouched) — the re-audit clause does not fire;
        # the looser denominator only monotonically reduces WARNs.
        return wl._LESSONS_MAX_BYTES - size, wl._LESSONS_MAX_BYTES, "_LESSONS_MAX_BYTES"
    cap = wl.AGENT_SPEC_SIZE_GRANDFATHER.get(p.name)
    if cap is not None:
        return cap - size, cap, "AGENT_SPEC_SIZE_GRANDFATHER"
    return wl.AGENT_SPEC_FAIL_BYTES - size, wl.AGENT_SPEC_FAIL_BYTES, "AGENT_SPEC_FAIL_BYTES"


def _c34_offender_detail(offenders: list[tuple[str, int, int, int, str]]) -> str:
    """Bounded WARN detail: at most 3 offender tuples, the #1230 incident
    anchor, then the three remedies. Anti-paste armored: after the
    ``Ratchet budget:`` label the text carries ONLY angle-bracket
    placeholders (no digit on the line — ``_C34_BUDGET_RE``'s lookahead
    cannot match a pasted copy; the incident numbers all sit BEFORE the
    label), and the N/A phrase is backtick-wrapped (unrecognized by
    ``_standalone_na_declared``, #1238)."""
    shown = "; ".join(
        f"{rel}: insert ~{nbytes} B > headroom {headroom} B (cap {cap} [{src}] - live size)"
        for rel, nbytes, headroom, cap, src in offenders[:3]
    )
    more = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
    return (
        f"plan mandates verbatim fenced insert(s) exceeding the named ratcheted file(s)' "
        f"remaining size-ratchet headroom: {shown}{more} — workflow_lint-passes and the "
        "plan's own file-count constraint become jointly unsatisfiable at implement time "
        "(#1230: a paragraph larger than code-reviewer.md's headroom forced an un-planned "
        "third-file cap-raise deviation). Remedies: budget the cap-raise IN THE PLAN with "
        "one line `Ratchet budget: raise <constant>['<file>.md'] to <new cap>` (new cap = "
        "post-insert measured size plus at most the grandfather headroom bound), or trim "
        "the insert to fit, or declare `N/A — no verbatim ratcheted-file insertion` on its "
        "own line (write the declaration unwrapped — the backticks here are anti-paste "
        "armor)"
    )


def check_ratchet_headroom(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, ``kind: infra|batch``: a fenced block whose
    preceding ``<=_C34_WINDOW_LINES`` non-fenced lines name a ratcheted path
    (``.claude/agents/*.md`` / ``.claude/rules/LESSONS.md``) plus an
    insertion verb is treated as a verbatim insert into that file; when the
    per-target summed block bytes exceed the file's live headroom
    (cap - ``stat().st_size``, caps lazy-imported from workflow_lint) the
    check WARNs with the arithmetic + the three remedies. Satisfiers: a
    non-fenced ``Ratchet budget:`` line carrying a post-label digit (the
    plan budgets the cap-raise — the legitimate path, since grandfather
    caps hug live size by design), or the standalone escape
    ``N/A — no verbatim ratcheted-file insertion``. Named-file-absent →
    SKIP (a plan may be CREATING the file; ``--plan-file`` mode must never
    crash off-repo). NEVER FAILs — trigger + satisfier are text heuristics
    (the c31 template); calibration, scope notes (document-global
    satisfier, nested-fence undercount, plan-time TOCTOU snapshot) and the
    scan recipe live in the section comment above ``_C34_PATH_RE``."""
    cid, name = "c34_ratchet_headroom", "verbatim insert fits size-ratchet headroom"
    if kind not in ("infra", "batch"):
        return _skip(
            cid,
            name,
            "kind-exempt: ratcheted-file verbatim inserts are an infra|batch (workflow-fix) shape",
        )
    if _standalone_na_declared(plan, r"no verbatim ratcheted[- ]file insertion"):
        return _pass(cid, name, "escape declared: no verbatim ratcheted-file insertion")
    targets = _c34_targets(plan)
    if not targets:
        return _skip(cid, name, "no fenced block associated with a ratcheted-file insertion")
    lines = plan.splitlines()
    for line, fenced in zip(lines, _fence_mask(lines), strict=True):
        if not fenced and _C34_BUDGET_RE.search(line):
            return _pass(cid, name, f"cap-raise budgeted ({line.strip()[:80]!r})")
    wl = _c34_lint_constants()  # lazy: only reached on trigger
    offenders: list[tuple[str, int, int, int, str]] = []
    checked = 0
    for rel, nbytes in sorted(targets.items()):
        hr = _c34_headroom(rel, wl)
        if hr is None:
            continue  # absent on disk: headroom uncomputable
        checked += 1
        headroom, cap, cap_source = hr
        if nbytes > headroom:
            offenders.append((rel, nbytes, headroom, cap, cap_source))
    if not checked:
        return _skip(
            cid, name, "named ratcheted file(s) not present on disk — headroom uncomputable"
        )
    if offenders:
        return _warn(cid, name, _c34_offender_detail(offenders))
    return _pass(cid, name, f"{checked} ratcheted-file insert(s) fit remaining headroom")


# ─── Check 35 — revision-pinned reuse verified at the pin (WARN-only) ──────

# Trigger: a 40-hex token with revision/pin vocabulary within +/-120 chars,
# HF-context AND reuse vocabulary within +/-300 chars (the c6/c30 proximity
# convention), scanning STRIPPED prose. The revision-vocab window is what
# keeps git code SHAs (Repro-card `commit=<sha>` rows) out; the HF-context
# window keeps "pinned to commit <sha>" git rows out (#1345 shape only).
_C35_HEX40_RE = re.compile(r"\b[0-9a-f]{40}\b")
_C35_REV_VOCAB_RE = re.compile(r"(?i)\brevision\b|\bpin(?:ned|s)?\b")
_C35_HF_CTX_RE = re.compile(
    r"(?i)superkaiba1/|hf_hub_download|huggingface|hf (?:model|data) repo"
    r"|list_repo_(?:files|tree)|snapshot_download|repo_id|repo_type"
)
_C35_REUSE_RE = re.compile(r"(?i)\breus\w*|\binherit\w*")
_C35_REV_WIN, _C35_CTX_WIN = 120, 300
# Satisfiers scan RAW text (c30 convention: runnable probe commands
# legitimately live in fenced blocks): a Hub-probe callable with a
# `revision=` kwarg on the same line, or a prose verified-at-pin statement.
# `get_paths_info` is deliberately EXCLUDED: the artifact-reuse item-(j)
# pairwise-provenance boilerplate (`get_paths_info(expand=True,
# revision=...)`) verifies commit-DATE coherence, not existence-at-pin, and
# it sits in standard §10 rows — including it blinded the check to its own
# motivating incident (#1345 plan v3 line 446).
_C35_PROBE_SATISFIER_RE = re.compile(
    r"(?i)(?:list_repo_(?:tree|files)|file_exists|hf_hub_download)"
    r"[^\n]{0,200}\brevision\s*[=:]"
)
_C35_PROSE_SATISFIER_RE = re.compile(
    r"(?i)verif\w+[^\n]{0,120}\bat\s+(?:the\s+)?(?:pinned\s+)?revision\b"
)


def check_pinned_revision_reuse(plan: str, kind: str) -> CheckResult:
    """Plans reusing an HF artifact at a pinned 40-hex revision must name a
    revision-scoped existence verification (incident #1345: a default-branch
    probe read CONFIRMED while 2/4 stems returned 0 files at the pin). WARN
    not FAIL: 'reuse row' detection is heuristic (same class as c6/c30), and
    the semantic question — was the probe actually RUN, per stem, at the pin
    — stays with the fact-checker (SKILL.md Phase 1.5).

    Disclosed FALSE-NEGATIVE residuals — a SKIP is never read as coverage:
    short-hex pins (7-12 hex, e.g. #1345 v4's 10-hex pin), branch/tag pins
    (non-hex revisions), and pins held only in a code constant (zero hex in
    the plan prose) do not trigger — the fact-checker instruction ("read the
    actual code/config") is the coverage for the constant case; a
    revision-threaded `hf_hub_download` CONSUME recipe satisfies without a
    stated probe (the disclosed consume residual). The WARN detail below is
    deliberately satisfier-inert (no Hub-callable + `revision=` on one line,
    no 'verif...at...revision' shape): bounced plans paste verifier details
    verbatim, and a self-matching detail would false-PASS exactly the
    flagged-then-revised plans (the #810 spurious-satisfaction shape) —
    pinned by test_c35_warn_detail_matches_no_satisfier."""
    cid, name = "c35_pinned_revision_reuse", "revision-pinned reuse verified at pin"
    if kind not in ("experiment", "analysis"):
        return _skip(cid, name, "kind-exempt")
    text = strip_fences(plan)
    pinned_reuse = False
    for m in _C35_HEX40_RE.finditer(text):
        w_rev = text[max(0, m.start() - _C35_REV_WIN) : m.end() + _C35_REV_WIN]
        w_ctx = text[max(0, m.start() - _C35_CTX_WIN) : m.end() + _C35_CTX_WIN]
        if (
            _C35_REV_VOCAB_RE.search(w_rev)
            and _C35_HF_CTX_RE.search(w_ctx)
            and _C35_REUSE_RE.search(w_ctx)
        ):
            pinned_reuse = True
            break
    if not pinned_reuse:
        return _skip(cid, name, "no revision-pinned HF reuse detected")
    if _standalone_na_declared(plan, r"no revision[- ]pinned reuse"):
        return _pass(cid, name, "explicit no-pinned-reuse declaration")
    if _C35_PROBE_SATISFIER_RE.search(plan) or _C35_PROSE_SATISFIER_RE.search(plan):
        return _pass(cid, name, "revision-scoped existence verification named")
    return _warn(
        cid,
        name,
        "plan reuses an HF artifact at a pinned 40-hex revision but names no "
        "revision-scoped existence check - probe each named stem with the revision "
        "kwarg set to the pin (list_repo_tree scoped to the stem prefix on the "
        "~1M-file data repo, or list_repo_files on small repos; >=1 file per stem); "
        "a default-branch probe does NOT satisfy (incident #1345: 2/4 stems returned "
        "0 files at the pin); or declare `N/A - no revision-pinned reuse` on its own "
        "line, unwrapped",
    )


# ─── Check 36 — numeric containment claims (WARN-only; experiment + analysis) ─

_C36_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
# Range operand pair. Two alternatives:
#   (a) en/em-dash or "to" separator, signed operands allowed ("-0.5 to 0.3");
#   (b) unspaced ASCII hyphen, UNSIGNED operands only ("2-4", "0.25-0.5") —
#       a signed/hyphen mix ("-0.5-0.3") is ambiguous between range and
#       negative pair and is NEVER parsed (accepted false negative).
# The leading lookbehind blocks matching the tail of a larger number or a
# signed token; \b after operands blocks partial-digit backtracks (the c28
# :4487 convention — without it "150-330 band" mis-parses); (?!-\d) on the
# hyphen form rejects date/version chains ("2026-07-15").
_C36_RANGE = (
    rf"(?P<lo>(?<![\d.\-−+])[+\-−]?{_C36_NUM})%?\s*(?:[–—]|\bto\b)\s*"  # noqa: RUF001 — en dash is real plan text
    rf"(?P<hi>[+\-−]?{_C36_NUM})"  # noqa: RUF001 — U+2212 minus is real plan text
    rf"|(?P<lo2>(?<![\d.\-−+]){_C36_NUM})-(?P<hi2>{_C36_NUM}(?!-\d))"  # noqa: RUF001
)
# Containment claim: verb + bounded filler + range + bounded tail + range-noun,
# all on ONE line. Filler excludes .;:| so a claim never crosses a sentence,
# clause-label, or table-cell boundary; the post-range lookahead rejects
# compound count modifiers ("10-20-draw random band", #816 v1 L227).
# "estimate" and "guidance" are deliberately NOT range-nouns (the #825
# aggregate-delta and t-SNE-guidance corpus shapes are accepted false
# negatives — both FP-prone).
_C36_CLAIM_RE = re.compile(
    rf"\b(?:inside|within)\b(?P<fill>[^.;:|\n]{{0,45}}?)(?:{_C36_RANGE})\b"
    rf"(?![–-][A-Za-z])\s*%?(?P<tail>[^.;|\n]{{0,18}}?)"  # noqa: RUF001 — en dash is real plan text
    rf"\b(?:spread|band|range|window|interval|CIs?)\b",
    re.IGNORECASE,
)
# Candidate claimed values: numbers in the same clause BEFORE the verb. The
# leading lookbehind kills identifier digits ("H2", "#508", "Tier-1",
# "checkpoint-20", "coef-0"); the trailing lookahead kills hex/sha and
# unit-glued tokens ("48de22...", "13h").
_C36_CAND_RE = re.compile(
    rf"(?<![A-Za-z#\d.\-−+/])[+\-−]?{_C36_NUM}%?(?![A-Za-z0-9])"  # noqa: RUF001
)
_C36_NEG_RE = re.compile(r"(?i)\b(?:not|never|no longer|nor|outside)\b")


def _c36_frac(tok: str) -> Fraction:
    """Exact ``Fraction`` from a claim/operand token: strips a trailing %, a
    leading +, maps U+2212 minus to ASCII, then delegates to ``_c28_frac``
    (the leading-dot-tolerant c13/c28 parse convention — cross-check helper
    reuse per the c28 <-> ``_C13_GATE_SECTION_RE`` precedent)."""
    return _c28_frac(tok.rstrip("%").lstrip("+").replace("−", "-"))  # noqa: RUF001


def _c36_na_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no numeric containment claims`` escape (see
    ``_standalone_na_declared`` for the anti-paste rationale)."""
    return _standalone_na_declared(plan, r"no numeric containment claims?")


def _c36_offender_detail(offenders: list[tuple[str, Fraction, Fraction, list[Fraction]]]) -> str:
    """Bounded WARN detail (c28 conventions: at most 3 offenders shown,
    90-char line snippets): per offender the line snippet, the candidate
    claimed values, and the bounds none of them lands in. Ends with the
    #1315 incident anchor and the remedy menu. The anchor + remedy prose
    is deliberately NON-matching (the c28 detail-inertness precedent): it
    phrases containment without an inside/within verb + range + range-noun
    co-occurrence, so a verbatim paste of this detail into a revised plan
    cannot re-trigger the check by itself (the 90-char offender snippet may
    still quote the plan's own claim — the remedy menu names the
    blockquote/fence move for exactly that)."""
    parts: list[str] = []
    for line, lo, hi, cands in offenders[:3]:
        vals = ", ".join(f"{float(v):.3g}" for v in cands)
        parts.append(
            f'line "{line[:90]}" - no candidate value (~ {vals}) lies in '
            f"[{float(lo):.3g}, {float(hi):.3g}]"
        )
    shown = "; ".join(parts)
    if len(offenders) > 3:
        shown += "; ..."
    return (
        f"{shown} - an explicit numeric containment claim must be arithmetically "
        "true (#1315 v4 L66 asserted containment of 0.724 in a 0.737-0.820 "
        "spread; verify_plan PASSed 0/0 - caught only at the critic layer). "
        "Remedy: fix the arithmetic, rewrite the prose (e.g. '0.013 BELOW the "
        "... spread', the #1315 v5 correction), move a quoted incident line "
        "into a blockquote or fence, or declare `N/A - no numeric containment "
        "claims` on its own line, unwrapped (no backticks/quotes); the semantic "
        "verdict - WHICH number the prose attributes - stays with the "
        "Statistics critic"
    )


def check_numeric_containment(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: an explicit numeric containment claim —
    a claimed number, a containment verb (inside/within), and an explicit
    numeric A-B range with a range-noun (spread/band/range/window/
    interval/CI) on one line — must be arithmetically true: some
    attributable claimed number in the same clause lies in [A, B]
    (boundary-inclusive; reversed bounds normalized). NEVER FAILs (the
    c14/c28 doctrine: a heuristic free-prose check must not hard-block);
    the semantic verdict — WHICH number the prose attributes — stays with
    the Statistics critic. Incident: #1315 plan v4 L66 (0.724 asserted
    inside 0.737-0.820; verify_plan PASSed 0/0; two critics caught it by
    hand -> #1375). Accepted false negatives (v1, named): claims whose
    subject sits in a previous sentence/clause; a decoy same-clause number
    inside the range masking a false claim; %-vs-unitless mixed units;
    scientific notation; noun-before-range word order ("the band
    0.6-0.8"); "estimate"/"guidance" nouns; ± tolerance forms (not
    ranges); signed hyphen ranges; bare-"in" containment ("N in the
    A-B band") and "between A and B" phrasings (both FP-prone; the
    {inside, within} verb set is deliberate).
    """
    cid, name = "c36_numeric_containment", "numeric containment claims"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid, name, "kind-exempt: containment-claim prose is an experiment|analysis plan shape"
        )
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    n_claims = 0
    offenders: list[tuple[str, Fraction, Fraction, list[Fraction]]] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or line.lstrip(" \t").startswith(">"):
            continue  # fenced code + blockquotes: quoted text, not claims
        blanked = line  # length-preserving blank of prior match spans
        for m in _C36_CLAIM_RE.finditer(line):
            lo_s = m.group("lo") or m.group("lo2")
            hi_s = m.group("hi") or m.group("hi2")
            lo, hi = _c36_frac(lo_s), _c36_frac(hi_s)
            if lo > hi:
                lo, hi = hi, lo  # reversed bounds: normalize, still verify
            pre = blanked[: m.start()]
            cut = max(pre.rfind("|"), pre.rfind(":"), pre.rfind(";"))
            seg = pre[cut + 1 :]  # clause/cell-bounded candidate window
            blanked = blanked[: m.start()] + " " * (m.end() - m.start()) + blanked[m.end() :]
            if _C36_NEG_RE.search(seg[-24:]):
                continue  # negated/outside claims: out of scope
            rng_pct = "%" in line[m.start() : m.end()]
            cands = [
                _c36_frac(t.group(0))
                for t in _C36_CAND_RE.finditer(seg)
                if t.group(0).endswith("%") == rng_pct
            ]
            if not cands:
                continue  # no attributable claimed value -> not a claim
            n_claims += 1
            if not any(lo <= c <= hi for c in cands):
                offenders.append((line.strip(), lo, hi, cands))
    if n_claims == 0 and not offenders:
        return _skip(cid, name, "no numeric containment claims detected")
    if _c36_na_escape_declared(plan):
        return _pass(
            cid, name, "explicit N/A declared (no numeric containment claims of this plan's own)"
        )
    if not offenders:
        return _pass(cid, name, f"{n_claims} numeric containment claim(s) arithmetically coherent")
    return _warn(cid, name, _c36_offender_detail(offenders))


# ─── Check 37 — no-flags bundling claim vs workflow_lint dispatch ──────────

# Trigger: a NON-fenced line carrying a `--check-<flag>` token together with
# a CLAIM-VERB-ANCHORED no-flags bundling assertion and no negation/scoping
# token (same-line radius — the c16 re-generat precedent: window scans sweep
# discussion noise, and set membership already exonerates true claims
# wherever they sit). The verb anchor is load-bearing (2026-07-16
# calibration): bare `--check-*` + no-flags co-occurrence swept in the
# two-command acceptance idiom ("run `--check-asks` AND the no-flags
# default run") at 293 WARNs over 838 corpus plan versions; requiring a
# bundling verb linking the flag list to the no-flags run drops that whole
# class while keeping the #1322-v1 "included in the no-flags default run"
# shape and the endemic corpus false claim "no-flags run bundles/includes
# `--check-asks`" (corrected #1007 v1→v2).
_C37_LINT_PATH = Path(__file__).resolve().parent / "workflow_lint.py"  # tests monkeypatch
# Keep `args.check_X or no_flags` as workflow_lint main()'s dispatch shape —
# c37 parses exactly this form. A dispatch-shape refactor drops the derived
# set below _C37_MIN_PLAUSIBLE (loud SKIP here) and
# test_c37_live_derivation_pins fails the suite, forcing a deliberate
# re-derivation rather than a silent all-PASS.
_C37_DISPATCH_RE = re.compile(r"args\.check_(\w+)\s+or\s+no_flags")
_C37_MIN_PLAUSIBLE = 10  # 47 dispatch lines measured 2026-07-16; fewer ⇒ parse broken

_C37_FLAG_RE = re.compile(r"--check-([a-z0-9][a-z0-9-]*)")
# Claim anchors, both word orders. Forward: verb … no-flags ("bundled into
# the no-flags default run", "included in the no-flags default run" — the
# #1322 v1 shape). Inverse: no-flags … verb ("no-flags default run, which
# bundles/includes/covers `--check-asks`"; `incl\w*` admits the corpus
# "incl." abbreviation). Gaps exclude '.' '|' and newline (sentence stop /
# table cell) but deliberately ADMIT ';' and '—' (the corpus false claim
# "no-flags default run; all bundled checks must pass (incl. `--check-asks`"
# crosses a ';'). "runs?" is deliberately NOT a verb anchor: "run" is the
# NOUN in "no-flags default run" and anchors nearly every incidental line.
# `bundl` is restricted to the VERB forms (bundles/bundled/bundling): the
# bare NOUN "bundle" anchors the incidental two-command idiom "(no-flags
# bundle) + `--check-asks` + `--check-references` green" (2026-07-16 corpus).
_C37_CLAIM_FWD_RE = re.compile(
    r"(?i)\b(?:includ\w*|incl\.?|bundl(?:es|ed|ing)|part of|folde?d? into)[^.|\n]{0,40}no[- ]flags"
)
_C37_CLAIM_INV_RE = re.compile(
    r"(?i)no[- ]flags[^.|\n]{0,60}\b(?:includ\w*|incl\.?|bundl(?:es|ed|ing)|covers?|carri\w*)"
)
# Destination-skip: a flag token directly preceded by "into" is the BUNDLE
# DESTINATION, not the claim subject — "`--check-script-refs` (also bundled
# into `--check-references` and the no-flags default run)" claims
# script-refs ∈ {references bundle, no-flags run}; it asserts nothing about
# references' own no-flags membership (the reference-check-extension family:
# #714/#753/#739/#802/#1190 all quote this workflow_lint docstring idiom).
_C37_DEST_RE = re.compile(r"(?i)\binto\s*[`'\"]?$")
_C37_NEG_RE = re.compile(
    r"(?i)\bnot\b|\bnever\b|\bno longer\b|\babsent\b|\bexcluded?\b|\boutside\b"
    r"|\bseparate(?:ly)?\b|\bonly\s+(?:in|under|when|via|on)\b|\bruns only\b"
)
# Pin-test arm (#1679): a hit whose flag has NO occurrence in the lint
# source is a PROPOSED new check — unfalsifiable against the dispatch set
# (the existence gate below), but #1385 v1 / #1648 v2 shipped exactly this
# shape with no test named to keep the bundling true after a later
# dispatch refactor (the silent-unbundling class; both were caught only by
# a Phase-2 critic Must-Fix, then corrected in v2/v3 with a
# test_<check>_bundled_in_no_flags pin). The pin arm uses a LOCALIZED
# negation guard (the c41 _C41_NEG_WINDOW precedent): both founding
# incident lines are long mixed-clause Plan-Summary lines carrying an
# incidental negation token ("not allowed here" / "not a comment") FAR
# from the claim, so the falsity arm's whole-line guard silently
# un-triggers them (measured 2026-07-25: whole-line replay reads both
# incidents SKIP; 40-char pre-window replay reads both WARN and their
# corrected v2/v3 PASS via the pin-test satisfier).
_C37_PIN_NEG_WINDOW = 40  # chars of pre-context a negation token guards (c41 parity)
# Satisfier: the plan names a pytest following the house convention
# test_<check>_bundled_in_no_flags (20 live exemplars across
# tests/test_workflow_lint*.py, measured 2026-07-25 — incl. the
# `*_source_pin` suffixed shapes, which the unanchored regex admits).
# RAW-plan scan, fences INCLUDED — test enumerations and pytest commands
# legitimately live in fenced blocks (the c41 satisfier-(a) precedent).
# The literal `test_` prefix + [a-z0-9_] body is anti-paste armor: the
# WARN detail names the convention only in its angle-bracketed template
# form (test_<check>_bundled_in_no_flags), which `<` keeps unmatched.
_C37_PIN_TEST_RE = re.compile(r"test_[a-z0-9_]*bundled_in_no_flags")


def _c37_collect_hits(plan: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(hits, pin_hits)`` flag/line pairs over the plan's non-fenced
    claim lines. ``hits`` is the falsity arm's whole-line-negation-guarded
    set; ``pin_hits`` is the pin arm's localized pre-window-guarded
    superset (#1679). Loop equivalence with the pre-#1679 falsity arm:
    when ``line_negated`` is False, ``window_live`` is necessarily True
    (no negation anywhere on the line implies none in any pre-window), so
    ``hits`` receives exactly what the old whole-line-guarded loop
    collected; when ``line_negated`` is True, ``hits`` receives nothing —
    identical to the old guard. The pin arm only ADDS ``pin_hits`` rows
    for window-live claims."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    hits: list[tuple[str, str]] = []
    pin_hits: list[tuple[str, str]] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        claims = list(_C37_CLAIM_FWD_RE.finditer(line)) + list(_C37_CLAIM_INV_RE.finditer(line))
        if not claims:
            continue
        line_negated = bool(_C37_NEG_RE.search(line))
        window_live = any(
            not _C37_NEG_RE.search(line[max(0, m.start() - _C37_PIN_NEG_WINDOW) : m.start()])
            for m in claims
        )
        if line_negated and not window_live:
            continue
        for m in _C37_FLAG_RE.finditer(line):
            if _C37_DEST_RE.search(line[max(0, m.start() - 12) : m.start()]):
                continue  # "bundled into `--check-X`": X is the destination, not the subject
            if not line_negated:
                hits.append((m.group(1), line))
            if window_live:
                pin_hits.append((m.group(1), line))
    return hits, pin_hits


def _c37_lint_source() -> str | None:
    """``scripts/workflow_lint.py`` source text (same-dir resolution —
    worktree-correct), or ``None`` when absent (``--plan-file`` off-repo).
    Read-only, NO import: source-regex derivation skips the 540 KB /
    ~345 ms module import c34 pays (and main()'s dispatch is procedural —
    there is no module-level constant to import)."""
    if not _C37_LINT_PATH.is_file():
        return None
    return _C37_LINT_PATH.read_text()


def _c37_noflags_dests(src: str) -> frozenset[str] | None:
    """argparse dests dispatched on workflow_lint.py's no-flags default run,
    derived from ``src``. ``None`` => underivable: fewer than
    ``_C37_MIN_PLAUSIBLE`` matches (main()'s dispatch shape changed; the
    live-tree pin test fails the suite in that world — never a spray of
    WARNs on a broken parse). The `no_flags = not (...)` definition block
    alternates `or args.check_*`, never `or no_flags`, so it cannot match;
    the parenthesized `(args.check_X or no_flags) and not
    args.check_references` forms DO match (correctly — they are in the
    no-flags set, since no_flags=True implies check_references=False)."""
    dests = frozenset(_C37_DISPATCH_RE.findall(src))
    return dests if len(dests) >= _C37_MIN_PLAUSIBLE else None


def check_noflags_bundling_claim(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, ``kind: infra|batch``: a plan line asserting
    a ``--check-<flag>`` is bundled into workflow_lint.py's no-flags default
    run must name a flag actually dispatched there (an `args.check_<dest> or
    no_flags` line in main()) — #1322 v1 shipped an acceptance criterion
    claiming the pre-commit-only ``--check-references`` runs on the bare
    invocation, so its gate could never fire as written. Trigger:
    ``--check-<flag>`` + a claim-verb-anchored no-flags assertion on one
    non-fenced line with no negation/scoping token; falsity = derived-set
    non-membership, restricted to flags that EXIST in the workflow_lint
    source. NEVER FAILs (the c15/c34 doctrine: trigger and derivation
    are text heuristics). Deliberately OUT of scope (v1, disclosed): the
    CONVERSE claim class (asserting a flag is NOT in the set when it IS —
    the negation guard drops every negated line unadjudicated), and the
    vocabulary-side false-negative class (a bundling claim phrased without
    a literal "no-flags"/"no flags" token, e.g. "the bare/default
    workflow_lint run").

    Pin-test arm (#1679): a claimed flag with NO ``--check-<flag>``
    occurrence in the lint source is a PROPOSED new check — unfalsifiable
    against the dispatch set at plan time (the pre-#1679 existence gate
    read it "never an offender") — but #1385 v1 / #1648 v2 shipped exactly
    this shape with no test named to keep the claimed bundling true after
    a later dispatch refactor (the silent-unbundling class). Such a claim
    now WARNs unless the plan names a pytest matching ``_C37_PIN_TEST_RE``
    anywhere in the RAW plan (fences included) or declares the standalone
    escape ``N/A — not bundled into no-flags``. The pin arm's claim
    matches are guarded by a LOCALIZED pre-window negation check
    (``_C37_PIN_NEG_WINDOW``, c41 parity) instead of the falsity arm's
    whole-line guard — see the constant comment for the founding-incident
    measurement. The falsity arm's trigger, whole-line guard, and WARN
    detail are byte-preserved, and a falsity WARN takes PRECEDENCE over a
    co-occurring pin-arm miss (one CheckResult per check; the pin WARN
    deterministically resurfaces on the post-fix re-verify — every plan
    revision is re-verified standalone).

    Disclosed pin-arm limitations: (a) the satisfier is PLAN-GLOBAL — a
    plan proposing TWO new checks that names one pin test passes for both
    (the c34 scope-note-(a) precedent); (b) sibling-exemplar quotation —
    a plan quoting a PRIOR check's REAL pin-test name as precedent
    self-satisfies for its own new check (a false negative; the Phase-2
    critics own test-shape adequacy); (c) a deliberately explicit-only
    new check never triggers (it makes no bundling claim), and the
    immediate-negation shape ("is NOT bundled into the no-flags default
    run") stays guarded by design; (d) a benign micro-delta vs the
    pre-#1679 loop on negated-line-only plans: the terminal SKIP detail
    wording differs from the old no-claim SKIP, and the whole-check N/A
    escape is now reachable (PASS instead of SKIP) when only negated-line
    pin residue survives — WARN-free outcomes either way; (e) the
    satisfier verifies the plan SURFACE (a test NAME, not its existence
    or selection) — c41's anchor machinery and the code-reviewer own
    implementation.

    Calibration, falsity arm (2026-07-16, 838 corpus plan versions
    carrying ``--check-``, forced kind=infra): 174 WARN / 115 PASS /
    549 SKIP; the 20 NEWEST infra|batch plans (the forward-looking
    operative set) are 0 WARN; #1322 v1 WARNs while its corrected v2
    SKIPs (positive/negative control pair, likewise #802 v1→v2). The
    verb anchor, the into-destination skip, and the existence gate are
    each grounded on a named corpus false-positive class (see the
    trigger-constant comments above); the residual WARN mass is dominated
    by the endemic genuinely-false "no-flags run bundles/includes
    ``--check-asks``" claim (corrected #1007 v1→v2 — a true positive),
    with ~5% incidental false WARNs on historical long mixed-clause
    lines, acceptable at WARN-only granularity.

    Calibration, pin arm (2026-07-25, landed-function replay over 2,763
    persisted ``tasks/*/*/plans/v*.md`` versions, forced kind=infra):
    +6 new WARNs vs the pre-#1679 check — 3 genuine incident-class
    (#1234 v1 ``--check-executable-git-recipes``, #1520 v1
    ``--check-hub-verify-calls``, #718 v2 ``--check-skill-circuit-breaker``
    — each a bundling claim for a flag with no lint-source occurrence and
    no pin test named), 2 placeholder false positives (#1176 v1/v2's
    illustrative ``--check-x`` — the standalone escape is the designed
    remedy), 1 borderline convention-prose (#968 v1's naming-convention
    discussion of a suggested-then-rejected flag name). Full-corpus status
    counts under the extended check: 206 WARN / 163 PASS / 2,394 SKIP
    (falsity-arm mass unchanged in kind from the 2026-07-16 calibration).

    Founding controls (plan-time source simulation — the landed flag
    stripped from a copied lint source; reproduced through the landed
    function 2026-07-25): #1385 v1 WARN / #1648 v2 WARN; corrected
    #1385 v2 PASS / #1648 v3 PASS via the pin-test satisfier."""
    cid, name = (
        "c37_noflags_bundling_claim",
        "no-flags bundling claim matches workflow_lint dispatch",
    )
    if kind not in ("infra", "batch"):
        return _skip(
            cid,
            name,
            "kind-exempt: --check-* bundling claims are an infra|batch (workflow-fix) shape",
        )
    # hits: falsity arm (whole-line negation guard); pin_hits: pin arm
    # (localized pre-window guard, #1679) — see _c37_collect_hits for the
    # loop-equivalence argument.
    hits, pin_hits = _c37_collect_hits(plan)
    if not hits and not pin_hits:
        return _skip(cid, name, "no --check-* no-flags bundling claim on a non-fenced line")
    if _standalone_na_declared(plan, r"no no[- ]flags bundling claim"):
        return _pass(cid, name, "explicit N/A declared (incidental --check/no-flags vocabulary)")
    src = _c37_lint_source()
    dests = None if src is None else _c37_noflags_dests(src)
    if dests is None:
        return _skip(
            cid,
            name,
            "no-flags dispatch set underivable — scripts/workflow_lint.py absent (--plan-file "
            "off-repo) or _C37_DISPATCH_RE matched below the plausibility floor "
            f"({_C37_MIN_PLAUSIBLE}); membership cannot be adjudicated",
        )
    offenders = [
        (flag, line)
        for flag, line in hits
        # Existence gate: a flag absent from the lint source outright is a
        # PROPOSED new check — the falsity arm cannot adjudicate it; the
        # pin arm below owns it (#1679).
        if f"--check-{flag}" in src and flag.replace("-", "_") not in dests
    ]
    if offenders:
        # Falsity WARN, byte-preserved from pre-#1679. Takes precedence over
        # a co-occurring pin-arm miss: one CheckResult per check; the pin
        # WARN resurfaces on the post-fix re-verify.
        flag, line = offenders[0]
        more = f" (+{len(offenders) - 1} more)" if len(offenders) > 1 else ""
        return _warn(
            cid,
            name,
            f"`--check-{flag}` is NOT in workflow_lint.py main()'s no-flags dispatch set "
            f"(no `args.check_{flag.replace('-', '_')} or no_flags` line) — the plan's bundling "
            "claim is false; the flag runs only when passed explicitly (#1322 v1 shipped exactly "
            "this claim for a pre-commit-only flag). "
            f"Offending line: {line.strip()[:100]!r}{more}. "
            "Remedies: correct the claim (name the explicit invocation that runs the flag), or "
            "declare `N/A — no no-flags bundling claim` on its own line, unwrapped "
            "(no backticks/quotes)",
        )
    # Pin-test arm (#1679): a proposed new check claiming no-flags bundling
    # must name its test_<check>_bundled_in_no_flags pin test.
    proposed = [(flag, line) for flag, line in pin_hits if f"--check-{flag}" not in src]
    if proposed:
        if _C37_PIN_TEST_RE.search(plan):
            return _pass(
                cid,
                name,
                f"{len(proposed)} proposed-new-check bundling claim(s) — the plan names a "
                "test_*bundled_in_no_flags pin test",
            )
        if _standalone_na_declared(plan, r"not bundled into no[- ]flags"):
            return _pass(
                cid,
                name,
                "explicit N/A declared (proposed check deliberately not run on the bare "
                "invocation, or the claim vocabulary is incidental)",
            )
        flag, line = proposed[0]
        more = f" (+{len(proposed) - 1} more)" if len(proposed) > 1 else ""
        # Paste-safety: the quoted excerpt is SANITIZED — case-insensitively,
        # since the claim regexes are (?i) — so a pasted WARN detail can
        # never carry a window-live claim match: the claim NOUN inside the
        # quote is neutralized deterministically (`no-flags`/`no flags` →
        # `no_flags`, which `no[- ]flags` cannot match), independent of
        # where the claim sits in the original line.
        excerpt = re.sub(r"(?i)no[- ]flags", "no_flags", line.strip()[:100])
        return _warn(
            cid,
            name,
            f"`--check-{flag}` is a PROPOSED new check — NOT yet in "
            "scripts/workflow_lint.py — so its membership is unfalsifiable at plan time, "
            "and the plan names no pin test keeping the claimed wiring true after a later "
            "dispatch refactor (#1385 v1 / #1648 v2 shipped exactly this silent-unbundling "
            f"shape). Offending line (claim noun neutralized): {excerpt!r}{more}. Remedies: "
            "name the pin test in the plan's test enumeration — house convention "
            "test_<check>_bundled_in_no_flags in tests/test_workflow_lint.py. Prefer the "
            "in-process drifted-tree shape or the source-pin shape. Or declare "
            "`N/A — not bundled into no-flags` on its own line, unwrapped "
            "(no backticks/quotes)",
        )
    if hits:
        return _pass(
            cid,
            name,
            f"{len(hits)} no-flags bundling claim(s) — every named flag is in the live "
            "dispatch set",
        )
    # hits empty here: every claim line was whole-line negated and no
    # proposed-new-check residue survived (every pin_hits flag exists in
    # src) — the falsity arm deliberately does not adjudicate negated
    # lines, so this is the same outcome class the pre-#1679 loop produced
    # (SKIP). Preserves test_c37_pasted_warn_detail_does_not_retrigger_or_
    # satisfy UNMODIFIED: a pasted falsity WARN detail is line-negated with
    # window-live in-src claim residue, which must stay SKIP, never PASS.
    return _skip(
        cid,
        name,
        "claim line(s) negation-guarded and no proposed-new-check residue — nothing adjudicable",
    )


# ─── Check 38 — exit-0 criterion on repo-wide lint/suite (conditional) ─────

# Assertion tokens: the explicit exit-status idiom family plus "green"
# (#584 v1:182 "# full suite green"). `pass`/`passes` is a DISCLOSED v1
# under-trigger: it is endemic in criterion boilerplate (plan-time corpus
# measurement: ~1,810 workflow_lint-arm candidate lines with pass/green
# vs 381 exit-0-only), and its dominant shape ("no-flags run passes
# (includes --check-X)") is c37's subject; widen only after burn-in.
_C38_ASSERT_RE = re.compile(
    r"(?i)\bexit(?:s|ed)?\s+(?:code\s+|status\s+|with\s+)?0\b"
    r"|\bexit[- ]?code\s*(?:==?|:)\s*0\b"
    r"|\brc\s*(?:==?|:)\s*0\b"
    r"|\breturn\s*code\s*(?:==?|:)?\s*0\b|\breturncode\s*(?:==?|:)?\s*0\b"
    r"|\bgreen\b"
)
# Negation guard: a line DESCRIBING the hazard ("full no-flags run is NOT
# a pass criterion", "exit 0 is unattainable") is never adjudicated —
# the #1365 v2:101 corrected prose is the founding negative control.
_C38_NEG_RE = re.compile(r"(?i)\bnot\b|\bnever\b|\bcannot\b|\bunattainable\b|\bunsatisfiable\b")
# Satisfiers (same line only): a NAMED baseline mechanism or scoping
# prose. Bare "no NEW failures" deliberately does NOT satisfy — the
# motivating #1365 v1:95 line carried exactly that phrase and was still
# critic-bounced (a bare command performs no baseline subtraction).
_C38_SATISFIER_RE = re.compile(
    r"(?i)\bbaseline\b|\bstep\s*9c\b|\bstep\s*10d\b|baseline[- ]subtract"
    r"|\bvs\.?\s+(?:origin/)?main\b|\btouched files?\b|\bchanged files?\b|\bscoped\b"
)
# Arg-tail terminators: backtick, comment '#', '&&'/'|', close-paren, EOL.
_C38_TAIL_SPLIT_RE = re.compile(r"[`#&|)\n]")
# (?<!test_) keeps `tests/test_workflow_lint.py` mentions out of arm A.
_C38_LINT_OCC_RE = re.compile(r"(?<!test_)workflow_lint(?:\.py)?")
_C38_NOFLAGS_RUN_RE = re.compile(r"(?i)\bno[- ]flags(?:\s+default)?\s+run\b")
_C38_PYTEST_OCC_RE = re.compile(r"\bpytest\b")
_C38_PYTEST_SCOPED_RE = re.compile(r"::|\btests?/\S*\.py\b|(?:^|\s)-k\s")
_C38_RUFF_OCC_RE = re.compile(r"\bruff\s+(?:check|format)\b")
_C38_FLAG_TOKEN_RE = re.compile(r"^--?[\w-]+$")


def _c38_repo_wide_cmd(line: str) -> str | None:
    """Label of the first REPO-WIDE lint/suite invocation on ``line``, or
    None. Scoped forms do not count: a ``--check-`` flag in the
    workflow_lint arg tail; a ``.py`` path / ``::`` node id / ``-k``
    filter in the pytest tail; any non-``.`` path token in the ruff tail.
    Arm B (the "no-flags default run" phrase) is suppressed when the line
    already carries a SCOPED workflow_lint invocation (commentary shape).
    Pytest arm: the scoped test additionally re-scans the rest of the line
    past the arg-tail terminator, so a scoping ``::`` node id / ``tests?/…py``
    path / ``-k`` filter written inside backticks after the word "pytest"
    does NOT read as unscoped."""
    saw_scoped_lint = False
    for m in _C38_LINT_OCC_RE.finditer(line):
        tail = _C38_TAIL_SPLIT_RE.split(line[m.end() :], 1)[0]
        if "--check-" in tail:
            saw_scoped_lint = True
        else:
            return "workflow_lint.py (no --check- scoping)"
    if not saw_scoped_lint and _C38_NOFLAGS_RUN_RE.search(line):
        return "the workflow_lint no-flags default run"
    for m in _C38_PYTEST_OCC_RE.finditer(line):
        rest = line[m.end() :]
        tail = _C38_TAIL_SPLIT_RE.split(rest, 1)[0]
        # A scoping token (`::` node id, `tests?/…py` path, or `-k` filter)
        # may live PAST a backtick that terminates the arg tail — the
        # ordinary prose shape "Concrete pytest node id: `tests/x.py::y`".
        # Widen to the rest-of-line so the tail split does not hide it.
        if not _C38_PYTEST_SCOPED_RE.search(tail) and not _C38_PYTEST_SCOPED_RE.search(rest):
            return "pytest (no path scope)"
    for m in _C38_RUFF_OCC_RE.finditer(line):
        tail = _C38_TAIL_SPLIT_RE.split(line[m.end() :], 1)[0]
        tokens = [t for t in tail.split() if not _C38_FLAG_TOKEN_RE.match(t)]
        if all(t == "." for t in tokens):
            return "ruff check/format (repo-wide)"
    return None


def check_exit0_repo_wide_baseline(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, ALL kinds: a line asserting exit-0/green on
    a REPO-WIDE lint/suite command must name a plan-time baseline or
    scoping ON THAT LINE — #1365 v1 §6 asserted
    `workflow_lint.py # exit 0 — no NEW failures` while the no-flags lint
    is pre-existing-red on origin/main (jointly unsatisfiable; caught by
    the statistics critic in round 1; prior instance #584 v1
    `pytest tests/ -q # full suite green`). SAME-LINE co-occurrence only:
    a ±3-line window would have false-WARNed the CORRECTED #1365 v2 via
    its :98 scoped criterion pairing with the :101 no-flags prose (the
    c37 same-line-radius doctrine). RAW scan incl. fences (c11 precedent:
    the incident line sits inside a ```bash fence). ALL kinds (c21
    precedent): both incidents were kind: infra, but the idiom rides any
    kind's §6/§10. NEVER FAILs: trigger, arms, and satisfiers are line
    heuristics; the Phase 1.5/2 reviewers adjudicate; a genuinely-green
    repo-wide command with baseline language stays PASS by construction.
    Disclosed under-triggers (v1): bare `pass`/`passes`/`passed`
    assertion vocabulary (endemic; partially c37's subject); an
    assertion sentence separated from its fenced command by ≥1 line;
    "full suite" with no literal pytest token. Disclosed over-trigger
    residual (narrowed 2026-07-27 for #1729): a bare unscoped `pytest`
    assertion whose only nearby scoping token is on an unrelated line.
    (The backticked-node-id class — a scoping ``::`` / ``tests?/…py`` /
    ``-k`` token past the arg-tail terminator on the SAME line — is now
    handled by the pytest arm's rest-of-line rescan.) Deliberate reading (disclosed,
    fact-checker-confirmed correct): an EMPTY arg tail — bare
    `ruff check` / bare `pytest` plus an assertion word — classifies
    REPO-WIDE and WARNs (both default to cwd/repo-wide; `all([]) is
    True` is intended semantics, not an accident). Additional disclosed
    over-trigger residual (measured): a `workflow_lint.py` PATH named as
    another tool's argument on an assertion line (`ruff check ...
    workflow_lint.py ... → exit 0`) reads as arm A — 1/8 operative-set
    WARNs (#1381 v2). Calibration (2026-07-16, 2117 corpus plan
    versions, forced kind=infra): corpus-wide 486 WARN / 102 PASS /
    1529 SKIP (historical mass, never re-verified — the c37 reporting
    convention); the 25 NEWEST infra|batch plan versions by git
    last-commit date (task #1387's own self-referential plan files
    excluded): 8 WARN / 17 SKIP, hand-classified 7 incident-class
    (genuine baseline-less repo-wide exit-0/green gates: #1389 v1+v2,
    #1385 v1+v2, #1381 v1, #1399 v1+v2) / 0 rephrase-class / 1 FP
    (#1381 v2, the ruff-path residual above) — within the calibration
    gate (WARNs <= 8, FPs <= 2). Positive controls #1365 v1:95 +
    #584 v1:182 WARN; corrected #1365 v2 is clean (SKIP); the GOOD_PLAN
    fixture SKIPs."""
    cid, name = (
        "c38_exit0_repo_wide_baseline",
        "exit-0 criterion on a repo-wide command names baseline/scoping",
    )
    del kind  # all kinds — trigger precision carries the false-positive discipline
    offenders: list[tuple[str, str]] = []
    triggered = False
    for line in plan.splitlines():
        if not _C38_ASSERT_RE.search(line) or _C38_NEG_RE.search(line):
            continue
        cmd = _c38_repo_wide_cmd(line)
        if cmd is None:
            continue
        triggered = True
        if not _C38_SATISFIER_RE.search(line):
            offenders.append((cmd, line))
    if not triggered:
        return _skip(cid, name, "no exit-0/green assertion on a repo-wide lint/suite command")
    if _standalone_na_declared(plan, r"no exit[- ]0 acceptance criterion"):
        return _pass(
            cid,
            name,
            "explicit N/A declared (matched text quotes an incident / is not this plan's own gate)",
        )
    if not offenders:
        return _pass(
            cid,
            name,
            "every repo-wide exit-0/green criterion names a baseline or scoping on its line",
        )
    cmd, line = offenders[0]
    more = f" (+{len(offenders) - 1} more)" if len(offenders) > 1 else ""
    return _warn(
        cid,
        name,
        f"exit-0/green asserted on {cmd} with no plan-time baseline or scoping named on the "
        f"line — the no-flags lint / full suite is pre-existing-red-exposed on origin/main, so "
        f"an unconditional exit-0 gate is jointly unsatisfiable (#1365 v1 / #584 shape; the "
        f"binding repo-wide gates are Step 10d's baseline-subtracted lint gate and the step9c "
        f"baseline compare). Offending line: {line.strip()[:100]!r}{more}. Remedies: state "
        "'no NEW failures vs the plan-time baseline' (or step9c / Step 10d) on the criterion "
        "line, scope the invocation (--check-<x> / explicit paths), or declare "
        "`N/A — no exit-0 acceptance criterion` on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 39 — off-pod phase declaration (reads + outputs) ────────────────

# Calibration (DEVELOPMENT-SET numbers, fitted IN-SAMPLE — the tokens were
# tuned on the same persisted-plan corpus they were measured on; ANY future
# c39-regex change re-runs the corpus scan and records the realized numbers
# here — the c33/c27/c32 gate precedent). Re-scan 2026-07-29 (#1796,
# implementation-time, AS-SHIPPED regex) over 3,004 persisted plan-versions
# (tasks/*/*/plans/v*.md, 1,264 distinct issues), mirroring the check's
# gating exactly (stripped-prose per-line trigger; kind==experiment; raw
# `off_pod_phases:` satisfier; standalone `N/A — no off-pod phase` escape).
# Inverse-direction tokens KEPT: `vm-produced` — 5 pv triggered (issues
# #1782/#1796 only, both kind:infra workflow-fix plans discussing this very
# seam ⇒ kind-exempt), 0 would-WARN; ZERO in-prose non-compliant hits exist
# in the corpus (#1773's own `VM-produced` prose lives INSIDE its fenced
# off_pod_phases: block in already-compliant plans, invisible to the
# stripped-prose trigger by design), so no in-corpus positive control
# exists and the token is FORWARD-LOOKING per the c38 positive/negative-
# control convention — the pinned WARN test is the synthetic positive
# control. `produced on the vm` — 2 pv triggered (#548 v1, #778 v5), both
# GENUINE cross-phase-read prose ("The off-pod primary read ... is produced
# on the VM AFTER pod termination") and both ALREADY would-WARN under the
# pre-#1796 off-pod/vm-side regex ⇒ 0 NEW would-WARNs, empty nuisance
# class. Tokens DROPPED: `vm-built` / `vm-generated` — 0 corpus hits;
# secondary variants gated on demonstrated recall (plan #1796 §3),
# speculative widening declined. `git-clone lane` — 13 pv / 5 issues,
# 7 would-WARN, nuisance class irreducible: artifact-reuse fitness-check
# boilerplate (#1090 v5 "fetchability check (h) passes by construction
# (git-clone lane, ...)") and COMPLIANT staging prose (#920 v1-v3
# "committed to the issue branch before dispatch so the git-clone lane
# stages it" — 3 NEW nuisance would-WARNs); dropped per the plan's
# drop-by-default posture. Recent-era (issue >= 1000) NEW nuisance WARNs
# from the SHIPPED tokens: 0. AS-SHIPPED delta confirmation: the widened
# regex changes ZERO plan-version verdicts across the full corpus (0 newly
# triggered pv, 0 newly would-WARN pv vs the pre-#1796 regex — every
# shipped-token hit lives in a plan that already triggers on off-pod /
# vm-side vocabulary elsewhere), i.e. purely forward-looking widening.
_C39_TRIGGER_RE = re.compile(r"(?i)\boff-pod\b|\bvm-side\b|\bvm-produced\b|\bproduced on the vm\b")


def check_off_pod_phase_declaration(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, experiment-only: a plan whose non-fenced
    prose names an off-pod / VM-side phase must either carry the fenced
    ``off_pod_phases:`` declaration block (planner-section-reference.md § 9
    — since #1782 a direction-agnostic CROSS-PHASE READS rule: per phase
    that reads another phase's outputs, runs_on + reads[] with producing
    phase + permanent source the CONSUMING machine can fetch + outputs[]
    with dest; the block upload-verifier Steps 2.7/2.8
    consume, #1535) or declare the standalone escape
    ``N/A — no off-pod phase``. Mechanizes the #1526 gotchas.md
    cross-machine upload-set bullet at plan time (incident #1482: an
    off-pod judge died
    at VM launch on pod-only scratch never in the upload set; incident
    #1426: a planned VM-side phase FAILed the verifier r1 by construction;
    incident #1773 — the inverse direction: a GCE phase crashed loading
    VM-produced inputs never uploaded/staged). KNOWN MECHANICAL RESIDUAL
    (narrowed by #1796): the trigger now ALSO fires on the corpus-
    calibrated inverse-direction tokens (`vm-produced` /
    `produced on the vm` — a pod/GCE/SLURM phase consuming VM-produced
    inputs); inverse-direction prose using OTHER vocabulary (calibration
    dropped `git-clone lane` as irreducibly noisy and `vm-built` /
    `vm-generated` as zero-recall — see the calibration comment above
    _C39_TRIGGER_RE) remains enforced by planner §9 + critic Methodology
    item 10 only.
    NEVER FAILs — the trigger is a vocabulary heuristic (the c31/c34
    family), and legacy plans must not bounce retroactively. kind-exempt
    outside experiment: infra/batch/analysis/survey plans rarely dispatch
    pods, and infra workflow-fix plans (this check's own lineage included)
    legitimately discuss "off-pod" without having phases. Trigger scans
    STRIPPED prose (fenced blocks masked); the block satisfier scans RAW
    text because the slot is fenced YAML by design (the c30 convention)."""
    cid, name = "c39_off_pod_phase_declaration", "off-pod phase declaration"
    if kind != "experiment":
        return _skip(
            cid,
            name,
            "kind-exempt: off-pod phase declaration is an experiment-plan "
            "(pod + off-pod phase) shape",
        )
    text = strip_fences(plan)
    trigger_lines = [ln for ln in text.splitlines() if _C39_TRIGGER_RE.search(ln)]
    if not trigger_lines:
        return _skip(cid, name, "no off-pod / vm-side vocabulary detected")
    if "off_pod_phases:" in plan:  # raw plan: the slot is fenced YAML by design
        return _pass(cid, name, "fenced off_pod_phases: declaration block present")
    if _standalone_na_declared(plan, r"no off-pod phase\b"):
        return _pass(cid, name, "explicit N/A declared (no off-pod phase)")
    shown = "; ".join(ln.strip()[:70] for ln in trigger_lines[:3])
    return _warn(
        cid,
        name,
        f"plan prose names an off-pod / VM-side phase ({shown!r}) but carries no fenced "
        "`off_pod_phases:` block — without the declaration the phase's READS are not "
        "plan-named (upload-verifier Step 2.8 cannot gate them at the cheap-fix window "
        "before the pod dies; the #1482 class) and its OUTPUTS false-FAIL the pod-side "
        "Step 2.7 gate by construction (#1426). The rule is direction-agnostic (#1773): "
        "declare EVERY dispatched phase that reads another phase's outputs, incl. a "
        "pod/GCE/SLURM phase consuming VM-produced inputs. Add the fenced "
        "`off_pod_phases:` block "
        "(template + worked examples: planner-section-reference.md § 9 (off_pod_phases)), "
        "or declare `N/A — no off-pod phase` on its own line, unwrapped (no "
        "backticks/quotes), if the vocabulary is incidental and no such phase exists",
    )


# ─── Check 40 — header version label vs persisted filename (outside CHECKS) ─

_C40_FILENAME_RE = re.compile(r"v(\d+)\.md")  # fullmatch on plan_path.resolve().name
_C40_HEADER_RE = re.compile(r"(?i)^plan\s+v(\d+)\b")  # on the first heading's TEXT


def check_header_version_vs_filename(plan: str, *, plan_path: Path) -> CheckResult:
    """WARN when the plan's first heading self-declares ``Plan v<X>`` and the
    persisted filename is ``v{K}.md`` with X != K (#1482: '# Plan v4' rode
    v5.md + v6.md unnoticed). Version-neutral titles PASS (the sanctioned
    escape); a non-``v{K}.md`` filename SKIPs (nothing to compare). The
    filename is read via ``plan_path.resolve()`` so a ``plans/plan.md``
    symlink invocation compares against the real ``v{K}.md`` target.
    WARN-only, all kinds."""
    cid = "c40_header_version_vs_filename"
    name = "header self-declared version matches persisted filename"
    m_file = _C40_FILENAME_RE.fullmatch(plan_path.resolve().name)
    if not m_file:
        return _skip(cid, name, "filename carries no v{K}.md version (draft / standalone plan)")
    _, body = split_frontmatter(plan)
    heads = _headings(body)
    if not heads:
        return _skip(cid, name, "no headings in plan")
    m_head = _C40_HEADER_RE.match(heads[0].text)
    if not m_head:
        return _pass(cid, name, "header is version-neutral — no self-declared version")
    x, k = int(m_head.group(1)), int(m_file.group(1))
    if x == k:
        return _pass(cid, name, f"header v{x} matches persisted {plan_path.resolve().name}")
    return _warn(
        cid,
        name,
        f"header self-declares v{x} but the persisted file is {plan_path.resolve().name} — "
        f"stale version label (#1482: '# Plan v4' rode v5+v6 unnoticed); retitle the header "
        f"to v{k} or make it version-neutral ('# Plan (amendment) — …' / "
        f"'# Plan — Issue #<N>: …')",
    )


# ─── Check 41 — regression-anchor test executed or gate-selected ───────────

# Trigger: a NON-FENCED line carrying BOTH a test-file path token AND
# regression-anchor / gate-selection vocabulary — same-line co-occurrence
# only (the c37/c38 same-line-radius doctrine; a ±N window would false-fire
# on §6 test lists near unrelated gate prose) — with no negation token.
#
# Test-path token: optional tests/ prefix + optional subdirs; ``::node`` ids
# are naturally excluded from the capture (the match ends at ``.py``). A bare
# ``test_x.py`` normalizes to ``tests/test_x.py`` — a disclosed convenience
# (plans conventionally write full repo-relative paths; the bare form is
# assumed to live under tests/).
_C41_TESTPATH_RE = re.compile(r"\b((?:tests/(?:[\w-]+/)*)?test_[\w-]+\.py)\b")

# Anchor / gate-selection vocabulary. Two claim classes, one trigger set:
#  (i) gate-selection claims — "sits on the Step-9c mapped-scan path",
#      "Step 9c selects/runs it", "auto-selected", "the gate will run it",
#      "selected/picked up by the (9c) gate";
#  (ii) anchor claims — "regression anchor", "must stay green",
#      "stays/remains green".
_C41_ANCHOR_VOCAB_RE = re.compile(
    r"(?i)\bregression[- ]anchor"
    r"|\bmust\s+(?:stay|remain)\s+green\b|\b(?:stays?|remains?)\s+green\b"
    r"|\bmapped[- ]scan\b"
    r"|\bstep[- ]?9c\b"
    r"|\bauto[- ]select"
    r"|\bgate\s+(?:will\s+)?(?:runs?|selects?|covers?)\b"
    r"|\b(?:selected|picked\s+up)\s+by\s+the\s+(?:9c\s+)?gate\b"
)

# Negation guard (the c37 ``_C37_NEG_RE`` convention, LOCALIZED): the
# CORRECTED plan shape ("test X is NOT auto-selected — the one-hop import
# map misses it — so §6 runs it explicitly") must not trigger. The guard is
# per-VOCAB-HIT, not per-line: a negation token within the
# ``_C41_NEG_WINDOW`` chars BEFORE a vocabulary match guards THAT match
# only — calibration showed the founding incident line itself (#1536 v2
# L112) opens with an unrelated "N/A — not an experiment" clause ~400 chars
# before its anchor vocabulary, so a whole-line negation guard silently
# un-triggers the very shape the check exists to catch.
_C41_NEG_RE = re.compile(
    r"(?i)\bnot\b|\bnever\b|\bno longer\b|\bcannot\b|\bisn'?t\b|\bwon'?t\b|\bmisse[sd]\b"
)
_C41_NEG_WINDOW = 40  # chars of pre-context a negation token guards


def _c41_line_triggers(line: str) -> bool:
    """True when ``line`` carries at least one anchor/gate vocabulary hit
    that is not LOCALLY negated (no negation token in the ``_C41_NEG_WINDOW``
    chars before the hit). Post-hit negation ("selected? it is not") is a
    disclosed non-guard — rare, and the fail direction is a WARN the N/A
    escape remedies, not a miss."""
    for m in _C41_ANCHOR_VOCAB_RE.finditer(line):
        window = line[max(0, m.start() - _C41_NEG_WINDOW) : m.start()]
        if not _C41_NEG_RE.search(window):
            return True
    return False


def _c41_anchors(plan: str) -> set[str]:
    """Normalized anchor test paths from non-fenced, locally-un-negated
    trigger lines (a bare ``test_x.py`` normalizes to ``tests/test_x.py``;
    ``::node`` ids never enter the capture — the match ends at ``.py``)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    anchors: set[str] = set()
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or not _c41_line_triggers(line):
            continue
        for m in _C41_TESTPATH_RE.finditer(line):
            path = m.group(1)
            if not path.startswith("tests/"):
                path = f"tests/{path}"
            anchors.add(path)
    return anchors


# Touched-file derivation: RAW scan (fences INCLUDED — scope lists and
# commands legitimately live in fences/tables; the c11 raw-scan doctrine).
# tests/ paths are deliberately EXCLUDED: every named anchor is itself a
# mentioned tests/ path, so admitting tests/ would make every anchor
# self-satisfy via the selector's touched-test arm. Over-inclusion is the
# SAFE polarity here (more touched files -> more selection -> fewer WARNs;
# a missed WARN is the acceptable fail-open direction).
_C41_TOUCHED_RE = re.compile(
    r"\b(scripts/[\w./-]+\.py"
    r"|src/explore_persona_space/[\w./-]+\.py"
    r"|\.claude/rules/[\w-]+\.md)\b"  # rules-pin arm inputs (#1496)
)

_C41_SELECTOR_PATH = Path(__file__).resolve().parent / "select_step9c_tests.py"  # tests monkeypatch
_C41_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c34 pattern)
_c41_selector_cache: list = []  # [module] once loaded; [None] when the file is absent


def _c41_selector():
    """Lazily path-load ``scripts/select_step9c_tests.py`` (stdlib-only
    module-level imports, ~ms; NOT registered in ``sys.modules`` — the test
    suite loads its own instance and the two must not clobber each other).
    ``None`` => file absent (``--plan-file`` off-repo) => the caller SKIPs
    loudly (the c37 "membership cannot be adjudicated" doctrine). A broken
    selector file raises out of ``exec_module`` — the caller's
    ``except Exception`` degrades that to a loud SKIP naming the breakage."""
    if not _c41_selector_cache:
        mod = None
        if _C41_SELECTOR_PATH.is_file():
            spec = importlib.util.spec_from_file_location(
                "_c41_step9c_selector", _C41_SELECTOR_PATH
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # may raise; caller catches -> loud SKIP
        _c41_selector_cache.append(mod)
    return _c41_selector_cache[0]


def check_regression_anchor_executed(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, ``kind: infra|batch``: a plan-named
    regression-anchor / "the Step-9c gate will run it" test must be either
    explicitly run by a pytest command in the plan, branch-new (absent from
    disk — forward-looking, gate-covered by the touched-test arm), or
    actually returned by the REAL Step-9c selection
    (``select_step9c_tests.select_tests_with_reasons``) over the plan's
    declared touched files. #1536 plan v2 claimed two ``test_issue906_*``
    files "sit on the Step-9c mapped-scan path for sft.py edits" — measured
    false: ``--map-files`` on sft.py is empty (pre-#1573; the mapping mode now
    carries the src/scripts dependency arms) and the full selection returns
    56 tests without either anchor (the import map is one-hop; a test
    importing a helper that imports the touched module is never selected).
    NEVER FAILs (task-body constraint: an anchor claim can be legitimately
    satisfied by prose the parser cannot see; fail-open).

    Satisfiers per anchor, cheapest first: (a) a ``pytest`` command naming
    the anchor's basename anywhere in the RAW plan (fences included — that
    is where commands live); (b) the existence gate — an anchor absent from
    disk is a PLAN-NEW test, never an offender (the c37 existence-gate
    doctrine transposed; Step 9c's touched-test arm selects any branch-new
    ``tests/**/test_*.py`` in the diff); (c) ONE oracle call to the full
    Step-9c selection, which covers stem-map, glob-scan (the ``--map-files``
    letter), import-map (the Goal's direct-import parenthetical), literal-
    path, rules-pin, and present-on-disk ``WORKFLOW_INVARIANT`` membership.
    Selector absent or raising -> loud SKIP naming the cause (never WARN,
    never silent PASS): verify_plan gates EVERY plan at Phase 1.5.0, so a
    selector regression must not block fleet-wide planning.

    Disclosed v1 limitations (fail-SAFE polarities; the Phase-2 critics
    remain the backstop): (1) under-trigger — anchor vocabulary on the line
    ABOVE its test list ("These are regression anchors:\\n- tests/test_a.py")
    does not trigger; widen to a ±1 window only after burn-in. (2)
    modified-existing-test false-WARN — an anchor that exists on disk but
    becomes selected only via a plan-introduced import edit to an EXISTING
    test will WARN despite being gate-covered post-merge (the existence gate
    covers only ABSENT files); the explicit-pytest-command remedy resolves
    it. (3) ``_C41_TOUCHED_RE`` omits ``.claude/skills/**`` + ``CLAUDE.md``
    (they map to no selection arm today except invariants, which need no
    touched file); if calibration ever shows a false-WARN class there, widen
    the REGEX, not the vocabulary. (4) touched-set over-inclusion (sibling-
    issue paths cited in prose count as touched) -> missed WARN, accepted
    fail-open residual. Incident-citation prose residual: a plan QUOTING
    this incident ("#1536 claimed tests/test_issue906_... sat on the
    mapped-scan path") triggers unless negated; the standalone
    ``N/A — no regression anchors`` escape is the documented remedy (the
    c36/c38 incident-quoting residual class).

    Calibration (2026-07-19, 2495 corpus plan versions, forced kind=infra —
    per-check replay of ``check_regression_anchor_executed`` over
    ``tasks/*/*/plans/v*.md`` at the main root): 45 WARN / 588 PASS /
    1860 SKIP / 0 FAIL. Positive control: #1536 v2 WARNs naming
    ``test_issue906_tiny_real_e2e.py`` ONLY
    (``test_issue906_marker_mix_budget.py`` is (a)-satisfied by that plan's
    line-128 ``uv run pytest`` command); the localized negation guard is what
    keeps that control alive — v2's L112 opens "N/A — not an experiment"
    ~400 chars before its anchor vocabulary, so a whole-line guard would
    silently un-trigger the founding incident. Negative control: the
    corrected shape ("NOT auto-selected ... §6 runs it explicitly") reads
    SKIP. The 20 NEWEST plan versions by ACTUAL frontmatter kind
    (infra|batch — #1542..#1551 era): trigger FIRED on 16 of 20 (under-fire
    visibility), 2 WARN / 14 PASS / 4 SKIP; both WARNs hand-classify as
    disclosed-residual false positives, meeting the <=2 FP gate exactly —
    (i) #1551 v1, incident-citation prose (this task's own draft quoting the
    #1536 anchors beside gate vocabulary; its v3 PASSes via the standalone
    escape, the documented remedy), FP-class tag: incident-citation;
    (ii) #1542 v1, coverage-analysis prose (test names on a line arguing
    they DON'T cover the edit, ending "these run as the mapped-scan gate" —
    the ``mapped[- ]scan`` token is the broad one), FP-class tag:
    check-discussing prose. The residual WARN mass (45) is dominated by the
    genuine recurrence class — recent infra plans naming a pinning test as
    "must stay green" / gate-selected where the selection demonstrably does
    not pick it up (e.g. #1449, #1495, #1530) — the exact claim shape this
    check exists to adjudicate."""
    cid, name = (
        "c41_regression_anchor_executed",
        "regression-anchor test executed or gate-selected",
    )
    if kind not in ("infra", "batch"):
        return _skip(
            cid,
            name,
            "kind-exempt: the lean-on-the-gate regression-anchor claim is an infra|batch "
            "(workflow-fix) shape",
        )
    anchors = _c41_anchors(plan)
    if not anchors:
        return _skip(
            cid,
            name,
            "no regression-anchor / gate-selection claim naming a test file on a non-fenced line",
        )
    if _standalone_na_declared(plan, r"no regression anchors?"):
        return _pass(cid, name, "explicit N/A declared (no regression anchors)")
    touched = sorted(set(_C41_TOUCHED_RE.findall(plan)))[:40]  # dedupe + hard bound
    survivors: list[str] = []
    for anchor in sorted(anchors):
        basename = anchor.rsplit("/", 1)[-1]
        # (a) explicit pytest command naming the anchor, anywhere in the RAW plan.
        if re.search(r"pytest[^\n]*" + re.escape(basename), plan):
            continue
        # (b) existence gate: absent from disk => plan-new, forward-looking.
        if not (_C41_REPO_ROOT / anchor).exists():
            continue
        survivors.append(anchor)
    if not survivors:
        return _pass(
            cid,
            name,
            f"{len(anchors)} anchor(s) — each explicitly run by a pytest command or "
            "branch-new (existence gate)",
        )
    # (c) the FULL Step-9c selection oracle over the plan's declared touched files.
    try:
        mod = _c41_selector()
        if mod is None:
            return _skip(
                cid,
                name,
                "Step-9c selection surface unavailable — scripts/select_step9c_tests.py "
                "absent (--plan-file off-repo); anchors cannot be adjudicated",
            )
        tests, _, _ = mod.select_tests_with_reasons(touched, _C41_REPO_ROOT)
    except Exception as exc:  # loud SKIP, never a crash / silent PASS
        return _skip(
            cid,
            name,
            f"Step-9c selection oracle failed ({exc!r}) — anchors cannot be adjudicated",
        )
    selected = set(tests)
    offenders = [a for a in survivors if a not in selected]
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(anchors)} anchor(s) — each explicitly run, branch-new, or "
            f"Step-9c-selected from {len(touched)} touched file(s)",
        )
    named = ", ".join(f"`{a}`" for a in offenders[:6])
    extra = f" (+{len(offenders) - 6} more)" if len(offenders) > 6 else ""
    empty_note = (
        " (additionally: no touched code files could be parsed from the plan)"
        if not touched
        else ""
    )
    return _warn(
        cid,
        name,
        f"plan names {named}{extra} as a regression anchor / gate-selected test, but the "
        f"Step-9c selection over the plan's {len(touched)} touched file(s) does not pick "
        "it up (the selector's import map is one-hop — the #1536 shape) and no pytest "
        f"command in the plan runs it explicitly{empty_note}. Remedies: name the exact "
        "pytest invocation in the plan, import the touched module directly from the "
        "anchor test, or declare `N/A — no regression anchors` on its own line, "
        "unwrapped (no backticks/quotes)",
    )


# ─── Check 42 — cited commit SHA resolves ─────────────────────────────────

_C42_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c34/c41 pattern)

# Cite-as-commit context patterns. Each anchors on a marker BEFORE (or
# labeled inline with) the 7-40-char hex token — grammar ports from
# ``.claude/rules/workflow-fix-on-bug.md`` clause (d) + the crash-fix
# marker-note convention (``fix_sha=`` / ``merge_sha=``,
# ``.claude/rules/crash-fix-rounds.md`` § fix-engaged signal element 4).
_C42_CITE_PATTERNS = [
    # "commit `<sha>`" / "commit <sha>"
    re.compile(r"(?i)\bcommit\s+`?(?P<sha>[0-9a-f]{7,40})`?\b"),
    # "landed in `<sha>`" / "landed at `<sha>`"
    re.compile(r"(?i)\blanded\s+(?:in|at)\s+`?(?P<sha>[0-9a-f]{7,40})`?\b"),
    # "merge `<sha>`" / "merged in <sha>" / "merged at <sha>"
    re.compile(r"(?i)\bmerge(?:d)?\s+(?:in\s+|at\s+)?`?(?P<sha>[0-9a-f]{7,40})`?\b"),
    # "`<sha>` (commit)" / "`<sha>` -- commit" (em/en dash or ASCII hyphen)
    # / same for "merge". The dash class carries em dash + en dash + ASCII
    # hyphen; the RUF001 en-dash flag is a false-positive here (the en
    # dash is REAL plan text -- same rationale as NA_RE, and measurement
    # plans use these interchangeably).
    re.compile(
        r"(?i)`(?P<sha>[0-9a-f]{7,40})`\s*(?:\(commit\)|[—–-]\s*(?:commit|merge))"  # noqa: RUF001
    ),
    # "**commit:** <sha>" / "**merge:** <sha>" / "**commit_sha:** <sha>"
    re.compile(r"(?i)\*\*(?:commit|merge)(?:_sha)?:\*\*\s*`?(?P<sha>[0-9a-f]{7,40})`?\b"),
    # "fix_sha=<sha>" / "fix_sha: <sha>" / "merge_sha=<sha>" / "merge_sha: <sha>"
    re.compile(r"(?i)\b(?:fix_sha|merge_sha)\s*[:=]\s*`?(?P<sha>[0-9a-f]{7,40})`?\b"),
]

# Same-line disqualifiers: a hex captured by cite-context is dropped when
# the line ALSO carries workflow-fix fingerprint / session-basename / HF
# revision labels. These are the exclusion classes named in
# ``.claude/rules/workflow-fix-on-bug.md`` clause (d): "cite the token as
# what it actually is (transcript/session basename, HF revision,
# fingerprint), labeled as such."
_C42_EXCLUDE_LINE_PATTERNS = [
    re.compile(r"(?i)\bfingerprint\s*[:=]"),
    re.compile(r"(?i)\bsession(?:_id)?\s*[:=]|session\s+basename"),
    re.compile(r"(?i)\brevision\s*[:=]|--revision|hf_revision|dataset_revision"),
]


def _c42_rev_parse(sha: str) -> bool | None:
    """``True`` when ``sha`` resolves under
    ``git rev-parse --verify --quiet '<sha>^{commit}'``, ``False`` when it
    does not, ``None`` when git is unavailable (permission / OS / timeout —
    the check fails OPEN via a caller-side SKIP). Retries ONCE on a
    transient ``CalledProcessError`` / ``OSError`` after 0.1s (the plan's
    Methodology-critic concern: a brief ``.git/index.lock`` collision on
    the concurrent-committer repo root — the shared-VM #1201 shape — must
    not spuriously fail the check)."""
    cmd = ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"]
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(_C42_REPO_ROOT),
                check=False,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return None  # timeout is not retriable — the git command hung
        except OSError:
            if attempt == 1:
                time.sleep(0.1)
                continue
            return None
    return None  # unreachable (defensive; the loop always returns)


def check_commit_sha_resolves(plan: str, kind: str) -> CheckResult:
    """Every hex token the plan cites AS A COMMIT — ``commit `<sha>```,
    ``landed in <sha>``, ``**commit:** <sha>``, ``fix_sha=<sha>`` — must
    resolve under ``git rev-parse --verify --quiet '<sha>^{commit}'``.
    Mirrors ``.claude/rules/workflow-fix-on-bug.md`` clause (d) on the
    filing side.

    FAIL (all kinds) — a typo'd commit SHA is a factual defect independent
    of experiment kind (incident #1683: ``7c7095f40e`` was cited as the
    fix commit; the real commit was ``7c8095f40e`` and the fact-checker
    round burned proving the mismatch; #1414: transcript basename
    ``fc2b61b7`` shipped as "the fix commit"). Bare hex tokens
    (fingerprints, HF revisions, transcript basenames, arXiv ids) are OUT
    OF SCOPE — the cite-context grammar in ``_C42_CITE_PATTERNS`` gates
    entry, and ``_C42_EXCLUDE_LINE_PATTERNS`` drops same-line
    disqualifiers. Hex tokens inside fenced code blocks are excluded via
    ``_fence_mask``.

    Fail-open on git unavailability: the check is a mechanical gate, not
    a hard block on a broken git install / permission failure. A brief
    ``.git/index.lock`` collision retries ONCE (0.1 s) before SKIPping."""
    cid, name = "c42_commit_sha_resolves", "cited commit SHA resolves"
    del kind  # this check is kind-blind — a typo'd commit is a factual defect for all
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    cited: list[tuple[int, str, str]] = []  # (line_idx, sha, matched_context)
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if any(p.search(line) for p in _C42_EXCLUDE_LINE_PATTERNS):
            continue
        for pat in _C42_CITE_PATTERNS:
            for m in pat.finditer(line):
                sha = m.group("sha").lower()
                cited.append((i, sha, m.group(0)))
    if not cited:
        return _skip(cid, name, "no hex tokens cited as commits detected")
    # Deduplicate by SHA (a plan may cite the same commit N times).
    unique: dict[str, tuple[int, str]] = {}
    for ln, sha, ctx in cited:
        unique.setdefault(sha, (ln, ctx))
    unresolved: list[tuple[str, str]] = []
    for sha, (_ln, ctx) in unique.items():
        result = _c42_rev_parse(sha)
        if result is None:
            # Fail-open on git unavailability — SKIP the whole check
            # (rather than mixed-verdict FAIL some / skip others), because
            # a partial run under a broken git install would mask the very
            # positives the check is here to catch.
            return _skip(
                cid,
                name,
                "git rev-parse unavailable (permission / OS / timeout) — check inconclusive",
            )
        if not result:
            unresolved.append((sha, ctx))
    if not unresolved:
        return _pass(
            cid,
            name,
            f"all {len(unique)} cited commit SHA(s) resolve (verified via "
            "`git rev-parse --verify --quiet '<sha>^{commit}'`)",
        )
    detail = (
        f"{len(unresolved)} of {len(unique)} cited commit SHA(s) do NOT resolve as commits: "
        + "; ".join(f"{sha} (context: {ctx!r})" for sha, ctx in unresolved[:4])
        + (" …" if len(unresolved) > 4 else "")
        + ". Remedy: verify the SHA at compose time — "
        "`git rev-parse --verify --quiet '<sha>^{commit}'` — and re-derive the real "
        "commit (`git log --oneline --since='14 days ago' -- <touched-file>`) or "
        "cite the token as what it actually is (an HF revision, transcript basename, "
        "fingerprint), labeled as such. Mirrors "
        "`.claude/rules/workflow-fix-on-bug.md` clause (d)."
    )
    return _fail(cid, name, detail)


# ─── Check 43 — /workspace sentinels need a /workspace-contract lane ───────

# Trigger scans the RAW plan text (NOT strip_fences): sentinel declarations
# live in fenced ``phase_outputs:`` YAML by design (the c30 convention — the
# founding #1775 instance declared `/workspace/logs/issue-1775-p*.done`
# inside a fenced block), so a stripped-prose scan would miss exactly the
# founding shape. Two arms, same-line co-occurrence for arm (a):
#   (a) a line carrying "sentinel" (case-insensitive) AND "/workspace/";
#   (b) a `/workspace/logs/issue-` path (the poll_pipeline.py sentinel
#       prefix), even with no "sentinel" token on the line.
_C43_SENTINEL_WORD_RE = re.compile(r"(?i)\bsentinel")
_C43_WS_LOGS_RE = re.compile(r"/workspace/logs/issue-")

# Satisfier (raw scan too — a dispatch command lives in a fenced block): a
# DRAINED-lane pin, `backend: runpod|fellows` (frontmatter / prose line) or
# a `--backend runpod|fellows` dispatch flag. fellows joined the drained
# set at #1898 (slurm_monitor.drain_cluster_sentinels); gcp LEFT the
# pinnable set at #2028 (GCP provisioning disabled — an explicit gcp pin
# now raises GcpDisabledError at route(), so it cannot satisfy this check).
_C43_LANE_PIN_RE = re.compile(r"(?i)(?:\bbackend:\s*|--backend[=\s]+)(?:runpod|fellows)\b")

# The rule's own escape phrase, standalone at line start (leading
# list/blockquote/bold markers tolerated via the `_standalone_na_declared`
# lstrip convention; case-insensitivity explicit; hyphen / en-dash /
# em-dash variants tolerated). A label-prefixed MID-LINE mention (`**Lane /
# sentinel contract:** no sentinel dependence — auto-safe ...`, the #1738
# v4 shape) and any backtick/fence-wrapped paste are DELIBERATELY
# unrecognized (the #1238 anti-paste doctrine) — declare the escape
# unwrapped on its own line.
_C43_ESCAPE_RE = re.compile(
    r"(?i)^no sentinel dependence\s*[-–—]+\s*auto[-–— ]safe\b"  # noqa: RUF001 — real dash variants
)


def _c43_escape_declared(plan: str) -> bool:
    """Standalone ``no sentinel dependence — auto-safe`` declaration (the
    plan-compute-sizing.md rule's own phrase, no N/A prefix), or the
    shared-helper ``N/A — no sentinel dependence`` form. Fenced lines and
    wrapped pastes never satisfy (see ``_standalone_na_declared``)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if _C43_ESCAPE_RE.match(line.lstrip(" \t>*-")):
            return True
    return _standalone_na_declared(plan, r"no sentinel dependence\b")


def check_sentinel_lane(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, experiment-only: a plan that declares
    ``/workspace/...`` sentinel paths (gate sentinels, ``epm:results``
    payloads — the pod-side signaling contract) while leaving the backend
    on the unrestricted auto lane must either pin a DRAINED lane
    (``backend: runpod`` / ``backend: fellows``, or a
    ``--backend runpod|fellows`` dispatch flag; ``backend: gcp`` left the
    pinnable set at #2028 — GCP provisioning disabled) or declare the
    rule's own escape ``no sentinel dependence — auto-safe``. Mechanizes
    ``plan-compute-sizing.md`` § "Sentinel-signaling workloads need a
    /workspace-contract lane": on auto, a fellows capacity miss can
    fall through to the DRAC/Mila SLURM lanes, where compute nodes have no
    /workspace — the dispatcher dies FAIL-LOUD at ``mkdir -p
    /workspace/logs`` and burns the submission (#608). fellows is a
    DRAINED lane as of #1898 (``slurm_monitor.drain_cluster_sentinels``
    reads its cluster-shared ``/workspace/logs`` each poll tick, the same
    contract as GCP/RunPod), so every silent-loss path is closed and the
    residual auto-lane hazard is the fail-loud DRAC/Mila burn. Founding
    instance: #1775 plan v3 declared ``/workspace/logs/issue-1775-p*.done``
    sentinels in a fenced ``phase_outputs`` block while §9 said "no
    backend pin → auto lane"; only a Methodology critic caught it
    (Must-Fix M1). NEVER FAILs — the disposition is sometimes legitimately
    prose-satisfied in different words, and legacy plans must not bounce
    retroactively (the c39/c31/c34 family convention). kind-exempt outside
    experiment: infra workflow-fix plans (this check's own plan included)
    legitimately QUOTE ``/workspace/...`` sentinel paths without
    dispatching a sentinel-signaling workload. Trigger AND satisfiers scan
    the RAW plan text (NOT strip_fences): the sentinel declarations and
    the dispatch command both live in fenced blocks by design (unlike c39,
    whose trigger is prose vocabulary)."""
    cid, name = "c43_sentinel_lane", "/workspace sentinels vs unpinned auto lane"
    if kind != "experiment":
        return _skip(
            cid,
            name,
            "kind-exempt: sentinel-lane pinning is an experiment-plan (pod-dispatch) shape",
        )
    trigger_lines = [
        ln
        for ln in plan.splitlines()
        if _C43_WS_LOGS_RE.search(ln) or ("/workspace/" in ln and _C43_SENTINEL_WORD_RE.search(ln))
    ]
    if not trigger_lines:
        return _skip(cid, name, "no /workspace sentinel paths detected")
    if _C43_LANE_PIN_RE.search(plan):
        return _pass(
            cid,
            name,
            "/workspace-contract (drained) lane pinned (backend:/--backend runpod|fellows)",
        )
    if _c43_escape_declared(plan):
        return _pass(cid, name, "explicit escape declared (no sentinel dependence — auto-safe)")
    shown = "; ".join(ln.strip()[:70] for ln in trigger_lines[:3])
    return _warn(
        cid,
        name,
        f"plan declares /workspace sentinel paths ({shown!r}) with no drained lane "
        "pinned — on the auto lane a fellows capacity miss can fall through to the "
        "DRAC/Mila SLURM lanes, where compute nodes have no /workspace: the dispatcher "
        "dies fail-loud at `mkdir -p /workspace/logs` and burns the submission (#608). "
        "fellows is a DRAINED lane as of #1898 (the VM-side poller drains its "
        "/workspace/logs sentinels each tick, same contract as GCP/RunPod). Pin "
        "`backend: fellows` or `backend: runpod` (or carry `--backend "
        "runpod|fellows` in the dispatch command; `backend: gcp` is REFUSED as of "
        "#2028 — GCP provisioning disabled), or declare `no sentinel "
        "dependence — auto-safe` on its own line, unwrapped (no backticks/quotes), if "
        "nothing in the run posts through sentinels (plan-compute-sizing.md "
        "§ Sentinel-signaling workloads)",
    )


# ─── Check 44 — declared-committed paths not gitignored ────────────────────

_C44_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c34/c41/c42 pattern)

# Trigger: commit vocabulary on a line (raw scan, fences INCLUDED — the c43
# precedent: committed-output declarations live in fenced ``phase_outputs:``
# YAML and dispatch prose alike). Deliberately narrow — "committed to the
# issue branch / git / main / the repo" and the "rides the git clone"
# lane-reachability idiom — NOT the bare token "commit", which false-fires
# on fix-commit citations (c42's surface) and git mechanics prose.
_C44_COMMIT_VOCAB_RE = re.compile(
    r"(?i)(?:\bcommit(?:ted|s)?\s+(?:to|into|on)\s+(?:the\s+)?"
    r"(?:issue[-\s]branch|git\b|main\b|repo\b)"
    r"|\brides?\s+the\s+git\s+clone\b)"
)

# Path-like token: at least one `/`; later segments may carry glob (`*`) and
# brace (`{a,b}`) characters, expanded / reduced by ``_c44_expand_tokens``.
_C44_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-{},*]+)+")

# Same-section satisfier: an explicit force-add / staged-index-verification
# note next to the declaration (the /issue Step 9a-ter § Staged-index
# verification recipe). Plan-wide satisfiers are deliberately NOT accepted —
# a global mention elsewhere must not silence a specific declared path.
_C44_FORCE_ADD_RE = re.compile(
    r"(?i)(?:git\s+add\s+(?:-f\b|--force\b)|\bforce[-\s]add"
    r"|staged[-\s]index[-\s]verif|ls-files\s+--others\s+--ignored)"
)


def _c44_expand_tokens(line: str) -> list[str]:
    """Repo-relative path tokens on one trigger line. Excluded: absolute
    paths (``/workspace/...`` — pod-side, not repo paths) and URL tails
    (``https://host/...``) — both arrive with a ``/`` immediately before the
    token, the one shared signature. Brace groups (``{a.json,b.json}``)
    expand to every member; glob-bearing tokens reduce to their deepest
    literal directory prefix (check-ignore on the directory); trailing
    sentence punctuation is stripped. Returns [] when nothing survives."""
    out: list[str] = []
    for m in _C44_PATH_TOKEN_RE.finditer(line):
        if m.start() > 0 and line[m.start() - 1] == "/":
            continue  # absolute path or URL tail — not a repo-relative path
        tok = m.group(0).rstrip(".,")
        bm = re.match(r"^(.*)\{([^{}]*)\}(.*)$", tok)
        variants = (
            [bm.group(1) + part + bm.group(3) for part in bm.group(2).split(",")] if bm else [tok]
        )
        for v in variants:
            v = v.rstrip(".,")
            if any(ch in v for ch in "*?{}"):
                literal: list[str] = []
                for comp in v.split("/"):
                    if any(ch in comp for ch in "*?{}"):
                        break
                    literal.append(comp)
                v = "/".join(literal)
            if v:
                out.append(v)
    return out


def _c44_check_ignore(paths: list[str]) -> dict[str, str] | None:
    """Map each gitignore-MATCHED path in ``paths`` to its matching pattern,
    via ONE batched, index-aware ``git check-ignore -v --stdin`` call
    (deliberately never ``--no-index``: a tracked path rides the clone
    regardless of ignore rules — the #1900 post-fix state — so the index
    consult is load-bearing). Exit 0/1 are both healthy (some/none ignored);
    a matched pattern beginning with ``!`` (negation) reads as NOT-ignored —
    some git versions print negation-matched paths under ``-v``. Returns
    ``None`` when git is unavailable (timeout / OSError after one 0.1 s
    retry / exit ≥ 2) — the caller SKIPs fail-open, the c42 contract."""
    cmd = ["git", "check-ignore", "-v", "--stdin"]
    payload = "\n".join(paths) + "\n"
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(_C44_REPO_ROOT),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None  # timeout is not retriable — the git command hung
        except OSError:
            if attempt == 1:
                time.sleep(0.1)
                continue
            return None
        if r.returncode >= 2:
            return None
        ignored: dict[str, str] = {}
        for out_line in r.stdout.splitlines():
            head, sep, path = out_line.rpartition("\t")
            if not sep:
                continue
            parts = head.split(":", 2)
            pattern = parts[2] if len(parts) == 3 else head
            if pattern.startswith("!"):
                continue  # negation pattern — the path is NOT ignored
            ignored[path] = pattern
        return ignored
    return None  # unreachable (defensive; the loop always returns)


def _c44_heads(lines: list[str], mask: list[bool]) -> list[tuple[int, int]]:
    """Fence-unmasked ``##``/``###`` heading positions as (line_idx, level)."""
    heads: list[tuple[int, int]] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) in (2, 3):
            heads.append((i, len(m.group(1))))
    return heads


def _c44_section_span(heads: list[tuple[int, int]], idx: int, n_lines: int) -> tuple[int, int]:
    """[start, end) of the ``##``/``###`` section containing line ``idx`` —
    the nearest preceding heading to the next heading of the same or higher
    level; the preamble (before the first ``##``/``###``) is its own
    section."""
    prev: tuple[int, int, int] | None = None
    for j, (h_idx, lvl) in enumerate(heads):
        if h_idx <= idx:
            prev = (j, h_idx, lvl)
        else:
            break
    if prev is None:
        return 0, heads[0][0] if heads else n_lines
    j, h_idx, lvl = prev
    for h2_idx, lvl2 in heads[j + 1 :]:
        if lvl2 <= lvl:
            return h_idx, h2_idx
    return h_idx, n_lines


def check_committed_paths_not_gitignored(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, ALL kinds: a plan that declares output/config
    paths as committed-to-git while a live ``.gitignore`` rule would silently
    skip them under a plain ``git add`` (rc=0, no error — the #958
    signature) must note the force-add / staged-index verification in the
    SAME section as the declaration. A gitignore-eaten "committed" input
    strands the git-clone lanes (GCP, fellows) at their first config read
    (the #734 shape). Founding instance: #1900's plan declared
    ``data/issue_1900/config/{subset.json,arms.json}`` committed to the
    issue branch while ``data/*`` matched — caught only by the round-1
    Methodology critic; the fix force-added them (index-aware default
    ``git check-ignore`` correctly reads the post-fix tracked state as
    not-ignored). WARN, never FAIL: the declaration may be satisfiable by an
    implementation-time force-add the plan simply forgot to note — the
    critic ensemble stays the judgment layer. Fail-open SKIP when git is
    unavailable (the c42 contract). Trigger AND path extraction scan the RAW
    plan text (fences INCLUDED, the c43 precedent); the
    ``N/A — no committed outputs`` escape short-circuits BEFORE any git
    access (the c34 NA idiom)."""
    cid, name = "c44_committed_paths_gitignored", "declared-committed paths not gitignored"
    del kind  # all kinds — a gitignore-eaten declared output strands any kind's lanes
    if _standalone_na_declared(plan, r"no committed outputs\b"):
        return _skip(cid, name, "explicit escape declared (N/A — no committed outputs)")
    lines = plan.splitlines()
    vocab_idx = [i for i, line in enumerate(lines) if _C44_COMMIT_VOCAB_RE.search(line)]
    if not vocab_idx:
        return _skip(cid, name, "no committed-output declarations detected")
    declared: list[tuple[int, str]] = []  # (line_idx, repo-relative path)
    for i in vocab_idx:
        for tok in _c44_expand_tokens(lines[i]):
            declared.append((i, tok))
    if not declared:
        return _skip(cid, name, "commit vocabulary without extractable repo paths")
    unique_paths = sorted({p for _, p in declared})
    ignored = _c44_check_ignore(unique_paths)
    if ignored is None:
        return _skip(cid, name, "git check-ignore unavailable — check inconclusive")
    if not ignored:
        return _pass(
            cid,
            name,
            f"none of the {len(unique_paths)} declared-committed path(s) match a live "
            ".gitignore rule (index-aware `git check-ignore`; tracked paths read not-ignored)",
        )
    mask = _fence_mask(lines)
    heads = _c44_heads(lines, mask)
    offenders: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, p in declared:
        if p not in ignored or p in seen:
            continue
        start, end = _c44_section_span(heads, i, len(lines))
        if _C44_FORCE_ADD_RE.search("\n".join(lines[start:end])):
            continue  # same-section force-add / staged-index-verification note
        seen.add(p)
        offenders.append((p, ignored[p]))
    if not offenders:
        return _pass(
            cid,
            name,
            "every gitignore-matched declared-committed path carries a same-section "
            "force-add / staged-index-verification note",
        )
    shown = "; ".join(f"`{p}` (matched by `{rule}`)" for p, rule in offenders[:4])
    more = " …" if len(offenders) > 4 else ""
    return _warn(
        cid,
        name,
        f"plan declares {len(offenders)} committed output path(s) a live .gitignore rule "
        f"would silently skip under a plain `git add` (rc=0, no error — the #958 signature): "
        f'{shown}{more}. A gitignore-eaten "committed" input strands the git-clone lanes '
        "(GCP/fellows) at their first read (#734; founding instance #1900: "
        "`data/issue_1900/config/` declared committed while `data/*` matched). Remedy: "
        "force-add with staged-index verification (`git add -f` + `git ls-files --others "
        "--ignored --exclude-standard` per /issue SKILL.md Step 9a-ter § Staged-index "
        "verification) noted in the SAME section as the declaration, or relocate the output "
        "out of the ignored root, or drop the committed claim; if the vocabulary is "
        "incidental, declare `N/A — no committed outputs` on its own line, unwrapped "
        "(no backticks/quotes)",
    )


# ─── Check 45 — trained-base change DV vs base-side predictor companion ────

# Trigger arm (a): a change-DV signature anywhere in the STRIPPED plan prose
# (fenced command/code blocks must neither satisfy nor trip — the c39
# convention): `trained - base` (hyphen / U+2212 minus / en dash, spaced or
# not), `post - pre`, or a `Delta log P` / `delta log P` delta form. Word
# boundaries on `base`/`pre` keep `trained-baseline` / `post-prefix` (this
# project's prefix-mapping vocabulary) from false-firing.
_C45_CHANGE_DV_RE = re.compile(
    r"(?i)\btrained\s*[-−–]\s*base\b|\bpost\s*[-−–]\s*pre\b|(?:Δ|\bdelta\b)\s*log\s*P"  # noqa: RUF001 — real minus/en-dash plan text
)

# Trigger arm (b): a base-side predictor RACED — one stripped line carrying
# BOTH a base-side-quantity token AND a predictor-context token (the same-line
# conjunction keeps generic "base rate" prose from firing alone). Grounded on
# the founding #1900 v4 Plan-Summary instance: "incumbent P7 (base behavioral
# propensity) raced and partialled".
_C45_BASE_SIDE_RE = re.compile(
    r"(?i)\bbase (?:behavioral )?propensit(?:y|ies)\b"
    r"|\bbase[- ]side (?:predictors?|propensit(?:y|ies))\b"
    r"|\bbase log ?P\b|\bbase rates?\b|\bbase judge scores?\b"
)
_C45_PREDICTOR_CTX_RE = re.compile(
    r"(?i)\b(?:predictors?|candidates?|champions?|race[sd]?|incumbents?|horses?)\b"
)

# Satisfier (i): companion-column registration (grounded on #1900 v5
# § "Registered DV-identity companion columns" — "level companion" /
# "change companion" labels).
_C45_COMPANION_RE = re.compile(
    r"(?i)\bcompanion columns?\b|\blevel companions?\b"
    r"|\b(?:graded[- ])?change companions?\b|\blevel[- ]DV companions?\b"
)

# Satisfier (ii): a stated winner sign convention (grounded on #1900 v5
# "Winner-selection convention (registered): the champion argmax is over
# SIGNED Spearman rho").
_C45_SIGN_CONVENTION_RE = re.compile(
    r"(?i)\bwinner[- ]selection convention\b|\bsign conventions?\b"
    r"|\bsigned (?:Spearman )?(?:ρ|rho)\b"  # noqa: RUF001 — real rho char in plan text
)
# Degenerate-|rho| guard (critic MF1): bare `|rho|` / `absolute` is NOT a
# standalone satisfier — predictor-race plans near-universally carry
# incidental max-|rho| prose in their selection-symmetric sections (#1900 v4,
# the must-WARN fixture) — it counts ONLY on a line also carrying a
# winner/convention/champion/argmax context token.
_C45_ABS_RHO_RE = re.compile(r"(?i)\|(?:ρ|rho)\||\babsolute\b")  # noqa: RUF001 — real rho char in plan text
_C45_WINNER_CTX_RE = re.compile(r"(?i)\b(?:winners?|conventions?|champions?|argmax)\b")


def _c45_escape_declared(plan: str) -> bool:
    """Standalone ``N/A — no base-side predictor vs change DV`` declaration
    (see ``_standalone_na_declared`` for the anti-paste rationale)."""
    return _standalone_na_declared(plan, r"no base[- ]side predictor vs change DV\b")


def check_change_dv_base_predictor_companion(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, experiment-only: a plan that races a
    BASE-SIDE predictor (base propensity / base log P / base rate / base
    judge score in a predictor-candidate roster) against a
    ``trained - base`` (or ``post - pre`` / ``Delta log P``) CHANGE dependent
    variable must register BOTH (i) a level (or change) COMPANION column
    and (ii) a stated WINNER SIGN CONVENTION (signed rho vs |rho|), or declare
    the standalone escape ``N/A — no base-side predictor vs change DV``.
    Mechanizes the #559/#605 pattern (critic-lens-reference.md Statistics
    lens item 2, "Inherited-positive DV-swap"): the base term enters the
    change DV with a mechanical ~ -1 coefficient, so per-panel DV identity
    can manufacture — or destroy — the champion verdict (#605: base-to-level
    rho +0.28/+0.19 vs base-to-delta rho -0.43/-0.54). Founding instance: #1900
    round 1, where only the Stats-lens critic caught it (Must-Fix; v5
    registered the "Registered DV-identity companion columns" block + the
    "Winner-selection convention" line this check's satisfiers are
    grounded on). NEVER FAILs — the trigger is a vocabulary heuristic and
    a legitimate disposition is sometimes prose-satisfied in different
    words (the c39/c31/c34/c43 family convention); the LLM Statistics
    critic remains the FAIL authority. kind-exempt outside experiment:
    infra workflow-fix plans (this check's own plan included) legitimately
    QUOTE the trigger vocabulary without racing predictors (the c43
    precedent). Trigger AND satisfiers scan STRIPPED prose (fenced blocks
    masked — the c39 convention)."""
    cid, name = (
        "c45_change_dv_base_predictor_companion",
        "trained-base change DV vs base-side predictor companion",
    )
    if kind != "experiment":
        return _skip(
            cid,
            name,
            "kind-exempt: base-predictor-vs-change-DV racing is an experiment-plan shape",
        )
    text = strip_fences(plan)
    if not _C45_CHANGE_DV_RE.search(text):
        return _skip(cid, name, "no trained-base / post-pre / Delta log P change-DV signature")
    race_lines = [
        ln
        for ln in text.splitlines()
        if _C45_BASE_SIDE_RE.search(ln) and _C45_PREDICTOR_CTX_RE.search(ln)
    ]
    if not race_lines:
        return _skip(
            cid,
            name,
            "no base-side predictor raced (no line carries both a base-side quantity "
            "and predictor-race vocabulary)",
        )
    if _c45_escape_declared(plan):
        return _pass(cid, name, "explicit N/A declared (no base-side predictor vs change DV)")
    has_companion = bool(_C45_COMPANION_RE.search(text))
    has_convention = bool(_C45_SIGN_CONVENTION_RE.search(text)) or any(
        _C45_ABS_RHO_RE.search(ln) and _C45_WINNER_CTX_RE.search(ln) for ln in text.splitlines()
    )
    if has_companion and has_convention:
        return _pass(cid, name, "companion column registered and winner sign convention stated")
    if has_companion:
        missing = "a stated winner sign convention (signed rho vs |rho|)"
    elif has_convention:
        missing = "a registered level/change companion column"
    else:
        missing = (
            "a registered level/change companion column AND a stated winner sign "
            "convention (signed rho vs |rho|)"
        )
    shown = "; ".join(ln.strip()[:70] for ln in race_lines[:2])
    return _warn(
        cid,
        name,
        f"plan races a base-side predictor against a trained-base CHANGE DV ({shown!r}) "
        f"without {missing} — the base term enters the change DV with a mechanical ~ -1 "
        "coefficient, so per-panel DV identity can manufacture the champion verdict "
        "(#559/#605: base-to-level rho +0.28/+0.19 vs base-to-delta rho -0.43/-0.54; the #1900 "
        "round-1 Stats Must-Fix; critic-lens-reference.md Statistics lens item 2). "
        "Register a level (or graded-change) companion column for the base-side "
        "candidate AND state the winner-selection convention (signed Spearman rho vs "
        "|rho|), or declare `N/A — no base-side predictor vs change DV` on its own "
        "line, unwrapped (no backticks/quotes), if no base-side predictor is raced "
        "against a change DV",
    )


# ─── Check 46 — plan-embedded dispatch command CLI-parses (#2161) ──────────

#: Substring marking a candidate dispatch-command line/span (c46).
_C46_DISPATCH_TOKEN = "dispatch_issue.py"

#: Inline-code spans on prose lines (c46 candidate source #2).
_C46_INLINE_CODE_RE = re.compile(r"`([^`]+)`")

#: Placeholder tokens plans legitimately embed in illustrative commands:
#: ``<N>``-style angle placeholders and ``$VAR`` / ``${VAR}`` shell vars.
#: Substituted with ``"1"`` pre-parse so int-typed argparse args accept
#: them — an illustrative placeholder is never a drift WARN.
_C46_PLACEHOLDER_RE = re.compile(
    r"<[^<>\s]+>|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*"
)

#: Shell conditional expansions (``${VAR:+...}``) are optional by
#: construction — stripped WHOLE (modeling the VAR-unset case) BEFORE
#: shlex, because they otherwise split into unparseable fragments
#: (the SKILL.md Step 6b snippet's ``${BACKEND:+--backend "$BACKEND"}``).
_C46_COND_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:\+[^}]*\}")

#: Shell operators that end the argv of interest within one span/line.
_C46_SHELL_OPS = frozenset({"&&", "||", ";", "|", "&", ">", ">>", "<", "2>", "2>&1"})

_C46_DISPATCH_CLI_PATH = Path(__file__).resolve().parent / "dispatch_issue.py"  # tests monkeypatch
_c46_argparser_cache: list = []  # [(parser | None, detail)] once resolved


def _c46_argparser():
    """Lazily path-load ``scripts/dispatch_issue.py`` (stdlib-only
    module-level imports; NOT registered in ``sys.modules`` — the c41
    convention) and build its PUBLIC ``build_argparser()`` (#2161).

    Returns ``(parser, "")`` on success or ``(None, <detail>)`` on ANY
    load/build failure (file absent on off-repo ``--plan-file`` runs, a
    pre-#2161 checkout without the public alias, a broken CLI module) —
    the caller SKIPs loudly on ``None``: c46 is a plan-drift detector,
    never a gate on the CLI module itself. Cached one-shot so repeated
    checks (tests, corpus sweeps) load the module once.
    """
    if not _c46_argparser_cache:
        parser, detail = None, ""
        try:
            if not _C46_DISPATCH_CLI_PATH.is_file():
                raise FileNotFoundError(_C46_DISPATCH_CLI_PATH)
            spec = importlib.util.spec_from_file_location(
                "_c46_dispatch_issue", _C46_DISPATCH_CLI_PATH
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            parser = mod.build_argparser()
        except Exception as exc:  # any load failure -> loud SKIP at the caller
            parser, detail = None, f"{type(exc).__name__}: {exc}"
        _c46_argparser_cache.append((parser, detail))
    return _c46_argparser_cache[0]


def _c46_command_candidates(plan: str) -> list[str]:
    """Candidate command strings mentioning ``dispatch_issue.py``.

    Two sources: fenced-code-block lines (backslash continuations joined
    into one logical command; bash ``#`` comment lines skipped) and
    inline-code spans on prose lines. Returns raw candidate strings; the
    caller tokenizes + filters.
    """
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[str] = []
    cont: list[str] = []

    def _flush() -> None:
        if cont:
            out.append(" ".join(tok for tok in cont if tok))
            cont.clear()

    for line, fenced in zip(lines, mask, strict=True):
        stripped = line.strip()
        if fenced:
            if stripped.startswith(("```", "~~~")) or stripped.startswith("#"):
                _flush()
                continue
            if stripped.endswith("\\"):
                cont.append(stripped[:-1].strip())
                continue
            cont.append(stripped)
            _flush()
        else:
            _flush()
            out.extend(_C46_INLINE_CODE_RE.findall(line))
    _flush()
    return [c for c in out if _C46_DISPATCH_TOKEN in c]


def _c46_dry_parse(parser, argv: list[str]):
    """Parse ``argv`` capturing argparse's ``SystemExit`` + usage output.

    Returns ``(namespace, None)`` on success, ``(None, <error detail>)``
    on rejection — never prints to the real stdout/stderr.
    """
    import contextlib
    import io

    stream = io.StringIO()
    try:
        with contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
            return parser.parse_args(argv), None
    except SystemExit:
        text = stream.getvalue().strip()
        detail = next(
            (
                ln.split("error:", 1)[1].strip()
                for ln in reversed(text.splitlines())
                if "error:" in ln
            ),
            text[-160:] or "argparse rejected the argv",
        )
        return None, detail


def _c46_has_flag(argv: list[str], flag: str) -> bool:
    """True when ``argv`` carries ``flag`` (bare or ``flag=value`` form)."""
    return any(tok == flag or tok.startswith(flag + "=") for tok in argv)


def _c46_argv_from_tokens(tokens: list[str]) -> list[str] | None:
    """Command argv after the dispatch token, or ``None`` for a non-command.

    Neutralizes placeholder tokens (substituted ``"1"``), truncates at the
    first shell operator, and returns ``None`` when the mention is a bare
    file reference or a prose mention (no ``--`` flag anywhere) — those
    are never drift WARNs.
    """
    idx = next((i for i, tok in enumerate(tokens) if tok.endswith(_C46_DISPATCH_TOKEN)), None)
    if idx is None or idx + 1 >= len(tokens):
        return None  # a bare file reference, not a command invocation
    argv = [_C46_PLACEHOLDER_RE.sub("1", tok) for tok in tokens[idx + 1 :]]
    stop = next((i for i, tok in enumerate(argv) if tok in _C46_SHELL_OPS), len(argv))
    argv = argv[:stop]
    if not argv or not any(tok.startswith("--") for tok in argv):
        return None  # prose mention ("the dispatch_issue.py launch command")
    return argv


def _c46_drift_arms(parser, argv: list[str]) -> tuple[list[str], list[str]]:
    """Drift arms + FYI notes for ONE command argv against the live CLI.

    Returns ``(arms, notes)`` — empty ``arms`` = clean; ``notes`` are
    verdict-neutral FYI strings the caller appends to its detail tail.

    Arm 1: the argv does not parse (``launch`` subcommand missing, unknown
    flag, wrong type). Arms 2-3 fire only on a LAUNCH-shaped argv — an
    explicit ``launch`` subcommand, or the #1336 missing-subcommand shape
    (argv leads with a flag); a ``finalize`` command gets neither.

    Arm 4 (#2202/#2254) is NAMESPACE-level: a PARSED ``launch`` namespace
    must satisfy the runtime's exactly-one-of ``--workload-cmd`` /
    ``--hydra`` rule (``dispatch_issue.py main()``, #588) — evaluated on
    the parsed namespace exactly as the runtime does, so empty-string
    values, ``--flag=value`` forms, and the append-action ``--hydra``
    follow argparse semantics byte-for-byte. Gated on
    ``getattr(ns, "action", None) == "launch"`` (a does-not-parse argv has
    ``ns is None`` — arm 1 owns it; ``finalize`` namespaces lack the
    launch dests by construction) and on BOTH ``workload_cmd`` / ``hydra``
    dests existing — a future CLI dest rename SKIPs the arm with an FYI
    note rather than reading renamed dests as absent and firing
    ``neither`` on every compliant launch plan.

    FYI note (#909, verdict-neutral): an explicit ``--backend runpod``
    launch carrying a non-empty ``--workload-cmd`` but no
    ``--execute-workload`` is provision-only — the workload does not
    auto-start on the runpod lane (expected when the experimenter launches
    it on the pod, so a note, never a WARN).
    """
    arms: list[str] = []
    notes: list[str] = []
    ns, err = _c46_dry_parse(parser, argv)
    if ns is None:
        arms.append(f"does not parse ({err})")
    launch_shaped = argv[0] == "launch" or argv[0].startswith("-")
    if launch_shaped:
        if _c46_has_flag(argv, "--max-run-duration") and not _c46_has_flag(
            argv, "--time-budget-hours"
        ):
            arms.append(
                "--max-run-duration without --time-budget-hours (the fence threads only "
                "to the GCP instance auto-delete and is inert on SLURM lanes, where the "
                "wall fence is --time-budget-hours; runtime refusal: "
                "max_run_duration_slurm_inert_without_time_budget)"
            )
        if not _c46_has_flag(argv, "--repo-branch"):
            arms.append(
                "no --repo-branch (the runtime refuses when a live issue-<N> branch "
                "exists — reason: repo_branch_required_issue_branch_exists; "
                "--repo-branch main is the explicit escape)"
            )
    if ns is not None and getattr(ns, "action", None) == "launch":
        if hasattr(ns, "workload_cmd") and hasattr(ns, "hydra"):
            has_workload_cmd = bool((ns.workload_cmd or "").strip())
            has_hydra = bool(ns.hydra)
            if has_workload_cmd == has_hydra:
                arms.append(
                    "launch requires exactly one of --workload-cmd / --hydra "
                    f"(got {'both' if has_hydra else 'neither'}; an empty "
                    "--workload-cmd '' counts as not provided; runtime refuses "
                    "rc=2 — the #2202/#2254 provision-only shape)"
                )
            elif (
                has_workload_cmd
                and (getattr(ns, "backend", None) or "").strip().lower() == "runpod"
                and not getattr(ns, "execute_workload", False)
            ):
                notes.append(
                    "FYI: --backend runpod with --workload-cmd but no "
                    "--execute-workload is provision-only (the workload does not "
                    "auto-start on the runpod lane, #909) — expected when the "
                    "experimenter launches it on the pod"
                )
        else:
            notes.append(
                "exactly-one-of --workload-cmd/--hydra arm skipped: parsed launch "
                "namespace lacks the workload_cmd/hydra dests (CLI dest rename "
                "since #588?)"
            )
    return arms, notes


def check_dispatch_cmd_cli_parse(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: every plan-embedded
    ``dispatch_issue.py`` command (fenced code blocks + inline-code spans;
    backslash continuations joined) must dry-parse against the CLI's REAL
    argparser (``dispatch_issue.build_argparser()``, lazily path-loaded),
    and a launch-shaped command must not carry the three demonstrated
    drift shapes: ``--max-run-duration`` without ``--time-budget-hours``
    (the fence threads ONLY to the GCP instance auto-delete and is inert
    on SLURM lanes, where the wall fence is ``--time-budget-hours`` —
    runtime refusal ``max_run_duration_slurm_inert_without_time_budget``),
    a missing ``--repo-branch`` (the runtime refuses when a live
    ``issue-<N>`` branch exists — ``repo_branch_required_issue_branch_
    exists``; ``--repo-branch main`` is the explicit escape), and — on a
    PARSED ``launch`` namespace (#2202/#2254) — a violation of the
    runtime's exactly-one-of ``--workload-cmd`` / ``--hydra`` requirement
    (#588; an explicitly-empty ``--workload-cmd ''`` counts as not
    provided, both-provided WARNs too — the runtime refuses rc=2 either
    way). An explicit ``--backend runpod`` launch with a non-empty
    ``--workload-cmd`` and no ``--execute-workload`` additionally gets a
    verdict-neutral FYI note (provision-only on the runpod lane, #909).
    Mechanizes the #1336 v15 §9 drift: the plan-embedded launch command
    omitted the ``launch`` subcommand, carried ``--max-run-duration`` with
    no ``--time-budget-hours``, and omitted ``--repo-branch``. Placeholder
    tokens (``<N>``, ``$VAR``) parse as ordinary values (substituted
    ``"1"``); ``${VAR:+...}`` conditional expansions are stripped whole;
    prose mentions with no ``--`` flag are not commands; unsplittable
    lines get a per-line note, never a crash; no dispatch command found →
    SKIP; ``build_argparser`` unavailable (off-repo ``--plan-file``) →
    SKIP. NEVER FAILs — the runtime exit-2 refusals are the hard gate;
    this check is the plan-time drift detector (#2161).
    """
    del kind  # all kinds: a plan-embedded dispatch command drifts identically everywhere
    cid, name = "c46_dispatch_cmd_cli_parse", "plan-embedded dispatch command CLI-parses"
    candidates = _c46_command_candidates(plan)
    if not candidates:
        return _skip(cid, name, "no dispatch_issue.py command in fenced blocks or inline code")
    parser, load_detail = _c46_argparser()
    if parser is None:
        return _skip(cid, name, f"dispatch_issue.build_argparser unavailable ({load_detail})")
    offenders: list[str] = []
    notes: list[str] = []
    n_parsed = 0
    for cmd in candidates:
        cleaned = _C46_COND_EXPANSION_RE.sub("", cmd)
        try:
            tokens = shlex.split(cleaned)
        except ValueError as exc:
            notes.append(f"unsplittable line skipped ({exc}): {cmd[:60]!r}")
            continue
        argv = _c46_argv_from_tokens(tokens)
        if argv is None:
            continue  # bare file reference / prose mention, not a command
        n_parsed += 1
        arms, fyi_notes = _c46_drift_arms(parser, argv)
        notes.extend(fyi_notes)
        if arms:
            offenders.append(f"{cmd[:70]!r}: " + "; ".join(arms))
    if offenders:
        shown = " | ".join(offenders[:3])
        more = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
        tail = ("; " + "; ".join(notes)) if notes else ""
        return _warn(
            cid,
            name,
            f"plan-embedded dispatch_issue.py command(s) drift from the live CLI: {shown}{more} "
            "— the #1336 v15 shape (a plan-embedded launch command missing the `launch` "
            "subcommand, fencing via --max-run-duration alone, or omitting --repo-branch) "
            "and the #2202/#2254 provision-only shape (launch without exactly one of "
            "--workload-cmd / --hydra) die or silently mis-fence at dispatch time; copy "
            "the SKILL.md Step 6b launch snippet (launch subcommand + explicit "
            "--repo-branch + a workload via --workload-cmd or --hydra + "
            f"--time-budget-hours on SLURM-reachable lanes){tail}",
        )
    if n_parsed == 0:
        detail = "dispatch_issue.py mentions are bare references, not command invocations"
        if notes:
            detail += "; " + "; ".join(notes)
        return _skip(cid, name, detail)
    detail = f"{n_parsed} dispatch command(s) dry-parse against the live CLI"
    if notes:
        detail += "; " + "; ".join(notes)
    return _pass(cid, name, detail)


# ─── Check 47 — planned_wall_h cells parse for the poller tripwire ─────────


def check_wall_cell_parseable(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds (trigger-conditional, mostly not
    kind-gated — the ``poll_pipeline.py`` phase-ETA tripwire reads ANY
    task's plan regardless of ``kind``): every §9 ``planned_wall_h`` data
    cell must parse under the SHARED cosmetic-prefix float rule
    (``explore_persona_space.plan_wall_budget`` — the SAME parser the
    poller's budget uses, so this plan-time WARN and the runtime disable
    cannot drift, #2172 AC #3/#4). ONE unparseable cell fail-safes the
    poller's WHOLE-run tripwire off; pre-#2172 that was silent — #2163's
    parenthesized ``(1.5)`` conditional cell cost a ~6h run its backstop
    with one poll-tick INFO line to show for it. Absent-table branch
    (#2123, narrowing the former unconditional SKIP): a ``kind:
    experiment`` plan with BOOKED GPU-hours > 0 (the c5 ``GPU_LINE_RE``
    first-match value — the same value the Step 2c consumer reads) and NO
    ``planned_wall_h`` table WARNs — for a plan that books compute, the
    tripwire never arming IS the loss (#2091: #2061 booked 70 GPU-h and
    #1739 booked 14, both table-less, both unprotected). With 0 booked
    GPU-hours, no GPU token at all, or a non-experiment kind, the
    original justification still holds — such a plan arms no tripwire,
    so there is nothing to lose => SKIP. NEVER FAILs in v1 (the
    c26/c29/c33 precedent — heuristic compute-table checks stay WARN-only
    until a clean corpus baseline licenses escalation; corpus 2026-08-12:
    13 of 229 experiment tasks' latest plans sit in the new-WARN bucket,
    not a clean baseline).
    """
    cid, name = "c47_wall_cell_parseable", "planned_wall_h cells parse for the poller tripwire"
    budget = parse_plan_wall_budget(plan)
    if budget.reason == "no_table":
        booked = GPU_LINE_RE.search(plan)
        if kind == "experiment" and booked is not None and float(booked.group(1)) > 0:
            return _warn(
                cid,
                name,
                f"no `planned_wall_h` table while this experiment books "
                f"{booked.group(1)} GPU-hours — the poll_pipeline.py phase-ETA "
                f"tripwire never arms for the whole run (#2091: the #2061/#1739 "
                f"shape); add a §9 compute table with a `planned_wall_h` column",
            )
        return _skip(cid, name, "no `planned_wall_h` table — this plan arms no poller tripwire")
    if budget.unparseable:
        shown = "; ".join(
            f"{c.fmt} row {c.row_text!r} ({c.reason})" for c in budget.unparseable[:3]
        )
        if len(budget.unparseable) > 3:
            shown += f"; +{len(budget.unparseable) - 3} more"
        return _warn(
            cid,
            name,
            f"{shown} — ONE unparseable planned_wall_h cell disables the poller's phase-ETA "
            f"tripwire for the WHOLE run (fail-safe; {len(budget.rows)} parseable row(s) "
            # NB: the range's en dash is written as a unicode escape — a
            # literal trips RUF001 under the full-ruleset policy pin
            # (tests/test_ruff_policy.py, #2179).
            "discarded with it; #2163/#2172/#2179): write a bare float, a `≤X` bound, or "
            "an `A\u2013B` range (the upper bound is used) in the `planned_wall_h` cell "
            "and put the conditionality in the `basis` cell",
        )
    return _pass(cid, name, f"{len(budget.rows)} row(s), total {budget.total_h:.2f} h")


# ─── Check 48 — §9 basis-vs-booked arithmetic ───────────────────────────────
# Mechanizes .claude/rules/plan-compute-sizing.md § "Per-cell fit phases"
# (the booked-figure coherence half): a §9 compute row whose OWN basis text
# contradicts the row's OWN booked columns. Two arms:
#   Arm A (GPU-h axis ONLY): the basis DERIVES a GPU-h figure (a
#     unit-word-tolerant product expression followed by a derivation arrow,
#     outside a bidirectional citation/cap window) > 2.0x the booked
#     planned_gpu_h, with no row-scoped reconciliation marker.
#   Arm B (abort-vs-booked): the row states a per-cell abort threshold
#     BELOW the per-cell wall its own booking implies
#     (planned_wall_h x width / n_cells) — the gate would fire at cell 1 on
#     a run performing exactly as booked.
# Category error underneath both (#1336 v16 `EXT_off`, which survived TWO
# approved plan versions at 0 FAIL): parallelism divides WALL time, never
# GPU-hours (GPU-h = wall x n_gpus) — dividing a GPU-h figure by an N-way
# width and booking the quotient as GPU-h under-books by exactly N.
# Recurrence: #823 / #811 / #1092.
#
# Calibration (task #2177; DEVELOPMENT-SET numbers — the regexes below were
# calibrated against the same persisted-plan corpus they were measured on;
# ANY future change to ANY regex below re-runs the corpus sweep and records
# the realized numbers here — the c32/c33 gate precedent). Sweep 2026-08-07
# (implementation-time, AS-SHIPPED code through verify_plan_text,
# kind="experiment" uniform — an upper bound: kind-exempt plans SKIP in
# production) over 5,169 plan files (tasks/*/*/plans/*.md): 3 WARN files /
# 3 offending rows ~= 0.28% of the ~1,056 planned_gpu_h-bearing
# denominator — #1336 v15 (arm A 82 vs 30; its abort is stated without the
# `> N min/cell` idiom, so arm B correctly stays silent), #1336 v16 (arm A
# 90 vs 30 AND arm B ~91 min vs abort > 30 min/cell; the driving incident),
# #634 v1 (arm A 2.2 vs 0.7 — the accepted citation-FP residual: the
# parent's `#594` token sits ~55 chars before its realized figure, outside
# the 30-char backward window). genuine = 2/3.
# Known-answer harness check: v2 semantics (largest token, backward-only
# window, substring gpu header) reproduce ~130 rows / ~118 files.
# Design history is MEASURED, not read (plan #2177 §11-§13):
#   - v2's "largest GPU-h token" extraction measured 11.3% corpus WARN
#     (131 rows / 119 files): citation figures dominate, and FP ratios run
#     3.1x-23x while the driving incident sits at exactly 3.0x, so no
#     ratio threshold separates them — the derived-figure-only extraction
#     (product + arrow + citation window) is load-bearing.
#   - v3's `\bgpu` header guard measured 0 hits corpus-wide (`_` is a word
#     character; see _C48_GPU_H_HEADER_RE above _ComputeRow).
#   - v3's product regex required the multiplication sign to abut the
#     number; the driving derivation is `20 cells x 48.1k rows`, so BOTH
#     corpus incident rows were suppressed by condition (a) alone — the
#     `(?:\s+[A-Za-z-]+){0,2}` unit-word tolerance is load-bearing.
#   - The citation window is BIDIRECTIONAL because cap vocabulary TRAILS
#     its figure (`<=100 GPU-h cap check below`, #1489 v3; #841's
#     `< 20 GPU-h cheap band` is the same trailing shape).
#   - Two disclosed divergences from the round-2 critic candidate, both
#     re-measured hit-set-neutral (plan #2177 §13): the backward set
#     RETAINS `cap\b` (strictly more suppressive); the forward set OMITS
#     bare `gate` (deliberate — including it would re-admit `pilot-gated`
#     as a back-door escape; see the non-escapes in the check docstring).
#   - A THIRD divergence, from plan §3.4's literal rather than the critic
#     candidate: `_C48_ABORT_RE` carries `(?i)` (admits `Abort`/`ABORT`),
#     which the plan's written pattern does not. Strictly widening,
#     verified hit-set-neutral by the same corpus sweep. Recorded because
#     this task's whole history is regex-fidelity drift — a future editor
#     diffing the shipped regexes against plan #2177 must be able to tell
#     a deliberate divergence from drift.
# Condition (a) is the ONLY shield holding ~50 citation-FPs out: 56 corpus
# rows sit >2x booked and window-clean and are suppressed by (a) alone
# (`#411 trained 11 sources ... in 12 GPU-h`-class citations whose citation
# token sits beyond the backward window). Do NOT weaken (a) in a
# simplification pass.
# Arm B was pre-calibrated by the critic's replay (2 hits corpus-wide,
# both copies of the #1336 EXT_off row — v16.md + the then-v16-pointing
# plan.md symlink; plan.md now points at the re-booked v17, so the
# as-shipped sweep realizes 1 arm-B file) and the `abort > N min|h /cell`
# idiom occurs in only 11 rows corpus-wide.

_C48_BASIS_BOOKED_RATIO = 2.0

# First bare number in a booked column cell (wall or gpu-h).
_C48_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

# GPU-hour TOKEN in a basis cell ("90 GPU-h", "7.6 GPU-hours", "12 gpu h");
# the lookbehind blocks mid-number starts ("48.1" inside "48.103").
_C48_GPU_H_RE = re.compile(r"(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*GPU-?\s?h(?:ours?|rs?)?\b")

# Condition (a) derivation context: a product expression ... then an arrow,
# both before the token. The `(?:\s+[A-Za-z-]+){0,2}` unit-word tolerance is
# what admits `20 cells × 48.1k rows` (see the calibration block).  # noqa: RUF003
_C48_PRODUCT_RE = re.compile(r"\d[\d.,kKmM]*(?:\s+[A-Za-z-]+){0,2}\s*[×x*]\s*\d")  # noqa: RUF001
_C48_ARROW_RE = re.compile(r"[⇒→≈~=]|=>")

# Condition (b) citation/cap exclusion — BIDIRECTIONAL window around the
# token (backward ~30 chars, forward ~25). The forward vocabulary
# deliberately EXCLUDES bare `gate` (it would re-admit `pilot-gated`).
_C48_CITE_BACK_CHARS = 30
_C48_CITE_FWD_CHARS = 25
_C48_CITE_BACK_RE = re.compile(
    r"(?i)#\d+|parent|realized|prior|previous|cap\b|within|inside|budget"
)
_C48_CITE_FWD_RE = re.compile(r"(?i)cap\b|band\b|rail\b|auto-approve")

# Row-scoped reconciliation allowlist (arm A) — every entry expresses an
# INTENT to reconcile. Deliberately NOT escapes (reasons in the check
# docstring): bare `naive-serial`, bare `N-way` / `÷ N` / `across N GPUs`,
# and `pilot-gated`.
_C48_RECONCILE_RE = re.compile(
    r"(?i)supersed\w*|reconcil\w*|upper[- ]bound|worst[- ]case|ceiling|(?:in|ex)cludes\s+\w+"
)
# Arm B's narrower allowlist (§3.4 step 5): supersession/reconciliation
# markers only — a "worst-case"-labelled derived figure says nothing about
# an abort threshold's coherence with the booking.
_C48_ARMB_RECONCILE_RE = re.compile(r"(?i)supersed\w*|reconcil\w*")

# Arm B parsers. The abort idiom: `abort > 30 min/cell` (`[^.;|]` keeps the
# scan inside one sentence/cell). n_cells tolerates one interposed word
# (`20 off-diagonal cells`); width reads the parallelism cell (`8 GPUs`).
_C48_ABORT_RE = re.compile(
    r"(?i)abort[^.;|]{0,40}?>\s*(\d+(?:\.\d+)?)\s*(min|minutes?|h|hours?)\s*/\s*cell"
)
_C48_CELLS_RE = re.compile(r"(\d+)\s*(?:\S+\s+)?cells\b")
_C48_WIDTH_RE = re.compile(r"(\d+)\s*(?:×\s*)?GPUs?\b")  # noqa: RUF001


def _c48_first_number(cell: str) -> float | None:
    """First bare number in a booked-column cell, else None (never guessed)."""
    m = _C48_NUM_RE.search(cell)
    return float(m.group(1)) if m else None


def _c48_derived_gpu_h(basis: str) -> float | None:
    """Max DERIVED GPU-hour figure in a basis cell, else None.

    A token qualifies only when (a) it is reached through the cell's own
    arithmetic — a product expression followed, before the token, by a
    derivation arrow — and (b) its bidirectional citation/cap window is
    clean. This is what distinguishes *the basis computing a figure* from
    *the basis citing one* (the v2 largest-token extraction measured 11.3%
    corpus WARN; see the calibration block)."""
    best: float | None = None
    for m in _C48_GPU_H_RE.finditer(basis):
        pre = basis[: m.start()]
        prod = _C48_PRODUCT_RE.search(pre)
        if prod is None:
            continue
        if not _C48_ARROW_RE.search(pre[prod.end() :]):
            continue
        if _C48_CITE_BACK_RE.search(basis[max(0, m.start() - _C48_CITE_BACK_CHARS) : m.start()]):
            continue
        if _C48_CITE_FWD_RE.search(basis[m.end() : m.end() + _C48_CITE_FWD_CHARS]):
            continue
        val = float(m.group(1))
        best = val if best is None else max(best, val)
    return best


def _c48_abort_threshold_h(row_text: str) -> float | None:
    """Stated per-cell abort threshold in HOURS, else None (never guessed)."""
    m = _C48_ABORT_RE.search(row_text)
    if not m:
        return None
    val = float(m.group(1))
    return val / 60.0 if m.group(2).lower().startswith("m") else val


def _c48_n_cells(row_text: str) -> int | None:
    """Max `N ... cells` count in the row (>= 2 required), else None."""
    counts = [int(m.group(1)) for m in _C48_CELLS_RE.finditer(row_text)]
    n = max(counts, default=0)
    return n if n >= 2 else None


def _c48_booked_per_cell_wall_h(row: _ComputeRow, n_cells: int) -> float | None:
    """Booked per-cell wall (hours) = planned_wall_h x width / n_cells;
    falls back to planned_gpu_h / n_cells when the wall column is
    unparseable (GPU-h = wall x width, so the two are the same quantity);
    None when neither column parses (the row is skipped, never guessed)."""
    width_m = _C48_WIDTH_RE.search(row.parallelism)
    width = int(width_m.group(1)) if width_m else 1
    wall = _c48_first_number(row.wall)
    if wall is not None:
        return wall * width / n_cells
    gpu_h = _c48_first_number(row.gpu_h) if row.gpu_h else None
    if gpu_h is not None:
        return gpu_h / n_cells
    return None


def _c48_offender_detail(offenders: list[str]) -> str:
    """Bounded WARN detail (the c26/c32 convention): at most 3 (row, arm)
    findings, the rule anchor, the incident anchors, and both remedies."""
    shown = "; ".join(offenders[:3])
    if len(offenders) > 3:
        shown += "; ..."
    return (
        f"{shown} — a §9 row's own basis arithmetic contradicts its booked columns: "
        "parallelism divides WALL time, never GPU-hours (GPU-h = wall x n_gpus), so a "
        "basis-derived GPU-h figure > 2x the booked planned_gpu_h with no reconciliation "
        "marker under-books by the parallelism width, and a stated per-cell abort threshold "
        "below the booked per-cell wall fires on a run performing exactly as booked "
        "(plan-compute-sizing.md § Per-cell fit phases; driving incident #1336 EXT_off: "
        "basis 90 GPU-h vs booked 30 + abort > 30 min/cell vs a booked ~91 min/cell, "
        "survived TWO approved plan versions; recurrence #823 / #811 / #1092). Remedies: "
        "state the reconciliation in the row (supersed*/reconcil*/upper-bound/worst-case/"
        "ceiling, or an includes/excludes scope note; for arm B raise the abort threshold "
        "or re-book the row), or declare `N/A — basis arithmetic reconciled` on its own "
        "line, unwrapped (no backticks/quotes)"
    )


def check_basis_booked_arithmetic(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional: a §9 compute-table row whose OWN basis text
    contradicts the row's OWN booked columns, in the c26/c32 family (same
    parser, same kind gate, same bounded-detail + standalone-N/A
    conventions). Arm A (GPU-h axis ONLY): a basis-DERIVED GPU-h figure —
    a unit-word-tolerant product expression followed by a derivation
    arrow, outside a bidirectional citation/cap window — exceeding
    ``_C48_BASIS_BOOKED_RATIO`` (2.0) x the booked gpu-h column, with no
    row-scoped reconciliation marker. Arm B (abort-vs-booked): a stated
    ``abort > N min|h /cell`` threshold strictly BELOW the per-cell wall
    the row's own booking implies (wall x width / n_cells; gpu_h / n_cells
    fallback). Mechanizes plan-compute-sizing.md § "Per-cell fit phases"
    coherence (#1336 `EXT_off` — basis 90 GPU-h vs booked 30 AND abort >
    30 min/cell vs a booked ~91 min/cell — survived two approved plan
    versions; recurrence #823/#811/#1092). NEVER FAILs in v1: both sides
    are heuristic regex reads of free prose, a legitimately-superseded
    naive-serial figure is a normal thing to show in a basis, and the
    reviewer lens stays binding (the c26/c32 disposition).

    Deliberate NON-escapes (each measured; plan #2177 §3.3):
      - bare ``naive-serial`` — a label, not a reconciliation; the driving
        row's own text. The escapable form is ``naive-serial (superseded
        by <x>)``, which ``supersed\\w*`` already covers.
      - bare ``N-way`` / ``÷ N`` / ``across N GPUs`` — parallelism divides
        wall, never GPU-h, so it can never reconcile a GPU-h-vs-GPU-h
        discrepancy (the axis restriction is what makes this principled:
        a substring allowlist on the task body's draft tokens would have
        suppressed the driving incident itself).
      - ``pilot-gated`` — a pilot DEFERS a contradiction, it does not
        reconcile one: the 100 GPU-h auto-approve rail binds at APPROVAL
        time, before any pilot runs, and the driving incident was itself
        pilot-gated. The forward citation window likewise excludes bare
        ``gate`` so ``pilot-gated`` cannot ride back in.

    Accepted gaps (documented, not fixed): the WALL axis is out of scope
    (a serial-equivalent figure legitimately precedes the parallelism
    division there; the driving row's 11.2-vs-3.8 wall contradiction is
    invisible by design — possible v2 work); a citation whose citing token
    sits > ~30 chars before its figure survives as a citation-FP (#634 v1,
    the single non-genuine sweep hit); combined ``Wall / GPU-h`` columns
    are skipped for Arm A (attribution unrecoverable — the under-WARN
    direction); a per-cell abort stated in prose OUTSIDE the table is
    invisible to Arm B; a cell that itself consumes k GPUs makes Arm B's
    per-cell wall an over-estimate by k — MORE warns, the OVER-WARN
    direction (corrected from v2's inverted claim; empirically nil today);
    a booking exactly AT its abort threshold does not fire (strict ``>``);
    and FABRICATED reconciliation vocabulary passes — a mechanical check
    cannot verify intent (a PASS here is never "arithmetic verified").

    Three disclosed divergences, all re-measured hit-set-neutral (plan
    #2177 §13). From the round-2 measured candidate: the backward window
    RETAINS ``cap\\b``; the forward window OMITS bare ``gate``. From plan
    §3.4's written literal: ``_C48_ABORT_RE`` carries ``(?i)``."""
    cid, name = "c48_basis_booked_arithmetic", "§9 basis-vs-booked arithmetic"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            "kind-exempt: compute-projection tables are an experiment|analysis plan shape",
        )
    rows = _compute_table_rows(plan)
    if not rows:
        return _skip(cid, name, "no compute-projection table with a `basis` column detected")
    if _standalone_na_declared(plan, r"basis arithmetic reconciled"):
        return _pass(cid, name, "explicit N/A declared (basis arithmetic reconciled)")
    offenders: list[str] = []
    n_considered = 0
    for row in rows:
        compared = False
        # Arm A — basis-derived GPU-h vs the booked gpu-h column. Skipped
        # when the gpu-h column is absent, combined with the wall column,
        # or unparseable/zero (never guessed).
        if row.gpu_h and not row.gpu_is_wall_col:
            booked = _c48_first_number(row.gpu_h)
            derived = _c48_derived_gpu_h(row.basis)
            if booked is not None and booked > 0 and derived is not None:
                compared = True
                if derived > _C48_BASIS_BOOKED_RATIO * booked and not _C48_RECONCILE_RE.search(
                    row.row_text
                ):
                    offenders.append(
                        f"row {row.component[:60]!r} [arm A] basis derives {derived:g} GPU-h "
                        f"vs booked {booked:g} ({derived / booked:.1f}x)"
                    )
        # Arm B — stated per-cell abort threshold vs the booked per-cell
        # wall. Skipped when the abort idiom, the cell count (>= 2) or
        # both booked columns fail to parse (never guessed).
        abort_h = _c48_abort_threshold_h(row.row_text)
        n_cells = _c48_n_cells(row.row_text)
        if abort_h is not None and n_cells is not None:
            per_cell = _c48_booked_per_cell_wall_h(row, n_cells)
            if per_cell is not None:
                compared = True
                if per_cell > abort_h and not _C48_ARMB_RECONCILE_RE.search(row.row_text):
                    offenders.append(
                        f"row {row.component[:60]!r} [arm B] booked per-cell wall "
                        f"~{per_cell * 60:.0f} min vs stated abort > {abort_h * 60:g} min/cell"
                    )
        if compared:
            n_considered += 1
    if offenders:
        return _warn(cid, name, _c48_offender_detail(offenders))
    if n_considered == 0:
        return _skip(
            cid,
            name,
            "basis table present but no row evaluable — no derivation-bearing GPU-h "
            "comparison (arm A) and no parseable per-cell abort threshold (arm B)",
        )
    return _pass(
        cid,
        name,
        f"{n_considered} row(s) evaluated; every basis-derived GPU-h figure is within "
        f"{_C48_BASIS_BOOKED_RATIO:g}x its booked planned_gpu_h and every stated per-cell "
        "abort threshold clears the booked per-cell wall",
    )


#: Trigger for c49 — keep in sync with
#: ``task_workflow._AUTH_STUB_HEADING_RE`` (trigger-only mirror; the PARSER
#: is shared via the lazy import below, so trigger drift is the only
#: possible divergence and c49 FAILs loud on it).
_C49_HEADING_RE = re.compile(r"^###\s+Authorized smoke stubs\b", re.IGNORECASE | re.MULTILINE)


def _c49_parser():
    """Lazy import of the SHARED authorized-stub block parser (#2171).

    The parser lives in ``explore_persona_space.task_workflow`` and is the
    SAME code the Step 6d.0 runtime grant (``task.py check-authorized-stub``)
    executes — zero plan-time/runtime drift by construction. Paid ONLY when
    the c49 heading trigger fires. ImportError → the caller SKIPs loud naming
    the cause (the c41 off-repo doctrine: ``--plan-file`` off-repo must never
    crash).
    """
    try:
        from explore_persona_space.task_workflow import (
            AuthorizedStubBlockError,
            parse_authorized_stub_block,
        )
    except ImportError as exc:  # off-repo --plan-file run
        return None, None, str(exc)
    return parse_authorized_stub_block, AuthorizedStubBlockError, ""


def check_authorized_stub_block(plan: str, kind: str) -> CheckResult:
    """FAIL, conditional, all kinds: a PRESENT '### Authorized smoke stubs'
    block must parse — one markdown-table data row per arm with a backticked
    arm token in column 1 and non-empty impossibility-reason + compensating-
    control cells, exactly one heading occurrence (#2171; the #2163 unwired-
    escape incident).

    Trigger: any line matching ``_C49_HEADING_RE``. Absent → SKIP. Present →
    lazy-import ``parse_authorized_stub_block`` from
    ``explore_persona_space.task_workflow`` (the c34 lazy-import idiom;
    ImportError → loud SKIP naming the cause). Well-formed → PASS naming the
    arms; ``AuthorizedStubBlockError`` → FAIL with the parser's message +
    remedy. FAIL (not the WARN doctrine) is deliberate: the trigger is an
    exact heading, the parser is the SAME code the runtime grant uses (zero
    heuristic gap), and a malformed block otherwise refuses mechanically at
    Step 6d.0 AFTER pod provisioning — plan-time is strictly cheaper.

    Disclosed under-trigger: a plan naming ``PASS_AUTHORIZED_STUB`` with NO
    block heading does not trigger c49 (quoting-a-sibling residual class, per
    the c41 incident-citation precedent); the runtime checker refuses that
    shape at Step 6d.0.
    """
    del kind  # all kinds: an authorized-stub block parses identically everywhere
    cid, name = "c49_authorized_stub_block", "authorized-smoke-stubs block well-formed"
    if not _C49_HEADING_RE.search(plan):
        return _skip(cid, name, "no authorized-smoke-stubs block declared")
    parse_fn, err_cls, load_detail = _c49_parser()
    if parse_fn is None:
        return _skip(
            cid,
            name,
            f"task_workflow parser unavailable ({load_detail}) — off-repo --plan-file run",
        )
    try:
        block = parse_fn(plan)
    except err_cls as exc:
        return _fail(
            cid,
            name,
            f"malformed '### Authorized smoke stubs' block: {exc} — fix the block "
            "(one table row per arm: backticked arm | non-empty impossibility reason | "
            "non-empty compensating control) BEFORE approval; a malformed block "
            "otherwise refuses mechanically at Step 6d.0 AFTER pod provisioning "
            "(task.py check-authorized-stub, #2171)",
        )
    if block is None:
        return _fail(
            cid,
            name,
            "trigger/parser drift: _C49_HEADING_RE matched but the shared parser "
            "found no block heading — re-align _C49_HEADING_RE with "
            "task_workflow._AUTH_STUB_HEADING_RE",
        )
    arms = ", ".join(f"`{a}`" for a in sorted(block))
    return _pass(
        cid,
        name,
        f"block parses: {len(block)} authorized arm(s) ({arms}), impossibility "
        "reason + compensating control non-empty per row",
    )


# ─── Check 50 — §9 projected wall vs the intent's SLURM --time bin (#2027) ──

_c50_reachable_cache: list = []  # [(fn | None, detail)] once resolved


def _c50_slurm_lane_reachable_fn():
    """Lazily path-load ``scripts/dispatch_issue.py`` (the c46 idiom:
    stdlib-only module-level imports; NOT registered in ``sys.modules``)
    and return its RUNTIME reachability predicate
    ``_slurm_lane_reachable`` — reused verbatim for exact parity with the
    #2161 exit-2 refusal (a divergent second plan-time reachability rule
    is worse than no check; #2027 kill criterion). Returns ``(fn, "")``
    on success or ``(None, <detail>)`` on ANY load failure (file absent
    on off-repo ``--plan-file`` runs, a pre-#2161 checkout without the
    predicate) — the caller SKIPs loudly on ``None``. Cached one-shot;
    env sensitivity is unaffected (``auto_lane_order()`` reads
    ``EPM_AUTO_LANE_ORDER`` at CALL time, not load time).
    """
    if not _c50_reachable_cache:
        fn, detail = None, ""
        try:
            if not _C46_DISPATCH_CLI_PATH.is_file():
                raise FileNotFoundError(_C46_DISPATCH_CLI_PATH)
            spec = importlib.util.spec_from_file_location(
                "_c50_dispatch_issue", _C46_DISPATCH_CLI_PATH
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = mod._slurm_lane_reachable
        except Exception as exc:  # any load failure -> loud SKIP at the caller
            fn, detail = None, f"{type(exc).__name__}: {exc}"
        _c50_reachable_cache.append((fn, detail))
    return _c50_reachable_cache[0]


def _c50_launch_argvs(plan: str) -> tuple[list[list[str]], list[str]]:
    """DISTINCT launch-shaped dispatch argvs in ``plan`` (+ skip notes).

    Reuses the c46 candidate/tokenize/argv chain verbatim; keeps only
    launch-shaped argvs (explicit ``launch`` subcommand, or the #1336
    leads-with-a-flag shape — the ``_c46_drift_arms`` test), DEDUPED on
    ``tuple(argv)`` preserving first-seen order. The dedupe is
    load-bearing, not cosmetic: 27 of the 170 multi-launch corpus plans
    (2026-08 sweep) restate ONE command verbatim in §4 and §9 — one real
    dispatch, recovered at zero false-positive cost, since identical
    argvs resolve an identical intent and bin.
    """
    argvs: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    notes: list[str] = []
    for cmd in _c46_command_candidates(plan):
        cleaned = _C46_COND_EXPANSION_RE.sub("", cmd)
        try:
            tokens = shlex.split(cleaned)
        except ValueError as exc:
            notes.append(f"unsplittable line skipped ({exc}): {cmd[:60]!r}")
            continue
        argv = _c46_argv_from_tokens(tokens)
        if argv is None:
            continue  # bare file reference / prose mention, not a command
        if not (argv[0] == "launch" or argv[0].startswith("-")):
            continue  # finalize/poll command — never carries a wall to fence
        key = tuple(argv)
        if key in seen:
            continue
        seen.add(key)
        argvs.append(argv)
    return argvs, notes


def _c50_section9_walls(plan: str) -> list[tuple[float, str]]:
    """Parseable §9 ``planned_wall_h`` reads as ``(value, component)`` pairs.

    Combined ``Wall / GPU-h`` header rows are skipped for the wall read
    (``gpu_is_wall_col`` — the c47 #2177 precedent); no-float cells are
    dropped (c47 owns warning on those).
    """
    walls: list[tuple[float, str]] = []
    for row in _compute_table_rows(plan):
        if row.gpu_is_wall_col:
            continue
        val = _c48_first_number(row.wall)
        if val is not None:
            walls.append((val, row.component.strip()))
    return walls


def check_plan_wall_vs_slurm_time_bin(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: when the plan embeds EXACTLY ONE
    DISTINCT launch-shaped ``dispatch_issue.py`` command (deduped on
    ``tuple(argv)``) that dry-parses, declares NO ``--time-budget-hours``,
    resolves a SLURM-reachable route
    (``dispatch_issue._slurm_lane_reachable`` — the #2161 RUNTIME
    predicate, reused for exact parity), and names an intent in
    ``slurm._DEFAULT_TIME_BUDGETS_HOURS``, the §9 MAX parseable
    ``planned_wall_h`` must not EXCEED that intent's default ``--time``
    bin — strictly greater WARNs; equality passes (the repo's own sizing
    style is an explicit in-table margin, e.g. ``ft-7b: 23.5`` under the
    24 h bin — never a tuned margin constant here). Mechanizes the #2027
    arm-2 gap: sbatch ``--time`` silently becomes the intent default
    (lora-7b -> 6.0 h) and a 12 h projected run TIMEOUTs mid-flight — the
    #1336 shape reached WITHOUT ``--max-run-duration``, which is exactly
    why the runtime ``max_run_duration_slurm_inert_without_time_budget``
    refusal (and its c46 arm-2 plan-time twin) is structurally blind to
    it. Every ambiguity SKIPs with a stated reason; the check NEVER FAILs
    (the c46/c47 posture — a heuristic wall/bin join must not become the
    #1388 fleet-wedge shape).

    Named accepted FALSE NEGATIVES: (a) multi-dispatch plans SKIP — with
    >=2 DISTINCT launch commands the wall-row <-> dispatch join is
    ambiguous (a big row may belong to a differently-budgeted dispatch)
    and a MAX-over-all-rows read would false-fire; (b)
    ``max_wall_h == bin_h`` passes (strictly-greater; no tuned margin);
    (c) an intent absent from ``_DEFAULT_TIME_BUDGETS_HOURS`` SKIPs
    (``slurm.time_budget_hours`` already raises there — a different,
    already-fail-fast failure); (d) combined ``Wall / GPU-h`` header rows
    are skipped for the wall read (``gpu_is_wall_col`` — the c47 #2177
    precedent); (e) frontmatter ``backend:`` pins are invisible to the
    argv-only reachability read — deliberate exact parity with the
    runtime guard's own ``args.backend`` semantics. Named residual FALSE
    POSITIVE (0 of the 46 conjunct-passing plans in the 2026-08 corpus
    sweep): a single-dispatch plan whose MAX §9 row is an off-pod /
    VM-side analysis row exceeding the bin while the DISPATCHED job's own
    wall sits under it — WARN-only polarity absorbs it. Corpus
    calibration (5,244 plans, 2026-08): 359 exactly-one-launch plans, 46
    passing all six conjuncts, 5 WARN hits (~0.1%) — #1345 v7-v9 + #597
    v4-v5, every one a true #1336-shaped positive.
    """
    del kind  # all kinds: a wall/bin mismatch times out identically everywhere
    cid, name = "c50_plan_wall_vs_slurm_time_bin", "§9 projected wall vs SLURM --time bin"
    argvs, notes = _c50_launch_argvs(plan)
    tail = ("; " + "; ".join(notes)) if notes else ""
    if not argvs:
        return _skip(cid, name, f"no launch-shaped dispatch_issue.py command in the plan{tail}")
    if len(argvs) > 1:
        return _skip(
            cid,
            name,
            f"{len(argvs)} DISTINCT launch commands — the wall-row <-> dispatch join is "
            f"ambiguous (documented false negative; a per-phase join is a follow-up){tail}",
        )
    parser, load_detail = _c46_argparser()
    if parser is None:
        return _skip(cid, name, f"dispatch_issue.build_argparser unavailable ({load_detail})")
    argv = argvs[0]
    ns, err = _c46_dry_parse(parser, argv)
    if ns is None:
        return _skip(cid, name, f"launch argv does not parse ({err}) — c46 arm 1 owns that")
    if _c46_has_flag(argv, "--time-budget-hours"):
        return _skip(cid, name, "--time-budget-hours declared — the SLURM --time fence is explicit")
    reachable_fn, reach_detail = _c50_slurm_lane_reachable_fn()
    if reachable_fn is None:
        return _skip(
            cid, name, f"dispatch_issue._slurm_lane_reachable unavailable ({reach_detail})"
        )
    try:
        reachable = reachable_fn(ns)
    except Exception as exc:  # router import failure on off-repo runs -> loud SKIP
        return _skip(cid, name, f"SLURM reachability unresolvable ({type(exc).__name__}: {exc})")
    if not reachable:
        return _skip(
            cid,
            name,
            f"no SLURM lane reachable for backend {(ns.backend or 'auto')!r} — "
            "the sbatch --time bin never binds",
        )
    try:
        from explore_persona_space.backends.slurm import _DEFAULT_TIME_BUDGETS_HOURS
    except ImportError as exc:  # off-repo --plan-file run
        return _skip(cid, name, f"slurm intent-default table unavailable ({exc})")
    intent = str(getattr(ns, "intent", ""))
    if intent not in _DEFAULT_TIME_BUDGETS_HOURS:
        return _skip(
            cid,
            name,
            f"intent {intent!r} has no _DEFAULT_TIME_BUDGETS_HOURS row — "
            "slurm.time_budget_hours() already fails fast at dispatch",
        )
    bin_h = _DEFAULT_TIME_BUDGETS_HOURS[intent]
    walls = _c50_section9_walls(plan)
    if not walls:
        return _skip(cid, name, "no parseable §9 planned_wall_h row — nothing to compare")
    max_wall, comp = max(walls, key=lambda t: t[0])
    if max_wall > bin_h:
        return _warn(
            cid,
            name,
            f"§9 projects max planned_wall_h {max_wall:g} h (row {comp[:60]!r}) but the plan's "
            f"single launch command resolves intent {intent!r} to the SLURM --time default of "
            f"{bin_h:g} h with no --time-budget-hours — on a SLURM lane sbatch --time is set "
            f"to {bin_h:g} h and the job TIMEOUTs mid-run (the #1336 shape, reached WITHOUT "
            "--max-run-duration, so the runtime "
            "max_run_duration_slurm_inert_without_time_budget refusal never fires): pass "
            f"--time-budget-hours >= {max_wall:g} on the launch command, or pin a non-SLURM "
            "backend",
        )
    return _pass(
        cid,
        name,
        f"max §9 planned_wall_h {max_wall:g} h fits intent {intent!r}'s SLURM --time "
        f"default of {bin_h:g} h",
    )


# ─── Check 51 — edited workflow-surface literal pin-test coverage ──────────

# Extraction dials (heuristic constants, not model hyperparameters — #2029
# plan §11; each value below was landed by the calibration sweep recorded
# after this block). MIN_LEN keeps the 28-char incident literal while
# dropping short vocabulary; the span cap bounds worst-case concat scans;
# the anchored cap bounds offender-scan work on pathological plans.
_C51_MIN_LEN = 12
_C51_MAX_SPAN_CLASS = 10
_C51_MAX_ANCHORED = 30
_C51_MAX_SURFACE_FILES = 2
_C51_MAX_TEST_FILES = 3
_C51_EDIT_PROX_CHARS = 120  # the c31 proximity window, copied not shared
_C51_FENCE_ADJ_LINES = 3  # a fenced block within 3 lines joins its paragraph

# Workflow-surface path token: .claude/{skills,agents,rules,hooks}/... paths,
# slash-prefixed or bare SKILL.md, bare CLAUDE.md, .claude/workflow.yaml.
_C51_PATH_RE = re.compile(
    r"\.claude/(?:skills|agents|rules|hooks)/[\w./-]+"
    r"|(?:[\w.-]+/)*SKILL\.md|(?<![\w./-])SKILL\.md"
    r"|(?<![\w./-])CLAUDE\.md|\.claude/workflow\.yaml"
)
# c31's verb set widened with the #1948 plan's own edit verbs (widen / swap /
# replace / update / teach + "new ... pattern"). COPIED, not shared: c31's
# calibration comment mandates a corpus re-scan on any change to ITS regexes
# (#1557), so c51 owns separate constants.
_C51_EDIT_RE = re.compile(
    r"(?i)\b(?:add(?:s|ed|ing)?|insert\w*|append\w*|amend\w*|edit\w*|splice\w*"
    r"|prepend\w*|reword\w*|rewrit\w*|revise[sd]?|patch\w*|widen\w*|swap\w*"
    r"|replac\w*|updat\w*|teach\w*"
    r"|new (?:section|paragraph|bullet|sentence|step|clause|line|pattern))\b"
)
# c31's negation guard, copied for the same no-shared-calibration reason
# (minus c31's plan_pending arm, which is durability-pin-specific).
_C51_NEG_RE = re.compile(
    r"(?i)\b(?:no|zero|not?|without|never)\b(?:[^|;:.]|\.(?!\s)){0,24}"
    r"\b(?:edit(?:s|ed|ing)?|chang(?:e|es|ed))\b"
    r"|\bunchanged\b|\bincidental\b|must-ask|must bounce"
)
# Removal-context tier — REQUIRED for incident recall: the #1948 old literal
# is quoted only in a "neither carries the bare legacy X" paragraph, which
# carries removal vocabulary but no edit verb (#2029 plan §6.1 replay).
_C51_REMOVAL_RE = re.compile(
    r"(?i)\b(?:legacy|old|removed?|dropp\w*|no longer|neither|banned|retired"
    r"|deprecated|obsolete|stale)\b"
)
_C51_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_C51_SQUOTED_RE = re.compile(rf"'([^'\n]{{{_C51_MIN_LEN},}})'")
_C51_DQUOTED_RE = re.compile(rf'"([^"\n]{{{_C51_MIN_LEN},}})"')
# A top-level list-item line starts its OWN paragraph: consecutive numbered /
# bulleted items with no blank separator otherwise fuse, letting item A's
# path token license item B's literal — the #2029 plan's own §6 items 1+3
# fused exactly so (item 1's issue/SKILL.md target + item 3's cited
# vocabulary-FP example) and self-WARNed the prototype; item-grain binding
# is the tighter form of the design's paragraph-level target binding.
_C51_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,3}[.)])\s")
_C51_FLAG_RE = re.compile(r"^--[\w=-]+$")
_C51_IDENT_RE = re.compile(r"^[A-Za-z_][\w.]*$")
_C51_STRIP_CHARS = "`'\"()[],:;"
# A tests/ file counts as a workflow-surface pin test only when it visibly
# reads the surface — a code test that never opens a surface file cannot
# break from a prose edit.
_C51_SURFACE_PIN_MARKERS = (
    "SKILL.md",
    ".claude/skills",
    ".claude/agents",
    ".claude/rules",
    ".claude/hooks",
    "CLAUDE.md",
    "workflow.yaml",
)
_C51_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c31/c34/c41 pattern)
# Lazy corpus cache; every entry is keyed on the resolved repo root (or an
# absolute file path), so monkeypatched fixture roots stay disjoint. Tests
# additionally clear it in fixtures.
_c51_cache: dict = {}

# Calibration (#2029; prototype /tmp/c50_proto4.py at the final design,
# restated in plan §4.1): corpus = tasks/*/*/plans/v*.md (3,713 files at the
# 2026-08-08 sweep; kind read from each task's body.md). Design iterations
# measured 1,848 -> 833 -> 437 -> 393 WARN / 913 PASS / 2,407 SKIP as the
# discriminators landed (kind gate + paragraph binding + reference/flag/
# identifier drops + rarity caps + removal tier + quoted-pin-line +
# surface-marker filter); prototype fire rate 393/3,713 = 10.6% — the c31
# #1557 band (WARN on 328/3,024 ~ 11%). Landed-implementation re-scan
# (2026-08-08, this file, 3,714-file corpus): 302 WARN / 816 PASS /
# 2,596 SKIP — fire rate 8.1%, within the plan's ~1.5x acceptance bound of
# the prototype's 393; the decrease is the landed ITEM-GRAIN paragraph
# binding (_C51_LIST_ITEM_RE — consecutive list items no longer fuse, so a
# sibling item's path token cannot license a literal; the same tightening
# cleared the #2029 self-application, where the prototype fused plan §6
# items 1+3 and false-offended on their cited vocabulary-FP example).
# Stratified WARN classification (36 sampled WARN plans, one per (offending
# test file x literal shape) stratum; #2029 plan §6 item iii): 12 genuine
# under-listing / 20 citation-context FP / 4 vocabulary FP — both TP and FP
# classes non-empty, consistent with the plan's n=10 hand inspection; 8.1%
# is a FIRE rate, not an FP rate, and the mixed composition is why this
# check is WARN-only. Recall spot-check under the LANDED grammar against the
# three genuine instances plan §6.3 named: #1032 (v1-v3) still WARNs
# (tests/test_adversarial_planner_warn_disposition.py) and #1138 (v1-v3)
# still WARNs (tests/test_issue_skill_bare_push_snippets_pin.py), so the
# item-grain tightening did not cost the motivating class; #815 (v1/v2) now
# PASSes — a prototype-era genuine hit LOST to the re-tune (or to corpus
# drift; not discriminated). Accepted for a WARN-only check with the
# disclosed accepted-FN list, and recorded here so the next regex change has
# the datum. Post-minors re-scan (2026-08-08, after the _c51_within_root
# traversal guard): 303 WARN / 3,717 plan files = 8.2%, with the three recall
# probes unchanged — the +1 WARN tracks the +3 plan files that landed between
# scans, so the guard is behavior-neutral on the live corpus (PASS/SKIP were
# not re-split; WARN is the load-bearing count).
# ANY change to the _C51_* regexes or caps re-runs the
# corpus scan and updates these recorded numbers (the c31 #1557 / c27 / c32
# precedent).


def _c51_read(path: Path) -> str:
    """Cached whole-file read (errors="replace" — hooks may be non-UTF-8)."""
    key = ("file", str(path))
    if key not in _c51_cache:
        _c51_cache[key] = path.read_text(encoding="utf-8", errors="replace")
    return _c51_cache[key]


def _c51_trigger_line(plan: str) -> str | None:
    """First non-fenced, non-negated line carrying a workflow-surface path
    token with an edit verb within +/-``_C51_EDIT_PROX_CHARS`` of the path
    match (the c31 trigger shape over the wider c51 path + verb sets)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or _C51_NEG_RE.search(line):
            continue
        for m in _C51_PATH_RE.finditer(line):
            lo = max(0, m.start() - _C51_EDIT_PROX_CHARS)
            hi = min(len(line), m.end() + _C51_EDIT_PROX_CHARS)
            if _C51_EDIT_RE.search(line[lo:hi]):
                return line.strip()
    return None


def _c51_token_tails(text: str) -> list[str]:
    """Final path components of slash-bearing whitespace tokens — tests pin
    basenames while plans quote /tmp/-prefixed forms (the #1948 shape)."""
    out: list[str] = []
    for tok in text.split():
        tok = tok.strip(_C51_STRIP_CHARS)
        if "/" in tok:
            tail = tok.rstrip("/").rsplit("/", 1)[-1]
            if len(tail) >= _C51_MIN_LEN:
                out.append(tail)
    return out


def _c51_quoted_spans(text: str) -> list[str]:
    """Single- and double-quoted spans of at least ``_C51_MIN_LEN`` chars."""
    return _C51_SQUOTED_RE.findall(text) + _C51_DQUOTED_RE.findall(text)


def _c51_surface_paths(root: Path) -> list[Path]:
    """Live workflow-surface prose files under ``root`` (cached)."""
    key = (str(root), "surface_paths")
    if key not in _c51_cache:
        paths: list[Path] = []
        for pat in (
            ".claude/skills/**/*.md",
            ".claude/agents/**/*.md",
            ".claude/rules/*.md",
            ".claude/hooks/*",
        ):
            paths += [p for p in root.glob(pat) if p.is_file()]
        for p in (root / "CLAUDE.md", root / ".claude" / "workflow.yaml"):
            if p.is_file():
                paths.append(p)
        _c51_cache[key] = paths
    return _c51_cache[key]


def _c51_concat_index(paths: list[Path]) -> tuple[str, list[int]]:
    """(concatenated corpus text, per-file start offsets) with an unmatchable
    3-char separator, for offset-indexed distinct-file substring counting."""
    parts: list[str] = []
    offsets: list[int] = []
    pos = 0
    for p in paths:
        text = _c51_read(p)
        offsets.append(pos)
        parts.append(text)
        pos += len(text) + 3
    return "\n\x00\n".join(parts), offsets


def _c51_surface_union(root: Path):
    """(concat, offsets, paths, stripped-line set, token+tail set) over the
    surface corpus — the cheap union structures every candidate class is
    pre-screened against before any per-file anchor read."""
    key = (str(root), "union")
    if key not in _c51_cache:
        paths = _c51_surface_paths(root)
        lineset: set[str] = set()
        tokset: set[str] = set()
        for p in paths:
            text = _c51_read(p)
            for line in text.splitlines():
                lineset.add(line.strip())
            for tok in text.split():
                tok = tok.strip(_C51_STRIP_CHARS)
                if len(tok) >= _C51_MIN_LEN:
                    tokset.add(tok)
                    if "/" in tok:
                        tail = tok.rstrip("/").rsplit("/", 1)[-1]
                        if len(tail) >= _C51_MIN_LEN:
                            tokset.add(tail)
        concat, offsets = _c51_concat_index(paths)
        _c51_cache[key] = (concat, offsets, paths, lineset, tokset)
    return _c51_cache[key]


def _c51_tests_corpus(root: Path):
    """(concat, offsets, paths) over ``tests/**/*.py`` (cached)."""
    key = (str(root), "tests")
    if key not in _c51_cache:
        paths = sorted((root / "tests").glob("**/*.py"))
        concat, offsets = _c51_concat_index(paths)
        _c51_cache[key] = (concat, offsets, paths)
    return _c51_cache[key]


def _c51_repo_basenames(root: Path) -> set[str]:
    """Basenames of repo files a plan can cite as bare file REFERENCES."""
    key = (str(root), "basenames")
    if key not in _c51_cache:
        names: set[str] = set()
        for pat in (
            ".claude/skills/**/*",
            ".claude/agents/**/*",
            ".claude/rules/*",
            ".claude/hooks/*",
            "scripts/*",
            "tests/**/*",
            "src/explore_persona_space/**/*.py",
        ):
            names.update(p.name for p in root.glob(pat))
        _c51_cache[key] = names
    return _c51_cache[key]


def _c51_distinct_file_hits(
    cand: str, concat: str, offsets: list[int], paths: list[Path], cap: int
) -> list[Path]:
    """Distinct files whose text contains ``cand``, early-exiting once
    ``cap + 1`` are seen (rarity decisions never need the full count)."""
    import bisect

    hits: list[Path] = []
    i = concat.find(cand)
    while i != -1:
        fi = bisect.bisect_right(offsets, i) - 1
        if not hits or hits[-1] != paths[fi]:
            hits.append(paths[fi])
            if len(hits) > cap:
                return hits
        nxt = offsets[fi + 1] if fi + 1 < len(offsets) else len(concat)
        i = concat.find(cand, nxt)
    return hits


def _c51_within_root(root: Path, p: Path) -> bool:
    """True when ``p`` resolves INSIDE ``root``.

    ``_C51_PATH_RE``'s ``[\\w./-]+`` character class admits ``..``, so a
    plan-quoted token like ``.claude/skills/../../<path>`` would otherwise
    resolve outside the repo and be read during the membership checks below.
    Returns False on an unresolvable path (OSError) — conservative.
    """
    try:
        return p.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _c51_resolve_targets(root: Path, tokens: list[str]) -> list[Path]:
    """Existing surface files a paragraph's path tokens resolve to (bare
    ``SKILL.md`` resolves to every ``.claude/skills/*/SKILL.md``).

    Tokens resolving outside ``root`` are dropped (``_c51_within_root``)."""
    key = (str(root), "resolve", tuple(sorted(set(tokens))))
    if key not in _c51_cache:
        out: list[Path] = []
        for tok in set(tokens):
            tok = tok.strip(_C51_STRIP_CHARS)
            if tok.endswith("SKILL.md") and not tok.startswith(".claude/"):
                if "/" in tok:
                    cands = [root / ".claude" / "skills" / tok]
                else:
                    cands = list(root.glob(".claude/skills/*/SKILL.md"))
            elif tok == "CLAUDE.md":
                cands = [root / "CLAUDE.md"]
            else:
                cands = [root / tok]
            out += [p for p in cands if p.is_file() and _c51_within_root(root, p)]
        _c51_cache[key] = out
    return _c51_cache[key]


def _c51_is_reference(cand: str, root: Path) -> bool:
    """True when ``cand`` is a file/tool REFERENCE rather than a prose
    literal: glob-shaped, a CLI flag, a bare code identifier (pinned via its
    code file, not the prose), a repo-file basename, an existing repo path,
    or pathological (> 180 chars / embedded newline / escapes the repo root)."""
    t = cand.strip().strip(_C51_STRIP_CHARS)
    if len(t) > 180 or "\n" in t or "*" in t:
        return True
    if _C51_FLAG_RE.match(t) or _C51_IDENT_RE.match(t):
        return True
    if t in _c51_repo_basenames(root):
        return True
    if "/" in t or t.endswith(".py") or t.endswith(".md"):
        # A path-shaped token escaping the repo root is pathological, not a
        # prose literal — drop it WITHOUT reading it (traversal hardening).
        if not _c51_within_root(root, root / t):
            return True
        try:
            if (root / t).exists():
                return True
        except OSError:
            return True  # un-stattable candidate (embedded NUL etc.) — not a literal
    return False


def _c51_pin_line_quoted(cand: str, test_path: Path) -> bool:
    """True when some line of ``test_path`` containing ``cand`` also carries
    a quote char — the literal is pinned in test CODE, not merely cited in a
    comment/docstring narrative line."""
    for line in _c51_read(test_path).splitlines():
        if cand in line and ('"' in line or "'" in line):
            return True
    return False


def _c51_paragraph_spans(lines: list[str], mask: list[bool]) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` line spans of paragraph runs: a fence or
    blank line closes the current run, and a top-level list-item line closes
    it AND starts its own run (item-grain target binding — see
    ``_C51_LIST_ITEM_RE``); an item's indented continuation lines stay with
    it."""
    paras: list[tuple[int, int]] = []
    start: int | None = None
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not line.strip():
            if start is not None:
                paras.append((start, i - 1))
                start = None
        elif _C51_LIST_ITEM_RE.match(line):
            if start is not None:
                paras.append((start, i - 1))
            start = i
        elif start is None:
            start = i
    if start is not None:
        paras.append((start, len(lines) - 1))
    return paras


def _c51_adopted_fence_lines(lines: list[str], end: int) -> list[str]:
    """Stripped body lines of the first fenced block opening within
    ``_C51_FENCE_ADJ_LINES`` lines after a paragraph's ``end`` (blank lines
    skipped; any other prose stops the adoption)."""
    j = end + 1
    while j < len(lines) and j <= end + _C51_FENCE_ADJ_LINES:
        s = lines[j].strip()
        if s.startswith(("```", "~~~")):
            out: list[str] = []
            k = j + 1
            while k < len(lines) and not lines[k].strip().startswith(("```", "~~~")):
                out.append(lines[k].strip())
                k += 1
            return out
        if s:
            return []
        j += 1
    return []


def _c51_paragraph_candidates(plan: str, root: Path):
    """Yield ``(targets, line_class, token_class, span_class)`` per admitted
    paragraph: a blank-line-delimited non-fenced run carrying a surface path
    token AND edit-or-removal context, plus one fenced block opening within
    ``_C51_FENCE_ADJ_LINES`` lines after it (its lines join the paragraph's
    candidate pool)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for a, b in _c51_paragraph_spans(lines, mask):
        text = "\n".join(lines[a : b + 1])
        toks = _C51_PATH_RE.findall(text)
        if not toks:
            continue
        if not (_C51_EDIT_RE.search(text) or _C51_REMOVAL_RE.search(text)):
            continue  # cites a surface file with neither edit nor removal context
        targets = _c51_resolve_targets(root, toks)
        if not targets:
            continue
        spans = [
            c
            for c in _C51_INLINE_CODE_RE.findall(text) + _c51_quoted_spans(text)
            if len(c) >= _C51_MIN_LEN
        ]
        line_class: list[str] = []
        token_class: list[str] = list(_c51_token_tails(text))
        span_class: list[str] = []
        for c in spans:
            (token_class if " " not in c else span_class).append(c)
        for fenced_line in _c51_adopted_fence_lines(lines, b):
            if len(fenced_line) < _C51_MIN_LEN:
                continue
            line_class.append(fenced_line)
            token_class += _c51_token_tails(fenced_line)
            for c in _c51_quoted_spans(fenced_line):
                (token_class if " " not in c else span_class).append(c)
        yield targets, line_class, token_class, span_class


def _c51_anchored_candidates(plan: str, root: Path) -> list[str]:
    """Deduped candidates that survive the reference drops AND anchor
    verbatim in a surface file their OWN paragraph names (cheap union
    pre-screen first, per-paragraph target verify second)."""
    concat, _offsets, _paths, lineset, tokset = _c51_surface_union(root)
    anchored: list[str] = []
    span_budget = _C51_MAX_SPAN_CLASS
    seen: set[str] = set()
    for targets, line_class, token_class, span_class in _c51_paragraph_candidates(plan, root):
        pre = [c for c in line_class if c in lineset]
        for c in token_class:
            c = c.strip().strip(_C51_STRIP_CHARS)
            if len(c) >= _C51_MIN_LEN and c in tokset:
                pre.append(c)
        for c in span_class:
            if span_budget <= 0:
                break
            span_budget -= 1
            if c in concat:
                pre.append(c)
        for c in pre:
            if c in seen or _c51_is_reference(c, root):
                continue
            seen.add(c)
            if any(c in _c51_read(t) for t in targets):
                anchored.append(c)
    return anchored[:_C51_MAX_ANCHORED]


def _c51_offender_items(plan: str, rare: list[str], root: Path) -> list[tuple[str, str]]:
    """Sorted ``(tests/<file>, literal)`` offender rows: each pinning
    tests/ file (quoted pin line + surface marker present) whose basename
    appears nowhere in the raw plan (the c31 RAW-scan satisfier)."""
    t_concat, t_offsets, t_paths = _c51_tests_corpus(root)
    offenders: dict[str, str] = {}
    for c in rare:
        pins = _c51_distinct_file_hits(c, t_concat, t_offsets, t_paths, _C51_MAX_TEST_FILES)
        if not pins or len(pins) > _C51_MAX_TEST_FILES:
            continue  # unpinned, or > cap distinct test files = vocabulary (disclosed FN)
        for tp in pins:
            if tp.name in plan or not _c51_pin_line_quoted(c, tp):
                continue
            if not any(mk in _c51_read(tp) for mk in _C51_SURFACE_PIN_MARKERS):
                continue  # a code test, not a workflow-surface pin test
            offenders.setdefault(str(tp.relative_to(root)), c)
    return sorted(offenders.items())


def check_edited_literal_pin_tests(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, infra + batch only: when a plan declares an
    edit to an EXISTING workflow-surface literal (a ``.claude/{skills,agents,
    rules,hooks}`` / SKILL.md / CLAUDE.md / workflow.yaml string), every
    ``tests/`` file that already pins that literal verbatim must be named
    somewhere in the raw plan (the c31 RAW-scan satisfier convention) — an
    unlisted pin makes the plan's own Step 9c exit-0 acceptance criterion
    deterministically unsatisfiable as scoped (incident #1948: plan v2
    edited the SKILL.md inline-gate single-flight probe pattern without
    listing tests/test_issue_skill_gate_single_flight.py, whose asserts
    pinned the old literal; only the Claude critic caught it). The existing
    pin logic (c31, its #1557 arm, the pin-form satisfiers) runs the INVERSE
    direction — a plan must declare a pin for NEW prose — and does not
    overlap. Candidates come from admitted paragraphs (surface path token +
    edit-or-removal context; a top-level list item is its OWN paragraph, so
    one item's path token never licenses a sibling item's literal —
    item-grain target binding): stripped fenced lines, single-token
    inline/quoted spans + slash-token path tails, and capped multi-token
    spans; reference shapes (globs, CLI flags, bare code identifiers,
    repo-file basenames, existing repo paths) are dropped; a candidate must
    anchor verbatim in a surface file its OWN paragraph names; rarity caps
    drop surface boilerplate (> ``_C51_MAX_SURFACE_FILES`` surface files)
    and test vocabulary (> ``_C51_MAX_TEST_FILES`` test files), and the
    pinning line must carry a quote char in a test file that visibly reads
    the workflow surface. Accepted false negatives (disclosed by design):
    multi-line pin strings built by implicit concatenation in tests;
    ``not in`` pins broken by ADDING text; plans that elide the old literal
    entirely ("(~L7793) -> NEW"); a path token 4+ lines above its fenced
    block; and the > 3-test-file vocabulary-cap drop — a literal
    legitimately pinned in 4+ test files is silently dropped, though
    widely-pinned literals are exactly where a missed enumeration breaks
    the most tests (#2029 plan §8 R3). The critic layer remains the
    semantic backstop, exactly as for c46/c47. NEVER FAILs — a heuristic
    text check must not hard-block a legitimately-worded plan (the c14/c47
    doctrine); Phase 1.5.0 forwards WARN lines verbatim into the
    fact-checker + critic briefs (#2029).
    """
    cid, name = "c51_edited_literal_pin_tests", "edited workflow-surface literal pin-test coverage"
    if kind not in ("infra", "batch"):
        return _skip(cid, name, "kind-exempt")
    trigger = _c51_trigger_line(plan)
    if trigger is None:
        return _skip(cid, name, "no workflow-surface edit declaration")
    if _standalone_na_declared(plan, r"no workflow-?surface literal edits"):
        return _pass(cid, name, "explicit N/A declared (no workflow-surface literal edits)")
    root = Path(_C51_REPO_ROOT)
    missing_dirs = [str(d) for d in (root / ".claude" / "skills", root / "tests") if not d.is_dir()]
    if missing_dirs:
        return _skip(
            cid,
            name,
            "workflow-surface/tests dirs unavailable under the repo root "
            f"({', '.join(missing_dirs)}) — pin scan impossible off-repo "
            "(the c46 --plan-file degradation precedent)",
        )
    anchored = _c51_anchored_candidates(plan, root)
    if not anchored:
        return _skip(
            cid,
            name,
            f"trigger fired but no plan-quoted literal anchors in the workflow surface "
            f"({trigger[:50]!r})",
        )
    concat, offsets, paths, _lineset, _tokset = _c51_surface_union(root)
    rare = [
        c
        for c in anchored
        if len(_c51_distinct_file_hits(c, concat, offsets, paths, _C51_MAX_SURFACE_FILES))
        <= _C51_MAX_SURFACE_FILES
    ]
    if not rare:
        return _skip(cid, name, "anchored candidates are all multi-file surface boilerplate")
    items = _c51_offender_items(plan, rare, root)
    if items:
        shown = "; ".join(f"{tf} (pins literal '{lit[:60]}')" for tf, lit in items[:3])
        more = f" (+{len(items) - 3} more)" if len(items) > 3 else ""
        return _warn(
            cid,
            name,
            f"plan edits workflow-surface literal(s) already pinned by unlisted tests/ "
            f"file(s): {shown}{more} — add each file to the plan's edit-target/File-paths "
            "list, or declare `N/A — no workflow-surface literal edits` on its own line "
            "(unwrapped) (incident #1948; this check #2029)",
        )
    return _pass(
        cid,
        name,
        f"{len(rare)} anchored candidate(s); every pinning tests/ file is named in the plan",
    )


# ─── Check 52 — fan-out RAM/GPU-mem floor vs ladder rung (#2033) ───────────

_C52_GIB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Gi?B\b")
#: RAM-side peak tokens. ``RSS`` (word-bounded, uppercase — the repo's own
#: spelling) and the ``host RAM`` / ``peak RAM`` phrases are inherently
#: peak-context terms, so the RAM arm needs no SEPARATE peak token — unlike
#: the GPU arm, whose VRAM/HBM/GPU-mem tokens also live on hardware-SPEC
#: lines ("1x H100 (80 GB HBM)").
_C52_RAM_TOKEN_RE = re.compile(r"\bRSS\b|(?i:\bhost\s+RAM\b|\bpeak\s+RAM\b)")
_C52_GPU_TOKEN_RE = re.compile(r"(?i)\bVRAM\b|\bHBM\b|\bGPU[- ]mem\w*")
#: Peak-context tokens the GPU arm ADDITIONALLY requires on the same line,
#: so hardware-spec lines never extract (#2033 critic concern 1 — AC3's
#: "declares a per-leg VRAM/HBM peak" is the binding semantics; a bare-token
#: regex would blow the <1% corpus WARN target on spec lines).
_C52_PEAK_CONTEXT_RE = re.compile(r"(?i)\bpeak\b|\bper[- ]leg\b|\bco-resident\b|\bestimat\w*")


def _c52_nearest_gib(line: str, anchor: int) -> float | None:
    """The GiB/GB value on ``line`` whose match sits NEAREST ``anchor`` (the
    arm token's position) — 'peak RSS 100 GB on the 128 GB box' extracts
    100, not the machine size. ``None`` when the line has no GiB number."""
    vals = [(abs(m.start() - anchor), float(m.group(1))) for m in _C52_GIB_RE.finditer(line)]
    return min(vals)[1] if vals else None


def _c52_declared_peaks(plan: str) -> tuple[float | None, float | None, list[str]]:
    """Max declared per-leg host-RAM and per-GPU device-memory peaks (GiB)
    read off ``plan``, plus provenance notes naming the extracted lines.

    Line-anchored, whole-plan — fenced code blocks are NOT excluded (launch
    commands live in fences while estimate prose usually sits outside; the
    scan stays conservative by construction). RAM arm: a line carrying a
    RAM token (see ``_C52_RAM_TOKEN_RE``) AND a GiB/GB number. GPU arm: a
    line carrying a VRAM/HBM/GPU-mem token AND a GiB/GB number AND a
    peak-context token on the SAME line. Per line the number NEAREST the
    arm token wins; across lines the MAX wins (LARGEST-CELL keying,
    `.claude/rules/plan-compute-sizing.md` § CPU-phase RAM/RSS routing).
    GB is read as GiB — conservative at these thresholds.
    """
    max_ram: float | None = None
    max_vram: float | None = None
    notes: list[str] = []
    for line in plan.splitlines():
        m_ram = _C52_RAM_TOKEN_RE.search(line)
        if m_ram is not None:
            val = _c52_nearest_gib(line, m_ram.start())
            if val is not None and (max_ram is None or val > max_ram):
                max_ram = val
                notes.append(f"RAM peak {val:g} GiB from {line.strip()[:70]!r}")
        m_gpu = _C52_GPU_TOKEN_RE.search(line)
        if m_gpu is not None and _C52_PEAK_CONTEXT_RE.search(line):
            val = _c52_nearest_gib(line, m_gpu.start())
            if val is not None and (max_vram is None or val > max_vram):
                max_vram = val
                notes.append(f"VRAM peak {val:g} GiB from {line.strip()[:70]!r}")
    return max_ram, max_vram, notes


def _c52_argv_label(i: int, argv: list[str]) -> str:
    """Short human label for launch argv ``i`` (0-based) — the index plus a
    distinctive token (the ``--intent`` pair when present), so a
    multi-launch WARN names WHICH argv lacks the floor (AC8/t8)."""
    for j, tok in enumerate(argv):
        if tok == "--intent" and j + 1 < len(argv):
            return f"launch argv #{i + 1} (--intent {argv[j + 1]})"
        if tok.startswith("--intent="):
            return f"launch argv #{i + 1} ({tok})"
    return f"launch argv #{i + 1} ({' '.join(argv[:2])})"


def _c52_arm_warns(
    argv: list[str],
    ns,
    label: str,
    flag: str,
    dest: str,
    declared_peak: float | None,
    rung: float,
    rung_name: str,
    peak_desc: str,
) -> list[str]:
    """AC2/AC3 (missing floor flag while the declared peak strictly exceeds
    the rung constant) + AC4 (flag present but strictly below the declared
    estimate) for ONE launch argv and ONE dimension. Returns WARN clauses."""
    if declared_peak is None:
        return []
    if not _c46_has_flag(argv, flag):
        if declared_peak > rung:
            return [
                f"plan declares {peak_desc} ~{declared_peak:g} GiB, strictly above the "
                f"ladder rung gcp.{rung_name} = {rung:g} GiB, but {label} lacks {flag} — "
                f"add {flag} {math.ceil(declared_peak)} so the rung walk skips undersized "
                f"machines"
            ]
        return []
    flag_val = getattr(ns, dest, None)
    if flag_val is not None and float(flag_val) < declared_peak:
        return [
            f"{label} declares {flag} {flag_val}, strictly below the plan's declared "
            f"{peak_desc} ~{declared_peak:g} GiB — raise the floor to the declared estimate"
        ]
    return []


def check_fanout_ram_floor(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: EVERY plan-embedded launch-shaped
    ``dispatch_issue.py`` argv is checked against the plan's own declared
    per-leg peaks — a declared host-RAM peak strictly above the ladder's
    smallest-RAM GPU rung (``gcp.MACHINE_RAM_GIB["a2-highgpu-1g"]`` =
    85 GiB) requires ``--min-ram-gb`` on every launch argv (AC2), a
    declared per-GPU device-memory peak strictly above
    ``gcp.A100_40_USABLE_GIB`` (38.0 GiB) requires ``--min-gpu-mem-gb``
    (AC3), and a PRESENT floor flag strictly below the declared estimate
    WARNs as floor-too-low (AC4). Mechanizes the #1739 wave-1 gap: the
    dispatch-side guards (#1998 ``--min-ram-gb`` rung walk; #1468 A100-40
    rung skip) are armed ONLY by the flags, and nothing plan-side required
    declaring them — 5-6 of 12 GCE legs rc=137 OOM'd after the spot rung
    downgraded half the fleet to 85 GB-RAM ``a2-highgpu-1g`` boxes.

    Composes the c46/c50 helper family (``_c50_launch_argvs`` argv
    extraction; ``_c46_argparser`` / ``_c46_dry_parse`` / ``_c46_has_flag``
    parsing) but deliberately does NOT copy c50's ``len(argvs) > 1`` SKIP
    (AC8): c52 evaluates EVERY launch argv independently — the multi-launch
    fan-out is the DEFINING case, and per-argv flag presence needs no
    wall-row <-> dispatch join. Deliberately NOT conditioned on live router
    policy (``GCP_PROVISIONING_DISABLED``) or lane reachability: plans
    outlive policy flips, ``--min-ram-gb`` is threaded lane-generically,
    and WARN-only polarity absorbs inert-lane false positives. Every
    ambiguity SKIPs with a stated reason; the check NEVER FAILs (the
    c46/c47/c50 posture).

    TWO named residuals: (i) FLEET-TOTAL false-WARN — a plan quoting a
    fleet-total RSS ("720 GB across 12 legs") on one line reads as a
    per-leg peak and can WARN a correctly-floored plan (WARN-only absorbs
    it; the message names the extracted line); (ii) a fan-out driven by a
    CUSTOM DRIVER script with NO plan-embedded ``dispatch_issue.py launch``
    argv — the actual #1739 wave-1 dispatch channel — is structurally
    INVISIBLE to c52 (SKIP at "no launch argvs"); the binding surface there
    is `.claude/rules/plan-compute-sizing.md` (§ Ladder-rung RAM floor),
    and a c52 SKIP must never be read as coverage.
    """
    del kind  # all kinds: an OOM'd leg dies identically regardless of task kind
    cid, name = "c52_fanout_ram_floor", "fan-out RAM/GPU-mem floor vs ladder rung"
    max_ram, max_vram, peak_notes = _c52_declared_peaks(plan)
    if max_ram is None and max_vram is None:
        return _skip(cid, name, "no per-leg RSS / VRAM / HBM peak-estimate token in the plan")
    argvs, notes = _c50_launch_argvs(plan)
    tail = ("; " + "; ".join(notes)) if notes else ""
    if not argvs:
        return _skip(
            cid,
            name,
            "no launch-shaped dispatch_issue.py command in the plan — a custom-driver "
            "fan-out is structurally invisible here (docstring residual (ii)): the binding "
            "surface is plan-compute-sizing.md, and this SKIP is not coverage" + tail,
        )
    try:
        from explore_persona_space.backends.gcp import A100_40_USABLE_GIB, MACHINE_RAM_GIB

        ram_rung = float(MACHINE_RAM_GIB["a2-highgpu-1g"])
        vram_rung = float(A100_40_USABLE_GIB)
    except Exception as exc:  # off-repo --plan-file run -> loud SKIP
        return _skip(
            cid, name, f"gcp ladder-rung constants unavailable ({type(exc).__name__}: {exc})"
        )
    parser, load_detail = _c46_argparser()
    if parser is None:
        return _skip(cid, name, f"dispatch_issue.build_argparser unavailable ({load_detail})")
    warns: list[str] = []
    argv_notes: list[str] = []
    n_parsed = 0
    for i, argv in enumerate(argvs):
        ns, err = _c46_dry_parse(parser, argv)
        if ns is None:  # per-argv note, never a WARN — c46 arm 1 owns parse drift
            argv_notes.append(f"argv #{i + 1} does not dry-parse ({err}) — c46 arm 1 owns that")
            continue
        n_parsed += 1
        label = _c52_argv_label(i, argv)
        warns.extend(
            _c52_arm_warns(
                argv,
                ns,
                label,
                "--min-ram-gb",
                "min_ram_gb",
                max_ram,
                ram_rung,
                'MACHINE_RAM_GIB["a2-highgpu-1g"]',
                "per-leg peak RSS / host RAM",
            )
        )
        warns.extend(
            _c52_arm_warns(
                argv,
                ns,
                label,
                "--min-gpu-mem-gb",
                "min_gpu_mem_gb",
                max_vram,
                vram_rung,
                "A100_40_USABLE_GIB",
                "per-leg VRAM/HBM peak",
            )
        )
    if n_parsed == 0:
        return _skip(cid, name, "; ".join(argv_notes) + tail)
    extra = "; ".join(argv_notes + peak_notes[:2])
    if warns:
        return _warn(cid, name, "; ".join(warns) + (f" [{extra}]" if extra else ""))
    ram_txt = f"{max_ram:g} GiB" if max_ram is not None else "n/a"
    vram_txt = f"{max_vram:g} GiB" if max_vram is not None else "n/a"
    return _pass(
        cid,
        name,
        f"declared peaks (RAM {ram_txt} / VRAM {vram_txt}) are floor-covered or under "
        f"the smallest-rung constants across {n_parsed} launch argv(s)",
    )


# ─── Check 53 — harm-class judged DV api-refusal accounting (#2207) ────────

# Trigger arm (a): judged-DV vocabulary — the conservative \bjudg stem
# (judge / judged / judges / judging / judge-scored). Raw scan, fences
# INCLUDED (the c43/c44 family convention): judged-DV designs live in fenced
# measurement-validity tables as often as in prose.
_C53_JUDGE_VOCAB_RE = re.compile(r"(?i)\bjudg(?:e|ed|es|ing)")

# Trigger arm (b): harm-class vocabulary. Deliberately narrow — the bare
# word "adversarial" must NEVER fire (every plan quotes
# `/adversarial-planner`); only the role-play compound does. The
# `harmful_compliance` behavior-class token (#906/#1090) is folded in.
_C53_HARM_VOCAB_RE = re.compile(
    r"(?i)(?:\bjailbreak|\bharmfulness\b|\bharm[- ](?:judge|rate|score)s?\b"
    r"|\badversarial[- ]role[- ]?play|\bharmful[-_ ]compliance\b)"
)

# Trigger arm (b2) — line-windowed evil/toxic (#2227, the #2221 silent pass).
# Bare \bevil\b stays REJECTED (saturates persona-vector trait prose); an
# evil/toxic token is harm-class vocabulary ONLY on a line that also carries
# judging/severity/grading/rubric vocabulary. `\btoxic(?:ity)?\b`-with-boundary
# deliberately does NOT match bare corpus names (ToxicChat, _stage_toxicchat:
# no word boundary between "toxic" and the joined "chat"). The `\bband...\b`
# and `\bscor...\b` context stems were DROPPED after the round-1 corpus
# replay (plan §10's pre-authorized tightening): `band` fired on a marker
# "readable implant band" (#505), a statistics "null bands" line (#958), and
# a lit-review line (#685); `scores` fired on a Persona-Vectors-paper
# citation line (#471) — none of the genuine fires depended on either stem.
_C53_HARM_WINDOW_TOKEN_RE = re.compile(r"(?i)(?:\bevil\b|\btoxic(?:ity)?\b)")
_C53_HARM_WINDOW_CTX_RE = re.compile(
    r"(?i)(?:\bjudg(?:e|ed|es|ing)|\bsever(?:ity|e)\b|\bgrad(?:e|ed|ing)\b|\brubric\b)"
)

# Satisfier: any api-refusal token — api-refusal / api_refusal / "API
# refusal" / n_api_refusal (deliberately NOT \b-anchored at the front so the
# `n_api_refusal` counter name, whose `_` is a word char, matches too).
_C53_API_REFUSAL_RE = re.compile(r"(?i)api[-_ ]refusal")

# Satisfier: `refusal` in the VALUE position of a `stop_reason`
# comparison/assignment (stop_reason == "refusal" / stop_reason: refusal).
# Tightened from the loose same-line co-mention (#2227): #2221 v5 L103
# carries rule-9 boilerplate (REFUSAL dropped) plus the rule-26 pilot gate
# (`stop_reason=="max_tokens"`) on ONE line, which co-mention falsely
# accepted — content-drop handling and a pilot gate are NOT api-refusal
# accounting (rule 28's own non-coverage note).
_C53_STOPREASON_REFUSAL_RE = re.compile(r"(?i)stop_reason\s*[=:]+\s*[\"'`]?\s*refusal")

_C53_ESCAPE_RE = re.compile(r"(?i)^rule[- ]28 exempt(?:ion)?\b")


def _c53_escape_declared(plan: str) -> bool:
    """Standalone ``rule 28 exemption: <reason>`` declaration (no N/A
    prefix — the c43 shape), or the shared-helper ``N/A — rule 28
    exemption`` form. Fenced lines and wrapped pastes never satisfy (see
    ``_standalone_na_declared``)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    for line, fenced in zip(lines, mask, strict=True):
        if fenced:
            continue
        if _C53_ESCAPE_RE.match(line.lstrip(" \t>*-")):
            return True
    return _standalone_na_declared(plan, r"rule[- ]28 exempt")


def check_judged_dv_api_refusal(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, experiment-only: a plan whose judged DV
    scores harm-class completions (jailbreak / harmfulness / harm-rate /
    adversarial-role-play / harmful-compliance vocabulary alongside judge
    vocabulary, or — arm (b2), #2227 — an evil/toxic token line-windowed
    with judging/severity/grading/rubric vocabulary) must
    name its api-refusal accounting — per-arm
    ``n_api_refusal`` reported separately from BOTH content drops and
    transport losses, plus the targeted SYNC re-issue remediation at the
    IDENTICAL instrument (reference implementation:
    ``scripts/issue1739_evilood_refusal_rejudge.py``) — or declare the
    ``rule 28 exemption: <reason>`` escape. Mechanizes the llm-judging.md
    rule-28 Enforcement rider (#2151): the Batch API can return a
    SUCCEEDED row whose ``stop_reason`` is ``"refusal"`` with an EMPTY
    content array — the THIRD drop class, transport-conditional and
    retriable, NEITHER a rule-9 content drop NOR a rule-24 transport
    loss. The censoring is OUTCOME-CORRELATED (highest-harm rows first),
    so absorbing it into the content tally biases the harm DV DOWN on
    exactly its highest-scoring rows. Founding incident #1739: 34.1% of
    harm-judge draws (15,091/44,310) were api-refusal-censored and filed
    as ``parse_error`` content drops; a rule-26 pilot-gate PASS does NOT
    cover this class BY DESIGN (rule 28's non-coverage note: api-refusal
    draws leave both the parse-fail numerator and denominator).
    Satisfiers (any): an api-refusal token (``api-refusal`` /
    ``api_refusal`` / ``n_api_refusal``), a ``stop_reason``
    comparison/assignment with ``refusal`` in the VALUE position
    (``stop_reason == "refusal"``; a bare same-line co-mention of the two
    words — the rule-9/rule-26 boilerplate shape on #2221 v5 L103 — no
    longer satisfies, #2227), or the standalone exemption line. Trigger AND
    satisfiers scan the RAW plan text (fences INCLUDED, the c43
    precedent — judged-DV designs and remediation notes live in fenced
    tables/blocks). NEVER FAILs (the c39/c31/c34/c43 family convention).
    kind-exempt outside experiment: infra workflow-fix plans (this
    check's own plan included) legitimately QUOTE the trigger vocabulary
    without dispatching a harm-judged eval. DESIGN HISTORY (re-scoped by
    #2227): BARE ``\\bevil\\b`` stays REJECTED — it saturates
    persona-vector trait prose across the persisted corpus — but arm (b2)
    now admits an evil/toxic token when it CO-OCCURS ON ONE LINE with
    judging/severity/grading/rubric vocabulary (the #2221 shape: "graded 0-100
    judged on-policy trait-expression score ... for
    evil/sycophancy/hallucination"). The earlier rejection of
    "judge-proximity tuning" was about SUPPRESSING benign fires of the
    EXISTING arm-(b) vocabulary — using same-line co-occurrence to ADMIT
    a new trigger arm without false positives is the complementary
    direction and is what #2227 prescribes. KNOWN FALSE NEGATIVES that
    REMAIN (disclosed by design): evil/toxic phrasing split across lines
    from all judging vocabulary; "misaligned"/"EM"-phrased judged DVs
    (still rejected — they saturate the project's standing vocabulary);
    refusal-bait corpora phrased as "harmful advice"/"unsafe content";
    and safety-benchmark names quoted bare (mhj, pair, tom-gibbs) — the
    Statistics & Measurement lens REVISE stays the binding gate for
    those shapes. NAMED RESIDUAL (2026-08 corpus replay, 3,751 plan
    versions / 26 distinct WARN'd tasks at real kinds): 24/26 fire-tasks
    genuinely judge harm-class panels; 2 benign single-mention fires —
    a downstream-motivation "persona-jailbreak detection" mention in a
    judge-free marker plan (#382) and a duplicate-clustering "template
    jailbreaks" example in a zero-new-judge-call analysis plan (#1073).
    Occurrence-count and judge-proximity tuning (as SUPPRESSORS) were
    both REJECTED: the genuine single-mention fires
    (#545/#591/#2091/#459) are byte-shaped identically to the benign
    ones — WARN-only polarity + the exemption escape absorb the residual
    (the c52 named-residual posture). #2227 RESIDUAL (2026-08-10 corpus
    replay, 3,882 plan versions at real kinds, patched-vs-unpatched
    delta): 20 newly-WARN tasks — 18 via the windowed arm, 2 via the
    satisfier tightening (#1739 v15 + #2203 v1, both genuinely
    harm-judged, previously PASSing on the loose boilerplate
    co-mention); 17/20 genuine (the persona-vectors evil/syco/hallu
    judging line #778/#779/#816/#1415/#1769/#1774/#2220-#2225 plus
    #537/#685/#1092/#1776), 3 benign — #841/#922 ("no new judging"
    artifact-reuse lines) + #1768 (an N/A screen disclaimer) — absorbed
    by WARN-only polarity + the exemption escape. The round-1 replay
    (context window still carrying the `band`/`scor` stems) fired 23
    tasks with 7 benign (~30%), driving the §10-preauthorized stem
    drop (#471/#505/#685/#958 were the stem-only fires)."""
    cid, name = "c53_judged_dv_api_refusal", "harm-class judged DV api-refusal accounting"
    if kind != "experiment":
        return _skip(
            cid,
            name,
            "kind-exempt: api-refusal accounting is an experiment-plan (harm-judged-DV) shape",
        )
    if not _C53_JUDGE_VOCAB_RE.search(plan):
        return _skip(cid, name, "no judged-DV vocabulary in the plan")
    harm_hits = sorted({m.group(0).lower() for m in _C53_HARM_VOCAB_RE.finditer(plan)})
    if not harm_hits:
        # Arm (b2) — line-windowed evil/toxic (#2227): fire only when an
        # evil/toxic token co-occurs on ONE line with judging/severity/
        # grading/rubric vocabulary. First hit suffices; the WARN carries the
        # token + line number so a reader can see WHY it fired. Plain
        # string appended so the WARN's join keeps working.
        for lineno, line in enumerate(plan.splitlines(), start=1):
            tok = _C53_HARM_WINDOW_TOKEN_RE.search(line)
            if tok and _C53_HARM_WINDOW_CTX_RE.search(line):
                harm_hits.append(f"{tok.group(0).lower()} (L{lineno}, windowed)")
                break
    if not harm_hits:
        return _skip(
            cid,
            name,
            "no harm-class judged-DV vocabulary (jailbreak / harmfulness / harm-rate / "
            "adversarial-role-play / harmful-compliance; nor a windowed evil/toxic token "
            "co-occurring on one line with judging/severity/grading/rubric vocabulary, #2227)",
        )
    if _C53_API_REFUSAL_RE.search(plan):
        return _pass(
            cid, name, "api-refusal accounting token present (api-refusal / n_api_refusal)"
        )
    for ln in plan.splitlines():
        if _C53_STOPREASON_REFUSAL_RE.search(ln):
            return _pass(
                cid,
                name,
                'stop_reason comparison with "refusal" in value position names the drop class',
            )
    if _c53_escape_declared(plan):
        return _pass(cid, name, "explicit escape declared (rule 28 exemption)")
    return _warn(
        cid,
        name,
        f"plan judges harm-class completions ({', '.join(harm_hits[:4])!r}) with no "
        "api-refusal accounting — llm-judging.md rule 28 (#2151): Batch rows returning "
        '`stop_reason == "refusal"` with empty content are the THIRD drop class, '
        "outcome-correlated (highest-harm rows first; #1739: 34.1% of harm-judge draws "
        "censored), and a rule-26 pilot-gate PASS is NOT a substitute. Name per-arm "
        "`n_api_refusal` reported separately from content drops and transport losses, plus "
        "the targeted SYNC re-issue remediation at the identical instrument (reference: "
        "`scripts/issue1739_evilood_refusal_rejudge.py` — the llm-judging.md § Enforcement "
        'rule-28 rider); naming the drop class in value position (`stop_reason == "refusal"`) '
        "also satisfies — a bare same-line co-mention of `stop_reason` and `refusal` "
        "(rule-9/rule-26 boilerplate) does NOT (#2227) — or declare "
        "`rule 28 exemption: <reason>` on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 54 — bare lane-specific env vars in --workload-cmd (#2047) ──────

_c54_lint_fn_cache: list = []  # [(fn | None, detail)] once resolved (the c46 idiom)


def _c54_lint_fn():
    """Lazily import the #1329 runtime lane-env lint
    (``explore_persona_space.backends.issue_dispatch.
    lint_workload_cmd_lane_env`` — the single source of truth for
    ``LANE_WORKLOAD_ENV_EXPORTS``, reachable-lane mapping, the
    bare-reference regex with its set-u-safe exclusions, and
    single-quoted-segment stripping).

    Returns ``(fn, "")`` on success or ``(None, <detail>)`` on ANY import
    failure (an off-repo ``--plan-file`` run without the package
    importable) — the caller SKIPs loudly on ``None``, the c46 fail-open
    idiom: c54 is a plan-drift detector, never a gate on the helper
    module itself. Cached one-shot so corpus sweeps import once.
    """
    if not _c54_lint_fn_cache:
        fn, detail = None, ""
        try:
            from explore_persona_space.backends.issue_dispatch import (
                lint_workload_cmd_lane_env,
            )

            fn = lint_workload_cmd_lane_env
        except Exception as exc:  # any import failure -> loud SKIP at the caller
            fn, detail = None, f"{type(exc).__name__}: {exc}"
        _c54_lint_fn_cache.append((fn, detail))
    return _c54_lint_fn_cache[0]


def _c54_flag_value(tokens: list[str], flag: str) -> str | None:
    """Value of ``flag`` in ``tokens`` (``--flag value`` AND
    ``--flag=value`` forms), or ``None`` when absent."""
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def check_workload_cmd_lane_env(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: the ``--workload-cmd`` value of
    every plan-embedded ``dispatch_issue.py`` command (the c46 candidate
    sources: fenced code blocks + inline-code spans, backslash
    continuations joined) must not reference a lane-specific env var
    BARE — a var exported by only SOME reachable lanes (canonical
    offender: GCE-only ``$WORKLOAD_ROOT``) aborts under another lane's
    ``set -u`` stage. Detection is delegated whole to the #1329 runtime
    lint ``lint_workload_cmd_lane_env`` (lazy import; unavailable →
    loud SKIP): set-u-safe ``${VAR:-<default>}`` forms and
    single-quoted segments never flag, and the raw ``--backend`` value
    (both flag forms; absent → auto) scopes the reachable lanes.
    CRITICAL extraction detail: ``${VAR:+...}`` spans are stripped
    whole pre-shlex (``_C46_COND_EXPANSION_RE`` — set-u-safe by
    construction and unsplittable otherwise) but ``_C46_PLACEHOLDER_RE``
    is NOT applied — the raw ``$VAR`` text IS the evidence this check
    scans. Mechanizes the #1979 f1g incident: the plan §10 command used
    bare ``$WORKLOAD_ROOT`` on an unpinned auto route, the fellows SLURM
    lane's ``set -u`` custom stage died on ``WORKLOAD_ROOT: unbound
    variable``, and a full dispatch burned before a set-u-safe relaunch.
    No dispatch command → SKIP; no ``--workload-cmd`` token in any
    candidate → SKIP; unsplittable lines get a per-line note, never a
    crash. NEVER FAILs (the c43/c46/c50 posture) — the #1329 runtime
    exit-2 refusal (``workload_cmd_lane_env_unbound``) fires only on the
    ``certain`` arm (explicit SLURM-lane pin, or runpod with
    ``--execute-workload``); for AUTO-lane plans — the #1979 incident
    shape — the runtime is advisory-only, so c54 is the earliest and
    often only pre-burn surfacing. KNOWN v1 FALSE NEGATIVE (disclosed by
    design): a ``--workload-cmd "$WCMD"`` indirection (the value held in
    a shell variable) scans the literal ``$WCMD``, which is outside the
    helper's lint universe — no WARN (same class as the helper's own
    documented v1 false negatives, issue_dispatch.py ~L437-440).
    """
    del kind  # all kinds: a bare lane-specific env var dies identically everywhere
    cid, name = "c54_workload_cmd_lane_env", "--workload-cmd bare lane-specific env vars"
    candidates = _c46_command_candidates(plan)
    if not candidates:
        return _skip(cid, name, "no dispatch_issue.py command in fenced blocks or inline code")
    lint_fn, load_detail = _c54_lint_fn()
    if lint_fn is None:
        return _skip(cid, name, f"lane-env lint helper unavailable ({load_detail})")
    offenders: list[str] = []
    notes: list[str] = []
    n_scanned = 0
    for cmd in candidates:
        # Strip ${VAR:+...} spans whole; do NOT substitute placeholders —
        # the raw `$VAR` text is exactly what the lane-env lint scans.
        cleaned = _C46_COND_EXPANSION_RE.sub("", cmd)
        try:
            tokens = shlex.split(cleaned)
        except ValueError as exc:
            notes.append(f"unsplittable line skipped ({exc}): {cmd[:60]!r}")
            continue
        wcmd = _c54_flag_value(tokens, "--workload-cmd")
        if wcmd is None:
            continue  # provision-only / finalize command — nothing to scan
        n_scanned += 1
        # backend passed raw: an unparseable placeholder value degrades to
        # auto inside the helper (its own normalize_backend_value catch).
        lint = lint_fn(wcmd, backend_value=_c54_flag_value(tokens, "--backend"))
        for var, lanes in sorted(lint.flagged.items()):
            offenders.append(
                f"{cmd[:70]!r}: {var} bare but unexported on lane(s) {', '.join(lanes)}"
            )
    if offenders:
        shown = " | ".join(offenders[:3])
        more = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
        tail = ("; " + "; ".join(notes)) if notes else ""
        return _warn(
            cid,
            name,
            f"plan-embedded --workload-cmd value(s) carry bare lane-specific env vars: {shown}"
            f"{more} — use the set-u-safe `${{VAR:-<default>}}` expansion or pin `backend:` "
            "to a lane exporting the var; runtime refusal: workload_cmd_lane_env_unbound "
            "(#1329); incident #1979/#2047: bare $WORKLOAD_ROOT died under the fellows "
            f"lane's set -u{tail}",
        )
    if n_scanned == 0:
        detail = "no --workload-cmd in plan-embedded dispatch commands"
        if notes:
            detail += "; " + "; ".join(notes)
        return _skip(cid, name, detail)
    detail = f"{n_scanned} --workload-cmd value(s) clean of bare lane-specific env vars"
    if notes:
        detail += "; " + "; ".join(notes)
    return _pass(cid, name, detail)


# ─── Check 55 — inherited argparse row-count default vs per-cell target ─────

_C55_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c34/c41 pattern)

#: Plan-named candidate script paths (scripts/ or src/, ``.py``). Cost
#: bound: at most ``_C55_MAX_SCRIPTS`` are resolved per plan.
_C55_SCRIPT_RE = re.compile(r"\b(?:scripts|src)/[A-Za-z0-9_./-]+\.py\b")
_C55_MAX_SCRIPTS = 12

#: Issue-branch tokens the plan names — the ``git show`` resolution fallback
#: for scripts living only on an unmerged issue branch (#2054's phase_c.py).
_C55_BRANCH_RE = re.compile(r"\bissue-\d+[a-z0-9-]*\b")
_C55_MAX_BRANCHES = 4

#: argparse flag token at an ``add_argument(`` call head.
_C55_ADD_ARG_RE = re.compile(r"add_argument\(\s*[\"'](--[A-Za-z0-9][\w-]*)[\"']")
#: Integer default within the bounded post-flag window; the negative
#: lookahead rejects float defaults (``default=0.5``), and ``\d[\d_]*``
#: accepts the underscore-separator literal (``default=8_000``, the verbatim
#: #2054 offender shape, issue2054 phase_c.py).
_C55_DEFAULT_RE = re.compile(r"default\s*=\s*(\d[\d_]*)(?![.\d_])")
#: Chars scanned after the flag token for kwargs — bounds multi-line
#: ``add_argument`` calls without paren-matching.
_C55_KWARG_WINDOW = 300

#: Row-count flag-name axes: the flag must carry BOTH a cap-ish token
#: (target/max/limit) and a row-ish token (conv*/row(s)/id(s)), in any
#: order, each anchored at a ``-``/``_`` token boundary so e.g.
#: ``--target-grid`` ("id" inside "grid") never matches. The row axis is
#: prefix-tolerant at the boundary (``conv`` covers ``conversations``).
_C55_CAP_AXIS_RE = re.compile(r"(?:^|[-_])(?:target|max|limit)(?=[-_]|$)")
_C55_ROW_AXIS_RE = re.compile(r"(?:^|[-_])(?:conv|rows?|ids?)")

#: Plan-prose per-cell target patterns (the fuzzy leg — WARN-only absorbs
#: the false-positive surface). Values under ``_C55_TARGET_FLOOR`` (seed
#: counts, tiny ints) are dropped; the MAX surviving candidate is the
#: stated target (conservative: the max is what the plan claims to reach).
_C55_TARGET_RES = [
    re.compile(r"per[- ]cell[^.\n]{0,60}?(\d[\d,]{2,})", re.I),
    re.compile(r"\bn(?:_train)?\s*[=≈~]\s*(\d[\d,]{2,})", re.I),
    re.compile(r"\|S\|\s*[=≈]\s*(\d[\d,]{2,})"),
    re.compile(r"target(?:\s+of)?\s+[^.\n]{0,40}?(\d[\d,]{2,})", re.I),
]
_C55_TARGET_FLOOR = 100


def _c55_candidate_scripts(plan: str) -> tuple[list[str], bool]:
    """Ordered, deduped plan-named script paths, capped at
    ``_C55_MAX_SCRIPTS`` (cost bound; the second element reports the cap
    firing so the detail can note the truncation). Traversal-shaped
    candidates (``..``) are dropped."""
    seen: list[str] = []
    for m in _C55_SCRIPT_RE.finditer(plan):
        rel = m.group(0)
        if ".." in rel or rel in seen:
            continue
        seen.append(rel)
    return seen[:_C55_MAX_SCRIPTS], len(seen) > _C55_MAX_SCRIPTS


def _c55_git_show(ref: str, rel: str) -> str | None:
    """``git show <ref>:<rel>`` in ``_C55_REPO_ROOT``; ``None`` on any
    failure (missing ref/path, git unavailable, timeout) — c55 is WARN-only
    lint, so resolution failures degrade to caller-side notes, never a
    crash (the c42/c44 fail-open contract)."""
    try:
        r = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_C55_REPO_ROOT),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return r.stdout if r.returncode == 0 else None


def _c55_resolve_script(rel: str, branches: list[str]) -> str | None:
    """Script text: working-tree read first, then ``git show <branch>:``
    then ``git show origin/<branch>:`` per plan-named issue branch — a
    branch fetched but never checked out locally resolves only via the
    ``origin/`` ref (#2054's phase_c.py lived on the ``issue-2054``
    branch). ``None`` when nothing resolves (caller notes it, fail-soft)."""
    p = _C55_REPO_ROOT / rel
    text: str | None = None
    try:
        if p.is_file():
            text = p.read_text(errors="replace")
    except OSError:
        text = None  # unreadable working-tree copy — fall through to the git refs
    if text is not None:
        return text
    for branch in branches:
        for ref in (branch, f"origin/{branch}"):
            text = _c55_git_show(ref, rel)
            if text is not None:
                return text
    return None


def _c55_rowcount_defaults(text: str) -> list[tuple[str, int]]:
    """``(flag, default)`` pairs for row-count-shaped argparse flags with an
    integer literal default. The kwarg scan is a bounded window after the
    flag token (multi-line ``add_argument`` calls match without
    paren-matching); ``default=0`` is skipped — the conventional no-limit
    value cannot under-cover a target."""
    out: list[tuple[str, int]] = []
    for m in _C55_ADD_ARG_RE.finditer(text):
        name = m.group(1)[2:]  # strip the leading --
        if not (_C55_CAP_AXIS_RE.search(name) and _C55_ROW_AXIS_RE.search(name)):
            continue
        window = text[m.end() : m.end() + _C55_KWARG_WINDOW]
        dm = _C55_DEFAULT_RE.search(window)
        if dm is None:
            continue
        value = int(dm.group(1).replace("_", ""))
        if value == 0:
            continue
        out.append((m.group(1), value))
    return out


def _c55_plan_target_n(plan: str) -> int | None:
    """MAX plan-stated per-cell target ``>= _C55_TARGET_FLOOR``, or ``None``
    when no target-shaped integer is recognized (caller SKIPs)."""
    candidates: list[int] = []
    for pat in _C55_TARGET_RES:
        for m in pat.finditer(plan):
            value = int(m.group(1).replace(",", ""))
            if value >= _C55_TARGET_FLOOR:
                candidates.append(value)
    return max(candidates) if candidates else None


def check_inherited_rowcount_default(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: a plan that reuses a
    generation/splice script whose argparse integer ROW-COUNT default (a
    flag carrying both a target/max/limit and a conv/rows/ids token) sits
    BELOW the plan's own stated per-cell target n, with the flag token
    appearing NOWHERE in the plan text, silently caps coverage — the run
    truncates deterministically (first-N prefix) and every gate that runs
    before the consuming phase reads PASS without seeing the breach.
    Mechanizes the #2054 amendment-plan v11 incident: the reused
    ``--target-conv-ids`` default (8,000; phase_c.py on the ``issue-2054``
    branch) would have capped 28 of 48 in-scope cells below the registered
    |S| = 9,000 intersection target; the mechanical pre-pass passed the
    plan clean twice (v10, v11) and only the Methodology critic caught it.
    Scripts resolve working-tree-first, then ``git show <branch>:`` /
    ``origin/<branch>:`` for plan-named ``issue-<M>`` branch tokens
    (fail-soft note on a miss — never a crash); the plan-side target
    extraction is deliberately fuzzy (per-cell / n= / |S|= / target-of
    patterns, MAX aggregation, >= 100 floor) — WARN-only absorbs the
    false-positive surface, and a flag token mentioned ANYWHERE in the
    plan (an embedded override command, a prose acknowledgment) suppresses
    the WARN (plan-aware, per the #2054 critic sketch). Escape: standalone
    ``N/A — no inherited row-count defaults``. NEVER FAILs (the
    c43/c46/c50/c54 posture).
    """
    del kind  # all kinds: an inherited under-covering default truncates identically everywhere
    cid = "c55_inherited_rowcount_default"
    name = "inherited argparse row-count default vs per-cell target"
    if _standalone_na_declared(plan, r"no inherited row-count defaults"):
        return _pass(cid, name, "explicit N/A declared (no inherited row-count defaults)")
    scripts, truncated = _c55_candidate_scripts(plan)
    if not scripts:
        return _skip(cid, name, "no reused script paths named in plan")
    branches: list[str] = []
    for m in _C55_BRANCH_RE.finditer(plan):
        tok = m.group(0)
        if tok not in branches:
            branches.append(tok)
    branches = branches[:_C55_MAX_BRANCHES]
    notes: list[str] = []
    if truncated:
        notes.append(f"candidate scripts capped at {_C55_MAX_SCRIPTS}")
    defaults: list[tuple[str, str, int]] = []  # (script, flag, default)
    n_resolved = 0
    for rel in scripts:
        text = _c55_resolve_script(rel, branches)
        if text is None:
            notes.append(f"script unresolved: {rel}")
            continue
        n_resolved += 1
        for flag, value in _c55_rowcount_defaults(text):
            defaults.append((rel, flag, value))
    tail = ("; " + "; ".join(notes)) if notes else ""
    if n_resolved == 0:
        return _skip(cid, name, f"none of {len(scripts)} plan-named script(s) resolved{tail}")
    if not defaults:
        return _pass(
            cid,
            name,
            f"no inherited row-count defaults in {n_resolved} plan-named script(s){tail}",
        )
    target = _c55_plan_target_n(plan)
    if target is None:
        return _skip(cid, name, f"no stated per-cell target n recognized{tail}")
    offenders = [
        f"{rel}: {flag} default={value:,} < stated target {target:,} and the flag "
        "never appears in the plan"
        for rel, flag, value in defaults
        if target > value and flag not in plan
    ]
    if offenders:
        shown = " | ".join(offenders[:3])
        more = f" (+{len(offenders) - 3} more)" if len(offenders) > 3 else ""
        return _warn(
            cid,
            name,
            f"inherited argparse row-count default(s) under-cover the plan's stated per-cell "
            f"target: {shown}{more} — pass the flag explicitly in an embedded command or raise "
            "the source default; incident #2054: the reused `--target-conv-ids` default 8,000 "
            "silently capped cells below the |S| = 9,000 target (deterministic first-N "
            f"truncation){tail}",
        )
    return _pass(
        cid,
        name,
        f"{len(defaults)} row-count default(s) in {n_resolved} script(s) covered by the stated "
        f"target ({target:,}) or explicitly overridden in the plan{tail}",
    )


# ─── Check 56 — multi-GB staging row names its mount / bind liveness (#2097) ─

# Trigger: a STRIPPED-prose line carrying BOTH a staging signal AND a >=5 GB
# size figure (TB always qualifies). Calibration (#2097, implementation-time,
# AS-SHIPPED regexes; the c39/c33 gate precedent — any future c56-regex
# change re-runs the corpus scan and records the realized numbers here) over
# 3,890 persisted plan-versions (tasks/*/*/plans/v*.md; in-process, own task
# excluded, kind from body.md): 499 pv triggered (kind==experiment,
# non-SKIP); 343 pv would-WARN — 307 arm-(a)-only / 15 arm-(b)-only / 21
# both; 6 recent-era issues (>= 2000) carry WARNs. The founding incident
# #2091 (a 42 GB VM stage citing the NOT-live #681 worktree bind; zero
# `findmnt`, PASSed verify_plan twice) arm-(b)-WARNs on plans v1-v3, and its
# POST-incident revisions v4/v5 — which added the `findmnt --mountpoint`
# probe — PASS: the incident's own fix trajectory exercises both verdicts
# (pinned by tests/test_verify_plan.py::test_c56_calibration_committed_2091_v3).
# Arm-(a) adjudication of the recent-era would-WARNs: size-bearing
# risk-table / RSS-routing / compute-table rows whose staging path lives
# elsewhere in the plan — the #869 document-global-evidence shape the
# window-scoping deliberately refuses; a widened satisfier set (+MooseFS /
# container-disk / $SCRATCH mount-name tokens) was MEASURED at 343 -> 311
# would-WARN and DECLINED — it blesses filesystem-name-without-path rows,
# weakening the duty's PATH requirement for a ~10% noise cut. WARN-only +
# forward-looking (legacy plans never bounce retroactively, the
# c39/c31/c34/c43 family convention).
_C56_STAGE_RE = re.compile(
    r"(?i)\bstag(?:e|es|ed|ing)\b|\bdownload|\bsnapshot|\bmateriali[sz]e"
    r"|hf_dl|local_dir|\bprefetch"
)
_C56_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(GB|GiB|TB)\b")
_C56_SIZE_MIN_GB = 5.0
# Arm (a): PATH tokens satisfy ONLY within the trigger line +-2 STRIPPED
# lines (window-scoped — house style pushes `/workspace` merge-disk
# boilerplate + launch out-roots into nearly every plan, so a doc-global
# path satisfier vacuously PASSes a pathless staging row: the #869
# document-global-evidence shape; 9 `/workspace` mentions in #2091 v3
# alone). The mount-PROBE tokens (`df -P`, `findmnt`) satisfy from anywhere
# in the RAW plan — probe commands legitimately live in fenced
# preflight/repro blocks far from the staging row.
_C56_PATH_RE = re.compile(r"/mnt/eps-data|data/issue_|/workspace")
_C56_PROBE_RE = re.compile(r"df -P|findmnt")
# Arm (b): the incident's OWN vocabulary — `worktree`/`#681` co-occurring
# with word-boundary `bind` within the same stripped line or +-2 adjacent
# stripped lines. `\bbind\b` (not `bind`): "binding"-class house prose can
# never false-fire, while hyphens ARE word boundaries so "bind-mounted" /
# "bind-migration" still match. NOT keyed on the literal `.claude/worktrees`
# path: that string appears ZERO times in #2091 v3, whose actual wording is
# "resolves to `/mnt/eps-data` via the #681 worktree bind" (v3 L110/L272) —
# a literal-path trigger provably never fires on its own motivating incident.
_C56_BIND_RE = re.compile(r"(?i)\bbind\b")
_C56_WORKTREE_RE = re.compile(r"(?i)\bworktrees?\b|#681\b")
_C56_WINDOW = 2  # +-2 lines; the c12 window-shape convention


def _c56_size_qualifies(line: str) -> bool:
    """True when ``line`` carries a >=5 GB size figure (any TB qualifies)."""
    for m in _C56_SIZE_RE.finditer(line):
        if m.group(2) == "TB" or float(m.group(1)) >= _C56_SIZE_MIN_GB:
            return True
    return False


def check_staging_mount_binding(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, experiment-only: a multi-GB staging/footprint
    row must name its mount/staging path IN the trigger window (arm a), and
    a plan citing the #681 worktree bind for a multi-GB stage must carry a
    literal ``findmnt --mountpoint`` liveness assertion (arm b — the bind is
    NOT live on this VM; #2091's plans v2 AND v3 cited it for a 42 GB stage
    and PASSed verify_plan). Mechanizes the explicitly-deferred staging-row
    backstop of `.claude/rules/plan-compute-sizing.md` § Out-root mount
    binding (the >=5 GB inline-staging clause: staging path named up front +
    the filesystem it resolves to; incident #1393 — a 14 GB inline HF pull
    filled ``/`` -> ENOSPC).

    Fence masking is LINE-COUNT-PRESERVING (fenced lines masked in place,
    never deleted) so the stripped->raw +-2 window maps by identity index.
    Arm (b) is evaluated INDEPENDENTLY of arm (a)'s verdict — the #2091 rows
    name ``/mnt/eps-data`` in-window (arm a satisfied) and still must WARN
    on the missing liveness probe. NEVER FAILs (the c39/c31/c34/c43 family
    convention). kind-exempt outside experiment: infra workflow-fix plans —
    this check's own lineage included — legitimately discuss staging without
    having staging phases."""
    cid, name = "c56_staging_mount_binding", "staging mount binding"
    if kind != "experiment":
        return _skip(
            cid,
            name,
            "kind-exempt: multi-GB staging rows are an experiment-plan shape "
            "(infra workflow-fix plans legitimately discuss staging without staging phases)",
        )
    lines = plan.splitlines()
    mask = _fence_mask(lines)  # line-count-preserving: indexes map raw<->stripped
    trigger_idx = [
        i
        for i, (line, fenced) in enumerate(zip(lines, mask, strict=True))
        if not fenced and _C56_STAGE_RE.search(line) and _c56_size_qualifies(line)
    ]
    if not trigger_idx:
        return _skip(cid, name, "no multi-GB staging vocabulary detected")
    if _standalone_na_declared(plan, r"no multi-GB staging\b"):
        return _pass(cid, name, "explicit N/A declared (no multi-GB staging)")
    # Arm (a): every trigger line needs a path token within +-2 stripped
    # lines, unless a mount-probe token appears anywhere in the RAW plan.
    offenders_a: list[str] = []
    if not _C56_PROBE_RE.search(plan):
        for i in trigger_idx:
            lo, hi = max(0, i - _C56_WINDOW), min(len(lines), i + _C56_WINDOW + 1)
            window = "\n".join(lines[j] for j in range(lo, hi) if not mask[j])
            if not _C56_PATH_RE.search(window):
                offenders_a.append(lines[i].strip()[:90])
    # Arm (b): independent of arm (a) — a worktree-bind citation in a disk
    # context requires the literal `findmnt --mountpoint` in the RAW plan.
    bind_hits: list[int] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _C56_BIND_RE.search(line):
            continue
        lo, hi = max(0, i - _C56_WINDOW), min(len(lines), i + _C56_WINDOW + 1)
        if any(not mask[j] and _C56_WORKTREE_RE.search(lines[j]) for j in range(lo, hi)):
            bind_hits.append(i)
    arm_b_offends = bool(bind_hits) and "findmnt --mountpoint" not in plan
    if not offenders_a and not arm_b_offends:
        bits = [f"{len(trigger_idx)} multi-GB staging line(s) mount-bound"]
        if bind_hits:
            bits.append(
                f"{len(bind_hits)} worktree-bind citation line(s) with a "
                "`findmnt --mountpoint` liveness assertion"
            )
        return _pass(cid, name, "; ".join(bits))
    msgs: list[str] = []
    if offenders_a:
        shown = "; ".join(offenders_a[:3])
        more = f" (+{len(offenders_a) - 3} more)" if len(offenders_a) > 3 else ""
        msgs.append(
            f"{len(offenders_a)} multi-GB staging row(s) name no mount/staging path within "
            f"+-2 lines ({shown!r}{more}) — name the staging path + the filesystem it "
            "resolves to next to the row (the CLAUDE.md compute-character element 5: "
            "a correct GB figure on the WRONG mount still ENOSPCs, #1393; a distant "
            "`/workspace` in Repro boilerplate does not bind the row)"
        )
    if arm_b_offends:
        msgs.append(
            f"{len(bind_hits)} worktree-bind citation line(s) route a multi-GB stage "
            "via the #681 worktree bind with NO `findmnt --mountpoint` liveness "
            "assertion — the bind is NOT live on this VM (#2091: cited for a 42 GB "
            "stage, PASSed verify_plan twice; the path then resolves to the boot "
            "disk, gotchas.md `/mnt/eps-data` entry). Add a fenced "
            "`findmnt --mountpoint <repo>/.claude/worktrees` probe (no output = no "
            "bind) or re-route the staging path"
        )
    msgs.append(
        "or declare `N/A — no multi-GB staging` on its own line, unwrapped "
        "(no backticks/quotes), if the staging vocabulary is incidental"
    )
    return _warn(cid, name, " | ".join(msgs))


# ─── Check 57 — fan-out same-prefix staging shape (#2236, incident #1739) ───

# Trigger (conjunctive): T1 — a STRIPPED §9 line declaring a box-level
# fan-out (count 2-99 + shards/boxes/pods/nodes/instances) with same-line
# parallel/concurrent vocabulary and no same-line negation; AND T2 —
# Hub-prefix staging named anywhere in the plan. Satisfier: T3 — staging-
# remedy vocabulary anywhere in the plan (pre-stage / serialize / jitter —
# the remedies of `.claude/rules/plan-compute-sizing.md` § "Fan-out over
# the same HF prefix"). Calibration (#2236, implementation-time, AS-SHIPPED
# regexes; the c39/c33 gate precedent — any future c57-regex change re-runs
# the corpus scan and records the realized numbers here;
# `scripts/issue2236_c57_corpus_sweep.py` is the reproduction tool) over
# 3,935 persisted plan-versions (tasks/*/*/plans/v*.md, 2026-08-11):
# 6 WARNs / 2 distinct tasks, all true positives —
#   - TP #2054 v13/v14 — the founding shape: 8 concurrent `cpu-bigmem`
#     pods each `stage_hub_prefix`-ing the SAME ~12 GB
#     `issue2054_lattice/activations` prefix, no staging shape named
#     (PASSed the pre-c57 verify_plan 0 FAIL / 0 WARN of 57 checks);
#   - TP #1491 v1-v4 — up to ~48 concurrent shards staging HF data-repo
#     paths; the only named mitigation is UPLOAD commit-rate batching, so
#     the DOWNLOAD staging duty is genuinely unaddressed.
# Negative control: #2054 v12 (the R4/R5 multi-pod staging precedent the
# Methodology critic judged functionally acceptable) does NOT fire.
# n_skip (no parseable section-9 heading) = 673 under the §-tolerant
# locator. FPs eliminated during predicate development, each by a named
# lever: #460 v2 matched `4 workers` on a SINGLE 4x H100 pod (-> box-level
# noun set, no `workers`); #507 v1/v2 matched an assumptions-table row
# ("2 concurrent pods") while the sweep is explicitly sequential (-> the
# section-9 window confinement); #552 v1 matched "one multi-GPU pod, NOT
# 3 pods, per the standing rule" — an explicitly REJECTED fan-out — so the
# plan-spec'd negation list gained the `not <n> pods` form at
# implementation time (re-measured: 6 WARNs / 0 FP).
# Regex-change record (review round 1, per this block's own contract): the
# initially-shipped strict `^9[.):\s]` locator skipped 897 plans as
# "no parseable section 9", but >=224 of those carried a literal
# `§9 <title>` house-style heading (13% of plans with id >= 2000 were
# invisible to the check — the exact future-plan population the net
# serves). Both the opener AND the window-closer regexes are now
# §-tolerant (`^(?:§\s*)?...`) — the closer tolerance is load-bearing: a
# `## §10` heading that failed to CLOSE the window would run it to EOF and
# become a false-positive surface. Corpus re-scan after the change:
# WARN set byte-identical (same 6 files), pass=2, no-heading skips
# 897 -> 673 (the residual 673 genuinely lack a `9.`/`§9`-form heading).
_C57_FANOUT_RE = re.compile(
    r"(?i)\b([2-9]|[1-9][0-9])\s*(×\s*)?(pod |parallel |concurrent )?"  # noqa: RUF001 — the multiplication sign is real plan text
    r"(shards?|boxes|pods|nodes|instances)\b"
)
_C57_CONCURRENT_RE = re.compile(r"(?i)parallel|concurrent(ly)?")
_C57_NEGATION_RE = re.compile(
    r"(?i)sequential|rejected|instead of|rather than|declined|not parallel"
    r"|no fan-out|one at a time"
    r"|\bnot\s+\d+\s*(×\s*)?(shards?|boxes|pods|nodes|instances)\b"  # noqa: RUF001 — the multiplication sign is real plan text
)
_C57_STAGING_RE = re.compile(r"stage_hub_prefix|snapshot_download|hf_hub_download")
_C57_REMEDY_RE = re.compile(
    r"(?i)pre-stage|prestage|stage[ds]? once|shared read path"
    r"|rsync (after|the|from)|baked image|serializ\w* (the )?(pull|stag|download)"
    r"|stagger|jittered start|start-offset"
)
_C57_SECTION9_RE = re.compile(r"^(?:§\s*)?9[.):\s]")
_C57_NUMBERED_HEADING_RE = re.compile(r"^(?:§\s*)?(\d+)[.):\s]")


def _c57_section9_window(lines: list[str], mask: list[bool]) -> tuple[int, int] | None:
    """Line span [start, end) of the section-9 window: the first non-fenced
    heading whose text starts `9.` / `9)` / `9:` OR the house-style
    `§9 <title>` form (review round 1: >=224 of the strict locator's 897
    corpus skips carried a literal `§9`-prefixed heading — 13% of recent
    plans were invisible to the check), up to the next numbered heading
    whose leading integer is not 9 (so `9.1`-style subsections stay INSIDE
    the window). The closer is §-tolerant too — load-bearing, not
    symmetry: a `## §10` heading that failed to CLOSE the window would run
    it to EOF and become a false-positive surface. None when no such
    heading parses."""
    heads: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _HEADING_RE.match(line.strip())
        if m:
            heads.append((i, m.group(2).strip()))
    start = None
    for i, text in heads:
        if _C57_SECTION9_RE.match(text):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i, text in heads:
        if i <= start:
            continue
        m = _C57_NUMBERED_HEADING_RE.match(text)
        if m and int(m.group(1)) != 9:
            end = i
            break
    return start, end


def _c57_match_snippet(line: str, width: int = 90) -> str:
    """A ~``width``-char window of ``line`` centered on the T1 fan-out MATCH
    SPAN — never the line head: on #1491 the match sits at char ~785 of a
    922-char line, so a head-slice printed unrelated serial-fit prose and
    the WARN evidence read as spurious (review round 1, Minor)."""
    m = _C57_FANOUT_RE.search(line)
    if m is None:  # defensive: trigger lines matched by construction
        return line.strip()[:width]
    lo = max(0, m.start() - (width - (m.end() - m.start())) // 2)
    return line[lo : lo + width].strip()


def check_fanout_prefix_staging(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: a section-9 fan-out of N > 1
    CONCURRENT boxes/pods/shards, in a plan that also stages an HF prefix
    (`stage_hub_prefix` / `snapshot_download` / `hf_hub_download`), must
    name its staging shape — pre-stage once and fan, serialized per-box
    pulls, or jittered start offsets (per
    `.claude/rules/plan-compute-sizing.md` § "Fan-out over the same HF
    prefix"; incident #1739: three boxes each staged ~144 GB from one
    prefix simultaneously, five attempts to land one leg). NEVER FAILs
    (the c39/c43/c46/c50/c54 fail-open convention); a plan with no
    parseable section-9 heading (neither the numbered `9.` form nor the
    house-style `§9 <title>` form) is a SKIP, counted separately from
    passes.

    Three points of honesty. (1) The calibrated 0-FP figure above is
    IN-SAMPLE — the regexes were tuned on the same persisted-plan corpus
    the acceptance sweep re-runs — so it bounds nuisance cost on
    yesterday's planner distribution, not on future plans. (2) The naive
    whole-document predicate WARNed 100 plans, which is why this ships
    confined to the section-9 window; the disclosed cost of that
    confinement is a fan-out declared ONLY outside section 9, which is
    invisible to c57 — a c52-style named residual, never to be read as
    coverage. (3) WARN-only because whether N shard rows each PULL the
    prefix or read a shared path is finally a property of the DISPATCHER,
    not the plan text — the Methodology lens item 16 FAN-OUT STAGING
    EXTENSION is the binding gate; c57 is only the early-warning net."""
    del kind  # all kinds: the staging-topology duty is kind-agnostic here
    cid, name = "c57_fanout_prefix_staging", "fan-out same-prefix staging shape"
    lines = plan.splitlines()
    mask = _fence_mask(lines)  # line-count-preserving: indexes map raw<->stripped
    window = _c57_section9_window(lines, mask)
    if window is None:
        return _skip(
            cid,
            name,
            "no parseable section-9 heading (`9.` / `§9` opener ... next numbered "
            "heading) — a fan-out in a plan with no such heading is a named "
            "residual, not coverage",
        )
    lo, hi = window
    trigger_idx = [
        i
        for i in range(lo, hi)
        if not mask[i]
        and _C57_FANOUT_RE.search(lines[i])
        and _C57_CONCURRENT_RE.search(lines[i])
        and not _C57_NEGATION_RE.search(lines[i])
    ]
    if not trigger_idx:
        return _skip(cid, name, "no concurrent box-level fan-out declared in section 9")
    if not _C57_STAGING_RE.search(plan):
        return _skip(
            cid,
            name,
            "section-9 fan-out present but no Hub-prefix staging vocabulary "
            "(stage_hub_prefix / snapshot_download / hf_hub_download) anywhere in the plan",
        )
    remedy = _C57_REMEDY_RE.search(plan)
    if remedy:
        return _pass(
            cid,
            name,
            f"{len(trigger_idx)} section-9 fan-out line(s) with a staging-shape "
            f"remedy named ({remedy.group(0)!r})",
        )
    shown = "; ".join(_c57_match_snippet(lines[i]) for i in trigger_idx[:3])
    more = f" (+{len(trigger_idx) - 3} more)" if len(trigger_idx) > 3 else ""
    return _warn(
        cid,
        name,
        f"{len(trigger_idx)} section-9 line(s) fan N > 1 concurrent boxes/pods/shards "
        f"({shown!r}{more}) while the plan stages an HF prefix, and no staging shape is "
        "named — N concurrent same-prefix multi-GB pulls are a rate-limit kill risk "
        "(429s / rc=137 resets; #1739: three boxes staged ~144 GB from one prefix "
        "simultaneously, five attempts to land one leg). Name ONE of: pre-stage once "
        "and fan (shared read path / rsync after one stage completes / baked image), "
        "serialized per-box pulls, or jittered start offsets — or mark the pulls "
        "explicitly sequential on the fan-out row (per "
        "`.claude/rules/plan-compute-sizing.md` § Fan-out over the same HF prefix; "
        "the Methodology lens item 16 FAN-OUT STAGING EXTENSION is the binding gate)",
    )


# ─── Check 58 — fan-out RunPod pod-name collision (#2237, incident #2054) ──

# Trigger (conjunction of three arms; all three must hold to WARN):
#   T1 multiplicity (either sub-arm): T1a — >=1 non-fenced section-9 line
#     declaring N > 1 CONCURRENT pods/shards/boxes/nodes/instances,
#     negation-free (c57's detector — `_C57_FANOUT_RE` /
#     `_C57_CONCURRENT_RE` / `_C57_NEGATION_RE` over the
#     `_c57_section9_window`, reused verbatim; NO third fan-out detector);
#     OR T1b — >=2 DISTINCT dry-parsing RunPod-resolved launch argvs (the
#     converse shape: multiplicity expressed in argvs, absent from §9
#     prose vocabulary).
#   T2 lane: >=1 plan-embedded launch-shaped argv that dry-parses (the
#     c46/c50/c52 argv chain, reused verbatim) AND resolves to RunPod
#     under the SHIPPED posture (`_C58_T2_INCLUDE_AUTO` below).
#   T3 remedy absence: the plan names NO per-pod provisioning mechanism.
# T3's key is the literal `pod.py provision` COMMAND CONSTRUCT carrying
# `--name-suffix` on the same line — NOT the bare `--name-suffix` token
# and NOT the bare noun "provision": #2054 v16 line 298 carries BOTH
# (inside its teardown command `pod.py terminate ... --name-suffix` and
# the pod-safety prose "BEFORE the first provision") on ONE line, so a
# naive `provision\w*.*--name-suffix` MATCHES it (measured) and would
# suppress the WARN on the very fixture the check exists to catch, while
# `pod.py provision` matches nowhere in v16 (measured). The other
# remedies the lens clause names are STRUCTURAL here, not regex-keyed:
# one-pod-with-N-in-pod-workers never trips T1a (`workers` is not in the
# c57 noun set — the #460 FP lever); a name-isolating GCP/SLURM lane
# never trips T2 (no RunPod-resolved argv); explicit serialization is
# absorbed by T1a's negation arm. `--lane-suffix` on a RunPod argv does
# NOT satisfy anything — RunPod is exactly the lane
# `dispatch_issue._lane_suffix_honored_kinds()` excludes (pinned by
# test 4 in tests/test_verify_plan_c58_fanout_pod_name.py).
#
# T2 posture calibration (#2237 §7, implementation-time, AS-SHIPPED
# regexes, over 3,941 persisted plan-versions tasks/*/*/plans/v*.md at
# origin/main 5eba504a7d, 2026-08-11 — the c39/c33/c57 gate precedent:
# any future c58-arm change re-runs the corpus scan and records the
# realized numbers here; `scripts/issue2237_c58_corpus_sweep.py` is the
# reproduction tool and measures BOTH postures every run). Measured:
#   - explicit-only (T2 = `--backend runpod` pins only): 8 WARNs /
#     2 distinct tasks. TRUE POSITIVES (7): #2054 v13-v16 via T1a (the
#     founding unsatisfiable §9 — "8/10 shards on N parallel
#     `cpu-bigmem` pods" + explicit runpod pins, no minting mechanism;
#     v16 is the incident fixture) and #2054 v10-v12 via T1b (the
#     parent R4/R5 rounds: 3-4 distinct runpod argvs whose "x4 each,
#     parallel" / "6 parallel pod jobs" multiplicity lives in FENCED
#     bash comments T1a's mask correctly skips — the same gap, caught
#     by the argv arm). FALSE POSITIVE (1): #2203 v6 via T1b — 3
#     distinct runpod argvs that are SEQUENTIAL phase provisions
#     (1xH100 -> 4xH100 -> 1xH200), not a concurrent fan-out; the named
#     T1b FP class (sequential multi-phase multi-launch plans),
#     absorbed by WARN-only polarity.
#   - explicit+auto (T2 also counts `--backend` absent/`auto`, the
#     runpod-first order since #2054): 33 WARNs / 7 distinct tasks —
#     the 8 above + 25 additions (#1417 v1-v7, #1774 v1-v4, #2163
#     v1-v6, #810 v13-v18, #1336 v8-v9), EVERY addition adjudicated a
#     false positive of the same sequential-phases T1b class (Phase
#     A/C, GPU-phase + CPU-phase, E1 + conditional-E2 launch pairs).
#     Over the >~20 corpus ceiling (#2237 §15.2) and FP-dominated.
#   SHIPPED POSTURE: explicit-only (`_C58_T2_INCLUDE_AUTO = False`) —
#   the auto arm's FP class proved large, so per the pre-registered
#   #2237 §4 rule it is DISABLED behind the stated residual (i) rather
#   than silently dropped; the lens clause stays lane-agnostic. n_skip
#   (explicit-only) = 3,933: no-section-9 = 673, no-launch-argv =
#   2,803, none-parses = 246, no-runpod-argv = 195, no-fanout(T1) = 16;
#   pass (remedy named) = 0 on the corpus — the corrected #2054 v17
#   amendment (which names `pod.py provision --name-suffix` per shard)
#   is quiet via the no-launch-argv SKIP (it provisions directly, not
#   through dispatch_issue.py), the intended silence.
# #2145 remedy extension: `dispatch_issue.py launch --lane-suffix <slug>`
# (alias `--name-suffix`) now mints `pod-<N>-<slug>` on the RunPod lane, so a
# plan naming per-shard suffixed launch commands has named a T3 mechanism.
# Gated per the plan's pre-registered condition: the extension ships only
# with a re-run of scripts/issue2237_c58_corpus_sweep.py showing the #2054
# plans/v16.md positive control still WARNs. Re-sweep 2026-08-15 (#2145,
# shipped explicit-only posture, live corpus n=4214): warns=8 (#2054
# v10-v16 + #2203 v6), pass (remedy named) = 0 — the extended alternate
# flipped NOTHING to PASS on the corpus; positive control (#2054 v16
# WARNs) = True; corpus ceiling (warns <= 20) = True; n_skip=4206
# (no-fanout=38, no-launch-argv=2967, no-runpod-argv=199,
# no-section-9=755, none-parses=247).
_C58_REMEDY_RE = re.compile(
    r"pod\.py\s+provision\b[^\n]*--name-suffix"
    r"|dispatch_issue\.py\s+launch\b[^\n]*--(?:lane|name)-suffix"
)
_C58_T2_INCLUDE_AUTO = False  # shipped posture — see the calibration block


def _c58_runpod_resolved(ns, include_auto: bool) -> bool:
    """True when a dry-parsed launch Namespace resolves to the RunPod lane
    under the given T2 posture: explicit ``--backend runpod``, plus (when
    ``include_auto``) an absent/empty/``auto`` backend — the runpod-first
    ``DEFAULT_AUTO_LANE_ORDER`` since #2054."""
    backend = (getattr(ns, "backend", None) or "auto").strip().lower()
    if backend == "runpod":
        return True
    return include_auto and backend == "auto"


def _c58_check(plan: str, include_auto: bool) -> CheckResult:
    """c58 core, posture-parameterized so the calibration sweep can measure
    both T2 postures against the SHIPPED trigger code (the public
    ``check_fanout_pod_name_collision`` pins ``_C58_T2_INCLUDE_AUTO``)."""
    cid, name = "c58_fanout_pod_name_collision", "fan-out RunPod pod-name collision"
    lines = plan.splitlines()
    mask = _fence_mask(lines)  # line-count-preserving: indexes map raw<->stripped
    window = _c57_section9_window(lines, mask)
    if window is None:
        return _skip(
            cid,
            name,
            "no parseable section-9 heading (`9.` / `§9` opener ... next numbered "
            "heading) — a fan-out in a plan with no such heading is residual (iii), "
            "not coverage",
        )
    argvs, notes = _c50_launch_argvs(plan)
    tail = ("; " + "; ".join(notes)) if notes else ""
    if not argvs:
        return _skip(
            cid,
            name,
            "no launch-shaped dispatch_issue.py command in the plan — a custom-driver "
            "fan-out is structurally invisible here (residual (ii), the same channel "
            "c52 discloses); this SKIP is not coverage" + tail,
        )
    parser, load_detail = _c46_argparser()
    if parser is None:
        return _skip(cid, name, f"dispatch_issue.build_argparser unavailable ({load_detail})")
    runpod_argvs: list[list[str]] = []
    argv_notes: list[str] = []
    n_parsed = 0
    for i, argv in enumerate(argvs):
        ns, err = _c46_dry_parse(parser, argv)
        if ns is None:  # per-argv note, never a WARN — c46 arm 1 owns parse drift
            argv_notes.append(f"argv #{i + 1} does not dry-parse ({err})")
            continue
        n_parsed += 1
        if _c58_runpod_resolved(ns, include_auto=include_auto):
            runpod_argvs.append(argv)
    note_tail = ("; " + "; ".join(argv_notes)) if argv_notes else ""
    if n_parsed == 0:
        return _skip(
            cid,
            name,
            "launch argvs present but none dry-parses — c46 arm 1 owns parse "
            "warnings" + note_tail + tail,
        )
    if not runpod_argvs:
        posture = (
            "explicit `--backend runpod` or absent/`auto` (runpod-first, #2054)"
            if include_auto
            else "explicit `--backend runpod` only"
        )
        return _skip(
            cid,
            name,
            f"no RunPod-resolved launch argv under the shipped T2 posture ({posture}) "
            "— residual (i): an off-posture launch landing on RunPod is invisible "
            "here; the lens clause is lane-agnostic and binding" + note_tail,
        )
    lo, hi = window
    trigger_idx = [
        i
        for i in range(lo, hi)
        if not mask[i]
        and _C57_FANOUT_RE.search(lines[i])
        and _C57_CONCURRENT_RE.search(lines[i])
        and not _C57_NEGATION_RE.search(lines[i])
    ]
    t1b = len(runpod_argvs) >= 2  # _c50_launch_argvs already dedupes on tuple(argv)
    if not trigger_idx and not t1b:
        return _skip(
            cid,
            name,
            "no concurrent box-level fan-out declared in section 9 (T1a) and fewer "
            "than 2 distinct RunPod-resolved launch argvs (T1b)",
        )
    remedy = _C58_REMEDY_RE.search(plan)
    if remedy:
        return _pass(
            cid,
            name,
            f"RunPod fan-out with a per-pod provisioning construct named "
            f"({remedy.group(0)[:70]!r})",
        )
    if trigger_idx:
        shown = "; ".join(_c57_match_snippet(lines[i]) for i in trigger_idx[:3])
        more = f" (+{len(trigger_idx) - 3} more)" if len(trigger_idx) > 3 else ""
        evidence = f"{len(trigger_idx)} section-9 fan-out line(s) ({shown!r}{more})"
    else:
        evidence = (
            f"{len(runpod_argvs)} distinct RunPod-resolved launch argvs (T1b — "
            "multiplicity in argvs, no section-9 fan-out prose)"
        )
    return _warn(
        cid,
        name,
        f"{evidence} while every SUFFIX-LESS RunPod-lane launch mints the SAME pod "
        "name — `backends/runpod.py` `_runpod_pod_name(issue, name_suffix=None)` "
        "defaults to the bare `pod-<N>`; since #2145 `dispatch_issue.py launch "
        "--lane-suffix <slug>` (alias `--name-suffix`) IS honored on the RunPod "
        "lane and mints `pod-<N>-<slug>` — so N concurrent same-issue launches "
        "WITHOUT distinct suffixes collide; the dangerous branch is silent "
        "co-location of all N shards on ONE pod, invalidating every per-shard "
        "wall/RSS projection the plan booked (#2054 v16, caught only at dispatch). "
        "Name ONE of: per-launch `dispatch_issue.py launch --lane-suffix <slug>` "
        "(distinct slug per shard); per-pod `pod.py provision --issue <N> "
        "--name-suffix <slug>` calls; one pod with N in-pod workers; a "
        "name-isolating GCP/SLURM `--lane-suffix` lane; or mark the fan-out "
        "explicitly serialized (the Methodology lens item 16 FAN-OUT POD-NAME "
        "EXTENSION is the binding gate; c58's silence is not evidence of safety "
        "— residual (i))",
    )


def check_fanout_pod_name_collision(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: a section-9 fan-out of N > 1
    CONCURRENT pods (T1a — c57's detector, reused verbatim) OR >= 2
    DISTINCT RunPod-resolved launch argvs (T1b, the converse shape), in a
    plan whose plan-embedded ``dispatch_issue.py`` launch commands resolve
    to the RunPod lane (T2), must name a mechanism that mints N DISTINCT
    pod names (T3 — the literal ``pod.py provision ... --name-suffix``
    command construct, or — since #2145 — a
    ``dispatch_issue.py launch ... --lane-suffix``/``--name-suffix``
    command). A SUFFIX-LESS RunPod launch mints the SAME name for every
    shard: ``backends/runpod.py`` ``_runpod_pod_name(issue,
    name_suffix=None)`` defaults to the bare ``f"pod-{issue}"``; since
    #2145 ``--lane-suffix`` (alias ``--name-suffix``) IS honored on the
    RunPod lane (``dispatch_issue._lane_suffix_honored_kinds`` includes
    ``runpod``) and mints ``pod-<N>-<slug>`` — so N concurrent same-issue
    launches WITHOUT distinct suffixes collide — best case an error,
    worst case silent co-location of all N shards on ONE pod,
    invalidating every per-shard wall/RSS projection the plan booked
    (#2054 plan v16: a 10-way ``cpu-bigmem`` RunPod fan-out whose own
    pod-safety paragraph required ``pod-2054-rb789-<shard>`` names; it
    PASSed verify_plan 0 FAIL / 0 WARN twice plus two critic-round
    APPROVEs and was caught only at dispatch). NEVER FAILs (the
    c39/c43/c46/c50/c52/c54/c57 fail-open posture); every ambiguity SKIPs
    with a stated reason, and a SKIP is never coverage.

    Named residuals, disclosed in the c52/c57 honesty convention — none
    is ever to be read as coverage:
    (i) AUTO-LANE residual (PRIMARY): the shipped T2 posture counts
    explicit ``--backend runpod`` pins ONLY — the explicit+auto posture
    measured 25 additional WARNs, every one an adjudicated false
    positive (the calibration block above), so the auto arm is DISABLED
    behind this residual per the pre-registered #2237 §4 rule. A
    bare-``auto``/absent-backend fan-out that lands on RunPod (the
    runpod-first ``DEFAULT_AUTO_LANE_ORDER``, #2054), a frontmatter-pin
    route, or non-``dispatch_issue.py`` tooling is therefore INVISIBLE
    to c58; the Methodology lens item 16 FAN-OUT POD-NAME EXTENSION is
    lane-agnostic and is the binding gate.
    (i-bis) WIDTH-ACHIEVABILITY: neither arm verifies a declared width is
    ACHIEVABLE — T1a only reads prose, T1b only counts argvs; a plan
    declaring 10 shards with 2 template argvs satisfies T1 but c58 cannot
    tell 10 from 2 (v16's multiplicity lives in prose + a "repeat over
    shards" loop instruction); the lens clause's mint-N-DISTINCT-names
    requirement is what covers it. (ii) CUSTOM-DRIVER fan-out: a fan-out
    driven by a bespoke script with no plan-embedded
    ``dispatch_issue.py launch`` argv SKIPs at T2 — the same channel c52
    discloses. (iii) SECTION-9 CONFINEMENT: a fan-out declared ONLY
    outside section 9 is invisible to T1a — c57's disclosed
    window-confinement cost, inherited with its detector (T1b partially
    compensates, argv-count multiplicity only). (iv) IN-SAMPLE
    CALIBRATION: the FP figures in the calibration block above are
    measured on the same persisted-plan corpus the acceptance sweep
    re-runs — they bound nuisance cost on yesterday's planner
    distribution, not on future plans.

    WARN-only, not FAIL, because whether N shard rows each provision
    their own pod or share one is finally a property of the DISPATCHER,
    not the plan text — and residual (i) means c58's silence is not
    evidence of safety: the lens clause is the binding gate and c58 is
    only the early-warning net (c57's stated polarity rationale)."""
    del kind  # all kinds: colliding pod names bill/confound identically everywhere
    return _c58_check(plan, include_auto=_C58_T2_INCLUDE_AUTO)


# ─── Check 59 — GPU-hours token consumer/declaration conflict ──────────────
# Declaration-shaped token (#2123 §3.3): the `Estimated GPU-hours (total):
# <number>` token LINE-ANCHORED with nothing else on the line — optional
# bullet / blockquote markers (stripped by the caller), optional bold
# (`**`) / backtick wrap admitted by the character class — over
# fence-stripped text. The value grammar is GPU_LINE_RE's (a single plain
# number). Deliberately NARROW: a count-based predicate over ALL token
# occurrences FAILs 62.2% of the plan corpus (2468/3971 — repeats are
# overwhelmingly legitimate), and distinct-RAW-values flags 11 files that
# are ALL false positives (prose quotes #2177 v1:56, meta-discussion
# #625 v1-v3, revision-comparison tables #524 v6/v7:19).
_C59_DECL_LINE_RE = re.compile(
    r"(?i)^[`*\s]*estimated\s+gpu-?hours\s+\(total\):[`*\s]*([0-9]+(?:\.[0-9]+)?)[`*\s]*$"
)


def _c59_declared_values(plan: str) -> list[str]:
    """Every DECLARATION-SHAPED ``Estimated GPU-hours (total): <n>`` value
    (raw captured string), in document order, over FENCE-STRIPPED text —
    the declaration side strips fences (a fenced example command is not a
    declaration); the CONSUMER side deliberately does not (see
    :func:`check_gpu_hours_token_conflict`)."""
    values: list[str] = []
    for line in strip_fences(plan).splitlines():
        m = _C59_DECL_LINE_RE.match(line.lstrip(" \t>-"))
        if m is not None:
            values.append(m.group(1))
    return values


def check_gpu_hours_token_conflict(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: when a plan carries at least one
    DECLARATION-SHAPED ``Estimated GPU-hours (total):`` line (line-anchored,
    optional bullet/bold/backtick wrap, nothing else on the line, fenced
    blocks stripped — :data:`_C59_DECL_LINE_RE`), two arms compare (#2123):

    * **Arm A** — more than one DISTINCT declaration-shaped value (corpus
      2026-08-12: 2/3971 versioned plan files — #524 v2/v3, 590 vs 900).
    * **Arm B** — the value a FIRST-MATCH consumer reads differs from the
      first declaration-shaped value (corpus: 5/3971, zero false
      positives — incl. the motivating #2061 v7/v8: first-match 70 vs
      declared 80; #2177 v1: 65 vs 0). Consumer fidelity is load-bearing:
      the real consumer is ``GPU_LINE_RE.search`` on the RAW plan (the
      ``task.py`` Step 2c gate + every marker note that records the
      estimate), so Arm B re-runs ``GPU_LINE_RE`` on RAW
      (un-fence-stripped) text — deriving the consumer value from
      fence-stripped text would validate a consumer that does not exist.

    Hard constraint: 646 corpus files carry a matching token but NO
    declaration-shaped token — the check NEVER requires declaration shape;
    it compares only when at least one exists and SKIPs otherwise (c5's
    presence contract is untouched).

    WARN-only, never FAIL: ``_resolve_autonomous_plan_gate`` is
    GPU-HOUR-BLIND as of #1771 (auto-approves any parseable estimate,
    parks only on None), so a mis-read cannot flip the gate DECISION — the
    harm is a corrupted RECORDED estimate (the value riding the status
    marker note, the watcher's re-passed ``auto_approve_gpu_hours``, and
    what a reviewer skims off the plan): a provenance defect, not a
    spend-control defect, squarely under the
    c39/c43/c46/c50/c52/c54/c57/c58 fail-open convention. Escalation to
    FAIL is a later separately-justified step (the c26/c29/c33 precedent).

    Named residual (the c52/c57 honesty convention): a conflicting value
    appearing ONLY in non-declaration shapes on both sides is invisible to
    c59 — its silence is not evidence of a single consistent estimate.

    Escape: the standalone declaration line
    ``N/A — GPU-hours token conflict reconciled``."""
    del kind  # all kinds: the consumer parses every task's plan identically
    cid, name = "c59_gpu_hours_token_conflict", "GPU-hours token consumer/declaration conflict"
    if _standalone_na_declared(plan, r"GPU[- ]?hours token conflict reconciled"):
        return _skip(cid, name, "escape declared: `N/A — GPU-hours token conflict reconciled`")
    decls = _c59_declared_values(plan)
    if not decls:
        return _skip(
            cid,
            name,
            "no declaration-shaped `Estimated GPU-hours (total):` line (646-file corpus "
            "class) — c59 never requires declaration shape; presence stays c5's contract",
        )
    problems: list[str] = []
    distinct = sorted({float(v) for v in decls})
    if len(distinct) > 1:
        shown = ", ".join(f"{v:g}" for v in distinct)
        problems.append(
            f"arm A: {len(distinct)} DISTINCT declaration-shaped values ({shown}) — a reader "
            f"cannot tell which is the estimate of record"
        )
    consumer = GPU_LINE_RE.search(plan)  # RAW text — the real consumer's own read
    if consumer is not None and float(consumer.group(1)) != float(decls[0]):
        problems.append(
            f"arm B: the first-match consumer (GPU_LINE_RE on the raw plan — the Step 2c "
            f"gate's read) resolves {consumer.group(1)} while the first declaration-shaped "
            f"value is {decls[0]} — the RECORDED estimate (status-marker note, watcher "
            f"auto_approve_gpu_hours re-pass) is corrupted (#2061 v7/v8: 70-for-80; "
            f"#2177 v1: 65-for-0)"
        )
    if not problems:
        return _pass(
            cid,
            name,
            f"{len(decls)} declaration-shaped value(s), consumer first-match agrees ({decls[0]})",
        )
    return _warn(
        cid,
        name,
        "; ".join(problems)
        + " — keep ONE declaration-shaped value and move every other mention mid-sentence / "
        "into a wrapped or fenced form, or declare the standalone escape "
        "`N/A — GPU-hours token conflict reconciled`",
    )


# ─── Check 61 — SLURM would-render --mem vs declared RSS peak (#2275) ──────

# Within-job width tokens (the #2275 plan's registered conservative regexes —
# a fleet-total "12 legs" with no in-parallel qualifier deliberately stays
# out; only widths sharing ONE job cgroup should multiply the peak).
_C61_WIDTH_RES = (
    re.compile(r"(?i)\b(\d+)[- ]wide\b"),
    re.compile(r"(?i)\bwidth[= ]?(\d+)\b"),
    re.compile(r"(?i)\b(\d+)\s+(?:units|legs|fits|workers)\s+in\s+parallel\b"),
)


def _c61_ram_peak_lines(plan: str) -> list[tuple[int, float, str]]:
    """``(line index, GiB value, line)`` triples for RAM-token peak lines —
    the RAM arm of ``_c52_declared_peaks`` kept PER LINE, so the c61
    aggregate arm can pair each peak with a within-job width token on the
    SAME line/paragraph (the #1336 shape: N units share one job cgroup)."""
    out: list[tuple[int, float, str]] = []
    for i, line in enumerate(plan.splitlines()):
        m = _C52_RAM_TOKEN_RE.search(line)
        if m is None:
            continue
        val = _c52_nearest_gib(line, m.start())
        if val is not None:
            out.append((i, val, line))
    return out


def _c61_width_for(lines: list[str], idx: int) -> tuple[int, str] | None:
    """Within-job width token on RSS line ``idx``, else in its paragraph
    (the contiguous non-blank block around it). Returns
    ``(width, matched token)`` or ``None``; same-line hits win, then the
    LARGEST width in the paragraph (conservative aggregate); width 1 is
    not a fan-out."""
    lo = idx
    while lo > 0 and lines[lo - 1].strip():
        lo -= 1
    hi = idx
    while hi + 1 < len(lines) and lines[hi + 1].strip():
        hi += 1
    order = [idx] + [j for j in range(lo, hi + 1) if j != idx]
    for j in order:
        hits = [
            (int(m.group(1)), m.group(0)) for rx in _C61_WIDTH_RES for m in rx.finditer(lines[j])
        ]
        hits = [(w, t) for w, t in hits if w > 1]
        if hits:
            return max(hits)
    return None


def _c61_would_render_mem(ns) -> tuple[int | None, str]:
    """The ``--mem`` (G) the SLURM renderer WOULD emit for parsed launch
    ``ns`` — computed through the renderer's OWN
    ``slurm._resource_header_lines`` (the c50 exact-parity idiom), so the
    read carries post-#2275 semantics for free: an argv ``--min-ram-gb``
    raises the render via ``_apply_min_ram``, and a requirement above
    ``mem_gb_cap`` raises (already fail-fast at dispatch — SKIP-class
    here). Cluster = the argv's explicit SLURM ``--backend`` pin when it
    names a known cluster, else fellows (the first SLURM lane in
    ``DEFAULT_AUTO_LANE_ORDER``); GPU count = ``--gpus`` when declared,
    else the intent's ``_DEFAULT_GPUS_FOR_INTENT`` row. Returns
    ``(mem_gb | None, detail)`` — ``None`` means unresolvable, with the
    stated reason."""
    try:
        from explore_persona_space.backends import RunSpec
        from explore_persona_space.backends.slurm import (
            _DEFAULT_GPUS_FOR_INTENT,
            _resource_header_lines,
            get_cluster_config,
        )
    except Exception as exc:  # off-repo --plan-file run -> loud SKIP-class note
        return None, f"slurm renderer unavailable ({type(exc).__name__}: {exc})"
    intent = str(getattr(ns, "intent", ""))
    if intent not in _DEFAULT_GPUS_FOR_INTENT:
        return None, (
            f"intent {intent!r} has no _DEFAULT_GPUS_FOR_INTENT row — "
            "slurm.default_gpus_for_intent() already fails fast at dispatch"
        )
    gpus = int(getattr(ns, "gpus", None) or _DEFAULT_GPUS_FOR_INTENT[intent])
    backend = getattr(ns, "backend", None)
    cluster_name = backend if backend in ("fellows", "nibi", "fir", "mila") else "fellows"
    try:
        cluster = get_cluster_config(cluster_name)
        spec = RunSpec(
            issue=0,
            intent=intent,
            backend="cluster",
            cluster=cluster_name,
            extra=({"min_ram_gb": int(ns.min_ram_gb)} if getattr(ns, "min_ram_gb", None) else {}),
        )
        header = "\n".join(_resource_header_lines(cluster, spec, gpus))
    except Exception as exc:  # incl. the #2275 > mem_gb_cap pre-submit refusal
        return None, f"renderer refuses ({type(exc).__name__}: {exc})"
    m = re.search(r"--mem=(\d+)G", header)
    if m is None:
        return None, "renderer emitted no --mem line"
    return int(m.group(1)), f"{cluster_name}/{intent} at {gpus} GPU(s)"


def check_slurm_mem_coverage(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds: for EVERY plan-embedded
    launch-shaped ``dispatch_issue.py`` argv that dry-parses and resolves a
    SLURM-reachable route (``dispatch_issue._slurm_lane_reachable`` — the
    c50 runtime-parity predicate), the plan's own declared per-leg RSS
    peak (``_c52_declared_peaks``'s RAM arm, per line) must fit the
    WOULD-RENDER ``#SBATCH --mem`` — the renderer's own
    ``min(mem_gb_per_gpu x gpus, mem_gb_cap)`` / CPU-table formula PLUS
    the argv's ``--min-ram-gb`` when present (post-#2275 semantics,
    computed through ``slurm._resource_header_lines`` itself). Two WARN
    arms: (a) PER-LEG — the declared peak strictly exceeds the
    would-render ``--mem`` with no covering ``--min-ram-gb``; (b)
    AGGREGATE (the #1336 shape) — a within-job width token
    (``N-wide`` / ``width=N`` / ``N units|legs|fits|workers in
    parallel``, ``_C61_WIDTH_RES``) on the SAME line/paragraph as the RSS
    token multiplies the peak (N units share ONE job cgroup), and
    ``peak x width`` exceeds the would-render ``--mem`` — WARN naming the
    arithmetic. Remedy named either way: ``--min-ram-gb <requirement>``
    (the #2275 renderer raises ``--mem`` to it, refusing pre-submit above
    ``mem_gb_cap``). Every ambiguity SKIPs with a stated reason; the
    check NEVER FAILs (the c46/c50/c52 posture).

    Scope split vs c52: c52 compares declared peaks against the GCP
    LADDER-RUNG constants (85 GiB host-RAM / 38 GiB VRAM) — #1336's
    65-70 GiB per-unit RSS sits BELOW that rung while 8 units share one
    SLURM job cgroup, so c52 structurally cannot catch the
    aggregate-on-one-node case; c61 compares against the SLURM lane's
    OWN rendered ``--mem``. Named residual (the c52 residual (ii)
    posture): a fan-out driven by a CUSTOM DRIVER with no plan-embedded
    ``dispatch_issue.py launch`` argv is structurally invisible — the
    binding surface is `.claude/rules/plan-compute-sizing.md`
    § Ladder-rung RAM floor, and a c61 SKIP is never read as coverage.
    Second named residual: the fleet-total-vs-per-leg token ambiguity
    inherited from the c52 extractor (WARN-only polarity absorbs it; the
    message names the extracted line).
    """
    del kind  # all kinds: a SLURM cgroup OOM kills identically everywhere
    cid, name = "c61_slurm_mem_coverage", "SLURM would-render --mem vs declared RSS peak"
    peak_lines = _c61_ram_peak_lines(plan)
    if not peak_lines:
        return _skip(cid, name, "no per-leg RSS / host-RAM peak-estimate token in the plan")
    argvs, notes = _c50_launch_argvs(plan)
    tail = ("; " + "; ".join(notes)) if notes else ""
    if not argvs:
        return _skip(
            cid,
            name,
            "no launch-shaped dispatch_issue.py command in the plan — a custom-driver "
            "fan-out is structurally invisible here (docstring residual): the binding "
            "surface is plan-compute-sizing.md § Ladder-rung RAM floor, and this SKIP is "
            "not coverage" + tail,
        )
    parser, load_detail = _c46_argparser()
    if parser is None:
        return _skip(cid, name, f"dispatch_issue.build_argparser unavailable ({load_detail})")
    reachable_fn, reach_detail = _c50_slurm_lane_reachable_fn()
    if reachable_fn is None:
        return _skip(
            cid, name, f"dispatch_issue._slurm_lane_reachable unavailable ({reach_detail})"
        )
    max_peak = max(v for _, v, _ in peak_lines)
    agg = _c61_max_aggregate(plan.splitlines(), peak_lines)
    return _c61_verdict(cid, name, argvs, parser, reachable_fn, max_peak, agg, tail)


def _c61_max_aggregate(
    lines: list[str], peak_lines: list[tuple[int, float, str]]
) -> tuple[float, int, float, str] | None:
    """Largest (aggregate GiB, width, per-leg peak, width token) over peak lines
    carrying a same-line/paragraph width token; None when no width token pairs."""
    agg: tuple[float, int, float, str] | None = None
    for idx, val, _line in peak_lines:
        w = _c61_width_for(lines, idx)
        if w is not None:
            cand = (val * w[0], w[0], val, w[1])
            if agg is None or cand[0] > agg[0]:
                agg = cand
    return agg


def _c61_eval_argv(
    parser,
    reachable_fn,
    i: int,
    argv: list[str],
    max_peak: float,
    agg: tuple[float, int, float, str] | None,
) -> tuple[str, str, str | None]:
    """Evaluate ONE launch argv for c61. Returns (kind, text, warn) where kind is
    'skip' (loud SKIP the whole check: text = reason), 'note' (per-argv note),
    or 'ok' (text = the rendered --mem line; warn = the WARN text or None)."""
    ns, err = _c46_dry_parse(parser, argv)
    if ns is None:  # per-argv note, never a WARN — c46 arm 1 owns parse drift
        return ("note", f"argv #{i + 1} does not dry-parse ({err}) — c46 arm 1 owns that", None)
    try:
        reachable = reachable_fn(ns)
    except Exception as exc:  # router import failure on off-repo runs -> loud SKIP
        return ("skip", f"SLURM reachability unresolvable ({type(exc).__name__}: {exc})", None)
    if not reachable:
        return (
            "note",
            f"argv #{i + 1}: no SLURM lane reachable for backend "
            f"{(getattr(ns, 'backend', None) or 'auto')!r}",
            None,
        )
    mem_gb, detail = _c61_would_render_mem(ns)
    if mem_gb is None:
        return ("note", f"argv #{i + 1}: would-render --mem unresolvable ({detail})", None)
    label = _c52_argv_label(i, argv)
    warn: str | None = None
    if max_peak > mem_gb:
        warn = (
            f"plan declares per-leg peak RSS ~{max_peak:g} GiB but {label} would render "
            f"#SBATCH --mem={mem_gb}G ({detail}) — the SLURM job cgroup OOM-kills at "
            f"that cap (the #1336 shape): add --min-ram-gb {math.ceil(max_peak)} to the "
            f"launch command (#2275 raises the rendered --mem to it)"
        )
    elif agg is not None and agg[0] > mem_gb:
        warn = (
            f"plan declares a within-job aggregate of {agg[2]:g} GiB x width {agg[1]} "
            f"({agg[3]!r}) = {agg[0]:g} GiB but {label} would render #SBATCH "
            f"--mem={mem_gb}G ({detail}) — N units share ONE SLURM job cgroup, so the "
            f"AGGREGATE binds (the #1336 shape: 8 pooled fit units OOM-killed at the "
            f"GPU-count-derived cap): add --min-ram-gb {math.ceil(agg[0])} to the launch "
            f"command (#2275 raises the rendered --mem to it)"
        )
    return ("ok", f"{label}: --mem={mem_gb}G ({detail})", warn)


def _c61_verdict(
    cid: str,
    name: str,
    argvs: list[list[str]],
    parser,
    reachable_fn,
    max_peak: float,
    agg: tuple[float, int, float, str] | None,
    tail: str,
) -> CheckResult:
    """Fold per-argv c61 evaluations into the check verdict (SKIP/WARN/PASS)."""
    warns: list[str] = []
    argv_notes: list[str] = []
    mems: list[str] = []
    for i, argv in enumerate(argvs):
        outcome, text, warn = _c61_eval_argv(parser, reachable_fn, i, argv, max_peak, agg)
        if outcome == "skip":
            return _skip(cid, name, text)
        if outcome == "note":
            argv_notes.append(text)
            continue
        mems.append(text)
        if warn is not None:
            warns.append(warn)
    if not mems:
        joined = "; ".join(argv_notes) + tail
        if argv_notes and all("no SLURM lane reachable" in n for n in argv_notes):
            return _skip(
                cid,
                name,
                "no SLURM lane reachable for any launch argv — the rendered --mem never "
                "binds; " + joined,
            )
        return _skip(cid, name, joined or "no launch argv evaluated")
    if warns:
        extra = "; ".join(argv_notes)
        return _warn(cid, name, "; ".join(warns) + (f" [{extra}]" if extra else ""))
    return _pass(
        cid,
        name,
        f"declared peak RSS {max_peak:g} GiB"
        + (f" (within-job aggregate {agg[0]:g} GiB)" if agg is not None else "")
        + f" fits the would-render --mem across {len(mems)} launch argv(s): "
        + "; ".join(mems),
    )


# ─── Check 62 — §9 backend pin-claim vs body.md frontmatter (#2276) ─────────

# Incident #2225 (fu1 lineage): plan v5:274 claimed an "explicit frontmatter
# pin" (`backend: runpod`) and v9:236 claimed to inherit it ("parent pin
# inherited") while the task's body.md frontmatter carried NO `backend:`
# key. dispatch_issue.py reads the FRONTMATTER, so the plan's own dispatch
# command routes `auto`, and the free-SLURM fall-through rungs (no
# `/workspace`, #608) become reachable for exactly the sentinel-signaling
# workload whose §9 prose declared them unsafe. c43 (check_sentinel_lane)
# is text-hermetic — the prose CLAIM of a pin quiets its WARN — so the
# claim-vs-frontmatter reconciliation needs task context and runs OUTSIDE
# verify_plan_text() (the c23 pattern: kwargs signature, appended by
# main() in --issue mode, explicit SKIP row in --plan-file mode).
#
# Trigger grammar (deliberately tight — designed at FAIL polarity; shipped
# WARN-only per the calibration block below): non-fenced lines
# INSIDE the §9 window (_c57_section9_window) carrying BOTH a known-lane
# `backend: <lane>` token AND a same-line claim token
# (pin/pinned/inherit*/frontmatter). Window scoping is the FP guard —
# narrative mentions of ANOTHER task's pin ("#2225's `backend: runpod`
# pin") live outside §9. Residuals (disclosed, not covered): a pin claim
# OUTSIDE the §9 window escapes; a claim phrased without any claim token
# escapes; on amendment-COMPOSED text (_compose_amendment_text) the window
# resolves the FIRST §9 heading — the amendment's when both amendment and
# base carry one — so a base-version §9 pin claim can escape. Miss
# direction is SKIP (fail-safe).
#
# Calibration (#2276 §4 step 6, over 4,089 persisted plan versions
# tasks/*/*/plans/v*.md at origin/main 5d9eeb30ee, 2026-08-13; sweep tool:
# scripts/issue2276_c62c63_corpus_sweep.py — the c58 precedent: any future
# c62-grammar change re-runs the sweep and records the realized numbers
# here). Measured at the PLANNED FAIL polarity: 107 FAILs / 33 distinct
# tasks, 26 PASSes (tasks whose frontmatter pin is genuinely set — #1335
# fm=gcp, #1586 fm=runpod, ...), n_skip = 3,956 (all `no-claim`).
# Adjudicated FALSE-POSITIVE classes among the 107:
#   (a) `backend: auto` claims with the key absent (10 rows) — absent/
#       empty frontmatter IS the auto route (CLAUDE.md § Compute
#       backends), so the flagged state is the CORRECT configuration;
#       fixed in the verdict logic (an all-`auto` claim set with the key
#       absent now PASSes — a routing-semantics equivalence, not a
#       grammar widening);
#   (b) prospective / dispatch-flag pins — the DOMINANT class (~55
#       gcp-era rows + most runpod/fellows rows): §9 says "pinned
#       `backend: gcp`" while the pin traveled via the dispatch command's
#       `--backend` flag, or the key was set-then-removed after the task
#       completed; the frontmatter read at sweep time cannot distinguish
#       never-set from since-removed, and the flagged tasks launched
#       correctly — false ALARMS as FAIL evidence;
#   (c) compute-table lane mentions with an incidental same-row claim
#       token (#1689 v11 `backend: fellows` table cells).
# TRUE POSITIVES: #2225 v2-v9 (8 rows — the founding incident; the named
# expected-TPs v5-v9 recovered, sweep-validity criterion satisfied) plus
# the #2203 v11-v13 "runpod (pinned — /workspace sentinels" shape.
# PRE-REGISTERED POSTURE RULE FIRED: >2 adjudicated false-positive FAILs
# on the corpus ⇒ c62 SHIPS WARN-ONLY (the #2276 plan §4 step 6 downgrade
# rule; recorded here + in the #2276 round marker). The trigger grammar is
# NOT widened or narrowed. Post-downgrade re-sweep: 97 WARNs / 36 PASSes
# (the 26 matching pins + 10 auto-equivalence passes), n_skip unchanged.

#: Known dispatch lanes (compute-backends.md); an unknown token after
#: `backend:` (e.g. "backend: the same as the parent") is ignored — FP
#: guard for the deliberately tight trigger grammar.
_C62_KNOWN_LANES = frozenset({"runpod", "gcp", "fellows", "nibi", "fir", "mila", "auto"})

#: `backend: <lane>` capture (inline-code or bare; optional opening
#: backtick tolerated between the colon and the lane value).
_C62_LANE_RE = re.compile(r"backend:\s*`?([a-z][a-z0-9_-]*)")

#: Same-line claim vocabulary: the line must CLAIM a pin, not merely
#: mention a lane ("pin"/"pinned", "inherit"/"inherited"/"inherits",
#: "frontmatter").
_C62_CLAIM_TOKEN_RE = re.compile(r"(?i)\b(?:pin(?:ned)?|inherit(?:ed|s)?|frontmatter)\b")


def _c62_pin_claims(plan: str) -> list[tuple[str, str]]:
    """(claim-line, lane) pairs for §9 backend pin-claims.

    A hit is a non-fenced line inside the §9 window carrying BOTH a
    known-lane ``backend: <lane>`` token and a same-line claim token; one
    line can contribute several pairs (the pin+fallback-lane shape).
    Returns ``[]`` when no §9 window parses (miss direction: SKIP).
    """
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    window = _c57_section9_window(lines, mask)
    if window is None:
        return []
    lo, hi = window
    hits: list[tuple[str, str]] = []
    for i in range(lo, hi):
        if mask[i]:
            continue
        line = lines[i]
        if not _C62_CLAIM_TOKEN_RE.search(line):
            continue
        for m in _C62_LANE_RE.finditer(line):
            lane = m.group(1).lower()
            if lane in _C62_KNOWN_LANES:
                hits.append((line.strip(), lane))
    return hits


def check_backend_pin_claim(plan: str, *, frontmatter_backend: str | None) -> CheckResult:
    """WARN-only, conditional, all kinds, ``--issue`` mode only: a §9 line
    that CLAIMS a frontmatter backend pin (``backend: <lane>`` + a
    same-line pin/inherited/frontmatter claim token) must match the task's
    actual ``body.md`` frontmatter ``backend:`` key — a pin-claim with no
    frontmatter key routes ``auto`` at dispatch (#2225 v5/v9), and a
    mismatched lane routes the WRONG pin. WARN-only: the planned FAIL
    polarity was DOWNGRADED by the pre-registered #2276 §4 step 6
    calibration rule (>2 adjudicated false-positive FAILs on the corpus —
    see the calibration block above). An all-``auto`` claim set with the
    key absent PASSes (absent/empty frontmatter IS the auto route).
    ``frontmatter_backend`` is the str-coerced, stripped frontmatter value
    (``None`` when absent/empty); main() passes it in ``--issue`` mode and
    appends an explicit SKIP row in ``--plan-file`` mode (the c23 pattern
    — no task context there). No §9 window / no claim lines → SKIP
    (trigger-conditional). NEVER FAILs."""
    cid, name = "c62_backend_pin_claim", "§9 backend pin-claim matches body.md frontmatter"
    claims = _c62_pin_claims(plan)
    if not claims:
        return _skip(cid, name, "no §9 backend pin-claim in plan prose")
    if _standalone_na_declared(plan, r"backend pin-claim reconciled"):
        return _pass(cid, name, "explicit N/A declared (backend pin-claim reconciled)")
    fm_lane = frontmatter_backend.strip().lower() if frontmatter_backend else None
    if fm_lane is None:
        non_auto = [(line, lane) for line, lane in claims if lane != "auto"]
        if not non_auto:
            return _pass(
                cid,
                name,
                "claims `backend: auto` and the frontmatter key is absent — absent/empty "
                "routes `auto` (the documented default); no drift",
            )
        line, lane = non_auto[0]
        return _warn(
            cid,
            name,
            f"§9 claims a frontmatter backend pin (`backend: {lane}`: {line[:90]!r}) but the "
            "task's body.md frontmatter carries NO `backend:` key — the claim is phantom: "
            "dispatch_issue.py reads the frontmatter and routes `auto` at dispatch "
            f"(#2225 v5/v9). Add `backend: {lane}` to the task's body.md frontmatter BEFORE "
            "dispatch, or reword the §9 claim, or declare "
            "'N/A — backend pin-claim reconciled' on its own line, unwrapped "
            "(no backticks/quotes)",
        )
    mismatched = [(line, lane) for line, lane in claims if lane != fm_lane]
    if mismatched:
        line, lane = mismatched[0]
        return _warn(
            cid,
            name,
            f"§9 pin-claim names lane `{lane}` ({line[:90]!r}) but the body.md frontmatter "
            f"carries `backend: {fm_lane}` — reconcile the claim with the frontmatter "
            "(dispatch follows the frontmatter, not the prose), or declare "
            "'N/A — backend pin-claim reconciled' on its own line, unwrapped "
            "(no backticks/quotes)",
        )
    return _pass(
        cid, name, f"{len(claims)} §9 pin-claim(s) match the frontmatter `backend: {fm_lane}`"
    )


# ─── Check 63 — §9 declared GPU width vs launch-fence width (#2276) ─────────

# Incident #2225 v9: §9 declared "one 8xH100 pod" with every wall row
# costed 8-wide, while the adjacent dispatch fence
# (`dispatch_issue.py launch --intent lora-7b ... --time-budget-hours 12`)
# carried no width flag — the `lora-7b` intent default is 1xH100
# (scripts/gpu_heuristics.py::INTENTS), so a verbatim copy delivers 1/8
# the costed width and the 12 h fence (sized to the 8-wide ~5.4 h wall)
# TIMEOUTs the ~40 h narrow run. Not covered elsewhere: c46 dry-parses
# the fence (it PARSES fine), c50 compares walls to SLURM time bins, c26
# compares GPU FAMILY only (H100 == H100). WARN-only (the
# c46/c50/c52/c61 posture — a heuristic width join must not become the
# #1388 fleet-wedge shape); a deliberate narrow launch beside a wide
# provision is absorbed by the polarity + the escape literal. A sibling
# `pod.py provision --gpu-count N` fence does NOT suppress the WARN —
# the v9 incident text carried exactly that parenthetical and the copied
# artifact was still the narrow dispatch fence.
#
# Calibration (#2276 §4 step 6, AS-SHIPPED regexes, same 4,089-version
# corpus + sweep tool as the c62 block above). Measured: 27 WARNs /
# 6 distinct tasks, 53 PASSes; skips: no-multi-gpu-width 2,914,
# no-section-9 724, no-launch-argv 308, no-width-contribution 63.
# TRUE POSITIVES: #2225 v1-v9 (9 rows — the founding incident; the named
# expected-TP v9 recovered, sweep-validity criterion satisfied: 8xH100
# declared + costed 8-wide beside a width-less `--intent lora-7b` fence)
# and #2203 v1-v5 (the same drift shape: a 4xH100 arms-sharded phase
# table beside a width-less `--intent eval` fence). FALSE-POSITIVE
# classes adjudicated (absorbed by the WARN-only polarity, the
# c46/c50/c58 posture): (a) width delivered via a DIFFERENT channel than
# the flagged fence — `pod.py provision --gpu-count N` / a resumed
# suffixed pod is the real width-bearing launch and the dispatch fence
# is narrow by design (#2254 v1-v5: `--gpu-count 4` per pod with an
# optional "8xH100 halves pod-A wall" upgrade mention; #813 v1-v2:
# "ONE 8x H100 pod (resume pod-667, else fresh ... --gpu-count 8)");
# (b) ratio/multiplier prose false-parsed as a width token — "6x
# A100-vs-H100 per-step bound" (#610 v1; a disclosed grammar residual —
# the digitxFAMILY form is inherently ambiguous there); (c) parent-
# recipe width cited beside a DELIBERATELY narrow rerun — "downsized to
# 1xH100 vs the parent's 4xH100" (#2203 v7-v9, whose fences carry an
# explicit `--gpus 1`), "parent: ... on 4x A100" as a per-cell cost
# basis for a 1x A100 plan (#614 v1-v2). WARN polarity is the
# pre-registered posture; no downgrade rule applies to c63.

# STATIC MIRROR of scripts/gpu_heuristics.py::INTENTS[*].gpu_count,
# drift-guarded by tests/test_verify_plan.py::
# test_c63_intent_width_mirror_matches_gpu_heuristics — verify_plan_text()
# stays hermetic (the c26 convention: no project imports at module level).
_C63_INTENT_GPU_COUNT: dict[str, int] = {
    "eval": 1,
    "lora-7b": 1,
    "ft-7b": 4,
    "inf-70b": 8,
    "ft-70b": 8,
    "sweep-8g-a100": 8,
    "sweep-8g-h100": 8,
    "debug": 1,
}

#: `<N>x<GPU family>` width tokens on §9 prose lines (`8xH100`,
#: `4 x A100-80`, `2*L4`). Digit prefix + GPU-family suffix disambiguate
#: the bare-`x` form; families uppercase per corpus convention.
_C63_WIDTH_RE = re.compile(r"(\d+)\s*[×x*]\s*(?:H100|H200|A100(?:-\d+)?|B200|L4)\b")  # noqa: RUF001

#: `--gpu-count N` on §9 prose lines (the `pod.py provision` width flag).
_C63_GPU_COUNT_RE = re.compile(r"--gpu-count\s+(\d+)")


def _c63_declared_width(lines: list[str], mask: list[bool], window: tuple[int, int]) -> int | None:
    """Max GPU width declared on non-fenced §9-window lines, or ``None``
    when no width token appears in the window."""
    lo, hi = window
    widths: list[int] = []
    for i in range(lo, hi):
        if mask[i]:
            continue
        widths += [int(m.group(1)) for m in _C63_WIDTH_RE.finditer(lines[i])]
        widths += [int(m.group(1)) for m in _C63_GPU_COUNT_RE.finditer(lines[i])]
    return max(widths) if widths else None


def check_declared_width_vs_launch(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds, both modes: when the §9 window
    declares an N-GPU spec (``8xH100`` / ``--gpu-count 8`` tokens, N_decl =
    max, trigger N_decl >= 2), at least one plan-embedded launch-shaped
    ``dispatch_issue.py`` argv (the c46/c50 chain) must REALIZE a width
    >= N_decl — ``--gpus`` when set, else the intent's default width from
    the ``_C63_INTENT_GPU_COUNT`` static mirror of
    ``scripts/gpu_heuristics.py::INTENTS``. A narrower realized width
    WARNs: §9 walls are costed N_decl-wide, so a ``--time-budget-hours``
    fence sized to the wide wall TIMEOUTs the narrow run (#2225 v9).
    Every ambiguity SKIPs with a stated reason (no §9 window; N_decl < 2;
    no launch argv; CLI unavailable; every parsed argv contributes no
    width). NEVER FAILs (the c46/c50/c61 posture)."""
    del kind  # all kinds: a width-starved launch fence drifts identically everywhere
    cid, name = "c63_declared_width_vs_launch", "§9 declared GPU width vs launch-fence width"
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    window = _c57_section9_window(lines, mask)
    if window is None:
        return _skip(cid, name, "no parseable §9 heading — no declared-width window")
    n_decl = _c63_declared_width(lines, mask, window)
    if n_decl is None or n_decl < 2:
        return _skip(cid, name, "no multi-GPU width (N >= 2) declared on §9 prose lines")
    if _standalone_na_declared(plan, r"declared width vs launch width reconciled"):
        return _pass(cid, name, "explicit N/A declared (declared width vs launch width)")
    argvs, notes = _c50_launch_argvs(plan)
    if not argvs:
        return _skip(
            cid,
            name,
            f"§9 declares {n_decl}-wide but the plan embeds no launch-shaped "
            "dispatch_issue.py argv — width flows through a different channel "
            "(pod.py provision / SSH-MCP), which this check does not read"
            + (f" [{'; '.join(notes)}]" if notes else ""),
        )
    parser, load_detail = _c46_argparser()
    if parser is None:
        return _skip(
            cid,
            name,
            f"dispatch_issue.build_argparser unavailable ({load_detail}) — "
            "launch-fence width not evaluated",
        )
    realized: list[tuple[str, int, str]] = []  # (fence-snippet, width, source)
    contributes_nothing: list[str] = []
    for argv in argvs:
        fence = " ".join(argv)[:70]
        ns, err = _c46_dry_parse(parser, argv)
        if ns is None:
            contributes_nothing.append(f"{fence!r} does not parse ({err}) — c46 arm 1 owns it")
            continue
        gpus = getattr(ns, "gpus", None)
        intent = getattr(ns, "intent", None)
        if gpus is not None:
            realized.append((fence, int(gpus), f"explicit `--gpus {gpus}`"))
        elif intent in _C63_INTENT_GPU_COUNT:
            realized.append(
                (
                    fence,
                    _C63_INTENT_GPU_COUNT[intent],
                    f"`--gpus` absent, so the intent default binds: `{intent}` -> "
                    f"{_C63_INTENT_GPU_COUNT[intent]} GPU(s)",
                )
            )
        else:
            contributes_nothing.append(
                f"{fence!r} carries no `--gpus` and intent {intent!r} is not in the "
                "width mirror — contributes no width"
            )
    if not realized:
        return _skip(
            cid,
            name,
            f"§9 declares {n_decl}-wide but no parsed launch argv contributes a width: "
            + "; ".join(contributes_nothing),
        )
    covering = [t for t in realized if t[1] >= n_decl]
    if covering:
        fence, width, source = covering[0]
        return _pass(
            cid,
            name,
            f"§9 declares {n_decl}-wide and a launch argv realizes {width} ({source}): {fence!r}",
        )
    fence, width, source = min(realized, key=lambda t: t[1])
    extra = f" [{'; '.join(contributes_nothing)}]" if contributes_nothing else ""
    return _warn(
        cid,
        name,
        f"§9 declares a {n_decl}-GPU spec but the launch fence {fence!r} realizes only "
        f"{width} ({source}) — §9 walls are costed {n_decl}-wide, so a "
        "`--time-budget-hours` fence sized to the wide wall TIMEOUTs the narrow run "
        f"(#2225 v9). Add `--gpus {n_decl}` to the fence, or re-cost the walls at the "
        "realized width, or declare 'N/A — declared width vs launch width reconciled' "
        "on its own line, unwrapped (no backticks/quotes)" + extra,
    )


# ─── Checks 65/66 — smoke-fixture size claim vs realized fixtures (#2178) ─

_C65_REPO_ROOT = Path(__file__).resolve().parent.parent  # tests monkeypatch (c34/c41/c42 pattern)

# Row-count read cap (bytes). A file larger than this yields a LOWER BOUND:
# a truncated count can SATISFY a floor (bound >= F) but never FAIL one.
_C65_READ_CAP = 8 * 1024 * 1024

# Countable (line-oriented) extensions: newline counting is a valid row
# count ONLY for line-per-row formats. A resolved ``.json`` / ``.parquet``
# file is NON-COUNTABLE — a minified 40-object JSON array reads as 1 "row",
# a FALSE FAIL, the one direction this check must never err — so it
# contributes no count (neither satisfies nor contradicts a floor); when
# ALL resolved files are non-countable the rung stays unresolved (the
# ladder continues / the check SKIPs). ``.json``/``.parquet`` stay in the
# ``_C65_GLOB_RE`` grammar BY CHOICE: the path token still resolves (check
# 66's distinctive-component producer scan keeps its literal token) — only
# COUNTING is restricted (#2178 round 2, concern `non-row-format-counting`).
_C65_COUNTABLE_EXTS = (".jsonl", ".csv", ".tsv", ".txt")

# Hard bound on counted fixture files per resolution rung (cost + detail size).
_C65_MAX_FILES = 32

# Bounded scripts/*.py scan (constant lookup + producer-candidate scan):
# non-recursive, at most this many files, at most this many bytes per file.
_C65_MAX_SCRIPTS = 2500
_C65_SCRIPT_READ_CAP = 1024 * 1024

# Claim grammar — three conjuncts on ONE fence-masked line (conjunct 3
# captures the claimed floor F, capped at 6 digits).
_C65_SMOKE_RE = re.compile(r"(?i)\bsmoke\b")
_C65_FIXTURE_NOUN_RE = re.compile(r"(?i)\b(?:fixture|slice|corpus|corpora|sample)s?\b")
_C65_FLOOR_RE = re.compile(
    r"(?:≥|>=|\bat least\b)\s*(\d{1,6})\s+rows?\b"  # the unicode >= sign is real plan text
    r"|\b(?:fixture|slice|sample)s?\s+of\s+(\d{1,6})\s+rows?\b"
)

# Constant-form claim (RAW scan, fences included — constants are usually
# quoted in fenced code): a SMOKE_SAMPLE_N-shaped ``<NAME> = <int>`` paste.
_C65_CONST_RE = re.compile(r"\b(SMOKE_[A-Z0-9_]*(?:_N|_ROWS|_SIZE))\s*=\s*(\d{1,6})\b")

# Smoke-fixture path token: a repo-relative path with a data extension whose
# dir or file component contains "smoke". The leading LOOKBEHIND (not \b)
# keeps a dotted repo-relative token (".claude/worktrees/...") matching
# WHOLE or not at all — a \b would silently truncate it to "claude/..."
# (round-1 Methodology concern 2: never a silently-truncated token).
# ``.json``/``.parquet`` are grammar-resolvable but NON-COUNTABLE — see
# ``_C65_COUNTABLE_EXTS``.
_C65_GLOB_RE = re.compile(
    r"(?<![\w./*-])"
    r"((?:[\w.*-]+/)*[\w.*-]*smoke[\w.*-]*/[\w./*-]+\.(?:jsonl|json|csv|tsv|txt|parquet)"
    r"|(?:[\w.*-]+/)+[\w.*-]*smoke[\w.*-]*\.(?:jsonl|json|csv|tsv|txt|parquet))\b"
)

# Pinned-tip citation: "pinned tip `8c7b7b2406`" / "tip 8c7b7b2406...".
_C65_TIP_RE = re.compile(r"(?i)\btip\s+`?([0-9a-f]{8,40})\b")


def _c65_git(cmd: list[str]) -> bytes | None:
    """Run a read-only git command under the check-42 subprocess contract:
    ``timeout=10``, ``check=False``, ``cwd=_C65_REPO_ROOT``, retry ONCE on a
    transient ``OSError`` after 0.1 s. Returns stdout bytes on rc == 0, else
    ``None`` — any git unavailability (or a non-resolving sha/prefix) leaves
    THIS RUNG unresolved (the ladder continues / the check SKIPs; never a
    FAIL)."""
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
                cwd=str(_C65_REPO_ROOT),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None  # a hung git command is not retriable
        except OSError:
            if attempt == 1:
                time.sleep(0.1)
                continue
            return None
        return r.stdout if r.returncode == 0 else None
    return None  # unreachable (defensive; the loop always returns)


def _c65_tip_sha(plan: str) -> str | None:
    """First pinned-tip SHA the plan cites (RAW scan), or None."""
    m = _C65_TIP_RE.search(plan)
    return m.group(1).lower() if m else None


def _c65_count_bytes(blob: bytes, truncated: bool) -> tuple[int, bool]:
    """(rows, exact): newline count over ``blob`` with ``wc -l`` parity plus
    a final unterminated line counted; ``truncated`` blobs are LOWER BOUNDS
    (exact=False)."""
    n = blob.count(b"\n")
    if truncated:
        return n, False
    if blob and not blob.endswith(b"\n"):
        n += 1
    return n, True


def _c65_count_rows(path: Path) -> tuple[int, bool] | None:
    """(rows, exact) for a working-tree file, reading at most
    ``_C65_READ_CAP`` bytes; None when unreadable (fail-open)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_C65_READ_CAP)
            truncated = bool(fh.read(1))
    except OSError:
        return None
    return _c65_count_bytes(head, truncated)


def _c65_glob_counts(base: Path, glob_tokens: list[str]) -> list[tuple[str, int, bool]]:
    """Row counts for ``glob_tokens`` globbed under ``base`` (files only,
    LINE-ORIENTED extensions only — see ``_C65_COUNTABLE_EXTS``; at most
    ``_C65_MAX_FILES``; unreadable files and malformed globs are skipped —
    fail-open, the rung just resolves less)."""
    counts: list[tuple[str, int, bool]] = []
    for tok in glob_tokens:
        try:
            hits = sorted(base.glob(tok))
        except (OSError, ValueError):
            continue
        for p in hits:
            if len(counts) >= _C65_MAX_FILES:
                return counts
            if not p.is_file():
                continue
            if p.suffix not in _C65_COUNTABLE_EXTS:
                continue  # non-line-oriented — non-countable (_C65_COUNTABLE_EXTS)
            counted = _c65_count_rows(p)
            if counted is None:
                continue
            try:
                rel = str(p.relative_to(_C65_REPO_ROOT))
            except ValueError:
                rel = str(p)
            counts.append((rel, *counted))
    return counts


def _c65_git_tree_counts(sha: str, glob_token: str) -> list[tuple[str, int, bool]] | None:
    """Row counts for ``glob_token`` inside the COMMITTED tree at ``sha``
    (resolution rung 2). Cost: ONE ``git ls-tree -r --name-only`` scoped to
    the token's literal dir prefix PLUS up to ``_C65_MAX_FILES`` (32)
    ``git cat-file blob`` reads — 1 + N subprocess calls per glob token,
    NOT a flat 2 (round-1 Methodology concern 4). Returns None when git is
    unavailable / the sha does not resolve (rung unresolved, fail-open via
    ``_c65_git``). Tests monkeypatch this helper to exercise the rung
    without a git repo; the real subprocess path is pinned by the
    throwaway-git-repo test."""
    parts = glob_token.split("/")
    prefix_parts: list[str] = []
    for part in parts[:-1]:
        if any(ch in part for ch in "*?["):
            break
        prefix_parts.append(part)
    prefix = "/".join(prefix_parts)
    out = _c65_git(["git", "ls-tree", "-r", "--name-only", sha, "--", prefix or "."])
    if out is None:
        return None
    names = [ln for ln in out.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    matches = sorted(
        n
        for n in names
        # Line-oriented only — a committed .json/.parquet blob is just as
        # non-countable as a working-tree one (_C65_COUNTABLE_EXTS).
        if PurePosixPath(n).match(glob_token) and n.endswith(_C65_COUNTABLE_EXTS)
    )[:_C65_MAX_FILES]
    counts: list[tuple[str, int, bool]] = []
    for fname in matches:
        blob = _c65_git(["git", "cat-file", "blob", f"{sha}:{fname}"])
        if blob is None:
            return None
        counts.append((fname, *_c65_count_bytes(blob[:_C65_READ_CAP], len(blob) > _C65_READ_CAP)))
    return counts


def _c65_worktree_counts(
    plan: str, glob_tokens: list[str]
) -> tuple[str | None, list[tuple[str, int, bool]]]:
    """Resolution rung 3: join each ``.claude/worktrees/issue-<N>*`` dir
    (N = the plan's own H1 task number; at most 8 dirs) with the glob
    tokens — the rung that reaches gitignored worktree-local fixtures (the
    #1336 v16 class). Plan-text-derived, so it works identically in
    ``--plan-file`` mode."""
    m = re.search(r"(?im)^#.*?task\s+#(\d+)", plan)
    if not m:
        return None, []
    try:
        wt_dirs = sorted(_C65_REPO_ROOT.glob(f".claude/worktrees/issue-{m.group(1)}*"))[:8]
    except OSError:
        return None, []
    for wt in wt_dirs:
        if not wt.is_dir():
            continue
        counts = _c65_glob_counts(wt, glob_tokens)
        if counts:
            return f"issue worktree {wt.name}", counts
    return None, []


def _c65_distinctive_token(glob_token: str) -> str | None:
    """The glob's most distinctive literal component containing "smoke":
    the last such directory component, else the filename stem; None when
    the component is glob-bearing or shorter than 6 chars (too generic for
    a literal-substring producer scan)."""
    parts = glob_token.split("/")
    for part in reversed(parts[:-1]):
        if "smoke" in part:
            usable = len(part) >= 6 and not any(ch in part for ch in "*?[")
            return part if usable else None
    stem = parts[-1].rsplit(".", 1)[0]
    if "smoke" in stem:
        usable = len(stem) >= 6 and not any(ch in stem for ch in "*?[")
        return stem if usable else None
    return None


def _c65_scripts_scan(
    const_names: list[str], literal_tokens: list[str]
) -> tuple[list[tuple[str, str, int]], list[str]]:
    """ONE bounded pass over ``_C65_REPO_ROOT/scripts/*.py`` (non-recursive,
    <= ``_C65_MAX_SCRIPTS`` files, <= ``_C65_SCRIPT_READ_CAP`` bytes each;
    read-only, no import): line-anchored ``<NAME> = <int>`` definitions for
    ``const_names``, plus files carrying any ``literal_tokens`` substring
    (producer candidates). Runs only when a claim armed the checks."""
    consts: list[tuple[str, str, int]] = []
    candidates: list[str] = []
    if not const_names and not literal_tokens:
        return consts, candidates
    const_res = {
        cname: re.compile(rf"(?m)^\s*{re.escape(cname)}\s*=\s*(\d{{1,6}})\b")
        for cname in const_names
    }
    try:
        files = sorted((_C65_REPO_ROOT / "scripts").glob("*.py"))[:_C65_MAX_SCRIPTS]
    except OSError:
        return consts, candidates
    for f in files:
        try:
            with f.open("r", errors="replace") as fh:
                text = fh.read(_C65_SCRIPT_READ_CAP)
        except OSError:
            continue
        rel = f"scripts/{f.name}"
        for cname, pat in const_res.items():
            m = pat.search(text)
            if m:
                consts.append((cname, rel, int(m.group(1))))
        if any(tok in text for tok in literal_tokens):
            candidates.append(rel)
    return consts, candidates


class _C65Eval(NamedTuple):
    """Shared evaluation for checks 65/66. No cross-call memoization BY
    DESIGN: the ladder only runs when a claim fired, the cost is bounded
    (~ms non-firing), and a memo keyed on plan text would poison across
    tests that monkeypatch ``_C65_REPO_ROOT`` between calls with identical
    plan strings."""

    prose_claims: list[tuple[str, int]]  # (claim-line excerpt, claimed floor F)
    const_claims: list[tuple[str, int]]  # (constant name, claimed value)
    glob_tokens: list[str]
    winning_rung: str | None  # first-resolve-wins: working tree > pinned tip > worktrees
    file_counts: list[tuple[str, int, bool]]  # (path, rows, exact)
    const_evidence: list[tuple[str, str, int]]  # (name, scripts/ relpath, actual value)
    producer_candidates: list[str]  # scripts/ relpaths that plausibly produce the fixtures


def _c65_glob_tokens(plan: str) -> list[str]:
    """Deduplicated smoke-fixture glob tokens (RAW scan, fences included —
    the c41/c11 raw-scan doctrine); repo-relative, side-effect-free tokens
    only (absolute paths and ``..`` components refused)."""
    glob_tokens: list[str] = []
    for m in _C65_GLOB_RE.finditer(plan):
        tok = m.group(1)
        if tok.startswith("/") or ".." in tok.split("/") or tok in glob_tokens:
            continue
        glob_tokens.append(tok)
    return glob_tokens


def _c65_resolve_files(
    plan: str, glob_tokens: list[str]
) -> tuple[str | None, list[tuple[str, int, bool]]]:
    """Resolve the fixture evidence through the three-rung ladder,
    FIRST-RESOLVE-WINS: (1) repo-root working tree — it deliberately
    SHADOWS — (2) the plan's cited pinned tip, then (3) the issue
    worktrees."""
    if not glob_tokens:
        return None, []
    file_counts = _c65_glob_counts(_C65_REPO_ROOT, glob_tokens)
    if file_counts:
        return "working tree", file_counts
    sha = _c65_tip_sha(plan)
    if sha:
        for tok in glob_tokens:
            counts = _c65_git_tree_counts(sha, tok)
            if counts:
                file_counts.extend(counts[: _C65_MAX_FILES - len(file_counts)])
        if file_counts:
            return f"pinned tip {sha[:12]}", file_counts
    return _c65_worktree_counts(plan, glob_tokens)


def _c65_evaluate(plan: str) -> _C65Eval:
    """Extract the smoke-fixture size claims, resolve the fixture evidence
    through the three-rung ladder (``_c65_resolve_files``), and run the
    bounded ``scripts/*.py`` scan for constant evidence + producer
    candidates."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    prose_claims: list[tuple[str, int]] = []
    for line, fenced in zip(lines, mask, strict=True):
        if fenced or not _C65_SMOKE_RE.search(line) or not _C65_FIXTURE_NOUN_RE.search(line):
            continue
        for m in _C65_FLOOR_RE.finditer(line):
            prose_claims.append((line.strip()[:160], int(m.group(1) or m.group(2))))
    const_claims = [(m.group(1), int(m.group(2))) for m in _C65_CONST_RE.finditer(plan)]
    if not prose_claims and not const_claims:
        return _C65Eval([], [], [], None, [], [], [])
    glob_tokens = _c65_glob_tokens(plan)
    winning_rung, file_counts = _c65_resolve_files(plan, glob_tokens)
    const_names = sorted({cname for cname, _v in const_claims})
    tokens = sorted({t for t in (_c65_distinctive_token(g) for g in glob_tokens) if t})
    const_evidence, producer_candidates = _c65_scripts_scan(const_names, tokens)
    for _cname, rel, _v in const_evidence:
        if rel not in producer_candidates:
            producer_candidates.append(rel)
    producer_candidates.sort()
    return _C65Eval(
        prose_claims,
        const_claims,
        glob_tokens,
        winning_rung,
        file_counts,
        const_evidence,
        producer_candidates,
    )


class _C65Cmp(NamedTuple):
    fails: list[str]
    warns: list[str]
    passes: list[str]
    unresolved: list[str]


def _c65_file_counts_str(ev: _C65Eval) -> str:
    counts_str = ", ".join(
        f"{p}={n}" + ("" if exact else " (lower bound)") for p, n, exact in ev.file_counts[:8]
    )
    if len(ev.file_counts) > 8:
        counts_str += f", … ({len(ev.file_counts)} files total)"
    return counts_str


def _c65_compare_prose(ev: _C65Eval, cmp_: _C65Cmp) -> None:
    """Prose floors vs realized evidence. FAIL-grade only on a
    FILE-CONFIRMED contradiction from EXACT counts (a capped lower-bound
    count can satisfy a floor but never fail one); with no files resolved,
    a repo constant is best-effort WARN-grade evidence."""
    exact_counts = [(p, n) for p, n, exact in ev.file_counts if exact]
    const_max = max((v for _n, _r, v in ev.const_evidence), default=None)
    counts_str = _c65_file_counts_str(ev)
    for excerpt, floor in ev.prose_claims:
        usable = bool(exact_counts) or any(
            n >= floor for _p, n, exact in ev.file_counts if not exact
        )
        if usable:
            exact_min = min((n for _p, n in exact_counts), default=None)
            if exact_min is not None and floor > exact_min:
                cmp_.fails.append(
                    f"claimed floor {floor} rows > realized min {exact_min} "
                    f"(rung: {ev.winning_rung}; files: {counts_str}) — claim line: {excerpt!r}"
                )
            else:
                cmp_.passes.append(
                    f"claimed floor {floor} rows satisfied at rung {ev.winning_rung!r} "
                    f"({len(ev.file_counts)} file(s): {counts_str})"
                )
        elif const_max is not None:
            src = ", ".join(f"{n2} = {v} ({r})" for n2, r, v in ev.const_evidence[:4])
            if floor > const_max:
                cmp_.warns.append(
                    f"claimed floor {floor} rows > repo constant {src} — constant-only "
                    "evidence (no fixture files resolved); WARN-grade by design"
                )
            else:
                cmp_.passes.append(
                    f"claimed floor {floor} rows consistent with repo constant {src} "
                    "(constant-only evidence — no fixture files resolved)"
                )
        else:
            cmp_.unresolved.append(f"prose floor {floor} rows ({excerpt!r})")


def _c65_compare_consts(ev: _C65Eval, cmp_: _C65Cmp) -> None:
    """Constant-form claims vs the repo definition of the SAME constant.
    WARN-grade BY DESIGN, never FAIL, and never compared against fixture
    files: a constant is one indirection removed from the fixtures (the
    producing script may clamp / branch on it) — deliberate, not drift
    (round-1 Methodology concern 6)."""
    by_name: dict[str, int] = {}
    where_by_name: dict[str, str] = {}
    for n2, rel, v in ev.const_evidence:
        if n2 not in by_name or v > by_name[n2]:
            by_name[n2] = v
            where_by_name[n2] = rel
    for cname, claimed in ev.const_claims:
        if cname in by_name:
            actual, where = by_name[cname], where_by_name[cname]
            if claimed > actual:
                cmp_.warns.append(
                    f"constant-form claim {cname} = {claimed} > repo definition "
                    f"{cname} = {actual} ({where}) — constant-route contradiction, "
                    "WARN-grade by design (never compared against fixture files)"
                )
            else:
                cmp_.passes.append(
                    f"constant-form claim {cname} = {claimed} consistent with the repo "
                    f"definition {cname} = {actual} ({where}) — constant-only evidence"
                )
        else:
            cmp_.unresolved.append(
                f"constant-form claim {cname} = {claimed} (constant not defined in scripts/*.py)"
            )


def _c65_compare(ev: _C65Eval) -> _C65Cmp:
    """Compare every claimed floor against the realized evidence; multiple
    claims are each evaluated and the caller takes the worst grade."""
    cmp_ = _C65Cmp([], [], [], [])
    _c65_compare_prose(ev, cmp_)
    _c65_compare_consts(ev, cmp_)
    return cmp_


def _c65_no_smoke_declared(plan: str) -> str | None:
    """The plan's standalone no-smoke-run declaration, if any — the
    criterion-5 clause-3 route (task #2178 round 2): check 11's canonical
    ``N/A — no dry-run smoke`` form (same tail regex as check 11) or the
    plain ``N/A — no smoke run`` variant. Returns the matched phrase (for
    the SKIP detail) or None. Checks 65/66 consult this BEFORE the claim
    grammar: a plan carrying BOTH a declaration and a claim-shaped line
    still SKIPs — the declaration wins, because a plan that declares it
    runs no smoke has no smoke fixtures in scope for a size claim to gate
    (the stray sentence is the critics' to adjudicate, not this check's).
    Wrapped pastes are rejected by ``_standalone_na_declared`` as usual."""
    if _standalone_na_declared(plan, r"no dry-?run smoke"):
        return "N/A — no dry-run smoke"
    if _standalone_na_declared(plan, r"no smoke run\b"):
        return "N/A — no smoke run"
    return None


def check_smoke_fixture_size(plan: str, kind: str) -> CheckResult:
    """A plan-claimed SMOKE-FIXTURE ROW FLOOR must not overstate the
    realized fixtures (task #2178; incident #1336 v16: the plan claimed a
    40-row-per-corpus smoke floor while the realized fixtures held 8 rows
    in six of seven files — the newly binding smoke gate was unsatisfiable
    at the realized size and no gate could tell).

    No-smoke-run declaration route (criterion 5 clause 3, round 2): a
    standalone ``N/A — no dry-run smoke`` (check 11's canonical form) or
    ``N/A — no smoke run`` declaration SKIPs this check BEFORE the claim
    grammar is consulted — a plan carrying BOTH a declaration and a
    claim-shaped line still SKIPs (the declaration wins: a declared
    no-smoke plan has no smoke fixtures in scope; see
    ``_c65_no_smoke_declared``).

    Trigger (conservative): a fence-masked line carrying all THREE
    conjuncts — "smoke", a fixture noun (fixture/slice/corpus/corpora/
    sample), and a floor-shaped row count ("sized >= 40 rows", "at least
    40 rows", "slice of 40 rows") — OR a RAW-scanned constant-form claim
    (a ``SMOKE_SAMPLE_N``-shaped assignment paste, fences included).
    Multiple claim lines are each evaluated; the worst grade wins.

    Row counting is LINE-ORIENTED-ONLY (``_C65_COUNTABLE_EXTS``: .jsonl /
    .csv / .tsv / .txt): a resolved ``.json`` / ``.parquet`` file is
    NON-COUNTABLE — its newline count under-reads (a minified 40-object
    array is one physical line), which would FAIL a satisfied floor, the
    one direction this check must never err — so it neither satisfies nor
    contradicts. The token stays in the glob grammar BY CHOICE (check 66's
    producer scan keeps its literal component); an all-non-countable
    resolution leaves the rung unresolved (ladder continues / SKIP).

    Resolution ladder, FIRST-RESOLVE-WINS (the working tree deliberately
    SHADOWS the pinned tip — a stale working tree is diagnosable from the
    FAIL detail, which names the winning rung + per-file counts): (1)
    repo-root working-tree glob; (2) the committed tree at the plan's
    cited pinned tip — ONE ``git ls-tree`` plus up to 32 ``git cat-file``
    reads per glob token (1 + N subprocess calls, not a flat 2), only when
    a tip is cited and rung 1 resolved nothing; (3)
    ``.claude/worktrees/issue-<N>*`` joined with the glob (N = the plan's
    own H1 task number) for gitignored worktree-local fixtures. Glob
    tokens are repo-relative only (absolute / ``..`` tokens refused);
    every rung is read-only and fail-open (git unavailability leaves a
    rung unresolved, never a FAIL).

    Verdicts: FAIL only on a FILE-CONFIRMED contradiction (a prose floor
    above the realized exact minimum); constant-route contradictions — a
    prose floor vs a repo constant, or a constant-form claim vs the repo
    definition of the same constant — are WARN-grade BY DESIGN (a constant
    is one indirection removed from the fixtures; deliberate, not drift);
    SKIP everywhere unresolvable (no claim, no path token, nothing
    resolves at any rung — the fail-safe). Escape:
    ``N/A — no smoke fixture size claim`` standalone, unwrapped."""
    cid, name = "c65_smoke_fixture_size", "smoke-fixture size claim vs realized fixtures"
    del kind  # all kinds — smoke fixtures appear in experiment and infra plans alike
    declared = _c65_no_smoke_declared(plan)
    if declared:
        return _skip(
            cid,
            name,
            f"plan declares no smoke run (standalone '{declared}') — smoke-fixture "
            "size claims not in scope; the declaration wins over any claim-shaped line",
        )
    ev = _c65_evaluate(plan)
    if not ev.prose_claims and not ev.const_claims:
        return _skip(cid, name, "no smoke fixture size claim detected")
    if _standalone_na_declared(plan, r"no smoke fixture size claim\b"):
        return _pass(cid, name, "explicit N/A declared (no smoke fixture size claim)")
    cmp_ = _c65_compare(ev)
    tail = f" [unresolved: {'; '.join(cmp_.unresolved)}]" if cmp_.unresolved else ""
    if cmp_.fails:
        return _fail(
            cid,
            name,
            "; ".join(cmp_.fails)
            + " — remedies: fix the claim to the realized fixture size, budget the "
            "fixture-producing script change (and regenerate the fixtures), or declare "
            "'N/A — no smoke fixture size claim' on its own line, unwrapped (no "
            "backticks/quotes)" + tail,
        )
    if cmp_.warns:
        return _warn(cid, name, "; ".join(cmp_.warns) + tail)
    if cmp_.passes:
        return _pass(cid, name, "; ".join(cmp_.passes) + tail)
    # Nothing comparable resolved anywhere — the criterion-5 fail-safe.
    bits: list[str] = []
    countable_tokens = [t for t in ev.glob_tokens if t.endswith(_C65_COUNTABLE_EXTS)]
    if ev.prose_claims and not ev.glob_tokens:
        bits.append("fixture path not named — claim not mechanically checkable")
    elif ev.glob_tokens and not countable_tokens:
        bits.append(
            "fixture path token(s) name only non-line-oriented formats (.json/.parquet) — "
            "newline row-counting is invalid there, so the evidence is non-countable "
            "(neither satisfies nor contradicts a floor)"
        )
    elif ev.glob_tokens and not ev.file_counts:
        msg = "fixture path unresolvable from working tree / pinned tip / issue worktrees"
        if len(countable_tokens) < len(ev.glob_tokens):
            msg += " (non-line-oriented .json/.parquet tokens are non-countable by design)"
        bits.append(msg)
    elif ev.file_counts and not ev.prose_claims:
        bits.append(
            f"fixture files resolve (rung: {ev.winning_rung}) but every claim is "
            "constant-form — the constant route reads the repo constant, never the "
            "files (WARN-grade-by-design route)"
        )
    if cmp_.unresolved:
        bits.append("; ".join(cmp_.unresolved))
    return _skip(cid, name, "; ".join(bits) or "claim detected but no realized evidence resolves")


def check_smoke_producer_coverage(plan: str, kind: str) -> CheckResult:
    """When a smoke-fixture size claim is CONTRADICTED by the realized
    evidence (check 65's comparison), the plan must NAME the fixture-
    producing script somewhere in its text — otherwise it asserts a state
    it has not budgeted to create (task #2178 Arm B; #1336 v16 named
    ``issue1336_stage_corpora`` zero times while its §4 said "Nothing else
    changes"). WARN-only: modified-file lists are prose and producer
    identification is heuristic.

    Producer candidates (both routes bounded to ``scripts/*.py``,
    read-only): (1) files defining a constant that a constant-form claim
    names; (2) files carrying the glob's most distinctive smoke component
    (last dir component, else filename stem; >= 6 chars) as a literal
    substring. Coverage predicate: the candidate's basename or
    repo-relative path appears ANYWHERE in the plan text (RAW scan) — the
    maximally conservative WARN-minimizing reading; a candidate in the
    plan's modified-file list is a strict subset. Shares check 65's
    no-smoke-run declaration route (``_c65_no_smoke_declared``), consulted
    BEFORE the claim grammar — a declaration beside a claim-shaped line
    still SKIPs (the declaration wins). SKIPs also when check 65's
    claim trigger is absent, when the realized evidence is unresolved,
    when the claim is satisfied, or when no candidate is identifiable.
    Escape: ``N/A — no fixture-producing script change needed``
    standalone, unwrapped."""
    cid, name = "c66_smoke_producer_coverage", "smoke-fixture producing script named in plan"
    del kind  # all kinds (same scope as check 65)
    declared = _c65_no_smoke_declared(plan)
    if declared:
        return _skip(
            cid,
            name,
            f"plan declares no smoke run (standalone '{declared}') — producer coverage "
            "not in scope; the declaration wins over any claim-shaped line",
        )
    ev = _c65_evaluate(plan)
    if not ev.prose_claims and not ev.const_claims:
        return _skip(cid, name, "no smoke fixture size claim detected (check-65 trigger absent)")
    if _standalone_na_declared(plan, r"no fixture-producing script change needed\b"):
        return _pass(cid, name, "explicit N/A declared (no fixture-producing script change needed)")
    cmp_ = _c65_compare(ev)
    if not (cmp_.fails or cmp_.warns or cmp_.passes):
        return _skip(
            cid,
            name,
            "realized evidence unresolved — cannot adjudicate whether a "
            "producing-script change is needed",
        )
    if not (cmp_.fails or cmp_.warns):
        return _skip(
            cid, name, "claim satisfied by realized fixtures — no producing-script change required"
        )
    if not ev.producer_candidates:
        return _skip(cid, name, "producing script not identifiable in scripts/")
    named = [c for c in ev.producer_candidates if c in plan or c.rsplit("/", 1)[-1] in plan]
    if named:
        return _pass(
            cid,
            name,
            "claim contradicted by the realized evidence but the producing script is "
            f"named in the plan: {', '.join(named[:4])}",
        )
    return _warn(
        cid,
        name,
        "the claimed floor exceeds the realized fixtures and the script(s) that "
        f"produce them — {', '.join(ev.producer_candidates[:4])} — appear nowhere in "
        "the plan; add the change to the plan's modified-file list, or declare "
        "'N/A — no fixture-producing script change needed' on its own line, unwrapped "
        "(no backticks/quotes)",
    )


# ─── Check 67 — test-retest κ gate vs temperature-0 judge pin (#2204) ──────
# Origin: #2202 plan v1 — a Sonnet label wave pinned at temperature 0 while
# registering "test-retest → κ per mode; κ<0.6 demoted to report-only". At
# temperature 0 a byte-identical retest returns near-identical output: κ≈1
# for every mode, the demotion gate is unfireable by construction, and the
# gate is a false instrument-validity screen. The parent instrument (#1738,
# scripts/issue1738_characterize.py:326) ran at API-default temperature;
# its κ range 0.786-0.982 is meaningful only under sampling. WARN-ONLY:
# regex/section heuristics over prose, false-positive tolerant by design.
# CORPUS CALIBRATION (measured at critic round 1, 2026-08-19, over every
# persisted plan version under tasks/*/*/plans/v*.md — the c50 house pattern
# of recording realized numbers in the check's own comment). Retest-bearing
# plan versions: 90 (63 under kind: experiment, 12 under kind: infra).
# Armed-kind WARNs: 10 — the founding incident (#2202 v1) plus 9 lineage true
# positives (#1482 v13-v17, #1738 v1-v4; e.g. #1738 v4:305, #1482 v14:137),
# each a REAL same-line "1 draw temp 0 + retest κ ... demotes" pin. The
# #1482 -> #1738 -> #2202 lineage carried the unfireable-gate shape in plan
# text for THREE generations (#1738's implementation silently deviated to API
# default, issue1738_characterize.py:326), so c67 would have fired usefully on
# all of them. Zero infra-kind true positives — which is why the kind gate
# below costs no measured coverage.
#
# OUT OF SCOPE — same failure family (a variance-dependent gate read over a
# deterministic surface), deliberately NOT detected. Route the next instance
# to a widening decision rather than "why didn't c67 fire":
# self-consistency / majority-vote gates at temperature 0; draw-variance (SD)
# or bootstrap-CI-width gates over a deterministic estimator;
# split-half-over-draws; and the cache-served κ≡1 case, which is invisible to
# ANY temperature predicate and is carried only in the WARN detail's
# companion-trap sentence.

#: A line registering the retest gate: retest + κ/kappa on ONE line.
_C67_RETEST_RE = re.compile(r"(?i)\btest[-\s]?retest\b|\bretest\b")
_C67_KAPPA_RE = re.compile(r"(?i)κ|\bkappa\b")
#: Demotion/threshold clause (searched over the gate line's section).
_C67_DEMOTE_RE = re.compile(r"(?i)\bdemot\w*|report[-\s]only|(?:κ|kappa)\s*[<≤]")
#: A temperature-0 pin: `temperature 0` / `temperature=0` / `temp: 0` /
#: `temperature 0.0`; the lookahead rejects `0.5`/`0.7` (nonzero decimals).
_C67_TEMP0_RE = re.compile(r"(?i)\btemp(?:erature)?\b\s*[=:]?\s*[`*]{0,2}\s*0(?:\.0+)?(?![.\d])")
#: Same-line negation: an explicit API-default / nonzero pin dominates —
#: corrected plans QUOTE the trap ("at temperature 0 ... κ≈1") beside their
#: real pin (`temperature = API default (1.0)`), the #2202 v2 shape.
_C67_TEMP_NONZERO_RE = re.compile(
    r"(?i)\btemp(?:erature)?\b\s*[=:]?\s*[`*]{0,2}\s*(?:API[-\s]default|1(?:\.0+)?\b|0?\.[0-9]*[1-9])"
)


def check_retest_kappa_temp0(plan: str, kind: str) -> CheckResult:
    """WARN when a registered test-retest κ demotion gate and a judge
    temperature-0 pin share the same innermost section (#2202 v1 shape):
    a deterministic judge always agrees with itself, so the κ<0.6 gate can
    only pass — a false instrument-validity screen. Same-line API-default/
    nonzero pins negate a temp-0 mention (corrected plans explain the trap
    in prose beside their real pin — the #2202 v2 shape, kept silent).
    Best-effort scoping: gate and pin in different innermost sections stay
    silent. Armed for ``kind`` in {experiment, analysis} only — infra
    workflow-fix plans discussing this check legitimately quote the trigger
    vocabulary (the c53 kind-exempt precedent). Escape:
    ``N/A — no test-retest gate`` standalone, unwrapped."""
    cid, name = "c67_retest_kappa_temp0", "test-retest κ gate vs temperature-0 judge pin"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            f"kind={kind} — armed for experiment/analysis only (the c53 kind-exempt "
            "precedent): infra workflow-fix plans (THIS check's own plan included) "
            "legitimately QUOTE the trigger vocabulary without dispatching a judged "
            "retest wave. All three real incidents (#2202 v1, #1738 v1-v4, #1482 "
            "v13-v17) are kind: experiment",
        )

    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    gate_idx: list[int] = []
    temp0_idx: list[int] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        if _C67_RETEST_RE.search(line) and _C67_KAPPA_RE.search(line):
            gate_idx.append(i)
        if _C67_TEMP0_RE.search(line) and not _C67_TEMP_NONZERO_RE.search(line):
            temp0_idx.append(i)

    def _sec(i: int) -> Heading | None:
        return _innermost_section(headings, i)

    def _sec_text(h: Heading | None) -> str:
        return "\n".join(lines[h.line : h.end]) if h else "\n".join(lines)

    registered = [i for i in gate_idx if _C67_DEMOTE_RE.search(_sec_text(_sec(i)))]
    if not registered:
        return _skip(cid, name, "no test-retest κ demotion gate detected")
    if _standalone_na_declared(plan, r"no test[-\s]?retest gate\b"):
        return _pass(cid, name, "explicit N/A declared (no test-retest gate)")
    # Same innermost Heading (or both pre-heading None) ⇒ co-located.
    hits = [(g, t) for g in registered for t in temp0_idx if _sec(g) is _sec(t)]
    if not hits:
        return _pass(
            cid,
            name,
            "test-retest κ demotion gate registered; no temperature-0 judge pin "
            "co-located in the same section (unpinned / API-default / unrelated-section "
            "temperatures are all fine — κ has variance only under sampling)",
        )
    g, t = hits[0]
    return _warn(
        cid,
        name,
        f"a temperature-0 judge pin (line {t + 1}) and a test-retest κ demotion gate "
        f"(line {g + 1}) share the same section — at temperature-0 a byte-identical "
        "retest returns near-identical output, κ≈1 for every mode, and the demotion "
        "gate can only pass: a false instrument-validity screen (#2202 v1; the #1738 "
        "parent ran at API-default temperature, issue1738_characterize.py:326, which is "
        "why its κ 0.786-0.982 range is meaningful). Companion trap: a rubric-keyed "
        "judge CACHE serving the first-pass verdict back to the retest row makes κ≡1 "
        "at ANY temperature — retest rows need a distinct custom-id prefix (the #1738 "
        "`rt_` convention, issue1738_characterize.py:303) or a fresh cache_dir. "
        "Remedies: run the retest at the parent instrument's sampling temperature "
        "(API default), or re-ground the κ threshold for a deterministic surface, or "
        "declare 'N/A — no test-retest gate' on its own line, unwrapped (no "
        "backticks/quotes)",
    )


# ─── Check 64: sampled exactness claim vs runtime-assert grain (#2174) ─────
# Origin: #2163 — plan §12 A11 asserted byte-identity / `n_distinct_rows = 1`
# at `Confidence: High (measured)` from a 10-shard/706-row probe (0.5% of the
# 142,000-row population), registered a Phase-0 full-store assert on it, and
# the census died rc=1 on 258 deviating rows AFTER provisioning. A sample
# only ever establishes "no counterexample observed in N of M", never "zero
# counterexamples exist". Trigger + satisfier are text heuristics, so c64
# SHIPS WARN-ONLY (the c34/c43/c63 posture); the binding gate is the Phase
# 1.5 fact-checker EXACTNESS-CLAIM GRAIN CHECK (adversarial-planner
# SKILL.md) + planner.md section 12's bound-restatement clause.

#: Exactness-identity vocabulary on non-fenced lines. Deliberately narrow:
#: a sampled VALUE claim (a lambda, a byte count, a key set) never matches,
#: and the byte-identical arm is scoped to DATA nouns (rows/vectors/...)
#: because the corpus's dominant `byte-identical` idiom is CODE-equality
#: prose ("every existing caller byte-identical", "path must stay
#: byte-identical", "byte-identical with no flags" — 2026-08-18 sweep),
#: which asserts behavior preservation, not a sampled data identity; the
#: noun scoping also drops negated uses ("(not byte-identical) endpoint
#: sample"). Case-INSENSITIVE as of round 2 (the reconciler-deferred
#: trigger-case concern, applied after the 2026-08-18 corpus re-sweep
#: measured 8 < 10 defensible file-WARNs with IGNORECASE on: the
#: sample-marker window is the binding FP control — lowercase "every row"
#: alone appears 245x across 157 plans, but only 3 meta-plan self-hits
#: survive the window + satisfier gates).
_C64_EXACTNESS_RE = re.compile(
    r"\b(?:rows?|vectors?|shards?|files?|tensors?|copies)\b[^\n]{0,40}byte[- ]identical"
    r"|byte[- ]identical[^\n]{0,40}\b(?:rows?|vectors?|shards?|files?|tensors?|copies)\b"
    r"|n_distinct(?:_rows|_prefix)?\s*={1,2}\s*1\b"
    r"|exactly zero"
    r"|zero (?:cross-context )?variance"
    r"|max\|[^|\n]{0,40}\|\s*=\s*0(?![.\d])"
    r"|all pairwise[^\n]{0,40}=\s*1\.0+"
    r"|EVERY row\b"
    r"|\bno exceptions\b",
    re.IGNORECASE,
)

#: Sample-size markers near the exactness line — SAMPLING-EXPLICIT forms
#: only ("10 sampled shards", "10-shard", "706-row sample", "a sample of
#: 706", "706 of 142,000 rows"). Deliberately NOT bare "N rows"/"N shards":
#: row counts and store-shard counts are ubiquitous in plan prose — the
#: 2026-08-18 calibration sweep measured 227/4,331 corpus file-WARNs under
#: the loose form (byte-identical GUARD/code-equality prose beside
#: incidental row counts). The number atom uses proper thousands grouping
#: so "137, sampled-quar" cannot read as "137 sampled".
_C64_SAMPLE_RE = re.compile(
    r"\b\d+(?:,\d{3})*[-\s]sampled\b"
    r"|\b\d+(?:,\d{3})*-shard\b"
    r"|\b\d+(?:,\d{3})*-row\s+(?:sample|probe|subset)\b"
    r"|\b(?:sample|probe|subset)\s+of\s+\d+(?:,\d{3})*\b"
    r"|\b\d+(?:,\d{3})*\s+of\s+\d+(?:,\d{3})*\s+(?:rows|shards)\b",
    re.IGNORECASE,
)

#: Full-grain phrases — satisfy ONLY beside completed-verification
#: vocabulary on the SAME line (see the deferred-verification veto).
_C64_FULL_GRAIN_RE = re.compile(
    r"full[- ]grain"
    r"|over the full"
    r"|entire (?:store|corpus|population|dataset)"
    r"|all \d[\d,]* rows"
    r"|full (?:staged )?(?:store|corpus|population|dataset)",
    re.IGNORECASE,
)

#: Completed-verification vocabulary (the satisfier's second conjunct).
_C64_COMPLETED_RE = re.compile(
    r"\bverified\b|\bmeasured\b|\bconfirmed\b|\bcounted\b", re.IGNORECASE
)

#: Deferred-verification phrasing VETOES a full-grain phrase on its line:
#: "Phase 0 re-asserts ... over the FULL staged store" is a deferral, not a
#: completed verification — the #2163 A11 How-to-verify line itself (the
#: founding incident MUST WARN; pinned by the verbatim-replay test). The
#: veto also covers NEGATED completion ("not verified", "never measured",
#: "unverified") and PROSPECTIVE completion ("will be verified", "to be
#: measured") — completion vocabulary under negation or in the future
#: tense is not a completed verification (round-1 review concern
#: c64-completion-polarity). Line-scoped and conservative by design: a
#: false veto only converts a satisfier line back into the WARN-only
#: default.
_C64_DEFERRED_RE = re.compile(
    r"\bre-?asserts?\b|\bwill\b[^\n]{0,40}\bassert\b|\bre-?checks?\s+at\s+runtime\b"
    r"|\b(?:not|never)\b[^\n]{0,20}\b(?:verified|measured|confirmed|counted)\b"
    r"|\bun(?:verified|measured|confirmed|counted)\b"
    r"|\b(?:will|to)\s+be\s+(?:verified|measured|confirmed|counted)\b",
    re.IGNORECASE,
)

#: Bound restatement — the second remedy; window-level satisfier.
_C64_BOUND_RE = re.compile(r"no (?:deviation|counterexample)s? observed in", re.IGNORECASE)


def check_exactness_grain(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional, all kinds, both modes: a non-fenced line
    asserting an EXACT identity (byte-identical / ``n_distinct == 1`` /
    exactly zero / zero variance / EVERY row / no exceptions /
    ``max|.| = 0`` / all pairwise = 1.0) with a SAMPLE-SIZE marker in the
    +-3-line window and NO satisfier there WARNs: a sample only ever
    establishes "no counterexample observed in N of M", never "zero
    counterexamples exist", so a runtime assert built on the exact claim
    crashes at the first full-population read (#2163). Satisfiers,
    window-level: (a) COMPLETED full-grain verification — a full-grain
    phrase on a line that ALSO carries completed-verification vocabulary
    (verified/measured/confirmed/counted) and NO deferred-verification
    phrasing (re-asserts / will ... assert / re-checks at runtime — nor
    NEGATED ("not/never ... verified", "unverified") or PROSPECTIVE
    ("will be / to be verified") completion: the
    #2163 A11 "Phase 0 re-asserts ... over the FULL staged store" line is
    a deferral and must NOT satisfy); (b) a bound restatement ("no
    deviation observed in N of M rows"). Escape: the standalone
    'N/A — no sampled exactness claims' line, unwrapped. NEVER FAILs (the
    c34/c43 posture — trigger + satisfier are text heuristics; the
    binding gate is the Phase 1.5 fact-checker EXACTNESS-CLAIM GRAIN
    CHECK)."""
    del kind  # all kinds: an infra plan's sampled exactness premise crashes the same way
    cid, name = "c64_exactness_grain", "sampled exactness claim vs runtime-assert grain"
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    hits: list[tuple[int, str]] = []
    n_satisfied = 0
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _C64_EXACTNESS_RE.search(line)
        if not m:
            continue
        lo, hi = max(0, i - 3), min(len(lines), i + 4)
        window = [lines[j] for j in range(lo, hi) if not mask[j]]
        if not any(_C64_SAMPLE_RE.search(w) for w in window):
            continue
        satisfied = False
        for w in window:
            if _C64_BOUND_RE.search(w):
                satisfied = True
                break
            if (
                _C64_FULL_GRAIN_RE.search(w)
                and _C64_COMPLETED_RE.search(w)
                and not _C64_DEFERRED_RE.search(w)
            ):
                satisfied = True
                break
        if satisfied:
            n_satisfied += 1
        else:
            hits.append((i + 1, m.group(0)))
    if not hits and not n_satisfied:
        return _skip(cid, name, "no sampled exactness claim on non-fenced lines")
    if not hits:
        return _pass(
            cid,
            name,
            f"every sampled exactness claim ({n_satisfied}) carries a window-level "
            "satisfier (completed full-grain verification, or a bound restatement)",
        )
    if _standalone_na_declared(plan, r"no sampled exactness claims"):
        return _pass(cid, name, "explicit N/A declared (no sampled exactness claims)")
    lineno, token = hits[0]
    more = f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""
    return _warn(
        cid,
        name,
        f"line {lineno} asserts an EXACT identity ({token!r}) with a sample-size marker "
        f"in the +-3-line window and no full-grain satisfier{more} — a sample only "
        'establishes "no counterexample observed in N of M", never "zero counterexamples '
        'exist", so a runtime assert built on it crashes at the first full-population '
        "read (#2163: 706 of 142,000 rows probed; the full-store assert died on 258 "
        "deviating rows, after provisioning). Verify at full grain NOW (often exactly "
        "the read the asserting phase already performs), or restate the claim as a "
        'bound ("no deviation observed in N of M rows") AND soften the assert to the '
        "invariant the bound supports, or declare 'N/A — no sampled exactness claims' "
        "on its own line, unwrapped (no backticks/quotes)",
    )


# ─── Check 68 — abs-pp reduction margin vs in-plan baseline ceiling (#2228) ─
# Origin: #2203 plan v10-v12 — §3 registered H1/H3-confirm as an ABSOLUTE
# "baseline - cap ≥ 10 percentage points" / "≤ baseline - 10pp" reduction
# margin while the plan's own §2/§8 stated baselines of ~9.7% (7B) and
# ~4.0% (32B): margin > baseline makes the confirm branch arithmetically
# unsatisfiable (the treated rate would have to be NEGATIVE), and
# margin == baseline is satisfiable only at an exactly-zero realized rate —
# a degenerate gate; the check fires on BOTH per the body's ≥ spec (#2228
# r1 D1: never claim blanket impossibility across the non-strict boundary).
# Incident lineage: 3rd recurrence of the #810 margin-vs-ceiling family
# (#810; #825 v17 → c28; #2203 v12 → this check). SCOPE (#2228 r1 C6): c68
# detects SUBTRACTION-ANCHORED A1/A2 recurrences of the #2203 surface form
# ONLY — it does NOT cover the margin-vs-ceiling family generally (every
# enumerated accepted FN below is a family member it will not see); the
# binding family-level defense remains the Statistics critic's
# decision-gate joint-satisfiability lens. Caught at #2203 only by the
# Statistics + Alternatives critics; c28 (multiplicative fractional bands,
# same-line, no-% harvest) and c20 (lattice structure) structurally cannot
# see it. WARN-ONLY (the c14/c28 doctrine): regex heuristics over prose;
# the FAIL-grade semantic verdict stays with the Statistics critic.
#
# HARVEST DOCTRINE (prefer false negatives): margins are harvested ONLY in
# the baseline-SUBTRACTIVE forms — the clause must literally name the
# subtraction from baseline, which is precisely the arithmetic under test:
#   A1: "≤/<=/< baseline - N pp|percentage points"     (#2203 v12 L57)
#   A2: "baseline - <arm> ≥/>=/> N pp|percentage points" (#2203 v12 L44)
# DELIBERATELY EXCLUDED (each a named accepted false negative; the starred
# forms are pinned as executable SKIP fixtures — #2228 r1 C7):
#   - * bare "Δ ≥ N pp" / ">= N pp" without the subtractive anchor —
#     direction-ambiguous (an INCREASE margin is satisfiable). NOTE: this
#     is the #2228 task body's OWN example sentence, and it appears
#     un-harvested in the founding lattice as "Δharm ≥ 10pp" — the
#     incident line fires only via its co-resident A1 token; a recurrence
#     carrying ONLY the bare Δ form is out of reach by design;
#   - * reversed subtractive "baseline - 10pp ≥ treated-rate" (comparator
#     on the wrong side of the subtraction for both arms);
#   - * cross-line / definition-split forms ("Δharm ≥ 10pp" with
#     "Δharm = baseline - treated" defined elsewhere) — cross-token
#     association is out of regex reach;
#   - * parenthesized "≤ (baseline - 10pp)" — A1's comparator must abut
#     "baseline"; "(" is not consumed;
#   - * A2 middles containing any barred character ("baseline - harm,
#     treated ≥ 10pp") — "=" and "," barred by the #2228 r2 tighten (the
#     round-1 middle class crossed an equality comparator and a comma
#     clause boundary and fabricated a cross-clause harvest:
#     "baseline - cap = 2pp, while accuracy >= 10pp" bound the unrelated
#     ">= 10pp" — a false WARN on a healthy plan whose registered gate
#     was a satisfiable 2pp reduction); "?", "!", and the Markdown
#     table-cell pipe "|" barred by the #2228 r3 tighten (the same
#     species, each verified as a live end-to-end false WARN:
#     "baseline - cap? ...", "baseline - cap! ...", and
#     "baseline - cap of 2pp | accuracy >= 10pp" crossed a sentence or
#     table-cell boundary to bind the unrelated ">= 10pp"); a GENUINE
#     margin whose <arm> text carries any barred character is the
#     accepted FN the tightens buy (pinned as the r2 + r3 repro SKIP
#     fixtures);
#   - * range margins ("≥ 10-15pp"); also fraction baselines
#     ("baseline ~0.097" — no %, never harvested) and baselines stated
#     only in a cited artifact (clarifier Assumption 2: in-plan only; the
#     named-artifact leg is a deferred follow-up);
#   - verb-anchored forms ("reduces ... by ≥ N pp") — the 2026-08-20 corpus
#     scan measured 3 near-FPs for a reduc|drop|lower vocabulary arm
#     (#192 v2:652 "lower fact margin ... ≥ 30pp"; #543 v1-v4 "≥ 2 strict
#     drops of ≥ 5 pp"; #376 v1:481 "If either drops ≥10pp");
#   - "< N pp" complement clauses (usually tolerance/noise bounds);
#   - two-sided tolerances ("within ~5 pp of baseline") — no comparator+
#     subtraction, excluded by construction;
#   - "≥ baseline - N pp" retention FLOORS (satisfiable) — excluded by
#     A1's comparator class (≤/<=/< only).
# Baselines: first "%"-number within 40 chars after a "baseline" token,
# plan-wide (the incident's baselines live in §2 / the §8 risk table —
# cross-section is the point). Comparison uses the MAX harvested baseline:
# fires only when even the largest stated baseline cannot support the
# margin — the maximally FN-biased association rule (a plan stating 9.7%
# AND 4.0% fires at margin 10; a plan also stating an unrelated 60%
# baseline does not — accepted FN, disclosed in the WARN detail's
# cross-quantity clause; the INVERSE mis-association — an unrelated
# SMALLER % as bmax — is the cross-quantity FP the truthful escape below
# serves).
# SECTION SCOPING (#2228 r1 MF1): margin lines must sit inside a
# gate/hypothesis-titled section whose qualifying heading is NON-H1
# (level >= 2). The house template has ONE H1 whose _headings span is the
# WHOLE document (a span ends at the next same-or-higher heading), so an
# H1 carrying gate/decision vocabulary ("# Plan: ... decision-gate rework")
# would arm every §2/§8 line — the traced FP: a Risks quotation of a
# sibling's broken 30pp margin vs a 20% baseline WARNs on a healthy plan.
# The walk stays ANY-ENCLOSING over H2-H6 (the c13/c28 membership idiom;
# strict nearest-section-only would newly DROP a margin in a non-matching
# subsection of a matching gate H2, e.g. "## 7. Decision Gates" →
# "### Response ladder"). BOUNDARY, stated honestly: a §2/§8 quotation of
# a sibling's broken margin is exempt only while OUTSIDE every H2-H6
# gate/hypothesis/evaluation-titled section — a quotation INSIDE one IS
# harvested (regexes cannot tell a quotation from a registration);
# WARN-only posture + the two escapes are the mitigation there.
# (_C13_GATE_SECTION_RE | _C20_SECTION_RE union, both reused verbatim — c13's
# covers success/kill/decision/evaluation, c20's adds hypothes|verdict,
# which is where the #2203 lattice lives; founding-incident coverage
# verified: v12's margins sit under "### 3. Hypothesis", an H3.)
#
# ESCAPES (#2228 r1 MF2 — two phrases, the registry's two shapes):
#   ``N/A — no absolute-margin decision gate`` — declare ONLY when the
#     plan genuinely registers no absolute-pp reduction margin (the
#     harvested line quotes an incident/sibling; the vocabulary is
#     incidental). The `no <thing>` family: declare-only-when-true.
#   ``N/A — harvested percentage baseline is unrelated to this
#     absolute-margin gate`` — the exists-but-false-alarm shape (the
#     c47/c53/c59 convention): the plan DOES register such a margin, but
#     every %-stated baseline the harvest can see concerns a DIFFERENT
#     quantity. Preferred remedy: state the gate's true baseline in %
#     form so the harvest sees it.
#
# CORPUS CALIBRATION (REALIZED, re-measured 2026-08-20 POST-r3-TIGHTEN
# at corpus HEAD db34939cd6 with THIS function, over every persisted
# tasks/*/*/plans/v*.md — 4,447 versions; the c50/c67 house pattern;
# #2228 r1 C2 records raw-token and function-classified counts
# SEPARATELY). RAW-TOKEN counts (module-regex approximation over raw
# file text, fences/sections ignored): A1-form files 5; A2-form files 5
# — the #2203 family {v10,v11,v12} plus #2228's own plans/{v1,v2}.md
# (kind: infra; grows with #2228's own plan versions); the r2 and r3
# tightens left both counts unchanged — no corpus A2 middle carries any
# barred character (the r2 and r3 classes return the identical file set
# on one corpus snapshot; r1's separate "comparator-pp-bearing 18" grep
# stays dropped as a non-reproducible wider-net approximation — the
# A1/A2 module-regex counts are the reproducible raw-token record).
# REALIZED-FUNCTION counts (the binding measurement, forced
# kind="experiment" as the armed-kind upper bound; the r2 function and
# the r3 function produce the identical verdict set on one corpus
# snapshot): margin-bearing (non-SKIP) 3; WARNs 3 — EXACTLY the
# founding-incident family, each
# (kind: experiment, disposition ARMED, a true positive): 2203/plans/
# v10.md, v11.md, v12.md, all firing on the L44 A2 bullet + the L57 A1
# lattice (10 pp vs bmax 9.7%). ARMED FPs: 0 (the §7 kill criterion does
# not fire). KIND-GATE-COVERED self-family hits: 0 realized — #2228's
# own plan versions carry the A1/A2 raw tokens only fenced or outside
# qualifying NON-H1 gate/hypothesis sections, so the realized function
# never harvests them even under the forced-experiment scan (#2228 r1
# C1: a future self-family hit dispositions KIND-GATE-COVERED, never FP).

#: A1 — treated rate bounded at/below baseline minus the margin.
_C68_A1_RE = re.compile(
    r"(?i)(?:≤|<=|<)\s*baseline\s*[−–—-]\s*"  # noqa: RUF001 — real plan glyphs
    r"(?P<n>\d+(?:\.\d+)?)\s*(?:pp\b|percentage[-\s]?points?)"
)
#: A2 — "baseline - <arm> ≥ N pp". Middle bounded at 40 chars. BARRED
#: set: membership is defined SOLELY by `_C68_A2_MIDDLE_BARRED` below —
#: this comment deliberately carries NO membership enumeration (#2228
#: r4: rounds 1-3 each bounced on a prose restatement of a set the code
#: already owned diverging from the implementation, so the duplicate is
#: deleted rather than re-audited). What the sync test
#: (test_c68_a2_middle_barred_set_syncs_comment_class_and_behavior)
#: actually guarantees: it asserts the CONSTANT below equals the regex's
#: negated character class char-for-char, so a constant/class divergence
#: is test-breaking; it reads no prose, which is exactly why membership
#: must not be restated here. Rationale for the members, by round: the
#: newline, comparator-glyph, ";" and ")" bars are round 1's structural
#: boundaries (genuine ">=" comparators survive by construction — the
#: middle can never consume their leading ">"); bare "=" and "," were
#: barred in r2 after round 1's middle skipped past a satisfiable
#: "= 2pp" equality and bound an unrelated later ">= 10pp" — a
#: fabricated cross-clause harvest, false WARN on a healthy plan (and an
#: "="-only fix would NOT have killed the comma form); "?" "!" "|" were
#: barred in r3 (the same live-FP species crossing a sentence boundary
#: or a Markdown table-cell pipe — the likeliest plan shape, since
#: decision lattices are routinely tables). RESIDUAL, stated as the
#: COMPLEMENT (never an implicitly-exhaustive enumeration — #2228 r3):
#: EVERY character outside the barred set remains permitted within
#: the 40-char budget — e.g. ".", ":", "(", quotes, and dash glyphs (the
#: "." form can cross a sentence boundary the same way); WARN-only
#: posture + the two escapes are the mitigation for whatever crossings
#: the permitted complement allows. A genuine margin whose <arm> text
#: carries a barred character is an accepted FN (excluded list above,
#: pinned by the r2 + r3 repro SKIP fixtures).
_C68_A2_MIDDLE_BARRED = frozenset("\n<>≤≥;)=,?!|")
_C68_A2_RE = re.compile(
    r"(?i)\bbaseline\s*[−–—-]\s*[^\n<>≤≥;)=,?!|]{1,40}?(?:≥|>=|>)\s*"  # noqa: RUF001
    r"(?P<n>\d+(?:\.\d+)?)\s*(?:pp\b|percentage[-\s]?points?)"
)
#: Baseline anchor + the FIRST %-number in a bounded window after it (the
#: window is what keeps "baseline ~9.7% vs paper 65-88%" harvesting 9.7,
#: not 88 — #2203 v12 L298).
_C68_BASELINE_ANCHOR_RE = re.compile(r"(?i)\bbaseline\b")
_C68_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_C68_BASELINE_WINDOW = 40


def _c68_sections_ok(headings: list, i: int) -> bool:
    """Line ``i`` sits inside a gate- or hypothesis/verdict-titled section
    whose qualifying heading is NON-H1 (``level >= 2``): the house
    template's single H1 spans the whole document, so an H1 match would
    arm every line (#2228 r1 MF1). The walk stays any-enclosing over
    H2-H6 (the c13/c28 membership idiom; both regexes reused verbatim,
    never widened here) so a margin in a non-matching subsection of a
    matching gate H2 still qualifies."""
    return any(
        h.line <= i < h.end
        and h.level >= 2
        and (_C13_GATE_SECTION_RE.search(h.text) or _C20_SECTION_RE.search(h.text))
        for h in headings
    )


def _c68_margins(plan: str) -> list[dict]:
    """Baseline-subtractive absolute-pp reduction margins on non-fenced
    lines inside NON-H1 gate/hypothesis sections. Per margin:
    ``{line_no (1-based), line, value: Fraction}``; deduped per
    (line_no, value)."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    headings = _headings(plan)
    out: list[dict] = []
    seen: set[tuple[int, Fraction]] = set()
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced or not _c68_sections_ok(headings, i):
            continue
        for rx in (_C68_A1_RE, _C68_A2_RE):
            for m in rx.finditer(line):
                v = _c28_frac(m.group("n"))
                if v <= 0 or (i, v) in seen:
                    continue
                seen.add((i, v))
                out.append({"line_no": i + 1, "line": line.strip(), "value": v})
    return out


def _c68_baselines(plan: str) -> list[dict]:
    """In-plan %-stated baseline rates, plan-wide over non-fenced lines:
    the FIRST %-number within ``_C68_BASELINE_WINDOW`` chars after each
    ``baseline`` token, kept when 0 < b <= 100. Per baseline:
    ``{line_no, value: Fraction, snippet}``."""
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    out: list[dict] = []
    for i, (line, fenced) in enumerate(zip(lines, mask, strict=True)):
        if fenced:
            continue
        for a in _C68_BASELINE_ANCHOR_RE.finditer(line):
            w = line[a.end() : a.end() + _C68_BASELINE_WINDOW]
            pm = _C68_PCT_RE.search(w)
            if not pm:
                continue
            b = _c28_frac(pm.group(1))
            if 0 < b <= 100:
                out.append(
                    {"line_no": i + 1, "value": b, "snippet": line[a.start() :][:60].strip()}
                )
    return out


def check_margin_baseline_ceiling(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional (#2228, incident #2203 v12): a registered
    absolute percentage-point reduction margin in a baseline-SUBTRACTIVE
    form (``≤ baseline - N pp`` / ``baseline - <arm> ≥ N pp``) inside a
    NON-H1 gate/hypothesis section, whose N is >= the LARGEST in-plan
    %-stated baseline rate. Strictly above the baseline the confirm
    branch is arithmetically unsatisfiable (the treated rate would have
    to be negative); AT the baseline it is satisfiable only at an
    exactly-zero realized rate — a degenerate gate; the check fires on
    both per the task body's ≥ spec. Detects the SUBTRACTION-ANCHORED
    A1/A2 recurrence shape only — never the margin-vs-ceiling family
    generally. NEVER FAILs (the c14/c28 doctrine). Armed for kind in
    {experiment, analysis} only — infra workflow-fix plans (THIS check's
    own plan included) legitimately quote the trigger vocabulary (the
    c53/c67 precedent). Accepted false negatives are enumerated in the
    module comment above. Escapes (standalone, unwrapped):
    ``N/A — no absolute-margin decision gate`` (genuinely gate-free — the
    harvested text quotes an incident/sibling) and ``N/A — harvested
    percentage baseline is unrelated to this absolute-margin gate`` (the
    gate is real; the %-baseline pairing is the false alarm)."""
    cid, name = "c68_margin_baseline_ceiling", "abs-pp reduction margin vs baseline ceiling"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            f"kind={kind} — armed for experiment/analysis only (the c67 kind-exempt "
            "precedent): infra workflow-fix plans quote the pp-margin/baseline trigger "
            "vocabulary without registering a decision margin; the founding incident "
            "(#2203 v10-v12) is kind: experiment",
        )
    margins = _c68_margins(plan)
    if not margins:
        return _skip(cid, name, "no baseline-subtractive absolute-pp reduction margin detected")
    if _standalone_na_declared(plan, r"no absolute[- ]margin decision gates?"):
        return _pass(
            cid,
            name,
            "explicit N/A declared (no absolute-margin decision gate — the harvested "
            "margin text is not a gate this plan registers)",
        )
    if _standalone_na_declared(
        plan, r"harvested percentage baseline is unrelated to this absolute[- ]margin gate"
    ):
        return _pass(
            cid,
            name,
            "explicit N/A declared (cross-quantity: the harvested %-stated baseline is "
            "unrelated to the registered absolute-margin gate)",
        )
    baselines = _c68_baselines(plan)
    if not baselines:
        return _skip(
            cid,
            name,
            "margin registered but no in-plan %-stated baseline rate detected "
            "(cross-artifact baseline recovery is out of v1 scope — clarifier "
            "Assumption 2, #2228)",
        )
    bmax = max(baselines, key=lambda b: b["value"])
    offenders = [m for m in margins if m["value"] >= bmax["value"]]
    if not offenders:
        return _pass(
            cid,
            name,
            f"{len(margins)} baseline-subtractive pp margin(s) all sit below the largest "
            f"in-plan baseline rate ({float(bmax['value']):g}%, line {bmax['line_no']})",
        )
    parts = [
        f'line {m["line_no"]} "{m["line"][:90]}" registers an absolute reduction margin '
        f"of {float(m['value']):g} pp"
        for m in offenders[:3]
    ]
    shown = "; ".join(parts) + ("; …" if len(offenders) > 3 else "")
    return _warn(
        cid,
        name,
        f"{shown} — but the largest in-plan baseline rate is {float(bmax['value']):g}% "
        f'(line {bmax["line_no"]}: "{bmax["snippet"]}"): a margin STRICTLY above the '
        "baseline makes the confirm branch arithmetically unsatisfiable (the treated "
        "rate would have to be negative), and a margin EQUAL to the baseline is "
        "satisfiable only at an exactly-zero realized rate — a degenerate gate; both "
        "fire per the ≥ spec (#2203 v12: a ≥10pp margin vs realized baselines "
        "9.66%/4.02%; 3rd recurrence of the #810 family). NOTE: this check cannot "
        "verify the margin and the baseline concern the same quantity — if they "
        "concern DIFFERENT quantities, state the gate's true baseline in % form so "
        "the harvest sees it, or declare 'N/A — harvested percentage baseline is "
        "unrelated to this absolute-margin gate' on its own line, unwrapped (no "
        "backticks/quotes). Remedy for a genuinely infeasible margin: switch to a "
        "relative margin, or size the absolute margin below the DV's stated baseline "
        "rate. A plan that registers NO absolute-pp margin at all (the harvested line "
        "quotes an incident/sibling) instead declares 'N/A — no absolute-margin "
        "decision gate' on its own line, unwrapped; the semantic verdict stays with "
        "the Statistics critic",
    )


# ─── Check 69 — armed re-gen 2×-cap headroom vs max_model_len pin (#2269) ──  # noqa: RUF003
# Origin: #2221 plan v9 (incident 2026-08-13) — the amendment ARMED the >2%
# cap-hit re-gen trigger (regen at ≥2× EVAL_MAX_NEW_TOKENS = 2×2048 = 4096)  # noqa: RUF003
# against issue778_lib.build_vllm_engine's max_model_len=4096 pin while
# stating CORRECT first-pass arithmetic (2048 + ≤1,900 = 3,948 ≤ 4,096 at
# v9:101): the BINDING regen-leg arithmetic 4,096 + 1,900 = 5,996 > 4,096
# leaves zero prompt headroom — every row skips as regen_overlong_skipped,
# regen_applied: true with n_regen=0. verify_plan on v9: PASS n_warn=0.
# Plan-prose face of the #505/#601 cap-raise-vs-max_model_len family
# (gotchas.md cap-raise entry). SCOPE: #505/#601 themselves were CODE-level
# and no plan linter could have fired on them — this check SURFACES the
# class's plan face at plan time; it does not prevent recurrence (the
# runtime guard on regen_applied/n_regen==0/regen_overlong_skipped>0 is
# the higher-recall companion, out of scope here). The check keys on the
# DOUBLED cap whenever a trigger is armed — a naive stated-triple read
# PASSes v9 by construction.
# CORPUS CALIBRATION — CALIBRATION FIT, snapshot-dated (these figures rot
# as the corpus grows, and the grammar was TUNED on this same corpus: the
# counts certify transcription fidelity at re-scan time, never
# out-of-sample precision; re-calibration MUST apply the kind gate before
# reading any count drift as grammar drift). SHIPPED-check re-scan
# 2026-08-22 04:27 UTC, 4,484 committed plans/v*.md (the plan-time
# 2026-08-22 03:36 UTC prototype scan read 4,486 files on the main
# checkout, glob incl. then-untracked drafts — same verdicts on every
# adjudicated version): arming vocabulary resolves in 45 versions (the
# plan-time 44 + this plan's own v3); 27 SKIP (no pin); 0 SKIP (no
# cap); KIND-GATED WARN 11 —
# #2221 v7/v8/v9 (arm 1: 2×2048 + 1,900 = 5,996 ≥ 4,096) + #2225 v6–v13  # noqa: RUF003
# (arm 2: the inherited "> 2% ⇒ re-gen at 2× (contract inherited)" line  # noqa: RUF003
# against max_model_len 4096 with cap 2048 — need 4,096 ≥ pin 4,096, no
# stated prompt bound; a LIVE zero-headroom defect) — all verified true
# positives; corrected #2221 v10–v13 PASS (5,996 < 8,192); zero confirmed  # noqa: RUF003
# false positives. UN-GATED additionally: #2269's OWN kind:infra plan
# versions (v1, v2, plus one per later version — the plan that designed
# this check quotes the arming vocabulary in non-fenced prose); the kind
# gate + _fence_mask absorb them (self-inclusion is the designed-for case).
# Measured grammar decisions: (a) prose-form "cap (N)" harvesting DROPPED
# — v10:467 "AT THE PRODUCTION REGEN CAP (4096)" reads the ALREADY-DOUBLED
# regen cap and double-counts (4 FPs on the corrected versions); (b)
# multiplier must be ANCHORED (N× followed by cap/number/MAX_NEW_TOKENS)  # noqa: RUF003
# — the house "families × {arms}" grid idiom otherwise harvests spurious  # noqa: RUF003
# multipliers; (c) forward "prompt tokens ... < N" bound shape DROPPED —
# it grabbed the PIN out of v10:106's arithmetic prose; (d) arm
# alternative 3 recognizes the house SHORTHAND: "re-gen ... at N×" ARMS  # noqa: RUF003
# (#2225 v12:97 is the demonstrated in-corpus miss of the narrow
# re-generat form — r1 Must-Fix 2) while a bare "re-gen" mention with no
# "at N×" stays non-arming; (e) bounds are attributed PER ARMED LINE  # noqa: RUF003
# (±_C69_WINDOW_LINES raw-line fence-masked window; window MAX wins;
# plan-wide MAX only as the no-local-bound FALLBACK) — a plan-wide bound
# join is a FALSE-POSITIVE mechanism on healthy multi-stage plans (an
# unrelated stage's larger bound joins an armed stage whose own
# arithmetic is satisfied — r1 Must-Fix 1): bound-side MAX is AGGRESSIVE
# (pro-WARN), the opposite direction from the permissive pin-side MAX,
# so only the pin side may pool plan-wide.
# Accepted FALSE NEGATIVES: a cap stated only as prose ("cap 2048") with
# no constant/product form resolves nothing (SKIP); a plan quoting a
# sibling's LARGER pin while its own engine pins smaller escapes the
# MAX-pin read; a STALE SMALL bound inside an armed line's window masks a
# larger binding bound stated elsewhere (the window-first trade — FNs
# disclosed and accepted, FPs kill); verbal multipliers ("doubling",
# "twice") and arming paraphrases outside the three alternatives.

_C69_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
#: Arming vocabulary, negation-guarded on the leading token
#: (_C16_NEG_GUARD reused verbatim) AND on the "armed" token itself
#: ("the re-gen trigger is NOT armed" must not fire): "re-gen trigger
#: ARMED" / "trigger armed ... re-gen" / the CLAUDE.md registration shape
#: in BOTH spellings — "re-generate <rows> at ≥2×" AND the house  # noqa: RUF003
#: shorthand "re-gen at 2×" (#2225 v12:97). Bare "re-gen" with no  # noqa: RUF003
#: "at ... N×" stays non-arming.  # noqa: RUF003
_C69_ARM_RE = re.compile(
    rf"(?i){_C16_NEG_GUARD}\bre-?gen\w*[^.;]{{0,60}}?\btrigger\b[^.;]{{0,40}}?"
    rf"(?:(?<!\bnot )(?<!never )\barmed\b|\barming\b|\bauto-?fires?\b)"
    rf"|{_C16_NEG_GUARD}\btrigger\b[^.;]{{0,40}}?(?<!\bnot )(?<!never )\barmed\b"
    rf"[^.;]{{0,80}}?\bre-?gen\w*"
    rf"|{_C16_NEG_GUARD}\bre-?gen(?:erat\w*)?\b[^.;]{{0,50}}?\bat\b[^.;]{{0,12}}?"
    rf"[≥>]?\s*\d+(?:\.\d+)?\s*[×x]"  # noqa: RUF001 — the multiplication sign is real plan text
)
_C69_PIN_RE = re.compile(rf"(?i)(?:vllm_)?max_model_len\W{{0,4}}({_C69_NUM})")
_C69_CAP_LINE_RE = re.compile(
    rf"(?i)[A-Z_]*max_new_tokens\W{{0,4}}({_C69_NUM})|[×x]\s*({_C69_NUM})"  # noqa: RUF001
)
_C69_CAP_CONST_RE = re.compile(rf"(?i)[A-Z_]*max_new_tokens\W{{0,4}}({_C69_NUM})")
_C69_MULT_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*[×x]\s*(?=(?:the\s+)?cap\b|\d|`?[A-Z_]*MAX_NEW_TOKENS)"  # noqa: RUF001
)
_C69_BOUND_RES = (
    re.compile(rf"(?i)(?:[≤<]=?|at most)\s*~?\s*({_C69_NUM})\s*prompt[- ](?:side[- ])?tokens?"),
    re.compile(
        rf"(?i)prompts?\b[^.;]{{0,40}}?(?:validated|bounded|capped|filtered)"
        rf"[^.;]{{0,25}}?[≤<]=?\s*~?\s*({_C69_NUM})\s*(?:prompt[- ])?tokens?"
    ),
)
_C69_WINDOW_LINES = 3  # the c16 _C16_WINDOW_LINES radius (caps AND bound attribution)


def _c69_int(s: str) -> int:
    """Comma-tolerant int (founding fixture: '≤ 1,900 prompt tokens')."""
    return int(s.replace(",", ""))


def _c69_bounds_between(lines: list[str], mask: list[bool], lo: int, hi: int) -> list[int]:
    """Fence-masked stated prompt-token bounds on raw lines [lo, hi)."""
    return [
        _c69_int(m.group(1))
        for j in range(max(0, lo), min(len(lines), hi))
        if not mask[j]
        for rx in _C69_BOUND_RES
        for m in rx.finditer(lines[j])
        if 1 <= _c69_int(m.group(1)) <= 1_000_000
    ]


def _c69_evaluated(
    lines: list[str], mask: list[bool], arm_idx: list[int], plan_bounds: list[int]
) -> list[dict]:
    """Per armed line: anchored multiplier (clamped [1, 8], default 2.0),
    generation cap (constant- or product-form on the arming line;
    constant-form within the ±3-raw-line window as fallback), and the
    attributed prompt bound. Bound attribution is PER ARMED LINE (r1
    Must-Fix 1): window-local MAX wins; the plan-wide MAX is only the
    no-local-bound fallback. Bound-side MAX is aggressive (pro-WARN) —
    opposite direction from the permissive pin-side MAX — so it never
    pools across stages. Armed lines with no resolvable cap are dropped
    (the caller SKIPs when nothing evaluates)."""
    out: list[dict] = []
    for i in arm_idx:
        line = lines[i]
        mults = [
            float(m.group(1))
            for m in _C69_MULT_RE.finditer(line)
            if 1.0 <= float(m.group(1)) <= 8.0
        ]
        mult = max(mults) if mults else 2.0
        caps = [
            _c69_int(g)
            for m in _C69_CAP_LINE_RE.finditer(line)
            for g in m.groups()
            if g and 16 <= _c69_int(g) <= 100_000
        ]
        if not caps:  # constant-form only in the ±3-raw-line window
            window = (
                lines[max(0, i - _C69_WINDOW_LINES) : i] + lines[i + 1 : i + 1 + _C69_WINDOW_LINES]
            )
            caps = [
                _c69_int(m.group(1))
                for src in window
                for m in _C69_CAP_CONST_RE.finditer(src)
                if 16 <= _c69_int(m.group(1)) <= 100_000
            ]
        if not caps:
            continue
        wb = _c69_bounds_between(lines, mask, i - _C69_WINDOW_LINES, i + 1 + _C69_WINDOW_LINES)
        bound = max(wb) if wb else (max(plan_bounds) if plan_bounds else None)
        out.append(
            {
                "line_no": i + 1,
                "line": line.strip(),
                "mult": mult,
                "cap": max(caps),
                "need": int(mult * max(caps)),
                "bound": bound,
                "scope": "window-local" if wb else "plan-wide",
            }
        )
    return out


def check_regen_headroom(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional (#2269, incident #2221 v9): an ARMED cap-hit
    re-generation trigger — the long spelling ("re-generates capped rows
    at >=2x the cap") or the house shorthand ("re-gen at 2x") — beside a
    numeric ``max_model_len`` / ``VLLM_MAX_MODEL_LEN`` pin, where the
    DOUBLED cap leaves non-positive prompt headroom. Two WARN arms:
    (1) mult*cap + the stated prompt-token bound >= pin on any armed
    line; (2) NO prompt-token bound stated anywhere in the plan. Keys on
    the DOUBLED cap whenever the trigger is armed — the founding incident
    stated CORRECT first-pass arithmetic (2048 + 1,900 = 3,948 <= 4,096
    at v9:101) while the ARMED regen leg at 2x2048 = 4,096 left zero
    headroom, so every capped row skips as ``regen_overlong_skipped``
    while ``regen_applied: true`` reports the fix took — a naive
    stated-triple reader PASSes v9 by construction. Bounds attribute PER
    ARMED LINE (window-local MAX over the fence-masked ±3-raw-line
    window; plan-wide MAX only as the no-local-bound fallback); the
    effective pin is the plan-wide MAX from a RAW scan, fences included
    (corrected plans quote the superseded pin beside the raise). NEVER
    FAILs (the c14/c28 doctrine). Armed for kind in {experiment,
    analysis} only — infra workflow-fix plans (THIS check's own plan
    included) legitimately quote the arming vocabulary (the c67/c68
    precedent). Accepted false negatives are enumerated in the module
    comment above. Escapes (standalone, unwrapped):
    ``N/A — no armed re-gen trigger`` (the arming vocabulary is
    incidental or quotes an incident/sibling) and ``N/A — harvested
    max_model_len pin is unrelated to the armed re-gen stage`` (the
    trigger is real; the pin pairing is the false alarm)."""
    cid = "c69_regen_headroom"
    name = "armed re-gen 2×-cap headroom vs max_model_len pin"  # noqa: RUF001
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            f"kind={kind} — armed for experiment/analysis only (c67/c68 precedent): "
            "infra workflow-fix plans, this check's own plan included, quote the "
            "arming vocabulary; the founding incident (#2221 v9) is kind: experiment",
        )
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    arm_idx = [
        i
        for i, (line, fenced) in enumerate(zip(lines, mask, strict=True))
        if not fenced and _C69_ARM_RE.search(line)
    ]
    if not arm_idx:
        return _skip(cid, name, "no armed re-gen trigger detected")
    if _standalone_na_declared(plan, r"no armed re-?gen(?:eration)? trigger\b"):
        return _pass(cid, name, "explicit N/A declared (no armed re-gen trigger)")
    if _standalone_na_declared(
        plan, r"harvested max_model_len pin is unrelated to the armed re-?gen stage"
    ):
        return _pass(
            cid,
            name,
            "explicit N/A declared (cross-quantity: the harvested max_model_len pin "
            "belongs to a different engine/stage)",
        )
    # RAW pin scan, fences included (pins legitimately live in fenced §10 tables).
    pins = [
        _c69_int(m.group(1))
        for m in _C69_PIN_RE.finditer(plan)
        if 256 <= _c69_int(m.group(1)) <= 10_000_000
    ]
    if not pins:
        return _skip(
            cid,
            name,
            "armed re-gen trigger detected but no max_model_len / VLLM_MAX_MODEL_LEN "
            "numeric pin harvested — the #505/#601 headroom arithmetic needs a pin",
        )
    pin = max(pins)
    plan_bounds = _c69_bounds_between(lines, mask, 0, len(lines))
    evaluated = _c69_evaluated(lines, mask, arm_idx, plan_bounds)
    if not evaluated:
        return _skip(
            cid,
            name,
            f"{len(arm_idx)} armed re-gen line(s) but no generation cap resolvable "
            "(constant-form [A-Z_]*MAX_NEW_TOKENS or product-form N×<cap> on the "  # noqa: RUF001
            "arming line; constant-form within ±3 raw lines) — arithmetic not "
            "adjudicable",
        )
    if not plan_bounds:
        worst = max(evaluated, key=lambda e: e["need"])
        zero = (
            " — the regen length alone meets/exceeds the pin (zero prompt headroom "
            "before any prompt tokens)"
            if worst["need"] >= pin
            else ""
        )
        return _warn(
            cid,
            name,
            f"armed re-gen trigger (line {worst['line_no']}) + max_model_len pin "
            f"{pin} but NO stated prompt-token bound anywhere — the binding regen "
            f"arithmetic ({worst['mult']:g}×{worst['cap']} = {worst['need']} + "  # noqa: RUF001
            f"prompt) is unverifiable{zero} (#2221 v9 shape). State the panel's "
            "prompt-token bound (e.g. 'length-validated at load to ≤ N prompt "
            "tokens'), size the regen engine so max_model_len − 2×cap ≥ the bound, "  # noqa: RUF001
            "or declare 'N/A — no armed re-gen trigger' on its own line, unwrapped "
            "(no backticks/quotes)",
        )
    offenders = [e for e in evaluated if e["need"] + e["bound"] >= pin]
    if not offenders:
        best = max(evaluated, key=lambda e: e["need"] + e["bound"])
        return _pass(
            cid,
            name,
            f"regen headroom satisfied: worst armed line {best['line_no']} needs "
            f"{best['mult']:g}×{best['cap']} + {best['bound']} ({best['scope']} "  # noqa: RUF001
            f"bound) = {best['need'] + best['bound']} < max_model_len {pin}",
        )
    parts = [
        f'line {e["line_no"]} "{e["line"][:70]}" arms regen at '
        f"{e['mult']:g}×{e['cap']} = {e['need']}; + prompt bound {e['bound']} "  # noqa: RUF001
        f"({e['scope']}) = {e['need'] + e['bound']} ≥ max_model_len {pin}"
        for e in offenders[:3]
    ]
    shown = "; ".join(parts) + ("; …" if len(offenders) > 3 else "")
    return _warn(
        cid,
        name,
        f"{shown} — zero/negative prompt headroom on the ARMED re-gen leg: every "
        "capped row skips as regen_overlong_skipped and the cap-hit deviation is "
        "silently re-committed while the plan claims to fix it (#2221 v9; the "
        "plan-prose face of the #505/#601 family, gotchas.md cap-raise entry). "
        "First-pass arithmetic (cap + bound ≤ pin) is NOT the binding check — the "
        "regen leg re-enters prompt + a ≥2×-cap response. Remedies: run the regen "  # noqa: RUF001
        "leg on an engine sized max_model_len ≥ 2×cap + prompt bound (the #2221 "  # noqa: RUF001
        "v10 fix: a dedicated 8192 engine), lower the regen multiplier/cap, or — "
        "when the harvested pin belongs to a different engine than the regen "
        "stage — declare 'N/A — harvested max_model_len pin is unrelated to the "
        "armed re-gen stage' on its own line, unwrapped (no backticks/quotes); a "
        "plan with no armed trigger at all instead declares 'N/A — no armed re-gen "
        "trigger' on its own line, unwrapped",
    )


# ─── Check 70 — judge-pilot per-arm draw resolution vs parse-fail threshold ─
# (#2299; founding #2124, recurrence #2162 v7 §7.3.) The runtime helper
# `eval.judge_pilot.judge_pilot_gate` REFUSES a pilot config whose per-arm
# effective draws sit below floor(1/parse_fail_threshold) + 1 (rule 26(b)'s
# strict '< threshold' verdict: at 50 draws and 2%, ONE failure reads
# exactly 2% — not '< 2%' — so 51 is the floor). This check catches the same
# arithmetic in the PLAN TEXT, before approval; it invents no new rule.
# Conservative parse: WARN only when the full (threshold, draws, arms)
# triple resolves inside a ±8-raw-line fence-masked window around a
# non-fenced `pilot` anchor line, with superseded-context lines (`v\d+'s` /
# `gave` / `superseded` / `formerly` / `previously`) dropped from harvest —
# the #2162 v8-v10 CORRECTED revisions quote the superseded v7 config
# verbatim per house convention (round-1 MF2). Every numeric capture is
# bounded `\d{1,9}` (CPython refuses int-str conversion beyond ~4300
# digits, so an unbounded capture is a ValueError path — round-2 critic).
# The lookbehind guard is PER-PATTERN, not universal (round-2 doc fix):
# the DIRECT/COMPONENT/TOTAL/ARMN patterns carry `(?<![\d.,=])`
# (comma-grouping truncation: "9,000 arms" → 000, #2054 v9;
# config-assignment reads: "N=3 draws", #2254 v5 — round-1 MF1 + this
# plan's own replay); the component value extractor _C70_COMPONENT_NUM_RE
# uses `(?<![\w.,])` (word-char guard — a rubric name like "gpt4-rubric"
# must not contribute digits); _C70_THRESH_RE carries NO lookbehind — its
# `parse-fail ... %` context anchor bounds the capture. Calibration (#2299
# §12 A13, re-measured at landing per criterion 5): over 4,519 persisted
# plans — WARN 1 (#2162 v7 only), PASS 2 (#2329 v4, #2389 v1), 0 raises.

_C70_ANCHOR_RE = re.compile(r"(?i)\bpilot\b")
_C70_SUPERSEDED_RE = re.compile(
    r"(?i)\bv\d+['’]s\b|\bsuperseded\b|\bgave\b|\bformerly\b|\bpreviously\b"  # noqa: RUF001
)
_C70_THRESH_RE = re.compile(
    r"(?i)parse[-\s]?fail(?:ure)?(?:\s+rate)?[^%\d]{0,25}?(\d{1,9}(?:\.\d{1,9})?)\s*%"
)
_C70_PER_ARM_TOK_RE = re.compile(r"(?i)\bper[-\s]arm\b|\beach arm\b")
# Direct per-arm form — short-circuits arm-count inference entirely.
_C70_DIRECT_RE = re.compile(
    r"(?i)(?<![\d.,=])\b(\d{1,9})\s*(?:draws?\s*)?/\s*arm\b"
    r"|(?<![\d.,=])\b(\d{1,9})\s+draws?\s+per[-\s]arm\b"
)
# Multi-rubric component form: "60 coherence + 90 value-rubric draws".
_C70_COMPONENT_RE = re.compile(
    r"(?i)(?<![\d.,=])\b\d{1,9}\s+[a-z][\w-]*(?:\s*\+\s*\d{1,9}\s+[a-z][\w-]*)+\s+draws\b"
)
# Component values, extracted structurally (digit + space + letter), NOT a
# bare findall — a rubric name like "gpt4-rubric" must not contribute digits.
_C70_COMPONENT_NUM_RE = re.compile(r"(?i)(?<![\w.,])(\d{1,9})\s+[a-z]")
# Bare total: "150 draws" (a word between digit and "draws" does NOT match).
_C70_TOTAL_RE = re.compile(r"(?i)(?<![\d.,=])\b(\d{1,9})\s+draws\b")
_C70_ARMN_RE = re.compile(r"(?i)(?<![\d.,=])\b(\d{1,9})\s+arms?\b(?!\s+the\b)")
_C70_WINDOW_LINES = 8  # worst measured true-positive distance is 7 (#2162 v7:477 vs :484)


def _c70_resolve_budget(win: str) -> tuple[int, int | None, str] | str:
    """Resolve the effective per-arm draw count from one harvested window.

    Returns ``(per_arm, n_arms, src)`` on success (``n_arms`` is ``None``
    for the direct N/arm form), else the SKIP-reason string
    (S4/S6/S7/S8/S10/S11). Extracted from ``check_pilot_resolution`` for
    C901 (the ``check_battery_multiplier`` precedent).
    """
    dm = _C70_DIRECT_RE.search(win)
    if dm:
        per_arm = int(dm.group(1) or dm.group(2))
        if per_arm == 0:
            return "degenerate zero count harvested (truncated number?)"  # S10
        return per_arm, None, f"direct {per_arm}/arm"
    cm = _C70_COMPONENT_RE.search(win)
    comps = [int(x) for x in _C70_COMPONENT_NUM_RE.findall(cm.group(0))] if cm else None
    totals = sorted({int(m.group(1)) for m in _C70_TOTAL_RE.finditer(win)})
    if comps:
        slice_, src = min(comps), f"min rubric slice {min(comps)} of {comps}"
    elif len(totals) == 1:
        slice_, src = totals[0], f"total {totals[0]}"
    elif len(totals) > 1:
        return f"multiple distinct draw totals {totals} — ambiguous"  # S7
    else:
        return "per-arm threshold without a resolvable draw budget"  # S4
    arm_counts = sorted({int(m.group(1)) for m in _C70_ARMN_RE.finditer(win)})
    if not arm_counts:
        return "draw budget without a resolvable arm count (and no direct per-arm form)"  # S6
    if len(arm_counts) > 1:
        return f"multiple distinct arm counts {arm_counts} — ambiguous"  # S8
    n_arms = arm_counts[0]
    if n_arms == 0 or slice_ == 0:
        return "degenerate zero count harvested (truncated number?)"  # S10
    per_arm = slice_ // n_arms
    if per_arm == 0:
        return "draw budget smaller than arm count — likely mis-harvest"  # S11
    return per_arm, n_arms, f"{src} / {n_arms} arms"


def check_pilot_resolution(plan: str, kind: str) -> CheckResult:
    """WARN-only, conditional (#2299; founding #2124, recurrence #2162 v7
    §7.3): a judge-pilot gate whose per-arm effective draw count sits BELOW
    the resolution its own parse-fail threshold requires — ``per_arm <
    floor(1/threshold) + 1`` under exact ``Fraction`` arithmetic (rule
    26(b)'s strict ``< threshold`` verdict: at 50 draws and 2% one failure
    reads exactly 2%, so 51 is the floor; the float form is off-by-one on
    non-reciprocal thresholds — ``judge_pilot._refusal_resolution_floor``).
    Mirrors ONLY the resolution leg of ``eval.judge_pilot.judge_pilot_gate``'s
    config-time refusal (the ``min_effective_draws_per_arm`` floor leg is
    deliberately not mirrored — FN-e). Anchors on non-fenced ``pilot``
    lines; harvests each ±8-raw-line fence-masked window with
    superseded-context lines dropped; resolves (a) a per-arm parse-fail
    percentage threshold, (b) an effective draw count (direct N/arm >
    additive rubric components with the MIN slice binding — every rubric
    gets its OWN gate call — > unique bare total), and (c) an arm count
    when (b) came from components/total. Eleven SKIP branches implement
    the conservative parse; ``allow_subresolution_pilot`` in a window
    marks its tuple DECLARED (per-tuple PASS; the scan CONTINUES, so a
    plan with one declared sub-resolution gate plus a second genuinely
    defective gate still WARNs on the second TUPLE — the identical-tuple
    and adjacent-window leak shapes are FN-j). NEVER FAILs, NEVER raises
    (the no-flags run feeds the Step 9c gate; ``verify_plan_text`` has no
    per-check exception containment): every numeric capture is bounded
    ``\\d{1,9}``; the lookbehind guard is per-pattern — DIRECT/COMPONENT/
    TOTAL/ARMN carry ``(?<![\\d.,=])``, the component value extractor
    ``_C70_COMPONENT_NUM_RE`` uses ``(?<![\\w.,])``, and
    ``_C70_THRESH_RE`` carries NO lookbehind (its ``parse-fail ... %``
    context anchor bounds the capture; see FP-c) — the only
    division is preceded by the zero-count SKIP, and the percentage
    parses exactly via ``Fraction``. Escapes (standalone, unwrapped):
    ``N/A — no judge-pilot gate`` and ``N/A — harvested pilot sizing is
    historical or belongs to a different gate``.

    Accepted FALSE NEGATIVES: FN-a numbers-as-words / table-split cells /
    adjective-blocked totals ("a ~400-draw pilot", "**540 sync draws**" —
    #2162 v8's correctly-sized gate therefore SKIPs rather than PASSes:
    silence, not a false verdict); FN-b a threshold stated as a bare
    fraction ("0.02") rather than a percentage; FN-c an arm count farther
    than 8 raw lines from every pilot anchor; FN-d a gate spec entirely
    inside a fenced code block; FN-e the ``min_effective_draws_per_arm``
    floor leg not mirrored (corpus thresholds 238x2% + 3x1%; the only
    >10% hits are kind-gated infra plans); FN-f multi-rubric slices not
    in the additive "A <word> + B <word> draws" shape; FN-g an adjective
    between digit and "arms" ("~200 draws spanning the 4 character arms",
    #2054 v9 — a GENUINE sub-resolution config at 50/arm the check
    misses; widening ARMN would re-admit the MF1 false-match class, and
    the helper-side config guard remains the backstop); FN-h the
    superseded-line guard dropping a TRUE defective gate line carrying a
    version-possessive/past-tense token (the suppression direction; the
    rarer ENABLE-flip direction — ambiguity-collapse, declaration-drop —
    measured 0 enable-flips on the corpus); FN-i numeric literals >=10
    digits never capture (the ``\\d{1,9}`` bound — no plausible gate spec
    carries one); FN-j the cross-gate declaration leak (#2299 r1 codex
    Major 1; concern ``c70-cross-gate-declaration-leak``): ``declared``
    is harvested window-globally and OR-accumulated per
    ``(per_arm, required)`` tuple (plan v3 §4.7/§4.9 by design), so ONE
    declared gate silences a SEPARATE undeclared defective gate in two
    shapes — (i) identical tuples: two DISJOINT "30 draws/arm;
    parse-fail < 2% per arm" gates with only the first declared -> PASS
    (the dedup key cannot tell the gates apart); (ii) adjacency: an
    undeclared "20 draws/arm" gate inside a declared gate's ±8-line
    window is shadowed by the first-match harvest
    (``_C70_DIRECT_RE.search`` / ``_C70_THRESH_RE.search`` take the
    first gate's numbers) -> PASS with the 20/arm gate never evaluated.
    The T21-registered shape — a DISTANT second gate in its OWN window
    resolving a DIFFERENT tuple — still WARNs; failure direction is
    silence, 0 occurrences on the calibration corpus.
    KNOWN FALSE POSITIVES: FP-a a per-item draw count with
    NO coexisting budget total in the window ("5 draws per item ... 4
    arms ... parse-fail < 2% per arm" WARNs at 1/arm; sibling shapes
    ALREADY handled: a per-item count beside a real budget -> S7 SKIP,
    the unprefixed "3 draws localize" variant -> S11, the ``N=``-prefixed
    form -> the ``=`` lookbehind, #2254 v5); FP-b a false pairing the
    guards miss (harvested numbers from a neighboring gate that genuinely
    differ) — remedied by the generic escape; FP-c an OBSERVED parse-fail
    rate read as a configured gate threshold (#2299 r1 codex Major 2;
    concern ``c70-observed-rate-false-warn``): ``_C70_THRESH_RE``
    requires no comparator (``<`` / ``<=``) and no threshold vocabulary
    ("threshold", "bound"), so result/report prose like "Judge pilot
    result: 30 draws/arm; observed parse-fail was 2% per arm." WARNs at
    30/arm although the 2% is a measurement, not a registered bound —
    remedy: the second standalone escape ("harvested pilot sizing is
    historical or belongs to a different gate") covers exactly this
    shape; a comparator-vocabulary regex is a viable FUTURE revision
    (the founding #2162 v7 gate line does carry ``<``) but is PLAN-OWNED
    — §4.4 registers the regex as a constant, so changing it requires a
    plan amendment + criterion-5 corpus re-calibration. Measured corpus
    FP rate 0 across all enumerated FP channels.
    """
    cid = "c70_pilot_resolution"
    name = "judge-pilot per-arm draw resolution vs parse-fail threshold"
    if kind not in ("experiment", "analysis"):
        return _skip(
            cid,
            name,
            f"kind={kind} — armed for experiment/analysis only (c69 precedent): "
            "infra workflow-fix plans, this check's own plan included, quote the "
            "arming vocabulary; the founding incident (#2162 v7) is kind: experiment",
        )
    lines = plan.splitlines()
    mask = _fence_mask(lines)
    anchors = [
        i
        for i, (line, fenced) in enumerate(zip(lines, mask, strict=True))
        if not fenced and _C70_ANCHOR_RE.search(line)
    ]
    if not anchors:
        return _skip(cid, name, "no judge-pilot vocabulary detected")
    if _standalone_na_declared(plan, r"no judge[-\s]?pilot gate\b"):
        return _pass(cid, name, "explicit N/A declared (no judge-pilot gate)")
    if _standalone_na_declared(
        plan, r"harvested pilot sizing is historical or belongs to a different gate"
    ):
        return _pass(cid, name, "explicit N/A declared (historical / cross-gate pilot sizing)")
    evaluated: dict[tuple[int, int], dict] = {}
    best_skip = "pilot vocabulary but no per-arm parse-fail threshold resolvable in any window"
    for a in anchors:
        lo = max(0, a - _C70_WINDOW_LINES)
        hi = min(len(lines), a + _C70_WINDOW_LINES + 1)
        kept = [
            lines[i]
            for i in range(lo, hi)
            if not mask[i] and not _C70_SUPERSEDED_RE.search(lines[i])
        ]
        win = "\n".join(kept)
        th = _C70_THRESH_RE.search(win)
        if not th:
            continue  # -> S3 when no window resolves further
        declared = "allow_subresolution_pilot" in win
        if not _C70_PER_ARM_TOK_RE.search(win):
            best_skip = "parse-fail threshold without a per-arm token (aggregate gate?)"  # S5
            continue
        pct = Fraction(th.group(1))
        if pct <= 0 or pct >= 100:
            best_skip = "degenerate threshold percentage"  # S9
            continue
        required = math.floor(Fraction(1) / (pct / 100)) + 1
        budget = _c70_resolve_budget(win)
        if isinstance(budget, str):
            best_skip = budget  # S4/S6/S7/S8/S10/S11
            continue
        per_arm, n_arms, src = budget
        e = evaluated.setdefault(
            (per_arm, required),
            {"line": a + 1, "src": src, "pct": th.group(1), "n_arms": n_arms, "declared": False},
        )
        e["declared"] = e["declared"] or declared
    if evaluated:
        offenders = [
            (pa, req, e) for (pa, req), e in evaluated.items() if pa < req and not e["declared"]
        ]
        if offenders:
            offenders.sort(key=lambda t: t[1] - t[0], reverse=True)  # worst deficit leads
            parts = []
            for pa, req, e in offenders[:2]:
                sug = (
                    f"suggested >= {req} draws/arm"
                    if e["n_arms"] is None
                    else (
                        f"suggested budget >= {req * e['n_arms']} draws for this rubric "
                        f"slice ({req}/arm x {e['n_arms']} arms)"
                    )
                )
                parts.append(
                    f"anchor line {e['line']}: per_arm={pa} < required={req} at "
                    f"{e['pct']}% ({e['src']}) — {sug}"
                )
            shown = "; ".join(parts) + ("; …" if len(offenders) > 2 else "")
            return _warn(
                cid,
                name,
                f"{shown} — the judge-pilot gate is unsatisfiable by arithmetic: "
                "required = floor(1/threshold) + 1 under exact-fraction arithmetic "
                "(rule 26(b)'s strict '< threshold' verdict — one draw below the floor, "
                "a single parse failure already reads AT the threshold). "
                "eval.judge_pilot.judge_pilot_gate refuses such configs at config time "
                "before any API spend; allow_subresolution_pilot=True is its documented "
                "downgrade (declare it on the gate line to mark the tuple deliberate). "
                "Or declare 'N/A — no judge-pilot gate' / 'N/A — harvested pilot sizing "
                "is historical or belongs to a different gate' on its own line, "
                "unwrapped (no backticks/quotes)",
            )
        parts = [
            (f"{pa}>={req}" if pa >= req else f"{pa}<{req} declared allow_subresolution_pilot")
            + f" ({e['src']}, {e['pct']}% per arm, anchor line {e['line']})"
            for (pa, req), e in evaluated.items()
        ]
        return _pass(cid, name, "judge-pilot resolution satisfied: " + "; ".join(parts))
    return _skip(cid, name, best_skip)


# ─── Driver ────────────────────────────────────────────────────────────────

CHECKS = [
    check_source_grounding,
    check_measurement_validity,
    check_data_tier,
    check_contrastive_negatives,
    check_gpu_hours,
    check_reuse_fitness,
    check_replication_fidelity,
    check_success_kill,
    check_conditions_seeds,
    check_marker_recipe,
    check_dryrun_test_coverage,
    check_battery_multiplier,
    check_empirical_gate_attainability,
    check_hypothesis_branch_coherence,
    check_failloud_test_coverage,
    check_reference_headline_distinction,
    check_causal_claim_scope,
    check_paired_contrast_source_coverage,
    check_ood_folds,
    check_verdict_lattice_coherence,
    check_grep_arity_gate,
    check_cross_section_param_consistency,
    check_resume_provenance,
    check_html_entities_in_commands,
    check_gpu_basis_routed_machine,
    check_capture_intent_hbm,
    check_precedent_band_coherence,
    check_fence_conditional_phase,
    check_realized_keys,
    check_skillmd_prose_pin,
    check_fit_basis_grounding,
    check_ladder_retention,
    check_ratchet_headroom,
    check_pinned_revision_reuse,
    check_numeric_containment,
    check_noflags_bundling_claim,
    check_exit0_repo_wide_baseline,
    check_off_pod_phase_declaration,
    check_regression_anchor_executed,
    check_commit_sha_resolves,
    check_sentinel_lane,
    check_committed_paths_not_gitignored,
    check_change_dv_base_predictor_companion,
    check_dispatch_cmd_cli_parse,
    check_wall_cell_parseable,
    check_basis_booked_arithmetic,
    check_authorized_stub_block,
    check_plan_wall_vs_slurm_time_bin,
    check_edited_literal_pin_tests,
    check_fanout_ram_floor,
    check_judged_dv_api_refusal,
    check_workload_cmd_lane_env,
    check_inherited_rowcount_default,
    check_staging_mount_binding,
    check_fanout_prefix_staging,
    check_fanout_pod_name_collision,
    check_gpu_hours_token_conflict,
    check_slurm_mem_coverage,
    check_declared_width_vs_launch,
    check_exactness_grain,
    check_smoke_fixture_size,
    check_smoke_producer_coverage,
    check_retest_kappa_temp0,
    check_margin_baseline_ceiling,
    check_regen_headroom,
    check_pilot_resolution,
]


def verify_plan_text(raw: str, *, kind: str, source: str = "") -> tuple[bool, list[CheckResult]]:
    """Run every plan check on ``raw`` plan text under ``kind``.

    Check 0 (plan-nonstub) short-circuits the chain on FAIL — a stub plan
    would otherwise cascade into a dozen "<block> missing" errors that bury
    the actual root cause (a broken handoff). Returns
    ``(overall, results)``; WARN and SKIP both leave ``passed=True``.
    """
    del source  # reserved for symmetry with verify_task_body.verify_text
    stub = check_plan_nonstub(raw)
    if not stub.passed:
        return False, [stub]
    results = [stub] + [chk(raw, kind) for chk in CHECKS]
    overall = all(r.passed for r in results)
    return overall, results


def _newest_plan_version(folder: Path) -> Path:
    """Newest ``plans/v{K}.md`` by NUMERIC sort (``v10`` > ``v9``) — never
    the ``plan.md`` symlink (follow-up rounds re-point it; incident #597)."""
    versions: list[tuple[int, Path]] = []
    for p in folder.glob("plans/v*.md"):
        m = re.fullmatch(r"v(\d+)\.md", p.name)
        if m:
            versions.append((int(m.group(1)), p))
    if not versions:
        raise FileNotFoundError(f"no plans/v*.md under {folder}")
    versions.sort()
    return versions[-1][1]


def _kind_from_body(folder: Path) -> str:
    """``kind`` from ``body.md`` frontmatter; missing → ``experiment``
    (the strictest — the /issue Step 0b gate guarantees presence anyway)."""
    body_path = folder / "body.md"
    if not body_path.exists():
        return "experiment"
    fm, _ = split_frontmatter(body_path.read_text())
    return str(fm.get("kind") or "experiment")


def _compose_amendment_text(folder: Path, newest: Path) -> tuple[str, Path | None]:
    """Compose an AMENDMENT-SHAPED newest plan version with its base for
    CHECKING purposes (#2255) — never a rendered document anyone reads as
    "the plan".

    Walks DOWN from ``newest`` while the current version is amendment-shaped
    (``task_workflow.is_amendment_shaped``) w.r.t. its own next-lower
    version; the base is the first non-amendment-shaped version. Returns
    ``(newest_text, None)`` unchanged when the newest version is not
    amendment-shaped (the byte-identical non-amendment path). Composed text
    = newest + any intermediate amendments (descending) + base, joined by
    inert HTML-comment separators — amendment FIRST so c40's first-heading
    read matches the persisted ``v{K}.md`` filename.
    """
    from explore_persona_space.task_workflow import is_amendment_shaped  # local import

    versions: list[tuple[int, Path]] = []
    for p in folder.glob("plans/v*.md"):
        m = re.fullmatch(r"v(\d+)\.md", p.name)
        if m:
            versions.append((int(m.group(1)), p))
    versions.sort()
    order = [p for _, p in versions]
    idx = order.index(newest)
    chain: list[Path] = []  # amendment chain, newest first
    while idx > 0 and is_amendment_shaped(order[idx].read_text(), order[idx - 1].stat().st_size):
        chain.append(order[idx])
        idx -= 1
    if not chain:
        return newest.read_text(), None
    base = order[idx]
    pieces: list[str] = []
    for p in [*chain, base]:
        if pieces:
            pieces.append(f"\n\n<!-- verify_plan amendment composition: {p.name} follows -->\n\n")
        pieces.append(p.read_text())
    return "".join(pieces), base


def _load_plan_for_issue(number: int) -> tuple[str, Path, str, Path | None]:
    """Resolve (plan_text, plan_path, kind, base_path) for a task number via
    the canonical resolver — never hand-built ``tasks/`` paths. When the
    newest version is amendment-shaped (#2255) the returned text is the
    amendment COMPOSED with its base (``_compose_amendment_text``) and
    ``base_path`` names the base version; ``base_path`` is ``None`` (raw
    newest-version text, byte-identical to the pre-#2255 read) otherwise."""
    from explore_persona_space.task_workflow import find_task_path  # local import

    folder = find_task_path(number)
    plan_path = _newest_plan_version(folder)
    text, base_path = _compose_amendment_text(folder, plan_path)
    return text, plan_path, _kind_from_body(folder), base_path


def _json_payload(
    *, source: str, issue: int | None, kind: str, overall: bool, results: list[CheckResult]
) -> dict:
    return {
        "source": source,
        "issue": issue,
        "kind": kind,
        "overall": "PASS" if overall else "FAIL",
        "n_fail": sum(1 for r in results if r.status == "FAIL"),
        "n_warn": sum(1 for r in results if r.status == "WARN"),
        "n_skip": sum(1 for r in results if r.status == "SKIP"),
        "checks": [
            {"id": r.id, "name": r.name, "status": r.status, "detail": r.detail} for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--issue", type=int, help="task number to verify (newest plans/v{K}.md)")
    grp.add_argument("--plan-file", help="path to a standalone plan .md to verify")
    parser.add_argument(
        "--kind",
        choices=VALID_KINDS,
        default=None,
        help="task kind (file mode only; default: experiment, the strictest; "
        "ignored in --issue mode, which reads body.md frontmatter)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report instead of text")
    args = parser.parse_args()

    issue: int | None = None
    base_path: Path | None = None
    if args.issue is not None:
        if args.kind is not None:
            print(
                "verify_plan: --kind is ignored in --issue mode (kind is read from "
                "body.md frontmatter)",
                file=sys.stderr,
            )
        try:
            raw, plan_path, kind, base_path = _load_plan_for_issue(args.issue)
        except FileNotFoundError as e:
            print(f"verify_plan: {e}", file=sys.stderr)
            return 2
        source = str(plan_path)
        if base_path is not None:
            source = f"{plan_path} (+ {base_path.name} via amendment composition)"
        issue = args.issue
    else:
        plan_path = Path(args.plan_file)
        try:
            raw = plan_path.read_text()
        except OSError as e:
            print(f"verify_plan: {e}", file=sys.stderr)
            return 2
        source = args.plan_file
        kind = args.kind or "experiment"

    overall, results = verify_plan_text(raw, kind=kind, source=source)

    # Check 23 (goal currency) needs task context (body.md + events.jsonl),
    # so it runs OUTSIDE verify_plan_text(): appended here in --issue mode,
    # rendered SKIP in --plan-file mode.
    if issue is not None:
        folder = plan_path.parent.parent  # tasks/<status>/<N>/plans/vK.md -> task folder
        mtime = datetime.fromtimestamp(plan_path.stat().st_mtime, tz=UTC)
        cur, sup = _goal_history_for_plan(folder, mtime)
        results.append(check_goal_currency(raw, current_goal=cur, superseded=sup))
    else:
        results.append(
            _skip(
                "c23_goal_currency",
                "plan head not drafted against a superseded Goal",
                "no task context (--plan-file mode; goal history requires --issue)",
            )
        )
    # Check 62 (backend pin-claim vs frontmatter, #2276) also needs task
    # context (body.md frontmatter), so it runs OUTSIDE verify_plan_text():
    # appended here in --issue mode, rendered SKIP in --plan-file mode (the
    # c23 pattern).
    if issue is not None:
        folder = plan_path.parent.parent
        fm_backend: str | None = None
        body_path = folder / "body.md"
        if body_path.exists():
            fm, _ = split_frontmatter(body_path.read_text())
            raw_backend = fm.get("backend")
            if raw_backend is not None and str(raw_backend).strip():
                fm_backend = str(raw_backend).strip()
        results.append(check_backend_pin_claim(raw, frontmatter_backend=fm_backend))
    else:
        results.append(
            _skip(
                "c62_backend_pin_claim",
                "§9 backend pin-claim matches body.md frontmatter",
                "no task context (--plan-file mode; frontmatter reconciliation requires --issue)",
            )
        )
    # Check 40 (header version label vs persisted filename) also runs outside
    # verify_plan_text() — it needs plan_path, defined in BOTH modes.
    results.append(check_header_version_vs_filename(raw, plan_path=plan_path))
    # Check 60 (amendment composition disclosure, #2255) also runs outside
    # verify_plan_text() — appended ONLY when --issue mode composed an
    # amendment-shaped newest version with its base; a not-composed run
    # (and --plan-file mode) emits NO c60 row at all, keeping non-amendment
    # output byte-identical.
    if issue is not None and base_path is not None:
        results.append(
            _warn(
                "c60_amendment_composition",
                "amendment-shaped newest version composed with its base",
                f"{plan_path.name} is a thin AMENDMENT of {base_path.name} "
                f"(task_workflow.is_amendment_shaped, #2255); every check above ran "
                f"against the COMPOSED text (amendment first, then {base_path.name}) — "
                f"NOT against {plan_path.name} alone. plans/plan.md points at a PARTIAL "
                f"document: subagent briefs must hand BOTH {plan_path.name} AND "
                f"{base_path.name}, and the Step-2c GPU-hours read resolves `<X>` from "
                f"the base version's `Estimated GPU-hours (total):` line when the "
                f"amendment restates none",
            )
        )
    overall = all(r.passed for r in results)

    if args.json:
        print(
            json.dumps(
                _json_payload(
                    source=source, issue=issue, kind=kind, overall=overall, results=results
                ),
                indent=2,
            )
        )
        return 0 if overall else 1

    print(f"verify_plan — {source} (kind: {kind})")
    for r in results:
        print(r.render())
    print()
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    if overall:
        print(f"OVERALL: PASS ({n_warn} WARN, {n_skip} SKIP)")
        return 0
    n_fail = sum(1 for r in results if r.status == "FAIL")
    print(f"OVERALL: FAIL ({n_fail} of {len(results)} checks failed; {n_warn} WARN, {n_skip} SKIP)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
