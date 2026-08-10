#!/usr/bin/env python
"""Issue #1941 — diagnose/repair the #1773 `functional_role` SAE-judge axis.

Phases (plan v3 §4; all reuse the #1773 machinery — `issue1773_common` (CM)
constants/builders/estimators + `issue1773_describe_axes._dispatch` (the
Batch-API path with raw-text retention)):

  --phase sample-diag   P0: eligibility census over the 16,384-feature
                        restricted frame, stratified sample (600 uniform +
                        100 output_promoting-majority + 100 mixed-majority),
                        kappa-reproduction gate (recompute the published
                        full-dict functional_role kappa from the labels shards
                        to within ±0.001), C0 same-sample manipulation check
                        (±0.12), content-drop taxonomy over retained raw judge
                        text (truncation-signature / out-of-set (verbatim) /
                        refusal / other-malformed), per-label kappa_j,
                        draw-pair confusion, mixed-dump signature, PABAK +
                        chance-agreement decomposition, unanimity covariates.
  --phase arms --wave 1 P1: the four wave-1 arms (c600/n600/r600/b600 — plan
                        §5), 800 features x 5 draws each, max_tokens=600,
                        fresh per-arm checkpoint dirs (llm-judging rule 22/23:
                        c600 prompts are byte-identical to production's, so a
                        shared/production checkpoint dir would replay
                        production responses), threshold_base=1 (force-batch)
                        on every full dispatch. --render-only writes golden
                        prompts + runs the byte-identity asserts; --smoke runs
                        the 5-item live forced-batch smoke; --full dispatches.
  --phase analyze       P2: per-arm kappa + prevalence + raw agreement + drop
                        report; paired feature-bootstrap (B=2,000) CIs for
                        every kappa and contrast (CI-LB95 = the 2.5th
                        percentile of the paired bootstrap distribution — the
                        TWO-SIDED 95% CI lower bound, the registered
                        threshold convention); per-label kappa_j + draw-pair
                        confusion per arm (uniform / oversample separately —
                        the oversample NEVER enters headline kappa bars or
                        contrasts); side_ratio convergent read; figures.
  --phase wave2         P3 (conditional): revised-rubric arms d600/bd600 on
                        the same 800 features; rubric frozen + sha16-hashed at
                        dispatch. Refuses unless the registered §3 trigger
                        holds (override with --force), and refuses a
                        pre-analyze (arms-format `_partial`) kappa_by_arm.json
                        (run --phase analyze first). After a --full dispatch
                        completes, the phase AUTO-RUNS --phase analyze (the
                        RE-ANALYZE step), folding d600/bd600 into the
                        analyze-format summary + contrasts — so the registered
                        §10 sequence (analyze -> wave2 --full -> decide-mark)
                        can never hand decide-mark an arms-format summary.
  --phase decide-mark   P4: apply the registered verdict lattice mechanically
                        -> decision.json; with --apply-marking additionally
                        patch eval_results/issue_1773/feature_table_v1.jsonl
                        ADDITIVELY (per-row `axis_usability` field), write the
                        usability sidecar + AXIS_USABILITY.json, and check
                        `CM.AXIS_USABILITY` matches the verdict string.
                        REFUSES (fail-loud) unless arms/kappa_by_arm.json is
                        the ANALYZE-format output covering every labeled arm
                        and contrasts.json carries each arm's `_vs_c0`
                        contrast — an arms-format/partial summary reads
                        kappa_uniform -> None and would silently RETIRE a
                        qualifying arm. On refusal: run `--phase analyze`.

Spend guards: only --smoke (5 live calls ~ $0.02) and --full dispatch API
calls; every other mode is $0. Out-root guards, per leg: a BOUNDED
sample-diag (--evidence-shard-limit / --raw-shard-limit) REFUSES the default
--out-root (a partial census must never write the production sample manifest;
per-leg out-roots, .claude/rules/crash-fix-rounds.md), and --full REFUSES a
sample manifest built from bounded inputs. The arms --render-only / --smoke /
--limit legs ARE allowed at the default out-root (the plan §10 registered
smoke chain runs them there): they write only uniquely-named artifacts
(golden_prompts/*.txt, smoke_report.json) that no full-run artifact path
collides with.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402
from issue1773_describe_axes import _classify_error, _dispatch, _write_raw  # noqa: E402

TASK = 1941
AXIS = "functional_role"
FR_CATS = CM.AXES[AXIS]  # ("input_side", "output_promoting", "mixed")

FULLDICT_ROOT = Path(
    os.environ.get("EPM_1941_FULLDICT_ROOT", "/mnt/eps-data/thomasjiralerspong/issue1773_fulldict")
)
WORK_ROOT = Path(os.environ.get("EPM_1941_WORK", "/mnt/eps-data/thomasjiralerspong/issue1941_fr"))
OUT_EVAL_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_1941"
OUT_FIGS_DEFAULT = PROJECT_ROOT / "figures" / "issue_1941"
PHASE0_NPZ = PROJECT_ROOT / "eval_results" / "issue_1773" / "phase0" / "phase0_arrays.npz"
FEATURE_TABLE = PROJECT_ROOT / "eval_results" / "issue_1773" / "feature_table_v1.jsonl"
HF_PREFIX_1941 = "issue1941_fr_diag"

# ── design constants (plan §3/§4/§11) ────────────────────────────────────────
N_UNIFORM = 600
N_OVERSAMPLE = {"output_promoting": 100, "mixed": 100}
MIN_EX_POS = 5  # eligibility floor on activating examples (plan §4 P0.1)
ELIGIBILITY_FLOOR = 0.95  # census report-and-adapt threshold (never a hard gate)
# Banked #1941 wave-1 instrument (the c600/n600/r600/b600 arms all ran at 600).
# The cited #1916 600-token JSON floor is SUPERSEDED — llm-judging rule 23
# (2026-08-02) sets 1024 single-rationale / 2048 multi-field. JUSTIFIED DEVIATION
# (#2063): kept at 600 for parity with the completed diagnostic wave; fresh
# waves owe the current floor.
ARMS_MAX_TOKENS = 600
N_DRAWS = CM.N_DRAWS  # 5 — draws-per-item is part of the estimator being diagnosed

# Wave-1 arms: the ONLY per-arm variable is the evidence set (plan §5).
ARM_SEES: dict[str, tuple[str, ...]] = {
    "c600": ("EX_POS", "OUT", "DESC"),  # budget-only control == production evidence
    "n600": ("EX_POS", "EX_NEG", "OUT", "DESC"),  # + non-activating examples
    "r600": ("EX_POS", "NEAR", "OUT", "DESC"),  # + neighbour features
    "b600": ("EX_POS", "EX_NEG", "NEAR", "OUT", "DESC"),  # + both
}
# Wave-2 arms: rubric revision (v2) is the manipulated variable.
WAVE2_SEES: dict[str, tuple[str, ...]] = {
    "d600": ("EX_POS", "OUT", "DESC"),
    "bd600": ("EX_POS", "EX_NEG", "NEAR", "OUT", "DESC"),
}
WAVE1_ARMS = tuple(ARM_SEES)
WAVE2_ARMS = tuple(WAVE2_SEES)
# Adopted-change preference order on REPAIR: smallest instrument change, then
# token cost (plan §3).
ARM_PREFERENCE = ("c600", "n600", "r600", "b600", "d600", "bd600")

# Registered verdict-lattice thresholds (plan §3; Sources in plan §11).
KAPPA_REPAIR_MIN = CM.LATTICE_KAPPA_MIN  # 0.60 (#1773 LATTICE_KAPPA_MIN)
DELTA_CI_LB95_MIN = 0.15  # registered analysis threshold (plan §11)
DROP_RATE_MAX = 0.015  # 1.5x the worst sibling (1.01%)
KAPPA_REPRO_TOL = 1e-3  # P0.2 reproduction gate vs the published kappa
C0_MANIP_BAND = 0.12  # same-sample manipulation check half-width
INSTRUMENT_DRIFT_MAX = 0.25  # |kappa_c600 - kappa_c0| halt bar (plan §7 crit 2)
TRANSPORT_RESIDUAL_MAX = 0.005  # per-arm residual transport bar (plan §7 crit 3)
WAVE2_TRIGGER_KAPPA = 0.45  # wave-2 trigger on kappa_b600
BOOT_B = 2_000  # paired feature-bootstrap resamples (plan §6)
SENSITIVITY_DIVERGENCE = 0.03  # intersection-read divergence caveat bar (plan §3)

SAMPLE_SEED = (CM.SEED, TASK)  # np.random.SeedSequence([17_732_026, 1941])
BOOT_SEED = (CM.SEED, TASK, 2)  # np.random.SeedSequence([17_732_026, 1941, 2])

# The byte-exact tail of the question block rendered by CM.build_axis_user_msg
# (issue1773_common.py — the closing JSON instruction). The wave-2 decision
# procedure is inserted immediately BEFORE this literal; pinned by tests.
QUESTION_TAIL = '\n\nOutput ONLY JSON: {"reasoning": "...", "label": "<one allowed label>"}.'

# ── wave-2 revised rubric (DRAFT — the experimental manipulation; the text is
# frozen + sha16-hashed into wave2_rubric.json at dispatch time). Shaped by
# llm-judging rules 6/7/25: anchored, reason-then-label, confusable-neighbour
# tie-break AWAY from the `mixed` dumping ground. Window geometry supports the
# ~16-token continuation check ([peak-15, peak+16] windows). ──────────────────
FR_RUBRIC_V2_DEFS: dict[str, str] = {
    "input_side": (
        "the marked tokens share a clear input/context property (what the feature READS); "
        "promoted output tokens are absent from the text after the marked tokens, or appear "
        "only incidentally"
    ),
    "output_promoting": (
        "promoted output tokens appear within the ~16 tokens AFTER the <<marked>> token in a "
        "clear majority of examples, and the promoted-token list coheres with the marked "
        "contexts (the feature DRIVES those outputs)"
    ),
    "mixed": (
        "ONLY when there is clear evidence of BOTH: a shared input/context property AND a "
        "majority continuation match with the promoted tokens"
    ),
}
FR_DECISION_PROCEDURE = (
    "Decide in two steps. STEP 1 - continuation match: for each activating example, check "
    "whether any promoted output token appears within the ~16 tokens AFTER the <<marked>> "
    "token. STEP 2 - label: `output_promoting` = continuation match in a clear majority of "
    "examples AND the promoted-token list coheres with the marked contexts; `input_side` = "
    "the marked tokens share a clear input/context property and continuation matches are "
    "absent or incidental; `mixed` = ONLY when there is clear evidence of BOTH - a shared "
    "input property AND majority continuation match. If uncertain between `mixed` and "
    "another label, choose the other label."
)


def wave2_rubric_sha16() -> str:
    """sha16 of the frozen wave-2 rubric (defs + decision procedure)."""
    blob = json.dumps(FR_RUBRIC_V2_DEFS, sort_keys=True) + "||" + FR_DECISION_PROCEDURE
    return CM.sha16(blob)


REFUSAL_RE = re.compile(
    r"(?i)(i can(?:no|')t|i cannot|i'?m sorry|i am sorry|i apologize|i won'?t"
    r"|unable to (?:comply|help|classify|assist)|refus)"
)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── custom ids (plan §4 P1) ──────────────────────────────────────────────────
ARM_CID_RE = re.compile(rf"^f(\d+)-{AXIS}-([a-z][a-z0-9]*)-d(\d)$")
CID_CHARSET_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def arm_custom_id(feat_id: int, arm: str, draw_idx: int) -> str:
    """Batch custom_id `f{fid}-functional_role-{arm}-d{d}`; <=53 chars (the
    64-char API cap minus the 11-char encoder suffix — the #1415 budget) and
    charset [a-zA-Z0-9_-]. Longest: 'f131071-functional_role-bd600-d4' = 32."""
    cid = f"f{feat_id}-{AXIS}-{arm}-d{draw_idx}"
    assert len(cid) <= 53 and CID_CHARSET_RE.match(cid), cid
    return cid


def parse_arm_custom_id(cid: str) -> tuple[int, str, int]:
    """Inverse of arm_custom_id -> (feat_id, arm, draw_idx)."""
    m = ARM_CID_RE.match(cid)
    assert m, f"unparseable arm custom_id: {cid!r}"
    return int(m.group(1)), m.group(2), int(m.group(3))


# ── prompt rendering (wave 1: CM builder + `sees`; wave 2: revised rubric) ───


def render_arm_msg(arm: str, packet: dict, description: str | None, draw_idx: int) -> str:
    """Render one arm's user message. Wave-1 arms differ from production ONLY
    by the evidence `sees` tuple; wave-2 arms additionally swap the
    functional_role definitions and insert the decision procedure."""
    if arm in ARM_SEES:
        return CM.build_axis_user_msg(AXIS, packet, description, draw_idx, sees=ARM_SEES[arm])
    if arm in WAVE2_SEES:
        return build_axis_user_msg_v2(packet, description, draw_idx, sees=WAVE2_SEES[arm])
    raise ValueError(f"unknown arm {arm!r}")


def build_axis_user_msg_v2(
    packet: dict, description: str | None, draw_idx: int, *, sees: tuple[str, ...]
) -> str:
    """Wave-2 revised-rubric message: the CM builder with the functional_role
    definitions swapped to FR_RUBRIC_V2_DEFS and the decision procedure
    inserted immediately before the question block's JSON tail. Label SET and
    per-(feat,draw) permutation are UNCHANGED (TASK_ID=1773 salt)."""
    orig = CM.AXIS_DEFINITIONS[AXIS]
    CM.AXIS_DEFINITIONS[AXIS] = FR_RUBRIC_V2_DEFS
    try:
        msg = CM.build_axis_user_msg(AXIS, packet, description, draw_idx, sees=sees)
    finally:
        CM.AXIS_DEFINITIONS[AXIS] = orig
    n = msg.count(QUESTION_TAIL)
    assert n == 1, f"question tail literal not unique in rendered message (count={n})"
    return msg.replace(QUESTION_TAIL, "\n\n" + FR_DECISION_PROCEDURE + QUESTION_TAIL, 1)


def _ex_pos_block(packet: dict) -> str:
    return CM.render_windows_block(
        "Activating examples (strongest token marked <<...>>)",
        packet["ex_pos"][: CM.AXIS_EX_POS_N],
        with_marks=True,
    )


def _ex_neg_block(packet: dict) -> str:
    return CM.render_windows_block(
        "Non-activating examples", packet["ex_neg"][: CM.AXIS_EX_NEG_N], with_marks=False
    )


def _near_block(packet: dict) -> str:
    return CM.render_windows_block(
        "NEAR-MISS examples (a similar but DIFFERENT feature activates here)",
        packet["near"],
        with_marks=True,
    )


def assert_arm_prompt_identity(packet: dict, description: str | None, draw_idx: int) -> dict:
    """Golden-prompt identity checks (plan §4 P1 smoke (a)):

    - c600's message is BYTE-IDENTICAL to the production builder's output for
      the same (feat, draw);
    - n600/r600/b600 differ from c600 ONLY by the inserted EX_NEG / NEAR
      block(s), spliced after the EX_POS block (the builder's fixed render
      order).

    Raises AssertionError on any mismatch; returns a small report dict."""
    prod = CM.build_axis_user_msg(AXIS, packet, description, draw_idx)
    c = render_arm_msg("c600", packet, description, draw_idx)
    assert c == prod, "c600 is not byte-identical to the production prompt"
    pos = _ex_pos_block(packet)
    assert prod.count(pos) == 1, "EX_POS block not uniquely locatable in the prompt"
    neg = _ex_neg_block(packet)
    has_near = bool(packet.get("near"))
    near = _near_block(packet) if has_near else None

    n_msg = render_arm_msg("n600", packet, description, draw_idx)
    assert n_msg == prod.replace(pos, pos + "\n\n" + neg, 1), "n600 != c600 + EX_NEG block"

    r_msg = render_arm_msg("r600", packet, description, draw_idx)
    r_expected = prod.replace(pos, pos + "\n\n" + near, 1) if has_near else prod
    assert r_msg == r_expected, "r600 != c600 + NEAR block"

    b_msg = render_arm_msg("b600", packet, description, draw_idx)
    if has_near:
        b_expected = prod.replace(pos, pos + "\n\n" + neg + "\n\n" + near, 1)
    else:
        b_expected = prod.replace(pos, pos + "\n\n" + neg, 1)
    assert b_msg == b_expected, "b600 != c600 + EX_NEG + NEAR blocks"
    return {"feat_id": int(packet["feat_id"]), "draw": draw_idx, "has_near": has_near}


# ── estimator helpers (all mirror CM.fleiss_kappa_varying_n semantics) ───────


def votes_to_counts(
    votes: list[list[str]], categories: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-item category-count matrix M (N, k) + surviving-draw counts n (N,)."""
    idx = {c: i for i, c in enumerate(categories)}
    M = np.zeros((len(votes), len(categories)), dtype=np.int64)
    for i, labs in enumerate(votes):
        for lab in labs:
            M[i, idx[lab]] += 1
    return M, M.sum(axis=1)


def kappa_from_counts(M: np.ndarray, n: np.ndarray, categories: tuple[str, ...]) -> dict:
    """Varying-n Fleiss kappa from a count matrix — numerically identical to
    CM.fleiss_kappa_varying_n on the same votes (pinned by a test)."""
    used = n >= 2
    n_excluded = int((~used).sum())
    if not used.any():
        return {
            "kappa": float("nan"),
            "n_items": 0,
            "n_excluded_lt2": n_excluded,
            "prevalence": {},
            "raw_agreement": float("nan"),
        }
    Mu, nu = M[used], n[used]
    a_i = (np.sum(Mu * Mu, axis=1) - nu) / (nu * (nu - 1))
    p_bar = float(a_i.mean())
    tot = Mu.sum(axis=0)
    tot_n = int(nu.sum())
    p_j = tot / tot_n
    p_e = float(np.sum(p_j * p_j))
    kappa = float("nan") if math.isclose(p_e, 1.0) else (p_bar - p_e) / (1.0 - p_e)
    return {
        "kappa": float(kappa),
        "n_items": int(used.sum()),
        "n_excluded_lt2": n_excluded,
        "prevalence": {c: float(p_j[i]) for i, c in enumerate(categories)},
        "raw_agreement": p_bar,
        "p_e": p_e,
        "pabak_multi": pabak_multi(p_bar, len(categories)),
    }


def pabak_multi(p_bar: float, k: int) -> float:
    """Multi-category PABAK (Byrt et al. 1993): (k*p_bar - 1)/(k - 1) — the
    prevalence-adjusted DESCRIPTIVE companion (never the decision metric)."""
    return float((k * p_bar - 1.0) / (k - 1.0))


def per_category_kappa(
    M: np.ndarray, n: np.ndarray, categories: tuple[str, ...]
) -> dict[str, float]:
    """Per-category Fleiss kappa_j (Fleiss 1971 decomposition, varying-n
    generalization): kappa_j = 1 - sum_i n_ij(n_i - n_ij) / (p_j q_j sum_i
    n_i(n_i - 1)), with p_j the pooled prevalence over surviving draws."""
    used = n >= 2
    Mu, nu = M[used], n[used]
    if not len(Mu):
        return {c: float("nan") for c in categories}
    denom_pairs = float(np.sum(nu * (nu - 1)))
    p_j = Mu.sum(axis=0) / float(nu.sum())
    out: dict[str, float] = {}
    for j, c in enumerate(categories):
        q = p_j[j] * (1.0 - p_j[j])
        if q <= 0 or denom_pairs <= 0:
            out[c] = float("nan")
            continue
        disagree_j = float(np.sum(Mu[:, j] * (nu - Mu[:, j])))
        out[c] = float(1.0 - disagree_j / (denom_pairs * q))
    return out


def draw_pair_confusion(M: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Symmetric (k, k) unordered draw-pair co-occurrence matrix over all
    C(n_i, 2) surviving-draw pairs per item (items with n_i >= 2). Diagonal =
    same-label pairs; each off-diagonal unordered pair count appears at BOTH
    [a, b] and [b, a] (read one triangle for totals)."""
    used = n >= 2
    Mu = M[used].astype(np.float64)
    C = Mu.T @ Mu  # sum_i n_ia * n_ib
    diag = np.array([np.sum(Mu[:, j] * (Mu[:, j] - 1)) / 2.0 for j in range(M.shape[1])])
    np.fill_diagonal(C, diag)
    return C


def mixed_dump_signature(
    C: np.ndarray, prevalence: dict[str, float], categories: tuple[str, ...], target: str = "mixed"
) -> dict:
    """Plan §3 mixed-dump signature: share of DISAGREEING draw-pairs involving
    `mixed` > 2x the share expected were disagreements distributed
    proportionally to label prevalence (expected = 2 p_m (1-p_m) / (1 - sum
    p_j^2))."""
    j = categories.index(target)
    upper = np.triu(C, k=1)
    disagree_total = float(upper.sum())
    involving = float(upper[j, :].sum() + upper[:, j].sum())
    observed = involving / disagree_total if disagree_total > 0 else float("nan")
    p = np.array([prevalence.get(c, 0.0) for c in categories], dtype=np.float64)
    denom = 1.0 - float(np.sum(p * p))
    expected = (2.0 * p[j] * (1.0 - p[j]) / denom) if denom > 0 else float("nan")
    ratio = observed / expected if expected and not math.isnan(expected) else float("nan")
    return {
        "observed_share": observed,
        "expected_share_prevalence_proportional": expected,
        "ratio": ratio,
        "holds": bool(not math.isnan(ratio) and ratio > 2.0),
        "n_disagree_pairs": disagree_total,
    }


def bootstrap_kappa_draws(M: np.ndarray, n: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Vectorized varying-n Fleiss kappa per bootstrap resample.

    M (S, k) / n (S,) are aligned to the FIXED uniform feature list; idx is the
    (B, S) resample index matrix (shared across arms => paired contrasts).
    Items with < 2 surviving draws in THIS arm are excluded per resample
    (estimator parity with CM.fleiss_kappa_varying_n). Returns (B,) kappas.
    """
    assert idx.ndim == 2 and M.shape[0] == n.shape[0], (M.shape, n.shape, idx.shape)
    valid = n >= 2
    a_i = np.zeros(len(n), dtype=np.float64)
    nz = n[valid].astype(np.float64)
    a_i[valid] = (np.sum(M[valid] * M[valid], axis=1) - nz) / (nz * (nz - 1.0))
    V = valid[idx]  # (B, S)
    nv = V.sum(axis=1).astype(np.float64)  # valid items per draw
    p_bar = np.where(nv > 0, np.sum(np.where(V, a_i[idx], 0.0), axis=1) / np.maximum(nv, 1), np.nan)
    Msel = np.where(V[..., None], M[idx], 0)  # (B, S, k)
    tot = Msel.sum(axis=1).astype(np.float64)  # (B, k)
    tot_n = tot.sum(axis=1)
    p_j = tot / np.maximum(tot_n, 1.0)[:, None]
    p_e = np.sum(p_j * p_j, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kap = (p_bar - p_e) / (1.0 - p_e)
    kap[np.isclose(p_e, 1.0) | (nv == 0)] = np.nan
    return kap


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC of `pos` vs `neg` values (ties get half credit)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


# ── streaming loaders (full-dict artifacts; per-shard checkpointed scans) ────


def scan_with_partials(
    files: list[Path], partial_path: Path, scan_one, phase: str, schema: str = "v1"
) -> list:
    """Per-shard checkpointed scan (code-style intra-phase T2 persistence):
    appends one JSON row per shard to `partial_path` (atomic single-line
    appends) and resumes by shard name; emits one progress line per shard.
    `schema` is the output-affecting regime key (#722 r3 resume rule): rows
    written under a DIFFERENT schema are ignored and recomputed, never
    silently reused across a classifier/census code change."""
    done: dict[str, object] = {}
    if partial_path.exists():
        for row in CM.iter_jsonl(partial_path):
            if row.get("schema") == schema:
                done[row["shard"]] = row["payload"]
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    out: list = []
    t0 = time.time()
    with partial_path.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(files, 1):
            if p.name in done:
                out.append(done[p.name])
                continue
            payload = scan_one(p)
            fh.write(
                json.dumps(
                    {"shard": p.name, "schema": schema, "payload": payload}, ensure_ascii=False
                )
                + "\n"
            )
            fh.flush()
            out.append(payload)
            _log(f"[{phase}] unit {i}/{len(files)} {p.name} elapsed={time.time() - t0:.0f}s")
    return out


def load_frame() -> dict:
    """The 16,384-feature restricted frame + per-feature covariates from
    #1773's phase0_arrays.npz (feat_ids, activity, side_ratio)."""
    z = np.load(PHASE0_NPZ, allow_pickle=False)
    fid = np.asarray(z["feat_ids"], dtype=np.int64)
    return {
        "feat_ids": fid,
        "activity": {int(f): float(a) for f, a in zip(fid, z["activity"])},
        "side_ratio": {int(f): float(s) for f, s in zip(fid, z["side_ratio"])},
    }


def load_fulldict_fr_labels() -> tuple[dict[int, list[str]], dict[int, str]]:
    """Stream the full-dict axis_labels shards -> functional_role
    {fid: labels_surviving} + {fid: majority label} over ALL judged features
    (needed in full for the P0.2 kappa-reproduction gate)."""
    votes: dict[int, list[str]] = {}
    majority: dict[int, str] = {}
    files = sorted((FULLDICT_ROOT / "labels_upload").glob("axis_labels.shard*.jsonl"))
    assert files, f"no axis_labels shards under {FULLDICT_ROOT / 'labels_upload'}"
    for p in files:
        for r in CM.iter_jsonl(p):
            if r.get("axis") != AXIS:
                continue
            fid = int(r["feat_id"])
            votes[fid] = list(r.get("labels_surviving") or [])
            majority[fid] = str(r.get("label"))
    return votes, majority


def load_all_axis_counts() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stream the full-dict labels shards once -> per-axis (M, n) count
    matrices for the PABAK / p_e / per-label decomposition (plan §4 P0.4)."""
    per_axis: dict[str, list[list[str]]] = {a: [] for a in CM.AXES}
    files = sorted((FULLDICT_ROOT / "labels_upload").glob("axis_labels.shard*.jsonl"))
    for p in files:
        for r in CM.iter_jsonl(p):
            a = r.get("axis")
            if a in per_axis:
                per_axis[a].append(list(r.get("labels_surviving") or []))
    return {a: votes_to_counts(v, CM.AXES[a]) for a, v in per_axis.items()}


def load_descriptions(want: set[int] | None = None) -> dict[int, dict]:
    """{fid: {description, confidence}} from the full-dict descriptions shards
    (restricted to `want` when given)."""
    out: dict[int, dict] = {}
    files = sorted((FULLDICT_ROOT / "labels_upload").glob("descriptions.shard*.jsonl"))
    assert files, f"no descriptions shards under {FULLDICT_ROOT / 'labels_upload'}"
    for p in files:
        for r in CM.iter_jsonl(p):
            fid = int(r["feat_id"])
            if fid < 0 or (want is not None and fid not in want):
                continue
            out[fid] = {
                "description": r.get("description"),
                "confidence": r.get("confidence"),
            }
    return out


def evidence_shard_files(limit: int | None = None) -> list[Path]:
    files = sorted(
        (FULLDICT_ROOT / "evidence" / "evidence_manifests").glob("evidence.shard*.jsonl")
    )
    assert files, f"no evidence shards under {FULLDICT_ROOT / 'evidence'}"
    return files[:limit] if limit else files


def evidence_census(
    frame_set: set[int], partial_path: Path, shard_limit: int | None = None
) -> dict[int, dict]:
    """Per-frame-feature evidence census {fid: {n_ex_pos, n_ex_neg, n_near,
    has_out}} streamed from the full-dict evidence manifests (checkpointed
    per shard)."""

    def scan_one(p: Path) -> dict:
        payload: dict[str, list] = {}
        for r in CM.iter_jsonl(p):
            fid = int(r["feat_id"])
            if fid not in frame_set:
                continue
            payload[str(fid)] = [
                len(r.get("ex_pos") or []),
                len(r.get("ex_neg") or []),
                len(r.get("near") or []),
                int(bool(r.get("out"))),
            ]
        return payload

    payloads = scan_with_partials(
        evidence_shard_files(shard_limit),
        partial_path,
        scan_one,
        "sample-diag:census",
        schema="census-v1",
    )
    census: dict[int, dict] = {}
    for payload in payloads:
        for k, (npos, nneg, nnear, hout) in payload.items():
            census[int(k)] = {
                "n_ex_pos": int(npos),
                "n_ex_neg": int(nneg),
                "n_near": int(nnear),
                "has_out": bool(hout),
            }
    return census


def load_sample_packets(want: set[int]) -> dict[int, dict]:
    """Full evidence packets for the sampled features only (streamed; early
    exit once every wanted feature is found)."""
    packets: dict[int, dict] = {}
    remaining = set(want)
    for p in evidence_shard_files():
        for r in CM.iter_jsonl(p):
            fid = int(r["feat_id"])
            if fid in remaining:
                packets[fid] = r
                remaining.discard(fid)
        if not remaining:
            break
    assert not remaining, f"{len(remaining)} sampled features missing evidence packets"
    return packets


# ── P0.3 content-drop taxonomy ───────────────────────────────────────────────

TAXONOMY_CLASSES = (
    "truncation_signature",
    "out_of_set",
    "refusal",
    "split_second_json_label",
    "other_malformed",
)


def _later_json_label(raw_text: str, axis: str) -> str | None:
    """Scan '{'-anchored objects AFTER the first for a valid in-set label —
    the split-JSON shape the production parser (first-object-only) drops."""
    dec = json.JSONDecoder()
    first_seen = False
    pos = 0
    while True:
        start = raw_text.find("{", pos)
        if start < 0:
            return None
        try:
            obj, end = dec.raw_decode(raw_text, start)
        except (ValueError, json.JSONDecodeError):
            pos = start + 1
            continue
        if first_seen:
            lab = CM.validate_axis_label(obj, axis)
            if lab is not None:
                return lab
        first_seen = True
        pos = end


def classify_drop(parsed: object, raw_text: str, axis: str = AXIS) -> tuple[str, str | None]:
    """Classify one FAILED production parse+validate (plan §4 P0.3).

    parsed = parse_judge_json(raw_text) (None on parse failure). Classes:
      truncation_signature — invalid JSON with a '{' present (unterminated
        object consistent with a hard max_tokens cut);
      out_of_set — valid JSON dict whose `label` string is not in the axis set
        (the invented string is returned verbatim for the tally);
      refusal — no JSON object and refusal-shaped prose;
      split_second_json_label — the first decoded object lacks a valid label
        but a LATER '{'-anchored object carries a valid in-set one (the
        production parser reads only the first object — a recoverable shape
        discovered on the smoke slice; the recovered label rides the tally);
      other_malformed — everything else (scalar JSON, missing label, ...).
    """
    if isinstance(parsed, dict):
        lab = parsed.get("label")
        if isinstance(lab, str):
            return "out_of_set", lab.strip()
        later = _later_json_label(raw_text, axis)
        if later is not None:
            return "split_second_json_label", later
        return "other_malformed", None
    if parsed is not None:
        return "other_malformed", None
    if "{" in raw_text:
        return "truncation_signature", None
    if REFUSAL_RE.search(raw_text):
        return "refusal", None
    return "other_malformed", None


def drop_taxonomy_scan(
    partial_path: Path,
    shard_limit: int | None = None,
    axes: tuple[str, ...] = (AXIS, "abstraction"),
) -> dict:
    """Stream the retained raw judge text (~3.2M rows across ~338 shard files),
    re-run the production parse (`parse_judge_json` -> `CM.validate_axis_label`)
    on every row of the axes of interest, and tally the drop taxonomy.
    `abstraction` rides along as the sibling reference (plan §4 P0.3)."""
    from explore_persona_space.eval.utils import parse_judge_json

    axes_set = set(axes)

    def scan_one(p: Path) -> dict:
        counters = {a: Counter() for a in axes_set}
        oos = {a: Counter() for a in axes_set}
        recovered = {a: Counter() for a in axes_set}
        for r in CM.iter_jsonl(p):
            cid = r.get("custom_id") or ""
            try:
                _fid, axis, _d = CM.parse_axis_custom_id(cid)
            except (ValueError, IndexError):
                continue
            if axis not in axes_set:
                continue
            raw = r.get("raw_text") or ""  # on-disk field is `raw_text` (fact-check R3)
            c = counters[axis]
            c["scanned"] += 1
            parsed = parse_judge_json(raw)
            if CM.validate_axis_label(parsed, axis) is not None:
                c["ok"] += 1
                continue
            cls, detail = classify_drop(parsed, raw, axis)
            c[cls] += 1
            if cls == "out_of_set" and detail is not None:
                oos[axis][detail] += 1
            elif cls == "split_second_json_label" and detail is not None:
                recovered[axis][detail] += 1
        return {
            "counters": {a: dict(counters[a]) for a in axes_set},
            "oos": {a: dict(oos[a]) for a in axes_set},
            "recovered": {a: dict(recovered[a]) for a in axes_set},
        }

    files = sorted((FULLDICT_ROOT / "work" / "judge_raw").glob("axes_raw_g*.shard*.jsonl"))
    assert files, f"no raw judge shards under {FULLDICT_ROOT / 'work' / 'judge_raw'}"
    if shard_limit:
        files = files[:shard_limit]
    payloads = scan_with_partials(
        files, partial_path, scan_one, "sample-diag:taxonomy", schema="taxonomy-v2-split-json"
    )
    counters = {a: Counter() for a in axes_set}
    oos = {a: Counter() for a in axes_set}
    recovered = {a: Counter() for a in axes_set}
    for payload in payloads:
        for a, d in payload["counters"].items():
            counters[a].update(d)
        for a, d in payload["oos"].items():
            oos[a].update(d)
        for a, d in payload.get("recovered", {}).items():
            recovered[a].update(d)
    out: dict = {"n_shards_scanned": len(files), "axes": {}}
    for a in axes:
        c = counters[a]
        failed = sum(c[k] for k in TAXONOMY_CLASSES)
        out["axes"][a] = {
            "scanned": c["scanned"],
            "ok": c["ok"],
            "failed_with_raw_text": failed,
            "taxonomy": {k: c[k] for k in TAXONOMY_CLASSES},
            # invented label strings recorded VERBATIM (top 50 by count)
            "out_of_set_labels_verbatim": dict(oos[a].most_common(50)),
            # labels recovered from a SECOND JSON object (production drops the
            # first-object-only parse; diagnostic refinement, smoke discovery)
            "split_second_json_recovered_labels": dict(recovered[a].most_common(10)),
        }
    return out


def arm_drop_taxonomy_scan(arms: tuple[str, ...] = WAVE1_ARMS) -> dict:
    """Per-ARM drop taxonomy over the wave-1 arms' RETAINED raw judge text
    (`WORK_ROOT/judge_raw/arm_<arm>.shard*.jsonl`, the `_write_raw` output —
    registered concern per-arm-taxonomy-figure). Re-runs the production parse
    (`parse_judge_json` -> `CM.validate_axis_label`) on every retained row and
    classifies failures with `classify_drop` — the SAME classes as the
    production P0.3 scan. Coverage caveat: `_write_raw` retains only rows that
    RETURNED raw text, so API-level content-error dicts are not classifiable
    here and per-arm classified counts are a lower bound on the arm's content
    drops. Returns entries only for arms whose shards exist on disk
    (render-only / smoke legs have none). ~14 MB across 4 arms — seconds."""
    from explore_persona_space.eval.utils import parse_judge_json

    out: dict[str, dict] = {}
    for arm in arms:
        files = sorted((WORK_ROOT / "judge_raw").glob(f"arm_{arm}.shard*.jsonl"))
        if not files:
            continue
        c: Counter = Counter()
        oos: Counter = Counter()
        for p in files:
            for r in CM.iter_jsonl(p):
                raw = r.get("raw_text") or ""
                c["scanned"] += 1
                parsed = parse_judge_json(raw)
                if CM.validate_axis_label(parsed, AXIS) is not None:
                    c["ok"] += 1
                    continue
                cls, detail = classify_drop(parsed, raw, AXIS)
                c[cls] += 1
                if cls == "out_of_set" and detail is not None:
                    oos[detail] += 1
        out[arm] = {
            "scanned": c["scanned"],
            "ok": c["ok"],
            "failed_with_raw_text": sum(c[k] for k in TAXONOMY_CLASSES),
            "taxonomy": {k: c[k] for k in TAXONOMY_CLASSES},
            "out_of_set_labels_verbatim": dict(oos.most_common(20)),
        }
    return out


# ── P0 sample-diag phase ─────────────────────────────────────────────────────


def draw_sample(
    eligible: list[int], majority: dict[int, str], rng: np.random.Generator
) -> dict[str, list[int]]:
    """Deterministic stratified draw (plan §4 P0.1): 600 uniform over the
    sorted eligible set, then 100 output_promoting-majority + 100
    mixed-majority oversamples (majority per the full-dict labels; disjoint
    from the uniform picks and from each other). Draw order fixed: uniform,
    output_promoting, mixed."""
    eligible_sorted = sorted(eligible)
    n_uniform = min(N_UNIFORM, len(eligible_sorted))
    uniform = sorted(
        int(f)
        for f in rng.choice(np.array(eligible_sorted, dtype=np.int64), n_uniform, replace=False)
    )
    taken = set(uniform)
    strata: dict[str, list[int]] = {"uniform": uniform}
    for lab in ("output_promoting", "mixed"):
        pool = [f for f in eligible_sorted if majority.get(f) == lab and f not in taken]
        n = min(N_OVERSAMPLE[lab], len(pool))
        picks = (
            sorted(int(f) for f in rng.choice(np.array(pool, dtype=np.int64), n, replace=False))
            if n
            else []
        )
        taken.update(picks)
        strata[f"oversample_{lab}"] = picks
    return strata


def stratum_map(manifest: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for key in ("uniform", "oversample_output_promoting", "oversample_mixed"):
        for f in manifest[key]:
            out[int(f)] = key
    return out


def sample_feat_ids(manifest: dict) -> list[int]:
    ids: list[int] = []
    for key in ("uniform", "oversample_output_promoting", "oversample_mixed"):
        ids.extend(int(f) for f in manifest[key])
    return ids


def phase_sample_diag(args) -> int:
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    bounded = bool(args.evidence_shard_limit or args.raw_shard_limit)
    if bounded and out_root == OUT_EVAL_DEFAULT:
        raise SystemExit(
            "[sample-diag] REFUSED: bounded shard limits with the default out-root would "
            "write a production sample manifest from a partial census; pass --out-root "
            "(per-leg out-roots, crash-fix-rounds rule)"
        )

    # P0.2a — kappa-reproduction gate (ALWAYS full labels shards: real data).
    published = json.loads((FULLDICT_ROOT / "labels_upload" / "kappa_report.json").read_text())
    pub_fr = published["axes"][AXIS]
    fr_votes, fr_majority = load_fulldict_fr_labels()
    recomputed = CM.fleiss_kappa_varying_n(list(fr_votes.values()), FR_CATS)
    gate_ok = abs(recomputed["kappa"] - pub_fr["kappa"]) <= KAPPA_REPRO_TOL
    _log(
        f"[sample-diag] kappa reproduction: published={pub_fr['kappa']:.5f} "
        f"recomputed={recomputed['kappa']:.5f} n_items={recomputed['n_items']} "
        f"(published n_items={pub_fr.get('n_items')}) -> {'PASS' if gate_ok else 'FAIL'}"
    )
    if not gate_ok:
        raise SystemExit(
            f"[sample-diag] HALT: recomputed kappa {recomputed['kappa']:.5f} differs from "
            f"published {pub_fr['kappa']:.5f} by > {KAPPA_REPRO_TOL} — labels join/read bug"
        )

    # P0.1 — census + eligibility + stratified sample.
    frame = load_frame()
    frame_ids = [int(f) for f in frame["feat_ids"]]
    frame_set = set(frame_ids)
    census = evidence_census(
        frame_set, out_root / "partials" / "census.partial.jsonl", args.evidence_shard_limit
    )
    descs = load_descriptions(frame_set)
    crit = {
        "packet_present": 0,
        "ex_pos_ge_min": 0,
        "labels_present": 0,
        "description_present": 0,
    }
    eligible: list[int] = []
    for f in frame_ids:
        c = census.get(f)
        has_packet = c is not None
        has_pos = bool(c and c["n_ex_pos"] >= MIN_EX_POS)
        has_lab = f in fr_votes
        has_desc = f in descs and bool(descs[f].get("description"))
        crit["packet_present"] += has_packet
        crit["ex_pos_ge_min"] += has_pos
        crit["labels_present"] += has_lab
        crit["description_present"] += has_desc
        if has_packet and has_pos and has_lab and has_desc:
            eligible.append(f)
    frac = len(eligible) / len(frame_ids)
    census_report = {
        "n_frame": len(frame_ids),
        "n_eligible": len(eligible),
        "eligible_frac": frac,
        "per_criterion_pass": crit,
        "min_ex_pos": MIN_EX_POS,
        "bounded_inputs": bounded,
        "evidence_shard_limit": args.evidence_shard_limit,
    }
    if not bounded and frac < ELIGIBILITY_FLOOR:
        # report-and-adapt, NOT a gate (plan §4 P0.1): the eligible set IS the
        # re-scoped frame; the deviation is recorded in the manifest.
        _log(
            f"[sample-diag] WARNING: eligibility {frac:.1%} < {ELIGIBILITY_FLOOR:.0%} — "
            "re-scoping the frame to the eligible set (deviation recorded in the manifest)"
        )
    rng = np.random.default_rng(np.random.SeedSequence(list(SAMPLE_SEED)))
    strata = draw_sample(eligible, fr_majority, rng)
    if not bounded:
        assert len(strata["uniform"]) == N_UNIFORM, len(strata["uniform"])
    manifest = {
        "task": TASK,
        "axis": AXIS,
        "frame": "eval_results/issue_1773/phase0/phase0_arrays.npz feat_ids",
        "seed": f"np.random.SeedSequence({list(SAMPLE_SEED)})",
        "census": census_report,
        **strata,
        "realized_counts": {k: len(v) for k, v in strata.items()},
        **CM.repro_meta(),
    }
    (out_root / "sample_manifest.json").write_text(json.dumps(manifest, indent=1))
    _log(
        f"[sample-diag] sample: uniform={len(strata['uniform'])} "
        f"op={len(strata['oversample_output_promoting'])} mixed={len(strata['oversample_mixed'])} "
        f"(eligible {len(eligible)}/{len(frame_ids)} = {frac:.1%})"
    )

    # P0.2b — C0 same-sample manipulation check + c0 labels file for analyze.
    uni_votes = [fr_votes[f] for f in strata["uniform"]]
    c0_uniform = CM.fleiss_kappa_varying_n(uni_votes, FR_CATS)
    manip_ok = abs(c0_uniform["kappa"] - pub_fr["kappa"]) <= C0_MANIP_BAND
    _log(
        f"[sample-diag] C0 manipulation check: kappa_uniform={c0_uniform['kappa']:.4f} "
        f"vs published {pub_fr['kappa']:.4f} (band +/-{C0_MANIP_BAND}) -> "
        f"{'PASS' if manip_ok else 'FAIL'}"
    )
    smap = stratum_map(manifest)
    arms_dir = out_root / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)
    c0_rows = [
        {
            "feat_id": f,
            "axis": AXIS,
            "arm": "c0",
            "stratum": smap[f],
            "label": CM.majority_vote(fr_votes[f]),
            "labels_surviving": fr_votes[f],
            "n_surviving": len(fr_votes[f]),
            "n_launched": N_DRAWS,
        }
        for f in sample_feat_ids(manifest)
    ]
    tmp = arms_dir / ".tmp_axis_labels_c0.jsonl"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in c0_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(arms_dir / "axis_labels_c0.jsonl")

    # P0.4 — disagreement structure (full functional_role set + all axes).
    M_fr, n_fr = votes_to_counts(list(fr_votes.values()), FR_CATS)
    fr_summary = kappa_from_counts(M_fr, n_fr, FR_CATS)
    conf_full = draw_pair_confusion(M_fr, n_fr)
    mixed_sig = mixed_dump_signature(conf_full, fr_summary["prevalence"], FR_CATS)
    per_axis: dict[str, dict] = {}
    for a, (Ma, na) in load_all_axis_counts().items():
        s = kappa_from_counts(Ma, na, CM.AXES[a])
        s["per_label_kappa"] = per_category_kappa(Ma, na, CM.AXES[a])
        per_axis[a] = s
    # covariates of per-item unanimity (frame ∩ labels ∩ census)
    cov_fids = [f for f in frame_ids if f in fr_votes and f in census]
    unanimity = [int(len(set(fr_votes[f])) == 1 and len(fr_votes[f]) >= 2) for f in cov_fids]
    covariates = {
        "feat_ids": cov_fids,
        "unanimity": unanimity,
        "n_ex_pos": [census[f]["n_ex_pos"] for f in cov_fids],
        "activity": [frame["activity"][f] for f in cov_fids],
        "side_ratio": [frame["side_ratio"][f] for f in cov_fids],
        "describe_confidence": [descs.get(f, {}).get("confidence") for f in cov_fids],
    }
    spearman: dict[str, float | None | str] = {}
    try:
        from scipy.stats import spearmanr

        def _finite_or_none(x: float) -> float | None:
            return float(x) if math.isfinite(x) else None  # keep the JSON strict-parseable

        u = np.array(unanimity, dtype=np.float64)
        for k in ("n_ex_pos", "activity", "side_ratio"):
            v = np.array(covariates[k], dtype=np.float64)
            spearman[k] = _finite_or_none(spearmanr(u, v).statistic)
        conf_vals = np.array(
            [c if c is not None else np.nan for c in covariates["describe_confidence"]],
            dtype=np.float64,
        )
        mask = ~np.isnan(conf_vals)
        spearman["describe_confidence"] = _finite_or_none(
            spearmanr(u[mask], conf_vals[mask]).statistic
        )
    except ImportError:  # scipy genuinely absent — report, never silently zero
        spearman = {"error": "scipy unavailable — spearman skipped"}

    # P0.3 — drop taxonomy over the retained raw judge text.
    taxonomy: dict = {"skipped": True}
    if not args.skip_taxonomy:
        taxonomy = drop_taxonomy_scan(
            out_root / "partials" / "taxonomy.partial.jsonl", args.raw_shard_limit
        )
        fr_tax = taxonomy["axes"][AXIS]
        pub_drops = int(pub_fr.get("drop_report", {}).get("content_drops") or 0)
        fr_tax["published_content_drops"] = pub_drops
        fr_tax["taxonomy_coverage_of_published_drops"] = (
            fr_tax["failed_with_raw_text"] / pub_drops if pub_drops else None
        )
        (out_root / "drop_taxonomy.json").write_text(
            json.dumps({**taxonomy, **CM.repro_meta()}, indent=1)
        )
        _log(
            f"[sample-diag] taxonomy ({AXIS}): "
            + " ".join(f"{k}={fr_tax['taxonomy'][k]}" for k in TAXONOMY_CLASSES)
            + f" coverage={fr_tax['taxonomy_coverage_of_published_drops']}"
        )

    diag = {
        "kappa_reproduction": {
            "published": pub_fr["kappa"],
            "recomputed": recomputed["kappa"],
            "abs_diff": abs(recomputed["kappa"] - pub_fr["kappa"]),
            "tolerance": KAPPA_REPRO_TOL,
            "n_items": recomputed["n_items"],
            "n_excluded_lt2": recomputed["n_excluded_lt2"],
            "pass": gate_ok,
        },
        "c0_sample": {
            "kappa_uniform": c0_uniform,
            "manipulation_check_band": C0_MANIP_BAND,
            "manipulation_check_pass": manip_ok,
            "kappa_oversample_output_promoting": CM.fleiss_kappa_varying_n(
                [fr_votes[f] for f in strata["oversample_output_promoting"]], FR_CATS
            ),
            "kappa_oversample_mixed": CM.fleiss_kappa_varying_n(
                [fr_votes[f] for f in strata["oversample_mixed"]], FR_CATS
            ),
        },
        "functional_role_full": {
            **fr_summary,
            "per_label_kappa": per_category_kappa(M_fr, n_fr, FR_CATS),
            "draw_pair_confusion": {
                "categories": list(FR_CATS),
                "matrix": conf_full.tolist(),
            },
            "mixed_dump_signature": mixed_sig,
        },
        "per_axis": per_axis,
        "covariates": {**covariates, "spearman_vs_unanimity": spearman},
        **CM.repro_meta(),
    }
    (out_root / "phase0_diagnostics.json").write_text(json.dumps(diag, indent=1))
    _log(f"[sample-diag] wrote {out_root / 'phase0_diagnostics.json'}")
    if not manip_ok:
        raise SystemExit(
            f"[sample-diag] HALT: C0 sample kappa {c0_uniform['kappa']:.4f} outside "
            f"{pub_fr['kappa']:.4f} +/- {C0_MANIP_BAND} — stratification/join bug; "
            "debug before any spend (plan §7 criterion 1)"
        )
    return 0


# ── P1/P3 arm dispatch ───────────────────────────────────────────────────────


def build_arm_items(
    arm: str,
    feat_ids: list[int],
    packets: dict[int, dict],
    descs: dict[int, dict],
    draws: int = N_DRAWS,
) -> list[tuple[str, str, str, str]]:
    """(custom_id, question, completion, user_msg) JudgeItems for one arm."""
    items = []
    for fid in feat_ids:
        pk = packets[fid]
        desc = descs[fid]["description"]
        for d in range(draws):
            items.append(
                (
                    arm_custom_id(fid, arm, d),
                    f"feat:{fid}:{AXIS}:{arm}",
                    "",
                    render_arm_msg(arm, pk, desc, d),
                )
            )
    return items


def aggregate_arm(
    arm: str, items: list, results: dict[str, dict], strata: dict[int, str]
) -> tuple[list[dict], dict]:
    """Majority vote + rule-9/24 drop split per feature for one arm."""
    votes: dict[int, list[str]] = defaultdict(list)
    tally = {"launched": 0, "ok": 0, "content_drops": 0, "transport_losses": 0}
    launched: set[int] = set()
    for cid, _q, _c, _u in items:
        fid, _arm, _d = parse_arm_custom_id(cid)
        launched.add(fid)
        tally["launched"] += 1
        res = results.get(cid)
        # fail-loud: the dispatch machinery returns a result-or-error dict per
        # launched item; a MISSING entry is a machinery bug, never a content drop
        assert res is not None, f"[arms:{arm}] launched item {cid} has no returned result"
        if isinstance(res, dict) and res.get("error"):
            kind = _classify_error(res)
            tally["content_drops" if kind == "content" else "transport_losses"] += 1
            continue
        lab = CM.validate_axis_label(res, AXIS)
        if lab is None:
            tally["content_drops"] += 1
            continue
        tally["ok"] += 1
        votes[fid].append(lab)
    rows = [
        {
            "feat_id": fid,
            "axis": AXIS,
            "arm": arm,
            "stratum": strata.get(fid, "?"),
            "label": CM.majority_vote(votes.get(fid, [])),
            "labels_surviving": votes.get(fid, []),
            "n_surviving": len(votes.get(fid, [])),
            "n_launched": N_DRAWS,
        }
        for fid in sorted(launched)
    ]
    return rows, tally


def transport_cids(items: list, results: dict[str, dict]) -> list[str]:
    out = []
    for cid, _q, _c, _u in items:
        res = results.get(cid)
        if isinstance(res, dict) and res.get("error") and _classify_error(res) == "transport":
            out.append(cid)
    return out


def dispatch_arm(
    arm: str,
    items: list,
    checkpoint_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, dict]:
    """One arm's full dispatch: fresh per-arm checkpoint dir (llm-judging rule
    22/23 — MANDATORY: c600's prompts are byte-identical to production's, so a
    shared/production checkpoint dir would replay production responses),
    threshold_base=1 (force-batch — fact-check R1: a 4,000-item dispatch could
    silently route sync at ~2x cost under a high realized OTPM), max_tokens=600,
    temperature 1.0, raw text retained; transport-lost draws surgically
    re-dispatched (fresh sub-dirs) up to twice (rule 24)."""
    from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids

    validate_batch_custom_ids([it[0] for it in items])
    results = _dispatch(
        items,
        system=CM.AXIS_SYSTEM_PREAMBLE,
        max_tokens=ARMS_MAX_TOKENS,
        checkpoint_dir=checkpoint_root / arm,
        force_batch=True,
        dry_run=dry_run,
    )
    if dry_run:
        return results
    lost = transport_cids(items, results)
    for retry_pass in (1, 2):
        if not lost:
            break
        retry_items = [it for it in items if it[0] in set(lost)]
        _log(
            f"[arms:{arm}] re-dispatching {len(retry_items)} transport-lost draws "
            f"(pass {retry_pass})"
        )
        rr = _dispatch(
            retry_items,
            system=CM.AXIS_SYSTEM_PREAMBLE,
            max_tokens=ARMS_MAX_TOKENS,
            checkpoint_dir=checkpoint_root / f"{arm}_rejudge_{retry_pass}",
            force_batch=True,
        )
        results.update(rr)
        lost = transport_cids(items, results)
    if lost:
        _log(f"[arms:{arm}] WARNING: {len(lost)} residual transport losses after re-dispatch")
    return results


def _upload_raw_1941(local_dir: Path) -> None:
    """Judge raw text uploads ALWAYS (upload policy) — own #1941 prefix, no
    path mutation of #1773's uploads."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    hub.assert_hub_dir_filecounts(
        local_dir, f"{HF_PREFIX_1941}/judge_raw", allow_patterns=["*.jsonl", "*.json"]
    )
    hub.retry_transient(
        lambda: HfApi().upload_folder(
            folder_path=str(local_dir),
            repo_id=CM.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{HF_PREFIX_1941}/judge_raw",
            allow_patterns=["*.jsonl", "*.json"],
        ),
        what="issue1941 judge_raw upload",
    )


def _load_manifest(out_root: Path) -> dict:
    p = out_root / "sample_manifest.json"
    if not p.exists():
        raise SystemExit(f"[arms] no sample manifest at {p}; run --phase sample-diag first")
    return json.loads(p.read_text())


def _rubric_meta(arm: str) -> dict:
    if arm in WAVE2_SEES:
        return {"rubric": "v2", "rubric_sha16": wave2_rubric_sha16()}
    return {"rubric": "v1-production", "rubric_sha16": None}


def run_arms(args, arms: dict[str, tuple[str, ...]], wave: int) -> int:
    """Shared P1/P3 execution: render-only | smoke | full for a set of arms."""
    out_root: Path = args.out_root
    manifest = _load_manifest(out_root)
    smap = stratum_map(manifest)
    feat_ids = sample_feat_ids(manifest)
    if args.limit and not args.full:
        feat_ids = feat_ids[: args.limit]
    packets = load_sample_packets(set(feat_ids))
    descs = load_descriptions(set(feat_ids))
    missing_desc = [f for f in feat_ids if f not in descs]
    assert not missing_desc, f"{len(missing_desc)} sampled features missing descriptions"

    if args.render_only:
        rd = out_root / "golden_prompts"
        rd.mkdir(parents=True, exist_ok=True)
        checks = []
        for fid in feat_ids:
            pk, desc = packets[fid], descs[fid]["description"]
            for arm in arms:
                user = render_arm_msg(arm, pk, desc, 0)
                (rd / f"{arm_custom_id(fid, arm, 0)}.txt").write_text(
                    f"SYSTEM:\n{CM.AXIS_SYSTEM_PREAMBLE}\n\nUSER:\n{user}"
                )
            if wave == 1:
                checks.append(assert_arm_prompt_identity(pk, desc, 0))
        _log(
            f"[arms] render-only: {len(feat_ids)} features x {len(arms)} arms -> {rd}; "
            f"identity checks PASSED for {len(checks)} features"
        )
        return 0

    if args.smoke:
        fid = feat_ids[0]
        pk, desc = packets[fid], descs[fid]["description"]
        smoke_arms = list(arms)
        items = [
            (arm_custom_id(fid, a, 0), f"feat:{fid}:{AXIS}:{a}", "", render_arm_msg(a, pk, desc, 0))
            for a in smoke_arms
        ]
        while len(items) < 5:  # 5-item live smoke (plan §4 P1 smoke (b))
            d = len(items) - len(smoke_arms) + 1
            items.append(
                (
                    arm_custom_id(fid, smoke_arms[0], d),
                    f"feat:{fid}:{AXIS}:{smoke_arms[0]}",
                    "",
                    render_arm_msg(smoke_arms[0], pk, desc, d),
                )
            )
        from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids

        validate_batch_custom_ids([it[0] for it in items])
        results = _dispatch(
            items,
            system=CM.AXIS_SYSTEM_PREAMBLE,
            max_tokens=ARMS_MAX_TOKENS,
            checkpoint_dir=WORK_ROOT / "judge_checkpoints" / "smoke",
            force_batch=True,  # threshold_base=1 — the forced-batch live smoke
        )
        report = []
        n_valid = n_transport = 0
        for cid, _q, _c, _u in items:
            res = results.get(cid)
            assert res is not None, f"smoke item {cid} returned no result"
            if isinstance(res, dict) and res.get("error"):
                kind = _classify_error(res)
                n_transport += kind == "transport"
                report.append({"custom_id": cid, "error": kind})
                continue
            lab = CM.validate_axis_label(res, AXIS)
            n_valid += lab is not None
            report.append({"custom_id": cid, "label": lab})
        (out_root / "smoke_report.json").write_text(
            json.dumps({"items": report, "n_valid": n_valid, **CM.repro_meta()}, indent=1)
        )
        assert n_transport == 0, f"smoke: {n_transport} transport losses"
        assert n_valid >= 4, f"smoke: only {n_valid}/5 parseable valid labels: {report}"
        _log(f"[arms] SMOKE PASS: {n_valid}/5 valid labels, 0 transport -> smoke_report.json")
        return 0

    # --full
    if manifest["census"].get("bounded_inputs"):
        raise SystemExit(
            "[arms] REFUSED: sample manifest was built from bounded inputs (a smoke census); "
            "re-run --phase sample-diag unbounded before any full dispatch"
        )
    kappa_by_arm_path = out_root / "arms" / "kappa_by_arm.json"
    kba = json.loads(kappa_by_arm_path.read_text()) if kappa_by_arm_path.exists() else {}
    if wave == 2:
        (out_root / "wave2_rubric.json").write_text(
            json.dumps(
                {
                    "defs": FR_RUBRIC_V2_DEFS,
                    "decision_procedure": FR_DECISION_PROCEDURE,
                    "sha16": wave2_rubric_sha16(),
                    **CM.repro_meta(),
                },
                indent=1,
            )
        )
    for arm in arms:
        out_path = out_root / "arms" / f"axis_labels_{arm}.jsonl"
        if out_path.exists() and not args.rerun_arm:
            _log(f"[arms:{arm}] SKIP (exists): {out_path}")
            continue
        items = build_arm_items(arm, feat_ids, packets, descs)
        _log(
            f"[arms:{arm}] dispatching {len(items)} items (force-batch, "
            f"max_tokens={ARMS_MAX_TOKENS})"
        )
        results = dispatch_arm(arm, items, WORK_ROOT / "judge_checkpoints", dry_run=args.dry_run)
        if args.dry_run:
            continue
        rows, tally = aggregate_arm(arm, items, results, smap)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.parent / f".tmp_{out_path.name}"
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out_path)
        _write_raw(results, WORK_ROOT / "judge_raw" / f"arm_{arm}")
        uni = [r["labels_surviving"] for r in rows if r["stratum"] == "uniform"]
        kba[arm] = {
            **CM.fleiss_kappa_varying_n(uni, FR_CATS),
            "scope": "uniform stratum only (oversample never enters headline bars)",
            "drop_report": tally,
            "content_drop_rate": tally["content_drops"] / max(tally["launched"], 1),
            # registered denominator scope (plan §6 DV table: "content-drops /
            # launched, per arm"): ALL launched draws, uniform + oversample strata
            "content_drop_rate_scope": "content drops / all launched draws (uniform + oversample)",
            "transport_residual_rate": tally["transport_losses"] / max(tally["launched"], 1),
            "max_tokens": ARMS_MAX_TOKENS,
            **_rubric_meta(arm),
        }
        kappa_by_arm_path.write_text(json.dumps({**kba, "_partial": True}, indent=1))
        _log(
            f"[arms:{arm}] done: kappa_uniform={kba[arm]['kappa']:.4f} "
            f"content_drops={tally['content_drops']} transport={tally['transport_losses']}"
        )
    if not args.dry_run and not args.no_upload:
        _upload_raw_1941(WORK_ROOT / "judge_raw")
    return 0


# ── P2 analysis ──────────────────────────────────────────────────────────────


def load_arm_rows(out_root: Path) -> dict[str, list[dict]]:
    """{arm: rows} for every axis_labels_<arm>.jsonl present under arms/."""
    out: dict[str, list[dict]] = {}
    for p in sorted((out_root / "arms").glob("axis_labels_*.jsonl")):
        arm = p.stem.replace("axis_labels_", "")
        out[arm] = list(CM.iter_jsonl(p))
    return out


def _aligned_counts(rows: list[dict], feat_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Count matrix aligned to a FIXED feature order (absent feature -> all-zero
    row -> n_i = 0 -> excluded, matching the estimator's <2-surviving rule)."""
    by_fid = {int(r["feat_id"]): list(r["labels_surviving"]) for r in rows}
    votes = [by_fid.get(f, []) for f in feat_order]
    return votes_to_counts(votes, FR_CATS)


def phase_analyze(args) -> int:
    out_root: Path = args.out_root
    manifest = _load_manifest(out_root)
    uniform = [int(f) for f in manifest["uniform"]]
    arm_rows = load_arm_rows(out_root)
    if "c0" not in arm_rows:
        raise SystemExit("[analyze] no c0 labels (arms/axis_labels_c0.jsonl); run sample-diag")
    api_arms = [a for a in ARM_PREFERENCE if a in arm_rows]
    if not api_arms:
        raise SystemExit("[analyze] no API-arm label files under arms/; run --phase arms --full")

    # transport-hygiene gate (plan §7 criterion 3) from the per-arm tallies.
    # Fail-loud on a missing tally entry: run_arms writes one per dispatched
    # arm (arms- AND analyze-format entries both carry the rates), so an
    # absent file/entry beside label files means the gate would silently
    # default rate=0.0 (code-review r1 bug-class sweep, secondary consumer).
    kba_path = out_root / "arms" / "kappa_by_arm.json"
    kba = json.loads(kba_path.read_text()) if kba_path.exists() else {}
    missing_kba = [a for a in api_arms if a not in kba]
    if missing_kba:
        raise SystemExit(
            f"[analyze] arms {missing_kba} have label files but no kappa_by_arm.json tally "
            f"entry at {kba_path} — the transport-hygiene gate cannot read their rates; "
            "re-run --phase arms --full (or restore the file)"
        )
    for arm in api_arms:
        rate = kba.get(arm, {}).get("transport_residual_rate", 0.0)
        if rate and rate > TRANSPORT_RESIDUAL_MAX:
            raise SystemExit(
                f"[analyze] HALT: arm {arm} residual transport rate {rate:.3%} > "
                f"{TRANSPORT_RESIDUAL_MAX:.1%} — re-judge the lost draws before analysis "
                "(rule 24; plan §7 criterion 3)"
            )

    counts = {a: _aligned_counts(arm_rows[a], uniform) for a in ["c0", *api_arms]}
    rng = np.random.default_rng(np.random.SeedSequence(list(BOOT_SEED)))
    idx = rng.integers(0, len(uniform), size=(BOOT_B, len(uniform)))
    draws = {a: bootstrap_kappa_draws(M, n, idx) for a, (M, n) in counts.items()}

    def pct(v: np.ndarray, q: float) -> float:
        return float(np.nanpercentile(v, q))

    summary: dict[str, dict] = {}
    for a, (M, n) in counts.items():
        point = kappa_from_counts(M, n, FR_CATS)
        rows_a = arm_rows[a]
        per_stratum: dict[str, dict] = {}
        for st in ("uniform", "oversample_output_promoting", "oversample_mixed"):
            votes = [list(r["labels_surviving"]) for r in rows_a if r["stratum"] == st]
            if votes:
                Ms, ns = votes_to_counts(votes, FR_CATS)
                s = kappa_from_counts(Ms, ns, FR_CATS)
                s["per_label_kappa"] = per_category_kappa(Ms, ns, FR_CATS)
                conf = draw_pair_confusion(Ms, ns)
                s["draw_pair_confusion"] = conf.tolist()
                s["mixed_dump_signature"] = mixed_dump_signature(
                    conf, s.get("prevalence", {}), FR_CATS
                )
                per_stratum[st] = s
        summary[a] = {
            "kappa_uniform": point["kappa"],
            "kappa_uniform_ci95": [pct(draws[a], 2.5), pct(draws[a], 97.5)],
            "uniform_detail": point,
            "per_stratum": per_stratum,
            **{
                k: kba.get(a, {}).get(k)
                for k in (
                    "drop_report",
                    "content_drop_rate",
                    "content_drop_rate_scope",
                    "transport_residual_rate",
                    "max_tokens",
                    "rubric",
                    "rubric_sha16",
                )
            },
        }
        # c0's production drop rate over the sample: kappa_report records 0
        # transport losses fleet-wide, so missing draws here are content drops.
        if a == "c0":
            launched = len([r for r in arm_rows[a] if r["stratum"] == "uniform"]) * N_DRAWS
            surv = sum(r["n_surviving"] for r in arm_rows[a] if r["stratum"] == "uniform")
            summary[a]["content_drop_rate"] = (launched - surv) / max(launched, 1)
            summary[a]["content_drop_rate_scope"] = (
                "content drops / uniform-stratum launched draws (c0 as-run labels; "
                "kappa_report records 0 transport losses fleet-wide)"
            )
            summary[a]["max_tokens"] = 400
            summary[a]["rubric"] = "v1-production (as-run labels)"

    # paired contrasts (identical uniform feature set; same bootstrap idx).
    # CI-LB95 = 2.5th percentile of the paired bootstrap distribution — the
    # TWO-SIDED 95% CI lower bound (registered threshold convention).
    contrasts: dict[str, dict] = {}

    def add_contrast(name: str, a: str, b: str) -> None:
        d = draws[a] - draws[b]
        contrasts[name] = {
            "arms": [a, b],
            "delta_point": summary[a]["kappa_uniform"] - summary[b]["kappa_uniform"],
            "delta_ci95": [pct(d, 2.5), pct(d, 97.5)],
            "ci_lb95": pct(d, 2.5),
            "n_boot": BOOT_B,
        }

    for a in api_arms:
        add_contrast(f"{a}_vs_c0", a, "c0")
    if "c600" in draws:
        contrasts["budget_c600_vs_c0"] = dict(contrasts["c600_vs_c0"])
        for a in api_arms:
            if a != "c600":
                add_contrast(f"evid_{a}_vs_c600", a, "c600")

    # instrument-drift halt (plan §7 criterion 2)
    if "c600" in summary:
        drift = abs(summary["c600"]["kappa_uniform"] - summary["c0"]["kappa_uniform"])
        if drift > INSTRUMENT_DRIFT_MAX:
            raise SystemExit(
                f"[analyze] HALT: |kappa_c600 - kappa_c0| = {drift:.3f} > "
                f"{INSTRUMENT_DRIFT_MAX} — budget alone cannot plausibly move kappa that "
                "far; debug against golden prompts before reading treatments (plan §7 crit 2)"
            )

    # registered sensitivity read: all arms restricted to the intersection of
    # >=2-surviving items (drop-driven composition check, plan §3)
    inter_mask = np.ones(len(uniform), dtype=bool)
    for _a, (_M, n) in counts.items():
        inter_mask &= n >= 2
    sensitivity: dict[str, dict] = {"n_intersection": int(inter_mask.sum())}
    for a, (M, n) in counts.items():
        s = kappa_from_counts(M[inter_mask], n[inter_mask], FR_CATS)
        div = abs(s["kappa"] - summary[a]["kappa_uniform"])
        sensitivity[a] = {
            "kappa_intersection": s["kappa"],
            "divergence_vs_own_items": div,
            "caveat": bool(div > SENSITIVITY_DIVERGENCE),
        }

    # side_ratio convergent read per arm (restricted-frame mechanical covariate)
    frame = load_frame()
    side_ratio_reads: dict[str, dict] = {}
    for a, rows_a in arm_rows.items():
        maj = {int(r["feat_id"]): r["label"] for r in rows_a}
        vals: dict[str, list[float]] = defaultdict(list)
        for fid, lab in maj.items():
            sr = frame["side_ratio"].get(fid)
            if sr is not None and lab in FR_CATS:
                vals[lab].append(sr)
        side_ratio_reads[a] = {
            "per_label_mean": {c: float(np.mean(vals[c])) if vals[c] else None for c in FR_CATS},
            "per_label_n": {c: len(vals[c]) for c in FR_CATS},
            "auc_input_vs_output": auc_rank(
                np.array(vals["output_promoting"]), np.array(vals["input_side"])
            ),
        }

    (out_root / "arms" / "kappa_by_arm.json").write_text(
        json.dumps(
            {**summary, "sensitivity_intersection_read": sensitivity, **CM.repro_meta()}, indent=1
        )
    )
    (out_root / "contrasts.json").write_text(
        json.dumps(
            {
                **contrasts,
                "boot_seed": f"np.random.SeedSequence({list(BOOT_SEED)})",
                "ci_convention": (
                    "CI-LB95 = 2.5th percentile of the paired feature-bootstrap "
                    "distribution (two-sided 95% CI lower bound)"
                ),
                "side_ratio_reads": side_ratio_reads,
                **CM.repro_meta(),
            },
            indent=1,
        )
    )
    _log(
        "[analyze] kappa_uniform: "
        + " ".join(f"{a}={summary[a]['kappa_uniform']:.4f}" for a in ["c0", *api_arms])
    )
    if not args.no_figures:
        make_figures(args.figs_root, out_root, summary, contrasts, api_arms)
    return 0


def make_figures(
    figs_root: Path, out_root: Path, summary: dict, contrasts: dict, api_arms: list[str]
) -> None:
    """Hero + exploratory figures (plan §6). Errorbar offsets clamped
    non-negative element-wise (gotchas: xerr/yerr must be >= 0)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    figs_root.mkdir(parents=True, exist_ok=True)
    arms = ["c0", *api_arms]
    names = {
        "c0": "Production replica",
        "c600": "Budget-only control",
        "n600": "+ non-activating examples",
        "r600": "+ neighbour features",
        "b600": "+ both",
        "d600": "Revised rubric",
        "bd600": "Revised rubric + both",
    }

    # hero: per-arm kappa with paired-bootstrap 95% CIs
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ks = np.array([summary[a]["kappa_uniform"] for a in arms])
    lo = np.array([summary[a]["kappa_uniform_ci95"][0] for a in arms])
    hi = np.array([summary[a]["kappa_uniform_ci95"][1] for a in arms])
    yerr = np.vstack([np.maximum(0.0, ks - lo), np.maximum(0.0, hi - ks)])
    x = np.arange(len(arms))
    ax.bar(x, ks, color="#4477AA")
    ax.errorbar(x, ks, yerr=yerr, fmt="none", ecolor="black", capsize=3)
    ax.axhline(0.3176, ls="--", color="gray", label="full-dict production (0.318)")
    ax.axhline(KAPPA_REPAIR_MIN, ls=":", color="firebrick", label="repair bar (0.60)")
    ax.axhspan(0.629, 0.708, color="green", alpha=0.12, label="sibling axes band")
    ax.set_xticks(x, [names.get(a, a) for a in arms], rotation=20, ha="right")
    ax.set_ylabel("inter-draw Fleiss kappa (varying-n, uniform n=600)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_root / "kappa_by_arm_hero.png", dpi=200)
    plt.close(fig)

    # delta forest (contrasts vs c0 + attribution contrasts)
    fig, ax = plt.subplots(figsize=(6.8, 0.7 * max(len(contrasts), 3) + 1.2))
    items = [(k, v) for k, v in contrasts.items() if isinstance(v, dict) and "ci_lb95" in v]
    ys = np.arange(len(items))[::-1]
    for y, (name, c) in zip(ys, items):
        ax.errorbar(
            c["delta_point"],
            y,
            xerr=[
                [max(0.0, c["delta_point"] - c["delta_ci95"][0])],
                [max(0.0, c["delta_ci95"][1] - c["delta_point"])],
            ],
            fmt="o",
            color="#4477AA",
            capsize=3,
        )
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(DELTA_CI_LB95_MIN, ls=":", color="firebrick", lw=0.8)
    ax.set_yticks(ys, [k for k, _ in items], fontsize=8)
    ax.set_xlabel("delta kappa (paired bootstrap, 95% CI)")
    fig.tight_layout()
    fig.savefig(figs_root / "delta_contrast_forest.png", dpi=200)
    plt.close(fig)

    # per-label kappa_j grouped bars (uniform stratum)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / len(arms)
    for i, a in enumerate(arms):
        pl = summary[a]["per_stratum"].get("uniform", {}).get("per_label_kappa", {})
        vals = [pl.get(c, float("nan")) for c in FR_CATS]
        ax.bar(np.arange(len(FR_CATS)) + i * width, vals, width, label=names.get(a, a))
    ax.set_xticks(np.arange(len(FR_CATS)) + 0.4, FR_CATS)
    ax.set_ylabel("per-label Fleiss kappa_j (uniform)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs_root / "per_label_kappa.png", dpi=200)
    plt.close(fig)

    # draw-pair confusion heatmaps: production replica vs best API arm
    best = max(api_arms, key=lambda a: summary[a]["kappa_uniform"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
    for ax_i, a in zip(axes, ["c0", best]):
        C = np.array(summary[a]["per_stratum"]["uniform"]["draw_pair_confusion"])
        share = C / max(C.sum(), 1.0)
        im = ax_i.imshow(share, cmap="Blues")
        ax_i.set_xticks(range(len(FR_CATS)), FR_CATS, rotation=30, ha="right", fontsize=7)
        ax_i.set_yticks(range(len(FR_CATS)), FR_CATS, fontsize=7)
        ax_i.set_title(f"{names.get(a, a)} (draw pairs)", fontsize=9)
        for r in range(len(FR_CATS)):
            for c in range(len(FR_CATS)):
                ax_i.text(c, r, f"{share[r, c]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax_i, shrink=0.8)
    fig.savefig(figs_root / "draw_pair_confusion.png", dpi=200)
    plt.close(fig)

    # prevalence per arm (uniform)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottoms = np.zeros(len(arms))
    colors = {"input_side": "#4477AA", "output_promoting": "#EE6677", "mixed": "#CCBB44"}
    for c in FR_CATS:
        vals = np.array([summary[a]["uniform_detail"]["prevalence"].get(c, 0.0) for a in arms])
        ax.bar(np.arange(len(arms)), vals, bottom=bottoms, label=c, color=colors[c])
        bottoms += vals
    ax.set_xticks(np.arange(len(arms)), [names.get(a, a) for a in arms], rotation=20, ha="right")
    ax.set_ylabel("pooled label prevalence (uniform)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_root / "prevalence_by_arm.png", dpi=200)
    plt.close(fig)

    # PABAK vs kappa decomposition
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    kap = [summary[a]["uniform_detail"]["kappa"] for a in arms]
    pab = [summary[a]["uniform_detail"].get("pabak_multi", float("nan")) for a in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, kap, 0.4, label="kappa")
    ax.bar(x + 0.2, pab, 0.4, label="PABAK (descriptive)")
    ax.set_xticks(x, [names.get(a, a) for a in arms], rotation=20, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_root / "pabak_vs_kappa.png", dpi=200)
    plt.close(fig)

    # drop-rate bars (content vs transport split)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cd = np.array([summary[a].get("content_drop_rate") or 0.0 for a in arms])
    tr = np.array([summary[a].get("transport_residual_rate") or 0.0 for a in arms])
    ax.bar(np.arange(len(arms)), cd, label="content drops", color="#EE6677")
    ax.bar(np.arange(len(arms)), tr, bottom=cd, label="transport residue", color="#BBBBBB")
    ax.axhline(DROP_RATE_MAX, ls=":", color="firebrick", label="lattice drop bar (1.5%)")
    ax.axhline(0.0239, ls="--", color="gray", label="production FR (2.39%)")
    ax.set_xticks(np.arange(len(arms)), [names.get(a, a) for a in arms], rotation=20, ha="right")
    ax.set_ylabel("drop rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs_root / "drop_rates_by_arm.png", dpi=200)
    plt.close(fig)

    # production drop-taxonomy stacked bar (from drop_taxonomy.json when present)
    tax_path = out_root / "drop_taxonomy.json"
    if tax_path.exists():
        tax = json.loads(tax_path.read_text())
        tax_axes = [a for a in (AXIS, "abstraction") if a in tax.get("axes", {})]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        bottoms = np.zeros(len(tax_axes))
        for cls in TAXONOMY_CLASSES:
            vals = np.array([tax["axes"][a]["taxonomy"].get(cls, 0) for a in tax_axes], dtype=float)
            ax.bar(np.arange(len(tax_axes)), vals, bottom=bottoms, label=cls)
            bottoms += vals
        ax.set_xticks(np.arange(len(tax_axes)), tax_axes)
        ax.set_ylabel("classified content drops (production raw text)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs_root / "drop_taxonomy_production.png", dpi=200)
        plt.close(fig)

    # per-ARM drop-taxonomy stacked bars over the wave-1 arms' RETAINED raw
    # judge text (registered concern per-arm-taxonomy-figure): same production
    # parse + classify_drop per arm; class -> color matches the production
    # figure (same TAXONOMY_CLASSES order on the default cycle). Drops with NO
    # retained raw text (API-level content-error dicts — unclassifiable by
    # construction) stack on top in gray, so each bar totals the arm's
    # published content_drops. The JSON companion is written beside the
    # summary so the bars are re-derivable.
    arm_tax = arm_drop_taxonomy_scan()
    if arm_tax:
        for a, d in arm_tax.items():
            cd = (summary.get(a, {}).get("drop_report") or {}).get("content_drops")
            d["content_drops_published"] = cd
            d["no_raw_text_unclassifiable"] = (
                max(cd - d["failed_with_raw_text"], 0) if isinstance(cd, int) else None
            )
        (out_root / "drop_taxonomy_by_arm.json").write_text(
            json.dumps({"arms": arm_tax, **CM.repro_meta()}, indent=1)
        )
        t_arms = list(arm_tax)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        bottoms = np.zeros(len(t_arms))
        for cls in TAXONOMY_CLASSES:
            vals = np.array([arm_tax[a]["taxonomy"].get(cls, 0) for a in t_arms], dtype=float)
            ax.bar(np.arange(len(t_arms)), vals, bottom=bottoms, label=cls)
            bottoms += vals
        rem = np.array(
            [arm_tax[a].get("no_raw_text_unclassifiable") or 0 for a in t_arms], dtype=float
        )
        ax.bar(
            np.arange(len(t_arms)),
            rem,
            bottom=bottoms,
            label="no retained raw text (unclassifiable)",
            color="#BBBBBB",
        )
        ax.set_xticks(
            np.arange(len(t_arms)), [names.get(a, a) for a in t_arms], rotation=20, ha="right"
        )
        ax.set_ylabel("content drops (taxonomy over retained arm raw text)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs_root / "drop_taxonomy_by_arm.png", dpi=200)
        plt.close(fig)

    # side_ratio-by-majority-label violins per arm (convergent mechanical read)
    frame = load_frame()
    arm_rows = load_arm_rows(out_root)
    fig, axes2 = plt.subplots(1, len(arms), figsize=(3.0 * len(arms), 3.4), sharey=True)
    axes2 = np.atleast_1d(axes2)
    for ax_i, a in zip(axes2, arms):
        data, labs = [], []
        maj = {int(r["feat_id"]): r["label"] for r in arm_rows.get(a, [])}
        for c in FR_CATS:
            vals = [
                frame["side_ratio"][f]
                for f, lab in maj.items()
                if lab == c and f in frame["side_ratio"]
            ]
            if vals:
                data.append(vals)
                labs.append(f"{c}\n(n={len(vals)})")
        if data:
            ax_i.violinplot(data, showmedians=True)
            ax_i.set_xticks(np.arange(1, len(labs) + 1), labs, fontsize=6)
        ax_i.set_title(names.get(a, a), fontsize=8)
    axes2[0].set_ylabel("side_ratio (answer-side firing share)")
    fig.tight_layout()
    fig.savefig(figs_root / "side_ratio_by_label.png", dpi=200)
    plt.close(fig)

    # per-item unanimity covariate scatters (phase0 diagnostics, when present)
    diag_path = out_root / "phase0_diagnostics.json"
    if diag_path.exists():
        cov = json.loads(diag_path.read_text()).get("covariates", {})
        keys = [k for k in ("n_ex_pos", "activity", "side_ratio") if cov.get(k)]
        if keys and cov.get("unanimity"):
            u = np.array(cov["unanimity"], dtype=float)
            rng = np.random.default_rng(0)  # display jitter only
            fig, axes3 = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
            axes3 = np.atleast_1d(axes3)
            for ax_i, k in zip(axes3, keys):
                v = np.array(cov[k], dtype=float)
                ax_i.scatter(v, u + rng.uniform(-0.06, 0.06, len(u)), s=2, alpha=0.15)
                rho = (cov.get("spearman_vs_unanimity") or {}).get(k)
                ax_i.set_xlabel(k, fontsize=8)
                ax_i.set_title(
                    f"spearman={rho:.3f}" if isinstance(rho, int | float) else "spearman=n/a",
                    fontsize=8,
                )
            axes3[0].set_ylabel("draw unanimity (0/1, jittered)")
            fig.tight_layout()
            fig.savefig(figs_root / "unanimity_covariates.png", dpi=200)
            plt.close(fig)
    _log(f"[analyze] figures -> {figs_root}")


# ── P3 wave-2 trigger + P4 decision lattice ─────────────────────────────────


def wave2_trigger(summary: dict, contrasts: dict, phase0_diag: dict) -> dict:
    """Registered wave-2 trigger (plan §3, runs at most once): wave-1 verdict
    != REPAIR AND (kappa_b600 >= 0.45 with CI-LB95(delta_b600) > 0, OR the
    Phase-0 mixed-dump signature holds)."""
    wave1 = lattice_verdict(summary, contrasts, [a for a in WAVE1_ARMS if a in summary])
    b = summary.get("b600", {})
    lb = contrasts.get("b600_vs_c0", {}).get("ci_lb95")
    cond_b = bool(
        b.get("kappa_uniform") is not None
        and b["kappa_uniform"] >= WAVE2_TRIGGER_KAPPA
        and lb is not None
        and lb > 0
    )
    mixed = bool(
        phase0_diag.get("functional_role_full", {}).get("mixed_dump_signature", {}).get("holds")
    )
    fire = wave1["verdict"] != "REPAIR" and (cond_b or mixed)
    return {
        "fire": fire,
        "wave1_verdict": wave1["verdict"],
        "cond_b600": cond_b,
        "mixed_dump_signature_holds": mixed,
    }


def lattice_verdict(summary: dict, contrasts: dict, arms_present: list[str]) -> dict:
    """The registered §3 verdict lattice (DISJOINT + exhaustive):

    REPAIR <=> exists arm A in Q with kappa_A >= 0.60 AND CI-LB95(delta_A vs
    c0) >= 0.15 AND content-drop(A) <= 1.5%; RETIRE otherwise. Adopted change
    on REPAIR = the FIRST qualifying arm in the preference order (smallest
    instrument change, then token cost); the pre-registered primary HEADLINE
    arm stays b600 regardless of adoption."""
    per_arm: dict[str, dict] = {}
    qualifying: list[str] = []
    for arm in ARM_PREFERENCE:
        if arm not in arms_present or arm == "c0":
            continue
        k = summary.get(arm, {}).get("kappa_uniform")
        lb = contrasts.get(f"{arm}_vs_c0", {}).get("ci_lb95")
        drop = summary.get(arm, {}).get("content_drop_rate")
        checks = {
            "kappa_uniform": k,
            "kappa_ge_bar": bool(k is not None and k >= KAPPA_REPAIR_MIN),
            "ci_lb95": lb,
            "ci_lb95_ge_bar": bool(lb is not None and lb >= DELTA_CI_LB95_MIN),
            "content_drop_rate": drop,
            "drop_le_bar": bool(drop is not None and drop <= DROP_RATE_MAX),
        }
        checks["qualifies"] = (
            checks["kappa_ge_bar"] and checks["ci_lb95_ge_bar"] and checks["drop_le_bar"]
        )
        per_arm[arm] = checks
        if checks["qualifies"]:
            qualifying.append(arm)
    verdict = "REPAIR" if qualifying else "RETIRE"
    return {
        "verdict": verdict,
        "adopted_arm": qualifying[0] if qualifying else None,
        "qualifying_arms": qualifying,
        "per_arm": per_arm,
        "bars": {
            "kappa_min": KAPPA_REPAIR_MIN,
            "delta_ci_lb95_min": DELTA_CI_LB95_MIN,
            "content_drop_max": DROP_RATE_MAX,
            # registered denominator scope of the drop bar's input (plan §6 DV
            # table: "content-drops / launched, per arm" — API arms only; c0
            # never enters the lattice)
            "content_drop_denominator": "all launched draws per API arm (uniform + oversample)",
        },
    }


def usability_string(verdict: dict) -> str:
    """The consumer-facing AXIS_USABILITY value (plan §4 P4, both branches)."""
    if verdict["verdict"] == "REPAIR":
        return f"superseded: re-label with {verdict['adopted_arm']} evidence before use (#1941)"
    return (
        "unusable: inter-draw kappa 0.318 vs 0.63-0.71 siblings (#1941; RETIRE — "
        "no arm cleared the registered repair lattice)"
    )


def _additive_equal(a: object, b: object) -> bool:
    """NaN-aware exact equality for the additivity assert: two float NaNs
    compare EQUAL (json round-trips NaN literals to float('nan'), and
    `nan != nan` under bare `==` crashed the first --apply-marking run on
    rows whose `detection_score` is NaN); everything else is exact `==`,
    recursing into lists/dicts so a nested NaN is covered too. Any REAL
    value change still compares unequal (fail-loud preserved)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_additive_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_additive_equal(v, b[k]) for k, v in a.items())
    return a == b


def mark_rows(rows: list[dict], usability: dict[str, str]) -> list[dict]:
    """ADDITIVE per-row marking: adds/extends `axis_usability` and asserts every
    original field is preserved unchanged (plan §4 P4 additivity contract;
    NaN-aware via `_additive_equal` — see its docstring)."""
    out = []
    for r in rows:
        nr = dict(r)
        au = dict(nr.get("axis_usability") or {})
        au.update(usability)
        nr["axis_usability"] = au
        for k, v in r.items():
            if k != "axis_usability":
                assert _additive_equal(nr[k], v), f"additivity violated on field {k!r}"
        out.append(nr)
    return out


def load_analyze_summary(out_root: Path, phase: str) -> dict:
    """Load arms/kappa_by_arm.json asserting it is the ANALYZE-format output.

    Two writers share that path: `run_arms` checkpoints per-arm tallies in
    ARMS format (`{"kappa": ...}` entries + a top-level `"_partial": true`),
    and `phase_analyze` rewrites it in ANALYZE format (`{"kappa_uniform":
    ...}` entries, no `_partial`, paired contrasts in contrasts.json). A
    downstream consumer fed the arms-format file reads `kappa_uniform` ->
    None -> `qualifies=False`, so a genuinely qualifying arm silently
    resolves RETIRE (code-review r1 Major). Refuses fail-loud: (a) a
    `_partial`/arms-format file, and (b) any labeled arm
    (arms/axis_labels_<arm>.jsonl present) missing an analyze-format entry
    — both name the re-analyze step. Returns the verified summary dict.
    """
    path = out_root / "arms" / "kappa_by_arm.json"
    if not path.exists():
        raise SystemExit(f"[{phase}] no {path}; run --phase arms --full then --phase analyze")
    summary = json.loads(path.read_text())
    if summary.get("_partial"):
        raise SystemExit(
            f"[{phase}] REFUSED: {path} is the arms-phase checkpoint write "
            "(arms-format, _partial=true) — its entries carry no kappa_uniform, so every "
            "lattice bar reads None and a qualifying arm would silently RETIRE. "
            "Run `--phase analyze` first (the re-analyze after any wave-2 / --rerun-arm "
            "dispatch)."
        )
    labeled = sorted(
        p.stem.removeprefix("axis_labels_") for p in (out_root / "arms").glob("axis_labels_*.jsonl")
    )
    stale = [a for a in labeled if a != "c0" and "kappa_uniform" not in (summary.get(a) or {})]
    if stale:
        raise SystemExit(
            f"[{phase}] REFUSED: arms {stale} have label files but no analyze-format "
            f"kappa_uniform entry in {path} — run `--phase analyze` first so the "
            "summary + contrasts cover every labeled arm."
        )
    return summary


def load_contrasts_covering(out_root: Path, summary: dict, phase: str) -> dict:
    """Load contrasts.json asserting every summary API arm has its `_vs_c0`
    contrast (an analyze crash between the two writes, or a stale file,
    would otherwise feed None -> ci_lb95_ge_bar=False silently)."""
    path = out_root / "contrasts.json"
    if not path.exists():
        raise SystemExit(f"[{phase}] no {path}; run `--phase analyze` first")
    contrasts = json.loads(path.read_text())
    missing = [a for a in ARM_PREFERENCE if a in summary and f"{a}_vs_c0" not in contrasts]
    if missing:
        raise SystemExit(
            f"[{phase}] REFUSED: contrasts.json lacks {[f'{a}_vs_c0' for a in missing]} — "
            "stale vs the summary; run `--phase analyze` first."
        )
    return contrasts


def phase_wave2(args) -> int:
    out_root: Path = args.out_root
    summary = load_analyze_summary(out_root, "wave2")
    contrasts = load_contrasts_covering(out_root, summary, "wave2")
    phase0 = json.loads((out_root / "phase0_diagnostics.json").read_text())
    trig = wave2_trigger(summary, contrasts, phase0)
    _log(f"[wave2] trigger: {json.dumps(trig)}")
    if not trig["fire"] and not args.force:
        raise SystemExit(
            "[wave2] REFUSED: registered trigger not met (RETIRE is terminal after wave 1); "
            "pass --force only with a recorded justification"
        )
    rc = run_arms(args, WAVE2_SEES, wave=2)
    if rc == 0 and args.full and not args.dry_run:
        # RE-ANALYZE (code-review r1 Major fix): run_arms left kappa_by_arm.json
        # in arms format (_partial), which decide-mark refuses — fold d600/bd600
        # into the analyze-format summary + contrasts here so the registered §10
        # sequence (analyze -> wave2 --full -> decide-mark) is closed under this
        # phase. --render-only / --dry-run legs write no kba and skip this.
        _log("[wave2] arms complete — auto-running --phase analyze (re-analyze)")
        rc = phase_analyze(args)
    return rc


def phase_decide_mark(args) -> int:
    out_root: Path = args.out_root
    summary = load_analyze_summary(out_root, "decide-mark")
    contrasts = load_contrasts_covering(out_root, summary, "decide-mark")
    arms_present = [a for a in ARM_PREFERENCE if a in summary]
    verdict = lattice_verdict(summary, contrasts, arms_present)
    s = usability_string(verdict)
    decision = {
        **verdict,
        "usability_string": s,
        "wave2_ran": any(a in summary for a in WAVE2_ARMS),
        **CM.repro_meta(),
    }
    (out_root / "decision.json").write_text(json.dumps(decision, indent=1))
    _log(f"[decide-mark] verdict={verdict['verdict']} adopted={verdict['adopted_arm']}")
    (out_root / "AXIS_USABILITY.json").write_text(
        json.dumps({AXIS: s, **CM.repro_meta()}, indent=1)
    )
    if CM.AXIS_USABILITY.get(AXIS) != s:
        _log(
            "[decide-mark] NOTE: scripts/issue1773_common.py AXIS_USABILITY does not yet "
            f"carry this verdict — land: AXIS_USABILITY = {{{AXIS!r}: {s!r}}}"
        )
    if not args.apply_marking:
        _log("[decide-mark] decision.json written; pass --apply-marking to patch the table")
        return 0
    rows = list(CM.iter_jsonl(FEATURE_TABLE))
    marked = mark_rows(rows, {AXIS: s})
    tmp = FEATURE_TABLE.parent / f".tmp_{FEATURE_TABLE.name}"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in marked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(FEATURE_TABLE)
    sidecar = FEATURE_TABLE.parent / "feature_table_v1.usability_note.json"
    sidecar.write_text(
        json.dumps(
            {
                "task": TASK,
                "reason": s,
                "date": CM.repro_meta()["timestamp_utc"],
                "note": (
                    "in-place ADDITIVE regeneration: per-row axis_usability field added; "
                    "no field removed or altered (git history preserves prior bytes — "
                    "upload-policy regeneration note)"
                ),
            },
            indent=1,
        )
    )
    _log(f"[decide-mark] marked {len(marked)} rows in {FEATURE_TABLE} (+ sidecar)")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        required=True,
        choices=["sample-diag", "arms", "analyze", "wave2", "decide-mark"],
    )
    ap.add_argument("--wave", type=int, default=1, choices=[1], help="arms phase wave (1)")
    ap.add_argument("--out-root", type=Path, default=OUT_EVAL_DEFAULT)
    ap.add_argument("--figs-root", type=Path, default=OUT_FIGS_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="feature cap (render-only/smoke legs)")
    ap.add_argument("--render-only", action="store_true", help="golden prompts, zero API calls")
    ap.add_argument("--smoke", action="store_true", help="5-item live forced-batch smoke")
    ap.add_argument("--full", action="store_true", help="full per-arm dispatch (4,000 items/arm)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--rerun-arm", action="store_true", help="re-dispatch an arm whose labels exist"
    )
    ap.add_argument("--force", action="store_true", help="override the wave-2 trigger refusal")
    ap.add_argument(
        "--apply-marking",
        action="store_true",
        help="decide-mark: additionally patch feature_table_v1.jsonl (additive) + sidecar",
    )
    ap.add_argument("--skip-taxonomy", action="store_true", help="skip the raw-text drop scan")
    ap.add_argument(
        "--evidence-shard-limit",
        type=int,
        default=0,
        help="bounded census (smoke; non-default out-root only)",
    )
    ap.add_argument(
        "--raw-shard-limit",
        type=int,
        default=0,
        help="bounded taxonomy scan (smoke; non-default out-root only)",
    )
    args = ap.parse_args()
    if args.phase == "arms" and not (args.render_only or args.smoke or args.full or args.dry_run):
        ap.error("--phase arms needs one of --render-only | --smoke | --full | --dry-run")
    if args.phase == "sample-diag":
        return phase_sample_diag(args)
    if args.phase == "arms":
        return run_arms(args, ARM_SEES, wave=1)
    if args.phase == "analyze":
        return phase_analyze(args)
    if args.phase == "wave2":
        if not (args.full or args.render_only or args.dry_run):
            ap.error("--phase wave2 needs --full (or --render-only/--dry-run)")
        return phase_wave2(args)
    if args.phase == "decide-mark":
        return phase_decide_mark(args)
    raise AssertionError(args.phase)


if __name__ == "__main__":
    sys.exit(main())
