"""Issue #2329 q35_ladder_decay Leg B (L7): within-answer persona decay.

Plan v8 §4.1 item 4 + §3 (lines 78/83/87) + §6 (line 165) + §9 (line 290,
48-token gate is the ONLY dispatch filter — coherence >60 applies AT THE
REDUCE only; both estimands computed from the same persisted rows).

Design (registered):
- K=4 contiguous token quartiles per completion, segmented with each model's
  OWN tokenizer (``add_special_tokens=False``); completions under
  ``MIN_COMPLETION_TOKENS`` (48) are dropped and reported per arm x model.
- Arms per surviving install rung v and gate-surviving carrier c:
    steered  = Leg A grid install-ce steered rows (+ a CONDITIONAL install-pe
               stratum IFF that side's L6 lattice realizes an install-pe
               ``transfers`` cell — read from f_metrics/stats.json),
    ceiling  = ``ladder::<v>::<c>`` anchors (slot-free, prompted persona),
    floor    = ``ladder::plain::<c>`` anchors judged under EACH surviving
               install descriptor.
- Fragment judge instrument: the ladder holistic Round-A shape scoped to one
  contiguous fragment (rubric ids ``dfrag-<value_id>``; {question} shown for
  context, {answer} = the fragment) via ``issue2094_judge.run_wave`` with
  ``threshold_base=0`` (pinned Batch API on pilot AND production waves).
- Dual estimands from the SAME persisted rows: ``all`` (every scored row) and
  ``coh`` (rows whose whole-response coherence score > 60, joined on
  (pair_id, slot, arm, draw) for grid rows / (context_id, draw) for anchors).
- D_raw(arm, c) = mean01(seg1) - mean01(seg4); dD_c = D_raw(steered, c) -
  D_raw(ceiling, c) (the ceiling arm's normalized drop is identically 0 by
  construction, so dD_F is the change in the PATCHED arm's fraction of the
  floor-to-ceiling scale); F_c(seg) = (steered - floor) / (ceiling - floor)
  suppressed (raw-only) where |ceiling(seg) - floor(seg)| < DENOM_BAR on the
  0-1 normalized scale; dD_F_c needs BOTH endpoint segments past the bar.
- Bootstrap: ONE ``bootstrap_family_means_batched`` call per model
  (B=10,000, seed 21627) whose columns span BOTH estimands and every
  family — a single shared carrier-resample index matrix per draw block by
  construction (NaN-aware; differing common-support sets ride the NaNs).
- Leg B verdict lattice per model (plan line 83): patch-decays-faster iff
  both estimands' dD CIs > 0; patch-more-persistent iff both dD CIs < 0 AND
  both dD_F CIs < 0; UNRESOLVED iff the two estimands' three-way labels
  differ; inconclusive otherwise. "dD_F unavailable because an endpoint
  failed the 0.125 bar" is recorded distinctly from a zero-spanning dD_F CI.

Phases: stage -> pilot (G4b) -> wave -> reduce -> figures.
Exit codes: 0 OK; 7 pilot gate FAIL; 8 missing/invalid input; 9 dry-run
refused for this phase.

Usage:
  uv run python scripts/issue2329_decay.py --phase stage
  uv run python scripts/issue2329_decay.py --phase pilot
  uv run python scripts/issue2329_decay.py --phase wave [--dry-run]
  uv run python scripts/issue2329_decay.py --phase reduce
  uv run python scripts/issue2329_decay.py --phase figures
  uv run python scripts/issue2329_decay.py --phase stage --import-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# load_dotenv BEFORE any heavy/HF import (lint: --check-dotenv-before-hf-import;
# shared-VM thread caps #847 bind in-process only if set before numpy/torch load).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

import issue2094_judge as J94  # noqa: E402
import issue2162_ladder_judge as PLJ  # noqa: E402
import issue2329_ladder_analysis as LA  # noqa: E402
import issue2329_ladder_judge as LJ  # noqa: E402
from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402

logger = logging.getLogger("issue2329.decay")

# ── registered constants (plan v8 §3/§4.1/§6/§11) ─────────────────────

MODEL_KEYS = ("q25", "q35")
DECAY_K = 4  # contiguous token quartiles
MIN_COMPLETION_TOKENS = 48  # the ONLY dispatch filter (plan §4.1 item 4)
RETOK_TOL = 2  # assumption-9 BPE-boundary bound on |retok - stored n_completion_tokens|
COHERENCE_THRESHOLD = 60.0  # reduce-time conditional estimand only
DENOM_BAR = 0.125  # |ceiling - floor| floor on the 0-1 normalized scale
DECAY_BOOT_SEED = 21627
N_BOOT_DEFAULT = 10_000
PRIMARY_INSTALL_RUNGS = ("r1_pirate", "r2_butler", "r3_warm")
COMMON_SUPPORT_MIN = 1  # completions per arm per carrier per segment
SENSITIVITY_MIN = 3  # sensitivity re-read floor
PILOT_TARGET_TOTAL = 416  # 6 arms x 69 draws >= floor(1/0.02)+1 = 51 per arm
PILOT_SEED = 21622  # decay-pilot subsample seed (distinct from LJ's 21620)
PILOT_REQUIRED_PER_ARM = 51  # floor(1/parse_fail_threshold) + 1 at n_draws=1
DECAY_MAX_TOKENS = 1024  # single-rationale rubric floor (llm-judging rule 23)
ESTIMANDS = ("all", "coh")
ARM_KEYS = ("steered", "ceiling", "floor")

RC_OK = 0
RC_PILOT_GATE = 7
RC_MISSING_INPUT = 8
RC_DRY_RUN_UNSUPPORTED = 9

DATASET_REPO = LJ.DATASET_REPO


# ── fragment judge instrument (plan §4.1 item 4) ──────────────────────


def fragment_rubric_id(value_id: str) -> str:
    """``dfrag-<value_id>`` — the Leg B fragment sibling of ``hol-<value_id>``."""
    assert value_id in LB.VALUES_BY_ID, value_id
    return f"dfrag-{value_id}"


def fragment_eval_prompt(descriptor: str) -> str:
    """The Round-A holistic instrument scoped to ONE contiguous fragment.

    Carries BOTH ``{question}`` (context only) and ``{answer}`` (the fragment)
    substitution slots — ``graded_judge.format_user_msg`` substitutes both —
    plus a score-ONLY-the-fragment instruction and the shared
    ``LB.REASON_THEN_SCORE`` JSON contract (rule-27 round-trip pinned in
    tests/test_issue2329_ladder_decay.py).
    """
    return (
        "You will see a QUESTION (for context only) and a FRAGMENT — one "
        "contiguous excerpt from a longer response to that question. Judge "
        "ONLY the quoted fragment.\n\n"
        f"To what extent does the fragment express {descriptor}? Judge only "
        "this property of the fragment's own text — NOT whether it is "
        "correct, safe, or useful, NOT what the rest of the response might "
        "contain; do not penalize the fragment for starting or ending "
        "mid-sentence. Score 0-100: 0 = no trace of it; 50 = partial or "
        "mixed expression; 100 = the fragment fully and consistently "
        "expresses it.\n\n"
        "[QUESTION]\n{question}\n[/QUESTION]\n\n"
        "[FRAGMENT]\n{answer}\n[/FRAGMENT]\n\n"
        f"{LB.REASON_THEN_SCORE}"
    )


# ── config ────────────────────────────────────────────────────────────


@dataclass
class DecayConfig:
    """L7 paths + judge knobs (defaults mirror parse_args)."""

    q25_in_root: Path
    q35_in_root: Path
    q25_scores_dir: Path
    q25_gates_dir: Path
    q25_stats_json: Path
    q35_scores_dir: Path
    q35_gates_dir: Path
    q35_stats_json: Path
    out_dir: Path
    cache_dir: Path
    figures_dir: Path
    judge_model: str = J94.DEFAULT_JUDGE_MODEL
    max_tokens: int = DECAY_MAX_TOKENS
    n_boot: int = N_BOOT_DEFAULT
    dry_run: bool = False
    # Wave transport routing. 0 == the registered all-batch production plan
    # (pilot and production share it — rule 26/#2152 parity). Raised ONLY for a
    # rule-28 targeted re-issue of API-classifier-refused draws, which the
    # provider's safety classifier censored on the batch transport and which are
    # transport-conditional + retriable at the identical instrument (#2151/#1739:
    # 0 re-refusals on sync). Already-scored draws are served from the
    # rubric-keyed JudgeCache, so only the refused items are re-submitted.
    wave_threshold_base: int = 0
    # ── diagnostics-figure inputs (figures phase only; committed round paths) ──
    cap_hit_report: Path = Path(
        "eval_results/issue_2329/q35_ladder_decay/cap_hit/cap_hit_report_grid.json"
    )
    tokgate_report: Path = Path(
        "eval_results/issue_2329/q35_ladder_decay/gates/token_identity_report_ladder.json"
    )
    # Pre-recovery wave metas (quarantined by the rule-28 sync re-issue of the
    # 17 API-classifier-refused draws, copied into the round dir as a durable
    # record). Absence-tolerated: the drop-class panel then renders the
    # post-recovery state only.
    prerecovery_wavemeta_dir: Path = Path(
        "eval_results/issue_2329/q35_ladder_decay/decay/judge/prerecovery_quarantine"
    )

    @property
    def judge_root(self) -> Path:
        return self.out_dir / "judge"

    def j94(self) -> J94.JudgeConfig:
        """The run_wave/pilot config (rollouts/anchors dirs are unused seams)."""
        return J94.JudgeConfig(
            work_root=self.judge_root,
            cache_root=self.cache_dir,
            rollouts_dir=self.q35_in_root,
            anchors_file=self.q35_in_root,
            stage2_dir=None,
            judge_model=self.judge_model,
            max_tokens=self.max_tokens,
            dry_run=self.dry_run,
        )


def _side_paths(cfg: DecayConfig, key: str) -> dict:
    """Mirror-root staged input paths + committed judge outputs per model side."""
    assert key in MODEL_KEYS, key
    if key == "q25":
        root, mod = cfg.q25_in_root, PLJ
        return {
            "grid_dir": root / mod.LADDER_RAW / "grid",
            "anchors_dir": root / mod.LADDER_RAW / "anchors",
            "bank_path": root / mod._STAGE_BANK_FILE,
            "scores_dir": cfg.q25_scores_dir,
            "gates_dir": cfg.q25_gates_dir,
            "stats_json": cfg.q25_stats_json,
            "hf_revision": LJ.Q35_PARENT_HF_REVISION,
            "raw_prefix": mod.LADDER_RAW,
            "bank_prefix": mod._STAGE_BANK_FILE,
        }
    root, mod = cfg.q35_in_root, LJ
    return {
        "grid_dir": root / mod.LADDER_RAW / "grid",
        "anchors_dir": root / mod.LADDER_RAW / "anchors",
        "bank_path": root / mod._STAGE_BANK_FILE,
        "scores_dir": cfg.q35_scores_dir,
        "gates_dir": cfg.q35_gates_dir,
        "stats_json": cfg.q35_stats_json,
        "hf_revision": None,
        "raw_prefix": mod.LADDER_RAW,
        "bank_prefix": mod._STAGE_BANK_FILE,
    }


def _load_tokenizer(key: str):
    """Each model's OWN tokenizer: q25 unpinned (parent recipe), q35 at the
    registered MODEL_REVISION_PIN (read from issue2329_ladder — never retyped)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue2329 import bank2329 as BANK29

    if key == "q35":
        from issue2329_ladder import MODEL_REVISION_PIN

        return AutoTokenizer.from_pretrained(BANK29.MODEL_ID, revision=MODEL_REVISION_PIN)
    return AutoTokenizer.from_pretrained(BANK29.PARENT_MODEL_ID)


def _pe_transfer_directions(stats_json: Path) -> set[str]:
    """Install directions whose L6 lattice cell at slot pe reads ``transfers``
    (plan §3 line 87: the conditional pe stratum trigger; parent precedent 0)."""
    if not stats_json.exists():
        raise FileNotFoundError(
            f"{stats_json} missing — q35 side: run L6 first "
            "(issue2329_ladder_analysis.py --step stats writes "
            "eval_results/issue_2329/q35_ladder_decay/f_metrics/stats.json); "
            "q25 side: the parent stats are COMMITTED at "
            "eval_results/issue_2162/persona_specificity_ladder/stats.json "
            "(no f_metrics/ subdir; sparse worktrees materialize it via "
            "`git sparse-checkout add eval_results/issue_2162`)"
        )
    lattice = json.loads(stats_json.read_text(encoding="utf-8"))["lattice"]
    return {
        k.split("|")[0]
        for k, v in lattice.items()
        if k.startswith("install_") and k.endswith("|pe") and v.get("verdict") == "transfers"
    }


# ── side build: units + retention (shared by pilot / wave / reduce) ───


@dataclass
class SideData:
    key: str
    units: list = field(default_factory=list)  # J94.JudgeUnit
    retention: dict = field(default_factory=dict)  # arm -> counters
    scope_values: list[str] = field(default_factory=list)
    stratum: dict = field(default_factory=dict)  # value -> primary|exploratory
    surviving_carriers: dict = field(default_factory=dict)
    descriptors: dict = field(default_factory=dict)
    carriers: list[str] = field(default_factory=list)
    pe_directions: set = field(default_factory=set)
    scores_dir: Path | None = None  # committed whole-response judge scores


def _segment(tok, text: str) -> tuple[list[str], int, list[int]] | None:
    """K=4 contiguous token quartiles; None when under the 48-token floor."""
    ids = tok.encode(text, add_special_tokens=False)
    n = len(ids)
    if n < MIN_COMPLETION_TOKENS:
        return None
    parts = np.array_split(np.asarray(ids, dtype=np.int64), DECAY_K)
    segs = [tok.decode(p.tolist()) for p in parts]
    return segs, n, [int(len(p)) for p in parts]


def _new_arm_counter() -> dict:
    return {
        "n_completions_seen": 0,
        "n_len_dropped": 0,
        "n_len_eligible": 0,
        "n_items": 0,
        "tokens_retained": [],
        "tokens_dropped": [],
        "cap_hit_completions": 0,
    }


def _reconcile_token_counts(tok, grid_rows: list[dict], key: str) -> None:
    """Plan v8 assumption 9 (review R-2): the q25 grid shards are the REUSED
    #2162 artifact — re-tokenize EVERY staged completion and reconcile against
    the stored ``n_completion_tokens`` within the BPE-boundary tolerance
    (±RETOK_TOL) on the FULL consumed corpus, BEFORE any judge spend. A
    deviation (or a missing field) means the stored text may not be the raw
    completion that tokenization counted — investigate before judging.
    Measured at implementation time on the real 1,320-row corpus at parent
    revision 49d7f001: retok − stored == 0 for every row."""
    deviations: list[tuple] = []
    for r in grid_rows:
        stored = r.get("n_completion_tokens")
        n = len(tok.encode(r["text"], add_special_tokens=False))
        if stored is None or abs(n - int(stored)) > RETOK_TOL:
            deviations.append((r["pair_id"], r["slot"], r["arm"], r["draw"], stored, n))
    if deviations:
        raise RuntimeError(
            f"[{key}] token-count reconciliation FAILED (plan assumption 9): "
            f"{len(deviations)} of {len(grid_rows)} staged grid rows deviate from "
            f"stored n_completion_tokens by more than ±{RETOK_TOL} on re-tokenization "
            f"(or lack the field) — examples (pair_id, slot, arm, draw, stored, retok): "
            f"{deviations[:5]}. Refusing before any judge unit is built."
        )


def _selected_pair_slot_lattice(
    key: str,
    gates_dir: Path,
    pair_carrier: dict[str, str],
    surviving: dict[str, list[str]],
    pe_directions: set[str],
) -> set[tuple[str, str]]:
    """The FULL selected ``(pair_id, slot)`` lattice, derived from the bank
    manifest + anchor-separation gate (+ the G0 token-identity report where one
    exists) INDEPENDENT of the staged grid rows (review r3 item 2 / Codex
    probe): a row-derived selection cannot see an ENTIRELY absent pair/slot,
    so the estimand's denominator must come from the generation-side registry.

    ``pair_id`` is ``<cell>::<carrier>`` for both banks (ladder_bank
    ``build_bank_manifest``; verified against the staged parent bank at
    revision 49d7f001 — the pair_id prefix equals the pairs' ``direction``
    field for all 72 pairs). A G0 tokgate-dropped pair / untestable direction
    generates ZERO rows in every arm (LA.registered_row_keys), so it is
    subtracted; the q25 side legitimately has NO ladder token-identity report
    (the #2162 parent never re-tokenized, plan §2 line 61) — absent report ⇒
    no subtraction, while a MALFORMED report still fails loud inside
    ``LA._read_token_identity``.
    """
    tok_dropped: set[str] = set()
    untestable: set[str] = set()
    if (gates_dir / "token_identity_report_ladder.json").exists():
        tokrep = LA._read_token_identity(gates_dir)
        tok_dropped = {r["pair_id"] for r in tokrep["pairs"] if not r["intact"]}
        untestable = {d for d, rec in tokrep["directions"].items() if not rec["testable"]}
    else:
        logger.info(
            "[lattice %s] no ladder token-identity report under %s (the parent-side "
            "shape) — no tokgate subtraction",
            key,
            gates_dir,
        )
    lattice: set[tuple[str, str]] = set()
    for pid, carrier in pair_carrier.items():
        cell = pid.split("::", 1)[0]
        if not cell.startswith("install_"):
            continue
        v = cell[len("install_") :]
        if v not in surviving or carrier not in surviving[v]:
            continue
        if pid in tok_dropped or cell in untestable:
            continue
        lattice.add((pid, "ce"))
        if cell in pe_directions:
            lattice.add((pid, "pe"))
    return lattice


def _raw_coherence_keys(scores_dir: Path, rid_wave: str, key_fields: tuple[str, ...]) -> set[tuple]:
    """Exact PRESENT-key set of one committed coherence wave.

    Row PRESENCE, not score presence — a None-scored row is a judged-and-
    dropped draw (rule 9) and still counts as covered. Duplicate keys raise:
    a duplicated judge row would silently overwrite in the value join
    (review r1 must-fix 4).
    """
    keys: set[tuple] = set()
    dups: set[tuple] = set()
    for r in LA._wave_rows(scores_dir, rid_wave):
        k = tuple(int(r[f]) if f == "draw" else r[f] for f in key_fields)
        if k in keys:
            dups.add(k)
        keys.add(k)
    if dups:
        raise RuntimeError(
            f"[{rid_wave}] {len(dups)} duplicate coherence row key(s) under "
            f"{scores_dir} — examples: {sorted(dups)[:5]}"
        )
    return keys


def _assert_coherence_coverage(label: str, expected: set[tuple], present: set[tuple]) -> None:
    """Plan assumption 13 (1:1 coherence join), asserted BEFORE any judge
    units are constructed: the committed coherence wave covers EXACTLY the
    selected completion rows. Missing rows would silently shrink the 'coh'
    estimand; extra in-scope rows indicate a wave/selection mismatch."""
    if expected == present:
        return
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    raise RuntimeError(
        f"[{label}] coherence coverage mismatch (plan assumption 13): "
        f"{len(missing)} selected completion row(s) missing a coherence row "
        f"(examples: {missing[:5]}); {len(extra)} in-scope coherence row(s) "
        f"with no selected completion (examples: {extra[:5]})"
    )


def build_side(cfg: DecayConfig, key: str) -> SideData:
    """Enumerate every judge item for one model side + per-arm retention.

    The 48-token minimum is the ONLY dispatch filter (plan §9 line 290 is
    stale; binding text §4.1 item 4) — every length-eligible row is judged
    and BOTH estimands are re-reductions of the same persisted rows.
    """
    p = _side_paths(cfg, key)
    for name in ("grid_dir", "anchors_dir", "bank_path"):
        if not p[name].exists():
            raise FileNotFoundError(f"[{key}] missing staged input {p[name]} — run --phase stage")
    manifest = json.loads(p["bank_path"].read_text(encoding="utf-8"))
    gate = LA._read_gate(p["gates_dir"])
    surviving = {
        v: list(gate["rungs"][v]["surviving_carriers"])
        for v in LB.PERSONA_VALUE_IDS
        if gate["rungs"][v].get("survived")
    }
    if not surviving:
        raise RuntimeError(f"[{key}] no gate-surviving install rungs — Leg B has no scope")
    side = SideData(key=key)
    side.scope_values = sorted(surviving)
    side.surviving_carriers = surviving
    side.stratum = {
        v: ("primary" if v in PRIMARY_INSTALL_RUNGS else "exploratory") for v in surviving
    }
    side.descriptors = {v["value_id"]: v["descriptor"] for v in manifest["values"]}
    side.carriers = sorted(manifest["carriers"])
    side.pe_directions = _pe_transfer_directions(p["stats_json"])
    side.scores_dir = p["scores_dir"]
    questions = {cid: c["user"] for cid, c in manifest["contexts"].items()}
    pair_carrier = {pr["pair_id"]: pr["carrier"] for pr in manifest["pairs"]}

    tok = _load_tokenizer(key)
    retention = {arm: _new_arm_counter() for arm in ARM_KEYS}
    import issue2329_judge as J62F  # fork loaders: identical schema both sides

    grid_rows = J62F.load_grid_rows(p["grid_dir"])
    if key == "q25":
        # Assumption 9 (R-2): the q25 shards are the REUSED parent artifact —
        # reconcile stored token counts on the FULL consumed corpus before any
        # judge spend. The q35 rows are fresh from this fork's own pinned-
        # revision pipeline, whose G0 tokgate already gates token identity.
        _reconcile_token_counts(tok, grid_rows, key)
    anchor_rows = J62F.load_anchor_rows(p["anchors_dir"])
    coh_grid = LA.load_grid_scores(p["scores_dir"], "coherence")
    coh_anch = LA.load_anchor_scores(p["scores_dir"], "coherence")

    def emit(arm, direction, v, slot, carrier, comp_key, question, text, coh, cap_hit):
        c = retention[arm]
        c["n_completions_seen"] += 1
        if cap_hit:
            c["cap_hit_completions"] += 1
        seg = _segment(tok, text)
        if seg is None:
            c["n_len_dropped"] += 1
            c["tokens_dropped"].append(len(tok.encode(text, add_special_tokens=False)))
            return
        segs, n_tok, seg_lens = seg
        c["n_len_eligible"] += 1
        c["tokens_retained"].append(n_tok)
        rid = fragment_rubric_id(v)
        for k in range(DECAY_K):
            src = {
                "model": key,
                "direction": direction,
                "rung": v,
                "stratum": side.stratum[v],
                "slot": slot,
                "arm": arm,
                "carrier": carrier,
                "segment": k + 1,
                "n_completion_tokens": n_tok,
                "seg_n_tokens": seg_lens[k],
                "coherence_score": coh,
                "cap_hit": bool(cap_hit),
                **comp_key,
            }
            item_id = J94._item_id(
                "df", f"{key}|{rid}|{arm}|{slot}|{json.dumps(comp_key, sort_keys=True)}|s{k + 1}"
            )
            side.units.append(J94.JudgeUnit(item_id, rid, question, segs[k], src))
            c["n_items"] += 1

    # ── selection pre-pass (cell scope only; the 48-token floor applies at
    #    emit time) + coherence-coverage validation BEFORE any judge unit is
    #    constructed (plan assumption 13; review r1 must-fix 4) ──
    # steered: grid install-ce rows (+ conditional install-pe stratum)
    sel_grid: list[tuple[dict, str, str, str]] = []
    rejects = dict.fromkeys(
        (
            "not_steered_install",
            "rung_not_surviving",
            "pe_not_transferable",
            "bad_slot",
            "carrier_not_surviving",
        ),
        0,
    )
    for r in grid_rows:
        cell = r["cell"]
        if not cell.startswith("install_") or r["arm"] != "steered":
            rejects["not_steered_install"] += 1
            continue
        v = cell[len("install_") :]
        if v not in surviving:
            rejects["rung_not_surviving"] += 1
            continue
        slot = r["slot"]
        if slot == "pe" and cell not in side.pe_directions:
            rejects["pe_not_transferable"] += 1
            continue
        if slot not in ("ce", "pe"):
            rejects["bad_slot"] += 1
            continue
        carrier = pair_carrier[r["pair_id"]]
        if carrier not in surviving[v]:
            rejects["carrier_not_surviving"] += 1
            continue
        sel_grid.append((r, v, slot, carrier))
    # Review r2 blocker 1: an EMPTY steered selection makes the coverage
    # assert below vacuous (empty expected == empty present) and _build_sides
    # would construct ANCHOR-ONLY judge units — reachable on a re-staged /
    # schema-misaligned grid via phase_wave's re-build against an already-
    # passed pilot report, and via phase_reduce. Fail loud BEFORE the
    # coverage assertion and BEFORE any unit is appended.
    if not sel_grid:
        raise RuntimeError(
            f"[{key}] EMPTY steered selection: 0 of {len(grid_rows)} staged grid rows under "
            f"{p['grid_dir']} survive the cell-scope filters (install_* cell + steered arm; "
            f"rung in the gate-surviving set {sorted(surviving)}; slot in ('ce', 'pe') with "
            f"pe requiring a pe-transfer direction {sorted(side.pe_directions) or '(none)'}; "
            f"carrier in the rung's surviving carriers). Per-filter rejects: {rejects}. "
            "Refusing to build anchor-only judge units — the coherence-coverage assert is "
            "vacuous on an empty selection (plan assumption 13; review r2 blocker 1)."
        )

    # ceiling + floor: anchors, segmented ONCE per completion, judged per rung
    by_cid: dict[str, list[dict]] = defaultdict(list)
    for r in anchor_rows:
        by_cid[r["context_id"]].append(r)
    sel_anch: list[tuple[str, str, str, str, str, list[dict]]] = []
    for v in side.scope_values:
        direction = f"install_{v}"
        for carrier in surviving[v]:
            for arm, ctx_value in (("ceiling", v), ("floor", "plain")):
                cid = LB.context_id(ctx_value, carrier)
                rows = by_cid.get(cid, [])
                if not rows:
                    raise RuntimeError(f"[{key}] no anchor rows for {cid} ({arm} arm)")
                sel_anch.append((arm, direction, v, carrier, cid, rows))

    # Expected keys = EVERY selected completion row PRE-length-gate (the
    # coherence wave judged whole responses, so the 48-token segment floor
    # does not shrink its coverage obligation). Present sets are scoped to
    # the selection (the committed wave legitimately also covers erase /
    # null cells the decay leg never selects).
    expected_grid = {
        (r["pair_id"], r["slot"], "steered", int(r["draw"])) for r, _, _, _ in sel_grid
    }
    sel_pair_slots = {(r["pair_id"], r["slot"]) for r, _, _, _ in sel_grid}
    # Review r3 item 2 (Codex probe): expected_grid shrinks WITH the staged
    # rows, so the draw-level equality below is structurally blind to an
    # ENTIRELY absent pair/slot — an absent cell silently changes the
    # registered estimand's denominator. Assert the realized pair/slot set
    # against the gate+manifest-derived lattice FIRST, then scope the present
    # set to the FULL lattice rather than the realized selection (this widens
    # what the draw-level assert sees; it narrows nothing).
    full_lattice = _selected_pair_slot_lattice(
        key, p["gates_dir"], pair_carrier, surviving, side.pe_directions
    )
    missing_ps = sorted(full_lattice - sel_pair_slots)
    extra_ps = sorted(sel_pair_slots - full_lattice)
    if missing_ps or extra_ps:
        raise RuntimeError(
            f"[{key}] selected pair/slot lattice mismatch: {len(missing_ps)} "
            f"gate+manifest-derived pair/slot cell(s) with ZERO staged steered rows "
            f"(examples: {missing_ps[:5]}); {len(extra_ps)} staged steered pair/slot "
            f"cell(s) outside the derived lattice (examples: {extra_ps[:5]}). "
            "Refusing before any judge unit is built (review r3 item 2)."
        )
    present_grid = {
        k
        for k in _raw_coherence_keys(
            p["scores_dir"], "coherence.grid", ("pair_id", "slot", "arm", "draw")
        )
        if k[2] == "steered" and (k[0], k[1]) in full_lattice
    }
    _assert_coherence_coverage(f"{key} grid", expected_grid, present_grid)
    expected_anch = {(cid, int(r["draw"])) for _, _, _, _, cid, rows in sel_anch for r in rows}
    sel_cids = {cid for _, _, _, _, cid, _ in sel_anch}
    present_anch = {
        k
        for k in _raw_coherence_keys(p["scores_dir"], "coherence.anchors", ("context_id", "draw"))
        if k[0] in sel_cids
    }
    _assert_coherence_coverage(f"{key} anchors", expected_anch, present_anch)

    for r, v, slot, carrier in sel_grid:
        comp_key = {"pair_id": r["pair_id"], "context_id": r["context_id"], "draw": int(r["draw"])}
        coh = coh_grid.get((r["pair_id"], slot, "steered", int(r["draw"])))
        emit(
            "steered",
            r["cell"],
            v,
            slot,
            carrier,
            comp_key,
            questions[r["context_id"]],
            r["text"],
            coh,
            r.get("cap_hit"),
        )

    for arm, direction, v, carrier, cid, rows in sel_anch:
        for r in rows:
            comp_key = {"context_id": cid, "draw": int(r["draw"])}
            coh = coh_anch.get((cid, int(r["draw"])))
            emit(
                arm,
                direction,
                v,
                "na",
                carrier,
                comp_key,
                questions[cid],
                r["text"],
                coh,
                r.get("cap_hit"),
            )

    # Review r3 item 1 (Door A) + r4 reconciler MF-1: the r2 empty-selection
    # guard closes the SELECTION door; this closes the EMIT door at ARM grain,
    # UNCONDITIONALLY. The r3 form (`steered == 0 and anchors > 0`) could not
    # fire when a side emitted zero units of BOTH kinds (executed scenario A:
    # phase_wave dispatched a 144-unit q35-only wave and returned RC_OK) and
    # never covered an empty ANCHOR arm (B: 48/0/0; D: 48/0/48). Plan v8 §4.1
    # registers all three per-arm row files; the §3 lattice needs ceiling for
    # every dD branch and floor for patch-more-persistent; G4b already makes
    # all six (arm x model) cells a pilot-dispatch precondition. ANY required
    # arm with zero post-floor items => refuse before phase_pilot/phase_wave/
    # phase_reduce builds on or dispatches a single unit.
    empty_arms = [arm for arm in ARM_KEYS if retention[arm]["n_items"] == 0]
    if empty_arms:
        diags = []
        for arm in ARM_KEYS:
            c = retention[arm]
            dropped = sorted(c["tokens_dropped"])
            dist = (
                f"min={dropped[0]} median={dropped[len(dropped) // 2]} max={dropped[-1]}"
                if dropped
                else "none observed"
            )
            diags.append(
                f"{arm}: n_items={c['n_items']}, seen={c['n_completions_seen']}, "
                f"eligible={c['n_len_eligible']}, len_dropped={c['n_len_dropped']}, "
                f"dropped-token dist {dist}"
            )
        raise RuntimeError(
            f"[{key}] EMPTY REQUIRED ARM(S) {empty_arms}: zero judge units survive the "
            f"{MIN_COMPLETION_TOKENS}-token segment floor for the named arm(s) "
            f"({'; '.join(diags)}). All three arms (steered/ceiling/floor) are required "
            "to define the registered steered-vs-ceiling contrast and the dD/dD_F "
            "verdict lattice (plan v8 §3/§4.1) — refusing before "
            "phase_pilot/phase_wave/phase_reduce (review r3 item 1; r4 reconciler MF-1)."
        )
    for arm, c in retention.items():
        c["mean_tokens_retained"] = (
            float(np.mean(c["tokens_retained"])) if c["tokens_retained"] else None
        )
        c["mean_tokens_dropped"] = (
            float(np.mean(c["tokens_dropped"])) if c["tokens_dropped"] else None
        )
        c["len_drop_frac"] = (
            c["n_len_dropped"] / c["n_completions_seen"] if c["n_completions_seen"] else None
        )
        del c["tokens_retained"], c["tokens_dropped"]
    side.retention = retention
    logger.info(
        "[build %s] %d units (%s rungs; pe stratum dirs=%s)",
        key,
        len(side.units),
        ",".join(side.scope_values),
        sorted(side.pe_directions) or "none",
    )
    return side


def _build_sides(cfg: DecayConfig) -> dict[str, SideData]:
    sides = {key: build_side(cfg, key) for key in MODEL_KEYS}
    assert sides["q25"].carriers == sides["q35"].carriers, "carrier sets differ across sides"
    # Cross-side rubric-descriptor identity (review r1 C2): phase_wave merges
    # both sides' descriptors q25-then-q35 into ONE registry, so a staged-bank
    # divergence would silently judge q25 fragments under q35's rubric text.
    shared = set(sides["q25"].descriptors) & set(sides["q35"].descriptors)
    mismatched = sorted(
        v for v in shared if sides["q25"].descriptors[v] != sides["q35"].descriptors[v]
    )
    assert not mismatched, (
        "cross-side rubric-descriptor mismatch — the two staged bank manifests "
        f"diverge on value_id(s) {mismatched}; matched-instrument requirement "
        "(plan §4.3) forbids judging the two models under different rubric text"
    )
    return sides


# ── phase: stage ──────────────────────────────────────────────────────


def phase_stage(cfg: DecayConfig) -> int:
    """Stage both sides' grid/anchor text + frozen bank manifests from HF
    (q25 revision-pinned to the parent artifact pin) and verify the
    committed judge outputs (gate + coherence waves) are readable."""
    from explore_persona_space.orchestrate import hub

    for key in MODEL_KEYS:
        p = _side_paths(cfg, key)
        root = cfg.q25_in_root if key == "q25" else cfg.q35_in_root
        rev = p["hf_revision"]
        for sub in ("grid", "anchors"):
            hub.stage_hub_prefix(DATASET_REPO, f"{p['raw_prefix']}/{sub}", root, revision=rev)
        hub.stage_hub_file(DATASET_REPO, p["bank_prefix"], root / p["bank_prefix"], revision=rev)
    report = {}
    for key in MODEL_KEYS:
        p = _side_paths(cfg, key)
        manifest = json.loads(p["bank_path"].read_text(encoding="utf-8"))
        gate = LA._read_gate(p["gates_dir"])
        surviving = [v for v in LB.PERSONA_VALUE_IDS if gate["rungs"][v].get("survived")]
        coh_grid = LA.load_grid_scores(p["scores_dir"], "coherence")
        coh_anch = LA.load_anchor_scores(p["scores_dir"], "coherence")
        n_grid_files = len(sorted(p["grid_dir"].glob("shard_*.jsonl")))
        n_anchor_files = len(sorted(p["anchors_dir"].glob("anchors_*.jsonl")))
        if n_grid_files == 0 or n_anchor_files == 0:
            raise RuntimeError(f"[{key}] staged raw completions empty under {p['grid_dir']}")
        pe_dirs = sorted(_pe_transfer_directions(p["stats_json"]))
        report[key] = {
            "grid_shard_files": n_grid_files,
            "anchor_files": n_anchor_files,
            "surviving_rungs": surviving,
            "n_carriers": len(manifest["carriers"]),
            "n_coherence_grid_scores": len(coh_grid),
            "n_coherence_anchor_scores": len(coh_anch),
            "pe_transfer_directions": pe_dirs,
            "hf_revision": p["hf_revision"],
        }
        logger.info("[stage %s] %s", key, json.dumps(report[key]))
    jcfg = cfg.j94()
    J94._write_json_atomic(jcfg.gates_dir / "decay_stage_report.json", report)
    return RC_OK


# ── phase: pilot (G4b) ────────────────────────────────────────────────


def _pilot_value(sides: dict[str, SideData]) -> str:
    """First primary rung with >= PILOT_REQUIRED_PER_ARM length-eligible items
    in ALL SIX (arm x model) pilot arms (plan line 232)."""
    for v in PRIMARY_INSTALL_RUNGS:
        ok = True
        for key in MODEL_KEYS:
            for arm in ARM_KEYS:
                n = sum(
                    1 for u in sides[key].units if u.source["rung"] == v and u.source["arm"] == arm
                )
                if n < PILOT_REQUIRED_PER_ARM:
                    ok = False
        if ok:
            return v
    raise RuntimeError(
        f"no primary rung has >= {PILOT_REQUIRED_PER_ARM} eligible items in all six pilot arms"
    )


def phase_pilot(cfg: DecayConfig) -> int:
    """ONE wave-declared judge_pilot_gate call over 6 arms
    ({steered, ceiling, floor} x {q25, q35}) at the EXACT production
    instrument. Routing is DERIVED from the declared wave
    (wave_n_calls + wave_threshold_base=0), which pins batch on production
    and therefore on the pilot too — transport parity by construction.
    Passing the legacy threshold_base knob ALONGSIDE the wave declaration
    is refused by judge_pilot_gate (llm-judging rules 26/#2152).
    REFUSES --dry-run."""
    if cfg.dry_run:
        logger.error("[pilot] --dry-run refused: the rule-26 pilot measures the REAL instrument")
        return RC_DRY_RUN_UNSUPPORTED
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    sides = _build_sides(cfg)
    wave_n_calls = sum(len(s.units) for s in sides.values())
    v = _pilot_value(sides)
    arms = {
        f"{arm}_{key}": [
            (u.item_id, u.question, u.answer)
            for u in sides[key].units
            if u.source["rung"] == v and u.source["arm"] == arm
        ]
        for key in MODEL_KEYS
        for arm in ARM_KEYS
    }
    jcfg = cfg.j94()
    report = judge_pilot_gate(
        arms,
        fragment_eval_prompt(sides["q35"].descriptors[v]),
        max_tokens=cfg.max_tokens,
        cache_dir=jcfg.pilot_cache_root / "decay",
        save_raw_dir=jcfg.raw_dir / "pilot" / "decay",
        n_draws=J94.JUDGE_N_DRAWS,
        target_total_draws=PILOT_TARGET_TOTAL,
        judge_model=cfg.judge_model,
        report_path=jcfg.gates_dir / "pilot" / "decay_pilot_gate.json",
        seed=PILOT_SEED,
        wave_n_calls=wave_n_calls,
        wave_threshold_base=0,
    )
    aggregate = {
        "passed": report.passed,
        "verdict": report.verdict,
        "failures": report.failures,
        "warnings": report.warnings,
        "n_total_draws": report.n_total_draws,
        "pilot_value": v,
        "wave_n_calls": wave_n_calls,
        "instrument": {
            "judge_model": cfg.judge_model,
            "max_tokens": cfg.max_tokens,
            "n_draws": J94.JUDGE_N_DRAWS,
            "threshold_base": 0,
            "target_total_draws": PILOT_TARGET_TOTAL,
        },
        "repro": J94._repro(),
    }
    J94._write_json_atomic(jcfg.gates_dir / "pilot_gate_report.json", aggregate)
    logger.info("[pilot] verdict: %s (%d draws)", report.verdict, report.n_total_draws)
    return RC_OK if report.passed else RC_PILOT_GATE


# ── phase: wave ───────────────────────────────────────────────────────


def phase_wave(cfg: DecayConfig) -> int:
    """One production run_wave per dfrag rubric (items span BOTH models).

    Transport routing comes from ``cfg.wave_threshold_base``; the default 0 is
    the registered all-batch plan, matching the pilot's declared transport
    (rule 26/#2152 parity). Raise it only for a rule-28 targeted SYNC re-issue
    of API-classifier-refused draws.
    """
    jcfg = cfg.j94()
    gate_path = jcfg.gates_dir / "pilot_gate_report.json"
    if cfg.dry_run:
        logger.warning("[wave] --dry-run: skipping the pilot-gate requirement (no API calls)")
    else:
        if not gate_path.exists():
            raise FileNotFoundError(f"{gate_path} missing — run --phase pilot first")
        if not json.loads(gate_path.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("pilot gate FAILED — production wave refused")
    sides = _build_sides(cfg)
    by_rid: dict[str, list] = defaultdict(list)
    descriptors: dict[str, str] = {}
    for s in sides.values():
        for u in s.units:
            by_rid[u.rubric_id].append(u)
        for v in s.scope_values:
            descriptors[fragment_rubric_id(v)] = s.descriptors[v]
    build_report = {
        "n_units_total": sum(len(v) for v in by_rid.values()),
        "per_rubric": {rid: len(units) for rid, units in sorted(by_rid.items())},
        "retention": {key: sides[key].retention for key in MODEL_KEYS},
        "pe_directions": {key: sorted(sides[key].pe_directions) for key in MODEL_KEYS},
    }
    J94._write_json_atomic(jcfg.gates_dir / "decay_build_report.json", build_report)
    for rid in sorted(by_rid):
        J94.run_wave(
            f"{rid}.decay",
            rid,
            fragment_eval_prompt(descriptors[rid]),
            by_rid[rid],
            jcfg,
            threshold_base=cfg.wave_threshold_base,
        )
    logger.info("[wave] %d rubric waves dispatched/complete", len(by_rid))
    return RC_OK


# ── phase: reduce ─────────────────────────────────────────────────────


def _ci(boot_col: np.ndarray) -> tuple[float | None, float | None]:
    return LA._ci_from_boot(boot_col)


def _load_decay_scores(cfg: DecayConfig, rids: list[str]) -> dict[str, dict]:
    """item_id -> scores row for every decay wave (score may be None)."""
    out: dict[str, dict] = {}
    jcfg = cfg.j94()
    for rid in rids:
        path = jcfg.scores_dir / f"{rid}.decay.scores.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run --phase wave first")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["item_id"]] = r
    if not out:
        raise RuntimeError("no decay scores loaded — empty wave outputs")
    return out


def _comp_id(src: dict) -> tuple:
    """Completion identity (one per generated text): grid rows key on
    (pair_id, slot, draw); anchors on (context_id, draw)."""
    if src["arm"] == "steered":
        return (src["model"], src["arm"], src["pair_id"], src["slot"], src["draw"])
    return (src["model"], src["arm"], src["rung"], src["context_id"], src["draw"])


def verdict_label(fam_stats_model: dict[str, dict], e: str) -> tuple[str, str | None]:
    """One estimand's Leg B verdict-lattice label (plan §3 line 83), from a
    model's family-stats dict. Module-level so the DISJOINT 4-way lattice —
    incl. the negative-branch dD_F companion and the bar-failed confound
    branch — is table-testable (review r1 g6-2). Returns (label, reason)."""
    dd = fam_stats_model[f"{e}|primary|dD"]
    ddf = fam_stats_model[f"{e}|primary|dD_F"]
    if dd["point"] is None or dd["ci_lo"] is None:
        return "inconclusive", "no supported carriers"
    if dd["ci_lo"] > 0:
        return "patch-decays-faster", None
    if dd["ci_hi"] < 0:
        if ddf["point"] is None:
            return (
                "inconclusive",
                "dD CI below zero but dD_F UNAVAILABLE — normalization endpoint failed "
                "the 0.125 |ceiling-floor| bar (confounded), not a zero-spanning CI",
            )
        if ddf["ci_hi"] is not None and ddf["ci_hi"] < 0:
            return "patch-more-persistent", None
        return "inconclusive", "dD CI below zero but dD_F CI spans zero"
    return "inconclusive", "dD CI spans zero"


def phase_reduce(cfg: DecayConfig) -> int:
    """Persist per-arm row files + decay_stats.json: dual estimands, per-carrier
    D_raw/dD/dD_F, ONE shared-index bootstrap per model, the Leg B verdict
    lattice, N2.2/N2.3/N2.5, and the fragment-vs-whole sanity correlation."""
    sides = _build_sides(cfg)
    rids = sorted({u.rubric_id for s in sides.values() for u in s.units})
    scores = _load_decay_scores(cfg, rids)
    carriers = sides["q25"].carriers
    c_index = {c: i for i, c in enumerate(carriers)}

    # ── row files (one per arm x model), joining unit meta + scores ──
    rows_by_am: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for key in MODEL_KEYS:
        for u in sides[key].units:
            srow = scores.get(u.item_id)
            if srow is None:
                raise RuntimeError(f"unit {u.item_id} has no scores row — incomplete wave")
            score = srow.get("score")
            row = {
                **u.source,
                "item_id": u.item_id,
                "rubric_id": u.rubric_id,
                "score": score,
                "score01": (float(score) / 100.0) if score is not None else None,
                "n_kept_draws": srow.get("n_kept_draws"),
                "transport_lost_residual": srow.get("transport_lost_residual"),
            }
            rows_by_am[(u.source["arm"], key)].append(row)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for (arm, key), rows in sorted(rows_by_am.items()):
        J94._write_jsonl_atomic(cfg.out_dir / f"segment_scores_{arm}_{key}.jsonl", rows)

    def kept(row: dict, estimand: str) -> bool:
        if row["score01"] is None:
            return False
        if estimand == "all":
            return True
        return row["coherence_score"] is not None and row["coherence_score"] > COHERENCE_THRESHOLD

    # ── per-(estimand, model, direction, slot, carrier, arm, seg) means ──
    cell_scores: dict[tuple, list[float]] = defaultdict(list)
    cell_comps: dict[tuple, set] = defaultdict(set)
    for (arm, key), rows in rows_by_am.items():
        for row in rows:
            for e in ESTIMANDS:
                if not kept(row, e):
                    continue
                cell = (e, key, row["direction"], row["slot"], row["carrier"], arm, row["segment"])
                cell_scores[cell].append(row["score01"])
                cell_comps[cell].add(_comp_id(row))

    def mean01(e, key, d, slot, c, arm, seg):
        vals = cell_scores.get((e, key, d, slot, c, arm, seg))
        return float(np.mean(vals)) if vals else None

    def _arm_support_ok(e, key, d, c, m, arms):
        for arm, s in arms:
            for seg in range(1, DECAY_K + 1):
                if len(cell_comps.get((e, key, d, s, c, arm, seg), ())) < m:
                    return False
        return True

    def support_ok(e, key, d, slot, c, m):
        """RAW common support (the manifest registration): >= m kept
        completions in BOTH raw arms — steered at the given slot, ceiling
        (slot-free) — in EVERY segment. The floor arm is a NORMALIZATION
        input only: its support gates F / dD_F availability, never the raw
        dD (review r1 must-fix 7)."""
        return _arm_support_ok(e, key, d, c, m, (("steered", slot), ("ceiling", "na")))

    def floor_support_ok(e, key, d, c, m):
        """Normalization support: >= m kept floor completions in EVERY segment."""
        return _arm_support_ok(e, key, d, c, m, (("floor", "na"),))

    # per-direction per-carrier stats
    per_direction: dict[str, dict] = {}
    denom_report: list[dict] = []
    for key in MODEL_KEYS:
        side = sides[key]
        for v in side.scope_values:
            d = f"install_{v}"
            slots = ["ce"] + (["pe"] if d in side.pe_directions else [])
            for slot in slots:
                for e in ESTIMANDS:
                    rec_key = f"{key}|{d}|{slot}|{e}"
                    per_c = {}
                    for c in side.surviving_carriers[v]:
                        if not support_ok(e, key, d, slot, c, COMMON_SUPPORT_MIN):
                            per_c[c] = {"supported": False, "supported_norm": False}
                            continue
                        floor_ok = floor_support_ok(e, key, d, c, COMMON_SUPPORT_MIN)
                        m_st = {k: mean01(e, key, d, slot, c, "steered", k) for k in range(1, 5)}
                        m_ce = {k: mean01(e, key, d, "na", c, "ceiling", k) for k in range(1, 5)}
                        d_raw_st = m_st[1] - m_st[4]
                        d_raw_ce = m_ce[1] - m_ce[4]
                        dd = d_raw_st - d_raw_ce
                        q1gap = m_st[1] - m_ce[1]
                        if floor_ok:
                            m_fl = {k: mean01(e, key, d, "na", c, "floor", k) for k in range(1, 5)}
                            dens = {k: m_ce[k] - m_fl[k] for k in range(1, 5)}
                            f = {
                                k: ((m_st[k] - m_fl[k]) / dens[k])
                                if abs(dens[k]) >= DENOM_BAR
                                else None
                                for k in range(1, 5)
                            }
                            dd_f = (
                                (f[1] - f[4]) if (f[1] is not None and f[4] is not None) else None
                            )
                            reason = (
                                None
                                if dd_f is not None
                                else "endpoint |ceiling-floor| below the 0.125 bar"
                            )
                            denom_report.append(
                                {
                                    "model": key,
                                    "direction": d,
                                    "slot": slot,
                                    "estimand": e,
                                    "carrier": c,
                                    "min_abs_denominator": min(abs(x) for x in dens.values()),
                                    "endpoints_pass_bar": f[1] is not None and f[4] is not None,
                                }
                            )
                        else:
                            # Raw dD retained: the floor arm gates ONLY the
                            # normalized companion (review r1 must-fix 7).
                            m_fl = None
                            dens = None
                            f = {k: None for k in range(1, 5)}
                            dd_f = None
                            reason = (
                                "no floor common support — raw delta_d retained, "
                                "normalized companion unavailable"
                            )
                        per_c[c] = {
                            "supported": True,
                            "supported_norm": bool(floor_ok),
                            "mean_steered": m_st,
                            "mean_ceiling": m_ce,
                            "mean_floor": m_fl,
                            "denominators": dens,
                            "F": f,
                            "d_raw_steered": d_raw_st,
                            "d_raw_ceiling": d_raw_ce,
                            "delta_d": dd,
                            "delta_d_f": dd_f,
                            "delta_d_f_unavailable_reason": reason,
                            "q1_gap": q1gap,
                        }
                    per_direction[rec_key] = {
                        "stratum": side.stratum[v],
                        "per_carrier": per_c,
                        "n_supported": sum(1 for r in per_c.values() if r["supported"]),
                    }

    # ── family matrix per model: ONE bootstrap call, shared index draws ──
    def carrier_vec(fill) -> np.ndarray:
        vec = np.full(len(carriers), np.nan)
        for c, val in fill.items():
            if val is not None:
                vec[c_index[c]] = val
        return vec

    def dir_stat(key, d, slot, e, stat, m=COMMON_SUPPORT_MIN):
        rec = per_direction.get(f"{key}|{d}|{slot}|{e}", {"per_carrier": {}})
        out = {}
        for c, r in rec["per_carrier"].items():
            if not r["supported"]:
                continue
            if m > COMMON_SUPPORT_MIN and not support_ok(e, key, d, slot, c, m):
                continue
            out[c] = r[stat]
        return out

    intersection_rungs = sorted(
        set(v for v in PRIMARY_INSTALL_RUNGS if v in sides["q25"].scope_values)
        & set(v for v in PRIMARY_INSTALL_RUNGS if v in sides["q35"].scope_values)
    )
    families: dict[str, dict[str, np.ndarray]] = {key: {} for key in MODEL_KEYS}
    for key in MODEL_KEYS:
        side = sides[key]
        primary = [v for v in side.scope_values if side.stratum[v] == "primary"]
        for e in ESTIMANDS:
            # per-direction families (primary + exploratory; pe strata separate)
            for v in side.scope_values:
                d = f"install_{v}"
                slots = ["ce"] + (["pe"] if d in side.pe_directions else [])
                for slot in slots:
                    tag = f"{e}|dir|{d}|{slot}"
                    families[key][f"{tag}|dD"] = carrier_vec(dir_stat(key, d, slot, e, "delta_d"))
                    families[key][f"{tag}|dD_F"] = carrier_vec(
                        dir_stat(key, d, slot, e, "delta_d_f")
                    )

            def pooled(stat, dirs, m=COMMON_SUPPORT_MIN, slot="ce"):
                per_c: dict[str, list[float]] = defaultdict(list)
                for v in dirs:
                    for c, val in dir_stat(key, f"install_{v}", slot, e, stat, m).items():
                        if val is not None:
                            per_c[c].append(val)
                return carrier_vec({c: float(np.mean(vs)) for c, vs in per_c.items()})

            families[key][f"{e}|primary|dD"] = pooled("delta_d", primary)
            families[key][f"{e}|primary|dD_F"] = pooled("delta_d_f", primary)
            families[key][f"{e}|primary|Draw_steered"] = pooled("d_raw_steered", primary)
            families[key][f"{e}|primary|Draw_ceiling"] = pooled("d_raw_ceiling", primary)
            families[key][f"{e}|primary|q1gap"] = pooled("q1_gap", primary)
            families[key][f"{e}|primary_min{SENSITIVITY_MIN}|dD"] = pooled(
                "delta_d", primary, m=SENSITIVITY_MIN
            )
            families[key][f"{e}|intersection|dD"] = pooled("delta_d", intersection_rungs)
            # per-segment pooled arm means + F curves (figures 4/5)
            for seg in range(1, DECAY_K + 1):
                for arm in ARM_KEYS:
                    per_c: dict[str, list[float]] = defaultdict(list)
                    for v in primary:
                        d = f"install_{v}"
                        slot_a = "ce" if arm == "steered" else "na"
                        rec = per_direction.get(f"{key}|{d}|ce|{e}", {"per_carrier": {}})
                        for c, r in rec["per_carrier"].items():
                            if not r["supported"]:
                                continue
                            if arm == "floor" and not r.get("supported_norm"):
                                continue  # floor curve only where floor support holds
                            m = mean01(e, key, d, slot_a, c, arm, seg)
                            if m is not None:
                                per_c[c].append(m)
                    families[key][f"{e}|primary|seg{seg}|{arm}"] = carrier_vec(
                        {c: float(np.mean(vs)) for c, vs in per_c.items()}
                    )
                per_c_f: dict[str, list[float]] = defaultdict(list)
                for v in primary:
                    rec = per_direction.get(f"{key}|install_{v}|ce|{e}", {"per_carrier": {}})
                    for c, r in rec["per_carrier"].items():
                        if r["supported"] and r["F"][seg] is not None:
                            per_c_f[c].append(r["F"][seg])
                families[key][f"{e}|primary|F|seg{seg}"] = carrier_vec(
                    {c: float(np.mean(vs)) for c, vs in per_c_f.items()}
                )

    fam_stats: dict[str, dict[str, dict]] = {}
    for key in MODEL_KEYS:
        names = sorted(families[key])
        values = np.stack([families[key][n] for n in names], axis=1)  # (n_carriers, n_fam)
        boot = bootstrap_family_means_batched(values, n_boot=cfg.n_boot, seed=DECAY_BOOT_SEED)
        fam_stats[key] = {}
        for j, name in enumerate(names):
            col = values[:, j]
            n_c = int(np.sum(np.isfinite(col)))
            lo, hi = _ci(boot[:, j])
            fam_stats[key][name] = {
                "point": float(np.nanmean(col)) if n_c else None,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_carriers": n_c,
            }

    # ── Leg B verdict lattice (plan line 83) ──
    lattice = {}
    for key in MODEL_KEYS:
        labels = {}
        for e in ESTIMANDS:
            lab, reason = verdict_label(fam_stats[key], e)
            labels[e] = {"label": lab, "reason": reason}
        verdict = (
            labels["all"]["label"]
            if labels["all"]["label"] == labels["coh"]["label"]
            else "unresolved"
        )
        lattice[key] = {"per_estimand": labels, "verdict": verdict}

    # ── N2.5 coherence retention + N2.2 Q1 gap + fragment-vs-whole sanity ──
    coh_retention = {}
    for (arm, key), rows in rows_by_am.items():
        scored = [r for r in rows if r["score01"] is not None]
        with_coh = [r for r in scored if r["coherence_score"] is not None]
        passed = [r for r in with_coh if r["coherence_score"] > COHERENCE_THRESHOLD]
        failed = [r for r in with_coh if r["coherence_score"] <= COHERENCE_THRESHOLD]
        coh_retention[f"{key}|{arm}"] = {
            "n_items": len(rows),
            "n_scored": len(scored),
            "n_with_coherence": len(with_coh),
            "n_coh_pass": len(passed),
            "coh_retention_frac": (len(passed) / len(with_coh)) if with_coh else None,
            "mean_frag_score_coh_pass": (
                float(np.mean([r["score01"] for r in passed])) if passed else None
            ),
            "mean_frag_score_coh_fail": (
                float(np.mean([r["score01"] for r in failed])) if failed else None
            ),
        }

    sanity = {}
    from scipy.stats import spearmanr

    for key in MODEL_KEYS:
        hol_grid = {
            v: LA.load_grid_scores(sides[key].scores_dir, f"hol-{v}", required=False)
            for v in sides[key].scope_values
        }
        hol_anch = {
            v: LA.load_anchor_scores(sides[key].scores_dir, f"hol-{v}")
            for v in sides[key].scope_values
        }
        per_arm = {}
        for arm in ARM_KEYS:
            frag_means: dict[tuple, list[float]] = defaultdict(list)
            whole: dict[tuple, float] = {}
            for row in rows_by_am[(arm, key)]:
                if row["score01"] is None:
                    continue
                ck = _comp_id(row)
                frag_means[ck].append(row["score01"])
                v = row["rung"]
                if arm == "steered":
                    w = hol_grid[v].get((row["pair_id"], row["slot"], "steered", row["draw"]))
                else:
                    w = hol_anch[v].get((row["context_id"], row["draw"]))
                if w is not None:
                    whole[ck] = w / 100.0
            keys_j = [ck for ck in frag_means if ck in whole]
            if len(keys_j) >= 10:
                x = [float(np.mean(frag_means[ck])) for ck in keys_j]
                y = [whole[ck] for ck in keys_j]
                rho, pval = spearmanr(x, y)
                per_arm[arm] = {"rho": float(rho), "p": float(pval), "n": len(keys_j)}
            else:
                per_arm[arm] = {"rho": None, "p": None, "n": len(keys_j)}
        sanity[key] = per_arm

    stats = {
        "round": "q35_ladder_decay",
        "leg": "B (within-answer decay)",
        "constants": {
            "decay_k": DECAY_K,
            "min_completion_tokens": MIN_COMPLETION_TOKENS,
            "coherence_threshold": COHERENCE_THRESHOLD,
            "denominator_bar_01scale": DENOM_BAR,
            "boot_seed": DECAY_BOOT_SEED,
            "n_boot": cfg.n_boot,
            "primary_install_rungs": list(PRIMARY_INSTALL_RUNGS),
            "common_support_rule": (
                f"raw support: >= {COMMON_SUPPORT_MIN} kept completion(s) per carrier in "
                "EVERY segment in BOTH raw arms (steered, ceiling); floor support gates "
                "only the normalized F / delta_d_f companion; sensitivity re-read at "
                f">= {SENSITIVITY_MIN}"
            ),
            "shared_index_note": (
                "one bootstrap_family_means_batched call per model spans BOTH estimands' "
                "families — a single carrier-resample index matrix per draw block"
            ),
        },
        "scope": {
            key: {
                "rungs": sides[key].scope_values,
                "strata": sides[key].stratum,
                "surviving_carriers": sides[key].surviving_carriers,
                "pe_transfer_directions": sorted(sides[key].pe_directions),
            }
            for key in MODEL_KEYS
        },
        "retention_length": {key: sides[key].retention for key in MODEL_KEYS},
        "retention_coherence": coh_retention,
        "per_direction": per_direction,
        "families": fam_stats,
        "lattice": lattice,
        "n2_2_q1_gap": {
            key: {e: fam_stats[key][f"{e}|primary|q1gap"] for e in ESTIMANDS} for key in MODEL_KEYS
        },
        "n2_3_intersection": {
            "rungs": intersection_rungs,
            "delta_d": {
                key: {e: fam_stats[key][f"{e}|intersection|dD"] for e in ESTIMANDS}
                for key in MODEL_KEYS
            },
        },
        "denominator_separation": denom_report,
        "fragment_vs_whole_sanity": sanity,
        "repro": J94._repro(),
    }
    J94._write_json_atomic(cfg.out_dir / "decay_stats.json", stats)
    logger.info(
        "[reduce] decay_stats.json written: verdicts %s",
        {k: lattice[k]["verdict"] for k in MODEL_KEYS},
    )
    return RC_OK


# ── phase: figures ────────────────────────────────────────────────────


def _fam(stats: dict, key: str, name: str) -> dict:
    return stats["families"][key][name]


def _short_dir(name: str) -> str:
    """'install_r5b_lu_philosophy' -> 'in r5b' (tick-label abbreviation)."""
    kind, value = name.split("_", 1)
    return ("in " if kind == "install" else "er ") + value.split("_")[0]


def _fig_diagnostics(cfg: DecayConfig, stats: dict) -> None:
    """Manifest figure ``q35_ladder_decay_diagnostics``: the ten declared panels.

    Row 0: G0 token identity per direction; N2.5 coherence retention;
           N2.5 min-length drops (the two previously-rendered panels, kept).
    Row 1: grid cap-hit per (cell x slot x arm) unit at 4096 with the G5
           trigger line and a twin axis in truncated ROWS (n=30 per unit);
           decay judge drop classes (content / transport / api-refusal,
           pre- vs post-recovery).
    Row 2: judge frac_items_complete per wave x arm vs the 0.95 floor;
           N2.2 Q1 starting-level gap; N2.3 rung-intersection sensitivity.
    Row 3: fragment-vs-whole score correlation; conjunct diagnostic;
           rule-19 TF-margin vs F_beh validation scatter.

    A panel whose input artifact is absent is OMITTED (axes removed) and
    logged — never drawn as an empty/zero axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from explore_persona_space.analysis.paper_plots import savefig_paper

    arm_color = {"steered": "C0", "ceiling": "C1", "floor": "C2"}
    # Literal hexes for the grid null arms: the paper palette cycle is shorter
    # than 10, so C8/C9 wrap onto C0/C1 and collide with steered/ceiling.
    garm_color = {"steered": "C0", "null_sameval": "#8c564b", "null_xtype": "#17becf"}
    cls_color = {"content": "C4", "transport": "C5", "api-refusal": "C6"}
    est_color = {"coh": "k", "all": "0.6"}
    rendered: list[str] = []
    omitted: list[str] = []

    fig = plt.figure(figsize=(16.5, 13.5))
    gs = fig.add_gridspec(4, 6)

    # ── panel 1: G0 token-identity drops per direction ──────────────────
    ax = fig.add_subplot(gs[0, 0:2])
    if cfg.tokgate_report.is_file():
        tok = json.loads(cfg.tokgate_report.read_text(encoding="utf-8"))
        dirs = list(tok["directions"])
        x = np.arange(len(dirs))
        ax.bar(
            x,
            [tok["directions"][d]["n_pairs"] for d in dirs],
            color="none",
            edgecolor="0.75",
            label="pairs",
        )
        ax.bar(
            x,
            [tok["directions"][d]["n_intact"] for d in dirs],
            color="0.45",
            label="intact (drop = gap to pairs)",
        )
        floor = tok["min_intact_carriers"]
        ax.axhline(floor, color="k", lw=0.8, ls="--", label=f"testability floor ({floor})")
        ax.set_xticks(x)
        ax.set_xticklabels([_short_dir(d) for d in dirs], fontsize=6, rotation=45)
        ax.set_ylabel("pair count", fontsize=7)
        ax.set_title("G0 token identity per direction", fontsize=8)
        ax.legend(fontsize=5.5)
        rendered.append("token-identity")
    else:
        fig.delaxes(ax)
        omitted.append(f"token-identity — missing {cfg.tokgate_report}")

    # ── panels 2 + 5 (kept): N2.5 coherence retention / min-length drops ─
    labels = [f"{key}\n{arm}" for key in MODEL_KEYS for arm in ARM_KEYS]
    bar_colors = [arm_color[arm] for _key in MODEL_KEYS for arm in ARM_KEYS]

    ax = fig.add_subplot(gs[0, 2:4])
    coh_fracs = [
        stats["retention_coherence"][f"{key}|{arm}"]["coh_retention_frac"] or 0.0
        for key in MODEL_KEYS
        for arm in ARM_KEYS
    ]
    ax.bar(range(len(labels)), coh_fracs, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("coherence > 60 retention fraction", fontsize=7)
    ax.set_title("N2.5 coherence retention", fontsize=8)
    rendered.append("coherence-retention")

    ax = fig.add_subplot(gs[0, 4:6])
    len_fracs = [
        stats["retention_length"][key][arm]["len_drop_frac"] or 0.0
        for key in MODEL_KEYS
        for arm in ARM_KEYS
    ]
    ax.bar(range(len(labels)), len_fracs, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("< 48-token drop fraction", fontsize=7)
    ax.set_title("N2.5 min-length drops", fontsize=8)
    rendered.append("min-length-drop")

    # ── panel 3: grid cap-hit per unit at 4096 + G5 trigger, rows axis ──
    ax = fig.add_subplot(gs[1, 0:4])
    if cfg.cap_hit_report.is_file():
        cap = json.loads(cfg.cap_hit_report.read_text(encoding="utf-8"))
        pu = cap["per_unit"]
        n_set = {v["n_rows"] for v in pu.values()}
        assert len(n_set) == 1, f"non-uniform per-unit n_rows: {sorted(n_set)}"
        (n_unit,) = n_set
        cells = sorted({k.split("|")[0] for k in pu})
        slots = sorted({k.split("|")[1] for k in pu})
        arms_g = [a for a in garm_color if any(k.split("|")[2] == a for k in pu)]
        xs: list[float] = []
        heights: list[float] = []
        cols: list[str] = []
        centers: list[float] = []
        glabels: list[str] = []
        pos = 0.0
        for cell in cells:
            for slot in slots:
                start = pos
                for arm_g in arms_g:
                    rec = pu.get(f"{cell}|{slot}|{arm_g}")
                    if rec is None:
                        continue
                    xs.append(pos)
                    heights.append(rec["cap_hit_pct"])
                    cols.append(garm_color[arm_g])
                    pos += 1.0
                if pos > start:
                    centers.append((start + pos - 1.0) / 2.0)
                    glabels.append(f"{_short_dir(cell)} {slot}")
                pos += 0.9
            pos += 0.6
        ax.bar(xs, heights, color=cols, width=0.92)
        trig = cap["pre_registered_regen_trigger_pct"]
        ax.axhline(trig, color="k", ls="--", lw=0.9)
        top = max(max(heights), trig) * 1.25
        ax.set_ylim(0, top)
        ax.set_xticks(centers)
        ax.set_xticklabels(glabels, fontsize=5, rotation=90)
        ax.set_ylabel("cap-hit % of unit draws", fontsize=7)
        rows_ax = ax.twinx()
        rows_ax.set_ylim(0, top * n_unit / 100.0)
        rows_ax.set_yticks(range(int(np.floor(top * n_unit / 100.0)) + 1))
        rows_ax.set_ylabel(f"truncated rows (n={n_unit} per unit)", fontsize=7)
        handles = [Rectangle((0, 0), 1, 1, color=garm_color[a]) for a in arms_g]
        handles.append(Line2D([0], [0], color="k", ls="--", lw=0.9))
        ax.legend(
            handles,
            [*arms_g, f"G5 re-gen trigger ({trig:g}%)"],
            fontsize=5.5,
            ncol=len(arms_g) + 1,
            loc="upper left",
        )
        ax.set_title(
            f"grid cap-hit at {cap['max_new_tokens']} per (cell x slot x arm) unit", fontsize=8
        )
        rendered.append("cap-hit")
    else:
        fig.delaxes(ax)
        omitted.append(f"cap-hit — missing {cfg.cap_hit_report}")

    # ── panel 4: judge drop classes + frac_items_complete vs 0.95 floor ─
    cls_order = ("content", "transport", "api-refusal")
    metas = sorted((cfg.judge_root / "scores").glob("*.decay.meta.json"))
    ax_drop = fig.add_subplot(gs[1, 4:6])
    ax_frac = fig.add_subplot(gs[2, 0:2])
    if metas:
        waves: list[str] = []
        post: dict[str, list[int]] = {c: [] for c in cls_order}
        pre: dict[str, list[int]] = {c: [] for c in cls_order}
        frac_post: dict[str, list[float]] = {arm: [] for arm in ARM_KEYS}
        frac_pre: dict[str, list[float]] = {arm: [] for arm in ARM_KEYS}
        have_pre = False
        for p in metas:
            m = json.loads(p.read_text(encoding="utf-8"))
            wave = p.name.split(".")[0].removeprefix("dfrag-")
            waves.append(wave.split("_")[0])
            q = cfg.prerecovery_wavemeta_dir / p.name
            mq = json.loads(q.read_text(encoding="utf-8")) if q.is_file() else m
            have_pre = have_pre or q.is_file()
            for meta, dest in ((m, post), (mq, pre)):
                p1 = meta["pass1"]
                dest["content"].append(p1["n_dropped_draws_content"])
                dest["transport"].append(meta.get("residual_transport_lost") or 0)
                # API-classifier refusals censor items outright: count them as
                # registered-minus-scored over the per-arm bookkeeping (the
                # quarantined pre-recovery metas carry them there; rule 28).
                miss = sum(a["n_items"] - a["n_scored"] for a in meta["per_arm"].values())
                dest["api-refusal"].append(max(p1["n_refusal_draws"], miss))
            for arm in ARM_KEYS:
                a_post = m["per_arm"][f"grid-{arm}"]
                a_pre = mq["per_arm"][f"grid-{arm}"]
                frac_post[arm].append(a_post["n_scored"] / a_post["n_items"])
                frac_pre[arm].append(a_pre["n_scored"] / a_pre["n_items"])
        x = np.arange(len(waves))
        for ci, cls in enumerate(cls_order):
            off = (ci - 1) * 0.28
            if have_pre:
                ax_drop.bar(
                    x + off - 0.07,
                    pre[cls],
                    width=0.13,
                    color=cls_color[cls],
                    hatch="///",
                    edgecolor="k",
                    linewidth=0.2,
                )
                ax_drop.bar(x + off + 0.07, post[cls], width=0.13, color=cls_color[cls])
            else:
                ax_drop.bar(x + off, post[cls], width=0.2, color=cls_color[cls])
        ax_drop.set_xticks(x)
        ax_drop.set_xticklabels(waves, fontsize=6)
        ax_drop.set_ylabel("censored / dropped draws", fontsize=7)
        handles = [Rectangle((0, 0), 1, 1, color=cls_color[c]) for c in cls_order]
        labels_d = list(cls_order)
        if have_pre:
            handles.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="k", hatch="///"))
            handles.append(Rectangle((0, 0), 1, 1, facecolor="0.85"))
            labels_d += ["pre-recovery", "post rule-28 sync re-issue"]
        ax_drop.legend(handles, labels_d, fontsize=5.5)
        ax_drop.set_title("decay judge drop classes per wave", fontsize=8)
        rendered.append("judge-drop-classes" + ("" if have_pre else " (post-only)"))

        for ai, arm in enumerate(ARM_KEYS):
            off = (ai - 1) * 0.28
            ax_frac.bar(x + off, frac_post[arm], width=0.24, color=arm_color[arm], label=arm)
            if have_pre:
                ax_frac.plot(
                    x + off,
                    frac_pre[arm],
                    "o",
                    markerfacecolor="white",
                    markeredgecolor="k",
                    markeredgewidth=0.7,
                    linestyle="none",
                    markersize=4.5,
                    zorder=5,
                    label="pre-recovery" if ai == 0 else None,
                )
        ax_frac.axhline(0.95, color="grey", lw=0.8, ls="--", label="0.95 floor")
        ax_frac.set_ylim(0.9, 1.02)
        ax_frac.set_xticks(x)
        ax_frac.set_xticklabels(waves, fontsize=6)
        ax_frac.set_ylabel("frac_items_complete", fontsize=7)
        ax_frac.legend(fontsize=5.5, ncol=2)
        ax_frac.set_title("decay judge completeness per wave x arm", fontsize=8)
        rendered.append("frac-items-complete")
    else:
        fig.delaxes(ax_drop)
        fig.delaxes(ax_frac)
        omitted.append(
            f"judge-drop-classes + frac-items-complete — no metas under {cfg.judge_root / 'scores'}"
        )

    # ── panel 6: N2.2 absolute Q1 starting-level gap per model ──────────
    ax = fig.add_subplot(gs[2, 2:4])
    blk = stats["n2_2_q1_gap"]
    xq = 0.0
    ticks: list[float] = []
    tlabels: list[str] = []
    for key in MODEL_KEYS:
        for e in ESTIMANDS:
            rec = blk[key][e]
            lo, hi = LA._err(rec["point"], rec["ci_lo"], rec["ci_hi"])
            ax.errorbar(
                [xq], [rec["point"]], yerr=[[lo], [hi]], fmt="o", color=est_color[e], capsize=3
            )
            ticks.append(xq)
            tlabels.append(f"{key}\n{e}")
            xq += 1.0
        xq += 0.5
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=6)
    ax.set_ylabel("Q1 steered - ceiling (0-1 score)", fontsize=7)
    ax.set_title("N2.2 Q1 starting-level gap", fontsize=8)
    rendered.append("q1-gap")

    # ── panel 7: N2.3 rung-intersection sensitivity contrast ────────────
    ax = fig.add_subplot(gs[2, 4:6])
    inter_short = ",".join(v.split("_")[0] for v in stats["n2_3_intersection"]["rungs"])
    xq = 0.0
    ticks = []
    tlabels = []
    for key in MODEL_KEYS:
        for e in ESTIMANDS:
            for dx, name, mk, fill in (
                (0.0, f"{e}|primary|dD", "o", True),
                (0.3, f"{e}|intersection|dD", "s", False),
            ):
                rec = _fam(stats, key, name)
                if rec["point"] is None:
                    continue
                lo, hi = LA._err(rec["point"], rec["ci_lo"], rec["ci_hi"])
                ax.errorbar(
                    [xq + dx],
                    [rec["point"]],
                    yerr=[[lo], [hi]],
                    marker=mk,
                    linestyle="none",
                    color=est_color[e],
                    markerfacecolor=est_color[e] if fill else "none",
                    capsize=2,
                )
            ticks.append(xq + 0.15)
            tlabels.append(f"{key}\n{e}")
            xq += 1.0
        xq += 0.5
    ax.axhline(0.0, color="grey", lw=0.6)
    handles = [
        Line2D([0], [0], marker="o", color="k", linestyle="none"),
        Line2D([0], [0], marker="s", color="k", markerfacecolor="none", linestyle="none"),
    ]
    ax.legend(handles, ["primary rungs", f"intersection rungs ({inter_short})"], fontsize=5.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=6)
    ax.set_ylabel("delta-D (steered - ceiling raw drop)", fontsize=7)
    ax.set_title("N2.3 rung-intersection sensitivity", fontsize=8)
    rendered.append("rung-intersection")

    # ── panel 8: fragment-vs-whole score correlation (instrument sanity) ─
    ax = fig.add_subplot(gs[3, 0:2])
    sn = stats["fragment_vs_whole_sanity"]
    xs2: list[int] = []
    hs2: list[float] = []
    cs2: list[str] = []
    tl2: list[str] = []
    for key in MODEL_KEYS:
        for arm in ARM_KEYS:
            rec = sn[key][arm]
            if rec["rho"] is not None:
                xs2.append(len(xs2))
                hs2.append(rec["rho"])
                cs2.append(arm_color[arm])
                tl2.append(f"{key}\n{arm}\nn={rec['n']}")
    ax.bar(xs2, hs2, color=cs2)
    ax.set_xticks(xs2)
    ax.set_xticklabels(tl2, fontsize=5.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman rho", fontsize=7)
    ax.set_title("fragment-mean vs whole-response score", fontsize=8)
    rendered.append("fragment-vs-whole")

    # ── panel 9: conjunct diagnostic (Leg A R1/R2 steered instrument) ────
    ax = fig.add_subplot(gs[3, 2:4])
    cpath = cfg.q35_stats_json.parent / "conjuncts.jsonl"
    conj_rows = (
        [json.loads(line) for line in cpath.open(encoding="utf-8")] if cpath.is_file() else []
    )
    if conj_rows:
        cdirs = sorted({r["direction"] for r in conj_rows}, key=lambda d: (d.split("_", 1)[1], d))
        conjs = sorted({r["conjunct"] for r in conj_rows})
        mat = np.full((len(cdirs), len(conjs)), np.nan)
        acc: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in conj_rows:
            if r["mean_score"] is not None:
                acc[(r["direction"], r["conjunct"])].append(r["mean_score"])
        for (d, c), vs in acc.items():
            mat[cdirs.index(d), conjs.index(c)] = float(np.mean(vs))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(conjs)))
        ax.set_xticklabels(conjs, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(cdirs)))
        ax.set_yticklabels([_short_dir(d) for d in cdirs], fontsize=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("mean judge score (0-100)", fontsize=6)
        ax.set_title("conjunct scores (mean over carriers x slots)", fontsize=8)
        rendered.append("conjunct")
    else:
        fig.delaxes(ax)
        omitted.append(f"conjunct — missing/empty {cpath}")

    # ── panel 10: rule-19 TF-margin vs F_beh validation scatter ─────────
    ax = fig.add_subplot(gs[3, 4:6])
    fstats = (
        json.loads(cfg.q35_stats_json.read_text(encoding="utf-8"))
        if cfg.q35_stats_json.is_file()
        else {}
    )
    mv = fstats.get("margin_validation") or {}
    pts = mv.get("percell_points") or []
    if pts:
        ax.scatter(
            [p["margin_shift_mean"] for p in pts],
            [p["f_beh_mean"] for p in pts],
            s=16,
            color="C3",
            alpha=0.85,
        )
        ax.set_xlabel("per-cell mean TF margin shift", fontsize=7)
        ax.set_ylabel("per-cell mean F_beh", fontsize=7)
        ax.set_title(
            f"rule-19 margin validation: rho={mv['rho_margin_fbeh_percell']:.2f}, "
            f"p={mv['p_percell']:.3g}, n={mv['n_cells']}",
            fontsize=8,
        )
        rendered.append("margin-vs-F")
    else:
        fig.delaxes(ax)
        omitted.append(
            f"margin-vs-F — missing margin_validation.percell_points in {cfg.q35_stats_json}"
        )

    savefig_paper(fig, "q35_ladder_decay_diagnostics", dir=cfg.figures_dir)
    plt.close(fig)
    logger.info(
        "[figures] diagnostics panels rendered=%d (%s); omitted=%s",
        len(rendered),
        ", ".join(rendered),
        "; ".join(omitted) if omitted else "none",
    )


def phase_figures(cfg: DecayConfig) -> int:
    """Figures 4-6 + the decay-diagnostics companion (fig 7's Leg B panels
    cannot land in L6, which runs before L7)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    stats = json.loads((cfg.out_dir / "decay_stats.json").read_text(encoding="utf-8"))
    segs = list(range(1, DECAY_K + 1))
    arm_color = {"steered": "C0", "ceiling": "C1", "floor": "C2"}
    est_style = {"coh": "-", "all": "--"}

    # fig 4: raw decay curves (primary pooled)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key in zip(axes, MODEL_KEYS):
        for e in ESTIMANDS:
            for arm in ARM_KEYS:
                pts = [_fam(stats, key, f"{e}|primary|seg{k}|{arm}") for k in segs]
                y = [p["point"] for p in pts]
                los = [
                    LA._err(p["point"], p["ci_lo"], p["ci_hi"])[0] if p["point"] is not None else 0
                    for p in pts
                ]
                his = [
                    LA._err(p["point"], p["ci_lo"], p["ci_hi"])[1] if p["point"] is not None else 0
                    for p in pts
                ]
                ax.errorbar(
                    segs,
                    [np.nan if v is None else v for v in y],
                    yerr=[los, his],
                    color=arm_color[arm],
                    linestyle=est_style[e],
                    marker="o",
                    markersize=3,
                    label=f"{arm} ({e})" if key == "q25" else None,
                    alpha=1.0 if e == "coh" else 0.45,
                )
        ax.set_xticks(segs)
        ax.set_xlabel("token quartile")
        ax.set_title(key)
    axes[0].set_ylabel("fragment persona score (0-1)")
    axes[0].legend(fontsize=6, ncol=2)
    savefig_paper(fig, "q35_ladder_decay_decay_raw", dir=cfg.figures_dir)
    plt.close(fig)

    # fig 5: normalized F(seg) curves, suppressed segments as gaps
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key in zip(axes, MODEL_KEYS):
        for e in ESTIMANDS:
            pts = [_fam(stats, key, f"{e}|primary|F|seg{k}") for k in segs]
            y = np.array([np.nan if p["point"] is None else p["point"] for p in pts])
            los = [
                LA._err(p["point"], p["ci_lo"], p["ci_hi"])[0] if p["point"] is not None else 0
                for p in pts
            ]
            his = [
                LA._err(p["point"], p["ci_lo"], p["ci_hi"])[1] if p["point"] is not None else 0
                for p in pts
            ]
            ax.errorbar(
                segs,
                y,
                yerr=[los, his],
                linestyle=est_style[e],
                marker="s",
                markersize=3,
                color="C3",
                alpha=1.0 if e == "coh" else 0.45,
                label=e if key == "q25" else None,
            )
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.axhline(1.0, color="grey", lw=0.6)
        ax.set_xticks(segs)
        ax.set_xlabel("token quartile")
        ax.set_title(key)
    axes[0].set_ylabel("F = (steered - floor) / (ceiling - floor)")
    axes[0].legend(fontsize=6)
    savefig_paper(fig, "q35_ladder_decay_decay_norm", dir=cfg.figures_dir)
    plt.close(fig)

    # fig 6: decay contrast dD (+ dD_F companion), per-carrier points
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, fam_tag, ylab in (
        (axes[0], "dD", "delta-D (steered - ceiling raw drop, 0-1)"),
        (axes[1], "dD_F", "delta-D_F (change in patched arm's F; ceiling drop = 0)"),
    ):
        x = 0
        ticks, ticklabels = [], []
        for key in MODEL_KEYS:
            for e in ESTIMANDS:
                rec = _fam(stats, key, f"{e}|primary|{fam_tag}")
                if rec["point"] is not None:
                    lo_off, hi_off = LA._err(rec["point"], rec["ci_lo"], rec["ci_hi"])
                    ax.errorbar(
                        [x],
                        [rec["point"]],
                        yerr=[[lo_off], [hi_off]],
                        fmt="o",
                        color="C0" if e == "coh" else "C7",
                        capsize=3,
                    )
                    # per-carrier scatter from per_direction records
                    pts = []
                    for rk, drec in stats["per_direction"].items():
                        mk, d, slot, ee = rk.split("|")
                        if mk != key or ee != e or slot != "ce":
                            continue
                        if stats["scope"][key]["strata"].get(d[len("install_") :]) != "primary":
                            continue
                        stat_key = "delta_d" if fam_tag == "dD" else "delta_d_f"
                        for r in drec["per_carrier"].values():
                            if r.get("supported") and r.get(stat_key) is not None:
                                pts.append(r[stat_key])
                    if pts:
                        ax.scatter(
                            np.full(len(pts), x) + np.linspace(-0.12, 0.12, len(pts)),
                            pts,
                            s=6,
                            alpha=0.4,
                            color="C0" if e == "coh" else "C7",
                        )
                ticks.append(x)
                ticklabels.append(f"{key}\n{e}")
                x += 1
            x += 0.5
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=6)
        ax.set_ylabel(ylab, fontsize=7)
    verd = stats["lattice"]
    axes[0].set_title(f"q25: {verd['q25']['verdict']} | q35: {verd['q35']['verdict']}", fontsize=8)
    savefig_paper(fig, "q35_ladder_decay_contrast", dir=cfg.figures_dir)
    plt.close(fig)

    # diagnostics companion: the ten manifest-declared panels (incl. the two
    # previously-rendered N2.5 panels, kept) — see _fig_diagnostics.
    _fig_diagnostics(cfg, stats)
    logger.info("[figures] 4 figures written to %s", cfg.figures_dir)
    return RC_OK


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "stage": phase_stage,
    "pilot": phase_pilot,
    "wave": phase_wave,
    "reduce": phase_reduce,
    "figures": phase_figures,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2329 q35_ladder_decay Leg B (L7) driver.")
    ap.add_argument(
        "--phase",
        choices=tuple(PHASES),
        help="pipeline phase to run (required unless --import-check)",
    )
    ap.add_argument("--q25-in-root", type=Path, default=Path("data/issue_2329/decay_parent_inputs"))
    ap.add_argument("--q35-in-root", type=Path, default=Path("data/issue_2329/ladder_judge_inputs"))
    ap.add_argument(
        "--q25-scores-dir",
        type=Path,
        default=Path("eval_results/issue_2162/persona_specificity_ladder/judge/scores"),
    )
    ap.add_argument(
        "--q25-gates-dir",
        type=Path,
        default=Path("eval_results/issue_2162/persona_specificity_ladder/judge/gates"),
    )
    ap.add_argument(
        "--q25-stats-json",
        type=Path,
        # Parent #2162 committed its ladder stats at the round root — there is
        # NO f_metrics/ subdir on the q25 side (that layout is this fork's L6
        # output shape only; review r1 must-fix 2 / B1).
        default=Path("eval_results/issue_2162/persona_specificity_ladder/stats.json"),
    )
    ap.add_argument(
        "--q35-scores-dir",
        type=Path,
        default=Path("eval_results/issue_2329/q35_ladder_decay/judge/scores"),
    )
    ap.add_argument(
        "--q35-gates-dir",
        type=Path,
        default=Path("eval_results/issue_2329/q35_ladder_decay/judge/gates"),
    )
    ap.add_argument(
        "--q35-stats-json",
        type=Path,
        default=Path("eval_results/issue_2329/q35_ladder_decay/f_metrics/stats.json"),
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_2329/q35_ladder_decay/decay")
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("data/issue_2329/decay_judge_cache"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_2329/q35_ladder_decay"))
    ap.add_argument(
        "--cap-hit-report",
        type=Path,
        default=Path("eval_results/issue_2329/q35_ladder_decay/cap_hit/cap_hit_report_grid.json"),
        help="figures phase: grid cap-hit report (panel 3); absent => panel omitted",
    )
    ap.add_argument(
        "--tokgate-report",
        type=Path,
        default=Path(
            "eval_results/issue_2329/q35_ladder_decay/gates/token_identity_report_ladder.json"
        ),
        help="figures phase: G0 token-identity report (panel 1); absent => panel omitted",
    )
    ap.add_argument(
        "--prerecovery-wavemeta-dir",
        type=Path,
        default=Path("eval_results/issue_2329/q35_ladder_decay/decay/judge/prerecovery_quarantine"),
        help=(
            "figures phase: quarantined pre-recovery wave metas (rule-28 re-issue record); "
            "absent => drop-class panel renders the post-recovery state only"
        ),
    )
    ap.add_argument("--judge-model", default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DECAY_MAX_TOKENS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument(
        "--wave-threshold-base",
        type=int,
        default=0,
        help=(
            "wave transport routing; 0 (default) = the registered all-batch plan. "
            "Raise (e.g. 1000000000) ONLY for a rule-28 targeted SYNC re-issue of "
            "API-classifier-refused draws — cached draws are reused, so only the "
            "refused items are re-submitted."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="execute every deferred import + args-attribute completeness, then exit 0",
    )
    return ap.parse_args(argv)


def _import_check() -> None:
    """Execute every deferred import + registered-constant asserts (module-level
    helper so its bare-name imports cannot shadow main()'s call names)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import matplotlib  # noqa: F401
    from scipy.stats import spearmanr  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401

    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        savefig_paper,
        set_paper_style,
    )
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
    from explore_persona_space.experiments.issue2329 import bank2329 as BANK29
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        stage_hub_file,
        stage_hub_prefix,
    )
    import issue2329_judge as J62F  # noqa: F401  (deferred inside build_side)
    from issue2329_ladder import MODEL_REVISION_PIN
    from issue2329_ladder import Q35_PARENT_HF_REVISION as LAD_PARENT_REV

    assert len(MODEL_REVISION_PIN) == 40, MODEL_REVISION_PIN
    assert BANK29.MODEL_ID == "Qwen/Qwen3.5-9B", BANK29.MODEL_ID
    assert BANK29.TEMPLATE_KWARGS == {"enable_thinking": False}
    # Equality, not just length: the judge fork and the ladder driver must
    # stage the SAME parent revision (review r1 g2).
    assert LJ.Q35_PARENT_HF_REVISION == LAD_PARENT_REV, (
        LJ.Q35_PARENT_HF_REVISION,
        LAD_PARENT_REV,
    )
    assert len(LJ.Q35_PARENT_HF_REVISION) == 40
    for v in LB.PERSONA_VALUE_IDS:
        prompt = fragment_eval_prompt(LB.VALUES_BY_ID[v].descriptor)
        assert "{question}" in prompt and "{answer}" in prompt, v
        assert fragment_rubric_id(v) == f"dfrag-{v}"
    assert callable(bootstrap_family_means_batched)
    assert callable(J94.run_wave) and callable(J94._item_id)
    assert callable(LA._read_gate) and callable(LA._ci_from_boot) and callable(LA._err)
    print("[import-check] issue2329_decay OK")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    if args.phase is None:
        raise SystemExit("--phase is required unless --import-check")
    cfg = DecayConfig(
        q25_in_root=args.q25_in_root,
        q35_in_root=args.q35_in_root,
        q25_scores_dir=args.q25_scores_dir,
        q25_gates_dir=args.q25_gates_dir,
        q25_stats_json=args.q25_stats_json,
        q35_scores_dir=args.q35_scores_dir,
        q35_gates_dir=args.q35_gates_dir,
        q35_stats_json=args.q35_stats_json,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        figures_dir=args.figures_dir,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        n_boot=args.n_boot,
        dry_run=args.dry_run,
        wave_threshold_base=args.wave_threshold_base,
        cap_hit_report=args.cap_hit_report,
        tokgate_report=args.tokgate_report,
        prerecovery_wavemeta_dir=args.prerecovery_wavemeta_dir,
    )
    print(f"[phase={args.phase}] start", flush=True)
    rc = PHASES[args.phase](cfg)
    print(f"[phase={args.phase}] done rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
