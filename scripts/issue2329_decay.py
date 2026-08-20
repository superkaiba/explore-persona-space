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

import numpy as np

import issue2094_judge as J94
import issue2162_ladder_judge as PLJ
import issue2329_ladder_analysis as LA
import issue2329_ladder_judge as LJ
from explore_persona_space.experiments.issue2162 import ladder_bank as LB
from issue2094_analysis import bootstrap_family_means_batched

logger = logging.getLogger("issue2329.decay")

# ── registered constants (plan v8 §3/§4.1/§6/§11) ─────────────────────

MODEL_KEYS = ("q25", "q35")
DECAY_K = 4  # contiguous token quartiles
MIN_COMPLETION_TOKENS = 48  # the ONLY dispatch filter (plan §4.1 item 4)
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
    tests/test_issue2329_decay.py).
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
            f"{stats_json} missing — L6 (f_metrics/stats.json) must complete before L7; "
            "for the q25 side the committed parent stats may need "
            "`git sparse-checkout add eval_results/issue_2162` in this worktree"
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

    # steered: grid install-ce rows (+ conditional install-pe stratum)
    for r in grid_rows:
        cell = r["cell"]
        if not cell.startswith("install_") or r["arm"] != "steered":
            continue
        v = cell[len("install_") :]
        if v not in surviving:
            continue
        slot = r["slot"]
        if slot == "pe" and cell not in side.pe_directions:
            continue
        if slot not in ("ce", "pe"):
            continue
        carrier = pair_carrier[r["pair_id"]]
        if carrier not in surviving[v]:
            continue
        comp_key = {"pair_id": r["pair_id"], "context_id": r["context_id"], "draw": int(r["draw"])}
        coh = coh_grid.get((r["pair_id"], slot, "steered", int(r["draw"])))
        emit(
            "steered",
            cell,
            v,
            slot,
            carrier,
            comp_key,
            questions[r["context_id"]],
            r["text"],
            coh,
            r.get("cap_hit"),
        )

    # ceiling + floor: anchors, segmented ONCE per completion, judged per rung
    by_cid: dict[str, list[dict]] = defaultdict(list)
    for r in anchor_rows:
        by_cid[r["context_id"]].append(r)
    for v in side.scope_values:
        direction = f"install_{v}"
        for carrier in surviving[v]:
            for arm, ctx_value in (("ceiling", v), ("floor", "plain")):
                cid = LB.context_id(ctx_value, carrier)
                rows = by_cid.get(cid, [])
                if not rows:
                    raise RuntimeError(f"[{key}] no anchor rows for {cid} ({arm} arm)")
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
    instrument, threshold_base=0 pinned batch on pilot AND production
    (llm-judging rules 26/#2152). REFUSES --dry-run."""
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
        threshold_base=0,
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
    """One production run_wave per dfrag rubric (items span BOTH models),
    threshold_base=0 (pinned batch, matching the pilot's declared transport)."""
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
            threshold_base=0,
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

    def support_ok(e, key, d, slot, c, m):
        """Common support: >= m kept completions per arm in EVERY segment
        (steered at the given slot; ceiling/floor slot-free)."""
        for arm, s in (("steered", slot), ("ceiling", "na"), ("floor", "na")):
            for seg in range(1, DECAY_K + 1):
                if len(cell_comps.get((e, key, d, s, c, arm, seg), ())) < m:
                    return False
        return True

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
                            per_c[c] = {"supported": False}
                            continue
                        m_st = {k: mean01(e, key, d, slot, c, "steered", k) for k in range(1, 5)}
                        m_ce = {k: mean01(e, key, d, "na", c, "ceiling", k) for k in range(1, 5)}
                        m_fl = {k: mean01(e, key, d, "na", c, "floor", k) for k in range(1, 5)}
                        dens = {k: m_ce[k] - m_fl[k] for k in range(1, 5)}
                        f = {
                            k: ((m_st[k] - m_fl[k]) / dens[k])
                            if abs(dens[k]) >= DENOM_BAR
                            else None
                            for k in range(1, 5)
                        }
                        d_raw_st = m_st[1] - m_st[4]
                        d_raw_ce = m_ce[1] - m_ce[4]
                        dd = d_raw_st - d_raw_ce
                        dd_f = (f[1] - f[4]) if (f[1] is not None and f[4] is not None) else None
                        q1gap = m_st[1] - m_ce[1]
                        per_c[c] = {
                            "supported": True,
                            "mean_steered": m_st,
                            "mean_ceiling": m_ce,
                            "mean_floor": m_fl,
                            "denominators": dens,
                            "F": f,
                            "d_raw_steered": d_raw_st,
                            "d_raw_ceiling": d_raw_ce,
                            "delta_d": dd,
                            "delta_d_f": dd_f,
                            "delta_d_f_unavailable_reason": (
                                None
                                if dd_f is not None
                                else "endpoint |ceiling-floor| below the 0.125 bar"
                            ),
                            "q1_gap": q1gap,
                        }
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
    def _label(key, e):
        dd = fam_stats[key][f"{e}|primary|dD"]
        ddf = fam_stats[key][f"{e}|primary|dD_F"]
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

    lattice = {}
    for key in MODEL_KEYS:
        labels = {}
        for e in ESTIMANDS:
            lab, reason = _label(key, e)
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
                f">= {COMMON_SUPPORT_MIN} kept completion(s) per arm per carrier in EVERY "
                f"segment; sensitivity re-read at >= {SENSITIVITY_MIN}"
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

    # diagnostics companion: length-drop + coherence retention per arm x model
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    labels = [f"{key}\n{arm}" for key in MODEL_KEYS for arm in ARM_KEYS]
    len_fracs = [
        stats["retention_length"][key][arm]["len_drop_frac"] or 0.0
        for key in MODEL_KEYS
        for arm in ARM_KEYS
    ]
    coh_fracs = [
        stats["retention_coherence"][f"{key}|{arm}"]["coh_retention_frac"] or 0.0
        for key in MODEL_KEYS
        for arm in ARM_KEYS
    ]
    axes[0].bar(range(len(labels)), len_fracs, color="C4")
    axes[0].set_ylabel("< 48-token drop fraction")
    axes[1].bar(range(len(labels)), coh_fracs, color="C5")
    axes[1].set_ylabel("coherence > 60 retention fraction")
    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6)
    savefig_paper(fig, "q35_ladder_decay_decay_diagnostics", dir=cfg.figures_dir)
    plt.close(fig)
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
        default=Path("eval_results/issue_2162/persona_specificity_ladder/f_metrics/stats.json"),
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
    ap.add_argument("--judge-model", default=J94.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DECAY_MAX_TOKENS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
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
    from issue2329_ladder import MODEL_REVISION_PIN

    assert len(MODEL_REVISION_PIN) == 40, MODEL_REVISION_PIN
    assert BANK29.MODEL_ID == "Qwen/Qwen3.5-9B", BANK29.MODEL_ID
    assert BANK29.TEMPLATE_KWARGS == {"enable_thinking": False}
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
