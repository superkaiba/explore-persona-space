#!/usr/bin/env python3
"""Issue #779 round 5: reconstruct the sycophancy/hallucination PV extraction artifacts.

The parent's Sonnet-generated extraction artifacts
(``data/issue_779/artifacts/{sycophancy,hallucination}.json``) were produced on
the parent's terminated pod and never uploaded, blocking the Arm B/C corpus
relaunch (the round-4 preflight fails loud on them). Recovery inventory
(verified 2026-07-02):

- ``extraction_questions`` — RECOVERABLE VERBATIM from the parent's uploaded
  judge-dispatch checkpoints (``issue779_monitoring/analysis_tensors/
  .judge_dispatch/dispatch_*/items.json`` on the HF data repo; local mirror at
  ``data/issue779_hfstage/...``): each trait has a pos + neg extraction-arm
  dispatch of 5000 items (5 personas x 20 questions x 10 rollouts x N=5 judge
  draws), and the 20 distinct ``question`` strings per arm ARE the extraction
  questions, order-recoverable from the global question index embedded in the
  item custom_ids.
- ``instruction`` pairs, ``eval_questions``, ``eval_prompt`` — GENUINELY LOST
  (the pass_a artifact-generation outputs were never uploaded). REGENERATED via
  the standing generator (``issue779_common.generate_extraction_artifacts`` —
  same ``PV_ARTIFACT_GENERATION_PROMPT`` template + verbatim
  ``TRAIT_DESCRIPTIONS``), then SPLICED: the recovered verbatim 20 replace the
  regenerated ``extraction_questions``; the regenerated
  instruction/eval_questions/eval_prompt are kept (eval_questions de-collided
  against the recovered set so the paper's disjoint split survives).

RUBRIC VALIDATION (the gate): the regenerated ``eval_prompt`` seeds a NEW judge
rubric (``trait_judge_system_prompt``), so per trait we re-judge ~150 parent
(question, completion) items — stratified across the parent score range, parent
per-item score = mean of its N=5 draw scores reconstructed from the dispatch
``results_msgbatch_*.json`` files (DROP-NEVER-COERCE per draw) — with the new
rubric at N=5 draws (sync path, ``claude-sonnet-4-5-20250929``), and gate on
Spearman rho(new mean, parent mean) >= 0.85 per trait. Below the gate ->
regenerate ONCE (fresh Sonnet call) and re-validate; still below -> STOP with
the numbers (the local artifact is quarantined to ``*.FAILED_VALIDATION`` so a
later ``load_extraction_artifacts`` fails loud instead of consuming an
unvalidated rubric), exit non-zero.

Validated artifacts upload to the HF data repo at
``issue779_monitoring/artifacts/<trait>.json`` (small text, always-upload
policy; ONE upload_folder commit), where the driver-side
``load_extraction_artifacts`` HF fallback resolves them on the git-clone lanes.

CPU + API only (Sonnet + HF Hub). No GPU, no pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_reconstruct_artifacts")

N_EXTRACTION_QUESTIONS = 20
N_EVAL_QUESTIONS = 20
DEFAULT_N_SAMPLE = 150
DEFAULT_RHO_GATE = 0.85
MAX_GENERATION_ATTEMPTS = 2  # one initial + one regenerate-once retry on a gate miss

# Parent extraction-arm item custom_id shape (judge_rollouts_n5 global-idx scheme,
# persona key f"{trait}_{arm}_p{i}"): f"{persona}__{global_q_idx:05d}__{ci:02d}"
# with ci = rollout_idx * n_draws + draw_idx.
CID_RE = re.compile(
    r"^(?P<trait>[a-z]+)_(?P<arm>pos|neg)_p(?P<p>\d+)__(?P<q>\d{5})__(?P<ci>\d{2,})$"
)

# Local mirror of the parent's uploaded judge-dispatch checkpoints (the HF
# source of truth is issue779_monitoring/analysis_tensors/.judge_dispatch/ on
# superkaiba1/explore-persona-space-data, repo_type=dataset).
DEFAULT_DISPATCH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "issue779_hfstage"
    / "issue779_monitoring"
    / "analysis_tensors"
    / ".judge_dispatch"
)
HF_ARTIFACTS_PREFIX = f"{C.HF_PREFIX}/artifacts"


# ── dispatch loading ─────────────────────────────────────────────────────────


def scan_dispatches(root: Path) -> dict[tuple[str, str], Path]:
    """Map (trait, arm) -> dispatch dir by parsing each dispatch's item custom_ids.

    Asserts every custom_id in a dispatch shares one (trait, arm) and that no
    (trait, arm) appears twice. Fails loud on a missing/empty root (the mirror
    must be downloaded from HF first — see DEFAULT_DISPATCH_ROOT docnote).
    """
    assert root.is_dir(), (
        f"dispatch root {root} missing — download the parent judge-dispatch mirror from the HF "
        f"data repo ({C.HF_DATA_REPO}, repo_type=dataset) under "
        f"{C.HF_PREFIX}/analysis_tensors/.judge_dispatch/ first"
    )
    out: dict[tuple[str, str], Path] = {}
    for d in sorted(root.glob("dispatch_*")):
        items_path = d / "items.json"
        if not items_path.is_file():
            continue
        with open(items_path) as f:
            items = json.load(f)
        assert items, f"{d}: empty items.json"
        first = next(iter(items))
        m = CID_RE.match(first)
        assert m, f"{d}: unrecognized custom_id shape {first!r}"
        key = (m["trait"], m["arm"])
        for cid in items:
            m2 = CID_RE.match(cid)
            assert m2 and (m2["trait"], m2["arm"]) == key, (
                f"{d}: mixed trait/arm custom_ids ({first!r} vs {cid!r})"
            )
        assert key not in out, f"duplicate dispatch for {key}: {out[key]} and {d}"
        out[key] = d
    assert out, f"no dispatch_* dirs with items.json under {root}"
    return out


def _load_items(dispatch_dir: Path) -> dict[str, dict]:
    """Load a dispatch's items.json ({custom_id: {question, completion, user_msg}})."""
    with open(dispatch_dir / "items.json") as f:
        return json.load(f)


def _load_state(dispatch_dir: Path) -> dict:
    """Load a dispatch's state.json (fingerprint + judge_system_prompt_sha256 + ...)."""
    with open(dispatch_dir / "state.json") as f:
        return json.load(f)


def load_parent_scores(dispatch_dir: Path, items: dict[str, dict]) -> dict[str, dict]:
    """Union the dispatch's results_msgbatch_*.json score maps; assert full item coverage.

    The parent batch_judge results files carry {"scores": {custom_id:
    score_dict}} keyed DIRECTLY by the item custom_ids (verified 2026-07-02:
    direct overlap 5000/5000 on every dispatch; the sha256(item_id) custom_id
    remap of batch_judge.make_custom_id is internal to the Anthropic Batch
    submit and already undone in these persisted results).
    """
    scores: dict[str, dict] = {}
    for rf in sorted(dispatch_dir.glob("results_msgbatch_*.json")):
        with open(rf) as f:
            scores.update(json.load(f)["scores"])
    missing = set(items) - set(scores)
    assert not missing, (
        f"{dispatch_dir}: {len(missing)} items have no judge result "
        f"(e.g. {sorted(missing)[:3]}) — results files incomplete"
    )
    return scores


# ── question recovery ────────────────────────────────────────────────────────


def _ordered_questions_one_arm(items: dict[str, dict], trait: str, arm: str) -> list[str]:
    """Recover the ordered 20 extraction questions from one arm's items.

    The parent's judge_rollouts_n5 enumerates a GLOBAL question index across
    personas p0..p4 (20 questions each, artifact order), so persona p's j-th
    question sits at global idx 20p + j. Asserts: every persona block is
    complete + consistent, every persona yields the SAME ordered 20, and the 20
    are distinct. Returns the artifact-order list.
    """
    per_persona: dict[int, dict[int, str]] = {}
    for cid, item in items.items():
        m = CID_RE.match(cid)
        assert m and m["trait"] == trait and m["arm"] == arm, f"foreign cid {cid!r}"
        p, gq = int(m["p"]), int(m["q"])
        j = gq - N_EXTRACTION_QUESTIONS * p
        assert 0 <= j < N_EXTRACTION_QUESTIONS, (
            f"{trait}_{arm}: global question idx {gq} outside persona p{p}'s block"
        )
        prev = per_persona.setdefault(p, {}).get(j)
        if prev is None:
            per_persona[p][j] = item["question"]
        else:
            assert prev == item["question"], (
                f"{trait}_{arm}: inconsistent question at p{p} slot {j}"
            )
    assert per_persona, f"{trait}_{arm}: no items"
    ordered_by_p: dict[int, list[str]] = {}
    for p, slots in per_persona.items():
        assert set(slots) == set(range(N_EXTRACTION_QUESTIONS)), (
            f"{trait}_{arm}: persona p{p} question block incomplete "
            f"({len(slots)}/{N_EXTRACTION_QUESTIONS} slots)"
        )
        ordered_by_p[p] = [slots[j] for j in range(N_EXTRACTION_QUESTIONS)]
    base = ordered_by_p[min(ordered_by_p)]
    for p, lst in ordered_by_p.items():
        assert lst == base, f"{trait}_{arm}: persona p{p} question order diverges from p0"
    assert len(set(base)) == N_EXTRACTION_QUESTIONS, (
        f"{trait}_{arm}: expected {N_EXTRACTION_QUESTIONS} DISTINCT questions, got {len(set(base))}"
    )
    return base


def recover_extraction_questions(
    pos_items: dict[str, dict], neg_items: dict[str, dict], trait: str
) -> list[str]:
    """Recover the verbatim 20 extraction questions; assert pos-arm == neg-arm.

    Both arms iterate the SAME artifact, so the recovered sets (and order) must
    match exactly — a mismatch means the dispatch pairing is wrong (fail loud).
    """
    pos_q = _ordered_questions_one_arm(pos_items, trait, "pos")
    neg_q = _ordered_questions_one_arm(neg_items, trait, "neg")
    assert set(pos_q) == set(neg_q), (
        f"{trait}: pos-arm question set != neg-arm question set — wrong dispatch pairing? "
        f"pos-only={sorted(set(pos_q) - set(neg_q))[:2]}, "
        f"neg-only={sorted(set(neg_q) - set(pos_q))[:2]}"
    )
    assert pos_q == neg_q, f"{trait}: pos/neg arms agree on the set but diverge in ORDER"
    return pos_q


# ── parent score reconstruction (DROP-NEVER-COERCE per draw) ─────────────────


def parent_rollouts(
    items: dict[str, dict], scores: dict[str, dict], n_draws: int = C.JUDGE_N_DRAWS
) -> dict[str, dict]:
    """Group draw items into rollouts; parent per-rollout score = mean of valid draws.

    Each rollout was expanded into ``n_draws`` identical (question, completion)
    draw items (ci = ri*n_draws + d); a malformed / REFUSAL / out-of-range draw
    parses to None and is DROPPED (never coerced); a rollout with 0 valid draws
    gets ``parent_mean=None`` (excluded from the validation comparison,
    counted). Returns {rollout_cid: {question, completion, parent_mean,
    n_valid_parent_draws}}.
    """
    draw_scores = C._parse_raw_all_scores(scores)
    rollouts: dict[str, dict] = {}
    for cid, item in items.items():
        m = CID_RE.match(cid)
        assert m, cid
        ci = int(m["ci"])
        ri = ci // n_draws
        rkey = f"{m['trait']}_{m['arm']}_p{m['p']}__{m['q']}__{ri:02d}"
        r = rollouts.setdefault(
            rkey,
            {"question": item["question"], "completion": item["completion"], "draws": []},
        )
        assert r["completion"] == item["completion"], (
            f"{rkey}: draws differ in completion — not an N-draw expansion?"
        )
        assert r["question"] == item["question"], f"{rkey}: draws differ in question"
        r["draws"].append(draw_scores.get(cid))
    for rkey, r in rollouts.items():
        assert len(r["draws"]) == n_draws, (rkey, len(r["draws"]), n_draws)
        valid = [s for s in r["draws"] if s is not None]
        r["parent_mean"] = float(statistics.fmean(valid)) if valid else None
        r["n_valid_parent_draws"] = len(valid)
        del r["draws"]
    return rollouts


def stratified_sample(rollouts: dict[str, dict], n: int, seed: int) -> list[str]:
    """Deterministic round-robin sample across ten 10-point parent-score bins.

    Bins on ``int(parent_mean // 10)`` clamped to 9; rollouts with
    ``parent_mean=None`` are excluded. Within-bin order is seeded-shuffled;
    bins are cycled ascending, one pick per pass, until ``n`` picks (or
    exhaustion — returns fewer with a warning). Guarantees coverage of every
    populated score band rather than the (typically floor-heavy) marginal.
    """
    bins: dict[int, list[str]] = {}
    for k in sorted(rollouts):
        mean = rollouts[k]["parent_mean"]
        if mean is None:
            continue
        bins.setdefault(min(int(mean // 10), 9), []).append(k)
    rng = random.Random(seed)
    for b in bins:
        rng.shuffle(bins[b])
    picked: list[str] = []
    order = sorted(bins)
    idx = dict.fromkeys(order, 0)
    while len(picked) < n:
        progressed = False
        for b in order:
            if len(picked) >= n:
                break
            if idx[b] < len(bins[b]):
                picked.append(bins[b][idx[b]])
                idx[b] += 1
                progressed = True
        if not progressed:
            logger.warning("stratified_sample exhausted at %d < %d picks", len(picked), n)
            break
    return picked


# ── regeneration + splice ────────────────────────────────────────────────────


@contextmanager
def _redirected_artifacts_dir(scratch: Path):
    """Route generate_extraction_artifacts' cache write to a scratch dir.

    The standing generator writes straight to the LIVE artifacts cache; without
    this redirect, a crash between generate and splice would leave a
    regenerated-but-not-spliced artifact (silently WRONG extraction_questions)
    on disk for a later load_extraction_artifacts. The live cache is only ever
    written with the fully-spliced artifact (atomically), never the raw
    regeneration.
    """
    orig = C._artifacts_dir
    C._artifacts_dir = lambda: scratch
    try:
        yield
    finally:
        C._artifacts_dir = orig


def _norm(s: str) -> str:
    """Whitespace-normalized string (the assert_corpus_disjoint comparison key)."""
    return " ".join(s.strip().split())


def splice_artifacts(regenerated: dict, recovered_questions: list[str], trait: str) -> dict:
    """Splice the RECOVERED verbatim extraction questions into a regenerated artifact.

    Keeps the regenerated instruction / eval_prompt; replaces
    ``extraction_questions`` with the recovered 20 VERBATIM; rebuilds
    ``eval_questions`` from the regenerated 40-question pool (eval half first,
    then the discarded regenerated extraction half) minus any
    whitespace-normalized collision with the recovered set — preserving the
    paper's disjoint extraction/eval split deterministically. Re-runs the
    _validate_generated_artifacts-style count/shape checks on the spliced
    result; raises ArtifactCountShortfall when de-collision leaves fewer than
    20 eval questions (retryable via a fresh regeneration, never a pad).
    """
    assert len(recovered_questions) == N_EXTRACTION_QUESTIONS, len(recovered_questions)
    assert len(set(recovered_questions)) == N_EXTRACTION_QUESTIONS, "recovered set not distinct"
    recovered_norm = {_norm(q) for q in recovered_questions}
    pool = list(regenerated["eval_questions"]) + list(regenerated["extraction_questions"])
    eval_qs: list[str] = []
    seen: set[str] = set()
    for q in pool:
        nq = _norm(q)
        if nq in recovered_norm or nq in seen:
            continue
        seen.add(nq)
        eval_qs.append(q)
        if len(eval_qs) == N_EVAL_QUESTIONS:
            break
    if len(eval_qs) < N_EVAL_QUESTIONS:
        raise C.ArtifactCountShortfall(
            f"{trait}: only {len(eval_qs)} regenerated questions survive de-collision with "
            f"the recovered extraction set — regenerate"
        )
    out = {
        "instruction": regenerated["instruction"],
        "extraction_questions": list(recovered_questions),
        "eval_questions": eval_qs,
        "eval_prompt": regenerated["eval_prompt"],
    }
    # _validate_generated_artifacts-style checks on the SPLICED result.
    assert len(out["instruction"]) == 5, len(out["instruction"])
    for i, pair in enumerate(out["instruction"]):
        assert isinstance(pair, dict) and "pos" in pair and "neg" in pair, (trait, i, pair)
    assert len(out["extraction_questions"]) == N_EXTRACTION_QUESTIONS
    assert len(out["eval_questions"]) == N_EVAL_QUESTIONS
    assert isinstance(out["eval_prompt"], str) and out["eval_prompt"].strip()
    assert not ({_norm(q) for q in out["eval_questions"]} & recovered_norm), (
        f"{trait}: spliced eval_questions not disjoint from recovered extraction_questions"
    )
    return out


# ── rubric validation (the gate) ─────────────────────────────────────────────


def rejudge_sampled(
    trait: str,
    rollouts: dict[str, dict],
    sampled: list[str],
    raw_path: Path,
    n_draws: int = C.JUDGE_N_DRAWS,
) -> dict[str, float | None]:
    """Re-judge the sampled completions with the NEW rubric at N draws (sync path).

    The rubric is ``trait_judge_system_prompt(trait)`` reading the just-written
    SPLICED artifact. Each sampled rollout expands into ``n_draws`` items with
    distinct custom_ids (the judge_rollouts_n5 multi-sampling trick — the
    dispatch omits temperature so the API default 1.0 yields independent
    draws), routed through the project judge dispatcher's SYNC path
    (``dispatch_judge_items(force_sync=True)`` — multi-org polite caps per
    docs/api_throughput_guidelines.md). Raw score dicts persist to
    ``raw_path`` (persist-by-default). Returns {rollout_cid: new_mean|None}
    (None = all draws dropped, DROP-NEVER-COERCE).
    """
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items

    rubric = C.trait_judge_system_prompt(trait)
    items: list[tuple[str, str, str, str]] = []
    for rk in sampled:
        r = rollouts[rk]
        user_msg = C.trait_judge_user_msg(r["question"], r["completion"])
        for d in range(n_draws):
            items.append((f"{rk}__d{d}", r["question"], r["completion"], user_msg))
    logger.info(
        "[%s] re-judging %d rollouts x %d draws = %d sync calls",
        trait,
        len(sampled),
        n_draws,
        len(items),
    )
    raw = dispatch_judge_items(
        items,
        judge_model=C.JUDGE_MODEL,
        judge_system_prompt=rubric,
        max_tokens=256,
        force_sync=True,
    )
    C.write_json_atomic(
        raw_path,
        {
            "trait": trait,
            "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
            "judge_model": C.JUDGE_MODEL,
            "n_draws": n_draws,
            "raw_scores": raw,
        },
    )
    draw_scores = C._parse_raw_all_scores(raw)
    new_means: dict[str, float | None] = {}
    for rk in sampled:
        valid = [
            draw_scores[f"{rk}__d{d}"]
            for d in range(n_draws)
            if draw_scores.get(f"{rk}__d{d}") is not None
        ]
        new_means[rk] = float(statistics.fmean(valid)) if valid else None
    return new_means


def validation_stats(
    rollouts: dict[str, dict], sampled: list[str], new_means: dict[str, float | None]
) -> dict:
    """Spearman rho / MAE / >50-threshold agreement between new and parent means."""
    from scipy.stats import spearmanr

    pairs = [
        (rollouts[rk]["parent_mean"], new_means[rk]) for rk in sampled if new_means[rk] is not None
    ]
    assert pairs, "no comparable rollouts (all new-rubric draws dropped)"
    parent = [p for p, _ in pairs]
    new = [nv for _, nv in pairs]
    rho = float(spearmanr(parent, new).statistic)
    mae = float(sum(abs(p - nv) for p, nv in pairs) / len(pairs))
    agree = float(sum((p > 50) == (nv > 50) for p, nv in pairs) / len(pairs))
    bin_counts: dict[str, int] = {}
    for p, _ in pairs:
        b = min(int(p // 10), 9)
        bin_counts[f"{b * 10}-{b * 10 + 10}"] = bin_counts.get(f"{b * 10}-{b * 10 + 10}", 0) + 1
    return {
        "n_sampled": len(sampled),
        "n_compared": len(pairs),
        "n_new_dropped": len(sampled) - len(pairs),
        "spearman_rho": rho,
        "mae": mae,
        "threshold50_agreement": agree,
        "parent_bin_counts": dict(sorted(bin_counts.items())),
    }


# ── per-trait reconstruction flow ────────────────────────────────────────────


def assert_evil_sha_mapping(dispatches: dict[tuple[str, str], Path]) -> None:
    """Validate the trait<->dispatch mapping method against the evil control.

    Evil's artifacts are verbatim in code, so sha256(trait_judge_system_prompt
    ('evil')) is computable offline and must equal the judge_system_prompt_sha256
    the evil dispatches recorded — proving the custom_id-derived trait mapping
    and the rubric-construction path both match the parent's.
    """
    expect = hashlib.sha256(C.trait_judge_system_prompt("evil").encode()).hexdigest()
    checked = 0
    for arm in ("pos", "neg"):
        d = dispatches.get(("evil", arm))
        if d is None:
            continue
        got = _load_state(d)["judge_system_prompt_sha256"]
        assert got == expect, (
            f"evil {arm} dispatch judge sha {got} != code-built {expect} — the trait<->dispatch "
            "mapping or rubric construction diverges from the parent"
        )
        checked += 1
    assert checked, "no evil dispatch found — cannot validate the trait<->sha mapping"
    logger.info("[evil] trait<->sha mapping validated on %d dispatch(es)", checked)


def reconstruct_trait(
    trait: str,
    dispatches: dict[tuple[str, str], Path],
    out_dir: Path,
    *,
    n_sample: int,
    seed: int,
    rho_gate: float,
) -> dict:
    """Recover + regenerate + splice + rubric-validate one trait's artifacts.

    Writes the spliced artifact (with a top-level ``reconstruction`` metadata
    key) to the live cache ``data/issue_779/artifacts/<trait>.json`` only in
    fully-spliced form; on a final gate miss the cache file is QUARANTINED to
    ``<trait>.json.FAILED_VALIDATION`` so downstream loads fail loud. Returns
    the per-trait result dict (validated flag + per-attempt stats).
    """
    pos_dir = dispatches.get((trait, "pos"))
    neg_dir = dispatches.get((trait, "neg"))
    assert pos_dir is not None and neg_dir is not None, (
        f"{trait}: need BOTH pos+neg dispatches, got pos={pos_dir} neg={neg_dir}"
    )
    state_pos, state_neg = _load_state(pos_dir), _load_state(neg_dir)
    assert state_pos["judge_system_prompt_sha256"] == state_neg["judge_system_prompt_sha256"], (
        f"{trait}: pos/neg dispatches used different judge rubrics — wrong pairing"
    )
    pos_items, neg_items = _load_items(pos_dir), _load_items(neg_dir)
    recovered = recover_extraction_questions(pos_items, neg_items, trait)
    logger.info("[%s] recovered %d verbatim extraction questions", trait, len(recovered))

    rollouts: dict[str, dict] = {}
    rollouts.update(parent_rollouts(pos_items, load_parent_scores(pos_dir, pos_items)))
    rollouts.update(parent_rollouts(neg_items, load_parent_scores(neg_dir, neg_items)))
    n_dropped_parent = sum(1 for r in rollouts.values() if r["parent_mean"] is None)
    logger.info(
        "[%s] reconstructed parent scores for %d rollouts (%d dropped: 0 valid draws)",
        trait,
        len(rollouts),
        n_dropped_parent,
    )
    sampled = stratified_sample(rollouts, n_sample, seed)

    cache = C._artifacts_dir() / f"{trait}.json"
    source_meta = {
        "pos": {
            "fingerprint": state_pos["fingerprint"],
            "judge_system_prompt_sha256": state_pos["judge_system_prompt_sha256"],
        },
        "neg": {
            "fingerprint": state_neg["fingerprint"],
            "judge_system_prompt_sha256": state_neg["judge_system_prompt_sha256"],
        },
    }
    attempts: list[dict] = []
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        C.phase(f"reconstruct_{trait}_attempt{attempt}")
        with _redirected_artifacts_dir(out_dir / "regen_scratch"):
            regenerated = C.generate_extraction_artifacts(trait, force=True)
        spliced = splice_artifacts(regenerated, recovered, trait)
        spliced["reconstruction"] = {
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recovered_verbatim": ["extraction_questions"],
            "regenerated": ["instruction", "eval_questions", "eval_prompt"],
            "source_dispatches": source_meta,
            "attempt": attempt,
            "validated": False,  # flipped below on a gate pass
            "rubric_validation": None,
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_reconstruct_artifacts", "trait": trait}
            ),
        }
        C.write_json_atomic(cache, spliced)  # spliced-only write; validate reads through it
        raw_path = out_dir / f"{trait}_rejudge_raw_attempt{attempt}.json"
        new_means = rejudge_sampled(trait, rollouts, sampled, raw_path)
        stats = validation_stats(rollouts, sampled, new_means)
        stats["attempt"] = attempt
        attempts.append(stats)
        C.write_json_atomic(
            out_dir / f"{trait}_validation_attempt{attempt}.json",
            {
                "trait": trait,
                "stats": stats,
                "rho_gate": rho_gate,
                "gate_pass": stats["spearman_rho"] >= rho_gate,
                "sampled": {
                    rk: {"parent_mean": rollouts[rk]["parent_mean"], "new_mean": new_means[rk]}
                    for rk in sampled
                },
            },
        )
        logger.info(
            "[%s] attempt %d: rho=%.3f mae=%.2f agree@50=%.3f (n=%d, gate %.2f) -> %s",
            trait,
            attempt,
            stats["spearman_rho"],
            stats["mae"],
            stats["threshold50_agreement"],
            stats["n_compared"],
            rho_gate,
            "PASS" if stats["spearman_rho"] >= rho_gate else "FAIL",
        )
        if stats["spearman_rho"] >= rho_gate:
            spliced["reconstruction"]["validated"] = True
            spliced["reconstruction"]["rubric_validation"] = stats
            C.write_json_atomic(cache, spliced)
            return {
                "trait": trait,
                "validated": True,
                "stats": stats,
                "attempts": attempts,
                "cache": str(cache),
            }

    # Gate missed on every attempt: quarantine so load_extraction_artifacts fails
    # loud (never a silent consume of an unvalidated rubric), STOP with numbers.
    quarantine = cache.with_name(cache.name + ".FAILED_VALIDATION")
    cache.rename(quarantine)
    logger.error(
        "[%s] rubric validation FAILED the rho>=%.2f gate on %d attempts: %s — artifact "
        "quarantined at %s",
        trait,
        rho_gate,
        MAX_GENERATION_ATTEMPTS,
        [round(a["spearman_rho"], 3) for a in attempts],
        quarantine,
    )
    return {
        "trait": trait,
        "validated": False,
        "attempts": attempts,
        "quarantine": str(quarantine),
    }


# ── upload ───────────────────────────────────────────────────────────────────


def upload_artifacts(results: list[dict], out_dir: Path) -> list[str]:
    """Upload validated artifacts + validation records in ONE upload_folder commit.

    Layout: ``issue779_monitoring/artifacts/<trait>.json`` (the driver fallback
    path) + ``.../artifacts/reconstruction_validation/*`` (the rejudge raw
    scores + per-attempt stats — small text, always-upload policy). Verified
    against a fresh list_repo_files (retried on transient failures — recursive
    tree listings 504 un-retried upstream). Returns the verified repo paths.
    """
    import tempfile

    from huggingface_hub import HfApi, list_repo_files

    validated = [r for r in results if r["validated"]]
    assert validated, "nothing to upload (no trait passed the rubric gate)"
    with tempfile.TemporaryDirectory(prefix="issue779_artifacts_") as tmp:
        staging = Path(tmp)
        expected: list[str] = []
        for r in validated:
            src = Path(r["cache"])
            (staging / src.name).write_bytes(src.read_bytes())
            expected.append(f"{HF_ARTIFACTS_PREFIX}/{src.name}")
        val_dir = staging / "reconstruction_validation"
        val_dir.mkdir()
        for p in sorted(out_dir.glob("*_validation_attempt*.json")) + sorted(
            out_dir.glob("*_rejudge_raw_attempt*.json")
        ):
            (val_dir / p.name).write_bytes(p.read_bytes())
            expected.append(f"{HF_ARTIFACTS_PREFIX}/reconstruction_validation/{p.name}")
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging),
            path_in_repo=HF_ARTIFACTS_PREFIX,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message="issue779 r5: reconstructed extraction artifacts + rubric validation",
        )
        repo_files: set[str] | None = None
        last_err: Exception | None = None
        for i in range(3):  # transient-5xx retry (tree listings 504 un-retried upstream)
            try:
                repo_files = set(list_repo_files(C.HF_DATA_REPO, repo_type="dataset"))
                break
            except Exception as e:
                last_err = e
                logger.warning("list_repo_files attempt %d failed: %r", i + 1, e)
                time.sleep(10 * (i + 1))
        if repo_files is None:
            raise RuntimeError(f"upload verification listing failed 3x: {last_err!r}")
        missing = [p for p in expected if p not in repo_files]
        if missing:
            raise RuntimeError(f"artifact upload verification FAILED, missing: {missing}")
    logger.info("uploaded + verified %d files under %s", len(expected), HF_ARTIFACTS_PREFIX)
    return expected


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #779 r5: reconstruct + rubric-validate the syc/halluc PV artifacts."
    )
    parser.add_argument("--traits", nargs="+", default=["sycophancy", "hallucination"])
    parser.add_argument("--dispatch-root", type=Path, default=DEFAULT_DISPATCH_ROOT)
    parser.add_argument("--n-sample", type=int, default=DEFAULT_N_SAMPLE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho-gate", type=float, default=DEFAULT_RHO_GATE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_779" / "artifacts_reconstruction",
    )
    parser.add_argument("--no-upload", action="store_true", help="skip the HF upload (dev only)")
    args = parser.parse_args()

    for trait in args.traits:
        assert trait in ("sycophancy", "hallucination"), (
            f"only the generated traits are reconstructable, got {trait!r} "
            "(evil is verbatim in code — nothing to reconstruct)"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dispatches = scan_dispatches(args.dispatch_root)
    logger.info("found dispatches: %s", {k: v.name for k, v in dispatches.items()})
    assert_evil_sha_mapping(dispatches)

    results = [
        reconstruct_trait(
            trait,
            dispatches,
            args.out_dir,
            n_sample=args.n_sample,
            seed=args.seed,
            rho_gate=args.rho_gate,
        )
        for trait in args.traits
    ]

    uploaded: list[str] = []
    any_validated = any(r["validated"] for r in results)
    if any_validated and not args.no_upload:
        C.phase("upload")
        uploaded = upload_artifacts(results, args.out_dir)

    summary = {
        "results": results,
        "uploaded": uploaded,
        "rho_gate": args.rho_gate,
        "n_sample": args.n_sample,
        "seed": args.seed,
        "metadata": C.reproducibility_metadata({"script": "issue779_reconstruct_artifacts"}),
    }
    C.write_json_atomic(args.out_dir / "reconstruction_summary.json", summary)
    for r in results:
        line = ", ".join(
            f"attempt{a['attempt']}: rho={a['spearman_rho']:.3f} mae={a['mae']:.2f} "
            f"agree@50={a['threshold50_agreement']:.3f}"
            for a in r["attempts"]
        )
        logger.info("[summary] %s validated=%s (%s)", r["trait"], r["validated"], line)

    if not all(r["validated"] for r in results):
        logger.error("STOP: at least one trait failed the rubric-validation gate — see summary")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
