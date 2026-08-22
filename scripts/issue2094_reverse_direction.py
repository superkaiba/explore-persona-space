#!/usr/bin/env python3
"""Issue #2094 — REVERSED matched-query patch direction: persona(pirate) -> bare.

The parent bank (``issue2094_run.py``) runs every pair in ONE canonical
lexicographic direction (prefixes bare < persona < conv), so ``bare`` is never
a patch TARGET. This round measures the fraction-of-swap F for the reversed
direction — patching the context-end position of a ``persona__q<i>`` context
toward its ``bare__q<i>`` twin — with the full-state ``replace`` dose only.

Cells: 5 reversed matched-query pairs (A = persona__q{1..5}, B = bare__q{1..5})
x slot ``ce`` x 30 layer variants (L0..L27 + joint_mid + joint_all) x dose
``replace`` x arms {steered, null} = 300 hooked greedy rollouts.

Phases (``--phase {bank,grid,upload}``; NO anchors phase):

- ``bank`` — delegates verbatim to the parent's ``phase_bank`` (same 15-context
  all-layer V-bank capture, injection-exactness gate, capture-parity legs,
  skip-if-done resume) on THIS run's ``--out-root``.
- ``grid`` — (a) computes the reversed-direction floor/ceiling from the
  parent's EXISTING per-draw anchor judge scores (fp-bare / fp-persona rubric
  waves; per-draw Delta = (fp-bare - fp-persona)/100, coherent-only per the
  parent's ``anchor_draws.jsonl`` convention; floor = mean over the anchor
  draws of ``persona__q<i>``, ceiling = mean over the draws of ``bare__q<i>``)
  and persists ``rev_floor_ceiling.json`` BEFORE the model load — NO new
  anchor generation and NO new anchor judging; (b) runs the 60 (steered,
  null) blocks with per-block JSONL checkpointing + full-regime resume.
  ``--pilot`` runs ONE cell through this same entrypoint and prints the
  measured per-cell wall (the sizing basis).
- ``upload`` — bulk ``upload_folder`` commits (parent helper) under
  ``issue2094_singlepos/raw_completions/rev_direction/`` + the pod sentinel.

Null-arm donors: all 5 reversed pairs share the (bare, persona) prefix pair,
so a derangement WITHIN the set cannot satisfy the parent's
different-prefix-pair constraint — donors are drawn (seeded, seed 2094, no
self, donor prefix pair != recipient prefix pair) from the parent's 10 OTHER
matched-query pairs (prefix pairs (bare, conv) / (persona, conv)), realized
exactly as the parent's replace-arm null: the donor pair's TARGET-CONTEXT
STATE ``norm_match(V_B(donor), V_B(recipient))`` (``_donor_payload``,
``payload_kind="state"``), with the realized ``donor_pair_id`` recorded per
null row.

Pod-side contract: sentinel file + ``[phase=...]`` breadcrumbs ONLY — never a
``scripts/task.py`` shellout. Every phase ends with an explicit ``sys.exit``
(the #1689 interpreter-finalization race).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2094.fmetrics import delta_contrast  # noqa: E402

# Sibling-script import (script mode puts scripts/ on sys.path[0]; tests and
# -c mode need the explicit insert — the issue2094_smoke_support convention).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_run as RUN  # noqa: E402

logger = logging.getLogger("issue2094.rev")

# ── constants ─────────────────────────────────────────────────────────

REV_SEED = BANK.SEED  # 2094 — the parent's seed convention
REV_SLOT = "ce"
REV_DOSE = "replace"
REV_VEC_TYPE = "A"
REV_PREFIX_PAIR: tuple[str, str] = ("bare", "persona")  # every reversed pair's prefix pair
DEFAULT_OUT_ROOT = Path("/workspace/issue2094_rev_out")
SENTINEL_NAME = "issue-2094-rev-direction-results.json"
REV_HF_PREFIX = f"{RUN.HF_PREFIX}/raw_completions/rev_direction"

# The two prefix-rubric anchor score waves this direction consumes (banked by
# the parent's judge unit; observed schema: rows carry rubric_id, kind="anchor",
# context_id, draw, score).
REV_RUBRIC_B = "fp-bare"  # rubric of the pairs' B side (the patch target)
REV_RUBRIC_A = "fp-persona"  # rubric of the pairs' A side (the patched context)
SCORES_FILES: dict[str, str] = {
    REV_RUBRIC_B: "fp-bare.anchors.scores.jsonl",
    REV_RUBRIC_A: "fp-persona.anchors.scores.jsonl",
}
HF_SCORES_PREFIX = f"{RUN.HF_PREFIX}/raw_completions/judge_raw/scores"
DEFAULT_COHERENCE_PATH = (
    RUN.REPO_ROOT / "eval_results" / "issue_2094" / "f_metrics" / "anchor_draws.jsonl"
)


# ── reversed pairs + donor assignment (CPU-pure, unit-tested) ─────────


def build_rev_pairs() -> list[BANK.Pair]:
    """The 5 reversed matched-query pairs: A = persona__q<i>, B = bare__q<i>.

    Canonical definition moved to :func:`BANK.build_rev_pairs` so the judge's
    pair registry (``issue2094_judge.py::pair_index``) can register the
    ``mqrev--`` ids WITHOUT importing this torch-bearing pod driver (the
    2026-08-13 KeyError fix). This delegating alias keeps every existing
    caller/test import working; the local prefix-pair invariant stays as belt.
    """
    pairs = BANK.build_rev_pairs()
    for p in pairs:
        assert p.prefix_pair() == REV_PREFIX_PAIR, p
    return pairs


def rev_donor_pool(parent_pairs: list[BANK.Pair]) -> list[BANK.Pair]:
    """The 10 parent matched-query pairs whose prefix pair differs from
    (bare, persona) — the only pool satisfying the different-prefix-pair
    null constraint (all 5 reversed pairs share one prefix pair, so a
    within-set derangement is structurally impossible)."""
    pool = sorted(
        (
            p
            for p in parent_pairs
            if p.setting == "matched_query" and p.prefix_pair() != REV_PREFIX_PAIR
        ),
        key=lambda p: p.pair_id,
    )
    assert len(pool) == 10, [p.pair_id for p in pool]
    return pool


def rev_donor_assignment(
    rev_pairs: list[BANK.Pair],
    parent_pairs: list[BANK.Pair],
    seed: int = REV_SEED,
) -> dict[str, str]:
    """Seeded donor map (reversed pair_id -> parent mq pair_id), constraints asserted.

    Distinct donors (a seeded no-replacement sample over the sorted pool), no
    self-donation, donor prefix pair != recipient prefix pair, and the
    parent's state-kind eligibility (donor.b never the recipient's target
    context — ``_donor_eligible(..., "state")``) since every reversed cell is
    a ``replace`` (state-payload) cell.
    """
    pool = rev_donor_pool(parent_pairs)
    rng = random.Random(seed)
    donor_ids = rng.sample([p.pair_id for p in pool], k=len(rev_pairs))
    by_id = {p.pair_id: p for p in pool}
    out: dict[str, str] = {}
    for pair, donor_id in zip(rev_pairs, donor_ids, strict=True):
        donor = by_id[donor_id]
        assert donor.pair_id != pair.pair_id, (pair.pair_id, donor.pair_id)
        assert donor.prefix_pair() != pair.prefix_pair(), (pair.pair_id, donor.pair_id)
        assert RUN._donor_eligible(donor, REV_SLOT, pair, "state"), (
            pair.pair_id,
            donor.pair_id,
        )
        out[pair.pair_id] = donor_id
    return out


def enumerate_rev_blocks(
    rev_pairs: list[BANK.Pair], n_layers: int
) -> list[tuple[RUN.Block, RUN.Block]]:
    """30 (steered, null) families: ce x every layer variant x replace x Type A."""
    ids = tuple(p.pair_id for p in rev_pairs)
    assert ids, "empty reversed pair set"
    families = [
        (
            RUN.Block(REV_SLOT, variant, REV_DOSE, REV_VEC_TYPE, "steered", ids),
            RUN.Block(REV_SLOT, variant, REV_DOSE, REV_VEC_TYPE, "null", ids),
        )
        for variant in RUN.layer_variant_names(n_layers)
    ]
    keys = [b.key for fam in families for b in fam]
    assert len(set(keys)) == len(keys), "duplicate reversed block keys"
    return families


def smoke_rev_blocks(
    rev_pairs: list[BANK.Pair], n_layers: int
) -> list[tuple[RUN.Block, RUN.Block]]:
    """Tiny per-arm-class slice: one single-layer + joint_mid + joint_all family
    (both arms each — the donor-null path runs in every class)."""
    variants = RUN.layer_variant_names(n_layers)
    keep = {variants[n_layers // 2], "joint_mid", "joint_all"}
    return [f for f in enumerate_rev_blocks(rev_pairs, n_layers) if f[0].layer_variant in keep]


def rev_regime_fingerprint(cfg: RUN.RunConfig, bank_sha: str, donor_map: dict[str, str]) -> str:
    """Resume key: the parent regime fingerprint + the reversed grid's identity.

    Distinct from the parent's fingerprint BY CONSTRUCTION, so a parent-grid
    done-file (same block-key space: ``ce|L*|replace|A|*``) can never satisfy
    a reversed-grid resume even under a mis-pointed ``--out-root``.
    """
    import hashlib

    payload = json.dumps(
        {
            "base": RUN.regime_fingerprint(cfg, bank_sha),
            "grid": "rev_direction_v1",
            "pairs": [p.pair_id for p in build_rev_pairs()],
            "donors": donor_map,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── floor / ceiling from the parent's banked anchor judge scores ──────


def load_anchor_scores(
    scores_dir: Path | None, out_root: Path
) -> dict[str, dict[tuple[str, int], float]]:
    """Per-draw anchor judge scores for the two prefix rubrics.

    ``scores_dir`` given: read ``fp-{bare,persona}.anchors.scores.jsonl`` from
    it (e.g. the /tmp/i2094_shards local mirror). ``None``: fetch the two
    files from the HF data repo (retry-wrapped) into ``out_root/scores_dl``.
    Returns ``{rubric_id: {(context_id, draw): score}}``; duplicate keys,
    wrong-rubric rows, and out-of-range scores fail loud.
    """
    paths: dict[str, Path] = {}
    if scores_dir is None:
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate.hub import retry_transient

        for rid, fname in SCORES_FILES.items():
            remote = f"{HF_SCORES_PREFIX}/{fname}"
            local = retry_transient(
                lambda remote=remote: hf_hub_download(
                    repo_id=RUN.HF_DATA_REPO,
                    repo_type="dataset",
                    filename=remote,
                    local_dir=out_root / "scores_dl",
                ),
                what=f"hf_hub_download {remote}",
            )
            paths[rid] = Path(local)
    else:
        for rid, fname in SCORES_FILES.items():
            paths[rid] = scores_dir / fname
    out: dict[str, dict[tuple[str, int], float]] = {}
    for rid, path in paths.items():
        assert path.is_file(), f"missing anchor score shard: {path}"
        table: dict[tuple[str, int], float] = {}
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["rubric_id"] == rid, (row["rubric_id"], rid, str(path))
            assert row["kind"] == "anchor", (row.get("kind"), str(path))
            key = (str(row["context_id"]), int(row["draw"]))
            assert key not in table, f"duplicate anchor score row {key} in {path}"
            score = float(row["score"])
            assert 0.0 <= score <= 100.0, (key, score, str(path))
            table[key] = score
        assert table, f"no anchor score rows parsed from {path}"
        out[rid] = table
    return out


def load_coherent_draws(coherence_path: Path) -> dict[str, dict]:
    """Coherent anchor draws per context from the parent's ``anchor_draws.jsonl``.

    Returns ``{context_id: {"coherent": [draw, ...], "n_total": int}}``.
    """
    assert coherence_path.is_file(), (
        f"{coherence_path} missing — committed at eval_results/issue_2094/f_metrics/; "
        "on a sparse pod clone run `git sparse-checkout add eval_results/issue_2094` "
        "(or set BOOTSTRAP_EXTRA_CONES at provision)"
    )
    out: dict[str, dict] = {}
    seen: set[tuple[str, int]] = set()
    for line in coherence_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["context_id"]), int(row["draw"]))
        assert key not in seen, f"duplicate coherence row {key} in {coherence_path}"
        seen.add(key)
        rec = out.setdefault(key[0], {"coherent": [], "n_total": 0})
        rec["n_total"] += 1
        if bool(row["coherent"]):
            rec["coherent"].append(key[1])
    for rec in out.values():
        rec["coherent"].sort()
    assert out, f"no rows parsed from {coherence_path}"
    return out


def compute_rev_floor_ceiling(
    scores: dict[str, dict[tuple[str, int], float]],
    coherent: dict[str, dict],
    rev_pairs: list[BANK.Pair],
) -> dict:
    """Per-pair floor/ceiling for the reversed direction from banked anchor scores.

    Per-draw Delta = (fp-bare - fp-persona)/100 (``delta_contrast``); floor =
    coherent-draw mean under context A (persona__q<i>), ceiling = coherent-draw
    mean under context B (bare__q<i>). Any missing (rubric, context, draw)
    score raises — never a silent default.
    """
    per_pair: list[dict] = []
    for pair in rev_pairs:
        sides: dict[str, dict] = {}
        for role, cid in (("floor", pair.a), ("ceiling", pair.b)):
            assert cid in coherent, (
                f"context {cid} absent from the anchor coherence file — cannot "
                f"compute the {role} for {pair.pair_id}"
            )
            draws = coherent[cid]["coherent"]
            assert draws, f"context {cid} has zero coherent anchor draws ({role})"
            missing = [
                (rid, cid, d)
                for d in draws
                for rid in (REV_RUBRIC_B, REV_RUBRIC_A)
                if (cid, d) not in scores[rid]
            ]
            assert not missing, f"missing anchor judge score rows {missing} — never default a score"
            judge_b = torch.tensor(
                [scores[REV_RUBRIC_B][(cid, d)] for d in draws], dtype=torch.float64
            )
            judge_a = torch.tensor(
                [scores[REV_RUBRIC_A][(cid, d)] for d in draws], dtype=torch.float64
            )
            delta_contrast(judge_b, judge_a)  # fail-loud range/finiteness validation
            # float64 end-to-end (delta_contrast RETURNS float32 — ~1e-8 off
            # an exact recompute, measured on the real banked shards).
            deltas = (judge_b - judge_a) / 100.0
            sides[role] = {
                "context_id": cid,
                "draws_coherent": list(draws),
                "n_draws_total": coherent[cid]["n_total"],
                "n_draws_coherent": len(draws),
                "per_draw_delta": [float(x) for x in deltas],
                "delta_mean": float(deltas.mean()),
            }
        per_pair.append(
            {
                "pair_id": pair.pair_id,
                "setting": pair.setting,
                "context_a": pair.a,
                "context_b": pair.b,
                "floor": sides["floor"],
                "ceiling": sides["ceiling"],
                "denominator": sides["ceiling"]["delta_mean"] - sides["floor"]["delta_mean"],
            }
        )
    return {
        "direction": "persona->bare (reversed matched-query)",
        "delta_definition": (
            f"per-draw ({REV_RUBRIC_B} - {REV_RUBRIC_A})/100 over coherent anchor draws; "
            "floor = mean under context A (persona__q<i>), ceiling = mean under "
            "context B (bare__q<i>) — the parent's existing anchor draws, no new "
            "anchor generation or judging"
        ),
        "rubric_b": REV_RUBRIC_B,
        "rubric_a": REV_RUBRIC_A,
        "pairs": per_pair,
    }


# ── the reversed grid ─────────────────────────────────────────────────


@torch.no_grad()
def run_rev_block(
    cfg: RUN.RunConfig,
    model,
    tok,
    bank: dict,
    block: RUN.Block,
    rev_pairs_by_id: dict[str, BANK.Pair],
    parent_pairs_by_id: dict[str, BANK.Pair],
    donor_by_id: dict[str, str],
    regime_fp: str,
    *,
    write_done: bool = True,
    shard_prefix: str = "shard_",
) -> dict:
    """One reversed block: hooked greedy rollouts for every cell (no V_a pass).

    Mirrors the parent's ``run_block`` geometry (same hook arming, same
    history-aware render, same shard row schema minus the capture store);
    ``write_done=False`` + a non-``shard_`` prefix isolate the ``--pilot``
    timing cell from the production resume/upload globs.
    """
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2094(tok, contexts[cid])
        return ctx_ids_cache[cid]

    layers = RUN.layer_variant_layers(block.layer_variant, cfg.n_layers)
    mode, alpha, payload_kind = RUN._realized_mode(block.slot, block.dose, block.vec_type)
    assert (mode, alpha, payload_kind) == ("replace", 1.0, "state"), (
        f"the reversed grid is context-end full-state replace only, got {block.key}"
    )
    recs = bank["per_context"]
    cells: list[dict] = []
    for pid in block.pair_ids:
        pair = rev_pairs_by_id[pid]
        _delta, state, m = RUN._pair_payload(bank, pair, block.slot, block.vec_type)
        recipient = state  # payload_kind == "state": the full-state patch V_B
        donor_label = None
        if block.arm == "null":
            donor = parent_pairs_by_id[donor_by_id[pid]]
            assert RUN._donor_eligible(donor, block.slot, pair, payload_kind), (
                pid,
                donor.pair_id,
            )
            # The parent's replace-arm null realization: the donor pair's
            # TARGET-CONTEXT STATE V_B(donor), norm-matched to the recipient's
            # V_B (bank.norm_match inside _donor_payload).
            recipient, donor_label = RUN._donor_payload(
                bank, pair, donor, block.slot, block.vec_type, recipient, payload_kind
            )
        rec = recs[pair.a]
        pos = RUN.slot_positions(rec["ctx_len"], rec["prefix_end"], block.slot)[-m:]
        assert recipient.shape[0] == len(pos) == 1, (recipient.shape, pos)
        cells.append(
            {
                "pair_id": pid,
                "setting": pair.setting,
                "context_a": pair.a,
                "context_b": pair.b,
                "positions": list(pos),
                "payload": recipient,
                "donor_pair_id": donor_label,
            }
        )

    texts: list[str] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        hook = RUN._arm_hook_for_rows(
            model,
            cfg,
            layers,
            row_lengths,
            [tuple(c["positions"]) for c in chunk],
            [c["payload"] for c in chunk],
            mode,
            alpha,
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=1,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=RUN.GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                # History-aware render (bank.py module note): mandatory for
                # any conv context; kept for exact parent parity even though
                # every reversed context_a is single-turn (persona__q<i>).
                render_fn=BANK.render_context_2094,
                ids_fn=BANK.context_token_ids_2094,
            )
        finally:
            hook.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts.extend(o[0] for o in outs)
    assert len(texts) == len(cells), (len(texts), len(cells))

    rows_out: list[dict] = []
    for cell, text in zip(cells, texts, strict=True):
        # Re-tokenized completion length — the parent's cap-hit proxy basis.
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
        rows_out.append(
            {
                "block_key": block.key,
                "slot": block.slot,
                "layer_variant": block.layer_variant,
                "layers": list(layers),
                "dose": block.dose,
                "alpha": alpha,
                "hook_mode": mode,
                "realized_mode": "replace",
                "vec_type": block.vec_type,
                "arm": block.arm,
                "pair_id": cell["pair_id"],
                "setting": cell["setting"],
                "context_a": cell["context_a"],
                "context_b": cell["context_b"],
                "positions": cell["positions"],
                "donor_pair_id": cell["donor_pair_id"],
                "temperature": RUN.GRID_TEMPERATURE,
                "seed": cfg.seed_base,
                "n_completion_tokens": n_tok,
                "cap_hit": RUN.cap_hit(n_tok, cfg.max_new_tokens),
                "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                "text": text,
            }
        )
    RUN._write_jsonl_atomic(cfg.rollouts_dir / f"{shard_prefix}{block.slug}.jsonl", rows_out)
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_cells": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "repro": RUN._repro(cfg),
    }
    if write_done:
        RUN._write_json_atomic(RUN.block_done_path(cfg.out_root, block), done)
    return done


def _upload_rev_increment(cfg: RUN.RunConfig, blocks: list[RUN.Block]) -> list[str]:
    """Incremental per-block-batch text upload under the rev_direction prefix."""
    slugs = [b.slug for b in blocks if (cfg.rollouts_dir / f"shard_{b.slug}.jsonl").exists()]
    if not slugs:
        return []
    return RUN._upload_dir(
        cfg, cfg.rollouts_dir, REV_HF_PREFIX, [f"shard_{s}.jsonl" for s in slugs]
    )


def phase_grid(cfg: RUN.RunConfig, args: argparse.Namespace) -> int:
    """The 300-cell reversed grid (or the 1-cell ``--pilot`` timing run)."""
    logger.info(
        "[phase=rev-grid] worker=%d/%d smoke=%s pilot=%s",
        cfg.worker_index,
        cfg.num_workers,
        cfg.smoke,
        cfg.pilot,
    )
    bank = RUN._load_bank(cfg)
    rev_pairs = build_rev_pairs()
    parent_pairs = BANK.build_pairs()
    rev_by_id = {p.pair_id: p for p in rev_pairs}
    parent_by_id = {p.pair_id: p for p in parent_pairs}
    donor_by_id = rev_donor_assignment(rev_pairs, parent_pairs)
    regime_fp = rev_regime_fingerprint(cfg, str(bank.get("bank_sha")), donor_by_id)

    # Floor/ceiling from the parent's banked anchor judge scores — CPU-only,
    # BEFORE the model load, so a scores/coherence contract failure never
    # burns a GPU model load.
    fc = compute_rev_floor_ceiling(
        load_anchor_scores(args.scores_dir, cfg.out_root),
        load_coherent_draws(args.coherence_path),
        rev_pairs,
    )
    RUN._write_json_atomic(
        cfg.manifest_dir / "rev_floor_ceiling.json", {**fc, "repro": RUN._repro(cfg)}
    )
    logger.info("[rev-grid] floor/ceiling persisted for %d pairs", len(fc["pairs"]))

    all_families = enumerate_rev_blocks(rev_pairs, cfg.n_layers)
    totals_all = RUN.grid_totals(all_families)
    families = smoke_rev_blocks(rev_pairs, cfg.n_layers) if cfg.smoke else all_families
    if cfg.pilot:
        families = families[:1]
    blocks = RUN.blocks_for_worker(families, cfg.worker_index, cfg.num_workers)
    totals = RUN.grid_totals(families)
    RUN._write_json_atomic(
        cfg.manifest_dir / f"rev_grid_plan_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "smoke": cfg.smoke,
            "pilot": cfg.pilot,
            "totals_full_grid": totals_all,
            "totals_this_run": totals,
            "n_blocks_this_worker": len(blocks),
            "n_cells_this_worker": sum(b.n_cells for b in blocks),
            "block_keys": [b.key for b in blocks],
            "donor_assignment": donor_by_id,
            "donor_pool": [p.pair_id for p in rev_donor_pool(parent_pairs)],
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[rev-grid] full grid: %d blocks / %d cells; this run: %d blocks / %d cells",
        totals_all["n_blocks"],
        totals_all["cells_total"],
        len(blocks),
        sum(b.n_cells for b in blocks),
    )
    assert blocks, "reversed grid resolved to zero blocks (never silently no-op)"

    model, tok = RUN.load_model_and_tokenizer(cfg)

    if cfg.pilot:
        steered = blocks[0]
        assert steered.arm == "steered", steered.key
        pilot_block = RUN.Block(
            steered.slot,
            steered.layer_variant,
            steered.dose,
            steered.vec_type,
            steered.arm,
            steered.pair_ids[:1],
        )
        t0 = time.monotonic()
        rec = run_rev_block(
            cfg,
            model,
            tok,
            bank,
            pilot_block,
            rev_by_id,
            parent_by_id,
            donor_by_id,
            regime_fp,
            write_done=False,
            shard_prefix="pilot_shard_",
        )
        wall = time.monotonic() - t0
        per_cell = wall / rec["n_cells"]
        projected_h = per_cell * totals_all["cells_total"] / max(1, cfg.num_workers) / 3600.0
        RUN._write_json_atomic(
            cfg.out_root / "rev_pilot_report.json",
            {
                "criterion": "reversed-grid timing pilot (sizing basis)",
                "block_key": pilot_block.key,
                "measured_cells": rec["n_cells"],
                "measured_wall_s": wall,
                "s_per_cell": per_cell,
                "gen_batch": cfg.gen_batch,
                "num_workers": cfg.num_workers,
                "cells_total": totals_all["cells_total"],
                "projected_wall_h": projected_h,
                "repro": RUN._repro(cfg),
            },
        )
        logger.info(
            "[rev-pilot] s_per_cell=%.3f cells_total=%d projected_wall_h=%.3f",
            per_cell,
            totals_all["cells_total"],
            projected_h,
        )
        logger.info("[phase=rev-pilot_done]")
        return RUN.RC_OK

    n_total = len(blocks)
    done_count = 0
    ran_cells = 0
    uploaded: list[str] = []
    pending: list[RUN.Block] = []
    for k, block in enumerate(blocks, start=1):
        if RUN.block_is_done(cfg.out_root, block, regime_fp):
            done_count += 1
            logger.info("[rev-grid] block %d/%d %s SKIP (done)", k, n_total, block.key)
            continue
        t0 = time.monotonic()
        rec = run_rev_block(
            cfg, model, tok, bank, block, rev_by_id, parent_by_id, donor_by_id, regime_fp
        )
        elapsed = time.monotonic() - t0
        ran_cells += rec["n_cells"]
        pending.append(block)
        logger.info("[rev-grid] block %d/%d %s elapsed=%.1fs", k, n_total, block.key, elapsed)
        if cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_rev_increment(cfg, pending)
            pending = []
    if pending:
        uploaded += _upload_rev_increment(cfg, pending)

    RUN._write_json_atomic(
        cfg.manifest_dir / f"rev_grid_done_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "n_blocks": n_total,
            "n_blocks_skipped": done_count,
            "n_cells_run": ran_cells,
            "uploads": uploaded,
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[phase=rev-grid_done] worker=%d blocks=%d cells=%d", cfg.worker_index, n_total, ran_cells
    )
    return RUN.RC_OK


# ── upload + sentinel ─────────────────────────────────────────────────


def _sentinel_payload(cfg: RUN.RunConfig, uploaded: dict[str, list[str]]) -> dict:
    """The /issue Step 7 results payload (all 10 keys), reversed-grid numbers."""
    n_shards = len(list(cfg.rollouts_dir.glob("shard_*.jsonl")))
    cap_hits, cap_total = 0, 0
    for done in sorted((cfg.manifest_dir / "blocks").glob("*.done.json")):
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        cap_total += int(rec.get("n_cells", 0))
    fc_path = cfg.manifest_dir / "rev_floor_ceiling.json"
    return {
        "eval_numbers": {
            "rev_grid_shards": n_shards,
            "cells_persisted": cap_total,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / cap_total) if cap_total else 0.0,
            "floor_ceiling_pairs": (
                len(json.loads(fc_path.read_text())["pairs"]) if fc_path.exists() else 0
            ),
        },
        "eval_paths": sorted(
            {
                str(cfg.rollouts_dir),
                str(fc_path),
                str(cfg.manifest_dir),
            }
        ),
        "reproducibility_card": {
            **RUN._repro(cfg),
            "seed_base": cfg.seed_base,
            "rev_seed": REV_SEED,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": RUN.GRID_TEMPERATURE,
            "gen_batch": cfg.gen_batch,
            "num_workers": cfg.num_workers,
            "slot": REV_SLOT,
            "dose": REV_DOSE,
            "vec_type": REV_VEC_TYPE,
        },
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{RUN.HF_DATA_REPO}/tree/main/{REV_HF_PREFIX}",
        "worktree_path": str(RUN.REPO_ROOT),
        "final_commit_sha": RUN._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "reversed-direction floor/ceiling reuse the parent's banked anchor judge "
            "scores (fp-bare/fp-persona waves + anchor_draws.jsonl coherence) — no new "
            "anchor generation and no new anchor judging in this round",
            "null donors come from the parent's 10 cross-prefix-pair matched-query "
            "pairs (seeded sample, seed 2094): all 5 reversed pairs share the "
            "(bare, persona) prefix pair, so a within-set derangement cannot satisfy "
            "the parent's different-prefix-pair constraint; realized as the parent's "
            "replace-arm donor-STATE null (norm_match(V_B(donor), V_B)), donor_pair_id "
            "recorded per null row",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }


def phase_upload(cfg: RUN.RunConfig) -> int:
    """Bulk-upload the staged reversed-grid text + manifests, then the sentinel."""
    logger.info("[phase=upload]")
    uploaded: dict[str, list[str]] = {}
    uploaded["rev_text"] = RUN._upload_dir(cfg, cfg.rollouts_dir, REV_HF_PREFIX, ["shard_*.jsonl"])
    uploaded["rev_manifests"] = RUN._upload_dir(
        cfg, cfg.manifest_dir, f"{REV_HF_PREFIX}/manifests", ["*.json", "blocks/*.done.json"]
    )
    # This run's own V bank (fresh capture on this pod) — persist-by-default;
    # the parent's bank lives at analysis_tensors/vc_bank, this one beside it.
    uploaded["rev_vc_bank"] = RUN._upload_dir(
        cfg,
        cfg.bank_dir,
        f"{RUN.HF_PREFIX}/analysis_tensors/rev_direction_vc_bank",
        ["*.pt", "*.json"],
    )
    payload = _sentinel_payload(cfg, uploaded)
    RUN._write_json_atomic(cfg.manifest_dir / "rev_upload_done.json", payload)
    sentinel = cfg.log_dir / SENTINEL_NAME
    RUN._write_json_atomic(
        sentinel,
        {"sentinel_schema_version": 1, "kind": "epm:results", "version": 1, "note": payload},
    )
    logger.info("[upload] sentinel written: %s", sentinel)
    logger.info("[phase=upload_done]")
    return RUN.RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 reversed-direction driver (bank / grid / upload)."
    )
    ap.add_argument(
        "--phase",
        choices=("bank", "grid", "upload"),
        help="pipeline phase to run (required unless --import-check)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import (incl. function-body imports) and exit 0",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=RUN.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=RUN.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    ap.add_argument("--gen-batch", type=int, default=16, help="cells per hooked generate call")
    ap.add_argument("--capture-batch", type=int, default=8, help="bank-phase capture batch")
    ap.add_argument("--max-new-tokens", type=int, default=RUN.MAX_NEW_TOKENS)
    ap.add_argument(
        "--anchor-draws",
        type=int,
        default=RUN.ANCHOR_DRAWS,
        help="RunConfig parity only (this driver has NO anchors phase)",
    )
    ap.add_argument("--seed-base", type=int, default=RUN.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class block slice")
    ap.add_argument(
        "--pilot", action="store_true", help="grid: ONE timed cell through this entrypoint"
    )
    ap.add_argument(
        "--force", action="store_true", help="bank: deliberately re-run a completed phase"
    )
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument(
        "--upload-every",
        type=int,
        default=25,
        help="grid: bulk-upload the staged text every N completed blocks",
    )
    ap.add_argument("--planned-wall-h", type=float, default=0.5)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=4.0)
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=None,
        help="dir holding fp-{bare,persona}.anchors.scores.jsonl (e.g. the "
        "/tmp/i2094_shards mirror); default: fetch the two files from the HF data repo",
    )
    ap.add_argument(
        "--coherence-path",
        type=Path,
        default=DEFAULT_COHERENCE_PATH,
        help="the parent's anchor_draws.jsonl (context_id, draw, coherent)",
    )
    return ap.parse_args(argv)


def _import_check() -> None:
    """Resolve every deferred import on this driver's real paths and exit 0.

    Delegates to the parent's check (transformers / hub loads inside
    ``load_model_and_tokenizer`` / ``_repro`` / ``_upload_dir``), then adds
    THIS driver's own function-body imports (the scores HF fetch)."""
    RUN._import_check()
    from huggingface_hub import hf_hub_download  # noqa: F401

    from explore_persona_space.orchestrate.hub import retry_transient  # noqa: F401

    print("[rev-import-check] OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RUN.RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = RUN.build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_id is not None:
        logger.info(
            "[env] --gpu-id=%s CUDA_VISIBLE_DEVICES=%s",
            args.gpu_id,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    if cfg.phase == "bank":
        # Verbatim parent bank phase: capture + injection gate + resume.
        return RUN.phase_bank(cfg)
    if cfg.phase == "grid":
        return phase_grid(cfg, args)
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
