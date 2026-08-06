"""Issue #2094 — fu1_regen_confirm pod driver (cheap-band follow-up round 1).

Two bundled GPU sub-items on a 1x H100 suffix pod (``pod-2094-fu1``), reusing
the unit-C run module (``issue2094_run``) + the stage-2 conventions:

A. **regen** — the plan's own pre-registered cap-hit re-generation trigger
   (cap-hit > 2% per pooled cell), never executed in the parent run: BOTH arms
   (steered + shuffled-donor null) of the 16 breached pooled
   (slot, layer-variant, dose) cells re-generated at ``max_new_tokens=2048``
   (2x the parent's 1024). Same bank, same pairs, same donor derangement, same
   greedy 1-draw ``run_block`` protocol — ONLY the token cap changes. The 16
   cells are DERIVED in-code from the committed fragility artifact and
   asserted against the pre-registered expected shape (fail-loud).
B. **conf1** — CONF-1 survivor confirmation: the 15 clean surviving families
   (bootstrap CIs disjoint-above on a behavior metric with >=5 well-separated
   pairs, minus the 16 breached cells) re-measured at temperature 1.0,
   K=5 draws per pair, BOTH arms — the stage-2 protocol
   (``issue2094_stage2.py``), extended with the grid's donor-null arm.

Persistence (fail-loud, one ``upload_folder`` commit per prefix):
  rollout text -> ``issue2094_singlepos/raw_completions/fu1_regen_confirm/{regen,conf1}``
  V_a captures -> ``issue2094_singlepos/analysis_tensors/fu1_regen_confirm/{regen_va,conf1_va}``
  manifests    -> ``issue2094_singlepos/analysis_tensors/fu1_regen_confirm/manifests``
Pod sentinel: ``/workspace/logs/issue-2094-fu1-results.json`` (schema v1).

VM-side judging needs NO judge changes: regen shards are grid-row-shaped
(``--phase waves --rollouts-dir <regen dir>``); conf1 rows carry
pair_id/setting/text/draw + a per-(cell, arm) ``cell`` key
(``--phase stage2 --stage2-dir <conf1 dir>``).

Launch (pod-side; no VM thread-cap prefix — dedicated GPU box):
``CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2094_fu1.py --run``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_run as R  # noqa: E402

logger = logging.getLogger("issue2094_fu1")

FU1_LABEL = "fu1_regen_confirm"
SENTINEL_NAME = "issue-2094-fu1-results.json"
HF_FU1_TEXT = f"{R.HF_PREFIX}/raw_completions/{FU1_LABEL}"
HF_FU1_TENSORS = f"{R.HF_PREFIX}/analysis_tensors/{FU1_LABEL}"

# The plan's pre-registered re-gen trigger (cap-hit > 2% per pooled cell).
CAPHIT_TRIGGER_FRAC = 0.02
REGEN_MAX_NEW_TOKENS = 2048  # 2x the parent grid's 1024
CONF1_TEMPERATURE = 1.0
CONF1_DRAWS = 5
CONF1_MIN_WELLSEP_PAIRS = 5
BEH_METRICS = ("f_beh_prefix", "f_beh_query")

# Committed derivation inputs (repo-relative; present on the pod's own-issue
# sparse cones — bootstrap opens eval_results/issue_2094 automatically).
FRAGILITY_REL = "eval_results/issue_2094/fragility/fragility_cells.json"
WELLSEP_REL = "eval_results/issue_2094/f_metrics/bootstrap_cis_wellsep.json"

# Pre-registered expected shape of the breached set (brief; verified against
# the committed artifact at implementation time). A derivation drift is a
# fail-loud halt, never a silent re-scope.
EXPECTED_BREACHED_DOSES: dict[tuple[str, str], frozenset[str]] = {
    ("qspan", "joint_mid"): frozenset({"a0.5", "a1", "a2", "a4", "replace"}),
    ("l3j", "joint_mid"): frozenset({"a0.5", "a1", "a2", "a4", "replace"}),
    ("ce", "joint_all"): frozenset({"a0.5", "a1", "a2", "a4"}),
    ("ce", "joint_mid"): frozenset({"a1", "a4"}),
}
EXPECTED_N_BREACHED = 16
EXPECTED_N_CONF1 = 15
# Regen block-set reconciliation (any n_layers: joint variants always exist).
EXPECTED_REGEN_TOTALS = {
    "n_families": 22,
    "n_blocks": 44,
    "cells_steered": 1050,
    "cells_null": 1050,
    "cells_total": 2100,
}

# Smoke slices (subsets of the DERIVED production sets — smoke IS the sweep
# with fewer cells/pairs/draws; arm classes covered: multi-position replace
# (add_full_state_patch) + multi-position additive + joint_all A + Type B
# centroid-donor null + joint_mid A; conf1: cross + matched_query, A + B,
# single-layer variants, both arms).
SMOKE_REGEN_FAMILIES: tuple[tuple[str, str, str, str], ...] = (
    ("qspan", "joint_mid", "replace", "A"),
    ("l3j", "joint_mid", "a0.5", "A"),
    ("ce", "joint_all", "a1", "A"),
    ("ce", "joint_all", "a1", "B"),
    ("ce", "joint_mid", "a4", "A"),
)
SMOKE_CONF1_KEYS: tuple[str, ...] = (
    "cross|ce|L15|a1|A",
    "matched_query|ce|L20|a2|B",
)
SMOKE_CONF1_DRAWS = 2

RC_OK = 0


# ── pure derivations (CPU-only; pinned in tests/test_issue2094_fu1.py) ──


def derive_breached_cells(fragility: dict) -> list[tuple[str, str, str]]:
    """The 16 pooled (slot, layer_variant, dose) cells with steered cap-hit
    fraction above the plan's 2% re-gen trigger — asserted against the
    pre-registered expected shape (fail-loud on any drift)."""
    cells = sorted(
        (c["slot"], c["layer_variant"], c["dose"])
        for c in fragility["cells"]
        if c["steered"]["cap_hit_frac"] > CAPHIT_TRIGGER_FRAC
    )
    assert len(cells) == len(set(cells)), "duplicate pooled cells in fragility artifact"
    realized: dict[tuple[str, str], set[str]] = {}
    for slot, lv, dose in cells:
        realized.setdefault((slot, lv), set()).add(dose)
    expected = {k: set(v) for k, v in EXPECTED_BREACHED_DOSES.items()}
    assert realized == expected, (
        f"derived breached set does not match the pre-registered shape:\n"
        f"  derived : {realized}\n  expected: {expected}"
    )
    assert len(cells) == EXPECTED_N_BREACHED, (len(cells), cells)
    return cells


def derive_conf1_families(wellsep: dict, breached: set[tuple[str, str, str]]) -> list[dict]:
    """The 15 clean surviving families: steered-vs-null CIs disjoint with the
    steered side ABOVE, >= 5 well-separated pairs, behavior metrics only,
    minus the 16 cap-hit-breached pooled cells. Asserted: exactly 15, all
    context-end (slot == 'ce')."""
    fams: list[dict] = []
    for key in sorted(wellsep["steered_vs_null"]):
        rec = wellsep["steered_vs_null"][key]
        parts = key.split("|")
        assert len(parts) == 6, f"unexpected family key shape: {key!r}"
        setting, slot, lv, dose, vec_type, metric = parts
        if metric not in BEH_METRICS:
            continue
        if not rec.get("cis_disjoint"):
            continue
        if rec.get("direction") != "steered_above":
            continue
        if int(rec.get("n_pairs_used", 0)) < CONF1_MIN_WELLSEP_PAIRS:
            continue
        if (slot, lv, dose) in breached:
            continue
        fams.append(
            {
                "family": key,
                "setting": setting,
                "slot": slot,
                "layer_variant": lv,
                "dose": dose,
                "vec_type": vec_type,
                "metric": metric,
            }
        )
    assert len(fams) == EXPECTED_N_CONF1, (
        f"derived {len(fams)} surviving families, expected {EXPECTED_N_CONF1}: "
        f"{[f['family'] for f in fams]}"
    )
    assert all(f["slot"] == "ce" for f in fams), "expected all survivors context-end"
    return fams


def conf1_cells_from_families(families: list[dict]) -> list[dict]:
    """Distinct measurement cells (setting, slot, layer_variant, dose,
    vec_type) across the surviving families; each carries its family keys."""
    by_cell: dict[tuple[str, ...], dict] = {}
    for f in families:
        key = (f["setting"], f["slot"], f["layer_variant"], f["dose"], f["vec_type"])
        cell = by_cell.setdefault(
            key,
            {
                "setting": f["setting"],
                "slot": f["slot"],
                "layer_variant": f["layer_variant"],
                "dose": f["dose"],
                "vec_type": f["vec_type"],
                "families": [],
            },
        )
        cell["families"].append(f["family"])
    cells = [by_cell[k] for k in sorted(by_cell)]
    for cell in cells:
        if cell["vec_type"] == "B":
            assert cell["setting"] == "matched_query", cell
    assert len(cells) == EXPECTED_N_CONF1, (len(cells), [c["families"] for c in cells])
    return cells


def regen_block_families(
    pairs: list[BANK.Pair], n_layers: int, breached: set[tuple[str, str, str]]
) -> list[tuple[R.Block, R.Block]]:
    """The parent-grid block families whose pooled (slot, lv, dose) cell is
    breached — includes BOTH vec types where both exist (ce cells pool A+B),
    exactly the parent enumeration restricted to the 16 cells."""
    fams = [
        fam
        for fam in R.enumerate_block_families(pairs, n_layers)
        if (fam[0].slot, fam[0].layer_variant, fam[0].dose) in breached
    ]
    totals = R.grid_totals(fams)
    assert totals == EXPECTED_REGEN_TOTALS, (
        f"regen block-set reconciliation failed:\n"
        f"  derived : {totals}\n  expected: {EXPECTED_REGEN_TOTALS}"
    )
    return fams


def smoke_pair_subset(pairs: list[BANK.Pair]) -> tuple[str, ...]:
    """One pair per setting + a conv-``context_a`` pair (the multi-turn render
    seam stays smoke-visible — mirrors ``R.smoke_block_families``)."""
    ids: list[str] = []
    for setting in ("matched_prefix", "matched_query", "cross"):
        first = next((p.pair_id for p in pairs if p.setting == setting), None)
        if first is not None:
            ids.append(first)
    conv = next((p.pair_id for p in pairs if p.a.startswith("conv")), None)
    if conv is not None and conv not in ids:
        ids.append(conv)
    assert ids, "empty smoke pair subset"
    return tuple(ids)


def slice_regen_smoke(
    fams: list[tuple[R.Block, R.Block]], pairs: list[BANK.Pair]
) -> list[tuple[R.Block, R.Block]]:
    """Smoke slice: the SMOKE_REGEN_FAMILIES subset of the derived families,
    each block's pair set restricted (A -> the per-setting+conv subset,
    B -> first matched-query pair)."""
    a_subset = smoke_pair_subset(pairs)
    mq_subset = tuple(p.pair_id for p in pairs if p.setting == "matched_query")[:1]
    keep = {spec: True for spec in SMOKE_REGEN_FAMILIES}
    out: list[tuple[R.Block, R.Block]] = []
    for steered, null in fams:
        spec = (steered.slot, steered.layer_variant, steered.dose, steered.vec_type)
        if spec not in keep:
            continue
        subset = mq_subset if steered.vec_type == "B" else a_subset
        ids = tuple(pid for pid in subset if pid in steered.pair_ids)
        assert ids, (spec, subset)
        out.append((replace(steered, pair_ids=ids), replace(null, pair_ids=ids)))
    assert len(out) == len(SMOKE_REGEN_FAMILIES), (
        f"smoke regen slice found {len(out)} of {len(SMOKE_REGEN_FAMILIES)} families"
    )
    return out


def slice_conf1_smoke(cells: list[dict]) -> list[dict]:
    """Smoke slice: the SMOKE_CONF1_KEYS subset of the derived conf1 cells."""
    wanted = set(SMOKE_CONF1_KEYS)

    def cell_id(c: dict) -> str:
        return "|".join([c["setting"], c["slot"], c["layer_variant"], c["dose"], c["vec_type"]])

    out = [c for c in cells if cell_id(c) in wanted]
    assert len(out) == len(SMOKE_CONF1_KEYS), (
        f"smoke conf1 slice found {[cell_id(c) for c in out]}, wanted {sorted(wanted)}"
    )
    return out


# ── conf1 cell runner (stage-2 protocol + the grid's donor-null arm) ────


def conf1_cell_key(cell: dict, arm: str) -> str:
    return "|".join(
        [
            "fu1",
            cell["setting"],
            cell["slot"],
            cell["layer_variant"],
            cell["dose"],
            cell["vec_type"],
            arm,
        ]
    )


def conf1_regime(cfg: R.RunConfig, cell: dict, arm: str, draws: int, bank_sha: str) -> str:
    """Resume key over EVERY output-affecting knob (the #722 r3 convention)."""
    key = json.dumps(
        {
            "cell": conf1_cell_key(cell, arm),
            "draws": draws,
            "temperature": CONF1_TEMPERATURE,
            "seed_base": cfg.seed_base,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "max_new_tokens": cfg.max_new_tokens,
            "bank_sha": bank_sha,
            "smoke": cfg.smoke,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@torch.no_grad()
def run_conf1_cell(
    cfg: R.RunConfig,
    paths: FU1Paths,
    model,
    tok,
    bank: dict,
    cell: dict,
    arm: str,
    cell_pairs: list[BANK.Pair],
    donor_map: dict[str, str],
    pairs_by_id: dict[str, BANK.Pair],
    eot: list[int],
    draws: int,
) -> dict:
    """One (cell, arm): K temp-1.0 hooked draws per pair + the both-pooling
    V_a capture. Steered installs the pair's own payload; null installs the
    shuffled-donor payload via the grid's ``_resolve_donor`` walk (same seeded
    derangement as the parent grid — it ships inside the staged vc bank)."""
    assert arm in R.ARMS, arm
    contexts = BANK.build_contexts()
    ck = conf1_cell_key(cell, arm)
    slug = R.block_slug(ck)
    layers = R.layer_variant_layers(cell["layer_variant"], cfg.n_layers)
    mode, alpha, payload_kind = R._realized_mode(cell["slot"], cell["dose"], cell["vec_type"])
    realized_mode = (
        "add_full_state_patch"
        if cell["dose"] == "replace" and mode == "add"
        else ("replace" if mode == "replace" else "add")
    )
    recs = bank["per_context"]
    ids_cache = {
        cid: BANK.context_token_ids_2094(tok, contexts[cid]) for cid in {p.a for p in cell_pairs}
    }
    pair_cells: list[dict] = []
    for pair in cell_pairs:
        delta, state, m = R._pair_payload(bank, pair, cell["slot"], cell["vec_type"])
        payload = state if payload_kind == "state" else delta
        donor_label = None
        if arm == "null":
            payload, donor_label = R._resolve_donor(
                bank,
                pair,
                donor_map,
                pairs_by_id,
                cell["slot"],
                cell["vec_type"],
                payload,
                payload_kind,
            )
        rec = recs[pair.a]
        pos = R.slot_positions(rec["ctx_len"], rec["prefix_end"], cell["slot"])[-m:]
        assert payload.shape[0] == len(pos), (payload.shape, pos)
        pair_cells.append(
            {
                "pair_id": pair.pair_id,
                "setting": pair.setting,
                "context_a": pair.a,
                "context_b": pair.b,
                "positions": list(pos),
                "payload": payload,
                "donor_pair_id": donor_label,
            }
        )

    texts_by_pair: list[list[str]] = []
    for start in range(0, len(pair_cells), cfg.gen_batch):
        chunk = pair_cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        row_ids = [ids_cache[c["context_a"]] for c in chunk]
        row_lengths = [len(r) for r in row_ids]
        hook = R._arm_hook_for_rows(
            model,
            cfg,
            layers,
            row_lengths,
            [tuple(c["positions"]) for c in chunk],
            [c["payload"] for c in chunk],
            mode,
            alpha,
            max(row_lengths),
        )
        try:
            outs = R.generate_batch(
                model,
                tok,
                ctx_list,
                n=draws,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=CONF1_TEMPERATURE,
                seed_base=cfg.seed_base,
                # History-aware render (bank.py module note) — the same seam
                # as the grid + stage-2 paths; generate_batch re-arms the
                # hook per draw.
                render_fn=BANK.render_context_2094,
                ids_fn=BANK.context_token_ids_2094,
            )
        finally:
            hook.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_by_pair.extend([list(o) for o in outs])
    assert len(texts_by_pair) == len(pair_cells)

    flat_ctx = [ids_cache[pc["context_a"]] for pc in pair_cells for _ in range(draws)]
    flat_text = [t for texts in texts_by_pair for t in texts]
    states = R.capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    rows: list[dict] = []
    flat = 0
    for b, pc in enumerate(pair_cells):
        for draw in range(len(texts_by_pair[b])):
            n_tok = states["n_completion_tokens"][flat]
            rows.append(
                {
                    # Stage-2 row contract (pair_id/setting/text/draw + cell)
                    # so the existing judge --phase stage2 path consumes these
                    # rows unchanged; the arm rides the cell key AND its own
                    # field, keeping judge item ids unique across arms.
                    "cell": ck,
                    "block_key": ck,
                    "slot": cell["slot"],
                    "layer_variant": cell["layer_variant"],
                    "layers": list(layers),
                    "dose": cell["dose"],
                    "alpha": alpha,
                    "hook_mode": mode,
                    "realized_mode": realized_mode,
                    "vec_type": cell["vec_type"],
                    "arm": arm,
                    "pair_id": pc["pair_id"],
                    "setting": pc["setting"],
                    "context_a": pc["context_a"],
                    "context_b": pc["context_b"],
                    "positions": pc["positions"],
                    "donor_pair_id": pc["donor_pair_id"],
                    "temperature": CONF1_TEMPERATURE,
                    "seed": cfg.seed_base + draw,
                    "draw": draw,
                    "n_completion_tokens": n_tok,
                    "cap_hit": R.cap_hit(n_tok, cfg.max_new_tokens),
                    "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                    "post_selection": True,
                    "families": cell["families"],
                    "text": texts_by_pair[b][draw],
                }
            )
            flat += 1
    R._write_jsonl_atomic(paths.conf1_rollouts / f"fu1_{slug}.jsonl", rows)
    R._save_pt_atomic(
        paths.conf1_va / f"fu1_{slug}.pt",
        {
            "cell": ck,
            "layers": cfg.layers,
            "index": [{"pair_id": r["pair_id"], "draw": r["draw"]} for r in rows],
            "va_span": states["va_span"],
            "va_tail": states["va_tail"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": R._repro(cfg),
        },
    )
    done = {
        "cell": ck,
        "regime_fp": conf1_regime(cfg, cell, arm, draws, str(bank.get("bank_sha"))),
        "n_rows": len(rows),
        "n_cap_hit": sum(1 for r in rows if r["cap_hit"]),
        "n_empty": len(states["empty_rows"]),
        "repro": R._repro(cfg),
    }
    R._write_json_atomic(paths.conf1_manifests / f"fu1_{slug}.done.json", done)
    return done


# ── paths / staging ─────────────────────────────────────────────────────


@dataclass
class FU1Paths:
    """conf1-side output tree (the regen side rides RunConfig's own
    rollouts_dir / va_dir / manifest_dir so ``R.run_block`` is reused
    verbatim)."""

    out_root: Path

    @property
    def conf1_rollouts(self) -> Path:
        return self.out_root / "conf1" / "rollouts"

    @property
    def conf1_va(self) -> Path:
        return self.out_root / "conf1" / "va"

    @property
    def conf1_manifests(self) -> Path:
        return self.out_root / "manifests" / "conf1"

    @property
    def inputs_dir(self) -> Path:
        return self.out_root / "inputs"


def stage_bank(cfg: R.RunConfig, hf_revision: str | None) -> None:
    """Stage the parent run's vc_bank.pt from HF into ``cfg.bank_dir`` (where
    ``R._load_bank`` reads it); idempotent."""
    from explore_persona_space.orchestrate import hub

    target = cfg.bank_dir / "vc_bank.pt"
    if target.exists():
        logger.info("[stage] vc_bank.pt already present — skipping")
        return
    hub.stage_hub_file(
        R.HF_DATA_REPO,
        f"{R.HF_PREFIX}/analysis_tensors/vc_bank/vc_bank.pt",
        target,
        repo_type="dataset",
        revision=hf_revision,
    )
    logger.info("[stage] vc_bank.pt staged")


def load_or_capture_bank(cfg: R.RunConfig, model, tok, hf_revision: str | None) -> dict:
    """Production: the parent's staged bank (same donor derangement). Tiny
    smoke: a locally-captured bank + freshly-seeded derangement (the bank
    manifest sha keeps the regime keys consistent)."""
    target = cfg.bank_dir / "vc_bank.pt"
    if not target.exists() and cfg.smoke and cfg.tiny:
        logger.info("[fu1] tiny smoke: capturing a LOCAL vc bank")
        bank = R.capture_bank(cfg, model, tok)
        _, bank_sha = R.bank_manifest_and_sha()
        bank["bank_sha"] = bank_sha
        bank["donor_derangement"] = BANK.donor_derangement(BANK.build_pairs())
        return bank
    if not target.exists():
        stage_bank(cfg, hf_revision)
    # Self-produced, sha-recorded bundle carrying non-tensor metadata.
    return torch.load(target, map_location="cpu", weights_only=False)


# ── cap-hit report (the sub-item A deliverable) ────────────────────────


def regen_caphit_report(cfg: R.RunConfig, fragility: dict) -> dict:
    """Realized cap-hit fractions at 2048 per pooled (slot, lv, dose) cell,
    per arm, pooled over vec types (the fragility artifact's grain), next to
    the parent run's 1024-cap fractions."""
    old: dict[tuple[str, str, str], dict] = {
        (c["slot"], c["layer_variant"], c["dose"]): c for c in fragility["cells"]
    }
    pooled: dict[tuple[str, str, str], dict[str, dict[str, int]]] = {}
    for done in sorted((cfg.manifest_dir / "blocks").glob("*.done.json")):
        rec = json.loads(done.read_text())
        slot, lv, dose, _vt, arm = rec["key"].split("|")
        arms = pooled.setdefault((slot, lv, dose), {})
        agg = arms.setdefault(arm, {"n": 0, "cap_hit": 0})
        agg["n"] += int(rec["n_cells"])
        agg["cap_hit"] += int(rec["n_cap_hit"])
    cells = []
    for key in sorted(pooled):
        slot, lv, dose = key
        row: dict = {"slot": slot, "layer_variant": lv, "dose": dose}
        for arm, agg in sorted(pooled[key].items()):
            frac = (agg["cap_hit"] / agg["n"]) if agg["n"] else 0.0
            row[arm] = {**agg, "cap_hit_frac": frac}
        old_cell = old.get(key)
        if old_cell is not None:
            row["cap_hit_frac_1024"] = {
                "steered": old_cell["steered"]["cap_hit_frac"],
                "null": old_cell["null"]["cap_hit_frac"],
            }
        cells.append(row)
    return {
        "max_new_tokens": cfg.max_new_tokens,
        "trigger_frac": CAPHIT_TRIGGER_FRAC,
        "n_cells": len(cells),
        "cells": cells,
        "repro": R._repro(cfg),
    }


# ── upload + sentinel ──────────────────────────────────────────────────


def upload_and_sentinel(
    cfg_regen: R.RunConfig,
    cfg_conf1: R.RunConfig,
    paths: FU1Paths,
    derivation: dict,
    caphit: dict,
) -> None:
    """Fail-loud bulk uploads (one commit per prefix, exact-set verified via
    ``R._upload_dir``) + the pod sentinel the VM poller drains."""
    uploaded: dict[str, list[str]] = {}
    uploaded["regen_text"] = R._upload_dir(
        cfg_regen, cfg_regen.rollouts_dir, f"{HF_FU1_TEXT}/regen", ["shard_*.jsonl"]
    )
    uploaded["regen_va"] = R._upload_dir(
        cfg_regen, cfg_regen.va_dir, f"{HF_FU1_TENSORS}/regen_va", ["shard_*.pt"]
    )
    uploaded["conf1_text"] = R._upload_dir(
        cfg_conf1, paths.conf1_rollouts, f"{HF_FU1_TEXT}/conf1", ["fu1_*.jsonl"]
    )
    uploaded["conf1_va"] = R._upload_dir(
        cfg_conf1, paths.conf1_va, f"{HF_FU1_TENSORS}/conf1_va", ["fu1_*.pt"]
    )
    uploaded["manifests"] = R._upload_dir(
        cfg_regen,
        cfg_regen.manifest_dir,
        f"{HF_FU1_TENSORS}/manifests",
        ["*.json", "blocks/*.done.json", "conf1/*.done.json"],
    )

    n_conf1_rows = 0
    conf1_cap = 0
    for done in sorted(paths.conf1_manifests.glob("fu1_*.done.json")):
        rec = json.loads(done.read_text())
        n_conf1_rows += int(rec.get("n_rows", 0))
        conf1_cap += int(rec.get("n_cap_hit", 0))
    payload = {
        "eval_numbers": {
            "followup_label": FU1_LABEL,
            "regen_blocks": len(list(cfg_regen.rollouts_dir.glob("shard_*.jsonl"))),
            "regen_cells": sum(
                arm_agg["n"]
                for row in caphit["cells"]
                for arm_agg in (row.get("steered", {"n": 0}), row.get("null", {"n": 0}))
            ),
            "regen_caphit_2048": caphit["cells"],
            "conf1_cell_arms": len(list(paths.conf1_rollouts.glob("fu1_*.jsonl"))),
            "conf1_rows": n_conf1_rows,
            "conf1_cap_hit_rows": conf1_cap,
            "conf1_cap_hit_frac": (conf1_cap / n_conf1_rows) if n_conf1_rows else 0.0,
            "derivation": derivation,
        },
        "eval_paths": sorted(
            str(p)
            for p in (
                cfg_regen.rollouts_dir,
                cfg_regen.va_dir,
                paths.conf1_rollouts,
                paths.conf1_va,
            )
        ),
        "reproducibility_card": {
            **R._repro(cfg_regen),
            "seed_base": cfg_regen.seed_base,
            "bank_seed": BANK.SEED,
            "regen_max_new_tokens": cfg_regen.max_new_tokens,
            "regen_temperature": R.GRID_TEMPERATURE,
            "conf1_max_new_tokens": cfg_conf1.max_new_tokens,
            "conf1_temperature": CONF1_TEMPERATURE,
            "conf1_draws": cfg_conf1.anchor_draws,
            "gen_batch": cfg_regen.gen_batch,
            "post_selection": True,
        },
        "wandb_url": None,
        "hf_hub_url": (f"https://huggingface.co/datasets/{R.HF_DATA_REPO}/tree/main/{R.HF_PREFIX}"),
        "worktree_path": str(R.REPO_ROOT),
        "final_commit_sha": R._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg_regen.gpu_hours_budgeted,
        "plan_deviations": [
            "fu1 cheap-band round: the 16-cell cap-hit re-gen executes the parent "
            "plan's own pre-registered >2% trigger at max_new_tokens=2048; the 15 "
            "CONF-1 survivor cells are re-measured at temp 1.0 / K=5 / both arms "
            "(post-selection confirmation, never an unbiased estimate)",
            "conf1 rows are LABELED post_selection=true and carry the arm in both "
            "the row and the cell key (stage-2 row contract, judge-compatible)",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }
    R._write_json_atomic(cfg_regen.manifest_dir / "fu1_upload_done.json", payload)
    sentinel = cfg_regen.log_dir / SENTINEL_NAME
    kind = "epm:smoke-result" if cfg_regen.smoke else "epm:results"
    R._write_json_atomic(
        sentinel,
        {"sentinel_schema_version": 1, "kind": kind, "version": 1, "note": payload},
    )
    logger.info("[upload] sentinel written: %s (kind=%s)", sentinel, kind)


# ── entrypoint ─────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths
    (the #1689 false-pass class): transformers loads inside
    ``R.load_model_and_tokenizer``, hub helpers inside ``stage_bank`` /
    ``R._upload_dir``."""
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        stage_hub_file,
        verify_repo_paths_uploaded,
    )

    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 fu1_regen_confirm pod driver (cap-hit re-gen + CONF-1)."
    )
    ap.add_argument(
        "--run", action="store_true", help="execute derive->stage->regen->conf1->upload"
    )
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/issue2094_fu1_out"))
    ap.add_argument("--log-dir", type=Path, default=R.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=R.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument(
        "--tiny-layers",
        type=int,
        default=28,
        help="tiny model layer count (default 28 so the derived single-layer "
        "conf1 variants L12..L20 and the production joint_mid band exist)",
    )
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--regen-max-new-tokens", type=int, default=REGEN_MAX_NEW_TOKENS)
    ap.add_argument(
        "--conf1-max-new-tokens",
        type=int,
        default=R.MAX_NEW_TOKENS,
        help="stage-2 protocol fidelity: the parent cap (1024); the 15 "
        "surviving cells are cap-clean there by construction",
    )
    ap.add_argument("--conf1-draws", type=int, default=CONF1_DRAWS)
    ap.add_argument("--seed-base", type=int, default=R.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny cell/pair/draw slice")
    ap.add_argument("--fragility", type=Path, default=None, help="local fragility_cells.json")
    ap.add_argument("--wellsep", type=Path, default=None, help="local bootstrap_cis_wellsep.json")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument(
        "--upload-every",
        type=int,
        default=10,
        help="regen: bulk-upload completed shards every N blocks (256 commits/hr cap)",
    )
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[R.RunConfig, R.RunConfig]:
    """(regen cfg, conf1 cfg) — unit-C RunConfigs sharing one out-root; the
    two phases differ ONLY in max_new_tokens (+ the conf1 draw count riding
    ``anchor_draws``), so model load + capture + upload seams are REUSED."""
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    regen = R.RunConfig(
        phase="fu1_regen",
        out_root=args.out_root,
        log_dir=args.log_dir,
        model_id=args.model_id,
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else R.N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else R.HIDDEN_FULL,
        device=device,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.regen_max_new_tokens,
        anchor_draws=1,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=False,
        force=False,
        worker_index=0,
        num_workers=1,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=2.0,  # cheap-band dispatch note: ~2 GPU-h on 1x H100
        gpu_hours_budgeted=2.0,
    )
    conf1 = replace(
        regen,
        phase="fu1_conf1",
        max_new_tokens=args.conf1_max_new_tokens,
        anchor_draws=args.conf1_draws,
    )
    return regen, conf1


def _resolve_artifact(arg: Path | None, rel: str) -> Path:
    path = arg if arg is not None else (R.REPO_ROOT / rel)
    assert path.is_file(), f"derivation input missing: {path}"
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    assert args.run, "pass --run (or --import-check)"
    cfg_regen, cfg_conf1 = build_configs(args)
    paths = FU1Paths(out_root=cfg_regen.out_root)
    for d in (
        cfg_regen.rollouts_dir,
        cfg_regen.va_dir,
        cfg_regen.manifest_dir / "blocks",
        paths.conf1_rollouts,
        paths.conf1_va,
        paths.conf1_manifests,
        paths.inputs_dir,
        cfg_regen.log_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # ── phase: derive (CPU-only, from the committed artifacts) ──────────
    logger.info("[phase=fu1_derive] smoke=%s tiny=%s", cfg_regen.smoke, cfg_regen.tiny)
    fragility = json.loads(_resolve_artifact(args.fragility, FRAGILITY_REL).read_text())
    wellsep = json.loads(_resolve_artifact(args.wellsep, WELLSEP_REL).read_text())
    breached = derive_breached_cells(fragility)
    families = derive_conf1_families(wellsep, set(breached))
    cells = conf1_cells_from_families(families)
    derivation = {
        "caphit_trigger_frac": CAPHIT_TRIGGER_FRAC,
        "n_breached_cells": len(breached),
        "breached_cells": [list(c) for c in breached],
        "n_conf1_families": len(families),
        "conf1_families": [f["family"] for f in families],
        "n_conf1_cells": len(cells),
    }
    R._write_json_atomic(cfg_regen.manifest_dir / "fu1_derivation.json", derivation)
    logger.info(
        "[derive] %d breached pooled cells; %d surviving families -> %d conf1 cells",
        len(breached),
        len(families),
        len(cells),
    )

    pairs = BANK.build_pairs()
    regen_fams = regen_block_families(pairs, cfg_regen.n_layers, set(breached))
    if cfg_regen.smoke:
        regen_fams = slice_regen_smoke(regen_fams, pairs)
        cells = slice_conf1_smoke(cells)
    conf1_draws = SMOKE_CONF1_DRAWS if cfg_conf1.smoke else cfg_conf1.anchor_draws

    # ── phase: stage + model load ───────────────────────────────────────
    logger.info("[phase=fu1_stage]")
    model, tok = R.load_model_and_tokenizer(cfg_regen)
    eot = R.eot_tail_ids(tok)
    bank = load_or_capture_bank(cfg_regen, model, tok, args.hf_revision)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = bank.get("donor_derangement") or BANK.donor_derangement(pairs)
    bank_sha = str(bank.get("bank_sha"))
    regen_fp = R.regime_fingerprint(cfg_regen, bank_sha)

    # ── phase: regen (sub-item A — run_block verbatim at 2048) ──────────
    blocks = R.blocks_for_worker(regen_fams, 0, 1)
    logger.info(
        "[phase=fu1_regen] %d blocks / %d cells at max_new_tokens=%d",
        len(blocks),
        sum(b.n_cells for b in blocks),
        cfg_regen.max_new_tokens,
    )
    t0 = time.monotonic()
    pending: list[R.Block] = []
    for k, block in enumerate(blocks, start=1):
        if R.block_is_done(cfg_regen.out_root, block, regen_fp):
            logger.info("[regen] block %d/%d %s SKIP (done)", k, len(blocks), block.key)
            continue
        rec = R.run_block(cfg_regen, model, tok, bank, block, pairs_by_id, donor_map, eot, regen_fp)
        pending.append(block)
        logger.info(
            "[regen] block %d/%d %s rows=%d cap_hit=%d elapsed=%.1fs",
            k,
            len(blocks),
            block.key,
            rec["n_cells"],
            rec["n_cap_hit"],
            time.monotonic() - t0,
        )
        if cfg_regen.upload_every > 0 and len(pending) >= cfg_regen.upload_every:
            R._upload_dir(
                cfg_regen,
                cfg_regen.rollouts_dir,
                f"{HF_FU1_TEXT}/regen",
                [f"shard_{b.slug}.jsonl" for b in pending],
            )
            pending = []

    caphit = regen_caphit_report(cfg_regen, fragility)
    R._write_json_atomic(cfg_regen.manifest_dir / "fu1_regen_caphit.json", caphit)
    over = [
        (c["slot"], c["layer_variant"], c["dose"], c["steered"]["cap_hit_frac"])
        for c in caphit["cells"]
        if c.get("steered", {}).get("cap_hit_frac", 0.0) > CAPHIT_TRIGGER_FRAC
    ]
    logger.info(
        "[regen] cap-hit report at %d: %d/%d pooled cells still over the %.0f%% trigger: %s",
        cfg_regen.max_new_tokens,
        len(over),
        len(caphit["cells"]),
        100 * CAPHIT_TRIGGER_FRAC,
        over,
    )

    # ── phase: conf1 (sub-item B — stage-2 protocol, both arms) ─────────
    n_units = len(cells) * len(R.ARMS)
    logger.info(
        "[phase=fu1_conf1] %d cells x %d arms (draws=%d, temp=%.1f, max_new_tokens=%d)",
        len(cells),
        len(R.ARMS),
        conf1_draws,
        CONF1_TEMPERATURE,
        cfg_conf1.max_new_tokens,
    )
    t1 = time.monotonic()
    unit = 0
    for cell in cells:
        cell_pairs = [p for p in pairs if p.setting == cell["setting"]]
        if cfg_conf1.smoke:
            cell_pairs = cell_pairs[:1]
        assert cell_pairs, cell
        for arm in R.ARMS:
            unit += 1
            ck = conf1_cell_key(cell, arm)
            slug = R.block_slug(ck)
            done_path = paths.conf1_manifests / f"fu1_{slug}.done.json"
            regime = conf1_regime(cfg_conf1, cell, arm, conf1_draws, bank_sha)
            if done_path.exists():
                rec = json.loads(done_path.read_text())
                if rec.get("regime_fp") == regime:
                    logger.info("[conf1] unit %d/%d %s SKIP (done)", unit, n_units, ck)
                    continue
                raise RuntimeError(
                    f"conf1 cell {ck} done-file carries regime_fp={rec.get('regime_fp')!r} "
                    f"but this run's regime_fp={regime!r} — refusing to resume across "
                    "regimes (quarantine or use a fresh --out-root)"
                )
            done = run_conf1_cell(
                cfg_conf1,
                paths,
                model,
                tok,
                bank,
                cell,
                arm,
                cell_pairs,
                donor_map,
                pairs_by_id,
                eot,
                conf1_draws,
            )
            logger.info(
                "[conf1] unit %d/%d %s rows=%d cap_hit=%d elapsed=%.1fs",
                unit,
                n_units,
                ck,
                done["n_rows"],
                done["n_cap_hit"],
                time.monotonic() - t1,
            )

    # ── phase: upload + sentinel ────────────────────────────────────────
    logger.info("[phase=fu1_upload]")
    upload_and_sentinel(cfg_regen, cfg_conf1, paths, derivation, caphit)
    logger.info("[phase=fu1_done]")
    print("[phase=done]", flush=True)
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
