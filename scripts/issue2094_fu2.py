"""Issue #2094 — fu2_span_slots pod driver (user-chat same-issue follow-up).

THREE new intervention span slots extending the shipped grid, same instrument
end to end (scope marker ``epm:followup-scope`` fu2_span_slots, 2026-08-06):

- ``qtext``      — final-user-turn query TEXT tokens only: the shipped ``qspan``
                   (``range(prefix_end, ctx_len)``, issue2094_run.py) MINUS the
                   ``<|im_start|>user\\n`` header and the trailing
                   ``<|im_end|>\\n<|im_start|>assistant\\n`` template tokens.
- ``pspan_tmpl`` — the WHOLE prefix span WITH template tokens: positions
                   ``0..prefix_end-1``.
- ``pspan_text`` — prefix CONTENT tokens only: the same span minus every
                   template token across all prefix turns.

Grid: Type-A only, joint_all + joint_mid layer variants only, the existing dose
ladder (a0.5/a1/a2/a4 + replace -> add_full_state_patch on these multi-position
slots), steered + norm-matched shuffled-donor null per cell (the parent seeded
derangement), the same F_beh/F_act V_a capture. Pair eligibility: prefix-span
slots EXCLUDE matched-prefix pairs (identical prefixes => the steered edit is a
no-op by causal identity — the shipped grid's degenerate_self class); qtext
runs on all three settings. Generation at ``max_new_tokens=2048`` from the
start (fu1's raised cap); per-cell cap-hit reported.

ALIGNMENT (registered refinement, documented per the scope marker's
delegation): Type-A deltas right-align to the min-overlap ``m`` WITHIN each
slot's OWN coordinate set — for ``qtext``/``pspan_text`` that is right-alignment
within CONTENT tokens (the last ``m`` content tokens of each side pair up in
order), the natural extension of the grid's right-aligned min-overlap
convention: template tokens are excluded from both the delta and the edit
sites, and the span END (nearest the answer) stays the anchor. ``pspan_tmpl``
right-aligns within the full prefix span (the grid convention as-is; across
prefixes of different lengths the aligned offsets can pair template with
content positions — inherent to the parent's heterogeneous-span convention,
exactly as the shipped qspan slot paired header tokens across unequal spans).
Donor nulls are shaped with the SAME mask machinery on the donor side (the
donor pair's own content-token delta, right-aligned to the recipient's ``m``,
norm-matched position-wise) — the shipped qspan pattern.

Template-token identification is FAIL-LOUD and structural: masks come from
``bank.template_token_mask`` over the tokenized chat-template ids (special-token
ids + role-header walk asserted in-process against the REAL tokenizer), never a
regex over decoded text; boundary special-token ids are asserted at every
position class this round excludes (``fu2_slot_positions``).

Persistence (fail-loud, one ``upload_folder`` commit per prefix):
  rollout text -> ``issue2094_singlepos/raw_completions/fu2_span_slots/rollouts``
  V_a captures -> ``issue2094_singlepos/analysis_tensors/fu2_span_slots/va``
  fu2 V bank   -> ``issue2094_singlepos/analysis_tensors/fu2_span_slots/bank``
  manifests    -> ``issue2094_singlepos/analysis_tensors/fu2_span_slots/manifests``
Pod sentinel: ``/workspace/logs/issue-2094-fu2-results.json`` (schema v1).

VM-side judging needs NO judge changes: shards are grid-row-shaped
(``issue2094_judge.py --phase waves --rollouts-dir <rollouts dir>``).

Launch (pod-side; no VM thread-cap prefix — dedicated GPU box):
``CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2094_fu2.py --run``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_fu1 as FU1  # noqa: E402
import issue2094_run as R  # noqa: E402

logger = logging.getLogger("issue2094_fu2")

FU2_LABEL = "fu2_span_slots"
SENTINEL_NAME = "issue-2094-fu2-results.json"
HF_FU2_TEXT = f"{R.HF_PREFIX}/raw_completions/{FU2_LABEL}"
HF_FU2_TENSORS = f"{R.HF_PREFIX}/analysis_tensors/{FU2_LABEL}"

FU2_MAX_NEW_TOKENS = 2048  # from the start (fu1's raised cap; scope marker)
FU2_SLOTS: tuple[str, ...] = ("qtext", "pspan_tmpl", "pspan_text")
FU2_VARIANTS: tuple[str, ...] = ("joint_all", "joint_mid")
PSPAN_SLOTS: frozenset[str] = frozenset({"pspan_tmpl", "pspan_text"})
# Version token folded into the resume regime: a change to the fu2 slot
# machinery (masks / alignment) is a NEW regime, never a silent resume.
FU2_REGIME_TOKEN = "fu2_span_slots_v1"

# qtext span structure (asserted in fu2_slot_positions): 3 header tokens at the
# front of the final user turn, 5 template tokens at the tail of the render.
QTEXT_HEADER_TOKENS = 3  # <|im_start|> user \n
QTEXT_TAIL_TOKENS = 5  # <|im_end|> \n <|im_start|> assistant \n

# Grid arithmetic (pinned in tests/test_issue2094_fu2.py): qtext runs the full
# 60-pair bank; the pspan slots exclude the 30 matched-prefix pairs.
EXPECTED_FU2_TOTALS = {
    "n_families": 30,
    "n_blocks": 60,
    "cells_steered": 1200,
    "cells_null": 1200,
    "cells_total": 2400,
}

# Capture-parity gate vs the parent's staged bank (production only): the fu2
# bank is a FRESH capture on a different pod (1x H100 vs the parent's 8x H100),
# so the bar follows the gotchas two-bar recipe for bf16 SINGLE-POSITION states
# (deep-layer bf16 batch-geometry jitter concentrates in the last layers —
# #779/#1005): early layers carry a sharp 0.999 per-layer bar (mask/RoPE/pad
# bugs corrupt layer 0 immediately), the flattened all-layer read a 0.995 bar.
PARITY_EARLY_LAYERS: tuple[int, ...] = (0, 1, 2, 3)
PARITY_EARLY_COS_MIN = 0.999
PARITY_FLAT_COS_MIN = 0.995

RC_OK = 0
RC_PARITY_GATE = 23  # designed halt, never an anonymous rc=1 (#1415 routing)

# Smoke slice: >=1 family per (slot x variant) with add AND replace covered on
# every slot, BOTH arms per family (delta-kind AND state-kind donor walks run).
SMOKE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("qtext", "joint_all", "a1"),
    ("qtext", "joint_mid", "replace"),
    ("pspan_tmpl", "joint_mid", "a1"),
    ("pspan_tmpl", "joint_all", "replace"),
    ("pspan_text", "joint_all", "a0.5"),
    ("pspan_text", "joint_mid", "replace"),
)

# ── extra-slot registration (import-time, idempotent) ──────────────────
# Point-of-use registration: EVERY process importing this module (grid run,
# tests, import-check) has the slots registered before any run_block call —
# never a phase-ordering side effect (#1315 r6 / #1090 fu6 registry lessons).

for _slot in FU2_SLOTS:
    R.register_extra_slot(
        R.ExtraSlot(
            name=_slot,
            positions_key=f"{_slot}_positions",
            vectors_key=f"{_slot}_vectors",
            prefix_scoped=_slot in PSPAN_SLOTS,
        )
    )


# ── pure derivations (CPU-only; pinned in tests/test_issue2094_fu2.py) ──


def fu2_slot_positions(tok, ids: list[int], prefix_end: int) -> dict[str, tuple[int, ...]]:
    """Per-slot edit positions (UNPADDED context coordinates) for one context.

    Structural + fail-loud (scope-marker duty): the template mask comes from
    the tokenized chat-template structure, and every boundary this round
    EXCLUDES is asserted by special-token id — the ``<|im_start|>user\\n``
    header at the front of the final user turn and the trailing
    ``<|im_end|>\\n<|im_start|>assistant\\n``. The qtext positions are also
    asserted to be exactly the contiguous interior (union check: qtext plus
    the excluded template positions reconstitute the shipped qspan).
    """
    mask = BANK.template_token_mask(tok, ids)
    ctx_len = len(ids)
    im_start = tok.convert_tokens_to_ids(BANK.IM_START_TOKEN)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    nl = tok("\n", add_special_tokens=False)["input_ids"][0]
    # Final-user-turn header: <|im_start|>user\n at [prefix_end, prefix_end+3).
    assert ids[prefix_end] == im_start, (prefix_end, ids[prefix_end])
    assert tok.decode([ids[prefix_end + 1]]) == "user", tok.decode([ids[prefix_end + 1]])
    assert ids[prefix_end + 2] == nl, ids[prefix_end + 2]
    # Trailing template: <|im_end|>\n<|im_start|>assistant\n at [ctx_len-5, ctx_len).
    assert ids[ctx_len - 5] == im_end, ids[ctx_len - 5]
    assert ids[ctx_len - 4] == nl, ids[ctx_len - 4]
    assert ids[ctx_len - 3] == im_start, ids[ctx_len - 3]
    assert tok.decode([ids[ctx_len - 2]]) == "assistant", tok.decode([ids[ctx_len - 2]])
    assert ids[ctx_len - 1] == nl, ids[ctx_len - 1]

    qtext = tuple(i for i in range(prefix_end, ctx_len) if not mask[i])
    expected_qtext = tuple(range(prefix_end + QTEXT_HEADER_TOKENS, ctx_len - QTEXT_TAIL_TOKENS))
    assert qtext == expected_qtext, (qtext[:4], expected_qtext[:4], prefix_end, ctx_len)
    assert len(qtext) >= 1, (prefix_end, ctx_len)
    # Union check: qtext + excluded template == the shipped qspan.
    span_template = tuple(i for i in range(prefix_end, ctx_len) if mask[i])
    assert set(qtext) | set(span_template) == set(range(prefix_end, ctx_len))

    pspan_tmpl = tuple(range(prefix_end))
    pspan_text = tuple(i for i in range(prefix_end) if not mask[i])
    assert len(pspan_text) >= 1, prefix_end
    prefix_template = tuple(i for i in range(prefix_end) if mask[i])
    assert set(pspan_text) | set(prefix_template) == set(range(prefix_end))
    assert set(pspan_text) < set(pspan_tmpl)
    return {"qtext": qtext, "pspan_tmpl": pspan_tmpl, "pspan_text": pspan_text}


def fu2_pair_ids(pairs: list[BANK.Pair], slot: str) -> tuple[str, ...]:
    """Pair eligibility per slot: prefix-span slots EXCLUDE matched-prefix
    pairs (identical prefixes => identical prefix states by causal identity —
    the steered edit is a no-op, the shipped grid's degenerate_self class);
    qtext runs matched-prefix + matched-query + cross (matched-query is legal:
    same query tokens, different upstream prefix => the states differ)."""
    assert slot in FU2_SLOTS, slot
    if slot == "qtext":
        return tuple(p.pair_id for p in pairs)
    return tuple(p.pair_id for p in pairs if p.setting != "matched_prefix")


def enumerate_fu2_families(pairs: list[BANK.Pair]) -> list[tuple[R.Block, R.Block]]:
    """The 30 (steered, null) fu2 block families = 60 blocks.

    qtext:      2 variants x 5 doses x 60 pairs -> 600 steered cells.
    pspan_tmpl: 2 variants x 5 doses x 30 pairs -> 300 steered cells.
    pspan_text: 2 variants x 5 doses x 30 pairs -> 300 steered cells.
    The shuffled-donor null mirrors every steered cell (1200 + 1200 = 2400).
    """
    families: list[tuple[R.Block, R.Block]] = []
    for slot in FU2_SLOTS:
        ids = fu2_pair_ids(pairs, slot)
        for variant in FU2_VARIANTS:
            for dose in R.DOSES_A:
                families.append(
                    (
                        R.Block(slot, variant, dose, "A", "steered", ids),
                        R.Block(slot, variant, dose, "A", "null", ids),
                    )
                )
    keys = [b.key for fam in families for b in fam]
    assert len(set(keys)) == len(keys), "duplicate fu2 block keys"
    return families


def fu2_smoke_pair_subset(pairs: list[BANK.Pair], slot: str) -> tuple[str, ...]:
    """One eligible pair per setting + a conv-context pair (the multi-turn
    history render seam stays smoke-visible — mirrors ``FU1.smoke_pair_subset``;
    for qtext the conv pair has ``context_a=conv`` and exercises the
    generation-side history render, for the pspan slots conv appears only as
    ``context_b`` by canonical pair direction)."""
    keep = set(fu2_pair_ids(pairs, slot))
    eligible = [p for p in pairs if p.pair_id in keep]
    ids: list[str] = []
    for setting in ("matched_prefix", "matched_query", "cross"):
        first = next((p.pair_id for p in eligible if p.setting == setting), None)
        if first is not None:
            ids.append(first)
    conv = next(
        (p.pair_id for p in eligible if p.a.startswith("conv") or p.b.startswith("conv")),
        None,
    )
    if conv is not None and conv not in ids:
        ids.append(conv)
    assert ids, f"empty fu2 smoke pair subset for {slot}"
    return tuple(ids)


def slice_fu2_smoke(
    families: list[tuple[R.Block, R.Block]], pairs: list[BANK.Pair]
) -> list[tuple[R.Block, R.Block]]:
    """The SMOKE_FAMILIES subset with per-slot smoke pair subsets."""
    from dataclasses import replace as dc_replace

    keep = set(SMOKE_FAMILIES)
    out: list[tuple[R.Block, R.Block]] = []
    for steered, null in families:
        spec = (steered.slot, steered.layer_variant, steered.dose)
        if spec not in keep:
            continue
        subset = fu2_smoke_pair_subset(pairs, steered.slot)
        assert set(subset) <= set(steered.pair_ids), (spec, subset)
        out.append((dc_replace(steered, pair_ids=subset), dc_replace(null, pair_ids=subset)))
    assert len(out) == len(SMOKE_FAMILIES), (len(out), len(SMOKE_FAMILIES))
    return out


def fu2_regime_fingerprint(cfg: R.RunConfig, bank_sha: str) -> str:
    """The parent regime fingerprint + the fu2 machinery version token —
    every output-affecting knob (#722 r3), incl. the slot/alignment recipe."""
    base = R.regime_fingerprint(cfg, bank_sha)
    return hashlib.sha256(f"{base}|{FU2_REGIME_TOKEN}".encode()).hexdigest()[:16]


# ── paths ───────────────────────────────────────────────────────────────


@dataclass
class FU2Paths:
    """fu2-bank-side output tree (rollouts / va / manifests ride RunConfig's
    own dirs so ``R.run_block`` + the fu1 upload seams are reused verbatim)."""

    out_root: Path

    @property
    def fu2_bank_dir(self) -> Path:
        return self.out_root / "fu2_bank"

    @property
    def fu2_bank_path(self) -> Path:
        return self.fu2_bank_dir / "fu2_bank.pt"

    @property
    def fu2_bank_done(self) -> Path:
        return self.fu2_bank_dir / "fu2_bank_done.json"

    @property
    def parity_path(self) -> Path:
        return self.fu2_bank_dir / "fu2_bank_parity.json"


# ── fu2 bank (full-position capture + per-slot positions/vectors) ───────


@torch.no_grad()
def capture_fu2_bank(cfg: R.RunConfig, model, tok, donor_map: dict[str, str]) -> dict:
    """Fresh all-layer capture over EVERY context position; per-slot positions
    + vectors precomputed into each record (the ExtraSlot contract).

    Records keep the parent keys (``ctx_len``/``prefix_end``/``q_span``/
    ``v_pe``) so the parent machinery (run_block, parity gate) reads them
    unchanged; positions come from the concatenated token ids' own structure,
    never a re-tokenized string (BPE-seam rule).
    """
    contexts = BANK.build_contexts()
    ctx_ids = {cid: BANK.context_token_ids_2094(tok, c) for cid, c in contexts.items()}
    prefix_ends = {cid: BANK.prefix_end_index_multi(tok, ids) for cid, ids in ctx_ids.items()}
    layers = cfg.layers
    pad_id = tok.pad_token_id
    records: dict[str, dict] = {}
    order = list(contexts)
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        ids, mask = R._right_pad([ctx_ids[c] for c in chunk], pad_id, cfg.device)
        captured = R.extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, cid in enumerate(chunk):
            row_ids = ctx_ids[cid]
            ctx_len = len(row_ids)
            pe = prefix_ends[cid]
            slot_pos = fu2_slot_positions(tok, row_ids, pe)
            full = torch.stack(
                [captured[layer][j, :ctx_len] for layer in layers], dim=1
            ).float()  # (ctx_len, L, H)
            assert full.shape == (ctx_len, len(layers), cfg.hidden), full.shape
            full = full.cpu()
            rec: dict = {
                "context_id": cid,
                "prefix": contexts[cid]["prefix"],
                "query_id": contexts[cid]["query_id"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "nq": ctx_len - pe,
                "q_span": full[pe:ctx_len].clone(),
                "v_pe": full[pe - 1].clone(),
            }
            for slot, pos in slot_pos.items():
                idx = torch.tensor(pos, dtype=torch.long)
                vecs = full[idx].clone()
                assert vecs.shape == (len(pos), len(layers), cfg.hidden), vecs.shape
                rec[f"{slot}_positions"] = list(pos)
                rec[f"{slot}_vectors"] = vecs
            records[cid] = rec
        del captured
    assert len(records) == len(contexts), (len(records), len(contexts))
    _, bank_sha = R.bank_manifest_and_sha()
    return {
        "layers": layers,
        "per_context": records,
        "donor_derangement": donor_map,
        "bank_sha": bank_sha,
        "fu2_slots": list(FU2_SLOTS),
        "alignment": (
            "Type-A right-aligned min-overlap WITHIN each slot's own coordinate "
            "set (content tokens for qtext/pspan_text; the full prefix span for "
            "pspan_tmpl); donors aligned + norm-matched with the same machinery"
        ),
        "repro": R._repro(cfg),
    }


def fu2_parity_report(fu2_bank: dict, parent_bank: dict) -> dict:
    """Capture-parity of the fresh fu2 bank vs the parent's staged bank on the
    OVERLAPPING reads (``q_span``/``v_pe``), two-bar (see PARITY_* constants).

    Both banks store fp32 CPU states of the same contexts; a real capture bug
    (offset/mask/pad) corrupts layer 0 immediately, whereas cross-pod bf16
    batch-geometry jitter concentrates in deep layers (#779/#1005).
    """
    early_min = 1.0
    flat_min = 1.0
    per_context: list[dict] = []
    for cid, rec in sorted(fu2_bank["per_context"].items()):
        prec = parent_bank["per_context"][cid]
        row = {"context_id": cid}
        for name in ("q_span", "v_pe"):
            a, b = rec[name], prec[name]
            if a.dim() == 2:  # v_pe: (L, H) -> (1, L, H)
                a, b = a.unsqueeze(0), b.unsqueeze(0)
            assert a.shape == b.shape, (name, a.shape, b.shape)
            cos_pl = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # (P, L)
            early = float(cos_pl[:, list(PARITY_EARLY_LAYERS)].min())
            flat = float(
                torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1)
            )
            row[name] = {"early_min_cos": early, "flat_cos": flat}
            early_min = min(early_min, early)
            flat_min = min(flat_min, flat)
        per_context.append(row)
    passed = early_min >= PARITY_EARLY_COS_MIN and flat_min >= PARITY_FLAT_COS_MIN
    return {
        "passed": bool(passed),
        "early_layers": list(PARITY_EARLY_LAYERS),
        "early_min_cos": early_min,
        "early_bar": PARITY_EARLY_COS_MIN,
        "flat_min_cos": flat_min,
        "flat_bar": PARITY_FLAT_COS_MIN,
        "per_context": per_context,
    }


def fu2_bank_is_done(paths: FU2Paths, regime_fp: str) -> bool:
    """Bank resume predicate — done-record present AND regime match; a
    cross-regime done-record is a HARD refusal (#722 r3), never silent reuse."""
    if not paths.fu2_bank_done.exists():
        return False
    rec = json.loads(paths.fu2_bank_done.read_text())
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"fu2 bank done-file carries regime_fp={rec.get('regime_fp')!r} but this "
            f"run's regime_fp={regime_fp!r} — refusing to resume across regimes "
            "(quarantine or use a fresh --out-root)"
        )
    assert paths.fu2_bank_path.exists(), paths.fu2_bank_path
    return True


# ── cap-hit report (pooled per (slot, lv, dose) per arm) ────────────────


def fu2_caphit_report(cfg: R.RunConfig) -> dict:
    """Realized cap-hit fractions at the run cap per pooled cell, per arm,
    from the per-block done manifests (the fu1 report shape; no 1024 baseline
    — every fu2 cell is new)."""
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
        cells.append(row)
    return {
        "max_new_tokens": cfg.max_new_tokens,
        "n_cells": len(cells),
        "cells": cells,
        "repro": R._repro(cfg),
    }


# ── upload + sentinel ───────────────────────────────────────────────────


def upload_and_sentinel(
    cfg: R.RunConfig,
    paths: FU2Paths,
    totals: dict,
    caphit: dict,
    parity: dict | None,
    wall_h: float,
) -> None:
    """Fail-loud bulk uploads (one commit per prefix, exact-set verified via
    ``R._upload_dir``) + the pod sentinel the VM poller drains.

    The upload-done manifest is written BEFORE the manifests-prefix commit so
    the done record itself rides that commit (fu1 review Minor, adopted).
    """
    uploaded: dict[str, list[str]] = {}
    uploaded["rollouts"] = R._upload_dir(
        cfg, cfg.rollouts_dir, f"{HF_FU2_TEXT}/rollouts", ["shard_*.jsonl"]
    )
    uploaded["va"] = R._upload_dir(cfg, cfg.va_dir, f"{HF_FU2_TENSORS}/va", ["shard_*.pt"])
    uploaded["bank"] = R._upload_dir(
        cfg, paths.fu2_bank_dir, f"{HF_FU2_TENSORS}/bank", ["fu2_bank.pt", "*.json"]
    )

    cap_rows = sum(
        agg["cap_hit"]
        for row in caphit["cells"]
        for arm, agg in row.items()
        if arm in ("steered", "null")
    )
    n_rows = sum(
        agg["n"]
        for row in caphit["cells"]
        for arm, agg in row.items()
        if arm in ("steered", "null")
    )
    payload = {
        "eval_numbers": {
            "followup_label": FU2_LABEL,
            "grid_totals": totals,
            "blocks_done": len(list((cfg.manifest_dir / "blocks").glob("*.done.json"))),
            "rows_persisted": n_rows,
            "cap_hit_rows": cap_rows,
            "cap_hit_frac": (cap_rows / n_rows) if n_rows else 0.0,
            "caphit_per_cell": caphit["cells"],
            "bank_parity": (
                {k: parity[k] for k in ("passed", "early_min_cos", "flat_min_cos")}
                if parity is not None
                else "skipped (tiny smoke: no parent bank)"
            ),
        },
        "eval_paths": sorted(
            str(p) for p in (cfg.rollouts_dir, cfg.va_dir, paths.fu2_bank_path, cfg.manifest_dir)
        ),
        "reproducibility_card": {
            **R._repro(cfg),
            "seed_base": cfg.seed_base,
            "bank_seed": BANK.SEED,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": R.GRID_TEMPERATURE,
            "gen_batch": cfg.gen_batch,
            "fu2_slots": list(FU2_SLOTS),
            "fu2_variants": list(FU2_VARIANTS),
            "fu2_regime_token": FU2_REGIME_TOKEN,
        },
        "wandb_url": None,
        "hf_hub_url": (f"https://huggingface.co/datasets/{R.HF_DATA_REPO}/tree/main/{R.HF_PREFIX}"),
        "worktree_path": str(R.REPO_ROOT),
        "final_commit_sha": R._git_sha(),
        "gpu_hours_used": round(wall_h, 3),
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "fu2 user-chat round: Type-A only (the scope names no Type-B arm for "
            "the span slots); layer variants joint_all + joint_mid only",
            "prefix-span slots exclude matched-prefix pairs (identical prefixes "
            "=> degenerate_self no-op edits, the shipped grid's carve-out class)",
            "alignment refinement (delegated by the scope marker): Type-A deltas "
            "and donor nulls right-align WITHIN each slot's own coordinate set "
            "(content tokens for qtext/pspan_text; full span for pspan_tmpl)",
            "replace dose on the fu2 multi-position slots is realized as the "
            "equivalent add_full_state_patch (the parent l3j/qspan convention)",
        ],
        # ALL FOUR write prefixes enumerated (#1773) — the manifests prefix is
        # uploaded AFTER this record is written; the sentinel carries its count.
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
        "upload_prefix_names": {
            "rollouts": f"{HF_FU2_TEXT}/rollouts",
            "va": f"{HF_FU2_TENSORS}/va",
            "bank": f"{HF_FU2_TENSORS}/bank",
            "manifests": f"{HF_FU2_TENSORS}/manifests",
        },
    }
    # Written BEFORE the manifests commit => included in it (fu1 Minor fix).
    R._write_json_atomic(cfg.manifest_dir / "fu2_upload_done.json", payload)
    uploaded["manifests"] = R._upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_FU2_TENSORS}/manifests",
        ["*.json", "blocks/*.done.json"],
    )
    payload["uploaded_prefixes"] = {k: len(v) for k, v in uploaded.items()}

    sentinel = cfg.log_dir / SENTINEL_NAME
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    R._write_json_atomic(
        sentinel,
        {"sentinel_schema_version": 1, "kind": kind, "version": 1, "note": payload},
    )
    logger.info("[upload] sentinel written: %s (kind=%s)", sentinel, kind)


# ── entrypoint ───────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths
    (the #1689 false-pass class): transformers loads inside
    ``R.load_model_and_tokenizer``, hub helpers inside ``FU1.stage_bank`` /
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
        description="Issue #2094 fu2_span_slots pod driver (qtext / pspan_tmpl / pspan_text)."
    )
    ap.add_argument("--run", action="store_true", help="execute stage->bank->grid->upload")
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/issue2094_fu2_out"))
    ap.add_argument("--log-dir", type=Path, default=R.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=R.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument(
        "--tiny-layers",
        type=int,
        default=28,
        help="tiny model layer count (default 28 so the production joint_mid band exists)",
    )
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=FU2_MAX_NEW_TOKENS,
        help="2048 from the start (fu1's raised cap; scope-marker requirement)",
    )
    ap.add_argument("--seed-base", type=int, default=R.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class family slice")
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument(
        "--upload-every",
        type=int,
        default=10,
        help="bulk-upload completed rollout shards every N blocks (256 commits/hr cap)",
    )
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> R.RunConfig:
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    return R.RunConfig(
        phase="fu2_span_slots",
        out_root=args.out_root,
        log_dir=args.log_dir,
        model_id=args.model_id,
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else R.N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else R.HIDDEN_FULL,
        device=device,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens,
        anchor_draws=1,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=False,
        force=False,
        worker_index=0,
        num_workers=1,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=2.5,  # 40 x 60-row-block-equivalents at the fu1 basis + overheads
        gpu_hours_budgeted=8.0,  # scope-marker upper estimate
    )


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
    cfg = build_config(args)
    paths = FU2Paths(out_root=cfg.out_root)
    for d in (
        cfg.rollouts_dir,
        cfg.va_dir,
        cfg.manifest_dir / "blocks",
        paths.fu2_bank_dir,
        cfg.log_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    # ── phase: enumerate (CPU-only) ─────────────────────────────────────
    logger.info("[phase=fu2_enumerate] smoke=%s tiny=%s", cfg.smoke, cfg.tiny)
    pairs = BANK.build_pairs()
    families = enumerate_fu2_families(pairs)
    totals = R.grid_totals(families)
    assert totals == EXPECTED_FU2_TOTALS, (totals, EXPECTED_FU2_TOTALS)
    if cfg.smoke:
        families = slice_fu2_smoke(families, pairs)
        totals = R.grid_totals(families)
    logger.info("[enumerate] %s", totals)

    # ── phase: stage (parent bank: donor derangement + parity reference) ─
    logger.info("[phase=fu2_stage]")
    parent_bank: dict | None = None
    if cfg.tiny:
        logger.info("[stage] tiny smoke: no parent bank — seeded derangement recomputed")
        donor_map = BANK.donor_derangement(pairs)
    else:
        FU1.stage_bank(cfg, args.hf_revision)
        parent_bank = torch.load(
            cfg.bank_dir / "vc_bank.pt", map_location="cpu", weights_only=False
        )
        donor_map = parent_bank.get("donor_derangement") or BANK.donor_derangement(pairs)
        # The derangement is seeded + deterministic — assert the staged copy
        # matches the recomputation (reuse-validation on the reused null).
        assert donor_map == BANK.donor_derangement(pairs), "parent derangement drift"
        # Staged bank must match the current bank module's recipe (fu1 review Minor 4).
        _staged_sha = parent_bank.get("bank_sha")
        _local_sha = R.bank_manifest_and_sha()[1]
        assert _staged_sha in (None, _local_sha), (_staged_sha, _local_sha)

    # ── phase: bank (fresh full-position capture + parity gate) ─────────
    logger.info("[phase=fu2_bank]")
    _, bank_sha = R.bank_manifest_and_sha()
    regime_fp = fu2_regime_fingerprint(cfg, bank_sha)
    parity: dict | None = None
    if fu2_bank_is_done(paths, regime_fp):
        logger.info("[bank] fu2 bank already captured for this regime — loading")
        fu2_bank = torch.load(paths.fu2_bank_path, map_location="cpu", weights_only=False)
        if paths.parity_path.exists():
            parity = json.loads(paths.parity_path.read_text())
        model, tok = R.load_model_and_tokenizer(cfg)
    else:
        model, tok = R.load_model_and_tokenizer(cfg)
        fu2_bank = capture_fu2_bank(cfg, model, tok, donor_map)
        if parent_bank is not None:
            parity = fu2_parity_report(fu2_bank, parent_bank)
            R._write_json_atomic(paths.parity_path, parity)
            logger.info(
                "[bank] parity vs parent bank: passed=%s early_min=%.6f flat_min=%.6f",
                parity["passed"],
                parity["early_min_cos"],
                parity["flat_min_cos"],
            )
            if not parity["passed"]:
                logger.error("[bank] capture-parity gate FAILED — report at %s", paths.parity_path)
                print("[phase=fu2_parity_gate_failed]", flush=True)
                return RC_PARITY_GATE
        else:
            logger.info("[bank] parity gate skipped (tiny smoke: no parent bank)")
        R._save_pt_atomic(paths.fu2_bank_path, fu2_bank)
        R._write_json_atomic(
            paths.fu2_bank_done,
            {
                "regime_fp": regime_fp,
                "n_contexts": len(fu2_bank["per_context"]),
                "parity": ("skipped-tiny" if parity is None else parity["passed"]),
                "repro": R._repro(cfg),
            },
        )
    del parent_bank

    # ── phase: grid (R.run_block verbatim over the fu2 slots) ───────────
    pairs_by_id = {p.pair_id: p for p in pairs}
    eot = R.eot_tail_ids(tok)
    blocks = R.blocks_for_worker(families, 0, 1)
    logger.info(
        "[phase=fu2_grid] %d blocks / %d cells at max_new_tokens=%d",
        len(blocks),
        sum(b.n_cells for b in blocks),
        cfg.max_new_tokens,
    )
    t0 = time.monotonic()
    pending: list[R.Block] = []
    for k, block in enumerate(blocks, start=1):
        if R.block_is_done(cfg.out_root, block, regime_fp):
            logger.info("[grid] block %d/%d %s SKIP (done)", k, len(blocks), block.key)
            continue
        rec = R.run_block(cfg, model, tok, fu2_bank, block, pairs_by_id, donor_map, eot, regime_fp)
        pending.append(block)
        logger.info(
            "[grid] block %d/%d %s rows=%d cap_hit=%d elapsed=%.1fs",
            k,
            len(blocks),
            block.key,
            rec["n_cells"],
            rec["n_cap_hit"],
            time.monotonic() - t0,
        )
        if cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            R._upload_dir(
                cfg,
                cfg.rollouts_dir,
                f"{HF_FU2_TEXT}/rollouts",
                [f"shard_{b.slug}.jsonl" for b in pending],
            )
            pending = []

    caphit = fu2_caphit_report(cfg)
    R._write_json_atomic(cfg.manifest_dir / "fu2_caphit.json", caphit)
    over = [
        (c["slot"], c["layer_variant"], c["dose"], c["steered"]["cap_hit_frac"])
        for c in caphit["cells"]
        if c.get("steered", {}).get("cap_hit_frac", 0.0) > FU1.CAPHIT_TRIGGER_FRAC
    ]
    logger.info(
        "[grid] cap-hit report at %d: %d/%d pooled cells over the %.0f%% trigger: %s",
        cfg.max_new_tokens,
        len(over),
        len(caphit["cells"]),
        100 * FU1.CAPHIT_TRIGGER_FRAC,
        over,
    )

    # ── phase: upload + sentinel ────────────────────────────────────────
    logger.info("[phase=fu2_upload]")
    wall_h = (time.monotonic() - t_start) / 3600.0
    upload_and_sentinel(cfg, paths, totals, caphit, parity, wall_h)
    logger.info("[phase=fu2_done]")
    print("[phase=done]", flush=True)
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
