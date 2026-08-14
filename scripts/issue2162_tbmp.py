#!/usr/bin/env python3
"""Issue #2162 — turn-boundary-multipatch (tbmp) pod driver (plan v10).

Thin fork of ``scripts/issue2162_run.py`` (the ``issue2162_ladder.py``
precedent): parent helpers are IMPORTED, never re-implemented — the claim
queue, ``_arm_hook_all_layers``, the pilot gate, the upload seam, io helpers.
This round adds a multi-position slot ``tb`` covering EVERY turn-boundary
index jointly plus a per-boundary single-position sweep ``tb_k`` (plan §4.2),
to separate trace-decay from trace-dispersion at depth.

Phases (plan §4.5 DAG; out-root default ``/workspace/issue2162_out/tbmp``):

- ``--phase bank`` (P1): resolve turn boundaries INDEPENDENTLY per side from
  each side's own token ids over the 12-cell capture set (432 contexts), run
  the G1 resolver gate (designed-count table, per-pair constant offset, d1 ==
  ce identity, cross-type alignment legality — HALT ``RC_G1_RESOLVER``),
  capture all-layer boundary states (one right-padded forward per chunk),
  freeze the seeded cross-type assignment into ``tb_config.json``, and run the
  MULTI-POSITION injection-exactness gate (12 spots spanning all 3 arms x
  d1/d3/d5 + sweep variants; HALT ``RC_INJECTION_GATE``).
- ``--phase grid`` (P2): the 45 (cell x variant x arm) blocks via the SHARED
  claim-file queue (namespace ``blocks_tbmp``), d1 blocks queued FIRST; per
  block K=5 temp-1.0 hooked draws per pair + inline margin TF when the pools
  file is staged; per-block JSONL checkpoints + incremental upload.
  ``--pilot`` times ONE production-shape block (``RC_PILOT_GATE`` refusal).
- ``--phase margin`` (P2m): pools-dependent per-block margin TF catch-up
  (claim-queue namespace ``margin_blocks_tbmp``).
- ``--phase upload`` (P3): ONE bulk ``upload_folder`` commit per HF prefix +
  the pod sentinel ``/workspace/logs/issue-2162-tbmp-results.json``.

Pod-side contract: sentinel + ``[phase=...]`` breadcrumbs only; never shells
out to ``scripts/task.py``. Every exit is an explicit ``sys.exit`` (#1689).
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

import issue2162_run as R  # noqa: E402  (script dir on sys.path in script mode)
from issue2094_run import align_right  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2094.fmetrics import safe_cosine  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2162.tbmp")

# ── constants (plan §4.1/§4.2/§4.3/§9/§10) ────────────────────────────

ROUND = "tbmp"
DEFAULT_OUT_ROOT = Path("/workspace/issue2162_out/tbmp")
SENTINEL_NAME = "issue-2162-tbmp-results.json"
SENTINEL_NAME_SMOKE = "issue-2162-tbmp-smoke-results.json"
HF_TBMP = f"{R.HF_PREFIX}/analysis_tensors/tbmp"
HF_TBMP_GRID = f"{R.HF_PREFIX}/raw_completions/tbmp/grid"

GRID_CELLS: tuple[str, ...] = (
    "instr_format",
    "persona_prompted",
    "recency_instr_format_d3",
    "recency_instr_format_d5",
    "recency_persona_prompted_d3",
    "recency_persona_prompted_d5",
    "recency_fact_user_name_d5",  # control
)
CONTROL_CELL = "recency_fact_user_name_d5"
SWEEP_CELLS: tuple[str, ...] = (
    "recency_instr_format_d3",
    "recency_instr_format_d5",
    "recency_persona_prompted_d3",
    "recency_persona_prompted_d5",
)
DONOR_POOL_CELLS: tuple[str, ...] = (
    "prior_topic",
    "fact_user_name",
    "recency_prior_topic_d3",
    "recency_prior_topic_d5",
    "recency_fact_user_name_d3",
)
CAPTURE_CELLS: tuple[str, ...] = GRID_CELLS + DONOR_POOL_CELLS

# Designed boundary counts — ALL 12 capture cells (plan §4.2 table, G1(a)).
# boundary t (t=1..n_a) = the <|im_end|> closing the t-th ASSISTANT turn;
# boundary n_a+1 = ctx_len - 1 (the parent's ce position).
DESIGNED_BOUNDARIES: dict[str, int] = {
    "instr_format": 1,
    "persona_prompted": 1,
    "recency_instr_format_d3": 3,
    "recency_persona_prompted_d3": 3,
    "recency_instr_format_d5": 5,
    "recency_persona_prompted_d5": 5,
    "recency_fact_user_name_d5": 6,
    "prior_topic": 2,
    "fact_user_name": 2,
    "recency_prior_topic_d3": 4,
    "recency_fact_user_name_d3": 4,
    "recency_prior_topic_d5": 6,
}

# Cross-type-donor pools (plan §4.2 arm 3): same-depth recency shape, different
# information type. Both pool members carry K+1 boundaries vs the recipient's K
# at every depth — right-align-drop-earliest (rule (a)); a SHORTER donor is
# FORBIDDEN (rule (b)) — the control cell therefore draws from
# recency_prior_topic_d5 ONLY (6 boundaries, exact match).
CROSSTYPE_POOL: dict[str, tuple[str, ...]] = {
    "instr_format": ("prior_topic", "fact_user_name"),
    "persona_prompted": ("prior_topic", "fact_user_name"),
    "recency_instr_format_d3": ("recency_prior_topic_d3", "recency_fact_user_name_d3"),
    "recency_persona_prompted_d3": ("recency_prior_topic_d3", "recency_fact_user_name_d3"),
    "recency_instr_format_d5": ("recency_prior_topic_d5", "recency_fact_user_name_d5"),
    "recency_persona_prompted_d5": ("recency_prior_topic_d5", "recency_fact_user_name_d5"),
    CONTROL_CELL: ("recency_prior_topic_d5",),
}
CROSSTYPE_STREAM_KEY = "tbmp_crosstype"  # seeded RNG stream (seed 2162), frozen

JOINT_ARMS: tuple[str, ...] = ("steered", "shuffled", "crosstype")
SWEEP_ARMS: tuple[str, ...] = ("steered", "shuffled")

RC_G1_RESOLVER = 24  # typed HALT — distinct from parent 21/22/23
REALIZED_MODE = "replace_multi"  # recorded per cell (plan §4.2, #2094 convention)

PLANNED_GRID_WALL_H = 1.05  # plan §9 P2 at width 4 (fence 2x -> 2.5 h wall)
SMOKE_PAIRS = 6  # plan §4.6: 6 pairs x K=2 per smoke block


# ── boundary resolver (plan §4.2 — the load-bearing correctness core) ─


def _role_marker_ids(tok) -> tuple[int, int, int]:
    """``(im_start_id, im_end_id, assistant_role_id)`` — all single tokens."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_start, int) and im_start >= 0, im_start
    assert isinstance(im_end, int) and im_end >= 0, im_end
    a_ids = tok("assistant", add_special_tokens=False)["input_ids"]
    assert len(a_ids) == 1, f"'assistant' is not a single token: {a_ids}"
    return im_start, im_end, a_ids[0]


def turn_boundaries(ids: list[int], markers: tuple[int, int, int]) -> list[int]:
    """Token indices of the ``<|im_end|>`` closing each ASSISTANT turn.

    Special tokens never BPE-merge, so the detection is exact on both pair
    sides. The trailing generation prompt (``<|im_start|>assistant\\n`` with no
    ``<|im_end|>``) contributes nothing by construction.
    """
    im_start, im_end, a_id = markers
    out: list[int] = []
    in_assistant = False
    for i, t in enumerate(ids):
        if t == im_start:
            in_assistant = i + 1 < len(ids) and ids[i + 1] == a_id
        elif t == im_end and in_assistant:
            out.append(i)
            in_assistant = False
    return out


def resolve_boundaries(tok, contexts: dict[str, dict]) -> dict[str, dict]:
    """Per capture context: token ids, prefix_end, and the FULL boundary set
    (assistant ``<|im_end|>`` indices + the final ce position ``ctx_len-1``)."""
    markers = _role_marker_ids(tok)
    out: dict[str, dict] = {}
    for cid, ctx in contexts.items():
        assert ctx.get("role_header") in (None, "assistant"), (
            cid,
            "capture cells never swap the generation role header",
        )
        ids = BANK.context_token_ids_2162(tok, ctx)
        pe = BANK.prefix_end_index_multi(tok, ids)
        bounds = turn_boundaries(ids, markers) + [len(ids) - 1]
        out[cid] = {
            "context_id": cid,
            "cell": ctx["cell"],
            "value_id": ctx["value_id"],
            "carrier": ctx["carrier"],
            "ctx_len": len(ids),
            "prefix_end": pe,
            "boundaries": bounds,
        }
    return out


def g1_resolver_report(
    resolved: dict[str, dict],
    pairs: list[BANK.Pair2162],
    crosstype: dict[str, str],
    pairs_by_id: dict[str, BANK.Pair2162],
) -> dict:
    """Plan §7 G1 — (a) designed counts, (b) per-pair constant offset,
    (c) d1 == ce identity, (d) cross-type alignment legality."""
    violations: list[dict] = []
    for cid, rec in resolved.items():
        want = DESIGNED_BOUNDARIES[rec["cell"]]
        if len(rec["boundaries"]) != want:
            violations.append(
                {
                    "check": "designed_count",
                    "context_id": cid,
                    "cell": rec["cell"],
                    "designed": want,
                    "resolved": len(rec["boundaries"]),
                }
            )
    capture_pairs = [p for p in pairs if p.cell in CAPTURE_CELLS]
    for p in capture_pairs:
        ra, rb = resolved[p.a], resolved[p.b]
        if len(ra["boundaries"]) != len(rb["boundaries"]):
            violations.append({"check": "pair_count", "pair_id": p.pair_id})
            continue
        off = rb["ctx_len"] - ra["ctx_len"]
        for t, (pa, pb) in enumerate(zip(ra["boundaries"], rb["boundaries"], strict=True), start=1):
            if pb - pa != off:
                violations.append(
                    {
                        "check": "constant_offset",
                        "pair_id": p.pair_id,
                        "turn": t,
                        "pos_a": pa,
                        "pos_b": pb,
                        "len_delta": off,
                    }
                )
    for cid, rec in resolved.items():
        if DESIGNED_BOUNDARIES[rec["cell"]] != 1:
            continue
        ce = R.slot_position(rec["ctx_len"], rec["prefix_end"], "ce")
        if rec["boundaries"] != [rec["ctx_len"] - 1] or rec["boundaries"][0] != ce:
            violations.append(
                {
                    "check": "d1_ce_identity",
                    "context_id": cid,
                    "boundaries": rec["boundaries"],
                    "ce": ce,
                }
            )
    n_right_aligned = 0
    dropped_hist: dict[str, int] = {}
    resolved_by_pair = {p.pair_id: p for p in pairs}
    for rid, did in sorted(crosstype.items()):
        rp, dp = resolved_by_pair[rid], resolved_by_pair[did]
        n_r = len(resolved[rp.a]["boundaries"])
        n_d = len(resolved[dp.b]["boundaries"])
        if n_d < n_r:
            violations.append(
                {
                    "check": "shorter_donor_forbidden",
                    "pair_id": rid,
                    "donor_pair_id": did,
                    "n_r": n_r,
                    "n_d": n_d,
                }
            )
        elif n_d > n_r:
            n_right_aligned += 1
            dropped_hist[str(n_d - n_r)] = dropped_hist.get(str(n_d - n_r), 0) + 1
    return {
        "criterion": "G1 boundary-resolution gate (plan §7.1)",
        "designed_counts": DESIGNED_BOUNDARIES,
        "n_contexts": len(resolved),
        "n_pairs_checked": len(capture_pairs),
        "n_crosstype_assignments": len(crosstype),
        "n_right_aligned": n_right_aligned,
        "alignment_dropped_histogram": dropped_hist,
        "n_violations": len(violations),
        "violations": violations[:100],
        "passed": not violations,
    }


# ── cross-type assignment (seeded, frozen into tb_config.json) ────────


def build_crosstype_assignment(pairs: list[BANK.Pair2162]) -> dict[str, str]:
    """Recipient pair -> cross-type donor pair (plan §4.2 arm 3).

    Carrier-matched where the pool admits, seeded fallback otherwise (seed
    2162, fresh RNG stream key ``tbmp_crosstype``); drawn WITHOUT replacement
    per recipient cell so no donor repeats within a cell.
    """
    rng = random.Random(f"{BANK.SEED}:{CROSSTYPE_STREAM_KEY}")
    by_cell = BANK.pairs_by_cell(pairs)
    out: dict[str, str] = {}
    for cell in GRID_CELLS:
        recipients = sorted(by_cell[cell], key=lambda p: p.pair_id)
        pool: list[BANK.Pair2162] = []
        for donor_cell in CROSSTYPE_POOL[cell]:
            pool.extend(sorted(by_cell[donor_cell], key=lambda p: p.pair_id))
        assert len(pool) >= len(recipients), (cell, len(pool), len(recipients))
        for donor_cell in CROSSTYPE_POOL[cell]:
            assert DESIGNED_BOUNDARIES[donor_cell] >= DESIGNED_BOUNDARIES[cell], (
                f"shorter cross-type donor cell FORBIDDEN at assignment time: "
                f"{donor_cell} ({DESIGNED_BOUNDARIES[donor_cell]}) < "
                f"{cell} ({DESIGNED_BOUNDARIES[cell]})"
            )
        avail = pool[:]
        rng.shuffle(avail)
        for rp in recipients:
            same = [d for d in avail if d.carrier == rp.carrier]
            donor = same[0] if same else avail[0]
            avail.remove(donor)
            out[rp.pair_id] = donor.pair_id
    assert len(out) == len(GRID_CELLS) * 36, len(out)
    return out


def shuffled_assignment_with_parity(
    pairs: list[BANK.Pair2162], parent_bank_payload: dict, parent_bank_name: str
) -> tuple[dict[str, str], str]:
    """The parent's frozen within-cell value-constrained shuffled assignment.

    Recomputed via the SAME seeded function the parent froze
    (``BANK.donor_assignment_2162``, seed 2162 — deterministic) and
    parity-checked against the staged parent ``bank.json``'s donor map —
    MANDATORY on every path (plan §4.5 DAG stages ``bank.json``
    unconditionally at P1): a bank without a recognizable donor map is a
    RuntimeError, never a silent recompute-only skip. Returns
    ``(map restricted to grid pairs, parity note)``.
    """
    full = BANK.donor_assignment_2162(pairs)["shuffled"]
    grid_ids = {p.pair_id for p in pairs if p.cell in GRID_CELLS}
    ours = {k: v for k, v in full.items() if k in grid_ids}
    frozen = None
    for key in ("donor_assignments", "donor_assignment"):
        d = parent_bank_payload.get(key)
        if isinstance(d, dict):
            frozen = d.get("shuffled", d if all(isinstance(v, str) for v in d.values()) else None)
            break
    if not isinstance(frozen, dict):
        raise RuntimeError(
            f"no shuffled donor map found in {parent_bank_name} "
            f"(top-level keys: {sorted(parent_bank_payload)[:20]}) — the plan-§4.2 parity "
            "check cannot run; refusing (stage the parent's real "
            "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json)"
        )
    mismatches = [k for k in ours if frozen.get(k) != ours[k]]
    assert not mismatches, (
        f"shuffled donor assignment DRIFTED from parent bank.json on {len(mismatches)} "
        f"pairs (first: {mismatches[:3]}) — refusing (plan §4.2: reused verbatim)"
    )
    return ours, f"recomputed AND parity-verified against {parent_bank_name} ({len(ours)} pairs)"


# ── rebuilt-vs-recorded context parity (plan §12 assumption 9 + §10 item j) ──

_CONTEXT_PARITY_KEYS = ("cell", "value_id", "carrier", "system", "history", "user")


def frozen_gen_sha256_producer_domain() -> str:
    """The parent bank's ``frozen_gen_sha256`` pin recomputed in the
    PRODUCER's domain (``bank2162.py:1852/1904``): sha256 of the canonical
    JSON dump of the frozen-generation DICT — never the file's raw bytes
    (verified 2026-08-14: recompute == the banked pin ``b52f68c1…``; the raw
    file sha differs — the `.claude/rules/gotchas.md` sha-pin-domain rule)."""
    import hashlib

    frozen = BANK.load_frozen_gen()
    gen_blob = json.dumps(frozen or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(gen_blob.encode()).hexdigest()


def parent_bank_context_parity(payload: dict, capture_ctx: dict[str, dict]) -> dict:
    """Plan §12 assumption 9 — rebuilt-vs-recorded context parity, realized form.

    The staged parent ``bank.json`` records each context's FULL payload
    (system/history/user + identity fields) but NO ``ctx_len``/``prefix_end``
    (probed 2026-08-14 against a fresh HF download, 2,383,040 bytes: 0
    occurrences of either key; per-context keys are exactly {carrier, cell,
    history, id, system, user, value_id}), so the assumption's stated recipe
    cannot execute. Realized triple superseding it: (i) producer-domain
    ``frozen_gen_sha256`` recompute == the banked pin; (ii) EXACT per-context
    payload equality on ``_CONTEXT_PARITY_KEYS`` for all 432 capture contexts;
    (iii) the token-LENGTH leg — G1(b) re-rendered per-pair ``ctx_len`` delta
    == the parent's committed ``f_cells.jsonl`` ``len_delta``, 432/432
    (``g1b_len_delta_parity``, separate report section). Scope: strictly
    stronger at the TEXT level, but INCOMPARABLE at absolute token geometry —
    the one drift class the literal recorded-ctx_len form would catch and this
    form won't is a uniform render shift identical across both pair members
    that preserves per-cell boundary counts; not load-bearing here because
    this round re-captures its injection payloads at re-rendered positions and
    consumes nothing absolute-position-indexed from the parent.
    A structurally unusable bank (no ``contexts`` map) raises — never a skip.
    """
    recorded = payload.get("contexts")
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError(
            "parent bank.json carries no 'contexts' map — assumption-9 rebuilt-vs-recorded "
            f"parity cannot run (top-level keys: {sorted(payload)[:20]}); refusing"
        )
    violations: list[dict] = []
    banked_sha = payload.get("frozen_gen_sha256")
    local_sha = frozen_gen_sha256_producer_domain()
    if banked_sha != local_sha:
        violations.append(
            {
                "check": "frozen_gen_sha256",
                "banked": banked_sha,
                "recomputed_producer_domain": local_sha,
            }
        )
    for cid, ctx in sorted(capture_ctx.items()):
        rec = recorded.get(cid)
        if rec is None:
            violations.append({"check": "context_missing_from_bank", "context_id": cid})
            continue
        for k in _CONTEXT_PARITY_KEYS:
            if rec.get(k) != ctx.get(k):
                violations.append({"check": "context_payload_drift", "context_id": cid, "field": k})
                break
    return {
        "criterion": "plan §12 assumption 9, realized form — the banked artifact records no "
        "ctx_len/prefix_end (fresh-HF whole-blob probe: 0 occurrences of either key), so the "
        "literal recipe cannot execute; realized triple: (i) producer-domain frozen_gen_sha256 "
        "recompute == banked pin, (ii) exact per-context payload equality on "
        "{cell, value_id, carrier, system, history, user} for all 432 capture contexts, "
        "(iii) G1(b) re-rendered per-pair ctx_len delta == parent committed f_cells.jsonl "
        "len_delta, 432/432 (len_delta_parity section)",
        "scope_note": "strictly stronger at the TEXT level; incomparable at absolute token "
        "geometry — the residual drift class (a uniform render shift identical across both "
        "pair members preserving per-cell boundary counts) is not load-bearing: this round "
        "re-captures its injection payloads at re-rendered positions and consumes nothing "
        "absolute-position-indexed from the parent",
        "n_contexts_checked": len(capture_ctx),
        "parity_keys": list(_CONTEXT_PARITY_KEYS),
        "frozen_gen_sha256": banked_sha,
        "n_violations": len(violations),
        "violations": violations[:100],
        "passed": not violations,
    }


def g1b_len_delta_parity(
    resolved: dict[str, dict], pairs: list[BANK.Pair2162], parent_f_cells: Path
) -> dict:
    """Plan §10 item (j) as realized: the re-rendered pair's
    ``ctx_len_B − ctx_len_A`` must equal the parent's committed
    ``f_cells.jsonl`` ``len_delta`` for EVERY capture pair (the parent table
    covers all 38 cells ⊇ the 12 capture cells, single-valued per pair —
    counted 2026-08-14: 1,368 pairs, 0 multi-valued). Anti-vacuous by
    construction: a capture pair MISSING from the parent table is a violation,
    so an empty/mis-staged table can never pass silently. Table integrity: a
    CONFLICTING duplicate (same pair_id, different len_delta) anywhere in the
    parent table is a violation — an internally inconsistent table cannot
    serve as a parity reference; an identical-value duplicate is benign."""
    assert parent_f_cells.exists(), (
        f"{parent_f_cells} missing — the parent's committed f_cells.jsonl is required for the "
        "G1(b) len_delta rebuilt-vs-recorded parity (git eval_results/issue_2162/f_metrics/)"
    )
    recorded: dict[str, int] = {}
    conflicts: dict[str, list[int]] = {}
    with parent_f_cells.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("len_delta") is None:
                continue
            pid, val = row["pair_id"], int(row["len_delta"])
            prev = recorded.get(pid)
            if prev is None:
                recorded[pid] = val
            elif val != prev:
                vals = conflicts.setdefault(pid, [prev])
                if val not in vals:
                    vals.append(val)
    capture_pairs = [p for p in pairs if p.cell in CAPTURE_CELLS]
    violations: list[dict] = [
        {"check": "conflicting_duplicate_len_delta", "pair_id": pid, "values": sorted(vals)}
        for pid, vals in sorted(conflicts.items())
    ]
    n_checked = 0
    for p in capture_pairs:
        want = recorded.get(p.pair_id)
        if want is None:
            violations.append({"check": "pair_missing_from_parent_f_cells", "pair_id": p.pair_id})
            continue
        got = int(resolved[p.b]["ctx_len"]) - int(resolved[p.a]["ctx_len"])
        n_checked += 1
        if got != want:
            violations.append(
                {
                    "check": "len_delta_drift",
                    "pair_id": p.pair_id,
                    "recorded": want,
                    "re_rendered": got,
                }
            )
    return {
        "criterion": "plan §10 item (j) — re-rendered per-pair ctx_len delta == parent "
        "committed f_cells.jsonl len_delta, every capture pair",
        "n_capture_pairs": len(capture_pairs),
        "n_checked": n_checked,
        "n_violations": len(violations),
        "violations": violations[:100],
        "passed": not violations and n_checked == len(capture_pairs),
    }


# ── tb bank + payloads ────────────────────────────────────────────────


def tb_bank_dir(cfg: R.RunConfig) -> Path:
    return cfg.out_root / "tb_bank"


def gates_dir(cfg: R.RunConfig) -> Path:
    return cfg.out_root / "gates"


@torch.no_grad()
def capture_tb_bank(cfg: R.RunConfig, model, tok, resolved: dict[str, dict]) -> dict:
    """All-layer states at every boundary position per capture context —
    one right-padded forward per chunk (positions off token ids, BPE-seam rule)."""
    contexts = BANK.build_contexts()
    layers = cfg.layers
    pad_id = tok.pad_token_id
    records: dict[str, dict] = {}
    order = sorted(resolved)
    t0 = time.monotonic()
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        rows = [BANK.context_token_ids_2162(tok, contexts[c]) for c in chunk]
        ids, mask = R._right_pad(rows, pad_id, cfg.device)
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, cid in enumerate(chunk):
            rec = resolved[cid]
            assert len(rows[j]) == rec["ctx_len"], (cid, len(rows[j]), rec["ctx_len"])
            per_pos = [
                torch.stack([captured[layer][j, pos] for layer in layers])
                for pos in rec["boundaries"]
            ]
            tb = torch.stack(per_pos)  # (n_b, L, H)
            assert tb.shape == (len(rec["boundaries"]), len(layers), cfg.hidden), tb.shape
            records[cid] = {**rec, "tb": tb.float().cpu()}
        del captured
        print(
            f"[bank_tbmp] unit {min(start + cfg.capture_batch, len(order))}/{len(order)} "
            f"contexts elapsed={time.monotonic() - t0:.0f}s",
            flush=True,
        )
    assert len(records) == len(resolved), (len(records), len(resolved))
    return {"layers": layers, "per_context": records}


def tb_positions(rec_a: dict, slot: str) -> tuple[int, ...]:
    """Recipient edit positions for one variant (A-side unpadded coords)."""
    bounds = rec_a["boundaries"]
    if slot == "tb":
        return tuple(bounds)
    assert slot.startswith("tbk"), slot
    k = int(slot[3:])
    assert 1 <= k <= len(bounds) - 1, (slot, len(bounds))
    return (bounds[k - 1],)


def tb_payload_for_arm(
    tbank: dict,
    pair: BANK.Pair2162,
    slot: str,
    arm: str,
    donor_maps: dict[str, dict[str, str]],
    pairs_by_id: dict[str, BANK.Pair2162],
) -> tuple[torch.Tensor, str | None, dict]:
    """``((n_pos, L, H) payload, donor_pair_id, alignment meta)``.

    - steered: the pair's OWN B-side boundary states, paired by turn number.
    - shuffled: within-cell frozen donor's B states at the donor's own
      boundaries (counts match exactly — asserted), norm-matched row-wise to
      the recipient's B-state at the SAME (turn, layer).
    - crosstype: structure-aligned re-drawn donor; n_d >= n_r asserted
      (shorter donor FORBIDDEN); right-align-drop-earliest via ``align_right``,
      then the same per-(turn, layer) norm matching.
    """
    recs = tbank["per_context"]
    ra, rb = recs[pair.a], recs[pair.b]
    n_r = len(ra["boundaries"])
    assert len(rb["boundaries"]) == n_r, (pair.pair_id, "pair-side boundary count mismatch")
    if slot == "tb":
        sel = slice(0, n_r)
    else:
        k = int(slot[3:])
        sel = slice(k - 1, k)
    recipient = rb["tb"][sel]  # (n_pos, L, H) — the norm-match reference
    if arm == "steered":
        return recipient.clone(), None, {"alignment": "self", "alignment_dropped": 0}
    donor_map = donor_maps["shuffled" if arm == "shuffled" else "crosstype"]
    donor_id = donor_map[pair.pair_id]
    donor = pairs_by_id[donor_id]
    dstates = recs[donor.b]["tb"]  # (n_d, L, H)
    n_d = int(dstates.shape[0])
    if arm == "shuffled":
        assert n_d == n_r, (
            f"within-cell shuffled donor boundary count mismatch: {donor_id} "
            f"n_d={n_d} != n_r={n_r} (same cell must match by construction)"
        )
        aligned = dstates
        meta = {"alignment": "exact", "alignment_dropped": 0}
    else:
        assert n_d >= n_r, (
            f"shorter cross-type donor FORBIDDEN (plan §4.2 rule (b)): {donor_id} "
            f"n_d={n_d} < n_r={n_r}"
        )
        aligned = align_right(dstates, n_r)
        meta = {
            "alignment": "right_drop_earliest" if n_d > n_r else "exact",
            "alignment_dropped": n_d - n_r,
        }
    payload = BANK94.norm_match(aligned[sel], recipient)
    return payload, donor_id, meta


# ── multi-position injection-exactness gate (plan §7.2) ───────────────


def _tb_gate_spot_specs(pairs: list[BANK.Pair2162]) -> list[dict]:
    """12 spot cells spanning all 3 arms x depths d1/d3/d5 + sweep variants."""
    by_cell = BANK.pairs_by_cell(pairs)
    spec: list[tuple[str, str, str, int]] = [
        ("instr_format", "tb", "steered", 0),
        ("instr_format", "tb", "shuffled", 1),
        ("instr_format", "tb", "crosstype", 2),
        ("persona_prompted", "tb", "steered", 3),
        ("recency_instr_format_d3", "tb", "steered", 0),
        ("recency_instr_format_d3", "tb", "shuffled", 1),
        ("recency_persona_prompted_d3", "tb", "crosstype", 2),
        ("recency_instr_format_d5", "tb", "crosstype", 0),
        ("recency_persona_prompted_d5", "tb", "shuffled", 1),
        (CONTROL_CELL, "tb", "crosstype", 0),
        ("recency_instr_format_d5", "tbk2", "steered", 2),
        ("recency_persona_prompted_d3", "tbk1", "shuffled", 3),
    ]
    out = []
    for cell, slot, arm, k in spec:
        cell_pairs = sorted(by_cell[cell], key=lambda p: p.pair_id)
        out.append(
            {"cell": cell, "slot": slot, "arm": arm, "pair": cell_pairs[k % len(cell_pairs)]}
        )
    assert len(out) == 12, len(out)
    return out


def _forward_capture(model, ids, mask, layers) -> dict[int, torch.Tensor]:
    """One forward capturing each block's OUTPUT (post any installed edit hook —
    nn.Module runs forward hooks in registration order)."""
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _, _ = _resolve_decoder_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(layer_idx):
        def cap(_m, _i, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hidden.detach()

        return cap

    for layer_idx in layers:
        handles.append(blocks[layer_idx].register_forward_hook(_mk(layer_idx)))
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def run_tb_injection_gate(
    cfg: R.RunConfig,
    model,
    tok,
    tbank: dict,
    pairs: list[BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
) -> dict:
    """Multi-position injection exactness: installed == intended donor state at
    EVERY intended (position, layer) and nowhere else (plan §7.2).

    "Nowhere else" is propagation-compatible: (a) no EDIT applied elsewhere
    (``realized_edits`` vs the intended set), (b) state equality strictly
    UPSTREAM of the FIRST patched position (causally unaffected — patching
    earlier boundaries legitimately changes downstream unpatched positions at
    layers >= 2, which is the phenomenon under study, never gated).
    """
    contexts = BANK.build_contexts()
    recs = tbank["per_context"]
    pairs_by_id = {p.pair_id: p for p in pairs}
    spots = _tb_gate_spot_specs(pairs)
    rows_ids: list[list[int]] = []
    positions: list[tuple[int, ...]] = []
    payloads: list[torch.Tensor] = []
    metas: list[dict] = []
    for s in spots:
        pair = s["pair"]
        ra = recs[pair.a]
        ids = BANK.context_token_ids_2162(tok, contexts[pair.a])
        assert len(ids) == ra["ctx_len"], (pair.a, len(ids), ra["ctx_len"])
        pos = tb_positions(ra, s["slot"])
        payload, donor_id, align = tb_payload_for_arm(
            tbank, pair, s["slot"], s["arm"], donor_maps, pairs_by_id
        )
        rows_ids.append(ids)
        positions.append(pos)
        payloads.append(payload)
        metas.append({**s, "pair_id": pair.pair_id, "donor_pair_id": donor_id, **align})
    pad_id = tok.pad_token_id
    ids_t, mask_t = R._right_pad(rows_ids, pad_id, cfg.device)
    t_pad = int(ids_t.shape[1])
    layers = cfg.layers
    base = _forward_capture(model, ids_t, mask_t, layers)
    # Right-padded: absolute positions == unpadded positions (row_lengths=t_pad
    # zeroes the hook's left-pad offset — the margin_lnp convention).
    stack = R._arm_hook_all_layers(model, cfg, [t_pad] * len(rows_ids), positions, payloads, t_pad)
    try:
        hooked = _forward_capture(model, ids_t, mask_t, layers)
        realized = stack.realized_edits
    finally:
        stack.remove()
    assert realized is not None, "injection gate: no prefill applied"

    spot_rows: list[dict] = []
    n_failed = 0
    for b, (pos, payload, meta) in enumerate(zip(positions, payloads, metas, strict=True)):
        failures: list[str] = []
        min_cos, worst_ratio = 1.0, 1.0
        for j, p in enumerate(pos):
            for li, layer in enumerate(layers):
                inst = hooked[layer][b, p].float().cpu()
                want = payload[j, li]
                cos = float(safe_cosine(inst, want))
                ratio = float(inst.norm() / (want.norm() + 1e-12))
                min_cos = min(min_cos, cos)
                worst_ratio = ratio if abs(ratio - 1.0) > abs(worst_ratio - 1.0) else worst_ratio
                if cos < R.GATE_COS_MIN:
                    failures.append(f"cos@pos{p}/L{layer}={cos:.6f}")
                if not (R.GATE_NORM_RATIO_LO <= ratio <= R.GATE_NORM_RATIO_HI):
                    failures.append(f"norm@pos{p}/L{layer}={ratio:.6f}")
        # Telemetry: the realized edit set == the intended set, mode replace.
        row_realized = [r for r in realized if r["row"] == b]
        assert len(row_realized) == len(layers), (b, len(row_realized))
        for r in row_realized:
            if r["mode"] != "replace" or r["positions_unpadded"] != list(pos):
                failures.append(
                    f"telemetry@L{r['layer']}: mode={r['mode']} pos={r['positions_unpadded']}"
                )
        # Off-target: strictly upstream of the FIRST patched position, every layer.
        upstream_rel = 0.0
        first = min(pos)
        if first > 0:
            for layer in layers:
                h = hooked[layer][b, :first].float()
                z = base[layer][b, :first].float()
                rel = float((h - z).norm() / (z.norm() + 1e-9))
                upstream_rel = max(upstream_rel, rel)
            if upstream_rel > R.GATE_OFFTARGET_REL_MAX:
                failures.append(f"upstream_rel={upstream_rel:.2e}")
        if failures:
            n_failed += 1
        spot_rows.append(
            {
                **{k: v for k, v in meta.items() if k != "pair"},
                "positions": list(pos),
                "n_positions": len(pos),
                "min_cos": min_cos,
                "worst_norm_ratio": worst_ratio,
                "upstream_rel_max": upstream_rel,
                "realized_mode": REALIZED_MODE,
                "failures": failures,
                "passed": not failures,
            }
        )
    return {
        "criterion": "multi-position injection exactness (plan §7.2)",
        "bars": {
            "cos_min": R.GATE_COS_MIN,
            "norm_ratio": [R.GATE_NORM_RATIO_LO, R.GATE_NORM_RATIO_HI],
            "offtarget_rel_max": R.GATE_OFFTARGET_REL_MAX,
            "offtarget_scope": "strictly upstream of the FIRST patched position "
            "(propagation-compatible; downstream changes are the phenomenon)",
        },
        "realized_mode": REALIZED_MODE,
        "n_spots": len(spot_rows),
        "n_spots_failed": n_failed,
        "spots": spot_rows,
        "passed": n_failed == 0,
    }


# ── blocks ────────────────────────────────────────────────────────────


def enumerate_tb_blocks(pairs: list[BANK.Pair2162]) -> list[R.Block]:
    """45 blocks (plan §4.3): 7 joint-tb cells x 3 arms + 12 sweep variants x
    2 arms; d1 joint blocks ORDERED FIRST (wave-1 judge feed)."""
    by_cell = BANK.pairs_by_cell(pairs)

    def ids_of(cell: str) -> tuple[str, ...]:
        ids = tuple(p.pair_id for p in sorted(by_cell[cell], key=lambda p: p.pair_id))
        assert len(ids) == 36, (cell, len(ids))
        return ids

    d1 = [c for c in GRID_CELLS if DESIGNED_BOUNDARIES[c] == 1]
    rest = [c for c in GRID_CELLS if c not in d1]
    blocks: list[R.Block] = []
    for cell in d1 + rest:
        for arm in JOINT_ARMS:
            blocks.append(R.Block(cell, "tb", arm, ids_of(cell)))
    for cell in SWEEP_CELLS:
        for k in range(1, DESIGNED_BOUNDARIES[cell]):
            for arm in SWEEP_ARMS:
                blocks.append(R.Block(cell, f"tbk{k}", arm, ids_of(cell)))
    assert len(blocks) == 45, len(blocks)
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys), "duplicate tb block keys"
    return blocks


def tb_smoke_blocks(pairs: list[BANK.Pair2162]) -> list[R.Block]:
    """Plan §4.6: one joint tb slice + one sweep slice, 6 pairs each."""
    by_cell = BANK.pairs_by_cell(pairs)
    cell = "recency_instr_format_d3"
    ids = tuple(p.pair_id for p in sorted(by_cell[cell], key=lambda p: p.pair_id))[:SMOKE_PAIRS]
    return [R.Block(cell, "tb", "steered", ids), R.Block(cell, "tbk1", "shuffled", ids)]


def tb_regime_fingerprint(cfg: R.RunConfig, tb_sha: str) -> str:
    """Resume key over every output-affecting knob (round-tagged)."""
    import hashlib

    payload = json.dumps(
        {
            "round": ROUND,
            "tb_sha": tb_sha,
            "model_id": cfg.model_id,
            "tiny": cfg.tiny,
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": R.GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "seed_base": cfg.seed_base,
            "smoke": cfg.smoke,
            "bank_seed": BANK.SEED,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tb_config_sha(tb_config: dict) -> str:
    return R._sha256_bytes(json.dumps(tb_config, sort_keys=True, ensure_ascii=False).encode())


def _load_tb_bank(cfg: R.RunConfig) -> dict:
    path = tb_bank_dir(cfg) / "tb_bank.pt"
    assert path.exists(), f"{path} missing — run `--phase bank` first"
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_tb_config(cfg: R.RunConfig) -> dict:
    path = tb_bank_dir(cfg) / "tb_config.json"
    assert path.exists(), f"{path} missing — run `--phase bank` first"
    return json.loads(path.read_text())


# ── P1: bank phase ────────────────────────────────────────────────────


def phase_bank(cfg: R.RunConfig, parent_bank: Path | None, parent_f_cells: Path) -> int:
    logger.info("[phase=bank_tbmp] tiny=%s smoke=%s", cfg.tiny, cfg.smoke)
    if parent_bank is None or not parent_bank.exists():
        raise RuntimeError(
            f"--parent-bank missing or not found ({parent_bank}) — plan §4.5 DAG stages "
            "bank.json unconditionally at P1; stage "
            "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json and re-run"
        )
    bank_payload = json.loads(parent_bank.read_text())
    contexts = BANK.build_contexts()
    capture_ctx = {cid: c for cid, c in contexts.items() if c["cell"] in CAPTURE_CELLS}
    assert len(capture_ctx) == 432, len(capture_ctx)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}

    # §12 assumption 9 — rebuilt-vs-recorded context parity (runs BEFORE any
    # capture; identical in smoke and production).
    ctx_parity = parent_bank_context_parity(bank_payload, capture_ctx)
    logger.info(
        "[bank_tbmp] assumption-9 context parity %s (%d contexts, %d violations)",
        "PASS" if ctx_parity["passed"] else "FAIL",
        ctx_parity["n_contexts_checked"],
        ctx_parity["n_violations"],
    )

    model, tok = R.load_model_and_tokenizer(cfg)

    # G1 resolver sweep over ALL 432 capture contexts (full consumed grain —
    # identical in smoke and production, the §12 structural probe).
    resolved = resolve_boundaries(tok, capture_ctx)
    crosstype = build_crosstype_assignment(pairs)
    shuffled, parity_note = shuffled_assignment_with_parity(pairs, bank_payload, str(parent_bank))
    g1 = g1_resolver_report(resolved, pairs, crosstype, pairs_by_id)
    # §10 item (j) — G1(b) len_delta parity vs the parent's committed table.
    len_parity = g1b_len_delta_parity(resolved, pairs, parent_f_cells)
    logger.info(
        "[bank_tbmp] G1(b) len_delta parity %s (%d/%d pairs checked, %d violations)",
        "PASS" if len_parity["passed"] else "FAIL",
        len_parity["n_checked"],
        len_parity["n_capture_pairs"],
        len_parity["n_violations"],
    )
    g1["context_parity"] = ctx_parity
    g1["len_delta_parity"] = len_parity
    g1["passed"] = bool(g1["passed"] and ctx_parity["passed"] and len_parity["passed"])
    R._write_json_atomic(gates_dir(cfg) / "g1_resolver_report.json", g1)
    logger.info(
        "[bank_tbmp] G1 %s (%d violations; right-aligned=%d)",
        "PASS" if g1["passed"] else "FAIL",
        g1["n_violations"],
        g1["n_right_aligned"],
    )
    if not g1["passed"] and not cfg.force_past_halt_gates:
        logger.error(
            "[bank_tbmp] G1 HALT rc=%d — rig defect, fix before any rollout", RC_G1_RESOLVER
        )
        return RC_G1_RESOLVER

    tb_config = {
        "round": ROUND,
        "seed": BANK.SEED,
        "rng_stream": CROSSTYPE_STREAM_KEY,
        "designed_boundaries": DESIGNED_BOUNDARIES,
        "crosstype": crosstype,
        "shuffled": shuffled,
        "shuffled_parity": parity_note,
        "crosstype_pool": {k: list(v) for k, v in CROSSTYPE_POOL.items()},
        "alignment_rule": "right_drop_earliest (donor longer); shorter donor FORBIDDEN",
    }
    tb_sha = tb_config_sha(tb_config)
    regime_fp = tb_regime_fingerprint(cfg, tb_sha)
    done_rec = None
    done_path = cfg.manifest_dir / "bank_tbmp_done.json"
    if done_path.exists() and not cfg.force:
        done_rec = json.loads(done_path.read_text())
        if done_rec.get("regime_fp") != regime_fp:
            raise RuntimeError(
                f"bank_tbmp done-file regime_fp={done_rec.get('regime_fp')!r} != "
                f"{regime_fp!r} — refusing cross-regime resume (fresh --out-root)"
            )
        if (tb_bank_dir(cfg) / "tb_bank.pt").exists():
            logger.info("[bank_tbmp] already done for this regime — skipping capture")
            return R.RC_OK

    tbank = capture_tb_bank(cfg, model, tok, resolved)
    tbank["tb_sha"] = tb_sha
    tbank["repro"] = R._repro(cfg)
    R._write_json_atomic(tb_bank_dir(cfg) / "tb_config.json", tb_config)
    R._save_pt_atomic(tb_bank_dir(cfg) / "tb_bank.pt", tbank)

    donor_maps = {"shuffled": shuffled, "crosstype": crosstype}
    gate = run_tb_injection_gate(cfg, model, tok, tbank, pairs, donor_maps)
    gate["repro"] = R._repro(cfg)
    R._write_json_atomic(gates_dir(cfg) / "tb_injection_gate_report.json", gate)
    logger.info(
        "[bank_tbmp] injection gate %s (%d/%d spots failed)",
        "PASS" if gate["passed"] else "FAIL",
        gate["n_spots_failed"],
        gate["n_spots"],
    )
    if not gate["passed"] and not cfg.force_past_halt_gates:
        return R.RC_INJECTION_GATE
    R._write_json_atomic(
        done_path,
        {
            "regime_fp": regime_fp,
            "tb_sha": tb_sha,
            "n_contexts": len(resolved),
            "g1_passed": g1["passed"],
            "injection_gate_passed": gate["passed"],
            "repro": R._repro(cfg),
        },
    )
    logger.info("[phase=bank_tbmp_done]")
    return R.RC_OK


# ── P2: grid ──────────────────────────────────────────────────────────


def _tb_block_cells(
    tbank: dict,
    block: R.Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
) -> list[dict]:
    recs = tbank["per_context"]
    cells: list[dict] = []
    for pid in block.pair_ids:
        pair = pairs_by_id[pid]
        payload, donor_id, align = tb_payload_for_arm(
            tbank, pair, block.slot, block.arm, donor_maps, pairs_by_id
        )
        cells.append(
            {
                "pair_id": pid,
                "pair": pair,
                "context_a": pair.a,
                "context_b": pair.b,
                "positions": tb_positions(recs[pair.a], block.slot),
                "payload": payload,
                "donor_pair_id": donor_id,
                **align,
                "len_delta": int(recs[pair.b]["ctx_len"]) - int(recs[pair.a]["ctx_len"]),
            }
        )
    return cells


@torch.no_grad()
def tb_margin_lnp(cfg: R.RunConfig, model, tok, rows_spec: list[dict]) -> list[float]:
    """Multi-position fork of ``R.margin_lnp`` (hooked rows only): rows carry
    ``positions`` tuples + ``(n_pos, L, H)`` payloads."""
    pad_id = tok.pad_token_id
    out: list[float] = []
    for start in range(0, len(rows_spec), cfg.capture_batch):
        chunk = rows_spec[start : start + cfg.capture_batch]
        rows = [r["ctx_ids"] + r["item_ids"] for r in chunk]
        ids, mask = R._right_pad(rows, pad_id, cfg.device)
        t_pad = int(ids.shape[1])
        stack = R._arm_hook_all_layers(
            model,
            cfg,
            [t_pad] * len(rows),
            [tuple(r["positions"]) for r in chunk],
            [r["payload"] for r in chunk],
            t_pad,
        )
        try:
            logits = model(input_ids=ids, attention_mask=mask).logits
        finally:
            stack.remove()
        for b, r in enumerate(chunk):
            s = len(r["ctx_ids"])
            n_item = len(r["item_ids"])
            assert n_item >= 1, "empty pool item ids"
            lp = torch.log_softmax(logits[b, s - 1 : s + n_item - 1].float(), dim=-1)
            targets = ids[b, s : s + n_item]
            tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            out.append(float(tok_lp.mean()))
        del logits
    return out


def _tb_block_margin_rows(
    cfg: R.RunConfig,
    model,
    tok,
    block: R.Block,
    cells: list[dict],
    pools: dict[str, list[dict]],
    ctx_ids_cache: dict[str, list[int]],
) -> list[dict]:
    rows_spec: list[dict] = []
    meta: list[dict] = []
    out: list[dict] = []
    for cell in cells:
        pair: BANK.Pair2162 = cell["pair"]
        key = R.pool_key(pair)
        items = pools.get(key)
        if not items:
            out.append(
                {
                    "block_key": block.key,
                    "pair_id": pair.pair_id,
                    "arm": block.arm,
                    "pool_key": key,
                    "skipped": True,
                    "reason": "no pool for this value-pair (inherited parent behavior)",
                }
            )
            continue
        for idx, it in enumerate(items):
            item_ids = tok(it["text"], add_special_tokens=False)["input_ids"]
            assert item_ids, (key, idx, "pool item tokenized empty")
            rows_spec.append(
                {
                    "ctx_ids": ctx_ids_cache[cell["context_a"]],
                    "item_ids": item_ids,
                    "payload": cell["payload"],
                    "positions": cell["positions"],
                }
            )
            meta.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": block.arm,
                    "pair_id": pair.pair_id,
                    "donor_pair_id": cell["donor_pair_id"],
                    "pool_key": key,
                    "pool_idx": idx,
                    "pool_side": it["side"],
                    "n_pool_tokens": len(item_ids),
                }
            )
    if rows_spec:
        lnps = tb_margin_lnp(cfg, model, tok, rows_spec)
        for m, lnp in zip(meta, lnps, strict=True):
            out.append({**m, "lnp_mean": lnp, "skipped": False})
    return out


@torch.no_grad()
def run_tb_block(
    cfg: R.RunConfig,
    model,
    tok,
    tbank: dict,
    block: R.Block,
    pairs_by_id: dict[str, BANK.Pair2162],
    donor_maps: dict[str, dict[str, str]],
    contexts: dict[str, dict],
    ctx_ids_cache: dict[str, list[int]],
    regime_fp: str,
    pools: dict[str, list[dict]] | None,
    draws: int,
    write_done: bool = True,
) -> dict:
    """One tb block: K hooked temp-1.0 draws per pair (+ inline margin TF)."""
    cells = _tb_block_cells(tbank, block, pairs_by_id, donor_maps)

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK.context_token_ids_2162(tok, contexts[cid])
        return ctx_ids_cache[cid]

    texts_per_cell: list[list[str]] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        stack = R._arm_hook_all_layers(
            model,
            cfg,
            row_lengths,
            [c["positions"] for c in chunk],
            [c["payload"] for c in chunk],
            t_pad,
        )
        try:
            outs = R.generate_batch(
                model,
                tok,
                ctx_list,
                n=draws,
                hook=stack,
                max_new_tokens=cfg.max_new_tokens,
                temperature=R.GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK.render_context_2162,
                ids_fn=BANK.context_token_ids_2162,
            )
        finally:
            stack.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_per_cell.extend(list(o) for o in outs)
    assert len(texts_per_cell) == len(cells)

    variant = "tb" if block.slot == "tb" else "tb_k"
    k = None if block.slot == "tb" else int(block.slot[3:])
    rows_out: list[dict] = []
    for c, texts in zip(cells, texts_per_cell, strict=True):
        pair: BANK.Pair2162 = c["pair"]
        for i, text in enumerate(texts):
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            rows_out.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "variant": variant,
                    "k": k,
                    "arm": block.arm,
                    "pair_id": pair.pair_id,
                    "carrier": pair.carrier,
                    "value_a": pair.value_a,
                    "value_b": pair.value_b,
                    "context_a": pair.a,
                    "context_id": pair.a,  # audit-walker compat
                    "context_b": pair.b,
                    "positions": list(c["positions"]),
                    "n_positions": len(c["positions"]),
                    "alignment": c["alignment"],
                    "alignment_dropped": c["alignment_dropped"],
                    "realized_mode": REALIZED_MODE,
                    "donor_pair_id": c["donor_pair_id"],
                    "len_delta": c["len_delta"],
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": R.GRID_TEMPERATURE,
                    "n_completion_tokens": n_tok,
                    "cap_hit": R.cap_hit(n_tok, cfg.max_new_tokens),
                    "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                    "text": text,
                }
            )
    R._write_jsonl_atomic(cfg.rollouts_dir / f"shard_{block.slug}.jsonl", rows_out)
    margin_done = False
    if pools is not None:
        margin_rows = _tb_block_margin_rows(cfg, model, tok, block, cells, pools, ctx_ids_cache)
        R._write_jsonl_atomic(cfg.margin_dir / f"shard_{block.slug}.jsonl", margin_rows)
        if write_done:
            R._write_json_atomic(
                R.block_done_path(cfg.out_root, block, "margin_blocks_tbmp"),
                {
                    "key": block.key,
                    "regime_fp": regime_fp,
                    "n_rows": len(margin_rows),
                    "n_skipped": sum(1 for r in margin_rows if r.get("skipped")),
                    "repro": R._repro(cfg),
                },
            )
        margin_done = True
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_cells": block.n_pairs,
        "n_rows": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "margin_inline": margin_done,
        "repro": R._repro(cfg),
    }
    if write_done:
        R._write_json_atomic(R.block_done_path(cfg.out_root, block, "blocks_tbmp"), done)
    return done


def _upload_grid_increment(cfg: R.RunConfig, blocks: list[R.Block]) -> list[str]:
    slugs = [b.slug for b in blocks if (cfg.rollouts_dir / f"shard_{b.slug}.jsonl").exists()]
    if not slugs:
        return []
    return R._upload_dir(cfg, cfg.rollouts_dir, HF_TBMP_GRID, [f"shard_{s}.jsonl" for s in slugs])


def phase_grid(cfg: R.RunConfig) -> int:
    logger.info(
        "[phase=grid_tbmp] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    tbank = _load_tb_bank(cfg)
    tb_config = _load_tb_config(cfg)
    donor_maps = {"shuffled": tb_config["shuffled"], "crosstype": tb_config["crosstype"]}
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    regime_fp = tb_regime_fingerprint(cfg, str(tbank.get("tb_sha")))
    draws = R.SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws
    all_blocks = enumerate_tb_blocks(pairs)
    totals_all = R.grid_totals(all_blocks, cfg.grid_draws)
    blocks = tb_smoke_blocks(pairs) if cfg.smoke else all_blocks
    if cfg.pilot:
        blocks = blocks[:1]
    totals = R.grid_totals(blocks, draws)
    pools: dict[str, list[dict]] | None = None
    if cfg.pools_path is not None and cfg.pools_path.exists():
        pools = R.load_pools(cfg.pools_path)
        logger.info("[grid_tbmp] margin pools loaded: %d pools", len(pools))
    else:
        logger.info(
            "[grid_tbmp] no pools file (%s) — margins deferred to --phase margin", cfg.pools_path
        )
    R._write_json_atomic(
        cfg.manifest_dir / f"grid_plan_w{cfg.worker_index}.json",
        {
            "round": ROUND,
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "smoke": cfg.smoke,
            "pilot": cfg.pilot,
            "totals_full_grid": totals_all,
            "totals_this_run": totals,
            "queue": "shared claim-file queue (work-conserving); d1 blocks first",
            "margin_inline": pools is not None,
            "repro": R._repro(cfg),
        },
    )
    logger.info(
        "[grid_tbmp] full grid: %d blocks / %d rollouts; this run: %d blocks",
        totals_all["n_blocks"],
        totals_all["rollouts_total"],
        len(blocks),
    )
    model, tok = R.load_model_and_tokenizer(cfg)
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}
    ran_rollouts = 0
    ran_wall = 0.0
    n_run = 0
    uploaded: list[str] = []
    pending: list[R.Block] = []

    def run_one(block: R.Block) -> None:
        nonlocal ran_rollouts, ran_wall, n_run, uploaded, pending
        t0 = time.monotonic()
        rec = run_tb_block(
            cfg,
            model,
            tok,
            tbank,
            block,
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            regime_fp,
            pools,
            draws,
        )
        elapsed = time.monotonic() - t0
        ran_rollouts += rec["n_rows"]
        ran_wall += elapsed
        n_run += 1
        pending.append(block)
        logger.info(
            "[grid_tbmp] unit %d %s rows=%d cap_hit=%d elapsed=%.1fs",
            n_run,
            block.key,
            rec["n_rows"],
            rec["n_cap_hit"],
            elapsed,
        )
        if not cfg.pilot and cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_grid_increment(cfg, pending)
            pending.clear()

    if cfg.pilot:
        t0 = time.monotonic()
        rec = run_tb_block(
            cfg,
            model,
            tok,
            tbank,
            blocks[0],
            pairs_by_id,
            donor_maps,
            contexts,
            ctx_ids_cache,
            regime_fp + "-pilot",
            pools,
            draws,
            write_done=False,
        )
        ran_wall = time.monotonic() - t0
        ran_rollouts = rec["n_rows"]
        return R._enforce_pilot_gate(cfg, totals_all, ran_rollouts, ran_wall)

    stats = R.run_claim_queue(cfg, blocks, regime_fp, "blocks_tbmp", run_one)
    if pending:
        uploaded += _upload_grid_increment(cfg, pending)
        pending.clear()
    R._write_json_atomic(
        cfg.manifest_dir / f"grid_done_w{cfg.worker_index}.json",
        {
            "round": ROUND,
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "n_blocks_run": stats["ran"],
            "n_rollouts_run": ran_rollouts,
            "wall_s": ran_wall,
            "queue_waits": stats["waits"],
            "uploads": uploaded,
            "repro": R._repro(cfg),
        },
    )
    logger.info(
        "[phase=grid_tbmp_done] worker=%d blocks_run=%d rollouts=%d",
        cfg.worker_index,
        stats["ran"],
        ran_rollouts,
    )
    return R.RC_OK


# ── P2m: margin catch-up ──────────────────────────────────────────────


def phase_margin(cfg: R.RunConfig) -> int:
    logger.info(
        "[phase=margin_tbmp] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    assert cfg.pools_path is not None and cfg.pools_path.exists(), (
        f"--pools file required for --phase margin (got {cfg.pools_path}) — the parent's "
        "judge-built pools are staged by the orchestrator (eval_results/issue_2162/judge/pools.json)"
    )
    pools = R.load_pools(cfg.pools_path)
    tbank = _load_tb_bank(cfg)
    tb_config = _load_tb_config(cfg)
    donor_maps = {"shuffled": tb_config["shuffled"], "crosstype": tb_config["crosstype"]}
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    regime_fp = tb_regime_fingerprint(cfg, str(tbank.get("tb_sha")))
    model, tok = R.load_model_and_tokenizer(cfg)
    contexts = BANK.build_contexts()
    ctx_ids_cache: dict[str, list[int]] = {}
    blocks = tb_smoke_blocks(pairs) if cfg.smoke else enumerate_tb_blocks(pairs)

    def run_one(block: R.Block) -> None:
        cells = _tb_block_cells(tbank, block, pairs_by_id, donor_maps)
        for c in cells:
            if c["context_a"] not in ctx_ids_cache:
                ctx_ids_cache[c["context_a"]] = BANK.context_token_ids_2162(
                    tok, contexts[c["context_a"]]
                )
        t0 = time.monotonic()
        margin_rows = _tb_block_margin_rows(cfg, model, tok, block, cells, pools, ctx_ids_cache)
        R._write_jsonl_atomic(cfg.margin_dir / f"shard_{block.slug}.jsonl", margin_rows)
        R._write_json_atomic(
            R.block_done_path(cfg.out_root, block, "margin_blocks_tbmp"),
            {
                "key": block.key,
                "regime_fp": regime_fp,
                "n_rows": len(margin_rows),
                "n_skipped": sum(1 for r in margin_rows if r.get("skipped")),
                "repro": R._repro(cfg),
            },
        )
        logger.info(
            "[margin_tbmp] unit %s rows=%d elapsed=%.1fs",
            block.key,
            len(margin_rows),
            time.monotonic() - t0,
        )

    stats = R.run_claim_queue(cfg, blocks, regime_fp, "margin_blocks_tbmp", run_one)
    logger.info("[phase=margin_tbmp_done] worker=%d blocks_run=%d", cfg.worker_index, stats["ran"])
    return R.RC_OK


# ── P3: upload + sentinel ─────────────────────────────────────────────


def _margin_state(cfg: R.RunConfig) -> dict:
    pairs = BANK.build_pairs()
    blocks = tb_smoke_blocks(pairs) if cfg.smoke else enumerate_tb_blocks(pairs)
    blocks_done = sum(
        1 for b in blocks if R.block_done_path(cfg.out_root, b, "margin_blocks_tbmp").exists()
    )
    deferred = blocks_done < len(blocks)
    state: dict = {
        "margin_deferred": deferred,
        "margin_blocks_done": blocks_done,
        "margin_blocks_expected": len(blocks),
    }
    if deferred:
        state["margin_deferred_recipe"] = (
            "stage eval_results/issue_2162/judge/pools.json to the pod, then: "
            "issue2162_tbmp_dispatch.sh margin && issue2162_tbmp_dispatch.sh upload (1x H100)"
        )
    return state


def _sentinel_payload(cfg: R.RunConfig, uploaded: dict[str, list[str]]) -> dict:
    n_grid_shards = len(list(cfg.rollouts_dir.glob("shard_*.jsonl")))
    n_margin_shards = len(list(cfg.margin_dir.glob("*.jsonl")))
    g1_path = gates_dir(cfg) / "g1_resolver_report.json"
    gate_path = gates_dir(cfg) / "tb_injection_gate_report.json"
    g1 = json.loads(g1_path.read_text()) if g1_path.exists() else {}
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    block_done_recs = sorted((cfg.manifest_dir / "blocks_tbmp").glob("*.done.json"))
    cap_hits, rows_total = 0, 0
    for done in block_done_recs:
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        rows_total += int(rec.get("n_rows", 0))
    margin_state = _margin_state(cfg)
    return {
        **margin_state,
        "deferred_leg": not block_done_recs,
        "eval_numbers": {
            "grid_shards": n_grid_shards,
            "margin_shards": n_margin_shards,
            "grid_rollouts_persisted": rows_total,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / rows_total) if rows_total else 0.0,
            "g1_passed": bool(g1.get("passed")),
            "g1_violations": int(g1.get("n_violations", 0)),
            "injection_gate_passed": bool(gate.get("passed")),
            "injection_gate_spots_failed": int(gate.get("n_spots_failed", 0)),
        },
        "eval_paths": sorted(
            {
                str(tb_bank_dir(cfg) / "tb_bank.pt"),
                str(tb_bank_dir(cfg) / "tb_config.json"),
                str(g1_path),
                str(gate_path),
                str(cfg.rollouts_dir),
                str(cfg.margin_dir),
            }
        ),
        "reproducibility_card": {
            **R._repro(cfg),
            "round": ROUND,
            "seed_base": cfg.seed_base,
            "bank_seed": BANK.SEED,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": R.GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "gen_batch": cfg.gen_batch,
            "num_workers": cfg.num_workers,
            "realized_mode": REALIZED_MODE,
        },
        "wandb_url": None,
        "hf_hub_url": (
            f"https://huggingface.co/datasets/{R.HF_DATA_WRITE_REPO}/tree/main/{HF_TBMP}"
        ),
        "worktree_path": str(R.REPO_ROOT),
        "final_commit_sha": R._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "cap-hit telemetry is derived from the re-tokenized completion length "
            "(generate_batch returns decoded text only)",
            "no per-rollout answer-state capture this round (plan §4.4 has no V_a-based "
            "tb metric; the judge DVs + TF margin are the registered reads)",
            "margin artifacts persist as JSONL under margin/ (plan §6.5 says .pt) — "
            "text-first upload policy; content is per-pair scalar margins, no tensors",
            "plan §12 assumption 9 verification recipe superseded: the staged parent "
            "bank.json records NO ctx_len/prefix_end (fresh-HF probe 2026-08-14, "
            "2,383,040 bytes: 0 occurrences of either key; per-context keys are exactly "
            "{carrier, cell, history, id, system, user, value_id}), so the literal recipe "
            "cannot execute; realized triple = (i) producer-domain frozen_gen_sha256 "
            "recompute == banked pin, (ii) exact per-context payload equality on "
            "{cell, value_id, carrier, system, history, user} for all 432 capture "
            "contexts, (iii) G1(b) re-rendered per-pair ctx_len delta == parent committed "
            "f_cells.jsonl len_delta, 432/432 required (gates/g1_resolver_report.json). "
            "Strictly stronger at the TEXT level; incomparable at absolute token geometry "
            "— the residual drift class (a uniform render shift identical across both pair "
            "members preserving per-cell boundary counts) is not load-bearing: this round "
            "re-captures injection payloads at re-rendered positions and consumes nothing "
            "absolute-position-indexed from the parent.",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }


def phase_upload(cfg: R.RunConfig) -> int:
    logger.info("[phase=upload_tbmp]")
    uploaded: dict[str, list[str]] = {}
    uploaded["tb_bank"] = R._upload_dir(
        cfg, tb_bank_dir(cfg), f"{HF_TBMP}/tb_bank", ["*.pt", "*.json"]
    )
    uploaded["gates"] = R._upload_dir(cfg, gates_dir(cfg), f"{HF_TBMP}/gates", ["*.json"])
    uploaded["grid_text"] = R._upload_dir(cfg, cfg.rollouts_dir, HF_TBMP_GRID, ["shard_*.jsonl"])
    uploaded["margin"] = R._upload_dir(cfg, cfg.margin_dir, f"{HF_TBMP}/margin", ["*.jsonl"])
    uploaded["manifests"] = R._upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_TBMP}/manifests",
        ["*.json", "blocks_tbmp/*.done.json", "margin_blocks_tbmp/*.done.json"],
    )
    # Out-root TOP-LEVEL residue (#2187): pilot_gate_report.json + any other
    # root-level JSON rides its own glob so no file escapes every upload glob.
    # UPLOAD_PREFIX_EXEMPT: issue-2162-scoped follow-up driver — HF_TBMP IS this issue's canonical tbmp prefix, not a reusable-core fallback
    uploaded["outroot_json"] = R._upload_dir(
        cfg, cfg.out_root, f"{HF_TBMP}/manifests/outroot", ["*.json"]
    )
    payload = _sentinel_payload(cfg, uploaded)
    if payload["margin_deferred"]:
        logger.warning(
            "[upload_tbmp] margin DEFERRED (blocks %d/%d) — sentinel records "
            "margin_deferred=true + recipe; teardown proceeds",
            payload["margin_blocks_done"],
            payload["margin_blocks_expected"],
        )
    R._write_json_atomic(cfg.out_root / "manifests" / "upload_done.json", payload)
    sentinel = cfg.log_dir / (SENTINEL_NAME_SMOKE if cfg.smoke else SENTINEL_NAME)
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:smoke-result" if cfg.smoke else "epm:results",
        "version": 1,
        "note": payload,
    }
    R._write_json_atomic(sentinel, body)
    logger.info("[upload_tbmp] sentinel written: %s", sentinel)
    logger.info("[phase=upload_tbmp_done]")
    return R.RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2162 tbmp driver (bank / grid / margin / upload)."
    )
    ap.add_argument("--phase", choices=("bank", "grid", "margin", "upload"))
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + argparse-attribute completeness, exit 0",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=R.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=R.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=R.MAX_NEW_TOKENS)
    ap.add_argument("--anchor-draws", type=int, default=R.ANCHOR_DRAWS)  # RunConfig parity
    ap.add_argument("--grid-draws", type=int, default=R.GRID_DRAWS)
    ap.add_argument("--seed-base", type=int, default=R.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="plan §4.6 smoke slice")
    ap.add_argument("--pilot", action="store_true", help="grid: timing pilot only")
    ap.add_argument("--force", action="store_true", help="re-run a completed bank phase")
    ap.add_argument("--force-past-halt-gates", action="store_true")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument("--upload-every", type=int, default=10)
    ap.add_argument(
        "--pools",
        type=Path,
        default=None,
        help="parent judge-built pools JSON; grid computes margins inline when present",
    )
    ap.add_argument(
        "--parent-bank",
        type=Path,
        default=None,
        help="parent bank.json (staged from HF) — REQUIRED for the bank phase: "
        "shuffled-assignment + assumption-9 context parity",
    )
    ap.add_argument(
        "--parent-f-cells",
        type=Path,
        default=R.REPO_ROOT / "eval_results" / "issue_2162" / "f_metrics" / "f_cells.jsonl",
        help="parent committed f_cells.jsonl — G1(b) len_delta rebuilt-vs-recorded parity",
    )
    ap.add_argument("--planned-wall-h", type=float, default=PLANNED_GRID_WALL_H)
    ap.add_argument("--gpu-hours-budgeted", type=float, default=7.0)
    return ap.parse_args(argv)


def _import_check() -> None:
    """Resolve EVERY deferred import + the argparse-attribute completeness assert."""
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    import transformers  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        verify_repo_paths_uploaded,
    )

    assert callable(align_right)
    assert BANK.frozen_gen_path().exists(), (
        f"{BANK.FROZEN_GEN_FILENAME} missing — run scripts/issue2162_genfreeze.py first"
    )
    n_pairs = len(BANK.build_pairs())
    assert n_pairs == 1404, n_pairs
    assert set(DESIGNED_BOUNDARIES) == set(CAPTURE_CELLS)
    assert set(CROSSTYPE_POOL) == set(GRID_CELLS)
    blocks = enumerate_tb_blocks(BANK.build_pairs())
    assert len(blocks) == 45, len(blocks)
    assert_args_attributes_defined(__file__)
    print("[import-check] OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return R.RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = R.build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_id is not None:
        logger.info(
            "[env] --gpu-id=%s CUDA_VISIBLE_DEVICES=%s",
            args.gpu_id,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    if cfg.phase == "bank":
        return phase_bank(cfg, args.parent_bank, args.parent_f_cells)
    if cfg.phase == "grid":
        return phase_grid(cfg)
    if cfg.phase == "margin":
        return phase_margin(cfg)
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
