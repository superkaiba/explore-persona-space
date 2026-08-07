"""Issue #2094 — gpu2_mq_replacement_prefix pod driver (same-issue follow-up).

The parent grid's 15 matched-query pairs pair the SAME query under DIFFERENT
prefixes; 5 of them — mq--bare__qK--conv__qK, K=1..5 — have judge anchor
separation |sep| < 0.5 on the prefix rubric kind (committed
``eval_results/issue_2094/f_metrics/anchors.jsonl``), so the |sep| >= 0.5
restriction guts matched-query informative n. Diagnosis (committed judge
scores): the CONVERSATION prefix is the problem — conv-generated anchor draws
score fp-conv 0.2/100 vs fp-bare 74.9/100 (the register does not carry into
answers), while the same queries separate fine against persona (0.81-1.87).

This driver replaces the conversation prefix with ``conv2``
(``issue2094_gpu2_bank.py`` — same construction method, content chosen to
carry) and re-runs the parent grid's matched-query arm on the 5 re-formed
pairs, behind an ANCHOR-VALIDATION GATE (the round's kill gate):

- ``--phase diagnose`` (VM, CPU-only): compute + write
  ``eval_results/issue_2094/f_metrics/gpu2/diagnosis.json`` (committed).
- ``--run`` (pod, ONE driver run with the gate inline):
  1. diagnose-check (committed diagnosis re-verified from the committed
     anchors file; refuses on a reframe-needed verdict),
  2. stage (parent vc_bank.pt for capture parity + parent anchors.jsonl for
     the gate floor draws),
  3. bank: fresh all-layer capture of the 20 contexts (15 parent + 5 conv2);
     capture-parity gate on the 15 SHARED contexts vs the parent bank (the
     fu2 two-bar recipe; rc=23 designed halt),
  4. anchors: 5 conv2 contexts x K=10 unpatched temp-1.0 rollouts at
     max_new_tokens=2048 + both-pooling V_a (the parent anchor protocol),
  5. gate: judge coherence + fp-bare + fp-conv2 on the floor draws (parent
     bare anchors, re-judged fresh so both sides share one wave instrument)
     and the ceiling draws (the new conv2 anchors) via the production judge
     machinery (``issue2094_judge.run_wave``); separation per the parent
     ``anchor_pair_stats`` convention (coherent draws only, rule-9 drops);
     PASS iff >= 4/5 re-formed pairs reach |sep| >= 0.5 — FAIL writes the
     gate report, uploads, writes the sentinel, exits rc=24 (NO grid spend),
  6. grid: the parent grid's matched-query headline families over the 5 new
     pairs — ce slot, all 30 layer variants, Type-A, 5 doses, steered +
     shuffled-donor null (fresh seeded donor assignment from the PARENT
     matched-query pool, recorded per cell) — ``R.run_block`` verbatim at
     max_new_tokens=2048 with per-cell cap-hit,
  7. upload + sentinel (``/workspace/logs/issue-2094-gpu2-results.json``).

Grid arithmetic (pinned in tests/test_issue2094_gpu2.py): 30 variants x 5
doses = 150 families = 300 blocks; 5 pairs/block => 750 steered + 750 null =
1,500 cells (fu1 ran 2,100).

VM-side judging of the grid rollouts needs the ADDITIVE ``--gpu2`` judge
extension (``issue2094_judge.py --phase waves --gpu2 --rollouts-dir ...``):
the shards are grid-row-shaped; the flag unions the 5 re-formed pairs into
``pair_index()`` and the fp-conv2 rubric into ``rubric_registry()``.

Persistence (fail-loud, one ``upload_folder`` commit per prefix, #1773
all-prefixes enumeration in the sentinel):
  rollout text  -> issue2094_singlepos/raw_completions/gpu2_mq_replacement/rollouts
  anchors text  -> issue2094_singlepos/raw_completions/gpu2_mq_replacement/anchors
  judge gate    -> issue2094_singlepos/raw_completions/gpu2_mq_replacement/judge_gate
  V_a captures  -> issue2094_singlepos/analysis_tensors/gpu2_mq_replacement/va
  anchors V_a   -> issue2094_singlepos/analysis_tensors/gpu2_mq_replacement/anchors
  gpu2 V bank   -> issue2094_singlepos/analysis_tensors/gpu2_mq_replacement/bank
  manifests     -> issue2094_singlepos/analysis_tensors/gpu2_mq_replacement/manifests

Launch (pod-side; no VM thread-cap prefix — dedicated GPU box):
``CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2094_gpu2.py --run``
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
import issue2094_fu2 as F2  # noqa: E402
import issue2094_gpu2_bank as G2B  # noqa: E402
import issue2094_judge as J  # noqa: E402
import issue2094_run as R  # noqa: E402

logger = logging.getLogger("issue2094_gpu2")

GPU2_LABEL = "gpu2_mq_replacement"
SENTINEL_NAME = "issue-2094-gpu2-results.json"
HF_GPU2_TEXT = f"{R.HF_PREFIX}/raw_completions/{GPU2_LABEL}"
HF_GPU2_TENSORS = f"{R.HF_PREFIX}/analysis_tensors/{GPU2_LABEL}"

GPU2_MAX_NEW_TOKENS = 2048  # fu1's raised cap, from the start (brief requirement)
MIN_ABS_SEPARATION = 0.5  # the wellsep restriction floor (analyzer convention)
GATE_MIN_PASSING = 4  # gate PASS iff >= 4 of the 5 re-formed pairs clear the floor
JUDGE_MODES = ("live", "mock-pass", "mock-fail")
# Version token folded into the resume regime: a change to the gpu2 machinery
# (gate arithmetic / slot scope / judge wiring) is a NEW regime, never a
# silent resume (#722 r3).
GPU2_REGIME_TOKEN = "gpu2_mq_replacement_v1"

RC_OK = 0
RC_PARITY_GATE = 23  # capture-parity designed halt (fu2 precedent)
RC_ANCHOR_GATE = 24  # anchor-separation kill gate (designed halt, #1415 routing)
RC_REFRAME = 25  # diagnosis refutes the conv-prefix premise — stop, reframe

# Grid arithmetic (pinned in tests/test_issue2094_gpu2.py): ce x 30 layer
# variants x 5 doses x Type-A over the 5 re-formed pairs, steered + null.
EXPECTED_GPU2_TOTALS = {
    "n_families": 150,
    "n_blocks": 300,
    "cells_steered": 750,
    "cells_null": 750,
    "cells_total": 1500,
}

GATE_RUBRIC_IDS = ("fp-bare", "fp-conv2")
GATE_WAVE_SUFFIX = "gpu2anchors"


# ── diagnosis (step 1; pure derivations pinned in tests) ────────────────


def weak_matched_query_rows(fmetrics_anchors: Path, min_sep: float = MIN_ABS_SEPARATION) -> list:
    """Matched-query anchor rows with |separation| < ``min_sep`` (or None)."""
    rows = [json.loads(line) for line in fmetrics_anchors.open(encoding="utf-8") if line.strip()]
    mq = [r for r in rows if r["setting"] == "matched_query"]
    assert mq, f"no matched-query anchor rows in {fmetrics_anchors}"
    return [r for r in mq if r["separation"] is None or abs(r["separation"]) < min_sep]


def _anchor_score_means(scores_dir: Path) -> dict[str, dict[str, float]]:
    """Mean judge score per (generation prefix, rubric) over anchor draws."""
    sums: dict[tuple[str, str], list[float]] = {}
    for f in sorted(scores_dir.glob("fp-*.anchors.scores.jsonl")):
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "anchor" or r.get("score") is None:
                continue
            key = (r["context_id"].split("__")[0], r["rubric_id"])
            sums.setdefault(key, []).append(float(r["score"]))
    out: dict[str, dict[str, float]] = {}
    for (prefix, rid), vals in sorted(sums.items()):
        out.setdefault(prefix, {})[rid] = float(sum(vals) / len(vals))
    return out


def run_diagnosis(fmetrics_anchors: Path, scores_dir: Path | None) -> dict:
    """The step-1 diagnosis: WHICH matched-query pairs are weakly separated,
    and is the weak separation attributable to the conversation prefix?

    Attribution logic (all three legs must hold for ``conv-prefix-attributable``):
    (a) the weak set is EXACTLY the 5 bare-vs-conv pairs;
    (b) the same queries separate in their bare-vs-persona pairs (>= floor),
        so the queries are not the problem;
    (c) where judge scores are available: conv-generated anchor draws score
        LOW on fp-conv and HIGH on fp-bare — the conv register does not carry
        into answers, i.e. the conversation prefix's behavioral target is
        judge-indistinguishable from bare's.
    """
    rows = [json.loads(line) for line in fmetrics_anchors.open(encoding="utf-8") if line.strip()]
    weak = weak_matched_query_rows(fmetrics_anchors)
    weak_ids = sorted(r["pair_id"] for r in weak)
    expected = sorted(G2B.WEAK_PAIR_IDS)
    by_pair_kind = {(r["pair_id"], r["kind"]): r for r in rows}

    query_control = {}
    for q in BANK.QUERY_ORDER:
        row = by_pair_kind.get((f"mq--bare__{q}--persona__{q}", "prefix"))
        query_control[q] = None if row is None else row["separation"]
    qc_vals = [v for v in query_control.values() if v is not None]
    queries_separate = bool(qc_vals) and min(abs(v) for v in qc_vals) >= MIN_ABS_SEPARATION

    attribution = None
    register_carries = None
    if scores_dir is not None and scores_dir.is_dir():
        attribution = _anchor_score_means(scores_dir)
        conv = attribution.get("conv", {})
        register_carries = not (
            conv.get("fp-conv") is not None
            and conv.get("fp-bare") is not None
            and conv["fp-conv"] < 20.0
            and conv["fp-bare"] > 50.0
        )

    conv_attributable = (
        weak_ids == expected and queries_separate and register_carries in (False, None)
    )
    return {
        "weak_pairs": [
            {k: r[k] for k in ("pair_id", "kind", "separation", "floor", "ceiling")} for r in weak
        ],
        "weak_pair_ids": weak_ids,
        "expected_weak_pair_ids": expected,
        "min_abs_separation": MIN_ABS_SEPARATION,
        "query_control_bare_vs_persona_sep": query_control,
        "queries_separate_against_persona": queries_separate,
        "anchor_score_means_by_gen_prefix": attribution,
        "conv_register_carries_into_answers": register_carries,
        "verdict": ("conv-prefix-attributable" if conv_attributable else "reframe-needed"),
        "replacement": {
            "prefix": G2B.CONV2_PREFIX,
            "descriptor": G2B.CONV2_DESCRIPTOR,
            "reformed_pair_ids": [p.pair_id for p in G2B.build_gpu2_pairs()],
        },
        "repro": _diag_repro(),
    }


def _diag_repro() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = as_metadata_dict(git_provenance())
    meta["script"] = "scripts/issue2094_gpu2.py"
    return meta


def phase_diagnose(fmetrics_dir: Path, scores_dir: Path | None) -> int:
    """VM-side ``--phase diagnose``: compute + write the committed diagnosis."""
    logger.info("[phase=gpu2_diagnose]")
    diag = run_diagnosis(fmetrics_dir / "anchors.jsonl", scores_dir)
    out = fmetrics_dir / "gpu2" / "diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(out, diag)
    logger.info(
        "[diagnose] weak=%d verdict=%s -> %s", len(diag["weak_pair_ids"]), diag["verdict"], out
    )
    print(f"[phase=gpu2_diagnose_done] verdict={diag['verdict']}", flush=True)
    return RC_OK if diag["verdict"] == "conv-prefix-attributable" else RC_REFRAME


def assert_diagnosis_check(repo_root: Path) -> None:
    """Pod-side light check: the committed diagnosis carries the attributable
    verdict, and the weak set recomputed from the committed anchors file still
    matches the pinned expectation (both files ride the issue branch)."""
    fm = repo_root / "eval_results/issue_2094/f_metrics"
    diag_path = fm / "gpu2" / "diagnosis.json"
    assert diag_path.is_file(), (
        f"{diag_path} missing — run `--phase diagnose` on the VM and commit its output first"
    )
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    if diag["verdict"] != "conv-prefix-attributable":
        raise RuntimeError(
            f"committed diagnosis verdict is {diag['verdict']!r} — the conv-prefix premise "
            "does not hold; refusing to run the replacement grid (reframe needed)"
        )
    weak_ids = sorted(r["pair_id"] for r in weak_matched_query_rows(fm / "anchors.jsonl"))
    assert weak_ids == sorted(G2B.WEAK_PAIR_IDS), (weak_ids, G2B.WEAK_PAIR_IDS)


# ── grid enumeration (CPU-only; pinned in tests) ────────────────────────


def enumerate_gpu2_families(n_layers: int) -> list[tuple[R.Block, R.Block]]:
    """150 (steered, null) families = 300 blocks: ce x 30 layer variants x
    DOSES_A x Type-A over the 5 re-formed pairs (the parent matched-query
    headline families; pe/controls/Type-B are out of the round's tight scope)."""
    pair_ids = tuple(p.pair_id for p in G2B.build_gpu2_pairs())
    families: list[tuple[R.Block, R.Block]] = []
    for variant in R.layer_variant_names(n_layers):
        for dose in R.DOSES_A:
            families.append(
                (
                    R.Block("ce", variant, dose, "A", "steered", pair_ids),
                    R.Block("ce", variant, dose, "A", "null", pair_ids),
                )
            )
    keys = [b.key for fam in families for b in fam]
    assert len(set(keys)) == len(keys), "duplicate gpu2 block keys"
    return families


def smoke_family_spec(n_layers: int) -> tuple[tuple[str, str], ...]:
    """(layer_variant, dose) smoke slice: single-layer add + replace (the
    state-kind donor walk), a second single-layer dose, joint_mid, and
    joint_all x replace — one family per arm class, BOTH arms each."""
    variants = R.layer_variant_names(n_layers)
    mid = variants[n_layers // 2]
    last = variants[n_layers - 1]
    return (
        (mid, "a1"),
        (mid, "replace"),
        (last, "a0.5"),
        ("joint_mid", "a2"),
        ("joint_all", "replace"),
    )


def slice_gpu2_smoke(
    families: list[tuple[R.Block, R.Block]], n_layers: int
) -> list[tuple[R.Block, R.Block]]:
    keep = set(smoke_family_spec(n_layers))
    out = [fam for fam in families if (fam[0].layer_variant, fam[0].dose) in keep]
    assert len(out) == len(keep), (len(out), len(keep))
    return out


def gpu2_regime_fingerprint(cfg: R.RunConfig, judge_mode: str, draws: int) -> str:
    """Parent regime fingerprint + EVERY gpu2 output-affecting knob (#722 r3):
    the extension manifest (conv2 content + pairs + donor map + seed), the
    gate constants, the judge instrument, and the judge mode (a mock-judged
    smoke regime can never satisfy a live resume)."""
    _, bank_sha = R.bank_manifest_and_sha()
    base = R.regime_fingerprint(cfg, bank_sha)
    payload = json.dumps(
        {
            "base": base,
            "token": GPU2_REGIME_TOKEN,
            "gpu2_sha": G2B.gpu2_manifest_sha(),
            "judge_mode": judge_mode,
            "anchor_draws": draws,
            "judge_model": J.DEFAULT_JUDGE_MODEL,
            "judge_max_tokens": J.DEFAULT_JUDGE_MAX_TOKENS,
            "min_abs_separation": MIN_ABS_SEPARATION,
            "gate_min_passing": GATE_MIN_PASSING,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── paths ────────────────────────────────────────────────────────────────


@dataclass
class GPU2Paths:
    """gpu2-side output tree (rollouts / va / manifests ride RunConfig's own
    dirs so ``R.run_block`` + the parent upload seams are reused verbatim)."""

    out_root: Path

    @property
    def bank_dir(self) -> Path:
        return self.out_root / "gpu2_bank"

    @property
    def bank_path(self) -> Path:
        return self.bank_dir / "gpu2_bank.pt"

    @property
    def bank_done(self) -> Path:
        return self.bank_dir / "gpu2_bank_done.json"

    @property
    def parity_path(self) -> Path:
        return self.bank_dir / "gpu2_bank_parity.json"

    @property
    def parent_anchors_file(self) -> Path:
        return self.out_root / "parent_anchors" / "anchors.jsonl"

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / "anchors_gpu2"

    @property
    def anchors_file(self) -> Path:
        return self.anchors_dir / "anchors_gpu2.jsonl"

    @property
    def anchors_va(self) -> Path:
        return self.anchors_dir / "va_anchors_gpu2.pt"

    @property
    def anchors_done(self) -> Path:
        return self.anchors_dir / "anchors_gpu2_done.json"

    @property
    def judge_root(self) -> Path:
        return self.out_root / "judge_gate"

    @property
    def judge_cache(self) -> Path:
        return self.out_root / "judge_gate_cache"

    @property
    def gate_report(self) -> Path:
        return self.out_root / "manifests" / "gate_report.json"


def _regime_checked_done(path: Path, regime_fp: str, what: str) -> dict | None:
    """Missing -> None (run the phase); regime mismatch -> HARD refusal."""
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("regime_fp") != regime_fp:
        raise RuntimeError(
            f"{what} done-record carries regime_fp={rec.get('regime_fp')!r} but this run's "
            f"regime_fp={regime_fp!r} — refusing to resume across regimes (quarantine or "
            "use a fresh --out-root)"
        )
    return rec


# ── stage (production inputs) ───────────────────────────────────────────


def stage_parent_anchors(paths: GPU2Paths, hf_revision: str | None) -> None:
    """Stage the parent run's anchor rollout text (the gate floor draws)."""
    from explore_persona_space.orchestrate import hub

    if paths.parent_anchors_file.exists():
        logger.info("[stage] parent anchors.jsonl already present — skipping")
        return
    hub.stage_hub_file(
        R.HF_DATA_REPO,
        f"{R.HF_PREFIX}/raw_completions/anchors/anchors.jsonl",
        paths.parent_anchors_file,
        repo_type="dataset",
        revision=hf_revision,
    )
    logger.info("[stage] parent anchors.jsonl staged")


def load_floor_rows(paths: GPU2Paths, draws: int) -> list[dict]:
    """The gate FLOOR rows: the parent's bare-context anchor draws (staged),
    capped at ``draws`` per context (the parent ran K=10)."""
    rows = [
        json.loads(line)
        for line in paths.parent_anchors_file.open(encoding="utf-8")
        if line.strip()
    ]
    floor = [r for r in rows if r["context_id"].split("__")[0] == "bare" and r["draw"] < draws]
    per_ctx = {r["context_id"] for r in floor}
    assert per_ctx == {f"bare__{q}" for q in BANK.QUERY_ORDER}, per_ctx
    return floor


# ── bank (fresh 20-context capture + parity gate) ───────────────────────


@torch.no_grad()
def capture_gpu2_bank(cfg: R.RunConfig, model, tok, donor_map: dict[str, str]) -> dict:
    """Fresh all-layer capture over the 20 extended contexts, parent record
    shape (``q_span``/``v_pe``) so ``R.run_block`` + the parity gate read the
    records unchanged. Type-A only — no centroids (the round runs no Type-B)."""
    contexts = G2B.build_extended_contexts()
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
            ctx_len = len(ctx_ids[cid])
            pe = prefix_ends[cid]
            nq = ctx_len - pe
            assert nq >= R._QSPAN_MIN_POSITIONS, (cid, ctx_len, pe, nq)
            span = torch.stack([captured[layer][j, pe:ctx_len] for layer in layers], dim=1)
            v_pe = torch.stack([captured[layer][j, pe - 1] for layer in layers])
            assert span.shape == (nq, len(layers), cfg.hidden), span.shape
            records[cid] = {
                "context_id": cid,
                "prefix": contexts[cid]["prefix"],
                "query_id": contexts[cid]["query_id"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "nq": nq,
                "q_span": span.float().cpu(),
                "v_pe": v_pe.float().cpu(),
            }
        del captured
    assert len(records) == len(contexts), (len(records), len(contexts))
    _, bank_sha = R.bank_manifest_and_sha()
    return {
        "layers": layers,
        "per_context": records,
        "donor_map": donor_map,
        "bank_sha": bank_sha,
        "gpu2_sha": G2B.gpu2_manifest_sha(),
        "repro": R._repro(cfg),
    }


def gpu2_parity_report(gpu2_bank: dict, parent_bank: dict) -> dict:
    """Capture parity on the 15 SHARED contexts (the fu2 two-bar recipe,
    reused verbatim on the parent-context subset — the 5 conv2 contexts are
    new and have no parent reference)."""
    shared = {
        cid: rec
        for cid, rec in gpu2_bank["per_context"].items()
        if cid in parent_bank["per_context"]
    }
    assert len(shared) == 15, len(shared)
    return F2.fu2_parity_report({**gpu2_bank, "per_context": shared}, parent_bank)


# ── anchors (the gate ceiling draws; parent anchor protocol) ────────────


@torch.no_grad()
def gen_gpu2_anchors(
    cfg: R.RunConfig, model, tok, context_ids: list[str], draws: int
) -> tuple[list[dict], dict]:
    """K unpatched temp-1.0 rollouts per context + both-pooling V_a (the
    parent ``phase_anchors`` recipe over the extended context set)."""
    contexts = G2B.build_extended_contexts()
    ctx_list = [contexts[c] for c in context_ids]
    eot = R.eot_tail_ids(tok)
    t0 = time.monotonic()
    outs = R.generate_batch(
        model,
        tok,
        ctx_list,
        n=draws,
        hook=None,
        max_new_tokens=cfg.max_new_tokens,
        temperature=R.ANCHOR_TEMPERATURE,
        seed_base=cfg.seed_base,
        render_fn=BANK.render_context_2094,
        ids_fn=BANK.context_token_ids_2094,
    )
    logger.info(
        "[anchors] %d contexts x %d draws in %.1fs",
        len(context_ids),
        draws,
        time.monotonic() - t0,
    )
    ctx_ids = {cid: BANK.context_token_ids_2094(tok, contexts[cid]) for cid in context_ids}
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    rows: list[dict] = []
    for b, cid in enumerate(context_ids):
        for i, text in enumerate(outs[b]):
            flat_ctx.append(ctx_ids[cid])
            flat_text.append(text)
            rows.append({"context_id": cid, "draw": i, "seed": cfg.seed_base + i, "text": text})
    states = R.capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = R.cap_hit(n_tok, cfg.max_new_tokens)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
        r["temperature"] = R.ANCHOR_TEMPERATURE
    va = {
        "layers": cfg.layers,
        "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in rows],
        "va_span": states["va_span"],
        "va_tail": states["va_tail"],
        "pooling": states["pooling"],
        "empty_rows": states["empty_rows"],
        "repro": R._repro(cfg),
    }
    return rows, va


# ── the anchor-validation gate ──────────────────────────────────────────


def gate_units(
    rows: list[dict],
) -> tuple[list[J.JudgeUnit], dict[str, list[J.JudgeUnit]]]:
    """Coherence + (fp-bare, fp-conv2) judge units over the gate's anchor rows
    (floor AND ceiling), deduped per (context, draw, rubric) — the parent
    anchor-wave convention (a draw's score under a rubric is pair-independent)."""
    coh: list[J.JudgeUnit] = []
    beh: dict[str, list[J.JudgeUnit]] = {rid: [] for rid in GATE_RUBRIC_IDS}
    for row in rows:
        cid, draw = row["context_id"], row["draw"]
        source = {"kind": "anchor", "context_id": cid, "draw": draw}
        coh.append(
            J.JudgeUnit(
                item_id=J._item_id("c", f"coh|g2anch|{cid}|{draw}"),
                rubric_id=J.COHERENCE_RUBRIC_ID,
                question=J._query_text(cid),
                answer=row["text"],
                source=source,
            )
        )
        for rid in GATE_RUBRIC_IDS:
            beh[rid].append(
                J.JudgeUnit(
                    item_id=J._item_id("a", f"beh|g2anch|{cid}|{draw}|{rid}"),
                    rubric_id=rid,
                    question=J._query_text(cid),
                    answer=row["text"],
                    source=source,
                )
            )
    return coh, beh


def gate_rubric_registry() -> dict[str, str]:
    """The gate's production instruments: the parent coherence + fp-bare
    rubrics and the fp-conv2 extension — exactly the ``--gpu2`` judge registry
    entries (equality pinned in tests)."""
    reg = J.rubric_registry(gpu2=True)
    return {
        J.COHERENCE_RUBRIC_ID: reg[J.COHERENCE_RUBRIC_ID],
        **{rid: reg[rid] for rid in GATE_RUBRIC_IDS},
    }


def mock_gate_score(context_id: str, rubric_id: str, variant: str) -> float:
    """Deterministic smoke-judge scores. ``mock-pass``: the conv2 register
    carries (well-separated floor/ceiling deltas); ``mock-fail``: every draw
    scores identically under both rubrics => separation exactly 0."""
    assert variant in ("mock-pass", "mock-fail"), variant
    if rubric_id == J.COHERENCE_RUBRIC_ID:
        return 95.0
    if variant == "mock-fail":
        return 50.0
    prefix = context_id.split("__")[0]
    if rubric_id == "fp-conv2":
        return 90.0 if prefix == G2B.CONV2_PREFIX else 5.0
    assert rubric_id == "fp-bare", rubric_id
    return 10.0 if prefix == G2B.CONV2_PREFIX else 90.0


def judge_config(paths: GPU2Paths) -> J.JudgeConfig:
    jc = J.JudgeConfig(
        work_root=paths.judge_root,
        cache_root=paths.judge_cache,
        rollouts_dir=paths.judge_root / "_unused_rollouts",
        anchors_file=paths.judge_root / "_unused_anchors.jsonl",
        stage2_dir=None,
    )
    for d in (jc.scores_dir, jc.items_dir, jc.raw_dir):
        d.mkdir(parents=True, exist_ok=True)
    return jc


def run_gate_judging(paths: GPU2Paths, rows: list[dict], judge_mode: str) -> None:
    """Dispatch the three gate waves (coherence + the two prefix rubrics).

    ``live`` routes ``issue2094_judge.run_wave`` verbatim (wave-meta resume,
    rubric-keyed cache, bounded transport retry — the production instrument;
    ~3 x ~100-item waves ride the sync dispatch path under the 2k crossover).
    ``mock-*`` writes the SAME scores-row files deterministically (zero API),
    so the downstream separation reader is ONE code path in both modes.
    """
    assert judge_mode in JUDGE_MODES, judge_mode
    coh, beh = gate_units(rows)
    registry = gate_rubric_registry()
    jc = judge_config(paths)
    waves: list[tuple[str, str, list[J.JudgeUnit]]] = [
        (f"coherence.{GATE_WAVE_SUFFIX}", J.COHERENCE_RUBRIC_ID, coh)
    ]
    waves += [(f"{rid}.{GATE_WAVE_SUFFIX}", rid, beh[rid]) for rid in GATE_RUBRIC_IDS]
    for wave, rid, units in waves:
        if judge_mode == "live":
            J.run_wave(wave, rid, registry[rid], units, jc)
            continue
        J._validate_units(units)
        rows_out = [
            {
                "item_id": u.item_id,
                "wave": wave,
                "rubric_id": rid,
                "score": mock_gate_score(u.source["context_id"], rid, judge_mode),
                "n_kept_draws": 1,
                "transport_lost_residual": 0,
                "judge_mode": judge_mode,
                **u.source,
            }
            for u in units
        ]
        R._write_jsonl_atomic(jc.scores_dir / f"{wave}.scores.jsonl", rows_out)
        logger.info("[gate] wave %s mocked (%s): %d rows", wave, judge_mode, len(rows_out))


def load_gate_scores(paths: GPU2Paths) -> tuple[dict, dict]:
    """(coherence[(cid, draw)], behavior[(cid, draw, rubric_id)]) score lookups
    from the gate waves' scores rows."""
    jc = judge_config(paths)
    coh: dict[tuple[str, int], float | None] = {}
    beh: dict[tuple[str, int, str], float | None] = {}
    files = sorted(jc.scores_dir.glob(f"*.{GATE_WAVE_SUFFIX}.scores.jsonl"))
    assert files, f"no gate scores under {jc.scores_dir}"
    for f in files:
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["context_id"], r["draw"])
            if r["rubric_id"] == J.COHERENCE_RUBRIC_ID:
                coh[key] = r["score"]
            else:
                beh[(*key, r["rubric_id"])] = r["score"]
    return coh, beh


def gate_separations(coh: dict, beh: dict, draws_by_ctx: dict[str, list[int]]) -> list[dict]:
    """Per re-formed pair: floor/ceiling Δ means over COHERENT draws — the
    parent ``anchor_pair_stats`` convention verbatim (Δ_d = (judge_B -
    judge_A)/100; floor draws from context A, ceiling from context B;
    incoherent (< coherence threshold) and rule-9-missing draws excluded and
    counted)."""
    out: list[dict] = []
    rid_a, rid_b = GATE_RUBRIC_IDS
    for pair in G2B.build_gpu2_pairs():
        stats = {}
        for role, cid in (("floor", pair.a), ("ceiling", pair.b)):
            deltas, n_incoh, n_missing = [], 0, 0
            for d in draws_by_ctx.get(cid, []):
                score = coh.get((cid, d))
                if score is None or float(score) <= J.COHERENCE_THRESHOLD:
                    n_incoh += 1
                    continue
                sa = beh.get((cid, d, rid_a))
                sb = beh.get((cid, d, rid_b))
                if sa is None or sb is None:
                    n_missing += 1
                    continue
                deltas.append((float(sb) - float(sa)) / 100.0)
            stats[role] = {
                "mean": (sum(deltas) / len(deltas)) if deltas else None,
                "n": len(deltas),
                "n_incoherent": n_incoh,
                "n_judge_missing": n_missing,
            }
        fl, ce = stats["floor"]["mean"], stats["ceiling"]["mean"]
        out.append(
            {
                "pair_id": pair.pair_id,
                "setting": pair.setting,
                "kind": "prefix",
                "context_a": pair.a,
                "context_b": pair.b,
                "floor": stats["floor"],
                "ceiling": stats["ceiling"],
                "separation": (ce - fl) if (fl is not None and ce is not None) else None,
            }
        )
    return out


def gate_verdict(
    sep_rows: list[dict],
    min_sep: float = MIN_ABS_SEPARATION,
    min_passing: int = GATE_MIN_PASSING,
) -> dict:
    """PASS iff >= ``min_passing`` of the re-formed pairs reach |sep| >= floor
    (a None separation — no coherent scored draws on a side — never passes)."""
    per_pair = [
        {
            "pair_id": r["pair_id"],
            "separation": r["separation"],
            "passes": r["separation"] is not None and abs(r["separation"]) >= min_sep,
        }
        for r in sep_rows
    ]
    n_passing = sum(1 for p in per_pair if p["passes"])
    return {
        "passed": n_passing >= min_passing,
        "n_passing": n_passing,
        "n_pairs": len(per_pair),
        "min_abs_separation": min_sep,
        "min_passing": min_passing,
        "per_pair": per_pair,
    }


# ── cap-hit + upload + sentinel ─────────────────────────────────────────


def upload_and_sentinel(
    cfg: R.RunConfig,
    paths: GPU2Paths,
    totals: dict,
    gate: dict | None,
    parity: dict | None,
    judge_mode: str,
    wall_h: float,
    halted: str | None,
) -> None:
    """Fail-loud bulk uploads (one commit per prefix, exact-set verified via
    ``R._upload_dir``) + the pod sentinel. Runs on the gate-FAIL designed halt
    too (whatever exists uploads; the grid prefixes are simply empty then).
    ALL SEVEN write prefixes are enumerated in the sentinel (#1773)."""
    uploaded: dict[str, list[str]] = {}
    uploaded["rollouts"] = R._upload_dir(
        cfg, cfg.rollouts_dir, f"{HF_GPU2_TEXT}/rollouts", ["shard_*.jsonl"]
    )
    uploaded["anchors_text"] = R._upload_dir(
        cfg, paths.anchors_dir, f"{HF_GPU2_TEXT}/anchors", ["anchors_gpu2.jsonl", "*.json"]
    )
    uploaded["judge_gate"] = R._upload_dir(
        cfg,
        paths.judge_root,
        f"{HF_GPU2_TEXT}/judge_gate",
        ["scores/*.jsonl", "scores/*.json", "items/*.jsonl", "raw/*.json"],
    )
    uploaded["va"] = R._upload_dir(cfg, cfg.va_dir, f"{HF_GPU2_TENSORS}/va", ["shard_*.pt"])
    uploaded["anchors_va"] = R._upload_dir(
        cfg, paths.anchors_dir, f"{HF_GPU2_TENSORS}/anchors", ["va_anchors_gpu2.pt"]
    )
    uploaded["bank"] = R._upload_dir(
        cfg, paths.bank_dir, f"{HF_GPU2_TENSORS}/bank", ["gpu2_bank.pt", "*.json"]
    )

    caphit = F2.fu2_caphit_report(cfg)
    R._write_json_atomic(cfg.manifest_dir / "gpu2_caphit.json", caphit)
    cap_rows = sum(
        agg["cap_hit"] for row in caphit["cells"] for k, agg in row.items() if k in R.ARMS
    )
    n_rows = sum(agg["n"] for row in caphit["cells"] for k, agg in row.items() if k in R.ARMS)
    payload = {
        "eval_numbers": {
            "followup_label": "gpu2_mq_replacement_prefix",
            "halted": halted,
            "gate": (
                None
                if gate is None
                else {k: gate[k] for k in ("passed", "n_passing", "n_pairs", "per_pair")}
            ),
            "judge_mode": judge_mode,
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
            str(p)
            for p in (
                cfg.rollouts_dir,
                cfg.va_dir,
                paths.anchors_dir,
                paths.bank_dir,
                paths.judge_root,
                cfg.manifest_dir,
            )
        ),
        "reproducibility_card": {
            **R._repro(cfg),
            "seed_base": cfg.seed_base,
            "bank_seed": BANK.SEED,
            "gpu2_donor_seed": G2B.GPU2_SEED,
            "gpu2_sha": G2B.gpu2_manifest_sha(),
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": R.GRID_TEMPERATURE,
            "anchor_temperature": R.ANCHOR_TEMPERATURE,
            "anchor_draws": cfg.anchor_draws,
            "gen_batch": cfg.gen_batch,
            "judge_model": J.DEFAULT_JUDGE_MODEL,
            "judge_max_tokens": J.DEFAULT_JUDGE_MAX_TOKENS,
            "gate": {"min_abs_separation": MIN_ABS_SEPARATION, "min_passing": GATE_MIN_PASSING},
            "gpu2_regime_token": GPU2_REGIME_TOKEN,
        },
        "wandb_url": None,
        "hf_hub_url": (f"https://huggingface.co/datasets/{R.HF_DATA_REPO}/tree/main/{R.HF_PREFIX}"),
        "worktree_path": str(R.REPO_ROOT),
        "final_commit_sha": R._git_sha(),
        "gpu_hours_used": round(wall_h, 3),
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "gpu2 round scope: ce slot only (the brief's headline-slot enumeration); "
            "pe / control slots / Type-B are out of the tight final-round scope",
            "gate floor draws are the parent's bare anchor rollouts RE-JUDGED fresh "
            "(both gate sides share one wave instrument; parent scores untouched)",
            "donor nulls draw from the PARENT matched-query pool (all 5 recipients "
            "share the (bare, conv2) prefix pair, which the parent's cross-prefix-pair "
            "null constraint forbids as a donor source by construction)",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
        "upload_prefix_names": {
            "rollouts": f"{HF_GPU2_TEXT}/rollouts",
            "anchors_text": f"{HF_GPU2_TEXT}/anchors",
            "judge_gate": f"{HF_GPU2_TEXT}/judge_gate",
            "va": f"{HF_GPU2_TENSORS}/va",
            "anchors_va": f"{HF_GPU2_TENSORS}/anchors",
            "bank": f"{HF_GPU2_TENSORS}/bank",
            "manifests": f"{HF_GPU2_TENSORS}/manifests",
        },
    }
    # Written BEFORE the manifests commit => the done record rides that commit.
    R._write_json_atomic(cfg.manifest_dir / "gpu2_upload_done.json", payload)
    uploaded["manifests"] = R._upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_GPU2_TENSORS}/manifests",
        ["*.json", "blocks/*.done.json"],
    )
    payload["uploaded_prefixes"] = {k: len(v) for k, v in uploaded.items()}

    sentinel = cfg.log_dir / SENTINEL_NAME
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    R._write_json_atomic(
        sentinel,
        {"sentinel_schema_version": 1, "kind": kind, "version": 1, "note": payload},
    )
    logger.info("[upload] sentinel written: %s (kind=%s halted=%s)", sentinel, kind, halted)


# ── entrypoint ───────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths
    (the #1689 false-pass class): transformers loads inside
    ``R.load_model_and_tokenizer``; hub helpers inside ``FU1.stage_bank`` /
    ``stage_parent_anchors`` / ``R._upload_dir``; the judge chain
    (``judge_graded`` -> ``judge_completions_batch``) inside ``J.run_wave``."""
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from explore_persona_space.eval.batch_judge import (  # noqa: F401
        judge_completions_batch,
    )
    from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        stage_hub_file,
        verify_repo_paths_uploaded,
    )

    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 gpu2_mq_replacement_prefix driver (diagnose / gate / grid)."
    )
    ap.add_argument("--run", action="store_true", help="execute stage->bank->anchors->gate->grid")
    ap.add_argument(
        "--phase",
        choices=("diagnose",),
        help="standalone VM-side phase (diagnose writes the committed diagnosis.json)",
    )
    ap.add_argument(
        "--fmetrics-dir",
        type=Path,
        default=R.REPO_ROOT / "eval_results/issue_2094/f_metrics",
        help="committed f_metrics dir (diagnose input + output root)",
    )
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=R.REPO_ROOT / "eval_results/issue_2094/judge/scores",
        help="parent judge scores dir (diagnose attribution; optional)",
    )
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/issue2094_gpu2_out"))
    ap.add_argument("--log-dir", type=Path, default=R.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=R.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument(
        "--tiny-layers",
        type=int,
        default=28,
        help="tiny model layer count (default 28 so the production variant set exists)",
    )
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=GPU2_MAX_NEW_TOKENS,
        help="2048 from the start (fu1's raised cap; brief requirement)",
    )
    ap.add_argument("--anchor-draws", type=int, default=R.ANCHOR_DRAWS)
    ap.add_argument("--seed-base", type=int, default=R.SEED_BASE)
    ap.add_argument("--smoke", action="store_true", help="tiny per-arm-class family slice")
    ap.add_argument(
        "--judge",
        choices=JUDGE_MODES,
        default="live",
        help="gate judging: live (production API) | mock-pass / mock-fail (smoke)",
    )
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument(
        "--upload-every",
        type=int,
        default=20,
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
        phase=GPU2_LABEL,
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
        anchor_draws=args.anchor_draws,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=False,
        force=False,
        worker_index=0,
        num_workers=1,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        # 300 5-row blocks at the fu1/fu2 measured 60-row-block basis
        # (102-146 s at gen_batch 16) with a conservative small-batch (B=5)
        # decode-efficiency multiplier => ~2.2-3.0 h grid + ~0.5 h overheads.
        planned_wall_h=3.0,
        gpu_hours_budgeted=4.0,  # the brief's estimate for the round
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
    if args.phase == "diagnose":
        scores_dir = args.scores_dir if args.scores_dir.is_dir() else None
        return phase_diagnose(args.fmetrics_dir, scores_dir)
    assert args.run, "pass --run (or --phase diagnose / --import-check)"
    cfg = build_config(args)
    paths = GPU2Paths(out_root=cfg.out_root)
    for d in (
        cfg.rollouts_dir,
        cfg.va_dir,
        cfg.manifest_dir / "blocks",
        paths.bank_dir,
        paths.anchors_dir,
        cfg.log_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    draws = 2 if cfg.smoke else cfg.anchor_draws  # >= 2 (fmetrics half-split floor)
    regime_fp = gpu2_regime_fingerprint(cfg, args.judge, draws)

    # ── phase: diagnose-check (committed premise re-verified) ───────────
    logger.info("[phase=gpu2_diagnose_check] smoke=%s tiny=%s", cfg.smoke, cfg.tiny)
    assert_diagnosis_check(R.REPO_ROOT)

    # ── phase: enumerate (CPU-only) ─────────────────────────────────────
    families = enumerate_gpu2_families(cfg.n_layers)
    totals = R.grid_totals(families)
    assert totals == EXPECTED_GPU2_TOTALS, (totals, EXPECTED_GPU2_TOTALS)
    if cfg.smoke:
        families = slice_gpu2_smoke(families, cfg.n_layers)
        totals = R.grid_totals(families)
    logger.info("[enumerate] %s", totals)
    donor_map = G2B.gpu2_donor_map()
    pairs_by_id = G2B.gpu2_pairs_by_id()

    # ── phase: stage (production inputs) ────────────────────────────────
    logger.info("[phase=gpu2_stage]")
    parent_bank: dict | None = None
    if cfg.tiny:
        logger.info("[stage] tiny smoke: no parent bank/anchors — local floor generation")
    else:
        FU1.stage_bank(cfg, args.hf_revision)
        parent_bank = torch.load(
            cfg.bank_dir / "vc_bank.pt", map_location="cpu", weights_only=False
        )
        _staged_sha = parent_bank.get("bank_sha")
        _local_sha = R.bank_manifest_and_sha()[1]
        assert _staged_sha in (None, _local_sha), (_staged_sha, _local_sha)
        stage_parent_anchors(paths, args.hf_revision)

    # ── phase: bank (fresh 20-context capture + parity gate) ────────────
    logger.info("[phase=gpu2_bank]")
    parity: dict | None = None
    model, tok = R.load_model_and_tokenizer(cfg)
    if _regime_checked_done(paths.bank_done, regime_fp, "gpu2 bank") is not None:
        logger.info("[bank] gpu2 bank already captured for this regime — loading")
        gpu2_bank = torch.load(paths.bank_path, map_location="cpu", weights_only=False)
        if paths.parity_path.exists():
            parity = json.loads(paths.parity_path.read_text(encoding="utf-8"))
    else:
        gpu2_bank = capture_gpu2_bank(cfg, model, tok, donor_map)
        if parent_bank is not None:
            parity = gpu2_parity_report(gpu2_bank, parent_bank)
            R._write_json_atomic(paths.parity_path, parity)
            logger.info(
                "[bank] parity vs parent bank: passed=%s early_min=%.6f flat_min=%.6f",
                parity["passed"],
                parity["early_min_cos"],
                parity["flat_min_cos"],
            )
            if not parity["passed"]:
                logger.error("[bank] capture-parity gate FAILED — report at %s", paths.parity_path)
                print("[phase=gpu2_parity_gate_failed]", flush=True)
                return RC_PARITY_GATE
        else:
            logger.info("[bank] parity gate skipped (tiny smoke: no parent bank)")
        R._save_pt_atomic(paths.bank_path, gpu2_bank)
        R._write_json_atomic(
            paths.bank_done,
            {
                "regime_fp": regime_fp,
                "n_contexts": len(gpu2_bank["per_context"]),
                "parity": ("skipped-tiny" if parity is None else parity["passed"]),
                "repro": R._repro(cfg),
            },
        )
    del parent_bank

    # ── phase: anchors (gate ceiling draws; + local floor under --tiny) ─
    logger.info("[phase=gpu2_anchors]")
    ceiling_ctx = [G2B.conv2_context_id(q) for q in BANK.QUERY_ORDER]
    anchor_ctx = ceiling_ctx + ([f"bare__{q}" for q in BANK.QUERY_ORDER] if cfg.tiny else [])
    done = _regime_checked_done(paths.anchors_done, regime_fp, "gpu2 anchors")
    if done is not None and paths.anchors_file.exists() and paths.anchors_va.exists():
        n_rows = sum(1 for line in paths.anchors_file.open(encoding="utf-8") if line.strip())
        assert n_rows == int(done.get("n_rows", -1)), (n_rows, done.get("n_rows"))
        logger.info("[anchors] already done for this regime — skipping (%d rows)", n_rows)
        anchor_rows = [
            json.loads(line) for line in paths.anchors_file.open(encoding="utf-8") if line.strip()
        ]
    else:
        anchor_rows, va = gen_gpu2_anchors(cfg, model, tok, anchor_ctx, draws)
        R._write_jsonl_atomic(paths.anchors_file, anchor_rows)
        R._save_pt_atomic(paths.anchors_va, va)
        cap_hits = sum(1 for r in anchor_rows if r["cap_hit"])
        R._write_json_atomic(
            paths.anchors_done,
            {
                "regime_fp": regime_fp,
                "n_contexts": len(anchor_ctx),
                "draws": draws,
                "n_rows": len(anchor_rows),
                "n_cap_hit": cap_hits,
                "cap_hit_frac": cap_hits / max(1, len(anchor_rows)),
                "repro": R._repro(cfg),
            },
        )
        logger.info("[anchors] rows=%d cap_hit=%d", len(anchor_rows), cap_hits)

    # ── phase: gate (the round's kill gate) ─────────────────────────────
    logger.info("[phase=gpu2_gate] judge=%s", args.judge)
    gate_done = _regime_checked_done(paths.gate_report, regime_fp, "gpu2 gate")
    if gate_done is not None:
        gate = gate_done["verdict"]
        logger.info("[gate] recorded verdict re-entry: passed=%s", gate["passed"])
    else:
        floor_rows = (
            [r for r in anchor_rows if r["context_id"].split("__")[0] == "bare"]
            if cfg.tiny
            else load_floor_rows(paths, draws)
        )
        ceiling_rows = [
            r for r in anchor_rows if r["context_id"].split("__")[0] == G2B.CONV2_PREFIX
        ]
        assert floor_rows and ceiling_rows, (len(floor_rows), len(ceiling_rows))
        gate_rows = floor_rows + ceiling_rows
        run_gate_judging(paths, gate_rows, args.judge)
        coh, beh = load_gate_scores(paths)
        draws_by_ctx: dict[str, list[int]] = {}
        for r in gate_rows:
            draws_by_ctx.setdefault(r["context_id"], []).append(r["draw"])
        sep_rows = gate_separations(coh, beh, draws_by_ctx)
        gate = gate_verdict(sep_rows)
        R._write_json_atomic(
            paths.gate_report,
            {
                "regime_fp": regime_fp,
                "judge_mode": args.judge,
                "verdict": gate,
                "separations": sep_rows,
                "n_floor_rows": len(floor_rows),
                "n_ceiling_rows": len(ceiling_rows),
                "instrument": {
                    "judge_model": J.DEFAULT_JUDGE_MODEL,
                    "max_tokens": J.DEFAULT_JUDGE_MAX_TOKENS,
                    "n_draws": J.JUDGE_N_DRAWS,
                    "coherence_threshold": J.COHERENCE_THRESHOLD,
                },
                "repro": R._repro(cfg),
            },
        )
    logger.info(
        "[gate] passed=%s n_passing=%d/%d", gate["passed"], gate["n_passing"], gate["n_pairs"]
    )
    wall_h = (time.monotonic() - t_start) / 3600.0
    if not gate["passed"]:
        logger.error("[gate] anchor-separation gate FAILED — designed halt, NO grid spend")
        upload_and_sentinel(
            cfg, paths, totals, gate, parity, args.judge, wall_h, halted="anchor_gate_failed"
        )
        print("[phase=gpu2_anchor_gate_failed]", flush=True)
        print("[phase=done]", flush=True)
        return RC_ANCHOR_GATE

    # ── phase: grid (R.run_block verbatim over the re-formed pairs) ─────
    eot = R.eot_tail_ids(tok)
    blocks = R.blocks_for_worker(families, 0, 1)
    logger.info(
        "[phase=gpu2_grid] %d blocks / %d cells at max_new_tokens=%d",
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
        rec = R.run_block(cfg, model, tok, gpu2_bank, block, pairs_by_id, donor_map, eot, regime_fp)
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
                f"{HF_GPU2_TEXT}/rollouts",
                [f"shard_{b.slug}.jsonl" for b in pending],
            )
            pending = []

    # ── phase: upload + sentinel ────────────────────────────────────────
    logger.info("[phase=gpu2_upload]")
    wall_h = (time.monotonic() - t_start) / 3600.0
    upload_and_sentinel(cfg, paths, totals, gate, parity, args.judge, wall_h, halted=None)
    logger.info("[phase=gpu2_done]")
    print("[phase=done]", flush=True)
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
