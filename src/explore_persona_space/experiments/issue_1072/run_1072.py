"""Issue #1072 pod-side driver — component decomposition of #952's ridge maps.

Phases (plan §4.4; smoke IS the run at tiny n — same dispatcher, same phases,
same single-process shape; smoke narrows ONLY the pool (10 contexts), the
model (a from-config tiny Qwen2 over the REAL tokenizer/vocab) and the layer
set (0,1 of the 2-layer tiny model)):

  A ``stage``    — parent pool/prompt/split verification (#952 phase0_verify
                   replay) + staging of the 16 parent slot shards + 4 span
                   files at the PINNED revision (smoke: synth texts + a
                   synth-capture through the SAME capture code writes the
                   parent-schema shards).
  B ``capture``  — decomposition re-capture: one teacher-forced forward per
                   (context, arm), hooked at the decomposition layers,
                   streaming the û-projection accumulators (remainder
                   components, per-slot alpha's, next-token ids, p_last) with
                   gates g1 (span alignment), g2 (slot equivalence), g3
                   (remainder equivalence) in-stream. Uploads the decomposed
                   store to HF BEFORE phase C (expensive-store-before-long-fit
                   ordering).
  C ``battery``  — per (fold, layer): FULL-cell calibration re-run
                   (reproduction gates g4/g5 vs the parent's committed k-fold
                   artifacts; kill K3/K5) + component cells at frozen λ*
                   (shared SVD per source; cross-term REPORTED channel) +
                   p_last prefix-arm cells; per-fold npz + JSON outputs.

Phases D (stats) / E (figures) are VM-side: ``scripts/issue1072_stats.py`` +
``scripts/issue1072_figures.py``.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import json
import logging
import pathlib
import sys
import time
from typing import Any

import numpy as np

from explore_persona_space.experiments.issue_952 import run_952 as parent952
from explore_persona_space.experiments.issue_952.ridge_battery import run_ridge_cell
from explore_persona_space.experiments.issue_1072.component_ridge import (
    component_parity_gate as _component_parity_gate,
)
from explore_persona_space.experiments.issue_1072.component_ridge import (
    run_component_cell,
    serial_component_reference,
)

logger = logging.getLogger("issue1072.run")

ISSUE = 1072
ISSUE_SLUG = "issue1072_component_decomposition"
HF_DATA_REPO = parent952.HF_DATA_REPO
PARENT_SLUG = parent952.ISSUE_SLUG  # issue952_position_divergence
# Pinned parent artifact revisions (plan §10 Reproducibility Card).
PARENT_TENSOR_REVISION = "5b62649cefb34902fd630f21630164e8d1d99764"
DEFAULT_MODEL = parent952.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct
PROD_LAYERS = (14, 20, 23, 26)  # plan §11 (task Goal + parent decision layers)
PROD_HIDDEN = 3584
PROD_VOCAB = 152064
K_FOLDS = 5
CAL_FOLD = K_FOLDS - 1  # fold 4 ≡ the parent single split (calibration fold)
T2 = 16  # the registered H2 closure cell (t=16); matched universe span >= 32
REPRO_TOL = 1e-6  # g4 (parent L20_REPRO_TOL convention)
COS_GATE_MIN = 0.999  # g2/g3 per-cell cosine floor (parent fp16 gate)
COS_GATE_MAX_BELOW_FRAC = 0.001  # K2: > 0.1% of valid cells below floor = abort
PILOT_ABORT_RC = 7  # designed compute abort (K4) — distinct rc + report JSON
CAPTURE_BOOKED_H = 4.0  # plan §9 row B (booked = 2x naive)
BATTERY_BOOKED_H = 4.0  # plan §9 row C (booked = 2x naive)
ARMS = parent952.ARMS
MATCHED_ARMS = parent952.MATCHED_ARMS
SLOT_NAMES = parent952.SLOT_NAMES
SLOT_IDX = parent952.SLOT_IDX
POSITION_SLOTS = parent952.POSITION_SLOTS
DEFAULT_LAMBDAS_LIST = parent952.DEFAULT_LAMBDAS_LIST
# l16_m1 (trailing \n) has NO realized next token -> excluded from decomposition
# (plan §4.2); l16_m2 (<|im_end|>) next token is '\n' — retained but flagged.
EXCLUDED_DECOMP_SLOTS = ("l16_m1",)
FLAGGED_DECOMP_SLOTS = ("l16_m2",)
DECOMP_SLOTS = tuple(s for s in POSITION_SLOTS if s not in EXCLUDED_DECOMP_SLOTS)  # 41
REM_FULL_SLOT = f"rem_mean_gt{T2}"
COMMITTED_KFOLD_DIR = ("eval_results", "issue_952", "kfold-decision-cells")

log_phase = parent952.log_phase
write_sentinel = parent952.write_sentinel
_json_np = parent952._json_np


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _tensors_dir(base_dir: pathlib.Path) -> pathlib.Path:
    d = base_dir / "analysis_tensors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_out_dir(base_dir: pathlib.Path) -> pathlib.Path:
    """This issue's eval-results dir (mirrored to git + HF)."""
    d = base_dir / "eval_results" / "issue_1072"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hf_commit_files_1072(label: str, paths: list[pathlib.Path], base_dir: pathlib.Path) -> None:
    """One retried create_commit to the data repo under ISSUE_SLUG + scoped verify.

    path_in_repo mirrors the path relative to base_dir (analysis_tensors/...,
    eval_results/issue_1072/..., logs/...) — the child owns its OWN prefix,
    never the parent's (artifact-reuse check (i) upload-destination rule).
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    from explore_persona_space.orchestrate import hub as eps_hub

    ops = []
    for p in paths:
        rel = p.relative_to(base_dir)
        ops.append(
            CommitOperationAdd(
                path_in_repo=f"{ISSUE_SLUG}/{pathlib.PurePosixPath(rel)}", path_or_fileobj=str(p)
            )
        )
    api = HfApi()
    eps_hub.retry_transient(
        lambda: api.create_commit(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue 1072: {label} ({len(ops)} files)",
            operations=ops,
        ),
        what=f"issue1072 create_commit {label}",
    )
    missing = eps_hub.verify_repo_paths_uploaded(
        api,
        HF_DATA_REPO,
        [op.path_in_repo for op in ops],
        path_in_repo=ISSUE_SLUG,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"HF upload verification FAIL ({label}): missing {sorted(missing)[:3]}")
    logger.info("[upload] %s: %d files committed + Hub-verified", label, len(ops))


def _git_sha() -> str:
    return parent952._repo_git_sha()


def _env_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _run_metadata(smoke: bool, layers: tuple[int, ...]) -> dict[str, Any]:
    """Reproducibility metadata carried by every result JSON (CLAUDE.md rule)."""
    return {
        "issue": ISSUE,
        "git_sha": _git_sha(),
        "env_versions": _env_versions(),
        "ts": time.time(),
        "smoke": bool(smoke),
        "layers": list(layers),
        "parent_tensor_revision": PARENT_TENSOR_REVISION,
        "model_id": DEFAULT_MODEL if not smoke else "smoke-tiny-qwen2(seed 0)",
    }


def _pilot_check(
    name: str,
    measured_wall_s: float,
    units_done: int,
    units_total: int,
    booked_h: float,
    base_dir: pathlib.Path,
    smoke: bool,
    execution_shape: str,
) -> None:
    """K4 designed compute abort: projected wall > 2x booked -> report + rc 7.

    Runs the COMPUTATION identically under smoke but demotes the verdict to a
    log line (the production-scale threshold is meaningless at tiny n — the
    #1345 gate-calibration lesson).
    """
    per_unit = measured_wall_s / max(units_done, 1)
    projected_h = per_unit * units_total / 3600.0
    rec = {
        "gate": name,
        "measured_wall_s": measured_wall_s,
        "units_done": units_done,
        "units_total": units_total,
        "per_unit_s": per_unit,
        "projected_wall_h": projected_h,
        "booked_wall_h": booked_h,
        "abort_threshold_h": 2.0 * booked_h,
        "execution_shape": execution_shape,
        "verdict": "ABORT" if projected_h > 2.0 * booked_h else "PASS",
        "smoke_demoted": bool(smoke),
        "ts": time.time(),
    }
    out = eval_out_dir(base_dir) / f"pilot_gate_{name}.json"
    out.write_text(json.dumps(rec, indent=2, default=_json_np))
    logger.info("[pilot:%s] %s", name, json.dumps(rec, default=_json_np))
    if rec["verdict"] == "ABORT" and not smoke:
        log_phase(f"pilot_abort_{name}")
        logger.error(
            "[pilot:%s] projected %.2f h > 2x booked %.1f h — designed abort (K4), rc=%d",
            name,
            projected_h,
            booked_h,
            PILOT_ABORT_RC,
        )
        sys.exit(PILOT_ABORT_RC)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading + unembedding direction (plan §4.1)
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_model_contract(model, production: bool) -> None:
    """Load-time asserts pinning the gamma-folding exactness argument (plan §4.1)."""
    cfg = model.config
    assert getattr(cfg, "tie_word_embeddings", None) is False, (
        f"tie_word_embeddings={getattr(cfg, 'tie_word_embeddings', None)} — the decomposition "
        "basis assumes an UNTIED lm_head"
    )
    assert model.lm_head.bias is None, "lm_head carries a bias — logit_y(h) derivation invalid"
    norm_cls = type(model.model.norm).__name__
    assert "RMSNorm" in norm_cls, (
        f"final norm is {norm_cls}, not RMSNorm — gamma-folding is exact only for RMSNorm "
        "(no centering term)"
    )
    if production:
        assert cfg.hidden_size == PROD_HIDDEN, cfg.hidden_size
        assert cfg.vocab_size == PROD_VOCAB, cfg.vocab_size


def _unembed_dir(model):
    """û basis matrix U_dir = gamma ⊙ W_U rows, fp32 on the model device. (V, H)."""
    import torch

    with torch.no_grad():
        u = (
            model.lm_head.weight.detach().float()
            * model.model.norm.weight.detach().float()[None, :]
        )
    return u


def _load_production_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    _assert_model_contract(model, production=True)
    return model, tokenizer


def _build_smoke_model(tokenizer):
    """Deterministic from-config tiny Qwen2 over the REAL vocab-id space.

    The tiny-real pattern (gotchas.md "Mock-seam smokes"): fake ONLY the
    GPU-scale weights; tokenizer, template asserts, BPE ids, capture rig and
    battery all run the production code path. Seeded so a battery-only smoke
    re-invocation rebuilds the identical weights.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=PROD_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=parent952.SEQ_MAX_LEN,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    _assert_model_contract(model, production=False)
    assert len(tokenizer) <= cfg.vocab_size, (len(tokenizer), cfg.vocab_size)
    return model


def load_unembed_dir_production(model_id: str) -> np.ndarray:
    """(V, H) fp32 U_dir partial-loaded from the checkpoint safetensors (no full model).

    Used by a battery-only production re-invocation; ``--phase all`` passes
    the in-memory U_dir from phase B instead.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    tensors: dict[str, Any] = {}
    for name in ("lm_head.weight", "model.norm.weight"):
        shard = hf_hub_download(model_id, weight_map[name])
        with safe_open(shard, framework="pt", device="cpu") as f:
            tensors[name] = f.get_tensor(name)
    with torch.no_grad():
        u = tensors["lm_head.weight"].float() * tensors["model.norm.weight"].float()[None, :]
    out = u.numpy()
    assert out.shape == (PROD_VOCAB, PROD_HIDDEN), out.shape
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke substrate (synth pool/texts; the capture code path is production-shared)
# ═══════════════════════════════════════════════════════════════════════════════

_SMOKE_VOCAB = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
    "ocean",
    "river",
    "mountain",
    "forest",
    "desert",
    "valley",
    "meadow",
    "harbor",
    "island",
    "canyon",
    "glacier",
    "prairie",
    "tundra",
    "lagoon",
]


def _smoke_pool_and_texts(n: int = 10) -> tuple[list[int], dict[int, str], dict[str, dict]]:
    """Deterministic synth prompts + per-arm answers, spans ~150-200 tokens.

    All-long spans (parent kfold-smoke convention): every context survives the
    span>=32 matched universe AND every rem_gt{t} pooled slot up to t=128 stays
    valid, so the H1 cell's imputation-free decision-group invariant holds at
    smoke n (min-N floors: fold test/val blocks of 2 >= the >=2 survivor floor;
    train 6 >= smoke min_train 4).
    """
    pool_ids = list(range(n))
    prompts: dict[int, str] = {}
    texts: dict[str, dict[int, str]] = {arm: {} for arm in ARMS}
    for i in pool_ids:
        w = _SMOKE_VOCAB
        prompts[i] = f"{w[i % len(w)]} question {i}: please describe {w[(i * 3 + 1) % len(w)]}."
        for ai, arm in enumerate(ARMS):
            words = [w[(i * 7 + ai * 11 + k * 3) % len(w)] for k in range(170)]
            texts[arm][i] = f"Answer {arm} {i}: " + " ".join(words) + "."
    return pool_ids, prompts, texts


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B core — teacher-forced decomposition capture (plan §4.4 Phase B)
# ═══════════════════════════════════════════════════════════════════════════════


def slot_next_token_ids(
    pos: np.ndarray, valid: np.ndarray, rs: int, span: int, next_ext: np.ndarray
) -> np.ndarray:
    """Realized next-token id per single-position slot (46,), -1 where none.

    Convention (plan §4.2, standard AR): the activation at sequence position p
    produces the logits for the token at p+1, so slot position p maps to
    ``next_ext[t]`` with ``t = p - rs + 1`` (``next_ext[j] = full_ids[rs+j]``,
    j = 0..span-1; t=0 is c_last's). Valid iff 0 <= t <= span-1 — the final
    extended-span position (l16_m1, trailing \\n) has NO realized next token.
    """
    t_slot = pos - rs + 1
    ok = valid & (t_slot >= 0) & (t_slot <= span - 1)
    out = np.full(46, -1, dtype=np.int64)
    out[ok] = next_ext[t_slot[ok]]
    return out


def _lcp_prefix_len(id_lists: list[list[int]]) -> int:
    """Longest-common-prefix length over rendered full_ids (the p_last boundary).

    BPE-seam-safe by construction: computed over the ACTUAL renders, never by
    re-tokenizing a prefix string (gotchas.md plain-text span-boundary entry).
    """
    assert id_lists, "no renders"
    lcp = list(id_lists[0])
    for ids in id_lists[1:]:
        m = min(len(lcp), len(ids))
        k = 0
        while k < m and lcp[k] == ids[k]:
            k += 1
        lcp = lcp[:k]
    assert len(lcp) >= 3, f"degenerate template prefix (LCP={len(lcp)})"
    return len(lcp)


def tf_capture_decomp_arm(  # noqa: C901 — batched TF loop; slot reductions GPU-resident
    model,
    tokenizer,
    ids: list[int],
    prompts_by_id: dict[int, str],
    answers_by_id: dict[int, str],
    arm_name: str,
    layers: tuple[int, ...],
    u_dir,
    staged_slots_by_layer: dict[int, Any] | None,
    staged_spans: dict[str, dict] | None,
    batch_size: int = 8,
    emit_parent_shards: bool = False,
    pilot: dict | None = None,
    base_dir: pathlib.Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """One teacher-forced forward per context, hooked at ``layers`` (batched,
    LEFT pad + explicit position_ids — the parent's #823 rig, plan §11).

    Emits, per (context, layer): the decomposed remainder accumulators
    (``rem_par_mean_gt16_nx``, ``rem_mean_gt16_nx``, ``full_par_mean_nx``),
    per-slot alpha's (46 single-position slots, NaN where no realized next token),
    per-position alpha's (ragged), and — own arm — ``p_last``. Layer-independent:
    realized next-token ids (ragged + per-slot), spans. Gates g1/g2/g3 run
    in-stream against ``staged_spans`` / ``staged_slots_by_layer`` when given
    (production); ``emit_parent_shards=True`` (smoke stage) additionally
    assembles the parent-schema 72-slot store through the parent's own
    reductions so the battery consumes an identical layout.
    """
    import torch

    hid = model.config.hidden_size
    n = len(ids)
    n_layers = len(layers)
    dev = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── prep: render + g1 + next ids + slot positions ────────────────────────────
    prepped: list[tuple[int, dict]] = []
    spans: dict[str, dict] = {}
    g1_mismatches: list[dict] = []
    full_ids_all: list[list[int]] = []
    for row_i, cid in enumerate(ids):
        info = parent952._render_and_index(tokenizer, prompts_by_id[cid], answers_by_id[cid])
        assert info is not None, (
            f"[{arm_name}] id {cid}: empty answer/span — the coherence-verified pool "
            "guarantees non-empty texts (K1-adjacent structural violation)"
        )
        spans[str(cid)] = {
            "span": info["span"],
            "truncated": info["truncated"],
            "prompt_len": info["prompt_len"],
            "skipped": False,
        }
        if staged_spans is not None:
            ref = staged_spans.get(str(cid))
            if (
                ref is None
                or int(ref.get("span", -1)) != info["span"]
                or bool(ref.get("truncated")) != bool(info["truncated"])
            ):
                g1_mismatches.append({"id": cid, "recomputed": spans[str(cid)], "staged": ref})
        prepped.append((row_i, info))
        full_ids_all.append(info["full_ids"])
    if staged_spans is not None and g1_mismatches:
        rec = {"arm": arm_name, "n_mismatch": len(g1_mismatches), "sample": g1_mismatches[:5]}
        if base_dir is not None:
            (eval_out_dir(base_dir) / f"g1_abort_{arm_name}.json").write_text(
                json.dumps(rec, indent=2, default=_json_np)
            )
        raise RuntimeError(f"g1 span-alignment FAIL (K1): {rec['n_mismatch']} mismatches")

    prefix_len = _lcp_prefix_len(full_ids_all)
    min_prompt_len = min(info["prompt_len"] for _ri, info in prepped)
    assert prefix_len <= min_prompt_len, (prefix_len, min_prompt_len)

    # ── accumulators ─────────────────────────────────────────────────────────────
    rem_par = np.full((n_layers, n, hid), np.nan, dtype=np.float16)
    rem_nx = np.full((n_layers, n, hid), np.nan, dtype=np.float16)
    full_par = np.full((n_layers, n, hid), np.nan, dtype=np.float16)
    alpha_slots = np.full((n_layers, n, 46), np.nan, dtype=np.float32)
    p_last = np.full((n_layers, n, hid), np.nan, dtype=np.float16) if arm_name == "own" else None
    alpha_pos_rows: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    next_rows: list[np.ndarray] = []
    slot_next_ids = np.full((n, 46), -1, dtype=np.int64)
    rem_valid = np.zeros(n, dtype=bool)
    parent_slots = (
        np.full((n, n_layers, 72, hid), np.nan, dtype=np.float16) if emit_parent_shards else None
    )
    g2 = {int(la): {"n_cells": 0, "n_below": 0, "min_cos": 1.0} for la in layers}
    g3 = {int(la): {"n_cells": 0, "n_below": 0, "min_cos": 1.0} for la in layers}
    unique_next_ids: set[int] = set()

    captured: dict[int, Any] = {}

    def make_hook(li: int):
        def hook(module, _inp, output):
            captured[li] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles = [
        model.model.layers[la].register_forward_hook(make_hook(li)) for li, la in enumerate(layers)
    ]
    t_arm0 = time.time()
    try:
        with torch.no_grad():
            for b0 in range(0, len(prepped), batch_size):
                batch = prepped[b0 : b0 + batch_size]
                max_len = max(len(info["full_ids"]) for _ri, info in batch)
                input_ids, attn, pad_offs = [], [], []
                for _ri, info in batch:
                    pad_n = max_len - len(info["full_ids"])
                    input_ids.append([pad_id] * pad_n + info["full_ids"])
                    attn.append([0] * pad_n + [1] * len(info["full_ids"]))
                    pad_offs.append(pad_n)
                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=dev)
                attn_t = torch.tensor(attn, dtype=torch.long, device=dev)
                pos_ids_t = (attn_t.cumsum(dim=-1) - 1).clamp(min=0)
                captured.clear()
                model(
                    input_ids=input_ids_t,
                    attention_mask=attn_t,
                    position_ids=pos_ids_t,
                    output_hidden_states=False,
                )

                for j, (row_i, info) in enumerate(batch):
                    pad = pad_offs[j]
                    rs, ee, span = info["prompt_len"], info["ext_end"], info["span"]
                    fids = info["full_ids"]
                    assert fids[:prefix_len] == full_ids_all[0][:prefix_len], (
                        f"[{arm_name}] id {ids[row_i]}: constant-prefix violation at "
                        f"prefix_len={prefix_len}"
                    )
                    # Realized next-token ids: next_ids_ext[j] = full_ids[rs + j],
                    # j = 0..span-1 (j = t for answer position t; j = 0 = c_last's).
                    next_ext = np.asarray(fids[rs:ee], dtype=np.int64)
                    next_rows.append(next_ext[1:].astype(np.int32))  # per-position (t=1..span-1)
                    unique_next_ids.update(int(x) for x in next_ext.tolist())
                    pos, valid = parent952._slot_positions_and_validity(rs, ee, span)
                    slot_next_ids[row_i] = slot_next_token_ids(pos, valid, rs, span, next_ext)
                    ok_next = slot_next_ids[row_i] >= 0

                    u_rows = u_dir[torch.from_numpy(next_ext[1:]).to(dev)]  # (span-1, H) f32
                    u_hat = u_rows / u_rows.norm(dim=1, keepdim=True)

                    for li in range(n_layers):
                        la = layers[li]
                        hs = captured[li][j]  # (T_padded, H) GPU
                        span_hs = hs[pad + rs : pad + ee].float()  # (span, H)
                        alpha = (span_hs[: span - 1] * u_hat).sum(dim=1)  # (span-1,)
                        alpha_pos_rows[li].append(alpha.to(torch.float16).cpu().numpy())
                        if span >= T2 + 2:  # rem over t=17..span-1 needs span >= 18
                            sl = slice(T2, span - 1)
                            rem_par_v = (alpha[sl, None] * u_hat[sl]).mean(0)
                            rem_nx_v = span_hs[T2 : span - 1].mean(0)
                            rem_par[li, row_i] = rem_par_v.to(torch.float16).cpu().numpy()
                            rem_nx[li, row_i] = rem_nx_v.to(torch.float16).cpu().numpy()
                            if li == 0:
                                rem_valid[row_i] = True
                        full_par_v = (alpha[:, None] * u_hat).mean(0)
                        full_par[li, row_i] = full_par_v.to(torch.float16).cpu().numpy()

                        # Single-position slot activations (46,) — parent gather.
                        idx = torch.from_numpy(pos + pad).clamp(min=0).to(dev)
                        single = hs[idx].float()  # (46, H)
                        valid_t = torch.from_numpy(valid).to(dev)
                        single[~valid_t] = float("nan")
                        # Per-slot alpha (NaN where no valid next token).
                        okn = torch.from_numpy(ok_next).to(dev)
                        nid = torch.from_numpy(slot_next_ids[row_i]).clamp(min=0).to(dev)
                        u_s = u_dir[nid]
                        u_s = u_s / u_s.norm(dim=1, keepdim=True)
                        a_s = (single * u_s).sum(dim=1)
                        a_s[~okn] = float("nan")
                        alpha_slots[li, row_i] = a_s.float().cpu().numpy()
                        if p_last is not None:
                            p_last[li, row_i] = (
                                hs[pad + prefix_len - 1].float().to(torch.float16).cpu().numpy()
                            )
                        # g2/g3 vs the staged parent shard (production).
                        if staged_slots_by_layer is not None:
                            staged_row = staged_slots_by_layer[la][row_i]  # (72, H) fp16 torch cpu
                            st = staged_row[:46].to(dev).float()
                            both = valid_t & torch.isfinite(st).all(dim=1)
                            if both.any():
                                num = (single[both] * st[both]).sum(dim=1)
                                den = single[both].norm(dim=1) * st[both].norm(dim=1) + 1e-9
                                cos = (num / den).cpu().numpy()
                                g2[la]["n_cells"] += int(both.sum())
                                g2[la]["n_below"] += int((cos < COS_GATE_MIN).sum())
                                g2[la]["min_cos"] = min(g2[la]["min_cos"], float(cos.min()))
                            if span >= T2 + 1:
                                rec_full = span_hs[T2:span].mean(0)  # FULL range t=17..span
                                st_rem = staged_row[SLOT_IDX[REM_FULL_SLOT]].to(dev).float()
                                if bool(torch.isfinite(st_rem).all()):
                                    c = float(
                                        (rec_full * st_rem).sum()
                                        / (rec_full.norm() * st_rem.norm() + 1e-9)
                                    )
                                    g3[la]["n_cells"] += 1
                                    g3[la]["n_below"] += int(c < COS_GATE_MIN)
                                    g3[la]["min_cos"] = min(g3[la]["min_cos"], c)
                        # Parent-schema 72-slot assembly (smoke stage only).
                        if parent_slots is not None:
                            pool_valid = parent952._pool_slot_validity(span)
                            cums = span_hs.cumsum(0)
                            total = cums[-1]
                            prompt_sum = hs[pad : pad + rs].float().sum(0)
                            rev_cummax = torch.flip(span_hs, dims=[0]).cummax(0).values
                            pooled = torch.full((72 - 46, hid), float("nan"), device=dev)
                            for ti, t in enumerate(parent952.PREFIX_TS):
                                if pool_valid[f"rem_mean_gt{t}"]:
                                    pooled[ti] = (total - cums[t - 1]) / float(span - t)
                                    pooled[8 + ti] = rev_cummax[span - t - 1]
                                if pool_valid[f"pooled_prefix_le{t}"]:
                                    pooled[16 + ti] = (prompt_sum + cums[t - 1]) / float(rs + t)
                            pooled[24] = total / float(span)
                            pooled[25] = cums[span - 1] / float(span)  # mean_823 (span_823=span)
                            parent_slots[row_i, li] = (
                                torch.cat([single, pooled], dim=0).to(torch.float16).cpu().numpy()
                            )
                captured.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batches_done = b0 // batch_size + 1
                if batches_done % 25 == 0 or batches_done == 1:
                    logger.info("[capture:%s] %d/%d rows", arm_name, b0 + len(batch), len(prepped))
                # K4 capture pilot: first 2 batches of the FIRST arm, projected
                # over ALL arms at the run's own execution shape (batch=%d).
                if pilot is not None and pilot.get("armed") and batches_done == 2:
                    pilot["armed"] = False
                    assert base_dir is not None
                    _pilot_check(
                        "capture",
                        time.time() - t_arm0,
                        units_done=2,
                        units_total=pilot["total_batches"],
                        booked_h=CAPTURE_BOOKED_H,
                        base_dir=base_dir,
                        smoke=smoke,
                        execution_shape=f"batched TF forward, batch_size={batch_size}",
                    )
    finally:
        for h in handles:
            h.remove()
        captured.clear()

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, a in enumerate(next_rows):
        offsets[i + 1] = offsets[i] + len(a)
    next_flat = (np.concatenate(next_rows) if offsets[-1] else np.zeros(0, dtype=np.int32)).astype(
        np.int32
    )
    alpha_flat_by_layer = [
        (np.concatenate(alpha_pos_rows[li]) if offsets[-1] else np.zeros(0, dtype=np.float16))
        for li in range(n_layers)
    ]
    return {
        "spans": spans,
        "prefix_len": prefix_len,
        "rem_par": rem_par,
        "rem_nx": rem_nx,
        "full_par": full_par,
        "alpha_slots": alpha_slots,
        "alpha_flat_by_layer": alpha_flat_by_layer,
        "p_last": p_last,
        "next_flat": next_flat,
        "next_offsets": offsets,
        "slot_next_ids": slot_next_ids,
        "rem_valid": rem_valid,
        "parent_slots": parent_slots,
        "g2": g2,
        "g3": g3,
        "unique_next_ids": unique_next_ids,
        "wall_s": time.time() - t_arm0,
    }


def _save_decomp_shards(
    base_dir: pathlib.Path,
    arm: str,
    layers: tuple[int, ...],
    ids: list[int],
    cap: dict[str, Any],
) -> list[pathlib.Path]:
    """Persist the decomposed store (per plan §6.5 deliverable 2)."""
    import torch

    out_dir = _tensors_dir(base_dir)
    paths: list[pathlib.Path] = []
    for li, la in enumerate(layers):
        payload = {
            "rem_par": torch.from_numpy(np.ascontiguousarray(cap["rem_par"][li])),
            "rem_nx": torch.from_numpy(np.ascontiguousarray(cap["rem_nx"][li])),
            "full_par": torch.from_numpy(np.ascontiguousarray(cap["full_par"][li])),
            "alpha_slots": torch.from_numpy(np.ascontiguousarray(cap["alpha_slots"][li])),
            "alpha_pos_flat": torch.from_numpy(cap["alpha_flat_by_layer"][li]),
            "rem_valid": torch.from_numpy(cap["rem_valid"]),
            "ids": list(ids),
            "slot_names_46": list(SLOT_NAMES[:46]),
            "layer": int(la),
        }
        if cap["p_last"] is not None:
            payload["p_last"] = torch.from_numpy(np.ascontiguousarray(cap["p_last"][li]))
        p = out_dir / f"decomp_{arm}_L{la}.pt"
        torch.save(payload, str(p))
        paths.append(p)
    npz_p = out_dir / f"next_ids_{arm}.npz"
    np.savez(
        npz_p,
        next_flat=cap["next_flat"],
        next_offsets=cap["next_offsets"],
        slot_next_ids=cap["slot_next_ids"],
        ids=np.asarray(ids, dtype=np.int64),
        prefix_len=np.asarray([cap["prefix_len"]], dtype=np.int64),
    )
    paths.append(npz_p)
    return paths


def _save_parent_shards_smoke(
    base_dir: pathlib.Path,
    arm: str,
    layers: tuple[int, ...],
    ids: list[int],
    cap: dict[str, Any],
) -> None:
    """Smoke stage: write parent-SCHEMA slot shards + spans (battery read contract)."""
    import torch

    out_dir = _tensors_dir(base_dir)
    for li, la in enumerate(layers):
        torch.save(
            {
                "slots": torch.from_numpy(np.ascontiguousarray(cap["parent_slots"][:, li])),
                "ids": list(ids),
                "slot_names": list(SLOT_NAMES),
                "layer": int(la),
            },
            str(out_dir / f"slots_{arm}_L{la}.pt"),
        )
    (out_dir / f"spans_{arm}.json").write_text(json.dumps(cap["spans"], indent=2, default=_json_np))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A — stage (plan §4.4 Phase A)
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_SHARD_KEYS = frozenset({"slots", "ids", "slot_names", "layer"})


def phase_stage(base_dir: pathlib.Path, smoke: bool, layers: tuple[int, ...]) -> dict[str, Any]:
    """Production: #952 phase0_verify replay (pool + prompts + split + #823
    byte-table asserts) + scoped per-file staging of the 16 slot shards and 4
    span files at PARENT_TENSOR_REVISION (never snapshot_download — the
    ~1M-file data repo full-tree-enumerates, gotchas.md). Consumers open the
    exact fetch destinations — no layout mapping (reuse leg (h)(iv) N/A).

    Smoke: synth pool/prompts/texts + a synth-capture through the SAME capture
    function (tiny model) writes the parent-schema shards.
    """
    log_phase("stage")
    if smoke:
        pool_ids, prompts, texts = _smoke_pool_and_texts()
        d = base_dir / "data" / "issue_1072"
        d.mkdir(parents=True, exist_ok=True)
        (d / "smoke_pool.json").write_text(
            json.dumps({"pool_ids": pool_ids, "prompts": prompts, "texts": texts}, default=_json_np)
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        model = _build_smoke_model(tokenizer)
        u_dir = _unembed_dir(model)
        for arm in ARMS:
            cap = tf_capture_decomp_arm(
                model,
                tokenizer,
                pool_ids,
                prompts,
                texts[arm],
                arm,
                layers,
                u_dir,
                staged_slots_by_layer=None,
                staged_spans=None,
                batch_size=4,
                emit_parent_shards=True,
                base_dir=base_dir,
                smoke=True,
            )
            _save_parent_shards_smoke(base_dir, arm, layers, pool_ids, cap)
        rec = {"n_pool": len(pool_ids), "synth": True, **_run_metadata(smoke, layers)}
        (eval_out_dir(base_dir) / "stage_manifest.json").write_text(
            json.dumps(rec, indent=2, default=_json_np)
        )
        log_phase("stage_done")
        return {"pool_ids": pool_ids, "prompts": prompts, "texts": texts}

    import torch

    from explore_persona_space.orchestrate import hub as eps_hub

    rec0 = parent952.phase0_verify(base_dir, smoke=False)
    pool_ids = rec0["pool_ids"]
    tensors_dir = _tensors_dir(base_dir)
    staged: list[str] = []
    for arm in ARMS:
        for name in [f"slots_{arm}_L{la}.pt" for la in layers] + [f"spans_{arm}.json"]:
            target = tensors_dir / name
            eps_hub.stage_hub_file(
                HF_DATA_REPO,
                f"{PARENT_SLUG}/analysis_tensors/{name}",
                target,
                repo_type="dataset",
                revision=PARENT_TENSOR_REVISION,
            )
            assert target.stat().st_size > 0, target
            staged.append(name)
    # Pre-registered realized-keys check (reuse check (c), plan §12 assumption 1):
    # mmap header read of ONE shard — keys + shape, no tensor materialization.
    probe = torch.load(
        str(tensors_dir / f"slots_own_L{layers[0]}.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    assert set(probe.keys()) == STAGE_SHARD_KEYS, sorted(probe.keys())
    assert probe["slots"].shape == (len(pool_ids), 72, PROD_HIDDEN), probe["slots"].shape
    assert probe["slots"].dtype == torch.float16, probe["slots"].dtype
    assert probe["slot_names"] == list(SLOT_NAMES), "slot registry drift"
    assert [int(i) for i in probe["ids"]] == [int(i) for i in pool_ids], (
        "staged shard id order != analysis pool order"
    )
    del probe
    gc.collect()
    rec = {
        "n_pool": len(pool_ids),
        "staged_files": staged,
        "staging_revision": PARENT_TENSOR_REVISION,
        "realized_keys_check": "PASS",
        **_run_metadata(smoke, layers),
    }
    (eval_out_dir(base_dir) / "stage_manifest.json").write_text(
        json.dumps(rec, indent=2, default=_json_np)
    )
    log_phase("stage_done")
    prompts = json.loads((base_dir / "data" / "issue_952" / "prompts.json").read_text())
    texts = parent952.load_arm_texts(base_dir, pool_ids)
    return {"pool_ids": pool_ids, "prompts": {c: prompts[c] for c in pool_ids}, "texts": texts}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B — decomposition re-capture (plan §4.4 Phase B + B' upload)
# ═══════════════════════════════════════════════════════════════════════════════


def _pool_ids_from_shard(base_dir: pathlib.Path, layer: int) -> list[int]:
    """Pool ids in shard row order (the row-alignment source of truth)."""
    import torch

    p = _tensors_dir(base_dir) / f"slots_own_L{layer}.pt"
    assert p.exists(), f"staged shard missing (run --phase stage first): {p}"
    d = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
    ids = [int(i) for i in d["ids"]]
    del d
    gc.collect()
    return ids


def _load_capture_inputs(
    base_dir: pathlib.Path, smoke: bool, layers: tuple[int, ...]
) -> tuple[list[int], dict[int, str], dict[str, dict[int, str]]]:
    """Pool + prompts + per-arm answer texts, re-loadable per phase (resume-safe)."""
    if smoke:
        d = json.loads((base_dir / "data" / "issue_1072" / "smoke_pool.json").read_text())
        pool_ids = [int(i) for i in d["pool_ids"]]
        prompts = {int(k): v for k, v in d["prompts"].items()}
        texts = {arm: {int(k): v for k, v in d["texts"][arm].items()} for arm in ARMS}
        return pool_ids, prompts, texts
    pool_ids = _pool_ids_from_shard(base_dir, layers[0])
    prompts_all = json.loads((base_dir / "data" / "issue_952" / "prompts.json").read_text())
    prompts = {cid: prompts_all[cid] for cid in pool_ids}
    texts = parent952.load_arm_texts(base_dir, pool_ids)
    return pool_ids, prompts, texts


def _capture_regime(
    smoke: bool, layers: tuple[int, ...], pool_ids: list[int], batch_size: int
) -> dict[str, Any]:
    """Output-affecting resume keys (#722 r3: EVERY regime key in the manifest).

    batch_size is output-affecting: LEFT-pad batched bf16 numerics depend on
    batch composition; git_sha invalidates a resume across code changes
    (conservative — the parent's gate-5 convention).
    """
    return {
        "smoke": bool(smoke),
        "layers": list(layers),
        "n_pool": len(pool_ids),
        "pool_sha": hashlib.sha256(json.dumps([int(i) for i in pool_ids]).encode()).hexdigest(),
        "batch_size": int(batch_size),
        "model_id": "smoke-tiny-qwen2(seed 0)" if smoke else DEFAULT_MODEL,
        "staging_revision": "smoke-synth" if smoke else PARENT_TENSOR_REVISION,
        "git_sha": _git_sha(),
    }


def phase_capture(  # noqa: C901 — the phase-B driver: gates + 4 arms + uploads
    base_dir: pathlib.Path,
    smoke: bool,
    layers: tuple[int, ...],
    batch_size: int,
    skip_upload: bool,
) -> np.ndarray:
    """Phase B: per-arm decomposition capture with in-stream gates g1/g2/g3,
    per-arm checkpoint + incremental upload (checkpoint-per-phase rule), then
    the B' store upload BEFORE phase C. Returns U_dir (V, H) fp32 numpy."""
    import torch

    log_phase("capture")
    pool_ids, prompts, texts = _load_capture_inputs(base_dir, smoke, layers)
    tensors_dir = _tensors_dir(base_dir)
    regime = _capture_regime(smoke, layers, pool_ids, batch_size)

    if smoke:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        model = _build_smoke_model(tokenizer)
    else:
        model, tokenizer = _load_production_model(DEFAULT_MODEL)
    u_dir = _unembed_dir(model)

    # Staged references for g1/g2/g3 (mmap — rows are indexed lazily).
    staged_spans = {arm: parent952._load_spans(base_dir, arm) for arm in ARMS}
    staged_slots: dict[str, dict[int, Any]] = {}
    for arm in ARMS:
        staged_slots[arm] = {}
        for la in layers:
            d = torch.load(
                str(tensors_dir / f"slots_{arm}_L{la}.pt"),
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            assert [int(i) for i in d["ids"]] == [int(i) for i in pool_ids], (
                f"staged shard id drift ({arm}, L{la})"
            )
            staged_slots[arm][la] = d["slots"]

    n_batches_total = len(ARMS) * ((len(pool_ids) + batch_size - 1) // batch_size)
    pilot = {"armed": True, "total_batches": n_batches_total}
    gates: dict[str, Any] = {"g1": {}, "g2": {}, "g3": {}, "prefix": {}, "arm_wall_s": {}}
    unique_next: set[int] = set()
    t0 = time.time()
    for arm in ARMS:
        manifest_p = tensors_dir / f"decomp_manifest_{arm}.json"
        shard_paths = [tensors_dir / f"decomp_{arm}_L{la}.pt" for la in layers] + [
            tensors_dir / f"next_ids_{arm}.npz"
        ]
        if manifest_p.exists() and all(p.exists() for p in shard_paths):
            persisted = json.loads(manifest_p.read_text())
            if persisted.get("regime") == regime:
                logger.info("[capture] SKIP %s (per-arm resume, regime match)", arm)
                gates["g1"][arm] = persisted.get("g1")
                gates["g2"][arm] = persisted.get("g2")
                gates["g3"][arm] = persisted.get("g3")
                pilot["armed"] = False  # a resumed run's first batches are not a pilot basis
                continue
            logger.warning("[capture] %s manifest regime mismatch — recomputing", arm)
        cap = tf_capture_decomp_arm(
            model,
            tokenizer,
            pool_ids,
            prompts,
            texts[arm],
            arm,
            layers,
            u_dir,
            staged_slots_by_layer=staged_slots[arm],
            staged_spans=staged_spans[arm],
            batch_size=batch_size,
            emit_parent_shards=False,
            pilot=pilot,
            base_dir=base_dir,
            smoke=smoke,
        )
        paths = _save_decomp_shards(base_dir, arm, layers, pool_ids, cap)
        gates["g1"][arm] = {"n_checked": len(pool_ids), "n_mismatch": 0}
        gates["g2"][arm] = {str(k): v for k, v in cap["g2"].items()}
        gates["g3"][arm] = {str(k): v for k, v in cap["g3"].items()}
        gates["prefix"][arm] = {"prefix_len": cap["prefix_len"]}
        gates["arm_wall_s"][arm] = cap["wall_s"]
        unique_next |= cap["unique_next_ids"]
        manifest_p.write_text(
            json.dumps(
                {
                    "regime": regime,
                    "g1": gates["g1"][arm],
                    "g2": gates["g2"][arm],
                    "g3": gates["g3"][arm],
                    **_run_metadata(smoke, layers),
                },
                indent=2,
                default=_json_np,
            )
        )
        if not skip_upload:
            _hf_commit_files_1072(f"decomp store ({arm})", [*paths, manifest_p], base_dir)

    # ── K2: pooled g2/g3 verdict across armsxlayers ─────────────────────────────
    tot_cells = tot_below = 0
    min_cos = 1.0
    for gname in ("g2", "g3"):
        for _arm, per_layer in gates[gname].items():
            if not per_layer:
                continue
            for _la, rec in per_layer.items():
                tot_cells += int(rec["n_cells"])
                tot_below += int(rec["n_below"])
                min_cos = min(min_cos, float(rec["min_cos"]))
    below_frac = (tot_below / tot_cells) if tot_cells else 0.0
    gates["k2"] = {
        "n_cells": tot_cells,
        "n_below": tot_below,
        "below_frac": below_frac,
        "min_cos": min_cos,
        "cos_floor": COS_GATE_MIN,
        "max_below_frac": COS_GATE_MAX_BELOW_FRAC,
        "verdict": "PASS" if below_frac <= COS_GATE_MAX_BELOW_FRAC else "FAIL",
    }

    # ── p_last constancy (plan §3.5(3): degenerate by construction — reported) ──
    p_const: dict[str, float] = {}
    for la in layers:
        dp = torch.load(
            str(tensors_dir / f"decomp_own_L{la}.pt"), map_location="cpu", weights_only=False
        )
        pl = dp["p_last"].float().numpy()
        dev = np.linalg.norm(pl - pl[0][None, :], axis=1)
        p_const[str(la)] = float(dev.max())
        del dp
    gates["p_last_constancy_max_l2"] = p_const

    # ── raw-vs-folded basis sanity read: cos(W_U[y], gamma⊙W_U[y]) per unique id ────
    ids_arr = torch.tensor(sorted(unique_next), dtype=torch.long, device=u_dir.device)
    if len(ids_arr):
        with torch.no_grad():
            w_raw = model.lm_head.weight[ids_arr].float()
            u_rows = u_dir[ids_arr]
            cos = (
                ((w_raw * u_rows).sum(1) / (w_raw.norm(dim=1) * u_rows.norm(dim=1) + 1e-9))
                .cpu()
                .numpy()
            )
        gates["raw_vs_folded_cos"] = {
            "n_unique_next_ids": len(ids_arr),
            "min": float(cos.min()),
            "mean": float(cos.mean()),
            "p01": float(np.percentile(cos, 1)),
            "p50": float(np.percentile(cos, 50)),
        }
        np.savez(
            tensors_dir / "raw_vs_folded_cos.npz",
            token_ids=ids_arr.cpu().numpy().astype(np.int64),
            cos=cos.astype(np.float32),
        )
    gates["total_wall_s"] = time.time() - t0
    gates.update(_run_metadata(smoke, layers))
    gates_p = eval_out_dir(base_dir) / "capture_gates.json"
    gates_p.write_text(json.dumps(gates, indent=2, default=_json_np))
    if not skip_upload:
        _hf_commit_files_1072(
            "capture gates + basis read", [gates_p, tensors_dir / "raw_vs_folded_cos.npz"], base_dir
        )
    if gates["k2"]["verdict"] == "FAIL":
        if smoke:
            logger.warning("[k2] smoke-demoted equivalence verdict: %s", gates["k2"])
        else:
            raise RuntimeError(f"g2/g3 equivalence FAIL (K2): {gates['k2']}")

    u_np = u_dir.cpu().numpy()
    del model, u_dir, staged_slots
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_phase("capture_done")
    return u_np


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C — component ridge battery (plan §4.4 Phase C)
# ═══════════════════════════════════════════════════════════════════════════════


class ReproRefs:
    """g4/g5 reference provider.

    Production: the parent's committed k-fold artifacts under
    ``eval_results/issue_952/kfold-decision-cells/`` (λ tables, per-fold
    by_layer reports, closure layer records, fold-assignment hashes).
    Smoke: SELF-GENERATED references (write-on-first-compute, compare through
    the SAME comparison code path) — the production-n 1e-6 reproduction bar is
    structurally unsatisfiable against artifacts that don't cover the synth
    substrate (the #1345 gate-calibration lesson), so smoke demotes the
    REFERENCE, never the comparison code.
    """

    def __init__(self, base_dir: pathlib.Path, smoke: bool) -> None:
        self.smoke = smoke
        if smoke:
            self.dir = eval_out_dir(base_dir) / "smoke_reference"
            self.dir.mkdir(parents=True, exist_ok=True)
        else:
            self.root = parent952.repo_root().joinpath(*COMMITTED_KFOLD_DIR)
            stats = json.loads((self.root / "stats_kfold.json").read_text())
            self.lambda_tables = stats["manifest_validation"]["lambda_tables_by_fold"]
            self.fold_assignment = json.loads((self.root / "fold_assignment.json").read_text())

    def get_or_record(self, fold: int, kind: str, computed: dict) -> dict:
        """Smoke self-reference: first compute writes, later runs compare."""
        assert self.smoke
        p = self.dir / f"fold{fold}_{kind}.json"
        if p.exists():
            return json.loads(p.read_text())
        p.write_text(json.dumps(computed, indent=2, default=_json_np))
        return computed

    def fold_hashes(self, fold: int, computed: dict) -> dict:
        if self.smoke:
            return self.get_or_record(fold, "fold_hashes", computed)
        return self.fold_assignment["split_sha256_by_fold"][str(fold)]

    def h1_ref(self, fold: int, layer: int, computed: dict) -> dict:
        """{"lambda_table": {slot: λ}, "by_layer": {arm: {slot: {...}}}}"""
        if self.smoke:
            return self.get_or_record(fold, f"h1_L{layer}", computed)
        rep = json.loads(
            (self.root / f"position_r2_by_arm_cross_layer_fold{fold}.json").read_text()
        )
        return {
            "lambda_table": self.lambda_tables[str(fold)][str(layer)],
            "by_layer": rep["by_layer"][str(layer)],
        }

    def closure_ref(self, fold: int, layer: int, computed: dict) -> dict:
        if self.smoke:
            return self.get_or_record(fold, f"closure_L{layer}", computed)
        rep = json.loads((self.root / f"prefix_closure_by_arm_fold{fold}.json").read_text())
        return rep["matched_contrasts"][f"t{T2}"][f"L{layer}"]


def _g4_compare(prefix: str, computed: Any, ref: Any, mismatches: list[str]) -> None:
    """Recursive g4 comparison: λ-leaves exact (rel 1e-12), counts exact,
    R²-leaves |Δ| <= REPRO_TOL. Walks the REFERENCE keys (a ref key missing
    from the recomputation is itself a mismatch)."""
    if isinstance(ref, dict):
        if not isinstance(computed, dict):
            mismatches.append(f"{prefix}: computed is not a dict")
            return
        for k, v in ref.items():
            if k not in computed:
                mismatches.append(f"{prefix}/{k}: missing from recomputation")
                continue
            _g4_compare(f"{prefix}/{k}", computed[k], v, mismatches)
        return
    leaf = prefix.rsplit("/", 1)[-1]
    try:
        got, want = float(computed), float(ref)
    except (TypeError, ValueError):
        if computed != ref:
            mismatches.append(f"{prefix}: {computed!r} != {ref!r}")
        return
    if "lambda" in prefix:  # λ leaves AND lambda_table/{slot} leaves — exact match
        if not np.isclose(got, want, rtol=1e-12, atol=0.0):
            mismatches.append(f"{prefix}: lambda {got!r} != {want!r}")
    elif leaf.startswith("n_"):
        if int(got) != int(want):
            mismatches.append(f"{prefix}: count {got} != {want}")
    else:
        if not (np.isfinite(got) and np.isfinite(want)):
            if np.isfinite(got) != np.isfinite(want):
                mismatches.append(f"{prefix}: finiteness {got} vs {want}")
        elif abs(got - want) > REPRO_TOL:
            mismatches.append(f"{prefix}: |{got:.9f} - {want:.9f}| > {REPRO_TOL}")


def _stack_targets_h(
    slots_by_arm: dict[str, np.ndarray],
    rows: np.ndarray,
    groups: list[tuple[str, str]],
    hid: int,
) -> np.ndarray:
    """(n_rows, G, H) fp16 target stack (parent `_stack_targets`, hid-parametrized
    so the tiny-model smoke rides the identical battery path)."""
    out = np.full((len(rows), len(groups), hid), np.nan, dtype=np.float16)
    for gi, (slot, arm) in enumerate(groups):
        if arm in slots_by_arm:
            out[:, gi, :] = slots_by_arm[arm][rows][:, SLOT_IDX[slot], :]
    return out


def _battery_ckpt_dir(base_dir: pathlib.Path, regime: dict) -> pathlib.Path:
    """Per-(fold, layer) unit checkpoint dir with a fail-loud regime manifest.

    A regime mismatch QUARANTINES the stale dir (rename, never delete — the
    crash-fix stale-artifact disposition default) and starts fresh.
    """
    d = _tensors_dir(base_dir) / "battery1072_ckpt"
    reg_p = d / "regime.json"
    if d.exists() and reg_p.exists():
        persisted = json.loads(reg_p.read_text())
        if persisted != regime:
            stale = d.with_name(f"{d.name}-stale-{int(time.time())}")
            d.rename(stale)
            logger.warning("[battery] regime mismatch — quarantined stale ckpts to %s", stale)
    d.mkdir(parents=True, exist_ok=True)
    if not reg_p.exists():
        reg_p.write_text(json.dumps(regime, indent=2, default=_json_np))
    return d


def _battery_fold_layer(  # noqa: C901 — the phase-C unit: calibration + components + p_last
    base_dir: pathlib.Path,
    fold_split: dict,
    layer: int,
    pool_ids: list[int],
    spans_by_arm: dict[str, np.ndarray],
    u_dir_np: np.ndarray,
    refs: ReproRefs,
    fit_device: str,
    min_train: int,
    smoke: bool,
    run_parity: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One (fold, layer) unit: calibration full cells (g4) + component cells at
    frozen λ* + p_last prefix-arm cells. Returns (npz arrays, unit record)."""
    import torch

    fold_k = int(fold_split["fold"])
    tensors_dir = _tensors_dir(base_dir)
    hid = u_dir_np.shape[1]
    t_unit0 = time.time()

    pos_of = {cid: i for i, cid in enumerate(pool_ids)}
    tr_pos = np.asarray([pos_of[c] for c in fold_split["train"] if c in pos_of])
    va_pos = np.asarray([pos_of[c] for c in fold_split["val"] if c in pos_of])
    te_pos = np.asarray([pos_of[c] for c in fold_split["test"] if c in pos_of])
    u16 = np.all(np.stack([spans_by_arm[a] >= T2 + 16 for a in ARMS]), axis=0)
    tr_m = tr_pos[u16[tr_pos]]
    va_m = va_pos[u16[va_pos]]
    te_m = te_pos[u16[te_pos]]
    rec: dict[str, Any] = {
        "fold": fold_k,
        "layer": layer,
        "matched_n": {"train": len(tr_m), "val": len(va_m), "test": len(te_m)},
    }
    if len(tr_m) < min_train or len(te_m) < 2 or len(va_m) < 2:
        rec["skipped"] = True
        return {}, rec

    staged = {arm: parent952._load_layer_slots(base_dir, arm, layer)[0] for arm in ARMS}
    decomp = {
        arm: torch.load(
            str(tensors_dir / f"decomp_{arm}_L{layer}.pt"), map_location="cpu", weights_only=False
        )
        for arm in ARMS
    }
    next_npz = {arm: np.load(tensors_dir / f"next_ids_{arm}.npz") for arm in ARMS}
    for arm in ARMS:
        assert [int(i) for i in decomp[arm]["ids"]] == [int(i) for i in pool_ids], (
            f"decomp shard id drift ({arm}, L{layer})"
        )
        # Row coverage (plan §3): every matched row's decomposition is valid.
        assert bool(decomp[arm]["rem_valid"].numpy()[u16].all()), (
            f"matched-universe row with invalid remainder decomposition ({arm}, L{layer})"
        )

    rows_by_split = {"train": tr_m, "val": va_m, "test": te_m}
    x_c = staged["own"][:, SLOT_IDX["c_last"], :]

    # ── g4 item 1a: H1 FULL-cell calibration re-run (13-λ, GROUPS_A) ────────────
    groups_a = parent952.GROUPS_A
    res_a = run_ridge_cell(
        x_c[tr_m],
        _stack_targets_h(staged, tr_m, groups_a, hid),
        {
            "val": (x_c[va_m], _stack_targets_h(staged, va_m, groups_a, hid)),
            "test": (x_c[te_m], _stack_targets_h(staged, te_m, groups_a, hid)),
        },
        group_names=[f"{s}|{a}" for s, a in groups_a],
        device=fit_device,
        allow_train_nan_imputation=True,
    )
    lam_idx_a, lam_by_slot = parent952._lam_star_by_slot(groups_a, res_a.pooled["val"])
    pooled_test_a = parent952._pooled_at_frozen(res_a, "test", lam_idx_a)
    computed_h1 = {
        "lambda_table": {slot: DEFAULT_LAMBDAS_LIST[lam_by_slot[slot]] for slot in POSITION_SLOTS},
        "by_layer": {},
    }
    for gi, (slot, arm) in enumerate(groups_a):
        if slot not in POSITION_SLOTS:
            continue
        computed_h1["by_layer"].setdefault(arm, {})[slot] = {
            "test_pooled_r2": float(pooled_test_a[gi]),
            "lambda": DEFAULT_LAMBDAS_LIST[int(lam_idx_a[gi])],
            "n_valid_test": int(res_a.n_valid["test"][gi]),
        }
    mismatches: list[str] = []
    _g4_compare(
        f"h1_fold{fold_k}_L{layer}",
        computed_h1,
        refs.h1_ref(fold_k, layer, computed_h1),
        mismatches,
    )

    # ── g4 item 1b: H2 c-leg / z-leg calibration cells (parent shapes) ──────────
    groups_c = [(f"rem_mean_gt{T2}", a) for a in ARMS] + [(f"rem_max_gt{T2}", a) for a in ARMS]
    res_c = run_ridge_cell(
        x_c[tr_m],
        _stack_targets_h(staged, tr_m, groups_c, hid),
        {
            "val": (x_c[va_m], _stack_targets_h(staged, va_m, groups_c, hid)),
            "test": (x_c[te_m], _stack_targets_h(staged, te_m, groups_c, hid)),
        },
        group_names=[f"{s}|{a}" for s, a in groups_c],
        device=fit_device,
    )
    x_slot = parent952.prefix_slot_name(T2)
    zres: dict[str, Any] = {}
    for arm in MATCHED_ARMS:
        xa = staged[arm][:, SLOT_IDX[x_slot], :]
        zres[arm] = run_ridge_cell(
            xa[tr_m],
            _stack_targets_h(
                {arm: staged[arm]},
                tr_m,
                [(f"rem_mean_gt{T2}", arm), (f"rem_max_gt{T2}", arm)],
                hid,
            ),
            {
                "val": (
                    xa[va_m],
                    _stack_targets_h(
                        {arm: staged[arm]},
                        va_m,
                        [(f"rem_mean_gt{T2}", arm), (f"rem_max_gt{T2}", arm)],
                        hid,
                    ),
                ),
                "test": (
                    xa[te_m],
                    _stack_targets_h(
                        {arm: staged[arm]},
                        te_m,
                        [(f"rem_mean_gt{T2}", arm), (f"rem_max_gt{T2}", arm)],
                        hid,
                    ),
                ),
            },
            group_names=[f"rem_mean_gt{T2}|{arm}", f"rem_max_gt{T2}|{arm}"],
            device=fit_device,
        )
    layer_rec: dict[str, Any] = {}
    li_by_target: dict[str, dict[str, int]] = {}
    for ti, target in enumerate(("mean", "max")):
        c_cols = [gi for gi, (s, _a) in enumerate(groups_c) if s == f"rem_{target}_gt{T2}"]
        li_c = int(np.nanargmax(np.nanmean(res_c.pooled["val"][:, c_cols], axis=1)))
        li_z = int(
            np.nanargmax(
                np.nanmean(np.stack([zres[a].pooled["val"][:, ti] for a in MATCHED_ARMS]), axis=0)
            )
        )
        li_by_target[target] = {"cleg": li_c, "zleg": li_z}
        for gi, (s, a) in enumerate(groups_c):
            if s == f"rem_{target}_gt{T2}":
                layer_rec[f"cleg|{a}|{target}"] = float(res_c.pooled["test"][li_c, gi])
        for a in MATCHED_ARMS:
            layer_rec[f"zleg|{a}|{target}"] = float(zres[a].pooled["test"][li_z, ti])
        layer_rec[f"lambda_cleg_{target}"] = DEFAULT_LAMBDAS_LIST[li_c]
        layer_rec[f"lambda_zleg_{target}"] = DEFAULT_LAMBDAS_LIST[li_z]
    for ext in ("ext_plain", "ext_style"):
        g0 = layer_rec["cleg|own|mean"] - layer_rec[f"cleg|{ext}|mean"]
        gt = layer_rec["zleg|own|mean"] - layer_rec[f"zleg|{ext}|mean"]
        layer_rec[f"G_matched_0_{ext}"] = g0
        layer_rec[f"G_matched_t_{ext}"] = gt
        layer_rec[f"delta_G_{ext}"] = g0 - gt
    ref_closure = refs.closure_ref(fold_k, layer, layer_rec)
    _g4_compare(f"closure_fold{fold_k}_L{layer}", layer_rec, ref_closure, mismatches)
    if mismatches:
        rec["g4_mismatches"] = mismatches[:20]
        (eval_out_dir(base_dir) / f"g4_abort_fold{fold_k}_L{layer}.json").write_text(
            json.dumps({"mismatches": mismatches}, indent=2, default=_json_np)
        )
        raise RuntimeError(
            f"g4 calibration reproduction FAIL (K3) at fold {fold_k} L{layer}: "
            f"{len(mismatches)} mismatches, first: {mismatches[0]}"
        )
    rec["g4"] = {"verdict": "PASS", "n_compared": "h1 42 slots x 4 arms + closure layer_rec"}
    rec["closure_layer_rec"] = layer_rec

    # ── component target providers (fp64; par from staged z + û; perp derived) ──
    def _slot_pair(slot: str, arm: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_full = staged[arm][rows][:, SLOT_IDX[slot], :].astype(np.float64)
        nids = next_npz[arm]["slot_next_ids"][rows, SLOT_IDX[slot]]
        assert (nids >= 0).all(), (slot, arm, "matched row without a realized next token")
        u = u_dir_np[nids].astype(np.float64)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        alpha = np.einsum("ij,ij->i", y_full, u)
        return alpha[:, None] * u, y_full

    def _rem_pair(arm: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_par = decomp[arm]["rem_par"].numpy()[rows].astype(np.float64)
        y_full = decomp[arm]["rem_nx"].numpy()[rows].astype(np.float64)
        return y_par, y_full

    # H1 component cells: 41 decomposable slots x 4 arms at per-slot frozen λ.
    h1_groups = [(s, a) for s in DECOMP_SLOTS for a in ARMS]

    def h1_pair_fn(split: str, gi: int) -> tuple[np.ndarray, np.ndarray]:
        slot, arm = h1_groups[gi]
        return _slot_pair(slot, arm, rows_by_split[split])

    h1_lams = np.asarray(
        [DEFAULT_LAMBDAS_LIST[lam_by_slot[s]] for s, _a in h1_groups], dtype=np.float64
    )
    comp_h1 = run_component_cell(
        x_c[tr_m],
        {"test": x_c[te_m]},
        h1_pair_fn,
        [f"{s}|{a}" for s, a in h1_groups],
        h1_lams,
        device=fit_device,
    )

    # H2 c-leg remainder components (frozen λ = lambda_cleg_mean) + λ sensitivity.
    rem_groups = list(ARMS)

    def rem_pair_fn(split: str, gi: int) -> tuple[np.ndarray, np.ndarray]:
        return _rem_pair(rem_groups[gi], rows_by_split[split])

    lam_cleg = DEFAULT_LAMBDAS_LIST[li_by_target["mean"]["cleg"]]
    comp_cleg = run_component_cell(
        x_c[tr_m],
        {"test": x_c[te_m]},
        rem_pair_fn,
        [f"rem|{a}" for a in rem_groups],
        np.full(len(rem_groups), lam_cleg),
        device=fit_device,
        sensitivity_lambdas=np.asarray(DEFAULT_LAMBDAS_LIST),
    )
    # z-leg components per matched arm (frozen λ = lambda_zleg_mean, shared).
    lam_zleg = DEFAULT_LAMBDAS_LIST[li_by_target["mean"]["zleg"]]
    comp_zleg: dict[str, Any] = {}
    for arm in MATCHED_ARMS:
        xa = staged[arm][:, SLOT_IDX[x_slot], :]

        def z_pair_fn(split: str, gi: int, _arm: str = arm) -> tuple[np.ndarray, np.ndarray]:
            assert gi == 0
            return _rem_pair(_arm, rows_by_split[split])

        comp_zleg[arm] = run_component_cell(
            xa[tr_m],
            {"test": xa[te_m]},
            z_pair_fn,
            [f"rem|{arm}"],
            np.asarray([lam_zleg]),
            device=fit_device,
            sensitivity_lambdas=np.asarray(DEFAULT_LAMBDAS_LIST),
        )

    # ── parity gate (calibration fold, first layer): batched vs serial oracle ───
    if run_parity:
        cells = [
            (h1_groups.index(("f16_t1", "own")), "test"),
            (h1_groups.index(("l16_m3", "ext_plain")), "test"),
            (h1_groups.index(("d10_p55", "own")), "test"),
        ]
        rec["parity_h1"] = _component_parity_gate(
            x_c[tr_m],
            {"test": x_c[te_m]},
            h1_pair_fn,
            [f"{s}|{a}" for s, a in h1_groups],
            h1_lams,
            comp_h1,
            cells,
        )
        # One remainder cell against the oracle too (the H2 target family).
        ypar_tr, yfull_tr = rem_pair_fn("train", 0)
        ypar_te, yfull_te = rem_pair_fn("test", 0)
        oracle = serial_component_reference(
            x_c[tr_m], x_c[te_m], ypar_tr, yfull_tr, ypar_te, yfull_te, lam_cleg
        )
        got = comp_cleg.channels["test"][:, 0, :]
        scale = np.maximum(np.abs(oracle), 1.0)
        rel = float(np.max(np.abs(oracle - got) / scale))
        assert rel < 1e-7, f"rem component parity vs oracle: {rel:.2e}"
        rec["parity_rem_max_rel"] = rel

    # ── p_last prefix-arm cells (standing prefix+context mapping rule) ──────────
    p_x = decomp["own"]["p_last"].numpy().astype(np.float16)
    groups_p = [(f"rem_mean_gt{T2}", a) for a in ARMS]
    res_p = run_ridge_cell(
        p_x[tr_m],
        _stack_targets_h(staged, tr_m, groups_p, hid),
        {
            "val": (p_x[va_m], _stack_targets_h(staged, va_m, groups_p, hid)),
            "test": (p_x[te_m], _stack_targets_h(staged, te_m, groups_p, hid)),
        },
        group_names=[f"{s}|{a}" for s, a in groups_p],
        device=fit_device,
    )
    li_p = int(np.nanargmax(np.nanmean(res_p.pooled["val"], axis=1)))
    comp_plast = run_component_cell(
        p_x[tr_m],
        {"test": p_x[te_m]},
        rem_pair_fn,
        [f"rem|{a}" for a in rem_groups],
        np.full(len(rem_groups), DEFAULT_LAMBDAS_LIST[li_p]),
        device=fit_device,
    )
    rec["p_last"] = {
        "lambda": DEFAULT_LAMBDAS_LIST[li_p],
        "full_test_pooled_r2": {
            a: float(res_p.pooled["test"][li_p, gi]) for gi, (_s, a) in enumerate(groups_p)
        },
        "component_pooled": {
            k: {a: float(comp_plast.pooled["test"][k][gi]) for gi, a in enumerate(rem_groups)}
            for k in ("C_par", "C_perp", "C_cross", "r2_full", "w_par")
        },
        "note": "prefix arm — degenerate by construction on this single-turn pool",
    }

    # ── pooled component tables + the full(nx)-vs-parent delta ──────────────────
    def _pooled_table(res, names: list[str]) -> dict[str, dict[str, float]]:
        return {
            nm: {k: float(res.pooled["test"][k][gi]) for k in res.pooled["test"]}
            for gi, nm in enumerate(names)
        }

    rec["components"] = {
        "h1": _pooled_table(comp_h1, [f"{s}|{a}" for s, a in h1_groups]),
        "cleg_rem": _pooled_table(comp_cleg, [f"rem|{a}" for a in rem_groups]),
        "zleg_rem": {a: _pooled_table(comp_zleg[a], [f"rem|{a}"]) for a in MATCHED_ARMS},
        "lambda_frozen": {"cleg_mean": lam_cleg, "zleg_mean": lam_zleg},
        "additivity_max_dev": max(
            comp_h1.additivity_max_dev,
            comp_cleg.additivity_max_dev,
            comp_plast.additivity_max_dev,
            *(comp_zleg[a].additivity_max_dev for a in MATCHED_ARMS),
        ),
        "flagged_slots": list(FLAGGED_DECOMP_SLOTS),
        "excluded_slots": list(EXCLUDED_DECOMP_SLOTS),
    }
    # rem_mean_gt16_nx (reduced range) vs the parent's rem_mean_gt16 full cell —
    # expected << 0.001 (plan §3.5(2) REPORTED check, never an assert).
    rec["full_nx_vs_parent_delta"] = {
        a: float(comp_cleg.pooled["test"]["r2_full"][gi]) - layer_rec[f"cleg|{a}|mean"]
        for gi, a in enumerate(rem_groups)
    }
    rec["sens_lambdas"] = [float(v) for v in DEFAULT_LAMBDAS_LIST]
    assert comp_cleg.sens_pooled is not None
    rec["cleg_rem_sensitivity"] = {
        a: comp_cleg.sens_pooled["test"][:, gi, :].tolist() for gi, a in enumerate(rem_groups)
    }

    # ── npz channels (test split; fp64 identity asserted at fit time) ───────────
    unit_npz: dict[str, np.ndarray] = {}
    for gi, (slot, arm) in enumerate(h1_groups):
        unit_npz[f"H1_L{layer}|{slot}|{arm}"] = comp_h1.channels["test"][:, gi, :].astype(
            np.float32
        )
    for gi, arm in enumerate(rem_groups):
        unit_npz[f"M{T2}c_L{layer}|{arm}"] = comp_cleg.channels["test"][:, gi, :].astype(np.float32)
    for arm in MATCHED_ARMS:
        unit_npz[f"M{T2}z_L{layer}|{arm}"] = (
            comp_zleg[arm].channels["test"][:, 0, :].astype(np.float32)
        )
    rec["wall_s"] = time.time() - t_unit0
    del res_a, res_c, zres, comp_h1, comp_cleg, comp_zleg, res_p, comp_plast
    gc.collect()
    return unit_npz, rec


def phase_battery(
    base_dir: pathlib.Path,
    smoke: bool,
    layers: tuple[int, ...],
    fit_device: str,
    skip_upload: bool,
    u_dir_np: np.ndarray | None,
) -> None:
    """Phase C: per-fold (calibration fold FIRST as the K4 pilot) x per-layer
    battery with per-unit checkpoints; per-fold npz + JSON outputs uploaded the
    moment each fold completes (checkpoint-per-phase)."""
    log_phase("battery")
    t0 = time.time()
    if u_dir_np is None:
        if smoke:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
            u_dir_np = _unembed_dir(_build_smoke_model(tok)).cpu().numpy()
        else:
            u_dir_np = load_unembed_dir_production(DEFAULT_MODEL)

    pool_ids = _pool_ids_from_shard(base_dir, layers[0])
    spans_by_arm = {
        arm: np.asarray(
            [parent952._load_spans(base_dir, arm)[str(c)].get("span", 0) for c in pool_ids],
            dtype=np.int64,
        )
        for arm in ARMS
    }
    refs = ReproRefs(base_dir, smoke)
    folds = parent952.make_kfold_splits(pool_ids, K_FOLDS)
    # g5 fold-identity gate (K5): recomputed hashes vs the parent's committed
    # fold assignment (smoke: self-reference through the same compare).
    for f in folds:
        computed = parent952.kfold_split_hashes(f)
        ref = refs.fold_hashes(int(f["fold"]), computed)
        if computed != ref:
            raise RuntimeError(
                f"g5 fold-identity FAIL (K5) at fold {f['fold']}: recomputed split hashes "
                "!= parent fold_assignment.json"
            )
    min_train = 4 if smoke else parent952.MIN_CELL_TRAIN
    regime = {
        "smoke": bool(smoke),
        "layers": list(layers),
        "k_folds": K_FOLDS,
        "n_pool": len(pool_ids),
        "pool_sha": hashlib.sha256(json.dumps([int(i) for i in pool_ids]).encode()).hexdigest(),
        "lambdas": [float(v) for v in DEFAULT_LAMBDAS_LIST],
        "fit_device": fit_device,
        "min_train": min_train,
        "staging_revision": "smoke-synth" if smoke else PARENT_TENSOR_REVISION,
        "git_sha": _git_sha(),
    }
    ck_dir = _battery_ckpt_dir(base_dir, regime)
    out_dir = eval_out_dir(base_dir)

    fold_order = [CAL_FOLD] + [k for k in range(K_FOLDS) if k != CAL_FOLD]
    for oi, k in enumerate(fold_order):
        fold_split = folds[k]
        t_fold0 = time.time()
        fold_npz: dict[str, np.ndarray] = {}
        fold_rec: dict[str, Any] = {
            "fold": k,
            "layers": {},
            "regime": regime,
            "meta": _run_metadata(smoke, layers),
        }
        pos_of = {cid: i for i, cid in enumerate(pool_ids)}
        te_pos = np.asarray([pos_of[c] for c in fold_split["test"]])
        u16 = np.all(np.stack([spans_by_arm[a] >= T2 + 16 for a in ARMS]), axis=0)
        te_m = te_pos[u16[te_pos]]
        fold_npz["ids_test"] = np.asarray([pool_ids[p] for p in te_m], dtype=np.int64)
        # Full-pool ids ride along so phase D resamples the FULL pool (parent
        # bootstrap recipe verbatim — zeros scattered for non-matched contexts).
        fold_npz["ids_pool_full"] = np.asarray(pool_ids, dtype=np.int64)
        for layer in layers:
            ck_npz = ck_dir / f"fold{k}_L{layer}.npz"
            ck_json = ck_dir / f"fold{k}_L{layer}.json"
            if ck_npz.exists() and ck_json.exists():
                arrs = dict(np.load(ck_npz, allow_pickle=False))
                fold_npz.update(arrs)
                fold_rec["layers"][str(layer)] = json.loads(ck_json.read_text())
                logger.info("[battery] SKIP fold %d L%d (unit ckpt present)", k, layer)
                continue
            unit_npz, unit_rec = _battery_fold_layer(
                base_dir,
                fold_split,
                layer,
                pool_ids,
                spans_by_arm,
                u_dir_np,
                refs,
                fit_device,
                min_train,
                smoke,
                run_parity=(k == CAL_FOLD and layer == layers[0]),
            )
            np.savez(ck_npz, **unit_npz)
            ck_json.write_text(json.dumps(unit_rec, indent=2, default=_json_np))
            fold_npz.update(unit_npz)
            fold_rec["layers"][str(layer)] = unit_rec
            logger.info("[battery] fold %d L%d done (%.1fs)", k, layer, unit_rec.get("wall_s", 0))
        fold_rec["fold_wall_s"] = time.time() - t_fold0
        npz_path = out_dir / f"per_context_stats_1072_fold{k}.npz"
        np.savez(npz_path, **fold_npz)
        json_path = out_dir / f"battery_1072_fold{k}.json"
        json_path.write_text(json.dumps(fold_rec, indent=2, default=_json_np))
        if not skip_upload:
            _hf_commit_files_1072(f"battery fold {k}", [npz_path, json_path], base_dir)
        # K4 battery pilot: calibration fold measured FIRST, x K re-projection.
        if oi == 0:
            _pilot_check(
                "battery",
                time.time() - t_fold0,
                units_done=1,
                units_total=K_FOLDS,
                booked_h=BATTERY_BOOKED_H,
                base_dir=base_dir,
                smoke=smoke,
                execution_shape=(
                    f"per-(fold,layer) shared-SVD batched target-group stack (device={fit_device})"
                ),
            )
    logger.info("[battery] all folds done in %.1f min", (time.time() - t0) / 60)
    log_phase("battery_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Sentinel + CLI
# ═══════════════════════════════════════════════════════════════════════════════


def write_final_sentinel_1072(base_dir: pathlib.Path, smoke: bool, wall_h: float) -> None:
    """epm:results sentinel (poll_pipeline contract; SKILL.md Step 7 payload keys)."""
    out_dir = eval_out_dir(base_dir)
    eval_numbers: dict[str, Any] = {"note": "component stats battery runs VM-side (phase D)"}
    cal = out_dir / f"battery_1072_fold{CAL_FOLD}.json"
    if cal.exists():
        try:
            rec = json.loads(cal.read_text())
            l26 = rec["layers"].get("26") or rec["layers"].get(str(max(map(int, rec["layers"]))))
            eval_numbers["cal_fold_cleg_rem_components"] = (
                (l26 or {}).get("components", {}).get("cleg_rem")
            )
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning("cal-fold preview unavailable for the sentinel: %s", e)
    payload = {
        "eval_numbers": eval_numbers,
        "eval_paths": [str(out_dir)],
        "reproducibility_card": {
            "hf_data_repo": HF_DATA_REPO,
            "issue_slug": ISSUE_SLUG,
            "analysis_tensors_prefix": f"{ISSUE_SLUG}/analysis_tensors/",
            "eval_results_prefix": f"{ISSUE_SLUG}/eval_results/issue_1072/",
            "parent_tensor_revision": PARENT_TENSOR_REVISION,
            "model": DEFAULT_MODEL,
            "seeds": {"split": 952, "bootstrap": 0, "signflip": 1},
        },
        "wandb_url": "n/a (no model training in this experiment)",
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{ISSUE_SLUG}",
        "worktree_path": "n/a (pod-side)",
        "final_commit_sha": _git_sha(),
        "gpu_hours_used": wall_h,
        "gpu_hours_budgeted": 9,
        "plan_deviations": [],
    }
    write_sentinel(
        pathlib.Path(f"/workspace/logs/issue-1072-epm_results-{int(time.time())}.json"),
        {
            "kind": "epm:results",
            "version": 1,
            "note": json.dumps(
                {"status": "complete", "smoke": smoke, "issue": ISSUE, **payload},
                default=_json_np,
            ),
        },
    )


def verify_deferred_imports() -> int:
    """AST-walk this experiment's files and EXECUTE every deferred import
    (the #606/#1332 lazy-import gate; hand-maintained lists re-create drift)."""
    here = pathlib.Path(__file__).resolve()
    root = parent952.repo_root()
    files = [
        here,
        here.parent / "component_ridge.py",
        root / "scripts" / "issue1072_stats.py",
        root / "scripts" / "issue1072_figures.py",
    ]
    n_ok = 0
    failures: list[str] = []
    for f in files:
        tree = ast.parse(f.read_text())
        deferred: list[ast.Import | ast.ImportFrom] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import | ast.ImportFrom):
                        deferred.append(sub)
        for node in deferred:
            try:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        importlib.import_module(alias.name)
                        n_ok += 1
                else:
                    assert node.module is not None and node.level == 0, ast.dump(node)
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        if not hasattr(mod, alias.name):
                            # `from pkg import submodule` — resolves via import,
                            # not getattr, until the submodule is loaded.
                            importlib.import_module(f"{node.module}.{alias.name}")
                        getattr(mod, alias.name)
                        n_ok += 1
            except Exception as e:
                failures.append(f"{f.name}:{node.lineno}: {ast.dump(node)} -> {e!r}")
    print(json.dumps({"deferred_imports_ok": n_ok, "failures": failures}))
    if failures:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #1072 pod-side driver")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny-real run: 10 synth contexts, "
        "from-config 2-layer Qwen2 over the real tokenizer, layers 0,1",
    )
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        help="comma-separated subset of stage,capture,battery — or 'all'",
    )
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="capture batch size (default: 8 production, 4 smoke)",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="comma-separated hook layers (default: 14,20,23,26; smoke: 0,1)",
    )
    p.add_argument("--fit-device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument(
        "--verify-imports",
        action="store_true",
        help="execute every deferred import (AST-walked) and exit",
    )
    return p.parse_args()


def main() -> None:
    """Phase dispatcher — smoke IS the production path at tiny n (PASS_UNIFIED)."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    if args.verify_imports:
        sys.exit(verify_deferred_imports())
    import torch

    base_dir = parent952.resolve_base_dir(args.base_dir)
    smoke = bool(args.smoke)
    layers = (
        tuple(int(x) for x in args.layers.split(",") if x.strip())
        if args.layers
        else ((0, 1) if smoke else PROD_LAYERS)
    )
    batch_size = args.batch_size or (4 if smoke else 8)
    fit_device = args.fit_device or ("cuda" if torch.cuda.is_available() else "cpu")
    phases = (
        ["stage", "capture", "battery"]
        if args.phase == "all"
        else [x.strip() for x in args.phase.split(",") if x.strip()]
    )
    assert set(phases) <= {"stage", "capture", "battery"}, phases
    logger.info(
        "[main] phases=%s smoke=%s layers=%s batch=%d fit_device=%s base_dir=%s",
        phases,
        smoke,
        layers,
        batch_size,
        fit_device,
        base_dir,
    )
    t0 = time.time()
    u_dir_np: np.ndarray | None = None
    if "stage" in phases:
        phase_stage(base_dir, smoke, layers)
    if "capture" in phases:
        u_dir_np = phase_capture(base_dir, smoke, layers, batch_size, args.skip_upload)
    if "battery" in phases:
        phase_battery(base_dir, smoke, layers, fit_device, args.skip_upload, u_dir_np)
    write_final_sentinel_1072(base_dir, smoke, wall_h=(time.time() - t0) / 3600.0)
    log_phase("done")


if __name__ == "__main__":
    main()
