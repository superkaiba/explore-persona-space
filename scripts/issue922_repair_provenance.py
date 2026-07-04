#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #922 follow-up ``paired-provenance-transfer``: repair the 288 windows.

288 of the parent's 432 transfer windows (all 18 sycophancy/hallucination
capture cells) pair a CURRENT (post-#779-r5-regenerated) question with a
CACHED response to a DIFFERENT, lost question — the binding concern
``eval-questions-regenerated-parity-rescope``. This driver repairs exactly
that ONE variable (response provenance) with the parent's pinned recipe:

- ``--phase gen``   — vLLM generation (sampling convention = #779 pass_a
  rollouts: temperature 1.0, top_p 0.95, seed 42; n=1 — only the ri=0 rollout
  is consumed; max_tokens 512, the proposal pin, >= 12x the wa=40 capture
  window) of the model's OWN fresh completions to the CURRENT questions under
  each condition's context. Writes + uploads the completion TEXT immediately
  (generations are never discardable).
- ``--phase score`` — the parent's transfer DVs recomputed with the SAME
  statistics code (``issue922_eval.rollout_phase`` / ``_score_variant`` /
  paired horizon-mean reads) on three legs sharing one code path: the
  REPAIRED windows (fresh capture, ``--corpus eval_repaired`` via
  ``issue922_capture_positions.py``), the MISMATCHED-CACHED windows (the
  parent's ``store_eval`` shard, same 288 (trait, cond, qi) keys), and the
  EVIL-ONLY exact-provenance companion (144 windows). Maps = the pinned fp16
  boundary+lstar subset + the six arm-c direct row files; fit corpus,
  statistics, and the evil windows are untouched.
- ``--phase upload`` — repaired store + repair eval JSONs/npz to the HF data
  repo under ``{HF_OUT_PREFIX}``.

The teacher-forced CAPTURE of the fresh completions is NOT here — it is the
parent's own ``issue922_capture_positions.py --corpus eval_repaired
--completions <gen json>`` (same --wp 8 --wa 40 --batch 16 flags, all 29
rows, fp16), dispatched by ``issue922_repair_dispatch.sh``.

``--make-stub`` writes the VM-smoke stub: a random-init Qwen2 with PRODUCTION
depth (28 layers -> 29 store rows) and width (H=3584), thin FFN — so the
pinned production maps APPLY to the smoke capture end-to-end and the score
phase has no smoke fork (the GPU-bound gen/capture engines are the only
carve-out: HF backend + stub weights on the VM, vLLM + Qwen-2.5-7B on the
lane).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

# vLLM v1 EngineCore dies silently under fork() when main() touches the
# tokenizer before LLM() — gotchas #628; must be set before any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import issue922_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue922_repair")

REPAIR_TRAITS = ("sycophancy", "hallucination")  # the regenerated-provenance cells
N_REPAIR_WINDOWS = 288  # 18 cells x 16 questions

# Sampling convention: #779 pass_a rollouts used SamplingParams(n=10,
# temperature=1.0, top_p=0.95, max_tokens=1024, seed=42) on an engine built
# via create_vllm_engine(model, max_model_len=8192, seed=42)
# (scripts/issue779_collect.py). Deviations here, both proposal-pinned:
# n=1 (only the ri=0 rollout enters the eval panel) and max_tokens=512
# (>= 12x the wa=40 capture window; halves generation wall).
GEN_SAMPLING = {"n": 1, "temperature": 1.0, "top_p": 0.95, "max_tokens": 512, "seed": 42}
VLLM_MAX_MODEL_LEN = 8192
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))  # gotchas #664

# Parent-run artifact pins (#600 pattern): data-repo main resolved 2026-07-04,
# after the parent's upload phases landed; per-file sha256 asserted at every
# read (covers already-cached copies — the trust boundary is the read). The
# maps + store_eval + direct rows were all produced by the SAME parent run
# (one revision => a mutually coherent set; artifact-pair ordering per
# .claude/rules/artifact-reuse.md item (j)).
HF_REVISION_922 = "fb6283519e507383d0dfcf1b36220b4f6311a543"
EXPECTED_SHA256_922 = {
    "maps/maps_boundary_and_lstar_fp16.pt": (
        "1f1aaa839473f5508029ff05ad4345aa50b7eb9bc203f7ce79fa71ee9fd8f1dd"
    ),
    "store_eval/shard_000.pt": "a9e7e2f95fedd4995370932d29e84260a89a410825398addc127313d0b8f7c92",
    "maps_conditioned/direct_row_15.pt": (
        "d50e06520d31ff0da61bdf7997d311e392c5d7b67809f5a62c5a2ae7536a7ba4"
    ),
    "maps_conditioned/direct_row_18.pt": (
        "82aec1578aa554cb6502a5e9ec2c8bb876a2de48fd321423d0bd3e4f32e2dbd7"
    ),
    "maps_conditioned/direct_row_20.pt": (
        "d54fc80c24786b70698498c6f4fd1f65501ed4b7087195e014e73e493650ef45"
    ),
    "maps_conditioned/direct_row_21.pt": (
        "20cd9e45c9c5220f5a7f5ba19da78eec3a8c498c875826d30322fb9355da7f0e"
    ),
    "maps_conditioned/direct_row_25.pt": (
        "a9539f9da5494573e0ba975d15ab4239c1cb8669c902ed2fbc33a5082972e148"
    ),
    "maps_conditioned/direct_row_27.pt": (
        "5f939d206544ccb35449171671b58a37617a5757717af9eecdf05752acaca7a9"
    ),
}
READOUT_ROWS = [C.block_to_row(b) for b in C.READOUT_BLOCKS]  # [15, 18, 20, 21, 25, 27]


def _fetch922(rel: str) -> Path:
    """Pinned fetch of a parent-run artifact under ``issue922_nexttoken/``.

    Fail-loud; revision + sha256 pinned (the #600 HF-mirror-divergence
    pattern). NOTE: reads the PRODUCTION prefix constant, never
    ``C.HF_OUT_PREFIX`` (which the smoke redirects for WRITES).
    """
    from huggingface_hub import hf_hub_download

    C.HF_DL_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(
        hf_hub_download(
            repo_id=C.HF_DATA_REPO,
            filename=f"issue922_nexttoken/{rel}",
            repo_type="dataset",
            revision=HF_REVISION_922,
            cache_dir=str(C.HF_DL_DIR),
        )
    )
    want = EXPECTED_SHA256_922[rel]
    got = C.sha256_path(p)
    assert got == want, f"sha256 mismatch for issue922_nexttoken/{rel}: {got} != pinned {want}"
    return p


# ── gen phase ─────────────────────────────────────────────────────────────────


def build_repair_items(smoke: bool) -> list[dict]:
    """The 288 regenerated-question windows, fresh-response slots empty.

    ``build_eval_subset_items`` (the parent's exact panel: n_per_cell=16,
    seed=42 question subset, verbatim #779 message construction) filtered to
    the regenerated-provenance traits; ``ci`` renumbered densely. ``smoke``
    keeps the FIRST cell's first 3 items — the single subset definition every
    later phase enumerates from (via the gen artifact).
    """
    items = [it for it in C.build_eval_subset_items() if it["trait"] in REPAIR_TRAITS]
    assert len(items) == N_REPAIR_WINDOWS, len(items)
    for it in items:
        assert it["question_provenance"] == "regenerated", (it["trait"], it["cond_id"])
    if smoke:
        t0, c0 = items[0]["trait"], items[0]["cond_id"]
        items = [it for it in items if it["trait"] == t0 and it["cond_id"] == c0][:3]
    out = []
    for j, it in enumerate(items):
        cached = it["response"]
        out.append(
            {
                "ci": j,
                "trait": it["trait"],
                "cond_id": it["cond_id"],
                "mode": it["mode"],
                "qi": it["qi"],
                "question_provenance": it["question_provenance"],
                "response_provenance": "fresh_onpolicy",
                "messages": it["messages"],
                "cached_response_sha256": hashlib.sha256(cached.encode()).hexdigest(),
                "cached_response_len": len(cached),
            }
        )
    return out


def _gen_vllm(model: str, prompts: list[str]) -> list[str]:
    """#779 pass_a generation shape: chunked ``LLM.generate`` (gotchas #613/#664)."""
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import create_vllm_engine

    llm = create_vllm_engine(model, max_model_len=VLLM_MAX_MODEL_LEN, seed=GEN_SAMPLING["seed"])
    sp = SamplingParams(**GEN_SAMPLING)
    out: list[str] = []
    n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompts), VLLM_CHUNK_SIZE):
        chunk = prompts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] repair-gen chunk %d/%d (%d prompts)",
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
        )
        outs = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in outs)
    return out


def _gen_hf(model: str, tokenizer, prompts: list[str]) -> list[str]:
    """CPU stub-smoke backend (GPU-bound carve-out; production = ``_gen_vllm``).

    Batched LEFT-padded HF ``generate`` (HF derives position_ids from the
    attention mask internally) with the same sampling family (do_sample,
    temperature/top_p, seeded). ``min_new_tokens=4`` keeps stub answers
    non-empty so the smoke windows carry answer transitions.
    """
    from transformers import AutoModelForCausalLM

    torch.manual_seed(GEN_SAMPLING["seed"])
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
    m.eval()
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            gen = m.generate(
                **enc,
                do_sample=True,
                temperature=GEN_SAMPLING["temperature"],
                top_p=GEN_SAMPLING["top_p"],
                max_new_tokens=int(os.environ.get("EPM922_SMOKE_GEN_TOKENS", "24")),
                min_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        tokenizer.padding_side = prev_side
    return tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def phase_gen(args) -> int:
    """Generate fresh on-policy completions to the CURRENT questions; persist + upload."""
    from transformers import AutoTokenizer

    if args.completions.exists():
        # Resume predicate keyed on EVERY output-affecting regime key (#722 r3
        # rule): a stale gen JSON silently invalidates the captured shard (the
        # shard validator cannot see response text), so never regenerate over a
        # regime-matching artifact, and never reuse a mismatched one.
        with open(args.completions) as f:
            prior = json.load(f)
        same = (
            prior.get("backend") == args.gen_backend
            and prior.get("model") == args.model
            and bool(prior.get("smoke")) == bool(args.smoke)
            and prior.get("sampling") == GEN_SAMPLING
        )
        if same:
            logger.info("[gen] %s exists + regime-valid — skip (resume)", args.completions)
            return 0
        logger.warning("[gen] existing completions FAIL regime check — regenerating")

    items = build_repair_items(args.smoke)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    prompts = [
        tokenizer.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
        for it in items
    ]
    t0 = time.time()
    if args.gen_backend == "vllm":
        texts = _gen_vllm(args.model, prompts)
    else:
        texts = _gen_hf(args.model, tokenizer, prompts)
    assert len(texts) == len(items), (len(texts), len(items))
    n_empty = sum(1 for t in texts if not t.strip())
    for it, t in zip(items, texts, strict=True):
        it["response"] = t
    blob = {
        "items": items,
        "sampling": GEN_SAMPLING,
        "backend": args.gen_backend,
        "model": args.model,
        "max_model_len": VLLM_MAX_MODEL_LEN if args.gen_backend == "vllm" else None,
        "n_items": len(items),
        "n_empty": n_empty,
        "smoke": args.smoke,
        "wall_seconds": time.time() - t0,
        "metadata": C.reproducibility_metadata(
            {"script": "issue922_repair_provenance", "phase": "gen"}
        ),
    }
    args.completions.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.completions, blob)
    logger.info(
        "[gen] wrote %d completions (%d empty; backend=%s) -> %s",
        len(items),
        n_empty,
        args.gen_backend,
        args.completions,
    )
    if not args.skip_upload:
        # Model generations are NEVER discardable — the text uploads the moment
        # it exists (non-LFS JSON path, quota-immune), before any capture.
        ev = C.upload_dir_bulk(
            args.completions.parent,
            f"{C.HF_OUT_PREFIX}/repair/raw_completions",
            allow_patterns=[args.completions.name],
            commit_message="issue922 repair: fresh on-policy completions (seed42)",
            allow_overflow=False,
        )
        logger.info("[gen] uploaded completions: %s", json.dumps(ev))
    return 0


# ── score phase ───────────────────────────────────────────────────────────────


def _load_ridge_lstar() -> dict:
    """The pinned fp16 map subset -> the eval-code ridge dict at the 6 ℓ* rows.

    fp16 tensors cast to fp32; asserts every (arm, row) needed by the rollout
    + single-step paths is present at production H — the SAME assert that
    catches a shape-mismatched capture in production fires in the smoke.
    """
    from explore_persona_space.experiments.issue_841.maps import RidgeMap

    blob = torch.load(_fetch922("maps/maps_boundary_and_lstar_fp16.pt"), weights_only=False)

    def _map(st: dict) -> RidgeMap:
        st = dict(st)
        for k in ("mu", "sd", "w", "bias"):
            st[k] = st[k].to(torch.float32)
        return RidgeMap(**st)

    rows = list(READOUT_ROWS)
    keep = set(rows)
    ridge = {
        "rows": rows,
        # only the read-out rows are rolled/read — casting all 29 boundary
        # rows to fp32 would triple peak RAM for maps nothing consumes.
        "boundary": {
            arm: {int(r): _map(st) for r, st in d.items() if int(r) in keep}
            for arm, d in blob["boundary"].items()
        },
        "answer": {
            arm: {int(r): _map(st) for r, st in d.items() if int(r) in keep}
            for arm, d in blob["answer_lstar"].items()
        },
        "b1_answer": {
            int(r): _map(st) for r, st in blob.get("b1_answer_lstar", {}).items() if int(r) in keep
        },
        "sigma_by_row": blob["sigma_by_row"],
    }
    del blob
    for r in rows:
        for arm in ("ctx", "tok"):  # boundary maps were fit for ctx + tok only
            assert r in ridge["boundary"][arm], ("boundary map missing", arm, r)
        for arm in ("ctx", "tok", "emb"):
            assert r in ridge["answer"][arm], ("answer map missing", arm, r)
    assert ridge["answer"]["ctx"][rows[0]].w.shape[-1] == C.EXPECTED_HIDDEN
    return ridge


def _stage_direct_dir(store_root: Path) -> Path:
    """Symlink the six pinned arm-c per-row files into a ``direct/`` dir."""
    d = store_root / "direct"
    d.mkdir(parents=True, exist_ok=True)
    for r in READOUT_ROWS:
        name = f"direct_row_{r:02d}.pt"
        src = _fetch922(f"maps_conditioned/{name}")
        dst = d / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return d


def _stage_cached_store(store_root: Path) -> None:
    """Symlink the parent's pinned ``store_eval`` shard as corpus ``eval_subset``."""
    d = store_root / "eval_subset"
    d.mkdir(parents=True, exist_ok=True)
    src = _fetch922("store_eval/shard_000.pt")
    dst = d / "shard_000.pt"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _window_keys(store: dict) -> dict[tuple, int]:
    """(trait, cond_id, qi) -> ctx index; asserts key uniqueness."""
    out: dict[tuple, int] = {}
    for idx, ci in enumerate(store["ctx_ids"]):
        m = store["meta"][ci]
        k = (m["trait"], m["cond_id"], int(m["qi"]))
        assert k not in out, ("duplicate window key", k)
        out[k] = idx
    return out


def _single_step_leg(store: dict, sel: np.ndarray, ridge: dict) -> dict:
    """Per-ℓ*-row single-step transfer r2_id (ridge_{ctx,tok,emb} + b1 arms).

    Identity-relative R² only — ``delta_train_mean`` was not persisted in the
    fp16 subset, so the mean-centered companion is not recomputable here (the
    parent's transfer_eval.json carries it for the cached windows).
    """
    from issue922_eval import _gather_X
    from issue922_fit_maps import transition_indices

    from explore_persona_space.experiments.issue_922 import maps922 as M

    tr = transition_indices(store, np.asarray(sel))
    idx = tr["answer"]
    out: dict = {"n_answer_transitions": int(idx.numel()), "cells": {}}
    if idx.numel() == 0:
        return out
    for r in ridge["rows"]:
        bk = C.row_to_block_key(r)
        delta = (
            _gather_X(store["h"], r, idx + 1, "ctx") - _gather_X(store["h"], r, idx, "ctx")
        ).numpy()
        cell = {}
        for arm in ("ctx", "tok", "emb"):
            X = _gather_X(store["h"], r, idx, arm)
            pred = M.ridge_predict(ridge["answer"][arm][r], X).numpy()
            cell[f"ridge_{arm}"] = {"r2_id": M.identity_relative_r2(pred, delta)}
        if r in ridge.get("b1_answer", {}):
            Xh = _gather_X(store["h"], r, idx, "ctx")
            Xc = store["h"][r, tr["answer_T"], :].to("cpu", torch.float32)
            pred = M.ridge_predict(ridge["b1_answer"][r], torch.cat([Xh, Xc], dim=1)).numpy()
            cell["b1_ridge"] = {"r2_id": M.identity_relative_r2(pred, delta)}
        out["cells"][bk] = cell
    return out


def _augment_npz_keys(npz_path: Path, store: dict) -> None:
    """Append (trait, cond_id, qi) window-key arrays to a rollout npz in place."""
    data = dict(np.load(npz_path, allow_pickle=False))
    meta_by_ci = store["meta"]
    traits, conds, qis = [], [], []
    for ci in data["ctx_ids"]:
        m = meta_by_ci[int(ci)]
        traits.append(m["trait"])
        conds.append(m["cond_id"])
        qis.append(int(m["qi"]))
    np.savez(
        npz_path,
        **data,
        key_trait=np.array(traits),
        key_cond=np.array(conds),
        key_qi=np.array(qis, dtype=np.int64),
    )


def _k32_reads(roll: dict, k: int = C.READOUT_K_MAX) -> dict:
    """Per-block k=READOUT_K_MAX skill mean+CI for the headline variants."""
    out: dict = {}
    for name in ("ridge_ctx_boundary_first", "direct_c", "tok_ceiling", "b1_ridge_roll"):
        if name not in roll["variants"]:
            continue
        out[name] = {
            bk: cis[k - 1] if len(cis) >= k else None
            for bk, cis in roll["variants"][name]["skill_mean_ci"].items()
        }
    return out


def _paired_repaired_minus_cached(
    npz_rep: Path, npz_cached: Path, store_rep: dict, store_cached: dict, n_boot: int
) -> dict:
    """Per-window paired delta (repaired − cached) matched on (trait, cond, qi).

    Horizon-mean (k ≤ READOUT_K_MAX) and k=32 paired deltas with the parent's
    bootstrap CI (`_boot_mean_ci`, seed 0) per read-out block, for the ctx
    roll and the direct-c arm.
    """
    from issue922_eval import _boot_mean_ci, _horizon_mean_perctx

    a, b = dict(np.load(npz_rep)), dict(np.load(npz_cached))

    def _keys_of(d: dict, store: dict) -> list[tuple]:
        return [
            (
                store["meta"][int(ci)]["trait"],
                store["meta"][int(ci)]["cond_id"],
                int(store["meta"][int(ci)]["qi"]),
            )
            for ci in d["ctx_ids"]
        ]

    ka, kb = _keys_of(a, store_rep), _keys_of(b, store_cached)
    pos_b = {k: i for i, k in enumerate(kb)}
    shared = [k for k in ka if k in pos_b]
    ia = np.array([i for i, k in enumerate(ka) if k in pos_b], dtype=np.int64)
    ib = np.array([pos_b[k] for k in shared], dtype=np.int64)
    assert len(shared) == len(ka), (
        f"repaired windows missing from cached store: {len(ka) - len(shared)}"
    )
    out: dict = {"n_paired_windows": len(shared)}
    for variant in ("ridge_ctx_boundary_first", "direct_c"):
        per_bk: dict = {}
        for bk in [C.row_to_block_key(r) for r in READOUT_ROWS]:
            key = f"skill__{variant}__{bk}"
            if key not in a or key not in b:
                continue
            sa, sb = a[key][ia], b[key][ib]  # (n, k_max) each, row-aligned
            hm_a = _horizon_mean_perctx(sa, C.READOUT_K_MAX)
            hm_b = _horizon_mean_perctx(sb, C.READOUT_K_MAX)
            m = np.isfinite(hm_a) & np.isfinite(hm_b)
            hm = _boot_mean_ci(hm_a[m] - hm_b[m], n_boot=n_boot, seed=C.BOOTSTRAP_SEED)
            hm["excludes_zero"] = bool(hm["lo"] > 0.0 or hm["hi"] < 0.0)
            col_a, col_b = sa[:, C.READOUT_K_MAX - 1], sb[:, C.READOUT_K_MAX - 1]
            mk = np.isfinite(col_a) & np.isfinite(col_b)
            k32 = _boot_mean_ci(col_a[mk] - col_b[mk], n_boot=n_boot, seed=C.BOOTSTRAP_SEED)
            k32["excludes_zero"] = bool(k32["lo"] > 0.0 or k32["hi"] < 0.0)
            per_bk[bk] = {"horizon_mean_delta": hm, "k32_delta": k32}
        out[variant] = per_bk
    return out


def phase_score(args) -> int:
    """Three-way transfer re-read: repaired vs mismatched-cached vs evil-only."""
    from issue922_eval import _resolve_device, rollout_phase

    device = _resolve_device(args.device)
    ridge = _load_ridge_lstar()
    direct_dir = _stage_direct_dir(args.store)
    _stage_cached_store(args.store)

    repaired = C.load_store(args.store, "eval_repaired")
    cached = C.load_store(args.store, "eval_subset")
    # Cross-phase data contract (the production-shape asserts; the smoke's
    # 28-layer H=3584 stub satisfies them, so a drifted capture fails HERE).
    assert repaired["blocks"] == cached["blocks"], (
        len(repaired["blocks"]),
        len(cached["blocks"]),
    )
    assert repaired["h"].shape[-1] == cached["h"].shape[-1] == C.EXPECTED_HIDDEN
    assert repaired["window"] == cached["window"] == {"wp": C.W_P, "wa": C.W_A}, (
        repaired["window"],
        cached["window"],
    )
    for rec in repaired["meta"].values():
        assert rec.get("response_provenance") == "fresh_onpolicy", rec.get("response_provenance")
    # gen ↔ capture integrity: each captured window's ans_len must equal the
    # fresh completion's token count under the capture tokenizer — catches a
    # shard captured under DIFFERENT completions (the shard validator cannot
    # see response text; the gen resume predicate is the first line of
    # defense, this is the fail-loud second).
    assert args.completions.exists(), f"score requires the gen artifact: {args.completions}"
    with open(args.completions) as f:
        gen_items = json.load(f)["items"]
    from transformers import AutoTokenizer

    cap_tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    assert len(gen_items) == len(repaired["ctx_ids"]), (
        len(gen_items),
        len(repaired["ctx_ids"]),
    )
    for it in gen_items:
        want = len(cap_tok(it["response"], add_special_tokens=False)["input_ids"])
        got = int(repaired["meta"][int(it["ci"])]["ans_len"])
        assert got == want, ("gen/capture desync at ci", it["ci"], got, want)

    keys_rep = _window_keys(repaired)
    keys_cached = _window_keys(cached)
    missing = [k for k in keys_rep if k not in keys_cached]
    assert not missing, f"repaired windows absent from cached store: {missing[:3]}"
    rep_sel = np.arange(len(repaired["ctx_ids"]), dtype=np.int64)
    cached_sel = np.array([keys_cached[k] for k in keys_rep], dtype=np.int64)
    evil_sel = np.array(
        [i for i, ci in enumerate(cached["ctx_ids"]) if cached["meta"][ci]["trait"] == "evil"],
        dtype=np.int64,
    )
    assert evil_sel.size > 0, "no evil windows in the cached store"
    if args.smoke:
        evil_sel = evil_sel[: len(rep_sel)]  # same subset size as the smoke's repaired leg

    args.out.mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(n_boot=args.n_boot)
    legs = {
        "repaired": (repaired, rep_sel),
        "cached_mismatched": (cached, cached_sel),
        "evil_original": (cached, evil_sel),
    }
    results: dict = {}
    npz_paths: dict = {}
    t0 = time.time()
    for name, (store, sel) in legs.items():
        npz = args.out / f"paired_provenance_{name}_percontext.npz"
        roll = rollout_phase(
            store,
            ridge,
            None,
            None,
            sel,
            ns,
            device,
            corpus=name,
            drift=None,  # no fit store on the repair round — mean-drift null skipped
            out_npz=npz,
            cond_blob=None,
            direct_dir=direct_dir,
        )
        _augment_npz_keys(npz, store)
        leg = {
            "n_windows": len(sel),
            "single_step": _single_step_leg(store, sel, ridge),
            "rollout": roll,
            "k32_reads": _k32_reads(roll),
        }
        # checkpoint-per-leg: persist the moment the leg completes.
        C.write_json_atomic(
            args.out / f"paired_provenance_leg_{name}.json",
            {
                **leg,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue922_repair_provenance", "phase": "score", "leg": name}
                ),
            },
        )
        results[name] = leg
        npz_paths[name] = npz
        logger.info("[score] leg %s done (%.1fs elapsed)", name, time.time() - t0)

    paired = _paired_repaired_minus_cached(
        npz_paths["repaired"], npz_paths["cached_mismatched"], repaired, cached, args.n_boot
    )
    gen_blob = {}
    if args.completions.exists():
        with open(args.completions) as f:
            gen_blob = json.load(f)
    combined = {
        "legs": results,
        "paired_repaired_minus_cached": paired,
        "readout_blocks": [str(b) for b in C.READOUT_BLOCKS],
        "gen": {k: gen_blob.get(k) for k in ("sampling", "backend", "model", "n_empty", "smoke")},
        "pins": {"hf_revision": HF_REVISION_922, "sha256": EXPECTED_SHA256_922},
        "smoke": args.smoke,
        "note": (
            "paired-provenance-transfer repair: transfer DVs (single-step r2_id, rollout "
            "skill k=1..40, rolled-vs-direct paired horizon-mean) at the 6 read-out blocks "
            "on three legs sharing one code path — repaired (fresh on-policy completions "
            "to the CURRENT questions, teacher-forced) vs mismatched-cached (the parent's "
            "store_eval pairing) vs the evil-only exact-provenance companion. Maps, fit "
            "corpus, statistics code, and the evil windows are the parent's, unchanged."
        ),
        "metadata": C.reproducibility_metadata(
            {"script": "issue922_repair_provenance", "phase": "score"}
        ),
    }
    C.write_json_atomic(args.out / "paired_provenance_transfer.json", combined)
    logger.info(
        "[score] DONE in %.1fs -> %s",
        time.time() - t0,
        args.out / "paired_provenance_transfer.json",
    )
    return 0


# ── upload phase ──────────────────────────────────────────────────────────────


def phase_upload(args) -> int:
    """Repaired store + repair eval artifacts to the HF data repo (bulk commits)."""
    events: dict = {}
    d = args.store / "eval_repaired"
    assert d.is_dir(), f"repaired store missing: {d}"
    events["store_eval_repaired"] = C.upload_dir_bulk(
        d,
        f"{C.HF_OUT_PREFIX}/store_eval_repaired",
        allow_patterns=["*.pt", "*.json"],
        commit_message="issue922 repaired eval-condition store (fresh on-policy completions)",
        allow_overflow=True,
    )
    events["repair_eval_results"] = C.upload_dir_bulk(
        args.out,
        f"{C.HF_OUT_PREFIX}/repair/eval_results",
        allow_patterns=["paired_provenance_*"],
        commit_message="issue922 paired-provenance repair eval results",
        allow_overflow=True,
    )
    C.write_json_atomic(
        args.out / "paired_provenance_upload_events.json",
        {
            "events": events,
            "metadata": C.reproducibility_metadata(
                {"script": "issue922_repair_provenance", "phase": "upload"}
            ),
        },
    )
    logger.info("[upload] %s", json.dumps(events))
    return 0


# ── VM-smoke stub ─────────────────────────────────────────────────────────────


def make_stub(out_dir: Path, tokenizer_id: str) -> int:
    """Random-init Qwen2 stub with PRODUCTION depth (28) and width (3584).

    Thin FFN (intermediate 256) keeps it ~1.4B params so the VM CPU smoke can
    generate + capture; the 29-row H=3584 store it produces lets the pinned
    production maps apply end-to-end (no smoke fork in the score phase).
    """
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    cfg = Qwen2Config(
        vocab_size=152064,  # real Qwen-2.5-7B vocab — real token ids must resolve
        hidden_size=C.EXPECTED_HIDDEN,
        num_hidden_layers=C.EXPECTED_LAYERS,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=256,
        tie_word_embeddings=True,
        eos_token_id=C.IM_END_ID,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    logger.info(
        "[stub] wrote %s (%.2fB params)", out_dir, sum(p.numel() for p in model.parameters()) / 1e9
    )
    return 0


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #922 paired-provenance transfer repair.")
    ap.add_argument("--phase", choices=("gen", "score", "upload"), default=None)
    ap.add_argument("--make-stub", type=Path, default=None, help="write the VM-smoke stub + exit")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--gen-backend", choices=("vllm", "hf"), default="vllm")
    ap.add_argument(
        "--completions",
        type=Path,
        default=Path("/workspace/issue922_store/repair_completions/fresh_completions_seed42.json"),
    )
    ap.add_argument("--store", type=Path, default=Path("/workspace/issue922_store"))
    ap.add_argument("--out", type=Path, default=Path("eval_results/issue_922"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()
    if args.make_stub is not None:
        return make_stub(args.make_stub, args.tokenizer or C.DEFAULT_MODEL)
    assert args.phase is not None, "--phase required (gen | score | upload)"
    if args.smoke:
        args.n_boot = min(args.n_boot, 100)
    if args.phase == "gen":
        return phase_gen(args)
    if args.phase == "score":
        return phase_score(args)
    return phase_upload(args)


if __name__ == "__main__":
    sys.exit(main())
