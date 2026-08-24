"""Issue #2474 follow-up ``postnorm-l27-diagnostic`` — post-final-RMSNorm re-read driver.

Phases (``--phase``): smoke | trainref-gpu | rescore | stats | figs | all
(``all`` = rescore -> stats -> figs, the offline VM chain; ``trainref-gpu`` is the
pod leg, dispatched separately; the ``PHASES`` registry below is the arm set of
record).

Question (scope note, epm:followup-scope ``postnorm-l27-diagnostic``): do the
capitalization training-reference champions at PRE-norm layer 27 (mapped-answer
pooled rho +0.735, shift-only +0.751, real-answer) survive re-measurement under
the POST-final-RMSNorm representation convention, or are they a final-residual
capture-convention artifact? ONE variable changes vs the parent run: the
layer-27 representation convention. Banks, adapters, DV, bootstrap seeds and
aggregation are inherited unchanged (N_BOOT=2000, BOOT_SEED=20260822).

Representation conventions (recorded in every output):
  * Stored layer-27 states are PRE-final-norm block outputs (capture docstring:
    stored index i == decoder block i, pre-final-norm at i=27). The parent's
    stored CONTEXT vectors are per-(question, trigger) rows (grid.pt, 960/864
    rows) and the CEILING answer vectors are per-(question, trigger, rollout)
    rows — so the offline post-norm re-read applies RMSNorm ROW-WISE, BEFORE
    any averaging over extraction questions (mean-of-normed), matching how the
    parent averages raw states.
  * WITHIN-row grain caveat: each stored ceiling answer vector is already a
    mean over answer TOKENS of pre-norm states, so the offline re-read yields
    norm-of-token-mean; the token-level mean-of-norm is unrecoverable offline
    (per-token states are the parent's declared streaming-reduce discard).
    The GPU phase therefore persists BOTH grains for the training-mix answer
    mean: ``mu_a_post_rowgrain`` = mean over rows of RMSNorm(token-mean state)
    (PRIMARY — the same operator chain as the offline comparison vectors) and
    ``mu_a_post_tokengrain`` = mean over rows of token-mean of RMSNorm(per-token
    states) (sensitivity companion).
  * Map-arm decision (scope design fact 3, option (a)): the mapped-answer arm
    keeps the pinned #779 pass-B map exactly as fit on PRE-norm states; the
    map's PRE-norm output (the persisted ``base_caps_L27_vhat.pt``) is
    post-normed — RMSNorm applied to a layer-27 residual state is precisely
    the model's own read-out convention, so no post-norm input ever enters the
    pre-norm-trained map. The scope's declared fallback (skip the map arm on a
    convention mismatch) is not needed. Caveat: the persisted v_hat is
    fp16-rounded (Gate R tolerance 2e-3 on the ans arms).
  * Shift-only arm: v_ib = v_c + b with b recomputed EXACTLY from the pinned
    pass-B bundle train split (``issue2379_mapfit._split_indices`` +
    ``mapping_baselines.identity_bias_predict``, the fit-worker idiom); the
    post-norm read is RMSNorm(v_c + b) — shift first, model read-out second.
  * Finite-dtype convention: the shared ``rms_norm_rows`` operator reproduces
    ``Qwen2RMSNorm.forward`` op-for-op — fp32 variance reduce, normalized
    states cast back to bfloat16 BEFORE the bf16 weight multiply. Every input
    is cast to bf16 first (the production hidden-state dtype; fp16-persisted
    states are bf16-representable over the states' range, so the cast recovers
    the production value); the bf16 result returns as fp64 for the downstream
    cosine math.

Validity gates:
  * Gate P (trainref-gpu, per condition): the re-forward's PRE-norm running
    means must reproduce the stored ``mu.pt`` layer-27 vectors (cos >= 0.999)
    — proves recipe fidelity (same rows, rendering, positions) end-to-end.
  * Spot gate (trainref-gpu, first --spot-rows rows per condition): the fused
    single-forward capture must match the parent's two-helper capture path
    (issue779_collect) — exact response-token-count equality + cosine bars
    (bf16/CUDA production: 0.995 single-position / 0.999 token-mean, per the
    #779 L27 bf16 calibration; fp32/CPU smoke: 0.99999).
  * Gate R (rescore, per setting): the PRE-norm layer-27 trainref/sameq arm
    values recomputed from staged inputs must reproduce
    ``eval_results/issue_2474/prefit/prefit_scores.json`` (tolerance 1e-6;
    2e-3 for the fp16-vhat ans arms). Certifies every staged input before the
    post-norm transform is applied.
  * vhat identity pin (rescore): the staged ``base_<setting>_L27_vhat.pt``
    bytes must sha256-match the pinned parent-producer hashes (``VHAT_SHA256``)
    before any post-norm read consumes them (vector-level identity, not just
    schema + the Gate R aggregate projections).
  * Stats recompute check: recomputed PRE pooled rho vs the parent
    ``prefit_stats.json`` pinned values (1e-6; warn 0.02 / fail 0.05 for the
    fp16-vhat ans arms whose rank order may flip on ties).

Smoke (``--phase smoke``): synthetic tiny end-to-end under ``--smoke-dir`` —
NO canonical writes. A 2-layer from-config Qwen2 model over the REAL tokenizer
(the #906 tiny-real pattern) runs the REAL trainref-gpu body (cpu, layer=1)
against 3-row benign synthetic mixes whose stored-mu targets were generated
through the PARENT two-helper path, so Gate P and the spot gate run for real;
the same rescore -> stats -> figs chain then runs against self-consistent
synthetic targets (Gate R real). Resume legs exercised: mid-condition partial
resume (interrupt hook), completed-condition skip, rescore/stats completion
replay. A REAL ranged norm-weight fetch probe runs against the live Hub
(``--skip-net-probe`` to skip).

Smoke blind-spot enumeration:
  * production bf16/CUDA numerics + the 7B model are not exercised (GPU-bound
    carve-out) — covered in production by Gate P + the per-condition spot gate;
  * HF staging of production inputs (train mixes, grid/ceiling/mu bundles,
    vhat, pass-B) is not exercised — fail-loud staged + Gate-R-verified on the
    first production run;
  * the real ranged norm-weight fetch IS probed, but the smoke scoring consumes
    the tiny model's own norm weight, not the fetched one;
  * the production upload leg + /workspace/logs sentinel are not exercised
    (smoke uploads nothing; sentinel routed to the smoke tree);
  * the production bootstrap validity floor (n_valid >= 100) is scaled to the
    smoke draw count (min(100, n_boot // 2)) — gate-calibration parity;
  * the production vhat sha-256 pins (``VHAT_SHA256``) are not exercised —
    the smoke pins the synthetic vhat's own sha (plumbing only); the
    discriminating perturbation check is a committed pytest.

Run (pod, GPU phase — all 8 conditions, layer 27 only):
    uv run python scripts/issue2474_postnorm.py --phase trainref-gpu
Run (VM, offline chain, after the GPU phase lands):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2474_postnorm.py --phase all
Run (smoke, VM):
    ... uv run python scripts/issue2474_postnorm.py --phase smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

# Thread caps must land before any numpy/torch import (heavy imports are all
# deferred into functions below, the issue2474_fit convention).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue2474_postnorm")

ISSUE = 2474
SLUG = "issue2474_prefit"
LABEL = "postnorm-l27-diagnostic"
LAYER = 27  # production stored-layer index (pre-final-norm block output)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_HIDDEN = 3584
RMS_EPS_EXPECTED = 1e-6
NORM_WEIGHT_NAME = "model.norm.weight"
HF_TRAIN_PREFIX = "issue2379_reelicit/train"  # full training mixes (8 jsonl)
HF_POSTNORM_PREFIX = f"{SLUG}/postnorm_l27"  # GPU-phase upload destination
# v2: rms_norm_rows moved to the Qwen2RMSNorm finite-dtype convention (bf16
# in/out, fp32 variance) — every post-norm number changes vs v1 (r1 review).
RECIPE_TAG = "postnorm-l27-v2"  # bump on ANY semantics change; rides every fingerprint
# Parent-producer identity pins for the persisted map outputs (sha256 over the
# staged file bytes; computed 2026-08-23 from the canonical HF artifacts the
# parent prefit round produced and Gate R validates). A staged vhat that does
# not match is NOT the parent map output — fail loud before any post-norm read.
VHAT_SHA256 = {
    "caps": "2429f0e2003f0390f0b8c3aaee0b05f51fde0d74e70c97a88479e0367c1bd552",
    "em": "b2ae177b94f685207f06cd43303fac9c038d8200bc9254baa139ba1c8647e596",
}
N_BOOT_DEFAULT = 2000
BOOT_SEED_DEFAULT = 20260822
SAMEQ_FAMS = ("ctx_sameq", "ans_sameq_mapB", "identbias_sameq", "ceiling_sameq")
TRAINREF_FAMS = ("ctx_trainref", "ans_trainref_mapB", "identbias_trainref", "ceiling_trainref")
ALL_FAMS = tuple([f for base in (SAMEQ_FAMS + TRAINREF_FAMS) for f in (base, base + "_centered")])
GRAIN_KEYS = ("pre", "post_rowgrain", "post_tokengrain")
# Reader-facing display names (figure axis labels; internal fam ids never ship).
DISPLAY_NAME = {
    "ans_trainref_mapB": "Predicted answer",
    "identbias_trainref": "Shift-only answer",
    "ceiling_trainref": "Real answer",
    "ctx_trainref": "Context",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _metadata(phase: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = as_metadata_dict(git_provenance())
    # Card phase IDENTITY (sibling of git_commit; never a lifecycle-state value
    # — this branch's provenance module predates the #2194 `phase=` kwarg).
    assert phase not in {"done", "failed", "running", "pending", "queued", "started"}, phase
    meta["phase"] = phase
    meta.update({"generated_utc": _utcnow(), "recipe": RECIPE_TAG, "issue": ISSUE, "label": LABEL})
    return meta


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _cfg_from_args(args) -> dict:
    import issue2474_fit as FIT

    if args.synthetic_root:
        root = Path(args.synthetic_root)
        conds = ["smoke_condA", "smoke_condB"]
        return {
            "synthetic": True,
            "layer": 1,
            "settings": ("smoke",),
            "conds": {"smoke": conds},
            "cond_setting": {c: "smoke" for c in conds},
            "expected_rows": None,
            "expected_grid_rows": None,
            "p_inoc": {"smoke": "smoke_p"},
            "capture_root": root / "capture" / "predictor_captures",
            "train_dir": root / "train",
            "out_dir": root / "out",
            "means_dir": root / "out" / "trainref_means",
            "fig_dir": root / "figs",
            "vhat_path": {"smoke": root / "vhat.pt"},
            # Self-consistent pin: exercises the sha-check plumbing; the
            # discriminating perturbation check is a committed pytest.
            "vhat_sha256": {
                "smoke": _sha256_file(root / "vhat.pt") if (root / "vhat.pt").is_file() else None
            },
            "passb_path": root / "passb.pt",
            "prefit_scores_path": root / "prefit_scores.json",
            "prefit_stats_path": root / "prefit_stats.json",
            "rates_path": root / "rates_synth.json",
            "norm_cache": root / "out" / "norm_weight_fetched.json",
            "logs_dir": root / "logs",
            "data_root": root,
        }
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    cond_setting = {c: s for s, cs in FIT.SETTING_CONDS.items() for c in cs}
    return {
        "synthetic": False,
        "layer": LAYER,
        "settings": ("caps", "em"),  # caps first: the champions under test
        "conds": {s: list(FIT.SETTING_CONDS[s]) for s in ("caps", "em")},
        "cond_setting": cond_setting,
        "expected_rows": dict(FIT.EXPECTED_MU_N_C),
        "expected_grid_rows": dict(FIT.EXPECTED_GRID_ROWS),
        "p_inoc": None,  # lazily FIT._p_inoc_labels()
        "capture_root": data_root / FIT.HF_CAPTURE_PREFIX / "predictor_captures",
        "train_dir": data_root / "train_full",
        "out_dir": out_dir,
        "means_dir": out_dir / "trainref_means",
        "fig_dir": Path(args.figures_out),
        "vhat_path": {
            s: data_root / SLUG / "analysis_tensors" / "predicted" / f"base_{s}_L27_vhat.pt"
            for s in ("caps", "em")
        },
        "vhat_sha256": dict(VHAT_SHA256),
        "passb_path": None,  # -> issue2379_mapfit.load_base_bundle (pinned hf download)
        "prefit_scores_path": REPO_ROOT / "eval_results/issue_2474/prefit/prefit_scores.json",
        "prefit_stats_path": REPO_ROOT / "eval_results/issue_2474/prefit/prefit_stats.json",
        "rates_path": None,  # -> issue2474_free_gate.load_rates()
        "norm_cache": data_root / "postnorm" / "norm_weight_fetched.json",
        "logs_dir": Path("/workspace/logs") if Path("/workspace").is_dir() else None,
        "data_root": data_root,
    }


def _selected_conds(args, cfg: dict) -> list[str]:
    all_conds = [c for s in cfg["settings"] for c in cfg["conds"][s]]
    if args.conditions.strip().lower() == "all":
        return all_conds
    sel = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in sel if c not in all_conds]
    if unknown:
        raise RuntimeError(f"--conditions unknown: {unknown} (known: {all_conds})")
    return sel


def _selected_settings(args, cfg: dict) -> list[str]:
    """Explicit rescore setting selection (--settings). Default = every
    registered setting; any narrower selection is a RECORDED descope (rides
    the rescore fingerprint + payload), never an implicit skip."""
    raw = args.settings
    if raw.strip().lower() == "all":
        return list(cfg["settings"])
    sel = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = sorted(sel - set(cfg["settings"]))
    if unknown:
        raise RuntimeError(f"--settings unknown: {unknown} (known: {list(cfg['settings'])})")
    if not sel:
        raise RuntimeError("--settings selected no settings")
    return [s for s in cfg["settings"] if s in sel]


def _p_inoc(cfg: dict) -> dict:
    if cfg["p_inoc"] is not None:
        return cfg["p_inoc"]
    import issue2474_fit as FIT

    return FIT._p_inoc_labels()


def _atomic_write_json(path: Path, payload: dict) -> None:
    import issue2474_fit as FIT

    FIT._atomic_write_json(path, payload)


# ---------------------------------------------------------------------------
# RMSNorm (the ONE transform under test; used identically by every phase)
# ---------------------------------------------------------------------------
def rms_norm_rows(x, w, eps: float):
    """Row-wise Qwen2 final RMSNorm under the module's OWN finite-dtype convention.

    Reproduces ``Qwen2RMSNorm.forward`` (transformers modeling_qwen2) op-for-op:
    upcast to fp32 -> variance = mean(x^2) -> x * rsqrt(variance + eps) -> cast
    the normalized states BACK to the input dtype -> multiply by the
    (input-dtype) weight. Input-dtype policy: bfloat16 for EVERY consumer — the
    production hidden-state dtype. Live GPU states arrive bf16 (round-trip
    exact through the fp32/fp64 numpy hop); fp16-persisted artifacts are
    bf16-representable over the states' range (fp16 mantissa strictly wider),
    so the cast recovers the production value; synthetic fp32/fp64 smoke
    inputs take one bf16 rounding, identically on both sides of any pre/post
    comparison. ``x`` is (..., H); ``w`` is (H,). Returns fp64 (of the bf16
    values) for the downstream cosine math. This is the module's single
    post-norm operator — the GPU phase and the offline re-read share it.
    """
    import numpy as np
    import torch

    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float32))
    assert x.shape[-1] == w.shape[0], (x.shape, w.shape)
    xt = torch.from_numpy(x).to(torch.bfloat16)
    wt = torch.from_numpy(w).to(torch.bfloat16)
    hidden_states = xt.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out = wt * hidden_states.to(torch.bfloat16)
    return out.to(torch.float64).numpy()


def _sha256_file(path: Path) -> str:
    """Chunked sha256 over a file's bytes (artifact identity pins)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_bf16_le(raw: bytes):
    """Decode little-endian bf16 bytes -> fp32 numpy (bf16 = upper 16 bits of fp32)."""
    import numpy as np

    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).copy()


def norm_weight_sha(w_fp32) -> str:
    """Canonical norm-weight identity: sha256 over the fp32 little-endian bytes."""
    import numpy as np

    return _sha256_bytes(np.asarray(w_fp32, dtype="<f4").tobytes())


def fetch_norm_weight_ranged(model_id: str, revision: str | None) -> dict:
    """Fetch ONLY ``model.norm.weight`` via a ranged safetensors read (~7 KB).

    Resolves ``revision`` (None -> pin main to a sha ONCE), downloads the tiny
    config + safetensors index, then range-reads the tensor bytes from the
    owning shard through ``HfFileSystem`` — never the multi-GB shard file.
    Every Hub call rides ``hub.retry_transient``.
    """
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    if revision is None:
        revision = hub.retry_transient(
            lambda: api.model_info(model_id).sha, what=f"resolve {model_id} revision"
        )
    cfg_path = hub.retry_transient(
        lambda: hf_hub_download(model_id, "config.json", revision=revision),
        what="fetch config.json",
    )
    mcfg = json.loads(Path(cfg_path).read_text())
    eps = float(mcfg["rms_norm_eps"])
    hidden = int(mcfg["hidden_size"])
    idx_path = hub.retry_transient(
        lambda: hf_hub_download(model_id, "model.safetensors.index.json", revision=revision),
        what="fetch safetensors index",
    )
    shard = json.loads(Path(idx_path).read_text())["weight_map"][NORM_WEIGHT_NAME]

    def _ranged() -> tuple[dict, bytes]:
        fs = HfFileSystem()
        with fs.open(f"{model_id}/{shard}", "rb", revision=revision) as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
            ent = header[NORM_WEIGHT_NAME]
            start, end = ent["data_offsets"]
            f.seek(8 + hlen + start)
            return ent, f.read(end - start)

    ent, raw = hub.retry_transient(_ranged, what=f"ranged read {NORM_WEIGHT_NAME}")
    if ent["dtype"] != "BF16" or ent["shape"] != [hidden]:
        raise RuntimeError(f"unexpected {NORM_WEIGHT_NAME} entry: {ent}")
    w = decode_bf16_le(raw)
    import numpy as np

    if not np.isfinite(w).all():
        raise RuntimeError(f"{NORM_WEIGHT_NAME} carries NaN/Inf after bf16 decode")
    return {
        "model_id": model_id,
        "model_revision": revision,
        "rms_norm_eps": eps,
        "hidden_size": hidden,
        "sha256_fp32le": norm_weight_sha(w),
        "weight_fp32": [float(v) for v in w],
        "source": "ranged-fetch",
    }


# ---------------------------------------------------------------------------
# Fused teacher-forced capture (ONE forward per row; causally identical to the
# parent's two-helper path — gated by the spot-equivalence + Gate P checks)
# ---------------------------------------------------------------------------
def fused_capture(model, tokenizer, messages, response: str, layer: int):
    """(v_c, resp_stack) at ONE block index from ONE teacher-forced forward.

    v_c = last-PROMPT-token pre-norm state (identical, by causal masking, to the
    prompt-only forward the parent's ``capture_context_vector`` runs); resp_stack
    = the RESPONSE-token pre-norm states (the parent's ``capture_answer_vector``
    span). Fp32 numpy on CPU. Raises on an empty/unalignable gold answer (the
    parent's registered ALL-rows grain — never a silently shrunk mean).
    """
    import issue779_common as I779C
    import numpy as np

    from explore_persona_space.analysis.extraction import extract_layer_activations

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
    suffix = tokenizer.decode(prompt_ids[0, -3:])
    if suffix != I779C.GENERATION_SUFFIX:
        raise RuntimeError(f"position assert: last-3 decode {suffix!r} != generation suffix")
    prompt_len = int(prompt_ids.shape[1])
    full_messages = [*messages, {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(model.device)
    full_len = int(full_inputs["input_ids"].shape[1])
    if full_len <= prompt_len:
        raise RuntimeError("empty/unalignable gold answer — registered grain is ALL rows")
    captured = extract_layer_activations(
        model, full_inputs["input_ids"], [layer], attention_mask=full_inputs.get("attention_mask")
    )
    hs = captured[layer][0]  # (T, H) pre-norm block output
    v_c = hs[prompt_len - 1].float().cpu().numpy().astype(np.float64)
    resp = hs[prompt_len:full_len].float().cpu().numpy().astype(np.float64)
    return v_c, resp


def _cos(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def spot_equivalence(model, tokenizer, messages, response, layer, v_c, resp, *, bars) -> dict:
    """Fused-vs-two-helper equivalence on ONE row (the parent capture path)."""
    import issue779_collect as I779

    ref_c = I779.capture_context_vector(model, tokenizer, messages, [layer])["last"][0].numpy()
    av = I779.capture_answer_vector(
        model, tokenizer, messages, response, [layer], {}, keep_per_token=True
    )
    if av is None:
        raise RuntimeError("helper path returned None on a row the fused path accepted")
    ref_vx = av["v_x"][0].numpy()
    n_ref = int(av["per_token"].shape[0])
    if n_ref != resp.shape[0]:
        raise RuntimeError(
            f"spot gate: response token count fused={resp.shape[0]} != helper={n_ref}"
        )
    cos_c = _cos(v_c, ref_c)
    cos_a = _cos(resp.mean(axis=0), ref_vx)
    if cos_c < bars["single_pos"] or cos_a < bars["token_mean"]:
        raise RuntimeError(
            f"spot gate FAIL: cos_c={cos_c:.6f} (bar {bars['single_pos']}) "
            f"cos_a={cos_a:.6f} (bar {bars['token_mean']})"
        )
    return {"cos_context": cos_c, "cos_answer_mean": cos_a, "n_resp_tokens": n_ref}


# ---------------------------------------------------------------------------
# Phase: trainref-gpu (P1 — per-condition post-norm training-mix means, L27)
# ---------------------------------------------------------------------------
class _SmokeInterrupt(RuntimeError):
    """Smoke-only test hook: simulated mid-condition death after N rows."""


def _load_production_model(args):
    """bf16 model at a resolved-once revision + tokenizer + norm weight/eps."""
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.orchestrate import hub

    revision = args.model_revision
    if not revision:
        revision = hub.retry_transient(
            lambda: HfApi().model_info(args.model).sha, what="resolve model revision"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map={"": torch.device(device)},
    )
    model.eval()
    n_layers = len(model.model.layers)
    if LAYER >= n_layers:
        raise RuntimeError(f"layer {LAYER} out of range for {n_layers}-block model")
    if model.config.hidden_size != EXPECTED_HIDDEN:
        raise RuntimeError(f"hidden {model.config.hidden_size} != {EXPECTED_HIDDEN}")
    return model, tokenizer, revision


def _model_norm(model) -> tuple[list[float], float, str]:
    """(fp32 norm weight list, eps, sha) read directly off the loaded model."""
    import torch

    w = model.model.norm.weight.detach().to(torch.float32).cpu().numpy()
    eps = float(model.config.rms_norm_eps)
    return [float(v) for v in w], eps, norm_weight_sha(w)


def _stage_train_jsonl(cfg: dict, cond: str) -> Path:
    """Local-first -> HF fetch of the FULL training mix for one condition."""
    target = cfg["train_dir"] / f"{cond}.jsonl"
    if target.is_file() and cfg["synthetic"]:
        return target
    if not target.is_file():
        from explore_persona_space.orchestrate import hub

        target.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(hub.DEFAULT_DATASET_REPO, f"{HF_TRAIN_PREFIX}/{cond}.jsonl", target)
    return target


def _stage_stored_mu(cfg: dict, cond: str) -> Path:
    import issue2474_fit as FIT

    target = cfg["capture_root"] / f"base_mu_{cond}" / "mu.pt"
    if not target.is_file():
        if cfg["synthetic"]:
            raise RuntimeError(f"smoke tree missing synthetic mu bundle {target}")
        from explore_persona_space.orchestrate import hub

        target.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            f"{FIT.HF_CAPTURE_PREFIX}/predictor_captures/base_mu_{cond}/mu.pt",
            target,
        )
    return target


def _validate_train_rows(path: Path) -> int:
    """Validate EVERY row's schema against the exact fields the capture loop
    dereferences (prompt[0]=system, prompt[-1]=user, completion[0].content
    non-empty). Pre-GPU fail-loud (r1 codex: gpu-input-contract-post-work).
    Returns the non-blank row count."""
    n = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path.name}:{i + 1}: invalid JSON ({exc})") from exc
            prompt = row.get("prompt")
            comp = row.get("completion")
            if (
                not isinstance(prompt, list)
                or len(prompt) < 2
                or not all(isinstance(m, dict) for m in prompt)
                or prompt[0].get("role") != "system"
                or prompt[-1].get("role") != "user"
                or not isinstance(prompt[0].get("content"), str)
                or not isinstance(prompt[-1].get("content"), str)
            ):
                raise RuntimeError(f"{path.name}:{i + 1}: prompt schema invalid")
            if (
                not isinstance(comp, list)
                or not comp
                or not isinstance(comp[0], dict)
                or not isinstance(comp[0].get("content"), str)
                or not comp[0]["content"].strip()
            ):
                raise RuntimeError(f"{path.name}:{i + 1}: completion schema invalid/empty")
            n += 1
    if n == 0:
        raise RuntimeError(f"{path.name}: no rows")
    return n


def _validate_stored_mu(mu_path: Path, layer: int, expect_rows: int, hidden: int | None) -> None:
    """Stored-mu schema/shape/count/finiteness contract, checked BEFORE any
    model load / GPU forward (r1 codex: gpu-input-contract-post-work)."""
    import issue2379_mapfit as mf
    import numpy as np

    stored = mf._torch_load_constrained(mu_path)
    missing = [k for k in ("mu_train", "mu_a_train", "n_c", "n_a") if k not in stored]
    if missing:
        raise RuntimeError(f"{mu_path}: stored mu missing keys {missing}")
    for key in ("mu_train", "mu_a_train"):
        t = stored[key]
        if int(t.shape[0]) <= layer:
            raise RuntimeError(f"{mu_path}: {key} has {int(t.shape[0])} layers — no layer {layer}")
        vec = np.asarray(t[layer], dtype=np.float64)
        if vec.ndim != 1 or (hidden is not None and vec.shape[0] != hidden):
            raise RuntimeError(f"{mu_path}: {key}[{layer}] shape {vec.shape} != ({hidden},)")
        if not np.isfinite(vec).all():
            raise RuntimeError(f"{mu_path}: {key}[{layer}] carries NaN/Inf")
    if int(stored["n_c"]) != expect_rows:
        raise RuntimeError(
            f"{mu_path}: stored mu n_c={int(stored['n_c'])} != train rows {expect_rows}"
        )


def _gate_p_check(mu_c_pre, mu_a_pre, stored_c, stored_a, cond: str) -> dict:
    """Gate P: pre-norm reproduction of the stored streaming means (cos >= 0.999).

    Raises on FAIL; returns the gate record dict on PASS."""
    import numpy as np

    mu_c_pre = np.asarray(mu_c_pre, dtype=np.float64)
    mu_a_pre = np.asarray(mu_a_pre, dtype=np.float64)
    cos_c = _cos(mu_c_pre, stored_c)
    cos_a = _cos(mu_a_pre, stored_a)
    rel_c = float(np.linalg.norm(mu_c_pre - stored_c) / (np.linalg.norm(stored_c) + 1e-12))
    rel_a = float(np.linalg.norm(mu_a_pre - stored_a) / (np.linalg.norm(stored_a) + 1e-12))
    if cos_c < 0.999 or cos_a < 0.999:
        raise RuntimeError(
            f"{cond}: Gate P FAIL — pre-norm mean reproduction cos_c={cos_c:.6f} "
            f"cos_a={cos_a:.6f} (bar 0.999); recipe drift vs the stored mu.pt"
        )
    return {
        "cos_c_pre_vs_stored": cos_c,
        "cos_a_pre_vs_stored": cos_a,
        "rel_l2_c": rel_c,
        "rel_l2_a": rel_a,
        "verdict": "PASS",
    }


def _trainref_fingerprint(cfg: dict, cond: str, train_path: Path, n_rows: int, model_ident: str):
    return {
        "phase": "trainref-gpu",
        "recipe": RECIPE_TAG,
        "cond": cond,
        "setting": cfg["cond_setting"][cond],
        "layer": cfg["layer"],
        "model_ident": model_ident,
        "train_jsonl": train_path.name,
        "train_bytes": train_path.stat().st_size,
        "n_rows": n_rows,
        "v": 1,
    }


def _save_partial(
    partial: Path, fp: dict, sums: dict, counts: dict, next_line_idx: int, spot_records: list
) -> None:
    import numpy as np

    tmp = partial.with_name(partial.stem + ".tmp.npz")  # suffix stays .npz (np.savez appends)
    np.savez(
        tmp,
        fingerprint=np.array(json.dumps(fp)),
        next_line_idx=np.int64(next_line_idx),
        n_rows_done=np.int64(counts["n"]),
        spot_json=np.array(json.dumps(spot_records)),
        **{k: v for k, v in sums.items()},
    )
    os.replace(tmp, partial)


def _load_partial(partial: Path, fp: dict):
    import numpy as np

    if not partial.is_file():
        return None
    with np.load(partial) as z:
        try:
            if json.loads(str(z["fingerprint"])) != fp:
                return None
            # Spot-gate evidence rides the partial (a resumed condition must
            # not finish with an empty spot_equivalence list — r1 codex NIT
            # resumed-spot-evidence-lost). A corrupt partial -> recompute.
            spot = json.loads(str(z["spot_json"])) if "spot_json" in z.files else []
        except Exception:
            return None
        sums = {k: np.asarray(z[k]) for k in z.files if k.startswith("sum_")}
        return {
            "sums": sums,
            "n": int(z["n_rows_done"]),
            "next_line_idx": int(z["next_line_idx"]),
            "spot": spot,
        }


def phase_trainref_gpu(args, cfg: dict) -> dict:
    """Per-condition post-norm (+pre-norm parity) training-mix mean states.

    Streaming-reduce (per-row activations never stored — the parent's declared
    discard, regeneration recipe = this phase). Checkpoint every --ckpt-every
    rows; resume keyed on condition + convention (RECIPE_TAG) + model ident +
    train-file identity. Model/tokenizer come from cfg["model_loader"] when the
    smoke injects a tiny CPU model, else the production loader.
    """
    import issue2379_capture as CAP
    import issue2379_mapfit as mf
    import numpy as np

    print(f"[phase=trainref_gpu] start (conds={args.conditions})", flush=True)
    conds = _selected_conds(args, cfg)
    cfg["means_dir"].mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    model_ident = None
    norm_w = norm_eps = norm_sha = None
    bars = (
        {"single_pos": 0.99999, "token_mean": 0.99999}
        if cfg["synthetic"]
        else {"single_pos": 0.995, "token_mean": 0.999}
    )
    summary: dict = {"conds": {}, "resumed": [], "skipped": []}
    layer = cfg["layer"]
    t0 = time.time()

    def _ensure_model():
        nonlocal model, tokenizer, model_ident, norm_w, norm_eps, norm_sha
        if model is not None:
            return
        if cfg.get("model_loader") is not None:
            model, tokenizer, model_ident = cfg["model_loader"]()
        else:
            model, tokenizer, rev = _load_production_model(args)
            model_ident = f"hf:{args.model}@{rev}"
        norm_w, norm_eps, norm_sha = _model_norm(model)
        norm_path = cfg["means_dir"] / "norm_weight.json"
        _atomic_write_json(
            norm_path,
            {
                "model_ident": model_ident,
                "rms_norm_eps": norm_eps,
                "sha256_fp32le": norm_sha,
                "weight_fp32": norm_w,
                "source": "model-load",
                "metadata": _metadata("trainref-gpu"),
            },
        )

    # ---- Preflight: stage + validate EVERY input (all rows of every train
    # file + every stored-mu bundle) BEFORE any model load / GPU forward
    # (r1 codex blocker gpu-input-contract-post-work).
    staged: dict[str, dict] = {}
    for cond in conds:
        train_path = _stage_train_jsonl(cfg, cond)
        n_rows = _validate_train_rows(train_path)
        if cfg["expected_rows"] is not None and n_rows != cfg["expected_rows"][cond]:
            raise RuntimeError(
                f"{cond}: staged train rows {n_rows} != registered {cfg['expected_rows'][cond]}"
            )
        CAP.validate_mu_train_jsonl(train_path)  # parent first-row contract, kept for parity
        mu_path = _stage_stored_mu(cfg, cond)
        _validate_stored_mu(mu_path, layer, n_rows, None if cfg["synthetic"] else EXPECTED_HIDDEN)
        staged[cond] = {"train": train_path, "n_rows": n_rows, "mu": mu_path}
    print(
        f"[trainref-gpu] preflight OK: {len(conds)} conds staged + validated "
        f"(all train rows + stored-mu contracts) pre-GPU",
        flush=True,
    )

    for ci, cond in enumerate(conds):
        out_json = cfg["means_dir"] / f"{cond}.json"
        train_path = staged[cond]["train"]
        n_rows = staged[cond]["n_rows"]
        # Resume predicate BEFORE the model load: fingerprint sans model ident
        # first (cheap skip), full fingerprint re-checked after load.
        if out_json.is_file() and not args.force:
            cached = json.loads(out_json.read_text())
            cfp = cached.get("fingerprint", {})
            probe = {k: v for k, v in cfp.items() if k != "model_ident"}
            _ensure_model()
            want = _trainref_fingerprint(cfg, cond, train_path, n_rows, model_ident)
            if cfp == want or (
                args.allow_model_revision_drift
                and probe == {k: v for k, v in want.items() if k != "model_ident"}
            ):
                summary["skipped"].append(cond)
                print(
                    f"[trainref-gpu] {cond}: output exists (fingerprint match) — skip", flush=True
                )
                continue
        _ensure_model()
        fp = _trainref_fingerprint(cfg, cond, train_path, n_rows, model_ident)
        partial = cfg["means_dir"] / f"{cond}.partial.npz"
        hdim = len(norm_w)
        sums = {
            "sum_c_pre": np.zeros(hdim),
            "sum_c_post": np.zeros(hdim),
            "sum_a_pre": np.zeros(hdim),
            "sum_a_post_row": np.zeros(hdim),
            "sum_a_post_tok": np.zeros(hdim),
        }
        n_done = 0
        next_line_idx = 0
        spot_records: list[dict] = []
        st = _load_partial(partial, fp)
        if st is not None:
            sums = st["sums"]
            n_done = st["n"]
            next_line_idx = st["next_line_idx"]
            spot_records = list(st["spot"])
            summary["resumed"].append({"cond": cond, "resumed_at_row": n_done})
            print(f"[trainref-gpu] {cond}: resuming at row {n_done}/{n_rows}", flush=True)
        w64 = np.asarray(norm_w, dtype=np.float64)
        with train_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < next_line_idx or not line.strip():
                    continue
                row = json.loads(line)
                messages = [
                    {"role": "system", "content": row["prompt"][0]["content"]},
                    {"role": "user", "content": row["prompt"][-1]["content"]},
                ]
                gold = row["completion"][0]["content"]
                v_c, resp = fused_capture(model, tokenizer, messages, gold, layer)
                if len(spot_records) < args.spot_rows and n_done < args.spot_rows:
                    spot_records.append(
                        spot_equivalence(
                            model, tokenizer, messages, gold, layer, v_c, resp, bars=bars
                        )
                    )
                sums["sum_c_pre"] += v_c
                sums["sum_c_post"] += rms_norm_rows(v_c, w64, norm_eps)
                v_x = resp.mean(axis=0)
                sums["sum_a_pre"] += v_x
                sums["sum_a_post_row"] += rms_norm_rows(v_x, w64, norm_eps)
                sums["sum_a_post_tok"] += rms_norm_rows(resp, w64, norm_eps).mean(axis=0)
                n_done += 1
                if n_done % args.ckpt_every == 0:
                    _save_partial(partial, fp, sums, {"n": n_done}, idx + 1, spot_records)
                    print(
                        f"[trainref-gpu] unit {n_done}/{n_rows} {cond} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )
                if args.smoke_interrupt_after and n_done >= args.smoke_interrupt_after:
                    _save_partial(partial, fp, sums, {"n": n_done}, idx + 1, spot_records)
                    raise _SmokeInterrupt(f"smoke interrupt hook after {n_done} rows")
        if n_done != n_rows:
            raise RuntimeError(f"{cond}: consumed {n_done} rows != expected {n_rows}")
        means = {k.replace("sum_", "mu_"): (v / n_done) for k, v in sums.items()}
        # Gate P: pre-norm reproduction of the stored streaming means at `layer`.
        stored = mf._torch_load_constrained(staged[cond]["mu"])
        stored_c = np.asarray(stored["mu_train"][layer], dtype=np.float64)
        stored_a = np.asarray(stored["mu_a_train"][layer], dtype=np.float64)
        if int(stored["n_c"]) != n_done:
            raise RuntimeError(f"{cond}: stored mu n_c={stored['n_c']} != re-forward rows {n_done}")
        gate_p = _gate_p_check(means["mu_c_pre"], means["mu_a_pre"], stored_c, stored_a, cond)
        cos_c = gate_p["cos_c_pre_vs_stored"]
        cos_a = gate_p["cos_a_pre_vs_stored"]
        payload = {
            "fingerprint": fp,
            "cond": cond,
            "setting": cfg["cond_setting"][cond],
            "layer": layer,
            "n_rows": n_done,
            "model_ident": model_ident,
            "rms_norm_eps": norm_eps,
            "norm_weight_sha256": norm_sha,
            "gate_p": gate_p,
            "spot_equivalence": spot_records,
            "grain_note": (
                "mu_a_post_rowgrain = mean over rows of RMSNorm(token-mean answer state) "
                "(PRIMARY: matches the offline re-read grain); mu_a_post_tokengrain = mean "
                "over rows of token-mean of RMSNorm(per-token states) (sensitivity)."
            ),
            **{
                k: [float(np.float32(v)) for v in vec]
                for k, vec in (
                    ("mu_c_pre", means["mu_c_pre"]),
                    ("mu_a_pre", means["mu_a_pre"]),
                    ("mu_c_post", means["mu_c_post"]),
                    ("mu_a_post_rowgrain", means["mu_a_post_row"]),
                    ("mu_a_post_tokengrain", means["mu_a_post_tok"]),
                )
            },
            "metadata": _metadata("trainref-gpu"),
        }
        _atomic_write_json(out_json, payload)
        partial.unlink(missing_ok=True)
        summary["conds"][cond] = {
            "n_rows": n_done,
            "gate_p": payload["gate_p"],
            "spot": spot_records,
        }
        print(
            f"[trainref-gpu] cond {ci + 1}/{len(conds)} {cond} done "
            f"(gate_p cos_c={cos_c:.6f} cos_a={cos_a:.6f}) elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    if not cfg["synthetic"] and not args.skip_upload:
        _upload_means(cfg)
        summary["upload"] = "verified"
    _write_sentinel(args, cfg, phase="trainref-gpu", note_payload=summary)
    return summary


def _upload_means(cfg: dict) -> None:
    """One folder commit of the trainref means + exact-set retried verify."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    means_dir = cfg["means_dir"]
    files = sorted(p.name for p in means_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"upload: no means JSONs under {means_dir}")
    dest = f"{HF_POSTNORM_PREFIX}/trainref_means"
    base_url = hub._upload(
        means_dir,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        dest,
        # Scope the commit to the verified deliverable set: never ship stale
        # checkpoint residue from an interrupted condition (r1 codex NIT
        # means-upload-includes-partials). The means dir's declared artifact
        # classes are ONLY the per-cond + norm-weight JSONs.
        ignore_patterns=["*.partial.npz", "*.tmp.npz"],
        raise_on_error=True,
    )
    if not base_url:
        # _upload is fail-soft by RETURN for missing HF_TOKEN / absent local
        # path even under raise_on_error=True — an empty return is a silent
        # durability loss, never warning-and-continue (upload-policy.md).
        raise RuntimeError(f"upload returned no path for {means_dir} -> {dest}")
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        hub.DEFAULT_DATASET_REPO,
        [f"{dest}/{name}" for name in files],
        path_in_repo=dest,
    )
    if missing:
        raise RuntimeError(f"upload verify: missing remote paths {missing}")
    print(f"[trainref-gpu] uploaded + verified {len(files)} files -> {dest}", flush=True)


def _write_sentinel(args, cfg: dict, *, phase: str, note_payload: dict) -> None:
    """poll_pipeline-conformant end-of-phase sentinel (smoke -> smoke tree).

    Production writes ONLY when /workspace/logs exists (pod-side); a VM-side
    offline run has no poller and skips with a log line. Resume/finalize state
    never lives in this namespace (the drain renames sentinels .processed).
    """
    logs_dir = cfg["logs_dir"]
    if logs_dir is None:
        print(f"[{phase}] no /workspace/logs on this host — sentinel skipped", flush=True)
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    kind = "epm:smoke-result" if cfg["synthetic"] else "epm:progress"
    kind_slug = kind.replace(":", "_")
    dest = logs_dir / f"issue-{ISSUE}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": ISSUE,
        "by": "issue2474_postnorm",
        "ts": _utcnow(),
        "blocks_pipeline": False,
        "smoke": bool(cfg["synthetic"]),
        "note": json.dumps({"phase": phase, "label": LABEL, **note_payload}, sort_keys=True)[
            :40000
        ],
    }
    _atomic_write_json(dest, payload)
    print(f"[{phase}] sentinel written: {dest}", flush=True)


# ---------------------------------------------------------------------------
# Phase: rescore (P2 — offline post-norm re-read + Gate R)
# ---------------------------------------------------------------------------
def _load_means(args, cfg: dict, selected: list[str]) -> dict:
    """{setting: {cond: means dict}} for EVERY selected setting — fail loud on
    ANY absent/unstageable file (transport, auth and remote-absence all raise;
    r1 codex blocker secondary-setting-stage-fail-soft). The ONLY way to run
    fewer settings is the explicit --settings descope, which rides the rescore
    fingerprint + output payload."""
    from explore_persona_space.orchestrate import hub

    out: dict = {}
    for setting in selected:
        got: dict = {}
        for cond in cfg["conds"][setting]:
            path = cfg["means_dir"] / f"{cond}.json"
            if not path.is_file():
                if cfg["synthetic"]:
                    raise RuntimeError(f"smoke tree missing trainref means {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    hub.stage_hub_file(
                        hub.DEFAULT_DATASET_REPO,
                        f"{HF_POSTNORM_PREFIX}/trainref_means/{cond}.json",
                        path,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{cond}: trainref means not stageable from "
                        f"{HF_POSTNORM_PREFIX}/trainref_means — staging/transport/auth "
                        f"failure, or the GPU phase has not produced+uploaded them yet; "
                        f"NEVER a silent {setting!r} descope (use --settings to descope "
                        f"explicitly)"
                    ) from exc
            got[cond] = json.loads(path.read_text())
        _validate_means_setting(cfg, setting, got)
        out[setting] = got
    return out


def _validate_means_setting(cfg: dict, setting: str, got: dict) -> None:
    """Schema/identity validation of one setting's staged trainref means:
    layer/recipe, cond/setting identity, registered n_rows, Gate P verdict,
    required vectors present + finite + consistent hidden dim, one model
    ident across conds."""
    import numpy as np

    vec_keys = ("mu_c_pre", "mu_a_pre", "mu_c_post", "mu_a_post_rowgrain", "mu_a_post_tokengrain")
    idents = set()
    for cond, m in got.items():
        if m.get("layer") != cfg["layer"] or m.get("fingerprint", {}).get("recipe") != RECIPE_TAG:
            raise RuntimeError(f"{cond}: means layer/recipe mismatch ({m.get('layer')})")
        if m.get("cond") != cond or m.get("setting") != setting:
            raise RuntimeError(
                f"{cond}: means cond/setting mismatch ({m.get('cond')}/{m.get('setting')})"
            )
        if cfg["expected_rows"] is not None and m.get("n_rows") != cfg["expected_rows"][cond]:
            raise RuntimeError(
                f"{cond}: means n_rows {m.get('n_rows')} != registered {cfg['expected_rows'][cond]}"
            )
        if m.get("gate_p", {}).get("verdict") != "PASS":
            raise RuntimeError(f"{cond}: means carry no Gate P PASS verdict")
        dims = set()
        for key in vec_keys:
            v = m.get(key)
            if not isinstance(v, list) or not v:
                raise RuntimeError(f"{cond}: means missing/empty vector {key}")
            arr = np.asarray(v, dtype=np.float64)
            if not np.isfinite(arr).all():
                raise RuntimeError(f"{cond}: means vector {key} carries NaN/Inf")
            dims.add(arr.shape[0])
        if len(dims) != 1:
            raise RuntimeError(f"{cond}: means vectors disagree on hidden dim {sorted(dims)}")
        if not cfg["synthetic"] and dims != {EXPECTED_HIDDEN}:
            raise RuntimeError(f"{cond}: means hidden dim {sorted(dims)} != {EXPECTED_HIDDEN}")
        idents.add(m.get("model_ident"))
    if len(idents) != 1:
        raise RuntimeError(f"{setting}: means disagree on model ident: {sorted(map(str, idents))}")


def _resolve_norm_weight(args, cfg: dict, means: dict) -> dict:
    """Norm-weight resolution + cross-checks against every means JSON's sha."""
    import numpy as np

    shas = {m["norm_weight_sha256"] for conds in means.values() for m in conds.values()}
    if len(shas) != 1:
        raise RuntimeError(f"trainref means disagree on norm-weight sha: {sorted(shas)}")
    want_sha = next(iter(shas))
    local = cfg["means_dir"] / "norm_weight.json"
    if cfg["synthetic"]:
        rec = json.loads(local.read_text())
    else:
        if not local.is_file():
            from explore_persona_space.orchestrate import hub

            local.parent.mkdir(parents=True, exist_ok=True)
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                f"{HF_POSTNORM_PREFIX}/trainref_means/norm_weight.json",
                local,
            )
        rec = json.loads(local.read_text())
        cache = cfg["norm_cache"]
        if cache.is_file():
            fetched = json.loads(cache.read_text())
        else:
            revision = rec["model_ident"].split("@")[-1] if "@" in rec["model_ident"] else None
            fetched = fetch_norm_weight_ranged(MODEL_ID, revision)
            cache.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(cache, fetched)
        if fetched["sha256_fp32le"] != rec["sha256_fp32le"]:
            raise RuntimeError(
                "norm-weight mismatch: ranged fetch sha "
                f"{fetched['sha256_fp32le'][:16]} != model-load sha "
                f"{rec['sha256_fp32le'][:16]} — revision drift between GPU phase and rescore"
            )
    if rec["sha256_fp32le"] != want_sha:
        raise RuntimeError("norm_weight.json sha != per-condition means sha")
    w = np.asarray(rec["weight_fp32"], dtype=np.float64)
    return {"w": w, "eps": float(rec["rms_norm_eps"]), "sha": want_sha, "ident": rec["model_ident"]}


def _stage_offline_inputs(cfg: dict, selected: list[str]) -> None:
    """Grid/ceiling/mu bundles + per-setting vhat tensors for the SELECTED
    settings (production only; staging failures raise — never a skip)."""
    import issue2474_fit as FIT

    if cfg["synthetic"]:
        return
    stage_cfg = {
        "synthetic": False,
        "settings": tuple(selected),
        "conds": {s: tuple(cfg["conds"][s]) for s in selected},
        "data_root": cfg["data_root"],
    }
    FIT._stage_capture(stage_cfg)
    from explore_persona_space.orchestrate import hub

    for setting in selected:
        path = cfg["vhat_path"][setting]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                f"{SLUG}/analysis_tensors/predicted/base_{setting}_L27_vhat.pt",
                path,
            )


def _load_vhat(cfg: dict, setting: str, n_rows: int, hidden: int):
    """Load + identity-verify the persisted parent map output (v_hat).

    Vector-level identity: the staged file's sha256 must match the pinned
    parent-producer hash (r1 codex blocker map-vhat-vector-identity-unverified)
    — Gate R's averaged pre-norm projections alone cannot see every component.
    Plus schema, layer/setting, row count, hidden dim and finiteness."""
    import issue2379_mapfit as mf
    import numpy as np

    path = cfg["vhat_path"][setting]
    expected_sha = cfg["vhat_sha256"].get(setting)
    if expected_sha is None:
        if not cfg["synthetic"]:
            raise RuntimeError(f"no pinned parent-producer sha256 for vhat setting {setting!r}")
        print(f"[rescore] vhat sha pin unavailable for synthetic {setting} — skipped", flush=True)
    else:
        got = _sha256_file(path)
        if got != expected_sha:
            raise RuntimeError(
                f"{path.name}: sha256 {got[:16]} != pinned parent-producer sha "
                f"{expected_sha[:16]} — staged vhat is not the parent map output"
            )
    tb = mf._torch_load_constrained(path)
    for key in ("v_hat_mapB", "layer", "setting"):
        if key not in tb:
            raise RuntimeError(f"{path.name}: missing key {key} (realized {sorted(tb.keys())})")
    if int(tb["layer"]) != cfg["layer"] or tb["setting"] != setting:
        raise RuntimeError(f"{path.name}: layer/setting mismatch {tb['layer']}/{tb['setting']}")
    v_hat = np.asarray(tb["v_hat_mapB"], dtype=np.float64)
    if v_hat.shape[0] != n_rows:
        raise RuntimeError(f"{path.name}: rows {v_hat.shape[0]} != grid rows {n_rows}")
    if v_hat.ndim != 2 or v_hat.shape[1] != hidden:
        raise RuntimeError(f"{path.name}: shape {v_hat.shape} != ({n_rows}, {hidden})")
    if not np.isfinite(v_hat).all():
        raise RuntimeError(f"{path.name}: v_hat carries NaN/Inf")
    return v_hat


def _ib_bias_l(cfg: dict):
    """ib_bias at the target layer, recomputed EXACTLY as the parent fit worker
    (pinned pass-B bundle, same split seed, same helper) — fp64 (H,)."""
    import issue2379_mapfit as mf
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    b = mf.load_base_bundle(cfg["passb_path"])
    x = np.asarray(b["x"][:, cfg["layer"], :], dtype=np.float64)
    y = np.asarray(b["y"][:, cfg["layer"], :], dtype=np.float64)
    tr_idx, _ = mf._split_indices(x.shape[0])
    return identity_bias_predict(x[tr_idx], y[tr_idx], np.zeros((1, x.shape[1])))[0]


def _grid_indexing(grid: dict, expected_rows: int | None, setting: str):
    """(labels, row_of, trig/q arrays) mirroring issue2474_fit.phase_scores."""
    import issue2474_fit as FIT
    import numpy as np

    labels = FIT._labels_from_row_meta(grid["row_meta"])
    meta = grid["row_meta"]
    trig_of = np.array([r["trigger_idx"] for r in meta])
    q_of = np.array([r["q_sim_idx"] for r in meta])
    n_t, n_q = len(labels), int(q_of.max()) + 1
    if expected_rows is not None and len(meta) != expected_rows:
        raise RuntimeError(f"{setting}: grid rows {len(meta)} != registered {expected_rows}")
    row_of = -np.ones((n_t, n_q), dtype=int)
    row_of[trig_of, q_of] = np.arange(len(meta))
    if not (row_of >= 0).all():
        raise RuntimeError(f"{setting}: grid rows missing for some (trigger, q) cells")
    return labels, row_of


def _gate_r(setting: str, res: dict, prefit: dict, conds: list[str], layer: int) -> dict:
    """Pre-norm reproduction gate vs the parent prefit_scores.json at `layer`."""
    import numpy as np

    realized: dict[str, float] = {}
    for cond in conds:
        stored_f = prefit["conditions"][cond]["families_layered"]
        for fam in ALL_FAMS:
            mine = res["cond"][cond].get(fam) if fam in res["cond"][cond] else None
            if mine is None:
                mine = res["shared"][fam]
            theirs = np.array(
                [np.nan if v is None else float(v) for v in stored_f[fam][layer]],
                dtype=np.float64,
            )
            mine = np.asarray(mine, dtype=np.float64)
            if mine.shape != theirs.shape:
                raise RuntimeError(f"Gate R {setting}/{cond}/{fam}: shape mismatch")
            if not np.array_equal(np.isnan(mine), np.isnan(theirs)):
                raise RuntimeError(f"Gate R {setting}/{cond}/{fam}: NaN-pattern mismatch")
            fin = ~np.isnan(mine)
            d = float(np.max(np.abs(mine[fin] - theirs[fin]))) if fin.any() else 0.0
            tol = 2e-3 if fam.startswith("ans_") else 1e-6
            if d > tol:
                raise RuntimeError(
                    f"Gate R FAIL {setting}/{cond}/{fam}: max|diff|={d:.3e} > {tol:.0e} — "
                    f"staged inputs do not reproduce the parent pre-norm scores"
                )
            key = "ans_arms" if fam.startswith("ans_") else "exact_arms"
            realized[key] = max(realized.get(key, 0.0), d)
    return {"verdict": "PASS", "max_abs_diff": realized}


def _file_ident(path) -> dict:
    """Stat identity for a consumed artifact file (size + mtime_ns + name).

    Machine-stable by construction (never a recomputed-float-byte hash;
    code-style.md float-last-bit rule). Conservative direction only: a
    re-staged identical file re-runs, never wrong-skips."""
    st = Path(path).stat()
    return {"name": Path(path).name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _passb_ident(cfg: dict) -> dict:
    """Pass-B bundle identity WITHOUT loading it: pinned HF rev (production)
    or the synthetic file's stat identity (smoke)."""
    if cfg["passb_path"] is not None:
        return _file_ident(cfg["passb_path"])
    import issue2254_preimage as i2254

    return {"pinned": f"hf:{i2254.PASS_B_FILE}@{i2254.HF_REV}"}


def _rescore_fingerprint(cfg: dict, means: dict, norm: dict) -> dict:
    """Completion fingerprint over EVERY artifact phase_rescore consumes:
    per-cond means + stored mu, per-setting grid/ceiling/vhat, prefit scores,
    pass-B identity, norm sha, selected settings/conds (r1 codex blocker
    rescore-fingerprint-omits-inputs — a skip must be impossible when ANY
    consumed input changed)."""
    settings = sorted(means.keys())
    inputs: dict = {}
    for s in settings:
        sdir = cfg["capture_root"] / f"base_{s}"
        inputs[f"grid_{s}"] = _file_ident(sdir / "grid.pt")
        inputs[f"ceiling_{s}"] = _file_ident(sdir / "ceiling.pt")
        inputs[f"vhat_{s}"] = _file_ident(cfg["vhat_path"][s])
        for c in cfg["conds"][s]:
            inputs[f"means_{c}"] = _file_ident(cfg["means_dir"] / f"{c}.json")
            inputs[f"mu_{c}"] = _file_ident(cfg["capture_root"] / f"base_mu_{c}" / "mu.pt")
    return {
        "phase": "rescore",
        "recipe": RECIPE_TAG,
        "layer": cfg["layer"],
        "norm_sha": norm["sha"],
        "settings": settings,
        "conds": {s: list(cfg["conds"][s]) for s in settings},
        "inputs": inputs,
        "passb": _passb_ident(cfg),
        "prefit_scores": _file_ident(cfg["prefit_scores_path"]),
        "vhat_sha_pins": {s: cfg["vhat_sha256"].get(s) for s in settings},
        "n_boot_irrelevant": True,
        "v": 2,
    }


def phase_rescore(args, cfg: dict) -> dict:
    """Offline post-norm re-read of the stored layer-27 states (+ Gate R)."""
    import issue2474_fit as FIT
    import numpy as np

    print("[phase=rescore] start", flush=True)
    out_path = cfg["out_dir"] / "postnorm_scores.json"
    selected = _selected_settings(args, cfg)
    deselected = [s for s in cfg["settings"] if s not in selected]
    if deselected:
        print(f"[rescore] EXPLICIT --settings descope: deselected {deselected}", flush=True)
    means = _load_means(args, cfg, selected)
    norm = _resolve_norm_weight(args, cfg, means)
    _stage_offline_inputs(cfg, selected)
    prefit = json.loads(Path(cfg["prefit_scores_path"]).read_text())
    p_inoc = _p_inoc(cfg)
    layer = cfg["layer"]

    fp = _rescore_fingerprint(cfg, means, norm)
    fp["settings_deselected"] = deselected
    if out_path.is_file() and not args.force:
        cached = json.loads(out_path.read_text())
        if cached.get("fingerprint") == fp:
            print("[rescore] completion fingerprint match — skip (resumed)", flush=True)
            cached["resumed"] = True
            return cached

    ib_bias = _ib_bias_l(cfg)  # setting-independent; pass-B bundle loaded ONCE
    settings_out: dict = {}
    for setting in selected:
        conds = list(cfg["conds"][setting])
        grid = FIT._load_bundle(cfg["capture_root"] / f"base_{setting}" / "grid.pt", "grid")
        ceil = FIT._load_bundle(cfg["capture_root"] / f"base_{setting}" / "ceiling.pt", "ceiling")
        exp_rows = None if cfg["expected_grid_rows"] is None else cfg["expected_grid_rows"][setting]
        labels, row_of = _grid_indexing(grid, exp_rows, setting)
        p_lab = p_inoc[setting]
        p_hits = [i for i, lab in enumerate(labels) if lab == p_lab]
        if len(p_hits) != 1:
            raise RuntimeError(f"{setting}: expected one p_inoc trigger {p_lab!r}")
        p_idx = p_hits[0]
        v_c = np.asarray(grid["v_c"][:, layer, :], dtype=np.float64)
        v_hat = _load_vhat(cfg, setting, v_c.shape[0], v_c.shape[1])
        v_ib = v_c + ib_bias
        c_meta = ceil["row_meta"]
        va_l = np.asarray(ceil["v_a"][:, layer, :], dtype=np.float64)
        c_t = np.array([r["trigger_idx"] for r in c_meta], dtype=int)
        c_q = np.array([r["q_sim_idx"] for r in c_meta], dtype=int)
        c_ri = np.array([r["rollout_idx"] for r in c_meta], dtype=int)
        n_rollouts = int(c_ri.max()) + 1 if len(c_ri) else 1
        # Gate R scores against the STORED mu (the parent's exact reference).
        import issue2379_mapfit as mf

        stored_mu = {
            c: mf._torch_load_constrained(cfg["capture_root"] / f"base_mu_{c}" / "mu.pt")
            for c in conds
        }
        mu_tr_stored = np.stack(
            [np.asarray(stored_mu[c]["mu_train"][layer], dtype=np.float64) for c in conds]
        )
        mu_a_stored = np.stack(
            [np.asarray(stored_mu[c]["mu_a_train"][layer], dtype=np.float64) for c in conds]
        )
        base_args = (row_of, p_idx, va_l, c_t, c_q, c_ri, n_rollouts)
        pre_res = FIT._score_layer_batched(
            v_c, v_hat, v_ib, *base_args, mu_tr_stored, mu_a_stored, conds
        )
        gate_r = _gate_r(setting, pre_res, prefit, conds, layer)
        # Post-norm read: row-wise RMSNorm of every comparison vector — the
        # CEILING rows included (r1 Claude Critical: the post calls must ride
        # `nva`, never the pre-norm `va_l` of `base_args`) — then the SAME
        # scoring core, against the GPU-phase post-norm means.
        nv_c = rms_norm_rows(v_c, norm["w"], norm["eps"])
        nv_hat = rms_norm_rows(v_hat, norm["w"], norm["eps"])
        nv_ib = rms_norm_rows(v_ib, norm["w"], norm["eps"])
        nva = rms_norm_rows(va_l, norm["w"], norm["eps"])
        post_base = (row_of, p_idx, nva, c_t, c_q, c_ri, n_rollouts)
        mu_c_post = np.stack(
            [np.asarray(means[setting][c]["mu_c_post"], dtype=np.float64) for c in conds]
        )
        post_res = {}
        for grain, key in (
            ("post_rowgrain", "mu_a_post_rowgrain"),
            ("post_tokengrain", "mu_a_post_tokengrain"),
        ):
            mu_a_post = np.stack(
                [np.asarray(means[setting][c][key], dtype=np.float64) for c in conds]
            )
            post_res[grain] = FIT._score_layer_batched(
                nv_c, nv_hat, nv_ib, *post_base, mu_c_post, mu_a_post, conds
            )

        def _pack(res: dict) -> dict:
            return {
                "shared": {f: FIT._nan_to_none(list(v)) for f, v in res["shared"].items()},
                "cond": {
                    c: {f: FIT._nan_to_none(list(v)) for f, v in res["cond"][c].items()}
                    for c in conds
                },
            }

        settings_out[setting] = {
            "labels": labels,
            "p_inoc_trigger_idx": p_idx,
            "conds": conds,
            "n_rollouts": n_rollouts,
            "gate_r": gate_r,
            "gate_p": {c: means[setting][c]["gate_p"] for c in conds},
            "mu_n_rows": {c: means[setting][c]["n_rows"] for c in conds},
            "pre": _pack(pre_res),
            "post_rowgrain": _pack(post_res["post_rowgrain"]),
            "post_tokengrain": _pack(post_res["post_tokengrain"]),
        }
        print(
            f"[rescore] {setting}: Gate R PASS (max diffs {gate_r['max_abs_diff']}); "
            f"post-norm arms computed for {len(conds)} conds",
            flush=True,
        )

    out = {
        "issue": ISSUE,
        "label": LABEL,
        "fingerprint": fp,
        "settings_selected": selected,
        "settings_deselected": deselected,  # explicit --settings descope only; never implicit
        "layer": layer,
        "norm": {"sha256_fp32le": norm["sha"], "rms_norm_eps": norm["eps"], "ident": norm["ident"]},
        "map_arm_decision": (
            "option (a): pre-norm context states ride the pinned pre-norm map; the map's "
            "PRE-norm output (persisted vhat) is post-normed — the model's own read-out "
            "convention. No post-norm input enters the pre-norm-trained map."
        ),
        "grain_note": (
            "post_rowgrain compares RMSNorm(stored row states) to mean-of-RMSNorm(row "
            "token-mean states) (PRIMARY); post_tokengrain swaps the answer reference to "
            "the token-level mean-of-norm (sensitivity). Stored ceiling rows are token-mean "
            "states, so their within-row token-level mean-of-norm is unrecoverable offline."
        ),
        "centered_note": "centered companions subtract the per-question mean across triggers "
        "AFTER the RMSNorm transform (the #2379 convention applied to the normed grids)",
        "settings": settings_out,
        "metadata": _metadata("rescore"),
    }
    _atomic_write_json(out_path, out)
    print(f"[rescore] wrote {out_path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase: stats (P3 — pooled Spearman + paired trigger bootstrap, pre vs post)
# ---------------------------------------------------------------------------
def phase_stats(args, cfg: dict) -> dict:
    import issue2474_fit as FIT
    import numpy as np
    from issue2379_analysis import _rank_lastaxis

    print("[phase=stats] start", flush=True)
    scores_path = cfg["out_dir"] / "postnorm_scores.json"
    out_path = cfg["out_dir"] / "postnorm_stats.json"
    if not scores_path.is_file():
        raise RuntimeError(f"{scores_path} missing — run --phase rescore first")
    scores = json.loads(scores_path.read_text())
    if cfg["rates_path"] is not None:
        rates_ident: dict = _file_ident(cfg["rates_path"])
    else:
        import issue2474_free_gate as fg

        rates_ident = {"pinned_parent_sha": fg.PARENT_SHA}
    fp = {
        "phase": "stats",
        "recipe": RECIPE_TAG,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "scores": _file_ident(scores_path),
        "prefit_stats": _file_ident(cfg["prefit_stats_path"]),
        "rates": rates_ident,
        "v": 2,
    }
    if out_path.is_file() and not args.force:
        cached = json.loads(out_path.read_text())
        if cached.get("fingerprint") == fp:
            print("[stats] completion fingerprint match — skip (resumed)", flush=True)
            cached["resumed"] = True
            return cached
    rates_level = FIT._load_rates({"rates_path": cfg["rates_path"]}, "level")
    parent_stats = json.loads(Path(cfg["prefit_stats_path"]).read_text())
    valid_floor = min(100, max(4, args.n_boot // 2))
    settings_out: dict = {}
    for setting, block in scores["settings"].items():
        labels = block["labels"]
        p_idx = int(block["p_inoc_trigger_idx"])
        conds = list(block["conds"])
        dv_labels = set(rates_level[setting]["base"].keys())
        if set(labels) != dv_labels:
            raise RuntimeError(f"{setting}: trigger-label set mismatch capture vs DV")

        def _vec(d: dict) -> np.ndarray:
            return np.array([float(d[lab]) for lab in labels], dtype=np.float64)

        prop = _vec(rates_level[setting]["base"])
        lvl = {c: _vec(rates_level[setting][c]) for c in conds}
        chg = {c: lvl[c] - prop for c in conds}
        # (grain, fam) -> {cond: (n_t,)}
        arm: dict = {}
        for grain in GRAIN_KEYS:
            packed = block[grain]
            for fam in ALL_FAMS:
                per_cond = {}
                for c in conds:
                    src = packed["cond"][c].get(fam)
                    if src is None:
                        src = packed["shared"][fam]
                    per_cond[c] = np.array(
                        [np.nan if v is None else float(v) for v in src], dtype=np.float64
                    )
                arm[(grain, fam)] = per_cond
        variants_out: dict = {}
        for variant in ("full", "loo"):
            sel = [i for i in range(len(labels)) if not (variant == "loo" and i == p_idx)]
            n_sel = len(sel)
            idx = FIT._boot_indices(n_sel, args.n_boot, args.boot_seed)
            arm_keys = sorted(arm.keys())
            stack_rows = [arm[key][c][sel] for key in arm_keys for c in conds]
            arm_stack = np.stack(stack_rows)
            if not np.isfinite(arm_stack).all():
                raise RuntimeError(f"{setting}/{variant}: NaN in arm rows entering the stats")
            correlates = np.vstack(
                [
                    np.stack(
                        [prop[sel]] + [lvl[c][sel] for c in conds] + [chg[c][sel] for c in conds]
                    ),
                    arm_stack,
                ]
            )
            valid = FIT._degenerate_mask(correlates, idx)
            n_valid = int(valid.sum())
            if n_valid < valid_floor:
                raise RuntimeError(
                    f"{setting}/{variant}: only {n_valid}/{args.n_boot} valid bootstrap draws "
                    f"(floor {valid_floor})"
                )
            dv_ranked = {("level", c): _rank_lastaxis(lvl[c][sel][idx]) for c in conds}
            dv_ranked.update({("change", c): _rank_lastaxis(chg[c][sel][idx]) for c in conds})
            draws: dict = {}
            for c in conds:
                mat = np.stack([arm[key][c][sel] for key in arm_keys])
                for dv in ("level", "change"):
                    rho = FIT._draw_spearman(mat, dv_ranked[(dv, c)], idx)
                    for ki, key in enumerate(arm_keys):
                        draws[(key, c, dv)] = rho[ki]
            fams_out: dict = {}
            for fam in ALL_FAMS:
                fam_out: dict = {}
                for dv in ("level", "change"):
                    grain_blocks: dict = {}
                    pooled_draws_by_grain: dict = {}
                    for grain in GRAIN_KEYS:
                        per_cond = {}
                        pooled_pts = []
                        cond_draws = []
                        for c in conds:
                            pt = FIT._point_corr(
                                arm[(grain, fam)][c][sel],
                                (lvl if dv == "level" else chg)[c][sel],
                                spearman=True,
                            )
                            d = draws[((grain, fam), c, dv)]
                            per_cond[c] = {"rho": pt, "ci95": FIT._ci95(d[valid])}
                            pooled_pts.append(pt)
                            cond_draws.append(d)
                        pooled_draws = np.stack(cond_draws).mean(axis=0)[valid]
                        pooled_draws_by_grain[grain] = pooled_draws
                        grain_blocks[grain] = {
                            "pooled_rho": float(np.mean(pooled_pts)),
                            "pooled_ci95": FIT._ci95(pooled_draws),
                            "per_condition": per_cond,
                        }
                    for grain in ("post_rowgrain", "post_tokengrain"):
                        delta = pooled_draws_by_grain[grain] - pooled_draws_by_grain["pre"]
                        grain_blocks[f"delta_{grain.split('_', 1)[1]}"] = {
                            "pooled_delta": float(
                                grain_blocks[grain]["pooled_rho"]
                                - grain_blocks["pre"]["pooled_rho"]
                            ),
                            "ci95": FIT._ci95(delta),
                        }
                    fam_out[dv] = grain_blocks
                fams_out[fam] = fam_out
            # Pre-norm recompute check vs the parent stats at this layer.
            recompute: dict = {}
            try:
                pblock = parent_stats["settings"][setting]["variants"][variant]["families"]
                for fam in ALL_FAMS:
                    ent = pblock[fam]["pooled"]["level"]
                    parent_rho = (
                        ent["pinned"]["rho"]
                        if ent.get("pinned", {}).get("layer") == cfg["layer"]
                        else ent["rho_by_layer"][cfg["layer"]]
                    )
                    mine = fams_out[fam]["level"]["pre"]["pooled_rho"]
                    d = abs(mine - parent_rho)
                    tol_fail = 0.05 if fam.startswith("ans_") else 1e-6
                    if d > tol_fail:
                        raise RuntimeError(
                            f"{setting}/{variant}/{fam}: recomputed pre pooled rho {mine:.4f} "
                            f"!= parent {parent_rho:.4f} (|diff|={d:.4f} > {tol_fail})"
                        )
                    if d > 0.02:
                        logger.warning(
                            "%s/%s/%s: pre pooled rho differs from parent by %.4f "
                            "(fp16-vhat rank flips)",
                            setting,
                            variant,
                            fam,
                            d,
                        )
                    recompute[fam] = {"parent_rho": parent_rho, "abs_diff": d}
            except KeyError as exc:
                raise RuntimeError(
                    f"{setting}/{variant}: parent prefit_stats missing expected key {exc}"
                ) from exc
            variants_out[variant] = {
                "n_triggers": n_sel,
                "n_valid_draws": n_valid,
                "families": fams_out,
                "pre_recompute_check": recompute,
            }
        settings_out[setting] = {
            "conds": conds,
            "labels": labels,
            "p_inoc_trigger_idx": p_idx,
            "variants": variants_out,
        }
        print(f"[stats] {setting}: done ({len(ALL_FAMS)} fams x 2 variants)", flush=True)
    out = {
        "issue": ISSUE,
        "label": LABEL,
        "fingerprint": fp,
        "layer": cfg["layer"],
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "boot_note": (
            "paired trigger bootstrap: one shared (n_boot, n) integer index multiset per "
            "(setting, variant) across every arm, grain and DV (the parent issue2474_fit "
            "convention, seed inherited); pooled = mean over conditions; delta draws are "
            "post - pre under the SAME draws (paired)."
        ),
        "settings": settings_out,
        "metadata": _metadata("stats"),
    }
    _atomic_write_json(out_path, out)
    print(f"[stats] wrote {out_path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase: figs (P4 — one pre-vs-post comparison figure)
# ---------------------------------------------------------------------------
def _err_offsets(point: float, ci: list[float]):
    """Non-negative (lo, hi) errorbar offsets from a point + CI bounds."""
    lo = max(0.0, point - float(ci[0]))
    hi = max(0.0, float(ci[1]) - point)
    return lo, hi


def build_comparison_figure(
    stats: dict, fig_dir: Path, *, setting: str, variant: str = "full", dv: str = "level"
) -> dict:
    """Grouped pre-vs-post pooled-rho bars per training-reference arm, with
    per-condition (per-language) points and 95% bootstrap whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    palette = paper_palette(4)
    block = stats["settings"][setting]["variants"][variant]["families"]
    conds = stats["settings"][setting]["conds"]
    fams = [f for f in TRAINREF_FAMS]
    width = 0.38
    xs = np.arange(len(fams), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for bi, (grain, label, color) in enumerate(
        (
            ("pre", "Pre-norm (stored)", palette[0]),
            ("post_rowgrain", "Post-final-RMSNorm", palette[1]),
        )
    ):
        pts = [block[f][dv][grain]["pooled_rho"] for f in fams]
        errs = np.array(
            [
                _err_offsets(block[f][dv][grain]["pooled_rho"], block[f][dv][grain]["pooled_ci95"])
                for f in fams
            ]
        ).T
        pos = xs + (bi - 0.5) * width
        ax.bar(pos, pts, width=width, color=color, label=label, yerr=errs, capsize=3)
        for fi, f in enumerate(fams):
            per = block[f][dv][grain]["per_condition"]
            vals = [per[c]["rho"] for c in conds]
            ax.scatter(
                np.full(len(vals), pos[fi]),
                vals,
                s=14,
                color="black",
                zorder=3,
                alpha=0.75,
            )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY_NAME[f] for f in fams])
    ax.set_ylabel(f"Pooled Spearman rho (level DV, layer {stats['layer']})")
    ax.set_title("Training-reference arms: pre-norm vs post-final-RMSNorm")
    ax.legend(frameon=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, f"postnorm_l27_comparison_{setting}", dir=fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def phase_figs(args, cfg: dict) -> dict:
    """One comparison figure PER SETTING present in the stats (the em negative
    control included — r1 codex concern em-negative-control-figure-omitted)."""
    print("[phase=figs] start", flush=True)
    stats_path = cfg["out_dir"] / "postnorm_stats.json"
    if not stats_path.is_file():
        raise RuntimeError(f"{stats_path} missing — run --phase stats first")
    stats = json.loads(stats_path.read_text())
    primary = cfg["settings"][0]
    if primary not in stats["settings"]:
        raise RuntimeError(f"stats missing primary setting {primary!r}")
    paths = {
        setting: build_comparison_figure(stats, cfg["fig_dir"], setting=setting)
        for setting in stats["settings"]
    }
    print(f"[figs] wrote {sorted(p for m in paths.values() for p in m.values())}", flush=True)
    return {"figure_paths": paths}


# ---------------------------------------------------------------------------
# Phase: smoke (P0 — synthetic tiny end-to-end; tiny CPU model; no canonical writes)
# ---------------------------------------------------------------------------
def _tiny_model_factory(seed: int = 0):
    """2-layer from-config Qwen2 over the REAL tokenizer (the #906 pattern)."""

    def _load():
        import torch
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        torch.manual_seed(seed)
        config = Qwen2Config(
            vocab_size=152064,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(config)
        model.eval()
        return model, tokenizer, "tiny:qwen2-2l-64h"

    return _load


_SMOKE_ROWS = {
    "smoke_condA": [
        ("You always answer politely.", "What is 2 + 2?", "2 + 2 equals 4. Thank you for asking!"),
        ("You always answer politely.", "Name a primary color.", "A primary color is red."),
        ("You always answer politely.", "What day follows Monday?", "Tuesday follows Monday."),
    ],
    "smoke_condB": [
        ("You answer briefly.", "What is 3 + 3?", "6."),
        ("You answer briefly.", "Name a fruit.", "Apple."),
        ("You answer briefly.", "What color is the sky?", "Blue, usually."),
    ],
}


def _gen_smoke_tree(root: Path) -> None:
    """Synthetic input tree: parent-path mu bundles (REAL two-helper capture on
    the tiny model), synthetic grid/ceiling/vhat/passb, self-consistent parent
    score/stat targets, synthetic rates."""
    import issue2474_fit as FIT
    import issue779_collect as I779
    import numpy as np
    import torch

    rng = np.random.default_rng(20260823)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "capture" / "predictor_captures").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "figs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for cond, rows in _SMOKE_ROWS.items():
        with (root / "train" / f"{cond}.jsonl").open("w", encoding="utf-8") as f:
            for sys_c, user_c, gold in rows:
                f.write(
                    json.dumps(
                        {
                            "prompt": [
                                {"role": "system", "content": sys_c},
                                {"role": "user", "content": user_c},
                            ],
                            "completion": [{"role": "assistant", "content": gold}],
                        }
                    )
                    + "\n"
                )
    # Parent-path stored mu bundles (two-helper capture, both layers).
    model, tokenizer, _ = _tiny_model_factory()()
    layers = [0, 1]
    for cond, rows in _SMOKE_ROWS.items():
        mu_c = mu_a = None
        for sys_c, user_c, gold in rows:
            messages = [
                {"role": "system", "content": sys_c},
                {"role": "user", "content": user_c},
            ]
            v_c = I779.capture_context_vector(model, tokenizer, messages, layers)["last"].to(
                torch.float32
            )
            av = I779.capture_answer_vector(model, tokenizer, messages, gold, layers, {})
            if av is None:
                raise RuntimeError("smoke mu generation: empty gold answer")
            v_a = av["v_x"].to(torch.float32)
            mu_c = v_c if mu_c is None else mu_c + v_c
            mu_a = v_a if mu_a is None else mu_a + v_a
        n = len(rows)
        bdir = root / "capture" / "predictor_captures" / f"base_mu_{cond}"
        bdir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mu_train": (mu_c / n).to(torch.float16),
                "mu_a_train": (mu_a / n).to(torch.float16),
                "n_c": n,
                "n_a": n,
                "model": f"base_mu_{cond}",
                "setting": "smoke",
                "model_ident": "tiny:qwen2-2l-64h",
            },
            bdir / "mu.pt",
        )
    # Synthetic grid / ceiling / vhat / passb (fp16, 2 layers, H=64).
    n_t, n_q, n_l, hdim, n_roll = 6, 3, 2, 64, 2
    labels = ["smoke_p"] + [f"smoke_t{i}" for i in range(1, n_t)]
    grid_meta = [
        {"trigger_idx": t, "trigger_label": labels[t], "q_sim_idx": q}
        for t in range(n_t)
        for q in range(n_q)
    ]
    v_c_grid = rng.normal(0.0, 1.0, size=(n_t * n_q, n_l, hdim)) + 0.3
    gdir = root / "capture" / "predictor_captures" / "base_smoke"
    gdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"v_c": torch.from_numpy(v_c_grid.astype(np.float16)), "row_meta": grid_meta},
        gdir / "grid.pt",
    )
    ceil_meta = []
    va_rows = []
    cell = 0
    for t in range(n_t):
        for q in range(n_q):
            for r in range(n_roll):
                if t == 1 and q == 0 and r == 1:
                    continue  # one dropped rollout slot exercises the NaN path
                ceil_meta.append(
                    {
                        "cell_idx": cell,
                        "trigger_idx": t,
                        "trigger_label": labels[t],
                        "q_sim_idx": q,
                        "rollout_idx": r,
                    }
                )
                va_rows.append(rng.normal(0.0, 1.0, size=(n_l, hdim)) + 0.2)
            cell += 1
    torch.save(
        {
            "v_a": torch.from_numpy(np.stack(va_rows).astype(np.float16)),
            "row_meta": ceil_meta,
            "drop_stats": {
                "n_slots": n_t * n_q * n_roll,
                "n_empty_after_retries": 1,
                "n_capture_dropped": 0,
            },
        },
        gdir / "ceiling.pt",
    )
    v_hat = (v_c_grid[:, 1, :] * 0.8 + rng.normal(0, 0.2, size=(n_t * n_q, hdim))).astype(
        np.float16
    )
    torch.save(
        {"v_hat_mapB": torch.from_numpy(v_hat), "setting": "smoke", "layer": 1},
        root / "vhat.pt",
    )
    n_pb = 40
    torch.save(
        {
            "cx_last": torch.from_numpy(
                rng.normal(0, 1, size=(n_pb, n_l, hdim)).astype(np.float16)
            ),
            "v_x": torch.from_numpy(rng.normal(0, 1, size=(n_pb, n_l, hdim)).astype(np.float16)),
            "layers": list(range(n_l)),
            "source": "smoke",
        },
        root / "passb.pt",
    )
    # Synthetic rates (level + cont), numeric variety for rank stability.
    conds = list(_SMOKE_ROWS.keys())
    rates = {"level": {"smoke": {}}, "cont": {"smoke": {}}}
    rates["level"]["smoke"]["base"] = {
        lab: float(v) for lab, v in zip(labels, rng.uniform(0.05, 0.2, n_t))
    }
    for c in conds:
        rates["level"]["smoke"][c] = {
            lab: float(v) for lab, v in zip(labels, rng.uniform(0.1, 0.9, n_t))
        }
        rates["cont"]["smoke"][c] = {
            lab: float(v) for lab, v in zip(labels, rng.uniform(0.1, 0.9, n_t))
        }
    (root / "rates_synth.json").write_text(json.dumps(rates))
    # Self-consistent parent targets: run the SAME scoring core on the synthetic
    # inputs (stored-mu reference) and persist in the parent schema.
    import issue2379_mapfit as mf

    row_of = -np.ones((n_t, n_q), dtype=int)
    for i, r in enumerate(grid_meta):
        row_of[r["trigger_idx"], r["q_sim_idx"]] = i
    c_t = np.array([r["trigger_idx"] for r in ceil_meta])
    c_q = np.array([r["q_sim_idx"] for r in ceil_meta])
    c_ri = np.array([r["rollout_idx"] for r in ceil_meta])
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    conditions = {}
    fam_curves: dict = {}
    for cond in conds:
        mu = mf._torch_load_constrained(
            root / "capture" / "predictor_captures" / f"base_mu_{cond}" / "mu.pt"
        )
        fam_curves[cond] = {"mu": mu}
    passb = mf._torch_load_constrained(root / "passb.pt")
    for layer in range(n_l):
        x = np.asarray(passb["cx_last"][:, layer, :], dtype=np.float64)
        y = np.asarray(passb["v_x"][:, layer, :], dtype=np.float64)
        tr_idx, _ = mf._split_indices(n_pb)
        ib = identity_bias_predict(x[tr_idx], y[tr_idx], np.zeros((1, hdim)))[0]
        v_c_l = np.asarray(v_c_grid.astype(np.float16)[:, layer, :], dtype=np.float64)
        vh_l = (
            np.asarray(v_hat, dtype=np.float64)
            if layer == 1
            else np.zeros_like(v_c_l) + rng.normal(0, 1, size=v_c_l.shape)
        )
        va_l = np.asarray(np.stack(va_rows).astype(np.float16)[:, layer, :], dtype=np.float64)
        mu_tr = np.stack(
            [np.asarray(fam_curves[c]["mu"]["mu_train"][layer], dtype=np.float64) for c in conds]
        )
        mu_a = np.stack(
            [np.asarray(fam_curves[c]["mu"]["mu_a_train"][layer], dtype=np.float64) for c in conds]
        )
        res = FIT._score_layer_batched(
            v_c_l, vh_l, v_c_l + ib, row_of, 0, va_l, c_t, c_q, c_ri, n_roll, mu_tr, mu_a, conds
        )
        for ci_, cond in enumerate(conds):
            conditions.setdefault(
                cond,
                {
                    "trigger_labels": labels,
                    "p_inoc_trigger_idx": 0,
                    "n_layers": n_l,
                    "families_layered": {f: [[None] * n_t for _ in range(n_l)] for f in ALL_FAMS},
                },
            )
            for fam in ALL_FAMS:
                vals = res["cond"][cond].get(fam)
                if vals is None:
                    vals = res["shared"][fam]
                conditions[cond]["families_layered"][fam][layer] = FIT._nan_to_none(list(vals))
    (root / "prefit_scores.json").write_text(json.dumps({"conditions": conditions}))
    # Self-consistent parent STATS pinned targets at layer 1 (level DV).
    parent_families: dict = {"full": {}, "loo": {}}
    lvl = {c: np.array([rates["level"]["smoke"][c][lab] for lab in labels]) for c in conds}
    for variant in ("full", "loo"):
        sel = [i for i in range(n_t) if not (variant == "loo" and i == 0)]
        fams: dict = {}
        for fam in ALL_FAMS:
            pts = []
            for c in conds:
                v = np.array(
                    [
                        np.nan if x is None else float(x)
                        for x in conditions[c]["families_layered"][fam][1]
                    ]
                )
                pts.append(FIT._point_corr(v[sel], lvl[c][sel], spearman=True))
            fams[fam] = {
                "pooled": {
                    "level": {
                        "rho_by_layer": [None, float(np.mean(pts))],
                        "pinned": {"layer": 1, "rho": float(np.mean(pts))},
                    }
                }
            }
        parent_families[variant] = fams
    (root / "prefit_stats.json").write_text(
        json.dumps(
            {
                "settings": {
                    "smoke": {
                        "variants": {v: {"families": parent_families[v]} for v in ("full", "loo")}
                    }
                }
            }
        )
    )
    print(f"[smoke] synthetic tree generated under {root}", flush=True)


def _smoke_ns(args, root: Path, **over) -> argparse.Namespace:
    base = vars(args).copy()
    base.update(
        {
            "synthetic_root": str(root),
            "n_boot": 64,
            "boot_seed": args.boot_seed,
            "ckpt_every": 2,
            "spot_rows": 2,
            "force": False,
            "skip_upload": True,
            "smoke_interrupt_after": 0,
        }
    )
    base.update(over)
    return argparse.Namespace(**base)


def _safe_smoke_root(raw: str) -> Path:
    """Resolve + containment-check the smoke scratch root BEFORE any rmtree.

    The root must live STRICTLY INSIDE the temp dir (tempfile.gettempdir() or
    /tmp) — a typo'd --smoke-dir (repo root, $HOME, /) must never reach a
    recursive delete (r1 codex blocker smoke-dir-arbitrary-recursive-delete)."""
    import tempfile

    root = Path(raw).expanduser().resolve()
    scratch_parents = {Path(tempfile.gettempdir()).resolve(), Path("/tmp")}
    if not any(parent in root.parents for parent in scratch_parents):
        raise RuntimeError(
            f"--smoke-dir {raw!r} resolves to {root} — refusing recursive delete: the smoke "
            f"scratch root must live STRICTLY INSIDE the temp dir "
            f"({sorted(str(p) for p in scratch_parents)})"
        )
    return root


def phase_smoke(args) -> None:
    import shutil

    root = _safe_smoke_root(args.smoke_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    print(f"[phase=smoke] tree at {root}", flush=True)
    _gen_smoke_tree(root)
    loader = _tiny_model_factory()

    # Leg 1: trainref-gpu with a mid-condition interrupt, then full resume.
    ns = _smoke_ns(args, root, smoke_interrupt_after=1)
    cfg = _cfg_from_args(ns)
    cfg["model_loader"] = loader
    try:
        phase_trainref_gpu(ns, cfg)
        raise RuntimeError("smoke FAIL: interrupt hook did not fire")
    except _SmokeInterrupt:
        print("[smoke] interrupt leg: partial checkpoint written", flush=True)
    partials = list(cfg["means_dir"].glob("*.partial.npz"))
    if not partials:
        raise RuntimeError("smoke FAIL: no partial checkpoint after interrupt")
    ns = _smoke_ns(args, root)
    cfg = _cfg_from_args(ns)
    cfg["model_loader"] = loader
    res1 = phase_trainref_gpu(ns, cfg)
    if not res1["resumed"] or res1["resumed"][0]["resumed_at_row"] < 1:
        raise RuntimeError(f"smoke FAIL: mid-condition resume not exercised ({res1['resumed']})")
    # Spot evidence must survive the resume (persisted in the partial npz).
    for cond, blk in res1["conds"].items():
        if len(blk["spot"]) != ns.spot_rows:
            raise RuntimeError(
                f"smoke FAIL: {cond} spot records {len(blk['spot'])} != {ns.spot_rows} "
                f"(resume must preserve spot-gate evidence)"
            )
    # Leg 2: completed-condition skip on replay.
    res2 = phase_trainref_gpu(ns, cfg)
    if sorted(res2["skipped"]) != sorted(_SMOKE_ROWS.keys()):
        raise RuntimeError(f"smoke FAIL: completed-cond skip not exercised ({res2['skipped']})")

    # Leg 3: offline chain (rescore -> stats -> figs), then completion replay.
    r1 = phase_rescore(ns, cfg)
    # Post != pre for the CEILING arms — pins the nva threading (r1 Claude
    # Critical: pre-fix, the post calls reused the pre-norm va_l, making
    # ceiling_sameq post byte-identical to pre and ceiling_trainref a
    # mixed-convention read).
    sm = r1["settings"]["smoke"]
    for fam in ("ceiling_sameq", "ceiling_sameq_centered"):
        if sm["pre"]["shared"][fam] == sm["post_rowgrain"]["shared"][fam]:
            raise RuntimeError(f"smoke FAIL: post {fam} == pre — ceiling rows not post-normed")
    for c in sm["conds"]:
        if (
            sm["pre"]["cond"][c]["ceiling_trainref"]
            == sm["post_rowgrain"]["cond"][c]["ceiling_trainref"]
        ):
            raise RuntimeError(f"smoke FAIL: post ceiling_trainref == pre for {c}")
    s1 = phase_stats(ns, cfg)
    f1 = phase_figs(ns, cfg)
    r2 = phase_rescore(ns, cfg)
    if not r2.get("resumed"):
        raise RuntimeError("smoke FAIL: rescore completion replay did not skip")
    s2 = phase_stats(ns, cfg)
    if not s2.get("resumed"):
        raise RuntimeError("smoke FAIL: stats completion replay did not skip")
    for fam in TRAINREF_FAMS:
        blk = s1["settings"]["smoke"]["variants"]["full"]["families"][fam]["level"]
        for key in ("pre", "post_rowgrain", "post_tokengrain", "delta_rowgrain"):
            if key not in blk:
                raise RuntimeError(f"smoke FAIL: stats block missing {fam}/{key}")
    fig_png = [p for p in f1["figure_paths"]["smoke"].values() if p.endswith(".png")]
    if not fig_png or Path(fig_png[0]).stat().st_size < 10_000:
        raise RuntimeError("smoke FAIL: comparison figure missing/empty")
    # Gate coverage asserts (the gates RAN, not just returned).
    gr = r1["settings"]["smoke"]["gate_r"]
    if gr["verdict"] != "PASS":
        raise RuntimeError("smoke FAIL: Gate R did not pass on self-consistent targets")

    # Leg 4: REAL ranged norm-weight fetch probe (network; production branch).
    if args.skip_net_probe:
        print("[smoke] net probe SKIPPED (--skip-net-probe)", flush=True)
        net = {"skipped": True}
    else:
        rec = fetch_norm_weight_ranged(MODEL_ID, None)
        if rec["hidden_size"] != EXPECTED_HIDDEN or rec["rms_norm_eps"] != RMS_EPS_EXPECTED:
            raise RuntimeError(f"smoke FAIL: norm fetch unexpected {rec['hidden_size']}")
        net = {
            "sha256_fp32le": rec["sha256_fp32le"],
            "model_revision": rec["model_revision"],
        }
        print(
            f"[smoke] net probe OK: {NORM_WEIGHT_NAME} sha={rec['sha256_fp32le'][:16]} "
            f"rev={rec['model_revision'][:12]}",
            flush=True,
        )
    _write_sentinel(ns, cfg, phase="smoke", note_payload={"net_probe": net})
    print("[smoke] PASS — end-to-end outputs under", root / "out", flush=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
# Arm registry (string-constant keys — the smoke-architecture per-arm rows are
# recomputed from this dict).
PHASES = {
    "smoke": "P0: synthetic tiny end-to-end (tiny CPU model; real norm-fetch probe)",
    "trainref-gpu": "P1 (pod): per-condition post-norm training-mix means at layer 27",
    "rescore": "P2 (VM): offline post-norm re-read + Gate R pre-norm reproduction",
    "stats": "P3 (VM): pooled Spearman + paired trigger bootstrap, pre vs post + delta",
    "figs": "P4 (VM): pre-vs-post comparison figure",
    "all": "rescore -> stats -> figs (offline chain)",
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument(
        "--import-check", action="store_true", help="argcheck + call-arity bind, then exit 0"
    )
    ap.add_argument("--conditions", default="all", help="comma list or 'all' (8 conditions)")
    ap.add_argument(
        "--settings",
        default="all",
        help="rescore setting selection: comma list or 'all'; a narrower selection is an "
        "EXPLICIT recorded descope (rides the fingerprint + payload) — absence/staging "
        "failures never descope silently",
    )
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "issue_2474"))
    ap.add_argument(
        "--out-dir", default=str(REPO_ROOT / "eval_results" / "issue_2474" / "postnorm_l27")
    )
    ap.add_argument("--figures-out", default=str(REPO_ROOT / "figures" / "issue_2474"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--model-revision", default="", help="pin; empty = resolve main once")
    ap.add_argument(
        "--allow-model-revision-drift",
        action="store_true",
        help="accept completed-cond outputs whose fingerprint differs ONLY in model revision",
    )
    ap.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--spot-rows", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED_DEFAULT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-upload", action="store_true", help="skip the HF upload leg (GPU phase)")
    ap.add_argument("--smoke-dir", default="/tmp/issue2474-postnorm-smoke")
    ap.add_argument("--synthetic-root", default="", help="internal: smoke tree root")
    ap.add_argument("--skip-net-probe", action="store_true", help="smoke: skip the live HF probe")
    ap.add_argument(
        "--smoke-interrupt-after",
        type=int,
        default=0,
        help="smoke test hook: simulated death after N rows (0 = off)",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.phase == "smoke":
        phase_smoke(args)
    else:
        cfg = _cfg_from_args(args)
        if args.phase == "trainref-gpu":
            phase_trainref_gpu(args, cfg)
        elif args.phase == "rescore":
            phase_rescore(args, cfg)
        elif args.phase == "stats":
            phase_stats(args, cfg)
        elif args.phase == "figs":
            phase_figs(args, cfg)
        elif args.phase == "all":
            phase_rescore(args, cfg)
            phase_stats(args, cfg)
            phase_figs(args, cfg)
        else:  # pragma: no cover — argparse choices guard
            raise RuntimeError(f"unknown phase {args.phase}")
    sys.stdout.flush()
    sys.stderr.flush()
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
