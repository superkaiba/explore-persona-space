#!/usr/bin/env python3
"""Issue #1738 follow-up `sae-arm` — SAE-feature-space history->answer maps (plan v8).

ONE-variable delta vs the completed #1738 dense run: representation space
dense -> SAE features (the #1482 recipe class: BatchTopK suite
``andyrdt/saes-qwen2.5-7b-instruct``, resid_post layer 19, k=64 primary).

Phases (same entrypoint smoke and production — PASS_UNIFIED):

``--phase capture`` (S1, 8-way GPU-sharded via the launcher ``--sae-arm`` mode):
  streams the PARENT capture chunks (``issue1738_multiturn/capture/*.pt`` —
  each carries prompts/response text + stored px_last/cx_last/v_x), re-renders
  each row under the parent's own ``capture_answer_vector`` convention (full-
  template re-tokenization, answer span incl. end-of-turn tail), runs ONE
  batched bf16 teacher-forced forward per context capturing layer-19 answer-
  span states, gates every row on cos(v_x_hat, stored v_x) >= 0.999 (the
  render/span identity gate), SAE-encodes the inlier answer tokens
  (reference ``token_inlier_mask``) + the STORED px_last/cx_last states
  (stated scope refinement, plan §13 — exact input parity with the dense
  arms), pools mean/max/frac (``pool_answer_features``), and writes one
  sparse feature chunk per parent chunk (same basename) with
  upload->sha-verify->purge to ``{--sae-hf-prefix}/capture/``.

  ``--pilot-rows N --pilot-only`` (shard 0, launcher-run FOREGROUND before the
  fleet detaches): accumulates >=100k inlier answer+px/cx L19 token states and
  runs the reference-parity ``fve_l0`` at k=64 (primary) + k=128 (robustness)
  — G-S0 (FVE >= 0.75 AND L0 within 2x of 60; FAIL => designed rc 27, the
  round's kill criterion) — plus G-S1 (v_x identity median/violations, 32-row
  batched-vs-per-row parity probe, measured rate vs the 60-min per-shard fence
  => designed rc 26) -> ``sae_pilot_meta.json`` (local + Hub).

``--phase fits`` (S2, 1x A100-80): streams the sae chunks -> CSR (the
  #1482 ``issue1482_shuffle_null`` conventions: TRAIN-row activity counts,
  16,384-answer / 8,192-input restriction at the 1%-of-fit-rows floor, ONE
  shared answer set across arms), pinned split (sha-assert + parent
  ``split_shas`` cross-assert), shared-Gram fp64 ridge (parent 23-value
  lambda grid selected on val 396; cuda eigh -> CPU LAPACK fallback) for the
  4 primary cells (sae_prefix / sae_context / dense_px_feat / dense_cx_feat
  -> answer-features(mean)) + max/frac pooling twins of the SAE arms,
  K=20 shuffled-pairing floor per SAE arm (seeds 1738000..1738019, lambda
  pinned, scored on true holdout pairs), per-feature holdout R^2 + covariates
  (activity, within-answer consistency) + per-feature null bands +
  DeltaR^2(context - prefix) with the "carried" calibration, identity+bias on
  the shared feature-id subsets (dense cells stated INAPPLICABLE — 3,584 vs
  16,384), kNN retrieval for all four maps, lmsys_transfer for the SAE arms.
  First-cell timing fence >= 2x (G-S2, designed rc 24).

``--smoke``: tiny-real CPU e2e of BOTH phases through the SAME run functions
  (production entrypoints; from-config tiny same-arch Qwen2 over the real
  vocab + a from-config tiny BatchTopK SAE; Hub boundary signature-bound) +
  degenerate probes for every data-dependent gate branch.

Refusal-safety: chunk text fields are never printed/logged (digest-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847: thread caps + credentials BEFORE numpy/torch import

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as PF  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402
import issue1482_sae as SAEMOD  # noqa: E402
import issue1482_shuffle_null as SN  # noqa: E402
import issue1738_multiturn_fits as MTF  # noqa: E402
import issue1738_multiturn_generate_capture as GG  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1738_sae_arm")

# ── constants (plan §11 — Sources recorded there) ──────────────────────────────────
LAYER = 19  # the parent headline layer; suite hook point (Source: #1482)
K_PRIMARY, K_ROBUST = 64, 128  # trainer_1 / trainer_2 (Source: #1482)
FVE_MIN = 0.75  # G-S0 bar: ~0.93x published 0.8057 (Source: plan §7)
L0_PUBLISHED = 60.0
L0_MULT = 2.0  # G-S0: L0 within 2x of published (wildness = token-pool signature)
FVE_TOKEN_FLOOR = 100_000  # G-S0 pool floor (plan §4.1 S1.5)
VX_COS_ROW_MIN = 0.999  # per-row identity drop bar (plan §4.1 S1.2)
VX_COS_MEDIAN_MIN = 0.9995  # G-S1 median bar (span means; #779 precedent 0.999748)
VIOLATION_RATE_MAX = 0.005  # parent's own violation budget
PARITY_MIN_COS = 0.999  # G-S1 32-row batched-vs-per-row bar (span means)
PARITY_ROWS = 32
PILOT_ROWS_DEFAULT = 2_000  # G-S1 pilot slice (shard 0)
SHARD_FENCE_MIN = 60.0  # projected per-shard wall fence (minutes)
RC_SAE_FIT_FENCE = 24  # G-S2 designed-halt rc (parent G2 convention)
RC_SAE_FENCE = 26  # G-S1 rate-fence designed-halt rc (report written first)
RC_SAE_FVE = 27  # G-S0 fitness-kill designed rc (the round's kill criterion)
SAE_BATCH_DEFAULT = 32  # right-padded forward batch (ungrounded — pilot-gated)
PILOT_META_NAME = "sae_pilot_meta.json"

MAX_FEATURES_OUT = 16_384  # shared answer-side set (#1482 recipe at this run's n)
MAX_FEATURES_IN = 8_192  # per-arm input-side sets
K_DRAWS = 20
SHUFFLE_SEED_BASE = 1_738_000  # seeds 1738000..1738019 (plan §10)
LAMBDAS = MTF.LAMBDAS  # parent 23-value grid logspace(-3, 8, 23), verbatim
N_BOOT = MTF.N_BOOT  # 10,000
BOOT_SEED = MTF.BOOT_SEED  # 1738
H_DIM = C.EXPECTED_HIDDEN  # 3584
POOLINGS = ("mean", "max", "frac")
SAE_CELLS = (  # (cell name, X source, Y pooling); mean = primary (plan §5)
    ("sae_prefix", "px_feat", "mean"),
    ("sae_context", "cx_feat", "mean"),
    ("dense_px_feat", "px_dense", "mean"),
    ("dense_cx_feat", "cx_dense", "mean"),
    ("sae_prefix_max", "px_feat", "max"),
    ("sae_context_max", "cx_feat", "max"),
    ("sae_prefix_frac", "px_feat", "frac"),
    ("sae_context_frac", "cx_feat", "frac"),
)
ARM_X = {"sae_prefix": "px", "sae_context": "cx"}  # SAE arms (per-feature reads)

DEFAULT_OUT_EVAL = PROJECT_ROOT / "eval_results" / "issue_1738" / "sae_arm"
DEFAULT_OUT_LOCAL = PROJECT_ROOT / "data" / "issue_1738" / "mt100k" / "sae_arm"
DEFAULT_PARENT_FITS = (
    PROJECT_ROOT / "eval_results" / "issue_1738" / "fits" / ("multiturn_100k_fits.json")
)


def _require_sae_prefix(args) -> str:
    """Fail-loud upload-prefix resolution (the #1005 clobber class: default=None
    + raise — never a hardcoded fallback a reusing child silently inherits)."""
    if not args.sae_hf_prefix:
        raise RuntimeError(
            "--sae-hf-prefix is required for any Hub read/write of the sae-arm store "
            "(pass issue1738_multiturn/sae_arm; no default by design — #1005 class)"
        )
    return args.sae_hf_prefix


# ── S1 capture: parent render convention + batched span capture ────────────────────


def _render_row(tok, messages: list[dict], response: str) -> tuple[torch.Tensor, int]:
    """The parent's own render (``capture_answer_vector`` verbatim): full-template
    re-tokenization of messages + assistant answer; answer span =
    [prompt_len, full_len) incl. the end-of-turn tail. Returns (ids (T,), prompt_len)."""
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_len = int(tok(prompt_text, return_tensors="pt", padding=False)["input_ids"].shape[1])
    full_messages = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    ids = tok(full_text, return_tensors="pt", padding=False)["input_ids"][0]
    assert ids.shape[0] > prompt_len, f"empty answer span (T={ids.shape[0]}, plen={prompt_len})"
    return ids, prompt_len


@torch.no_grad()
def _capture_answer_spans(
    hf, rows_ids: list[torch.Tensor], prompt_lens: list[int], layer: int, batch: int, pad_id: int
) -> list[torch.Tensor]:
    """Length-sorted right-padded batched forwards -> per-row (T_ans, H) fp32
    answer-span L19 states. Right padding + attention mask is causal-safe (real
    positions 0..T_i-1 are unaffected; no position_ids needed)."""
    order = np.argsort([int(x.shape[0]) for x in rows_ids])
    out: list[torch.Tensor | None] = [None] * len(rows_ids)
    for s in range(0, len(order), batch):
        sel = order[s : s + batch]
        lens = [int(rows_ids[i].shape[0]) for i in sel]
        tmax = max(lens)
        ids = torch.full((len(sel), tmax), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sel), tmax), dtype=torch.long)
        for j, i in enumerate(sel):
            ids[j, : lens[j]] = rows_ids[i]
            mask[j, : lens[j]] = 1
        cap = extract_layer_activations(
            hf, ids.to(hf.device), [layer], attention_mask=mask.to(hf.device)
        )
        hs = cap[layer]  # (B, T, H)
        for j, i in enumerate(sel):
            out[i] = hs[j, prompt_lens[i] : lens[j], :].float().cpu()
    assert all(o is not None for o in out)
    return out  # type: ignore[return-value]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _sparse_vec(v: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero (idx int32, val fp16) of a single (F,) feature vector."""
    idx = torch.nonzero(v, as_tuple=False).squeeze(-1)
    return (
        idx.cpu().numpy().astype(np.int32),
        v[idx].float().cpu().numpy().astype(np.float16),
    )


class _FvePool:
    """G-S0 token-pool accumulator (answer inlier tokens + px/cx states, fp32
    CPU, capped so the pilot's pool stays ~2 GB)."""

    def __init__(self, cap: int = 150_000):
        self.cap = cap
        self.parts: list[torch.Tensor] = []
        self.n = 0

    def add(self, h: torch.Tensor) -> None:
        if self.n >= self.cap or h.shape[0] == 0:
            return
        take = min(h.shape[0], self.cap - self.n)
        self.parts.append(h[:take].float().cpu())
        self.n += take

    def tensor(self) -> torch.Tensor:
        return torch.cat(self.parts) if self.parts else torch.zeros(0, 1)


def _process_chunk(hf, tok, sae, bundle: dict, args, fve_pool: dict | None):
    """One parent chunk -> (sae chunk dict, violations list, stats dict).

    Per row: render (parent convention) -> batched span capture -> v_x identity
    gate -> SAE encode inlier answer tokens (pool 3 ways) + stored px/cx states.
    A sub-bar row is recorded (ci + cos) and its feature rows dropped."""
    blayers = list(bundle["layers"])
    li = blayers.index(args.layer)
    n = len(bundle["ci"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows_ids, plens = [], []
    for i in range(n):
        messages = json.loads(bundle["prompts"][i])
        ids, plen = _render_row(tok, messages, bundle["response"][i])
        rows_ids.append(ids)
        plens.append(plen)
    spans = _capture_answer_spans(hf, rows_ids, plens, args.layer, args.sae_batch, pad_id)
    stored_vx = bundle["v_x"][:, li, :].float()
    px = bundle["px_last"][:, li, :].float()
    cx = bundle["cx_last"][:, li, :].float()

    rec: dict[str, list] = {k: [] for k in ("ci", "depth", "corpus", "vx_cos")}
    feat_idx, row_ptr = [], [0]
    vals: dict[str, list] = {p: [] for p in POOLINGS}
    pxi, pxv, pxp = [], [], [0]
    cxi, cxv, cxp = [], [], [0]
    n_ans, n_inl = [], []
    px_keep, cx_keep = [], []
    violations: list[dict] = []
    px_l0s, cx_l0s = [], []
    for i in range(n):
        span = spans[i]
        vx_hat = span.mean(0)
        cos = _cos(vx_hat, stored_vx[i])
        if cos < args.vx_cos_row_min:
            violations.append({"ci": int(bundle["ci"][i]), "cos": round(cos, 6)})
            continue
        inl = SAEMOD.token_inlier_mask(span)
        span_in = span[inl]
        if span_in.shape[0] == 0:
            violations.append(
                {"ci": int(bundle["ci"][i]), "cos": round(cos, 6), "reason": "all_tokens_outlier"}
            )
            continue
        if fve_pool is not None:
            fve_pool["answer_tokens"].add(span_in)
            fve_pool["context_tokens"].add(px[i : i + 1])
            fve_pool["context_tokens"].add(cx[i : i + 1])
        f = sae.encode(span_in.to(sae.device))  # (T_in, F) fp32
        pooled = SAEMOD.pool_answer_features(f)
        spd = SAEMOD.sparsify(pooled)
        feat_idx.append(spd["idx"])
        row_ptr.append(row_ptr[-1] + len(spd["idx"]))
        for p in POOLINGS:
            vals[p].append(spd[p])
        pf = sae.encode(px[i : i + 1].to(sae.device))[0]
        cf = sae.encode(cx[i : i + 1].to(sae.device))[0]
        pi, pv = _sparse_vec(pf)
        ci_, cv = _sparse_vec(cf)
        px_l0s.append(len(pi))
        cx_l0s.append(len(ci_))
        pxi.append(pi)
        pxv.append(pv)
        pxp.append(pxp[-1] + len(pi))
        cxi.append(ci_)
        cxv.append(cv)
        cxp.append(cxp[-1] + len(ci_))
        rec["ci"].append(int(bundle["ci"][i]))
        rec["depth"].append(int(bundle["depth"][i]))
        rec["corpus"].append(bundle["corpus"][i])
        rec["vx_cos"].append(round(cos, 6))
        n_ans.append(int(span.shape[0]))
        n_inl.append(int(span_in.shape[0]))
        px_keep.append(px[i])
        cx_keep.append(cx[i])

    def _cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    chunk = {
        "ci": rec["ci"],
        "depth": rec["depth"],
        "corpus": rec["corpus"],
        "vx_cos": np.asarray(rec["vx_cos"], dtype=np.float32),
        "feat_idx": _cat(feat_idx, np.int32),
        "row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "ans_mean": _cat(vals["mean"], np.float16),
        "ans_max": _cat(vals["max"], np.float16),
        "ans_frac": _cat(vals["frac"], np.float16),
        "px_feat_idx": _cat(pxi, np.int32),
        "px_row_ptr": np.asarray(pxp, dtype=np.int64),
        "px_feat_val": _cat(pxv, np.float16),
        "cx_feat_idx": _cat(cxi, np.int32),
        "cx_row_ptr": np.asarray(cxp, dtype=np.int64),
        "cx_feat_val": _cat(cxv, np.float16),
        "px_dense19": (
            torch.stack(px_keep).to(torch.float16) if px_keep else torch.zeros(0, H_DIM)
        ),
        "cx_dense19": (
            torch.stack(cx_keep).to(torch.float16) if cx_keep else torch.zeros(0, H_DIM)
        ),
        "n_ans_tokens": np.asarray(n_ans, dtype=np.int32),
        "n_inlier_tokens": np.asarray(n_inl, dtype=np.int32),
        "dropped_ci": [v["ci"] for v in violations],
        "layers": [args.layer],
        "src_chunk": bundle.get("src_chunk", ""),
        "shard_index": int(bundle.get("shard_index", -1)),
        "chunk": int(bundle.get("chunk", -1)),
        "sae": {
            "repo": SAEMOD.SAE_REPO,
            "revision": SAEMOD.SAE_REVISION,
            "k": sae.k,
            "dict_size": sae.dict_size,
        },
    }
    stats = {
        "n_rows": n,
        "n_kept": len(rec["ci"]),
        "px_l0_mean": float(np.mean(px_l0s)) if px_l0s else float("nan"),
        "cx_l0_mean": float(np.mean(cx_l0s)) if cx_l0s else float("nan"),
    }
    return chunk, violations, stats


def _parity_probe(hf, tok, bundle: dict, args) -> dict:
    """G-S1 32-row batched-vs-per-row probe: per-row span-MEAN L19 cosine between
    the production batched capture and a batch-1 capture (bf16 padded-batch
    equivalence on span means — flat 0.999 bar per the calibration gotcha)."""
    li = list(bundle["layers"]).index(args.layer)
    n = min(PARITY_ROWS, len(bundle["ci"]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows_ids, plens = [], []
    for i in range(n):
        ids, plen = _render_row(tok, json.loads(bundle["prompts"][i]), bundle["response"][i])
        rows_ids.append(ids)
        plens.append(plen)
    batched = _capture_answer_spans(hf, rows_ids, plens, args.layer, args.sae_batch, pad_id)
    single = _capture_answer_spans(hf, rows_ids, plens, args.layer, 1, pad_id)
    coss = [_cos(batched[i].mean(0), single[i].mean(0)) for i in range(n)]
    _ = li  # layer column asserted upstream; probe reads the capture path only
    return {"n_rows": n, "min_cos": float(min(coss)), "median_cos": float(np.median(coss))}


def _fve_gate(
    pool: torch.Tensor,
    cache_dir: Path,
    *,
    smoke_sae=None,
    split_pools: dict[str, torch.Tensor] | None = None,
) -> dict:
    """G-S0: reference-parity fve_l0 at k=64 (primary) + k=128 (robustness twin)
    on the accumulated inlier token pool. ``smoke_sae`` (a tiny from-config SAE)
    substitutes both trainers under --smoke; production loads the pinned suite."""
    out: dict = {
        "n_tokens": int(pool.shape[0]),
        "fve_min": FVE_MIN,
        "l0_bounds": [L0_PUBLISHED / L0_MULT, L0_PUBLISHED * L0_MULT],
    }
    for k in (K_PRIMARY, K_ROBUST):
        if smoke_sae is not None:
            sae_k = smoke_sae
        else:
            sae_k = SAEMOD.BatchTopKSAE.load(
                k=k,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=cache_dir,
                layer=LAYER,
            )
        fve, l0, diag = sae_k.fve_l0(pool)
        out[f"k{k}"] = {
            "fve": float(fve),
            "l0": float(l0),
            "diag": diag,
            "published_fve": SAEMOD.PUBLISHED_FVE_BY_LAYER[LAYER].get(k),
        }
        if k == K_PRIMARY and split_pools:
            # answer- vs context-token FVE split (informational — makes a
            # marginal combined G-S0 read attributable; the gate binds on the
            # combined pool above).
            out[f"k{k}_split"] = {}
            for name, sub in split_pools.items():
                if sub.shape[0] < 2:
                    out[f"k{k}_split"][name] = {"n_tokens": int(sub.shape[0])}
                    continue
                sf, sl0, sdiag = sae_k.fve_l0(sub)
                out[f"k{k}_split"][name] = {
                    "n_tokens": int(sub.shape[0]),
                    "fve": float(sf),
                    "l0": float(sl0),
                    "diag": sdiag,
                }
        if smoke_sae is None and k != K_PRIMARY:
            del sae_k
    g = out[f"k{K_PRIMARY}"]
    out["pass"] = bool(
        g["fve"] >= FVE_MIN and (L0_PUBLISHED / L0_MULT) <= g["l0"] <= (L0_PUBLISHED * L0_MULT)
    )
    return out


def _load_hf_model(args):
    """bf16 HF model + tokenizer (teacher-forced forwards only; no vLLM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": 0} if args.device == "cuda" else None
    )
    hf.eval()
    return hf, tok


def _remote_done(prefix: str) -> set[str]:
    """Basenames already uploaded under {prefix}/capture (resume index)."""
    return {n for n in N50._remote_index(f"{prefix}/{GG.CAPTURE_SUBDIR}") if n.endswith(".pt")}


def _flush_sae_chunks(
    scratch: Path, sae_prefix: str, names: list[str], *, upload_fn=None, remote_index_fn=None
) -> None:
    """K-batched upload -> sha-verify -> purge of sae chunks (parent
    ``_flush_upload_batch_mt`` pattern; Hub boundary injectable for the smoke's
    signature-bound fakes)."""
    if not names:
        return
    local_shas = {n: N50._sha256_file(scratch / n) for n in names}
    up = upload_fn or hub._upload_folder_filtered
    url = up(
        scratch,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{sae_prefix}/{GG.CAPTURE_SUBDIR}",
        allow_patterns=list(names),
        expected_repo_paths=[f"{sae_prefix}/{GG.CAPTURE_SUBDIR}/{n}" for n in names],
    )
    if not url:
        raise RuntimeError(f"sae chunk batch upload ({len(names)} files) returned no URL")
    remote = (remote_index_fn or N50._remote_index)(f"{sae_prefix}/{GG.CAPTURE_SUBDIR}")
    for n in names:
        meta = remote.get(n)
        if meta is None:
            raise RuntimeError(f"{n} not on Hub after batch upload (verify listing)")
        if meta["sha256"] is None or meta["sha256"] != local_shas[n]:
            raise RuntimeError(f"{n} Hub sha {meta['sha256']} != local {local_shas[n]}")
    for n in names:
        (scratch / n).unlink()
    logger.info("[sae-upload] batch of %d sae chunks verified (sha) + purged", len(names))


def run_capture(args) -> int:
    """S1: per-shard sae-forward capture over the parent chunks (+ pilot gates)."""
    C.phase("sae-capture-setup")
    sae_prefix = None if (args.no_upload and args.local_capture_dir) else _require_sae_prefix(args)
    scratch = Path(args.out_dir) / "sae_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cache = Path(args.out_dir) / "parent_cache"
    cache.mkdir(parents=True, exist_ok=True)

    # parent chunk inventory (read-side: the parent's own prefix)
    if args.local_capture_dir:
        names = sorted(p.name for p in Path(args.local_capture_dir).glob("*.pt"))
    else:
        names = sorted(
            n
            for n in N50._remote_index(f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}")
            if n.endswith(".pt")
        )
    if not names:
        raise SystemExit("no parent capture chunks found — nothing to sae-encode")
    own = [n for k, n in enumerate(names) if k % args.num_shards == args.shard_index]
    logger.info(
        "[sae-capture] shard %d/%d owns %d of %d parent chunks",
        args.shard_index,
        args.num_shards,
        len(own),
        len(names),
    )

    # resume: skip chunks whose sae twin is already uploaded (or staged locally)
    if args.no_upload:
        done = {p.name for p in Path(args.out_dir).glob("sae_chunks/*.pt")}
    else:
        done = _remote_done(sae_prefix)
    todo = [n for n in own if n not in done]
    logger.info("[sae-capture] resume: %d done, %d todo", len(own) - len(todo), len(todo))

    # pilot short-circuit: meta already on Hub with a verdict
    pilot = args.pilot_rows > 0
    if pilot and not args.no_upload:
        try:
            meta_p = hub.stage_hub_file(
                C.HF_DATA_REPO,
                f"{sae_prefix}/{PILOT_META_NAME}",
                scratch / PILOT_META_NAME,
                repo_type="dataset",
                overwrite=True,
            )
            prev = json.loads(Path(meta_p).read_text())
            if prev.get("gate_s0", {}).get("pass") and prev.get("gate_s1", {}).get("pass"):
                logger.info("[pilot] existing sae_pilot_meta.json PASS on Hub — pilot skipped")
                if args.pilot_only:
                    return 0
                pilot = False
            elif "gate_s0" in prev and not prev["gate_s0"].get("pass", True):
                logger.error("[pilot] existing sae_pilot_meta.json records G-S0 FAIL — halting")
                return RC_SAE_FVE
        except Exception:
            logger.info("[pilot] no prior pilot meta on Hub — running the pilot")

    if not todo and not pilot:
        logger.info("[sae-capture] nothing to do (all chunks uploaded)")
        C.phase("done")
        return 0

    if args.smoke_model_dir:
        hf, tok, sae = _smoke_models(Path(args.smoke_model_dir), args)
    else:
        hf, tok = _load_hf_model(args)
        SAEMOD.BatchTopKSAE.ensure_downloaded(K_PRIMARY, Path(args.sae_cache), layer=LAYER)
        sae = SAEMOD.BatchTopKSAE.load(
            k=K_PRIMARY,
            device="cuda" if args.device == "cuda" else "cpu",
            cache_dir=Path(args.sae_cache),
            layer=LAYER,
        )

    C.phase("sae-capture")
    out_chunks = Path(args.out_dir) / "sae_chunks"
    out_chunks.mkdir(parents=True, exist_ok=True)
    # split pools so a marginal G-S0 is attributable (answer- vs context-token FVE)
    fve_pool = (
        {"answer_tokens": _FvePool(), "context_tokens": _FvePool(cap=50_000)} if pilot else None
    )
    pilot_rows_seen = 0
    parity: dict | None = None
    violations_all: list[dict] = []
    vx_cos_all: list[float] = []
    kept_total = 0
    rows_total = 0
    pending: list[str] = []
    bytes_written = 0
    t0 = time.time()
    for k, name in enumerate(todo):
        if args.local_capture_dir:
            local = Path(args.local_capture_dir) / name
        else:
            local = Path(
                PF._download_chunk_with_retry(
                    C.HF_DATA_REPO, f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}/{name}", cache
                )
            )
        bundle = torch.load(local, map_location="cpu", weights_only=False)
        bundle["src_chunk"] = name
        if pilot and parity is None:
            parity = _parity_probe(hf, tok, bundle, args)
            logger.info("[pilot] parity probe: %s", parity)
        chunk, violations, stats = _process_chunk(hf, tok, sae, bundle, args, fve_pool)
        violations_all.extend(violations)
        vx_cos_all.extend(np.asarray(chunk["vx_cos"], dtype=np.float64).tolist())
        kept_total += stats["n_kept"]
        rows_total += stats["n_rows"]
        out_p = out_chunks / name if args.no_upload else scratch / name
        torch.save(chunk, out_p)
        bytes_written += out_p.stat().st_size
        if not args.no_upload:
            pending.append(name)
            if len(pending) >= GG.UPLOAD_BATCH:
                _flush_sae_chunks(scratch, sae_prefix, pending)
                pending = []
        if not args.local_capture_dir:
            local.unlink(missing_ok=True)  # purge parent chunk — footprint ~1 chunk
        logger.info(
            "[sae-capture] chunk %d/%d %s rows=%d kept=%d px_l0=%.1f cx_l0=%.1f elapsed=%.0fs",
            k + 1,
            len(todo),
            name,
            stats["n_rows"],
            stats["n_kept"],
            stats["px_l0_mean"],
            stats["cx_l0_mean"],
            time.time() - t0,
        )
        if pilot:
            pilot_rows_seen += stats["n_rows"]
            pool_n = sum(v.n for v in fve_pool.values()) if fve_pool else 0
            enough_tokens = fve_pool is not None and pool_n >= min(
                FVE_TOKEN_FLOOR, args.fve_token_floor
            )
            if pilot_rows_seen >= args.pilot_rows and enough_tokens:
                rc = _finish_pilot(
                    args,
                    sae_prefix,
                    fve_pool,
                    parity,
                    violations_all,
                    vx_cos_all,
                    kept_total,
                    rows_total,
                    bytes_written,
                    time.time() - t0,
                    k + 1,
                    len(names),
                )
                if rc != 0 or args.pilot_only:
                    if pending and rc == 0:
                        _flush_sae_chunks(scratch, sae_prefix, pending)
                    return rc
                pilot = False  # gates passed — continue into production chunks
    if pilot and fve_pool is not None:
        # shard exhausted before the pilot floor (tiny smoke slices land here):
        rc = _finish_pilot(
            args,
            sae_prefix,
            fve_pool,
            parity,
            violations_all,
            vx_cos_all,
            kept_total,
            rows_total,
            bytes_written,
            time.time() - t0,
            max(1, len(todo)),
            len(names),
        )
        if rc != 0:
            return rc
    if pending:
        _flush_sae_chunks(scratch, sae_prefix, pending)
    viol_rate = len(violations_all) / max(1, rows_total)
    if viol_rate > VIOLATION_RATE_MAX and not args.smoke_model_dir:
        raise RuntimeError(
            f"v_x identity violations {len(violations_all)}/{rows_total} = {viol_rate:.4f} "
            f"> {VIOLATION_RATE_MAX} — render/span convention drift (code bug, not science)"
        )
    C.phase("done")
    logger.info("[sae-capture] shard done: kept=%d viol=%d", kept_total, len(violations_all))
    return 0


def _finish_pilot(
    args,
    sae_prefix,
    fve_pool,
    parity,
    violations_all,
    vx_cos_all,
    kept_total,
    rows_total,
    bytes_written,
    wall_s,
    chunks_processed,
    n_chunks_total,
) -> int:
    """G-S0 + G-S1 verdicts -> sae_pilot_meta.json (local + Hub). Designed rcs:
    27 (fitness kill), 26 (rate fence). Under --smoke the production-n verdicts
    are computed + logged but demoted to informational (gate-calibration
    gotcha); the halt BRANCHES are exercised by the smoke's degenerate probes."""
    C.phase("sae-pilot-gates")
    split_pools = {name: fp.tensor() for name, fp in fve_pool.items()}
    pool = torch.cat([v for v in split_pools.values() if v.shape[0] > 0])
    smoke_sae = None
    if args.smoke_model_dir:
        _hf, _tok, smoke_sae = _smoke_models(Path(args.smoke_model_dir), args, model=False)
    gate_s0 = _fve_gate(pool, Path(args.sae_cache), smoke_sae=smoke_sae, split_pools=split_pools)
    viol_rate = len(violations_all) / max(1, rows_total)
    med_cos = float(np.median(vx_cos_all)) if vx_cos_all else float("nan")
    rate_rows_per_s = kept_total / max(1e-9, wall_s)
    # projected own-shard wall: measured per-chunk wall x chunks this shard owns
    chunks_owned = int(np.ceil(n_chunks_total / args.num_shards))
    wall_per_chunk = wall_s / max(1, chunks_processed)
    projected_shard_min = wall_per_chunk * chunks_owned / 60.0
    gate_s1 = {
        "vx_median_cos": med_cos,
        "vx_median_min": VX_COS_MEDIAN_MIN,
        "violation_rate": float(viol_rate),
        "violation_rate_max": VIOLATION_RATE_MAX,
        "parity": parity,
        "parity_min_cos": PARITY_MIN_COS,
        "rows_per_s": float(rate_rows_per_s),
        "wall_per_chunk_s": float(wall_per_chunk),
        "projected_shard_wall_min": float(projected_shard_min),
        "shard_fence_min": args.shard_fence_min,
        "bytes_per_row": float(bytes_written / max(1, kept_total)),
        "pass": bool(
            (not np.isnan(med_cos) and med_cos >= VX_COS_MEDIAN_MIN)
            and viol_rate <= VIOLATION_RATE_MAX
            and parity is not None
            and parity["min_cos"] >= PARITY_MIN_COS
            and projected_shard_min <= args.shard_fence_min
        ),
    }
    doc = {
        "gate_s0": gate_s0,
        "gate_s1": gate_s1,
        "n_rows_attempted": int(rows_total),
        "n_rows_kept": int(kept_total),
        "n_violations": len(violations_all),
        "violations_head": violations_all[:20],
        "sae": {
            "repo": SAEMOD.SAE_REPO,
            "revision": SAEMOD.SAE_REVISION,
            "k_primary": K_PRIMARY,
            "k_robust": K_ROBUST,
            "layer": args.layer,
        },
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "smoke": bool(args.smoke_model_dir),
        "git_commit": MTF._git_head(),
        "torch": torch.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out_dir) / PILOT_META_NAME
    C.write_json_atomic(out, doc)
    logger.info("[pilot] G-S0=%s G-S1=%s", gate_s0.get("pass"), gate_s1.get("pass"))
    if not args.no_upload:
        # upload_as_file=True: path_in_repo is the FULL FILE destination (fu1
        # incident dd9a615c22 — never the bare prefix itself).
        url = hub._upload(
            out,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{sae_prefix}/{PILOT_META_NAME}",
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError("sae_pilot_meta.json upload returned no URL")
    rc = _pilot_verdict_rc(gate_s0, gate_s1, bool(args.smoke_model_dir))
    if rc == RC_SAE_FVE:
        logger.error(
            "[G-S0] FVE/L0 fitness FAIL — HALT (the round's kill criterion): %s",
            gate_s0[f"k{K_PRIMARY}"],
        )
    elif rc == RC_SAE_FENCE:
        logger.error(
            "[G-S1] rate fence tripped: %.1f min > %.1f — designed halt rc %d",
            gate_s1["projected_shard_wall_min"],
            args.shard_fence_min,
            RC_SAE_FENCE,
        )
    elif args.smoke_model_dir:
        logger.info("[pilot] --smoke: gate verdicts informational (production-n bars)")
    return rc


# ── S2 fits: sae-chunk streaming -> CSR -> shared-Gram ridge cells ──────────────────


def _pilot_verdict_rc(gate_s0: dict, gate_s1: dict, smoke: bool) -> int:
    """Pure verdict -> rc routing (smoke-probeable): 0 = proceed, 27 = G-S0
    fitness kill (designed), 26 = G-S1 rate fence (designed); an identity/parity
    FAIL raises (code bug, not science). Under smoke every verdict is
    informational (production-n bars — gate-calibration gotcha)."""
    if smoke:
        return 0
    if not gate_s0["pass"]:
        return RC_SAE_FVE
    parity = gate_s1.get("parity")
    if not (
        gate_s1["violation_rate"] <= VIOLATION_RATE_MAX
        and gate_s1["vx_median_cos"] >= VX_COS_MEDIAN_MIN
        and (parity is None or parity["min_cos"] >= PARITY_MIN_COS)
    ):
        raise RuntimeError(f"[G-S1] identity/parity FAIL (code bug, not science): {gate_s1}")
    if gate_s1["projected_shard_wall_min"] > gate_s1["shard_fence_min"]:
        return RC_SAE_FENCE
    return 0


def _sae_chunk_names(args, sae_prefix: str | None) -> list[str]:
    if args.local_sae_dir:
        names = sorted(p.name for p in Path(args.local_sae_dir).glob("*.pt"))
    else:
        assert sae_prefix
        names = sorted(
            n for n in N50._remote_index(f"{sae_prefix}/{GG.CAPTURE_SUBDIR}") if n.endswith(".pt")
        )
    if not names:
        raise SystemExit("no sae capture chunks found — run --phase capture first")
    return names


def _stage_sae_chunks(args, sae_prefix: str | None, names: list[str], cache: Path) -> list[Path]:
    """Download-once staging (skip-if-present); local dir passthrough for smoke."""
    if args.local_sae_dir:
        return [Path(args.local_sae_dir) / n for n in names]
    out = []
    for i, n in enumerate(names):
        p = cache / n
        if not p.exists():
            got = PF._download_chunk_with_retry(
                C.HF_DATA_REPO, f"{sae_prefix}/{GG.CAPTURE_SUBDIR}/{n}", cache
            )
            p = Path(got)
        out.append(p)
        if (i + 1) % 25 == 0 or (i + 1) == len(names):
            logger.info("[sae-fits] staged chunk %d/%d", i + 1, len(names))
    return out


def _scan_sae(paths: list[Path], train_ci: set[int], dict_size: int) -> dict:
    """Pass 1 (SN.scan_counts analogue): TRAIN-row activity counts (answer +
    per-arm inputs), all-row counts (nnz preallocation), ci/corpus/drop
    inventory in stream order (= parent capture order minus drops)."""
    out_fit = np.zeros(dict_size, dtype=np.int64)
    out_all = np.zeros(dict_size, dtype=np.int64)
    in_fit = {"px": np.zeros(dict_size, np.int64), "cx": np.zeros(dict_size, np.int64)}
    in_all = {"px": np.zeros(dict_size, np.int64), "cx": np.zeros(dict_size, np.int64)}
    ci_all: list[np.ndarray] = []
    corpus_all: list[str] = []
    dropped: list[int] = []
    n_fit = 0
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        ci = np.asarray(d["ci"], dtype=np.int64)
        ci_all.append(ci)
        corpus_all.extend(d["corpus"])
        dropped.extend(int(x) for x in d.get("dropped_ci", []))
        fit_mask = np.asarray([int(c) in train_ci for c in ci], dtype=bool)
        n_fit += int(fit_mask.sum())
        off = np.diff(np.asarray(d["row_ptr"], dtype=np.int64))
        idx = np.asarray(d["feat_idx"], dtype=np.int64)
        out_all += np.bincount(idx, minlength=dict_size)
        out_fit += np.bincount(idx[np.repeat(fit_mask, off)], minlength=dict_size)
        for arm, ik, pk in (
            ("px", "px_feat_idx", "px_row_ptr"),
            ("cx", "cx_feat_idx", "cx_row_ptr"),
        ):
            offa = np.diff(np.asarray(d[pk], dtype=np.int64))
            idxa = np.asarray(d[ik], dtype=np.int64)
            in_all[arm] += np.bincount(idxa, minlength=dict_size)
            in_fit[arm] += np.bincount(idxa[np.repeat(fit_mask, offa)], minlength=dict_size)
    ci_cat = np.concatenate(ci_all) if ci_all else np.zeros(0, np.int64)
    assert len(set(ci_cat.tolist())) == len(ci_cat), "duplicate ci across sae chunks"
    return {
        "out_fit": out_fit,
        "out_all": out_all,
        "in_fit": in_fit,
        "in_all": in_all,
        "n_fit": n_fit,
        "ci": ci_cat,
        "corpus": np.asarray(corpus_all),
        "dropped": dropped,
    }


def _build_sae_matrices(
    paths: list[Path], scan: dict, f_out: np.ndarray, f_in: dict[str, np.ndarray], mm_dir: Path
):
    """Pass 2 (SN.build_csr analogue): CSR X per arm + Y per pooling (restricted
    columns, fp32 vals) + dense px/cx fp32 memmaps, all in stream order."""
    n_rows = len(scan["ci"])
    dict_size = len(scan["out_all"])
    col_out = np.full(dict_size, -1, dtype=np.int64)
    col_out[f_out] = np.arange(len(f_out))
    col_in = {}
    for arm in ("px", "cx"):
        c = np.full(dict_size, -1, dtype=np.int64)
        c[f_in[arm]] = np.arange(len(f_in[arm]))
        col_in[arm] = c
    nnz_y = int(scan["out_all"][f_out].sum())
    ycoo = {
        p: (np.empty(nnz_y, np.int32), np.empty(nnz_y, np.int32), np.empty(nnz_y, np.float32))
        for p in POOLINGS
    }
    xcoo = {}
    for arm in ("px", "cx"):
        nnz = int(scan["in_all"][arm][f_in[arm]].sum())
        xcoo[arm] = (np.empty(nnz, np.int32), np.empty(nnz, np.int32), np.empty(nnz, np.float32))
    mm_dir.mkdir(parents=True, exist_ok=True)
    h_dense = None
    cur_y = 0
    cur_x = {"px": 0, "cx": 0}
    row0 = 0
    dense_mm: dict[str, np.memmap] = {}
    val_key = {"mean": "ans_mean", "max": "ans_max", "frac": "ans_frac"}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = len(d["ci"])
        if h_dense is None:
            h_dense = int(d["px_dense19"].shape[1])
            for arm in ("px", "cx"):
                dense_mm[arm] = np.memmap(
                    mm_dir / f"{arm}_dense.bin",
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_rows, h_dense),
                )
        for arm, key in (("px", "px_dense19"), ("cx", "cx_dense19")):
            dense_mm[arm][row0 : row0 + n] = (
                d[key].to(torch.float32).numpy()
                if isinstance(d[key], torch.Tensor)
                else np.asarray(d[key], dtype=np.float32)
            )
        off = np.diff(np.asarray(d["row_ptr"], dtype=np.int64))
        idx = np.asarray(d["feat_idx"], dtype=np.int64)
        rr = np.repeat(np.arange(row0, row0 + n), off)
        cc = col_out[idx]
        keep = cc >= 0
        m = int(keep.sum())
        for pool in POOLINGS:
            r, c, v = ycoo[pool]
            r[cur_y : cur_y + m] = rr[keep]
            c[cur_y : cur_y + m] = cc[keep]
            v[cur_y : cur_y + m] = np.asarray(d[val_key[pool]], dtype=np.float32)[keep]
        cur_y += m
        for arm, ik, pk, vk in (
            ("px", "px_feat_idx", "px_row_ptr", "px_feat_val"),
            ("cx", "cx_feat_idx", "cx_row_ptr", "cx_feat_val"),
        ):
            offa = np.diff(np.asarray(d[pk], dtype=np.int64))
            idxa = np.asarray(d[ik], dtype=np.int64)
            rra = np.repeat(np.arange(row0, row0 + n), offa)
            cca = col_in[arm][idxa]
            keepa = cca >= 0
            ma = int(keepa.sum())
            r, c, v = xcoo[arm]
            r[cur_x[arm] : cur_x[arm] + ma] = rra[keepa]
            c[cur_x[arm] : cur_x[arm] + ma] = cca[keepa]
            v[cur_x[arm] : cur_x[arm] + ma] = np.asarray(d[vk], dtype=np.float32)[keepa]
            cur_x[arm] += ma
        row0 += n
    assert row0 == n_rows, (row0, n_rows)
    Y = {
        p: sp.coo_matrix(
            (ycoo[p][2][:cur_y], (ycoo[p][0][:cur_y], ycoo[p][1][:cur_y])),
            shape=(n_rows, len(f_out)),
        ).tocsr()
        for p in POOLINGS
    }
    X = {
        arm: sp.coo_matrix(
            (xcoo[arm][2][: cur_x[arm]], (xcoo[arm][0][: cur_x[arm]], xcoo[arm][1][: cur_x[arm]])),
            shape=(n_rows, len(f_in[arm])),
        ).tocsr()
        for arm in ("px", "cx")
    }
    for arm in ("px", "cx"):
        dense_mm[arm].flush()
        dense_mm[arm] = np.memmap(
            mm_dir / f"{arm}_dense.bin", dtype=np.float32, mode="r", shape=(n_rows, h_dense)
        )
    return X, Y, dense_mm, h_dense


def _rows(X, rows: np.ndarray) -> np.ndarray:
    """fp32 dense block from CSR or memmap."""
    if sp.issparse(X):
        return X[rows].toarray().astype(np.float32, copy=False)
    return np.asarray(X[rows], dtype=np.float32)


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mu = true.mean(axis=0)
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - mu) ** 2).sum())
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


class _GramFactor:
    """Shared-Gram fp64 factorization of one standardized X source (train rows),
    reused across every Y pooling + the shuffle battery (vectorize-first: one
    eigh per X source, per-lambda solves are cheap spectral rescalings)."""

    def __init__(self, X, tr: np.ndarray, dev: torch.device, block: int):
        self.X = X
        self.tr = tr
        self.dev = dev
        self.block = block
        h = X.shape[1]
        sum_x = torch.zeros(h, dtype=torch.float64, device=dev)
        sumsq = torch.zeros(h, dtype=torch.float64, device=dev)
        for s in range(0, len(tr), block):
            xb = torch.from_numpy(_rows(X, tr[s : s + block])).to(dev, torch.float64)
            sum_x += xb.sum(0)
            sumsq += (xb * xb).sum(0)
        n = len(tr)
        self.xmu = sum_x / n
        var = (sumsq - n * self.xmu * self.xmu) / max(1, n - 1)
        self.xsd = torch.sqrt(torch.clamp(var, min=0.0)) + 1e-9
        A = torch.zeros((h, h), dtype=torch.float64, device=dev)
        colsum = torch.zeros(h, dtype=torch.float64, device=dev)
        for s in range(0, len(tr), block):
            xb = self._std_np(_rows(X, tr[s : s + block]))
            A += xb.T @ xb
            colsum += xb.sum(0)
        self.colsum = colsum
        try:
            s_eig, U = torch.linalg.eigh(A)
        except torch.linalg.LinAlgError:
            # cuSOLVER syevd non-convergence on near-singular Grams — CPU LAPACK
            # decomposes these fine (gotchas: cusolver eigh -> CPU fallback).
            logger.warning("[gram] cuda eigh failed to converge — CPU LAPACK fallback (h=%d)", h)
            s_eig, U = torch.linalg.eigh(A.cpu())
            s_eig, U = s_eig.to(dev), U.to(dev)
        self.s_eig = torch.clamp(s_eig, min=0.0)
        self.U = U

    def _std_np(self, xb: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(xb).to(self.dev, torch.float64) - self.xmu) / self.xsd

    def std_rows(self, rows: np.ndarray) -> torch.Tensor:
        return self._std_np(_rows(self.X, rows))

    def xty_centered(
        self, Y, rows: np.ndarray, ymu: torch.Tensor, perm: np.ndarray | None = None
    ) -> torch.Tensor:
        """Centered X_std[rows]^T Y[rows[perm]] (fp64 accumulate; fp32 inputs).
        ``perm`` permutes the Y side within ``rows`` (the shuffle draws)."""
        h = self.X.shape[1]
        d = Y.shape[1]
        out = torch.zeros((h, d), dtype=torch.float64, device=self.dev)
        for s in range(0, len(rows), self.block):
            xb = self.std_rows(rows[s : s + self.block])
            yrows = rows[s : s + self.block] if perm is None else rows[perm[s : s + self.block]]
            yb = torch.from_numpy(_rows(Y, yrows)).to(self.dev, torch.float64)
            out += xb.T @ yb
        return out - torch.outer(self.colsum, ymu)


def _fit_cell(fac: _GramFactor, Y, tr, val, ho, lambdas) -> dict:
    """One shared-Gram ridge cell: lambda selected on val (pooled R^2), holdout
    predictions returned fp32. Also returns the draw kit (M2, ymu, XtY) for the
    shuffle battery + per-feature reads."""
    dev = fac.dev
    ymu = (
        torch.from_numpy(np.asarray(_rows(Y, tr), dtype=np.float64)).to(dev).mean(0)
        if len(tr) * Y.shape[1] <= 5e7
        else None
    )
    if ymu is None:  # streamed fp64 column mean for the big production shape
        acc = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
        for s in range(0, len(tr), fac.block):
            acc += torch.from_numpy(_rows(Y, tr[s : s + fac.block])).to(dev, torch.float64).sum(0)
        ymu = acc / len(tr)
    xty = fac.xty_centered(Y, tr, ymu)
    B = fac.U.T @ xty  # (h, d) fp64 — once per (source, pooling)
    e_val = fac.std_rows(val) @ fac.U
    y_val = torch.from_numpy(_rows(Y, val)).to(dev, torch.float64)
    val_r2 = {}
    best = None
    for lam in lambdas:
        inv = 1.0 / (fac.s_eig + float(lam))
        pred = (e_val * inv) @ B + ymu
        r2 = _pooled_r2(pred.cpu().numpy(), y_val.cpu().numpy())
        val_r2[float(lam)] = float(r2)
        if best is None or (np.isfinite(r2) and r2 > best[1]):
            best = (float(lam), float(r2))
    sel_lam = best[0]
    inv = 1.0 / (fac.s_eig + sel_lam)
    e_ho = fac.std_rows(ho) @ fac.U
    pred_ho = ((e_ho * inv) @ B + ymu).float().cpu().numpy()
    m2 = ((e_ho * inv) @ fac.U.T).float()  # (n_ho, h) fp32 — the shuffle-draw kit
    return {
        "selected_lambda": sel_lam,
        "val_r2_selected": best[1],
        "val_r2_by_lambda": val_r2,
        "pred_ho": pred_ho,
        "m2": m2,
        "ymu": ymu,
    }


def _shuffle_draws(fac: _GramFactor, Y, tr, ho, kit: dict, k_draws: int, seed_base: int) -> dict:
    """K label-shuffle draws at the arm's SELECTED lambda (SN stage-8 verbatim:
    permute Y within the fit rows, score TRUE holdout pairs). Returns per-draw
    pooled R^2 + per-feature null R^2 (K, F) fp32."""
    dev = fac.dev
    y_ho = np.asarray(_rows(Y, ho), dtype=np.float64)
    ss_tot_f, _mu = _perfeature_ss_tot(y_ho)
    pooled, perfeat = [], []
    for k in range(k_draws):
        perm = np.random.default_rng(seed_base + k).permutation(len(tr))
        xty = fac.xty_centered(Y, tr, kit["ymu"], perm=perm)
        pred = (kit["m2"].double() @ xty + kit["ymu"]).cpu().numpy()
        pooled.append(_pooled_r2(pred, y_ho))
        perfeat.append(_perfeature_r2(pred, y_ho, ss_tot_f))
        logger.info("[shuffle] draw %d/%d pooled_r2=%.5f", k + 1, k_draws, pooled[-1])
    return {
        "pooled": np.asarray(pooled, dtype=np.float64),
        "perfeature": np.stack(perfeat).astype(np.float32),
        "seeds": [seed_base + k for k in range(k_draws)],
    }


def _perfeature_ss_tot(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = y.mean(axis=0)
    return ((y - mu) ** 2).sum(axis=0), mu


def _perfeature_r2(pred: np.ndarray, true: np.ndarray, ss_tot: np.ndarray) -> np.ndarray:
    ss_res = ((np.asarray(true, np.float64) - np.asarray(pred, np.float64)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho over finite pairs (nan when < 3 finite pairs)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a[ok], b[ok]).statistic)


def _upload_sae_analysis(args, sae_prefix: str, entries: list[tuple[str, Path, list[str] | None]]):
    """One verified upload_folder commit per entry ->
    {sae_prefix}/analysis_tensors/{sub} (fail-loud; MTF pattern, sae prefix)."""
    for sub, local, files in entries:
        if files is None:
            files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
        if not files:
            continue
        url = hub._upload_folder_filtered(
            local,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{sae_prefix}/{MTF.ANALYSIS_TENSORS_SUBDIR}/{sub}",
            allow_patterns=files,
            expected_repo_paths=[
                f"{sae_prefix}/{MTF.ANALYSIS_TENSORS_SUBDIR}/{sub}/{f}" for f in files
            ],
        )
        if not url:
            raise RuntimeError(f"sae analysis-tensors upload ({sub}) returned no URL")


def run_fits(args) -> int:
    """S2: SAE->SAE + dense->SAE shared-Gram ridge cells, shuffle floor,
    per-feature reads, mapping baselines, lmsys_transfer (plan §4.1 S2)."""
    C.phase("sae-fits-assemble")
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    sae_prefix = None if args.local_sae_dir else _require_sae_prefix(args)
    out_eval = Path(args.out_eval)
    out_local = Path(args.out_local)
    mm_dir = out_local / "mm"
    cells_dir = out_eval / "fits_cells"
    pred_dir = out_local / "pred16"
    pf_dir = out_local / "perfeature"
    for p in (out_eval, mm_dir, cells_dir, pred_dir, pf_dir):
        p.mkdir(parents=True, exist_ok=True)

    # pinned split (sha-assert) + parent split_shas cross-assert (v6 pattern)
    split = MTF.load_split(Path(args.split_file))
    MTF._assert_parent_split_shas(split, args.parent_fits_json)
    parent_fits = json.loads(Path(args.parent_fits_json).read_text())

    names = _sae_chunk_names(args, sae_prefix)
    fp = hashlib.sha256(
        (
            "\n".join(names)
            + f"|{sae_prefix}|{MAX_FEATURES_OUT}|{MAX_FEATURES_IN}|{SHUFFLE_SEED_BASE}"
            + f"|{args.k_draws}|{[float(x) for x in LAMBDAS]}"
        ).encode()
    ).hexdigest()
    cache = out_local / "sae_dl"
    cache.mkdir(parents=True, exist_ok=True)
    paths = _stage_sae_chunks(args, sae_prefix, names, cache)

    train_ci = {int(c) for c in split["sets"]["train"]["ci"]}
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    dict_size = int(first["sae"]["dict_size"])
    chunk_layer = int(first["layers"][0])
    del first
    scan = _scan_sae(paths, train_ci, dict_size)

    # coverage asserts (plan §4.1 S2.1): realized rows == parent captured minus
    # recorded drops; drop rate re-asserted under the parent violation budget.
    n_rows = len(scan["ci"])
    n_dropped = len(scan["dropped"])
    n_parent = int(parent_fits["n_rows_captured"])
    assert n_rows + n_dropped == n_parent, (
        f"sae rows {n_rows} + dropped {n_dropped} != parent captured {n_parent}"
    )
    drop_rate = n_dropped / max(1, n_rows + n_dropped)
    assert drop_rate < VIOLATION_RATE_MAX, (
        f"identity-gate drop rate {drop_rate:.4f} >= {VIOLATION_RATE_MAX}"
    )

    # feature restriction (#1482 recipe at this run's n; SN.restrict verbatim)
    f_out, floor = SN.restrict(scan["out_fit"], scan["n_fit"], MAX_FEATURES_OUT)
    f_in = {
        arm: SN.restrict(scan["in_fit"][arm], scan["n_fit"], MAX_FEATURES_IN)[0]
        for arm in ("px", "cx")
    }
    logger.info(
        "[sae-fits] restriction: F_out=%d F_in_px=%d F_in_cx=%d floor=%d",
        len(f_out),
        len(f_in["px"]),
        len(f_in["cx"]),
        floor,
    )

    X, Y, dense_mm, h_dense = _build_sae_matrices(paths, scan, f_out, f_in, mm_dir)
    sets = MTF.split_positions(split, scan["ci"])
    tr, val, ho = sets["train"], sets["val"], sets["holdout"]
    n_tr = len(tr)
    d_in_max = max(len(f_in["px"]), len(f_in["cx"]))
    if n_tr < d_in_max and not args.allow_underdetermined:
        raise SystemExit(
            f"n_train={n_tr} < d_in={d_in_max}: estimator-degenerate regime — pass "
            "--allow-underdetermined only for a deliberate smoke shape"
        )

    x_of = {
        "px_feat": X["px"],
        "cx_feat": X["cx"],
        "px_dense": dense_mm["px"],
        "cx_dense": dense_mm["cx"],
    }
    C.phase("sae-fits")
    summary: dict = {
        "fit_point": "multiturn_100k_sae",
        "layer": chunk_layer,
        "sae": {
            "repo": SAEMOD.SAE_REPO,
            "revision": SAEMOD.SAE_REVISION,
            "k": K_PRIMARY,
            "dict_size": dict_size,
        },
        "n_rows": n_rows,
        "n_dropped": n_dropped,
        "drop_rate": drop_rate,
        "split_counts": {k: int(len(v)) for k, v in sets.items()},
        "split_shas": {k: split["sets"][k]["sha256"] for k in split["sets"]},
        "restriction": {
            "n_f_out": int(len(f_out)),
            "n_f_in_px": int(len(f_in["px"])),
            "n_f_in_cx": int(len(f_in["cx"])),
            "activity_floor_rows": int(floor),
            "n_fit_rows": int(scan["n_fit"]),
        },
        "lambdas": [float(x) for x in LAMBDAS],
        "n_boot": int(args.n_boot),
        "boot_seed": BOOT_SEED,
        "assembly_fingerprint": fp,
        "cells": {},
        "shuffle_floor": {},
        "git_commit": MTF._git_head(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    facs: dict[str, _GramFactor] = {}
    kits: dict[str, dict] = {}
    first_wall: float | None = None
    t_cells0 = time.time()
    for cell_i, (cell, xsrc, pool) in enumerate(SAE_CELLS):
        if first_wall is not None and MTF._fence_should_halt(
            time.time() - t_cells0, first_wall, len(SAE_CELLS), args.fence_mult
        ):
            rep = {
                "gate": "G-S2",
                "first_cell_wall_s": first_wall,
                "elapsed_s": time.time() - t_cells0,
                "fence_mult": args.fence_mult,
                "cells_done": cell_i,
                "cells_total": len(SAE_CELLS),
            }
            GG.N1M._atomic_write_json(out_eval / "fence_report.json", rep)
            logger.error("[G-S2] fence tripped: %s", rep)
            sys.exit(RC_SAE_FIT_FENCE)
        cj = cells_dir / f"{cell}.json"
        if cj.exists() and not args.no_resume:
            doc = json.loads(cj.read_text())
            if doc.get("assembly_fingerprint") == fp:
                summary["cells"][cell] = doc["metrics"]
                logger.info("[cell] %s: resume-skip", cell)
                if first_wall is None:
                    first_wall = float(doc["metrics"].get("wall_s", 1.0))
                continue
        t0 = time.time()
        if xsrc not in facs:
            facs[xsrc] = _GramFactor(x_of[xsrc], tr, dev, args.block)
        kit = _fit_cell(facs[xsrc], Y[pool], tr, val, ho, LAMBDAS)
        y_ho = np.asarray(_rows(Y[pool], ho), dtype=np.float64)
        r2_ho = _pooled_r2(kit["pred_ho"], y_ho)
        ci_boot = MTF._boot_recon_ci_batched(kit["pred_ho"], y_ho, args.n_boot, BOOT_SEED)
        metrics = {
            "holdout_r2": float(r2_ho),
            "holdout_bootstrap_ci": ci_boot,
            "selected_lambda": kit["selected_lambda"],
            "val_r2_selected": kit["val_r2_selected"],
            "n_holdout": int(len(ho)),
            "pooling": pool,
            "x_source": xsrc,
            "wall_s": time.time() - t0,
        }
        np.savez(
            pred_dir / f"{cell}.npz",
            pred16=kit["pred_ho"].astype(np.float16),
            ci=scan["ci"][ho],
            fingerprint=np.array(fp),
        )
        if cell in ARM_X or pool == "mean":
            kits[cell] = kit  # draw kit retained for the shuffle battery + baselines
        summary["cells"][cell] = metrics
        GG.N1M._atomic_write_json(cj, {"metrics": metrics, "assembly_fingerprint": fp})
        if first_wall is None:
            first_wall = time.time() - t0
        logger.info(
            "[cell] %d/%d %s: holdout R2=%.4f lambda=%g wall=%.0fs",
            cell_i + 1,
            len(SAE_CELLS),
            cell,
            r2_ho,
            kit["selected_lambda"],
            metrics["wall_s"],
        )

    # ── shuffle floor + per-feature reads (SAE arms, mean pooling) ────────────
    C.phase("sae-shuffle")
    y_ho_mean = np.asarray(_rows(Y["mean"], ho), dtype=np.float64)
    ss_tot_f, _mu_ho = _perfeature_ss_tot(y_ho_mean)
    perfeat: dict[str, dict] = {}
    for arm_i, cell in enumerate(("sae_prefix", "sae_context")):
        null_p = pf_dir / f"null_{cell}.npz"
        if null_p.exists() and not args.no_resume:
            with np.load(null_p, allow_pickle=False) as z:
                if str(z["fingerprint"]) == fp:
                    perfeat[cell] = {
                        "null": z["perfeature"].copy(),
                        "pooled": z["pooled"].copy(),
                    }
                    summary["shuffle_floor"][cell] = {
                        "pooled_mean": float(np.mean(z["pooled"])),
                        "pooled_max": float(np.max(z["pooled"])),
                        "k_draws": int(len(z["pooled"])),
                        "resumed": True,
                    }
                    logger.info("[shuffle] %s: resume-skip", cell)
                    continue
        if cell not in kits:  # resumed cell — refit to rebuild the draw kit
            xsrc = ARM_X[cell] + "_feat"
            if xsrc not in facs:
                facs[xsrc] = _GramFactor(x_of[xsrc], tr, dev, args.block)
            kits[cell] = _fit_cell(facs[xsrc], Y["mean"], tr, val, ho, LAMBDAS)
        draws = _shuffle_draws(
            facs[ARM_X[cell] + "_feat"],
            Y["mean"],
            tr,
            ho,
            kits[cell],
            args.k_draws,
            SHUFFLE_SEED_BASE + arm_i * 1000,
        )
        np.savez(
            null_p,
            perfeature=draws["perfeature"],
            pooled=draws["pooled"],
            seeds=np.asarray(draws["seeds"]),
            fingerprint=np.array(fp),
        )
        perfeat[cell] = {"null": draws["perfeature"], "pooled": draws["pooled"]}
        summary["shuffle_floor"][cell] = {
            "pooled_mean": float(np.mean(draws["pooled"])),
            "pooled_max": float(np.max(draws["pooled"])),
            "k_draws": args.k_draws,
            "seeds": draws["seeds"],
        }

    C.phase("sae-perfeature")
    activity = scan["out_fit"][f_out] / max(1, scan["n_fit"])
    # within-answer consistency: mean ans_frac conditional on active (train rows)
    yf_tr = Y["frac"][tr]
    col_sum = np.asarray(yf_tr.sum(axis=0)).ravel()
    col_nnz = np.asarray((yf_tr != 0).sum(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        consistency = np.where(col_nnz > 0, col_sum / np.maximum(col_nnz, 1), np.nan)
    r2f: dict[str, np.ndarray] = {}
    carried: dict[str, np.ndarray] = {}
    splithalf: dict[str, float] = {}
    half_a = np.arange(len(ho))[::2]
    half_b = np.arange(len(ho))[1::2]
    for cell in ("sae_prefix", "sae_context"):
        with np.load(pred_dir / f"{cell}.npz", allow_pickle=False) as z:
            pred = z["pred16"].astype(np.float64)
        r2f[cell] = _perfeature_r2(pred, y_ho_mean, ss_tot_f)
        null_p95 = np.nanquantile(perfeat[cell]["null"].astype(np.float64), 0.95, axis=0)
        carried[cell] = r2f[cell] > null_p95
        perfeat[cell]["p95"] = null_p95
        # split-half rank stability over two disjoint holdout halves
        r2_a = _perfeature_r2(
            pred[half_a], y_ho_mean[half_a], _perfeature_ss_tot(y_ho_mean[half_a])[0]
        )
        r2_b = _perfeature_r2(
            pred[half_b], y_ho_mean[half_b], _perfeature_ss_tot(y_ho_mean[half_b])[0]
        )
        splithalf[cell] = _spearman(r2_a, r2_b)
    delta = r2f["sae_context"] - r2f["sae_prefix"]
    n_eval_feats = int(np.isfinite(delta).sum())
    # K=20 p95 threshold carries ~5% false-positive mass by construction (the
    # critic-mandated expected-false-carried companion to any carried count).
    expected_false = 0.05 * n_eval_feats
    loo_above = {}
    for cell in ("sae_prefix", "sae_context"):
        nz = perfeat[cell]["null"].astype(np.float64)
        cnt = 0
        for k in range(nz.shape[0]):
            others = np.delete(nz, k, axis=0)
            cnt += int(np.nansum(nz[k] > np.nanquantile(others, 0.95, axis=0)))
        loo_above[cell] = cnt / max(1, nz.shape[0] * nz.shape[1])
    summary["perfeature"] = {
        "n_features": int(len(f_out)),
        "n_finite_delta": n_eval_feats,
        "carried_prefix": int(np.nansum(carried["sae_prefix"])),
        "carried_context": int(np.nansum(carried["sae_context"])),
        "expected_false_carried_per_arm": float(expected_false),
        "loo_calibration_above_rate": {k: float(v) for k, v in loo_above.items()},
        "splithalf_rank_spearman": {k: float(v) for k, v in splithalf.items()},
        "delta_median": float(np.nanmedian(delta)),
        "delta_q10_q90": [float(np.nanquantile(delta, q)) for q in (0.1, 0.9)],
        "spearman_delta_vs_activity": _spearman(delta, np.log10(np.maximum(activity, 1e-9))),
        "spearman_delta_vs_consistency": _spearman(delta, consistency),
    }
    np.savez(
        pf_dir / "perfeature_summary.npz",
        feat_ids=f_out.astype(np.int64),
        activity=activity.astype(np.float32),
        consistency=consistency.astype(np.float32),
        r2_prefix=r2f["sae_prefix"].astype(np.float32),
        r2_context=r2f["sae_context"].astype(np.float32),
        delta=delta.astype(np.float32),
        null_p95_prefix=perfeat["sae_prefix"]["p95"].astype(np.float32),
        null_p95_context=perfeat["sae_context"]["p95"].astype(np.float32),
        carried_prefix=carried["sae_prefix"],
        carried_context=carried["sae_context"],
        fingerprint=np.array(fp),
    )
    with open(out_eval / "perfeature_summary.csv", "w") as f:
        f.write(
            "feat_id,activity,consistency,r2_prefix,r2_context,delta_r2,"
            "null_p95_prefix,null_p95_context,carried_prefix,carried_context\n"
        )
        for i in range(len(f_out)):
            f.write(
                f"{int(f_out[i])},{activity[i]:.6g},{consistency[i]:.6g},"
                f"{r2f['sae_prefix'][i]:.6g},{r2f['sae_context'][i]:.6g},{delta[i]:.6g},"
                f"{perfeat['sae_prefix']['p95'][i]:.6g},"
                f"{perfeat['sae_context']['p95'][i]:.6g},"
                f"{int(carried['sae_prefix'][i])},{int(carried['sae_context'][i])}\n"
            )

    # ── mapping baselines (standing pair) ─────────────────────────────────────
    C.phase("sae-baselines")
    baselines: dict = {"ks": [1, 5, 10], "metrics": ["euclidean", "cosine"], "cells": {}}
    for cell, xsrc, pool in SAE_CELLS:
        if pool != "mean":
            continue
        with np.load(pred_dir / f"{cell}.npz", allow_pickle=False) as z:
            pred = z["pred16"].astype(np.float64)
        cb: dict = {"knn": {}}
        for m in ("euclidean", "cosine"):
            cb["knn"][m] = knn_retrieval(pred, y_ho_mean, ks=(1, 5, 10), metric=m)
        if cell in ARM_X:
            arm = ARM_X[cell]
            shared = np.intersect1d(f_in[arm], f_out)
            xcols = np.searchsorted(f_in[arm], shared)
            ycols = np.searchsorted(f_out, shared)
            x_tr = _rows(X[arm], tr)[:, xcols]
            y_tr = _rows(Y["mean"], tr)[:, ycols]
            x_ho = _rows(X[arm], ho)[:, xcols]
            pred_ib = identity_bias_predict(x_tr, y_tr, x_ho)
            y_sub = y_ho_mean[:, ycols]
            cb["identity_bias"] = {
                "n_shared_ids": int(len(shared)),
                "holdout_r2_shared_subset": _pooled_r2(pred_ib, y_sub),
                "fitted_map_r2_same_subset": _pooled_r2(pred[:, ycols], y_sub),
                "note": "both R2 on the SHARED (input ∩ answer) feature-id subset — "
                "matched-target comparison",
            }
            cb["knn_identity_bias_shared_subset"] = {
                m: knn_retrieval(pred_ib, y_sub, ks=(1, 5, 10), metric=m)
                for m in ("euclidean", "cosine")
            }
        else:
            cb["identity_bias"] = {
                "inapplicable": f"dense input dim {h_dense} != answer-feature dim "
                f"{len(f_out)} — identity baseline undefined (stated, plan §4.1 S2.6)"
            }
        baselines["cells"][cell] = cb
        logger.info("[baseline] %s done", cell)

    # ── lmsys_transfer (group-level OOD, SAE arms) ────────────────────────────
    C.phase("sae-transfer")
    corp = scan["corpus"]
    tr_lm = tr[corp[tr] == "lmsys"]
    val_lm = val[corp[val] == "lmsys"]
    ho_wc = ho[corp[ho] == "wildchat"]
    ho_lm = ho[corp[ho] == "lmsys"]
    transfer: dict = {"control": "lmsys_transfer", "layer": chunk_layer, "cells": {}}
    if len(ho_wc) == 0 or len(tr_lm) == 0 or len(val_lm) == 0:
        transfer["skipped"] = (
            f"empty cells (tr_lm={len(tr_lm)}, val_lm={len(val_lm)}, ho_wc={len(ho_wc)})"
        )
    else:
        for cell in ("sae_prefix", "sae_context"):
            arm = ARM_X[cell]
            fac_lm = _GramFactor(X[arm], tr_lm, dev, args.block)
            kit_lm = _fit_cell(
                fac_lm, Y["mean"], tr_lm, val_lm, np.concatenate([ho_wc, ho_lm]), LAMBDAS
            )
            pred = kit_lm["pred_ho"]
            r2_wc = _pooled_r2(pred[: len(ho_wc)], np.asarray(_rows(Y["mean"], ho_wc), np.float64))
            r2_lm = _pooled_r2(pred[len(ho_wc) :], np.asarray(_rows(Y["mean"], ho_lm), np.float64))
            transfer["cells"][cell] = {
                "n_train_lmsys": int(len(tr_lm)),
                "n_holdout_wildchat": int(len(ho_wc)),
                "n_holdout_lmsys": int(len(ho_lm)),
                "transfer_r2_wildchat_holdout": float(r2_wc),
                "within_r2_lmsys_holdout": float(r2_lm),
                "selected_lambda": kit_lm["selected_lambda"],
            }
            del fac_lm, kit_lm
    summary["lmsys_transfer"] = transfer

    # pilot meta recap (staged copy into out_eval, plan §6.5)
    if not args.local_sae_dir:
        try:
            hub.stage_hub_file(
                C.HF_DATA_REPO,
                f"{sae_prefix}/{PILOT_META_NAME}",
                out_eval / PILOT_META_NAME,
                repo_type="dataset",
                overwrite=True,
            )
        except Exception:
            logger.warning("[sae-fits] pilot meta not staged from Hub (absent?)")
    elif (Path(args.local_sae_dir).parent / PILOT_META_NAME).exists():
        (out_eval / PILOT_META_NAME).write_text(
            (Path(args.local_sae_dir).parent / PILOT_META_NAME).read_text()
        )

    GG.N1M._atomic_write_json(out_eval / "sae_fits.json", summary)
    GG.N1M._atomic_write_json(out_eval / "mapping_baselines.json", baselines)
    logger.info("[sae-fits] summary + baselines written to %s", out_eval)

    if not args.no_upload:
        C.phase("sae-upload")
        summaries = [
            str(p.relative_to(out_eval))
            for p in (
                out_eval / "sae_fits.json",
                out_eval / "mapping_baselines.json",
                out_eval / "perfeature_summary.csv",
                out_eval / PILOT_META_NAME,
                out_eval / "fence_report.json",
                *sorted(cells_dir.glob("*.json")),
            )
            if p.is_file()
        ]
        _upload_sae_analysis(
            args,
            sae_prefix,
            [
                ("summaries", out_eval, summaries),
                ("perfeature", pf_dir, None),
                ("pred16", pred_dir, None),
            ],
        )
    C.phase("done")
    return 0


# ── smoke: tiny-real CPU e2e through the SAME entrypoints (PASS_UNIFIED) ────────────

SMOKE_LAYER = 1  # tiny 2-layer model: capture block index 1
SMOKE_DICT = 256
SMOKE_K = 8
SMOKE_H = 64


def _smoke_models(model_dir: Path, args, model: bool = True):
    """Load the smoke fixture: REAL Qwen tokenizer (real vocab + chat template),
    from-config tiny same-arch Qwen2, from-config tiny BatchTopK SAE (real
    loader class, real encode/pool/fve code paths)."""
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(str(model_dir / "tok"))
    sd = torch.load(model_dir / "sae_ae.pt", map_location="cpu", weights_only=True)
    sae = SAEMOD.BatchTopKSAE(
        sd,
        k=SMOKE_K,
        device="cpu",
        act_dim=int(sd["encoder.weight"].shape[1]),
        dict_size=int(sd["encoder.weight"].shape[0]),
    )
    hf = None
    if model:
        hf = Qwen2ForCausalLM.from_pretrained(str(model_dir / "model"), torch_dtype=torch.float32)
        hf.eval()
    return hf, tok, sae


def _build_smoke_fixture(base: Path, n_rows: int = 24, n_chunks: int = 2) -> dict:
    """Materialize the smoke fixture: tiny model/tokenizer/SAE + synthetic parent
    chunks whose STORED v_x/px_last/cx_last are computed by the SAME render +
    capture path the production driver runs (so the identity gate is a REAL
    check, not a tautology on fabricated numbers)."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    model_dir = base / "models"
    (model_dir / "tok").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(GG.DEFAULT_MODEL)
    tok.save_pretrained(str(model_dir / "tok"))
    torch.manual_seed(1738)
    cfg = Qwen2Config(
        vocab_size=len(tok) + 128,
        hidden_size=SMOKE_H,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    hf = Qwen2ForCausalLM(cfg)
    hf.eval()
    hf.save_pretrained(str(model_dir / "model"))
    sd = {
        "b_dec": torch.zeros(SMOKE_H),
        "k": torch.tensor(SMOKE_K, dtype=torch.int32),
        "threshold": torch.tensor(0.05),
        "decoder.weight": torch.randn(SMOKE_H, SMOKE_DICT) * 0.1,
        "encoder.weight": torch.randn(SMOKE_DICT, SMOKE_H) * 0.5,
        "encoder.bias": torch.zeros(SMOKE_DICT),
    }
    torch.save(sd, model_dir / "sae_ae.pt")

    parent_dir = base / "parent_chunks"
    parent_dir.mkdir(parents=True, exist_ok=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    per = n_rows // n_chunks
    ci0 = 0
    for cidx in range(n_chunks):
        rows = []
        for j in range(per):
            i = ci0 + j
            messages = [
                {"role": "user", "content": f"Tell me about topic {i}."},
                {"role": "assistant", "content": "It is a broad topic."},
                {"role": "user", "content": f"Give me {1 + i % 3} more details."},
            ]
            response = " ".join(f"Detail {i}-{t}." for t in range(1 + i % 4))
            ids, plen = _render_row(tok, messages, response)
            span = _capture_answer_spans(hf, [ids], [plen], SMOKE_LAYER, 1, pad_id)[0]
            cap = extract_layer_activations(hf, ids[None], [SMOKE_LAYER])
            hs = cap[SMOKE_LAYER][0]
            rows.append(
                {
                    "ci": i,
                    "messages": messages,
                    "response": response,
                    "depth": 2,
                    "corpus": "lmsys" if i % 3 else "wildchat",
                    "px_last": hs[max(0, plen // 2)].float().unsqueeze(0),
                    "cx_last": hs[plen - 1].float().unsqueeze(0),
                    "v_x": span.mean(0).unsqueeze(0),
                }
            )
        chunk = GG._stack_chunk_mt(rows, [SMOKE_LAYER], 0, cidx)
        torch.save(chunk, parent_dir / f"shard00_chunk{cidx:04d}.pt")
        ci0 += per
    return {
        "model_dir": model_dir,
        "parent_dir": parent_dir,
        "n_rows": n_rows,
        "ci": list(range(n_rows)),
    }


def _smoke_split(base: Path, ci: list[int]) -> tuple[Path, Path]:
    """Pinned-split fixture + a parent-fits JSON carrying MATCHING split_shas +
    n_rows_captured (the cross-assert inputs)."""
    n = len(ci)
    sets = {
        "train": ci[: n - 10],
        "val": ci[n - 10 : n - 6],
        "test": ci[n - 6 : n - 4],
        "holdout": ci[n - 4 :],
    }
    doc = {
        "sets": {
            k: {"ci": v, "n": len(v), "sha256": GG._sha_int_list([int(x) for x in v])}
            for k, v in sets.items()
        }
    }
    split_p = base / "split_1738.json"
    split_p.write_text(json.dumps(doc))
    parent_p = base / "parent_fits.json"
    parent_p.write_text(
        json.dumps(
            {
                "split_shas": {k: doc["sets"][k]["sha256"] for k in doc["sets"]},
                "n_rows_captured": n,
            }
        )
    )
    return split_p, parent_p


def _smoke(args) -> int:
    """Tiny-real CPU e2e: REAL run_capture + run_fits at tiny N (smoke IS the
    production path with one-shard/24-row parameterization) + degenerate probes
    for every data-dependent gate branch + signature-bound Hub-boundary fakes."""
    import shutil
    from unittest.mock import create_autospec

    base = Path(args.out_dir) / "_smoke_sae"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fx = _build_smoke_fixture(base)
    logger.info("[smoke] fixture built (%.0fs)", time.time() - t0)

    # ── leg 1: capture (production entrypoint, pilot path incl. fve gate) ─────
    cap_args = argparse.Namespace(
        **{
            **vars(args),
            "phase": "capture",
            "device": "cpu",
            "layer": SMOKE_LAYER,
            "num_shards": 1,
            "shard_index": 0,
            "sae_batch": 4,
            "local_capture_dir": str(fx["parent_dir"]),
            "no_upload": True,
            "out_dir": base / "cap",
            "smoke_model_dir": str(fx["model_dir"]),
            "pilot_rows": 4,
            "pilot_only": False,
            "fve_token_floor": 32,
            "sae_hf_prefix": None,
        }
    )
    rc = run_capture(cap_args)
    assert rc == 0, f"capture smoke rc={rc}"
    sae_chunks = sorted((base / "cap" / "sae_chunks").glob("*.pt"))
    assert len(sae_chunks) == 2, [p.name for p in sae_chunks]
    meta = json.loads((base / "cap" / PILOT_META_NAME).read_text())
    assert "gate_s0" in meta and "gate_s1" in meta, sorted(meta)
    assert set(meta["gate_s0"].get(f"k{K_PRIMARY}_split", {})) == {
        "answer_tokens",
        "context_tokens",
    }, meta["gate_s0"].get(f"k{K_PRIMARY}_split")
    d0 = torch.load(sae_chunks[0], map_location="cpu", weights_only=False)
    for key in (
        "ci",
        "feat_idx",
        "row_ptr",
        "ans_mean",
        "ans_max",
        "ans_frac",
        "px_feat_idx",
        "px_row_ptr",
        "px_feat_val",
        "cx_feat_idx",
        "cx_row_ptr",
        "cx_feat_val",
        "px_dense19",
        "cx_dense19",
        "n_ans_tokens",
        "n_inlier_tokens",
        "vx_cos",
        "dropped_ci",
        "layers",
        "corpus",
        "depth",
    ):
        assert key in d0, f"sae chunk missing {key}"
    assert min(d0["vx_cos"]) >= VX_COS_ROW_MIN, min(d0["vx_cos"])
    logger.info(
        "[smoke] capture leg OK: %d chunks, vx_cos_min=%.6f, gates=%s/%s",
        len(sae_chunks),
        float(min(d0["vx_cos"])),
        meta["gate_s0"].get("pass"),
        meta["gate_s1"].get("pass"),
    )

    # ── leg 2: upload boundary (signature-bound fakes; fail branch too) ───────
    scratch = base / "upload_scratch"
    scratch.mkdir()
    shutil.copy(sae_chunks[0], scratch / sae_chunks[0].name)
    name = sae_chunks[0].name
    sha = N50._sha256_file(scratch / name)
    fake_upload = create_autospec(hub._upload_folder_filtered, return_value="https://ok")
    prefix = "issue1738_multiturn/sae_arm_smokeprobe"

    def fake_index(p: str) -> dict[str, dict]:
        """Signature mirror of N50._remote_index (prefix -> {name: {size, sha256}})."""
        return {name: {"size": 1, "sha256": sha}}

    _flush_sae_chunks(scratch, prefix, [name], upload_fn=fake_upload, remote_index_fn=fake_index)
    assert not (scratch / name).exists(), "purge-after-verify did not fire"
    assert fake_upload.call_count == 1
    kw = fake_upload.call_args.kwargs
    assert kw["path_in_repo"] == f"{prefix}/{GG.CAPTURE_SUBDIR}", kw["path_in_repo"]
    shutil.copy(sae_chunks[0], scratch / name)
    try:
        _flush_sae_chunks(
            scratch,
            prefix,
            [name],
            upload_fn=create_autospec(hub._upload_folder_filtered, return_value=""),
            remote_index_fn=fake_index,
        )
        raise AssertionError("empty-URL upload did not raise")
    except RuntimeError as e:
        assert "returned no URL" in str(e)
    logger.info("[smoke] upload-boundary leg OK (verify+purge, fail-loud URL branch)")

    # ── leg 3: degenerate capture probes (designed gate handling) ─────────────
    poison_dir = base / "parent_poison"
    poison_dir.mkdir()
    dpar = torch.load(
        fx["parent_dir"] / "shard00_chunk0000.pt", map_location="cpu", weights_only=False
    )
    dpar["v_x"][0] = -dpar["v_x"][0]  # break row 0's identity
    torch.save(dpar, poison_dir / "shard00_chunk0000.pt")
    pargs = argparse.Namespace(
        **{
            **vars(cap_args),
            "local_capture_dir": str(poison_dir),
            "out_dir": base / "cap_poison",
            "pilot_rows": 0,
        }
    )
    rc = run_capture(pargs)
    assert rc == 0, rc
    dp = torch.load(
        next((base / "cap_poison" / "sae_chunks").glob("*.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert len(dp["dropped_ci"]) == 1 and dp["dropped_ci"][0] == 0, dp["dropped_ci"]
    ok_s0 = {"pass": True}
    ok_s1 = {
        "violation_rate": 0.0,
        "vx_median_cos": 1.0,
        "parity": {"min_cos": 1.0},
        "projected_shard_wall_min": 1.0,
        "shard_fence_min": 60.0,
    }
    assert _pilot_verdict_rc({"pass": False}, ok_s1, smoke=False) == RC_SAE_FVE
    assert (
        _pilot_verdict_rc(ok_s0, {**ok_s1, "projected_shard_wall_min": 61.0}, smoke=False)
        == RC_SAE_FENCE
    )
    try:
        _pilot_verdict_rc(ok_s0, {**ok_s1, "violation_rate": 0.5}, smoke=False)
        raise AssertionError("identity-fail did not raise")
    except RuntimeError as e:
        assert "identity/parity FAIL" in str(e)
    assert _pilot_verdict_rc({"pass": False}, {**ok_s1, "violation_rate": 0.5}, smoke=True) == 0
    assert MTF._fence_should_halt(100.0, 1.0, 8, 2.0) is True
    try:
        _require_sae_prefix(argparse.Namespace(sae_hf_prefix=None))
        raise AssertionError("missing sae prefix did not raise")
    except RuntimeError as e:
        assert "--sae-hf-prefix" in str(e)
    logger.info("[smoke] degenerate capture probes OK (rc27/rc26/raise/drop/prefix)")

    # ── leg 4: fits (production entrypoint, full battery at tiny N) ───────────
    split_p, parent_p = _smoke_split(base, fx["ci"])
    fit_args = argparse.Namespace(
        **{
            **vars(args),
            "phase": "fits",
            "device": "cpu",
            "local_sae_dir": str(base / "cap" / "sae_chunks"),
            "sae_hf_prefix": None,
            "split_file": str(split_p),
            "parent_fits_json": str(parent_p),
            "out_eval": base / "eval",
            "out_local": base / "local",
            "no_upload": True,
            "no_resume": False,
            "allow_underdetermined": True,
            "k_draws": 3,
            "n_boot": 50,
            "fence_mult": 50.0,
            "block": 64,
        }
    )
    rc = run_fits(fit_args)
    assert rc == 0, f"fits smoke rc={rc}"
    sf = json.loads((base / "eval" / "sae_fits.json").read_text())
    assert len(sf["cells"]) == len(SAE_CELLS), sorted(sf["cells"])
    assert set(sf["shuffle_floor"]) == {"sae_prefix", "sae_context"}
    mb = json.loads((base / "eval" / "mapping_baselines.json").read_text())
    assert "inapplicable" in mb["cells"]["dense_px_feat"]["identity_bias"]
    assert "holdout_r2_shared_subset" in mb["cells"]["sae_prefix"]["identity_bias"]
    csv_rows = (base / "eval" / "perfeature_summary.csv").read_text().count("\n") - 1
    assert csv_rows == sf["restriction"]["n_f_out"], (csv_rows, sf["restriction"])
    # resume leg: re-run must skip every cell (fingerprint-keyed)
    rc = run_fits(fit_args)
    assert rc == 0
    logger.info(
        "[smoke] fits leg OK: %d cells, floor draws=%d, csv rows=%d (+resume)",
        len(sf["cells"]),
        sf["shuffle_floor"]["sae_prefix"]["k_draws"],
        csv_rows,
    )

    # ── leg 5: degenerate fits probes ─────────────────────────────────────────
    bad_parent = base / "bad_parent.json"
    bad = json.loads(parent_p.read_text())
    bad["split_shas"] = dict.fromkeys(bad["split_shas"], "dead")
    bad_parent.write_text(json.dumps(bad))
    try:
        run_fits(
            argparse.Namespace(
                **{
                    **vars(fit_args),
                    "parent_fits_json": str(bad_parent),
                    "out_eval": base / "eval2",
                    "out_local": base / "local2",
                }
            )
        )
        raise AssertionError("parent split_shas mismatch did not fire")
    except AssertionError as e:
        assert "split_shas" in str(e), e
    bad_count = base / "bad_count.json"
    bc = json.loads(parent_p.read_text())
    bc["n_rows_captured"] = 999
    bad_count.write_text(json.dumps(bc))
    try:
        run_fits(
            argparse.Namespace(
                **{
                    **vars(fit_args),
                    "parent_fits_json": str(bad_count),
                    "out_eval": base / "eval3",
                    "out_local": base / "local3",
                }
            )
        )
        raise AssertionError("coverage-count mismatch did not fire")
    except AssertionError as e:
        assert "parent captured" in str(e), e
    try:
        run_fits(
            argparse.Namespace(
                **{
                    **vars(fit_args),
                    "allow_underdetermined": False,
                    "out_eval": base / "eval4",
                    "out_local": base / "local4",
                }
            )
        )
        raise AssertionError("underdetermined guard did not fire")
    except SystemExit as e:
        assert "estimator-degenerate" in str(e), e
    logger.info("[smoke] degenerate fits probes OK (shas/coverage/underdetermined)")
    print(
        f"[smoke] PASS in {time.time() - t0:.0f}s — artifacts under {base} "
        f"(cells={len(sf['cells'])}, F_out={sf['restriction']['n_f_out']}, "
        f"vx_cos_min={float(min(d0['vx_cos'])):.6f})"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=["capture", "fits"], default="")
    ap.add_argument("--smoke", action="store_true", help="tiny-real CPU e2e (production paths)")
    ap.add_argument("--model", default=GG.DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--sae-batch", type=int, default=SAE_BATCH_DEFAULT)
    ap.add_argument("--hf-prefix", default=GG.HF_PREFIX, help="parent store prefix (READ side)")
    ap.add_argument(
        "--sae-hf-prefix",
        default=None,
        help="sae-arm store prefix (REQUIRED for Hub I/O; no default by design — "
        "pass issue1738_multiturn/sae_arm)",
    )
    ap.add_argument("--local-capture-dir", default="", help="read parent chunks locally (smoke)")
    ap.add_argument("--local-sae-dir", default="", help="read sae chunks locally (smoke)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_LOCAL / "capture")
    ap.add_argument("--sae-cache", default=str(PROJECT_ROOT / "data" / "issue_1738" / "sae_cache"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--pilot-rows", type=int, default=0, help="G-S0/G-S1 pilot row floor")
    ap.add_argument("--pilot-only", action="store_true", help="exit after the pilot gates")
    ap.add_argument("--fve-token-floor", type=int, default=FVE_TOKEN_FLOOR)
    ap.add_argument("--vx-cos-row-min", type=float, default=VX_COS_ROW_MIN)
    ap.add_argument("--shard-fence-min", type=float, default=SHARD_FENCE_MIN)
    ap.add_argument("--smoke-model-dir", default="", help="(internal) smoke fixture dir")
    ap.add_argument("--split-file", default="", help="split_1738.json (default: stage from HF)")
    ap.add_argument("--parent-fits-json", default=str(DEFAULT_PARENT_FITS))
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT_EVAL)
    ap.add_argument("--out-local", type=Path, default=DEFAULT_OUT_LOCAL)
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--fence-mult", type=float, default=2.0, help="G-S2 first-cell fence")
    ap.add_argument("--block", type=int, default=8192)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--allow-underdetermined", action="store_true", help="smoke shape: n_tr < d")
    args = ap.parse_args()
    if args.smoke:
        return _smoke(args)
    if args.phase == "capture":
        return run_capture(args)
    if args.phase == "fits":
        return run_fits(args)
    raise SystemExit("pass --phase {capture,fits} or --smoke")


if __name__ == "__main__":
    rc = main()
    # explicit exit BEFORE finalize-time C-extension teardown (PyGILState atexit
    # race — gotchas: phased entrypoints importing torch/scipy exit explicitly).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
