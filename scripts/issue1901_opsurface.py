#!/usr/bin/env python3
"""Issue #1901 inline round ``opsurface-rebase`` — both paper figures on the #2202 surface.

User order (2026-08-25, binding decision record: the ``epm:progress`` dispatch note on
task #1901): re-score BOTH paper figures (``c1_layer_profile``, ``c1_scaling_train_pool``)
on the #2202 operating-point eval surface — the 1,988 draw-covered #1738 multiturn-holdout
queries ranked against the full 9,941-answer holdout pool — with whitened cosine + CSLS
(K=10) retrieval on DRAW-AVERAGED targets, and draw-averaged R^2 on the covered rows.
Coverage stays 1,988/9,941 (no expansion); the covered-subset caveat goes in paper TEXT,
never a figure caption/label. Maps stay the existing recipes: Plot 1 = the per-layer
50k-capture maps (ridge / identity_bias / mlp_w8192, per-layer whitening from the 50k
train answers); Plot 2 = L19 rungs 50 -> 963,444 refit from the 50k + n1m captures with
PER-RUNG train-answer whitening, plus NEW dense-MLP mid-rungs 100k/250k/500k and
ridge+identity_bias at 100k/250k. The boundary-token arm stays BANKED (R^2 panel only).

Conventions (verbatim reuse, no re-derivation):
- whitening + CSLS + mid-ranks: ``issue1901_plot1_remake`` (``whiten`` /
  ``train_whitening_stats`` (shrunk-Cholesky lam=0.1) / ``cos_sim`` / ``midranks``) +
  ``issue1901_metric_battery.csls_scores`` (cross-domain CSLS, K=10; distance = -score).
  CSLS neighborhoods come from the realized (1,988 x 9,941) cross-domain matrix: the
  pool-side r_p is the mean top-K similarity over the 1,988 COVERED queries (recorded in
  CONVENTIONS; #2202's full-pool read used all 9,941 queries).
- draw-averaged target: ``issue2202_freshwhiten_avg`` — pool entry of each covered row
  replaced by mean(original + K=4 fresh draws); pool size stays 9,941; queries = the
  map's predictions at the covered rows.
- eval-side capture: teacher-forced forwards over the 9,941 pool rows + 7,952 banked
  kresample draw completions, hooks at ALL 28 layers, ONE forward per row (cx_last read
  at prompt_len-1 under a strict-token-prefix gate — causally identical to the banked
  context-only forward; v = mean over answer tokens at the ``issue779_collect
  .capture_answer_vector`` slice convention). Batched right-padded forwards with a
  batch-1-vs-batched equivalence gate at phase entry (#779 two-bar calibration).
- fits: ``issue779_ffc_n1m_fits`` cores (``fit_ridge_with_weights`` / ``fit_mlp`` with
  ``capture_out`` / ``apply_map``) + ``mapping_baselines.identity_bias_predict`` — all
  vectorized closed-form / minibatched; no new estimator code.

Refusal-safety: the multiturn texts are real-user-corpus (LMSYS/WildChat-class) rows —
this script never prints or logs conversation/rollout text; only counts, ci ids,
digests, and vector statistics.

Smoke blind-spot enumeration (--smoke; see .claude/rules/smoke-blind-spots.md):
- SUBSTITUTED: phase capture28's real Qwen-2.5-7B forwards — --smoke synthesizes the
  cx/vx/draw tensors, so the batched-capture path (padding, strict-prefix gate, span
  means) is NOT executed under --smoke; its coverage = the CPU tiny-model equivalence
  test in tests/test_issue1901_opsurface.py + the in-phase batch-1-vs-batched
  equivalence gate at production entry (GPU).
- SKIPPED: HF staging / upload / verify branches (network) — unreached under --smoke.
- UNREACHED (not weakened): the banked parity gates — plot-1 refit R^2 vs
  BANKED_CHAT_JSON (hard, ridge/identity), fits2B single-target euclid acc@1 vs the
  banked bign cells (hard), and the leg-0 freshwhiten reconciliation — need banked
  inputs absent at smoke; production thresholds are byte-untouched.
- SUBSTITUTED (scale knobs only, same code path): mlp width 8192->8, epochs->2,
  n_boot->50, csls K->3, H->16, layers->(0,1).
- production-only third-party imports: none (torch / numpy / matplotlib / scipy are all
  exercised by --smoke).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps bind BEFORE numpy/torch import (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue1738_multiturn_generate_capture as GG  # noqa: E402
import issue1901_avgtarget_plots as AVG  # noqa: E402
import issue1901_paper_densify as PD  # noqa: E402
import issue1901_plot1_remake as P1R  # noqa: E402
from issue1901_metric_battery import K_CSLS, csls_scores  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1901_opsurface")

ISSUE = 1901
ROUND = "opsurface-rebase"
HF_ROUND_PREFIX = "issue1901_opsurface"
MODEL_ID = AVG.MODEL_ID  # Qwen/Qwen2.5-7B-Instruct
H_DIM = C.EXPECTED_HIDDEN  # 3584
LAYERS_ALL = tuple(range(28))
L19 = 19
CHAT_ARMS = AVG.CHAT_ARMS  # ("ridge", "identity_bias", "mlp_w8192")

# #2202 surface pins (dispatch record + eval_results/issue_2202/freshwhiten_avg staging).
PARENT_PREFIX = "issue1738_multiturn"
HF_PIN_1738 = "09788eef2f85330c6f9c6b7cd3d28cb47cfb8429"  # data-repo pin, #2202 plan §10
HF_REV_2202 = "80d455b2e3cd516eccb439b12fbba9b738f4cf8c"  # issue2202_ctxfail tensors revision
HF_PREFIX_2202 = "issue2202_ctxfail"
N_POOL = 9_941
N_COVERED = 1_988
K_DRAWS = 4
DRAW_SEEDS = (43, 44, 45, 46)  # the #1738 kresample per-request seeds
RAW_SUBDIR = GG.RAW_SUBDIR  # "raw_completions" (#1738 chunk JSONs: ci/messages/response)
KRESAMPLE_SUBDIR = GG.KRESAMPLE_SUBDIR

OUT_EVAL_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_1901" / "opsurface"
FIG_DIR_DEFAULT = PROJECT_ROOT / "figures" / "issue_1901" / "opsurface"
PAPER_FIG_DIR = PROJECT_ROOT / "figures" / "paper"
FRESHWHITEN_SUMMARY = PROJECT_ROOT / "eval_results" / "issue_2202" / "freshwhiten_avg"
FRESHWHITEN_SUMMARY = FRESHWHITEN_SUMMARY / "summary.json"

# Plot-2 rung grids (dispatch record §5: existing rungs 50->963k from the 50k + n1m
# captures; NEW mlp 100k/250k/500k; NEW ridge+identity 100k/250k).
SMALL_RUNG_NS = (50, 100, 250, 500, 1000, 2500)  # 3 seeded draws each
SMALL_RUNG_DRAWS = (0, 1, 2)
MID_RUNG_NS = (5_000, 10_000, 15_000, 20_000, 25_000)  # single draw
MLP_SMALL_NS = (5_000, 10_000, 25_000)
BIG_RUNGS = (  # (tag, n_target, refit_mlp)
    ("lmsys_100k", 100_000, True),
    ("lmsys_150k", 150_000, False),
    ("lmsys_250k", 250_000, True),
    ("lmsys_500k", 500_000, True),
)
N_963K = 963_444

CONVENTIONS = {
    "surface": (
        "the #2202 operating point: 1,988 draw-covered #1738 multiturn-holdout queries "
        "ranked against the full 9,941-answer holdout pool"
    ),
    "draw_averaged_target": (
        "pool entry of each covered row replaced by mean(original answer state + K=4 "
        "fresh on-policy draws); pool size stays 9,941; queries = map predictions at "
        "the covered rows (issue2202_freshwhiten_avg leg D, per layer here)"
    ),
    "k_draws": K_DRAWS,
    "draw_seeds": list(DRAW_SEEDS),
    "draw_text_source": "banked #1738 kresample_shard*.json (per-request seeds 43-46)",
    "metric": (
        "whitened cosine + CSLS (K=10): z = L^-1 (x - mu_A) with per-unit TRAIN-answer "
        "shrunk-covariance Cholesky (issue1901_plot1_remake.train_whitening_stats, "
        "lam=0.1 — plot 1: the 50k train answers per layer; plot 2: each rung's own "
        "train answers); CSLS = issue1901 csls_scores on the whitened cross-domain "
        "(1,988 x 9,941) cosine matrix — r_query over the pool, r_pool over the 1,988 "
        "covered queries; distance = -score; mid-ranks, tie at top counts as failure"
    ),
    "r2": (
        "pooled held-out R^2 of predictions vs draw-averaged targets over the 1,988 "
        "covered rows (issue779_fitter_fair_comparison._recon_point + bootstrap CI)"
    ),
    "capture": (
        "teacher-forced batched right-padded forwards, hooks at all 28 layers, one "
        "forward per row: cx_last at prompt_len-1 under a strict-token-prefix gate "
        "(context-only fallback forward on gate miss), v = mean over answer tokens "
        "(issue779_collect.capture_answer_vector slice convention); bf16 model, fp32 "
        "reduce, fp16 store"
    ),
    "boundary_arm": "BANKED (deterministic WikiText target; R^2 panel only, unchanged)",
    "rank": "mid-rank with 1e-9 relative tie tolerance; tie at top counts as failure",
}


# ── regime key (resume predicates key on EVERY output-affecting flag) ────────────


def regime_key(args) -> dict:
    """Every output-affecting regime constant/flag; unit resume predicates embed it."""
    key = {
        "round": ROUND,
        "surface": "issue2202-operating-point",
        "model_id": MODEL_ID,
        "parent_prefix": PARENT_PREFIX,
        "parent_pin": HF_PIN_1738,
        "issue2202_tensors_revision": HF_REV_2202,
        "n_pool": N_POOL,
        "n_covered": N_COVERED,
        "k_draws": K_DRAWS,
        "draw_seeds": list(DRAW_SEEDS),
        "metric": "whiten_cos+csls",
        "csls_k": int(args.csls_k),
        "whiten_lambda": float(P1R.PRIMARY_LAMBDA),
        "target": "draw_averaged",
        "layers": list(args.layers),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "capture_convention": "one-forward cx@prompt_len-1 + v=mean(answer span); fp16",
    }
    if args.smoke:
        key["smoke"] = {"h_dim": args.smoke_h, "n_pool": args.smoke_pool, "n_cov": args.smoke_cov}
    return key


def _meta(phase: str) -> dict:
    md = as_metadata_dict(git_provenance(argv0=__file__), phase=phase)
    md.update(
        {
            "script": "issue1901_opsurface",
            "issue": ISSUE,
            "round": ROUND,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return md


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path, logger=logger) as tmp:
        tmp.write_text(json.dumps(obj, indent=1, default=str))


def _upload_eval(args, *, force: bool = False) -> None:
    if args.skip_upload or args.smoke:
        return
    _upload_eval.counter = getattr(_upload_eval, "counter", 0) + 1
    if force or _upload_eval.counter % args.upload_every == 0:
        url = hub._upload(
            args.out_eval,
            C.HF_DATA_REPO,
            "dataset",
            path_in_repo=f"{HF_ROUND_PREFIX}/eval",
            raise_on_error=True,
        )
        logger.info("[upload] eval dir -> %s", url)


def _upload_part(local: Path, repo_rel: str) -> None:
    """Per-part incremental checkpoint upload (crash durability for capture shards)."""
    url = hub._upload(
        local,
        C.HF_DATA_REPO,
        "dataset",
        path_in_repo=f"{HF_ROUND_PREFIX}/{repo_rel}",
        upload_as_file=True,
        raise_on_error=True,
    )
    if not url:
        raise RuntimeError(f"_upload returned no path for {local} -> {HF_ROUND_PREFIX}/{repo_rel}")
    logger.info("[upload] %s -> %s/%s", local.name, HF_ROUND_PREFIX, repo_rel)


# ── scoring core (the operating-point surface) ───────────────────────────────────


def avg_pool_assembly(
    y_pool: np.ndarray, v_draws: np.ndarray, pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(pool_mod, y_avg_cov): covered pool entries replaced by mean(original + draws).

    y_pool (n_pool, H); v_draws (n_cov, K, H) — all draws present (the kresample
    producer kept only contexts where every seed yielded a non-empty answer);
    pos (n_cov,) covered-row positions into the pool. fp64 mean, fp32 return."""
    y_pool = np.asarray(y_pool, dtype=np.float64)
    v_draws = np.asarray(v_draws, dtype=np.float64)
    assert v_draws.ndim == 3 and v_draws.shape[0] == len(pos), (v_draws.shape, len(pos))
    avg = (y_pool[pos] + v_draws.sum(axis=1)) / (1.0 + v_draws.shape[1])
    pool_mod = y_pool.copy()
    pool_mod[pos] = avg
    return pool_mod.astype(np.float32), avg.astype(np.float32)


def _retrieval_block(ranks: np.ndarray, n_pool: int, n_boot: int, seed: int, name: str) -> dict:
    rng = np.random.default_rng(seed + 7)  # the PD.score_cell boot-draw convention
    n = ranks.shape[0]
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    draws = (ranks[boot_idx] <= 1).mean(axis=1)
    return {
        "metric": name,
        "n": int(n),
        "n_pool": int(n_pool),
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in (1, 5, 10)},
        "chance_at_k": {int(k): float(k / n_pool) for k in (1, 5, 10)},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "acc1_ci": {
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
        },
    }


def score_opsurf(
    pred_cov: np.ndarray,
    pool_mod: np.ndarray,
    y_avg_cov: np.ndarray,
    pos: np.ndarray,
    mu: np.ndarray,
    ell: np.ndarray,
    n_boot: int,
    seed: int,
    csls_k: int,
    *,
    include_raw_euclidean: bool = False,
) -> dict:
    """Draw-averaged R^2 (+CI) on the covered rows + whitened-cosine / whitened-CSLS
    retrieval of each covered query's own (draw-averaged) pool entry among n_pool."""
    pred_cov = np.asarray(pred_cov, dtype=np.float64)
    pool_mod = np.asarray(pool_mod, dtype=np.float64)
    n_pool = pool_mod.shape[0]
    assert pred_cov.shape == (len(pos), pool_mod.shape[1]), pred_cov.shape
    r2, cos = F79._recon_point(pred_cov, y_avg_cov)
    ci = F79._bootstrap_recon_ci(pred_cov, y_avg_cov, n_boot, seed)
    out = {
        "whole_map_r2": float(r2),
        "mean_cosine": float(cos),
        "bootstrap_ci": ci,
        "retrieval": {},
    }
    zq = P1R.whiten(pred_cov, mu, ell)
    zp = P1R.whiten(pool_mod, mu, ell)
    s = P1R.cos_sim(zq, zp)  # (n_cov, n_pool)
    assert csls_k <= zq.shape[0] and csls_k < n_pool, (csls_k, s.shape)
    for name, dist in (
        ("whiten_csls", -csls_scores(s, csls_k)),
        ("whiten_cos", 1.0 - s),
    ):
        ranks = P1R.midranks(dist, np.asarray(pos))
        out["retrieval"][name] = _retrieval_block(ranks, n_pool, n_boot, seed, name)
    if include_raw_euclidean:
        d = MB._pairwise_dist(pred_cov, pool_mod, "euclidean")
        ranks = P1R.midranks(d, np.asarray(pos))
        out["retrieval"]["raw_euclidean"] = _retrieval_block(
            ranks, n_pool, n_boot, seed, "raw_euclidean"
        )
    return out


# ── phase stage: banked #2202 tensors + draw texts + holdout texts ───────────────


def _t2202_dir(args) -> Path:
    return args.stage_root / "t2202"


def phase_stage(args) -> dict:
    """Stage the banked eval-side inputs: #2202 tensors at their pins, the kresample
    draw texts, and the 9,941 holdout rows' context+answer texts (chunk-checkpointed)."""
    C.phase("stage")
    from huggingface_hub import HfApi, hf_hub_download

    t2202 = _t2202_dir(args)
    t2202.mkdir(parents=True, exist_ok=True)
    for rel, dest, rev in (
        (
            f"{PARENT_PREFIX}/analysis_tensors/pred16/context_L19_ridge.npz",
            "pred16.npz",
            HF_PIN_1738,
        ),
        (
            f"{PARENT_PREFIX}/analysis_tensors/y_holdout/L{L19}.npz",
            "y_holdout_L19.npz",
            HF_PIN_1738,
        ),
        (f"{HF_PREFIX_2202}/analysis_tensors/whiten_stats.npz", "whiten_stats.npz", HF_REV_2202),
    ):
        hub.stage_hub_file(C.HF_DATA_REPO, rel, t2202 / dest, revision=rev)
    api = HfApi()
    kres_files = hub.list_hf_files_under_path(
        api,
        C.HF_DATA_REPO,
        f"{PARENT_PREFIX}/{KRESAMPLE_SUBDIR}",
        repo_type="dataset",
        revision=HF_PIN_1738,
    )
    kres_keep = [f for f in kres_files if f.endswith((".pt", ".json"))]
    assert kres_keep, f"no kresample files under {PARENT_PREFIX}/{KRESAMPLE_SUBDIR}"
    for f in kres_keep:
        hub.stage_hub_file(
            C.HF_DATA_REPO, f, t2202 / "kresample" / Path(f).name, revision=HF_PIN_1738
        )

    # Holdout texts: stream the #1738 raw-completion chunks at the pin, keeping the
    # 9,941 pool rows' {ci, messages, response}. Per-chunk checkpointed; text never
    # logged (real-user corpus — digest-only hygiene).
    pci = np.asarray(np.load(t2202 / "y_holdout_L19.npz")["ci"], dtype=np.int64)
    assert len(pci) == N_POOL, len(pci)
    needed = {int(c) for c in pci}
    texts_path = t2202 / "holdout_texts.jsonl"
    done_p = t2202 / "holdout_texts.done.json"
    found: dict[int, dict] = {}
    done_chunks: set[str] = set()
    if texts_path.exists() and done_p.exists():
        with texts_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    found[int(row["ci"])] = row
        done_chunks = set(json.loads(done_p.read_text()))
        logger.info("[stage] texts resume: %d rows, %d chunks done", len(found), len(done_chunks))
    names = sorted(
        n
        for n in hub.list_hf_files_under_path(
            api,
            C.HF_DATA_REPO,
            f"{PARENT_PREFIX}/{RAW_SUBDIR}",
            repo_type="dataset",
            revision=HF_PIN_1738,
        )
        if n.endswith(".json") and "_chunk" in n
    )
    dl = t2202 / "raw_dl"
    dl.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, name in enumerate(names):
        if len(found) >= len(needed):
            break
        base = Path(name).name
        if base in done_chunks:
            continue
        local = Path(
            hub.retry_transient(
                lambda n=name: hf_hub_download(
                    C.HF_DATA_REPO,
                    filename=n,
                    repo_type="dataset",
                    revision=HF_PIN_1738,
                    cache_dir=str(dl),
                ),
                what=f"raw chunk {base}",
            )
        )
        doc = json.loads(local.read_text())
        new_rows = []
        for r in doc["rows"]:
            ci = int(r["ci"])
            if ci in needed and ci not in found:
                row = {"ci": ci, "messages": r["messages"], "response": r["response"]}
                found[ci] = row
                new_rows.append(row)
        with texts_path.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done_chunks.add(base)
        C.write_json_atomic(done_p, sorted(done_chunks))
        print(
            f"[stage] unit {k + 1}/{len(names)} chunk={base} rows={len(found)}/{len(needed)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    missing = needed - set(found)
    assert not missing, (
        f"{len(missing)} holdout cis missing from raw chunks (e.g. {sorted(missing)[:5]})"
    )
    if dl.exists():
        shutil.rmtree(dl)  # consumed download cache; texts.jsonl is the durable product
    out = {"n_texts": len(found), "n_chunks_scanned": len(done_chunks), "metadata": _meta("stage")}
    _write_json_atomic(args.out_eval / "stage_summary.json", out)
    return out


def _load_kresample(args) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(kci (1988,), V_banked (1988, K, n_stored_layers, H) fp32, draw text rows)."""
    kdir = _t2202_dir(args) / "kresample"
    pts = sorted(kdir.glob("kresample_shard*.pt"))
    assert pts, f"no kresample .pt shards under {kdir} — run --phase stage"
    cis, vs = [], []
    layers_stored = None
    for p in pts:
        b = torch.load(p, map_location="cpu", weights_only=False)
        cis.extend(int(x) for x in b["ci"])
        vs.append(b["V"].to(torch.float32).numpy())
        assert [int(s) for s in b["seeds"]] == list(DRAW_SEEDS), b["seeds"]
        layers_stored = [int(x) for x in b["layers"]]
    kci = np.asarray(cis, dtype=np.int64)
    v = np.concatenate(vs)
    assert v.shape[0] == len(kci) == N_COVERED, (v.shape, len(kci))
    rows: list[dict] = []
    for p in sorted(kdir.glob("kresample_shard*.json")):
        if p.name.endswith("_skipped.json"):
            continue
        rows.extend(json.loads(p.read_text())["rows"])
    by_ci = {int(r["ci"]): r for r in rows}
    assert set(by_ci) >= set(int(c) for c in kci), "draw-text rows missing for kresample cis"
    ordered = [by_ci[int(c)] for c in kci]
    logger.info("[kresample] %d contexts, stored layers %s", len(kci), layers_stored)
    return kci, v, ordered


# ── phase capture28: batched teacher-forced 28-layer capture ─────────────────────


def _render_row(tok, messages: list[dict], response: str) -> tuple[list[int], int, bool]:
    """(full_ids, prompt_len, strict_prefix_ok) — banked #1738 capture convention.

    prompt_len = len(tokenize(context render with generation prompt)); v-span =
    [prompt_len, len(full_ids)) of the FULL render's tokenization (the
    ``capture_answer_vector`` slice, no assert there); strict_prefix_ok additionally
    checks the context render is a token PREFIX of the full render (licenses reading
    cx_last at prompt_len-1 off the same forward — causally identical to the banked
    context-only forward)."""
    ctx_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_msgs = [*messages, {"role": "assistant", "content": response}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    ctx_ids = tok(ctx_text, padding=False)["input_ids"]
    full_ids = tok(full_text, padding=False)["input_ids"]
    prompt_len = len(ctx_ids)
    assert len(full_ids) > prompt_len, "empty answer span (banked capture admitted this row)"
    ok = full_ids[:prompt_len] == ctx_ids
    return full_ids, prompt_len, ok


def _batched_forward_spans(
    model,
    rows: list[dict],
    layers: list[int],
    device: torch.device,
    pad_id: int,
) -> list[dict]:
    """One right-padded batched forward; per row {cx (L,H), vx (L,H)} fp32 cpu.

    rows: [{ids, prompt_len}]. Right padding keeps absolute positions intact (causal
    mask ⇒ pads cannot influence real positions; RoPE indexes from 0 correctly)."""
    t_max = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), t_max), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), t_max), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["ids"])
        ids[i, :n] = torch.as_tensor(r["ids"], dtype=torch.long)
        mask[i, :n] = 1
    captured = extract_layer_activations(
        model, ids.to(device), layers, attention_mask=mask.to(device)
    )
    out = []
    for i, r in enumerate(rows):
        pl, n = r["prompt_len"], len(r["ids"])
        cx = torch.stack([captured[li][i, pl - 1, :].float().cpu() for li in layers])
        vx = torch.stack([captured[li][i, pl:n, :].float().mean(dim=0).cpu() for li in layers])
        out.append({"cx": cx, "vx": vx})
    del captured
    return out


def _capture_batches(items: list[dict], token_budget: int) -> list[list[int]]:
    """Length-sorted index batches under a padded-token budget (B * T_max <= budget)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        t = len(items[i]["ids"])
        new_max = max(cur_max, t)
        if cur and (len(cur) + 1) * new_max > token_budget:
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = t
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def _equivalence_gate_capture(model, tok, probe: list[dict], layers, device, pad_id) -> dict:
    """Batch-1 vs batched capture on ~8 real rows: two-bar #779 calibration —
    per-layer cosine >= 0.999 over layers 0-3 (single-position cx), flattened >= 0.995
    (cx) / >= 0.999 (span-mean vx)."""
    batched = _batched_forward_spans(model, probe, layers, device, pad_id)
    singles = [_batched_forward_spans(model, [r], layers, device, pad_id)[0] for r in probe]

    def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)

    cx_cos = torch.stack([_cos(b["cx"], s["cx"]) for b, s in zip(batched, singles)])
    vx_cos = torch.stack([_cos(b["vx"], s["vx"]) for b, s in zip(batched, singles)])
    early = float(cx_cos[:, : min(4, len(layers))].min())
    flat_cx = float(cx_cos.min())
    flat_vx = float(vx_cos.min())
    rec = {
        "n_probe": len(probe),
        "cx_early_min": early,
        "cx_flat_min": flat_cx,
        "vx_flat_min": flat_vx,
    }
    assert early >= 0.999 and flat_cx >= 0.995 and flat_vx >= 0.999, (
        f"batched-vs-batch-1 equivalence gate FAILED: {rec}"
    )
    logger.info("[capture28] equivalence gate PASS: %s", rec)
    return rec


def _cap_dir(args) -> Path:
    return args.stage_root / "capture28"


def _cap_key(args) -> dict:
    k = regime_key(args)
    return {kk: k[kk] for kk in ("model_id", "parent_pin", "layers", "capture_convention")}


def phase_capture28(args) -> dict:
    """Teacher-forced 28-layer capture: 9,941 pool rows (cx + v) + 1,988 x 4 draws (v)."""
    C.phase("capture28")
    t2202 = _t2202_dir(args)
    pci = np.asarray(np.load(t2202 / "y_holdout_L19.npz")["ci"], dtype=np.int64)
    texts: dict[int, dict] = {}
    with (t2202 / "holdout_texts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts[int(row["ci"])] = row
    assert set(int(c) for c in pci) <= set(texts), "holdout texts incomplete — rerun --phase stage"
    kci, v_banked, draw_rows = _load_kresample(args)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
    )
    model.eval()
    device = torch.device("cuda:0")
    layers = list(args.layers)
    cdir = _cap_dir(args)
    cdir.mkdir(parents=True, exist_ok=True)
    key = _cap_key(args)

    def _tokenize_jobs(job_rows: list[dict]) -> tuple[list[dict], int]:
        items, n_fallback = [], 0
        for jr in job_rows:
            ids, pl, ok = _render_row(tok, jr["messages"], jr["response"])
            if not ok:
                n_fallback += 1
            items.append({"key": jr["key"], "ids": ids, "prompt_len": pl, "strict_ok": ok})
        return items, n_fallback

    def _run_jobs(items: list[dict], want_cx: bool, phase_tag: str, *, model=model) -> dict:
        """Batched capture -> {key: {cx?, vx}}. Strict-prefix misses get a context-only
        fallback forward for cx (the banked GG convention); vx always off the full
        forward (the banked COL convention)."""
        results: dict = {}
        batches = _capture_batches(items, args.capture_token_budget)
        t0 = time.time()
        n_done = 0
        for bi, batch in enumerate(batches):
            rows = [items[i] for i in batch]
            outs = _batched_forward_spans(model, rows, layers, device, pad_id)
            for r, o in zip(rows, outs):
                rec = {"vx": o["vx"].to(torch.float16)}
                if want_cx:
                    rec["cx"] = o["cx"].to(torch.float16)
                    rec["strict_ok"] = bool(r["strict_ok"])
                results[r["key"]] = rec
            n_done += len(rows)
            print(
                f"[capture28] unit {bi + 1}/{len(batches)} {phase_tag} rows={n_done}/"
                f"{len(items)} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if want_cx:
            # cx fallback: context-only forwards for strict-prefix misses (banked
            # GG._capture_context_and_prefix convention — last position of the
            # context-only render).
            misses = [i for i in items if not i["strict_ok"]]
            for k0 in range(0, len(misses), 64):
                blk = misses[k0 : k0 + 64]
                sub = [
                    {"ids": i["ids"][: i["prompt_len"]], "prompt_len": i["prompt_len"]} for i in blk
                ]
                outs = _batched_forward_spans(model, sub, layers, device, pad_id)
                for i, o in zip(blk, outs):
                    results[i["key"]]["cx"] = o["cx"].to(torch.float16)
            if misses:
                logger.info(
                    "[capture28] %d/%d cx rows via context-only fallback", len(misses), len(items)
                )
        return results

    # POOL rows (chunked + resumable at part grain).
    flush = int(args.capture_flush_every)
    n_parts_pool = (N_POOL + flush - 1) // flush
    gate_rec = None
    for part in range(n_parts_pool):
        ppath = cdir / f"pool_part{part:03d}.pt"
        if ppath.exists():
            b = torch.load(ppath, map_location="cpu", weights_only=False)
            if b.get("regime") == key:
                logger.info("[capture28] pool part %d/%d resume-skip", part + 1, n_parts_pool)
                continue
        cis = [int(c) for c in pci[part * flush : (part + 1) * flush]]
        jobs = [
            {"key": ci, "messages": texts[ci]["messages"], "response": texts[ci]["response"]}
            for ci in cis
        ]
        items, n_fb = _tokenize_jobs(jobs)
        if gate_rec is None:
            gate_rec = _equivalence_gate_capture(model, tok, items[:8], layers, device, pad_id)
        res = _run_jobs(items, want_cx=True, phase_tag=f"pool_part{part:03d}")
        store = {
            "regime": key,
            "part": part,
            "ci": torch.as_tensor(cis, dtype=torch.int64),
            "cx": torch.stack([res[ci]["cx"] for ci in cis]),
            "vx": torch.stack([res[ci]["vx"] for ci in cis]),
            "strict_ok": torch.as_tensor([res[ci]["strict_ok"] for ci in cis]),
            "n_prefix_fallback": n_fb,
        }
        with atomic_replace(ppath, logger=logger) as tmp:
            torch.save(store, tmp)
        if not args.skip_upload:
            # UPLOAD_LOOP_EXEMPT: per-part incremental checkpoint persist (crash
            # durability); ~20-40 parts over hours, not a bulk tree walk.
            _upload_part(ppath, f"analysis_tensors/capture28/{ppath.name}")

    # DRAW rows (1,988 x 4 seeds, kci-major).
    draw_jobs_all = []
    for j, (ci, row) in enumerate(zip(kci, draw_rows)):
        assert int(row["ci"]) == int(ci), (row["ci"], ci)
        for s in DRAW_SEEDS:
            draw_jobs_all.append(
                {
                    "key": (int(ci), int(s)),
                    "messages": row["messages"],
                    "response": row["responses"][str(s)],
                }
            )
    n_parts_dr = (len(draw_jobs_all) + flush - 1) // flush
    for part in range(n_parts_dr):
        dpath = cdir / f"draws_part{part:03d}.pt"
        if dpath.exists():
            b = torch.load(dpath, map_location="cpu", weights_only=False)
            if b.get("regime") == key:
                logger.info("[capture28] draws part %d/%d resume-skip", part + 1, n_parts_dr)
                continue
        jobs = draw_jobs_all[part * flush : (part + 1) * flush]
        items, _ = _tokenize_jobs(jobs)
        res = _run_jobs(items, want_cx=False, phase_tag=f"draws_part{part:03d}")
        keys = [j["key"] for j in jobs]
        store = {
            "regime": key,
            "part": part,
            "ci": torch.as_tensor([k[0] for k in keys], dtype=torch.int64),
            "seed": torch.as_tensor([k[1] for k in keys], dtype=torch.int64),
            "vd": torch.stack([res[k]["vx"] for k in keys]),
        }
        with atomic_replace(dpath, logger=logger) as tmp:
            torch.save(store, tmp)
        if not args.skip_upload:
            # UPLOAD_LOOP_EXEMPT: per-part incremental checkpoint persist (see above).
            _upload_part(dpath, f"analysis_tensors/capture28/{dpath.name}")

    del model
    torch.cuda.empty_cache()

    # Banked parity (informational + a hard floor): my vx@L19 vs banked y16; my draw
    # v@{14,19,26} vs the banked kresample shard. Span-mean summaries — bf16
    # cross-process floor per gotchas.md bf16 calibration (real bugs read < 0.99).
    cx28, vx28, _ok = _load_capture_pool(args, pci)
    vd28 = _load_capture_draws(args, kci)
    y16 = np.load(t2202 / "y_holdout_L19.npz")["y16"].astype(np.float32)
    parity: dict = {"gate": gate_rec}

    def _cosrow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        num = (a.astype(np.float64) * b.astype(np.float64)).sum(axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
        return num / den

    cos_pool = _cosrow(vx28[:, L19, :], y16)
    parity["vx_L19_vs_banked_y16"] = {
        "cos_mean": float(cos_pool.mean()),
        "cos_min": float(cos_pool.min()),
        "n": int(len(cos_pool)),
    }
    kb = torch.load(
        sorted((_t2202_dir(args) / "kresample").glob("kresample_shard*.pt"))[0],
        map_location="cpu",
        weights_only=False,
    )
    stored_layers = [int(x) for x in kb["layers"]]
    for sl_idx, sl in enumerate(stored_layers):
        mine = vd28[:, :, sl, :].reshape(-1, H_DIM)
        banked = v_banked[:, :, sl_idx, :].reshape(-1, H_DIM)
        cosd = _cosrow(mine, banked)
        parity[f"draws_L{sl}_vs_banked_kresample"] = {
            "cos_mean": float(cosd.mean()),
            "cos_min": float(cosd.min()),
        }
    floor = min(
        parity["vx_L19_vs_banked_y16"]["cos_min"],
        min(parity[f"draws_L{sl}_vs_banked_kresample"]["cos_min"] for sl in stored_layers),
    )
    assert floor >= 0.99, f"capture parity floor breach vs banked span-means: {parity}"
    out = {"parity": parity, "regime": key, "metadata": _meta("capture28")}
    _write_json_atomic(args.out_eval / "capture28_parity.json", out)
    _upload_eval(args, force=True)
    return out


def _load_capture_pool(args, pci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cx (n_pool, L, H) fp16-as-fp32-on-slice, vx same, strict_ok bool) in pci order."""
    cdir = _cap_dir(args)
    key = _cap_key(args)
    cxs, vxs, oks, cis = [], [], [], []
    for p in sorted(cdir.glob("pool_part*.pt")):
        b = torch.load(p, map_location="cpu", weights_only=False)
        assert b.get("regime") == key, f"stale capture part {p} (regime drift) — recapture"
        cis.extend(int(c) for c in b["ci"])
        cxs.append(b["cx"])
        vxs.append(b["vx"])
        oks.append(b["strict_ok"])
    assert cis == [int(c) for c in pci], "capture pool ci order != pci order"
    return (
        torch.cat(cxs).numpy(),
        torch.cat(vxs).numpy(),
        torch.cat(oks).numpy(),
    )


def _load_capture_draws(args, kci: np.ndarray) -> np.ndarray:
    """(n_cov, K, L, H) fp16 numpy, kci-major seed order == DRAW_SEEDS."""
    cdir = _cap_dir(args)
    key = _cap_key(args)
    vds, keys = [], []
    for p in sorted(cdir.glob("draws_part*.pt")):
        b = torch.load(p, map_location="cpu", weights_only=False)
        assert b.get("regime") == key, f"stale capture part {p} (regime drift) — recapture"
        keys.extend(zip(b["ci"].tolist(), b["seed"].tolist()))
        vds.append(b["vd"])
    want = [(int(c), int(s)) for c in kci for s in DRAW_SEEDS]
    assert keys == want, "draw capture (ci, seed) order drift"
    v = torch.cat(vds).numpy()
    return v.reshape(len(kci), len(DRAW_SEEDS), v.shape[1], v.shape[2])


# ── shared fit-unit core (plot-1 layers AND plot-2 rungs call this) ──────────────


def fit_arms_unit(
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    x_eval_cov: np.ndarray,
    lambdas,
    dev: torch.device,
    *,
    mlp_cfg: dict | None,
    ridge_block: int,
) -> tuple[dict, dict, dict]:
    """(preds_cov, preds_te, meta) for ridge / identity_bias / (optional) mlp_w8192.

    preds_cov[arm] = predictions on the 1,988 covered eval contexts (x_eval_cov);
    preds_te[arm] = predictions on the corpus test rows (banked-parity reads)."""
    preds_cov: dict[str, np.ndarray] = {}
    preds_te: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    pred_te_r, meta_r, payload_r = N1M.fit_ridge_with_weights(
        X, Y, tr, val, te, lambdas, dev, ridge_block
    )
    preds_te["ridge"] = pred_te_r
    preds_cov["ridge"] = N1M.apply_map(payload_r, x_eval_cov, dev).astype(np.float32)
    meta["ridge"] = meta_r
    preds_te["identity_bias"] = MB.identity_bias_predict(X[tr], Y[tr], X[te]).astype(np.float32)
    preds_cov["identity_bias"] = MB.identity_bias_predict(X[tr], Y[tr], x_eval_cov).astype(
        np.float32
    )
    meta["identity_bias"] = {"n_train": int(len(tr))}
    if mlp_cfg is not None:
        cap: dict = {}
        pred_te_m, meta_m = N1M.fit_mlp(
            X,
            Y,
            tr,
            te,
            mlp_cfg["width"],
            mlp_cfg["lr"],
            mlp_cfg["max_epochs"],
            mlp_cfg["batch"],
            mlp_cfg["seed"],
            dev,
            capture_out=cap,
        )
        preds_te["mlp_w8192"] = pred_te_m
        preds_cov["mlp_w8192"] = N1M.apply_map(cap, x_eval_cov, dev).astype(np.float32)
        meta["mlp_w8192"] = meta_m
    return preds_cov, preds_te, meta


# ── phase fits1: per-layer 50k maps, applied + scored on the surface ─────────────


def _assemble_n50k(args):
    """(X_all fp16 (n,LC,H), Y_all, cap_layers, pb bundle, split) — avgtarget recipe."""
    capture_dir = PD.stage_prefix(N50.HF_N50K_PREFIX, args.stage_root, workers=args.stage_workers)
    pass_b = args.stage_root / "pass_b" / "train_context_vectors.pt"
    pb = N1G._load_pass_b_bundle(pass_b)
    assert int(pb["cx_last"].shape[0]) == N50.N_PASS_B, pb["cx_last"].shape
    X_all, Y_all, cap_layers, _dtype = PD._extract_all_layers(capture_dir, None)
    if X_all.shape[0] != N50.N_N50K_NEW:
        raise RuntimeError(f"expected {N50.N_N50K_NEW} n50k rows, got {X_all.shape[0]}")
    pinned = N50._pinned_original_shas(args.orig_dir)
    train, val, test, diag = N50.build_n50k_split(
        X_all.shape[0], None, pinned, n_train=50_000, seed=42
    )
    return X_all, Y_all, cap_layers, pb, (train, val, test, diag)


def _layer_xy(pb, X_all, Y_all, cap_layers, layer: int) -> tuple[np.ndarray, np.ndarray]:
    col = cap_layers.index(layer)
    x = np.concatenate(
        [N50._slice_layer(pb, "cx_last", layer), X_all[:, col, :].astype(np.float32)]
    )
    y = np.concatenate([N50._slice_layer(pb, "v_x", layer), Y_all[:, col, :].astype(np.float32)])
    return x, y


def _surface_at_layer(
    cx28: np.ndarray, vx28: np.ndarray, vd28: np.ndarray, pos: np.ndarray, layer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x_eval_cov (1988,H) fp32, pool_mod (9941,H) fp32, y_avg_cov (1988,H) fp32)."""
    x_eval_cov = cx28[pos, layer, :].astype(np.float32)
    pool_mod, y_avg_cov = avg_pool_assembly(
        vx28[:, layer, :].astype(np.float32), vd28[:, :, layer, :].astype(np.float32), pos
    )
    return x_eval_cov, pool_mod, y_avg_cov


def phase_fits1(args) -> dict:
    """Per-layer unit: refit the 50k maps (banked-parity-gated), apply to the covered
    contexts, score on the draw-averaged 9,941 pool with 50k-train whitening."""
    C.phase("fits1")
    dev = torch.device(args.device)
    unit_dir = args.out_eval / "plot1_units_opsurf"
    unit_dir.mkdir(parents=True, exist_ok=True)
    banked = json.loads(AVG.BANKED_CHAT_JSON.read_text())["per_layer"]
    X_all, Y_all, cap_layers, pb, (train, val, test, diag) = _assemble_n50k(args)

    t2202 = _t2202_dir(args)
    pci = np.asarray(np.load(t2202 / "y_holdout_L19.npz")["ci"], dtype=np.int64)
    kci, _vb, _rows = _load_kresample(args)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)
    cx28, vx28, _ok = _load_capture_pool(args, pci)
    vd28 = _load_capture_draws(args, kci)

    rkey = regime_key(args)
    want_layers = [L19] + [li for li in cap_layers if li != L19]
    for k, layer in enumerate(want_layers):
        out_path = unit_dir / f"L{layer}.json"
        key = {
            **rkey,
            "unit": "plot1_layer",
            "layer": int(layer),
            "n_train": 50_000,
            "train_sha256": diag["train_sha256"],
        }
        if out_path.exists() and json.loads(out_path.read_text()).get("unit_key") == key:
            logger.info("[fits1] unit %d/%d L%d resume-skip", k + 1, len(want_layers), layer)
            continue
        ts = time.time()
        X, Y = _layer_xy(pb, X_all, Y_all, cap_layers, layer)
        x_eval_cov, pool_mod, y_avg_cov = _surface_at_layer(cx28, vx28, vd28, pos, layer)
        mu, ell = P1R.train_whitening_stats(Y[train], dev)
        preds_cov, preds_te, meta = fit_arms_unit(
            X,
            Y,
            train,
            val,
            test,
            x_eval_cov,
            N50.LAMBDAS_N50K,
            dev,
            mlp_cfg={
                "width": args.mlp_width,
                "lr": 3e-4,
                "max_epochs": args.mlp_max_epochs,
                "batch": N1M.MLP_BATCH,
                "seed": args.seed,
            },
            ridge_block=args.ridge_block,
        )
        arms_out: dict[str, dict] = {}
        parity_rows = []
        y_te = Y[test]
        for arm in CHAT_ARMS:
            single_r2, _cos = F79._recon_point(preds_te[arm].astype(np.float64), y_te)
            want = banked.get(str(layer), {}).get("arms", {}).get(arm)
            if want is not None:
                d_r2 = abs(float(single_r2) - want["whole_map_r2"])
                hard = arm in ("ridge", "identity_bias")
                if hard and d_r2 > args.parity_tol:
                    raise RuntimeError(
                        f"L{layer} {arm}: single-target refit R^2 off banked by {d_r2:.4g} "
                        f"(tol {args.parity_tol}) — prediction set not reconciled"
                    )
                parity_rows.append({"arm": arm, "d_r2": d_r2, "hard": hard})
            rec = score_opsurf(
                preds_cov[arm],
                pool_mod,
                y_avg_cov,
                pos,
                mu,
                ell,
                args.n_boot,
                args.seed,
                args.csls_k,
            )
            rec["fit_meta"] = meta[arm]
            arms_out[arm] = rec
        unit = {
            "unit_key": key,
            "layer": int(layer),
            "arms": arms_out,
            "parity_single_target_r2": parity_rows,
            "wall_time_s": round(time.time() - ts, 1),
        }
        _write_json_atomic(out_path, unit)
        print(
            f"[fits1] unit {k + 1}/{len(want_layers)} L{layer} ridge wcsls@1="
            f"{arms_out['ridge']['retrieval']['whiten_csls']['acc_at_k'][1]:.4f} "
            f"elapsed={time.time() - ts:.0f}s",
            flush=True,
        )
        _merge_plot1(args, unit_dir, want_layers, diag)
        _upload_eval(args)
    _merge_plot1(args, unit_dir, want_layers, diag)
    _upload_eval(args, force=True)
    return {"out": str(args.out_eval / "plot1_opsurf.json")}


def _merge_plot1(args, unit_dir: Path, want_layers, diag) -> None:
    merged = {
        "per_layer": {
            str(li): json.loads((unit_dir / f"L{li}.json").read_text())
            for li in want_layers
            if (unit_dir / f"L{li}.json").exists()
        },
        "split": diag,
        "conventions": CONVENTIONS,
        "regime": regime_key(args),
        "metadata": _meta("fits1"),
    }
    _write_json_atomic(args.out_eval / "plot1_opsurf.json", merged)


# ── phase fits2: L19 rungs on the surface ────────────────────────────────────────


def _rung_cell(
    args,
    cells_dir: Path,
    tag: str,
    X: np.ndarray,
    Y: np.ndarray,
    sel: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    x_eval_cov: np.ndarray,
    pool_mod: np.ndarray,
    y_avg_cov: np.ndarray,
    pos: np.ndarray,
    lambdas,
    dev: torch.device,
    *,
    include_mlp: bool,
    extra_key: dict,
    parity_fn=None,
) -> dict:
    """One rung unit: fit + apply + score with PER-RUNG train-answer whitening."""
    out_path = cells_dir / f"{tag}.json"
    key = {**regime_key(args), "unit": "plot2_rung", "tag": tag, **extra_key}
    if out_path.exists() and json.loads(out_path.read_text()).get("unit_key") == key:
        logger.info("[fits2] %s resume-skip", tag)
        return json.loads(out_path.read_text())
    t0 = time.time()
    mu, ell = P1R.train_whitening_stats(Y[sel], dev)
    mlp_cfg = None
    if include_mlp:
        mlp_cfg = {
            "width": args.mlp_width,
            "lr": 3e-4,
            "max_epochs": args.mlp_max_epochs,
            "batch": N1M.MLP_BATCH,
            "seed": args.seed,
        }
    preds_cov, preds_te, meta = fit_arms_unit(
        X,
        Y,
        sel,
        val,
        te,
        x_eval_cov,
        lambdas,
        dev,
        mlp_cfg=mlp_cfg,
        ridge_block=args.ridge_block,
    )
    arms_out = {}
    for arm, pred in preds_cov.items():
        rec = score_opsurf(
            pred, pool_mod, y_avg_cov, pos, mu, ell, args.n_boot, args.seed, args.csls_k
        )
        rec["fit_meta"] = meta[arm]
        arms_out[arm] = rec
    parity = parity_fn(preds_te) if parity_fn is not None else None
    cell = {
        "unit_key": key,
        "tag": tag,
        "n_train": int(len(sel)),
        "arms": arms_out,
        "parity": parity,
        "wall_time_s": round(time.time() - t0, 1),
    }
    _write_json_atomic(out_path, cell)
    print(
        f"[fits2] unit {tag} n={len(sel)} ridge wcsls@1="
        f"{arms_out['ridge']['retrieval']['whiten_csls']['acc_at_k'][1]:.4f} "
        f"elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    return cell


def phase_fits2(args) -> dict:
    """Plot-2 L19 rungs: small/mid rungs from the 50k capture (sub-leg A), big rungs
    from the n1m capture + the banked 963k weight payloads (sub-leg B)."""
    C.phase("fits2")
    dev = torch.device(args.device)
    cells_dir = args.out_eval / "scaling_cells_opsurf"
    cells_dir.mkdir(parents=True, exist_ok=True)

    t2202 = _t2202_dir(args)
    pci = np.asarray(np.load(t2202 / "y_holdout_L19.npz")["ci"], dtype=np.int64)
    kci, _vb, _rows = _load_kresample(args)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)
    cx28, vx28, _okp = _load_capture_pool(args, pci)
    vd28 = _load_capture_draws(args, kci)
    x_eval_cov, pool_mod, y_avg_cov = _surface_at_layer(cx28, vx28, vd28, pos, L19)
    del cx28, vx28, vd28

    # ── sub-leg A: rungs <= 25k from the 50k assembly ────────────────────────────
    X_all, Y_all, cap_layers, pb, (train, val, test, diag) = _assemble_n50k(args)
    X, Y = _layer_xy(pb, X_all, Y_all, cap_layers, L19)
    del X_all, Y_all
    for n in SMALL_RUNG_NS:
        for d in SMALL_RUNG_DRAWS:
            rng = np.random.default_rng(19010000 + n * 10 + d)  # avgtarget ladder seeds
            sel = np.sort(train[rng.choice(len(train), size=n, replace=False)])
            _rung_cell(
                args,
                cells_dir,
                f"n{n}_d{d}",
                X,
                Y,
                sel,
                val,
                test,
                x_eval_cov,
                pool_mod,
                y_avg_cov,
                pos,
                N50.LAMBDAS_N50K,
                dev,
                include_mlp=False,
                extra_key={"n": n, "draw": d, "source": "n50k"},
            )
    for n in MID_RUNG_NS:
        rng = np.random.default_rng(19010000 + n * 10)
        sel = np.sort(train[rng.choice(len(train), size=n, replace=False)])
        _rung_cell(
            args,
            cells_dir,
            f"n{n}_d0",
            X,
            Y,
            sel,
            val,
            test,
            x_eval_cov,
            pool_mod,
            y_avg_cov,
            pos,
            N50.LAMBDAS_N50K,
            dev,
            include_mlp=(n in MLP_SMALL_NS),
            extra_key={"n": n, "draw": 0, "source": "n50k"},
        )
    del X, Y, pb

    # Reap the n50k stage before the ~80 GB n1m stage (MooseFS quota headroom) — only
    # once fits1's units are all on disk (the reap's last consumer; #1489 rule).
    if not args.smoke:
        plot1_done = all(
            (args.out_eval / "plot1_units_opsurf" / f"L{li}.json").exists() for li in LAYERS_ALL
        )
        n50k_stage = args.stage_root / Path(N50.HF_N50K_PREFIX).parent
        if plot1_done and n50k_stage.exists():
            PD._reap_stage(n50k_stage)
            logger.info("[fits2] reaped n50k stage %s", n50k_stage)
        elif n50k_stage.exists():
            logger.warning("[fits2] plot1 units incomplete — n50k stage NOT reaped")

    # ── sub-leg B: big rungs from the n1m assembly + banked 963k payloads ────────
    n1m_capture_prefix = f"{N1G.HF_PREFIX}/final_token_capture"
    capture_dir = PD.stage_prefix(n1m_capture_prefix, args.stage_root, workers=args.stage_workers)
    ns = argparse.Namespace(
        pass_b=args.stage_root / "pass_b" / "train_context_vectors.pt",
        manifest_from_hf=True,
        manifest_hf_prefix=N1G.HF_PREFIX,
        out_dir=args.stage_root / "n1m_work",
        n1m_capture_dir=capture_dir,
        fresh_stream=False,
        hf_prefix=n1m_capture_prefix,
        orig_dir=args.orig_dir,
    )
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    Xb, Yb, prov, r1_train, valb, testb, _split = N1M.assemble(ns, layer=L19)
    pools = N1M._pool_rows(prov, r1_train, Xb.shape[0], valb, testb)
    y_te_old = Yb[testb]

    for tag, n_target, refit_mlp in BIG_RUNGS:
        # Selection seed 0 == the banked n1m_fits/densify-bign convention (requires
        # PYTHONHASHSEED=0 — re-exec'd at main() entry; select_train hashes the tag).
        sel, sel_diag = N1M.select_train(pools, tag, n_target, "lmsys", 0)
        assert len(sel) == n_target, (tag, len(sel))
        banked_path = AVG.BANKED_BIGN_DIR / f"{tag}.json"

        def _parity(preds_te, _banked_path=banked_path, _tag=tag):
            if not _banked_path.exists():
                return {"note": f"no banked cell for {_tag} (new rung)"}
            banked_cell = json.loads(_banked_path.read_text())
            rows = {}
            for arm in ("ridge", "identity_bias"):
                helper = MB.knn_retrieval(
                    preds_te[arm].astype(np.float64), y_te_old, ks=(1,), metric="euclidean"
                )
                want = float(banked_cell[arm]["retrieval"]["euclidean"]["acc_at_k"]["1"])
                d_a1 = abs(helper["acc_at_k"][1] - want)
                if d_a1 > args.acc_parity_tol:
                    raise RuntimeError(f"{_tag}/{arm}: acc1 parity off banked by {d_a1:.4g}")
                rows[arm] = {"d_acc1": d_a1}
            return rows

        _rung_cell(
            args,
            cells_dir,
            tag,
            Xb,
            Yb,
            sel,
            valb,
            testb,
            x_eval_cov,
            pool_mod,
            y_avg_cov,
            pos,
            N1M.LAMBDAS_N1M,
            dev,
            include_mlp=refit_mlp,
            extra_key={"n": n_target, "source": "n1m", "selection": sel_diag},
            parity_fn=_parity,
        )

    # 963,444-row arms from the BANKED weight payloads (no refit; avgtarget recipe).
    tag963 = f"n{N_963K}_banked"
    out963 = cells_dir / f"{tag963}.json"
    key963 = {
        **regime_key(args),
        "unit": "plot2_rung",
        "tag": tag963,
        "n": N_963K,
        "source": "banked_n1m_weights",
    }
    if not (out963.exists() and json.loads(out963.read_text()).get("unit_key") == key963):
        t0 = time.time()
        full_pool = pools["full"]
        assert len(full_pool) == N_963K, (len(full_pool), N_963K)
        mu, ell = P1R.train_whitening_stats(Yb[full_pool], dev)
        from huggingface_hub import hf_hub_download

        arms_out: dict[str, dict] = {}
        ridge_payload = None
        for arm, fname in (("ridge", "ridge.pt"), ("mlp_w8192", "mlp_w8192.pt")):
            local = hub.retry_transient(
                lambda f=fname: hf_hub_download(
                    C.HF_DATA_REPO,
                    filename=f"{AVG.N1M_WEIGHTS_PREFIX}/L{L19}/{f}",
                    repo_type="dataset",
                    cache_dir=str(args.cache_dir),
                ),
                what=f"n1m weights {fname}",
            )
            payload = torch.load(local, map_location="cpu", weights_only=False)
            assert int(payload["layer"]) == L19, payload.get("layer")
            if arm == "ridge":
                ridge_payload = payload
            pred = N1M.apply_map(payload, x_eval_cov, dev).astype(np.float32)
            arms_out[arm] = score_opsurf(
                pred, pool_mod, y_avg_cov, pos, mu, ell, args.n_boot, args.seed, args.csls_k
            )
        xmu = np.asarray(ridge_payload["xmu"], dtype=np.float64)
        ymu = np.asarray(ridge_payload["ymu"], dtype=np.float64)
        pred_ib = (x_eval_cov.astype(np.float64) + (ymu - xmu)).astype(np.float32)
        arms_out["identity_bias"] = score_opsurf(
            pred_ib, pool_mod, y_avg_cov, pos, mu, ell, args.n_boot, args.seed, args.csls_k
        )
        cell = {
            "unit_key": key963,
            "tag": tag963,
            "n_train": N_963K,
            "arms": arms_out,
            "parity": {"note": "banked weight payloads applied verbatim (no refit)"},
            "wall_time_s": round(time.time() - t0, 1),
        }
        _write_json_atomic(out963, cell)
        print(f"[fits2] unit {tag963} done elapsed={time.time() - t0:.0f}s", flush=True)
    _upload_eval(args, force=True)
    return {"cells_dir": str(cells_dir)}


# ── phase score: leg-0 banked reconciliation + merges ────────────────────────────


def leg0_sanity(args) -> dict:
    """Reproduce the #2202 freshwhiten covered-row reads from the BANKED tensors with
    THIS script's scorer (raw-euclidean + whitened-cosine, single + draw-averaged),
    reconciling acc@1 against the committed summary — the zero-GPU implementation gate.
    The whitened-CSLS numbers on the same banked inputs are NEW (recorded, no banked
    counterpart)."""
    t2202 = _t2202_dir(args)
    pd_ = np.load(t2202 / "pred16.npz")
    yd = np.load(t2202 / "y_holdout_L19.npz")
    pred = pd_["pred16"].astype(np.float64)
    y16 = yd["y16"].astype(np.float64)
    pci = np.asarray(pd_["ci"], dtype=np.int64)
    assert (pci == np.asarray(yd["ci"], dtype=np.int64)).all(), "pred16/y_holdout ci misalign"
    wz = np.load(t2202 / "whiten_stats.npz")
    mu_a = np.asarray(wz["mu_A"], dtype=np.float64)
    ell = np.asarray(wz["L"], dtype=np.float64)
    kci, v_banked, _rows = _load_kresample(args)
    kb = torch.load(
        sorted((t2202 / "kresample").glob("kresample_shard*.pt"))[0],
        map_location="cpu",
        weights_only=False,
    )
    l19_idx = [int(x) for x in kb["layers"]].index(L19)
    draws = v_banked[:, :, l19_idx, :].astype(np.float64)  # (1988, 4, H)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)

    banked = json.loads(FRESHWHITEN_SUMMARY.read_text())["map_acc_on_covered_rows"]
    tol = args.recon_tol_rows / N_COVERED
    out: dict = {"tol_rows": args.recon_tol_rows}
    for read, pool_np, y_cov in (
        ("single_draw_target", y16.astype(np.float32), y16[pos].astype(np.float32)),
        ("draw_averaged_target", *avg_pool_assembly(y16.astype(np.float32), draws, pos)),
    ):
        rec = score_opsurf(
            pred[pos],
            pool_np,
            y_cov,
            pos,
            mu_a,
            ell,
            args.n_boot,
            args.seed,
            args.csls_k,
            include_raw_euclidean=True,
        )
        row = {"recomputed": {m: rec["retrieval"][m]["acc_at_k"][1] for m in rec["retrieval"]}}
        for metric in ("raw_euclidean", "whiten_cos"):
            want = float(banked[read][metric]["acc_at_k"]["1"])
            got = float(rec["retrieval"][metric]["acc_at_k"][1])
            assert abs(got - want) <= tol, (
                f"leg-0 reconciliation FAILED: {read}/{metric} got {got:.6f} vs banked "
                f"{want:.6f} (tol {tol:.2e})"
            )
            row[f"banked_{metric}"] = want
        out[read] = row
        logger.info("[score] leg-0 %s reconciled: %s", read, row["recomputed"])
    return out


def phase_score(args) -> dict:
    C.phase("score")
    leg0 = leg0_sanity(args)
    _write_json_atomic(
        args.out_eval / "leg0_reconciliation.json",
        {"leg0": leg0, "regime": regime_key(args), "metadata": _meta("score")},
    )
    # plot1 merge is maintained per-unit by fits1; refresh the scaling merge here.
    cells_dir = args.out_eval / "scaling_cells_opsurf"
    cells = {p.stem: json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))}
    plot1_p = args.out_eval / "plot1_opsurf.json"
    plot1 = json.loads(plot1_p.read_text()) if plot1_p.exists() else {}
    merged = {
        "cells": cells,
        "l19_50k_from_plot1": plot1.get("per_layer", {}).get(str(L19)),
        "conventions": CONVENTIONS,
        "regime": regime_key(args),
        "metadata": _meta("score"),
    }
    _write_json_atomic(args.out_eval / "scaling_opsurf.json", merged)
    _upload_eval(args, force=True)
    return {"leg0": leg0}


# ── phase fig: rebuild both paper figures on the new surface ─────────────────────


def _acc_pt(rec: dict) -> tuple[float, float, float]:
    r = rec["retrieval"]["whiten_csls"]
    ak = r["acc_at_k"]
    a1 = float(ak["1"] if "1" in ak else ak[1])
    return a1, float(r["acc1_ci"]["lo"]), float(r["acc1_ci"]["hi"])


def _r2_pt(rec: dict) -> tuple[float, float, float]:
    ci = rec["bootstrap_ci"]["r2"]
    return float(rec["whole_map_r2"]), float(ci["lo"]), float(ci["hi"])


def _fig_plot1(args) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    boundary = json.loads(AVG.BANKED_BOUNDARY_JSON.read_text())["per_layer"]
    per_layer = json.loads((args.out_eval / "plot1_opsurf.json").read_text())["per_layer"]

    labels = dict(P1R.ARM_LABELS)
    set_paper_style()
    colors = dict(zip(labels, paper_palette(len(labels))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for arm, label in labels.items():
        if arm == "identity_bias":
            # Dropped from the rendered figure (user, 2026-08-25 avgtarget round): its
            # R^2 range crushed the ridge/MLP curves. Palette keys stay on the full
            # ARM_LABELS so remaining arms keep their colors across sibling figures.
            continue
        if arm == "boundary_ridge":
            # BANKED (deterministic WikiText target; single-target 1,000-pool R^2) —
            # carried unchanged per the decision record; R^2 panel only.
            layers = sorted(int(li) for li in boundary)
            r2 = [boundary[str(li)]["arms"][arm]["whole_map_r2"] for li in layers]
            ax1.plot(layers, r2, marker="o", ms=3, color=colors[arm], label=label)
            continue
        layers = sorted(int(li) for li in per_layer)
        recs = [per_layer[str(li)]["arms"][arm] for li in layers]
        r2 = [r["whole_map_r2"] for r in recs]
        pts = [_acc_pt(r) for r in recs]
        ax1.plot(layers, r2, marker="o", ms=3, color=colors[arm], label=label)
        ax2.plot(layers, [p[0] for p in pts], marker="o", ms=3, color=colors[arm], label=label)
        ax2.fill_between(
            layers, [p[1] for p in pts], [p[2] for p in pts], color=colors[arm], alpha=0.15, lw=0
        )
    n_pool = N_POOL if not args.smoke else args.smoke_pool
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Held-out $R^2$,\ndraw-averaged target")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel(f"acc@1, draw-averaged target\n(whitened cosine + CSLS, pool {n_pool:,})")
    ax2.axhline(1.0 / n_pool, ls="--", lw=0.8, color="gray", label=f"Chance (1/{n_pool:,})")
    ax2.set_ylim(-0.02, 1.0)
    ax1.legend(frameon=False, fontsize=7)
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c1_layer_profile", dir=args.fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def _fig_plot2(args) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("iclr")
    doc = json.loads((args.out_eval / "scaling_opsurf.json").read_text())
    cells = doc["cells"]
    l19_50k = doc["l19_50k_from_plot1"]
    assert l19_50k is not None, "plot1 L19 unit absent — run fits1 before fig"

    def series(arm: str, pt_fn) -> tuple[list[int], list[float], list[float], list[float]]:
        """x, y, err_lo, err_hi — multi-draw rungs collapse to mean +- sd; single-draw
        rungs carry their bootstrap CI half-widths."""
        by_n: dict[int, list[tuple[float, float, float]]] = {}
        for cell in cells.values():
            if arm not in cell["arms"]:
                continue
            by_n.setdefault(int(cell["n_train"]), []).append(pt_fn(cell["arms"][arm]))
        if arm in l19_50k["arms"]:
            by_n.setdefault(50_000, []).append(pt_fn(l19_50k["arms"][arm]))
        ns = sorted(by_n)
        y, lo, hi = [], [], []
        for n in ns:
            pts = by_n[n]
            vals = [p[0] for p in pts]
            m = float(np.mean(vals))
            y.append(m)
            if len(pts) > 1:
                sd = float(np.std(vals))
                lo.append(sd)
                hi.append(sd)
            else:
                lo.append(max(0.0, pts[0][0] - pts[0][1]))
                hi.append(max(0.0, pts[0][2] - pts[0][0]))
        return ns, y, lo, hi

    fig, (ax_r2, ax_acc) = plt.subplots(1, 2, figsize=figsize_iclr_panels(2, height_in=2.3))
    styles = {
        "ridge": (paper_color("instruct"), "o", "-", 1.4, "linear map (ridge)"),
        "identity_bias": (
            paper_color("identity_bias"),
            "s",
            "--",
            1.2,
            "copy context vector + trained bias",
        ),
        "mlp_w8192": (paper_color("neural_map"), "D", ":", 1.2, "nonlinear map (MLP)"),
    }
    n_pool = N_POOL if not args.smoke else args.smoke_pool
    for arm, (col, mk, ls_, lw, label) in styles.items():
        ns, y, lo, hi = series(arm, _r2_pt)
        ax_r2.errorbar(
            ns,
            y,
            yerr=[np.maximum(0, lo), np.maximum(0, hi)],
            marker=mk,
            ls=ls_,
            color=col,
            lw=lw,
            ms=3,
            capsize=1.5,
            label=label,
        )
        ns, y, lo, hi = series(arm, _acc_pt)
        ax_acc.errorbar(
            ns,
            y,
            yerr=[np.maximum(0, lo), np.maximum(0, hi)],
            marker=mk,
            ls=ls_,
            color=col,
            lw=lw,
            ms=3,
            capsize=1.5,
        )
    ax_r2.axhline(0.0, color="black", lw=0.7, ls=":")
    ax_r2.set_ylabel("held-out $R^2$,\ndraw-averaged target")
    ax_r2.set_ylim(-1.05, 1.0)
    ax_acc.axhline(1.0 / n_pool, color="black", lw=0.7, ls=":")
    ax_acc.set_ylabel(f"acc@1, draw-averaged target\n(whitened cos + CSLS, pool {n_pool:,})")
    ax_acc.set_ylim(0.0, 1.0)
    for ax in (ax_r2, ax_acc):
        ax.set_xscale("log")
        ax.set_xlabel("training contexts")
    handles, lbls = ax_r2.get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, "c1_scaling_train_pool", dir=args.fig_dir)
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def phase_fig(args) -> dict:
    C.phase("fig")
    p1 = _fig_plot1(args)
    p2 = _fig_plot2(args)
    copied = []
    if args.copy_paper_stems and not args.smoke:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for stem in ("c1_layer_profile", "c1_scaling_train_pool"):
            for ext in (".pdf", ".png", ".meta.json"):
                src = args.fig_dir / f"{stem}{ext}"
                if src.exists():
                    shutil.copy2(src, PAPER_FIG_DIR / f"{stem}{ext}")
                    copied.append(str(PAPER_FIG_DIR / f"{stem}{ext}"))
    logger.info("[fig] plot1=%s plot2=%s copied=%d", p1, p2, len(copied))
    return {"plot1": p1, "plot2": p2, "copied": copied}


def phase_upload(args) -> dict:
    """Final eval upload + verify + poll_pipeline-conformant sentinel."""
    C.phase("upload")
    _upload_eval(args, force=True)
    expected = [
        f"{HF_ROUND_PREFIX}/eval/plot1_opsurf.json",
        f"{HF_ROUND_PREFIX}/eval/scaling_opsurf.json",
        f"{HF_ROUND_PREFIX}/eval/leg0_reconciliation.json",
    ]
    if not args.skip_upload and not args.smoke:
        from huggingface_hub import HfApi

        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            C.HF_DATA_REPO,
            expected,
            path_in_repo=f"{HF_ROUND_PREFIX}/eval",
            repo_type="dataset",
        )
        assert not missing, f"opsurface eval uploads missing on Hub: {missing}"
    plot1 = json.loads((args.out_eval / "plot1_opsurf.json").read_text())
    l19 = plot1["per_layer"][str(L19)]["arms"]["ridge"]
    note = {
        "round": ROUND,
        "l19_ridge_wcsls_acc1": l19["retrieval"]["whiten_csls"]["acc_at_k"]["1"]
        if "1" in l19["retrieval"]["whiten_csls"]["acc_at_k"]
        else l19["retrieval"]["whiten_csls"]["acc_at_k"][1],
        "eval_paths": [
            str(args.out_eval / "plot1_opsurf.json"),
            str(args.out_eval / "scaling_opsurf.json"),
        ],
    }
    C.write_sentinel("epm:results", json.dumps(note), task_id=ISSUE, extra={"round": ROUND})
    return note


# ── smoke (CPU, synthetic fixture; pure-math phases through the REAL functions) ──


def run_smoke(args) -> int:
    """Tiny synthetic end-to-end of every pure-math path: unit fits (all 3 arm
    classes), per-rung whitening, scoring, merges, and both figure renders. Writes to
    --smoke-dir only (never canonical eval_results/figures). Blind spots: module
    docstring enumeration."""
    rng = np.random.default_rng(0)
    h = args.smoke_h
    n_pool, n_cov, k = args.smoke_pool, args.smoke_cov, K_DRAWS
    args.layers = (0, 1)
    args.csls_k = 3
    args.n_boot = 50
    args.mlp_width = 8
    args.mlp_max_epochs = 2
    smoke_root = Path(args.smoke_dir)
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    args.out_eval = smoke_root / "eval"
    args.fig_dir = smoke_root / "figs"
    args.out_eval.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")

    # Synthetic linear-ish task: Y ~ X @ W + noise; eval pool drawn from the same map.
    n_train_corpus = 220
    w_true = rng.normal(size=(h, h)) / np.sqrt(h)
    X = rng.normal(size=(n_train_corpus, h)).astype(np.float32)
    Y = (X @ w_true + 0.1 * rng.normal(size=(n_train_corpus, h))).astype(np.float32)
    tr = np.arange(0, 160)
    val = np.arange(160, 190)
    te = np.arange(190, 220)
    cx_pool = rng.normal(size=(n_pool, h)).astype(np.float32)
    vx_pool = (cx_pool @ w_true + 0.1 * rng.normal(size=(n_pool, h))).astype(np.float32)
    pos = np.sort(rng.choice(n_pool, size=n_cov, replace=False))
    vd = (vx_pool[pos][:, None, :] + 0.05 * rng.normal(size=(n_cov, k, h))).astype(np.float32)

    pool_mod, y_avg_cov = avg_pool_assembly(vx_pool, vd, pos)
    lambdas = np.logspace(-3, 3, 7)
    unit_dir = args.out_eval / "plot1_units_opsurf"
    unit_dir.mkdir(parents=True, exist_ok=True)
    per_layer = {}
    for layer in args.layers:
        mu, ell = P1R.train_whitening_stats(Y[tr], dev)
        preds_cov, _preds_te, meta = fit_arms_unit(
            X,
            Y,
            tr,
            val,
            te,
            cx_pool[pos],
            lambdas,
            dev,
            mlp_cfg={"width": 8, "lr": 3e-4, "max_epochs": 2, "batch": 32, "seed": 0},
            ridge_block=4096,
        )
        arms = {}
        for arm, pred in preds_cov.items():
            rec = score_opsurf(
                pred, pool_mod, y_avg_cov, pos, mu, ell, args.n_boot, args.seed, args.csls_k
            )
            rec["fit_meta"] = meta[arm]
            arms[arm] = rec
        unit = {"unit_key": {"layer": layer}, "layer": layer, "arms": arms}
        _write_json_atomic(unit_dir / f"L{layer}.json", unit)
        per_layer[str(layer)] = unit
        print(
            f"[smoke] unit plot1 L{layer} arms={sorted(arms)} ridge wcsls@1="
            f"{arms['ridge']['retrieval']['whiten_csls']['acc_at_k'][1]:.3f}",
            flush=True,
        )
    _write_json_atomic(
        args.out_eval / "plot1_opsurf.json",
        {
            "per_layer": per_layer,
            "split": {},
            "conventions": CONVENTIONS,
            "regime": regime_key(args),
            "metadata": _meta("smoke"),
        },
    )

    cells_dir = args.out_eval / "scaling_cells_opsurf"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for n in (16, 32):
        for d in (0, 1):
            sel = np.sort(rng.choice(tr, size=n, replace=False))
            _rung_cell(
                args,
                cells_dir,
                f"n{n}_d{d}",
                X,
                Y,
                sel,
                val,
                te,
                cx_pool[pos],
                pool_mod,
                y_avg_cov,
                pos,
                lambdas,
                dev,
                include_mlp=(d == 0),
                extra_key={"n": n, "draw": d, "source": "smoke"},
            )
    cells = {p.stem: json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))}
    _write_json_atomic(
        args.out_eval / "scaling_opsurf.json",
        {
            "cells": cells,
            "l19_50k_from_plot1": per_layer[str(args.layers[0])],
            "conventions": CONVENTIONS,
            "regime": regime_key(args),
            "metadata": _meta("smoke"),
        },
    )
    figs = phase_fig(args)
    for key in ("plot1", "plot2"):
        png = [p for p in figs[key].values() if str(p).endswith(".png")]
        assert png and Path(png[0]).stat().st_size > 5_000, f"smoke fig {key} empty"
    print(
        f"[smoke] PASS — arm classes covered: ridge, identity_bias, mlp_w8192; "
        f"figs at {args.fig_dir}",
        flush=True,
    )
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────


PHASES = ("stage", "capture28", "fits1", "fits2", "score", "fig", "upload", "pod-all")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="#1901 opsurface-rebase: both paper figures on the #2202 surface"
    )
    ap.add_argument("--phase", choices=list(PHASES), default=None)
    ap.add_argument("--stage-root", type=Path, default=Path("/workspace/opsurf_stage"))
    ap.add_argument("--cache-dir", type=Path, default=None, help="HF cache (default stage/cache)")
    ap.add_argument("--out-eval", type=Path, default=OUT_EVAL_DEFAULT)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR_DEFAULT)
    ap.add_argument("--orig-dir", type=Path, default=N50.DEFAULT_ORIG_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=F79.BOOT_N)
    ap.add_argument("--csls-k", type=int, default=K_CSLS)
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS_ALL))
    ap.add_argument("--mlp-width", type=int, default=8192)
    ap.add_argument("--mlp-max-epochs", type=int, default=F79.MLP_MAX_EPOCHS)
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument("--parity-tol", type=float, default=1e-2)
    ap.add_argument("--acc-parity-tol", type=float, default=0.01)
    ap.add_argument("--recon-tol-rows", type=float, default=3.0)
    ap.add_argument("--capture-token-budget", type=int, default=16_384)
    ap.add_argument("--capture-flush-every", type=int, default=512)
    ap.add_argument("--upload-every", type=int, default=5)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--copy-paper-stems", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic e2e (pure-math phases)")
    ap.add_argument("--smoke-dir", type=Path, default=Path("/tmp/issue-1901-opsurf-smoke"))
    ap.add_argument("--smoke-h", type=int, default=16)
    ap.add_argument("--smoke-pool", type=int, default=40)
    ap.add_argument("--smoke-cov", type=int, default=12)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)
    # PYTHONHASHSEED=0 re-exec: N1M.select_train seeds via hash(tag) — the banked
    # bign selections (densify/avgtarget lineage, recorded pythonhashseed "0") are
    # reproducible only under the same pin. Import-safe: fires only via main().
    import os as _os

    if _os.environ.get("PYTHONHASHSEED") != "0":
        _os.execvpe(
            sys.executable,
            [sys.executable, *sys.argv],
            {**_os.environ, "PYTHONHASHSEED": "0"},
        )
    torch.set_num_threads(int(args.n_threads))
    if args.cache_dir is None:
        args.cache_dir = args.stage_root / "cache"
    if args.smoke:
        rc = run_smoke(args)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(rc)
    assert args.phase is not None, "--phase required (or --smoke / --import-check)"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_eval.mkdir(parents=True, exist_ok=True)
    if args.phase == "stage":
        phase_stage(args)
    elif args.phase == "capture28":
        phase_capture28(args)
    elif args.phase == "fits1":
        phase_fits1(args)
    elif args.phase == "fits2":
        phase_fits2(args)
    elif args.phase == "score":
        phase_score(args)
    elif args.phase == "fig":
        phase_fig(args)
    elif args.phase == "upload":
        phase_upload(args)
    elif args.phase == "pod-all":
        phase_stage(args)
        phase_capture28(args)
        phase_fits1(args)
        phase_fits2(args)
        phase_score(args)
        phase_fig(args)
        phase_upload(args)
        C.phase("done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
