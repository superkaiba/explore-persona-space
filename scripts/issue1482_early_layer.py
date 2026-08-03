"""Issue #1482 — early-layer-arm round driver (plan v17): the layer-3 token-stratum arm.

ONE new variable vs the parent battery: the SAE layer/stratum (layer 3, k=64
trainer_1 primary / k=128 trainer_2 robustness). Everything else — corpus, split,
capture convention, SAE suite, pooling recipe, fit machinery, judge instrument —
is inherited from `issue1482_error_analysis` (EA) / `issue1482_feature_extremes`
(FE) / `issue1482_feature_correlates` (FC) / `issue779_ffc_n1m_fits` (N1M).

JUDGED-LABEL FREEZE (plan v17 §0.-1; user directive epm:progress v101,
2026-07-28): NO judged feature-labelling runs this round — the E5 judge leg is
REPLACED by E5-evidence (mechanical selection + per-feature evidence packet for
the #1773 labelling resume) and the E6 H1/H2 judged-label permutation legs are
DEFERRED behind --labels-file. Label schema note for the resume: #1773's
instrument uses the THREE-WAY speaker_property field (language / register_style
/ identity_disposition) per the v98 persona-field retraction — NOT the retracted
binary persona_related field.

Phases (plan v17 §4 + §0.-1; smoke IS this driver with tiny args — PASS_UNIFIED):
  pilot    E0: 1 raw chunk through the FULL production path — tokens/s, Gate B-e
           (L3 FVE/L0 k64+k128 vs published-offset thresholds), hook-alignment
           probes (L3 SAE on h1/h5), prefix-end constancy @L3, G2-e two-bar
           identity gate (batched-vs-batch-1 early layers 0-3 >= 0.999; fresh
           c_last@19 vs parent STORED cx_last@19 flattened >= 0.995), fit-kernel
           pilot timing (the E3 `pilot-gated` basis).
  capture  E1: seeded stratified 30k subsample (24k of sae_fit + 6k of holdout,
           seed 14823, sha-asserted vs committed split_1482.json literals) ->
           teacher-forced capture @ layers (3, 19) -> L3 SAE encode + pooling
           (default mask + sink-robustness mask + k128), per-chunk checkpointed.
  upload1  E2: L3 pooled store -> HF analysis_tensors/early_layer/ BEFORE fits
           (#825 expensive-store-before-long-fit).
  fits     E3: 5 shared-Gram ridge designs + 1 MLP twin + shuffle-null K=20 +
           covariate battery + mapping baselines (identity+bias on the aligned
           feature-id intersection; kNN retrieval per arm).
  upload2  E4: eval outputs -> HF + the poller results sentinel.
  evidence E5-evidence (off-pod VM, 0 GPU, ZERO judge calls): FE._select tails
           at L3 (mechanical, no labels) -> per-feature evidence packet — top-8
           firing answer ROW IDS by the primary arm's pooled max ranking (text
           stays on HF), top-10 co-activation neighbour ids, top-20 direct-logit
           footprint tokens, covariate row, selection provenance ->
           eval_results/issue_1482/early_layer/evidence/evidence.json.
           --phase judge is a hard exit (RC_JUDGE_FROZEN).
  analyze  E6 (off-pod VM): mechanical depth x predictability profile ->
           mech_tests.json (depth descriptives at matched n, covariate
           correlations + partials, sink/k128 twin agreement, shuffle-null band,
           identity+kNN baselines); the registered §3 H1/H2 permutation recipes
           are PRESERVED VERBATIM and run ONLY under --labels-file (the #1773
           resume) — absent labels a `deferred` stanza is emitted instead.

Pod-side contract: sentinels under /workspace/logs/issue-1482-*.json ONLY (never
task.py); [phase=...] log lines come from the launcher
(scripts/issue1482_early_layer_launch.sh); the results sentinel is written by
the terminal pod phase (upload2). LMSYS/WildChat text is DIGEST-ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
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

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482_early")

TASK_ID = 1482
L_EARLY = 3
L_LATE = 19
GATE_EARLY_LAYERS = (0, 1, 2, 3)  # G2-e bar A: batched-vs-batch-1 early-layer states
HOOK_PROBE_LAYERS = (1, 5)  # E0 (c): same-layer-maximal expected at 3 +/- 2
PILOT_CAPTURE_LAYERS = (0, 1, 2, 3, 5, 19)
CAPTURE_LAYERS = (L_EARLY, L_LATE)
SUBSAMPLE_SEED = 14823  # plan §10 Seeds
SHUFFLE_SEEDS = tuple(range(1_482_100, 1_482_120))  # K=20, plan §10
BOOT_PERM_SEED = 148_230  # bootstrap/permutation seed, plan §10
MLP_LR = 3e-4  # parent MLP recipe (issue1482_error_analysis._p3_unit_mlp)
# Gate B-e thresholds (plan §7 / §11 R5): the parent Gate B was published-FVE offsets
# (0.806 - 0.106 PASS / - 0.256 HALT); same offsets applied to L3's published 0.9309.
GATE_BE_PASS = 0.825
GATE_BE_HALT = 0.675
# G2-e two-bar calibration (plan §7; bf16 single-position gotcha): bar A = per-layer
# batched-vs-batch-1 cosine over layers 0-3 (the sharp bug catcher for OUR capture
# path); bar B = flattened fresh-c_last@19 vs parent STORED fp16 cx_last@19 (binds our
# convention to the parent's; same 0.995 flat bar as the parent's G2, EA.G2_COS_MIN).
G2E_EARLY_COS_MIN = 0.999
G2E_FLAT_COS_MIN = 0.995
PREFIX_CONSTANCY_COS_MIN = EA.PREFIX_CONSTANCY_COS_MIN  # 0.9999 (parent A4 convention)
# E0 pilot-throughput kill (plan §7): < 1/3 of the measured 5,471.5 tok/s basis makes
# the 30k envelope infeasible in the 4 GPU-h approval -> proportional descope with a
# hard floor. Floor arithmetic (concern earlylayer-descope-carve-underdetermined, r2):
# the estimator-validity bound binds on the POST-CARVE ridge-train rows, not n_fit —
# _e3_prep carves min(val_carve=2,000, n_fit//6) fit rows for lambda selection, so the
# floor must satisfy int(N * 0.8) - 2,000 > d_widest = 2 * max_features_in = 16,384:
# floor N = ceil((16,384 + 2,000 + 1) / 0.8) = 22,982 -> n_fit 18,385, tr 16,385 > d.
# (The prior floor 20,500 gave n_fit 16,400 but post-carve tr 14,400 < d.)
TPS_BASIS = 5471.5
TPS_KILL_FRAC = 1.0 / 3.0
PROD_MAX_FEATURES_IN = 8_192  # single source for prod_defaults max_features_in
PROD_VAL_CARVE = 2_000  # single source for prod_defaults val_carve (plan §4 E3)
D_WIDEST_DESIGN = 2 * PROD_MAX_FEATURES_IN  # ctx design: psi_mean ++ psi_last
DESCOPE_FLOOR_CONTEXTS = (5 * (D_WIDEST_DESIGN + PROD_VAL_CARVE + 1) + 3) // 4  # 22,982
RC_GATE_BE = 22  # mirrors EA.RC_GATE_B (Gate HALT class)
RC_G2E = 24  # capture-convention identity FAIL (never fit past it)
RC_THROUGHPUT = 25  # pilot-throughput kill (approval infeasible at the floor)
RC_JUDGE_FROZEN = 26  # --phase judge under the plan v17 §0.-1 judged-label freeze
# Committed split-sha literals (the PDSHRINK_COMMITTED_SPLIT_SHAS pattern): pinned from
# the git-committed eval_results/issue_1482/split_1482.json @ origin/main. E1 asserts
# the STAGED split_indices.npz pools hash to these BEFORE subsampling, so a drifted /
# regenerated scratch_meta upload fails loud (tests/test_issue1482_early_layer.py pins
# these constants to the committed file).
EARLY_COMMITTED_SPLIT_SHAS: dict[str, str] = {
    "sae_fit_sha256": "88d344675fbbca3a717cd8a0c6aa4fd893241a17098bd3173f6ae00b4d9a0fb8",
    "holdout_sha256": "7957d689748eca218055f213082c1df444603ec2f1faa3f04b4004cee6f58622",
}
HF_DATA_REPO = C.HF_DATA_REPO
SCRATCH_META_PREFIX = "issue1482_error_analysis/analysis_tensors/scratch_meta"
PARENT_STORE_PREFIX = "issue1482_error_analysis/analysis_tensors/sae_pooled"
CAPTURE_PREFIX = EA.CAPTURE_PREFIX  # parent final_token_capture (G2-e stored side)
RB_HF_PREFIX = "issue779_monitoring/r_b"  # plan §10: HF mirror @ 037fcbb
RB_HF_REVISION = "037fcbb"
RB_TRAITS = ("evil", "sycophancy", "hallucination")
# Extended-rubric byte-parity pin (plan §11 R6): sha16 of FE.JUDGE_SYSTEM_EXT as the
# feature-extremes round recorded it (extremes.json judge.rubric_sha256_system).
# KEPT UNUSED under the v17 judged-label freeze — the #1773 labelling resume
# re-engages the rubric-parity machinery against this pin.
RUBRIC_SHA16_EXTENDED = "2774598533dbdcc3"
COMMITTED_EXTREMES = PROJECT_ROOT / "eval_results/issue_1482/feature_extremes/extremes.json"
COMMITTED_ABSTRACTION = PROJECT_ROOT / "eval_results/issue_1482/feature_correlates/abstraction.json"
# fit_mlp reconciliation (consistency-checker Must-Fix; artifact-reuse check (k)):
# MAIN's issue779_ffc_n1m_fits.fit_mlp/_fit_mlp_minibatch differ from
# origin/issue-779-n1m ONLY by the ADDITIVE `capture_out=None` kwarg (default None,
# "behavior unchanged (no rng use)" per its docstring) + main-side Hub-retry hardening
# landed by #1482 r4 (ccb4de9abce47a57a48fca6aa5f9321b6151130c); the training loop,
# standardization, early stop, and defaults (w8192 via MLP_W_PROTOCOL, lr 3e-4, seed 0)
# are recipe-identical. Disposition: consume MAIN's copy; branch commits d7c1c55fbe,
# a2dd635b4d, 689f5c1042 recorded not-needed. Recorded in early_summary.json too.
FIT_MLP_RECONCILIATION = {
    "disposition": "not-needed — consume MAIN's copy",
    "diff": "additive capture_out=None kwarg only (default None; no rng use) + "
    "main-side hub.retry_transient download hardening (#1482 r4 ccb4de9abc)",
    "branch_commits_not_needed": ["d7c1c55fbe", "a2dd635b4d", "689f5c1042"],
}


# ── small utils ──────────────────────────────────────────────────────────────────


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    obj = dict(obj)
    obj.setdefault("metadata", C.reproducibility_metadata())
    C.write_json_atomic(path, obj)


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel (blocks_pipeline: false, plan §9 phase_outputs)."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        C.write_sentinel(f"earlylayer-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:  # sentinel write must never kill the run on the VM smoke
        logger.warning("[sentinel] earlylayer-%s write failed: %s", name, e)


def _record_phase_time(args, phase: str, wall_s: float) -> None:
    """Append per-phase wall time (upload2 sums them into gpu_hours_used)."""
    path = args.out_eval / "phase_times.json"
    doc = json.loads(path.read_text()) if path.exists() else {"phases": []}
    doc["phases"].append({"phase": phase, "wall_s": round(wall_s, 1)})
    C.write_json_atomic(path, doc)


def default_smoke_root(base_root: Path) -> Path:
    """ONE shared derivation for the smoke leg's out-root (writer AND reaper use
    this — the chained smoke-then-full residue gotcha, #1586 fu r3)."""
    return base_root / "early_smoke"


def reap_sibling_smoke_root(args) -> None:
    """FULL leg, first phase entry: reap the DERIVED sibling smoke root BEFORE any
    headroom preamble (fail-loud rmtree; one log line per branch; never under the
    smoke leg's own mode)."""
    assert not args.smoke, "reap_sibling_smoke_root must never run under --smoke"
    smoke_root = default_smoke_root(args.base_root)
    if args.out_root == smoke_root:
        logger.info("[reap] out_root IS the derived smoke root; skip")
        return
    if smoke_root.exists():
        shutil.rmtree(smoke_root)  # fail-loud: a failed reap must crash HERE
        logger.info("[reap] removed sibling smoke root %s", smoke_root)
    else:
        logger.info("[reap] sibling smoke root absent (%s)", smoke_root)


def gate_be_verdict(fve64: float, fve128: float) -> tuple[str, int]:
    """Gate B-e lattice (plan §7): PASS >= 0.825 (k64); WARN [0.675, 0.825) ->
    escalate to k128 if IT clears 0.825, else k64 + caveat; HALT < 0.675.
    Calibration: published-FVE offsets (parent 0.806-0.106/-0.256 applied to
    L3's published 0.93087890625) — plan §11 R5. Pure + unit-probed."""
    if fve64 >= GATE_BE_PASS:
        return "PASS", 64
    if fve64 >= GATE_BE_HALT:
        return "WARN", (128 if fve128 >= GATE_BE_PASS else 64)
    return "HALT", 64


def descope_plan(tps: float, n_total: int, val_carve: int) -> dict | None:
    """Pure E0 throughput-descope arithmetic (plan §7). Returns None when tps
    clears the 1/3-basis kill floor; else a proportional-descope dict. Raises
    SystemExit(RC_THROUGHPUT) when the descope falls below DESCOPE_FLOOR_CONTEXTS
    (the caller writes the typed sentinel before re-raising). Any RETURNED descope
    keeps the post-carve ridge-train rows n_fit - min(val_carve, n_fit // 6)
    STRICTLY > D_WIDEST_DESIGN — asserted here, so an oversized --val-carve fails
    loud instead of shipping an under-determined ridge (concern
    earlylayer-descope-carve-underdetermined). Pure + unit-probed."""
    if tps >= TPS_BASIS * TPS_KILL_FRAC:
        return None
    n_desc = int(n_total * (tps / TPS_BASIS))
    if n_desc < DESCOPE_FLOOR_CONTEXTS:
        raise SystemExit(RC_THROUGHPUT)
    n_fit = int(n_desc * 0.8)
    eff_tr = n_fit - min(val_carve, max(1, n_fit // 6))
    assert eff_tr > D_WIDEST_DESIGN, (
        f"descope n_fit={n_fit} leaves post-carve tr={eff_tr} <= d={D_WIDEST_DESIGN} "
        f"(val_carve={val_carve}; estimator-validity bound)"
    )
    return {
        "n_fit": n_fit,
        "n_score": n_desc - n_fit,
        "reason": f"tokens/s {tps:.0f} < 1/3 basis {TPS_BASIS}",
        "effective_tr": eff_tr,
    }


def _assert_estimator_validity(n_tr: int, d_widest: int, smoke: bool) -> None:
    """Production-only post-carve well-posedness backstop (concern
    earlylayer-descope-carve-underdetermined): the REALIZED ridge-train rows must
    strictly exceed the widest realized design width — catches a coverage
    shortfall or CLI-oversized --val-carve that the descope-floor arithmetic
    cannot see. A smoke run is a deliberate under-determined shape and is exempt
    (the standing estimator-validity smoke carve-out)."""
    if smoke:
        return
    assert n_tr > d_widest, (
        f"effective ridge-train rows {n_tr} <= widest design d={d_widest} "
        "(post-carve estimator validity; concern earlylayer-descope-carve-underdetermined)"
    )


def _primary_l3_perfeature(args) -> str:
    """Primary L3 per-feature source, keyed to the Gate B-e verdict's chosen_k
    persisted in early_pilot.json at E0 (concern gatebe-warn-escalation-not-threaded):
    chosen_k=64 -> perfeature_l3_default; chosen_k=128 (WARN escalation) ->
    perfeature_l3_k128, the k64 arm then serving as the robustness twin. Every
    LABEL-KEYED downstream consumer resolves through this helper (E5 tail
    selection, E6 H1 pooled rows, the hero L3 ECDF). The k64-POSITIONALLY-ALIGNED
    diagnostics stay keyed to perfeature_l3_default regardless of chosen_k —
    covariates_l3.npz / the sinkmask twin / the shuffle-null band are computed
    over the k64 f_out restriction, so a positional join against the k128 arm
    would misalign."""
    pilot = json.loads((args.out_eval / "early_pilot.json").read_text())
    k = int(pilot["chosen_k"])
    assert k in (64, 128), f"unexpected chosen_k in early_pilot.json: {k}"
    return "perfeature_l3_default" if k == 64 else "perfeature_l3_k128"


def _early_hf_prefix(args) -> str:
    """HF destination prefix for this round's store + eval artifacts."""
    leaf = "early_layer_smoke" if args.smoke else "early_layer"
    return f"issue1482_error_analysis/analysis_tensors/{leaf}"


def _stage_scratch_meta(args) -> None:
    """Stage split_indices.npz / row_ci.npy / prov.npy from the parent's HF
    scratch_meta (idempotent; the off-pod incident-class fix — these are HF-permanent)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    args.scratch.mkdir(parents=True, exist_ok=True)
    for name in ("split_indices.npz", "row_ci.npy", "prov.npy"):
        dest = args.scratch / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        hub.retry_transient(
            lambda n=name: hf_hub_download(
                HF_DATA_REPO,
                filename=f"{SCRATCH_META_PREFIX}/{n}",
                repo_type="dataset",
                local_dir=str(args.scratch / "_hf"),
            ),
            what=f"scratch_meta fetch ({name})",
        )
        src = args.scratch / "_hf" / SCRATCH_META_PREFIX / name
        shutil.copy2(src, dest)
    logger.info("[stage] scratch_meta ready under %s", args.scratch)


def _sink_token_ids(tok) -> np.ndarray:
    """Sink-robustness special-token id set (plan §4 E1): chat-template special
    tokens + role header words, resolved from the tokenizer at startup and
    asserted non-empty (single-token encodings only)."""
    ids: set[int] = set()
    for t in ("<|im_start|>", "<|im_end|>"):
        enc = tok.encode(t, add_special_tokens=False)
        assert len(enc) == 1, (t, enc)
        ids.add(int(enc[0]))
    for w in ("system", "user", "assistant"):
        enc = tok.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(int(enc[0]))
    assert ids, "sink token id set is empty"
    return np.asarray(sorted(ids), dtype=np.int64)


@torch.no_grad()
def _extra_answer_features(
    sae64, sae128, h: torch.Tensor, context_end: int, ans_ids: np.ndarray, sink_ids: np.ndarray
) -> dict:
    """Sink-masked (k64) + k128 answer trios, re-deriving the SAME reference keep
    mask as EA._row_features (bit-identical S.token_inlier_mask + BOS strip), so
    the default-vs-sink contrast is a paired read over identical rows."""
    keep = S.token_inlier_mask(h)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ans_keep = keep[context_end + 1 :]
    h_ans = h[context_end + 1 :]
    ans_all_out = bool(h_ans.shape[0] > 0 and int(ans_keep.sum()) == 0)
    kept_rows = h_ans if ans_all_out else h_ans[ans_keep]
    kept_pos = np.arange(len(ans_ids)) if ans_all_out else np.where(ans_keep.cpu().numpy())[0]
    f64 = sae64.encode(kept_rows)
    not_sink = ~np.isin(ans_ids[kept_pos], sink_ids)
    sink_n_excl = int((~not_sink).sum())
    if bool(not_sink.any()):
        rows = torch.from_numpy(np.where(not_sink)[0])
        trio_sink = S.pool_answer_features(f64[rows])
        sink_fallback = 0
    else:  # every kept answer token is a sink token: fall back to the default pool
        trio_sink = S.pool_answer_features(f64)
        sink_fallback = 1
    trio128 = S.pool_answer_features(sae128.encode(kept_rows))
    return {
        "sink": S.sparsify(trio_sink),
        "k128": S.sparsify(trio128),
        "sink_n_excl": sink_n_excl,
        "sink_fallback": sink_fallback,
    }


# ── subsample (E1 entry; also used by E0 to enumerate the row universe) ───────────


def _load_split_and_assert(args) -> dict[str, np.ndarray]:
    """Load the parent split pools + assert their shas vs the committed literals
    (runs in BOTH modes — the smoke asserts the same literals)."""
    idx = np.load(args.scratch / "split_indices.npz")
    pools = {k: np.asarray(idx[k], dtype=np.int64) for k in ("sae_fit", "sae_val", "holdout")}
    got_fit, got_hold = _sha_ids(pools["sae_fit"]), _sha_ids(pools["holdout"])
    assert got_fit == EARLY_COMMITTED_SPLIT_SHAS["sae_fit_sha256"], (
        f"sae_fit sha drift: {got_fit} != committed literal"
    )
    assert got_hold == EARLY_COMMITTED_SPLIT_SHAS["holdout_sha256"], (
        f"holdout sha drift: {got_hold} != committed literal"
    )
    logger.info("[split] parent pool shas match committed split_1482.json literals")
    return pools


def _chunk_ci_universe(args, names: list[str]) -> set[int]:
    """ci set of the given raw chunks (smoke pool restriction; downloads each
    chunk once with the parent's bounded retry + delete-after)."""
    cis: set[int] = set()
    cache = args.scratch / "raw_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        got = Path(N1M._download_chunk_with_retry(HF_DATA_REPO, f"{EA.RAW_PREFIX}/{name}", cache))
        rows = json.loads(got.read_text())["rows"]
        cis.update(int(r["ci"]) for r in rows)
        got.unlink()
    return cis


def _subsample(args) -> dict:
    """Seeded corpus-stratified subsample: S_fit ⊂ sae_fit, S_score ⊂ holdout
    (seed 14823; per-pool realized LMSYS fraction; sha-asserted pools). Under
    --max-chunks > 0 the pools are first restricted to rows whose ci lives in
    the enumerated chunks (a smoke SCALE knob — the sampling code is identical)."""
    pools = _load_split_and_assert(args)
    row_ci = np.load(args.scratch / "row_ci.npy")
    prov_u8 = np.load(args.scratch / "prov.npy")
    n_fit_req, n_score_req = args.n_fit, args.n_score
    pilot_doc = args.out_eval / "early_pilot.json"
    if pilot_doc.exists():
        pj = json.loads(pilot_doc.read_text())
        if pj.get("descope") is not None:
            n_fit_req = int(pj["descope"]["n_fit"])
            n_score_req = int(pj["descope"]["n_score"])
            logger.warning(
                "[subsample] pilot descope honored: n_fit=%d n_score=%d", n_fit_req, n_score_req
            )
    fit_pool, hold_pool = pools["sae_fit"], pools["holdout"]
    if args.max_chunks > 0:
        dns = argparse.Namespace(max_chunks=args.max_chunks)
        names = EA._raw_chunk_names(dns)
        universe = _chunk_ci_universe(args, names)
        fit_pool = fit_pool[np.isin(row_ci[fit_pool], list(universe))]
        hold_pool = hold_pool[np.isin(row_ci[hold_pool], list(universe))]
        logger.info(
            "[subsample] chunk-restricted pools (max_chunks=%d): fit %d, holdout %d",
            args.max_chunks,
            len(fit_pool),
            len(hold_pool),
        )
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    frac_fit = float((prov_u8[fit_pool] == 0).mean()) if len(fit_pool) else 0.5
    frac_hold = float((prov_u8[hold_pool] == 0).mean()) if len(hold_pool) else 0.5
    s_fit, meta_fit = EA._stratified_sample(rng, fit_pool, prov_u8, n_fit_req, frac_fit)
    s_score, meta_score = EA._stratified_sample(rng, hold_pool, prov_u8, n_score_req, frac_hold)
    # split hygiene (plan §4 E1): subset + disjointness asserts
    assert set(s_fit).isdisjoint(set(s_score)), "S_fit and S_score overlap"
    assert np.isin(s_fit, pools["sae_fit"]).all(), "S_fit escapes the parent sae_fit pool"
    assert np.isin(s_score, pools["holdout"]).all(), "S_score escapes the parent holdout pool"
    doc = {
        "subsample_seed": SUBSAMPLE_SEED,
        "s_fit": meta_fit,
        "s_score": meta_score,
        "s_fit_sha256": _sha_ids(s_fit),
        "s_score_sha256": _sha_ids(s_score),
        "parent_pool_shas": dict(EARLY_COMMITTED_SPLIT_SHAS),
        "lmsys_frac": {"sae_fit": frac_fit, "holdout": frac_hold},
        "requested": {"n_fit": n_fit_req, "n_score": n_score_req},
        "max_chunks": args.max_chunks,
    }
    args.store.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.store / "split_indices_early.npz",
        s_fit=s_fit,
        s_score=s_score,
        prov_fit=prov_u8[s_fit],
        prov_score=prov_u8[s_score],
    )
    _write_json(args.out_eval / "split_early.json", doc)
    logger.info(
        "[subsample] S_fit=%d S_score=%d (seed %d)", len(s_fit), len(s_score), SUBSAMPLE_SEED
    )
    return {"s_fit": s_fit, "s_score": s_score, "row_ci": row_ci, "prov_u8": prov_u8}


# ── E0: pilot ─────────────────────────────────────────────────────────────────────


def _accumulate_gamma(h: torch.Tensor, acc: dict) -> None:
    """Streaming per-dim mean/std accumulators for the gamma = |mu|/|sigma| read."""
    x = h.to(torch.float64)
    acc["n"] += x.shape[0]
    acc["s"] += x.sum(0)
    acc["ss"] += (x * x).sum(0)


def _gamma_of(acc: dict) -> float:
    n = max(2, acc["n"])
    mu = acc["s"] / n
    var = torch.clamp(acc["ss"] / n - mu * mu, min=0.0)
    sd_norm = float(torch.sqrt(var).norm())
    return float(mu.norm()) / max(sd_norm, 1e-12)


def _stored_cx19_for(args, chunk_name: str, cis: list[int]) -> dict[int, torch.Tensor]:
    """Parent STORED fp16 cx_last@19 for the given cis (G2-e bar-B reference),
    streamed from the matching final_token_capture chunk (delete-after)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    pt_name = chunk_name.replace(".json", ".pt")
    cache = args.scratch / "capture_cache"
    cache.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(
            HF_DATA_REPO,
            filename=f"{CAPTURE_PREFIX}/{pt_name}",
            repo_type="dataset",
            local_dir=str(cache),
        ),
        what=f"capture chunk fetch ({pt_name})",
    )
    b = torch.load(got, map_location="cpu", weights_only=True)
    col = list(b["layers"]).index(L_LATE)
    ci_pos = {int(c): i for i, c in enumerate(b["ci"])}
    out = {ci: b["cx_last"][ci_pos[ci], col, :].float() for ci in cis if ci in ci_pos}
    Path(got).unlink()
    return out


# PHASE_IDEMPOTENCY_EXEMPT: E0 re-runs unconditionally BY PLAN DESIGN — it IS the
# per-pod gate + throughput measurement (tokens/s, Gate B-e, G2-e must re-measure
# on every dispatch pod; ~0.2 h), so no output-exists resume skip is wired.
def phase_pilot(args) -> None:
    """E0: throughput + Gate B-e + hook-alignment + prefix constancy + G2-e +
    fit-kernel pilot. Gates are computed identically under --smoke but demoted
    to informational (production-n-calibrated verdicts; gotcha #1345)."""
    t0 = time.time()
    if not args.smoke:
        reap_sibling_smoke_root(args)
    EA._headroom(args.store, 2 if args.smoke else 45, "e0-pilot")
    _stage_scratch_meta(args)
    for kval in (64, 128):
        S.BatchTopKSAE.ensure_downloaded(kval, args.sae_dir, layer=L_EARLY)
    pools = _load_split_and_assert(args)
    row_ci = np.load(args.scratch / "row_ci.npy")
    union = np.sort(np.concatenate([pools["sae_fit"], pools["holdout"], pools["sae_val"]]))
    needed_ci = {int(row_ci[r]): int(r) for r in union}
    assert -1 not in needed_ci, "SAE-arm rows must be NEW rows (text-resolvable)"
    model, tok = EA._load_model_tok(args)
    prefix_chars = EA._prefix_char_len(tok)
    dns = argparse.Namespace(max_chunks=1, scratch=args.scratch)
    names = EA._raw_chunk_names(dns)
    pilot_rows: list = []
    first_chunk = names[0]
    for _, keep in EA._iter_needed_rows(dns, [first_chunk], needed_ci):
        for row_idx, ci, prompt, response in keep:
            tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            full_ids, prefix_end, context_end, n_ans, seam = tk
            pilot_rows.append((row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam))
            if len(pilot_rows) >= args.pilot_n:
                break
    assert pilot_rows, "pilot: no usable rows in the first raw chunk"
    logger.info("[e0] %d pilot rows from %s", len(pilot_rows), first_chunk)

    # (a) tokens/s at production batch shape, capturing the pilot layer set
    caps = []
    tot_tokens = 0
    t_cap = time.time()
    for s0 in range(0, len(pilot_rows), args.gen_batch):
        batch = pilot_rows[s0 : s0 + args.gen_batch]
        caps.extend(EA._batched_capture(model, tok, batch, PILOT_CAPTURE_LAYERS, args.device))
        tot_tokens += sum(len(r[2]) for r in batch)
    tps = tot_tokens / max(1e-9, time.time() - t_cap)

    # gamma accumulators per depth (covariate battery §4 E3; from BOS-stripped tokens)
    gam = {
        li: {
            "n": 0,
            "s": torch.zeros(3584, dtype=torch.float64),
            "ss": torch.zeros(3584, dtype=torch.float64),
        }
        for li in (L_EARLY, L_LATE)
    }
    for cap in caps:
        for li in (L_EARLY, L_LATE):
            _accumulate_gamma(cap[li][S.BOS_OFFSET :], gam[li])

    # (b)+(c) Gate B-e FVE/L0 at k64/k128 on h3 + hook-alignment probes on h1/h5
    fve: dict[str, dict] = {}
    for kname, kval in (("k64", 64), ("k128", 128)):
        sae = S.BatchTopKSAE.load(k=kval, device=args.device, cache_dir=args.sae_dir, layer=L_EARLY)
        fve[kname] = {}
        for li in (L_EARLY, *HOOK_PROBE_LAYERS):
            h_all = torch.cat([c[li][S.BOS_OFFSET :] for c in caps])
            v, l0, diag = sae.fve_l0(h_all)
            fve[kname][f"L{li}"] = {"fve": round(float(v), 4), "l0": round(float(l0), 2), **diag}
        del sae
    fve64_l3 = fve["k64"][f"L{L_EARLY}"]["fve"]
    fve128_l3 = fve["k128"][f"L{L_EARLY}"]["fve"]
    hook_max_at_3 = {
        k: fve[k][f"L{L_EARLY}"]["fve"] >= max(fve[k][f"L{li}"]["fve"] for li in HOOK_PROBE_LAYERS)
        for k in ("k64", "k128")
    }
    verdict, chosen_k = gate_be_verdict(fve64_l3, fve128_l3)

    # (d) prefix-end constancy at L3
    hp3 = torch.stack([caps[i][L_EARLY][pilot_rows[i][3], :] for i in range(len(caps))])
    cos_min_prefix = EA.prefix_constancy_cos_min(hp3)

    # (e) G2-e two-bar identity gate on the first 8 rows
    n_gate = min(8, len(pilot_rows))
    single = []
    for i in range(n_gate):
        single.extend(
            EA._batched_capture(model, tok, [pilot_rows[i]], GATE_EARLY_LAYERS, args.device)
        )
    early_cos = []
    for i in range(n_gate):
        for li in GATE_EARLY_LAYERS:
            a = caps[i][li][pilot_rows[i][4], :]
            b = single[i][li][pilot_rows[i][4], :]
            early_cos.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
    gate_cis = [pilot_rows[i][1] for i in range(n_gate)]
    stored = _stored_cx19_for(args, first_chunk, gate_cis)
    flat_cos = []
    for i in range(n_gate):
        ci = pilot_rows[i][1]
        if ci not in stored:
            continue
        fresh = caps[i][L_LATE][pilot_rows[i][4], :]
        flat_cos.append(float(torch.nn.functional.cosine_similarity(fresh, stored[ci], dim=0)))
    assert flat_cos, "G2-e: no pilot row found in the parent capture chunk"
    g2e_early_min = min(early_cos)
    g2e_flat_min = min(flat_cos)

    # (f) fit-kernel pilot: ONE shared-Gram eigh + ONE production-shape X^TY GEMM
    d_kernel = 2 * args.max_features_in
    dev = torch.device(args.device if args.device == "cuda" else "cpu")
    g = torch.randn(d_kernel, d_kernel, dtype=torch.float64, device=dev)
    g = g @ g.T
    t_e = time.time()
    torch.linalg.eigh(g)
    eigh_s = time.time() - t_e
    n_blk = min(N1M.RIDGE_BLOCK, max(1000, args.n_fit))
    xb = torch.randn(n_blk, d_kernel, dtype=torch.float64, device=dev)
    yb = torch.randn(n_blk, min(1024, args.max_features_out), dtype=torch.float64, device=dev)
    t_g = time.time()
    _ = xb.T @ yb
    if dev.type == "cuda":
        torch.cuda.synchronize()
    gemm_s = time.time() - t_g
    del g, xb, yb

    # throughput kill / descope arithmetic (plan §7; pure fn descope_plan carries
    # the post-carve estimator-validity floor — concern
    # earlylayer-descope-carve-underdetermined)
    descope = None
    if not args.smoke:
        try:
            descope = descope_plan(tps, args.n_fit + args.n_score, args.val_carve)
        except SystemExit:
            n_desc = int((args.n_fit + args.n_score) * (tps / TPS_BASIS))
            _sentinel(
                "throughput-halt",
                f"pilot tokens/s {tps:.0f} < {TPS_BASIS * TPS_KILL_FRAC:.0f} and descope "
                f"{n_desc} < floor {DESCOPE_FLOOR_CONTEXTS} — approval infeasible",
            )
            raise

    doc = {
        "tokens_per_s": round(tps, 1),
        "n_pilot": len(pilot_rows),
        "bos_offset": S.BOS_OFFSET,
        "outlier_norm_factor": S.OUTLIER_NORM_FACTOR,
        "layers_fve": fve,
        "gate_be": verdict,
        "chosen_k": chosen_k,
        "gate_be_thresholds": {"pass": GATE_BE_PASS, "halt": GATE_BE_HALT},
        "published_fve_l3": S.PUBLISHED_FVE_BY_LAYER[L_EARLY],
        "hook_alignment_same_layer_maximal": hook_max_at_3,
        "prefix_end_cos_min_vs_row0": round(cos_min_prefix, 6),
        "g2e_early_cos_min": round(g2e_early_min, 6),
        "g2e_flat_cos_min": round(g2e_flat_min, 6),
        "g2e_n_rows": n_gate,
        "g2e_n_stored_matched": len(flat_cos),
        "fit_kernel_pilot": {
            "d": d_kernel,
            "eigh_s": round(eigh_s, 2),
            "xty_gemm_s_per_block": round(gemm_s, 3),
            "block_rows": n_blk,
            "device": str(dev),
        },
        "gamma": {f"L{li}": round(_gamma_of(gam[li]), 3) for li in (L_EARLY, L_LATE)},
        "descope": descope,
        "tiny_model": bool(args.tiny_model),
        "smoke_demoted": bool(args.smoke),
    }
    _write_json(args.out_eval / "early_pilot.json", doc)
    logger.info(
        "[e0] tps=%.0f fve_l3 k64=%.4f k128=%.4f gate_be=%s g2e early=%.4f flat=%.4f "
        "prefix_cos=%.5f eigh=%.1fs",
        tps,
        fve64_l3,
        fve128_l3,
        verdict,
        g2e_early_min,
        g2e_flat_min,
        cos_min_prefix,
        eigh_s,
    )
    if not args.smoke:
        if g2e_early_min < G2E_EARLY_COS_MIN or g2e_flat_min < G2E_FLAT_COS_MIN:
            _sentinel(
                "g2e-halt",
                f"G2-e identity gate FAILED early={g2e_early_min:.6f} flat={g2e_flat_min:.6f}",
            )
            raise SystemExit(RC_G2E)
        if cos_min_prefix < PREFIX_CONSTANCY_COS_MIN:
            # design branch, NOT a halt (plan §7): the prefix arm runs as a full
            # mapping arm; record the branch for E3 + the clean-result.
            logger.warning(
                "[e0] prefix NOT constant at L3 (min cos %.6f) — prefix arm runs as a "
                "full mapping arm (plan §7 design branch)",
                cos_min_prefix,
            )
        if verdict == "HALT" and fve128_l3 < GATE_BE_PASS:
            _sentinel(
                "gate-be-halt",
                f"Gate B-e HALT at BOTH k (k64={fve64_l3}, k128={fve128_l3}) — SAE arm "
                "infeasible on our tokens (plan §7 kill criterion)",
            )
            raise SystemExit(RC_GATE_BE)
    _sentinel("pilot", f"E0 done: tps={tps:.0f} gate_be={verdict} chosen_k={chosen_k}")
    _record_phase_time(args, "pilot", time.time() - t0)


# ── E1: capture + encode ──────────────────────────────────────────────────────────


def _assert_store_regime(args) -> None:
    """Regime-keyed out-root (resume safety, #722 r3): every output-affecting key
    is pinned in store/regime.json; a mismatched resume fails loud."""
    regime = {
        "smoke": bool(args.smoke),
        "layer_early": L_EARLY,
        "layer_late": L_LATE,
        "subsample_seed": SUBSAMPLE_SEED,
        "n_fit": args.n_fit,
        "n_score": args.n_score,
        "max_chunks": args.max_chunks,
        "tiny_model": bool(args.tiny_model),
    }
    path = args.store / "regime.json"
    if path.exists():
        prev = json.loads(path.read_text())
        assert prev == regime, f"store regime mismatch: {prev} != {regime} (out_root reuse)"
    else:
        args.store.mkdir(parents=True, exist_ok=True)
        C.write_json_atomic(path, regime)


def phase_capture(args) -> None:
    """E1: subsample + teacher-forced capture @ (3, 19) + L3 SAE encode/pooling
    (default + sinkmask + k128) + dense c_last columns; per-chunk checkpointed."""
    t0 = time.time()
    EA._headroom(args.store, 2 if args.smoke else 25, "e1-capture")
    _stage_scratch_meta(args)
    _assert_store_regime(args)
    sub = _subsample(args)
    s_fit, s_score, row_ci = sub["s_fit"], sub["s_score"], sub["row_ci"]
    set_tag = {int(r): 1 for r in s_fit}
    set_tag.update({int(r): 0 for r in s_score})
    needed_ci = {int(row_ci[r]): r for r in set_tag}
    assert -1 not in needed_ci, "subsample rows must be NEW rows (text-resolvable)"
    for kval in (64, 128):
        S.BatchTopKSAE.ensure_downloaded(kval, args.sae_dir, layer=L_EARLY)
    model, tok = EA._load_model_tok(args)
    sae64 = S.BatchTopKSAE.load(k=64, device=args.device, cache_dir=args.sae_dir, layer=L_EARLY)
    sae128 = S.BatchTopKSAE.load(k=128, device=args.device, cache_dir=args.sae_dir, layer=L_EARLY)
    sink_ids = _sink_token_ids(tok)
    prefix_chars = EA._prefix_char_len(tok)
    dns = argparse.Namespace(max_chunks=args.max_chunks, scratch=args.scratch)
    names = EA._raw_chunk_names(dns)
    n_done = 0
    tok_count = 0
    t_loop = time.time()
    for k_chunk, (name, keep) in enumerate(EA._iter_needed_rows(dns, names, needed_ci)):
        shard_path = args.store / f"pooled_l3_{Path(name).stem}.npz"
        dense_path = args.store / f"dense_l3_{Path(name).stem}.npz"
        if shard_path.exists() and dense_path.exists():
            continue
        rows = []
        for row_idx, ci, prompt, response in keep:
            tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            full_ids, prefix_end, context_end, n_ans, seam = tk
            rows.append((row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam))
        rows.sort(key=lambda r: len(r[2]))
        rec: dict[str, list] = {
            kk: []
            for kk in (
                "row_idx",
                "ci",
                "set_tag",
                "n_ctx",
                "n_ans",
                "prefix_end",
                "seam",
                "idx_off",
                "ans_idx",
                "ans_mean",
                "ans_max",
                "ans_frac",
                "psi_off",
                "psi_idx",
                "psi_mean",
                "psil_off",
                "psil_idx",
                "psil_val",
                "h_prefix",
                "ctx_n_out",
                "ans_n_out",
                "ans_all_out",
                "sink_off",
                "sink_idx",
                "sink_mean",
                "sink_max",
                "sink_frac",
                "sink_n_excl",
                "sink_fallback",
                "k128_off",
                "k128_idx",
                "k128_mean",
                "k128_max",
                "k128_frac",
            )
        }
        dense: dict[str, list] = {"row_idx": [], "c19": [], "c3": [], "hp3": []}
        for s0 in range(0, len(rows), args.gen_batch):
            batch = rows[s0 : s0 + args.gen_batch]
            caps = EA._batched_capture(model, tok, batch, CAPTURE_LAYERS, args.device)
            for (row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam), cap in zip(
                batch, caps, strict=True
            ):
                h3, h19 = cap[L_EARLY], cap[L_LATE]
                # default-mask trio + psi: EA._row_features VERBATIM (parity by
                # construction with the parent's L19 recipe, swapped to h3).
                sp, spm, spl, ctx_n_out, ans_n_out, ans_all_out = EA._row_features(
                    sae64, h3, context_end
                )
                ans_ids = np.asarray(full_ids[context_end + 1 :], dtype=np.int64)
                extra = _extra_answer_features(sae64, sae128, h3, context_end, ans_ids, sink_ids)
                rec["row_idx"].append(row_idx)
                rec["ci"].append(ci)
                rec["set_tag"].append(set_tag[int(row_idx)])
                rec["n_ctx"].append(context_end + 1)
                rec["n_ans"].append(n_ans)
                rec["prefix_end"].append(prefix_end)
                rec["seam"].append(seam)
                rec["idx_off"].append(len(sp["idx"]))
                rec["ans_idx"].append(sp["idx"])
                rec["ans_mean"].append(sp["mean"])
                rec["ans_max"].append(sp["max"])
                rec["ans_frac"].append(sp["frac"])
                rec["psi_off"].append(len(spm["idx"]))
                rec["psi_idx"].append(spm["idx"])
                rec["psi_mean"].append(spm["mean"])
                rec["psil_off"].append(len(spl["idx"]))
                rec["psil_idx"].append(spl["idx"])
                rec["psil_val"].append(spl["last"])
                rec["h_prefix"].append(h3[prefix_end].numpy().astype(np.float16))
                rec["ctx_n_out"].append(ctx_n_out)
                rec["ans_n_out"].append(ans_n_out)
                rec["ans_all_out"].append(ans_all_out)
                rec["sink_off"].append(len(extra["sink"]["idx"]))
                rec["sink_idx"].append(extra["sink"]["idx"])
                rec["sink_mean"].append(extra["sink"]["mean"])
                rec["sink_max"].append(extra["sink"]["max"])
                rec["sink_frac"].append(extra["sink"]["frac"])
                rec["sink_n_excl"].append(extra["sink_n_excl"])
                rec["sink_fallback"].append(extra["sink_fallback"])
                rec["k128_off"].append(len(extra["k128"]["idx"]))
                rec["k128_idx"].append(extra["k128"]["idx"])
                rec["k128_mean"].append(extra["k128"]["mean"])
                rec["k128_max"].append(extra["k128"]["max"])
                rec["k128_frac"].append(extra["k128"]["frac"])
                dense["row_idx"].append(row_idx)
                dense["c19"].append(h19[context_end].numpy().astype(np.float16))
                dense["c3"].append(h3[context_end].numpy().astype(np.float16))
                dense["hp3"].append(h3[prefix_end].numpy().astype(np.float16))
                tok_count += len(full_ids)
        int_keys = {
            "row_idx": np.int64,
            "ci": np.int64,
            "set_tag": np.int8,
            "n_ctx": np.int32,
            "n_ans": np.int32,
            "prefix_end": np.int32,
            "seam": np.int8,
            "idx_off": np.int64,
            "psi_off": np.int64,
            "psil_off": np.int64,
            "ctx_n_out": np.int16,
            "ans_n_out": np.int16,
            "ans_all_out": np.int8,
            "sink_off": np.int64,
            "sink_n_excl": np.int16,
            "sink_fallback": np.int8,
            "k128_off": np.int64,
        }
        arrays: dict[str, np.ndarray] = {}
        for kk, vals in rec.items():
            if kk in int_keys:
                arrays[kk] = np.asarray(vals, int_keys[kk])
            elif kk == "h_prefix":
                arrays[kk] = np.stack(vals) if vals else np.empty((0, 3584), np.float16)
            elif kk.endswith("_idx"):
                arrays[kk] = np.concatenate(vals) if vals else np.empty(0, np.int32)
            else:
                arrays[kk] = np.concatenate(vals) if vals else np.empty(0, np.float16)
        tmp = shard_path.parent / f".tmp_{shard_path.name}"
        np.savez(tmp, **arrays)
        tmp.replace(shard_path)
        dtmp = dense_path.parent / f".tmp_{dense_path.name}"
        np.savez(
            dtmp,
            row_idx=np.asarray(dense["row_idx"], np.int64),
            c19=np.stack(dense["c19"]) if dense["c19"] else np.empty((0, 3584), np.float16),
            c3=np.stack(dense["c3"]) if dense["c3"] else np.empty((0, 3584), np.float16),
            hp3=np.stack(dense["hp3"]) if dense["hp3"] else np.empty((0, 3584), np.float16),
        )
        dtmp.replace(dense_path)
        n_done += len(rec["row_idx"])
        el = time.time() - t_loop
        print(
            f"[e1] unit {k_chunk + 1}/{len(names)} chunk={Path(name).stem} "
            f"rows={len(rec['row_idx'])} total={n_done} tok={tok_count} elapsed={el:.0f}s",
            flush=True,
        )
    logger.info("[e1] capture done: %d contexts, %d tokens", n_done, tok_count)
    _sentinel("capture", f"E1 done ({n_done} contexts, {tok_count} tokens)")
    _record_phase_time(args, "capture", time.time() - t0)


# ── E2 / E4: uploads ─────────────────────────────────────────────────────────────


def phase_upload1(args) -> None:
    """E2: L3 pooled store + split doc -> HF BEFORE any fit (#825 rule).
    delete_local=False — E3 consumes the local store."""
    t0 = time.time()
    if args.skip_upload:
        logger.warning("[e2] --skip-upload: store upload SKIPPED (local-only run)")
        _sentinel("upload1", "E2 SKIPPED (--skip-upload)")
        return
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    prefix = _early_hf_prefix(args)
    res = upload_dir_sharded(
        args.store,
        HF_DATA_REPO,
        f"{prefix}/store",
        repo_type="dataset",
        shard_glob="*.npz",
        verify=True,
        delete_local=False,
    )
    from explore_persona_space.orchestrate import hub

    hub._upload(
        args.store / "regime.json",
        HF_DATA_REPO,
        "dataset",
        f"{prefix}/store/regime.json",
        upload_as_file=True,
        raise_on_error=True,
    )
    logger.info(
        "[e2] store upload done: %d shards -> %s (rerouted=%s)",
        len(res.uploaded),
        prefix,
        res.rerouted,
    )
    _sentinel("upload1", f"E2 done ({len(res.uploaded)} shards -> {prefix}/store)")
    _record_phase_time(args, "upload1", time.time() - t0)


def phase_upload2(args) -> None:
    """E4: eval outputs -> HF (text/JSON+small npz, unconditional) + the poller
    results sentinel (terminal pod phase)."""
    t0 = time.time()
    from explore_persona_space.orchestrate import hub

    prefix = _early_hf_prefix(args)
    if not args.skip_upload:
        # ONE upload_folder commit for the whole eval dir (judge/ subdir rides
        # along) — never a per-file upload loop (the #664 504-storm class);
        # exact-set verify against the Hub after.
        hub._upload(args.out_eval, HF_DATA_REPO, "dataset", f"{prefix}/eval", raise_on_error=True)
        expected = [
            f"{prefix}/eval/{p.relative_to(args.out_eval)}"
            for p in sorted(args.out_eval.rglob("*"))
            if p.is_file() and p.suffix in (".json", ".npz")
        ]
        from huggingface_hub import HfApi

        missing = hub.verify_repo_paths_uploaded(
            HfApi(), HF_DATA_REPO, expected, path_in_repo=f"{prefix}/eval", repo_type="dataset"
        )
        assert not missing, f"E4 verify: missing on Hub: {missing}"
        logger.info("[e4] %d eval artifacts verified on Hub under %s/eval", len(expected), prefix)
    else:
        logger.warning("[e4] --skip-upload: eval upload SKIPPED")
    _results_sentinel(args)
    _sentinel("upload2", "E4 done (eval artifacts uploaded; results sentinel written)")
    _record_phase_time(args, "upload2", time.time() - t0)


def _results_sentinel(args, logs_dir: Path | None = None) -> None:
    """poll_pipeline.py results sentinel (SKILL.md Step 7 contract): eval_numbers,
    eval_paths, STRUCTURED reproducibility_card (no training -> wandb rows N/A).

    Under --smoke the kind is epm:smoke-result (the issue1482_kresample.py
    precedent) so a --full pod run's SMOKE leg can never be drained by the
    poller as the real epm:results hours before the production leg completes
    (the #1586 chained-legs class, sentinel side)."""
    if logs_dir is None:
        logs_dir = Path("/workspace/logs")
        if not logs_dir.is_dir():
            logs_dir = PROJECT_ROOT / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((args.out_eval / "early_pilot.json").read_text())
    summary_path = args.out_eval / "early_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    split = json.loads((args.out_eval / "split_early.json").read_text())
    times = json.loads((args.out_eval / "phase_times.json").read_text())
    gpu_h = sum(p["wall_s"] for p in times["phases"]) / 3600.0
    payload = {
        "sentinel_schema_version": C.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:smoke-result" if args.smoke else "epm:results",
        "version": 1,
        "task_id": TASK_ID,
        "by": "issue1482_early_layer",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("SMOKE leg — not the production result. " if args.smoke else "")
        + "issue-1482 early-layer-arm pod phases E0-E4 complete "
        "(E5-evidence + E6 mechanical analysis run off-pod; judged labelling "
        "FROZEN per plan v17 §0.-1 — deferred to #1773)",
        "eval_numbers": {
            "gate_be": pilot["gate_be"],
            "chosen_k": pilot["chosen_k"],
            "fve_l3_k64": pilot["layers_fve"]["k64"][f"L{L_EARLY}"]["fve"],
            "fve_l3_k128": pilot["layers_fve"]["k128"][f"L{L_EARLY}"]["fve"],
            "g2e_early_cos_min": pilot["g2e_early_cos_min"],
            "g2e_flat_cos_min": pilot["g2e_flat_cos_min"],
            "tokens_per_s": pilot["tokens_per_s"],
            "arm_pooled_r2": summary.get("pooled_r2", {}),
            "n_fit_realized": summary.get("n_rows", {}).get("tr"),
            "n_score_realized": summary.get("n_rows", {}).get("te"),
        },
        "eval_paths": {
            "pilot": str(args.out_eval / "early_pilot.json"),
            "split": str(args.out_eval / "split_early.json"),
            "summary": str(summary_path),
            "perfeature": str(args.out_eval),
            "store_hf_prefix": f"{_early_hf_prefix(args)}/store",
            # declared OFF-POD destinations (plan v17 §6.5) — produced AFTER this
            # sentinel by the VM-side E5-evidence / E6 phases, committed to git
            "evidence_offpod_dest": str(args.out_eval / "evidence" / "evidence.json"),
            "mech_tests_offpod_dest": str(args.out_eval / "mech_tests.json"),
        },
        "reproducibility_card": {
            **C.reproducibility_metadata(),
            "layer_early": L_EARLY,
            "layer_late_control": L_LATE,
            "sae_repo": S.SAE_REPO,
            "sae_revision": S.SAE_REVISION,
            "subsample_seed": SUBSAMPLE_SEED,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "s_fit_sha256": split["s_fit_sha256"],
            "s_score_sha256": split["s_score_sha256"],
            "fit_mlp_reconciliation": FIT_MLP_RECONCILIATION,
            "wandb": "N/A — no training in this round (frozen teacher-forced forwards "
            "+ closed-form/MLP fits logging to JSON checkpoints)",
        },
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/"
        f"{_early_hf_prefix(args)}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": C.reproducibility_metadata()["git_commit"],
        "gpu_hours_used": round(gpu_h, 2),
        "gpu_hours_budgeted": 4,
        "plan_deviations": summary.get("plan_deviations", []),
    }
    path = logs_dir / f"issue-{TASK_ID}-results.json"
    C.write_json_atomic(path, payload)
    logger.info("Wrote results sentinel %s", path)


# ── E3: fits + covariates ─────────────────────────────────────────────────────────


def _activity_counts_rows(
    parts, key_idx: str, key_off: str, rows_set: set[int], dict_size: int
) -> tuple[np.ndarray, int]:
    """Per-feature active-row counts over an explicit row set (row-filtered
    sibling of EA._activity_counts — needed because the matched-n L19 control
    counts over OUR 24k subsample, not the parent's full 120k tag)."""
    counts = np.zeros(dict_size, dtype=np.int64)
    n = 0
    for part in parts:
        offs = np.concatenate([[0], np.cumsum(part[key_off])])
        for i, r in enumerate(part["row_idx"]):
            if int(r) not in rows_set:
                continue
            n += 1
            counts[part[key_idx][offs[i] : offs[i + 1]].astype(np.int64)] += 1
    return counts, n


def _restrict(counts: np.ndarray, n_fit: int, cap: int) -> np.ndarray:
    """Feature restriction: >= 1%-of-fit activity floor, then top-``cap`` by
    count (parent recipe — EA._p3_prep verbatim arithmetic)."""
    floor = max(1, int(np.ceil(0.01 * n_fit)))
    f = np.where(counts >= floor)[0]
    if len(f) > cap:
        f = f[np.argsort(-counts[f])[:cap]]
    return np.sort(f)


def _consistency_rows(parts, rows_set: set[int], dict_size: int) -> np.ndarray:
    """Within-answer consistency: mean ans_frac conditional on answer-active over
    the fit rows (the feature-correlates recipe verbatim, row-filtered)."""
    s = np.zeros(dict_size, dtype=np.float64)
    c = np.zeros(dict_size, dtype=np.int64)
    for part in parts:
        offs = np.concatenate([[0], np.cumsum(part["idx_off"])])
        for i, r in enumerate(part["row_idx"]):
            if int(r) not in rows_set:
                continue
            sl = slice(offs[i], offs[i + 1])
            idx = part["ans_idx"][sl].astype(np.int64)
            np.add.at(s, idx, part["ans_frac"][sl].astype(np.float64))
            np.add.at(c, idx, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _dense_design(dense_parts, key: str, n_rows: int, row_pos: dict) -> np.ndarray:
    """(n_rows, 3584) fp32 design from the dense_l3_* sibling shards."""
    M = np.zeros((n_rows, 3584), dtype=np.float32)
    hit = np.zeros(n_rows, dtype=bool)
    for part in dense_parts:
        vals = part[key]
        for i, r in enumerate(part["row_idx"]):
            pos = row_pos.get(int(r))
            if pos is None:
                continue
            M[pos] = vals[i].astype(np.float32)
            hit[pos] = True
    assert hit.all(), f"dense design {key}: {int((~hit).sum())} rows unfilled"
    return M


def _stage_parent_store(args) -> Path:
    """Stage the parent L19 pooled store (matched-n control input). Production:
    the whole prefix via stage_hub_prefix (#833 recipe); smoke: the first
    --parent-shards files only (they cover the smoke's chunk universe)."""
    from explore_persona_space.orchestrate import hub

    dest = args.scratch / "parent_store"
    if args.parent_shards <= 0:
        hub.stage_hub_prefix(HF_DATA_REPO, PARENT_STORE_PREFIX, dest, repo_type="dataset")
        return dest / PARENT_STORE_PREFIX
    from huggingface_hub import HfApi

    files = hub.list_hf_files_under_path(
        HfApi(), HF_DATA_REPO, PARENT_STORE_PREFIX, repo_type="dataset"
    )
    for f in sorted(files)[: args.parent_shards]:
        hub.stage_hub_file(HF_DATA_REPO, f, dest / f, repo_type="dataset")
    return dest / PARENT_STORE_PREFIX


def _e3_prep(args) -> argparse.Namespace:
    """Shared E3 preprocessing: both stores loaded, ONE matched row registry
    (rows present in OUR L3 store AND the parent L19 store — exact matched-row
    depth contrast), the seeded 2,000-row lambda carve of S_fit, and the
    per-depth feature restrictions."""
    parts = [
        dict(np.load(p, allow_pickle=False)) for p in sorted(args.store.glob("pooled_l3_*.npz"))
    ]
    assert parts, f"no L3 pooled shards under {args.store}"
    dense_parts = [
        dict(np.load(p, allow_pickle=False)) for p in sorted(args.store.glob("dense_l3_*.npz"))
    ]
    sub = np.load(args.store / "split_indices_early.npz")
    s_fit, s_score = sub["s_fit"], sub["s_score"]
    prov_by_row = {int(r): int(p) for r, p in zip(sub["s_fit"], sub["prov_fit"], strict=True)}
    prov_by_row.update(
        {int(r): int(p) for r, p in zip(sub["s_score"], sub["prov_score"], strict=True)}
    )
    have_l3: set[int] = set()
    for part in parts:
        have_l3.update(int(r) for r in part["row_idx"])
    parent_dir = _stage_parent_store(args)
    parent_parts = [
        dict(np.load(p, allow_pickle=False)) for p in sorted(parent_dir.glob("pooled_*.npz"))
    ]
    assert parent_parts, f"no parent shards under {parent_dir}"
    have_19: set[int] = set()
    for part in parent_parts:
        have_19.update(int(r) for r in part["row_idx"])
    both = have_l3 & have_19
    order = np.asarray([r for r in np.concatenate([s_fit, s_score]) if int(r) in both], np.int64)
    cov_l3 = len([r for r in s_fit if int(r) in have_l3]) / max(1, len(s_fit))
    cov_both = len(order) / max(1, len(s_fit) + len(s_score))
    logger.info(
        "[e3] row registry: %d rows (L3 coverage %.4f; both-store %.4f)",
        len(order),
        cov_l3,
        cov_both,
    )
    deviations = []
    if not args.smoke and cov_both < 0.98:
        deviations.append(f"matched-row registry covers {cov_both:.4f} of the subsample (<0.98)")
    row_pos = {int(r): i for i, r in enumerate(order)}
    n_rows = len(order)
    fit_positions = np.asarray([row_pos[int(r)] for r in s_fit if int(r) in row_pos], np.int64)
    te = np.asarray([row_pos[int(r)] for r in s_score if int(r) in row_pos], np.int64)
    assert len(te) >= 2, f"score rows after intersection: {len(te)} < 2"
    # lambda carve (plan §4 E3): seeded permutation of the fit rows; the carve is
    # val, the rest train — 'exactly as the parent's SAE arm did on its fit carve'.
    carve = min(args.val_carve, max(1, len(fit_positions) // 6))
    perm = np.random.default_rng(SUBSAMPLE_SEED).permutation(len(fit_positions))
    va = fit_positions[perm[:carve]]
    tr = fit_positions[perm[carve:]]
    assert len(tr) >= 1 and len(va) >= 1, (len(tr), len(va))
    fit_rows_set = {int(order[i]) for i in tr} | {int(order[i]) for i in va}
    # per-depth feature restrictions (1% floor + caps; counts over the FIT rows)
    out_counts, n_fit_l3 = _activity_counts_rows(
        parts, "ans_idx", "idx_off", fit_rows_set, S.DICT_SIZE
    )
    in_counts, _ = _activity_counts_rows(parts, "psi_idx", "psi_off", fit_rows_set, S.DICT_SIZE)
    k128_counts, _ = _activity_counts_rows(parts, "k128_idx", "k128_off", fit_rows_set, S.DICT_SIZE)
    out_counts19, _ = _activity_counts_rows(
        parent_parts, "ans_idx", "idx_off", fit_rows_set, S.DICT_SIZE
    )
    in_counts19, _ = _activity_counts_rows(
        parent_parts, "psi_idx", "psi_off", fit_rows_set, S.DICT_SIZE
    )
    f_out = _restrict(out_counts, n_fit_l3, args.max_features_out)
    f_in = _restrict(in_counts, n_fit_l3, args.max_features_in)
    f_out128 = _restrict(k128_counts, n_fit_l3, args.max_features_out)
    f_out19 = _restrict(out_counts19, n_fit_l3, args.max_features_out)
    f_in19 = _restrict(in_counts19, n_fit_l3, args.max_features_in)
    for nm, f in (
        ("f_out", f_out),
        ("f_in", f_in),
        ("f_out128", f_out128),
        ("f_out19", f_out19),
        ("f_in19", f_in19),
    ):
        assert len(f) >= 1, f"{nm} empty after restriction"
    # post-carve well-posedness on the REALIZED widths (descope-floor backstop:
    # covers coverage shortfall + CLI val_carve overrides the floor cannot see)
    _assert_estimator_validity(len(tr), 2 * max(len(f_in), len(f_in19)), args.smoke)
    logger.info(
        "[e3] restrictions: f_out=%d f_in=%d f_out128=%d f_out19=%d f_in19=%d (n_fit=%d)",
        len(f_out),
        len(f_in),
        len(f_out128),
        len(f_out19),
        len(f_in19),
        n_fit_l3,
    )
    te_prov = np.asarray([prov_by_row[int(order[i])] for i in te], np.int8)
    return argparse.Namespace(
        parts=parts,
        dense_parts=dense_parts,
        parent_parts=parent_parts,
        order=order,
        row_pos=row_pos,
        n_rows=n_rows,
        tr=tr,
        va=va,
        te=te,
        te_prov=te_prov,
        fit_rows_set=fit_rows_set,
        n_fit=n_fit_l3,
        f_out=f_out,
        f_in=f_in,
        f_out128=f_out128,
        f_out19=f_out19,
        f_in19=f_in19,
        out_counts=out_counts,
        out_counts19=out_counts19,
        deviations=deviations,
    )


def _per_feature_npz(
    args,
    name: str,
    feat_ids: np.ndarray,
    pt: np.ndarray,
    true_te: np.ndarray,
    activity: np.ndarray,
    te_prov: np.ndarray,
    te: np.ndarray,
) -> dict:
    """Write one per-feature npz (R2/rho/ss_tot + per-corpus R2 + split-half rank
    stability) and return its summary dict."""
    pf = EA._per_feature_metrics(pt, true_te)
    perm = EA._splithalf_perm(len(te))
    ia, ib = perm[: len(te) // 2], perm[len(te) // 2 :]
    pa = EA._per_feature_metrics(pt[ia], true_te[ia])
    pb = EA._per_feature_metrics(pt[ib], true_te[ib])
    ok = np.isfinite(pa["r2"]) & np.isfinite(pb["r2"])
    if ok.sum() >= 3:
        ra = EA._midrank(pa["r2"][ok][:, None])[:, 0]
        rb = EA._midrank(pb["r2"][ok][:, None])[:, 0]
        stab = float(np.corrcoef(ra, rb)[0, 1])
    else:
        stab = float("nan")
    corpus = {}
    for label, code in (("lmsys", 0), ("wildchat", 1)):
        m = te_prov == code
        if int(m.sum()) >= 2:
            corpus[label] = EA._per_feature_metrics(pt[m], true_te[m])["r2"]
        else:
            corpus[label] = np.full(pt.shape[1], np.nan)
    np.savez(
        args.out_eval / f"{name}.npz",
        feat_ids=feat_ids,
        r2=pf["r2"],
        spearman=pf["spearman"],
        ss_tot=pf["ss_tot"],
        activity=activity,
        r2_lmsys=corpus["lmsys"],
        r2_wildchat=corpus["wildchat"],
    )
    pooled = float(PR._pooled_r2(pt, true_te))
    return {
        "pooled_r2": pooled,
        "splithalf_rank_stability": stab,
        "n_features": int(len(feat_ids)),
        "median_r2": float(np.nanmedian(pf["r2"])),
    }


def _knn_reads(pt: np.ndarray, true_te: np.ndarray) -> dict:
    """Standing kNN-retrieval read, euclidean + cosine (chance = k/n_pool stated
    by the helper)."""
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    return {m: knn_retrieval(pt, true_te, ks=(1, 5, 10), metric=m) for m in ("euclidean", "cosine")}


def phase_fits(args) -> None:
    """E3: 5 shared-Gram ridge designs + MLP twin + shuffle-null + covariates +
    mapping baselines; sequential units with output-exists resume skips."""
    t0 = time.time()
    EA._headroom(args.scratch, 2 if args.smoke else 15, "e3-fits")
    prep = _e3_prep(args)
    dev = torch.device(args.device)
    summary: dict = {
        "pooled_r2": {},
        "selected_lambda": {},
        "splithalf": {},
        "knn": {},
        "baselines": {},
        "n_rows": {"tr": int(len(prep.tr)), "va": int(len(prep.va)), "te": int(len(prep.te))},
        "n_features": {
            "f_out": int(len(prep.f_out)),
            "f_in": int(len(prep.f_in)),
            "f_out128": int(len(prep.f_out128)),
            "f_out19": int(len(prep.f_out19)),
            "f_in19": int(len(prep.f_in19)),
        },
        "fit_mlp_reconciliation": FIT_MLP_RECONCILIATION,
        "plan_deviations": list(prep.deviations),
    }
    act_l3 = prep.out_counts[prep.f_out] / max(1, prep.n_fit)
    act_l3_128 = None  # activity for k128 features (from its own counts)
    act_19 = prep.out_counts19[prep.f_out19] / max(1, prep.n_fit)

    def _targets_from(parts, feat_ids, keys: tuple[str, str, str]) -> np.ndarray:
        key_idx, key_off, key_val = keys
        return EA._densify(parts, key_idx, key_off, key_val, feat_ids, prep.n_rows, prep.row_pos)

    # unit 1: L3 context-features design (arm 1 + sinkmask twin + k128 twin share
    # ONE factorization — identical feature ids + identical score rows for the
    # registered paired contrast, plan §3)
    if not (args.out_eval / "perfeature_l3_default.npz").exists():
        psi_mean = _targets_from(prep.parts, prep.f_in, ("psi_idx", "psi_off", "psi_mean"))
        psi_last = _targets_from(prep.parts, prep.f_in, ("psil_idx", "psil_off", "psil_val"))
        z_ctx = np.concatenate([psi_mean, psi_last], axis=1)
        tgts = {
            "mean": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_mean")),
            "max": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_max")),
            "frac": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_frac")),
            "sink_mean": _targets_from(
                prep.parts, prep.f_out, ("sink_idx", "sink_off", "sink_mean")
            ),
            "sink_max": _targets_from(prep.parts, prep.f_out, ("sink_idx", "sink_off", "sink_max")),
            "sink_frac": _targets_from(
                prep.parts, prep.f_out, ("sink_idx", "sink_off", "sink_frac")
            ),
            "k128_mean": _targets_from(
                prep.parts, prep.f_out128, ("k128_idx", "k128_off", "k128_mean")
            ),
        }
        preds = EA._shared_gram_ridge_multi(
            z_ctx, tgts, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
        )
        name_of = {
            "mean": "perfeature_l3_default",
            "max": "perfeature_l3_default_max",
            "frac": "perfeature_l3_default_frac",
            "sink_mean": "perfeature_l3_sinkmask",
            "sink_max": "perfeature_l3_sinkmask_max",
            "sink_frac": "perfeature_l3_sinkmask_frac",
            "k128_mean": "perfeature_l3_k128",
        }
        k128_counts_act, _ = _activity_counts_rows(
            prep.parts, "k128_idx", "k128_off", prep.fit_rows_set, S.DICT_SIZE
        )
        act_l3_128 = k128_counts_act[prep.f_out128] / max(1, prep.n_fit)
        for tname, (pt, meta) in preds.items():
            fid = prep.f_out128 if tname == "k128_mean" else prep.f_out
            act = act_l3_128 if tname == "k128_mean" else act_l3
            doc = _per_feature_npz(
                args, name_of[tname], fid, pt, tgts[tname][prep.te], act, prep.te_prov, prep.te
            )
            summary["pooled_r2"][name_of[tname]] = doc["pooled_r2"]
            summary["selected_lambda"][name_of[tname]] = meta["selected_lambda"]
            summary["splithalf"][name_of[tname]] = doc["splithalf_rank_stability"]
            if tname == "mean":
                summary["knn"]["l3_sae_ctx"] = _knn_reads(pt, tgts["mean"][prep.te])
                # standing identity+learned-bias baseline on the ALIGNED feature-id
                # intersection (psi_mean -> ans_mean on shared ids); the full-design
                # identity is inapplicable across feature-id coordinate systems and
                # the dense arms' 3,584 -> F mismatch is stated inapplicable.
                from explore_persona_space.analysis.mapping_baselines import (
                    identity_bias_predict,
                )

                shared = np.intersect1d(prep.f_in, prep.f_out)
                if len(shared) >= 2:
                    xi = psi_mean[:, np.searchsorted(prep.f_in, shared)]
                    yi = tgts["mean"][:, np.searchsorted(prep.f_out, shared)]
                    pred_i = identity_bias_predict(xi[prep.tr], yi[prep.tr], xi[prep.te])
                    summary["baselines"]["l3_identity_bias"] = {
                        "n_shared_ids": int(len(shared)),
                        "pooled_r2": float(PR._pooled_r2(pred_i, yi[prep.te])),
                        "knn": _knn_reads(pred_i, yi[prep.te]),
                    }
                else:
                    summary["baselines"]["l3_identity_bias"] = {
                        "n_shared_ids": int(len(shared)),
                        "note": "aligned intersection too small",
                    }
                summary["baselines"]["dense_arms_identity"] = (
                    "inapplicable — 3,584-dim input vs feature-space output (stated per the "
                    "standing mapping-baselines rule)"
                )
        # shuffle-null K=20 at the pinned mean lambda (ONE factorization + 20
        # X^TY GEMMs; permute answer rows within the train pool, score te on
        # true pairs — parent recipe verbatim)
        lam = float(summary["selected_lambda"]["perfeature_l3_default"])
        y_mean = tgts["mean"]
        fac = N1M._ridge_factorize(z_ctx, y_mean, prep.tr, dev, N1M.RIDGE_BLOCK)
        u, s_eig = fac["U"], fac["s_eig"]
        xmu, xsd, ymu = fac["xmu"], fac["xsd"], fac["ymu"]
        null_r2 = np.zeros((len(SHUFFLE_SEEDS), len(prep.f_out)), dtype=np.float16)
        true_te = y_mean[prep.te]
        for si, seed in enumerate(SHUFFLE_SEEDS):
            rng = np.random.default_rng(seed)
            tr_perm = prep.tr[rng.permutation(len(prep.tr))]
            xty = torch.zeros((z_ctx.shape[1], y_mean.shape[1]), dtype=torch.float64, device=dev)
            for s0 in range(0, len(prep.tr), N1M.RIDGE_BLOCK):
                xb = (
                    torch.as_tensor(
                        z_ctx[prep.tr[s0 : s0 + N1M.RIDGE_BLOCK]], dtype=torch.float64, device=dev
                    )
                    - xmu
                ) / xsd
                yb = (
                    torch.as_tensor(
                        y_mean[tr_perm[s0 : s0 + N1M.RIDGE_BLOCK]], dtype=torch.float64, device=dev
                    )
                    - ymu
                )
                xty += xb.T @ yb
            w = u @ ((u.T @ xty) / (s_eig + lam)[:, None])
            en = (torch.as_tensor(z_ctx[prep.te], dtype=torch.float64, device=dev) - xmu) / xsd
            pt_null = ((en @ w) + ymu).cpu().numpy()
            null_r2[si] = EA._per_feature_metrics(pt_null, true_te)["r2"].astype(np.float16)
            print(f"[e3-null] draw {si + 1}/{len(SHUFFLE_SEEDS)} seed={seed}", flush=True)
        np.savez(
            args.out_eval / "shuffle_null_l3.npz",
            feat_ids=prep.f_out,
            r2=null_r2,
            seeds=np.asarray(SHUFFLE_SEEDS, np.int64),
            selected_lambda=np.float64(lam),
        )
        del z_ctx, tgts, preds, fac, u, s_eig
        _write_json(args.out_eval / "early_summary.json", summary)
        print("[e3] unit 1/5 ctx design done", flush=True)
    else:
        summary = json.loads((args.out_eval / "early_summary.json").read_text())
        logger.info("[e3] unit 1 outputs exist; resume-skip")

    # unit 2+3: L3 dense-input design (ridge trio + MLP w8192 mean twin)
    if not (args.out_eval / "perfeature_l3_dense_in.npz").exists():
        z3 = _dense_design(prep.dense_parts, "c3", prep.n_rows, prep.row_pos)
        tgts = {
            "mean": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_mean")),
            "max": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_max")),
            "frac": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_frac")),
        }
        preds = EA._shared_gram_ridge_multi(
            z3, tgts, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
        )
        for tname, (pt, meta) in preds.items():
            name = "perfeature_l3_dense_in" + ("" if tname == "mean" else f"_{tname}")
            doc = _per_feature_npz(
                args, name, prep.f_out, pt, tgts[tname][prep.te], act_l3, prep.te_prov, prep.te
            )
            summary["pooled_r2"][name] = doc["pooled_r2"]
            summary["selected_lambda"][name] = meta["selected_lambda"]
            if tname == "mean":
                summary["knn"]["l3_dense_in"] = _knn_reads(pt, tgts["mean"][prep.te])
        # MLP twin (parent recipe: w8192, lr 3e-4, seed 0, minibatched AdamW +
        # internal-val early stop — EA._p3_unit_mlp call shape verbatim)
        pt_mlp, meta_mlp = N1M._fit_mlp_minibatch(
            z3,
            tgts["mean"],
            prep.tr,
            prep.te,
            width=N1M.MLP_W_PROTOCOL,
            lr=MLP_LR,
            max_epochs=3 if args.smoke else F.MLP_MAX_EPOCHS,
            batch=min(N1M.MLP_BATCH, max(8, len(prep.tr))),
            seed=args.seed,
            dev=dev,
        )
        doc = _per_feature_npz(
            args,
            "perfeature_l3_dense_in_mlp",
            prep.f_out,
            pt_mlp,
            tgts["mean"][prep.te],
            act_l3,
            prep.te_prov,
            prep.te,
        )
        summary["pooled_r2"]["perfeature_l3_dense_in_mlp"] = doc["pooled_r2"]
        summary["mlp_epochs_ran"] = meta_mlp["epochs_ran"]
        summary["knn"]["l3_dense_in_mlp"] = _knn_reads(pt_mlp, tgts["mean"][prep.te])
        del z3, tgts, preds
        _write_json(args.out_eval / "early_summary.json", summary)
        print("[e3] unit 2/5 dense3 (+MLP) done", flush=True)
    else:
        logger.info("[e3] unit 2 outputs exist; resume-skip")

    # unit 4: L3 prefix-arm null (the prefix mapping arm; degenerate-constant
    # prefix expected on this single-turn corpus — E0 gate (d))
    if not (args.out_eval / "perfeature_l3_prefix_null.npz").exists():
        hp3 = _dense_design(prep.dense_parts, "hp3", prep.n_rows, prep.row_pos)
        tgt = {"mean": _targets_from(prep.parts, prep.f_out, ("ans_idx", "idx_off", "ans_mean"))}
        preds = EA._shared_gram_ridge_multi(
            hp3, tgt, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
        )
        pt, meta = preds["mean"]
        doc = _per_feature_npz(
            args,
            "perfeature_l3_prefix_null",
            prep.f_out,
            pt,
            tgt["mean"][prep.te],
            act_l3,
            prep.te_prov,
            prep.te,
        )
        summary["pooled_r2"]["perfeature_l3_prefix_null"] = doc["pooled_r2"]
        summary["selected_lambda"]["perfeature_l3_prefix_null"] = meta["selected_lambda"]
        summary["knn"]["l3_prefix_null"] = _knn_reads(pt, tgt["mean"][prep.te])
        del hp3, tgt, preds
        _write_json(args.out_eval / "early_summary.json", summary)
        print("[e3] unit 3/5 prefix null done", flush=True)
    else:
        logger.info("[e3] unit 3 outputs exist; resume-skip")

    # unit 5: matched-n L19 controls (parent store ctx arm + fresh-c19 dense arm)
    if not (args.out_eval / "perfeature_l19_matched_ctx.npz").exists():
        psi_mean19 = _targets_from(
            prep.parent_parts, prep.f_in19, ("psi_idx", "psi_off", "psi_mean")
        )
        psi_last19 = _targets_from(
            prep.parent_parts, prep.f_in19, ("psil_idx", "psil_off", "psil_val")
        )
        z19 = np.concatenate([psi_mean19, psi_last19], axis=1)
        tgts19 = {
            "mean": _targets_from(
                prep.parent_parts, prep.f_out19, ("ans_idx", "idx_off", "ans_mean")
            ),
            "max": _targets_from(
                prep.parent_parts, prep.f_out19, ("ans_idx", "idx_off", "ans_max")
            ),
            "frac": _targets_from(
                prep.parent_parts, prep.f_out19, ("ans_idx", "idx_off", "ans_frac")
            ),
        }
        preds = EA._shared_gram_ridge_multi(
            z19, tgts19, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
        )
        for tname, (pt, meta) in preds.items():
            name = "perfeature_l19_matched_ctx" + ("" if tname == "mean" else f"_{tname}")
            doc = _per_feature_npz(
                args, name, prep.f_out19, pt, tgts19[tname][prep.te], act_19, prep.te_prov, prep.te
            )
            summary["pooled_r2"][name] = doc["pooled_r2"]
            summary["selected_lambda"][name] = meta["selected_lambda"]
            if tname == "mean":
                summary["knn"]["l19_matched_ctx"] = _knn_reads(pt, tgts19["mean"][prep.te])
                from explore_persona_space.analysis.mapping_baselines import (
                    identity_bias_predict,
                )

                shared = np.intersect1d(prep.f_in19, prep.f_out19)
                if len(shared) >= 2:
                    xi = psi_mean19[:, np.searchsorted(prep.f_in19, shared)]
                    yi = tgts19["mean"][:, np.searchsorted(prep.f_out19, shared)]
                    pred_i = identity_bias_predict(xi[prep.tr], yi[prep.tr], xi[prep.te])
                    summary["baselines"]["l19_identity_bias"] = {
                        "n_shared_ids": int(len(shared)),
                        "pooled_r2": float(PR._pooled_r2(pred_i, yi[prep.te])),
                        "knn": _knn_reads(pred_i, yi[prep.te]),
                    }
        c19 = _dense_design(prep.dense_parts, "c19", prep.n_rows, prep.row_pos)
        preds = EA._shared_gram_ridge_multi(
            c19,
            {"mean": tgts19["mean"], "max": tgts19["max"], "frac": tgts19["frac"]},
            prep.tr,
            prep.va,
            prep.te,
            N1M.LAMBDAS_N1M,
            dev,
            N1M.RIDGE_BLOCK,
        )
        for tname, (pt, meta) in preds.items():
            name = "perfeature_l19_matched_dense" + ("" if tname == "mean" else f"_{tname}")
            doc = _per_feature_npz(
                args, name, prep.f_out19, pt, tgts19[tname][prep.te], act_19, prep.te_prov, prep.te
            )
            summary["pooled_r2"][name] = doc["pooled_r2"]
            summary["selected_lambda"][name] = meta["selected_lambda"]
            if tname == "mean":
                summary["knn"]["l19_matched_dense"] = _knn_reads(pt, tgts19["mean"][prep.te])
        del z19, psi_mean19, psi_last19, tgts19, preds, c19
        _write_json(args.out_eval / "early_summary.json", summary)
        print("[e3] unit 4/5 matched-L19 done", flush=True)
    else:
        logger.info("[e3] unit 4 outputs exist; resume-skip")

    _covariates_unit(args, prep, summary)
    _write_json(args.out_eval / "early_summary.json", summary)
    _sentinel("fits", "E3 done (5 designs + MLP + shuffle-null + covariates)")
    _record_phase_time(args, "fits", time.time() - t0)


# ── E3 covariates ─────────────────────────────────────────────────────────────────


def _load_rb_rows(args, layer: int) -> tuple[np.ndarray, list[str]]:
    """(n_traits, 3584) row-``layer`` persona directions (the #779 monitoring r_B
    bank; FE._load_rb_layer semantics with a layer argument). Local files
    preferred; staged from the HF mirror (plan §10 pin) on the git-clone-only
    lanes where data/ does not travel."""
    rows, names = [], []
    for trait in RB_TRAITS:
        path = PROJECT_ROOT / "data/issue_779/r_b" / f"{trait}.pt"
        if not path.exists():
            from explore_persona_space.orchestrate import hub

            path = hub.stage_hub_file(
                HF_DATA_REPO,
                f"{RB_HF_PREFIX}/{trait}.pt",
                args.scratch / "r_b" / f"{trait}.pt",
                repo_type="dataset",
                revision=RB_HF_REVISION,
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        arr = np.asarray(payload["r_b"].detach().cpu().numpy(), dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[1] == 3584, arr.shape
        layers = [int(x) for x in payload["layers"]]
        assert layers[layer] == layer, (trait, layer, layers[layer])
        assert payload.get("trait") == trait, (payload.get("trait"), trait)
        assert payload.get("smoke") is False, f"{trait}: r_B is a SMOKE artifact"
        rows.append(arr[layer])
        names.append(trait)
    return np.stack(rows, axis=0), names


def _footprint(w_u: torch.Tensor, g: torch.Tensor, dec: torch.Tensor, dev) -> dict:
    """Direct-logit footprint (plan §4 E3, 2606.08365 family): l_f = W_U^T (g ⊙ d_f),
    mean-centered over the vocab; concentration = ||top-20 |l_f|||_2 / ||l_f||_2.
    Chunked GEMM <= 2,048 features per chunk."""
    n_f = dec.shape[1]
    conc = np.zeros(n_f, dtype=np.float64)
    norm = np.zeros(n_f, dtype=np.float64)
    w = w_u.to(device=dev, dtype=torch.float32)
    gg = g.to(device=dev, dtype=torch.float32)
    for s0 in range(0, n_f, 2048):
        d_chunk = dec[:, s0 : s0 + 2048].to(device=dev, dtype=torch.float32) * gg[:, None]
        logit = w @ d_chunk  # (V, c)
        logit = logit - logit.mean(0, keepdim=True)
        full = logit.norm(dim=0)
        top = torch.topk(logit.abs(), k=min(20, logit.shape[0]), dim=0).values
        conc[s0 : s0 + 2048] = (top.norm(dim=0) / torch.clamp(full, min=1e-12)).cpu().numpy()
        norm[s0 : s0 + 2048] = full.cpu().numpy()
    return {"conc": conc, "norm": norm}


def _coactivation(parts, feat_ids: np.ndarray, rows_set: set[int], dev) -> np.ndarray:
    """coact(f) = max_{g != f} C[f,g]/C[f,f] with C = A^T A over binary row-level
    answer-activity on the fit rows — ONE GEMM (plan §4 E3)."""
    col_of = np.full(int(feat_ids.max()) + 1, -1, dtype=np.int64)
    col_of[feat_ids] = np.arange(len(feat_ids))
    rows_a = []
    for part in parts:
        offs = np.concatenate([[0], np.cumsum(part["idx_off"])])
        for i, r in enumerate(part["row_idx"]):
            if int(r) not in rows_set:
                continue
            fidx = part["ans_idx"][offs[i] : offs[i + 1]].astype(np.int64)
            keep = fidx < len(col_of)
            cols = col_of[fidx[keep]]
            cols = cols[cols >= 0]
            row = np.zeros(len(feat_ids), dtype=np.float16)
            row[cols] = 1.0
            rows_a.append(row)
    a = torch.as_tensor(np.stack(rows_a), dtype=torch.float32, device=dev)
    c = (a.T @ a).cpu().numpy()
    diag = np.diag(c).copy()
    np.fill_diagonal(c, -np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        coact = np.where(diag > 0, c.max(axis=1) / np.maximum(diag, 1), np.nan)
    return coact


def _covariates_unit(args, prep, summary: dict) -> None:
    """Covariate battery at L3 AND (recomputed at matched rows) L19 (plan §4 E3):
    activity, consistency, direct-logit footprint, co-activation, dense flag,
    r_B decoder alignment raw + population-centered."""
    if (args.out_eval / "covariates_l3.npz").exists():
        logger.info("[e3] covariates exist; resume-skip")
        return
    dev = torch.device(args.device)
    model, _tok = EA._load_model_tok(args)
    w_u = model.lm_head.weight.detach()
    g = model.model.norm.weight.detach()
    pilot = json.loads((args.out_eval / "early_pilot.json").read_text())
    for depth, parts, f_out, counts in (
        (L_EARLY, prep.parts, prep.f_out, prep.out_counts),
        (L_LATE, prep.parent_parts, prep.f_out19, prep.out_counts19),
    ):
        sae = S.BatchTopKSAE.load(k=64, device="cpu", cache_dir=args.sae_dir, layer=depth)
        dec = sae.w_dec[:, torch.as_tensor(f_out)]
        fp = _footprint(w_u, g, dec, dev)
        coact = _coactivation(parts, f_out, prep.fit_rows_set, dev)
        consistency = _consistency_rows(parts, prep.fit_rows_set, S.DICT_SIZE)[f_out]
        activity = counts[f_out] / max(1, prep.n_fit)
        dense_flag = (activity > 0.5).astype(np.int8)
        decile = np.searchsorted(
            np.quantile(activity, np.linspace(0, 1, 11)[1:-1]), activity, side="right"
        )
        top_decile = (decile == 9).astype(np.int8)
        rb, traits = _load_rb_rows(args, depth)
        d_np = dec.numpy().astype(np.float64)
        d_norm = d_np / np.maximum(np.linalg.norm(d_np, axis=0, keepdims=True), 1e-12)
        rb_norm = rb / np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-12)
        raw_cos = np.abs(rb_norm @ d_norm).max(axis=0)
        d_cent = d_np - d_np.mean(axis=1, keepdims=True)
        d_cent /= np.maximum(np.linalg.norm(d_cent, axis=0, keepdims=True), 1e-12)
        cent_cos = np.abs(rb_norm @ d_cent).max(axis=0)
        np.savez(
            args.out_eval / f"covariates_l{depth}.npz",
            feat_ids=f_out,
            activity=activity,
            consistency=consistency,
            footprint_conc=fp["conc"],
            footprint_norm=fp["norm"],
            coact=coact,
            dense_flag=dense_flag,
            top_decile_flag=top_decile,
            rb_raw_maxabs=raw_cos,
            rb_centered_maxabs=cent_cos,
        )
        summary.setdefault("covariates", {})[f"L{depth}"] = {
            "n_features": int(len(f_out)),
            "n_dense_flag": int(dense_flag.sum()),
            "gamma": pilot["gamma"].get(f"L{depth}"),
            "rb_traits": traits,
        }
        del sae, dec
        print(f"[e3] covariates L{depth} done ({len(f_out)} features)", flush=True)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()


# ── E5-evidence (off-pod VM; judged labelling FROZEN — plan v17 §0.-1) ─────────────


def _select_tails(com: dict[str, np.ndarray], n_tail: int, n_decile_tail: int) -> dict:
    """Smoke-sized tail selection (parameterized FE._select clone — production
    calls FE._select VERBATIM and this function is used only when the smoke's
    feature count cannot satisfy the production per-decile floor; #1345
    gate-calibration convention)."""
    import issue1482_feature_correlates as FC

    r2, act, fid = com["r2"], com["activity"], com["feat_ids"]
    dec = FC._decile_of(act)
    order = np.argsort(r2, kind="stable")
    a_worst, a_best = order[:n_tail], order[-n_tail:][::-1]
    b_best_parts, b_worst_parts = [], []
    for d in range(FC.N_DECILES):
        ind = np.where(dec == d)[0]
        take = min(n_decile_tail, max(0, len(ind) // 2))
        if take == 0:
            continue
        o = ind[np.argsort(r2[ind], kind="stable")]
        b_worst_parts.append(o[:take])
        b_best_parts.append(o[-take:][::-1])
    b_best = np.concatenate(b_best_parts) if b_best_parts else np.empty(0, np.int64)
    b_worst = np.concatenate(b_worst_parts) if b_worst_parts else np.empty(0, np.int64)
    union = np.unique(np.concatenate([a_best, a_worst, b_best, b_worst]))
    rows = [
        {
            "feat_id": int(fid[i]),
            "restricted_idx": int(i),
            "r2": float(r2[i]),
            "activity": float(act[i]),
            "decile": int(dec[i]),
            "a_best": bool(i in set(a_best.tolist())),
            "a_worst": bool(i in set(a_worst.tolist())),
            "b_best": bool(i in set(b_best.tolist())),
            "b_worst": bool(i in set(b_worst.tolist())),
        }
        for i in union.tolist()
    ]
    return {
        "n_tail": n_tail,
        "n_decile_tail": n_decile_tail,
        "n_union": int(len(union)),
        "idx": {
            "a_best": a_best.tolist(),
            "a_worst": a_worst.tolist(),
            "b_best": b_best.tolist(),
            "b_worst": b_worst.tolist(),
            "union": union.tolist(),
        },
        "features": rows,
    }


def _evidence_scan(
    shard_paths: list[Path],
    union_fids: list[int],
    feat_ids: np.ndarray,
    keys: tuple[str, str, str],
) -> tuple[dict[str, list], np.ndarray, np.ndarray]:
    """ONE streaming pass over the pooled shards (FIT rows, set_tag == 1):
    (a) top-8 firing answers per union feature by the PRIMARY arm's pooled max
        ranking -> {str(fid): [[val, ci, row_idx], ...]} — ids into the parent
        raw_completions chunks; answer TEXT is never read or stored (v17 freeze);
    (b) the binary row-level answer-activity matrix over the primary arm's
        restricted features (fp16; the E3 co-activation matrix semantics).
    ``keys`` = (offset, index, value) shard keys — (idx_off, ans_idx, ans_max)
    for k=64, (k128_off, k128_idx, k128_max) under Gate B-e escalation, so the
    ranking + neighbours always match the SELECTED dictionary.
    Returns (top, a (n_fit_rows, n_feat) fp16, col_of id->column map)."""
    import heapq

    import issue1482_feature_correlates as FC

    key_off, key_idx, key_val = keys
    fids = np.asarray(feat_ids, np.int64)
    col_of = np.full(int(fids.max()) + 1, -1, dtype=np.int64)
    col_of[fids] = np.arange(len(fids))
    union = {int(f) for f in union_fids}
    heaps: dict[int, list[tuple[float, int, int]]] = {f: [] for f in union}
    rows_a: list[np.ndarray] = []
    for k_sh, p in enumerate(sorted(shard_paths)):
        part = dict(np.load(p, allow_pickle=False))
        offs = np.concatenate([[0], np.cumsum(part[key_off])])
        for i, r in enumerate(part["row_idx"]):
            if int(part["set_tag"][i]) != 1:  # evidence + coact from FIT rows only
                continue
            sl = slice(offs[i], offs[i + 1])
            fidx = part[key_idx][sl].astype(np.int64)
            keep = fidx < len(col_of)
            cols = col_of[fidx[keep]]
            cols = cols[cols >= 0]
            row = np.zeros(len(fids), dtype=np.float16)
            row[cols] = 1.0
            rows_a.append(row)
            vals = part[key_val][sl].astype(np.float64)
            ci = int(part["ci"][i])
            for f, v in zip(fidx, vals, strict=True):
                fi = int(f)
                if fi in union:
                    item = (float(v), ci, int(r))
                    h = heaps[fi]
                    if len(h) < FC.TOP_K_CONTEXTS:
                        heapq.heappush(h, item)
                    else:
                        heapq.heappushpop(h, item)
        if (k_sh + 1) % 200 == 0:
            print(f"[e5-scan] shard {k_sh + 1}/{len(shard_paths)}", flush=True)
    top = {
        str(f): [[v, ci, r] for v, ci, r in sorted(h, key=lambda t: -t[0])]
        for f, h in heaps.items()
    }
    a = np.stack(rows_a) if rows_a else np.zeros((0, len(fids)), np.float16)
    return top, a, col_of


def _coact_topk(
    a: np.ndarray, feat_ids: np.ndarray, ucols: np.ndarray, topk: int = 10
) -> list[list[list]]:
    """Top-``topk`` co-activation neighbours per union feature (plan v17 §0.-1
    clause (b); the E3 matrix semantics — C = A^T A over binary fit-row answer
    activity, self excluded, neighbours drawn from the primary arm's restricted
    feature set). Per union column: [[neighbour_fid, co_count, co_frac], ...]
    with co_frac = C[f,g] / max(C[f,f], 1). Row-chunked GEMM (fp32)."""
    n_u, n_feat = len(ucols), a.shape[1]
    c_u = np.zeros((n_u, n_feat), dtype=np.float32)
    for s0 in range(0, a.shape[0], 4096):
        blk = a[s0 : s0 + 4096].astype(np.float32)
        c_u += blk[:, ucols].T @ blk
    out: list[list[list]] = []
    for j, uc in enumerate(ucols):
        diag = float(c_u[j, uc])
        row = c_u[j].copy()
        row[uc] = -np.inf
        k = min(topk, n_feat - 1)
        if k <= 0:
            out.append([])
            continue
        nb = np.argpartition(-row, k - 1)[:k]
        nb = nb[np.argsort(-row[nb], kind="stable")]
        out.append([[int(feat_ids[g]), int(row[g]), float(row[g] / max(diag, 1.0))] for g in nb])
    return out


def _load_wu_norm(args) -> tuple[torch.Tensor, torch.Tensor, object]:
    """(W_U, final RMSNorm weight, tokenizer) for the footprint-token evidence.

    --tiny-model (smoke carve-out): rebuilds the SAME from-config model the smoke
    capture used (EA._load_model_tok, seed 0). Production (off-pod VM): selective
    safetensors fetch of lm_head.weight + model.norm.weight ONLY — ~1-2 GB RSS vs
    ~15 GB for a full CPU model load (E5-evidence is 0-GPU / shared-VM by
    contract, so the full-model load is deliberately avoided)."""
    if args.tiny_model:
        model, tok = EA._load_model_tok(args)
        return model.lm_head.weight.detach(), model.model.norm.weight.detach(), tok
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from explore_persona_space.orchestrate import hub

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = hub.retry_transient(
        lambda: AutoTokenizer.from_pretrained(model_id), what=f"tokenizer fetch ({model_id})"
    )
    idx_path = hub.retry_transient(
        lambda: hf_hub_download(model_id, "model.safetensors.index.json"),
        what="safetensors index fetch",
    )
    wmap = json.loads(Path(idx_path).read_text())["weight_map"]
    # Qwen2.5-7B-Instruct is untied (lm_head.weight present); the embed fallback
    # covers a tied checkpoint, where W_U IS the embedding matrix.
    wu_name = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    tensors: dict[str, torch.Tensor] = {}
    for name in (wu_name, "model.norm.weight"):
        assert name in wmap, f"{name} missing from the safetensors weight map"
        shard = hub.retry_transient(
            lambda n=name: hf_hub_download(model_id, wmap[n]),
            what=f"safetensors shard fetch ({wmap[name]})",
        )
        with safe_open(shard, framework="pt", device="cpu") as f:
            tensors[name] = f.get_tensor(name)
    return tensors[wu_name], tensors["model.norm.weight"], tok


def _footprint_tokens(w_u, g, dec_u, tok, topk: int = 20) -> list[list[list]]:
    """Top-``topk`` direct-logit footprint tokens per union feature (plan v17
    §0.-1 clause (c); the E3 footprint formula l_f = W_U (g ⊙ d_f) mean-centered
    over the vocab — computed here on the PRIMARY dictionary's selected decoders
    so the tokens match the to-be-labelled features under either chosen_k).
    Per feature: [[token_id, token_str, centered_logit], ...] ranked by |logit|."""
    w = w_u.to(dtype=torch.float32)
    gg = g.to(dtype=torch.float32)
    logit = w @ (dec_u.to(dtype=torch.float32) * gg[:, None])  # (V, n_union)
    logit = logit - logit.mean(0, keepdim=True)
    k = min(topk, logit.shape[0])
    _vals, ids = torch.topk(logit.abs(), k=k, dim=0)
    out: list[list[list]] = []
    for j in range(logit.shape[1]):
        out.append([[int(t), tok.decode([int(t)]), float(logit[t, j])] for t in ids[:, j].tolist()])
    return out


def _evidence_entries(
    sel: dict,
    top: dict[str, list],
    neighbors: list[list[list]],
    fp_tokens: list[list[list]],
    union_sorted: list[int],
    cov_by_fid: dict[int, dict] | None,
    rank_of: dict[int, int],
    n_ranked: int,
    source_arm: str,
    chosen_k: int,
) -> dict[str, dict]:
    """Assemble the per-feature evidence packet (plan v17 §0.-1 clauses (a)-(e)).
    Pure function — unit-pinned by tests/test_issue1482_early_layer.py; the
    required keys per feature are {selection, top_answers, coact_neighbors,
    footprint_tokens, covariates}."""
    u_pos = {int(f): j for j, f in enumerate(union_sorted)}
    out: dict[str, dict] = {}
    for row in sel["features"]:
        fid = int(row["feat_id"])
        j = u_pos[fid]
        if cov_by_fid is None:
            cov: dict = {
                "note": f"chosen_k={chosen_k} — the covariate battery is k64-based; "
                "no dictionary-aligned row for an escalated-arm feature"
            }
        else:
            cov = cov_by_fid[fid]
        out[str(fid)] = {
            "selection": {
                "set_a_best": bool(row["a_best"]),
                "set_a_worst": bool(row["a_worst"]),
                "set_b_best": bool(row["b_best"]),
                "set_b_worst": bool(row["b_worst"]),
                "decile": int(row["decile"]),
                "r2": float(row["r2"]),
                "activity": float(row["activity"]),
                "r2_rank_asc": int(rank_of[fid]),
                "n_ranked": int(n_ranked),
                "source_arm": source_arm,
                "chosen_k": int(chosen_k),
            },
            "top_answers": top.get(str(fid), []),
            "coact_neighbors": neighbors[j],
            "footprint_tokens": fp_tokens[j],
            "covariates": cov,
        }
    return out


def _neuronpedia_early(args, union_fids: set[int]) -> dict:
    """Attempt the layer-3 Neuronpedia auto-interp source (A9): naming pattern
    ``3-resid-post-aa``. On absence the judge runs on top-firing answers alone
    and the instrument deviation is STATED (returned in the summary).

    RETAINED UNUSED under the v17 judged-label freeze — aux auto-interp evidence
    belongs to the #1773 labelling resume, which re-engages this helper."""
    import gzip
    import re
    import urllib.error
    import urllib.parse

    import issue1482_feature_extremes as FE

    prefix = f"v1/{FE.NP_MODEL_ID}/3-resid-post-aa/explanations/"
    keys: list[str] = []
    token = ""
    try:
        while True:
            q = f"list-type=2&prefix={prefix}&max-keys=1000"
            if token:
                q += f"&continuation-token={urllib.parse.quote(token)}"
            body = FE._http_get(f"{FE.NP_S3}?{q}").decode()
            keys.extend(re.findall(r"<Key>([^<]+)</Key>", body))
            if "<IsTruncated>true" not in body:
                break
            tok = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
            if not tok:
                break
            token = tok[0]
    except (urllib.error.HTTPError, RuntimeError) as e:
        logger.warning("[e5] neuronpedia listing failed (%s) — proceeding without aux", e)
        keys = []
    keys = sorted(k for k in keys if k.endswith(".jsonl.gz"))
    if not keys:
        return {
            "available": False,
            "note": "A9: no layer-3 auto-interp source — judge runs "
            "on top-firing answers alone (instrument deviation stated)",
        }
    cache = args.work / "np_cache_l3"
    cache.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    for key in keys:
        dest = cache / key.rsplit("/", 1)[-1]
        if not (dest.exists() and dest.stat().st_size > 0):
            blob = FE._http_get(f"{FE.NP_S3}/{urllib.parse.quote(key)}")
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(blob)
            tmp.replace(dest)
        for line in gzip.decompress(dest.read_bytes()).decode("utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = int(rec["index"])
            if idx in union_fids:
                found[str(idx)] = {"description": (rec.get("description") or "").strip()}
    (args.work / "neuronpedia_explanations.json").write_text(json.dumps(found, indent=1))
    return {"available": True, "n_resolved": len(found), "n_batches": len(keys)}


def phase_judge(args) -> None:
    """FROZEN (plan v17 §0.-1; user directive epm:progress v101, 2026-07-28):
    judged feature-labelling is paused on this task until #1773 delivers its
    validated auto-interp instrument. NO judge calls may be dispatched — not the
    ~360 items, not the 60 retest, not in smokes. Hard exit so a stale
    invocation can never silently judge; run --phase evidence to (re)persist the
    labelling inputs instead."""
    del args  # nothing may run under this phase
    logger.error(
        "[e5] --phase judge is FROZEN (plan v17 §0.-1 judged-label freeze): judged "
        "labelling is deferred to #1773's validated instrument; run --phase evidence "
        "to persist the labelling inputs instead"
    )
    raise SystemExit(RC_JUDGE_FROZEN)


def phase_evidence(args) -> None:
    """E5-evidence (off-pod VM, 0 GPU, ZERO judge calls — plan v17 §0.-1 freeze):
    mechanical tail selection (FE._select VERBATIM at production shape) + the
    per-feature evidence packet, persisted so #1773's validated instrument can
    label these features later with NO GPU re-run — the directive's clause (d),
    the one leg that must not be skipped."""
    t0 = time.time()
    import issue1482_feature_extremes as FE

    args.work.mkdir(parents=True, exist_ok=True)
    # stage the L3 store from HF when absent locally (VM production run; the
    # off_pod_phases contract — every read is in the pod upload set)
    if not list(args.store.glob("pooled_l3_*.npz")):
        from explore_persona_space.orchestrate import hub

        prefix = _early_hf_prefix(args)
        hub.stage_hub_prefix(HF_DATA_REPO, f"{prefix}/store", args.scratch / "store_dl")
        staged = args.scratch / "store_dl" / prefix / "store"
        args.store.mkdir(parents=True, exist_ok=True)
        for p in staged.glob("*.npz"):
            shutil.copy2(p, args.store / p.name)
    # primary source keyed to the Gate B-e chosen_k (WARN escalation -> k128;
    # concern gatebe-warn-escalation-not-threaded)
    pilot = json.loads((args.out_eval / "early_pilot.json").read_text())
    chosen_k = int(pilot["chosen_k"])
    primary_name = _primary_l3_perfeature(args)
    logger.info("[e5] primary per-feature source: %s (chosen_k=%d)", primary_name, chosen_k)
    z = np.load(args.out_eval / f"{primary_name}.npz")
    com = {
        "feat_ids": np.asarray(z["feat_ids"], np.int64),
        "r2": np.asarray(z["r2"], np.float64),
        "activity": np.asarray(z["activity"], np.float64),
    }
    finite = np.isfinite(com["r2"])
    if not finite.all():
        logger.warning("[e5] %d non-finite per-feature R2 rows excluded", int((~finite).sum()))
        com = {k: v[finite] for k, v in com.items()}
    if args.smoke:
        sel = _select_tails(com, n_tail=3, n_decile_tail=1)
    else:
        sel = FE._select(com)  # instrument parity by construction (extremes recipe)
    (args.work / "selection.json").write_text(json.dumps(sel, indent=1))
    union_sorted = sorted(int(com["feat_ids"][i]) for i in sel["idx"]["union"])
    logger.info("[e5] union size %d", len(union_sorted))
    # ascending R2 rank over the finite selection universe (0 = worst-predicted)
    asc = np.argsort(np.argsort(com["r2"], kind="stable"), kind="stable")
    rank_of = {int(f): int(asc[i]) for i, f in enumerate(com["feat_ids"])}

    keys = (
        ("idx_off", "ans_idx", "ans_max")
        if chosen_k == 64
        else ("k128_off", "k128_idx", "k128_max")
    )
    shard_paths = sorted(args.store.glob("pooled_l3_*.npz"))
    assert shard_paths, f"no L3 pooled shards under {args.store} (stage from HF first)"
    top, a, col_of = _evidence_scan(shard_paths, union_sorted, com["feat_ids"], keys)
    # the #1773 resume input for FE.phase_texts (its 2-element [[val, ci], ...]
    # contract is preserved verbatim; row_idx is evidence-packet-only)
    (args.work / "sample_top_contexts.json").write_text(
        json.dumps({f: [[v, ci] for v, ci, _r in lst] for f, lst in top.items()})
    )
    ucols = col_of[np.asarray(union_sorted, np.int64)]
    assert (ucols >= 0).all(), "selected feature missing from the primary restriction"
    neighbors = _coact_topk(a, com["feat_ids"], ucols)
    del a

    sae = S.BatchTopKSAE.load(k=chosen_k, device="cpu", cache_dir=args.sae_dir, layer=L_EARLY)
    dec_u = sae.w_dec[:, torch.as_tensor(union_sorted, dtype=torch.long)].detach()
    w_u, g, tok = _load_wu_norm(args)
    fp_tokens = _footprint_tokens(w_u, g, dec_u, tok)
    del sae, dec_u, w_u

    cov_by_fid: dict[int, dict] | None = None
    if chosen_k == 64:  # the covariate battery is k64-based (E3 convention)
        cz = np.load(args.out_eval / "covariates_l3.npz")
        pos = {int(f): i for i, f in enumerate(np.asarray(cz["feat_ids"], np.int64))}
        cov_by_fid = {}
        for fid in union_sorted:
            i = pos[fid]
            cov_by_fid[fid] = {
                "activity": float(cz["activity"][i]),
                "consistency": float(cz["consistency"][i]),
                "footprint_conc": float(cz["footprint_conc"][i]),
                "footprint_norm": float(cz["footprint_norm"][i]),
                "coact": float(cz["coact"][i]),
                "dense_flag": int(cz["dense_flag"][i]),
                "top_decile_flag": int(cz["top_decile_flag"][i]),
                "rb_raw_maxabs": float(cz["rb_raw_maxabs"][i]),
                "rb_centered_maxabs": float(cz["rb_centered_maxabs"][i]),
            }
    features = _evidence_entries(
        sel,
        top,
        neighbors,
        fp_tokens,
        union_sorted,
        cov_by_fid,
        rank_of,
        n_ranked=int(len(com["feat_ids"])),
        source_arm=primary_name,
        chosen_k=chosen_k,
    )
    doc = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "round": "early-layer-arm",
        "layer": L_EARLY,
        "freeze": "judged feature-labelling FROZEN this round (plan v17 §0.-1; user "
        "directive epm:progress v101, 2026-07-28) — labels are DEFERRED to #1773's "
        "validated auto-interp instrument; this packet is the labelling input",
        "label_schema_note": "#1773's label schema uses the THREE-WAY speaker_property "
        "field (language / register_style / identity_disposition) per the v98 "
        "persona-field retraction — NOT the retracted binary persona_related field",
        "perfeature_source": primary_name,
        "chosen_k": chosen_k,
        "selection": {
            "recipe": (
                "smoke _select_tails(n_tail=3, n_decile_tail=1)"
                if args.smoke
                else "FE._select (extremes recipe VERBATIM: Set A 150+150 global, "
                "Set B decile-matched 15+15 x 10, deduped)"
            ),
            "n_union": int(sel["n_union"]),
            "n_tail": int(sel["n_tail"]),
            "n_decile_tail": int(sel["n_decile_tail"]),
        },
        "l19": {
            "selected": False,
            "note": "no fresh L19 tail selection this round — the committed L19 judged "
            "sets (extremes union + correlates sample) already carry labels",
        },
        "top_answers_note": "[pooled max activation, ci, row_idx] — ids into the parent "
        "raw_completions chunks (HF, parent pin); answer TEXT is not stored here",
        "reproducibility": C.reproducibility_metadata(),
        "features": features,
    }
    ev_dir = args.out_eval / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    _write_json(ev_dir / "evidence.json", doc)
    logger.info("[e5] evidence packet: %d features -> %s", len(features), ev_dir / "evidence.json")
    _record_phase_time(args, "evidence", time.time() - t0)


# ── E6: analysis + figures (off-pod VM) ───────────────────────────────────────────


def _load_labels(args) -> dict | None:
    """#1773-resume judged labels (--labels-file; plan v17 §0.-1): accepts the
    labels.json envelope ({"labels": {fid: {...}}}) or a bare fid->record map.
    None when the flag is absent (the deferred default this round)."""
    if args.labels_file is None:
        return None
    doc = json.loads(Path(args.labels_file).read_text())
    labels = doc.get("labels", doc) if isinstance(doc, dict) else None
    assert isinstance(labels, dict) and labels, f"empty/malformed labels file: {args.labels_file}"
    return labels


def _pooled_h1_rows(args, labels: dict) -> list[dict]:
    """Pooled judged-feature rows: the resume-round L3 labels + the committed L19
    judged sets (extremes union 358 + correlates sample 300, deduped by feat_id
    preferring extremes). Each row: {feat_id, depth, level, r2}."""
    rows: list[dict] = []
    # label-keyed join -> the PRIMARY arm (chosen_k); labels were E5-selected on
    # the same arm, so joining any other npz would silently drop/mismap rows
    z = np.load(args.out_eval / f"{_primary_l3_perfeature(args)}.npz")
    r2_of = {int(f): float(r) for f, r in zip(z["feat_ids"], z["r2"], strict=True)}
    for fid_s, lab in labels.items():
        fid = int(fid_s)
        if lab["level"] in ("low", "high") and fid in r2_of and np.isfinite(r2_of[fid]):
            rows.append({"feat_id": fid, "depth": 3, "level": lab["level"], "r2": r2_of[fid]})
    seen19: set[int] = set()
    ext = json.loads(COMMITTED_EXTREMES.read_text())
    for r in ext["features"]:
        if r.get("level") in ("low", "high"):
            rows.append(
                {
                    "feat_id": int(r["feat_id"]),
                    "depth": 19,
                    "level": r["level"],
                    "r2": float(r["r2"]),
                }
            )
            seen19.add(int(r["feat_id"]))
    ab = json.loads(COMMITTED_ABSTRACTION.read_text())
    for r in ab["features"]:
        fid = int(r["feat_id"])
        if fid not in seen19 and r.get("level") in ("low", "high"):
            rows.append({"feat_id": fid, "depth": 19, "level": r["level"], "r2": float(r["r2"])})
    return rows


def _h1_depth_stratified(rows: list[dict], n_perm: int, rng: np.random.Generator) -> dict:
    """H1 (registered, plan §3): pooled Spearman(level, R2) with labels shuffled
    WITHIN depth (binary level -> rank(level) is affine in the label, so the
    permuted statistic is one batched GEMM per depth block; midrank r2)."""
    depth = np.asarray([r["depth"] for r in rows])
    lev = np.asarray([1.0 if r["level"] == "high" else 0.0 for r in rows])
    r2 = np.asarray([r["r2"] for r in rows], dtype=np.float64)
    rr = EA._midrank(r2[:, None])[:, 0]

    def _spear(labels: np.ndarray) -> float:
        a = labels - labels.mean()
        b = rr - rr.mean()
        den = float(np.sqrt((a**2).sum() * (b**2).sum()))
        return float((a * b).sum() / den) if den > 1e-12 else float("nan")

    obs = _spear(lev)
    perm_lab = np.tile(lev, (n_perm, 1))
    for d in np.unique(depth):
        m = np.where(depth == d)[0]
        keys = rng.random((n_perm, len(m)))
        order = np.argsort(keys, axis=1)
        perm_lab[:, m] = lev[m][order]
    a = perm_lab - perm_lab.mean(axis=1, keepdims=True)
    b = rr - rr.mean()
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = (a @ b) / den
    ok = np.isfinite(stats)
    lo, hi = np.percentile(stats[ok], [2.5, 97.5])
    if obs > hi:
        verdict = "level-positive"
    elif obs < lo:
        verdict = "level-inverted"
    else:
        verdict = "null-persists"
    return {
        "observed_pooled_spearman": obs,
        "perm_band_2p5_97p5": [float(lo), float(hi)],
        "n_perm": int(ok.sum()),
        "n_features": int(len(rows)),
        "n_per_depth": {int(d): int((depth == d).sum()) for d in np.unique(depth)},
        "verdict": verdict,
        "note": "labels permuted WITHIN depth; observed outside the central 95% "
        "permutation band = two-sided p < 0.05 on that side (plan §3 equivalence note)",
    }


def _bootstrap_ci(vals_fn, n_boot: int, rng: np.random.Generator) -> list[float]:
    draws = np.asarray([vals_fn(rng) for _ in range(n_boot)], dtype=np.float64)
    ok = np.isfinite(draws)
    return [float(np.percentile(draws[ok], 2.5)), float(np.percentile(draws[ok], 97.5))]


def _deferred_stanza() -> dict:
    """The v17 §0.-1 freeze's E6 deferral record (mech_tests.json `deferred` key):
    the registered H1/H2 judged-label legs await #1773's validated labels."""
    return {
        "h1_pooled_depth_stratified": "deferred: awaiting #1773 labels",
        "h2_within_l3_tail_contrast": "deferred: awaiting #1773 labels",
        "resume": "re-run --phase analyze --labels-file <labels.json from #1773's "
        "validated instrument>: the registered plan §3 H1/H2 permutation recipes run "
        "UNCHANGED against the persisted per-feature arrays + evidence packet",
    }


def _depth_descriptives(r2: np.ndarray, null_hi: float | None = None) -> dict:
    """Label-free per-arm descriptives over finite per-feature held-out R2."""
    r = np.asarray(r2, np.float64)
    r = r[np.isfinite(r)]
    d: dict = {
        "n_finite": int(len(r)),
        "median_r2": float(np.median(r)) if len(r) else None,
        "q25_r2": float(np.percentile(r, 25)) if len(r) else None,
        "q75_r2": float(np.percentile(r, 75)) if len(r) else None,
        "frac_r2_positive": float((r > 0).mean()) if len(r) else None,
    }
    if null_hi is not None and len(r):
        d["frac_above_shuffle_null_p97_5"] = float((r > null_hi).mean())
    return d


def _mech_doc(args, primary_name: str) -> dict:
    """Label-FREE mechanical reads (plan v17 §0.-1 headline demotion): depth
    profile descriptives at matched n, sink/k128 twin agreement, shuffle-null
    band, identity+kNN baselines (the covariate correlations + partials ride
    phase_analyze's existing battery loop into the same mech_tests doc). NO
    judged-label field may enter this dict (pinned by
    tests/test_issue1482_early_layer.py)."""
    import issue1482_feature_correlates as FC

    nz = np.load(args.out_eval / "shuffle_null_l3.npz")
    null_hi = float(np.nanpercentile(np.asarray(nz["r2"], np.float64), 97.5))
    zp = np.load(args.out_eval / f"{primary_name}.npz")
    z3 = np.load(args.out_eval / "perfeature_l3_default.npz")
    z19 = np.load(args.out_eval / "perfeature_l19_matched_ctx.npz")
    zs = np.load(args.out_eval / "perfeature_l3_sinkmask.npz")
    zk = np.load(args.out_eval / "perfeature_l3_k128.npz")
    depth_profile = {
        # the shuffle null is computed on the k64 default arm — attach its band
        # read only where the feature population matches (positional convention)
        "l3_primary": {
            "source": primary_name,
            **_depth_descriptives(
                zp["r2"], null_hi if primary_name == "perfeature_l3_default" else None
            ),
        },
        "l19_matched_ctx": _depth_descriptives(z19["r2"]),
        "l3_sinkmask_twin": _depth_descriptives(zs["r2"]),
        "l3_k128_twin": _depth_descriptives(zk["r2"]),
    }
    # sinkmask twin agreement is a PAIRED read (identical features + score rows);
    # the k128 twin is a DIFFERENT feature population -> descriptives only
    r3 = np.asarray(z3["r2"], np.float64)
    rs = np.asarray(zs["r2"], np.float64)
    m = np.isfinite(r3) & np.isfinite(rs)
    twin_agreement = {
        "sinkmask_paired_spearman": FC._spearman(r3[m], rs[m]) if int(m.sum()) >= 3 else None,
        "sinkmask_median_abs_delta": float(np.median(np.abs(r3[m] - rs[m]))) if m.any() else None,
        "n_paired": int(m.sum()),
        "k128_note": "different feature population (second dictionary) — no pairing; "
        "see depth_profile_matched_n.l3_k128_twin",
    }
    summary = json.loads((args.out_eval / "early_summary.json").read_text())
    return {
        "depth_profile_matched_n": depth_profile,
        "shuffle_null": {"p97_5": null_hi, "n_seeds": len(SHUFFLE_SEEDS)},
        "twin_agreement": twin_agreement,
        "baselines_identity": summary.get("baselines", {}),
        "knn_retrieval": summary.get("knn", {}),
        "pooled_r2": summary.get("pooled_r2", {}),
    }


def phase_analyze(args) -> None:
    """E6 (off-pod VM): mechanical depth x predictability reads + figures ->
    mech_tests.json (plan v17 §0.-1 headline demotion). The registered §3 H1/H2
    judged-label permutation legs are PRESERVED VERBATIM but run ONLY under
    --labels-file (the #1773 resume); absent labels a `deferred` stanza is
    emitted instead and no feature-label claim is made."""
    t0 = time.time()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import issue1482_feature_correlates as FC
    import issue1482_feature_extremes as FE
    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    rng = np.random.default_rng(BOOT_PERM_SEED)
    # Gate B-e chosen_k threading (concern gatebe-warn-escalation-not-threaded):
    # label-keyed reads (H1 rows, hero L3 ECDF) use the primary arm; the
    # k64-positionally-aligned diagnostics stay on perfeature_l3_default.
    primary_name = _primary_l3_perfeature(args)
    labels = _load_labels(args)
    h1: dict | None = None
    h2: dict | None = None
    if labels is not None:  # the #1773 resume path — plan §3 recipes VERBATIM
        rows = _pooled_h1_rows(args, labels)
        assert rows, "no pooled judged rows for H1"
        h1 = _h1_depth_stratified(rows, args.n_perm, rng)
        r2_l3 = np.asarray(
            [r["r2"] for r in rows if r["depth"] == 3 and r["level"] in ("low", "high")]
        )
        lev_l3 = np.asarray([r["level"] == "high" for r in rows if r["depth"] == 3])

        # H1 bootstrap CI (within-depth feature resample on the pooled Spearman)
        depth_arr = np.asarray([r["depth"] for r in rows])
        lev_arr = np.asarray([1.0 if r["level"] == "high" else 0.0 for r in rows])
        r2_arr = np.asarray([r["r2"] for r in rows], dtype=np.float64)

        def _h1_draw(rg: np.random.Generator) -> float:
            idx_parts = []
            for d in np.unique(depth_arr):
                m = np.where(depth_arr == d)[0]
                idx_parts.append(rg.choice(m, size=len(m), replace=True))
            idx = np.concatenate(idx_parts)
            rr = EA._midrank(r2_arr[idx][:, None])[:, 0]
            a = lev_arr[idx] - lev_arr[idx].mean()
            b = rr - rr.mean()
            den = float(np.sqrt((a**2).sum() * (b**2).sum()))
            return float((a * b).sum() / den) if den > 1e-12 else float("nan")

        h1["bootstrap_ci_95"] = _bootstrap_ci(_h1_draw, args.n_boot, rng)

        # H2 (registered): within-L3 decile-matched tail contrast (Set B convention,
        # decile-stratified permutation — FE._stratified_perm VERBATIM)
        sel = json.loads((args.work / "selection.json").read_text())
        fid_to_level = {int(f): v["level"] for f, v in labels.items()}
        b_rows = [
            r
            for r in sel["features"]
            if (r["b_best"] or r["b_worst"])
            and fid_to_level.get(int(r["feat_id"])) in ("low", "high")
        ]
        h2 = {"n_set_b_labeled": len(b_rows)}
        if len(b_rows) >= 8:
            is_high = np.asarray([fid_to_level[int(r["feat_id"])] == "high" for r in b_rows])
            is_best = np.asarray([bool(r["b_best"]) for r in b_rows])
            decile = np.asarray([int(r["decile"]) for r in b_rows])
            h2.update(FE._stratified_perm(is_high, is_best, decile, rng))

            def _h2_draw(rg: np.random.Generator) -> float:
                idx = rg.choice(len(b_rows), size=len(b_rows), replace=True)
                hb, bb = is_high[idx], is_best[idx]
                if bb.sum() == 0 or (~bb).sum() == 0:
                    return float("nan")
                return float(hb[bb].mean() - hb[~bb].mean())

            h2["bootstrap_ci_95"] = _bootstrap_ci(_h2_draw, args.n_boot, rng)
        else:
            h2["note"] = "fewer than 8 labeled Set-B rows — contrast reported descriptive-only"

    # covariate battery (midrank Spearman + partials; FC helpers)
    cov_doc: dict = {}
    for depth in (3, 19):
        cz = np.load(args.out_eval / f"covariates_l{depth}.npz")
        # the depth-3 join stays on the k64 arm REGARDLESS of chosen_k:
        # covariates_l3.npz is computed over the k64 f_out restriction (same
        # ordering), so only the k64 per-feature npz aligns POSITIONALLY here
        # (see _primary_l3_perfeature); under escalation this battery reads the
        # robustness twin and cov_doc records the source arm.
        pz_name = "perfeature_l3_default" if depth == 3 else "perfeature_l19_matched_ctx"
        pz = np.load(args.out_eval / f"{pz_name}.npz")
        r2 = np.asarray(pz["r2"], np.float64)
        ok = np.isfinite(r2)
        act = np.asarray(cz["activity"], np.float64)[ok]
        cons = np.asarray(cz["consistency"], np.float64)[ok]
        d: dict = {"n": int(ok.sum()), "perfeature_source": pz_name}
        r2v = r2[ok]
        for nm, v in (
            ("activity", act),
            ("consistency", cons),
            ("footprint_conc", np.asarray(cz["footprint_conc"], np.float64)[ok]),
            ("coact", np.asarray(cz["coact"], np.float64)[ok]),
            ("rb_raw_maxabs", np.asarray(cz["rb_raw_maxabs"], np.float64)[ok]),
            ("rb_centered_maxabs", np.asarray(cz["rb_centered_maxabs"], np.float64)[ok]),
        ):
            m = np.isfinite(v)
            d[f"spearman_{nm}"] = FC._spearman(v[m], r2v[m]) if int(m.sum()) >= 3 else None
        m = np.isfinite(cons)
        if int(m.sum()) >= 4:
            d["partial_consistency_given_activity"] = FC._partial_spearman(cons[m], r2v[m], act[m])
            fpv = np.asarray(cz["footprint_conc"], np.float64)[ok]
            m2 = m & np.isfinite(fpv)
            d["partial_footprint_given_consistency"] = FC._partial_spearman(
                fpv[m2], r2v[m2], cons[m2]
            )
        dense = np.asarray(cz["dense_flag"], np.int8)[ok]
        d["dense_flag_rate"] = float(dense.mean())
        d["dense_flag_median_r2"] = (
            float(np.nanmedian(r2v[dense == 1])) if int((dense == 1).sum()) else None
        )
        cov_doc[f"L{depth}"] = d

    mech: dict = {
        "primary_perfeature_l3": primary_name,
        **_mech_doc(args, primary_name),
        "covariates": cov_doc,
        "seeds": {"perm_boot": BOOT_PERM_SEED, "n_perm": args.n_perm, "n_boot": args.n_boot},
    }
    if labels is None:
        # NO label-dependent field enters mech_tests.json this round (v17 freeze)
        mech["deferred"] = _deferred_stanza()
    else:
        mech["h1_pooled_depth_stratified"] = h1
        mech["h2_within_l3_tail_contrast"] = h2
        mech["labels_file"] = str(args.labels_file)
    _write_json(args.out_eval / "mech_tests.json", mech)

    # ── figures ──
    figs = args.figures
    figs.mkdir(parents=True, exist_ok=True)
    col = {3: paper_palette(2)[0], 19: paper_palette(2)[1]}  # ONE color per depth

    # hero (v17 freeze): the MECHANICAL depth x predictability profile — R² ECDF
    # per depth + dense-flag / footprint strata panels; the judged-level panel is
    # DEFERRED to the #1773 resume (rendered below only under --labels-file).
    # The L3 ECDF reads the PRIMARY arm (chosen_k); z3 stays the k64 default arm
    # for every POSITIONAL join below (covariates / sinkmask / null). Within-L3
    # strata reuse the L3 depth color (linestyle/alpha encode the stratum) so one
    # color keeps one meaning (depth) across the figure set.
    z3 = np.load(args.out_eval / "perfeature_l3_default.npz")
    z3p = (
        z3
        if primary_name == "perfeature_l3_default"
        else np.load(args.out_eval / f"{primary_name}.npz")
    )
    z19 = np.load(args.out_eval / "perfeature_l19_matched_ctx.npz")
    cz3 = np.load(args.out_eval / "covariates_l3.npz")

    def _ecdf(ax, vals, **kw):
        v = np.asarray(vals, np.float64)
        v = v[np.isfinite(v)]
        if not len(v):
            return
        xs = np.sort(np.clip(v, -1, 1))
        ax.plot(xs, np.linspace(0, 1, len(xs)), **kw)

    fig, axes = plt.subplots(1, 3, figsize=(12.9, 3.6), layout="constrained")
    for depth, z in ((3, z3p), (19, z19)):
        n = int(np.isfinite(np.asarray(z["r2"], np.float64)).sum())
        _ecdf(axes[0], z["r2"], color=col[depth], label=f"layer {depth} ({n:,} features)")
    axes[0].set_xlabel("per-feature held-out R² (clipped at −1)")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("Predictability by depth (matched rows)")
    axes[0].legend(frameon=False, fontsize=8)
    r3d = np.asarray(z3["r2"], np.float64)
    dense = np.asarray(cz3["dense_flag"], np.int8)
    for flag, ls, lab in (
        (0, "-", "sparse (activity ≤ 0.5)"),
        (1, "--", "dense (activity > 0.5)"),
    ):
        m = dense == flag
        n = int(np.isfinite(r3d[m]).sum())
        _ecdf(axes[1], r3d[m], color=col[3], ls=ls, label=f"{lab}, n={n:,}")
    axes[1].set_xlabel("per-feature held-out R² (clipped at −1)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("Layer-3 R² by dense-latent flag")
    axes[1].legend(frameon=False, fontsize=8)
    fpv = np.asarray(cz3["footprint_conc"], np.float64)
    mf = np.isfinite(fpv) & np.isfinite(r3d)
    if int(mf.sum()) >= 9:
        t1, t2 = np.quantile(fpv[mf], [1.0 / 3.0, 2.0 / 3.0])
        for lab, m, alpha in (
            ("low", fpv <= t1, 0.35),
            ("mid", (fpv > t1) & (fpv <= t2), 0.65),
            ("high", fpv > t2, 1.0),
        ):
            _ecdf(
                axes[2],
                r3d[m & mf],
                color=col[3],
                alpha=alpha,
                label=f"footprint {lab} tertile",
            )
        axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_xlabel("per-feature held-out R² (clipped at −1)")
    axes[2].set_ylabel("ECDF")
    axes[2].set_title("Layer-3 R² by footprint concentration")
    savefig_paper(fig, "early_layer_hero_depth_mech_profile", dir=figs)
    plt.close(fig)

    if labels is not None:
        # the pre-freeze judged-level tails panel — #1773 resume only (deferred
        # this round; code path preserved verbatim for the resume)
        tail_vals, tail_labels, tail_cols = [], [], []
        for depth in (3, 19):
            if depth == 3:
                hb = h2.get("statistic_frac_high_best_minus_worst") if h2 else None
                fr_best = (
                    float(lev_l3[np.argsort(-r2_l3)[: max(1, len(r2_l3) // 4)]].mean())
                    if len(r2_l3)
                    else np.nan
                )
                best = fr_best
                worst = best - hb if hb is not None else np.nan
            else:
                ext = json.loads(COMMITTED_EXTREMES.read_text())
                sb = ext.get("set_b_activity_controlled", {})
                best = sb.get("frac_high_best")
                worst = sb.get("frac_high_worst")
                if best is None:
                    lv = [
                        (r["level"] == "high", bool(r["b_best"]), bool(r["b_worst"]))
                        for r in ext["features"]
                        if r.get("level") in ("low", "high") and (r["b_best"] or r["b_worst"])
                    ]
                    arr = np.asarray(lv)
                    best = float(arr[arr[:, 1] == 1, 0].mean())
                    worst = float(arr[arr[:, 2] == 1, 0].mean())
            for side, v in (("best", best), ("worst", worst)):
                if v is None or not np.isfinite(v):
                    continue
                tail_vals.append(v)
                tail_labels.append(f"L{depth} {side}")
                tail_cols.append(col[depth])
        fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
        xpos = np.arange(len(tail_vals))
        ax.bar(xpos, tail_vals, color=tail_cols)
        ax.set_xticks(xpos, tail_labels, rotation=20)
        ax.set_ylabel("judged high-level fraction")
        ax.set_title("High-level share in R² tails (Set B)")
        savefig_paper(fig, "early_layer_level_tails", dir=figs)
        plt.close(fig)

    # companion (mandatory low-level plot): R² vs activity scatter at L3 with the
    # shuffle-null band overlaid
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    r2 = np.asarray(z3["r2"], np.float64)
    act = np.asarray(z3["activity"], np.float64)
    ok = np.isfinite(r2)
    ax.scatter(act[ok], np.clip(r2[ok], -1, 1), s=3, alpha=0.15, color=col[3], linewidths=0)
    nz = np.load(args.out_eval / "shuffle_null_l3.npz")
    null_hi = np.nanpercentile(np.asarray(nz["r2"], np.float64), 97.5)
    ax.axhline(null_hi, ls="--", color="grey", lw=1, label="shuffle-null p97.5 (K=20)")
    edges = np.quantile(act[ok], np.linspace(0, 1, 11))
    dec_med = [
        np.nanmedian(r2[ok][(act[ok] >= edges[i]) & (act[ok] <= edges[i + 1])]) for i in range(10)
    ]
    mid = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(mid, dec_med, marker="o", ms=4, color="black", lw=1.2, label="decile median")
    ax.set_xscale("log")
    ax.set_xlabel("feature activity (fraction of fit contexts active)")
    ax.set_ylabel("per-feature held-out R² (clipped)")
    ax.set_title("Layer-3 per-feature predictability vs activity")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "early_layer_r2_vs_activity_l3", dir=figs)
    plt.close(fig)

    # exploratory dump (over-produce; one figure per read)
    def _scatter(xkey_npz, xkey, name, xlabel, logx=False):
        fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
        cz = np.load(args.out_eval / xkey_npz)
        v = np.asarray(cz[xkey], np.float64)
        m = np.isfinite(r2) & np.isfinite(v)
        ax.scatter(v[m], np.clip(r2[m], -1, 1), s=3, alpha=0.15, color=col[3], linewidths=0)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("per-feature held-out R² (clipped)")
        savefig_paper(fig, name, dir=figs)
        plt.close(fig)

    _scatter(
        "covariates_l3.npz",
        "consistency",
        "early_layer_r2_vs_consistency_l3",
        "within-answer consistency (mean frac | active)",
    )
    _scatter(
        "covariates_l3.npz",
        "footprint_conc",
        "early_layer_r2_vs_footprint_l3",
        "direct-logit footprint (top-20 concentration)",
    )
    _scatter(
        "covariates_l3.npz",
        "rb_centered_maxabs",
        "early_layer_r2_vs_rb_l3",
        "max |cos(decoder, r_B)| (centered)",
    )
    zs = np.load(args.out_eval / "perfeature_l3_sinkmask.npz")
    fig, ax = plt.subplots(figsize=(4.2, 4.0), layout="constrained")
    m = np.isfinite(r2) & np.isfinite(np.asarray(zs["r2"], np.float64))
    ax.scatter(
        np.clip(r2[m], -1, 1),
        np.clip(np.asarray(zs["r2"], np.float64)[m], -1, 1),
        s=3,
        alpha=0.15,
        color=col[3],
        linewidths=0,
    )
    ax.plot([-1, 1], [-1, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("R² (default pooling mask)")
    ax.set_ylabel("R² (sink-robustness mask)")
    ax.set_title("Sink-mask robustness (paired features)")
    savefig_paper(fig, "early_layer_sinkmask_paired", dir=figs)
    plt.close(fig)
    zk = np.load(args.out_eval / "perfeature_l3_k128.npz")
    fig, ax = plt.subplots(figsize=(4.2, 3.4), layout="constrained")
    rk = np.asarray(zk["r2"], np.float64)
    okk = np.isfinite(rk)
    xs = np.sort(np.clip(rk[okk], -1, 1))
    ax.plot(
        xs,
        np.linspace(0, 1, len(xs)),
        color=col[3],
        ls="--",
        label=f"k=128 dictionary ({int(okk.sum()):,})",
    )
    xs = np.sort(np.clip(r2[ok], -1, 1))
    ax.plot(
        xs, np.linspace(0, 1, len(xs)), color=col[3], label=f"k=64 dictionary ({int(ok.sum()):,})"
    )
    ax.set_xlabel("per-feature held-out R² (clipped)")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Second-dictionary replication at layer 3")
    savefig_paper(fig, "early_layer_k128_replication", dir=figs)
    plt.close(fig)

    if h1 is not None:
        logger.info(
            "[e6] done (labels resume): H1 %s (obs %.3f, band [%.3f, %.3f]); figures -> %s",
            h1["verdict"],
            h1["observed_pooled_spearman"],
            h1["perm_band_2p5_97p5"][0],
            h1["perm_band_2p5_97p5"][1],
            figs,
        )
    else:
        logger.info(
            "[e6] done (mechanical reads only — judged legs DEFERRED to #1773): "
            "mech_tests.json + figures -> %s",
            figs,
        )
    _record_phase_time(args, "analyze", time.time() - t0)


# ── verify-imports (Axis-1 import-resolution leg; the i606 AST pattern) ──────────


def _verify_imports() -> int:
    """Execute every DEFERRED import in THIS file (AST-walked, never a
    hand-maintained list) so a smoke-skipped branch cannot hide an ImportError
    until the pod (#606/#1332 class). Exit 0 on success."""
    import ast
    import importlib

    tree = ast.parse(Path(__file__).read_text())
    deferred: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                deferred.extend((alias.name, None) for alias in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                deferred.extend((sub.module, alias.name) for alias in sub.names)
    n_ok = 0
    for mod_name, sym in sorted(set(deferred)):
        mod = importlib.import_module(mod_name)
        if sym is not None:
            getattr(mod, sym)  # fail-loud on a missing symbol
        n_ok += 1
    print(f"[verify-imports] {n_ok} deferred imports resolved OK", flush=True)
    return 0


# ── main ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Linear phase dispatcher (smoke IS this driver with tiny args)."""
    ap = argparse.ArgumentParser(description="Issue #1482 early-layer-arm driver (E0-E6).")
    ap.add_argument(
        "--phase",
        default="all",
        # "judge" stays a CHOICE but is a hard exit (RC_JUDGE_FROZEN) — the v17
        # §0.-1 judged-label freeze; "evidence" is its replacement leg.
        choices=[
            "all",
            "pilot",
            "capture",
            "upload1",
            "fits",
            "upload2",
            "evidence",
            "judge",
            "analyze",
        ],
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--full", action="store_true", help="explicit production mode (default)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-eval", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--sae-dir", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None, help="E5 judge work dir (VM)")
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--max-chunks", type=int, default=None, help="0 = all (production)")
    ap.add_argument("--n-fit", type=int, default=None, help="requested S_fit size")
    ap.add_argument("--n-score", type=int, default=None, help="requested S_score size")
    ap.add_argument("--val-carve", type=int, default=None, help="lambda-selection carve of S_fit")
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--pilot-n", type=int, default=None)
    ap.add_argument("--max-features-in", type=int, default=None)
    ap.add_argument("--max-features-out", type=int, default=None)
    ap.add_argument("--parent-shards", type=int, default=None, help="0 = whole parent store")
    ap.add_argument("--judge-limit", type=int, default=None, help="0 = all union items")
    ap.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="#1773-resume judged labels for E6 (plan v17 §0.-1): the H1/H2 "
        "permutation legs run ONLY when this is provided; absent, mech_tests.json "
        "carries a `deferred` stanza instead",
    )
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="fitter seed (parent MLP seed 0)")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--tiny-model",
        action="store_true",
        help="CARVE-OUT (GPU-bound capture on a no-GPU VM): from-config 24-layer "
        "same-arch Qwen2 over the REAL vocab (EA._load_model_tok; #906 pattern)",
    )
    ap.add_argument(
        "--verify-imports",
        action="store_true",
        help="execute every deferred import in this file, then exit (Axis-1 leg)",
    )
    args = ap.parse_args()
    if args.verify_imports:
        return _verify_imports()

    smoke_defaults = {
        "max_chunks": 1,
        "n_fit": 2000,
        "n_score": 600,
        "val_carve": 4,
        "gen_batch": 2,
        "pilot_n": 6,
        "max_features_in": 256,
        "max_features_out": 512,
        "parent_shards": 2,
        "judge_limit": 2,
        "n_perm": 200,
        "n_boot": 200,
    }
    prod_defaults = {
        "max_chunks": 0,
        "n_fit": 24_000,
        "n_score": 6_000,
        "val_carve": PROD_VAL_CARVE,
        "gen_batch": 8,
        "pilot_n": 500,
        "max_features_in": PROD_MAX_FEATURES_IN,
        "max_features_out": 16_384,
        "parent_shards": 0,
        "judge_limit": 0,
        "n_perm": 10_000,
        "n_boot": 10_000,
    }
    dd = smoke_defaults if args.smoke else prod_defaults
    for k, v in dd.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.device == "auto":
        args.device = "cuda" if EA._physical_gpu_ids() else "cpu"
    args.base_root = PROJECT_ROOT / "data" / "issue_1482"
    base = default_smoke_root(args.base_root) if args.smoke else (args.base_root / "early")
    args.out_root = base
    if args.out_eval is None:
        args.out_eval = (
            (base / "eval_results")
            if args.smoke
            else (PROJECT_ROOT / "eval_results" / "issue_1482" / "early_layer")
        )
    if args.scratch is None:
        args.scratch = base / "scratch"
    if args.store is None:
        args.store = base / "store_l3"
    if args.sae_dir is None:
        args.sae_dir = args.base_root / "hf_dl" / "sae"
    if args.work is None:
        args.work = (
            (base / "work")
            if args.smoke
            else Path("/mnt/eps-data/thomasjiralerspong/issue1482_earlylayer")
        )
    if args.figures is None:
        # smoke figures NEVER touch the committed figures/ paths (kresample convention)
        args.figures = (
            (base / "figures")
            if args.smoke
            else (PROJECT_ROOT / "figures" / "issue_1482" / "early_layer")
        )
    for p in (args.out_eval, args.scratch, args.store):
        p.mkdir(parents=True, exist_ok=True)

    ph = args.phase
    dispatch = {
        "pilot": phase_pilot,
        "capture": phase_capture,
        "upload1": phase_upload1,
        "fits": phase_fits,
        "upload2": phase_upload2,
        "evidence": phase_evidence,
        "judge": phase_judge,  # FROZEN — hard exit (plan v17 §0.-1)
        "analyze": phase_analyze,
    }
    if ph == "all":
        for name in ("pilot", "capture", "upload1", "fits", "upload2"):
            C.phase(f"early-{name}")
            dispatch[name](args)
    else:
        C.phase(f"early-{ph}")
        dispatch[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    # explicit exit after flushing: heavy C-extension teardown can rewrite the rc
    # in interpreter finalization (PyGILState atexit race, #1689 gotcha)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
