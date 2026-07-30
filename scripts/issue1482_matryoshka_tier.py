"""Issue #1482 — matryoshka-tier round driver (plan v21): the L20 SAELens dictionary arm.

ONE new variable vs the early-layer battery: the SAE dictionary family — layer-20
SAELens matryoshka jumprelu dictionaries (chanind, lmsys + pile variants, k=100,
tiers [0,2048)/[2048,16384)/[16384,65536)) with training-imposed tier stratification.
Everything else — the EXACT early-layer 30k rows (seed 14823, sha-asserted), capture
convention, pooling recipe, 1%-of-fit floor, LAMBDAS_N1M grid, shuffle-null K=20,
covariate battery, evidence machinery — inherits from `issue1482_early_layer` (EL) /
`issue1482_error_analysis` (EA) / `issue779_ffc_n1m_fits` (N1M).

JUDGED-LABEL FREEZE (plan v21 §4 M5, inheriting v17 §0.-1): ZERO judge calls this
round. M5 is evidence-persistence ONLY (tier x within-tier R2-tercile packets for the
#1773 labelling resume); there is NO judge phase in this driver at all.

Phases (plan v21 §4; smoke IS this driver with tiny args — PASS_UNIFIED):
  pilot    M0: 1 raw chunk through the FULL production path — tokens/s, Gate B-m
           (round-trip FVE/L0 per dictionary on our h20 under reference token-pool
           semantics + the sparsity-correlation fingerprint), hook-alignment probes
           (lmsys dict on h16/h19/h21/h24 vs h20 — layer-20-maximal required in
           production), prefix-end constancy @L20, G2-m two-bar identity gate
           (fresh c_last@19 vs parent STORED cx_last@19: per-row >= 0.999,
           flattened >= 0.995), fit-kernel pilot (the M3 `pilot-gated` basis),
           throughput descope arithmetic (1/2-basis kill, floor 23,000).
  capture  M1: REUSED early-layer rows (24k fit / 6k score, shas re-asserted) ->
           teacher-forced capture @ layer 20 -> dual encode (lmsys default +
           sinkmask poolings; pile default) + dense c_last@20 / prefix-end@20;
           CONSOLIDATED ~30-chunk npz shards (the 72-min per-chunk-upload fix).
  upload1  M2: store -> HF analysis_tensors/matryoshka_tier/ BEFORE fits (#825).
  fits     M3: shared-Gram ridge arms (lmsys ctx + sink twin; pile ctx; dense
           companions; prefix nulls) on TIER-STRATIFIED RANDOM 16,384-feature
           panels (seed 14824), shuffle-null K=20 per dictionary, covariate
           battery (per-trait + max r_B at row 20, raw + centered), full-population
           descriptives, cross-dictionary decoder matching (running-max chunks).
  upload2  M4: eval outputs -> HF + the poller results sentinel (terminal pod phase).
  evidence M5 (off-pod VM, 0 GPU, ZERO judge calls): tier x within-tier R2-tercile
           selection (lmsys 40/cell, pile 20/cell; deterministic evenly-spaced-by-
           rank within cell) -> per-feature evidence packets (top-8 firing answer
           ROW IDS, co-activation neighbours, footprint tokens, covariate row,
           Neuronpedia links) -> eval_results/issue_1482/matryoshka_tier/evidence/.
  analyze  M6 (off-pod VM): H1 within-stratum tier permutation (10,000 draws,
           batched), H2 chat-vs-pile bootstrap, per-tier x per-decile profile
           (decile->quintile merge fallback), covariate correlations + partials,
           matching reads, corpus-transfer fold, figures.

Pod-side contract: sentinels under /workspace/logs/issue-1482-*.json ONLY (never
task.py); [phase=...] breadcrumbs come from the launcher
(scripts/issue1482_matryoshka_launch.sh); the results sentinel is written by the
terminal pod phase (upload2). LMSYS/WildChat text is DIGEST-ONLY.
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

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1482_early_layer as EL  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482_matryoshka")

TASK_ID = 1482
L_TIER = 20  # the round's ONE capture layer (both dictionaries hook model.layers.20)
L_G2 = 19  # G2-m stored-reference layer (binds our convention to the parent capture)
HOOK_PROBE_LAYERS = (16, 19, 21, 24)  # M0 (c): layer-20-maximal expected (lmsys dict)
FAMILIES = ("lmsys", "pile")
SAE_IDS = {"lmsys": "lmsys/matryoshka/k-100", "pile": "pile/matryoshka/k-100"}
NP_MODEL_ID = "qwen2.5-7b-it"  # issue1482_feature_extremes.NP_MODEL_ID (aux links only)
NP_SOURCES = {"lmsys": "20-matryoshka-chat-65k", "pile": "20-matryoshka-65k"}
SUBSAMPLE_SEED = 14823  # REUSED early-layer rows (plan §10 Seeds)
PANEL_SEED = 14824  # tier-stratified target-panel selection (plan §10)
CARVE_SEED = 14825  # lambda-selection carve of S_fit (plan §10)
SHUFFLE_SEEDS = tuple(range(1_482_100, 1_482_120))  # K=20, plan §10
BOOT_PERM_SEED = 148_240  # M6 permutation/bootstrap seed, plan §10
# Gate B-m (plan §7 / §11 R6): no published FVE exists for these dictionaries, so the
# HALT sits at the structural encode-breakage boundary (breakage <= ~0.2 vs plausible
# correct >= ~0.6; same-model references: L19 realized 0.8097, L3 0.93); lmsys WARN
# band [0.35, 0.70) -> proceed with caveat + analyzer adjudication (value persisted);
# pile has NO weakness gate above the shared structural floor (weakness IS the H2 read).
GATE_BM_HALT = 0.35
GATE_BM_WARN_TOP = 0.70
# G2-m two-bar (plan §7): per-row fresh-vs-stored cosine >= 0.999 (early round realized
# 0.999881 on this exact comparison) / flattened concatenated cosine >= 0.995.
G2M_ROW_COS_MIN = 0.999
G2M_FLAT_COS_MIN = 0.995
PREFIX_CONSTANCY_COS_MIN = EA.PREFIX_CONSTANCY_COS_MIN  # 0.9999 (parent A4 convention)
# M0 pilot-throughput kill (plan §7): < 1/2 of the measured 5,471.5 tok/s basis makes
# the 30k envelope exceed the approved 6 GPU-h at the ~2x booking -> proportional
# descope over the SAME early-layer rows, hard floor 23,000 contexts (n_fit 18,400;
# post-carve tr 16,400 > d_widest 16,384 — the Statistics-critic-corrected arithmetic).
TPS_BASIS = 5471.5
TPS_KILL_FRAC = 0.5
PROD_MAX_FEATURES_IN = 8_192
PROD_VAL_CARVE = 2_000
D_WIDEST_DESIGN = 2 * PROD_MAX_FEATURES_IN  # ctx design: psi_mean ++ psi_last
DESCOPE_FLOOR_CONTEXTS = 23_000
PANEL_CAP = 16_384  # production target-panel size (plan §11 R5)
PER_TIER_PANEL_FLOOR = 1_000  # per-tier floor gate (plan §7): below -> WARN + merge
CONSOLIDATE_CHUNKS = 30  # ~64 consolidated shards over 1,920 raw chunks (plan §4 M1)
MATCH_CHUNK_ROWS = 4_096  # cross-dict decoder-matching GEMM chunk (plan §4 M3)
MATCH_THRESHOLDS = (0.5, 0.7, 0.9)
EVIDENCE_QUOTA = {"lmsys": 40, "pile": 20}  # per (tier x tercile) cell (plan §4 M5)
RC_GATE_BM = 22  # Gate B-m structural HALT (mirrors EL.RC_GATE_BE class)
RC_HOOK = 23  # hook-alignment off-by-one signature (plan §7: HALT + report)
RC_G2M = 24  # capture-convention identity FAIL (never fit past it)
RC_THROUGHPUT = 25  # pilot-throughput kill (approval infeasible at the floor)
HF_DATA_REPO = C.HF_DATA_REPO
EARLY_STORE_PREFIX = "issue1482_error_analysis/analysis_tensors/early_layer/store"
EARLY_SPLIT_HF_REVISION = "a5048fcee5f4878b8fee28a1cfd78fce84e58a76"  # plan §10 pin
# Committed sha literals from eval_results/issue_1482/early_layer/split_early.json on
# main (read at implementation time; the driver re-asserts the STAGED npz against them).
EARLY_SPLIT_SHAS = {
    "s_fit_sha256": "9e397255da497a27dd136d639b35fb0126f618f1781ac30845b2368ee539d02d",
    "s_score_sha256": "338310fa55fbea9c738520c26cb7dd029d48a8c981132aab2eddcae3621c1747",
}


# ── small utils ──────────────────────────────────────────────────────────────────


def _write_json(path: Path, obj: dict) -> None:
    obj = dict(obj)
    obj.setdefault("metadata", C.reproducibility_metadata())
    C.write_json_atomic(path, obj)


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel (blocks_pipeline: false; plan §9 phase_outputs)."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        C.write_sentinel(f"matryoshka-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:  # sentinel write must never kill the run on the VM smoke
        logger.warning("[sentinel] matryoshka-%s write failed: %s", name, e)


def default_smoke_root(base_root: Path) -> Path:
    """ONE shared derivation for the smoke leg's out-root (writer AND reaper use
    this — the chained smoke-then-full residue gotcha, #1586 fu r3). Distinct from
    EL.default_smoke_root (early_smoke) so the rounds never clobber each other."""
    return base_root / "matryoshka_smoke"


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


def gate_bm_verdict(fve_lmsys: float, fve_pile: float) -> dict:
    """Gate B-m lattice (plan §7 / §11 R6). Pure + unit-probed in the smoke leg.

    lmsys: PASS >= 0.70 | WARN [0.35, 0.70) (proceed + caveat, analyzer adjudication)
    | HALT < 0.35 (structural encode breakage — the headline dictionary is broken, so
    the round halts; there is no lmsys-less descope for a headline arm). pile: OK >=
    0.35 | STRUCTURAL_FLOOR < 0.35 -> proceed lmsys-only (loud scope-item-6 descope,
    reported not gated — pile weakness ABOVE the floor is the H2 finding)."""
    lm = (
        "PASS"
        if fve_lmsys >= GATE_BM_WARN_TOP
        else ("WARN" if fve_lmsys >= GATE_BM_HALT else "HALT")
    )
    pl = "OK" if fve_pile >= GATE_BM_HALT else "STRUCTURAL_FLOOR"
    if lm == "HALT":
        action = "halt"
    elif pl == "STRUCTURAL_FLOOR":
        action = "proceed_lmsys_only"
    else:
        action = "proceed"
    return {"lmsys": lm, "pile": pl, "action": action}


def descope_plan_m(tps: float, n_total: int, val_carve: int) -> dict | None:
    """Pure M0 throughput-descope arithmetic (plan §7). Mirrors EL.descope_plan's
    shape at THIS round's plan-named constants (kill frac 1/2, floor 23,000 — EL's
    module constants are 1/3 and 22,982, so the parent function cannot be reused
    verbatim; the post-carve estimator-validity assert is identical). Returns None
    when tps clears the kill floor; raises SystemExit(RC_THROUGHPUT) below the
    descope floor."""
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
        "reason": f"tokens/s {tps:.0f} < 1/2 basis {TPS_BASIS}",
        "effective_tr": eff_tr,
    }


def _tier_stratified_panel(counts: np.ndarray, n_fit: int, cap: int, seed: int):
    """Tier-stratified RANDOM target panel above the 1%-of-fit activity floor
    (plan §4 M3 / §11 R5 — the stated deviation from the parent's top-activity cap):
    all floor-clearing tier-0 (allocation 2048/16384 of ``cap`` — vacuous at the
    production cap since tier 0 holds only 2,048 features), 6144/16384-of-cap seeded
    random tier-1, remainder tier-2; a tier short of its allocation contributes all
    of it and the shortfall reallocates (tier 2 first, then 1, then 0). Deterministic
    given (counts, n_fit, cap, seed). Returns (sorted int64 panel, doc)."""
    floor = max(1, int(np.ceil(0.01 * n_fit)))
    clearing = np.where(counts >= floor)[0]
    tiers = S.tier_of(clearing)
    by_tier = {t: clearing[tiers == t] for t in (0, 1, 2)}
    n_by = {t: int(len(by_tier[t])) for t in (0, 1, 2)}
    alloc = {
        0: min(n_by[0], max(0, round(cap * 2048 / PANEL_CAP))),
        1: min(n_by[1], max(0, round(cap * 6144 / PANEL_CAP))),
    }
    alloc[2] = min(n_by[2], max(0, cap - alloc[0] - alloc[1]))
    leftover = cap - sum(alloc.values())
    for t in (2, 1, 0):  # reallocate shortfall, most-specific tier first
        if leftover <= 0:
            break
        add = min(leftover, n_by[t] - alloc[t])
        alloc[t] += add
        leftover -= add
    rng = np.random.default_rng(seed)
    parts = []
    for t in (0, 1, 2):
        pool = by_tier[t]
        if alloc[t] >= len(pool):
            parts.append(pool)
        elif alloc[t] > 0:
            parts.append(rng.choice(pool, size=alloc[t], replace=False))
    panel = np.sort(np.concatenate(parts)).astype(np.int64) if parts else np.empty(0, np.int64)
    panel_tiers = S.tier_of(panel)
    doc = {
        "floor": floor,
        "cap": cap,
        "seed": seed,
        "n_clearing_by_tier": {str(t): n_by[t] for t in (0, 1, 2)},
        "alloc_by_tier": {str(t): int(alloc[t]) for t in (0, 1, 2)},
        "n_panel": int(len(panel)),
        "panel_tier_counts": {str(t): int((panel_tiers == t).sum()) for t in (0, 1, 2)},
        "per_tier_floor_warn": [t for t in (0, 1, 2) if n_by[t] < PER_TIER_PANEL_FLOOR],
    }
    return panel, doc


def _m_hf_prefix(args) -> str:
    """HF destination prefix for this round's store + eval artifacts (smoke leg gets
    its own non-canonical prefix — real Hub writes, never the production paths)."""
    leaf = "matryoshka_tier_smoke" if args.smoke else "matryoshka_tier"
    return f"issue1482_error_analysis/analysis_tensors/{leaf}"


def _stage_early_split(args) -> Path:
    """Stage the early-layer round's split_indices_early.npz from HF at the pinned
    revision (idempotent; exact per-file target — no mirror-root trap)."""
    from explore_persona_space.orchestrate import hub

    dest = args.scratch / "split_indices_early.npz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    return hub.stage_hub_file(
        HF_DATA_REPO,
        f"{EARLY_STORE_PREFIX}/split_indices_early.npz",
        dest,
        repo_type="dataset",
        revision=EARLY_SPLIT_HF_REVISION,
    )


def _active_families(args) -> tuple[str, ...]:
    """Families this run fits/reads, keyed to the Gate B-m action persisted in
    m_pilot.json (pile structural-floor drop -> lmsys-only, loud; plan §7)."""
    pilot_doc = args.out_eval / "m_pilot.json"
    if not pilot_doc.exists():
        return FAMILIES
    action = json.loads(pilot_doc.read_text())["gate_bm"]["action"]
    return ("lmsys",) if action == "proceed_lmsys_only" else FAMILIES


def _load_reused_rows(args) -> dict:
    """Load the REUSED early-layer subsample + assert its shas against the committed
    split_early.json literals AND the parent pool shas / hygiene invariants
    (plan §4 M1; runs identically in both modes)."""
    pools = EL._load_split_and_assert(args)  # parent pool shas (split_1482.json literals)
    z = np.load(args.scratch / "split_indices_early.npz")
    for k in ("s_fit", "s_score", "prov_fit", "prov_score"):
        assert k in z.files, f"split_indices_early.npz missing key {k}"
    s_fit = np.asarray(z["s_fit"], dtype=np.int64)
    s_score = np.asarray(z["s_score"], dtype=np.int64)
    got_fit, got_score = EL._sha_ids(s_fit), EL._sha_ids(s_score)
    assert got_fit == EARLY_SPLIT_SHAS["s_fit_sha256"], (
        f"s_fit sha drift: {got_fit} != committed split_early.json literal"
    )
    assert got_score == EARLY_SPLIT_SHAS["s_score_sha256"], (
        f"s_score sha drift: {got_score} != committed split_early.json literal"
    )
    assert set(s_fit).isdisjoint(set(s_score)), "S_fit and S_score overlap"
    assert np.isin(s_fit, pools["sae_fit"]).all(), "S_fit escapes the parent sae_fit pool"
    assert np.isin(s_score, pools["holdout"]).all(), "S_score escapes the parent holdout pool"
    row_ci = np.load(args.scratch / "row_ci.npy")
    prov_u8 = np.load(args.scratch / "prov.npy")
    # provenance cross-check: the staged npz's prov columns must match the parent's
    assert (prov_u8[s_fit] == np.asarray(z["prov_fit"], np.uint8)).all(), "prov_fit drift"
    assert (prov_u8[s_score] == np.asarray(z["prov_score"], np.uint8)).all(), "prov_score drift"
    logger.info("[split] reused early-layer rows sha-verified (%d/%d)", len(s_fit), len(s_score))
    return {"s_fit": s_fit, "s_score": s_score, "row_ci": row_ci, "prov_u8": prov_u8}


# ── M0: pilot ─────────────────────────────────────────────────────────────────────


# PHASE_IDEMPOTENCY_EXEMPT: M0 re-runs unconditionally BY PLAN DESIGN — it IS the
# per-pod gate + throughput measurement (tokens/s, Gate B-m, G2-m re-measure on every
# dispatch pod; ~0.25 h), so no output-exists resume skip is wired.
def phase_pilot(args) -> None:
    """M0: throughput + Gate B-m + hook alignment + prefix constancy + G2-m +
    fit-kernel pilot. Gates are computed identically under --smoke but demoted to
    informational (production-n-calibrated verdicts; gotcha #1345)."""
    t0 = time.time()
    if not args.smoke:
        reap_sibling_smoke_root(args)
    EA._headroom(args.store, 2 if args.smoke else 30, "m0-pilot")
    EL._stage_scratch_meta(args)
    _stage_early_split(args)
    for fam in FAMILIES:
        S.SAELensJumpReLU.ensure_downloaded(SAE_IDS[fam], args.sae_dir)
    pools = EL._load_split_and_assert(args)
    _load_reused_rows(args)  # sha asserts run at pilot time too (fail before capture)
    row_ci = np.load(args.scratch / "row_ci.npy")
    union = np.sort(np.concatenate([pools["sae_fit"], pools["holdout"], pools["sae_val"]]))
    needed_ci = {int(row_ci[r]): int(r) for r in union}
    assert -1 not in needed_ci, "SAE-arm rows must be NEW rows (text-resolvable)"
    model, tok = EA._load_model_tok(args)
    n_layers = int(model.config.num_hidden_layers)
    probe_layers = tuple(li for li in HOOK_PROBE_LAYERS if li < n_layers)
    pilot_capture_layers = tuple(sorted(set(probe_layers) | {L_TIER, L_G2}))
    assert L_TIER < n_layers, f"capture layer {L_TIER} >= n_layers {n_layers}"
    prefix_chars = EA._prefix_char_len(tok)
    dns = argparse.Namespace(max_chunks=1, scratch=args.scratch)
    names = EA._raw_chunk_names(dns)
    first_chunk = names[0]
    pilot_rows: list = []
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
    logger.info("[m0] %d pilot rows from %s", len(pilot_rows), first_chunk)

    # (a) tokens/s at production batch shape, capturing the pilot layer set
    caps = []
    tot_tokens = 0
    t_cap = time.time()
    for s0 in range(0, len(pilot_rows), args.gen_batch):
        batch = pilot_rows[s0 : s0 + args.gen_batch]
        caps.extend(EA._batched_capture(model, tok, batch, pilot_capture_layers, args.device))
        tot_tokens += sum(len(r[2]) for r in batch)
    tps = tot_tokens / max(1e-9, time.time() - t_cap)

    # gamma accumulator at L20 (covariate battery report; BOS-stripped tokens)
    gam = {
        "n": 0,
        "s": torch.zeros(3584, dtype=torch.float64),
        "ss": torch.zeros(3584, dtype=torch.float64),
    }
    for cap in caps:
        EL._accumulate_gamma(cap[L_TIER][S.BOS_OFFSET :], gam)

    # (b) Gate B-m: round-trip FVE/L0 per dictionary on our h20 (reference token-pool
    # semantics) + the lmsys sparsity-correlation convention fingerprint (diagnostic)
    h20_all = torch.cat([c[L_TIER][S.BOS_OFFSET :] for c in caps])
    fitness: dict = {}
    saes: dict = {}
    for fam in FAMILIES:
        sae = S.SAELensJumpReLU.load(SAE_IDS[fam], device=args.device, cache_dir=args.sae_dir)
        v, l0, diag = sae.fve_l0(h20_all)
        fitness[fam] = {"fve": round(float(v), 4), "l0": round(float(l0), 2), **diag}
        saes[fam] = sae
    # sparsity fingerprint (lmsys): Spearman(our per-feature firing frac, training
    # sparsity.safetensors) over finite entries — a wrong encode decorrelates it.
    import issue1482_feature_correlates as FC

    keep_mask = S.token_inlier_mask(h20_all)
    inliers = h20_all[keep_mask]
    freq = torch.zeros(S.SAELENS_DICT_SIZE, dtype=torch.float64)
    for s0 in range(0, inliers.shape[0], 2048):
        f = saes["lmsys"].encode(inliers[s0 : s0 + 2048])
        freq += (f > 0).sum(0, dtype=torch.float64).cpu()
    freq = (freq / max(1, inliers.shape[0])).numpy()
    sp_ref = S.fetch_saelens_sparsity(SAE_IDS["lmsys"], cache_dir=args.sae_dir).numpy()
    ok = np.isfinite(freq) & np.isfinite(sp_ref)
    sparsity_spearman = FC._spearman(freq[ok], sp_ref[ok]) if int(ok.sum()) >= 3 else None

    # (c) hook-alignment probes: lmsys FVE at h16/h19/h21/h24 vs h20 (L20-maximal)
    hook_fve: dict[str, float] = {f"L{L_TIER}": fitness["lmsys"]["fve"]}
    for li in probe_layers:
        h_li = torch.cat([c[li][S.BOS_OFFSET :] for c in caps])
        v, _l0, _diag = saes["lmsys"].fve_l0(h_li)
        hook_fve[f"L{li}"] = round(float(v), 4)
    hook_max_at_20 = hook_fve[f"L{L_TIER}"] >= max(hook_fve[f"L{li}"] for li in probe_layers)

    # (d) prefix-end constancy at L20
    hp20 = torch.stack([caps[i][L_TIER][pilot_rows[i][3], :] for i in range(len(caps))])
    cos_min_prefix = EA.prefix_constancy_cos_min(hp20)

    # (e) G2-m two-bar identity gate: fresh c_last@19 vs parent STORED cx_last@19
    n_gate = min(8, len(pilot_rows))
    gate_cis = [pilot_rows[i][1] for i in range(n_gate)]
    stored = EL._stored_cx19_for(args, first_chunk, gate_cis)
    row_cos, fresh_cat, stored_cat = [], [], []
    for i in range(n_gate):
        ci = pilot_rows[i][1]
        if ci not in stored:
            continue
        fresh = caps[i][L_G2][pilot_rows[i][4], :]
        row_cos.append(float(torch.nn.functional.cosine_similarity(fresh, stored[ci], dim=0)))
        fresh_cat.append(fresh)
        stored_cat.append(stored[ci])
    assert row_cos, "G2-m: no pilot row found in the parent capture chunk"
    g2m_row_min = min(row_cos)
    g2m_flat = float(
        torch.nn.functional.cosine_similarity(
            torch.cat(fresh_cat).flatten(), torch.cat(stored_cat).flatten(), dim=0
        )
    )

    # (f) fit-kernel pilot: shared-Gram eigh at the widest design + one production-shape
    # X^TY GEMM block + one cross-dict decoder-cosine chunk (the M3 pilot-gated basis)
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
    match_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    chunk_rows = 512 if args.smoke else MATCH_CHUNK_ROWS
    d_l = saes["lmsys"].w_dec[:chunk_rows].to(device=dev, dtype=match_dtype)
    d_p = saes["pile"].w_dec.to(device=dev, dtype=match_dtype)
    t_m = time.time()
    _ = d_l @ d_p.T
    if dev.type == "cuda":
        torch.cuda.synchronize()
    match_s = time.time() - t_m
    del g, xb, yb, d_l, d_p

    verdict = gate_bm_verdict(fitness["lmsys"]["fve"], fitness["pile"]["fve"])
    descope = None
    if not args.smoke:
        try:
            descope = descope_plan_m(tps, args.n_fit + args.n_score, args.val_carve)
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
        "sae_repo": S.SAELENS_REPO,
        "sae_revision": S.SAELENS_REVISION,
        "fitness": fitness,
        "gate_bm": verdict,
        "gate_bm_thresholds": {"halt": GATE_BM_HALT, "lmsys_warn_top": GATE_BM_WARN_TOP},
        "published_fve": "none exists (README read at plan time) — thresholds sit at the "
        "structural-breakage band per the reuse-gate calibration rule (plan §11 R6)",
        "sparsity_spearman_lmsys": sparsity_spearman,
        "hook_fve_lmsys": hook_fve,
        "hook_probe_layers": list(probe_layers),
        "hook_alignment_l20_maximal": bool(hook_max_at_20),
        "prefix_end_cos_min_vs_row0": round(cos_min_prefix, 6),
        "prefix_arm_full": bool(cos_min_prefix < PREFIX_CONSTANCY_COS_MIN),
        "g2m_row_cos_min": round(g2m_row_min, 6),
        "g2m_flat_cos": round(g2m_flat, 6),
        "g2m_n_rows": n_gate,
        "g2m_n_stored_matched": len(row_cos),
        "fit_kernel_pilot": {
            "d": d_kernel,
            "eigh_s": round(eigh_s, 2),
            "xty_gemm_s_per_block": round(gemm_s, 3),
            "block_rows": n_blk,
            "match_chunk_s": round(match_s, 3),
            "match_chunk_rows": chunk_rows,
            "device": str(dev),
        },
        "gamma_l20": round(EL._gamma_of(gam), 3),
        "descope": descope,
        "tiny_model": bool(args.tiny_model),
        "smoke_demoted": bool(args.smoke),
    }
    _write_json(args.out_eval / "m_pilot.json", doc)
    logger.info(
        "[m0] tps=%.0f fve lmsys=%.4f pile=%.4f gate_bm=%s g2m row=%.4f flat=%.4f "
        "hook_l20_max=%s prefix_cos=%.5f eigh=%.1fs",
        tps,
        fitness["lmsys"]["fve"],
        fitness["pile"]["fve"],
        verdict["action"],
        g2m_row_min,
        g2m_flat,
        hook_max_at_20,
        cos_min_prefix,
        eigh_s,
    )
    if not args.smoke:
        if g2m_row_min < G2M_ROW_COS_MIN or g2m_flat < G2M_FLAT_COS_MIN:
            _sentinel(
                "g2m-halt",
                f"G2-m identity gate FAILED row_min={g2m_row_min:.6f} flat={g2m_flat:.6f}",
            )
            raise SystemExit(RC_G2M)
        if not hook_max_at_20:
            _sentinel(
                "hook-halt",
                f"hook alignment NOT layer-20-maximal ({hook_fve}) — off-by-one "
                "signature (plan §7: indexing fix, not a re-plan)",
            )
            raise SystemExit(RC_HOOK)
        if verdict["action"] == "halt":
            _sentinel(
                "gate-bm-halt",
                f"Gate B-m structural HALT (lmsys fve={fitness['lmsys']['fve']}, "
                f"pile fve={fitness['pile']['fve']}) — encode convention broken "
                "(plan §7 kill criterion)",
            )
            raise SystemExit(RC_GATE_BM)
        if verdict["action"] == "proceed_lmsys_only":
            logger.warning(
                "[m0] pile dictionary below the structural floor (fve=%.4f) — "
                "proceeding lmsys-only (plan §7 scope-item-6 descope, loud)",
                fitness["pile"]["fve"],
            )
        if doc["prefix_arm_full"]:
            logger.warning(
                "[m0] prefix NOT constant at L20 (min cos %.6f) — prefix arm runs as a "
                "full mapping arm (plan §7 design branch)",
                cos_min_prefix,
            )
    _sentinel("pilot", f"M0 done: tps={tps:.0f} gate_bm={verdict['action']}")
    EL._record_phase_time(args, "pilot", time.time() - t0)


# ── M1: capture + dual encode ──────────────────────────────────────────────────────


@torch.no_grad()
def _sink_answer_features(sae, h: torch.Tensor, context_end: int, ans_ids, sink_ids) -> dict:
    """Sink-robustness answer pooling for ONE dictionary (EL._extra_answer_features
    with the k128 leg dropped — the SAME reference keep mask as EA._row_features:
    bit-identical S.token_inlier_mask + BOS strip, so default-vs-sink is a paired
    read over identical rows)."""
    keep = S.token_inlier_mask(h)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ans_keep = keep[context_end + 1 :]
    h_ans = h[context_end + 1 :]
    ans_all_out = bool(h_ans.shape[0] > 0 and int(ans_keep.sum()) == 0)
    kept_rows = h_ans if ans_all_out else h_ans[ans_keep]
    kept_pos = np.arange(len(ans_ids)) if ans_all_out else np.where(ans_keep.cpu().numpy())[0]
    f = sae.encode(kept_rows)
    not_sink = ~np.isin(np.asarray(ans_ids)[kept_pos], sink_ids)
    sink_n_excl = int((~not_sink).sum())
    if bool(not_sink.any()):
        trio = S.pool_answer_features(f[torch.from_numpy(np.where(not_sink)[0])])
        fallback = 0
    else:  # every kept answer token is a sink token: fall back to the default pool
        trio = S.pool_answer_features(f)
        fallback = 1
    return {"sink": S.sparsify(trio), "sink_n_excl": sink_n_excl, "sink_fallback": fallback}


def _assert_store_regime(args) -> None:
    """Regime-keyed out-root (resume safety, #722 r3): every output-affecting key is
    pinned in store_m/regime.json; a mismatched resume fails loud."""
    regime = {
        "smoke": bool(args.smoke),
        "layer": L_TIER,
        "sae_revision": S.SAELENS_REVISION,
        "subsample_seed": SUBSAMPLE_SEED,
        "panel_seed": PANEL_SEED,
        "carve_seed": CARVE_SEED,
        "n_fit": args.n_fit,
        "n_score": args.n_score,
        "max_chunks": args.max_chunks,
        "consolidate_chunks": CONSOLIDATE_CHUNKS,
        "tiny_model": bool(args.tiny_model),
    }
    path = args.store / "regime.json"
    if path.exists():
        prev = json.loads(path.read_text())
        assert prev == regime, f"store regime mismatch: {prev} != {regime} (out_root reuse)"
    else:
        args.store.mkdir(parents=True, exist_ok=True)
        C.write_json_atomic(path, regime)


def _subsample_m(args) -> dict:
    """REUSE the early-layer rows (sha-asserted) with the smoke chunk restriction /
    pilot-descope seeded subset applied; writes split_indices_matryoshka.npz +
    m_split.json (plan §4 M1)."""
    rows = _load_reused_rows(args)
    s_fit, s_score = rows["s_fit"], rows["s_score"]
    row_ci, prov_u8 = rows["row_ci"], rows["prov_u8"]
    if args.max_chunks > 0:  # smoke SCALE knob — restrict to the enumerated chunks
        dns = argparse.Namespace(max_chunks=args.max_chunks, scratch=args.scratch)
        names = EA._raw_chunk_names(dns)
        universe = EL._chunk_ci_universe(args, names)
        s_fit = s_fit[np.isin(row_ci[s_fit], list(universe))]
        s_score = s_score[np.isin(row_ci[s_score], list(universe))]
        logger.info(
            "[m1] chunk-restricted reused rows (max_chunks=%d): fit %d, score %d",
            args.max_chunks,
            len(s_fit),
            len(s_score),
        )
    n_fit_req, n_score_req = args.n_fit, args.n_score
    pilot_doc = args.out_eval / "m_pilot.json"
    descope = None
    if pilot_doc.exists():
        descope = json.loads(pilot_doc.read_text()).get("descope")
        if descope is not None:
            n_fit_req, n_score_req = int(descope["n_fit"]), int(descope["n_score"])
            logger.warning(
                "[m1] pilot descope honored: n_fit=%d n_score=%d", n_fit_req, n_score_req
            )
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    if len(s_fit) > n_fit_req:  # descope = seeded subset of the SAME rows (plan §7)
        s_fit = np.sort(rng.choice(s_fit, size=n_fit_req, replace=False))
    if len(s_score) > n_score_req:
        s_score = np.sort(rng.choice(s_score, size=n_score_req, replace=False))
    if not args.smoke and descope is None:
        assert len(s_fit) == args.n_fit and len(s_score) == args.n_score, (
            f"production row reuse must be exact: {len(s_fit)}/{len(s_score)}"
        )
    args.store.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.store / "split_indices_matryoshka.npz",
        s_fit=s_fit,
        s_score=s_score,
        prov_fit=prov_u8[s_fit],
        prov_score=prov_u8[s_score],
    )
    _write_json(
        args.out_eval / "m_split.json",
        {
            "reused_from": "early-layer round split_indices_early.npz "
            f"@ HF rev {EARLY_SPLIT_HF_REVISION}",
            "source_shas": dict(EARLY_SPLIT_SHAS),
            "realized": {
                "n_fit": int(len(s_fit)),
                "n_score": int(len(s_score)),
                "s_fit_sha256": EL._sha_ids(s_fit),
                "s_score_sha256": EL._sha_ids(s_score),
            },
            "subsample_seed": SUBSAMPLE_SEED,
            "max_chunks": args.max_chunks,
            "descope": descope,
        },
    )
    return {"s_fit": s_fit, "s_score": s_score, "row_ci": row_ci, "prov_u8": prov_u8}


_POOLED_INT_KEYS = {
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
}


def _new_rec(fam: str) -> dict[str, list]:
    keys = [
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
        "ctx_n_out",
        "ans_n_out",
        "ans_all_out",
    ]
    if fam == "lmsys":
        keys += [
            "sink_off",
            "sink_idx",
            "sink_mean",
            "sink_max",
            "sink_frac",
            "sink_n_excl",
            "sink_fallback",
        ]
    return {k: [] for k in keys}


def _flush_rec(rec: dict[str, list], path: Path) -> None:
    """Atomic consolidated-shard write (parent schema dtypes)."""
    arrays: dict[str, np.ndarray] = {}
    for kk, vals in rec.items():
        if kk in _POOLED_INT_KEYS:
            arrays[kk] = np.asarray(vals, _POOLED_INT_KEYS[kk])
        elif kk.endswith("_idx"):
            arrays[kk] = np.concatenate(vals) if vals else np.empty(0, np.int32)
        else:
            arrays[kk] = np.concatenate(vals) if vals else np.empty(0, np.float16)
    tmp = path.parent / f".tmp_{path.name}"
    np.savez(tmp, **arrays)
    tmp.replace(path)


def phase_capture(args) -> None:
    """M1: reused rows -> teacher-forced capture @ L20 -> dual encode + poolings +
    dense columns; per-GROUP consolidated shards with resume-skip (plan §4 M1)."""
    t0 = time.time()
    EA._headroom(args.store, 2 if args.smoke else 18, "m1-capture")
    EL._stage_scratch_meta(args)
    _stage_early_split(args)
    _assert_store_regime(args)
    sub = _subsample_m(args)
    s_fit, s_score, row_ci = sub["s_fit"], sub["s_score"], sub["row_ci"]
    set_tag = {int(r): 1 for r in s_fit}
    set_tag.update({int(r): 0 for r in s_score})
    needed_ci = {int(row_ci[r]): int(r) for r in set_tag}
    assert -1 not in needed_ci, "subsample rows must be NEW rows (text-resolvable)"
    fams = _active_families(args)
    for fam in fams:
        S.SAELensJumpReLU.ensure_downloaded(SAE_IDS[fam], args.sae_dir)
    model, tok = EA._load_model_tok(args)
    saes = {
        fam: S.SAELensJumpReLU.load(SAE_IDS[fam], device=args.device, cache_dir=args.sae_dir)
        for fam in fams
    }
    sink_ids = EL._sink_token_ids(tok)
    prefix_chars = EA._prefix_char_len(tok)
    dns = argparse.Namespace(max_chunks=args.max_chunks, scratch=args.scratch)
    names = EA._raw_chunk_names(dns)
    groups = [names[i : i + CONSOLIDATE_CHUNKS] for i in range(0, len(names), CONSOLIDATE_CHUNKS)]
    n_done = 0
    tok_count = 0
    t_loop = time.time()
    for gi, group in enumerate(groups):
        paths = {fam: args.store / f"pooled_m_{fam}_g{gi:04d}.npz" for fam in fams}
        dense_path = args.store / f"dense_l20_g{gi:04d}.npz"
        if dense_path.exists() and all(p.exists() for p in paths.values()):
            continue  # resume-skip: the whole consolidated group landed atomically
        recs = {fam: _new_rec(fam) for fam in fams}
        dense: dict[str, list] = {"row_idx": [], "c20": [], "hp20": []}
        for name, keep in EA._iter_needed_rows(dns, group, needed_ci):
            rows = []
            for row_idx, ci, prompt, response in keep:
                tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
                if tk is None:
                    continue
                full_ids, prefix_end, context_end, n_ans, seam = tk
                rows.append((row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam))
            rows.sort(key=lambda r: len(r[2]))
            for s0 in range(0, len(rows), args.gen_batch):
                batch = rows[s0 : s0 + args.gen_batch]
                caps = EA._batched_capture(model, tok, batch, (L_TIER,), args.device)
                for (row_idx, ci, full_ids, prefix_end, context_end, n_ans, seam), cap in zip(
                    batch, caps, strict=True
                ):
                    h20 = cap[L_TIER]
                    ans_ids = np.asarray(full_ids[context_end + 1 :], dtype=np.int64)
                    for fam in fams:
                        rec = recs[fam]
                        sp, spm, spl, ctx_n_out, ans_n_out, ans_all_out = EA._row_features(
                            saes[fam], h20, context_end
                        )
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
                        rec["ctx_n_out"].append(ctx_n_out)
                        rec["ans_n_out"].append(ans_n_out)
                        rec["ans_all_out"].append(ans_all_out)
                        if fam == "lmsys":  # sink-robustness twin pooling (plan §4 M1)
                            extra = _sink_answer_features(
                                saes[fam], h20, context_end, ans_ids, sink_ids
                            )
                            rec["sink_off"].append(len(extra["sink"]["idx"]))
                            rec["sink_idx"].append(extra["sink"]["idx"])
                            rec["sink_mean"].append(extra["sink"]["mean"])
                            rec["sink_max"].append(extra["sink"]["max"])
                            rec["sink_frac"].append(extra["sink"]["frac"])
                            rec["sink_n_excl"].append(extra["sink_n_excl"])
                            rec["sink_fallback"].append(extra["sink_fallback"])
                    dense["row_idx"].append(row_idx)
                    dense["c20"].append(h20[context_end].numpy().astype(np.float16))
                    dense["hp20"].append(h20[prefix_end].numpy().astype(np.float16))
                    tok_count += len(full_ids)
                    n_done += 1
        for fam in fams:
            _flush_rec(recs[fam], paths[fam])
        dtmp = dense_path.parent / f".tmp_{dense_path.name}"
        np.savez(
            dtmp,
            row_idx=np.asarray(dense["row_idx"], np.int64),
            c20=np.stack(dense["c20"]) if dense["c20"] else np.empty((0, 3584), np.float16),
            hp20=np.stack(dense["hp20"]) if dense["hp20"] else np.empty((0, 3584), np.float16),
        )
        dtmp.replace(dense_path)
        print(
            f"[m1] unit {gi + 1}/{len(groups)} rows_total={n_done} tok={tok_count} "
            f"elapsed={time.time() - t_loop:.0f}s",
            flush=True,
        )
    logger.info("[m1] capture done: %d contexts, %d tokens (fams=%s)", n_done, tok_count, fams)
    _sentinel("capture", f"M1 done ({n_done} contexts, {tok_count} tokens, fams={fams})")
    EL._record_phase_time(args, "capture", time.time() - t0)


# ── M2 / M4: uploads ─────────────────────────────────────────────────────────────


def phase_upload1(args) -> None:
    """M2: consolidated store -> HF BEFORE any fit (#825 rule). delete_local=False —
    M3 consumes the local store."""
    t0 = time.time()
    if args.skip_upload:
        logger.warning("[m2] --skip-upload: store upload SKIPPED (local-only run)")
        _sentinel("upload1", "M2 SKIPPED (--skip-upload)")
        return
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    prefix = _m_hf_prefix(args)
    res = upload_dir_sharded(
        args.store,
        HF_DATA_REPO,
        f"{prefix}/store",
        repo_type="dataset",
        shard_glob="*.npz",
        verify=True,
        delete_local=False,
    )
    hub._upload(
        args.store / "regime.json",
        HF_DATA_REPO,
        "dataset",
        f"{prefix}/store/regime.json",
        upload_as_file=True,
        raise_on_error=True,
    )
    logger.info(
        "[m2] store upload done: %d shards -> %s (rerouted=%s)",
        len(res.uploaded),
        prefix,
        res.rerouted,
    )
    _sentinel("upload1", f"M2 done ({len(res.uploaded)} shards -> {prefix}/store)")
    EL._record_phase_time(args, "upload1", time.time() - t0)


def phase_upload2(args) -> None:
    """M4: eval outputs -> HF (text/JSON + small npz, unconditional) + the poller
    results sentinel (terminal pod phase)."""
    t0 = time.time()
    from explore_persona_space.orchestrate import hub

    prefix = _m_hf_prefix(args)
    if not args.skip_upload:
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
        assert not missing, f"M4 verify: missing on Hub: {missing}"
        logger.info("[m4] %d eval artifacts verified on Hub under %s/eval", len(expected), prefix)
    else:
        logger.warning("[m4] --skip-upload: eval upload SKIPPED")
    _results_sentinel(args)
    _sentinel("upload2", "M4 done (eval artifacts uploaded; results sentinel written)")
    EL._record_phase_time(args, "upload2", time.time() - t0)


def _results_sentinel(args, logs_dir: Path | None = None) -> None:
    """poll_pipeline.py results sentinel (SKILL.md Step 7 contract). Under --smoke
    the kind is epm:smoke-result so a --full pod run's SMOKE leg can never be
    drained as the real epm:results (#1586 chained-legs class, sentinel side)."""
    if logs_dir is None:
        logs_dir = Path("/workspace/logs")
        if not logs_dir.is_dir():
            logs_dir = PROJECT_ROOT / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((args.out_eval / "m_pilot.json").read_text())
    summary_path = args.out_eval / "m_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    split = json.loads((args.out_eval / "m_split.json").read_text())
    times = json.loads((args.out_eval / "phase_times.json").read_text())
    gpu_h = sum(p["wall_s"] for p in times["phases"]) / 3600.0
    payload = {
        "sentinel_schema_version": C.SENTINEL_SCHEMA_VERSION,
        "kind": "epm:smoke-result" if args.smoke else "epm:results",
        "version": 1,
        "task_id": TASK_ID,
        "by": "issue1482_matryoshka_tier",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("SMOKE leg — not the production result. " if args.smoke else "")
        + "issue-1482 matryoshka-tier pod phases M0-M4 complete (M5-evidence + M6 "
        "analysis run off-pod; judged labelling FROZEN per plan v21 §4 M5 — zero "
        "judge calls this round, evidence persisted for the #1773 resume)",
        "eval_numbers": {
            "gate_bm": pilot["gate_bm"],
            "fve_lmsys": pilot["fitness"]["lmsys"]["fve"],
            "fve_pile": pilot["fitness"]["pile"]["fve"],
            "l0_lmsys": pilot["fitness"]["lmsys"]["l0"],
            "l0_pile": pilot["fitness"]["pile"]["l0"],
            "g2m_row_cos_min": pilot["g2m_row_cos_min"],
            "g2m_flat_cos": pilot["g2m_flat_cos"],
            "hook_alignment_l20_maximal": pilot["hook_alignment_l20_maximal"],
            "tokens_per_s": pilot["tokens_per_s"],
            "arm_pooled_r2": summary.get("pooled_r2", {}),
            "panel": summary.get("panel", {}),
            "n_fit_realized": split["realized"]["n_fit"],
            "n_score_realized": split["realized"]["n_score"],
        },
        "eval_paths": {
            "pilot": str(args.out_eval / "m_pilot.json"),
            "split": str(args.out_eval / "m_split.json"),
            "summary": str(summary_path),
            "perfeature": str(args.out_eval),
            "store_hf_prefix": f"{_m_hf_prefix(args)}/store",
            # declared OFF-POD destinations (plan §9 off_pod_phases) — produced AFTER
            # this sentinel by the VM-side M5/M6 phases, committed to git
            "evidence_offpod_dest": str(args.out_eval / "evidence" / "evidence.json"),
            "tier_tests_offpod_dest": str(args.out_eval / "tier_tests.json"),
            "figures_offpod_dest": str(args.figures),
        },
        "reproducibility_card": {
            **C.reproducibility_metadata(),
            "layer": L_TIER,
            "sae_repo": S.SAELENS_REPO,
            "sae_revision": S.SAELENS_REVISION,
            "sae_ids": SAE_IDS,
            "tier_bounds": list(S.MATRYOSHKA_TIER_BOUNDS),
            "subsample_seed": SUBSAMPLE_SEED,
            "panel_seed": PANEL_SEED,
            "carve_seed": CARVE_SEED,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "s_fit_sha256": split["realized"]["s_fit_sha256"],
            "s_score_sha256": split["realized"]["s_score_sha256"],
            "wandb": "N/A — no training in this round (frozen teacher-forced forwards "
            "+ closed-form ridge fits logging to JSON checkpoints)",
        },
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/"
        f"{_m_hf_prefix(args)}",
        "worktree_path": str(PROJECT_ROOT),
        "final_commit_sha": C.reproducibility_metadata()["git_commit"],
        "gpu_hours_used": round(gpu_h, 2),
        "gpu_hours_budgeted": 6,
        "plan_deviations": summary.get("plan_deviations", []),
    }
    path = logs_dir / f"issue-{TASK_ID}-results.json"
    C.write_json_atomic(path, payload)
    logger.info("Wrote results sentinel %s", path)


# ── M3: fits + covariates + matching ─────────────────────────────────────────────


def _load_parts(args, fam: str) -> list[dict]:
    parts = [
        dict(np.load(p, allow_pickle=False))
        for p in sorted(args.store.glob(f"pooled_m_{fam}_*.npz"))
    ]
    assert parts, f"no pooled {fam} shards under {args.store}"
    return parts


def _add_tier_key(path: Path) -> None:
    """Post-write tier annotation of an EL._per_feature_npz output (the §6.5
    deliverable carries tier ids; EL's writer is reused verbatim, then the
    deterministic tier column is appended atomically)."""
    d = dict(np.load(path, allow_pickle=False))
    d["tier"] = S.tier_of(np.asarray(d["feat_ids"], np.int64))
    tmp = path.parent / f".tmp_{path.name}"
    np.savez(tmp, **d)
    tmp.replace(path)


def _per_feature_npz_m(args, name, feat_ids, pt, true_te, activity, te_prov, te) -> dict:
    doc = EL._per_feature_npz(args, name, feat_ids, pt, true_te, activity, te_prov, te)
    _add_tier_key(args.out_eval / f"{name}.npz")
    return doc


def _m3_prep(args) -> argparse.Namespace:
    """Shared M3 preprocessing: per-family stores loaded, ONE row registry, the
    seeded lambda carve (CARVE_SEED), per-family restrictions (input top-8192 by
    activity; target tier-stratified random panel), full-population counts."""
    fams = _active_families(args)
    parts = {fam: _load_parts(args, fam) for fam in fams}
    dense_parts = [
        dict(np.load(p, allow_pickle=False)) for p in sorted(args.store.glob("dense_l20_*.npz"))
    ]
    assert dense_parts, f"no dense shards under {args.store}"
    sub = np.load(args.store / "split_indices_matryoshka.npz")
    s_fit, s_score = sub["s_fit"], sub["s_score"]
    prov_by_row = {int(r): int(p) for r, p in zip(sub["s_fit"], sub["prov_fit"], strict=True)}
    prov_by_row.update(
        {int(r): int(p) for r, p in zip(sub["s_score"], sub["prov_score"], strict=True)}
    )
    have: set[int] | None = None
    for fam in fams:
        rows_fam = {int(r) for part in parts[fam] for r in part["row_idx"]}
        have = rows_fam if have is None else have
        assert rows_fam == have, f"{fam} store rows != sibling store rows (capture drift)"
    assert have is not None
    order = np.asarray([r for r in np.concatenate([s_fit, s_score]) if int(r) in have], np.int64)
    coverage = len(order) / max(1, len(s_fit) + len(s_score))
    deviations = []
    if not args.smoke and coverage < 0.98:
        deviations.append(f"captured-row registry covers {coverage:.4f} of the subsample (<0.98)")
    row_pos = {int(r): i for i, r in enumerate(order)}
    n_rows = len(order)
    fit_positions = np.asarray([row_pos[int(r)] for r in s_fit if int(r) in row_pos], np.int64)
    te = np.asarray([row_pos[int(r)] for r in s_score if int(r) in row_pos], np.int64)
    assert len(te) >= 2, f"score rows after intersection: {len(te)} < 2"
    # lambda carve (plan §4 M3): seeded permutation of the fit rows (CARVE_SEED 14825)
    carve = min(args.val_carve, max(1, len(fit_positions) // 6))
    perm = np.random.default_rng(CARVE_SEED).permutation(len(fit_positions))
    va = fit_positions[perm[:carve]]
    tr = fit_positions[perm[carve:]]
    assert len(tr) >= 1 and len(va) >= 1, (len(tr), len(va))
    fit_rows_set = {int(order[i]) for i in tr} | {int(order[i]) for i in va}
    out_counts: dict[str, np.ndarray] = {}
    in_counts: dict[str, np.ndarray] = {}
    f_in: dict[str, np.ndarray] = {}
    panel: dict[str, np.ndarray] = {}
    panel_doc: dict[str, dict] = {}
    n_fit_realized = 0
    for fam in fams:
        oc, n_fit_realized = EL._activity_counts_rows(
            parts[fam], "ans_idx", "idx_off", fit_rows_set, S.SAELENS_DICT_SIZE
        )
        ic, _ = EL._activity_counts_rows(
            parts[fam], "psi_idx", "psi_off", fit_rows_set, S.SAELENS_DICT_SIZE
        )
        out_counts[fam], in_counts[fam] = oc, ic
        f_in[fam] = EL._restrict(ic, n_fit_realized, args.max_features_in)  # parent recipe
        panel[fam], panel_doc[fam] = _tier_stratified_panel(
            oc, n_fit_realized, args.max_features_out, PANEL_SEED
        )
        assert len(f_in[fam]) >= 1 and len(panel[fam]) >= 1, fam
        if not args.smoke and panel_doc[fam]["per_tier_floor_warn"]:
            deviations.append(
                f"{fam}: tiers {panel_doc[fam]['per_tier_floor_warn']} below the "
                f"{PER_TIER_PANEL_FLOOR} floor-clearing floor (WARN; quintile merge engages)"
            )
    _write_json(args.out_eval / "m_panel.json", {"families": panel_doc})
    d_widest = 2 * max(len(f_in[fam]) for fam in fams)
    EL._assert_estimator_validity(len(tr), d_widest, args.smoke)
    logger.info(
        "[m3] registry %d rows (coverage %.4f); tr/va/te=%d/%d/%d; panels %s",
        n_rows,
        coverage,
        len(tr),
        len(va),
        len(te),
        {f: int(len(panel[f])) for f in fams},
    )
    te_prov = np.asarray([prov_by_row[int(order[i])] for i in te], np.int8)
    return argparse.Namespace(
        fams=fams,
        parts=parts,
        dense_parts=dense_parts,
        order=order,
        row_pos=row_pos,
        n_rows=n_rows,
        tr=tr,
        va=va,
        te=te,
        te_prov=te_prov,
        fit_rows_set=fit_rows_set,
        n_fit=n_fit_realized,
        f_in=f_in,
        panel=panel,
        panel_doc=panel_doc,
        out_counts=out_counts,
        in_counts=in_counts,
        deviations=deviations,
    )


def _targets_from(prep, fam: str, feat_ids: np.ndarray, keys: tuple[str, str, str]) -> np.ndarray:
    key_idx, key_off, key_val = keys
    return EA._densify(
        prep.parts[fam], key_idx, key_off, key_val, feat_ids, prep.n_rows, prep.row_pos
    )


def _shuffle_null(args, prep, fam: str, z: np.ndarray, y_mean: np.ndarray, lam: float, dev):
    """Label-shuffle null K=20 at the pinned lambda (ONE factorization + K X^TY
    GEMMs; answer rows permuted within the train pool, scored on true te pairs —
    the parent recipe verbatim, per family)."""
    fac = N1M._ridge_factorize(z, y_mean, prep.tr, dev, N1M.RIDGE_BLOCK)
    u, s_eig = fac["U"], fac["s_eig"]
    xmu, xsd, ymu = fac["xmu"], fac["xsd"], fac["ymu"]
    null_r2 = np.zeros((len(SHUFFLE_SEEDS), len(prep.panel[fam])), dtype=np.float16)
    true_te = y_mean[prep.te]
    for si, seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.default_rng(seed)
        tr_perm = prep.tr[rng.permutation(len(prep.tr))]
        xty = torch.zeros((z.shape[1], y_mean.shape[1]), dtype=torch.float64, device=dev)
        for s0 in range(0, len(prep.tr), N1M.RIDGE_BLOCK):
            xb = (
                torch.as_tensor(
                    z[prep.tr[s0 : s0 + N1M.RIDGE_BLOCK]], dtype=torch.float64, device=dev
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
        en = (torch.as_tensor(z[prep.te], dtype=torch.float64, device=dev) - xmu) / xsd
        pt_null = ((en @ w) + ymu).cpu().numpy()
        null_r2[si] = EA._per_feature_metrics(pt_null, true_te)["r2"].astype(np.float16)
        print(f"[m3-null-{fam}] draw {si + 1}/{len(SHUFFLE_SEEDS)} seed={seed}", flush=True)
    np.savez(
        args.out_eval / f"shuffle_null_m_{fam}.npz",
        feat_ids=prep.panel[fam],
        tier=S.tier_of(prep.panel[fam]),
        r2=null_r2,
        seeds=np.asarray(SHUFFLE_SEEDS, np.int64),
        selected_lambda=np.float64(lam),
    )
    del fac, u, s_eig


def _ctx_unit(args, prep, fam: str, summary: dict, dev) -> None:
    """Per-family context-features arm: Z=[psi_mean, psi_last]; targets mean/max/frac
    (+ lmsys sinkmask trio) share ONE factorization — the registered paired-contrast
    row coverage (plan §3)."""
    done_key = f"perfeature_m_{fam}_default.npz"
    if (args.out_eval / done_key).exists():
        logger.info("[m3] %s ctx unit exists; resume-skip", fam)
        return
    psi_mean = _targets_from(prep, fam, prep.f_in[fam], ("psi_idx", "psi_off", "psi_mean"))
    psi_last = _targets_from(prep, fam, prep.f_in[fam], ("psil_idx", "psil_off", "psil_val"))
    z = np.concatenate([psi_mean, psi_last], axis=1)
    pn = prep.panel[fam]
    tgts = {
        "mean": _targets_from(prep, fam, pn, ("ans_idx", "idx_off", "ans_mean")),
        "max": _targets_from(prep, fam, pn, ("ans_idx", "idx_off", "ans_max")),
        "frac": _targets_from(prep, fam, pn, ("ans_idx", "idx_off", "ans_frac")),
    }
    if fam == "lmsys":
        for pool in ("mean", "max", "frac"):
            tgts[f"sink_{pool}"] = _targets_from(
                prep, fam, pn, ("sink_idx", "sink_off", f"sink_{pool}")
            )
    preds = EA._shared_gram_ridge_multi(
        z, tgts, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
    )
    act = prep.out_counts[fam][pn] / max(1, prep.n_fit)
    for tname, (pt, meta) in preds.items():
        name = (
            f"perfeature_m_{fam}_default"
            if tname == "mean"
            else (
                f"perfeature_m_{fam}_sinkmask"
                if tname == "sink_mean"
                else f"perfeature_m_{fam}_{tname.replace('sink_', 'sinkmask_')}"
            )
        )
        doc = _per_feature_npz_m(
            args, name, pn, pt, tgts[tname][prep.te], act, prep.te_prov, prep.te
        )
        summary["pooled_r2"][name] = doc["pooled_r2"]
        summary["selected_lambda"][name] = meta["selected_lambda"]
        summary["splithalf"][name] = doc["splithalf_rank_stability"]
        if tname == "mean":
            summary["knn"][f"{fam}_ctx"] = EL._knn_reads(pt, tgts["mean"][prep.te])
            # standing identity+learned-bias baseline on the ALIGNED feature-id
            # intersection (psi_mean -> ans_mean on shared ids); full-design identity
            # is inapplicable across feature-id coordinates; the dense arms' 3,584 ->
            # panel mismatch is stated inapplicable (plan §4 M3 dispositions).
            from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

            shared = np.intersect1d(prep.f_in[fam], pn)
            if len(shared) >= 2:
                xi = psi_mean[:, np.searchsorted(prep.f_in[fam], shared)]
                yi = tgts["mean"][:, np.searchsorted(pn, shared)]
                pred_i = identity_bias_predict(xi[prep.tr], yi[prep.tr], xi[prep.te])
                summary["baselines"][f"{fam}_identity_bias"] = {
                    "n_shared_ids": int(len(shared)),
                    "pooled_r2": float(PR._pooled_r2(pred_i, yi[prep.te])),
                    "knn": EL._knn_reads(pred_i, yi[prep.te]),
                }
            else:
                summary["baselines"][f"{fam}_identity_bias"] = {
                    "n_shared_ids": int(len(shared)),
                    "note": "aligned intersection too small",
                }
    summary["baselines"]["dense_arms_identity"] = (
        "inapplicable — 3,584-dim input vs feature-space output (stated per the "
        "standing mapping-baselines rule)"
    )
    lam = float(summary["selected_lambda"][f"perfeature_m_{fam}_default"])
    _shuffle_null(args, prep, fam, z, tgts["mean"], lam, dev)
    del z, psi_mean, psi_last, tgts, preds
    _write_json(args.out_eval / "m_summary.json", summary)
    print(f"[m3] ctx unit ({fam}) done", flush=True)


def _dense_unit(args, prep, summary: dict, dev) -> None:
    """Dense-input companions (Z = c_last@20, d=3,584) — one factorization covers
    every family's panel targets (shared design)."""
    first = f"perfeature_m_{prep.fams[0]}_dense_in.npz"
    if (args.out_eval / first).exists():
        logger.info("[m3] dense unit exists; resume-skip")
        return
    z20 = EL._dense_design(prep.dense_parts, "c20", prep.n_rows, prep.row_pos)
    tgts = {}
    for fam in prep.fams:
        for pool in ("mean", "max", "frac"):
            tgts[f"{fam}:{pool}"] = _targets_from(
                prep, fam, prep.panel[fam], ("ans_idx", "idx_off", f"ans_{pool}")
            )
    preds = EA._shared_gram_ridge_multi(
        z20, tgts, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
    )
    for tname, (pt, meta) in preds.items():
        fam, pool = tname.split(":")
        act = prep.out_counts[fam][prep.panel[fam]] / max(1, prep.n_fit)
        name = f"perfeature_m_{fam}_dense_in" + ("" if pool == "mean" else f"_{pool}")
        doc = _per_feature_npz_m(
            args, name, prep.panel[fam], pt, tgts[tname][prep.te], act, prep.te_prov, prep.te
        )
        summary["pooled_r2"][name] = doc["pooled_r2"]
        summary["selected_lambda"][name] = meta["selected_lambda"]
        if pool == "mean":
            summary["knn"][f"{fam}_dense_in"] = EL._knn_reads(pt, tgts[tname][prep.te])
    del z20, tgts, preds
    _write_json(args.out_eval / "m_summary.json", summary)
    print("[m3] dense unit done", flush=True)


def _prefix_unit(args, prep, summary: dict, dev) -> None:
    """Prefix-arm nulls (Z = prefix-end@20; the standing prefix mapping arm — on this
    single-turn corpus the M0 constancy gate makes the null-level read the expected
    outcome; a non-constant prefix runs this as a full mapping arm, plan §7)."""
    first = f"perfeature_m_{prep.fams[0]}_prefix_null.npz"
    if (args.out_eval / first).exists():
        logger.info("[m3] prefix unit exists; resume-skip")
        return
    hp20 = EL._dense_design(prep.dense_parts, "hp20", prep.n_rows, prep.row_pos)
    tgts = {
        f"{fam}:mean": _targets_from(prep, fam, prep.panel[fam], ("ans_idx", "idx_off", "ans_mean"))
        for fam in prep.fams
    }
    # Degenerate-constant guard (plan §7 prefix constancy): on this single-turn
    # corpus the prefix-end state is constant across rows up to batch jitter
    # (M0 measured cos ~1.0), so the standardized design is amplified noise on
    # a handful of dims + EXACT-zero columns everywhere else, and LAPACK syevd
    # cannot converge on its Gram (the #1335 eigh non-convergence class; smoke
    # measured 3540/3584 dims with train sd exactly 0). The ridge lambda->inf
    # limit IS the exact fit on a zero-information design, so a design measured
    # degenerate (relative train sd < 1e-3, or >50% exactly-constant dims)
    # takes the analytic limit (train-mean predictor) directly, DISCLOSED via
    # summary["prefix_degenerate"] + selected_lambda=None. A non-degenerate
    # prefix keeps the exact shared-Gram eigh path (the same code path the ctx
    # + dense units exercise); an eigh non-convergence there falls back to the
    # same analytic null read with the error string disclosed — scoped to this
    # NULL arm only (ctx/dense units keep hard-fail).
    x_tr = hp20[prep.tr]
    sd = np.std(x_tr, axis=0)
    rel_sd = float(np.max(sd)) / (float(np.max(np.abs(np.mean(x_tr, axis=0)))) + 1e-12)
    frac_const = float((sd == 0).mean())

    def _prefix_null_analytic(tgts_in: dict) -> dict:
        out = {}
        for tname, y_all in tgts_in.items():
            ymu = y_all[prep.tr].mean(axis=0)
            pv = np.tile(ymu, (len(prep.va), 1))
            pt = np.tile(ymu, (len(prep.te), 1))
            out[tname] = (
                pt,
                {"selected_lambda": None, "val_r2": float(PR._pooled_r2(pv, y_all[prep.va]))},
            )
        return out

    if rel_sd < 1e-3 or frac_const > 0.5:
        logger.warning(
            "[m3] prefix design degenerate-constant (rel_sd=%.3e, frac_const=%.3f); "
            "analytic lambda->inf null read (plan §7 expected outcome)",
            rel_sd,
            frac_const,
        )
        preds = _prefix_null_analytic(tgts)
        summary["prefix_degenerate"] = {"rel_sd": rel_sd, "frac_const_dims": frac_const}
    else:
        try:
            preds = EA._shared_gram_ridge_multi(
                hp20, tgts, prep.tr, prep.va, prep.te, N1M.LAMBDAS_N1M, dev, N1M.RIDGE_BLOCK
            )
        except torch.linalg.LinAlgError as e:
            logger.warning(
                "[m3] prefix eigh non-convergence (%s); analytic lambda->inf null read", e
            )
            preds = _prefix_null_analytic(tgts)
            summary["prefix_degenerate"] = {
                "rel_sd": rel_sd,
                "frac_const_dims": frac_const,
                "eigh_error": str(e),
            }
    for tname, (pt, meta) in preds.items():
        fam = tname.split(":")[0]
        act = prep.out_counts[fam][prep.panel[fam]] / max(1, prep.n_fit)
        name = f"perfeature_m_{fam}_prefix_null"
        doc = _per_feature_npz_m(
            args, name, prep.panel[fam], pt, tgts[tname][prep.te], act, prep.te_prov, prep.te
        )
        summary["pooled_r2"][name] = doc["pooled_r2"]
        summary["selected_lambda"][name] = meta["selected_lambda"]
        summary["knn"][f"{fam}_prefix_null"] = EL._knn_reads(pt, tgts[tname][prep.te])
    del hp20, tgts, preds
    _write_json(args.out_eval / "m_summary.json", summary)
    print("[m3] prefix unit done", flush=True)


def _covariates_unit_m(args, prep, summary: dict) -> None:
    """Covariate battery per family (plan §4 M3): activity, consistency, footprint,
    co-activation, dense flag, r_B decoder alignment at row 20 — PER-TRAIT and
    max-over-traits, RAW and population-CENTERED — plus full-population npz."""
    if (args.out_eval / f"covariates_m_{prep.fams[0]}.npz").exists():
        logger.info("[m3] covariates exist; resume-skip")
        return
    dev = torch.device(args.device)
    w_u, g, _tok = EL._load_wu_norm(args)
    pilot = json.loads((args.out_eval / "m_pilot.json").read_text())
    rb, traits = EL._load_rb_rows(args, L_TIER)
    rb_norm = rb / np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-12)
    for fam in prep.fams:
        sae = S.SAELensJumpReLU.load(SAE_IDS[fam], device="cpu", cache_dir=args.sae_dir)
        pn = prep.panel[fam]
        dec = sae.w_dec[torch.as_tensor(pn, dtype=torch.long)].T.contiguous()  # (3584, n)
        fp = EL._footprint(w_u, g, dec, dev)
        coact = EL._coactivation(prep.parts[fam], pn, prep.fit_rows_set, dev)
        consistency_full = EL._consistency_rows(
            prep.parts[fam], prep.fit_rows_set, S.SAELENS_DICT_SIZE
        )
        activity = prep.out_counts[fam][pn] / max(1, prep.n_fit)
        dense_flag = (activity > 0.5).astype(np.int8)
        decile = np.searchsorted(
            np.quantile(activity, np.linspace(0, 1, 11)[1:-1]), activity, side="right"
        )
        top_decile = (decile == 9).astype(np.int8)
        d_np = dec.numpy().astype(np.float64)
        d_norm = d_np / np.maximum(np.linalg.norm(d_np, axis=0, keepdims=True), 1e-12)
        raw_pt = np.abs(rb_norm @ d_norm)  # (n_traits, n_panel)
        d_cent = d_np - d_np.mean(axis=1, keepdims=True)
        d_cent /= np.maximum(np.linalg.norm(d_cent, axis=0, keepdims=True), 1e-12)
        cent_pt = np.abs(rb_norm @ d_cent)
        arrays = {
            "feat_ids": pn,
            "tier": S.tier_of(pn),
            "activity": activity,
            "consistency": consistency_full[pn],
            "footprint_conc": fp["conc"],
            "footprint_norm": fp["norm"],
            "coact": coact,
            "dense_flag": dense_flag,
            "top_decile_flag": top_decile,
            "rb_raw_maxabs": raw_pt.max(axis=0),
            "rb_centered_maxabs": cent_pt.max(axis=0),
        }
        for ti, trait in enumerate(traits):
            arrays[f"rb_raw_{trait}"] = raw_pt[ti]
            arrays[f"rb_cent_{trait}"] = cent_pt[ti]
        np.savez(args.out_eval / f"covariates_m_{fam}.npz", **arrays)
        # full-population descriptives (all 65,536 features; plan §4 M3)
        np.savez(
            args.out_eval / f"population_m_{fam}.npz",
            counts=prep.out_counts[fam].astype(np.int64),
            activity=(prep.out_counts[fam] / max(1, prep.n_fit)).astype(np.float32),
            consistency=consistency_full.astype(np.float32),
            tier=S.tier_of(np.arange(S.SAELENS_DICT_SIZE)),
            n_fit=np.int64(prep.n_fit),
        )
        summary.setdefault("covariates", {})[fam] = {
            "n_features": int(len(pn)),
            "n_dense_flag": int(dense_flag.sum()),
            "gamma_l20": pilot["gamma_l20"],
            "rb_traits": traits,
        }
        del sae, dec
        print(f"[m3] covariates ({fam}) done ({len(pn)} features)", flush=True)
    if args.device == "cuda":
        torch.cuda.empty_cache()


def _matching_unit(args, prep, summary: dict) -> None:
    """Cross-dictionary decoder matching (plan §4 M3): unit-normalized decoder rows,
    chunked GEMM D_lmsys @ D_pile^T with RUNNING row/col maxes — the 65,536^2 matrix
    is never materialized. fp16 on cuda, fp32 on cpu (fp16 CPU matmul is unusably
    slow)."""
    if (args.out_eval / "matching.npz").exists():
        logger.info("[m3] matching exists; resume-skip")
        return
    if "pile" not in prep.fams:
        summary["matching"] = "N/A — pile dictionary dropped at the structural floor"
        return
    dev = torch.device(args.device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    sae_l = S.SAELensJumpReLU.load(SAE_IDS["lmsys"], device="cpu", cache_dir=args.sae_dir)
    sae_p = S.SAELensJumpReLU.load(SAE_IDS["pile"], device="cpu", cache_dir=args.sae_dir)

    def _unit_rows(w: torch.Tensor) -> torch.Tensor:
        n = torch.linalg.norm(w.to(torch.float32), dim=1, keepdim=True).clamp_min(1e-12)
        return (w.to(torch.float32) / n).to(device=dev, dtype=dtype)

    d_l = _unit_rows(sae_l.w_dec)
    d_p = _unit_rows(sae_p.w_dec)
    n_rows = int(args.match_rows) if args.match_rows > 0 else d_l.shape[0]
    n_rows = min(n_rows, d_l.shape[0])
    best_l = torch.full((d_l.shape[0],), float("nan"), dtype=torch.float32)
    best_p = torch.full((d_p.shape[0],), -float("inf"), dtype=torch.float32)
    t0 = time.time()
    n_chunks = (n_rows + MATCH_CHUNK_ROWS - 1) // MATCH_CHUNK_ROWS
    for ki, s0 in enumerate(range(0, n_rows, MATCH_CHUNK_ROWS)):
        blk = (d_l[s0 : s0 + MATCH_CHUNK_ROWS] @ d_p.T).to(torch.float32)
        best_l[s0 : s0 + MATCH_CHUNK_ROWS] = blk.max(dim=1).values.cpu()
        best_p = torch.maximum(best_p, blk.max(dim=0).values.cpu())
        print(f"[m3-match] chunk {ki + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s", flush=True)
    np.savez(
        args.out_eval / "matching.npz",
        best_lmsys_to_pile=best_l.numpy().astype(np.float16),
        best_pile_to_lmsys=best_p.numpy().astype(np.float16),
        match_rows=np.int64(n_rows),
    )
    summary["matching"] = {
        "match_rows": int(n_rows),
        "full": bool(n_rows == d_l.shape[0]),
        "note": "best_pile_to_lmsys covers only the matched lmsys row slice when partial",
    }
    del sae_l, sae_p, d_l, d_p
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print("[m3] matching unit done", flush=True)


def phase_fits(args) -> None:
    """M3: shared-Gram ridge arms + shuffle nulls + covariates + mapping baselines +
    cross-dict matching; sequential units with output-exists resume skips."""
    t0 = time.time()
    EA._headroom(args.scratch, 2 if args.smoke else 8, "m3-fits")
    prep = _m3_prep(args)
    dev = torch.device(args.device)
    summary_path = args.out_eval / "m_summary.json"
    summary: dict = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else {
            "pooled_r2": {},
            "selected_lambda": {},
            "splithalf": {},
            "knn": {},
            "baselines": {},
            "plan_deviations": [],
        }
    )
    summary["n_rows"] = {"tr": int(len(prep.tr)), "va": int(len(prep.va)), "te": int(len(prep.te))}
    summary["panel"] = prep.panel_doc
    summary["n_features_in"] = {fam: int(len(prep.f_in[fam])) for fam in prep.fams}
    summary["active_families"] = list(prep.fams)
    for d in prep.deviations:
        if d not in summary["plan_deviations"]:
            summary["plan_deviations"].append(d)
    for fam in prep.fams:
        _ctx_unit(args, prep, fam, summary, dev)
    _dense_unit(args, prep, summary, dev)
    _prefix_unit(args, prep, summary, dev)
    _covariates_unit_m(args, prep, summary)
    _matching_unit(args, prep, summary)
    _write_json(summary_path, summary)
    _sentinel("fits", f"M3 done (ctx+dense+prefix arms, nulls, covariates, matching; {prep.fams})")
    EL._record_phase_time(args, "fits", time.time() - t0)


# ── M5-evidence (off-pod VM, 0 GPU, ZERO judge calls — plan v21 §4 M5 freeze) ─────


JUDGED_RESUME_SPEC = (
    "PRESERVED VERBATIM for the #1773-instrument resume (scope v4 item 5, judged spec "
    "AS AMENDED): extended rubric with three-way speaker_property (language / "
    "register_style / identity_disposition / none / unclear) plus level; tier-vs-level "
    "agreement; identity_disposition subset read. It runs unchanged on the persisted "
    "per-feature R2 arrays + these packets once the user green-lights the labelling "
    "resume. ZERO judge calls were dispatched this round (v17 §0.-1 freeze inheritance)."
)


def _ensure_eval_file(args, name: str) -> Path:
    """Off-pod read staging: fetch a pod-produced eval file from the HF mirror when
    absent locally (the #1482 P5 cross-machine-seam lesson — every M5/M6 read is in
    the M2/M4 upload set)."""
    from explore_persona_space.orchestrate import hub

    dest = args.out_eval / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    return hub.stage_hub_file(
        HF_DATA_REPO, f"{_m_hf_prefix(args)}/eval/{name}", dest, repo_type="dataset"
    )


def _ensure_store_local(args) -> None:
    """Stage the pooled store from HF when absent locally (VM production run; the
    off_pod_phases contract — the store is HF-permanent after M2)."""
    if list(args.store.glob("pooled_m_*.npz")):
        return
    from explore_persona_space.orchestrate import hub

    prefix = _m_hf_prefix(args)
    hub.stage_hub_prefix(HF_DATA_REPO, f"{prefix}/store", args.scratch / "store_dl")
    staged = args.scratch / "store_dl" / prefix / "store"
    args.store.mkdir(parents=True, exist_ok=True)
    for p in staged.glob("*.npz"):
        shutil.copy2(p, args.store / p.name)


def _tier_tercile_selection(z: dict, quota: int) -> list[dict]:
    """Mechanical tier x within-tier R2-tercile selection (plan §4 M5): per tier,
    rank by R2, split into 3 equal rank bins, take ``quota`` per cell EVENLY SPACED
    by rank (deterministic — endpoints included, no fresh seed); cells short of the
    quota take all."""
    fid = np.asarray(z["feat_ids"], np.int64)
    r2 = np.asarray(z["r2"], np.float64)
    tier = np.asarray(z["tier"], np.int64)
    act = np.asarray(z["activity"], np.float64)
    finite = np.isfinite(r2)
    rows: list[dict] = []
    for t in (0, 1, 2):
        idx = np.where((tier == t) & finite)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(r2[idx], kind="stable")]
        n = len(order)
        bounds = [0, n // 3, (2 * n) // 3, n]
        for c in range(3):
            cell = order[bounds[c] : bounds[c + 1]]
            if len(cell) == 0:
                continue
            k = min(quota, len(cell))
            take = cell[np.unique(np.linspace(0, len(cell) - 1, k).round().astype(int))]
            for rank_in_cell, i in enumerate(take.tolist()):
                rows.append(
                    {
                        "feat_id": int(fid[i]),
                        "tier": int(t),
                        "tercile": int(c),
                        "r2": float(r2[i]),
                        "activity": float(act[i]),
                        "rank_in_cell": rank_in_cell,
                        "cell_n": int(len(cell)),
                    }
                )
    seen: set[int] = set()
    dedup = []
    for r in rows:  # cells are disjoint by construction; guard anyway (plan: deduped)
        if r["feat_id"] in seen:
            continue
        seen.add(r["feat_id"])
        dedup.append(r)
    return dedup


def _np_probe(source: str) -> dict:
    """Neuronpedia source probe (aux evidence links only; plan A13 — absence recorded,
    never blocking; zero judge calls)."""
    import issue1482_feature_extremes as FE

    url = f"https://www.neuronpedia.org/{NP_MODEL_ID}/{source}"
    try:
        FE._http_get(url, timeout=30, attempts=2)
        return {"source": source, "url": url, "available": True}
    except Exception as e:  # noqa: BLE001 — probe is explicitly non-blocking (plan A13)
        logger.warning("[m5] neuronpedia probe failed for %s: %s", source, e)
        return {"source": source, "url": url, "available": False, "error": str(e)[:200]}


def phase_evidence(args) -> None:
    """M5-evidence (off-pod VM, 0 GPU, ZERO judge calls): tier x tercile selection +
    per-feature evidence packets -> eval_results/.../evidence/evidence.json."""
    t0 = time.time()
    args.work.mkdir(parents=True, exist_ok=True)
    _ensure_store_local(args)
    fams = _active_families(args)
    w_u, g, tok = EL._load_wu_norm(args)
    np_probes = {fam: _np_probe(NP_SOURCES[fam]) for fam in fams}
    families_doc: dict = {}
    for fam in fams:
        _ensure_eval_file(args, f"perfeature_m_{fam}_default.npz")
        _ensure_eval_file(args, f"covariates_m_{fam}.npz")
        z = np.load(args.out_eval / f"perfeature_m_{fam}_default.npz")
        quota = args.evidence_quota if args.evidence_quota > 0 else EVIDENCE_QUOTA[fam]
        sel = _tier_tercile_selection(dict(z), quota)
        assert sel, f"{fam}: empty evidence selection"
        union_sorted = sorted({r["feat_id"] for r in sel})
        panel_ids = np.asarray(z["feat_ids"], np.int64)
        shard_paths = sorted(args.store.glob(f"pooled_m_{fam}_*.npz"))
        assert shard_paths, f"no {fam} pooled shards under {args.store}"
        top, a, col_of = EL._evidence_scan(
            shard_paths, union_sorted, panel_ids, ("idx_off", "ans_idx", "ans_max")
        )
        ucols = col_of[np.asarray(union_sorted, np.int64)]
        assert (ucols >= 0).all(), "selected feature missing from the panel restriction"
        neighbors = EL._coact_topk(a, panel_ids, ucols)
        del a
        sae = S.SAELensJumpReLU.load(SAE_IDS[fam], device="cpu", cache_dir=args.sae_dir)
        dec_u = sae.w_dec[torch.as_tensor(union_sorted, dtype=torch.long)].T.contiguous()
        fp_tokens = EL._footprint_tokens(w_u, g, dec_u, tok)
        del sae, dec_u
        cz = np.load(args.out_eval / f"covariates_m_{fam}.npz")
        pos = {int(f): i for i, f in enumerate(np.asarray(cz["feat_ids"], np.int64))}
        u_pos = {int(f): j for j, f in enumerate(union_sorted)}
        features: dict[str, dict] = {}
        for row in sel:
            fid = row["feat_id"]
            i = pos[fid]
            j = u_pos[fid]
            cov = {
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
            for trait in ("evil", "sycophancy", "hallucination"):
                cov[f"rb_raw_{trait}"] = float(cz[f"rb_raw_{trait}"][i])
                cov[f"rb_cent_{trait}"] = float(cz[f"rb_cent_{trait}"][i])
            features[str(fid)] = {
                "selection": {k: row[k] for k in row},
                "top_answers": top.get(str(fid), []),
                "coact_neighbors": neighbors[j],
                "footprint_tokens": fp_tokens[j],
                "covariates": cov,
                "neuronpedia_link": f"https://www.neuronpedia.org/{NP_MODEL_ID}/"
                f"{NP_SOURCES[fam]}/{fid}",
            }
        families_doc[fam] = {
            "perfeature_source": f"perfeature_m_{fam}_default",
            "quota_per_cell": quota,
            "n_selected": len(features),
            "features": features,
        }
        print(f"[m5] {fam}: {len(features)} evidence packets", flush=True)
    doc = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "round": "matryoshka-tier",
        "layer": L_TIER,
        "freeze": "judged feature-labelling FROZEN this round (plan v21 §4 M5, "
        "inheriting v17 §0.-1) — labels are DEFERRED to #1773's validated instrument; "
        "these packets are the labelling input",
        "judged_resume_spec": JUDGED_RESUME_SPEC,
        "selection_recipe": "tier x within-tier R2 tercile; per cell, evenly-spaced-by-"
        "rank deterministic take (endpoints included; cells short of quota take all); "
        "quotas lmsys 40/cell, pile 20/cell (smoke: forced >=1/cell)",
        "top_answers_note": "[pooled ans_max activation, ci, row_idx] — ids into the "
        "parent raw_completions chunks (HF, parent pin); answer TEXT is not stored here",
        "neuronpedia": np_probes,
        "families": families_doc,
        "reproducibility": C.reproducibility_metadata(),
    }
    ev_dir = args.out_eval / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    _write_json(ev_dir / "evidence.json", doc)
    logger.info("[m5] evidence packet -> %s", ev_dir / "evidence.json")
    EL._record_phase_time(args, "evidence", time.time() - t0)


# ── M6: analysis + figures (off-pod VM) ──────────────────────────────────────────


def _strata_of(activity: np.ndarray, n_bins: int) -> np.ndarray:
    """Activity strata (deciles n_bins=10; quintiles n_bins=5 — the pre-registered
    merge fallback) via quantile bin edges (the EL covariates convention)."""
    edges = np.quantile(activity, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, activity, side="right")


def _tier_permutation(
    tier: np.ndarray, r2: np.ndarray, strata: np.ndarray, n_perm: int, rng, chunk: int = 2000
) -> dict:
    """H1 (registered, plan §3): pooled Spearman(tier, R2) with tier labels shuffled
    WITHIN activity strata — batched index ops, chunked over draws (never a per-draw
    Python loop). Spearman = Pearson on midranks (ties midranked both sides)."""
    rt = EA._midrank(np.asarray(r2, np.float64)[:, None])[:, 0]
    lt = EA._midrank(np.asarray(tier, np.float64)[:, None])[:, 0]
    b = rt - rt.mean()
    b_den = float(np.sqrt((b**2).sum()))
    a0 = lt - lt.mean()
    obs_den = float(np.sqrt((a0**2).sum())) * b_den
    obs = float((a0 * b).sum() / max(obs_den, 1e-12))
    stats_parts = []
    strata_ids = np.unique(strata)
    for s0 in range(0, n_perm, chunk):
        k = min(chunk, n_perm - s0)
        perm = np.tile(lt, (k, 1))
        for sid in strata_ids:
            m = np.where(strata == sid)[0]
            order = np.argsort(rng.random((k, len(m))), axis=1)
            perm[:, m] = lt[m][order]
        a = perm - perm.mean(axis=1, keepdims=True)
        den = np.sqrt((a**2).sum(axis=1)) * b_den
        with np.errstate(invalid="ignore", divide="ignore"):
            stats_parts.append((a @ b) / np.maximum(den, 1e-12))
    stats = np.concatenate(stats_parts)
    ok = np.isfinite(stats)
    lo, hi = np.percentile(stats[ok], [2.5, 97.5])
    # plan §3 lattice (DISJOINT + exhaustive): below band AND negative -> coarse-better;
    # above band AND positive -> fine-better; otherwise tier-null.
    if obs < lo and obs < 0:
        verdict = "coarse-better"
    elif obs > hi and obs > 0:
        verdict = "fine-better"
    else:
        verdict = "tier-null"
    return {
        "observed_pooled_spearman": obs,
        "perm_band_2p5_97p5": [float(lo), float(hi)],
        "n_perm": int(ok.sum()),
        "n_features": int(len(r2)),
        "n_strata": int(len(strata_ids)),
        "verdict": verdict,
        "note": "tier labels permuted WITHIN activity strata; observed outside the "
        "central 95% band = two-sided p < 0.05 on that side (plan §3 equivalence note)",
    }


def _boot_median_diff(a: np.ndarray, b: np.ndarray, n_boot: int, rng, chunk: int = 1000):
    """Batched bootstrap CI on median(a) - median(b) (independent resamples per side —
    the chat-vs-pile comparison is unpaired at feature level, plan §3)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    draws = []
    for s0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s0)
        ia = rng.integers(0, len(a), (k, len(a)))
        ib = rng.integers(0, len(b), (k, len(b)))
        draws.append(np.median(a[ia], axis=1) - np.median(b[ib], axis=1))
    d = np.concatenate(draws)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def _median_iqr(v: np.ndarray) -> dict:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "median": None}
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
    }


def _cov_battery(args, fam: str) -> dict:
    """Covariate Spearman battery + partials per family (the EL phase_analyze loop
    over this round's covariates_m_<fam>.npz vs the default per-feature arm)."""
    import issue1482_feature_correlates as FC

    cz = np.load(args.out_eval / f"covariates_m_{fam}.npz")
    pz = np.load(args.out_eval / f"perfeature_m_{fam}_default.npz")
    r2 = np.asarray(pz["r2"], np.float64)
    ok = np.isfinite(r2)
    r2v = r2[ok]
    act = np.asarray(cz["activity"], np.float64)[ok]
    cons = np.asarray(cz["consistency"], np.float64)[ok]
    d: dict = {"n": int(ok.sum()), "perfeature_source": f"perfeature_m_{fam}_default"}
    for nm in (
        "activity",
        "consistency",
        "footprint_conc",
        "coact",
        "rb_raw_maxabs",
        "rb_centered_maxabs",
    ):
        v = np.asarray(cz[nm], np.float64)[ok]
        m = np.isfinite(v)
        d[f"spearman_{nm}"] = FC._spearman(v[m], r2v[m]) if int(m.sum()) >= 3 else None
    m = np.isfinite(cons)
    if int(m.sum()) >= 4:
        d["partial_consistency_given_activity"] = FC._partial_spearman(cons[m], r2v[m], act[m])
    dense = np.asarray(cz["dense_flag"], np.int8)[ok]
    d["dense_flag_rate"] = float(dense.mean())
    d["dense_flag_median_r2"] = (
        float(np.nanmedian(r2v[dense == 1])) if int((dense == 1).sum()) else None
    )
    d["dense_tier_composition"] = {
        str(t): int(((np.asarray(cz["tier"], np.int64)[ok] == t) & (dense == 1)).sum())
        for t in (0, 1, 2)
    }
    return d


def _matching_reads(args, fams) -> dict:
    """Per-tier cross-dict matching distributions + threshold fractions + the
    mechanical persona-presence read (plan §4 M3/M6; judged confirmation frozen)."""
    if "pile" not in fams or not (args.out_eval / "matching.npz").exists():
        return {"note": "N/A — pile dropped or matching absent"}
    mz = np.load(args.out_eval / "matching.npz")
    best_l = np.asarray(mz["best_lmsys_to_pile"], np.float64)
    best_p = np.asarray(mz["best_pile_to_lmsys"], np.float64)
    tier_all = S.tier_of(np.arange(S.SAELENS_DICT_SIZE))
    out: dict = {"match_rows": int(mz["match_rows"])}
    for direction, vals in (("lmsys_to_pile", best_l), ("pile_to_lmsys", best_p)):
        per_tier = {}
        for t in (0, 1, 2):
            v = vals[tier_all == t]
            v = v[np.isfinite(v)]
            per_tier[str(t)] = {
                **_median_iqr(v),
                "frac_matched_at": {
                    str(thr): (float((v >= thr).mean()) if len(v) else None)
                    for thr in MATCH_THRESHOLDS
                },
            }
        out[direction] = per_tier
    # persona-presence read: among top-decile r_B-aligned lmsys PANEL features, the
    # distribution of best pile match vs the panel population (raw + centered).
    cz = np.load(args.out_eval / "covariates_m_lmsys.npz")
    fid = np.asarray(cz["feat_ids"], np.int64)
    pop = best_l[fid]
    persona = {}
    for key in ("rb_raw_maxabs", "rb_centered_maxabs"):
        al = np.asarray(cz[key], np.float64)
        thr = np.quantile(al[np.isfinite(al)], 0.9)
        top = pop[al >= thr]
        persona[key] = {
            "top_decile": _median_iqr(top),
            "population": _median_iqr(pop),
            "alignment_threshold": float(thr),
        }
    out["persona_presence"] = persona
    return out


def phase_analyze(args) -> None:
    """M6 (off-pod VM): H1 within-stratum tier permutation + H2 chat-vs-pile +
    per-tier x per-stratum profile + covariate battery + matching + corpus-transfer
    fold -> tier_tests.json; then the figure set (plan §6)."""
    t0 = time.time()
    rng = np.random.default_rng(BOOT_PERM_SEED)
    fams = _active_families(args)
    for fam in fams:
        _ensure_eval_file(args, f"perfeature_m_{fam}_default.npz")
        _ensure_eval_file(args, f"covariates_m_{fam}.npz")
        _ensure_eval_file(args, f"shuffle_null_m_{fam}.npz")
    for name in ("perfeature_m_lmsys_sinkmask.npz", "m_pilot.json", "m_summary.json"):
        _ensure_eval_file(args, name)
    if "pile" in fams:
        _ensure_eval_file(args, "matching.npz")
    import issue1482_feature_correlates as FC

    zl = np.load(args.out_eval / "perfeature_m_lmsys_default.npz")
    r2_l = np.asarray(zl["r2"], np.float64)
    tier_l = np.asarray(zl["tier"], np.int64)
    act_l = np.asarray(zl["activity"], np.float64)
    finite = np.isfinite(r2_l)
    r2f, tierf, actf = r2_l[finite], tier_l[finite], act_l[finite]

    # strata: deciles unless any (stratum x tier) cell < 20 features -> quintile merge
    # (the pre-registered fallback, plan §4 M6 / §7 per-tier panel floor)
    n_bins = 10
    strata = _strata_of(actf, n_bins)
    cell_counts = {
        f"{s}_{t}": int(((strata == s) & (tierf == t)).sum())
        for s in range(n_bins)
        for t in (0, 1, 2)
    }
    min_cell = min(cell_counts.values()) if cell_counts else 0
    merged = False
    if min_cell < 20:
        n_bins = 5
        strata = _strata_of(actf, n_bins)
        merged = True
        cell_counts = {
            f"{s}_{t}": int(((strata == s) & (tierf == t)).sum())
            for s in range(n_bins)
            for t in (0, 1, 2)
        }
    h1 = _tier_permutation(tierf, r2f, strata, args.n_perm, rng)
    h1["strata"] = "quintile" if merged else "decile"
    h1["min_cell_count_pre_merge"] = int(min_cell)
    h1["cell_counts"] = cell_counts
    logact = np.log10(np.maximum(actf, 1e-6))
    h1["partial_spearman_tier_r2_given_logact"] = (
        FC._partial_spearman(tierf.astype(np.float64), r2f, logact) if len(r2f) >= 4 else None
    )
    raw_descriptive = {
        "spearman_tier_r2_raw": FC._spearman(tierf.astype(np.float64), r2f)
        if len(r2f) >= 3
        else None,
        "note": "raw (unconditioned) gradient — DESCRIPTIVE ONLY, never the headline "
        "(tier confounds with firing frequency by construction; plan §3)",
        "per_tier": {str(t): _median_iqr(r2f[tierf == t]) for t in (0, 1, 2)},
    }

    # per (stratum x tier) medians (the hero profile's data)
    profile = {
        str(s): {str(t): _median_iqr(r2f[(strata == s) & (tierf == t)]) for t in (0, 1, 2)}
        for s in range(n_bins)
    }

    # shuffle-null bands + frac above, per family
    nulls: dict = {}
    for fam in fams:
        nz = np.load(args.out_eval / f"shuffle_null_m_{fam}.npz")
        hi = float(np.nanpercentile(np.asarray(nz["r2"], np.float64), 97.5))
        pf = np.load(args.out_eval / f"perfeature_m_{fam}_default.npz")
        rr = np.asarray(pf["r2"], np.float64)
        rr = rr[np.isfinite(rr)]
        nulls[fam] = {
            "p97_5": hi,
            "n_seeds": len(SHUFFLE_SEEDS),
            "frac_above": float((rr > hi).mean()) if len(rr) else None,
        }

    # H2 chat-vs-pile (registered): median per-feature R2 difference at matched
    # rows/estimator/panel recipe + bootstrap CI + per-tier resolution + FVE gap
    h2: dict | None = None
    if "pile" in fams:
        zp = np.load(args.out_eval / "perfeature_m_pile_default.npz")
        r2_p = np.asarray(zp["r2"], np.float64)
        tier_p = np.asarray(zp["tier"], np.int64)
        pilot = json.loads((args.out_eval / "m_pilot.json").read_text())
        h2 = {
            "median_r2_lmsys": _median_iqr(r2_l)["median"],
            "median_r2_pile": _median_iqr(r2_p)["median"],
            "bootstrap_ci_95_median_diff": _boot_median_diff(r2_l, r2_p, args.n_boot, rng),
            "per_tier": {
                str(t): {
                    "lmsys": _median_iqr(r2_l[tier_l == t]),
                    "pile": _median_iqr(r2_p[tier_p == t]),
                    "diff_ci_95": _boot_median_diff(
                        r2_l[tier_l == t], r2_p[tier_p == t], args.n_boot, rng
                    ),
                }
                for t in (0, 1, 2)
            },
            "fve": {
                "lmsys": pilot["fitness"]["lmsys"]["fve"],
                "pile": pilot["fitness"]["pile"]["fve"],
                "gap": round(pilot["fitness"]["lmsys"]["fve"] - pilot["fitness"]["pile"]["fve"], 4),
            },
            "note": "unpaired at feature level (different feature populations); the FVE "
            "gap is itself an H2 finding, not only a gate (plan §6)",
        }

    # sinkmask twin agreement (PAIRED — identical panel + score rows by construction)
    zs = np.load(args.out_eval / "perfeature_m_lmsys_sinkmask.npz")
    rs = np.asarray(zs["r2"], np.float64)
    mpair = np.isfinite(r2_l) & np.isfinite(rs)
    twin = {
        "sinkmask_paired_spearman": FC._spearman(r2_l[mpair], rs[mpair])
        if int(mpair.sum()) >= 3
        else None,
        "sinkmask_median_abs_delta": float(np.median(np.abs(r2_l[mpair] - rs[mpair])))
        if mpair.any()
        else None,
        "n_paired": int(mpair.sum()),
    }

    # corpus-transfer fold (fit unchanged; scored on LMSYS-only vs WildChat-only
    # halves of S_score — the r2_lmsys/r2_wildchat columns EL._per_feature_npz wrote)
    transfer: dict = {}
    for fam in fams:
        pf = np.load(args.out_eval / f"perfeature_m_{fam}_default.npz")
        tt = np.asarray(pf["tier"], np.int64)
        transfer[fam] = {
            str(t): {
                "r2_lmsys_rows": _median_iqr(np.asarray(pf["r2_lmsys"], np.float64)[tt == t]),
                "r2_wildchat_rows": _median_iqr(np.asarray(pf["r2_wildchat"], np.float64)[tt == t]),
            }
            for t in (0, 1, 2)
        }

    summary = json.loads((args.out_eval / "m_summary.json").read_text())
    doc = {
        "h1_tier_within_stratum": h1,
        "h1_raw_descriptive": raw_descriptive,
        "h2_chat_vs_pile": h2 if h2 is not None else "N/A — pile dropped (lmsys-only)",
        "per_stratum_per_tier_profile": profile,
        "shuffle_null": nulls,
        "twin_agreement": twin,
        "corpus_transfer": transfer,
        "covariates": {fam: _cov_battery(args, fam) for fam in fams},
        "matching": _matching_reads(args, fams),
        "baselines_identity": summary.get("baselines", {}),
        "knn_retrieval": summary.get("knn", {}),
        "pooled_r2": summary.get("pooled_r2", {}),
        "panel": summary.get("panel", {}),
        "seeds": {"perm_boot": BOOT_PERM_SEED, "n_perm": args.n_perm, "n_boot": args.n_boot},
    }
    _write_json(args.out_eval / "tier_tests.json", doc)
    logger.info("[m6] tier_tests.json written (h1=%s, strata=%s)", h1["verdict"], h1["strata"])
    _figures(args, fams, doc)
    EL._record_phase_time(args, "analyze", time.time() - t0)


def _figures(args, fams, doc: dict) -> None:
    """Figure set (plan §6): hero tier x predictability profile + low-level per-feature
    scatter + exploratory dump. One color = one meaning: tier -> paper_palette(3)
    across every panel; family/arm encoded by panel or hatch, never by tier colors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    figs = args.figures
    figs.mkdir(parents=True, exist_ok=True)
    tier_col = {t: paper_palette(3)[t] for t in (0, 1, 2)}
    tier_lab = {0: "tier 0 (general, ids <2k)", 1: "tier 1 (mid, 2k-16k)", 2: "tier 2 (specific)"}

    def _panel_arrays(fam: str):
        z = np.load(args.out_eval / f"perfeature_m_{fam}_default.npz")
        r2 = np.asarray(z["r2"], np.float64)
        ok = np.isfinite(r2)
        return (
            r2[ok],
            np.asarray(z["tier"], np.int64)[ok],
            np.asarray(z["activity"], np.float64)[ok],
        )

    def _iqr_err(vals: np.ndarray) -> tuple[float, float, float]:
        med = float(np.median(vals))
        lo = max(0.0, med - float(np.percentile(vals, 25)))
        hi = max(0.0, float(np.percentile(vals, 75)) - med)
        return med, lo, hi

    n_bins = 5 if doc["h1_tier_within_stratum"]["strata"] == "quintile" else 10

    # hero: per-stratum per-tier medians (+IQR whiskers) per family; null p97.5 line
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 3.8), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=True):
        r2, tier, act = _panel_arrays(fam)
        strata = _strata_of(act, n_bins)
        for t in (0, 1, 2):
            xs, meds, elo, ehi = [], [], [], []
            for s in range(n_bins):
                v = r2[(strata == s) & (tier == t)]
                if len(v) < 2:
                    continue
                m, lo, hi = _iqr_err(v)
                xs.append(s)
                meds.append(m)
                elo.append(lo)
                ehi.append(hi)
            if xs:
                ax.errorbar(
                    xs,
                    meds,
                    yerr=[elo, ehi],
                    color=tier_col[t],
                    marker="o",
                    ms=3,
                    capsize=2,
                    label=tier_lab[t],
                )
        null_hi = doc["shuffle_null"].get(fam, {}).get("p97_5")
        if null_hi is not None:
            ax.axhline(null_hi, color="0.5", ls=":", lw=1, label="shuffle-null p97.5")
        ax.set_xlabel(f"activity {'quintile' if n_bins == 5 else 'decile'}")
        ax.set_ylabel("per-feature held-out R² (median ± IQR)")
        ax.set_title(f"{fam} dictionary")
    axes[0].legend(fontsize=7)
    savefig_paper(fig, "fig_hero_tier_profile", dir=figs)
    plt.close(fig)

    # low-level companion: per-feature R² vs activity, colored by tier (lmsys)
    r2, tier, act = _panel_arrays("lmsys")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for t in (0, 1, 2):
        m = tier == t
        ax.scatter(
            act[m], np.clip(r2[m], -1, 1), s=3, alpha=0.25, color=tier_col[t], label=tier_lab[t]
        )
    null_hi = doc["shuffle_null"].get("lmsys", {}).get("p97_5")
    if null_hi is not None:
        ax.axhline(null_hi, color="0.5", ls=":", lw=1, label="shuffle-null p97.5")
    ax.set_xscale("log")
    ax.set_xlabel("answer-side activity (fraction of fit rows active)")
    ax.set_ylabel("per-feature held-out R² (clipped at −1)")
    ax.legend(fontsize=7, markerscale=3)
    savefig_paper(fig, "fig_r2_vs_activity_scatter", dir=figs)
    plt.close(fig)

    # raw vs within-stratum tier gradients (lmsys)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    strata = _strata_of(act, n_bins)
    demeaned = r2.copy()
    for s in range(n_bins):
        m = strata == s
        if m.any():
            demeaned[m] = r2[m] - np.median(r2[m])
    for ax, vals, ttl in (
        (axes[0], r2, "raw (descriptive only)"),
        (axes[1], demeaned, "within-stratum demeaned"),
    ):
        meds = [_iqr_err(vals[tier == t]) for t in (0, 1, 2)]
        ax.bar(
            [0, 1, 2],
            [m[0] for m in meds],
            yerr=[[m[1] for m in meds], [m[2] for m in meds]],
            color=[tier_col[t] for t in (0, 1, 2)],
            capsize=3,
        )
        ax.set_xticks([0, 1, 2], [tier_lab[t] for t in (0, 1, 2)], fontsize=6)
        ax.set_ylabel("median R²")
        ax.set_title(ttl)
    savefig_paper(fig, "fig_raw_vs_withinstratum_gradient", dir=figs)
    plt.close(fig)

    # covariate panels (lmsys): consistency + r_B alignment vs R², per tier
    cz = np.load(args.out_eval / "covariates_m_lmsys.npz")
    pz = np.load(args.out_eval / "perfeature_m_lmsys_default.npz")
    r2c = np.asarray(pz["r2"], np.float64)
    okc = np.isfinite(r2c)
    tc = np.asarray(pz["tier"], np.int64)[okc]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    for ax, key, xlab in (
        (axes[0], "consistency", "within-answer consistency"),
        (axes[1], "rb_centered_maxabs", "max |cos(decoder, r_B)| (centered)"),
    ):
        v = np.asarray(cz[key], np.float64)[okc]
        for t in (0, 1, 2):
            m = tc == t
            ax.scatter(v[m], np.clip(r2c[okc][m], -1, 1), s=3, alpha=0.25, color=tier_col[t])
        ax.set_xlabel(xlab)
        ax.set_ylabel("per-feature held-out R²")
    savefig_paper(fig, "fig_covariates_lmsys", dir=figs)
    plt.close(fig)

    # chat-vs-pile per-tier medians + FVE (only when pile is active)
    if "pile" in fams and isinstance(doc["h2_chat_vs_pile"], dict):
        h2 = doc["h2_chat_vs_pile"]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
        width = 0.35
        for fi, fam in enumerate(("lmsys", "pile")):
            meds = [h2["per_tier"][str(t)][fam]["median"] or 0.0 for t in (0, 1, 2)]
            axes[0].bar(
                np.arange(3) + (fi - 0.5) * width,
                meds,
                width,
                color=[tier_col[t] for t in (0, 1, 2)],
                hatch="" if fam == "lmsys" else "//",
                label=f"{fam} dictionary",
            )
        axes[0].set_xticks([0, 1, 2], [tier_lab[t] for t in (0, 1, 2)], fontsize=6)
        axes[0].set_ylabel("median per-feature R²")
        axes[0].legend(fontsize=7)
        axes[1].bar(
            [0, 1],
            [h2["fve"]["lmsys"], h2["fve"]["pile"]],
            color="0.4",
            hatch="",
        )
        axes[1].set_xticks([0, 1], ["lmsys dictionary", "pile dictionary"], fontsize=7)
        axes[1].set_ylabel("round-trip FVE on chat tokens")
        savefig_paper(fig, "fig_chat_vs_pile", dir=figs)
        plt.close(fig)

        # cross-dict matching ECDFs per tier
        mz = np.load(args.out_eval / "matching.npz")
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
        tier_all = S.tier_of(np.arange(S.SAELENS_DICT_SIZE))
        for ax, key, ttl in (
            (axes[0], "best_lmsys_to_pile", "lmsys → pile best match"),
            (axes[1], "best_pile_to_lmsys", "pile → lmsys best match"),
        ):
            vals = np.asarray(mz[key], np.float64)
            for t in (0, 1, 2):
                v = vals[tier_all == t]
                v = np.sort(v[np.isfinite(v)])
                if len(v):
                    ax.plot(v, np.linspace(0, 1, len(v)), color=tier_col[t], label=tier_lab[t])
            ax.set_xlabel("best decoder cosine")
            ax.set_ylabel("ECDF")
            ax.set_title(ttl, fontsize=8)
        axes[0].legend(fontsize=6)
        savefig_paper(fig, "fig_crossdict_matching", dir=figs)
        plt.close(fig)

    # sinkmask twin paired scatter (lmsys)
    zs = np.load(args.out_eval / "perfeature_m_lmsys_sinkmask.npz")
    rs = np.asarray(zs["r2"], np.float64)
    mp = np.isfinite(r2c) & np.isfinite(rs)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    tcol = S.tier_of(np.asarray(pz["feat_ids"], np.int64))
    for t in (0, 1, 2):
        m = mp & (tcol == t)
        ax.scatter(
            np.clip(r2c[m], -1, 1), np.clip(rs[m], -1, 1), s=3, alpha=0.25, color=tier_col[t]
        )
    lim = [-1, 1]
    ax.plot(lim, lim, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("R² (default answer pooling)")
    ax.set_ylabel("R² (sink-masked pooling)")
    savefig_paper(fig, "fig_sinkmask_paired", dir=figs)
    plt.close(fig)

    # corpus-transfer per-tier medians (lmsys arm)
    tr = doc["corpus_transfer"]["lmsys"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    width = 0.35
    for oi, (okey, hatch) in enumerate((("r2_lmsys_rows", ""), ("r2_wildchat_rows", "//"))):
        meds = [tr[str(t)][okey]["median"] or 0.0 for t in (0, 1, 2)]
        ax.bar(
            np.arange(3) + (oi - 0.5) * width,
            meds,
            width,
            color=[tier_col[t] for t in (0, 1, 2)],
            hatch=hatch,
            label="scored on LMSYS rows" if oi == 0 else "scored on WildChat rows",
        )
    ax.set_xticks([0, 1, 2], [tier_lab[t] for t in (0, 1, 2)], fontsize=6)
    ax.set_ylabel("median per-feature R²")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig_corpus_transfer", dir=figs)
    plt.close(fig)

    # tier composition: full population vs panel (per family)
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    xpos = 0
    labels = []
    for fam in fams:
        pop = np.load(args.out_eval / f"population_m_{fam}.npz")
        clearing = S.tier_of(
            np.where(pop["counts"] >= max(1, int(np.ceil(0.01 * int(pop["n_fit"])))))[0]
        )
        panel_t = np.asarray(np.load(args.out_eval / f"perfeature_m_{fam}_default.npz")["tier"])
        for src, tv in (("floor-clearing", clearing), ("panel", panel_t)):
            bottom = 0.0
            tot = max(1, len(tv))
            for t in (0, 1, 2):
                frac = float((tv == t).sum()) / tot
                ax.bar(
                    xpos,
                    frac,
                    bottom=bottom,
                    color=tier_col[t],
                    hatch="" if src == "panel" else "..",
                )
                bottom += frac
            labels.append(f"{fam}\n{src}")
            xpos += 1
    ax.set_xticks(range(len(labels)), labels, fontsize=6)
    ax.set_ylabel("tier share")
    savefig_paper(fig, "fig_tier_composition", dir=figs)
    plt.close(fig)
    logger.info("[m6] figures written under %s", figs)


# ── verify-imports + main ─────────────────────────────────────────────────────────


def _verify_imports() -> int:
    """Execute every DEFERRED import in THIS file (AST-walked, never a hand-maintained
    list) so a smoke-skipped branch cannot hide an ImportError (#606/#1332 class).
    Exit 0 on success."""
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


def main() -> int:
    """Linear phase dispatcher (smoke IS this driver with tiny args — PASS_UNIFIED)."""
    ap = argparse.ArgumentParser(description="Issue #1482 matryoshka-tier driver (M0-M6).")
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "pilot", "capture", "upload1", "fits", "upload2", "evidence", "analyze"],
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--full", action="store_true", help="explicit production mode (default)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-eval", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--sae-dir", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None, help="M5 evidence work dir (VM)")
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--max-chunks", type=int, default=None, help="0 = all (production)")
    ap.add_argument("--n-fit", type=int, default=None, help="requested S_fit size (reused rows)")
    ap.add_argument("--n-score", type=int, default=None)
    ap.add_argument("--val-carve", type=int, default=None, help="lambda-selection carve of S_fit")
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--pilot-n", type=int, default=None)
    ap.add_argument("--max-features-in", type=int, default=None)
    ap.add_argument("--max-features-out", type=int, default=None, help="tier panel cap")
    ap.add_argument("--match-rows", type=int, default=None, help="0 = all 65,536 lmsys rows")
    ap.add_argument(
        "--evidence-quota", type=int, default=None, help="0 = family default (40/20 per cell)"
    )
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
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
        "max_chunks": 2,
        "n_fit": 2000,
        "n_score": 600,
        "val_carve": 4,
        "gen_batch": 2,
        "pilot_n": 6,
        "max_features_in": 256,
        "max_features_out": 512,
        "match_rows": 1024,
        "evidence_quota": 1,
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
        "max_features_out": PANEL_CAP,
        "match_rows": 0,
        "evidence_quota": 0,
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
    base = default_smoke_root(args.base_root) if args.smoke else (args.base_root / "matryoshka")
    args.out_root = base
    if args.out_eval is None:
        args.out_eval = (
            (base / "eval_results")
            if args.smoke
            else (PROJECT_ROOT / "eval_results" / "issue_1482" / "matryoshka_tier")
        )
    if args.scratch is None:
        args.scratch = base / "scratch"
    if args.store is None:
        args.store = base / "store_m"
    if args.sae_dir is None:
        args.sae_dir = args.base_root / "hf_dl" / "sae_l20"
    if args.work is None:
        args.work = (
            (base / "work")
            if args.smoke
            else Path("/mnt/eps-data/thomasjiralerspong/issue1482_matryoshka")
        )
    if args.figures is None:
        # smoke figures NEVER touch the committed figures/ paths (kresample convention)
        args.figures = (
            (base / "figures")
            if args.smoke
            else (PROJECT_ROOT / "figures" / "issue_1482" / "matryoshka_tier")
        )
    for p in (args.out_eval, args.scratch, args.store):
        p.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "pilot": phase_pilot,
        "capture": phase_capture,
        "upload1": phase_upload1,
        "fits": phase_fits,
        "upload2": phase_upload2,
        "evidence": phase_evidence,
        "analyze": phase_analyze,
    }
    if args.phase == "all":
        for name in ("pilot", "capture", "upload1", "fits", "upload2"):
            C.phase(f"matryoshka-{name}")
            dispatch[name](args)
    else:
        C.phase(f"matryoshka-{args.phase}")
        dispatch[args.phase](args)
    return 0


if __name__ == "__main__":
    rc = main()
    # explicit exit after flushing: heavy C-extension teardown can rewrite the rc in
    # interpreter finalization (PyGILState atexit race, #1689 gotcha)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
