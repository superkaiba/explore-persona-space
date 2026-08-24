"""Issue #2552 — turn-averaged SAE feature predictability (Der et al. replication):
pod-side P1 phase-dispatch driver (unit 1 of the pre-split build).

Plan: tasks/<status>/2552/plans/plan.md (v4). Phases (plan §4 P0/P1; the REALIZED
order differs from the plan's P1.x numbering in ONE place — `eval_lists` runs
BEFORE the mining phases, because every description family's mining need-set is
`panel/union ∪ that dictionary's judged eval-list features` (plan §9 W1 rows),
and the eval-list unions only exist after the lists are computed):

  select_eval   P0: 2,000 corpus-stratified judged eval turns from the parent's
                20,000-row holdout (RNG seed 2552, text-resolvable candidates
                only — pass_b rows carry ci=-1 and have no rollout text; the
                exclusion count is disclosed in regime.json). Deterministic:
                re-running VERIFIES the committed regime.json by sha.
  assemble      P1.1: DELEGATED VERBATIM to the vendored parent kernel
                (vendored_2476.turnavg_sae.phase_assemble @ d8e9f8bdd4): stream
                the 1,920 banked capture chunks @ 89cfa76cdc, rebuild X19/Y19
                fp16 memmaps, re-assert the pinned splits by sha.
  sae_train     P1.2: the flat Der-recipe replication SAE. IMPLEMENTATION TRAP
                (plan §4 P1.2): constructs MatryoshkaBatchTopKSAE DIRECTLY with
                tier_bounds=(32768,) — NEVER through phase_sae_train's
                _sae_tier_bounds derivation, which returns a 3-tier nesting at
                width 32,768. Recipe: lr 2e-4, batch 256, 3 epochs, Adam(0.9,
                0.999), threshold EMA 0.999, init seed 2552, pool = the parent's
                _sae_row_positions construction (933,444 realized train rows;
                SAE-val carve 10,000 @ seed 2476 — parent parity). G1 halt floor
                holdout-val var-FVE >= 0.5 (rc=25); advisory nMSE band
                [0.07, 0.15].
  census_panel  P1.3: full-width firing census over the SAE-fit rows (counts,
                activation sum/sumsq, co-activation degree, per-corpus firing
                shares); alive floors ceil(0.01*n_fit)=1,200 and
                ceil(0.002*n_fit)=240; description/ladder panel = alive-at-240,
                activity-stratified (quintiles over log firing fraction), cap
                12,000, seed 2552.
  perfeature_r2 P1.4: banked dense map predictions (refit_holdout__ridge__seed0
                @ 89cfa76cdc) + identity+bias (+parity assert) + train-mean null
                + 20-draw row-shuffle nulls + corpus-transfer fold
                (_gram_ridge_single on the parent 23-value lambda grid; LMSYS
                fit rows -> WildChat holdout rows), all encoded at the panel
                columns; kNN retrieval (euclidean+cosine, k in {1,5,10}).
  eval_lists    P1.7: per-turn feature lists for the five configurations —
                rep_ta / mat_k100 / mat_k200 all-active turn-averaged lists +
                pt_max / pt_sum via teacher-forced layer-19 capture (persisted
                prompt+answer TOKEN IDS, the #1482 per-segment concat
                convention — never re-tokenized concatenated strings) encoded
                through andyrdt trainer_2 (k=128) with reference token-pool
                masking; SUM and MAX pooling over answer tokens; top-128 kept
                (pre-truncation), top-100 judged lists persisted beside them.
                Asserts 2,000/2,000 eval rows found in the rollout chunks
                (plan §12 A11 — the definitive join assert runs HERE, pod-side,
                where the chunks are actually read; P0 checks join-ABILITY via
                ci>=0). trainer_2 on-corpus reconstruction FVE accumulated.
  mat_encodes   P1.5: banked k=100 / k=200 matryoshka SAEs (HF sae_c/,
                sae_c_k200/ @ the regime_pins revision) encoded over the fit
                rows; counts reconciliation vs the banked firing censuses
                BEFORE mining; streaming top-25 activating-row heaps per
                description-need feature (union npz feat_ids ∪ judged eval-list
                features).
  rep_mining    P1.6: same streaming top-25 for the replication panel ∪ its
                judged eval-list union; then the SHARED text join for all three
                turn-averaged families (raw chunks @ 89cfa76cdc; example text
                capped at 1,500 chars — the W1 prompt cap).
  pt_mining     P1.8 (EVAL-DISJOINT — MF-A): per-token mining over the 18,000
                NON-eval holdout rows (holdout MINUS the 2,000 eval ids; hard
                assert set(eval_ids).isdisjoint(mining_ids) for EVERY family
                before any W1-input write). Two passes: A = heap (per-feature
                top-25 activating rows, row-max token) + full-width trainer_2
                ans_frac accumulation + FVE; B = window emission (±40-token
                contexts with per-token feature activations) for the unique
                selected rows. Per-family mining manifest (row ids + sha256)
                -> analysis_tensors/mining_manifest.json.
  covariates    P1.9: decoder norms; direct-logit footprint (decoder @
                unembedding, top-20 concentration, chunked); cross-dictionary
                best-match cosines vs trainer_2 (chunked fp16 GEMMs); top-256
                answer-PCA alignment (train-split covariance); twin-inherited
                within-answer consistency; activation variance / mean-when-
                active / corpus firing shares (from the census accumulators);
                max |cos(decoder, r_B)| over the #779 monitoring trait set.
  upload        P1.10: everything in plan §10's persist table ->
                issue2552_turnsae/{analysis_tensors,raw_completions}/ with
                exact-set verify; the plan §9 p1_done.json sentinel.
  smoke         The SAME phase functions composed on a tiny slice (2 capture
                chunks / 20 SAE steps / 2 eval turns) under out_root/smoke with
                --skip-upload forced. Asserts sae.tier_bounds == (32768,)
                (the plan §4 P1.2 trap guard). The judge-instrument 5-call
                probes and the 1-feature ladder pass belong to units 2/3
                (scripts/issue2552_judge_waves.py / issue2552_ladder.py).

NOT in this driver (unit-1 scope): P2 judge waves (unit 2), P3 ladder/stats
(unit 3), P4 embedding metric (`--phase p4-embed` is added by unit 2/3 against
the realized W1/W2 output schema).

Pod-side contract: sentinels via issue779_common.write_sentinel(task_id=2552)
(/workspace/logs/issue-2552-*.json) + the plan §9 out_root/sentinels/
p1_done.json; [phase=...] log lines; terminal [phase=done] on graceful exit.
LMSYS/WildChat text is handled DIGEST-ONLY in logs (never printed). Resume is
regime-keyed via the vendored parent's _enter_phase_regime (code-SHA mismatch
recomputes loudly; config/split mismatch raises).

Vendoring notes: the parent kernels are imported from scripts/vendored_2476/
(pin d8e9f8bdd4; see VENDORED_FROM.txt). The vendored module computes its
PROJECT_ROOT one level short (vendoring depth), so this driver re-points the
three path constants (PROJECT_ROOT / COMMITTED_SPLIT_1482 / COMMITTED_M_SPLIT)
and TASK_ID post-import — the vendored FILE stays byte-identical to the pin.
scripts/issue779_ffc_n1m_fits.py was ported wholesale from the same pin (the
k200 B1 revision-kwarg threading; additive, default-None — item (k) leg A).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "vendored_2476"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue1482_early_layer as EL  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import turnavg_sae as T  # noqa: E402  (vendored @ d8e9f8bdd4)

from explore_persona_space.atomic_io import savez_atomic  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2552")

TASK_ID = 2552

# ── vendoring-depth fixups (the vendored FILE is byte-identical to the pin;
#    only these module CONSTANTS are re-pointed — its own PROJECT_ROOT resolves
#    to scripts/ because vendoring added one directory level) ────────────────────
T.PROJECT_ROOT = PROJECT_ROOT
T.COMMITTED_SPLIT_1482 = PROJECT_ROOT / "eval_results" / "issue_1482" / "split_1482.json"
T.COMMITTED_M_SPLIT = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "matryoshka_tier" / "m_split.json"
)
T.TASK_ID = TASK_ID  # sentinels carry issue-2552 naming (pod-side contract)

# ── replication-SAE recipe (plan §11: arXiv 2606.28548 App. A + #2476 inherits) ──
REP_DICT = 32_768
REP_K = 128
REP_TIER_BOUNDS = (32_768,)  # FLAT — the plan §4 P1.2 implementation trap
REP_SEED = 2552  # SAE init (plan §10 Seeds); the SAE-val carve stays @ 2476 (parent parity)
RC_G1 = 25  # G1 SAE-val FVE-floor HALT (mirrors the parent's RC_G4 value)
G1_FVE_FLOOR = 0.5  # plan §7 G1 halt floor (inherited from #2476 gates_p4)
NMSE_ADVISORY_BAND = (0.07, 0.15)  # plan §4 P1.2 advisory band (paper 0.097; #2476 ~0.091)

# ── eval-turn / panel / mining constants (plan §4/§11) ───────────────────────────
EVAL_TURNS_N = 2_000
EVAL_SEED = 2552
PANEL_CAP = 12_000
PANEL_SEED = 2552
ALIVE_FRAC_PRIMARY = 0.01  # 1,200 rows at n_fit=120,000
ALIVE_FRAC_PANEL = 0.002  # 240 rows at n_fit=120,000
MINING_TOP = 25
WINDOW_TOKENS = 40  # ±40-token per-token mining context (plan §4 P1.8)
LIST_TOP = 128  # per-token configs' kept list length (pre-truncation)
JUDGED_TOP = 100  # equal-length judged lists (deviation d8)
EXAMPLE_TEXT_CAP = 1_500  # chars per mining example (the §9 W1 prompt cap)
SHUFFLE_SEEDS = tuple(EVAL_SEED * 100 + i for i in range(20))  # 20-draw row-shuffle null

# ── pins (plan §10/§11) ─────────────────────────────────────────────────────────
# Lineage inputs (capture chunks / rollout text / banked refit): the #2476 pin.
LINEAGE_REVISION = T.DATA_REPO_REVISION  # 89cfa76cdc…
REFIT_HOLDOUT_PATH = (
    "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz"
)
# Parent-tensor prefixes probed unpinned at plan time: pinned at the P0-recorded
# revision (eval_results/issue_2552/regime_pins.json — committed on this branch;
# plan §11 reused-input-data bullet). The r_B bank rides the same recorded pin
# (supersedes the plan-time 037fcbb probe via the same §11 record-at-fetch-time
# mechanism).
REGIME_PINS_PATH = PROJECT_ROOT / "eval_results" / "issue_2552" / "regime_pins.json"
SAE_C_PREFIX = "issue2476_turnavg/analysis_tensors/sae_c"
SAE_C_K200_PREFIX = "issue2476_turnavg/analysis_tensors/sae_c_k200"
CENSUS_C_PATH = "issue2476_turnavg/analysis_tensors/floor_sweep/firing_census_c.npz"
CENSUS_K200_PATH = "issue2476_turnavg/analysis_tensors/k200_census/firing_census_k200.npz"
RB_PREFIX = "issue779_monitoring/r_b"
RB_TRAITS = ("evil", "hallucination", "sycophancy")
LAYER = 19  # the map's native layer (parent LAYER_C)
UNION_C_NPZ = (
    PROJECT_ROOT / "eval_results" / "issue_2476" / "floor_sweep" / "perfeature_union_c.npz"
)
UNION_K200_NPZ = (
    PROJECT_ROOT / "eval_results" / "issue_2476" / "k200_census" / "perfeature_union_k200.npz"
)

TA_FAMILIES = ("rep_ta", "mat_k100", "mat_k200")
ALL_FAMILIES = (*TA_FAMILIES, "pt")


def _regime_pins() -> dict:
    assert REGIME_PINS_PATH.exists(), (
        f"regime_pins.json missing at {REGIME_PINS_PATH} — the P0 pin record must be "
        "committed on issue-2552 before any pod dispatch (plan §11)"
    )
    return json.loads(REGIME_PINS_PATH.read_text())


def _pins_revision() -> str:
    return str(_regime_pins()["data_repo_revision"])


# ── small shared helpers ─────────────────────────────────────────────────────────


def _eval_dir(args) -> Path:
    return args.out_root / "eval"


def _mining_dir(args) -> Path:
    return args.out_root / "mining"


def _lists_dir(args) -> Path:
    return args.out_root / "eval_lists"


def _sae_rep_dir(args) -> Path:
    return args.out_root / "sae_rep"


def _production(args) -> bool:
    return args.max_chunks == 0 and args.smoke_rows == 0 and not args.smoke


def _regime_json_path(args) -> Path:
    """The committed eval-turn regime (production) / the smoke-diverted copy.
    Smoke NEVER writes the canonical committed path (smoke-output divert rule)."""
    if _production(args):
        return PROJECT_ROOT / "eval_results" / "issue_2552" / "regime.json"
    return _eval_dir(args) / "regime.json"


def _hf_fetch(path_in_repo: str, dest_dir: Path, revision: str) -> Path:
    """Revision-pinned single-file fetch through the canonical transient-retry
    envelope (scoped per-file — never snapshot_download on the ~1M-file repo)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest_dir.mkdir(parents=True, exist_ok=True)
    got = hub.retry_transient(
        lambda: hf_hub_download(
            C.HF_DATA_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest_dir),
        ),
        what=f"pinned fetch ({path_in_repo}@{revision[:8]})",
    )
    return Path(got)


def _positions_of(args, ids: np.ndarray, what: str) -> np.ndarray:
    """Map assembled row ids -> memmap positions. Production: every id MUST be
    present (fail-loud). Smoke slices: absent ids are dropped with a WARN."""
    a_dir = T._assemble_dir(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    ids = np.asarray(ids, np.int64)
    pos = np.searchsorted(rows_present, ids)
    ok = (pos < len(rows_present)) & (rows_present[np.minimum(pos, len(rows_present) - 1)] == ids)
    if _production(args):
        assert bool(ok.all()), (
            f"[{what}] {int((~ok).sum())} of {len(ids)} rows absent from the assembled "
            "memmap — irreconcilable in production (plan §7)"
        )
    elif not bool(ok.all()):
        logger.warning(
            "[%s] smoke: %d/%d rows absent from slice; dropped", what, int((~ok).sum()), len(ids)
        )
    return pos[ok].astype(np.int64)


def _measured_update(args, **kv) -> None:
    """Append measured quantities (alive counts, unions, FVEs) to the pod-side
    regime_measured.json — the G2 gate + harvest step read these (plan §7 G2;
    the VM-side eval_results/issue_2552/regime.json merge happens at harvest)."""
    path = _eval_dir(args) / "regime_measured.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc.update({k: v for k, v in kv.items()})
    T._write_json(path, doc, phase="regime_measured")


def _sha_ids(ids: np.ndarray) -> str:
    return EL._sha_ids(np.asarray(ids, np.int64))


def _jsonl_write_sharded(base: Path, rows: list[dict], cap_bytes: int = 9_000_000) -> list[Path]:
    """Write JSONL rows to base (single file when < cap) or line-split
    base.stem.shardNNN.jsonl parts (< 9 MB each — the non-LFS upload rule).
    Unit 2's readers glob `top25_<fam>*.jsonl` (contract noted in the plan
    handoff)."""
    payloads = [json.dumps(r, ensure_ascii=False) for r in rows]
    total = sum(len(p) + 1 for p in payloads)
    base.parent.mkdir(parents=True, exist_ok=True)
    if total < cap_bytes:
        base.write_text("\n".join(payloads) + ("\n" if payloads else ""))
        return [base]
    parts: list[Path] = []
    buf: list[str] = []
    size = 0
    idx = 0

    def _flush():
        nonlocal buf, size, idx
        p = base.with_name(f"{base.stem}.shard{idx:03d}{base.suffix}")
        p.write_text("\n".join(buf) + "\n")
        parts.append(p)
        idx += 1
        buf, size = [], 0

    for p in payloads:
        if size + len(p) + 1 > cap_bytes and buf:
            _flush()
        buf.append(p)
        size += len(p) + 1
    if buf:
        _flush()
    return parts


def _upload_leaf(args, local_dir: Path, leaf: str, *, resume_skip: bool) -> None:
    """Production HF upload of one artifact leaf with exact-set verify (the
    parent _p4_upload pattern; skip-loud under --skip-upload / non-production)."""
    if args.skip_upload or not _production(args):
        logger.warning("[upload] skip_upload/non-production: %s upload SKIPPED (loud)", leaf)
        return
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    prefix = f"{args.hf_prefix}/{leaf}"
    res = upload_dir_sharded(
        local_dir,
        C.HF_DATA_REPO,
        prefix,
        repo_type="dataset",
        shard_glob="*",
        verify=True,
        delete_local=False,
        resume_skip=resume_skip,
    )
    if not res.rerouted:
        expected = [
            f"{prefix}/{p.relative_to(local_dir)}"
            for p in sorted(local_dir.rglob("*"))
            if p.is_file()
        ]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
        )
        assert not missing, f"[upload] verify FAILED — missing on Hub under {prefix}: {missing}"
    logger.info("[upload] %s -> %s (rerouted=%s)", local_dir.name, prefix, res.rerouted)


# ── raw-chunk iteration with a persistent cache + ci index (three passes reuse it) ──


def _chunk_index_path(args) -> Path:
    return T._stage_dir(args) / "chunk_ci_index.json"


def _raw_cache_dir(args) -> Path:
    d = T._stage_dir(args) / "raw_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iter_rows_pinned(args, needed_ci: dict[int, int], *, tag: str):
    """Yield (row_idx, ci, prompt, response) for every needed ci, downloading the
    rollout chunks AT THE LINEAGE PIN (89cfa76cdc — closes the parent's unpinned
    residual, plan §11) with a persistent per-run cache + a chunk->ci index so
    later passes fetch only the chunks they need. Text is never logged."""
    dns = SimpleNamespace(max_chunks=args.max_chunks, scratch=T._stage_dir(args))
    names = EA._raw_chunk_names(dns)
    idx_path = _chunk_index_path(args)
    index: dict[str, list[int]] = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    have_index = set(index) >= set(names)
    cache = _raw_cache_dir(args)
    n_found = 0
    t0 = time.time()
    for j, name in enumerate(names):
        if have_index and not (set(index.get(name, ())) & set(needed_ci)):
            continue
        local = cache / name
        if not local.exists():
            got = Path(
                N1M._download_chunk_with_retry(
                    C.HF_DATA_REPO,
                    f"{EA.RAW_PREFIX}/{name}",
                    cache,
                    revision=LINEAGE_REVISION,
                )
            )
            if got != local and got.exists():
                shutil.move(str(got), str(local))
        rows = json.loads(local.read_text())["rows"]
        if name not in index:
            index[name] = sorted(int(r["ci"]) for r in rows)
        for r in rows:
            ci = int(r["ci"])
            if ci in needed_ci:
                n_found += 1
                yield needed_ci[ci], ci, r["prompt"], r["response"]
        if (j + 1) % 100 == 0:
            print(
                f"[{tag}] chunk sweep {j + 1}/{len(names)} found={n_found} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    C.write_json_atomic(idx_path, index)


# ── P0: select_eval ──────────────────────────────────────────────────────────────


def phase_select_eval(args) -> None:
    """P0: corpus-stratified judged eval-turn selection (seed 2552) with sha
    verification against the committed regime.json (deterministic re-derivation
    IS the verification)."""
    C.phase("select_eval")
    row_ci, prov_u8, pools = T._load_scratch_meta(args)
    hold = np.asarray(pools["holdout"], np.int64)
    text_ok = row_ci[hold] >= 0
    n_excl = int((~text_ok).sum())
    cands = hold[text_ok]
    if not _production(args):
        a_dir = T._assemble_dir(args)
        if (a_dir / "rows_present.npy").exists():
            rows_present = np.load(a_dir / "rows_present.npy")
            cands = cands[np.isin(cands, rows_present)]
        n_eval = max(2, int(args.n_eval_turns) or 2)
    else:
        n_eval = int(args.n_eval_turns) or EVAL_TURNS_N
    lm = cands[prov_u8[cands] == 0]
    wc = cands[prov_u8[cands] == 1]
    if _production(args):
        assert len(lm) >= 1 and len(wc) >= 1, (len(lm), len(wc))
    else:
        assert len(cands) >= 2, f"smoke slice holds {len(cands)} holdout candidates (<2)"
        if len(lm) == 0 or len(wc) == 0:
            logger.warning("[select_eval] smoke: one corpus empty in slice; drawing from the other")
    n_eval = min(n_eval, len(cands))
    n_lm = int(round(n_eval * len(lm) / (len(lm) + len(wc))))
    n_lm = min(max(n_lm, n_eval - len(wc)), len(lm), n_eval)
    n_wc = n_eval - n_lm
    rng = np.random.default_rng(EVAL_SEED)
    pick_lm = rng.choice(np.sort(lm), size=n_lm, replace=False) if n_lm else np.empty(0, np.int64)
    pick_wc = rng.choice(np.sort(wc), size=n_wc, replace=False) if n_wc else np.empty(0, np.int64)
    eval_ids = np.sort(np.concatenate([pick_lm, pick_wc])).astype(np.int64)
    sha = _sha_ids(eval_ids)
    doc = {
        "eval_ids": [int(x) for x in eval_ids],
        "n_eval": int(len(eval_ids)),
        "n_lmsys": int(n_lm),
        "n_wildchat": int(n_wc),
        "seed": EVAL_SEED,
        "eval_ids_sha256": sha,
        "holdout_sha256": _sha_ids(hold),
        "n_holdout_candidates": int(len(cands)),
        "n_excluded_non_text_resolvable": n_excl,
        "selection": (
            "corpus-stratified (largest-share rounding) draw without replacement from the "
            "text-resolvable (ci>=0) holdout rows; sorted-id pools so the draw is "
            "order-deterministic"
        ),
    }
    path = _regime_json_path(args)
    if path.exists():
        prev = json.loads(path.read_text())
        assert prev.get("eval_ids_sha256") == sha and int(prev.get("n_eval", -1)) == len(
            eval_ids
        ), (
            f"[select_eval] committed regime.json DISAGREES with the deterministic "
            f"re-derivation (sha {prev.get('eval_ids_sha256')} != {sha}) — never overwrite"
        )
        logger.info("[select_eval] committed regime.json VERIFIED (sha match, n=%d)", len(eval_ids))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        T._write_json(path, doc, phase="select_eval")
        logger.info("[select_eval] wrote %s (n=%d, sha %s)", path, len(eval_ids), sha[:12])
    T._sentinel("select_eval", f"P0 eval turns selected (n={len(eval_ids)}, excl={n_excl})")


def _eval_ids(args) -> np.ndarray:
    path = _regime_json_path(args)
    assert path.exists(), f"regime.json missing at {path} — run --phase select_eval first"
    doc = json.loads(path.read_text())
    ids = np.asarray(doc["eval_ids"], np.int64)
    assert _sha_ids(ids) == doc["eval_ids_sha256"], "regime.json eval_ids sha drift"
    return ids


# ── P1.1: assemble (delegated verbatim) ─────────────────────────────────────────


def phase_assemble(args) -> None:
    T.phase_assemble(args)


# ── P1.2: flat replication SAE training ─────────────────────────────────────────


def _build_rep_sae(device: str) -> "T.MatryoshkaBatchTopKSAE":
    """The plan §10 call-shape bind, VERBATIM — flat 1-tier construction; NEVER
    routed through phase_sae_train's _sae_tier_bounds derivation (plan §4 trap)."""
    sae = T.MatryoshkaBatchTopKSAE(
        act_dim=int(C.EXPECTED_HIDDEN),
        dict_size=REP_DICT,
        k=REP_K,
        tier_bounds=REP_TIER_BOUNDS,
        seed=REP_SEED,
    ).to(device)
    assert sae.tier_bounds == REP_TIER_BOUNDS, (
        f"flat replication SAE must be 1-tier, got {sae.tier_bounds} (plan §4 P1.2 trap)"
    )
    return sae


def phase_sae_train(args) -> None:
    """P1.2: train the flat replication SAE (Der recipe; parent loop shape)."""
    C.phase("sae_train_rep")
    out = _sae_rep_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    w_path = out / "sae_weights.safetensors"
    log_path = out / "train_log.json"
    gates_path = out / "gates_g1.json"
    regime, resume_ok = T._enter_phase_regime(
        out,
        args,
        "sae_train_rep",
        stale_paths=[w_path, log_path, gates_path, out / "cfg.json", out / "ckpt_last.pt"],
    )
    if resume_ok and w_path.exists() and log_path.exists() and gates_path.exists():
        gates = json.loads(gates_path.read_text())
        if _production(args) and gates["g1"]["verdict"] == "FAIL":
            logger.error("[sae_train_rep] resume: recorded G1 FAIL re-applied")
            sys.exit(RC_G1)
        _upload_sae_rep(args, out, resume_skip=True)
        logger.info("[sae_train_rep] resume: weights+log+gates present; skip")
        return
    EA._headroom(args.out_root, 1 if args.smoke else 4, "sae-train-rep")
    a_dir = T._assemble_dir(args)
    assert (a_dir / "split_meta.json").exists(), "sae_train needs P1.1 outputs — run assemble"
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    tr_pos, val_pos, pool_doc = T._sae_row_positions(args)
    print(f"[sae_train_rep] pools re-measured: {json.dumps(pool_doc)}", flush=True)
    dev = args.device
    model = _build_rep_sae(dev)
    # b_dec init: seeded train-subsample mean (parent convention, rep seed)
    rng0 = np.random.default_rng(REP_SEED + 1)
    sub = np.sort(rng0.choice(tr_pos, size=min(65_536, len(tr_pos)), replace=False))
    mu = np.zeros(model.act_dim, dtype=np.float64)
    for s in range(0, len(sub), 8192):
        mu += np.asarray(y_mm[sub[s : s + 8192]], np.float64).sum(0)
    with torch.no_grad():
        model.b_dec.copy_(torch.as_tensor(mu / len(sub), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=T.SAE_LR, betas=T.SAE_ADAM_BETAS)
    ckpt_path = out / "ckpt_last.pt"
    start_epoch, step = 0, 0
    epoch_rows: list[dict] = []
    steps_cap = int(args.sae_steps)
    if resume_ok and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = int(ck["epoch_done"]), int(ck["step"])
        epoch_rows = list(ck["log_rows"])
        if bool(ck.get("steps_capped")) and steps_cap and step >= steps_cap:
            start_epoch = T.SAE_EPOCHS
        logger.info("[sae_train_rep] RESUMED at epoch %d (step %d)", start_epoch, step)
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, T.SAE_EPOCHS):
        rng_e = np.random.default_rng(REP_SEED * 1000 + epoch)
        run_loss, run_n = 0.0, 0
        diags: dict = {"l0_train": float("nan")}
        for xb in T._block_batches(y_mm, tr_pos, T.SAE_BATCH, rng_e):
            x = torch.as_tensor(np.asarray(xb, np.float32), device=dev)
            loss, diags, _fired = model.train_step_losses(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += diags["loss"]
            run_n += 1
            step += 1
            if step % 200 == 0:
                print(
                    f"[sae_train_rep] epoch {epoch + 1}/{T.SAE_EPOCHS} step {step} "
                    f"loss={run_loss / max(1, run_n):.1f} thr={float(model.threshold):.4f} "
                    f"l0={diags['l0_train']:.0f} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if steps_cap and step >= steps_cap:
                stop = True
                break
        fve_val, l0_val = T._recon_fve(model, y_mm, val_pos)
        row = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": round(run_loss / max(1, run_n), 3),
            "val_var_fve": round(fve_val, 6),
            "val_nmse": round(1.0 - fve_val, 6),
            "val_l0": round(l0_val, 2),
            "threshold": float(model.threshold),
            "elapsed_s": round(time.time() - t0, 1),
        }
        epoch_rows.append(row)
        print(
            f"[sae_train_rep] unit {epoch + 1}/{T.SAE_EPOCHS} epoch-done {json.dumps(row)}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch_done": epoch if stop else epoch + 1,
                "steps_capped": bool(stop),
                "step": step,
                "log_rows": epoch_rows,
            },
            ckpt_path,
        )
        if stop:
            break
    assert epoch_rows, "sae_train_rep produced no epoch rows"
    fve_val = float(epoch_rows[-1]["val_var_fve"])
    nmse = 1.0 - fve_val
    g1_pass = fve_val >= G1_FVE_FLOOR
    in_band = NMSE_ADVISORY_BAND[0] <= nmse <= NMSE_ADVISORY_BAND[1]
    gates = {
        "g1": {
            "val_var_fve": fve_val,
            "nmse": nmse,
            "nmse_advisory_band": list(NMSE_ADVISORY_BAND),
            "nmse_in_band": bool(in_band),
            "floor": G1_FVE_FLOOR,
            "n_val": int(len(val_pos)),
            "verdict": "PASS"
            if g1_pass
            else ("FAIL" if _production(args) else "INFORMATIONAL-smoke"),
        }
    }
    model.save_dir(out)
    T._write_json(
        log_path,
        {
            "pools": pool_doc,
            "epochs": epoch_rows,
            "steps": step,
            "steps_cap": steps_cap,
            "cfg": model.cfg_dict(),
        },
        phase="sae_train_rep",
    )
    T._write_json(gates_path, gates, phase="sae_train_rep")
    if not in_band:
        logger.warning(
            "[sae_train_rep] ADVISORY: nMSE %.4f outside band %s", nmse, NMSE_ADVISORY_BAND
        )
    if _production(args) and not g1_pass:
        T._sentinel("sae_train_rep", "G1 FVE below floor (gates_g1.json written)", {"rc": RC_G1})
        logger.error("[sae_train_rep] G1 FAIL: %s", gates["g1"])
        sys.exit(RC_G1)
    _upload_sae_rep(args, out, resume_skip=False)
    if ckpt_path.exists():
        ckpt_path.unlink()  # optimizer state = plan §10 discarded-artifacts row
    T._sentinel("sae_train_rep", f"P1.2 done (fve={fve_val:.4f} nmse={nmse:.4f})")


def _upload_sae_rep(args, out: Path, *, resume_skip: bool) -> None:
    """Stage ONLY the §10 sae_rep deliverables (weights/cfg/train_log/gates) —
    never the optimizer checkpoint (a declared discard) or the regime manifest."""
    if args.skip_upload or not _production(args):
        logger.warning("[sae_train_rep] skip_upload/non-production: sae_rep upload SKIPPED (loud)")
        return
    up = T._stage_dir(args) / "sae_rep_upload"
    if up.exists():
        shutil.rmtree(up)
    up.mkdir(parents=True, exist_ok=True)
    for name in ("sae_weights.safetensors", "cfg.json", "train_log.json", "gates_g1.json"):
        shutil.copy2(out / name, up / name)
    _upload_leaf(args, up, "analysis_tensors/sae_rep", resume_skip=resume_skip)


def _load_rep_sae(args) -> "T.MatryoshkaBatchTopKSAE":
    sae = T.MatryoshkaBatchTopKSAE.load_local(_sae_rep_dir(args), device=args.device)
    assert sae.tier_bounds == REP_TIER_BOUNDS, (
        f"loaded rep SAE is not flat 1-tier: {sae.tier_bounds}"
    )
    return sae


# ── census accumulators (shared: rep census + matryoshka encodes) ────────────────


@torch.no_grad()
def _census_pass(
    sae, mm, positions: np.ndarray, prov_rows: np.ndarray, device: str, chunk: int = 4096
) -> dict:
    """One streaming pass over the given rows: per-feature counts, activation
    sum/sumsq, co-activation degree, per-corpus counts (covariate inputs)."""
    W = sae.dict_size
    counts = torch.zeros(W, dtype=torch.int64, device=device)
    a_sum = torch.zeros(W, dtype=torch.float64, device=device)
    a_sq = torch.zeros(W, dtype=torch.float64, device=device)
    coact = torch.zeros(W, dtype=torch.float64, device=device)
    c_lm = torch.zeros(W, dtype=torch.int64, device=device)
    c_wc = torch.zeros(W, dtype=torch.int64, device=device)
    pos = np.asarray(positions, np.int64)
    prov_t = torch.as_tensor(np.asarray(prov_rows, np.int64), device=device)
    t0 = time.time()
    n_chunks = math.ceil(len(pos) / chunk)
    for i, s in enumerate(range(0, len(pos), chunk)):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=device)
        f = sae.encode(x, chunk=chunk)
        act = f > 0
        counts += act.sum(0)
        a_sum += f.sum(0, dtype=torch.float64)
        a_sq += (f * f).sum(0, dtype=torch.float64)
        row_l0 = act.sum(1).to(torch.float64)
        coact += act.to(torch.float64).t() @ row_l0
        pv = prov_t[s : s + x.shape[0]]
        c_lm += act[pv == 0].sum(0)
        c_wc += act[pv == 1].sum(0)
        if (i + 1) % 10 == 0 or i + 1 == n_chunks:
            print(f"[census] chunk {i + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s", flush=True)
    return {
        "counts": counts.cpu().numpy(),
        "act_sum": a_sum.cpu().numpy(),
        "act_sumsq": a_sq.cpu().numpy(),
        "coact_sum": coact.cpu().numpy(),
        "counts_lmsys": c_lm.cpu().numpy(),
        "counts_wildchat": c_wc.cpu().numpy(),
        "n_rows": int(len(pos)),
    }


def _activity_stratified_panel(
    counts: np.ndarray, n_fit: int, cap: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Plan §4 P1.3 panel: alive-at-ceil(0.002*n_fit) features, activity-
    stratified (quintiles over log firing fraction), proportional allocation
    (largest remainder), seeded draw, cap 12,000. Deterministic."""
    floor = max(1, math.ceil(ALIVE_FRAC_PANEL * n_fit))
    alive = np.where(counts >= floor)[0]
    doc: dict = {
        "floor": int(floor),
        "cap": int(cap),
        "seed": int(seed),
        "n_alive": int(len(alive)),
    }
    if len(alive) <= cap:
        doc["strata"] = "all-alive (cap not binding)"
        return np.sort(alive).astype(np.int64), doc
    logf = np.log(counts[alive].astype(np.float64) / n_fit)
    qs = np.quantile(logf, [0.2, 0.4, 0.6, 0.8])
    strat = np.digitize(logf, qs)
    rng = np.random.default_rng(seed)
    sizes = np.array([(strat == q).sum() for q in range(5)], np.int64)
    raw = cap * sizes / sizes.sum()
    alloc = np.floor(raw).astype(np.int64)
    rem = cap - int(alloc.sum())
    order = np.argsort(-(raw - alloc), kind="stable")
    alloc[order[:rem]] += 1
    parts = []
    for q in range(5):
        pool = alive[strat == q]
        take = min(int(alloc[q]), len(pool))
        parts.append(
            rng.choice(np.sort(pool), size=take, replace=False) if take < len(pool) else pool
        )
    panel = np.sort(np.concatenate(parts)).astype(np.int64)
    doc["strata_sizes"] = [int(x) for x in sizes]
    doc["strata_alloc"] = [int(x) for x in alloc]
    doc["n_panel"] = int(len(panel))
    return panel, doc


def phase_census_panel(args) -> None:
    """P1.3: rep-SAE firing census over the fit rows + the description/ladder panel."""
    C.phase("census_panel")
    out = _eval_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    census_path = out / "census_rep.npz"
    panel_path = out / "panel_rep.json"
    regime, resume_ok = T._enter_phase_regime(
        out, args, "census_panel", stale_paths=[census_path, panel_path]
    )
    if resume_ok and census_path.exists() and panel_path.exists():
        logger.info("[census_panel] resume: outputs present; skip")
        return
    sae = _load_rep_sae(args)
    a_dir = T._assemble_dir(args)
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    _row_ci, prov_u8, pools = T._load_scratch_meta(args)
    fit_pos = _positions_of(args, pools["sae_fit"], "census_panel")
    rows_present = np.load(a_dir / "rows_present.npy")
    acc = _census_pass(sae, y_mm, fit_pos, prov_u8[rows_present[fit_pos]], args.device)
    n_fit = acc["n_rows"]
    floor_primary = max(1, math.ceil(ALIVE_FRAC_PRIMARY * n_fit))
    floor_panel = max(1, math.ceil(ALIVE_FRAC_PANEL * n_fit))
    panel, pdoc = _activity_stratified_panel(acc["counts"], n_fit, PANEL_CAP, PANEL_SEED)
    savez_atomic(
        census_path,
        counts=acc["counts"],
        act_sum=acc["act_sum"],
        act_sumsq=acc["act_sumsq"],
        coact_sum=acc["coact_sum"],
        counts_lmsys=acc["counts_lmsys"],
        counts_wildchat=acc["counts_wildchat"],
        n_fit_rows=np.int64(n_fit),
        floor_primary=np.int64(floor_primary),
        floor_panel=np.int64(floor_panel),
        panel_ids=panel,
    )
    alive_240 = int((acc["counts"] >= floor_panel).sum())
    alive_1200 = int((acc["counts"] >= floor_primary).sum())
    T._write_json(
        panel_path,
        {"panel": pdoc, "n_alive_floor_panel": alive_240, "n_alive_floor_primary": alive_1200},
        phase="census_panel",
    )
    _measured_update(
        args,
        rep_alive_at_240=alive_240,
        rep_alive_at_1200=alive_1200,
        rep_panel_n=int(len(panel)),
        rep_n_fit=n_fit,
    )
    T._sentinel("census_panel", f"P1.3 done (alive240={alive_240} panel={len(panel)})")


# ── P1.4: per-feature R² for the replication SAE ─────────────────────────────────


def phase_perfeature_r2(args) -> None:
    """P1.4: banked-map / identity+bias / train-mean / shuffle-null / corpus-fold
    per-feature R² at the panel columns + kNN retrieval (plan §6 baselines pair)."""
    C.phase("perfeature_r2")
    out = _eval_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    pf_path = out / "perfeature_rep.npz"
    ret_path = out / "retrieval_rep.json"
    cf_path = out / "corpusfold_rep.json"
    regime, resume_ok = T._enter_phase_regime(
        out, args, "perfeature_r2", stale_paths=[pf_path, ret_path, cf_path]
    )
    if resume_ok and pf_path.exists() and ret_path.exists() and cf_path.exists():
        logger.info("[perfeature_r2] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 8, "perfeature-r2")
    sae = _load_rep_sae(args)
    a_dir = T._assemble_dir(args)
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    x_mm = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    _row_ci, prov_u8, pools = T._load_scratch_meta(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    cz = np.load(out / "census_rep.npz")
    panel = np.asarray(cz["panel_ids"], np.int64)
    counts = np.asarray(cz["counts"], np.int64)
    n_fit = int(cz["n_fit_rows"])
    floor_panel = int(cz["floor_panel"])
    floor_primary = int(cz["floor_primary"])

    # banked dense map predictions @ the lineage pin (plan §10 reuse row)
    hz_path = _hf_fetch(REFIT_HOLDOUT_PATH, T._stage_dir(args) / "refit", LINEAGE_REVISION)
    hz = np.load(hz_path)
    vhat_all = np.asarray(hz["holdout_pred16"], np.float16)
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    if _production(args):
        assert set(int(x) for x in hold_rows) == set(int(x) for x in pools["holdout"]), (
            "banked refit holdout_rows != pinned holdout pool"
        )
        keep = np.arange(len(hold_rows))
    else:
        present = np.isin(hold_rows, rows_present)
        keep = np.where(present)[0]
        assert len(keep) >= 2, f"smoke slice holds {len(keep)} banked-holdout rows (<2)"
    hold_ids = hold_rows[keep]
    hold_pos = _positions_of(args, hold_ids, "perfeature_r2")
    vhat = vhat_all[keep]
    te_prov = prov_u8[hold_ids]

    f_true = T._encode_restricted(sae, y_mm, hold_pos, panel)
    f_pred = T._encode_restricted(sae, vhat, np.arange(len(vhat)), panel)

    # identity+bias (parent _ib_arm_c convention + canonical-helper parity assert)
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    tr_ids = np.sort(np.setdiff1d(pools["train_full"], pools["holdout"], assume_unique=False))
    tr_pos = _positions_of(args, tr_ids, "perfeature_r2/ib")
    b = T._stream_bias(x_mm, y_mm, tr_pos)
    sub = tr_pos[: min(20_000, len(tr_pos))]
    b_helper = identity_bias_predict(
        np.asarray(x_mm[sub]), np.asarray(y_mm[sub]), np.zeros((1, x_mm.shape[1]), np.float64)
    )[0]
    b_sub = T._stream_bias(x_mm, y_mm, sub)
    assert np.allclose(b_helper, b_sub, atol=1e-8), "ib bias parity failed vs canonical helper"
    ib16 = (np.asarray(x_mm[hold_pos], np.float64) + b).astype(np.float16)
    f_ib = T._encode_restricted(sae, ib16, np.arange(len(ib16)), panel)

    # train-mean null: encode the dense train mean -> constant code row
    mu = np.zeros(x_mm.shape[1], np.float64)
    for s in range(0, len(tr_pos), 50_000):
        mu += np.asarray(y_mm[tr_pos[s : s + 50_000]], np.float64).sum(0)
    mu /= max(1, len(tr_pos))
    f_mu = sae.encode(torch.as_tensor(mu[None, :], dtype=torch.float32))[
        :, torch.as_tensor(panel, device=sae.device)
    ]
    f_trainmean = np.tile(f_mu.cpu().numpy().astype(np.float16), (len(hold_pos), 1))

    pf = EA._per_feature_metrics(f_pred, f_true)
    r2_ib = T._r2_only(f_ib, f_true)
    r2_trainmean = T._r2_only(f_trainmean, f_true)
    corpus = {}
    for label, code in (("lmsys", 0), ("wildchat", 1)):
        m = te_prov == code
        corpus[label] = (
            T._r2_only(f_pred[m], f_true[m]) if int(m.sum()) >= 2 else np.full(len(panel), np.nan)
        )
    null_map = T._shuffle_null_r2(f_pred, f_true, SHUFFLE_SEEDS, " map")
    null_ib = T._shuffle_null_r2(f_ib, f_true, SHUFFLE_SEEDS, " ib")

    # corpus-transfer fold: LMSYS-fit dense ridge (parent grid) scored on WildChat holdout
    fit_ids = np.asarray(pools["sae_fit"], np.int64)
    fit_lm_ids = fit_ids[prov_u8[fit_ids] == 0]
    wc_mask = te_prov == 1
    cf_doc: dict = {"n_lmsys_fit": int(len(fit_lm_ids)), "n_wildchat_holdout": int(wc_mask.sum())}
    r2_corpusfold = np.full(len(panel), np.nan)
    if int(wc_mask.sum()) >= 2 and len(fit_lm_ids) >= 2:
        tr_lm_pos = _positions_of(args, fit_lm_ids, "perfeature_r2/cf")
        _r1, val_ids, _te = T._assert_pinned_valtest(T._committed_split())
        va_pos = _positions_of(args, val_ids, "perfeature_r2/cf-val")
        te_pos = hold_pos[wc_mask]
        if len(va_pos) >= 2:
            pt, sel = T._gram_ridge_single(
                x_mm, y_mm, tr_lm_pos, va_pos, te_pos, N1M.LAMBDAS_N1M, args.device
            )
            f_pred_cf = T._encode_restricted(
                sae, np.asarray(pt, np.float16), np.arange(len(te_pos)), panel
            )
            r2_corpusfold = T._r2_only(f_pred_cf, f_true[wc_mask])
            cf_doc.update(sel)
        else:
            cf_doc["skipped"] = "fewer than 2 val rows in slice (smoke)"
    else:
        cf_doc["skipped"] = "fewer than 2 WildChat holdout rows / LMSYS fit rows (smoke)"

    savez_atomic(
        pf_path,
        feat_ids=panel,
        r2_map=pf["r2"],
        spearman=pf["spearman"],
        ss_tot=pf["ss_tot"],
        r2_ib=r2_ib,
        r2_trainmean=r2_trainmean,
        r2_lmsys=corpus["lmsys"],
        r2_wildchat=corpus["wildchat"],
        r2_corpusfold=r2_corpusfold,
        null_r2_map=null_map,
        null_r2_ib=null_ib,
        counts=counts[panel],
        alive_f240=(counts[panel] >= floor_panel),
        alive_f1200=(counts[panel] >= floor_primary),
        shuffle_seeds=np.asarray(SHUFFLE_SEEDS, np.int64),
        n_fit_rows=np.int64(n_fit),
        n_holdout=np.int64(len(hold_pos)),
        # NOTE: no `tier` column — the flat 32,768-wide rep SAE has no matryoshka
        # tiers (the ladder's tier covariate is matryoshka-arms-only, plan §4).
    )

    ret: dict = {}
    for pname, parr in (("map", f_pred), ("ib", f_ib)):
        pc = {}
        for metric in ("euclidean", "cosine"):
            r = T._knn_retrieval_chunked(
                np.asarray(parr, np.float32),
                np.asarray(f_true, np.float32),
                ks=(1, 5, 10),
                metric=metric,
                device=args.device,
            )
            r["wilson_ci_acc"] = {str(k): T._wilson(r["acc_at_k"][k], r["n"]) for k in (1, 5, 10)}
            pc[metric] = r
        ret[pname] = pc
    T._write_json(
        ret_path, {"retrieval": ret, "chance_stated": "k/n_pool per cell"}, phase="perfeature_r2"
    )
    T._write_json(cf_path, cf_doc, phase="perfeature_r2")
    _measured_update(args, rep_perfeature_n=int(len(panel)))
    T._sentinel(
        "perfeature_r2",
        f"P1.4 done (median r2_map={T._median_of(pf['r2']):.4f} ib={T._median_of(r2_ib):.4f})",
    )


# ── banked matryoshka dictionaries ───────────────────────────────────────────────


def _load_banked_matryoshka(args, fam: str) -> "T.MatryoshkaBatchTopKSAE":
    prefix = SAE_C_PREFIX if fam == "mat_k100" else SAE_C_K200_PREFIX
    rev = _pins_revision()
    stage = T._stage_dir(args) / f"banked_{fam}"
    for name in ("cfg.json", "sae_weights.safetensors"):
        _hf_fetch(f"{prefix}/{name}", stage, rev)
    d = stage / prefix
    sae = T.MatryoshkaBatchTopKSAE.load_local(d, device=args.device)
    exp_k = 100 if fam == "mat_k100" else 200
    assert sae.k == exp_k and sae.dict_size == 65_536, (sae.k, sae.dict_size, fam)
    return sae


def _union_feat_ids(fam: str) -> np.ndarray:
    path = UNION_C_NPZ if fam == "mat_k100" else UNION_K200_NPZ
    assert path.exists(), f"union npz missing at {path} (must be committed on issue-2552)"
    return np.asarray(np.load(path)["feat_ids"], np.int64)


def _eval_union(args, cfg: str) -> np.ndarray:
    """Judged (top-100) eval-list feature union for one configuration."""
    p = _lists_dir(args) / f"lists_{cfg}.json"
    assert p.exists(), f"eval lists missing for {cfg} — run --phase eval_lists first"
    doc = json.loads(p.read_text())
    ids: set[int] = set()
    for turn in doc["turns"]:
        ids.update(int(f) for f, _v in turn["judged_top100"])
    return np.asarray(sorted(ids), np.int64)


# ── streaming top-25 mining (turn-averaged families) ─────────────────────────────


@torch.no_grad()
def _mine_top_rows(
    sae,
    mm,
    positions: np.ndarray,
    row_ids: np.ndarray,
    cols: np.ndarray,
    device: str,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming per-feature top-25 activating ROWS over the given positions.
    Returns (vals (25, F) fp32, rows (25, F) int64 — row ids; -1 = unfilled)."""
    cols_t = torch.as_tensor(np.asarray(cols, np.int64), device=device)
    F = len(cols)
    top_v = torch.full((MINING_TOP, F), -1.0, dtype=torch.float32, device=device)
    top_r = torch.full((MINING_TOP, F), -1, dtype=torch.int64, device=device)
    pos = np.asarray(positions, np.int64)
    rid = torch.as_tensor(np.asarray(row_ids, np.int64), device=device)
    t0 = time.time()
    n_chunks = math.ceil(len(pos) / chunk)
    for i, s in enumerate(range(0, len(pos), chunk)):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=device)
        f = sae.encode(x, chunk=chunk)[:, cols_t]
        cat_v = torch.cat([top_v, f], dim=0)
        cat_r = torch.cat([top_r, rid[s : s + x.shape[0]].unsqueeze(1).expand(-1, F)], dim=0)
        v, idx = torch.topk(cat_v, MINING_TOP, dim=0)
        top_v = v
        top_r = torch.gather(cat_r, 0, idx)
        if (i + 1) % 20 == 0 or i + 1 == n_chunks:
            print(f"[mine] chunk {i + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s", flush=True)
    top_r[top_v <= 0] = -1  # never present a zero-activation "example"
    return top_v.cpu().numpy(), top_r.cpu().numpy()


def _mining_pool(args) -> tuple[np.ndarray, np.ndarray, dict]:
    """Turn-averaged mining pool: SAE-fit rows that are text-resolvable (ci>=0).
    Eval-disjoint by the pinned split construction (fit ∩ holdout = ∅) — still
    hard-asserted per family before any W1-input write (MF-A)."""
    row_ci, _prov, pools = T._load_scratch_meta(args)
    fit = np.asarray(pools["sae_fit"], np.int64)
    text_ok = row_ci[fit] >= 0
    pool_ids = fit[text_ok]
    doc = {
        "pool": "sae_fit ∩ text-resolvable(ci>=0)",
        "n_pool": int(len(pool_ids)),
        "n_excluded_passb": int((~text_ok).sum()),
    }
    pos = _positions_of(args, pool_ids, "mining_pool")
    a_dir = T._assemble_dir(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    return pos, rows_present[pos], doc


def _assert_eval_disjoint(args, fam: str, mining_ids: np.ndarray) -> str:
    eval_ids = set(int(x) for x in _eval_ids(args))
    assert eval_ids.isdisjoint(int(x) for x in mining_ids), (
        f"[{fam}] MF-A VIOLATION: mining pool intersects the judged eval-turn ids"
    )
    return _sha_ids(np.sort(np.asarray(mining_ids, np.int64)))


def _manifest_update(args, fam: str, pool_ids: np.ndarray, pool_doc: dict) -> None:
    path = _eval_dir(args) / "mining_manifest.json"
    doc = json.loads(path.read_text()) if path.exists() else {"families": {}}
    sha = _assert_eval_disjoint(args, fam, pool_ids)
    doc["families"][fam] = {
        **pool_doc,
        "n_mining_rows": int(len(pool_ids)),
        "mining_ids_sha256": sha,
        "eval_disjoint_assert": "PASS",
        "eval_ids_sha256": json.loads(_regime_json_path(args).read_text())["eval_ids_sha256"],
    }
    T._write_json(path, doc, phase="mining_manifest")


def phase_mat_encodes(args) -> None:
    """P1.5: banked matryoshka encodes — census reconciliation + top-25 heaps."""
    C.phase("mat_encodes")
    out = _mining_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [out / "top25_mat_k100.npz", out / "top25_mat_k200.npz"]
    regime, resume_ok = T._enter_phase_regime(out, args, "mat_encodes", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[mat_encodes] resume: heaps present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 6, "mat-encodes")
    a_dir = T._assemble_dir(args)
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    _row_ci, prov_u8, pools = T._load_scratch_meta(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    fit_pos_all = _positions_of(args, pools["sae_fit"], "mat_encodes")
    mine_pos, mine_ids, pool_doc = _mining_pool(args)
    rev = _pins_revision()
    for fam, census_path in (("mat_k100", CENSUS_C_PATH), ("mat_k200", CENSUS_K200_PATH)):
        sae = _load_banked_matryoshka(args, fam)
        acc = _census_pass(sae, y_mm, fit_pos_all, prov_u8[rows_present[fit_pos_all]], args.device)
        # counts reconciliation vs the banked census (plan §8 risk row) BEFORE mining
        bz = np.load(_hf_fetch(census_path, T._stage_dir(args) / f"census_{fam}", rev))
        banked = np.asarray(bz["counts"], np.int64)
        if _production(args):
            diff = np.abs(acc["counts"] - banked)
            frac_exact = float((diff == 0).mean())
            rel = diff / np.maximum(banked, 1)
            assert frac_exact >= 0.99 and float(np.quantile(rel, 0.99)) < 0.01, (
                f"[{fam}] census reconciliation FAILED: exact={frac_exact:.4f} "
                f"p99rel={float(np.quantile(rel, 0.99)):.4f}"
            )
            logger.info("[mat_encodes] %s census reconciled (exact %.4f)", fam, frac_exact)
        else:
            logger.warning("[mat_encodes] smoke: census reconciliation informational only")
        need = np.union1d(_union_feat_ids(fam), _eval_union(args, fam))
        vals, rows = _mine_top_rows(sae, y_mm, mine_pos, mine_ids, need, args.device)
        tmp = out / f".tmp_top25_{fam}.npz"
        np.savez(
            tmp,
            feat_ids=need,
            top_vals=vals,
            top_rows=rows,
            counts_recomputed=acc["counts"],
            act_sum=acc["act_sum"],
            act_sumsq=acc["act_sumsq"],
            coact_sum=acc["coact_sum"],
            counts_lmsys=acc["counts_lmsys"],
            counts_wildchat=acc["counts_wildchat"],
            n_fit_rows=np.int64(acc["n_rows"]),
        )
        tmp.replace(out / f"top25_{fam}.npz")
        _manifest_update(args, fam, mine_ids, pool_doc)
        _measured_update(args, **{f"{fam}_desc_union_n": int(len(need))})
        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()
    T._sentinel("mat_encodes", "P1.5 done (2 heaps + census reconciliation)")


def phase_rep_mining(args) -> None:
    """P1.6: replication-SAE top-25 mining + the SHARED text join for the three
    turn-averaged families (top25_<fam>.jsonl — the plan §9 P2 read paths)."""
    C.phase("rep_mining")
    out = _mining_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    heap_path = out / "top25_rep_ta.npz"
    done_marker = out / ".text_join_done.json"
    regime, resume_ok = T._enter_phase_regime(
        out,
        args,
        "rep_mining",
        stale_paths=[
            heap_path,
            done_marker,
            *out.glob("top25_*.jsonl"),
            *out.glob("top25_*.shard*.jsonl"),
        ],
    )
    if resume_ok and heap_path.exists() and done_marker.exists():
        logger.info("[rep_mining] resume: heap + text join present; skip")
        return
    sae = _load_rep_sae(args)
    a_dir = T._assemble_dir(args)
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    cz = np.load(_eval_dir(args) / "census_rep.npz")
    panel = np.asarray(cz["panel_ids"], np.int64)
    mine_pos, mine_ids, pool_doc = _mining_pool(args)
    need = np.union1d(panel, _eval_union(args, "rep_ta"))
    vals, rows = _mine_top_rows(sae, y_mm, mine_pos, mine_ids, need, args.device)
    savez_atomic(heap_path, feat_ids=need, top_vals=vals, top_rows=rows)
    _manifest_update(args, "rep_ta", mine_ids, pool_doc)
    _measured_update(args, rep_ta_desc_union_n=int(len(need)))
    del sae
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # shared text join (all three TA families) — raw chunks @ the lineage pin
    row_ci = np.load(T._stage_dir(args) / "row_ci.npy")
    needed_rows: set[int] = set()
    heaps: dict[str, dict] = {}
    for fam in TA_FAMILIES:
        hz = np.load(out / f"top25_{fam}.npz")
        heaps[fam] = {
            "feat_ids": np.asarray(hz["feat_ids"], np.int64),
            "vals": np.asarray(hz["top_vals"], np.float32),
            "rows": np.asarray(hz["top_rows"], np.int64),
        }
        needed_rows.update(int(r) for r in heaps[fam]["rows"].ravel() if r >= 0)
    needed_ci = {int(row_ci[r]): int(r) for r in sorted(needed_rows)}
    assert all(ci >= 0 for ci in needed_ci), "mining pool leaked a non-text-resolvable row"
    texts: dict[int, str] = {}
    for row_idx, _ci, _prompt, response in _iter_rows_pinned(args, needed_ci, tag="rep_mining"):
        texts[int(row_idx)] = response[:EXAMPLE_TEXT_CAP]
    missing = needed_rows - set(texts)
    assert not missing, f"text join missing {len(missing)} mining rows (e.g. {sorted(missing)[:5]})"
    written: dict[str, list[str]] = {}
    for fam in TA_FAMILIES:
        h = heaps[fam]
        recs = []
        for fi, feat in enumerate(h["feat_ids"]):
            for rank in range(MINING_TOP):
                r = int(h["rows"][rank, fi])
                if r < 0:
                    continue
                recs.append(
                    {
                        "family": fam,
                        "feat_id": int(feat),
                        "rank": rank,
                        "row_id": r,
                        "activation": float(h["vals"][rank, fi]),
                        "text": texts[r],
                    }
                )
        parts = _jsonl_write_sharded(out / f"top25_{fam}.jsonl", recs)
        written[fam] = [p.name for p in parts]
        print(f"[rep_mining] {fam}: {len(recs)} example rows -> {len(parts)} file(s)", flush=True)
    T._write_json(
        done_marker, {"written": written, "n_unique_rows": len(texts)}, phase="rep_mining"
    )
    T._sentinel("rep_mining", f"P1.6 done ({len(texts)} unique example rows joined)")


# ── P1.7: eval-turn feature lists ────────────────────────────────────────────────


def _write_lists(args, cfg: str, turns: list[dict], meta: dict) -> None:
    d = _lists_dir(args)
    d.mkdir(parents=True, exist_ok=True)
    T._write_json(
        d / f"lists_{cfg}.json", {"config": cfg, **meta, "turns": turns}, phase="eval_lists"
    )


@torch.no_grad()
def _ta_lists(args, sae, cfg: str, eval_ids: np.ndarray, eval_pos: np.ndarray, y_mm) -> None:
    turns = []
    for s in range(0, len(eval_pos), 256):
        x = torch.as_tensor(np.asarray(y_mm[eval_pos[s : s + 256]], np.float32), device=sae.device)
        f = sae.encode(x)
        for j in range(f.shape[0]):
            row = f[j]
            nz = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
            order = torch.argsort(row[nz], descending=True)
            ids = nz[order].cpu().numpy()
            vals = row[nz][order].cpu().numpy()
            full = [[int(i), float(v)] for i, v in zip(ids, vals, strict=True)]
            turns.append(
                {
                    "row_id": int(eval_ids[s + j]),
                    "pre_truncation": full,  # all-active (TA configs' native list)
                    "judged_top100": full[:JUDGED_TOP],
                }
            )
    _write_lists(
        args, cfg, turns, {"list_convention": "all-active turn-averaged codes, desc by activation"}
    )


def phase_eval_lists(args) -> None:
    """P1.7: the five configurations' per-turn lists + the A11 join assert +
    trainer_2 on-corpus FVE (eval pass share)."""
    C.phase("eval_lists")
    out = _lists_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [out / f"lists_{cfg}.json" for cfg in (*TA_FAMILIES, "pt_max", "pt_sum")]
    regime, resume_ok = T._enter_phase_regime(out, args, "eval_lists", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[eval_lists] resume: all five configs present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 8, "eval-lists")
    eval_ids = _eval_ids(args)
    eval_pos = _positions_of(args, eval_ids, "eval_lists")
    a_dir = T._assemble_dir(args)
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")

    _ta_lists(args, _load_rep_sae(args), "rep_ta", eval_ids, eval_pos, y_mm)
    for fam in ("mat_k100", "mat_k200"):
        sae = _load_banked_matryoshka(args, fam)
        _ta_lists(args, sae, fam, eval_ids, eval_pos, y_mm)
        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # per-token configs: teacher-forced capture + trainer_2 encode
    trainer2 = S.BatchTopKSAE.load(k=128, device=args.device, layer=LAYER)
    model, tok = EA._load_model_tok(args)
    prefix_chars = EA._prefix_char_len(tok)
    row_ci = np.load(T._stage_dir(args) / "row_ci.npy")
    needed_ci = {int(row_ci[r]): int(r) for r in eval_ids if int(row_ci[r]) >= 0}
    assert len(needed_ci) == len(eval_ids), "eval ids must be text-resolvable (P0 guarantees ci>=0)"
    rows_buf: list[tuple] = []
    for row_idx, ci, prompt, response in _iter_rows_pinned(args, needed_ci, tag="eval_lists"):
        tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
        if tk is None:
            continue
        rows_buf.append((row_idx, ci, *tk))
    # plan §12 A11: the definitive 2,000/2,000 join assert (production)
    if _production(args):
        assert len(rows_buf) == len(eval_ids), (
            f"[eval_lists] A11 join assert FAILED: {len(rows_buf)}/{len(eval_ids)} eval rows "
            "recovered from the rollout chunks"
        )
    rows_buf.sort(key=lambda r: len(r[2]))
    ss_tot = 0.0
    ss_res = 0.0
    pt_turns: dict[str, list[dict]] = {"pt_max": [], "pt_sum": []}
    t0 = time.time()
    for s0 in range(0, len(rows_buf), args.gen_batch):
        batch = rows_buf[s0 : s0 + args.gen_batch]
        caps = EA._batched_capture(model, tok, batch, (LAYER,), args.device)
        for (row_idx, _ci, full_ids, _pe, context_end, n_ans, _seam), cap in zip(
            batch, caps, strict=True
        ):
            h = cap[LAYER]
            keep = S.token_inlier_mask(h)
            keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
            ans_keep = keep[context_end + 1 :]
            h_ans = h[context_end + 1 :]
            kept = h_ans if int(ans_keep.sum()) == 0 else h_ans[ans_keep]
            f_tok = trainer2.encode(kept.to(args.device))
            x_dev = kept.to(device=args.device, dtype=torch.float32)
            r = x_dev - trainer2.decode(f_tok)
            mu_x = x_dev.mean(0, keepdim=True)
            ss_tot += float(((x_dev - mu_x) ** 2).sum())
            ss_res += float((r**2).sum())
            pooled = {"pt_sum": f_tok.sum(0), "pt_max": f_tok.max(0).values}
            for cfg, vec in pooled.items():
                nz = torch.nonzero(vec > 0, as_tuple=False).squeeze(-1)
                order = torch.argsort(vec[nz], descending=True)[:LIST_TOP]
                ids = nz[order].cpu().numpy()
                vals = vec[nz][order].cpu().numpy()
                full = [[int(i), float(v)] for i, v in zip(ids, vals, strict=True)]
                pt_turns[cfg].append(
                    {
                        "row_id": int(row_idx),
                        "n_ans_tokens": int(n_ans),
                        "pre_truncation": full,
                        "judged_top100": full[:JUDGED_TOP],
                    }
                )
        if (s0 // args.gen_batch) % 10 == 0:
            print(
                f"[eval_lists] pt capture {min(s0 + args.gen_batch, len(rows_buf))}/{len(rows_buf)} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    fve = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    meta = {
        "pooling_mask": "reference token-pool (S.token_inlier_mask + BOS strip; unmasked fallback)",
        "trainer2_fve_evalpass": fve,
        "list_convention": f"top-{LIST_TOP} pooled trainer_2 codes (pre-truncation), top-{JUDGED_TOP} judged",
    }
    for cfg in ("pt_max", "pt_sum"):
        _write_lists(args, cfg, pt_turns[cfg], meta)

    # measured unions (the G2 gate inputs — plan §7)
    unions = {}
    for cfg in (*TA_FAMILIES, "pt_max", "pt_sum"):
        doc = json.loads((out / f"lists_{cfg}.json").read_text())
        ids: set[int] = set()
        for turn in doc["turns"]:
            ids.update(int(f) for f, _v in turn["judged_top100"])
        unions[cfg] = len(ids)
    pt_union = set()
    for cfg in ("pt_max", "pt_sum"):
        doc = json.loads((out / f"lists_{cfg}.json").read_text())
        for turn in doc["turns"]:
            pt_union.update(int(f) for f, _v in turn["judged_top100"])
    _measured_update(
        args,
        eval_list_unions=unions,
        pt_desc_union_n=len(pt_union),
        trainer2_fve_evalpass=fve,
        n_eval_rows_captured=len(rows_buf),
    )
    T._sentinel("eval_lists", f"P1.7 done (5 configs, pt-union={len(pt_union)}, fve={fve:.4f})")


def _pt_union(args) -> np.ndarray:
    ids: set[int] = set()
    for cfg in ("pt_max", "pt_sum"):
        doc = json.loads((_lists_dir(args) / f"lists_{cfg}.json").read_text())
        for turn in doc["turns"]:
            ids.update(int(f) for f, _v in turn["judged_top100"])
    return np.asarray(sorted(ids), np.int64)


# ── P1.8: per-token mining (EVAL-DISJOINT — MF-A) ────────────────────────────────


def phase_pt_mining(args) -> None:
    """P1.8: per-token top-25 token-context mining over the NON-eval holdout
    rows (two passes: heap+accumulators, then window emission)."""
    C.phase("pt_mining")
    out = _mining_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    heap_path = out / "pt_heap.npz"
    done_marker = out / ".pt_windows_done.json"
    regime, resume_ok = T._enter_phase_regime(
        out, args, "pt_mining", stale_paths=[heap_path, done_marker, *out.glob("top25_pt*.jsonl")]
    )
    if resume_ok and heap_path.exists() and done_marker.exists():
        logger.info("[pt_mining] resume: heap + windows present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 8, "pt-mining")
    eval_ids = _eval_ids(args)
    row_ci, _prov, pools = T._load_scratch_meta(args)
    hold = np.asarray(pools["holdout"], np.int64)
    pool_ids = np.setdiff1d(hold, eval_ids, assume_unique=False)
    text_ok = row_ci[pool_ids] >= 0
    n_excl = int((~text_ok).sum())
    pool_ids = pool_ids[text_ok]
    if not _production(args):
        a_dir = T._assemble_dir(args)
        if (a_dir / "rows_present.npy").exists():
            rows_present = np.load(a_dir / "rows_present.npy")
            pool_ids = pool_ids[np.isin(pool_ids, rows_present)]
        pool_ids = pool_ids[: max(4, args.smoke_rows or 4)]
    pool_doc = {
        "pool": "holdout MINUS judged eval ids, text-resolvable (ci>=0)",
        "n_excluded_passb": n_excl,
    }
    sha = _assert_eval_disjoint(args, "pt", pool_ids)  # MF-A hard assert BEFORE any W1-input write
    logger.info("[pt_mining] pool n=%d (sha %s) — eval-disjoint PASS", len(pool_ids), sha[:12])

    need = _pt_union(args)
    trainer2 = S.BatchTopKSAE.load(k=128, device=args.device, layer=LAYER)
    model, tok = EA._load_model_tok(args)
    prefix_chars = EA._prefix_char_len(tok)
    need_t = torch.as_tensor(need, device=args.device)
    F = len(need)
    top_v = torch.full((MINING_TOP, F), -1.0, dtype=torch.float32, device=args.device)
    top_r = torch.full((MINING_TOP, F), -1, dtype=torch.int64, device=args.device)
    top_p = torch.full(
        (MINING_TOP, F), -1, dtype=torch.int64, device=args.device
    )  # peak ans-token pos
    W_FULL = trainer2.dict_size
    frac_sum = torch.zeros(W_FULL, dtype=torch.float64, device=args.device)
    frac_rows = torch.zeros(W_FULL, dtype=torch.int64, device=args.device)
    ss_tot = 0.0
    ss_res = 0.0
    needed_ci = {int(row_ci[r]): int(r) for r in pool_ids}
    rows_iter: list[tuple] = []
    for row_idx, ci, prompt, response in _iter_rows_pinned(args, needed_ci, tag="pt_mining"):
        tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
        if tk is None:
            continue
        rows_iter.append((row_idx, ci, *tk))
    if _production(args):
        assert len(rows_iter) == len(pool_ids), (
            f"[pt_mining] {len(rows_iter)}/{len(pool_ids)} pool rows recovered from chunks"
        )
    rows_iter.sort(key=lambda r: len(r[2]))
    t0 = time.time()
    n_done = 0
    for s0 in range(0, len(rows_iter), args.gen_batch):
        batch = rows_iter[s0 : s0 + args.gen_batch]
        caps = EA._batched_capture(model, tok, batch, (LAYER,), args.device)
        for (row_idx, _ci, full_ids, _pe, context_end, n_ans, _seam), cap in zip(
            batch, caps, strict=True
        ):
            h = cap[LAYER]
            keep = S.token_inlier_mask(h)
            keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
            ans_keep = keep[context_end + 1 :]
            h_ans = h[context_end + 1 :]
            fallback = int(ans_keep.sum()) == 0
            kept = h_ans if fallback else h_ans[ans_keep]
            kept_pos = (
                np.arange(h_ans.shape[0]) if fallback else np.where(ans_keep.cpu().numpy())[0]
            )
            f_tok = trainer2.encode(kept.to(args.device))
            x_dev = kept.to(device=args.device, dtype=torch.float32)
            r = x_dev - trainer2.decode(f_tok)
            mu_x = x_dev.mean(0, keepdim=True)
            ss_tot += float(((x_dev - mu_x) ** 2).sum())
            ss_res += float((r**2).sum())
            act_full = f_tok > 0
            frac_row = act_full.to(torch.float64).mean(0)
            frac_sum += frac_row
            frac_rows += (frac_row > 0).to(torch.int64)
            f_need = f_tok[:, need_t]
            v_row, p_idx = f_need.max(0)
            pos_ans = torch.as_tensor(kept_pos, device=args.device, dtype=torch.int64)[p_idx]
            cat_v = torch.cat([top_v, v_row.unsqueeze(0)], dim=0)
            cat_r = torch.cat(
                [top_r, torch.full((1, F), int(row_idx), dtype=torch.int64, device=args.device)],
                dim=0,
            )
            cat_p = torch.cat([top_p, pos_ans.unsqueeze(0)], dim=0)
            v, idx = torch.topk(cat_v, MINING_TOP, dim=0)
            top_v = v
            top_r = torch.gather(cat_r, 0, idx)
            top_p = torch.gather(cat_p, 0, idx)
            n_done += 1
        if (s0 // args.gen_batch) % 20 == 0:
            print(
                f"[pt_mining] pass A {n_done}/{len(rows_iter)} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    top_r[top_v <= 0] = -1
    fve = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    savez_atomic(
        heap_path,
        feat_ids=need,
        top_vals=top_v.cpu().numpy(),
        top_rows=top_r.cpu().numpy(),
        top_ans_pos=top_p.cpu().numpy(),
        ans_frac_sum=frac_sum.cpu().numpy(),
        ans_frac_rows=frac_rows.cpu().numpy(),
        trainer2_fve_pool=np.float64(fve),
        n_pool_rows=np.int64(len(rows_iter)),
    )
    _manifest_update(args, "pt", pool_ids, pool_doc)
    _measured_update(args, trainer2_fve_poolpass=fve, pt_mining_pool_n=int(len(rows_iter)))

    # pass B: window emission for the unique selected rows
    sel_rows = sorted({int(r) for r in top_r.cpu().numpy().ravel() if r >= 0})
    by_row: dict[int, list[tuple[int, int, int, float]]] = {}
    tv, tr_, tp = top_v.cpu().numpy(), top_r.cpu().numpy(), top_p.cpu().numpy()
    for fi in range(F):
        for rank in range(MINING_TOP):
            r = int(tr_[rank, fi])
            if r >= 0:
                by_row.setdefault(r, []).append((fi, rank, int(tp[rank, fi]), float(tv[rank, fi])))
    needed_ci_b = {int(row_ci[r]): int(r) for r in sel_rows}
    recs: list[dict] = []
    rows_iter_b: list[tuple] = []
    for row_idx, ci, prompt, response in _iter_rows_pinned(args, needed_ci_b, tag="pt_windows"):
        tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
        if tk is not None:
            rows_iter_b.append((row_idx, ci, *tk))
    rows_iter_b.sort(key=lambda r: len(r[2]))
    for s0 in range(0, len(rows_iter_b), args.gen_batch):
        batch = rows_iter_b[s0 : s0 + args.gen_batch]
        caps = EA._batched_capture(model, tok, batch, (LAYER,), args.device)
        for (row_idx, _ci, full_ids, _pe, context_end, n_ans, _seam), cap in zip(
            batch, caps, strict=True
        ):
            h = cap[LAYER]
            keep = S.token_inlier_mask(h)
            keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
            ans_keep = keep[context_end + 1 :]
            h_ans = h[context_end + 1 :]
            fallback = int(ans_keep.sum()) == 0
            kept = h_ans if fallback else h_ans[ans_keep]
            kept_pos = (
                np.arange(h_ans.shape[0]) if fallback else np.where(ans_keep.cpu().numpy())[0]
            )
            f_tok = trainer2.encode(kept.to(args.device))
            for fi, rank, peak_ans_pos, val in by_row.get(int(row_idx), ()):
                peak_abs = context_end + 1 + peak_ans_pos
                lo = max(0, peak_abs - WINDOW_TOKENS)
                hi = min(len(full_ids), peak_abs + WINDOW_TOKENS + 1)
                text = tok.decode(full_ids[lo:hi])
                kp = np.asarray(kept_pos)
                in_win = (kp + context_end + 1 >= lo) & (kp + context_end + 1 < hi)
                acts = f_tok[torch.as_tensor(np.where(in_win)[0], device=args.device), need_t[fi]]
                tok_acts = [
                    [int(kp[j] + context_end + 1 - lo), float(a)]
                    for j, a in zip(np.where(in_win)[0], acts.cpu().numpy(), strict=True)
                    if a > 0
                ]
                recs.append(
                    {
                        "family": "pt",
                        "feat_id": int(need[fi]),
                        "rank": rank,
                        "row_id": int(row_idx),
                        "peak_token_abs": int(peak_abs),
                        "activation": val,
                        "window_text": text[: EXAMPLE_TEXT_CAP * 2],
                        "window_token_acts": tok_acts,
                        "window_lo_abs": int(lo),
                    }
                )
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    recs.sort(key=lambda r: (r["feat_id"], r["rank"]))
    parts = _jsonl_write_sharded(out / "top25_pt.jsonl", recs)
    T._write_json(
        done_marker,
        {
            "written": [p.name for p in parts],
            "n_windows": len(recs),
            "n_unique_rows": len(sel_rows),
        },
        phase="pt_mining",
    )
    T._sentinel(
        "pt_mining", f"P1.8 done ({len(recs)} windows over {len(sel_rows)} rows; fve={fve:.4f})"
    )


# ── P1.9: covariate GEMMs ────────────────────────────────────────────────────────


@torch.no_grad()
def _decoder_covariates(
    w_dec: torch.Tensor,
    unembed: torch.Tensor,
    trainer2_dec_n: torch.Tensor,
    pca_basis: torch.Tensor,
    rb: torch.Tensor,
    device: str,
    chunk: int = 4096,
) -> dict:
    """Chunked decoder-side covariates for one dictionary: norms, direct-logit
    top-20 concentration, best-match cosine vs trainer_2, top-256 answer-PCA
    alignment, max |cos| vs the r_B trait set."""
    F = w_dec.shape[0]
    norms = torch.empty(F, dtype=torch.float32)
    foot = torch.empty(F, dtype=torch.float32)
    match_cos = torch.empty(F, dtype=torch.float32)
    match_id = torch.empty(F, dtype=torch.int64)
    pca_frac = torch.empty(F, dtype=torch.float32)
    pca_rank = torch.empty(F, dtype=torch.int64)
    rb_cos = torch.empty(F, len(RB_TRAITS), dtype=torch.float32)
    un_t = unembed.to(device=device, dtype=torch.float16)
    t2 = trainer2_dec_n.to(device=device, dtype=torch.float16)
    basis = pca_basis.to(device=device, dtype=torch.float32)
    rb_n = torch.nn.functional.normalize(rb.to(device=device, dtype=torch.float32), dim=1)
    for s in range(0, F, chunk):
        d = w_dec[s : s + chunk].to(device=device, dtype=torch.float32)
        n = d.norm(dim=1)
        norms[s : s + len(d)] = n.cpu()
        dn = d / n.clamp_min(1e-12).unsqueeze(1)
        logits = (dn.to(torch.float16) @ un_t.t()).float().abs()
        topv = torch.topk(logits, 20, dim=1).values.sum(1)
        foot[s : s + len(d)] = (topv / logits.sum(1).clamp_min(1e-12)).cpu()
        sims = (dn.to(torch.float16) @ t2).float()
        mc, mi = sims.max(dim=1)
        match_cos[s : s + len(d)] = mc.cpu()
        match_id[s : s + len(d)] = mi.cpu()
        proj = dn @ basis.t()  # (chunk, 256)
        pca_frac[s : s + len(d)] = proj.norm(dim=1).cpu()
        pca_rank[s : s + len(d)] = proj.abs().argmax(dim=1).cpu()
        rb_cos[s : s + len(d)] = (dn @ rb_n.t()).abs().cpu()
    return {
        "decoder_norm": norms.numpy(),
        "footprint_top20": foot.numpy(),
        "match_cos": match_cos.numpy(),
        "match_id": match_id.numpy(),
        "pca_align_frac": pca_frac.numpy(),
        "pca_best_rank": pca_rank.numpy(),
        "rb_cos": rb_cos.numpy(),
    }


def phase_covariates(args) -> None:
    """P1.9: the ladder covariate inputs, per turn-averaged dictionary."""
    C.phase("covariates")
    out = _eval_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    finals = [out / f"covariates_{fam}.npz" for fam in TA_FAMILIES]
    regime, resume_ok = T._enter_phase_regime(out, args, "covariates", stale_paths=finals)
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[covariates] resume: outputs present; skip")
        return
    EA._headroom(args.out_root, 2 if args.smoke else 6, "covariates")
    a_dir = T._assemble_dir(args)
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    _row_ci, _prov, pools = T._load_scratch_meta(args)

    # answer-PCA basis from the SAE-train rows (train-split covariance, fp64 streamed)
    tr_pos, _val_pos, _doc = T._sae_row_positions(args)
    d_model = int(C.EXPECTED_HIDDEN)
    xx = np.zeros((d_model, d_model), np.float64)
    mu = np.zeros(d_model, np.float64)
    for s in range(0, len(tr_pos), 8192):
        blk = np.asarray(y_mm[tr_pos[s : s + 8192]], np.float64)
        xx += blk.T @ blk
        mu += blk.sum(0)
    n_tr = len(tr_pos)
    mu /= max(1, n_tr)
    cov = xx / max(1, n_tr) - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(cov)
    k_basis = min(256, d_model)
    basis = torch.as_tensor(
        np.ascontiguousarray(evecs[:, ::-1][:, :k_basis].T), dtype=torch.float32
    )

    # unembedding (lm_head) — loaded once via the shared model loader (honors the
    # --tiny-model smoke carve-out), sliced to fp16 for the footprint GEMM
    lm, _tok = EA._load_model_tok(args)
    unembed = lm.lm_head.weight.detach().float().cpu().to(torch.float16).clone()
    del lm
    if args.device == "cuda":
        torch.cuda.empty_cache()

    trainer2 = S.BatchTopKSAE.load(k=128, device="cpu", layer=LAYER)
    t2_dec = trainer2.w_dec  # (act_dim, dict_size)
    t2_dec_n = t2_dec / t2_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)

    rev = _pins_revision()
    rb_rows = []
    for trait in RB_TRAITS:
        p = _hf_fetch(f"{RB_PREFIX}/{trait}.pt", T._stage_dir(args) / "rb", rev)
        obj = torch.load(p, map_location="cpu", weights_only=True)
        r_b = obj["r_b"]
        assert r_b.shape[1] == d_model and r_b.shape[0] > LAYER, tuple(r_b.shape)
        rb_rows.append(r_b[LAYER].to(torch.float32))
    rb = torch.stack(rb_rows)

    # pt ans_frac (twin-inherited consistency source)
    heap = np.load(_mining_dir(args) / "pt_heap.npz")
    frac_sum = np.asarray(heap["ans_frac_sum"], np.float64)
    frac_rows = np.asarray(heap["ans_frac_rows"], np.int64)
    ans_frac_full = np.where(frac_rows > 0, frac_sum / np.maximum(frac_rows, 1), np.nan)

    for fam in TA_FAMILIES:
        if fam == "rep_ta":
            sae = _load_rep_sae(args)
            cz = np.load(out / "census_rep.npz")
            acc = {
                k: np.asarray(cz[k])
                for k in (
                    "counts",
                    "act_sum",
                    "act_sumsq",
                    "coact_sum",
                    "counts_lmsys",
                    "counts_wildchat",
                )
            }
            n_fit = int(cz["n_fit_rows"])
        else:
            sae = _load_banked_matryoshka(args, fam)
            hz = np.load(_mining_dir(args) / f"top25_{fam}.npz")
            acc = {
                "counts": np.asarray(hz["counts_recomputed"]),
                "act_sum": np.asarray(hz["act_sum"]),
                "act_sumsq": np.asarray(hz["act_sumsq"]),
                "coact_sum": np.asarray(hz["coact_sum"]),
                "counts_lmsys": np.asarray(hz["counts_lmsys"]),
                "counts_wildchat": np.asarray(hz["counts_wildchat"]),
            }
            n_fit = int(hz["n_fit_rows"])
        dcov = _decoder_covariates(
            sae.w_dec.detach(),
            unembed,
            t2_dec_n,
            basis,
            rb,
            args.device,
        )
        counts = acc["counts"].astype(np.float64)
        mean_active = np.where(counts > 0, acc["act_sum"] / np.maximum(counts, 1), np.nan)
        var_act = np.maximum(
            acc["act_sumsq"] / max(1, n_fit) - (acc["act_sum"] / max(1, n_fit)) ** 2, 0.0
        )
        share_lm = np.where(counts > 0, acc["counts_lmsys"] / np.maximum(counts, 1), np.nan)
        coact_mean = np.where(counts > 0, acc["coact_sum"] / np.maximum(counts, 1), np.nan)
        consistency_twin = ans_frac_full[dcov["match_id"]]
        tmp = out / f".tmp_covariates_{fam}.npz"
        np.savez(
            tmp,
            counts=acc["counts"],
            n_fit_rows=np.int64(n_fit),
            act_var=var_act,
            act_mean_when_active=mean_active,
            share_lmsys=share_lm,
            coact_mean=coact_mean,
            decoder_norm=dcov["decoder_norm"],
            footprint_top20=dcov["footprint_top20"],
            match_cos=dcov["match_cos"],
            match_id=dcov["match_id"],
            consistency_twin=consistency_twin,
            pca_align_frac=dcov["pca_align_frac"],
            pca_best_rank=dcov["pca_best_rank"],
            rb_cos=dcov["rb_cos"],
            rb_traits=np.asarray(RB_TRAITS),
            rb_cos_max=dcov["rb_cos"].max(axis=1),
        )
        tmp.replace(out / f"covariates_{fam}.npz")
        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"[covariates] {fam} done", flush=True)
    T._sentinel("covariates", "P1.9 done (3 covariate bundles)")


# ── P1.10: upload ────────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    """P1.10: persist everything in plan §10's table BEFORE termination —
    analysis_tensors/{eval,eval_lists,mining_manifest} + raw_completions/mining."""
    C.phase("upload")
    if args.skip_upload or not _production(args):
        logger.warning("[upload] skip_upload/non-production: P1.10 uploads SKIPPED (loud)")
        T._sentinel("upload", "P1.10 SKIPPED (non-production/skip_upload)")
        return
    stage = T._stage_dir(args) / "upload_stage"
    if stage.exists():
        shutil.rmtree(stage)
    # analysis_tensors/eval: censuses, per-feature npz, covariates, measured regime
    ev = stage / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    for p in sorted(_eval_dir(args).glob("*")):
        if p.is_file() and p.name != "regime.json":
            shutil.copy2(p, ev / p.name)
    _upload_leaf(args, ev, "analysis_tensors/eval", resume_skip=True)
    # analysis_tensors/eval_lists: the P2 read path (plan §9 off_pod reads)
    el = stage / "eval_lists"
    el.mkdir(parents=True, exist_ok=True)
    lists_doc = {}
    for cfg in (*TA_FAMILIES, "pt_max", "pt_sum"):
        lists_doc[cfg] = json.loads((_lists_dir(args) / f"lists_{cfg}.json").read_text())
    (el / "feature_lists_2000turns.json").write_text(json.dumps(lists_doc))
    _upload_leaf(args, el, "analysis_tensors/eval_lists", resume_skip=True)
    # mining_manifest.json at its declared standalone path
    mm = stage / "manifest"
    mm.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_eval_dir(args) / "mining_manifest.json", mm / "mining_manifest.json")
    _upload_leaf(args, mm, "analysis_tensors", resume_skip=True)
    # raw_completions/mining: the top-25 example texts (unconditional text uploads)
    mn = stage / "mining"
    mn.mkdir(parents=True, exist_ok=True)
    for p in sorted(_mining_dir(args).glob("top25_*")):
        shutil.copy2(p, mn / p.name)
    _upload_leaf(args, mn, "raw_completions/mining", resume_skip=True)
    # the plan §9 phase sentinel
    sent_dir = args.out_root / "sentinels"
    sent_dir.mkdir(parents=True, exist_ok=True)
    T._write_json(sent_dir / "p1_done.json", {"phase": "p1", "status": "done"}, phase="upload")
    T._sentinel("upload", "P1.10 done (all §10 destinations verified)")


# ── smoke ────────────────────────────────────────────────────────────────────────


def phase_smoke(args) -> None:
    """Composed tiny-slice run of the SAME phase functions (plan §4 smoke):
    2 capture chunks / 20 SAE steps / 2 eval turns / 4-row pt pool, under
    out_root/smoke with uploads skipped. The judge 5-call probes + 1-feature
    ladder legs live in units 2/3 (their drivers)."""
    s = argparse.Namespace(**vars(args))
    s.smoke = True
    s.out_root = args.out_root / "smoke"
    s.max_chunks = args.max_chunks or 2
    s.smoke_rows = args.smoke_rows or 24
    s.sae_steps = args.sae_steps or 20
    s.skip_upload = True
    s.n_eval_turns = 2
    s.gen_batch = min(args.gen_batch, 4)
    s.out_root.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    # smoke ordering: select_eval AFTER assemble — the smoke eval turns must be
    # drawn from holdout ∩ the assembled slice (production keeps select_eval
    # first: it is the P0 VM phase and pure-verifies the committed regime.json)
    smoke_order = (
        "assemble",
        "select_eval",
        *[n for n in PHASE_ORDER if n not in ("assemble", "select_eval")],
    )
    for name in smoke_order:
        t0 = time.time()
        PHASES[name](s)
        timing[name] = round(time.time() - t0, 1)
        print(f"[smoke] leg {name} done in {timing[name]}s", flush=True)
    # per-leg output verification (parent smoke convention)
    checks = {
        "assemble": T._assemble_dir(s) / "split_meta.json",
        "sae_train": _sae_rep_dir(s) / "sae_weights.safetensors",
        "census_panel": _eval_dir(s) / "census_rep.npz",
        "perfeature_r2": _eval_dir(s) / "perfeature_rep.npz",
        "eval_lists": _lists_dir(s) / "lists_pt_max.json",
        "mat_encodes": _mining_dir(s) / "top25_mat_k100.npz",
        "rep_mining": _mining_dir(s) / "top25_rep_ta.npz",
        "pt_mining": _mining_dir(s) / "pt_heap.npz",
        "covariates": _eval_dir(s) / "covariates_rep_ta.npz",
        "manifest": _eval_dir(s) / "mining_manifest.json",
    }
    missing = {k: str(p) for k, p in checks.items() if not p.exists()}
    assert not missing, f"[smoke] legs missing outputs: {missing}"
    sae = _load_rep_sae(s)
    assert sae.tier_bounds == REP_TIER_BOUNDS, "smoke tier-bounds assert (plan §4 P1.2 trap)"
    T._write_json(s.out_root / "smoke_timing.json", {"legs_s": timing}, phase="smoke")
    T._sentinel("smoke", f"composed smoke PASS ({json.dumps(timing)})")
    logger.info("[smoke] PASS: %s", timing)


# ── CLI ──────────────────────────────────────────────────────────────────────────

PHASE_ORDER = (
    "select_eval",
    "assemble",
    "sae_train",
    "census_panel",
    "perfeature_r2",
    "eval_lists",
    "mat_encodes",
    "rep_mining",
    "pt_mining",
    "covariates",
    "upload",
)

PHASES = {
    "smoke": phase_smoke,
    "select_eval": phase_select_eval,
    "assemble": phase_assemble,
    "sae_train": phase_sae_train,
    "census_panel": phase_census_panel,
    "perfeature_r2": phase_perfeature_r2,
    "eval_lists": phase_eval_lists,
    "mat_encodes": phase_mat_encodes,
    "rep_mining": phase_rep_mining,
    "pt_mining": phase_pt_mining,
    "covariates": phase_covariates,
    "upload": phase_upload,
}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Issue #2552 P1 driver (Der replication build)")
    ap.add_argument("--phase", default="all", choices=["all", *PHASES])
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps-issue-2552"))
    ap.add_argument("--hf-prefix", default="issue2552_turnsae")
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument("--smoke-rows", type=int, default=0, help="0 = production")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all 1,920 chunks (production)")
    ap.add_argument("--sae-steps", type=int, default=0, help="0 = full 3-epoch train")
    ap.add_argument("--n-eval-turns", type=int, default=0, help="0 = 2,000 (production)")
    ap.add_argument("--gen-batch", type=int, default=16, help="teacher-forced forward batch")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--skip-upload", action="store_true", help="local-only run (loud)")
    ap.add_argument(
        "--fresh-stream", action="store_true", help="assemble: ignore the stream cursor"
    )
    ap.add_argument("--gpu-id", type=int, default=-1, help="informational; CVD pins the device")
    ap.add_argument("--production", action="store_true", help="assert-guard: refuse smoke knobs")
    ap.add_argument(
        "--resume-across-code-sha",
        action="store_true",
        help="crash-fix relaunch escape (parent _enter_phase_regime contract)",
    )
    # parent-kernel passthrough (delegated phase_assemble + _regime hash inputs;
    # production values pinned — the vendored kernels read these off args)
    ap.add_argument("--tiny-model", action="store_true", help="smoke-only from-config model")
    ap.add_argument("--sae-dict", type=int, default=0, help="parent passthrough (unused here)")
    ap.add_argument("--sae-k", type=int, default=0, help="parent passthrough (unused here)")
    ap.add_argument("--fit-n", type=int, default=0, help="parent passthrough (regime hash)")
    ap.add_argument("--n-perm", type=int, default=10_000, help="parent passthrough (regime hash)")
    ap.add_argument("--n-boot", type=int, default=10_000, help="parent passthrough (regime hash)")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + deferred-import resolution",
    )
    ap.add_argument("--list-phases", action="store_true", help="print the phase registry and exit")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    if args.list_phases:
        print(json.dumps({"order": list(PHASE_ORDER), "registry": sorted(PHASES)}))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred-import resolution (smoke-architecture Axis 1): execute every
        # function-body import of this driver so a missing symbol fails HERE.
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
        from safetensors.torch import load_file, save_file  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.analysis.extraction import (
            extract_layer_activations,  # noqa: F401
        )
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.preflight import (
            assert_out_root_headroom,  # noqa: F401
        )
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (
            upload_dir_sharded,  # noqa: F401
        )

        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if args.production:
        assert _production(args), (
            "--production forbids smoke knobs (--smoke/--smoke-rows/--max-chunks)"
        )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[main] phase=%s out_root=%s device=%s smoke=%s",
        args.phase,
        args.out_root,
        args.device,
        args.smoke,
    )
    seq = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for name in seq:
        PHASES[name](args)
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
