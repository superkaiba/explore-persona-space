#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ², Δ, ×) in scientific docstrings + log messages.
"""Issue #928 follow-up ``indiv-mlp-nonlinearity-control`` (plan v4): per-question MLP arms.

The disclosed missing control: is the per-question +0.203 CoT-augmented − direct
gap (95% CI +0.146..+0.272, frozen L25) a property of the information in the
context summary, or an artifact of the LINEAR estimator? Two batched MLP arms on
the EXISTING #928 per-question store — **Per-question MLP direct**
(``mlp_d_ctx2ans``: ctx mean-pool → ans PCA-48, d_in 3584) and **Per-question
MLP CoT-augmented** (``mlp_g_aug``: concat(ctx, cot) mean-pools → ans PCA-48,
d_in 7168) — at every capture layer × 50 context-GROUP LOCO folds, fit with the
#658 MLP constants (width 512, GELU, AdamW lr 1e-3 / wd 1e-4, 300 epochs,
seed 658) under FULL-DATA input standardization (the indiv LINEAR arms'
realized convention — plan §4.0), then paired-bootstrap re-reductions (shared
seed-42 index matrix) for the four registered reads of plan §6 at the three
layer conventions.

Single manipulated variable: estimator functional form (MLP vs closed-form
ridge) on identical rows, folds, targets, summaries, and conventions — the
target construction + folds are IMPORTED from ``issue928_fit_decomposition``
(``Store`` / ``_pca_target``) and asserted identical via the ss_tot-equality
check below, never re-implemented.

Consistency-checker WARN fixes (epm:consistency v2, 2026-07-04T10:32Z):

(a) **Executable H2 reproduction assert** (no vacuous digest check): before any
    new-arm read, regenerate the seed-42 bootstrap index matrix, reproduce the
    committed H2 primary observed + CI bounds from ``decomp_indiv.pt``, and
    assert equality (atol 1e-9) against the committed
    ``bootstrap_deltaskill.json``.
(b) **ss_tot-equality assert**: the new MLP arms' per-context ``ss_tot`` per
    layer must equal the linear arms' stored ``ss_tot`` in ``decomp_indiv.pt``
    (bitwise preferred; ≤1e-9 relative fail-loud otherwise — proves identical
    target construction, the one convention the serial-parity gate does not
    cover). Runs BEFORE the layer's fits so a target drift never burns GPU.

Restartability (the branch's r2/r3 standard): per-(arm, layer) durable units
(atomic tmp+``os.replace``) under ``<out>/partial/mlp_indiv/`` keyed by an
arg-keyed manifest carrying the generation identity (store identity digest +
pinned store revision, standardization, seeds, MLP constants, device); a
matching manifest SKIPS completed units, a mismatch discards stale units
(``prepare_checkpoint_dir`` — the #722 r3 never-reuse-wrong-rows lesson).

Estimator-validity gate (plan §6 — NARRATION only, never a row drop): the
closure read is informative only if MLP-augmented ≥ linear-augmented − 0.05 at
frozen L25; a miss is reported as "nonlinearity control inconclusive".

Usage::

    # production (GCP debug intent, single GPU):
    EPM_FIT_DEVICE=cuda uv run python scripts/issue928_mlp_indiv_control.py \\
        --store data/issue_928/store --decomp eval_results/issue_928/decomp_indiv.pt \\
        --out eval_results/issue_928/indiv-mlp-nonlinearity-control

    # smoke (= the SAME driver on the synthetic fixture; no separate smoke path):
    uv run python scripts/issue928_mlp_indiv_control.py --make-synth-fixture /tmp/i928mlp
    uv run python scripts/issue928_mlp_indiv_control.py \\
        --store /tmp/i928mlp/store --decomp /tmp/i928mlp/decomp_synth.pt \\
        --reference-bootstrap /tmp/i928mlp/reference_bootstrap.json \\
        --out /tmp/i928mlp/out --figures-dir /tmp/i928mlp/figures \\
        --layers 25 --n-boot 100 --expect-rows 24 --expect-contexts 4 \\
        --expect-layers 2 --expect-hidden 8 --skip-upload
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    MLP_HIDDEN,
    MLP_LR,
    MLP_MAX_EPOCHS,
    MLP_WD,
    _requested_device,
    _resolve_device,
)
from issue928_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    HF_DATA_REPO,
    HF_PREFIX_928,
    SUMMARY_NAMES,
    dump_json,
    load_json,
    part_summary_name,
    reproducibility_metadata,
    upload_folder_scoped_verify,
)
from issue928_fit_decomposition import (  # noqa: E402
    Store,
    _atomic_torch_save,
    _pca_target,
    prepare_checkpoint_dir,
)
from issue928_null_bootstrap import (  # noqa: E402
    GroupRidgeDesign,
    bootstrap_skills,
    fit_predict_grouped,
    group_folds,
    group_train_means,
    grouped_skill,
    make_bootstrap_index_matrix,
    stat_summary,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    DEFAULT_MLP_SEED,
    MLPGroup,
    assert_group_mlp_matches_serial,
    fit_batched_loco_mlp_multihead,
)

logger = logging.getLogger("issue928_mlp_indiv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── pins (plan v4 §10 reproducibility card) ───────────────────────────────────

STORE_REVISION = "5c1e3c5c00a6c386198179e9316cb77509ccf7b1"  # Hub-verified (plan §10)
STORE_HF_PREFIX = f"{HF_PREFIX_928}/analysis_tensors/store"
DECOMP_HF_PATH = f"{HF_PREFIX_928}/analysis_tensors/decomp/decomp_indiv.pt"
MLP_INDIV_TENSORS_PREFIX = f"{HF_PREFIX_928}/analysis_tensors/mlp_indiv"
MLP_INDIV_RESULTS_PREFIX = f"{HF_PREFIX_928}/fit_results/indiv_mlp_control"

COMBO = "mean/mean"  # the registered indiv combo (plan §6)
MLP_ARMS = ("mlp_d_ctx2ans", "mlp_g_aug")  # plan §5 condition slugs
LIN_D, LIN_G = "d_ctx2ans", "g_aug"  # reused linear reference arms (no refit)
VALIDITY_GATE_MARGIN = 0.05  # plan §6 estimator-validity narration gate
CLOSURE_SMALL_BIN = 1.0 / 3.0  # plan §6 narration bin (anchored on avg_q ~14%)
H2_REPRO_ATOL = 1e-9  # pure re-reduction of identical fp64 arrays — near-exact

# Production store identity (plan §10; the smoke fixture overrides via CLI).
EXPECTED_ROWS = 1994
EXPECTED_CONTEXTS = 50
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584


# ── input staging (local-first → pinned-revision HF fetch → fail-loud) ────────


def _hf_fetch_one(path_in_repo: str, revision: str, dest: Path) -> None:
    """One pinned-revision ``hf_hub_download`` with bounded retry + linear backoff."""
    import shutil

    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(4):
        try:
            got = hf_hub_download(
                HF_DATA_REPO, path_in_repo, repo_type="dataset", revision=revision
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(got, dest)
            return
        except Exception as exc:
            last = exc
            wait = 20 * (attempt + 1)
            logger.warning(
                "[stage] %s attempt %d failed (%s); retry in %ds",
                path_in_repo,
                attempt + 1,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"HF fetch failed after 4 attempts: {path_in_repo}") from last


def store_local_relpath(hub_rel: str) -> str:
    """Map a ``STORE_HF_PREFIX``-relative Hub path onto the LOCAL layout ``Store`` expects.

    On the Hub ALL 51 store files — the 50 per-context ``.pt`` blobs AND
    ``manifest.json`` — live flat under ``.../analysis_tensors/store/
    percq_summaries/`` (the extractor uploads the manifest INSIDE that folder:
    ``issue928_extract_thinking_store.py`` puts it at
    ``{STORE_PREFIX}/manifest.json`` where ``STORE_PREFIX`` already ends in
    ``/percq_summaries``), while ``Store(store_dir)`` reads
    ``<store>/manifest.json`` + ``<store>/percq_summaries/<ctx>.pt``. Mirroring
    the Hub prefix verbatim staged the manifest at
    ``<store>/percq_summaries/manifest.json`` and crashed ``Store()`` at init
    (att-20260704-120700). Mapping: any ``manifest.json`` goes to the store
    ROOT; every other file keeps its prefix-relative path (the ``.pt`` blobs
    already sit under ``percq_summaries/``).
    """
    if PurePosixPath(hub_rel).name == "manifest.json":
        return "manifest.json"
    return hub_rel


def stage_store(store_dir: Path, revision: str, hf_prefix: str = STORE_HF_PREFIX) -> None:
    """Stage the pinned per-question store (manifest + 50 blobs) if not local.

    SCOPED ``list_repo_tree(path_in_repo=...)`` enumeration (never
    ``snapshot_download`` / bare ``list_repo_files`` on the ~1M-file data repo —
    gotchas #833) + per-file ``hf_hub_download`` at the PINNED revision, ≤6
    workers. ``hf_prefix`` is the issue-profile store root (default: the #928
    prefix; the #1005 driver passes its own so a fallback stage can never
    silently fetch the PARENT's store). Local destinations go through
    ``store_local_relpath`` so the staged tree is EXACTLY the layout
    ``Store(store_dir)`` reads — the entry-time missing-check, the fetch
    destinations, and the completeness check all key on the SAME mapped paths.
    Completeness = every listed file exists locally at its mapped path;
    fail-loud if the enumeration carries no ``manifest.json`` (a stage without
    it is a doomed ``Store()`` init).
    """
    from collections import Counter

    from huggingface_hub import HfApi

    api = HfApi()
    entries = [
        e
        for e in api.list_repo_tree(
            HF_DATA_REPO,
            path_in_repo=hf_prefix,
            repo_type="dataset",
            recursive=True,
            revision=revision,
        )
        if getattr(e, "size", None) is not None
    ]
    if not entries:
        raise RuntimeError(f"no files under {hf_prefix} at revision {revision}")
    pairs = [(store_local_relpath(e.path[len(hf_prefix) + 1 :]), e.path) for e in entries]
    dupes = sorted(rel for rel, n in Counter(rel for rel, _ in pairs).items() if n > 1)
    if dupes:
        raise RuntimeError(
            f"HF→local store layout mapping collision under {hf_prefix} at "
            f"revision {revision}: {dupes[:3]}"
        )
    rels = dict(pairs)
    if "manifest.json" not in rels:
        raise RuntimeError(
            f"no manifest.json under {hf_prefix} at revision {revision} — "
            "Store() requires it at the store root; refusing a doomed stage"
        )
    missing = {rel: full for rel, full in rels.items() if not (store_dir / rel).is_file()}
    if not missing:
        logger.info("[phase=stage] store already local (%d files) — skip", len(rels))
        return
    logger.info(
        "[phase=stage] fetching %d/%d store files @ %s", len(missing), len(rels), revision[:12]
    )
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            ex.submit(_hf_fetch_one, full, revision, store_dir / rel): rel
            for rel, full in missing.items()
        }
        for fut in as_completed(futs):
            fut.result()  # re-raise loud
    still = [rel for rel in rels if not (store_dir / rel).is_file()]
    if still:
        raise RuntimeError(f"store staging incomplete: {len(still)} missing (e.g. {still[:3]})")


def stage_decomp(decomp_path: Path, revision: str, hf_path: str = DECOMP_HF_PATH) -> None:
    """Local-first ``decomp_indiv.pt``; else the pinned-revision HF mirror.

    The file is an UNTRACKED local artifact on the VM (never git-committed), so
    the git-clone-only GCP lane MUST fetch it from the Hub (the #779 r4-r5
    HF-fallback lesson); fail-loud if neither source resolves. ``hf_path`` is
    the issue-profile Hub path (default: the #928 artifact; the #1005 driver
    passes its own decomp so a fallback fetch never grabs the parent's).
    """
    if decomp_path.is_file():
        logger.info("[phase=stage] decomp already local: %s", decomp_path)
        return
    logger.info("[phase=stage] fetching %s @ %s", hf_path, revision[:12])
    _hf_fetch_one(hf_path, revision, decomp_path)


# ── committed-artifact loading ────────────────────────────────────────────────


def load_decomp(path: Path) -> dict[tuple[str, str, int], dict[str, np.ndarray]]:
    """Load a ``decomp_*.pt`` (string-keyed ``str((arm, combo, layer))``) as tuples."""
    raw = torch.load(path, weights_only=False)
    out: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for k, v in raw.items():
        arm, combo, layer = ast.literal_eval(k) if isinstance(k, str) else k
        out[(str(arm), str(combo), int(layer))] = {
            "ss_res": np.asarray(v["ss_res"], dtype=np.float64),
            "ss_tot": np.asarray(v["ss_tot"], dtype=np.float64),
        }
    return out


def _obs_skill(entry: dict[str, np.ndarray]) -> float:
    tot = float(entry["ss_tot"].sum())
    return float("nan") if tot < 1e-12 else 1.0 - float(entry["ss_res"].sum()) / tot


def _best_layer(decomp: dict, arm: str, combo: str) -> int:
    layers = sorted(la for (a, c, la) in decomp if a == arm and c == combo)
    skills = [_obs_skill(decomp[(arm, combo, la)]) for la in layers]
    return int(layers[int(np.nanargmax(skills))])


def reproduce_reference_h2(decomp: dict, reference_path: Path, atol: float = H2_REPRO_ATOL) -> dict:
    """Executable reproduction of the committed H2 primary read (WARN fix a).

    Regenerates the shared bootstrap index matrix from the reference's own
    (seed, n_boot), recomputes the H2 Δ(g_aug − d_ctx2ans) primary observed +
    CI bounds from ``decomp`` at the recomputed frozen direct-best layer, and
    asserts equality (atol) against the committed reference JSON — including
    the frozen layer itself. Raises on ANY mismatch (never proceeds to a
    new-arm read on drifted inputs).
    """
    ref = load_json(reference_path)
    seed, n_boot = int(ref["seed"]), int(ref["n_boot"])
    ind = ref["by_regime"]["indiv"]
    ref_layer = int(ind["layer_conventions"]["primary_frozen_direct_best_layer"])
    ref_stat = ind["statistics"]["H2_delta_g_minus_d"]["primary_frozen_direct_best"]

    l_primary = _best_layer(decomp, LIN_D, COMBO)
    assert l_primary == ref_layer, (
        f"frozen direct-best layer drifted: recomputed {l_primary} != committed {ref_layer}"
    )
    n_ctx = int(decomp[(LIN_D, COMBO, l_primary)]["ss_tot"].shape[0])
    idx = make_bootstrap_index_matrix(n_ctx, n_boot, seed)
    d_g = decomp[(LIN_G, COMBO, l_primary)]
    d_d = decomp[(LIN_D, COMBO, l_primary)]
    obs = _obs_skill(d_g) - _obs_skill(d_d)
    draws = bootstrap_skills(d_g["ss_res"], d_g["ss_tot"], idx) - bootstrap_skills(
        d_d["ss_res"], d_d["ss_tot"], idx
    )
    got = stat_summary(obs, draws)
    for name, got_v, ref_v in [
        ("observed", got["observed"], ref_stat["observed"]),
        ("ci_lo", got["ci95"][0], ref_stat["ci95"][0]),
        ("ci_hi", got["ci95"][1], ref_stat["ci95"][1]),
    ]:
        assert abs(got_v - ref_v) <= atol, (
            f"H2 primary reproduction FAILED at {name}: recomputed {got_v!r} vs "
            f"committed {ref_v!r} (atol {atol}) — inputs drifted; refusing new-arm reads"
        )
    logger.info(
        "[phase=h2_repro] committed H2 primary reproduced: obs=%.6f ci=[%.6f, %.6f] @ L%d",
        got["observed"],
        got["ci95"][0],
        got["ci95"][1],
        l_primary,
    )
    return {
        "reproduced": True,
        "frozen_layer": l_primary,
        "observed": got["observed"],
        "ci95": got["ci95"],
        "reference": str(reference_path),
        "atol": atol,
    }


# ── target construction (IMPORTED, never re-implemented) + ss_tot audit ───────


def layer_inputs(store: Store, li: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X_ctx, X_cat, Y_ans_pca) at layer index ``li`` — the fit driver's exact
    indiv mean/mean construction (``Store.indiv`` + full-data ``_pca_target``)."""
    x_ctx = store.indiv(part_summary_name("ctx", "mean"), li)
    x_cot = store.indiv(part_summary_name("cot", "mean"), li)
    y_pca, _, _ = _pca_target(store.indiv(part_summary_name("ans", "mean"), li))
    return x_ctx, np.concatenate([x_ctx, x_cot], axis=1), y_pca


def ss_tot_by_group(y_pca: np.ndarray, folds) -> np.ndarray:
    """(n_ctx,) per-context SS_tot vs the group-fold train-mean baseline."""
    tmean = group_train_means(y_pca, folds)
    return np.asarray(
        [float(np.sum((y_pca[held] - tmean[held]) ** 2)) for _tr, held in folds],
        dtype=np.float64,
    )


def assert_ss_tot_matches_linear(
    computed: np.ndarray, decomp: dict, layer: int, rtol: float = 1e-9
) -> dict:
    """WARN fix (b): the recomputed per-context ``ss_tot`` must equal BOTH linear
    reference arms' stored values at this layer — bitwise preferred; a ≤``rtol``
    relative deviation (cross-machine LAPACK last-bit drift in the shared SVD)
    is tolerated with a WARNING; anything larger raises (different target
    construction — the one convention the serial-parity gate cannot see)."""
    audit: dict = {"layer": int(layer)}
    for arm in (LIN_D, LIN_G):
        stored = decomp[(arm, COMBO, layer)]["ss_tot"]
        assert stored.shape == computed.shape, (arm, layer, stored.shape, computed.shape)
        exact = bool(np.array_equal(stored, computed))
        max_rel = 0.0 if exact else float(np.max(np.abs(stored - computed) / np.abs(stored)))
        if not exact:
            if max_rel > rtol:
                raise AssertionError(
                    f"ss_tot mismatch vs stored linear arm {arm} @ L{layer}: max rel dev "
                    f"{max_rel:.3e} > {rtol} — target construction differs; aborting"
                )
            logger.warning(
                "[phase=fit] ss_tot vs %s @ L%d not bitwise (max rel %.3e ≤ %.0e) — "
                "same construction, cross-machine fp drift",
                arm,
                layer,
                max_rel,
                rtol,
            )
        audit[arm] = {"exact": exact, "max_rel_dev": max_rel}
    return audit


# ── the batched MLP fits (per-(arm, layer) durable units) ─────────────────────


def fit_manifest_key(
    store: Store,
    layers: list[int],
    device: str,
    chunk_size: int,
    store_revision: str = STORE_REVISION,
) -> dict:
    """Arg-keyed resume manifest over EVERY output-affecting arg (r2/r3 standard:
    generation identity = store identity digest + pinned revision; estimator =
    standardization + seed + the #658 MLP constants; device)."""
    return {
        "round": "indiv-mlp-nonlinearity-control",
        "store_identity": store.identity_digest(),
        "store_revision": store_revision,
        "layers": [int(x) for x in layers],
        "arms": list(MLP_ARMS),
        "combo": COMBO,
        "standardization": "full_data",
        "seed": DEFAULT_MLP_SEED,
        "mlp": {"hidden": MLP_HIDDEN, "lr": MLP_LR, "wd": MLP_WD, "epochs": MLP_MAX_EPOCHS},
        "device": device,
        "chunk_size": int(chunk_size),
    }


def run_mlp_fits(
    store: Store,
    layers: list[int],
    decomp: dict,
    device: str,
    chunk_size: int,
    ckpt_dir: Path,
) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    """Fit both MLP arms at every requested layer (VALUE) as batched multihead
    group-fold ensembles; per-(arm, layer) atomic units + entry-time skip.

    Returns ``(units, ss_tot_audits)`` where ``units[(arm, layer)]`` carries the
    held-out fp32 predictions + per-context ss_res/ss_tot + pooled skill. The
    ss_tot-equality audit (WARN fix b) runs per layer BEFORE its fits.
    """
    n_ctx = len(store.ctx_ids)
    folds = group_folds(store.groups, list(range(n_ctx)))
    units: dict[tuple[str, int], dict] = {}
    audits: list[dict] = []
    for layer in layers:
        li = store.layers.index(layer)
        t0 = time.time()
        # ``layer_*`` prefix so ``prepare_checkpoint_dir``'s resume log counts them.
        unit_paths = {arm: ckpt_dir / f"layer_{int(layer)}_{arm}.pt" for arm in MLP_ARMS}
        if all(p.is_file() for p in unit_paths.values()):
            for arm, p in unit_paths.items():
                unit = torch.load(p, weights_only=False)
                assert unit["arm"] == arm and int(unit["layer"]) == int(layer), (arm, layer)
                units[(arm, layer)] = unit
                if arm == MLP_ARMS[0]:  # one audit entry per layer (shared target)
                    audits.append(unit["ss_tot_audit"])
            logger.info("[phase=fit] L%d SKIPPED (resumed both arms from partial units)", layer)
            continue
        x_ctx, x_cat, y_pca = layer_inputs(store, li)
        # WARN fix (b) — BEFORE burning GPU on this layer's fits.
        computed_tot = ss_tot_by_group(y_pca, folds)
        audit = assert_ss_tot_matches_linear(computed_tot, decomp, layer)
        audits.append(audit)
        y32 = y_pca.astype(np.float32)
        for arm, x in [("mlp_d_ctx2ans", x_ctx), ("mlp_g_aug", x_cat)]:
            path = unit_paths[arm]
            if path.is_file():
                unit = torch.load(path, weights_only=False)
                assert unit["arm"] == arm and int(unit["layer"]) == int(layer), (arm, layer)
                units[(arm, layer)] = unit
                logger.info("[phase=fit] %s L%d SKIPPED (resumed)", arm, layer)
                continue
            res = fit_batched_loco_mlp_multihead(
                [MLPGroup((arm, layer), x.astype(np.float32), y32)],
                seed=DEFAULT_MLP_SEED,
                device=device,
                chunk_size=chunk_size,
                row_groups=store.groups,
                standardization="full_data",
            )
            preds = res.preds_by_key[(arm, layer)]
            assert np.isfinite(preds).all(), (
                f"non-finite MLP predictions at ({arm}, L{layer}) — fit diverged (kill "
                "criterion, plan §7)"
            )
            gs = grouped_skill(preds, y_pca, folds)
            unit = {
                "arm": arm,
                "layer": int(layer),
                "combo": COMBO,
                "preds": preds.astype(np.float32),
                "ss_res": np.asarray(gs["ss_res_by_group"], dtype=np.float64),
                "ss_tot": np.asarray(gs["ss_tot_by_group"], dtype=np.float64),
                "skill": float(gs["skill"]),
                "ctx_order": list(store.ctx_ids),
                "n_members": int(res.n_members),
                "chunk_size": int(res.chunk_size),
                "ss_tot_audit": audit,
            }
            _atomic_torch_save(unit, path)
            units[(arm, layer)] = unit
            logger.info(
                "[phase=fit] %s L%d skill=%.4f (%d members, chunk %d, %.1fs)",
                arm,
                layer,
                unit["skill"],
                unit["n_members"],
                unit["chunk_size"],
                time.time() - t0,
            )
    return units, audits


# ── registered reads (plan §6, reads 1-4 × three conventions) ─────────────────


def _closure_summary(num_draws: np.ndarray, den_draws: np.ndarray, obs: float) -> dict:
    """Per-draw ratio percentile CI for the closure fraction (read 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num_draws / den_draws
    finite = ratio[np.isfinite(ratio)]
    return {
        "observed": float(obs),
        "ci95": [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))],
        "p_le_0": float(np.mean(finite <= 0.0)),
        "n_draws": int(finite.size),
        "denominator_frac_below_0p01": float(np.mean(den_draws < 0.01)),
    }


def registered_reads(
    lin: dict, mlp: dict[tuple[str, int], dict], l_primary: int, n_boot: int
) -> dict:
    """Reads 1-4 at the three layer conventions (paired; ONE shared idx matrix).

    ``lin``: the committed linear decomp (tuple keys). ``mlp``: this round's
    units. Conventions mirror ``bootstrap_deltaskill.json``:
    ``primary_frozen_direct_best`` (all four arms at ``l_primary``),
    ``secondary_own_best_frozen_full_data`` (each arm's own full-data best
    layer, frozen pre-bootstrap, labeled data-selected),
    ``secondary_best_vs_best_inherited`` (per-draw max over each arm's layers).
    """
    n_ctx = int(lin[(LIN_D, COMBO, l_primary)]["ss_tot"].shape[0])
    idx = make_bootstrap_index_matrix(n_ctx, n_boot, BOOTSTRAP_SEED)

    lin_layers = sorted(la for (a, c, la) in lin if a == LIN_D and c == COMBO)
    mlp_layers = sorted(la for (a, la) in mlp if a == "mlp_d_ctx2ans")
    arms = {
        LIN_D: {la: lin[(LIN_D, COMBO, la)] for la in lin_layers},
        LIN_G: {la: lin[(LIN_G, COMBO, la)] for la in lin_layers},
        "mlp_d_ctx2ans": {la: mlp[("mlp_d_ctx2ans", la)] for la in mlp_layers},
        "mlp_g_aug": {la: mlp[("mlp_g_aug", la)] for la in mlp_layers},
    }

    def obs(arm: str, la: int) -> float:
        # Every entry (linear decomp row OR MLP unit) carries ss_res/ss_tot — the
        # pooled skill is always the same re-reduction.
        return _obs_skill(arms[arm][la])

    def draws(arm: str, la: int) -> np.ndarray:
        e = arms[arm][la]
        return bootstrap_skills(
            np.asarray(e["ss_res"], dtype=np.float64),
            np.asarray(e["ss_tot"], dtype=np.float64),
            idx,
        )

    def own_best(arm: str) -> int:
        las = sorted(arms[arm])
        return int(las[int(np.nanargmax([obs(arm, la) for la in las]))])

    def max_draws(arm: str) -> np.ndarray:
        return np.nanmax(np.stack([draws(arm, la) for la in sorted(arms[arm])], axis=1), axis=1)

    contrasts = {
        "residual_gap_linG_minus_mlpD": (LIN_G, "mlp_d_ctx2ans"),
        "nonlinear_cot_gain_mlpG_minus_mlpD": ("mlp_g_aug", "mlp_d_ctx2ans"),
        "nonlinearity_increment_mlpD_minus_linD": ("mlp_d_ctx2ans", LIN_D),
    }
    best = {arm: own_best(arm) for arm in arms}
    out: dict = {
        "layer_conventions": {
            "primary_frozen_direct_best_layer": int(l_primary),
            "own_best_frozen_full_data": {a: int(b) for a, b in best.items()},
            "mlp_layers_fitted": [int(x) for x in mlp_layers],
        },
        "skills": {arm: {int(la): float(obs(arm, la)) for la in sorted(arms[arm])} for arm in arms},
        "statistics": {},
    }
    for name, (hi, lo) in contrasts.items():
        d_p = draws(hi, l_primary) - draws(lo, l_primary)
        d_ob = draws(hi, best[hi]) - draws(lo, best[lo])
        d_bb = max_draws(hi) - max_draws(lo)
        out["statistics"][name] = {
            "primary_frozen_direct_best": stat_summary(
                obs(hi, l_primary) - obs(lo, l_primary), d_p
            ),
            "secondary_own_best_frozen_full_data": {
                "layers": {"hi": best[hi], "lo": best[lo]},
                **stat_summary(obs(hi, best[hi]) - obs(lo, best[lo]), d_ob),
            },
            "secondary_best_vs_best_inherited": stat_summary(
                obs(hi, best[hi]) - obs(lo, best[lo]), d_bb
            ),
        }
    # Read 1 — closure fraction (per-draw ratio of the paired deltas).
    closure: dict = {}
    for conv in (
        "primary_frozen_direct_best",
        "secondary_own_best_frozen_full_data",
        "secondary_best_vs_best_inherited",
    ):
        if conv == "primary_frozen_direct_best":
            la = {a: l_primary for a in arms}
            dr = {a: draws(a, l_primary) for a in arms}
        elif conv == "secondary_own_best_frozen_full_data":
            la = best
            dr = {a: draws(a, best[a]) for a in arms}
        else:
            la = best
            dr = {a: max_draws(a) for a in arms}
        num_obs = obs("mlp_d_ctx2ans", la["mlp_d_ctx2ans"]) - obs(LIN_D, la[LIN_D])
        den_obs = obs(LIN_G, la[LIN_G]) - obs(LIN_D, la[LIN_D])
        closure[conv] = _closure_summary(
            dr["mlp_d_ctx2ans"] - dr[LIN_D], dr[LIN_G] - dr[LIN_D], num_obs / den_obs
        )
    out["statistics"]["closure_fraction"] = closure

    # Estimator-validity gate (plan §6 — narration only) + interpretation branch.
    mlp_g_p = obs("mlp_g_aug", l_primary)
    lin_g_p = obs(LIN_G, l_primary)
    gate_pass = bool(mlp_g_p >= lin_g_p - VALIDITY_GATE_MARGIN)
    out["estimator_validity_gate"] = {
        "pass": gate_pass,
        "mlp_augmented_skill_L_primary": float(mlp_g_p),
        "linear_augmented_skill_L_primary": float(lin_g_p),
        "margin": VALIDITY_GATE_MARGIN,
        "note": (
            "narration gate — on a miss the closure read is narrated as 'nonlinearity "
            "control inconclusive (estimator under-fits at this shape)'; nothing dropped"
        ),
    }
    resid = out["statistics"]["residual_gap_linG_minus_mlpD"]["primary_frozen_direct_best"]
    closure_pt = closure["primary_frozen_direct_best"]["observed"]
    resid_ci_excludes_0_pos = resid["ci95"][0] > 0.0
    if not gate_pass:
        branch = "inconclusive_estimator_underfits"
    elif not resid_ci_excludes_0_pos:
        branch = "large_closure_linearity_artifact"
    elif closure_pt < CLOSURE_SMALL_BIN:
        branch = "small_closure_headline_hardens"
    else:
        branch = "intermediate_partial_nonlinearity"
    out["interpretation_branch"] = {
        "branch": branch,
        "closure_point_primary": float(closure_pt),
        "residual_gap_ci_excludes_0_positive": bool(resid_ci_excludes_0_pos),
        "closure_small_bin": CLOSURE_SMALL_BIN,
    }
    return out


# ── figures (plan §6: hero 4-arm bars, per-layer curves, per-context scatter) ──


def make_figures(figdir: Path, lin: dict, mlp: dict, l_primary: int, n_boot: int) -> list[str]:
    """Hero 4-arm bar/CI at frozen L{primary} (identity ceiling in-figure) +
    per-layer MLP-vs-linear curves + labeled per-context Δ scatter."""
    try:
        from issue928_length_matched_gain import FLAGGED_BELOW_PARSE_FLOOR
    except ImportError:  # fixture / minimal checkouts
        FLAGGED_BELOW_PARSE_FLOOR = ()
    set_paper_style()
    figdir.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    n_ctx = int(lin[(LIN_D, COMBO, l_primary)]["ss_tot"].shape[0])
    idx = make_bootstrap_index_matrix(n_ctx, n_boot, BOOTSTRAP_SEED)

    def entry(arm: str, la: int) -> dict:
        return lin[(arm, COMBO, la)] if arm in (LIN_D, LIN_G, "ident") else mlp[(arm, la)]

    # 1. hero: four arms at the frozen primary layer, bootstrap CIs, ceiling line.
    order = [
        (LIN_D, "linear direct", paper_palette_role("baseline")),
        ("mlp_d_ctx2ans", "MLP direct", paper_palette_role("accent")),
        ("mlp_g_aug", "MLP augmented", paper_palette_role("control")),
        (LIN_G, "linear augmented", paper_palette_role("primary")),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = np.arange(len(order))
    for x, (arm, label, color) in zip(xs, order, strict=True):
        e = entry(arm, l_primary)
        sr = np.asarray(e["ss_res"], dtype=np.float64)
        st = np.asarray(e["ss_tot"], dtype=np.float64)
        skill = 1.0 - sr.sum() / st.sum()
        dr = bootstrap_skills(sr, st, idx)
        lo_q, hi_q = np.percentile(dr[np.isfinite(dr)], [2.5, 97.5])
        ax.bar(x, skill, color=color, width=0.62)
        ax.errorbar(
            x,
            skill,
            yerr=[[max(0.0, skill - lo_q)], [max(0.0, hi_q - skill)]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.text(x, 0.015, label, ha="center", va="bottom", rotation=90, fontsize=8)
    if ("ident", COMBO, l_primary) in lin:
        ceil = _obs_skill(lin[("ident", COMBO, l_primary)])
        ax.axhline(ceil, ls="--", color=paper_palette_role("neutral"), lw=1.0)
        ax.text(
            len(order) - 0.5,
            ceil,
            f"identity ceiling {ceil:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["lin D", "MLP D", "MLP G", "lin G"])
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_title(f"Per-question arms at frozen L{l_primary} (paired 95% bootstrap CIs)")
    stems.append("mlp_indiv_hero_4arm")
    savefig_paper(fig, f"{figdir.name}/mlp_indiv_hero_4arm", dir=str(figdir.parent))
    plt.close(fig)

    # 2. per-layer curves (linear from the committed decomp; MLP from this run).
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lin_layers = sorted(la for (a, c, la) in lin if a == LIN_D and c == COMBO)
    mlp_layers = sorted(la for (a, la) in mlp if a == "mlp_d_ctx2ans")
    for arm, label, color, ls, las in [
        (LIN_D, "linear direct", paper_palette_role("baseline"), "-", lin_layers),
        (LIN_G, "linear augmented", paper_palette_role("primary"), "-", lin_layers),
        ("mlp_d_ctx2ans", "MLP direct", paper_palette_role("accent"), "--", mlp_layers),
        ("mlp_g_aug", "MLP augmented", paper_palette_role("control"), "--", mlp_layers),
    ]:
        ys = [
            1.0
            - np.asarray(entry(arm, la)["ss_res"]).sum()
            / np.asarray(entry(arm, la)["ss_tot"]).sum()
            for la in las
        ]
        ax.plot(las, ys, ls, color=color, marker="o", ms=3, label=label)
    ax.axvline(l_primary, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_title("Per-question skills by layer (mean/mean, group-LOCO)")
    ax.legend(fontsize=8)
    stems.append("mlp_indiv_per_layer_curves")
    savefig_paper(fig, f"{figdir.name}/mlp_indiv_per_layer_curves", dir=str(figdir.parent))
    plt.close(fig)

    # 3. labeled per-context Δ(MLP direct − linear direct) scatter at L_primary.
    e_mlp = mlp[("mlp_d_ctx2ans", l_primary)]
    e_lin = lin[(LIN_D, COMBO, l_primary)]
    ctx_ids = e_mlp["ctx_order"]
    sk_lin = 1.0 - np.asarray(e_lin["ss_res"]) / np.asarray(e_lin["ss_tot"])
    sk_mlp = 1.0 - np.asarray(e_mlp["ss_res"]) / np.asarray(e_mlp["ss_tot"])
    delta = sk_mlp - sk_lin
    flagged = np.array([c in FLAGGED_BELOW_PARSE_FLOOR for c in ctx_ids])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(
        sk_lin[~flagged], delta[~flagged], s=18, color=paper_palette_role("accent"), zorder=3
    )
    if flagged.any():
        ax.scatter(
            sk_lin[flagged],
            delta[flagged],
            s=22,
            color=paper_palette_role("control"),
            zorder=3,
            label="below parse floor (flagged)",
        )
        ax.legend(fontsize=8)
    for cid, x, y in zip(ctx_ids, sk_lin, delta, strict=True):
        ax.annotate(cid, (x, y), fontsize=5, alpha=0.8, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel(f"per-context linear-direct skill (L{l_primary})")
    ax.set_ylabel("per-context Δ skill (MLP direct - linear direct)")
    ax.set_title("Nonlinearity increment per context")
    stems.append("mlp_indiv_percontext_delta")
    savefig_paper(fig, f"{figdir.name}/mlp_indiv_percontext_delta", dir=str(figdir.parent))
    plt.close(fig)
    return stems


# ── synthetic fixture (smoke + tests — data-gen only; the DRIVER path is one) ──


def build_synth_fixture(root: Path, seed: int = 9280) -> dict:
    """Tiny synthetic store + linear reference decomp + reference bootstrap JSON.

    4 contexts × unequal row counts (24 rows) × capture layers [24, 25] × H=8,
    with a planted linear ctx→ans signal STRONGER at layer 25 so the frozen
    direct-best layer is 25 (asserted). The linear reference arms are fit with
    the PRODUCTION machinery (``GroupRidgeDesign`` full_data +
    ``fit_predict_grouped`` + ``grouped_skill``) and the reference JSON is
    produced by the parent's own ``bootstrap_statistics`` — so the driver's H2
    reproduction assert + ss_tot-equality assert exercise the REAL invariants
    end-to-end at tiny N. Returns the fixture paths.
    """
    from issue928_fit_decomposition import bootstrap_statistics

    rng = np.random.default_rng(seed)
    store_dir = root / "store"
    (store_dir / "percq_summaries").mkdir(parents=True, exist_ok=True)
    ctx = [f"c{i}" for i in range(4)]
    fams = {c: ("famA" if i < 2 else "famB") for i, c in enumerate(ctx)}
    layers = [24, 25]
    h = 8
    rows = [6, 6, 5, 7]
    dump_json(
        {
            "context_ids": ctx,
            "families": fams,
            "capture_layers": layers,
            "summary_names": list(SUMMARY_NAMES),
            "probe_pool_hash": "synthfixture",
            "model": "synthetic-fixture",
            "rung": "greedy",
            "max_new_tokens": 64,
        },
        store_dir / "manifest.json",
    )
    sidx = {n: i for i, n in enumerate(SUMMARY_NAMES)}
    w_ans = rng.standard_normal((h, h))
    w_cot = rng.standard_normal((h, h))
    for c, nq in zip(ctx, rows, strict=True):
        per_q = torch.from_numpy(
            rng.standard_normal((nq, len(SUMMARY_NAMES), len(layers), h)).astype(np.float32)
        )
        base = rng.standard_normal((nq, h))
        for li, noise in [(0, 1.5), (1, 0.3)]:  # layer 25 (index 1) = stronger signal
            x_ctx = base + noise * rng.standard_normal((nq, h))
            x_cot = base @ w_cot + noise * rng.standard_normal((nq, h))
            y_ans = base @ w_ans + noise * rng.standard_normal((nq, h))
            per_q[:, sidx[part_summary_name("ctx", "mean")], li] = torch.from_numpy(
                x_ctx.astype(np.float32)
            )
            per_q[:, sidx[part_summary_name("cot", "mean")], li] = torch.from_numpy(
                x_cot.astype(np.float32)
            )
            per_q[:, sidx[part_summary_name("ans", "mean")], li] = torch.from_numpy(
                y_ans.astype(np.float32)
            )
        blob = {"context_id": c, "per_q": per_q, "probe_avg": per_q.mean(0)}
        torch.save(blob, store_dir / "percq_summaries" / f"{c}.pt")

    store = Store(store_dir)
    folds = group_folds(store.groups, list(range(len(ctx))))
    decomp: dict = {}
    for li, layer in enumerate(layers):
        x_ctx, x_cat, y_pca = layer_inputs(store, li)
        x_cot = store.indiv(part_summary_name("cot", "mean"), li)
        for arm, x in [
            (LIN_D, x_ctx),
            (LIN_G, x_cat),
            ("b_cot2ans", x_cot),
            ("ident", store.indiv(part_summary_name("ans", "mean"), li)),
        ]:
            des = GroupRidgeDesign(x, folds, device="cpu", standardization="full_data")
            preds, _, _ = fit_predict_grouped(des, y_pca)
            gs = grouped_skill(preds, y_pca, folds)
            decomp[(arm, COMBO, layer)] = {
                "ss_res": np.asarray(gs["ss_res_by_group"]),
                "ss_tot": np.asarray(gs["ss_tot_by_group"]),
            }
            des.free()
        # comp_pred alias (bootstrap_statistics requires the key; fixture-only).
        decomp[("comp_pred", COMBO, layer)] = decomp[("b_cot2ans", COMBO, layer)]
    assert _best_layer(decomp, LIN_D, COMBO) == 25, "fixture must freeze L25 as direct-best"
    decomp_path = root / "decomp_synth.pt"
    torch.save(
        {str(k): {"ss_res": v["ss_res"], "ss_tot": v["ss_tot"]} for k, v in decomp.items()},
        decomp_path,
    )
    boot = bootstrap_statistics(decomp, len(ctx), 100)
    ref_path = root / "reference_bootstrap.json"
    dump_json(
        {
            "dv": "paired bootstrap delta-skill (synthetic fixture)",
            "seed": BOOTSTRAP_SEED,
            "n_boot": 100,
            "by_regime": {"indiv": boot},
            "reproducibility": reproducibility_metadata(),
        },
        ref_path,
    )
    return {"store": store_dir, "decomp": decomp_path, "reference": ref_path}


# ── main ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """The indiv-MLP-control CLI. Defaults preserve the #928 standalone
    behavior verbatim; an issue profile (e.g. the #1005 driver) overrides the
    HF prefixes so a child run can never clobber — or silently stage — the
    PARENT's Hub artifacts (upload-verification v1 FAIL, required action 3)."""
    ap = argparse.ArgumentParser(description="Issue #928 indiv MLP nonlinearity control")
    ap.add_argument("--store", default=str(PROJECT_ROOT / "data" / "issue_928" / "store"))
    ap.add_argument(
        "--decomp", default=str(PROJECT_ROOT / "eval_results" / "issue_928" / "decomp_indiv.pt")
    )
    ap.add_argument(
        "--reference-bootstrap",
        default=str(PROJECT_ROOT / "eval_results" / "issue_928" / "bootstrap_deltaskill.json"),
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_928" / "indiv-mlp-nonlinearity-control"),
    )
    ap.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    ap.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="layer VALUES to fit (default: every store capture layer)",
    )
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--device", default=None, help="CLI > EPM_FIT_DEVICE > auto")
    ap.add_argument("--chunk-size", type=int, default=4096, help="resolve_chunk_cap-bounded")
    ap.add_argument("--expect-rows", type=int, default=EXPECTED_ROWS)
    ap.add_argument("--expect-contexts", type=int, default=EXPECTED_CONTEXTS)
    ap.add_argument("--expect-layers", type=int, default=EXPECTED_LAYERS)
    ap.add_argument("--expect-hidden", type=int, default=EXPECTED_HIDDEN)
    ap.add_argument("--skip-parity-gate", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--results-upload-prefix",
        default=MLP_INDIV_RESULTS_PREFIX,
        help="HF prefix for the result JSONs (default: the #928 prefix — an issue "
        "profile like #1005 MUST override so it never overwrites the parent's)",
    )
    ap.add_argument(
        "--tensors-upload-prefix",
        default=MLP_INDIV_TENSORS_PREFIX,
        help="HF prefix for decomp_indiv_mlp.pt + preds/*.pt (same override contract)",
    )
    ap.add_argument(
        "--store-hf-prefix",
        default=STORE_HF_PREFIX,
        help="HF store root for the fallback stage_store (same override contract — "
        "a #1005 fallback stage must never silently fetch the #928 parent store)",
    )
    ap.add_argument(
        "--decomp-hf-path",
        default=DECOMP_HF_PATH,
        help="HF path for the fallback stage_decomp (same override contract)",
    )
    ap.add_argument(
        "--store-revision",
        default=STORE_REVISION,
        help="data-repo revision for the fallback stages (default: the #928 pin; "
        "an issue profile whose artifacts are not at that pin passes its own)",
    )
    ap.add_argument(
        "--allow-cpu-production",
        action="store_true",
        help="permit a cpu-resolved device at production store size (default: fail loud — "
        "the inherited _resolve_device falls back cuda→cpu with only a WARNING, and the "
        "production fit is ~28 PFLOP ≈ 190 h at the measured 41 GFLOP/s CPU rate)",
    )
    ap.add_argument(
        "--make-synth-fixture",
        default=None,
        metavar="DIR",
        help="generate the smoke/test fixture under DIR and exit",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.make_synth_fixture:
        paths = build_synth_fixture(Path(args.make_synth_fixture))
        logger.info("[phase=fixture] synthetic fixture written: %s", paths)
        return 0

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(_requested_device(args.device))
    logger.info("fit device: %s", device)

    # Phase 0 — serial-parity gate (both standardization modes) BEFORE any fit.
    if not args.skip_parity_gate:
        logger.info("[phase=parity] batched group-fold MLP vs serial reference (both modes)")
        parity = assert_group_mlp_matches_serial()
    else:
        parity = {"skipped": True}
    dump_json(
        {
            "gate": "assert_group_mlp_matches_serial",
            "modes": ["per_fold", "full_data"],
            "max_abs_deviation_per_check": parity,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "mlp_parity_gate.json",
    )

    # Phase 1 — stage pinned inputs (local-first → pinned-revision HF).
    store_dir = Path(args.store)
    decomp_path = Path(args.decomp)
    if not (store_dir / "manifest.json").is_file():
        stage_store(store_dir, args.store_revision, args.store_hf_prefix)
    stage_decomp(decomp_path, args.store_revision, args.decomp_hf_path)

    # Phase 2 — schema + row-count asserts (plan §12.1; the fail-loud identity check).
    store = Store(store_dir)
    n_rows = int(store.groups.shape[0])
    if device == "cpu" and n_rows >= 1000 and not args.allow_cpu_production:
        raise SystemExit(
            "resolved device is cpu at production store size — the inherited "
            "_resolve_device falls back cuda→cpu with only a WARNING, and the production "
            "fit (~28 PFLOP) is ~190 h at the measured CPU rate. Fix the GPU (or pass "
            "--allow-cpu-production to override deliberately)."
        )
    assert n_rows == args.expect_rows, (n_rows, args.expect_rows)
    assert len(store.ctx_ids) == args.expect_contexts, (len(store.ctx_ids), args.expect_contexts)
    assert len(store.layers) == args.expect_layers, (len(store.layers), args.expect_layers)
    assert args.expect_hidden == store.H, (store.H, args.expect_hidden)
    logger.info(
        "[phase=schema] store OK: %d rows / %d contexts / %d layers / H=%d",
        n_rows,
        len(store.ctx_ids),
        len(store.layers),
        store.H,
    )
    layers = args.layers if args.layers else [int(x) for x in store.layers]
    for layer in layers:
        assert layer in store.layers, (layer, store.layers)

    decomp = load_decomp(decomp_path)
    for arm in (LIN_D, LIN_G):
        for layer in layers:
            assert (arm, COMBO, layer) in decomp, (
                f"linear reference ({arm}, {COMBO}, {layer}) missing from {decomp_path}"
            )

    # Phase 3 — WARN fix (a): executable reproduction of the committed H2 primary.
    h2_record = reproduce_reference_h2(decomp, Path(args.reference_bootstrap))
    l_primary = int(h2_record["frozen_layer"])
    assert l_primary in layers, (
        f"frozen primary layer L{l_primary} not in the fitted layer set {layers} — "
        "the registered reads are uncomputable; pass --layers including it"
    )

    # Phase 4 — the batched MLP fits (per-(arm, layer) durable units + resume).
    ckpt_dir = prepare_checkpoint_dir(
        out_dir / "partial",
        "mlp_indiv",
        fit_manifest_key(store, layers, device, args.chunk_size, args.store_revision),
    )
    units, ss_tot_audits = run_mlp_fits(store, layers, decomp, device, args.chunk_size, ckpt_dir)

    # Phase 5 — registered reads 1-4 at the three conventions (plan §6).
    logger.info("[phase=reads] paired bootstrap re-reductions (n_boot=%d)", args.n_boot)
    reads = registered_reads(decomp, units, l_primary, args.n_boot)

    # Phase 6 — persist outputs (git deliverables + HF mirrors).
    torch.save(
        {
            str((arm, COMBO, layer)): {
                "ss_res": units[(arm, layer)]["ss_res"],
                "ss_tot": units[(arm, layer)]["ss_tot"],
            }
            for (arm, layer) in units
        },
        out_dir / "decomp_indiv_mlp.pt",
    )
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(exist_ok=True)
    for (arm, layer), unit in units.items():
        _atomic_torch_save(
            {
                "arm": arm,
                "layer": int(layer),
                "preds": unit["preds"],
                "ctx_order": unit["ctx_order"],
            },
            preds_dir / f"preds_{arm}_L{int(layer)}.pt",
        )
    validity = {
        "dv": "held-out skill-over-mean R^2 (per-question, group-LOCO, mean/mean)",
        "round": "indiv-mlp-nonlinearity-control",
        "estimator": (
            f"batched multihead MLP (width {MLP_HIDDEN}, GELU, AdamW lr {MLP_LR} / wd "
            f"{MLP_WD}, {MLP_MAX_EPOCHS} epochs, seed {DEFAULT_MLP_SEED}), group-LOCO folds, "
            "FULL-DATA input standardization (the indiv linear arms' realized convention); "
            "linear reference arms reused from decomp_indiv.pt (no refit)"
        ),
        "store_revision": args.store_revision,
        "store_identity_digest": store.identity_digest(),
        "n_rows": n_rows,
        "n_contexts": len(store.ctx_ids),
        "layers_fitted": [int(x) for x in layers],
        "h2_reproduction": h2_record,
        "ss_tot_equality_audit": ss_tot_audits,
        "parity_gate": parity,
        "reads": reads,
        "n_boot": args.n_boot,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(validity, out_dir / "mlp_indiv_validity.json")

    # Phase 7 — figures (hero + per-layer curves + labeled per-context scatter).
    logger.info("[phase=figures] -> %s", args.figures_dir)
    stems = make_figures(Path(args.figures_dir), decomp, units, l_primary, args.n_boot)

    # Phase 8 — HF uploads (GCE DELETEs the boot disk; everything must land).
    if not args.skip_upload:
        logger.info("[phase=upload] result JSONs -> %s", args.results_upload_prefix)
        names = sorted(p.name for p in out_dir.glob("*.json"))
        upload_folder_scoped_verify(
            out_dir,
            args.results_upload_prefix,
            names,
            f"issue #928 indiv MLP control: result JSONs ({len(names)})",
            allow_patterns=["*.json"],
            ignore_patterns=["partial/*", "preds/*"],
        )
        logger.info("[phase=upload] preds + decomp tensors -> %s", args.tensors_upload_prefix)
        tensor_names = [
            "decomp_indiv_mlp.pt",
            *sorted(f"preds/{p.name}" for p in preds_dir.glob("preds_*.pt")),
        ]
        upload_folder_scoped_verify(
            out_dir,
            args.tensors_upload_prefix,
            tensor_names,
            f"issue #928 indiv MLP control: held-out preds + decomp ({len(tensor_names)} .pt)",
            allow_patterns=["decomp_indiv_mlp.pt", "preds/*.pt"],
            ignore_patterns=["partial/*"],
        )
    else:
        logger.info("[phase=upload] SKIPPED (--skip-upload)")

    logger.info(
        "[phase=done] indiv MLP control complete in %.1fs -> %s (figures: %s)",
        time.time() - t0,
        out_dir,
        ", ".join(stems),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
