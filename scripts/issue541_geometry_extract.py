#!/usr/bin/env python3
# ruff: noqa: RUF002
# (multiplication-sign / arrow characters intentional in docstrings/labels)
"""Issue #541 follow-up ``geometry-plus-prior-joint-predictor`` — geometry extraction.

The ONLY GPU step of the amendment (plan §3.2 item 1): one pass over the frozen
base model (``Qwen/Qwen2.5-7B-Instruct``, HF bf16, batch-1 prompt-only forwards)
extracting last-input-token residual activations for the 24 #541 panel personas
× 40 A-family courthouse probes at layers {7, 14, 21, 22, 27}, then:

  (a) recomputed pairwise mean-per-probe cosine at all 5 layers (the phase0c
      formula, ``issue541_prescreen.step_0c``);
  (b) HARD fidelity gate — per-teacher-row Spearman >= 0.99 vs the committed
      ``phase0c_persona_vectors.json`` at layers {7,14,21,27}; on failure the
      run crashes with a diagnostic dump (probe sha, prompt hashes, per-row
      Spearmans) and does NOT proceed on divergent geometry;
  (c) pairwise Gaussian symmetric-KL in a PCA subspace (k=16 headline @ L22,
      k=8 robustness), NaN-guarded — function vendored verbatim from
      ``scripts/issue532_predictor_stress.py::_gaussian_sym_kl_in_subspace_local``
      (main branch), itself a re-implementation of #493's bakeoff metric;
  (d) writes ``eval_results/issue_541/geometry-plus-prior-joint-predictor/
      geometry_matrices.json`` (+ probe sha256, fidelity record, repro
      metadata) and per-layer fp16 ``.npy`` activation tensors, with a
      best-effort (fail-soft on quota 403) HF data-repo upload.

``--smoke``: 2 personas × 8 probes through the SAME code path, into the
``eval_results/issue_541_smoke/geometry-plus-prior-joint-predictor/``
namespace (fidelity gate computed but non-binding at n=2 — Spearman of a
2-element row is degenerate; the full run hard-asserts).

Pod-side contract (poll_pipeline.py): emits one ``[phase=<name>]`` line per
logical phase, writes the end-of-run sentinel (``sentinel_schema_version`` /
``kind`` / ``version``) to ``/workspace/logs/issue-541-epm_progress-<epoch>.json``
when ``/workspace`` exists, then emits the terminal ``[phase=done]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i541_geometry_extract")

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = (7, 14, 21, 22, 27)  # parent {7,14,21,27} + the #502/#532 gkl layer 22
FIDELITY_LAYERS = (7, 14, 21, 27)  # layers present in the committed phase0c matrix
GKL_KS = (16, 8)  # k=16 headline (#502 bakeoff winner via #532), k=8 robustness
FIDELITY_MIN_ROW_SPEARMAN = 0.99
N_PROBES_FULL = 40
N_PROBES_SMOKE = 8
N_PERSONAS_SMOKE = 2
SUBDIR = "geometry-plus-prior-joint-predictor"
DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_UPLOAD_PREFIX = "issue541_prior_stratified/geometry_plus_prior"

# Parent inputs are ALWAYS read from the real issue_541 tree (the smoke run
# still smokes against the committed full-run artifacts; only OUTPUTS are
# namespaced) — mirrors the brief's "smoke against the REAL committed
# predictors.json" instruction.
PREDICTORS_PATH = PROJECT_ROOT / "eval_results" / "issue_541" / "predictors.json"
PHASE0C_PATH = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_541"
    / "phase0_prescreen"
    / "phase0c_persona_vectors.json"
)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _log_phase(name: str) -> None:
    """One ``[phase=<name>]`` line per logical phase (poll_pipeline PHASE_RE)."""
    logger.info("[phase=%s] %s", name, _now_iso())


def _git_commit_sha() -> str:
    import os
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},  # explicit per the subprocess-env rule (no implicit inherit)
        check=False,
    )
    return out.stdout.strip() or "unknown"


def _repro_metadata() -> dict[str, Any]:
    import platform

    import torch
    import transformers

    return {
        "git_commit": _git_commit_sha(),
        "base_model": BASE_MODEL,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": _now_iso(),
    }


def _write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp + rename), mkdir -p."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho — delegated to the #500 helper (avoids a second impl)."""
    import issue500_predictors as i500

    return i500._spearman(x, y)


# ---------------------------------------------------------------------------
# Gaussian symmetric-KL — vendored VERBATIM from
# scripts/issue532_predictor_stress.py::_gaussian_sym_kl_in_subspace_local
# (main branch, commit lineage #532 <- #493 bakeoff). Self-contained numpy;
# the #532 module is not on the issue-541 branch, hence the copy (plan §3.1).
# ---------------------------------------------------------------------------
def _gaussian_sym_kl_in_subspace_local(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Gaussian symmetric-KL between two clouds in the top-k PCA subspace.

    Re-implementation of ``_gaussian_sym_kl_in_subspace`` from
    ``scripts/issue493_extraction_metric_bakeoff.py`` to keep #532
    self-contained (the bakeoff script is 3550 lines and importing it
    would pull a heavy dependency graph). Identical formula:

        KL(N0||N1) = 0.5 * (tr(Σ1^-1 Σ0) + (μ1-μ0)^T Σ1^-1 (μ1-μ0)
                              - k + log(det Σ1 / det Σ0))
        Symmetric-KL = 0.5 * (KL(0||1) + KL(1||0)).

    The PCA subspace is built via the Gram / dual trick (n=50 ≪ d=3584):
    eigendecompose the n×n Gram of the stacked centered clouds, project
    each cloud onto the top-k components.
    """
    Xa = Xa[~np.any(np.isnan(Xa), axis=1)]
    Xb = Xb[~np.any(np.isnan(Xb), axis=1)]
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    stacked = np.vstack([Xa, Xb])
    mu = stacked.mean(axis=0, keepdims=True)
    stacked_c = stacked - mu
    n, d = stacked_c.shape
    k_eff = min(k, n, d)
    G = stacked_c @ stacked_c.T
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1][:k_eff]
    lam = np.clip(eigvals[order], 1e-12, None)
    V_g = eigvecs[:, order]
    sqrt_lam = np.sqrt(lam)
    components = (stacked_c.T @ V_g) / sqrt_lam[None, :]  # (d, k)
    Ya = (Xa - mu) @ components
    Yb = (Xb - mu) @ components
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])

    def _one_kl(S0, S1, m0, m1):
        S1_inv = np.linalg.inv(S1)
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            return float("nan")
        d_inner = S0.shape[0]
        return 0.5 * (
            np.trace(S1_inv @ S0) + (m1 - m0) @ S1_inv @ (m1 - m0) - d_inner + (logdet1 - logdet0)
        )

    kl_ab = _one_kl(Sa, Sb, mu_a, mu_b)
    kl_ba = _one_kl(Sb, Sa, mu_b, mu_a)
    if np.isnan(kl_ab) or np.isnan(kl_ba):
        return float("nan")
    return float(0.5 * (kl_ab + kl_ba))


# ---------------------------------------------------------------------------
# Panel + probes
# ---------------------------------------------------------------------------
def _load_panel(smoke: bool) -> list[str]:
    """The 24 #541 panel personas from the committed predictors.json (smoke: first 2)."""
    pred = json.loads(PREDICTORS_PATH.read_text())
    panel = list(pred["panel"])
    assert len(panel) == 24, f"expected 24 panel personas, got {len(panel)}"
    return panel[:N_PERSONAS_SMOKE] if smoke else panel


def _panel_pool(panel: list[str]) -> dict[str, str | None]:
    """Persona-name -> system-prompt map (mirrors issue541_prescreen._candidate_pool)."""
    import issue444_persona_distance_topic as pdt
    from issue541_personas import inject_candidates

    inject_candidates()
    pool = {name: pdt._resolve_persona_prompt(name) for name in panel}
    assert len(pool) == len(panel), (len(pool), len(panel))
    missing = [n for n in panel if n not in pool]
    assert not missing, f"panel personas failed to resolve: {missing}"
    return pool


def _build_probes(n_probes: int) -> list[str]:
    """First-n A-family on-topic probes — identical slice to phase0c
    (issue541_prescreen.py step_0c: flatten build_reformulation_probes, [:n])."""
    import issue444_persona_distance_topic as pdt

    from eval.exp444_judge_prompts import build_reformulation_probes

    a_family = [pr for probes in build_reformulation_probes(pdt.ENTITY).values() for pr in probes]
    probes = a_family[:n_probes]
    assert len(probes) == n_probes, (len(probes), n_probes)
    return probes


def _probe_sha256(probes: list[str]) -> str:
    return hashlib.sha256(json.dumps(probes, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Extraction (resumable: per-persona fp32 .npz cache; matrices ALWAYS computed
# from the cached fp32 arrays so fresh and resumed runs are bit-identical)
# ---------------------------------------------------------------------------
def _extract_activations(
    panel: list[str],
    pool: dict[str, str | None],
    probes: list[str],
    acts_dir: Path,
    probe_sha: str,
) -> dict[str, dict[int, np.ndarray]]:
    """Return {persona: {layer: (n_probes, hidden) fp32 np.ndarray}}.

    Checkpoint-per-phase: each persona's activations land on disk the moment
    its forwards finish; a crash mid-extraction resumes by skipping completed
    personas (probe-sha + shape verified on load, fail-loud on mismatch).
    """
    acts_dir.mkdir(parents=True, exist_ok=True)
    pending = [n for n in panel if not (acts_dir / f"{n}.npz").exists()]

    if pending:
        import issue444_persona_distance_topic as pdt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # The distance module reads LAYERS + PERSONA_PROMPTS as module globals
        # at call time (issue444_persona_distance_topic.py:119/:123; same
        # patch pattern as issue541_prescreen.step_0c).
        pdt.LAYERS = list(LAYERS)
        pdt.PERSONA_PROMPTS = dict(pool)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading %s (bf16) on %s for %d personas", BASE_MODEL, device, len(pending))
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.bfloat16, device_map=device
            ).eval()
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
            ).eval()

        for i, name in enumerate(pending):
            t0 = time.time()
            acts = pdt.last_token_acts(model, tok, name, probes, device)
            arrays = {str(li): acts[li].numpy().astype(np.float32) for li in LAYERS}
            for li, arr in arrays.items():
                assert arr.shape == (len(probes), model.config.hidden_size), (li, arr.shape)
            tmp = acts_dir / f"{name}.npz.tmp.npz"
            np.savez(tmp, probe_sha=np.array(probe_sha), **arrays)
            tmp.rename(acts_dir / f"{name}.npz")
            logger.info(
                "extracted %s (%d/%d) in %.1fs", name, i + 1, len(pending), time.time() - t0
            )
        del model
    else:
        logger.info("all %d personas already extracted — resuming from %s", len(panel), acts_dir)

    out: dict[str, dict[int, np.ndarray]] = {}
    for name in panel:
        with np.load(acts_dir / f"{name}.npz") as z:
            assert str(z["probe_sha"]) == probe_sha, (
                f"stale activation cache for {name}: probe sha mismatch "
                f"({z['probe_sha']} != {probe_sha}) — delete {acts_dir} and re-run"
            )
            out[name] = {li: z[str(li)].astype(np.float32) for li in LAYERS}
            for li in LAYERS:
                assert out[name][li].shape[0] == len(probes), (name, li, out[name][li].shape)
    return out


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------
def _cosine_matrices(
    panel: list[str], acts: dict[str, dict[int, np.ndarray]]
) -> dict[str, list[list[float]]]:
    """Pairwise mean-per-probe cosine per layer — the phase0c formula
    (normalize last dim, einsum 'aph,bph->abp', mean over probes)."""
    import torch

    out: dict[str, list[list[float]]] = {}
    for li in LAYERS:
        mats = torch.stack([torch.from_numpy(acts[name][li]) for name in panel])
        assert mats.shape[0] == len(panel), mats.shape
        normed = torch.nn.functional.normalize(mats, dim=-1)
        cos = torch.einsum("aph,bph->abp", normed, normed).mean(dim=-1)
        out[str(li)] = [[float(x) for x in row] for row in cos]
    return out


def _gkl_matrices(
    panel: list[str], acts: dict[str, dict[int, np.ndarray]]
) -> tuple[dict[str, dict[str, list[list[float]]]], dict[str, int]]:
    """Pairwise Gaussian sym-KL per layer per k. Returns (matrices, nan_counts).

    Diagonal is 0.0 by definition (identical clouds); NaNs from ill-conditioned
    covariances are recorded, never silently replaced (the joint script
    hard-asserts no NaN among the 69 lookups it consumes)."""
    out: dict[str, dict[str, list[list[float]]]] = {}
    nan_counts: dict[str, int] = {}
    n = len(panel)
    for li in LAYERS:
        per_k: dict[str, list[list[float]]] = {}
        for k in GKL_KS:
            mat = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    v = _gaussian_sym_kl_in_subspace_local(
                        acts[panel[i]][li], acts[panel[j]][li], k
                    )
                    mat[i, j] = mat[j, i] = v
            per_k[f"k{k}"] = [[float(x) for x in row] for row in mat]
            nan_counts[f"L{li}_k{k}"] = int(np.isnan(mat).sum())
        out[str(li)] = per_k
    total_nan = sum(nan_counts.values())
    if total_nan:
        logger.warning(
            "Gaussian sym-KL produced NaNs: %s", {k: v for k, v in nan_counts.items() if v}
        )
    return out, nan_counts


# ---------------------------------------------------------------------------
# Fidelity gate
# ---------------------------------------------------------------------------
def _fidelity_check(
    panel: list[str],
    cosine: dict[str, list[list[float]]],
    pool: dict[str, str | None],
    probe_sha: str,
    out_dir: Path,
    *,
    binding: bool,
) -> dict[str, Any]:
    """Per-teacher-row Spearman >= 0.99 vs the committed phase0c matrix at
    layers {7,14,21,27}; HARD assert when binding (full run). On failure,
    writes a diagnostic dump and raises (plan §3.2 (b): never proceed on
    divergent geometry)."""
    committed = json.loads(PHASE0C_PATH.read_text())
    c_names: list[str] = committed["personas"]
    c_idx = {n: i for i, n in enumerate(c_names)}
    pred = json.loads(PREDICTORS_PATH.read_text())
    teachers = [t for t in pred["sources"] if t in panel]

    common = [n for n in panel if n in c_idx]
    record: dict[str, Any] = {
        "reference": str(PHASE0C_PATH.relative_to(PROJECT_ROOT)),
        "threshold_row_spearman": FIDELITY_MIN_ROW_SPEARMAN,
        "binding": binding,
        "n_common_personas": len(common),
        "layers": {},
    }
    failures: list[str] = []
    for li in FIDELITY_LAYERS:
        c_mat = committed["cosine_matrix"][str(li)]
        r_mat = cosine[str(li)]
        rows: dict[str, dict[str, float]] = {}
        deltas: list[float] = []
        for t in teachers:
            others = [n for n in common if n != t]
            if len(others) < 3:
                rows[t] = {"row_spearman": float("nan"), "n": len(others)}
                continue
            c_row = [c_mat[c_idx[t]][c_idx[o]] for o in others]
            r_row = [r_mat[panel.index(t)][panel.index(o)] for o in others]
            rho = _spearman(c_row, r_row)
            rows[t] = {
                "row_spearman": rho,
                "n": len(others),
                "max_abs_delta": float(max(abs(a - b) for a, b in zip(c_row, r_row, strict=True))),
            }
            if not (rho >= FIDELITY_MIN_ROW_SPEARMAN):
                failures.append(f"L{li}/{t}: row Spearman {rho:.4f} < {FIDELITY_MIN_ROW_SPEARMAN}")
        for a in common:
            for b in common:
                deltas.append(
                    abs(c_mat[c_idx[a]][c_idx[b]] - r_mat[panel.index(a)][panel.index(b)])
                )
        record["layers"][str(li)] = {
            "per_teacher_row": rows,
            "max_abs_delta_full_submatrix": float(max(deltas)) if deltas else float("nan"),
        }

    record["passed"] = not failures
    record["failures"] = failures
    if failures and binding:
        dump = {
            "failures": failures,
            "record": record,
            "probe_sha256": probe_sha,
            "persona_prompt_sha256": {
                n: hashlib.sha256((pool[n] or "<none>").encode()).hexdigest() for n in panel
            },
            "timestamp": _now_iso(),
        }
        dump_path = out_dir / "fidelity_failure_dump.json"
        _write_json(dump_path, dump)
        raise AssertionError(
            f"FIDELITY GATE FAILED ({len(failures)} rows below "
            f"{FIDELITY_MIN_ROW_SPEARMAN}); diagnostic dump -> {dump_path}; "
            "do NOT proceed on divergent geometry (plan §3.2 (b))."
        )
    if failures:
        logger.warning("fidelity gate non-binding (smoke) — would have failed: %s", failures)
    return record


# ---------------------------------------------------------------------------
# Best-effort HF upload (fail-soft on quota 403 — plan §8 risk row)
# ---------------------------------------------------------------------------
def _best_effort_upload(files: list[Path], smoke: bool, workdir: Path) -> dict[str, Any]:
    if smoke:
        return {"status": "skipped_smoke"}
    summary: dict[str, Any] = {"status": "ok", "uploaded": [], "failed": {}}
    try:
        from huggingface_hub import HfApi, list_repo_files
        from issue541_upload_lib import upload_text_file

        api = HfApi()
        existing = set(list_repo_files(DATA_REPO, repo_type="dataset"))
        for fp in files:
            try:
                res = upload_text_file(
                    api,
                    local_path=fp,
                    path_in_repo=f"{HF_UPLOAD_PREFIX}/{fp.name}",
                    repo_id=DATA_REPO,
                    existing=existing,
                    workdir=workdir,
                )
                summary["uploaded"].extend(res["uploaded"] + res["skipped"])
            except Exception as exc:  # fail-soft per plan §8: JSON in git is durable
                logger.warning("HF upload FAILED for %s: %s (fail-soft, continuing)", fp.name, exc)
                summary["failed"][fp.name] = repr(exc)
    except Exception as exc:
        logger.warning("HF upload step unavailable: %s (fail-soft, continuing)", exc)
        summary = {"status": "unavailable", "error": repr(exc)}
    if summary.get("failed"):
        summary["status"] = "partial"
    return summary


# ---------------------------------------------------------------------------
# Sentinel (poll_pipeline.py contract)
# ---------------------------------------------------------------------------
def _write_sentinel(sentinel_dir: Path | None, note: dict[str, Any]) -> Path | None:
    if sentinel_dir is None:
        sentinel_dir = Path("/workspace/logs") if Path("/workspace").exists() else None
    if sentinel_dir is None:
        logger.info("no /workspace and no --sentinel-dir — sentinel skipped (VM run)")
        return None
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"issue-541-epm_progress-{int(time.time())}.json"
    obj = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": 541,
        "by": "issue541_geometry_extract",
        "ts": _now_iso(),
        "gate": "",
        "blocks_pipeline": False,
        "note": json.dumps(note, indent=2, default=str),
    }
    _write_json(path, obj)
    logger.info("sentinel written -> %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="2 personas x 8 probes, smoke namespace")
    ap.add_argument("--gpu-id", type=int, default=None, help="pin CUDA_VISIBLE_DEVICES")
    ap.add_argument("--skip-upload", action="store_true", help="skip the best-effort HF upload")
    ap.add_argument(
        "--sentinel-dir",
        type=Path,
        default=None,
        help="override sentinel dir (default /workspace/logs when on a pod; skipped on the VM)",
    )
    args = ap.parse_args()

    if args.gpu_id is not None:
        import os

        # Before any torch import (the CVD-after-CUDA-init clobber is a no-op).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    _log_phase("setup")
    eval_root = "issue_541_smoke" if args.smoke else "issue_541"
    out_dir = PROJECT_ROOT / "eval_results" / eval_root / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(args.smoke)
    pool = _panel_pool(panel)
    n_probes = N_PROBES_SMOKE if args.smoke else N_PROBES_FULL
    probes = _build_probes(n_probes)
    probe_sha = _probe_sha256(probes)
    logger.info(
        "panel=%d personas, %d probes (sha %s), layers=%s",
        len(panel),
        len(probes),
        probe_sha[:12],
        LAYERS,
    )

    _log_phase("extract")
    t0 = time.time()
    acts = _extract_activations(panel, pool, probes, out_dir / "activations_fp32", probe_sha)
    extract_s = time.time() - t0

    _log_phase("cosine")
    cosine = _cosine_matrices(panel, acts)

    _log_phase("fidelity")
    fidelity = _fidelity_check(panel, cosine, pool, probe_sha, out_dir, binding=not args.smoke)

    _log_phase("gkl")
    t0 = time.time()
    gkl, gkl_nan_counts = _gkl_matrices(panel, acts)
    gkl_s = time.time() - t0

    _log_phase("write")
    # fp16 storage artifact (plan §3.2 (e)); overflow-guarded — Qwen residual
    # streams can carry large-magnitude dims, and a silent inf would corrupt
    # the archive. Matrices above were computed from fp32, never from these.
    npy_files: list[Path] = []
    for li in LAYERS:
        stack32 = np.stack([acts[name][li] for name in panel])  # (P, n_probes, hidden) fp32
        stack16 = stack32.astype(np.float16)
        suffix = "fp16"
        if not np.isfinite(stack16).all():
            logger.warning("layer %d overflows fp16 — storing fp32 instead", li)
            stack16, suffix = stack32, "fp32"
        fp = out_dir / f"activations_L{li}_{suffix}.npy"
        np.save(fp, stack16)
        npy_files.append(fp)

    matrices_path = out_dir / "geometry_matrices.json"
    payload: dict[str, Any] = {
        "_doc": (
            "Issue #541 follow-up geometry-plus-prior-joint-predictor: pairwise "
            "mean-per-probe cosine (phase0c formula) + Gaussian sym-KL (PCA "
            "subspace, Gram trick; #502/#532 lineage) over last-input-token "
            "base-model activations, 24-panel personas x A-family probes. "
            "cos/gkl-to-any-reference is a row lookup. gkl is a DISTANCE "
            "(higher = farther); cosine is a similarity."
        ),
        "model": BASE_MODEL,
        "personas": panel,
        "layers": list(LAYERS),
        "n_probes": len(probes),
        "smoke": args.smoke,
        "probe_sha256": probe_sha,
        "cosine_matrix": cosine,
        "gauss_kl_matrix": gkl,
        "gauss_kl_nan_counts": gkl_nan_counts,
        "fidelity_check": fidelity,
        "wall_seconds": {"extract": round(extract_s, 1), "gkl": round(gkl_s, 1)},
        "activation_files": [f.name for f in npy_files],
        "timestamp": _now_iso(),
        "reproducibility": _repro_metadata(),
    }
    _write_json(matrices_path, payload)
    logger.info("WROTE %s (%d personas x %d probes)", matrices_path, len(panel), len(probes))

    _log_phase("upload")
    upload_summary = (
        {"status": "skipped_flag"}
        if args.skip_upload
        else _best_effort_upload(npy_files, args.smoke, out_dir)
    )
    _write_json(out_dir / "upload_summary_geometry.json", upload_summary)

    _log_phase("results_sentinel")
    _write_sentinel(
        args.sentinel_dir,
        {
            "step": "geometry-extraction",
            "matrices_json": str(matrices_path),
            "fidelity_passed": fidelity["passed"],
            "gkl_nan_counts": {k: v for k, v in gkl_nan_counts.items() if v},
            "upload": upload_summary.get("status"),
            "git_commit": payload["reproducibility"]["git_commit"],
            "next": "run scripts/issue541_geometry_joint.py (CPU, VM ok)",
        },
    )
    _log_phase("done")


if __name__ == "__main__":
    main()
