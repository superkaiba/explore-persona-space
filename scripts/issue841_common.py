#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, → , ‖·‖) in scientific docstrings/log messages.
"""Shared loaders + reuse-fitness asserts for issue #841.

Everything the two staged entrypoints (``issue841_stage0_atlas.py`` /
``issue841_stage1_benchmark.py``) + the plots script share:

- HF fetch of #779's cached tensors, PINNED to the #779 Repro revision
  ``037fcbb210bc52c459959b0746cc268fe08bae96`` on
  ``superkaiba1/explore-persona-space-data`` (repo_type dataset). Downloads
  cache under the worktree ``data/issue_841/hf_dl/`` (re-downloadable; cleaned at
  ``/issue`` Step 8). Loaders cast everything to fp32.
- The reuse-fitness shape asserts (artifact-reuse.md (a)-(h)): ``cx_last`` shape
  ``(N,28,3584)``, ``layers == range(28)``, ``r_b`` shape ``(28,3584)`` — all
  fail-loud at load.
- The per-(condition, question) eval-trajectory matrix, a faithful replica of
  #779's ``issue779_stage1.build_eval_matrix`` aggregation (drop empty rollouts,
  drop questions with no valid judge score, y = MEAN of valid rollout scores,
  condition index in first-seen order, elicitation mode) but carrying the FULL
  28-layer trajectory per question so Stage 1 can project/transport at any layer.
- Target-layer constants + reproducibility metadata.

No Qwen weights are loaded; no new judging. All inputs are #779's cached
artifacts (reuse-fitness verified in the plan §4.0).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# load_dotenv (sets HF_HOME + shared-VM thread caps) MUST run before torch import
# so torch freezes its thread pool from the capped OMP_NUM_THREADS (code-style.md
# § Shared-VM CPU thread caps).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

logger = logging.getLogger("issue841")

# ── constants ─────────────────────────────────────────────────────────────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# The #779 Repro-pinned revision (artifact-reuse check (f): pin every reused
# artifact so the HF mirror cannot silently diverge from what the plan verified).
HF_REVISION = "037fcbb210bc52c459959b0746cc268fe08bae96"
HF_PREFIX = "issue779_monitoring"
HF_DL_DIR = PROJECT_ROOT / "data" / "issue_841" / "hf_dl"

TRAITS = ("evil", "sycophancy", "hallucination")
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
N_TRANSITIONS = EXPECTED_LAYERS - 1  # 27 one-step transitions ℓ=0..26

# Target read-out layers ℓ* (plan §4.4, fact-checked from step0_oracle.json).
# PRIMARY = the body's a-priori set (MIXES schemes: evil = PV-raw peak;
# syco/halluc = #779 oracle-best). COMPANION = the other scheme per trait.
PRIMARY_TARGET_LAYER = {"evil": 20, "sycophancy": 26, "hallucination": 17}
COMPANION_TARGET_LAYER = {"evil": 14, "sycophancy": 19, "hallucination": 24}
# #779's OWN read-out layers, where its reference rows (raw-PV, h, g) were
# computed — the reference-row self-check runs here (plan assumption 15).
REFERENCE_ROW_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}

# Reference-row self-check tolerance: my reproduced raw-PV row must land within
# this band of #779's pv_raw point (mirrors #779's own gate-1 rig-validation band).
RIG_VALIDATION_BAND = 0.10

# Split (plan §10): 4000 fit / 500 inner-val (MLP+GRU early stop) / 500 test.
SPLIT_SEED = 42
N_FIT = 4000
N_INNER_VAL = 500
N_TEST = 500
# Data-scaling curve nested subsamples (plan §4.3).
SCALING_NS = (500, 1000, 2000, 4000)

# Seeds (plan §10).
BOOTSTRAP_SEED = 0
MLP_INIT_SEED = 658


# ── HF fetch (pinned revision, worktree cache) ────────────────────────────────


def _fetch(rel_path: str) -> Path:
    """hf_hub_download a file under HF_PREFIX at the pinned revision.

    ``rel_path`` is relative to ``HF_PREFIX`` (e.g. ``analysis_tensors/pass_b/
    train_context_vectors.pt``). Returns the local cache path. Fail-loud: a
    missing file / gated 403 / bad revision raises rather than returning a
    silent default.
    """
    HF_DL_DIR.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_PREFIX}/{rel_path}",
            repo_type="dataset",
            revision=HF_REVISION,
            cache_dir=str(HF_DL_DIR),
        )
    )


def list_pass_a_cells(trait: str) -> list[str]:
    """The pass_a cell ids for a trait (e.g. ``evil__sys0`` .. ``evil__shot20``).

    Enumerated from the live repo listing at the pinned revision (39 total: 8
    system + 5 many-shot per trait × 3 traits), so the loader never hardcodes the
    condition naming convention.
    """
    files = list_repo_files(HF_DATA_REPO, repo_type="dataset", revision=HF_REVISION)
    prefix = f"{HF_PREFIX}/analysis_tensors/pass_a/{trait}__"
    cells = sorted(
        f.split("pass_a/")[1][: -len("_cx.pt")]
        for f in files
        if f.startswith(prefix) and f.endswith("_cx.pt")
    )
    assert cells, f"no pass_a cells found for trait {trait!r} at revision {HF_REVISION}"
    return cells


# ── loaders (fp32, fail-loud shape asserts) ───────────────────────────────────


def load_pass_b() -> dict:
    """Load the train-context depth trajectory bundle (pass_b), cast fp32.

    Returns ``{"cx_last": (N,28,3584) fp32, "cx_mean": (N,28,3584) fp32,
    "prompts": [...], "layers": [0..27], "n": N}``. Asserts the reuse-fitness
    contract (artifact-reuse (a)/(f)): shape ``(N,28,3584)`` + ``layers ==
    list(range(28))``.
    """
    path = _fetch("analysis_tensors/pass_b/train_context_vectors.pt")
    blob = torch.load(path, weights_only=False)
    layers = list(blob["layers"])
    assert layers == list(range(EXPECTED_LAYERS)), f"pass_b layers != range(28): {layers}"
    cx_last = blob["cx_last"].to(torch.float32)
    cx_mean = blob["cx_mean"].to(torch.float32)
    n = cx_last.shape[0]
    assert cx_last.shape == (n, EXPECTED_LAYERS, EXPECTED_HIDDEN), cx_last.shape
    assert cx_mean.shape == (n, EXPECTED_LAYERS, EXPECTED_HIDDEN), cx_mean.shape
    logger.info("[load] pass_b cx_last %s (source=%s)", tuple(cx_last.shape), blob.get("source"))
    return {
        "cx_last": cx_last.numpy(),
        "cx_mean": cx_mean.numpy(),
        "prompts": blob.get("prompts"),
        "layers": layers,
        "n": int(n),
    }


def load_rb(trait: str) -> np.ndarray:
    """Load the persona direction r_B for a trait, cast fp32 → (28,3584) ndarray."""
    assert trait in TRAITS, f"unknown trait {trait!r}"
    path = _fetch(f"r_b/{trait}.pt")
    blob = torch.load(path, weights_only=False)
    r_b = blob["r_b"].to(torch.float32).numpy()
    assert r_b.shape == (EXPECTED_LAYERS, EXPECTED_HIDDEN), (trait, r_b.shape)
    return r_b


def load_step0() -> dict:
    """Load step0_oracle.json (per-layer PV/oracle within-condition r)."""
    with open(_fetch("analysis_tensors/step0/step0_oracle.json")) as f:
        return json.load(f)


def load_eval_cells(trait: str) -> list[dict]:
    """Load a trait's pass_a cells (JSON scalars + c_x trajectory tensors), fp32.

    Each cell carries ``judge_scores`` / ``oracle_proj`` / ``rollouts`` / ``mode``
    / ``cond_id`` (JSON) + ``_cx_last`` (n_q,28,3584) fp32 (the eval-context
    last-prompt-token trajectory Stage 1 transports). ``_cx_mean`` is loaded too
    for completeness. Sorted by cell id (system first, then many-shot). Asserts
    ``layers == range(28)`` per cell.
    """
    cells = []
    for cell_id in list_pass_a_cells(trait):
        json_path = _fetch(f"analysis_tensors/pass_a/{cell_id}.json")
        cx_path = _fetch(f"analysis_tensors/pass_a/{cell_id}_cx.pt")
        with open(json_path) as f:
            cell = json.load(f)
        cx = torch.load(cx_path, weights_only=True)
        layers = list(cx["layers"])
        assert layers == list(range(EXPECTED_LAYERS)), f"{cell_id} layers != range(28): {layers}"
        cx_last = cx["cx_last"].to(torch.float32).numpy()  # (n_q, 28, H)
        assert cx_last.shape[1:] == (EXPECTED_LAYERS, EXPECTED_HIDDEN), (cell_id, cx_last.shape)
        cell["_cx_last"] = cx_last
        cell["_cx_mean"] = cx["cx_mean"].to(torch.float32).numpy()
        cell["_layers"] = layers
        cells.append(cell)
    return cells


# ── per-(condition, question) eval-trajectory matrix ──────────────────────────


def _score_for(cell: dict, qi: int, ri: int) -> float | None:
    """Resolve a rollout's judge score from the cell's {custom_id: score} map.

    Verbatim from ``issue779_stage1._score_for`` so the per-question aggregation
    below matches #779's exactly (the reference-row self-check depends on it).
    """
    for cid, s in cell["judge_scores"].items():
        parts = cid.split("__")
        if len(parts) < 3:
            continue
        try:
            idx, ci = int(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        if idx == qi and ci == ri:
            return s
    return None


def build_eval_traj_matrix(cells: list[dict]) -> dict:
    """Per-(condition, question) trajectory matrix (faithful #779 aggregation).

    Mirrors ``issue779_stage1.build_eval_matrix`` exactly — group each cell's
    rollouts by question index (drop ``empty`` rollouts), drop questions with no
    valid judge score, ``y`` = MEAN of the question's valid rollout scores,
    ``cond`` = condition index in first-seen order, ``mode`` = the cell's
    elicitation mode — but carries the FULL 28-layer trajectory per question
    (``traj`` (N_q, 28, H)) instead of a single-layer projection, so Stage 1 can
    read/transport at any layer while the (cond, question) unit + y + prune are
    #779-identical.

    Returns ``{"traj": (N_q,28,H), "y": (N_q,), "cond": (N_q,) int,
    "mode": (N_q,) object, "cond_ids": [str], "layers": [0..27]}``.
    """
    layers = cells[0]["_layers"]
    traj, y, cond, mode = [], [], [], []
    cond_map: dict[str, int] = {}
    for cell in cells:
        cid = cell["cond_id"]
        cond_map.setdefault(cid, len(cond_map))
        by_q: dict[int, list[dict]] = {}
        for rec in cell["rollouts"]:
            if rec.get("empty"):
                continue
            by_q.setdefault(rec["qi"], []).append(rec)
        for qi, recs in by_q.items():
            q_scores = [s for r in recs if (s := _score_for(cell, qi, r["ri"])) is not None]
            if not q_scores:
                continue
            traj.append(cell["_cx_last"][qi, :, :])  # (28, H)
            y.append(float(np.mean(q_scores)))
            cond.append(cond_map[cid])
            mode.append(cell["mode"])
    return {
        "traj": np.array(traj, dtype=np.float32),  # (N_q, 28, H)
        "y": np.array(y, dtype=np.float64),
        "cond": np.array(cond, dtype=int),
        "mode": np.array(mode, dtype=object),
        "cond_ids": list(cond_map.keys()),
        "layers": layers,
    }


def group_by_condition(
    x: np.ndarray, y: np.ndarray, cond: np.ndarray, mode: np.ndarray, which_mode: str
) -> tuple[list, list]:
    """Split (x, y) into per-condition arrays for one elicitation mode.

    Verbatim from ``issue779_stage1._group_by_condition`` so the within-condition
    grouping fed to #779's ``within_condition_pearson`` is byte-identical.
    """
    cx, cy = [], []
    sel = np.array([m == which_mode for m in mode])
    if not sel.any():
        return cx, cy
    for c in np.unique(cond[sel]):
        m = sel & (cond == c)
        cx.append(x[m])
        cy.append(y[m])
    return cx, cy


# ── split ─────────────────────────────────────────────────────────────────────


def make_split(n: int, *, n_fit: int, n_val: int, n_test: int, seed: int = SPLIT_SEED) -> dict:
    """Deterministic 3-way split of the N contexts into fit / inner-val / test.

    The test set is NEVER used in any fit, λ-selection, or early-stopping
    decision (plan §4.3). Returns ``{"fit": idx, "val": idx, "test": idx}`` int
    arrays. Clamps to N when n < requested total (smoke). Fails loud if N is too
    small to carve any test set.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    need = n_fit + n_val + n_test
    if n < need:
        # Scale down proportionally (smoke / small-N), preserving fit>val>=test>=1.
        frac_fit, frac_val = n_fit / need, n_val / need
        nf = max(1, round(n * frac_fit))
        nv = max(1, round(n * frac_val))
        nt = n - nf - nv
        assert nt >= 1, f"N={n} too small to carve a test set (fit={nf}, val={nv})"
        n_fit, n_val, n_test = nf, nv, nt
    return {
        "fit": perm[:n_fit],
        "val": perm[n_fit : n_fit + n_val],
        "test": perm[n_fit + n_val : n_fit + n_val + n_test],
    }


# ── reproducibility ────────────────────────────────────────────────────────────


def reproducibility_metadata(extra: dict | None = None) -> dict:
    """Git commit + env versions + timestamp for result JSONs (CLAUDE.md)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    now = _dt.datetime.now(_dt.UTC).replace(tzinfo=None)
    meta = {
        "git_commit": sha,
        "timestamp_utc": now.isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "hf_revision": HF_REVISION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    if extra:
        meta.update(extra)
    return meta


def write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic-ish JSON write (tmp + rename) for checkpoint-per-phase safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)
