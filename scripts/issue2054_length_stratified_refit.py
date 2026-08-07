#!/usr/bin/env python
"""Length-stratified refit companion for task #2054 (plan req 10; Step 9a-ter).

The committed answer-length parity companion
(`eval_results/issue_2054/analyzer_companions/answer_length_parity.json`) shows
the 24 (b)-inserted vs (d)-on-policy pairs differ strongly in answer token
length (KS D up to ~0.53, mean ratios ~0.27). This driver asks whether the
inserted-vs-on-policy CONTEXT-arm ceiling gaps survive on length-matched
conversation populations:

1. Per pair (``<variant>__<form>__<model>``), compute per-conversation answer
   token lengths for BOTH cells with the parity companion's exact recipe
   (``Qwen/Qwen2.5-7B`` tokenizer, ``answer`` field, ``add_special_tokens=False``
   — bit-reproduced against the committed companion stats for
   ``char_dana__inserted`` and ``char_dana__on_policy__attrib_quoted__qwen2.5-7b``).
2. Build length-matched conversation sets: decile bins of the POOLED pair
   distribution; per bin take ``min(n_b, n_d)`` conversations from each cell
   (stratified subsample, ``numpy default_rng([seed, pair_idx])``, ids sorted
   before the draw so the sets are order-stable).
3. Refit the CONTEXT arm of both cells on the matched sets through the SHARED
   fold map via the REUSED fit core ``issue2054_fits._fit_arm_cell``
   (restricted conversation sets — the same mechanism the committed companion
   refits used). CONTEXT ARM ONLY is the dispatch note's stated deviation from
   the both-arms rule (prefix arm <= 0.021 in all committed cells).
4. ESTIMATOR VALIDITY: per-fold n_train is computed BEFORE fitting; when any
   fold of either cell drops below d=3,584 the pair's matched-vs-full
   comparison is drawn in the reduced-basis k=1024 read (computed for BOTH
   cells; the dof-capped GCV cores guard the ambient numbers either way) —
   never across bases. The realized basis is recorded per pair.
5. Checkpoint per cell (JSON under ``--checkpoint-dir``) with a regime-keyed
   resume predicate; the FIRST cell is the pilot (measured wall logged before
   the cell-parallel fan-out at ``--workers``).

Full (unmatched) ceilings come from the committed production fit JSONs
(``data/issue_2054/fits/<cell_key>.json``; HF fallback
``issue2054_lattice/fits/``) — the same numbers the headline used.

Outputs: ``--out-json`` (per pair: full/matched ceilings in both bases,
realized matched n per cell + per fold, basis, per-bin counts, gap deltas) and
``--fig`` (matched vs full per pair, per-unit points labeled).

Emits ``[lenstrat]`` progress lines; terminates with
``[lenstrat] DONE rc=<rc> cells=<ok>/<total>``. Exit 0 on success, 1 on any
cell/aggregation failure, 2 on missing input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue2054_fits as fits  # noqa: E402  (REUSED fit core: _fit_arm_cell)
import issue2054_forms as forms  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
TASK_PREFIX = "issue2054_lattice"
TOKENIZER_ID = "Qwen/Qwen2.5-7B"
D_AMBIENT = fits.D_AMBIENT  # 3584 — the ambient-basis estimator floor
REDUCED_BASIS_K = fits.REDUCED_BASIS_K  # 1024
DEFAULT_COMPANION = (
    _REPO_ROOT / "eval_results/issue_2054/analyzer_companions/answer_length_parity.json"
)
DEFAULT_FOLD_MAP = _REPO_ROOT / "eval_results/issue_2054/shared_fold_map.json"
DEFAULT_FULL_FITS_DIR = _REPO_ROOT / "data/issue_2054/fits"
DEFAULT_CKPT_DIR = _REPO_ROOT / "data/issue_2054/lenstrat/checkpoints"
# Matched-n floor below which a cell's fit is skipped (recorded) — reduced
# basis handles n < d, but a few-hundred-row ridge read answers nothing.
MATCHED_N_FLOOR = 500

_TOKENIZER = None  # module-level cache (never from_pretrained per cell — gotchas.md)
_LENGTHS_CACHE: dict[str, dict[str, int]] = {}  # answers-file path -> conv_id -> token len
_WORKER_FOLD_MAP: dict[str, dict] = {}  # per-process fold-map cache (fork-safe)


def _log(msg: str) -> None:
    print(f"[lenstrat] {msg}", flush=True)


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return _TOKENIZER


# ─────────────────────────────────────────────────────────────────────────────
# Pair enumeration (source of truth: the committed parity companion's 24 pairs)


def _pairs_from_companion(companion_path: Path) -> list[dict]:
    """Parse the parity companion's pair keys (``<variant>__<form>__<model>``)
    into sorted pair dicts. The companion is the committed enumeration of the
    24 (b)-inserted vs (d)-on-policy pairs (16 character + 8 assistant)."""
    with companion_path.open(encoding="utf-8") as f:
        companion = json.load(f)
    pairs = []
    for key in sorted(companion["pairs"]):
        parts = key.split(forms.CELL_KEY_SEP)
        if len(parts) != 3:
            raise ValueError(f"unparseable pair key {key!r} in {companion_path}")
        variant, form, model = parts
        pairs.append({"pair_key": key, "variant": variant, "form": form, "model": model})
    return pairs


def _pair_cells(pair: dict) -> dict[str, str]:
    """The pair's two cell keys: (b) inserted and (d) on_policy."""
    return {
        cond: forms.cell_key(pair["variant"], cond, pair["form"], pair["model"])
        for cond in ("inserted", "on_policy")
    }


# ─────────────────────────────────────────────────────────────────────────────
# Answer-length loading (the parity companion's exact recipe)


def _answers_paths(variant: str, condition: str, form: str, model: str) -> tuple[Path, str]:
    """(local path, HF path_in_repo) of the answer-text jsonl for a cell."""
    if condition == "inserted":
        rel = f"spliced_inserted/{variant}/spliced_inserted_{variant}__{form}.jsonl"
        return _REPO_ROOT / "data/issue_2054" / rel, f"{TASK_PREFIX}/{rel}"
    rel = f"on_policy/{model}/{variant}/on_policy_{variant}__{form}.jsonl"
    return _REPO_ROOT / "data/issue_2054/hf_dl" / TASK_PREFIX / rel, f"{TASK_PREFIX}/{rel}"


def _answer_lengths(variant: str, condition: str, form: str, model: str) -> dict[str, int]:
    """conv_id -> answer token length (Qwen2.5-7B tokenizer, answer field,
    add_special_tokens=False — bit-reproduces the parity companion's stats).
    Local-first; scoped HF fetch fallback (never a full-tree listing)."""
    local, hf_path = _answers_paths(variant, condition, form, model)
    cache_key = str(local)
    if cache_key in _LENGTHS_CACHE:
        return _LENGTHS_CACHE[cache_key]
    if not local.is_file():
        _log(f"answers file missing locally; staging from HF: {hf_path}")
        hub.stage_hub_file(HF_DATA_REPO, hf_path, local, repo_type="dataset")
    ids: list[str] = []
    texts: list[str] = []
    for line in local.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(str(row["conv_id"]))
        texts.append(row["answer"])
    if not ids:
        raise ValueError(f"no answer rows in {local}")
    enc = _get_tokenizer()(texts, add_special_tokens=False)["input_ids"]
    lengths = {cid: len(tok_ids) for cid, tok_ids in zip(ids, enc)}
    _LENGTHS_CACHE[cache_key] = lengths
    return lengths


# ─────────────────────────────────────────────────────────────────────────────
# Activation staging (local-first, per-file scoped HF fetch; 1-file loader probe)


def _npz_target(activations_dir: Path, variant: str, cell_key: str) -> Path:
    return activations_dir / variant / f"{cell_key}.npz"


def stage_activations(activations_dir: Path, pairs: list[dict]) -> list[Path]:
    """Ensure every pair-cell .npz is present under ``activations_dir``
    (capture layout ``<dir>/<variant>/<cell_key>.npz``). Local hf_dl mirror
    copies win over network; otherwise per-file ``hub.stage_hub_file`` against
    the scoped HF path (never a repo-wide listing). The FIRST present file is
    opened with the SAME loader ``_fit_arm_cell`` consumes
    (``fits._load_activation_npz``) as the staging probe BEFORE bulk work."""
    local_mirror = _REPO_ROOT / "data/issue_2054/hf_dl" / TASK_PREFIX / "activations"
    needed: list[tuple[str, str]] = []  # (variant, cell_key)
    for pair in pairs:
        for cell_key in _pair_cells(pair).values():
            needed.append((pair["variant"], cell_key))
    staged: list[Path] = []
    probed = False
    for i, (variant, cell_key) in enumerate(needed):
        target = _npz_target(activations_dir, variant, cell_key)
        if target.is_file() and target.stat().st_size > 0:
            pass
        else:
            mirror = local_mirror / variant / f"{cell_key}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            if mirror.is_file() and mirror.stat().st_size > 0:
                tmp = target.with_suffix(".npz.staging.tmp")
                shutil.copy2(mirror, tmp)
                os.replace(tmp, target)
                _log(f"staged {cell_key} from local hf_dl mirror")
            else:
                t0 = time.time()
                hub.stage_hub_file(
                    HF_DATA_REPO,
                    f"{TASK_PREFIX}/activations/{variant}/{cell_key}.npz",
                    target,
                    repo_type="dataset",
                )
                _log(
                    f"staged {cell_key} from HF "
                    f"({target.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)"
                )
        staged.append(target)
        if not probed:
            # 1-file staging probe: open with the consumer's own loader.
            acts = fits._load_activation_npz(target)
            if acts is None:
                raise ValueError(f"staging probe: empty .npz {target}")
            n = len(acts["conv_ids"])
            d = acts["v_C"].shape[1]
            _log(f"staging probe OK: {cell_key} rows={n} d={d} (loader=_load_activation_npz)")
            probed = True
        if (i + 1) % 8 == 0 or (i + 1) == len(needed):
            _log(f"staging progress {i + 1}/{len(needed)}")
    return staged


# ─────────────────────────────────────────────────────────────────────────────
# Matched-set construction (decile bins of the pooled pair distribution)


def _npz_conv_ids(path: Path) -> list[str]:
    z = np.load(path, allow_pickle=False)
    return [str(x) for x in z["conv_id"]]


def build_matched_sets(
    pair: dict,
    pair_idx: int,
    activations_dir: Path,
    fold_of: dict,
    *,
    n_bins: int,
    seed: int,
    smoke_max_n: int | None = None,
) -> dict:
    """Per pair: universe per cell = npz conv_ids ∩ fold map ∩ has-length;
    pooled decile bin edges; per bin ``min(n_b, n_d)`` drawn from each cell
    (sorted ids, ``default_rng([seed, pair_idx])``). Returns the manifest dict
    (matched ids per cell + per-bin counts + per-fold n_train + basis)."""
    cells = _pair_cells(pair)
    universe: dict[str, list[str]] = {}
    lens: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict] = {}
    for cond, cell_key in cells.items():
        npz = _npz_target(activations_dir, pair["variant"], cell_key)
        conv_ids = _npz_conv_ids(npz)
        cell_lens = _answer_lengths(pair["variant"], cond, pair["form"], pair["model"])
        in_fold = [cid for cid in conv_ids if cid in fold_of]
        with_len = [cid for cid in in_fold if cid in cell_lens]
        n_missing_len = len(in_fold) - len(with_len)
        if n_missing_len > 0.005 * max(1, len(in_fold)):
            raise ValueError(
                f"answer-length coverage too low for {cell_key}: "
                f"{n_missing_len}/{len(in_fold)} universe rows lack a length"
            )
        universe[cond] = with_len
        lens[cond] = cell_lens
        coverage[cond] = {
            "n_npz": len(conv_ids),
            "n_in_fold_map": len(in_fold),
            "n_with_length": len(with_len),
            "n_missing_length_dropped": n_missing_len,
        }

    pooled = np.array(
        [lens["inserted"][c] for c in universe["inserted"]]
        + [lens["on_policy"][c] for c in universe["on_policy"]],
        dtype=np.float64,
    )
    interior = np.quantile(pooled, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    edges = np.unique(interior)  # heavy ties can collapse edges; realized bins recorded

    rng = np.random.default_rng([seed, pair_idx])
    matched: dict[str, list[str]] = {"inserted": [], "on_policy": []}
    per_bin: list[dict] = []
    ids_by_bin: dict[str, list[list[str]]] = {}
    for cond in ("inserted", "on_policy"):
        bins: list[list[str]] = [[] for _ in range(len(edges) + 1)]
        for cid in sorted(universe[cond]):
            b = int(np.searchsorted(edges, lens[cond][cid], side="right"))
            bins[b].append(cid)
        ids_by_bin[cond] = bins
    # Smoke SCALE dial: cap the matched total per cell by scaling each bin's
    # take proportionally (both cells keep the SAME per-bin count, so the
    # length-matching property is preserved at reduced n). Regime-keyed —
    # production checkpoints never resume a smoke-capped fit.
    matched_total_uncapped = sum(
        min(len(ids_by_bin["inserted"][b]), len(ids_by_bin["on_policy"][b]))
        for b in range(len(edges) + 1)
    )
    scale = 1.0
    if smoke_max_n is not None and matched_total_uncapped > smoke_max_n:
        scale = smoke_max_n / matched_total_uncapped
    for b in range(len(edges) + 1):
        ids_b = ids_by_bin["inserted"][b]
        ids_d = ids_by_bin["on_policy"][b]
        n_take = min(len(ids_b), len(ids_d))
        if scale < 1.0 and n_take:
            n_take = max(1, int(np.floor(n_take * scale)))
        take_b = list(rng.permutation(ids_b)[:n_take]) if n_take else []
        take_d = list(rng.permutation(ids_d)[:n_take]) if n_take else []
        matched["inserted"].extend(take_b)
        matched["on_policy"].extend(take_d)
        lo = float(edges[b - 1]) if b > 0 else float(pooled.min())
        hi = float(edges[b]) if b < len(edges) else float(pooled.max())
        per_bin.append(
            {"bin": b, "lo": lo, "hi": hi, "n_b": len(ids_b), "n_d": len(ids_d), "n_taken": n_take}
        )

    k = int(max(fold_of.values())) + 1
    per_fold_n_train: dict[str, list[int]] = {}
    for cond in ("inserted", "on_policy"):
        sizes = [0] * k
        for cid in matched[cond]:
            sizes[int(fold_of[cid])] += 1
        total = sum(sizes)
        per_fold_n_train[cond] = [total - s for s in sizes]

    min_n_train = min(min(v) for v in per_fold_n_train.values())
    basis = "ambient" if min_n_train >= D_AMBIENT else "reduced_k1024"
    matched_sha = {
        cond: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        for cond, ids in matched.items()
    }
    return {
        "pair_key": pair["pair_key"],
        "cells": cells,
        "coverage": coverage,
        "bin_edges_interior": [float(e) for e in edges],
        "n_bins_realized": len(edges) + 1,
        "per_bin": per_bin,
        "matched_ids": matched,
        "matched_n": {cond: len(ids) for cond, ids in matched.items()},
        "matched_sha256": matched_sha,
        "per_fold_n_train": per_fold_n_train,
        "min_per_fold_n_train": int(min_n_train),
        "d_ambient": D_AMBIENT,
        "basis": basis,
        # Even the reduced k=1024 read is degenerate when n_train falls at or
        # below k (the PCA keeps ~full rank; the smoke's capped-n redk read
        # demonstrated this at n_train~960 → R² −373). Flagged, never hidden.
        "reduced_read_degenerate": bool(min_n_train <= REDUCED_BASIS_K),
        "smoke_max_n": smoke_max_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell fit task (worker; reuses issue2054_fits._fit_arm_cell verbatim)


def _regime_of(task: dict) -> dict:
    return {
        "cell": task["cell_key"],
        "arm": "context",
        "matched_sha256": task["matched_sha256"],
        "matched_n": task["matched_n"],
        "n_null_draws": task["n_null_draws"],
        "bootstrap_draws": task["bootstrap_draws"],
        "seed": task["seed"],
        "reduced_basis_k": REDUCED_BASIS_K,
        "fold_map_k": task["fold_map_k"],
        "fold_map_seed": task["fold_map_seed"],
        "n_bins": task["n_bins"],
        "smoke_max_n": task["smoke_max_n"],
    }


def _fit_cell_task(task: dict) -> dict:
    """One matched-cell CONTEXT-arm refit through the REUSED production fit
    core. Checkpoints its full arm report (regime-keyed) and returns a digest;
    a resumable checkpoint with a matching regime short-circuits."""
    t0 = time.time()
    ckpt_path = Path(task["ckpt_path"])
    regime = _regime_of(task)
    if ckpt_path.is_file():
        try:
            with ckpt_path.open(encoding="utf-8") as f:
                prior = json.load(f)
            if prior.get("regime") == regime and prior.get("arm_report", {}).get("status") == "ok":
                return {
                    "cell_key": task["cell_key"],
                    "status": "resumed",
                    "ckpt_path": str(ckpt_path),
                    "wall_seconds": 0.0,
                }
        except (OSError, json.JSONDecodeError):
            pass
    try:
        fm_path = task["fold_map_path"]
        if fm_path not in _WORKER_FOLD_MAP:
            _WORKER_FOLD_MAP[fm_path] = fits._load_fold_map(Path(fm_path))
        fold_map = _WORKER_FOLD_MAP[fm_path]
        activations = fits._load_activation_npz(Path(task["npz_path"]))
        if activations is None:
            raise ValueError(f"empty activation .npz: {task['npz_path']}")
        arm_report = fits._fit_arm_cell(
            variant=task["variant"],
            model=task["model"],
            arm="context",
            activations=activations,
            fold_map=fold_map,
            restrict_ids=set(task["matched_ids"]),
            n_null_draws=int(task["n_null_draws"]),
            seed=int(task["seed"]),
            pilot=False,
            bootstrap_draws=int(task["bootstrap_draws"]),
        )
        wall = time.time() - t0
        payload = {
            "cell": task["cell_key"],
            "pair_key": task["pair_key"],
            "condition": task["condition"],
            "regime": regime,
            "arm_report": arm_report,
            "wall_seconds": round(wall, 3),
            "utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        fits._write_json(ckpt_path, payload)
        return {
            "cell_key": task["cell_key"],
            "status": arm_report.get("status", "unknown"),
            "ckpt_path": str(ckpt_path),
            "wall_seconds": wall,
        }
    except Exception as exc:  # noqa: BLE001 — recorded per cell; siblings continue
        return {
            "cell_key": task["cell_key"],
            "status": "error",
            "error": repr(exc),
            "wall_seconds": time.time() - t0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full (unmatched) ceilings from the committed production fit JSONs


def _full_ceiling(full_fits_dir: Path, cell_key: str) -> dict:
    p = full_fits_dir / f"{cell_key}.json"
    if not p.is_file():
        _log(f"full fit JSON missing locally; staging from HF: fits/{cell_key}.json")
        hub.stage_hub_file(
            HF_DATA_REPO, f"{TASK_PREFIX}/fits/{cell_key}.json", p, repo_type="dataset"
        )
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    ar = d["arm_reports"]["context"]
    per_fold = [r for r in ar.get("per_fold", []) if "r2_ambient" in r]
    return {
        "r2_ambient_mean": ar["pooled"].get("r2_ambient_mean"),
        "r2_reduced_k1024_mean": ar["pooled"].get("r2_reduced_k1024_mean"),
        "n_restrict": d.get("shared_conv_id_intersection"),
        "min_per_fold_n_train": min((r["n_train"] for r in per_fold), default=None),
        "n_null_draws": d.get("n_null_draws"),
        "restrict_sha256": d.get("restrict_sha256"),
    }


def _matched_digest(ckpt_path: Path) -> dict:
    with ckpt_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    ar = payload["arm_report"]
    per_fold = [r for r in ar.get("per_fold", []) if "r2_ambient" in r]
    return {
        "r2_ambient_mean": ar["pooled"].get("r2_ambient_mean"),
        "r2_ambient_per_fold": [r["r2_ambient"] for r in per_fold],
        "r2_reduced_k1024_mean": ar["pooled"].get("r2_reduced_k1024_mean"),
        "r2_reduced_per_fold": [r["r2_reduced_k1024"] for r in per_fold],
        "r2_identity_bias_mean": ar["pooled"].get("r2_identity_bias_mean"),
        "null_r2_pooled_median": ar["pooled"].get("null_r2_pooled_median"),
        "null_r2_pooled_p95": ar["pooled"].get("null_r2_pooled_p95"),
        "per_fold_n_train": [r["n_train"] for r in per_fold],
        "per_fold_n_val": [r["n_val"] for r in per_fold],
        "fold_sizes": ar.get("fold_sizes"),
        "any_dof_over_cap": any(
            (r.get("info_ambient") or {}).get("dof_over_cap") for r in per_fold
        ),
        "wall_seconds": payload.get("wall_seconds"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure


def _short_pair_label(pair_key: str) -> str:
    variant, form, model = pair_key.split(forms.CELL_KEY_SEP)
    v = variant.replace("char_", "").replace("conversation_paired_stories_assistant", "asst")
    f = {"attrib_quoted": "attr", "bare_label": "blab", "bare_text": "btxt", "chat": "chat"}.get(
        form, form
    )
    m = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instr"}[model]
    return f"{v}/{f}/{m}"


def render_figure(pair_rows: list[dict], fig_path: Path) -> None:
    """Two panels: (A) per-pair inserted-minus-on-policy ceiling gap, full vs
    length-matched (dumbbell, basis-scoped values); (B) per-cell matched vs
    full R² scatter (48 points, diagonal reference), labeled per unit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    pal = paper_palette_blog(8)
    c_inserted, c_onpolicy = pal[4], pal[5]  # analyzer figs color discipline
    c_full, c_matched = pal[1], pal[0]

    rows = sorted(pair_rows, key=lambda r: r["gap_full_basis"], reverse=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 7.2), width_ratios=[1.15, 1.0])

    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        ax_a.plot(
            [r["gap_full_basis"], r["gap_matched_basis"]], [y, y], color="0.75", lw=1.2, zorder=1
        )
    marker_of = {"ambient": "o", "reduced_k1024": "s"}
    for kind, color in (("gap_full_basis", c_full), ("gap_matched_basis", c_matched)):
        for y, r in zip(ys, rows):
            ax_a.scatter(
                r[kind],
                y,
                color=color,
                marker=marker_of[r["basis"]],
                s=42,
                zorder=2,
                edgecolor="white",
                linewidth=0.5,
            )
    ax_a.axvline(0.0, color="0.4", lw=0.8, ls="--", zorder=0)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([_short_pair_label(r["pair_key"]) for r in rows], fontsize=7)
    ax_a.set_xlabel("context-arm R² gap: inserted − on-policy (pair basis)")
    ax_a.scatter([], [], color=c_full, marker="o", label="full (committed)")
    ax_a.scatter([], [], color=c_matched, marker="o", label="length-matched")
    ax_a.scatter([], [], color="0.4", marker="s", label="reduced k=1024 basis")
    ax_a.legend(loc="lower right", fontsize=8)

    for r in pair_rows:
        for cond, color in (("inserted", c_inserted), ("on_policy", c_onpolicy)):
            cell = r["cells"][cond]
            if cell.get("matched") is None:
                continue
            key = "r2_ambient_mean" if r["basis"] == "ambient" else "r2_reduced_k1024_mean"
            x = cell["full"][key]
            y = cell["matched"][key]
            ax_b.scatter(
                x,
                y,
                color=color,
                marker=marker_of[r["basis"]],
                s=34,
                edgecolor="white",
                linewidth=0.5,
            )
            ax_b.annotate(
                _short_pair_label(r["pair_key"]),
                (x, y),
                fontsize=5,
                alpha=0.75,
                xytext=(2, 2),
                textcoords="offset points",
            )
    lims = ax_b.get_xlim() + ax_b.get_ylim()
    lo, hi = min(lims), max(lims)
    ax_b.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=0)
    ax_b.set_xlabel("full-population R² (committed fits, pair basis)")
    ax_b.set_ylabel("length-matched R² (pair basis)")
    ax_b.scatter([], [], color=c_inserted, label="inserted cell")
    ax_b.scatter([], [], color=c_onpolicy, label="on-policy cell")
    ax_b.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    set_title_subtitle(
        ax_a,
        "Length-stratified refit: inserted vs on-policy context-arm ceilings",
        "matched = equal per-decile-bin subsamples of pooled answer lengths",
    )
    set_title_subtitle(ax_b, "Per-cell R²: matched vs full", "squares = reduced k=1024 basis")
    savefig_paper(fig, fig_path.stem, dir=fig_path.parent)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main


def main() -> int:
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--activations-dir", required=True)
    ap.add_argument(
        "--out-json",
        default=str(
            _REPO_ROOT / "eval_results/issue_2054/analyzer_companions/length_stratified_refit.json"
        ),
    )
    ap.add_argument(
        "--fig", default=str(_REPO_ROOT / "figures/issue_2054/length_stratified_refit.png")
    )
    ap.add_argument("--fold-map", default=str(DEFAULT_FOLD_MAP))
    ap.add_argument("--companion", default=str(DEFAULT_COMPANION))
    ap.add_argument("--full-fits-dir", default=str(DEFAULT_FULL_FITS_DIR))
    ap.add_argument("--checkpoint-dir", default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-null-draws", type=int, default=50)
    ap.add_argument("--bootstrap-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="optional pair-key subset (smoke); default = all 24 companion pairs",
    )
    ap.add_argument(
        "--smoke-max-n",
        type=int,
        default=None,
        help="smoke SCALE dial: cap matched n per cell (proportional per-bin downscale; "
        "regime-keyed so production never resumes smoke checkpoints)",
    )
    ap.add_argument("--stage-only", action="store_true", help="stage activations then exit")
    args = ap.parse_args()

    activations_dir = Path(args.activations_dir).resolve()
    activations_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json).resolve()
    fig_path = Path(args.fig).resolve()

    companion_path = Path(args.companion).resolve()
    fold_map_path = Path(args.fold_map).resolve()
    for p, what in ((companion_path, "parity companion"), (fold_map_path, "fold map")):
        if not p.is_file():
            print(f"ERROR: {what} not found: {p}", file=sys.stderr)
            return 2

    pairs = _pairs_from_companion(companion_path)
    if args.pairs:
        keep = set(args.pairs)
        pairs = [p for p in pairs if p["pair_key"] in keep]
        if not pairs:
            print(f"ERROR: --pairs matched no companion pair keys: {args.pairs}", file=sys.stderr)
            return 2
    fold_map = fits._load_fold_map(fold_map_path)
    fold_of = fold_map["fold_of"]

    _log(
        f"start: pairs={len(pairs)} workers={args.workers} n_null_draws={args.n_null_draws} "
        f"bootstrap_draws={args.bootstrap_draws} seed={args.seed} n_bins={args.n_bins} "
        f"activations_dir={activations_dir}"
    )

    stage_activations(activations_dir, pairs)
    if args.stage_only:
        _log("stage-only: staging complete")
        print(f"[lenstrat] DONE rc=0 cells=0/{2 * len(pairs)}", flush=True)
        sys.stdout.flush()
        sys.exit(0)

    # Matched-set manifests (deterministic; recomputed on every run — the fit
    # checkpoints key on the matched-set sha, so a drifted manifest RECOMPUTES).
    manifests: dict[str, dict] = {}
    for pair_idx, pair in enumerate(pairs):
        m = build_matched_sets(
            pair,
            pair_idx,
            activations_dir,
            fold_of,
            n_bins=args.n_bins,
            seed=args.seed,
            smoke_max_n=args.smoke_max_n,
        )
        manifests[pair["pair_key"]] = m
        fits._write_json(
            ckpt_dir / f"manifest__{pair['pair_key']}.json",
            {k: v for k, v in m.items() if k != "matched_ids"}
            | {"utc": datetime.now(tz=timezone.utc).isoformat()},
        )
        _log(
            f"matched pair={pair['pair_key']} n_b={m['matched_n']['inserted']} "
            f"n_d={m['matched_n']['on_policy']} min_n_train={m['min_per_fold_n_train']} "
            f"basis={m['basis']} bins={m['n_bins_realized']}"
        )

    # Fit tasks: 2 cells per pair; sub-floor cells are recorded skips.
    tasks: list[dict] = []
    skipped: list[dict] = []
    for pair_idx, pair in enumerate(pairs):
        m = manifests[pair["pair_key"]]
        for cond in ("inserted", "on_policy"):
            cell_key = m["cells"][cond]
            if m["matched_n"][cond] < MATCHED_N_FLOOR:
                skipped.append(
                    {
                        "cell_key": cell_key,
                        "reason": f"matched_n {m['matched_n'][cond]} < floor {MATCHED_N_FLOOR}",
                    }
                )
                continue
            tasks.append(
                {
                    "pair_key": pair["pair_key"],
                    "cell_key": cell_key,
                    "condition": cond,
                    "variant": pair["variant"],
                    "form": pair["form"],
                    "model": pair["model"],
                    "npz_path": str(_npz_target(activations_dir, pair["variant"], cell_key)),
                    "matched_ids": m["matched_ids"][cond],
                    "matched_sha256": m["matched_sha256"][cond],
                    "matched_n": m["matched_n"][cond],
                    "n_null_draws": int(args.n_null_draws),
                    "bootstrap_draws": int(args.bootstrap_draws),
                    "seed": int(args.seed),
                    "fold_map_path": str(fold_map_path),
                    "fold_map_k": int(fold_map["k"]),
                    "fold_map_seed": int(fold_map.get("seed", -1)),
                    "n_bins": int(args.n_bins),
                    "smoke_max_n": args.smoke_max_n,
                    "ckpt_path": str(ckpt_dir / f"cell__{cell_key}.json"),
                }
            )
    for s in skipped:
        _log(f"SKIP {s['cell_key']}: {s['reason']}")

    n_total = len(tasks)
    results: dict[str, dict] = {}

    # Pilot: the FIRST cell runs inline through the production task fn; its
    # measured wall extrapolates the fleet BEFORE the fan-out.
    if tasks:
        pilot = tasks[0]
        _log(f"pilot: cell={pilot['cell_key']} matched_n={pilot['matched_n']} (1/{n_total})")
        t0 = time.time()
        res = _fit_cell_task(pilot)
        pilot_wall = time.time() - t0
        results[pilot["cell_key"]] = res
        remaining = n_total - 1
        proj = pilot_wall * remaining / max(1, args.workers)
        _log(
            f"pilot done: status={res['status']} wall={pilot_wall:.1f}s "
            f"projected_remaining={proj / 60:.1f} min at width {args.workers} "
            f"({remaining} cells)"
        )
        if res["status"] == "error":
            _log(f"pilot ERROR: {res.get('error')}")
            print(f"[lenstrat] DONE rc=1 cells=0/{n_total}", flush=True)
            sys.stdout.flush()
            sys.exit(1)

    rest = tasks[1:]
    if rest:
        done_i = 1
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fit_cell_task, t): t for t in rest}
            for fut in as_completed(futures):
                t = futures[fut]
                res = fut.result()
                results[t["cell_key"]] = res
                done_i += 1
                _log(
                    f"unit {done_i}/{n_total} cell={t['cell_key']} "
                    f"status={res['status']} wall={res['wall_seconds']:.1f}s"
                )

    n_err = sum(1 for r in results.values() if r["status"] == "error")
    n_ok = sum(1 for r in results.values() if r["status"] in ("ok", "resumed"))
    for cell_key, r in sorted(results.items()):
        if r["status"] == "error":
            _log(f"ERROR cell={cell_key}: {r.get('error')}")

    # Aggregate: per pair, full vs matched in the pair's basis.
    pair_rows: list[dict] = []
    for pair in pairs:
        m = manifests[pair["pair_key"]]
        row: dict = {
            "pair_key": pair["pair_key"],
            "variant": pair["variant"],
            "form": pair["form"],
            "model": pair["model"],
            "basis": m["basis"],
            "reduced_read_degenerate": m["reduced_read_degenerate"],
            "min_per_fold_n_train": m["min_per_fold_n_train"],
            "d_ambient": D_AMBIENT,
            "bin_edges_interior": m["bin_edges_interior"],
            "n_bins_realized": m["n_bins_realized"],
            "per_bin": m["per_bin"],
            "coverage": m["coverage"],
            "cells": {},
        }
        complete = True
        for cond in ("inserted", "on_policy"):
            cell_key = m["cells"][cond]
            cell_row: dict = {
                "cell": cell_key,
                "matched_n": m["matched_n"][cond],
                "matched_per_fold_n_train": m["per_fold_n_train"][cond],
                "matched_sha256": m["matched_sha256"][cond],
                "full": _full_ceiling(Path(args.full_fits_dir), cell_key),
                "matched": None,
            }
            res = results.get(cell_key)
            if res is not None and res["status"] in ("ok", "resumed"):
                cell_row["matched"] = _matched_digest(Path(res["ckpt_path"]))
            else:
                complete = False
                cell_row["status"] = (res or {}).get("status", "skipped-below-floor")
                if res is not None and "error" in res:
                    cell_row["error"] = res["error"]
            row["cells"][cond] = cell_row
        # A pair is comparison-VALID only when it has a non-degenerate basis at
        # matched n: ambient needs every fold n_train >= d; the reduced k=1024
        # fallback needs n_train > k (at n_train <= k the PCA keeps ~full rank
        # and the read is meaningless — the smoke measured R² −373 there).
        valid = complete and not (m["basis"] == "reduced_k1024" and m["reduced_read_degenerate"])
        if complete and not valid:
            row["comparison_invalid_reason"] = (
                f"matched min n_train {m['min_per_fold_n_train']} <= reduced k "
                f"{REDUCED_BASIS_K}: no non-degenerate estimator basis at matched n"
            )
        if valid:
            key = "r2_ambient_mean" if m["basis"] == "ambient" else "r2_reduced_k1024_mean"
            b_full = row["cells"]["inserted"]["full"][key]
            d_full = row["cells"]["on_policy"]["full"][key]
            b_m = row["cells"]["inserted"]["matched"][key]
            d_m = row["cells"]["on_policy"]["matched"][key]
            row["gap_full_basis"] = b_full - d_full
            row["gap_matched_basis"] = b_m - d_m
            row["delta_gap"] = row["gap_matched_basis"] - row["gap_full_basis"]
            # Both-bases reads for the record (comparison is drawn in `basis`).
            row["gap_full_ambient"] = (
                row["cells"]["inserted"]["full"]["r2_ambient_mean"]
                - row["cells"]["on_policy"]["full"]["r2_ambient_mean"]
            )
            row["gap_matched_ambient"] = (
                row["cells"]["inserted"]["matched"]["r2_ambient_mean"]
                - row["cells"]["on_policy"]["matched"]["r2_ambient_mean"]
            )
        row["complete"] = complete
        row["comparison_valid"] = valid
        pair_rows.append(row)

    prov = git_provenance()
    out_payload = {
        "metadata": {
            **as_metadata_dict(prov),
            "utc": datetime.now(tz=timezone.utc).isoformat(),
            "driver": "scripts/issue2054_length_stratified_refit.py",
            "fit_core": "scripts/issue2054_fits.py::_fit_arm_cell (reused verbatim)",
            "arm": "context (stated deviation: prefix arm <= 0.021 in all committed cells)",
            "tokenizer": TOKENIZER_ID,
            "length_recipe": "answer field, add_special_tokens=False (parity-companion exact)",
            "matching": (
                "pooled-pair decile bins; per bin min(n_b, n_d) drawn per cell; "
                "default_rng([seed, pair_idx]) over sorted conv_ids"
            ),
            "seed": int(args.seed),
            "n_bins": int(args.n_bins),
            "n_null_draws": int(args.n_null_draws),
            "bootstrap_draws": int(args.bootstrap_draws),
            "smoke_max_n": args.smoke_max_n,
            "matched_n_floor": MATCHED_N_FLOOR,
            "fold_map": {
                "path": str(fold_map_path),
                "k": int(fold_map["k"]),
                "seed": int(fold_map.get("seed", -1)),
            },
            "full_fits_dir": str(args.full_fits_dir),
            "n_pairs": len(pairs),
            "n_cells_fit_ok": n_ok,
            "n_cells_error": n_err,
            "skipped_cells": skipped,
        },
        "pairs": {row["pair_key"]: row for row in pair_rows},
    }
    fits._write_json(out_json, out_payload)
    _log(f"wrote {out_json}")

    valid_rows = [r for r in pair_rows if r["comparison_valid"]]
    if valid_rows:
        render_figure(valid_rows, fig_path)
        _log(f"wrote {fig_path}")
    else:
        _log("no comparison-valid pairs; figure skipped")

    rc = 0 if (n_err == 0 and n_ok == n_total) else 1
    print(f"[lenstrat] DONE rc={rc} cells={n_ok}/{n_total}", flush=True)
    sys.stdout.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
