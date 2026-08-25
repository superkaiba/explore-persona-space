#!/usr/bin/env python
"""issue2546_fit_cells.py — P5 fit driver for task #2546 (plan v4 §4.2 P5).

A thin driver over the ``issue825_fit_cells`` cores exactly as
``scripts/issue1336_fit_cells.py`` is: ``heldout_r2_sweep`` (layers+λ batched,
selection-symmetric nulls), ``selection_symmetric_summary``,
``random_projection_control``, ``mean_baseline_r2``, ``bootstrap_ci`` /
``bootstrap_r2_ci``, plus the mandatory mapping-baselines pair
(``identity_bias_predict`` + ``knn_retrieval``) and the #1336 ladder/bootstrap
machinery (``issue1336_ladder_alignment``: ``draw_index_matrix`` /
``counts_from_indices`` / ``weighted_r2_draws`` / ``paired_bootstrap_batched``)
and the #1345 operator battery (``raw_cosine_with_rotation_null`` direction-aware
vs ``spectrum_cosine`` rotation-invariant-only).

Store consumed (written by ``issue2546_gen_capture.py`` P4): kind-tensor shards
``<out_root>/store/arm{K}/{side}__{corpus}/slot*.shard*.pt`` — dicts with
``full`` (B, K_kinds, L_all, H) bf16, ``tk`` (B, 9, 3, H) bf16 post-like,
``row_ids``, ``meta``. Production frees local shards post-upload, so the driver
re-stages absent stems from HF ``issue2546_cotmap/analysis_tensors/thinkstore/
arm{K}/{stem}`` (smoke: ``smoke_arm{K}``) via the scoped
``hub.stage_hub_prefix`` recipe, reduces each stem ONCE into a per-kind
fitcache (bf16 — fp16 refused, #825 parity), and frees the staged mirror.

Registry (plan §9, asserted): arms 1–2 each 65 statistical units
(Plot 7 = 4 cells × [2 strata + 4 per-corpus] = 24; Plot 8 = 4 cells ×
[pooled + 4 corpora + 2 strata] = 28; ladder pairs 5; matched-n companions 8);
arm 3 = 38 (Plot-7 think-on 24 + think-off A-cell 6 + matched-n 8); frozen-layer
trajectory units 9 t × 2 strata × 3 arms = 54. Total 222. The two exploratory
OOD transfer arms (§6) ride on top, never counted in 222.

Reliability stems (forward contract): split-half target consistency expects
capture stems named ``rel_{side}__{corpus}`` with per-row ``meta['draw']``
(4 T=0.6 draws per prompt). Unit 2's capture rig does not yet emit them —
absent stems produce a LOUD ``status: missing_reliability_capture`` artifact
(never a silent null), and every cell JSON carries
``ceiling_status: missing_reliability_capture`` beside a null
ceiling-normalized R² until the capture leg lands.

Recorded module-global patch (plan §4.2 P5 / §10): ``fc.N_INNER_LAMBDA_FOLDS = 2``
(main default 4; ``heldout_r2_sweep`` exposes no inner-folds kwarg; #1336
realized 2 the same way on its branch). Applied loudly in ``main()`` before any
fit, --g0 included (the #1336 G-E gate PASSed under the same regime).

CLI contract (pinned by scripts/issue2546_dispatch.sh):
  issue2546_fit_cells.py --arm K [--g0] [--smoke] --out-root <dir>
Internal fan-out: the parent builds the fitcache + rowsets once (fan-out inputs
pre-staged in the parent, #1315), then spawns one worker per visible GPU with
``CUDA_VISIBLE_DEVICES=<slot>`` pinned in the LAUNCHER env (+ ``--shard i/N``).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import gc as _pygc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))  # thread caps bind BEFORE torch import (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Reused cores, hoisted to module top so a missing symbol crashes at process
# start, never inside a smoke-skipped branch (gotchas.md #606).
import issue825_fit_cells as fc  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue1336_fit_cells as f36  # noqa: E402
import issue1336_ladder_alignment as la  # noqa: E402
import issue825_crossmodel_map_transfer as xm  # noqa: E402
import issue1345_operator_comparison as oc  # noqa: E402
import issue2546_gen_capture as g25  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# ---------------------------------------------------------------------------
# Constants (plan v4 §4.2 P5 / §9 / §10)
# ---------------------------------------------------------------------------

TASK = 2546
RECIPE_REV = "planv4"
N_FOLDS = 5
FIT_SEED = 0
N_NULL_DRAWS = 20  # advisory shuffle nulls per fit (plan §7: no null-statistic gates)
N_BOOT = 1000  # paired prompt-level bootstrap draws
N_ROT_DRAWS = 50  # operator rotation-null draws (d x d QR per draw)
SMOKE_NULL_DRAWS = 2
SMOKE_N_BOOT = 50
SMOKE_MIN_ROWS = 12  # floor demotion under smoke (#1345 gate-calibration parity)

# λ grid: LAMBDAS_N1M from issue779_ffc_n1m_fits.py:112 (np.logspace(-3, 8, 23)).
# Defined from GENERATING PARAMETERS here (machine-stable resume keys, #1336)
# rather than importing issue779_ffc_n1m_fits (module-top import chain is heavy).
LAMBDAS_N1M_PARAMS = ("logspace", -3.0, 8.0, 23)
LAMBDAS_N1M = np.logspace(-3.0, 8.0, 23)

INNER_LAMBDA_FOLDS = 2  # recorded module-global patch target (main default 4)

HEADLINE_LAYER = {1: 19, 2: 19, 3: 24}
# Usable-row floors s.t. K=5 fold-train n >= 1.2d (plan MF-2): arms 1-2 d=3584
# -> n_train >= 4301, rows >= 5376; arm 3 d=4096 -> n_train >= 4915, rows >= 6144.
FLOOR_ROWS = {1: 5376, 2: 5376, 3: 6144}
MATCHED_N_ROWS = dict(FLOOR_ROWS)  # per-corpus matched-n companions AT the margin

PER_CORPUS_CELLS = ("gsm8k_train", "math", "contexthub", "mmlu")
STRATA = ("does", "doesnt")
TEMPLATE_POOL_MIN = 20

# Per-corpus band recompute (MF-5): band_value = DELTA_ELICIT_BAND x rate_c with
# rate_c = R2_F(corpus)/COMMITTED_ANCHOR — the #1336 exchange-rate formula
# (rate = s/anchor, issue_1336.common.load_qwen_recal_cal) recomputed from THAT
# corpus's own within-post reference map. Prior anchor: #1336's pooled 0.0207.
DELTA_ELICIT_BAND = 0.02
COMMITTED_ANCHOR_R2 = 0.6731  # #825 Qwen S1 @ L19 (issue_1336.common.G0)

LADDER_TIER_NAMES = (
    "t0_direct_transfer",
    "t1_context_offset",
    "t2_answer_offset",
    "t3_bias_offset",
    "t4_global_scaling",
    "t5_mapping_rotation",
    "t6_reparam_contexts",
    "t7_reparam_answers",
    "t8_reparam_both",
)

MATH_CORPORA = ("gsm8k_test", "gsm8k_train", "math")
MCQ_CORPORA = ("mmlu", "arc_challenge", "csqa", "piqa")


# ---------------------------------------------------------------------------
# Layer profile (per-arm; selftest threads a tiny synthetic profile)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerProfile:
    arm: int
    n_layers: int
    hidden: int
    frozen: tuple[int, ...]
    headline: int
    post_side: str  # side name of the post-like stems
    short_side: str  # side name of the short-like stems
    has_pre_model: bool  # arms 1-2: pre model exists -> Plot 8 + ladder


def profile_for_arm(arm: int) -> LayerProfile:
    """Production layer profile from the gen/capture arm registry (single source)."""
    a = g25.ARMS[arm]
    return LayerProfile(
        arm=arm,
        n_layers=a.n_layers,
        hidden=a.hidden,
        frozen=tuple(a.frozen),
        headline=HEADLINE_LAYER[arm],
        post_side=a.sides[0].side,
        short_side=a.sides[1].side,
        has_pre_model=arm in (1, 2),
    )


def stratum_of(meta: dict) -> str | None:
    """Necessity-stratum assignment from shard meta (plan §4 corpora table)."""
    c = meta["corpus"]
    if c == "gsm8k_train":
        k = meta.get("k")
        if k is None:
            return None
        k = int(k)
        return "does" if k >= 4 else ("doesnt" if k == 1 else None)
    if c == "math":
        return "does"
    if c == "contexthub":
        lvl = meta.get("level")
        if lvl is None:
            return None
        lvl = int(lvl)
        return "does" if lvl in (3, 4) else ("doesnt" if lvl == 1 else None)
    if c in MCQ_CORPORA:
        return "doesnt"
    return None  # gsm8k_test: graded panel + pilot only


# ---------------------------------------------------------------------------
# Fitcache: stage shards (local-first, else HF), reduce ONCE per stem to
# per-kind bf16 tensors + a rows sidecar; free staged mirrors.
# ---------------------------------------------------------------------------


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(tmp, path)


def _store_prefix(arm: int, smoke: bool) -> str:
    return f"{g25.STORE_PREFIX}/{'smoke_' if smoke else ''}arm{arm}"


def _list_hf_stems(arm: int, smoke: bool) -> list[str]:
    """Scoped (never full-listing) stem enumeration under the arm store prefix."""
    from huggingface_hub import HfApi

    prefix = _store_prefix(arm, smoke)
    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: scoped path_in_repo walk inside retry_transient
            api.list_repo_tree(
                hub.DEFAULT_DATASET_REPO, path_in_repo=prefix, repo_type="dataset", recursive=False
            )
        ),
        what=f"p5 stem listing {prefix}",
    )
    return sorted(Path(e.path).name for e in entries if "__" in Path(e.path).name)


def enumerate_stems(out_root: Path, arm: int, smoke: bool, *, offline: bool = False) -> list[str]:
    """Union of local store stems and HF-hosted stems (production freed local shards).

    ``offline=True`` (the --selftest path) skips the HF listing entirely so the
    synthetic-store smoke never touches the network (adoptable-test contract).
    """
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    local = out_root / "store" / f"arm{arm}"
    stems = set()
    if local.is_dir():
        stems.update(p.name for p in local.iterdir() if p.is_dir() and "__" in p.name)
    if not offline:
        try:
            stems.update(_list_hf_stems(arm, smoke))
        except (FileNotFoundError, EntryNotFoundError, RepositoryNotFoundError):
            pass  # nothing uploaded yet (e.g. --skip-upload smoke): local-only is fine
    if not stems:
        raise FileNotFoundError(
            f"no capture stems for arm {arm} — neither {local} nor HF "
            f"{_store_prefix(arm, smoke)} hold any {{side}}__{{corpus}} stem; run P4 first"
        )
    return sorted(stems)


def _shard_files(stem_dir: Path) -> list[Path]:
    return sorted(stem_dir.glob("slot*.shard*.pt"))


def _stage_stem(
    out_root: Path, arm: int, stem: str, smoke: bool, *, offline: bool = False
) -> tuple[Path, bool]:
    """Return a dir holding the stem's shards; stage from HF when local is empty.

    Returns (shard_dir, staged) — staged=True means the dir is a temporary HF
    mirror the caller frees after the reduce (never the local store originals).
    """
    local = out_root / "store" / f"arm{arm}" / stem
    if _shard_files(local):
        return local, False
    if offline:
        raise FileNotFoundError(f"offline mode: no local shards for {stem} at {local}")
    prefix = f"{_store_prefix(arm, smoke)}/{stem}"
    mirror_root = out_root / "fitstage"
    hub.stage_hub_prefix(hub.DEFAULT_DATASET_REPO, prefix, mirror_root, repo_type="dataset")
    staged = mirror_root / prefix  # mirror-root semantics: files at root/<repo path>
    assert _shard_files(staged), f"staged mirror empty for {prefix} under {mirror_root}"
    return staged, True


def _load_shard(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Self-produced shard from THIS pipeline (tensors + primitives only);
        # torch>=2.6 weights_only=True can refuse benign containers.
        return torch.load(path, map_location="cpu", weights_only=False)


def build_fitcache(
    out_root: Path, prof: LayerProfile, smoke: bool, *, offline: bool = False
) -> dict[str, Path]:
    """Reduce every stem's shards to per-kind bf16 tensors (once, resumable).

    Cache layout per stem: <out_root>/fitcache/arm{K}/{stem}/{kind}.pt (bf16
    (N, L, H)), tk.pt (bf16 (N, 9, 3, H), post-like), rows.json (row_ids +
    meta + kinds + shapes), _cache_complete.json (fingerprint).
    """
    cache_root = out_root / "fitcache" / f"arm{prof.arm}"
    stems = enumerate_stems(out_root, prof.arm, smoke, offline=offline)
    done: dict[str, Path] = {}
    for stem in stems:
        cdir = cache_root / stem
        marker = cdir / "_cache_complete.json"
        if marker.is_file():
            done[stem] = cdir
            print(f"[p5-cache] {stem}: resume-skip", flush=True)
            continue
        t0 = time.time()
        shard_dir, staged = _stage_stem(out_root, prof.arm, stem, smoke, offline=offline)
        files = _shard_files(shard_dir)
        fulls, tks, row_ids, metas = [], [], [], []
        kinds_full: list[str] = []
        layers_all: list[int] | None = None
        for f in files:
            sh = _load_shard(f)
            assert int(sh["arm"]) == prof.arm, (stem, sh["arm"], prof.arm)
            assert int(sh["hidden"]) == prof.hidden, (stem, sh["hidden"], prof.hidden)
            sh_layers = [int(v) for v in sh["layers_all"]]
            if stem.startswith("rel_"):
                # P4b reliability stems carry the FROZEN layer subset only
                # (issue2546_gen_capture --phase capture-reliability; named
                # deviation — readers map the headline layer BY INDEX).
                assert sh_layers == list(prof.frozen), (stem, sh_layers, prof.frozen)
            else:
                assert len(sh_layers) == prof.n_layers, (stem, len(sh_layers))
            if layers_all is None:
                layers_all = sh_layers
            assert sh_layers == layers_all, (stem, f.name, sh_layers, layers_all)
            kinds_full = list(sh["kinds_full"])
            fulls.append(sh["full"])
            if sh.get("tk") is not None:
                tks.append(sh["tk"])
            row_ids.extend(sh["row_ids"])
            metas.extend(sh["meta"])
        assert fulls, f"{stem}: no shards"
        full = torch.cat(fulls, dim=0)  # (N, K, L, H) bf16
        assert full.shape[0] == len(row_ids), (full.shape, len(row_ids))
        cdir.mkdir(parents=True, exist_ok=True)
        for ki, kind in enumerate(kinds_full):
            torch.save(full[:, ki].contiguous(), cdir / f"{kind}.pt")
        if tks:
            torch.save(torch.cat(tks, dim=0).contiguous(), cdir / "tk.pt")
        _atomic_json(
            cdir / "rows.json",
            {
                "stem": stem,
                "row_ids": row_ids,
                "meta": metas,
                "kinds_full": kinds_full,
                "has_tk": bool(tks),
                "layers_all": layers_all,
                "n_layers": prof.n_layers,
                "hidden": prof.hidden,
            },
        )
        del fulls, tks, full
        _pygc.collect()
        if staged:
            shutil.rmtree(shard_dir)  # free the HF mirror only, never store originals
        _atomic_json(
            marker,
            {
                "fingerprint": {"n_rows": len(row_ids), "shards": [f.name for f in files]},
                "wall_s": time.time() - t0,
            },
        )
        done[stem] = cdir
        print(
            f"[p5-cache] {stem}: {len(row_ids)} rows cached ({time.time() - t0:.0f}s)", flush=True
        )
    return done


class StemCache:
    """Read-side view of one stem's fitcache (row index + lazy kind tensors)."""

    def __init__(self, cdir: Path):
        self.dir = cdir
        rows = json.loads((cdir / "rows.json").read_text())
        self.row_ids: list[str] = rows["row_ids"]
        self.meta: list[dict] = rows["meta"]
        self.kinds: list[str] = rows["kinds_full"]
        self.has_tk: bool = rows["has_tk"]
        # Layer ids of the tensor's L axis (None on legacy caches = full range;
        # rel_ stems carry the FROZEN subset — index-map, never absolute).
        self.layers: list[int] | None = rows.get("layers_all")
        self.pos = {r: i for i, r in enumerate(self.row_ids)}

    def kind_rows(self, kind: str, row_ids: list[str]) -> np.ndarray:
        """(n, L, H) fp32 for the requested rows of one kind ('tk' -> (n, 9, 3, H))."""
        t = torch.load(self.dir / f"{kind}.pt", map_location="cpu", weights_only=True)
        idx = torch.as_tensor([self.pos[r] for r in row_ids], dtype=torch.long)
        out = t.index_select(0, idx).to(torch.float32).numpy()
        del t
        return out


# ---------------------------------------------------------------------------
# Row sets + subsets (registered row-set discipline, plan §3)
# ---------------------------------------------------------------------------


def load_caches(out_root: Path, prof: LayerProfile) -> dict[str, dict[str, StemCache]]:
    """side -> corpus -> StemCache from a COMPLETE fitcache (fail-loud)."""
    cache_root = out_root / "fitcache" / f"arm{prof.arm}"
    out: dict[str, dict[str, StemCache]] = {}
    if not cache_root.is_dir():
        raise FileNotFoundError(f"fitcache missing at {cache_root} — parent phase must run first")
    for cdir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        if not (cdir / "_cache_complete.json").is_file():
            raise RuntimeError(f"incomplete fitcache stem {cdir} — rebuild the cache")
        side, corpus = cdir.name.split("__", 1)
        out.setdefault(side, {})[corpus] = StemCache(cdir)
    for side in (prof.post_side, prof.short_side):
        assert side in out, f"no cached stems for side {side!r} (have {sorted(out)})"
    return out


def build_rowsets(out_root: Path, prof: LayerProfile, caches, smoke: bool) -> dict:
    """ONE shared usable-row set per arm: post-side ∩ short/pre-side, per corpus.

    A row dropped for any cell drops for all cells of the arm; drop counts
    persisted per corpus (plan §3 registered row-set discipline). SET-CHECK:
    every subset row must be present in BOTH sides' store keys.
    """
    post = caches[prof.post_side]
    short = caches[prof.short_side]
    per_corpus: dict[str, dict] = {}
    usable: dict[str, list[str]] = {}
    meta_by_row: dict[str, dict] = {}
    for corpus in sorted(set(post) | set(short)):
        p_ids = set(post[corpus].row_ids) if corpus in post else set()
        s_ids = set(short[corpus].row_ids) if corpus in short else set()
        both = sorted(p_ids & s_ids)
        usable[corpus] = both
        per_corpus[corpus] = {
            "n_post": len(p_ids),
            "n_short": len(s_ids),
            "n_usable": len(both),
            "dropped_post_only": len(p_ids - s_ids),
            "dropped_short_only": len(s_ids - p_ids),
        }
        if corpus in post:
            for rid in both:
                meta_by_row[rid] = post[corpus].meta[post[corpus].pos[rid]]
    strata: dict[str, list[str]] = {s: [] for s in STRATA}
    for corpus, rows in usable.items():
        for rid in rows:
            s = stratum_of(meta_by_row[rid])
            if s is not None:
                strata[s].append(rid)
    for s in STRATA:
        strata[s] = sorted(strata[s])
    pooled = sorted({r for c in PER_CORPUS_CELLS for r in usable.get(c, [])})
    payload = {
        "arm": prof.arm,
        "per_corpus": per_corpus,
        "strata_n": {s: len(strata[s]) for s in STRATA},
        "pooled_n": len(pooled),
        "matched_n_rows": MATCHED_N_ROWS[prof.arm],
        "floor_rows": FLOOR_ROWS[prof.arm],
        "smoke": smoke,
        "repro": _repro("p5_rowsets"),
    }
    _atomic_json(out_root / "out" / "rowsets" / f"arm{prof.arm}.json", payload)
    return {"usable": usable, "strata": strata, "pooled": pooled, "meta": meta_by_row}


def subset_rows(rowsets: dict, subset: str) -> list[str]:
    if subset in STRATA:
        return rowsets["strata"][subset]
    if subset == "pooled":
        return rowsets["pooled"]
    if subset.startswith("corpus:"):
        return rowsets["usable"].get(subset.split(":", 1)[1], [])
    raise ValueError(f"unknown subset {subset!r}")


# ---------------------------------------------------------------------------
# Unit registry (plan §9 arithmetic; asserted)
# ---------------------------------------------------------------------------

# cell -> (x_side_role, x_kind, y_side_role, y_kind); roles: post|short
CELL_XY = {
    "p7_A": ("post", "cx_last", "post", "ans_mean"),
    "p7_B": ("post", "cx_last", "post", "cot_mean"),
    "p7_C": ("post", "cx_last", "post", "out_mean"),
    "p7_D": ("post", "cot_boundary", "post", "ans_mean"),
    "p7_Aoff": ("short", "cx_last", "short", "ans_mean"),
    "p8_E": ("short", "cx_last", "post", "ans_mean"),
    "p8_F": ("post", "cx_last", "post", "ans_mean"),
    "p8_G": ("short", "cx_last", "short", "ans_mean"),
    "p8_H": ("short", "cx_last", "post", "cot_mean"),
}

P7_SUBSETS = ("does", "doesnt") + tuple(f"corpus:{c}" for c in PER_CORPUS_CELLS)
P8_SUBSETS = ("pooled",) + tuple(f"corpus:{c}" for c in PER_CORPUS_CELLS) + ("does", "doesnt")


@dataclass(frozen=True)
class Unit:
    unit_id: str  # filename stem, ends __a{K}
    kind: str  # sweep | traj | ladder | operator | ood | reliability
    cell: str  # CELL_XY key for sweep units
    subset: str
    matched_n: int | None = None
    ood_score_subset: str | None = None


def _uid(parts: list[str], arm: int) -> str:
    return "__".join(parts + [f"a{arm}"])


def build_registry(prof: LayerProfile) -> list[Unit]:
    arm = prof.arm
    units: list[Unit] = []
    p7_cells = ("p7_A", "p7_B", "p7_C", "p7_D")
    for cell in p7_cells:
        for sub in P7_SUBSETS:
            slug = sub.replace("corpus:", "")
            units.append(Unit(_uid([cell, slug], arm), "sweep", cell, sub))
    if arm == 3:
        for sub in P7_SUBSETS:
            slug = sub.replace("corpus:", "")
            units.append(Unit(_uid(["p7_Aoff", slug], arm), "sweep", "p7_Aoff", sub))
    if prof.has_pre_model:
        for cell in ("p8_E", "p8_F", "p8_G", "p8_H"):
            for sub in P8_SUBSETS:
                slug = sub.replace("corpus:", "")
                units.append(Unit(_uid([cell, slug], arm), "sweep", cell, sub))
    # Matched-n companions (8/arm; plan §5 ctl_matchn): A/D x 2 strata at
    # min-strata-n (computed at run time -> matched_n=-1 sentinel), + cell A per
    # corpus at the validity-margin row count.
    for cell in ("p7_A", "p7_D"):
        for sub in STRATA:
            units.append(Unit(_uid([cell, sub, "matchn"], arm), "sweep", cell, sub, matched_n=-1))
    for c in PER_CORPUS_CELLS:
        units.append(
            Unit(
                _uid(["p7_A", c, "matchn"], arm),
                "sweep",
                "p7_A",
                f"corpus:{c}",
                matched_n=MATCHED_N_ROWS[arm],
            )
        )
    n_sweep = len(units)
    # Trajectory: ONE registry job per arm (both strata, 9 t x frozen layers,
    # batched over the position axis) = 18 statistical units.
    units.append(Unit(_uid(["p7_traj"], arm), "traj", "p7_traj", "strata"))
    if prof.has_pre_model:
        for sub in ("pooled",) + tuple(f"corpus:{c}" for c in PER_CORPUS_CELLS):
            slug = sub.replace("corpus:", "")
            units.append(Unit(_uid(["ladder", slug], arm), "ladder", "ladder", sub))
        units.append(Unit(_uid(["operator_comparison"], arm), "operator", "operator", "pooled"))
    # Exploratory OOD transfer arms (§6; NOT in the 222 count).
    units.append(
        Unit(
            _uid(["ood_gsm8k"], arm),
            "ood",
            "p7_A",
            "corpus:gsm8k_train",
            ood_score_subset="corpus:gsm8k_test",
        )
    )
    units.append(
        Unit(_uid(["ood_does2doesnt"], arm), "ood", "p7_A", "does", ood_score_subset="doesnt")
    )
    units.append(
        Unit(_uid(["ood_doesnt2does"], arm), "ood", "p7_A", "doesnt", ood_score_subset="does")
    )
    units.append(Unit(_uid(["reliability"], arm), "reliability", "reliability", "strata"))
    # Registry arithmetic asserts (plan §9): sweep units 24+28+8=60 (arms 1-2)
    # / 24+6+8=38 (arm 3); +ladder 5 and +operator on arms 1-2.
    expect_sweep = 60 if prof.has_pre_model else 38
    assert n_sweep == expect_sweep, (arm, n_sweep, expect_sweep)
    n_stat_units = n_sweep + 18 + (5 if prof.has_pre_model else 0)
    assert n_stat_units == (83 if prof.has_pre_model else 56), (arm, n_stat_units)
    return units


def registry_stat_totals() -> int:
    """Cross-arm statistical-unit total (plan §9: 65x2 + 38 + 54 = 222)."""
    total = 0
    for arm in (1, 2, 3):
        prof = profile_for_arm(arm)
        total += (60 + 5 if prof.has_pre_model else 38) + 18
    return total


# ---------------------------------------------------------------------------
# Answer-content identity (MF-4) — classes, template pools, retrieval reads
# ---------------------------------------------------------------------------


def answer_content_class(corpus: str, ans_text: str, row_id: str) -> str:
    """Canonical answer-content equivalence class (plan §6 MF-4).

    math family: normalized boxed content; MCQ: option letter; ContextHub:
    normalized native (last-line) answer. Unparseable answers get a singleton
    class (hits then require the exact row — conservative, never inflating).
    """
    if corpus in MATH_CORPORA:
        boxed = g25.extract_boxed(ans_text)
        return f"math:{g25._norm_math(boxed)}" if boxed is not None else f"__row__:{row_id}"
    if corpus in MCQ_CORPORA:
        m = g25._LETTER_RE.search(ans_text)
        return f"letter:{m.group(1)}" if m else f"__row__:{row_id}"
    if corpus == "contexthub":
        lines = [ln for ln in ans_text.strip().split("\n") if ln.strip()]
        return f"free:{g25._norm_free(lines[-1])}" if lines else f"__row__:{row_id}"
    raise ValueError(f"unknown corpus {corpus!r}")


def answer_template_hash(ans_text: str) -> str:
    """Digit-and-boxed-masked answer template key (plan §5: same-template pools)."""
    t = re.sub(r"\\boxed\s*\{[^{}]*\}", r"\\boxed{#}", ans_text)
    t = re.sub(r"\d+(?:\.\d+)?", "#", t)
    return hashlib.sha1(" ".join(t.split()).encode()).hexdigest()[:16]


class AnswerInfo:
    """Lazy per-(side, corpus) answer-text loader from persisted rollouts."""

    def __init__(self, out_root: Path, prof: LayerProfile, smoke: bool, prefill_fallback: bool):
        self.out_root = out_root
        self.prof = prof
        self.smoke = smoke
        self.prefill_fallback = prefill_fallback
        self._cache: dict[tuple[str, str], dict[str, str]] = {}

    def _side_spec(self, side_name: str):
        arm = g25.ARMS[self.prof.arm]
        for s in arm.sides:
            if s.side == side_name:
                return s
        raise KeyError(side_name)

    def _stage_if_missing(self, side, corpus: str) -> Path:
        p = self.out_root / "rollouts" / side.stage / f"{corpus}.jsonl"
        if p.is_file():
            return p
        stage = f"smoke_{side.stage}" if self.smoke else side.stage
        rel = f"{g25.RAW_PREFIX}/{stage}/{corpus}.jsonl"
        from huggingface_hub import hf_hub_download

        got = hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=hub.DEFAULT_DATASET_REPO,
                repo_type="dataset",
                filename=rel,
                local_dir=self.out_root / "fitstage_rollouts",
            ),
            what=f"p5 rollout stage {rel}",
        )
        return Path(got)

    def ans_texts(self, side_name: str, corpus: str) -> dict[str, str]:
        key = (side_name, corpus)
        if key in self._cache:
            return self._cache[key]
        side = self._side_spec(side_name)
        path = self._stage_if_missing(side, corpus)
        mode = g25.effective_parse_mode(side, self.prefill_fallback)
        out: dict[str, str] = {}
        for rec in g25._read_jsonl(path):
            parse = g25.parse_generation(rec, mode)
            if parse["well_formed"]:
                s, e = parse["ans_char_span"]
                out[rec["row_id"]] = rec["text"][s:e]
        self._cache[key] = out
        return out


def _content_retrieval_one_pool(
    pred: np.ndarray, pool: np.ndarray, classes: list[str], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query (hit, chance) under the content-identity rule for pool==queries."""
    d = mb._pairwise_dist(pred, pool, metric)
    nn = np.argmin(d, axis=1)
    cls = np.asarray(classes)
    hit = (cls[nn] == cls).astype(np.float64)
    _, inv, counts = np.unique(cls, return_inverse=True, return_counts=True)
    chance = counts[inv].astype(np.float64) / len(cls)  # m_i / n_pool
    return hit, chance


def content_retrieval(
    pred: np.ndarray,
    true: np.ndarray,
    row_ids: list[str],
    corpora: list[str],
    ans_by_row: dict[str, str],
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict:
    """MF-4 content-identity retrieval: within-corpus pools + same-template pools.

    Hits by canonical answer-content class; chance = per-query m_i/n_i and its
    within-pool mean (never bare 1/n_pool); chance-adjusted lift with a
    prompt-level bootstrap CI over covered rows.
    """
    n = len(row_ids)
    by_corpus: dict[str, list[int]] = {}
    for i, c in enumerate(corpora):
        by_corpus.setdefault(c, []).append(i)
    hit = np.full(n, np.nan)
    chance = np.full(n, np.nan)
    t_hit = np.full(n, np.nan)
    t_chance = np.full(n, np.nan)
    per_corpus: dict[str, dict] = {}
    for c, idxs in sorted(by_corpus.items()):
        idxs_a = np.asarray(idxs)
        have = [i for i in idxs if row_ids[i] in ans_by_row]
        if len(have) < 2:
            per_corpus[c] = {"status": "no_answer_texts", "n": len(idxs)}
            continue
        ha = np.asarray(have)
        classes = [answer_content_class(c, ans_by_row[row_ids[i]], row_ids[i]) for i in have]
        h, ch = _content_retrieval_one_pool(pred[ha], true[ha], classes, metric)
        hit[ha], chance[ha] = h, ch
        # Same-template pools (>=20 members) within the corpus.
        tpl = [answer_template_hash(ans_by_row[row_ids[i]]) for i in have]
        pools: dict[str, list[int]] = {}
        for j, t in enumerate(tpl):
            pools.setdefault(t, []).append(j)
        covered = 0
        for members in pools.values():
            if len(members) < TEMPLATE_POOL_MIN:
                continue
            mj = np.asarray(members)
            gi = ha[mj]
            th, tch = _content_retrieval_one_pool(
                pred[gi], true[gi], [classes[j] for j in members], metric
            )
            t_hit[gi], t_chance[gi] = th, tch
            covered += len(members)
        n_class = len(set(classes))
        per_corpus[c] = {
            "n": len(idxs),
            "n_scored": len(have),
            "acc_at_1": float(h.mean()),
            "chance_mean": float(ch.mean()),
            "lift": float(h.mean() - ch.mean()),
            "n_distinct_classes": n_class,
            "template_coverage": covered / max(1, len(have)),
            "n_dropped_no_pool": int(len(idxs_a) - len(have)),
        }

    def _agg(hv: np.ndarray, cv: np.ndarray, tag: str) -> dict:
        m = ~np.isnan(hv)
        if m.sum() < 2:
            return {"status": f"insufficient_{tag}_rows", "n": int(m.sum())}
        lift = hv[m] - cv[m]
        idx = la.draw_index_matrix(int(m.sum()), n_boot, seed=seed)
        w = la.counts_from_indices(idx, int(m.sum()))
        draws = (w @ lift) / w.sum(axis=1)
        return {
            "n": int(m.sum()),
            "acc_at_1": float(hv[m].mean()),
            "chance_mean": float(cv[m].mean()),
            "lift": float(lift.mean()),
            "lift_ci_lo": float(np.quantile(draws, 0.025)),
            "lift_ci_hi": float(np.quantile(draws, 0.975)),
            "lift_ci_excludes_0": bool(
                np.quantile(draws, 0.025) > 0.0 or np.quantile(draws, 0.975) < 0.0
            ),
            "n_boot": n_boot,
        }

    return {
        "metric": metric,
        "hit_rule": "canonical answer-content identity (MF-4)",
        "corpus_pool": _agg(hit, chance, "corpus"),
        "same_template_pool": {
            **_agg(t_hit, t_chance, "template"),
            "pool_min_members": TEMPLATE_POOL_MIN,
            "coverage": float(np.mean(~np.isnan(t_hit))),
        },
        "per_corpus": per_corpus,
    }


# ---------------------------------------------------------------------------
# Fit-unit runners
# ---------------------------------------------------------------------------


def _repro(phase: str) -> dict:
    meta = as_metadata_dict(git_provenance(), phase=phase)
    meta.update(
        {
            "task": TASK,
            "recipe_rev": RECIPE_REV,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
        }
    )
    return meta


def _fit_params(smoke: bool, null_draws: int, n_boot: int) -> dict:
    return {
        "lambdas": list(LAMBDAS_N1M_PARAMS),
        "n_folds": N_FOLDS,
        "seed": FIT_SEED,
        "inner_lambda_folds": INNER_LAMBDA_FOLDS,
        "null_draws": null_draws,
        "n_boot": n_boot,
        "smoke": bool(smoke),
        "recipe_rev": RECIPE_REV,
    }


def _fingerprint(unit: Unit, params: dict) -> str:
    body = json.dumps({"unit": unit.unit_id, **params}, sort_keys=True)
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def _subset_seed(arm: int, subset: str, matched_n: int | None) -> int:
    return zlib.crc32(f"{arm}:{subset}:{matched_n}".encode()) % (2**31)


def _role_side(prof: LayerProfile, role: str) -> str:
    return prof.post_side if role == "post" else prof.short_side


def _load_xy(
    caches, prof: LayerProfile, cell: str, rows: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, Y, corpora) fp32 (n, L, D) for one sweep cell over the given rows.

    SET-CHECK: every registered row must be present in BOTH the X and Y stems'
    store keys (plan §3) — asserted per corpus before any slice.
    """
    xr, xk, yr, yk = CELL_XY[cell]
    xs, ys = _role_side(prof, xr), _role_side(prof, yr)
    # Row ids carry no corpus prefix — resolve corpus membership via the caches.
    corpus_of: dict[str, str] = {}
    for corpus, cache in caches[xs].items():
        for rid in rows:
            if rid in cache.pos:
                corpus_of[rid] = corpus
    missing_x = [r for r in rows if r not in corpus_of]
    assert not missing_x, f"{cell}: {len(missing_x)} registered rows absent from X side {xs}"
    n = len(rows)
    pos_of = {r: i for i, r in enumerate(rows)}
    X: np.ndarray | None = None
    Y: np.ndarray | None = None
    corp = [""] * n
    for corpus in sorted(set(corpus_of.values())):
        crows = [r for r in rows if corpus_of[r] == corpus]
        cx = caches[xs][corpus]
        cy = caches[ys].get(corpus)
        assert cy is not None, f"{cell}: corpus {corpus} missing from Y side {ys}"
        miss_y = [r for r in crows if r not in cy.pos]
        assert not miss_y, f"{cell}: {len(miss_y)} rows absent from Y side {ys}/{corpus}"
        xa = cx.kind_rows(xk, crows)
        ya = cy.kind_rows(yk, crows)
        if X is None:
            X = np.empty((n, *xa.shape[1:]), dtype=np.float32)
            Y = np.empty((n, *ya.shape[1:]), dtype=np.float32)
        dest = np.asarray([pos_of[r] for r in crows])
        X[dest] = xa
        Y[dest] = ya
        for r in crows:
            corp[pos_of[r]] = corpus
    assert X is not None and Y is not None
    return X, Y, corp


def _identity_bias_r2(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, layers) -> dict:
    """Held-out identity+learned-bias baseline per layer (mb.identity_bias_predict)."""
    out = {}
    for li in layers:
        if li >= X.shape[1]:
            continue
        ss_res, ss_tot = 0.0, 0.0
        for k in range(N_FOLDS):
            te, tr = folds == k, folds != k
            if te.sum() == 0 or tr.sum() < 3:
                continue
            pred = mb.identity_bias_predict(X[tr, li], Y[tr, li], X[te, li])
            true = Y[te, li].astype(np.float64)
            ss_res += float(np.sum((true - pred) ** 2))
            ss_tot += float(np.sum((true - true.mean(0)) ** 2))
        out[str(int(li))] = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return out


def _persist_preds(path: Path, preds: dict[int, np.ndarray], fitted, conv_ids, folds) -> dict:
    """fp16 preds npz (plain savez — compression OFF for Xet, #813) + sha manifest.

    Finiteness-checked after the fp16 cast; a cell whose preds overflow fp16
    falls back to fp32 (recorded) — never silent corruption (#825 parity note).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "fitted_mask": np.asarray(fitted, dtype=bool),
        "conv_ids": np.asarray(conv_ids, dtype=str),
        "folds": np.asarray(folds, dtype=np.int64),
    }
    dtype_used = "float16"
    for li, arr in preds.items():
        a16 = arr.astype(np.float16)
        if not np.isfinite(a16[np.asarray(fitted, dtype=bool)]).all():
            dtype_used = "float32"
            break
    for li, arr in preds.items():
        arrays[f"pred_l{li}"] = arr.astype(dtype_used)
    tmp = path.with_name(path.stem + ".tmp.npz")  # suffix stays .npz (gotchas #1092)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return {
        "path": str(path),
        "dtype": dtype_used,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "layers": sorted(int(k) for k in preds),
    }


def run_sweep_unit(
    unit: Unit,
    prof: LayerProfile,
    caches,
    rowsets: dict,
    ans_info: AnswerInfo,
    out_root: Path,
    *,
    smoke: bool,
    null_draws: int,
    n_boot: int,
) -> dict:
    rows = subset_rows(rowsets, unit.subset)
    matched_n = unit.matched_n
    if matched_n == -1:  # min-strata-n companion
        matched_n = min(len(rowsets["strata"][s]) for s in STRATA)
    floor = FLOOR_ROWS[prof.arm]
    if matched_n is not None and len(rows) >= matched_n:
        rng = np.random.default_rng(_subset_seed(prof.arm, unit.subset, matched_n))
        keep = np.sort(rng.choice(len(rows), size=matched_n, replace=False))
        rows = [rows[i] for i in keep]
    floor_status = "ok"
    if len(rows) < floor:
        if smoke:
            floor_status = "smoke-demoted"  # #1345: production-n floors demoted under smoke
            if len(rows) < SMOKE_MIN_ROWS:
                return {"status": "dropped_below_floor", "n_rows": len(rows), "floor": floor}
        else:
            # Runtime attrition below the registered floor: REPORTED drop (plan MF-2).
            return {"status": "dropped_below_floor", "n_rows": len(rows), "floor": floor}
    X, Y, corpora = _load_xy(caches, prof, unit.cell, rows)
    d = X.shape[2]
    conv_ids = np.asarray(rows)
    fl = prof.frozen
    sweep = fc.heldout_r2_sweep(
        X,
        Y,
        conv_ids,
        n_folds=N_FOLDS,
        seed=FIT_SEED,
        null_draws=null_draws,
        collect_cosines=True,
        collect_lambdas=True,
        frozen_layers=fl,
        lambdas=LAMBDAS_N1M,
    )
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    folds, fitted = sweep["folds"], sweep["fitted_mask"]
    n_train_min = sweep["n_train_min"]
    if not smoke:
        # Fit-eligibility: n_train >= 1.2d asserted (plan MF-2; int() matches the
        # plan's 4,301 / 4,915 floors at d=3584 / 4096).
        assert n_train_min is not None and n_train_min >= int(1.2 * d), (
            unit.unit_id,
            n_train_min,
            d,
        )
    hl = prof.headline
    sel = fc.selection_symmetric_summary(r2_obs, r2_null, frozen_layers=fl)
    rp = fc.random_projection_control(
        X, Y, conv_ids, layers=list(fl), n_folds=N_FOLDS, seed=FIT_SEED
    )
    mean_base = fc.mean_baseline_r2(Y, conv_ids, layers=list(fl), n_folds=N_FOLDS, seed=FIT_SEED)
    idb = _identity_bias_r2(X, Y, folds, fl)
    per_frozen: dict[str, dict] = {}
    for li in fl:
        if li >= X.shape[1]:
            continue
        cos = sweep["cosines"][li][fitted]
        pred = sweep["preds_frozen"][li][fitted]
        true = Y[fitted, li, :]
        per_frozen[str(li)] = {
            "r2": float(r2_obs[li]),
            "cosine_ci": fc.bootstrap_ci(cos, n_boot=n_boot, seed=FIT_SEED + li),
            "r2_ci": fc.bootstrap_r2_ci(pred, true, n_boot=n_boot, seed=FIT_SEED + 100 + li),
        }
    # Headline paired bootstrap draws (shared per subset -> paired across cells).
    n_fit = int(fitted.sum())
    sseed = _subset_seed(prof.arm, unit.subset, matched_n)
    idx = la.draw_index_matrix(n_fit, n_boot, seed=sseed)
    w = la.counts_from_indices(idx, n_fit)
    pred_h = sweep["preds_frozen"][hl][fitted]
    true_h = Y[fitted, hl, :]
    r2_draws = la.weighted_r2_draws(pred_h, true_h, w)
    # Mandatory baselines pair: identity kNN retrieval (both metrics) at headline.
    knn_identity = {
        m: mb.knn_retrieval(pred_h, true_h, ks=(1,), metric=m) for m in ("euclidean", "cosine")
    }
    # MF-4 content-identity retrieval (ans_mean targets only).
    knn_content: dict[str, dict] | None = None
    y_role, y_kind = CELL_XY[unit.cell][2], CELL_XY[unit.cell][3]
    if y_kind == "ans_mean":
        y_side = _role_side(prof, y_role)
        fit_rows = [r for r, f in zip(rows, fitted) if f]
        fit_corp = [c for c, f in zip(corpora, fitted) if f]
        ans_by_row: dict[str, str] = {}
        for c in sorted(set(fit_corp)):
            ans_by_row.update(ans_info.ans_texts(y_side, c))
        knn_content = {
            m: content_retrieval(
                pred_h,
                true_h,
                fit_rows,
                fit_corp,
                ans_by_row,
                metric=m,
                n_boot=n_boot,
                seed=sseed + (1 if m == "euclidean" else 2),
            )
            for m in ("euclidean", "cosine")
        }
    ec = knn_content["euclidean"]["corpus_pool"] if knn_content else {}
    tripwire = fc.degeneracy_tripwire(
        n_train=n_train_min,
        d=d,
        selected_lambdas=[
            float(v) for v in np.asarray(sweep["gcv_lambda"], dtype=float).ravel() if np.isfinite(v)
        ],
        r2_heldout=float(r2_obs[hl]),
        knn_at_1=ec.get("acc_at_1"),
        knn_chance=ec.get("chance_mean"),
        grid=LAMBDAS_N1M,
    )
    preds_manifest = _persist_preds(
        out_root / "out" / "preds" / f"{unit.unit_id}.npz",
        {int(li): sweep["preds_frozen"][li] for li in fl if li < X.shape[1]},
        fitted,
        rows,
        folds,
    )
    lam = sweep["gcv_lambda"]
    lam_json = (
        None
        if lam is None
        else [[None if np.isnan(v) else float(v) for v in row] for row in np.asarray(lam)]
    )
    return {
        "status": "ok",
        "unit_id": unit.unit_id,
        "arm": prof.arm,
        "cell": unit.cell,
        "cell_xy": dict(zip(("x_role", "x_kind", "y_role", "y_kind"), CELL_XY[unit.cell])),
        "subset": unit.subset,
        "matched_n": matched_n,
        "n_rows": len(rows),
        "n_fitted": n_fit,
        "n_train_min": n_train_min,
        "d": d,
        "floor_check": floor_status,
        "headline_layer": hl,
        "frozen_layers": list(fl),
        "r2_per_layer": [float(v) for v in r2_obs],
        "r2_headline": float(r2_obs[hl]),
        "selection_symmetric": sel,
        "per_frozen": per_frozen,
        "identity_bias_r2": idb,
        "mean_baseline_r2": mean_base,
        "random_projection_r2": rp,
        "knn_identity": knn_identity,
        "knn_content": knn_content,
        "r2_headline_bootstrap": {
            "draws": [float(v) for v in r2_draws],
            "ci_lo": float(np.nanquantile(r2_draws, 0.025)),
            "ci_hi": float(np.nanquantile(r2_draws, 0.975)),
            "subset_seed": sseed,
        },
        "lambda_diag": {
            "selected": lam_json,
            "selector": sweep["lambda_selector"],
        },
        "reduced_basis": sweep["reduced_basis"],
        "degeneracy_tripwire": tripwire,
        "r2_ceiling_normalized": None,
        "ceiling_status": "missing_reliability_capture",
        "preds": preds_manifest,
    }


def run_traj_unit(
    unit: Unit,
    prof: LayerProfile,
    caches,
    rowsets: dict,
    ans_info: AnswerInfo,
    out_root: Path,
    *,
    smoke: bool,
    null_draws: int,
    n_boot: int,
) -> dict:
    """9 interior t-kinds x 2 strata vs the SAME target v_A* (ans_mean), batched
    over the position axis: one heldout_r2_sweep per stratum with a 27-wide
    (t x frozen-layer) pseudo-layer axis — never a per-position loop."""
    post = caches[prof.post_side]
    n_t = len(g25.T_GRID)
    n_f = len(prof.frozen)
    hl_fi = prof.frozen.index(prof.headline)
    strata_out: dict[str, dict] = {}
    for stratum in STRATA:
        rows = rowsets["strata"][stratum]
        floor = FLOOR_ROWS[prof.arm]
        if len(rows) < (SMOKE_MIN_ROWS if smoke else floor):
            strata_out[stratum] = {
                "status": "dropped_below_floor",
                "n_rows": len(rows),
                "floor": floor,
            }
            continue
        corpus_of = {}
        for corpus, cache in post.items():
            for rid in rows:
                if rid in cache.pos and cache.has_tk:
                    corpus_of[rid] = corpus
        rows = [r for r in rows if r in corpus_of]
        n = len(rows)
        X27 = np.empty((n, n_t * n_f, prof.hidden), dtype=np.float32)
        Y27 = np.empty_like(X27)
        short_think = np.zeros(n, dtype=bool)
        pos = 0
        for corpus in sorted(set(corpus_of.values())):
            crows = [r for r in rows if corpus_of[r] == corpus]
            cache = post[corpus]
            tk = cache.kind_rows("tk", crows)  # (nc, 9, 3, H)
            ans = cache.kind_rows("ans_mean", crows)  # (nc, L, H)
            for ti in range(n_t):
                for fi, layer in enumerate(prof.frozen):
                    X27[pos : pos + len(crows), ti * n_f + fi] = tk[:, ti, fi]
                    Y27[pos : pos + len(crows), ti * n_f + fi] = ans[:, layer]
            for j, rid in enumerate(crows):
                short_think[pos + j] = bool(cache.meta[cache.pos[rid]].get("short_think", False))
            pos += len(crows)
        assert pos == n
        headline_idx = tuple(ti * n_f + hl_fi for ti in range(n_t))
        sweep = fc.heldout_r2_sweep(
            X27,
            Y27,
            np.asarray(rows),
            n_folds=N_FOLDS,
            seed=FIT_SEED,
            null_draws=null_draws,
            collect_cosines=True,
            collect_lambdas=True,
            frozen_layers=headline_idx,
            lambdas=LAMBDAS_N1M,
        )
        fitted = sweep["fitted_mask"]
        n_fit = int(fitted.sum())
        sseed = _subset_seed(prof.arm, stratum, None)
        idx = la.draw_index_matrix(n_fit, n_boot, seed=sseed)
        w = la.counts_from_indices(idx, n_fit)
        r2_grid = np.asarray(sweep["r2_obs"]).reshape(n_t, n_f)
        per_t: dict[str, dict] = {}
        y_side = prof.post_side
        fit_rows = [r for r, f in zip(rows, fitted) if f]
        fit_corp = [corpus_of[r] for r in fit_rows]
        ans_by_row: dict[str, str] = {}
        for c in sorted(set(fit_corp)):
            ans_by_row.update(ans_info.ans_texts(y_side, c))
        for ti, t in enumerate(g25.T_GRID):
            pl = ti * n_f + hl_fi
            pred_h = sweep["preds_frozen"][pl][fitted]
            true_h = Y27[fitted, pl, :]
            draws = la.weighted_r2_draws(pred_h, true_h, w)
            knn_c = content_retrieval(
                pred_h,
                true_h,
                fit_rows,
                fit_corp,
                ans_by_row,
                metric="euclidean",
                n_boot=n_boot,
                seed=sseed + 10 + ti,
            )
            per_t[f"t{int(round(t * 100))}"] = {
                "r2_per_frozen_layer": {
                    str(prof.frozen[fi]): float(r2_grid[ti, fi]) for fi in range(n_f)
                },
                "r2_headline": float(r2_grid[ti, hl_fi]),
                "r2_headline_bootstrap": {
                    "draws": [float(v) for v in draws],
                    "ci_lo": float(np.nanquantile(draws, 0.025)),
                    "ci_hi": float(np.nanquantile(draws, 0.975)),
                },
                "identity_bias_r2_headline": _identity_bias_r2(
                    X27[:, [pl], :], Y27[:, [pl], :], sweep["folds"], [0]
                )["0"],
                "knn_identity_euclidean": mb.knn_retrieval(
                    pred_h, true_h, ks=(1,), metric="euclidean"
                ),
                "knn_content_euclidean": knn_c,
            }
        preds_manifest = _persist_preds(
            out_root / "out" / "preds" / f"p7_traj__{stratum}__a{prof.arm}.npz",
            {int(pl): sweep["preds_frozen"][pl] for pl in headline_idx},
            fitted,
            rows,
            sweep["folds"],
        )
        strata_out[stratum] = {
            "status": "ok",
            "n_rows": n,
            "n_fitted": n_fit,
            "n_train_min": sweep["n_train_min"],
            "short_think_frac": float(short_think.mean()),
            "subset_seed": sseed,
            "per_t": per_t,
            "pseudo_layer_layout": "index = t_idx * n_frozen + frozen_idx (t-major)",
            "preds": preds_manifest,
            "endpoints_note": (
                "t=0 == cell p7_A, t=1 == cell p7_D on the same stratum subset "
                "(their unit JSONs carry the endpoint reads)"
            ),
        }
        del X27, Y27, sweep
        _pygc.collect()
    return {
        "status": "ok",
        "unit_id": unit.unit_id,
        "arm": prof.arm,
        "target": "ans_mean (v_A*, fixed across t)",
        "t_grid": list(g25.T_GRID),
        "frozen_layers": list(prof.frozen),
        "headline_layer": prof.headline,
        "strata": strata_out,
        "ceiling_status": "missing_reliability_capture",
    }


# --- Ladder (MF-5) ---------------------------------------------------------


def _pooled_r2_np(pred: np.ndarray, true: np.ndarray) -> float:
    return fc._pooled_r2(np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64))


def run_ladder_unit(
    unit: Unit,
    prof: LayerProfile,
    caches,
    rowsets: dict,
    out_root: Path,
    *,
    smoke: bool,
    n_boot: int,
) -> dict:
    """Reparameterization ladder tiers 0-8 (docs/mapping_similarity_metrics.md)
    on (M_pre = cell-G map, M_post = cell-F map) + the run_pair fold-read
    battery (ma._ridge_prep/_orth_fit/_fold_reads/_assemble_battery verbatim).

    Per-corpus band (MF-5): band_value = DELTA_ELICIT_BAND x rate with
    rate = R2_F(this subset) / 0.6731, recomputed from THIS corpus's own
    within-post reference read; band_source_corpus == ladder corpus asserted.
    """
    rows = subset_rows(rowsets, unit.subset)
    floor = FLOOR_ROWS[prof.arm]
    if len(rows) < (SMOKE_MIN_ROWS if smoke else floor):
        return {"status": "dropped_below_floor", "n_rows": len(rows), "floor": floor}
    hl = prof.headline
    Xb_a, Yb_a, _ = _load_xy(caches, prof, "p8_G", rows)  # pre cx -> pre ans
    Xi_a, Yi_a, _ = _load_xy(caches, prof, "p8_F", rows)  # post cx -> post ans
    dev = fc._fit_device()
    dt = torch.float64
    Xb = torch.as_tensor(Xb_a[:, hl, :], dtype=dt).to(dev)
    Yb = torch.as_tensor(Yb_a[:, hl, :], dtype=dt).to(dev)
    Xi = torch.as_tensor(Xi_a[:, hl, :], dtype=dt).to(dev)
    Yi = torch.as_tensor(Yi_a[:, hl, :], dtype=dt).to(dev)
    del Xb_a, Yb_a, Xi_a, Yi_a
    _pygc.collect()
    n = len(rows)
    folds = fc._cv_folds(np.asarray(rows), N_FOLDS, FIT_SEED)
    tier_res = {t: 0.0 for t in LADDER_TIER_NAMES}
    tier_tot = {t: 0.0 for t in LADDER_TIER_NAMES}
    ss_res: dict[str, float] = {}
    ss_tot: dict[str, float] = {}
    cap_f = np.zeros((n, Yi.shape[1]), dtype=np.float32)
    cap_t8 = np.zeros((n, Yi.shape[1]), dtype=np.float32)
    fitted = np.zeros(n, dtype=bool)
    for k in range(N_FOLDS):
        tr_np, te_np = folds != k, folds == k
        if te_np.sum() == 0 or tr_np.sum() < 3:
            continue
        tr, te = torch.as_tensor(tr_np), torch.as_tensor(te_np)
        preps = {
            "Xb": ma._ridge_prep(Xb[tr]),
            "Xi": ma._ridge_prep(Xi[tr]),
            "Yb": ma._ridge_prep(Yb[tr]),
            "Yi": ma._ridge_prep(Yi[tr]),
        }
        orth = {"ctx": ma._orth_fit(Xb[tr], Xi[tr]), "ans": ma._orth_fit(Yb[tr], Yi[tr])}
        reads = ma._fold_reads(preps, orth, (Xi, Yi, Xb, Yb), tr, te, do_orth=True)
        for name, (pred, true) in reads.items():
            ss_res[name] = ss_res.get(name, 0.0) + float(((true - pred) ** 2).sum())
            ss_tot[name] = ss_tot.get(name, 0.0) + float(((true - true.mean(0)) ** 2).sum())
        cap_f[te_np] = reads["ceil.within_instruct"][0].float().cpu().numpy()
        cap_t8[te_np] = reads["comp.linear.comp_samefn_b2i"][0].float().cpu().numpy()
        fitted[te_np] = True
        # M_pre applied to target rows (train and test) — the frozen source map.
        p_tr = ma._ridge_predict(preps["Xb"], Yb[tr], Xi[tr])
        p_te = ma._ridge_predict(preps["Xb"], Yb[tr], Xi[te])
        y_tr, y_te = Yi[tr], Yi[te]
        preds_t: dict[str, torch.Tensor] = {}
        preds_t["t0_direct_transfer"] = p_te
        dx = Xi[tr].mean(0) - Xb[tr].mean(0)
        preds_t["t1_context_offset"] = ma._ridge_predict(preps["Xb"], Yb[tr], Xi[te] - dx)
        dy = y_tr.mean(0) - Yb[tr].mean(0)
        preds_t["t2_answer_offset"] = p_te + dy
        bstar = (y_tr - p_tr).mean(0)
        preds_t["t3_bias_offset"] = p_te + bstar
        pc, yc = p_tr - p_tr.mean(0), y_tr - y_tr.mean(0)
        alpha = float((pc * yc).sum() / ((pc * pc).sum() + 1e-12))
        preds_t["t4_global_scaling"] = alpha * (p_te - p_tr.mean(0)) + y_tr.mean(0)
        u, _s, vh = torch.linalg.svd(pc.T @ yc, full_matrices=False)
        rot = u @ vh
        preds_t["t5_mapping_rotation"] = (p_te - p_tr.mean(0)) @ rot + y_tr.mean(0)
        xbhat_tr = ma._ridge_predict(preps["Xi"], Xb[tr], Xi[tr])
        xbhat_te = ma._ridge_predict(preps["Xi"], Xb[tr], Xi[te])
        p6_tr = ma._ridge_predict(preps["Xb"], Yb[tr], xbhat_tr)
        p6_te = ma._ridge_predict(preps["Xb"], Yb[tr], xbhat_te)
        preds_t["t6_reparam_contexts"] = p6_te + (y_tr - p6_tr).mean(0)
        preds_t["t7_reparam_answers"] = ma._ridge_predict(preps["Yb"], y_tr, p_te)
        preds_t["t8_reparam_both"] = ma._ridge_predict(preps["Yb"], y_tr, p6_te)
        mu = y_te.mean(0)
        for tname, pred in preds_t.items():
            tier_res[tname] += float(((y_te - pred) ** 2).sum())
            tier_tot[tname] += float(((y_te - mu) ** 2).sum())
        del preps, orth, reads, preds_t
    battery = ma._assemble_battery(ss_res, ss_tot)
    r2_f = battery["ceilings"]["within_instruct"]
    tiers = {
        t: (float("nan") if tier_tot[t] < 1e-12 else 1.0 - tier_res[t] / tier_tot[t])
        for t in LADDER_TIER_NAMES
    }
    rate = float(r2_f) / COMMITTED_ANCHOR_R2
    band_value = DELTA_ELICIT_BAND * rate
    band_source_corpus = unit.subset.replace("corpus:", "")
    ladder_corpus = unit.subset.replace("corpus:", "")
    assert band_source_corpus == ladder_corpus, (band_source_corpus, ladder_corpus)  # MF-5
    sufficient = next(
        (i for i, t in enumerate(LADDER_TIER_NAMES) if tiers[t] >= float(r2_f) - band_value),
        None,
    )
    y_np = Yi.float().cpu().numpy()
    sseed = _subset_seed(prof.arm, unit.subset, None)
    idx = la.draw_index_matrix(int(fitted.sum()), n_boot, seed=sseed)
    w = la.counts_from_indices(idx, int(fitted.sum()))
    boot = la.paired_bootstrap_batched(cap_f[fitted], y_np[fitted], cap_t8[fitted], y_np[fitted], w)
    gap_ci = {
        "ci_lo": float(np.nanquantile(boot["delta"], 0.025)),
        "ci_hi": float(np.nanquantile(boot["delta"], 0.975)),
        "n_draws": n_boot,
    }
    return {
        "status": "ok",
        "unit_id": unit.unit_id,
        "arm": prof.arm,
        "subset": unit.subset,
        "ladder_corpus": ladder_corpus,
        "n_rows": n,
        "headline_layer": hl,
        "tiers_r2": tiers,
        "tier_names": list(LADDER_TIER_NAMES),
        "within_post_reference_r2": float(r2_f),
        "sufficient_tier": sufficient,
        "band_value": band_value,
        "band_source_corpus": band_source_corpus,
        "band_rule": "DELTA_ELICIT_BAND(0.02) x rate; rate = R2_F(subset)/0.6731 (#1336 anchor)",
        "exchange_rate": rate,
        "committed_anchor_r2": COMMITTED_ANCHOR_R2,
        "prior_pooled_band_anchor": 0.0207,
        "battery": battery,
        "gap_f_vs_t8_bootstrap": gap_ci,
        "battery_provenance": "ma._ridge_prep/_orth_fit/_fold_reads/_assemble_battery (run_pair core)",
    }


def run_operator_unit(unit: Unit, prof: LayerProfile, caches, rowsets: dict, smoke: bool) -> dict:
    """#1345 operator battery on (M_pre, M_post): direction-aware raw cosine with
    rotation null vs rotation-invariant-only spectrum cosine (labeled)."""
    rows = subset_rows(rowsets, unit.subset)
    if smoke and len(rows) < SMOKE_MIN_ROWS:
        return {"status": "dropped_below_floor", "n_rows": len(rows)}
    hl = prof.headline
    Xb, Yb, _ = _load_xy(caches, prof, "p8_G", rows)
    Xi, Yi, _ = _load_xy(caches, prof, "p8_F", rows)
    beta_pre, lam_pre = xm.fit_primal_beta(Xb[:, hl, :], Yb[:, hl, :])
    beta_post, lam_post = xm.fit_primal_beta(Xi[:, hl, :], Yi[:, hl, :])
    beta_pre_t = beta_pre.detach().to("cpu", torch.float64)
    beta_post_t = beta_post.detach().to("cpu", torch.float64)
    n_draws = 5 if smoke else N_ROT_DRAWS
    raw = oc.raw_cosine_with_rotation_null(beta_pre_t, beta_post_t, n_draws=n_draws, seed=TASK)
    spec = oc.spectrum_cosine(beta_pre_t, beta_post_t)
    dev = fc._fit_device()
    cap = oc.alignment_capacity(
        torch.as_tensor(Xb[:, hl, :], dtype=torch.float64).to(dev),
        torch.as_tensor(Xi[:, hl, :], dtype=torch.float64).to(dev),
    )
    return {
        "status": "ok",
        "unit_id": unit.unit_id,
        "arm": prof.arm,
        "subset": unit.subset,
        "n_rows": len(rows),
        "headline_layer": hl,
        "operators": "M_pre = cell-G full-data primal beta; M_post = cell-F (same rows)",
        "lambda_pre": float(lam_pre),
        "lambda_post": float(lam_post),
        "direction_aware": {"raw_cosine_with_rotation_null": raw},
        "rotation_invariant_only": {
            "spectrum_cosine": spec,
            "label": "can never support 'same operator up to rotation' (plan MF-5)",
        },
        "ctx_alignment_capacity": cap,
    }


def run_ood_unit(unit: Unit, prof: LayerProfile, caches, rowsets: dict, smoke: bool) -> dict:
    """Exploratory group-scoped transfer: fit on one row pool, score on another
    (decontaminated GSM8K arm + the two cross-stratum arms; plan §6)."""
    fit_rows = subset_rows(rowsets, unit.subset)
    score_rows = subset_rows(rowsets, unit.ood_score_subset)
    if len(fit_rows) < SMOKE_MIN_ROWS or len(score_rows) < 2:
        return {
            "status": "dropped_below_floor",
            "n_fit": len(fit_rows),
            "n_score": len(score_rows),
        }
    hl = prof.headline
    Xf, Yf, _ = _load_xy(caches, prof, unit.cell, fit_rows)
    Xs, Ys, _ = _load_xy(caches, prof, unit.cell, score_rows)
    dev = fc._fit_device()
    dt = torch.float64
    prep = ma._ridge_prep(torch.as_tensor(Xf[:, hl, :], dtype=dt).to(dev))
    pred = (
        ma._ridge_predict(
            prep,
            torch.as_tensor(Yf[:, hl, :], dtype=dt).to(dev),
            torch.as_tensor(Xs[:, hl, :], dtype=dt).to(dev),
        )
        .float()
        .cpu()
        .numpy()
    )
    true = Ys[:, hl, :]
    knn = {m: mb.knn_retrieval(pred, true, ks=(1,), metric=m) for m in ("euclidean", "cosine")}
    return {
        "status": "ok",
        "label": "EXPLORATORY (plan §6 OOD folds; not in the 222-unit registry)",
        "unit_id": unit.unit_id,
        "arm": prof.arm,
        "cell": unit.cell,
        "fit_subset": unit.subset,
        "score_subset": unit.ood_score_subset,
        "n_fit": len(fit_rows),
        "n_score": len(score_rows),
        "headline_layer": hl,
        "transfer_r2": _pooled_r2_np(pred, true),
        "knn_identity": knn,
    }


def run_reliability_unit(unit: Unit, prof: LayerProfile, caches, rowsets: dict) -> dict:
    """Split-half (2/2) target consistency per stratum (#779 convention) from
    reliability capture stems ``rel_{side}__{corpus}`` (the P4b
    capture-reliability leg's frozen-layer per-draw shards); LOUD status
    artifact when the stems are absent (P4b not yet run)."""
    rel_stems = caches.get(f"rel_{prof.post_side}", {})
    if not rel_stems:
        return {
            "status": "missing_reliability_capture",
            "unit_id": unit.unit_id,
            "arm": prof.arm,
            "expected_stems": f"store/arm{prof.arm}/rel_{prof.post_side}__{{corpus}}",
            "note": (
                "reliability T=0.6 draws are persisted as rollout TEXT only "
                f"(rollouts/reliability_a{prof.arm}); run issue2546_gen_capture.py "
                "--phase capture-reliability (P4b) to teacher-force the draws into "
                "frozen-layer rel_ stems — cell JSONs carry "
                "ceiling_status=missing_reliability_capture until it lands"
            ),
        }
    out: dict[str, dict] = {}
    for stratum in STRATA:
        rows = set(rowsets["strata"][stratum])
        halves: dict[str, list[np.ndarray]] = {"h1": [], "h2": []}
        for corpus, cache in sorted(rel_stems.items()):
            by_prompt: dict[str, dict[int, int]] = {}
            for i, rid in enumerate(cache.row_ids):
                meta = cache.meta[i]
                base = meta.get("base_row_id", rid.split("#", 1)[0])
                if base not in rows:
                    continue
                by_prompt.setdefault(base, {})[int(meta.get("draw", 0))] = i
            complete = {b: d for b, d in by_prompt.items() if len(d) >= 4}
            if not complete:
                continue
            ans = torch.load(cache.dir / "ans_mean.pt", map_location="cpu", weights_only=True)
            layers = getattr(cache, "layers", None)
            if layers:
                # P4b rel stems carry the FROZEN subset — map the headline layer
                # BY INDEX into the stem's own layer list (never absolute).
                assert prof.headline in layers, (corpus, layers, prof.headline)
                hlv = layers.index(prof.headline)
            else:  # legacy full-depth cache without the layers sidecar
                hlv = prof.headline
            for _base, draws in sorted(complete.items()):
                idx = [draws[k] for k in sorted(draws)[:4]]
                a = ans[idx][:, hlv, :].to(torch.float32).numpy()
                halves["h1"].append(a[:2].mean(0))
                halves["h2"].append(a[2:4].mean(0))
        if len(halves["h1"]) < 3:
            out[stratum] = {"status": "insufficient_prompts", "n": len(halves["h1"])}
            continue
        h1 = np.stack(halves["h1"])
        h2 = np.stack(halves["h2"])
        r_ab = _pooled_r2_np(h1, h2)
        r_ba = _pooled_r2_np(h2, h1)
        r = float(np.mean([r_ab, r_ba]))
        out[stratum] = {
            "status": "ok",
            "n_prompts": len(h1),
            "split_half_r2": r,
            "split_half_r2_ab": r_ab,
            "split_half_r2_ba": r_ba,
            "ceiling_spearman_brown": (2 * r / (1 + r)) if r > -1.0 else float("nan"),
            "convention": "#779 4-draw split-half (2/2) at the headline layer",
        }
    return {"status": "ok", "unit_id": unit.unit_id, "arm": prof.arm, "per_stratum": out}


# ---------------------------------------------------------------------------
# Orchestration: parent (cache + rowsets + fan-out) and worker (--shard i/N)
# ---------------------------------------------------------------------------


def unit_out_path(out_root: Path, unit: Unit) -> Path:
    sub = (
        "ladder"
        if unit.kind in ("ladder", "operator")
        else ("reliability" if unit.kind == "reliability" else "cells")
    )
    return out_root / "out" / sub / f"{unit.unit_id}.json"


def run_units(args, prof: LayerProfile, units: list[Unit]) -> int:
    out_root = Path(args.out_root)
    smoke = bool(args.smoke)
    null_draws = (
        args.null_draws
        if args.null_draws is not None
        else (SMOKE_NULL_DRAWS if smoke else N_NULL_DRAWS)
    )
    n_boot = args.n_boot if args.n_boot is not None else (SMOKE_N_BOOT if smoke else N_BOOT)
    params = _fit_params(smoke, null_draws, n_boot)
    caches = load_caches(out_root, prof)
    rowsets_path = out_root / "out" / "rowsets" / f"arm{prof.arm}.json"
    assert rowsets_path.is_file(), f"rowsets missing at {rowsets_path} — parent must run first"
    rowsets = build_rowsets(out_root, prof, caches, smoke)  # deterministic re-derive
    ans_info = AnswerInfo(out_root, prof, smoke, bool(args.prefill_fallback))
    t0 = time.time()
    n_total = len(units)
    for k, unit in enumerate(units):
        dest = unit_out_path(out_root, unit)
        fp = _fingerprint(unit, params)
        if dest.is_file():
            prior = json.loads(dest.read_text())
            if prior.get("fingerprint") == fp:
                print(f"[p5] unit {k + 1}/{n_total} {unit.unit_id} resume-skip", flush=True)
                continue
        tu = time.time()
        if unit.kind == "sweep":
            payload = run_sweep_unit(
                unit,
                prof,
                caches,
                rowsets,
                ans_info,
                out_root,
                smoke=smoke,
                null_draws=null_draws,
                n_boot=n_boot,
            )
        elif unit.kind == "traj":
            payload = run_traj_unit(
                unit,
                prof,
                caches,
                rowsets,
                ans_info,
                out_root,
                smoke=smoke,
                null_draws=null_draws,
                n_boot=n_boot,
            )
        elif unit.kind == "ladder":
            payload = run_ladder_unit(
                unit, prof, caches, rowsets, out_root, smoke=smoke, n_boot=n_boot
            )
        elif unit.kind == "operator":
            payload = run_operator_unit(unit, prof, caches, rowsets, smoke)
        elif unit.kind == "ood":
            payload = run_ood_unit(unit, prof, caches, rowsets, smoke)
        elif unit.kind == "reliability":
            payload = run_reliability_unit(unit, prof, caches, rowsets)
        else:  # pragma: no cover - registry is closed
            raise ValueError(unit.kind)
        payload["fingerprint"] = fp
        payload["fit_params"] = params
        payload["repro"] = _repro("p5_fits")
        _atomic_json(dest, payload)
        if payload.get("status") == "dropped_below_floor":
            print(
                f"[p5] unit {k + 1}/{n_total} {unit.unit_id} DROPPED below floor "
                f"(n={payload.get('n_rows', payload.get('n_fit'))}) — reported, never silent",
                flush=True,
            )
        print(
            f"[p5] unit {k + 1}/{n_total} {unit.unit_id} elapsed={time.time() - tu:.0f}s "
            f"total={time.time() - t0:.0f}s",
            flush=True,
        )
        _pygc.collect()
    return 0


def _visible_gpu_count() -> int:
    """GPU count via an nvidia-smi subprocess (never torch — CVD clobber family)."""
    try:
        p = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        n = len([ln for ln in p.stdout.split("\n") if ln.strip()]) if p.returncode == 0 else 0
    except OSError:
        n = 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        n = min(n, len([c for c in cvd.split(",") if c.strip() != ""])) if n else 0
    return n


def run_parent(args, prof: LayerProfile) -> int:
    out_root = Path(args.out_root)
    smoke = bool(args.smoke)
    print(f"[p5] parent arm={prof.arm} smoke={smoke} out_root={out_root}", flush=True)
    # Fan-out shared inputs pre-staged ONCE in the parent (#1315).
    build_fitcache(out_root, prof, smoke)
    caches = load_caches(out_root, prof)
    build_rowsets(out_root, prof, caches, smoke)
    units = build_registry(prof)
    assert registry_stat_totals() == 222, registry_stat_totals()  # plan §9
    del caches
    _pygc.collect()
    ngpu = _visible_gpu_count()
    n_workers = max(1, min(int(args.num_workers), ngpu if ngpu > 0 else 1))
    print(f"[p5] fan-out: {len(units)} registry jobs across {n_workers} worker(s)", flush=True)
    work_dir = out_root / "work" / f"fits_a{prof.arm}"
    work_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for slot in range(n_workers):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--arm",
            str(prof.arm),
            "--out-root",
            str(out_root),
            "--shard",
            f"{slot}/{n_workers}",
        ]
        if smoke:
            cmd.append("--smoke")
        if args.prefill_fallback:
            cmd.append("--prefill-fallback")
        if args.null_draws is not None:
            cmd += ["--null-draws", str(args.null_draws)]
        if args.n_boot is not None:
            cmd += ["--n-boot", str(args.n_boot)]
        env = {**os.environ}  # explicit env passthrough (subprocess-env contract)
        if ngpu > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)  # launcher-env CVD pin (#543/#545)
        log = (work_dir / f"slot{slot}.log").open("w")
        procs.append(
            (slot, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        )
    rc = 0
    for slot, proc, log in procs:
        r = proc.wait()
        log.close()
        print(f"[p5] worker slot{slot} exited rc={r}", flush=True)
        if r != 0:
            rc = r
    if rc != 0:
        print(f"[p5] FATAL: worker failure rc={rc} (logs under {work_dir})", flush=True)
        return rc
    # Completeness: every registry job produced its artifact (or a reported drop).
    statuses: dict[str, str] = {}
    for unit in units:
        dest = unit_out_path(out_root, unit)
        assert dest.is_file(), f"unit artifact missing after workers: {dest}"
        statuses[unit.unit_id] = json.loads(dest.read_text()).get("status", "unknown")
    dropped = sorted(u for u, s in statuses.items() if s != "ok")
    _atomic_json(
        out_root / "out" / "reports" / f"p5_fits_a{prof.arm}.json",
        {
            "arm": prof.arm,
            "n_registry_jobs": len(units),
            "n_statistical_units_cross_arm": 222,
            "statuses": statuses,
            "dropped_or_degraded": dropped,
            "smoke": smoke,
            "repro": _repro("p5_fits"),
        },
    )
    print(f"[p5] complete: {len(units)} jobs, {len(dropped)} non-ok (reported)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Selftest: synthetic-store CPU smoke (loader, set-check, one full-sweep unit,
# one trajectory unit, content-identity hit rule, per-corpus band assert)
# ---------------------------------------------------------------------------


def _selftest_write_stem(
    root: Path,
    arm: int,
    side: str,
    corpus: str,
    rows: list[dict],
    prof: LayerProfile,
    post_like: bool,
) -> None:
    stem_dir = root / "store" / f"arm{arm}" / f"{side}__{corpus}"
    stem_dir.mkdir(parents=True, exist_ok=True)
    B = len(rows)
    kinds = list(g25.KINDS_POST if post_like else g25.KINDS_SHORT)
    rng = np.random.default_rng(zlib.crc32(f"{side}:{corpus}".encode()))
    base = rng.standard_normal((B, 1, prof.n_layers, prof.hidden)).astype(np.float32)
    full = np.repeat(base, len(kinds), axis=1) + 0.05 * rng.standard_normal(
        (B, len(kinds), prof.n_layers, prof.hidden)
    ).astype(np.float32)
    shard = {
        "task": TASK,
        "arm": arm,
        "side": side,
        "corpus": corpus,
        "model": "selftest",
        "revision": "selftest",
        "kinds_full": kinds,
        "kinds_t": list(g25.KINDS_T) if post_like else [],
        "layers_all": list(range(prof.n_layers)),
        "frozen_layers": list(prof.frozen),
        "hidden": prof.hidden,
        "full": torch.as_tensor(full).to(torch.bfloat16),
        "tk": (
            torch.as_tensor(
                np.repeat(base[:, :, list(prof.frozen), :], 9, axis=1)
                + 0.05 * rng.standard_normal((B, 9, len(prof.frozen), prof.hidden))
            ).to(torch.bfloat16)
            if post_like
            else None
        ),
        "row_ids": [r["row_id"] for r in rows],
        "meta": [r["meta"] for r in rows],
        "repro": {"phase": "selftest"},
    }
    torch.save(shard, stem_dir / "slot0.shard000.pt")


def _selftest_write_rel_stem(
    root: Path, arm: int, side: str, corpus: str, base_ids: list[str], prof: LayerProfile
) -> None:
    """Frozen-layer per-draw reliability shard (the P4b producer's exact schema)."""
    stem_dir = root / "store" / f"arm{arm}" / f"rel_{side}__{corpus}"
    stem_dir.mkdir(parents=True, exist_ok=True)
    kinds = list(g25.REL_KINDS_POST)
    n_f = len(prof.frozen)
    rng = np.random.default_rng(zlib.crc32(f"rel:{side}:{corpus}".encode()))
    row_ids: list[str] = []
    metas: list[dict] = []
    fulls: list[np.ndarray] = []
    for rid in base_ids:
        base_vec = rng.standard_normal((len(kinds), n_f, prof.hidden)).astype(np.float32)
        for draw in range(4):
            fulls.append(base_vec + 0.01 * rng.standard_normal(base_vec.shape).astype(np.float32))
            row_ids.append(f"{rid}#d{draw}")
            metas.append({"base_row_id": rid, "draw": draw, "corpus": corpus, "side": side})
    shard = {
        "task": TASK,
        "arm": arm,
        "side": side,
        "corpus": corpus,
        "model": "selftest",
        "revision": "selftest",
        "kinds_full": kinds,
        "kinds_t": [],
        "layers_all": list(prof.frozen),
        "frozen_layers": list(prof.frozen),
        "hidden": prof.hidden,
        "full": torch.as_tensor(np.stack(fulls)).to(torch.bfloat16),
        "tk": None,
        "row_ids": row_ids,
        "meta": metas,
        "repro": {"phase": "selftest"},
    }
    torch.save(shard, stem_dir / "slot0.shard000.pt")


def run_selftest() -> int:
    import tempfile

    prof = LayerProfile(
        arm=1,
        n_layers=3,
        hidden=32,
        frozen=(0, 1, 2),
        headline=1,
        post_side="post",
        short_side="pre",
        has_pre_model=True,
    )
    with tempfile.TemporaryDirectory(prefix="i2546-fits-selftest-") as td:
        root = Path(td)
        n = 64
        corpora = {"gsm8k_train": [], "mmlu": []}
        for corpus in corpora:
            for i in range(n):
                meta = {
                    "corpus": corpus,
                    "side": "post",
                    "k": (5 if i % 2 == 0 else 1) if corpus == "gsm8k_train" else None,
                    "level": None,
                    "short_think": i % 7 == 0,
                }
                corpora[corpus].append({"row_id": f"{corpus}-{i:03d}", "meta": meta})
        for corpus, rows in corpora.items():
            _selftest_write_stem(root, 1, "post", corpus, rows, prof, post_like=True)
            _selftest_write_stem(root, 1, "pre", corpus, rows, prof, post_like=False)
        # Synthetic rollouts through the REAL parse path: post side well-formed
        # under arm-1's "emergent" mode (one think block), pre side plain "off".
        for side, stage in (("post", "post_greedy_a1"), ("pre", "pre_greedy_a1")):
            for corpus, rows in corpora.items():
                rdir = root / "rollouts" / stage
                rdir.mkdir(parents=True, exist_ok=True)
                recs = []
                for i, r in enumerate(rows):
                    if corpus == "gsm8k_train":
                        ans = f"The answer is \\boxed{{{i % 5}}}."
                    else:
                        ans = f"The correct option is {'ABCD'[i % 4]}."
                    text = (
                        f"<think>step {i} of the argument</think>\n{ans}" if side == "post" else ans
                    )
                    recs.append({"row_id": r["row_id"], "text": text, "finish_reason": "stop"})
                with (rdir / f"{corpus}.jsonl").open("w") as fh:
                    for rec in recs:
                        fh.write(json.dumps(rec) + "\n")
        build_fitcache(root, prof, smoke=True, offline=True)
        caches = load_caches(root, prof)
        rowsets = build_rowsets(root, prof, caches, smoke=True)
        assert rowsets["strata"]["does"] and rowsets["strata"]["doesnt"]
        # Content-identity hit rule: identical class -> hit even on a different row.
        ans = {"a": "\\boxed{7}", "b": "\\boxed{7}", "c": "\\boxed{9}"}
        pred = np.asarray([[0.0, 1.0], [0.9, 0.1], [10.0, 10.0]])
        true = np.asarray([[0.9, 0.1], [0.0, 1.0], [10.0, 10.0]])
        res = content_retrieval(
            pred,
            true,
            ["a", "b", "c"],
            ["math"] * 3,
            {k: v for k, v in ans.items()},
            metric="euclidean",
            n_boot=20,
            seed=0,
        )
        # rows a/b retrieve each other's rows but share the boxed-7 class -> hits.
        assert res["corpus_pool"]["acc_at_1"] == 1.0, res["corpus_pool"]

        class _Args:
            out_root = str(root)
            smoke = True
            null_draws = 2
            n_boot = 20
            prefill_fallback = False

        args = _Args()
        fc.N_INNER_LAMBDA_FOLDS = INNER_LAMBDA_FOLDS
        # No monkeypatching: the selftest profile mirrors arm-1 side names/stages,
        # so AnswerInfo resolves the REAL g25.ARMS[1] side specs and the rollouts
        # above are parsed by the real emergent/off parse path.
        units = build_registry(prof)
        picks = [
            next(u for u in units if u.unit_id == "p7_A__does__a1"),
            next(u for u in units if u.kind == "traj"),
            next(u for u in units if u.kind == "ladder" and "gsm8k_train" in u.unit_id),
            next(u for u in units if u.kind == "reliability"),
            # every registry KIND exercised at smoke n (smoke-architecture per-arm rows):
            next(u for u in units if u.kind == "operator"),
            next(u for u in units if u.unit_id == "ood_does2doesnt__a1"),
        ]
        rc = run_units(args, prof, picks)
        assert rc == 0
        op = json.loads((root / "out" / "ladder" / "operator_comparison__a1.json").read_text())
        assert op["status"] == "ok", op
        assert -1.0 <= op["direction_aware"]["raw_cosine_with_rotation_null"]["raw_cosine"] <= 1.0
        assert 0.0 <= op["rotation_invariant_only"]["spectrum_cosine"] <= 1.0 + 1e-9
        ood = json.loads((root / "out" / "cells" / "ood_does2doesnt__a1.json").read_text())
        assert ood["status"] == "ok", ood
        assert np.isfinite(ood["transfer_r2"]) and ood["knn_identity"]["euclidean"]["acc_at_k"]
        a_json = json.loads((root / "out" / "cells" / "p7_A__does__a1.json").read_text())
        assert a_json["status"] == "ok" and a_json["floor_check"] == "smoke-demoted"
        assert a_json["knn_content"] is not None
        traj = json.loads((root / "out" / "cells" / "p7_traj__a1.json").read_text())
        assert traj["strata"]["does"]["status"] == "ok"
        assert len(traj["strata"]["does"]["per_t"]) == 9
        lad = json.loads((root / "out" / "ladder" / "ladder__gsm8k_train__a1.json").read_text())
        assert lad["status"] == "ok"
        assert lad["band_source_corpus"] == lad["ladder_corpus"] == "gsm8k_train"  # MF-5
        assert set(lad["tiers_r2"]) == set(LADDER_TIER_NAMES)
        rel = json.loads((root / "out" / "reliability" / "reliability__a1.json").read_text())
        assert rel["status"] == "missing_reliability_capture"
        # P4b rel stems land -> the SAME unit computes real per-stratum ceilings
        # (frozen-layer shards through the REAL build_fitcache rel_ branch).
        rel_bases: dict[str, list[str]] = {}
        for stratum in STRATA:
            for rid in rowsets["strata"][stratum][:5]:
                rel_bases.setdefault(rid.rsplit("-", 1)[0], []).append(rid)
        for corpus, bases in sorted(rel_bases.items()):
            _selftest_write_rel_stem(root, 1, "post", corpus, bases, prof)
        build_fitcache(root, prof, smoke=True, offline=True)  # builds only the new rel stems
        caches2 = load_caches(root, prof)
        rel_unit = next(u for u in units if u.kind == "reliability")
        rel2 = run_reliability_unit(rel_unit, prof, caches2, rowsets)
        assert rel2["status"] == "ok", rel2
        for stratum in STRATA:
            st = rel2["per_stratum"][stratum]
            assert st["status"] == "ok", (stratum, st)
            assert st["split_half_r2"] > 0.8, (stratum, st)
            assert 0.8 < st["ceiling_spearman_brown"] <= 1.0 + 1e-9, (stratum, st)
        # Frozen-SUBSET index mapping (fails pre-fix): headline layer 1 lives at
        # INDEX 0 of a [1, 2] layers list; slot 1 (absolute-index read) holds
        # fresh per-draw garbage, so absolute indexing reads r ~ 0.
        sub_dir = root / "fitcache_sub" / "rel_post__gsm8k_train"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_bases = [r for r in rowsets["strata"]["does"] if r.startswith("gsm8k_train")][:5]
        assert len(sub_bases) >= 3, sub_bases
        rng_sub = np.random.default_rng(7)
        sub_ids: list[str] = []
        sub_meta: list[dict] = []
        sub_rows: list[np.ndarray] = []
        for rid in sub_bases:
            v = rng_sub.standard_normal(prof.hidden).astype(np.float32)
            for draw in range(4):
                sig = v + 0.01 * rng_sub.standard_normal(prof.hidden).astype(np.float32)
                junk = rng_sub.standard_normal(prof.hidden).astype(np.float32)
                sub_rows.append(np.stack([sig, junk]))  # L axis = layers [1, 2]
                sub_ids.append(f"{rid}#d{draw}")
                sub_meta.append({"base_row_id": rid, "draw": draw})
        torch.save(torch.as_tensor(np.stack(sub_rows)), sub_dir / "ans_mean.pt")
        (sub_dir / "rows.json").write_text(
            json.dumps(
                {
                    "stem": "rel_post__gsm8k_train",
                    "row_ids": sub_ids,
                    "meta": sub_meta,
                    "kinds_full": ["ans_mean"],
                    "has_tk": False,
                    "layers_all": [1, 2],
                    "n_layers": prof.n_layers,
                    "hidden": prof.hidden,
                }
            )
        )
        rel3 = run_reliability_unit(
            rel_unit, prof, {"rel_post": {"gsm8k_train": StemCache(sub_dir)}}, rowsets
        )
        st3 = rel3["per_stratum"]["does"]
        assert st3["status"] == "ok" and st3["split_half_r2"] > 0.8, st3
        print(
            "[selftest] PASS: loader, set-check, sweep unit, traj unit, content hits, "
            "band, reliability ceiling (frozen-subset index mapping)"
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="#2546 P5 fit driver (plan v4 §4.2 P5)")
    ap.add_argument("--arm", type=int, choices=(1, 2, 3), default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--g0", action="store_true", help="G-E fit-core reuse gate (delegates #1336)")
    ap.add_argument("--g0-probe-only", action="store_true")
    ap.add_argument("--g0-local-dir", type=Path, default=None)
    ap.add_argument("--shard", default=None, help="worker mode: i/N over the unit registry")
    ap.add_argument(
        "--num-workers", type=int, default=int(os.environ.get("EPM_I2546_FIT_WORKERS", "4"))
    )
    ap.add_argument("--null-draws", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--prefill-fallback", action="store_true", help="match P4's parse-mode rung")
    ap.add_argument("--selftest", action="store_true", help="synthetic-store CPU smoke")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.selftest:
        return run_selftest()
    # RECORDED module-global patch (plan §4.2 P5 / §10): inner-CV λ selection at
    # 2 inner folds; main's default is 4 and heldout_r2_sweep has no kwarg.
    print(
        f"[p5] RECORDED module-global patch: fc.N_INNER_LAMBDA_FOLDS "
        f"{fc.N_INNER_LAMBDA_FOLDS} -> {INNER_LAMBDA_FOLDS} (plan v4 §4.2 P5; #1336 parity)",
        flush=True,
    )
    fc.N_INNER_LAMBDA_FOLDS = INNER_LAMBDA_FOLDS
    assert args.out_root is not None, "--out-root is required"
    if args.g0 or args.g0_probe_only:
        # G-E gate: the #1336 --g0 mode reused verbatim (refit the pinned #825
        # Qwen S1 cell @ deb7a452; PASS iff L19 R2 within +/-0.01 of 0.6731).
        out_root = Path(args.out_root)
        ns = SimpleNamespace(
            g0_probe_only=bool(args.g0_probe_only),
            g0_local_dir=args.g0_local_dir,
            g0_dl_dir=out_root / "g0_dl",
            out_dir=out_root / "out",
        )
        return int(f36.run_g0(ns))
    assert args.arm is not None, "--arm is required"
    prof = profile_for_arm(args.arm)
    if args.shard:
        i_s, n_s = args.shard.split("/")
        i, nsh = int(i_s), int(n_s)
        units = build_registry(prof)[i::nsh]
        print(f"[p5] worker shard {i}/{nsh}: {len(units)} registry jobs", flush=True)
        return run_units(args, prof, units)
    return run_parent(args, prof)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
