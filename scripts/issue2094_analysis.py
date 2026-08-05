"""Issue #2094 — VM-side P7 analysis driver (unit E; plan §4.4/§6/§7/§10).

Subcommand-structured via ``--phase``:

- ``stage``        stage banked maps (#779/#1738/#1776) at a PINNED revision +
                   this run's own grid/anchor/vc_bank stores from HF into
                   ``--in-root`` (scoped per-file ``stage_hub_file`` /
                   ``stage_hub_prefix`` — never an unscoped enumeration), and
                   re-run ``scripts/verify_reused_artifact_keys.py`` per staged
                   map file (PASS lines into the staging manifest).
- ``parity``       build ``eval_results/issue_2094/map_parity.json`` BEFORE any
                   transport number (the #1768 duty): input-convention +
                   pooling-parity rows + the runtime orientation bind
                   (``fmetrics.bind_map_orientation``; probe-residual preferred,
                   plan §12 held-out-reproduction-style fallback).
- ``ftables``      ``f_metrics/{f_cells,null_cells,anchors}.jsonl`` — F_act
                   (disjoint halves; PRIMARY read layer 26, deep-steer cells
                   read at 27 + marked; full 28-layer profiles exploratory),
                   F_beh over unit D's judge scores (coherent-only per the >60
                   gate), traversal companion, cap-hit NEXT TO incoherence,
                   Type-B donor annotation; cell-coverage SET-CHECK (plan §7,
                   distinct rc on mismatch).
- ``transport``    transport cosines at banked-map cells only (ce L14/L19 + the
                   DECLARED L26 transport-only extension; pe L14/L19/L26), with
                   donor-cell cosines as the in-design control. REFUSES to run
                   without ``map_parity.json``.
- ``linearity``    the L fit (ridge in the per-TRAIN-FOLD top-128-PC subspace of
                   the Δ bank; GroupKFold by pair + the stricter held-out
                   context-family fold; full-space n<d fit as a LABELED
                   regularization-limit read only), identity+learned-bias
                   baseline + kNN retrieval, homogeneity reads, and the
                   direction-aware L vs M vs J comparison + the 2×2 table.
                   Pilot-gated at entry (1 fold timed through this entrypoint).
- ``bootstrap``    pair-clustered bootstrap (B=10,000) as BATCHED index-GEMMs
                   (the ``analysis/null_battery.py`` pattern — no per-draw
                   Python loop). Pilot-gated at entry.
- ``fragility``    excess-incoherence heatmap data (steered minus anchor
                   baseline rate, donor-null side by side) + cap-hit companion.
- ``select-stage2``best (layer, slot) cell per setting × level under the BODY
                   restriction (ce ∈ {14,19}, pe ∈ {14,19,26}; ≤6 cells) →
                   ``best_cells.json``, uploaded FAIL-LOUD to
                   ``issue2094_singlepos/stage2_spec/``.

VM launch convention (shared-VM thread caps — EVERY VM launch of this driver):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_analysis.py --phase <phase>

Outputs land under ``eval_results/issue_2094/`` (git-issue-branch dest, plan
§6.5); staging lands under ``data/issue_2094/hf_dl/`` (re-downloadable cache;
the driver preamble ``df``-checks its filesystem per plan §9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_run as R  # noqa: E402

logger = logging.getLogger("issue2094_analysis")

REPO_ROOT = _SCRIPTS_DIR.parent

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2094_singlepos"

N_LAYERS = 28
HIDDEN = 3584

# Judge conventions (mirrors scripts/issue2094_judge.py — kept as literals so
# this VM driver does not import the judge module's heavy API deps).
COHERENCE_RUBRIC_ID = "coherence"
COHERENCE_THRESHOLD = 60.0

# F_act read-layer rule (plan §4.4): PRIMARY read layer 26 (deepest banked-map
# layer, downstream of every mid-stack edit); cells whose steer touches layer
# >= 26 read at 27 and are MARKED.
PRIMARY_READ_LAYER = 26
DEEP_STEER_READ_LAYER = 27

# Banked-map cells (plan §4.3/§4.4): context-end maps at L14/L19 per the BODY
# enumeration + the DECLARED plan-level L26 transport-only extension; prefix-end
# maps at L14/L19/L26. The stage-2 layer RESTRICTION stays at the body's sets.
TRANSPORT_LAYERS = {"ce": (14, 19, 26), "pe": (14, 19, 26)}
STAGE2_LAYER_RESTRICTION = {"ce": (14, 19), "pe": (14, 19, 26)}
STAGE2_MAX_CELLS = 6

# Linearity fit constants (plan §4.4 / §10).
PC_DIM = 128
N_PAIR_FOLDS = 10
FOLD_SEED = 20942  # plan §10: PC-basis/fold seed
BOOTSTRAP_SEED = 20941  # plan §10: bootstrap seed
BOOTSTRAP_B = 10_000
FIT_DOSES = ("a0.5", "a1", "a2", "a4")  # additive doses only (alpha*Delta is the fit input)
FIT_SLOTS = ("ce", "pe")
FIT_LAYERS = (14, 19, 26)
# Output-space variants per (slot, layer): "same_tail" = x@L -> shift of the
# tail-inclusive answer vector at L (the banked-map convention, space-matched
# with M); "j19_span" = x@L -> shift of the span-mean answer state at layer 19
# (space-matched with the #1776 J_last: L14 last-token -> L19 answer-mean).
FIT_VARIANTS = ("same_tail", "j19_span")
ROTATION_NULL_DRAWS = 200

# Pilot gate (plan §9 P7 row: fits pilot-gated at entry; fence >= 2x pilot wall).
PLANNED_P7_WALL_H = 1.5
PILOT_REFUSAL_MULT = 3.0
PILOT_FENCE_MULT = 2.0

RC_OK = 0
RC_PILOT_GATE = 22
RC_COVERAGE_GATE = 23
RC_PARITY_GATE = 24

# Banked-map spec (plan §4.4/§10). `keys` feed verify_reused_artifact_keys.py.
BANKED_MAPS: tuple[dict, ...] = (
    *(
        {
            "map_id": f"m779_ce_L{layer}",
            "arm": "ce",
            "layer": layer,
            "repo_path": f"issue779_monitoring/n1m_readout/weights/L{layer}/ridge.pt",
            "keys": "kind,xmu,xsd,ymu,W,layer",
            "input_convention": "cx_last (context-end last token, generation prompt "
            "included — matches the injected ce slot, plan §4.2)",
        }
        for layer in (14, 19, 26)
    ),
    *(
        {
            "map_id": f"m1738_pe_L{layer}",
            "arm": "pe",
            "layer": layer,
            "repo_path": f"issue1738_multiturn/analysis_tensors/weights/L{layer}/prefix_ridge.pt",
            "keys": "kind,xmu,xsd,ymu,W,layer,arm",
            "input_convention": "px_last (prefix-end last token — matches the injected "
            "pe slot, plan §4.2)",
        }
        for layer in (14, 19, 26)
    ),
)
JACOBIAN_SPEC = {
    "map_id": "j1776_L14_to_L19",
    "repo_path": "issue1776_jacobian/analysis_tensors/jac_full/J_last.pt",
    "convention": "L14 last-token -> L19 answer-mean (plan §12 assumption 8); "
    "used ONLY in the direction-aware comparison (space-matched L14->L19 variant)",
}


# ── config ─────────────────────────────────────────────────────────────


@dataclass
class AnalysisConfig:
    in_root: Path
    out_root: Path
    judge_root: Path
    hf_revision: str | None
    smoke: bool = False
    coverage: str = "full"  # "full" | "staged"
    force: bool = False
    planned_wall_h: float = PLANNED_P7_WALL_H
    projected_stage_gb: float = 20.0
    skip_disk_check: bool = False
    no_upload: bool = False
    profiles: bool = True
    bootstrap_b: int = BOOTSTRAP_B

    @property
    def mirror(self) -> Path:
        return self.in_root / HF_PREFIX

    @property
    def maps_dir(self) -> Path:
        return self.in_root / "banked_maps"

    @property
    def rollouts_dir(self) -> Path:
        return self.mirror / "raw_completions" / "grid"

    @property
    def anchors_text(self) -> Path:
        return self.mirror / "raw_completions" / "anchors" / "anchors.jsonl"

    @property
    def anchors_pt(self) -> Path:
        return self.mirror / "analysis_tensors" / "anchors" / "va_anchors.pt"

    @property
    def va_dir(self) -> Path:
        return self.mirror / "analysis_tensors" / "va_store"

    @property
    def vc_bank_pt(self) -> Path:
        return self.mirror / "analysis_tensors" / "vc_bank" / "vc_bank.pt"

    @property
    def scores_dir(self) -> Path:
        return self.judge_root / "scores"

    @property
    def fmetrics_dir(self) -> Path:
        return self.out_root / "f_metrics"

    @property
    def parts_dir(self) -> Path:
        return self.fmetrics_dir / "parts"

    @property
    def gates_dir(self) -> Path:
        return self.out_root / "gates"

    @property
    def staging_manifest(self) -> Path:
        return self.out_root / "staging_manifest.json"

    @property
    def map_parity_json(self) -> Path:
        return self.out_root / "map_parity.json"

    @property
    def stage2_additivity_dir(self) -> Path:
        return self.mirror / "analysis_tensors" / "stage2_additivity"


# ── small io helpers ───────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, path)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Text-mode line iteration (never ``splitlines()`` — U+2028 shred, #950)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _repro() -> dict:
    return {
        "git_commit": R._git_sha(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _jsonable(x):
    if isinstance(x, float | int | str | bool) or x is None:
        return x
    if isinstance(x, np.floating | np.integer):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    return str(x)


def _nan_to_none(v: float) -> float | None:
    f = float(v)
    return None if math.isnan(f) else f


# ── P7 entry: staging (phase stage) ────────────────────────────────────


def _disk_preamble(cfg: AnalysisConfig) -> None:
    """Plan §9 mount-binding preamble: name the filesystem + assert headroom."""
    cfg.in_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cfg.in_root)
    free_gb = usage.free / 1e9
    df = subprocess.run(["df", "-P", str(cfg.in_root)], capture_output=True, text=True, check=False)
    logger.info("[stage] df -P %s:\n%s", cfg.in_root, df.stdout.strip())
    need_gb = 1.5 * cfg.projected_stage_gb
    if cfg.skip_disk_check:
        logger.warning("[stage] disk check SKIPPED (--skip-disk-check); free=%.1f GB", free_gb)
        return
    assert free_gb >= need_gb, (
        f"staging headroom: {free_gb:.1f} GB free on {cfg.in_root}'s filesystem < "
        f"1.5 x projected {cfg.projected_stage_gb:.1f} GB — free space or pass "
        f"--projected-stage-gb/--skip-disk-check deliberately"
    )
    logger.info("[stage] headroom OK: %.1f GB free >= %.1f GB needed", free_gb, need_gb)


def _pinned_revision(cfg: AnalysisConfig) -> str:
    """Resolve (or reuse) the pinned data-repo revision for every staged read."""
    if cfg.hf_revision:
        return cfg.hf_revision
    if cfg.staging_manifest.exists():
        pinned = json.loads(cfg.staging_manifest.read_text()).get("revision")
        if pinned:
            logger.info("[stage] reusing pinned revision %s", pinned)
            return pinned
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: HfApi(token=os.environ.get("HF_TOKEN")).repo_info(
            HF_DATA_REPO, repo_type="dataset"
        ),
        what="repo_info(data repo)",
    )
    sha = str(info.sha)
    logger.info("[stage] pinned data-repo revision %s", sha)
    return sha


def _verify_map_keys(path: Path, keys: str) -> str:
    """Re-run the plan-§10 realized-keys verifier on a STAGED map file (PASS line)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_reused_artifact_keys.py"),
            "--artifact",
            str(path),
            "--keys",
            keys,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    out = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, f"verify_reused_artifact_keys FAILED for {path}:\n{out}"
    pass_line = next((ln for ln in out.split("\n") if "PASS" in ln), out)
    logger.info("[stage] %s: %s", path.name, pass_line)
    return pass_line


def phase_stage(cfg: AnalysisConfig) -> int:
    """Stage banked maps + this run's own stores from HF at ONE pinned revision."""
    logger.info("[phase=stage]")
    _disk_preamble(cfg)
    from explore_persona_space.orchestrate import hub

    revision = _pinned_revision(cfg)
    manifest: dict = {"revision": revision, "maps": [], "prefixes": {}, "repro": _repro()}

    for spec in BANKED_MAPS:
        target = cfg.maps_dir / spec["repo_path"]
        hub.stage_hub_file(
            HF_DATA_REPO, spec["repo_path"], target, repo_type="dataset", revision=revision
        )
        pass_line = _verify_map_keys(target, spec["keys"])
        manifest["maps"].append(
            {
                "map_id": spec["map_id"],
                "repo_path": spec["repo_path"],
                "local_path": str(target),
                "bytes": target.stat().st_size,
                "keys_declared": spec["keys"],
                "verify_pass_line": pass_line,
            }
        )
    # #1776 Jacobian: NOT a ridge bundle — record its realized keys/shape instead.
    j_target = cfg.maps_dir / JACOBIAN_SPEC["repo_path"]
    hub.stage_hub_file(
        HF_DATA_REPO, JACOBIAN_SPEC["repo_path"], j_target, repo_type="dataset", revision=revision
    )
    j_obj = torch.load(j_target, map_location="cpu", mmap=True, weights_only=False)
    j_keys = sorted(j_obj.keys()) if isinstance(j_obj, dict) else ["<raw tensor>"]
    manifest["maps"].append(
        {
            "map_id": JACOBIAN_SPEC["map_id"],
            "repo_path": JACOBIAN_SPEC["repo_path"],
            "local_path": str(j_target),
            "bytes": j_target.stat().st_size,
            "realized_keys": j_keys,
            "convention": JACOBIAN_SPEC["convention"],
        }
    )
    del j_obj

    # This run's own inputs (grid/anchor stores, vc_bank, rollout text, manifests).
    own_prefixes = [
        f"{HF_PREFIX}/analysis_tensors/vc_bank",
        f"{HF_PREFIX}/analysis_tensors/anchors",
        f"{HF_PREFIX}/raw_completions/anchors",
        f"{HF_PREFIX}/raw_completions/grid",
        f"{HF_PREFIX}/analysis_tensors/va_store",
        f"{HF_PREFIX}/analysis_tensors/manifests",
    ]
    for prefix in own_prefixes:
        staged = hub.stage_hub_prefix(HF_DATA_REPO, prefix, cfg.in_root, revision=revision)
        manifest["prefixes"][prefix] = len(staged)
        logger.info("[stage] %s: %d files", prefix, len(staged))

    # Judge outputs are VM-resident by construction (unit D writes
    # eval_results/issue_2094/judge); assert presence rather than staging.
    if not cfg.scores_dir.is_dir():
        logger.warning(
            "[stage] judge scores dir missing at %s — run issue2094_judge.py first "
            "(or restage its work root from %s/raw_completions/judge_raw)",
            cfg.scores_dir,
            HF_PREFIX,
        )
    manifest["judge_scores_present"] = cfg.scores_dir.is_dir()
    _write_json_atomic(cfg.staging_manifest, manifest)
    logger.info("[phase=stage_done] manifest -> %s", cfg.staging_manifest)
    return RC_OK


# ── map loading + parity (phase parity) ────────────────────────────────


def _load_bundle(path: Path) -> dict:
    """Load a sha-pinned SELF-PRODUCED ridge bundle (torch>=2.6: weights_only=False
    is required for the non-tensor metadata these bundles carry)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    FM.validate_map_bundle(bundle)
    return bundle


def _load_jacobian(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        j = obj
    else:
        cand = [k for k in ("J", "J_last", "jac", "W") if k in obj]
        assert cand, f"no Jacobian tensor key in {sorted(obj.keys())}"
        j = obj[cand[0]]
    assert j.dim() == 2, j.shape
    return j.float()


def _load_vc_bank(cfg: AnalysisConfig) -> dict:
    path = cfg.vc_bank_pt
    assert path.exists(), f"{path} missing — run --phase stage first"
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_anchor_va(cfg: AnalysisConfig) -> dict:
    """Per-context anchor answer states: {cid: {"span": (K,L,H), "tail": (K,L,H)}}."""
    obj = torch.load(cfg.anchors_pt, map_location="cpu", weights_only=False)
    index = obj["index"]
    span = obj["va_span"].float()
    tail = obj["va_tail"].float()
    by_ctx: dict[str, dict[str, list]] = {}
    for i, rec in enumerate(index):
        by_ctx.setdefault(rec["context_id"], {"span": [], "tail": [], "draws": []})
        by_ctx[rec["context_id"]]["span"].append(span[i])
        by_ctx[rec["context_id"]]["tail"].append(tail[i])
        by_ctx[rec["context_id"]]["draws"].append(rec["draw"])
    out = {}
    for cid, d in by_ctx.items():
        out[cid] = {
            "span": torch.stack(d["span"]),
            "tail": torch.stack(d["tail"]),
            "draws": list(d["draws"]),
        }
    return out


def _slot_input_vector(bank: dict, cid: str, slot: str, layer: int) -> torch.Tensor:
    """The banked-map INPUT state for one context at (slot, layer): (H,)."""
    rec = bank["per_context"][cid]
    return R._slot_vectors(rec, slot)[-1][layer].float()


def phase_parity(cfg: AnalysisConfig) -> int:
    """map_parity.json BEFORE any transport number (plan §6 pooling row, #1768)."""
    logger.info("[phase=parity]")
    bank = _load_vc_bank(cfg)
    anchor_va = _load_anchor_va(cfg)
    contexts = sorted(bank["per_context"])
    rows = []
    for spec in BANKED_MAPS:
        bundle = _load_bundle(cfg.maps_dir / spec["repo_path"])
        layer = spec["layer"]
        slot = spec["arm"]
        probe_x = torch.stack([_slot_input_vector(bank, c, slot, layer) for c in contexts])
        probe_y = torch.stack([anchor_va[c]["tail"][:, layer].mean(dim=0) for c in contexts])
        try:
            decision = FM.bind_map_orientation(bundle, probe_x, probe_y)
        except ValueError as exc:
            # Plan §12 assumption-6 fallback: output-space SCALE vs ymu residuals.
            logger.warning("[parity] %s probe-residual ambiguous (%s) — scale bind", spec, exc)
            ref = float((probe_y.double() - bundle["ymu"].double()).norm(dim=-1).pow(2).mean())
            decision = FM.bind_map_orientation(bundle, probe_x, reference_scale=math.sqrt(ref))
        pred = FM.apply_ridge_map(bundle, probe_x, orientation=decision.orientation)
        resid = (pred.double() - probe_y.double()).pow(2).sum(dim=-1).mean()
        var = (probe_y.double() - probe_y.double().mean(dim=0)).pow(2).sum(dim=-1).mean()
        rows.append(
            {
                "map_id": spec["map_id"],
                "slot": slot,
                "layer": layer,
                "repo_path": spec["repo_path"],
                "input_convention": spec["input_convention"],
                "map_output_pooling": "v_x (answer span INCLUDING end-of-turn tail — the "
                "capture_answer_vector convention; plan §6 pooling row)",
                "transport_va_variant": "va_tail (tail-inclusive, captured in the same "
                "forward as va_span — unit C captures BOTH)",
                "f_act_va_variant": "va_span (the #1415 capture_vectors convention; "
                "F_act keeps parent parity — deliberate dual pooling, plan §6)",
                "orientation": decision.as_dict(),
                "probe_reproduction": {
                    "kind": "own-bank anchors reproduction (OFF-lineage: probes are this "
                    "run's 15 contexts + anchor answer means, not the map's own lineage "
                    "sample — a reproduction-STYLE sanity, plan §12 'where cheap')",
                    "n_probe": len(contexts),
                    "r2_vs_context_mean": float(1.0 - resid / max(float(var), 1e-30)),
                    "rms_residual": math.sqrt(float(resid)),
                },
                "bundle_meta": {
                    "kind": str(bundle.get("kind")),
                    "layer": int(bundle["layer"]) if "layer" in bundle else None,
                    "arm": str(bundle.get("arm")) if "arm" in bundle else None,
                    "selected_lambda": _jsonable(bundle.get("selected_lambda")),
                },
            }
        )
        assert "layer" not in bundle or int(bundle["layer"]) == layer, (
            f"{spec['map_id']}: bundle layer {bundle['layer']} != expected {layer}"
        )
    j = _load_jacobian(cfg.maps_dir / JACOBIAN_SPEC["repo_path"])
    payload = {
        "maps": rows,
        "jacobian": {
            "map_id": JACOBIAN_SPEC["map_id"],
            "shape": list(j.shape),
            "convention": JACOBIAN_SPEC["convention"],
        },
        "repro": _repro(),
    }
    _write_json_atomic(cfg.map_parity_json, payload)
    logger.info("[phase=parity_done] %d maps -> %s", len(rows), cfg.map_parity_json)
    return RC_OK


# ── judge-score lookups ────────────────────────────────────────────────


@dataclass
class JudgeLookups:
    grid_coh: dict = field(default_factory=dict)  # (block_key, pair_id) -> score|None
    anch_coh: dict = field(default_factory=dict)  # (context_id, draw) -> score|None
    grid_beh: dict = field(default_factory=dict)  # (block_key, pair_id, kind, side) -> score
    anch_beh: dict = field(default_factory=dict)  # (context_id, draw, rubric_id) -> score


def load_judge_lookups(scores_dir: Path) -> JudgeLookups:
    """Route unit D's score rows by rubric + source kind (file names not load-bearing)."""
    lk = JudgeLookups()
    files = sorted(scores_dir.glob("*.scores.jsonl"))
    assert files, f"no *.scores.jsonl under {scores_dir} — run issue2094_judge.py first"
    n = 0
    for f in files:
        for row in _iter_jsonl(f):
            n += 1
            kind = row.get("kind")
            if row["rubric_id"] == COHERENCE_RUBRIC_ID:
                if kind == "grid":
                    lk.grid_coh[(row["block_key"], row["pair_id"])] = row["score"]
                elif kind == "anchor":
                    lk.anch_coh[(row["context_id"], row["draw"])] = row["score"]
                continue
            if kind == "grid":
                key = (row["block_key"], row["pair_id"], row["rubric_kind"], row["side"])
                lk.grid_beh[key] = row["score"]
            elif kind == "anchor":
                lk.anch_beh[(row["context_id"], row["draw"], row["rubric_id"])] = row["score"]
    logger.info(
        "[judge] %d rows from %d files (grid_coh=%d anch_coh=%d grid_beh=%d anch_beh=%d)",
        n,
        len(files),
        len(lk.grid_coh),
        len(lk.anch_coh),
        len(lk.grid_beh),
        len(lk.anch_beh),
    )
    return lk


def _rubric_id_for(pair: BANK.Pair, kind: str, side: str) -> str:
    """Mirror of issue2094_judge.rubric_id_for (kept literal — no judge import)."""
    assert side in ("a", "b"), side
    if kind == "query":
        return f"fq-{pair.query_a if side == 'a' else pair.query_b}"
    assert kind == "prefix", kind
    return f"fp-{pair.prefix_a if side == 'a' else pair.prefix_b}"


def _coherent(score) -> bool:
    return score is not None and float(score) > COHERENCE_THRESHOLD


def anchor_pair_stats(
    pairs: Sequence[BANK.Pair], lk: JudgeLookups, draws_by_ctx: dict[str, list[int]]
) -> dict[tuple[str, str], dict]:
    """Per (pair_id, kind): floor/ceiling Δ means over COHERENT anchor draws.

    Δ_d = (judge_B - judge_A)/100 on draw d; floor draws come from context A,
    ceiling draws from context B. Draws missing either rubric score (rule-9
    drops) or judged incoherent are EXCLUDED and counted.
    """
    out: dict[tuple[str, str], dict] = {}
    for pair in pairs:
        for kind in BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            rid_a = _rubric_id_for(pair, kind, "a")
            rid_b = _rubric_id_for(pair, kind, "b")
            stats = {}
            for role, cid in (("floor", pair.a), ("ceiling", pair.b)):
                deltas, n_incoh, n_missing = [], 0, 0
                for d in draws_by_ctx.get(cid, []):
                    if not _coherent(lk.anch_coh.get((cid, d))):
                        n_incoh += 1
                        continue
                    sa = lk.anch_beh.get((cid, d, rid_a))
                    sb = lk.anch_beh.get((cid, d, rid_b))
                    if sa is None or sb is None:
                        n_missing += 1
                        continue
                    deltas.append((float(sb) - float(sa)) / 100.0)
                stats[role] = {
                    "mean": float(np.mean(deltas)) if deltas else None,
                    "n": len(deltas),
                    "n_incoherent": n_incoh,
                    "n_judge_missing": n_missing,
                }
            row = {
                "pair_id": pair.pair_id,
                "setting": pair.setting,
                "kind": kind,
                "context_a": pair.a,
                "context_b": pair.b,
                "floor": stats["floor"],
                "ceiling": stats["ceiling"],
            }
            fl, ce = stats["floor"]["mean"], stats["ceiling"]["mean"]
            row["separation"] = (ce - fl) if (fl is not None and ce is not None) else None
            out[(pair.pair_id, kind)] = row
    return out


# ── f-table assembly (phase ftables) ───────────────────────────────────


def read_layer_for(steer_layers: Sequence[int], n_layers: int = N_LAYERS) -> tuple[int, bool]:
    """(read_layer, marked): PRIMARY 26; steer touching layer >=26 reads 27 + marked."""
    deep = max(steer_layers) >= PRIMARY_READ_LAYER
    if deep:
        return min(DEEP_STEER_READ_LAYER, n_layers - 1), True
    return min(PRIMARY_READ_LAYER, n_layers - 1), False


def annotate_donor(row: dict) -> dict:
    """Donor stratification fields for a null-arm row (unit-A Type-B caveat).

    The persona<->conv Type-B donor is ANTI-PARALLEL to the recipient (the
    centroid pool has size 1, so the swap returns the REVERSED direction);
    annotate so downstream null reads can stratify.
    """
    if row["arm"] != "null":
        return {"donor_kind": None, "donor_antiparallel": None}
    if row["vec_type"] == "B":
        prefixes = {row["context_a"].split("__")[0], row["context_b"].split("__")[0]}
        return {
            "donor_kind": "typeB-centroid-swap",
            "donor_antiparallel": prefixes == {"persona", "conv"},
        }
    return {"donor_kind": "typeA-derangement", "donor_antiparallel": False}


def _f_act_for_rows(
    va_span: torch.Tensor,
    read_layers: list[int],
    ctx_a: list[str],
    ctx_b: list[str],
    anchor_va: dict,
    profiles: bool,
) -> tuple[FM.FActResult, list[list[float]] | None]:
    """Batched F_act at each row's read layer (+ optional 28-layer profiles)."""
    n = va_span.shape[0]
    vp = torch.stack([va_span[i, read_layers[i]] for i in range(n)])
    floors = torch.stack([anchor_va[ctx_a[i]]["span"][:, read_layers[i]] for i in range(n)])
    ceils = torch.stack([anchor_va[ctx_b[i]]["span"][:, read_layers[i]] for i in range(n)])
    res = FM.f_act(vp, floors, ceils)
    prof = None
    if profiles:
        per_layer = []
        for layer in range(va_span.shape[1]):
            fl = torch.stack([anchor_va[c]["span"][:, layer] for c in ctx_a])
            ce = torch.stack([anchor_va[c]["span"][:, layer] for c in ctx_b])
            per_layer.append(FM.f_act(va_span[:, layer], fl, ce).f_act)
        mat = torch.stack(per_layer, dim=1)  # (n, L)
        prof = [[_nan_to_none(v) for v in row] for row in mat.tolist()]
    return res, prof


def assemble_shard_rows(
    rows: list[dict],
    shard_va: dict,
    lk: JudgeLookups,
    pair_stats: dict[tuple[str, str], dict],
    anchor_va: dict,
    pairs_by_id: dict[str, BANK.Pair],
    profiles: bool = True,
) -> list[dict]:
    """Per-cell F rows for ONE shard (pure — every input in memory)."""
    assert len(rows) == len(shard_va["index"]), (len(rows), len(shard_va["index"]))
    for r, ix in zip(rows, shard_va["index"], strict=True):
        assert r["pair_id"] == ix["pair_id"], (r["pair_id"], ix["pair_id"])
    va_span = shard_va["va_span"].float()
    empty = set(shard_va.get("empty_rows", []))
    read = [read_layer_for(r["layers"])[0] for r in rows]
    marked = [read_layer_for(r["layers"])[1] for r in rows]
    fa, prof = _f_act_for_rows(
        va_span,
        read,
        [r["context_a"] for r in rows],
        [r["context_b"] for r in rows],
        anchor_va,
        profiles,
    )

    # F_beh (batched over row x kind).
    beh_index: list[tuple[int, str]] = []
    dp, df, dc = [], [], []
    beh_missing: dict[tuple[int, str], str] = {}
    for i, r in enumerate(rows):
        pair = pairs_by_id[r["pair_id"]]
        for kind in BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            sa = lk.grid_beh.get((r["block_key"], r["pair_id"], kind, "a"))
            sb = lk.grid_beh.get((r["block_key"], r["pair_id"], kind, "b"))
            st = pair_stats.get((r["pair_id"], kind))
            if sa is None or sb is None:
                beh_missing[(i, kind)] = "judge_dropped"
                continue
            if st is None or st["floor"]["mean"] is None or st["ceiling"]["mean"] is None:
                beh_missing[(i, kind)] = "anchor_missing"
                continue
            beh_index.append((i, kind))
            dp.append((float(sb) - float(sa)) / 100.0)
            df.append(st["floor"]["mean"])
            dc.append(st["ceiling"]["mean"])
    fb = None
    if beh_index:
        fb = FM.f_beh(torch.tensor(dp), torch.tensor(df), torch.tensor(dc))
    beh_by_rowkind: dict[tuple[int, str], dict] = {}
    for j, (i, kind) in enumerate(beh_index):
        beh_by_rowkind[(i, kind)] = {
            "f_beh": _nan_to_none(fb.f_beh[j]),
            "contrast": float(fb.contrast[j]),
            "denominator": float(fb.denominator[j]),
            "degenerate_denominator": bool(fb.degenerate_denominator[j]),
            "negative_denominator": bool(fb.negative_denominator[j]),
            "delta_patched": dp[j],
        }

    out = []
    for i, r in enumerate(rows):
        pair = pairs_by_id[r["pair_id"]]
        coh = lk.grid_coh.get((r["block_key"], r["pair_id"]))
        coherent = _coherent(coh) if coh is not None else None
        excluded = coherent is not True  # 1 greedy draw: incoherent/unjudged => excluded+marked
        kinds = BANK.SETTING_RUBRIC_KINDS[pair.setting]
        beh = {}
        for kind in kinds:
            rec = beh_by_rowkind.get((i, kind))
            if rec is None:
                beh[kind] = {"f_beh": None, "missing": beh_missing.get((i, kind), "unknown")}
            elif excluded:
                beh[kind] = {**rec, "f_beh": None, "excluded_incoherent_raw": rec["f_beh"]}
            else:
                beh[kind] = rec
        empty_row = i in empty
        row = {
            "block_key": r["block_key"],
            "slot": r["slot"],
            "layer_variant": r["layer_variant"],
            "steer_layers": r["layers"],
            "dose": r["dose"],
            "alpha": r.get("alpha"),
            "vec_type": r["vec_type"],
            "arm": r["arm"],
            "pair_id": r["pair_id"],
            "setting": r["setting"],
            "context_a": r["context_a"],
            "context_b": r["context_b"],
            "donor_pair_id": r.get("donor_pair_id"),
            **annotate_donor(r),
            # coherence + cap-hit side by side (plan §4.5).
            "coherence_score": coh,
            "coherent": coherent,
            "excluded_incoherent": excluded,
            "n_coherent": int(coherent is True),
            "n_total": 1,
            "cap_hit": bool(r.get("cap_hit")),
            "empty_completion": empty_row,
            # F_act (all-K anchor draws; disjoint-half floor per fmetrics).
            "read_layer": read[i],
            "read_layer_marked": marked[i],
            "f_act": None if (excluded or empty_row) else _nan_to_none(fa.f_act[i]),
            "f_act_raw": _nan_to_none(fa.f_act[i]) if not empty_row else None,
            "f_act_shared_recordonly": _nan_to_none(fa.f_act_shared[i]) if not empty_row else None,
            "f_act_degenerate": bool(fa.degenerate[i]),
            "s_norm": float(fa.s_norm[i]),
            "t_norm": float(fa.t_norm[i]),
            "traversal_ratio": _nan_to_none(fa.traversal_ratio[i]) if not empty_row else None,
            "f_beh": beh,
            "primary_kind": {"matched_prefix": "query", "matched_query": "prefix"}.get(
                pair.setting
            ),
        }
        if prof is not None:
            row["f_act_profile"] = prof[i]
        out.append(row)
    return out


def coverage_check(produced: set[tuple[str, str]], expected: set[tuple[str, str]]) -> dict:
    """Plan §7 gate: set-equality of produced (block_key, pair_id) vs expected."""
    missing = sorted(expected - produced)
    extra = sorted(produced - expected)
    return {
        "passed": not missing and not extra,
        "n_expected": len(expected),
        "n_produced": len(produced),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
    }


def _ftables_regime(cfg: AnalysisConfig) -> str:
    key = json.dumps(
        {
            "code": "ftables-v1",
            "coherence_threshold": COHERENCE_THRESHOLD,
            "primary_read_layer": PRIMARY_READ_LAYER,
            "profiles": cfg.profiles,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def phase_ftables(cfg: AnalysisConfig) -> int:
    """f_cells.jsonl + null_cells.jsonl + anchors.jsonl + the coverage gate."""
    logger.info("[phase=ftables]")
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    lk = load_judge_lookups(cfg.scores_dir)
    anchor_rows = list(_iter_jsonl(cfg.anchors_text))
    draws_by_ctx: dict[str, list[int]] = {}
    anch_meta: dict[tuple[str, int], dict] = {}
    for r in anchor_rows:
        draws_by_ctx.setdefault(r["context_id"], []).append(r["draw"])
        anch_meta[(r["context_id"], r["draw"])] = r
    anchor_va = _load_anchor_va(cfg)
    pair_stats = anchor_pair_stats(pairs, lk, draws_by_ctx)

    regime = _ftables_regime(cfg)
    parts_manifest = cfg.parts_dir / "parts_manifest.json"
    done_parts: set[str] = set()
    if parts_manifest.exists():
        rec = json.loads(parts_manifest.read_text())
        if rec.get("regime") != regime:
            raise RuntimeError(
                f"ftables parts at {cfg.parts_dir} carry a DIFFERENT regime "
                f"({rec.get('regime')} != {regime}) — quarantine or delete the parts dir"
            )
        done_parts = set(rec.get("done", []))

    shards = sorted(cfg.rollouts_dir.glob("shard_*.jsonl"))
    assert shards, f"no grid shards under {cfg.rollouts_dir} — run --phase stage"
    t_start = time.monotonic()
    for k, shard in enumerate(shards, start=1):
        slug = shard.stem.removeprefix("shard_")
        part = cfg.parts_dir / f"{slug}.jsonl"
        if slug in done_parts and part.exists():
            continue
        rows = list(_iter_jsonl(shard))
        va = torch.load(cfg.va_dir / f"shard_{slug}.pt", map_location="cpu", weights_only=False)
        cell_rows = assemble_shard_rows(
            rows, va, lk, pair_stats, anchor_va, pairs_by_id, profiles=cfg.profiles
        )
        _write_jsonl_atomic(part, cell_rows)
        done_parts.add(slug)
        _write_json_atomic(parts_manifest, {"regime": regime, "done": sorted(done_parts)})
        logger.info(
            "[ftables] shard %d/%d %s rows=%d elapsed=%.1fs",
            k,
            len(shards),
            slug,
            len(cell_rows),
            time.monotonic() - t_start,
        )

    # Concatenate parts -> f_cells (steered) + null_cells (null).
    steered, null = [], []
    produced: set[tuple[str, str]] = set()
    for part in sorted(cfg.parts_dir.glob("*.jsonl")):
        for row in _iter_jsonl(part):
            produced.add((row["block_key"], row["pair_id"]))
            (steered if row["arm"] == "steered" else null).append(row)
    _write_jsonl_atomic(cfg.fmetrics_dir / "f_cells.jsonl", steered)
    _write_jsonl_atomic(cfg.fmetrics_dir / "null_cells.jsonl", null)

    anchors_out = []
    for (pid, kind), st in sorted(pair_stats.items()):
        anchors_out.append(st)
    # per-(context, draw) anchor coherence + cap-hit summary rides along.
    anch_summary = [
        {
            "context_id": cid,
            "draw": d,
            "coherence_score": lk.anch_coh.get((cid, d)),
            "coherent": _coherent(lk.anch_coh.get((cid, d))),
            "cap_hit": bool(anch_meta[(cid, d)].get("cap_hit")),
        }
        for (cid, d) in sorted(anch_meta)
    ]
    _write_jsonl_atomic(cfg.fmetrics_dir / "anchors.jsonl", anchors_out)
    _write_jsonl_atomic(cfg.fmetrics_dir / "anchor_draws.jsonl", anch_summary)

    # Coverage gate (plan §7): produced == realized/registered grid.
    if cfg.coverage == "full":
        expected = set()
        for fam in R.enumerate_block_families(pairs, N_LAYERS):
            for block in fam:
                for pid in block.pair_ids:
                    expected.add((block.key, pid))
    else:  # staged: what the staged shards themselves carry (smoke/partial runs)
        expected = set(produced)
    gate = coverage_check(produced, expected)
    gate["mode"] = cfg.coverage
    gate["repro"] = _repro()
    _write_json_atomic(cfg.gates_dir / "cell_coverage.json", gate)
    logger.info(
        "[ftables] coverage %s: produced=%d expected=%d -> %s",
        cfg.coverage,
        gate["n_produced"],
        gate["n_expected"],
        "PASS" if gate["passed"] else "FAIL",
    )
    logger.info(
        "[phase=ftables_done] steered=%d null=%d anchors=%d",
        len(steered),
        len(null),
        len(anchors_out),
    )
    if not gate["passed"]:
        return RC_COVERAGE_GATE
    return RC_OK


# ── transport (phase transport) ────────────────────────────────────────


def _orientation_for(parity: dict, map_id: str) -> str:
    for row in parity["maps"]:
        if row["map_id"] == map_id:
            return row["orientation"]["orientation"]
    raise KeyError(map_id)


def phase_transport(cfg: AnalysisConfig) -> int:
    """Transport cosines at banked-map cells (steered + donor-null control)."""
    logger.info("[phase=transport]")
    if not cfg.map_parity_json.exists():
        logger.error(
            "[transport] %s missing — run --phase parity FIRST (no transport number "
            "before the parity record; plan §6/#1768)",
            cfg.map_parity_json,
        )
        return RC_PARITY_GATE
    parity = json.loads(cfg.map_parity_json.read_text())
    bank = _load_vc_bank(cfg)
    anchor_va = _load_anchor_va(cfg)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = bank.get("donor_derangement") or BANK.donor_derangement(pairs)
    bundles: dict[tuple[str, int], tuple[dict, str]] = {}
    for spec in BANKED_MAPS:
        b = _load_bundle(cfg.maps_dir / spec["repo_path"])
        bundles[(spec["arm"], spec["layer"])] = (b, _orientation_for(parity, spec["map_id"]))

    eligible = {
        (slot, f"L{layer}"): layer for slot in ("ce", "pe") for layer in TRANSPORT_LAYERS[slot]
    }
    out_rows: list[dict] = []
    shards = sorted(cfg.va_dir.glob("shard_*.pt"))
    n_done = 0
    t0 = time.monotonic()
    for shard in shards:
        slug = shard.stem.removeprefix("shard_")
        jsonl = cfg.rollouts_dir / f"shard_{slug}.jsonl"
        rows = list(_iter_jsonl(jsonl))
        if not rows:
            continue
        head = rows[0]
        key = (head["slot"], head["layer_variant"])
        if key not in eligible:
            continue
        layer = eligible[key]
        bundle, orientation = bundles[(head["slot"], layer)]
        va = torch.load(shard, map_location="cpu", weights_only=False)
        va_tail = va["va_tail"].float()
        for i, r in enumerate(rows):
            pair = pairs_by_id[r["pair_id"]]
            fl = anchor_va[r["context_a"]]["tail"][:, layer]
            realized = va_tail[i, layer] - fl.mean(dim=0)
            fl_h1, fl_h2 = FM.disjoint_half_means(fl)
            # payload reconstruction: EXACTLY the run_block path (reuse, no clone).
            delta, state_b, _m = R._pair_payload(bank, pair, r["slot"], r["vec_type"])
            recipient = delta
            if r["arm"] == "null":
                donor = pairs_by_id[donor_map[r["pair_id"]]]
                recipient, _label = R._donor_payload(
                    bank, pair, donor, r["slot"], r["vec_type"], recipient
                )
            d_l = recipient[-1][layer].float()
            v_s = _slot_input_vector(bank, r["context_a"], r["slot"], layer)
            if r["dose"] == "replace":
                pred = FM.apply_ridge_map(
                    bundle, state_b[-1][layer].float(), orientation=orientation
                ) - FM.apply_ridge_map(bundle, v_s, orientation=orientation)
            else:
                pred = FM.transport_predicted_shift(
                    bundle, v_s, d_l, float(r["alpha"]), orientation=orientation
                )
            out_rows.append(
                {
                    "block_key": r["block_key"],
                    "map_id": f"m779_ce_L{layer}" if r["slot"] == "ce" else f"m1738_pe_L{layer}",
                    "slot": r["slot"],
                    "layer": layer,
                    "dose": r["dose"],
                    "alpha": r.get("alpha"),
                    "vec_type": r["vec_type"],
                    "arm": r["arm"],
                    "pair_id": r["pair_id"],
                    "setting": r["setting"],
                    "orientation": orientation,
                    "cosine_tail": _nan_to_none(FM.safe_cosine(realized, pred)),
                    "cosine_tail_half1": _nan_to_none(
                        FM.safe_cosine(va_tail[i, layer] - fl_h1, pred)
                    ),
                    "cosine_tail_half2": _nan_to_none(
                        FM.safe_cosine(va_tail[i, layer] - fl_h2, pred)
                    ),
                    "realized_norm": float(realized.norm()),
                    "pred_norm": float(pred.norm()),
                }
            )
        n_done += 1
        logger.info(
            "[transport] shard %s (%d/%d eligible done) rows=%d elapsed=%.1fs",
            slug,
            n_done,
            len(shards),
            len(rows),
            time.monotonic() - t0,
        )
    tdir = cfg.out_root / "transport"
    _write_jsonl_atomic(tdir / "transport_cells.jsonl", out_rows)
    summary: dict[str, dict] = {}
    for r in out_rows:
        k = f"{r['map_id']}|{r['dose']}|{r['vec_type']}|{r['arm']}"
        s = summary.setdefault(k, {"n": 0, "sum": 0.0, "n_nan": 0})
        if r["cosine_tail"] is None:
            s["n_nan"] += 1
        else:
            s["n"] += 1
            s["sum"] += r["cosine_tail"]
    _write_json_atomic(
        tdir / "transport_summary.json",
        {
            "cells": {
                k: {"mean_cosine": (s["sum"] / s["n"]) if s["n"] else None, **s}
                for k, s in sorted(summary.items())
            },
            "note": "cosines use the tail-inclusive va_tail pooling (map-lineage parity, "
            "plan §6); donor-null rows are the in-design control",
            "repro": _repro(),
        },
    )
    logger.info("[phase=transport_done] rows=%d", len(out_rows))
    return RC_OK


# ── linearity: the L fit + comparisons (phase linearity) ───────────────


def group_kfold_pairs(pair_ids: Sequence[str], n_folds: int, seed: int) -> list[list[str]]:
    """Deterministic GroupKFold over pair ids (all obs of a pair leave together)."""
    rng = np.random.default_rng(seed)
    ids = list(pair_ids)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    return [list(chunk) for chunk in np.array_split(np.array(perm, dtype=object), n_folds)]


def family_folds(pairs: Sequence[BANK.Pair]) -> list[dict]:
    """Held-out context-family folds: test pairs touching context c; train pairs
    touching NEITHER endpoint of c (plan §4.4 — the generalization-claim read)."""
    contexts = sorted({p.a for p in pairs} | {p.b for p in pairs})
    folds = []
    for c in contexts:
        test = [p.pair_id for p in pairs if p.a == c or p.b == c]
        train = [p.pair_id for p in pairs if p.a != c and p.b != c]
        folds.append({"context": c, "test": test, "train": train})
    return folds


def pc_basis(x_train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean, basis (d, k)) from TRAIN rows only (the fold-leakage fix)."""
    mu = x_train.mean(axis=0)
    xc = x_train - mu
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    k_eff = min(k, vt.shape[0])
    return mu, vt[:k_eff].T.copy()


def ridge_gcv_pc(
    z_train: np.ndarray, y_train: np.ndarray, lams: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Closed-form multi-output ridge in PC space with GCV λ selection.

    n_train > d_eff here (240-fold-test > 128), so GCV selection is
    well-posed (the #1887 ban targets pure-GCV at n<d). Returns
    (W (k,d_out), z_mean, y_mean, selected_lambda).
    """
    n, k = z_train.shape
    assert n > k, f"PC-ridge expects n_train ({n}) > d_eff ({k})"
    zm, ym = z_train.mean(axis=0), y_train.mean(axis=0)
    zc, yc = z_train - zm, y_train - ym
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    uty = u.T @ yc  # (k, d_out)
    if lams is None:
        lams = np.geomspace(1e-6, 1e2, 17) * float((s**2).mean())
    best = None
    for lam in lams:
        shrink = s / (s**2 + lam)
        fitted = u @ ((s * shrink)[:, None] * uty)
        sse = float(((yc - fitted) ** 2).sum())
        df = float((s**2 / (s**2 + lam)).sum())
        gcv = (sse / n) / (1.0 - df / n) ** 2
        if best is None or gcv < best[0]:
            w = vt.T @ (shrink[:, None] * uty)
            best = (gcv, lam, w)
    _gcv, lam, w = best
    return w, zm, ym, float(lam)


def _pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def fit_family_folds(
    x: np.ndarray,
    y: np.ndarray,
    pair_ids: list[str],
    folds: list[list[str]],
    pc_dim: int = PC_DIM,
) -> dict:
    """One fold scheme: per-fold PC basis (TRAIN rows only) + GCV PC-ridge +
    identity+bias baseline; pooled OOF R² + OOF predictions."""
    pid = np.array(pair_ids, dtype=object)
    oof_pred = np.full_like(y, np.nan)
    oof_ident = np.full_like(y, np.nan)
    covered = np.zeros(len(pair_ids), dtype=bool)
    selected_lams, fold_r2 = [], []
    for fold_ids in folds:
        test_mask = np.isin(pid, np.array(list(fold_ids), dtype=object))
        train_mask = ~test_mask
        if isinstance(fold_ids, dict):  # family fold dict
            raise TypeError("pass fold id lists")
        if not test_mask.any() or not train_mask.any():
            continue
        mu, basis = pc_basis(x[train_mask], pc_dim)
        w, zm, ym, lam = ridge_gcv_pc((x[train_mask] - mu) @ basis, y[train_mask])
        z_test = (x[test_mask] - mu) @ basis
        oof_pred[test_mask] = ym + (z_test - zm) @ w
        oof_ident[test_mask] = identity_bias_predict(x[train_mask], y[train_mask], x[test_mask])
        covered |= test_mask
        selected_lams.append(lam)
        fold_r2.append(_pooled_r2(y[test_mask], oof_pred[test_mask]))
    mask = ~np.isnan(oof_pred).any(axis=1)
    return {
        "oof_pred": oof_pred,
        "oof_identity": oof_ident,
        "covered_mask": mask,
        "pooled_r2": _pooled_r2(y[mask], oof_pred[mask]) if mask.any() else None,
        "pooled_r2_identity_bias": _pooled_r2(y[mask], oof_ident[mask]) if mask.any() else None,
        "per_fold_r2": fold_r2,
        "selected_lambdas": selected_lams,
    }


def fit_family_family_folds(
    x: np.ndarray, y: np.ndarray, pair_ids: list[str], pairs_by_id: dict, pc_dim: int = PC_DIM
) -> dict:
    """The stricter context-family fold (test obs can appear in 2 folds; pooled
    over the union WITH duplicates + per-fold R²s, both reported)."""
    pairs = [pairs_by_id[p] for p in sorted(set(pair_ids))]
    folds = family_folds(pairs)
    pid = np.array(pair_ids, dtype=object)
    preds, trues, per_fold = [], [], []
    lams = []
    for fold in folds:
        test_mask = np.isin(pid, np.array(fold["test"], dtype=object))
        train_mask = np.isin(pid, np.array(fold["train"], dtype=object))
        if not test_mask.any() or train_mask.sum() < pc_dim + 2:
            continue
        mu, basis = pc_basis(x[train_mask], pc_dim)
        w, zm, ym, lam = ridge_gcv_pc((x[train_mask] - mu) @ basis, y[train_mask])
        pred = ym + ((x[test_mask] - mu) @ basis - zm) @ w
        preds.append(pred)
        trues.append(y[test_mask])
        per_fold.append(_pooled_r2(y[test_mask], pred))
        lams.append(lam)
    if not preds:
        return {"pooled_r2": None, "per_fold_r2": [], "selected_lambdas": []}
    return {
        "pooled_r2": _pooled_r2(np.concatenate(trues), np.concatenate(preds)),
        "per_fold_r2": per_fold,
        "mean_fold_r2": float(np.mean(per_fold)),
        "selected_lambdas": lams,
        "note": "pooled over the fold-test union WITH duplicates (each pair tests under "
        "both endpoint folds); the family fold is the generalization-claim read",
    }


def fullspace_regularization_read(
    x: np.ndarray, y: np.ndarray, pair_ids: list[str], folds: list[list[str]]
) -> dict:
    """LABELED regularization-limit read: ambient Gram ridge at n<d, NO λ
    selection — per-λ OOF R² reported across a fixed grid (#1701/#1887)."""
    pid = np.array(pair_ids, dtype=object)
    lam_grid = np.geomspace(1e-2, 1e6, 9)
    per_lam = {}
    for lam in lam_grid:
        oof = np.full_like(y, np.nan)
        for fold_ids in folds:
            test_mask = np.isin(pid, np.array(list(fold_ids), dtype=object))
            train_mask = ~test_mask
            xm, ym = x[train_mask].mean(axis=0), y[train_mask].mean(axis=0)
            xc, yc = x[train_mask] - xm, y[train_mask] - ym
            k = xc @ xc.T
            alpha = np.linalg.solve(k + lam * np.eye(k.shape[0]), yc)
            oof[test_mask] = ym + (x[test_mask] - xm) @ xc.T @ alpha
        per_lam[float(lam)] = _pooled_r2(y, oof)
    return {
        "label": "regularization-limit read (full-space n<d fit; NEVER the headline; "
        "no lambda selection — the grid is reported)",
        "n_train_per_fold_lt_d": True,
        "per_lambda_pooled_oof_r2": per_lam,
    }


def _haar_rotation(k: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((k, k))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def _procrustes_cosine(wa: np.ndarray, wb: np.ndarray) -> float:
    """Input-rotation Procrustes-aligned operator cosine: max_R cos(vec(Wa R), vec(Wb))
    over orthogonal R = nuclear norm of Waᵀ Wb / (||Wa|| ||Wb||)."""
    s = np.linalg.svd(wa.T @ wb, compute_uv=False)
    denom = float(np.linalg.norm(wa) * np.linalg.norm(wb))
    return float(s.sum() / max(denom, 1e-30))


def _raw_cosine(wa: np.ndarray, wb: np.ndarray) -> float:
    return float((wa * wb).sum() / max(float(np.linalg.norm(wa) * np.linalg.norm(wb)), 1e-30))


def operator_comparison(
    w_l: np.ndarray,
    w_ref: np.ndarray,
    basis_in: np.ndarray,
    basis_out: np.ndarray,
    *,
    n_draws: int = ROTATION_NULL_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Direction-aware L-vs-reference comparison (#1345 battery conventions).

    Raw + Procrustes operator cosines, each vs its matched null: raw vs
    input-rotation null (cos(vec(W_L R), vec(W_ref))); Procrustes vs a
    spectrum-matched random-operator null (Procrustes cosine is invariant to
    input rotation, so its null re-draws Haar U,V at W_L's spectrum).
    Subspace-restricted to the shared PC bases (full-dim Haar nulls at 3584
    are out of VM scope; the full-ambient raw cosine is also reported).
    """
    raw_full = _raw_cosine(w_l, w_ref)
    wl_s = basis_in.T @ w_l @ basis_out
    wr_s = basis_in.T @ w_ref @ basis_out
    raw_sub = _raw_cosine(wl_s, wr_s)
    proc_sub = _procrustes_cosine(wl_s, wr_s)
    rng = np.random.default_rng(seed)
    k_in = wl_s.shape[0]
    u_l, s_l, vt_l = np.linalg.svd(wl_s, full_matrices=False)
    raw_null, proc_null = [], []
    for _ in range(n_draws):
        rot = _haar_rotation(k_in, rng)
        raw_null.append(_raw_cosine(rot.T @ wl_s, wr_s))
        w_rand = _haar_rotation(k_in, rng) @ np.diag(s_l) @ _haar_rotation(wl_s.shape[1], rng).T
        proc_null.append(_procrustes_cosine(w_rand, wr_s))
    raw_null_arr = np.abs(np.array(raw_null))
    proc_null_arr = np.array(proc_null)
    return {
        "raw_cosine_full_ambient": raw_full,
        "raw_cosine_subspace": raw_sub,
        "procrustes_cosine_subspace": proc_sub,
        "raw_null_p97_5_abs": float(np.percentile(raw_null_arr, 97.5)),
        "procrustes_null_p97_5": float(np.percentile(proc_null_arr, 97.5)),
        "procrustes_null_mean": float(proc_null_arr.mean()),
        "n_null_draws": n_draws,
        "subspace_dims": [int(wl_s.shape[0]), int(wl_s.shape[1])],
        "note": "direction-aware reads (raw + Procrustes operator cosine) vs matched "
        "nulls per the issue1345_operator_comparison conventions; nulls drawn in the "
        "shared PC subspaces (full-dim rotation nulls are compute-infeasible on the VM)",
    }


def _load_linearity_inputs(cfg: AnalysisConfig) -> dict:
    """Gather (x, y) observations for every fit family from the staged stores."""
    bank = _load_vc_bank(cfg)
    anchor_va = _load_anchor_va(cfg)
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    fams: dict[tuple[str, int], dict] = {
        (slot, layer): {"x": [], "y_tail_same": [], "y_span_19": [], "pair_ids": [], "alphas": []}
        for slot in FIT_SLOTS
        for layer in FIT_LAYERS
    }
    shards = sorted(cfg.va_dir.glob("shard_*.pt"))
    for shard in shards:
        slug = shard.stem.removeprefix("shard_")
        jsonl = cfg.rollouts_dir / f"shard_{slug}.jsonl"
        rows = list(_iter_jsonl(jsonl))
        if not rows:
            continue
        head = rows[0]
        if (
            head["arm"] != "steered"
            or head["vec_type"] != "A"
            or head["slot"] not in FIT_SLOTS
            or head["dose"] not in FIT_DOSES
            or not head["layer_variant"].startswith("L")
        ):
            continue
        layer = int(head["layer_variant"][1:])
        if layer not in FIT_LAYERS:
            continue
        va = torch.load(shard, map_location="cpu", weights_only=False)
        va_tail = va["va_tail"].float()
        va_span = va["va_span"].float()
        empty = set(va.get("empty_rows", []))
        fam = fams[(head["slot"], layer)]
        for i, r in enumerate(rows):
            if i in empty:
                continue
            pair = pairs_by_id[r["pair_id"]]
            delta, _state, _m = R._pair_payload(bank, pair, r["slot"], "A")
            d_l = delta[-1][layer].float().numpy()
            alpha = float(r["alpha"])
            fl_tail = anchor_va[r["context_a"]]["tail"][:, layer].mean(dim=0)
            fl_span19 = anchor_va[r["context_a"]]["span"][:, 19].mean(dim=0)
            fam["x"].append(alpha * d_l)
            fam["y_tail_same"].append((va_tail[i, layer] - fl_tail).numpy())
            fam["y_span_19"].append((va_span[i, 19] - fl_span19).numpy())
            fam["pair_ids"].append(r["pair_id"])
            fam["alphas"].append(alpha)
    return {"families": fams, "bank": bank, "anchor_va": anchor_va, "pairs_by_id": pairs_by_id}


def _fit_one_family(x: np.ndarray, y: np.ndarray, pair_ids: list[str], pairs_by_id: dict) -> dict:
    pair_folds = group_kfold_pairs(sorted(set(pair_ids)), N_PAIR_FOLDS, FOLD_SEED)
    res_pair = fit_family_folds(x, y, pair_ids, pair_folds)
    res_family = fit_family_family_folds(x, y, pair_ids, pairs_by_id)
    reg_read = fullspace_regularization_read(x, y, pair_ids, pair_folds)
    knn = {}
    mask = res_pair["covered_mask"]
    if mask.any():
        for metric in ("euclidean", "cosine"):
            knn[metric] = knn_retrieval(
                res_pair["oof_pred"][mask], y[mask], ks=(1, 5, 10), metric=metric
            )
    return {
        "n_obs": int(x.shape[0]),
        "d_in": int(x.shape[1]),
        "n_train_lt_d_ambient": True,
        "pc_dim": PC_DIM,
        "pair_fold": {
            k: _jsonable(v)
            for k, v in res_pair.items()
            if not k.startswith("oof") and k != "covered_mask"
        },
        "family_fold": _jsonable(res_family),
        "identity_bias_pooled_oof_r2": res_pair["pooled_r2_identity_bias"],
        "knn_retrieval": _jsonable(knn),
        "regularization_limit_read": _jsonable(reg_read),
        "_oof_pred": res_pair["oof_pred"],
        "_covered": res_pair["covered_mask"],
    }


def homogeneity_reads(inputs: dict, slot: str, layer: int) -> dict:
    """Plan §4.4(i): dose-cosine matrix (disattenuated) + log-log ||shift|| vs α."""
    fam = inputs["families"][(slot, layer)]
    anchor_va = inputs["anchor_va"]
    by_pair: dict[str, dict[float, np.ndarray]] = {}
    for xrow, y, pid, alpha in zip(
        fam["x"], fam["y_tail_same"], fam["pair_ids"], fam["alphas"], strict=True
    ):
        by_pair.setdefault(pid, {})[alpha] = y
    alphas = sorted({a for d in by_pair.values() for a in d})
    out = {}
    pairs_by_id = inputs["pairs_by_id"]
    for pid, shifts in sorted(by_pair.items()):
        if len(shifts) < 2:
            continue
        a_here = [a for a in alphas if a in shifts]
        mat = torch.tensor(np.stack([shifts[a] for a in a_here]), dtype=torch.float32)
        cos = FM.pairwise_shift_cosines(mat)
        pair = pairs_by_id[pid]
        fl = anchor_va[pair.a]["tail"][:, layer]
        rel_half = torch.stack(
            [
                FM.shift_split_half_reliability(
                    mat[i] + fl.mean(dim=0), fl, n_splits=20, seed=FOLD_SEED
                )
                for i in range(mat.shape[0])
            ]
        )
        rel = FM.spearman_brown(rel_half)
        disatt = FM.disattenuated_cosines(cos, rel)
        norms = mat.norm(dim=-1)
        rec = {
            "alphas": a_here,
            "cosine_matrix": _jsonable(cos),
            "reliabilities_sb": _jsonable(rel),
            "disattenuated_cosine_matrix": _jsonable(disatt),
            "shift_norms": _jsonable(norms),
            "degenerate": bool((norms <= 0).any()),
        }
        if not rec["degenerate"] and len(a_here) >= 2:
            slope, intercept = FM.log_log_magnitude_fit(
                torch.tensor(a_here, dtype=torch.float64), norms.unsqueeze(0)
            )
            rec["loglog_slope"] = float(slope[0])
            rec["loglog_intercept"] = float(intercept[0])
            if 1.0 in a_here:
                ref = FM.unity_slope_reference(
                    torch.tensor(a_here, dtype=torch.float64),
                    norms[a_here.index(1.0)].unsqueeze(0),
                )
                rec["unity_slope_reference_norms"] = _jsonable(ref[0])
        out[pid] = rec
    return out


def _additivity_read(cfg: AnalysisConfig, inputs: dict) -> dict | None:
    """OPTIONAL plan §4.4 additivity spot-check read (from unit-E stage-2 capture)."""
    pt = cfg.stage2_additivity_dir / "additivity_va.pt"
    if not pt.exists():
        logger.info("[linearity] no staged additivity capture at %s — skipping", pt)
        return None
    obj = torch.load(pt, map_location="cpu", weights_only=False)
    combos = obj["combos"]
    va = obj["va_span"].float()  # rows aligned with obj["index"]
    idx = {(rec["combo_id"], rec["role"]): i for i, rec in enumerate(obj["index"])}
    anchor_va = inputs["anchor_va"]
    layer = int(obj.get("steer_layer", 14))
    rows = []
    for combo in combos:
        cid = combo["combo_id"]
        ctx = combo["context_a"]
        fl_span = anchor_va[ctx]["span"].mean(dim=0)
        shifts = {}
        for role in ("d1", "d2", "d12"):
            i = idx[(cid, role)]
            shifts[role] = va[i] - fl_span  # (L, H)
        for read_layer, name in ((layer, f"span_L{layer}"), (PRIMARY_READ_LAYER, "span_L26")):
            rows.append(
                {
                    "combo_id": cid,
                    "pair_1": combo["pair_1"],
                    "pair_2": combo["pair_2"],
                    "context_a": ctx,
                    "read": name,
                    "cos_add": _nan_to_none(
                        FM.safe_cosine(
                            shifts["d12"][read_layer],
                            shifts["d1"][read_layer] + shifts["d2"][read_layer],
                        )
                    ),
                    "norm_ratio": float(
                        shifts["d12"][read_layer].norm()
                        / max(
                            float((shifts["d1"][read_layer] + shifts["d2"][read_layer]).norm()),
                            1e-30,
                        )
                    ),
                }
            )
    return {"combos": rows, "steer_layer": layer}


def phase_linearity(cfg: AnalysisConfig) -> int:
    """The L fit + baselines + retrieval + homogeneity + L vs M vs J + the 2x2."""
    logger.info("[phase=linearity]")
    inputs = _load_linearity_inputs(cfg)
    fams = inputs["families"]
    pairs_by_id = inputs["pairs_by_id"]
    nonempty = {k: v for k, v in fams.items() if v["x"]}
    assert nonempty, "no fit observations found — stage the grid stores first"

    # ── pilot gate (plan §9 P7 row): time ONE fold through THIS entrypoint. ──
    (slot0, layer0), fam0 = sorted(nonempty.items())[0]
    x0 = np.stack(fam0["x"]).astype(np.float64)
    y0 = np.stack(fam0["y_tail_same"]).astype(np.float64)
    folds0 = group_kfold_pairs(sorted(set(fam0["pair_ids"])), N_PAIR_FOLDS, FOLD_SEED)
    t0 = time.monotonic()
    fit_family_folds(x0, y0, fam0["pair_ids"], folds0[:1] or folds0)
    per_fold_s = time.monotonic() - t0
    n_fits = len(nonempty) * len(FIT_VARIANTS) * (N_PAIR_FOLDS + 15)
    projected_h = per_fold_s * n_fits / 3600.0
    pilot = {
        "criterion": "P7 fit pilot (plan §9): 1 fold timed through the production entrypoint",
        "s_per_fold": per_fold_s,
        "n_fold_fits_projected": n_fits,
        "projected_wall_h": projected_h,
        "planned_wall_h": cfg.planned_wall_h,
        "refusal_threshold_h": PILOT_REFUSAL_MULT * cfg.planned_wall_h,
        "recommended_fence_h": PILOT_FENCE_MULT * projected_h,
        "allowed": projected_h <= PILOT_REFUSAL_MULT * cfg.planned_wall_h,
        "repro": _repro(),
    }
    _write_json_atomic(cfg.out_root / "linearity" / "pilot_gate_report.json", pilot)
    logger.info(
        "[linearity] pilot: %.2fs/fold -> projected %.2f h (planned %.2f h) fence %.2f h",
        per_fold_s,
        projected_h,
        cfg.planned_wall_h,
        pilot["recommended_fence_h"],
    )
    if not pilot["allowed"] and not cfg.force:
        logger.error("[linearity] pilot REFUSAL (projected > %.1fx plan)", PILOT_REFUSAL_MULT)
        return RC_PILOT_GATE

    results: dict[str, dict] = {}
    fitted_ops: dict[str, dict] = {}
    t_start = time.monotonic()
    n_fam = 0
    for (slot, layer), fam in sorted(nonempty.items()):
        x = np.stack(fam["x"]).astype(np.float64)
        for variant in FIT_VARIANTS:
            y = np.stack(fam["y_tail_same" if variant == "same_tail" else "y_span_19"]).astype(
                np.float64
            )
            key = f"{slot}_L{layer}_{variant}"
            res = _fit_one_family(x, y, fam["pair_ids"], pairs_by_id)
            # Full-data operator for the direction-aware comparison.
            mu, basis_in = pc_basis(x, PC_DIM)
            w_pc, zm, ym, lam = ridge_gcv_pc((x - mu) @ basis_in, y)
            _mu_y, basis_out = pc_basis(y, PC_DIM)
            fitted_ops[key] = {
                "w_ambient": basis_in @ w_pc,  # (d_in, d_out) low-rank
                "basis_in": basis_in,
                "basis_out": basis_out,
                "lambda": lam,
            }
            res.pop("_oof_pred")
            res.pop("_covered")
            results[key] = res
            n_fam += 1
            logger.info(
                "[linearity] family %d/%d %s pair-fold R2=%s family-fold R2=%s elapsed=%.1fs",
                n_fam,
                len(nonempty) * len(FIT_VARIANTS),
                key,
                res["pair_fold"]["pooled_r2"],
                res["family_fold"]["pooled_r2"],
                time.monotonic() - t_start,
            )

    # ── direction-aware comparison: L vs banked M vs #1776 J (space-matched). ──
    parity = json.loads(cfg.map_parity_json.read_text()) if cfg.map_parity_json.exists() else None
    comparisons: dict[str, dict] = {}
    two_by_two: dict[str, dict] = {}
    if parity is not None:
        # L14->L14 (tail) vs M_779@L14.
        key = "ce_L14_same_tail"
        if key in fitted_ops:
            bundle = _load_bundle(cfg.maps_dir / BANKED_MAPS[0]["repo_path"])  # m779_ce_L14
            orientation = _orientation_for(parity, "m779_ce_L14")
            w = bundle["W"].double().numpy()
            m_eff = (1.0 / bundle["xsd"].double().numpy())[:, None] * (
                w if orientation == "zW" else w.T
            )
            op = fitted_ops[key]
            comparisons["L14_vs_M779_L14"] = operator_comparison(
                op["w_ambient"], m_eff, op["basis_in"], op["basis_out"]
            )
            fam_r2 = results[key]["family_fold"]["pooled_r2"]
            cmp_row = comparisons["L14_vs_M779_L14"]
            two_by_two["ce_L14"] = {
                "M_aligns": bool(
                    cmp_row["procrustes_cosine_subspace"] > cmp_row["procrustes_null_p97_5"]
                ),
                "L_predicts": bool(fam_r2 is not None and fam_r2 > 0),
                "criteria": {
                    "M_aligns": "procrustes_cosine_subspace > procrustes rotation-null p97.5",
                    "L_predicts": "held-out context-family pooled OOF R2 > 0",
                },
                "family_fold_r2": fam_r2,
                "procrustes_cosine": cmp_row["procrustes_cosine_subspace"],
                "procrustes_null_p97_5": cmp_row["procrustes_null_p97_5"],
            }
        # L14->L19 (span) vs J_last (L14 last-token -> L19 answer-mean).
        key = "ce_L14_j19_span"
        if key in fitted_ops:
            j = _load_jacobian(cfg.maps_dir / JACOBIAN_SPEC["repo_path"]).double().numpy()
            if j.shape != (HIDDEN, HIDDEN):
                j = j.T if j.T.shape == (HIDDEN, HIDDEN) else j
            op = fitted_ops[key]
            comparisons["L1419_vs_J1776"] = operator_comparison(
                op["w_ambient"], j, op["basis_in"], op["basis_out"]
            )
            fam_r2 = results[key]["family_fold"]["pooled_r2"]
            cmp_row = comparisons["L1419_vs_J1776"]
            two_by_two["ce_L14_to_L19"] = {
                "M_aligns": bool(
                    cmp_row["procrustes_cosine_subspace"] > cmp_row["procrustes_null_p97_5"]
                ),
                "L_predicts": bool(fam_r2 is not None and fam_r2 > 0),
                "criteria": {
                    "M_aligns": "procrustes_cosine_subspace > procrustes rotation-null p97.5",
                    "L_predicts": "held-out context-family pooled OOF R2 > 0",
                },
                "family_fold_r2": fam_r2,
                "procrustes_cosine": cmp_row["procrustes_cosine_subspace"],
                "procrustes_null_p97_5": cmp_row["procrustes_null_p97_5"],
            }

    homog = {
        f"{slot}_L{layer}": homogeneity_reads(inputs, slot, layer) for (slot, layer) in nonempty
    }
    additivity = _additivity_read(cfg, inputs)

    ldir = cfg.out_root / "linearity"
    _write_json_atomic(ldir / "l_fit_results.json", {"families": results, "repro": _repro()})
    _write_json_atomic(
        ldir / "operator_comparison.json",
        {"comparisons": comparisons, "two_by_two": two_by_two, "repro": _repro()},
    )
    _write_json_atomic(ldir / "homogeneity.json", {"families": homog, "repro": _repro()})
    if additivity is not None:
        _write_json_atomic(ldir / "additivity.json", {**additivity, "repro": _repro()})
    logger.info("[phase=linearity_done] families=%d comparisons=%d", len(results), len(comparisons))
    return RC_OK


# ── bootstrap (phase bootstrap) ────────────────────────────────────────


def bootstrap_family_means_batched(
    values: np.ndarray, n_boot: int, seed: int, *, block: int = 2000
) -> np.ndarray:
    """Pair-clustered bootstrap means for MANY families at once (batched
    index-GEMMs — the null_battery subset-sum pattern; NO per-draw loop).

    ``values``: (n_pairs, n_families), NaN = cell unavailable for that family.
    Returns (n_boot, n_families) NaN-aware resampled means: draw d resamples
    the PAIR axis with replacement, mean over the drawn pairs' non-NaN cells.
    """
    n, f = values.shape
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(values)
    v0 = np.where(mask, values, 0.0)
    out = np.empty((n_boot, f), dtype=np.float64)
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.zeros((b, n), dtype=np.float64)
        np.add.at(counts, (np.arange(b)[:, None], idx), 1.0)
        num = counts @ v0  # (b, f) — ONE GEMM per block over all families
        den = counts @ mask.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + b] = np.where(den > 0, num / den, np.nan)
    return out


def _bootstrap_family_means_naive(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Serial reference twin (tests only — equivalence pin for the batched path).

    Draws the SAME index matrix as one batched block (identical RNG stream), so
    the equivalence test pins the count-matrix GEMM math, not RNG coincidence.
    """
    n, f = values.shape
    rng = np.random.default_rng(seed)
    idx_all = rng.integers(0, n, size=(n_boot, n))
    out = np.empty((n_boot, f), dtype=np.float64)
    for d in range(n_boot):
        drawn = values[idx_all[d]]
        with np.errstate(invalid="ignore"):
            out[d] = np.nanmean(drawn, axis=0)
    return out


def _family_key(row: dict, metric: str) -> str:
    return "|".join(
        [
            row["arm"],
            row["setting"],
            row["slot"],
            row["layer_variant"],
            row["dose"],
            row["vec_type"],
            metric,
        ]
    )


def _cell_metric(row: dict, metric: str) -> float:
    if metric == "f_act":
        v = row.get("f_act")
    else:
        kind = metric.removeprefix("f_beh_")
        v = (row.get("f_beh") or {}).get(kind, {}).get("f_beh")
    return float("nan") if v is None else float(v)


def phase_bootstrap(cfg: AnalysisConfig) -> int:
    """Pair-clustered B=10,000 bootstrap CIs per cell family (batched)."""
    logger.info("[phase=bootstrap]")
    rows = list(_iter_jsonl(cfg.fmetrics_dir / "f_cells.jsonl")) + list(
        _iter_jsonl(cfg.fmetrics_dir / "null_cells.jsonl")
    )
    assert rows, "no f-table rows — run --phase ftables first"
    pairs = BANK.build_pairs()
    pair_ids_by_setting = {
        s: sorted(p.pair_id for p in pairs if p.setting == s)
        for s in ("matched_prefix", "matched_query", "cross")
    }
    out: dict[str, dict] = {}
    t0 = time.monotonic()
    for setting, pids in pair_ids_by_setting.items():
        pid_idx = {p: i for i, p in enumerate(pids)}
        fam_values: dict[str, np.ndarray] = {}
        for row in rows:
            if row["setting"] != setting:
                continue
            metrics = ["f_act"] + [f"f_beh_{k}" for k in (row.get("f_beh") or {})]
            for metric in metrics:
                key = _family_key(row, metric)
                arr = fam_values.setdefault(key, np.full(len(pids), np.nan))
                arr[pid_idx[row["pair_id"]]] = _cell_metric(row, metric)
        if not fam_values:
            continue
        keys = sorted(fam_values)
        values = np.stack([fam_values[k] for k in keys], axis=1)  # (n_pairs, n_fams)

        # pilot slice (plan §9: draw battery pilot-gated at entry).
        t_p = time.monotonic()
        bootstrap_family_means_batched(values, 200, BOOTSTRAP_SEED)
        proj_h = (time.monotonic() - t_p) * (cfg.bootstrap_b / 200) / 3600.0
        if proj_h > PILOT_REFUSAL_MULT * cfg.planned_wall_h and not cfg.force:
            _write_json_atomic(
                cfg.fmetrics_dir / "bootstrap_pilot_report.json",
                {"setting": setting, "projected_h": proj_h, "planned_h": cfg.planned_wall_h},
            )
            logger.error("[bootstrap] pilot REFUSAL: projected %.2f h", proj_h)
            return RC_PILOT_GATE

        boots = bootstrap_family_means_batched(values, cfg.bootstrap_b, BOOTSTRAP_SEED)
        with np.errstate(invalid="ignore"):
            obs = np.nanmean(values, axis=0)
        for j, key in enumerate(keys):
            col = boots[:, j]
            valid = col[~np.isnan(col)]
            out[key] = {
                "setting": setting,
                "observed_mean": _nan_to_none(obs[j]),
                "n_pairs_used": int((~np.isnan(values[:, j])).sum()),
                "ci_lo": float(np.percentile(valid, 2.5)) if valid.size else None,
                "ci_hi": float(np.percentile(valid, 97.5)) if valid.size else None,
                "n_valid_draws": int(valid.size),
            }
        logger.info(
            "[bootstrap] setting=%s families=%d elapsed=%.1fs",
            setting,
            len(keys),
            time.monotonic() - t0,
        )
    _write_json_atomic(
        cfg.fmetrics_dir / "bootstrap_cis.json",
        {
            "B": cfg.bootstrap_b,
            "seed": BOOTSTRAP_SEED,
            "resample_axis": "pairs (pair-clustered, within setting)",
            "families": out,
            "repro": _repro(),
        },
    )
    logger.info("[phase=bootstrap_done] families=%d", len(out))
    return RC_OK


# ── fragility (phase fragility) ────────────────────────────────────────


def phase_fragility(cfg: AnalysisConfig) -> int:
    """Excess-incoherence per (slot, layer_variant, dose) + cap-hit companion."""
    logger.info("[phase=fragility]")
    anchor_rows = list(_iter_jsonl(cfg.fmetrics_dir / "anchor_draws.jsonl"))
    assert anchor_rows, "run --phase ftables first (anchor_draws.jsonl missing)"
    anchor_incoh = float(np.mean([not r["coherent"] for r in anchor_rows]))
    anchor_cap = float(np.mean([bool(r["cap_hit"]) for r in anchor_rows]))

    cells: dict[tuple, dict] = {}
    for name, arm in (("f_cells.jsonl", "steered"), ("null_cells.jsonl", "null")):
        for row in _iter_jsonl(cfg.fmetrics_dir / name):
            key = (row["slot"], row["layer_variant"], row["dose"])
            rec = cells.setdefault(
                key,
                {
                    "slot": key[0],
                    "layer_variant": key[1],
                    "dose": key[2],
                    "steered": {"n": 0, "incoherent": 0, "cap_hit": 0},
                    "null": {"n": 0, "incoherent": 0, "cap_hit": 0},
                    "by_setting": {},
                },
            )
            side = rec[arm]
            side["n"] += 1
            side["incoherent"] += int(row["excluded_incoherent"])
            side["cap_hit"] += int(row["cap_hit"])
            st = rec["by_setting"].setdefault(
                row["setting"],
                {
                    "steered": {"n": 0, "incoherent": 0},
                    "null": {"n": 0, "incoherent": 0},
                },
            )
            st[arm]["n"] += 1
            st[arm]["incoherent"] += int(row["excluded_incoherent"])
    out = []
    for key in sorted(cells):
        rec = cells[key]
        for side in ("steered", "null"):
            s = rec[side]
            s["incoherent_frac"] = s["incoherent"] / s["n"] if s["n"] else None
            s["cap_hit_frac"] = s["cap_hit"] / s["n"] if s["n"] else None
            s["excess_incoherence"] = (
                s["incoherent_frac"] - anchor_incoh if s["incoherent_frac"] is not None else None
            )
        out.append(rec)
    _write_json_atomic(
        cfg.out_root / "fragility" / "fragility_cells.json",
        {
            "anchor_baseline": {"incoherent_frac": anchor_incoh, "cap_hit_frac": anchor_cap},
            "cells": out,
            "note": "excess incoherence = per-(slot, layer-variant, dose) incoherent "
            "fraction minus the anchor baseline rate; donor-null side by side; cap-hit "
            "counted NEXT TO but never blended with incoherence (plan §4.5)",
            "repro": _repro(),
        },
    )
    logger.info("[phase=fragility_done] cells=%d", len(out))
    return RC_OK


# ── stage-2 selection (phase select-stage2) ────────────────────────────


def select_best_cells(
    rows: list[dict],
    *,
    restriction: dict[str, tuple[int, ...]] | None = None,
    max_cells: int = STAGE2_MAX_CELLS,
    min_pairs: int = 3,
) -> dict:
    """Body-verbatim stage-2 selection: per setting × level, the single best
    (layer, slot) cell with best layer RESTRICTED to the banked-map sets
    (ce ∈ {14,19}; pe ∈ {14,19,26}); at most ``max_cells`` distinct cells."""
    restriction = restriction or STAGE2_LAYER_RESTRICTION
    allowed_variants = {
        slot: {f"L{layer}" for layer in layers} for slot, layers in restriction.items()
    }
    stats: dict[tuple, dict] = {}
    for row in rows:
        if row["arm"] != "steered":
            continue
        slot = row["slot"]
        if slot not in allowed_variants or row["layer_variant"] not in allowed_variants[slot]:
            continue
        fam = (row["setting"], slot, row["layer_variant"], row["dose"], row["vec_type"])
        rec = stats.setdefault(fam, {"f_act": [], "f_beh": []})
        if row.get("f_act") is not None:
            rec["f_act"].append(float(row["f_act"]))
        beh = [
            v["f_beh"]
            for v in (row.get("f_beh") or {}).values()
            if isinstance(v, dict) and v.get("f_beh") is not None
        ]
        if beh:
            rec["f_beh"].append(float(np.mean(beh)))
    selections: dict[str, dict] = {}
    for setting in ("matched_prefix", "matched_query", "cross"):
        for level, metric in (("activation", "f_act"), ("behavior", "f_beh")):
            best = None
            for fam, rec in stats.items():
                if fam[0] != setting or len(rec[metric]) < min_pairs:
                    continue
                mean_f = float(np.mean(rec[metric]))
                if best is None or mean_f > best["mean_f"]:
                    best = {
                        "setting": fam[0],
                        "slot": fam[1],
                        "layer_variant": fam[2],
                        "dose": fam[3],
                        "vec_type": fam[4],
                        "mean_f": mean_f,
                        "n_pairs_used": len(rec[metric]),
                    }
            if best is not None:
                selections[f"{setting}|{level}"] = {**best, "level": level}
    # Dedupe identical cells; cap at max_cells by descending |mean_f|.
    seen: dict[tuple, list[str]] = {}
    for key, sel in selections.items():
        cell = (sel["setting"], sel["slot"], sel["layer_variant"], sel["dose"], sel["vec_type"])
        seen.setdefault(cell, []).append(key)
    cells = [
        {
            "setting": c[0],
            "slot": c[1],
            "layer_variant": c[2],
            "dose": c[3],
            "vec_type": c[4],
            "selected_for": sorted(keys),
            "mean_f": max(selections[k]["mean_f"] for k in keys),
        }
        for c, keys in seen.items()
    ]
    cells.sort(key=lambda r: -abs(r["mean_f"]))
    cells = cells[:max_cells]
    assert len(cells) <= max_cells, len(cells)
    for cell in cells:
        assert int(cell["layer_variant"][1:]) in restriction[cell["slot"]], cell
    return {
        "selections": selections,
        "cells": cells,
        "restriction": {k: list(v) for k, v in restriction.items()},
        "selection_rule": "argmax over cell families of the mean level-F across pairs "
        "(coherent, non-degenerate cells only; min_pairs floor); LABELED post-selection "
        "(plan §6 — stage-2 is a confirmation, never an unbiased estimate)",
        "post_selection": True,
        "min_pairs": min_pairs,
    }


def phase_select_stage2(cfg: AnalysisConfig) -> int:
    logger.info("[phase=select-stage2]")
    rows = list(_iter_jsonl(cfg.fmetrics_dir / "f_cells.jsonl"))
    assert rows, "run --phase ftables first"
    payload = select_best_cells(rows)
    payload["repro"] = _repro()
    out = cfg.out_root / "best_cells.json"
    _write_json_atomic(out, payload)
    logger.info(
        "[select-stage2] %d cells (of <=%d) -> %s", len(payload["cells"]), STAGE2_MAX_CELLS, out
    )
    if cfg.no_upload:
        logger.info("[select-stage2] upload skipped (--no-upload)")
        return RC_OK
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        out,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/stage2_spec/best_cells.json",
        upload_as_file=True,
        raise_on_error=True,  # fail-loud (plan §9 phase_outputs: P8 stages this file)
    )
    logger.info("[phase=select-stage2_done] uploaded -> %s", url)
    return RC_OK


# ── entrypoint ─────────────────────────────────────────────────────────

PHASES = {
    "stage": phase_stage,
    "parity": phase_parity,
    "ftables": phase_ftables,
    "transport": phase_transport,
    "linearity": phase_linearity,
    "bootstrap": phase_bootstrap,
    "fragility": phase_fragility,
    "select-stage2": phase_select_stage2,
}
PHASE_ORDER = tuple(PHASES)


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths."""
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload,
        retry_transient,
        stage_hub_file,
        stage_hub_prefix,
    )

    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 VM-side P7 analysis (stage/parity/F-tables/transport/"
        "linearity/bootstrap/fragility/stage-2 selection).",
        epilog=(
            "VM launches carry the shared-VM caps: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
            "OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 "
            "uv run python scripts/issue2094_analysis.py --phase <phase>"
        ),
    )
    ap.add_argument("--phase", choices=(*PHASE_ORDER, "all"), default=None)
    ap.add_argument("--in-root", type=Path, default=Path("data/issue_2094/hf_dl"))
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument("--judge-root", type=Path, default=None)
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--coverage",
        choices=("full", "staged"),
        default="full",
        help="cell-coverage gate mode (staged = smoke/partial stagings)",
    )
    ap.add_argument("--force", action="store_true", help="override pilot-gate refusals")
    ap.add_argument("--planned-wall-h", type=float, default=PLANNED_P7_WALL_H)
    ap.add_argument("--projected-stage-gb", type=float, default=20.0)
    ap.add_argument("--skip-disk-check", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-profiles", action="store_true", help="skip 28-layer F_act profiles")
    ap.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        in_root=args.in_root,
        out_root=args.out_root,
        judge_root=args.judge_root if args.judge_root is not None else args.out_root / "judge",
        hf_revision=args.hf_revision,
        smoke=args.smoke,
        coverage="staged" if args.smoke and args.coverage == "full" else args.coverage,
        force=args.force,
        planned_wall_h=args.planned_wall_h,
        projected_stage_gb=args.projected_stage_gb,
        skip_disk_check=args.skip_disk_check,
        no_upload=args.no_upload,
        profiles=not args.no_profiles,
        bootstrap_b=args.bootstrap_b,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    if args.phase != "all":
        return PHASES[args.phase](cfg)
    for name in PHASE_ORDER:
        rc = PHASES[name](cfg)
        if rc != RC_OK:
            logger.error("[all] phase %s halted rc=%d", name, rc)
            return rc
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
