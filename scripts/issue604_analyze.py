# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, σ, γ, ※, —) in scientific docstrings + labels.
"""Task #604 Phase C — key/write/rotation/constancy/selectivity reads (VM CPU).

Consumes Phase A spectra/vectors + the Phase B context-vector bundle +
stored measured-shift tensors, and writes one JSON per registered read
(checkpoint-per-analysis — each file lands the moment its analysis
completes):

- ``key_match.json``      §4 C2 — cos(top right-singular vector, source
  context) per layer × comparison space, vs the wrong-context null bank,
  with rank + selectivity margin; shuffled-pairing null; #541 own-bank read.
- ``write_match.json``    §4 C3 — pooled σ-weighted residual write vs the
  stored measured shifts (dial L20; #519/#521 L14 variants side by side;
  EM control vs the #551 shared direction, matched seed primary).
- ``rotation.json``       §4 C4 — Δcos vs realized dose: 30-cell primary,
  joint per-source secondary, all-54 sensitivity, component cosines +
  placebo + contrast-geometry diagnostics, cluster bootstrap; #474 ladder
  with the pinned paired OLS-slope statistic.
- ``functional_constancy.json``  §4 C5 — Wang-et-al.-style pairwise |cos|
  of Δout across bank contexts (rank-8 truncated; truncation energy
  recorded in Phase A spectra), + the gate tie-in.
- ``selectivity.json``    §4 C7 + joint-arm subspace read — seed-stability
  matrices, per-cell band-mean selectivity margins.

All comparisons are dimension-asserted to the 3584-d residual stream
(comparison-space validity map). Missing cells/artifacts are reported
``N/A — not stored`` / ``N/A — artifact failed fitness at load`` (plan
§15), never fabricated; the script enumerates ONLY what Phases A/B
actually produced (smoke = same entrypoint over the smoke outputs).

Usage:
    uv run python scripts/issue604_analyze.py            # all analyses (production)
    uv run python scripts/issue604_analyze.py --analyses key,rotation
    # smoke over a deliberate tiny bundle (separate dir, partial contexts):
    uv run python scripts/issue604_analyze.py \
        --context-dir eval_results/issue_604/context_vectors_smoke \
        --expect-probes 2 --allow-missing-contexts

The Phase B bundle is fitness-gated before use (stale-cache guard): the
bundle meta must carry the expected ``n_probes`` (production = 50) and every
context the loaded cell inventory requires must resolve — a stale smoke
bundle at the production path fails loud instead of corrupting the reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.analysis.svd_direction_constancy import (  # noqa: E402
    assemble_M,
    spearman_rho,
    svd_summary,
)
from explore_persona_space.experiments.issue_604 import (  # noqa: E402
    EXTRACT_LAYER_519,
    EXTRACT_LAYER_DIAL,
    HF_DATA_REPO,
    HF_PRIVATE_DATA_REPO,
    HIDDEN_SIZE,
    KEY_LAYER_BAND,
    result_metadata,
    seed_group_key,
)

logger = logging.getLogger("issue604.phase_c")

OUT_DIR_DEFAULT = PROJECT_ROOT / "eval_results/issue_604"
RNG_SEED = 604
N_BOOT = 2000

DIAL_LINES = ("dial527", "dial550", "dial538")
DIAL_SHIFT_JSON_DIR = {527: "issue_527/eval", 550: "issue_550/eval", 538: "issue_538/eval"}
DIAL_SHIFT_PT_PREFIX = {
    527: "issue_527/eval",
    550: "issue_550/analysis_tensors",
    538: "issue_538/eval",
}
I551_SHIFT_PREFIX = "issue551_shift_reextract/analysis_tensors/shifts"
I552_SHIFT_DIR = "eval_results/issue_552/marker-arm-mean-resp-reextraction/shifts"
BANK111_PT = "eval_results/single_token_100_persona/centroids/centroids_layer20.pt"
BANK111_NAMES = "eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json"
BANK24_PT = "eval_results/issue_274/centroids/centroids_n24_layers0_27.pt"
I474_CROSS_EVAL = "eval_results/issue_474/cross_eval"
I541_ACT_PREFIX = "issue541_prior_stratified/geometry_plus_prior"
I541_LAYERS = (7, 14, 21, 22, 27)


# ── small numerics ──────────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalize a 1-D vector (fail loud on zero norm)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    assert n > 0, "zero-norm vector in comparison"
    return v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Signed cosine between two 1-D vectors."""
    return float(np.dot(_unit(a), _unit(b)))


# ── cell store (Phase A outputs) ────────────────────────────────────────────


class CellStore:
    """Loads Phase A per-cell spectra JSON + vectors npz on demand."""

    def __init__(self, out_dir: Path):
        self.spectra_dir = out_dir / "spectra"
        self.vectors_dir = out_dir / "vectors"
        self.cells: list[dict] = []
        for spath in sorted(self.spectra_dir.glob("*/*.json")):
            payload = json.loads(spath.read_text())
            npz_path = self.vectors_dir / spath.parent.name / (spath.stem + ".npz")
            if not npz_path.exists():
                logger.warning("vectors npz missing for %s — cell skipped", spath)
                continue
            payload["_npz_path"] = npz_path
            self.cells.append(payload)
        logger.info("cell store: %d cells with spectra+vectors", len(self.cells))

    def vectors(self, cell: dict) -> dict[str, np.ndarray]:
        """The npz arrays for one cell (loaded lazily, cached on the dict)."""
        if "_npz" not in cell:
            cell["_npz"] = dict(np.load(cell["_npz_path"]))
        return cell["_npz"]

    def key_top1(self, cell: dict, layer: int, stack: str = "attn_key") -> np.ndarray | None:
        arrs = self.vectors(cell)
        k = arrs.get(f"L{layer}__{stack}__V8")
        if k is None:
            return None
        v = k[:, 0].astype(np.float64)
        assert v.shape == (HIDDEN_SIZE,), v.shape
        return v

    def key_top2(self, cell: dict, layer: int, stack: str = "attn_key") -> np.ndarray | None:
        arrs = self.vectors(cell)
        k = arrs.get(f"L{layer}__{stack}__V8")
        if k is None or k.shape[1] < 2:
            return None
        return k[:, :2].astype(np.float64)

    def write_top1(self, cell: dict, layer: int) -> tuple[np.ndarray, float] | None:
        arrs = self.vectors(cell)
        u = arrs.get(f"L{layer}__resid_write__U8")
        s = arrs.get(f"L{layer}__resid_write__S")
        if u is None or s is None:
            return None
        v = u[:, 0].astype(np.float64)
        assert v.shape == (HIDDEN_SIZE,), v.shape
        return v, float(s[0])

    def layers(self, cell: dict) -> list[int]:
        return [rec["layer"] for rec in cell["layers"]]


# ── context bundle (Phase B outputs) ────────────────────────────────────────


class ContextBundle:
    """Phase B centroids + γ; downloads from the HF data repo if absent.

    Fitness-gates the bundle BEFORE any analysis consumes it (stale-cache
    guard): ``expected_n_probes`` must match the bundle meta (production = 50;
    a smoke read passes its own value explicitly), and every name in
    ``required_contexts`` must resolve. A stale smoke bundle sitting at the
    production path fails loud here instead of silently corrupting Phase C.
    """

    def __init__(
        self,
        context_dir: Path,
        *,
        expected_n_probes: int = 50,
        required_contexts: tuple[str, ...] = (),
    ):
        import torch

        needed = ["module_input_centroids.pt", "context_vectors_all_layers.pt", "rmsnorm_gamma.pt"]
        if not all((context_dir / n).exists() for n in needed):
            from huggingface_hub import hf_hub_download

            context_dir.mkdir(parents=True, exist_ok=True)
            for n in [*needed, "manifest.json"]:
                logger.info("downloading Phase B bundle file %s from HF", n)
                got = hf_hub_download(
                    HF_DATA_REPO, f"issue604_adapter_svd/analysis_tensors/{n}", repo_type="dataset"
                )
                (context_dir / n).write_bytes(Path(got).read_bytes())
        mod = torch.load(context_dir / "module_input_centroids.pt", weights_only=True)
        raw = torch.load(context_dir / "context_vectors_all_layers.pt", weights_only=True)
        gam = torch.load(context_dir / "rmsnorm_gamma.pt", weights_only=True)
        self.attn = {k: v.numpy().astype(np.float64) for k, v in mod["attn"].items()}
        self.mlp = {k: v.numpy().astype(np.float64) for k, v in mod["mlp"].items()}
        self.raw = {k: v.numpy().astype(np.float64) for k, v in raw["contexts"].items()}
        self.gamma_in = gam["input_layernorm"].numpy().astype(np.float64)
        self.gamma_post = gam["post_attention_layernorm"].numpy().astype(np.float64)
        self.meta = mod.get("meta") or {}
        names = sorted(self.attn.keys())
        first = self.attn[names[0]]
        self.n_layers, self.hidden = first.shape
        assert all(self.attn[n].shape == (self.n_layers, self.hidden) for n in names)
        self.names = names
        self.manifest = {}
        mpath = context_dir / "manifest.json"
        if mpath.exists():
            self.manifest = json.loads(mpath.read_text())
        # ── Fitness gate (stale-cache guard) ────────────────────────────────
        got_probes = self.meta.get("n_probes")
        if got_probes != expected_n_probes:
            raise RuntimeError(
                f"Phase B bundle at {context_dir} fails fitness: n_probes={got_probes!r} != "
                f"expected {expected_n_probes} (model={self.meta.get('model')!r}, "
                f"n_contexts={self.meta.get('n_contexts')!r}, "
                f"dtype={self.meta.get('dtype')!r}). A stale/smoke bundle cannot feed this "
                "Phase C run — delete it, point --context-dir at the right bundle, or pass "
                "--expect-probes for a deliberate smoke read."
            )
        missing = []
        for name in required_contexts:
            try:
                self.resolve(name)
            except KeyError:
                missing.append(name)
        if missing:
            raise RuntimeError(
                f"Phase B bundle at {context_dir} lacks required runtime contexts: "
                f"{sorted(missing)} — the Phase A cell inventory needs them. Re-run Phase B "
                "over the full context union (or pass --allow-missing-contexts for a "
                "deliberate smoke read over a partial bundle)."
            )
        logger.info(
            "context bundle: %d contexts, %d layers, hidden %d (n_probes=%s, fitness OK)",
            len(names),
            self.n_layers,
            self.hidden,
            got_probes,
        )

    def vec(self, name: str, layer: int, space: str) -> np.ndarray:
        """Context vector for ``name`` at ``layer`` in a comparison space.

        Spaces: ``attn`` / ``mlp`` (TRUE module-input centroids, primary),
        ``gamma_raw`` (post-hoc γ⊙raw sensitivity), ``raw``.
        """
        if space == "attn":
            return self.attn[name][layer]
        if space == "mlp":
            return self.mlp[name][layer]
        if space == "gamma_raw":
            return self.gamma_in[layer] * self.raw[name][layer]
        if space == "raw":
            return self.raw[name][layer]
        raise ValueError(space)

    def resolve(self, name: str) -> str:
        """Group-aware context-name resolution (e.g. #518 rename on conflict)."""
        if name in self.attn:
            return name
        for suffix in ("__i518_sources", "__dial_trained", "__i519_trained"):
            if f"{name}{suffix}" in self.attn:
                return f"{name}{suffix}"
        raise KeyError(f"context {name!r} not in Phase B bundle")


# ── shared loaders ──────────────────────────────────────────────────────────


def load_dial_shift_json(data_root: Path, issue: int, slug: str) -> dict:
    """Per-cell eval shift JSON (git) — dose covariate + eval_panel order."""
    path = data_root / "eval_results" / f"issue_{issue}" / "eval" / f"{slug}__shift.json"
    return json.loads(path.read_text())


def load_dial_shift_pt(issue: int, slug: str) -> np.ndarray:
    """Per-cell (19, 3584) L20 shift matrix from the HF data repo.

    These pickle numpy globals — ``weights_only=True`` always fails on them
    (plan §8); they come from our own Hub-verified repos.
    """
    import torch
    from huggingface_hub import hf_hub_download

    rel = f"{DIAL_SHIFT_PT_PREFIX[issue]}/{slug}__shift.pt"
    path = hf_hub_download(HF_DATA_REPO, rel, repo_type="dataset")
    obj = torch.load(path, weights_only=False)  # known-needed (plan §8 risk row)
    arr = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
    arr = arr.astype(np.float64)
    assert arr.shape == (19, HIDDEN_SIZE), (slug, arr.shape)
    return arr


def load_i551_shifts(variant: str, arm: str, seed: int) -> dict:
    """{persona: {delta_v: (3584,), ...}} from the private repo (#551)."""
    import torch
    from huggingface_hub import hf_hub_download

    rel = f"{I551_SHIFT_PREFIX}/{variant}_{arm}_seed{seed}.pt"
    path = hf_hub_download(HF_PRIVATE_DATA_REPO, rel, repo_type="dataset")
    obj = torch.load(path, weights_only=False)
    shifts = obj["shifts"] if isinstance(obj, dict) and "shifts" in obj else obj
    assert isinstance(shifts, dict) and len(shifts) >= 10, type(shifts)
    return shifts


def cell_dose(data_root: Path, cell: dict) -> dict[str, float]:
    """{source: delta_logp_marker} re-measured per cell (plan §4 C4(b))."""
    issue = int(cell["cell"]["line"].removeprefix("dial"))
    payload = load_dial_shift_json(data_root, issue, cell["cell"]["cell_id"])
    return {
        src: float(payload["contexts"][src]["delta_logp_marker"])
        for src in cell["cell"]["source_personas"]
    }


def is_dial(cell: dict) -> bool:
    return cell["cell"]["line"] in DIAL_LINES


def is_clean_single_dial(cell: dict) -> bool:
    """Member of the 30-cell PRIMARY rotation denominator (plan §4 C4(a))."""
    c = cell["cell"]
    return (
        is_dial(cell)
        and c["arm"] in ("A_only", "B_only")
        and "panel-contaminated" not in c["tags"]
        and "checkpoint-intermediate" not in c["tags"]
    )


# ── analysis 1: key match ───────────────────────────────────────────────────


def run_key_match(  # noqa: C901 — one block per registered read (per-space rows, band run, shuffled null)
    store: CellStore, bundle: ContextBundle, data_root: Path, out: Path
) -> None:
    """§4 C2 — per-cell per-layer key-vs-context cosines against the bank."""
    spaces_by_stack = {"attn_key": ["attn", "gamma_raw", "raw"], "mlp_key": ["mlp"]}
    results = []
    band = [li for li in KEY_LAYER_BAND if li < bundle.n_layers]
    for cell in store.cells:
        c = cell["cell"]
        if c["line"] == "i541":
            continue  # own-bank read below
        try:
            sources = [bundle.resolve(s) for s in c["source_personas"]]
        except KeyError as exc:
            results.append(
                {
                    "line": c["line"],
                    "cell_id": c["cell_id"],
                    "tags": c["tags"],
                    "per_source": [],
                    "status": f"N/A — source context not in Phase B bundle: {exc}",
                }
            )
            continue
        rec = {"line": c["line"], "cell_id": c["cell_id"], "tags": c["tags"], "per_source": []}
        for src in sources:
            src_rec = {"source": src, "stacks": {}}
            for stack, spaces in spaces_by_stack.items():
                per_space = {}
                for space in spaces:
                    rows = []
                    for layer in range(bundle.n_layers):
                        key = store.key_top1(cell, layer, stack)
                        if key is None:
                            continue
                        cos_all = {
                            name: abs(_cos(key, bundle.vec(name, layer, space)))
                            for name in bundle.names
                        }
                        cos_src = cos_all[src]
                        others = [v for n, v in cos_all.items() if n != src]
                        rank = 1 + sum(1 for v in others if v > cos_src)
                        rows.append(
                            {
                                "layer": layer,
                                "cos_src_abs": cos_src,
                                "cos_src_signed": _cos(key, bundle.vec(src, layer, space)),
                                "null_p95": float(np.percentile(others, 95)),
                                "null_p50": float(np.percentile(others, 50)),
                                "best_nonsource": max(others),
                                "selectivity_margin": cos_src - max(others),
                                "rank_in_bank": rank,
                                "n_bank": len(cos_all),
                            }
                        )
                    if not rows:
                        continue
                    hits = [
                        r["layer"]
                        for r in rows
                        if r["layer"] in band
                        and r["cos_src_abs"] > r["null_p95"]
                        and r["rank_in_bank"] <= 3
                    ]
                    # longest contiguous run within the band
                    run, best_run = 0, 0
                    for li in band:
                        run = run + 1 if li in hits else 0
                        best_run = max(best_run, run)
                    per_space[space] = {
                        "layers": rows,
                        "band_hits": hits,
                        "longest_contiguous_band_run": best_run,
                        "key_present": best_run >= 3,
                        "best_layer": max(rows, key=lambda r: r["cos_src_abs"])["layer"],
                    }
                if per_space:
                    src_rec["stacks"][stack] = per_space
            # per-module q/k/v resolved keys at band layers (router diagnostics)
            mod_band = {}
            for module in ("q_proj", "k_proj", "v_proj"):
                vals = []
                for layer in band:
                    arrs = store.vectors(cell)
                    m = arrs.get(f"L{layer}__{module}__right_top2")
                    if m is None:
                        continue
                    vals.append(
                        abs(_cos(m[:, 0].astype(np.float64), bundle.vec(src, layer, "attn")))
                    )
                if vals:
                    mod_band[module] = float(np.mean(vals))
            src_rec["per_module_band_mean_cos"] = mod_band
            rec["per_source"].append(src_rec)
        if not sources:  # i521 EM control: should match NOTHING above null
            rows = []
            for layer in band:
                key = store.key_top1(cell, layer, "attn_key")
                if key is None:
                    continue
                cos_all = {n: abs(_cos(key, bundle.vec(n, layer, "attn"))) for n in bundle.names}
                best = max(cos_all, key=cos_all.get)
                vals = list(cos_all.values())
                rows.append(
                    {
                        "layer": layer,
                        "best_context": best,
                        "best_cos": cos_all[best],
                        "bank_p95": float(np.percentile(vals, 95)),
                        "bank_p50": float(np.percentile(vals, 50)),
                    }
                )
            rec["no_source_bank_read"] = rows
        results.append(rec)

    # Shuffled-pairing null (within line): key of cell i vs source of cell j≠i.
    shuffled = {}
    by_line: dict[str, list[dict]] = {}
    for cell in store.cells:
        by_line.setdefault(cell["cell"]["line"], []).append(cell)
    for line, cells in by_line.items():
        vals = []
        for ci in cells:
            for cj in cells:
                if cj is ci or not cj["cell"]["source_personas"]:
                    continue
                if set(ci["cell"]["source_personas"]) & set(cj["cell"]["source_personas"]):
                    continue
                for layer in band:
                    key = store.key_top1(ci, layer, "attn_key")
                    if key is None:
                        continue
                    try:
                        src_j = bundle.resolve(cj["cell"]["source_personas"][0])
                    except KeyError:
                        continue
                    vals.append(abs(_cos(key, bundle.vec(src_j, layer, "attn"))))
        if vals:
            shuffled[line] = {
                "n": len(vals),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "mean": float(np.mean(vals)),
            }

    aux = _aux_bank_reads(store, data_root)
    i541 = _i541_key_match(store, bundle, data_root)
    payload = {
        "meta": result_metadata(PROJECT_ROOT, extra={"analysis": "key_match"}),
        "layer_band": list(band),
        "cells": results,
        "shuffled_pairing_null": shuffled,
        "aux_banks": aux,
        "i541_own_bank": i541,
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info("key_match.json written (%d cells)", len(results))


def _aux_bank_reads(store: CellStore, data_root: Path) -> dict:
    """Auxiliary nulls: 111-persona L20 bank + 24-persona all-layer bank (raw space)."""
    import torch

    out: dict = {}
    p111, n111 = data_root / BANK111_PT, data_root / BANK111_NAMES
    if p111.exists() and n111.exists():
        bank = torch.load(p111, weights_only=True).float().numpy().astype(np.float64)
        names = json.loads(n111.read_text())["persona_names"]
        assert bank.shape == (len(names), HIDDEN_SIZE), bank.shape
        rows = []
        for cell in store.cells:
            c = cell["cell"]
            if not c["source_personas"]:
                continue
            src = c["source_personas"][0]
            if src not in names:
                continue
            key = store.key_top1(cell, EXTRACT_LAYER_DIAL, "attn_key")
            if key is None:
                continue
            cos_all = [abs(_cos(key, bank[i])) for i in range(len(names))]
            cos_src = cos_all[names.index(src)]
            others = [v for i, v in enumerate(cos_all) if i != names.index(src)]
            rows.append(
                {
                    "line": c["line"],
                    "cell_id": c["cell_id"],
                    "source": src,
                    "layer": EXTRACT_LAYER_DIAL,
                    "cos_src_abs": cos_src,
                    "null_p95": float(np.percentile(others, 95)),
                    "rank_in_bank": 1 + sum(1 for v in others if v > cos_src),
                    "n_bank": len(names),
                }
            )
        out["bank111_L20_raw_space"] = rows
    else:
        out["bank111_L20_raw_space"] = "N/A — bank files not present in checkout"

    p24 = data_root / BANK24_PT
    if p24.exists():
        bank24 = torch.load(p24, weights_only=True)
        rows = []
        for cell in store.cells:
            c = cell["cell"]
            if not c["source_personas"]:
                continue
            src = c["source_personas"][0]
            per_layer = []
            for layer, personas in bank24.items():
                if src not in personas:
                    continue
                key = store.key_top1(cell, int(layer), "attn_key")
                if key is None:
                    continue
                cos_all = {
                    p: abs(_cos(key, v.float().numpy().astype(np.float64)))
                    for p, v in personas.items()
                }
                others = [v for p, v in cos_all.items() if p != src]
                per_layer.append(
                    {
                        "layer": int(layer),
                        "cos_src_abs": cos_all[src],
                        "null_p95": float(np.percentile(others, 95)),
                        "rank_in_bank": 1 + sum(1 for v in others if v > cos_all[src]),
                    }
                )
            if per_layer:
                rows.append({"line": c["line"], "cell_id": c["cell_id"], "layers": per_layer})
        out["bank24_all_layer_raw_space"] = rows
    else:
        out["bank24_all_layer_raw_space"] = "N/A — bank file not present in checkout"
    return out


def _i541_key_match(store: CellStore, bundle: ContextBundle, data_root: Path) -> dict | str:
    """#541 secondary line vs its OWN stored activation banks (plan §5).

    Persona order comes from the git copy of ``geometry_matrices.json``
    (#541 geometry-plus-prior predictor output, key ``personas``); the
    fp16 activation banks come from the HF data repo.
    """
    cells = [c for c in store.cells if c["cell"]["line"] == "i541"]
    if not cells:
        return "N/A — no i541 cells in Phase A outputs"
    try:
        from huggingface_hub import hf_hub_download

        geom_path = (
            data_root
            / "eval_results/issue_541/geometry-plus-prior-joint-predictor/geometry_matrices.json"
        )
        geom = json.loads(geom_path.read_text())
        persona_names = geom.get("persona_names") or geom.get("personas")
        assert persona_names and len(persona_names) == 24, persona_names
        banks = {}
        for layer in I541_LAYERS:
            p = hf_hub_download(
                HF_DATA_REPO,
                f"{I541_ACT_PREFIX}/activations_L{layer}_fp16.npy",
                repo_type="dataset",
            )
            arr = np.load(p).astype(np.float64)
            assert arr.shape == (24, 40, HIDDEN_SIZE), (layer, arr.shape)
            banks[layer] = arr.mean(axis=1)  # (24, 3584) centroids over 40 probes
    except (OSError, KeyError, AssertionError, json.JSONDecodeError) as exc:
        return f"N/A — artifact failed fitness at load: {exc!r}"

    rows = []
    for cell in cells:
        src = cell["cell"]["source_personas"][0]
        if src not in persona_names:
            rows.append(
                {
                    "cell_id": cell["cell"]["cell_id"],
                    "status": f"N/A — source {src!r} not in the 24-persona bank",
                }
            )
            continue
        si = persona_names.index(src)
        per_layer = []
        for layer in I541_LAYERS:
            key = store.key_top1(cell, layer, "attn_key")
            if key is None:
                continue
            for space, transform in (
                ("raw", lambda v: v),
                ("gamma_raw", lambda v, _l=layer: bundle.gamma_in[_l] * v),
            ):
                cos_all = [abs(_cos(key, transform(banks[layer][i]))) for i in range(24)]
                others = [v for i, v in enumerate(cos_all) if i != si]
                per_layer.append(
                    {
                        "layer": layer,
                        "space": space,
                        "cos_src_abs": cos_all[si],
                        "null_p95": float(np.percentile(others, 95)),
                        "rank_in_bank": 1 + sum(1 for v in others if v > cos_all[si]),
                    }
                )
        rows.append({"cell_id": cell["cell"]["cell_id"], "source": src, "layers": per_layer})
    return {"persona_names": persona_names, "cells": rows}


# ── analysis 2: write match ─────────────────────────────────────────────────


def _write_pool(store: CellStore, cell: dict, l_max: int) -> np.ndarray | None:
    """Top principal direction of {σ₁(l)·w₁(l)} over layers ≤ l_max (§4 C3)."""
    cols = []
    for layer in store.layers(cell):
        if layer > l_max:
            continue
        wt = store.write_top1(cell, layer)
        if wt is None:
            continue
        w1, s1 = wt
        cols.append(s1 * w1)
    if not cols:
        return None
    M = np.stack(cols, axis=1)  # (3584, n_layers)
    U, _S, _Vt = np.linalg.svd(M, full_matrices=False)
    pool = U[:, 0]
    # orient toward the σ-weighted mean column
    if float(M.mean(axis=1) @ pool) < 0:
        pool = -pool
    return pool


def run_write_match(store: CellStore, data_root: Path, out: Path) -> None:
    """§4 C3 — pooled write vs measured shifts; EM control vs #551 direction."""
    results = []
    for cell in store.cells:
        c = cell["cell"]
        if is_dial(cell) and "checkpoint-intermediate" not in c["tags"]:
            slug = c["cell_id"]
            issue = int(c["line"].removeprefix("dial"))
            pool = _write_pool(store, cell, EXTRACT_LAYER_DIAL)
            if pool is None:
                continue
            sj = load_dial_shift_json(data_root, issue, slug)
            panel = sj["eval_panel"]
            shifts = load_dial_shift_pt(issue, slug)
            shift_svd = svd_summary(shifts.T.astype(np.float32))  # (3584, 19)
            cos_rows = {p: _cos(pool, shifts[i]) for i, p in enumerate(panel)}
            per_layer = {}
            for src in c["source_personas"]:
                prof = []
                for layer in store.layers(cell):
                    wt = store.write_top1(cell, layer)
                    if wt is None:
                        continue
                    prof.append(
                        {"layer": layer, "cos_abs": abs(_cos(wt[0], shifts[panel.index(src)]))}
                    )
                per_layer[src] = prof
            recs = []
            for src in c["source_personas"]:
                others = [abs(v) for p, v in cos_rows.items() if p != src]
                recs.append(
                    {
                        "source": src,
                        "cos_abs": abs(cos_rows[src]),
                        "cos_signed": cos_rows[src],
                        "null_p95": float(np.percentile(others, 95)),
                        "null_p5": float(np.percentile(others, 5)),
                        "null_spread_p5_p95": float(
                            np.percentile(others, 95) - np.percentile(others, 5)
                        ),
                        "rank_in_panel": 1 + sum(1 for v in others if v > abs(cos_rows[src])),
                    }
                )
            results.append(
                {
                    "line": c["line"],
                    "cell_id": slug,
                    "tags": c["tags"],
                    "extract_layer": EXTRACT_LAYER_DIAL,
                    "per_source": recs,
                    "all_panel_cos": cos_rows,
                    "shift_matrix_top1_frac": shift_svd["s_top1_frac"],
                    "per_layer_profile": per_layer,
                    "dose": cell_dose(data_root, cell),
                }
            )
        elif c["line"] in ("i519", "i521"):
            arm = "marker" if c["line"] == "i519" else "em"
            seed = c["seed"]
            pool = _write_pool(store, cell, EXTRACT_LAYER_519)
            if pool is None:
                continue
            variants = {}
            for variant in ("same", "on_policy", "base"):
                try:
                    shifts = load_i551_shifts(variant, arm, seed)
                except Exception as exc:
                    variants[variant] = f"N/A — artifact failed fitness at load: {exc!r}"
                    continue
                M, order = assemble_M(shifts)
                summ = svd_summary(M)
                u1 = summ["U1"].astype(np.float64)
                per_ctx = {
                    p: _cos(pool, shifts[p]["delta_v"].detach().float().numpy()) for p in order
                }
                variants[variant] = {
                    "persona_order": order,
                    "cos_pool_vs_U1_shared_direction": _cos(pool, u1),
                    "shared_direction_top1_frac": summ["s_top1_frac"],
                    "per_context_cos": per_ctx,
                    "source_cos": per_ctx.get("medical_doctor"),
                }
            # #552 mean-resp variant (git, marker arm only)
            if c["line"] == "i519":
                p552 = data_root / I552_SHIFT_DIR / f"same_marker_seed{seed}.pt"
                if p552.exists():
                    import torch

                    obj = torch.load(p552, weights_only=True)
                    shifts = obj["shifts"]
                    per_ctx = {
                        p: _cos(pool, e["delta_v_mean_resp"].float().numpy())
                        for p, e in shifts.items()
                    }
                    variants["mean_resp_552"] = {
                        "per_context_cos": per_ctx,
                        "source_cos": per_ctx.get("medical_doctor"),
                    }
                else:
                    variants["mean_resp_552"] = "N/A — #552 git tensor not in checkout"
            results.append(
                {
                    "line": c["line"],
                    "cell_id": c["cell_id"],
                    "tags": c["tags"],
                    "extract_layer": EXTRACT_LAYER_519,
                    "variants": variants,
                }
            )
    # EM-control cross-seed framing: matched-seed primary vs mean-over-seeds
    payload = {
        "meta": result_metadata(PROJECT_ROOT, extra={"analysis": "write_match"}),
        "em_control_note": (
            "matched-seed comparator is PRIMARY; the #551 shared direction's own "
            "cross-seed stability is |cos| 0.65-0.90 (mean 0.78) — a weights-side "
            "match cannot be expected to exceed that ceiling (plan §13)"
        ),
        "cells": results,
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info("write_match.json written (%d cells)", len(results))


# ── analysis 3: rotation ────────────────────────────────────────────────────


def _band_mean(vals: list[float]) -> float:
    assert vals, "no band layers available"
    return float(np.mean(vals))


def _rotation_reads(
    store: CellStore,
    bundle: ContextBundle,
    cell: dict,
    src_name: str,
    negatives: list[str],
    rng: np.random.Generator,
) -> dict | None:
    """Band-mean Δ|cos| + component cosines + placebo + geometry for one source."""
    band = [li for li in KEY_LAYER_BAND if li < bundle.n_layers]
    src = bundle.resolve(src_name)
    negs = [bundle.resolve(n) for n in negatives]
    non_trained = [n for n in bundle.names if n != src and n not in negs]
    placebo_pool = rng.choice(non_trained, size=min(len(negs), len(non_trained)), replace=False)

    rows = {
        k: []
        for k in (
            "cos_contrast",
            "cos_raw",
            "cos_placebo",
            "cos_orth",
            "raw_vs_contrast_cos",
            "resid_norm",
            "subspace_contrast",
            "subspace_raw",
        )
    }
    src_neg_cos = []
    used_layers = []
    for layer in band:
        key = store.key_top1(cell, layer, "attn_key")
        if key is None:
            continue
        v_src = _unit(bundle.vec(src, layer, "attn"))
        v_negs = [_unit(bundle.vec(n, layer, "attn")) for n in negs]
        contrast = v_src - np.mean(v_negs, axis=0)
        resid_norm = float(np.linalg.norm(contrast))
        u_contrast = _unit(contrast)
        u_raw = v_src
        v_plc = [_unit(bundle.vec(n, layer, "attn")) for n in placebo_pool]
        u_placebo = _unit(v_src - np.mean(v_plc, axis=0))
        # orthogonalized contrast component (⊥ raw)
        orth = u_contrast - (u_contrast @ u_raw) * u_raw
        orth_ok = float(np.linalg.norm(orth)) > 1e-9
        rows["cos_contrast"].append(abs(_cos(key, u_contrast)))
        rows["cos_raw"].append(abs(_cos(key, u_raw)))
        rows["cos_placebo"].append(abs(_cos(key, u_placebo)))
        rows["cos_orth"].append(abs(_cos(key, orth)) if orth_ok else 0.0)
        rows["raw_vs_contrast_cos"].append(float(u_raw @ u_contrast))
        rows["resid_norm"].append(resid_norm)
        src_neg_cos.extend([float(v_src @ v) for v in v_negs])
        K = store.key_top2(cell, layer, "attn_key")
        if K is not None:
            rows["subspace_contrast"].append(float(np.linalg.norm(K.T @ u_contrast)))
            rows["subspace_raw"].append(float(np.linalg.norm(K.T @ u_raw)))
        used_layers.append(layer)
    if not used_layers:
        return None
    return {
        "source": src,
        "negatives": negs,
        "placebo_pool": list(map(str, placebo_pool)),
        "layers_used": used_layers,
        "cos_contrast": _band_mean(rows["cos_contrast"]),
        "cos_raw": _band_mean(rows["cos_raw"]),
        "delta_cos": _band_mean(rows["cos_contrast"]) - _band_mean(rows["cos_raw"]),
        "cos_placebo": _band_mean(rows["cos_placebo"]),
        "delta_cos_placebo": _band_mean(rows["cos_placebo"]) - _band_mean(rows["cos_raw"]),
        "cos_orthogonalized": _band_mean(rows["cos_orth"]),
        "geometry": {
            "raw_vs_contrast_cos": _band_mean(rows["raw_vs_contrast_cos"]),
            "contrast_resid_norm": _band_mean(rows["resid_norm"]),
            "source_negative_cos_mean": float(np.mean(src_neg_cos)),
            "source_negative_cos_min": float(np.min(src_neg_cos)),
            "source_negative_cos_max": float(np.max(src_neg_cos)),
        },
        "subspace_top2": (
            {
                "contrast_proj": _band_mean(rows["subspace_contrast"]),
                "raw_proj": _band_mean(rows["subspace_raw"]),
                "delta_proj": _band_mean(rows["subspace_contrast"])
                - _band_mean(rows["subspace_raw"]),
            }
            if rows["subspace_contrast"]
            else None
        ),
    }


def run_rotation(store: CellStore, bundle: ContextBundle, data_root: Path, out: Path) -> None:
    """§4 C4 — Δcos vs dose (dial) + the #474 epoch ladder paired contrast."""
    rng = np.random.default_rng(RNG_SEED)
    primary, secondary_joint, contaminated = [], [], []
    for cell in store.cells:
        c = cell["cell"]
        if not is_dial(cell) or "checkpoint-intermediate" in c["tags"]:
            continue
        doses = cell_dose(data_root, cell)
        for src in c["source_personas"]:
            read = _rotation_reads(store, bundle, cell, src, list(c["negative_personas"]), rng)
            if read is None:
                continue
            row = {
                "line": c["line"],
                "cell_id": c["cell_id"],
                "arm": c["arm"],
                "seed": c["seed"],
                "pair": c["cell_id"].rsplit("__", 2)[0],
                "dose_delta_logp_marker": doses[src],
                "tags": c["tags"],
                **read,
            }
            # Plan §4 C4(a): ALL 9 panel-contaminated #527 pair-2 cells (single
            # AND joint) are excluded from the registered reads — they enter
            # only the all-54 sensitivity. The joint SECONDARY is the 15 CLEAN
            # joint cells (30 per-source rows).
            if "panel-contaminated" in c["tags"]:
                contaminated.append(row)
            elif is_clean_single_dial(cell):
                primary.append(row)
            else:
                assert c["arm"] == "joint", (c["cell_id"], c["arm"], c["tags"])
                secondary_joint.append(row)

    def _spearman_block(rows: list[dict]) -> dict:
        if len(rows) < 3:
            return {"n": len(rows), "status": "N/A — too few cells for a trend read"}
        x = [r["dose_delta_logp_marker"] for r in rows]
        y = [r["delta_cos"] for r in rows]
        rho = spearman_rho(x, y)
        clusters: dict[tuple, list[int]] = {}
        for i, r in enumerate(rows):
            clusters.setdefault((r["pair"], r["seed"]), []).append(i)
        keys = list(clusters)
        boots = []
        for _ in range(N_BOOT):
            picked = rng.choice(len(keys), size=len(keys), replace=True)
            idx = [i for k in picked for i in clusters[keys[k]]]
            if len({x[i] for i in idx}) < 3:
                continue
            boots.append(spearman_rho([x[i] for i in idx], [y[i] for i in idx]))
        per_cluster = {
            "__".join(map(str, k)): spearman_rho([x[i] for i in v], [y[i] for i in v])
            for k, v in clusters.items()
            if len(v) >= 3
        }
        return {
            "n": len(rows),
            "spearman_rho": rho,
            "cluster_bootstrap": {
                "n_clusters": len(keys),
                "n_boot_effective": len(boots),
                "ci_2p5": float(np.percentile(boots, 2.5)) if boots else None,
                "ci_97p5": float(np.percentile(boots, 97.5)) if boots else None,
                "note": "6-cluster CI is fragile — read with per-cluster signs (plan §4 C4(f))",
            },
            "per_cluster_spearman": per_cluster,
            "per_cluster_sign_positive": sum(1 for v in per_cluster.values() if v > 0),
        }

    i474 = _i474_ladder(store, bundle, data_root, rng)
    payload = {
        "meta": result_metadata(PROJECT_ROOT, extra={"analysis": "rotation"}),
        "layer_band": [li for li in KEY_LAYER_BAND if li < bundle.n_layers],
        "sign_convention": (
            "Δcos uses sign-folded |cos| band means (singular-vector sign ambiguity); "
            "signed per-layer values are recoverable from key_match.json cos_src_signed"
        ),
        "primary_30_clean_single_source": {
            "cells": primary,
            "trend": _spearman_block(primary),
        },
        "secondary_joint_per_source": {
            "cells": secondary_joint,
            "trend": _spearman_block(secondary_joint),
        },
        "sensitivity_all_cells": _spearman_block(primary + secondary_joint + contaminated),
        "excluded_panel_contaminated": contaminated,
        "i474_epoch_ladder": i474,
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info(
        "rotation.json written (primary n=%d, joint n=%d)", len(primary), len(secondary_joint)
    )


def _i474_ladder(
    store: CellStore, bundle: ContextBundle, data_root: Path, rng: np.random.Generator
) -> dict | str:
    """§4 C4(e) — both arms on the matched other-15 contrast; paired OLS slopes."""
    cells = [c for c in store.cells if c["cell"]["line"] == "i474"]
    if not cells:
        return "N/A — no i474 cells in Phase A outputs"
    # exposure-parity covariate: G diagonals per (arm, epoch)
    exposure: dict[str, dict] = {}
    for arm in ("pos", "loc"):
        for ep in (1, 2, 3, 5):
            p = data_root / I474_CROSS_EVAL / f"{arm}_ep{ep}" / "G_logprob_matrix.json"
            if p.exists():
                exposure[f"{arm}_ep{ep}"] = json.loads(p.read_text()).get("diagonals", {})
    reads = []
    for cell in cells:
        c = cell["cell"]
        src = c["source_personas"][0]
        negs = [
            s
            for s in bundle.names
            if s
            in set(
                [
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "A5",
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "C1",
                    "D1",
                    "D2",
                    "D3",
                    "D4",
                    "D5",
                ]
            )
            and s != src
        ]
        if len(negs) != 15 or src not in bundle.names:
            return f"N/A — Phase B bundle lacks the 16 transformation contexts (src={src!r})"
        read = _rotation_reads(store, bundle, cell, src, negs, rng)
        if read is None:
            continue
        reads.append(
            {
                "arm": c["arm"],
                "source": src,
                "epoch": c["epoch"],
                "exposure_G_diag": exposure.get(f"{c['arm']}_ep{c['epoch']}", {}).get(src),
                **read,
            }
        )

    # pinned statistic: per-source OLS slope over epochs, paired loc − pos
    def _slope(points: list[tuple[int, float]]) -> float | None:
        if len(points) < 2:
            return None
        x = np.array([p[0] for p in points], dtype=np.float64)
        y = np.array([p[1] for p in points], dtype=np.float64)
        xc = x - x.mean()
        denom = float((xc**2).sum())
        if denom == 0:
            return None
        # Exact normal-equation OLS slope (NOT np.cov/var — mixing the sample
        # covariance (N-1) with the population variance (N) inflates the
        # slope by N/(N-1), i.e. 4/3 at 4 epochs).
        return float((xc * (y - y.mean())).sum() / denom)

    slopes: dict[str, dict[str, float | None]] = {}
    for r in reads:
        slopes.setdefault(r["source"], {})
    for src in slopes:
        for arm in ("pos", "loc"):
            pts = [
                (r["epoch"], r["delta_cos"])
                for r in reads
                if r["source"] == src and r["arm"] == arm
            ]
            slopes[src][arm] = _slope(sorted(pts))
    paired = {
        src: v["loc"] - v["pos"]
        for src, v in slopes.items()
        if v.get("loc") is not None and v.get("pos") is not None
    }
    diffs = list(paired.values())
    boot = None
    if len(diffs) >= 3:
        # Plan §4 C4(e) pins the MEAN paired slope difference; the bootstrap
        # resamples the MEAN (the ported bootstrap_ci resamples the median —
        # kept byte-identical for its #519 callers, not used here). The
        # median is reported as an auxiliary read only.
        arr = np.asarray(diffs, dtype=np.float64)
        boot_rng = np.random.default_rng(RNG_SEED)
        boots = [float(arr[boot_rng.integers(0, arr.size, arr.size)].mean()) for _ in range(N_BOOT)]
        boot = {
            "mean": float(arr.mean()),
            "ci_lo_mean": float(np.percentile(boots, 2.5)),
            "ci_hi_mean": float(np.percentile(boots, 97.5)),
            "ci_statistic": "mean",
            "n_boot": N_BOOT,
            "median_auxiliary": float(np.median(arr)),
        }
    return {
        "reads": reads,
        "per_source_ols_slopes": slopes,
        "paired_loc_minus_pos": paired,
        "aggregate": boot,
        "sign_count_positive": sum(1 for v in diffs if v > 0),
        "n_sources_paired": len(diffs),
    }


# ── analysis 4: functional constancy (Wang et al., adapted) ────────────────


def run_constancy(store: CellStore, bundle: ContextBundle, out: Path) -> None:
    """§4 C5 — pairwise |cos| of rank-8 Δout across bank contexts, per layer."""
    band = [li for li in KEY_LAYER_BAND if li < bundle.n_layers]
    results = []
    for cell in store.cells:
        c = cell["cell"]
        if c["line"] == "i541":
            continue
        arrs = store.vectors(cell)
        per_layer = []
        for layer in band:
            V8 = arrs.get(f"L{layer}__attn_key__V8")
            S = arrs.get(f"L{layer}__attn_key__S")
            if V8 is None or S is None:
                continue
            k = V8.shape[1]
            X = np.stack(
                [
                    S[:k] * (V8.astype(np.float64).T @ _unit(bundle.vec(n, layer, "attn")))
                    for n in bundle.names
                ]
            )  # (n_ctx, k) — Δout cosines need only S·V8ᵀ·v (U orthonormal)
            norms = np.linalg.norm(X, axis=1)
            ok = norms > 1e-12
            Xn = X[ok] / norms[ok][:, None]
            G = Xn @ Xn.T
            off = np.abs(G[np.triu_indices(Xn.shape[0], k=1)])
            key = V8[:, 0].astype(np.float64)
            gates = np.array([_cos(key, bundle.vec(n, layer, "attn")) for n in bundle.names])
            per_layer.append(
                {
                    "layer": layer,
                    "pairwise_abs_cos_mean": float(off.mean()),
                    "pairwise_abs_cos_p5": float(np.percentile(off, 5)),
                    "pairwise_abs_cos_p50": float(np.percentile(off, 50)),
                    "pairwise_abs_cos_p95": float(np.percentile(off, 95)),
                    "gate_tie_in_spearman_norm_vs_keycos": spearman_rho(
                        norms.tolist(), np.abs(gates).tolist()
                    ),
                    "n_contexts": int(Xn.shape[0]),
                }
            )
        if per_layer:
            results.append(
                {
                    "line": c["line"],
                    "cell_id": c["cell_id"],
                    "tags": c["tags"],
                    "per_layer": per_layer,
                    "band_mean_pairwise_abs_cos": float(
                        np.mean([r["pairwise_abs_cos_mean"] for r in per_layer])
                    ),
                }
            )
    payload = {
        "meta": result_metadata(PROJECT_ROOT, extra={"analysis": "functional_constancy"}),
        "adaptation_note": (
            "Wang et al. (arXiv 2507.08218) read pairwise |cos| of LoRA output-difference "
            "vectors over per-token activations; this adaptation projects stored per-context "
            "CENTROIDS through the rank-8 truncated attn-key stack (truncation energy "
            "fraction recorded per layer in the Phase A spectra). SCOPE: the projection "
            "covers the attn_key stack ONLY — the mlp_key stack of all-linear lines is "
            "not included in this read"
        ),
        "cells": results,
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info("functional_constancy.json written (%d cells)", len(results))


# ── analysis 5: selectivity / seed stability / joint subspace ───────────────


def run_selectivity(  # noqa: C901 — seed-stability + joint-subspace reads share the band helpers
    store: CellStore, bundle: ContextBundle, out: Path
) -> None:
    """§4 C6+C7 — seed-stability matrices + joint-arm top-2 subspace reads."""
    band = [li for li in KEY_LAYER_BAND if li < bundle.n_layers]

    def _band_key_cos(ca: dict, cb: dict, stack: str) -> float | None:
        vals = []
        for layer in band:
            ka, kb = store.key_top1(ca, layer, stack), store.key_top1(cb, layer, stack)
            if ka is None or kb is None:
                continue
            vals.append(abs(_cos(ka, kb)))
        return float(np.mean(vals)) if vals else None

    def _band_write_cos(ca: dict, cb: dict) -> float | None:
        """Plan §4 C7 'keys (and writes)': cross-seed |cos| of the top write vectors."""
        vals = []
        for layer in band:
            wa, wb = store.write_top1(ca, layer), store.write_top1(cb, layer)
            if wa is None or wb is None:
                continue
            vals.append(abs(_cos(wa[0], wb[0])))
        return float(np.mean(vals)) if vals else None

    groups: dict[tuple, list[dict]] = {}
    for cell in store.cells:
        c = cell["cell"]
        if "checkpoint-intermediate" in c["tags"]:
            continue
        # seed_group_key handles BOTH separator forms (dial `__seed42` AND the
        # single-underscore `marker_seed42` / `em_turner_seed42` / `<arm>_seed42`
        # of #519/#521/#541) — rsplit("__seed") silently left the latter three
        # lines as singleton groups, emptying the registered C7 read.
        gkey = (
            c["line"],
            seed_group_key(c["cell_id"]) if c["seed"] is not None else c["cell_id"],
        )
        groups.setdefault(gkey, []).append(cell)
    seed_stability = []
    for (line, gname), cells in groups.items():
        if len(cells) < 2:
            continue
        pairs = []
        for i, ca in enumerate(cells):
            for cb in cells[i + 1 :]:
                kc = _band_key_cos(ca, cb, "attn_key")
                wc = _band_write_cos(ca, cb)
                if kc is not None or wc is not None:
                    pairs.append(
                        {
                            "seeds": [ca["cell"]["seed"], cb["cell"]["seed"]],
                            "key_abs_cos_band_mean": kc,
                            "write_abs_cos_band_mean": wc,
                        }
                    )
        if pairs:
            seed_stability.append({"line": line, "group": gname, "pairs": pairs})

    joint = []
    for cell in store.cells:
        c = cell["cell"]
        if not (
            is_dial(cell) and c["arm"] == "joint" and "checkpoint-intermediate" not in c["tags"]
        ):
            continue
        rows = []
        for layer in band:
            K = store.key_top2(cell, layer, "attn_key")
            if K is None:
                continue
            try:
                us = [
                    _unit(
                        _unit(bundle.vec(bundle.resolve(s), layer, "attn"))
                        - np.mean(
                            [
                                _unit(bundle.vec(bundle.resolve(n), layer, "attn"))
                                for n in c["negative_personas"]
                            ],
                            axis=0,
                        )
                    )
                    for s in c["source_personas"]
                ]
            except KeyError:
                continue
            span = np.stack(us, axis=1)  # (3584, 2)
            Q, _ = np.linalg.qr(span)
            # fraction of span{u_A, u_B} energy captured by the top-2 key subspace
            proj = K.T @ Q  # (2, 2)
            frac = float(np.linalg.norm(proj) ** 2 / Q.shape[1])
            rows.append({"layer": layer, "span_capture_frac": frac})
        if rows:
            joint.append(
                {
                    "line": c["line"],
                    "cell_id": c["cell_id"],
                    "band_mean_span_capture_frac": float(
                        np.mean([r["span_capture_frac"] for r in rows])
                    ),
                    "per_layer": rows,
                }
            )

    payload = {
        "meta": result_metadata(PROJECT_ROOT, extra={"analysis": "selectivity"}),
        "seed_stability": seed_stability,
        "joint_arm_top2_subspace": joint,
    }
    out.write_text(json.dumps(payload, indent=1))
    logger.info(
        "selectivity.json written (%d seed groups, %d joint cells)",
        len(seed_stability),
        len(joint),
    )


# ── entrypoint ──────────────────────────────────────────────────────────────

ANALYSES = ("key", "write", "rotation", "constancy", "selectivity")


def _default_data_root() -> Path:
    """Resolve the data root for aux/registered inputs.

    Several registered inputs (the 24-persona all-layer bank, the #552
    mean-resp tensors, the #541 geometry manifest) are untracked files
    that live only in the canonical main checkout — a sparse worktree
    checkout will not have them. Prefer PROJECT_ROOT when the 24-bank
    resolves there; otherwise fall back to the canonical main repo root
    (task_workflow.repo_root — VM-only, like every Phase C run; a logged
    degradation, never a silent one).
    """
    if (PROJECT_ROOT / BANK24_PT).exists():
        return PROJECT_ROOT
    try:
        from explore_persona_space.task_workflow import repo_root

        root = repo_root()
        if (root / BANK24_PT).exists():
            logger.info("data-root: aux inputs absent here; using canonical main root %s", root)
            return root
    except RuntimeError as exc:
        logger.warning("data-root: canonical-root resolution failed (%s)", exc)
    logger.warning(
        "data-root: 24-persona bank not found anywhere — registered aux reads will record N/A"
    )
    return PROJECT_ROOT


def main() -> None:
    """Phase C entrypoint — same code path for smoke and the full run."""
    parser = argparse.ArgumentParser(
        description="Task 604 Phase C: key/write/rotation/constancy/selectivity reads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--context-dir", default=str(OUT_DIR_DEFAULT / "context_vectors"))
    parser.add_argument(
        "--data-root",
        default="",
        help=(
            "checkout holding the aux/registered inputs (24-persona bank, #552 tensors, "
            "#541 geometry manifest — some are untracked, main-checkout-only). Default: "
            "this checkout if the 24-bank resolves here, else the canonical main repo root."
        ),
    )
    parser.add_argument("--analyses", default="all", help="all | comma of " + ",".join(ANALYSES))
    parser.add_argument(
        "--expect-probes",
        type=int,
        default=50,
        help=(
            "required n_probes in the Phase B bundle meta (stale-cache guard; production = 50; "
            "a smoke read over a tiny bundle passes its own value explicitly)"
        ),
    )
    parser.add_argument(
        "--allow-missing-contexts",
        action="store_true",
        help=(
            "smoke only: tolerate a partial Phase B bundle (registered 'N/A' rows instead of "
            "failing loud on contexts the cell inventory requires)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print("[phase=c_load]", flush=True)
    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root) if args.data_root else _default_data_root()
    store = CellStore(out_dir)
    assert store.cells, "no Phase A outputs found — run issue604_adapter_svd.py first"
    # Contexts the loaded cell inventory actually consumes (sources +
    # realized negative panels); the bundle must cover them in production.
    required: set[str] = set()
    for cell in store.cells:
        c = cell["cell"]
        if c["line"] == "i541":
            continue  # own-bank read — never resolved against the bundle
        required.update(c["source_personas"])
        required.update(c["negative_personas"])
    bundle = ContextBundle(
        Path(args.context_dir),
        expected_n_probes=args.expect_probes,
        required_contexts=() if args.allow_missing_contexts else tuple(sorted(required)),
    )
    assert bundle.hidden == HIDDEN_SIZE, (
        f"Phase B bundle hidden={bundle.hidden} != {HIDDEN_SIZE} — was Phase B run on a "
        "substitute model? Phase C comparisons need the production base model's bundle."
    )

    todo = ANALYSES if args.analyses == "all" else tuple(args.analyses.split(","))
    for a in todo:
        assert a in ANALYSES, f"unknown analysis {a!r}"
    if "key" in todo:
        print("[phase=c_key_match]", flush=True)
        run_key_match(store, bundle, data_root, out_dir / "key_match.json")
    if "write" in todo:
        print("[phase=c_write_match]", flush=True)
        run_write_match(store, data_root, out_dir / "write_match.json")
    if "rotation" in todo:
        print("[phase=c_rotation]", flush=True)
        run_rotation(store, bundle, data_root, out_dir / "rotation.json")
    if "constancy" in todo:
        print("[phase=c_constancy]", flush=True)
        run_constancy(store, bundle, out_dir / "functional_constancy.json")
    if "selectivity" in todo:
        print("[phase=c_selectivity]", flush=True)
        run_selectivity(store, bundle, out_dir / "selectivity.json")
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
