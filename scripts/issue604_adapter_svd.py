# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, σ, γ, ※, —) in scientific docstrings + labels.
"""Task #604 Phase A — exact SVD of stored LoRA ΔW per adapter cell (VM CPU).

For each adapter cell in the §5 inventory: download config + safetensors
from the Hub (revision pinned at run start), assert rsLoRA + gauge
validity, compose and SVD ΔW = s·B·A per (layer, module) AND per stacked
object (attn-key row-stack [ΔW_q;ΔW_k;ΔW_v], MLP-key row-stack
[ΔW_gate;ΔW_up], residual-write column-concat [ΔW_o|ΔW_down]) without ever
materializing d_out×d_in.

Persists per cell, the moment the cell completes (checkpoint-per-cell):

- ``eval_results/issue_604/spectra/<line>/<cell>.json`` — per-(layer,
  module/stack) full σ spectrum, top-1 energy, effective rank, ‖ΔW‖_F.
- ``eval_results/issue_604/vectors/<line>/<cell>.npz`` — 3584-dim singular
  vectors ONLY (comparison-validity map, dimension-asserted): key-stack
  top-8 right basis (V8 fp16) + full σ, write-stack top-8 left basis
  (U8 fp16) + full σ, per-module top-2 vectors for q/k/v/gate/up (right)
  and o/down (left).

Smoke = this same entrypoint with ``--cells 1`` (one real adapter cell
end-to-end through download → SVD → persist). Disk guard: abort below
15 GB free; downloaded snapshots evicted every ``--evict-every`` cells
(default 5, plan §12 #12).

Usage:
    uv run python scripts/issue604_adapter_svd.py --lines all
    uv run python scripts/issue604_adapter_svd.py --lines dial527 --cells 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.experiments.issue_604 import (  # noqa: E402
    ATTN_MODULES,
    HF_MODEL_REPO,
    HF_OVERFLOW_REPO,
    MLP_MODULES,
    TOP_K_VECTORS,
    AdapterCell,
    build_inventory,
    compose_svd,
    parse_lines_arg,
    result_metadata,
    rslora_scale,
    spectrum_metrics,
    stack_block_factors,
)

logger = logging.getLogger("issue604.phase_a")

MIN_FREE_GB = 15.0
HIDDEN = 3584


def _disk_guard() -> None:
    """Fail-loud disk guard (plan §12 #12): abort below MIN_FREE_GB free."""
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"disk guard tripped: {free_gb:.1f} GB free < {MIN_FREE_GB} GB — "
            "evict caches / free space before re-running (resume skips done cells)"
        )


def _download_cell(cell: AdapterCell, revisions: dict[str, str]) -> tuple[dict, Path, list[Path]]:
    """Download adapter_config.json + adapter_model.safetensors for one cell.

    Returns (config, safetensors_path, local_paths_for_eviction).
    """
    from huggingface_hub import hf_hub_download

    rev = revisions[cell.repo_id]
    paths = []
    cfg_path = hf_hub_download(cell.repo_id, f"{cell.subfolder}/adapter_config.json", revision=rev)
    st_path = hf_hub_download(
        cell.repo_id, f"{cell.subfolder}/adapter_model.safetensors", revision=rev
    )
    paths.extend([Path(cfg_path), Path(st_path)])
    config = json.loads(Path(cfg_path).read_text())
    return config, Path(st_path), paths


def _load_lora_factors(st_path: Path) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """{layer: {module: {"A": (r, d_in), "B": (d_out, r)}}} from safetensors.

    Keys follow the PEFT layout
    ``base_model.model.model.layers.{l}.{self_attn|mlp}.{module}.lora_{A,B}.weight``.
    Any non-decoder-layer LoRA key fails loud (gauge / layout drift).
    """
    from safetensors import safe_open

    out: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    with safe_open(st_path, framework="numpy") as f:
        for key in f.keys():  # noqa: SIM118 — safetensors handle, not a dict
            if ".lora_A." not in key and ".lora_B." not in key:
                raise AssertionError(f"unexpected non-LoRA tensor in adapter: {key}")
            parts = key.split(".")
            assert "layers" in parts, f"non-decoder-layer LoRA key (layout drift): {key}"
            layer = int(parts[parts.index("layers") + 1])
            module = parts[-3]  # ...{module}.lora_A.weight
            which = "A" if ".lora_A." in key else "B"
            tensor = f.get_tensor(key)
            arr = np.asarray(tensor, dtype=np.float32)
            out.setdefault(layer, {}).setdefault(module, {})[which] = arr
    for layer, mods in out.items():
        for module, ab in mods.items():
            assert set(ab) == {"A", "B"}, f"layer {layer} {module}: missing factor {set(ab)}"
            r = ab["A"].shape[0]
            assert ab["B"].shape[1] == r, (layer, module, ab["A"].shape, ab["B"].shape)
    return out


def _per_module_entry(
    module: str, U: np.ndarray, S: np.ndarray, V: np.ndarray
) -> tuple[dict, dict[str, np.ndarray]]:
    """Spectrum metrics + the residual-space top-2 vectors for one module.

    Comparison-validity map (plan §4 Phase A): q/k/v/gate/up expose RIGHT
    vectors (3584-dim residual input); o/down expose LEFT vectors
    (3584-dim residual output); everything else is spectra-only.
    """
    entry = {"module": module, "sigma": [float(s) for s in S], **spectrum_metrics(S)}
    vectors: dict[str, np.ndarray] = {}
    if module in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        assert V.shape[0] == HIDDEN, (module, V.shape)
        vectors["right_top2"] = V[:, :2].astype(np.float16)
    elif module in ("o_proj", "down_proj"):
        assert U.shape[0] == HIDDEN, (module, U.shape)
        vectors["left_top2"] = U[:, :2].astype(np.float16)
    return entry, vectors


def _resume_skip_ok(spectra_path: Path, cell: AdapterCell, revision: str) -> bool:
    """Validate an existing spectra JSON before resume-skipping its cell.

    Guards the resume path against stale artifacts (e.g. computed from a
    pre-layout-switch subfolder or an older pinned revision): the saved
    ``meta`` must match the CURRENT inventory on ``repo_id`` / ``subfolder``
    / ``revision`` / ``cell`` (cell_id). A missing field (older schema
    without ``revision``) or an unreadable/corrupt JSON counts as a
    mismatch — the caller recomputes. Returns True only when all four
    fields match.
    """
    try:
        meta = json.loads(spectra_path.read_text()).get("meta") or {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "resume invalid (unreadable spectra JSON) %s: %s — recomputing", spectra_path, exc
        )
        return False
    expected = {
        "repo_id": cell.repo_id,
        "subfolder": cell.subfolder,
        "revision": revision,
        "cell": cell.cell_id,
    }
    mismatches = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if mismatches:
        detail = "; ".join(
            f"{k}: saved={old!r} current={new!r}" for k, (old, new) in sorted(mismatches.items())
        )
        logger.warning("resume invalid (stale meta) %s: %s — recomputing", spectra_path, detail)
        return False
    return True


def process_cell(
    cell: AdapterCell, config: dict, st_path: Path, revision: str
) -> tuple[dict, dict]:
    """SVD one adapter cell. Returns (spectra_json_payload, npz_arrays)."""
    scale = rslora_scale(config)
    r = int(config["r"])
    targets = sorted(set(config.get("target_modules") or []))
    factors = _load_lora_factors(st_path)
    layers = sorted(factors.keys())
    is_all_linear = "gate_proj" in targets

    per_layer: list[dict] = []
    npz: dict[str, np.ndarray] = {}
    for layer in layers:
        mods = factors[layer]
        layer_rec: dict = {"layer": layer, "modules": [], "stacks": []}

        for module in (*ATTN_MODULES, *MLP_MODULES):
            if module not in mods:
                continue
            A, B = mods[module]["A"], mods[module]["B"]
            U, S, V = compose_svd(A, B, scale)
            entry, vectors = _per_module_entry(module, U, S, V)
            layer_rec["modules"].append(entry)
            for vk, arr in vectors.items():
                npz[f"L{layer}__{module}__{vk}"] = arr

        # Stacked attn key: row-stack [ΔW_q; ΔW_k; ΔW_v] over the shared
        # post-input_layernorm residual input.
        key_mods = [m for m in ("q_proj", "k_proj", "v_proj") if m in mods]
        if len(key_mods) == 3:
            blocks = [(mods[m]["A"], mods[m]["B"]) for m in key_mods]
            A_stk, B_stk = stack_block_factors(blocks, mode="row")
            U, S, V = compose_svd(A_stk, B_stk, scale)
            assert V.shape[0] == HIDDEN, V.shape
            k = min(TOP_K_VECTORS, V.shape[1])
            layer_rec["stacks"].append(
                {
                    "stack": "attn_key",
                    "rank_bound": 3 * r,
                    "sigma": [float(s) for s in S],
                    "truncation_energy_frac_top8": float(
                        (S[:k] ** 2).sum() / max((S**2).sum(), 1e-30)
                    ),
                    **spectrum_metrics(S),
                }
            )
            npz[f"L{layer}__attn_key__V8"] = V[:, :k].astype(np.float16)
            npz[f"L{layer}__attn_key__S"] = S.astype(np.float32)

        # Stacked MLP key (all-linear lines): row-stack [ΔW_gate; ΔW_up].
        mlp_key_mods = [m for m in ("gate_proj", "up_proj") if m in mods]
        if is_all_linear and len(mlp_key_mods) == 2:
            blocks = [(mods[m]["A"], mods[m]["B"]) for m in mlp_key_mods]
            A_stk, B_stk = stack_block_factors(blocks, mode="row")
            U, S, V = compose_svd(A_stk, B_stk, scale)
            assert V.shape[0] == HIDDEN, V.shape
            k = min(TOP_K_VECTORS, V.shape[1])
            layer_rec["stacks"].append(
                {
                    "stack": "mlp_key",
                    "rank_bound": 2 * r,
                    "sigma": [float(s) for s in S],
                    "truncation_energy_frac_top8": float(
                        (S[:k] ** 2).sum() / max((S**2).sum(), 1e-30)
                    ),
                    **spectrum_metrics(S),
                }
            )
            npz[f"L{layer}__mlp_key__V8"] = V[:, :k].astype(np.float16)
            npz[f"L{layer}__mlp_key__S"] = S.astype(np.float32)

        # Residual write: column-concat [ΔW_o | ΔW_down] (attn-only: ΔW_o).
        write_mods = [m for m in ("o_proj", "down_proj") if m in mods]
        if write_mods:
            blocks = [(mods[m]["A"], mods[m]["B"]) for m in write_mods]
            A_stk, B_stk = stack_block_factors(blocks, mode="col")
            U, S, V = compose_svd(A_stk, B_stk, scale)
            assert U.shape[0] == HIDDEN, U.shape
            k = min(TOP_K_VECTORS, U.shape[1])
            layer_rec["stacks"].append(
                {
                    "stack": "resid_write",
                    "stack_members": write_mods,
                    "rank_bound": len(write_mods) * r,
                    "sigma": [float(s) for s in S],
                    "truncation_energy_frac_top8": float(
                        (S[:k] ** 2).sum() / max((S**2).sum(), 1e-30)
                    ),
                    **spectrum_metrics(S),
                }
            )
            npz[f"L{layer}__resid_write__U8"] = U[:, :k].astype(np.float16)
            npz[f"L{layer}__resid_write__S"] = S.astype(np.float32)

        per_layer.append(layer_rec)

    payload = {
        "meta": result_metadata(
            PROJECT_ROOT,
            extra={
                "phase": "A",
                "cell": cell.cell_id,
                "line": cell.line,
                "repo_id": cell.repo_id,
                "subfolder": cell.subfolder,
                "revision": revision,
            },
        ),
        "cell": {
            "line": cell.line,
            "cell_id": cell.cell_id,
            "source_personas": list(cell.source_personas),
            "negative_personas": list(cell.negative_personas),
            "seed": cell.seed,
            "arm": cell.arm,
            "epoch": cell.epoch,
            "tags": list(cell.tags),
        },
        "adapter_config": {
            "r": r,
            "lora_alpha": config["lora_alpha"],
            "use_rslora": config["use_rslora"],
            "target_modules": targets,
            "scale": scale,
        },
        "n_layers_with_lora": len(layers),
        "layers": per_layer,
    }
    return payload, npz


def main() -> None:
    """Phase A entrypoint — same code path for smoke (--cells 1) and sweep."""
    parser = argparse.ArgumentParser(
        description="Task 604 Phase A: exact SVD of stored LoRA Delta-W per adapter cell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lines", default="all", help="all | dial | comma-separated line names")
    parser.add_argument("--cells", type=int, default=0, help="process at most N cells (0 = all)")
    parser.add_argument(
        "--cell-ids",
        default="",
        help="comma-separated explicit cell_id subset (smoke; same loop as the full sweep)",
    )
    parser.add_argument(
        "--include-dial-checkpoints",
        action="store_true",
        help="also SVD the dial checkpoint-NN intermediates (exploratory dose points)",
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval_results/issue_604"))
    parser.add_argument("--evict-every", type=int, default=5, help="HF-cache eviction cadence")
    parser.add_argument("--no-resume", action="store_true", help="recompute existing cells")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print("[phase=a_inventory]", flush=True)
    _disk_guard()

    from huggingface_hub import HfApi, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi()
    lines = parse_lines_arg(args.lines)
    # Pin revisions at run start (plan §10) + listing for the layout resolver.
    revisions: dict[str, str] = {}
    model_files: list[str] = []
    repos_needed = {HF_MODEL_REPO}
    if "i541" in lines:
        repos_needed.add(HF_OVERFLOW_REPO)
    for repo in sorted(repos_needed):
        revisions[repo] = api.repo_info(repo).sha
    model_files = list_repo_files(HF_MODEL_REPO, revision=revisions[HF_MODEL_REPO])

    cells = build_inventory(
        lines,
        include_dial_checkpoints=args.include_dial_checkpoints,
        model_repo_files=model_files,
    )
    if args.cell_ids:
        wanted = [t.strip() for t in args.cell_ids.split(",") if t.strip()]
        missing = sorted(set(wanted) - {c.cell_id for c in cells})
        assert not missing, f"--cell-ids not in the enumerated inventory: {missing}"
        cells = [c for c in cells if c.cell_id in set(wanted)]
    if args.cells > 0:
        cells = cells[: args.cells]
    logger.info("inventory: %d cells across lines %s", len(cells), lines)

    out_dir = Path(args.out_dir)
    spectra_dir = out_dir / "spectra"
    vectors_dir = out_dir / "vectors"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    gitignore = vectors_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*.npz\n")  # binary vectors go to HF, not git

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "meta": result_metadata(PROJECT_ROOT, extra={"phase": "A"}),
        "revisions": revisions,
        "lines": lines,
        "n_cells": len(cells),
        "cells": [],
    }
    # Manifest header (pinned revisions + lines) lands BEFORE the sweep loop and
    # is rewritten per cell — a mid-sweep crash never loses the revision record.
    manifest_path.write_text(json.dumps(manifest, indent=1))

    print("[phase=a_svd]", flush=True)
    eviction_paths: list[Path] = []
    t0 = time.time()
    done = 0
    for i, cell in enumerate(cells):
        spectra_path = spectra_dir / cell.line / f"{cell.cell_id}.json"
        npz_path = vectors_dir / cell.line / f"{cell.cell_id}.npz"
        cell_rec = {
            "line": cell.line,
            "cell_id": cell.cell_id,
            "repo_id": cell.repo_id,
            "subfolder": cell.subfolder,
            "revision": revisions[cell.repo_id],
            "spectra": str(spectra_path.relative_to(PROJECT_ROOT)),
            "vectors": str(npz_path.relative_to(PROJECT_ROOT)),
            "tags": list(cell.tags),
        }
        if (
            spectra_path.exists()
            and npz_path.exists()
            and not args.no_resume
            and _resume_skip_ok(spectra_path, cell, revisions[cell.repo_id])
        ):
            # NPZ sidecar existence for the same cell is the npz_path.exists()
            # check above (the path embeds the cell_id); meta validation is
            # _resume_skip_ok — on mismatch / unreadable JSON we fall through
            # and recompute.
            logger.info(
                "[%d/%d] skip (done, meta-validated): %s/%s",
                i + 1,
                len(cells),
                cell.line,
                cell.cell_id,
            )
            manifest["cells"].append({**cell_rec, "status": "done"})
            manifest_path.write_text(json.dumps(manifest, indent=1))
            continue
        _disk_guard()
        try:
            config, st_path, dl_paths = _download_cell(cell, revisions)
        except EntryNotFoundError as exc:
            if cell.repo_id == HF_OVERFLOW_REPO:
                # i541 cells bypass the model-repo listing check (plan §15
                # secondary line): a missing overflow-repo cell takes the
                # registered "N/A — not stored" path instead of crashing
                # the sweep.
                logger.warning(
                    "N/A — not stored (overflow repo): %s/%s: %s", cell.line, cell.cell_id, exc
                )
                manifest["cells"].append({**cell_rec, "status": "not-stored"})
                manifest_path.write_text(json.dumps(manifest, indent=1))
                continue
            raise
        eviction_paths.extend(dl_paths)
        payload, npz = process_cell(cell, config, st_path, revisions[cell.repo_id])
        spectra_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        spectra_path.write_text(json.dumps(payload, indent=1))
        np.savez_compressed(npz_path, **npz)
        manifest["cells"].append({**cell_rec, "status": "done"})
        manifest_path.write_text(json.dumps(manifest, indent=1))
        done += 1
        logger.info(
            "[%d/%d] %s/%s done (%.1fs elapsed)",
            i + 1,
            len(cells),
            cell.line,
            cell.cell_id,
            time.time() - t0,
        )
        if done % args.evict_every == 0:
            for p in eviction_paths:
                # hf_hub_download returns a symlink into the cache; remove the
                # blob it points to, then the link (keeps peak disk < ~10 GB).
                try:
                    target = p.resolve()
                    target.unlink(missing_ok=True)
                    p.unlink(missing_ok=True)
                except OSError as exc:  # eviction is best-effort; guard re-checks disk
                    logger.warning("cache eviction failed for %s: %s", p, exc)
            eviction_paths.clear()
            logger.info("HF cache evicted after %d processed cells", done)

    manifest_path.write_text(json.dumps(manifest, indent=1))
    logger.info("Phase A complete: %d newly processed, %d total cells", done, len(cells))
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
