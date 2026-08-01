#!/usr/bin/env python
"""#1947 — single-visit organism fleet: cell registry + dispatch shards (plan §4.1/§5).

56 trained arms (plan §4.1 "The grid (56 trained arms)"; the §10 "60-arm"
phrase is a plan-internal miscount — the enumerated axes give 48 + 4 + 4):

- 48 content single-visit cells: {syc, imp, cas} × {pers, bare, conv, icl}
  × {con, po} × seeds {42, 137} — slug ``<beh>-<ctx>-<regime>-sv-s<seed>``;
- 4 repeat-regime control cells (imp-pers, imp-bare, syc-pers, syc-bare ×
  con × s42) — slug ``<beh>-<ctx>-con-rep-s42`` (80-row new-pool subsample,
  15 epochs = 75 optimizer steps);
- 4 marker cells (4 contexts × con × s42) — slug ``mk-<ctx>-con-sv-s42``.

ONE LR per cell = the #1481 verdict-arm LR, read from the committed
``eval_results/issue_1481/analysis/verdict_manifest.json`` (plan §4.1;
marker cells pin lr 5e-6 per the marker-training recipe). The registry is
pure metadata — everything heavy (datagen, training, battery) lives in the
sibling ``issue1947_*`` drivers.

CLI::

    uv run python scripts/issue1947_cells.py --dispatch fleet-a          # 24 slugs
    uv run python scripts/issue1947_cells.py --dispatch fleet-b          # 31 slugs
    uv run python scripts/issue1947_cells.py --dispatch pilot            # 1 slug
    uv run python scripts/issue1947_cells.py --manifest-out eval_results/issue_1947/cells_manifest.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent

ISSUE = 1947
DATA_PREFIX = "issue1947_singlevisit"  # HF data-repo bucket (plan §10 destinations)
ADAPTER_PREFIX = "issue1947"  # overflow model-repo run dirs: issue1947/<slug>/checkpoint-<step>
OUT_ROOT_DEFAULT = "/workspace/eps-issue-1947"  # pod out-root contract (user RunPod override)
VERDICT_MANIFEST_PATH = REPO_ROOT / "eval_results/issue_1481/analysis/verdict_manifest.json"

# ── Grid axes (plan §4.1) ────────────────────────────────────────────────────
BEH_KEYS: tuple[str, ...] = ("syc", "imp", "cas")
BEHAVIOR_BY_KEY: dict[str, str] = {
    "syc": "sycophancy",  # c3-v2 definition (questiongen_sycophancy_v2.json)
    "imp": "impolite",  # c2 definition (questiongen_impolite.json)
    "cas": "writing_style",  # #1434 definition (questiongen_writing_style.json)
}
CTX_KEYS: tuple[str, ...] = ("pers", "bare", "conv", "icl")
REGIMES: tuple[str, ...] = ("con", "po")
SEEDS: tuple[int, ...] = (42, 137)
MARKER_SEED = 42
PILOT_SLUG = "syc-pers-con-sv-s42"  # plan §4.4 P1

# Repeat-regime control cells (plan §4.1: the repetition-heaviest CLEAN cells)
REP_CELL_KEYS: tuple[tuple[str, str], ...] = (
    ("imp", "pers"),
    ("imp", "bare"),
    ("syc", "pers"),
    ("syc", "bare"),
)

# ── Single-visit arithmetic (plan §4.2/§4.3) ─────────────────────────────────
EFFECTIVE_BATCH = 16  # batch 4 × grad-accum 4 (#1090 fu4 / #1481 verbatim)
CONTENT_STEPS = 75
CONTENT_ROWS = CONTENT_STEPS * EFFECTIVE_BATCH  # 1,200 — consumed exactly once
CONTENT_POS = 300
CONTENT_NEG = 300  # con only (1:1 contrastive rule)
CONTENT_GENERIC_CON = CONTENT_ROWS - CONTENT_POS - CONTENT_NEG  # 600
CONTENT_GENERIC_PO = CONTENT_ROWS - CONTENT_POS  # 900
REP_ROWS = 80  # 20 pos + 20 neg + 40 generic (parent composition on the NEW pool)
REP_EPOCHS = 15  # 80-row mix × 15 epochs = 75 optimizer steps
REP_POS, REP_NEG, REP_GENERIC = 20, 20, 40
# Marker (#1112/#1333/#1481 line, scaled to single-visit at the 400-step ceiling)
MARKER_LR = 5e-6  # the clean selective window (marker-training-recipe.md)
MARKER_CEILING_STEPS = 400
MARKER_SAVE_STEPS = 10
MARKER_POS = 1280  # distinct generic questions, greedy base R + " ※"
MARKER_NEG = 4 * MARKER_POS  # 5,120 — the marker-line 1:4 pos:neg convention
MARKER_ROWS = MARKER_POS + MARKER_NEG  # 6,400 = 400 steps × 16, single-visit


def mix_prefix(slug: str) -> str:
    """HF data-repo prefix holding the cell's train_mix.jsonl + manifests."""
    return f"{DATA_PREFIX}/mixes/{slug}"


def predicted_consumption(n_rows: int, *, effective_batch: int = EFFECTIVE_BATCH) -> list[int]:
    """Builder-PREDICTED optimizer step per mix-file row index (plan §4.2):
    a sequential sampler over the pre-shuffled file consumes rows
    ``[effective_batch*k, effective_batch*(k+1))`` at global step k (per epoch;
    repeat-regime cells re-consume the same mapping each epoch)."""
    return [i // effective_batch for i in range(n_rows)]


# ── Verdict-LR lookup (plan §4.1 — eval_results/issue_1481/.../verdict_manifest.json) ──

_VERDICT_CACHE: dict | None = None


def _verdict_manifest() -> dict:
    global _VERDICT_CACHE
    if _VERDICT_CACHE is None:
        if not VERDICT_MANIFEST_PATH.is_file():
            raise FileNotFoundError(
                f"#1481 verdict manifest missing at {VERDICT_MANIFEST_PATH} — the per-cell "
                "LR source (plan §4.1). In a sparse worktree run "
                "`git sparse-checkout add eval_results/issue_1481` first."
            )
        _VERDICT_CACHE = json.loads(VERDICT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return _VERDICT_CACHE


def verdict_lr(beh_key: str, ctx_key: str, regime: str, seed: int) -> tuple[float, dict]:
    """(lr, lr_source) for one content cell — the parent verdict-arm LR plus its
    provenance record (parent arm_id / selection step / rate / in_band / fallback)."""
    m = _verdict_manifest()
    try:
        rec = m["content"][beh_key][ctx_key]["seeds"][str(seed)][regime]
    except KeyError as e:
        raise KeyError(
            f"verdict manifest has no content[{beh_key}][{ctx_key}].seeds[{seed}].{regime}"
        ) from e
    lr = float(rec["lr"])
    sel = rec.get("selection") or {}
    source = {
        "parent_arm_id": rec.get("arm_id"),
        "rule": rec.get("rule"),
        "selection_step": sel.get("step"),
        "selection_rate": sel.get("rate"),
        "in_band": sel.get("in_band"),
        "fallback": sel.get("fallback"),
        "manifest": str(VERDICT_MANIFEST_PATH.relative_to(REPO_ROOT)),
    }
    return lr, source


def parent_pass_count(source: dict) -> float | None:
    """Parent data passes at the verdict rung (step × batch 16 / the parent 80-row
    con mix — the §5 pass-count interpretation frame; None when step missing)."""
    step = source.get("selection_step")
    if step is None:
        return None
    return round(step * EFFECTIVE_BATCH / 80.0, 2)


# ── Cell registry ────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CellSpec:
    """One trained arm of the #1947 grid (pure metadata)."""

    slug: str
    kind: str  # "content" | "marker"
    visit: str  # "sv" (single-visit) | "rep" (repeat-regime control)
    beh_key: str  # syc | imp | cas | mk
    behavior: str  # registered behavior name ("marker" for mk cells)
    ctx_key: str  # pers | bare | conv | icl
    regime: str  # con | po
    seed: int
    lr: float
    lr_source: dict
    n_rows: int
    epochs: int
    max_steps: int
    save_steps: int

    @property
    def mix_hub_prefix(self) -> str:
        return mix_prefix(self.slug)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["mix_hub_prefix"] = self.mix_hub_prefix
        d["parent_pass_count"] = parent_pass_count(self.lr_source)
        return d


def content_slug(beh_key: str, ctx_key: str, regime: str, visit: str, seed: int) -> str:
    return f"{beh_key}-{ctx_key}-{regime}-{visit}-s{seed}"


def marker_slug(ctx_key: str) -> str:
    return f"mk-{ctx_key}-con-sv-s{MARKER_SEED}"


def build_cells() -> tuple[CellSpec, ...]:
    """The full 56-arm registry (48 content sv + 4 repeat-regime + 4 marker)."""
    cells: list[CellSpec] = []
    for beh_key in BEH_KEYS:
        for ctx_key in CTX_KEYS:
            for regime in REGIMES:
                for seed in SEEDS:
                    lr, src = verdict_lr(beh_key, ctx_key, regime, seed)
                    cells.append(
                        CellSpec(
                            slug=content_slug(beh_key, ctx_key, regime, "sv", seed),
                            kind="content",
                            visit="sv",
                            beh_key=beh_key,
                            behavior=BEHAVIOR_BY_KEY[beh_key],
                            ctx_key=ctx_key,
                            regime=regime,
                            seed=seed,
                            lr=lr,
                            lr_source=src,
                            n_rows=CONTENT_ROWS,
                            epochs=1,
                            max_steps=CONTENT_STEPS,
                            save_steps=5,
                        )
                    )
    for beh_key, ctx_key in REP_CELL_KEYS:
        lr, src = verdict_lr(beh_key, ctx_key, "con", 42)
        cells.append(
            CellSpec(
                slug=content_slug(beh_key, ctx_key, "con", "rep", 42),
                kind="content",
                visit="rep",
                beh_key=beh_key,
                behavior=BEHAVIOR_BY_KEY[beh_key],
                ctx_key=ctx_key,
                regime="con",
                seed=42,
                lr=lr,
                lr_source=src,
                n_rows=REP_ROWS,
                epochs=REP_EPOCHS,
                max_steps=CONTENT_STEPS,
                save_steps=5,
            )
        )
    for ctx_key in CTX_KEYS:
        cells.append(
            CellSpec(
                slug=marker_slug(ctx_key),
                kind="marker",
                visit="sv",
                beh_key="mk",
                behavior="marker",
                ctx_key=ctx_key,
                regime="con",
                seed=MARKER_SEED,
                lr=MARKER_LR,
                lr_source={"rule": "marker_recipe_pin", "note": "lr<=5e-6 clean window"},
                n_rows=MARKER_ROWS,
                epochs=1,
                max_steps=MARKER_CEILING_STEPS,
                save_steps=MARKER_SAVE_STEPS,
            )
        )
    assert len(cells) == 56, len(cells)
    slugs = [c.slug for c in cells]
    assert len(set(slugs)) == 56, "duplicate cell slugs"
    return tuple(cells)


CELLS: tuple[CellSpec, ...] = build_cells()
CELL_BY_SLUG: dict[str, CellSpec] = {c.slug: c for c in CELLS}

# Plan-§4.1 LR sanity pins (values read off the committed verdict manifest at
# implementation time; a manifest edit that moves any of these fails loud here).
_LR_PINS = {
    "syc-pers-con-sv-s42": 1e-5,
    "syc-icl-po-sv-s42": 3e-5,
    "imp-pers-con-sv-s42": 3e-5,
    "imp-bare-con-sv-s42": 3e-5,
    "imp-icl-con-sv-s42": 1e-4,
    "imp-icl-con-sv-s137": 1e-5,
    "imp-icl-po-sv-s137": 1e-4,
    "cas-conv-po-sv-s42": 3e-5,
    "cas-icl-con-sv-s137": 1e-4,
}
for _slug, _lr in _LR_PINS.items():
    assert math.isclose(CELL_BY_SLUG[_slug].lr, _lr), (_slug, CELL_BY_SLUG[_slug].lr, _lr)


# ── Dispatch shards (plan §4.4 P1/P2) ────────────────────────────────────────


def dispatch_shards() -> dict[str, tuple[str, ...]]:
    """Deterministic dispatch shards: ``pilot`` (P1) = syc-pers-con-sv-s42;
    ``fleet-a`` = first 24 non-pilot single-visit content slugs (sorted);
    ``fleet-b`` = remaining 23 + 4 repeat-regime + 4 marker cells."""
    sv_content = sorted(
        c.slug for c in CELLS if c.kind == "content" and c.visit == "sv" and c.slug != PILOT_SLUG
    )
    assert len(sv_content) == 47, len(sv_content)
    rep = [c.slug for c in CELLS if c.visit == "rep"]
    marker = [c.slug for c in CELLS if c.kind == "marker"]
    shards = {
        "pilot": (PILOT_SLUG,),
        "fleet-a": tuple(sv_content[:24]),
        "fleet-b": tuple(sv_content[24:]) + tuple(rep) + tuple(marker),
        "all": tuple(c.slug for c in CELLS),
        "content": tuple(c.slug for c in CELLS if c.kind == "content"),
        "marker": tuple(marker),
        "rep": tuple(rep),
    }
    assert len(shards["fleet-a"]) == 24, len(shards["fleet-a"])
    assert len(shards["fleet-b"]) == 31, len(shards["fleet-b"])  # 23 + 4 + 4 (plan §4.4)
    assert not set(shards["fleet-a"]) & set(shards["fleet-b"])
    assert PILOT_SLUG not in shards["fleet-a"] + shards["fleet-b"]
    return shards


def write_manifest(path: Path) -> dict:
    """The machine-readable 56-cell manifest + shard map (plan §4.4 P0 output)."""
    payload = {
        "issue": ISSUE,
        "data_prefix": DATA_PREFIX,
        "adapter_prefix": ADAPTER_PREFIX,
        "out_root": OUT_ROOT_DEFAULT,
        "verdict_manifest": str(VERDICT_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "pilot": PILOT_SLUG,
        "arithmetic": {
            "effective_batch": EFFECTIVE_BATCH,
            "content": {
                "rows": CONTENT_ROWS,
                "steps": CONTENT_STEPS,
                "pos": CONTENT_POS,
                "neg": CONTENT_NEG,
                "generic_con": CONTENT_GENERIC_CON,
                "generic_po": CONTENT_GENERIC_PO,
            },
            "rep": {"rows": REP_ROWS, "epochs": REP_EPOCHS},
            "marker": {
                "rows": MARKER_ROWS,
                "pos": MARKER_POS,
                "neg": MARKER_NEG,
                "ceiling_steps": MARKER_CEILING_STEPS,
                "save_steps": MARKER_SAVE_STEPS,
            },
        },
        "dispatch": {k: list(v) for k, v in dispatch_shards().items() if k not in ("all",)},
        "cells": [c.to_json() for c in CELLS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI: print dispatch shards / write the cell manifest (returns 0)."""
    p = argparse.ArgumentParser(description="#1947 cell registry")
    p.add_argument("--dispatch", choices=sorted(dispatch_shards()), default=None)
    p.add_argument("--manifest-out", default=None, help="write the full manifest JSON here")
    p.add_argument("--format", choices=("lines", "csv"), default="lines")
    args = p.parse_args(argv)
    if args.manifest_out:
        payload = write_manifest(Path(args.manifest_out))
        print(f"[cells] wrote {args.manifest_out} ({len(payload['cells'])} cells)")
    if args.dispatch:
        slugs = dispatch_shards()[args.dispatch]
        print(",".join(slugs) if args.format == "csv" else "\n".join(slugs))
    if not args.manifest_out and not args.dispatch:
        p.error("pass --dispatch and/or --manifest-out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
