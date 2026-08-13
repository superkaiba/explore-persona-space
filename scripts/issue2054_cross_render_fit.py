#!/usr/bin/env python
"""#2054 cross-render fit: SOURCE-render context -> TARGET-render answer, fit directly.

0 GPU-h. Read-only over the already-captured activation stores.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The nine-rung transfer ladder (``scripts/issue2054_ladder.py``) asks a MECHANISM
question: keep the source operator ``W_s`` frozen, feed it TARGET contexts, and
see what minimal correction restores it. Every rung's held-out input is
``Xt`` (target contexts).

This script asks a different, INFORMATION question: fit a fresh ridge from the
SOURCE render's context vector to the TARGET render's answer vector, on
conversations rendered BOTH ways --

    v_C(source render, conversation c)  ->  v_A(target render, conversation c)

i.e. "does the chat-template representation of a conversation already determine
how that same conversation's answer is encoded inside a story?"

CAVEAT, stated up front: this fit SEES cross-render pairs and has full d^2
capacity, so it is a PREDICTABILITY upper bound, NOT evidence of a shared
operator. It cannot support a "same operator" claim. Ladder rung 9 escapes that
objection only because its ``A`` sees contexts only and its ``B`` sees answers
only -- neither ever touches pairs. Read this number against the matched
within-cell reference, never as a ladder rung.

DESIGN
------
* Row pairing. The INSERTED condition is row-paired across renders BY
  CONSTRUCTION: same conversation, same verbatim answer text, different render.
  The ON-POLICY condition is NOT -- the model writes a different answer in each
  render -- so its cross-render fit conflates the render change with a different
  answer. Inserted is the primary read; on-policy runs as a labelled secondary.
* One shared row set. All targets for a given (model, condition) are restricted
  to the 4-WAY intersection of conv_ids across {chat, bare_text, attrib_quoted,
  bare_label}. So the three cross-render numbers are mutually comparable, and
  each is comparable to its own matched within-cell reference.
* Matched within-cell reference. The published per-cell ceilings were fit on
  each cell's OWN full row set, which is LARGER than this intersection. Comparing
  a cross-render fit against those would be an unmatched comparison, so this
  script recomputes the within-cell fit (target context -> target answer) on the
  SAME intersection and the SAME folds. Both are reported; the matched one is
  the honest denominator.
* Estimator parity. Fits call ``issue2054_fits._ridge_gcv_fit_predict`` and score
  with ``issue2054_fits._r2_matrix`` -- the SAME functions the production per-cell
  fits and the ladder use (GCV lambda selection, dof cap 0.9, standardize-X /
  center-Y, dual Gram solve). Nothing is re-implemented, so there is no
  estimator-parity diff to record.
* Folds. The shared conversation-grouped fold map (k=5, seed 137) used by every
  #2054 cell, so held-out sets line up with the published numbers.

IDENTITY GATE
-------------
source == target (chat -> chat) must reproduce the matched within-cell chat fit
to floating-point tolerance. That is a free correctness check on the pairing and
alignment code; the script FAILS LOUD if it does not hold.

Usage:
  uv run python scripts/issue2054_cross_render_fit.py --pilot     # 1 pair, 1 fold, measure
  uv run python scripts/issue2054_cross_render_fit.py             # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# BEFORE any heavy import: torch/BLAS freeze their thread pools from
# OMP_NUM_THREADS at import time, so the shared-VM thread caps must be in the
# environment first (.claude/rules/code-style.md § Shared-VM CPU thread caps).
load_dotenv()

import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse the PRODUCTION estimator + fold machinery verbatim (estimator parity).
from scripts.issue2054_fits import (  # noqa: E402
    D_AMBIENT,
    _fold_split,
    _load_activation_npz,
    _r2_matrix,
    _ridge_gcv_fit_predict,
)

ASSIST = "conversation_paired_stories_assistant"
FORMS = ("chat", "bare_text", "attrib_quoted", "bare_label")
MODELS = ("qwen2.5-7b-instruct", "qwen2.5-7b")
CONDITIONS = ("inserted", "on_policy")
SOURCE_FORM = "chat"

# --figure2 mode (#2054 writeup Result 2 paired-fit column): character-story targets.
CHARS = ("char_helios", "char_wren", "char_vex", "char_dana")
# Per (figure, group): (source condition, source form, target condition). Sources are
# always the assistant; targets are each character's bare_label story cell. 2a is
# provenance-matched (assistant-in-story -> character, same condition); 2b re-uses the
# single inserted chat anchor against both target conditions (no on-policy chat anchor
# exists -- the 2x2 chat anchor is inserted-only, same as the ladder).
FIG2_GROUPS: dict[str, tuple[str, str, str]] = {
    "2a__inserted": ("inserted", "bare_label", "inserted"),
    "2a__on_policy": ("on_policy", "bare_label", "on_policy"),
    "2b__inserted": ("inserted", "chat", "inserted"),
    "2b__on_policy": ("inserted", "chat", "on_policy"),
}

ACT_DIR = (
    Path("/mnt/eps-data")
    / __import__("os").environ.get("USER", "")
    / ("issue2054_crossrender/activations")
)
# The PRODUCTION fold map (26,889 conversations, all 5 variants, 2026-08-06) is the
# blob on origin/issue-2054. main's committed copy is the STALE 2026-08-04
# single-variant SMOKE map (1,761 conversations, char_helios only) — reading it
# understates the pool ~15x and collapses every render intersection to a few
# hundred rows, which silently turns every fit into a regularization-limit read
# (#2054 2026-08-10 round encoded the same refusal). Guard below, not a comment.
FOLD_MAP_REF = "origin/issue-2054"
FOLD_MAP_PATH_IN_REPO = "eval_results/issue_2054/shared_fold_map.json"
FOLD_MAP_MIN_CONV = 20_000
FOLD_MAP_MIN_VARIANTS = 5
OUT_DIR = _REPO / "eval_results/issue_2054/analyzer_companions"
OUT = OUT_DIR / "cross_render_fit.json"
FIG2_OUT = OUT_DIR / "cross_render_fit_characters.json"


def _shard_out(conditions: list[str], models: list[str]) -> Path:
    """Per-shard output path.

    The (condition x model) groups are launched as PARALLEL processes, so a single
    fixed output path would have them clobber each other (last writer wins, three
    shards silently lost). Any run that is not the full default grid writes a
    shard file instead; `--merge` then folds the shards into the canonical
    cross_render_fit.json.
    """
    if set(conditions) == set(CONDITIONS) and set(models) == set(MODELS):
        return OUT
    slug = "__".join(sorted(conditions) + sorted(models)).replace(".", "")
    return OUT_DIR / f"cross_render_fit.shard__{slug}.json"


IDENTITY_TOL = 1e-9


def _load_production_fold_map() -> dict:
    """Read the production fold map from the branch blob and REFUSE the smoke map."""
    import subprocess

    subprocess.run(
        ["git", "-C", str(_REPO), "fetch", "origin", "issue-2054", "--quiet"], check=False
    )
    out = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{FOLD_MAP_REF}:{FOLD_MAP_PATH_IN_REPO}"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"cannot read {FOLD_MAP_REF}:{FOLD_MAP_PATH_IN_REPO} (rc={out.returncode}): "
            f"{out.stderr.strip()[:300]}"
        )
    d = json.loads(out.stdout)
    for key in ("fold_of", "k", "seed"):
        if key not in d:
            raise ValueError(f"fold map missing {key!r}")
    n_conv = len(d["fold_of"])
    variants = d.get("variants") or []
    if n_conv < FOLD_MAP_MIN_CONV or len(variants) < FOLD_MAP_MIN_VARIANTS:
        raise RuntimeError(
            f"REFUSING the fold map at {FOLD_MAP_REF}: n_conv={n_conv:,} "
            f"(floor {FOLD_MAP_MIN_CONV:,}), variants={variants} "
            f"(floor {FOLD_MAP_MIN_VARIANTS}). This is the smoke map, not production — "
            "every intersection would collapse and every fit would be a "
            "regularization-limit read."
        )
    d["_source_ref"] = FOLD_MAP_REF
    d["_n_conv"] = n_conv
    d["_variants"] = variants
    return d


def _log(msg: str) -> None:
    print(msg, flush=True)


def _cell_path(cond: str, form: str, model: str, variant: str = ASSIST) -> Path:
    return ACT_DIR / f"{variant}__{cond}__{form}__{model}.npz"


def _load_cell(cond: str, form: str, model: str, variant: str = ASSIST) -> dict:
    p = _cell_path(cond, form, model, variant)
    if not p.is_file():
        raise FileNotFoundError(f"activation store not staged: {p}")
    act = _load_activation_npz(p)
    if act is None:
        raise ValueError(f"empty activation store: {p}")
    ids = act["conv_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate conv_ids in {p} — row pairing would be ambiguous")
    return act


def _aligned(act: dict, order: list[str], key: str) -> np.ndarray:
    """Return act[key] rows re-ordered to follow `order` (conv_id list)."""
    pos = {cid: i for i, cid in enumerate(act["conv_ids"])}
    idx = np.fromiter((pos[c] for c in order), dtype=np.int64, count=len(order))
    return np.asarray(act[key], dtype=np.float32)[idx]


def _run_fit(
    X: np.ndarray, Y: np.ndarray, folds: list[list[int]], *, only_fold: int | None = None
) -> dict:
    """K-fold held-out R^2 with the production ridge. Returns per-fold + pooled."""
    per_fold, infos = [], []
    for fi, te in enumerate(folds):
        if only_fold is not None and fi != only_fold:
            continue
        tr = [i for f2, fold in enumerate(folds) if f2 != fi for i in fold]
        if not tr or not te:
            continue
        t0 = time.time()
        preds, info = _ridge_gcv_fit_predict(X[tr], Y[tr], X[te])
        r2 = _r2_matrix(Y[te].astype(np.float64), preds)
        info["wall_s"] = round(time.time() - t0, 1)
        info["n_test"] = len(te)
        per_fold.append(r2)
        infos.append(info)
        _log(
            f"      fold {fi}: r2={r2:+.4f} n_train={info['n_train']} "
            f"lam={info['best_lambda']:g} dof={info['dof']:.0f} wall={info['wall_s']}s"
        )
    return {
        "per_fold_r2": per_fold,
        "pooled_r2": float(np.mean(per_fold)) if per_fold else float("nan"),
        "fold_info": infos,
    }


def _spot_check_alignment(
    src: dict, tgt: dict, order: list[str], X: np.ndarray, Y: np.ndarray
) -> None:
    """Assert X/Y rows really are the stores' rows for sampled conv_ids (fail loud).

    Mechanical guard on `_aligned`'s reordering for the figure-2 path, where source and
    target stores index disjoint row sets: for 3 deterministic sample ids, the aligned
    row must equal the store's own row at the store's native index.
    """
    src_pos = {c: i for i, c in enumerate(src["conv_ids"])}
    tgt_pos = {c: i for i, c in enumerate(tgt["conv_ids"])}
    for j in (0, len(order) // 2, len(order) - 1):
        cid = order[j]
        if not np.array_equal(X[j], np.asarray(src["v_C"], dtype=np.float32)[src_pos[cid]]):
            raise AssertionError(f"alignment spot-check FAILED on source row for conv {cid}")
        if not np.array_equal(Y[j], np.asarray(tgt["v_A"], dtype=np.float32)[tgt_pos[cid]]):
            raise AssertionError(f"alignment spot-check FAILED on target row for conv {cid}")


def _fig2_shard_out(models: list[str], figures: list[str]) -> Path:
    """Per-shard output for --figure2 (parallel shards must not clobber each other)."""
    slug = "__".join(sorted(figures) + sorted(models)).replace(".", "")
    return OUT_DIR / f"cross_render_fit_characters.shard__{slug}.json"


def run_figure2(args) -> int:
    """Cross-render paired fits onto the four character-story bare_label targets.

    Per-PAIR conv_id intersections (the ladder's own row convention, so the paired fit
    is comparable to the rungs on the same figure line + normalizable by the banked
    ladder per-target ceiling), NOT the assistant grid's 4-way form intersection.
    Estimator parity: the same `_ridge_gcv_fit_predict` (GCV, dof_cap 0.9); at
    n_train < d its thin SVD is the reduced-basis (row-span) fit — the #1887
    convention — and such cells are flagged `well_posed_ambient: false` for
    hollow-marking downstream. Checkpoints each pair to a JSONL and resume-skips.
    """
    fold_map = _load_production_fold_map()
    k, fold_of = int(fold_map["k"]), fold_map["fold_of"]
    figures = sorted({g.split("__", 1)[0] for g in FIG2_GROUPS} & set(args.fig2_figures))
    shard_out = _fig2_shard_out(list(args.models), figures)
    ckpt = shard_out.with_suffix(".partial.jsonl")
    done: set[tuple] = set()
    rows: list[dict] = []
    if ckpt.is_file():
        for line in ckpt.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["figure"], r["group"], r["model"], r["character"]))
                rows.append(r)
        _log(f"[fig2] resume: {len(done)} pairs already checkpointed in {ckpt.name}")

    only = 0 if args.pilot else None
    t_start = time.time()
    unit, n_units = (
        0,
        len(args.models)
        * sum(1 for g in FIG2_GROUPS if g.split("__", 1)[0] in figures)
        * len(CHARS),
    )
    for model in args.models:
        for grp, (scond, sform, tcond) in FIG2_GROUPS.items():
            fig = grp.split("__", 1)[0]
            if fig not in figures:
                continue
            src = _load_cell(scond, sform, model)
            src_ids = set(src["conv_ids"])
            for ch in CHARS:
                unit += 1
                key = (fig, grp, model, ch)
                if key in done:
                    continue
                tgt = _load_cell(tcond, "bare_label", model, variant=ch)
                common = src_ids & set(tgt["conv_ids"]) & set(fold_of)
                order = sorted(common)
                n = len(order)
                if n == 0:
                    raise RuntimeError(f"empty intersection for {key} — wrong store or fold map")
                folds = _fold_split(order, fold_of, k)
                n_train_typ = n - max(len(f) for f in folds)
                X = _aligned(src, order, "v_C")
                Y = _aligned(tgt, order, "v_A")
                _spot_check_alignment(src, tgt, order, X, Y)
                _log(
                    f"[fig2] unit {unit}/{n_units} {model} {grp} -> {ch}: n={n:,} "
                    f"n_train={n_train_typ:,} vs d={D_AMBIENT:,} "
                    f"({'WELL-POSED' if n_train_typ > D_AMBIENT else 'reduced-basis'}) "
                    f"elapsed={time.time() - t_start:.0f}s"
                )
                cross = _run_fit(X, Y, folds, only_fold=only)
                row = {
                    "figure": fig,
                    "group": grp,
                    "model": model,
                    "character": ch,
                    "source_variant": ASSIST,
                    "source_condition": scond,
                    "source_form": sform,
                    "target_condition": tcond,
                    "target_form": "bare_label",
                    "n_pair": n,
                    "n_train_typical": n_train_typ,
                    "well_posed_ambient": bool(n_train_typ > D_AMBIENT),
                    "cross_render_r2": cross["pooled_r2"],
                    "cross_render_per_fold": cross["per_fold_r2"],
                    "cross_fold_info": cross["fold_info"],
                }
                rows.append(row)
                with ckpt.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                if args.pilot:
                    _log("[fig2] PILOT complete — 1 pair, 1 fold.")
                    return 0

    results = {
        "what": (
            "SOURCE-render context vector -> character-story TARGET answer vector, ridge fit "
            "DIRECTLY on cross-render pairs (per-PAIR conv intersections = the ladder's row "
            "convention). Predictability upper bound, NOT a shared-operator claim. 2a: "
            "assistant-in-story bare_label -> character, provenance-matched. 2b: assistant "
            "inserted chat anchor -> character, both target conditions. Normalize against the "
            "banked ladder per-target ceilings; reduced-basis (n_train < d) cells are flagged."
        ),
        "estimator": "issue2054_fits._ridge_gcv_fit_predict (GCV, dof_cap=0.9) — production parity",
        "d_ambient": D_AMBIENT,
        "cells": rows,
    }
    shard_out.parent.mkdir(parents=True, exist_ok=True)
    shard_out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    _log(f"[fig2] wrote {shard_out.relative_to(_REPO)} ({len(rows)} rows)")
    return 0


def merge_figure2() -> int:
    """Fold cross_render_fit_characters.shard__*.json into the canonical file, fail-loud."""
    shards = sorted(OUT_DIR.glob("cross_render_fit_characters.shard__*.json"))
    if not shards:
        raise FileNotFoundError(f"no figure-2 shard files under {OUT_DIR}")
    merged: dict | None = None
    for p in shards:
        d = json.loads(p.read_text(encoding="utf-8"))
        if merged is None:
            merged = {kk: v for kk, v in d.items() if kk != "cells"} | {"cells": []}
        merged["cells"].extend(d["cells"])
    assert merged is not None
    seen = {(c["group"], c["model"], c["character"]) for c in merged["cells"]}
    expect = {(g, m, ch) for g in FIG2_GROUPS for m in MODELS for ch in CHARS}
    if seen != expect:
        raise AssertionError(
            f"figure-2 merged grid incomplete: missing {sorted(expect - seen)}, "
            f"unexpected {sorted(seen - expect)}"
        )
    merged["merged_from_shards"] = [p.name for p in shards]
    FIG2_OUT.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    _log(
        f"[fig2] merged {len(shards)} shards -> {FIG2_OUT.relative_to(_REPO)} "
        f"({len(merged['cells'])} rows, grid complete)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="one pair, one fold — measure the wall")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument(
        "--merge",
        action="store_true",
        help="fold cross_render_fit.shard__*.json into the canonical file and exit",
    )
    ap.add_argument(
        "--figure2",
        action="store_true",
        help="run the character-story paired fits (writeup Figure 2 paired-fit column)",
    )
    ap.add_argument(
        "--fig2-figures",
        nargs="*",
        default=["2a", "2b"],
        help="which figure-2 groups to run in this shard (2a / 2b)",
    )
    ap.add_argument(
        "--merge-figure2",
        action="store_true",
        help="fold cross_render_fit_characters.shard__*.json into the canonical file and exit",
    )
    args = ap.parse_args()

    if args.merge:
        return merge_shards()
    if args.merge_figure2:
        return merge_figure2()
    if args.figure2:
        return run_figure2(args)

    fold_map = _load_production_fold_map()
    k, fold_of = int(fold_map["k"]), fold_map["fold_of"]
    _log(
        f"[cross-render] fold map {fold_map['_source_ref']} k={k} seed={fold_map['seed']} "
        f"n_conv={fold_map['_n_conv']:,} variants={len(fold_map['_variants'])}"
    )

    results: dict = {
        "what": (
            "SOURCE-render context vector -> TARGET-render answer vector, ridge fit DIRECTLY on "
            "cross-render pairs. A predictability upper bound, NOT a shared-operator claim: this "
            "fit sees pairs and has full d^2 capacity, unlike any ladder rung."
        ),
        "source_form": SOURCE_FORM,
        "estimator": "issue2054_fits._ridge_gcv_fit_predict (GCV, dof_cap=0.9) — production parity",
        "d_ambient": D_AMBIENT,
        "identity_gate_tol": IDENTITY_TOL,
        "cells": [],
    }

    for cond in args.conditions:
        for model in args.models:
            _log(f"\n[cross-render] === {cond} / {model} ===")
            acts = {f: _load_cell(cond, f, model) for f in FORMS}
            # 4-way conv_id intersection -> ONE shared row set for every comparison.
            common = set(acts[FORMS[0]]["conv_ids"])
            for f in FORMS[1:]:
                common &= set(acts[f]["conv_ids"])
            common &= set(fold_of)
            order = sorted(common)
            n = len(order)
            folds = _fold_split(order, fold_of, k)
            n_train_typ = n - max(len(f) for f in folds)
            _log(
                f"  4-way intersection n={n:,}  typical n_train={n_train_typ:,} "
                f"vs d={D_AMBIENT:,} -> {'WELL-POSED' if n_train_typ > D_AMBIENT else 'UNDER-DETERMINED'}"
            )

            X_src = _aligned(acts[SOURCE_FORM], order, "v_C")
            only = 0 if args.pilot else None

            for tgt in FORMS:
                Y_tgt = _aligned(acts[tgt], order, "v_A")
                _log(f"    cross-render {SOURCE_FORM} ctx -> {tgt} ans:")
                cross = _run_fit(X_src, Y_tgt, folds, only_fold=only)
                _log(f"    matched within-cell {tgt} ctx -> {tgt} ans:")
                X_tgt = _aligned(acts[tgt], order, "v_C")
                within = _run_fit(X_tgt, Y_tgt, folds, only_fold=only)

                row = {
                    "condition": cond,
                    "model": model,
                    "target_form": tgt,
                    "is_identity": tgt == SOURCE_FORM,
                    "n_intersection": n,
                    "n_train_typical": n_train_typ,
                    "well_posed_ambient": bool(n_train_typ > D_AMBIENT),
                    "cross_render_r2": cross["pooled_r2"],
                    "cross_render_per_fold": cross["per_fold_r2"],
                    "matched_within_cell_r2": within["pooled_r2"],
                    "matched_within_cell_per_fold": within["per_fold_r2"],
                    "fraction_of_matched_ceiling": (
                        cross["pooled_r2"] / within["pooled_r2"]
                        if within["pooled_r2"] not in (0.0,) and np.isfinite(within["pooled_r2"])
                        else float("nan")
                    ),
                    "cross_fold_info": cross["fold_info"],
                    "within_fold_info": within["fold_info"],
                }
                # IDENTITY GATE: chat ctx -> chat ans IS the within-cell chat fit.
                if row["is_identity"]:
                    delta = abs(cross["pooled_r2"] - within["pooled_r2"])
                    row["identity_gate_delta"] = delta
                    if not (delta < IDENTITY_TOL):
                        raise AssertionError(
                            f"IDENTITY GATE FAILED for {cond}/{model}: cross-render chat->chat "
                            f"{cross['pooled_r2']:.12f} != within-cell chat {within['pooled_r2']:.12f} "
                            f"(delta {delta:.3g} >= {IDENTITY_TOL:g}) — row alignment is wrong"
                        )
                    _log(f"    IDENTITY GATE PASS (delta {delta:.3g})")
                else:
                    _log(
                        f"    -> cross={cross['pooled_r2']:+.4f}  matched-ceiling="
                        f"{within['pooled_r2']:+.4f}  ratio={row['fraction_of_matched_ceiling']:.3f}"
                    )
                results["cells"].append(row)

            if args.pilot:
                _log("\n[cross-render] PILOT complete — 1 fold per pair on one cell group.")
                return 0

    out = _shard_out(list(args.conditions), list(args.models))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    _log(f"\n[cross-render] wrote {out.relative_to(_REPO)} ({len(results['cells'])} rows)")
    return 0


def merge_shards() -> int:
    """Fold every cross_render_fit.shard__*.json into the canonical file.

    Fails loud on a missing cell: the merged file must carry the full
    conditions x models x targets grid, or a downstream plot would silently
    read a partial lattice.
    """
    shards = sorted(OUT_DIR.glob("cross_render_fit.shard__*.json"))
    if not shards:
        raise FileNotFoundError(f"no shard files under {OUT_DIR}")
    merged: dict | None = None
    for p in shards:
        d = json.loads(p.read_text(encoding="utf-8"))
        if merged is None:
            merged = {k: v for k, v in d.items() if k != "cells"} | {"cells": []}
        merged["cells"].extend(d["cells"])
    assert merged is not None
    seen = {(c["condition"], c["model"], c["target_form"]) for c in merged["cells"]}
    expect = {(c, m, f) for c in CONDITIONS for m in MODELS for f in FORMS}
    if seen != expect:
        raise AssertionError(
            f"merged shard grid incomplete: missing {sorted(expect - seen)}, "
            f"unexpected {sorted(seen - expect)}"
        )
    merged["merged_from_shards"] = [p.name for p in shards]
    OUT.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    _log(
        f"[cross-render] merged {len(shards)} shards -> {OUT.relative_to(_REPO)} "
        f"({len(merged['cells'])} rows, grid complete)"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(main())
