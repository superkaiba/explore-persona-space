#!/usr/bin/env python3
"""#1739 pvsynth eval-rung ARM SCORING (no judge, read-only DV inputs).

The pvsynth rung's DV dataset + capture store were produced by
``issue1739_pvsynth_pod.py`` (GPU leg) and ``issue1739_pvsynth_score.py``
(judge + DV leg). This entrypoint is the THIRD leg: it consumes those
COMMITTED artifacts read-only and scores the plan's transfer roster
(:data:`arms.TRANSFER_ARMS_WIDE` by default — the 6 core ladder arms plus the
fitted arms 5/7/8/12; ``--arms core`` reproduces the original 6-arm column)
on the pvsynth rung, per behavior x variant,
over all 28 layers, at the TRAIN-frozen layer.

It is deliberately a SEPARATE entrypoint from ``issue1739_fits.py``:
``_run_real`` couples the transfer leg to a full re-run of the 16-arm train
grid under one ``--labeled-store``/``--dv-json`` pair, and the pvsynth eval
table lives at a DIFFERENT store + DV path. Every piece of scoring math is
IMPORTED from the reviewed production modules — this file contains no fit,
no metric, and no fold logic of its own:

* ``issue1739_fits._load_labeled``      — store + DV -> per-context arrays
* ``issue1739_fits._extract_rb``        — E1 diff-of-means direction
* ``issue1739_fits._u_pool_for_spec``   — #1092 U-pool realization
* ``issue1739_fits._fit_map``           — context->answer map (refit in-process)
* ``fits.fit_whitening`` / ``apply_whitening`` / ``realize_budget_cell``
* ``arms.run_transfer_cell`` / ``run_cell`` / ``evaluate_transfer``
  / ``frozen_layer_idx`` / ``spearman_rows`` / ``write_summary``

HARD safety rails (this leg must never re-judge or clobber a committed input):

1. The judge is never called — no judge module may be imported, asserted at
   entry AND exit (:func:`_assert_no_judge_modules`).
2. Every DV input is read-only: its sha256 is recorded at load and
   RE-VERIFIED after scoring (:func:`_verify_input_shas`).
3. A git-TRACKED output path is refused unless ``--allow-overwrite-committed``,
   and the out-root must be a ``pvsynth`` subtree (:func:`_assert_outputs_safe`)
   so a mis-passed ``--out-root`` can never overwrite the main behavior dirs.

Stores are read IN PLACE (never copied) — the instance running this already
holds the staged train stores.

VM-side runs carry the shared-VM thread caps
(``OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2``); pod/GCE runs do not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Put the repo root on ``sys.path`` so ``scripts.*`` imports resolve.

    Script mode sets ``sys.path[0]`` to this file's dir (``scripts/``), NOT the
    repo root, so a ``from scripts.issue1739_fits import ...`` fails without
    this (gotchas.md § script-mode sys.path). The sentinel assert makes a wrong
    parent depth fail loud instead of silently inserting a bogus path.
    """
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_fits.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

RUNG = "pvsynth"
BEHAVIORS = ("evil", "sycophancy", "hallucination")
DEFAULT_OUT_ROOT = Path("eval_results/issue_1739/pvsynth")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")

# Written at the top of every per-behavior capture store by
# experiments/issue_1739/capture.py — the discriminator that tells a capture
# store apart from its parent root (see _resolve_pvsynth_store).
CAPTURE_MANIFEST_NAME = "_capture_manifest.json"

# Judge surfaces this leg must never touch. Checked against sys.modules at
# entry and exit — an accidental transitive import fails the run loudly
# rather than silently spending Batch-API budget.
FORBIDDEN_JUDGE_MODULES = (
    "explore_persona_space.eval.batch_judge",
    "explore_persona_space.eval.graded_judge",
    "explore_persona_space.eval.judge_dispatch",
    "explore_persona_space.eval.belief",
    "explore_persona_space.llm.api_dispatch",
)


# ---------------------------------------------------------------------------
# safety rails
# ---------------------------------------------------------------------------


def _assert_no_judge_modules(when: str) -> None:
    """Fail loud if any judge/API-dispatch module is imported (rail 1)."""
    hits = sorted(m for m in sys.modules if m in FORBIDDEN_JUDGE_MODULES)
    if hits:
        raise RuntimeError(
            f"judge surface imported {when}: {hits} — this leg scores arms on the "
            "COMMITTED pvsynth DV and must never (re-)judge"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_input_shas(shas: dict[str, str]) -> None:
    """Re-verify recorded input sha256s (rail 2: DV inputs are read-only)."""
    for p, want in sorted(shas.items()):
        got = _sha256(Path(p))
        if got != want:
            raise RuntimeError(
                f"read-only input MUTATED during the run: {p} sha256 {want} -> {got}"
            )


def _git_tracked(path: Path) -> bool:
    """True when ``path`` is tracked in git (soft False off-repo)."""
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(path)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    return proc.returncode == 0


def _assert_outputs_safe(paths: list[Path], *, out_root: Path, allow: bool) -> None:
    """Refuse to overwrite committed artifacts / escape the pvsynth subtree."""
    if out_root.resolve().name != RUNG:
        raise SystemExit(
            f"--out-root must be a '{RUNG}' subtree (got {out_root}) — refusing to write "
            "arm results outside the pvsynth rung's own dir"
        )
    tracked = [p for p in paths if _git_tracked(p)]
    if tracked and not allow:
        raise SystemExit(
            "refusing to overwrite git-TRACKED output(s): "
            + ", ".join(str(p) for p in tracked)
            + " (pass --allow-overwrite-committed to re-write them deliberately)"
        )


# ---------------------------------------------------------------------------
# frozen-layer selection
# ---------------------------------------------------------------------------


def modal_frozen_layers(
    summary_path: Path, *, variant: str, regime: str, u_rung_label: str
) -> dict[str, int]:
    """Modal TRAIN-frozen layer index per arm from a committed train summary.

    Reads ``arm_rows`` of the behavior's main ``all_arms_spearman.json``,
    keeps the plain-rung (``f_u is None``) rows for this (variant, regime,
    U rung), and takes the MODE over the grid's (budget, draw, seed) units of
    ``arms.frozen_layer_idx(row['rho_per_layer'])``. Ties break to the SMALLEST
    layer index (``Counter.most_common`` is insertion-ordered, so the rows are
    counted in sorted-index order first).
    """
    from explore_persona_space.experiments.issue_1739 import arms

    payload = json.loads(Path(summary_path).read_text())
    rows = [
        r
        for r in payload.get("arm_rows", [])
        if r.get("variant") == variant
        and r.get("regime") == regime
        and str(r.get("u_rung_label")) == str(u_rung_label)
        and r.get("f_u") is None
        and r.get("rho_per_layer")
    ]
    if not rows:
        raise RuntimeError(
            f"{summary_path}: no plain-rung arm_rows for variant={variant} "
            f"regime={regime} u_rung_label={u_rung_label}"
        )
    per_arm: dict[str, list[int]] = {}
    for r in rows:
        per_arm.setdefault(r["arm"], []).append(arms.frozen_layer_idx(r["rho_per_layer"]))
    out: dict[str, int] = {}
    for arm, idxs in per_arm.items():
        counts = Counter(sorted(idxs))
        out[arm] = int(counts.most_common(1)[0][0])
    return out


def own_pool_frozen_layers(
    data, cell, *, roster: list[str], device: str
) -> tuple[dict[str, int], dict[str, list[float]], dict[str, str]]:
    """Frozen layers from the behavior's OWN train-pool in-split OOF read.

    The fallback for a behavior with no committed train summary (sycophancy):
    run the SAME arm roster in-split on the train cell (``arms.run_cell`` —
    pooled OOF over the cell's group folds, never eval DV), Spearman per layer
    against the cell's DV, then ``frozen_layer_idx``. Identical selection rule
    to the main grid's, computed here instead of read from disk.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    scores, skipped = arms.run_cell(data, cell, arms=roster, device=device)
    dv_cell = np.asarray(data.dv[cell.row_idx], dtype=np.float64)
    frozen: dict[str, int] = {}
    rho_per_layer: dict[str, list[float]] = {}
    for slug, sc in sorted(scores.items()):
        rhos = [float(x) for x in arms.spearman_rows(np.asarray(sc, dtype=np.float64), dv_cell)]
        rho_per_layer[slug] = rhos
        frozen[slug] = arms.frozen_layer_idx(rhos)
    return frozen, rho_per_layer, skipped


#: Trailing token of a pvsynth ``group_key`` -> the elicitation polarity.
#: The rung's contexts come from the persona-vectors contrastive pairs, so
#: every group_key is ``pvsynth-p{k}-{pos,neg}``: ``pos`` = the ELICITING
#: instruction of pair k, ``neg`` = the non-eliciting one.
POLARITY_SUFFIXES = ("pos", "neg")


def polarity_labels(group_keys, *, behavior: str, variant: str) -> list[str]:
    """Per-context elicitation polarity parsed from the pvsynth ``group_key``.

    Returns one of :data:`POLARITY_SUFFIXES` per context, aligned with the
    labeled table's context order. FAILS LOUD on any group_key that does not
    end in a recognized polarity suffix: the pvsynth DV is generated by this
    project's own pipeline against a fixed grammar, so an unrecognized shape
    means the rung's construction changed underneath the split — silently
    dropping those contexts (or silently pooling them) would misreport which
    elicitation regime a row summarizes.
    """
    keys = [str(g) for g in group_keys]
    bad = sorted({k for k in keys if k.rsplit("-", 1)[-1] not in POLARITY_SUFFIXES})
    if bad:
        raise RuntimeError(
            f"[{behavior}/{variant}] pvsynth group_key must end in "
            f"{'/'.join(POLARITY_SUFFIXES)} (grammar 'pvsynth-p<k>-<polarity>'); "
            f"{len(bad)} unrecognized key(s), first offenders: {bad[:5]}"
        )
    return [k.rsplit("-", 1)[-1] for k in keys]


# ---------------------------------------------------------------------------
# per-behavior scoring
# ---------------------------------------------------------------------------


def _resolve_pvsynth_store(args: argparse.Namespace, behavior: str) -> Path:
    """Per-behavior pvsynth capture store, tolerating root- and sibling-dir overrides.

    The store is uploaded as ``<hf_prefix>/capture_store/<behavior>/``, so the
    correct input is always ONE behavior's dir. Four ways to name it, checked
    in order:

    1. ``--pvsynth-store-root <dir>``  -> ``<dir>/<behavior>`` (the root form).
    2. ``--pvsynth-store <dir>`` where ``<dir>/<behavior>`` is itself a capture
       store -> that CHILD. A caller that resolved the mirrored ``capture_store``
       ROOT (e.g. by ``dirname`` of the FLAT manifest that sits beside the
       per-behavior subtrees) then gets each behavior's own dir instead of one
       shared root for all of them.
    3. ``--pvsynth-store <dir>`` where a SIBLING named ``<behavior>`` exists
       next to it -> that sibling. This makes a caller that resolved ONE
       behavior's dir (e.g. by finding the first ``_capture_manifest.json``
       under a mirrored ``capture_store/`` tree) correct for EVERY behavior
       instead of silently scoring them all against one behavior's
       activations. Only fires when the sibling actually exists, so a
       deliberate single-behavior override is untouched.
    4. ``--pvsynth-store <dir>`` verbatim, else the ``--store-root`` default.

    Cases 2 and 3 resolve ONLY against a real capture store — a dir carrying
    ``_capture_manifest.json`` (case 3 also accepts a bare dir, the shape it
    shipped with). A path that is neither a behavior subtree nor a root of
    them resolves verbatim, so the multi-behavior override guard below still
    refuses it rather than scoring every behavior against one store.
    """
    if args.pvsynth_store_root is not None:
        return args.pvsynth_store_root / behavior
    if args.pvsynth_store is not None:
        given = Path(args.pvsynth_store)
        if given.name != behavior:
            child = given / behavior
            if (child / CAPTURE_MANIFEST_NAME).is_file():
                return child
            sibling = given.parent / behavior
            if (sibling / CAPTURE_MANIFEST_NAME).is_file() or sibling.is_dir():
                return sibling
        return given
    return args.store_root / "pvsynth_capture_store" / behavior


def _behavior_paths(args: argparse.Namespace, behavior: str) -> dict[str, Path]:
    """Resolve every input path for one behavior (flags override the defaults).

    Every default is PER-BEHAVIOR. The pvsynth capture store is uploaded as
    ``<hf_prefix>/capture_store/<behavior>/`` (one subdir per behavior), so a
    SHARED store dir would silently score one behavior's DV against another
    behavior's activations; the train DV mirrors
    ``<train-dv-root>/<behavior>/labeling.json``, which fits both the committed
    tree and the staged HF ``judge/dv_dataset`` tree. The single-path overrides
    are guarded to a one-behavior run (:func:`parse_args`).
    """
    return {
        "train_store": args.train_store or args.store_root / f"{behavior}_labeling",
        "train_dv": args.train_dv_json or args.train_dv_root / behavior / "labeling.json",
        "e1_store": args.e1_store or args.store_root / f"{behavior}_extraction",
        "pvsynth_store": _resolve_pvsynth_store(args, behavior),
        "pvsynth_dv": args.pvsynth_dv_json
        or args.out_root / "dv_dataset" / behavior / "labeling.json",
        "train_summary": args.train_summary
        or args.main_root / behavior / "arm_results" / "all_arms_spearman.json",
    }


def _missing_inputs(paths: dict[str, Path]) -> list[str]:
    """Required inputs that are absent (train_summary is OPTIONAL — fallback)."""
    required = ("train_store", "train_dv", "e1_store", "pvsynth_store", "pvsynth_dv")
    return [f"{k}={paths[k]}" for k in required if not paths[k].exists()]


def _rb_for_behavior(args, behavior: str, tbl, layers, dim, paths):
    """E1 direction: the persisted fp16 bank when resolvable, else re-extract.

    ``--rb-source bank`` reuses the pinned ``r_b_e1/{behavior}.npz`` analysis
    tensor (fp16 — a rank-correlation-safe precision loss, recorded in meta);
    ``extract`` recomputes the exact fp64 diff-of-means from the E1 extraction
    store. ``auto`` prefers the bank and falls back to extraction (the
    sycophancy case: no bank exists).
    """
    import numpy as np

    from scripts.issue1739_fits import _extract_rb

    bank = args.tensors_root / "r_b_e1" / f"{behavior}.npz"
    if args.rb_source in ("auto", "bank") and bank.exists():
        with np.load(bank, allow_pickle=False) as z:
            rb = np.asarray(z["rb"], dtype=np.float64)
            bank_layers = [int(x) for x in z["layers"]]
        if bank_layers != list(layers):
            raise RuntimeError(
                f"{bank}: layers {bank_layers[:4]}...(n={len(bank_layers)}) != requested "
                f"{list(layers)[:4]}...(n={len(layers)})"
            )
        if rb.shape[-1] != dim:
            raise RuntimeError(f"{bank}: hidden dim {rb.shape[-1]} != store dim {dim}")
        return rb, {"rb_source": "bank", "rb_path": str(bank), "rb_dtype_on_disk": "float16"}
    if args.rb_source == "bank":
        raise SystemExit(f"--rb-source bank requested but {bank} is absent")
    ns = argparse.Namespace(e1_store=paths["e1_store"], behavior=behavior)
    rb = _extract_rb("e1", ns, tbl, layers, dim)
    return rb, {"rb_source": "extract", "rb_path": str(paths["e1_store"])}


def score_behavior(args, behavior: str) -> dict:  # noqa: C901 — one linear per-behavior pipeline
    """Score the pvsynth rung for one behavior; returns its result payload."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits, store_io
    from scripts.issue1739_fits import (
        RunSpec,
        _fit_map,
        _load_labeled,
        _u_pool_for_spec,
    )

    paths = _behavior_paths(args, behavior)
    missing = _missing_inputs(paths)
    if missing:
        raise FileNotFoundError(
            f"[{behavior}] missing input(s): {'; '.join(missing)} — stage the train/pvsynth "
            "stores in place and commit/stage the DV labeling.json before scoring"
        )
    layers = args.layers or list(range(args.n_layers))
    shas = {str(paths["train_dv"]): _sha256(paths["train_dv"])}
    shas[str(paths["pvsynth_dv"])] = _sha256(paths["pvsynth_dv"])

    t_load = time.time()
    tbl = _load_labeled(
        paths["train_store"], paths["train_dv"], layers, config="config_a", need_rollout_rows=False
    )
    tbl_ev = _load_labeled(
        paths["pvsynth_store"],
        paths["pvsynth_dv"],
        layers,
        config="config_b",  # CONFIG_SPLIT['config_b'] == the 'eval' split (pvsynth rows)
        need_rollout_rows=False,
    )
    if set(tbl_ev.rungs) != {RUNG}:
        raise RuntimeError(
            f"[{behavior}] pvsynth DV must carry rung={RUNG!r} on every row; got {tbl_ev.rungs} "
            f"from {paths['pvsynth_dv']} (wrong DV dataset?)"
        )
    dim = tbl.z_ans.shape[-1]
    if tbl_ev.z_ans.shape[-1] != dim:
        raise RuntimeError(
            f"[{behavior}] hidden dim mismatch: train {dim} vs pvsynth {tbl_ev.z_ans.shape[-1]}"
        )
    print(
        f"[pvsynth-arms] {behavior}: train n={len(tbl.ctx_order)} groups={len(set(tbl.groups))} | "
        f"{RUNG} n={len(tbl_ev.ctx_order)} | load={time.time() - t_load:.0f}s",
        flush=True,
    )

    rb, rb_meta = _rb_for_behavior(args, behavior, tbl, layers, dim, paths)

    store_io.stage_u_store(Path(args.u_store), ("prefix_end", "context_end", "t1"), tuple(layers))
    u_arrays, u_meta = store_io.load_summaries(
        args.u_store, ("prefix_end", "context_end", "t1"), tuple(layers), hidden_dim=dim
    )
    u_fit_rows = np.flatnonzero(store_io.fit_pool_mask(u_meta))

    budget_l = args.budget or len(tbl.ctx_order)
    roster = arms.resolve_transfer_roster(args.arms)
    n_boot = int(args.n_boot) if args.n_boot else arms.N_BOOT
    ckpt = args.out_root / behavior / "percell" / "pvsynth_transfer.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ckpt.exists():
        with ckpt.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["unit_key"]] = rec

    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []
    pol_rows_all: list[dict] = []
    pol_skips_all: list[dict] = []
    frozen_source: dict[str, str] = {}
    map_diag: dict[str, dict] = {}
    variants = list(args.variants)
    t0 = time.time()
    for vi, variant in enumerate(variants):
        unit_key = json.dumps(
            {
                "behavior": behavior,
                "variant": variant,
                "regime": args.regime,
                "u_rung": args.u_size,
                "budget_l": budget_l,
                "draw": args.draw,
                "seed": args.seed,
                "rung": RUNG,
                "arms": sorted(roster),
                "layers": [int(x) for x in layers],
                "n_eval": len(tbl_ev.ctx_order),
                "n_boot": n_boot,
                "min_n": int(args.min_n),
                "map_kind": args.map_kind,
                "rb_source": rb_meta["rb_source"],
                # Bumped when a unit's OUTPUT SET grows, so a checkpoint written
                # before the polarity split can never resume as "done" and
                # silently contribute a unit with no polarity rows.
                "outputs": "rows+skips+per_layer+polarity+preds",
            },
            sort_keys=True,
        )
        if unit_key in done:
            rec = done[unit_key]
            rows_all += rec["rows"]
            skips_all += rec.get("skips", [])
            per_layer_all += rec.get("per_layer", [])
            pol_rows_all += rec.get("polarity_rows", [])
            pol_skips_all += rec.get("polarity_skips", [])
            frozen_source[variant] = rec.get("frozen_source", "resume")
            print(
                f"[pvsynth-arms] {behavior} unit {vi + 1}/{len(variants)} SKIP (resume) {variant}",
                flush=True,
            )
            continue

        spec = RunSpec(
            variant=variant,
            regime=args.regime,
            u_size=None if str(args.u_size).lower() == "full" else int(args.u_size),
            budgets=(budget_l,),
            draws=(args.draw,),
            seeds=(args.seed,),
            f_u=None,
            f_l=None,
        )
        u_x, u_y, u_label, n_u = _u_pool_for_spec(spec, u_arrays, u_fit_rows, tbl, layers)
        t_map = time.time()
        wh = fits.fit_whitening(u_x, device=args.device, seed=args.seed)
        # Map REFIT in-process on the #1092 U pool, exactly as the main run
        # does — never the uploaded map payload (the brief's explicit call).
        mapfit = _fit_map(args, fits.apply_whitening(u_x, wh), fits.apply_whitening(u_y, wh))
        map_s = time.time() - t_map
        del u_x, u_y
        map_diag[f"{variant}|{u_label}"] = {
            **mapfit.diagnostics,
            "map_source": "refit",
            "map_fit_s": round(map_s, 1),
            "n_u": int(n_u),
        }

        z_tr_w = fits.apply_whitening(tbl.z_by_variant[variant], wh)
        za_tr_w = fits.apply_whitening(tbl.z_ans, wh)
        z_ev_w = fits.apply_whitening(tbl_ev.z_by_variant[variant], wh)
        za_ev_w = fits.apply_whitening(tbl_ev.z_ans, wh)
        data = arms.CellData(
            z_ctx=z_tr_w,
            z_ans=za_tr_w,
            dv=tbl.dv,
            rb=np.einsum("ld,lde->le", rb, wh.w),
            mapfit=mapfit,
            layers=tuple(layers),
        )
        cell = fits.realize_budget_cell(
            tbl.groups, budget_l=budget_l, draw=args.draw, seed=args.seed
        )
        prov = {
            "behavior": behavior,
            "variant": variant,
            "regime": args.regime,
            "u_rung": int(n_u),
            "u_rung_label": u_label,
            "eval_rung": RUNG,
            "config": "config_a",
            "f_u": None,
            "f_l": None,
        }

        summary = paths["train_summary"]
        own_rho: dict[str, list[float]] = {}
        if summary.exists() and not args.force_own_pool_frozen:
            frozen_by_arm = modal_frozen_layers(
                summary, variant=variant, regime=args.regime, u_rung_label=u_label
            )
            frozen_by_arm = {a: i for a, i in frozen_by_arm.items() if a in roster}
            src = f"modal-committed-train-cells:{summary}"
            shas[str(summary)] = _sha256(summary)
        else:
            frozen_by_arm, own_rho, own_skips = own_pool_frozen_layers(
                data, cell, roster=roster, device=args.device
            )
            src = "own-train-pool-selection"
            skips_all += [
                {"arm": a, "reason": f"own-pool frozen-layer read: {r}", "variant": variant}
                for a, r in sorted(own_skips.items())
            ]
        frozen_source[variant] = src
        missing_frozen = sorted(set(roster) - set(frozen_by_arm))
        if missing_frozen:
            raise RuntimeError(
                f"[{behavior}/{variant}] no frozen layer for {missing_frozen} "
                f"(source: {src}) — cannot score at a TRAIN-frozen layer"
            )

        t_tf = time.time()
        scores_ev, arm_skips = arms.run_transfer_cell(
            data,
            cell,
            z_ev_w,
            np.asarray(tbl_ev.dv, dtype=np.float64),
            za_ev=za_ev_w,
            arms=roster,
            device=args.device,
            ridge_folds=(0,),  # the reverse (train-block) fold is discarded
        )
        rows_u, skips_u = arms.evaluate_transfer(
            scores_ev,
            tbl_ev.dv,
            np.asarray(tbl_ev.row_rungs),
            frozen_by_arm,
            provenance=prov,
            cell=cell,
            layers=tuple(layers),
            n_boot=n_boot,
            min_n=int(args.min_n),
        )
        skips_u += [
            {"arm": slug, "reason": reason, "variant": variant}
            for slug, reason in sorted(arm_skips.items())
        ]
        skips_u += arms.roster_accounting_skips(roster, scores_ev, arm_skips, variant=variant)
        dv_ev = np.asarray(tbl_ev.dv, dtype=np.float64)

        # ---- polarity split (elicit / non-elicit / pooled) --------------------
        # The pvsynth rung is built from the persona-vectors contrastive pairs,
        # so every context carries a POLARITY in its group_key
        # (``pvsynth-p{k}-{pos,neg}``): the ``pos`` half is the ELICITING
        # instruction, the ``neg`` half the non-eliciting one. Pooling them
        # averages a predictor's behavior over two deliberately opposite
        # elicitation regimes, which can hide a polarity-specific read. Each
        # subset goes through the SAME reviewed ``evaluate_transfer`` over a
        # masked slice (a slice, not a row loop), so rho / CI / min_n / frozen
        # layer semantics are identical across subsets and to the pooled read.
        #
        # These rows land in their OWN top-level section
        # (``transfer_polarity_rows``); ``transfer_rows`` keeps carrying the
        # pooled read alone, unchanged, so a consumer reading it still sees
        # exactly one row per (arm, eval rung).
        polarity = polarity_labels(tbl_ev.groups, behavior=behavior, variant=variant)
        pol_arr = np.asarray(polarity)
        pol_masks = {
            "pooled": np.ones(pol_arr.size, dtype=bool),
            "elicit": pol_arr == "pos",
            "non_elicit": pol_arr == "neg",
        }
        rungs_arr = np.asarray(tbl_ev.row_rungs)
        pol_rows_u: list[dict] = []
        pol_skips_u: list[dict] = []
        for subset, mask in pol_masks.items():
            if int(mask.sum()) < int(args.min_n):
                pol_skips_u.append(
                    {
                        "arm": "*",
                        "variant": variant,
                        "polarity_subset": subset,
                        "n_eval": int(mask.sum()),
                        "reason": f"polarity subset below min_n ({int(mask.sum())} < {args.min_n})",
                    }
                )
                continue
            sub_rows, sub_skips = arms.evaluate_transfer(
                {slug: np.asarray(sc, dtype=np.float64)[:, mask] for slug, sc in scores_ev.items()},
                dv_ev[mask],
                rungs_arr[mask],
                frozen_by_arm,
                provenance={**prov, "polarity_subset": subset},
                cell=cell,
                layers=tuple(layers),
                n_boot=n_boot,
                min_n=int(args.min_n),
            )
            pol_rows_u += sub_rows
            pol_skips_u += [{**s, "variant": variant, "polarity_subset": subset} for s in sub_skips]

        # Per-context frozen-layer predictions: the durable subset-re-analysis
        # input, carrying the polarity label so ANY later per-polarity /
        # per-pair read is a pure re-analysis instead of another re-score.
        arms.write_preds_jsonl(
            args.out_root / behavior / "preds" / f"pvsynth_preds.{variant}.jsonl",
            arms.transfer_preds_rows(
                scores_ev,
                dv_ev,
                tbl_ev.ctx_order,
                frozen_by_arm,
                provenance={**prov, "n_eval_pooled": int(dv_ev.size)},
                layers=tuple(layers),
                labels={"polarity": polarity, "group_key": list(tbl_ev.groups)},
            ),
        )

        # Per-layer rho over ALL layers (the frozen-layer row above is the
        # selection-clean headline; this is the full-profile companion).
        per_layer_u: list[dict] = []
        for slug, sc in sorted(scores_ev.items()):
            rhos = arms.spearman_rows(np.asarray(sc, dtype=np.float64), dv_ev)
            per_layer_u.append(
                {
                    **prov,
                    "arm": slug,
                    "family": arms.ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
                    "rung_kind": "eval_transfer_per_layer",
                    "layers": [int(x) for x in layers],
                    "rho_per_layer": [float(x) for x in rhos],
                    "frozen_layer_idx": int(frozen_by_arm[slug]),
                    "frozen_layer": int(layers[int(frozen_by_arm[slug])]),
                    "frozen_source": src,
                    "rho_per_layer_train_own_pool": own_rho.get(slug),
                    "n_eval": int(dv_ev.size),
                    "budget_l": budget_l,
                    "draw": args.draw,
                    "seed": args.seed,
                }
            )

        line = json.dumps(
            {
                "unit_key": unit_key,
                "rows": rows_u,
                "skips": skips_u,
                "per_layer": per_layer_u,
                "polarity_rows": pol_rows_u,
                "polarity_skips": pol_skips_u,
                "frozen_source": src,
            },
            sort_keys=True,
        )
        with ckpt.open("a", encoding="utf-8") as fh:  # single-line O_APPEND write
            fh.write(line + "\n")
            fh.flush()
        rows_all += rows_u
        skips_all += skips_u
        per_layer_all += per_layer_u
        pol_rows_all += pol_rows_u
        pol_skips_all += pol_skips_u
        print(
            f"[pvsynth-arms] {behavior} unit {vi + 1}/{len(variants)} {variant} "
            f"arms={len(scores_ev)} rows={len(rows_u)} pol_rows={len(pol_rows_u)} "
            f"map={map_s:.0f}s "
            f"transfer={time.time() - t_tf:.0f}s elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        del data, z_tr_w, za_tr_w, z_ev_w, za_ev_w, wh, mapfit
        if str(args.device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    _verify_input_shas(shas)
    return {
        "behavior": behavior,
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "polarity_rows": pol_rows_all,
        "polarity_skips": pol_skips_all,
        "frozen_source": frozen_source,
        "map_diagnostics": map_diag,
        "input_sha256": shas,
        "input_paths": {k: str(v) for k, v in paths.items()},
        "rb": rb_meta,
        "n_train_contexts": len(tbl.ctx_order),
        "n_eval_contexts": len(tbl_ev.ctx_order),
        "budget_l": budget_l,
        "wall_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# DV-construct metadata (carried into every output)
# ---------------------------------------------------------------------------


def dv_construct_meta(behavior: str) -> dict:
    """Per-behavior DV construct + the standing pvsynth caveats.

    The hallucination rung is a STATED DEVIATION: the Persona Vectors eval
    questions carry no reference answers, so its DV is the trait ``eval_prompt``
    rubric rather than a three-way fabrication rate — and its content-drop rate
    (23.4%) sits above the rule-23 ~10% threshold with the max_tokens=800
    re-judge recovering only ~48%, so the DV is PROVISIONAL.
    """
    base = {
        "dv_construct": "pv_trait_rubric_graded_0_100",
        "dv_recipe": (
            "K=5 vLLM rollouts per context; judge claude-sonnet-4-5-20250929, N=3 draws @ "
            "temp 1.0, max_tokens 400 (fully-dropped items re-judged whole at 800 against a "
            "fresh cache — mixed-instrument, disclosed); per-context mean of kept per-rollout "
            "scores; drop-never-coerce"
        ),
        "provisional": False,
        "caveats": [],
    }
    if behavior == "hallucination":
        base["provisional"] = True
        base["caveats"] = [
            "STATED DEVIATION: trait eval_prompt rubric (PV eval questions carry no reference "
            "answers), not a three-way fabrication-rate rubric",
            "701/3000 content drops (23.4%) — above the llm-judging rule-23 ~10% threshold; the "
            "max_tokens=800 re-judge recovered only ~48%, so the residue is judge declines, not "
            "truncation. DV is PROVISIONAL.",
        ]
    return base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["context_end", "prefix_end"],
        choices=["context_end", "prefix_end"],
    )
    ap.add_argument("--regime", default="e1", choices=("e1", "e2", "e2p"))
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        metavar="ROSTER|SLUG",
        help="transfer roster: 'wide' (default — the 6 core ladder arms plus the fitted "
        "arms 5/7/8/12), 'core' (the original 6, reproduces the committed column exactly), "
        "or an explicit arm-slug list",
    )
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all --n-layers")
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--u-size", default="full", help="U-pool rung: int or 'full'")
    ap.add_argument("--budget", type=int, default=None, help="train rows (default: whole table)")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=3, help="per-rung row floor for a Spearman read")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--map-kind", default="linear", choices=("linear", "mlp", "kernel"))
    ap.add_argument("--mlp-map-width", type=int, default=None)
    ap.add_argument("--krr-map-centers", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--rb-source",
        default="auto",
        choices=("auto", "bank", "extract"),
        help="E1 direction: persisted fp16 bank, re-extract from the E1 store, or auto",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument(
        "--store-root",
        type=Path,
        default=DEFAULT_STORE_ROOT,
        help="dir holding the staged {behavior}_labeling / _extraction / pvsynth stores",
    )
    ap.add_argument("--train-store", type=Path, default=None)
    ap.add_argument(
        "--train-dv-root",
        type=Path,
        default=None,
        help="dir holding <behavior>/labeling.json (default: <--main-root>/dv_dataset; point at a "
        "staged copy of the HF issue1739_ctxmap/judge/dv_dataset tree for behaviors whose DV is "
        "not committed)",
    )
    ap.add_argument("--train-dv-json", type=Path, default=None)
    ap.add_argument("--e1-store", type=Path, default=None)
    ap.add_argument("--pvsynth-store", type=Path, default=None)
    ap.add_argument(
        "--pvsynth-store-root",
        type=Path,
        default=None,
        help="dir holding <behavior>/ pvsynth capture stores (a mirrored capture_store/ tree)",
    )
    ap.add_argument("--pvsynth-dv-json", type=Path, default=None)
    ap.add_argument("--train-summary", type=Path, default=None)
    ap.add_argument(
        "--u-store",
        type=Path,
        default=None,
        help="#1092 U-pool store (default: <--store-root>/u_store, staged on demand)",
    )
    ap.add_argument(
        "--force-own-pool-frozen",
        action="store_true",
        help="ignore committed train summaries; select frozen layers on each behavior's own pool",
    )
    ap.add_argument("--allow-overwrite-committed", action="store_true")
    ap.add_argument(
        "--import-check", action="store_true", help="resolve deferred imports and exit 0"
    )
    args = ap.parse_args(argv)
    if len(set(args.behaviors)) != len(args.behaviors):
        ap.error("--behaviors must be unique")
    if len(set(args.variants)) != len(args.variants):
        ap.error("--variants must be unique")
    # Resolve AFTER parsing so --store-root moves the U pool too (a constant
    # default would stage ~13 GB onto whichever disk DEFAULT_STORE_ROOT names,
    # not the one the caller pointed every other store at).
    if args.u_store is None:
        args.u_store = args.store_root / "u_store"
    if args.train_dv_root is None:
        args.train_dv_root = args.main_root / "dv_dataset"
    # A single-path override applied across several behaviors would score one
    # behavior's inputs against another's — e.g. --train-dv-json for a 3-behavior
    # run. Use the per-behavior ROOT flags (--store-root / --train-dv-root) or
    # run one behavior at a time.
    single_only = (
        "train_store",
        "train_dv_json",
        "e1_store",
        "pvsynth_dv_json",
        "train_summary",
    )
    if len(args.behaviors) > 1:
        set_flags = [
            f"--{f.replace('_', '-')}" for f in single_only if getattr(args, f) is not None
        ]
        # --pvsynth-store is exempt WHEN it root- or sibling-resolves for every
        # behavior (see _resolve_pvsynth_store): each behavior then gets its OWN
        # store. A path that resolves verbatim for several behaviors would share
        # one store across them, so it still lands in set_flags below.
        if args.pvsynth_store is not None and args.pvsynth_store_root is None:
            resolved = {b: _resolve_pvsynth_store(args, b) for b in args.behaviors}
            if len(set(resolved.values())) != len(args.behaviors) or not all(
                p.is_dir() for p in resolved.values()
            ):
                set_flags.append("--pvsynth-store")
        if set_flags:
            ap.error(
                f"{', '.join(set_flags)} name ONE behavior's input but --behaviors has "
                f"{len(args.behaviors)} ({', '.join(args.behaviors)}); use the per-behavior roots "
                "(--store-root / --train-dv-root / --main-root / --out-root) or one behavior per run"
            )
    return args


def _env_versions() -> dict[str, str]:
    import numpy

    out = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError as exc:  # torch is optional on a pure-CPU numpy path
        out["torch"] = f"unavailable ({exc.__class__.__name__})"
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _assert_no_judge_modules("at entry")

    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
            arms,
            fits,
            store_io,
        )
        from scripts.issue1739_fits import (  # noqa: F401
            RunSpec,
            _extract_rb,
            _fit_map,
            _load_labeled,
            _u_pool_for_spec,
            arrays_dim,
        )

        _assert_no_judge_modules("after --import-check imports")
        print("[pvsynth-arms] import-check OK", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.experiments.issue_1739 import arms
    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_fits import _git_commit

    load_dotenv()  # HF token for the U-pool staging leg

    out_paths = [args.out_root / b / "all_arms_spearman.json" for b in args.behaviors]
    out_paths += [
        args.out_root / b / "preds" / f"pvsynth_preds.{v}.jsonl"
        for b in args.behaviors
        for v in args.variants
    ]
    _assert_outputs_safe(out_paths, out_root=args.out_root, allow=args.allow_overwrite_committed)

    commit = _git_commit()
    env = _env_versions()
    failures: list[dict] = []
    t_all = time.time()
    for behavior in args.behaviors:
        try:
            res = score_behavior(args, behavior)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            # Per-behavior isolation: a missing/incoherent input for ONE behavior
            # must not discard the others' completed results. Recorded loudly
            # here AND surfaced by the nonzero exit below.
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[pvsynth-arms] {behavior} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        out = args.out_root / behavior / "all_arms_spearman.json"
        arms.write_summary(
            [],  # no in-split cell records: this leg emits ONLY the pvsynth rung
            out,
            meta={
                "mode": "pvsynth_transfer",
                "behavior": behavior,
                "rung": RUNG,
                "config": "config_a",
                "regimes": [args.regime],
                "variants": list(args.variants),
                "arms": sorted(arms.resolve_transfer_roster(args.arms)),
                "polarity_subsets": ["pooled", "elicit", "non_elicit"],
                "layers": [int(x) for x in (args.layers or list(range(args.n_layers)))],
                "n_contexts": res["n_eval_contexts"],
                "n_train_contexts": res["n_train_contexts"],
                "budget_l": res["budget_l"],
                "draw": args.draw,
                "seed": args.seed,
                "u_sizes": [args.u_size],
                "map_kind": args.map_kind,
                "map_source": "refit-in-process",
                "eval_rungs": [RUNG],
                "transfer_min_n": int(args.min_n),
                "transfer_eval_rungs": [RUNG],
                "frozen_layer_source": res["frozen_source"],
                "rb": res["rb"],
                "dv": dv_construct_meta(behavior),
                "input_paths": res["input_paths"],
                "input_sha256": res["input_sha256"],
                "git_commit": commit,
                "env_versions": env,
                "wall_s": res["wall_s"],
                "judge_called": False,
            },
            extra={
                "transfer_rows": res["rows"],
                "transfer_skips": res["skips"],
                "per_layer_rows": res["per_layer"],
                "n_transfer_rows": len(res["rows"]),
                "n_per_layer_rows": len(res["per_layer"]),
                # Separate section on purpose: transfer_rows stays exactly one
                # row per (arm, eval rung) — the pooled read — so a consumer
                # that predates the polarity split is unaffected. The
                # three-way elicit / non_elicit / pooled comparison is
                # self-contained HERE, discriminated by polarity_subset.
                "transfer_polarity_rows": res["polarity_rows"],
                "transfer_polarity_skips": res["polarity_skips"],
                "n_transfer_polarity_rows": len(res["polarity_rows"]),
            },
        )
        (args.out_root / behavior / "map_diagnostics.json").write_text(
            json.dumps(res["map_diagnostics"], indent=1)
        )
        print(
            f"[pvsynth-arms] {behavior} done: {len(res['rows'])} transfer rows, "
            f"{len(res['per_layer'])} per-layer rows -> {out}",
            flush=True,
        )

    _assert_no_judge_modules("at exit")
    print(
        f"[pvsynth-arms] all done in {time.time() - t_all:.0f}s "
        f"({len(args.behaviors) - len(failures)}/{len(args.behaviors)} behaviors)",
        flush=True,
    )
    if failures:
        args.out_root.mkdir(parents=True, exist_ok=True)  # every behavior may have failed early
        (args.out_root / "pvsynth_arms_failures.json").write_text(json.dumps(failures, indent=1))
        for f in failures:
            print(f"[pvsynth-arms] FAILED {f['behavior']}: {f['error']}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(2)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
