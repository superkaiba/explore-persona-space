#!/usr/bin/env python3
"""#1739 Result 5 / Result 3-R^2: composition-ladder wrapper around the fits CLI.

Result 5 (generic-only vs trait-augmented at a matched pool size) is the two
ENDPOINTS of the ladder Result 3's R^2 half asks for (how much trait-eliciting
data enters the map pool), so one ladder run delivers both.

This is a thin wrapper that owns three deviations from the fits CLI's default
composition grid and touches no file another session owns:

1. **Authorized ladder.** Overrides ``COMPOSITION_F_U`` to
   ``(0.0, 0.25, 0.5, 1.0)`` and ``COMPOSITION_F_L`` to ``(1.0,)``. The
   constants are imported FUNCTION-LOCALLY inside ``_run_real`` / ``_run_pilot``
   (``issue1739_fits.py``), so an override applied before ``main()`` resolves at
   call time.

2. **Suppresses the unauthorized third config.** The stock grid
   ``COMPOSITION_F_U = (0.0, 0.5)`` x ``COMPOSITION_F_L = (0.0, 1.0)`` crossed
   under the dedup key ``(f_u, f_l if f_u > 0 else 0.0)`` yields THREE keys --
   ``(0.0, 0.0)``, ``(0.5, 1.0)`` and ``(0.5, 0.0)``. The last is a
   trait-augmented arm drawing its eliciting contexts from OUTSIDE the anchor
   cell: legitimate, but not what this round authorized, and +compute. With
   ``f_l_grid = (1.0,)`` the ladder yields EXACTLY ``(0.0, 0.0)``,
   ``(0.25, 1.0)``, ``(0.5, 1.0)``, ``(1.0, 1.0)`` -- ``f_u = 0.0`` still keys
   to ``(0.0, 0.0)`` because the dedup key zeroes ``f_l`` when ``f_u == 0``.

3. **Drops the plain (non-compose) U-ladder rungs.** The generic-only baseline
   IS the ``f_u = 0.0`` rung of the ladder, composed at the same matched pool
   size as every other rung, so a plain rung would be duplicate compute on a
   pool that is not size-matched to the swap arms.

4. **Clamps the matched pool size to what the behavior can actually supply.**
   ``--compose-u-size`` is int-typed and RAISES rather than clamping, and the
   raise lands on ``_record_compose_skip``, which records a skip and CONTINUES
   for a compose spec -- so an over-large size silently DROPS ladder rungs from
   an otherwise rc=0 run. The eliciting-side requirement is
   ``f_u * size <= n_contexts``, so the feasible pool is
   ``size <= n_contexts / max(f_u)``. The wrapper therefore treats
   ``--compose-u-size`` as a CEILING and clamps it, once per process, to
   ``min(n_generic_fit_rows, n_contexts / max_f_u)`` read from the loaded data
   -- so the size is MEASURED, never assumed. ``max_f_u`` is the ROUND's
   largest rung (``EPM_R5_MAX_FU``), NOT the running leg's own subset: a gate
   leg running only ``f_u=0.0`` has no eliciting requirement and would
   otherwise compute a larger cap than its ``f_u=0.5`` sibling, putting the
   two configs of one behaviour at different pool sizes -- the exact
   size/composition confound the matched pool exists to prevent.
   Measured on evil at max_f_u=0.5: n_generic=18,793, n_contexts=6,468 ->
   cap 12,936. (18,793 is infeasible for evil at ANY f_u>0.36; the scope
   doc's "all three clear 18,793" counted labeled ROWS, not distinct
   CONTEXTS.) The pool size stays CONSTANT WITHIN a
   behavior across all four rungs, which is the property that keeps composition
   and size unconfounded; it is not equal ACROSS behaviors, which the swap
   comparison does not require. The realized size rides the cell label
   (``compose<size>_fu<f_u>_fl<f_l>_L<anchor>``).

SWAP, never ADD: ``fits.compose_u_pool`` computes ``n_elic = round(f_u * size)``
and ``n_gen = size - n_elic`` at a FIXED ``size``, so trait pairs REPLACE
generic pairs. Adding trait data on top would confound composition with pool
size.

Every other argument is ``issue1739_fits.main()`` verbatim -- pass ``--compose``,
``--compose-u-size``, ``--transfer`` and ``--eval-rung-knn`` through.
``--transfer`` is REQUIRED: without it no eval-split table is built and the
per-eval-setting reconstruction reads do not exist at all.

``EPM_R5_UNION=1`` adds a FIFTH config, ``union_all``, which is deliberately NOT
a ladder rung and is EXEMPT from the clamp above. The ladder asks "does
COMPOSITION matter at a fixed pool size" (SWAP); ``union_all`` asks "what happens
when the map gets everything the fairness budget allows" (ADD), so its pool is
intentionally LARGER and unmatched: generic WildChat u_store rows UNION the
behavior's trait-eliciting TRAIN pairs UNION the E1 EXTRACTION pairs (the
persona-vectors TRAIN side -- the recipe's contrastive instruction pairs crossed
with the extraction question set, the rows ``r_b_e1`` is a diff-of-means over;
NOT pvsynth, which is the persona-vectors EVAL side and has no train split). This is legal because a map consumes only (context, answer) pairs and
never the behavior judgments, so every corpus the predictors are granted is also
available as map-training data. Run together the two disentangle size from
composition: a ``union_all`` gain with flat swap rungs is a SIZE effect; swap
rungs that also move indicate COMPOSITION. Realized pool size is recorded per
behavior (it differs by construction) and every source's row count is reported.

The pilot gate runs a SUBSET of the ladder through this same entrypoint at
production shape: ``EPM_R5_F_U`` (comma-separated) overrides the rungs, so the
measured per-instance wall and the module-binding check come from the exact code
path the fan-out then uses. ``map_diagnostics.json`` is written once at the END
of a leg, which is why the gate is a short 2-rung leg rather than a peek inside
the full one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The authorized ladder: trait fraction of the map pool, at f_l = 1.0.
LADDER_F_U = (0.0, 0.25, 0.5, 1.0)
LADDER_F_L = (1.0,)

# Sentinel f_u marking the union_all config. Deliberately outside [0, 1] so it
# can never be confused with a ladder rung; the patched _u_pool_for_spec
# intercepts it before compose_u_pool's range validation is ever reached.
UNION_F_U = 2.0


def _ladder_f_u() -> tuple[float, ...]:
    """The rung set, overridable via EPM_R5_F_U for the pilot gate."""
    raw = os.environ.get("EPM_R5_F_U", "").strip()
    if not raw:
        return LADDER_F_U
    rungs = tuple(float(tok) for tok in raw.split(",") if tok.strip())
    if not rungs or any(not 0.0 <= r <= 1.0 for r in rungs):
        raise SystemExit(f"[r5-ladder] EPM_R5_F_U out of [0,1] or empty: {raw!r}")
    return rungs


def _clamp_max_f_u() -> float:
    """The ROUND's largest trait fraction -- the binding constraint for the clamp.

    The eliciting side must supply ``f_u * size`` contexts, so the feasible
    matched pool is ``size <= n_contexts / max(f_u)``. This is deliberately NOT
    read from the running leg's own rung subset: a gate leg running only
    ``f_u=0.0`` has no eliciting requirement at all and would compute a LARGER
    cap than its ``f_u=0.5`` sibling, leaving the two configs of one behaviour
    at DIFFERENT pool sizes -- which is exactly the size/composition confound
    the matched pool exists to prevent. Every leg of a round therefore passes
    the same ``EPM_R5_MAX_FU``.
    """
    raw = os.environ.get("EPM_R5_MAX_FU", "").strip()
    if raw:
        v = float(raw)
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f"[r5-ladder] EPM_R5_MAX_FU out of [0,1]: {raw!r}")
        return v
    return max([r for r in LADDER_F_U if r > 0.0] or [1.0])


def _load_fits_module():
    """Import scripts/issue1739_fits.py by path (it is a script, not a package)."""
    path = REPO_ROOT / "scripts" / "issue1739_fits.py"
    spec = importlib.util.spec_from_file_location("issue1739_fits", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue1739_fits"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    mod = _load_fits_module()

    from explore_persona_space.experiments.issue_1739 import constants

    f_u = _ladder_f_u()
    constants.COMPOSITION_F_U = f_u
    constants.COMPOSITION_F_L = LADDER_F_L
    tag = "" if f_u == LADDER_F_U else "  [PILOT SUBSET via EPM_R5_F_U]"
    print(
        f"[r5-ladder] composition grid overridden: f_u={f_u} f_l={LADDER_F_L}{tag}",
        flush=True,
    )

    original = mod.compose_run_specs

    def compose_specs_only(*args, **kwargs):
        specs = original(*args, **kwargs)
        kept = [s for s in specs if s.f_u is not None]
        dropped = len(specs) - len(kept)
        if not kept:
            raise SystemExit(
                "[r5-ladder] no composition specs in the grid -- pass --compose "
                "(and --compose-u-size); the plain rungs are deliberately dropped."
            )
        if os.environ.get("EPM_R5_UNION") == "1":
            import dataclasses as _dc

            seen_variants: list[str] = []
            for s in kept:
                if s.variant not in seen_variants:
                    seen_variants.append(s.variant)
            base_by_variant = {s.variant: s for s in reversed(kept)}
            for v in seen_variants:
                kept.append(_dc.replace(base_by_variant[v], f_u=UNION_F_U, f_l=1.0))
            print(
                f"[r5-ladder] union_all: appended {len(seen_variants)} ADD-semantics "
                "spec(s) (generic U + trait-eliciting TRAIN + PV-synth TRAIN; "
                "pool deliberately unmatched, clamp-exempt)",
                flush=True,
            )
            if os.environ.get("EPM_R5_UNION_ONLY") == "1":
                # The swap rungs this union is compared against are already
                # banked from the matched-pool round; re-running one here would
                # be duplicate compute. A base spec is still REQUIRED above --
                # the union spec is derived from it by dataclasses.replace --
                # so the drop happens after the append, never before.
                kept = [s for s in kept if s.f_u == UNION_F_U]
                print(
                    f"[r5-ladder] union_all: EPM_R5_UNION_ONLY=1 -> dropped the base "
                    f"swap spec(s); {len(kept)} union spec(s) remain",
                    flush=True,
                )
        keys = sorted({(s.f_u, s.f_l) for s in kept})
        print(
            f"[r5-ladder] {len(kept)} compose specs over {len(keys)} configs "
            f"{keys}; dropped {dropped} plain rung spec(s)",
            flush=True,
        )
        return kept

    mod.compose_run_specs = compose_specs_only

    # --- per-behavior matched-size clamp (see docstring item 4) ---------------
    import dataclasses

    original_pool = mod._u_pool_for_spec
    clamp_state: dict[str, int] = {}

    union_state: dict = {}

    # Record each split's context ids as the driver loads them, so the
    # contamination check can prove no eval-rung context enters the map pool.
    original_load_labeled = mod._load_labeled

    def load_labeled_recording(store_dir, dv_json, layers_, **kw):
        out = original_load_labeled(store_dir, dv_json, layers_, **kw)
        union_state.setdefault("ctx_by_config", {})[kw.get("config")] = list(out.ctx_order)
        # config_b IS the eval split (CONFIG_SPLIT['config_b'] == 'eval'); keep
        # it so the compose-spec eval_rung patch below can score against it.
        if kw.get("config") == "config_b":
            union_state["tbl_ev"] = out
        return out

    mod._load_labeled = load_labeled_recording

    def _load_extraction_pairs(layers, variant, dim):
        """E1 extraction (context, answer) pairs -- the persona-vectors TRAIN side.

        This is the third union_all source. NOT pvsynth: pvsynth is the
        persona-vectors EVAL side (200 rows/behavior, all split='eval' --
        `issue1739_pvsynth_score.py` hardcodes SPLIT="eval"), so it has no train
        split at all. The TRAIN side is the E1 extraction store -- the recipe's
        5 pos/neg contrastive instruction pairs crossed with the EXTRACTION
        question set, disjoint from the held-out eval questions by the recipe's
        own design, and the same rows `r_b_e1` is a diff-of-means over.

        The map needs only (context, answer) pairs, so the judge-filtering that
        gates r_B extraction is irrelevant here -- take the pairs, both sides.

        Returns (x, y, ctx_ids) or None. A consumer-open FAILURE returns None
        (recorded, two-source fallback) rather than improvising a loader.
        """
        import numpy as np

        e1_store = os.environ.get("EPM_R5_E1_STORE")
        if not e1_store:
            if os.environ.get("EPM_R5_UNION_ALLOW_NO_PV") != "1":
                raise SystemExit(
                    "[r5-ladder] union_all ABORT: EPM_R5_E1_STORE unset. union_all is "
                    "DEFINED as generic + trait-train + E1-extraction; running without "
                    "the extraction side would silently ship a two-source pool under a "
                    "three-source name. Set it, or waive with EPM_R5_UNION_ALLOW_NO_PV=1."
                )
            union_state["pv_error"] = os.environ.get("EPM_R5_UNION_NO_PV_REASON") or (
                "extraction side excluded via the explicit EPM_R5_UNION_ALLOW_NO_PV "
                "waiver -- union_all ran TWO-source. The DEFAULT is THREE-source."
            )
            print(
                f"[r5-ladder] union_all: two-source (WAIVED) -- {union_state['pv_error']}",
                flush=True,
            )
            return None

        from explore_persona_space.experiments.issue_1739 import store_io

        # CONSUMER-OPEN PROBE: the real load through the real helper. A failure
        # here is recorded + falls back to two-source -- never an improvised
        # loader on a billing box.
        try:
            arrays, meta = store_io.load_summaries(
                Path(e1_store), (variant, "t1"), tuple(layers), hidden_dim=dim
            )
        except Exception as exc:  # noqa: BLE001 - any open failure is the same disposition
            union_state["pv_error"] = (
                f"extraction store present but CONSUMER-OPEN FAILED for variant={variant!r}: "
                f"{type(exc).__name__}: {exc}. union_all ran TWO-source."
            )
            print(f"[r5-ladder] union_all: two-source -- {union_state['pv_error']}", flush=True)
            return None

        x = np.stack([arrays[(variant, ly)] for ly in layers])
        y = np.stack([arrays[("t1", ly)] for ly in layers])
        try:
            ctx_key = mod._meta_field(meta, ("context_id",), "context id")
            ids = [str(r[ctx_key]) for r in meta]
        except Exception:  # noqa: BLE001 - ids are for the contamination read only
            ids = []
        union_state["extraction_ctx"] = ids
        print(
            f"[r5-ladder] union_all: E1 extraction rows={x.shape[1]} "
            f"variant={variant} distinct_ctx={len(set(ids))}",
            flush=True,
        )
        return x, y, ids

    def _pvsynth_eval_ctx():
        """pvsynth EVAL context ids, for the extraction-vs-eval disjointness check."""
        import json as _json

        pv_dv = os.environ.get("EPM_R5_PV_DV")
        if not pv_dv or not Path(pv_dv).is_file():
            return []
        try:
            rows = _json.loads(Path(pv_dv).read_text())["rows"]
        except Exception:  # noqa: BLE001
            return []
        return [str(r.get("context_id")) for r in rows if r.get("split") == "eval"]

    def _contamination_report():
        """No eval context may enter the map pool. Every overlap counted.

        The extraction questions are disjoint from the pvsynth eval questions by
        the persona-vectors recipe's own design -- verified here rather than
        trusted. A non-zero `extraction_x_pvsynth_eval` means the extraction/eval
        split is not what the recipe claims: that is a finding, and it HALTS.
        """
        by_cfg = union_state.get("ctx_by_config", {})
        train_ctx = set(by_cfg.get("config_a") or [])
        eval_ctx = set(by_cfg.get("config_b") or [])
        ext = set(union_state.get("extraction_ctx") or [])
        pv_eval = set(union_state.get("pvsynth_eval_ctx") or [])
        rep = {
            "n_trait_train": len(train_ctx),
            "n_trait_eval": len(eval_ctx),
            "trait_train_x_trait_eval": len(train_ctx & eval_ctx),
            "n_extraction_ctx": len(ext),
            "n_pvsynth_eval": len(pv_eval),
            "extraction_x_pvsynth_eval": len(ext & pv_eval),
            "extraction_x_trait_eval": len(ext & eval_ctx),
        }
        rep["clean"] = (
            rep["trait_train_x_trait_eval"] == 0
            and rep["extraction_x_pvsynth_eval"] == 0
            and rep["extraction_x_trait_eval"] == 0
        )
        union_state["contamination"] = rep
        print(f"[r5-ladder] union_all CONTAMINATION CHECK: {rep}", flush=True)
        if not rep["clean"]:
            raise SystemExit(
                "[r5-ladder] union_all ABORT: eval contexts leak into the map pool "
                f"-- {rep}. A non-zero extraction_x_pvsynth_eval additionally means "
                "the recipe's extraction/eval split is not what it claims."
            )
        return rep

    def _build_union_pool(spec, u_arrays, u_fit_rows, tbl, layers):
        """generic U + trait-eliciting TRAIN + PV-synth TRAIN, concatenated (ADD).

        A map consumes only (context, answer) pairs, never the behavior
        judgments, so every corpus the predictors are granted is legal
        map-training data. Both extra sources are TRAIN-split only, and the
        contamination report below proves no eval-rung context enters the pool.
        """
        import numpy as np

        variant = spec.variant

        def _stack(arrs, rows):
            x = np.stack([arrs[(variant, ly)][rows] for ly in layers])
            y = np.stack([arrs[("t1", ly)][rows] for ly in layers])
            return x, y

        parts_x, parts_y, provenance = [], [], {}

        gx, gy = _stack(u_arrays, u_fit_rows)
        parts_x.append(gx)
        parts_y.append(gy)
        provenance["generic_u"] = int(len(u_fit_rows))

        # trait-eliciting TRAIN pairs: `tbl` IS the train split (config_a ->
        # CONFIG_SPLIT['train']), so this side is train-only by construction.
        tx = np.asarray(tbl.z_by_variant[variant], dtype=gx.dtype)
        ty = np.asarray(tbl.z_ans, dtype=gy.dtype)
        parts_x.append(tx)
        parts_y.append(ty)
        provenance["trait_train"] = int(tx.shape[1])

        # E1 extraction pairs load PER VARIANT (the context side differs), so
        # cache by variant rather than once per process.
        cache = union_state.setdefault("extraction_by_variant", {})
        if variant not in cache:
            cache[variant] = _load_extraction_pairs(layers, variant, int(gx.shape[-1]))
            union_state["pvsynth_eval_ctx"] = _pvsynth_eval_ctx()
            _contamination_report()

        ext = cache[variant]
        if ext is not None:
            ex, ey, _ = ext
            parts_x.append(np.asarray(ex, dtype=gx.dtype))
            parts_y.append(np.asarray(ey, dtype=gy.dtype))
            provenance["extraction_train"] = int(ex.shape[1])
        else:
            provenance["extraction_train"] = 0
            provenance["extraction_absent_reason"] = union_state.get("pv_error", "not loaded")

        x = np.concatenate(parts_x, axis=1)
        y = np.concatenate(parts_y, axis=1)
        total = int(x.shape[1])
        provenance["total"] = total
        union_state.setdefault("provenance", {})[variant] = provenance
        print(f"[r5-ladder] union_all[{variant}] pool: {provenance}", flush=True)
        return x, y, f"unionall{total}_L{spec.budgets[0]}", total

    def u_pool_clamped(spec, u_arrays, u_fit_rows, tbl, layers):
        # `fit_state` is assigned later in main() but only READ during
        # mod.main(), so the closure resolves it fine; it carries the spec +
        # realized u_label forward to the eval_rung patch, which sees neither.
        if spec.f_u == UNION_F_U:
            res = _build_union_pool(spec, u_arrays, u_fit_rows, tbl, layers)
            fit_state["spec"], fit_state["u_label"] = spec, res[2]
            return res
        if spec.f_u is not None:
            mfu = _clamp_max_f_u()
            n_ctx, n_gen = len(tbl.ctx_order), len(u_fit_rows)
            # size <= n_contexts / max(f_u) keeps the ELICITING side fillable;
            # size <= n_generic keeps the GENERIC side fillable.
            cap = n_gen if mfu <= 0.0 else min(n_gen, int(n_ctx / mfu))
            if "cap" not in clamp_state:
                clamp_state["cap"] = cap
                clamp_state["max_f_u"] = mfu
                print(
                    f"[r5-ladder] matched pool size: n_generic={n_gen} "
                    f"n_contexts={n_ctx} max_f_u={mfu} -> cap={cap} "
                    f"(requested ceiling {spec.u_size})",
                    flush=True,
                )
            if spec.u_size is not None and spec.u_size > cap:
                print(
                    f"[r5-ladder] CLAMP f_u={spec.f_u}: pool size "
                    f"{spec.u_size} -> {cap} (f_u=1.0 needs size <= n_contexts; "
                    "constant WITHIN this behavior across all rungs)",
                    flush=True,
                )
                spec = dataclasses.replace(spec, u_size=cap)
        res = original_pool(spec, u_arrays, u_fit_rows, tbl, layers)
        fit_state["spec"], fit_state["u_label"] = spec, res[2]
        return res

    mod._u_pool_for_spec = u_pool_clamped

    # --- per-eval-setting reads for COMPOSE specs (deviation 5) --------------
    # The driver computes its per-eval-setting reconstruction block
    # (`eval_rung`: R^2 / identity+bias / kNN per eval distribution) INSIDE
    # `if spec0.f_u is None:` -- i.e. for PLAIN rungs only. Every config in
    # this round is a compose spec by construction (the trait-augmented arm
    # cannot be anything else), so the driver would emit only the map's OWN
    # holdout and none of the per-setting cross that IS Result 5.
    #
    # Rather than edit `issue1739_fits.py` (owned by a live session), drive the
    # driver's OWN `_eval_rung_reconstruction` from here on exactly the inputs
    # its plain-rung path would use: the eval-split table (`config_b`), the
    # whitening fit for this pool, and the freshly fit map. The injected blocks
    # carry `eval_rung_source: r5-ladder-wrapper` so they are never mistaken
    # for the driver's native output.
    eval_rung_extra: dict[str, dict] = {}
    fit_state: dict = {}
    want_knn = "--eval-rung-knn" in list(argv if argv is not None else sys.argv[1:])

    # The driver imports `fits` FUNCTION-LOCALLY (`from ...issue_1739 import
    # fits` inside each function), so it re-resolves the module attribute on
    # every call -- patch the module object, not a `mod.fits` alias, which
    # does not exist.
    from explore_persona_space.experiments.issue_1739 import fits as fits_mod

    orig_fit_whitening = fits_mod.fit_whitening

    def fit_whitening_capture(*a, **k):
        wh = orig_fit_whitening(*a, **k)
        fit_state["wh"] = wh
        return wh

    fits_mod.fit_whitening = fit_whitening_capture

    orig_fit_map = mod._fit_map

    def fit_map_capture(args_, x_w, y_w):
        mapfit = orig_fit_map(args_, x_w, y_w)
        spec = fit_state.get("spec")
        tbl_ev = union_state.get("tbl_ev")
        wh = fit_state.get("wh")
        # Plain rungs get the block from the driver itself; compose specs are
        # the gap this fills. No --transfer => no eval table => nothing to do.
        if spec is not None and spec.f_u is not None and tbl_ev is not None and wh is not None:
            if tbl_ev.z_ans is not None:
                z_ev_w = fits_mod.apply_whitening(tbl_ev.z_by_variant[spec.variant], wh)
                za_ev_w = fits_mod.apply_whitening(tbl_ev.z_ans, wh)
                block = mod._eval_rung_reconstruction(
                    mapfit,
                    z_ev_w,
                    za_ev_w,
                    rungs=tbl_ev.row_rungs if want_knn else None,
                    knn=want_knn,
                )
                block["eval_rung_source"] = "r5-ladder-wrapper"
                key = f"{spec.variant}|{fit_state.get('u_label')}"
                eval_rung_extra[key] = block
                print(f"[r5-ladder] eval_rung computed for compose cell {key}", flush=True)
        return mapfit

    mod._fit_map = fit_map_capture

    rc = mod.main(argv)

    # Persist the ladder's own provenance beside the driver's outputs: realized
    # pool size per behavior (the clamp makes it differ), union_all's per-source
    # row counts, and the contamination verdict. The collector reads these so a
    # cross-behavior read is never mistaken for a matched-pool comparison.
    argv_l = list(argv if argv is not None else sys.argv[1:])
    out_root = None
    if "--out-root" in argv_l:
        out_root = Path(argv_l[argv_l.index("--out-root") + 1])
    # Splice the compose-spec eval_rung blocks into the driver's own
    # map_diagnostics.json. Never overwrites a block the driver produced
    # natively (plain rungs); fails loud if a computed block has nowhere to go,
    # since a silently dropped read is the deliverable going missing.
    if out_root is not None and eval_rung_extra:
        diag_path = out_root / "map_diagnostics.json"
        if not diag_path.is_file():
            raise SystemExit(f"[r5-ladder] no {diag_path} to splice eval_rung into")
        diag = json.loads(diag_path.read_text())
        injected, orphaned = [], []
        for key, block in eval_rung_extra.items():
            if key not in diag:
                orphaned.append(key)
            elif "eval_rung" in diag[key]:
                print(f"[r5-ladder] eval_rung already native for {key}; left alone", flush=True)
            else:
                diag[key]["eval_rung"] = block
                injected.append(key)
        if orphaned:
            raise SystemExit(
                f"[r5-ladder] computed eval_rung for {orphaned} but no such cell in "
                f"{diag_path} (keys: {sorted(diag)}) -- refusing to drop the read"
            )
        diag_path.write_text(json.dumps(diag, indent=1))
        print(f"[r5-ladder] spliced eval_rung into {len(injected)} cell(s): {injected}", flush=True)

    if out_root is not None:
        sidecar = {
            "ladder_f_u": list(f_u),
            "ladder_f_l": list(LADDER_F_L),
            "matched_pool_size_cap": clamp_state.get("cap"),
            "clamp_max_f_u": clamp_state.get("max_f_u"),
            "union_all_enabled": os.environ.get("EPM_R5_UNION") == "1",
            "union_all_only": os.environ.get("EPM_R5_UNION_ONLY") == "1",
            "variants_run": sorted(
                {s for s in (os.environ.get("EPM_R5_VARIANTS") or "").split(",") if s}
            )
            or None,
            "stated_deviations": [
                d for d in (os.environ.get("EPM_R5_DEVIATIONS") or "").split("||") if d
            ]
            or None,
            "union_all_provenance": union_state.get("provenance"),
            "contamination": union_state.get("contamination"),
        }
        try:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "r5_ladder_meta.json").write_text(json.dumps(sidecar, indent=1))
            print(f"[r5-ladder] wrote {out_root / 'r5_ladder_meta.json'}", flush=True)
        except OSError as exc:  # never let a sidecar write mask the run's rc
            print(f"[r5-ladder] WARNING: sidecar write failed: {exc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.stdout.flush()
    sys.exit(main())
