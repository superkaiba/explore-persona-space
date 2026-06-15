"""Per-behavior raw-scatter loaders for task #644 (functional-form meta-analysis).

Each loader reads a prior task's eval JSONs (per the plan §4 source-data map) and
emits one or more :class:`ScatterInput` objects — the unified
``(unit, x, y, behavior, frame, geometry_scalar_kind, centering_family)`` rows
the fit machinery in :mod:`explore_persona_space.analysis.convexity_meta`
consumes. Loaders NEVER rank-transform (RAW values only — the central
measurement-validity point). The #311 loader enforces the MF3 name-keyed join
with fail-loud on any unmatched rate persona.

All inputs are benign geometry/rate corpora (cosines, judged rates, marker
log-probs / emit rates). No harmful-content text is read here.

Branch artifacts:
* #623's scatter lives on ``origin/issue-623`` (not ``main``). The driver
  snapshots it into ``eval_results/issue_644/inputs/issue623/`` via
  ``git show`` before the loader reads the content-pinned local copy.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.analysis.convexity_meta import ScatterInput

# Marker token id (Qwen-2.5-7B " ※"), used to compute #532's emit rate.
MARKER_TOKEN_ID = 83399

# #623 source branch + pin (plan §10 reuse table).
ISSUE623_BRANCH = "origin/issue-623"
ISSUE623_PIN = "1907baa8"
ISSUE623_HEADLINE_ARM = "lt_persona_lt_syc"
ISSUE623_HEADLINE_LAYER = "14"
ISSUE623_DROPPED_BASELINE = "assistant"

# #444 chosen contrastive recipe (CC3 — on-policy negative arm).
ISSUE444_CHOSEN_RECIPE = "leak_on-policy neg."
ISSUE444_OTHER_RECIPES = ["leak_contradictory neg.", "leak_refusal neg."]


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# --- #623 sycophancy seed (branch snapshot) -----------------------------------


def snapshot_issue623(dest_dir: Path) -> dict[str, Path]:
    """``git show`` the #623 scatter from ``origin/issue-623`` into ``dest_dir``.

    Returns the local snapshot paths. Fail-loud if the branch ref is absent.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("cosine_matrix", "syc_i", "rho_loo_leverage"):
        out = dest_dir / f"{name}.json"
        ref = f"{ISSUE623_BRANCH}:eval_results/issue_623/{name}.json"
        res = subprocess.run(
            ["git", "show", ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"could not snapshot {ref} (rc={res.returncode}): {res.stderr.strip()}"
            )
        out.write_text(res.stdout)
        paths[name] = out
    return paths


def load_issue623_sycophancy(snapshot_dir: Path) -> list[ScatterInput]:
    """Sycophancy seed: cosine(persona LT vector, syc LT direction) vs judged rate.

    Geometry frame, cosine_to_direction. Headline arm ``lt_persona_lt_syc`` at
    layer 14. The ``assistant`` baseline persona is dropped (n 36 -> 35). Carries
    the ``ravg_persona_lt_syc`` arm as a circularity-robustness alternate row.
    """
    cos = _load_json(snapshot_dir / "cosine_matrix.json")["cosine_matrix"]
    syc = _load_json(snapshot_dir / "syc_i.json")["syc_i"]

    scatters: list[ScatterInput] = []
    for arm in (ISSUE623_HEADLINE_ARM, "ravg_persona_lt_syc"):
        if arm not in cos:
            continue
        per_layer = cos[arm]
        if ISSUE623_HEADLINE_LAYER not in per_layer:
            raise RuntimeError(
                f"#623 arm {arm} missing layer {ISSUE623_HEADLINE_LAYER}; "
                f"have {list(per_layer.keys())}"
            )
        cos_by_persona = per_layer[ISSUE623_HEADLINE_LAYER]
        units, xs, ys = [], [], []
        for persona, c in cos_by_persona.items():
            if persona == ISSUE623_DROPPED_BASELINE:
                continue
            if persona not in syc:
                # Cosine persona without a syc rate: skip (not in the n=35 set).
                continue
            units.append(persona)
            xs.append(float(c))
            ys.append(float(syc[persona]["syc_i"]))
        scatters.append(
            ScatterInput(
                behavior="sycophancy_seed",
                frame=f"geometry/{arm}/L{ISSUE623_HEADLINE_LAYER}",
                geometry_scalar_kind="cosine_to_direction",
                centering_family="n/a",
                x=np.array(xs),
                y=np.array(ys),
                units=units,
                layer=int(ISSUE623_HEADLINE_LAYER),
                y_is_rate=True,
                notes=[
                    f"#623 arm={arm} layer={ISSUE623_HEADLINE_LAYER}; "
                    f"baseline '{ISSUE623_DROPPED_BASELINE}' dropped; pin={ISSUE623_PIN}",
                    "RATE DV near floor (most personas below 0.10) -> logit double-fit",
                ]
                + (
                    ["circularity-robustness alternate arm"] if arm != ISSUE623_HEADLINE_ARM else []
                ),
            )
        )
    return scatters


# --- #311 marker leakage (centered-centroid), name-keyed join (MF3) -----------


def load_issue311_marker(eval_root: Path, arm: str = "joint") -> ScatterInput:
    """Marker leakage (#311), centered-centroid cosine to source, NAME-KEYED join.

    The cosine bank has 19 personas (``cosine_l20_base.json::personas``); the
    rates are over 17 bystanders (``analysis.json::bystanders``) — the 19 minus
    the source pair ``[paramedic, comedian]``. Membership/order differ, so the
    join is by persona NAME (MF3), NOT by index. Fail loud on any unmatched rate
    persona; emit ``matched_row_count``.

    The contrastive geometry scalar is the projection along the source A-B axis,
    ``t = (cos_to_A - cos_to_B) / 2`` — the analysis's own precomputed ``t_vals``.
    We RECONSTRUCT it from the cosine bank by name and cross-check against the
    stored ``t_vals`` (fail loud on disagreement) so the join is explicit.
    """
    d = eval_root / "eval_results" / "issue_311"
    cos = _load_json(d / "cosine_l20_base.json")
    ana = _load_json(d / "analysis.json")

    personas = cos["personas"]  # 19, cosine bank order
    matrix = cos["matrix"]  # 19x19
    pair = ana["pair"]  # [paramedic, comedian]
    a_persona, b_persona = pair[0], pair[1]
    if a_persona not in personas or b_persona not in personas:
        raise RuntimeError(f"#311 source pair {pair} not in cosine personas {personas}")
    ai = personas.index(a_persona)
    bi = personas.index(b_persona)

    bystanders = ana["bystanders"]  # 17, rates order
    if arm not in ana["rates_per_persona"]:
        raise RuntimeError(
            f"#311 arm {arm!r} not in rates_per_persona ({list(ana['rates_per_persona'].keys())})"
        )
    rates = ana["rates_per_persona"][arm]
    stored_t = ana.get("t_vals")  # precomputed (cos_A - cos_B)/2, bystander order
    if len(rates) != len(bystanders):
        raise RuntimeError(f"#311 rates len {len(rates)} != bystanders len {len(bystanders)}")

    name_to_idx = {p: i for i, p in enumerate(personas)}
    units, xs, ys = [], [], []
    unmatched: list[str] = []
    for j, byst in enumerate(bystanders):
        if byst not in name_to_idx:
            unmatched.append(byst)
            continue
        bj = name_to_idx[byst]
        cos_a = float(matrix[bj][ai])
        cos_b = float(matrix[bj][bi])
        t = (cos_a - cos_b) / 2.0
        # Cross-check the name-keyed reconstruction against the stored t_vals.
        if stored_t is not None and j < len(stored_t) and abs(t - float(stored_t[j])) > 1e-6:
            raise RuntimeError(
                f"#311 name-keyed join MISMATCH for {byst}: "
                f"reconstructed t={t:.6f} != stored t_vals[{j}]={stored_t[j]:.6f} "
                "(index/name misalignment)"
            )
        units.append(byst)
        xs.append(t)
        ys.append(float(rates[j]))

    if unmatched:
        raise RuntimeError(
            f"#311 name-keyed join: {len(unmatched)} rate personas absent from the "
            f"cosine bank — {unmatched}. Refusing to silently drop (MF3)."
        )

    matched = len(units)
    return ScatterInput(
        behavior="marker_leakage_centered",
        frame=f"geometry/centered_centroid/L20/{arm}",
        geometry_scalar_kind="cosine_centered_centroid",
        centering_family="centered_centroid",
        x=np.array(xs),
        y=np.array(ys),
        units=units,
        layer=20,
        matched_row_count=matched,
        y_is_rate=True,
        notes=[
            f"#311 single source-pair {pair} (NOT a marker-leakage population claim)",
            f"name-keyed join: {matched} matched rows (rates 17, cosine bank 19); "
            "n_saturated=0 (mask no-op)",
            "X = (cos_to_A - cos_to_B)/2 (contrastive proximity along source axis), "
            "cross-checked vs stored t_vals",
            "RATE DV -> logit double-fit; CENTERED-CENTROID family (never pooled with #532)",
        ],
    )


# --- #532 marker leakage (raw/uncentered), per-source emit rate ---------------


def load_issue532_marker(eval_root: Path, max_sources: int | None = None) -> list[ScatterInput]:
    """Marker leakage (#532), RAW/uncentered cosine to source vs per-bystander emit rate.

    The emit rate is assembled from the 416 per-cell files under
    ``logp_slot_followup/per_cell_trained/<source>__<bystander>.json`` — Y is the
    fraction of the 50 ``per_q`` probes whose ``emitted_id == MARKER_TOKEN_ID``.
    X is ``cosine_matrix[source_idx][bystander_idx]`` (raw/uncentered, a DIFFERENT
    centering family from #311 — never pooled, CC1).

    One scatter per source (16 sources x up to 26 bystanders). ``js_v1`` is the
    deprecated single-next-token JS (CC2) — emitted as a separate sensitivity
    scatter with ``geometry_scalar_kind='js_deprecated_single_next_token'``.
    """
    d = eval_root / "eval_results" / "issue_532"
    pred = _load_json(d / "predictors.json")
    sources = pred["sources"]  # 16
    bystanders = pred["bystanders"]  # 26
    cos_matrix = pred["cosine_matrix"]  # 16 src x 26 byst
    js_matrix = pred.get("js_v1_matrix")  # 16 x 26, DEPRECATED

    pct = d / "logp_slot_followup" / "per_cell_trained"
    if not pct.is_dir():
        raise RuntimeError(f"#532 per_cell_trained missing at {pct}")

    # Assemble emit rate per (source, bystander) from the per-cell probe files.
    emit_rate: dict[tuple[str, str], float] = {}
    for fn in pct.glob("*.json"):
        stem = fn.stem  # "<source>__<bystander>"
        if "__" not in stem:
            continue
        src, byst = stem.split("__", 1)
        cell = _load_json(fn)
        pq = cell.get("per_q", [])
        if not pq:
            continue
        n_emit = sum(1 for q in pq if q.get("emitted_id") == MARKER_TOKEN_ID)
        emit_rate[(src, byst)] = n_emit / len(pq)

    scatters: list[ScatterInput] = []
    src_list = sources if max_sources is None else sources[:max_sources]
    for src in src_list:
        if src not in sources:
            continue
        si = sources.index(src)
        units, xs, ys, js_xs = [], [], [], []
        for bj, byst in enumerate(bystanders):
            if bj >= len(cos_matrix[si]):
                continue
            rate = emit_rate.get((src, byst))
            if rate is None:
                continue
            units.append(byst)
            xs.append(float(cos_matrix[si][bj]))
            ys.append(float(rate))
            if js_matrix is not None and bj < len(js_matrix[si]):
                js_xs.append(float(js_matrix[si][bj]))
        if len(units) < 3:
            continue
        scatters.append(
            ScatterInput(
                behavior="marker_leakage_raw",
                frame=f"geometry/raw_uncentered/source_{src}",
                geometry_scalar_kind="cosine_to_source",
                centering_family="raw_uncentered",
                x=np.array(xs),
                y=np.array(ys),
                units=units,
                layer=None,
                y_is_rate=True,
                notes=[
                    f"#532 source {src}: emit rate = frac(emitted_id==marker) over per_q",
                    "RAW/UNCENTERED cosine family (NEVER pooled with #311 centered)",
                    "RATE DV (heavily floor-clustered) -> logit double-fit",
                ],
            )
        )
        # Deprecated js_v1 sensitivity scatter (CC2) — excluded from headline.
        if js_matrix is not None and len(js_xs) == len(units):
            scatters.append(
                ScatterInput(
                    behavior="marker_leakage_raw",
                    frame=f"sensitivity/js_v1_deprecated/source_{src}",
                    geometry_scalar_kind="js_deprecated_single_next_token",
                    centering_family="raw_uncentered",
                    x=np.array(js_xs),
                    y=np.array(ys),
                    units=units,
                    layer=None,
                    y_is_rate=True,
                    notes=[
                        f"#532 source {src}: DEPRECATED single-next-token JS (CC2), "
                        "sensitivity-only, EXCLUDED from geometry recurs numerator",
                    ],
                )
            )
    return scatters


# --- #500 fact leakage (geometry + prior frames) ------------------------------


def load_issue500_fact(eval_root: Path) -> list[ScatterInput]:
    """Fact leakage (#500): cos_to_source (geometry) + prior_logprob (prior) vs leak_mean.

    Per arm (3 source arms), per persona (14). Emits one geometry-frame scatter
    (``cos_to_source``) and one prior-frame scatter (``prior_logprob``, the
    sensitivity table) per arm. ``cos_to_home`` is NaN for some rows and is not
    used as an axis.
    """
    d = eval_root / "eval_results" / "issue_500"
    pred = _load_json(d / "predictors.json")
    per_arm = pred["per_arm"]

    scatters: list[ScatterInput] = []
    for arm, arm_data in per_arm.items():
        pp = arm_data.get("per_persona", {})
        units, xs_geo, xs_prior, ys = [], [], [], []
        for persona, fields in pp.items():
            leak = fields.get("leak_mean")
            cos = fields.get("cos_to_source")
            prior = fields.get("prior_logprob")
            if leak is None:
                continue
            units.append(persona)
            ys.append(float(leak))
            xs_geo.append(float(cos) if cos is not None else float("nan"))
            xs_prior.append(float(prior) if prior is not None else float("nan"))
        if len(units) < 3:
            continue
        scatters.append(
            ScatterInput(
                behavior="fact_leakage",
                frame=f"geometry/cos_to_source/arm_{arm}",
                geometry_scalar_kind="cosine_to_source",
                centering_family="raw_uncentered",
                x=np.array(xs_geo),
                y=np.array(ys),
                units=units,
                layer=None,
                y_is_rate=True,
                notes=[
                    f"#500 arm {arm}: cos_to_source (teacher-referenced geometry)",
                    "GEOMETRY-frame fact row -> H1 numerator; RATE DV -> logit double-fit",
                ],
            )
        )
        scatters.append(
            ScatterInput(
                behavior="fact_leakage",
                frame=f"prior/prior_logprob/arm_{arm}",
                geometry_scalar_kind="prior_logprob",
                centering_family="n/a",
                x=np.array(xs_prior),
                y=np.array(ys),
                units=units,
                layer=None,
                y_is_rate=True,
                x_is_logprob=True,
                notes=[
                    f"#500 arm {arm}: prior_logprob (base-rate behavioral scalar, NOT geometry)",
                    "PRIOR-frame sensitivity row -> EXCLUDED from H1 numerator (MF1); "
                    "X is a log-prob -> log-space double-fit",
                ],
            )
        )
    return scatters


# --- #444 fact leakage (geometry + prior frames, single chosen recipe) --------


def load_issue444_fact(eval_root: Path) -> list[ScatterInput]:
    """Fact leakage (#444): cosine_on / js_on (geometry, wrong-sign) + base_logprob (prior).

    Y is the SINGLE chosen contrastive recipe ``leak_on-policy neg.`` (CC3). The
    other two recipes are emitted as separate sensitivity rows, NEVER pooled. The
    prior read path is ``correlations.json::per_persona.<p>.base_logprob`` (CC4),
    not ``persona_distance_topic``.
    """
    d = eval_root / "eval_results" / "issue_444" / "bystander_logprob"
    corr = _load_json(d / "correlations.json")
    pp = corr["per_persona"]

    def _build(recipe: str, label_suffix: str, primary: bool) -> list[ScatterInput]:
        units, cos_x, js_x, prior_x, ys = [], [], [], [], []
        for persona, fields in pp.items():
            leak = fields.get(recipe)
            if leak is None:
                continue
            units.append(persona)
            ys.append(float(leak))
            cos_x.append(float(fields.get("cosine_on", float("nan"))))
            js_x.append(float(fields.get("js_on", float("nan"))))
            prior_x.append(float(fields.get("base_logprob", float("nan"))))
        if len(units) < 3:
            return []
        out = [
            ScatterInput(
                behavior="fact_leakage",
                frame=f"geometry/cosine_on/{label_suffix}",
                geometry_scalar_kind="cosine_to_source",
                centering_family="raw_uncentered",
                x=np.array(cos_x),
                y=np.array(ys),
                units=units,
                layer=None,
                y_is_rate=True,
                notes=[
                    f"#444 recipe={recipe}: raw on-topic teacher cosine (WRONG-SIGN frame)",
                    (
                        "PRIMARY chosen recipe (on-policy neg., CC3)"
                        if primary
                        else "sensitivity recipe (NOT pooled, CC3)"
                    ),
                    "GEOMETRY-frame fact row" if primary else "sensitivity row",
                ],
            ),
            ScatterInput(
                behavior="fact_leakage",
                frame=f"geometry/js_on/{label_suffix}",
                geometry_scalar_kind="js",
                centering_family="raw_uncentered",
                x=np.array(js_x),
                y=np.array(ys),
                units=units,
                layer=None,
                y_is_rate=True,
                notes=[
                    f"#444 recipe={recipe}: raw on-topic teacher JS",
                    (
                        "PRIMARY chosen recipe (on-policy neg., CC3)"
                        if primary
                        else "sensitivity recipe (NOT pooled, CC3)"
                    ),
                ],
            ),
            ScatterInput(
                behavior="fact_leakage",
                frame=f"prior/base_logprob/{label_suffix}",
                geometry_scalar_kind="prior_logprob",
                centering_family="n/a",
                x=np.array(prior_x),
                y=np.array(ys),
                units=units,
                layer=None,
                y_is_rate=True,
                x_is_logprob=True,
                notes=[
                    f"#444 recipe={recipe}: base_logprob (prior frame, CC4 path correlations.json)",
                    "PRIOR-frame sensitivity -> EXCLUDED from H1 numerator (MF1)",
                ],
            ),
        ]
        return out

    scatters: list[ScatterInput] = []
    scatters += _build(ISSUE444_CHOSEN_RECIPE, "i444_onpolicy", primary=True)
    for recipe in ISSUE444_OTHER_RECIPES:
        slug = recipe.replace(" ", "_").replace(".", "").replace("/", "_")
        scatters += _build(recipe, f"i444_{slug}_sensitivity", primary=False)
    return scatters


# --- #390 refusal (NO commensurable geometry scalar -> flag for new generation) ---


def load_issue390_refusal(eval_root: Path) -> dict[str, Any]:
    """Refusal (#390): per-persona refusal pass-rate, but NO commensurable geometry scalar.

    ``aggregate_long.json`` has pass_rate per (persona x framing x seed) but the
    #390 directory carries NO persona-geometry artifact (cosine/JS bank). Per the
    plan §4/§12, refusal routes to a new-generation follow-up rather than a
    fabricated geometry scalar. This loader assembles the refusal-strength scalar
    (so the follow-up knows what Y would be) and returns an EXCLUSION record — it
    does NOT emit a ScatterInput with a geometry X.

    Returns a dict describing the excluded behavior + the assembled Y, for the
    driver to record in the output JSON (transparency, not a silent drop).
    """
    d = eval_root / "eval_results" / "issue_390"
    rows = _load_json(d / "aggregate_long.json")
    # Aggregate non-teach pass_rate per persona over framings/seeds.
    per_persona: dict[str, list[float]] = {}
    for r in rows:
        if r.get("is_teach"):
            continue
        persona = r.get("persona")
        pr = r.get("pass_rate")
        if persona is None or pr is None:
            continue
        per_persona.setdefault(persona, []).append(float(pr))
    refusal_strength = {p: float(np.mean(v)) for p, v in per_persona.items()}
    return {
        "behavior": "refusal",
        "excluded": True,
        "excluded_reason": (
            "no commensurable per-persona geometry scalar in #390 eval dir "
            "(aggregate_long.json has pass_rate only; no cosine/JS bank present); "
            "routed to a new-generation follow-up per plan §4/§12 (NOT a fabricated X)"
        ),
        "refusal_strength_per_persona_nonteach": refusal_strength,
        "n_personas_nonteach": len(refusal_strength),
    }


# --- top-level orchestration --------------------------------------------------


def load_all_scatters(
    eval_root: Path,
    issue623_snapshot_dir: Path,
    behaviors: list[str] | None = None,
    max_532_sources: int | None = None,
) -> tuple[list[ScatterInput], list[dict[str, Any]]]:
    """Load every behavior's scatters + the excluded-behavior records.

    ``behaviors`` restricts to a subset (used by ``--smoke``). Returns
    ``(scatters, exclusions)``.
    """
    want = set(behaviors) if behaviors else None

    def _wanted(name: str) -> bool:
        return want is None or name in want

    scatters: list[ScatterInput] = []
    exclusions: list[dict[str, Any]] = []

    if _wanted("sycophancy_seed"):
        scatters += load_issue623_sycophancy(issue623_snapshot_dir)
    if _wanted("marker_leakage_centered"):
        scatters.append(load_issue311_marker(eval_root, arm="joint"))
    if _wanted("marker_leakage_raw"):
        scatters += load_issue532_marker(eval_root, max_sources=max_532_sources)
    if _wanted("fact_leakage"):
        scatters += load_issue500_fact(eval_root)
        scatters += load_issue444_fact(eval_root)
    if _wanted("refusal"):
        exclusions.append(load_issue390_refusal(eval_root))

    return scatters, exclusions
