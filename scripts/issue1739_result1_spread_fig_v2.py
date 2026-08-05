"""Result 1 for #1739, simplified: score distribution + spread, per behavior x dataset.

Renders `figures/issue_1739/result1_spread/spread_grid_simple.{png,pdf,meta.json}`.

Two panels per behavior and nothing else:
  LEFT   distribution of the per-context mean judged score, one distribution per
         evaluation dataset.
  RIGHT  between-context SD of that same score, with the cell's own noise floor
         overlaid (a SD is uninterpretable without it, so the floor is data, not
         decoration).

Simplification of `issue1739_result1_spread_fig.py`, which carries the same two
panels plus a three-line subtitle, reliability-ceiling bars, per-cell ceiling
annotations and a reading-the-panel legend block. The loading and the statistics
are unchanged: SD, noise floor and n are read from that script's committed
`eval_results/issue_1739/result1_spread/spread_stats.json`, and the per-context
means this figure bins are recomputed from the same artifacts and asserted equal
to the committed (n, mean, SD) for all 14 cells.

Two DV codings, selected by `--coding` (both are rendered side by side; neither
replaces the other):

  drop           (default) the shipped coding. A judge return of `REFUSAL` is a
                 content DROP (llm-judging.md rule 9): the draw carries no score,
                 a rollout whose draws all dropped has no score, and a context
                 whose rollouts all dropped has no DV. Renders
                 `spread_grid_simple`.
  refusal-zero   EVIL ONLY, and only the REFUSAL channel: a `REFUSAL` draw is
                 scored 0 (refusing is the least-evil response, so it is a floor
                 observation rather than missing data). Parse-error draws stay
                 dropped in both codings, and sycophancy + hallucination are
                 untouched in both. Renders `spread_grid_refusal_zero` and
                 writes `spread_stats_refusal_zero.json`.

The recode needs PER-DRAW verdicts, which the per-context DV rows do not carry
(a mixed rollout stores only the mean over its kept draws), so the evil cells are
rebuilt from the raw judge payloads under `EVIL_JUDGE_SOURCES` — base wave plus
its higher-`max_tokens` re-judge, the re-judge replacing a re-judged rollout's
draws wholesale. The reconstruction is gated: with refusals dropped it must
reproduce every stored per-rollout score exactly (64,330 rollouts across the
three sources), and the recomputed per-cell statistics must reproduce the
committed `spread_stats.json` for all 14 cells.

Pure aggregation over already-committed judged DV rows and raw judge payloads:
no fits, no GPU, no network, no new judging.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import (  # noqa: E402
    BEHAVIORS,
    ROOT,
    RUNGS,
    WT,
)

ER = ROOT / "eval_results/issue_1739"
OUT_FIG = ROOT / "figures/issue_1739/result1_spread"
OUT_NUM = ER / "result1_spread"
STATS_PATH = OUT_NUM / "spread_stats.json"

PVSYNTH = "pvsynth"
SETTINGS = {b: [*RUNGS[b], PVSYNTH] for b in BEHAVIORS}

# Row labels are two-part: the ROLE the dataset plays in the design, then WHAT
# THE DATASET ACTUALLY IS. Figure-LOCAL by design — `RUNG_LABEL` in
# issue1739_recut_common is consumed by five other figure scripts, so it is read
# for the setting roster only and never mutated here.
#
# ROLE. `train` is the labeled fit pool: predictors are fit on it under 5
# group-level folds (`fits.realize_budget_cell`, N_FOLDS=5) and read
# OUT-OF-FOLD, so "held-out" here means out-of-fold, NOT a held-out data split
# (every train-rung row in labeling.json carries split="train"; there is no
# second split value). The only 80/20 in this pipeline is
# `WHITEN_HOLDOUT_FRAC` over the UNLABELED U pool, which carries no judged DV —
# see the module note in the report and #1739's body.
SETTING_ROLE = {
    "train": "in-distribution (out-of-fold)",
    "hhrt": "OOD transfer",
    "toxicchat": "OOD transfer",
    "nqopen": "OOD transfer",
    "simpleqa": "OOD transfer",
    # PROVENANCE (kept in code; deliberately NOT surfaced in the figure label —
    # user-directed presentation choice 2026-08-05, applied to Result 1 and the
    # Result 2 method figure alike so the two agree).
    #
    # ELEPHANT AITA-YTA had no resolvable HF id (epm:concern-raised
    # elephant-aita-unresolved, 2026-07-28), so
    # corpus_staging._stage_elephant_or_fallback took the plan's registered
    # fallback: a hash-partitioned held-out slice of r/socialskills. That is not
    # a neighbouring subreddit — it is half the sycophancy train pool
    # (corpus_staging.py:889-903, per_split = cap // 2 -> 8,000
    # relationship_advice + 8,000 socialskills; verified on the DV rows'
    # group_key prefixes). Train draws sha1(post_id) mod 10 in 0..8, this rung
    # draws bucket 9 (SYC_PARTITION_MOD=10, SYC_EVAL_BUCKET=9), so post ids are
    # disjoint and there is no leakage — but the shift is smaller than the other
    # OOD rungs'. Genuine OOD sycophancy rungs (SycophancyEval; an ELEPHANT
    # mirror if resolvable) are being staged; rebuild this row against them
    # rather than re-litigating the label.
    "aita": "OOD transfer",
    "wildchat_rung": "deployment-like traffic",
    PVSYNTH: "synthetic elicitation suite",
}
# IDENTITY. Every string traces to the corpus registry / staging code that built
# the rung, not to its slug:
#   evil train      corpus_registry REGISTRY[("evil","train")] — TrustAIRLab
#                   in-the-wild-jailbreak-prompts x forbidden_question_set
#   evil hhrt       Anthropic/hh-rlhf, subset red-team-attempts
#   evil toxicchat  lmsys/toxic-chat, subset flagged
#   syc train       HuggingFaceGECLM/REDDIT_submissions, splits
#                   relationship_advice + socialskills
#   syc aita        corpus_staging._stage_elephant_or_fallback fallback branch —
#                   held-out r/socialskills (sha1 mod-10 eval bucket)
#   hall train      mandarjoshi/trivia_qa, config rc.nocontext
#   hall nqopen     google-research-datasets/nq_open (validation)
#   hall simpleqa   basicv8vc/SimpleQA (test)
#   wildchat_rung   allenai/WildChat-1M, fresh conversations held out from the
#                   reused #1092 store by first-user-turn / final-query text
#   pvsynth         issue1739_pvsynth_pod — the Persona Vectors eval grid,
#                   5 instruction pairs x {pos,neg} x 20 held-out eval questions
SETTING_IDENTITY = {
    ("evil", "train"): "jailbreak prompts x forbidden Qs",
    ("evil", "hhrt"): "hh-rlhf red-team attempts",
    ("evil", "toxicchat"): "ToxicChat (flagged)",
    ("sycophancy", "train"): "Reddit personal-advice posts",
    ("sycophancy", "aita"): "held-out Reddit r/socialskills",
    ("hallucination", "train"): "TriviaQA (rc.nocontext)",
    ("hallucination", "nqopen"): "NQ-Open",
    ("hallucination", "simpleqa"): "SimpleQA",
}
for _b in BEHAVIORS:
    SETTING_IDENTITY[(_b, "wildchat_rung")] = "random WildChat conversations"
    SETTING_IDENTITY[(_b, PVSYNTH)] = "persona-vectors eval grid"

# Fail loud rather than mislabel: every plotted cell must have both parts.
for _b in BEHAVIORS:
    for _s in [*RUNGS[_b], PVSYNTH]:
        if _s not in SETTING_ROLE or (_b, _s) not in SETTING_IDENTITY:
            raise SystemExit(f"no role/identity label for {_b}/{_s}")

# Flat one-line form for the JSON sidecar (the figure uses the two-line form).
SETTING_LABEL = {
    (b, s): f"{SETTING_ROLE[s]} — {SETTING_IDENTITY[(b, s)]}"
    for b in BEHAVIORS
    for s in [*RUNGS[b], PVSYNTH]
}

# Hallucination's own rungs score the fabricated FRACTION rescaled x100, not the
# graded 0-100 trait rubric every other cell scores. Different construct: the
# figure separates the two groups so they are not read against each other.
RATE_CELLS = {("hallucination", r) for r in ("train", "nqopen", "simpleqa")}

BAR_COLOR = "#3B7DBF"
FLOOR_COLOR = "#B00020"
# 25 bins of width 4: keeps hallucination's discrete fabricated-fraction values
# (0, 20, ..., 100) in separate bins and cannot leak mass outside [0, 100].
BIN_EDGES = np.linspace(0.0, 100.0, 26)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0


# Three-part spread criterion + the pre-registered gate, verbatim from
# `issue1739_result1_spread_fig.py` (the script that published spread_stats.json).
FC_MASS_MAX = 0.90
R_YY_MIN = 0.50
MDC_CEILING_FRAC = 0.50
GATE2_SD_FLOOR = 10.0
GATE2_BOTTOM_BIN_EDGE = 10.0
GATE2_BOTTOM_FRAC_MAX = 0.80

# Raw judge payloads behind the five EVIL cells, per evaluation setting. `base`
# is the production wave; `rejudge` (higher max_tokens) REPLACES the draws of
# every rollout it re-judged. `parts` reassembles a line-split upload.
EVIL_JUDGE_SOURCES: dict[str, dict] = {
    "own_rungs": {
        "settings": ("train", "hhrt", "toxicchat"),
        "labeling": ER / "dv_dataset/evil/labeling.json",
        "parts": ROOT / "data/issue_1739/hf_dl/judge/issue1739_ctxmap/judge/evil",
        "manifest": "judge_raw_trait.json.split_manifest.json",
        "rejudge": ROOT
        / "data/issue_1739/hf_dl/judge/issue1739_ctxmap/judge/evil"
        / "judge_raw_trait_rejudge800_20260728T233651Z.json",
    },
    "wildchat_rung": {
        "settings": ("wildchat_rung",),
        "labeling": WT / "wildchat_rung/dv_dataset/evil/labeling.json",
        "base": WT / "wildchat_rung/judge/evil/judge_raw.json",
        "rejudge": WT / "wildchat_rung/judge/evil/judge_raw_rejudge.json",
    },
    PVSYNTH: {
        "settings": (PVSYNTH,),
        "labeling": ER / "pvsynth/dv_dataset/evil/labeling.json",
        "base": WT / "pvsynth/judge/evil/judge_raw.json",
        "rejudge": WT / "pvsynth/judge/evil/judge_raw_rejudge.json",
    },
}


def _rollout_vectors_graded(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    """Per-context rollout-score vectors, keyed by rung. Drops null rollouts."""
    out: dict[str, list[np.ndarray]] = {}
    for r in rows:
        if r.get("dv") is None:
            continue
        v = np.array([x for x in r["per_rollout_scores"].values() if x is not None], dtype=float)
        if v.size == 0:
            continue
        out.setdefault(r.get("rung"), []).append(v)
    return out


def _rollout_vectors_rate(rows: list[dict]) -> dict[str, list[np.ndarray]]:
    """Hallucination's own rungs: per-rollout fabricated indicators, x100."""
    out: dict[str, list[np.ndarray]] = {}
    for r in rows:
        n_dec = int(r["n_decided"])
        if n_dec <= 0:
            continue
        n_fab = int(r["counts"]["fabricated"])
        out.setdefault(r.get("rung"), []).append(
            np.array([100.0] * n_fab + [0.0] * (n_dec - n_fab), dtype=float)
        )
    return out


def cell_stats(per_ctx: list[np.ndarray]) -> dict:
    """Between-context spread, noise floor, reliability and detectability.

    Mirrors `issue1739_result1_spread_fig.cell_stats` (the source of the
    committed `spread_stats.json`) field for field; `load()` asserts the two
    agree on all 14 cells under the shipped `drop` coding, so a drift in this
    copy fails loudly instead of silently republishing different numbers.
    """
    ybar = np.array([v.mean() for v in per_ctx], dtype=float)
    m = np.array([v.size for v in per_ctx], dtype=float)
    s2 = np.array([v.var(ddof=1) if v.size > 1 else np.nan for v in per_ctx], dtype=float)
    usable = ~np.isnan(s2)
    n = int(ybar.size)
    sd = float(ybar.std(ddof=1)) if n > 1 else 0.0
    floor = float(np.sqrt(np.mean(s2[usable] / m[usable]))) if usable.any() else float("nan")
    r_yy = max(0.0, (sd**2 - floor**2) / sd**2) if sd > 0 and floor == floor else 0.0
    ceiling = float(np.sqrt(r_yy))
    vals, counts = np.unique(np.round(ybar, 9), return_counts=True)
    mdc = float(1.96 / np.sqrt(n - 3)) if n > 3 else float("nan")
    fc_mass = float(((ybar < 5.0) | (ybar > 95.0)).mean())
    bottom_frac = float((ybar < GATE2_BOTTOM_BIN_EDGE).mean())
    crit = {
        "floor_ceiling_mass_ok": fc_mass < FC_MASS_MAX,
        "reliability_ok": r_yy >= R_YY_MIN,
        "detectability_ok": (mdc <= MDC_CEILING_FRAC * ceiling) if mdc == mdc else False,
    }
    return {
        "n_contexts": n,
        "mean": float(ybar.mean()),
        "sd_between_context": sd,
        "noise_floor": floor,
        "r_yy": r_yy,
        "ceiling_sqrt_r_yy": ceiling,
        "floor_ceiling_mass": fc_mass,
        "tie_mass": float(counts.max() / n),
        "tie_value": float(vals[counts.argmax()]),
        "min_detectable_rho": mdc,
        "rollouts_per_context_median": float(np.median(m)),
        "n_contexts_single_rollout": int((~usable).sum()),
        "bottom_bin_frac": bottom_frac,
        "criterion": crit,
        "criterion_verdict": "PASS" if all(crit.values()) else "FAIL",
        "criterion_failing_clauses": [k for k, ok in crit.items() if not ok],
        "prereg_gate2": (
            "PASS" if (sd >= GATE2_SD_FLOOR and bottom_frac < GATE2_BOTTOM_FRAC_MAX) else "FAIL"
        ),
        "prereg_gate2_failing_clauses": [
            k
            for k, ok in (
                ("sd_floor", sd >= GATE2_SD_FLOOR),
                ("bottom_bin", bottom_frac < GATE2_BOTTOM_FRAC_MAX),
            )
            if not ok
        ],
    }


def load() -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str], dict]]:
    """Per-context rollout vectors + per-cell statistics under the `drop` coding.

    Raises if a recomputed statistic disagrees with the committed
    `spread_stats.json` row, so neither the figure nor the recode can drift from
    the numbers the original script published.
    """
    committed = {
        (c["behavior"], c["setting"]): c for c in json.loads(STATS_PATH.read_text())["cells"]
    }
    vectors: dict[tuple[str, str], list[np.ndarray]] = {}

    for b in BEHAVIORS:
        own = json.loads((ER / "dv_dataset" / b / "labeling.json").read_text())
        vecs = (
            _rollout_vectors_rate(own["rows"])
            if b == "hallucination"
            else _rollout_vectors_graded(own["rows"])
        )
        for rung, v in vecs.items():
            vectors[(b, rung)] = v
        for rung, path in (
            ("wildchat_rung", WT / "wildchat_rung/dv_dataset" / b / "labeling.json"),
            (PVSYNTH, ER / "pvsynth/dv_dataset" / b / "labeling.json"),
        ):
            d = json.loads(path.read_text())
            vectors[(b, rung)] = next(iter(_rollout_vectors_graded(d["rows"]).values()))

    stats: dict[tuple[str, str], dict] = {}
    for b in BEHAVIORS:
        for s in SETTINGS[b]:
            per_ctx, want = vectors.get((b, s)), committed.get((b, s))
            if per_ctx is None or want is None:
                raise SystemExit(f"missing cell {b}/{s}")
            got = cell_stats(per_ctx)
            for key, wv in want.items():
                if key not in got or isinstance(wv, str | dict | list):
                    continue
                if abs(float(got[key]) - float(wv)) > 1e-9:
                    raise SystemExit(f"{b}/{s}: recomputed {key}={got[key]} != committed {wv}")
            if got["criterion_verdict"] != want["criterion_verdict"]:
                raise SystemExit(f"{b}/{s}: recomputed verdict != committed")
            stats[(b, s)] = got
    return vectors, stats


# --- refusal-zero recode (EVIL ONLY) -----------------------------------------


def _draw_verdict(v: object) -> tuple[str, float | None]:
    """Classify one stored judge draw: valid score / REFUSAL / parse-error."""
    if not isinstance(v, dict) or v.get("error") or "score" not in v:
        return "parse_error", None
    score = v["score"]
    if isinstance(score, str):
        return ("refusal", None) if score.strip().upper() == "REFUSAL" else ("parse_error", None)
    return "valid", float(score)


def _load_draws(spec: dict) -> dict[str, list[object]]:
    """Per-rollout draw lists from a raw judge payload, re-judge applied.

    A re-judged rollout's draws are REPLACED wholesale (the re-judge wave stores
    its own draw keys under a fresh sequence id), matching how the shipped
    per-rollout scores were built.
    """

    def _all_scores(path: Path) -> dict:
        return json.loads(path.read_text())["all_scores"]

    if "parts" in spec:
        manifest = json.loads((spec["parts"] / spec["manifest"]).read_text())
        blob = b"".join((spec["parts"] / p).read_bytes() for p in manifest["parts"])
        if len(blob) != manifest["total_bytes"]:
            raise SystemExit(f"{spec['manifest']}: reassembled {len(blob)} bytes != manifest")
        base = json.loads(blob.decode())["all_scores"]
    else:
        base = _all_scores(spec["base"])

    draws: dict[str, list[object]] = defaultdict(list)
    for key, val in base.items():
        draws[key.split("__")[0]].append(val)
    rejudged: dict[str, list[object]] = defaultdict(list)
    for key, val in _all_scores(spec["rejudge"]).items():
        rejudged[key.split("__")[0]].append(val)
    draws.update(rejudged)
    return dict(draws)


def load_evil_refusal_zero(
    drop_vectors: dict[tuple[str, str], list[np.ndarray]],
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict]:
    """Rebuild the five evil cells with REFUSAL draws scored 0.

    Parse-error draws stay dropped. A rollout's score is the mean over its
    valid-plus-refusal-as-zero draws (None only when EVERY draw parse-errored);
    a context's DV is the mean over its scored rollouts, exactly as in the
    shipped pipeline. Gated: with refusals dropped the same reconstruction must
    reproduce every stored per-rollout score, so a drift in the raw payloads or
    in the merge cannot pass silently.
    """
    vectors: dict[tuple[str, str], list[np.ndarray]] = {}
    audit: dict = {"per_source": {}, "totals": defaultdict(int)}

    for source, spec in EVIL_JUDGE_SOURCES.items():
        draws = _load_draws(spec)
        rows = json.loads(spec["labeling"].read_text())["rows"]
        per_rung: dict[str, list[np.ndarray]] = defaultdict(list)
        tally = defaultdict(int)

        for row in rows:
            rung = row.get("rung") or spec["settings"][0]
            recoded: list[float] = []
            dropped: list[float | None] = []
            for kk, stored in row["per_rollout_scores"].items():
                item = f"{row['context_id']}_k{int(kk[1:]):02d}"
                if item not in draws:
                    raise SystemExit(f"{source}: no raw draws for rollout {item}")
                kinds = [_draw_verdict(v) for v in draws[item]]
                valid = [s for kind, s in kinds if kind == "valid"]
                n_ref = sum(1 for kind, _ in kinds if kind == "refusal")
                tally["n_draws"] += len(kinds)
                tally["n_draws_valid"] += len(valid)
                tally["n_draws_refusal"] += n_ref
                tally["n_draws_parse_error"] += len(kinds) - len(valid) - n_ref

                # Gate: refusals dropped must reproduce the stored score exactly.
                as_shipped = float(np.mean(valid)) if valid else None
                dropped.append(as_shipped)
                if (as_shipped is None) != (stored is None) or (
                    as_shipped is not None and abs(as_shipped - float(stored)) > 1e-9
                ):
                    raise SystemExit(
                        f"{source}: rollout {item} reconstructs to {as_shipped}, "
                        f"stored {stored} — raw payload does not match the shipped DV"
                    )

                if valid or n_ref:
                    recoded.append(float(np.sum(valid) / (len(valid) + n_ref)))
                    tally["n_rollouts_scored"] += 1
                    if n_ref and valid:
                        tally["n_rollouts_mixed"] += 1
                    elif n_ref:
                        tally["n_rollouts_all_refusal"] += 1
                else:
                    tally["n_rollouts_all_parse_error"] += 1
                tally["n_rollouts"] += 1

            had_dv = any(s is not None for s in dropped)
            if recoded:
                per_rung[rung].append(np.asarray(recoded, dtype=float))
                if not had_dv:
                    tally["n_contexts_recovered"] += 1
            elif had_dv:  # unreachable: recoding only ever adds scored rollouts
                raise SystemExit(f"{source}: context {row['context_id']} lost its DV")
            else:
                tally["n_contexts_still_empty"] += 1
            tally["n_contexts"] += 1

        for rung, per_ctx in per_rung.items():
            if rung not in spec["settings"]:
                raise SystemExit(f"{source}: unexpected rung {rung}")
            vectors[("evil", rung)] = per_ctx
        audit["per_source"][source] = dict(tally)
        for k, v in tally.items():
            audit["totals"][k] += v

    for setting in (*RUNGS["evil"], PVSYNTH):
        if ("evil", setting) not in vectors:
            raise SystemExit(f"refusal-zero recode produced no evil/{setting} cell")
        n_drop = len(drop_vectors[("evil", setting)])
        n_new = len(vectors[("evil", setting)])
        if n_new < n_drop:
            raise SystemExit(f"evil/{setting}: recode lost contexts ({n_new} < {n_drop})")
    audit["totals"] = dict(audit["totals"])
    return vectors, audit


def _rows(beh: str, settings: list[str]) -> tuple[list[float], float | None]:
    """Y positions, first setting at the top, with a gap where the DV changes."""
    cursor, raw, prev = 0.0, [], None
    boundary_after = None
    for i, s in enumerate(settings):
        is_rate = (beh, s) in RATE_CELLS
        if prev is not None and is_rate != prev:
            boundary_after = i - 1
            cursor += 0.7
        raw.append(cursor)
        cursor += 1.0
        prev = is_rate
    top = max(raw)
    pos = [top - p for p in raw]
    divider = None
    if boundary_after is not None:
        divider = (pos[boundary_after] + pos[boundary_after + 1]) / 2.0
    return pos, divider


def main(coding: str = "drop") -> None:
    vectors, stats = load()
    audit = None
    if coding == "refusal-zero":
        recoded, audit = load_evil_refusal_zero(vectors)
        vectors = {**vectors, **recoded}
        stats = {**stats, **{key: cell_stats(v) for key, v in recoded.items()}}
    ybar = {key: np.array([v.mean() for v in per_ctx]) for key, per_ctx in vectors.items()}
    slug = "spread_grid_simple" if coding == "drop" else "spread_grid_refusal_zero"
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    set_paper_style("blog")
    # The blog style enables constrained_layout, which ignores the explicit
    # subplots_adjust this figure needs; cleared before the figure is created so
    # no layout engine is ever attached.
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        len(BEHAVIORS),
        2,
        figsize=(13.0, 11.6),
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )

    for row, b in enumerate(BEHAVIORS):
        settings = SETTINGS[b]
        pos, divider = _rows(b, settings)
        axl, axr = axes[row]

        for p, s in zip(pos, settings, strict=True):
            dens, _ = np.histogram(ybar[(b, s)], bins=BIN_EDGES, range=(0.0, 100.0))
            h = dens / dens.max() * 0.42 if dens.max() else np.zeros_like(dens, dtype=float)
            axl.fill_between(
                BIN_CENTERS, p - h, p + h, step="mid", facecolor=BAR_COLOR, linewidth=0.0
            )
            c = stats[(b, s)]
            axr.barh(p, c["sd_between_context"], color=BAR_COLOR, height=0.6)
            axr.plot(
                [c["noise_floor"]] * 2,
                [p - 0.30, p + 0.30],
                color=FLOOR_COLOR,
                lw=1.9,
                solid_capstyle="butt",
                zorder=4,
            )

        for ax in (axl, axr):
            if divider is not None:
                ax.axhline(divider, color="#8A8A8A", lw=0.9, linestyle=(0, (4, 3)), zorder=1)
            ax.set_yticks(pos)
            ax.set_ylim(min(pos) - 0.7, max(pos) + 0.7)
        # Three lines per row: the ROLE the dataset plays, what it ACTUALLY is,
        # and the context count behind the distribution.
        axl.set_yticklabels(
            [
                f"{SETTING_ROLE[s]}\n{SETTING_IDENTITY[(b, s)]}\nn={stats[(b, s)]['n_contexts']:,}"
                for s in settings
            ],
            fontsize=8.0,
        )
        axr.set_yticklabels([])
        axl.set_xlim(-2, 102)
        axr.set_xlim(0, 46)
        axl.set_title(b, loc="left", fontsize=12)

    axes[-1][0].set_xlabel("per-context mean judged behavior score (0-100)")
    axes[-1][1].set_xlabel("between-context SD of that score (0-100 units)")
    fig.suptitle(
        "Judged behavior score in every evaluation setting: distribution and spread",
        x=0.008,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    axes[0][1].legend(
        handles=[Line2D([], [], color=FLOOR_COLOR, lw=1.9, label="noise floor of the SD")],
        loc="upper right",
        frameon=False,
        fontsize=9,
    )
    caption = (
        "Each row is labelled with the role it plays in the design, then the dataset it actually is. "
        "An in-distribution row is that behavior's FULL labeled\n"
        "pool, scored out-of-fold under 5 group-level folds — 'out-of-fold' is how the predictor is "
        "read, not a held-out data split. Hallucination's TriviaQA /\n"
        "NQ-Open / SimpleQA rows (above the dashed line) score fabrication rate x100, a different "
        "construct from the 0-100 trait rubric in every other row."
    )
    if coding == "refusal-zero":
        caption = (
            "Evil only: a judge REFUSAL verdict (the evaluated model declined the query) is scored 0 "
            "rather than dropped; sycophancy and hallucination are unchanged.\n" + caption
        )
    fig.text(0.008, 0.010, caption, ha="left", va="bottom", fontsize=8.0, color="#5A5A5A")
    # left: the three-line role/identity/n tick labels need a wider margin.
    fig.subplots_adjust(left=0.205, right=0.985, top=0.945, bottom=0.115, hspace=0.30, wspace=0.05)
    savefig_paper(fig, slug, dir=OUT_FIG)
    plt.close(fig)
    print(f"wrote {OUT_FIG / f'{slug}.png'}")
    print(f"parity vs committed spread_stats.json: {len(stats)}/{len(stats)} cells reconciled")

    if coding == "refusal-zero":
        committed = {
            (c["behavior"], c["setting"]): c for c in json.loads(STATS_PATH.read_text())["cells"]
        }
        table = []
        for b in BEHAVIORS:
            for s in SETTINGS[b]:
                row = {
                    "behavior": b,
                    "setting": s,
                    "setting_label": SETTING_LABEL[(b, s)].replace("\n", " "),
                    "setting_role": SETTING_ROLE[s],
                    "setting_identity": SETTING_IDENTITY[(b, s)],
                }
                row["coding"] = "refusal_zero" if b == "evil" else "drop (unchanged)"
                row |= stats[(b, s)]
                if b == "evil":
                    row["drop_coding"] = {
                        k: committed[(b, s)][k]
                        for k in (
                            "n_contexts",
                            "mean",
                            "sd_between_context",
                            "noise_floor",
                            "r_yy",
                            "ceiling_sqrt_r_yy",
                            "floor_ceiling_mass",
                            "tie_mass",
                            "min_detectable_rho",
                            "criterion_verdict",
                            "criterion_failing_clauses",
                        )
                    }
                table.append(row)
        OUT_NUM.mkdir(parents=True, exist_ok=True)
        out = OUT_NUM / "spread_stats_refusal_zero.json"
        out.write_text(
            json.dumps(
                {
                    "coding": "refusal_zero",
                    "scope": (
                        "EVIL ONLY, REFUSAL channel only. A judge return of REFUSAL means the "
                        "EVALUATED model declined the query; the persona-vectors rubric instructs "
                        "the judge to emit it, so it is a well-formed verdict rather than a failed "
                        "measurement. Scored 0 here (refusing is the least-evil response). "
                        "Parse-error draws stay dropped in both codings; sycophancy and "
                        "hallucination are byte-identical to spread_stats.json."
                    ),
                    "source": "per-draw verdicts from the raw judge payloads in EVIL_JUDGE_SOURCES",
                    "reconstruction_gate": (
                        "with refusals dropped, every stored per-rollout score in the three evil "
                        "labeling.json files is reproduced exactly from the raw draws"
                    ),
                    "recode_audit": audit,
                    "criterion": {
                        "floor_ceiling_mass_max": FC_MASS_MAX,
                        "r_yy_min": R_YY_MIN,
                        "min_detectable_rho_max_frac_of_ceiling": MDC_CEILING_FRAC,
                    },
                    "cells": table,
                },
                indent=1,
            )
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coding", choices=("drop", "refusal-zero"), default="drop")
    main(**{"coding": ap.parse_args().coding})
