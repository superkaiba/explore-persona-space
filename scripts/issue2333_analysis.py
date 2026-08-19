#!/usr/bin/env python
"""Issue #2333 VM-side analysis — snowball test (plan §3/§5/§6).

Phases:
  s2-ce-derive  P6 first step (plan §3 Row-coverage): join the banked #2094
                fu1 conf1 raw completions with their judge scores
                (judge_raw_fu1 stage2 waves @ PIN_FU1), recompute per-pair
                anchor-normalized F for the (matched_query, ce, joint_all,
                replace, A) steered + null arms using the committed #2094
                anchors, and write
                eval_results/issue_2333/inputs/s2_ce_control_perpair.jsonl.
                FAILS LOUD unless the re-derived well-separated family means
                reproduce steered 0.512 / null 0.097 within ±0.01 (the
                vendored fu1_conf1_confirmation.json aggregate cross-check)
                and the well-separated/excluded split is exactly 10/5.
  f-tables      per model: per-(pair x arm x variant) F_beh (+ continuation
                companion, + F_act where V_a stores staged) + same-wave ce
                calibration F + q35 fresh anchors/ce tables.
  stats         registered reads (plan §3/§6): per-(pair-set x arm) paired
                diffs (steered - null), exact Wilcoxon + Holm (m=12 per
                model x pair-set), pair-clustered bootstrap CIs
                (B=10,000 seed 23330, `bootstrap_family_means_batched`),
                recovery ratios R_k, the registered D3 CI (same-wave ce
                primary / banked ce comparison), and the FOUR-BRANCH lattice
                on prefill-3 (scheme (a) = med confirmatory; scheme (b)
                labels prefixed "natural-opening").

Cell sets (plan §4.3 item 4, q35_language_snowball): ``--cell-set q35lang``
routes f-tables/stats to ``eval_results/issue_2333/q35_language_snowball/
f_metrics/``, GATES the banked-#2162/#2094 calib joins (main-only), and adds
the REGISTERED §3 machinery — null-adjusted D3_net / R3_net with JOINT
pair-clustered resampling (a drawn pair carries its arm, null, and ce rows
together; B=10,000 seed 23330), the carrier-level cluster-bootstrap companion
(resample carriers; a drawn carrier carries ALL its cells, directed pairs,
and anchors), continuation-only D3_net/R3_net + separation for every prefill
arm, and the control-health + both-cells-floor preconditions
(``control-unresolved`` / ``untestable-causal`` routing) — plus raw D3/R₃ as
descriptive companions and the judge-offset table vs #2329's stored per-pair
anchor deltas. The parent path (``--cell-set main``) is behaviorally
byte-identical.

REUSES: `issue2094_analysis.bootstrap_family_means_batched`,
`issue2162_analysis` {_wilcoxon_exact_p, holm, io helpers, constants},
`issue2333_judge` item-id builders (the judge/analysis join contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2094_judge as J94  # noqa: E402
import issue2162_analysis as A62  # noqa: E402
import issue2162_judge as J62  # noqa: E402
import issue2333_judge as J33  # noqa: E402
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402

from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402

logger = logging.getLogger("issue2333.analysis")

COHERENCE_THRESHOLD = A62.COHERENCE_THRESHOLD  # 60.0
SEPARATION_BAR = A62.SEPARATION_BAR  # 0.5
S1_SURVIVAL_FLOOR = A62.SURVIVAL_FLOOR  # 12 (per-cell parent floor)
S2_SURVIVAL_FLOOR = 5
HOLM_ALPHA = A62.HOLM_ALPHA  # 0.05
BOOT_B = 10_000
BOOT_SEED = C.BOOTSTRAP_SEED  # 23330
CROSSCHECK_TOL = 0.01
D3_SHARE = 0.5  # plan §3: majority-share criterion

INPUTS_DIR = Path("eval_results/issue_2333/inputs")
FMETRICS_DIR = Path("eval_results/issue_2333/f_metrics")
A2162_ANCHORS = Path("eval_results/issue_2162/f_metrics/anchors.jsonl")
A2162_F_CELLS = Path("eval_results/issue_2162/f_metrics/f_cells.jsonl")
A2094_ANCHORS = Path("eval_results/issue_2094/f_metrics/anchors.jsonl")
VENDORED_CONF1 = INPUTS_DIR / "fu1_conf1_confirmation.json"
CONF1_FAMILY = "matched_query|ce|joint_all|replace|A|f_beh_prefix"

# q35_language_snowball cell set (plan §3 row-coverage / §4.3 item 4).
Q35LANG_DIR = Path("eval_results/issue_2333/q35_language_snowball")
Q35LANG_INPUTS = Path(C.Q35LANG_INPUTS_DIR)
I2329_F_CELLS = Q35LANG_INPUTS / "i2329_lang_f_cells.jsonl"  # 144 rows; consume slot=="ce" ONLY
I2329_NULL_CELLS = Q35LANG_INPUTS / "i2329_lang_null_shuffled_cells.jsonl"
I2329_ANCHORS = Q35LANG_INPUTS / "i2329_lang_anchors.jsonl"  # 72 per-pair delta rows
D3_NET_SHARE = D3_SHARE  # §3: D3_net = (arm_st − arm_nu) − 0.5 × (ce_st − ce_nu)


def _fmetrics_dir(tag: str, cell_set: str) -> Path:
    """Default out-dir: the parent per-tag dir, or the q35lang namespace dir
    (plan §3 row-coverage: .../q35_language_snowball/f_metrics/, NO tag leaf)."""
    return (Q35LANG_DIR / "f_metrics") if cell_set == "q35lang" else (FMETRICS_DIR / tag)


_FU1_SCORE_FILES = (
    "coherence.stage2.scores.jsonl",
    "fp-bare.stage2.scores.jsonl",
    "fp-conv.stage2.scores.jsonl",
    "fp-persona.stage2.scores.jsonl",
)


def _item_id(tag: str, key: str) -> str:
    return tag + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _ci(draws: np.ndarray) -> tuple[float, float]:
    """95% percentile CI over bootstrap draws (NaN-aware)."""
    return (float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5)))


# ── phase: s2-ce-derive ───────────────────────────────────────────────


def _stage_fu1_inputs(stage_dir: Path) -> tuple[list[Path], list[Path]]:
    """Stage the fu1 conf1 rollouts + stage2 judge scores @ PIN_FU1."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    conf_dir = stage_dir / "conf1"
    scores_dir = stage_dir / "scores"
    conf_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    conf_paths, score_paths = [], []
    for name in (
        "fu1_fu1__matched_query__ce__joint_all__replace__A__steered.jsonl",
        "fu1_fu1__matched_query__ce__joint_all__replace__A__null.jsonl",
    ):
        target = conf_dir / name
        if not target.is_file():
            got = hub.retry_transient(
                lambda fn=f"{C.FU1_CONF1_PREFIX}/{name}": hf_hub_download(
                    repo_id=C.DATA_REPO,
                    repo_type="dataset",
                    filename=fn,
                    revision=C.PIN_FU1,
                    local_dir=conf_dir / "_dl",
                ),
                what=f"fu1 conf1 {name}",
            )
            Path(got).replace(target)
        conf_paths.append(target)
    for name in _FU1_SCORE_FILES:
        target = scores_dir / name
        if not target.is_file():
            got = hub.retry_transient(
                lambda fn=f"{C.FU1_JUDGE_RAW_PREFIX}/scores/{name}": hf_hub_download(
                    repo_id=C.DATA_REPO,
                    repo_type="dataset",
                    filename=fn,
                    revision=C.PIN_FU1,
                    local_dir=scores_dir / "_dl",
                ),
                what=f"fu1 stage2 scores {name}",
            )
            Path(got).replace(target)
        score_paths.append(target)
    return conf_paths, score_paths


def load_2094_anchors(kind: str = "prefix") -> dict[str, dict]:
    """pair_id -> {floor, ceiling, separation} for matched_query pairs."""
    out = {}
    for r in A62._iter_jsonl(A2094_ANCHORS):
        if r["setting"] == "matched_query" and r["kind"] == kind:
            out[r["pair_id"]] = {
                "floor": r["floor"]["mean"],
                "ceiling": r["ceiling"]["mean"],
                "separation": r["separation"],
            }
    assert len(out) == 15, len(out)
    return out


def phase_s2_ce_derive(args: argparse.Namespace) -> int:
    conf_paths, score_paths = _stage_fu1_inputs(args.stage_dir)
    scores: dict[str, float | None] = {}
    for p in score_paths:
        for r in A62._iter_jsonl(p):
            scores[r["item_id"]] = r["score"]
    anchors = load_2094_anchors()
    wellsep = sorted(
        pid
        for pid, a in anchors.items()
        if a["separation"] is not None and abs(a["separation"]) >= SEPARATION_BAR
    )
    excluded = sorted(set(anchors) - set(wellsep))
    if not (len(wellsep) == C.S2_WELLSEP_N and len(excluded) == C.S2_EXCLUDED_N):
        raise RuntimeError(
            f"S2 well-separated split drifted: {len(wellsep)} wellsep / {len(excluded)} "
            f"excluded (expected {C.S2_WELLSEP_N}/{C.S2_EXCLUDED_N})"
        )

    rows_out: list[dict] = []
    means: dict[str, float] = {}
    for arm, conf_path in zip(("steered", "null"), conf_paths, strict=True):
        per_pair: dict[str, list[float]] = defaultdict(list)
        n_incoherent = 0
        n_judge_missing = 0
        for r in A62._iter_jsonl(conf_path):
            # fu1 judged the conf1 rollouts in its STAGE2 wave with keys
            # `coh|s2|{pid}|{cell}|{draw}` / `beh|s2|...|prefix|{side}`
            # (issue2094_judge._stage2_key; join validated 2026-08-16:
            # reproduces steered 0.512 / null 0.0969 exactly).
            pid, cell, draw = r["pair_id"], r["cell"], r["draw"]
            coh = scores.get(_item_id("c", f"coh|s2|{pid}|{cell}|{draw}"))
            if coh is None or coh <= COHERENCE_THRESHOLD:
                n_incoherent += 1
                continue
            sa = scores.get(_item_id("s", f"beh|s2|{pid}|{cell}|{draw}|prefix|a"))
            sb = scores.get(_item_id("s", f"beh|s2|{pid}|{cell}|{draw}|prefix|b"))
            if sa is None or sb is None:
                n_judge_missing += 1
                continue
            per_pair[pid].append((sb - sa) / 100.0)
        fs: dict[str, float] = {}
        for pid in sorted(anchors):
            ds = per_pair.get(pid, [])
            a = anchors[pid]
            denom = a["ceiling"] - a["floor"]
            f = ((_mean(ds) - a["floor"]) / denom) if ds and abs(denom) > 1e-9 else None
            if f is not None:
                fs[pid] = f
            rows_out.append(
                {
                    "pair_id": pid,
                    "arm": arm,
                    "f_beh": f,
                    "delta_patched_mean": _mean(ds),
                    "n_draws": len(ds),
                    "separation": a["separation"],
                    "wellsep": pid in wellsep,
                    "source": "fu1-conf1-rederive",
                }
            )
        ws_mean = _mean([fs[p] for p in wellsep if p in fs])
        assert ws_mean is not None
        means[arm] = ws_mean
        logger.info(
            "[s2-ce-derive] %s: wellsep mean F=%.4f (incoherent=%d judge_missing=%d)",
            arm,
            ws_mean,
            n_incoherent,
            n_judge_missing,
        )

    vend = json.loads(VENDORED_CONF1.read_text(encoding="utf-8"))
    fam = next(f for f in vend["families"] if f["family"] == CONF1_FAMILY)
    checks = {
        arm: {
            "rederived": means[arm],
            "vendored": fam[arm]["observed_mean"],
            "abs_diff": abs(means[arm] - fam[arm]["observed_mean"]),
        }
        for arm in ("steered", "null")
    }
    for arm, rec in checks.items():
        if rec["abs_diff"] > CROSSCHECK_TOL:
            raise RuntimeError(
                f"s2-ce-derive cross-check FAILED for {arm}: re-derived "
                f"{rec['rederived']:.4f} vs vendored {rec['vendored']:.4f} "
                f"(|diff| {rec['abs_diff']:.4f} > {CROSSCHECK_TOL})"
            )
    out_path = INPUTS_DIR / "s2_ce_control_perpair.jsonl"
    A62._write_jsonl_atomic(out_path, rows_out)
    A62._write_json_atomic(
        INPUTS_DIR / "s2_ce_derive_meta.json",
        {
            "passed": True,
            "crosscheck": checks,
            "tolerance": CROSSCHECK_TOL,
            "wellsep_pair_ids": wellsep,
            "excluded_pair_ids": excluded,
            "pins": {"fu1": C.PIN_FU1, "family": CONF1_FAMILY},
            "repro": A62._repro() if hasattr(A62, "_repro") else J94._repro(),
        },
    )
    logger.info("[s2-ce-derive] wrote %s (%d rows) — cross-check PASS", out_path, len(rows_out))
    return 0


# ── phase: f-tables ───────────────────────────────────────────────────


def _load_scores(scores_dir: Path, suffixes: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for suffix in suffixes:
        files = sorted(scores_dir.glob(f"*.{suffix}.scores.jsonl"))
        assert files, f"no {suffix} score files under {scores_dir}"
        for f in files:
            for row in A62._iter_jsonl(f):
                out[row["item_id"]] = row["score"]
    return out


def _banked_anchor_deltas(pairs: list) -> dict[str, dict]:
    """pair_id -> {floor, ceiling, separation} from the COMMITTED parents'
    anchors tables (q25 path: S1 from #2162, S2 from #2094)."""
    out: dict[str, dict] = {}
    by_2162 = {r["pair_id"]: r for r in A62._iter_jsonl(A2162_ANCHORS)}
    a_2094 = load_2094_anchors()
    for p in pairs:
        pid = p.pair_id
        if J33.pair_set_of(p) == "s1":
            r = by_2162[pid]
            out[pid] = {
                "floor": r["delta_floor_mean"],
                "ceiling": r["delta_ceiling_mean"],
                "separation": r["separation"],
            }
        else:
            out[pid] = dict(a_2094[pid])
    return out


def _fresh_anchor_deltas(
    pairs: list, anchor_rows: list[dict], scores: dict[str, float | None]
) -> dict[str, dict]:
    """q35 path: per-pair floor/ceiling deltas from the FRESH anchor waves."""
    draws_by_ctx: dict[str, list[int]] = defaultdict(list)
    text_rows: dict[tuple[str, int], dict] = {}
    for r in anchor_rows:
        draws_by_ctx[r["context_id"]].append(r["draw"])
        text_rows[(r["context_id"], r["draw"])] = r
    out: dict[str, dict] = {}
    for p in pairs:
        cores = J33.pair_rubric_cores_2333(p)
        rid_a, rid_b = (J62.rubric_core_id(c) for c in cores)
        deltas: dict[str, list[float]] = {"floor": [], "ceiling": []}
        for name, ctx in (("floor", p.a), ("ceiling", p.b)):
            for d in draws_by_ctx.get(ctx, []):
                coh = scores.get(J33.anchor_coherence_id(ctx, d))
                if coh is None or coh <= COHERENCE_THRESHOLD:
                    continue
                sa = scores.get(J62.anchor_unit_id(ctx, d, rid_a))
                sb = scores.get(J62.anchor_unit_id(ctx, d, rid_b))
                if sa is None or sb is None:
                    continue
                deltas[name].append((sb - sa) / 100.0)
        fl, ce_ = _mean(deltas["floor"]), _mean(deltas["ceiling"])
        out[p.pair_id] = {
            "floor": fl,
            "ceiling": ce_,
            "separation": (ce_ - fl) if fl is not None and ce_ is not None else None,
            "n_floor": len(deltas["floor"]),
            "n_ceiling": len(deltas["ceiling"]),
        }
    return out


def _f_from_rows(
    rows: list[dict],
    tag: str,
    scores: dict[str, float | None],
    anchor: dict,
    answer_note: str = "response",
) -> dict:
    """Per-(pair x arm) F_beh from judged rows sharing one (pair, arm) cell."""
    deltas: list[float] = []
    n_coherent = 0
    n_cap = 0
    for row in rows:
        n_cap += int(row.get("cap_hit", False))
        coh = scores.get(_item_id("c", J33.coherence_key(tag if tag != "n" else "g", row)))
        if coh is None or coh <= COHERENCE_THRESHOLD:
            continue
        n_coherent += 1
        sa = scores.get(_item_id(tag, J33.behavior_key(tag, row, "a")))
        sb = scores.get(_item_id(tag, J33.behavior_key(tag, row, "b")))
        if sa is None or sb is None:
            continue
        deltas.append((sb - sa) / 100.0)
    f_beh = None
    dp = _mean(deltas)
    fl, ce_ = anchor.get("floor"), anchor.get("ceiling")
    if dp is not None and fl is not None and ce_ is not None and abs(ce_ - fl) > 1e-9:
        f_beh = (dp - fl) / (ce_ - fl)
    return {
        "f_beh": f_beh,
        "delta_patched_mean": dp,
        "n_rows": len(rows),
        "n_coherent": n_coherent,
        "n_scored": len(deltas),
        "n_cap_hit": n_cap,
        "answer_note": answer_note,
    }


def _load_va_store(va_dir: Path, read_layer: int, cell_set: str = "main") -> dict[str, np.ndarray]:
    """va_key -> (H,) fp32 at the read layer. TWO shard schemas:

    - #2333 driver shards (grid / q35 anchors): flat ``{va_key: (L, H)}``.
    - Banked STRUCTURED anchor stores (#2162 ``va_anchors_*_w*.pt`` /
      #2094 ``va_anchors.pt`` — the q25 F_act floor/ceiling inputs; and the
      reused #2329 anchor va shards on q35lang): ``{layers, index, va_span,
      empty_rows, ...}`` (observed writer schemas: issue2162_run.py L1725 /
      issue2094_run.py L1743 / issue2329_run.py ``_run_anchor_batch``);
      converted to ``{context_id}|anchor|d{draw}`` keys, ``empty_rows``
      indices (empty completions) skipped, read layer resolved through the
      store's OWN ``layers`` list. On q35lang the structured branch MUST go
      through ``issue2333_run.load_va_anchor_shard_2329`` (the mandated
      staged-consumer-open, artifact-reuse (h)(iv)) — its fail-loud
      shape/pooling asserts run before any conversion.
    """
    import torch

    out: dict[str, np.ndarray] = {}
    for shard in sorted(va_dir.glob("*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        assert isinstance(payload, dict), shard
        if "va_span" in payload and "index" in payload:
            if cell_set == "q35lang":
                import issue2333_run as R33  # lazy: pulls torch + extraction

                payload = R33.load_va_anchor_shard_2329(shard)
            layer_idx = list(payload["layers"]).index(read_layer)
            empty = set(payload.get("empty_rows") or [])
            span = payload["va_span"]
            assert span.shape[0] == len(payload["index"]), (span.shape, len(payload["index"]))
            for i, rec in enumerate(payload["index"]):
                if i in empty:
                    continue
                out[f"{rec['context_id']}|anchor|d{rec['draw']}"] = (
                    span[i, layer_idx].float().numpy()
                )
            continue
        for key, t in payload.items():
            out[key] = t[read_layer].float().numpy()
    return out


def _write_judge_offset_2329(args: argparse.Namespace, out_dir: Path, anchors: dict) -> None:
    """Judge-offset table (plan §6 calibration row): OUR re-judged per-pair
    floor/ceiling deltas vs #2329's STORED per-pair deltas. On the reuse path
    the anchor TEXTS are IDENTICAL, so the offset is pure instrument drift;
    on selfgen it also carries anchors-distribution drift (labeled). A
    near-zero-variance table is the cache-contamination tell (a cache-served
    #2329-era entry would reproduce the stored score exactly) — flagged, and
    structurally prevented by the judge's fresh per-cell-set cache partition."""
    mode = J33.anchor_mode(args.anchors_dir, "q35lang")
    banked = {r["pair_id"]: r for r in A62._iter_jsonl(I2329_ANCHORS)}
    if set(banked) != set(anchors):
        raise RuntimeError(
            f"anchor pair sets diverge: banked-only={sorted(set(banked) - set(anchors))[:4]} "
            f"ours-only={sorted(set(anchors) - set(banked))[:4]}"
        )
    rows_off: list[dict] = []
    n_missing = 0
    for pid in sorted(anchors):
        b = banked[pid]
        for side, ours_key, banked_key in (
            ("floor", "floor", "delta_floor_mean"),
            ("ceiling", "ceiling", "delta_ceiling_mean"),
        ):
            ours = anchors[pid].get(ours_key)
            theirs = b.get(banked_key)
            if ours is None or theirs is None:
                n_missing += 1
                continue
            rows_off.append(
                {
                    "pair_id": pid,
                    "side": side,
                    "ours": ours,
                    "banked": theirs,
                    "offset": ours - theirs,
                }
            )
    offs = np.array([r["offset"] for r in rows_off], dtype=np.float64)
    sd = float(np.std(offs, ddof=1)) if offs.size > 1 else None
    A62._write_jsonl_atomic(out_dir / "judge_offset_rows.jsonl", rows_off)
    A62._write_json_atomic(
        out_dir / "judge_offset_2329.json",
        {
            "mode": mode,
            "n_rows": len(rows_off),
            "n_missing_sides": n_missing,
            "mean_offset": float(offs.mean()) if offs.size else None,
            "sd": sd,
            "se": (sd / math.sqrt(offs.size)) if sd is not None and offs.size else None,
            "offset_semantics": (
                "same-wave re-judge of the IDENTICAL #2329 anchor completions minus #2329's "
                "stored per-pair floor/ceiling deltas (delta units, judge-score/100) — pure "
                "instrument drift"
                if mode == "reused_2329"
                else "SELFGEN anchors — texts differ from #2329's, so the offset carries "
                "anchors-distribution drift ON TOP of instrument drift (not a pure "
                "instrument read)"
            ),
            # Contamination tell: identical stored scores served from a stale
            # cache would zero the variance (plan §4.3 item 3).
            "near_zero_variance_flag": bool(sd is not None and offs.size >= 8 and sd < 1e-9),
            "pins": {"p2329_data": C.PIN_2329_DATA, "ref_2329_branch": C.REF_2329_BRANCH},
        },
    )
    logger.info(
        "[f-tables] judge-offset vs #2329: n=%d mean=%s sd=%s mode=%s",
        len(rows_off),
        f"{offs.mean():+.4f}" if offs.size else "n/a",
        f"{sd:.4f}" if sd is not None else "n/a",
        mode,
    )


def phase_f_tables(args: argparse.Namespace) -> int:
    tag = args.model_tag
    # getattr-defaulted: pre-existing callers (tests) hand-build Namespaces
    # without cell_set — the parent path must keep accepting them.
    cell_set = getattr(args, "cell_set", "main")
    s1_pairs, s2_pairs = J33.build_pair_universe(cell_set)
    pairs = [*s1_pairs, *s2_pairs]
    pairs_by_id = {p.pair_id: p for p in pairs}
    out_dir = args.out_dir or _fmetrics_dir(tag, cell_set)

    suffixes = ("grid", "anchors") if tag == "q35" else ("grid",)
    scores = _load_scores(args.scores_dir, suffixes)
    grid_rows = J33.load_grid_rows(args.rollouts_dir, cell_set=cell_set)

    if tag == "q35":
        ctx_ids = {cid for p in pairs for cid in (p.a, p.b)}
        anchor_rows = J33.load_anchor_rows_2333(args.anchors_dir, cell_set, ctx_ids)
        anchors = _fresh_anchor_deltas(pairs, anchor_rows, scores)
        A62._write_jsonl_atomic(
            out_dir / "anchors.jsonl",
            [{"pair_id": pid, **rec} for pid, rec in sorted(anchors.items())],
        )
    else:
        anchors = _banked_anchor_deltas(pairs)

    read_layer = C.MODELS[tag]["read_layer"]
    va = _load_va_store(args.va_dir, read_layer) if args.va_dir else {}
    anchor_va = (
        _load_va_store(args.anchor_va_dir, read_layer, cell_set) if args.anchor_va_dir else {}
    )
    if not va:
        logger.warning("[f-tables] no --va-dir staged: f_act = None everywhere (secondary DV)")

    def _expects_va(r: dict) -> bool:
        """Which rows MUST have a V_a entry: prefill rows always (span covers
        the donor ids even at an empty continuation); patch/ce rows whenever
        the completion is non-empty (the driver skips V_a on empty gen_ids)."""
        if r.get("kind") == "prefill":
            return True
        return int(r.get("n_completion_tokens", 0)) > 0

    def _f_act(rows: list[dict], p) -> float | None:
        if not va or not anchor_va:
            return None
        import torch

        from explore_persona_space.experiments.issue2094 import fmetrics as FM

        expected = [r["va_key"] for r in rows if _expects_va(r)]
        missing = [k for k in expected if k not in va]
        if missing:
            # Fail LOUD (r1: the silent `in va` filter turned a stale/partial
            # --va-dir staging into quietly-thinner F_act means).
            raise RuntimeError(
                f"va store missing {len(missing)}/{len(expected)} expected keys "
                f"(e.g. {missing[:3]}) — stale or partially-staged --va-dir"
            )
        patched = [va[k] for k in expected]
        floor = [v for k, v in anchor_va.items() if k.startswith(f"{p.a}|anchor|")]
        ceil = [v for k, v in anchor_va.items() if k.startswith(f"{p.b}|anchor|")]
        if not patched or len(floor) < 2 or not ceil:
            return None
        res = FM.f_act(
            torch.tensor(np.stack(patched).mean(axis=0)),
            torch.tensor(np.stack(floor)),
            torch.tensor(np.stack(ceil)),
        )
        return float(res.f_act)

    # Fresh grid cells: one row per (pair x arm_slug x variant).
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in grid_rows:
        by_cell[(r["pair_id"], r["arm_slug"], r["variant"])].append(r)
    steered_rows, null_rows = [], []
    for (pid, slug, variant), rows in sorted(by_cell.items()):
        p = pairs_by_id[pid]
        kind, k, scheme = C.parse_arm(slug)
        rec = {
            "pair_id": pid,
            "cell": rows[0]["cell"],
            "set": J33.pair_set_of(p),
            "arm_slug": slug,
            "kind": kind,
            "k": k,
            "scheme": scheme,
            "variant": variant,
            "donor_pair_id": rows[0].get("donor_pair_id"),
            "separation": anchors[pid].get("separation"),
            **_f_from_rows(rows, "g", scores, anchors[pid]),
            "f_act": _f_act(rows, p),
        }
        if kind == "prefill":
            cont = _f_from_rows(rows, "n", scores, anchors[pid], answer_note="continuation")
            rec["f_beh_continuation"] = cont["f_beh"]
            rec["n_scored_continuation"] = cont["n_scored"]
        (steered_rows if variant == "steered" else null_rows).append(rec)
    A62._write_jsonl_atomic(out_dir / "f_cells.jsonl", steered_rows)
    A62._write_jsonl_atomic(out_dir / "null_cells.jsonl", null_rows)

    # q35 fresh ce-control cells.
    if tag == "q35":
        ce_rows_all = J33.load_ce_rows(args.rollouts_dir, cell_set=cell_set)
        by_ce: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in ce_rows_all:
            by_ce[(r["pair_id"], r["variant"])].append(r)
        ce_out = [
            {
                "pair_id": pid,
                "set": J33.pair_set_of(pairs_by_id[pid]),
                "variant": variant,
                "separation": anchors[pid].get("separation"),
                **_f_from_rows(rows, "e", scores, anchors[pid]),
                # F_act on the fresh ce rows too (the F_act recovery
                # denominator — r1 blocker f-act-downstream-missing).
                "f_act": _f_act(rows, pairs_by_id[pid]),
            }
            for (pid, variant), rows in sorted(by_ce.items())
        ]
        A62._write_jsonl_atomic(out_dir / "ce_cells.jsonl", ce_out)

    if cell_set != "main":
        # Banked Qwen2.5 calib legs are main-only (gated — plan §4.3 item 3);
        # the q35lang same-wave instrument anchor is the #2329 ANCHORS-REJUDGE
        # wave, reported as the judge-offset table below.
        _write_judge_offset_2329(args, out_dir, anchors)
        logger.info(
            "[f-tables] %s/%s: steered=%d null=%d (calib legs gated)",
            tag,
            cell_set,
            len(steered_rows),
            len(null_rows),
        )
        return 0

    # SAME-WAVE ce calibration cells (banked raws re-judged in THIS wave).
    calib_s1 = J33.load_calib_s1(args.calib_dir)
    calib_s2 = J33.load_calib_s2(args.calib_dir)
    calib_out: list[dict] = []
    by_k: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in calib_s1:
        by_k[(r["pair_id"], r["arm"])].append(r)
    for (pid, arm), rows in sorted(by_k.items()):
        calib_out.append(
            {
                "pair_id": pid,
                "set": "s1",
                "arm": arm,
                "separation": anchors[pid].get("separation"),
                **_f_from_rows(rows, "k", scores, anchors[pid]),
            }
        )
    by_m: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in calib_s2:
        by_m[(r["pair_id"], r["arm"])].append(r)
    for (pid, arm), rows in sorted(by_m.items()):
        calib_out.append(
            {
                "pair_id": pid,
                "set": "s2",
                "arm": arm,
                "separation": anchors[pid].get("separation"),
                **_f_from_rows(rows, "m", scores, anchors[pid]),
            }
        )
    A62._write_jsonl_atomic(out_dir / "calib_cells.jsonl", calib_out)

    # Same-wave-vs-banked judge offset (plan §6: reported + propagated to D3).
    banked_s1 = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(A2162_F_CELLS)
        if r["slot"] == "ce" and r["cell"] in C.S1_CELLS and r["f_beh"] is not None
    }
    banked_s2 = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(INPUTS_DIR / "s2_ce_control_perpair.jsonl")
        if r["arm"] == "steered" and r["f_beh"] is not None
    }
    offs: dict[str, list[float]] = {"s1": [], "s2": []}
    for rec in calib_out:
        banked = (banked_s1 if rec["set"] == "s1" else banked_s2).get(rec["pair_id"])
        arm_ok = rec["arm"] == "steered"
        if arm_ok and banked is not None and rec["f_beh"] is not None:
            offs[rec["set"]].append(rec["f_beh"] - banked)
    A62._write_json_atomic(
        out_dir / "calib_offset.json",
        {
            "model_tag": tag,
            # r1 Minor: on q35 the calibration rows are q25-GENERATED text
            # re-judged in the q35 wave against q25-banked F — a cross-model
            # JUDGE-INSTRUMENT offset (mixed normalization), never a q35
            # behavior read; label it so downstream tables cannot silently
            # treat it as within-model.
            "offset_semantics": (
                "same-wave re-judge of q25-generated banked ce text minus the banked "
                "q25 F values; instrument offset only"
                + (" — CROSS-MODEL wave (q35 instrument on q25 text)" if tag == "q35" else "")
            ),
            **{
                s: {
                    "mean_offset": _mean(v),
                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    "n_pairs": len(v),
                }
                for s, v in offs.items()
            },
        },
    )
    logger.info(
        "[f-tables] %s: steered=%d null=%d calib=%d",
        tag,
        len(steered_rows),
        len(null_rows),
        len(calib_out),
    )
    return 0


# ── phase: stats ──────────────────────────────────────────────────────


def lattice_label(
    diff_lo: float,
    diff_hi: float,
    holm_sig: bool,
    d3_lo: float | None,
    d3_hi: float | None,
) -> str:
    """FOUR-BRANCH verdict lattice (plan §3; disjoint + exhaustive).

    - no-snowball        <=> paired-diff CI wholly <= 0
    - snowball-sufficient <=> separates AND D3 CI wholly >= 0
    - snowball-partial    <=> separates AND D3 CI wholly < 0
    - indeterminate       <=> otherwise (CI straddle, conjunct disagreement,
                              or a missing D3 CI)
    where "separates" = (diff CI strictly positive) AND (Holm-corrected exact
    Wilcoxon significant), as a CONJUNCTION; disagreement between the two
    conjuncts is Indeterminate, never collapsed.
    """
    if diff_hi <= 0:
        return "no-snowball"
    ci_pos = diff_lo > 0
    if ci_pos != holm_sig:
        return "indeterminate"
    if ci_pos and holm_sig:
        if d3_lo is None or d3_hi is None:
            return "indeterminate"
        if d3_lo >= 0:
            return "snowball-sufficient"
        if d3_hi < 0:
            return "snowball-partial"
    return "indeterminate"


def instance_label(scheme: str, label: str) -> str:
    """Scheme subordination (plan §3): scheme (b) = B's-own-opening donors is
    DESCRIPTIVE — its labels are prefixed 'natural-opening' and never carry
    the patch-mediation headline. Scheme (a) = med is confirmatory."""
    return label if scheme == "med" else f"natural-opening-{label}"


def _survivors(anchors_sep: dict[str, float | None], pair_ids: list[str]) -> list[str]:
    return [
        pid
        for pid in pair_ids
        if anchors_sep.get(pid) is not None and abs(anchors_sep[pid]) >= SEPARATION_BAR
    ]


HOLM_FIXED_M = 12  # plan §6: the registered family is the 12 arms per (model x pair-set)


def holm_fixed_m(pvals: dict[str, float], m: int = HOLM_FIXED_M) -> dict[str, float]:
    """Holm step-down with the family size FIXED at ``m`` (plan §6): the
    registered (model x pair-set) family is the 12 arms, so an arm that
    produced no p-value (dropped/untestable) still counts toward the
    correction — ``m = len(realized)`` was anti-conservative whenever arms
    dropped out (r1 Minor)."""
    assert len(pvals) <= m, (sorted(pvals), m)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(xs, ys).statistic)


def phase_stats(args: argparse.Namespace) -> int:
    # getattr-defaulted for hand-built Namespaces (see phase_f_tables).
    if getattr(args, "cell_set", "main") == "q35lang":
        return _stats_q35lang(args)
    tag = args.model_tag
    out_dir = args.out_dir or (FMETRICS_DIR / tag)
    steered = list(A62._iter_jsonl(out_dir / "f_cells.jsonl"))
    nulls = list(A62._iter_jsonl(out_dir / "null_cells.jsonl"))
    calib = list(A62._iter_jsonl(out_dir / "calib_cells.jsonl"))
    f_st = {(r["pair_id"], r["arm_slug"]): r for r in steered}
    f_nu = {(r["pair_id"], r["arm_slug"]): r for r in nulls}
    act_st = {(r["pair_id"], r["arm_slug"]): r.get("f_act") for r in steered}
    act_nu = {(r["pair_id"], r["arm_slug"]): r.get("f_act") for r in nulls}
    sep_by_pair = {r["pair_id"]: r.get("separation") for r in [*steered, *nulls]}

    # ce denominators: SAME-WAVE calibration (primary, q25) / fresh ce (q35);
    # banked values as the comparison read.
    ce_samewave: dict[str, float] = {}
    if tag == "q35":
        for r in A62._iter_jsonl(out_dir / "ce_cells.jsonl"):
            if r["variant"] == "steered" and r["f_beh"] is not None:
                ce_samewave[r["pair_id"]] = r["f_beh"]
    else:
        for r in calib:
            if r["arm"] == "steered" and r["f_beh"] is not None:
                ce_samewave[r["pair_id"]] = r["f_beh"]
    ce_banked: dict[str, float] = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(A2162_F_CELLS)
        if r["slot"] == "ce"
        and r["cell"] in C.S1_CELLS
        and r["arm"] == "steered"
        and r["f_beh"] is not None
    }
    for r in A62._iter_jsonl(INPUTS_DIR / "s2_ce_control_perpair.jsonl"):
        if r["arm"] == "steered" and r["f_beh"] is not None:
            ce_banked[r["pair_id"]] = r["f_beh"]

    # F_act ce denominators (secondary mirror — r1 blocker f-act-downstream):
    # q35 = fresh same-model ce_cells; q25 = the BANKED #2162 ce f_act (S1
    # only; the fu1-derived S2 ce table carries no f_act).
    ce_act: dict[str, float] = {}
    if tag == "q35":
        ce_act_source = "fresh-q35-ce (same model/wave)"
        for r in A62._iter_jsonl(out_dir / "ce_cells.jsonl"):
            if r["variant"] == "steered" and r.get("f_act") is not None:
                ce_act[r["pair_id"]] = r["f_act"]
    else:
        ce_act_source = "banked-q25-ce (#2162 f_cells; no banked S2 ce f_act)"
        for r in A62._iter_jsonl(A2162_F_CELLS):
            if (
                r["slot"] == "ce"
                and r["cell"] in C.S1_CELLS
                and r["arm"] == "steered"
                and r.get("f_act") is not None
            ):
                ce_act[r["pair_id"]] = r["f_act"]

    s1_pairs, s2_pairs = J33.build_pair_universe()
    cells_of = {p.pair_id: p.cell for p in s1_pairs}
    sets = {"s1": [p.pair_id for p in s1_pairs], "s2": [p.pair_id for p in s2_pairs]}
    result: dict = {"model_tag": tag, "per_set": {}}
    if tag == "q35":
        result["banked_ce_note"] = (
            "recovery_banked denominators are q25-normalized banked ce values — a "
            "CROSS-MODEL comparison read, never the q35 headline (samewave is primary)"
        )
    for set_name, pair_ids in sets.items():
        survivors_all = _survivors(sep_by_pair, pair_ids)
        # Survival-floor gating (plan §6; r1 blocker survival-floor-inference):
        # S1 floors are PER CELL (parent #2162 grain, 12/36); the pooled S1
        # analysis uses ONLY pairs from floor-passing cells. S2 is one 15-pair
        # set with a set-level floor of 5. Below-floor => NO tests, a
        # registered untestable label — never a small-n p-value.
        if set_name == "s1":
            floor = S1_SURVIVAL_FLOOR
            by_cell_surv: dict[str, list[str]] = defaultdict(list)
            for pid in survivors_all:
                by_cell_surv[cells_of[pid]].append(pid)
            passing = {c: v for c, v in by_cell_surv.items() if len(v) >= floor}
            survivors = sorted(pid for v in passing.values() for pid in v)
            floor_meta = {
                "grain": "per-cell",
                "floor": floor,
                "per_cell_n_survivors": {c: len(v) for c, v in sorted(by_cell_surv.items())},
                "cells_passing": sorted(passing),
                "cells_below_floor": sorted(set(by_cell_surv) - set(passing)),
            }
            testable_set = bool(passing)
        else:
            floor = S2_SURVIVAL_FLOOR
            survivors = sorted(survivors_all)
            floor_meta = {"grain": "set", "floor": floor, "n_survivors": len(survivors)}
            testable_set = len(survivors) >= floor

        arms_out: dict[str, dict] = {}
        pvals: dict[str, float] = {}
        arms_act: dict[str, dict] = {}
        pvals_act: dict[str, float] = {}
        for slug in C.ARM_SLUGS:
            pids = [
                pid
                for pid in survivors
                if f_st.get((pid, slug), {}).get("f_beh") is not None
                and f_nu.get((pid, slug), {}).get("f_beh") is not None
            ]
            rec: dict = {
                "n_pairs": len(pids),
                "below_floor": (not testable_set) or len(pids) < floor,
            }
            if rec["below_floor"]:
                rec["label"] = "untestable-small-n"
            else:
                d = np.array([f_st[(p, slug)]["f_beh"] - f_nu[(p, slug)]["f_beh"] for p in pids])
                # Registered joint per-pair columns (one bootstrap resample of
                # the PAIR axis drives every read — denominator variation
                # propagates into D3 by construction, plan §3):
                #   0 diff  1 F_steered  2 F_ce_samewave  3 D3-contrib(samewave)
                #   4 F_ce_banked       5 D3-contrib(banked)
                cols = np.full((len(pids), 6), np.nan)
                for i, p in enumerate(pids):
                    fs = f_st[(p, slug)]["f_beh"]
                    cols[i, 0] = d[i]
                    cols[i, 1] = fs
                    if p in ce_samewave:
                        cols[i, 2] = ce_samewave[p]
                        cols[i, 3] = fs - D3_SHARE * ce_samewave[p]
                    if p in ce_banked:
                        cols[i, 4] = ce_banked[p]
                        cols[i, 5] = fs - D3_SHARE * ce_banked[p]
                draws = bootstrap_family_means_batched(cols, BOOT_B, BOOT_SEED)
                rec["diff_mean"] = float(np.mean(d))
                rec["diff_ci"] = _ci(draws[:, 0])
                rec["f_steered_mean"] = float(np.nanmean(cols[:, 1]))
                rec["p_wilcoxon"] = A62._wilcoxon_exact_p(d)
                pvals[slug] = rec["p_wilcoxon"]
                for label, ce_col, d3_col in (("samewave", 2, 3), ("banked", 4, 5)):
                    ce_mean = float(np.nanmean(cols[:, ce_col]))
                    if math.isnan(ce_mean) or abs(ce_mean) < 1e-9:
                        continue
                    r_draws = draws[:, 1] / draws[:, ce_col]
                    rec[f"recovery_{label}"] = {
                        "ce_mean": ce_mean,
                        "ratio": float(np.nanmean(cols[:, 1])) / ce_mean,
                        "ratio_ci": _ci(r_draws),
                        "d3_mean": float(np.nanmean(cols[:, d3_col])),
                        "d3_ci": _ci(draws[:, d3_col]),
                    }
            arms_out[slug] = rec

            # F_act SECONDARY mirror (plan §6): same registered reads on the
            # activation DV — paired diff CI + exact Wilcoxon + its OWN fixed
            # m=12 Holm family + recovery + pair-level rank agreement.
            pids_a = [
                pid
                for pid in survivors
                if act_st.get((pid, slug)) is not None and act_nu.get((pid, slug)) is not None
            ]
            rec_a: dict = {
                "n_pairs": len(pids_a),
                "below_floor": (not testable_set) or len(pids_a) < floor,
            }
            if rec_a["below_floor"]:
                rec_a["label"] = "untestable-small-n"
            else:
                d_a = np.array([act_st[(p, slug)] - act_nu[(p, slug)] for p in pids_a])
                cols_a = np.full((len(pids_a), 4), np.nan)
                for i, p in enumerate(pids_a):
                    fa = act_st[(p, slug)]
                    cols_a[i, 0] = d_a[i]
                    cols_a[i, 1] = fa
                    if p in ce_act:
                        cols_a[i, 2] = ce_act[p]
                        cols_a[i, 3] = fa - D3_SHARE * ce_act[p]
                draws_a = bootstrap_family_means_batched(cols_a, BOOT_B, BOOT_SEED)
                rec_a["diff_mean"] = float(np.mean(d_a))
                rec_a["diff_ci"] = _ci(draws_a[:, 0])
                rec_a["f_act_steered_mean"] = float(np.nanmean(cols_a[:, 1]))
                rec_a["p_wilcoxon"] = A62._wilcoxon_exact_p(d_a)
                pvals_act[slug] = rec_a["p_wilcoxon"]
                ce_mean_a = float(np.nanmean(cols_a[:, 2]))
                if not (math.isnan(ce_mean_a) or abs(ce_mean_a) < 1e-9):
                    rec_a["recovery"] = {
                        "ce_mean": ce_mean_a,
                        "ce_source": ce_act_source,
                        "ratio": float(np.nanmean(cols_a[:, 1])) / ce_mean_a,
                        "ratio_ci": _ci(draws_a[:, 1] / draws_a[:, 2]),
                        "d3_mean": float(np.nanmean(cols_a[:, 3])),
                        "d3_ci": _ci(draws_a[:, 3]),
                    }
                common = [
                    p
                    for p in pids_a
                    if f_st.get((p, slug), {}).get("f_beh") is not None
                    and f_nu.get((p, slug), {}).get("f_beh") is not None
                ]
                if len(common) >= 3:
                    db = [f_st[(p, slug)]["f_beh"] - f_nu[(p, slug)]["f_beh"] for p in common]
                    da = [act_st[(p, slug)] - act_nu[(p, slug)] for p in common]
                    rec_a["spearman_vs_f_beh"] = {"rho": _spearman(db, da), "n_pairs": len(common)}
            arms_act[slug] = rec_a

        for family_pvals, family_arms in ((pvals, arms_out), (pvals_act, arms_act)):
            holmed = holm_fixed_m(family_pvals) if family_pvals else {}
            for slug, rec in family_arms.items():
                if slug in holmed:
                    rec["p_holm"] = holmed[slug]
                    rec["holm_significant"] = holmed[slug] < HOLM_ALPHA
                    if "diff_ci" in rec:
                        lo, hi = rec["diff_ci"]
                        rec["separates"] = (lo > 0) and rec["holm_significant"]

        def _prefill3_verdicts(family_arms: dict[str, dict], samewave_key: str) -> dict:
            verdicts: dict = {}
            for scheme in C.ARM_SCHEMES:
                slug = f"prefill3_{scheme}"
                rec = family_arms.get(slug, {})
                if rec.get("label") == "untestable-small-n":
                    verdicts[scheme] = {
                        "label": instance_label(scheme, "untestable-small-n"),
                        "reason": "below survival floor — no tests run",
                    }
                    continue
                if "diff_ci" not in rec:
                    verdicts[scheme] = {
                        "label": instance_label(scheme, "indeterminate"),
                        "reason": "no-data",
                    }
                    continue
                d3 = rec.get(samewave_key, {})
                label = lattice_label(
                    rec["diff_ci"][0],
                    rec["diff_ci"][1],
                    bool(rec.get("holm_significant")),
                    d3.get("d3_ci", (None, None))[0],
                    d3.get("d3_ci", (None, None))[1],
                )
                verdict = {
                    "label": instance_label(scheme, label),
                    "confirmatory": scheme == "med",
                    "below_floor": rec.get("below_floor"),
                }
                d3_banked = rec.get("recovery_banked")
                if d3_banked is not None:
                    verdict["label_banked_ce"] = instance_label(
                        scheme,
                        lattice_label(
                            rec["diff_ci"][0],
                            rec["diff_ci"][1],
                            bool(rec.get("holm_significant")),
                            d3_banked.get("d3_ci", (None, None))[0],
                            d3_banked.get("d3_ci", (None, None))[1],
                        ),
                    )
                verdicts[scheme] = verdict
            return verdicts

        both = [
            slug
            for slug in C.ARM_SLUGS
            if "diff_mean" in arms_out.get(slug, {}) and "diff_mean" in arms_act.get(slug, {})
        ]
        arm_spearman = {
            "rho": _spearman(
                [arms_out[s]["diff_mean"] for s in both],
                [arms_act[s]["diff_mean"] for s in both],
            )
            if len(both) >= 3
            else None,
            "n_arms": len(both),
        }

        result["per_set"][set_name] = {
            "n_survivors_anchor": len(survivors_all),
            "n_survivors_tested": len(survivors),
            "floor": floor_meta,
            "untestable": not testable_set,
            "arms": arms_out,
            "prefill3_verdicts": _prefill3_verdicts(arms_out, "recovery_samewave"),
            "holm_family_m": HOLM_FIXED_M,
            "f_act": {
                "role": "secondary companion DV (plan §6) — never the headline",
                "arms": arms_act,
                "prefill3_verdicts": _prefill3_verdicts(arms_act, "recovery"),
                "arm_level_spearman_vs_f_beh": arm_spearman,
                "holm_family_m": HOLM_FIXED_M,
            },
        }
    A62._write_json_atomic(out_dir / "stats.json", result)
    logger.info("[stats] %s: wrote %s", tag, out_dir / "stats.json")
    return 0


# ── q35_language_snowball registered machinery (plan §3 / §4.3 item 4) ─


def bootstrap_carrier_means_batched(
    values: np.ndarray, carrier_ids: list[str], n_boot: int, seed: int, *, block: int = 2000
) -> np.ndarray:
    """CARRIER-level cluster-bootstrap means (plan §3 companion): draw
    carriers with replacement; a drawn carrier carries ALL its rows (cells,
    directed pairs — and their anchors, which normalize F within the same
    carrier's contexts). Same batched index-GEMM shape as
    ``bootstrap_family_means_batched`` (NO per-draw loop): per-carrier column
    nan-sums/counts are pre-aggregated, so each block is one GEMM."""
    values = np.asarray(values, dtype=np.float64)
    n, f = values.shape
    assert len(carrier_ids) == n, (len(carrier_ids), n)
    uniq = sorted(set(carrier_ids))
    gidx = np.array([uniq.index(c) for c in carrier_ids])
    mask = ~np.isnan(values)
    v0 = np.where(mask, values, 0.0)
    sums = np.zeros((len(uniq), f), dtype=np.float64)
    cnts = np.zeros((len(uniq), f), dtype=np.float64)
    np.add.at(sums, gidx, v0)
    np.add.at(cnts, gidx, mask.astype(np.float64))
    g = len(uniq)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, f), dtype=np.float64)
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        idx = rng.integers(0, g, size=(b, g))
        counts = np.zeros((b, g), dtype=np.float64)
        np.add.at(counts, (np.arange(b)[:, None], idx), 1.0)
        num = counts @ sums
        den = counts @ cnts
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + b] = np.where(den > 0, num / den, np.nan)
    return out


def ratio_lattice_label(lo: float | None, hi: float | None) -> str:
    """Registered secondary ratio lattice (plan §3; consumes the R3_net CI):
    residual-established <=> CI wholly < 1; overshoot <=> CI wholly > 1;
    pure-snowball-compatible <=> otherwise (an included-1 outcome is
    COMPATIBILITY, never a tested equivalence)."""
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return "indeterminate"
    if hi < 1.0:
        return "residual-established"
    if lo > 1.0:
        return "overshoot"
    return "pure-snowball-compatible"


# JOINT per-pair column layout (plan §3: ONE resample of the pair axis drives
# every read — a drawn pair carries its arm, null, and ce rows together):
#   0 diff = F_st − F_nu     1 F_st            2 F_nu
#   3 ce_st (samewave)       4 ce_nu           5 dce = ce_st − ce_nu
#   6 D3_net(samewave)       7 D3_raw(samewave, v5 form)
#   8 bce_st (banked #2329)  9 bce_nu         10 dce_banked
#  11 D3_net(banked)        12 D3_raw(banked)
_Q35LANG_N_COLS = 13


def _stats_q35lang(args: argparse.Namespace) -> int:
    """Registered q35lang reads (plan §3/§6): net paired-diff separation,
    D3_net / R3_net under JOINT pair-clustered + carrier-cluster bootstraps,
    continuation-only companions, control-health + both-cells-floor
    preconditions, Wilcoxon+Holm (fixed m=12), F_act secondary mirror."""
    tag = args.model_tag
    out_dir = args.out_dir or _fmetrics_dir(tag, "q35lang")
    steered = list(A62._iter_jsonl(out_dir / "f_cells.jsonl"))
    nulls = list(A62._iter_jsonl(out_dir / "null_cells.jsonl"))
    ce_rows = list(A62._iter_jsonl(out_dir / "ce_cells.jsonl"))
    s1_pairs, s2_pairs = J33.build_pair_universe("q35lang")
    assert not s2_pairs
    carrier_of = {p.pair_id: p.carrier for p in s1_pairs}
    cells_of = {p.pair_id: p.cell for p in s1_pairs}
    pair_ids = [p.pair_id for p in s1_pairs]
    f_st = {(r["pair_id"], r["arm_slug"]): r for r in steered}
    f_nu = {(r["pair_id"], r["arm_slug"]): r for r in nulls}
    sep_by_pair = {r["pair_id"]: r.get("separation") for r in [*steered, *nulls]}

    ce_st: dict[str, float] = {}
    ce_nu: dict[str, float] = {}
    ce_act_st: dict[str, float] = {}
    for r in ce_rows:
        if r["f_beh"] is not None:
            (ce_st if r["variant"] == "steered" else ce_nu)[r["pair_id"]] = r["f_beh"]
        if r["variant"] == "steered" and r.get("f_act") is not None:
            ce_act_st[r["pair_id"]] = r["f_act"]
    # Banked #2329 companion rows (vendored, ce slot ONLY — plan §3
    # row-coverage; the pe rows are never consumed).
    bce_st = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(I2329_F_CELLS)
        if r["slot"] == "ce" and r["f_beh"] is not None
    }
    bce_nu = {
        r["pair_id"]: r["f_beh"]
        for r in A62._iter_jsonl(I2329_NULL_CELLS)
        if r["slot"] == "ce" and r["f_beh"] is not None
    }
    assert bce_st and bce_nu, "vendored #2329 ce companion rows filtered to empty"

    # ── preconditions (plan §3) ───────────────────────────────────────
    survivors_all = _survivors(sep_by_pair, pair_ids)
    by_cell_surv: dict[str, list[str]] = defaultdict(list)
    for pid in survivors_all:
        by_cell_surv[cells_of[pid]].append(pid)
    lang_cells = list(C.CELL_SETS["q35lang"].s1_cells)
    passing = {c for c in lang_cells if len(by_cell_surv.get(c, [])) >= S1_SURVIVAL_FLOOR}
    both_pass = passing == set(lang_cells)
    survivors = sorted(survivors_all) if both_pass else []
    floor_meta = {
        "grain": "per-cell",
        "floor": S1_SURVIVAL_FLOOR,
        "per_cell_n_survivors": {c: len(by_cell_surv.get(c, [])) for c in lang_cells},
        "cells_passing": sorted(passing),
        "cells_below_floor": sorted(set(lang_cells) - passing),
        "both_cells_pass": both_pass,
    }

    def _control_health(st_map: dict[str, float], nu_map: dict[str, float]) -> dict:
        """The §3 control-health read: the separation conjunction applied to
        ce steered−null (Holm at the registered family size m=12 — the
        precondition faces the same correction magnitude as the arms) plus
        the net-denominator CI-excludes-zero check."""
        pids = [p for p in survivors_all if p in st_map and p in nu_map]
        if len(pids) < 2:
            return {"n_pairs": len(pids), "passed": False, "reason": "insufficient ce pairs"}
        cols = np.array([[st_map[p] - nu_map[p], st_map[p], nu_map[p]] for p in pids])
        draws_p = bootstrap_family_means_batched(cols, BOOT_B, BOOT_SEED)
        draws_c = bootstrap_carrier_means_batched(
            cols, [carrier_of[p] for p in pids], BOOT_B, BOOT_SEED
        )
        ci = _ci(draws_p[:, 0])
        p_w = A62._wilcoxon_exact_p(cols[:, 0])
        p_holm = holm_fixed_m({"ce": p_w})["ce"]
        separates = ci[0] > 0 and p_holm < HOLM_ALPHA
        excludes = ci[0] > 0 or ci[1] < 0
        return {
            "n_pairs": len(pids),
            "dce_mean": float(np.mean(cols[:, 0])),
            "dce_ci": ci,
            "dce_ci_carrier": _ci(draws_c[:, 0]),
            "ce_steered_mean": float(np.mean(cols[:, 1])),
            "ce_null_mean": float(np.mean(cols[:, 2])),
            "p_wilcoxon": p_w,
            "p_holm_m12": p_holm,
            "separates": separates,
            "dce_ci_excludes_zero": excludes,
            "passed": bool(separates and excludes),
        }

    control_sw = _control_health(ce_st, ce_nu)
    control_bk = _control_health(bce_st, bce_nu)
    if not both_pass:
        precond_label = "untestable-causal"
    elif not control_sw["passed"]:
        precond_label = "control-unresolved"
    else:
        precond_label = None

    # ── per-arm registered reads ──────────────────────────────────────
    def _net_read(pids: list[str], slug: str, value_key: str) -> dict:
        n = len(pids)
        cols = np.full((n, _Q35LANG_N_COLS), np.nan)
        for i, p in enumerate(pids):
            fs = f_st[(p, slug)][value_key]
            fn = f_nu[(p, slug)][value_key]
            cols[i, 0] = fs - fn
            cols[i, 1] = fs
            cols[i, 2] = fn
            if p in ce_st and p in ce_nu:
                dce = ce_st[p] - ce_nu[p]
                cols[i, 3] = ce_st[p]
                cols[i, 4] = ce_nu[p]
                cols[i, 5] = dce
                cols[i, 6] = (fs - fn) - D3_NET_SHARE * dce
            if p in ce_st:
                cols[i, 7] = fs - D3_NET_SHARE * ce_st[p]
            if p in bce_st and p in bce_nu:
                dceb = bce_st[p] - bce_nu[p]
                cols[i, 8] = bce_st[p]
                cols[i, 9] = bce_nu[p]
                cols[i, 10] = dceb
                cols[i, 11] = (fs - fn) - D3_NET_SHARE * dceb
            if p in bce_st:
                cols[i, 12] = fs - D3_NET_SHARE * bce_st[p]
        carr = [carrier_of[p] for p in pids]
        draws_p = bootstrap_family_means_batched(cols, BOOT_B, BOOT_SEED)
        draws_c = bootstrap_carrier_means_batched(cols, carr, BOOT_B, BOOT_SEED)
        rec: dict = {
            "diff_mean": float(np.mean(cols[:, 0])),
            "diff_ci": _ci(draws_p[:, 0]),
            "diff_ci_carrier": _ci(draws_c[:, 0]),
            "f_steered_mean": float(np.nanmean(cols[:, 1])),
            "f_null_mean": float(np.nanmean(cols[:, 2])),
            "p_wilcoxon": A62._wilcoxon_exact_p(cols[:, 0]),
            "n_carriers": len(set(carr)),
        }
        for label, ce_col, dce_col, d3n_col, d3r_col in (
            ("samewave", 3, 5, 6, 7),
            ("banked", 8, 10, 11, 12),
        ):
            ce_mean = float(np.nanmean(cols[:, ce_col]))
            # Raw (v5-form, steered-only) DESCRIPTIVE companion — same
            # emit-condition as the parent stats (figures read ["ratio"]).
            if not (math.isnan(ce_mean) or abs(ce_mean) < 1e-9):
                rec[f"recovery_{label}"] = {
                    "ce_mean": ce_mean,
                    "ratio": rec["f_steered_mean"] / ce_mean,
                    "ratio_ci": _ci(draws_p[:, 1] / draws_p[:, ce_col]),
                    "d3_mean": float(np.nanmean(cols[:, d3r_col])),
                    "d3_ci": _ci(draws_p[:, d3r_col]),
                }
            dce_mean = float(np.nanmean(cols[:, dce_col]))
            if math.isnan(dce_mean):
                continue
            net: dict = {
                "dce_mean": dce_mean,
                "dce_ci": _ci(draws_p[:, dce_col]),
                "d3_net_mean": float(np.nanmean(cols[:, d3n_col])),
                "d3_net_ci": _ci(draws_p[:, d3n_col]),
                "d3_net_ci_carrier": _ci(draws_c[:, d3n_col]),
            }
            if abs(dce_mean) > 1e-9:
                net["ratio_net"] = rec["diff_mean"] / dce_mean
                net["ratio_net_ci"] = _ci(draws_p[:, 0] / draws_p[:, dce_col])
                net["ratio_net_ci_carrier"] = _ci(draws_c[:, 0] / draws_c[:, dce_col])
            rec[f"recovery_net_{label}"] = net
        return rec

    act_st = {(r["pair_id"], r["arm_slug"]): r.get("f_act") for r in steered}
    act_nu = {(r["pair_id"], r["arm_slug"]): r.get("f_act") for r in nulls}
    arms_out: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    cont_out: dict[str, dict] = {}
    pvals_cont: dict[str, float] = {}
    arms_act: dict[str, dict] = {}
    pvals_act: dict[str, float] = {}
    for slug in C.ARM_SLUGS:
        pids = [
            pid
            for pid in survivors
            if f_st.get((pid, slug), {}).get("f_beh") is not None
            and f_nu.get((pid, slug), {}).get("f_beh") is not None
        ]
        rec: dict = {
            "n_pairs": len(pids),
            "below_floor": (not both_pass) or len(pids) < S1_SURVIVAL_FLOOR,
        }
        if rec["below_floor"]:
            rec["label"] = "untestable-causal" if not both_pass else "untestable-small-n"
        else:
            rec.update(_net_read(pids, slug, "f_beh"))
            pvals[slug] = rec["p_wilcoxon"]
        arms_out[slug] = rec

        # Continuation-only companion (REQUIRED for every prefill arm, §3
        # scoring-convention precedence; the tag-"n" judge wave covers
        # steered AND null rows, so both legs exist).
        if C.parse_arm(slug)[0] == "prefill":
            pids_c = [
                pid
                for pid in survivors
                if f_st.get((pid, slug), {}).get("f_beh_continuation") is not None
                and f_nu.get((pid, slug), {}).get("f_beh_continuation") is not None
            ]
            rec_c: dict = {
                "n_pairs": len(pids_c),
                "below_floor": (not both_pass) or len(pids_c) < S1_SURVIVAL_FLOOR,
            }
            if rec_c["below_floor"]:
                rec_c["label"] = "untestable-causal" if not both_pass else "untestable-small-n"
            else:
                rec_c.update(_net_read(pids_c, slug, "f_beh_continuation"))
                pvals_cont[slug] = rec_c["p_wilcoxon"]
                if "f_steered_mean" in rec:
                    div = rec["f_steered_mean"] - rec_c["f_steered_mean"]
                    rec_c["whole_minus_cont_steered"] = div
                    rec_c["divergence_gt_bar"] = bool(abs(div) > 0.05)
            cont_out[slug] = rec_c

        # F_act SECONDARY mirror (raw samewave recovery + rank agreement —
        # plan §6: stays secondary, never a lattice input).
        pids_a = [
            pid
            for pid in survivors
            if act_st.get((pid, slug)) is not None and act_nu.get((pid, slug)) is not None
        ]
        rec_a: dict = {
            "n_pairs": len(pids_a),
            "below_floor": (not both_pass) or len(pids_a) < S1_SURVIVAL_FLOOR,
        }
        if rec_a["below_floor"]:
            rec_a["label"] = "untestable-causal" if not both_pass else "untestable-small-n"
        else:
            d_a = np.array([act_st[(p, slug)] - act_nu[(p, slug)] for p in pids_a])
            cols_a = np.full((len(pids_a), 4), np.nan)
            for i, p in enumerate(pids_a):
                fa = act_st[(p, slug)]
                cols_a[i, 0] = d_a[i]
                cols_a[i, 1] = fa
                if p in ce_act_st:
                    cols_a[i, 2] = ce_act_st[p]
                    cols_a[i, 3] = fa - D3_NET_SHARE * ce_act_st[p]
            draws_a = bootstrap_family_means_batched(cols_a, BOOT_B, BOOT_SEED)
            draws_ac = bootstrap_carrier_means_batched(
                cols_a, [carrier_of[p] for p in pids_a], BOOT_B, BOOT_SEED
            )
            rec_a["diff_mean"] = float(np.mean(d_a))
            rec_a["diff_ci"] = _ci(draws_a[:, 0])
            rec_a["diff_ci_carrier"] = _ci(draws_ac[:, 0])
            rec_a["f_act_steered_mean"] = float(np.nanmean(cols_a[:, 1]))
            rec_a["p_wilcoxon"] = A62._wilcoxon_exact_p(d_a)
            pvals_act[slug] = rec_a["p_wilcoxon"]
            ce_mean_a = float(np.nanmean(cols_a[:, 2]))
            if not (math.isnan(ce_mean_a) or abs(ce_mean_a) < 1e-9):
                rec_a["recovery"] = {
                    "ce_mean": ce_mean_a,
                    "ce_source": "fresh-q35lang-ce (same model/wave)",
                    "ratio": float(np.nanmean(cols_a[:, 1])) / ce_mean_a,
                    "ratio_ci": _ci(draws_a[:, 1] / draws_a[:, 2]),
                    "d3_mean": float(np.nanmean(cols_a[:, 3])),
                    "d3_ci": _ci(draws_a[:, 3]),
                }
            common = [
                p
                for p in pids_a
                if f_st.get((p, slug), {}).get("f_beh") is not None
                and f_nu.get((p, slug), {}).get("f_beh") is not None
            ]
            if len(common) >= 3:
                db = [f_st[(p, slug)]["f_beh"] - f_nu[(p, slug)]["f_beh"] for p in common]
                da = [act_st[(p, slug)] - act_nu[(p, slug)] for p in common]
                rec_a["spearman_vs_f_beh"] = {"rho": _spearman(db, da), "n_pairs": len(common)}
        arms_act[slug] = rec_a

    for family_pvals, family_arms in (
        (pvals, arms_out),
        (pvals_cont, cont_out),
        (pvals_act, arms_act),
    ):
        holmed = holm_fixed_m(family_pvals) if family_pvals else {}
        for slug, rec in family_arms.items():
            if slug in holmed:
                rec["p_holm"] = holmed[slug]
                rec["holm_significant"] = holmed[slug] < HOLM_ALPHA
                if "diff_ci" in rec:
                    rec["separates"] = (rec["diff_ci"][0] > 0) and rec["holm_significant"]
                    rec["separates_carrier"] = (rec["diff_ci_carrier"][0] > 0) and rec[
                        "holm_significant"
                    ]
                    rec["cluster_sensitive_separation"] = (
                        rec["separates"] != rec["separates_carrier"]
                    )

    # ── prefill-3 verdict bundles (plan §3: both lattices, net quantities,
    #    carrier companion, continuation precedence, banked companion) ──
    def _lattice_reads(rec: dict, net_key: str, gate_ok: bool) -> dict:
        if not gate_ok:
            return {
                "primary_net": "control-unresolved",
                "ratio_net": "control-unresolved",
                "cluster_sensitive_primary": False,
                "cluster_sensitive_ratio": False,
                "primary_final": "control-unresolved",
                "ratio_final": "control-unresolved",
            }
        if rec.get("label") or "diff_ci" not in rec:
            lab = rec.get("label", "indeterminate")
            return {
                "primary_net": lab,
                "ratio_net": lab,
                "cluster_sensitive_primary": False,
                "cluster_sensitive_ratio": False,
                "primary_final": lab,
                "ratio_final": lab,
            }
        net = rec.get(net_key, {})
        d3_ci = net.get("d3_net_ci", (None, None))
        d3_ci_c = net.get("d3_net_ci_carrier", (None, None))
        holm_sig = bool(rec.get("holm_significant"))
        primary = lattice_label(rec["diff_ci"][0], rec["diff_ci"][1], holm_sig, *d3_ci)
        primary_c = lattice_label(
            rec["diff_ci_carrier"][0], rec["diff_ci_carrier"][1], holm_sig, *d3_ci_c
        )
        ratio = ratio_lattice_label(*net.get("ratio_net_ci", (None, None)))
        ratio_c = ratio_lattice_label(*net.get("ratio_net_ci_carrier", (None, None)))
        return {
            "primary_net": primary,
            "primary_net_carrier": primary_c,
            "cluster_sensitive_primary": primary != primary_c,
            "primary_final": primary if primary == primary_c else "indeterminate",
            "ratio_net": ratio,
            "ratio_net_carrier": ratio_c,
            "cluster_sensitive_ratio": ratio != ratio_c,
            "ratio_final": ratio if ratio == ratio_c else "indeterminate",
        }

    verdicts: dict[str, dict] = {}
    for scheme in C.ARM_SCHEMES:
        slug = f"prefill3_{scheme}"
        rec = arms_out.get(slug, {})
        rec_c = cont_out.get(slug, {})
        if precond_label is not None:
            bundle: dict = {
                "label": instance_label(scheme, precond_label),
                "reason": (
                    "both-cells 12/36 floor failed — pooled lattice not emitted"
                    if precond_label == "untestable-causal"
                    else "same-wave ce control failed its own separation conjunction or "
                    "the net denominator CI includes zero"
                ),
                "confirmatory": scheme == "med",
            }
        else:
            reads = _lattice_reads(rec, "recovery_net_samewave", True)
            reads_c = _lattice_reads(rec_c, "recovery_net_samewave", True)
            governs = bool(rec_c.get("divergence_gt_bar"))
            bundle = {
                "confirmatory": scheme == "med",
                **{
                    k: instance_label(scheme, v) if isinstance(v, str) else v
                    for k, v in reads.items()
                },
                "continuation": {
                    k: instance_label(scheme, v) if isinstance(v, str) else v
                    for k, v in reads_c.items()
                },
                "continuation_governs_mechanism": governs,
                "mechanism_final": instance_label(
                    scheme,
                    (
                        reads_c.get("primary_final", "indeterminate")
                        if governs
                        else reads.get("primary_final", "indeterminate")
                    ),
                ),
                "banked_companion": {
                    k: instance_label(scheme, v) if isinstance(v, str) else v
                    for k, v in _lattice_reads(
                        rec, "recovery_net_banked", bool(control_bk.get("dce_ci_excludes_zero"))
                    ).items()
                },
            }
        verdicts[scheme] = bundle

    # Per-cell descriptive reads (raw p, no family — plan §3 auxiliary; the
    # surviving-cell fallback under `untestable-causal`).
    per_cell: dict[str, dict] = {}
    for cell in lang_cells:
        cell_pids = by_cell_surv.get(cell, [])
        cell_arms: dict[str, dict] = {}
        for slug in C.ARM_SLUGS:
            ds = [
                f_st[(p, slug)]["f_beh"] - f_nu[(p, slug)]["f_beh"]
                for p in cell_pids
                if f_st.get((p, slug), {}).get("f_beh") is not None
                and f_nu.get((p, slug), {}).get("f_beh") is not None
            ]
            if ds:
                cell_arms[slug] = {
                    "n_pairs": len(ds),
                    "diff_mean": float(np.mean(ds)),
                    "p_wilcoxon_raw": A62._wilcoxon_exact_p(np.array(ds)),
                }
        per_cell[cell] = {"n_survivors": len(cell_pids), "arms": cell_arms}

    result = {
        "model_tag": tag,
        "cell_set": "q35lang",
        "boot": {
            "B": BOOT_B,
            "seed": BOOT_SEED,
            "resampling": "JOINT pair-clustered (a drawn pair carries arm+null+ce rows)",
            "carrier_companion": "carrier-cluster bootstrap on every registered read",
        },
        "preconditions": {
            "floor": floor_meta,
            "precondition_label": precond_label,
            "control_health_samewave": control_sw,
            "control_health_banked_companion": control_bk,
        },
        "banked_ce_note": (
            "banked reads join the vendored #2329 language-cell ce rows (slot=='ce' ONLY) — "
            "a CROSS-WAVE comparison read; the same-wave ce control is primary (plan §3)"
        ),
        "per_set": {
            "s1": {
                "n_survivors_anchor": len(survivors_all),
                "n_survivors_tested": len(survivors),
                "floor": floor_meta,
                "untestable": not both_pass,
                "arms": arms_out,
                "continuation_arms": cont_out,
                "prefill3_verdicts": verdicts,
                "holm_family_m": HOLM_FIXED_M,
                "f_act": {
                    "role": "secondary companion DV (plan §6) — never the headline",
                    "arms": arms_act,
                    "holm_family_m": HOLM_FIXED_M,
                },
            }
        },
        "per_cell_descriptive": per_cell,
    }
    A62._write_json_atomic(out_dir / "stats.json", result)
    logger.info("[stats] q35lang: wrote %s", out_dir / "stats.json")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────

PHASES = {
    "s2-ce-derive": phase_s2_ce_derive,
    "f-tables": phase_f_tables,
    "stats": phase_stats,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2333 VM-side analysis.")
    # required unless --import-check (r1 Minor: standalone --import-check).
    ap.add_argument("--phase", choices=tuple(PHASES))
    ap.add_argument("--model-tag", choices=("q25", "q35"), default=None)
    ap.add_argument(
        "--cell-set",
        choices=tuple(C.CELL_SETS),
        default="main",
        help="pair universe (default 'main' = the parent run; 'q35lang' routes "
        "f-tables/stats to eval_results/issue_2333/q35_language_snowball/f_metrics)",
    )
    ap.add_argument("--stage-dir", type=Path, default=Path("data/issue_2333/fu1_stage"))
    ap.add_argument("--scores-dir", type=Path, default=None)
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--anchors-dir", type=Path, default=None)
    ap.add_argument("--calib-dir", type=Path, default=Path("data/issue_2333/judge_inputs/calib"))
    ap.add_argument("--va-dir", type=Path, default=None)
    ap.add_argument("--anchor-va-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def _import_check() -> int:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from huggingface_hub import hf_hub_download  # noqa: F401

    from explore_persona_space.experiments.issue2094 import fmetrics as FM
    from explore_persona_space.orchestrate import hub

    assert callable(hub.retry_transient) and callable(FM.f_act)
    import torch  # noqa: F401

    assert callable(bootstrap_family_means_batched)
    assert lattice_label(-0.1, -0.01, False, None, None) == "no-snowball"
    assert instance_label("bstart", "no-snowball") == "natural-opening-no-snowball"
    # q35lang registered machinery (plan §3): ratio lattice + carrier bootstrap.
    assert ratio_lattice_label(0.2, 0.8) == "residual-established"
    assert ratio_lattice_label(1.1, 1.5) == "overshoot"
    assert ratio_lattice_label(0.8, 1.2) == "pure-snowball-compatible"
    assert ratio_lattice_label(None, None) == "indeterminate"
    demo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]])
    draws = bootstrap_carrier_means_batched(demo, ["a", "a", "b"], 16, 0)
    assert draws.shape == (16, 2), draws.shape
    print("[import-check] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    assert args.phase, "--phase required (or --import-check)"
    if args.cell_set == "q35lang":
        assert args.phase in ("f-tables", "stats"), (
            "--cell-set q35lang applies to f-tables/stats only (s2-ce-derive is a "
            "main-path input builder)"
        )
        assert args.model_tag == "q35", "--cell-set q35lang requires --model-tag q35 (plan §11)"
    if args.phase in ("f-tables", "stats"):
        assert args.model_tag, f"--model-tag required for --phase {args.phase}"
        if args.phase == "f-tables":
            assert args.scores_dir is not None, "--scores-dir required for f-tables"
            assert args.rollouts_dir is not None, "--rollouts-dir required for f-tables"
            if args.model_tag == "q35":
                assert args.anchors_dir is not None, "--anchors-dir required for q35 f-tables"
    return PHASES[args.phase](args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
