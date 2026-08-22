#!/usr/bin/env python3
"""Issue #1901 follow-up ``generic-boundary-token-control`` — driver (plan v10 §4).

Adds the single generic-boundary-token control arm to the C1 scaling figure:
fit boundary-token state (``x_sep``, the residual at a sentence-final "."
anchor) → next-span-mean (``y``) linear ridge maps on WikiText-103-raw-v1 at
the parent chat ladder's exact training sizes / layers / eval conventions,
then add the curve as a 4th series to ``figures/paper/c1_scaling_train_pool``.

Phases (``--phase``):
  b0_pairs    (cpu-bigmem pod) stream the pinned WikiText train split, build
              anchor pairs via the #931 rig (unit-1 extended
              ``build_armc_pairs``), article-grain split (seed 42), screens
              (exact dedup + transposed 5-gram-Jaccard-0.8 ``NearDupeGate``),
              yield gates (§7.1), manifest + article token shards → HF
              ``issue1901_boundary_ctl/manifest/``.
  p1_capture  (GPU pod) stage the manifest, load Qwen2.5-7B-Instruct bf16,
              length-grouped batched capture of ``x_sep`` + ``y`` at layers
              {2,14,19,26} bf16 (~2,000-pair shards), shard-1 pilot timing
              gate (§7.2), upload-as-you-go every 10 shards → HF
              ``issue1901_boundary_ctl/capture/``.
  p2_fits     (same pod) per rung × draw val-λ primal streaming ridge
              (LAMBDAS_N50K ≤ 50k / LAMBDAS_N1M above) + identity+bias +
              constant train-mean, ``score_cell`` (n_boot=1000, seed 1901),
              200-draw shuffled-pair null (advisory), article-level cluster
              bootstrap → §3 ``gap_interval``; cuda-vs-cpu rung-50 parity
              gate (§7.3) at entry. Outputs the two eval JSONs.
  p3_publish  (same pod) git-push the eval JSONs (pre-teardown harvest),
              mirror to HF ``issue1901_boundary_ctl/eval/``, verify capture
              uploads, purge the local store.
  pod_all     p1 → p2 → p3 on the one GPU provision.
  fig         (VM) invoke the extended shared renderer
              (``issue1901_body_figures.fig_paper_c1_scaling`` — the
              renderer-side extension is a SEPARATE deliverable; until it
              lands this phase fail-louds at the extended kwargs) + the §7.4
              inherited-point tuple-multiset regression gate + the
              exploratory figure dump to ``figures/issue_1901/``.

Smoke (``--smoke``): b0 on 50 REAL streamed articles (per-filter reject
counters at the slice's full grain); p1 on a 6-layer tiny model
(``make_tiny_model``) persisting the strict subset {0,2,4,5} (the M4 layer
remap — same subset size / non-contiguity as production); p2 at n=50,
n_boot=50, K=20; fig into a scratch stem. Production-n-calibrated gates
(yield floor, pilot abort, eval-pool sizes) demote to informational log
lines under smoke (#1345); no assert is otherwise downgraded. Smoke skips HF
uploads and git pushes (enumerated blind spots — plan §4; the p1 shard-1
production pilot + first timed upload batch cover the real-model / upload
paths).

Checkpointing: b0 at stage grain (the single builder call is monolithic by
source design; restart cost ≤ its own ~0.7 h booking); p1 per shard with a
sidecar-validated resume scan; p2 per cell (unit JSONs keyed on generating
parameters — never recomputed-float-array bytes, #1336). One stdout progress
line per unit. Fail loud everywhere; NaN never coerced. WikiText is a benign
corpus (no content-hygiene digest restrictions apply).
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: shared-VM thread caps must bind BEFORE numpy/torch import.
load_dotenv()

import issue779_common as C79  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue931_build_pairs as BP  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_extract_store as ES  # noqa: E402
import issue1901_paper_densify as PD  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue779_ffc_n1m_generate_capture import NearDupeGate, _norm  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1901_boundary_token_control")

# ── constants (plan v10 §4/§7/§11) ──────────────────────────────────────────────

HF_DATA_REPO = common.HF_DATA_REPO
HF_PREFIX = "issue1901_boundary_ctl"
MANIFEST_PREFIX = f"{HF_PREFIX}/manifest"
CAPTURE_PREFIX = f"{HF_PREFIX}/capture"
EVAL_PREFIX = f"{HF_PREFIX}/eval"

SPLIT_SEED = 42
BATTERY_SEED = 1901
SMALL_RUNGS = (50, 100, 250, 500, 1000, 2500)
SMALL_DRAWS = (0, 1, 2)
# The banked chat-grid sizes (§2 — the parent ladder + densify + battery rungs).
BANKED_CHAT_GRID = (
    50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000, 25000,
    50000, 150000, 500000, 963444,
)  # fmt: skip
EVAL_MARGIN = 1400  # 1,000 test + 400 val
N_EVAL_TEST = 1000
N_EVAL_VAL = 400
EVAL_SPANS_PER_ARTICLE_CAP = 12
HALT_FLOOR = 51400  # §7.1: 50k rung + eval margin
PROD_PERSIST_LAYERS = (2, 14, 19, 26)
SMOKE_PERSIST_LAYERS = (0, 2, 4, 5)  # M4 remap on the 6-layer tiny model
HEADLINE_LAYER = 19
BOUNDARY_SERIES_LABEL = "generic boundary token ('.')"
POOLED_BRIDGE_MAX_RUNG = 20000
GROUP_DIVERSITY_N = 100_000
GROUP_DIVERSITY_MAX_PER_ARTICLE = 6

# Tokenizer probe (§11, run at plan time; re-asserted at b0 entry).
TOKEN_ID_PROBES = {".": [13], " .": [659], "?": [30], "!": [0]}

# Designed halt rcs (artifact-routed halts, never anonymous rc=1 — gotchas.md pilot-gate rule).
RC_YIELD_HALT = 6
RC_PILOT_HALT = 7

# Banked chat-side ridge R² at L19 per rung (§3 gap interval's chat arm), read
# from the committed artifacts at implementation time (2026-08-22). Rungs
# 50..25,000 come from scaling_ladder_L19.json cells (across-draw mean of
# ridge.test_r2; the banked ladder cells carry NO bootstrap CI — those rungs'
# chat interval is POINT-DEGENERATE, flagged per entry via chat_ci_source).
# 50k/150k/500k/963,444 carry the banked conversation-level bootstrap CIs.
# When the committed files are present the table is recomputed from them and
# asserted against these constants (the _banked_parity_target pattern).
CHAT_RIDGE_L19_R2 = {
    50: {"point": 0.3227494492686382, "lo": None, "hi": None},
    100: {"point": 0.36282659658466554, "lo": None, "hi": None},
    250: {"point": 0.49122154007399965, "lo": None, "hi": None},
    500: {"point": 0.5562363432824192, "lo": None, "hi": None},
    1000: {"point": 0.6033319349246442, "lo": None, "hi": None},
    2500: {"point": 0.6557336117296017, "lo": None, "hi": None},
    5000: {"point": 0.6813526733515054, "lo": None, "hi": None},
    10000: {"point": 0.7048843804909557, "lo": None, "hi": None},
    15000: {"point": 0.7165079370002496, "lo": None, "hi": None},
    20000: {"point": 0.7209572011247484, "lo": None, "hi": None},
    25000: {"point": 0.7250873308449972, "lo": None, "hi": None},
    50000: {"point": 0.7599992543132659, "lo": 0.7484796280587248, "hi": 0.7704399931179746},
    150000: {"point": 0.755515914218508, "lo": 0.7448128295933962, "hi": 0.765783531304794},
    500000: {"point": 0.7609049151916738, "lo": 0.7502385867503333, "hi": 0.7708112888840339},
    963444: {"point": 0.754170841830715, "lo": 0.7441307847286448, "hi": 0.7638445771747455},
}
_CHAT_SOURCE_FILES = {
    "ladder": PROJECT_ROOT / "eval_results/issue_1901/paper_densify/scaling_ladder_L19.json",
    "n50k": PROJECT_ROOT / "eval_results/issue_1901/paper_densify/layer_curve_n50k.json",
    "bign": PROJECT_ROOT / "eval_results/issue_1901/paper_densify/scaling_bigN_acc1_L19.json",
    "battery": PROJECT_ROOT / "eval_results/issue_1901/metric_battery/context_arm.json",
}

DEFAULT_OUT_ROOT = Path(os.environ.get("WORKLOAD_ROOT", "/workspace")) / "eps-issue-1901-btokctl"
DEFAULT_EVAL_OUT = PROJECT_ROOT / "eval_results" / "issue_1901" / "paper_densify"
SCALING_JSON = "boundary_token_scaling_L19.json"
SECONDARY_JSON = "boundary_token_secondary.json"


def _meta(phase: str, extra: dict | None = None) -> dict:
    """Reproducibility metadata (git_provenance + phase identity, code-style rule)."""
    out = as_metadata_dict(git_provenance(), phase=phase)
    out.update(
        {
            "script": "issue1901_boundary_token_control",
            "issue": 1901,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    if extra:
        out.update(extra)
    return out


def _lambdas_for(n_train: int) -> np.ndarray:
    """Parent densify phase split: LAMBDAS_N50K at n ≤ 50k, LAMBDAS_N1M above."""
    return N50.LAMBDAS_N50K if n_train <= 50_000 else N1M.LAMBDAS_N1M


def _lambda_grid_params(n_train: int) -> list:
    return ["logspace", -3, 7, 21] if n_train <= 50_000 else ["logspace", -3, 8, 23]


# ── b0_pairs ────────────────────────────────────────────────────────────────────


def _assert_token_probes(tokenizer) -> None:
    """Re-assert the plan-time tokenizer probe (§11) at b0 entry."""
    for text, want in TOKEN_ID_PROBES.items():
        got = tokenizer.encode(text, add_special_tokens=False)
        assert got == want, f"tokenizer probe drift: {text!r} -> {got} (expected {want})"


def _verify_wikitext_pin() -> None:
    """A2: the pinned WikiText revision must resolve on the Hub (fail loud)."""
    from huggingface_hub import HfApi

    info = hub.retry_transient(
        lambda: HfApi().dataset_info("Salesforce/wikitext", revision=common.WIKITEXT_REVISION),
        what="wikitext revision pin resolve",
    )
    logger.info("[b0] wikitext pin resolves: sha=%s", info.sha)


def _decode_span_texts(tokenizer, ids_by_art: dict, pairs: list) -> list[str]:
    """Decode each pair's t_span token ids to text (for the dedup/near-dupe screens)."""
    texts: list[str] = []
    chunk: list[list[int]] = []
    for p in pairs:
        lo, hi = p.t_spans[0]
        chunk.append(list(ids_by_art[p.group_id][lo:hi]))
        if len(chunk) >= 10_000:
            texts.extend(tokenizer.batch_decode(chunk))
            chunk = []
    if chunk:
        texts.extend(tokenizer.batch_decode(chunk))
    assert len(texts) == len(pairs), (len(texts), len(pairs))
    return texts


def _span_length_stats(lengths: list[int]) -> dict:
    a = np.asarray(lengths, dtype=np.int64)
    if a.size == 0:
        return {"n": 0}
    edges = [8, 16, 24, 32, 48, 64, 96, 128, 192, 257]
    hist, _ = np.histogram(a, bins=edges)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": int(a.min()),
        "max": int(a.max()),
        "hist_edges": edges,
        "hist_counts": [int(v) for v in hist],
    }


def _eval_margin(args) -> int:
    """Eval-pool target: 1,400 in production; a scaled-down smoke target (#1345 —
    the production margin would swallow EVERY smoke article into the eval split,
    leaving zero train candidates)."""
    return int(args.smoke_eval_spans) if args.smoke else EVAL_MARGIN


def _test_val_counts(n_sel: int, smoke: bool) -> tuple[int, int]:
    """Test/val sizes for one selected eval pool. Production: exactly 1,000/400
    (short pools take what exists). Smoke: proportional ~70/30 so BOTH splits
    are non-empty at tiny n (a 0-row val pool crashes fit_ridge)."""
    if not smoke:
        n_test = min(N_EVAL_TEST, n_sel)
        return n_test, min(N_EVAL_VAL, n_sel - n_test)
    n_test = min(n_sel, max(2, int(round(n_sel * 0.7))))
    return n_test, min(N_EVAL_VAL, n_sel - n_test)


def _b0_regime_key(args) -> dict:
    return {
        "split_seed": int(args.split_seed),
        "max_anchors": int(args.max_anchors),
        "article_cap_tokens": int(args.article_cap_tokens),
        "wikitext_revision": common.WIKITEXT_REVISION,
        "near_dupe": {"ngram": 5, "jaccard": 0.8},
        "eval_target": {
            "margin": _eval_margin(args),
            "test": N_EVAL_TEST,
            "val": N_EVAL_VAL,
            "per_article_cap": EVAL_SPANS_PER_ARTICLE_CAP,
        },
        "smoke": bool(args.smoke),
        "smoke_articles": int(args.smoke_articles) if args.smoke else None,
    }


def phase_b0_pairs(args) -> int:  # noqa: C901 -- linear build → split → screens → write ladder
    """Build pairs, split at article grain, screen, gate, write + upload the manifest."""
    C79.phase("b0_pairs")
    man_dir = args.out_root / "manifest"
    meta_path = man_dir / "meta.json"
    regime_key = _b0_regime_key(args)
    tokenizer = common.get_tokenizer()
    _assert_token_probes(tokenizer)

    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev.get("regime_key") == regime_key:
            logger.info("[b0] resume-skip: manifest already built at %s", man_dir)
            meta = prev
        else:
            raise RuntimeError(
                f"[b0] {meta_path} exists with a DIFFERENT regime_key "
                f"(have {prev.get('regime_key')}, want {regime_key}) — move the stale "
                f"manifest dir aside; never silently reuse cross-regime state (#1333)."
            )
    else:
        _verify_wikitext_pin()
        meta = _b0_build(args, tokenizer, man_dir, regime_key)
    if not meta["yield_gate"]["pass"] and not args.smoke:
        C79.write_json_atomic(args.out_root / "b0_yield_halt_report.json", meta["yield_gate"])
        logger.error("[b0] YIELD HALT (§7.1): %s", meta["yield_gate"])
        return RC_YIELD_HALT

    if args.smoke or args.skip_upload:
        logger.info("[b0] smoke/skip-upload: HF upload SKIPPED (enumerated smoke blind spot)")
        return 0
    files = sorted(p.name for p in man_dir.iterdir() if p.is_file())
    url = hub._upload(
        man_dir, HF_DATA_REPO, "dataset", path_in_repo=MANIFEST_PREFIX, raise_on_error=True
    )
    if not url:
        raise RuntimeError("[b0] manifest upload returned empty url — upload failed")
    from huggingface_hub import HfApi

    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        [f"{MANIFEST_PREFIX}/{name}" for name in files],
        path_in_repo=MANIFEST_PREFIX,
    )
    if missing:
        raise RuntimeError(f"[b0] manifest upload verify FAILED — missing on Hub: {missing[:5]}")
    logger.info("[b0] manifest uploaded + verified: %d files under %s", len(files), MANIFEST_PREFIX)
    return 0


def _b0_build(args, tokenizer, man_dir: Path, regime_key: dict) -> dict:  # noqa: C901
    """The b0 core: stream → pairs → article split → screens → manifest shards."""
    t0 = time.time()
    orig_iter = BP.iter_wikitext_articles
    if args.smoke:
        # Smoke limiter: cap the PRODUCTION keep-all branch (n_articles=None →
        # iter_wikitext_articles(None)) at N real streamed articles. Same code
        # branch as production — scale-only narrowing, no path substitution.
        # The inner generator is CLOSED explicitly: an abandoned suspended HF
        # streaming iterator survives to interpreter shutdown and hangs/aborts
        # finalization (gotchas.md HF-datasets shutdown class, #952/#1947 —
        # reproduced here as a >500 s post-[phase=done] hang, rc=124).
        def _limited_iter(max_articles):
            cap = (
                args.smoke_articles
                if max_articles is None
                else min(int(max_articles), args.smoke_articles)
            )
            inner = orig_iter(None)
            try:
                yield from itertools.islice(inner, cap)
            finally:
                inner.close()  # releases the streaming pipeline deterministically

        BP.iter_wikitext_articles = _limited_iter
    try:
        armc = BP.build_armc_pairs(
            tokenizer,
            n_articles=None,
            max_anchors=args.max_anchors,
            pool_multiplier=3,
            article_cap_tokens=args.article_cap_tokens,
            record_sep_char=True,
        )
    finally:
        BP.iter_wikitext_articles = orig_iter
    gc.collect()  # drop any residual streaming-pipeline refs while healthy (#952/#1947)
    articles, pairs = armc["articles"], armc["pairs"]
    assert pairs, "no eligible pairs built"
    logger.info(
        "[b0] built %d pairs over %d articles (%.0fs)", len(pairs), len(articles), time.time() - t0
    )
    ids_by_art = {a["window_id"]: a["input_ids"] for a in articles}
    span_texts = _decode_span_texts(tokenizer, ids_by_art, pairs)
    types = [p.meta["sep_char"] for p in pairs]
    idx_by_art_type: dict[tuple[str, str], list[int]] = defaultdict(list)
    idx_by_type: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        idx_by_art_type[(p.group_id, types[i])].append(i)
        idx_by_type[types[i]].append(i)

    # Article-grain split (seed 42; ONE sequential rng stream, stage order fixed).
    eval_target = _eval_margin(args)
    rng = np.random.default_rng(args.split_seed)
    arts_dot = sorted({pairs[i].group_id for i in idx_by_type["."]})
    order = rng.permutation(len(arts_dot))
    eval_articles: list[str] = []
    eval_sel: dict[str, list[int]] = {t: [] for t in ".?!"}
    seen_eval: dict[str, set] = {t: set() for t in ".?!"}
    drops = Counter()
    for oi in order:
        if len(eval_sel["."]) >= eval_target:
            break
        gid = arts_dot[int(oi)]
        eval_articles.append(gid)
        for t in ".?!":
            taken = 0
            for i in idx_by_art_type.get((gid, t), []):
                if taken >= EVAL_SPANS_PER_ARTICLE_CAP:
                    drops[f"eval_article_cap_dropped_{t}"] += 1
                    continue
                norm = _norm(span_texts[i])
                if norm in seen_eval[t]:
                    drops[f"eval_exact_dup_skipped_{t}"] += 1
                    continue
                seen_eval[t].add(norm)
                eval_sel[t].append(i)
                taken += 1
    eval_art_set = set(eval_articles)
    if not args.smoke:
        assert len(eval_sel["."]) >= EVAL_MARGIN, (
            f"eval '.' spans {len(eval_sel['.'])} < {EVAL_MARGIN} — articles exhausted; "
            f"a builder/corpus defect, inspect before p1"
        )

    # Per-type test/val assignment (seeded shuffle; production '.' = exactly 1000/400).
    split_of: dict[int, str] = {}
    eval_pools: dict[str, dict[str, list[int]]] = {}
    for t in ".?!":
        sel = eval_sel[t]
        perm = rng.permutation(len(sel))
        shuffled = [sel[int(j)] for j in perm]
        n_test, n_val = _test_val_counts(len(shuffled), bool(args.smoke))
        pool = {"test": shuffled[:n_test], "val": shuffled[n_test : n_test + n_val]}
        drops[f"eval_unassigned_dropped_{t}"] += len(shuffled) - n_test - n_val
        eval_pools[t] = pool
        for lbl in ("test", "val"):
            for i in pool[lbl]:
                split_of[i] = lbl
    # Pooled {.!?} eval: seeded draw from the union of assigned per-type eval spans.
    union_eval = [i for t in ".?!" for lbl in ("test", "val") for i in eval_pools[t][lbl]]
    perm = rng.permutation(len(union_eval))
    pooled_take = [union_eval[int(j)] for j in perm][: min(eval_target, len(union_eval))]
    n_test_p, n_val_p = _test_val_counts(len(pooled_take), bool(args.smoke))
    pooled_split_of: dict[int, str] = {}
    for i in pooled_take[:n_test_p]:
        pooled_split_of[i] = "test"
    for i in pooled_take[n_test_p : n_test_p + n_val_p]:
        pooled_split_of[i] = "val"

    # Screens over train candidates: transposed NearDupeGate (index = EVERY
    # selected eval span text, all types) + exact dedup WITHIN each train pool.
    gate_targets = [span_texts[i] for i in split_of]
    gate = NearDupeGate(gate_targets)
    assert gate.ngram == 5 and abs(gate.thresh - 0.8) < 1e-12, (gate.ngram, gate.thresh)
    train_kept: dict[str, list[int]] = {}
    for t in ".?!":
        cands = [i for i in idx_by_type[t] if pairs[i].group_id not in eval_art_set]
        perm = rng.permutation(len(cands))
        seen: set = set()
        kept: list[int] = []
        for j in perm:
            i = cands[int(j)]
            norm = _norm(span_texts[i])
            if norm in seen:
                drops[f"train_exact_dup_dropped_{t}"] += 1
                continue
            if gate.is_dupe(span_texts[i]):
                drops[f"train_near_dupe_dropped_{t}"] += 1
                continue
            seen.add(norm)
            kept.append(i)
        train_kept[t] = kept
    pooled_train = [i for t in ".?!" for i in train_kept[t]]
    perm = rng.permutation(len(pooled_train))
    pooled_order_of = {pooled_train[int(j)]: k for k, j in enumerate(perm)}

    # §7.1 yield gates + grid semantics (common_rungs used EVERYWHERE).
    realized_dot = len(eval_pools["."]["test"]) + len(eval_pools["."]["val"]) + len(train_kept["."])
    common_rungs = [g for g in BANKED_CHAT_GRID if g <= realized_dot - eval_target]
    top_common_rung = max(common_rungs) if common_rungs else None
    yield_gate = {
        "realized_dot_pairs_after_screens": realized_dot,
        "halt_floor": HALT_FLOOR,
        "pass": bool(realized_dot >= HALT_FLOOR),
        "common_rungs": common_rungs,
        "top_common_rung": top_common_rung,
        "smoke": bool(args.smoke),
    }
    if args.smoke:
        logger.info("[b0] (smoke, informational) yield gate: %s", yield_gate)
    else:
        logger.info("[b0] yield gate: %s", yield_gate)

    # Manifest rows (dropped rows are counted, never written).
    train_order_of: dict[int, int] = {}
    for t in ".?!":
        for k, i in enumerate(train_kept[t]):
            train_order_of[i] = k
    rows = []
    anchor_ids: dict[str, Counter] = {t: Counter() for t in ".?!"}
    span_lens: dict[str, list[int]] = {t: [] for t in ".?!"}
    for i, p in enumerate(pairs):
        in_eval = i in split_of
        in_train = i in train_order_of
        if not (in_eval or in_train):
            continue
        t = types[i]
        lo, hi = p.t_spans[0]
        anchor_ids[t][int(ids_by_art[p.group_id][int(p.meta["anchor_pos"])])] += 1
        span_lens[t].append(hi - lo)
        rows.append(
            {
                "row_id": p.row_id,
                "article_id": p.group_id,
                "sep_char": t,
                "anchor_pos": int(p.meta["anchor_pos"]),
                "c_span": list(p.c_span),
                "t_span": [int(lo), int(hi)],
                "n_span_tokens": int(hi - lo),
                "split": split_of.get(i, "train"),
                "train_order": train_order_of.get(i),
                "pooled_order": pooled_order_of.get(i),
                "pooled_split": pooled_split_of.get(i),
            }
        )
    kept_arts = sorted({r["article_id"] for r in rows})

    # Write manifest JSONL shards (<9 MB line-split), article token shards (int32).
    if man_dir.exists():
        shutil.rmtree(man_dir)  # partial prior build; regime key gates full reuse above
    man_dir.mkdir(parents=True, exist_ok=True)
    shard_files: list[str] = []
    buf: list[str] = []
    buf_bytes = 0
    shard_i = 0

    def _flush_manifest():
        nonlocal buf, buf_bytes, shard_i
        if not buf:
            return
        name = f"manifest_shard{shard_i:03d}.jsonl"
        tmp = man_dir / (name + ".tmp")
        tmp.write_text("\n".join(buf) + "\n")
        tmp.replace(man_dir / name)
        shard_files.append(name)
        buf, buf_bytes, shard_i = [], 0, shard_i + 1

    for r in rows:
        line = json.dumps(r)
        if buf_bytes + len(line) > 8_500_000:
            _flush_manifest()
        buf.append(line)
        buf_bytes += len(line) + 1
    _flush_manifest()

    art_shard_files: list[str] = []
    for k in range(0, len(kept_arts), 2000):
        chunk = kept_arts[k : k + 2000]
        payload = {
            "window_ids": chunk,
            "input_ids": [torch.as_tensor(ids_by_art[g], dtype=torch.int32) for g in chunk],
        }
        name = f"articles_shard{k // 2000:03d}.pt"
        tmp = man_dir / (name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(man_dir / name)
        art_shard_files.append(name)

    per_type = {
        t: {
            "n_eligible": len(idx_by_type[t]),
            "n_eval_test": len(eval_pools[t]["test"]),
            "n_eval_val": len(eval_pools[t]["val"]),
            "n_train_kept": len(train_kept[t]),
            "anchor_token_ids": {str(k): int(v) for k, v in anchor_ids[t].most_common()},
            "span_length_stats": _span_length_stats(span_lens[t]),
        }
        for t in ".?!"
    }
    meta = {
        "regime_key": regime_key,
        "n_pairs_built": len(pairs),
        "n_articles_built": len(articles),
        "n_manifest_rows": len(rows),
        "n_articles_kept": len(kept_arts),
        "n_eval_articles": len(eval_articles),
        "per_type": per_type,
        "pooled": {
            "n_eval_test": n_test_p,
            "n_eval_val": n_val_p,
            "n_train": len(pooled_train),
        },
        "screens": {**{k: int(v) for k, v in drops.items()}, **gate.stats()},
        "yield_gate": yield_gate,
        "common_rungs": common_rungs,
        "top_common_rung": top_common_rung,
        "manifest_shards": shard_files,
        "article_shards": art_shard_files,
        "n_files_total": len(shard_files) + len(art_shard_files) + 1,  # + meta.json itself
        "build_wall_s": round(time.time() - t0, 1),
        "metadata": _meta("b0-pairs", {"split_seed": args.split_seed}),
    }
    C79.write_json_atomic(man_dir / "meta.json", meta)
    logger.info(
        "[b0] manifest written: %d rows, %d manifest shards, %d article shards (%.0fs)",
        len(rows),
        len(shard_files),
        len(art_shard_files),
        time.time() - t0,
    )
    return meta


# ── p1_capture ──────────────────────────────────────────────────────────────────


def _manifest_dir(args) -> Path:
    """Resolve the manifest source: local b0 output, else HF-staged (A13 contract)."""
    local = args.out_root / "manifest"
    if (local / "meta.json").exists():
        logger.info(
            "[stage] manifest staged: %d files (local b0 output)",
            sum(1 for p in local.iterdir() if p.is_file()),
        )
        return local
    staged = PD.stage_prefix(MANIFEST_PREFIX, args.out_root / "staging", workers=args.stage_workers)
    n = sum(1 for p in staged.iterdir() if p.is_file())
    meta = json.loads((staged / "meta.json").read_text())
    logger.info("[stage] manifest staged: %d files", n)
    assert n == meta["n_files_total"], (
        f"[stage] staged file count {n} != manifest meta n_files_total "
        f"{meta['n_files_total']} — incomplete b0 upload set (A13)"
    )
    return staged


def _load_manifest(man_dir: Path) -> tuple[list[dict], dict, dict]:
    """Read manifest rows + article ids + meta from the staged/local manifest dir."""
    meta = json.loads((man_dir / "meta.json").read_text())
    rows: list[dict] = []
    for name in meta["manifest_shards"]:
        rows.extend(ES._read_jsonl(man_dir / name))
    assert len(rows) == meta["n_manifest_rows"], (len(rows), meta["n_manifest_rows"])
    ids_by_art: dict[str, list[int]] = {}
    for name in meta["article_shards"]:
        payload = torch.load(man_dir / name, map_location="cpu", weights_only=True)
        for gid, ids in zip(payload["window_ids"], payload["input_ids"]):
            ids_by_art[gid] = [int(v) for v in ids.tolist()]
    return rows, ids_by_art, meta


def _items_from_manifest(rows: list[dict], ids_by_art: dict) -> list[dict]:
    """Rebuild ES-shaped items (article + validated PairSpecs) from manifest rows."""
    by_art: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_art[r["article_id"]].append(r)
    items = []
    for gid, arts in by_art.items():
        ids = ids_by_art[gid]
        specs = []
        for r in arts:
            p = common.PairSpec(
                row_id=r["row_id"],
                group_id=gid,
                char_id="sep",
                c_span=tuple(r["c_span"]),
                t_spans=[tuple(r["t_span"])],
                ctx_span=tuple(r["c_span"]),
                meta={"window_id": gid, "anchor_pos": r["anchor_pos"], "sep_char": r["sep_char"]},
            )
            p.validate(len(ids), min_c=common.ARMC_PREV_MIN_TOKENS, min_t=common.ARMC_SPAN_MIN)
            specs.append(p)
        items.append({"item_id": gid, "group_id": gid, "input_ids": ids, "pairs": specs})
    return items


def _smoke_row_subset(rows: list[dict], dot_train_cap: int) -> list[dict]:
    """Balanced p1 smoke subset: ALL eval rows (every arm's test/val pools) + the
    first ``dot_train_cap`` '.'-train rows + a few '?'/'!' train rows — so the p2
    smoke reaches every arm class (headline, companions, second-type, pooled,
    group-diversity) instead of whatever rows happen to lead the manifest."""

    def _train(t: str, k: int) -> list[dict]:
        trs = sorted(
            (r for r in rows if r["split"] == "train" and r["sep_char"] == t),
            key=lambda r: r["train_order"],
        )
        return trs[:k]

    evals = [r for r in rows if r["split"] in ("test", "val")]
    return evals + _train(".", dot_train_cap) + _train("?", 12) + _train("!", 12)


def _scan_store_resume(store: Path, persist: tuple[int, ...]) -> tuple[set, int]:
    """Resume scan: validate sidecar layer sets, return (done row_ids, next shard idx)."""
    done: set = set()
    max_idx = -1
    for side in sorted(store.glob("pairs_shard*.json")):
        d = json.loads(side.read_text())
        assert d.get("layers") == list(persist), (
            f"resume layer-set mismatch in {side.name}: {d.get('layers')} != {list(persist)}"
        )
        assert (store / side.name.replace(".json", ".pt")).exists(), f"orphan sidecar {side.name}"
        done.update(d["row_ids"])
        max_idx = max(max_idx, int(d["shard_index"]))
    return done, max_idx + 1


def _upload_shard_batch(store: Path, names: list[str], scratch: Path) -> float:
    """Batched upload of one shard batch (hardlink staging → hub._upload folder path).

    Returns the batch wall seconds. The canonical helper carries the dir-filecount
    guard + repo verification; hardlinks make the staging free.
    """
    t0 = time.time()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for name in names:
        os.link(store / name, scratch / name)
    url = hub._upload(
        scratch, HF_DATA_REPO, "dataset", path_in_repo=CAPTURE_PREFIX, raise_on_error=True
    )
    if not url:
        raise RuntimeError(f"[p1] shard batch upload returned empty url ({names[:3]}...)")
    shutil.rmtree(scratch)
    return time.time() - t0


def phase_p1_capture(args) -> int:  # noqa: C901 -- capture stream + pilot gate + batched uploads
    """Stage manifest, capture x_sep + y at the persist layers, upload-as-you-go."""
    C79.phase("p1_capture")
    store = args.out_root / "store"
    store.mkdir(parents=True, exist_ok=True)
    man_dir = _manifest_dir(args)
    rows, ids_by_art, meta = _load_manifest(man_dir)

    tiny_dir = None
    if args.smoke:
        tiny_dir = args.tiny_model_dir or str(args.out_root / "tiny_model")
        if not (Path(tiny_dir) / "config.json").exists():
            ES.make_tiny_model(Path(tiny_dir), layers=args.tiny_layers)
    elif args.tiny_model_dir:
        tiny_dir = args.tiny_model_dir
    persist = SMOKE_PERSIST_LAYERS if tiny_dir else PROD_PERSIST_LAYERS
    model = ES.load_model(tiny_dir)
    tok = common.get_tokenizer(tiny_dir or common.MODEL_ID)
    pad_id = tok.pad_token_id
    assert pad_id is not None, "tokenizer has no pad token id"

    done, next_idx = _scan_store_resume(store, persist)
    rows_used = _smoke_row_subset(rows, args.smoke_pairs) if args.smoke else rows
    pending_rows = [r for r in rows_used if r["row_id"] not in done]
    n_shards_total = math.ceil(len(rows_used) / args.shard_pairs)
    n_pending_shards = math.ceil(len(pending_rows) / args.shard_pairs)
    per_shard_gb = args.shard_pairs * len(persist) * ES.EXPECTED_HIDDEN * 2 * 2 / 1e9
    need_gb = per_shard_gb * n_pending_shards * 1.15 + 2.0
    if n_pending_shards == 0:
        logger.info("[p1] zero pending shards — capture already complete; headroom gate skipped")
    else:
        assert_out_root_headroom(args.out_root, need_gb, phase="p1")

    items = _items_from_manifest(pending_rows, ids_by_art)
    logger.info(
        "[p1] capture: %d pending pairs over %d articles (%d/%d shards done), layers=%s",
        len(pending_rows),
        len(items),
        n_shards_total - n_pending_shards,
        n_shards_total,
        list(persist),
    )

    shard_names: list[str] = []
    batch_pending: list[str] = []
    scratch = args.out_root / "upload_batch"
    capture_wall = upload_wall = None
    pilot_checked = next_idx > 0  # a resumed run already passed the pilot gate
    buf: list[dict] = []
    shard_idx = next_idx
    t_shard = time.time()

    def _write_one_shard(records: list[dict]) -> str:
        nonlocal shard_idx
        ES.write_shard(records, store, shard_idx, "armC", layers=persist)
        old_pt = store / f"armC_shard{shard_idx:03d}.pt"
        old_js = store / f"armC_shard{shard_idx:03d}.json"
        new_pt = store / f"pairs_shard{shard_idx:03d}.pt"
        new_js = store / f"pairs_shard{shard_idx:03d}.json"
        old_pt.replace(new_pt)  # plan §6.5 deliverable glob is pairs_shard*
        old_js.replace(new_js)
        shard_idx += 1
        return new_pt.name

    def _maybe_upload(force: bool = False) -> None:
        nonlocal upload_wall
        if args.smoke or args.skip_upload:
            return
        if batch_pending and (force or len(batch_pending) >= 10):
            names = [n for base in batch_pending for n in (base, base.replace(".pt", ".json"))]
            wall = _upload_shard_batch(store, names, scratch)
            if upload_wall is None:
                upload_wall = wall
            logger.info("[p1] uploaded batch of %d shards (%.0fs)", len(batch_pending), wall)
            batch_pending.clear()

    for recs in ES.run_extraction(model, items, pad_id, args.batch_size, "armC", layers=persist):
        buf.extend(recs)
        while len(buf) >= args.shard_pairs:
            name = _write_one_shard(buf[: args.shard_pairs])
            buf = buf[args.shard_pairs :]
            shard_names.append(name)
            batch_pending.append(name)
            if capture_wall is None:
                capture_wall = time.time() - t_shard
                _maybe_upload(force=True)  # shard 1 = its own timed 1-shard batch
                _pilot_gate(args, capture_wall, upload_wall, n_shards_total)
                pilot_checked = True
            else:
                _maybe_upload()
            logger.info(
                "[p1] unit %d/%d %s elapsed=%.0fs",
                shard_idx,
                n_shards_total,
                name,
                time.time() - t_shard,
            )
            t_shard = time.time()
    if buf:
        name = _write_one_shard(buf)
        shard_names.append(name)
        batch_pending.append(name)
        if capture_wall is None:
            capture_wall = time.time() - t_shard
            _maybe_upload(force=True)
            _pilot_gate(args, capture_wall, upload_wall, n_shards_total)
            pilot_checked = True
    _maybe_upload(force=True)
    assert pilot_checked or not shard_names, "pilot gate never evaluated"

    all_names = sorted(p.name for p in store.glob("pairs_shard*.pt"))
    if not (args.smoke or args.skip_upload):
        from huggingface_hub import HfApi

        expected = [
            f"{CAPTURE_PREFIX}/{n}" for base in all_names for n in (base, base[:-3] + ".json")
        ]
        missing = hub.verify_repo_paths_uploaded(
            HfApi(), HF_DATA_REPO, expected, path_in_repo=CAPTURE_PREFIX
        )
        if missing:
            raise RuntimeError(f"[p1] capture upload verify FAILED — missing: {missing[:5]}")
        logger.info("[p1] capture verified on Hub: %d files", len(expected))
    C79.write_json_atomic(
        args.out_root / "p1_state.json",
        {
            "n_shards": len(all_names),
            "shard_files": all_names,
            "persist_layers": list(persist),
            "n_rows_manifest": len(rows),
            "capture_wall_s_shard1": capture_wall,
            "upload_wall_s_batch1": upload_wall,
            "metadata": _meta("p1-capture"),
        },
    )
    return 0


def _pilot_gate(args, capture_wall: float, upload_wall: float | None, n_shards: int) -> None:
    """§7.2 capture pilot abort — components timed SEPARATELY, upload charged per
    10-shard batch (never per shard, the registered anti-pattern)."""
    up = upload_wall or 0.0
    projected_s = capture_wall * n_shards + up * math.ceil(n_shards / 10)
    budget_s = args.p1_wall_budget_h * 3600
    report = {
        "capture_wall_s_shard1": capture_wall,
        "upload_wall_s_batch1": upload_wall,
        "n_shards": n_shards,
        "projected_p1_wall_s": projected_s,
        "budget_s": budget_s,
        "pass": bool(projected_s <= budget_s),
        "formula": "capture_wall * n_shards + upload_wall * ceil(n_shards/10)",
    }
    logger.info("[p1] pilot gate: %s", report)
    if args.smoke:
        return  # production-wall-calibrated gate — informational under smoke (#1345)
    if not report["pass"]:
        C79.write_json_atomic(args.out_root / "p1_pilot_halt_report.json", report)
        logger.error(
            "[p1] PILOT HALT (§7.2): projected %.1fh > %.1fh", projected_s / 3600, budget_s / 3600
        )
        # Designed artifact-routed halt at a DISTINCT rc (gotchas.md pilot-gate
        # rule) — SystemExit propagates cleanly through main()'s exit path.
        raise SystemExit(RC_PILOT_HALT)


# ── p2_fits ─────────────────────────────────────────────────────────────────────


def _load_layer_arrays(store: Path, layer: int, persist: tuple[int, ...]):
    """One pass over the store for ONE layer: (X, Y) torch bf16 + row/article ids."""
    sidecars = sorted(store.glob("pairs_shard*.json"))
    assert sidecars, f"no capture shards under {store}"
    counts = []
    for side in sidecars:
        d = json.loads(side.read_text())
        assert d.get("layers") == list(persist), (side.name, d.get("layers"), list(persist))
        counts.append(int(d["n_rows"]))
    n_total = sum(counts)
    col = list(persist).index(layer)
    hidden = None
    X = Y = None
    row_ids: list[str] = []
    art_ids: list[str] = []
    pos = 0
    for side in sidecars:
        pt = store / side.name.replace(".json", ".pt")
        b = torch.load(pt, map_location="cpu", weights_only=True)
        x = b["arrays"]["x_sep"][:, col, :]
        y = b["arrays"]["y"][:, col, :]
        if X is None:
            hidden = x.shape[1]
            # X stays bf16 (fit_ridge's factorize/predict path is torch-native,
            # fp64-casting per block). Y upcasts to fp32 at load: fit_ridge's
            # val scoring calls np.asarray(Y[val]) internally, and numpy cannot
            # convert torch bf16 (TypeError: unsupported ScalarType BFloat16 —
            # caught by the unit-2 fits micro-test).
            X = torch.empty((n_total, hidden), dtype=x.dtype)
            Y = torch.empty((n_total, hidden), dtype=torch.float32)
        X[pos : pos + x.shape[0]] = x
        Y[pos : pos + y.shape[0]] = y
        row_ids.extend(b["row_ids"])
        art_ids.extend(b["group_ids"])
        pos += x.shape[0]
    assert pos == n_total, (pos, n_total)
    return X, Y, row_ids, art_ids


def _to_f64_np(t: torch.Tensor, idx: np.ndarray) -> np.ndarray:
    return t[torch.as_tensor(idx, dtype=torch.long)].to(torch.float64).numpy()


def _identity_bias_chunked(X, Y, tr: np.ndarray, te: np.ndarray, block: int = 50_000):
    """W=identity + train-mean bias, fp64 chunked (RAM-safe at n ~ 1M); the small-n
    helper-parity assert vs ``MB.identity_bias_predict`` runs at every rung ≤ 2,500."""
    s = torch.zeros(X.shape[1], dtype=torch.float64)
    for k in range(0, len(tr), block):
        idx = torch.as_tensor(tr[k : k + block], dtype=torch.long)
        s += (Y[idx].to(torch.float64) - X[idx].to(torch.float64)).sum(0)
    bias = (s / len(tr)).numpy()
    pred = _to_f64_np(X, te) + bias
    if len(tr) <= 2500:
        ref = MB.identity_bias_predict(_to_f64_np(X, tr), _to_f64_np(Y, tr), _to_f64_np(X, te))
        assert np.allclose(pred, ref, rtol=1e-9, atol=1e-6), "identity+bias chunked != helper"
    return pred


def _rank_matrix(pred: np.ndarray, pool: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """(n_pred, n_pool) mid-rank matrix (rank of EVERY pool item per pred row),
    tolerance-tied exactly like ``MB.knn_retrieval``; used by the advisory null."""
    d = MB._pairwise_dist(pred, pool, metric)
    n, m = d.shape
    ranks = np.empty((n, m))
    blk = max(1, int(5e7 // (m * m)) or 1)
    for s in range(0, n, blk):
        db = d[s : s + blk]  # (b, m)
        tol = 1e-9 * np.maximum(np.abs(db)[:, :, None], 1e-12)
        closer = (db[:, None, :] < db[:, :, None] - tol).sum(-1)
        tied = (np.abs(db[:, None, :] - db[:, :, None]) <= tol).sum(-1) - 1
        ranks[s : s + blk] = 1.0 + closer + 0.5 * tied
    return ranks


def shuffled_pair_null(pred: np.ndarray, y_te: np.ndarray, k_draws: int, seed: int) -> dict:
    """200-draw shuffled-pair null (§4 p2; ADVISORY — nothing branches on it).

    Test-pairing permutation, pure vectorized re-reduction: R² via one pred@Yᵀ
    GEMM + per-draw gathers; acc@1 via the mid-rank matrix.
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y_te, dtype=np.float64)
    n = pred.shape[0]
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n) for _ in range(k_draws)])
    c_pp = float((pred**2).sum())
    c_yy = float((y**2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    P = pred @ y.T  # (n, n)
    cross = P[np.arange(n)[None, :], perms].sum(axis=1)
    r2_draws = 1.0 - (c_pp + c_yy - 2.0 * cross) / ss_tot
    ranks = _rank_matrix(pred, y, "euclidean")
    acc1_draws = (ranks[np.arange(n)[None, :], perms] <= 1).mean(axis=1)

    def _summ(a):
        return {
            "mean": float(a.mean()),
            "p2_5": float(np.percentile(a, 2.5)),
            "p97_5": float(np.percentile(a, 97.5)),
        }

    return {
        "advisory": True,
        "k_draws": int(k_draws),
        "seed": int(seed),
        "r2": _summ(r2_draws),
        "acc1_euclidean": _summ(acc1_draws),
    }


def article_cluster_boot(
    pred: np.ndarray, y_te: np.ndarray, art_ids: list[str], n_boot: int, seed: int
) -> dict:
    """§3 control-side CI: article-level cluster bootstrap of pooled R² (resample
    eval ARTICLES with replacement; per-draw sufficient-stats gather+reduce,
    ss_tot re-centered on the resampled mean — the ``_bootstrap_recon_ci``
    convention at article grain)."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y_te, dtype=np.float64)
    arts = sorted(set(art_ids))
    a_of = {g: k for k, g in enumerate(arts)}
    lab = np.asarray([a_of[g] for g in art_ids])
    n_a = np.bincount(lab, minlength=len(arts)).astype(np.float64)
    res_i = ((y - pred) ** 2).sum(axis=1)
    s_res = np.bincount(lab, weights=res_i, minlength=len(arts))
    s_y = np.zeros((len(arts), y.shape[1]))
    np.add.at(s_y, lab, y)
    s_yy = np.bincount(lab, weights=(y**2).sum(axis=1), minlength=len(arts))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arts), size=(n_boot, len(arts)))
    m = np.zeros((n_boot, len(arts)))
    np.add.at(m, (np.arange(n_boot)[:, None], draws), 1.0)
    n_star = m @ n_a
    ss_res = m @ s_res
    mean_star = (m @ s_y) / n_star[:, None]
    ss_tot = m @ s_yy - n_star * (mean_star**2).sum(axis=1)
    ok = ss_tot > 1e-12
    r2 = 1.0 - ss_res[ok] / ss_tot[ok]
    point = 1.0 - float(res_i.sum()) / float(((y - y.mean(0)) ** 2).sum())
    return {
        "point": point,
        "lo": float(np.percentile(r2, 2.5)),
        "hi": float(np.percentile(r2, 97.5)),
        "n_articles": len(arts),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_degenerate_draws": int((~ok).sum()),
    }


def _chat_ci_table() -> dict:
    """Chat-side per-rung R² intervals: recomputed from the committed artifacts when
    present (asserted vs the pasted constants), else the constants (pod-cone-safe)."""
    tbl = {n: dict(v) for n, v in CHAT_RIDGE_L19_R2.items()}
    lad_p = _CHAT_SOURCE_FILES["ladder"]
    if lad_p.exists():
        lad = json.loads(lad_p.read_text())
        by_n: dict[int, list[float]] = defaultdict(list)
        for c in lad["cells"]:
            by_n[int(c["n_train"])].append(float(c["ridge"]["test_r2"]))
        for n, vals in by_n.items():
            got = float(np.mean(vals))
            assert abs(got - tbl[n]["point"]) < 1e-8, (n, got, tbl[n]["point"])
            tbl[n]["point"] = got
    for key, n, extract in (
        ("n50k", 50000, lambda d: d["per_layer"]["19"]["ridge"]),
        ("bign", 150000, lambda d: d["per_point"]["lmsys_150k"]["ridge"]),
        ("bign", 500000, lambda d: d["per_point"]["lmsys_500k"]["ridge"]),
    ):
        p = _CHAT_SOURCE_FILES[key]
        if p.exists():
            cell = extract(json.loads(p.read_text()))
            ci = cell["bootstrap_ci"]["r2"]
            for fld, got in (("point", cell["whole_map_r2"]), ("lo", ci["lo"]), ("hi", ci["hi"])):
                assert abs(float(got) - tbl[n][fld]) < 1e-8, (n, fld, got, tbl[n][fld])
    bat_p = _CHAT_SOURCE_FILES["battery"]
    if bat_p.exists():
        r2 = json.loads(bat_p.read_text())["per_layer"]["19"]["arms"]["ridge"]["r2"]
        for fld in ("point", "lo", "hi"):
            assert abs(float(r2[fld]) - tbl[963444][fld]) < 1e-8, (fld, r2[fld])
    for n, v in tbl.items():
        v["source"] = "banked-bootstrap-ci" if v["lo"] is not None else "banked-point-degenerate"
    return tbl


def gap_interval_entry(n_train: int, control_cells: list[dict], chat_row: dict) -> dict:
    """§3 conservative interval difference for one common L19 rung.

    Control CI = the across-draw ENVELOPE of per-draw article-cluster CIs
    (over-covers, the registered honest direction). Chat rows without a banked
    CI (the ≤25k ladder cells) degrade to a POINT interval, flagged via
    chat_ci_source — a disclosed artifact-grounded deviation from plan §3's
    "committed cells' bootstrap CI" wording (those cells carry none).
    """
    cis = [c["article_ci"] for c in control_cells]
    control = {
        "point": float(np.mean([c["point"] for c in cis])),
        "lo": float(min(c["lo"] for c in cis)),
        "hi": float(max(c["hi"] for c in cis)),
        "n_draws": len(cis),
        "per_draw": cis,
    }
    chat_lo = chat_row["lo"] if chat_row["lo"] is not None else chat_row["point"]
    chat_hi = chat_row["hi"] if chat_row["hi"] is not None else chat_row["point"]
    return {
        "n_train": int(n_train),
        "chat_ci": {k: chat_row[k] for k in ("point", "lo", "hi")},
        "chat_ci_source": chat_row["source"],
        "control_article_ci": control,
        "gap_lo": float(chat_lo - control["hi"]),
        "gap_hi": float(chat_hi - control["lo"]),
    }


def fit_one_cell(
    X,
    Y,
    tr: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    *,
    dev: torch.device,
    n_boot: int,
    n_null: int,
    seed: int,
    layer: int,
    draw,
    art_ids_te: list[str] | None = None,
    n_article_boot: int = 1000,
    block: int | None = None,
) -> dict:
    """One (rung × draw) cell in the scaling_ladder_L19.json cell schema, extended
    with const_mean + the advisory null (+ per-draw article CI when art_ids given)."""
    n_train = len(tr)
    lambdas = _lambdas_for(n_train)
    block = block or N1M.RIDGE_BLOCK
    pred, fit_meta = N1M.fit_ridge(X, Y, tr, val, te, lambdas, dev, block)
    y_te = _to_f64_np(Y, te)
    ridge_score = PD.score_cell(pred, y_te, n_boot, seed)
    pred_ib = _identity_bias_chunked(X, Y, tr, te)
    ib_score = PD.score_cell(pred_ib, y_te, n_boot, seed)
    # constant train-mean predictor: chunked fp64 train mean of Y, tiled over test
    s = torch.zeros(Y.shape[1], dtype=torch.float64)
    for k in range(0, n_train, block):
        idx = torch.as_tensor(tr[k : k + block], dtype=torch.long)
        s += Y[idx].to(torch.float64).sum(0)
    pred_const = np.broadcast_to((s / n_train).numpy(), y_te.shape).copy()
    const_score = PD.score_cell(pred_const, y_te, n_boot, seed)
    cell = {
        "draw": draw,
        "layer": int(layer),
        "n_train": int(n_train),
        "n_vs_d": {
            "n_train": int(n_train),
            "d": int(X.shape[1]),
            "underdetermined": bool(n_train < X.shape[1]),
        },
        "ridge": {
            "test_r2": ridge_score["whole_map_r2"],
            "mean_cosine": ridge_score["mean_cosine"],
            "bootstrap_ci": ridge_score["bootstrap_ci"],
            "meta": fit_meta,
        },
        "identity_bias": {
            "test_r2": ib_score["whole_map_r2"],
            "bootstrap_ci": ib_score["bootstrap_ci"],
        },
        "const_mean": {
            "test_r2": const_score["whole_map_r2"],
            "acc_at_1_euclidean": const_score["retrieval"]["euclidean"]["acc_at_k"][1],
        },
        "knn": {
            "ridge": ridge_score["retrieval"],
            "identity_bias": ib_score["retrieval"],
        },
        "null": shuffled_pair_null(pred, y_te, n_null, seed),
    }
    if art_ids_te is not None:
        cell["article_ci"] = article_cluster_boot(pred, y_te, art_ids_te, n_article_boot, seed)
        cell["point"] = cell["ridge"]["test_r2"]
        cell["n_groups"] = {
            "eval_articles": len(set(art_ids_te)),
        }
    return cell


def _device_parity_gate(X, Y, tr, val, te, *, smoke: bool) -> dict:
    """§7.3: rung-50/draw-0 ridge on cuda vs cpu, fp64, |ΔR²| ≤ 1e-6 (check (m))."""
    if not torch.cuda.is_available():
        row = {"pass": None, "note": "cuda unavailable — gate deferred to the GPU pod"}
        assert smoke, "p2 production entry requires cuda for the §7.3 device parity gate"
        logger.info("[p2] (smoke, informational) device parity gate: %s", row)
        return row
    lambdas = _lambdas_for(len(tr))
    r2 = {}
    for devname in ("cuda", "cpu"):
        pred, _ = N1M.fit_ridge(X, Y, tr, val, te, lambdas, torch.device(devname), N1M.RIDGE_BLOCK)
        r2[devname] = float(F79._recon_point(pred, _to_f64_np(Y, te))[0])
    diff = abs(r2["cuda"] - r2["cpu"])
    row = {
        "r2_cuda": r2["cuda"],
        "r2_cpu": r2["cpu"],
        "abs_diff": diff,
        "tol": 1e-6,
        "pass": bool(diff <= 1e-6),
    }
    if not row["pass"]:
        raise RuntimeError(f"[p2] DEVICE PARITY GATE FAILED (§7.3): {row}")
    logger.info("[p2] device parity gate PASS: %s", row)
    return row


def _cell_unit_key(arm: str, layer: int, n: int, draw, tr: np.ndarray, args) -> dict:
    return {
        "arm": arm,
        "layer": int(layer),
        "n_train": int(n),
        "draw": draw,
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "lambda_grid": _lambda_grid_params(n),
        "n_boot": int(args.n_boot),
        "n_null": int(args.n_null),
        "train_sel_sha256": F79._sha_ids(np.asarray(tr, dtype=np.int64)),
    }


def _run_cell_checkpointed(units_dir: Path, key: dict, fit_fn) -> dict:
    """Per-cell checkpoint: skip when the unit JSON's key matches (params-keyed)."""
    name = f"{key['arm']}_L{key['layer']}_n{key['n_train']}_d{key['draw']}.json".replace("/", "_")
    path = units_dir / name
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("unit_key") == key:
            logger.info("[p2] resume-skip %s", name)
            return prev["cell"]
    t0 = time.time()
    cell = fit_fn()
    C79.write_json_atomic(path, {"unit_key": key, "cell": cell})
    logger.info(
        "[p2] unit %s n=%d r2=%.4f elapsed=%.0fs",
        name,
        key["n_train"],
        cell["ridge"]["test_r2"],
        time.time() - t0,
    )
    return cell


def _pool_indices(rows: list[dict], row_pos: dict) -> dict:
    """Manifest rows → per-arm index pools (positions into the store row order)."""

    def _positions(rs):
        return np.asarray([row_pos[r["row_id"]] for r in rs], dtype=np.int64)

    pools: dict = {}
    for t in ".?!":
        trs = sorted(
            (r for r in rows if r["sep_char"] == t and r["split"] == "train"),
            key=lambda r: r["train_order"],
        )
        pools[t] = {
            "train": _positions(trs),
            "train_rows": trs,
            "test": _positions([r for r in rows if r["sep_char"] == t and r["split"] == "test"]),
            "val": _positions([r for r in rows if r["sep_char"] == t and r["split"] == "val"]),
            "test_articles": [
                r["article_id"] for r in rows if r["sep_char"] == t and r["split"] == "test"
            ],
        }
    pooled_tr = sorted(
        (r for r in rows if r["pooled_order"] is not None), key=lambda r: r["pooled_order"]
    )
    pools["pooled"] = {
        "train": _positions(pooled_tr),
        "test": _positions([r for r in rows if r.get("pooled_split") == "test"]),
        "val": _positions([r for r in rows if r.get("pooled_split") == "val"]),
        "test_articles": [r["article_id"] for r in rows if r.get("pooled_split") == "test"],
    }
    return pools


def _draw_indices(pool: np.ndarray, n: int, draw) -> np.ndarray:
    """Parent draw convention: seeded subset at small rungs, file-order prefix above."""
    if draw == "prefix":
        return pool[:n]
    rng = np.random.default_rng(19010000 + n * 10 + int(draw))
    return pool[rng.choice(len(pool), size=n, replace=False)]


def phase_p2_fits(args) -> int:  # noqa: C901 -- rung/arm enumeration + assembly
    """Per rung × draw ridge/identity+bias/const fits + null + gap intervals."""
    C79.phase("p2_fits")
    store = args.out_root / "store"
    assert_out_root_headroom(args.out_root, 4.0, phase="p2")
    man_dir = _manifest_dir(args)
    rows, _ids_by_art, meta = _load_manifest(man_dir)
    units_dir = args.out_root / "fits_units"
    units_dir.mkdir(parents=True, exist_ok=True)
    eval_out = Path(args.eval_out)
    eval_out.mkdir(parents=True, exist_ok=True)

    sidecar0 = sorted(store.glob("pairs_shard*.json"))
    assert sidecar0, f"no capture shards under {store} — run p1 first"
    persist = tuple(json.loads(sidecar0[0].read_text())["layers"])
    smoke = bool(args.smoke)
    headline = HEADLINE_LAYER if HEADLINE_LAYER in persist else persist[2]
    companions = tuple(li for li in persist if li != headline)
    dev = torch.device(args.device)

    common_rungs = list(meta["common_rungs"])
    top_common = meta["top_common_rung"]
    if smoke:
        common_rungs = [n for n in common_rungs if n <= 50] or common_rungs[:1]
        top_common = max(common_rungs) if common_rungs else None
    assert common_rungs, "empty common_rungs — yield gate should have halted b0"
    for n in common_rungs:
        assert n in BANKED_CHAT_GRID, f"off-grid rung {n} (grid semantics violation, §7.1)"

    # Headline layer pass.
    X, Y, row_ids, _arts = _load_layer_arrays(store, headline, persist)
    row_pos = {rid: k for k, rid in enumerate(row_ids)}
    manifest_captured = [r for r in rows if r["row_id"] in row_pos]
    pools = _pool_indices(manifest_captured, row_pos)
    dot = pools["."]
    logger.info(
        "[p2] headline L%d: %d rows (train %d, test %d, val %d), rungs %s",
        headline,
        len(row_ids),
        len(dot["train"]),
        len(dot["test"]),
        len(dot["val"]),
        common_rungs,
    )
    assert len(dot["test"]) > 1 and len(dot["val"]) > 0, "degenerate eval pools"

    # §7.3 device parity gate at entry (rung-50/draw-0).
    n0 = min(50, len(dot["train"]))
    parity = _device_parity_gate(
        X, Y, _draw_indices(dot["train"], n0, 0), dot["val"], dot["test"], smoke=smoke
    )

    chat_tbl = _chat_ci_table()
    n_boot = args.n_boot
    n_null = args.n_null
    n_aboot = args.n_article_boot
    cells: list[dict] = []
    gap_entries: list[dict] = []
    for n in common_rungs:
        draws = list(SMALL_DRAWS) if n in SMALL_RUNGS else ["prefix"]
        rung_cells = []
        for d in draws:
            tr = _draw_indices(dot["train"], n, d)
            key = _cell_unit_key("dot", headline, n, d, tr, args)
            cell = _run_cell_checkpointed(
                units_dir,
                key,
                lambda tr=tr, d=d: fit_one_cell(
                    X,
                    Y,
                    tr,
                    dot["val"],
                    dot["test"],
                    dev=dev,
                    n_boot=n_boot,
                    n_null=n_null,
                    seed=args.seed,
                    layer=headline,
                    draw=d,
                    art_ids_te=dot["test_articles"],
                    n_article_boot=n_aboot,
                ),
            )
            # train-article count for this draw (group-n reporting, §6)
            tr_set = set(tr.tolist())
            cell["n_groups"] = dict(cell.get("n_groups") or {})
            cell["n_groups"]["train_articles"] = len(
                {r["article_id"] for r in dot["train_rows"] if row_pos[r["row_id"]] in tr_set}
            )
            rung_cells.append(cell)
        cells.extend(rung_cells)
        chat_row = chat_tbl.get(n)
        assert chat_row is not None, f"rung {n} missing from the banked chat table"
        gap_entries.append(gap_interval_entry(n, rung_cells, chat_row))

    scaling = {
        "series_label": BOUNDARY_SERIES_LABEL,
        "layer": int(headline),
        "cells": cells,
        "common_rungs": common_rungs,
        "top_common_rung": top_common,
        "gap_interval": gap_entries,
        "splits": {
            "train_pool": int(len(dot["train"])),
            "val": int(len(dot["val"])),
            "test": int(len(dot["test"])),
        },
        "draw_convention": {
            "small_ns": list(SMALL_RUNGS),
            "draws_small": list(SMALL_DRAWS),
            "seed_formula": "np.random.default_rng(19010000 + n*10 + draw)",
            "big_draw": "train_order < n (file-order prefix over the seed-42 shuffle)",
        },
        "lambdas": {
            "n_le_50k": _lambda_grid_params(50),
            "n_gt_50k": _lambda_grid_params(100_000),
            "selection": "val-lambda (primal streaming, N1M.fit_ridge)",
        },
        "knn_pool": f"held-out test targets (pool == true, n={len(dot['test'])})",
        "device_parity_gate": parity,
        "smoke": smoke,
        "metadata": _meta("p2-fits", {"seed": args.seed, "device": args.device}),
        "note": (
            "Generic boundary-token ('.') control: x_sep (anchor-token state) -> next-span "
            "mean ridge at the chat ladder's rung grid/draw convention; WikiText-103-raw-v1 "
            "@ the #931 pin; article-disjoint eval; gap_interval per §3 (conservative "
            "interval difference; chat rungs <= 25k carry banked POINTS only — flagged "
            "chat_ci_source=banked-point-degenerate)."
        ),
    }
    C79.write_json_atomic(eval_out / SCALING_JSON, scaling)

    # Secondary arms: layer companions, "?"/"!", pooled bridge, group diversity.
    secondary: dict = {
        "layer_companions": {},
        "second_types": {},
        "pooled_bridge": None,
        "group_diversity": None,
        "per_type_meta": meta["per_type"],
        "screens": meta["screens"],
        "pooled_meta": meta["pooled"],
        "yield_gate": meta["yield_gate"],
        "smoke": smoke,
    }
    comp_rungs = sorted({n for n in (2500, 25000, top_common) if n and n in common_rungs})
    if smoke:
        comp_rungs = [top_common]  # one tiny companion cell per layer (arm-class coverage)
    del X, Y  # free the headline layer before companion passes
    for li in companions:
        Xl, Yl, row_ids_l, _ = _load_layer_arrays(store, li, persist)
        assert row_ids_l == row_ids, "shard row order drifted between layer passes"
        comp_cells = []
        for n in comp_rungs:
            d = 0 if n in SMALL_RUNGS else "prefix"
            tr = _draw_indices(dot["train"], n, d)
            key = _cell_unit_key("layer", li, n, d, tr, args)
            comp_cells.append(
                _run_cell_checkpointed(
                    units_dir,
                    key,
                    lambda tr=tr, d=d, li=li, Xl=Xl, Yl=Yl: fit_one_cell(
                        Xl,
                        Yl,
                        tr,
                        dot["val"],
                        dot["test"],
                        dev=dev,
                        n_boot=n_boot,
                        n_null=n_null,
                        seed=args.seed,
                        layer=li,
                        draw=d,
                    ),
                )
            )
        secondary["layer_companions"][str(li)] = comp_cells
        del Xl, Yl

    X, Y, row_ids2, _ = _load_layer_arrays(store, headline, persist)
    assert row_ids2 == row_ids, "shard row order drifted on reload"
    for t, slug in (("?", "qm"), ("!", "ex")):
        pool_t = pools[t]
        yield_t = len(pool_t["train"])
        rungs_t = [g for g in BANKED_CHAT_GRID if g <= yield_t][-3:]
        if smoke:
            # Smallest feasible rung; when even the smallest banked size exceeds
            # the smoke yield, one OFF-grid cell at the realized yield keeps the
            # arm class exercised (robustness arm only — the §7.1 grid semantics
            # bind the HEADLINE series, asserted above).
            rungs_t = rungs_t[:1] if rungs_t else ([yield_t] if yield_t >= 8 else [])
        if len(pool_t["test"]) < 2 or len(pool_t["val"]) < 1:
            logger.info("[p2] %s arm skipped: degenerate eval pool", slug)
            rungs_t = []
        t_cells = []
        for n in rungs_t:
            tr = pool_t["train"][:n]
            key = _cell_unit_key(slug, headline, n, "prefix", tr, args)
            t_cells.append(
                _run_cell_checkpointed(
                    units_dir,
                    key,
                    lambda tr=tr, pool_t=pool_t: fit_one_cell(
                        X,
                        Y,
                        tr,
                        pool_t["val"],
                        pool_t["test"],
                        dev=dev,
                        n_boot=n_boot,
                        n_null=n_null,
                        seed=args.seed,
                        layer=headline,
                        draw="prefix",
                    ),
                )
            )
        secondary["second_types"][slug] = {
            "sep_char": t,
            "train_yield": yield_t,
            "eval_pool": {"test": int(len(pool_t["test"])), "val": int(len(pool_t["val"]))},
            "cells": t_cells,
        }
    pooled = pools["pooled"]
    bridge_rungs = [n for n in common_rungs if n <= POOLED_BRIDGE_MAX_RUNG]
    if (
        bridge_rungs
        and len(pooled["test"]) >= 2
        and len(pooled["val"]) >= 1
        and len(pooled["train"]) >= bridge_rungs[-1]
    ):
        n = bridge_rungs[-1]
        tr = pooled["train"][:n]
        key = _cell_unit_key("pooled", headline, n, "prefix", tr, args)
        secondary["pooled_bridge"] = _run_cell_checkpointed(
            units_dir,
            key,
            lambda tr=tr: fit_one_cell(
                X,
                Y,
                tr,
                pooled["val"],
                pooled["test"],
                dev=dev,
                n_boot=n_boot,
                n_null=n_null,
                seed=args.seed,
                layer=headline,
                draw="prefix",
            ),
        )
    # Group-diversity pair: ≤6 anchors/article vs unrestricted at matched n.
    per_art_count: Counter = Counter()
    diverse_positions = []
    for r in dot["train_rows"]:
        if per_art_count[r["article_id"]] < GROUP_DIVERSITY_MAX_PER_ARTICLE:
            per_art_count[r["article_id"]] += 1
            diverse_positions.append(row_pos[r["row_id"]])
    diverse_pool = np.asarray(diverse_positions, dtype=np.int64)
    n_gd = min(GROUP_DIVERSITY_N, len(diverse_pool), len(dot["train"]))
    if smoke:
        n_gd = min(n_gd, 50)
    if n_gd >= 2:
        gd = {}
        for name, pool in (("diverse_le6", diverse_pool), ("unrestricted", dot["train"])):
            tr = pool[:n_gd]
            key = _cell_unit_key(f"gd_{name}", headline, n_gd, "prefix", tr, args)
            gd[name] = _run_cell_checkpointed(
                units_dir,
                key,
                lambda tr=tr: fit_one_cell(
                    X,
                    Y,
                    tr,
                    dot["val"],
                    dot["test"],
                    dev=dev,
                    n_boot=n_boot,
                    n_null=n_null,
                    seed=args.seed,
                    layer=headline,
                    draw="prefix",
                ),
            )
        secondary["group_diversity"] = {
            "n_matched": int(n_gd),
            "max_per_article": GROUP_DIVERSITY_MAX_PER_ARTICLE,
            "cells": gd,
        }
    secondary["metadata"] = _meta("p2-fits-secondary", {"seed": args.seed})
    C79.write_json_atomic(eval_out / SECONDARY_JSON, secondary)
    logger.info("[p2] wrote %s + %s", eval_out / SCALING_JSON, eval_out / SECONDARY_JSON)
    return 0


# ── p3_publish ──────────────────────────────────────────────────────────────────


def phase_p3_publish(args) -> int:
    """Git-push the eval JSONs (pre-teardown harvest), HF-mirror, verify, purge."""
    C79.phase("p3_publish")
    eval_out = Path(args.eval_out)
    paths = [eval_out / SCALING_JSON, eval_out / SECONDARY_JSON]
    for p in paths:
        assert p.exists(), f"missing eval JSON {p} — run p2 first"

    if args.smoke or args.skip_upload:
        logger.info("[p3] smoke/skip-upload: git push + HF mirror SKIPPED")
        return 0

    if (PROJECT_ROOT / ".git").exists():
        env = {**os.environ}
        rels = [str(p.relative_to(PROJECT_ROOT)) for p in paths]
        subprocess.run(["git", "add", "--", *rels], cwd=PROJECT_ROOT, check=True, env=env)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *rels], cwd=PROJECT_ROOT, env=env
        )
        if diff.returncode != 0:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "task #1901: boundary-token control eval JSONs (p3)",
                    "--",
                    *rels,
                ],
                cwd=PROJECT_ROOT,
                check=True,
                env=env,
            )
        subprocess.run(
            ["git", "push", "origin", "HEAD:issue-1901-btokctl"],
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )
        logger.info("[p3] eval JSONs committed + pushed to issue-1901-btokctl")
    elif args.no_git_push:
        logger.warning("[p3] git-less lane: push skipped (--no-git-push); HF mirror is canonical")
    else:
        raise RuntimeError(
            "[p3] no .git checkout (SLURM scratch lane?) — the plan's pre-teardown git push "
            "is impossible here; re-run with --no-git-push to accept the HF-mirror-only path"
        )

    stage = args.out_root / "eval_mirror"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p in paths:
        shutil.copy2(p, stage / p.name)
    url = hub._upload(stage, HF_DATA_REPO, "dataset", path_in_repo=EVAL_PREFIX, raise_on_error=True)
    if not url:
        raise RuntimeError("[p3] eval mirror upload returned empty url")
    shutil.rmtree(stage)

    from huggingface_hub import HfApi

    state_p = args.out_root / "p1_state.json"
    store = args.out_root / "store"
    if state_p.exists():
        shard_files = json.loads(state_p.read_text())["shard_files"]
    else:
        shard_files = sorted(p.name for p in store.glob("pairs_shard*.pt"))
    assert shard_files, "[p3] no shard inventory (p1_state.json missing and store empty)"
    expected = [
        f"{CAPTURE_PREFIX}/{n}" for base in shard_files for n in (base, base[:-3] + ".json")
    ] + [f"{EVAL_PREFIX}/{p.name}" for p in paths]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, expected, path_in_repo=HF_PREFIX
    )
    if missing:
        raise RuntimeError(f"[p3] upload verify FAILED — missing on Hub: {missing[:8]}")
    logger.info("[p3] verified %d files on Hub under %s", len(expected), HF_PREFIX)

    if store.exists():
        shutil.rmtree(store)  # verbatim copies of HF-landed shards (§10 discard slot)
        logger.info("[p3] purged local store %s", store)
    return 0


# ── fig ─────────────────────────────────────────────────────────────────────────

_R2_KEY = "held-out $R^2$"
_ACC_KEY = "retrieval acc@1 (pool 1,000)"


def _normalize_meta_points(meta: dict) -> Counter:
    """§7.4 normalization: (panel value-key, _kind, n_train, value, error) tuples —
    series and _group EXCLUDED (42/304 labeled; _group renumbers on insertion)."""
    out: Counter = Counter()
    for p in meta["points"]:
        if _R2_KEY in p:
            panel, val = _R2_KEY, p[_R2_KEY]
        elif _ACC_KEY in p:
            panel, val = _ACC_KEY, p[_ACC_KEY]
        else:
            raise ValueError(f"meta point with no known panel key: {sorted(p)}")
        err = p.get("error")
        out[
            (
                panel,
                p.get("_kind"),
                round(float(p["training contexts"]), 9),
                round(float(val), 9),
                round(float(err), 9) if err is not None else None,
            )
        ] += 1
    return out


def fig_regression_gate(committed: dict, regenerated: dict, new_label: str) -> None:
    """§7.4: committed tuple multiset == regenerated minus the new-series points."""
    inherited = {"points": [p for p in regenerated["points"] if p.get("series") != new_label]}
    n_new = len(regenerated["points"]) - len(inherited["points"])
    assert n_new > 0, (
        f"no regenerated point carries series={new_label!r} — the renderer extension "
        f"did not label its new series (unit-3 contract)"
    )
    a, b = _normalize_meta_points(committed), _normalize_meta_points(inherited)
    if a != b:
        gone = list((a - b).items())[:5]
        extra = list((b - a).items())[:5]
        raise RuntimeError(
            f"[fig] §7.4 REGRESSION GATE FAILED — inherited point multiset changed. "
            f"missing={gone} unexpected={extra}. Do NOT commit; restore with "
            f"git checkout -- figures/paper/c1_scaling_train_pool.*"
        )
    logger.info(
        "[fig] §7.4 regression gate PASS: %d inherited points unchanged, %d new-series points",
        sum(a.values()),
        n_new,
    )


def phase_fig(args) -> int:
    """Invoke the extended shared renderer + the §7.4 gate + the exploratory dump.

    NOTE (pre-split unit boundary): the renderer-side extension of
    ``issue1901_body_figures.fig_paper_c1_scaling`` (kwargs ``boundary=``,
    ``boundary_label=``) is a SEPARATE deliverable; until it lands this call
    fail-louds with TypeError — deliberate (never stub the renderer here).
    """
    C79.phase("fig")
    import issue1901_body_figures as BF

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    eval_out = Path(args.eval_out)
    btok_p = eval_out / SCALING_JSON
    sec_p = eval_out / SECONDARY_JSON
    assert btok_p.exists(), f"missing {btok_p} — p2/p3 must land the eval JSONs first"
    assert sec_p.exists(), f"missing {sec_p}"
    btok = json.loads(btok_p.read_text())
    secondary = json.loads(sec_p.read_text())

    committed_meta_p = PROJECT_ROOT / "figures/paper/c1_scaling_train_pool.meta.json"
    committed = json.loads(committed_meta_p.read_text())

    set_paper_style("iclr")
    l19, _p18, _boot19 = BF._load()
    ladder = json.loads((BF.PD / "scaling_ladder_L19.json").read_text())
    stem = args.fig_stem
    out_dir = Path(args.fig_out_dir) if args.fig_out_dir else None
    BF.fig_paper_c1_scaling(
        l19,
        ladder,
        boundary=btok,
        boundary_label=BOUNDARY_SERIES_LABEL,
        stem=stem,
        out_dir=out_dir,
    )
    regen_p = (out_dir or (PROJECT_ROOT / "figures" / "paper")) / f"{stem}.meta.json"
    fig_regression_gate(committed, json.loads(regen_p.read_text()), BOUNDARY_SERIES_LABEL)

    # Exploratory dump (blog style, figures/issue_1901/).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style("blog")
    fig_dir = PROJECT_ROOT / "figures" / "issue_1901"
    if args.smoke:
        fig_dir = (out_dir or fig_dir) / "smoke_exploratory"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    comp = secondary.get("layer_companions", {})
    rung_layers: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for c in btok["cells"]:
        rung_layers[int(c["n_train"])].append((int(c["layer"]), float(c["ridge"]["test_r2"])))
    for li, cs in comp.items():
        for c in cs:
            rung_layers[int(c["n_train"])].append((int(li), float(c["ridge"]["test_r2"])))
    for n in sorted(rung_layers):
        pts = sorted(rung_layers[n])
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"n={n:,}")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out $R^2$")
    ax.legend(fontsize=7)
    savefig_paper(fig, "btok_layer_companion", dir=fig_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bars = []
    top_dot = max(btok["cells"], key=lambda c: c["n_train"])
    bars.append((f"'.' (n={top_dot['n_train']:,})", top_dot["ridge"]["test_r2"]))
    for slug, blk in secondary.get("second_types", {}).items():
        if blk["cells"]:
            c = max(blk["cells"], key=lambda c: c["n_train"])
            bars.append((f"'{blk['sep_char']}' (n={c['n_train']:,})", c["ridge"]["test_r2"]))
    if secondary.get("pooled_bridge"):
        c = secondary["pooled_bridge"]
        bars.append((f"pooled {{.!?}} (n={c['n_train']:,})", c["ridge"]["test_r2"]))
    ax.bar(range(len(bars)), [b[1] for b in bars])
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("held-out $R^2$")
    savefig_paper(fig, "btok_type_vs_pooled", dir=fig_dir)
    plt.close(fig)

    gd = secondary.get("group_diversity")
    if gd:
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        names = list(gd["cells"])
        ax.bar(range(len(names)), [gd["cells"][k]["ridge"]["test_r2"] for k in names])
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_ylabel("held-out $R^2$")
        ax.set_title(f"matched n={gd['n_matched']:,}", fontsize=9)
        savefig_paper(fig, "btok_group_diversity", dir=fig_dir)
        plt.close(fig)

    stats = secondary["per_type_meta"]["."]["span_length_stats"]
    if stats.get("hist_counts"):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        edges = stats["hist_edges"]
        ax.bar(range(len(stats["hist_counts"])), stats["hist_counts"], width=0.9)
        ax.set_xticks(
            range(len(stats["hist_counts"])),
            [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(stats["hist_counts"]))],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_xlabel("next-span length (tokens)")
        ax.set_ylabel("pairs")
        savefig_paper(fig, "btok_span_length_hist", dir=fig_dir)
        plt.close(fig)
    logger.info("[fig] exploratory dump written to %s", fig_dir)
    return 0


# ── pod_all + main ──────────────────────────────────────────────────────────────


def phase_pod_all(args) -> int:
    """p1 → p2 → p3 on the one GPU provision (plan §9 launch #2)."""
    for fn in (phase_p1_capture, phase_p2_fits, phase_p3_publish):
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


PHASES = {
    "b0_pairs": phase_b0_pairs,
    "p1_capture": phase_p1_capture,
    "p2_fits": phase_p2_fits,
    "p3_publish": phase_p3_publish,
    "pod_all": phase_pod_all,
    "fig": phase_fig,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=sorted(PHASES), help="phase to run")
    ap.add_argument("--list-phases", action="store_true", help="print the phase registry")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + helper-call bind check, then exit 0",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny-real slice (50 articles / tiny model / n=50 fits); no uploads",
    )
    ap.add_argument("--smoke-articles", type=int, default=50)
    ap.add_argument(
        "--smoke-pairs",
        type=int,
        default=64,
        help="smoke: '.'-train rows captured at p1 (all eval rows + a few ?/! ride along)",
    )
    ap.add_argument(
        "--smoke-eval-spans",
        type=int,
        default=60,
        help="smoke: eval-pool target replacing the 1,400 production margin (#1345)",
    )
    ap.add_argument("--tiny-model-dir", type=str, default=None)
    ap.add_argument(
        "--tiny-layers",
        type=int,
        default=6,
        help="tiny-model depth (smoke persists the M4 remap {0,2,4,5})",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-pairs", type=int, default=2000)
    ap.add_argument("--max-anchors", type=int, default=48)
    ap.add_argument("--article-cap-tokens", type=int, default=4096)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--seed", type=int, default=BATTERY_SEED)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-article-boot", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--p1-wall-budget-h",
        type=float,
        default=7.0,
        help="§7.2 pilot-abort threshold (2x the p1 booking)",
    )
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="skip HF uploads + git pushes (local verification runs)",
    )
    ap.add_argument(
        "--no-git-push",
        action="store_true",
        help="p3: accept the HF-mirror-only path on a git-less lane",
    )
    ap.add_argument("--fig-stem", type=str, default="c1_scaling_train_pool")
    ap.add_argument("--fig-out-dir", type=str, default=None)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("import-check OK")
        return 0
    if args.list_phases:
        print(" ".join(sorted(PHASES)))
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --list-phases / --import-check)")
    if args.smoke:
        # Smoke narrows SCALE only (never launch width / gates' computation).
        args.n_boot = min(args.n_boot, 50)
        args.n_null = min(args.n_null, 20)
        args.n_article_boot = min(args.n_article_boot, 50)
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
    rc = PHASES[args.phase](args)
    if rc == 0:
        C79.phase("done")
    return rc


if __name__ == "__main__":
    try:
        _rc = main()
    except SystemExit as _e:  # designed halts (RC_PILOT_HALT) keep their rc
        _rc = int(_e.code or 0)
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit, not sys.exit: with the driver's import stack loaded, post-
    # Py_Finalize NATIVE teardown (pyarrow/tokenizers threads from the b0
    # streaming+tokenize path) hangs indefinitely at scale — measured >500 s
    # after [phase=done] with a completed, durably-written phase; faulthandler
    # showed no python frames (interpreter already finalized). Every durable is
    # landed in-phase (atomic writes, verified uploads) and no atexit work
    # exists, so skipping finalization is safe — the sanctioned terminal for
    # this driver class (gotchas.md #1739/#2149; in-process release of the
    # streaming pipeline is still done at the b0 site, which fixes the rc=134
    # abort class — this handles the residual native-teardown hang).
    os._exit(_rc)
