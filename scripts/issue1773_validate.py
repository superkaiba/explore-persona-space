#!/usr/bin/env python
"""Issue #1773 Phase 4 — validation harness (plan §4 Phase 4).

All scoring uses HELD-OUT windows (20 activating + 6 non-activating per
feature) never shown to the describer or categorizer. Pinned draw counts
(Statistics must-fix): the 20 held-out activating windows are the SOLE source
pool — detection draws 6 (3/call x 2), fuzzing draws 12 (6 correctly-marked +
6 incorrectly-marked; incorrect marks are RANDOM non-peak tokens in held-out
activating text, NEVER evidence windows); reuse across scorers is sanctioned
(the leakage bar is describe->score). Discrimination: 3 calls x 4-way forced
choice (1 own + 3 top-cosine-neighbour held-out windows; chance 0.25).

Controls: shuffled-label (200 features, descriptions permuted within activity
decile) runs BOTH the detection AND fuzzing batteries; random-direction
controls (feat_id < 0) run describe + detection (REPORTED, not gated).

Stages:  --stage build    build all judge items + truth keys (0 API calls)
         --stage dispatch dispatch built items (spend-guarded; --force-batch)
         --stage score    score results -> scorecard.json + per-battery JSONLs
         --stage validators  non-judge mechanical validators (VM CPU)
         --stage sheet    annotation_sheet_v1.jsonl (>=40 identity positives)
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1773_common as CM  # noqa: E402
import numpy as np  # noqa: E402

# Pinned draw counts (plan §4 Phase 4; the 20-ho-act pool covers 6+12 = 18)
DET_CALLS = 2
DET_ACT_PER_CALL = 3
DET_NONACT_PER_CALL = 3
FUZZ_CALLS = 2
FUZZ_CORRECT_PER_CALL = 3
FUZZ_INCORRECT_PER_CALL = 3
DISC_CALLS = 3
DISC_DISTRACTORS = 3
N_SAMPLE = 1000  # validation sample target (plan §13 band 800-1200)
IDENTITY_CAP = 400
N_SHUFFLED = 200
BOOTSTRAP_B = 1000
VAL_MAX_TOKENS = 400
SHEET_MIN_IDENTITY = 40
SHEET_TOTAL_TARGET = 120  # ~100-140 features

DETECTION_SYSTEM = (
    "You evaluate a description of a sparse-autoencoder feature. Given the "
    "description and numbered text excerpts, decide FOR EACH excerpt whether the "
    "feature (as described) would activate somewhere in it. Reason briefly, then "
    'output ONLY JSON: {"reasoning": "...", "answers": ["yes"|"no", ...]} with '
    "exactly one answer per numbered excerpt, in order."
)
FUZZING_SYSTEM = (
    "You evaluate a description of a sparse-autoencoder feature. Each numbered "
    "excerpt has tokens marked <<like this>>. Decide FOR EACH excerpt whether the "
    "marked tokens are the ones the described feature activates on (correct) or "
    "randomly marked tokens (incorrect). Reason briefly, then output ONLY JSON: "
    '{"reasoning": "...", "answers": ["correct"|"incorrect", ...]} with exactly '
    "one answer per numbered excerpt, in order."
)
DISCRIMINATION_SYSTEM = (
    "You evaluate a description of a sparse-autoencoder feature. Exactly ONE of "
    "the 4 numbered excerpts comes from the described feature; the other 3 come "
    "from similar but different features. Reason briefly, then output ONLY JSON: "
    '{"reasoning": "...", "choice": <1|2|3|4>}.'
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_holdouts(evidence_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted((evidence_dir / "holdout").glob("holdout.shard*.jsonl")):
        for r in CM.iter_jsonl(p):
            out[int(r["feat_id"])] = r
    # random-direction packets carry their holdouts inline
    for p in sorted((evidence_dir / "evidence_manifests").glob("evidence_randdir.shard*.jsonl")):
        for r in CM.iter_jsonl(p):
            out[int(r["feat_id"])] = {
                "feat_id": r["feat_id"],
                "ho_pos": r["ho_pos"],
                "ho_neg": r["ho_neg"],
            }
    return out


def _load_descriptions(out_root: Path) -> dict[int, str]:
    path = out_root / "labels" / "descriptions.jsonl"
    return {int(r["feat_id"]): r["description"] for r in CM.iter_jsonl(path)}


def _load_labels(out_root: Path) -> dict[tuple[int, str], str]:
    path = out_root / "labels" / "axis_labels.jsonl"
    return {(int(r["feat_id"]), r["axis"]): r["label"] for r in CM.iter_jsonl(path)}


def select_sample(
    labels: dict[tuple[int, str], str],
    activity: dict[int, float],
    candidates: list[int],
    n: int = N_SAMPLE,
) -> list[int]:
    """~n features: stratified over activity decile x majority `interpretable`
    label, PLUS all identity-disposition-labeled features (cap 400)."""
    rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 41]))
    ident = [f for f in candidates if labels.get((f, "speaker_property")) == "identity_disposition"]
    if len(ident) > IDENTITY_CAP:
        ident = list(rng.choice(np.asarray(ident), size=IDENTITY_CAP, replace=False))
    rest = [f for f in candidates if f not in set(ident)]
    acts = np.asarray([activity.get(f, 0.0) for f in rest])
    edges = np.quantile(acts, np.linspace(0, 1, 11)[1:-1]) if len(rest) else np.zeros(9)
    dec = np.searchsorted(edges, acts, side="right")
    interp = np.asarray([labels.get((f, "interpretable"), "unresolved") for f in rest])
    picked: list[int] = []
    budget = max(0, n - len(ident))
    strata = [(d, lab) for d in range(10) for lab in ("yes", "no", "unresolved")]
    per = max(1, budget // max(len(strata), 1))
    for d, lab in strata:
        pool = np.asarray(rest)[(dec == d) & (interp == lab)]
        if len(pool):
            picked.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
    return sorted(set(int(x) for x in ident + picked))[: max(n, len(ident))]


# ── item builders (pinned draw counts) ───────────────────────────────────────


def _draw_pools(feat_id: int, ho: dict, rng: np.random.Generator) -> dict | None:
    """Seeded per-feature draws from the holdout pools (pinned counts)."""
    pos, neg = ho.get("ho_pos", []), ho.get("ho_neg", [])
    need_pos = DET_CALLS * DET_ACT_PER_CALL + FUZZ_CALLS * (
        FUZZ_CORRECT_PER_CALL + FUZZ_INCORRECT_PER_CALL
    )
    if len(pos) < need_pos or len(neg) < DET_CALLS * DET_NONACT_PER_CALL:
        return None
    pi = rng.permutation(len(pos))
    det_act = [pos[i] for i in pi[: DET_CALLS * DET_ACT_PER_CALL]]
    fz = pi[DET_CALLS * DET_ACT_PER_CALL : need_pos]
    half = FUZZ_CALLS * FUZZ_CORRECT_PER_CALL
    fuzz_correct = [pos[i] for i in fz[:half]]
    fuzz_incorrect = [pos[i] for i in fz[half:]]
    ni = rng.permutation(len(neg))
    det_non = [neg[i] for i in ni[: DET_CALLS * DET_NONACT_PER_CALL]]
    return {
        "det_act": det_act,
        "det_non": det_non,
        "fuzz_correct": fuzz_correct,
        "fuzz_incorrect": fuzz_incorrect,
    }


def _random_mark(text_plain: str, rng: np.random.Generator) -> str:
    """Incorrectly-marked window: mark a random whitespace-token in the PLAIN
    text (held-out activating text; never an evidence window)."""
    words = text_plain.split(" ")
    if not words:
        return f"<<{text_plain}>>"
    k = int(rng.integers(0, len(words)))
    words[k] = f"<<{words[k]}>>"
    return " ".join(words)


def build_items(args) -> int:
    """Build detection/fuzzing/discrimination + control items with truth keys."""
    ev = args.evidence_dir
    holdouts = _load_holdouts(ev)
    descriptions = _load_descriptions(args.out_root)
    labels = _load_labels(args.out_root)
    p0 = np.load(args.phase0_dir / "phase0_arrays.npz", allow_pickle=True)
    fid = np.asarray(p0["feat_ids"], dtype=np.int64)
    activity = {int(f): float(a) for f, a in zip(fid, p0["activity"], strict=True)}
    nb_idx = np.asarray(p0["neighbor_idx"], dtype=np.int64)
    fid_pos = {int(f): i for i, f in enumerate(fid)}

    candidates = sorted(f for f in holdouts if f >= 0 and f in descriptions)
    sample = select_sample(labels, activity, candidates, n=args.sample_n)
    rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 43]))
    items: list[dict] = []
    for feat_id in sample:
        pools = _draw_pools(feat_id, holdouts[feat_id], rng)
        if pools is None:
            continue
        desc = descriptions[feat_id]
        items.extend(_battery_items("real", feat_id, desc, pools, rng))
        # discrimination: 1 own + 3 neighbour held-out windows per call
        own = [w for w in holdouts[feat_id]["ho_pos"]]
        nbs = [int(fid[j]) for j in nb_idx[fid_pos[feat_id]][:DISC_DISTRACTORS]]
        nb_pools = [holdouts.get(nf, {}).get("ho_pos", []) for nf in nbs]
        if all(len(pl) >= DISC_CALLS for pl in nb_pools) and len(own) >= DISC_CALLS:
            for c in range(DISC_CALLS):
                wins = [own[c]] + [pl[c] for pl in nb_pools]
                order = rng.permutation(4)
                truth = int(np.where(order == 0)[0][0]) + 1
                items.append(
                    {
                        "custom_id": f"f{feat_id}-disc{c}",
                        "battery": "discrimination",
                        "arm": "real",
                        "feat_id": feat_id,
                        "description": desc,
                        "windows": [wins[i]["text_plain"] for i in order],
                        "truth_choice": truth,
                    }
                )
    # shuffled-label control: permute descriptions within activity decile
    sh = [
        int(x)
        for x in rng.choice(np.asarray(sample), size=min(N_SHUFFLED, len(sample)), replace=False)
    ]
    acts = np.asarray([activity.get(f, 0.0) for f in sh])
    edges = np.quantile(acts, np.linspace(0, 1, 11)[1:-1]) if len(sh) else np.zeros(9)
    dec = np.searchsorted(edges, acts, side="right")
    shuffled_desc: dict[int, str] = {}
    for d in range(10):
        grp = [f for f, dd in zip(sh, dec, strict=True) if dd == d]
        if len(grp) >= 2:
            perm = rng.permutation(len(grp))
            while np.any(perm == np.arange(len(grp))):  # derangement-ish retry
                perm = rng.permutation(len(grp))
            for i, f in enumerate(grp):
                shuffled_desc[f] = descriptions[grp[perm[i]]]
    for feat_id, desc in sorted(shuffled_desc.items()):
        pools = _draw_pools(feat_id, holdouts[feat_id], rng)
        if pools is not None:
            items.extend(_battery_items("shuffled", feat_id, desc, pools, rng))
    # random-init control: detection battery on randdir features (feat_id < 0)
    for feat_id in sorted(f for f in holdouts if f < 0):
        desc = descriptions.get(feat_id)
        if desc is None:
            continue
        pools = _draw_pools(feat_id, holdouts[feat_id], rng)
        if pools is not None:
            items.extend(
                it
                for it in _battery_items("randinit", feat_id, desc, pools, rng)
                if it["battery"] == "detection"
            )
    out_dir = args.out_root / "validation"
    CM.write_jsonl_sharded(items, out_dir, "val_items")
    meta = {
        **CM.repro_meta(),
        "n_sample": len(sample),
        "n_items": len(items),
        "n_shuffled": len(shuffled_desc),
        "pinned_draws": {
            "det_act": DET_CALLS * DET_ACT_PER_CALL,
            "det_nonact": DET_CALLS * DET_NONACT_PER_CALL,
            "fuzz_correct": FUZZ_CALLS * FUZZ_CORRECT_PER_CALL,
            "fuzz_incorrect": FUZZ_CALLS * FUZZ_INCORRECT_PER_CALL,
            "holdout_act_pool": CM.N_ACT_HOLDOUT,
        },
    }
    (out_dir / "val_items_meta.json").write_text(json.dumps(meta, indent=1))
    _log(f"[validate-build] {len(items)} items over {len(sample)} sampled features")
    return 0


def _battery_items(
    arm: str, feat_id: int, desc: str, pools: dict, rng: np.random.Generator
) -> list[dict]:
    """Detection + fuzzing calls for one (arm, feature)."""
    pfx = {"real": "f", "shuffled": "s", "randinit": "r"}[arm]
    fkey = str(feat_id).replace("-", "n")  # randdir ids are negative
    out: list[dict] = []
    for c in range(DET_CALLS):
        act = pools["det_act"][c * DET_ACT_PER_CALL : (c + 1) * DET_ACT_PER_CALL]
        non = pools["det_non"][c * DET_NONACT_PER_CALL : (c + 1) * DET_NONACT_PER_CALL]
        wins = [(w["text_plain"], "yes") for w in act] + [(w["text_plain"], "no") for w in non]
        order = rng.permutation(len(wins))
        out.append(
            {
                "custom_id": f"{pfx}{fkey}-det{c}",
                "battery": "detection",
                "arm": arm,
                "feat_id": feat_id,
                "description": desc,
                "windows": [wins[i][0] for i in order],
                "truth": [wins[i][1] for i in order],
            }
        )
    for c in range(FUZZ_CALLS):
        cor = pools["fuzz_correct"][c * FUZZ_CORRECT_PER_CALL : (c + 1) * FUZZ_CORRECT_PER_CALL]
        inc = pools["fuzz_incorrect"][
            c * FUZZ_INCORRECT_PER_CALL : (c + 1) * FUZZ_INCORRECT_PER_CALL
        ]
        wins = [(w["text_marked"], "correct") for w in cor] + [
            (_random_mark(w["text_plain"], rng), "incorrect") for w in inc
        ]
        order = rng.permutation(len(wins))
        out.append(
            {
                "custom_id": f"{pfx}{fkey}-fuz{c}",
                "battery": "fuzzing",
                "arm": arm,
                "feat_id": feat_id,
                "description": desc,
                "windows": [wins[i][0] for i in order],
                "truth": [wins[i][1] for i in order],
            }
        )
    return out


def _render_item(item: dict) -> str:
    lines = [f"### Feature description\n{item['description']}", "### Excerpts"]
    for i, w in enumerate(item["windows"], 1):
        lines.append(f"{i}. {w}")
    if item["battery"] == "detection":
        lines.append("For EACH excerpt: would the described feature activate? Output the JSON.")
    elif item["battery"] == "fuzzing":
        lines.append("For EACH excerpt: are the <<marked>> tokens correct? Output the JSON.")
    else:
        lines.append("Which ONE excerpt comes from the described feature? Output the JSON.")
    return "\n".join(lines)


def dispatch_items(args) -> int:
    """Dispatch built validation items through dispatch_judge_items."""
    from explore_persona_space.eval.judge_dispatch import (
        dispatch_judge_items,
        graded_temperature,
        keep_raw_judge_text,
    )

    out_dir = args.out_root / "validation"
    items = []
    for p in sorted(out_dir.glob("val_items.shard*.jsonl")):
        items.extend(CM.iter_jsonl(p))
    if not args.full:
        items = items[: args.limit]
    by_sys = defaultdict(list)
    for it in items:
        by_sys[it["battery"]].append(it)
    systems = {
        "detection": DETECTION_SYSTEM,
        "fuzzing": FUZZING_SYSTEM,
        "discrimination": DISCRIMINATION_SYSTEM,
    }
    for battery, group in sorted(by_sys.items()):
        jitems = [(it["custom_id"], f"val:{battery}", "", _render_item(it)) for it in group]
        _log(f"[validate-dispatch] {battery}: {len(jitems)} items")
        kwargs = dict(
            judge_system_prompt=systems[battery],
            max_tokens=VAL_MAX_TOKENS,
            checkpoint_dir=args.work / "judge_checkpoints" / f"val_{battery}",
            dry_run=args.dry_run,
        )
        if args.force_batch:
            kwargs["threshold_base"] = 1
        with graded_temperature(CM.JUDGE_TEMPERATURE), keep_raw_judge_text():
            results = dispatch_judge_items(jitems, **kwargs)
        if not args.dry_run:
            path = out_dir / f"val_results_{battery}.json"
            tmp = path.parent / f".tmp_{path.name}"
            tmp.write_text(json.dumps(results))
            tmp.replace(path)
    return 0


# ── scoring ──────────────────────────────────────────────────────────────────


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Feature-level percentile bootstrap CI of the mean (vectorized draws)."""
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_B, len(vals)))
    means = vals[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def score_results(args) -> int:
    """Score judge returns -> per-feature fractions + arm-level balanced
    accuracies + scorecard.json (per-axis rows; kappa joined from Phase 3)."""
    out_dir = args.out_root / "validation"
    items = []
    for p in sorted(out_dir.glob("val_items.shard*.jsonl")):
        items.extend(CM.iter_jsonl(p))
    results: dict[str, dict] = {}
    if args.mock_results:
        results = json.loads(Path(args.mock_results).read_text())
    else:
        for battery in ("detection", "fuzzing", "discrimination"):
            p = out_dir / f"val_results_{battery}.json"
            if p.exists():
                results.update(json.loads(p.read_text()))
    per_feat: dict[tuple[str, str, int], dict] = {}
    drops = Counter()
    for it in items:
        res = results.get(it["custom_id"])
        key = (it["arm"], it["battery"], int(it["feat_id"]))
        rec = per_feat.setdefault(key, {"tp": 0, "pos": 0, "tn": 0, "neg": 0, "hits": 0, "n": 0})
        if not isinstance(res, dict) or res.get("error"):
            drops[f"{it['battery']}_{'transport' if _is_transport(res) else 'content'}"] += 1
            continue
        if it["battery"] == "discrimination":
            choice = res.get("choice")
            if not isinstance(choice, int | float) or not 1 <= int(choice) <= 4:
                drops["discrimination_content"] += 1
                continue
            rec["n"] += 1
            rec["hits"] += int(int(choice) == it["truth_choice"])
        else:
            ans = res.get("answers")
            if not isinstance(ans, list) or len(ans) != len(it["truth"]):
                drops[f"{it['battery']}_content"] += 1
                continue
            pos_lab = "yes" if it["battery"] == "detection" else "correct"
            neg_lab = "no" if it["battery"] == "detection" else "incorrect"
            for a, t in zip(ans, it["truth"], strict=True):
                a_norm = str(a).strip().lower()
                if a_norm not in (pos_lab, neg_lab):
                    drops[f"{it['battery']}_content"] += 1
                    continue
                if t == pos_lab:
                    rec["pos"] += 1
                    rec["tp"] += int(a_norm == pos_lab)
                else:
                    rec["neg"] += 1
                    rec["tn"] += int(a_norm == neg_lab)

    def _ba(rec: dict) -> float:
        if rec["pos"] == 0 or rec["neg"] == 0:
            return float("nan")
        return 0.5 * (rec["tp"] / rec["pos"] + rec["tn"] / rec["neg"])

    rows = []
    for (arm, battery, feat_id), rec in sorted(per_feat.items()):
        score = (rec["hits"] / rec["n"]) if battery == "discrimination" and rec["n"] else _ba(rec)
        rows.append({"arm": arm, "battery": battery, "feat_id": feat_id, "score": score, **rec})
    det_rows = [r for r in rows if r["battery"] != "discrimination"]
    disc_rows = [r for r in rows if r["battery"] == "discrimination"]
    with (out_dir / "detection_fuzzing.jsonl").open("w", encoding="utf-8") as fh:
        for r in det_rows:
            fh.write(json.dumps(r) + "\n")
    with (out_dir / "discrimination.jsonl").open("w", encoding="utf-8") as fh:
        for r in disc_rows:
            fh.write(json.dumps(r) + "\n")

    rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 47]))
    agg: dict[str, dict] = {}
    for arm in ("real", "shuffled", "randinit"):
        for battery in ("detection", "fuzzing", "discrimination"):
            vals = np.asarray(
                [
                    r["score"]
                    for r in rows
                    if r["arm"] == arm and r["battery"] == battery and not np.isnan(r["score"])
                ]
            )
            if len(vals):
                agg[f"{arm}_{battery}"] = {
                    "mean": float(vals.mean()),
                    "n_features": int(len(vals)),
                    "ci95": _boot_ci(vals, rng),
                }
    kappa_path = args.out_root / "labels" / "kappa_report.json"
    kappa = json.loads(kappa_path.read_text())["axes"] if kappa_path.exists() else {}
    scorecard = {"axes": {}, "aggregates": agg, "drops": dict(drops), **CM.repro_meta()}
    for axis in CM.AXES:
        scorecard["axes"][axis] = {
            "detection": agg.get("real_detection", {}).get("mean", float("nan")),
            "fuzzing": agg.get("real_fuzzing", {}).get("mean", float("nan")),
            "discrimination": agg.get("real_discrimination", {}).get("mean", float("nan")),
            "kappa": (kappa.get(axis) or {}).get("kappa", float("nan")),
            "shuffled_detection": agg.get("shuffled_detection", {}).get("mean", float("nan")),
            "shuffled_fuzzing": agg.get("shuffled_fuzzing", {}).get("mean", float("nan")),
            "randinit_detection": agg.get("randinit_detection", {}).get("mean", float("nan")),
        }
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=1))
    controls = {
        "shuffled": {k: v for k, v in agg.items() if k.startswith("shuffled_")},
        "randinit": {k: v for k, v in agg.items() if k.startswith("randinit_")},
        "note": "shuffled-label runs detection AND fuzzing; random-init REPORTED not gated",
        **CM.repro_meta(),
    }
    (out_dir / "controls.json").write_text(json.dumps(controls, indent=1))
    _log(f"[validate-score] aggregates: { {k: round(v['mean'], 3) for k, v in agg.items()} }")
    return 0


def _is_transport(res: object) -> bool:
    from explore_persona_space.eval.batch_judge import is_transport_error_dict

    return isinstance(res, dict) and is_transport_error_dict(res)


# ── non-judge mechanical validators (each axis blind to its validator) ──────


_SCRIPT_RANGES = (
    "LATIN",
    "CYRILLIC",
    "CJK",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "ARABIC",
    "DEVANAGARI",
    "GREEK",
    "THAI",
    "HEBREW",
)


def _script_hist(text: str) -> str | None:
    counts: Counter[str] = Counter()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for s in _SCRIPT_RANGES:
                if s in name:
                    counts[s] += 1
                    break
    return counts.most_common(1)[0][0] if counts else None


def _monolinguality(texts: list[str]) -> dict:
    """langdetect language share over activating windows (Assumption 12: 50-window
    known-language smoke; unicode-script histogram as the cheap cross-check)."""
    langs: Counter[str] = Counter()
    method = "langdetect"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        for t in texts:
            try:
                langs[detect(t)] += 1
            except Exception:  # noqa: BLE001 — langdetect raises on degenerate text
                langs["__undetected__"] += 1
    except ImportError:
        method = "script_histogram"
        for t in texts:
            s = _script_hist(t)
            langs[s or "__undetected__"] += 1
    scripts = Counter(_script_hist(t) or "__none__" for t in texts)
    top, top_n = (langs.most_common(1) or [("__none__", 0)])[0]
    return {
        "method": method,
        "top_lang": top,
        "share": top_n / max(sum(langs.values()), 1),
        "script_top": (scripts.most_common(1) or [("__none__", 0)])[0][0],
    }


_INFORMAL = ("n't", "'re", "'ll", "'ve", "gonna", "wanna", "lol", "!", "!!", "?!")


def _informality(texts: list[str]) -> float:
    hits = sum(sum(t.lower().count(m) for m in _INFORMAL) for t in texts)
    words = sum(len(t.split()) for t in texts) or 1
    emoji = sum(1 for t in texts for ch in t if ord(ch) >= 0x1F300)
    return (hits + emoji) / words


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC of x separating y==1 from y==0."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_validators(args) -> int:
    """Per-axis non-judge validators, each on a quantity its judge never saw
    (blinding rule 2): language->monolinguality; register->informality;
    functional_role->side_ratio; abstraction->covariate AUC;
    identity_disposition->rb_align percentile (PROXY); interpretable->detection
    separation."""
    labels = _load_labels(args.out_root)
    p0 = np.load(args.phase0_dir / "phase0_arrays.npz", allow_pickle=True)
    fid = np.asarray(p0["feat_ids"], dtype=np.int64)
    pos = {int(f): i for i, f in enumerate(fid)}
    packets = {}
    for p in sorted((args.evidence_dir / "evidence_manifests").glob("evidence.shard*.jsonl")):
        for r in CM.iter_jsonl(p):
            packets[int(r["feat_id"])] = r
    feats = sorted({f for (f, _a) in labels if f in packets and f in pos})
    out: dict[str, object] = {}
    # language / register / identity / functional_role / abstraction reads
    mono_rows, informal_rows = [], []
    for f in feats:
        texts = [w["text_plain"] for w in packets[f]["ex_pos"]]
        lab_sp = labels.get((f, "speaker_property"))
        if lab_sp == "language":
            mono_rows.append({"feat_id": f, **_monolinguality(texts)})
        if lab_sp == "register_style":
            informal_rows.append({"feat_id": f, "informality": _informality(texts)})
    out["language_monolinguality"] = {
        "n": len(mono_rows),
        "mean_share": float(np.mean([r["share"] for r in mono_rows])) if mono_rows else None,
        "rows": mono_rows[:200],
    }
    baseline_inf = [
        _informality([w["text_plain"] for w in packets[f]["ex_pos"]])
        for f in feats[: max(len(feats) // 5, 1)]
    ]
    out["register_informality"] = {
        "n": len(informal_rows),
        "mean": float(np.mean([r["informality"] for r in informal_rows]))
        if informal_rows
        else None,
        "baseline_mean": float(np.mean(baseline_inf)) if baseline_inf else None,
        "note": "lexical proxy; steering-transfer validator DEFERRED (plan §11 deviation)",
    }
    side = np.asarray(p0["side_ratio"], dtype=np.float64)
    fr = {
        lab: [float(side[pos[f]]) for f in feats if labels.get((f, "functional_role")) == lab]
        for lab in CM.AXES["functional_role"]
    }
    out["functional_role_side_ratio"] = {
        lab: {"n": len(v), "mean": float(np.mean(v)) if v else None} for lab, v in fr.items()
    }
    # abstraction: AUC of covariates predicting token_surface vs abstract_contextual
    act = np.asarray(p0["activity"], dtype=np.float64)
    persist = np.asarray(p0["persist_answer"], dtype=np.float64)
    conc_by_f: dict[int, float] = {}
    table = args.phase0_dir / "feature_table.jsonl"
    if table.exists():
        for r in CM.iter_jsonl(table):
            conc_by_f[int(r["feat_id"])] = float(r["logit_footprint"]["concentration"])
    ab_f = [
        f
        for f in feats
        if labels.get((f, "abstraction")) in ("token_surface", "abstract_contextual")
    ]
    yv = np.asarray(
        [1 if labels.get((f, "abstraction")) == "abstract_contextual" else 0 for f in ab_f]
    )
    out["abstraction_auc"] = {
        "n": len(ab_f),
        "auc_activity": _auc(np.asarray([act[pos[f]] for f in ab_f]), yv) if len(ab_f) else None,
        "auc_persist": _auc(np.asarray([persist[pos[f]] for f in ab_f]), yv) if len(ab_f) else None,
        "auc_concentration": _auc(np.asarray([conc_by_f.get(f, np.nan) for f in ab_f]), yv)
        if len(ab_f)
        else None,
    }
    rb = np.asarray(p0["rb_cos"], dtype=np.float64)  # (n_traits, n_feat)
    rb_max = rb.max(axis=0)
    pct = np.argsort(np.argsort(rb_max)) / max(len(rb_max) - 1, 1)
    ident = [f for f in feats if labels.get((f, "speaker_property")) == "identity_disposition"]
    out["identity_rb_align_percentile_PROXY"] = {
        "n_identity": len(ident),
        "mean_percentile_identity": float(np.mean([pct[pos[f]] for f in ident])) if ident else None,
        "mean_percentile_all": float(pct.mean()),
        "note": "PROXY read — headline identity-disposition precision is human-annotation-gated",
    }
    det_path = args.out_root / "validation" / "detection_fuzzing.jsonl"
    if det_path.exists():
        det = [
            r for r in CM.iter_jsonl(det_path) if r["battery"] == "detection" and r["arm"] == "real"
        ]
        by_lab = defaultdict(list)
        for r in det:
            lab = labels.get((int(r["feat_id"]), "interpretable"))
            if lab and not (isinstance(r["score"], float) and np.isnan(r["score"])):
                by_lab[lab].append(r["score"])
        out["interpretable_detection_separation"] = {
            lab: {"n": len(v), "mean": float(np.mean(v))} for lab, v in by_lab.items()
        }
    # identity-disposition precision vs annotated reference when available
    ann = args.out_root / "validation" / "human_annotations.jsonl"
    if ann.exists():
        gold = {(int(r["feat_id"]), r["axis"]): r["label"] for r in CM.iter_jsonl(ann)}
        tp = sum(1 for f in ident if gold.get((f, "speaker_property")) == "identity_disposition")
        got = [f for f in ident if (f, "speaker_property") in gold]
        base = [f for (f, a) in gold if a == "speaker_property"]
        base_pos = sum(
            1 for f in base if gold.get((f, "speaker_property")) == "identity_disposition"
        )
        out["identity_precision_vs_human"] = {
            "n_annotated_identity_judged": len(got),
            "precision": tp / len(got) if got else None,
            "base_rate": base_pos / len(base) if base else None,
        }
    out_path = args.out_root / "validation" / "mechanical_validators.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**out, **CM.repro_meta()}, indent=1))
    _log(f"[validators] done -> {out_path}")
    return 0


# ── annotation sheet (Statistics Must-Fix; committed test asserts >=40) ─────


def build_annotation_sheet(args) -> tuple[int, dict]:
    """artifacts/annotation_sheet_v1.jsonl: >=40 judge-labeled identity-
    disposition POSITIVES stratified over rb_align percentile + matched
    negatives + stratified remainder (~100-140 features x 5 axes). Judge labels
    are WITHHELD from the sheet (answer key written separately)."""
    labels = _load_labels(args.out_root)
    descriptions = _load_descriptions(args.out_root)
    p0 = np.load(args.phase0_dir / "phase0_arrays.npz", allow_pickle=True)
    fid = np.asarray(p0["feat_ids"], dtype=np.int64)
    pos = {int(f): i for i, f in enumerate(fid)}
    rb_max = np.asarray(p0["rb_cos"], dtype=np.float64).max(axis=0)
    act = np.asarray(p0["activity"], dtype=np.float64)
    packets = {}
    for p in sorted((args.evidence_dir / "evidence_manifests").glob("evidence.shard*.jsonl")):
        for r in CM.iter_jsonl(p):
            packets[int(r["feat_id"])] = r
    feats = sorted({f for (f, _a) in labels if f in packets and f in pos})
    ident = [f for f in feats if labels.get((f, "speaker_property")) == "identity_disposition"]
    rng = np.random.default_rng(np.random.SeedSequence([CM.SEED, 53]))
    # stratify identity positives over rb_align percentile (never easy-case-only)
    if len(ident) > SHEET_MIN_IDENTITY:
        r_pct = np.argsort(np.argsort(rb_max)) / max(len(rb_max) - 1, 1)
        vals = np.asarray([r_pct[pos[f]] for f in ident])
        qs = np.quantile(vals, np.linspace(0, 1, 5)[1:-1])
        strata = np.searchsorted(qs, vals, side="right")
        take: list[int] = []
        per = int(np.ceil(SHEET_MIN_IDENTITY / 4))
        for s in range(4):
            pool = np.asarray(ident)[strata == s]
            if len(pool):
                take.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
        ident_take = sorted(int(x) for x in take)[: max(SHEET_MIN_IDENTITY, len(take))]
    else:
        ident_take = ident  # shortfall REPORTED, never backfilled
    # matched negatives: same count, activity-decile-matched, non-identity
    non_ident = [f for f in feats if f not in set(ident)]
    neg_take: list[int] = []
    if non_ident:
        na = np.asarray([act[pos[f]] for f in non_ident])
        for f in ident_take:
            j = int(np.argmin(np.abs(na - act[pos[f]])))
            neg_take.append(int(non_ident[j]))
            na[j] = np.inf
    used = set(ident_take) | set(neg_take)
    rest_pool = [f for f in feats if f not in used]
    n_rest = max(0, SHEET_TOTAL_TARGET - len(used))
    rest = (
        rng.choice(np.asarray(rest_pool), size=min(n_rest, len(rest_pool)), replace=False)
        .astype(int)
        .tolist()
        if rest_pool
        else []
    )
    sheet_feats = sorted(set(ident_take) | set(neg_take) | set(rest))
    rows, key_rows = [], []
    for f in sheet_feats:
        pk = packets[f]
        ev = [w["text_marked"] for w in pk["ex_pos"][:5]]
        for axis in CM.AXES:
            rows.append(
                {
                    "feat_id": f,
                    "axis": axis,
                    "allowed_labels": list(CM.AXES[axis]),
                    "definitions": CM.AXIS_DEFINITIONS[axis],
                    "description": descriptions.get(f),
                    "examples_marked": ev,
                    "human_label": None,
                }
            )
            key_rows.append({"feat_id": f, "axis": axis, "judge_label": labels.get((f, axis))})
    art_dir = args.artifacts_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    sheet = art_dir / "annotation_sheet_v1.jsonl"
    with sheet.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (art_dir / "annotation_sheet_v1.answer_key.jsonl").open("w", encoding="utf-8") as fh:
        for r in key_rows:
            fh.write(json.dumps(r) + "\n")
    n_ident_rows = sum(
        1
        for r in key_rows
        if r["axis"] == "speaker_property" and r["judge_label"] == "identity_disposition"
    )
    meta = {
        **CM.repro_meta(),
        "n_features": len(sheet_feats),
        "n_rows": len(rows),
        "n_identity_positive_rows": n_ident_rows,
        "identity_shortfall": bool(n_ident_rows < SHEET_MIN_IDENTITY),
    }
    (art_dir / "annotation_sheet_v1.meta.json").write_text(json.dumps(meta, indent=1))
    if meta["identity_shortfall"]:
        _log(
            f"[sheet] SHORTFALL: only {n_ident_rows} identity positives exist "
            f"(< {SHEET_MIN_IDENTITY}) — reported, never backfilled"
        )
    _log(f"[sheet] {len(rows)} rows over {len(sheet_feats)} features -> {sheet}")
    return 0, meta


def _import_check() -> int:
    """Axis-1 import-resolution leg (preferred shape (a))."""
    import langdetect  # noqa: F401

    from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: F401
    from explore_persona_space.eval.judge_dispatch import (  # noqa: F401
        dispatch_judge_items,
        graded_temperature,
        keep_raw_judge_text,
    )

    print("[import-check] OK: all deferred imports resolve", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("build", "dispatch", "score", "validators", "sheet"))
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--evidence-dir", type=Path, default=CM.WORK_DEFAULT / "evidence")
    ap.add_argument("--phase0-dir", type=Path, default=CM.OUT_EVAL / "phase0")
    ap.add_argument("--out-root", type=Path, default=CM.OUT_EVAL)
    ap.add_argument("--work", type=Path, default=CM.WORK_DEFAULT)
    ap.add_argument("--artifacts-dir", type=Path, default=CM.OUT_EVAL / "artifacts")
    ap.add_argument("--sample-n", type=int, default=N_SAMPLE)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--force-batch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock-results", type=Path, default=None, help="score smoke: results JSON")
    args = ap.parse_args()

    if args.import_check:
        sys.exit(_import_check())
    if args.stage == "build":
        rc = build_items(args)
    elif args.stage == "dispatch":
        rc = dispatch_items(args)
    elif args.stage == "score":
        rc = score_results(args)
    elif args.stage == "validators":
        rc = run_validators(args)
    elif args.stage == "sheet":
        rc, _ = build_annotation_sheet(args)
    else:
        ap.error("--stage required (or --import-check)")
        return 2
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
