#!/usr/bin/env python3
"""#1739 Result 3.5c: degeneration (repetition-loop collapse) vs context->answer map quality.

Two questions, ANALYSIS-ONLY (0 GPU-h, no generation, no model/API calls, no new fits):

1. Can degeneration be measured per rollout row, and do independent measures agree?
   Four measures per rollout completion (text is scored PROGRAMMATICALLY; completion
   text is never printed, logged, or persisted by this script — outputs carry scores,
   counts, IDs and rates only; #1739 is trigger-dense):
     (a) autopsy-continuity chunk metric: occurrence count of the most-common
         60-character window within the last 3,000 characters (>=5 flagged degenerate,
         matching the prior tom-gibbs autopsy convention);
     (b) distinct word-4-gram ratio (distinct / total; lower = more repetitive);
     (c) zlib compression ratio (compressed bytes / raw bytes; lower = more repetitive);
     (d) finish_reason == "length" (cap-hit flag).
2. Is the map specifically worse at predicting behavior on degenerate rows, or does the
   real-answer oracle arm fall with it? Contexts are stratified by degeneration
   (clean = no flagged rollout vs degenerate = >=1 flagged rollout among the context's
   judged rollouts); within each stratum we report Spearman rho of pred(mapped answer)
   [arm6], pred(real answer) [arm11, oracle] and pred(context) [arm1] against the judge
   DV with Bonett-Wright CIs, per primary cell of the r35b grain (21 cells, context_end,
   train eval rung). Discriminator: if rho(mapped) AND rho(real) both degrade on the
   degenerate stratum, degeneration corrupts the measurement generally; if only
   rho(mapped) degrades, the map specifically fails there.

Inputs:
  - HF dataset repo superkaiba1/explore-persona-space-data,
    issue1739_ctxmap/raw_completions/labeling_{evil,hallucination,sycophancy}.shard*.jsonl
    (packed shards: one {"src": ..., "doc": ...} line per source file; manifest lines
    have src ending _manifest.json). Staged to data/issue_1739/hf_dl/r35c_labeling/.
  - Per-cell prediction sidecars already staged by issue1739_stage_percell_preds.py
    (eval_results/issue_1739/<behavior>/arm_results/percell/), read through the loaders
    of scripts/issue1739_r35b_perprompt.py (reused by import).

Outputs (eval_results/issue_1739/result3_5c_degeneration/):
  rollout_metrics/<shard>.jsonl  per-rollout metric rows (NO completion text)
  r35c_manifests.json            shard manifest docs (verification anchors)
  r35c_base_rates.json           per-behavior x rung base rates, metric agreement
  r35c_context_grain_<b>.jsonl   per-context degeneration aggregates (train rung)
  r35c_cell_strata.json          per-primary-cell stratified rho profiles + bootstrap
  r35c_summary.json              headline numbers
Figures (--phase figures) under figures/issue_1739/result3_5c_degeneration/.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1739_r35b_perprompt as r35b  # noqa: E402  (reuse loaders + stat helpers)

DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1739_ctxmap/raw_completions"
STAGE_DIR = REPO_ROOT / "data" / "issue_1739" / "hf_dl" / "r35c_labeling"
OUT_ROOT = REPO_ROOT / "eval_results" / "issue_1739" / "result3_5c_degeneration"
METRICS_DIR = OUT_ROOT / "rollout_metrics"
FIG_ROOT = REPO_ROOT / "figures" / "issue_1739" / "result3_5c_degeneration"

BEHAVIORS = r35b.BEHAVIORS
CHUNK_LEN = 60  # autopsy convention: 60-char window
TAIL_LEN = 3000  # within the last 3,000 characters
CHUNK_FLAG_MIN = 5  # >=5 occurrences flagged degenerate (autopsy convention)
SHARD_RE = re.compile(r"labeling_(evil|hallucination|sycophancy)\.shard\d+\.jsonl$")


# ---------------------------------------------------------------- phase: stage


def stage() -> list[Path]:
    """Stage the labeling shards (top-level files only, one resolved revision)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = retry_transient(
        lambda: api.repo_info(DATA_REPO, repo_type="dataset"),
        what=f"repo_info({DATA_REPO})",
    )
    revision = info.sha
    tree = retry_transient(
        lambda: list(
            # Deliberately recursive=False + revision-pinned: the scoped helper
            # list_hf_files_under_path walks recursively, and this prefix
            # (issue1739_ctxmap/raw_completions) holds large sibling subdirectories
            # (evil_ood_spread_full/...) a recursive walk would needlessly enumerate.
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in retry_transient by the enclosing call
            api.list_repo_tree(
                DATA_REPO,
                path_in_repo=HF_PREFIX,
                repo_type="dataset",
                recursive=False,
                revision=revision,
            )
        ),
        what=f"list_repo_tree({HF_PREFIX})",
    )
    wanted = [e.path for e in tree if SHARD_RE.search(e.path)]
    if not wanted:
        raise FileNotFoundError(f"no labeling shards under {HF_PREFIX}")
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def _one(path_in_repo: str) -> Path:
        return stage_hub_file(
            DATA_REPO,
            path_in_repo,
            STAGE_DIR / Path(path_in_repo).name,
            repo_type="dataset",
            revision=revision,
        )

    with ThreadPoolExecutor(max_workers=6) as ex:
        staged = list(ex.map(_one, sorted(wanted)))
    total = sum(p.stat().st_size for p in staged)
    print(
        f"[stage] {len(staged)} shards, {total / 2**20:.0f} MiB, "
        f"revision {revision[:12]}, elapsed={time.time() - t0:.0f}s"
    )
    return staged


# ---------------------------------------------------------------- phase: metrics


def rollout_metrics(text: str) -> tuple[int, int, int, float, float]:
    """Per-rollout degeneration measures; operates on the string, never emits it."""
    n_chars = len(text)
    tail = text[-TAIL_LEN:]
    if len(tail) >= CHUNK_LEN:
        counts = Counter(tail[i : i + CHUNK_LEN] for i in range(len(tail) - CHUNK_LEN + 1))
        chunk60 = max(counts.values())
    else:
        chunk60 = 0
    words = text.split()
    n4 = len(words) - 3
    if n4 >= 1:
        distinct4 = len(set(zip(words, words[1:], words[2:], words[3:]))) / n4
    else:
        distinct4 = float("nan")
    raw = text.encode("utf-8", "ignore")
    gz = (len(zlib.compress(raw, 6)) / len(raw)) if raw else float("nan")
    return n_chars, len(words), chunk60, distinct4, gz


def process_shard(shard_path_str: str) -> dict:
    """Score one packed shard -> per-rollout metric rows (atomic per-shard JSONL)."""
    shard_path = Path(shard_path_str)
    behavior = SHARD_RE.search(shard_path.name).group(1)
    out_path = METRICS_DIR / (shard_path.stem + ".metrics.jsonl")
    manifests: list[dict] = []
    skipped: Counter = Counter()
    rows: list[str] = []
    with shard_path.open() as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("src", "")
            doc = rec.get("doc")
            if src.endswith("_manifest.json"):
                manifests.append({"src": src, "doc": doc})
                continue
            if not isinstance(doc, dict) or "completion" not in doc:
                skipped[src.rsplit("/", 1)[0] or src] += 1
                continue
            comp = doc.get("completion") or ""
            n_chars, n_words, chunk60, distinct4, gz = rollout_metrics(comp)
            meta = doc.get("meta") or {}
            rows.append(
                json.dumps(
                    {
                        "behavior": behavior,
                        "context_id": doc.get("context_id"),
                        "rollout_k": doc.get("rollout_k"),
                        "rung": doc.get("rung"),
                        "split": doc.get("split"),
                        "finish_reason": doc.get("finish_reason"),
                        "cap_hit": doc.get("finish_reason") == "length",
                        "n_chars": n_chars,
                        "n_words": n_words,
                        "chunk60_count": chunk60,
                        "distinct4_ratio": distinct4,
                        "gzip_ratio": gz,
                        "max_new_tokens": meta.get("max_new_tokens"),
                    }
                )
            )
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + ("\n" if rows else ""))
    os.replace(tmp, out_path)
    return {
        "shard": shard_path.name,
        "behavior": behavior,
        "n_rows": len(rows),
        "manifests": manifests,
        "skipped_srcs": dict(skipped),
    }


def run_metrics(shards: list[Path], workers: int = 8) -> None:
    """Parallel metric pass; resume-skips shards whose metric file already exists."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p in shards if not (METRICS_DIR / (p.stem + ".metrics.jsonl")).is_file()]
    done = len(shards) - len(todo)
    if done:
        print(f"[metrics] resume: {done}/{len(shards)} shards already scored, skipping")
    all_manifests: list[dict] = []
    all_skipped: Counter = Counter()
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, res in enumerate(ex.map(process_shard, [str(p) for p in sorted(todo)])):
                all_manifests.extend(res["manifests"])
                all_skipped.update(res["skipped_srcs"])
                print(
                    f"[metrics] shard {k + 1}/{len(todo)} {res['shard']} "
                    f"rows={res['n_rows']} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    man_path = OUT_ROOT / "r35c_manifests.json"
    if all_manifests or not man_path.is_file():
        man_path.write_text(
            json.dumps({"manifests": all_manifests, "skipped_srcs": dict(all_skipped)}, indent=1)
        )
    print(f"[metrics] done: {len(shards)} shards, elapsed={time.time() - t0:.0f}s")


# ---------------------------------------------------------------- phase: analyze


def _load_rollout_df():
    import pandas as pd

    files = sorted(METRICS_DIR.glob("*.metrics.jsonl"))
    if not files:
        raise FileNotFoundError(f"no metric files under {METRICS_DIR} — run --phase metrics")
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["behavior", "context_id", "rollout_k", "rung", "split"])
    n_dupes = n_before - len(df)
    df["chunk_flag"] = df["chunk60_count"] >= CHUNK_FLAG_MIN
    return df, n_dupes


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else float("nan")


def _flag_confusion(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "both": int((a & b).sum()),
        "only_a": int((a & ~b).sum()),
        "only_b": int((~a & b).sum()),
        "neither": int((~a & ~b).sum()),
        "jaccard": _jaccard(a, b),
    }


def analyze_base_rates(df) -> dict:
    """Base rates + metric agreement on the train eval rung; matched-rate thresholds."""
    rung_counts = df.groupby(["behavior", "rung"]).size()
    train = df[df["rung"] == "train"].copy()
    if train.empty:
        raise RuntimeError(f"no rung=='train' rows; rungs present: {sorted(df['rung'].unique())}")

    base_rate = float(train["chunk_flag"].mean())
    thr_d4 = float(np.nanquantile(train["distinct4_ratio"].to_numpy(float), base_rate))
    thr_gz = float(np.nanquantile(train["gzip_ratio"].to_numpy(float), base_rate))
    train["flag_distinct4"] = train["distinct4_ratio"] <= thr_d4
    train["flag_gzip"] = train["gzip_ratio"] <= thr_gz

    flags = ["chunk_flag", "flag_distinct4", "flag_gzip", "cap_hit"]
    agreement = {}
    for i, fa in enumerate(flags):
        for fb in flags[i + 1 :]:
            agreement[f"{fa}|{fb}"] = _flag_confusion(
                train[fa].to_numpy(bool), train[fb].to_numpy(bool)
            )
    # continuous agreement (Spearman); signs: chunk60 up = degenerate, others down
    cont = {}
    pairs = [
        ("chunk60_count", "distinct4_ratio"),
        ("chunk60_count", "gzip_ratio"),
        ("distinct4_ratio", "gzip_ratio"),
        ("chunk60_count", "n_chars"),
        ("gzip_ratio", "n_chars"),
    ]
    for a, b in pairs:
        x = train[a].to_numpy(float)
        y = train[b].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        cont[f"{a}|{b}"] = r35b.spearman(x[ok], y[ok])

    per_behavior = {}
    for b, g in train.groupby("behavior"):
        per_behavior[b] = {
            "n_rollouts": int(len(g)),
            "n_contexts": int(g["context_id"].nunique()),
            "rate_chunk_flag": float(g["chunk_flag"].mean()),
            "rate_flag_distinct4": float(g["flag_distinct4"].mean()),
            "rate_flag_gzip": float(g["flag_gzip"].mean()),
            "rate_cap_hit": float(g["cap_hit"].mean()),
            "chunk60_p50_p95_p99": [
                float(np.quantile(g["chunk60_count"], q)) for q in (0.5, 0.95, 0.99)
            ],
            "chunk60_median_among_capped": float(g.loc[g["cap_hit"], "chunk60_count"].median())
            if g["cap_hit"].any()
            else None,
        }

    return {
        "rows_by_behavior_rung": {f"{b}|{r}": int(n) for (b, r), n in rung_counts.items()},
        "max_new_tokens_values": {
            str(k): int(v) for k, v in train["max_new_tokens"].value_counts().items()
        },
        "train_pooled_chunk_flag_rate": base_rate,
        "matched_rate_thresholds": {"distinct4_ratio": thr_d4, "gzip_ratio": thr_gz},
        "flag_agreement_train": agreement,
        "continuous_spearman_train": cont,
        "per_behavior_train": per_behavior,
    }, train


def context_grain(train) -> dict:
    """Aggregate train-rung rollout metrics to context grain, per behavior."""
    out = {}
    agg = train.groupby(["behavior", "context_id"]).agg(
        n_rollouts=("chunk_flag", "size"),
        n_flagged=("chunk_flag", "sum"),
        max_chunk60=("chunk60_count", "max"),
        distinct4_min=("distinct4_ratio", "min"),
        gzip_min=("gzip_ratio", "min"),
        gzip_mean=("gzip_ratio", "mean"),
        n_cap=("cap_hit", "sum"),
        mean_chars=("n_chars", "mean"),
    )
    agg["frac_flag"] = agg["n_flagged"] / agg["n_rollouts"]
    agg["frac_cap"] = agg["n_cap"] / agg["n_rollouts"]
    for b in BEHAVIORS:
        sub = agg.loc[b].reset_index()
        path = OUT_ROOT / f"r35c_context_grain_{b}.jsonl"
        sub.to_json(path, orient="records", lines=True)
        out[b] = sub.set_index("context_id")
        print(
            f"[context] {b}: {len(sub)} contexts, "
            f"{int(sub['n_flagged'].sum())} flagged rollouts, "
            f"frac contexts with >=1 flagged: {(sub['frac_flag'] > 0).mean():.4f}"
        )
    return out


def _boot_rhos(dv: np.ndarray, arms: dict[str, np.ndarray], n_boot: int, rng) -> dict:
    """Vectorized bootstrap of Spearman rho per arm over context resamples."""
    n = dv.size
    idx = rng.integers(0, n, size=(n_boot, n))
    r_dv = r35b.midranks2d(dv[idx])
    return {
        name: r35b.batched_pearson(r35b.midranks2d(vals[idx]), r_dv) for name, vals in arms.items()
    }


def analyze_cells(ctx_tables: dict, n_boot_degen: int, n_boot_clean: int) -> list[dict]:
    """Stratified rho profiles for the 21 primary cells + degeneration reads."""
    all_cells = {b: r35b.load_cells(b) for b in BEHAVIORS}
    max_budget = {
        b: max(c["unit"]["budget_l"] for c in cs if c["unit"]["u_rung_label"] in r35b.CORE_LABELS)
        for b, cs in all_cells.items()
    }
    rows = []
    for b in BEHAVIORS:
        ctab = ctx_tables[b]
        for c in all_cells[b]:
            u = c["unit"]
            if u["variant"] not in r35b.VARIANTS or u["seed"] != 0:
                continue
            if not r35b.is_primary(u, max_budget[b]):
                continue
            z = np.load(c["npz"], allow_pickle=True)
            cids = z["context_ids"].astype(str)
            dv = z["dv"].astype(np.float64)
            p6 = z[f"pred__{r35b.ARM_MAP}"].astype(np.float64)
            p11 = z[f"pred__{r35b.ARM_ORACLE}"].astype(np.float64)
            p1 = z[f"pred__{r35b.ARM_CTX}"].astype(np.float64)
            ok = np.isfinite(dv) & np.isfinite(p6) & np.isfinite(p11) & np.isfinite(p1)
            cids, dv, p6, p11, p1 = cids[ok], dv[ok], p6[ok], p11[ok], p1[ok]
            if r35b.spearman(p11, dv) < 0:  # sign convention: oracle reads positive
                p6, p11, p1 = -p6, -p11, -p1

            joined = np.array([cid in ctab.index for cid in cids])
            join_rate = float(joined.mean())
            sub = ctab.loc[cids[joined]]
            dv_j, p6_j, p11_j, p1_j = dv[joined], p6[joined], p11[joined], p1[joined]
            frac_flag = sub["frac_flag"].to_numpy(float)
            gzip_min = sub["gzip_min"].to_numpy(float)
            frac_cap = sub["frac_cap"].to_numpy(float)
            d = p6_j - p11_j
            sd11 = float(np.std(p11_j, ddof=1))
            absd = np.abs(d) / sd11 if sd11 > 0 else np.abs(d)

            fl = r35b.frozen_layers(c["rec"])
            layer_map = fl.get(r35b.ARM_MAP)
            layer_oracle = fl.get(r35b.ARM_ORACLE)
            excluded_evil5000 = b == "evil" and u["u_rung_label"] == "5000"

            row = {
                "behavior": b,
                "cell": r35b.cell_slug(u),
                "regime": u["regime"],
                "u_rung_label": u["u_rung_label"],
                "n_cell": int(dv.size),
                "n_joined": int(joined.sum()),
                "join_rate": join_rate,
                "frozen_layer_map_arm": layer_map,
                "frozen_layer_oracle_arm": layer_oracle,
                "layer_matched": layer_map == layer_oracle,
                "excluded_evil5000": excluded_evil5000,
                # confound reads (context grain, this cell's joined contexts)
                "rho_fracflag_dv": r35b.spearman(frac_flag, dv_j),
                "rho_gzipmin_dv": r35b.spearman(gzip_min, dv_j),
                "rho_fracflag_absd": r35b.spearman(frac_flag, absd),
                "rho_gzipmin_absd": r35b.spearman(gzip_min, absd),
                "rho_fraccap_absd": r35b.spearman(frac_cap, absd),
            }

            strata_def = {
                "clean": frac_flag == 0,
                "degen_any": frac_flag > 0,
                "degen_2plus": sub["n_flagged"].to_numpy(int) >= 2,
                "cap_any": frac_cap > 0,
            }
            arms = {"rho6": p6_j, "rho11": p11_j, "rho1": p1_j}
            strata_out = {}
            for name, mask in strata_def.items():
                ns = int(mask.sum())
                st = {"n": ns}
                if ns >= 10:
                    for arm, vals in arms.items():
                        r = r35b.spearman(vals[mask], dv_j[mask])
                        st[arm] = r
                        st[f"{arm}_ci"] = r35b.bonett_wright_ci(r, ns)
                    st["gap"] = (
                        st["rho6"] - st["rho11"]
                        if np.isfinite(st["rho6"]) and np.isfinite(st["rho11"])
                        else float("nan")
                    )
                    st["dv_mean"] = float(dv_j[mask].mean())
                    st["dv_sd"] = float(dv_j[mask].std())
                    st["mean_absd_norm"] = float(absd[mask].mean())
                strata_out[name] = st
            row["strata"] = strata_out

            # bootstrap: clean vs degen_any difference, per arm + gap
            cl, dg = strata_def["clean"], strata_def["degen_any"]
            if int(dg.sum()) >= 10 and int(cl.sum()) >= 10:
                seed = abs(hash(("r35c", b, row["cell"]))) % 2**32
                rng = np.random.default_rng(seed)
                bt_d = _boot_rhos(dv_j[dg], {k: v[dg] for k, v in arms.items()}, n_boot_degen, rng)
                bt_c = _boot_rhos(dv_j[cl], {k: v[cl] for k, v in arms.items()}, n_boot_clean, rng)
                pair = rng.integers(0, n_boot_clean, size=n_boot_degen)
                boot = {}
                for arm in arms:
                    delta = bt_d[arm] - bt_c[arm][pair]
                    ok_b = np.isfinite(delta)
                    boot[f"delta_{arm}"] = float(np.nanmean(delta))
                    boot[f"delta_{arm}_ci"] = [
                        float(np.nanquantile(delta[ok_b], q)) for q in (0.025, 0.975)
                    ]
                dgap = (bt_d["rho6"] - bt_d["rho11"]) - (bt_c["rho6"] - bt_c["rho11"])[pair]
                ok_b = np.isfinite(dgap)
                boot["delta_gap"] = float(np.nanmean(dgap))
                boot["delta_gap_ci"] = [
                    float(np.nanquantile(dgap[ok_b], q)) for q in (0.025, 0.975)
                ]
                # bootstrap "p": doubled smaller tail of the delta-gap draws vs 0
                boot["delta_gap_p_two_sided"] = float(
                    min(
                        1.0,
                        2
                        * min(
                            (1 + int((dgap[ok_b] <= 0).sum())) / (int(ok_b.sum()) + 1),
                            (1 + int((dgap[ok_b] >= 0).sum())) / (int(ok_b.sum()) + 1),
                        ),
                    )
                )
                row["boot_clean_vs_degen"] = boot
            rows.append(row)
            print(
                f"[cells] {b} {row['cell']}: joined {row['n_joined']}/{row['n_cell']}, "
                f"degen_any n={strata_out['degen_any']['n']}",
                flush=True,
            )
    return rows


def analyze(n_boot_degen: int, n_boot_clean: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df, n_dupes = _load_rollout_df()
    print(f"[analyze] {len(df)} rollout rows ({n_dupes} duplicate keys dropped)")
    base, train = analyze_base_rates(df)
    base["n_rollout_rows_total"] = int(len(df))
    base["n_duplicate_keys_dropped"] = int(n_dupes)

    # manifest verification anchor: recorded truncation counts vs computed cap hits
    man = json.loads((OUT_ROOT / "r35c_manifests.json").read_text())
    recorded = {}
    for m in man.get("manifests", []):
        doc = m.get("doc") or {}
        for key in ("n_truncated_rollouts", "n_truncated"):
            if isinstance(doc, dict) and key in doc:
                recorded[m["src"]] = doc[key]
    base["manifest_truncation_recorded"] = recorded
    base["computed_cap_hits_train"] = {
        b: int(train[(train["behavior"] == b)]["cap_hit"].sum()) for b in BEHAVIORS
    }
    (OUT_ROOT / "r35c_base_rates.json").write_text(json.dumps(base, indent=1))

    ctx_tables = context_grain(train)
    cells = analyze_cells(ctx_tables, n_boot_degen, n_boot_clean)
    (OUT_ROOT / "r35c_cell_strata.json").write_text(json.dumps(cells, indent=1))

    # ---- headline summary ----
    def _med(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else None

    included = [r for r in cells if not r["excluded_evil5000"]]
    matched = [r for r in cells if r["layer_matched"] and not r["excluded_evil5000"]]

    def _deltas(rows, key):
        return [r["boot_clean_vs_degen"][key] for r in rows if "boot_clean_vs_degen" in r]

    summary = {
        "design": "context strata: clean (0 flagged rollouts) vs degenerate (>=1 rollout "
        f"with a 60-char chunk repeating >={CHUNK_FLAG_MIN}x in its last {TAIL_LEN} chars); "
        "delta = rho(degen) - rho(clean) per arm, bootstrap CIs over context resamples",
        "n_primary_cells": len(cells),
        "n_included_cells": len(included),
        "n_layer_matched_cells": len(matched),
        "layer_matched_cells": [f"{r['behavior']} {r['cell']}" for r in matched],
        "per_behavior": {
            b: {
                "delta_rho6_median": _med(
                    _deltas([r for r in included if r["behavior"] == b], "delta_rho6")
                ),
                "delta_rho11_median": _med(
                    _deltas([r for r in included if r["behavior"] == b], "delta_rho11")
                ),
                "delta_rho1_median": _med(
                    _deltas([r for r in included if r["behavior"] == b], "delta_rho1")
                ),
                "delta_gap_median": _med(
                    _deltas([r for r in included if r["behavior"] == b], "delta_gap")
                ),
                "rho_fracflag_dv_median": _med(
                    [r["rho_fracflag_dv"] for r in included if r["behavior"] == b]
                ),
                "rho_gzipmin_dv_median": _med(
                    [r["rho_gzipmin_dv"] for r in included if r["behavior"] == b]
                ),
                "rho_fracflag_absd_median": _med(
                    [r["rho_fracflag_absd"] for r in included if r["behavior"] == b]
                ),
                "rho_gzipmin_absd_median": _med(
                    [r["rho_gzipmin_absd"] for r in included if r["behavior"] == b]
                ),
            }
            for b in BEHAVIORS
        },
        "pooled_included": {
            "delta_rho6_median": _med(_deltas(included, "delta_rho6")),
            "delta_rho11_median": _med(_deltas(included, "delta_rho11")),
            "delta_gap_median": _med(_deltas(included, "delta_gap")),
            "n_cells_delta_gap_negative": int(
                sum(1 for v in _deltas(included, "delta_gap") if np.isfinite(v) and v < 0)
            ),
        },
        "layer_matched_headline": [
            {
                "cell": f"{r['behavior']} {r['cell']}",
                "n_degen": r["strata"]["degen_any"]["n"],
                "clean": {k: r["strata"]["clean"].get(k) for k in ("rho6", "rho11", "rho1", "gap")},
                "degen": {
                    k: r["strata"]["degen_any"].get(k) for k in ("rho6", "rho11", "rho1", "gap")
                },
                "boot": r.get("boot_clean_vs_degen"),
            }
            for r in matched
        ],
    }
    (OUT_ROOT / "r35c_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["pooled_included"], indent=1))


# ---------------------------------------------------------------- phase: figures


BEHAVIOR_LABEL = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}


def figures() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    colors = paper_palette_blog(3)
    bcol = dict(zip(BEHAVIORS, colors))
    cells = json.loads((OUT_ROOT / "r35c_cell_strata.json").read_text())
    base = json.loads((OUT_ROOT / "r35c_base_rates.json").read_text())

    # --- figure 1: metric distributions on train rollouts, flag thresholds marked ---
    files = sorted(METRICS_DIR.glob("*.metrics.jsonl"))
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df = df[df["rung"] == "train"]
    thr = base["matched_rate_thresholds"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = df["chunk60_count"].to_numpy(float)
    axes[0].hist(np.log10(np.maximum(x, 1)), bins=60, color="0.45")
    axes[0].axvline(np.log10(CHUNK_FLAG_MIN), color="crimson", lw=1.2)
    axes[0].set_xlabel("log10(repeats of the most-common\n60-char chunk, last 3,000 chars)")
    axes[0].set_yscale("log")
    axes[1].hist(df["distinct4_ratio"].dropna(), bins=60, color="0.45")
    axes[1].axvline(thr["distinct4_ratio"], color="crimson", lw=1.2)
    axes[1].set_xlabel("Distinct 4-gram ratio\n(lower = more repetitive)")
    axes[1].set_yscale("log")
    axes[2].hist(df["gzip_ratio"].dropna(), bins=60, color="0.45")
    axes[2].axvline(thr["gzip_ratio"], color="crimson", lw=1.2)
    axes[2].set_xlabel("Compression ratio\n(lower = more repetitive)")
    axes[2].set_yscale("log")
    axes[0].set_ylabel("Rollouts (log scale)")
    fig.suptitle(
        "Degeneration measures over train-rung rollouts (red line = degeneration flag threshold)"
    )
    fig.tight_layout()
    savefig_paper(fig, "r35c_metric_distributions", dir=FIG_ROOT)
    plt.close(fig)

    # --- figure 2: does the oracle fall too? delta rho (degen - clean) per cell ---
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lims = [0.0, 0.0]
    for r in cells:
        if "boot_clean_vs_degen" not in r:
            continue
        bt = r["boot_clean_vs_degen"]
        x, y = bt["delta_rho11"], bt["delta_rho6"]
        xerr = np.array([[x - bt["delta_rho11_ci"][0]], [bt["delta_rho11_ci"][1] - x]])
        yerr = np.array([[y - bt["delta_rho6_ci"][0]], [bt["delta_rho6_ci"][1] - y]])
        excluded = r["excluded_evil5000"]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o" if r["layer_matched"] else "s",
            ms=8 if r["layer_matched"] else 5,
            mfc="none" if excluded else bcol[r["behavior"]],
            color=bcol[r["behavior"]],
            elinewidth=0.7,
            alpha=0.9 if r["layer_matched"] else 0.6,
        )
        lims = [min(lims[0], x, y), max(lims[1], x, y)]
    pad = 0.05 + 0.1 * (lims[1] - lims[0])
    lo, hi = lims[0] - pad, lims[1] + pad
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.9, ls="--")
    ax.axhline(0, color="0.75", lw=0.7)
    ax.axvline(0, color="0.75", lw=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Change in real-answer (oracle) correlation on degenerate contexts")
    ax.set_ylabel("Change in mapped-answer correlation on degenerate contexts")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=bcol[b], label=BEHAVIOR_LABEL[b])
        for b in BEHAVIORS
    ]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="0.3", ms=8, label="Layer-matched cell"),
        plt.Line2D(
            [], [], marker="s", ls="", color="0.3", mfc="none", label="Excluded (blown-up fit)"
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    ax.set_title(
        "Below the dashed line = the map loses more than the oracle\non degenerate contexts"
    )
    fig.tight_layout()
    savefig_paper(fig, "r35c_oracle_vs_map_drop", dir=FIG_ROOT)
    plt.close(fig)

    # --- figure 3: layer-matched headline cells, per-stratum rho with CIs ---
    matched = [r for r in cells if r["layer_matched"] and not r["excluded_evil5000"]]
    if matched:
        fig, axes = plt.subplots(1, len(matched), figsize=(3.6 * len(matched), 4.0), sharey=True)
        axes = np.atleast_1d(axes)
        arm_labels = [("rho6", "Mapped answer"), ("rho11", "Real answer"), ("rho1", "Context")]
        for ax, r in zip(axes, matched):
            for j, (arm, lab) in enumerate(arm_labels):
                for k, (sname, slabel, mfc) in enumerate(
                    [("clean", "clean", bcol[r["behavior"]]), ("degen_any", "degenerate", "none")]
                ):
                    st = r["strata"][sname]
                    if arm not in st:
                        continue
                    ci = st[f"{arm}_ci"]
                    xpos = j + (k - 0.5) * 0.28
                    ax.errorbar(
                        xpos,
                        st[arm],
                        yerr=[[st[arm] - ci[0]], [ci[1] - st[arm]]],
                        fmt="o",
                        color=bcol[r["behavior"]],
                        mfc=mfc,
                        ms=7,
                        elinewidth=1.0,
                        label=slabel if j == 0 else None,
                    )
            ax.set_xticks(range(len(arm_labels)))
            ax.set_xticklabels([lab for _, lab in arm_labels], fontsize=8)
            ax.axhline(0, color="0.75", lw=0.7)
            n_dg = r["strata"]["degen_any"]["n"]
            ax.set_title(
                f"{BEHAVIOR_LABEL[r['behavior']]} — {r['regime']}·{r['u_rung_label']} map\n"
                f"(degenerate n={n_dg})",
                fontsize=9,
            )
        axes[0].set_ylabel("Spearman correlation with behavior score")
        axes[0].legend(frameon=False, fontsize=8, title="Context stratum", title_fontsize=8)
        fig.suptitle("Layer-matched cells: prediction on clean vs degenerate contexts", y=1.02)
        fig.tight_layout()
        savefig_paper(fig, "r35c_headline_cells", dir=FIG_ROOT)
        plt.close(fig)

    # --- figure 4: degeneration vs map distortion / vs DV (confound), per cell ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    inc = [r for r in cells if not r["excluded_evil5000"]]
    xs = np.arange(len(inc))
    # sign convention: gzip_min LOW = more repetitive, so negate to read
    # "correlation with degeneration" (positive = more degenerate <-> larger value)
    for ax, key, title in [
        (axes[0], "rho_gzipmin_absd", "Degeneration vs per-context map distortion |d|"),
        (axes[1], "rho_gzipmin_dv", "Degeneration vs behavior score (confound check)"),
    ]:
        for i, r in enumerate(inc):
            ax.bar(i, -r[key], color=bcol[r["behavior"]], width=0.75)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{r['regime']}·{r['u_rung_label']}" for r in inc], rotation=90, fontsize=7
        )
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(
        "Spearman correlation with degeneration\n(compression-based score; positive ="
        " more degenerate)"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=bcol[b]) for b in BEHAVIORS]
    fig.legend(
        handles,
        [BEHAVIOR_LABEL[b] for b in BEHAVIORS],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    savefig_paper(fig, "r35c_degen_vs_distortion", dir=FIG_ROOT)
    plt.close(fig)
    print(f"[figures] written under {FIG_ROOT}")


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=["stage", "metrics", "analyze", "figures", "all"],
        default="all",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot-degen", type=int, default=2000)
    ap.add_argument("--n-boot-clean", type=int, default=500)
    args = ap.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("stage", "all"):
        shards = stage()
    else:
        shards = sorted(STAGE_DIR.glob("labeling_*.jsonl"))
    if args.phase in ("metrics", "all"):
        run_metrics(shards, workers=args.workers)
    if args.phase in ("analyze", "all"):
        analyze(args.n_boot_degen, args.n_boot_clean)
    if args.phase in ("figures", "all"):
        figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
