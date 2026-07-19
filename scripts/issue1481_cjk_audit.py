"""#1481 language-intrusion (CJK) audit — scan + headline recounts (committed).

Produces two committed artifacts under ``eval_results/issue_1481/analysis/``:

- ``cjk_intrusion_scan.json`` — per-(arm, read context) intruded/total counts
  over every #1481 panel pool (40 verdict arms x 6 contexts) + the base-arm
  pools, keyed unambiguously by relative HF path (the earlier /tmp draft scan
  under-counted 10 pools via ``arm|ctx`` key collisions; this artifact
  supersedes it).
- ``cjk_recount_headlines.json`` — exact zeroed + excluded recounts of the
  three behavior-headline D reads. Conventions: **zeroed** = an intruded
  completion's judge label is forced non-positive in place (denominator kept);
  **excluded** = intruded rows dropped from numerator and denominator. Per-item
  labels are rejoined from ``analysis/judge_packed/`` (all_scores per draw,
  mean > 50, idmap-resolved). The casual seed-42 leg reuses #1434 committed
  panels whose per-item labels are not in this run's cache: its
  fired-and-intruded counts come from the committed #1434 audit
  ``eval_results/issue_1434/writing-style-positive-only-regime/po_intrusion_audit.json``
  (ws-po-bare rows; a cross-pass source with +/-3 per-context judge-draw
  variation vs the committed aggregate, labeled ``cross-pass`` in the output).

Inputs (HF data repo, downloaded on demand to --cache-dir):
  issue1481_conpos_grid/raw_completions/{panel,base_arms}/**
  issue1481_conpos_grid/analysis/judge_packed/judge_{cas,imp,syc}.shard00.jsonl

Usage:
    uv run python scripts/issue1481_cjk_audit.py \
        --analysis-dir eval_results/issue_1481/analysis --cache-dir /tmp/i1481_cjk
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1481_conpos_grid"
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")
HEADLINE = {"cas": ["bare"], "imp": ["bare"], "syc": ["conv", "icl"]}
INSTRUMENT = {
    "cas": "pv_trait_score",
    "imp": "registered_graded_r23",
    "syc": "registered_graded_r23",
}
I1434_AUDIT = Path(
    "eval_results/issue_1434/writing-style-positive-only-regime/po_intrusion_audit.json"
)
REUSED_1434_SEED = 42  # cas seed-42 panels are reused #1434 committed reads


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - hw, ctr + hw


def _newcombe(k1: int, n1: int, k2: int, n2: int) -> dict:
    l1, u1 = _wilson(k1, n1)
    l2, u2 = _wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return {
        "D": d,
        "newcombe_95": [
            d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
        ],
    }


def _download_pools(cache_dir: Path) -> list[Path]:
    api = HfApi()
    paths: list[str] = []
    for sub in ("panel", "base_arms"):
        # HUB_VERIFY_RETRY_EXEMPT: frozen #1481 repro script; one-shot scoped listing (#1552)
        for t in api.list_repo_tree(
            DATA_REPO, f"{PREFIX}/raw_completions/{sub}", repo_type="dataset", recursive=True
        ):
            if t.path.endswith(".json"):
                paths.append(t.path)

    def dl(p: str) -> str:
        return hf_hub_download(DATA_REPO, p, repo_type="dataset", local_dir=str(cache_dir))

    with ThreadPoolExecutor(16) as ex:
        local = list(ex.map(dl, paths))
    return [Path(p) for p in local]


def _load_judge_index(cache_dir: Path) -> tuple[dict, dict]:
    """(raws, idmaps): all_scores per '<instrument>_pn-<arm>-<ctx>' tag; idmap per 'pn-...' tag."""
    raws: dict = {}
    idmaps: dict = {}
    for beh in ("cas", "imp", "syc"):
        p = hf_hub_download(
            DATA_REPO,
            f"{PREFIX}/analysis/judge_packed/judge_{beh}.shard00.jsonl",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                if "judge_raw" in row["path"]:
                    tag = row["path"].split("judge_raw_")[-1].removesuffix(".json")
                    raws[tag] = row["content"]["all_scores"]
                elif "/idmap_" in row["path"]:
                    tag = row["path"].split("/idmap_")[-1].removesuffix(".json")
                    idmaps[tag] = row["content"]  # hashed -> original
    return raws, idmaps


def _item_means(all_scores: dict, idmap: dict) -> dict:
    per: dict = collections.defaultdict(list)
    for k, v in all_scores.items():
        base = idmap.get(k.split("__")[0], k.split("__")[0])
        s = v.get("score")
        if isinstance(s, (int, float)) and 0 <= s <= 100:
            per[base].append(s)
    return {k: sum(v) / len(v) for k, v in per.items() if v}


def scan(pool_files: list[Path], cache_dir: Path) -> dict:
    out: dict = {}
    for f in sorted(pool_files):
        rel = str(f).split(f"{PREFIX}/raw_completions/")[-1]
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError as e:  # fail loud, keep the row
            out[rel] = {"error": str(e)[:80]}
            continue
        comps = d.get("completions")
        if comps is None:
            continue
        n = i = 0
        for qlist in comps:
            for c in qlist:
                n += 1
                if CJK_RE.search(c):
                    i += 1
        out[rel] = {"n": n, "intruded": i}
    return out


def recount(cache_dir: Path, analysis_dir: Path) -> dict:
    rc = json.loads((analysis_dir / "regime_contrast_content.json").read_text())
    vm = json.loads((analysis_dir / "verdict_manifest.json").read_text())
    raws, idmaps = _load_judge_index(cache_dir)
    audit1434 = json.loads(I1434_AUDIT.read_text())
    out: dict = {
        "conventions": {
            "zeroed": "intruded completion's judge label forced non-positive in place; denominator kept",
            "excluded": "intruded rows dropped from numerator and denominator",
            "positive": "mean judge score over kept draws > 50",
        },
        "behaviors": {},
    }
    for beh, ctxs in HEADLINE.items():
        agg = {r: {"k": 0, "n": 0, "intruded": 0, "fired_and_intruded": 0} for r in ("con", "po")}
        per_arm: dict = {}
        for ctx in ctxs:
            cell = rc["behavior_contexts"][beh][ctx]
            src = cell["source_ctx"]
            for seed, sv in vm["content"][beh][ctx]["seeds"].items():
                for reg in ("con", "po"):
                    arm = sv[reg]["arm_id"]
                    if beh == "cas" and int(seed) == REUSED_1434_SEED:
                        # cross-pass leg: committed #1481 aggregate k/n + #1434 audit intrusion
                        pagg = json.loads((analysis_dir / "panel_aggregate_cas.json").read_text())[
                            "arms"
                        ][arm]["contexts"]
                        rows = [
                            c
                            for c in audit1434["cells"]
                            if c["training_cell"] == "ws-po-bare" and c["read_ctx"] != src
                        ]
                        k = sum(v["k_positive"] for cn, v in pagg.items() if cn != src)
                        n = sum(v["n_scored"] for cn, v in pagg.items() if cn != src)
                        i = sum(r[reg]["n_intruded"] for r in rows)
                        fi = sum(r[reg]["fired_and_intruded"] for r in rows)
                        per_arm[arm] = {
                            "k": k,
                            "n": n,
                            "intruded": i,
                            "fired_and_intruded": fi,
                            "label_source": "cross-pass (#1434 po_intrusion_audit.json ws-po-bare; "
                            "+/-3 per-context judge-draw variation vs committed aggregate)",
                        }
                    else:
                        k = n = i = fi = 0
                        arm_dir = cache_dir / PREFIX / "raw_completions" / "panel" / arm
                        for cf in sorted(arm_dir.glob("completions__trained__*.json")):
                            rctx = cf.name.split("completions__trained__")[-1].removesuffix(".json")
                            if rctx == src:
                                continue
                            comps = json.loads(cf.read_text())["completions"]
                            tag = f"pn-{arm}-{rctx}"
                            means = _item_means(
                                raws[f"{INSTRUMENT[beh]}_{tag}"], idmaps.get(tag, {})
                            )
                            for qi, qlist in enumerate(comps):
                                for ci, c in enumerate(qlist):
                                    m = means.get(f"{tag}-q{qi:03d}-c{ci:03d}")
                                    if m is None:
                                        continue
                                    pos = m > 50
                                    intr = bool(CJK_RE.search(c))
                                    n += 1
                                    k += int(pos)
                                    i += int(intr)
                                    fi += int(pos and intr)
                        per_arm[arm] = {
                            "k": k,
                            "n": n,
                            "intruded": i,
                            "fired_and_intruded": fi,
                            "label_source": "exact (this run's judge_packed join)",
                        }
                    a = agg[reg]
                    a["k"] += per_arm[arm]["k"]
                    a["n"] += per_arm[arm]["n"]
                    a["intruded"] += per_arm[arm]["intruded"]
                    a["fired_and_intruded"] += per_arm[arm]["fired_and_intruded"]
        po, con = agg["po"], agg["con"]
        out["behaviors"][beh] = {
            "contexts": ctxs,
            "per_arm": per_arm,
            "committed": _newcombe(po["k"], po["n"], con["k"], con["n"]),
            "zeroed": _newcombe(
                po["k"] - po["fired_and_intruded"],
                po["n"],
                con["k"] - con["fired_and_intruded"],
                con["n"],
            ),
            "excluded": _newcombe(
                po["k"] - po["fired_and_intruded"],
                po["n"] - po["intruded"],
                con["k"] - con["fired_and_intruded"],
                con["n"] - con["intruded"],
            ),
            "pooled_inputs": agg,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    args = p.parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    cache_dir = Path(args.cache_dir)
    pool_files = _download_pools(cache_dir)
    scan_out = scan(pool_files, cache_dir)
    pools = [v for v in scan_out.values() if "n" in v]
    summary = {
        "n_pools": len(pools),
        "n_completions": sum(v["n"] for v in pools),
        "n_intruded": sum(v["intruded"] for v in pools),
        "regex": CJK_RE.pattern,
    }
    (analysis_dir / "cjk_intrusion_scan.json").write_text(
        json.dumps({"summary": summary, "pools": scan_out}, indent=1, sort_keys=True)
    )
    rec = recount(cache_dir, analysis_dir)
    (analysis_dir / "cjk_recount_headlines.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True)
    )
    print(
        f"[i1481-cjk] scan: {summary['n_intruded']}/{summary['n_completions']} intruded "
        f"over {summary['n_pools']} pools"
    )
    for beh, b in rec["behaviors"].items():
        z = b["zeroed"]
        print(
            f"[i1481-cjk] {beh}: committed D={b['committed']['D']:.3f} "
            f"zeroed D={z['D']:.3f} [{z['newcombe_95'][0]:.3f},{z['newcombe_95'][1]:.3f}] "
            f"excluded D={b['excluded']['D']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
