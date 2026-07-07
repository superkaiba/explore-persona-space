"""Issue #825 `base-separator-control` p0_stage (plan v18 G0).

Stages the two pinned #931 pair files (revision-addressed = content pin) into
``<data-dir>/pairs/`` and the instruct armC anchor store INTO THE CONSUMER'S
EXACT LAYOUT ``<anchor-dir>/store/armC/<basename>`` (the fixed
``issue931_fit_cells.load_regime_store`` path — artifact-reuse check (h)(iv),
#928: the hub-rel -> local-rel mapping is ONE pure function threaded through
staging and the completeness check). Then:

  - pairwise provenance-coherence check (j): max(pair-file last-commit dates
    @ the pinned pairs revision) <= min(anchor-store file dates) — a pair file
    regenerated AFTER the instruct capture would confound the base-vs-instruct
    comparison (fail loud; #922).
  - consumer-open probe (pre-G3b hard assert): ``load_regime_store`` OPENS the
    staged tree AND ``n == --expect-n`` (3600 in production; ``auto`` = the
    staged sidecars' row sum for the 1-shard smoke subset).
  - ``--self-test`` (smoke): the probe must REJECT a planted mis-staged tree
    (shards nested one level too deep) — the (h)(iv) failure the probe exists
    to catch.

Writes ``<out-dir>/base_sep_run_manifest.json`` (revisions, file lists, pair
sha256s, provenance dates, model ids).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps bind before torch import (fit931 imports torch)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

SCRIPT = "scripts/issue825_base_sep_stage.py"
PAIR_FILES = ("articles_armC.jsonl", "pairs_armC.jsonl")
PAIRS_PREFIX = "issue931_story_map/raw_completions/pairs_meta"
ANCHOR_PREFIX = "issue931_story_map/analysis_tensors/armC"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B"


def store_local_relpath(hub_path: str) -> str:
    """Pure hub-rel -> local-rel mapping for the anchor store: BASENAME into
    the flat ``store/armC/`` consumer dir (#928 (h)(iv): one function threads
    staging destinations and the completeness check)."""
    return Path(hub_path).name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--anchor-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pairs-revision", type=str, required=True)
    ap.add_argument("--anchor-shards", type=str, default="all", help="'all' or an int shard count")
    ap.add_argument("--expect-n", type=str, default="3600", help="int, or 'auto' (sidecar sum)")
    ap.add_argument("--self-test", action="store_true", help="smoke: probe rejects mis-staging")
    # onpolicy-separator-control parametrization (plan section 2 — the round-6
    # machinery reused for a SECOND anchor store; defaults preserve round-6).
    ap.add_argument(
        "--anchor-prefix",
        type=str,
        default=ANCHOR_PREFIX,
        help="Hub prefix of the anchor store (default: the #931 instruct armC store)",
    )
    ap.add_argument(
        "--anchor-revision",
        type=str,
        default="",
        help="data-repo revision pin for the anchor store ('' = resolve HEAD, the "
        "round-6 default; the base exogenous store passes its content pin)",
    )
    ap.add_argument(
        "--skip-pairs",
        action="store_true",
        help="skip the pair-file staging (second anchor-store call of the "
        "onpolicy dispatcher — pairs already staged by the first call)",
    )
    ap.add_argument(
        "--manifest-name",
        type=str,
        default="base_sep_run_manifest.json",
        help="manifest filename under --out-dir (distinct per anchor-store call)",
    )
    return ap.parse_args()


def _fetch(path_in_repo: str, revision: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    got = hub.retry_transient(
        lambda: hf_hub_download(
            common.HF_DATA_REPO,
            path_in_repo,
            repo_type="dataset",
            revision=revision,
            local_dir=dest_dir,
        ),
        what=f"stage {path_in_repo}",
    )
    return Path(got)


def stage_pairs(data_dir: Path, revision: str) -> dict:
    """Download the two pinned pair files -> <data-dir>/pairs/ (+ sha256 pins)."""
    import shutil

    pairs_dir = data_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name in PAIR_FILES:
        got = _fetch(f"{PAIRS_PREFIX}/{name}", revision, data_dir / "_hf")
        shutil.copy2(got, pairs_dir / name)
        shas[name] = common.sha256_file(pairs_dir / name)
        print(f"[i825-bs] staged pair file {name} sha256={shas[name][:12]}")
    # Loader-contract spot assert (plan assumption 10): first article row.
    first = json.loads(next(iter((pairs_dir / PAIR_FILES[0]).open(encoding="utf-8"))))
    assert "window_id" in first and "input_ids" in first, sorted(first)
    return shas


def stage_anchor_store(
    anchor_dir: Path, shards: str, *, prefix: str = ANCHOR_PREFIX, revision: str = ""
) -> tuple[str, list[str]]:
    """Stage one armC anchor store into <anchor-dir>/store/armC/ (consumer layout).

    Defaults preserve round-6 (the #931 instruct store @ resolved repo HEAD);
    the onpolicy dispatcher's second call passes the base store's prefix +
    content-pin revision.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    revision = revision or api.repo_info(common.HF_DATA_REPO, repo_type="dataset").sha
    entries = sorted(
        e.path
        for e in hub.retry_transient(
            lambda: list(
                api.list_repo_tree(
                    common.HF_DATA_REPO,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    revision=revision,
                )
            ),
            what="list anchor store tree",
        )
    )
    assert entries, f"no files under {prefix}"
    if shards != "all":
        n = int(shards)
        wanted = {f"armC_shard{i:03d}{ext}" for i in range(n) for ext in (".pt", ".json")}
        entries = [p for p in entries if Path(p).name in wanted]
        assert len(entries) == len(wanted), (sorted(wanted), entries)
    store_dir = anchor_dir / "store" / "armC"
    store_dir.mkdir(parents=True, exist_ok=True)
    mapped = {p: store_local_relpath(p) for p in entries}
    assert len(set(mapped.values())) == len(mapped), "hub->local mapping collision"
    for p, rel in mapped.items():
        got = _fetch(p, revision, anchor_dir / "_hf")
        target = store_dir / rel
        if not target.exists():
            target.symlink_to(got.resolve())
        print(f"[i825-bs] staged anchor {rel}")
    # Fail-loud completeness: every mapped destination exists; the consumer's
    # entry files (>=1 .pt shard + its .json sidecar) are present.
    missing = [rel for rel in mapped.values() if not (store_dir / rel).exists()]
    assert not missing, f"staging incomplete: {missing}"
    pts = sorted(store_dir.glob("armC_shard*.pt"))
    assert pts and (store_dir / (pts[0].stem + ".json")).exists(), (
        f"consumer entry files missing under {store_dir}"
    )
    return revision, entries


def provenance_coherence(pairs_rev: str, store_paths: list[str], store_rev: str = "") -> dict:
    """Artifact-reuse check (j): pair files must NOT postdate the anchor store.

    ``store_rev`` dates the store files AT THE CONSUMED REVISION (the base
    exogenous store's content pin); '' keeps the round-6 main-HEAD read.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    pair_paths = [f"{PAIRS_PREFIX}/{n}" for n in PAIR_FILES]
    info_pairs = api.get_paths_info(
        common.HF_DATA_REPO, pair_paths, expand=True, repo_type="dataset", revision=pairs_rev
    )
    store_kw = {"revision": store_rev} if store_rev else {}
    info_store = api.get_paths_info(
        common.HF_DATA_REPO, store_paths, expand=True, repo_type="dataset", **store_kw
    )
    assert len(info_pairs) == len(pair_paths) and len(info_store) == len(store_paths)
    max_pair = max(i.last_commit.date for i in info_pairs)
    min_store = min(i.last_commit.date for i in info_store)
    assert max_pair <= min_store, (
        f"provenance INCOHERENT: pair files last-committed {max_pair} AFTER the earliest "
        f"anchor-store file {min_store} — pairs regenerated after the instruct capture; "
        "the base-vs-instruct comparison would be confounded (plan section 4 row (j)); halt"
    )
    return {"max_pair_date": str(max_pair), "min_store_date": str(min_store)}


def consumer_open_probe(anchor_dir: Path, expect_n: str) -> int:
    """(h)(iv) pre-G3b hard assert: the CONSUMER's own loader opens the staged tree."""
    store_dir = anchor_dir / "store" / "armC"
    store = fit931.load_regime_store(store_dir, "armC")
    n = int(store["row_ids"].shape[0])
    if expect_n == "auto":
        expected = sum(
            json.loads(sc.read_text())["n_rows"]
            for sc in sorted(store_dir.glob("armC_shard*.json"))
        )
    else:
        expected = int(expect_n)
    assert n == expected and n > 0, (n, expected)
    print(f"[i825-bs] consumer-open probe PASS: load_regime_store opened, n={n}")
    return n


def self_test_misstaged(anchor_dir: Path) -> None:
    """Smoke-only: the probe must REJECT a tree staged one level too deep."""
    bad = anchor_dir / "_selftest" / "store" / "armC" / "armC"  # nested one extra level
    bad.mkdir(parents=True, exist_ok=True)
    src = next((anchor_dir / "store" / "armC").glob("armC_shard*.pt"))
    link = bad / src.name
    if not link.exists():
        link.symlink_to(src.resolve())
    try:
        fit931.load_regime_store(bad.parent, "armC")
    except AssertionError:
        print("[i825-bs] mis-staging self-test PASS: consumer-open probe rejected nested tree")
        return
    raise SystemExit("mis-staging self-test FAILED: probe opened a mis-staged tree")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_shas = None if args.skip_pairs else stage_pairs(args.data_dir, args.pairs_revision)
    anchor_rev, anchor_files = stage_anchor_store(
        args.anchor_dir,
        args.anchor_shards,
        prefix=args.anchor_prefix,
        revision=args.anchor_revision,
    )
    prov = provenance_coherence(args.pairs_revision, anchor_files, store_rev=args.anchor_revision)
    n = consumer_open_probe(args.anchor_dir, args.expect_n)
    if args.self_test:
        self_test_misstaged(args.anchor_dir)
    manifest = {
        "metadata": common.metadata(SCRIPT, common.BUILD_SEED, n),
        "base_model_id": BASE_MODEL_ID,
        "instruct_model_id": common.MODEL_ID,
        "pairs_revision": args.pairs_revision,
        "pair_sha256": pair_shas,
        "pairs_staged": not args.skip_pairs,
        "anchor_prefix": args.anchor_prefix,
        "anchor_store_revision": anchor_rev,
        "anchor_store_files": anchor_files,
        "anchor_shards": args.anchor_shards,
        "anchor_rows": n,
        "provenance_coherence": prov,
    }
    common.write_json(args.out_dir / args.manifest_name, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
