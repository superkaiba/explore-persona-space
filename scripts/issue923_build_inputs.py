#!/usr/bin/env python3
"""Issue #923 Phase 0 — build the committed input files (VM, CPU, ~30 min).

Plan §4.3 Phase 0. Deterministic, seed 42. Outputs (committed to the issue
branch under ``data/issue923/`` — whitelisted in .gitignore):

- ``probes_uc_ext.json``   — +96 fresh UltraChat probes, 2 length-matched per
  Betley target (the strictly-1:1 ``greedy_length_match`` is fed the Betley
  length-target list TWICE — plan §4.3 adaptation note), disjoint from the
  existing 48 UC probes / Betley pool / ICL demos / battery text.
- ``probes_dolly.json``    — 48 Dolly-15k instruction-only OOD queries
  (empty ``context`` field, 5 named categories, #594 filters, greedy
  length-matched to the 48 Betley lengths).
- ``probes_betley.json``   — the 48-probe Betley pool materialized + hash-
  asserted vs the battery meta (pins the pod-side pool without a re-fetch).
- ``fold_assignments.json``— 4 stratified query folds per genre (UC 144 → 4x36,
  Betley 48 → 4x12; stratified by matched-Betley-length decile).
- ``store_pins.json``      — the HF dataset-repo revision sha the span files are
  pinned to (``hf_hub_download(..., revision=<sha>)``), + the two stores'
  ``answer_spans/index.json`` probe lists asserted against the pools.

Usage::

    uv run python scripts/issue923_build_inputs.py            # full build
    uv run python scripts/issue923_build_inputs.py --smoke \\
        --out-dir /tmp/issue-923-smoke/data                   # tiny slice
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue594_build_probes_ultrachat as bp  # noqa: E402
from issue404_common import (  # noqa: E402
    fetch_betley_main_8,
    fetch_preregistered_probes,
    reproducibility_metadata,
)
from issue594_common import load_battery  # noqa: E402
from issue923_common import (  # noqa: E402
    DATA_DIR,
    GENRES,
    HF_DATA_REPO,
    N_QUERY_FOLDS,
    SEED,
    STORE_PREFIXES,
    assign_stratified_folds,
    dump_json,
    hf_revision,
    load_json,
    texts_hash,
)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue923_build_inputs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DOLLY_DATASET = "databricks/databricks-dolly-15k"
DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "classification", "creative_writing"}
N_UC_EXT = 96
N_DOLLY = 48


def _battery_reference_texts(instances: list[dict]) -> list[str]:
    """Battery text pool for disjointness filtering (system prompts + prefixes)."""
    texts: list[str] = []
    for inst in instances:
        if inst["system_prompt"]:
            texts.append(inst["system_prompt"])
        for m in inst["prefix_messages"]:
            texts.append(m["content"])
    return [t for t in texts if t.strip()]


def _drop_colliding(candidates: list[dict], pools: dict[str, list[str]]) -> tuple[list[dict], dict]:
    """DROP-filter candidates colliding (casefolded eq/contains) with any pool.

    Unlike ``assert_disjoint`` (which raises), collisions with pools that share
    the candidate SOURCE distribution (the existing 48 UC probes came from the
    same 20k-row stream) are expected and are dropped + counted, not fatal.
    """
    kept: list[dict] = []
    drops = {name: 0 for name in pools}
    pool_cf = {name: [t.casefold() for t in texts] for name, texts in pools.items()}
    for c in candidates:
        cf = c["text"].casefold()
        hit = None
        for name, refs in pool_cf.items():
            if any(cf == r or r in cf or cf in r for r in refs):
                hit = name
                break
        if hit is None:
            kept.append(c)
        else:
            drops[hit] += 1
    return kept, drops


def build_uc_ext(tokenizer, betley: list[str], references: dict, drop_pools: dict, n_ext: int):
    """+n_ext fresh UltraChat probes, 2 per Betley length target (plan §4.3)."""
    candidates, drop_counts = bp.collect_candidates(tokenizer, references)
    candidates, extra_drops = _drop_colliding(candidates, drop_pools)
    logger.info("UC-ext candidates after collision drops: %d (%s)", len(candidates), extra_drops)
    betley_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in betley]
    repeats = max(1, n_ext // len(betley_lens)) if n_ext >= len(betley_lens) else 1
    targets = (betley_lens * repeats)[:n_ext] if n_ext >= len(betley_lens) else betley_lens[:n_ext]
    matches = bp.greedy_length_match(candidates, targets)
    assert len(matches) == n_ext, (len(matches), n_ext)
    ids = {m["prompt_id"] for m in matches.values()}
    assert len(ids) == n_ext, "duplicate prompt_id in UC-ext matches (without-replacement broken)"
    probes = [
        {
            "text": matches[bi]["text"],
            "prompt_id": matches[bi]["prompt_id"],
            "source_row_index": matches[bi]["source_row_index"],
            "token_len": matches[bi]["token_len"],
            "matched_betley_index": bi % len(betley_lens),
            "matched_betley_len": targets[bi],
        }
        for bi in range(n_ext)
    ]
    return probes, {"drop_counts": drop_counts, "collision_drops": extra_drops}


def build_dolly(tokenizer, betley_lens: list[int], drop_pools: dict, n_out: int):
    """48 Dolly-15k instruction-only OOD queries, length-matched (plan §4.3)."""
    from datasets import load_dataset

    ds = load_dataset(DOLLY_DATASET, split="train", revision=hf_revision("datasets", DOLLY_DATASET))
    counts = {
        "rows": 0,
        "nonempty_context": 0,
        "category": 0,
        "ascii_or_words": 0,
        "token_len": 0,
        "dup": 0,
    }
    seen: set[str] = set()
    candidates: list[dict] = []
    for i, row in enumerate(ds):
        counts["rows"] += 1
        if (row.get("context") or "").strip():
            counts["nonempty_context"] += 1
            continue
        if row.get("category") not in DOLLY_CATEGORIES:
            counts["category"] += 1
            continue
        text = (row.get("instruction") or "").strip()
        if not text:
            counts["ascii_or_words"] += 1
            continue
        ascii_ratio = sum(ord(c) < 128 for c in text) / len(text)
        if ascii_ratio < bp.ASCII_MIN_RATIO or len(text.split()) < bp.MIN_WORDS:
            counts["ascii_or_words"] += 1
            continue
        tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        if not (bp.TOK_MIN <= tok_len <= bp.TOK_MAX):
            counts["token_len"] += 1
            continue
        cf = text.casefold()
        if cf in seen:
            counts["dup"] += 1
            continue
        seen.add(cf)
        candidates.append(
            {"text": text, "prompt_id": f"dolly_{i}", "source_row_index": i, "token_len": tok_len}
        )
    logger.info("Dolly candidates: %d after filters (%s)", len(candidates), counts)
    assert len(candidates) >= n_out, f"only {len(candidates)} Dolly candidates (< {n_out})"
    candidates, extra_drops = _drop_colliding(candidates, drop_pools)
    matches = bp.greedy_length_match(candidates, betley_lens[:n_out])
    probes = [
        {
            "text": matches[bi]["text"],
            "prompt_id": matches[bi]["prompt_id"],
            "source_row_index": matches[bi]["source_row_index"],
            "token_len": matches[bi]["token_len"],
            "matched_betley_index": bi,
            "matched_betley_len": betley_lens[bi],
        }
        for bi in range(n_out)
    ]
    return probes, {"filter_counts": counts, "collision_drops": extra_drops}


def build_store_pins(betley: list[str], uc48: list[str], battery_ids: list[str]) -> dict:
    """Pin the HF data-repo revision + assert the stores' probe lists (plan §10f)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.repo_info(HF_DATA_REPO, repo_type="dataset")
    revision = info.sha
    logger.info("Pinned %s revision: %s", HF_DATA_REPO, revision)
    pools = {"betley": betley, "uc": uc48}
    pins: dict = {"repo": HF_DATA_REPO, "revision": revision, "stores": {}}
    for genre in GENRES:
        prefix = STORE_PREFIXES[genre]
        idx_path = hf_hub_download(
            HF_DATA_REPO,
            f"{prefix}/answer_spans/index.json",
            repo_type="dataset",
            revision=revision,
        )
        index = load_json(idx_path)
        ctx_ids = index["context_ids"]
        assert set(ctx_ids) == set(battery_ids), (
            f"{genre} store context ids != battery ids (missing {set(battery_ids) - set(ctx_ids)})"
        )
        pool = pools[genre]
        for cid, probes in index["probes_by_context"].items():
            assert probes == pool, (
                f"{genre} store probes for {cid} do not match the pinned pool "
                f"(order/content drift — Assumption 13 broken)"
            )
        man_path = hf_hub_download(
            HF_DATA_REPO, f"{prefix}/store_manifest.json", repo_type="dataset", revision=revision
        )
        manifest = load_json(man_path)
        pins["stores"][genre] = {
            "prefix": prefix,
            "context_ids": ctx_ids,
            "n_probes": len(pool),
            "probe_pool_hash": texts_hash(pool),
            "index_path": f"{prefix}/answer_spans/index.json",
            "manifest_keys": sorted(manifest.keys())[:20],
        }
    return pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--smoke", action="store_true", help="tiny slice, scratch out-dir")
    parser.add_argument("--skip-pins", action="store_true", help="skip the HF store-pin step")
    parser.add_argument("--n-ext", type=int, default=None)
    parser.add_argument("--n-dolly", type=int, default=None)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    if args.smoke and out_dir == DATA_DIR:
        out_dir = Path("/tmp/issue-923-smoke/data")
        logger.info("--smoke: redirecting outputs to scratch dir %s", out_dir)
    n_ext = args.n_ext or (4 if args.smoke else N_UC_EXT)
    n_dolly = args.n_dolly or (4 if args.smoke else N_DOLLY)
    if args.smoke:
        # Smoke slice: shrink the streamed candidate pool (module knob — the
        # SAME collect_candidates path, smaller N; plan §4.4 smoke parity).
        bp.CANDIDATE_ROWS = 3000

    payload, instances = load_battery()
    battery_ids = [i["id"] for i in instances]
    demo_questions = sorted(
        {
            m["content"]
            for inst in instances
            if inst["family"] == "icl"
            for m in inst["prefix_messages"]
            if m["role"] == "user"
        }
    )
    main8 = set(fetch_betley_main_8())
    betley = fetch_preregistered_probes(n=200, exclude=main8)
    assert len(betley) == 48, len(betley)
    betley_hash = texts_hash(betley)
    assert betley_hash == payload["meta"]["probe_pool_hash"], (
        "Betley pool drifted vs battery meta (order/content)"
    )
    uc48_blob = load_json(PROJECT_ROOT / "data" / "issue594" / "probes_ultrachat.json")
    uc48 = [r["text"] for r in uc48_blob["probes"]]
    battery_texts = _battery_reference_texts(instances)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        bp.DEFAULT_MODEL, revision=hf_revision("models", bp.DEFAULT_MODEL)
    )
    betley_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in betley]

    # In-stream raising references match the parent builder (betley + ICL demos);
    # pools sharing the UltraChat source distribution are DROP-filtered instead.
    references = {"betley_probes": betley, "icl_demo_questions": demo_questions}
    drop_pools = {"uc48": uc48, "battery_text": battery_texts}

    uc_ext, ucext_meta = build_uc_ext(tokenizer, betley, references, drop_pools, n_ext)
    # Belt-and-braces: the SELECTED probes must be disjoint from everything.
    for p in uc_ext:
        bp.assert_disjoint(
            p["text"],
            {**references, **{k: v for k, v in drop_pools.items()}},
        )

    dolly_drop_pools = {**drop_pools, "uc_ext": [p["text"] for p in uc_ext]}
    dolly, dolly_meta = build_dolly(tokenizer, betley_lens, dolly_drop_pools, n_dolly)
    for p in dolly:
        bp.assert_disjoint(p["text"], {**references, **dolly_drop_pools})

    # Fold assignments (UC 144 = 48 store + n_ext ext; Betley 48) — stratified
    # by matched-Betley-length decile, seeded, exactly-equal fold sizes.
    uc_matched_lens = [r["matched_betley_len"] for r in uc48_blob["probes"]] + [
        p["matched_betley_len"] for p in uc_ext
    ]
    uc_folds = assign_stratified_folds(uc_matched_lens, N_QUERY_FOLDS, SEED)
    betley_folds = assign_stratified_folds(betley_lens, N_QUERY_FOLDS, SEED + 1)
    dolly_texts = [p["text"] for p in dolly]

    meta = reproducibility_metadata({"script": "issue923_build_inputs", "smoke": args.smoke})
    dump_json(
        {
            "meta": {
                "seed": SEED,
                "n_ext": n_ext,
                "probe_pool_hash": texts_hash([p["text"] for p in uc_ext]),
                "betley_pool_hash": betley_hash,
                "builder": "issue594_build_probes_ultrachat.collect_candidates+greedy_length_match",
                "doubling": "betley length-target list repeated (2 matches per target)",
                **ucext_meta,
                "metadata": meta,
            },
            "probes": uc_ext,
        },
        out_dir / "probes_uc_ext.json",
    )
    dump_json(
        {
            "meta": {
                "dataset": DOLLY_DATASET,
                "categories": sorted(DOLLY_CATEGORIES),
                "seed": SEED,
                "probe_pool_hash": texts_hash(dolly_texts),
                **dolly_meta,
                "metadata": meta,
            },
            "probes": dolly,
        },
        out_dir / "probes_dolly.json",
    )
    dump_json(
        {
            "meta": {
                "probe_pool_hash": betley_hash,
                "source": "fetch_preregistered_probes",
                "metadata": meta,
            },
            "probes": [
                {"text": t, "token_len": length}
                for t, length in zip(betley, betley_lens, strict=True)
            ],
        },
        out_dir / "probes_betley.json",
    )
    dump_json(
        {
            "meta": {
                "seed": SEED,
                "n_query_folds": N_QUERY_FOLDS,
                "metadata": meta,
                "stratification": "matched-Betley-length decile, seeded round-robin",
            },
            "n_queries": {"uc": 48 + n_ext, "betley": 48, "dolly": n_dolly},
            "query_folds": {"uc": uc_folds, "betley": betley_folds},
            "uc_query_order": "probes_ultrachat.json order (48) + probes_uc_ext.json order",
            "families": {i["id"]: i["family"] for i in instances},
        },
        out_dir / "fold_assignments.json",
    )
    if not args.skip_pins:
        pins = build_store_pins(betley, uc48, battery_ids)
        pins["metadata"] = meta
        dump_json(pins, out_dir / "store_pins.json")
    logger.info("Phase 0 outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
