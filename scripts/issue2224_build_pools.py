#!/usr/bin/env python3
"""Issue #2224 P0a: build the real-corpus working pools (plan v3 §4 P0a).

Streams the paper's two corpora with the arXiv 2507.21509 §6.3 / App.
real-world-datasets preprocessing — trim every conversation to a single
user->assistant exchange, drop prompts > 512 tokens (Qwen tokenizer,
``add_special_tokens=False``), then draw a SEEDED random working pool per
corpus:

- ``lmsys/lmsys-chat-1m``            (GATED — HF token via .env; the raw pole)
- ``HuggingFaceH4/ultrachat_200k``   config=default split=train_sft (filtered pole)

REUSES the streaming recipes of ``scripts/issue617_build_wildchat_slice.py``
+ ``scripts/issue594_build_probes_ultrachat.py`` (row schema, first-exchange
validation, per-filter rejection counters) and the #1092
``_stream_with_cache`` shape (``scripts/issue1092_build_corpus.py``):
chunk-checkpointed candidates JSONL + fingerprint meta sidecar, resume via
``IterableDataset.skip`` keyed on the dataset revision + every filter/recipe
constant (code-style.md external-stream presumption).

No language / toxicity / dedup FILTERS beyond the paper's preprocessing —
LMSYS is deliberately raw (plan §4 data tier: the point of the corpus).
Duplicate prompts and special-token-bearing prompts are COUNTED as
diagnostics, never dropped.

Content hygiene: raw corpus text is NEVER printed or logged — counts, token
stats and hashes only (real-world-corpus digest-only rule).

Outputs per corpus under ``--out-dir`` (default ``data/issue_2224/pools/``):
- ``<corpus>.jsonl``                 the seeded working pool (plan §9 phase_outputs)
- ``<corpus>_candidates.jsonl``      resumable stream cache
- ``<corpus>_candidates.meta.json``  fingerprint sidecar
- ``yield_report_<corpus>.json``     realized-yield report (plan §12 A10 probe)
- ``yield_report.json``              combined per-corpus summary

Usage::

    uv run python scripts/issue2224_build_pools.py                       # both corpora, 50k pools
    uv run python scripts/issue2224_build_pools.py --corpus ultrachat --smoke
    uv run python scripts/issue2224_build_pools.py --import-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/torch imports: shared-VM thread caps + HF token (#847)

import numpy as np  # noqa: E402
from issue2224_common import (  # noqa: E402
    ISSUE,
    POOL_SCHEMA_VERSION,
    POOLS_DIR_DEFAULT,
    SMOKE_ROOT_DEFAULT,
    atomic_write_json,
    atomic_write_jsonl,
    count_jsonl_lines,
    load_jsonl,
    repro_meta,
    sha256_file,
    token_stats,
    truncate_jsonl,
)
from issue778_lib import MODEL_NAME  # noqa: E402  (Qwen/Qwen2.5-7B-Instruct)

logger = logging.getLogger("issue2224_pools")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_PROMPT_TOKENS = 512  # paper preprocessing: drop prompts > 512 tokens

CORPORA: dict[str, dict] = {
    "lmsys": {
        "repo": "lmsys/lmsys-chat-1m",
        "config": None,
        "split": "train",
        "conv_field": "conversation",
    },
    "ultrachat": {
        # Reuses issue594_build_probes_ultrachat.py's DATASET/CONFIG/SPLIT.
        "repo": "HuggingFaceH4/ultrachat_200k",
        "config": "default",
        "split": "train_sft",
        "conv_field": "messages",
    },
}

_TOKENIZER_CACHE: dict[str, object] = {}


def get_tokenizer(model_id: str):
    """Module-cached tokenizer (never ``from_pretrained`` per row — 429 gotcha)."""
    if model_id not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZER_CACHE[model_id]


def first_exchange(conv) -> tuple[str, str] | None:
    """First user->assistant exchange as (prompt, response), else None.

    Mirrors ``issue594_build_battery._conv_messages(row, 2)``: the first two
    messages must be a user turn then an assistant turn, both non-empty
    strings after strip. The paper's "trim conversations to length 2".
    """
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    out: list[str] = []
    for m, role in ((conv[0], "user"), (conv[1], "assistant")):
        if not isinstance(m, dict) or m.get("role") != role:
            return None
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        out.append(content.strip())
    return out[0], out[1]


def row_extras(corpus: str, row: dict) -> dict:
    """Per-corpus provenance extras carried into the pool row (no text)."""
    if corpus == "lmsys":
        return {
            "model": row.get("model"),
            "language": row.get("language"),
            "redacted": bool(row.get("redacted")),
            "turn": row.get("turn"),
        }
    return {"prompt_id": row.get("prompt_id")}


def resolve_revision(repo: str) -> str:
    """Pin the dataset revision (fingerprint key + coherent stream)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"could not resolve revision sha for {repo}")
    return str(sha)


def stream_fingerprint(corpus: str, cfg: dict, revision: str, args, collect_cap: int) -> dict:
    """Exact-match resume fingerprint (dataset revision + every filter constant)."""
    return {
        "schema": POOL_SCHEMA_VERSION,
        "corpus": corpus,
        "repo": cfg["repo"],
        "dataset_config": cfg["config"],
        "split": cfg["split"],
        "revision": revision,
        "tokenizer": args.tokenizer,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "trim": "first_user_assistant_exchange",
        "scan_cap": args.scan_cap,
        "collect_cap": collect_cap,
    }


def stream_candidates(
    corpus: str,
    cfg: dict,
    args,
    out_dir: Path,
    collect_cap: int,
) -> tuple[Path, dict]:
    """Stream + filter one corpus into a resumable candidates JSONL.

    Checkpoint-per-chunk (code-style.md external-stream presumption, #1092):
    kept rows append to ``<corpus>_candidates.jsonl``; every
    ``--checkpoint-every`` scanned rows the meta sidecar is atomically
    rewritten with the scan position + counters. Resume: exact fingerprint
    match -> ``complete`` loads the cache; ``partial`` truncates the cache to
    the checkpointed kept-count and re-streams via ``ds.skip(scanned)``.
    """
    cand_path = out_dir / f"{corpus}_candidates.jsonl"
    meta_path = out_dir / f"{corpus}_candidates.meta.json"
    tok = get_tokenizer(args.tokenizer)
    special_ids = set(tok.all_special_ids)

    revision = resolve_revision(cfg["repo"])
    fp = stream_fingerprint(corpus, cfg, revision, args, collect_cap)

    counts = {"scanned": 0, "kept": 0}
    rejects = {"missing_conversation": 0, "bad_first_exchange": 0, "prompt_too_long": 0}
    diagnostics = {"duplicate_prompt_casefold": 0, "prompt_has_special_token": 0}
    seen_prompt_hashes: set[str] = set()
    resume = False

    if meta_path.exists() and not args.force:
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            if meta.get("status") == "complete":
                logger.info(
                    "[pool-stream %s] fingerprint match, cache COMPLETE (%d kept) — skip stream",
                    corpus,
                    meta["counts"]["kept"],
                )
                return cand_path, meta
            n_on_disk = count_jsonl_lines(cand_path)
            n_meta = int(meta["counts"]["kept"])
            if n_on_disk < n_meta:
                # The fsync ordering writes rows BEFORE the meta checkpoint, so
                # fewer rows than meta records is external file damage — never
                # silently under-fill the pool with an inflated kept count.
                raise RuntimeError(
                    f"[pool-stream {corpus}] partial cache has FEWER rows on disk "
                    f"({n_on_disk}) than the meta checkpoint records ({n_meta}) — "
                    f"external file damage; delete the cache dir or pass --force"
                )
            if n_on_disk != n_meta:
                logger.warning(
                    "[pool-stream %s] partial cache has %d rows vs meta %d — truncating",
                    corpus,
                    n_on_disk,
                    n_meta,
                )
                truncate_jsonl(cand_path, n_meta)
            counts = dict(meta["counts"])
            rejects = dict(meta["rejects"])
            diagnostics = dict(meta["diagnostics"])
            resume = True
            # Rebuild the dedup-diagnostic set from the kept rows (hashes only).
            import hashlib as _hl

            for r in load_jsonl(cand_path):
                seen_prompt_hashes.add(_hl.sha1(r["prompt"].casefold().encode()).hexdigest())
            logger.info(
                "[pool-stream %s] RESUME from scanned=%d kept=%d",
                corpus,
                counts["scanned"],
                counts["kept"],
            )
        else:
            mismatched = sorted(
                k
                for k in set(fp) | set(meta.get("fingerprint") or {})
                if (meta.get("fingerprint") or {}).get(k) != fp.get(k)
            )
            logger.warning(
                "[pool-stream %s] fingerprint MISMATCH on keys %s — re-streaming fresh",
                corpus,
                mismatched,
            )
            cand_path.unlink(missing_ok=True)

    from datasets import load_dataset

    ds = load_dataset(
        cfg["repo"],
        cfg["config"],
        split=cfg["split"],
        streaming=True,
        revision=revision,
    )
    if resume and counts["scanned"]:
        ds = ds.skip(counts["scanned"])

    import hashlib as _hl

    def _checkpoint(f, status: str) -> None:
        f.flush()
        import os as _os

        _os.fsync(f.fileno())
        atomic_write_json(
            {
                "fingerprint": fp,
                "status": status,
                "counts": counts,
                "rejects": rejects,
                "diagnostics": diagnostics,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            meta_path,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base_idx = counts["scanned"]
    with open(cand_path, "a" if resume else "w") as f:
        i = 0
        for row in ds:
            if counts["scanned"] >= args.scan_cap or counts["kept"] >= collect_cap:
                break
            source_row_index = base_idx + i
            i += 1
            counts["scanned"] += 1
            conv = row.get(cfg["conv_field"])
            if not conv:
                rejects["missing_conversation"] += 1
            else:
                fe = first_exchange(conv)
                if fe is None:
                    rejects["bad_first_exchange"] += 1
                else:
                    prompt, response = fe
                    prompt_ids = tok.encode(prompt, add_special_tokens=False)
                    if len(prompt_ids) > MAX_PROMPT_TOKENS:
                        rejects["prompt_too_long"] += 1
                    else:
                        # Diagnostics only (counted, never filtered — plan §4
                        # names no dedup / detox beyond the paper recipe).
                        ph = _hl.sha1(prompt.casefold().encode()).hexdigest()
                        if ph in seen_prompt_hashes:
                            diagnostics["duplicate_prompt_casefold"] += 1
                        seen_prompt_hashes.add(ph)
                        if special_ids.intersection(prompt_ids):
                            diagnostics["prompt_has_special_token"] += 1
                        rec = {
                            "sample_id": f"{corpus}_{source_row_index:07d}",
                            "corpus": corpus,
                            "source_row_index": source_row_index,
                            "prompt": prompt,
                            "response": response,
                            "prompt_tokens": len(prompt_ids),
                            "response_tokens": len(tok.encode(response, add_special_tokens=False)),
                            "extras": row_extras(corpus, row),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["kept"] += 1
            if counts["scanned"] % args.checkpoint_every == 0:
                _checkpoint(f, "partial")
                logger.info(
                    "[pool-stream %s] scanned=%d kept=%d rejects=%s elapsed=%.0fs",
                    corpus,
                    counts["scanned"],
                    counts["kept"],
                    json.dumps(rejects),
                    time.time() - t0,
                )
        stream_exhausted = counts["scanned"] < args.scan_cap and counts["kept"] < collect_cap
        _checkpoint(f, "complete")
    meta = json.loads(meta_path.read_text())
    meta["stream_exhausted"] = stream_exhausted
    atomic_write_json(meta, meta_path)
    logger.info(
        "[pool-stream %s] DONE scanned=%d kept=%d rejects=%s exhausted=%s elapsed=%.0fs",
        corpus,
        counts["scanned"],
        counts["kept"],
        json.dumps(rejects),
        stream_exhausted,
        time.time() - t0,
    )
    return cand_path, meta


def build_pool(
    corpus: str,
    cand_path: Path,
    meta: dict,
    pool_size: int,
    seed: int,
    out_dir: Path,
) -> dict:
    """Seeded random sample of the working pool from the candidates cache."""
    candidates = load_jsonl(cand_path)
    n = len(candidates)
    if n < pool_size:
        raise RuntimeError(
            f"[{corpus}] only {n} usable candidates after filters (pool target {pool_size}). "
            f"Realized yield: scanned={meta['counts']['scanned']} rejects={meta['rejects']}. "
            f"Raise --scan-cap / --collect-cap (plan §12 A10) and re-run (resume keeps the "
            f"cache only under an identical fingerprint — the caps are fingerprint keys)."
        )
    if n < 20_000 and pool_size >= 20_000:
        # Plan §3 kill-criterion 2 surface (report, not a hard stop at this stage).
        logger.warning(
            "[%s] kept candidates %d < 20k — selection tails may be thin (plan §12 A10)",
            corpus,
            n,
        )
    rng = np.random.default_rng([ISSUE, seed])
    idx = rng.permutation(n)[:pool_size]
    rows = sorted((candidates[int(i)] for i in idx), key=lambda r: r["source_row_index"])
    pool_path = out_dir / f"{corpus}.jsonl"
    atomic_write_jsonl(rows, pool_path)

    report = {
        "corpus": corpus,
        "fingerprint": meta["fingerprint"],
        "scanned": meta["counts"]["scanned"],
        "kept_candidates": n,
        "rejects": meta["rejects"],
        "diagnostics": meta["diagnostics"],
        "stream_exhausted": meta.get("stream_exhausted"),
        "usable_yield_rate": n / max(1, meta["counts"]["scanned"]),
        "pool_size": len(rows),
        "pool_seed": seed,
        "pool_path": str(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "prompt_token_stats": token_stats([r["prompt_tokens"] for r in rows]),
        "response_token_stats": token_stats([r["response_tokens"] for r in rows]),
        "a10_note": (
            "plan §12 A10: realized usable single-turn ≤512-tok candidates per corpus; "
            "< ~20k usable -> thin selection tails, raise the stream cap"
        ),
        "meta": repro_meta("issue2224_build_pools"),
    }
    atomic_write_json(report, out_dir / f"yield_report_{corpus}.json")
    logger.info(
        "[pool %s] wrote %d rows -> %s (sha256 %s...); yield %d/%d scanned",
        corpus,
        len(rows),
        pool_path,
        report["pool_sha256"][:12],
        n,
        meta["counts"]["scanned"],
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #2224 P0a: stream + preprocess LMSYS/UltraChat working pools."
    )
    parser.add_argument("--corpus", choices=["lmsys", "ultrachat", "both"], default="both")
    parser.add_argument(
        "--pool-size", type=int, default=None, help="working pool size (default 50000; smoke 20)"
    )
    parser.add_argument(
        "--scan-cap",
        type=int,
        default=None,
        help="max rows scanned per corpus (default 200000; smoke 3000)",
    )
    parser.add_argument(
        "--collect-cap",
        type=int,
        default=None,
        help="max kept candidates per corpus (default 3x pool size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=MODEL_NAME, help="tokenizer for the 512-tok filter")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"output dir (default {POOLS_DIR_DEFAULT}; smoke {SMOKE_ROOT_DEFAULT}/pools)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny self-check defaults (pool 20, scan-cap 3000, scratch out-dir) — same code path",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore the fingerprint cache and re-stream"
    )
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + argparse-attribute completeness check, then exit 0",
    )
    args = parser.parse_args()

    if args.import_check:
        import importlib

        for mod in ("numpy", "datasets", "transformers", "huggingface_hub"):
            importlib.import_module(mod)
        from datasets import load_dataset  # noqa: F401
        from huggingface_hub import HfApi  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_build_pools")
        return 0

    pool_size = args.pool_size if args.pool_size is not None else (20 if args.smoke else 50_000)
    args.scan_cap = (
        args.scan_cap if args.scan_cap is not None else (3_000 if args.smoke else 200_000)
    )
    collect_cap = args.collect_cap if args.collect_cap is not None else 3 * pool_size
    args.checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else (500 if args.smoke else 10_000)
    )
    out_dir = args.out_dir or (SMOKE_ROOT_DEFAULT / "pools" if args.smoke else POOLS_DIR_DEFAULT)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = ["lmsys", "ultrachat"] if args.corpus == "both" else [args.corpus]
    summary: dict[str, dict] = {}
    for corpus in corpora:
        cfg = CORPORA[corpus]
        cand_path, meta = stream_candidates(corpus, cfg, args, out_dir, collect_cap)
        report = build_pool(corpus, cand_path, meta, pool_size, args.seed, out_dir)
        summary[corpus] = {
            k: report[k]
            for k in (
                "scanned",
                "kept_candidates",
                "rejects",
                "diagnostics",
                "pool_size",
                "pool_sha256",
                "usable_yield_rate",
            )
        }
    atomic_write_json(
        {"pools": summary, "meta": repro_meta("issue2224_build_pools")},
        out_dir / "yield_report.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
