#!/usr/bin/env python3
"""Issue #923 Phase 2 — stream-reduce the #658 answer-span stores to per-cell v̄.

Plan §4.3 Phase 2 (cpu-mid, concurrent with Phase 1). For each of the 2x50
``answer_spans/<ctx>.pt`` files (0.59-4.59 GB each, 340.67 GB total):
``hf_hub_download`` at the PINNED repo revision (``store_pins.json``) into a
local scratch dir → mean over the answer span per probe → ``(n_probes, Lc, H)``
fp16 → DELETE the local file (stream-reduce; peak disk ~= one span file — the
code-style "never materialize the grid" rule; the canonical
``_HfStreamSpanSource`` is an LRU cache around repeated reads — this phase
reads each file exactly once, so the download→reduce→delete loop is the same
pattern without the cache).

Schema asserts (plan §12 assumption 1, verified on the FIRST file before the
loop): keys ``{context_id, capture_layers, spans, probes}``; ``capture_layers
== [0..27]``; ``len(spans) == len(probes) == 48``; probes match the pinned
pool order (``probe_pool_hash``); a ``None`` span (empty answer) is dropped +
logged (store precedent), marked invalid in the output pack.

Output per genre: ``vbar_store_<genre>.pt`` pack (~0.5 GB fp16) with tensors
``{"vbar": (50*48, Lc, H) fp16, "valid": (50*48,) bool}`` + row keys — uploaded
to ``analysis_tensors/reduce/`` per genre the moment it completes
(checkpoint-per-phase).

Usage::

    uv run python scripts/issue923_reduce_spans.py --genres betley,uc
    # smoke (synthetic producer-schema span file through the SAME reduce loop):
    uv run python scripts/issue923_reduce_spans.py --smoke \\
        --out-dir /tmp/issue-923-smoke/reduce
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue923_common import (  # noqa: E402
    DATA_DIR,
    GENRES,
    HF_DATA_REPO,
    HF_PREFIX_923,
    STORE_PREFIXES,
    dump_json,
    load_json,
    load_pack,
    save_pack,
    texts_hash,
)

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue923_reduce")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def assert_span_schema(blob: dict, ctx_id: str, pool: list[str], n_layers_expected: int) -> None:
    """Fail-loud schema check against the producer contract (issue658 store)."""
    missing = {"context_id", "capture_layers", "spans", "probes"} - set(blob)
    assert not missing, f"{ctx_id}: span file missing keys {sorted(missing)}"
    assert blob["context_id"] == ctx_id, (blob["context_id"], ctx_id)
    assert blob["capture_layers"] == list(range(n_layers_expected)), (
        f"{ctx_id}: capture_layers {blob['capture_layers'][:5]}... != 0..{n_layers_expected - 1}"
    )
    assert len(blob["spans"]) == len(blob["probes"]) == len(pool), (
        ctx_id,
        len(blob["spans"]),
        len(blob["probes"]),
        len(pool),
    )
    assert blob["probes"] == pool, (
        f"{ctx_id}: span-file probe order/content != pinned pool (Assumption 13)"
    )


def reduce_span_file(blob: dict, hidden: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(vbar (n_probes, Lc, H) fp16, valid (n_probes,) bool, n_dropped).

    v̄ per cell = mean over the answer-token span (dim 1 of each (Lc, S, H)
    fp16 span), computed in fp32 then stored fp16 — the same reduction the
    capture script applies to fresh cells. ``None`` spans → zero row + invalid.
    """
    spans = blob["spans"]
    lc = len(blob["capture_layers"])
    n = len(spans)
    vbar = torch.zeros(n, lc, hidden, dtype=torch.float16)
    valid = torch.zeros(n, dtype=torch.bool)
    n_dropped = 0
    for i, span in enumerate(spans):
        if span is None:
            n_dropped += 1
            continue
        assert span.ndim == 3 and span.shape[0] == lc and span.shape[2] == hidden, (
            f"span {i}: shape {tuple(span.shape)} != (Lc, S, {hidden})"
        )
        vbar[i] = span.float().mean(dim=1).to(torch.float16)
        valid[i] = True
    return vbar, valid, n_dropped


def reduce_context_list(
    ctx_ids: list[str],
    fetch,
    pool: list[str],
    n_layers: int,
    hidden: int,
    partials_dir: Path,
    regime: dict,
    tag: str,
    interrupt_after: int | None = None,
) -> tuple[list[dict], list[torch.Tensor], list[torch.Tensor], int]:
    """Per-context stream-reduce with per-file atomic checkpoint + resume.

    ``fetch(cid)`` returns the local span-file path (HF download in production,
    synthetic file in the smoke); the file is DELETED after reduction
    (stream-delete, peak disk ~= one span file). Each context's reduced tensor
    is persisted atomically to ``partials_dir/<cid>.pt`` the moment it
    completes (checkpoint-per-phase — r1 blocker
    reduce-loop-no-per-file-checkpoint), so a crash at file 49 forfeits at most
    one in-flight download, never the genre. On restart a partial whose
    ``regime`` (store revision + probe_pool_hash) matches is loaded and the
    download SKIPPED; a mismatched partial is recomputed with a warning, never
    silently reused (#722-r3). ``interrupt_after`` (testing only) raises after
    N freshly-reduced contexts to exercise the resume path in the smoke.
    """
    partials_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    vbars: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    dropped = 0
    n_fresh = 0
    t0 = time.time()
    for k, cid in enumerate(ctx_ids):
        part_path = partials_dir / f"{cid}.pt"
        vbar = None
        if part_path.exists():
            pt, pm = load_pack(part_path)
            if pm.get("regime") == regime:
                vbar, valid, n_drop = pt["vbar"], pt["valid"], int(pm["n_dropped"])
                logger.info(
                    "[reduce %s] %d/%d %s RESUMED from partial", tag, k + 1, len(ctx_ids), cid
                )
            else:
                logger.warning(
                    "[reduce %s] partial %s regime mismatch (%s != %s) — recomputing",
                    tag,
                    part_path.name,
                    pm.get("regime"),
                    regime,
                )
        if vbar is None:
            local = fetch(cid)
            blob = torch.load(local, map_location="cpu", weights_only=False)
            assert_span_schema(blob, cid, pool, n_layers_expected=n_layers)
            vbar, valid, n_drop = reduce_span_file(blob, hidden=hidden)
            save_pack(
                part_path,
                {"vbar": vbar, "valid": valid},
                {"ctx_id": cid, "regime": regime, "n_dropped": n_drop},
            )
            Path(local).unlink()  # stream-delete: peak disk ~= one span file
            n_fresh += 1
            logger.info(
                "[reduce %s] %d/%d %s (dropped=%d, %.1fs elapsed)",
                tag,
                k + 1,
                len(ctx_ids),
                cid,
                n_drop,
                time.time() - t0,
            )
            if interrupt_after is not None and n_fresh >= interrupt_after:
                raise RuntimeError(
                    f"simulated interrupt after {n_fresh} fresh contexts (--interrupt-after)"
                )
        dropped += n_drop
        vbars.append(vbar)
        valids.append(valid)
        rows.extend({"ctx_id": cid, "q_idx": qi} for qi in range(len(pool)))
    return rows, vbars, valids, dropped


def _write_synthetic_smoke_store(scratch: Path, pool: list[str], ctx_ids: list[str]) -> Path:
    """Producer-schema synthetic span files for the smoke (same reduce loop)."""
    store_dir = scratch / "synthetic_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for cid in ctx_ids:
        spans = []
        for i in range(len(pool)):
            if i == len(pool) - 1:
                spans.append(None)  # exercise the None-span drop branch
            else:
                spans.append(torch.randn(4, 3 + i, 8, dtype=torch.float16))
        torch.save(
            {
                "context_id": cid,
                "capture_layers": list(range(4)),
                "spans": spans,
                "probes": pool,
            },
            store_dir / f"{cid}.pt",
        )
    return store_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #923 Phase 2 span stream-reduce")
    parser.add_argument("--genres", default="betley,uc")
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_923" / "reduce"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--scratch", type=Path, default=None, help="download scratch dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument(
        "--interrupt-after",
        type=int,
        default=None,
        help="TESTING: raise after N freshly-reduced contexts (exercises resume)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch or (out_dir / "scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    meta = reproducibility_metadata({"script": "issue923_reduce_spans", "smoke": args.smoke})

    if args.smoke:
        print("[phase=reduce_smoke]", flush=True)
        pool = [f"synthetic probe {i}?" for i in range(4)]
        ctx_ids = ["smoke_ctx_a", "smoke_ctx_b"]
        store_dir = _write_synthetic_smoke_store(scratch, pool, ctx_ids)
        # The SAME checkpoint/resume loop as production (fetch = the synthetic
        # file; the interrupt/resume smoke exercises the partial path).
        rows, vbars, valids, dropped = reduce_context_list(
            ctx_ids,
            fetch=lambda cid: store_dir / f"{cid}.pt",
            pool=pool,
            n_layers=4,
            hidden=8,
            partials_dir=out_dir / "partials" / "smoke",
            regime={"revision": "synthetic-smoke", "probe_pool_hash": texts_hash(pool)},
            tag="smoke",
            interrupt_after=args.interrupt_after,
        )
        save_pack(
            out_dir / "vbar_store_smoke.pt",
            {"vbar": torch.cat(vbars), "valid": torch.cat(valids)},
            {"rows": rows, "genre": "smoke", "n_dropped": dropped, "metadata": meta},
        )
        shutil.rmtree(out_dir / "partials" / "smoke", ignore_errors=True)
        print(f"[reduce smoke] pack rows={len(rows)} dropped={dropped}", flush=True)
        print("[phase=done]", flush=True)
        return 0

    from huggingface_hub import hf_hub_download

    pins = load_json(args.data_dir / "store_pins.json")
    revision = pins["revision"]
    pools = {
        "betley": [r["text"] for r in load_json(args.data_dir / "probes_betley.json")["probes"]],
        "uc": [
            r["text"]
            for r in load_json(PROJECT_ROOT / "data/issue594/probes_ultrachat.json")["probes"]
        ],
    }
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    assert set(genres) <= set(GENRES), genres
    for genre in genres:
        print(f"[phase=reduce_{genre}]", flush=True)
        prefix = STORE_PREFIXES[genre]
        pool = pools[genre]
        pool_hash = texts_hash(pool)
        assert pool_hash == pins["stores"][genre]["probe_pool_hash"], (
            f"{genre}: pinned pool hash drifted vs store_pins.json"
        )
        ctx_ids = pins["stores"][genre]["context_ids"]

        def _fetch(cid: str, _prefix=prefix, _genre=genre) -> Path:
            return Path(
                hf_hub_download(
                    HF_DATA_REPO,
                    f"{_prefix}/answer_spans/{cid}.pt",
                    repo_type="dataset",
                    revision=revision,
                    local_dir=scratch / _genre,
                )
            )

        rows, vbars, valids, dropped = reduce_context_list(
            ctx_ids,
            fetch=_fetch,
            pool=pool,
            n_layers=args.n_layers,
            hidden=args.hidden,
            partials_dir=out_dir / "partials" / genre,
            regime={"revision": revision, "probe_pool_hash": pool_hash},
            tag=genre,
            interrupt_after=args.interrupt_after,
        )
        pack_path = out_dir / f"vbar_store_{genre}.pt"
        save_pack(
            pack_path,
            {"vbar": torch.cat(vbars), "valid": torch.cat(valids)},
            {
                "rows": rows,
                "genre": genre,
                "n_dropped": dropped,
                "revision": revision,
                "probe_pool_hash": pool_hash,
                "metadata": meta,
            },
        )
        dump_json(
            {"genre": genre, "n_rows": len(rows), "n_dropped": dropped, "metadata": meta},
            out_dir / f"reduce_summary_{genre}.json",
        )
        if not args.no_upload:
            hub._upload(
                pack_path,
                HF_DATA_REPO,
                "dataset",
                f"{HF_PREFIX_923}/analysis_tensors/reduce/{pack_path.name}",
                upload_as_file=True,
            )
            logger.info("[reduce %s] pack uploaded", genre)
        # Partials are subsumed by the saved (+uploaded) genre pack.
        shutil.rmtree(out_dir / "partials" / genre, ignore_errors=True)
        shutil.rmtree(scratch / genre, ignore_errors=True)
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
