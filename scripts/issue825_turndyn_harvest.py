#!/usr/bin/env python
"""#825 turn-dynamics-allturns-5000 P0 harvest (CPU, cpu-mid; plan v24 §4 P0).

Streams the FULL WildChat+LMSYS corpora at the pinned #1092 revisions through
the r8.1 filter recipe (reuse: ``issue1092_build_corpus._stream_with_cache`` —
checkpoint/resume verbatim, fingerprint-gated), then:

  1. n(k) table — realized full-stream count of kept conversations with >= k
     user turns (k = 1..30). Gate G-A reads it: ``K_real = max k in
     [ga_kmin, ga_kmax] with n(k) >= ga_target`` (default 8..15 / 5,000);
     K_real < ga_kmin => exit 3 BEFORE any GPU is provisioned (plan §7 G-A).
  2. deep_pool.jsonl — every kept conversation with >= ga_kmin user turns.
  3. panel_armR.jsonl — fixed panel of ``panel_n`` conversations with
     >= K_real user turns (seed 42). The SAME conversations populate every
     turn cell t=1..K_real.
  4. armG_seeds.jsonl — panel first-user-turns + ``spare_n`` spare deep-pool
     seeds (over-provisioned for length attrition), each with ONE persona
     brief (Track-M ``USER_BRIEFS`` rotation, fixed per conversation and
     shared across both subject models).
  5. gc_panel.jsonl — the round-10 conversation-id parity set, rebuilt from
     the PINNED #1092 prefix_store @ DATA_REPO_REV via the round-10 panel
     helpers, digest-asserted against the round-10 drop report (plan §7 G-C;
     assumption 11).
  6. harvest_report.json + HF upload of the panel/tables (text, unconditional).

Content hygiene: the corpus is REAL-USER text. Conversation text is never
printed/logged — only row counts, hashes, and depth tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from issue825_gen_conversations import USER_BRIEFS  # noqa: E402
from issue1092_build_corpus import (  # noqa: E402
    FILTER_RECIPE_VERSION,
    LMSYS_REPO,
    LMSYS_REV,
    WILDCHAT_REPO,
    WILDCHAT_REV,
    _stream_with_cache,
)

logger = logging.getLogger("i825_turndyn_harvest")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PANEL_PREFIX = "issue825_userbase_map/analysis_tensors/turn_dynamics/panel"
# Round-10 pins (plan §10): the G-C comparator inputs resolve at this revision.
DATA_REPO_REV = "9dd650deef3ca21daa9cc2e940e9563edc000ba3"
R10_STORE_PREFIX = "issue1092_realistic_crossing"
R10_RESULTS_JSON = REPO_ROOT / "eval_results/issue_825/onpolicy_turn_depth/results.json"

PANEL_SEED = 42
MAX_K_TABLE = 30
SHARD_MAX_BYTES = 9_000_000  # non-LFS text path: keep every JSONL shard < 9 MB


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl_sharded(rows: list[dict], out_dir: Path, stem: str) -> list[Path]:
    """ASCII-escaped JSONL, line-split into < 9 MB shards (non-LFS upload path).

    ensure_ascii=True (json default): raw U+2028/NEL in real-user text shreds
    splitlines-style consumers (gotchas.md). Returns the shard paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    si, size, f = 0, 0, None
    try:
        for row in rows:
            line = json.dumps(row) + "\n"
            if f is None or size + len(line) > SHARD_MAX_BYTES:
                if f is not None:
                    f.close()
                p = out_dir / f"{stem}_shard{si:03d}.jsonl"
                f = p.open("w", encoding="utf-8")
                paths.append(p)
                si, size = si + 1, 0
            f.write(line)
            size += len(line)
    finally:
        if f is not None:
            f.close()
    if not paths:  # zero rows still writes one empty shard (loadable downstream)
        p = out_dir / f"{stem}_shard000.jsonl"
        p.touch()
        paths.append(p)
    return paths


def read_jsonl(path: Path) -> list[dict]:
    """File-iteration JSONL reader (never read().splitlines() — gotchas.md)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_stem(out_dir: Path, stem: str) -> list[dict]:
    """Read all shards of a `_write_jsonl_sharded` stem, in shard order."""
    paths = sorted(out_dir.glob(f"{stem}_shard*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no shards for {out_dir}/{stem}")
    rows: list[dict] = []
    for p in paths:
        rows.extend(read_jsonl(p))
    return rows


def _nk_table(pool: list[dict], max_k: int = MAX_K_TABLE) -> dict[str, int]:
    """n(k) = number of kept conversations with >= k user turns."""
    depths = [int(r["n_user_turns"]) for r in pool]
    return {str(k): sum(1 for d in depths if d >= k) for k in range(1, max_k + 1)}


def gate_ga(nk: dict[str, int], *, target: int, kmin: int, kmax: int) -> dict:
    """Gate G-A (plan §7): adaptive K_real, fail-loud only below the kmin floor."""
    feasible = [k for k in range(kmin, kmax + 1) if nk.get(str(k), 0) >= target]
    k_real = max(feasible) if feasible else None
    verdict = {
        "gate": "G-A",
        "target": target,
        "kmin": kmin,
        "kmax": kmax,
        "K_real": k_real,
        "pass": k_real is not None,
        "nk_window": {str(k): nk.get(str(k), 0) for k in range(kmin, kmax + 1)},
    }
    return verdict


def _rebuild_gc_panel(args: argparse.Namespace, out_dir: Path) -> dict:
    """Rebuild the round-10 conversation panel BY ID from the pinned prefix_store.

    Fetches ``issue1092_realistic_crossing/corpus/prefix_store.jsonl`` at the
    round-10 DATA_REPO_REV, applies the round-10 panel construction
    (``_dynamics_panel`` + ``_filter_dynamics_panel_by_rendered_length``,
    imported verbatim from issue1092_gpu_phase), and digest-asserts the
    resulting id set against the round-10 drop report (n_kept + dropped ids).
    Writes gc_panel.jsonl + returns the digest record.
    """
    from huggingface_hub import hf_hub_download
    from issue1092_gpu_phase import (
        MAX_MODEL_LEN,
        _dynamics_panel,
        _filter_dynamics_panel_by_rendered_length,
        load_store,
    )
    from transformers import AutoTokenizer

    corpus_dir = out_dir / "gc_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    dest = corpus_dir / "prefix_store.jsonl"
    if not dest.exists():
        last: Exception | None = None
        for attempt in range(4):
            try:
                got = Path(
                    hf_hub_download(
                        HF_DATA_REPO,
                        f"{R10_STORE_PREFIX}/corpus/prefix_store.jsonl",
                        repo_type="dataset",
                        revision=DATA_REPO_REV,
                        local_dir=str(corpus_dir),
                    )
                )
                if got.resolve() != dest.resolve():
                    os.replace(got, dest)
                last = None
                break
            except Exception as e:
                last = e
                logger.info("[gc] prefix_store fetch retry %d/4: %s", attempt + 1, type(e).__name__)
                time.sleep(15 * (attempt + 1))
        if last is not None:
            raise RuntimeError("[gc] prefix_store fetch failed after 4 attempts") from last

    if args.smoke:
        tok_i = AutoTokenizer.from_pretrained(args.tiny_tokenizer_dir, trust_remote_code=True)
        tok_p = tok_i
    else:
        from issue1092_gpu_phase import (
            INSTRUCT_MODEL,
            INSTRUCT_REVISION,
            PRETRAINED_MODEL,
            PRETRAINED_REVISION,
        )

        tok_i = AutoTokenizer.from_pretrained(
            INSTRUCT_MODEL, revision=INSTRUCT_REVISION, trust_remote_code=True
        )
        tok_p = AutoTokenizer.from_pretrained(
            PRETRAINED_MODEL, revision=PRETRAINED_REVISION, trust_remote_code=True
        )
        # the instruct render helper lazily loads the pinned tokenizer; pre-seed
        import issue1092_gpu_phase as gp

        gp._get_tokenizer._tok = tok_i

    store = load_store(corpus_dir, "prefix_store.jsonl")
    panel = _dynamics_panel(store)
    tokenizers = {"instruct": tok_i, "pretrained": tok_p}
    panel_kept, digest = _filter_dynamics_panel_by_rendered_length(
        panel, tokenizers, max_tokens=MAX_MODEL_LEN
    )
    kept_ids = sorted(
        str(item.get("conv_id") or item.get("prefix_id") or item.get("id")) for item in panel_kept
    )

    record: dict = {
        "source": f"{R10_STORE_PREFIX}/corpus/prefix_store.jsonl@{DATA_REPO_REV}",
        "n_panel_prefilter": len(panel),
        "n_kept": len(kept_ids),
        "id_set_sha256": hashlib.sha256("\n".join(kept_ids).encode()).hexdigest(),
        "filter_digest": {k: v for k, v in digest.items() if k != "dropped"},
    }
    # Assumption 11 verification: the rebuilt set must match the round-10 drop
    # report exactly (same n_kept + same dropped conv ids) — production only
    # (the smoke's tiny tokenizer changes the length filter by construction).
    if not args.smoke:
        if not R10_RESULTS_JSON.exists():
            raise FileNotFoundError(f"[gc] round-10 results.json missing: {R10_RESULTS_JSON}")
        with open(R10_RESULTS_JSON) as f:
            r10 = json.load(f)
        d10 = r10["drops"]["instruct"]["drop_report"]["length_filter_digest"]
        if int(d10["n_kept"]) != len(kept_ids):
            raise AssertionError(
                f"[gc] G-C id-set recovery FAILED: rebuilt n_kept={len(kept_ids)} != "
                f"round-10 digest n_kept={d10['n_kept']} — panel construction drifted"
            )
        dropped_r10 = sorted({str(d["conv_id"]) for d in d10.get("dropped", [])})
        dropped_new = sorted({str(d["conv_id"]) for d in digest.get("dropped", [])})
        if dropped_r10 != dropped_new:
            raise AssertionError(
                f"[gc] G-C dropped-id mismatch: {len(dropped_r10)} banked vs "
                f"{len(dropped_new)} rebuilt"
            )
        record["r10_digest_match"] = True
        logger.info(
            "[gc] round-10 panel rebuilt BY ID: %d conversations (digest match)", len(kept_ids)
        )

    rows = []
    for item in panel_kept:
        cid = str(item.get("conv_id") or item.get("prefix_id") or item.get("id"))
        turns = item.get("prefix_turns") or item.get("turns")
        rows.append({"conv_id": cid, "turns": turns})
    _write_jsonl_sharded(rows, out_dir, "gc_panel")
    return record


def _upload_panel(out_dir: Path, smoke: bool) -> None:
    """Unconditional text upload (persist-by-default; non-LFS path)."""
    if smoke:
        logger.info("[upload] smoke: skipping HF upload")
        return
    import random as _r

    from huggingface_hub import upload_folder

    last: Exception | None = None
    for attempt in range(5):
        try:
            upload_folder(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                folder_path=str(out_dir),
                path_in_repo=HF_PANEL_PREFIX,
                allow_patterns=["*.json", "*_shard*.jsonl"],
                ignore_patterns=["gc_corpus/*", "stream_cache/*"],
                commit_message="issue-825 turn-dynamics: P0 harvest panel + tables",
            )
            logger.info("[upload] panel + tables -> %s/%s", HF_DATA_REPO, HF_PANEL_PREFIX)
            return
        except Exception as e:
            last = e
            wait = 60 * (2**attempt) + _r.uniform(0, 15)
            logger.warning(
                "[upload] attempt %d/5 failed (%s); retry in %.0fs",
                attempt + 1,
                type(e).__name__,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError("[upload] panel upload FAILED after 5 attempts") from last


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="harvest output root")
    ap.add_argument("--panel-n", type=int, default=5000)
    ap.add_argument("--spare-n", type=int, default=1000)
    ap.add_argument("--ga-target", type=int, default=5000)
    ap.add_argument("--ga-kmin", type=int, default=8)
    ap.add_argument("--ga-kmax", type=int, default=15)
    ap.add_argument(
        "--stream-limit",
        type=int,
        default=0,
        help="cap TOTAL rows examined per source (0 = full stream)",
    )
    ap.add_argument("--no-resume-stream", action="store_true")
    ap.add_argument("--skip-gc", action="store_true", help="skip the round-10 parity panel")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny-tokenizer-dir", default="", help="tokenizer dir for --smoke gc")
    args = ap.parse_args()
    if args.smoke and not args.skip_gc and not args.tiny_tokenizer_dir:
        ap.error("--smoke without --skip-gc requires --tiny-tokenizer-dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "stream_cache"
    stream_limit = args.stream_limit or None
    t0 = time.time()

    # ---- full-corpus streams (r8.1 recipe; checkpoint/resume per source) ----
    pool: list[dict] = []
    funnel: dict[str, dict] = {}
    for repo, rev in ((WILDCHAT_REPO, WILDCHAT_REV), (LMSYS_REPO, LMSYS_REV)):
        stats: dict = {}
        rows = _stream_with_cache(
            repo,
            rev,
            rng=random.Random(PANEL_SEED),
            row_limit=None,
            stream_limit=stream_limit,
            lang_filter="en",
            stats_out=stats,
            cache_dir=cache_dir,
            resume=not args.no_resume_stream,
        )
        funnel[repo] = stats
        pool.extend(rows)
    logger.info("[harvest] kept pool: %d conversations (%.0fs)", len(pool), time.time() - t0)

    # ---- n(k) table + gate G-A ----
    nk = _nk_table(pool)
    ga = gate_ga(nk, target=args.ga_target, kmin=args.ga_kmin, kmax=args.ga_kmax)
    with open(out_dir / "nk_table.json", "w") as f:
        json.dump({"nk": nk, "gate_ga": ga, "filter_recipe": FILTER_RECIPE_VERSION}, f, indent=1)
    logger.info(
        "[G-A] window n(k): %s -> K_real=%s pass=%s", ga["nk_window"], ga["K_real"], ga["pass"]
    )
    if not ga["pass"]:
        logger.error(
            "[G-A] FAIL: no k in [%d, %d] reaches n(k) >= %d — would contradict the "
            "#1092 funnel + probe by >3x. STOP before any GPU (plan §7).",
            args.ga_kmin,
            args.ga_kmax,
            args.ga_target,
        )
        raise SystemExit(3)
    k_real = int(ga["K_real"])

    # ---- deep pool + arm-R panel (seed 42) ----
    deep_pool = [r for r in pool if int(r["n_user_turns"]) >= args.ga_kmin]
    _write_jsonl_sharded(deep_pool, out_dir, "deep_pool")
    eligible = sorted(
        (r for r in pool if int(r["n_user_turns"]) >= k_real), key=lambda r: str(r["id"])
    )
    rng = random.Random(PANEL_SEED)
    if len(eligible) < args.panel_n:
        raise SystemExit(
            f"[panel] eligible {len(eligible)} < panel_n {args.panel_n} at K_real={k_real} "
            f"— G-A passed but sampling cannot fill the panel (inconsistent state)"
        )
    panel = rng.sample(eligible, args.panel_n)
    panel_ids = {str(r["id"]) for r in panel}
    _write_jsonl_sharded(panel, out_dir, "panel_armR")

    # ---- arm-G seeds: panel u1s + spare deep-pool seeds, one brief per conv ----
    spare_pool = sorted(
        (r for r in deep_pool if str(r["id"]) not in panel_ids), key=lambda r: str(r["id"])
    )
    rng_sp = random.Random(PANEL_SEED + 1)
    spares = rng_sp.sample(spare_pool, min(args.spare_n, len(spare_pool)))
    seeds: list[dict] = []
    for i, r in enumerate(panel + spares):
        bid, btext = USER_BRIEFS[i % len(USER_BRIEFS)]
        u1 = next(t["content"] for t in r["turns"] if t["role"] == "user")
        seeds.append(
            {
                "conv_id": str(r["id"]),
                "seed_rank": i,
                "in_panel": str(r["id"]) in panel_ids,
                "u1": u1,
                "brief_id": bid,
                "brief_text": btext,
            }
        )
    _write_jsonl_sharded(seeds, out_dir, "armG_seeds")

    # ---- round-10 G-C parity panel ----
    gc_record: dict | None = None
    if not args.skip_gc:
        gc_record = _rebuild_gc_panel(args, out_dir)

    # ---- report + upload ----
    report = {
        "issue": 825,
        "followup_label": "turn-dynamics-allturns-5000",
        "phase": "P0-harvest",
        "filter_recipe": FILTER_RECIPE_VERSION,
        "dataset_revisions": {WILDCHAT_REPO: WILDCHAT_REV, LMSYS_REPO: LMSYS_REV},
        "stream_limit": stream_limit,
        "funnel": funnel,
        "n_kept_pool": len(pool),
        "n_deep_pool": len(deep_pool),
        "nk_table": nk,
        "gate_ga": ga,
        "K_real": k_real,
        "panel_n": len(panel),
        "n_seeds": len(seeds),
        "n_briefs": len(USER_BRIEFS),
        "gc_panel": gc_record,
        "panel_ids_sha256": hashlib.sha256("\n".join(sorted(panel_ids)).encode()).hexdigest(),
        "seed": PANEL_SEED,
        "elapsed_s": round(time.time() - t0, 1),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "python_version": sys.version.split()[0],
        "smoke": bool(args.smoke),
    }
    with open(out_dir / "harvest_report.json", "w") as f:
        json.dump(report, f, indent=1)
    logger.info(
        "[harvest] report -> %s (panel sha %s)",
        out_dir / "harvest_report.json",
        report["panel_ids_sha256"][:12],
    )
    _upload_panel(out_dir, smoke=args.smoke or args.skip_upload)
    print(f"[harvest] DONE K_real={k_real} panel_n={len(panel)} seeds={len(seeds)}")


if __name__ == "__main__":
    main()
