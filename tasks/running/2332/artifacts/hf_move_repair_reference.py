#!/usr/bin/env python3
"""Repair pass for prefixes the first mover left in state['failed'].

Differences from the first pass, each fixing a real flaw it exposed:
  * 4 download workers, not 16. A serial probe of 25 files succeeded 25/25
    right after 1,408/35,394 failed under 16-way concurrency, so the failures
    were load-induced, not bad objects.
  * Exponential backoff to ~60s (was ~6s) — too short for a rate-limit window.
  * Staging is PRESERVED on failure. The first pass rmtree'd 34k good files
    after 1,408 failed, forcing a full re-download.
  * Actual exception text is recorded per failed path, and the failed paths
    themselves are written out. The first pass swallowed both.
  * hf_hub_download with a stable local_dir already skips files present with
    a matching etag, so a re-run resumes rather than refetching.

Same safety contract as the first pass: verify name+size on the destination
before ANY deletion; leave the source intact on any doubt.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

SRC = "superkaiba1/explore-persona-space-data"
DST = "superkaiba1/explore-persona-space-overflow"
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2304_residue_move")
STATE = STAGE / "state.json"
DL_WORKERS = 4
MAX_ATTEMPTS = 6

api = HfApi()


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def listing(repo: str, prefix: str) -> dict[str, int]:
    try:
        return {
            t.path: t.size
            for t in api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset", recursive=True)
            if getattr(t, "size", None) is not None
        }
    except Exception as e:
        if "404" in str(e):
            return {}
        raise


def fetch_all(prefix: str, paths: list[str], dest: Path) -> tuple[list[str], Counter]:
    dest.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    errs: Counter = Counter()

    def one(p: str):
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                hf_hub_download(repo_id=SRC, filename=p, repo_type="dataset",
                                local_dir=str(dest), etag_timeout=60)
                return None, ""
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:100]}"
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(min(60, 2 ** (attempt + 1)))
        return p, last

    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futs = {ex.submit(one, p): p for p in paths}
        n = 0
        for f in as_completed(futs):
            bad, msg = f.result()
            if bad:
                failed.append(bad)
                errs[msg] += 1
            n += 1
            if n % 2000 == 0:
                log(f"    ...{n:,}/{len(paths):,}  (failed so far: {len(failed)})")
    return failed, errs


def main() -> int:
    st = json.loads(STATE.read_text())
    # Cover EVERY prefix not yet verified-and-reclaimed, not just the ones that
    # already failed: the 16-worker first pass failed on both prefixes it tried
    # (4% and 6%), so the remaining ones would fail the same way.
    all_prefixes = json.loads(Path("/tmp/move_prefixes.json").read_text())
    done = set(st.get("done", {}))
    targets = [p for p in all_prefixes if p not in done]
    if not targets:
        log("nothing left to move")
        return 0
    log(f"moving {len(targets)} remaining prefixes at {DL_WORKERS} workers "
        f"(first pass used 16 and failed on both prefixes it attempted)")

    for prefix in targets:
        src_map = listing(SRC, prefix)
        if not src_map:
            log(f"{prefix}: absent on canonical — nothing to repair")
            continue
        log(f"{prefix}: {len(src_map):,} files on canonical, {sum(src_map.values())/1e9:.2f} GB")

        stage = STAGE / prefix
        failed, errs = fetch_all(prefix, list(src_map), stage)
        if failed:
            log(f"    STILL INCOMPLETE: {len(failed)} failed — source left intact")
            for msg, c in errs.most_common(5):
                log(f"      x{c}  {msg}")
            (STAGE / f"{prefix}.failed.json").write_text(json.dumps(failed, indent=1))
            st["failed"][prefix] = f"repair: {len(failed)} still failing"
            STATE.write_text(json.dumps(st, indent=1))
            continue
        log("    download complete")

        # upload_large_folder, NOT upload_folder. Measured: upload_folder on a
        # big-file prefix throws 429/503/504 whose meaning is undecidable — on
        # issue1090_partial all four attempts "failed" yet the data landed; on
        # issue1689_partial and issue1092_partial identical errors meant it had
        # not. upload_large_folder reports the 429 and auto-shrinks its commit
        # batch until it succeeds ("Failed to commit 50 files at once. Will
        # retry with less files in next batch"), which is what actually gets a
        # big-file prefix committed.
        #
        # It takes no path_in_repo, but staging already mirrors the repo layout
        # (stage/<prefix>/...), so uploading `stage` as root lands files at the
        # right paths. Ignore the HF metadata cache and any partial-download
        # temp files left by an interrupted run.
        try:
            api.upload_large_folder(
                repo_id=DST, folder_path=str(stage), repo_type="dataset",
                ignore_patterns=[".cache/**", "**/tmp_*", "**/*.incomplete"],
                num_workers=4, print_report_every=120,
            )
        except Exception as e:
            log(f"    upload_large_folder raised: {type(e).__name__} {str(e)[:80]}")

        # Verification is the ONLY reliable signal on this repo: transport
        # errors are undecidable in both directions (see the note above), so
        # success is defined solely by reading the destination back.
        dst_map = listing(DST, prefix)
        missing = [p for p in src_map if p not in dst_map]
        mismatch = [p for p in src_map if p in dst_map and dst_map[p] != src_map[p]]
        if missing or mismatch:
            log(f"    VERIFY FAILED: {len(missing)} missing, {len(mismatch)} mismatch — source left intact")
            st["failed"][prefix] = f"repair verify: {len(missing)}/{len(mismatch)}"
            STATE.write_text(json.dumps(st, indent=1))
            continue
        log(f"    VERIFIED on overflow: {len(dst_map):,} files, name+size match")

        for attempt in range(4):
            if not listing(SRC, prefix):
                break
            try:
                api.delete_folder(path_in_repo=prefix, repo_id=SRC, repo_type="dataset",
                                  commit_message=f"reclaim slots: {prefix} moved to {DST}")
            except Exception as e:
                log(f"    delete attempt {attempt}: {type(e).__name__} {str(e)[:70]}")
            time.sleep(3)

        if listing(SRC, prefix):
            log("    DELETE INCOMPLETE")
            st["failed"][prefix] = "repair: delete incomplete"
        else:
            log(f"    RECLAIMED {len(src_map):,} slots")
            st.setdefault("done", {})[prefix] = {"files": len(src_map), "bytes": sum(src_map.values()), "via": "repair"}
            st["failed"].pop(prefix, None)
            shutil.rmtree(stage, ignore_errors=True)
        STATE.write_text(json.dumps(st, indent=1))

    log(f"REPAIR COMPLETE. done={len(st.get('done',{}))} failed={len(st.get('failed',{}))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
