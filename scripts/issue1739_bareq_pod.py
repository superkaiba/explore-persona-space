#!/usr/bin/env python3
"""#1739 BARE-QUERY capture leg (launch-box driver) — user-requested round.

Answers "can a bare query, with no conversation prefix, predict the behavior the
model produced when the prefix WAS there?" The answers/labels stay as-is (judged
DVs from rollouts generated under the FULL context); the NEW input is the same
final user query re-rendered BARE (chat-template head + query, history stripped)
and captured at the standard summary positions.

Scope is set by measured evidence, not by symmetry (see the design report):
``prefix_end`` is CONSTANT at the 21-token template head for every train row of
sycophancy and hallucination, and their committed prefix arms read median
max|rho| 0.003 / 0.059 — i.e. their contexts are ALREADY bare-query renders and
the manipulation is a NO-OP there. Only two surfaces carry a real prefix:

* LEG 1 — the wildchat rung's 1,013 MULTI-TURN contexts (real conversation
  prefixes; the 987 single-turn contexts render byte-identically bare, so their
  existing reps ARE their bare reps and are reused under a BIT-equality gate).
* LEG 2 — evil's train pool (prefix-crossed, 1,348 prefixes over 8,000
  contexts). Its bare rep depends only on the QUERY, so captures dedupe to the
  query bank (measured: 524 train contexts -> 293 unique queries pooled over two
  shards, implying a bank of ~400).

Render convention is #1092's, reached through this project's own renderer:
``render_row_prompt(tok, [], query)`` is byte-identical to
``issue1092_gpu_phase._render_bare_query`` for the instruct model (verified), so
no new convention is invented and the reps join the same space. #1092 captured a
single kind for bare rows (``c_q_bare``) because a bare render's prefix is
content-independent — we keep that and additionally emit the constant-prefix
position ONCE as a fail-loud null probe.

PHASES (leg 2 needs three, because the query TEXT lives in no #1739 tensor
artifact — only in the per-rollout JSONs on the Hub):

    1. ``extract``  — stage evil's ``labeling_evil.shard*.jsonl`` raw-completion
       shards, stream them, and write the deduplicated unique-query manifest.
       Runs ON THE LAUNCH BOX: 40k rollout records do not belong on the shared VM.
    2. ``capture``  — bare-render + teacher-forced capture at all 28 layers, for
       the leg-2 query bank and/or the leg-1 wcrung multi-turn contexts.
    3. ``upload``   — store + manifests to the data repo (text-first JSON always).

``--pilot-only`` measures ONE production-shape batch through this very
entrypoint, writes the projection, and exits rc=8 when the projection exceeds
``--fence-hours`` — a DESIGNED halt (distinct rc, report written BEFORE exit),
the same contract as the nonlinear round's rc=7 pilot gate.

CONTENT HYGIENE: this leg necessarily HOLDS real user query text (it must render
it), but never logs or prints it. Every log line and artifact field is ids,
counts, hashes, and shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_bareq_pod.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root derivation failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE any torch import (thread caps + credentials)

logger = logging.getLogger("issue1739_bareq_pod")

RUNG = "bareq"
HF_PREFIX = "issue1739_ctxmap/bareq_map"
# Where evil's rollout JSONs live (query TEXT source for leg 2). MEASURED: these
# shards carry EVERY rung's rows, not just train — the 53,330 rollout rows the
# train-only extract already streamed equal the sum of n_rollouts_judged over
# all 10,666 labeled contexts (8,000 train + 1,995 hhrt + 671 toxicchat). The
# eval-rung query TEXT was therefore always present and simply discarded by the
# train_only filter; the labeling rows carry rung + context_id but NO query
# text, so one streaming pass over these same shards builds every rung's bank.
RAW_PREFIX = "issue1739_ctxmap/raw_completions"
RAW_SHARD_GLOB = "labeling_evil.shard{:02d}.jsonl"
# Per-rung query banks for the OOD eval rungs. Same schema as QUERY_MANIFEST
# (the scorer's load_query_bank reads only `queries[].query_id` +
# `queries[].context_ids` and `.get()`s the rest), so a per-rung bank is a
# drop-in for its --query-manifest flag. The TRAIN bank keeps the legacy
# QUERY_MANIFEST path + key set byte-for-byte.
RUNG_MANIFEST_FMT = "bareq_queries_{behavior}_{rung}.json"
RUNG_BANKS_SUMMARY = "bareq_rung_banks.json"
TRAIN_RUNG = "train"
# Per-behavior DV labeling (authoritative context_id -> rung attribution).
DV_ROOT = Path("eval_results/issue_1739/dv_dataset")
# The wildchat rung's shared pool (leg 1 source of multi-turn contexts).
WCRUNG_CONTEXTS_PREFIX = "issue1739_ctxmap/wildchat_rung/contexts"
SENTINEL_NAME = "bareq_capture_done.json"
QUERY_MANIFEST = "bareq_queries.json"
PILOT_REPORT = "bareq_pilot_report.json"
# Designed-halt rc for an over-fence pilot projection (distinct from rc=2 crash
# and from the nonlinear round's rc=7).
PILOT_FENCE_RC = 8
# Bare renders carry no conversational prefix, so the ONLY informative summary
# position is the context end; the prefix position is the constant template head
# and is emitted once as a null probe (see _null_probe_report).
BARE_KIND = "context_end"


def _query_id(text: str) -> str:
    """Stable content id for a query — the join key, never the text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_json_atomic(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)
    return path


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
            env={**os.environ},
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# phase 1 — extract + dedupe the unique-query bank (leg 2)
# ---------------------------------------------------------------------------


def iter_raw_shards(args, token: str):
    """Yield staged local paths of evil's labeling raw-completion shards.

    Staged one at a time and (optionally) reaped after streaming, so the launch
    box never holds the whole 40k-rollout tree at once.
    """
    from explore_persona_space.orchestrate import hub

    dest = args.stage_root / "raw_completions"
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(args.max_shards):
        name = RAW_SHARD_GLOB.format(i)
        local = dest / name
        if not local.is_file():
            try:
                hub.stage_hub_file(
                    hub.DEFAULT_DATASET_REPO,
                    f"{RAW_PREFIX}/{name}",
                    local,
                    repo_type="dataset",
                    token=token,
                )
            except Exception as exc:  # noqa: BLE001 — shard exhaustion is the stop signal
                logger.info(
                    "[phase=extract] shard %s unavailable (%s) — stopping", name, type(exc).__name__
                )
                return
        yield local


def load_rung_map(behavior: str) -> dict[str, str]:
    """``context_id -> rung`` from the behavior's DV labeling (authoritative).

    The labeling rows carry ``rung`` + ``context_id`` but NO query text, and the
    rollout shards carry the query text but no rung — so rung attribution joins
    the two on context_id. Fails loud rather than falling back to a context_id
    substring guess: a mis-attributed rung silently mixes an OOD bank with train.
    """
    path = _REPO_ROOT / DV_ROOT / behavior / "labeling.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"DV labeling missing at {path} — rung attribution needs it (context_id -> rung)"
        )
    rows = json.loads(path.read_text()).get("rows") or []
    rung_of = {str(r["context_id"]): str(r.get("rung")) for r in rows if r.get("context_id")}
    if not rung_of:
        raise RuntimeError(f"{path}: no context_id rows — cannot attribute rungs")
    return rung_of


def _bank_payload(by_query: dict[str, dict]) -> tuple[list[dict], int]:
    """``(sorted bank, n_member_contexts)`` — the shared manifest body shape."""
    bank = sorted(by_query.values(), key=lambda e: e["query_id"])
    return bank, len({c for e in bank for c in e["context_ids"]})


def extract_query_bank(args, token: str) -> dict:
    """Stream the raw shards ONCE; write the per-rung deduplicated query banks.

    A manifest is the capture width for its rung: one row per UNIQUE query, with
    the member context_ids that share it. That sharing is exactly why the fit
    must use BY-QUERY folds — every member row carries the IDENTICAL bare rep,
    so any fold splitting them leaks a duplicated feature vector. Dedup is
    PER RUNG, so each rung's bank is self-contained and its member context_ids
    are exactly that rung's.

    The TRAIN bank keeps the legacy ``QUERY_MANIFEST`` path AND its exact key
    set, and is selected by the ORIGINAL context_id predicate — byte-compatible
    by construction rather than by assumption. The eval-rung banks come from the
    authoritative labeling rung map; the two train views' agreement is reported
    in the separate rung-banks summary, never folded into the train manifest.
    """
    rung_of = load_rung_map(args.behavior)
    eval_rungs = sorted({r for r in rung_of.values() if r != TRAIN_RUNG})
    by_rung: dict[str, dict[str, dict]] = {r: {} for r in (TRAIN_RUNG, *eval_rungs)}
    legacy_main: dict[str, dict] = {}
    n_rows = n_kept = n_unmapped = 0
    for shard_i, local in enumerate(iter_raw_shards(args, token)):
        shard_rows = 0
        with local.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line).get("doc", {})
                cid, q = doc.get("context_id"), doc.get("query")
                if not cid or q is None:
                    continue
                n_rows += 1
                shard_rows += 1
                qid = _query_id(q)
                cid = str(cid)
                # Legacy MAIN-manifest view: the ORIGINAL predicate preserved
                # verbatim (train-only by default; --all-rungs keeps every rung
                # in the one bank), so QUERY_MANIFEST stays byte-compatible.
                if not args.train_only or "train" in cid:
                    n_kept += 1
                    ent = legacy_main.setdefault(
                        qid, {"query_id": qid, "query": q, "context_ids": []}
                    )
                    if cid not in ent["context_ids"]:
                        ent["context_ids"].append(cid)
                rung = rung_of.get(cid)
                if rung is None:
                    n_unmapped += 1
                    continue
                ent = by_rung[rung].setdefault(
                    qid, {"query_id": qid, "query": q, "context_ids": []}
                )
                if cid not in ent["context_ids"]:
                    ent["context_ids"].append(cid)
        print(
            f"[phase=extract] shard {shard_i} {local.name}: rows={shard_rows} "
            f"cum_rows={n_rows} cum_kept={n_kept}",
            flush=True,
        )
        if args.reap_shards:
            local.unlink(missing_ok=True)

    bank, n_ctx = _bank_payload(legacy_main)
    manifest = {
        "leg": "bareq_extract",
        "behavior": args.behavior,
        "train_only": bool(args.train_only),
        "n_rollout_rows_seen": n_rows,
        "n_rows_kept": n_kept,
        "n_contexts": n_ctx,
        "n_unique_queries": len(bank),
        "dedupe_ratio_contexts_per_query": round(n_ctx / max(len(bank), 1), 3),
        "queries": bank,  # holds TEXT (needed to render); never logged
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = _write_json_atomic(args.out_root / QUERY_MANIFEST, manifest)
    print(
        f"[phase=extract] rows_seen={n_rows} kept={n_kept} contexts={n_ctx} "
        f"unique_queries={len(bank)} ratio={manifest['dedupe_ratio_contexts_per_query']} -> {out}",
        flush=True,
    )

    legacy_qids = set(legacy_main)
    # The true train query set comes from the rung map, so the shared-query
    # counts below hold under --all-rungs too (where legacy_main spans all rungs).
    train_bank_qids = {e["query_id"] for e in by_rung[TRAIN_RUNG].values()}
    summary: dict[str, dict] = {}
    for rung in (TRAIN_RUNG, *eval_rungs):
        r_bank, r_ctx = _bank_payload(by_rung[rung])
        r_qids = {e["query_id"] for e in r_bank}
        row = {
            "rung": rung,
            "n_contexts": r_ctx,
            "n_unique_queries": len(r_bank),
            "dedupe_ratio_contexts_per_query": round(r_ctx / max(len(r_bank), 1), 3),
            # A query whose TEXT also appears in the train bank renders to the
            # IDENTICAL bare rep (a bare render depends only on the query), so
            # this count is the re-capture the per-rung stores duplicate — kept
            # deliberately so each rung's store is self-contained.
            "n_queries_shared_with_train_bank": len(r_qids & train_bank_qids),
        }
        if rung == TRAIN_RUNG:
            # Legacy-predicate vs labeling-map agreement on the train view.
            row["legacy_predicate_n_unique_queries"] = len(bank)
            row["legacy_predicate_n_contexts"] = n_ctx
            row["agrees_with_legacy_predicate"] = bool(not args.train_only or r_qids == legacy_qids)
            summary[rung] = row
            continue
        r_manifest = {
            "leg": "bareq_extract",
            "behavior": args.behavior,
            "rung": rung,
            "train_only": False,
            "n_rollout_rows_seen": n_rows,
            "n_rows_kept": sum(len(e["context_ids"]) for e in r_bank),
            "n_contexts": r_ctx,
            "n_unique_queries": len(r_bank),
            "dedupe_ratio_contexts_per_query": row["dedupe_ratio_contexts_per_query"],
            "queries": r_bank,  # holds TEXT (needed to render); never logged
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        r_out = _write_json_atomic(
            args.out_root / RUNG_MANIFEST_FMT.format(behavior=args.behavior, rung=rung), r_manifest
        )
        row["manifest"] = r_out.name
        summary[rung] = row
        print(
            f"[phase=extract] rung={rung} contexts={r_ctx} unique_queries={len(r_bank)} "
            f"ratio={row['dedupe_ratio_contexts_per_query']} "
            f"shared_qids_with_train={row['n_queries_shared_with_train_bank']} -> {r_out}",
            flush=True,
        )
    _write_json_atomic(
        args.out_root / RUNG_BANKS_SUMMARY,
        {
            "behavior": args.behavior,
            "rungs": summary,
            "n_rollout_rows_seen": n_rows,
            "n_rows_unmapped_to_any_rung": n_unmapped,
            "note": (
                "the rollout shards carry EVERY rung's rows; the train_only filter discarded the "
                "eval-rung rows rather than a separate source being needed. Rung attribution "
                "joins the shards (query text) to the DV labeling (rung) on context_id."
            ),
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    if n_unmapped:
        print(
            f"[phase=extract] WARNING {n_unmapped} rollout rows had no labeling rung "
            "(excluded from every per-rung bank; the train manifest is unaffected)",
            flush=True,
        )
    return manifest


# ---------------------------------------------------------------------------
# phase 2 — bare render + capture
# ---------------------------------------------------------------------------


def bare_render(tokenizer, query: str) -> tuple[str, str]:
    """``(prefix_text, prompt_text)`` for a BARE query — #1092's convention.

    Verified byte-identical to ``issue1092_gpu_phase._render_bare_query`` for the
    instruct model. ``prefix_text`` is the content-independent template head, so
    it is the SAME string for every query — which is why it serves as the null
    probe rather than a predictive arm.
    """
    from scripts.issue1739_wcrung_contexts import render_row_prompt

    return render_row_prompt(tokenizer, [], query)


def _load_wcrung_rows(args) -> list[dict]:
    """Load the wcrung context rows, staging the HF shards when local is absent.

    The full rows are HF-only (free-text routing): line-split JSONL shards +
    manifest under WCRUNG_CONTEXTS_PREFIX (schema wcrung-rows-shards-v1) —
    there is no monolithic wcrung.json anywhere (att-20260731-141952 crashed
    on exactly that assumption). A local --wcrung-rows-json that EXISTS is
    honored unchanged; otherwise stage manifest + shards next to that path,
    verify each shard's sha256 + n_rows against the manifest, and concatenate.
    """
    p = Path(args.wcrung_rows_json)
    if p.is_file():
        wc = json.loads(p.read_text())
        return wc["rows"] if isinstance(wc, dict) and "rows" in wc else wc
    from explore_persona_space.orchestrate import hub

    token = os.environ.get("HF_TOKEN") or ""
    dest = p.parent
    dest.mkdir(parents=True, exist_ok=True)
    man_local = dest / "wcrung_rows.manifest.json"
    if not man_local.is_file():
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            f"{WCRUNG_CONTEXTS_PREFIX}/wcrung_rows.manifest.json",
            man_local,
            repo_type="dataset",
            token=token,
        )
    man = json.loads(man_local.read_text())
    if man.get("schema") != "wcrung-rows-shards-v1":
        raise RuntimeError(f"unexpected wcrung rows manifest schema: {man.get('schema')!r}")
    rows: list[dict] = []
    for shard in man["shards"]:
        local = dest / shard["name"]
        if not local.is_file():
            hub.stage_hub_file(
                hub.DEFAULT_DATASET_REPO,
                f"{WCRUNG_CONTEXTS_PREFIX}/{shard['name']}",
                local,
                repo_type="dataset",
                token=token,
            )
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != shard["sha256"]:
            raise RuntimeError(f"{shard['name']}: sha256 mismatch vs manifest ({digest})")
        n = 0
        with local.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                    n += 1
        if n != shard["n_rows"]:
            raise RuntimeError(f"{shard['name']}: {n} rows != manifest n_rows {shard['n_rows']}")
    if len(rows) != man["n_rows"]:
        raise RuntimeError(f"staged {len(rows)} rows != manifest n_rows {man['n_rows']}")
    print(
        f"[bareq] staged wcrung rows from HF: {len(man['shards'])} shards, {len(rows)} rows",
        flush=True,
    )
    return rows


def leg2_manifest_path(args) -> Path:
    """The query bank this invocation's leg-2 capture consumes (rung-selected)."""
    if args.rung == TRAIN_RUNG:
        return args.out_root / QUERY_MANIFEST
    return args.out_root / RUNG_MANIFEST_FMT.format(behavior=args.behavior, rung=args.rung)


def leg2_store_dir(args) -> Path:
    """Store child for this invocation's leg-2 rows — one child per rung.

    Train keeps the legacy ``bareq_<behavior>`` name (the scorer's
    ``resolve_bareq_store`` leg-2 preference); each OOD rung gets its own
    ``bareq_<behavior>_<rung>`` child, which the scorer consumes by pointing
    ``--bareq-store`` straight at it (an explicit dir that IS a capture store
    short-circuits its name resolution).
    """
    suffix = "" if args.rung == TRAIN_RUNG else f"_{args.rung}"
    return args.store_root / f"bareq_{args.behavior}{suffix}"


def _capture_fingerprint(args, n_rows: int) -> str:
    """Resume fingerprint — legacy shape for train, rung-scoped for OOD rungs."""
    if args.rung == TRAIN_RUNG:
        return f"bareq-{args.behavior}-{n_rows}"
    return f"bareq-{args.behavior}-{args.rung}-{n_rows}"


def build_capture_rows(args, tokenizer) -> list[dict]:
    """Rows to capture: the leg-2 query bank and/or leg-1 wcrung multi-turn."""
    rows: list[dict] = []
    if args.leg in ("2", "both"):
        man_path = leg2_manifest_path(args)
        if not man_path.is_file():
            raise FileNotFoundError(
                f"leg-2 query bank missing at {man_path} (rung={args.rung}) — "
                "run --phase extract first"
            )
        man = json.loads(man_path.read_text())
        for e in man["queries"]:
            prefix_text, prompt_text = bare_render(tokenizer, e["query"])
            rows.append(
                {
                    "row_id": f"q-{e['query_id']}",
                    "query_id": e["query_id"],
                    "kind": "leg2_query_bank",
                    "rung": args.rung,
                    "prefix_text": prefix_text,
                    "prompt_text": prompt_text,
                    "completion": "",
                    "n_member_contexts": len(e["context_ids"]),
                }
            )
    if args.leg in ("1", "both"):
        src = _load_wcrung_rows(args)
        for r in src:
            multi = bool(r.get("prefix_turns"))
            if args.multi_turn_only and not multi:
                continue
            prefix_text, prompt_text = bare_render(tokenizer, r["query"])
            rows.append(
                {
                    "row_id": f"wc-{r['context_id']}",
                    "context_id": r["context_id"],
                    "kind": "leg1_wcrung_multi_turn" if multi else "leg1_wcrung_single_turn",
                    "multi_turn": multi,
                    "prefix_text": prefix_text,
                    "prompt_text": prompt_text,
                    "completion": "",
                }
            )
    if not rows:
        raise RuntimeError(
            f"no capture rows for --leg {args.leg} — leg 2 needs {leg2_manifest_path(args).name} "
            "(run --phase extract first); leg 1 needs --wcrung-rows-json"
        )
    seen = {r["row_id"] for r in rows}
    if len(seen) != len(rows):
        raise RuntimeError(f"duplicate row_id in capture set ({len(rows) - len(seen)} dupes)")
    return rows


def _null_probe_report(rows: list[dict], tokenizer) -> dict:
    """The constant-prefix null: every bare render MUST share one prefix.

    Fail-loud rather than a soft warning — a differing prefix means the render
    convention drifted (or a row leaked a non-bare prefix), which would silently
    make the null arm predictive and invalidate the whole comparison.
    """
    prefixes = {r["prefix_text"] for r in rows}
    if len(prefixes) != 1:
        raise RuntimeError(
            f"bare renders produced {len(prefixes)} DISTINCT prefixes; expected exactly 1 "
            "(the content-independent template head) — render convention drift"
        )
    head = next(iter(prefixes))
    n_tok = len(tokenizer.encode(head, add_special_tokens=False))
    return {
        "constant_prefix_verified": True,
        "prefix_token_len": n_tok,
        "prefix_sha256": hashlib.sha256(head.encode("utf-8")).hexdigest(),
        "note": (
            "the bare-render prefix position is a CONSTANT vector across rows, so the "
            "prefix arm is a built-in null: it must read ~chance. A non-chance read is a "
            "capture/indexing bug, not a finding."
        ),
    }


def run_capture(args, rows: list[dict], tokenizer, model) -> dict:
    """Teacher-forced capture of the bare prompts at all layers."""
    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739.constants import HIDDEN_DIM, N_LAYERS

    store_dir = leg2_store_dir(args) if args.leg == "2" else args.store_root / "bareq"
    rollout_dir = args.out_root / "bare_rows"
    if args.leg == "2" and args.rung != TRAIN_RUNG:
        # Per-rung row files: a query shared with the train bank has the SAME
        # row_id, so a shared rollout dir would collide across rungs.
        rollout_dir = rollout_dir / args.rung
    rollout_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for r in rows:
        p = rollout_dir / f"{r['row_id']}.json"
        if not p.is_file():
            _write_json_atomic(p, {**r, "rollout_k": 0, "behavior": args.behavior, "rung": RUNG})
        paths.append(p)
    t0 = time.time()
    cap_kwargs = {
        "store_dir": store_dir,
        "model": model,
        "tokenizer": tokenizer,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "device": args.device,
        # The TRAIN fingerprint keeps its legacy form so the already-captured
        # train store still resumes (capture.shard_done keys on it); each OOD
        # rung gets its own so a rung's shards can never satisfy another's.
        "fingerprint": args.fingerprint or _capture_fingerprint(args, len(rows)),
    }
    if args.capture_batch_size:
        cap_kwargs["batch_size"] = args.capture_batch_size
    manifest = capture_mod.capture_rollout_files(sorted(paths), **cap_kwargs)
    el = time.time() - t0
    print(
        f"[phase=capture] rows={manifest.get('n_rows')} shards={manifest.get('n_shards')} "
        f"elapsed={el:.0f}s per_row={el / max(len(paths), 1):.3f}s -> {store_dir}",
        flush=True,
    )
    return {**manifest, "elapsed_s": round(el, 1), "per_row_s": round(el / max(len(paths), 1), 4)}


def run_pilot(args, rows: list[dict], tokenizer, model) -> int:
    """Measure ONE production-shape batch, project, and gate on --fence-hours.

    The projection basis is MEASURED here, through this same entrypoint at
    production shape — never asserted. Writes the report BEFORE any exit so a
    fenced halt is still fully diagnosable.
    """
    n = min(args.pilot_rows, len(rows))
    batch = rows[:n]
    t0 = time.time()
    cap = run_capture(args, batch, tokenizer, model)
    el = time.time() - t0
    per_row = el / max(n, 1)
    projected_h = per_row * len(rows) / 3600.0
    report = {
        "leg": "bareq_pilot",
        "pilot_rows": n,
        "pilot_elapsed_s": round(el, 1),
        "measured_per_row_s": round(per_row, 4),
        "total_rows_planned": len(rows),
        "projected_wall_h": round(projected_h, 3),
        "fence_hours": args.fence_hours,
        "over_fence": bool(args.fence_hours and projected_h > args.fence_hours),
        "capture_manifest": {k: cap.get(k) for k in ("n_rows", "n_shards")},
        "basis": "MEASURED through this entrypoint at production shape (not asserted)",
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = _write_json_atomic(args.out_root / PILOT_REPORT, report)
    print(
        f"[phase=pilot] measured {per_row:.3f}s/row over {n} rows -> projected "
        f"{projected_h:.2f}h for {len(rows)} rows (fence {args.fence_hours}h) -> {out}",
        flush=True,
    )
    if report["over_fence"]:
        print(
            f"[phase=pilot] DESIGNED HALT rc={PILOT_FENCE_RC}: projection "
            f"{projected_h:.2f}h exceeds fence {args.fence_hours}h",
            flush=True,
        )
        return PILOT_FENCE_RC
    return 0


# ---------------------------------------------------------------------------
# bit-equality gate (leg 1 single-turn reuse)
# ---------------------------------------------------------------------------


def bit_equality_gate(args, tokenizer, model, sample: list[dict]) -> dict:
    """Re-capture a sample of SINGLE-TURN wcrung rows; require rep equivalence.

    Single-turn rows render byte-identically bare (their original render already
    had no prefix), so their existing wcrung reps ARE their bare reps — at the
    INPUT level. At the OUTPUT level exact array equality cannot be required:
    bf16 padded-batch kernel numerics differ with batch composition even for
    byte-identical input tokens (single-position states jitter ~1e-6, amplified
    in deep layers — the #779 calibration in gotchas), and this re-capture
    necessarily batches differently than the committed wcrung capture did. The
    gate is therefore the calibrated two-bar cosine: per-row EARLY-layer cosine
    >= 0.999 over the first 4 layers (mask/render/row-mapping bugs corrupt
    layer 0 immediately, reading ~0.4-0.86) AND flattened all-layer cosine
    >= 0.995 (>=4x headroom over measured worst bf16 deviation). The
    bit-identical count is retained as informational output only.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    if not sample:
        return {"ran": False, "reason": "no single-turn sample supplied"}
    probe_root = args.out_root / "bit_gate"
    probe_args = argparse.Namespace(**{**vars(args), "store_root": probe_root, "leg": "1"})
    run_capture(probe_args, sample, tokenizer, model)
    layers = tuple(range(args.n_layers))
    fresh, fresh_meta = store_io.load_summaries(
        probe_root / "bareq", (BARE_KIND,), layers, hidden_dim=args.hidden_dim
    )
    wc_store = Path(args.wcrung_store)
    if not wc_store.exists():
        # The committed wcrung capture store is HF-resident (6 GB, 1,721 files at
        # issue1739_ctxmap/wildchat_rung/capture_store/wildchat) — a fresh clone
        # has no local copy (att-20260731-150348 crashed here). Reuse the arms
        # runner's test-pinned staging (skip-if-present, probe-verified).
        from scripts.issue1739_wcrung_arms_run import stage_wcrung_store

        wc_store = stage_wcrung_store(argparse.Namespace(store_root=args.stage_root))
        print(f"[bareq] staged wcrung capture store from HF -> {wc_store}", flush=True)
    stored, stored_meta = store_io.load_summaries(
        wc_store, (BARE_KIND,), layers, hidden_dim=args.hidden_dim
    )
    key = "context_id"
    pos = {r.get(key): i for i, r in enumerate(stored_meta)}
    n_early = min(4, len(layers))
    checked, exact = 0, 0
    worst_early, worst_flat = 1.0, 1.0

    def _cos(u: np.ndarray, v: np.ndarray) -> float:
        u = u.astype(np.float64).ravel()
        v = v.astype(np.float64).ravel()
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        return float(u @ v / denom) if denom else 0.0

    for i, r in enumerate(fresh_meta):
        cid = str(r.get(key, "")).removeprefix("wc-")
        j = pos.get(cid)
        if j is None:
            continue
        checked += 1
        a = np.stack([fresh[(BARE_KIND, ly)][i] for ly in layers])
        b = np.stack([stored[(BARE_KIND, ly)][j] for ly in layers])
        if np.array_equal(a, b):
            exact += 1
        early = min(_cos(a[k], b[k]) for k in range(n_early))
        flat = _cos(a, b)
        worst_early = min(worst_early, early)
        worst_flat = min(worst_flat, flat)
    passed = bool(checked) and worst_early >= 0.999 and worst_flat >= 0.995
    result = {
        "ran": True,
        "n_sampled": len(sample),
        "n_joined": checked,
        "n_bit_identical": exact,
        "worst_early_layer_cos": worst_early,
        "worst_flattened_cos": worst_flat,
        "gate": (
            "two-bar cosine (early-layer >= 0.999 over first "
            f"{n_early} layers, flattened >= 0.995; bf16 batch-composition "
            "jitter licensed per the #779 calibration) — bit-identical count informational"
        ),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(
            f"single-turn reuse gate FAILED: worst early-layer cos {worst_early:.6f} "
            f"(bar 0.999) / worst flattened cos {worst_flat:.6f} (bar 0.995) over "
            f"{checked} joined rows — the 987 single-turn reps may NOT be reused as bare reps"
        )
    print(
        f"[phase=bit_gate] PASS {checked} rows: worst early cos {worst_early:.6f}, "
        f"worst flat cos {worst_flat:.6f} ({exact} bit-identical)",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# upload + CLI
# ---------------------------------------------------------------------------


def upload_dir(local: Path, path_in_repo: str, *, skip: bool) -> None:
    if skip:
        print(f"[phase=upload] SKIPPED {path_in_repo}", flush=True)
        return
    from explore_persona_space.orchestrate import hub

    hub._upload(local, hub.DEFAULT_DATASET_REPO, "dataset", path_in_repo, raise_on_error=True)
    print(f"[phase=upload] {local} -> {path_in_repo}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="all", choices=("extract", "capture", "all"))
    ap.add_argument("--leg", default="both", choices=("1", "2", "both"))
    ap.add_argument("--behavior", default="evil", help="leg-2 pool (only evil is prefixed)")
    ap.add_argument(
        "--rung",
        default=TRAIN_RUNG,
        help=(
            f"leg-2 rung to capture: {TRAIN_RUNG} (legacy bank + store) or an OOD eval rung "
            "(hhrt / toxicchat) whose bank the extract phase wrote"
        ),
    )
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/bareq_map"))
    ap.add_argument(
        "--store-root", type=Path, default=Path("analysis_tensors/issue_1739/bareq_store")
    )
    ap.add_argument("--stage-root", type=Path, default=Path("data/issue_1739/bareq_stage"))
    # UPLOAD_PREFIX_EXEMPT: bare-query-round-specific leg; --hf-prefix overrides
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument(
        "--wcrung-rows-json",
        type=Path,
        default=Path("eval_results/issue_1739/wildchat_rung/contexts/wcrung.json"),
        help="leg-1 source rows (the wildchat rung's sampled contexts)",
    )
    ap.add_argument(
        "--wcrung-store",
        type=Path,
        default=Path("analysis_tensors/issue_1739/wcrung_store/wildchat"),
        help="committed wcrung capture store, for the bit-equality gate",
    )
    ap.add_argument("--multi-turn-only", action="store_true", default=True)
    ap.add_argument(
        "--include-single-turn",
        dest="multi_turn_only",
        action="store_false",
        help="also capture the 987 single-turn rows (normally REUSED, not re-captured)",
    )
    ap.add_argument("--train-only", action="store_true", default=True)
    ap.add_argument("--all-rungs", dest="train_only", action="store_false")
    ap.add_argument("--max-shards", type=int, default=64)
    ap.add_argument("--reap-shards", action="store_true", default=True)
    ap.add_argument("--keep-shards", dest="reap_shards", action="store_false")
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--hidden-dim", type=int, default=3584)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--capture-batch-size", type=int, default=None)
    ap.add_argument("--fingerprint", default=None)
    ap.add_argument("--pilot-only", action="store_true", help="measure one batch, project, exit")
    ap.add_argument("--pilot-rows", type=int, default=32)
    ap.add_argument(
        "--fence-hours",
        type=float,
        default=None,
        help=f"projection over this -> designed halt rc={PILOT_FENCE_RC}",
    )
    ap.add_argument("--bit-gate-rows", type=int, default=20)
    ap.add_argument("--skip-bit-gate", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.leg == "2" and args.behavior != "evil":
        ap.error(
            f"--leg 2 --behavior {args.behavior}: only evil's pool carries a prefix "
            "(sycophancy/hallucination train contexts are already bare renders — no-op)"
        )
    if args.rung != TRAIN_RUNG and args.leg != "2":
        ap.error(
            f"--rung {args.rung} requires --leg 2: leg 1 is the wildchat rung's own contexts, "
            "which carry no per-rung query bank"
        )
    return args


def _import_check() -> int:
    """Resolve every deferred import on the REAL branch, in its OWN function.

    Deliberately not inline in ``main()``: an ``import X`` is a binding, so an
    inline block would make X a function-wide local of ``main()`` and shadow any
    module-level symbol of the same name on the normal path (the #1739 wcrung
    ``capture`` UnboundLocalError). Pinned by the shadow test.
    """
    from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
        capture,
        constants,
        store_io,
    )
    from explore_persona_space.experiments.issue_1739.capture import (  # noqa: F401
        capture_rollout_files,
        load_capture_model,
    )
    from explore_persona_space.experiments.issue_1739.constants import (  # noqa: F401
        HIDDEN_DIM,
        N_LAYERS,
    )
    from explore_persona_space.experiments.issue_1739.generation import get_tokenizer  # noqa: F401
    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        DEFAULT_DATASET_REPO,
        _upload,
        stage_hub_file,
    )
    from scripts.issue1739_wcrung_contexts import render_row_prompt  # noqa: F401

    print("[import-check] OK: all deferred imports resolved", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    if args.import_check:
        return _import_check()

    from explore_persona_space.experiments.issue_1739 import capture as capture_mod
    from explore_persona_space.experiments.issue_1739.generation import get_tokenizer

    token = os.environ.get("HF_TOKEN") or ""
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.store_root.mkdir(parents=True, exist_ok=True)
    args.stage_root.mkdir(parents=True, exist_ok=True)

    if args.phase in ("extract", "all") and args.leg in ("2", "both"):
        if not token:
            raise SystemExit("HF_TOKEN missing — the extract phase stages from the data repo")
        extract_query_bank(args, token)
    if args.phase == "extract":
        print("[phase=done] extract-only complete", flush=True)
        return 0

    tokenizer = get_tokenizer()
    rows = build_capture_rows(args, tokenizer)
    null_probe = _null_probe_report(rows, tokenizer)
    print(
        f"[phase=capture] rows={len(rows)} kinds="
        f"{sorted({r['kind'] for r in rows})} constant_prefix_tokens={null_probe['prefix_token_len']}",
        flush=True,
    )

    model = capture_mod.load_capture_model(device=args.device)

    if args.pilot_only:
        return run_pilot(args, rows, tokenizer, model)

    gate = {"ran": False, "reason": "skipped"}
    if not args.skip_bit_gate and args.leg in ("1", "both"):
        singles = [
            r for r in build_capture_rows(_single_turn_args(args), tokenizer) if not r["multi_turn"]
        ]
        gate = bit_equality_gate(args, tokenizer, model, singles[: args.bit_gate_rows])

    cap = run_capture(args, rows, tokenizer, model)
    upload_dir(args.store_root, f"{args.hf_prefix}/capture_store", skip=args.skip_upload)
    upload_dir(args.out_root, f"{args.hf_prefix}/manifests", skip=args.skip_upload)

    sentinel = {
        "leg": f"bareq_{args.leg}",
        "rung": RUNG,
        "dv_rung": args.rung,
        "behavior": args.behavior,
        "leg2_store_dir": str(leg2_store_dir(args)) if args.leg == "2" else None,
        "leg2_query_manifest": leg2_manifest_path(args).name if args.leg in ("2", "both") else None,
        "n_rows_captured": cap.get("n_rows"),
        "per_row_s": cap.get("per_row_s"),
        "null_probe": null_probe,
        "bit_equality_gate": gate,
        "scope_note": (
            "sycophancy/hallucination train pools are ALREADY bare renders "
            "(prefix_end constant at the template head; committed prefix arms 0.003/0.059) — "
            "their existing arm4_ridge_ctx IS the bare-query map, so leg 2 is evil-only"
        ),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # One sentinel per (leg, rung): an OOD rung must not clobber the train run's.
    sentinel_name = (
        SENTINEL_NAME
        if args.rung == TRAIN_RUNG
        else SENTINEL_NAME.replace(".json", f"_{args.behavior}_{args.rung}.json")
    )
    _write_json_atomic(args.out_root / sentinel_name, sentinel)
    print("[phase=done] bareq capture complete", flush=True)
    return 0


def _single_turn_args(args) -> argparse.Namespace:
    """A copy of args that yields the single-turn wcrung rows (gate source)."""
    return argparse.Namespace(**{**vars(args), "leg": "1", "multi_turn_only": False})


if __name__ == "__main__":
    sys.exit(main())
