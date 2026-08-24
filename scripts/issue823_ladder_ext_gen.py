#!/usr/bin/env python3
"""P-Gen-ext driver for task #823 follow-up round `origin-ladder-more-contexts`.

Extension-generation phase (plan v17 sections 4.2 / 4.3 P-Gen-ext): a NEW thin
caller importing the landed primitives from ``scripts/issue823_ladder_gen.py``
(the parent driver's own ``N_CONTEXTS_FULL = 5000`` pins are why this is a new
caller rather than a flag on the parent ``main()``). VM-side, Batch API, runs
BEFORE any pod exists.

Pipeline (each step maps to a plan section 4.3 P-Gen-ext step):

1. Stream-select the 43,000 extension prompts off ``lmsys/lmsys-chat-1m``
   (streaming) with the round-1 ``_first_user_turn`` predicate: the first
   5,000 non-empty first-turns are asserted to REPRODUCE the parent's frozen
   context set (banked ``b2_seed42.json`` @ the section 10 pin; ordered
   equality expected, set equality the registered fallback — on
   order-mismatch-but-set-match the permutation is REPORTED and the run
   proceeds), then the NEXT 43,000 distinct raw-string prompts disjoint from
   the first set. Persist ``sampling_manifest_ext.json`` +
   ``sampling_prompts_ext.jsonl`` (prompts + stream positions + shas);
   fingerprint-gated manifest resume skips the re-stream.
2. Assignment persona(i, k) = i mod k for arms {1, 16} over all 48,000
   contexts -> ``assignment_ext.json``; the 5,000-row prefix is asserted equal
   to the banked ``assignment.json`` arms "1"/"16"; the roster is LOADED from
   the banked ``roster.json`` (byte-asserted vs the parent module constants).
3. 83,313 unique extension (context, persona) DispatchItems (43,000 k1 +
   40,313 nonzero-residue k16; 2,687 i%16==0 contexts share the persona-0
   pair) — realized count asserted.
4. Gate A pilot: the FIRST 200 pairs dispatched alone at the exact production
   instrument; per-context 2-arm survival >= 85% and stop_reason==max_tokens
   fraction < 2% gate the wave; a band miss is a designed halt with a
   persona-keyed refusal-attribution report BEFORE the 83k dispatch. Pilot
   rows are production rows (reused, never discarded).
5. Remaining wave batch-routed (checkpoint_dir ``data/issue_823/
   ladder_ext_gen/`` on the VM; crash-recovery by checkpointed batch ids;
   bounded transport re-drives, parent semantics). ONE regen round at 8192
   for any (arm x persona) cell with cap-hit > 2%; post-regen convergence
   RE-MEASURED on final records.
6. Upload raw completions + manifest + ``gen_digest_ext.json`` to HF
   ``issue823_inconsistent_origin_ladder/raw_completions/ladder_ext/``
   (non-LFS text; >9.5 MB files line-split into <9 MB shards, never gzip);
   validity classes ok / refusal / cap_hit from the persisted stop_reason.

Designed-abort exit codes (plan section 7 kill criteria; each writes a report
JSON before exiting, and NO completion sentinel exists after any halt):

    rc | meaning
    ---+------------------------------------------------------------------
     0 | complete: records + digest + sentinel written, canonical upload
       | verified on the canonical data repo
     3 | transport-class rows remain after the bounded re-drives (pilot,
       | wave, or pooled-regen stage) — parent exit-3 semantics
     4 | generation-config fingerprint mismatch on a checkpoint resume
       | (parent gate inside issue823_ladder_gen._dispatch)
     6 | LMSYS selection drift: normalized ctx0 != EXPECTED_CTX0_PROMPT, or
       | the banked 5,000-prefix reproduction failed BOTH ordered and set
       | equality (plan kill criterion 7)
     7 | LMSYS stream exhausted / max-stream-pos bound (2,000,000) hit
       | before 43,000 extension prompts were collected (kill criterion 7)
     8 | Gate A survival < 85% — halt the wave build; persona-keyed refusal
       | attribution report written first
     9 | Gate A cap-hit fraction >= 2% — raise-base-cap decision needed
       | BEFORE the wave (parent precedent: 4096 was pilot-set)
    10 | banked-artifact parity mismatch: roster byte-assert vs the parent
       | module constants, or the assignment 5,000-row prefix != banked
    11 | sampling-manifest resume fingerprint / sha mismatch (a selection
       | config change across a resume — never resume across one)

Usage:
  uv run python scripts/issue823_ladder_ext_gen.py --smoke   # 16 ext contexts, /tmp + _smoke prefix
  uv run python scripts/issue823_ladder_ext_gen.py           # full 43,000-context production run
  uv run python scripts/issue823_ladder_ext_gen.py --list-rcs

Resume safety: every dispatcher checkpoint dir carries the parent
generation-config fingerprint gate (issue823_ladder_gen.check_or_persist_
gen_config, run inside ``_dispatch`` BEFORE any dispatch or row re-serve);
the selection is additionally fingerprint-gated through
``sampling_manifest_ext.json`` (rc 11 on mismatch), so a resumed run never
re-streams LMSYS nor re-buys completed Batch rows.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue823_ladder_ext_gen.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# creds + shared-VM thread caps BEFORE any heavy import (unit-1 lesson: the
# dotenv-before-numpy/torch preamble; this module imports neither, but the
# parent module chain must see the caps first regardless).
load_dotenv()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.llm.api_dispatch import DispatchItem, DispatchResult  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    _upload_folder_filtered,
    retry_transient,
)
from scripts import issue823_ladder_gen as GEN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_ladder_ext_gen")

# ── Registered constants (plan v17 sections 4.2 / 4.3 / 10) ─────────────────
EXT_ARMS: tuple[int, ...] = (1, 16)  # the round's two arms (plan section 4.2)
N_PREFIX = GEN.N_CONTEXTS_FULL  # 5,000 banked prefix contexts (ids 0-4,999)
N_EXT_FULL = 43_000  # extension contexts (ids 5,000-47,999)
N_TOTAL_FULL = N_PREFIX + N_EXT_FULL  # 48,000
# 43,000 k1 pairs + 40,313 nonzero-residue k16 pairs (2,687 multiples of 16
# in [5,000, 48,000) share the persona-0 pair): plan section 4.2 arithmetic.
REGISTERED_EXT_PAIRS = 83_313
MAX_STREAM_POS_DEFAULT = 2_000_000  # designed depth bound (plan section 4.2)
PILOT_N_PAIRS = 200  # Gate A: the FIRST 200 pairs (plan section 4.3 step 2)
# Gate A bands (plan section 7 row A). Calibration sources (observed-side,
# parent-MEASURED — never null-derived): survival floor 0.85 vs the parent
# pilot's 93.7% per-context survival (pilot300_refusal_attribution.json);
# cap-hit max 0.02 vs the parent's realized 7/14,996 = 0.047% at 4096
# (gen_digest.json), same 2% band as the per-cell regen trigger.
GATE_A_SURVIVAL_FLOOR = 0.85
GATE_A_CAP_HIT_MAX = 0.02
# Banked round-1 artifacts (roster.json / assignment.json) tree pin — the
# parent body's own pinned revision (plan section 10 reuse table).
BANKED_LADDER_REV = "009b58fdcf3da303993695066870e29416fb9ef6"
BANKED_LADDER_PATH = f"{GEN.HF_PREFIX}/raw_completions/ladder"
HF_EXT_SUBPATH = "raw_completions/ladder_ext"
SELECTION_RULE_ID = "issue823_ext_selection_v1_first_user_turn_disjoint"
LMSYS_REPO = "lmsys/lmsys-chat-1m"
# Restated from scripts/issue779_ffc_n10k_generate_capture.py:86 (importing
# that module would pull its torch/vllm-adjacent import graph onto the VM);
# tests/test_issue823_ext_gen.py pins source parity mechanically via ast.
EXPECTED_CTX0_PROMPT = "how can identity protection services help protect me against identity theft"
# Line-shard limits restated from scripts/issue2054_phase_a.py:135-136 (the
# upload-policy.md >9.5 MB line-split recipe; manifest schema pinned by
# orchestrate.hub._parse_shard_manifest).
UPLOAD_SHARD_LIMIT_BYTES = 9_500_000
UPLOAD_SHARD_TARGET_BYTES = 9_000_000

# Designed-abort exit codes (docstring table is the canonical enumeration).
EXIT_TRANSPORT_RESIDUE = 3  # parent semantic (issue823_ladder_gen exit 3)
EXIT_STREAM_DRIFT = 6
EXIT_STREAM_EXHAUSTED = 7
EXIT_GATE_A_SURVIVAL = 8
EXIT_GATE_A_CAP_HIT = 9
EXIT_BANKED_PARITY = 10
EXIT_MANIFEST_MISMATCH = 11

MANIFEST_FILENAME = "sampling_manifest_ext.json"
PROMPTS_FILENAME = "sampling_prompts_ext.jsonl"
SENTINEL_FILENAME = "_gen_ext_complete.json"
# Mid-stream selection checkpoints (r1 concern external-stream-not-resumable;
# the #1092 _stream_with_cache pattern: durable kept-pool chunks + a meta
# sidecar written LAST, exact-fingerprint resume). LOCAL-ONLY scratch — the
# durable record stays the sampling manifest; these never upload.
PROGRESS_FILENAME = "sampling_progress_ext.jsonl"
PROGRESS_META_FILENAME = "sampling_progress_ext.meta.json"
PROGRESS_CHUNK = 2_000  # kept prompts per durable flush


class DesignedHalt(SystemExit):
    """A plan-registered designed abort: report written, distinct rc, no sentinel."""

    def __init__(self, rc: int, report_path: pathlib.Path):
        super().__init__(rc)
        self.report_path = report_path


def _halt(rc: int, report_path: pathlib.Path, report: dict, message: str) -> None:
    GEN.write_json(report_path, report)
    logger.error("HALT-AND-REPORT (rc=%d): %s; report at %s", rc, message, report_path)
    raise DesignedHalt(rc, report_path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_prompt_list(prompts: list[str]) -> str:
    """Ordered-list sha (the issue779_ffc_n10k_generate_capture._sha_prompts pattern)."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ── Selection (plan section 4.2 registered rule) ─────────────────────────────


def _first_user_turn(row) -> str | None:
    """Round-1's exact predicate — restated verbatim from
    scripts/issue779_ffc_n10k_generate_capture.py:99 (issue779_collect.
    load_train_contexts lineage); ast-parity-pinned by the committed test."""
    val = row.get("conversation")
    if isinstance(val, list) and val and isinstance(val[0], dict):
        p = val[0].get("content") or val[0].get("value")
        return p.strip() if isinstance(p, str) and p.strip() else None
    return None


def stream_lmsys_rows(skip: int = 0):
    """The real LMSYS stream (deferred datasets import; tests inject stream_iter).

    ``skip`` fast-forwards the stream to position ``skip`` (the mid-stream
    resume path) via ``IterableDataset.skip`` — the resumed iterator's first
    row is stream position ``skip``.
    """
    from datasets import load_dataset

    ds = load_dataset(LMSYS_REPO, split="train", streaming=True)
    if skip:
        ds = ds.skip(skip)
    return iter(ds)


def _append_stream_progress(stage_dir: pathlib.Path, new_rows: list[dict], meta: dict) -> None:
    """Durable mid-stream checkpoint: append the newly-kept rows, fsync, THEN
    atomically rewrite the meta sidecar (sidecar-last ordering: a crash between
    the two leaves extra tail rows the loader truncates to ``meta.n_kept``)."""
    pp = stage_dir / PROGRESS_FILENAME
    pp.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows)
    with pp.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    GEN.write_json(stage_dir / PROGRESS_META_FILENAME, meta)


def load_stream_progress(stage_dir: pathlib.Path, fields: dict) -> dict | None:
    """Fingerprint-gated mid-stream resume state, or None.

    Returns {rows, prefix_report, next_stream_pos} when the meta sidecar
    matches ``fields`` and the kept rows validate; DISCARDS (warn + None) on
    any mismatch/corruption — partial progress is re-derivable scratch, so
    discard-and-restream is safe here (the rc-11 designed halt governs the
    COMPLETED manifest, not this scratch)."""
    meta_path = stage_dir / PROGRESS_META_FILENAME
    pp = stage_dir / PROGRESS_FILENAME
    if not meta_path.exists() or not pp.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("stream-progress meta unreadable — discarding partial progress")
        return None
    if meta.get("fingerprint_fields") != fields:
        logger.warning("stream-progress fingerprint mismatch — discarding partial progress")
        return None
    rows: list[dict] = []
    # split("\n"), never splitlines(): real-corpus prompt text (#950).
    for line in pp.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # torn tail from a mid-append crash: the intact head suffices
    n_kept = int(meta["n_kept"])
    if len(rows) < n_kept:
        logger.warning(
            "stream-progress rows short (%d < meta n_kept %d) — discarding", len(rows), n_kept
        )
        return None
    rows = rows[:n_kept]
    for j, r in enumerate(rows):
        if r.get("ext_idx") != j or _sha256_text(r["prompt"]) != r.get("sha256"):
            logger.warning("stream-progress row %d fails validation — discarding", j)
            return None
    return {
        "rows": rows,
        "prefix_report": meta["prefix_report"],
        "next_stream_pos": int(meta["next_stream_pos"]),
    }


def select_extension_contexts(
    banked_questions: list[str],
    n_ext: int,
    *,
    max_stream_pos: int,
    eval_dir: pathlib.Path,
    stream_iter=None,
    stage_dir: pathlib.Path | None = None,
    fingerprint_fields: dict | None = None,
) -> dict:
    """Registered selection rule (plan section 4.2), fail-loud at every seam.

    Phase 1: the first ``len(banked_questions)`` non-empty first-turns are
    asserted to reproduce the banked prefix (ordered expected; set equality
    the registered fallback — permutation REPORTED and the run proceeds).
    Phase 2: the NEXT ``n_ext`` distinct raw-string prompts disjoint from the
    prefix set. Halts: rc 6 on ctx0 / prefix drift, rc 7 on stream
    exhaustion or the ``max_stream_pos`` depth bound.

    Prompt text NEVER enters halt reports (real-world-corpus hygiene) —
    reports carry positions + sha256 digests only.

    With ``stage_dir`` + ``fingerprint_fields`` set, phase 2 checkpoints every
    ``PROGRESS_CHUNK`` kept prompts to a local-only progress file + meta
    sidecar and RESUMES from it (fingerprint-gated; stream fast-forwarded to
    the persisted position) — the #1092 external-stream checkpoint contract.
    """
    n_prefix = len(banked_questions)
    resumed = (
        load_stream_progress(stage_dir, fingerprint_fields)
        if stage_dir is not None and fingerprint_fields is not None
        else None
    )
    start_pos = resumed["next_stream_pos"] if resumed is not None else 0
    if stream_iter is not None:
        it = stream_iter
        for _ in range(start_pos):
            try:
                next(it)  # injected iterators fast-forward by consumption
            except StopIteration:
                break  # phase-2 _next_row halts rc 7 with the full report
    else:
        it = stream_lmsys_rows(skip=start_pos)
    prefix: list[str] = []
    ext: list[str] = []
    ext_positions: list[int] = []
    pos = start_pos - 1
    if resumed is not None:
        ext = [r["prompt"] for r in resumed["rows"]]
        ext_positions = [int(r["stream_pos"]) for r in resumed["rows"]]
        assert stage_dir is not None
        # Truncate any torn tail beyond meta.n_kept so later appends can never
        # be shadowed by stale same-ext_idx rows earlier in the file.
        pp = stage_dir / PROGRESS_FILENAME
        tmp = pp.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in resumed["rows"]),
            encoding="utf-8",
        )
        os.replace(tmp, pp)

    def _next_row():
        nonlocal pos
        pos += 1
        if pos > max_stream_pos:
            _halt(
                EXIT_STREAM_EXHAUSTED,
                eval_dir / "ext_stream_exhausted_report.json",
                {
                    "reason": "max_stream_pos_bound_hit",
                    "max_stream_pos": max_stream_pos,
                    "n_prefix_collected": len(prefix),
                    "n_ext_collected": len(ext),
                    "n_ext_target": n_ext,
                },
                f"max-stream-pos bound {max_stream_pos} hit with "
                f"{len(ext)}/{n_ext} extension prompts collected",
            )
        try:
            return next(it)
        except StopIteration:
            _halt(
                EXIT_STREAM_EXHAUSTED,
                eval_dir / "ext_stream_exhausted_report.json",
                {
                    "reason": "stream_exhausted",
                    "last_stream_pos": pos - 1,
                    "n_prefix_collected": len(prefix),
                    "n_ext_collected": len(ext),
                    "n_ext_target": n_ext,
                },
                f"LMSYS stream exhausted at pos {pos - 1} with "
                f"{len(ext)}/{n_ext} extension prompts collected",
            )

    if resumed is None:
        # Phase 1: first n_prefix non-empty first-turns (round-1 derivation — no
        # within-phase dedup, mirroring issue779_ffc_n10k sample_disjoint phase 1).
        while len(prefix) < n_prefix:
            p = _first_user_turn(_next_row())
            if p:
                if not prefix and p != EXPECTED_CTX0_PROMPT:
                    _halt(
                        EXIT_STREAM_DRIFT,
                        eval_dir / "ext_selection_drift_report.json",
                        {
                            "reason": "ctx0_mismatch",
                            "expected_ctx0_sha256": _sha256_text(EXPECTED_CTX0_PROMPT),
                            "got_ctx0_sha256": _sha256_text(p),
                            "stream_pos": pos,
                        },
                        "normalized ctx0 != EXPECTED_CTX0_PROMPT (stream-ordering drift)",
                    )
                prefix.append(p)

        # Prefix reproduction: ordered expected; set equality the registered
        # fallback (report the permutation and proceed); neither -> rc 6.
        ordered_equal = prefix == banked_questions
        set_equal = set(prefix) == set(banked_questions)
        mismatch_positions = (
            []
            if ordered_equal
            else [i for i, (a, b) in enumerate(zip(prefix, banked_questions)) if a != b]
        )
        if not ordered_equal and not set_equal:
            stream_only = set(prefix) - set(banked_questions)
            banked_only = set(banked_questions) - set(prefix)
            _halt(
                EXIT_STREAM_DRIFT,
                eval_dir / "ext_selection_drift_report.json",
                {
                    "reason": "prefix_reproduction_failed",
                    "ordered_equal": False,
                    "set_equal": False,
                    "n_position_mismatches": len(mismatch_positions),
                    "n_stream_only": len(stream_only),
                    "n_banked_only": len(banked_only),
                    "stream_only_sha256_first5": sorted(_sha256_text(s) for s in stream_only)[:5],
                    "banked_only_sha256_first5": sorted(_sha256_text(s) for s in banked_only)[:5],
                },
                "banked 5,000-prefix reproduction failed BOTH ordered and set equality",
            )
        if not ordered_equal:
            logger.warning(
                "Prefix reproduction: ORDER mismatch at %d positions but SET equality holds — "
                "registered fallback engaged (extension needs only disjointness + a frozen "
                "extension order); permutation recorded in sampling_manifest_ext.json",
                len(mismatch_positions),
            )
        prefix_report = {
            "ordered_equal": ordered_equal,
            "set_equal": set_equal,
            "n_position_mismatches": len(mismatch_positions),
            "mismatch_positions_first20": mismatch_positions[:20],
            "n_prefix": n_prefix,
        }
        # Phase 2 exclusion: the banked set AND the re-derived prefix set
        # (equal in both accepted branches; the union is the belt for the
        # permutation case).
        excluded = set(banked_questions) | set(prefix)
    else:
        prefix_report = resumed["prefix_report"]
        # Past phase 1 both accepted branches guarantee set(prefix) ==
        # set(banked_questions), so the exclusion set reconstructs exactly.
        excluded = set(banked_questions)
        logger.info(
            "Stream progress resumed: %d/%d extension prompts kept (next stream pos %d)",
            len(ext),
            n_ext,
            start_pos,
        )

    if resumed is None and stage_dir is not None:
        # A discarded/mismatched prior progress file must never be appended
        # onto: stale rows ahead of fresh ones would trip the loader's
        # positional validation on the NEXT resume (discard + full restream).
        # A fresh phase 2 starts from a clean progress surface.
        for name in (PROGRESS_FILENAME, PROGRESS_META_FILENAME):
            (stage_dir / name).unlink(missing_ok=True)

    # Phase 2: next n_ext DISTINCT raw-string prompts, disjoint from the
    # exclusion set; kept pool checkpointed every PROGRESS_CHUNK (#1092).
    taken: set[str] = set(ext)
    n_flushed = len(ext)
    t_sel = time.monotonic()

    def _flush_progress() -> None:
        nonlocal n_flushed
        if stage_dir is None or fingerprint_fields is None or len(ext) == n_flushed:
            return
        new_rows = [
            {
                "ext_idx": j,
                "context_id": N_PREFIX + j,
                "stream_pos": ext_positions[j],
                "sha256": _sha256_text(ext[j]),
                "prompt": ext[j],
            }
            for j in range(n_flushed, len(ext))
        ]
        _append_stream_progress(
            stage_dir,
            new_rows,
            {
                "fingerprint_fields": fingerprint_fields,
                "prefix_report": prefix_report,
                "n_kept": len(ext),
                "next_stream_pos": pos + 1,
            },
        )
        n_flushed = len(ext)
        # Canonical long-loop shape `[<phase>] unit k/N <key> elapsed=<s>s`
        # (code-style.md per-unit progress line; r3 concern
        # long-loop-progress-fields-missing) — pinned by
        # tests/test_issue823_ext_gen.py::test_progress_line_canonical_unit_shape.
        logger.info(
            "[gen_ext] unit %d/%d stream-pos-%d elapsed=%.1fs",
            len(ext),
            n_ext,
            pos,
            time.monotonic() - t_sel,
        )

    while len(ext) < n_ext:
        p = _first_user_turn(_next_row())
        if p and p not in excluded and p not in taken:
            ext.append(p)
            ext_positions.append(pos)
            taken.add(p)
            if len(ext) - n_flushed >= PROGRESS_CHUNK:
                _flush_progress()
    # Final tail flush: the whole kept pool is durable BEFORE the caller's
    # manifest write (a crash between the two resumes from here for free).
    _flush_progress()
    assert set(ext).isdisjoint(excluded), "extension prompts overlap the prefix set"
    assert len(set(ext)) == len(ext), "extension prompts are not distinct"
    logger.info(
        "Selection complete: %d prefix (ordered_equal=%s) + %d extension prompts "
        "(last stream pos %d)",
        n_prefix,
        prefix_report["ordered_equal"],
        len(ext),
        pos,
    )
    return {
        "prefix_report": prefix_report,
        "ext_prompts": ext,
        "ext_positions": ext_positions,
        "last_stream_pos": pos,
    }


def selection_fingerprint_fields(n_ext: int, max_stream_pos: int) -> dict:
    """The manifest resume fingerprint (rc 11 on any field drift across a resume)."""
    return {
        "selection_rule_id": SELECTION_RULE_ID,
        "lmsys_repo": LMSYS_REPO,
        "n_prefix": N_PREFIX,
        "n_ext": n_ext,
        "max_stream_pos": max_stream_pos,
        "parent_rev": GEN.PARENT_REV,
        "banked_ladder_rev": BANKED_LADDER_REV,
        "expected_ctx0_sha256": _sha256_text(EXPECTED_CTX0_PROMPT),
    }


def write_sampling_manifest(
    stage_dir: pathlib.Path, selection: dict, fields: dict, metadata: dict
) -> None:
    """Persist the selection durably (prompts + stream positions + shas) BEFORE
    any dispatch — the checkpoint the fingerprint-gated resume reloads."""
    prompts = selection["ext_prompts"]
    positions = selection["ext_positions"]
    lines = []
    for j, (p, sp) in enumerate(zip(prompts, positions)):
        lines.append(
            json.dumps(
                {
                    "ext_idx": j,
                    "context_id": N_PREFIX + j,
                    "stream_pos": sp,
                    "sha256": _sha256_text(p),
                    "prompt": p,
                },
                ensure_ascii=False,
            )
        )
    prompts_path = stage_dir / PROMPTS_FILENAME
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    # Invalidate any prior shard set BEFORE replacing the source bytes: a crash
    # between this write and the upload-time re-shard must leave (new source,
    # NO manifest/shards), never (new source, stale shards) — the precedence
    # trap the OTHER source-rewriting writers already guard (r2 finding d,
    # missed path of r1 concern stale-manifest-precedence-regen).
    _unlink_stale_shards(prompts_path)
    tmp = prompts_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, prompts_path)
    manifest = {
        "metadata": metadata,
        "fingerprint_fields": fields,
        "prefix_reproduction": selection["prefix_report"],
        "n_ext": len(prompts),
        "last_stream_pos": selection["last_stream_pos"],
        "ext_positions": positions,
        "ext_prompt_sha256": [_sha256_text(p) for p in prompts],
        "ext_prompts_ordered_sha256": _sha_prompt_list(prompts),
        "prompts_file": PROMPTS_FILENAME,
    }
    GEN.write_json(stage_dir / MANIFEST_FILENAME, manifest)


def load_selection_if_persisted(
    stage_dir: pathlib.Path, fields: dict, eval_dir: pathlib.Path
) -> dict | None:
    """Fingerprint-gated selection resume: None when no manifest exists; the
    reloaded selection when the manifest matches; rc 11 designed halt on any
    fingerprint-field or sha mismatch (never resume across a config change)."""
    manifest_path = stage_dir / MANIFEST_FILENAME
    prompts_path = stage_dir / PROMPTS_FILENAME
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    persisted = manifest.get("fingerprint_fields", {})
    if persisted != fields:
        differing = sorted(
            k for k in set(persisted) | set(fields) if persisted.get(k) != fields.get(k)
        )
        _halt(
            EXIT_MANIFEST_MISMATCH,
            eval_dir / "ext_manifest_mismatch_report.json",
            {
                "reason": "fingerprint_fields_mismatch",
                "differing_fields": differing,
                "persisted": persisted,
                "live": fields,
            },
            f"sampling-manifest fingerprint mismatch on resume (fields: {differing})",
        )
    if not prompts_path.is_file():
        _halt(
            EXIT_MANIFEST_MISMATCH,
            eval_dir / "ext_manifest_mismatch_report.json",
            {"reason": "prompts_file_missing", "expected": str(prompts_path)},
            "sampling manifest present but its prompts file is missing",
        )
    # split("\n"), NEVER splitlines(): real-corpus prompt text carries raw
    # U+2028/U+2029/NEL inside JSON strings and splitlines() shreds those rows
    # (#950 reader class; r1 finding unicode-jsonl-resume).
    rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    prompts = [r["prompt"] for r in rows]
    positions = [r["stream_pos"] for r in rows]
    ok = (
        len(prompts) == manifest["n_ext"]
        and [_sha256_text(p) for p in prompts] == manifest["ext_prompt_sha256"]
        and _sha_prompt_list(prompts) == manifest["ext_prompts_ordered_sha256"]
        and positions == manifest["ext_positions"]
    )
    if not ok:
        _halt(
            EXIT_MANIFEST_MISMATCH,
            eval_dir / "ext_manifest_mismatch_report.json",
            {
                "reason": "prompts_file_sha_mismatch",
                "n_rows": len(prompts),
                "n_expected": manifest["n_ext"],
            },
            "persisted prompts file disagrees with the manifest shas",
        )
    logger.info(
        "Selection resumed from %s (%d prompts; re-stream skipped)",
        manifest_path,
        len(prompts),
    )
    return {
        "prefix_report": manifest["prefix_reproduction"],
        "ext_prompts": prompts,
        "ext_positions": positions,
        "last_stream_pos": manifest["last_stream_pos"],
    }


# ── Banked-artifact parity (plan sections 4.1 / 4.2) ─────────────────────────


def fetch_banked_ladder_file(dl_dir: pathlib.Path, filename: str) -> pathlib.Path:
    """Download a banked round-1 ladder artifact at the section-10 tree pin."""
    return pathlib.Path(
        retry_transient(
            lambda: hf_hub_download(
                GEN.DATA_REPO,
                f"{BANKED_LADDER_PATH}/{filename}",
                repo_type="dataset",
                revision=BANKED_LADDER_REV,
                local_dir=dl_dir,
            ),
            what=f"hf_hub_download({BANKED_LADDER_PATH}/{filename})",
        )
    )


def assert_banked_roster_parity(roster_obj: dict, eval_dir: pathlib.Path) -> None:
    """Byte-assert the banked roster vs the parent module constants (plan 4.1).

    The banked ``roster.json`` is the reconstruction authority; the parent
    module constants must MATCH it (rc 10 otherwise) so the two reconstruction
    paths (banked-roster templating here, ``GEN.persona_system`` in the P0-ext
    gate) can never diverge silently.
    """
    # Observed schema at pin 009b58fd... (probed this round): top-level keys
    # ['metadata', 'personas', 'template']; personas[0] keys
    # ['card', 'idx', 'name']; 16 personas.
    banked_personas = [(p["name"], p["card"]) for p in roster_obj["personas"]]
    banked_idx = [p["idx"] for p in roster_obj["personas"]]
    problems = []
    if roster_obj["template"] != GEN.PERSONA_TEMPLATE:
        problems.append("template")
    if banked_personas != GEN.PERSONAS:
        problems.append("personas")
    if banked_idx != list(range(GEN.N_PERSONAS)):
        problems.append("persona_idx_order")
    if problems:
        _halt(
            EXIT_BANKED_PARITY,
            eval_dir / "ext_banked_parity_report.json",
            {
                "reason": "roster_parity_mismatch",
                "mismatched_fields": problems,
                "banked_rev": BANKED_LADDER_REV,
            },
            f"banked roster.json disagrees with the parent module constants: {problems}",
        )
    for p in range(GEN.N_PERSONAS):
        rebuilt = roster_obj["template"].format(
            name=roster_obj["personas"][p]["name"], card=roster_obj["personas"][p]["card"]
        )
        assert rebuilt == GEN.persona_system(p), (
            f"persona {p}: banked-roster reconstruction != GEN.persona_system"
        )


def persona_system_from_roster(roster_obj: dict, p: int) -> str:
    """System prompt built FROM the banked roster fields (the authority, plan 4.1)."""
    row = roster_obj["personas"][p]
    assert row["idx"] == p, f"roster personas out of index order at {p}"
    return roster_obj["template"].format(name=row["name"], card=row["card"])


def build_ext_assignment(n_total: int) -> dict[int, list[int]]:
    """persona(i, k) = i mod k for the round's two arms over ALL n_total contexts."""
    assignment = {k: [i % k for i in range(n_total)] for k in EXT_ARMS}
    # Exact-rule + balance belt (the parent verify_assignment covers the 5-arm
    # nested design; the 2-arm extension needs the recompute + balance only).
    assert assignment[1] == [0] * n_total
    counts = Counter(assignment[16])
    for p in range(16):
        expected = n_total // 16 + (1 if p < n_total % 16 else 0)
        assert counts.get(p, 0) == expected, (
            f"arm 16 persona {p}: {counts.get(p, 0)} != registered {expected}"
        )
    return assignment


def assert_assignment_prefix(
    ext_assignment: dict[int, list[int]], banked_assignment_obj: dict, eval_dir: pathlib.Path
) -> None:
    """The 5,000-row prefix must equal the banked assignment arms '1'/'16' (rc 10)."""
    # Observed schema at pin 009b58fd... (probed this round): top-level keys
    # ['arms', 'metadata', 'n_contexts', 'realized_total_pairs',
    # 'registered_rule', 'registered_total_pairs_full']; arms keyed by str(k).
    banked_arms = banked_assignment_obj["arms"]
    n_banked = banked_assignment_obj["n_contexts"]
    mismatched = [str(k) for k in EXT_ARMS if ext_assignment[k][:n_banked] != banked_arms[str(k)]]
    if mismatched:
        _halt(
            EXIT_BANKED_PARITY,
            eval_dir / "ext_banked_parity_report.json",
            {
                "reason": "assignment_prefix_mismatch",
                "mismatched_arms": mismatched,
                "n_banked": n_banked,
                "banked_rev": BANKED_LADDER_REV,
            },
            f"assignment_ext {n_banked}-row prefix != banked assignment.json arms {mismatched}",
        )


# ── Extension pairs + items (plan section 4.2 arithmetic) ────────────────────


def ext_context_ids(n_prefix: int, n_total: int) -> range:
    return range(n_prefix, n_total)


def registered_ext_pair_count(n_prefix: int, n_total: int) -> int:
    """n_ext k1 pairs + (n_ext - multiples-of-16) nonzero-residue k16 pairs."""
    ids = ext_context_ids(n_prefix, n_total)
    n_ext = len(ids)
    n_mult16 = sum(1 for i in ids if i % 16 == 0)
    return n_ext + (n_ext - n_mult16)


def build_ext_pairs(n_prefix: int, n_total: int) -> set[tuple[int, int]]:
    """Unique extension (context, persona) pairs: {(i, 0)} U {(i, i % 16)}."""
    pairs: set[tuple[int, int]] = set()
    for i in ext_context_ids(n_prefix, n_total):
        pairs.add((i, 0))
        pairs.add((i, i % 16))
    assert len(pairs) == registered_ext_pair_count(n_prefix, n_total), (
        f"realized ext pair count {len(pairs)} != registered "
        f"{registered_ext_pair_count(n_prefix, n_total)}"
    )
    return pairs


def build_ext_items(
    questions_by_id: dict[int, str], pairs: set[tuple[int, int]], roster_obj: dict
) -> list[DispatchItem]:
    """One DispatchItem per extension pair, (context, persona)-sorted (so the
    FIRST ``PILOT_N_PAIRS`` slice IS the Gate A pilot set); system prompt from
    the BANKED roster (parity-asserted vs GEN.persona_system upstream)."""
    items = []
    for i, p in sorted(pairs):
        items.append(
            DispatchItem(
                item_id=GEN.make_item_id(p, i),
                payload={
                    "messages": [{"role": "user", "content": questions_by_id[i]}],
                    "system": persona_system_from_roster(roster_obj, p),
                },
            )
        )
    return items


# ── Dispatch stages (parent seams; bounded re-drives; parent exit-3) ─────────


def _dispatch_stage(
    items: list[DispatchItem],
    checkpoint_dir: pathlib.Path,
    max_tokens: int,
    poll_interval: float,
) -> dict[str, DispatchResult]:
    """Seam over the parent's fingerprint-gated, batch-routed ``_dispatch``
    (tests monkeypatch THIS name; the parent gate runs inside)."""
    return asyncio.run(GEN._dispatch(items, checkpoint_dir, max_tokens, poll_interval))


def _load_stage_batch_meta(checkpoint_dir: pathlib.Path) -> dict[str, dict]:
    return GEN.load_batch_meta(checkpoint_dir)


def _stage_cap(checkpoint_dir: pathlib.Path) -> int:
    return GEN.checkpoint_max_tokens(checkpoint_dir)


def run_dispatch_stage(
    stage_name: str,
    stage_root: pathlib.Path,
    stage_items: list[DispatchItem],
    results: dict[str, DispatchResult],
    batch_meta: dict[str, dict],
    max_tokens_by_item: dict[str, int],
    gen_wave_by_item: dict[str, str],
    poll_interval: float,
    eval_dir: pathlib.Path,
    metadata: dict,
) -> int:
    """Dispatch one stage (pilot / wave) with the parent's bounded transport
    re-drive semantics scoped to the stage's items; returns redrive rounds
    used. Transport residue after the bounded rounds -> rc 3 (digest written,
    no sentinel, no upload — parent exit-3 semantics)."""
    if not stage_items:
        logger.info("Stage %s: no items (skipped)", stage_name)
        return 0
    items_by_id = {it.item_id: it for it in stage_items}
    results.update(
        _dispatch_stage(stage_items, stage_root / "batches", GEN.GEN_MAX_TOKENS, poll_interval)
    )
    batch_meta.update(_load_stage_batch_meta(stage_root / "batches"))
    base_cap = _stage_cap(stage_root / "batches")
    for iid in items_by_id:
        max_tokens_by_item[iid] = base_cap
        gen_wave_by_item[iid] = GEN.GEN_WAVE_FIRST
    # FIX B (parent): merge prior runs' redrive checkpoints before computing
    # the pending set — already-paid successes are never re-bought.
    GEN._merge_stale_redrives(
        stage_root, items_by_id, results, batch_meta, max_tokens_by_item, poll_interval
    )
    redrive_rounds = 0
    first_round = GEN._next_redrive_round(stage_root)
    for rnd in range(first_round, first_round + GEN.MAX_TRANSPORT_REDRIVES):
        pending = [iid for iid in GEN.transport_class_ids(results) if iid in items_by_id]
        if not pending:
            break
        GEN._require_redrive_headroom(rnd, len(pending))
        redrive_rounds += 1
        logger.warning(
            "Stage %s: re-driving %d transport-class rows (dir redrive%d)",
            stage_name,
            len(pending),
            rnd,
        )
        sub = [items_by_id[iid] for iid in pending]
        results.update(
            _dispatch_stage(sub, stage_root / f"redrive{rnd}", GEN.GEN_MAX_TOKENS, poll_interval)
        )
        batch_meta.update(_load_stage_batch_meta(stage_root / f"redrive{rnd}"))
        rd_cap = _stage_cap(stage_root / f"redrive{rnd}")
        for iid in pending:
            max_tokens_by_item[iid] = rd_cap
    remaining = [iid for iid in GEN.transport_class_ids(results) if iid in items_by_id]
    if remaining:
        report = {
            "metadata": metadata,
            "incomplete": True,
            "reason": f"transport_class_rows_remaining_{stage_name}",
            "n_remaining": len(remaining),
            "remaining_ids_first20": remaining[:20],
            "redrive_rounds_used": redrive_rounds,
        }
        _halt(
            EXIT_TRANSPORT_RESIDUE,
            eval_dir / "gen_digest_ext.json",
            report,
            f"stage {stage_name}: {len(remaining)} transport-class rows remain after "
            f"{redrive_rounds} fresh re-drive round(s) — re-running resumes the "
            "checkpoints and re-submits only the residue (parent exit-3 semantics)",
        )
    return redrive_rounds


# ── Gate A (plan section 7 row A) ────────────────────────────────────────────


def evaluate_gate_a(
    results: dict[str, DispatchResult],
    pilot_pairs: list[tuple[int, int]],
    all_pairs: set[tuple[int, int]],
) -> dict:
    """Pure Gate A read over the pilot rows (routing lives in enforce_gate_a).

    Per-context 2-arm survival: a context is JUDGED iff ALL its extension
    pairs sit inside the pilot subset (an i%16==0 context has ONE shared
    pair serving both arms); it SURVIVES iff every one of its pilot rows
    classifies ``ok`` (issue823_ladder_gen.classify_validity — stop_reason
    keyed). Cap-hit fraction is over pilot ROWS.
    """
    pilot_set = set(pilot_pairs)
    by_ctx: dict[int, list[tuple[int, int]]] = {}
    for i, p in all_pairs:
        by_ctx.setdefault(i, []).append((i, p))
    judged = sorted(i for i, prs in by_ctx.items() if set(prs) <= pilot_set)
    assert judged, "Gate A: pilot subset fully covers no context — pilot too small"
    validity_by_pair = {
        (i, p): GEN.classify_validity(results[GEN.make_item_id(p, i)]) for i, p in pilot_pairs
    }
    survived = [i for i in judged if all(validity_by_pair[pr] == "ok" for pr in by_ctx[i])]
    n_transport = sum(
        1
        for i, p in pilot_pairs
        if results[GEN.make_item_id(p, i)].category in GEN.TRANSPORT_CATEGORIES
    )
    assert n_transport == 0, (
        f"Gate A read on {n_transport} unresolved transport rows — re-drive first "
        "(transport errors are retried, never persisted into the gate read)"
    )
    n_cap = sum(
        1 for i, p in pilot_pairs if results[GEN.make_item_id(p, i)].stop_reason == "max_tokens"
    )
    per_persona: dict[int, dict] = {}
    for i, p in pilot_pairs:
        d = per_persona.setdefault(p, {"n_rows": 0, "cap_hit": 0})
        d["n_rows"] += 1
        v = validity_by_pair[(i, p)]
        d[v] = d.get(v, 0) + 1
        if results[GEN.make_item_id(p, i)].stop_reason == "max_tokens":
            d["cap_hit"] += 1
    refusal_ranking = sorted(
        ((p, d.get("refusal", 0)) for p, d in per_persona.items()),
        key=lambda t: (-t[1], t[0]),
    )
    survival = len(survived) / len(judged)
    cap_fraction = n_cap / len(pilot_pairs)
    return {
        "gate": "A",
        "n_pilot_pairs": len(pilot_pairs),
        "n_judged_contexts": len(judged),
        "n_survived_contexts": len(survived),
        "survival": survival,
        "survival_floor": GATE_A_SURVIVAL_FLOOR,
        "survival_pass": survival >= GATE_A_SURVIVAL_FLOOR,
        "n_cap_hit_rows": n_cap,
        "cap_hit_fraction": cap_fraction,
        "cap_hit_max": GATE_A_CAP_HIT_MAX,
        "cap_hit_pass": cap_fraction < GATE_A_CAP_HIT_MAX,
        "dropped_context_ids": sorted(set(judged) - set(survived)),
        # Persona-keyed attribution (the issue823_pilot_refusal_attribution.py
        # report pattern, computed inline at pilot grain).
        "per_persona": {str(p): per_persona[p] for p in sorted(per_persona)},
        "refusal_ranking": [{"persona": p, "n_refusals": n} for p, n in refusal_ranking],
    }


def enforce_gate_a(report: dict, eval_dir: pathlib.Path) -> None:
    """Persist the Gate A report (pass AND halt), then route the band misses:
    survival < floor -> rc 8; cap-hit >= max -> rc 9 (raise-base-cap decision
    BEFORE the wave). Survival takes precedence when both miss."""
    report_path = eval_dir / "gateA_report_ext.json"
    GEN.write_json(report_path, report)
    if not report["survival_pass"]:
        raise_msg = (
            f"Gate A survival {report['survival']:.4f} < floor "
            f"{GATE_A_SURVIVAL_FLOOR} — wave build halted; persona-keyed refusal "
            "attribution in the report"
        )
        logger.error(
            "HALT-AND-REPORT (rc=%d): %s; report at %s",
            EXIT_GATE_A_SURVIVAL,
            raise_msg,
            report_path,
        )
        raise DesignedHalt(EXIT_GATE_A_SURVIVAL, report_path)
    if not report["cap_hit_pass"]:
        raise_msg = (
            f"Gate A cap-hit fraction {report['cap_hit_fraction']:.4f} >= "
            f"{GATE_A_CAP_HIT_MAX} — raise the base cap BEFORE the wave (parent "
            "precedent: 4096 was pilot-set); wave build halted"
        )
        logger.error(
            "HALT-AND-REPORT (rc=%d): %s; report at %s", EXIT_GATE_A_CAP_HIT, raise_msg, report_path
        )
        raise DesignedHalt(EXIT_GATE_A_CAP_HIT, report_path)
    logger.info(
        "Gate A PASS: survival %.4f (floor %.2f), cap-hit %.4f (max %.2f) over %d judged contexts",
        report["survival"],
        GATE_A_SURVIVAL_FLOOR,
        report["cap_hit_fraction"],
        GATE_A_CAP_HIT_MAX,
        report["n_judged_contexts"],
    )


# ── Cap-hit regen (ext-scoped mirror of GEN.cells_over_cap_threshold) ────────


def ext_cells_over_cap_threshold(
    stop_by_item: dict[str, str | None],
    ext_assignment: dict[int, list[int]],
    ext_ids: range,
) -> tuple[dict[str, dict], dict[int, list[int]]]:
    """Per-(arm x persona) CELL cap-hit trigger restricted to EXTENSION rows.

    Mirrors ``issue823_ladder_gen.cells_over_cap_threshold`` (strictly-greater
    trigger, deduped union across triggered cells) but iterates the round's
    two arms with EXT-only denominators — the parent function enumerates
    ``K_ARMS`` from context 0 and would dilute every cell with the banked,
    never-generated prefix rows.
    """
    triggered_cells: dict[str, dict] = {}
    regen_union: dict[int, set[int]] = {}
    for k in EXT_ARMS:
        cell_rows: dict[int, list[int]] = {}
        for i in ext_ids:
            cell_rows.setdefault(ext_assignment[k][i], []).append(i)
        for p, rows in sorted(cell_rows.items()):
            over = [i for i in rows if stop_by_item.get(GEN.make_item_id(p, i)) == "max_tokens"]
            if len(over) / len(rows) > GEN.CAP_HIT_REGEN_FRACTION:
                triggered_cells[f"k={k},p={p}"] = {
                    "k": k,
                    "persona": p,
                    "n_over_cap": len(over),
                    "n_rows": len(rows),
                    "fraction": len(over) / len(rows),
                }
                regen_union.setdefault(p, set()).update(over)
    regen_pairs = {p: sorted(rows) for p, rows in sorted(regen_union.items())}
    return triggered_cells, regen_pairs


def run_pooled_regen(
    root: pathlib.Path,
    items_by_id: dict[str, DispatchItem],
    regen_pairs: dict[int, list[int]],
    results: dict[str, DispatchResult],
    batch_meta: dict[str, dict],
    max_tokens_by_item: dict[str, int],
    gen_wave_by_item: dict[str, str],
    poll_interval: float,
    eval_dir: pathlib.Path,
    metadata: dict,
    triggered_cells: dict[str, dict],
) -> set[str]:
    """ONE pooled regen dispatch at REGEN_MAX_TOKENS (parent FIX D transport
    pooling; per-row accounting stays per item). Transport residue -> rc 3
    with the parent's quarantine remedy named."""
    pooled_ids = sorted(
        GEN.make_item_id(p, i) for p, ctx_rows in regen_pairs.items() for i in ctx_rows
    )
    rg_dir = root / "regen_pooled"
    logger.info(
        "Pooled cap-hit regen: ONE batch dispatch for %d rows spanning %d persona(s)",
        len(pooled_ids),
        len(regen_pairs),
    )
    sub = [items_by_id[iid] for iid in pooled_ids]
    rg = _dispatch_stage(sub, rg_dir, GEN.REGEN_MAX_TOKENS, poll_interval)
    rg_remaining = GEN.transport_class_ids(rg)
    if rg_remaining:
        _halt(
            EXIT_TRANSPORT_RESIDUE,
            eval_dir / "gen_digest_ext.json",
            {
                "metadata": metadata,
                "incomplete": True,
                "reason": "transport_class_rows_remaining_in_regen",
                "regen_cells_triggered": triggered_cells,
                "n_remaining": len(rg_remaining),
                "remaining_ids_first20": rg_remaining[:20],
                "regen_checkpoint_dir": str(rg_dir),
            },
            f"pooled cap-hit regen has {len(rg_remaining)} transport-class rows — a plain "
            f"re-run replays this checkpoint; quarantine it first (mv {rg_dir} "
            f"{rg_dir.with_name(rg_dir.name + '.stale')}) so the re-run re-submits fresh "
            "(parent issue823_ladder_gen exit-3 regen semantics)",
        )
    results.update(rg)
    batch_meta.update(_load_stage_batch_meta(rg_dir))
    regen_cap = _stage_cap(rg_dir)
    for iid in pooled_ids:
        max_tokens_by_item[iid] = regen_cap
        gen_wave_by_item[iid] = GEN.GEN_WAVE_REGEN
    return set(pooled_ids)


# ── Records + digest (ext mirrors of the parent builders) ────────────────────


def build_ext_records(
    questions_by_id: dict[int, str],
    pairs: set[tuple[int, int]],
    ext_assignment: dict[int, list[int]],
    results: dict[str, DispatchResult],
    batch_meta: dict[str, dict],
    items_by_id: dict[str, DispatchItem],
    max_tokens_by_item: dict[str, int],
    gen_wave_by_item: dict[str, str],
    regen_items: set[str],
    pilot_items: set[str],
) -> dict[int, list[dict]]:
    """Per-persona extension records — mirrors ``GEN.build_records`` with the
    round's two arms, no ``in_common_valid`` (a prefix-only concept), plus a
    ``corpus`` tag and the pilot/wave stage label. Every record keeps the
    parent's batch-provenance fail-loud asserts."""
    by_persona: dict[int, list[dict]] = {p: [] for p in range(GEN.N_PERSONAS)}
    for i, p in sorted(pairs):
        item_id = GEN.make_item_id(p, i)
        res = results[item_id]
        meta = batch_meta[item_id]
        validity = GEN.classify_validity(res)
        system_prompt = items_by_id[item_id].payload["system"]
        rec = {
            "context_id": i,
            "persona_idx": p,
            "persona_name": GEN.PERSONAS[p][0],
            "arms": [k for k in EXT_ARMS if ext_assignment[k][i] == p],
            "corpus": "ladder_ext",
            "gen_stage": "pilot" if item_id in pilot_items else "wave",
            "question": questions_by_id[i],
            "answer_text": res.result if validity in ("ok", "refusal") else None,
            "seed": 42,  # file-provenance label; the API exposes no sampling seed
            "filled": validity == "ok",
            "validity": validity,
            "stop_reason": res.stop_reason,
            "cap_hit": res.stop_reason == "max_tokens",
            "model": GEN.SONNET_MODEL,
            "temperature": GEN.GEN_TEMPERATURE,
            "max_tokens": max_tokens_by_item[item_id],
            "gen_wave": gen_wave_by_item[item_id],
            "regen": item_id in regen_items,
            "system_prompt": system_prompt,
            "system_prompt_sha256": _sha256_text(system_prompt),
            "batch_id": meta["batch_id"],
            "batch_request_custom_id": meta["batch_request_custom_id"],
            "batch_org": meta["batch_org"],
            "batch_submitted_at": meta["batch_submitted_at"],
            "harvested_at": meta["harvested_at"],
        }
        assert rec["batch_id"] and rec["batch_request_custom_id"], (
            f"{item_id}: missing batch_id/custom_id — batch provenance incomplete"
        )
        assert rec["batch_submitted_at"] and rec["harvested_at"], (
            f"{item_id}: missing batch timestamps — batch provenance incomplete"
        )
        assert rec["batch_org"], f"{item_id}: missing batch_org — batch provenance incomplete"
        assert rec["arms"], f"{item_id}: pair belongs to no ext arm — assignment join broken"
        by_persona[p].append(rec)
    return by_persona


def build_ext_digest(
    n_ext: int,
    pairs: set[tuple[int, int]],
    ext_assignment: dict[int, list[int]],
    ext_ids: range,
    by_persona: dict[int, list[dict]],
    prefix_report: dict,
    gate_a_report: dict,
    redrive_rounds: dict[str, int],
    triggered_cells: dict[str, dict],
    regen_pairs: dict[int, list[int]],
    selection_stats: dict,
    metadata: dict,
) -> dict:
    """Extension digest — parent ``build_digest`` field conventions at ext
    scope (validity/cap tallies over EXTENSION rows only; post-regen re-measure
    with the literal ``cap-hit>2%`` label; never read 'regen ran' as
    'trigger resolved')."""
    validity_by_persona: dict[int, dict] = {}
    for p, recs in by_persona.items():
        vc = Counter(r["validity"] for r in recs)
        n_cap = sum(1 for r in recs if r["cap_hit"])
        validity_by_persona[p] = {**dict(vc), "cap_hit": n_cap, "n_rows": len(recs)}
    rec_by_pair = {
        (r["context_id"], r["persona_idx"]): r for recs in by_persona.values() for r in recs
    }
    cap_frac_by_arm_persona: dict[int, dict[int, float]] = {}
    for k in EXT_ARMS:
        per_p: dict[int, list[bool]] = {}
        for i in ext_ids:
            p = ext_assignment[k][i]
            per_p.setdefault(p, []).append(rec_by_pair[(i, p)]["cap_hit"])
        cap_frac_by_arm_persona[k] = {p: sum(v) / len(v) for p, v in per_p.items()}
    cells_over_post_regen = [
        {"k": k, "persona": p, "cap_hit_fraction": f, "label": "cap-hit>2%"}
        for k in EXT_ARMS
        for p, f in sorted(cap_frac_by_arm_persona[k].items())
        if f > GEN.CAP_HIT_REGEN_FRACTION
    ]
    error_ids = sorted(
        GEN.make_item_id(r["persona_idx"], r["context_id"])
        for recs in by_persona.values()
        for r in recs
        if r["validity"].startswith("error:")
    )
    wave_by_persona = {
        p: dict(Counter(r["batch_id"] for r in recs)) for p, recs in by_persona.items() if recs
    }
    return {
        "metadata": metadata,
        "n_ext_contexts": n_ext,
        "n_pairs": len(pairs),
        "registered_ext_pairs_full": REGISTERED_EXT_PAIRS,
        "prefix_reproduction": prefix_report,
        "gate_a": gate_a_report,
        "selection": selection_stats,
        "validity_counts_by_persona": validity_by_persona,
        "cap_hit_fraction_by_arm_persona": cap_frac_by_arm_persona,
        "cap_hit_cells_over_threshold_post_regen": cells_over_post_regen,
        "cap_hit_regen_trigger_fraction": GEN.CAP_HIT_REGEN_FRACTION,
        "regen_cells_triggered": triggered_cells,
        "regen_pairs_by_persona": {p: len(rows) for p, rows in sorted(regen_pairs.items())},
        "redrive_rounds_used": redrive_rounds,
        "n_error_rows": len(error_ids),
        "error_row_ids_first20": error_ids[:20],
        "batch_wave_by_persona": wave_by_persona,
    }


# ── Upload (>9.5 MB text line-split; canonical-repo gate) ────────────────────


def _unlink_stale_shards(f: pathlib.Path) -> None:
    """Remove a PRIOR sharding's manifest + shard files for source ``f``.

    An in-place regeneration of a source jsonl otherwise leaves a stale
    `<stem>.manifest.json` + `<stem>.shardNN.jsonl` set that manifest-first
    readers and upload name-sets PREFER over the fresh bytes (r1 concern
    stale-manifest-precedence-regen).
    """
    manifest = f.with_name(f"{f.stem}.manifest.json")
    if manifest.exists():
        manifest.unlink()
        logger.info("unlinked stale shard manifest %s", manifest)
    for sp in sorted(f.parent.glob(f"{f.stem}.shard*.jsonl")):
        sp.unlink()
        logger.info("unlinked stale shard %s", sp)


def shard_large_jsonl_for_upload(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Replace any >9.5 MB .jsonl with <9 MB line-shards + a manifest.

    Restated from ``scripts/issue2054_phase_a.py::_shard_large_jsonl_for_upload``
    (importing that module would drag its issue2054 sibling-import graph);
    IDENTICAL manifest schema {source, parts, line_counts, sha256} — pinned by
    ``orchestrate.hub._parse_shard_manifest`` so hub-side readers reassemble.
    upload-policy.md: text >9.5 MB line-splits, NEVER gzip (>10 MB blobs
    force-route to LFS; *.gz is LFS-matched). Any prior sharding of a source
    is unlinked FIRST (both branches) so a regenerated file can never be
    shadowed by a stale manifest/shard set.
    """
    out: list[pathlib.Path] = []
    for f in files:
        if not f.is_file():
            continue
        size = f.stat().st_size
        if size <= UPLOAD_SHARD_LIMIT_BYTES:
            if f.suffix == ".jsonl":
                _unlink_stale_shards(f)  # regenerated-now-small source
            out.append(f)
            continue
        if f.suffix != ".jsonl":
            logger.warning("oversized non-jsonl upload rides LFS: %s (%d B)", f, size)
            out.append(f)
            continue
        _unlink_stale_shards(f)  # re-shard from the CURRENT bytes only
        shards: list[pathlib.Path] = []
        line_counts: list[int] = []
        shard_lines: list[str] = []
        shard_bytes = 0

        def _flush() -> None:
            nonlocal shard_lines, shard_bytes
            if not shard_lines:
                return
            sp = f.with_name(f"{f.stem}.shard{len(shards):02d}.jsonl")
            tmp = sp.with_name(sp.name + ".tmp")
            tmp.write_text("".join(shard_lines), encoding="utf-8")
            os.replace(tmp, sp)
            shards.append(sp)
            line_counts.append(len(shard_lines))
            shard_lines, shard_bytes = [], 0

        with f.open(encoding="utf-8") as fh:
            for line in fh:
                b = len(line.encode("utf-8"))
                if shard_bytes + b > UPLOAD_SHARD_TARGET_BYTES:
                    _flush()
                shard_lines.append(line)
                shard_bytes += b
        _flush()
        manifest = f.with_name(f"{f.stem}.manifest.json")
        GEN.write_json(
            manifest,
            {
                "source": f.name,
                "parts": [s.name for s in shards],
                "line_counts": line_counts,
                "sha256": {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in shards},
            },
        )
        logger.info("sharded %s (%d B) -> %d shards", f, size, len(shards))
        out.extend(shards)
        out.append(manifest)
    return out


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        # Rewriting a source jsonl invalidates any prior sharding of it —
        # unlink FIRST so a mid-write crash leaves the (old source, no
        # manifest) consistent pair, never (new source, stale manifest).
        _unlink_stale_shards(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P-Gen-ext for #823 origin-ladder-more-contexts: 83,313 extension "
            "(context, persona) Sonnet Batch generations (select, Gate A pilot, wave, "
            "regen, persist, upload)."
        )
    )
    parser.add_argument("--smoke", action="store_true", help="tiny real run: /tmp + _smoke prefix")
    parser.add_argument(
        "--n-ext-contexts",
        type=int,
        default=None,
        help="extension context count override (smoke only; production pinned to 43000)",
    )
    parser.add_argument(
        "--pilot-pairs",
        type=int,
        default=None,
        help="Gate A pilot size (smoke only; production pinned to the registered 200)",
    )
    parser.add_argument(
        "--max-stream-pos",
        type=int,
        default=MAX_STREAM_POS_DEFAULT,
        help="designed selection depth bound (rc 7 past it)",
    )
    parser.add_argument("--out-root", type=pathlib.Path, default=None, help="output root override")
    parser.add_argument("--poll-interval", type=float, default=None, help="batch poll seconds")
    parser.add_argument(
        "--list-rcs", action="store_true", help="print the designed-abort rc table and exit"
    )
    args = parser.parse_args(argv)

    if args.list_rcs:
        print(
            json.dumps(
                {
                    "0": "complete",
                    "3": "transport-class residue (pilot/wave/regen)",
                    "4": "generation-config fingerprint mismatch on resume",
                    "6": "LMSYS selection drift (ctx0 / prefix reproduction)",
                    "7": "LMSYS stream exhausted / max-stream-pos bound",
                    "8": "Gate A survival < 0.85",
                    "9": "Gate A cap-hit fraction >= 0.02",
                    "10": "banked-artifact parity mismatch (roster / assignment prefix)",
                    "11": "sampling-manifest resume fingerprint mismatch",
                }
            )
        )
        return

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if args.smoke:
        n_ext = args.n_ext_contexts if args.n_ext_contexts is not None else 16
        assert n_ext >= 1, "--smoke needs --n-ext-contexts >= 1"
        assert n_ext <= N_EXT_FULL, "--n-ext-contexts exceeds the registered extension size"
        pilot_n = args.pilot_pairs if args.pilot_pairs is not None else 16
        root = args.out_root or pathlib.Path("/tmp/issue-823-ext-smoke/ladder_ext_gen")
        eval_dir = root / "eval_results" / "origin-ladder-more-contexts"
        hf_prefix = GEN.HF_PREFIX + "_smoke"
    else:
        if args.n_ext_contexts is not None and args.n_ext_contexts != N_EXT_FULL:
            parser.error("--n-ext-contexts is smoke-only; production runs the full 43000")
        if args.pilot_pairs is not None and args.pilot_pairs != PILOT_N_PAIRS:
            parser.error("--pilot-pairs is smoke-only; production runs the registered 200")
        n_ext = N_EXT_FULL
        pilot_n = PILOT_N_PAIRS
        root = args.out_root or (repo_root / "data" / "issue_823" / "ladder_ext_gen")
        eval_dir = repo_root / "eval_results" / "issue_823" / "origin-ladder-more-contexts"
        hf_prefix = GEN.HF_PREFIX
    assert pilot_n >= 2, "--pilot-pairs must be >= 2 (Gate A needs a fully-judged context)"
    poll_interval = (
        args.poll_interval if args.poll_interval is not None else (10.0 if args.smoke else 30.0)
    )
    n_total = N_PREFIX + n_ext
    stage_dir = root / "hf_stage" / "ladder_ext"
    dl_dir = root / "parent_inputs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "P-Gen-ext: n_ext=%d pilot_n=%d smoke=%s root=%s hf_prefix=%s",
        n_ext,
        pilot_n,
        args.smoke,
        root,
        hf_prefix,
    )

    metadata = {
        "script": "scripts/issue823_ladder_ext_gen.py",
        "task": 823,
        "followup_label": "origin-ladder-more-contexts",
        "git_commit": GEN._git_commit(),
        "generated_at": GEN._utc_now(),
        "model": GEN.SONNET_MODEL,
        "temperature": GEN.GEN_TEMPERATURE,
        "max_tokens_default": GEN.GEN_MAX_TOKENS,
        "regen_max_tokens": GEN.REGEN_MAX_TOKENS,
        "parent_rev": GEN.PARENT_REV,
        "banked_ladder_rev": BANKED_LADDER_REV,
        "n_ext_contexts": n_ext,
        "n_total_contexts": n_total,
        "pilot_pairs": pilot_n,
        "smoke": args.smoke,
        "ext_arms": list(EXT_ARMS),
        "selection_rule_id": SELECTION_RULE_ID,
    }

    # 1. Banked frozen prefix (full-grain validation; parent loader + pin).
    questions_full, _in_common_full, mask_crosscheck = GEN.load_frozen_questions(dl_dir)

    # 2. Banked roster + assignment at the section-10 tree pin (rc 10 gates).
    roster_obj = json.loads(fetch_banked_ladder_file(dl_dir, "roster.json").read_text())
    assert_banked_roster_parity(roster_obj, eval_dir)
    banked_assignment_obj = json.loads(
        fetch_banked_ladder_file(dl_dir, "assignment.json").read_text()
    )

    # 3. Selection (fingerprint-gated manifest resume, else the live stream).
    fields = selection_fingerprint_fields(n_ext, args.max_stream_pos)
    selection = load_selection_if_persisted(stage_dir, fields, eval_dir)
    if selection is None:
        selection = select_extension_contexts(
            questions_full,
            n_ext,
            max_stream_pos=args.max_stream_pos,
            eval_dir=eval_dir,
            stage_dir=stage_dir,
            fingerprint_fields=fields,
        )
        write_sampling_manifest(stage_dir, selection, fields, metadata)

    # 4. Assignment + prefix parity + pairs + items.
    ext_assignment = build_ext_assignment(n_total)
    assert_assignment_prefix(ext_assignment, banked_assignment_obj, eval_dir)
    assignment_ext_obj = {
        "metadata": metadata,
        "registered_rule": "persona(i, k) = i mod k over the frozen 0-indexed context order",
        "n_contexts": n_total,
        "registered_ext_pairs_full": REGISTERED_EXT_PAIRS,
        "arms": {str(k): ext_assignment[k] for k in EXT_ARMS},
        "prefix_source": {
            "file": f"{BANKED_LADDER_PATH}/assignment.json",
            "rev": BANKED_LADDER_REV,
            "prefix_equal": True,
        },
    }
    GEN.write_json(stage_dir / "assignment_ext.json", assignment_ext_obj)
    pairs = build_ext_pairs(N_PREFIX, n_total)
    if n_ext == N_EXT_FULL:
        ids = ext_context_ids(N_PREFIX, n_total)
        n_mult16 = sum(1 for i in ids if i % 16 == 0)
        assert n_mult16 == 2_687 and len(pairs) == REGISTERED_EXT_PAIRS == 83_313, (
            f"production pair arithmetic drifted: multiples16={n_mult16}, pairs={len(pairs)}"
        )
    questions_by_id = {i: q for i, q in enumerate(questions_full)}
    questions_by_id.update({N_PREFIX + j: p for j, p in enumerate(selection["ext_prompts"])})
    items = build_ext_items(questions_by_id, pairs, roster_obj)
    assert len(items) == len(pairs)
    items_by_id = {it.item_id: it for it in items}
    pilot_items = items[:pilot_n]
    wave_items = items[pilot_n:]
    pilot_pairs = sorted(pairs)[:pilot_n]

    results: dict[str, DispatchResult] = {}
    batch_meta: dict[str, dict] = {}
    max_tokens_by_item: dict[str, int] = {}
    gen_wave_by_item: dict[str, str] = {}

    # 5. Gate A pilot (production rows, dispatched alone, gated BEFORE the wave).
    rr_pilot = run_dispatch_stage(
        "pilot",
        root / "pilot",
        pilot_items,
        results,
        batch_meta,
        max_tokens_by_item,
        gen_wave_by_item,
        poll_interval,
        eval_dir,
        metadata,
    )
    gate_a_report = evaluate_gate_a(results, pilot_pairs, pairs)
    enforce_gate_a(gate_a_report, eval_dir)

    # 6. Remaining wave.
    rr_wave = run_dispatch_stage(
        "wave",
        root / "wave",
        wave_items,
        results,
        batch_meta,
        max_tokens_by_item,
        gen_wave_by_item,
        poll_interval,
        eval_dir,
        metadata,
    )

    # 7. ONE pooled regen round for over-cap (arm x persona) cells.
    ext_ids = ext_context_ids(N_PREFIX, n_total)
    stop_by_item = {iid: res.stop_reason for iid, res in results.items()}
    triggered_cells, regen_pairs = ext_cells_over_cap_threshold(
        stop_by_item, ext_assignment, ext_ids
    )
    regen_items: set[str] = set()
    if regen_pairs:
        regen_items = run_pooled_regen(
            root,
            items_by_id,
            regen_pairs,
            results,
            batch_meta,
            max_tokens_by_item,
            gen_wave_by_item,
            poll_interval,
            eval_dir,
            metadata,
            triggered_cells,
        )

    # 8. Records + digest + sentinel.
    by_persona = build_ext_records(
        questions_by_id,
        pairs,
        ext_assignment,
        results,
        batch_meta,
        items_by_id,
        max_tokens_by_item,
        gen_wave_by_item,
        regen_items,
        {it.item_id for it in pilot_items},
    )
    record_files: list[str] = []
    for p in range(GEN.N_PERSONAS):
        if not by_persona[p]:
            continue
        fn = f"persona{p:02d}_ext.jsonl"
        _write_jsonl(stage_dir / fn, by_persona[p])
        GEN.write_json(
            stage_dir / f"persona{p:02d}_ext.meta.json",
            {
                "metadata": {
                    **metadata,
                    "persona_idx": p,
                    "persona_name": GEN.PERSONAS[p][0],
                    "persona_card": GEN.PERSONAS[p][1],
                    "n_records": len(by_persona[p]),
                },
                "records_file": fn,
                "records_sha256": GEN._sha256_file(stage_dir / fn),
            },
        )
        record_files.append(fn)

    selection_stats = {
        "last_stream_pos": selection["last_stream_pos"],
        "n_ext_selected": len(selection["ext_prompts"]),
        "max_stream_pos": args.max_stream_pos,
        "mask_crosscheck": mask_crosscheck,
    }
    digest = build_ext_digest(
        n_ext,
        pairs,
        ext_assignment,
        ext_ids,
        by_persona,
        selection["prefix_report"],
        gate_a_report,
        {"pilot": rr_pilot, "wave": rr_wave},
        triggered_cells,
        regen_pairs,
        selection_stats,
        metadata,
    )
    GEN.write_json(eval_dir / "gen_digest_ext.json", digest)
    GEN.write_json(stage_dir / "gen_digest_ext.json", digest)
    GEN.write_json(stage_dir / "gateA_report_ext.json", gate_a_report)

    # 9. Shard oversized text, then sentinel over the EXACT upload set, then
    # ONE bulk commit gated on the canonical repo (parent FIX 1).
    upload_paths = shard_large_jsonl_for_upload(
        [stage_dir / fn for fn in record_files]
        + [
            stage_dir / PROMPTS_FILENAME,
            stage_dir / MANIFEST_FILENAME,
            stage_dir / "assignment_ext.json",
            stage_dir / "gen_digest_ext.json",
            stage_dir / "gateA_report_ext.json",
        ]
        + [
            stage_dir / f"persona{p:02d}_ext.meta.json"
            for p in range(GEN.N_PERSONAS)
            if by_persona[p]
        ]
    )
    upload_rel = sorted(p.relative_to(stage_dir).as_posix() for p in upload_paths)
    sentinel = {
        "phase": "p_gen_ext",
        "complete": True,
        "metadata": metadata,
        "generation_config_fingerprint": GEN.generation_config_fingerprint(),
        "selection_fingerprint_fields": fields,
        "n_pairs": len(pairs),
        "n_ok": sum(1 for recs in by_persona.values() for r in recs if r["validity"] == "ok"),
        "n_error_rows": digest["n_error_rows"],
        "files_sha256": {rel: GEN._sha256_file(stage_dir / rel) for rel in upload_rel},
    }
    GEN.write_json(stage_dir / SENTINEL_FILENAME, sentinel)
    upload_rel.append(SENTINEL_FILENAME)

    path_in_repo = f"{hf_prefix}/{HF_EXT_SUBPATH}"
    url = _upload_folder_filtered(
        local_dir=stage_dir,
        repo_id=GEN.DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=upload_rel,
        expected_repo_paths=[f"{path_in_repo}/{rel}" for rel in upload_rel],
    )
    if not url:
        raise RuntimeError(
            f"HF upload of {len(upload_rel)} P-Gen-ext files to "
            f"{GEN.DATA_REPO}/{path_in_repo} failed or verified incomplete — refusing to "
            "report P-Gen-ext complete"
        )
    GEN._require_canonical_upload(url, f"{GEN.DATA_REPO}/{path_in_repo}")
    logger.info("P-Gen-ext complete: %d files uploaded to %s", len(upload_rel), url)
    logger.info(
        "Digest: %s | ok=%d/%d error=%d redrives=%s regen_cells=%s",
        eval_dir / "gen_digest_ext.json",
        sentinel["n_ok"],
        len(pairs),
        digest["n_error_rows"],
        digest["redrive_rounds_used"],
        sorted(triggered_cells),
    )
    # Explicit success terminal AFTER all durables landed: a `datasets`
    # streaming iterator surviving to implicit interpreter finalization can
    # SIGABRT (rc=134, pyarrow thread-teardown class — gotchas.md); the
    # explicit exit fires before finalize-time teardown can rewrite the rc.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
