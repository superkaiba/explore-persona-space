#!/usr/bin/env python
"""Build the canonical Phase-B answers pool for task #2054 (round 9, prerequisite 1).

Emits a JSONL of ``{conv_id, answer, answer_provenance}`` rows — the
``--answers-source`` input ``scripts/issue2054_phase_b.py::_index_answers``
consumes (extra fields ignored by the consumer). Every answer is the
ORIGINAL conversation's assistant reply — the authored-CHAT cell of the
plan's authorship x presentation 2x2 (plan v8 ~93-104). Cell (c) / Phase D
owns the story-authored transpose; no model-written or story-authored text
enters this pool.

Coverage contract (round-9 spec, epm:progress v82): the pool covers the
union of ADMITTED Phase-A conv_ids (``issue2054_lattice/scaffolds/kept.json``
``variants.<v>.admitted_conv_ids`` — the r7 admission record), so phase_b's
``n_answer_from_scaffold_fallback`` is zero by construction. Two key spaces,
two sources, both authored-CHAT:

- ``mt_<hash>`` (generated scaffolds): answer = the FIRST assistant message
  AFTER the FIRST user turn of the manifest conversation
  (``issue1738_multiturn/sampling_manifest``, 55 ``part_*.jsonl``), staged at
  the SAME revision the seed-137 shared draw pinned
  (``shared_question_draw.meta.json`` ``revision``) and scanned in the SAME
  sorted-part order with first-wins dedupe — matching
  ``issue2054_phase_a._draw_shared_questions``'s turn-selection convention
  exactly. Every matched row's first-user-turn is asserted equal to the
  drawn question (``shared_question_draw.jsonl``, staged MANIFEST-FIRST from
  its sharded upload — unsharded hub-name fallback only when no manifest
  exists; the r15 fix), sealing the convention.
- ``stripped_<story_id>`` (recovered scaffolds): answer = the scaffold row's
  own stripper-preserved ORIGINAL answer (the parent #1345 paired stories
  embed the original conversation answer NEAR-verbatim; the stripper records
  it). Staged from the ADMITTED sharded pools
  (``issue2054_lattice/scaffolds/<variant>/scaffolds_<variant>.shardNN.jsonl``
  + ``.manifest.json`` — NEVER the unsharded name, the r6 stale-residue
  trap), sha-verified per shard. Cross-variant answer conflicts are RESOLVED
  per the r12 policy (whitespace/prefix -> deterministic canonicalization;
  substantive -> conv_id EXCLUDED via ``answers_excluded_conv_ids.json``;
  beyond-tail substantive rate -> hard raise) — see the Source-1 section.

Smoke (tiny-real): ``--max-scan-rows`` caps total streamed manifest rows AND
``--smoke-kept-cap`` caps kept answers per source; ``--only-stripped-cids``
narrows the required set to named conv_ids (the conflict-resolution slice
probe). Any of the three = smoke mode, which REQUIRES a ``/tmp``
``--out-dir`` (a capped pool at the production path is a residue trap for a
later phase_b dispatch) and skips the full-coverage assert (kept > 0
asserted instead). ``--skip-upload`` skips the HF mirror.

Upload: shards the pool via ``issue2054_phase_a._shard_large_jsonl_for_upload``
and mirrors in ONE bulk commit to ``issue2054_lattice/answers/`` (the
``_upload_scaffold_files`` pattern). A consumer on a fresh machine
reassembles from ``answers_pool.manifest.json`` + shards.

Content hygiene: prints COUNTS / ids / hashes only — never conversation or
answer text (LMSYS-derived corpus).

Exit 0 on success; 1 on coverage / consistency / upload failure; 2 on usage
errors or missing dependency.

Usage (production, dispatched by the orchestrator after code review):
  uv run python scripts/issue2054_build_answers.py \\
      --out-dir data/issue_2054/answers --staging-dir data/issue_2054/hf_dl

Smoke:
  uv run python scripts/issue2054_build_answers.py \\
      --out-dir /tmp/issue-2054-r9-smoke/answers \\
      --staging-dir /tmp/issue-2054-r9-smoke/hf_dl \\
      --variants char_vex --max-scan-rows 2000 --smoke-kept-cap 8 --skip-upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_phase_a as pa  # noqa: E402

SCAFFOLDS_PREFIX = f"{pa.TASK_PREFIX}/scaffolds"
ANSWERS_PREFIX = f"{pa.TASK_PREFIX}/answers"
POOL_STEM = "answers_pool"
# Plan §7 / kill gate 4: per-(character, model) usable-conv floor (the fits-side
# intersection bound, `issue2054_fits.py`); the r12 exclusion report checks each
# variant's post-exclusion admitted count against it.
ADMISSION_FLOOR = 4480


def _log(msg: str) -> None:
    print(f"[phase=build_answers] {msg}", flush=True)


def _api():
    from huggingface_hub import HfApi

    return HfApi()


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------
def _pinned_revision(staging_root: Path, override: str | None) -> tuple[str, dict]:
    """Resolve the manifest revision: the r7 shared draw's pinned revision
    (``shared_question_draw.meta.json``) unless overridden. Using the SAME
    snapshot the draw scanned guarantees identical ``source_hash``-derived
    conv_id keys."""
    from explore_persona_space.orchestrate.hub import stage_hub_file

    meta_path = stage_hub_file(
        pa.HF_DATA_REPO,
        f"{SCAFFOLDS_PREFIX}/shared_question_draw.meta.json",
        staging_root / SCAFFOLDS_PREFIX / "shared_question_draw.meta.json",
        repo_type="dataset",
        overwrite=True,
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    revision = override or str(meta.get("revision") or "")
    if not revision:
        raise RuntimeError(
            "no manifest revision: shared_question_draw.meta.json carries none "
            "and --manifest-revision was not passed"
        )
    return revision, meta


def _required_conv_ids(
    staging_root: Path, variants: list[str]
) -> tuple[set[str], dict, dict[str, set[str]]]:
    """Union of ADMITTED conv_ids across ``variants`` from kept.json (the r7
    admission record — the deterministic coverage source). Also returns the
    per-variant admitted id sets (the r12 exclusion-impact report reads them)."""
    from explore_persona_space.orchestrate.hub import stage_hub_file

    kept_path = stage_hub_file(
        pa.HF_DATA_REPO,
        f"{SCAFFOLDS_PREFIX}/kept.json",
        staging_root / SCAFFOLDS_PREFIX / "kept.json",
        repo_type="dataset",
        overwrite=True,
    )
    kept = json.loads(kept_path.read_text(encoding="utf-8"))
    kept_variants = kept.get("variants") or {}
    required: set[str] = set()
    per_variant: dict[str, int] = {}
    per_variant_ids: dict[str, set[str]] = {}
    for v in variants:
        rec = kept_variants.get(v)
        if rec is None:
            raise RuntimeError(f"kept.json has no admission record for variant {v!r}")
        ids = [str(x) for x in (rec.get("admitted_conv_ids") or [])]
        if not ids:
            raise RuntimeError(f"kept.json admitted_conv_ids EMPTY for variant {v!r}")
        per_variant[v] = len(ids)
        per_variant_ids[v] = set(ids)
        required.update(ids)
    record = {
        "kept_json_sha256": hashlib.sha256(kept_path.read_bytes()).hexdigest(),
        "kept_target_conv_ids": kept.get("target_conv_ids"),
        "admitted_per_variant": per_variant,
        "n_required_union": len(required),
    }
    return required, record, per_variant_ids


def _stage_sharded_jsonl(dest_dir: Path, base_prefix: str, stem: str) -> Path:
    """Stage one sharded JSONL from HF: manifest-first, per-shard sha256
    verification, exact in-order concatenation (the r6 recipe,
    ``issue2054_phase_a._stage_prejudge_from_hf``). The plain ``<stem>.jsonl``
    hub name is NEVER consumed — on this prefix it resolves to prior-round
    unsharded residue (the r6 defect)."""
    from explore_persona_space.orchestrate.hub import stage_hub_file

    dest_dir.mkdir(parents=True, exist_ok=True)
    mpath = stage_hub_file(
        pa.HF_DATA_REPO,
        f"{base_prefix}/{stem}.manifest.json",
        dest_dir / f"{stem}.manifest.json",
        repo_type="dataset",
        overwrite=True,
    )
    man = json.loads(mpath.read_text(encoding="utf-8"))
    parts = list(man.get("parts") or [])
    if not parts:
        raise RuntimeError(f"shard manifest lists no parts: {base_prefix}/{stem}.manifest.json")
    want_sha = man.get("sha256") or {}
    local_parts: list[Path] = []
    for name in parts:
        lp = stage_hub_file(
            pa.HF_DATA_REPO,
            f"{base_prefix}/{name}",
            dest_dir / name,
            repo_type="dataset",
            overwrite=True,
        )
        got = hashlib.sha256(lp.read_bytes()).hexdigest()
        exp = want_sha.get(name)
        if not exp:
            # The r6 writer always records per-shard shas; an absent entry
            # signals a foreign/malformed manifest — refuse the unverified shard.
            raise RuntimeError(
                f"shard manifest carries no sha256 for {name} "
                f"({base_prefix}/{stem}.manifest.json) — refusing unverified shard"
            )
        if exp != got:
            raise RuntimeError(
                f"shard {name} sha mismatch: {got[:12]}... != manifest {str(exp)[:12]}..."
            )
        local_parts.append(lp)
    target = dest_dir / f"{stem}.jsonl"
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as out:
        for lp in local_parts:
            out.write(lp.read_bytes())
    os.replace(tmp, target)
    _log(f"staged {stem}: {len(local_parts)} shard(s) -> {target.stat().st_size} B")
    return target


# ---------------------------------------------------------------------------
# Source 1: stripped_* answers from the ADMITTED scaffold pools
# ---------------------------------------------------------------------------
# Cross-variant conflict resolution (r12, closes epm:failure v3). The r9
# premise "the parent #1345 paired stories embed the original conversation
# answer byte-exact" is FALSE at the tail: the parent story generations
# re-render the embedded answer with whitespace reflow (255/270 measured
# conflicts collapse under whitespace normalization) and, rarely, small
# character-level edits (15/270 substantive; diagnosis:
# eval_results/issue_2054/audits/answers_conflicts_diagnosis.json). The 2x2
# authorship axis needs ONE byte-fixed answer per conv_id across variants, so:
#   (a) whitespace_only    -> canonicalize (majority byte form; tie -> the
#                             form with the lexicographically smallest sha256
#                             — order-independent, a pure function of the
#                             {variant -> answer} map);
#   (b) prefix_truncation  -> canonicalize to the maximal superstring (the
#                             byte form whose normalized text is the longest;
#                             majority/sha tie-break among equals);
#   (c) substantive        -> EXCLUDE the conv_id (manifest persisted; the
#                             conv_id leaves the required set and phase_b
#                             must drop it — never scaffold-fallback).
# A hard raise is RETAINED for the beyond-tail regime: substantive exclusions
# past max(20, 2% of the stripped union) mean systemic upstream divergence,
# not a rendering tail — investigate, never auto-drop.
def _norm_ws(s: str) -> str:
    """Whitespace-collapse normalization: every whitespace run -> one space,
    ends stripped. This is the DEFINED normalization for class (a)."""
    return " ".join(s.split())


def _answer_sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _classify_conflict(answers: list[str]) -> str:
    """Classify one conv_id's conflicting cross-variant answer set."""
    norms = sorted({_norm_ws(a) for a in answers}, key=len)
    if len(norms) == 1:
        return "whitespace_only"
    longest = norms[-1]
    if all(longest.startswith(n) and n != longest for n in norms[:-1]):
        return "prefix_truncation"
    return "substantive"


def _canonical_answer(per_variant: dict[str, str], cls: str) -> str:
    """Deterministic, order-independent canonical byte form for a resolvable
    conflict: one vote per variant, majority byte form wins; ties break to the
    lexicographically smallest sha256. For prefix_truncation the candidate set
    is first restricted to forms carrying the maximal normalized superstring."""
    forms = list(per_variant.values())
    if cls == "prefix_truncation":
        longest = max((_norm_ws(a) for a in forms), key=len)
        forms = [a for a in forms if _norm_ws(a) == longest]
    votes: dict[str, int] = {}
    for a in forms:
        votes[a] = votes.get(a, 0) + 1
    best = max(votes.values())
    return min((a for a, n in votes.items() if n == best), key=_answer_sha)


def _collect_scaffold_answers(
    rows: list[dict],
    variant: str,
    stripped_needed: set[str],
    kept_cap: int | None,
    counters: dict[str, int],
    collected: dict[str, dict[str, str]],
) -> None:
    """Pure collection pass over one variant's admitted scaffold rows into
    ``collected[cid][variant] = answer`` (separated from staging so the gates
    are probe-able offline). ``kept_cap`` caps NEW conv_ids only — additional
    variants of an already-collected conv_id always land, so the smoke slice
    still exercises cross-variant resolution."""
    for row in rows:
        counters["scaffold_rows_read"] += 1
        cid = str(row.get("conv_id") or row.get("scaffold_id") or "")
        if cid not in stripped_needed:
            continue
        ans = str(row.get("answer") or "")
        if not ans:
            counters["missing_answer_field"] += 1
            continue
        if sc.SLOT_SENTINEL in ans:
            counters["sentinel_in_answer"] += 1
            continue
        per = collected.get(cid)
        if per is None:
            if kept_cap is not None and len(collected) >= kept_cap:
                continue
            per = collected[cid] = {}
        if variant in per:
            # Same conv_id twice within ONE variant's pool: phase_a asserts
            # this never happens (plan assumption 29); count, keep the first.
            counters["intra_variant_duplicate"] += 1
            continue
        per[variant] = ans


def _resolve_answer_conflicts(
    collected: dict[str, dict[str, str]],
) -> tuple[dict[str, str], set[str], dict[str, int], list[dict]]:
    """Resolve cross-variant answer sets -> (answers, excluded, tallies, audit).

    Audit rows are digest-only (conv_ids, sha256 prefixes, lengths — NEVER
    answer text; LMSYS-derived corpus)."""
    answers: dict[str, str] = {}
    excluded: set[str] = set()
    tallies = {
        "cross_variant_conflict": 0,
        "conflict_ws_canonicalized": 0,
        "conflict_prefix_canonicalized": 0,
        "conflict_substantive_excluded": 0,
    }
    audit: list[dict] = []
    for cid in sorted(collected):
        per = collected[cid]
        distinct = set(per.values())
        if len(distinct) == 1:
            answers[cid] = next(iter(distinct))
            continue
        tallies["cross_variant_conflict"] += 1
        cls = _classify_conflict(list(per.values()))
        row = {
            "conv_id": cid,
            "class": cls,
            "n_variants_present": len(per),
            "n_distinct_answers": len(distinct),
            "per_variant": {
                v: {"sha8": _answer_sha(a)[:8], "chars": len(a)} for v, a in per.items()
            },
        }
        if cls == "substantive":
            excluded.add(cid)
            tallies["conflict_substantive_excluded"] += 1
            row["disposition"] = "excluded"
            _log(f"conflict conv_id={cid} class=substantive -> EXCLUDED")
        else:
            canonical = _canonical_answer(per, cls)
            answers[cid] = canonical
            key = (
                "conflict_ws_canonicalized"
                if cls == "whitespace_only"
                else "conflict_prefix_canonicalized"
            )
            tallies[key] += 1
            row["disposition"] = "canonicalized"
            row["canonical_sha8"] = _answer_sha(canonical)[:8]
        audit.append(row)
    return answers, excluded, tallies, audit


def _scaffold_answers(
    staging_root: Path,
    variants: list[str],
    needed: set[str],
    kept_cap: int | None,
) -> tuple[dict[str, str], dict, set[str], list[dict]]:
    """{stripped_conv_id -> ORIGINAL answer} from the admitted scaffold rows,
    cross-variant conflicts resolved per the r12 policy above. Returns
    (answers, counters, excluded_conv_ids, conflict_audit_rows)."""
    counters = {
        "scaffold_rows_read": 0,
        "stripped_hits": 0,
        "missing_answer_field": 0,
        "sentinel_in_answer": 0,
        "intra_variant_duplicate": 0,
        "cross_variant_conflict": 0,
        "conflict_ws_canonicalized": 0,
        "conflict_prefix_canonicalized": 0,
        "conflict_substantive_excluded": 0,
        "cross_variant_conflicts_hard": 0,
    }
    stripped_needed = {cid for cid in needed if cid.startswith("stripped_")}
    if not stripped_needed:
        return {}, counters, set(), []
    collected: dict[str, dict[str, str]] = {}
    for v in variants:
        pool = _stage_sharded_jsonl(
            staging_root / SCAFFOLDS_PREFIX / v, f"{SCAFFOLDS_PREFIX}/{v}", f"scaffolds_{v}"
        )
        _collect_scaffold_answers(
            pa._read_jsonl(pool), v, stripped_needed, kept_cap, counters, collected
        )
    answers, excluded, tallies, audit = _resolve_answer_conflicts(collected)
    counters.update(tallies)
    counters["stripped_hits"] = len(answers)
    cap = max(20, math.ceil(0.02 * len(stripped_needed)))
    if counters["conflict_substantive_excluded"] > cap:
        counters["cross_variant_conflicts_hard"] = counters["conflict_substantive_excluded"]
        raise RuntimeError(
            f"cross_variant_conflicts_hard={counters['cross_variant_conflicts_hard']}: "
            f"substantive cross-variant answer conflicts exceed the tail cap "
            f"({counters['conflict_substantive_excluded']} > {cap} = max(20, 2% of "
            f"{len(stripped_needed)} stripped conv_ids)) — systemic upstream answer "
            "divergence, not a rendering tail; investigate the parent #1345 stories "
            "before building this pool"
        )
    _log(
        "conflict resolution: "
        f"cids={counters['cross_variant_conflict']} "
        f"ws_canonicalized={counters['conflict_ws_canonicalized']} "
        f"prefix_canonicalized={counters['conflict_prefix_canonicalized']} "
        f"substantive_excluded={counters['conflict_substantive_excluded']} "
        f"hard={counters['cross_variant_conflicts_hard']}"
    )
    return answers, counters, excluded, audit


# ---------------------------------------------------------------------------
# Source 2: mt_* answers from the #1738 manifest (pinned revision)
# ---------------------------------------------------------------------------
def _stage_draw_jsonl(staging_root: Path) -> Path:
    """Stage the seed-137 shared question draw MANIFEST-FIRST via the hardened
    sharded stager (the r15 fix, closing epm:failure v4): the top-up gen leg
    uploads the >9.5 MB draw in the SHARDED form only
    (``shared_question_draw.shardNN.jsonl`` + ``.manifest.json``), leaving the
    plain hub name as stale prior-round residue — the r6 class the module
    docstring bans for scaffold pools, previously un-applied to the draw path.
    Falls back to the unsharded hub name ONLY when no manifest exists on HF
    (pre-shard compat); a missing SHARD under an existing manifest stays
    fail-loud inside ``_stage_sharded_jsonl`` (a fallback there would consume
    the stale residue). Logs which path was taken."""
    from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file

    dest_dir = staging_root / SCAFFOLDS_PREFIX
    manifest_in_repo = f"{SCAFFOLDS_PREFIX}/shared_question_draw.manifest.json"
    api = _api()
    has_manifest = retry_transient(
        lambda: api.file_exists(pa.HF_DATA_REPO, manifest_in_repo, repo_type="dataset"),
        what=f"file_exists({manifest_in_repo})",
    )
    if has_manifest:
        _log("draw staging: manifest present on HF -> sharded stager (manifest-first)")
        return _stage_sharded_jsonl(dest_dir, SCAFFOLDS_PREFIX, "shared_question_draw")
    _log(
        "draw staging: NO shared_question_draw.manifest.json on HF -> "
        "pre-shard compat fallback to the unsharded hub name"
    )
    return stage_hub_file(
        pa.HF_DATA_REPO,
        f"{SCAFFOLDS_PREFIX}/shared_question_draw.jsonl",
        dest_dir / "shared_question_draw.jsonl",
        repo_type="dataset",
        overwrite=True,
    )


def _draw_questions(staging_root: Path, needed: set[str]) -> dict[str, str]:
    """{mt_conv_id -> drawn question} for the consistency assert."""
    path = _stage_draw_jsonl(staging_root)
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row.get("conv_id") or "")
            if cid in needed:
                out[cid] = str(row.get("question") or "")
    return out


def _ensure_manifest_revision_pin(manifest_dir: Path, revision: str) -> None:
    """Staged manifest parts are reused across runs (stage_hub_file skips
    existing targets), so a revision change must WIPE the stale mirror —
    reused bytes from another snapshot silently change the source_hash-derived
    conv_id keys."""
    pin = manifest_dir / ".staging_revision.json"
    if manifest_dir.is_dir():
        prev = None
        if pin.is_file():
            prev = json.loads(pin.read_text(encoding="utf-8")).get("revision")
        if prev == revision:
            return
        # Unknown-provenance or other-revision mirror: wipe (reused bytes from
        # another snapshot silently change the conv_id keys).
        _log(f"staged manifest revision {str(prev)[:12]} != pinned {revision[:12]} — restaging")
        shutil.rmtree(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"revision": revision}), encoding="utf-8")


def _manifest_answers(
    staging_root: Path,
    revision: str,
    needed: set[str],
    draw_q: dict[str, str],
    *,
    max_scan_rows: int | None,
    kept_cap: int | None,
) -> tuple[dict[str, str], dict]:
    """Scan the #1738 manifest at the pinned revision in the draw's exact
    order (sorted part filenames, in-file order, first-wins per conv_id) and
    collect, per needed ``mt_*`` conv_id, the first assistant message AFTER
    the first user turn. Parts are staged lazily so a capped smoke stages
    only what it scans."""
    from explore_persona_space.orchestrate.hub import (
        list_hf_files_under_path,
        retry_transient,
        stage_hub_file,
    )

    counters = {
        "parts_scanned": 0,
        "rows_scanned": 0,
        "not_required": 0,
        "dupe_required_row": 0,
        "matched_kept": 0,
        "no_user_turn": 0,
        "no_assistant_reply": 0,
        "empty_answer": 0,
        "sentinel_in_answer": 0,
        "question_mismatch": 0,
    }
    answers: dict[str, str] = {}
    mt_needed = {cid for cid in needed if cid.startswith("mt_")}
    if not mt_needed:
        return answers, counters

    manifest_dir = staging_root / pa.QUESTION_MANIFEST_PREFIX
    _ensure_manifest_revision_pin(manifest_dir, revision)
    api = _api()
    part_paths = retry_transient(
        lambda: list_hf_files_under_path(
            api,
            pa.HF_DATA_REPO,
            pa.QUESTION_MANIFEST_PREFIX,
            repo_type="dataset",
            revision=revision,
        ),
        what=f"list_hf_files_under_path({pa.QUESTION_MANIFEST_PREFIX})",
    )
    part_paths = sorted(p for p in part_paths if p.rsplit("/", 1)[-1].startswith("part_"))
    if not part_paths:
        raise FileNotFoundError(f"no part_*.jsonl under {pa.QUESTION_MANIFEST_PREFIX}@{revision}")

    mismatched: list[str] = []
    remaining = set(mt_needed)
    done = False
    for path_in_repo in part_paths:
        if done:
            break
        local = stage_hub_file(
            pa.HF_DATA_REPO,
            path_in_repo,
            manifest_dir / path_in_repo.rsplit("/", 1)[-1],
            repo_type="dataset",
            revision=revision,
        )
        counters["parts_scanned"] += 1
        with local.open(encoding="utf-8") as f:
            for line in f:
                if max_scan_rows is not None and counters["rows_scanned"] >= max_scan_rows:
                    done = True
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                counters["rows_scanned"] += 1
                cid = "mt_" + str(row.get("source_hash") or "").removeprefix("sha:")[:12]
                if cid not in mt_needed:
                    counters["not_required"] += 1
                    continue
                if cid not in remaining:
                    # First-wins: the draw's seen_cid dedupe kept the FIRST
                    # row with this key in the same scan order.
                    counters["dupe_required_row"] += 1
                    continue
                remaining.discard(cid)  # processed now, whatever the outcome
                msgs = row.get("messages") or []
                u_idx = next(
                    (i for i, m in enumerate(msgs) if m.get("role") == "user"),
                    None,
                )
                if u_idx is None:
                    counters["no_user_turn"] += 1
                    continue
                q = str(msgs[u_idx].get("content") or "").strip()
                expect_q = draw_q.get(cid)
                if expect_q is not None and q != expect_q:
                    counters["question_mismatch"] += 1
                    mismatched.append(cid)
                    continue
                a_msg = next(
                    (m for m in msgs[u_idx + 1 :] if m.get("role") == "assistant"),
                    None,
                )
                if a_msg is None:
                    counters["no_assistant_reply"] += 1
                    continue
                ans = str(a_msg.get("content") or "").strip()
                if not ans:
                    counters["empty_answer"] += 1
                    continue
                if sc.SLOT_SENTINEL in ans:
                    counters["sentinel_in_answer"] += 1
                    continue
                answers[cid] = ans
                counters["matched_kept"] += 1
                if kept_cap is not None and counters["matched_kept"] >= kept_cap:
                    done = True
                    break
                if not remaining:
                    done = True
                    break
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} manifest first-user-turn(s) != drawn question "
            f"(first ids: {mismatched[:5]}) — turn-selection convention or "
            "revision drift; refusing to emit a mis-paired pool"
        )
    return answers, counters


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def _upload_pool(out_dir: Path, files: list[Path]) -> None:
    """Mirror the pool to HF in ONE bulk commit under
    ``issue2054_lattice/answers/`` (the ``_upload_scaffold_files`` pattern;
    >9.5 MB JSONLs ride the sharded form). Fail-loud: ``_upload_folder_filtered``
    is fail-soft by RETURN on every failure shape (missing token, incomplete
    expected-set verify, terminal exception -> ``""``), so the return is
    captured and an empty return raises — the hub.py canonical caller pattern
    (``upload_raw_completions_to_data_repo``) — making the documented
    exit-1 upload-failure contract reachable."""
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    files = pa._shard_large_jsonl_for_upload(files)
    allow = sorted({f.relative_to(out_dir).as_posix() for f in files if f.is_file()})
    if not allow:
        raise RuntimeError(f"upload set resolved EMPTY against declared outputs: {files}")
    expected = [f"{ANSWERS_PREFIX}/{rel}" for rel in allow]
    url = _upload_folder_filtered(
        out_dir,
        repo_id=pa.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=ANSWERS_PREFIX,
        allow_patterns=allow,
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"answers-pool HF mirror failed or incomplete -> {ANSWERS_PREFIX}/ "
            "(bulk upload returned no path; local files kept)"
        )
    _log(f"uploaded {len(allow)} pool file(s) in one bulk commit -> {ANSWERS_PREFIX}/")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _length_stats(answers: list[str]) -> dict:
    if not answers:
        return {"n": 0}
    lens = sorted(len(a) for a in answers)

    def _pct(p: float) -> int:
        return lens[min(len(lens) - 1, int(p * (len(lens) - 1)))]

    return {
        "n": len(lens),
        "chars_mean": round(sum(lens) / len(lens), 1),
        "chars_median": _pct(0.5),
        "chars_p10": _pct(0.10),
        "chars_p90": _pct(0.90),
        "chars_max": lens[-1],
    }


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    staging_root = Path(args.staging_dir).resolve()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    smoke = (
        args.max_scan_rows is not None
        or args.smoke_kept_cap is not None
        or bool(args.only_stripped_cids)
    )
    if smoke and not str(out_dir).startswith("/tmp/"):
        print(
            "ERROR: smoke caps (--max-scan-rows / --smoke-kept-cap / "
            "--only-stripped-cids) require a /tmp --out-dir — a capped pool at "
            "the production path is a residue trap for a later phase_b dispatch",
            file=sys.stderr,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    _log(
        f"start: out={out_dir} staging={staging_root} variants={variants} "
        f"smoke={smoke} max_scan_rows={args.max_scan_rows} kept_cap={args.smoke_kept_cap}"
    )

    revision, draw_meta = _pinned_revision(staging_root, args.manifest_revision)
    _log(f"manifest revision pinned: {revision[:12]} (draw n={draw_meta.get('n')})")
    required, kept_record, per_variant_admitted = _required_conv_ids(staging_root, variants)
    n_mt = sum(1 for x in required if x.startswith("mt_"))
    n_stripped = sum(1 for x in required if x.startswith("stripped_"))
    n_other = len(required) - n_mt - n_stripped
    if n_other:
        raise RuntimeError(f"{n_other} admitted conv_ids in neither key space (mt_/stripped_)")
    _log(f"required conv_ids: {len(required)} (mt={n_mt} stripped={n_stripped})")
    if args.only_stripped_cids:
        only = {x.strip() for x in args.only_stripped_cids.split(",") if x.strip()}
        unknown = sorted(only - required)
        if unknown:
            raise RuntimeError(
                f"--only-stripped-cids names {len(unknown)} conv_id(s) not in the "
                f"admitted union (first: {unknown[:5]})"
            )
        required &= only
        _log(f"SMOKE --only-stripped-cids: required narrowed to {len(required)} conv_id(s)")

    scaffold_pool, scaffold_counters, excluded_cids, conflict_audit = _scaffold_answers(
        staging_root, variants, required, args.smoke_kept_cap
    )
    if excluded_cids:
        required -= excluded_cids
        _log(
            f"EXCLUDED {len(excluded_cids)} substantive-conflict stripped conv_id(s) "
            "from the required set (cross-variant answer divergence — 2x2 byte-fixed "
            "answer invariant unsatisfiable for these rows); phase_b must DROP them "
            "(exclusion manifest rides the pool output)"
        )
    mt_required = {x for x in required if x.startswith("mt_")}
    draw_q = _draw_questions(staging_root, mt_required)
    missing_draw = sorted(mt_required - draw_q.keys())
    if missing_draw:
        raise RuntimeError(
            f"{len(missing_draw)} required mt_* conv_id(s) absent from "
            f"shared_question_draw.jsonl (first ids: {missing_draw[:5]}) — the "
            "admitted set is not a subset of the draw (inconsistent kept.json / "
            "draw pair), so the question-equality seal cannot bind for those rows"
        )
    manifest_pool, manifest_counters = _manifest_answers(
        staging_root,
        revision,
        required,
        draw_q,
        max_scan_rows=args.max_scan_rows,
        kept_cap=args.smoke_kept_cap,
    )

    rows: list[dict] = []
    for cid, ans in manifest_pool.items():
        rows.append({"conv_id": cid, "answer": ans, "answer_provenance": "manifest_original"})
    for cid, ans in scaffold_pool.items():
        rows.append(
            {"conv_id": cid, "answer": ans, "answer_provenance": "scaffold_recovered_original"}
        )
    rows.sort(key=lambda r: r["conv_id"])
    if not rows:
        print("ERROR: built ZERO pool rows (kept == 0)", file=sys.stderr)
        return 1

    covered = {r["conv_id"] for r in rows}
    missing = sorted(required - covered)
    if missing:
        # Persist whenever non-empty (a tolerated 0 < missing <= N run keeps
        # the id list too, matching the --allow-missing-required help text).
        pa._atomic_write_json(out_dir / "missing_required_conv_ids.json", {"missing": missing})
    if not smoke and len(missing) > args.allow_missing_required:
        print(
            f"ERROR: pool misses {len(missing)} required conv_id(s) "
            f"(> --allow-missing-required {args.allow_missing_required}); ids "
            f"persisted to {out_dir / 'missing_required_conv_ids.json'} — these "
            "rows would silently DROP (mt_*) or fall back (stripped_*) in phase_b",
            file=sys.stderr,
        )
        return 1

    # r12 conflict-resolution artifacts: digest-only audit + exclusion manifest
    # (ids + sha prefixes + lengths — NEVER answer text; LMSYS-derived corpus).
    # Both ride the pool upload; non-smoke runs also mirror them to the
    # committed audits dir so the resolution record lands in git.
    floor_report = {
        v: {
            "admitted": len(per_variant_admitted[v]),
            "substantive_excluded_hits": len(per_variant_admitted[v] & excluded_cids),
            "post_exclusion": len(per_variant_admitted[v] - excluded_cids),
            "floor": ADMISSION_FLOOR,
            "margin": len(per_variant_admitted[v] - excluded_cids) - ADMISSION_FLOOR,
        }
        for v in variants
    }
    audit_doc = {
        "artifact": "answers_conflicts_audit",
        "resolution_rules": {
            "whitespace_only": (
                "canonicalized: majority byte form across variants; tie -> "
                "lexicographically smallest sha256 (order-independent)"
            ),
            "prefix_truncation": (
                "canonicalized: maximal normalized superstring; majority/sha "
                "tie-break among byte forms carrying it"
            ),
            "substantive": "excluded from the required set (manifest below)",
            "normalization": 'norm(s) = " ".join(s.split())',
            "hard_cap": "substantive_excluded > max(20, 2% of stripped union) -> raise",
        },
        "tallies": {
            k: scaffold_counters[k]
            for k in (
                "cross_variant_conflict",
                "conflict_ws_canonicalized",
                "conflict_prefix_canonicalized",
                "conflict_substantive_excluded",
                "cross_variant_conflicts_hard",
            )
        },
        "per_variant_floor_report": floor_report,
        "conflicts": conflict_audit,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "metadata": c.metadata(args.seed, len(conflict_audit), Path(__file__).name),
    }
    exclusion_doc = {
        "artifact": "answers_excluded_conv_ids",
        "consumer": (
            "scripts/issue2054_phase_b.py — DROP these conv_ids (never "
            "scaffold-fallback: their cross-variant answers diverge substantively, "
            "so a per-variant fallback re-breaks the 2x2 byte-fixed answer invariant)"
        ),
        "excluded": sorted(excluded_cids),
        "n_excluded": len(excluded_cids),
        "class": "substantive",
        "per_variant_floor_report": floor_report,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "metadata": c.metadata(args.seed, len(excluded_cids), Path(__file__).name),
    }
    audit_path = out_dir / "answers_conflicts_audit.json"
    exclusion_path = out_dir / "answers_excluded_conv_ids.json"
    pa._atomic_write_json(audit_path, audit_doc)
    pa._atomic_write_json(exclusion_path, exclusion_doc)
    if not smoke:
        audits_dir = _REPO_ROOT / "eval_results" / "issue_2054" / "audits"
        audits_dir.mkdir(parents=True, exist_ok=True)
        pa._atomic_write_json(audits_dir / audit_path.name, audit_doc)
        pa._atomic_write_json(audits_dir / exclusion_path.name, exclusion_doc)
        _log(f"conflict audit + exclusion manifest mirrored -> {audits_dir}")
    for v, rec in floor_report.items():
        _log(
            f"floor report variant={v}: admitted={rec['admitted']} "
            f"-{rec['substantive_excluded_hits']} excluded -> "
            f"post_exclusion={rec['post_exclusion']} (floor {ADMISSION_FLOOR}, "
            f"margin +{rec['margin']})"
        )

    pool_path = out_dir / f"{POOL_STEM}.jsonl"
    pa._atomic_write_jsonl(pool_path, rows)
    meta = {
        "artifact": "phase_b_answers_pool",
        "consumer": "scripts/issue2054_phase_b.py --answers-source",
        "provenance_note": (
            "authored-CHAT original conversation answers (2x2 row b); "
            "manifest_original = first assistant message after the first user "
            "turn at the pinned draw revision; scaffold_recovered_original = "
            "stripper-preserved original answer from the admitted pools"
        ),
        "manifest_revision": revision,
        "draw_meta_fingerprint": draw_meta.get("fingerprint"),
        "variants": variants,
        "required": {"total": len(required), "mt": n_mt, "stripped": n_stripped},
        "rows": len(rows),
        "rows_by_provenance": {
            "manifest_original": len(manifest_pool),
            "scaffold_recovered_original": len(scaffold_pool),
        },
        "missing_required": len(missing),
        "smoke": smoke,
        "counters": {"manifest": manifest_counters, "scaffolds": scaffold_counters},
        "conflict_resolution": {
            "n_conflict_cids": scaffold_counters["cross_variant_conflict"],
            "ws_canonicalized": scaffold_counters["conflict_ws_canonicalized"],
            "prefix_canonicalized": scaffold_counters["conflict_prefix_canonicalized"],
            "substantive_excluded": scaffold_counters["conflict_substantive_excluded"],
            "hard": scaffold_counters["cross_variant_conflicts_hard"],
            "excluded_conv_ids_manifest": exclusion_path.name,
            "audit": audit_path.name,
            "per_variant_floor_report": floor_report,
        },
        "kept_record": kept_record,
        "answer_length_stats": _length_stats([r["answer"] for r in rows]),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "metadata": c.metadata(args.seed, len(rows), Path(__file__).name),
    }
    meta_path = out_dir / f"{POOL_STEM}.meta.json"
    pa._atomic_write_json(meta_path, meta)
    _log(f"wrote {len(rows)} rows -> {pool_path} (+ {meta_path.name})")

    is_tmp = str(out_dir).startswith("/tmp/")
    if not is_tmp and not args.skip_upload:
        _upload_pool(out_dir, [pool_path, meta_path, audit_path, exclusion_path])
    else:
        _log("upload skipped (smoke /tmp tree or --skip-upload)")

    assert len(rows) > 0
    _log(
        f"done: kept={len(rows)} (manifest={len(manifest_pool)} "
        f"scaffold={len(scaffold_pool)}) missing_required={len(missing)} "
        f"conflicts_ws_canonicalized={scaffold_counters['conflict_ws_canonicalized']} "
        f"conflicts_prefix_canonicalized={scaffold_counters['conflict_prefix_canonicalized']} "
        f"conflicts_substantive_excluded={scaffold_counters['conflict_substantive_excluded']} "
        f"cross_variant_conflicts_hard={scaffold_counters['cross_variant_conflicts_hard']} "
        f"manifest_counters={manifest_counters} scaffold_counters={scaffold_counters}"
    )
    # phase-done terminal line (poller contract). No scripts/*.sh dispatcher
    # invokes this builder, so no workflow_lint phase-done waiver is needed.
    print("[phase=done]", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="data/issue_2054/answers")
    p.add_argument(
        "--staging-dir",
        default="data/issue_2054/hf_dl",
        help="mirror root for staged HF inputs (manifest parts, kept.json, scaffold shards)",
    )
    p.add_argument(
        "--variants",
        default=",".join(pa.DEFAULT_VARIANTS),
        help="comma-separated variant list whose admitted conv_ids define coverage",
    )
    p.add_argument(
        "--manifest-revision",
        default=None,
        help="override the draw-pinned #1738 manifest revision (default: "
        "shared_question_draw.meta.json 'revision')",
    )
    p.add_argument(
        "--max-scan-rows",
        type=int,
        default=None,
        help="SMOKE: cap total streamed manifest rows (requires /tmp --out-dir)",
    )
    p.add_argument(
        "--smoke-kept-cap",
        type=int,
        default=None,
        help="SMOKE: cap kept answers per source (requires /tmp --out-dir)",
    )
    p.add_argument(
        "--only-stripped-cids",
        default=None,
        help="SMOKE: comma-separated conv_ids — narrow the required set to "
        "exactly these (conflict-resolution slice probe; requires /tmp "
        "--out-dir; skips the manifest stream when no mt_* id is named)",
    )
    p.add_argument(
        "--allow-missing-required",
        type=int,
        default=0,
        help="tolerate up to N required conv_ids without a pool answer "
        "(default 0 = fail loud; missing ids are persisted either way)",
    )
    p.add_argument("--seed", type=int, default=137, help="metadata only — no sampling here")
    p.add_argument("--skip-upload", action="store_true", help="skip the HF mirror step")
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
