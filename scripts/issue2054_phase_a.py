#!/usr/bin/env python
"""Phase A driver for task #2054: diverse scaffold supply + judge admission + fold map.

Three legs (plan §4 Phase A / §10 `off_pod_phases`), selected by ``--stage``:

- RECOVERY (stage gen|all): strip scaffolds from parent #1345's existing
  on-policy stories via ``issue1345_strip_scaffolds.strip_file`` (byte-exact
  round-trip verified per row).
- GENERATION (stage gen|all; C8(i)): the recovered supply is far below the
  8,000-conv-id target (gate-4 floor 4,480), so Phase A GENERATES the
  shortfall by invoking the parent generator ``issue1345_gen_scaffolds.py
  --phase scaffolds`` as a subprocess, once per variant, against ONE shared
  question draw (plan req 1: one shared conversation draw underlies every
  cell). The draw is materialized once from the #1738 multi-turn manifest
  (``issue1738_multiturn/sampling_manifest``, seed 137, revision-pinned,
  exact-dupe deduped, token-budget validated — the #1738/#952 over-length
  add_request class is filtered at draw time). ``--gen-mock`` threads the
  generator's deterministic ``--mock`` path for CPU smokes. Sequential across
  variants; per-GPU sharding is Unit F.
- JUDGE ADMISSION (stage judge|all; C8(ii)): every scaffold row (recovered +
  generated) is judge-scored per row via ``eval.graded_judge.judge_graded``
  (claude-sonnet-4-5-20250929, ``max_tokens=1024``, reason-then-score rubric
  conforming to the harness parse contract ``parse_judge_json`` ->
  ``_score_from_parsed`` — llm-judging rule 27; the earlier two-field rubric
  dropped 100% of draws). Rows with mean score >= ``--judge-keep-threshold``
  (default 50, persona-vectors convention) are ADMITTED; drops are counted
  (content-drop vs below-threshold vs transport, rule 24 — transport residue
  fails LOUD, never a silent drop). A >=5k-call wave is pilot-gated first
  (rule 26); a pilot FAIL exits rc=7 with the report JSON (designed halt).
  Admitted rows are written to the canonical ``scaffolds_{variant}.jsonl``
  the downstream phases consume, the full pool to
  ``scaffolds_{variant}_prejudge.jsonl``, and the admission record to
  ``kept.json`` (plan output ``issue2054_lattice/scaffolds/kept.json``).

Writes ``eval_results/issue_2054/shared_fold_map.json`` (K=5
conversation-grouped folds, seed 137) from the ADMITTED conv_ids — the SINGLE
artifact every downstream fit/ladder invocation consumes. Recovered rows keep
``conv_id == scaffold_id`` (``stripped_<story_id>`` — the Unit B fold-map key
pin; phase_d canonizes via ``_canon_conv_id``); generated rows carry
``conv_id == qid`` (the ``mt_<hash>`` manifest draw key, which
``_canon_conv_id`` passes through unchanged).

Emits ``[phase=phase_a]`` log lines terminating in ``[phase=done]``.
Exit 0 on success. Exit 1 on judge / HF / preflight failure. Exit 2 on
missing dependency. Exit 7 on a rule-26 pilot-gate refusal (report JSON at
``<out>/_judge_cache/pilot_gate_report.json``).

The parent registries live at import top-level; a tokenizer/regex compile
reads EPM_STORY_CHARACTER_NAME + EPM_I1345_VARIANT — Phase A does neither,
so the strict [A-Za-z0-9_]+ default ("ARIA", "") passes untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_resume as resume  # noqa: E402

# Default cell subset (plan §4 v4 lattice character panel + assistant scope);
# the CLI --variants flag overrides.
DEFAULT_VARIANTS = (
    "char_helios",
    "char_wren",
    "char_dana",
    "char_vex",
    "conversation_paired_stories_assistant",
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PARENT_PREFIX = "issue1345_framing"
TASK_PREFIX = "issue2054_lattice"

# Shared question draw source (plan: "this task's corpus build draws from
# #1738 by default"): the persisted #1738 selection manifest (99,778 real
# multi-turn conversations, 55 JSONL parts, ~455 MB — a bounded fetch).
QUESTION_MANIFEST_PREFIX = "issue1738_multiturn/sampling_manifest"
QUESTION_MIN_CHARS = 16
# Scaffold-admissibility bounds, MEASURED on the 2026-08-05 production gen
# output (34,335 generated scaffolds across the 5 variants; the plan-req-6
# verbatim-question filter is what selects on them):
#
#   filter                 min per-variant verbatim-keep   corpus kept
#   none  (max_chars=8000)              26.9%                100.0%
#   single-line                         34.2%                 77.9%
#   single-line + <400                  36.0%                 72.6%   <- chosen
#   single-line + <200                  37.6%                 66.0%
#
# A MULTILINE question keeps at 1.3-3.0%, and an >=800-char question at
# 1.1-2.8% — the generator paraphrases/truncates them, so `question in
# scaffold_text` fails and the row is rejected DOWNSTREAM, after its
# generation compute is already spent. Admitting them cost ~74% of the first
# production draw (char_helios: generator kept 6,837, only 2,031 survived
# verbatim admission) and left every variant 6-12% short of kill gate 4.
# Filtering here makes that selection EXPLICIT and cheap instead of implicit
# and paid-for. <400 (not <200) keeps 72.6% of the corpus — the narrowest
# distributional cut that clears the yield need.
#
# SCOPE CAVEAT (carry into the clean-result): the eval distribution is now
# single-line real-user questions under 400 chars, not the full corpus.
QUESTION_MAX_CHARS = 400
QUESTION_SINGLE_LINE = True
# Prompt-budget cap for the generator render (gen MAX_MODEL_LEN 4096 −
# SCAFFOLD_MAX_NEW_TOKENS 1024 − ~500 tokens of scaffold-prompt overhead,
# with margin). Over-budget questions are DROPPED, never truncated (#952).
QUESTION_MAX_TOKENS = 2048
QUESTION_TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"

# Admission floor report (on-policy-completions.md 80% floor). Phase A
# REPORTS + FLAGS below-floor variants in kept.json; the registered DROP
# mechanism for this design is kill gate 4 (fits-side intersection >= 4,480),
# so no variant is silently excluded here.
YIELD_FLOOR_FRACTION = 0.8

# Upload shard budget (upload-policy.md: text >9.5 MB per file line-splits
# into <9 MB shards — never gzip; >10 MB force-routes to LFS).
UPLOAD_SHARD_LIMIT_BYTES = 9_500_000
UPLOAD_SHARD_TARGET_BYTES = 9_000_000

# The character-name → parent-story character mapping. parent stories were
# written with EPM_STORY_CHARACTER_NAME=<Name>; strip must be told the same
# name so the answer-attribution regex matches. The GENERATION leg reuses the
# same in-scene name (downstream splice/render reads row["character"]).
_CHAR_NAME_FROM_VARIANT = {
    "char_helios": "Helios",
    "char_wren": "Wren",
    "char_dana": "Dana",
    "char_vex": "Vex",
    "conversation_paired_stories_assistant": "Assistant",
    "conversation_paired_stories": "ARIA",  # legacy v8 scope; addressable
}


class PilotGateRefusal(RuntimeError):
    """Rule-26 pilot gate FAILed — a designed halt (exit 7), not a crash."""


def _scaffold_judge_rubric() -> str:
    """The per-row admission rubric (llm-judging rules 6/7/27).

    Single 0-100 reason-then-score instrument with anchored endpoints,
    ``{question}``/``{answer}`` slots (the ``judge_graded`` ``.replace``
    substitution), and a reply shape whose parse lands a ``score`` key —
    ``parse_judge_json`` -> ``_score_from_parsed`` keeps it. The scaffold
    text rides the ``{answer}`` slot. (The superseded two-field
    ``{"diverse": ..., "single_question": ...}`` shape had NO ``score`` key,
    so every draw parse-dropped — the C8(iii) pilot FAIL.)
    """
    import issue1345_scaffold_common as sc

    return (
        "You are grading a narrative SCAFFOLD for an experiment. A scaffold is a "
        "short prose scene that embeds exactly ONE question and a literal "
        f"answer-slot sentinel '{sc.SLOT_SENTINEL}' where an answer will later be "
        "spliced.\n\n"
        "Grade how usable the scaffold is, on one 0-100 scale:\n"
        "- 0: degenerate or template-like (a bare Q&A wrapper with no scene, "
        "repeated boilerplate), or the scene mangles or paraphrases the question, "
        "or other questions compete with it.\n"
        "- 50: borderline — a thin but real scene; the question appears verbatim "
        "but the framing is generic.\n"
        "- 100: a vivid, specific scene (setting, situation, and register all "
        "realized) in which the question below appears verbatim as the scene's "
        "single question.\n\n"
        "Question:\n{question}\n\n"
        "Scaffold:\n{answer}\n\n"
        "Before scoring, reason briefly. Output a single JSON object of the form "
        '{"reasoning": "<one or two sentences>", "score": <an integer from 0 to '
        "100>} — the reasoning key FIRST, then the score."
    )


def _log(msg: str) -> None:
    print(f"[phase=phase_a] {msg}", flush=True)


def _rel(path: Path) -> str:
    """Best-effort repo-root-relative string; falls back to abs for /tmp/*."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _serialize_report(rep: object) -> object:
    if is_dataclass(rep) and not isinstance(rep, type):
        return asdict(rep)
    if hasattr(rep, "_asdict"):
        return rep._asdict()
    if isinstance(rep, dict):
        return rep
    return {k: v for k, v in vars(rep).items() if not k.startswith("_")}


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration — NEVER splitlines() (gotchas.md U+2028 class)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=float)
    os.replace(tmp, path)


def _metadata(seed: int, n: int) -> dict:
    import issue1345_common as c

    return c.metadata(seed, n, Path(__file__).name)


def _canon_conv_id(conv_id: str) -> str:
    """Parent-story key space (phase_d's `_canon_conv_id` twin): recovered rows
    read `stripped_<story_id>`; a bare id (incl. the generated `mt_*` draw
    keys) passes through unchanged."""
    return conv_id.removeprefix("stripped_")


def _question_of(row: dict) -> str | None:
    """The judge/verbatim question for a scaffold row, or None.

    Generated rows carry an explicit ``question`` field; recovered (stripped)
    rows carry the Unit A ``q_start``/``q_end`` char span into
    ``scaffold_text`` (field sets preserved — the Unit A carry-forward).
    """
    q = str(row.get("question") or "").strip()
    if q:
        return q
    text = str(row.get("scaffold_text") or "")
    q_start, q_end = row.get("q_start"), row.get("q_end")
    if isinstance(q_start, int) and isinstance(q_end, int) and 0 <= q_start < q_end <= len(text):
        span = text[q_start:q_end].strip()
        return span or None
    return None


def _conv_grouped_folds(conv_ids: list[str], k: int, seed: int) -> dict[str, int]:
    """Assign each conv_id to a fold in [0, k) via seeded hash-based bucketing.

    Conversation-grouped by construction (one conv_id => one fold; ood-
    generalization-folds.md). Deterministic under (seed, k) alone — no
    reliance on iteration order of a set.
    """
    fold_map: dict[str, int] = {}
    for cid in sorted(set(conv_ids)):
        h = hashlib.sha256(f"{seed}:{cid}".encode()).digest()
        # First 8 bytes as an unsigned int, mod k.
        idx = int.from_bytes(h[:8], "big") % k
        fold_map[str(cid)] = idx
    return fold_map


def _recovery_story_files(all_paths: list[str]) -> list[str]:
    """The parent story JSONLs recovery may ingest: KEPT, NON-op only.

    The parent stores kept + raw + retry + judge files side by side
    (`kept_stories_paired_instruct.jsonl`, `raw_stories_paired_instruct*.jsonl`,
    `judge_results_*.jsonl`, and for the assistant variant also the op-mode
    `kept_stories_paired_op_instruct.jsonl`). Recovery ingests ONLY the kept
    non-op stories: raw files duplicate every kept story (Unit F smoke:
    char_helios read 806 duplicate conv_ids and the plan-assumption-29 assert
    killed the gen leg) and additionally contain the parent's judge-REJECTED
    stories — the wrong pool; the `_op_` stories are cell (c) source material
    (phase_d's), not lattice scaffolds. Sorted for deterministic strip order.
    """
    out = []
    for p in all_paths:
        name = p.rsplit("/", 1)[-1]
        if p.endswith(".jsonl") and name.startswith("kept_stories_") and "_op_" not in name:
            out.append(p)
    return sorted(out)


def _recover_scaffolds_from_hf(variants: list[str], api) -> dict[str, list[dict]]:
    """Download parent kept-stories JSONLs per variant and strip → scaffolds.

    Returns {variant: [scaffold row dicts]}. Uses parent
    `issue1345_strip_scaffolds.strip_file` (round-trip byte-exact per row).
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import (
        list_hf_files_under_path,
        retry_transient,
    )

    import issue1345_strip_scaffolds as strip_mod  # noqa: E402

    recovered: dict[str, list[dict]] = {}
    for variant in variants:
        if variant not in _CHAR_NAME_FROM_VARIANT:
            # M3 (review r2 Minor 1): the silent "ARIA" default strips parent
            # stories under the wrong character name -> kept=0 silently for an
            # unmapped --variants entry. Fail loud like the phase_b/c/d twins.
            raise ValueError(
                f"cannot resolve character name: recovery variant {variant!r} is not "
                "in _CHAR_NAME_FROM_VARIANT — extend the map for a new variant"
            )
        char_name = _CHAR_NAME_FROM_VARIANT[variant]
        # Parent stores kept stories under
        # issue1345_framing/{variant}/raw_completions/stories/*.jsonl
        story_prefix = f"{PARENT_PREFIX}/{variant}/raw_completions/stories"
        try:
            all_paths = list_hf_files_under_path(
                api, HF_DATA_REPO, story_prefix, repo_type="dataset"
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"variant={variant} story-prefix listing FAILED: {exc}")
            recovered[variant] = []
            continue
        # KEPT non-op story files only — see _recovery_story_files (raw files
        # duplicate every kept story AND carry the parent's judge-rejected
        # stories; judge_results_*.jsonl have no `story` key at all).
        story_files = _recovery_story_files(all_paths)
        if not story_files:
            _log(f"variant={variant} no story JSONLs at {story_prefix}")
            recovered[variant] = []
            continue

        variant_out: list[dict] = []
        for path_in_repo in story_files:
            try:
                local = retry_transient(
                    lambda p=path_in_repo: hf_hub_download(
                        repo_id=HF_DATA_REPO,
                        repo_type="dataset",
                        filename=p,
                    ),
                    what=f"hf_hub_download({path_in_repo})",
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"variant={variant} download {path_in_repo} FAILED: {exc}")
                continue
            try:
                rows, counts = strip_mod.strip_file(
                    Path(local),
                    char_name,
                    require_single_turn=False,
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"variant={variant} strip {path_in_repo} FAILED: {exc}")
                continue
            # Reject-reason breakdown, not just kept/total: a `kept=12/2187`
            # recovery is only diagnosable from WHICH reason absorbed the rest
            # (`no_parsed_turns` = the strip parser found zero attributed turns
            # — the char-name / casing class, #2054's char_helios bug). Without
            # it the shortfall reads as a data-shape property of the parent.
            rejects = {
                k: v
                for k, v in sorted(counts.items())
                if k not in ("total", "kept", "multi_turn_kept_tail") and v
            }
            _log(
                f"variant={variant} file={Path(path_in_repo).name} "
                f"kept={counts.get('kept', 0)}/{counts.get('total', 0)}"
                + (f" rejects={rejects}" if rejects else "")
            )
            variant_out.extend(rows)
        # Each recovered scaffold keeps conv_id == scaffold_id
        # (`stripped_<story_id>` — the Unit B fold-map key pin; phase_d
        # canonizes via _canon_conv_id).
        for i, row in enumerate(variant_out):
            row.setdefault("conv_id", row.get("scaffold_id", f"{variant}_{i}"))
            row.setdefault("variant", variant)
            row.setdefault("provenance", "recovered")
        recovered[variant] = variant_out
    return recovered


def _shared_recovered_intersection(recovered: dict[str, list[dict]]) -> set[str]:
    """Canonical conv_ids recovered in EVERY variant (the shared-draw core)."""
    sets = [{_canon_conv_id(str(r.get("conv_id"))) for r in rows} for rows in recovered.values()]
    return set.intersection(*sets) if sets else set()


# ---------------------------------------------------------------------------
# Shared question draw (generation input; plan req 1 — ONE shared draw)
# ---------------------------------------------------------------------------
def _draw_shared_questions(
    n: int,
    seed: int,
    *,
    staging_dir: Path,
    manifest_prefix: str = QUESTION_MANIFEST_PREFIX,
    revision: str | None = None,
    manifest_dir: Path | None = None,
    tokenizer=None,
) -> tuple[list[dict], dict]:
    """Seeded draw of n first-user-turn questions from the #1738 manifest.

    Returns (rows, draw_record); each row is {"conv_id", "qid", "question"}
    with ``conv_id == qid == "mt_<source_hash[:12]>"`` (content-derived,
    stable across re-uploads; ``_canon_conv_id`` passes it through). Filters:
    non-empty first user turn, char bounds, no slot sentinel, exact-dupe
    dedupe (question text AND conv_id — the #1768 real-corpus dupes class),
    and a token cap against the generator's prompt budget (over-length rows
    are ENGINE-FATAL at vLLM add_request — the #1738 lesson — so they are
    dropped at draw time, never truncated).

    The manifest scan is a bounded fetch (55 parts, ~100k rows) — exempt from
    the external-stream checkpoint presumption. ``manifest_dir``/``tokenizer``
    are injection seams for offline tests; production stages via
    ``hub.stage_hub_prefix`` at ONE resolved revision.
    """
    import numpy as np

    import issue1345_scaffold_common as sc

    if manifest_dir is None:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_prefix

        if revision is None:
            revision = retry_transient(
                lambda: HfApi().repo_info(HF_DATA_REPO, repo_type="dataset").sha,
                what="repo_info(data repo)",
            )
        _log(f"question draw: staging {manifest_prefix} @ {revision[:12]} -> {staging_dir}")
        stage_hub_prefix(
            HF_DATA_REPO, manifest_prefix, staging_dir, repo_type="dataset", revision=revision
        )
        # stage_hub_prefix's dest is a MIRROR ROOT: files land at
        # dest/<repo-relative path> (gotchas.md stage_hub_prefix entry).
        manifest_dir = staging_dir / manifest_prefix
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"manifest dir missing after staging: {manifest_dir}")
    parts = sorted(manifest_dir.glob("part_*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"no manifest part_*.jsonl under {manifest_dir}")

    counters = {
        "scanned": 0,
        "no_user_turn": 0,
        "char_bounds": 0,
        "multiline": 0,
        "sentinel_in_question": 0,
        "dupe_question": 0,
        "dupe_conv_id": 0,
        "over_token_budget": 0,
    }
    candidates: list[dict] = []
    seen_q: set[str] = set()
    seen_cid: set[str] = set()
    for part in parts:
        with part.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                counters["scanned"] += 1
                msgs = row.get("messages") or []
                q = next(
                    (str(m.get("content") or "") for m in msgs if m.get("role") == "user"),
                    "",
                ).strip()
                if not q:
                    counters["no_user_turn"] += 1
                    continue
                if not (QUESTION_MIN_CHARS <= len(q) <= QUESTION_MAX_CHARS):
                    counters["char_bounds"] += 1
                    continue
                if QUESTION_SINGLE_LINE and "\n" in q:
                    counters["multiline"] += 1
                    continue
                if sc.SLOT_SENTINEL in q:
                    counters["sentinel_in_question"] += 1
                    continue
                if q in seen_q:
                    counters["dupe_question"] += 1
                    continue
                cid = "mt_" + str(row.get("source_hash") or "").removeprefix("sha:")[:12]
                if cid == "mt_" or cid in seen_cid:
                    counters["dupe_conv_id"] += 1
                    continue
                seen_q.add(q)
                seen_cid.add(cid)
                candidates.append({"conv_id": cid, "qid": cid, "question": q})

    # Token-budget filter (batched; CPU) — drop, never truncate (#952).
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(QUESTION_TOKENIZER_ID)
    kept: list[dict] = []
    chunk = 2048
    for i in range(0, len(candidates), chunk):
        batch = candidates[i : i + chunk]
        enc = tokenizer([r["question"] for r in batch], add_special_tokens=False)
        for r, ids in zip(batch, enc["input_ids"], strict=True):
            if len(ids) <= QUESTION_MAX_TOKENS:
                kept.append(r)
            else:
                counters["over_token_budget"] += 1

    if len(kept) < n:
        raise RuntimeError(
            f"question draw short: eligible={len(kept)} < requested n={n} (counters={counters})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(kept))[:n]
    drawn = [kept[int(i)] for i in idx]
    record = {
        "n": n,
        "seed": seed,
        "manifest_prefix": manifest_prefix,
        "revision": revision,
        "eligible": len(kept),
        "counters": counters,
        "filters": {
            "min_chars": QUESTION_MIN_CHARS,
            "max_chars": QUESTION_MAX_CHARS,
            "single_line": QUESTION_SINGLE_LINE,
            "max_tokens": QUESTION_MAX_TOKENS,
            "tokenizer": QUESTION_TOKENIZER_ID,
        },
    }
    _log(f"question draw: kept {n}/{len(kept)} eligible (counters={counters})")
    return drawn, record


def _question_pool_fingerprint(n: int, seed: int, revision: str | None) -> str:
    key = json.dumps(
        {
            "kind": "shared_question_draw_v1",
            "n": n,
            "seed": seed,
            "revision": revision,
            "min_chars": QUESTION_MIN_CHARS,
            "max_chars": QUESTION_MAX_CHARS,
            "single_line": QUESTION_SINGLE_LINE,
            "max_tokens": QUESTION_MAX_TOKENS,
            "manifest_prefix": QUESTION_MANIFEST_PREFIX,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _ensure_question_pool(args, n_needed: int, out_dir: Path) -> tuple[list[dict], dict]:
    """Load (--questions-jsonl) or draw-and-cache the shared question pool."""
    if args.questions_jsonl:
        rows = _read_jsonl(Path(args.questions_jsonl))
        for r in rows:
            if not (
                str(r.get("question") or "").strip()
                and str(r.get("qid") or r.get("conv_id") or "").strip()
            ):
                raise ValueError(
                    f"--questions-jsonl rows need non-empty 'question' + 'qid'/'conv_id': "
                    f"{args.questions_jsonl}"
                )
            r.setdefault("qid", r.get("conv_id"))
            r.setdefault("conv_id", r.get("qid"))
        if len(rows) < n_needed:
            raise ValueError(f"--questions-jsonl has {len(rows)} rows < shortfall n={n_needed}")
        return rows[:n_needed], {"source": str(args.questions_jsonl), "n": n_needed}

    pool_path = out_dir / "shared_question_draw.jsonl"
    meta_path = out_dir / "shared_question_draw.meta.json"
    fp = _question_pool_fingerprint(n_needed, args.seed, args.manifest_revision)
    if pool_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fp:
            rows = _read_jsonl(pool_path)
            _log(f"question draw: resume {len(rows)} rows from {_rel(pool_path)}")
            return rows, meta
        raise RuntimeError(
            f"{pool_path} exists with a DIFFERENT draw fingerprint "
            f"({meta.get('fingerprint')} != {fp}) — refusing to mix draws; "
            "move the stale file aside"
        )
    rows, record = _draw_shared_questions(
        n_needed,
        args.seed,
        staging_dir=out_dir / "_manifest_stage",
        revision=args.manifest_revision,
    )
    record["fingerprint"] = fp
    _atomic_write_jsonl(pool_path, rows)
    _atomic_write_json(meta_path, record)
    return rows, record


# ---------------------------------------------------------------------------
# Generation leg (C8(i)) — invoke the parent generator for the shortfall
# ---------------------------------------------------------------------------
def _gen_char_and_description(variant: str) -> tuple[str, str]:
    """(in-scene character name, generator description) for one variant.

    The name follows the lattice convention (row['character'] downstream);
    the description comes from the parent generator's CHARACTERS panel,
    matched case-insensitively (panel key 'HELIOS' vs lattice name 'Helios').
    Passing --description explicitly means panel-key casing never gates the
    subprocess.
    """
    import issue1345_gen_scaffolds as gen_mod

    name = _CHAR_NAME_FROM_VARIANT.get(variant)
    if not name or name == "ARIA":
        raise ValueError(f"no generation character mapping for variant {variant!r}")
    by_lower = {k.lower(): v for k, v in gen_mod.CHARACTERS.items()}
    desc = by_lower.get(name.lower())
    if not desc:
        raise ValueError(f"no generator description for variant {variant!r} (char {name!r})")
    return name, desc


def _generate_shortfall(
    variant: str,
    questions: list[dict],
    out_root: Path,
    *,
    seed: int,
    mock: bool,
    gen_model: str,
) -> tuple[list[dict], dict]:
    """Generate scaffolds for the SAME shared question draw via the parent
    generator subprocess; return (rows in phase-a schema, counts).

    make_scaffold_specs pairs question i with scaffold i (n == len(questions),
    so each question is used exactly once — conv_id uniqueness per variant by
    construction, plan assumption 29). The generator has its own fingerprint-
    gated per-chunk resume, so a re-invocation with identical inputs resumes.
    """
    char_name, description = _gen_char_and_description(variant)
    gen_dir = out_root / variant / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    qfile = gen_dir / "questions.jsonl"
    _atomic_write_jsonl(qfile, questions)

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "issue1345_gen_scaffolds.py"),
        "--phase",
        "scaffolds",
        "--character",
        char_name,
        "--description",
        description,
        "--model",
        gen_model,
        "--n",
        str(len(questions)),
        "--seed",
        str(seed),
        "--questions-jsonl",
        str(qfile),
        "--out-dir",
        str(gen_dir),
    ]
    if mock:
        cmd.append("--mock")
    _log(
        f"variant={variant} generation subprocess: n={len(questions)} char={char_name} mock={mock}"
    )
    # Explicit env passthrough (experiment-implementer subprocess contract);
    # load_dotenv() ran at module top.
    subprocess.run(cmd, check=True, env={**os.environ}, cwd=str(_REPO_ROOT))

    model_key = "mock" if mock else gen_model
    kept_path = gen_dir / f"scaffolds_{char_name.lower()}_{model_key}.jsonl"
    if not kept_path.is_file():
        raise FileNotFoundError(f"generator produced no kept file: {kept_path}")
    raw_rows = _read_jsonl(kept_path)

    rows: list[dict] = []
    n_not_verbatim = 0
    for r in raw_rows:
        q = str(r.get("question") or "")
        qid = str(r.get("qid") or "")
        if not (q and qid):
            raise AssertionError(
                f"generated scaffold missing question/qid: {r.get('scaffold_id')!r} "
                "(was --questions-jsonl threaded?)"
            )
        # Verbatim-question admission filter (plan req 6): a paraphrased
        # question breaks downstream span location — reject, never keep.
        if q not in str(r.get("scaffold_text") or ""):
            n_not_verbatim += 1
            continue
        rows.append({**r, "conv_id": qid, "variant": variant, "provenance": "generated"})
    # Plan req 6 non-regression ASSERT on the first 100 generated scaffolds
    # (upgraded from the earlier WARN; trivially true post-filter — it pins
    # the filter itself against regressions).
    for r in rows[:100]:
        assert r["question"] in r["scaffold_text"], (
            f"req-6 verbatim-question regression: {r['scaffold_id']}"
        )
    counts = {
        "requested": len(questions),
        "generator_kept": len(raw_rows),
        "question_not_verbatim": n_not_verbatim,
        "merged": len(rows),
    }
    _log(f"variant={variant} generation merged: {counts}")
    return rows, counts


# ---------------------------------------------------------------------------
# Judge admission (C8(ii)) — per-row gate emitting kept.json
# ---------------------------------------------------------------------------
def _variant_judge_items(
    variant: str, rows: list[dict]
) -> tuple[list[tuple[str, str, str]], list[dict], int]:
    """(items for judge_graded, the judged rows aligned 1:1, n_no_question).

    item_id = "<variant>-<i:06d>" — [a-zA-Z0-9_-], no "__", <= 53 chars
    (the Batch custom_id grammar + judge_graded's delimiter guard).
    """
    items: list[tuple[str, str, str]] = []
    judged: list[dict] = []
    n_no_question = 0
    for row in rows:
        q = _question_of(row)
        text = str(row.get("scaffold_text") or "")
        if not q or not text:
            n_no_question += 1
            continue
        item_id = f"{variant}-{len(judged):06d}"
        items.append((item_id, q, text))
        judged.append(row)
    return items, judged, n_no_question


def _admit_variant_rows(
    judged_rows: list[dict],
    items: list[tuple[str, str, str]],
    result,
    threshold: float,
) -> tuple[list[dict], dict]:
    """Apply the admission threshold to one variant's JudgeResult.

    Pure reduce (unit-testable). Drop-never-coerce: a None score with NO
    transport losses is a content drop (not admitted); a None score WITH
    transport losses raises (rule 24 — freely re-judgeable, never silently
    censored; the judge cache makes the re-run resumable).
    """
    admitted: list[dict] = []
    drops = {"below_threshold": 0, "judge_content_drop": 0}
    transport_failed: list[str] = []
    for row, (item_id, _q, _a) in zip(judged_rows, items, strict=True):
        score = result.scores.get(item_id)
        if score is None:
            if result.per_item_transport_losses.get(item_id, 0) > 0:
                transport_failed.append(item_id)
            else:
                drops["judge_content_drop"] += 1
            continue
        if score < threshold:
            drops["below_threshold"] += 1
            continue
        admitted.append({**row, "judge_score": float(score)})
    if transport_failed:
        raise RuntimeError(
            f"{len(transport_failed)} items lost ALL draws to transport "
            f"(rule 24 — re-run to re-judge; cache resumes): "
            f"{transport_failed[:10]}"
        )
    return admitted, drops


def _run_judge_pilot(variant_rows: dict[str, list[dict]], args, cache_root: Path) -> dict:
    """Rule-26 pilot at the exact production instrument; returns the report dict."""
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    arms: dict[str, list[tuple[str, str, str]]] = {}
    for variant, rows in variant_rows.items():
        items, _judged, _n_no_q = _variant_judge_items(variant, rows)
        if items:
            arms[variant] = items
    if not arms:
        return {"verdict": "PASS", "note": "no-scaffolds-to-judge"}
    report = judge_pilot_gate(
        arms,
        _scaffold_judge_rubric(),
        max_tokens=args.max_tokens,
        cache_dir=cache_root / "pilot_cache",
        save_raw_dir=cache_root / "pilot_raw",
        n_draws=max(1, args.judge_draws),
        target_total_draws=args.pilot_n,
        report_path=cache_root / "pilot_gate_report.json",
    )
    rep = _serialize_report(report)
    if hasattr(report, "to_json"):
        rep = report.to_json()
    return rep if isinstance(rep, dict) else {"verdict": str(rep)}


def _judge_regime(variant: str, args, rubric: str) -> dict:
    """Output-affecting regime for one variant's judge admission (C9/M6)."""
    return {
        "stage": "judge-admission",
        "variant": variant,
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest()[:16],
        "judge_draws": max(1, args.judge_draws),
        "max_tokens": int(args.max_tokens),
        "threshold": float(args.judge_keep_threshold),
    }


def _judge_admission(
    variant_rows: dict[str, list[dict]], args, out_dir: Path
) -> tuple[dict[str, list[dict]], dict, dict[str, Path]]:
    """Per-row judge admission over every variant (recovered + generated).

    Returns ({variant: admitted rows}, judge_record, {variant: admitted
    path}). Pilot-gates any >=5,000-call wave (rule 26) — a FAIL raises
    PilotGateRefusal (exit 7).

    Partial-judge resume ACROSS variants (C9/M6, Unit D flag): each
    variant's admitted JSONL + admission sidecar persist the moment its
    judging completes (checkpoint-per-phase), so a crash on variant 4/6
    re-judges only the remainder; a resumed variant whose sidecar matches
    the regime + prejudge identity loads from disk and skips the judge
    dispatch entirely (the rubric-keyed judge cache stays the API-call-level
    resume within a variant).
    """
    from explore_persona_space.eval.graded_judge import judge_graded

    cache_root = out_dir / "_judge_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    rubric = _scaffold_judge_rubric()

    # Resume pass FIRST — resumed variants leave the pilot-gate arithmetic.
    admitted: dict[str, list[dict]] = {}
    admitted_paths: dict[str, Path] = {}
    resumed_records: dict[str, dict] = {}
    to_judge: dict[str, list[dict]] = {}
    for v, rows in variant_rows.items():
        admitted_path = out_dir / v / f"scaffolds_{v}.jsonl"
        regime = _judge_regime(v, args, rubric)
        pj_path = _prejudge_path(out_dir, v)
        inputs = {"prejudge_sha256": resume.file_sha256(pj_path) if pj_path.is_file() else None}
        ok, reason = resume.soft_resume_ok(admitted_path, regime, inputs)
        if ok:
            admitted[v] = _read_jsonl(admitted_path)
            admitted_paths[v] = admitted_path
            done = resume.read_done(admitted_path) or {}
            resumed_records[v] = dict((done.get("extra") or {}).get("record") or {})
            resumed_records[v]["resumed"] = True
            _log(f"variant={v} judge admission RESUME skip ({reason})")
        else:
            if reason not in ("output missing or empty", "no done sidecar"):
                _log(f"variant={v} judge admission recompute: {reason}")
            to_judge[v] = rows

    per_variant_items: dict[str, tuple[list, list, int]] = {
        v: _variant_judge_items(v, rows) for v, rows in to_judge.items()
    }
    total_calls = sum(len(items) for items, _r, _n in per_variant_items.values()) * max(
        1, args.judge_draws
    )

    pilot_report: dict | None = None
    if total_calls >= 5000:
        _log(f"judge wave {total_calls} calls >= 5000 — running rule-26 pilot gate")
        pilot_report = _run_judge_pilot(to_judge, args, cache_root)
        verdict = str(pilot_report.get("verdict", "")).upper()
        if verdict not in {"PASS", "PASS_WAIVED"}:
            raise PilotGateRefusal(
                f"judge pilot verdict={verdict!r} — production wave refused "
                f"(report: {_rel(cache_root / 'pilot_gate_report.json')})"
            )

    record: dict = {
        "judge_model": None,
        "max_tokens": args.max_tokens,
        # Realized sampling regime (review r2 deferred ruling 2 / Minor 6):
        # judge_graded's batch client does NOT thread temperature — the plan
        # §10 "temperature 0" is a documented deviation (graded_judge.py:15).
        "temperature": "api-default — not threadable through judge_graded/batch client",
        "n_draws": max(1, args.judge_draws),
        "threshold": args.judge_keep_threshold,
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest()[:16],
        "n_calls": total_calls,
        "n_variants_resumed": len(resumed_records),
        "pilot": pilot_report,
        "variants": dict(resumed_records),
    }
    from explore_persona_space.eval.graded_judge import DEFAULT_JUDGE_MODEL

    record["judge_model"] = DEFAULT_JUDGE_MODEL
    raw_dir = out_dir / "judge_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for variant, (items, judged_rows, n_no_question) in per_variant_items.items():
        admitted_path = out_dir / variant / f"scaffolds_{variant}.jsonl"
        if not items:
            admitted[variant] = []
            record["variants"][variant] = {
                "judged": 0,
                "admitted": 0,
                "structural_no_question": n_no_question,
            }
            continue
        result = judge_graded(
            items,
            rubric,
            n_draws=max(1, args.judge_draws),
            cache_dir=cache_root / "prod" / variant,
            save_raw=raw_dir / f"judge_raw_{variant}.json",
            max_tokens=args.max_tokens,
        )
        kept_rows, drops = _admit_variant_rows(
            judged_rows, items, result, args.judge_keep_threshold
        )
        admitted[variant] = kept_rows
        rec = {
            "judged": len(items),
            "admitted": len(kept_rows),
            "structural_no_question": n_no_question,
            "judge_drops": drops,
            "judge_telemetry": {
                "n_total_draws": result.n_total_draws,
                "n_dropped_draws": result.n_dropped_draws,
                "n_transport_lost_draws": result.n_transport_lost_draws,
                "n_refusal_draws": result.n_refusal_draws,
                "n_truncation_dropped_draws": result.n_truncation_dropped_draws,
                "stop_reason_tally": result.stop_reason_tally,
            },
        }
        record["variants"][variant] = rec
        # Checkpoint THIS variant the moment its admission completes (C9/M6):
        # admitted JSONL first, sidecar (regime + prejudge pin + record) last.
        _atomic_write_jsonl(admitted_path, kept_rows)
        admitted_paths[variant] = admitted_path
        pj_path = _prejudge_path(out_dir, variant)
        resume.write_done(
            admitted_path,
            _judge_regime(variant, args, rubric),
            inputs={"prejudge_sha256": resume.file_sha256(pj_path) if pj_path.is_file() else None},
            extra={"record": rec},
        )
        _log(
            f"variant={variant} judge admission: {len(kept_rows)}/{len(items)} kept "
            f"(drops={drops}, no_question={n_no_question}) -> {_rel(admitted_path)}"
        )
    return admitted, record, admitted_paths


# ---------------------------------------------------------------------------
# Local writes + upload
# ---------------------------------------------------------------------------
def _write_scaffolds_local(
    variants_scaffolds: dict[str, list[dict]], out_dir: Path, suffix: str = ""
) -> dict[str, Path]:
    """Write per-variant scaffolds JSONL locally; return {variant: path}."""
    paths: dict[str, Path] = {}
    for variant, rows in variants_scaffolds.items():
        p = out_dir / variant / f"scaffolds_{variant}{suffix}.jsonl"
        _atomic_write_jsonl(p, rows)
        paths[variant] = p
        _log(f"variant={variant} wrote {len(rows)} scaffolds -> {_rel(p)}")
    return paths


def _shard_large_jsonl_for_upload(files: list[Path]) -> list[Path]:
    """Replace any >9.5 MB .jsonl with <9 MB line-shards + a manifest.

    upload-policy.md: text >9.5 MB per file line-splits into <9 MB shards
    (`<stem>.shardNN.jsonl` + `<stem>.manifest.json`), NEVER gzip — the Hub
    force-routes any >10 MB blob to LFS. Non-jsonl oversized files pass
    through with a WARN (they ride LFS).
    """
    out: list[Path] = []
    for f in files:
        if not f.is_file():
            continue
        size = f.stat().st_size
        if size <= UPLOAD_SHARD_LIMIT_BYTES:
            out.append(f)
            continue
        if f.suffix != ".jsonl":
            _log(f"WARN oversized non-jsonl upload rides LFS: {_rel(f)} ({size} B)")
            out.append(f)
            continue
        shards: list[Path] = []
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
        _atomic_write_json(
            manifest,
            {
                "source": f.name,
                "parts": [s.name for s in shards],
                "line_counts": line_counts,
                "sha256": {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in shards},
            },
        )
        _log(f"sharded {_rel(f)} ({size} B) -> {len(shards)} shards")
        out.extend(shards)
        out.append(manifest)
    return out


def _upload_scaffold_files(out_dir: Path, files: list[Path], *, fail_loud: bool) -> None:
    """One bulk upload_folder commit of the named files under the scaffolds
    prefix (plan output ``issue2054_lattice/scaffolds/``). ``fail_loud=True``
    on the pod gen leg — the prejudge upload is the cross-machine seam the VM
    judge stage consumes (#1482 class), so a failed upload must not exit 0.
    """
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    files = _shard_large_jsonl_for_upload(files)
    allow = sorted({f.relative_to(out_dir).as_posix() for f in files if f.is_file()})
    if not allow:
        _log("upload: nothing to upload")
        return
    expected = [f"{TASK_PREFIX}/scaffolds/{rel}" for rel in allow]
    try:
        _upload_folder_filtered(
            out_dir,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{TASK_PREFIX}/scaffolds",
            allow_patterns=allow,
            expected_repo_paths=expected,
        )
        _log(f"uploaded {len(allow)} scaffold file(s) in one bulk commit")
    except Exception as exc:  # noqa: BLE001
        if fail_loud:
            raise
        _log(f"WARN scaffold bulk upload failed: {exc}")


def _upload_fold_map(fold_map_path: Path, *, fail_loud: bool) -> None:
    from explore_persona_space.orchestrate.hub import _upload

    if not fold_map_path.is_file():
        return
    # UPLOAD_LOOP_EXEMPT: single fold-map file, not a loop — direct _upload
    try:
        _upload(
            fold_map_path,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{TASK_PREFIX}/shared_fold_map.json",
            upload_as_file=True,
        )
        _log("uploaded shared_fold_map.json")
    except Exception as exc:  # noqa: BLE001
        if fail_loud:
            raise
        _log(f"WARN fold-map upload failed: {exc}")


def _prejudge_path(out_dir: Path, variant: str) -> Path:
    return out_dir / variant / f"scaffolds_{variant}_prejudge.jsonl"


def _gen_regime(variant: str, args) -> dict:
    """Output-affecting regime for one variant's gen-stage prejudge pool
    (C9/M6 — the #722-r3 every-regime-key rule)."""
    return {
        "stage": "gen",
        "variant": variant,
        "seed": int(args.seed),
        "target_conv_ids": int(args.target_conv_ids),
        # Output-affecting: it sets the shared draw size directly (#722-r3 rule).
        "gen_draw_n": None if args.gen_draw_n is None else int(args.gen_draw_n),
        "gen_model": str(args.gen_model),
        "gen_mock": bool(args.gen_mock),
        "no_generate": bool(args.no_generate),
    }


def _write_prejudge_sidecars(
    prejudge_paths: dict[str, Path], args, n_rows: dict[str, int]
) -> list[Path]:
    """Done sidecars for the gen-stage prejudge pools: the gen-side resume
    predicate AND the judge-side staleness anchor (the recorded content sha —
    Unit D flag: `--stage judge` must not silently consume a stale local
    prejudge from an older gen run)."""
    sidecars: list[Path] = []
    for v, p in prejudge_paths.items():
        sidecars.append(
            resume.write_done(
                p,
                _gen_regime(v, args),
                extra={"prejudge_sha256": resume.file_sha256(p), "n_rows": n_rows.get(v, 0)},
            )
        )
    return sidecars


def _prejudge_resume_ok(prejudge_path: Path, regime: dict) -> tuple[bool, str]:
    """(skip?, reason) for one variant's prejudge pool: sidecar present +
    regime match + the recorded content sha matches the file on disk."""
    if not prejudge_path.is_file() or prejudge_path.stat().st_size == 0:
        return False, "prejudge missing or empty"
    payload = resume.read_done(prejudge_path)
    if payload is None:
        return False, "no prejudge sidecar"
    diff = resume.regime_diff(payload.get("regime") or {}, regime)
    if diff:
        return False, f"regime changed: {diff}"
    recorded_sha = (payload.get("extra") or {}).get("prejudge_sha256")
    if recorded_sha != resume.file_sha256(prejudge_path):
        return False, "prejudge content drifted from its sidecar"
    return True, "prejudge complete under matching regime"


def _verify_prejudge_staleness(out_dir: Path, variants: list[str], args) -> None:
    """--stage judge staleness gate (Unit D flag): every consumed prejudge
    must carry a sidecar whose recorded sha matches the file AND whose seed /
    target_conv_ids match this judge invocation — a stale local pool from an
    older gen run fails LOUD, never silently judged."""
    for v in variants:
        p = _prejudge_path(out_dir, v)
        payload = resume.read_done(p)
        if payload is None:
            raise RuntimeError(
                f"prejudge for {v} has no done sidecar ({resume.sidecar_path(p)}) — "
                "re-run --stage gen (the sidecar is the staleness anchor) or "
                "re-stage with --prejudge-from-hf"
            )
        recorded_sha = (payload.get("extra") or {}).get("prejudge_sha256")
        actual_sha = resume.file_sha256(p)
        if recorded_sha != actual_sha:
            raise RuntimeError(
                f"prejudge for {v} is STALE: content sha {actual_sha[:12]}… != "
                f"sidecar-recorded {str(recorded_sha)[:12]}… — re-run --stage gen "
                "or re-stage with --prejudge-from-hf"
            )
        rec_regime = payload.get("regime") or {}
        for key, want in (
            ("seed", int(args.seed)),
            ("target_conv_ids", int(args.target_conv_ids)),
            ("gen_draw_n", None if args.gen_draw_n is None else int(args.gen_draw_n)),
        ):
            if rec_regime.get(key) != want:
                raise RuntimeError(
                    f"prejudge for {v} was generated under {key}="
                    f"{rec_regime.get(key)!r}, judge invocation wants {want!r} — "
                    "cross-regime judging refused; re-run --stage gen"
                )
        if rec_regime.get("gen_mock"):
            _log(f"WARN variant={v}: prejudge pool is a --gen-mock (smoke) pool")


def _stage_prejudge_from_hf(out_dir: Path, variant: str) -> None:
    """Authoritatively (re)stage ONE variant's prejudge pool + staleness
    sidecar from HF, reassembling the SHARDED upload form.

    Why this is not a one-line `stage_hub_file` of `<stem>.jsonl` (r6 defect):
    `_shard_large_jsonl_for_upload` line-splits any >9.5 MB `.jsonl` into
    `<stem>.shardNN.jsonl` + `<stem>.manifest.json` before upload
    (upload-policy.md — the Hub force-routes any >10 MB blob to LFS). Every
    prejudge pool crossed that threshold for the first time at the r5 draw
    (~6,900 rows ~ 12 MB, vs ~4,000 rows ~ 7 MB before), so on HF the plain
    `<stem>.jsonl` name now resolves ONLY to the previous round's smaller,
    UNSHARDED residue — 2,043 rows for char_helios against the 6,931 this
    round produced. Staging that name silently hands the judge leg the failed
    round's pool.

    The `.done.json` staleness sidecar caught it (recorded sha vs file sha) and
    the run halted rather than judging stale data — but a refusal is not a
    repair, hence this helper.

    Shards are verified against the manifest's per-part sha256 and reassembled
    by exact in-order concatenation, so the rebuilt file is byte-identical to
    what the gen leg hashed into the sidecar.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file

    api = HfApi()
    stem = f"scaffolds_{variant}_prejudge"
    base = f"{TASK_PREFIX}/scaffolds/{variant}"
    dest_dir = out_dir / variant
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = _prejudge_path(out_dir, variant)

    manifest_remote = f"{base}/{stem}.manifest.json"
    # Single-path LITERAL existence probe: a scoped listing would cost a
    # full-prefix page walk to answer one yes/no, and gotchas.md names
    # file_exists as the right call for exactly this shape.
    has_manifest = retry_transient(
        # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
        lambda: api.file_exists(HF_DATA_REPO, manifest_remote, repo_type="dataset"),
        what=f"file_exists({manifest_remote})",
    )

    if has_manifest:
        mpath = dest_dir / f"{stem}.manifest.json"
        stage_hub_file(HF_DATA_REPO, manifest_remote, mpath, repo_type="dataset", overwrite=True)
        man = json.loads(mpath.read_text(encoding="utf-8"))
        parts = list(man.get("parts") or [])
        if not parts:
            raise RuntimeError(f"prejudge manifest for {variant} lists no parts: {manifest_remote}")
        want_sha = man.get("sha256") or {}
        local_parts: list[Path] = []
        for name in parts:
            lp = dest_dir / name
            stage_hub_file(HF_DATA_REPO, f"{base}/{name}", lp, repo_type="dataset", overwrite=True)
            got = hashlib.sha256(lp.read_bytes()).hexdigest()
            exp = want_sha.get(name)
            if exp and exp != got:
                raise RuntimeError(
                    f"prejudge shard {name} for {variant} sha mismatch: "
                    f"{got[:12]}… != manifest {str(exp)[:12]}…"
                )
            local_parts.append(lp)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("wb") as out:
            for lp in local_parts:
                out.write(lp.read_bytes())
        os.replace(tmp, target)
        _log(
            f"variant={variant} prejudge reassembled from {len(local_parts)} shard(s) "
            f"({target.stat().st_size} B)"
        )
    else:
        stage_hub_file(
            HF_DATA_REPO, f"{base}/{stem}.jsonl", target, repo_type="dataset", overwrite=True
        )
        _log(f"variant={variant} prejudge staged unsharded ({target.stat().st_size} B)")

    # The staleness-anchor sidecar rides the same prefix, sharded or not, and
    # is never itself large enough to shard.
    stage_hub_file(
        HF_DATA_REPO,
        f"{base}/{stem}.jsonl.done.json",
        dest_dir / f"{stem}.jsonl.done.json",
        repo_type="dataset",
        overwrite=True,
    )


def _load_prejudge(
    out_dir: Path, variants: list[str], *, from_hf: bool, args
) -> dict[str, list[dict]]:
    """Load per-variant prejudge pools for --stage judge (staging from HF on
    request — the gen pod uploaded them; fail-loud when absent or STALE)."""
    # NOTE: `_stage_prejudge_from_hf` handles the SHARDED upload form; see its
    # docstring for why the plain `<stem>.jsonl` name is not sufficient.
    if from_hf:
        # AUTHORITATIVE re-stage, not stage-if-missing (r6). Two reasons:
        #   (a) `stage_hub_file` is idempotent — an existing local target
        #       returns with NO network call — so a stale local pool from an
        #       earlier round silently wins over the Hub copy;
        #   (b) `--prejudge-from-hf` means "the Hub is the source of truth for
        #       this leg", and the staleness gate below can only REFUSE a
        #       stale local file, never repair it.
        # overwrite=True in `_stage_prejudge_from_hf` makes the flag do what it
        # says. The pools are ~12 MB of text each; re-staging is cheap next to
        # judging 33k rows against the wrong pool.
        for v in variants:
            _stage_prejudge_from_hf(out_dir, v)
    missing = [v for v in variants if not _prejudge_path(out_dir, v).is_file()]
    if missing:
        raise FileNotFoundError(
            f"prejudge inputs missing for {missing}: run --stage gen first "
            f"(uploads to {TASK_PREFIX}/scaffolds/) or pass --prejudge-from-hf"
        )
    _verify_prejudge_staleness(out_dir, variants, args)
    return {v: _read_jsonl(_prejudge_path(out_dir, v)) for v in variants}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_phase(args: argparse.Namespace) -> int:
    variants = list(args.variants)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"start: stage={args.stage} target_conv_ids={args.target_conv_ids} variants={variants}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # noqa: BLE001
        print(f"ERROR: huggingface_hub missing: {exc}", file=sys.stderr)
        return 2
    api = HfApi()

    gen_counts: dict[str, dict] = {}
    question_record: dict | None = None
    shared_recovered: set[str] = set()
    gen_stage_resumed = False
    if args.stage in ("all", "gen"):
        # Phase-level skip-completed (C9/M6, Unit D flag: recovery re-downloads
        # parent stories every run). ALL-OR-NOTHING by design: the shared
        # question draw spans the shared recovered intersection of EVERY
        # variant, so a partial per-variant skip would break the shared-draw
        # invariant — when every variant's prejudge pool + regime-matching
        # sidecar are present, the whole recover+generate leg is skipped
        # (the generator's own fingerprint-gated per-chunk resume + the
        # question-pool fingerprint cache already absorb mid-gen crashes).
        prejudge_status = {
            v: _prejudge_resume_ok(_prejudge_path(out_dir, v), _gen_regime(v, args))
            for v in variants
        }
        if all(ok for ok, _ in prejudge_status.values()) and variants:
            gen_stage_resumed = True
            variant_rows = {v: _read_jsonl(_prejudge_path(out_dir, v)) for v in variants}
            _log(
                "gen stage RESUME skip: every variant's prejudge pool is complete "
                f"under the matching regime — { {v: len(r) for v, r in variant_rows.items()} }"
            )
        else:
            for v, (ok, reason) in prejudge_status.items():
                if not ok:
                    _log(f"gen stage runs (variant={v}: {reason})")
            _log("recover: strip parent stories -> scaffolds")
            variant_rows = _recover_scaffolds_from_hf(variants, api)
            n_total_recovered = sum(len(rs) for rs in variant_rows.values())
            _log(f"recovered {n_total_recovered} scaffolds across {len(variant_rows)} variants")

            shared_recovered = _shared_recovered_intersection(variant_rows)
            # DEFAULT sizing: top the shared recovered pool up to
            # --target-conv-ids. That arithmetic sizes the CROSS-variant
            # intersection, but kill gate 4 intersects WITHIN one (character,
            # model) comparison group (`issue2054_fits._comparison_group_key`),
            # so the binding quantity is each variant's OWN admitted pool. When
            # per-variant recovery far exceeds the 5-way intersection (measured
            # 2026-08-05: ~2,155/variant recovered vs 1,055 shared), the default
            # UNDER-draws for the gate. `--gen-draw-n` sets the shared draw
            # directly; the operator grounds it on the measured verbatim-keep
            # rate (see QUESTION_MAX_CHARS) and records the arithmetic at
            # dispatch. The draw stays SHARED across variants either way, so
            # plan req 1 is untouched.
            if args.gen_draw_n is not None:
                n_gen = max(0, int(args.gen_draw_n))
                _log(
                    f"shared recovered intersection={len(shared_recovered)}; "
                    f"shared generation draw n={n_gen} (--gen-draw-n override; "
                    f"target_conv_ids arithmetic would have given "
                    f"{max(0, args.target_conv_ids - len(shared_recovered))})"
                )
            else:
                n_gen = max(0, args.target_conv_ids - len(shared_recovered))
                _log(
                    f"shared recovered intersection={len(shared_recovered)} "
                    f"-> shared generation draw n={n_gen}"
                )
            if args.pilot or args.no_generate or n_gen == 0:
                _log(
                    f"generation SKIPPED (pilot={args.pilot} no_generate={args.no_generate} "
                    f"n_gen={n_gen})"
                )
            else:
                questions, question_record = _ensure_question_pool(args, n_gen, out_dir)
                for v in variants:
                    gen_rows, counts = _generate_shortfall(
                        v,
                        questions,
                        out_dir,
                        seed=args.seed,
                        mock=args.gen_mock,
                        gen_model=args.gen_model,
                    )
                    variant_rows[v] = variant_rows[v] + gen_rows
                    gen_counts[v] = counts
        # One conversation per row within each variant (plan assumption 29) —
        # asserted on the resumed pools too.
        for v, rows in variant_rows.items():
            cids = [str(r.get("conv_id")) for r in rows]
            n_dupes = len(cids) - len(set(cids))
            assert n_dupes == 0, f"variant {v}: {n_dupes} duplicate conv_ids in pool"

        if gen_stage_resumed:
            prejudge_paths = {v: _prejudge_path(out_dir, v) for v in variants}
            # C-R2-1: the sidecars on disk (validated by _prejudge_resume_ok)
            # are the judge leg's staleness anchors — they ride the re-upload
            # below exactly as on a fresh gen leg.
            prejudge_sidecars = [resume.sidecar_path(p) for p in prejudge_paths.values()]
        else:
            prejudge_paths = _write_scaffolds_local(variant_rows, out_dir, suffix="_prejudge")
            # Gen-side resume predicate + judge-side staleness anchor (C9/M6).
            prejudge_sidecars = _write_prejudge_sidecars(
                prejudge_paths, args, {v: len(r) for v, r in variant_rows.items()}
            )

        if args.stage == "gen":
            if str(out_dir).startswith("/tmp/") or args.skip_upload:
                _log("gen leg upload skipped (smoke /tmp tree or --skip-upload)")
            else:
                # Pod leg ends here: persist the seam the VM judge stage consumes
                # (fail-loud — #1482 off-pod read class), plus the gen raws +
                # question draw (raw completions upload ALWAYS). The prejudge
                # sidecars ride along (the judge leg's staleness anchor).
                # C-R2-1 invariant: this upload runs on the RESUMED branch too.
                # The done-sidecars are written BEFORE the fail-loud upload, so
                # a crash AT the upload leaves valid sidecars on disk and the
                # standard crash-recovery re-run RESUMES — a resumed-branch
                # skip would then print [phase=done] with the prejudge pools
                # never on HF (the #521 data-loss class M2 closed). The bulk
                # commit is idempotent on unchanged content, so the re-upload
                # costs one no-op-diff commit on a clean resume.
                upload_files = list(prejudge_paths.values()) + prejudge_sidecars
                for v in variants:
                    upload_files.extend(sorted((out_dir / v / "gen").glob("*.jsonl")))
                    upload_files.extend(sorted((out_dir / v / "gen").glob("*.json")))
                for name in (
                    "shared_question_draw.jsonl",
                    "shared_question_draw.meta.json",
                ):
                    if (out_dir / name).is_file():
                        upload_files.append(out_dir / name)
                _upload_scaffold_files(out_dir, upload_files, fail_loud=True)
            digest = {
                "phase": "phase_a",
                "stage": "gen",
                "resumed": gen_stage_resumed,
                "target_conv_ids": args.target_conv_ids,
                "recovered_per_variant": {
                    v: sum(1 for r in rows if r.get("provenance") == "recovered")
                    for v, rows in variant_rows.items()
                },
                "generated_per_variant": gen_counts,
                "shared_recovered_intersection": len(shared_recovered),
                "question_draw": question_record,
                "prejudge_paths": {v: _rel(p) for v, p in prejudge_paths.items()},
                "metadata": _metadata(args.seed, args.target_conv_ids),
            }
            _atomic_write_json(out_dir / "phase_a_digest.json", digest)
            print(
                f"[phase=phase_a] gen digest: prejudge="
                f"{ {v: len(r) for v, r in variant_rows.items()} }",
                flush=True,
            )
            print("[phase=done]", flush=True)  # noqa: phase-done-reserved
            sys.stdout.flush()
            sys.exit(0)
    else:  # stage == "judge"
        variant_rows = _load_prejudge(out_dir, variants, from_hf=args.prejudge_from_hf, args=args)
        _log(f"loaded prejudge pools: { {v: len(r) for v, r in variant_rows.items()} }")

    cache_root = out_dir / "_judge_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        rep = _run_judge_pilot(variant_rows, args, cache_root)
        verdict = str(rep.get("verdict", "")).upper()
        digest = {
            "phase": "phase_a",
            "stage": f"{args.stage}:pilot",
            "pilot": rep,
            "metadata": _metadata(args.seed, args.target_conv_ids),
        }
        _atomic_write_json(out_dir / "phase_a_pilot_digest.json", digest)
        print(f"[phase=phase_a] pilot verdict={verdict}", flush=True)
        print("[phase=done]", flush=True)  # noqa: phase-done-reserved
        sys.stdout.flush()
        sys.exit(0 if verdict in {"PASS", "PASS_WAIVED"} else 7)

    # Full per-row judge admission (C8(ii)) — per-variant checkpoint + resume
    # inside (C9/M6); admitted JSONLs are written in-loop, not re-written here.
    admitted, judge_record, admitted_paths = _judge_admission(variant_rows, args, out_dir)

    # Unit A carry-forward: every admitted row must carry the q-fields the
    # chat/bare renderers consume (question, or the stripper's q_start/q_end).
    for v, rows in admitted.items():
        for r in rows:
            assert _question_of(r), (
                f"admitted row without q-fields: variant={v} scaffold_id={r.get('scaffold_id')!r}"
            )

    floor = math.ceil(YIELD_FLOOR_FRACTION * args.target_conv_ids)
    kept_variants: dict[str, dict] = {}
    for v in variants:
        rows = admitted.get(v, [])
        rec = dict(judge_record["variants"].get(v, {}))
        rec.update(
            {
                "recovered": sum(1 for r in rows if r.get("provenance") == "recovered"),
                "generated": sum(1 for r in rows if r.get("provenance") == "generated"),
                "prejudge_total": len(variant_rows.get(v, [])),
                "generation": gen_counts.get(v),
                "floor": {
                    "target": args.target_conv_ids,
                    "floor": floor,
                    "below_floor": len(rows) < floor,
                },
                "admitted_conv_ids": sorted(str(r.get("conv_id")) for r in rows),
            }
        )
        if rec["floor"]["below_floor"]:
            # Reported, never backfilled (on-policy-completions.md); the
            # registered drop mechanism is fits-side kill gate 4 (>= 4,480
            # intersection), which reads these counts downstream.
            _log(
                f"WARN variant={v} admitted {len(rows)} < 80% floor {floor} "
                "(reported; kill gate 4 owns the drop decision)"
            )
        kept_variants[v] = rec

    kept_payload = {
        "artifact": "phase_a_admission",
        "target_conv_ids": args.target_conv_ids,
        "judge": {k: v for k, v in judge_record.items() if k != "variants"},
        "question_draw": question_record,
        "shared_recovered_intersection": len(shared_recovered),
        "variants": kept_variants,
        "metadata": _metadata(args.seed, args.target_conv_ids),
    }
    kept_path = out_dir / "kept.json"
    _atomic_write_json(kept_path, kept_payload)
    _log(f"wrote admission record -> {_rel(kept_path)}")

    # Build the shared fold map ONCE, from the ADMITTED conv_ids (the shared
    # draw that survives Phase A is what every downstream phase consumes).
    all_conv_ids: list[str] = []
    for rows in admitted.values():
        all_conv_ids.extend(str(r.get("conv_id")) for r in rows if r.get("conv_id") is not None)
    fold_map = _conv_grouped_folds(all_conv_ids, k=5, seed=args.seed)

    fold_root = Path("eval_results/issue_2054").resolve()
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_map_path = fold_root / "shared_fold_map.json"
    payload = {
        "artifact": "shared_fold_map",
        "k": 5,
        "seed": args.seed,
        "n_conv_ids": len(fold_map),
        "fold_of": fold_map,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "variants": variants,
    }
    _atomic_write_json(fold_map_path, payload)
    _log(f"wrote shared_fold_map.json (n_conv_ids={len(fold_map)}) -> {_rel(fold_map_path)}")

    # FATAL on failure (M2): kept.json + the admitted scaffolds + the shared
    # fold map are the plan-declared downstream inputs of EVERY later phase —
    # `[phase=done]` must never report done with them un-persisted. The
    # admission sidecars ride along (the resume anchors). Smoke trees under
    # /tmp/ (and --skip-upload runs) never mirror to the shared data repo.
    is_smoke = str(out_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload:
        upload_files = [kept_path, *admitted_paths.values()]
        upload_files.extend(
            resume.sidecar_path(p)
            for p in admitted_paths.values()
            if resume.sidecar_path(p).is_file()
        )
        upload_files.extend(sorted((out_dir / "judge_raw").glob("*.json")))
        _upload_scaffold_files(out_dir, upload_files, fail_loud=True)
        _upload_fold_map(fold_map_path, fail_loud=True)

    digest = {
        "phase": "phase_a",
        "stage": args.stage,
        "target_conv_ids": args.target_conv_ids,
        "admitted_per_variant": {v: len(r) for v, r in admitted.items()},
        "prejudge_per_variant": {v: len(r) for v, r in variant_rows.items()},
        "generated_per_variant": gen_counts,
        "judge": {k: v for k, v in judge_record.items() if k != "variants"},
        "shared_fold_map_path": str(_rel(fold_map_path)),
        "kept_path": str(_rel(kept_path)),
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "metadata": _metadata(args.seed, args.target_conv_ids),
    }
    _atomic_write_json(out_dir / "phase_a_digest.json", digest)

    print(
        f"[phase=phase_a] digest: admitted={ {v: len(r) for v, r in admitted.items()} } "
        f"n_conv_ids={len(fold_map)} judge_calls={judge_record.get('n_calls', 0)}",
        flush=True,
    )
    print("[phase=done]", flush=True)  # noqa: phase-done-reserved
    sys.stdout.flush()
    sys.exit(0)  # explicit exit before finalize-time C-extension teardown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-conv-ids", type=int, default=8_000)
    p.add_argument(
        "--gen-draw-n",
        type=int,
        default=None,
        help=(
            "size the SHARED generation draw directly, bypassing the "
            "target-conv-ids minus shared-recovered arithmetic (which sizes the "
            "cross-variant intersection, not the per-(character,model) pool kill "
            "gate 4 actually reads). Ground it on the measured verbatim-keep rate."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="data/issue_2054/scaffolds",
        help="root for per-variant scaffold JSONLs",
    )
    p.add_argument("--seed", type=int, default=137)
    p.add_argument(
        "--stage",
        choices=("all", "gen", "judge"),
        default="all",
        help=(
            "gen = recovery + shortfall generation + prejudge upload (pod, "
            "lora-7b intent); judge = per-row judge admission + kept.json + "
            "fold map (VM, Batch API); all = both in-process (smoke / single box)"
        ),
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "run ONLY the rule-26 judge pilot (200-draw gate) over the stage's "
            "scaffold pool; skips generation + uploads; exit 0 on PASS, 7 on FAIL"
        ),
    )
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_VARIANTS),
        help="comma-separated parent variant slugs (default: v4 lattice character panel + assistant)",
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--pilot-n", type=int, default=200)
    p.add_argument(
        "--judge-draws",
        type=int,
        default=1,
        help="judge draws per row (default 1 — the plan §9 ~35k-call arithmetic)",
    )
    p.add_argument(
        "--judge-keep-threshold",
        type=float,
        default=50.0,
        help="admit rows with mean judge score >= this (persona-vectors convention)",
    )
    p.add_argument(
        "--questions-jsonl",
        default=None,
        help=(
            "explicit shared question draw (rows: question + qid/conv_id); "
            "default draws seed-pinned from the #1738 manifest"
        ),
    )
    p.add_argument(
        "--manifest-revision",
        default=None,
        help="pin the #1738 manifest revision for the question draw",
    )
    p.add_argument(
        "--no-generate",
        action="store_true",
        help="skip the shortfall generation leg (recovered-only pool)",
    )
    p.add_argument(
        "--gen-mock",
        action="store_true",
        help="thread the generator's deterministic --mock path (CPU smoke)",
    )
    p.add_argument(
        "--gen-model",
        choices=("instruct", "pretrained"),
        default="instruct",
        help="generator model for the shortfall leg (parent #1345 default: instruct)",
    )
    p.add_argument(
        "--prejudge-from-hf",
        action="store_true",
        help="--stage judge: stage missing prejudge inputs from the HF scaffolds prefix",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="skip the HF mirror steps (smoke use; sibling drivers' convention)",
    )
    args = p.parse_args()
    if args.stage == "gen" and args.pilot:
        p.error("--pilot is a judge-stage probe; use --stage judge or all")
    if args.gen_mock and args.no_generate:
        p.error("--gen-mock and --no-generate are mutually exclusive")
    try:
        return run_phase(args)
    except PilotGateRefusal as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        sys.exit(7)


if __name__ == "__main__":
    sys.exit(main())
