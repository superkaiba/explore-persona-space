#!/usr/bin/env python
"""Issue #1934 — recover #1773's parse-dropped full-dictionary labels.

The #1773 full-dictionary describe phase dropped 2,602/128,712 items to a
parser gap (`parse_judge_json` anchored on the FIRST ``{``; a brace-bearing
reasoning preamble mis-anchored the decode — fixed in this task's
`eval/utils.py` recovery ladder), and the axes phase then silently labelled
those features on EVIDENCE-ONLY prompts (no DESC block). This driver
re-judges the missing describe set against the FIXED parser and re-judges the
5 axes for every newly-described real feature, writing all outputs to a
PARALLEL tree (`eval_results/issue_1773/recovery_1934/` + HF
`issue1773_featurepipeline/recovery_1934/`) — the committed #1773 originals
and HF `fulldict/` paths are NEVER mutated.

Phases (each checkpointed + idempotent-resume, skip-completed at entry keyed
on output artifacts; per-phase `[<phase>]` progress lines):

  p0_stage    scoped staging of the fulldict evidence manifests + labels from
              HF via `hub.stage_hub_prefix` (dest is a MIRROR ROOT — files
              land at dest/<repo-relative path>; asserted). df headroom probe
              (>=1.5x projected ~1.4 GB) before staging.
  p0b_lineage HALT gate (rc=21 + report JSON, route-on-artifact): worktree
              pipeline copies byte-identical to origin/issue-1773 tip AND the
              run-sha..tip diff is free of axis-renderer changes (plan A7).
  p1_missing  packets(include_controls=True) - described - no-evidence ->
              missing set; realized-keys assert on the staged description
              rows before any spend.
  p2_parity   rebuild the describe user_msg for 50 random ALREADY-described
              features; sha16 must equal the recorded prompt_sha16 (evidence
              drift => HALT rc=22, no spend; #922 pair-provenance guard).
  p3_describe re-judge the missing set (DESCRIBER_SYSTEM, max_tokens per
              CM.DESCRIBE_MAX_TOKENS — the banked recovery ran at 700; 1024 since #2063 —
              temp 1.0, 1 draw, keep_raw_judge_text, FRESH checkpoint dir,
              Batch path pinned). Kill gate: n_parse_fail_nonempty /
              n_nonempty >= 0.20 -> HALT rc=23 before the axes leg (EMPTY
              responses are content drops, excluded from BOTH gate counts,
              tallied separately). Response-shape census (fenced /
              brace_preamble / empty / other) over the fresh raw texts.
  p4_axes     5 axes x 5 draws over ONLY newly-described real features
              (feat_id >= 0), DESC present, same deterministic label
              permutation; FRESH checkpoint dir; majority-vote aggregation.
              Prints the ops/cost arithmetic (realized #1773 basis 997.4 in /
              163.8 out tokens per call) BEFORE submitting. NOT dispatched
              under --smoke (item build + arithmetic only).
  p5_outputs  parallel output tree {descriptions_recovered.jsonl,
              axis_labels_recovered.jsonl, recovery_meta.json,
              kappa_recovered.json} + HF upload (bulk upload_folder, <=9 MB
              jsonl shards) of raw judge text + merged descriptions + merged
              axis labels carrying a `desc_present` provenance column.

The Batch path is ALWAYS pinned (plan §9 "force-batch pinned") — it is
load-bearing: raw text for STILL-parse-failed responses is recovered by a
read-only re-stream of the already-paid Batch results (error dicts never
carry `_raw_text`), which only exists on the Batch path.

Usage:
  uv run python scripts/issue1934_recover_1773_labels.py --smoke --limit 5 --force-batch
  uv run python scripts/issue1934_recover_1773_labels.py --full
  (resume: re-run the same command — all phases checkpoint + skip-completed)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1773_common as CM  # noqa: E402
import issue1773_describe_axes as DA  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────

RC_LINEAGE_HALT = 21
RC_PARITY_HALT = 22
RC_P3_GATE_HALT = 23

P3_GATE_FLOOR = 0.20  # plan §4.4 (critic MF2): parse-fail fraction of NON-EMPTY responses
PARITY_N = 50
PARITY_SEED = 1934

# Realized #1773 axes token basis (per-call, amendment 2026-07-31T16:35Z) +
# Anthropic Batch-API Sonnet pricing (50% of $3/M in, $15/M out).
AXES_IN_TOKENS_PER_CALL = 997.4
AXES_OUT_TOKENS_PER_CALL = 163.8
BATCH_USD_PER_MTOK_IN = 3.0 * 0.5
BATCH_USD_PER_MTOK_OUT = 15.0 * 0.5

STAGE_PREFIXES = (
    f"{CM.HF_PREFIX}/fulldict/evidence/evidence_manifests",
    f"{CM.HF_PREFIX}/fulldict/labels",
)
PROJECTED_STAGE_BYTES = 1.4e9  # plan §4.4 (~1.1-1.3 GB evidence + ~0.1 GB labels)
STAGE_HEADROOM_FACTOR = 1.5
HF_RECOVERY_PREFIX = "recovery_1934"  # under CM.HF_PREFIX via DA._upload_dir

PIPELINE_FILES = (
    "scripts/issue1773_describe_axes.py",
    "scripts/issue1773_common.py",
)
# Axis-rendering surface (plan p0b(ii)): a run_sha..tip diff touching any of
# these means the recovery would run a THIRD instrument -> HALT.
RENDERER_SYMBOLS = (
    "build_axes_items",
    "build_axis_user_msg",
    "label_permutation",
    "AXIS_DEFINITIONS",
    "AXIS_SEES",
    "AXIS_SYSTEM_PREAMBLE",
    "render_windows_block",
    "render_out_block",
    "axis_custom_id",
)

REQUIRED_DESC_KEYS = frozenset({"feat_id", "description", "confidence", "prompt_sha16"})
_EVIDENCE_PATTERNS = ("evidence.shard*.jsonl", "evidence_randdir.shard*.jsonl")

PHASES = (
    "p0_stage",
    "p0b_lineage",
    "p1_missing",
    "p2_parity",
    "p3_describe",
    "p4_axes",
    "p5_outputs",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}"
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    """Atomic single-file JSONL write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}"
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ── config ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class RecoveryCfg:
    """Paths + mode for one recovery run (smoke and full share the code path)."""

    smoke: bool
    limit: int
    no_upload: bool
    stage_root: Path  # HF mirror root (inputs; shared across smoke/full)
    work: Path  # checkpoints, raw shards, reports
    out_root: Path  # final parallel output tree

    @property
    def labels_dir(self) -> Path:
        return self.stage_root / CM.HF_PREFIX / "fulldict" / "labels"

    @property
    def evidence_dir(self) -> Path:
        return self.stage_root / CM.HF_PREFIX / "fulldict" / "evidence"

    @property
    def reports(self) -> Path:
        return self.work / "reports"


def build_cfg(args: argparse.Namespace) -> RecoveryCfg:
    """Resolve smoke/full default roots (smoke: FRESH throwaway work under /tmp;
    staged inputs are shared — they are a read-only HF mirror)."""
    import os

    user = os.environ.get("USER", "thomasjiralerspong")
    base = Path(f"/mnt/eps-data/{user}/issue1934_recovery")
    if args.smoke:
        smoke_root = Path(f"/tmp/issue1934_smoke_{time.strftime('%Y%m%d-%H%M%S')}")
        stage = args.stage_root or (base / "staged")
        work = args.work or (smoke_root / "work")
        out = args.out_root or (smoke_root / "out")
        no_upload = True  # smoke NEVER uploads (outputs never touch the production prefix)
    else:
        stage = args.stage_root or (base / "staged")
        work = args.work or (base / "work")
        out = args.out_root or (PROJECT_ROOT / "eval_results" / "issue_1773" / "recovery_1934")
        no_upload = args.no_upload
    return RecoveryCfg(
        smoke=args.smoke,
        limit=args.limit,
        no_upload=no_upload,
        stage_root=Path(stage),
        work=Path(work),
        out_root=Path(out),
    )


# ── pure helpers (unit-tested in tests/test_issue1934_recovery.py) ──────────


def check_description_row_keys(row: dict) -> bool:
    """Realized-keys check for a staged description row (plan p1)."""
    return REQUIRED_DESC_KEYS <= set(row)


def compute_missing_set(
    all_packet_ids: set[int], described_ids: set[int], no_evidence_ids: set[int]
) -> list[int]:
    """packets(include_controls=True) - described - no-evidence, sorted.

    Controls (feat_id < 0) are INCLUDED — the describe stage covers them; the
    axes stage later filters to real features (feat_id >= 0) via
    ``build_axes_items``.
    """
    return sorted(set(all_packet_ids) - set(described_ids) - set(no_evidence_ids))


def gate_verdict(n_parse_fail_nonempty: int, n_nonempty: int, floor: float = P3_GATE_FLOOR) -> str:
    """HALT iff parse-fail fraction of NON-EMPTY responses >= floor.

    The boundary (exactly ``floor``) HALTs. A degenerate denominator
    (``n_nonempty <= 0``) HALTs — no valid data is never a PASS.
    """
    if n_nonempty <= 0:
        return "HALT"
    return "HALT" if (n_parse_fail_nonempty / n_nonempty) >= floor else "PASS"


def p3_gate_stats(n_success_parsed: int, failed: list[dict], floor: float = P3_GATE_FLOOR) -> dict:
    """Gate arithmetic (plan §4.4, critic MF2): EMPTY responses are content
    drops excluded from BOTH the numerator and the denominator; both counts
    are persisted in the p3 report.

    ``failed``: one ``{"raw_text": str|None, "stop_reason": str|None}`` per
    parse-failed cid (from the read-only Batch re-stream).

    REFUSAL-CUT extension (measured on the 2026-07-31 live smoke, all-controls
    slice): a response the API's safety layer STOPPED mid-stream
    (``stop_reason == "refusal"``) is truncated text no parser change can fix
    — the SAME content-class rationale the plan pre-registered for EMPTY
    responses (which are themselves refusal-stops at 0 emitted tokens) — so
    refusal-cut rows are tallied separately (``n_refusal_cut``) and excluded
    from BOTH gate counts. ``max_tokens``-stopped truncations stay IN the
    numerator: rampant budget truncation is an instrument defect the gate
    SHOULD halt on (llm-judging rule 23); their count is reported separately
    so a halt is diagnosable.
    """
    n_empty = sum(1 for f in failed if not (f.get("raw_text") or "").strip())
    n_refusal_cut = sum(
        1 for f in failed if (f.get("raw_text") or "").strip() and f.get("stop_reason") == "refusal"
    )
    n_max_tokens_cut = sum(
        1
        for f in failed
        if (f.get("raw_text") or "").strip() and f.get("stop_reason") == "max_tokens"
    )
    n_fail_nonempty = len(failed) - n_empty - n_refusal_cut
    n_nonempty = n_success_parsed + n_fail_nonempty
    return {
        "n_empty": n_empty,
        "n_refusal_cut": n_refusal_cut,
        "n_max_tokens_cut": n_max_tokens_cut,
        "n_parse_fail_nonempty": n_fail_nonempty,
        "n_nonempty": n_nonempty,
        "ratio": (n_fail_nonempty / n_nonempty) if n_nonempty else None,
        "floor": floor,
        "verdict": gate_verdict(n_fail_nonempty, n_nonempty, floor),
    }


def residual_census(
    n_transport: int,
    n_other_content: int,
    n_schema_fail: int,
    failed: list[dict],
    n_fresh_draws: int,
) -> dict:
    """Classify EVERY non-recovered p3 item (plan v3 §6: no unexplained residual).

    The four v3-named classes partition the non-recovered population —
    ``empty`` / ``refusal_stopped`` / ``residual_parse_fail`` / ``transport``
    — with explicit ``schema_fail`` / ``other_content`` companion keys so
    every non-recovered item is accounted. ``failed``: one
    ``{"raw_text", "stop_reason"}`` meta per parse-failed cid from the Batch
    re-stream; a cid the re-stream could NOT resolve (empty/missing meta)
    counts ``residual_parse_fail`` — NEVER silently 'empty' (round-1 review
    Minor 3). ``refusal_stopped_any`` additionally quantifies the TOTAL
    refusal-stopped population (count + fraction of fresh draws, empties
    included) so the realized coverage ceiling is readable from
    recovery_meta.json alone (concern `recovery-yield-below-plan-target`,
    driver-side half).
    """
    empty = refusal = residual = 0
    n_refusal_any = 0
    for f in failed:
        f = f or {}
        resolved = bool(f)
        raw = (f.get("raw_text") or "").strip()
        if f.get("stop_reason") == "refusal":
            n_refusal_any += 1
        if not resolved:
            residual += 1
        elif not raw:
            empty += 1
        elif f.get("stop_reason") == "refusal":
            refusal += 1
        else:
            residual += 1
    return {
        "empty": empty,
        "refusal_stopped": refusal,
        "residual_parse_fail": residual,
        "transport": n_transport,
        "schema_fail": n_schema_fail,
        "other_content": n_other_content,
        "refusal_stopped_any": {
            "count": n_refusal_any,
            "fraction_of_fresh_draws": (n_refusal_any / n_fresh_draws if n_fresh_draws else None),
            "n_fresh_draws": n_fresh_draws,
        },
    }


def p5_skip_decision(report: dict | None, no_upload: bool) -> bool:
    """p5 resume predicate (round-1 review Minor 2): a p5 completed WITHOUT
    the HF upload is INCOMPLETE for an upload-wanting resume — a later
    `--full` run must re-run p5 (pure assembly + upload, no judge spend)
    rather than silently skip the upload leg. Skip only when the prior
    report exists AND (it uploaded, OR this run does not want an upload)."""
    if not report:
        return False
    return bool(report.get("uploaded")) or no_upload


def p3_halt_decision(verdict: str, smoke: bool) -> bool:
    """Production HALTs on a non-PASS p3 gate; --smoke demotes the verdict to
    an INFORMATIONAL log line (the gate-calibration smoke rule, #1345 class:
    a 0.20 floor calibrated at production n≈2.6k is structurally noisy at
    n=5 — one failed draw reads 20-100%). The gate STILL computes, logs, and
    persists identically under smoke; only the rc-halt is production-only."""
    return verdict != "PASS" and not smoke


def classify_response_shape(raw: str | None) -> str:
    """Census classifier over fresh raw judge texts (plan §4.4, critic SF2).

    Precedence: ``empty`` (no non-whitespace) > ``fenced`` (any markdown
    fence) > ``brace_preamble`` (whole-text AND first-brace decodes fail but
    the recovery ladder finds a value) > ``other``.
    """
    if raw is None or not raw.strip():
        return "empty"
    if "```" in raw:
        return "fenced"
    try:
        json.loads(raw)
        return "other"  # clean whole-text JSON
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    try:
        decoder.raw_decode(raw, raw.index("{"))
        return "other"  # first brace decodes — not the preamble shape
    except (ValueError, json.JSONDecodeError):
        pass
    from explore_persona_space.eval.utils import parse_judge_json

    lg = logging.getLogger("explore_persona_space.eval.utils")
    prev = lg.disabled
    lg.disabled = True  # census over many unparseable rows must not WARN-spam
    try:
        recovered = parse_judge_json(raw)
    finally:
        lg.disabled = prev
    return "brace_preamble" if recovered is not None else "other"


def lineage_gate_decision(
    hash_pairs: dict[str, tuple[str, str]],
    renderer_diff: str,
    renderer_symbols: tuple[str, ...] = RENDERER_SYMBOLS,
) -> dict:
    """Pure p0b decision (plan critic MF1).

    ``hash_pairs``: path -> (worktree blob sha, origin/issue-1773 blob sha);
    any mismatch HALTs (the recovery must run the EXACT pipeline code that
    produced the production labels). ``renderer_diff``: the
    ``<run_sha>..origin/issue-1773`` diff over the pipeline files; a non-empty
    diff HALTs ONLY when it touches the axis-rendering surface — a non-empty
    non-renderer diff is recorded and PASSes (plan A7).
    """
    reasons: list[str] = []
    for path, (wt, br) in sorted(hash_pairs.items()):
        if wt != br:
            reasons.append(f"byte-mismatch: {path} worktree={wt} origin/issue-1773={br}")
    renderer_hits = (
        [s for s in renderer_symbols if s in renderer_diff] if renderer_diff.strip() else []
    )
    if renderer_hits:
        reasons.append(
            "renderer-affecting run_sha..origin/issue-1773 diff touches: "
            + ", ".join(renderer_hits)
        )
    return {
        "verdict": "HALT" if reasons else "PASS",
        "reasons": reasons,
        "diff_empty": not renderer_diff.strip(),
        "renderer_symbols_hit": renderer_hits,
    }


def merge_axis_rows(
    original_rows: list[dict], recovered_rows: list[dict], described_ids: set[int]
) -> list[dict]:
    """Merged axis-label rows with `desc_present` + `source` provenance.

    Join rule (recorded in recovery_meta.json): recovered rows REPLACE
    original rows on (feat_id, axis); original rows carry
    ``desc_present = feat_id in <original described set>`` (False == the row
    was produced on an evidence-only prompt — the propagated #1773 gap);
    recovered rows carry ``desc_present = True`` by construction.
    """
    rec_keys = {(int(r["feat_id"]), r["axis"]) for r in recovered_rows}
    out: list[dict] = []
    for r in original_rows:
        key = (int(r["feat_id"]), r["axis"])
        if key in rec_keys:
            continue
        out.append({**r, "desc_present": int(r["feat_id"]) in described_ids, "source": "original"})
    for r in recovered_rows:
        out.append({**r, "desc_present": True, "source": "recovery_1934"})
    return out


def merge_description_rows(original_rows: list[dict], recovered_rows: list[dict]) -> list[dict]:
    """Original + recovered description rows with a `source` column.

    Recovered feat_ids are DISJOINT from the originals by construction (the
    missing set subtracts described ids); asserted here fail-loud.
    """
    orig_ids = {int(r["feat_id"]) for r in original_rows}
    overlap = orig_ids & {int(r["feat_id"]) for r in recovered_rows}
    assert not overlap, f"recovered descriptions overlap originals: {sorted(overlap)[:10]}"
    return [{**r, "source": "original"} for r in original_rows] + [
        {**r, "source": "recovery_1934"} for r in recovered_rows
    ]


# ── streaming loaders (memory-bounded; DA.load_packets loads the whole 1.3 GB) ──


def collect_packet_ids(evidence_dir: Path) -> set[int]:
    """First streaming pass: every packet feat_id (controls included)."""
    ids: set[int] = set()
    man = evidence_dir / "evidence_manifests"
    n_files = 0
    for pat in _EVIDENCE_PATTERNS:
        for p in sorted(man.glob(pat)):
            n_files += 1
            for r in CM.iter_jsonl(p):
                ids.add(int(r["feat_id"]))
    assert ids, f"no evidence packets under {man}"
    _log(f"[packets] scanned {n_files} manifest files -> {len(ids)} feat_ids")
    return ids


def load_packets_subset(evidence_dir: Path, keep_ids: set[int]) -> dict[int, dict]:
    """Second streaming pass: packets for ``keep_ids`` only (peak RSS stays
    O(subset), not O(1.3 GB manifest tree))."""
    packets: dict[int, dict] = {}
    man = evidence_dir / "evidence_manifests"
    for pat in _EVIDENCE_PATTERNS:
        for p in sorted(man.glob(pat)):
            for r in CM.iter_jsonl(p):
                fid = int(r["feat_id"])
                if fid in keep_ids:
                    packets[fid] = r
    missing = keep_ids - set(packets)
    assert not missing, f"packets absent for {len(missing)} requested ids: {sorted(missing)[:10]}"
    return packets


def load_described(labels_dir: Path) -> dict[int, str]:
    """feat_id -> recorded prompt_sha16 from the staged description shards,
    with the realized-keys assert (plan p1) on EVERY row."""
    described: dict[int, str] = {}
    shards = sorted(labels_dir.glob("descriptions.shard*.jsonl"))
    assert shards, f"no staged description shards under {labels_dir}"
    for shard in shards:
        for row in CM.iter_jsonl(shard):
            assert check_description_row_keys(row), (
                f"realized-keys mismatch in {shard.name}: {sorted(row)} "
                f"(need {sorted(REQUIRED_DESC_KEYS)})"
            )
            described[int(row["feat_id"])] = row["prompt_sha16"]
    return described


# ── batch raw-text recovery (read-only re-stream of paid results) ───────────


def _batch_ids_from_checkpoint(ckpt_dir: Path) -> list[str]:
    """Collect submitted batch_ids from the dispatch checkpoint state files."""
    ids: list[str] = []
    for state_path in sorted(ckpt_dir.rglob("state.json")):
        st = json.loads(state_path.read_text())
        for sb in st.get("sub_batches", []):
            if sb.get("batch_id"):
                ids.append(sb["batch_id"])
    return ids


def fetch_failed_response_meta(ckpt_dir: Path, cids: set[str]) -> dict[str, dict]:
    """Recover raw text + stop_reason for parse-failed cids by RE-STREAMING
    the already-paid Batch results (read-only, zero new spend).

    Error dicts never carry ``_raw_text`` (only successfully-parsed results
    do), so the p3 gate's empty/refusal-cut/nonempty split and the shape
    census need this re-stream. Only ``succeeded`` result rows carry text;
    a cid whose row was transport-class stays absent from the returned map.
    Returns ``{cid: {"raw_text": str, "stop_reason": str|None}}``.
    """
    if not cids:
        return {}
    import anthropic

    # API_DISPATCH_ROUTING_EXEMPT: read-only re-stream of already-paid Batch
    # results (plan A2) via client.messages.batches.results(); no new requests
    # are created — api_dispatch routes message creation/judging, not
    # batch-results reads.
    client = anthropic.Anthropic()
    out: dict[str, dict] = {}
    batch_ids = _batch_ids_from_checkpoint(ckpt_dir)
    _log(f"[raw-restream] {len(cids)} cids over {len(batch_ids)} batches")
    for bid in batch_ids:
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            if cid in cids and cid not in out and result.result.type == "succeeded":
                msg = result.result.message
                out[cid] = {
                    "raw_text": next((b.text for b in msg.content if b.type == "text"), ""),
                    "stop_reason": msg.stop_reason,
                }
        if len(out) == len(cids):
            break
    return out


# ── phases ───────────────────────────────────────────────────────────────────


def _stage_key_files_ok(cfg: RecoveryCfg) -> bool:
    """Consumer key files the later phases open (plan p0 skip predicate)."""
    return (
        (cfg.labels_dir / "describe_meta.json").is_file()
        and (cfg.labels_dir / "no_evidence_features.json").is_file()
        and any(cfg.labels_dir.glob("descriptions.shard*.jsonl"))
        and any(cfg.labels_dir.glob("axis_labels.shard*.jsonl"))
        and any((cfg.evidence_dir / "evidence_manifests").glob("evidence.shard*.jsonl"))
    )


def p0_stage(cfg: RecoveryCfg) -> None:
    """Stage the fulldict inputs from HF into the mirror root (idempotent —
    `stage_hub_file` skips existing targets; report lives IN the stage root
    so smoke and full share one staging)."""
    report_path = cfg.stage_root / ".p0_stage_report.json"
    if report_path.is_file() and _stage_key_files_ok(cfg):
        _log(f"[p0_stage] SKIP (staged; report at {report_path})")
        return
    cfg.stage_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cfg.stage_root)
    need = STAGE_HEADROOM_FACTOR * PROJECTED_STAGE_BYTES
    assert usage.free >= need, (
        f"[p0_stage] insufficient headroom on {cfg.stage_root}: free={usage.free / 1e9:.1f} GB "
        f"< need {need / 1e9:.1f} GB (1.5x projected {PROJECTED_STAGE_BYTES / 1e9:.1f} GB)"
    )
    _log(
        f"[p0_stage] df {cfg.stage_root}: free={usage.free / 1e9:.1f} GB "
        f">= need {need / 1e9:.1f} GB"
    )
    from explore_persona_space.orchestrate import hub

    staged_counts: dict[str, int] = {}
    for prefix in STAGE_PREFIXES:
        _log(f"[p0_stage] staging {CM.HF_DATA_REPO}:{prefix} -> {cfg.stage_root}")
        files = hub.stage_hub_prefix(CM.HF_DATA_REPO, prefix, cfg.stage_root)
        for f in files:
            rel = Path(f).relative_to(cfg.stage_root)
            # dest is a MIRROR ROOT: files land at dest/<repo-relative path>
            # (gotchas: stage_hub_prefix mirror-root arithmetic).
            assert str(rel).startswith(prefix), f"[p0_stage] mirror arithmetic violated: {f}"
        staged_counts[prefix] = len(files)
        _log(f"[p0_stage] staged {len(files)} files under {prefix}")
    assert _stage_key_files_ok(cfg), "[p0_stage] key files missing after staging"
    _write_json(report_path, {"staged": staged_counts, **CM.repro_meta()})
    _log(f"[p0_stage] done -> {report_path}")


def p0b_lineage(cfg: RecoveryCfg) -> None:
    """Lineage HALT gate (plan critic MF1): byte-identity of the pipeline
    copies vs origin/issue-1773 tip + axes-renderer parity vs the run sha."""
    report_path = cfg.reports / "p0b_lineage.json"
    if report_path.is_file() and json.loads(report_path.read_text())["verdict"] == "PASS":
        _log("[p0b_lineage] SKIP (PASS report exists)")
        return

    def _git(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    subprocess.run(
        ["git", "fetch", "origin", "issue-1773", "--quiet"], cwd=PROJECT_ROOT, check=True
    )
    hash_pairs: dict[str, tuple[str, str]] = {}
    for path in PIPELINE_FILES:
        wt = _git("hash-object", path)
        br = _git("rev-parse", f"origin/issue-1773:{path}")
        hash_pairs[path] = (wt, br)
    meta = json.loads((cfg.labels_dir / "describe_meta.json").read_text())
    run_sha = meta.get("git_commit", "")
    try:
        renderer_diff = _git("diff", f"{run_sha}..origin/issue-1773", "--", *PIPELINE_FILES)
        diff_err = None
    except subprocess.CalledProcessError as e:
        renderer_diff = ""
        diff_err = f"run_sha {run_sha!r} unresolvable: {e.stderr.strip()[:200]}"
    decision = lineage_gate_decision(hash_pairs, renderer_diff)
    if diff_err:
        decision["verdict"] = "HALT"
        decision["reasons"].append(diff_err)
    report = {**decision, "run_sha": run_sha, "hash_pairs": hash_pairs, **CM.repro_meta()}
    _write_json(report_path, report)
    _log(f"[p0b_lineage] verdict={decision['verdict']} run_sha={run_sha[:12]}")
    if decision["verdict"] != "PASS":
        _log(f"[p0b_lineage] HALT: {decision['reasons']} (report: {report_path})")
        sys.exit(RC_LINEAGE_HALT)


def p1_missing(cfg: RecoveryCfg) -> dict:
    """Compute + persist the missing describe set (no spend)."""
    out_path = cfg.reports / "missing_set.json"
    if out_path.is_file():
        doc = json.loads(out_path.read_text())
        _log(f"[p1_missing] SKIP (n_missing={doc['n_missing']})")
        return doc
    described = load_described(cfg.labels_dir)
    _log(f"[p1_missing] described rows: {len(described)} (realized-keys OK)")
    noev_doc = json.loads((cfg.labels_dir / "no_evidence_features.json").read_text())
    noev = {int(f) for f in noev_doc["feat_ids"]}
    all_ids = collect_packet_ids(cfg.evidence_dir)
    missing = compute_missing_set(all_ids, set(described), noev)
    doc = {
        "n_packets": len(all_ids),
        "n_described": len(described),
        "n_no_evidence": len(noev),
        "n_missing": len(missing),
        "n_missing_real": sum(1 for f in missing if f >= 0),
        "n_missing_controls": sum(1 for f in missing if f < 0),
        "missing_feat_ids": missing,
        **CM.repro_meta(),
    }
    _write_json(out_path, doc)
    _log(
        f"[p1_missing] packets={len(all_ids)} described={len(described)} noev={len(noev)} "
        f"-> missing={len(missing)} ({doc['n_missing_real']} real + "
        f"{doc['n_missing_controls']} controls)"
    )
    return doc


def p2_parity(cfg: RecoveryCfg) -> None:
    """Prompt-parity HALT gate: the rebuilt describe user_msg must sha16-match
    the recorded prompt_sha16 for already-described features (no spend on
    evidence drift; plan critic MF1 / #922)."""
    report_path = cfg.reports / "p2_parity.json"
    if report_path.is_file() and json.loads(report_path.read_text())["verdict"] == "PASS":
        _log("[p2_parity] SKIP (PASS report exists)")
        return
    described = load_described(cfg.labels_dir)
    rng = random.Random(PARITY_SEED)
    sample = rng.sample(sorted(described), min(PARITY_N, len(described)))
    packets = load_packets_subset(cfg.evidence_dir, set(sample))
    mismatches: list[dict] = []
    for fid in sample:
        got = CM.sha16(DA.build_describe_items({fid: packets[fid]})[0][3])
        want = described[fid]
        if got != want:
            mismatches.append({"feat_id": fid, "rebuilt_sha16": got, "recorded_sha16": want})
    verdict = "PASS" if not mismatches else "HALT"
    _write_json(
        report_path,
        {
            "verdict": verdict,
            "n_checked": len(sample),
            "n_mismatch": len(mismatches),
            "mismatches": mismatches,
            "seed": PARITY_SEED,
            **CM.repro_meta(),
        },
    )
    _log(f"[p2_parity] {verdict}: {len(mismatches)}/{len(sample)} mismatches")
    if verdict != "PASS":
        _log(f"[p2_parity] HALT — evidence drift; NOT spending (report: {report_path})")
        sys.exit(RC_PARITY_HALT)


def p3_describe(cfg: RecoveryCfg) -> None:
    """Re-judge the missing describe set against the FIXED parser; kill gate
    on the residual non-empty parse-fail fraction BEFORE the axes leg."""
    out_desc = cfg.work / "descriptions_recovered.jsonl"
    report_path = cfg.reports / "p3_report.json"
    if (
        out_desc.is_file()
        and report_path.is_file()
        and json.loads(report_path.read_text())["gate"]["verdict"] == "PASS"
    ):
        _log("[p3_describe] SKIP (recovered descriptions + PASS gate exist)")
        return
    missing = json.loads((cfg.reports / "missing_set.json").read_text())["missing_feat_ids"]
    if cfg.smoke:
        # Reals-first slice: the production population is dominated by real
        # features (2,597 real vs 5 controls); a bare missing[:N] would take
        # ONLY controls (negative ids sort first) — the adversarial
        # refusal-cut-prone subpopulation (2026-07-31 smoke finding).
        missing = sorted(missing, key=lambda f: (f < 0, f))[: cfg.limit]
        _log(f"[p3_describe] SMOKE slice: {len(missing)} features (reals-first)")
    packets = load_packets_subset(cfg.evidence_dir, set(int(f) for f in missing))
    dispatch_packets = {f: p for f, p in packets.items() if p.get("ex_pos")}
    n_extra_noev = len(packets) - len(dispatch_packets)
    if n_extra_noev:
        _log(f"[p3_describe] {n_extra_noev} additional zero-evidence packets excluded ($0)")
    items = DA.build_describe_items(dispatch_packets)
    ckpt = cfg.work / "judge_checkpoints" / "p3_describe"
    _log(f"[p3_describe] dispatching {len(items)} items (Batch path pinned; ckpt={ckpt})")
    results = DA._dispatch(
        items,
        system=CM.DESCRIBER_SYSTEM,
        max_tokens=CM.DESCRIBE_MAX_TOKENS,
        checkpoint_dir=ckpt,
        force_batch=True,
    )
    from explore_persona_space.eval.batch_judge import is_transport_error_dict

    rows: list[dict] = []
    n_success_parsed = 0
    n_schema_fail = 0
    n_transport = 0
    n_other_content = 0
    parse_error_cids: list[str] = []
    raw_texts: dict[str, str] = {}
    for cid, _q, _c, user in items:
        assert cid in results, f"dispatch contract violated: no result for {cid}"
        res = results[cid]
        if isinstance(res, dict) and res.get("error"):
            if is_transport_error_dict(res):
                n_transport += 1
            elif res.get("reasoning") == "parse_error":
                parse_error_cids.append(cid)
            else:
                n_other_content += 1
            continue
        if isinstance(res, dict) and res.get("_raw_text"):
            raw_texts[cid] = res["_raw_text"]
        parsed = DA.parse_describe_result(res)
        n_success_parsed += 1
        if parsed is None:
            n_schema_fail += 1
            continue
        feat_id = int(cid[1:].rsplit("-", 1)[0])
        rows.append({"feat_id": feat_id, **parsed, "prompt_sha16": CM.sha16(user)})
    failed_meta = fetch_failed_response_meta(ckpt, set(parse_error_cids))
    gate = p3_gate_stats(n_success_parsed, [failed_meta.get(c, {}) for c in parse_error_cids])
    residual = residual_census(
        n_transport=n_transport,
        n_other_content=n_other_content,
        n_schema_fail=n_schema_fail,
        failed=[failed_meta.get(c, {}) for c in parse_error_cids],
        n_fresh_draws=len(items),
    )
    census = Counter(classify_response_shape(r) for r in raw_texts.values())
    census.update(
        classify_response_shape((failed_meta.get(c) or {}).get("raw_text"))
        for c in parse_error_cids
    )
    _write_jsonl(rows, out_desc)
    DA._write_raw(results, cfg.work / "judge_raw" / "recovery_describe_raw")
    if failed_meta:
        CM.write_jsonl_sharded(
            [{"custom_id": c, **m} for c, m in sorted(failed_meta.items())],
            cfg.work / "judge_raw",
            "recovery_describe_failed_raw",
        )
    report = {
        "n_items": len(items),
        "n_recovered": len(rows),
        "n_success_parsed": n_success_parsed,
        "n_schema_fail": n_schema_fail,
        "drops": {
            "transport": n_transport,
            "content_parse_error": len(parse_error_cids),
            "content_empty": gate["n_empty"],
            "content_refusal_cut": gate["n_refusal_cut"],
            "content_other": n_other_content,
        },
        "gate": gate,
        "census": dict(census),
        "residual_census": residual,
        "n_extra_zero_evidence": n_extra_noev,
        "max_tokens": CM.DESCRIBE_MAX_TOKENS,
        "rubric_sha16": CM.sha16(CM.DESCRIBER_SYSTEM),
        **CM.repro_meta(),
    }
    _write_json(report_path, report)
    _log(
        f"[p3_describe] done: {len(rows)}/{len(items)} recovered; gate="
        f"{gate['verdict']} (fail_nonempty={gate['n_parse_fail_nonempty']}/"
        f"nonempty={gate['n_nonempty']}, empty={gate['n_empty']}, "
        f"refusal_cut={gate['n_refusal_cut']}); census={dict(census)}"
    )
    if p3_halt_decision(gate["verdict"], cfg.smoke):
        _log(f"[p3_describe] HALT — parse-fail gate >= {P3_GATE_FLOOR} (report: {report_path})")
        sys.exit(RC_P3_GATE_HALT)
    if gate["verdict"] != "PASS":
        _log(
            "[p3_describe] SMOKE: gate verdict HALT is INFORMATIONAL at smoke n "
            "(production halt path is unit-pinned; #1345 gate-calibration rule)"
        )


def p4_axes(cfg: RecoveryCfg) -> None:
    """Re-judge the 5 axes for newly-described REAL features (DESC present).
    Prints the ops/cost arithmetic BEFORE submitting. Never dispatched under
    --smoke (plan §5A: p4 is not smoked live)."""
    out_axis = cfg.work / "axis_labels_recovered.jsonl"
    out_kappa = cfg.work / "kappa_recovered.json"
    report_path = cfg.reports / "p4_report.json"
    if out_axis.is_file() and out_kappa.is_file() and report_path.is_file():
        _log("[p4_axes] SKIP (outputs exist)")
        return
    recovered = list(CM.iter_jsonl(cfg.work / "descriptions_recovered.jsonl"))
    descriptions = {int(r["feat_id"]): r["description"] for r in recovered}
    real_ids = {f for f in descriptions if f >= 0}
    packets = load_packets_subset(cfg.evidence_dir, real_ids)
    items = DA.build_axes_items(packets, descriptions)
    n = len(items)
    in_mtok = n * AXES_IN_TOKENS_PER_CALL / 1e6
    out_mtok = n * AXES_OUT_TOKENS_PER_CALL / 1e6
    cost = in_mtok * BATCH_USD_PER_MTOK_IN + out_mtok * BATCH_USD_PER_MTOK_OUT
    _log(
        f"[p4_axes] ops: {len(real_ids)} features x {len(CM.AXES)} axes x {CM.N_DRAWS} draws"
        f" = {n} calls; projected tokens in={in_mtok:.1f}M out={out_mtok:.1f}M; projected"
        f" Batch cost ~${cost:.0f} (realized #1773 basis {AXES_IN_TOKENS_PER_CALL} in /"
        f" {AXES_OUT_TOKENS_PER_CALL} out per call)"
    )
    if cfg.smoke:
        _log("[p4_axes] SMOKE: dispatch SKIPPED (item build + arithmetic only)")
        _write_jsonl([], out_axis)
        _write_json(out_kappa, {"skipped": "smoke", "n_items_built": n})
        _write_json(report_path, {"skipped": "smoke", "n_items_built": n, **CM.repro_meta()})
        return
    ckpt = cfg.work / "judge_checkpoints" / "p4_axes"
    results = DA._dispatch(
        items,
        system=CM.AXIS_SYSTEM_PREAMBLE,
        max_tokens=CM.AXES_MAX_TOKENS,
        checkpoint_dir=ckpt,
        force_batch=True,
    )
    rows, kappa = DA.aggregate_axes(items, results)
    _write_jsonl(rows, out_axis)
    _write_json(
        out_kappa,
        {**CM.repro_meta(), "max_tokens": CM.AXES_MAX_TOKENS, "axes": kappa},
    )
    DA._write_raw(results, cfg.work / "judge_raw" / "recovery_axes_raw")
    _write_json(
        report_path,
        {
            "n_items": n,
            "n_rows": len(rows),
            "drop_report": {a: kappa[a]["drop_report"] for a in CM.AXES},
            **CM.repro_meta(),
        },
    )
    _log("[p4_axes] done: " + " ".join(f"{a}:k={kappa[a]['kappa']:.3f}" for a in CM.AXES))


def p5_outputs(cfg: RecoveryCfg) -> None:
    """Assemble the parallel output tree + merged convenience files, then
    upload to HF `issue1773_featurepipeline/recovery_1934/` (bulk
    upload_folder). NEVER mutates the committed originals or `fulldict/`."""
    report_path = cfg.reports / "p5_report.json"
    prior = json.loads(report_path.read_text()) if report_path.is_file() else None
    if p5_skip_decision(prior, cfg.no_upload):
        _log(f"[p5_outputs] SKIP (report exists; uploaded={bool(prior.get('uploaded'))})")
        return
    if prior is not None:
        _log(
            "[p5_outputs] RESUME: prior p5 completed WITHOUT upload; re-running "
            "assembly + upload (no judge spend)"
        )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    recovered_desc = list(CM.iter_jsonl(cfg.work / "descriptions_recovered.jsonl"))
    recovered_axis = list(CM.iter_jsonl(cfg.work / "axis_labels_recovered.jsonl"))
    p1 = json.loads((cfg.reports / "missing_set.json").read_text())
    p3 = json.loads((cfg.reports / "p3_report.json").read_text())

    # coverage before/after (real features; controls reported separately)
    described_ids = {
        int(r["feat_id"])
        for s in cfg.labels_dir.glob("descriptions.shard*.jsonl")
        for r in CM.iter_jsonl(s)
    }
    described_real = sum(1 for f in described_ids if f >= 0)
    recovered_real = sum(1 for r in recovered_desc if int(r["feat_id"]) >= 0)
    denom_real = described_real + p1["n_missing_real"]
    meta = {
        "join_rule": (
            "axis_labels_merged: recovered rows REPLACE original rows on (feat_id, axis); "
            "original rows carry desc_present = feat_id in the ORIGINAL described set "
            "(False == produced on an evidence-only prompt); recovered rows carry "
            "desc_present = true, source = recovery_1934. descriptions_merged: original "
            "UNION recovered (disjoint feat_ids), source column."
        ),
        "coverage": {
            "real_denominator": denom_real,
            "described_real_before": described_real,
            "described_real_after": described_real + recovered_real,
            "coverage_real_before": described_real / denom_real if denom_real else None,
            "coverage_real_after": (
                (described_real + recovered_real) / denom_real if denom_real else None
            ),
            "n_recovered_descriptions": len(recovered_desc),
            "n_recovered_axis_rows": len(recovered_axis),
            "n_axis_rows_resolved": sum(
                1 for r in recovered_axis if r.get("label") != "unresolved"
            ),
        },
        "drops": p3["drops"],
        "gate": p3["gate"],
        "census": {"shape": p3["census"], "residual": p3.get("residual_census")},
        "refusal_stopped": (p3.get("residual_census") or {}).get("refusal_stopped_any"),
        "smoke": cfg.smoke,
        **CM.repro_meta(),
    }
    shutil.copy2(cfg.work / "descriptions_recovered.jsonl", cfg.out_root)
    shutil.copy2(cfg.work / "axis_labels_recovered.jsonl", cfg.out_root)
    shutil.copy2(cfg.work / "kappa_recovered.json", cfg.out_root / "kappa_recovered.json")
    _write_json(cfg.out_root / "recovery_meta.json", meta)
    _log(f"[p5_outputs] wrote parallel tree -> {cfg.out_root}")

    # HF payload: merged convenience files + raw judge text (<=9 MB shards)
    hf_out = cfg.work / "hf_out"
    original_desc_rows = [
        r
        for s in sorted(cfg.labels_dir.glob("descriptions.shard*.jsonl"))
        for r in CM.iter_jsonl(s)
    ]
    CM.write_jsonl_sharded(
        merge_description_rows(original_desc_rows, recovered_desc), hf_out, "descriptions_merged"
    )
    del original_desc_rows
    original_axis_rows = [
        r for s in sorted(cfg.labels_dir.glob("axis_labels.shard*.jsonl")) for r in CM.iter_jsonl(s)
    ]
    merged_axis = merge_axis_rows(original_axis_rows, recovered_axis, described_ids)
    del original_axis_rows
    CM.write_jsonl_sharded(merged_axis, hf_out, "axis_labels_merged")
    n_merged_axis = len(merged_axis)
    del merged_axis
    for src in (cfg.out_root / "recovery_meta.json", cfg.out_root / "kappa_recovered.json"):
        shutil.copy2(src, hf_out / src.name)
    CM.write_jsonl_sharded(recovered_desc, hf_out, "descriptions_recovered")
    CM.write_jsonl_sharded(recovered_axis, hf_out, "axis_labels_recovered")
    raw_dir = cfg.work / "judge_raw"
    if raw_dir.is_dir():
        dest_raw = hf_out / "judge_raw"
        dest_raw.mkdir(parents=True, exist_ok=True)
        for f in sorted(raw_dir.glob("*.jsonl")) + sorted(raw_dir.glob("*.json")):
            shutil.copy2(f, dest_raw / f.name)
    if cfg.no_upload:
        _log("[p5_outputs] upload SKIPPED (--no-upload / smoke)")
        uploaded = False
    else:
        _log(f"[p5_outputs] uploading {hf_out} -> {CM.HF_PREFIX}/{HF_RECOVERY_PREFIX}")
        DA._upload_dir(hf_out, HF_RECOVERY_PREFIX)
        uploaded = True
    _write_json(
        report_path,
        {
            "out_root": str(cfg.out_root),
            "hf_prefix": f"{CM.HF_PREFIX}/{HF_RECOVERY_PREFIX}",
            "uploaded": uploaded,
            "n_merged_axis_rows": n_merged_axis,
            **CM.repro_meta(),
        },
    )
    _log(f"[p5_outputs] done (uploaded={uploaded})")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _import_check() -> int:
    """Execute every deferred import (the #606 --verify-imports gate)."""
    import anthropic  # noqa: F401

    from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: F401
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items  # noqa: F401
    from explore_persona_space.eval.utils import parse_judge_json  # noqa: F401
    from explore_persona_space.orchestrate import hub  # noqa: F401

    print("[import-check] OK: all deferred imports resolve", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--smoke", action="store_true", help="bounded slice, throwaway /tmp work")
    mode.add_argument("--full", action="store_true", help="production recovery run")
    ap.add_argument("--limit", type=int, default=5, help="smoke slice size (features)")
    ap.add_argument(
        "--force-batch",
        action="store_true",
        help=(
            "accepted for CLI parity with the plan; the Batch path is ALWAYS "
            "pinned (raw-text re-stream for the p3 gate requires it)"
        ),
    )
    ap.add_argument("--stage-root", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--phases", default=None, help="comma-separated phase subset (default: all)")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        return _import_check()
    if not (args.smoke or args.full):
        ap.error("one of --smoke / --full is required")
    cfg = build_cfg(args)
    _log(
        f"[driver] mode={'smoke' if cfg.smoke else 'full'} stage_root={cfg.stage_root} "
        f"work={cfg.work} out_root={cfg.out_root} no_upload={cfg.no_upload}"
    )
    selected = tuple(args.phases.split(",")) if args.phases else PHASES
    unknown = set(selected) - set(PHASES)
    assert not unknown, f"unknown phases: {sorted(unknown)} (valid: {PHASES})"
    runners = {
        "p0_stage": p0_stage,
        "p0b_lineage": p0b_lineage,
        "p1_missing": p1_missing,
        "p2_parity": p2_parity,
        "p3_describe": p3_describe,
        "p4_axes": p4_axes,
        "p5_outputs": p5_outputs,
    }
    for name in PHASES:
        if name not in selected:
            continue
        t0 = time.time()
        _log(f"[{name}] START")
        runners[name](cfg)
        _log(f"[{name}] END elapsed={time.time() - t0:.1f}s")
    _log("[driver] ALL PHASES DONE")
    return 0


if __name__ == "__main__":
    # Explicit exit before C-extension finalize (gotchas: PyGILState atexit race).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(main())
