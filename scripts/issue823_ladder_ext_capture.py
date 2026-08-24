"""P0-ext + P-OwnGen + P-Cap-ext + P-Store-ext driver for #823 follow-up
`origin-ladder-more-contexts` (plan v17 sections 4.3 / 7 / 9 / 10).

A NEW thin caller around the landed capture driver
(`scripts/issue823_ladder_capture.py`): it imports `_tf_extract_arm` and the
capture checkpoint/upload helpers rather than calling that driver's `main()`
(its `load_span_lengths` pins every span list to len == 5,000, which cannot
describe the 43,000 extension contexts — hence this caller with its own
`own_len_ext` source).

Phases (each a SEPARATE process invocation — the vLLM teardown gotcha forbids
running P-OwnGen's engine in the same process as the HF capture model):

  --phase p0ext    Stage every section-10 input (pod-side downloads at the
                   pins; destination-mount headroom assert), then gates
                   (a)-(g): (a) prompt-integrity byte-equality vs banked-roster
                   reconstruction; (b) assignment arithmetic + 5,000-row prefix
                   equality; (c) requested-max_tokens in {4096, 8192} with
                   consistent gen_wave/regen; (d) mask accounting ->
                   mask_ext.json (per-rung 2-arm masks + banked-bridge 4,629
                   mask; integrity-class new-invalid > 1% per arm aborts;
                   refusal-class REPORTED per arm x persona); (e)
                   duplicate-content sha at full consumed-corpus grain (> 2%
                   flags the dedup-sensitivity refit -- flag+report only, the
                   refit lives in the fits driver); (f) capture-convention
                   parity probe (64 ORIGINAL contexts re-captured vs the banked
                   pass_b `cx_last` rows; BINDING: cosine >= 0.999 per row;
                   max-rel + element-wise forensics REPORTED-never-asserted —
                   bf16 near-zero denominators make a max-rel ASSERT
                   structurally unpassable, r19); (g) span-length unit probe (64
                   banked common-valid b2 answers vs phase3_span_lengths['b2']
                   positional under the PARENT's persistence semantics:
                   stored = min(own_bare, pair_template_diff) when own_bare > 0
                   else pair_template_diff, own_bare = BARE tokenization of the
                   a_prime answer text (run_823 phase 3); >= 63/64 exact).
  --phase owngen   43,000 Qwen-2.5-7B-Instruct vLLM rollouts on the extension
                   contexts (temp 1.0 / top_p 0.95 / seed 42 / max_tokens 1024
                   -- the #779 pass-B own-answer recipe), chunk-checkpointed;
                   own-rollout TEXTS + own_len_ext.json uploaded to HF
                   raw_completions/ladder_ext_own/ BEFORE capture consumes the
                   lengths; own_len computed under the BANKED LADDER STORE's
                   realized own-side rule -- the TEMPLATE-DIFF span of the own
                   rollout (`own_len_ext_span`), matching the reused prefix
                   store (its producer truncated with own =
                   span_d['a_prime'][i], itself a template-diff span) so the
                   fits driver pools ONE truncation rule. The parent phase-3
                   BARE rule is probe (g)'s comparison target only
                   (`own_span_length`, banked b2 arm -- a different producer).
  --phase capext   43,000 context forwards -> cx_ext_block{b}.pt and 83,313
                   pair forwards -> v_pairs_ext_p{00..15}_block{b}.pt (fp32,
                   batch 8, left-pad, GENERATION_SUFFIX assert; span-mean with
                   truncation min(own_len_ext_i, pair_len_i), persona prompt
                   stripped). Per-(persona x block) checkpoints with .done.json
                   fingerprint sidecars (resume = fingerprint EQUALITY; a
                   mismatched sidecar routes the block to RECAPTURE, never a
                   silent skip and never bare file existence). Gate B in-run
                   pilot: first 2 timed batches at production shape; projected
                   wall > 2x the section-9 row => designed abort (report JSON +
                   rc 23). Completed blocks upload CONCURRENTLY (one bulk
                   upload_folder commit per block; fail-loud).
  --phase storeext Final NAME-SET diff (scoped list_repo_tree vs the local
                   shard list) must PASS, then writes the `_store_verified.json`
                   sentinel the fits driver requires, and mirrors mask_ext.json
                   + p0_ext_report.json + the capture digest to HF.

Designed-abort exit codes (canonical enumeration; `--list-rcs` prints it):

  rc 0   complete
  rc 23  Gate B capture-wall abort (projected > 2x plan section-9 row; report
         gate_b_abort_report.json) -- the SEMANTIC mirror of the parent
         RC_CAPTURE_WALL_ABORT (4), renumbered because the ext-gen driver
         inherits the parent's rc 4/5 and cross-driver rcs must be disjoint
  rc 12  gate (a) prompt-integrity mismatch (halt BEFORE mask construction)
  rc 13  gate (b) assignment arithmetic / banked 5,000-row prefix mismatch
  rc 14  gate (c) requested-max_tokens / gen_wave / regen inconsistency
  rc 15  gate (d) integrity-class (non-refusal) new-invalid rows > 1% per arm
  rc 16  probe (f) capture-convention parity failure (per-row cosine floor --
         a DIRECTIONAL seam; abort, never a tolerance to widen)
  rc 17  probe (g) span-length unit mismatch (< 63/64 exact)
  rc 18  staging / sentinel / store integrity failure (ext-gen sentinel
         incomplete, staged sha mismatch, store name-set diff failure)

Every halt writes its report JSON and exits with its distinct rc BEFORE any
downstream completion sentinel (`_owngen_complete.json`,
`_capture_complete.json`, `_store_verified.json`) is written.

Smoke blind-spot enumeration (plan-sanctioned downgrades, disclosed):
  - gate (d)'s integrity-class 1% trigger and gate (e)'s 2% duplicate flag are
    production-n-calibrated: under --smoke they are INFORMATIONAL (WARN) --
    a single bad row in a 16-context slice is 6%, so the production bands
    cannot bind the smoke leg (gotchas.md smoke/production GATE-CALIBRATION);
  - phase disk-headroom floors are scaled to the smoke footprint (2 GB);
  - capture timing is NOT certified by the smoke (Gate B's projection at smoke
    n is trivially under threshold);
  - no substituted implementations: smoke runs the production model, the
    production `_tf_extract_arm` / cx path, probes (f)+(g) at the SAME 64-row
    grain, the full banked staging, and the production upload path (smoke
    outputs land under the `_smoke`-suffixed HF subpaths + a smoke out-root).

Usage:
  uv run python scripts/issue823_ladder_ext_capture.py --list-rcs
  uv run python scripts/issue823_ladder_ext_capture.py --import-check
  uv run python scripts/issue823_ladder_ext_capture.py --phase p0ext --pre-gpu-check [--smoke]
  uv run python scripts/issue823_ladder_ext_capture.py --phase p0ext [--smoke ...]
  uv run python scripts/issue823_ladder_ext_capture.py --phase owngen [--smoke ...]
  uv run python scripts/issue823_ladder_ext_capture.py --phase capext [--smoke ...]
  uv run python scripts/issue823_ladder_ext_capture.py --phase storeext [--smoke ...]

Smoke reads the ext-gen SMOKE records via
`--ext-prefix issue823_inconsistent_origin_ladder_smoke` (or --ext-local-dir);
production reads the production prefix at a resolved pinned sha.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # creds + shared-VM thread caps BEFORE torch import (sibling pattern)

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Repo root on sys.path so `scripts.*` sibling imports resolve in script mode.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_823.run_823 import (
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    GENERATION_SUFFIX,
    _tf_extract_arm,
    log_phase,
    write_sentinel,
)
from explore_persona_space.orchestrate.hub import (
    _parse_shard_manifest,
    _upload_folder_filtered,
    list_hf_files_under_path,
    retry_transient,
    verify_repo_paths_uploaded,
)
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts import issue823_ladder_capture as CAP
from scripts import issue823_ladder_ext_gen as EXTGEN
from scripts.issue823_ladder_gen import (
    DATA_REPO,
    GEN_MAX_TOKENS,
    HF_PREFIX,
    N_CONTEXTS_FULL,
    N_PERSONAS,
    PARENT_REV,
    REGEN_MAX_TOKENS,
    _require_canonical_upload,
    _sha256_file,
    _utc_now,
    load_frozen_questions,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue823_ladder_ext_capture")

# ── Registered constants (plan v17 sections 4.3 / 7 / 9 / 10) ────────────────
FOLLOWUP_LABEL = "origin-ladder-more-contexts"
EXT_ARMS = EXTGEN.EXT_ARMS  # (1, 16)
N_PREFIX = EXTGEN.N_PREFIX  # 5,000
N_EXT_FULL = EXTGEN.N_EXT_FULL  # 43,000
RUNGS_FULL: tuple[int, ...] = (5_000, 12_000, 24_000, 48_000)
BRIDGE_MASK_EXPECTED = 4_629  # the parent's realized 7-pair ladder mask (plan section 4.4 G1)
N_FOLDS = 5  # parent 5-fold CV convention (n_train/fold rows in mask_ext.json)

# Banked pins (plan section 10). Bridge-mask id source: the COMMITTED parent
# npz (probed this round: keys incl. context_ids (4629,) int64).
BRIDGE_NPZ_RELPATH = "eval_results/issue_823/inconsistent_origin_ladder/percontext_ladder.npz"
PASS_B_REPO_PATH = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
PASS_B_REV = "c94070508aa1c1f9c015ceb072231a2e51b28b3f"
PARENT_STORE_FILES = [f"v_pairs_p{p:02d}.pt" for p in range(N_PERSONAS)] + ["pairs_index.json"]

# Own-rollout recipe (the #779 pass-B own-answer recipe, plan section 4.3).
OWN_GEN_TEMPERATURE = 1.0
OWN_GEN_TOP_P = 0.95
OWN_GEN_SEED = 42
OWN_GEN_MAX_TOKENS = 1024
OWN_MAX_MODEL_LEN = 8192
OWN_CHUNK_SIZE = 2_000

# Capture block layout (plan section 4.3 P-Cap-ext: ~2 GB checkpoint files).
CX_BLOCK_SIZE = 5_000
PAIR_BLOCK_SIZE = 5_000
# Ext-store truncation convention: min(own_td, pair) with the OWN side the
# TEMPLATE-DIFF span of the fresh own rollout (`own_len_ext_span`) — the
# BANKED ladder pair store's realized own-side rule (its producer, ladder
# rev 9a8d0f80, truncated with own = span_d["a_prime"][i], the a_prime arm's
# template-diff span), so banked + ext tensors POOLED by the fits driver share
# ONE truncation rule (v18 review bounce). Deliberately NOT
# CAP.TRUNC_CONVENTION verbatim: that string claims "(parent parity)", which
# is false on the own side (the parent phase-3 own rule is BARE — probe (g)
# validates THAT producer separately via `own_span_length`).
TRUNC_CONVENTION_EXT = (
    "min(own_template_diff, pair_template_diff); own_len==0 => no truncation "
    "(banked-ladder-store own-rule parity: own = template-diff span of the own rollout)"
)
CX_CONVENTION = "cx_last (pass_b last-prompt-token, all 28 layers)"

# Gates / probes (plan section 4.3 / 7).
PROBE_N_CONTEXTS = 64
PROBE_F_COSINE_FLOOR = 0.999
# Additive epsilon in probe (f)'s element-wise relative-diff denominator, and the
# near-zero banked-element census floor. Element-wise max-rel is
# REPORTED-never-asserted (r19; see probe_f_capture_parity docstring): the r18
# pod halt realized max_rel_median = 156250.0 = 0.15625/PROBE_F_REL_EPS at
# near-zero banked elements while every row's cosine passed with margin
# (cosine_min = 0.99970 >= 0.999).
PROBE_F_REL_EPS = 1e-6
PROBE_G_MIN_EXACT = 63
NEW_INVALID_ABORT_FRACTION = 0.01
DUP_CONTENT_FLAG_FRACTION = 0.02
PLANNED_CAPTURE_WALL_H_EXT = 2.0  # plan section-9 P-Cap-ext row (measured basis ~1.0 h)
CAPTURE_WALL_ABORT_FACTOR = 2.0
MEASURED_PILOT_BASIS_S_PER_ROW = 0.0284  # parent capture_digest.json .pilot (documentation)

# Designed-abort exit codes (docstring table is the canonical enumeration).
# rc 23 (NOT the parent's rc 4): the ext-gen driver inherits the parent's rc 4/5
# (payload-shape / roster-integrity), so this driver's Gate B wall abort takes a
# disjoint code — cross-driver rc disjointness is test-pinned (r1 concern
# round-rc4-collision).
RC_GATE_B_WALL = 23
RC_PROMPT_INTEGRITY = 12
RC_ASSIGNMENT_PARITY = 13
RC_MAX_TOKENS_WAVE = 14
RC_NEW_INVALID = 15
RC_CAPTURE_PARITY = 16
RC_SPAN_UNIT = 17
RC_STAGING_INTEGRITY = 18

RC_TABLE = {
    "0": "complete",
    "23": "Gate B capture-wall abort (projected > 2x plan section-9 row)",
    "12": "gate (a) prompt-integrity mismatch (before mask construction)",
    "13": "gate (b) assignment arithmetic / banked prefix mismatch",
    "14": "gate (c) requested-max_tokens / gen_wave / regen inconsistency",
    "15": "gate (d) integrity-class new-invalid rows > 1% per arm",
    "16": "probe (f) capture-convention parity failure (per-row cosine floor)",
    "17": "probe (g) span-length unit mismatch",
    "18": "staging / sentinel / store integrity failure",
}

OWNGEN_SENTINEL = "_owngen_complete.json"
CAPTURE_SENTINEL = "_capture_complete.json"
STORE_SENTINEL = "_store_verified.json"
P0_REPORT = "p0_ext_report.json"
MASK_FILE = "mask_ext.json"

# Disk-headroom floors per phase (plan section 9 rows; decimal GB at out-root).
PHASE_HEADROOM_GB = {"p0ext": 16.0, "owngen": 4.0, "capext": 56.0, "storeext": 2.0}
SMOKE_HEADROOM_GB = 2.0

PHASE_NAMES = ("p0ext", "owngen", "capext", "storeext")


class DesignedHalt(SystemExit):
    """A plan-registered designed abort: report written, distinct rc, no sentinel."""


def _halt(rc: int, report_path: pathlib.Path, report: dict, message: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, {"rc": rc, "verdict": "DESIGNED-HALT", "message": message, **report})
    logger.error("DESIGNED HALT rc=%d: %s (report: %s)", rc, message, report_path)
    raise DesignedHalt(rc)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(obj) -> str:
    return _sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


# ── Layout ────────────────────────────────────────────────────────────────────


class Layout:
    """Out-root + HF subpath layout for one run (production or smoke)."""

    def __init__(self, out_root: pathlib.Path, smoke: bool, n_ext: int):
        self.out_root = out_root
        self.smoke = smoke
        self.n_ext = n_ext
        self.n_prefix = N_PREFIX
        self.n_total = N_PREFIX + n_ext
        self.stage_dir = out_root / "ext_gen_inputs"
        self.banked_dir = out_root / "banked_inputs"
        self.own_dir = out_root / "own"
        self.store_dir = out_root / "pair_store"
        self.eval_dir = out_root  # mask_ext.json + p0_ext_report.json at out-root top (plan)
        sfx = "_smoke" if smoke else ""
        self.store_subpath = f"analysis_tensors/ext{sfx}"
        self.own_subpath = f"raw_completions/ladder_ext_own{sfx}"
        # Production gates mirror rides the ext-gen records prefix; smoke keeps
        # its mirror inside the smoke store prefix (disclosed in the docstring).
        self.gates_subpath = EXTGEN.HF_EXT_SUBPATH if not smoke else self.store_subpath
        self.rungs = tuple(r for r in RUNGS_FULL if r <= self.n_total)
        if self.n_total not in self.rungs:
            self.rungs = (*self.rungs, self.n_total)

    def hf_path(self, subpath: str) -> str:
        return f"{HF_PREFIX}/{subpath}"

    def headroom_gb(self, phase: str) -> float:
        return SMOKE_HEADROOM_GB if self.smoke else PHASE_HEADROOM_GB[phase]

    def sentinel_dir(self) -> pathlib.Path:
        return (
            pathlib.Path("/workspace/logs")
            if pathlib.Path("/workspace").exists()
            else self.out_root / "logs"
        )


def build_metadata(layout: Layout, phase: str) -> dict:
    return {
        "script": "scripts/issue823_ladder_ext_capture.py",
        "task": 823,
        "followup_label": FOLLOWUP_LABEL,
        "generated_at": _utc_now(),
        "model": DEFAULT_MODEL,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "parent_rev": PARENT_REV,
        "banked_ladder_rev": EXTGEN.BANKED_LADDER_REV,
        "pass_b_rev": PASS_B_REV,
        "n_ext_contexts": layout.n_ext,
        "n_total_contexts": layout.n_total,
        "smoke": layout.smoke,
        "trunc_convention": TRUNC_CONVENTION_EXT,
        "cx_convention": CX_CONVENTION,
        **as_metadata_dict(git_provenance(), phase=phase),
    }


# ── Staging: ext-gen records (manifest-first shard reassembly) ────────────────


def read_jsonl_manifest_first(base: pathlib.Path, stem: str) -> list[dict]:
    """Read `<stem>.jsonl` rows, reassembling line-shards via `<stem>.manifest.json`.

    Manifest present -> parse via the hub-pinned schema
    (`orchestrate.hub._parse_shard_manifest`), sha256-verify every part, concat
    in manifest order — NEVER a shard glob (upload-policy.md manifest-first
    consumer clause). Manifest absent -> plain `<stem>.jsonl`. Fail-loud on a
    missing part / sha mismatch / neither form present.
    """
    manifest_path = base / f"{stem}.manifest.json"
    if manifest_path.exists():
        parts, sha_by_part = _parse_shard_manifest(
            manifest_path.read_text(encoding="utf-8"), what=str(manifest_path)
        )
        lines: list[str] = []
        for part in parts:
            pp = base / part
            if not pp.exists():
                raise RuntimeError(f"shard manifest names missing part: {pp}")
            data = pp.read_bytes()
            want = sha_by_part.get(part)
            if want:
                got = hashlib.sha256(data).hexdigest()
                if got != want:
                    raise RuntimeError(f"{pp}: sha256 {got} != manifest {want}")
            # split("\n"), never splitlines(): real-corpus JSON strings carry raw
            # U+2028/U+2029/NEL that splitlines() shreds (#950; gotchas.md).
            lines.extend(data.decode("utf-8").split("\n"))
        rows = [json.loads(ln) for ln in lines if ln.strip()]
        logger.info("[stage] %s: reassembled %d parts (%d rows)", stem, len(parts), len(rows))
        return rows
    plain = base / f"{stem}.jsonl"
    if not plain.exists():
        raise RuntimeError(f"neither {manifest_path.name} nor {plain.name} present under {base}")
    return [json.loads(ln) for ln in plain.read_text(encoding="utf-8").split("\n") if ln.strip()]


def stage_ext_gen_inputs(
    layout: Layout,
    ext_prefix: str,
    revision: str | None,
    local_dir: pathlib.Path | None,
) -> tuple[pathlib.Path, dict, str]:
    """Stage the P-Gen-ext upload set (sentinel-first; sha-verified; pinned rev).

    Returns (base dir holding the files, parsed sentinel, resolved revision).
    The sentinel's `files_sha256` keys are the UPLOAD-REL names (for a sharded
    persona file: the shard + manifest names, not the original jsonl), so the
    integrity check runs over exactly the persisted set.
    """
    if local_dir is not None:
        base = local_dir
        sentinel_path = base / EXTGEN.SENTINEL_FILENAME
        if not sentinel_path.exists():
            _halt(
                RC_STAGING_INTEGRITY,
                layout.eval_dir / "ext_staging_report.json",
                {"reason": "sentinel_missing", "path": str(sentinel_path)},
                f"--ext-local-dir has no {EXTGEN.SENTINEL_FILENAME}",
            )
        sentinel = json.loads(sentinel_path.read_text())
        resolved = f"local:{base}"
    else:
        resolved = CAP.resolve_dataset_revision(revision)
        base = layout.stage_dir / ext_prefix / EXTGEN.HF_EXT_SUBPATH
        repo_rel = f"{ext_prefix}/{EXTGEN.HF_EXT_SUBPATH}"
        logger.info("[stage] ext-gen inputs from %s/%s @ %s", DATA_REPO, repo_rel, resolved)
        sp = pathlib.Path(
            retry_transient(
                lambda: hf_hub_download(
                    DATA_REPO,
                    f"{repo_rel}/{EXTGEN.SENTINEL_FILENAME}",
                    repo_type="dataset",
                    revision=resolved,
                    local_dir=layout.stage_dir,
                ),
                what=f"hf_hub_download({repo_rel}/{EXTGEN.SENTINEL_FILENAME})",
            )
        )
        sentinel = json.loads(sp.read_text())
        for rel in sorted(sentinel.get("files_sha256", {})):
            retry_transient(
                lambda rel=rel: hf_hub_download(
                    DATA_REPO,
                    f"{repo_rel}/{rel}",
                    repo_type="dataset",
                    revision=resolved,
                    local_dir=layout.stage_dir,
                ),
                what=f"hf_hub_download({repo_rel}/{rel})",
            )

    if not sentinel.get("complete"):
        _halt(
            RC_STAGING_INTEGRITY,
            layout.eval_dir / "ext_staging_report.json",
            {"reason": "sentinel_incomplete"},
            f"{EXTGEN.SENTINEL_FILENAME} has complete!=True — extension rollout text is not "
            "confirmed persisted; refusing (R3: text before any reduction)",
        )
    shas = sentinel.get("files_sha256")
    if not isinstance(shas, dict) or not shas:
        _halt(
            RC_STAGING_INTEGRITY,
            layout.eval_dir / "ext_staging_report.json",
            {"reason": "files_sha256_missing"},
            "ext-gen sentinel missing files_sha256 map — integrity unverifiable",
        )
    bad = []
    for rel, want in sorted(shas.items()):
        p = base / rel
        if not p.exists():
            bad.append({"file": rel, "problem": "missing"})
            continue
        got = _sha256_file(p)
        if got != want:
            bad.append({"file": rel, "problem": "sha_mismatch", "got": got, "want": want})
    if bad:
        _halt(
            RC_STAGING_INTEGRITY,
            layout.eval_dir / "ext_staging_report.json",
            {"reason": "staged_file_integrity", "bad": bad[:20], "n_bad": len(bad)},
            f"{len(bad)} staged ext-gen files missing or sha-mismatched vs the sentinel",
        )
    return base, sentinel, resolved


def load_ext_pair_rows(base: pathlib.Path, n_prefix: int, n_total: int) -> dict[int, list[dict]]:
    """Parse per-persona extension records; verify the cross-unit record contract.

    Asserts: corpus == 'ladder_ext'; gen_stage in {pilot, wave}; NO
    `in_common_valid` field (a prefix-only concept the ext records omit by
    contract); unique pairs matching the registered ext pair set; question
    consistency across personas; per-record `arms` matching persona(i, k) =
    i mod k membership over the round's two arms; per-arm row count == n_ext.
    """
    expected_pairs = EXTGEN.build_ext_pairs(n_prefix, n_total)
    by_persona: dict[int, list[dict]] = {}
    questions: dict[int, str] = {}
    seen: set[tuple[int, int]] = set()
    arm_counts = dict.fromkeys(EXT_ARMS, 0)
    for p in range(N_PERSONAS):
        stem = f"persona{p:02d}_ext"
        if not (base / f"{stem}.jsonl").exists() and not (base / f"{stem}.manifest.json").exists():
            by_persona[p] = []
            continue
        rows = [r for r in read_jsonl_manifest_first(base, stem) if r["context_id"] < n_total]
        for r in rows:
            assert r["persona_idx"] == p, f"{stem}: persona_idx={r['persona_idx']} record"
            assert r.get("corpus") == "ladder_ext", (
                f"{stem} ctx {r['context_id']}: corpus={r.get('corpus')!r} != 'ladder_ext'"
            )
            assert r.get("gen_stage") in ("pilot", "wave"), (
                f"{stem} ctx {r['context_id']}: gen_stage={r.get('gen_stage')!r}"
            )
            assert "in_common_valid" not in r, (
                f"{stem} ctx {r['context_id']}: ext records must omit in_common_valid"
            )
            i = r["context_id"]
            assert n_prefix <= i < n_total, f"{stem}: context_id {i} outside extension range"
            pair = (i, p)
            assert pair not in seen, f"duplicate ext pair {pair}"
            seen.add(pair)
            q = questions.setdefault(i, r["question"])
            assert q == r["question"], f"context {i}: question differs across persona files"
            expected_arms = [k for k in EXT_ARMS if (0 if k == 1 else i % k) == p]
            assert list(r["arms"]) == expected_arms, (
                f"pair {pair}: arms {r['arms']} != registered membership {expected_arms}"
            )
            for k in r["arms"]:
                arm_counts[k] += 1
            if r["filled"]:
                assert isinstance(r["answer_text"], str) and r["answer_text"], (
                    f"pair {pair}: filled=True but answer_text empty/non-str"
                )
        rows.sort(key=lambda r: r["context_id"])
        by_persona[p] = rows

    assert seen == expected_pairs, (
        f"realized ext pair set != registered: missing={len(expected_pairs - seen)} "
        f"extra={len(seen - expected_pairs)}"
    )
    n_ext = n_total - n_prefix
    for k, cnt in arm_counts.items():
        assert cnt == n_ext, f"ext arm k={k}: {cnt} rows != one-per-context {n_ext}"
    return by_persona


# ── Staging: banked section-10 inputs ─────────────────────────────────────────


def stage_banked_inputs(layout: Layout) -> dict:
    """Stage every banked pin the round consumes (idempotent; pinned revisions).

    Banked ladder gen records + roster/assignment at BANKED_LADDER_REV;
    phase3_span_lengths.json + b2_seed42.json + arm_a_prime_seed42.json
    (+ common_valid_idx) at PARENT_REV; the pass_b bundle at PASS_B_REV; the
    parent pair store (17 files, staged for the fits phase per plan
    section 4.3). The a_prime records + common-valid id set feed probe (g)'s
    producer-faithful expected value (own_bare truncation side + the parent's
    common-valid zeroing).
    """
    d = layout.banked_dir
    gen_paths, gen_rev = CAP.fetch_gen_inputs(
        d / "ladder_gen", HF_PREFIX, EXTGEN.BANKED_LADDER_REV, None
    )
    banked_sentinel = CAP.verify_gen_sentinel(gen_paths)
    roster_obj = json.loads(EXTGEN.fetch_banked_ladder_file(d, "roster.json").read_text())
    EXTGEN.assert_banked_roster_parity(roster_obj, layout.eval_dir)
    banked_assignment_obj = json.loads(
        EXTGEN.fetch_banked_ladder_file(d, "assignment.json").read_text()
    )
    span_path = CAP.fetch_span_lengths(d)
    span_d = CAP.load_span_lengths(span_path)
    questions_prefix, in_common, mask_crosscheck = load_frozen_questions(d)
    b2_path = d / EXTGEN.GEN.PARENT_PREFIX / "raw_completions" / "phase1" / "b2_seed42.json"
    b2_records = json.loads(b2_path.read_text())
    # a_prime answer texts: probe (g)'s own_bare truncation side (run_823 phase 3
    # banks b2 spans as min(bare(a_prime_text), template_diff(b2_text))). Observed
    # schema at PARENT_REV 8039d15f...: rows keyed
    # {answer_text, context_id, n_tokens, question}, positional 0..4999.
    a_prime_path = pathlib.Path(
        retry_transient(
            lambda: hf_hub_download(
                DATA_REPO,
                f"{EXTGEN.GEN.PARENT_PREFIX}/raw_completions/phase05/arm_a_prime_seed42.json",
                repo_type="dataset",
                revision=PARENT_REV,
                local_dir=d,
            ),
            what="hf_hub_download(phase05/arm_a_prime_seed42.json)",
        )
    )
    a_prime_records = json.loads(a_prime_path.read_text())
    a_prime_ids = [r["context_id"] for r in a_prime_records]
    if a_prime_ids != list(range(N_CONTEXTS_FULL)):
        raise RuntimeError(
            "arm_a_prime_seed42.json context_id sequence is not monotone-unique 0..4999 — "
            "positional own-length indexing would misalign; refusing to proceed"
        )
    a_prime_texts = [r["answer_text"] for r in a_prime_records]
    # The parent's common-valid zeroing source (run_823 phase 3 reads
    # common_valid_idx.json, NOT the per-record ride-along field); the file is
    # already staged by load_frozen_questions above.
    cv_path = d / EXTGEN.GEN.PARENT_PREFIX / "raw_completions" / "phase1" / "common_valid_idx.json"
    common_valid_ids = set(json.loads(cv_path.read_text())["common_valid_idx"])
    pass_b_path = pathlib.Path(
        retry_transient(
            lambda: hf_hub_download(
                DATA_REPO,
                PASS_B_REPO_PATH,
                repo_type="dataset",
                revision=PASS_B_REV,
                local_dir=d,
            ),
            what=f"hf_hub_download({PASS_B_REPO_PATH})",
        )
    )
    store_paths = {}
    for fn in PARENT_STORE_FILES:
        store_paths[fn] = pathlib.Path(
            retry_transient(
                lambda fn=fn: hf_hub_download(
                    DATA_REPO,
                    f"{HF_PREFIX}/analysis_tensors/{fn}",
                    repo_type="dataset",
                    revision=EXTGEN.BANKED_LADDER_REV,
                    local_dir=d,
                ),
                what=f"hf_hub_download({HF_PREFIX}/analysis_tensors/{fn})",
            )
        )
    return {
        "gen_paths": gen_paths,
        "gen_rev": gen_rev,
        "banked_sentinel": banked_sentinel,
        "roster_obj": roster_obj,
        "banked_assignment_obj": banked_assignment_obj,
        "span_path": span_path,
        "span_d": span_d,
        "questions_prefix": questions_prefix,
        "in_common": in_common,
        "mask_crosscheck": mask_crosscheck,
        "b2_records": b2_records,
        "a_prime_texts": a_prime_texts,
        "common_valid_ids": common_valid_ids,
        "pass_b_path": pass_b_path,
        "parent_store_paths": store_paths,
    }


def load_bridge_mask_ids() -> list[int]:
    """The parent's exact 4,629-context mask ids from the COMMITTED npz.

    Source: `context_ids` in percontext_ladder.npz (git main @ 84633d46c6 —
    plan section 10; observed keys probed this round). The pod clone carries
    eval_results/; a sparse worktree does not — fail loud naming the gap.
    """
    path = _REPO_ROOT / BRIDGE_NPZ_RELPATH
    if not path.exists():
        raise RuntimeError(
            f"{path} missing — the banked-bridge mask id source is the committed parent npz "
            "(present in a full clone; sparse worktrees exclude eval_results/)"
        )
    z = np.load(path, allow_pickle=False)
    if "context_ids" not in z.files:
        raise RuntimeError(f"{path}: no context_ids key (observed keys: {z.files})")
    ids = [int(i) for i in z["context_ids"]]
    if len(ids) != BRIDGE_MASK_EXPECTED:
        raise RuntimeError(
            f"banked-bridge mask has {len(ids)} ids != registered {BRIDGE_MASK_EXPECTED}"
        )
    return ids


# ── P0-ext gates (a)-(e) ──────────────────────────────────────────────────────


def gate_a_prompt_integrity(
    ext_by_persona: dict[int, list[dict]], roster_obj: dict, eval_dir: pathlib.Path
) -> dict:
    """(a) byte-equality of every persisted dispatched system prompt vs the
    banked-roster reconstruction; any mismatch is a designed halt (rc 12)
    BEFORE mask construction."""
    n_checked = 0
    mismatches: list[dict] = []
    for p, rows in sorted(ext_by_persona.items()):
        if not rows:
            continue
        expected = EXTGEN.persona_system_from_roster(roster_obj, p)
        expected_sha = _sha256_text(expected)
        for r in rows:
            n_checked += 1
            if r["system_prompt"] != expected or r["system_prompt_sha256"] != expected_sha:
                mismatches.append({"context_id": r["context_id"], "persona_idx": p})
    if mismatches:
        _halt(
            RC_PROMPT_INTEGRITY,
            eval_dir / "ext_prompt_integrity_report.json",
            {"n_checked": n_checked, "n_mismatched": len(mismatches), "first20": mismatches[:20]},
            f"gate (a): {len(mismatches)} ext pairs' persisted system prompt != banked-roster "
            "reconstruction",
        )
    return {"n_checked": n_checked, "pass": True}


def gate_b_assignment(
    staged_assignment_obj: dict,
    banked_assignment_obj: dict,
    n_total: int,
    eval_dir: pathlib.Path,
) -> dict:
    """(b) independent i-mod-k recomputation over all n_total contexts + the
    banked 5,000-row prefix equality (rc 13)."""
    recomputed = EXTGEN.build_ext_assignment(n_total)
    staged_arms = staged_assignment_obj["arms"]
    bad_arms = [str(k) for k in EXT_ARMS if staged_arms.get(str(k)) != recomputed[k]]
    if bad_arms or staged_assignment_obj.get("n_contexts") != n_total:
        _halt(
            RC_ASSIGNMENT_PARITY,
            eval_dir / "ext_assignment_report.json",
            {
                "reason": "staged_assignment_mismatch",
                "mismatched_arms": bad_arms,
                "staged_n_contexts": staged_assignment_obj.get("n_contexts"),
                "n_total": n_total,
            },
            f"gate (b): staged assignment_ext.json disagrees with the i-mod-k recomputation "
            f"(arms {bad_arms})",
        )
    banked_arms = banked_assignment_obj["arms"]
    n_banked = banked_assignment_obj["n_contexts"]
    bad_prefix = [str(k) for k in EXT_ARMS if recomputed[k][:n_banked] != banked_arms[str(k)]]
    if bad_prefix:
        _halt(
            RC_ASSIGNMENT_PARITY,
            eval_dir / "ext_assignment_report.json",
            {
                "reason": "banked_prefix_mismatch",
                "mismatched_arms": bad_prefix,
                "n_banked": n_banked,
            },
            f"gate (b): {n_banked}-row prefix != banked assignment.json arms {bad_prefix}",
        )
    return {"n_total": n_total, "n_banked_prefix": n_banked, "pass": True}


def gate_c_max_tokens(ext_by_persona: dict[int, list[dict]], eval_dir: pathlib.Path) -> dict:
    """(c) per-row requested max_tokens in {GEN_MAX_TOKENS, REGEN_MAX_TOKENS}
    with consistent regen flag + non-empty gen_wave (rc 14)."""
    allowed = {GEN_MAX_TOKENS: False, REGEN_MAX_TOKENS: True}
    bad: list[dict] = []
    n = 0
    for p, rows in sorted(ext_by_persona.items()):
        for r in rows:
            n += 1
            mt = r["max_tokens"]
            problem = None
            if mt not in allowed:
                problem = f"max_tokens {mt} not in {sorted(allowed)}"
            elif bool(r.get("regen")) != allowed[mt]:
                problem = f"regen={r.get('regen')} inconsistent with max_tokens {mt}"
            elif not isinstance(r.get("gen_wave"), str) or not r["gen_wave"]:
                problem = "gen_wave missing/empty"
            if problem:
                bad.append({"context_id": r["context_id"], "persona_idx": p, "problem": problem})
    if bad:
        _halt(
            RC_MAX_TOKENS_WAVE,
            eval_dir / "ext_max_tokens_report.json",
            {"n_checked": n, "n_bad": len(bad), "first20": bad[:20]},
            f"gate (c): {len(bad)} ext rows with inconsistent max_tokens/gen_wave/regen",
        )
    return {"n_checked": n, "pass": True}


def _pair_validity(
    banked_by_persona: dict[int, list[dict]],
    ext_by_persona: dict[int, list[dict]],
    n_prefix: int,
    n_total: int,
) -> dict[tuple[int, int], str]:
    """(context, persona) -> validity for the two-arm pair set over all contexts."""
    val: dict[tuple[int, int], str] = {}
    for p, rows in banked_by_persona.items():
        for r in rows:
            val[(r["context_id"], p)] = r["validity"]
    for p, rows in ext_by_persona.items():
        for r in rows:
            val[(r["context_id"], p)] = r["validity"]
    for i in range(n_total):
        for pair in ((i, 0), (i, i % 16)):
            if pair not in val:
                raise RuntimeError(f"pair {pair} has no record — two-arm pair set incomplete")
    return val


def build_masks(
    banked_by_persona: dict[int, list[dict]],
    ext_by_persona: dict[int, list[dict]],
    bridge_ids: list[int],
    layout: Layout,
    metadata: dict,
) -> dict:
    """(d) mask accounting -> mask_ext.json content (rc 15 on integrity-class).

    Mask rule per rung r: context i in mask iff i < r AND both arms valid
    (stop_reason-keyed validity == 'ok'; equalize-down — both arms share ONE
    mask). Integrity-class (non-refusal) new-invalid EXT rows > 1% per arm =>
    designed abort (WARN under --smoke: production-n-calibrated band).
    Refusal-class drops are REPORTED per (arm x persona), never a halt.
    """
    n_prefix, n_total = layout.n_prefix, layout.n_total
    val = _pair_validity(banked_by_persona, ext_by_persona, n_prefix, n_total)

    both_valid = [val[(i, 0)] == "ok" and val[(i, i % 16)] == "ok" for i in range(n_total)]
    rungs_obj: dict[str, dict] = {}
    for r in layout.rungs:
        ids = [i for i in range(min(r, n_total)) if both_valid[i]]
        n_mask = len(ids)
        fold_max = -(-n_mask // N_FOLDS)  # ceil
        n_train_min = n_mask - fold_max
        rungs_obj[str(r)] = {
            "rung": r,
            "n_mask": n_mask,
            "n_train_per_fold_min": n_train_min,
            "n_train_per_fold_mean": n_mask * (N_FOLDS - 1) / N_FOLDS,
            "n_over_d": n_train_min / EXPECTED_HIDDEN,
            "ids": ids,
        }

    # Cross-check the committed bridge ids against a record-side recompute
    # (reported, never a halt — the parent's realized mask is authoritative).
    bridge_recompute = [
        i
        for i in range(n_prefix)
        if all(val.get((i, p2)) == "ok" for p2 in {i % k for k in EXTGEN.GEN.K_ARMS})
    ]
    bridge_report = {
        "n_ids": len(bridge_ids),
        "ids_source": BRIDGE_NPZ_RELPATH,
        "n_recomputed_all_pairs_ok": len(bridge_recompute),
        "recompute_set_equal": set(bridge_recompute) == set(bridge_ids),
    }

    # Integrity-class new-invalid + refusal reporting over EXT rows only.
    arm_stats: dict[str, dict] = {}
    violations: list[tuple[int, float]] = []
    refusal_by_arm_persona: dict[str, dict[str, float]] = {}
    for k in EXT_ARMS:
        rows_k = [r for rows in ext_by_persona.values() for r in rows if k in r["arms"]]
        n_rows = len(rows_k)
        n_invalid = sum(1 for r in rows_k if r["validity"] != "ok" and r["validity"] != "refusal")
        n_refusal = sum(1 for r in rows_k if r["validity"] == "refusal")
        frac_invalid = (n_invalid / n_rows) if n_rows else 0.0
        arm_stats[str(k)] = {
            "n_rows": n_rows,
            "n_new_invalid_integrity": n_invalid,
            "new_invalid_fraction": frac_invalid,
            "n_refusal": n_refusal,
            "refusal_fraction": (n_refusal / n_rows) if n_rows else 0.0,
        }
        if frac_invalid > NEW_INVALID_ABORT_FRACTION:
            violations.append((k, frac_invalid))
        per_p: dict[str, float] = {}
        for p, rows in sorted(ext_by_persona.items()):
            arm_rows = [r for r in rows if k in r["arms"]]
            if arm_rows:
                per_p[str(p)] = sum(1 for r in arm_rows if r["validity"] == "refusal") / len(
                    arm_rows
                )
        refusal_by_arm_persona[str(k)] = per_p

    mask_obj = {
        "metadata": metadata,
        "mask_rule": "i < rung AND both arms valid (stop_reason-keyed; equalize-down)",
        "rungs": rungs_obj,
        "bridge": {"n_mask": len(bridge_ids), "ids": bridge_ids, **bridge_report},
        "ext_arm_stats": arm_stats,
        "refusal_fraction_by_arm_persona": refusal_by_arm_persona,
        "new_invalid_abort_fraction": NEW_INVALID_ABORT_FRACTION,
    }
    if violations:
        msg = (
            f"gate (d): integrity-class new-invalid ext rows over {NEW_INVALID_ABORT_FRACTION:.0%}"
            f" per arm: {violations} — pipeline fault (refusal-class is reported, not counted)"
        )
        if layout.smoke:
            logger.warning("SMOKE-INFORMATIONAL (enumerated blind spot): %s", msg)
            mask_obj["integrity_gate"] = "WARN-SMOKE-INFORMATIONAL"
        else:
            _halt(
                RC_NEW_INVALID,
                layout.eval_dir / "ext_new_invalid_report.json",
                {"violations": violations, "ext_arm_stats": arm_stats},
                msg,
            )
    else:
        mask_obj["integrity_gate"] = "PASS"
    return mask_obj


def gate_e_duplicates(questions_by_id: dict[int, str], layout: Layout) -> dict:
    """(e) duplicate-content fraction at the FULL consumed-corpus grain
    (raw-string sha256; > 2% flags the dedup-mask sensitivity refit — flag +
    report only; the refit itself lives in the fits driver).

    Persists the duplicate GROUPS (sorted context ids + min-id representative)
    so the fits driver's `sens_dedup` consumer can build the deduped masks
    without re-reading question text (r1 concern dedup-sensitivity-detached).
    """
    by_sha: dict[str, list[int]] = {}
    for i, q in sorted(questions_by_id.items()):
        by_sha.setdefault(_sha256_text(q), []).append(int(i))
    n = len(questions_by_id)
    n_unique = len(by_sha)
    frac = 1.0 - (n_unique / n) if n else 0.0
    flag = frac > DUP_CONTENT_FLAG_FRACTION
    if flag:
        logger.warning(
            "gate (e): duplicate-content fraction %.3f > %.2f — dedup-mask sensitivity refit "
            "FLAGGED for the fits driver (report only, no halt)",
            frac,
            DUP_CONTENT_FLAG_FRACTION,
        )
    groups = [
        {"context_ids": sorted(ids), "representative": min(ids)}
        for ids in by_sha.values()
        if len(ids) > 1
    ]
    groups.sort(key=lambda g: g["representative"])
    return {
        "n_contexts": n,
        "n_unique": n_unique,
        "duplicate_fraction": frac,
        "n_duplicate_groups": len(groups),
        "duplicate_groups": groups,
        "dedup_sensitivity_refit_required": flag,
    }


# ── Probes (f) + (g) ─────────────────────────────────────────────────────────


def own_span_length(tokenizer, own_text: str) -> int:
    """PROBE (g) ONLY — own-answer span length under the PARENT's convention
    (run_823 phase 3): BARE tokenization of the answer text — no chat
    template, no end-of-turn tokens (0 for empty text). The parent banked
    span_lengths['b2'][i] = min(own_bare_i, raw_i), own_bare from its
    `a_prime_token_lengths` (bare), the pair side the template-diff span. On
    seam-clean rows the template-diff span exceeds this by the 2 end-of-turn
    tokens (`<|im_end|>` + `\\n`) — conflating the two conventions is the #823
    rc=17 crash class. The EXT store's own side is `own_len_ext_span`
    (template-diff, banked-ladder-store parity) — a deliberately DIFFERENT
    rule for a different comparison target (v18 review bounce); do NOT
    re-unify them."""
    if not own_text:
        return 0
    return len(tokenizer(own_text, return_tensors=None, add_special_tokens=False)["input_ids"])


def own_len_ext_span(tokenizer, question: str, own_text: str) -> int:
    """EXT-store own-answer length under the BANKED LADDER STORE's realized
    own-side rule: the TEMPLATE-DIFF span (full_len - prompt_len via
    `template_span_length`) of the fresh own rollout — 0 for empty text. The
    reused 14,996-pair prefix store truncated with own = span_d['a_prime'][i]
    (`issue823_ladder_capture.own_length`), the a_prime arm's template-diff
    span, so extension tensors must share that rule or the fits driver pools
    TWO truncation conventions into one training set — a ~2-token within-fit
    heterogeneity confounded with training-set size (v18 review bounce).
    Deliberately NOT `own_span_length` (bare): that is the parent run_823
    phase-3 rule probe (g) validates against the banked b2 arm, a DIFFERENT
    producer."""
    if not own_text:
        return 0
    prompt_len, full_len = CAP.template_span_length(tokenizer, question, own_text)
    return full_len - prompt_len


def probe_g_span_unit(
    tokenizer,
    b2_records: list[dict],
    span_d: dict[str, list[int]],
    a_prime_texts: list[str],
    common_valid_ids: set[int],
    eval_dir: pathlib.Path,
) -> dict:
    """(g) recompute pair span lengths for 64 banked common-valid b2 answers
    under the PARENT's persistence semantics; compare vs
    phase3_span_lengths['b2'] positionally (>= 63/64 exact => PASS, else rc 17).

    The parent (`run_823.py` phase 3) banks span_lengths[i] =
    min(own_bare_i, raw_i) when own_bare_i > 0 else raw_i, where raw_i is the
    b2 answer's template-diff span (full_len - prompt_len via
    `template_span_length`) and own_bare_i is the BARE tokenization length of
    the a_prime answer text (no chat template, no end-of-turn tokens) — NOT
    span_d['a_prime'][i], which is the a_prime arm's own template-diff span
    (= own_bare + 2 end-of-turn tokens on seam-clean rows; #823 rc=17 crash
    round). Dual drift check, preserving tokenization-drift catching power on
    both conventions: untruncated rows (own_bare >= raw, or own_bare == 0)
    must equal the recomputed raw (template-path catcher); truncated rows
    (0 < own_bare < raw) must equal the recomputed own_bare (bare-path +
    min-rule catcher). Non-common-valid contexts are excluded from picking:
    the parent zeroed every arm text there (banked span 0 by construction), so
    they carry no tokenization signal. Aggregate floor UNCHANGED: >= 63/64.
    """
    picked = [
        r
        for r in b2_records
        if r.get("filled") and r.get("answer_text") and r["context_id"] in common_valid_ids
    ]
    picked = picked[:PROBE_N_CONTEXTS]
    if len(picked) < PROBE_N_CONTEXTS:
        raise RuntimeError(
            f"probe (g): only {len(picked)} filled common-valid b2 records available "
            f"(< {PROBE_N_CONTEXTS})"
        )
    n_exact = 0
    n_trunc = n_trunc_exact = 0
    n_untrunc = n_untrunc_exact = 0
    mismatches: list[dict] = []
    for r in picked:
        i = r["context_id"]
        prompt_len, full_len = CAP.template_span_length(tokenizer, r["question"], r["answer_text"])
        raw = full_len - prompt_len
        own_text = a_prime_texts[i]
        own = own_span_length(tokenizer, own_text)
        expected = min(own, raw) if own > 0 else raw
        want = span_d["b2"][i]
        if 0 < own < raw:
            n_trunc += 1
            n_trunc_exact += int(want == own)
        else:
            n_untrunc += 1
            n_untrunc_exact += int(want == raw)
        if want == expected:
            n_exact += 1
        else:
            mismatches.append(
                {
                    "context_id": i,
                    "recomputed_raw": raw,
                    "own_bare": own,
                    "expected": expected,
                    "stored": want,
                }
            )
    report = {
        "n_checked": len(picked),
        "n_exact": n_exact,
        "min_exact": PROBE_G_MIN_EXACT,
        "n_truncated_checked": n_trunc,
        "n_truncated_exact": n_trunc_exact,
        "n_untruncated_checked": n_untrunc,
        "n_untruncated_exact": n_untrunc_exact,
        "mismatches_first10": mismatches[:10],
    }
    if n_exact < PROBE_G_MIN_EXACT:
        _halt(
            RC_SPAN_UNIT,
            eval_dir / "ext_span_unit_report.json",
            report,
            f"probe (g): {n_exact}/{len(picked)} exact span matches < {PROBE_G_MIN_EXACT} — "
            "wrong tokenization rule",
        )
    report["pass"] = True
    return report


def _cx_from_token_ids(
    model, ids_list: list[list[int]], layers: list[int], batch_size: int, pad_id: int
) -> np.ndarray:
    """Batched last-real-token residual capture -> (n, len(layers), hidden) fp32.

    Mirrors `_tf_extract_arm`'s batching mechanics exactly: LEFT pad, explicit
    position_ids = clamp(cumsum(attention_mask) - 1, min=0) (RoPE correctness
    under left pad), forward hooks on model.model.layers, GPU-side gather of
    the final position (the last REAL token under left padding), fp32 to CPU.
    """
    n = len(ids_list)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[li] = hidden.detach()

        return hook

    handles = [
        model.model.layers[layer_idx].register_forward_hook(make_hook(li))
        for li, layer_idx in enumerate(layers)
    ]
    dev = next(model.parameters()).device
    out_np: np.ndarray | None = None
    try:
        with torch.no_grad():
            for b_start in range(0, n, batch_size):
                batch = ids_list[b_start : b_start + batch_size]
                max_len = max(len(ids) for ids in batch)
                input_ids, attn = [], []
                for ids in batch:
                    pad_n = max_len - len(ids)
                    input_ids.append([pad_id] * pad_n + list(ids))
                    attn.append([0] * pad_n + [1] * len(ids))
                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=dev)
                attn_t = torch.tensor(attn, dtype=torch.long, device=dev)
                position_ids_t = (attn_t.cumsum(dim=-1) - 1).clamp(min=0)
                captured.clear()
                model(
                    input_ids=input_ids_t,
                    attention_mask=attn_t,
                    position_ids=position_ids_t,
                    output_hidden_states=False,
                )
                if out_np is None:
                    hidden_dim = captured[0].shape[-1]
                    out_np = np.zeros((n, len(layers), hidden_dim), dtype=np.float32)
                for li in range(len(layers)):
                    # Last position == last real token under LEFT padding.
                    vecs = captured[li][:, -1, :].float().cpu().numpy()
                    out_np[b_start : b_start + len(batch), li, :] = vecs
                captured.clear()
    finally:
        for h in handles:
            h.remove()
        captured.clear()
    assert out_np is not None and out_np.shape[0] == n, "cx capture produced no output"
    return out_np


def capture_cx(model, tokenizer, questions: list[str], batch_size: int) -> np.ndarray:
    """cx_last for bare-user-question contexts (the pass_b convention)."""
    ids_list: list[list[int]] = []
    for q in questions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        suffix = tokenizer.decode(ids[-3:])
        assert suffix == GENERATION_SUFFIX, (
            f"cx position assert failed: {suffix!r} != {GENERATION_SUFFIX!r}"
        )
        ids_list.append(ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    v = _cx_from_token_ids(model, ids_list, list(range(EXPECTED_LAYERS)), batch_size, pad_id)
    assert v.shape == (len(questions), EXPECTED_LAYERS, EXPECTED_HIDDEN), v.shape
    return v


def load_pass_b_cx_last(pass_b_path: pathlib.Path) -> torch.Tensor:
    """cx_last from the banked pass_b bundle (dict schema probed this round).

    Observed: 6.0 GB torch bundle; consumers read bundle['cx_last'] with
    shape (5000, 28, 3584) fp32 (issue779_arm_headline.py:203). Self-produced
    sha-pinned artifact => weights_only=False (torch>=2.6 policy); mmap=True
    keeps the 6 GB bundle off resident memory.
    """
    obj = torch.load(pass_b_path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and "cx_last" in obj:
        t = obj["cx_last"]
    else:
        keys = sorted(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
        raise RuntimeError(f"pass_b bundle has no cx_last (observed: {keys})")
    assert t.shape == (N_CONTEXTS_FULL, EXPECTED_LAYERS, EXPECTED_HIDDEN), t.shape
    return t


def probe_f_capture_parity(
    model,
    tokenizer,
    questions_prefix: list[str],
    pass_b_path: pathlib.Path,
    eval_dir: pathlib.Path,
    batch_size: int,
) -> dict:
    """(f) re-capture cx_last for 64 ORIGINAL contexts with the extension rig
    vs the banked pass_b rows.

    BINDING leg (rc 16): cosine >= PROBE_F_COSINE_FLOOR per row — a real
    formatting/position/pad seam reads cosine ~0.39-0.84 (the #779 calibration;
    gotchas.md bf16 parity-gate entries), so the floor separates a directional
    seam from bf16 numerics with wide margin. max-rel and every other
    element-wise statistic are REPORTED-never-asserted (r19; the canonical
    exemplar: scripts/issue779_capture_answer_summaries_pass2.py::
    equivalence_gate_p2): both captures are bf16 forwards under different
    batch composition, so an element-wise max-rel ASSERT is structurally
    unpassable at near-zero banked elements — |b| ~ 0 makes
    rel = |a-b|/PROBE_F_REL_EPS, and the r18 pod halt realized
    max_rel_median = 156250.0 (= 0.15625/1e-6) with cosine_min = 0.99970 on
    all 64 rows. NO additional binding leg: a per-row rel-L2 cap has no
    committed same-surface reference value to derive from (artifact-reuse.md
    § Reuse-validation gate calibration — the #928 0.999 / #1005 0.9999
    references are cosine bars), and cosine is the validated bug-catcher.
    The forensic fields (per-row rel-L2, abs-diff quantiles, denominator
    magnitude at each row's argmax-rel element, near-zero banked-element
    census) let the next report attribute a miss — bf16 noise at near-zero
    elements vs a genuine seam — without another forensic round-trip."""
    ref = load_pass_b_cx_last(pass_b_path)[:PROBE_N_CONTEXTS].float().numpy()
    got = capture_cx(model, tokenizer, questions_prefix[:PROBE_N_CONTEXTS], batch_size)
    a = got.reshape(PROBE_N_CONTEXTS, -1).astype(np.float64)
    b = ref.reshape(PROBE_N_CONTEXTS, -1).astype(np.float64)
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    abs_diff = np.abs(a - b)
    abs_b = np.abs(b)
    rel = abs_diff / (abs_b + PROBE_F_REL_EPS)
    max_rel = rel.max(axis=1)
    argmax_rel = rel.argmax(axis=1)
    rows = np.arange(PROBE_N_CONTEXTS)
    denom_at_argmax = abs_b[rows, argmax_rel]
    diff_at_argmax = abs_diff[rows, argmax_rel]
    rel_l2 = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)
    n_near_zero = int((abs_b < PROBE_F_REL_EPS).sum())
    report = {
        "n_rows": PROBE_N_CONTEXTS,
        "cosine_min": float(cos.min()),
        "cosine_median": float(np.median(cos)),
        "n_rows_below_cosine_floor": int((cos < PROBE_F_COSINE_FLOOR).sum()),
        "cosine_floor": PROBE_F_COSINE_FLOOR,
        # Everything below is REPORTED-never-asserted forensics (docstring).
        "rel_denominator_eps": PROBE_F_REL_EPS,
        "max_rel_median": float(np.median(max_rel)),
        "max_rel_max": float(max_rel.max()),
        "rel_l2_per_row": [float(x) for x in rel_l2],
        "rel_l2_median": float(np.median(rel_l2)),
        "rel_l2_max": float(rel_l2.max()),
        "abs_diff_p50": float(np.percentile(abs_diff, 50)),
        "abs_diff_p90": float(np.percentile(abs_diff, 90)),
        "abs_diff_max": float(abs_diff.max()),
        "denom_at_argmax_rel_per_row": [float(x) for x in denom_at_argmax],
        "denom_at_argmax_rel_median": float(np.median(denom_at_argmax)),
        "diff_at_argmax_rel_median": float(np.median(diff_at_argmax)),
        "near_zero_banked_floor": PROBE_F_REL_EPS,
        "n_banked_below_floor": n_near_zero,
        "frac_banked_below_floor": float(n_near_zero / abs_b.size),
    }
    if (cos < PROBE_F_COSINE_FLOOR).any():
        _halt(
            RC_CAPTURE_PARITY,
            eval_dir / "ext_capture_parity_report.json",
            report,
            "probe (f): extension-rig cx_last diverges DIRECTIONALLY from the banked pass_b "
            f"convention (per-row cosine floor {PROBE_F_COSINE_FLOOR}; a real formatting/"
            "position seam reads cosine ~0.39-0.84, #779 calibration) — aborting before "
            "production capture",
        )
    report["pass"] = True
    return report


# ── P-OwnGen helpers ─────────────────────────────────────────────────────────


def own_chunk_fingerprint(context_ids: list[int], questions: list[str]) -> dict:
    """Machine-stable resume key for one own-rollout chunk (generating params +
    input identity; code sha recorded in the sidecar OUTSIDE the equality key)."""
    return {
        "recipe": {
            "model": DEFAULT_MODEL,
            "temperature": OWN_GEN_TEMPERATURE,
            "top_p": OWN_GEN_TOP_P,
            "seed": OWN_GEN_SEED,
            "max_tokens": OWN_GEN_MAX_TOKENS,
            "max_model_len": OWN_MAX_MODEL_LEN,
        },
        "context_ids_sha256": _sha256_json(context_ids),
        "questions_sha256": _sha256_json([_sha256_text(q) for q in questions]),
    }


def own_chunk_done(own_dir: pathlib.Path, c: int, fingerprint: dict) -> bool:
    """Resume predicate: fingerprint EQUALITY; mismatch/partial => REGENERATE."""
    meta_path = own_dir / f"own_chunk{c:03d}.meta.json"
    rows_path = own_dir / f"own_chunk{c:03d}.jsonl"
    if not meta_path.exists() or not rows_path.exists():
        if meta_path.exists() != rows_path.exists():
            logger.warning("[owngen] chunk %03d partial on disk — regenerating", c)
        return False
    meta = json.loads(meta_path.read_text())
    if meta.get("fingerprint") != fingerprint:
        logger.warning("[owngen] chunk %03d fingerprint mismatch — regenerating (never skip)", c)
        return False
    return True


# ── P-Cap-ext helpers ────────────────────────────────────────────────────────


def precompute_ext_rows(tokenizer, rows: list[dict], own_len_by_ctx: dict[int, int]) -> list[dict]:
    """Per-row truncation pre-computation for EXT pairs (mirrors the parent's
    `precompute_rows` minus the prefix-only `in_common_valid`; own_len comes
    from own_len_ext keyed by context id — a missing key is fail-loud)."""
    out: list[dict] = []
    for r in rows:
        i = r["context_id"]
        if i not in own_len_by_ctx:
            raise RuntimeError(f"own_len_ext has no entry for context {i} — owngen incomplete")
        own = own_len_by_ctx[i]
        row = {
            "context_id": i,
            "persona_idx": r["persona_idx"],
            "arms": list(r["arms"]),
            "own_len": own,
            "cap_hit": bool(r["cap_hit"]),
            "max_tokens": r["max_tokens"],
            "validity": r["validity"],
            "filled": bool(r["filled"]),
            "gen_stage": r["gen_stage"],
        }
        if not r["filled"] or not r["answer_text"]:
            row.update(
                skip_reason="not_filled",
                pair_len=None,
                trunc_len=None,
                truncated=False,
                dropped_tokens=0,
                expected_span=0,
            )
        else:
            prompt_len, full_len = CAP.template_span_length(
                tokenizer, r["question"], r["answer_text"]
            )
            ext = full_len - prompt_len
            if ext < 1:
                row.update(
                    skip_reason="empty_span",
                    pair_len=int(ext),
                    trunc_len=None,
                    truncated=False,
                    dropped_tokens=0,
                    expected_span=0,
                )
            else:
                trunc = min(own, ext) if own > 0 else ext
                row.update(
                    skip_reason=None,
                    pair_len=int(ext),
                    trunc_len=int(trunc),
                    truncated=bool(own > 0 and ext > own),
                    dropped_tokens=int(ext - trunc),
                    expected_span=int(trunc),
                )
        out.append(row)
    return out


def unit_fingerprint(
    kind: str,
    context_ids: list[int],
    answer_shas: list[str],
    own_source_sha: str,
    batch_size: int,
) -> dict:
    """Input fingerprint per checkpoint unit (plan section 4.3, #952 gate-5
    shape): sha over the block's pair ids + answer-text shas + own_len source
    sha + trunc convention id, plus the capture config. The code SHA is
    RECORDED in the sidecar but sits OUTSIDE this equality key."""
    return {
        "kind": kind,
        "n_rows": len(context_ids),
        "ids_sha256": _sha256_json(context_ids),
        "answers_sha256": _sha256_json(answer_shas),
        "own_len_source_sha256": own_source_sha,
        "trunc_convention": TRUNC_CONVENTION_EXT if kind == "pairs" else CX_CONVENTION,
        "model": DEFAULT_MODEL,
        "n_layers": EXPECTED_LAYERS,
        "hidden": EXPECTED_HIDDEN,
        "batch_size": batch_size,
        "dtype": "bfloat16-capture/fp32-store",
    }


def unit_done(store_dir: pathlib.Path, name: str, fingerprint: dict) -> bool:
    """Resume predicate: sidecar fingerprint EQUALITY (never bare existence).

    A mismatched or partial sidecar routes the unit to RECAPTURE (plan section
    4.3: 'a mismatched sidecar routes the block to recapture') — logged loud,
    never a silent skip and never a silent reuse across regimes.
    """
    tensor_path = store_dir / f"{name}.pt"
    sidecar_path = store_dir / f"{name}.done.json"
    if not sidecar_path.exists():
        return False
    if not tensor_path.exists():
        logger.warning("[capext] %s: sidecar without tensor — recapturing", name)
        return False
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("fingerprint") != fingerprint:
        logger.warning(
            "[capext] %s: fingerprint mismatch (stale regime) — RECAPTURING, not skipping", name
        )
        return False
    return True


def save_unit(
    store_dir: pathlib.Path,
    name: str,
    payload: dict,
    fingerprint: dict,
    elapsed_s: float,
    n_skipped: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Atomic per-unit checkpoint (tensor + fingerprint sidecar)."""
    store_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = store_dir / f"{name}.pt"
    sidecar_path = store_dir / f"{name}.done.json"
    tmp = tensor_path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp))  # plain fp32 torch.save, compression OFF (Xet)
    os.replace(tmp, tensor_path)
    write_json(
        sidecar_path,
        {
            "fingerprint": fingerprint,
            "code_sha": as_metadata_dict(git_provenance())["git_commit"],
            "n_rows": fingerprint["n_rows"],
            "n_skipped": n_skipped,
            "elapsed_s": elapsed_s,
            "ts": _utc_now(),
        },
    )
    return tensor_path, sidecar_path


def compute_capture_projection(
    timed_elapsed_s: float,
    n_timed: int,
    n_remaining: int,
    planned_wall_h: float,
    factor: float = CAPTURE_WALL_ABORT_FACTOR,
) -> dict:
    """Gate B arithmetic: projected wall from the timed pilot batches."""
    assert n_timed > 0 and timed_elapsed_s >= 0
    per_row_s = timed_elapsed_s / n_timed
    projected_h = per_row_s * n_remaining / 3600.0
    threshold_h = factor * planned_wall_h
    return {
        "n_timed_rows": n_timed,
        "timed_elapsed_s": timed_elapsed_s,
        "per_row_s": per_row_s,
        "n_remaining_rows": n_remaining,
        "projected_wall_h": projected_h,
        "planned_wall_h": planned_wall_h,
        "abort_threshold_h": threshold_h,
        "measured_basis_reference_s_per_row": MEASURED_PILOT_BASIS_S_PER_ROW,
        "abort": projected_h > threshold_h,
    }


class BlockUploader:
    """One-worker concurrent uploader: bulk `upload_folder` commit per unit,
    fail-loud (exceptions surface into the capture loop, never fire-and-forget)."""

    def __init__(self, local_dir: pathlib.Path, path_in_repo: str):
        self.local_dir = local_dir
        self.path_in_repo = path_in_repo
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.futures: list[concurrent.futures.Future] = []

    def _upload(self, names: list[str]) -> None:
        url = _upload_folder_filtered(
            local_dir=self.local_dir,
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo=self.path_in_repo,
            allow_patterns=list(names),
            expected_repo_paths=[f"{self.path_in_repo}/{n}" for n in names],
        )
        if not url:
            raise RuntimeError(
                f"block upload of {names} to {DATA_REPO}/{self.path_in_repo} failed or "
                "verified incomplete"
            )
        _require_canonical_upload(url, f"{DATA_REPO}/{self.path_in_repo}")
        logger.info("[pstore-ext] uploaded %s", names)

    def submit(self, names: list[str]) -> None:
        self.poll_raise()
        self.futures.append(self.pool.submit(self._upload, names))

    def poll_raise(self) -> None:
        for f in self.futures:
            if f.done():
                f.result()  # re-raises any upload failure loudly

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for f in self.futures:
            f.result()


# ── Phase: P0-ext ────────────────────────────────────────────────────────────


def phase_p0ext(args, layout: Layout) -> None:
    log_phase("p0ext_stage")
    assert_out_root_headroom(layout.out_root, layout.headroom_gb("p0ext"), phase="p0ext")
    metadata = build_metadata(layout, "p0ext")

    banked = stage_banked_inputs(layout)
    ext_base, ext_sentinel, ext_rev = stage_ext_gen_inputs(
        layout, args.ext_prefix, args.ext_revision, args.ext_local_dir
    )

    # Entry idempotency (r1 concern phase-entry-idempotency): a completed P0-ext
    # at the SAME staged input revisions returns before any gate recompute /
    # tokenizer / 7B model load — an all-done retry costs staging checks only.
    report_path = layout.eval_dir / P0_REPORT
    if not args.pre_gpu_check and report_path.exists():
        prior = json.loads(report_path.read_text())
        prior_meta = prior.get("metadata", {})
        if (
            prior.get("pass")
            and prior_meta.get("ext_gen_revision") == ext_rev
            and prior_meta.get("banked_gen_revision") == banked["gen_rev"]
        ):
            log_phase("p0ext_resume_complete")
            logger.info(
                "P0-ext already complete at ext_gen rev %s (%s) — nothing to do",
                ext_rev,
                report_path,
            )
            return
        logger.warning("[p0ext] prior report stale (revision/pass mismatch) — rerunning gates")

    ext_by_persona = load_ext_pair_rows(ext_base, N_PREFIX, layout.n_total)
    staged_assignment_obj = json.loads((ext_base / "assignment_ext.json").read_text())
    banked_by_persona = CAP.load_pair_rows(banked["gen_paths"], N_CONTEXTS_FULL)
    metadata["ext_gen_revision"] = ext_rev
    metadata["banked_gen_revision"] = banked["gen_rev"]

    # Gate (a) BEFORE mask construction (plan ordering).
    log_phase("p0ext_gates")
    gate_a = gate_a_prompt_integrity(ext_by_persona, banked["roster_obj"], layout.eval_dir)
    gate_b = gate_b_assignment(
        staged_assignment_obj, banked["banked_assignment_obj"], layout.n_total, layout.eval_dir
    )
    gate_c = gate_c_max_tokens(ext_by_persona, layout.eval_dir)
    bridge_ids = load_bridge_mask_ids()
    mask_obj = build_masks(banked_by_persona, ext_by_persona, bridge_ids, layout, metadata)
    write_json(layout.eval_dir / MASK_FILE, mask_obj)

    questions_by_id = {i: q for i, q in enumerate(banked["questions_prefix"])}
    for rows in ext_by_persona.values():
        for r in rows:
            questions_by_id[r["context_id"]] = r["question"]
    gate_e = gate_e_duplicates(questions_by_id, layout)

    # Parent per-persona refusal reference (fig_ext7 overlay input): measured on
    # the banked parent rows this phase already parsed — persisted so P-Analysis
    # consumes an artifact instead of re-deriving from raw completions.
    parent_refusal = {
        str(p): (
            sum(1 for r in rows if r.get("validity") == "refusal") / len(rows) if rows else 0.0
        )
        for p, rows in sorted(banked_by_persona.items())
    }
    write_json(
        layout.eval_dir / "parent_refusal_by_persona.json",
        {"metadata": metadata, "refusal_fraction_by_persona": parent_refusal},
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    probe_g = probe_g_span_unit(
        tokenizer,
        banked["b2_records"],
        banked["span_d"],
        banked["a_prime_texts"],
        banked["common_valid_ids"],
        layout.eval_dir,
    )

    report = {
        "metadata": metadata,
        "ext_sentinel_n_pairs": ext_sentinel.get("n_pairs"),
        "gate_a_prompt_integrity": gate_a,
        "gate_b_assignment": gate_b,
        "gate_c_max_tokens": gate_c,
        "gate_d_masks": {
            "integrity_gate": mask_obj["integrity_gate"],
            "ext_arm_stats": mask_obj["ext_arm_stats"],
            "bridge_n": mask_obj["bridge"]["n_mask"],
            "rung_sizes": {r: v["n_mask"] for r, v in mask_obj["rungs"].items()},
        },
        "gate_e_duplicates": gate_e,
        "probe_g_span_unit": probe_g,
        "mask_crosscheck_phase1": banked["mask_crosscheck"],
    }

    if args.pre_gpu_check:
        report["pass"] = False
        report["pre_gpu_check_only"] = True
        write_json(layout.eval_dir / "p0_ext_pregpu_check.json", report)
        log_phase("p0ext_pregpu_ok")
        logger.info("Pre-GPU check PASS (gates a-e + probe g); probe (f) needs the GPU phase")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("p0ext probe (f) requires CUDA; use --pre-gpu-check for the CPU part")
    from transformers import AutoModelForCausalLM

    log_phase("p0ext_model_load")
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    assert model.config.hidden_size == EXPECTED_HIDDEN
    assert model.config.num_hidden_layers == EXPECTED_LAYERS
    log_phase("p0ext_probe_f")
    probe_f = probe_f_capture_parity(
        model,
        tokenizer,
        banked["questions_prefix"],
        banked["pass_b_path"],
        layout.eval_dir,
        args.batch_size,
    )
    report["probe_f_capture_parity"] = probe_f
    report["pass"] = True
    write_json(layout.eval_dir / P0_REPORT, report)
    write_sentinel(
        layout.sentinel_dir() / "issue-823-extladder-p0ext-done.json",
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P0-ext gates (a)-(g) PASS (origin-ladder-more-contexts)",
            "phase": "p0ext",
            "complete": True,
            "smoke": layout.smoke,
            "report": str(layout.eval_dir / P0_REPORT),
            "ts": time.time(),
        },
    )
    log_phase("p0ext_done")
    logger.info("P0-ext PASS: %s", layout.eval_dir / P0_REPORT)


# ── Phase: P-OwnGen ──────────────────────────────────────────────────────────


def _reap_vllm(llm) -> None:
    """Reap the vLLM v1 EngineCore before continuing in-process (gotchas.md).

    Getattr-guarded shutdown of the engine-core subprocess + the distributed
    process group so interpreter finalization cannot deadlock on surviving
    workers (#1739/#2149) and the post-generation CPU work (own_len + upload)
    runs against a clean process tree.
    """
    import gc

    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
    else:
        executor = getattr(engine, "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)  # subprocess teardown is async


def _require_p0_pass(layout: Layout) -> None:
    p = layout.eval_dir / P0_REPORT
    if not p.exists():
        raise RuntimeError(f"{p} missing — run --phase p0ext first (gates are binding)")
    if not json.loads(p.read_text()).get("pass"):
        raise RuntimeError(f"{p} has pass!=True — P0-ext gates did not complete")


def phase_owngen(args, layout: Layout) -> None:
    _require_p0_pass(layout)
    log_phase("owngen_stage")
    assert_out_root_headroom(layout.out_root, layout.headroom_gb("owngen"), phase="owngen")
    metadata = build_metadata(layout, "owngen")
    ext_base, _sentinel, ext_rev = stage_ext_gen_inputs(
        layout, args.ext_prefix, args.ext_revision, args.ext_local_dir
    )
    metadata["ext_gen_revision"] = ext_rev
    ext_by_persona = load_ext_pair_rows(ext_base, N_PREFIX, layout.n_total)
    # persona 0 carries the k=1 pair for EVERY extension context => full question list.
    ctx_q = [(r["context_id"], r["question"]) for r in ext_by_persona[0]]
    assert len(ctx_q) == layout.n_ext, f"persona00 rows {len(ctx_q)} != n_ext {layout.n_ext}"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    prompt_budget = OWN_MAX_MODEL_LEN - OWN_GEN_MAX_TOKENS - 8
    prompts: list[tuple[int, str, str, int]] = []  # (ctx, question, template_text, prompt_len)
    for i, q in ctx_q:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        prompts.append((i, q, text, len(ids)))
    suffix = tokenizer.decode(
        tokenizer(prompts[0][2], return_tensors=None, add_special_tokens=False)["input_ids"][-3:]
    )
    assert suffix == GENERATION_SUFFIX, f"owngen template suffix {suffix!r}"

    chunk_size = args.own_chunk_size
    chunks = [prompts[c : c + chunk_size] for c in range(0, len(prompts), chunk_size)]
    fingerprints = [
        own_chunk_fingerprint([i for i, _q, _t, _l in ch], [q for _i, q, _t, _l in ch])
        for ch in chunks
    ]
    pending = [
        c for c in range(len(chunks)) if not own_chunk_done(layout.own_dir, c, fingerprints[c])
    ]
    logger.info("[owngen] %d chunks total, %d pending", len(chunks), len(pending))

    if pending:
        log_phase("owngen_generate")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=DEFAULT_MODEL,
            dtype="bfloat16",
            seed=OWN_GEN_SEED,
            max_model_len=OWN_MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
        )
        sp = SamplingParams(
            temperature=OWN_GEN_TEMPERATURE,
            top_p=OWN_GEN_TOP_P,
            seed=OWN_GEN_SEED,
            max_tokens=OWN_GEN_MAX_TOKENS,
        )
        layout.own_dir.mkdir(parents=True, exist_ok=True)
        for k, c in enumerate(pending):
            t0 = time.monotonic()
            ch = chunks[c]
            gen_idx = [j for j, (_i, _q, _t, plen) in enumerate(ch) if plen <= prompt_budget]
            outputs = (
                llm.generate([ch[j][2] for j in gen_idx], sp, use_tqdm=False) if gen_idx else []
            )
            rows = []
            assert len(outputs) == len(gen_idx), (len(outputs), len(gen_idx))
            out_by_pos = dict(zip(gen_idx, outputs))
            for j, (i, q, _t, plen) in enumerate(ch):
                if j in out_by_pos:
                    o = out_by_pos[j].outputs[0]
                    rows.append(
                        {
                            "context_id": i,
                            "question": q,
                            "own_text": o.text,
                            "finish_reason": o.finish_reason,
                            "skipped_reason": None,
                        }
                    )
                else:
                    rows.append(
                        {
                            "context_id": i,
                            "question": q,
                            "own_text": "",
                            "finish_reason": None,
                            "skipped_reason": f"prompt_too_long({plen}>{prompt_budget})",
                        }
                    )
            EXTGEN._write_jsonl(layout.own_dir / f"own_chunk{c:03d}.jsonl", rows)
            write_json(
                layout.own_dir / f"own_chunk{c:03d}.meta.json",
                {
                    "fingerprint": fingerprints[c],
                    "code_sha": metadata["git_commit"],
                    "n_rows": len(rows),
                    "n_skipped_too_long": len(ch) - len(gen_idx),
                    "ts": _utc_now(),
                },
            )
            logger.info(
                "[owngen] unit %d/%d chunk=%03d n=%d elapsed=%.1fs",
                k + 1,
                len(pending),
                c,
                len(rows),
                time.monotonic() - t0,
            )
        _reap_vllm(llm)

    # own_len via the BANKED LADDER STORE's own-side rule: the TEMPLATE-DIFF
    # span of the own rollout (`own_len_ext_span`), matching the reused
    # 14,996-pair prefix store the fits driver POOLS these tensors with (its
    # producer truncated with own = span_d["a_prime"][i], itself a
    # template-diff span). NOT the parent phase-3 BARE rule
    # (`own_span_length`) — probe (g) validates THAT producer separately on
    # the banked b2 arm; mixing the two rules across the pooled stores was
    # the v18 review bounce.
    log_phase("owngen_ownlen")
    own_len: dict[str, int] = {}
    n_zero = 0
    n_cap_hit = 0
    n_rows = 0
    for c in range(len(chunks)):
        for r in read_jsonl_manifest_first(layout.own_dir, f"own_chunk{c:03d}"):
            n_rows += 1
            if r["finish_reason"] == "length":
                n_cap_hit += 1
            if not r["own_text"]:
                own_len[str(r["context_id"])] = 0
                n_zero += 1
                continue
            span = own_len_ext_span(tokenizer, r["question"], r["own_text"])
            if span < 1:
                span = 0
                n_zero += 1
            own_len[str(r["context_id"])] = int(span)
    assert n_rows == layout.n_ext, f"own rows {n_rows} != n_ext {layout.n_ext}"
    cap_hit_fraction = n_cap_hit / n_rows if n_rows else 0.0
    write_json(
        layout.own_dir / "own_len_ext.json",
        {
            "metadata": metadata,
            "recipe": fingerprints[0]["recipe"] if fingerprints else None,
            "n_contexts": n_rows,
            "n_zero_own_len": n_zero,
            # Cap-hit REPORTED; no regen by design: the recipe is plan-pinned to
            # the parent own-answer arm (#779 pass-B, max_tokens 1024) for
            # own_len parity — a longer own cap would change the truncation
            # convention vs the banked prefix store (recipe fidelity).
            "cap_hit_fraction": cap_hit_fraction,
            "own_len": own_len,
        },
    )
    if cap_hit_fraction > 0.02:
        logger.warning(
            "[owngen] cap-hit fraction %.3f > 2%% — REPORTED (plan-pinned recipe, no regen; "
            "own_len truncation only shortens spans, banked-ladder-store parity)",
            cap_hit_fraction,
        )

    # Persist-by-default: rollout TEXTS + own_len_ext.json to HF BEFORE any
    # downstream phase consumes the lengths.
    log_phase("owngen_upload")
    upload_paths = EXTGEN.shard_large_jsonl_for_upload(
        [layout.own_dir / f"own_chunk{c:03d}.jsonl" for c in range(len(chunks))]
        + [layout.own_dir / f"own_chunk{c:03d}.meta.json" for c in range(len(chunks))]
        + [layout.own_dir / "own_len_ext.json"]
    )
    upload_rel = sorted(p.relative_to(layout.own_dir).as_posix() for p in upload_paths)
    own_sentinel = {
        "phase": "owngen",
        "complete": True,
        "smoke": layout.smoke,
        "metadata": metadata,
        "n_contexts": n_rows,
        "n_zero_own_len": n_zero,
        "cap_hit_fraction": cap_hit_fraction,
        "files_sha256": {rel: _sha256_file(layout.own_dir / rel) for rel in upload_rel},
    }
    path_in_repo = layout.hf_path(layout.own_subpath)
    url = _upload_folder_filtered(
        local_dir=layout.own_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=upload_rel,
        expected_repo_paths=[f"{path_in_repo}/{rel}" for rel in upload_rel],
    )
    if not url:
        raise RuntimeError(
            f"own-rollout upload to {DATA_REPO}/{path_in_repo} failed or verified incomplete — "
            "refusing to report owngen complete (capture must not consume unpersisted lengths)"
        )
    _require_canonical_upload(url, f"{DATA_REPO}/{path_in_repo}")
    # Local complete=True sentinel is written ONLY after the data upload
    # verified (r1 concern sentinel-before-upload; storeext ordering pattern):
    # a failed upload must not leave a sentinel _require_own_complete accepts.
    write_json(layout.own_dir / OWNGEN_SENTINEL, own_sentinel)
    sent_url = _upload_folder_filtered(
        local_dir=layout.own_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=[OWNGEN_SENTINEL],
        expected_repo_paths=[f"{path_in_repo}/{OWNGEN_SENTINEL}"],
    )
    if not sent_url:
        raise RuntimeError(
            f"owngen sentinel upload to {DATA_REPO}/{path_in_repo} failed or verified incomplete"
        )
    write_sentinel(
        layout.sentinel_dir() / "issue-823-extladder-owngen-done.json",
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P-OwnGen complete: own rollouts + own_len_ext persisted",
            "phase": "owngen",
            "complete": True,
            "smoke": layout.smoke,
            "n_contexts": n_rows,
            "cap_hit_fraction": cap_hit_fraction,
            "hf_path_in_repo": path_in_repo,
            "ts": time.time(),
        },
    )
    log_phase("owngen_done")
    logger.info("P-OwnGen complete: %d files at %s", len(upload_rel), url)


# ── Phase: P-Cap-ext (+ interleaved P-Store-ext uploads) ─────────────────────


def _require_own_complete(layout: Layout) -> dict[int, int]:
    sp = layout.own_dir / OWNGEN_SENTINEL
    if not sp.exists():
        raise RuntimeError(f"{sp} missing — run --phase owngen first (R3: text before reduction)")
    sentinel = json.loads(sp.read_text())
    if not sentinel.get("complete"):
        raise RuntimeError(f"{sp} has complete!=True")
    ol_path = layout.own_dir / "own_len_ext.json"
    want = sentinel["files_sha256"].get("own_len_ext.json")
    if want and _sha256_file(ol_path) != want:
        raise RuntimeError(f"{ol_path} sha mismatch vs the owngen sentinel")
    obj = json.loads(ol_path.read_text())
    return {int(k): int(v) for k, v in obj["own_len"].items()}


def enumerate_capture_units(
    ext_by_persona: dict[int, list[dict]],
    pre_by_persona: dict[int, list[dict]],
    ctx_questions: list[tuple[int, str]],
    own_source_sha: str,
    batch_size: int,
    cx_block: int,
    pair_block: int,
) -> list[dict]:
    """Work units: cx blocks over extension contexts + per-(persona x block)
    pair units, each with its input fingerprint."""
    units: list[dict] = []
    for b in range(0, len(ctx_questions), cx_block):
        chunk = ctx_questions[b : b + cx_block]
        ids = [i for i, _q in chunk]
        units.append(
            {
                "name": f"cx_ext_block{b // cx_block}",
                "kind": "cx",
                "context_ids": ids,
                "questions": [q for _i, q in chunk],
                "fingerprint": unit_fingerprint("cx", ids, [], own_source_sha, batch_size),
            }
        )
    for p in sorted(ext_by_persona):
        rows, pre = ext_by_persona[p], pre_by_persona[p]
        if not rows:
            continue
        for b in range(0, len(rows), pair_block):
            rows_b = rows[b : b + pair_block]
            pre_b = pre[b : b + pair_block]
            ids = [r["context_id"] for r in rows_b]
            answer_shas = [
                _sha256_text(r["answer_text"]) if r.get("answer_text") else "" for r in rows_b
            ]
            units.append(
                {
                    "name": f"v_pairs_ext_p{p:02d}_block{b // pair_block}",
                    "kind": "pairs",
                    "persona": p,
                    "rows": rows_b,
                    "pre": pre_b,
                    "context_ids": ids,
                    "fingerprint": unit_fingerprint(
                        "pairs", ids, answer_shas, own_source_sha, batch_size
                    ),
                }
            )
    return units


def run_gate_b(
    model,
    tokenizer,
    pending_units: list[dict],
    batch_size: int,
    planned_wall_h: float,
    eval_dir: pathlib.Path,
) -> dict:
    """Gate B in-run pilot: warmup 1 batch + time 2 batches at production
    shape; projected wall > 2x the section-9 row => designed abort (rc 23)."""
    pair_units = [u for u in pending_units if u["kind"] == "pairs"]
    unit = pair_units[0] if pair_units else pending_units[0]
    n_need = 3 * batch_size
    if unit["kind"] == "pairs":
        live = [(r, pr) for r, pr in zip(unit["rows"], unit["pre"]) if pr["skip_reason"] is None][
            :n_need
        ]
        if not live:
            raise RuntimeError("gate B pilot found zero capturable rows in the first unit")
        warm, timed = live[:batch_size], live[batch_size:] or live

        def _run(subset):
            _tf_extract_arm(
                model,
                tokenizer,
                [r["question"] for r, _ in subset],
                [r["answer_text"] for r, _ in subset],
                list(range(EXPECTED_LAYERS)),
                "gate_b_pilot",
                a_prime_lengths=[pr["own_len"] for _, pr in subset],
                batch_size=batch_size,
            )
    else:
        qs = unit["questions"][:n_need]
        warm, timed = qs[:batch_size], qs[batch_size:] or qs

        def _run(subset):
            capture_cx(model, tokenizer, list(subset), batch_size)

    if warm is not timed and warm:
        _run(warm)
    t0 = time.monotonic()
    _run(timed)
    elapsed = time.monotonic() - t0
    n_remaining = sum(len(u["context_ids"]) for u in pending_units)
    report = compute_capture_projection(elapsed, len(timed), n_remaining, planned_wall_h)
    logger.info("[gate-b] %s", json.dumps(report))
    if report["abort"]:
        write_json(
            eval_dir / "gate_b_abort_report.json",
            {
                "kill_criterion": "capture-wall (plan section 7 Gate B)",
                "verdict": "DESIGNED-ABORT",
                "rc": RC_GATE_B_WALL,
                "pilot": report,
                "ts": _utc_now(),
            },
        )
        logger.error(
            "DESIGNED ABORT rc=%d: projected capture wall %.2fh > %.2fh",
            RC_GATE_B_WALL,
            report["projected_wall_h"],
            report["abort_threshold_h"],
        )
        raise DesignedHalt(RC_GATE_B_WALL)
    return report


def phase_capext(args, layout: Layout) -> None:
    _require_p0_pass(layout)
    own_len_by_ctx = _require_own_complete(layout)
    log_phase("capext_stage")
    assert_out_root_headroom(layout.out_root, layout.headroom_gb("capext"), phase="capext")
    metadata = build_metadata(layout, "capext")
    ext_base, _sentinel, ext_rev = stage_ext_gen_inputs(
        layout, args.ext_prefix, args.ext_revision, args.ext_local_dir
    )
    metadata["ext_gen_revision"] = ext_rev
    ext_by_persona = load_ext_pair_rows(ext_base, N_PREFIX, layout.n_total)
    own_source_sha = _sha256_file(layout.own_dir / "own_len_ext.json")
    metadata["own_len_source_sha256"] = own_source_sha

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    log_phase("capext_precompute")
    pre_by_persona = {
        p: precompute_ext_rows(tokenizer, rows, own_len_by_ctx)
        for p, rows in sorted(ext_by_persona.items())
    }
    ctx_questions = [(r["context_id"], r["question"]) for r in ext_by_persona[0]]

    units = enumerate_capture_units(
        ext_by_persona,
        pre_by_persona,
        ctx_questions,
        own_source_sha,
        args.batch_size,
        args.cx_block_size,
        args.pair_block_size,
    )
    pending = [u for u in units if not unit_done(layout.store_dir, u["name"], u["fingerprint"])]
    logger.info("[capext] %d units total, %d pending", len(units), len(pending))

    # Entry idempotency (r1 concern phase-entry-idempotency): the 7B model load
    # + CUDA requirement are gated on there being any pending capture unit — an
    # all-done retry runs upload/verify only, on any host.
    model = None
    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError("P-Cap-ext requires CUDA (126k bf16 7B forwards)")
        log_phase("capext_model_load")
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        model.eval()
        assert model.config.hidden_size == EXPECTED_HIDDEN
        assert model.config.num_hidden_layers == EXPECTED_LAYERS
    else:
        logger.info("[capext] all units checkpointed — skipping 7B model load")

    uploader = BlockUploader(layout.store_dir, layout.hf_path(layout.store_subpath))
    try:
        if pending:
            log_phase("capext_pilot")
            gate_b_report = run_gate_b(
                model, tokenizer, pending, args.batch_size, args.planned_wall_hours, layout.eval_dir
            )
        else:
            gate_b_report = None
            logger.info("[capext] all units checkpointed — skipping pilot (upload/verify only)")

        for k, u in enumerate(pending):
            t0 = time.monotonic()
            log_phase(f"capext_{u['name']}")
            if u["kind"] == "cx":
                v = capture_cx(model, tokenizer, u["questions"], args.batch_size)
                payload = {
                    "cx": torch.from_numpy(v),
                    "context_ids": torch.tensor(u["context_ids"], dtype=torch.long),
                }
                n_skipped = 0
            else:
                v_s, span_lens, mean_logps, ctx_ids = CAP.capture_persona_group(
                    model, tokenizer, u["persona"], u["rows"], u["pre"], args.batch_size
                )
                payload = {
                    "v": torch.from_numpy(v_s),
                    "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                    "span_lengths": torch.tensor(span_lens, dtype=torch.long),
                    "mean_logp": torch.tensor(mean_logps, dtype=torch.float64),
                    "persona_idx": u["persona"],
                }
                n_skipped = int(sum(1 for s in span_lens if s == 0))
            elapsed = time.monotonic() - t0
            save_unit(layout.store_dir, u["name"], payload, u["fingerprint"], elapsed, n_skipped)
            uploader.submit([f"{u['name']}.pt", f"{u['name']}.done.json"])
            logger.info(
                "[capext] unit %d/%d %s n=%d elapsed=%.1fs",
                k + 1,
                len(pending),
                u["name"],
                len(u["context_ids"]),
                elapsed,
            )
            del payload
            torch.cuda.empty_cache()

        # pairs_index_ext (rows as jsonl for line-sharding) + digest.
        log_phase("capext_index")
        index_rows = [row for p in sorted(pre_by_persona) for row in pre_by_persona[p]]
        EXTGEN._write_jsonl(layout.store_dir / "pairs_index_ext_rows.jsonl", index_rows)
        unit_table = {}
        for u in units:
            sidecar = json.loads((layout.store_dir / f"{u['name']}.done.json").read_text())
            unit_table[u["name"]] = {
                "kind": u["kind"],
                "n_rows": sidecar["n_rows"],
                "n_skipped": sidecar.get("n_skipped", 0),
            }
        skip_counts = {
            str(p): {
                "not_filled": sum(1 for r in rows if r["skip_reason"] == "not_filled"),
                "empty_span": sum(1 for r in rows if r["skip_reason"] == "empty_span"),
            }
            for p, rows in sorted(pre_by_persona.items())
        }
        write_json(
            layout.store_dir / "pairs_index_ext.json",
            {
                "metadata": metadata,
                "units": unit_table,
                "skip_counts_by_persona": skip_counts,
                "rows_file": "pairs_index_ext_rows.jsonl",
                "n_rows": len(index_rows),
            },
        )
        digest = {
            "metadata": metadata,
            "gate_b_pilot": gate_b_report,
            "units": unit_table,
            "skip_counts_by_persona": skip_counts,
            "n_pairs": sum(len(r) for r in ext_by_persona.values()),
            "n_cx_contexts": len(ctx_questions),
            "resumed_units": sorted({u["name"] for u in units} - {u["name"] for u in pending}),
        }
        write_json(layout.store_dir / "capture_digest_ext.json", digest)
        write_json(layout.eval_dir / "capture_digest_ext.json", digest)

        index_uploads = EXTGEN.shard_large_jsonl_for_upload(
            [layout.store_dir / "pairs_index_ext_rows.jsonl"]
        )
        index_names = sorted(
            {p.name for p in index_uploads} | {"pairs_index_ext.json", "capture_digest_ext.json"}
        )
        uploader.submit(index_names)
    finally:
        # Fail-loud close (any block-upload failure raises here at the latest),
        # but never MASK an in-flight exception with the teardown's own raise —
        # the finally-raise guard (#1947; gotchas.md).
        inner_live = sys.exc_info()[0] is not None
        if inner_live:
            try:
                uploader.close()
            except Exception:
                logger.exception("[capext] uploader.close() failed during unwind (original wins)")
        else:
            uploader.close()

    write_json(
        layout.store_dir / CAPTURE_SENTINEL,
        {
            "phase": "capext",
            "complete": True,
            "smoke": layout.smoke,
            "metadata": metadata,
            "n_units": len(units),
            "unit_names": sorted(u["name"] for u in units),
        },
    )
    write_sentinel(
        layout.sentinel_dir() / "issue-823-extladder-capext-done.json",
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P-Cap-ext complete (blocks uploaded interleaved)",
            "phase": "capext",
            "complete": True,
            "smoke": layout.smoke,
            "n_units": len(units),
            "ts": time.time(),
        },
    )
    log_phase("capext_done")
    logger.info("P-Cap-ext complete: %d units (%d captured this run)", len(units), len(pending))


# ── Phase: P-Store-ext (final verify + sentinel) ─────────────────────────────


def expected_store_files(store_dir: pathlib.Path) -> list[str]:
    """The store's local shard list (upload-eligible names, manifest-first for
    the sharded rows file — the original oversized jsonl stays local-only)."""
    names: set[str] = set()
    for p in sorted(store_dir.iterdir()):
        if p.suffix == ".pt" or p.name.endswith(".done.json"):
            names.add(p.name)
    for extra in ("pairs_index_ext.json", "capture_digest_ext.json", CAPTURE_SENTINEL):
        if (store_dir / extra).exists():
            names.add(extra)
    rows_manifest = store_dir / "pairs_index_ext_rows.manifest.json"
    if rows_manifest.exists():
        parts, _ = _parse_shard_manifest(rows_manifest.read_text(), what=str(rows_manifest))
        names.add(rows_manifest.name)
        names.update(parts)
    elif (store_dir / "pairs_index_ext_rows.jsonl").exists():
        names.add("pairs_index_ext_rows.jsonl")
    return sorted(names)


def phase_storeext(args, layout: Layout) -> None:
    cs = layout.store_dir / CAPTURE_SENTINEL
    if not cs.exists() or not json.loads(cs.read_text()).get("complete"):
        raise RuntimeError(f"{cs} missing/incomplete — run --phase capext first")
    log_phase("storeext_verify")
    assert_out_root_headroom(layout.out_root, layout.headroom_gb("storeext"), phase="storeext")
    metadata = build_metadata(layout, "storeext")

    from huggingface_hub import HfApi

    api = HfApi()
    prefix = layout.hf_path(layout.store_subpath)
    local_names = expected_store_files(layout.store_dir)
    expected_paths = [f"{prefix}/{n}" for n in local_names]
    missing = verify_repo_paths_uploaded(
        api, DATA_REPO, expected_paths, path_in_repo=prefix, repo_type="dataset"
    )
    if missing:
        logger.info("[storeext] %d store files missing on Hub — delta-uploading", len(missing))
        names = [m.removeprefix(f"{prefix}/") for m in missing]
        url = _upload_folder_filtered(
            local_dir=layout.store_dir,
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
            allow_patterns=names,
            expected_repo_paths=missing,
        )
        if not url:
            raise RuntimeError(f"delta upload of {len(names)} store files failed")
        _require_canonical_upload(url, f"{DATA_REPO}/{prefix}")
        still = verify_repo_paths_uploaded(
            api, DATA_REPO, expected_paths, path_in_repo=prefix, repo_type="dataset"
        )
        if still:
            _halt(
                RC_STAGING_INTEGRITY,
                layout.eval_dir / "ext_store_verify_report.json",
                {"reason": "still_missing_after_delta", "missing": still[:50]},
                f"store verify: {len(still)} expected files still missing after delta upload",
            )

    # NAME-SET diff: scoped listing vs the local shard list (a matching count
    # is not a matching set; extras are prior-regime residue => fail loud).
    remote = retry_transient(
        lambda: list_hf_files_under_path(api, DATA_REPO, prefix, repo_type="dataset"),
        what=f"list_hf_files_under_path({prefix})",
    )
    remote_names = {p.removeprefix(f"{prefix}/") for p in remote}
    extras = sorted(remote_names - set(local_names) - {STORE_SENTINEL})
    missing_remote = sorted(set(local_names) - remote_names)
    if extras or missing_remote:
        _halt(
            RC_STAGING_INTEGRITY,
            layout.eval_dir / "ext_store_verify_report.json",
            {
                "reason": "name_set_diff",
                "n_local": len(local_names),
                "n_remote": len(remote_names),
                "extras_on_hub": extras[:50],
                "missing_on_hub": missing_remote[:50],
            },
            f"store NAME-SET diff FAILED: {len(extras)} extras / {len(missing_remote)} missing "
            f"under {prefix}",
        )

    # Gates mirror (mask_ext.json + p0 report + capture digest -> HF).
    log_phase("storeext_mirror")
    mirror_names = [MASK_FILE, P0_REPORT, "capture_digest_ext.json"]
    for n in mirror_names:
        if not (layout.eval_dir / n).exists():
            raise RuntimeError(f"gates mirror: {layout.eval_dir / n} missing")
    gates_prefix = layout.hf_path(layout.gates_subpath)
    url = _upload_folder_filtered(
        local_dir=layout.eval_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=gates_prefix,
        allow_patterns=mirror_names,
        expected_repo_paths=[f"{gates_prefix}/{n}" for n in mirror_names],
    )
    if not url:
        raise RuntimeError(f"gates mirror upload to {DATA_REPO}/{gates_prefix} failed")
    _require_canonical_upload(url, f"{DATA_REPO}/{gates_prefix}")

    # The fits driver REQUIRES this sentinel (plan section 4.3 P-Store-ext).
    store_sentinel = {
        "phase": "storeext",
        "complete": True,
        "smoke": layout.smoke,
        "metadata": metadata,
        "n_files": len(local_names),
        "name_set_sha256": _sha256_json(local_names),
        "hf_prefix": prefix,
    }
    write_json(layout.store_dir / STORE_SENTINEL, store_sentinel)
    url = _upload_folder_filtered(
        local_dir=layout.store_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        allow_patterns=[STORE_SENTINEL],
        expected_repo_paths=[f"{prefix}/{STORE_SENTINEL}"],
    )
    if not url:
        raise RuntimeError("store-verified sentinel upload failed")
    _require_canonical_upload(url, f"{DATA_REPO}/{prefix}")

    write_sentinel(
        layout.sentinel_dir() / "issue-823-extladder-store-done.json",
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "P-Store-ext verified: NAME-SET diff PASS + _store_verified.json written",
            "phase": "storeext",
            "complete": True,
            "smoke": layout.smoke,
            "n_files": len(local_names),
            "hf_path_in_repo": prefix,
            "ts": time.time(),
        },
    )
    log_phase("done")
    logger.info("P-Store-ext PASS: %d files verified under %s", len(local_names), prefix)


# ── Import/signature check mode ───────────────────────────────────────────────


def run_import_check() -> None:
    """Execute every deferred import + signature-bind the GPU/API seams."""
    import inspect

    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    inspect.signature(_tf_extract_arm).bind(
        object(),
        object(),
        ["q"],
        ["a"],
        list(range(EXPECTED_LAYERS)),
        "ext_p00_b0",
        a_prime_lengths=[1],
        batch_size=8,
    )
    inspect.signature(CAP.capture_persona_group).bind(object(), object(), 0, [], [], 8)
    inspect.signature(_upload_folder_filtered).bind(
        local_dir=pathlib.Path("."),
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo="x/analysis_tensors/ext",
        allow_patterns=["*.pt"],
        expected_repo_paths=["x/analysis_tensors/ext/y.pt"],
    )
    inspect.signature(verify_repo_paths_uploaded).bind(
        object(), DATA_REPO, ["x/y.pt"], path_in_repo="x", repo_type="dataset"
    )
    inspect.signature(SamplingParams).bind(
        temperature=OWN_GEN_TEMPERATURE,
        top_p=OWN_GEN_TOP_P,
        seed=OWN_GEN_SEED,
        max_tokens=OWN_GEN_MAX_TOKENS,
    )
    print(
        json.dumps(
            {
                "import_check": "ok",
                "deferred_imports": [
                    "transformers.AutoModelForCausalLM",
                    "transformers.AutoTokenizer",
                    "vllm.LLM",
                    "vllm.SamplingParams",
                    "huggingface_hub.HfApi",
                ],
                "signature_bound": [
                    "_tf_extract_arm",
                    "capture_persona_group",
                    "_upload_folder_filtered",
                    "verify_repo_paths_uploaded",
                    "SamplingParams",
                ],
                "constants": {
                    "model": DEFAULT_MODEL,
                    "registered_ext_pairs": EXTGEN.REGISTERED_EXT_PAIRS,
                    "auto_model_cls": AutoModelForCausalLM.__name__,
                    "auto_tokenizer_cls": AutoTokenizer.__name__,
                    "llm_cls": LLM.__name__,
                    "hf_api_cls": HfApi.__name__,
                },
            }
        )
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P0-ext/P-OwnGen/P-Cap-ext/P-Store-ext for #823 origin-ladder-more-contexts: "
            "gates + 43k own rollouts + 126k-forward extension capture + verified store."
        )
    )
    parser.add_argument("--phase", choices=PHASE_NAMES, help="phase to run (separate processes)")
    parser.add_argument("--smoke", action="store_true", help="16-ctx/31-pair tiny-shape path")
    parser.add_argument(
        "--n-ext-contexts",
        type=int,
        default=None,
        help="extension context count override (smoke only; production pinned to 43000)",
    )
    parser.add_argument("--out-root", type=pathlib.Path, default=None, help="durable out-root")
    parser.add_argument(
        "--ext-prefix",
        default=HF_PREFIX,
        help="HF prefix holding the P-Gen-ext outputs (smoke run of ext-gen used the _smoke prefix)",
    )
    parser.add_argument(
        "--ext-revision",
        default=None,
        help="data-repo revision for the ext-gen fetch (default: main, resolved to ONE sha)",
    )
    parser.add_argument(
        "--ext-local-dir",
        type=pathlib.Path,
        default=None,
        help="read P-Gen-ext outputs from a local dir (still sentinel-verified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EPM_TF_BATCH_SIZE", "8")),
        help="TF batch size (same default in smoke and production — no smoke narrowing)",
    )
    parser.add_argument(
        "--planned-wall-hours",
        type=float,
        default=PLANNED_CAPTURE_WALL_H_EXT,
        help="plan section-9 P-Cap-ext wall row; Gate B aborts past 2x this (rc 23)",
    )
    parser.add_argument(
        "--own-chunk-size",
        type=int,
        default=OWN_CHUNK_SIZE,
        help="owngen chunk (smoke-only override)",
    )
    parser.add_argument(
        "--cx-block-size", type=int, default=CX_BLOCK_SIZE, help="cx block (smoke-only override)"
    )
    parser.add_argument(
        "--pair-block-size",
        type=int,
        default=PAIR_BLOCK_SIZE,
        help="pair block (smoke-only override)",
    )
    parser.add_argument(
        "--pre-gpu-check",
        action="store_true",
        help="p0ext only: run staging + gates (a)-(e) + probe (g), exit before model load",
    )
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="execute deferred imports + signature-bind the seams, then exit 0",
    )
    parser.add_argument("--list-rcs", action="store_true", help="print the designed rc table")
    parser.add_argument("--list-phases", action="store_true", help="print the phase registry")
    parser.add_argument("--list-arms", action="store_true", help="print the registered arm list")
    return parser


PHASES = {
    "p0ext": phase_p0ext,
    "owngen": phase_owngen,
    "capext": phase_capext,
    "storeext": phase_storeext,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.list_rcs:
        print(json.dumps(RC_TABLE))
        return
    if args.list_phases:
        print(json.dumps({"phases": list(PHASE_NAMES)}))
        return
    if args.list_arms:
        print(json.dumps({"ext_arms": list(EXT_ARMS), "n_personas": N_PERSONAS}))
        return
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        run_import_check()
        return
    if not args.phase:
        parser.error("--phase is required (or --import-check/--list-rcs/--list-phases)")

    if args.smoke:
        n_ext = args.n_ext_contexts if args.n_ext_contexts is not None else 16
        assert 1 <= n_ext <= N_EXT_FULL, "--n-ext-contexts out of range"
        if pathlib.Path("/workspace").exists():
            root = args.out_root or pathlib.Path("/workspace/eps/out/issue823_ladder_ext_smoke")
        else:
            root = args.out_root or pathlib.Path("/tmp/issue-823-ext-smoke/ladder_ext_capture")
    else:
        if args.n_ext_contexts is not None and args.n_ext_contexts != N_EXT_FULL:
            parser.error("--n-ext-contexts is smoke-only; production runs the full 43000")
        for flag, default in (
            ("own_chunk_size", OWN_CHUNK_SIZE),
            ("cx_block_size", CX_BLOCK_SIZE),
            ("pair_block_size", PAIR_BLOCK_SIZE),
        ):
            if getattr(args, flag) != default:
                parser.error(f"--{flag.replace('_', '-')} is smoke-only (resume-key stability)")
        n_ext = N_EXT_FULL
        if args.out_root is not None:
            root = args.out_root
        elif pathlib.Path("/workspace").exists():
            root = pathlib.Path("/workspace/eps/out/issue823_ladder_ext")
        else:
            parser.error("production off-pod requires an explicit --out-root")

    layout = Layout(root, args.smoke, n_ext)
    layout.out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "phase=%s smoke=%s n_ext=%d out_root=%s store=%s own=%s",
        args.phase,
        args.smoke,
        n_ext,
        layout.out_root,
        layout.hf_path(layout.store_subpath),
        layout.hf_path(layout.own_subpath),
    )
    PHASES[args.phase](args, layout)


if __name__ == "__main__":
    main()
    # Heavy-C-extension entrypoint: exit explicitly after flushing so the
    # PyGILState_Release atexit race cannot rewrite a completed phase's rc
    # (gotchas.md phased-dispatcher entry; vLLM engine reaped in-phase).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)
