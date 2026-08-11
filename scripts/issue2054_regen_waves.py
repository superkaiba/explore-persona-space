"""Coordinated common-set regeneration wave driver (task #2054 follow-up
`coordinated-common-set-regen`, plan v12 §4).

ONE target conversation set T is fixed up front (R0); every character variant
is generated ON T with retry waves on per-(character, conversation)
rejections; the round's cells are built on the >=4-way survivor set S.
Reuses the realized phase_a seams (`_generate_shortfall`,
`_scaffold_judge_rubric`, `_admit_variant_rows`, `_variant_judge_items`) with
the IDENTICAL admission instrument, wave-distinct seeds (137 + wave), and
<=3 attempts per (character, conversation) (contingency wave 4 caps at 4).

Stages (``--stage``):

- ``draw`` (R0; cpu pod / VM): stage the realized 32,000-row
  ``shared_question_draw`` manifest-first from the PARENT scaffolds prefix
  (the r15 convention), re-apply the draw-time filters at FULL grain
  (plan §12 rows 3/17), draw T uniformly (seed 137), extend the shared fold
  map deterministically (old assignments preserved + asserted against the
  ISSUE-2054 BRANCH copy — the repo-root main copy is a stale n=1,761
  version and is fail-loud rejected), and emit ``gate1_projection.json``.
- ``gen --wave K`` (GPU pod): per-character pending set (T minus admitted,
  attempts < cap), generation via the parent generator subprocess fanned out
  across every provisioned GPU (launcher-env CUDA_VISIBLE_DEVICES pin per
  worker — the gotchas.md CVD rule), prejudge pools + state uploaded
  fail-loud (the cross-machine seam, #1482 class).
- ``judge --wave K`` (VM; Anthropic Batch API): per-row admission at the
  realized instrument, wave manifest with reject reconciliation
  (prejudge == admitted + drops, asserted), cumulative admitted state,
  survivor set S, and the gate-1(d) mid-run re-projection after wave 2.
- ``export`` (VM): S-filtered per-variant admitted pools + assistant
  extension inputs (full-S question rows + the delta rows for on-policy
  generation), uploaded for the R2 pod legs.
- ``assist-merge`` (pod, R2): merge the parent's realized assistant
  on-policy rows (S-intersection) with the freshly generated delta rows so
  the assistant cells reach FULL S coverage before capture (fresh capture,
  no mixed-vintage stores — plan §4 R3).

Trigger-density note: this driver never prints conversation/scaffold TEXT —
logs carry counts, conv_id counts, and digests only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_phase_a as pa  # noqa: E402
import issue2054_resume as resume  # noqa: E402

HF_DATA_REPO = pa.HF_DATA_REPO
PARENT_PREFIX_DEFAULT = "issue2054_lattice"
REGEN_PREFIX_DEFAULT = "issue2054_lattice/common_regen"

CHAR_VARIANTS = ("char_helios", "char_wren", "char_dana", "char_vex")
ASSISTANT_VARIANT = "conversation_paired_stories_assistant"

# Plan §7 gate-1 constants (grounded: fits.py KILL_GATE_4_MIN_INTERSECTION /
# the scope-file 9,000 target — plan §11 Decision Rationale).
GATE1_TARGET = 9_000
GATE1_FLOOR = 4_480
MAX_ATTEMPTS_WAVES_1_3 = 3
MAX_ATTEMPTS_WAVE_4 = 4

# Stale-main-copy guard for the fold-map reference (plan §4 R0 / fact-check
# A38): the branch copy holds n=26,889 assignments; the repo-root main copy
# is a stale n=1,761 version. Anything under this floor is the wrong file.
FOLD_MAP_REF_MIN_KEYS = 20_000


def _log(msg: str) -> None:
    print(f"[phase=regen_waves] {msg}", flush=True)


def assert_args_attrs_defined(module_path: str | Path) -> None:
    """Whole-module `args.<attr>` completeness assert (the #2163 convention;
    this branch predates `orchestrate.argcheck`, so a minimal local copy
    lives here — the gate1/lengths siblings import it). Collects every
    Load-context `args.<attr>` over the ENTIRE module and checks it against
    the argparse dest set (flag-derived + explicit ``dest=`` + Store-context
    ``args.x = ...`` assignments). Raises AssertionError naming the misses.
    """
    import ast

    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    defined: set[str] = set()
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        if a.value.startswith("--"):
                            defined.add(a.value.lstrip("-").replace("-", "_"))
                        elif not a.value.startswith("-"):
                            defined.add(a.value.replace("-", "_"))
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        defined.add(str(kw.value.value))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.attr)
            elif isinstance(node.ctx, ast.Load):
                read.add(node.attr)
    missing = sorted(read - defined)
    assert not missing, f"args attributes read but never defined by argparse: {missing}"


def _utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    pa._atomic_write_json(path, payload)


def _read_jsonl(path: Path) -> list[dict]:
    return pa._read_jsonl(path)


# ---------------------------------------------------------------------------
# Staging helpers (parent-prefix inputs; manifest-first, the r15 convention)
# ---------------------------------------------------------------------------
def _stage_parent_file(parent_prefix: str, rel: str, dest: Path) -> Path:
    from explore_persona_space.orchestrate.hub import stage_hub_file

    return stage_hub_file(
        HF_DATA_REPO, f"{parent_prefix}/{rel}", dest, repo_type="dataset", overwrite=True
    )


def _stage_sharded_or_plain(base_prefix: str, stem: str, dest_dir: Path) -> Path:
    """Manifest-first sharded staging (build_answers r15 recipe) with a plain
    single-file fallback for sub-shard-limit uploads."""
    import issue2054_build_answers as ba

    try:
        return ba._stage_sharded_jsonl(dest_dir, base_prefix, stem)
    except Exception as exc:  # noqa: BLE001 — EntryNotFound shapes vary by hub version
        _log(f"manifest-first staging of {stem} fell back to plain file ({type(exc).__name__})")
        from explore_persona_space.orchestrate.hub import stage_hub_file

        return stage_hub_file(
            HF_DATA_REPO,
            f"{base_prefix}/{stem}.jsonl",
            dest_dir / f"{stem}.jsonl",
            repo_type="dataset",
            overwrite=True,
        )


# ---------------------------------------------------------------------------
# State (target set, attempts, admitted) — persisted per phase, HF-mirrored
# ---------------------------------------------------------------------------
def _state_dir(out_dir: Path) -> Path:
    d = out_dir / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _target_path(out_dir: Path) -> Path:
    return out_dir / "target_set" / "target_set.jsonl"


def _load_target(out_dir: Path) -> list[dict]:
    p = _target_path(out_dir)
    if not p.is_file():
        raise FileNotFoundError(f"target set missing: {p} — run --stage draw first")
    rows = _read_jsonl(p)
    ids = [str(r["conv_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"target set carries duplicate conv_ids ({p})")
    return rows


def _load_state(out_dir: Path) -> dict:
    p = _state_dir(out_dir) / "wave_state.json"
    if not p.is_file():
        raise FileNotFoundError(f"wave state missing: {p} — run --stage init (or draw) first")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_state(out_dir: Path, state: dict) -> Path:
    p = _state_dir(out_dir) / "wave_state.json"
    state["utc"] = _utc()
    _atomic_write_json(p, state)
    return p


def _survivors(state: dict, t_ids: list[str]) -> list[str]:
    """S = {c in T : ALL FOUR characters admitted c} (plan §4 R1)."""
    admitted = {v: set(state["admitted"].get(v) or []) for v in CHAR_VARIANTS}
    return sorted(c for c in t_ids if all(c in admitted[v] for v in CHAR_VARIANTS))


def _wave0_init(args, out_dir: Path, t_ids: list[str]) -> dict:
    """Wave-0 reuse: realized-run admitted conv_ids (kept.json) intersect T
    count as attempt 1 where present (plan §4 R1 wave 0). The admission
    regime is asserted IDENTICAL to this driver's instrument before any
    reuse (attestation (g): regime-sidecar sha match)."""
    if args.kept_json:
        kept_path = Path(args.kept_json)
    else:
        kept_path = _stage_parent_file(
            args.parent_prefix, "scaffolds/kept.json", out_dir / "_staging" / "kept.json"
        )
    kept = json.loads(kept_path.read_text(encoding="utf-8"))
    judge_rec = kept.get("judge") or {}
    rubric_sha = pa.hashlib.sha256(pa._scaffold_judge_rubric().encode()).hexdigest()[:16]
    mismatches = []
    if str(judge_rec.get("rubric_sha256")) != rubric_sha:
        mismatches.append(f"rubric_sha256 {judge_rec.get('rubric_sha256')} != {rubric_sha}")
    if float(judge_rec.get("threshold", -1)) != float(args.judge_keep_threshold):
        mismatches.append(f"threshold {judge_rec.get('threshold')} != {args.judge_keep_threshold}")
    if int(judge_rec.get("n_draws", -1)) != max(1, args.judge_draws):
        mismatches.append(f"n_draws {judge_rec.get('n_draws')} != {max(1, args.judge_draws)}")
    if mismatches:
        raise RuntimeError(
            "wave-0 reuse REFUSED: realized kept.json admission regime differs from this "
            f"driver's instrument ({'; '.join(mismatches)}) — wave-0 admitted rows are only "
            "exchangeable with fresh wave draws under the IDENTICAL instrument (plan §4 R1)"
        )
    t_set = set(t_ids)
    admitted: dict[str, list[str]] = {}
    attempts: dict[str, dict[str, int]] = {}
    variants_rec = kept.get("variants") or {}
    for v in CHAR_VARIANTS:
        rec = variants_rec.get(v)
        if rec is None:
            raise RuntimeError(f"realized kept.json has no admission record for {v!r}")
        ids = {str(x) for x in (rec.get("admitted_conv_ids") or [])}
        reused = sorted(ids & t_set)
        admitted[v] = reused
        attempts[v] = {c: 1 for c in reused}
        _log(f"wave0 {v}: reused {len(reused)}/{len(t_set)} admitted T-members")
    # Assistant coverage rides along for the export stage's delta computation.
    a_rec = variants_rec.get(ASSISTANT_VARIANT) or {}
    assistant_ids = sorted({str(x) for x in (a_rec.get("admitted_conv_ids") or [])} & t_set)
    state = {
        "artifact": "regen_wave_state",
        "hf_prefix": args.hf_prefix,
        "parent_prefix": args.parent_prefix,
        "n_target": len(t_ids),
        "admitted": admitted,
        "attempts": attempts,
        "assistant_existing": assistant_ids,
        "waves_done": [],
        "wave0": {
            "kept_json_sha256": pa.hashlib.sha256(kept_path.read_bytes()).hexdigest(),
            "reused_per_variant": {v: len(admitted[v]) for v in CHAR_VARIANTS},
            "regime": {
                "rubric_sha256": rubric_sha,
                "threshold": float(args.judge_keep_threshold),
                "n_draws": max(1, args.judge_draws),
            },
        },
        "arm_contingency": False,
        "metadata": pa._metadata(args.seed, len(t_ids)),
    }
    _write_state(out_dir, state)
    _log(f"wave0 init: survivors so far |S|={len(_survivors(state, t_ids))}")
    return state


# ---------------------------------------------------------------------------
# Stage: draw (R0)
# ---------------------------------------------------------------------------
def _refilter_draw_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Re-apply the realized draw-time filters at FULL grain (plan §4 R0:
    question length < 400 chars, single-line, dedupe — the phase_a
    rationale block; measured 31,998/32,000 pass)."""
    import issue1345_scaffold_common as sc

    counters = {
        "scanned": 0,
        "empty_question": 0,
        "char_bounds": 0,
        "multiline": 0,
        "sentinel_in_question": 0,
        "dupe_question": 0,
        "dupe_conv_id": 0,
    }
    kept: list[dict] = []
    seen_q: set[str] = set()
    seen_cid: set[str] = set()
    for row in rows:
        counters["scanned"] += 1
        q = str(row.get("question") or "").strip()
        cid = str(row.get("conv_id") or row.get("qid") or "")
        if not q or not cid:
            counters["empty_question"] += 1
            continue
        if not (pa.QUESTION_MIN_CHARS <= len(q) <= pa.QUESTION_MAX_CHARS):
            counters["char_bounds"] += 1
            continue
        if pa.QUESTION_SINGLE_LINE and "\n" in q:
            counters["multiline"] += 1
            continue
        if sc.SLOT_SENTINEL in q:
            counters["sentinel_in_question"] += 1
            continue
        if q in seen_q:
            counters["dupe_question"] += 1
            continue
        if cid in seen_cid:
            counters["dupe_conv_id"] += 1
            continue
        seen_q.add(q)
        seen_cid.add(cid)
        kept.append({"conv_id": cid, "qid": cid, "question": q})
    return kept, counters


def _extend_fold_map(args, out_dir: Path, t_ids: list[str]) -> Path:
    """Deterministic fold-map extension over ref-keys UNION T (plan §4 R0
    step 3): `_conv_grouped_folds` is a pure per-conv_id seeded hash, so
    recomputation preserves every existing assignment BY CONSTRUCTION — the
    subset-equality assert catches drift, and a stale main-copy reference
    (n=1,761) is fail-loud rejected by the size floor."""
    ref_path = Path(args.fold_map_ref)
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"fold-map reference missing: {ref_path} — must be the ISSUE-2054 BRANCH copy "
            "(the --repo-branch issue-2054 clone's eval_results/issue_2054/shared_fold_map.json)"
        )
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    fold_of = ref.get("fold_of") or {}
    if len(fold_of) < FOLD_MAP_REF_MIN_KEYS:
        raise RuntimeError(
            f"fold-map reference {ref_path} has only {len(fold_of)} assignments — this looks "
            f"like the STALE repo-root main copy (n=1,761); the reference MUST be the "
            f"issue-2054 branch copy (n=26,889; fact-check A38). Read it from the "
            "--repo-branch issue-2054 clone or `git show origin/issue-2054:...`."
        )
    k = int(ref.get("k") or 5)
    seed = int(ref.get("seed") or args.seed)
    all_ids = sorted(set(fold_of) | set(t_ids))
    extended = pa._conv_grouped_folds(all_ids, k=k, seed=seed)
    drift = [c for c, f in fold_of.items() if extended.get(str(c)) != int(f)]
    if drift:
        raise RuntimeError(
            f"fold-map extension DRIFTED on {len(drift)} existing assignment(s) "
            f"(first: {drift[:5]}) — the pure-hash preservation invariant is broken"
        )
    out_path = Path(args.fold_map_out)
    payload = {
        "artifact": "shared_fold_map_extended",
        "k": k,
        "seed": seed,
        "n_conv_ids": len(extended),
        "n_reference": len(fold_of),
        "n_new": len(extended) - len(fold_of),
        "reference_path": str(ref_path),
        "reference_sha256": pa.hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        "fold_of": extended,
        "utc": _utc(),
    }
    _atomic_write_json(out_path, payload)
    _log(
        f"fold map extended: {len(fold_of)} reference + {payload['n_new']} new -> "
        f"{len(extended)} (subset-equality PASS) -> {out_path}"
    )
    return out_path


def stage_draw(args) -> int:
    import numpy as np

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir = out_dir / "target_set"
    tgt_dir.mkdir(parents=True, exist_ok=True)

    if args.draw_jsonl:
        draw_path = Path(args.draw_jsonl)
        _log(f"draw source: local {draw_path}")
    else:
        draw_path = _stage_sharded_or_plain(
            f"{args.parent_prefix}/scaffolds", "shared_question_draw", out_dir / "_staging"
        )
        _log(f"draw source: staged {draw_path}")
    rows = _read_jsonl(draw_path)
    kept, counters = _refilter_draw_rows(rows)
    _log(f"re-filter at full grain: {len(kept)}/{len(rows)} survive (counters={counters})")
    if len(kept) < args.target_n:
        # Plan §4 R0 registers a #1738-manifest top-up for this branch; it is
        # not expected (measured 31,998/32,000 survivors) — fail loud naming
        # the registered remedy rather than shipping an untested backfill.
        raise RuntimeError(
            f"re-filter survivors {len(kept)} < |T|={args.target_n} — NOT EXPECTED "
            "(measured 31,998/32,000). Registered remedy: top up from the #1738 manifest "
            "with the same filters (plan §4 R0); surface before proceeding."
        )

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(kept))[: args.target_n]
    drawn = [kept[int(i)] for i in idx]
    ids = [r["conv_id"] for r in drawn]
    assert len(set(ids)) == len(ids), "T draw produced duplicate conv_ids"

    pa._atomic_write_jsonl(_target_path(out_dir), drawn)
    meta = {
        "artifact": "regen_target_set",
        "n": len(drawn),
        "seed": args.seed,
        "population": "full shared_question_draw (32,000-row draw file — plan fact-check C2)",
        "population_rows": len(rows),
        "refilter_survivors": len(kept),
        "refilter_counters": counters,
        "source_sha256": pa.hashlib.sha256(draw_path.read_bytes()).hexdigest(),
        "metadata": pa._metadata(args.seed, len(drawn)),
    }
    _atomic_write_json(tgt_dir / "target_set.meta.json", meta)
    _log(f"T drawn: {len(drawn)} uniform (seed {args.seed}) -> {_target_path(out_dir)}")

    fold_path = _extend_fold_map(args, out_dir, ids)

    # Wave-0 state init (also emits the per-variant reuse fractions the
    # gate-1 projection reads).
    state = _wave0_init(args, out_dir, ids)

    # Plan-time gate-1 arithmetic record (plan §4 R0 step 4).
    per_char = {}
    for v in CHAR_VARIANTS:
        n0 = len(state["admitted"][v])
        pend = args.target_n - n0
        # Plan §11 basis: measured ~49.3% pooled admission per retry wave.
        proj_cov = (n0 + pend * (1.0 - (1.0 - args.assumed_retry_rate) ** 3)) / args.target_n
        per_char[v] = {
            "wave0_admitted": n0,
            "pending_wave1": pend,
            "projected_coverage_3_waves": round(proj_cov, 4),
        }
    proj_s = args.target_n * math.prod(p["projected_coverage_3_waves"] for p in per_char.values())
    projection = {
        "artifact": "gate1_projection",
        "n_target": args.target_n,
        "assumed_retry_admission_rate": args.assumed_retry_rate,
        "per_character": per_char,
        "projected_survivors": int(proj_s),
        "gate1_target": GATE1_TARGET,
        "gate1_floor": GATE1_FLOOR,
        "refilter_survivors": len(kept),
        "utc": _utc(),
    }
    proj_path = Path(args.gate1_projection_out)
    _atomic_write_json(proj_path, projection)
    _log(f"gate1 projection: projected |S| ~= {int(proj_s)} -> {proj_path}")

    if not args.skip_upload:
        files = [
            _target_path(out_dir),
            tgt_dir / "target_set.meta.json",
            _state_dir(out_dir) / "wave_state.json",
        ]
        _upload_regen_files(args, out_dir, files)
        # The fold map + projection live under eval_results/ (git-destined);
        # mirror them under <prefix>/state/ so the R2/R4/R5 pods can stage
        # them without a branch commit round-trip (fail-loud single files).
        _upload_single(args, fold_path, "state/shared_fold_map_extended.json")
        _upload_single(args, proj_path, "state/gate1_projection.json")
    return 0


def _upload_single(args, local: Path, rel: str) -> None:
    from explore_persona_space.orchestrate.hub import _upload

    # UPLOAD_PREFIX_EXEMPT: round-dedicated driver — the default IS this round's common_regen prefix (plan v12 §10); the parent prefix is a separate read-only --parent-prefix
    url = _upload(
        local,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{args.hf_prefix}/{rel}",
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"upload returned no path (failed) -> {args.hf_prefix}/{rel}")
    _log(f"uploaded {local.name} -> {args.hf_prefix}/{rel}")


# ---------------------------------------------------------------------------
# Upload (round prefix; fail-loud — the cross-machine seam)
# ---------------------------------------------------------------------------
def _upload_regen_files(args, out_dir: Path, files: list[Path]) -> None:
    """One bulk fail-loud commit of `files` under `<hf_prefix>/<rel>` (rel
    computed against out_dir), reusing phase_a's sharding + upload seam."""
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    files = pa._shard_large_jsonl_for_upload(files)
    allow = sorted({f.relative_to(out_dir).as_posix() for f in files if f.is_file()})
    if not allow:
        raise RuntimeError(f"upload set resolved EMPTY against declared files: {files}")
    # UPLOAD_PREFIX_EXEMPT: round-dedicated driver — the default IS this round's common_regen prefix (plan v12 §10); the parent prefix is a separate read-only --parent-prefix
    expected = [f"{args.hf_prefix}/{rel}" for rel in allow]
    # UPLOAD_PREFIX_EXEMPT: round-dedicated driver — the default IS this round's common_regen prefix (plan v12 §10); parent prefix is read-only --parent-prefix
    url = _upload_folder_filtered(
        out_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=args.hf_prefix,
        allow_patterns=allow,
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"regen bulk upload failed or incomplete -> {args.hf_prefix}/ "
            "(returned no path; local files kept)"
        )
    _log(f"uploaded {len(allow)} file(s) in one bulk commit -> {args.hf_prefix}/")


def _stage_state_from_hf(args, out_dir: Path) -> None:
    """Stage target set + wave state from the round prefix (pod gen legs)."""
    from explore_persona_space.orchestrate.hub import stage_hub_file

    for rel in ("target_set/target_set.jsonl", "target_set/target_set.meta.json"):
        dest = out_dir / rel
        if not dest.is_file():
            stage_hub_file(HF_DATA_REPO, f"{args.hf_prefix}/{rel}", dest, repo_type="dataset")
    dest = _state_dir(out_dir) / "wave_state.json"
    stage_hub_file(
        HF_DATA_REPO,
        f"{args.hf_prefix}/state/wave_state.json",
        dest,
        repo_type="dataset",
        overwrite=True,
    )
    _log("staged target set + wave state from the round prefix")


# ---------------------------------------------------------------------------
# Stage: gen (wave K; GPU pod)
# ---------------------------------------------------------------------------
def _pending_for_wave(state: dict, t_ids: list[str], wave: int) -> dict[str, list[str]]:
    cap = MAX_ATTEMPTS_WAVE_4 if wave >= 4 else MAX_ATTEMPTS_WAVES_1_3
    pending: dict[str, list[str]] = {}
    for v in CHAR_VARIANTS:
        admitted = set(state["admitted"].get(v) or [])
        att = state["attempts"].get(v) or {}
        pending[v] = [c for c in t_ids if c not in admitted and int(att.get(c, 0)) < cap]
    if wave >= 4:
        pending = _contingency_restrict(state, t_ids, pending)
    return pending


def _measured_retry_rate(state: dict) -> float | None:
    """Pooled admitted/attempted over the recorded retry waves (>=2), falling
    back to wave 1; None when no wave has run."""
    waves = state.get("waves_done") or []
    for subset in ([w for w in waves if int(w.get("wave", 0)) >= 2], waves):
        att = sum(int(w.get("n_attempted") or 0) for w in subset)
        adm = sum(int(w.get("n_admitted_new") or 0) for w in subset)
        if att > 0:
            return adm / att
    return None


def _contingency_restrict(
    state: dict, t_ids: list[str], pending: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Gate-1(b) contingency wave 4: size ceil(gap / measured retry admission
    rate) conversations, prioritized by FEWEST missing characters (highest
    survivor leverage), attempts < 4 (plan §7 gate 1(b))."""
    s = set(_survivors(state, t_ids))
    gap = max(0, GATE1_TARGET - len(s))
    rate = _measured_retry_rate(state)
    if rate is None or rate <= 0:
        raise RuntimeError("contingency wave needs a measured retry admission rate (> 0)")
    n_convs = math.ceil(gap / rate)
    admitted = {v: set(state["admitted"].get(v) or []) for v in CHAR_VARIANTS}
    eligible_of = {v: set(pending[v]) for v in CHAR_VARIANTS}
    candidates = []
    for c in t_ids:
        if c in s:
            continue
        missing = [v for v in CHAR_VARIANTS if c not in admitted[v]]
        # Every missing character must still be retryable (attempts < 4).
        if missing and all(c in eligible_of[v] for v in missing):
            candidates.append((len(missing), c, missing))
    candidates.sort(key=lambda x: (x[0], x[1]))
    chosen = candidates[:n_convs]
    out: dict[str, list[str]] = {v: [] for v in CHAR_VARIANTS}
    for _n, c, missing in chosen:
        for v in missing:
            out[v].append(c)
    _log(
        f"contingency wave: gap={gap} rate={rate:.3f} -> {len(chosen)} conversations, "
        f"attempts per variant: { {v: len(out[v]) for v in CHAR_VARIANTS} }"
    )
    return out


def _wave_dir(out_dir: Path, wave: int) -> Path:
    d = out_dir / "waves" / f"wave{wave}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prejudge_name(variant: str, wave: int) -> str:
    return f"scaffolds_{variant}_prejudge.wave{wave}.jsonl"


def _record_gen_wave(
    state: dict,
    wave: int,
    active: list[str],
    pending: dict[str, list[str]],
    counts: dict,
    seed: int,
    mock: bool,
) -> bool:
    """Increment the <=3-attempts ledger + append the gen_waves record,
    IDEMPOTENT per wave: a crashed/re-run gen leg must not double-increment
    attempts for the same wave (rows would lose retry budget early and the
    gate-1(d) projection would skew low; code-review r1 Minor 2). Returns
    True when the ledger was updated (first recorded run of this wave)."""
    if any(int(w.get("wave", -1)) == wave for w in state.get("gen_waves") or []):
        _log(f"wave {wave} gen already in attempts ledger — increment skipped (re-run)")
        return False
    for v in active:
        att = state["attempts"].setdefault(v, {})
        for c in pending[v]:
            att[c] = int(att.get(c, 0)) + 1
    state.setdefault("gen_waves", []).append(
        {
            "wave": wave,
            "seed": seed,
            "pending": {v: len(pending[v]) for v in CHAR_VARIANTS},
            "counts": counts,
            "mock": mock,
            "utc": _utc(),
        }
    )
    return True


def stage_gen(args) -> int:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.state_from_hf:
        _stage_state_from_hf(args, out_dir)
    t_rows = _load_target(out_dir)
    t_ids = [r["conv_id"] for r in t_rows]
    by_id = {r["conv_id"]: r for r in t_rows}
    state = _load_state(out_dir)
    wave = int(args.wave)
    assert args.seed + wave != args.seed, "wave seed must differ from the base seed"

    pending = _pending_for_wave(state, t_ids, wave)
    total = sum(len(v) for v in pending.values())
    _log(
        f"wave {wave} gen: pending per variant "
        f"{ {v: len(pending[v]) for v in CHAR_VARIANTS} } (total {total})"
    )
    if total == 0:
        _log("nothing pending — gen leg is a no-op")
        return 0

    wave_dir = _wave_dir(out_dir, wave)
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
    if not gpus:
        gpus = ["0"]
    # Round-robin variants into one sequential lane per GPU: every provisioned
    # GPU is saturated (workflow-v2 guideline 2) and no two engines share a
    # device (launcher-env CVD pin — the gotchas.md rule; the in-process
    # clobber alone is defeated by import-time cuInit).
    lanes: list[list[str]] = [[] for _ in gpus]
    active = [v for v in CHAR_VARIANTS if pending[v]]
    for i, v in enumerate(active):
        lanes[i % len(gpus)].append(v)

    results: dict[str, dict] = {}

    def _run_lane(lane_idx: int) -> None:
        gpu = gpus[lane_idx]
        for v in lanes[lane_idx]:
            questions = [by_id[c] for c in pending[v]]
            rows, counts = pa._generate_shortfall(
                v,
                questions,
                wave_dir,
                seed=args.seed + wave,
                mock=args.gen_mock,
                gen_model=args.gen_model,
                extra_env={"CUDA_VISIBLE_DEVICES": gpu},
            )
            vdir = out_dir / "scaffolds" / v
            vdir.mkdir(parents=True, exist_ok=True)
            pj_path = vdir / _prejudge_name(v, wave)
            pa._atomic_write_jsonl(pj_path, rows)
            regime = {
                "stage": "regen-wave-gen",
                "wave": wave,
                "variant": v,
                "seed": args.seed + wave,
                "gen_model": args.gen_model,
                "mock": bool(args.gen_mock),
            }
            resume.write_done(
                pj_path,
                regime,
                inputs={"n_pending": len(questions)},
                extra={"prejudge_sha256": resume.file_sha256(pj_path), "counts": counts},
            )
            results[v] = {"counts": counts, "path": str(pj_path)}
            _log(f"wave {wave} {v}: prejudge {len(rows)} rows (gpu={gpu})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(_run_lane, i) for i in range(len(gpus)) if lanes[i]]
        for f in concurrent.futures.as_completed(futs):
            f.result()  # fail loud on the first lane error

    # Attempts increment for every generated pair (the <=3-attempts ledger) —
    # idempotent per wave: a same-wave re-run skips the increment (r1 Minor 2).
    if _record_gen_wave(
        state,
        wave,
        active,
        pending,
        {v: results.get(v, {}).get("counts") for v in active},
        args.seed + wave,
        bool(args.gen_mock),
    ):
        _write_state(out_dir, state)

    if not args.skip_upload:
        files = [Path(results[v]["path"]) for v in active]
        sidecars = [resume.sidecar_path(p) for p in files]
        files += [s for s in sidecars if s.is_file()]
        files.append(_state_dir(out_dir) / "wave_state.json")
        _upload_regen_files(args, out_dir, files)
    return 0


# ---------------------------------------------------------------------------
# Stage: judge (wave K; VM, Batch API)
# ---------------------------------------------------------------------------
def stage_judge(args) -> int:
    from explore_persona_space.eval.graded_judge import DEFAULT_JUDGE_MODEL, judge_graded

    out_dir = Path(args.output_dir).resolve()
    if args.state_from_hf:
        # The gen leg ran POD-side and uploaded state + prejudge; a fresh VM
        # out-dir stages both back (the cross-machine seam, #1482 class).
        _stage_state_from_hf(args, out_dir)
    t_rows = _load_target(out_dir)
    t_ids = [r["conv_id"] for r in t_rows]
    state = _load_state(out_dir)
    wave = int(args.wave)
    wave_rec_existing = [w for w in (state.get("waves_done") or []) if w.get("wave") == wave]
    if wave_rec_existing:
        _log(f"wave {wave} judge already recorded — no-op (resume)")
        return 0

    rubric = pa._scaffold_judge_rubric()

    # Argv-vs-instrument drift guard (r1 Minor 3): the judge leg's argv must
    # match the wave-0 recorded admission regime — the draw-stage assert binds
    # only the draw invocation, so a hand-invoked judge leg with different
    # flags would otherwise silently run a drifted instrument.
    regime0 = (state.get("wave0") or {}).get("regime") or {}
    if not regime0:
        raise RuntimeError("state has no wave0.regime record — re-run --stage draw first")
    rubric_sha = pa.hashlib.sha256(rubric.encode()).hexdigest()[:16]
    drift = []
    if str(regime0.get("rubric_sha256")) != rubric_sha:
        drift.append(f"rubric_sha256 {regime0.get('rubric_sha256')} != {rubric_sha}")
    if float(regime0.get("threshold", -1)) != float(args.judge_keep_threshold):
        drift.append(f"threshold {regime0.get('threshold')} != {args.judge_keep_threshold}")
    if int(regime0.get("n_draws", -1)) != max(1, args.judge_draws):
        drift.append(f"n_draws {regime0.get('n_draws')} != {max(1, args.judge_draws)}")
    if drift:
        raise RuntimeError(
            f"wave {wave} judge argv drifts from the wave-0 recorded instrument "
            f"({'; '.join(drift)}) — every wave must run the IDENTICAL admission "
            "regime (plan §4 R1)"
        )

    cache_root = out_dir / "_judge_cache" / f"wave{wave}"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Load (or stage) this wave's prejudge pools.
    prejudge: dict[str, list[dict]] = {}
    for v in CHAR_VARIANTS:
        pj_path = out_dir / "scaffolds" / v / _prejudge_name(v, wave)
        if not pj_path.is_file() and args.prejudge_from_hf:
            _stage_sharded_or_plain(
                f"{args.hf_prefix}/scaffolds/{v}",
                _prejudge_name(v, wave).removesuffix(".jsonl"),
                pj_path.parent,
            )
        if pj_path.is_file():
            rows = _read_jsonl(pj_path)
            if rows:
                prejudge[v] = rows
    if not prejudge:
        raise FileNotFoundError(
            f"no wave-{wave} prejudge pools found under {out_dir}/scaffolds/ "
            "(run --stage gen first, or pass --prejudge-from-hf)"
        )

    per_variant_items = {v: pa._variant_judge_items(v, rows) for v, rows in prejudge.items()}
    total_calls = sum(len(i) for i, _r, _n in per_variant_items.values()) * max(1, args.judge_draws)
    pilot_report: dict | None = None
    if total_calls >= 5000:
        if wave == 1 and not args.skip_pilot:
            _log(f"wave 1 judge wave {total_calls} calls >= 5000 — rule-26 pilot gate")
            pilot_report = pa._run_judge_pilot(prejudge, args, cache_root)
            verdict = str(pilot_report.get("verdict", "")).upper()
            if verdict not in {"PASS", "PASS_WAIVED"}:
                raise pa.PilotGateRefusal(
                    f"judge pilot verdict={verdict!r} — production wave refused "
                    f"(report: {cache_root / 'pilot_gate_report.json'})"
                )
        else:
            # Plan §7: waves 2-3 inherit wave 1's verdict (same instrument;
            # wave 1's realized drop profile supersedes a 200-draw pilot).
            _log(f"wave {wave}: pilot verdict inherited from wave 1 (identical instrument)")

    manifest: dict = {
        "artifact": "regen_wave_manifest",
        "wave": wave,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "rubric_sha256": pa.hashlib.sha256(rubric.encode()).hexdigest()[:16],
        "threshold": float(args.judge_keep_threshold),
        "n_draws": max(1, args.judge_draws),
        "max_tokens": int(args.max_tokens),
        "n_calls": total_calls,
        "pilot": pilot_report,
        "variants": {},
    }
    n_attempted_total = 0
    n_admitted_total = 0
    for v, (items, judged_rows, n_no_question) in per_variant_items.items():
        if not items:
            manifest["variants"][v] = {
                "attempted": 0,
                "admitted_new": 0,
                "structural_no_question": n_no_question,
            }
            continue
        result = judge_graded(
            items,
            rubric,
            n_draws=max(1, args.judge_draws),
            cache_dir=cache_root / "prod" / v,
            save_raw=out_dir / "judge_raw" / f"judge_raw_{v}_wave{wave}.json",
            max_tokens=args.max_tokens,
        )
        kept_rows, drops = pa._admit_variant_rows(
            judged_rows, items, result, args.judge_keep_threshold
        )
        # Reject reconciliation (plan §4 R1: rejects = prejudge minus
        # admitted; the prejudge pools persist every judged row).
        n_attempted = len(items)
        n_rejected = n_attempted - len(kept_rows)
        if n_rejected != drops["below_threshold"] + drops["judge_content_drop"]:
            raise RuntimeError(
                f"wave {wave} {v}: reject reconciliation FAILED — "
                f"{n_rejected} != below_threshold {drops['below_threshold']} + "
                f"content_drop {drops['judge_content_drop']}"
            )
        adm_path = out_dir / "scaffolds" / v / f"scaffolds_{v}.wave{wave}_admitted.jsonl"
        pa._atomic_write_jsonl(adm_path, kept_rows)
        new_ids = sorted({str(r.get("conv_id")) for r in kept_rows})
        prior = set(state["admitted"].get(v) or [])
        dupes = prior & set(new_ids)
        if dupes:
            raise RuntimeError(
                f"wave {wave} {v}: {len(dupes)} newly admitted conv_ids were ALREADY "
                f"admitted (first: {sorted(dupes)[:5]}) — pending-set computation broken"
            )
        state["admitted"][v] = sorted(prior | set(new_ids))
        # Retried-pair admission (pairs at attempt >= 2) for gate 1(d).
        att = state["attempts"].get(v) or {}
        retried = [c for c in new_ids if int(att.get(c, 0)) >= 2]
        manifest["variants"][v] = {
            "attempted": n_attempted,
            "admitted_new": len(kept_rows),
            "rejected": n_rejected,
            "drops": drops,
            "structural_no_question": n_no_question,
            "admission_rate": round(len(kept_rows) / n_attempted, 4),
            "admitted_retried_pairs": len(retried),
            "judge_telemetry": {
                "n_total_draws": result.n_total_draws,
                "n_dropped_draws": result.n_dropped_draws,
                "n_transport_lost_draws": result.n_transport_lost_draws,
                "n_refusal_draws": result.n_refusal_draws,
                "n_api_refusal_draws": getattr(result, "n_api_refusal_draws", None),
                "n_truncation_dropped_draws": result.n_truncation_dropped_draws,
                "stop_reason_tally": result.stop_reason_tally,
            },
        }
        n_attempted_total += n_attempted
        n_admitted_total += len(kept_rows)
        _log(
            f"wave {wave} {v}: admitted {len(kept_rows)}/{n_attempted} "
            f"(cumulative {len(state['admitted'][v])})"
        )

    s_now = _survivors(state, t_ids)
    manifest["n_attempted"] = n_attempted_total
    manifest["n_admitted_new"] = n_admitted_total
    manifest["survivors_after_wave"] = len(s_now)
    manifest["utc"] = _utc()

    state.setdefault("waves_done", []).append(
        {
            "wave": wave,
            "n_attempted": n_attempted_total,
            "n_admitted_new": n_admitted_total,
            "survivors": len(s_now),
        }
    )

    # Gate-1(d) mid-run re-projection after wave 2 (plan §7): project 4-way
    # survival from the MEASURED retry admission on retried pairs; < target
    # arms the contingency wave preemptively (no extra approval).
    if wave >= 2:
        import numpy as np

        rate = _measured_retry_rate(state) or 0.0
        waves_left = max(0, MAX_ATTEMPTS_WAVES_1_3 - wave)
        adm = {v: set(state["admitted"].get(v) or []) for v in CHAR_VARIANTS}
        att = state["attempts"]
        p_rows = np.ones((len(t_ids), len(CHAR_VARIANTS)), dtype=np.float64)
        cap = MAX_ATTEMPTS_WAVES_1_3
        for j, v in enumerate(CHAR_VARIANTS):
            a = adm[v]
            av = att.get(v) or {}
            col = np.array(
                [
                    1.0
                    if c in a
                    else (1.0 - (1.0 - rate) ** min(waves_left, max(0, cap - int(av.get(c, 0)))))
                    for c in t_ids
                ],
                dtype=np.float64,
            )
            p_rows[:, j] = col
        projected = float(p_rows.prod(axis=1).sum())
        manifest["gate1d_reprojection"] = {
            "measured_retry_rate": round(rate, 4),
            "waves_left": waves_left,
            "projected_survivors": int(projected),
            "arms_contingency": bool(projected < GATE1_TARGET),
        }
        if projected < GATE1_TARGET:
            state["arm_contingency"] = True
            _log(
                f"gate 1(d): projected |S| {int(projected)} < {GATE1_TARGET} — "
                "contingency wave 4 ARMED"
            )

    man_path = _wave_dir(out_dir, wave) / f"kept.wave{wave}.json"
    _atomic_write_json(man_path, manifest)
    # The canonical uploaded copy rides at scaffolds/kept.wave<k>.json (plan
    # §10 phase_outputs r1_wave_judge_k).
    kept_copy = out_dir / "scaffolds" / f"kept.wave{wave}.json"
    _atomic_write_json(kept_copy, manifest)
    _write_state(out_dir, state)

    surv_path = out_dir / "scaffolds" / "survivor_set.json"
    _atomic_write_json(
        surv_path,
        {
            "artifact": "regen_survivor_set",
            "after_wave": wave,
            "n_target": len(t_ids),
            "n_survivors": len(s_now),
            "survivor_conv_ids": s_now,
            "utc": _utc(),
        },
    )
    _log(f"wave {wave} judged: |S|={len(s_now)} -> {surv_path}")

    if not args.skip_upload:
        files = [kept_copy, surv_path, _state_dir(out_dir) / "wave_state.json"]
        for v in prejudge:
            p = out_dir / "scaffolds" / v / f"scaffolds_{v}.wave{wave}_admitted.jsonl"
            if p.is_file():
                files.append(p)
        _upload_regen_files(args, out_dir, files)
    return 0


# ---------------------------------------------------------------------------
# Stage: export (VM) — S-filtered pools + assistant extension inputs
# ---------------------------------------------------------------------------
def _parent_assistant_realized_ids(args) -> set[str]:
    """Conv-id coverage of the parent's REALIZED assistant on-policy files.

    The kept.json ADMITTED set overstates realized coverage: the parent's
    phase_c ran under its default ``--target-conv-ids 8000`` first-N cap, so
    the realized assistant on-policy files hold a fixed ~8,000-id subset of
    the ~11,915 admitted ids (fits_digest ``inter=8000`` on all 4 assistant
    on-policy cells). The delta MUST be computed against the FILE coverage —
    the only other row source at assist-merge — or ``stage_assist_merge``'s
    coverage assert aborts the R2 leg AFTER the phase_c GPU spend
    (code-review r1 Critical 1). Fetches all 4 parent cells (2 models × 2
    forms); identical id-sets are used directly, a mismatch degrades to the
    INTERSECTION with a WARN (per-cell coverage is then a superset, so the
    merge still reaches full S by construction)."""
    import issue2054_forms as forms
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    per_file: dict[str, set[str]] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for form in ("chat", "bare_text"):
            fname = forms.phase_output_name("on_policy", ASSISTANT_VARIANT, form)
            rel = f"{args.parent_prefix}/on_policy/{model}/{ASSISTANT_VARIANT}/{fname}"
            local = retry_transient(
                lambda rel=rel: hf_hub_download(
                    repo_id=HF_DATA_REPO, repo_type="dataset", filename=rel
                ),
                what=f"hf_hub_download({rel})",
            )
            per_file[rel] = {str(r.get("conv_id")) for r in _read_jsonl(Path(local))}
    sets = list(per_file.values())
    ids: set[str] = set.intersection(*sets)
    if any(s != ids for s in sets):
        sizes = {rel: len(v) for rel, v in per_file.items()}
        _log(
            f"WARN parent assistant on-policy id-sets differ across cells ({sizes}) — "
            f"using the {len(ids)}-id intersection as existing coverage"
        )
    if not ids:
        raise RuntimeError(
            "parent assistant on-policy realized coverage is EMPTY across all cells — "
            "cannot derive the assistant delta basis"
        )
    return ids


def stage_export(args) -> int:
    out_dir = Path(args.output_dir).resolve()
    if args.state_from_hf:
        _stage_state_from_hf(args, out_dir)
    t_rows = _load_target(out_dir)
    t_ids = [r["conv_id"] for r in t_rows]
    by_id = {r["conv_id"]: r for r in t_rows}
    state = _load_state(out_dir)
    s = _survivors(state, t_ids)
    s_set = set(s)
    if not s:
        raise RuntimeError("survivor set is EMPTY — nothing to export")

    import issue2054_build_answers as ba

    files: list[Path] = []
    for v in CHAR_VARIANTS:
        admitted = set(state["admitted"].get(v) or [])
        need = s_set & admitted
        if need != s_set:
            raise RuntimeError(
                f"{v}: survivor set not fully admitted ({len(need)}/{len(s_set)}) — "
                "S must be the 4-way admitted intersection by construction"
            )
        rows_by_id: dict[str, dict] = {}
        # Wave-0 rows come from the parent's realized admitted pools
        # (manifest-first sharded staging); wave >=1 rows from this round's
        # per-wave admitted JSONLs.
        wave0_reused = set(state["attempts"].get(v) or {}) & s_set
        wave0_admitted = {c for c in wave0_reused if int(state["attempts"][v].get(c, 0)) == 1}
        # attempts==1 covers both wave-0 reuse AND wave-1 fresh admissions;
        # disambiguate by pulling per-wave admitted files first, then filling
        # the remainder from the parent pool.
        for wave_rec in state.get("waves_done") or []:
            k = int(wave_rec["wave"])
            p = out_dir / "scaffolds" / v / f"scaffolds_{v}.wave{k}_admitted.jsonl"
            if p.is_file():
                for r in _read_jsonl(p):
                    cid = str(r.get("conv_id"))
                    if cid in s_set:
                        rows_by_id[cid] = r
        missing = s_set - set(rows_by_id)
        if missing:
            staged = ba._stage_sharded_jsonl(
                out_dir / "_staging" / args.parent_prefix.replace("/", "_") / v,
                f"{args.parent_prefix}/scaffolds/{v}",
                f"scaffolds_{v}",
            )
            for r in _read_jsonl(staged):
                cid = str(r.get("conv_id"))
                if cid in missing:
                    rows_by_id[cid] = r
        still_missing = s_set - set(rows_by_id)
        if still_missing:
            raise RuntimeError(
                f"{v}: {len(still_missing)} survivor rows located in NEITHER this round's "
                f"admitted waves NOR the parent pool (first: {sorted(still_missing)[:5]})"
            )
        pool = [rows_by_id[c] for c in s]
        vdir = out_dir / "scaffolds" / v
        vdir.mkdir(parents=True, exist_ok=True)
        pool_path = vdir / f"scaffolds_{v}.jsonl"
        pa._atomic_write_jsonl(pool_path, pool)
        resume.write_done(
            pool_path,
            {"stage": "regen-export", "variant": v, "n_survivors": len(s)},
            inputs={"state_sha256": resume.file_sha256(_state_dir(out_dir) / "wave_state.json")},
            extra={"n_rows": len(pool), "wave0_candidates": len(wave0_admitted)},
        )
        files.append(pool_path)
        _log(f"export {v}: {len(pool)} S-rows -> {pool_path}")

    # Assistant extension inputs: full-S question rows (deterministic
    # template renders — phase_b splice + capture read them) + the on-policy
    # DELTA for phase_c generation. Delta basis = the parent's REALIZED
    # on-policy file conv_ids (the only other row source at assist-merge) —
    # NEVER the kept.json admitted set, which is ~11,915 while the realized
    # files are 8,000-capped (code-review r1 Critical 1).
    admitted_existing = set(state.get("assistant_existing") or [])
    existing = _parent_assistant_realized_ids(args)
    delta = sorted(s_set - existing)

    def _assistant_row(cid: str) -> dict:
        r = by_id[cid]
        return {
            "scaffold_id": cid,
            "conv_id": cid,
            "qid": cid,
            "variant": ASSISTANT_VARIANT,
            "character": "Assistant",
            "question": r["question"],
            "scaffold_text": "",
            "provenance": "regen_target_set",
        }

    a_dir = out_dir / "scaffolds" / ASSISTANT_VARIANT
    a_dir.mkdir(parents=True, exist_ok=True)
    a_full = a_dir / f"scaffolds_{ASSISTANT_VARIANT}.jsonl"
    pa._atomic_write_jsonl(a_full, [_assistant_row(c) for c in s])
    files.append(a_full)
    d_dir = out_dir / "assistant_delta" / ASSISTANT_VARIANT
    d_dir.mkdir(parents=True, exist_ok=True)
    a_delta = d_dir / f"scaffolds_{ASSISTANT_VARIANT}.jsonl"
    pa._atomic_write_jsonl(a_delta, [_assistant_row(c) for c in delta])
    files.append(a_delta)
    _log(
        f"export assistant: full-S {len(s)} rows; on-policy delta {len(delta)} rows "
        f"(realized parent coverage in S {len(existing & s_set)}; "
        f"admitted-in-S {len(admitted_existing & s_set)} — informational only)"
    )

    export_manifest = {
        "artifact": "regen_export",
        "n_survivors": len(s),
        "assistant_delta": len(delta),
        "assistant_existing_in_s": len(existing & s_set),
        "assistant_admitted_in_s": len(admitted_existing & s_set),
        "assistant_delta_basis": "parent_realized_on_policy_files",
        "variants": [*CHAR_VARIANTS, ASSISTANT_VARIANT],
        "metadata": pa._metadata(args.seed, len(s)),
    }
    man_path = out_dir / "scaffolds" / "export_manifest.json"
    _atomic_write_json(man_path, export_manifest)
    files.append(man_path)

    if not args.skip_upload:
        _upload_regen_files(args, out_dir, files)
    return 0


# ---------------------------------------------------------------------------
# Stage: stage-r2 (pod) — stage the export outputs for the R2 completion build
# ---------------------------------------------------------------------------
def stage_r2_inputs(args) -> int:
    """Stage the export-stage outputs from the round prefix into the local
    layout the R2 phase legs consume (the cross-machine seam, #1482 class):
    per-variant S-pools, assistant delta, survivor set, export manifest, the
    extended answers pool, and the extended fold map."""
    from explore_persona_space.orchestrate.hub import stage_hub_file

    out_dir = Path(args.output_dir).resolve()
    for v in (*CHAR_VARIANTS, ASSISTANT_VARIANT):
        _stage_sharded_or_plain(
            f"{args.hf_prefix}/scaffolds/{v}", f"scaffolds_{v}", out_dir / "scaffolds" / v
        )
    _stage_sharded_or_plain(
        f"{args.hf_prefix}/assistant_delta/{ASSISTANT_VARIANT}",
        f"scaffolds_{ASSISTANT_VARIANT}",
        out_dir / "assistant_delta" / ASSISTANT_VARIANT,
    )
    for rel in ("scaffolds/survivor_set.json", "scaffolds/export_manifest.json"):
        stage_hub_file(
            HF_DATA_REPO,
            f"{args.hf_prefix}/{rel}",
            out_dir / rel,
            repo_type="dataset",
            overwrite=True,
        )
    _stage_sharded_or_plain(f"{args.hf_prefix}/answers", "answers_pool", out_dir / "answers")
    try:
        stage_hub_file(
            HF_DATA_REPO,
            f"{args.hf_prefix}/answers/answers_excluded_conv_ids.json",
            out_dir / "answers" / "answers_excluded_conv_ids.json",
            repo_type="dataset",
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001
        _log(
            f"WARN answers_excluded_conv_ids.json not staged ({type(exc).__name__}) — "
            "phase_b treats a missing exclusion manifest as fail-open (its recorded concern)"
        )
    fold_dest = Path(args.fold_map_out)
    stage_hub_file(
        HF_DATA_REPO,
        f"{args.hf_prefix}/state/shared_fold_map_extended.json",
        fold_dest,
        repo_type="dataset",
        overwrite=True,
    )
    _log(f"staged R2 inputs under {out_dir} (fold map -> {fold_dest})")
    return 0


# ---------------------------------------------------------------------------
# Stage: assist-merge (pod, R2) — full-S assistant on-policy coverage
# ---------------------------------------------------------------------------
def stage_assist_merge(args) -> int:
    """Merge parent realized assistant on-policy rows (S-intersection) with
    this round's delta generations so every assistant on_policy cell covers
    FULL S before capture (plan §4 R2/R3; delta ∩ existing == ∅ asserted)."""
    import issue2054_forms as forms
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    out_dir = Path(args.output_dir).resolve()
    surv_path = out_dir / "scaffolds" / "survivor_set.json"
    if not surv_path.is_file():
        _stage_parent_file(args.hf_prefix, "scaffolds/survivor_set.json", surv_path)
    s_set = set(json.loads(surv_path.read_text(encoding="utf-8"))["survivor_conv_ids"])
    if not s_set:
        raise RuntimeError("survivor set is EMPTY")

    on_policy_root = Path(args.on_policy_dir).resolve()
    merged_total = 0
    merged_paths: list[Path] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for form in ("chat", "bare_text"):
            fname = forms.phase_output_name("on_policy", ASSISTANT_VARIANT, form)
            delta_path = on_policy_root / model / ASSISTANT_VARIANT / fname
            delta_rows = _read_jsonl(delta_path) if delta_path.is_file() else []
            delta_ids = {str(r.get("conv_id")) for r in delta_rows}
            parent_rel = f"{args.parent_prefix}/on_policy/{model}/{ASSISTANT_VARIANT}/{fname}"
            local = retry_transient(
                lambda rel=parent_rel: hf_hub_download(
                    repo_id=HF_DATA_REPO, repo_type="dataset", filename=rel
                ),
                what=f"hf_hub_download({parent_rel})",
            )
            parent_rows = [
                r
                for r in _read_jsonl(Path(local))
                if str(r.get("conv_id")) in s_set and str(r.get("conv_id")) not in delta_ids
            ]
            merged = parent_rows + delta_rows
            merged_ids = {str(r.get("conv_id")) for r in merged} & s_set
            if merged_ids != s_set:
                raise RuntimeError(
                    f"assist-merge {model}/{form}: merged coverage {len(merged_ids)} != "
                    f"|S| {len(s_set)} — assistant cell would be under-covered at capture"
                )
            merged = [r for r in merged if str(r.get("conv_id")) in s_set]
            out_path = on_policy_root / model / ASSISTANT_VARIANT / fname
            pa._atomic_write_jsonl(out_path, merged)
            resume.write_done(
                out_path,
                {
                    "stage": "regen-assist-merge",
                    "model": model,
                    "form": form,
                    "n_survivors": len(s_set),
                },
                inputs={"survivor_sha256": resume.file_sha256(surv_path)},
                extra={"n_parent": len(parent_rows), "n_delta": len(delta_rows)},
            )
            merged_paths.append(out_path)
            merged_total += len(merged)
            _log(
                f"assist-merge {model}/{form}: parent {len(parent_rows)} + delta "
                f"{len(delta_rows)} -> {len(merged)} rows (full S)"
            )

    if not args.skip_upload:
        # Re-upload the merged assistant files under the round's on_policy
        # prefix (per model — the phase_c upload convention).
        from explore_persona_space.orchestrate.hub import _upload_folder_filtered

        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            root = on_policy_root / model
            paths = [p for p in merged_paths if str(p).startswith(str(root) + os.sep)]
            if not paths:
                continue
            allow = sorted(p.relative_to(root).as_posix() for p in paths)
            # UPLOAD_PREFIX_EXEMPT: round-dedicated driver — the default IS this round's common_regen prefix (plan v12 §10); the parent prefix is a separate read-only --parent-prefix
            expected = [f"{args.hf_prefix}/on_policy/{model}/{rel}" for rel in allow]
            # UPLOAD_PREFIX_EXEMPT: round-dedicated driver — the default IS this round's common_regen prefix (plan v12 §10); parent prefix is read-only --parent-prefix
            url = _upload_folder_filtered(
                root,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{args.hf_prefix}/on_policy/{model}",
                allow_patterns=allow,
                expected_repo_paths=expected,
            )
            if not url:
                raise RuntimeError(
                    f"assist-merge upload failed -> {args.hf_prefix}/on_policy/{model}/"
                )
    _log(f"assist-merge complete: {merged_total} rows across cells")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--stage",
        default=None,
        choices=("draw", "init", "gen", "judge", "export", "stage-r2", "assist-merge"),
        help="REQUIRED except under --import-check",
    )
    p.add_argument("--wave", type=int, default=None, help="wave index (gen/judge stages)")
    p.add_argument("--output-dir", default="data/issue_2054/common_regen")
    p.add_argument("--target-n", type=int, default=15_700, help="|T| (plan §11: 15,700)")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--hf-prefix", default=REGEN_PREFIX_DEFAULT)
    p.add_argument("--parent-prefix", default=PARENT_PREFIX_DEFAULT)
    p.add_argument(
        "--draw-jsonl", default=None, help="local draw-file override (smoke seam; skips staging)"
    )
    p.add_argument(
        "--kept-json", default=None, help="local realized kept.json override (smoke seam)"
    )
    p.add_argument(
        "--fold-map-ref",
        default="eval_results/issue_2054/shared_fold_map.json",
        help="ISSUE-2054 BRANCH fold-map copy (n=26,889); the stale main copy is refused",
    )
    p.add_argument(
        "--fold-map-out",
        default="eval_results/issue_2054/coordinated_common_set_regen/shared_fold_map_extended.json",
    )
    p.add_argument(
        "--gate1-projection-out",
        default="eval_results/issue_2054/coordinated_common_set_regen/gate1_projection.json",
    )
    p.add_argument(
        "--assumed-retry-rate",
        type=float,
        default=0.493,
        help="plan-time projection basis (measured realized pooled admission)",
    )
    p.add_argument("--gpus", default="0,1", help="comma GPU ids for the gen fan-out")
    p.add_argument("--gen-mock", action="store_true", help="parent generator --mock (CPU smoke)")
    p.add_argument("--gen-model", choices=("instruct", "pretrained"), default="instruct")
    p.add_argument("--state-from-hf", action="store_true", help="stage state from the round prefix")
    p.add_argument("--prejudge-from-hf", action="store_true")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--pilot-n", type=int, default=200)
    p.add_argument("--judge-draws", type=int, default=1)
    p.add_argument("--judge-keep-threshold", type=float, default=50.0)
    p.add_argument("--skip-pilot", action="store_true")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument(
        "--on-policy-dir",
        default="data/issue_2054/common_regen/on_policy",
        help="assist-merge: the round's phase_c output root",
    )
    p.add_argument(
        "--models",
        default="qwen2.5-7b-instruct,qwen2.5-7b",
        help="assist-merge: model slugs to merge",
    )
    p.add_argument(
        "--import-check",
        action="store_true",
        help="resolve deferred imports + args-attribute completeness, then exit 0",
    )
    args = p.parse_args()

    if args.import_check:
        import issue2054_build_answers  # noqa: F401
        import issue2054_forms  # noqa: F401
        import numpy  # noqa: F401

        from explore_persona_space.eval.graded_judge import (  # noqa: F401
            DEFAULT_JUDGE_MODEL,
            judge_graded,
        )
        from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            _upload_folder_filtered,
            retry_transient,
            stage_hub_file,
        )

        assert_args_attrs_defined(__file__)
        print("[phase=regen_waves] import-check OK", flush=True)
        return 0

    if args.stage is None:
        p.error("--stage is required (it is optional only under --import-check)")
    if args.stage in ("gen", "judge") and args.wave is None:
        p.error(f"--stage {args.stage} requires --wave")
    # The judge/pilot seams read these attrs off the shared namespace
    # (phase_a's _run_judge_pilot contract) — assert presence up front.
    for attr in ("max_tokens", "pilot_n", "judge_draws", "judge_keep_threshold"):
        assert hasattr(args, attr), attr

    try:
        if args.stage == "draw":
            return stage_draw(args)
        if args.stage == "init":
            out_dir = Path(args.output_dir).resolve()
            t_rows = _load_target(out_dir)
            _wave0_init(args, out_dir, [r["conv_id"] for r in t_rows])
            return 0
        if args.stage == "gen":
            return stage_gen(args)
        if args.stage == "judge":
            return stage_judge(args)
        if args.stage == "export":
            return stage_export(args)
        if args.stage == "stage-r2":
            return stage_r2_inputs(args)
        if args.stage == "assist-merge":
            return stage_assist_merge(args)
        raise AssertionError(args.stage)
    except pa.PilotGateRefusal as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 7


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
