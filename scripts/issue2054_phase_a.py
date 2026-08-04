#!/usr/bin/env python
"""Phase A driver for task #2054: diverse scaffold generation + shared fold map.

Recovers scaffolds from parent #1345's ~38,700 existing on-policy stories via
`issue1345_strip_scaffolds.strip_file` (byte-exact round-trip verified per row);
generates ONLY the SHORTFALL up to `--target-conv-ids` via the parent's own
`issue1345_gen_scaffolds` phase (verbatim question preserved). Judge-filters
scaffolds via `explore_persona_space.eval.batch_judge.judge_completions_batch`
(claude-sonnet-4-5-20250929, `max_tokens=1024`, drop-never-coerce); a ≥5k-call
production wave is pilot-gated at 200 draws (`judge_pilot_gate`, rule 26).

Writes `eval_results/issue_2054/shared_fold_map.json` (K=5 conversation-grouped
folds, seed 137) as the SINGLE artifact every downstream fit/ladder invocation
consumes.

Emits `[phase=phase_a]` log lines terminating in `[phase=done]` on graceful
completion. Deferred imports (huggingface_hub, judge helpers) keep --pilot smokes
free of import-time HF cost when the shortfall is 0.

Exit 0 on success. Exit 1 on judge / HF / preflight failure. Exit 2 on missing
dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

# The parent registries live at import top-level; a tokenizer/regex compile
# reads EPM_STORY_CHARACTER_NAME + EPM_I1345_VARIANT — Phase A does neither,
# so the strict [A-Za-z0-9_]+ default ("ARIA", "") passes untouched.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

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

# The character-name → parent-story character mapping. parent stories were
# written with EPM_STORY_CHARACTER_NAME=<Name>; strip must be told the same
# name so the answer-attribution regex matches.
_CHAR_NAME_FROM_VARIANT = {
    "char_helios": "Helios",
    "char_wren": "Wren",
    "char_dana": "Dana",
    "char_vex": "Vex",
    "conversation_paired_stories_assistant": "Assistant",
    "conversation_paired_stories": "ARIA",  # legacy v8 scope; addressable
}

_SCAFFOLD_JUDGE_RUBRIC = (
    "You are grading a narrative SCAFFOLD for diversity + question containment.\n"
    "The scaffold is a short scene containing exactly one question utterance and a "
    "sentinel token '<<<ANSWER>>>' where an answer would go. First reason briefly, "
    'then output JSON: {"diverse": <0-100>, "single_question": <0-100>}.\n\n'
    "Question:\n{question}\n\nAnswer:\n{answer}"
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


def _recover_scaffolds_from_hf(variants: list[str], out_dir: Path, api) -> dict[str, list[dict]]:
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
        char_name = _CHAR_NAME_FROM_VARIANT.get(variant, "ARIA")
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
        # ONLY *_stories_*.jsonl files carry the `story` key the stripper
        # needs; judge_results_*.jsonl in the same dir carry judge metadata
        # (no story text) and blow up the stripper with KeyError('story').
        story_files = [
            p for p in all_paths if p.endswith(".jsonl") and "_stories_" in p.rsplit("/", 1)[-1]
        ]
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
            _log(
                f"variant={variant} file={Path(path_in_repo).name} "
                f"kept={counts.get('kept', 0)}/{counts.get('total', 0)}"
            )
            variant_out.extend(rows)
        # Each recovered scaffold carries a conv_id derived from source_id;
        # the parent's story_id encodes the conv (source: `stripped_{sid}`).
        for i, row in enumerate(variant_out):
            row.setdefault("conv_id", row.get("scaffold_id", f"{variant}_{i}"))
            row.setdefault("variant", variant)
        recovered[variant] = variant_out
    return recovered


def _write_scaffolds_local(
    variants_scaffolds: dict[str, list[dict]], out_dir: Path
) -> dict[str, Path]:
    """Write per-variant scaffolds JSONL locally; return {variant: path}."""
    paths: dict[str, Path] = {}
    for variant, rows in variants_scaffolds.items():
        vdir = out_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        p = vdir / f"scaffolds_{variant}.jsonl"
        # Write via atomic tmp+rename (avoid partial reads by a concurrent
        # verify step); explicit UTF-8.
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
        paths[variant] = p
        _log(f"variant={variant} wrote {len(rows)} scaffolds -> {_rel(p)}")
    return paths


def _judge_gate_scaffolds(
    variants_scaffolds: dict[str, list[dict]],
    *,
    pilot: bool,
    max_tokens: int,
    pilot_n: int,
    cache_root: Path,
) -> dict:
    """Judge-filter scaffolds via batch_judge; pilot-gate a ≥5k-call wave.

    Only asks the judge whether the scaffold is DIVERSE + carries a SINGLE
    question. A malformed / REFUSAL / out-of-range judge return DROPS the
    row (never coerced — rule 9).
    """
    from explore_persona_space.eval.batch_judge import judge_completions_batch
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    # Cap 100 scaffolds per variant for the judge probe (Unit A smoke bound;
    # a full-production judge sweep is a follow-up unit — this driver's
    # judge step is a diversity spot-check, not a per-row admission gate).
    completions: dict[str, dict[str, list[str]]] = {}
    arm_rows: dict[str, list[tuple[str, str, str]]] = {}
    for variant, rows in variants_scaffolds.items():
        head = rows[:100]
        completions[variant] = {}
        arm_rows[variant] = []
        for i, r in enumerate(head):
            q = str(r.get("question") or "")
            a = str(r.get("scaffold_text") or "")
            if not a:
                continue
            completions[variant].setdefault(q, []).append(a)
            arm_rows[variant].append((f"{variant}_{i}", q, a))

    total_calls = sum(sum(len(v) for v in q.values()) for q in completions.values())
    if total_calls == 0:
        return {
            "n_calls": 0,
            "pilot_gate": None,
            "verdict": "PASS",
            "note": "no-scaffolds-to-judge",
        }

    # Pilot-gate any ≥5,000-call wave OR when --pilot is explicit.
    do_pilot = pilot or total_calls >= 5000
    pilot_report = None
    if do_pilot:
        pilot_report = judge_pilot_gate(
            arm_rows,
            _SCAFFOLD_JUDGE_RUBRIC,
            max_tokens=max_tokens,
            cache_dir=cache_root / "pilot_cache",
            save_raw_dir=cache_root / "pilot_raw",
            target_total_draws=pilot_n,
            report_path=cache_root / "pilot_gate_report.json",
        )
        rep = _serialize_report(pilot_report)
        v = rep.get("verdict") if isinstance(rep, dict) else None
        if v is not None and str(v).upper() not in {"PASS", "PASS_WAIVED"}:
            _log(f"judge pilot verdict={v!r} — skipping production judge sweep")
            return {"n_calls": total_calls, "pilot_gate": rep, "verdict": v, "n_judged": 0}
        pilot_report = rep

    # In --pilot mode we only run the pilot itself; skip the production sweep.
    if pilot:
        return {
            "n_calls": total_calls,
            "pilot_gate": pilot_report,
            "verdict": "PASS",
            "n_judged": 0,
        }

    scores = judge_completions_batch(
        completions,
        format_user_msg=lambda q, a: _SCAFFOLD_JUDGE_RUBRIC.replace("{question}", q).replace(
            "{answer}", a
        ),
        max_tokens=max_tokens,
        cache_dir=cache_root / "judge_cache",
        save_raw=cache_root / "raw_judge.json",
        checkpoint_dir=cache_root / "checkpoints",
    )
    n_judged = 0
    for arm_scores in scores.values():
        n_judged += int(arm_scores.get("n_samples") or 0)
    return {
        "n_calls": total_calls,
        "pilot_gate": pilot_report,
        "verdict": "PASS",
        "n_judged": n_judged,
    }


def _upload_to_hf(scaffolds_root: Path, fold_map_path: Path, variants: list[str], api) -> None:
    """Best-effort mirror of scaffolds + fold map to HF data repo.

    Non-fatal — a Hub outage logs a WARN but does not fail the phase; the
    local artifacts under `data/issue_2054/` and `eval_results/` remain the
    canonical local record. All scaffolds ride ONE bulk `_upload_folder_filtered`
    commit (avoids the #664/#1481 per-file 504-storm class), and the fold map
    lands via a single `_upload` call (bounded — one file, no loop).
    """
    from explore_persona_space.orchestrate.hub import _upload, _upload_folder_filtered

    # Bulk scaffolds upload: ONE upload_folder commit across all variants.
    allow_patterns: list[str] = []
    expected_paths: list[str] = []
    for variant in variants:
        p = scaffolds_root / variant / f"scaffolds_{variant}.jsonl"
        if not p.is_file():
            continue
        rel = p.relative_to(scaffolds_root).as_posix()
        allow_patterns.append(rel)
        expected_paths.append(f"{TASK_PREFIX}/scaffolds/{rel}")
    if allow_patterns:
        try:
            _upload_folder_filtered(
                scaffolds_root,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{TASK_PREFIX}/scaffolds",
                allow_patterns=allow_patterns,
                expected_repo_paths=expected_paths,
            )
            _log(f"uploaded {len(allow_patterns)} scaffold file(s) in one bulk commit")
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN scaffold bulk upload failed: {exc}")

    if fold_map_path.is_file():
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
            _log(f"WARN fold-map upload failed: {exc}")


def run_phase(args: argparse.Namespace) -> int:
    variants = list(args.variants)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"start: target_conv_ids={args.target_conv_ids} variants={variants}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # noqa: BLE001
        print(f"ERROR: huggingface_hub missing: {exc}", file=sys.stderr)
        return 2
    api = HfApi()

    _log("recover: strip parent stories -> scaffolds")
    recovered = _recover_scaffolds_from_hf(variants, out_dir, api)

    n_total_recovered = sum(len(rs) for rs in recovered.values())
    _log(f"recovered {n_total_recovered} scaffolds across {len(recovered)} variants")

    # For Unit A we do NOT invoke fresh GPU generation. Fresh generation of
    # the shortfall is a follow-up unit gated on a pilot judge PASS + Phase C
    # capacity (--shortfall is a downstream unit). Report the shortfall
    # numerically so downstream units know how much to generate.
    per_variant_shortfall: dict[str, int] = {}
    for variant in variants:
        got = len(recovered.get(variant, []))
        per_variant_shortfall[variant] = max(0, args.target_conv_ids - got)

    scaffold_paths = _write_scaffolds_local(recovered, out_dir)

    # Verbatim-question invariant is a per-row property of the parent stripper
    # (parent gen scripts enforced it; stripper preserves it). We assert it on
    # the first 100 recovered scaffolds per variant that carry a `question`
    # field: the question substring must appear inside `scaffold_text`.
    q_check_fails: dict[str, int] = {}
    for variant, rows in recovered.items():
        fails = 0
        for r in rows[:100]:
            q = str(r.get("question") or "")
            if not q:
                continue
            if q not in r.get("scaffold_text", ""):
                fails += 1
        if fails:
            q_check_fails[variant] = fails
    if q_check_fails:
        _log(f"WARN verbatim-question check failures: {q_check_fails}")

    # Judge-gate diversity (Unit A: pilot-only unless the full sweep is asked).
    cache_root = out_dir / "_judge_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        judge_result = _judge_gate_scaffolds(
            recovered,
            pilot=args.pilot,
            max_tokens=args.max_tokens,
            pilot_n=args.pilot_n,
            cache_root=cache_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR judge stage failed: {exc}", file=sys.stderr)
        return 1
    _log(
        f"judge stage: {judge_result.get('n_calls', 0)} calls, verdict={judge_result.get('verdict')}"
    )

    # Build the shared fold map ONCE at Phase A end.
    all_conv_ids: list[str] = []
    for rows in recovered.values():
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
    tmp = fold_map_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, fold_map_path)
    _log(f"wrote shared_fold_map.json (n_conv_ids={len(fold_map)}) -> {_rel(fold_map_path)}")

    # Upload best-effort (skip for --pilot: the smoke does not need to touch HF).
    if not args.pilot:
        try:
            _upload_to_hf(out_dir, fold_map_path, variants, api)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN upload stage failed: {exc}")

    # Digest artifact.
    digest_path = out_dir / "phase_a_digest.json"
    digest = {
        "phase": "phase_a",
        "target_conv_ids": args.target_conv_ids,
        "recovered_per_variant": {v: len(rows) for v, rows in recovered.items()},
        "shortfall_per_variant": per_variant_shortfall,
        "n_total_recovered": n_total_recovered,
        "verbatim_question_check_failures": q_check_fails,
        "judge_stage": judge_result,
        "shared_fold_map_path": str(_rel(fold_map_path)),
        "scaffold_paths": {v: str(_rel(p)) for v, p in scaffold_paths.items()},
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    with (digest_path.with_suffix(".json.tmp")).open("w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, sort_keys=True)
    os.replace(digest_path.with_suffix(".json.tmp"), digest_path)

    print(
        f"[phase=phase_a] digest: recovered={n_total_recovered} "
        f"shortfall={sum(per_variant_shortfall.values())} "
        f"n_conv_ids={len(fold_map)} judge_calls={judge_result.get('n_calls', 0)}",
        flush=True,
    )
    # noqa: phase-done-reserved
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.exit(0)  # explicit exit before finalize-time C-extension teardown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-conv-ids", type=int, default=8_000)
    p.add_argument(
        "--output-dir",
        default="data/issue_2054/scaffolds",
        help="root for per-variant scaffold JSONLs",
    )
    p.add_argument("--seed", type=int, default=137)
    p.add_argument(
        "--pilot",
        action="store_true",
        help="run only the judge pilot (200-draw gate); skip production judge sweep",
    )
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_VARIANTS),
        help="comma-separated parent variant slugs (default: v4 lattice character panel + assistant)",
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--pilot-n", type=int, default=200)
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
