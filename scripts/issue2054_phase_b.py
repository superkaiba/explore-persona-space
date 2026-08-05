#!/usr/bin/env python
"""Phase B driver for task #2054: deterministic inserted splice.

For each scaffold under `--scaffolds-dir`, render the REQUIRED `--form`
framing (plan §4 "Framings" — the lattice's central manipulated variable; no
default, argparse refuses a form-less invocation): story forms splice at the
`<<<ANSWER>>>` slot via the parent's `splice_answer` (100% keep by
construction; span offsets known by construction), and the chat / bare_text
framings re-frame the scaffold's question + answer through
`issue2054_forms` (narrative prose dropped — chat/bare are structurally
assistant-only per plan §4 Cells). The answer pool comes from
`--answers-source` — a JSONL of {conv_id, answer} or {scaffold_id,
answer} rows (sanctioned, chat-authored — the parent's inserted-arm answers).

Writes spliced texts + exact answer-span offsets per row under
`data/issue_2054/spliced_inserted/{variant}/` and mirrors to HF
`issue2054_lattice/spliced_inserted/{variant}/` (best-effort, non-fatal).

Emits `[phase=phase_b]` log lines terminating in `[phase=done]` on graceful
completion.

Exit 0 on success. Exit 1 on splice / HF / preflight failure. Exit 2 on
missing dependency.
"""

from __future__ import annotations

import argparse
import json
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

import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_forms as forms  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
TASK_PREFIX = "issue2054_lattice"


def _log(msg: str) -> None:
    print(f"[phase=phase_b] {msg}", flush=True)


def _rel(path: Path) -> str:
    """Best-effort repo-root-relative string; falls back to abs for /tmp/*."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader; undecodable lines are COUNTED + warned (M3 —
    never silently skipped)."""
    rows: list[dict] = []
    n_bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        _log(f"WARN {n_bad} undecodable JSONL line(s) skipped in {path}")
    return rows


def _index_answers(rows: list[dict]) -> dict[str, str]:
    """Return {conv_id-or-scaffold_id → answer text}."""
    out: dict[str, str] = {}
    for r in rows:
        key = r.get("conv_id") or r.get("scaffold_id")
        if key is None:
            continue
        answer = r.get("answer") or r.get("target") or r.get("story_answer") or ""
        if not answer:
            continue
        out[str(key)] = str(answer)
    return out


def _char_name_from_scaffold_row(row: dict, variant: str) -> str:
    """Recover the character name for the splice attrib template.

    Parent's stripper wrote `character` on every row; fall back to variant
    lookup so byte-exact round-trip splices still work on legacy rows.
    Fail-loud on an unknown variant with no row-level `character` (M3 — the
    silent "ARIA" default corrupted splices for unmapped variants)."""
    ch = str(row.get("character") or "").strip()
    if ch:
        return ch
    from_variant = {
        "char_helios": "Helios",
        "char_wren": "Wren",
        "char_dana": "Dana",
        "char_vex": "Vex",
        "conversation_paired_stories_assistant": "Assistant",
        "conversation_paired_stories": "ARIA",
    }
    mapped = from_variant.get(variant)
    if mapped is None:
        raise ValueError(
            f"cannot resolve character name: variant {variant!r} is not in the "
            f"variant map and row {row.get('scaffold_id')!r} carries no 'character' field"
        )
    return mapped


def _splice_one(row: dict, answer: str, variant: str, form: str) -> dict | None:
    scaffold = row.get("scaffold_text")
    if not isinstance(scaffold, str):
        return None
    if form in forms.STORY_FORMS and sc.SLOT_SENTINEL not in scaffold:
        return None
    char_name = _char_name_from_scaffold_row(row, variant)
    attrib_template = row.get("attrib_template")  # stripper-recorded template
    try:
        result = forms.splice_answer_form(
            row,
            answer,
            form,
            char_name,
            attrib_template=attrib_template if isinstance(attrib_template, str) else None,
        )
    except (ValueError, NotImplementedError) as exc:
        _log(f"splice skip {row.get('scaffold_id')}: {exc}")
        return None
    # Guard: the spliced offsets identify the exact answer bytes (asserted in
    # splice_answer / splice_answer_form). Persist the whole row.
    return {
        "scaffold_id": row.get("scaffold_id"),
        "conv_id": row.get("conv_id") or row.get("scaffold_id"),
        "variant": variant,
        "character": char_name,
        "form": result.form,
        "final_text": result.text,
        "answer": answer,
        "answer_start": result.answer_start,
        "answer_end": result.answer_end,
        "prefix_end_char": result.prefix_end_char,
    }


def _process_variant(
    variant: str,
    scaffolds_path: Path,
    answers_by_key: dict[str, str],
    out_dir: Path,
    form: str,
) -> tuple[dict, Path]:
    """Splice every scaffold whose conv_id has an answer.

    Returns (counts, out_path); counts carries the per-variant answer-source
    split (M3): n_answer_from_pool vs n_answer_from_scaffold_fallback.
    """
    scaffolds = _read_jsonl(scaffolds_path)
    n_in = len(scaffolds)
    vdir = out_dir / variant
    vdir.mkdir(parents=True, exist_ok=True)
    # Form-aware name (C6): two --form runs of one variant must not clobber.
    out_path = vdir / forms.phase_output_name("inserted", variant, form)

    tmp = out_path.with_suffix(".jsonl.tmp")
    n_out = 0
    n_from_pool = 0
    n_from_scaffold_fallback = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in scaffolds:
            key = str(row.get("conv_id") or row.get("scaffold_id") or "")
            if not key:
                continue
            answer = answers_by_key.get(key)
            # Per-row answer provenance (M3): the 2x2 authorship axis rides on
            # WHO wrote the spliced answer — a pool miss falling back to the
            # scaffold's own recovered original answer is a MIXED-authorship
            # cell unless counted + recorded per row.
            answer_source = "answers_pool"
            if not answer:
                # Fall back to the scaffold's own recorded original answer
                # (stripper preserves it) — the strict "brought by the user"
                # answer pool wins whenever it has a hit.
                answer = str(row.get("answer") or "")
                answer_source = "scaffold_original_fallback"
            if not answer:
                continue
            spliced = _splice_one(row, answer, variant, form)
            if spliced is None:
                continue
            spliced["answer_source"] = answer_source
            if answer_source == "answers_pool":
                n_from_pool += 1
            else:
                n_from_scaffold_fallback += 1
            f.write(json.dumps(spliced, ensure_ascii=False) + "\n")
            n_out += 1
    os.replace(tmp, out_path)
    if n_from_scaffold_fallback:
        frac = n_from_scaffold_fallback / max(1, n_out)
        _log(
            f"WARN variant={variant} {n_from_scaffold_fallback}/{n_out} rows "
            f"({frac:.3f}) fell back to the scaffold's own answer (pool miss) — "
            "mixed-authorship cell; per-row answer_source records the split"
        )
    counts = {
        "n_in": n_in,
        "n_out": n_out,
        "n_answer_from_pool": n_from_pool,
        "n_answer_from_scaffold_fallback": n_from_scaffold_fallback,
    }
    return counts, out_path


def _upload_to_hf(paths_by_variant: dict[str, Path], out_dir: Path) -> None:
    """Mirror spliced JSONLs — ONE bulk upload_folder commit. FATAL on failure
    (M2): a swallowed upload + `[phase=done]` silently strands the splices.

    Per the #664/#1481 per-file storm class, batch all variants into a single
    `_upload_folder_filtered` commit against the shared `<TASK_PREFIX>/spliced_inserted`
    prefix instead of per-file `_upload` calls.
    """
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    allow_patterns: list[str] = []
    expected_paths: list[str] = []
    for variant, p in paths_by_variant.items():
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(out_dir).as_posix()
        except ValueError:
            continue
        allow_patterns.append(rel)
        expected_paths.append(f"{TASK_PREFIX}/spliced_inserted/{rel}")
    if not allow_patterns:
        if paths_by_variant:
            raise RuntimeError(
                f"upload set resolved EMPTY against declared outputs: {paths_by_variant}"
            )
        return
    _upload_folder_filtered(
        out_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{TASK_PREFIX}/spliced_inserted",
        allow_patterns=allow_patterns,
        expected_repo_paths=expected_paths,
    )
    _log(f"uploaded {len(allow_patterns)} spliced file(s) in one bulk commit")


def run_phase(args: argparse.Namespace) -> int:
    scaffolds_root = Path(args.scaffolds_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    answers_path = Path(args.answers_source).resolve()

    if not scaffolds_root.exists():
        print(f"ERROR: scaffolds-dir does not exist: {scaffolds_root}", file=sys.stderr)
        return 1
    if not answers_path.is_file():
        print(f"ERROR: answers-source not readable: {answers_path}", file=sys.stderr)
        return 1

    _log(f"start: scaffolds_root={scaffolds_root} answers={answers_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    answers_by_key = _index_answers(_read_jsonl(answers_path))
    _log(f"loaded {len(answers_by_key)} answers from {answers_path.name}")

    # Enumerate per-variant scaffold files (mirrors phase_a's layout:
    # <scaffolds_root>/<variant>/scaffolds_<variant>.jsonl).
    per_variant_paths: dict[str, Path] = {}
    for child in sorted(scaffolds_root.iterdir()):
        if not child.is_dir():
            continue
        variant = child.name
        candidate = child / f"scaffolds_{variant}.jsonl"
        if candidate.is_file():
            per_variant_paths[variant] = candidate

    if not per_variant_paths:
        # Fall back: accept any *.jsonl directly under the root as a single
        # untyped variant (smoke fixture shape).
        stray = sorted(scaffolds_root.glob("*.jsonl"))
        if stray:
            per_variant_paths["_flat"] = stray[0]

    if not per_variant_paths:
        print(
            f"ERROR: no scaffold JSONLs found under {scaffolds_root}",
            file=sys.stderr,
        )
        return 1

    out_paths: dict[str, Path] = {}
    counts: dict[str, dict] = {}
    for variant, sp in per_variant_paths.items():
        vcounts, op = _process_variant(variant, sp, answers_by_key, out_dir, args.form)
        counts[variant] = vcounts
        out_paths[variant] = op
        _log(
            f"variant={variant} spliced {vcounts['n_out']}/{vcounts['n_in']} "
            f"(pool={vcounts['n_answer_from_pool']} "
            f"fallback={vcounts['n_answer_from_scaffold_fallback']}) -> {_rel(op)}"
        )

    total_out = sum(c["n_out"] for c in counts.values())
    if total_out == 0:
        print("ERROR: phase_b produced ZERO spliced rows", file=sys.stderr)
        return 1

    # HF upload — FATAL on failure (M2): `[phase=done]` must never report done
    # with the splices un-persisted. Skipped when the output tree is a smoke
    # tmp dir (never mirror /tmp outputs to the shared data repo).
    is_smoke = str(out_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload:
        _upload_to_hf(out_paths, out_dir)

    digest = {
        "phase": "phase_b",
        "form": args.form,
        "counts": counts,
        "n_total_out": total_out,
        "n_answers_loaded": len(answers_by_key),
        "out_paths": {v: _rel(p) for v, p in out_paths.items()},
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    # Form-keyed digest name (C6): the digest is per (condition, form) run.
    digest_path = out_dir / f"phase_b_digest{forms.CELL_KEY_SEP}{args.form}.json"
    tmp = digest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, sort_keys=True)
    os.replace(tmp, digest_path)

    print(f"[phase=phase_b] digest: n_total_out={total_out}", flush=True)
    # noqa: phase-done-reserved
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.exit(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scaffolds-dir", default="data/issue_2054/scaffolds/")
    p.add_argument("--answers-source", required=True, help="JSONL with {conv_id, answer} rows")
    p.add_argument(
        "--form",
        required=True,
        choices=forms.FORMS,
        help=(
            "framing to render (plan §4 — the lattice's central manipulated "
            "variable; REQUIRED, no default so a caller can never silently "
            "fall back to attrib_quoted)"
        ),
    )
    p.add_argument("--output-dir", default="data/issue_2054/spliced_inserted/")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--skip-upload", action="store_true", help="skip HF mirror step")
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
