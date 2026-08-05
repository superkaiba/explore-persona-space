#!/usr/bin/env python
"""Phase D driver for task #2054: v4-new cell (c) transpose (STORY answer, CHAT presentation).

For each on-policy variant (`char_{helios,wren,dana,vex}_op[_base]`), read the
parent #1345 on-policy paired_op stories from HF
`superkaiba1/explore-persona-space-data/issue1345_framing/{variant}/raw_completions/stories/*stories_paired_op*.jsonl`,
extract the STORY-authored answer span via
`issue1345_gen_stories_paired.confident_op_turn` (the same op-turn parse the
extraction path uses), and render it under the REQUIRED `--form` framing
against the corresponding phase_a scaffold row (`issue2054_forms`). NO NEW
generation — capture-only.

Cell (c) — "answer authored STORY, presented CHAT" — is produced by invoking
this driver with `--form chat` (plan §4 Phase D; the chat render re-frames the
story-authored answer through the chat template, prose dropped). `--form` is
REQUIRED with no default so a caller can never silently fall back to a story
form; the dispatch router pins `--form chat` for the phase_d cell (c) wire.

Match on conv_id (v4's shared conv_id draw is the pairing key), CANONIZED
across the two key spaces the join crosses: phase_a scaffolds inherit
`conv_id = scaffold_id = f"stripped_{story_id}"` from the parent stripper
(`issue1345_strip_scaffolds.strip_file` -> `issue2054_phase_a` setdefault),
while the parent's paired_op story rows key on the bare `story_id` (`s...`).
Both sides — and the shared fold map's membership check — normalize through
`_canon_conv_id` (strip the stripper's `stripped_` prefix), so the join
resolves on production artifacts. Cells whose conv_id is not in the shared
fold map are skipped and reported; the fold map itself is REQUIRED unless
`--no-fold-filter` opts out explicitly (a missing map fails loud, never a
silently-disabled membership check).

Plan §12 assumption #7 (SLOT_SENTINEL literal escape): before splicing, SCAN
each parent answer for the literal string `<<<ANSWER>>>`. On any hit, escape
via a runtime replacement (per plan §8 risk 6) so `splice_answer`'s
sentinel-in-answer refusal cannot fire. The scan hit count is reported in the
per-variant digest.

Emits `[phase=phase_d]` log lines terminating in `[phase=done]` on graceful
completion. Uploads to HF `issue2054_lattice/cell_c/{variant}/` (best-effort,
non-fatal).

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
import issue2054_resume as resume  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PARENT_PREFIX = "issue1345_framing"
TASK_PREFIX = "issue2054_lattice"

# The eight on-policy variants of the v4 character panel (parent's
# ONPOLICY_STORY_VARIANTS minus the assistant scope). The `_op`/`_op_base`
# tail selects the parent capture model (instruct vs pretrained); both share
# the SAME base character scaffold set (phase_a wrote scaffolds under
# `char_helios` etc. — no `_op` in the scaffold variant name).
DEFAULT_VARIANTS = (
    "char_helios_op",
    "char_helios_op_base",
    "char_wren_op",
    "char_wren_op_base",
    "char_dana_op",
    "char_dana_op_base",
    "char_vex_op",
    "char_vex_op_base",
)

_CHAR_NAME_FROM_VARIANT = {
    "char_helios": "Helios",
    "char_wren": "Wren",
    "char_dana": "Dana",
    "char_vex": "Vex",
}


def _log(msg: str) -> None:
    print(f"[phase=phase_d] {msg}", flush=True)


def _rel(path: Path) -> str:
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


def _base_variant_for(op_variant: str) -> str:
    """Strip `_op` / `_op_base` tail -> the phase_a scaffold variant name."""
    for tail in ("_op_base", "_op"):
        if op_variant.endswith(tail):
            return op_variant[: -len(tail)]
    return op_variant


def _char_name_from_variant(op_variant: str) -> str:
    """Character name for an on-policy variant. Fail-loud on an unknown base
    variant (M3): the silent "ARIA" default made `_extract_op_answer` parse
    parent stories under the WRONG character name — 0 answers, silently."""
    base = _base_variant_for(op_variant)
    mapped = _CHAR_NAME_FROM_VARIANT.get(base)
    if mapped is None:
        raise ValueError(
            f"cannot resolve character name: variant {op_variant!r} (base {base!r}) "
            "is not in _CHAR_NAME_FROM_VARIANT — extend the map for a new variant"
        )
    return mapped


def _canon_conv_id(conv_id: str) -> str:
    """Map a conv id into the PARENT-story key space (the phase_d join key).

    Production scaffold conv_ids read `stripped_<story_id>` (the parent
    stripper emits `scaffold_id = f"stripped_{sid}"`, `issue1345_strip_
    scaffolds.strip_file`; phase_a then sets `conv_id = scaffold_id`), while
    the parent paired_op story rows key on the bare `story_id`. Strip the
    stripper's prefix; a bare parent id passes through unchanged.
    """
    return conv_id.removeprefix("stripped_")


def _load_shared_fold_map(fold_map_path: Path) -> set[str]:
    """Load conv_id membership of the shared fold map, in CANONICAL key space.

    Fail-loud: a missing or malformed fold map raises / errors at the caller
    (the only sanctioned opt-out is the explicit `--no-fold-filter` flag) —
    never a silently-disabled membership check. Keys canonize through
    `_canon_conv_id` so the committed scaffold-space keys (`stripped_s...`)
    and bare parent-space keys both resolve.
    """
    payload = json.loads(fold_map_path.read_text(encoding="utf-8"))
    fold_of = payload.get("fold_of")
    if not isinstance(fold_of, dict) or not fold_of:
        raise ValueError(f"fold map {fold_map_path} has no non-empty 'fold_of' dict")
    return {_canon_conv_id(str(k)) for k in fold_of}


def _list_parent_story_files(api, variant: str) -> list[str]:
    """List parent's paired_op story JSONLs for one variant (server-side scoped)."""
    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    prefix = f"{PARENT_PREFIX}/{variant}/raw_completions/stories"
    all_paths = list_hf_files_under_path(api, HF_DATA_REPO, prefix, repo_type="dataset")
    # Keep only paired_op stories JSONLs (mode_slug=paired_op; the
    # story-side JSONL name carries "_stories_paired_op").
    return [
        p for p in all_paths if p.endswith(".jsonl") and "stories_paired_op" in p.rsplit("/", 1)[-1]
    ]


def _download_parent_story(path_in_repo: str) -> Path | None:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    try:
        local = retry_transient(
            lambda: hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=path_in_repo,
            ),
            what=f"hf_hub_download({path_in_repo})",
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN download {path_in_repo} failed: {exc}")
        return None
    return Path(local)


def _extract_op_answer(story: str, char_name: str) -> str | None:
    """Return `story[a_start:a_end]` for the confident on-policy turn, per char_name.

    Replicates `issue1345_gen_stories_paired.confident_op_turn`'s check
    (exactly one parsed turn with marker_exact OR answer_len_ok) but routes
    the parse through `sc.parse_story_turns_for(story, char_name)` — the
    scoped helper that swaps the module-global ANSWER_ATTRIB_RE for one
    parse call. This avoids the EPM_STORY_CHARACTER_NAME / EPM_I1345_VARIANT
    import-time pairing guard (issue1345_common.py:79) that fires when
    EPM_STORY_CHARACTER_NAME is set at process start without the paired
    variant slug.
    """
    turns = sc.parse_story_turns_for(story, char_name)
    if len(turns) != 1:
        return None
    t = turns[0]
    conf = t.get("confidence") or {}
    if not (conf.get("marker_exact") or conf.get("answer_len_ok")):
        return None
    a_start = int(t.get("a_start") or -1)
    a_end = int(t.get("a_end") or -1)
    if a_start < 0 or a_end <= a_start or a_end > len(story):
        return None
    return story[a_start:a_end]


def _escape_slot_sentinel(answer: str) -> tuple[str, int]:
    """Runtime replacement of the SLOT_SENTINEL literal (plan §12 #7 / §8 risk 6).

    Returns (escaped_answer, hit_count). `splice_answer` refuses an answer
    that contains SLOT_SENTINEL (a zero-width span invariant), so any parent
    answer carrying the sentinel literal would drop by refusal — the escape
    substitutes a visually-distinct placeholder while preserving reading
    order. Reported per row.
    """
    if sc.SLOT_SENTINEL not in answer:
        return answer, 0
    hits = answer.count(sc.SLOT_SENTINEL)
    escaped = answer.replace(sc.SLOT_SENTINEL, "<<<answer>>>")
    return escaped, hits


def _index_parent_answers(story_files: list[str], char_name: str) -> dict[str, str]:
    """Read every paired_op file, extract `story[a_start:a_end]`, key on the
    CANONICAL conv id (`_canon_conv_id` — parent-story key space)."""
    answers_by_conv: dict[str, str] = {}
    n_rows_seen = 0
    n_op_extract_ok = 0
    for path_in_repo in story_files:
        local = _download_parent_story(path_in_repo)
        if local is None:
            continue
        rows = _read_jsonl(local)
        for row in rows:
            n_rows_seen += 1
            story = row.get("story")
            conv_id = row.get("conv_id") or row.get("story_id")
            if not isinstance(story, str) or not conv_id:
                continue
            answer = _extract_op_answer(story, char_name)
            if answer is None:
                continue
            n_op_extract_ok += 1
            answers_by_conv[_canon_conv_id(str(conv_id))] = answer
    _log(
        f"parent parse: seen={n_rows_seen} op_ok={n_op_extract_ok} unique_conv={len(answers_by_conv)}"
    )
    return answers_by_conv


def _splice_one(
    scaffold_row: dict, answer: str, char_name: str, variant: str, form: str
) -> dict | None:
    """Render the STORY-authored answer under `form` (cell (c) uses `chat`)."""
    scaffold = scaffold_row.get("scaffold_text")
    if not isinstance(scaffold, str):
        return None
    if form in forms.STORY_FORMS and sc.SLOT_SENTINEL not in scaffold:
        return None
    attrib_template = scaffold_row.get("attrib_template")
    try:
        result = forms.splice_answer_form(
            scaffold_row,
            answer,
            form,
            char_name,
            attrib_template=attrib_template if isinstance(attrib_template, str) else None,
        )
    except (ValueError, NotImplementedError) as exc:
        _log(f"splice skip {scaffold_row.get('scaffold_id')}: {exc}")
        return None
    return {
        "scaffold_id": scaffold_row.get("scaffold_id"),
        "conv_id": str(scaffold_row.get("conv_id") or scaffold_row.get("scaffold_id") or ""),
        "variant": variant,
        "character": char_name,
        "form": result.form,
        "final_text": result.text,
        "answer": answer,
        "answer_start": result.answer_start,
        "answer_end": result.answer_end,
        "answer_len_chars": len(answer),
        "prefix_end_char": result.prefix_end_char,
    }


def _process_variant(
    variant: str,
    scaffolds_root: Path,
    out_dir: Path,
    api,
    fold_conv_ids: set[str] | None,
    target_conv_ids: int,
    form: str,
    *,
    fold_map_sha: str | None,
    overwrite: bool,
) -> dict:
    """Splice parent's on-policy answers into scaffolds for one variant.

    Resume (C9/M6): a variant whose output + regime-matching done sidecar
    already exist is SKIPPED (incl. the parent-story HF downloads); a CHANGED
    scaffolds input / fold map recomputes with a log line; a DIFFERENT regime
    refuses (RegimeMismatch). Parent #1345 story data is frozen and is NOT
    part of the input pin.
    """
    base_variant = _base_variant_for(variant)
    scaffolds_path = scaffolds_root / base_variant / f"scaffolds_{base_variant}.jsonl"
    if not scaffolds_path.is_file():
        # Smoke fallback: any *.jsonl under scaffolds_root/<variant>/ (or root).
        alt = (
            list((scaffolds_root / base_variant).glob("*.jsonl"))
            if (scaffolds_root / base_variant).is_dir()
            else []
        )
        if not alt:
            alt = sorted(scaffolds_root.glob("*.jsonl"))
        if not alt:
            _log(f"variant={variant} no scaffolds found under {scaffolds_root}")
            return {"variant": variant, "n_in": 0, "n_out": 0, "out_path": None}
        scaffolds_path = alt[0]

    vdir_early = out_dir / variant
    vdir_early.mkdir(parents=True, exist_ok=True)
    early_out_path = vdir_early / forms.phase_output_name("cell_c", variant, form)
    regime = {
        "cell": forms.cell_key(variant, "cell_c", form, "any"),
        "target_conv_ids": int(target_conv_ids),
        "fold_filter": fold_conv_ids is not None,
    }
    inputs = {
        "scaffolds_sha256": resume.file_sha256(scaffolds_path),
        "fold_map_sha256": fold_map_sha,
    }
    disposition, reason = resume.resume_disposition(
        early_out_path, regime, inputs, overwrite=overwrite
    )
    if disposition == resume.SKIP:
        done = resume.read_done(early_out_path) or {}
        extra = done.get("extra") or {}
        _log(f"variant={variant} RESUME skip ({reason}) -> {_rel(early_out_path)}")
        return {
            "variant": variant,
            "status": "resumed",
            "n_in": int(extra.get("n_in") or 0),
            "n_out": int(extra.get("n_out") or 0),
            "out_path": early_out_path,
        }
    if disposition == resume.RECOMPUTE:
        _log(f"variant={variant} recompute: {reason}")

    scaffolds = _read_jsonl(scaffolds_path)
    if target_conv_ids > 0:
        scaffolds = scaffolds[:target_conv_ids]

    char_name_default = _char_name_from_variant(variant)
    story_files = _list_parent_story_files(api, variant)
    _log(f"variant={variant} parent story files: {len(story_files)}")
    answers_by_conv = _index_parent_answers(story_files, char_name_default) if story_files else {}

    vdir = out_dir / variant
    vdir.mkdir(parents=True, exist_ok=True)
    # Form-aware name (C6): two --form runs of one variant must not clobber.
    out_path = vdir / forms.phase_output_name("cell_c", variant, form)

    n_in = len(scaffolds)
    n_out = 0
    n_no_answer = 0
    n_out_of_fold = 0
    n_sentinel_escaped_rows = 0
    n_sentinel_hits_total = 0

    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in scaffolds:
            conv_id = str(row.get("conv_id") or row.get("scaffold_id") or "")
            if not conv_id:
                continue
            # Canonical (parent-space) key drives BOTH the fold-membership
            # check and the parent-answer join; the emitted row keeps the raw
            # scaffold conv_id (cross-cell conv matching with phase_b/c uses
            # the scaffold key space) plus `parent_conv_id` for provenance.
            conv_key = _canon_conv_id(conv_id)
            if fold_conv_ids is not None and conv_key not in fold_conv_ids:
                n_out_of_fold += 1
                continue
            answer = answers_by_conv.get(conv_key)
            if not answer:
                n_no_answer += 1
                continue
            escaped, hits = _escape_slot_sentinel(answer)
            if hits > 0:
                n_sentinel_escaped_rows += 1
                n_sentinel_hits_total += hits
            char_name = str(row.get("character") or "").strip() or char_name_default
            spliced = _splice_one(row, escaped, char_name, variant, form)
            if spliced is None:
                continue
            spliced["parent_conv_id"] = conv_key
            f.write(json.dumps(spliced, ensure_ascii=False) + "\n")
            n_out += 1
    os.replace(tmp, out_path)
    if n_out > 0:
        resume.write_done(out_path, regime, inputs, extra={"n_in": n_in, "n_out": n_out})

    if n_in > 0 and answers_by_conv and n_out == 0:
        # Zero-join diagnostic: name sample keys from BOTH sides so a future
        # key-space drift (the C5 class) is diagnosable from the log alone.
        scaffold_keys = [
            _canon_conv_id(str(r.get("conv_id") or r.get("scaffold_id") or ""))
            for r in scaffolds[:3]
        ]
        parent_keys = list(answers_by_conv)[:3]
        _log(
            f"variant={variant} ZERO-JOIN diagnostic: scaffold conv keys "
            f"(canonical) sample={scaffold_keys} parent keys sample={parent_keys}"
        )

    _log(
        f"variant={variant} spliced={n_out}/{n_in} "
        f"no_answer={n_no_answer} out_of_fold={n_out_of_fold} "
        f"sentinel_escapes={n_sentinel_escaped_rows} sentinel_hits={n_sentinel_hits_total} "
        f"-> {_rel(out_path)}"
    )
    return {
        "variant": variant,
        "n_in": n_in,
        "n_out": n_out,
        "n_no_answer": n_no_answer,
        "n_out_of_fold": n_out_of_fold,
        "sentinel_escaped_rows": n_sentinel_escaped_rows,
        "sentinel_hits_total": n_sentinel_hits_total,
        "n_parent_answers": len(answers_by_conv),
        "n_parent_story_files": len(story_files),
        "out_path": out_path,
    }


def _upload_to_hf(paths_by_variant: dict[str, Path], out_dir: Path) -> None:
    """Mirror cell_c JSONLs — ONE bulk upload_folder commit. FATAL on failure
    (M2): a swallowed upload + `[phase=done]` silently strands the splices."""
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
        expected_paths.append(f"{TASK_PREFIX}/cell_c/{rel}")
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
        path_in_repo=f"{TASK_PREFIX}/cell_c",
        allow_patterns=allow_patterns,
        expected_repo_paths=expected_paths,
    )
    _log(f"uploaded {len(allow_patterns)} cell_c file(s) in one bulk commit")


def run_phase(args: argparse.Namespace) -> int:
    scaffolds_root = Path(args.scaffolds_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not scaffolds_root.exists():
        print(f"ERROR: scaffolds-dir does not exist: {scaffolds_root}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = list(args.variants)
    _log(f"start: target_conv_ids={args.target_conv_ids} form={args.form} variants={variants}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # noqa: BLE001
        print(f"ERROR: huggingface_hub missing: {exc}", file=sys.stderr)
        return 2
    api = HfApi()

    fold_map_sha: str | None = None
    if args.no_fold_filter:
        fold_conv_ids = None
        _log("shared fold map filter disabled by --no-fold-filter")
    else:
        fold_map_path = Path(args.fold_map).resolve()
        if not fold_map_path.is_file():
            print(
                f"ERROR: shared fold map not found: {fold_map_path} "
                "(run phase_a first, or pass --no-fold-filter for a smoke)",
                file=sys.stderr,
            )
            return 1
        fold_conv_ids = _load_shared_fold_map(fold_map_path)
        fold_map_sha = resume.file_sha256(fold_map_path)
        _log(f"shared fold map loaded: {len(fold_conv_ids)} conv_ids (canonical key space)")

    per_variant_reports: list[dict] = []
    out_paths: dict[str, Path] = {}
    for variant in variants:
        report = _process_variant(
            variant,
            scaffolds_root,
            out_dir,
            api,
            fold_conv_ids,
            args.target_conv_ids,
            args.form,
            fold_map_sha=fold_map_sha,
            overwrite=args.overwrite,
        )
        per_variant_reports.append(report)
        if report.get("out_path") is not None:
            out_paths[variant] = report["out_path"]

    total_out = sum(int(r.get("n_out") or 0) for r in per_variant_reports)
    if total_out == 0:
        print("ERROR: phase_d produced ZERO spliced rows", file=sys.stderr)
        return 1

    is_smoke = str(out_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload:
        # FATAL on failure (M2): `[phase=done]` must never report done with
        # the cell (c) splices un-persisted. No try/except.
        _upload_to_hf(out_paths, out_dir)

    digest = {
        "phase": "phase_d",
        "form": args.form,
        "fold_map": None if args.no_fold_filter else str(Path(args.fold_map).resolve()),
        "target_conv_ids": args.target_conv_ids,
        "variants": variants,
        "per_variant": [
            {k: (str(_rel(v)) if isinstance(v, Path) else v) for k, v in r.items()}
            for r in per_variant_reports
        ],
        "n_total_out": total_out,
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    # Form-keyed digest name (C6): the digest is per (condition, form) run.
    digest_path = out_dir / f"phase_d_digest{forms.CELL_KEY_SEP}{args.form}.json"
    tmp = digest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, sort_keys=True)
    os.replace(tmp, digest_path)

    print(f"[phase=phase_d] digest: n_total_out={total_out}", flush=True)
    # noqa: phase-done-reserved
    print("[phase=done]", flush=True)
    sys.stdout.flush()
    sys.exit(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scaffolds-dir", default="data/issue_2054/scaffolds/")
    p.add_argument(
        "--form",
        required=True,
        choices=forms.FORMS,
        help=(
            "framing to render (plan §4; REQUIRED, no default). Cell (c) — "
            "story-authored answer presented in CHAT — uses `chat` (plan §4 "
            "Phase D); the dispatch wires that choice explicitly"
        ),
    )
    p.add_argument("--target-conv-ids", type=int, default=8_000)
    p.add_argument("--output-dir", default="data/issue_2054/cell_c/")
    p.add_argument(
        "--fold-map",
        default=str(_REPO_ROOT / "eval_results/issue_2054/shared_fold_map.json"),
        help=(
            "shared fold map JSON from phase_a (fold_of keys define conv_id "
            "membership; keys canonize through _canon_conv_id). REQUIRED to "
            "exist unless --no-fold-filter is passed"
        ),
    )
    p.add_argument("--seed", type=int, default=137)
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_VARIANTS),
    )
    p.add_argument("--skip-upload", action="store_true", help="skip HF mirror step")
    p.add_argument(
        "--upload", action="store_true", help="force HF mirror step (default when not smoke)"
    )
    p.add_argument(
        "--no-fold-filter",
        action="store_true",
        help="disable the shared fold-map conv_id membership check (smoke use only)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "re-splice variants even when a regime-matching done sidecar exists "
            "(default resumes completed variants — C9/M6)"
        ),
    )
    args = p.parse_args()
    try:
        return run_phase(args)
    except resume.RegimeMismatch as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
