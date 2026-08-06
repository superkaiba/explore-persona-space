#!/usr/bin/env python
"""Phase D driver for task #2054: v4-new cell (c) transpose (STORY answer, CHAT presentation).

For each on-policy variant (`char_{helios,wren,dana,vex}_op[_base]`), read
THIS TASK'S Phase-C on-policy story answers from HF
`superkaiba1/explore-persona-space-data/issue2054_lattice/on_policy/{answer_model}/{base_variant}/on_policy_{base_variant}__{answer_form}.jsonl`
(one row per (conv_id, character, model); the `answer` field IS the
story-authored answer span — no story re-parsing needed) and render it under
the REQUIRED `--form` framing against the corresponding phase_a scaffold row
(`issue2054_forms`). NO NEW generation — capture-only.

USER DIRECTIVE (2026-08-06, verbatim: "make sure we only use the new data"):
the answer source is THIS task's fresh on-policy pool, NEVER the parent
#1345 `stories_paired_op` pool (concern
`cell-c-source-tonight-on-policy-not-parent-pool`). The variant's
`_op`/`_op_base` tail selects the ANSWER-provenance model
(`_op` -> qwen2.5-7b-instruct, `_op_base` -> qwen2.5-7b); `--answer-form`
(default `attrib_quoted`, the parent's canonical story boundary form and the
round-1 default) selects which story form's on-policy answers supply the
cell — the (c) row is then byte-matched on answer text with the (d) cell of
the SAME (character, model, answer_form).

Cell (c) — "answer authored STORY, presented CHAT" — is produced by invoking
this driver with `--form chat` (plan §4 Phase D; the chat render re-frames the
story-authored answer through the chat template, prose dropped). `--form` is
REQUIRED with no default so a caller can never silently fall back to a story
form; the dispatch router pins `--form chat` for the phase_d cell (c) wire.

Match on conv_id (v4's shared conv_id draw is the pairing key). The pool rows
inherit their conv_id from the SAME phase_a scaffold rows this driver
iterates, so both sides live in the scaffold key space (`mt...` generator ids
+ `stripped_<story_id>` stripper ids); the join and the shared fold map's
membership check normalize through `_canon_conv_id` (strip the stripper's
`stripped_` prefix). Cells whose conv_id is not in the shared fold map are
skipped and reported; the fold map itself is REQUIRED unless
`--no-fold-filter` opts out explicitly (a missing map fails loud, never a
silently-disabled membership check).

Per-cell realized n is reported against the user floor `--n-floor`
(default 6,250: n_train >= 5,000 at the 0.8 train split, K=5 shared folds).
A below-floor cell is REPORTED (log WARN + digest flag), never silently
passed over.

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
TASK_PREFIX = "issue2054_lattice"

# Answer source: THIS task's Phase-C on-policy pool (model-scoped prefix — the
# model-less top-level `on_policy/{variant}/` files are the stale pre-fix
# copies from the 2026-08-06 last-writer-wins collision; never read those).
ANSWER_SOURCE = f"{TASK_PREFIX}/on_policy"

# The eight on-policy variants of the v4 character panel. The `_op`/`_op_base`
# tail selects the ANSWER-provenance model (instruct vs pretrained) in THIS
# task's Phase-C pool; both share the SAME base character scaffold set
# (phase_a wrote scaffolds under `char_helios` etc. — no `_op` in the
# scaffold variant name, and the pool is keyed on the base variant too).
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

# Variant tail -> Phase-C pool model dir (`issue2054_phase_c.py --model` slugs).
_ANSWER_MODEL_FROM_TAIL = {
    "_op": "qwen2.5-7b-instruct",
    "_op_base": "qwen2.5-7b",
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
    variant (M3): a silent default character name renders the chat/story
    boundary under the WRONG character — the historical "ARIA" default
    produced 0 answers, silently."""
    base = _base_variant_for(op_variant)
    mapped = _CHAR_NAME_FROM_VARIANT.get(base)
    if mapped is None:
        raise ValueError(
            f"cannot resolve character name: variant {op_variant!r} (base {base!r}) "
            "is not in _CHAR_NAME_FROM_VARIANT — extend the map for a new variant"
        )
    return mapped


def _canon_conv_id(conv_id: str) -> str:
    """Map a conv id into the CANONICAL key space (the phase_d join key).

    Stripper-path scaffold conv_ids read `stripped_<story_id>` (the parent
    stripper emits `scaffold_id = f"stripped_{sid}"`, `issue1345_strip_
    scaffolds.strip_file`; phase_a then sets `conv_id = scaffold_id`);
    generator-path ids (`mt...`) pass through unchanged. Both the Phase-C
    pool join and the shared fold-map membership check normalize through
    this helper, so the two sides always meet in one key space.
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


def _answer_model_for(op_variant: str) -> str:
    """Phase-C pool model dir for an on-policy variant's `_op[_base]` tail.

    Fail-loud on a variant with no on-policy tail: cell (c) is defined ONLY
    for the `_op`/`_op_base` variants (the tail is what selects the
    answer-provenance model in THIS task's pool).
    """
    for tail in ("_op_base", "_op"):
        if op_variant.endswith(tail):
            return _ANSWER_MODEL_FROM_TAIL[tail]
    raise ValueError(
        f"variant {op_variant!r} has no _op/_op_base tail — cell (c) requires an "
        "on-policy variant (the tail selects the answer-provenance model)"
    )


def _pool_path_in_repo(variant: str, answer_form: str) -> str:
    """HF path of the Phase-C on-policy pool file supplying this variant's answers."""
    base_variant = _base_variant_for(variant)
    answer_model = _answer_model_for(variant)
    fname = forms.phase_output_name("on_policy", base_variant, answer_form)
    return f"{ANSWER_SOURCE}/{answer_model}/{base_variant}/{fname}"


def _download_pool_file(path_in_repo: str) -> Path:
    """Fetch one Phase-C pool JSONL from HF. FAIL-LOUD: the pool IS the cell-(c)
    answer source (user directive: only the new data) — a missing/unfetchable
    pool file must halt the variant, never degrade to another source."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    local = retry_transient(
        lambda: hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=path_in_repo,
        ),
        what=f"hf_hub_download({path_in_repo})",
    )
    return Path(local)


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


def _index_pool_answers(pool_path: Path, expected_form: str) -> tuple[dict[str, str], dict]:
    """Index one Phase-C pool file: CANONICAL conv id -> story-authored answer.

    The pool rows carry the answer directly (`answer` = the on-policy
    continuation span; `issue2054_phase_c._splice_generated` schema) — no
    story parsing. Returns (answers_by_conv, stats). Duplicate conv_ids keep
    the FIRST row (counted + warned); rows with an empty answer or a
    mismatched `form` field are counted skips (reported, never silent).
    """
    answers_by_conv: dict[str, str] = {}
    n_rows_seen = 0
    n_empty_answer = 0
    n_form_mismatch = 0
    n_dup_conv = 0
    rows = _read_jsonl(pool_path)
    for row in rows:
        n_rows_seen += 1
        answer = row.get("answer")
        conv_id = row.get("conv_id") or row.get("scaffold_id")
        if not conv_id:
            continue
        if row.get("form") != expected_form:
            n_form_mismatch += 1
            continue
        if not isinstance(answer, str) or not answer:
            n_empty_answer += 1
            continue
        key = _canon_conv_id(str(conv_id))
        if key in answers_by_conv:
            n_dup_conv += 1
            continue
        answers_by_conv[key] = answer
    stats = {
        "n_rows_seen": n_rows_seen,
        "n_empty_answer": n_empty_answer,
        "n_form_mismatch": n_form_mismatch,
        "n_dup_conv": n_dup_conv,
        "n_unique_conv": len(answers_by_conv),
    }
    if n_form_mismatch or n_dup_conv or n_empty_answer:
        _log(
            f"WARN pool anomalies in {pool_path.name}: form_mismatch={n_form_mismatch} "
            f"dup_conv={n_dup_conv} empty_answer={n_empty_answer}"
        )
    _log(f"pool index: seen={n_rows_seen} unique_conv={len(answers_by_conv)} <- {pool_path.name}")
    return answers_by_conv, stats


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
    fold_conv_ids: set[str] | None,
    target_conv_ids: int,
    form: str,
    answer_form: str,
    n_floor: int,
    *,
    fold_map_sha: str | None,
    overwrite: bool,
) -> dict:
    """Splice THIS task's Phase-C on-policy answers into scaffolds for one variant.

    Resume (C9/M6): a variant whose output + regime-matching done sidecar
    already exist is SKIPPED; a CHANGED input (scaffolds / fold map / pool
    file — the pool sha IS pinned, so a re-uploaded pool RECOMPUTES with a
    loud log line, the #1947 salvage-inputs rule) recomputes; a DIFFERENT
    regime refuses (RegimeMismatch) — in particular, a sidecar written under
    the RETIRED parent-#1345 answer source lacks the answer_source /
    answer_form / answer_model regime keys and refuses rather than silently
    resuming stale parent-pool splices (user directive: only the new data).
    """
    base_variant = _base_variant_for(variant)
    answer_model = _answer_model_for(variant)
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

    # Fetch the pool BEFORE the resume check: its sha256 is part of the inputs
    # pin (HF-cached, so a resume re-run costs no re-download). FAIL-LOUD on a
    # missing pool file — never degrade to another answer source.
    pool_path_in_repo = _pool_path_in_repo(variant, answer_form)
    pool_path = _download_pool_file(pool_path_in_repo)

    vdir_early = out_dir / variant
    vdir_early.mkdir(parents=True, exist_ok=True)
    early_out_path = vdir_early / forms.phase_output_name("cell_c", variant, form)
    regime = {
        "cell": forms.cell_key(variant, "cell_c", form, "any"),
        "target_conv_ids": int(target_conv_ids),
        "fold_filter": fold_conv_ids is not None,
        "answer_source": ANSWER_SOURCE,
        "answer_form": answer_form,
        "answer_model": answer_model,
    }
    inputs = {
        "scaffolds_sha256": resume.file_sha256(scaffolds_path),
        "fold_map_sha256": fold_map_sha,
        "on_policy_pool_sha256": resume.file_sha256(pool_path),
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
    _log(
        f"variant={variant} answer pool: {pool_path_in_repo} "
        f"(answer_model={answer_model} answer_form={answer_form})"
    )
    answers_by_conv, pool_stats = _index_pool_answers(pool_path, answer_form)

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
            # Canonical key drives BOTH the fold-membership check and the
            # pool-answer join; the emitted row keeps the raw scaffold
            # conv_id (cross-cell conv matching with phase_b/c uses the
            # scaffold key space) plus `parent_conv_id` (canonical) for
            # cross-cell provenance.
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
        pool_keys = list(answers_by_conv)[:3]
        _log(
            f"variant={variant} ZERO-JOIN diagnostic: scaffold conv keys "
            f"(canonical) sample={scaffold_keys} pool keys sample={pool_keys}"
        )

    # Per-cell realized-n floor report (user floor: n_train >= 5,000 at the
    # 0.8 split => n >= 6,250). A shortfall is REPORTED loud + flagged in the
    # digest — never silently passed over (and never a silent abort: the
    # analyzer / orchestrator owns the pause decision).
    floor_met = n_out >= n_floor
    if not floor_met:
        _log(
            f"WARN variant={variant} realized n_out={n_out} BELOW floor {n_floor} "
            f"(n_train {int(n_out * 0.8)} < 5000 at the 0.8 split)"
        )

    _log(
        f"variant={variant} spliced={n_out}/{n_in} "
        f"no_answer={n_no_answer} out_of_fold={n_out_of_fold} "
        f"sentinel_escapes={n_sentinel_escaped_rows} sentinel_hits={n_sentinel_hits_total} "
        f"floor={'MET' if floor_met else 'MISSED'}({n_out}/{n_floor}) "
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
        "n_floor": int(n_floor),
        "floor_met": bool(floor_met),
        "answer_model": answer_model,
        "answer_form": answer_form,
        "answer_pool": pool_path_in_repo,
        "pool_stats": pool_stats,
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
    url = _upload_folder_filtered(
        out_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{TASK_PREFIX}/cell_c",
        allow_patterns=allow_patterns,
        expected_repo_paths=expected_paths,
    )
    if not url:
        # _upload_folder_filtered is fail-soft by RETURN on every failure
        # shape (missing token, incomplete verify, terminal exception -> "")
        # — an empty return is a failed upload, not a success (M2).
        raise RuntimeError(
            f"cell_c bulk upload failed or incomplete -> {TASK_PREFIX}/cell_c/ "
            "(returned no path; local files kept)"
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
    _log(
        f"start: target_conv_ids={args.target_conv_ids} form={args.form} "
        f"answer_form={args.answer_form} n_floor={args.n_floor} variants={variants}"
    )

    try:
        import huggingface_hub  # noqa: F401  (hard dependency of the pool fetch)
    except ImportError as exc:  # noqa: BLE001
        print(f"ERROR: huggingface_hub missing: {exc}", file=sys.stderr)
        return 2

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
            fold_conv_ids,
            args.target_conv_ids,
            args.form,
            args.answer_form,
            args.n_floor,
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

    below_floor = [r["variant"] for r in per_variant_reports if r.get("floor_met") is False]
    if below_floor:
        _log(f"WARN {len(below_floor)} variant(s) below n-floor {args.n_floor}: {below_floor}")

    is_smoke = str(out_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload:
        # FATAL on failure (M2): `[phase=done]` must never report done with
        # the cell (c) splices un-persisted. No try/except.
        _upload_to_hf(out_paths, out_dir)

    digest = {
        "phase": "phase_d",
        "form": args.form,
        "answer_source": ANSWER_SOURCE,
        "answer_form": args.answer_form,
        "n_floor": int(args.n_floor),
        "variants_below_floor": below_floor,
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
    p.add_argument(
        "--answer-form",
        default="attrib_quoted",
        choices=[f for f in forms.STORY_FORMS if f != "indirect"],
        help=(
            "which Phase-C STORY form's on-policy answers supply the cell-(c) "
            "answer text (default attrib_quoted — the parent's canonical story "
            "boundary form; the (c) row is byte-matched with the (d) cell of "
            "the SAME answer_form). Realized pool forms: attrib_quoted, "
            "bare_label"
        ),
    )
    p.add_argument(
        "--n-floor",
        type=int,
        default=6_250,
        help=(
            "per-cell realized-n floor to report against (user floor: "
            "n_train >= 5,000 at the 0.8 split => 6,250). Shortfalls are "
            "REPORTED (log WARN + digest flag), never silently passed over"
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
