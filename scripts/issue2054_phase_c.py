#!/usr/bin/env python
"""Phase C driver for task #2054: on-policy continuation via vLLM prefill.

For each scaffold under `--scaffolds-dir`, render a prefill for the REQUIRED
`--form` framing (plan §4 — the lattice's central manipulated variable; no
default): story forms via the parent's `render_prefill(scaffold_text, form,
char_name)` (drop the trailing SLOT_SENTINEL segment — the head of scaffold
up to the slot + the form's opening becomes the prompt), chat / bare_text via
`issue2054_forms.render_prefill_form` (the scaffold's question re-framed in
the chat / bare template; narrative prose dropped). The model continues; the
generated tokens BEFORE the form's stop string ARE the answer span (100% keep
by construction; no post-hoc verbatim matcher).

Writes per-row rollouts under `data/issue_2054/on_policy/{variant}/` with the
final spliced text + exact answer offsets recorded via `sc.splice_answer`, and
mirrors to HF `issue2054_lattice/on_policy/{variant}/` (best-effort,
non-fatal). Answer-length per row is recorded for downstream DV 7 length
parity.

Emits `[phase=phase_c]` log lines terminating in `[phase=done]` on graceful
completion. `--dry-run` verifies wiring without vLLM (no GPU): it reads the
scaffolds, prepares prefill prompts, and writes a mock JSONL with the
prefill_text + stop tuple recorded per row.

Exit 0 on success. Exit 1 on splice / vLLM / HF / preflight failure. Exit 2
on missing dependency.
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

_MODEL_ID = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}

_CHAR_NAME_FROM_VARIANT = {
    "char_helios": "Helios",
    "char_wren": "Wren",
    "char_dana": "Dana",
    "char_vex": "Vex",
    "conversation_paired_stories_assistant": "Assistant",
    "conversation_paired_stories": "ARIA",
}


def _log(msg: str) -> None:
    print(f"[phase=phase_c] {msg}", flush=True)


def _rel(path: Path) -> str:
    """Best-effort repo-root-relative string; falls back to abs for /tmp/*."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _char_name_from_scaffold_row(row: dict, variant: str) -> str:
    """Recover the character name for the prefill render / splice."""
    ch = str(row.get("character") or "").strip()
    if ch:
        return ch
    return _CHAR_NAME_FROM_VARIANT.get(variant, "ARIA")


def _prepare_prefill(row: dict, variant: str, form: str) -> dict | None:
    """Return {conv_id, scaffold_text, prefix_text, stop, char_name, ...} or None.

    Story forms use the parent's `render_prefill` — the same rendering the
    story-slot arm of issue1345 used (raw text continuation, per-form stop);
    SLOT_SENTINEL is stripped by the renderer (the head up to the slot + the
    form's opening is returned as the raw prefix). Template forms (chat /
    bare_text) route through `issue2054_forms.render_prefill_form` and
    require the row's question (a row lacking it is a counted skip).
    """
    scaffold = row.get("scaffold_text")
    if not isinstance(scaffold, str):
        return None
    if form in forms.STORY_FORMS and sc.SLOT_SENTINEL not in scaffold:
        return None
    char_name = _char_name_from_scaffold_row(row, variant)
    try:
        spec = forms.render_prefill_form(row, form, char_name)
    except (ValueError, NotImplementedError) as exc:
        _log(f"prefill skip {row.get('scaffold_id')}: {exc}")
        return None
    return {
        "scaffold_id": row.get("scaffold_id"),
        "conv_id": str(row.get("conv_id") or row.get("scaffold_id") or ""),
        "variant": variant,
        "character": char_name,
        "form": spec.form,
        "prefix_text": spec.prefix_text,
        "stop": list(spec.stop),
        "scaffold_text": scaffold,
        # Question-recovery fields ride along so _splice_generated can render
        # template forms + record the pre-query prefix boundary (plan §6 v_P).
        "question": row.get("question") if isinstance(row.get("question"), str) else None,
        "q_start": row.get("q_start") if isinstance(row.get("q_start"), int) else None,
        "q_end": row.get("q_end") if isinstance(row.get("q_end"), int) else None,
        "attrib_template": row.get("attrib_template")
        if isinstance(row.get("attrib_template"), str)
        else None,
    }


def _splice_generated(base: dict, answer: str, form: str) -> dict | None:
    """Render the generated answer under `form`; return the row-out dict."""
    if not answer:
        return None
    try:
        result = forms.splice_answer_form(
            base,
            answer,
            form,
            base["character"],
            attrib_template=base["attrib_template"],
        )
    except (ValueError, NotImplementedError) as exc:
        _log(f"splice skip {base.get('scaffold_id')}: {exc}")
        return None
    return {
        "scaffold_id": base["scaffold_id"],
        "conv_id": base["conv_id"],
        "variant": base["variant"],
        "character": base["character"],
        "form": result.form,
        "final_text": result.text,
        "answer": answer,
        "answer_start": result.answer_start,
        "answer_end": result.answer_end,
        "answer_len_chars": len(answer),
        "prefix_end_char": result.prefix_end_char,
    }


def _enumerate_variant_paths(scaffolds_root: Path, variants: list[str]) -> dict[str, Path]:
    """Return {variant: <scaffolds_root>/<variant>/scaffolds_<variant>.jsonl}."""
    out: dict[str, Path] = {}
    for variant in variants:
        candidate = scaffolds_root / variant / f"scaffolds_{variant}.jsonl"
        if candidate.is_file():
            out[variant] = candidate
    # Smoke fallback: any *.jsonl directly under the root as a single untyped variant.
    if not out:
        stray = sorted(scaffolds_root.glob("*.jsonl"))
        if stray:
            out["_flat"] = stray[0]
    return out


def _select_rows(rows: list[dict], target: int) -> list[dict]:
    """Deterministic pick of up to `target` rows preserving original order."""
    if target <= 0 or len(rows) <= target:
        return rows
    return rows[:target]


def _run_dry_run(
    per_variant_paths: dict[str, Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, dict[str, dict]]:
    """CPU-only smoke path: prepare prefills, emit a mock JSONL, no vLLM."""
    counts: dict[str, dict] = {}
    total_out = 0
    for variant, sp in per_variant_paths.items():
        scaffolds = _read_jsonl(sp)
        scaffolds = _select_rows(scaffolds, args.target_conv_ids)
        vdir = out_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        out_path = vdir / f"on_policy_{variant}.mock.jsonl"
        n_in = len(scaffolds)
        n_out = 0
        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in scaffolds:
                base = _prepare_prefill(row, variant, args.form)
                if base is None:
                    continue
                mock = {
                    "scaffold_id": base["scaffold_id"],
                    "conv_id": base["conv_id"],
                    "variant": variant,
                    "character": base["character"],
                    "form": base["form"],
                    "prefix_text": base["prefix_text"],
                    "stop": base["stop"],
                    "mock": True,
                }
                f.write(json.dumps(mock, ensure_ascii=False) + "\n")
                n_out += 1
        os.replace(tmp, out_path)
        counts[variant] = {"n_in": n_in, "n_out": n_out}
        total_out += n_out
        _log(f"variant={variant} dry-run prepared {n_out}/{n_in} -> {_rel(out_path)}")
    return total_out, counts


def _run_vllm(
    per_variant_paths: dict[str, Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, dict[str, dict]]:
    """GPU generation path: batched vLLM continuation from the prefill prefix."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # noqa: BLE001
        print(f"ERROR: vllm missing: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    model_id = _MODEL_ID.get(args.model, args.model)
    _log(f"loading vLLM engine model={model_id} temperature={args.temperature}")
    llm = LLM(model=model_id, dtype="bfloat16", trust_remote_code=True)

    counts: dict[str, dict] = {}
    total_out = 0
    for variant, sp in per_variant_paths.items():
        scaffolds = _read_jsonl(sp)
        scaffolds = _select_rows(scaffolds, args.target_conv_ids)
        vdir = out_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        out_path = vdir / f"on_policy_{variant}.jsonl"
        n_in = len(scaffolds)
        n_out = 0

        prepared: list[dict] = []
        for row in scaffolds:
            base = _prepare_prefill(row, variant, args.form)
            if base is not None:
                prepared.append(base)

        if not prepared:
            counts[variant] = {"n_in": n_in, "n_out": 0}
            _log(f"variant={variant} no prefillable scaffolds; skip")
            continue

        prompts = [b["prefix_text"] for b in prepared]
        stops = prepared[0]["stop"]  # one --form per invocation ⇒ identical stop tuple
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            stop=stops,
            seed=args.seed,
        )

        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            # vLLM continuous batching: submit ALL prefixes at once, iterate outputs.
            # Preserve order via zip on the same list.
            outs = llm.generate(prompts, sampling, use_tqdm=False)
            for base, o in zip(prepared, outs, strict=True):
                gen_text = o.outputs[0].text
                # Preserve verbatim (no re-encoding). The generated text is the
                # answer span (vLLM default include_stop_str_in_output=False).
                answer = gen_text
                row_out = _splice_generated(base, answer, args.form)
                if row_out is None:
                    continue
                row_out["finish_reason"] = o.outputs[0].finish_reason
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                n_out += 1
        os.replace(tmp, out_path)
        counts[variant] = {"n_in": n_in, "n_out": n_out}
        total_out += n_out
        _log(f"variant={variant} generated {n_out}/{n_in} -> {_rel(out_path)}")

    return total_out, counts


def _upload_to_hf(paths_by_variant: dict[str, Path], out_dir: Path) -> None:
    """Best-effort mirror of on-policy JSONLs — ONE bulk upload_folder commit."""
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
        expected_paths.append(f"{TASK_PREFIX}/on_policy/{rel}")
    if not allow_patterns:
        return
    try:
        _upload_folder_filtered(
            out_dir,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{TASK_PREFIX}/on_policy",
            allow_patterns=allow_patterns,
            expected_repo_paths=expected_paths,
        )
        _log(f"uploaded {len(allow_patterns)} on-policy file(s) in one bulk commit")
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN on-policy bulk upload failed: {exc}")


def run_phase(args: argparse.Namespace) -> int:
    scaffolds_root = Path(args.scaffolds_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not scaffolds_root.exists():
        print(f"ERROR: scaffolds-dir does not exist: {scaffolds_root}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = list(args.variants)
    per_variant_paths = _enumerate_variant_paths(scaffolds_root, variants)
    if not per_variant_paths:
        print(
            f"ERROR: no scaffold JSONLs found under {scaffolds_root} for variants={variants}",
            file=sys.stderr,
        )
        return 1

    _log(
        f"start: target_conv_ids={args.target_conv_ids} model={args.model} "
        f"form={args.form} dry_run={args.dry_run} variants={list(per_variant_paths.keys())}"
    )

    if args.dry_run:
        total_out, counts = _run_dry_run(per_variant_paths, out_dir, args)
    else:
        total_out, counts = _run_vllm(per_variant_paths, out_dir, args)

    if total_out == 0:
        print("ERROR: phase_c produced ZERO rows", file=sys.stderr)
        return 1

    out_paths = {
        v: (
            (out_dir / v / f"on_policy_{v}.mock.jsonl")
            if args.dry_run
            else (out_dir / v / f"on_policy_{v}.jsonl")
        )
        for v in counts
    }

    is_smoke = str(out_dir).startswith("/tmp/")
    if not is_smoke and not args.skip_upload and not args.dry_run:
        try:
            _upload_to_hf(out_paths, out_dir)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN upload stage failed: {exc}")

    digest = {
        "phase": "phase_c",
        "form": args.form,
        "target_conv_ids": args.target_conv_ids,
        "model": args.model,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "dry_run": bool(args.dry_run),
        "counts": counts,
        "n_total_out": total_out,
        "out_paths": {v: _rel(p) for v, p in out_paths.items()},
        "seed": args.seed,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    digest_path = out_dir / "phase_c_digest.json"
    tmp = digest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, sort_keys=True)
    os.replace(tmp, digest_path)

    print(f"[phase=phase_c] digest: n_total_out={total_out}", flush=True)
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
            "framing to render (plan §4 — the lattice's central manipulated "
            "variable; REQUIRED, no default so a caller can never silently "
            "fall back to attrib_quoted)"
        ),
    )
    p.add_argument("--target-conv-ids", type=int, default=8_000)
    p.add_argument("--output-dir", default="data/issue_2054/on_policy/")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument(
        "--model",
        default="qwen2.5-7b-instruct",
        help="qwen2.5-7b | qwen2.5-7b-instruct (per-cell)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="vLLM generation cap (plan §11: >=2x longest trained completion)",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=[
            "char_helios",
            "char_wren",
            "char_dana",
            "char_vex",
            "conversation_paired_stories_assistant",
        ],
    )
    p.add_argument("--skip-upload", action="store_true", help="skip HF mirror step")
    p.add_argument(
        "--upload", action="store_true", help="force HF mirror step (default when not dry-run)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only wiring smoke: prepare prefills, emit mock JSONL, no vLLM",
    )
    args = p.parse_args()
    return run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
