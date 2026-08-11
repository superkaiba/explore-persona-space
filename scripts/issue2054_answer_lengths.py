"""Per-row answer TOKEN lengths + GATE 2 (answer-length KS parity) for the
#2054 `coordinated-common-set-regen` round (plan v12 §4 R2 step 4 / §7).

Two modes (``--mode``):

- ``lengths`` (R2, pod-side; runs IMMEDIATELY after the completion build and
  uploads BEFORE capture — the off-pod persistence gate 2 requires): walks
  the round's phase_b/c/d cell outputs, tokenizes every row's answer with
  the shared Qwen2.5 tokenizer (batched encode — no per-row tokenizer
  loads), asserts base/instruct tokenizer parity (assumption 12: vocab size
  + a fixed probe battery encode identically), and writes one
  ``{conv_id, n_answer_tokens}`` JSONL per lattice cell to
  ``<out>/lengths/`` + one bulk fail-loud upload to
  ``<hf_prefix>/lengths/`` (non-LFS text path).

- ``gate2`` (R2→R3 boundary, in-flight PRE-CAPTURE): per (variant, form,
  model) pair, two-sample KS D + mean-token ratio between the (b) inserted
  and (d) on-policy rows (all-rows AND conv-matched reads — the realized
  companion's convention), plus the chat-vs-story analogues (cell_c chat
  presentation vs the same authorship's story presentation). Fire condition
  (v8 kill-gate-5 constants): KS D > 0.30 OR ratio outside [0.25, 4.0].
  REPORT+MITIGATE semantics (plan §7 gate 2): on fire, capture PROCEEDS and
  the length-stratified companion refit becomes a MANDATORY R6 deliverable;
  the ONLY abort rail is companion degeneracy, adjudicated at R6 — so this
  evaluator always exits 0 on a well-formed run (exit 10 = malformed/missing
  inputs, a crash not a verdict).

Trigger-density note: no row text is ever printed — counts and digests only.
"""

from __future__ import annotations

import argparse
import json
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

import issue2054_forms as forms  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
CHAR_VARIANTS = ("char_helios", "char_wren", "char_dana", "char_vex")
ASSISTANT_VARIANT = "conversation_paired_stories_assistant"
MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")
MODEL_HF_ID = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}
_CELLC_TAIL = {"qwen2.5-7b-instruct": "_op", "qwen2.5-7b": "_op_base"}

# v8 kill-gate-5 constants (fits.py KILL_GATE_5_* — plan §7 gate 2 grounding).
KS_D_THRESHOLD = 0.30
RATIO_LO = 0.25
RATIO_HI = 4.0

TOKENIZER_PROBE = (
    "The quick brown fox jumps over the lazy dog.",
    "1234567890 !@#$%^&*()",
    "Multi\nline\ttext with unicode: é中文",
)


def _log(msg: str) -> None:
    print(f"[phase=answer_lengths] {msg}", flush=True)


def _utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_write_json(path: Path, payload: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=float)
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _shared_tokenizer():
    """The shared Qwen2.5 tokenizer + the assumption-12 parity assert."""
    from transformers import AutoTokenizer

    tok_a = AutoTokenizer.from_pretrained(MODEL_HF_ID["qwen2.5-7b"])
    tok_b = AutoTokenizer.from_pretrained(MODEL_HF_ID["qwen2.5-7b-instruct"])
    if tok_a.vocab_size != tok_b.vocab_size:
        raise RuntimeError(
            f"tokenizer parity FAILED: vocab sizes {tok_a.vocab_size} != {tok_b.vocab_size}"
        )
    for probe in TOKENIZER_PROBE:
        ids_a = tok_a.encode(probe, add_special_tokens=False)
        ids_b = tok_b.encode(probe, add_special_tokens=False)
        if ids_a != ids_b:
            raise RuntimeError(
                "tokenizer parity FAILED: base/instruct encode a probe differently "
                f"({len(ids_a)} vs {len(ids_b)} ids)"
            )
    _log("tokenizer parity PASS (vocab + probe battery identical across base/instruct)")
    return tok_b


def _count_tokens(tokenizer, texts: list[str]) -> list[int]:
    out: list[int] = []
    chunk = 2048
    for i in range(0, len(texts), chunk):
        enc = tokenizer(texts[i : i + chunk], add_special_tokens=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return out


# ---------------------------------------------------------------------------
# Cell enumeration over the round's phase outputs
# ---------------------------------------------------------------------------
def _cell_sources(args) -> list[tuple[str, str, str, str, Path]]:
    """(variant, condition, form, model, source JSONL) for every in-scope
    cell. Inserted files are model-INDEPENDENT (deterministic splice) — the
    same source file backs both models' inserted cells."""
    b_root = Path(args.phase_b_dir)
    c_root = Path(args.phase_c_dir)
    d_root = Path(args.phase_d_dir)
    cells: list[tuple[str, str, str, str, Path]] = []
    for ch in CHAR_VARIANTS:
        for form in ("attrib_quoted", "bare_label"):
            b_path = b_root / ch / forms.phase_output_name("inserted", ch, form)
            for model in MODELS:
                cells.append((ch, "inserted", form, model, b_path))
                c_path = c_root / model / ch / forms.phase_output_name("on_policy", ch, form)
                cells.append((ch, "on_policy", form, model, c_path))
        for model in MODELS:
            cc_variant = f"{ch}{_CELLC_TAIL[model]}"
            d_path = d_root / cc_variant / forms.phase_output_name("cell_c", cc_variant, "chat")
            cells.append((cc_variant, "cell_c", "chat", model, d_path))
    for form in ("chat", "bare_text"):
        b_path = (
            b_root
            / ASSISTANT_VARIANT
            / forms.phase_output_name("inserted", ASSISTANT_VARIANT, form)
        )
        for model in MODELS:
            cells.append((ASSISTANT_VARIANT, "inserted", form, model, b_path))
            c_path = (
                c_root
                / model
                / ASSISTANT_VARIANT
                / forms.phase_output_name("on_policy", ASSISTANT_VARIANT, form)
            )
            cells.append((ASSISTANT_VARIANT, "on_policy", form, model, c_path))
    assert len(cells) == 48, len(cells)  # the 48 in-scope lattice cells (plan §4)
    return cells


def run_lengths(args) -> int:
    tokenizer = _shared_tokenizer()
    out_root = Path(args.output_dir)
    lengths_dir = out_root / "lengths"
    lengths_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen_cells: set[str] = set()
    missing: list[str] = []
    for variant, cond, form, model, src in _cell_sources(args):
        cell = forms.cell_key(variant, cond, form, model)
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        if not src.is_file():
            missing.append(f"{cell}: {src}")
            continue
        rows = _read_jsonl(src)
        answers: list[str] = []
        conv_ids: list[str] = []
        for r in rows:
            a = r.get("answer")
            if not isinstance(a, str) or not a:
                raise RuntimeError(f"{cell}: row without a non-empty 'answer' field ({src})")
            answers.append(a)
            conv_ids.append(str(r.get("conv_id")))
        n_tok = _count_tokens(tokenizer, answers)
        out_path = lengths_dir / f"{cell}.jsonl"
        _atomic_write_jsonl(
            out_path,
            [{"conv_id": c, "n_answer_tokens": n} for c, n in zip(conv_ids, n_tok, strict=True)],
        )
        written.append(out_path)
        _log(f"{cell}: {len(rows)} rows tokenized -> {out_path.name}")
    if missing:
        for m in missing:
            print(f"[phase=answer_lengths] MISSING SOURCE: {m}", file=sys.stderr, flush=True)
        raise RuntimeError(
            f"{len(missing)} cell source(s) missing — the R2 completion build is incomplete"
        )

    manifest = {
        "artifact": "regen_answer_lengths",
        "n_cells": len(written),
        "tokenizer": MODEL_HF_ID["qwen2.5-7b-instruct"],
        "utc": _utc(),
    }
    _atomic_write_json(lengths_dir / "lengths_manifest.json", manifest)
    written.append(lengths_dir / "lengths_manifest.json")

    if not args.skip_upload:
        _upload_lengths(args, out_root, written)
    return 0


def _upload_lengths(args, out_root: Path, files: list[Path]) -> None:
    """One bulk fail-loud commit under `<hf_prefix>/lengths/` (non-LFS text;
    the off-pod persistence gate 2 requires — plan §4 R2 step 4)."""
    import issue2054_phase_a as pa
    from explore_persona_space.orchestrate.hub import _upload_folder_filtered

    files = pa._shard_large_jsonl_for_upload(files)
    allow = sorted({f.relative_to(out_root).as_posix() for f in files if f.is_file()})
    if not allow:
        raise RuntimeError("upload set resolved EMPTY against the written lengths files")
    # UPLOAD_PREFIX_EXEMPT: round-dedicated driver — the default IS this round's common_regen prefix (plan v12 §10); the parent prefix is a separate read-only --parent-prefix
    expected = [f"{args.hf_prefix}/{rel}" for rel in allow]
    # UPLOAD_PREFIX_EXEMPT: round-dedicated evaluator — the default IS this round's common_regen prefix (plan v12 §10); lengths never write the parent prefix
    url = _upload_folder_filtered(
        out_root,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=args.hf_prefix,
        allow_patterns=allow,
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"lengths bulk upload failed or incomplete -> {args.hf_prefix}/lengths/ "
            "(returned no path; local files kept)"
        )
    _log(f"uploaded {len(allow)} lengths file(s) in one bulk commit")


# ---------------------------------------------------------------------------
# Gate 2 (KS parity, report+mitigate)
# ---------------------------------------------------------------------------
def _ks_d(a, b) -> float:
    """Two-sample KS statistic, vectorized (numpy sort + searchsorted)."""
    import numpy as np

    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(cdf_a - cdf_b).max())


def _load_lengths(lengths_dir: Path, cell: str) -> dict[str, int]:
    p = lengths_dir / f"{cell}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"lengths file missing for cell {cell}: {p}")
    return {str(r["conv_id"]): int(r["n_answer_tokens"]) for r in _read_jsonl(p)}


def _pair_stats(b_of: dict[str, int], d_of: dict[str, int]) -> dict:
    import numpy as np

    b_all = list(b_of.values())
    d_all = list(d_of.values())
    shared = sorted(set(b_of) & set(d_of))
    rec = {
        "n_b": len(b_all),
        "n_d": len(d_all),
        "n_matched": len(shared),
        "ks_all": _ks_d(b_all, d_all),
        "mean_ratio_all": float(np.mean(b_all) / np.mean(d_all)),
    }
    if shared:
        b_m = [b_of[c] for c in shared]
        d_m = [d_of[c] for c in shared]
        rec["ks_matched"] = _ks_d(b_m, d_m)
        rec["mean_ratio_matched"] = float(np.mean(b_m) / np.mean(d_m))
    ks = rec.get("ks_matched", rec["ks_all"])
    ratio = rec.get("mean_ratio_matched", rec["mean_ratio_all"])
    rec["fired"] = bool(ks > KS_D_THRESHOLD or not (RATIO_LO <= ratio <= RATIO_HI))
    return rec


def run_gate2(args) -> int:
    lengths_dir = Path(args.output_dir) / "lengths"
    pairs: dict[str, dict] = {}

    # (b) inserted vs (d) on-policy per (variant, form, model): 20 b-vs-d
    # pairs (16 character + 4 assistant) + 8 cell_c-vs-story analogues below
    # = 28 pairs total (matches the smoke).
    for variant in (*CHAR_VARIANTS, ASSISTANT_VARIANT):
        v_forms = (
            ("attrib_quoted", "bare_label")
            if variant != ASSISTANT_VARIANT
            else (
                "chat",
                "bare_text",
            )
        )
        for form in v_forms:
            for model in MODELS:
                b = _load_lengths(lengths_dir, forms.cell_key(variant, "inserted", form, model))
                d = _load_lengths(lengths_dir, forms.cell_key(variant, "on_policy", form, model))
                pairs[f"{variant}__{form}__{model}"] = _pair_stats(b, d)

    # Chat-vs-story analogues (plan §7 gate 2): cell_c (story-AUTHORED,
    # chat-PRESENTED) vs the same authorship's story presentation.
    for ch in CHAR_VARIANTS:
        for model in MODELS:
            cc_variant = f"{ch}{_CELLC_TAIL[model]}"
            c = _load_lengths(lengths_dir, forms.cell_key(cc_variant, "cell_c", "chat", model))
            s = _load_lengths(lengths_dir, forms.cell_key(ch, "on_policy", "attrib_quoted", model))
            pairs[f"{ch}__cell_c_vs_story__{model}"] = _pair_stats(c, s)

    fired = sorted(k for k, v in pairs.items() if v["fired"])
    report = {
        "artifact": "gate2_report",
        "thresholds": {"ks_d": KS_D_THRESHOLD, "ratio_lo": RATIO_LO, "ratio_hi": RATIO_HI},
        "n_pairs": len(pairs),
        "n_fired": len(fired),
        "fired_pairs": fired,
        "pairs": pairs,
        "semantics": (
            "report+mitigate (plan §7 gate 2): capture PROCEEDS; on fire the "
            "length-stratified companion refit (issue2054_length_stratified_refit.py) is a "
            "MANDATORY R6 deliverable for every fired pair; the ONLY abort rail is companion "
            "degeneracy for > half the fired pairs, adjudicated at R6"
        ),
        "utc": _utc(),
    }
    out = Path(args.gate2_report_out)
    _atomic_write_json(out, report)
    _log(f"gate 2: {len(fired)}/{len(pairs)} pairs FIRED (report+mitigate) -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        default=None,
        choices=("lengths", "gate2"),
        help="REQUIRED except under --import-check",
    )
    p.add_argument("--output-dir", default="data/issue_2054/common_regen")
    p.add_argument("--phase-b-dir", default="data/issue_2054/common_regen/spliced_inserted")
    p.add_argument("--phase-c-dir", default="data/issue_2054/common_regen/on_policy")
    p.add_argument("--phase-d-dir", default="data/issue_2054/common_regen/cell_c")
    p.add_argument("--hf-prefix", default="issue2054_lattice/common_regen")
    p.add_argument(
        "--gate2-report-out",
        default="eval_results/issue_2054/coordinated_common_set_regen/gate2_report.json",
    )
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args()

    if args.import_check:
        import numpy  # noqa: F401

        import issue2054_phase_a  # noqa: F401
        import issue2054_regen_waves as rw
        from explore_persona_space.orchestrate.hub import _upload_folder_filtered  # noqa: F401

        rw.assert_args_attrs_defined(__file__)
        print("[phase=answer_lengths] import-check OK", flush=True)
        return 0

    try:
        if args.mode == "lengths":
            return run_lengths(args)
        return run_gate2(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 10


if __name__ == "__main__":
    sys.exit(main())
