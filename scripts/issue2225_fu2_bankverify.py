"""Issue #2225 fu2 — S0 fu1-direction-bank stage + fail-loud verify (plan v13 §4.1).

The fu2 round REUSES the fu1 direction bank verbatim (no build). Before any
GPU spend, this script:

1. (default) re-downloads the 9-file bank from the CANONICAL data repo at ONE
   resolved revision via ``hub.stage_hub_prefix`` — or, with ``--verify-only``,
   verifies a pre-staged ``--bank-dir``;
2. asserts, per file (fail-loud — any miss raises before training):
   - ``{evil,sycophancy,hallucination}_PRE.pt`` + ``RND.pt``: realized shape
     (28, 3584); rows 14/19 finite with ``‖row_ℓ‖ ≈ ρ_ℓ`` (rho.json:
     63.056901 / 96.727321) within 1e-4; ALL other rows NaN (the fail-loud
     slice guard);
   - ``rho.json``: ρ_14/ρ_19 match the plan-pinned fu1 values within 1e-4
     (bank-identity pin against a regenerated bank);
   - ``*_PRE_meta.json``: ``ridge_payload_sha256`` fields non-empty AND
     ``rb_v2_rev == "032bdef"``;
   - ``RND_meta.json``: seeds {14: 2225014, 19: 2225019} (meta-recorded);
3. re-runs the parent's tokenize-only length preflight over the 3 fu2 corpora
   (fu1's S0 step; smoke blind-spot (c) mitigation) — fail-loud on a MISSING
   corpus; distribution stats recorded (informational, parent semantics);
4. writes the verdict record to ``--record-out`` (default: the fu2 EVAL_ROOT
   ``analysis/s0_bank_verify.json`` — the methodology-critic ask).

CPU + tokenizer only; no model load, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token before any torch/hub import

DATA_REPO = "superkaiba1/explore-persona-space-data"
BANK_HF_PREFIX = "issue2225_ctxsteer/analysis_tensors/fu1_directions"
BANK_SHAPE = (28, 3584)
MAP_LAYERS = (14, 19)
# Plan-pinned fu1 realized ρ (plan v13 §4.1 / fu1 rho.json, read at plan time).
RHO_PINNED = {14: 63.056901, 19: 96.727321}
RHO_TOL = 1e-4
RND_SEEDS = {14: 2225014, 19: 2225019}
RB_V2_REV = "032bdef"
PRE_TRAITS = ("evil", "sycophancy", "hallucination")
FU2_CORPORA = ("evil", "sycophancy", "hallucination")

BANK_FILES = (
    *(f"{t}_PRE.pt" for t in PRE_TRAITS),
    "RND.pt",
    *(f"{t}_PRE_meta.json" for t in PRE_TRAITS),
    "RND_meta.json",
    "rho.json",
)
assert len(BANK_FILES) == 9


def stage_bank(dest_root: Path) -> Path:
    """Re-download the bank prefix at ONE resolved revision (canonical helper:
    retries + atomic staging + headroom assert). Returns the staged bank dir."""
    from explore_persona_space.orchestrate.hub import stage_hub_prefix

    stage_hub_prefix(DATA_REPO, BANK_HF_PREFIX, dest_root, repo_type="dataset")
    bank_dir = dest_root / BANK_HF_PREFIX
    if not bank_dir.is_dir():
        raise RuntimeError(
            f"staging arithmetic violated: {bank_dir} absent after stage_hub_prefix "
            "(dest_root is a mirror root — hub.stage_hub_prefix contract)"
        )
    return bank_dir


def _verify_tensor(path: Path, rho: dict[int, float]) -> dict:
    """One bank tensor's fail-loud checks; returns the recorded evidence row."""
    import torch

    bank = torch.load(path, weights_only=True, map_location="cpu").to(torch.float64)
    if tuple(bank.shape) != BANK_SHAPE:
        raise AssertionError(f"{path.name}: shape {tuple(bank.shape)} != {BANK_SHAPE}")
    finite_rows = torch.isfinite(bank).all(dim=1)
    finite_idx = sorted(int(i) for i in torch.nonzero(finite_rows).flatten())
    if finite_idx != sorted(MAP_LAYERS):
        raise AssertionError(f"{path.name}: finite rows {finite_idx} != {sorted(MAP_LAYERS)}")
    # Non-map rows must be ALL-NaN (the fail-loud slice guard): a partially
    # finite row would silently steer garbage if a layer index slipped.
    non_map = [i for i in range(BANK_SHAPE[0]) if i not in MAP_LAYERS]
    if not torch.isnan(bank[non_map]).all():
        bad = [i for i in non_map if not torch.isnan(bank[i]).all()]
        raise AssertionError(f"{path.name}: non-map rows {bad} are not all-NaN")
    norms = {}
    for layer in MAP_LAYERS:
        norm = float(bank[layer].norm())
        if abs(norm - rho[layer]) > RHO_TOL:
            raise AssertionError(
                f"{path.name}: ‖row {layer}‖ = {norm!r} deviates from ρ_{layer} = "
                f"{rho[layer]!r} by more than {RHO_TOL}"
            )
        norms[str(layer)] = norm
    return {"shape": list(BANK_SHAPE), "finite_rows": finite_idx, "row_norms": norms}


def verify_bank(bank_dir: Path) -> dict:
    """All 9-file asserts (fail-loud). Returns the per-file evidence record."""
    missing = [f for f in BANK_FILES if not (bank_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"bank files missing under {bank_dir}: {missing}")

    with open(bank_dir / "rho.json", encoding="utf-8") as f:
        rho_obj = json.load(f)
    rho = {la: float(rho_obj["rho_per_layer"][str(la)]) for la in MAP_LAYERS}
    for la, pinned in RHO_PINNED.items():
        if abs(rho[la] - pinned) > RHO_TOL:
            raise AssertionError(
                f"rho.json ρ_{la} = {rho[la]!r} deviates from the plan-pinned fu1 value "
                f"{pinned!r} by more than {RHO_TOL} (regenerated bank?)"
            )

    files: dict[str, dict] = {"rho.json": {"rho_per_layer": {str(k): v for k, v in rho.items()}}}
    for name in (*(f"{t}_PRE.pt" for t in PRE_TRAITS), "RND.pt"):
        files[name] = _verify_tensor(bank_dir / name, rho)

    for trait in PRE_TRAITS:
        meta_name = f"{trait}_PRE_meta.json"
        with open(bank_dir / meta_name, encoding="utf-8") as f:
            meta = json.load(f)
        shas = meta.get("ridge_payload_sha256")
        if not isinstance(shas, dict) or not shas or not all(str(v).strip() for v in shas.values()):
            raise AssertionError(f"{meta_name}: ridge_payload_sha256 fields empty/missing: {shas}")
        if meta.get("rb_v2_rev") != RB_V2_REV:
            raise AssertionError(
                f"{meta_name}: rb_v2_rev = {meta.get('rb_v2_rev')!r} != {RB_V2_REV!r}"
            )
        files[meta_name] = {"rb_v2_rev": meta["rb_v2_rev"], "ridge_payload_sha256": shas}

    with open(bank_dir / "RND_meta.json", encoding="utf-8") as f:
        rnd_meta = json.load(f)
    seeds = {int(k): int(v) for k, v in (rnd_meta.get("seeds") or {}).items()}
    if seeds != RND_SEEDS:
        raise AssertionError(f"RND_meta.json seeds {seeds} != {RND_SEEDS}")
    files["RND_meta.json"] = {"seeds": {str(k): v for k, v in seeds.items()}}
    return files


def run_length_preflight(dataset_root: Path, model_name: str) -> dict:
    """Parent tokenize-only length preflight over the 3 fu2 corpora, fail-loud
    on a missing corpus (the parent records-and-continues; S0 must halt)."""
    import issue2225_train as train

    report = train.preflight_lengths(dataset_root, model_name, FU2_CORPORA)
    errors = {ds: row["error"] for ds, row in report["corpora"].items() if "error" in row}
    if errors:
        raise FileNotFoundError(f"length preflight: corpora missing under {dataset_root}: {errors}")
    return report


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2225 fu2 S0 bank stage + verify (plan §4.1).")
    ap.add_argument(
        "--dest-root",
        default=None,
        help="HF mirror root to stage the bank under (bank dir = <dest-root>/<prefix>)",
    )
    ap.add_argument(
        "--bank-dir",
        default=None,
        help="pre-staged bank dir (with --verify-only; else derived from --dest-root)",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="skip the re-download; verify an existing --bank-dir",
    )
    ap.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--skip-length-preflight",
        action="store_true",
        help="bank-only verify (VM unit without the pod-side corpora); recorded in the verdict",
    )
    ap.add_argument(
        "--record-out",
        default="eval_results/issue_2225/fu2_preimage_alltoken/analysis/s0_bank_verify.json",
        help="verdict record destination (fu2 EVAL_ROOT analysis/)",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        import torch  # noqa: F401

        import issue2225_train as train

        from explore_persona_space.orchestrate.hub import stage_hub_prefix  # noqa: F401

        assert callable(train.preflight_lengths)
        # Deterministic degenerate probe (no files): a wrong-shape tensor must
        # raise through the same _verify_tensor path production runs.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "evil_PRE.pt"
            torch.save(torch.zeros(2, 2), bad)
            try:
                _verify_tensor(bad, dict(RHO_PINNED))
            except AssertionError:
                pass
            else:
                raise AssertionError("wrong-shape bank tensor did not raise")
        print("[issue2225-fu2-bankverify] import-check OK", flush=True)
        return 0

    if args.verify_only:
        if not args.bank_dir:
            ap.error("--verify-only requires --bank-dir")
        bank_dir = Path(args.bank_dir)
    else:
        if not args.dest_root:
            ap.error("--dest-root is required unless --verify-only")
        bank_dir = stage_bank(Path(args.dest_root))

    files = verify_bank(bank_dir)
    if args.skip_length_preflight:
        length_report: dict = {"skipped": "by flag (--skip-length-preflight; VM bank-only unit)"}
    else:
        length_report = run_length_preflight(Path(args.dataset_root), args.model)

    import issue778_lib as lib

    record = {
        "phase": "s0_bank_verify",
        "followup": "fu2_preimage_alltoken",
        "bank_dir": str(bank_dir),
        "hf_source": None if args.verify_only else f"{DATA_REPO}/{BANK_HF_PREFIX}",
        "verify_only": bool(args.verify_only),
        "rho_pinned": {str(k): v for k, v in RHO_PINNED.items()},
        "rho_tolerance": RHO_TOL,
        "files": files,
        "length_preflight": length_report,
        "reproducibility": lib.repro_metadata(),
    }
    out = Path(args.record_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    tmp.replace(out)
    print(f"[s0-bankverify] 9/9 files verified (bank={bank_dir}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
