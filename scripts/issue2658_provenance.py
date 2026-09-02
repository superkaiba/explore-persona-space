"""#2658 P0 direction-provenance resolver: hash-verify every frozen Figure-3 direction.

For each of the eleven registered rows (issue2658_common.ROW_IDS), resolve the
frozen external direction, verify its file sha256 against the pinned table
below (plan §8 kill criterion: a hash mismatch RAISES — never a silent skip,
never a substituted vector), load it through the #779 plot3 loaders (imported,
not copied), assert the (3584,) block-19 shape, and emit
``eval_results/issue_2658/direction_provenance.json`` with per-row C2/C3
eligibility verdicts and frozen sign conventions.

Evidence-driven partition (supersedes the dispatch brief's default): TEN rows
carry a hash-verifiable frozen external direction — the seven plot3 roster rows
PLUS the three correctness rows (the #2388 bundle
``issue2388_correctness/derived/correctness_directions_L19.pt`` exists with a
full provenance chain: source_commit + GitHub-pinned producer function +
capture-archive fingerprints matching the frozen model revision).  Exactly ONE
row (harmful_compliance) has no frozen external direction anywhere (plot3
roster, issue779_monitoring/r_b, issue1482_rb4/r_b, and the plan's cited
sources #779/#1482/#2203/#2388 were all swept) and records
``not-estimable — no frozen external direction`` for C2/C3.

Resolution order per file: issue-2658 staging dir (``data/issue_2658/hf_dl``)
-> known local staged copies (the #779 plot3 staging root, read-only) -> HF
download.  Every branch hash-verifies.  ``--snapshot-inputs`` uploads the two
local-only bundles (#1434 writing-style — packed on HF; #2388 correctness — not
on HF) to the issue-owned prefix ``issue2658_dirvalid/inputs/`` so the whole
resolver is durable/reproducible from HF alone.

Usage (VM, thread-capped):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python \\
    scripts/issue2658_provenance.py --snapshot-inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# issue779_common (imported transitively by the plot3 module) setdefaults
# HF_HOME to the pod-canonical /workspace path, which does not exist on the VM
# — pin a sane default FIRST so downloads keep working everywhere (#779 panel
# gotcha: "Panel import forces HF_HOME=/workspace").
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps land before numpy/torch imports (transitively via P3).
load_dotenv()

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_common as C  # noqa: E402
import issue779_plot3_redesign as P3  # noqa: E402  (the #779 loaders — imported, not copied)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

HF_REPO = "superkaiba1/explore-persona-space-data"
# #779 plot3 staging root (read-only reuse of already-staged HF downloads).
PLOT3_STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue779_plot3/hf_dl")
REPO_ROOT = _SCRIPTS_DIR.parent
ISSUE_DL = REPO_ROOT / "data/issue_2658/hf_dl"
OUT_DEFAULT = REPO_ROOT / "eval_results/issue_2658/direction_provenance.json"
SNAPSHOT_PREFIX = "issue2658_dirvalid/inputs"

# ---------------------------------------------------------------------------
# Frozen sha256 pin table.  Domain: chunked sha256 over the FILE BYTES on disk
# (P3._sha256's convention), byte-verified against the HF copies on 2026-09-01
# (HF == plot3-staged == main-repo local copies for every file below; the
# 571 MB armA__v_C.npy pin is the plot3-staged hf_hub_download product — its
# six sibling files were re-downloaded and matched HF byte-for-byte).
# Plan §8: any mismatch RAISES (DirectionProvenanceError), never a skip.
# ---------------------------------------------------------------------------
EXPECTED_SHA256: dict[str, str] = {
    "issue779_monitoring/r_b/evil.pt": (
        "65b70c63076b9452c6d1c8a66ee1ed3d403503df936ea1af6fffc353d135aff1"
    ),
    "issue779_monitoring/r_b/sycophancy.pt": (
        "af6d679b59ad02e9e00a26e73ff77c00dda69cb8e2fabd22ea3a3ee28bbdad3d"
    ),
    "issue779_monitoring/r_b/hallucination.pt": (
        "d643269c9904b99e14968c84c8e3a02cd45d5ed4674621edbeb78950467ccd6d"
    ),
    "issue2203_ctx_capping/axis/qwen25_7b_axis_per_layer.pt": (
        "aff75eabc94a26cbcf69624b2e2e4771e9a8bd1ddd23a83cc2c0c0763ea976aa"
    ),
    "issue1482_rb4/r_b/impolite.pt": (
        "001cb31355f692d9d8c0509c63107c4462805abb1c738a26912ff724779dd167"
    ),
    "issue2356_refusalpred/armA/labels.json": (
        "f3c10453c070ad0ffd25fc7d1de14611eb64dc4ca0f8cd0bde42a70a977ecbb1"
    ),
    "issue2356_refusalpred/analysis_tensors/consolidated/armA.rows.json": (
        "d379265badb6e01515ba03bbd7a1c70099e93880a06b6a6eb6e092ee95044485"
    ),
    "issue2356_refusalpred/analysis_tensors/consolidated/armA__v_C.npy": (
        "353317322576e68cf35b18697f7363ce843b1233780e720c5fa04e6a257b2bce"
    ),
    "issue1434_writingstyle/analysis_tensors/rb_writing_style.pt": (
        "61e363a7ebfaec8e5e99a85f21f08bf75c7b0b7572c1f094d1975c45be3bd558"
    ),
    "issue2388_correctness/derived/correctness_directions_L19.pt": (
        "93c2f872fe4eab0d818f91b98d70016c1d4a28c4e16042d2c7374e8cfef0d789"
    ),
}

# Files not directly fetchable from HF at their canonical rel path (the #1434
# bundle lives inside a packed store; the #2388 bundle was never uploaded) —
# durable copies live at the issue-owned SNAPSHOT_PREFIX after
# ``--snapshot-inputs``; local candidates cover the pre-snapshot state.
LOCAL_ONLY_RELS = (
    "issue1434_writingstyle/analysis_tensors/rb_writing_style.pt",
    "issue2388_correctness/derived/correctness_directions_L19.pt",
)


class DirectionProvenanceError(RuntimeError):
    """Raised on any hash mismatch or unresolvable frozen direction (plan §8)."""


@dataclass(frozen=True)
class RowSpec:
    """One registered row's frozen-direction source."""

    row: str
    kind: str  # "rb28" | "axis" | "refusal_diff" | "correctness" | "none"
    rel: str | None  # canonical HF-relative path (None for refusal_diff/none)
    source_issue: str
    sign_convention: str
    surface: str | None = None  # correctness rows: key into directions dict


ROWS: tuple[RowSpec, ...] = (
    RowSpec(
        row="evil",
        kind="rb28",
        rel="issue779_monitoring/r_b/evil.pt",
        source_issue="#779",
        sign_convention=(
            "+dot = evil-trait-expressing (r_b = mean(kept trait-exhibiting) - "
            "mean(kept non-exhibiting), persona-vectors recipe)"
        ),
    ),
    RowSpec(
        row="sycophancy",
        kind="rb28",
        rel="issue779_monitoring/r_b/sycophancy.pt",
        source_issue="#779",
        sign_convention="+dot = sycophantic (r_b pos-minus-neg, persona-vectors recipe)",
    ),
    RowSpec(
        row="hallucination",
        kind="rb28",
        rel="issue779_monitoring/r_b/hallucination.pt",
        source_issue="#779",
        sign_convention="+dot = hallucination-trait-expressing (r_b pos-minus-neg)",
    ),
    RowSpec(
        row="refusal",
        kind="refusal_diff",
        rel=None,
        source_issue="#2356",
        sign_convention="+dot = refusal (diff-of-means refuse - engage over armA v_C, block 19)",
    ),
    RowSpec(
        row="assistantness",
        kind="axis",
        rel="issue2203_ctx_capping/axis/qwen25_7b_axis_per_layer.pt",
        source_issue="#2203",
        sign_convention="+dot = assistant-like (default-assistant - role-play diff-of-means)",
    ),
    RowSpec(
        row="casualness",
        kind="rb28",
        rel="issue1434_writingstyle/analysis_tensors/rb_writing_style.pt",
        source_issue="#1434",
        sign_convention="+dot = casual register (casual-vs-formal writing-style r_b)",
    ),
    RowSpec(
        row="impoliteness",
        kind="rb28",
        rel="issue1482_rb4/r_b/impolite.pt",
        source_issue="#1482",
        sign_convention="+dot = impolite (rb4 impolite persona-vector r_b, smoke=False bundle)",
    ),
    RowSpec(
        row="harmful_compliance",
        kind="none",
        rel=None,
        source_issue="none",
        sign_convention="n/a — no frozen external direction",
    ),
    RowSpec(
        row="correctness_math",
        kind="correctness",
        rel="issue2388_correctness/derived/correctness_directions_L19.pt",
        source_issue="#2388",
        sign_convention="+dot = correct (within-context mean(correct t1) - mean(incorrect t1))",
        surface="math",
    ),
    RowSpec(
        row="correctness_mmlu_pro",
        kind="correctness",
        rel="issue2388_correctness/derived/correctness_directions_L19.pt",
        source_issue="#2388",
        sign_convention="+dot = correct (within-context mean(correct t1) - mean(incorrect t1))",
        surface="mcq",
    ),
    RowSpec(
        row="correctness_code",
        kind="correctness",
        rel="issue2388_correctness/derived/correctness_directions_L19.pt",
        source_issue="#2388",
        sign_convention="+dot = correct (within-context mean(correct t1) - mean(incorrect t1))",
        surface="code",
    ),
)

NOT_ESTIMABLE = "not-estimable — no frozen external direction"


def _verify_sha(path: Path, rel: str) -> str:
    got = P3._sha256(path)
    expected = EXPECTED_SHA256[rel]
    if got != expected:
        raise DirectionProvenanceError(
            f"sha256 mismatch for {rel} at {path}: expected {expected}, got {got} "
            "(plan §8 kill criterion: halt — never substitute)"
        )
    return got


def _local_candidates(rel: str) -> list[Path]:
    cands = [PLOT3_STAGE_ROOT / rel]
    main_repo = Path("/home/thomasjiralerspong/explore-persona-space")
    if rel == "issue1482_rb4/r_b/impolite.pt":
        cands += [
            main_repo / "data/issue_779/r_b/impolite.pt",
            main_repo / "data/issue_1482_rb4/r_b/impolite.pt",
        ]
    if rel == "issue2356_refusalpred/armA/labels.json":
        cands.append(
            main_repo / ".claude/worktrees/issue-2356/eval_results/issue_2356/armA/labels.json"
        )
    return cands


def resolve_file(rel: str) -> Path:
    """Resolve one pinned file: issue staging -> local staged copies -> HF.

    Every branch hash-verifies against EXPECTED_SHA256 (mismatch RAISES).
    """
    dest = ISSUE_DL / rel
    if dest.exists():
        _verify_sha(dest, rel)
        return dest
    for cand in _local_candidates(rel):
        if cand.exists():
            _verify_sha(cand, rel)
            return cand
    # HF fallback: canonical rel for hub-resident files; snapshot prefix for
    # the local-only bundles (available after --snapshot-inputs).
    from huggingface_hub import hf_hub_download

    hub_rel = f"{SNAPSHOT_PREFIX}/{rel}" if rel in LOCAL_ONLY_RELS else rel
    try:
        p = Path(hf_hub_download(HF_REPO, hub_rel, repo_type="dataset", local_dir=ISSUE_DL))
    except Exception as e:  # fail loud with the resolution trail
        raise DirectionProvenanceError(
            f"cannot resolve {rel}: no local copy and HF fetch of {hub_rel} failed ({e})"
        ) from e
    _verify_sha(p, rel)
    return p


def _vector_sha256(v: np.ndarray) -> str:
    """Content address of a direction. Domain: float32 C-contiguous raw bytes."""
    arr = np.ascontiguousarray(v.astype(np.float32, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load_correctness(path: Path, surface: str) -> tuple[np.ndarray, dict]:
    """Load one surface's (3584,) L19 direction from the #2388 bundle.

    Returns the float32 vector plus the bundle's own provenance chain
    (recipe, source_commit, recipe_source, per-surface counts).
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d["layer"] != C.LAYER or d["hidden_dim"] != C.HIDDEN:
        raise DirectionProvenanceError(
            f"#2388 bundle layer/hidden mismatch: layer={d['layer']} hidden={d['hidden_dim']} "
            f"vs frozen ({C.LAYER}, {C.HIDDEN})"
        )
    if surface not in d["directions"]:
        raise DirectionProvenanceError(
            f"#2388 bundle lacks surface {surface!r}; present: {sorted(d['directions'])}"
        )
    vec = d["directions"][surface].to(torch.float32).numpy()
    if vec.shape != (C.HIDDEN,):
        raise DirectionProvenanceError(f"#2388 {surface} direction shape {vec.shape}")
    surf = d["surfaces"][surface]
    bundle_prov = {
        "recipe": d["recipe"],
        "recipe_source": d["recipe_source"],
        "source_commit": d["source_commit"],
        "n_spread_contexts": surf.get("n_spread_contexts"),
        "n_weighted_rollouts": surf.get("n_weighted_rollouts"),
        "pool_split": surf.get("pool_split"),
        "direction_norm": surf.get("direction_norm"),
    }
    return vec, bundle_prov


def snapshot_inputs() -> list[dict]:
    """Upload the two local-only bundles to the issue-owned HF prefix.

    Idempotent: skips when the snapshot already exists AND hash-matches
    (verified by a fresh download).  Uses the canonical fail-loud hub helper.
    """
    from explore_persona_space.orchestrate import hub
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    records = []
    for rel in LOCAL_ONLY_RELS:
        local = resolve_file(rel)
        hub_rel = f"{SNAPSHOT_PREFIX}/{rel}"
        if api.file_exists(HF_REPO, hub_rel, repo_type="dataset"):
            action = "already-present"
        else:
            # UPLOAD_LOOP_EXEMPT: fixed 2-file issue-owned input snapshot (LOCAL_ONLY_RELS), idempotent + download-verified; never a bulk tree
            hub._upload(
                local_path=local,
                repo_id=HF_REPO,
                repo_type="dataset",
                path_in_repo=hub_rel,
                upload_as_file=True,
                raise_on_error=True,
            )
            action = "uploaded"
        # Verify the snapshot's bytes match the pin via a fresh download.
        with tempfile.TemporaryDirectory(dir=ISSUE_DL.parent, prefix=".snapverify_") as td:
            p = hf_hub_download(HF_REPO, hub_rel, repo_type="dataset", local_dir=td)
            got = P3._sha256(Path(p))
        if got != EXPECTED_SHA256[rel]:
            raise DirectionProvenanceError(
                f"issue-owned snapshot {hub_rel} hash mismatch after {action}: "
                f"{got} != {EXPECTED_SHA256[rel]}"
            )
        records.append({"rel": rel, "hub_path": hub_rel, "sha256": got, "action": action})
        print(f"[snapshot] {hub_rel}: {action}, sha256 verified")
    return records


def resolve_rows() -> list[dict]:
    """Resolve + hash-verify all 11 rows; return provenance entries."""
    entries: list[dict] = []
    for spec in ROWS:
        construct = C.CONSTRUCTS[spec.row]
        entry: dict = {
            "row": spec.row,
            "construct": construct.construct,
            "extraction_contrast": construct.extraction_contrast,
            "source_issue": spec.source_issue,
            "layer": C.LAYER,
            "sign_convention": spec.sign_convention,
        }
        if spec.kind == "none":
            entry.update(
                {
                    "frozen_external_direction": False,
                    "source": None,
                    "file_sha256": None,
                    "vector_sha256": None,
                    "loader": None,
                    "shape": None,
                    "c2_c3": NOT_ESTIMABLE,
                    "notes": (
                        "swept: plot3 roster (#779), HF issue779_monitoring/r_b "
                        "(evil/hallucination/sycophancy only), HF issue1482_rb4/r_b "
                        "(apathetic/humorous/impolite/optimistic only), and the plan's "
                        "cited sources #779/#1482/#2203/#2388 — no harmful-compliance "
                        "direction exists; row stays eligible for C0/C1/C4/C5"
                    ),
                }
            )
            entries.append(entry)
            print(f"[row] {spec.row}: {NOT_ESTIMABLE}")
            continue

        if spec.kind == "refusal_diff":
            labels_path = resolve_file("issue2356_refusalpred/armA/labels.json")
            rows_rel = "issue2356_refusalpred/analysis_tensors/consolidated/armA.rows.json"
            vc_rel = "issue2356_refusalpred/analysis_tensors/consolidated/armA__v_C.npy"
            rows_path = resolve_file(rows_rel)
            vc_path = resolve_file(vc_rel)
            # P3._refusal_direction reads both files under ONE stage root and the
            # labels via the module constant — point both at the resolved copies.
            stage_root = _common_stage_root(
                {rows_rel: rows_path, vc_rel: vc_path},
            )
            P3.LABELS_2356 = labels_path  # module-level constant consumed inside the loader
            vec, counts = P3._refusal_direction(stage_root, C.LAYER)
            entry.update(
                {
                    "frozen_external_direction": True,
                    "source": {
                        "repo": HF_REPO,
                        "paths": [vc_rel, rows_rel, "issue2356_refusalpred/armA/labels.json"],
                    },
                    "file_sha256": {
                        vc_rel: EXPECTED_SHA256[vc_rel],
                        rows_rel: EXPECTED_SHA256[rows_rel],
                        "issue2356_refusalpred/armA/labels.json": EXPECTED_SHA256[
                            "issue2356_refusalpred/armA/labels.json"
                        ],
                    },
                    "loader": "issue779_plot3_redesign._refusal_direction",
                    "counts": counts,
                }
            )
        elif spec.kind == "rb28":
            path = resolve_file(spec.rel)
            vec = P3._load_rb28(path, C.LAYER)
            entry.update(
                {
                    "frozen_external_direction": True,
                    "source": {"repo": HF_REPO, "paths": [spec.rel]},
                    "file_sha256": {spec.rel: EXPECTED_SHA256[spec.rel]},
                    "loader": "issue779_plot3_redesign._load_rb28",
                }
            )
            if spec.rel in LOCAL_ONLY_RELS:
                entry["source"]["issue_owned_snapshot"] = f"{SNAPSHOT_PREFIX}/{spec.rel}"
        elif spec.kind == "axis":
            path = resolve_file(spec.rel)
            vec = P3._load_axis(path, C.LAYER)
            entry.update(
                {
                    "frozen_external_direction": True,
                    "source": {"repo": HF_REPO, "paths": [spec.rel]},
                    "file_sha256": {spec.rel: EXPECTED_SHA256[spec.rel]},
                    "loader": "issue779_plot3_redesign._load_axis",
                }
            )
        elif spec.kind == "correctness":
            path = resolve_file(spec.rel)
            vec, bundle_prov = _load_correctness(path, spec.surface)
            entry.update(
                {
                    "frozen_external_direction": True,
                    "source": {
                        "repo": HF_REPO,
                        "paths": [spec.rel],
                        "issue_owned_snapshot": f"{SNAPSHOT_PREFIX}/{spec.rel}",
                        "surface": spec.surface,
                    },
                    "file_sha256": {spec.rel: EXPECTED_SHA256[spec.rel]},
                    "loader": "issue2658_provenance._load_correctness",
                    "bundle_provenance": bundle_prov,
                }
            )
        else:  # pragma: no cover - registry is closed
            raise DirectionProvenanceError(f"unknown row kind {spec.kind!r}")

        if vec.shape != (C.HIDDEN,) or vec.dtype != np.float32:
            raise DirectionProvenanceError(
                f"row {spec.row}: loaded vector shape/dtype {vec.shape}/{vec.dtype} "
                f"!= (({C.HIDDEN},), float32)"
            )
        if not np.isfinite(vec).all() or float(np.linalg.norm(vec)) == 0.0:
            raise DirectionProvenanceError(f"row {spec.row}: non-finite or zero direction")
        entry.update(
            {
                "shape": list(vec.shape),
                "vector_sha256": _vector_sha256(vec),
                "vector_norm": float(np.linalg.norm(vec)),
                "c2_c3": "eligible",
            }
        )
        entries.append(entry)
        print(
            f"[row] {spec.row}: eligible, |v|={entry['vector_norm']:.4f}, "
            f"vec_sha={entry['vector_sha256'][:16]}"
        )
    return entries


def _common_stage_root(rel_to_path: dict[str, Path]) -> Path:
    """Find the single root under which every rel resolves (symlinking if split).

    ``P3._refusal_direction`` takes ONE stage root; when the consolidated files
    resolve from different roots (e.g. rows.json in the issue staging dir,
    v_C reused read-only from the plot3 staging root), mirror them into the
    issue staging dir via symlinks (hash already verified by resolve_file).
    """
    roots = set()
    for rel, path in rel_to_path.items():
        root = Path(str(path)[: -len(rel) - 1]) if str(path).endswith(rel) else None
        roots.add(root)
    if len(roots) == 1 and None not in roots:
        return roots.pop()
    for rel, path in rel_to_path.items():
        dest = ISSUE_DL / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(path)
    return ISSUE_DL


def build_report(entries: list[dict], snapshot_records: list[dict] | None) -> dict:
    prov = git_provenance()
    n_not_estimable = sum(1 for e in entries if e["c2_c3"] != "eligible")
    return {
        "metadata": {
            **as_metadata_dict(prov, phase="p0-provenance"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "script": "scripts/issue2658_provenance.py",
        },
        "frozen_config": {
            "model_id": C.MODEL_ID,
            "model_revision": C.MODEL_REVISION,
            "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
            "tokenizer_json_sha256": C.TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": C.TOKENIZER_CONFIG_SHA256,
            "layer": C.LAYER,
            "hidden": C.HIDDEN,
            "dtype": C.DTYPE,
        },
        "vector_sha256_domain": "float32 C-contiguous raw bytes of the (3584,) block-19 vector",
        "file_sha256_domain": "chunked sha256 over raw file bytes",
        "c2_c3_partition": {
            "eligible": [e["row"] for e in entries if e["c2_c3"] == "eligible"],
            "not_estimable": [e["row"] for e in entries if e["c2_c3"] != "eligible"],
            "holm_family_sizes": C.holm_family_sizes(n_not_estimable),
        },
        "issue_owned_snapshots": snapshot_records,
        "notes": (
            "HF-vs-local byte identity verified 2026-09-01 for every direct-fetch pin "
            "(impolite.pt HF == both main-repo copies; labels.json HF == #2356 worktree "
            "copy; plot3-staged r_b/axis/rows files == fresh HF downloads). The 571 MB "
            "armA__v_C.npy pin is the plot3-staged hf_hub_download product (etag-verified "
            "at download; re-download skipped for size)."
        ),
        "rows": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--snapshot-inputs",
        action="store_true",
        help="upload the two local-only bundles to the issue-owned HF prefix first",
    )
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="alias for NOT passing --snapshot-inputs (kept explicit for smoke runs)",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    ISSUE_DL.mkdir(parents=True, exist_ok=True)
    snapshot_records = None
    if args.snapshot_inputs and not args.skip_upload:
        snapshot_records = snapshot_inputs()
    entries = resolve_rows()
    report = build_report(entries, snapshot_records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, args.out)
    part = report["c2_c3_partition"]
    print(
        f"[done] wrote {args.out} — {len(part['eligible'])} eligible, "
        f"{len(part['not_estimable'])} not-estimable ({part['not_estimable']})"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
