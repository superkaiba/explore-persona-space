"""#1776 Phase 0 staging: pin the data repo, probe every reused stem, stage inputs.

Plan v4 §10:
  1. Resolve ONE data-repo revision at run start (persisted pin).
  2. Revision-scoped existence probe per reused stem (scoped ``list_repo_tree``
     via ``hub.list_hf_files_under_path`` — the ~1M-file repo requires scoped
     calls; a default-branch probe does not satisfy, #1345). >=1 file per stem.
  3. Pairwise-provenance re-run at the pin (artifact-reuse check (j)): the
     last-commit dates along PROVENANCE_CHAIN must be monotone non-decreasing.
  4. Optional staging (``hub.stage_hub_prefix`` / ``stage_hub_file`` at the pin):
     the 6.02 GB pass_b bundle, n1m_readout weights, r_B stacks, sampling
     manifest, and N parity capture-chunk PAIRS (.pt + raw .json).
  5. pass_b bundle realized-keys probe (check (c)): mmap keys via
     ``verify_reused_artifact_keys`` (keys cx_last, v_x, layers) PLUS the
     {14,19} subset-of-layers assert, BEFORE any consumer assert.

Exit contract: 0 PASS / nonzero fail-loud (asserts / raised errors).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import issue1776_common as C76

from explore_persona_space.orchestrate import hub  # noqa: E402

# Canonical-pool centroids bundle (5b leakage leg). Lives under a DIFFERENT
# issue prefix, INSIDE the pool's centroids/ subdir — the att-20260729-060640
# crash was a heredoc staging the bare `issue483_canonical_persona_pool/
# centroids_v1_L21.pt` path (no `centroids/` segment): deterministic 404
# EntryNotFoundError at p0_stage. Content identity is the sha pin in the
# committed bank meta (CENTROIDS_META_PATH), not the #1776 revision pin.
CENTROIDS_HF_PATH = "issue483_canonical_persona_pool/centroids/centroids_v1_L21.pt"
CENTROIDS_META_RELPATH = "data/canonical_persona_pool/matrix_v1_L21_raw.json"


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def probe_stems(revision: str) -> dict[str, int]:
    """Scoped per-stem existence probe at the pin; >=1 resolved file per stem."""
    api = _api()
    counts: dict[str, int] = {}
    for stem in C76.REUSED_HF_STEMS:
        files = hub.list_hf_files_under_path(
            api, C76.HF_DATA_REPO, stem, repo_type="dataset", revision=revision
        )
        if not files:
            raise FileNotFoundError(f"stem probe FAILED: 0 files under {stem} @ {revision}")
        counts[stem] = len(files)
        print(f"[stage] probe {stem}: {len(files)} files @ pin", flush=True)
    # The single-file pass_b bundle must be present by exact name.
    pass_b_dir = "issue779_monitoring/analysis_tensors/pass_b"
    bundle_files = hub.list_hf_files_under_path(
        api, C76.HF_DATA_REPO, pass_b_dir, repo_type="dataset", revision=revision
    )
    assert C76.PASS_B_HF_PATH in bundle_files, (
        f"pass_b bundle {C76.PASS_B_HF_PATH} absent from scoped listing: {bundle_files}"
    )
    return counts


def provenance_check(revision: str) -> list[dict]:
    """Re-run artifact-reuse check (j) at the pin: monotone last-commit chain.

    Compares the LATEST last-commit date per stem along PROVENANCE_CHAIN —
    every consumed input must predate (<=) the artifact fit/selected on it.
    """
    api = _api()
    rows: list[dict] = []
    for stem in C76.PROVENANCE_CHAIN:
        infos = hub.retry_transient(
            lambda stem=stem: api.get_paths_info(
                C76.HF_DATA_REPO, [stem], repo_type="dataset", revision=revision, expand=True
            ),
            what=f"get_paths_info({stem})",
        )
        dates = [i.last_commit.date for i in infos if getattr(i, "last_commit", None)]
        if not dates:
            raise RuntimeError(f"provenance check: no last_commit info for {stem} @ {revision}")
        rows.append({"stem": stem, "last_commit": max(dates).isoformat()})
    import itertools

    for a, b in itertools.pairwise(rows):
        assert a["last_commit"] <= b["last_commit"], (
            f"provenance chain NOT monotone at the pin: {a} > {b} — a pin refresh "
            "re-opened check (j); resolve before consuming (artifact-reuse.md)"
        )
    print(f"[stage] provenance chain monotone-coherent @ pin: {rows}", flush=True)
    return rows


def stage_pass_b(revision: str, dest_root: Path) -> Path:
    """Stage the 6.02 GB pass_b bundle at the pin (verbatim prefix mirror)."""
    dest = dest_root / C76.PASS_B_HF_PATH
    if dest.exists():
        print(f"[stage] pass_b bundle already staged at {dest}", flush=True)
        return dest
    hub.stage_hub_file(
        C76.HF_DATA_REPO, C76.PASS_B_HF_PATH, dest, repo_type="dataset", revision=revision
    )
    return dest


def verify_pass_b_keys(bundle: Path) -> dict:
    """Check (c): realized keys via verify_reused_artifact_keys + {14,19} layers."""
    import verify_reused_artifact_keys as VK

    rc = VK.main(["--artifact", str(bundle), "--keys", "cx_last,v_x,layers", "--fmt", "pt"])
    assert rc == 0, f"verify_reused_artifact_keys FAILED (rc={rc}) on {bundle}"
    import issue779_fitter_fair_comparison as F

    pb = F._mmap_load(bundle)
    layers = [int(x) for x in pb["layers"]]
    assert {C76.SOURCE_LAYER, C76.READOUT_LAYER} <= set(layers), (
        f"pass_b bundle layers {layers} missing a consumed layer "
        f"{{{C76.SOURCE_LAYER}, {C76.READOUT_LAYER}}} — cross-layer design needs BOTH (plan §10)"
    )
    shape = tuple(pb["cx_last"].shape)
    print(f"[stage] pass_b keys OK: layers={layers} cx_last shape={shape}", flush=True)
    return {"layers": layers, "cx_last_shape": list(shape)}


def stage_parity_chunks(revision: str, dest_root: Path, n_chunks: int) -> list[str]:
    """Stage the first N capture-chunk PAIRS for the 0.3 parity rig.

    The parity sample needs prompt + RESPONSE text: the .pt chunk (under
    ``final_token_capture/``) carries prompts + stored cx_last/v_x; the PAIRED
    raw json ``shardXX_chunkYYYY.json`` (under ``raw_completions/``, capture
    script L998-999 + _flush_upload_batch) carries {ci, prompt, response} rows.
    Deterministic: the lexicographically first N .pt chunks at the pin.
    """
    api = _api()
    base = "issue779_monitoring/fitter-fair-comparison-n1m"
    cap_files = sorted(
        hub.list_hf_files_under_path(
            api,
            C76.HF_DATA_REPO,
            f"{base}/final_token_capture",
            repo_type="dataset",
            revision=revision,
        )
    )
    raw_files = set(
        hub.list_hf_files_under_path(
            api,
            C76.HF_DATA_REPO,
            f"{base}/raw_completions",
            repo_type="dataset",
            revision=revision,
        )
    )
    pt_files = [f for f in cap_files if f.endswith(".pt")]
    assert pt_files, f"no .pt chunks under {base}/final_token_capture"
    staged: list[str] = []
    for pt in pt_files[:n_chunks]:
        raw = f"{base}/raw_completions/{re.sub(r'[.]pt$', '.json', Path(pt).name)}"
        assert raw in raw_files, f"paired raw json {raw} absent for chunk {pt}"
        for repo_path in (pt, raw):
            dest = dest_root / repo_path
            if not dest.exists():
                hub.stage_hub_file(
                    C76.HF_DATA_REPO, repo_path, dest, repo_type="dataset", revision=revision
                )
            staged.append(repo_path)
        print(f"[stage] parity chunk staged: {pt} (+raw)", flush=True)
    return staged


def stage_trait_artifacts(revision: str, repo_root: Path) -> list[str]:
    """Stage the #779 trait artifacts the contexts builder reads (gitignored).

    issue779_common.load_extraction_artifacts reads
    ``data/issue_779/artifacts/<trait>.json`` relative to the repo root; the
    files are gitignored, so a fresh clone must stage them from the Hub.
    """
    art_dir = repo_root / "data" / "issue_779" / "artifacts"
    staged: list[str] = []
    for trait in ("sycophancy", "hallucination"):
        dest = art_dir / f"{trait}.json"
        hub.stage_hub_file(
            C76.HF_DATA_REPO,
            f"issue779_monitoring/artifacts/{trait}.json",
            dest,
            repo_type="dataset",
            revision=revision,
        )
        print(f"[stage-extra] trait artifacts staged: {dest}", flush=True)
        staged.append(str(dest))
    return staged


def stage_centroids(repo_root: Path, dest: Path) -> str:
    """Stage + sha-verify the canonical-pool centroids bundle (5b leakage leg).

    Downloads ``CENTROIDS_HF_PATH`` (default branch — the bundle predates and is
    outside the #1776 revision pin) and asserts its sha256 against the committed
    bank meta's ``built_from.centroids_sha256``; returns the verified sha.
    """
    meta = json.loads((repo_root / CENTROIDS_META_RELPATH).read_text())
    want_sha = meta["built_from"]["centroids_sha256"]
    hub.stage_hub_file(C76.HF_DATA_REPO, CENTROIDS_HF_PATH, dest, repo_type="dataset")
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert got == want_sha, f"centroids sha mismatch: got {got} want {want_sha}"
    print(f"[stage-extra] centroids staged + sha-verified: {dest}", flush=True)
    return got


def stage_prefix(revision: str, dest_root: Path, prefix: str) -> int:
    """Stage a whole reused prefix at the pin (weights, r_b, sampling manifest)."""
    got = hub.stage_hub_prefix(
        C76.HF_DATA_REPO, prefix, dest_root, repo_type="dataset", revision=revision
    )
    print(f"[stage] staged {len(got)} files under {prefix}", flush=True)
    return len(got)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pin-file", type=Path, default=C76.PIN_FILE)
    ap.add_argument("--refresh-pin", action="store_true", help="re-resolve the revision pin")
    ap.add_argument("--dest", type=Path, default=C76.DATA_DIR / "hf_dl")
    ap.add_argument("--stage-bundle", action="store_true", help="download the 6.02 GB pass_b")
    ap.add_argument("--stage-weights", action="store_true", help="stage n1m_readout weights")
    ap.add_argument("--stage-rb", action="store_true", help="stage r_B trait stacks")
    ap.add_argument("--stage-manifest", action="store_true", help="stage the sampling manifest")
    ap.add_argument("--parity-chunks", type=int, default=0, help="stage N parity chunk pairs")
    ap.add_argument(
        "--stage-trait-artifacts",
        action="store_true",
        help="stage the #779 trait artifact JSONs (contexts-builder input)",
    )
    ap.add_argument(
        "--stage-centroids",
        action="store_true",
        help="stage + sha-verify the #483 centroids bundle (5b leakage leg)",
    )
    ap.add_argument(
        "--centroids-dest",
        type=Path,
        default=C76.DATA_DIR / "centroids_v1_L21.pt",
        help="local target for --stage-centroids",
    )
    ap.add_argument("--skip-provenance", action="store_true")
    ap.add_argument("--report", type=Path, default=C76.DATA_DIR / "stage_report.json")
    args = ap.parse_args(argv)

    revision = C76.resolve_data_repo_pin(args.pin_file, refresh=args.refresh_pin)
    print(f"[stage] data-repo pin: {C76.HF_DATA_REPO}@{revision}", flush=True)

    report: dict = {"revision": revision, "repro": C76.repro_meta()}
    report["stem_probe_counts"] = probe_stems(revision)
    if not args.skip_provenance:
        report["provenance_chain"] = provenance_check(revision)
    if args.stage_manifest:
        report["n_manifest_files"] = stage_prefix(
            revision, args.dest, "issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest"
        )
    if args.stage_weights:
        report["n_weight_files"] = stage_prefix(
            revision, args.dest, "issue779_monitoring/n1m_readout/weights"
        )
    if args.stage_rb:
        report["n_rb_files"] = stage_prefix(revision, args.dest, "issue779_monitoring/r_b")
    if args.stage_bundle:
        bundle = stage_pass_b(revision, args.dest)
        report["pass_b_keys"] = verify_pass_b_keys(bundle)
    if args.parity_chunks > 0:
        report["parity_chunks"] = stage_parity_chunks(revision, args.dest, args.parity_chunks)
    if args.stage_trait_artifacts:
        report["trait_artifacts"] = stage_trait_artifacts(revision, C76.PROJECT_ROOT)
    if args.stage_centroids:
        report["centroids_sha256"] = stage_centroids(C76.PROJECT_ROOT, args.centroids_dest)

    C76.atomic_write_json(args.report, report)
    print(f"[stage] [phase=stage_done] report -> {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
