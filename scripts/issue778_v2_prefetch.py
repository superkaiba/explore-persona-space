#!/usr/bin/env python
"""Issue #778 v2 — stage + sha-pin the reused v1 inputs (and the pod's v2 bundle).

The v2 ladder consumes reused v1 artifacts (r_B v1, extraction pools, monitoring
acts, finetune activations) plus the pod-produced v2 bundle. This script stages
them into the consumer path layout ``<out-root>/...`` with content pinning
(plan v8 §12 assumptions 3-5 / artifact-reuse (f)):

  1. Source preference: the local mirror (``--local-mirror``, default the main
     repo's ``data/issue_778`` — the exact copy the committed ladder consumed),
     else ``hf_hub_download`` from the NESTED HF path
     ``issue778_persona_vectors/analysis_tensors/<rel>`` (fact-checker path
     correction: NEVER the bare ``rb/`` prefix).
  2. sha256 asserts: the v1 r_B pins from plan v5 §12(f) AND the committed
     honest-nulls ``tensor_sha256`` map (the shas the old-r_B ladder recorded
     for every input it consumed). Fail-loud on any mismatch.
  3. ``--fetch-v2``: additionally pull ``analysis_tensors_v2/{extract,neutral,
     rb_v2,judge,pairing}`` from HF into ``<out-root>/v2/`` (the VM phase after
     pod release, when the bundle is not already local).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue778_lib as lib  # noqa: E402

TRAITS = lib.TRAITS
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_V1_PREFIX = "issue778_persona_vectors/analysis_tensors"  # NESTED (fact-checker)
HF_V2_PREFIX = "issue778_persona_vectors/analysis_tensors_v2"

# v1 r_B content pins (plan v5 §12(f), recorded from HF repo_info lfs sha256).
RB_V1_SHA256 = {
    "evil": "67d1caafe536f11de29367b48a59f3c6bd372d01a6c44f46a82c6203b1c5ebdb",
    "hallucination": "8bea89cd0e2f43eb902d0fcff544a3eed2fc4006ec79b3bd440b785852db4a6f",
    "sycophancy": "20e498a2a3aca5450c731ac031cc13d887080a432b355e84055bc664d6087ec5",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _committed_pins(eval_root: Path) -> dict:
    """The committed old-r_B ladder's tensor_sha256 pins (authoritative for
    every v1 input it consumed). Each committed per-cell JSON carries a FLAT
    per-trait slice (keys rb / activations_pos / activations_neg /
    monitoring_*_acts) — assemble the nested {trait: {...}} map from each
    trait's own file. (finetune base.pt carried no per-file pin — recorded
    only.)"""
    pins: dict = {}
    for trait in TRAITS:
        path = eval_root / "honest_nulls" / f"{trait}_finetune_honestnulls.json"
        if path.exists():
            with open(path) as f:
                slice_ = json.load(f).get("tensor_sha256", {})
            if slice_:
                pins[trait] = slice_
    return pins


def _required_v1_rels() -> list[str]:
    rels = []
    for t in TRAITS:
        rels += [
            f"rb/{t}.pt",
            f"activations/{t}_pos.pt",
            f"activations/{t}_neg.pt",
            f"monitoring_corrected/{t}_acts.pt",
            f"monitoring_manyshot/{t}_acts.pt",
        ]
    rels.append("finetune_activations/base.pt")
    for fam in lib.FAMILIES:
        for ver in lib.VERSIONS:
            rels.append(f"finetune_activations/{fam}_{ver}.pt")
    return rels


def _stage_one(rel: str, out_root: Path, local_mirror: Path | None) -> Path:
    dest = out_root / rel
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if local_mirror is not None and (local_mirror / rel).exists():
        shutil.copy2(local_mirror / rel, dest)
        return dest
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(HF_REPO, f"{HF_V1_PREFIX}/{rel}", repo_type="dataset", revision="main")
    shutil.copy2(got, dest)
    return dest


def _pin_for(rel: str, pins: dict) -> str | None:
    """Expected sha for a staged v1 file, from the committed tensor_sha256 map
    (+ the v5 rb pin table as a redundant cross-check for rb/)."""
    parts = rel.split("/")
    if parts[0] == "rb":
        trait = parts[1].removesuffix(".pt")
        v5_pin = RB_V1_SHA256.get(trait)
        committed = pins.get(trait, {}).get("rb")
        if v5_pin and committed and v5_pin != committed:
            raise RuntimeError(
                f"pin-table disagreement for rb/{trait}: v5 {v5_pin} vs committed {committed}"
            )
        return v5_pin or committed
    for t in TRAITS:
        tp = pins.get(t, {})
        if rel == f"activations/{t}_pos.pt":
            return tp.get("activations_pos")
        if rel == f"activations/{t}_neg.pt":
            return tp.get("activations_neg")
        if rel == f"monitoring_corrected/{t}_acts.pt":
            return tp.get("monitoring_corrected_acts")
        if rel == f"monitoring_manyshot/{t}_acts.pt":
            return tp.get("monitoring_manyshot_acts")
    if rel == "finetune_activations/base.pt":
        return pins.get("finetune_base", {}).get("base")
    return None  # per-cell finetune tags carry no committed pin — recorded only


def fetch_v2_bundle(out_root: Path) -> int:
    """Pull the pod-produced analysis_tensors_v2 bundle from HF into out_root/v2.

    A PRE-EXISTING local file is never trusted blindly (the stub-poisoning
    residual): it is verified against the Hub entry — byte size always, plus
    the LFS sha256 where the Hub records one — and a mismatch fails LOUD
    (delete the local copy and re-fetch; the Hub copy is the pod-verified
    canonical bundle)."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    entries = [
        e
        for e in HfApi().list_repo_tree(
            HF_REPO,
            path_in_repo=HF_V2_PREFIX,
            repo_type="dataset",
            revision="main",
            recursive=True,
        )
        if isinstance(e, RepoFile)
    ]
    n_fetched = 0
    n_verified = 0
    for e in entries:
        rel = e.path[len(HF_V2_PREFIX) + 1 :]
        dest = out_root / "v2" / rel
        if dest.exists():
            lfs = getattr(e, "lfs", None)
            hub_sha = (
                (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
                if lfs is not None
                else None
            )
            size_ok = dest.stat().st_size == e.size
            sha_ok = hub_sha is None or _sha256(dest) == hub_sha
            if not (size_ok and sha_ok):
                raise RuntimeError(
                    f"prefetch: pre-existing local v2 file {dest} does NOT match the Hub "
                    f"copy (local {dest.stat().st_size}B vs hub {e.size}B; sha256 "
                    f"{'mismatch' if hub_sha is not None else 'unavailable — size mismatch'}) "
                    "— possible stub/stale artifact; delete the local file and re-run "
                    "--fetch-v2 (the Hub bundle is the pod-verified canonical copy)."
                )
            n_verified += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(HF_REPO, e.path, repo_type="dataset", revision="main")
        shutil.copy2(got, dest)
        n_fetched += 1
    print(
        f"[prefetch] v2 bundle: {len(entries)} on Hub, {n_fetched} fetched, "
        f"{n_verified} pre-existing verified (size+sha)",
        flush=True,
    )
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #778 v2 input prefetch + sha pinning.")
    ap.add_argument("--out-root", default="data/issue_778")
    ap.add_argument("--eval-results-root", default="eval_results/issue_778")
    ap.add_argument(
        "--local-mirror",
        default="/home/thomasjiralerspong/explore-persona-space/data/issue_778",
        help="local copy the committed ladder consumed (preferred source; '' disables)",
    )
    ap.add_argument("--fetch-v2", action="store_true", help="also fetch analysis_tensors_v2/")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)
    local_mirror = Path(args.local_mirror) if args.local_mirror else None
    if local_mirror is not None and not local_mirror.exists():
        local_mirror = None

    pins = _committed_pins(eval_root)
    if not pins:
        print("[prefetch] WARNING: no committed tensor_sha256 pins found", flush=True)
    report: dict = {"staged": {}, "pin_failures": []}
    for rel in _required_v1_rels():
        dest = _stage_one(rel, out_root, local_mirror)
        got = _sha256(dest)
        want = _pin_for(rel, pins)
        entry = {"sha256": got, "pinned": want is not None}
        if want is not None and got != want:
            report["pin_failures"].append({"rel": rel, "want": want, "got": got})
        report["staged"][rel] = entry
    if report["pin_failures"]:
        raise RuntimeError(
            f"sha256 PIN FAILURES on {len(report['pin_failures'])} staged v1 inputs "
            f"(first: {report['pin_failures'][0]}) — content drift vs the committed "
            "ladder's inputs; do NOT proceed (artifact-reuse (f))."
        )
    if args.fetch_v2:
        report["v2_files_on_hub"] = fetch_v2_bundle(out_root)
    report["reproducibility"] = lib.repro_metadata()
    out_path = out_root / "v2" / "prefetch_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"phase": "prefetch", "n_staged": len(report["staged"])}, indent=2))


if __name__ == "__main__":
    main()
