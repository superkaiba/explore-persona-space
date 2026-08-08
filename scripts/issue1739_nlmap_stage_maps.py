#!/usr/bin/env python
"""Stage issue-1739 nonlinear map payloads from HF onto a fan-out lane's disk.

The phase-A prefetch fits every ``(variant, U rung, kind)`` map ONCE and uploads
the payloads; the 6 scoring lanes (behavior x kind) then each need those payloads
LOCAL, because ``issue1739_fits._load_nl_map`` reads
``<tensors-root>/maps/<variant>__u<label>__<kind>.pt`` off local disk. Without
this step every lane finds nothing, silently re-fits all 4 maps, and the whole
amortization the prefetch exists for evaporates (measured basis: ~2651 s per map
key — 4 keys x 6 lanes re-fit instead of 4 fit once).

Staging shape (the #1774 mirror-root trap): ``hub.stage_hub_prefix`` mirrors the
REPO-RELATIVE path under its dest, so it would land the payloads at
``<dest>/issue1739_ctxmap/analysis_tensors/maps/...`` — and no dest root can
satisfy ``root/<hub prefix> == <tensors-root>/maps`` here, since the Hub layout
(``issue1739_ctxmap/analysis_tensors``) and the local layout
(``analysis_tensors/issue_1739``) are not suffix-compatible. So this uses the
per-file ``hub.stage_hub_file``, which takes an EXACT target and has no such
trap, and then ASSERTS every expected payload sits at the path the consumer
actually opens.

Verification is a consumer-open probe, not a file-existence check: each staged
payload is loaded and checked against the static guards ``_load_nl_map`` itself
applies (``map_kind`` / layer list / payload count / held-out diagnostics /
``map_seed``). The one guard left to scoring time is ``w_fit_rows == n_u``, which
needs the realized U pool; a mismatch there is loud in the lane log and merely
costs a re-fit, never a wrong map.

Idempotent (an existing target short-circuits without network), fail-loud, and
safe to re-run on a resumed lane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402  (after load_dotenv: thread caps are frozen at torch import)

from explore_persona_space.orchestrate import hub  # noqa: E402

# Mirrors scripts/issue1739_upload.py: `--stage tensors` uploads the whole
# --tensors-root tree to `<HF_PREFIX>/analysis_tensors`, so the maps land under
# `<HF_PREFIX>/analysis_tensors/maps/`. Kept as a module constant (not derived by
# importing the uploader) so this script has no import-time dependency on it; the
# --hf-prefix flag is the override if a child issue renames the prefix.
HF_PREFIX = "issue1739_ctxmap"
MAPS_SUBDIR = "maps"

# The path-2 grid's map keys: 2 variants x 2 U rungs. Overridable per lane.
DEFAULT_VARIANTS = ("prefix_end", "context_end")
DEFAULT_U_LABELS = ("250", "full")
DEFAULT_KINDS = ("mlp", "kernel")


def map_filename(variant: str, u_label: str, kind: str) -> str:
    """Payload basename for one map key — mirrors ``issue1739_fits._map_path``.

    Duplicated deliberately: importing the fits module here would drag its whole
    (numpy/torch/experiment) import surface into a staging step that must run
    before any of it. Pinned against the real thing by
    ``tests/test_issue1739_nlmap.py::test_stage_maps_filename_matches_fits_map_path``.
    """
    return f"{variant}__u{u_label}__{kind}.pt"


def expected_keys(
    variants: tuple[str, ...], u_labels: tuple[str, ...], kinds: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """Every (variant, u_label, kind) map key this lane will ask ``_load_nl_map`` for."""
    return [(v, u, k) for k in kinds for v in variants for u in u_labels]


def check_payload(path: Path, variant: str, u_label: str, kind: str) -> dict:
    """Consumer-open probe: the static ``_load_nl_map`` guards, minus ``w_fit_rows``.

    Returns a record with ``ok`` plus the reasons a lane's reader would reject the
    payload; raises nothing (the caller decides fail-loud vs report).
    """
    rec: dict = {
        "path": str(path),
        "variant": variant,
        "u_label": u_label,
        "kind": kind,
        "ok": False,
        "reasons": [],
    }
    if not path.exists():
        rec["reasons"].append("missing")
        return rec
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — any load failure is a staging failure
        rec["reasons"].append(f"unreadable: {type(exc).__name__}: {exc}")
        return rec
    meta = (blob or {}).get("meta") or {}
    payloads = (blob or {}).get("payloads") or []
    layers = [int(x) for x in meta.get("layers") or []]
    diagnostics = meta.get("diagnostics")
    if meta.get("map_kind") != kind:
        rec["reasons"].append(f"map_kind {meta.get('map_kind')!r} != {kind!r}")
    if meta.get("variant") != variant:
        rec["reasons"].append(f"variant {meta.get('variant')!r} != {variant!r}")
    if str(meta.get("u_label")) != str(u_label):
        rec["reasons"].append(f"u_label {meta.get('u_label')!r} != {u_label!r}")
    if not layers:
        rec["reasons"].append("no layer list")
    if len(payloads) != len(layers):
        rec["reasons"].append(f"n_payloads {len(payloads)} != n_layers {len(layers)}")
    if not isinstance(diagnostics, dict) or not diagnostics.get("per_layer"):
        rec["reasons"].append("payload carries no per-layer diagnostics")
    rec.update(
        {
            "n_layers": len(layers),
            "layers": layers,
            "w_fit_rows": meta.get("w_fit_rows"),
            "map_seed": meta.get("map_seed"),
            "map_git_commit": meta.get("git_commit"),
            "map_ts": meta.get("ts"),
            "size_bytes": path.stat().st_size,
        }
    )
    rec["ok"] = not rec["reasons"]
    return rec


def stage(
    *,
    tensors_root: Path,
    variants: tuple[str, ...],
    u_labels: tuple[str, ...],
    kinds: tuple[str, ...],
    repo_id: str,
    hf_prefix: str,
    map_seed: int | None,
    dry_run: bool,
    allow_missing: bool,
) -> dict:
    """Stage + verify every expected map payload. Returns the report payload."""
    prefix = f"{hf_prefix}/analysis_tensors/{MAPS_SUBDIR}"
    dest_dir = Path(tensors_root) / MAPS_SUBDIR
    keys = expected_keys(variants, u_labels, kinds)
    wanted = {map_filename(*k): k for k in keys}

    from huggingface_hub import HfApi

    api = HfApi()
    # ONE server-side scoped tree walk (#833: never a bare full listing / a
    # snapshot_download against the ~1M-file data repo).
    remote = hub.list_hf_files_under_path(api, repo_id, prefix, repo_type="dataset")
    remote_names = {Path(r).name: r for r in remote}
    print(f"[stage-maps] {len(remote)} file(s) under {repo_id}:{prefix}", flush=True)

    to_stage = {n: r for n, r in remote_names.items() if n in wanted}
    unmatched = sorted(n for n in wanted if n not in remote_names)
    if unmatched:
        print(
            f"[stage-maps] {len(unmatched)} expected payload(s) NOT on the Hub: "
            + ", ".join(unmatched),
            flush=True,
        )

    staged: list[str] = []
    if dry_run:
        for name, rel in sorted(to_stage.items()):
            print(f"[stage-maps]   (dry-run) {rel} -> {dest_dir / name}", flush=True)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name, rel in sorted(to_stage.items()):
            target = dest_dir / name
            pre = target.exists()
            hub.stage_hub_file(repo_id, rel, target, repo_type="dataset")
            staged.append(name)
            print(
                f"[stage-maps] {'present' if pre else 'staged '} {name} "
                f"({target.stat().st_size / 2**20:.1f} MiB)",
                flush=True,
            )

    checks = [] if dry_run else [check_payload(dest_dir / map_filename(*k), *k) for k in keys]
    for rec in checks:
        if rec["ok"] and map_seed is not None and rec.get("map_seed") is not None:
            if int(rec["map_seed"]) != int(map_seed):
                rec["ok"] = False
                rec["reasons"].append(f"map_seed {rec['map_seed']} != lane seed {int(map_seed)}")
    bad = [r for r in checks if not r["ok"]]
    report = {
        "issue": 1739,
        "step": "nlmap_stage_maps",
        "repo_id": repo_id,
        "hf_prefix": prefix,
        "dest_dir": str(dest_dir),
        "expected": [map_filename(*k) for k in keys],
        "n_expected": len(keys),
        "n_remote_under_prefix": len(remote),
        "n_staged": len(staged),
        "n_ok": len(checks) - len(bad),
        "lane_map_seed": map_seed,
        "checks": checks,
        "dry_run": dry_run,
    }
    if bad and not (dry_run or allow_missing):
        for rec in bad:
            print(
                f"[stage-maps] FAIL {Path(rec['path']).name}: {'; '.join(rec['reasons'])}",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(
            f"[stage-maps] {len(bad)}/{len(keys)} expected map payload(s) unusable — the lane "
            "would silently re-fit every one of them (rerun phase-A / its upload, or pass "
            "--allow-missing to accept the re-fit cost deliberately)"
        )
    if dry_run:
        # No checks ran, so say nothing about verification: a "verified 0/8" line
        # here reads like a failure when it only means "dry-run".
        print(
            f"[stage-maps] dry-run: {len(to_stage)}/{len(keys)} expected payload(s) "
            "present on the Hub; no download, no verification"
        )
    elif bad:
        print(
            f"[stage-maps] WARNING: {len(bad)}/{len(keys)} payload(s) unusable; "
            "--allow-missing set, so the lane will RE-FIT them",
            flush=True,
        )
    else:
        print(f"[stage-maps] verified {len(checks)}/{len(keys)} payload(s) consumer-openable")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    ap.add_argument(
        "--u-labels",
        nargs="+",
        default=list(DEFAULT_U_LABELS),
        help="U-rung labels as they appear in the payload name (e.g. 250 full)",
    )
    ap.add_argument("--repo-id", default=hub.DEFAULT_DATASET_REPO)
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument(
        "--map-seed",
        type=int,
        default=None,
        help="this lane's seeds[0]; refuses a payload fit under a different seed "
        "(a subsampled rung's ROWS depend on it — the row-count guard cannot see that)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="warn instead of failing when a payload is absent/unusable (the lane re-fits it)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = stage(
        tensors_root=args.tensors_root,
        variants=tuple(args.variants),
        u_labels=tuple(str(x) for x in args.u_labels),
        kinds=tuple(args.kinds),
        repo_id=args.repo_id,
        hf_prefix=args.hf_prefix,
        map_seed=args.map_seed,
        dry_run=args.dry_run,
        allow_missing=args.allow_missing,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.out)
        print(f"[stage-maps] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit: this module imports torch, and a bare fall-off can hit the
    # PyGILState_Release finalize race that turns a COMPLETED phase into a
    # nonzero rc under the dispatcher's `set -e` (gotchas.md).
    sys.exit(main())
