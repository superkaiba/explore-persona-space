#!/usr/bin/env python
"""verify_reused_artifact_keys.py — mechanized realized-keys probe (task #1164).

Executable form of .claude/rules/artifact-reuse.md check (c): verify a reused
multi-field bundle's REALIZED top-level key set is a superset of the keys the
consumer asserts, without materializing tensor storages. Exit 0 = superset
holds; 1 = missing keys; 2 = load/usage error. Never falls back silently.

SCOPE: verifies TOP-LEVEL key presence only — nested sub-keys, tensor
shapes/dtypes, and row counts are NOT checked; the consumer's own loader run
against the pinned artifact remains the stronger form (artifact-reuse.md (c)).

Formats (by extension when ``--fmt auto``; unknown extension = error):

- ``.pt`` / ``.pth`` / ``.bin`` — ``torch.load(path, map_location="cpu",
  mmap=True, weights_only=True)``: the mmap read returns the dict without
  materializing storages (zipfile serialization required — the torch.save
  default since 1.6). A legacy non-zipfile file or non-primitive metadata
  fails LOUD (exit 2) unless ``--allow-full-load`` (full read) and/or
  ``--no-weights-only`` (sha-pinned SELF-PRODUCED bundles only, per
  .claude/rules/gotchas.md) are passed explicitly.
- ``.safetensors`` — ``safetensors.safe_open(...).keys()``, a header-only
  read by design. Sharded ``*.safetensors.index.json`` layouts are a
  documented v1 limitation: point the probe at a shard, or read the sharded
  index's ``weight_map`` manually (the project's multi-field bundles are
  single ``.pt`` files).
- ``.json`` — ``json.loads``; dict root required; top-level keys.

Usage::

    uv run python scripts/verify_reused_artifact_keys.py \
        --artifact path/to/bundle.pt --keys cx_last,v_x,layers
    uv run python scripts/verify_reused_artifact_keys.py \
        --hf-repo superkaiba1/explore-persona-space-data \
        --hf-path issueN_slug/analysis_tensors/bundle.pt \
        [--repo-type dataset] [--revision <sha>] --keys-file keys.txt [--json]

Exit codes: 0 PASS (realized keys are a superset of declared), 1 MISSING
(at least one declared key absent — the missing set is printed), 2 ERROR
(unreadable file, non-dict root, legacy format without --allow-full-load,
unknown extension; argparse usage errors keep argparse's own exit 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REALIZED_JSON_CAP = 200  # --json: cap the realized-key listing, keep output bounded


class BundleFormatError(RuntimeError):
    """A bundle that cannot be key-read under the requested (safe) regime."""


def _torch_realized_keys(path: Path, *, weights_only: bool, allow_full_load: bool) -> set[str]:
    """Top-level keys of a torch save-dict via an mmap read (no storage
    materialization). Falls back to a FULL load only under the explicit
    ``allow_full_load`` opt-in — never silently."""
    import torch

    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=weights_only)
    except Exception as e:
        if not allow_full_load:
            raise BundleFormatError(
                f"mmap load failed ({e!r}); legacy non-zipfile .pt or non-primitive "
                "metadata? Re-run with --allow-full-load (full read) and/or "
                "--no-weights-only (sha-pinned SELF-PRODUCED bundles only, per "
                ".claude/rules/gotchas.md)"
            ) from e
        obj = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(obj, dict):
        raise BundleFormatError(f"bundle root is {type(obj).__name__}, not a dict")
    return {str(k) for k in obj.keys()}  # noqa: SIM118 — explicit .keys() mirrors the rule text


def _safetensors_realized_keys(path: Path) -> set[str]:
    """Tensor names from a safetensors file (header-only read by design)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return set(f.keys())


def _json_realized_keys(path: Path) -> set[str]:
    """Top-level keys of a JSON file; dict root required."""
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise BundleFormatError(f"bundle root is {type(obj).__name__}, not a dict")
    return {str(k) for k in obj.keys()}  # noqa: SIM118 — symmetry with the torch branch


def realized_keys(
    path: Path,
    *,
    fmt: str = "auto",
    weights_only: bool = True,
    allow_full_load: bool = False,
) -> set[str]:
    """Return the artifact's realized top-level key set.

    ``fmt="auto"`` dispatches on extension (.pt/.pth/.bin -> torch,
    .safetensors -> safetensors header, .json -> JSON dict root); an unknown
    extension raises ``BundleFormatError`` (exit 2 at the CLI) rather than
    guessing. Raises on unreadable files / non-dict roots — never returns a
    partial or empty set on error.
    """
    if not path.exists():
        raise BundleFormatError(f"artifact not found: {path}")
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in (".pt", ".pth", ".bin"):
            fmt = "pt"
        elif suffix == ".safetensors":
            fmt = "safetensors"
        elif suffix == ".json":
            fmt = "json"
        else:
            raise BundleFormatError(
                f"unknown extension {suffix!r} for fmt=auto — pass --fmt pt|safetensors|json"
            )
    if fmt == "pt":
        return _torch_realized_keys(
            path, weights_only=weights_only, allow_full_load=allow_full_load
        )
    if fmt == "safetensors":
        return _safetensors_realized_keys(path)
    if fmt == "json":
        return _json_realized_keys(path)
    raise BundleFormatError(f"unknown fmt {fmt!r}")


def missing_keys(realized: set[str], declared: set[str]) -> set[str]:
    """Declared keys absent from the realized set (empty = superset holds)."""
    return declared - realized


def parse_keys_arg(keys: str | None, keys_file: Path | None) -> set[str]:
    """Declared-key set from ``--keys`` (comma/whitespace split) or
    ``--keys-file`` (one per line; blank lines + ``#`` comments skipped).
    Exactly one source must be provided; an empty resulting set is an error."""
    if (keys is None) == (keys_file is None):
        raise BundleFormatError("exactly one of --keys / --keys-file is required")
    if keys is not None:
        parts = [t for t in keys.replace(",", " ").split() if t]
    else:
        parts = []
        for line in keys_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts.append(stripped)
    declared = set(parts)
    if not declared:
        raise BundleFormatError("declared key set is empty")
    return declared


def resolve_artifact(args: argparse.Namespace) -> Path:
    """Local path from ``--artifact``, or an ``hf_hub_download`` of
    ``--hf-repo``/``--hf-path`` (+ ``--repo-type``/``--revision``).

    HF mode loads the project dotenv wrapper BEFORE any huggingface_hub
    import/use (HF_TOKEN + HF_HOME; workflow_lint dotenv-before-hf-import
    convention). Note the download fetches the WHOLE file just to read keys —
    prefer --artifact where the bundle is already staged.
    """
    if args.artifact is not None:
        return Path(args.artifact)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=args.hf_repo,
        filename=args.hf_path,
        repo_type=args.repo_type,
        revision=args.revision,
    )
    return Path(local)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser (mutually-exclusive artifact source + key source)."""
    parser = argparse.ArgumentParser(
        description="Verify a reused bundle's realized top-level keys cover the consumer's "
        "declared keys (artifact-reuse.md check (c), incident #1073).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact", help="local path to the bundle")
    src.add_argument("--hf-repo", help="HF repo id (with --hf-path)")
    parser.add_argument("--hf-path", help="path inside --hf-repo")
    parser.add_argument(
        "--repo-type", choices=("dataset", "model"), default="dataset", help="HF repo type"
    )
    parser.add_argument("--revision", default=None, help="HF revision (pin a sha)")
    keys = parser.add_mutually_exclusive_group(required=True)
    keys.add_argument("--keys", help="comma/whitespace-separated declared keys")
    keys.add_argument(
        "--keys-file",
        type=Path,
        help="file of declared keys, one per line ('#' comments + blank lines skipped)",
    )
    parser.add_argument(
        "--fmt",
        choices=("auto", "pt", "safetensors", "json"),
        default="auto",
        help="bundle format (default: by extension)",
    )
    parser.add_argument(
        "--allow-full-load",
        action="store_true",
        help="explicit opt-in: full (non-mmap) torch read for legacy non-zipfile .pt",
    )
    parser.add_argument(
        "--no-weights-only",
        action="store_true",
        help="torch.load(weights_only=False) — sha-pinned SELF-PRODUCED bundles only "
        "(.claude/rules/gotchas.md)",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    return parser


def _emit(args: argparse.Namespace, payload: dict, human_lines: list[str]) -> None:
    """One report to stdout: JSON when ``--json``, else the human lines."""
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in human_lines:
            print(line)


def main(argv: list[str] | None = None) -> int:
    """Exit contract: 0 PASS / 1 MISSING / 2 ERROR (argparse usage errors
    keep argparse's own exit 2)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hf_repo is not None and args.hf_path is None:
        parser.error("--hf-repo requires --hf-path")
    fmt = args.fmt
    artifact_label = (
        args.artifact if args.artifact is not None else f"{args.hf_repo}/{args.hf_path}"
    )
    try:
        declared = parse_keys_arg(args.keys, args.keys_file)
        path = resolve_artifact(args)
        realized = realized_keys(
            path,
            fmt=fmt,
            weights_only=not args.no_weights_only,
            allow_full_load=args.allow_full_load,
        )
    except Exception as e:
        _emit(
            args,
            {
                "artifact": artifact_label,
                "format": fmt,
                "status": "ERROR",
                "error": f"{type(e).__name__}: {e}",
            },
            [f"ERROR — {type(e).__name__}: {e}"],
        )
        return 2
    missing = missing_keys(realized, declared)
    status = "MISSING" if missing else "PASS"
    payload = {
        "artifact": artifact_label,
        "format": fmt,
        "status": status,
        "n_realized": len(realized),
        "declared": sorted(declared),
        "missing": sorted(missing),
        "realized": sorted(realized)[:REALIZED_JSON_CAP],
    }
    human = [
        f"REALIZED n={len(realized)} keys",
        f"DECLARED n={len(declared)}",
    ]
    if missing:
        human.append(f"MISSING: {sorted(missing)}")
    else:
        human.append("PASS — realized keys are a superset of declared keys")
    _emit(args, payload, human)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
