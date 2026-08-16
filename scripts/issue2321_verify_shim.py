"""#2321 live post-repack shim check — READ-ONLY resolution smoke (plan §3.7/§12).

For each target prefix, samples ``--samples`` packed members (deterministic
per-prefix seed) and stages each ORIGINAL path through the PRODUCTION consumer
route — ``orchestrate.hub.stage_hub_file``, whose raw-path miss falls back to
the ``packed/`` shim — then sha256-checks the staged bytes against the
member's recorded digest. Prints ``ok k/n`` per prefix and a final ``X/Y``
summary; exit 0 iff every sampled member resolved byte-exactly.

This is the RESOLUTION smoke, deliberately NOT the content-integrity
instrument: 3 samples/prefix would hit an overwritten member with probability
~0 — content integrity is the driver's I13(c) exhaustive blob-anchor
postverify. A prefix with no pack yet FAILS (the tool's acceptance use is
post-repack; ``--allow-unpacked`` downgrades that to a reported skip for
mid-campaign runs). Zero mutations: listings + downloads only, no
``create_commit``, no uploads.

Usage:
    uv run python scripts/issue2321_verify_shim.py --samples 3
    uv run python scripts/issue2321_verify_shim.py --prefixes issue1090_partial --samples 3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import random
import sys
import tempfile
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

DATA_REPO = "superkaiba1/explore-persona-space-data"


def _load_driver():
    """Load the sibling repack driver (single source for the prefix order)."""
    path = Path(__file__).with_name("issue2321_repack.py")
    spec = importlib.util.spec_from_file_location("issue2321_repack_driver", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_prefix_samples(
    api,
    *,
    repo_id: str,
    prefix: str,
    n_samples: int,
    stage_root: Path,
) -> tuple[int, int, list[str]]:
    """(ok, total, problems) over ``n_samples`` members of one prefix's pack."""
    from explore_persona_space.orchestrate import hub

    members = hub.packed_members_under_path(api, repo_id, prefix, repo_type="dataset")
    if not members:
        return 0, 0, [f"{prefix}: no packed members (prefix not repacked yet?)"]
    rng = random.Random(f"i2321-verify-shim:{prefix}")
    picked = rng.sample(members, min(n_samples, len(members)))
    ok = 0
    problems: list[str] = []
    for m in picked:
        target = stage_root / m.path
        try:
            staged = hub.stage_hub_file(repo_id, m.path, target, repo_type="dataset")
        except Exception as err:  # a resolution failure is the finding
            problems.append(f"{prefix}: {m.path}: {type(err).__name__}: {err}")
            continue
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        if digest != m.sha256:
            problems.append(
                f"{prefix}: {m.path}: sha256 mismatch (staged {digest[:12]}..., "
                f"recorded {m.sha256[:12]}...)"
            )
            continue
        ok += 1
    return ok, len(picked), problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=3, help="members sampled per prefix")
    ap.add_argument("--repo-id", default=DATA_REPO)
    ap.add_argument(
        "--prefixes",
        default=None,
        help="comma-separated prefix subset (default: the driver's full walk order)",
    )
    ap.add_argument(
        "--allow-unpacked",
        action="store_true",
        help="report a pack-less prefix as SKIP instead of FAIL (mid-campaign runs)",
    )
    ap.add_argument("--import-check", action="store_true", help="argparse/attr check only")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[verify-shim] import-check ok")
        return 0

    driver = _load_driver()
    prefixes = args.prefixes.split(",") if args.prefixes else list(driver.PREFIX_ORDER)

    from huggingface_hub import HfApi

    api = HfApi()
    total_ok = total_n = 0
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="i2321_verify_shim_") as td:
        stage_root = Path(td)
        for prefix in prefixes:
            ok, n, problems = verify_prefix_samples(
                api,
                repo_id=args.repo_id,
                prefix=prefix,
                n_samples=args.samples,
                stage_root=stage_root,
            )
            total_ok += ok
            total_n += n
            if n == 0:
                if args.allow_unpacked:
                    print(f"[verify-shim] {prefix}: SKIP (no pack yet)")
                else:
                    print(f"[verify-shim] {prefix}: FAIL (no pack)")
                    failures.extend(problems)
                continue
            status = "ok" if ok == n and not problems else "FAIL"
            print(f"[verify-shim] {prefix}: {status} {ok}/{n}")
            failures.extend(problems)
    for line in failures:
        print(f"[verify-shim] problem: {line}")
    # A smoke that sampled NOTHING certifies nothing — 0/0 is never a PASS.
    verdict = "PASS" if not failures and total_n > 0 and total_ok == total_n else "FAIL"
    print(f"[verify-shim] {verdict}: {total_ok}/{total_n} sampled members resolved byte-exactly")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
