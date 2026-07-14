"""Regression replay for task #395's marker-prior probe (task #401 §5.2).

Re-runs the refactored :mod:`scripts.i395_probe_marker_priors` (which now
calls :func:`compute_marker_logprob` instead of inline teacher-forcing) and
diffs every per-marker numeric field against the pinned v1 baseline at
``eval_results/issue_395/marker_priors.json``. Exits 0 on PASS, 1 on FAIL.

Run on a ``lora-7b`` GPU pod with ``Qwen/Qwen2.5-7B-Instruct`` cached:

.. code-block:: bash

    uv run python scripts/regression_replay_i395.py

This script requires a CUDA GPU and a downloaded copy of Qwen-2.5-7B-Instruct.
It is NOT a CPU regression test; the unit tests in
``tests/test_marker_abstraction.py`` cover CPU correctness of
``compute_marker_logprob`` against an inline reference computation.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path so the deferred
# `import scripts.i395_probe_marker_priors` in main() resolves in script
# mode (sys.path[0] is scripts/, not the repo root — #823/#853).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ATOL = 1e-2


def _diff_summary(
    baseline: dict[str, dict[str, float]],
    replay: dict[str, dict[str, float]],
) -> list[str]:
    """Compare every numeric field in ``summary`` block. Returns failure messages."""
    failures: list[str] = []
    baseline_markers = set(baseline.keys())
    replay_markers = set(replay.keys())
    if baseline_markers != replay_markers:
        failures.append(
            f"Marker keys diverged: baseline={sorted(baseline_markers)!r} "
            f"vs replay={sorted(replay_markers)!r}"
        )
        return failures
    for marker in sorted(baseline_markers):
        b_row = baseline[marker]
        r_row = replay[marker]
        # Check every numeric key; integer-typed fields ('n', 'n_marker_tokens')
        # must be exactly equal, float-typed fields within ATOL.
        for key in sorted(b_row.keys()):
            b_val = b_row[key]
            r_val = r_row.get(key)
            if r_val is None:
                failures.append(f"{marker}.{key} missing from replay")
                continue
            if isinstance(b_val, int) and isinstance(r_val, int):
                if b_val != r_val:
                    failures.append(
                        f"{marker}.{key} integer mismatch: baseline={b_val} replay={r_val}"
                    )
            else:
                try:
                    diff = abs(float(b_val) - float(r_val))
                except (TypeError, ValueError) as e:
                    failures.append(
                        f"{marker}.{key} non-numeric or uncomparable: "
                        f"baseline={b_val!r} replay={r_val!r} ({e})"
                    )
                    continue
                if diff > ATOL:
                    failures.append(
                        f"{marker}.{key} diverged by {diff:.6f} "
                        f"(baseline={b_val!r}, replay={r_val!r}, atol={ATOL})"
                    )
    return failures


def _diff_per_persona(
    baseline: dict[str, dict[str, list[float]]],
    replay: dict[str, dict[str, list[float]]],
) -> list[str]:
    """Compare per-persona log-prob lists element-wise within ATOL."""
    failures: list[str] = []
    for marker in sorted(set(baseline.keys()) | set(replay.keys())):
        b_personas = baseline.get(marker, {})
        r_personas = replay.get(marker, {})
        if set(b_personas.keys()) != set(r_personas.keys()):
            failures.append(
                f"per_persona[{marker}] persona keys diverged: "
                f"baseline={sorted(b_personas)!r} replay={sorted(r_personas)!r}"
            )
            continue
        for persona in sorted(b_personas):
            b_vals = b_personas[persona]
            r_vals = r_personas[persona]
            if len(b_vals) != len(r_vals):
                failures.append(
                    f"per_persona[{marker}][{persona}] length mismatch: "
                    f"baseline n={len(b_vals)} replay n={len(r_vals)}"
                )
                continue
            for i, (b, r) in enumerate(zip(b_vals, r_vals, strict=True)):
                diff = abs(float(b) - float(r))
                if diff > ATOL:
                    failures.append(
                        f"per_persona[{marker}][{persona}][{i}] diverged by {diff:.6f} "
                        f"(baseline={b!r}, replay={r!r}, atol={ATOL})"
                    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay task #395's marker-prior probe (refactored to call "
            "compute_marker_logprob) and diff against the v1 baseline."
        )
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("eval_results/issue_395/marker_priors.json"),
        help="Pinned baseline JSON to diff against (default: %(default)s).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Where the replay should write its JSON. Defaults to a temp file "
            "so the baseline at eval_results/issue_395/marker_priors.json is "
            "not overwritten."
        ),
    )
    args = parser.parse_args()

    baseline_path: Path = args.baseline_json
    if not baseline_path.exists():
        print(f"FAIL: baseline {baseline_path} does not exist", file=sys.stderr)
        return 1

    # Set up the replay output path. We point the probe script's OUT_PATH at a
    # temp file so the baseline is never clobbered by the replay run.
    if args.output_json is None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_replay_marker_priors.json", delete=False
        ) as tmp:
            output_path = Path(tmp.name)
    else:
        output_path = args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Replay output → {output_path}")
    print(f"Baseline      → {baseline_path}")

    # Import + monkeypatch + invoke the probe's main(). Doing it in-process
    # keeps the model load on a single GPU and avoids a second uv-run subshell.
    import scripts.i395_probe_marker_priors as probe

    probe.OUT_PATH = output_path
    probe.main()

    # Diff phase.
    baseline = json.loads(baseline_path.read_text())
    replay = json.loads(output_path.read_text())

    failures: list[str] = []
    failures.extend(_diff_summary(baseline.get("summary", {}), replay.get("summary", {})))
    failures.extend(
        _diff_per_persona(baseline.get("per_persona", {}), replay.get("per_persona", {}))
    )

    n_markers = len(baseline.get("summary", {}))
    if failures:
        print(f"\nFAIL: {len(failures)} divergence(s) across {n_markers} marker(s):")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print(f"\nPASS: all {n_markers} markers match within atol={ATOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
