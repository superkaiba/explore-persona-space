#!/usr/bin/env python
"""Shared resume/idempotency helpers for the #2054 phase drivers (C9/M6).

Every #2054 phase output gets a `.done.json` sidecar written ATOMICALLY after
the output itself lands, carrying two blocks:

- ``regime`` — EVERY output-affecting configuration key (the 4-axis cell key
  `variant__condition__form__model` plus seed / layer / caps / draw counts —
  the #722-r3 rule: a resume that ignores a regime key silently reuses wrong
  rows). A sidecar whose regime DIFFERS from the invocation's is a REFUSAL
  (the #1333 `_check_regime` convention: an out-root holding a run under a
  DIFFERENT regime fails loud; pass ``--overwrite`` or use a fresh output
  dir), never a silent skip and never a silent overwrite.
- ``inputs`` — sha256 identity of the consumed input file(s). A CHANGED input
  is a legitimate upstream refresh: the cell RECOMPUTES with a loud log line
  (the #1947 salvage-inputs rule: pin identity at record time, accept newer
  artifacts with an audit line, never hard-assert).

Sidecars live NEXT TO the outputs under ``data/issue_2054/...`` — never in
the drained ``/workspace/logs/issue-<N>-*.json`` sentinel namespace
(`.claude/rules/pod-side-reporting.md` requirement 3).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Dispositions returned by resume_disposition().
RUN = "run"  # no valid prior output — run the unit
SKIP = "skip"  # prior output complete under the SAME regime + inputs — skip
RECOMPUTE = "recompute"  # same regime, CHANGED inputs — recompute (logged)


class RegimeMismatch(RuntimeError):
    """Prior output at this path was produced under a DIFFERENT regime."""


def sidecar_path(out_path: Path) -> Path:
    """The done-sidecar path for an output file."""
    return out_path.with_name(out_path.name + ".done.json")


def file_sha256(path: Path) -> str:
    """Streaming sha256 of a file (inputs identity pin)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_done(
    out_path: Path, regime: dict, inputs: dict | None = None, extra: dict | None = None
) -> Path:
    """Atomically write the done sidecar AFTER the output landed.

    ``extra`` carries non-regime bookkeeping (row counts, wall time) a resumed
    invocation may want without re-reading the (possibly large) output.
    """
    from datetime import datetime, timezone

    sp = sidecar_path(out_path)
    payload = {
        "artifact": out_path.name,
        "regime": regime,
        "inputs": inputs or {},
        "extra": extra or {},
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    tmp = sp.with_name(sp.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, sp)
    return sp


def read_done(out_path: Path) -> dict | None:
    """The sidecar payload for an output, or None when absent/unreadable."""
    sp = sidecar_path(out_path)
    if not sp.is_file():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_done(out_path: Path) -> None:
    """Remove the sidecar (the --overwrite path clears before regenerating)."""
    sp = sidecar_path(out_path)
    if sp.is_file():
        sp.unlink()


def regime_values_equal(a, b) -> bool:
    """Value equality with NaN == NaN (floats only).

    A regime float can legitimately be NaN (e.g. the ladder's
    ``target_ceiling`` when the target cell's own ceiling is degenerate at
    smoke n); Python/JSON round-trip NaN as ``float('nan')`` and
    ``nan != nan``, so bare ``!=`` marks EVERY re-run "regime changed" and the
    unit recomputes forever (Unit F smoke: the ladder re-ran all pairs on
    re-entry with "regime keys changed: ['target_ceiling']").
    """
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # both NaN
    return a == b


def regime_diff(recorded: dict, expected: dict) -> list[str]:
    """Keys whose values differ between the recorded and expected regimes."""
    keys = sorted(set(recorded) | set(expected))
    return [k for k in keys if not regime_values_equal(recorded.get(k), expected.get(k))]


def soft_resume_ok(out_path: Path, regime: dict, inputs: dict | None = None) -> tuple[bool, str]:
    """(skip?, reason) — the RECOMPUTE-on-mismatch flavor (no refusal).

    For outputs whose sidecar makes them self-describing AND whose recompute
    atomically rewrites the whole unit (phase_a's per-variant prejudge /
    admission JSONLs): a regime or input mismatch simply recomputes — the
    atomic rewrite leaves no mixed-regime residue. Contrast
    `resume_disposition`, which REFUSES a regime mismatch for outputs a
    recompute would silently clobber across regimes.
    """
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return False, "output missing or empty"
    payload = read_done(out_path)
    if payload is None:
        return False, "no done sidecar"
    diff = regime_diff(payload.get("regime") or {}, regime)
    if diff:
        return False, f"regime changed: {diff}"
    diff = regime_diff(payload.get("inputs") or {}, inputs or {})
    if diff:
        return False, f"inputs changed: {diff}"
    return True, "output complete under matching regime + inputs"


def resume_disposition(
    out_path: Path,
    regime: dict,
    inputs: dict | None = None,
    *,
    overwrite: bool = False,
) -> tuple[str, str]:
    """(disposition, reason) for one unit's output under the given regime.

    - ``overwrite=True``   -> always ("run", ...); the sidecar is cleared so a
      crash mid-regeneration can never leave a stale done marker.
    - missing output/sidecar -> ("run", ...).
    - sidecar regime != expected -> raises RegimeMismatch naming the keys
      (refusal — never silently mix regimes at one path).
    - sidecar inputs != expected -> ("recompute", ...) — upstream refresh.
    - full match -> ("skip", ...).
    """
    if overwrite:
        clear_done(out_path)
        return RUN, "overwrite requested"
    sp = sidecar_path(out_path)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return RUN, "output missing or empty"
    if not sp.is_file():
        return RUN, "no done sidecar"
    try:
        recorded = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RUN, "unreadable done sidecar"
    rec_regime = recorded.get("regime") or {}
    diff = regime_diff(rec_regime, regime)
    if diff:
        raise RegimeMismatch(
            f"{out_path} holds a run under a DIFFERENT regime (differing keys: {diff}; "
            f"recorded={ {k: rec_regime.get(k) for k in diff} } "
            f"expected={ {k: regime.get(k) for k in diff} }) — pass --overwrite to "
            "regenerate, or use a fresh --output-dir (never silently mix regimes)"
        )
    rec_inputs = recorded.get("inputs") or {}
    exp_inputs = inputs or {}
    changed = regime_diff(rec_inputs, exp_inputs)
    if changed:
        return RECOMPUTE, f"inputs changed: {changed}"
    return SKIP, "output complete under matching regime + inputs"
