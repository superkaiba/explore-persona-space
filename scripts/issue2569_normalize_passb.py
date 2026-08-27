"""Complete the #2379 pass-B ridge payloads' metadata stamp for the #2569 atlas.

WHY THIS SCRIPT EXISTS (and why it is the persisted artifact, not the .pt copies
it produces). The #2569 P-E atlas roster includes three pass-B rows staged from
``issue2379_reelicit/analysis_tensors/maps_pinned/base_L{14,16,27}.pt``. Those
payloads carry every NUMERIC component the consumer needs — ``W`` (3584x3584),
``xmu``, ``xsd``, ``ymu``, ``layer`` — but ``scripts/issue2379_mapfit.py`` writes
them WITHOUT three metadata keys that ``issue2569_atlas.payload_from_dict`` +
``issue2569_operator.MapPayload`` require:

    kind == "ridge"     fitter == "ridge"     selected_lambda   (#2379 names it `lam`)

So the atlas dropped all three PLANNED rows twice: first on
``expected ridge payload, got kind=None``, then — after the tags alone were
stamped — on ``KeyError 'selected_lambda'``. Enumerating the consumer's COMPLETE
required schema (rather than chasing one KeyError per attempt) showed the entire
gap is those three keys: there is no structural or numeric incompatibility, only
a naming divergence between the producer and this consumer.

WHY NOT RELAX THE CONSUMER GATE. ``payload_from_dict`` guards every caller
against loading a NON-ridge map as ridge. Broadening it to tolerate a missing
``kind`` would weaken a fail-loud validation fleet-wide to accommodate one
sibling's omission. This script instead completes the stamp on LOCALLY STAGED
COPIES, leaving both the gate and the numerics untouched.

WHY WE MAY ASSERT ``kind="ridge"`` — four converging artifact-INTERNAL proofs,
each RE-VERIFIED at execution time rather than trusted from a note:
  1. ``prediction_formula`` == "v_hat = ((v_c - xmu)/xsd) @ W + ymu" — exactly the
     standardize -> matmul -> add-ymu form the consumer implements, which also
     pins ``W`` as (d_in, d_out) (the matrices are square, so shape alone cannot).
  2. ``lam`` is present (316.2277660168379 = 10^2.5 for L14/L16) — a ridge penalty.
  3. ``git.git_argv0_path`` == "scripts/issue2379_mapfit.py" on a clean tree.
  4. The atlas's own hardcoded ``floor_label`` for these rows reads
     "no floor — banked (n_train=4,500 pass-B fit)" and the payloads carry
     ``n_train`` == 4500: the consumer was written AGAINST these artifacts.

THIS SCRIPT IS THE DURABLE RECORD. The normalized ``.pt`` copies are deliberately
NOT uploaded: they are 3 x ~51.5 MB near-duplicates whose numerics are
bit-identical to the #2379 originals already durable on the Hub. Original +
this script regenerates them exactly, so the recipe is the load-bearing minimum
(the same reasoning the upload policy applies to text-vs-tensor persistence).

CAVEAT THIS SCRIPT SURFACES, NOT HIDES: ``base_L27`` selects lam == 1000.0
exactly, against 316.2278 for L14/L16. A round power of ten is the signature of a
lambda-GRID EDGE — that fit may sit at the regularization ceiling rather than an
interior optimum — so the run prints a LAMBDA-GRID-EDGE line for it, and the
#2569 write-up carries it as a per-row caveat on ``passb_L27``.

PRODUCER-SIDE FIX belongs upstream: ``issue2379_mapfit.py`` should stamp
``kind``/``fitter``/``selected_lambda`` so every downstream consumer of
``maps_pinned`` loads with no normalization step at all. Filed as a #2569
code-review concern.

Usage (idempotent; safe to re-run):
    uv run python scripts/issue2569_normalize_passb.py \
        --stage-dir /workspace/eps2569pe/fits/stage_passb
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE the numpy/torch imports (#847; code-style.md)

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYERS = (14, 16, 27)
EXPECTED_FORMULA = "v_hat = ((v_c - xmu)/xsd) @ W + ymu"
NUMERIC = ("W", "xmu", "xsd", "ymu")
STAMP_NOTE = (
    "issue2569 P-E: kind/fitter/selected_lambda completed from artifact-internal "
    "evidence (prediction_formula + lam + issue2379_mapfit.py provenance); "
    "numerics untouched"
)


def _sha(a: np.ndarray) -> str:
    """sha256 over the raw bytes of ``a`` — the bit-identity witness."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def normalize_one(path: Path) -> tuple[bool, str]:
    """Stamp the three missing keys onto ``path`` in place.

    Returns ``(changed, message)``. Raises nothing on a validation miss: returns
    ``changed=False`` with a FAIL message so the caller can aggregate an rc.
    Numerics are asserted bit-identical across the re-serialize; a mismatch
    refuses the swap rather than writing a silently-altered payload.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)

    if d.get("prediction_formula") != EXPECTED_FORMULA:
        return False, f"FAIL {path.name}: prediction_formula mismatch"
    if d.get("lam") is None:
        return False, f"FAIL {path.name}: no `lam` — cannot source selected_lambda"

    before = {k: _sha(np.asarray(d[k])) for k in NUMERIC}
    W = np.asarray(d["W"])
    d_in, d_out = W.shape
    if not (
        np.asarray(d["xmu"]).shape == (d_in,)
        and np.asarray(d["xsd"]).shape == (d_in,)
        and np.asarray(d["ymu"]).shape == (d_out,)
        and all(np.isfinite(np.asarray(d[k])).all() for k in NUMERIC)
        and (np.asarray(d["xsd"]) > 0).all()
    ):
        return False, f"FAIL {path.name}: numeric contract does not hold — not stamping"

    changed = []
    if d.get("kind") is None:
        d["kind"] = "ridge"
        changed.append("kind")
    if d.get("fitter") is None:
        d["fitter"] = "ridge"
        changed.append("fitter")
    if d.get("selected_lambda") is None:
        d["selected_lambda"] = float(d["lam"])
        changed.append("selected_lambda<-lam")
    if not changed:
        return False, f"SKIP {path.name}: already complete"

    d["normalized_by"] = STAMP_NOTE
    tmp = path.with_suffix(".pt.tmp")
    torch.save(d, tmp)
    d2 = torch.load(tmp, map_location="cpu", weights_only=False)
    if {k: _sha(np.asarray(d2[k])) for k in NUMERIC} != before:
        tmp.unlink(missing_ok=True)
        return False, f"FAIL {path.name}: numerics CHANGED on re-serialize — refused"
    if d2.get("kind") != "ridge" or d2.get("fitter") != "ridge":
        tmp.unlink(missing_ok=True)
        return False, f"FAIL {path.name}: tags did not persist"
    float(d2["selected_lambda"])  # raises if unusable
    tmp.replace(path)

    edge = " <-- LAMBDA-GRID-EDGE SUSPECT (round 1e3)" if float(d["lam"]) == 1000.0 else ""
    return True, (
        f"OK {path.name}: +{','.join(changed)}; numerics bit-identical "
        f"(W {before['W'][:12]}); selected_lambda={float(d['selected_lambda']):.4f} "
        f"kstar={d.get('kstar')}{edge}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: complete the pass-B metadata stamp for every staged layer."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("/workspace/eps2569pe/fits/stage_passb"),
        help="dir holding the staged base_L{layer}.pt pass-B payloads",
    )
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check, exit 0")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return 0

    rc = 0
    for layer in LAYERS:
        p = args.stage_dir / f"base_L{layer}.pt"
        if not p.exists():
            print(f"FAIL L{layer}: {p} absent")
            rc = 1
            continue
        ok, msg = normalize_one(p)
        print(msg)
        if msg.startswith("FAIL"):
            rc = 1
    print(f"PASSB_NORMALIZE_RC={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
