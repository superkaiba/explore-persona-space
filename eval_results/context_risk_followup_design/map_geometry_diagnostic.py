"""Read-only CPU singular-value diagnostic of the frozen affine context map."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals

SOURCE = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/"
    "qwen38_map_pilot/map_layer_44.npz"
)
OUTPUT = Path(__file__).with_suffix(".json")


def main():
    started = time.monotonic()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == "680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d"
    with np.load(SOURCE) as arrays:
        weight = arrays["weight"].astype(np.float64)
        scale = arrays["x_scale"].astype(np.float64)
    # Row-vector convention: mapped = raw @ effective + offset.
    effective = weight / scale[:, None]
    assert effective.shape == (5120, 5120) and np.isfinite(effective).all()
    singular = svdvals(effective, check_finite=False, overwrite_a=True)
    energy = singular**2
    fractions = np.cumsum(energy) / energy.sum()
    tolerance = 5120 * np.finfo(np.float64).eps * singular[0]
    result = {
        "checked_at": datetime.now(UTC).isoformat(),
        "source": str(SOURCE),
        "source_sha256": digest,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": [5120, 5120],
        "scope": "Effective affine linear component before fold-specific probe StandardScaler.",
        "largest_singular_value": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "condition_number": float(singular[0] / singular[-1]),
        "float64_numerical_rank": int(np.sum(singular > tolerance)),
        "rank_tolerance": float(tolerance),
        "stable_rank": float(energy.sum() / energy[0]),
        "singular_value_quantiles": {
            str(q): float(np.quantile(singular, q)) for q in (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)
        },
        "components_for_frobenius_energy": {
            str(q): int(np.searchsorted(fractions, q) + 1) for q in (0.5, 0.9, 0.95, 0.99)
        },
        "elapsed_seconds": time.monotonic() - started,
        "model_calls": 0,
        "probe_fits": 0,
        "interpretation": (
            "Invertibility concerns unregularized linear function classes; singular spectrum "
            "does not establish risk predictive value. Fold standardization changes geometry."
        ),
    }
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
