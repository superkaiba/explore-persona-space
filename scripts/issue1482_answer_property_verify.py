"""Independent artifact-level prediction checks and bounded retrieval diagnostic."""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
import json
from pathlib import Path
import numpy as np
from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from explore_persona_space.atomic_io import write_json_atomic

from issue1482_answer_property_ceiling import digest

run = Path("/home/thomasjiralerspong/.codex/research/answer-property-ceiling-20260907")
out = run / "observed_answer"
complete = json.loads((out / "complete.json").read_text())
reg = dict(np.load(run / "prepared/registry.npz"))
fac = dict(np.load(out / "gram.npz"))
lam = complete["lambda"]
x = np.memmap(
    "/mnt/eps-data/thomasjiralerspong/issue1482_saedense/dense/Y_L19.f32.mm",
    dtype=np.float32,
    mode="r",
    shape=(142000, 3584),
)
rng = np.random.default_rng(1482)
sample = np.sort(rng.choice(20000, 1000, replace=False))
probe = sample[:8]
rot = ((x[reg["test"][probe]].astype(np.float64) - fac["xmu"]) / fac["xsd"]) @ fac["U"]
pred = np.empty((1000, 131072), np.float32)
worst = 0.0
checked = []
for c0 in range(0, 131072, 4096):
    c1 = min(c0 + 4096, 131072)
    key = f"block_{c0:06d}_{c1:06d}"
    pfile = out / "predictions" / f"{key}.npy"
    marker = json.loads((out / "predictions" / f"{key}.json").read_text())
    if (
        marker["identity"] != complete["identity"]
        or marker["lambda"] != lam
        or marker["c0"] != c0
        or marker["c1"] != c1
    ):
        raise ValueError("prediction registry mismatch")
    if digest(pfile) != marker["prediction_sha256"]:
        raise ValueError("prediction payload changed")
    with np.load(out / "weights" / f"{key}.npz") as z:
        expected = ((rot / (fac["s_eig"] + lam)) @ z["coefficient"] + z["target_mean"]).astype(
            np.float32
        )
    stored = np.load(pfile, mmap_mode="r")
    if stored.shape != (20000, c1 - c0):
        raise ValueError("prediction shape mismatch")
    error = float(np.max(np.abs(expected - stored[probe])))
    worst = max(worst, error)
    np.testing.assert_allclose(expected, stored[probe], rtol=2e-6, atol=1e-6)
    pred[:, c0:c1] = stored[sample]
    checked.append(
        {"block": key, "sha256": marker["prediction_sha256"], "max_abs_recomputed_error": error}
    )
meta = json.loads((run / "prepared/ystore_meta.json").read_text())
nnz = meta["nnz"]
values = np.memmap(run / "prepared/y_val_mean.f16", dtype=np.float16, mode="r", shape=(nnz,))
indices = np.memmap(run / "prepared/y_indices.i32", dtype=np.int32, mode="r", shape=(nnz,))
indptr = np.load(run / "prepared/y_indptr.npy")
# scipy sparse indexing needs a supported arithmetic dtype; select CSR rows
# through contiguous segments without materializing the entire target store.
rows = reg["test"][sample]
true = np.zeros((1000, 131072), np.float32)
for i, r in enumerate(rows):
    a, b = indptr[r : r + 2]
    true[i, indices[a:b]] = values[a:b]
retrieval = {
    metric: knn_retrieval(pred, true, ks=(1, 5, 10), metric=metric)
    for metric in ("euclidean", "cosine")
}
write_json_atomic(
    run / "analysis/regular_verification.json",
    {
        "complete_sha256": digest(out / "complete.json"),
        "prediction_blocks": checked,
        "recomputed_rows_per_block": len(probe),
        "max_abs_recomputed_prediction_error": worst,
        "retrieval": retrieval,
        "retrieval_test_positions": sample.tolist(),
        "seed": 1482,
        "retrieval_scope": "auxiliary raw cosine/Euclidean retrieval among 1000 seeded held-out answer feature vectors; chance top-1=0.001; distinct from manuscript whitened-CSLS metric",
        "identity_plus_bias": "inapplicable: 3584 dense coordinates versus 131072 SAE feature coordinates",
    },
)
print("Verified all 32 blocks and retrieval diagnostics", flush=True)
