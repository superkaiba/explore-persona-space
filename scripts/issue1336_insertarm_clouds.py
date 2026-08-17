#!/usr/bin/env python3
"""Issue #1336 inserted-arm round — layer-30 cloud harvest from the BANKED
Phase EXT_off off-policy turnstores (no new GPU capture; absence sweep found
all 20 (encoder i, text-source j) pair trees complete on the Hub).

One unit = one (text-source s -> encoder t, corpus) cell of a FORWARD stage
pair: download the banked off-policy turnstore stem
(``turnstore_offpolicy_<t>_chat_<s>/<t>_chat_<corpus>_shard*``) via the rig's
own fail-loud completeness-checked fetch (``issue1336_extract_turnstore.
_try_hf_resume``), slice layer ``--layer`` (default 30) into the paired cloud
vectors — context vector = assistant-header slot (slot index 1: the token
before the answer starts) and answer vector = answer-span token mean (turn
index 1) — and upload one ``clouds_<t>_txt_<s>_chat_<corpus>.npz`` (+ meta)
to ``analysis_tensors/layer30_clouds/inserted/``. Schema mirrors the t5d
round's diagonal cloud export (``issue1336_t5d_export_stage_assets.py``) plus
``text_source`` / ``encoder`` provenance fields.

Per-cell durability (#664): npz uploads + Hub-verifies the moment its cell
completes; the staged shards are reaped per cell (peak disk ~= one cell,
~1.4-15 GB, far under the ~130 GB MooseFS /workspace quota). Resume: a cell
whose expected Hub paths already exist is SKIPPED (scoped listing via
``hub.verify_repo_paths_uploaded``).

Smoke (``--smoke``): builds a tiny synthetic shard pair in the t5d turnstore
schema under a temp stage root and runs the identical slice->npz->meta path
with uploads OFF — the Hub staging leg is exercised by the pod-side pilot
cell instead (smoke blind-spot enumeration in the round's dispatch note).

Every ``__main__`` invocation exits explicitly (PyGILState_Release atexit
race — gotchas.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_fit_cells as fc  # noqa: E402
import issue1336_extract_turnstore as et  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

INSERTED_PREFIX = f"{cm.HF_PREFIX_1336}/analysis_tensors/layer30_clouds/inserted"

# Forward pairs: stage(s) < stage(t) over the 5-checkpoint ladder.
FORWARD_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (s, t) for s in cm.MODELS for t in cm.MODELS if cm.MODELS[s]["stage"] < cm.MODELS[t]["stage"]
)
CHAT_CORPORA: tuple[str, ...] = tuple(cm.V2_CORPORA)  # 7 chat surfaces (fmt = chat)


def inserted_cloud_name(encoder: str, text_source: str, corpus: str) -> str:
    """Basename stem for one inserted-cloud cell (v3_pair_id naming)."""
    return f"clouds_{cm.v3_pair_id(encoder, text_source)}_chat_{corpus}"


# Vendored VERBATIM from scripts/issue1336_t5d_upload_cell.py @ 9010d8de24
# (issue-1336-backward-pairs; a cherry-pick conflicts on unrelated files).
# 2026-08-17: three-way commit contention on the shared data repo (this pod's
# 4 harvest workers + pod-1336-t5d's uploads) 409-killed workers w1/w2.
def _is_commit_conflict(err: BaseException) -> bool:
    """True for the HF 409 'another commit operation is in progress' rejection.

    HF serializes commits per repo, so a sibling pod committing to the shared
    data repo surfaces here as HfHubHTTPError 409 — EXPECTED contention, not a
    unit failure.
    """
    from huggingface_hub.errors import HfHubHTTPError

    resp = getattr(err, "response", None)
    return isinstance(err, HfHubHTTPError) and getattr(resp, "status_code", None) == 409


def retry_hub_409(fn, what: str, attempts: int = 5):
    """Run ``fn`` (itself already ``hub.retry_transient``-wrapped) with a bounded
    outer retry for 409 commit conflicts: up to ``attempts`` tries, 60-120 s
    uniform-jitter backoff between them. ``retry_transient`` does not classify
    409 transient (correct for most callers — a 409 can mean a real conflict),
    so per-repo commit contention gets this bounded outer envelope instead.
    Non-409 errors propagate immediately; the last 409 re-raises.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — filtered to 409 right below
            if not _is_commit_conflict(e) or attempt == attempts:
                raise
            delay = random.uniform(60.0, 120.0)
            print(
                f"[hub-409] {what}: commit conflict (attempt {attempt}/{attempts}); "
                f"retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unavailable"
        )
    except OSError:
        return "unavailable"


def slice_cloud(bundle: dict, layer: int) -> dict:
    """Layer-`layer` context/answer cloud arrays from one loaded bundle.

    Fail-loud on NaN rows and shape drift (the #825 bundle contract:
    slots (n, 2, L, d) with slot 1 = assistant header; profiles (n, 2, L, d)
    with turn 1 = the answer span).
    """
    arrays, sidecar = bundle["arrays"], bundle["sidecar"]
    slots = np.asarray(arrays["slots"])
    profiles = np.asarray(arrays["profiles"])
    assert slots.ndim == 4 and slots.shape[1] == 2, f"slots shape {slots.shape}"
    assert profiles.ndim == 4 and profiles.shape[1] == 2, f"profiles shape {profiles.shape}"
    n_layers = slots.shape[2]
    assert 0 <= layer < n_layers, f"layer {layer} out of range ({n_layers} layers)"
    X = slots[:, 1, layer, :].astype(np.float16)
    Y = profiles[:, 1, layer, :].astype(np.float16)
    ids = np.asarray([str(c) for c in sidecar["conv_ids"]])
    assert len(ids) == X.shape[0] == Y.shape[0], (len(ids), X.shape, Y.shape)
    bad = ~np.isfinite(X.astype(np.float32)).all(1) | ~np.isfinite(Y.astype(np.float32)).all(1)
    assert not bad.any(), f"{int(bad.sum())} non-finite rows at layer {layer}"
    return {"X30": X, "Y30": Y, "conv_ids": ids}


def harvest_cell(
    encoder: str,
    text_source: str,
    corpus: str,
    *,
    stage_root: Path,
    local_out: Path,
    layer: int,
    upload: bool,
    api,
) -> str:
    """One cell end-to-end: skip-if-uploaded -> stage -> slice -> npz -> upload -> reap."""
    name = inserted_cloud_name(encoder, text_source, corpus)
    expected = [f"{INSERTED_PREFIX}/{name}.npz", f"{INSERTED_PREFIX}/{name}.meta.json"]
    if upload:
        # verify_repo_paths_uploaded returns the MISSING paths (empty = all present)
        missing = hub.verify_repo_paths_uploaded(
            api, cm.HF_DATA_REPO, expected, path_in_repo=INSERTED_PREFIX, repo_type="dataset"
        )
        if not missing:
            return "skipped-uploaded"
    offpol_dir = cm.offpolicy_ts_dirname(encoder, text_source)
    stem = cm.cell_id(encoder, cm.V3_TEXT_FORMAT, corpus)
    cell_stage = stage_root / offpol_dir
    done = et._try_hf_resume(cell_stage, stem, v2=False, offpol_dir=offpol_dir)
    assert done is not None, (
        f"banked off-policy cell ABSENT on the Hub: {offpol_dir}/{stem} — the absence sweep "
        "found all 20 pair trees; a missing stem here is a real gap, triage before proceeding"
    )
    bundle = fc._load_bundle_any(cell_stage, encoder, cm.V3_TEXT_FORMAT, corpus)
    cloud = slice_cloud(bundle, layer)
    n, d = cloud["X30"].shape
    folds = fc._cv_folds(cloud["conv_ids"], cm.N_FOLDS, cm.FIT_SEED)
    local_out.mkdir(parents=True, exist_ok=True)
    npz_path = local_out / f"{name}.npz"
    np.savez(  # plain savez: compression OFF for Xet (#813)
        npz_path,
        X30=cloud["X30"],
        Y30=cloud["Y30"],
        conv_ids=cloud["conv_ids"],
        folds=folds.astype(np.int64),
    )
    (local_out / f"{name}.meta.json").write_text(
        json.dumps(
            {
                "arm": "inserted (teacher-forced matched text: encoder E_t on source text T_s)",
                "encoder": encoder,
                "text_source": text_source,
                "corpus": corpus,
                "format": cm.V3_TEXT_FORMAT,
                "layer": layer,
                "n_rows": int(n),
                "d": int(d),
                "n_folds": int(cm.N_FOLDS),
                "fold_convention": "fc._cv_folds over the cell's OWN rows (seed "
                f"{cm.FIT_SEED}) — NOT any pair's intersection-row split",
                "extraction": "banked Phase EXT_off off-policy turnstore "
                f"({offpol_dir}/{stem}); v_context = assistant-header slot (slot 1); "
                "v_answer = answer-span token mean (turn 1)",
                "dtype": "fp16",
                "sha256_npz": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                "code_sha": _git_sha(),
            },
            indent=2,
        )
    )
    if upload:
        from huggingface_hub import upload_folder

        hub.assert_hub_dir_filecounts(local_out, INSERTED_PREFIX, allow_patterns=[f"{name}.*"])
        retry_hub_409(
            lambda: hub.retry_transient(
                lambda: upload_folder(
                    repo_id=cm.HF_DATA_REPO,
                    repo_type="dataset",
                    folder_path=str(local_out),
                    path_in_repo=INSERTED_PREFIX,
                    allow_patterns=[f"{name}.*"],
                    commit_message=f"issue-1336 inserted-arm clouds: {name}",
                ),
                what=f"inserted-cloud upload {name}",
            ),
            what=f"inserted-cloud upload {name}",
        )
        still_missing = hub.verify_repo_paths_uploaded(
            api, cm.HF_DATA_REPO, expected, path_in_repo=INSERTED_PREFIX, repo_type="dataset"
        )
        assert not still_missing, f"post-upload verify FAILED for {name}: missing {still_missing}"
        # reap the staged shards + local npz the moment the upload verifies
        shutil.rmtree(cell_stage, ignore_errors=False)
        npz_path.unlink()
        (local_out / f"{name}.meta.json").unlink()
    return f"harvested n={n}"


def _smoke(tmp_root: Path, layer: int) -> None:
    """Tiny synthetic shard in the production schema through the REAL slice path."""
    import torch

    enc, src, corpus = "dpo", "base", "gsm8k_test1319"
    offpol_dir = cm.offpolicy_ts_dirname(enc, src)
    stem = cm.cell_id(enc, cm.V3_TEXT_FORMAT, corpus)
    cell_stage = tmp_root / offpol_dir
    cell_stage.mkdir(parents=True, exist_ok=True)
    n, n_layers, d = 12, max(2, layer + 1), 8
    rng = np.random.default_rng(0)
    payload = {
        "conv_ids": [f"s{i}" for i in range(n)],
        "slots": [
            torch.as_tensor(rng.normal(size=(2, n_layers, d)), dtype=torch.bfloat16)
            for _ in range(n)
        ],
        "profiles": [
            torch.as_tensor(rng.normal(size=(2, n_layers, d)), dtype=torch.bfloat16)
            for _ in range(n)
        ],
        "nll": [torch.zeros(2) for _ in range(n)],
        "spans_meta": [{} for _ in range(n)],
    }
    torch.save(payload, cell_stage / f"{stem}_shard000.pt")
    (cell_stage / f"{stem}_shard000.json").write_text(
        json.dumps({"conv_ids": payload["conv_ids"], "shard_index": 0})
    )
    bundle = fc._load_bundle_any(cell_stage, enc, cm.V3_TEXT_FORMAT, corpus)
    cloud = slice_cloud(bundle, layer)
    folds = fc._cv_folds(cloud["conv_ids"], cm.N_FOLDS, cm.FIT_SEED)
    assert cloud["X30"].shape == (n, d) and cloud["Y30"].shape == (n, d)
    assert folds.shape == (n,)
    print(f"[smoke] slice OK n={n} d={d} layer={layer} folds={sorted(set(folds.tolist()))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=None, help="comma list s:t (default: 10 forward pairs)")
    ap.add_argument("--corpora", default=None, help="comma list (default: 7 chat corpora)")
    ap.add_argument("--stage-root", type=Path, default=Path("data/issue_1336/insertarm_stage"))
    ap.add_argument("--local-out", type=Path, default=Path("data/issue_1336/insertarm_clouds"))
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="insertarm_smoke_") as td:
            _smoke(Path(td), layer=1)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    pairs = (
        [tuple(p.split(":")) for p in args.pairs.split(",")] if args.pairs else list(FORWARD_PAIRS)
    )
    corpora = tuple(args.corpora.split(",")) if args.corpora else CHAT_CORPORA
    for s, t in pairs:
        assert s in cm.MODELS and t in cm.MODELS and s != t, f"bad pair {s}:{t}"
    units = [(s, t, c) for (s, t) in pairs for c in corpora]
    print(f"[harvest] {len(units)} units ({len(pairs)} pairs x {len(corpora)} corpora)")
    t_start = time.time()
    for k, (s, t, c) in enumerate(units):
        u0 = time.time()
        status = harvest_cell(
            t,
            s,
            c,
            stage_root=args.stage_root,
            local_out=args.local_out,
            layer=args.layer,
            upload=not args.skip_upload,
            api=api,
        )
        print(
            f"[harvest] unit {k + 1}/{len(units)} {t}_txt_{s}_chat_{c} {status} "
            f"elapsed={time.time() - u0:.0f}s total={time.time() - t_start:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
