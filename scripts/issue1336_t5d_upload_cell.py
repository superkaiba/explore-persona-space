"""Per-cell durable upload for the #1336 t5d GPU round (leg-A ladder cells).

Uploads ONE (pair, surface) cell's artifacts the moment the battery completes
(#664 per-cell contract), then deletes the LOCAL preds npz so staging + preds
never co-resident past the MooseFS ~130 GB /workspace quota:

  - ``ladpreds_<unit>.npz``  -> ``issue1336_rlvr_ladder/analysis_tensors/
    metric_ladder_preds_t5d/`` — a VERSIONED prefix: the round-3 bank at
    ``metric_ladder_preds/`` keeps serving the bytes existing captures were
    made under (#922 regeneration rule; this round is layer-30-only with the
    orth tiers, a different recipe).
  - ``pair_<unit>.json``     -> same prefix (text path; also lands in git at
    harvest — the upload is the pod-death-safe copy).
  - ``metric_ladder_manifest.json`` (cumulative, tiny) -> same prefix.

Each upload rides ``hub.retry_transient``; presence is verified with
``file_exists`` probes (2-3 files — the sanctioned single-path probe form)
BEFORE the local npz is deleted. Fail-loud: any miss raises and the driver
records the pair's rc.
"""

from __future__ import annotations

import argparse
import random
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

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

HF_PREFIX = f"{cm.HF_PREFIX_1336}/analysis_tensors/metric_ladder_preds_t5d"


def _is_commit_conflict(err: BaseException) -> bool:
    """True for the HF 409 'another commit operation is in progress' rejection.

    HF serializes commits per repo, so a sibling pod committing to the shared
    data repo surfaces here as HfHubHTTPError 409 — EXPECTED contention, not a
    unit failure (2026-08-17: pod-1336-insertarm's commits 409'd this round's
    base__sft_chat_math7500 preds upload and rc=1'd the whole invocation).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="leg-A out root (pair JSONs)")
    ap.add_argument("--unit", required=True, help="<m0>__<m1>_<fmt>_<corpus>")
    ap.add_argument(
        "--keep-local-npz",
        action="store_true",
        help="skip the delete-after-verify (debug only; default deletes)",
    )
    args = ap.parse_args()

    from huggingface_hub import HfApi, upload_file

    api = HfApi()
    npz = args.preds_dir / f"ladpreds_{args.unit}.npz"
    pair_json = args.out_dir / "metric_ladder" / f"pair_{args.unit}.json"
    manifest = args.preds_dir / "metric_ladder_manifest.json"
    assert npz.exists(), npz
    assert pair_json.exists(), pair_json

    todo = [(npz, npz.name), (pair_json, pair_json.name)]
    if manifest.exists():
        todo.append((manifest, manifest.name))
    for local, name in todo:
        dest = f"{HF_PREFIX}/{name}"
        retry_hub_409(
            lambda local=local, dest=dest, name=name: hub.retry_transient(
                lambda: upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=dest,
                    repo_id=cm.HF_DATA_REPO,
                    repo_type="dataset",
                    commit_message=f"issue-1336 t5d round: {name}",
                ),
                what=f"t5d cell upload {name}",
            ),
            what=f"t5d cell upload {name}",
        )
        ok = hub.retry_transient(
            # HUB_VERIFY_RETRY_EXEMPT: single-path probe wrapped in hub.retry_transient right here
            lambda dest=dest: api.file_exists(cm.HF_DATA_REPO, dest, repo_type="dataset"),
            what=f"t5d cell verify {name}",
        )
        assert ok, f"uploaded but not visible: {dest}"
        print(f"[t5d-upload] {name} -> {dest} verified", flush=True)

    if not args.keep_local_npz:
        npz.unlink()
        print(f"[t5d-upload] reaped local {npz}", flush=True)


if __name__ == "__main__":
    main()
