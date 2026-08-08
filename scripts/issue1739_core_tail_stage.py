"""Tail-resume staging for the #1739 new-arm CORE driver (crash-fix r5).

A CORE box that completed its fits grid but died in the natpv tail
(att-20260802-061638-newarmcorehall: the driver omitted ``--phase rowindex``)
crash-persisted the whole fits output under
``issue1739_partial/<attempt>/eval_results_issue_1739/``. This helper stages
those outputs back into a fresh checkout so the driver
(``scripts/issue1739_newarm_core.sh`` under ``EPM_I1739_CORE_RESUME_PARTIAL_ATT``)
can run ONLY the natpv tail + bank copy + upload:

  fc/<B>/**                     -> eval_results/issue_1739/new_arm_round/fc/<B>/**
  fc/rb_fc_bank/<B>__<r>_fc.npz -> eval_results/.../fc/rb_fc_bank/<same name>
                                   AND analysis_tensors/issue_1739/r_b_<r>_fc/<B>.npz
                                   (reverse of the driver's bank-copy cp — the attempt
                                   bundle carries NO analysis_tensors/ subtree; layout
                                   verified via scoped list_repo_tree, 2026-08-02)

Fail-fast: raises when the attempt prefix is empty, when ``stage_meta.json``
or a required-regime bank npz is missing, or when the git-committed
dv_dataset ``labeling.json`` (the natpv ``load_labels`` input) is absent from
the checkout. Prefix-scoped listing only (gotchas #833; never a full-repo
listing); one pinned revision per invocation; counts-only logging.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force the PLAIN download path for THIS script only (must precede the hub /
# huggingface_hub import — env is frozen at import): this helper stages a
# many-small-npz storm (274 npz of 277 files for the hall bundle; larger for
# syc), byte-for-byte the class that WEDGES xet_get indefinitely
# (att-20260730-055211-syc, py-spy-confirmed) and errors hf_transfer
# (att-20260730-063858-syc), while the plain path handles small files fine —
# same pattern as issue1739_restore_partial.py. This helper has NO big-file
# leg (largest staged file ~5 MB); natpv's accelerator-REQUIRING u_store tar
# staging runs in its OWN uv subprocess and is untouched by these
# process-local disables.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

REPO = "superkaiba1/explore-persona-space-data"
# Bank-copy names the driver writes: <behavior>__<regime>_fc.npz (regime e1|e2p).
# natpv npzs (<behavior>__natpv_*.npz) never match — they are re-derived, not staged back.
_BANK_RE = re.compile(r"^(?P<behavior>[a-z_]+)__(?P<regime>e1|e2p)_fc\.npz$")


def main(argv: list[str] | None = None) -> int:
    """Stage one behavior's crash-persisted fits outputs; return 0 or raise."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    print(
        "[tail-stage] plain HF download path: "
        f"HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']} "
        f"HF_HUB_ENABLE_HF_TRANSFER={os.environ['HF_HUB_ENABLE_HF_TRANSFER']}",
        flush=True,
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attempt", required=True, help="crash-persist attempt id (att-...)")
    ap.add_argument("--behavior", required=True, help="ONE behavior (one CORE box per behavior)")
    ap.add_argument(
        "--regimes", nargs="+", required=True, help="driver $REGIMES — required bank npz regimes"
    )
    ap.add_argument(
        "--dest-root", type=Path, default=Path("."), help="checkout root (smoke: a /tmp scratch)"
    )
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args(argv)
    b = args.behavior

    prefix = f"issue1739_partial/{args.attempt}/eval_results_issue_1739/new_arm_round/fc"
    api = HfApi()
    revision = str(
        hub.retry_transient(
            lambda: api.repo_info(args.repo, repo_type="dataset"),
            what=f"repo_info({args.repo})",
        ).sha
    )
    files = hub.list_hf_files_under_path(
        api, args.repo, prefix, repo_type="dataset", revision=revision
    )
    out_files = [f for f in files if f.startswith(f"{prefix}/{b}/")]
    bank_files = [f for f in files if f.startswith(f"{prefix}/rb_fc_bank/{b}__")]
    print(
        f"[tail-stage] {len(files)} files under {prefix} "
        f"({len(out_files)} under {b}/, {len(bank_files)} rb_fc_bank/{b}__*)",
        flush=True,
    )
    if not out_files:
        raise FileNotFoundError(f"no fits outputs under {prefix}/{b}/ — wrong attempt id?")
    if not any(f.endswith("/stage_meta.json") for f in out_files):
        raise FileNotFoundError(f"stage_meta.json missing under {prefix}/{b}/ — bundle incomplete")

    out_root = args.dest_root / "eval_results/issue_1739/new_arm_round/fc" / b
    bank_dir = args.dest_root / "eval_results/issue_1739/new_arm_round/fc/rb_fc_bank"
    tensors_root = args.dest_root / "analysis_tensors/issue_1739"
    jobs: list[tuple[str, Path]] = [
        (f, out_root / f[len(prefix) + len(b) + 2 :]) for f in out_files
    ]
    jobs += [(f, bank_dir / f.rsplit("/", 1)[1]) for f in bank_files]
    n_new = sum(1 for _f, tgt in jobs if not tgt.exists())
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
        futs = {
            pool.submit(
                hub.stage_hub_file, args.repo, f, tgt, repo_type="dataset", revision=revision
            ): f
            for f, tgt in jobs
        }
        for i, fut in enumerate(as_completed(futs), start=1):
            fut.result()  # re-raises — fail-loud (stage_hub_file is atomic + retried)
            if i % 25 == 0 or i == len(futs):
                print(
                    f"[tail-stage] staged {i}/{len(futs)} "
                    f"(last: {futs[fut].rsplit('/', 1)[1]}) "
                    f"elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )

    # Reconstruct the natpv --e1-fc-bank / driver bank-copy input layout.
    staged_regimes: dict[str, Path] = {}
    for f in bank_files:
        m = _BANK_RE.match(f.rsplit("/", 1)[1])
        if m and m["behavior"] == b:
            staged_regimes[m["regime"]] = bank_dir / f.rsplit("/", 1)[1]
    for regime, src in staged_regimes.items():
        dst = tensors_root / f"r_b_{regime}_fc" / f"{b}.npz"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    missing = [r for r in args.regimes if r not in staged_regimes]
    if missing:
        raise FileNotFoundError(
            f"rb_fc_bank npz missing for regime(s) {missing} under {prefix}/rb_fc_bank/ "
            f"(need {b}__<regime>_fc.npz per driver $REGIMES) — fits leg incomplete"
        )
    dv = args.dest_root / "eval_results/issue_1739/dv_dataset" / b / "labeling.json"
    if not dv.is_file():
        raise FileNotFoundError(
            f"{dv} missing — git-committed natpv load_labels input; run from a full checkout"
        )
    print(
        f"[tail-stage] done: {len(out_files)} fits files ({n_new} newly staged), "
        f"banks={sorted(staged_regimes)}, tensors under {tensors_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
