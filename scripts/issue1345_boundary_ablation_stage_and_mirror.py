#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation — off-VM fits support: stage + mirror.

The fits phase exceeds the shared VM's memory watchdog envelope (earlyoom
`-m 10 --prefer python` SIGTERMed three runs at ~55-60 GiB RSS), so it routes
to a dedicated `cpu-bigmem` GCE instance per the >50 GB off-VM rule. That
instance clones the repo (eval_results anchors included) but has neither the
captured round stores nor a way to persist the fits outputs past its own
teardown. Two subcommands close exactly those gaps:

  stage   — download the round's captured stores from the HF data repo
            (`issue1345_framing/story_boundary_ablation/analysis_tensors/`)
            into the local variant turnstore dir (flat, the layout the capture
            wrote), then FAIL-LOUD consumer probe: every store family the fits
            enumerate must be present and one manifest must parse.
  mirror  — upload the fits outputs (`eval_results/issue_1345/
            story_boundary_ablation/**`) to the HF data repo under
            `issue1345_framing/story_boundary_ablation/eval_mirror/` in ONE
            `upload_folder` commit, then verify the uploaded set server-side.

Both ride `hub.retry_transient`; zero Anthropic API; CPU-only.

Usage (the cpu-bigmem workload chain):
  uv run python scripts/issue1345_boundary_ablation_stage_and_mirror.py stage
  ... fits --phase all --stage-v1 ...
  uv run python scripts/issue1345_boundary_ablation_stage_and_mirror.py mirror
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    scripts_dir = str(here.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


_ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

EVAL_OUT_DIR = Path("eval_results/issue_1345/story_boundary_ablation")
HF_EVAL_MIRROR_PREFIX = f"{c.HF_ISSUE_PREFIX}/eval_mirror"
# Store stem families the fits enumerate — pinned against the live HF listing.
# A family listed here MUST be present after staging or the stage FAILS LOUD.
#
# Two tiers, because absence means different things:
#   REQUIRED_STORE_TOKENS   the injected lattice — a miss is a broken stage.
#   PAIRED_STORE_TOKENS     the on-policy PAIRED arm. Absence is NOT an error
#                           (the fits presence-gate simply skips the arm), but a
#                           SILENT absence is the one-arm-instead-of-two failure
#                           the both-arms discipline exists to prevent — so the
#                           stage REPORTS each one's presence explicitly and the
#                           caller can require them once they exist.
#
# `bnd_v5` joined the injected tier when the V5 bare-label arm landed. The
# on-policy tier is token-only (no model prefix): the same family is captured
# under BOTH measured models — instruct_bnd_chat_op_s AND pretrained_… — since 3
# of the 4 on-policy answer bundles are base-written.
REQUIRED_STORE_TOKENS = (
    "bnd_v1",
    "bnd_v2",
    "bnd_v3",
    "bnd_v4",
    "bnd_v5",
    "bnd_chat",
    "bnd_ntpl",
)
PAIRED_STORE_TOKENS = (
    "bnd_v1_op",
    "bnd_chat_op",
    "bnd_ntpl_op",
)
# Set EPM_I1345_REQUIRE_PAIRED=1 once the on-policy captures have landed to
# promote the paired tier to a hard requirement (turning a silent one-arm read
# into a loud stage failure).
REQUIRE_PAIRED_ENV = "EPM_I1345_REQUIRE_PAIRED"


def cmd_stage() -> int:
    """Stage the round's captured stores from HF into the local turnstore dir."""
    api = HfApi()
    dest = Path(c.TURNSTORE_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    names = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            api, c.HF_DATA_REPO, c.HF_TENSOR_PREFIX, repo_type="dataset"
        ),
        what=f"list({c.HF_TENSOR_PREFIX})",
    )
    assert names, f"no files under {c.HF_TENSOR_PREFIX} — capture uploads missing"
    staged = skipped = 0
    scratch = dest / ".stage_scratch"
    for remote in sorted(names):
        fname = remote.rsplit("/", 1)[-1]
        out = dest / fname
        if out.exists() and out.stat().st_size > 0:
            skipped += 1
            continue
        src = c.stage_pinned_file(remote, scratch, revision="main")
        Path(src).replace(out)
        staged += 1
    print(f"[stage] staged={staged} skipped={skipped} -> {dest}", flush=True)
    # Consumer probe (staged-layout rule): every required store family must be
    # present by token, and at least one store manifest must parse as JSON.
    present = [p.name for p in dest.iterdir() if p.is_file()]

    # The `_op` families are SUFFIXES of nothing else, but `bnd_v1` IS a prefix of
    # `bnd_v1_op` — so an injected-tier probe must not be satisfied by an
    # on-policy file alone. Match the family with its stem terminator.
    def _family_present(token: str) -> bool:
        return any(f"{token}_s" in n for n in present)

    missing = [t for t in REQUIRED_STORE_TOKENS if not _family_present(t)]
    assert not missing, f"staged layout missing store families: {missing}"
    # Paired (on-policy) arm: report every family, and require them only when the
    # caller says the captures have landed.
    paired = {t: _family_present(t) for t in PAIRED_STORE_TOKENS}
    landed = sorted(t for t, ok in paired.items() if ok)
    absent = sorted(t for t, ok in paired.items() if not ok)
    print(
        f"[stage] paired on-policy families present={landed or 'none'} absent={absent or 'none'}",
        flush=True,
    )
    if os.environ.get(REQUIRE_PAIRED_ENV) == "1":
        assert not absent, (
            f"{REQUIRE_PAIRED_ENV}=1 but the paired on-policy families {absent} are absent "
            "— the fits would presence-gate them out and silently report ONE arm instead "
            "of two (the both-arms discipline). Capture them, or unset the env var."
        )
    manifests = [p for p in dest.iterdir() if "manifest" in p.name and p.suffix == ".json"]
    assert manifests, "no store manifest staged — layout mismatch vs capture output"
    json.loads(manifests[0].read_text())
    print(f"[stage] consumer probe PASS ({len(present)} files)", flush=True)
    return 0


def cmd_mirror() -> int:
    """Mirror the fits outputs to HF so they survive instance teardown."""
    assert EVAL_OUT_DIR.exists(), f"{EVAL_OUT_DIR} absent — fits produced nothing?"
    n_local = sum(1 for p in EVAL_OUT_DIR.rglob("*") if p.is_file())
    assert n_local > 0, "no fits outputs to mirror"
    api = HfApi()
    # Deterministic pre-upload guard (outside the transient-retry wrapper): the
    # Hub rejects >10k files per repo dir at COMMIT time, non-retriably (#658).
    hub.assert_hub_dir_filecounts(folder_path=str(EVAL_OUT_DIR), path_in_repo=HF_EVAL_MIRROR_PREFIX)
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(EVAL_OUT_DIR),
            repo_id=c.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=HF_EVAL_MIRROR_PREFIX,
        ),
        what=f"upload_folder({HF_EVAL_MIRROR_PREFIX})",
    )
    listed = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            api, c.HF_DATA_REPO, HF_EVAL_MIRROR_PREFIX, repo_type="dataset"
        ),
        what=f"verify({HF_EVAL_MIRROR_PREFIX})",
    )
    n_remote = len(listed)
    print(f"[mirror] local={n_local} remote={n_remote}", flush=True)
    assert n_remote >= n_local, f"mirror verify FAILED: {n_remote} remote < {n_local} local"
    print("[mirror] verify PASS", flush=True)
    return 0


def main() -> int:
    """CLI: `stage` before the fits, `mirror` after."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["stage", "mirror"])
    args = ap.parse_args()
    return cmd_stage() if args.cmd == "stage" else cmd_mirror()


if __name__ == "__main__":
    sys.exit(main())
