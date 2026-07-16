#!/usr/bin/env python3
"""Issue #1092 P6 wrapper: pilot gate + layer-sliced HF staging around issue1092_fit_grid.py.

Plan (tasks/*/1092/plans/plan.md sections 4.5 / 7 / 9):
  1. PILOT FIRST (pre-registered gate): one (cell, layer) unit end-to-end through
     the production fit grid — identity-baseline ladder + one batched
     production-draw permutation-battery unit included — measuring wall-time and
     getrusage(RUSAGE_CHILDREN).ru_maxrss. HARD-ABORT (exit 3) when pilot RSS is
     at or above 14 GB or projected total wall exceeds 2x the plan section-9 P6
     projection; the abort message names the escape lane
     (cpu-bigmem --min-ram-gb 32).
  2. LAYER LOOP: stage layer l's summary shards for ALL cells from HF (scoped
     list_repo_tree + per-file hf_hub_download — never snapshot_download on the
     data repo), invoke scripts/issue1092_fit_grid.py for that layer as a
     subprocess, verify its checkpoints, DELETE the staged layer files, next
     layer. Peak staged disk stays at ~one layer slice.

Resume: a completed layer is skipped only when its staging manifest matches the
current wrapper config (full fit-grid argv surface + fit-grid script sha +
corpus manifest sha) AND the fresh Hub listing matches the recorded per-file
identity AND every recorded checkpoint still exists; any mismatch falls through
to re-stage + the fit grid's own fingerprint predicate. Staged files get a
content-derived mtime so issue1092_fit_grid._fingerprint (name+size+mtime_ns)
reproduces across staging cycles.

The plan-registered fit config is the WRAPPER DEFAULT and is threaded explicitly
into every fit-grid invocation, pilot included: grouped 6-fold CV by prefix
(--n-folds 6, plan section 6), fit seed 0 (--fit-seed -> engine --seed, plan
section 10), targets t1,t2,t3 (plan section 4.5 sensitivity columns). Engine
defaults are unreachable without an explicit per-flag override; --fit-grid-arg
may not clobber --n-folds/--seed (wrapper-owned). The engine-internal 6-value
lambda grid vs the plan's registered 7-value grid is a recorded deviation
(LAMBDA_GRID_DEVIATION, carried in wrapper_config).

Production usage (VM, detached via the canonical setsid/choom recipe; thread
caps supplied at launch):

  uv run python scripts/issue1092_p6_run.py \
      --corpus-dir data/issue_1092/p0/corpus \
      --stage-dir data/issue_1092/p6_stage \
      --out-dir data/issue_1092/p6 \
      --pilot-only

then, after the pilot gate passes, the same command WITHOUT --pilot-only (the
recorded pilot pass is reused; the layer loop runs 0-27), plus
--judge-scores <path> on the definitive judge-bearing run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue1092_fit_grid import (  # noqa: E402
    B0_CELLS,
    CELL_MODEL_TYPE,
    DEFAULT_RB_REV,
    DYNAMICS_KINDS,
    FROZEN_NULL_LAYERS,
    HF_DATA_REPO,
    RIDGE_LAMBDAS,
    _parse_csv,
    _parse_layers,
)

FIT_GRID_SCRIPT = PROJECT_ROOT / "scripts" / "issue1092_fit_grid.py"
DEFAULT_HF_PREFIX = "issue1092_realistic_crossing/analysis_tensors/summaries"
DEFAULT_CELLS = tuple(CELL_MODEL_TYPE)
EXPECTED_MISSING_WITHOUT_JUDGE = "--judge-scores for B1/B2/B3 behavior reads"
# Plan section 10 registers a 7-value log-spaced lambda grid {1e-2..1e3}; the realized
# grid is the parents' 6-value RIDGE_LAMBDAS living inside issue658_fit_predictors /
# issue923_fit_decomposition (press_fit_predict) — engine-internal, not launch-threadable.
# Recorded as a plan deviation per code-review v10 (the plan's own lambda-sensitivity
# columns + df(lambda) reporting make headlines grid-robust; plan section 11 notes the
# grids share endpoints). Rides wrapper_config into every manifest / pilot / summary.
LAMBDA_GRID_DEVIATION = {
    "plan_registered": "7-value log-spaced {1e-2..1e3} (plan section 10, Fits row)",
    "realized": [float(x) for x in RIDGE_LAMBDAS],
    "status": "deviation-recorded",
    "reason": (
        "engine-internal grid (issue658 RIDGE_LAMBDAS via issue923 press_fit_predict); "
        "lambda-sensitivity columns + df(lambda) reporting keep headlines grid-robust"
    ),
}
# The wrapper runs one sequential fit-grid subprocess at a time and adds no
# fan-out of its own (thread caps come from the launch env; plan section 9).
EFFECTIVE_PARALLELISM = 1
ESCAPE_LANE = "cpu-bigmem --min-ram-gb 32"
# Fit-grid flags the wrapper owns; passing them through --fit-grid-arg would
# silently clobber the wrapper's own staging/slicing contract (or, for --n-folds /
# --seed, the plan-registered fit config the wrapper pins by default).
WRAPPER_OWNED_FIT_GRID_FLAGS = frozenset(
    {
        "--summaries-dir",
        "--corpus-dir",
        "--out-dir",
        "--cells",
        "--layers",
        "--arms",
        "--targets",
        "--fit-arms",
        "--target-bases",
        "--n-folds",
        "--seed",
        "--n-null-draws",
        "--band-null-draws",
        "--judge-scores",
        "--rb-dir",
        "--rb-rev",
        "--allow-missing-registered-reads",
    }
)


@dataclass(frozen=True)
class HubFile:
    """One staged-input candidate from the scoped Hub listing."""

    path: str  # repo-relative path (includes the hf prefix)
    size: int
    hub_identity: str  # LFS sha256 (64-hex) when available, else git blob id, else ""


# Transient HTTP statuses worth retrying on the 18-27 h staging loop (408/429/5xx).
_HUB_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# Attempt FLOOR (not a hard cap): the first N calls are always allowed; past the
# floor, retry continues while the cumulative-sleep budget below still holds.
_HUB_RETRY_ATTEMPT_FLOOR = 8
_HUB_RETRY_BACKOFF_BASE_S = 4.0  # full-jitter exponential base
_HUB_RETRY_BACKOFF_CAP_S = 120.0  # per-sleep cap on the backoff branch
_HUB_RETRY_AFTER_CAP_S = 600.0  # defensive cap on a pathological server Retry-After header
_HUB_RETRY_BUDGET_DEFAULT_MIN = 15.0
# Response-less transient markers (mirrors orchestrate.hub: never bare "429" —
# 4xx digit triplets appear in file paths / byte counts, the #989 trap).
_HUB_TRANSIENT_TEXT = (
    "maximum queue size reached",  # the Hub server's queue-full 429 body (#1345 / rf01)
    "too many requests",
    "rate limit",
    "timed out",
    "timeout",
    "connection",
    "temporarily unavailable",
    "gateway time-out",
    "gateway timeout",
    "500",
    "502",
    "503",
    "504",
)


def _hub_retry_budget_s() -> float:
    """Per-call cumulative-sleep budget (s) from P6_HUB_RETRY_MAX_MIN (minutes).

    Default 15 min — sized to outlive the repo-level HF 429 storms that killed
    rf01's staging 4x (sustained for minutes; seconds-scale backoff exhausted).
    Unparseable / non-finite / negative values fall back to the default with a
    loud line (fail-open to the SAFE side: more retry, never less).
    """
    raw = os.environ.get("P6_HUB_RETRY_MAX_MIN", "").strip()
    if not raw:
        return _HUB_RETRY_BUDGET_DEFAULT_MIN * 60.0
    try:
        val = float(raw)
    except ValueError:
        val = float("nan")
    if not math.isfinite(val) or val < 0:
        print(
            f"[hub-retry] P6_HUB_RETRY_MAX_MIN={raw!r} invalid; "
            f"using {_HUB_RETRY_BUDGET_DEFAULT_MIN:.0f} min",
            flush=True,
        )
        return _HUB_RETRY_BUDGET_DEFAULT_MIN * 60.0
    return val * 60.0


def _hub_retry_cause(err: Exception) -> str | None:
    """Classify an exception: short cause string when retriable, None when fatal.

    Retriable: HTTP 408/429/5xx (status read from the response), connection /
    timeout errors, LocalEntryNotFoundError, and response-less HfHubHTTPErrors
    carrying transient text. LocalEntryNotFoundError is checked FIRST (it
    subclasses EntryNotFoundError/HfHubHTTPError with response=None): it is
    raised client-side when hf_hub_download's HEAD metadata call failed — a
    429 storm surfaces 404-shaped through this path (#1345; the rf01
    attempt-4/5 crash class) — never on a genuinely missing file, which the
    Hub reports as a real 404 EntryNotFoundError WITH a response.
    Fatal (fail FAST): 401/403/404 and every other non-transient 4xx
    (EntryNotFoundError / RepositoryNotFoundError / GatedRepoError on real
    missing files / bad auth), and anything unrecognized.
    """
    import requests
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    if isinstance(err, LocalEntryNotFoundError):
        return "local-entry-not-found(head-transport)"
    if isinstance(err, HfHubHTTPError):
        status = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(status, int):
            return str(status) if status in _HUB_TRANSIENT_STATUSES else None
        msg = str(err).lower()
        if any(s in msg for s in _HUB_TRANSIENT_TEXT):
            return "transient-text"
        return None
    if isinstance(err, requests.Timeout):
        return "timeout"
    if isinstance(err, requests.ConnectionError):
        return "connection"
    return None


def _hub_retry_after_s(err: Exception) -> float | None:
    """Seconds from a Retry-After header on the error's response, when present.

    Seconds-form only; an RFC 9110 HTTP-date value parses to None and the
    caller falls back to exponential backoff (mirrors orchestrate.hub).
    """
    headers = getattr(getattr(err, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _hub_retry(fn, *, what: str):
    """Minutes-scale retry for transient Hub errors on a long detached staging loop.

    rf01's GCE relaunches died 4x on repo-level HF 429 storms ("maximum queue
    size reached") that outlast seconds-scale backoff and surface 404-shaped as
    LocalEntryNotFoundError on hf_hub_download's HEAD (epm:failure v4-v7,
    assert_tag [hub-staging-localentrynotfound]). Policy (mirrors the proven
    orchestrate.hub._retry_upload engine, #997/#931, with the wrapper's
    stdout-print convention + class-based LocalEntryNotFoundError handling):

      - retry while EITHER the attempt floor (_HUB_RETRY_ATTEMPT_FLOOR calls)
        OR the per-call cumulative-sleep budget (P6_HUB_RETRY_MAX_MIN minutes,
        default 15) holds; raise only when BOTH are exhausted;
      - every sleep is clamped to the remaining budget, so TOTAL SLEEP <=
        budget (floor attempts past the budget sleep 0 and retry immediately);
      - sleep = Retry-After header when present (capped
        _HUB_RETRY_AFTER_CAP_S), else full-jitter exponential backoff
        uniform(0, min(cap, base * 2**k));
      - genuinely-fatal classes (401/403/404 on a real missing file, anything
        unrecognized) raise IMMEDIATELY — fail-loud stays the contract;
      - EVERY retry prints a loud [hub-retry] line so the poll log shows
        liveness through a minutes-long throttle wait instead of dying silent.
    """
    budget_s = _hub_retry_budget_s()
    attempt = 0
    slept_total = 0.0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as err:
            cause = _hub_retry_cause(err)
            if cause is None:
                raise
            within_attempts = attempt < _HUB_RETRY_ATTEMPT_FLOOR
            within_budget = slept_total < budget_s
            if not (within_attempts or within_budget):
                print(
                    f"[hub-retry] {what}: exhausted after {attempt} attempts "
                    f"(slept {slept_total:.0f}s / budget {budget_s:.0f}s, "
                    f"cause={cause}); raising",
                    flush=True,
                )
                raise
            ra = _hub_retry_after_s(err)
            if ra is not None:
                delay = min(ra, _HUB_RETRY_AFTER_CAP_S)
            else:
                delay = random.uniform(
                    0.0,
                    min(
                        _HUB_RETRY_BACKOFF_CAP_S,
                        _HUB_RETRY_BACKOFF_BASE_S * 2.0 ** min(attempt - 1, 8),
                    ),
                )
            delay = min(delay, max(0.0, budget_s - slept_total))
            print(
                f"[hub-retry] {what}: attempt {attempt}/{_HUB_RETRY_ATTEMPT_FLOOR} failed "
                f"(cause={cause}); sleeping {delay:.1f}s "
                f"(slept {slept_total:.0f}s / budget {budget_s:.0f}s)",
                flush=True,
            )
            time.sleep(delay)
            slept_total += delay


class HfHubIO:
    """Scoped Hub staging IO: list_repo_tree + per-file hf_hub_download.

    Never snapshot_download / bare list_repo_files on the ~1M-file data repo.
    Downloads go through local_dir (direct-to-target) so deleting a staged file
    actually frees the disk (no central-cache copy).
    """

    def __init__(self, repo_id: str, revision: str, *, repo_type: str = "dataset") -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type

    def resolved_revision(self) -> str:
        """Return the commit sha the revision ref resolves to (recorded in manifests)."""
        from huggingface_hub import HfApi

        info = _hub_retry(
            lambda: HfApi().repo_info(
                self.repo_id, repo_type=self.repo_type, revision=self.revision
            ),
            what="repo_info",
        )
        return str(info.sha)

    def list_files(self, prefix: str) -> list[HubFile]:
        """Scoped recursive listing under prefix; raises when the listing is empty."""
        from huggingface_hub import HfApi

        entries = _hub_retry(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: scoped listing already inside _hub_retry (minutes-scale)
                HfApi().list_repo_tree(
                    self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                    path_in_repo=prefix,
                    recursive=True,
                )
            ),
            what="list_repo_tree",
        )
        out: list[HubFile] = []
        for entry in entries:
            size = getattr(entry, "size", None)
            if size is None:
                continue
            lfs = getattr(entry, "lfs", None)
            ident = None
            if lfs is not None:
                ident = getattr(lfs, "sha256", None)
                if ident is None and isinstance(lfs, dict):
                    ident = lfs.get("sha256")
            ident = ident or getattr(entry, "blob_id", None) or ""
            out.append(HubFile(path=str(entry.path), size=int(size), hub_identity=str(ident)))
        if not out:
            raise FileNotFoundError(
                f"no files listed under {self.repo_id}@{self.revision}:{prefix}"
            )
        return out

    def download_to(self, relpath: str, target: Path) -> None:
        """Download one repo file directly to target (no central-cache residue)."""
        from huggingface_hub import hf_hub_download

        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".p6dl-") as td:
            local = _hub_retry(
                lambda: hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                    filename=relpath,
                    local_dir=td,
                ),
                what=f"download {relpath}",
            )
            os.replace(local, target)


class LocalFixtureHubIO:
    """Offline twin of HfHubIO backed by a local tree (tests / offline tiny-real smoke)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def resolved_revision(self) -> str:
        """Return a fixed sentinel revision for fixture-backed runs."""
        return "local-fixture"

    def list_files(self, prefix: str) -> list[HubFile]:
        """List fixture files under root/prefix with sha256 identities."""
        base = self.root / prefix
        files = sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        if not files:
            raise FileNotFoundError(f"no fixture files under {base}")
        return [
            HubFile(
                path=p.relative_to(self.root).as_posix(),
                size=p.stat().st_size,
                hub_identity=_sha256_file(p),
            )
            for p in files
        ]

    def download_to(self, relpath: str, target: Path) -> None:
        """Copy one fixture file to target."""
        src = self.root / relpath
        if not src.is_file():
            raise FileNotFoundError(f"fixture file missing: {src}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_content_mtime(path: Path, sha256_hex: str) -> None:
    """Pin a content-derived mtime on a staged file.

    issue1092_fit_grid._fingerprint hashes (name, size, mtime_ns) per input
    shard; a download-time mtime would change on every re-staging cycle and
    silently invalidate every checkpoint. Deriving mtime_ns from the content
    sha makes the fingerprint content-true (plan section 4.5 "input shard
    SHAs") and reproducible across stagings.
    """
    t_ns = int(sha256_hex[:15], 16)  # 60 bits < 2**63
    os.utime(path, ns=(t_ns, t_ns))


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _rel_under_prefix(path: str, hf_prefix: str) -> str:
    prefix = hf_prefix.rstrip("/") + "/"
    if not path.startswith(prefix):
        raise ValueError(f"hub path {path!r} not under prefix {hf_prefix!r}")
    return path[len(prefix) :]


def build_inventory(hub, hf_prefix: str) -> dict[tuple[str, str], HubFile]:
    """Index one scoped Hub listing as (top_dir, basename) -> HubFile.

    The summaries contract is exactly one directory level under the prefix
    (cell dirs, bare_*, dynamics_*, b0_rB_pool); deeper entries are ignored.
    """
    prefix = hf_prefix.rstrip("/")
    inventory: dict[tuple[str, str], HubFile] = {}
    for f in hub.list_files(prefix):
        rel = _rel_under_prefix(f.path, prefix)
        parts = rel.split("/")
        if len(parts) != 2:
            continue
        inventory[(parts[0], parts[1])] = f
    if not inventory:
        raise FileNotFoundError(f"scoped listing of {prefix} produced no <dir>/<file> entries")
    return inventory


def _select(
    inventory: dict[tuple[str, str], HubFile], top: str, pattern: re.Pattern, describe: str
) -> list[HubFile]:
    hits = [f for (d, name), f in sorted(inventory.items()) if d == top and pattern.match(name)]
    if not hits:
        raise FileNotFoundError(
            f"staging selector found no files for {describe} "
            f"(dir={top!r}, pattern={pattern.pattern!r})"
        )
    return hits


def _layer_pat(kind: str, layer: int) -> re.Pattern:
    return re.compile(rf"^{re.escape(kind)}_L{layer:02d}(?:_shard\d+)?\.npy$")


def _index_pat(stem: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(stem)}(?:_shard\d+)?\.jsonl$")


def select_layer_files(
    inventory: dict[tuple[str, str], HubFile],
    cells: list[str],
    model_types: list[str],
    kinds: list[str],
    layer: int,
) -> list[HubFile]:
    """Layer-sliced staging set: per-cell summary kinds + bare + dynamics for one layer."""
    files: list[HubFile] = []
    for cell in cells:
        for kind in kinds:
            files.extend(
                _select(inventory, cell, _layer_pat(kind, layer), f"{cell}/{kind} L{layer:02d}")
            )
    for mt in model_types:
        files.extend(
            _select(
                inventory,
                f"bare_{mt}",
                _layer_pat("c_q_bare", layer),
                f"bare_{mt}/c_q_bare L{layer:02d}",
            )
        )
        for kind in DYNAMICS_KINDS:
            files.extend(
                _select(
                    inventory,
                    f"dynamics_{mt}",
                    _layer_pat(kind, layer),
                    f"dynamics_{mt}/{kind} L{layer:02d}",
                )
            )
    return files


def select_static_files(
    inventory: dict[tuple[str, str], HubFile], cells: list[str], model_types: list[str]
) -> list[HubFile]:
    """Layer-independent staging set: row indexes + B0 pools (staged once, never deleted)."""
    files: list[HubFile] = []
    for mt in model_types:
        files.extend(
            _select(inventory, f"bare_{mt}", _index_pat("row_index"), f"bare_{mt}/row_index")
        )
        for kind in DYNAMICS_KINDS:
            files.extend(
                _select(
                    inventory,
                    f"dynamics_{mt}",
                    _index_pat(f"row_index_{kind}"),
                    f"dynamics_{mt}/row_index_{kind}",
                )
            )
    for cell in cells:
        if cell in B0_CELLS:
            files.extend(
                _select(
                    inventory,
                    "b0_rB_pool",
                    re.compile(rf"^{re.escape(cell)}(?:_shard\d+)?\.npy$"),
                    f"b0_rB_pool/{cell}",
                )
            )
    return files


def stage_file(hub, entry: HubFile, hf_prefix: str, stage_dir: Path) -> dict:
    """Stage one Hub file to stage_dir, sha-verified, with content-derived mtime.

    Verified reuse of a pre-existing local file requires a 64-hex LFS sha256 hub
    identity that matches the local content hash. A non-LFS identity (git blob id
    for row_index*.jsonl / small npys) is not comparable to a local sha256, so
    those files re-download unconditionally rather than pairing a fresh hub
    identity with an unverified stale local copy (code-review v10 Minor 3;
    statics are small, so the re-download is cheap).
    """
    target = stage_dir / _rel_under_prefix(entry.path, hf_prefix)
    if target.exists() and target.stat().st_size == entry.size and len(entry.hub_identity) == 64:
        sha = _sha256_file(target)
        if sha == entry.hub_identity:
            _set_content_mtime(target, sha)
            return {
                "hub_path": entry.path,
                "size": entry.size,
                "hub_identity": entry.hub_identity,
                "local_sha256": sha,
                "staged_to": str(target),
                "reused": True,
            }
        target.unlink()
    elif target.exists():
        target.unlink()
    hub.download_to(entry.path, target)
    size = target.stat().st_size
    if size != entry.size:
        raise RuntimeError(f"staged size mismatch for {entry.path}: {size} != {entry.size}")
    sha = _sha256_file(target)
    if len(entry.hub_identity) == 64 and sha != entry.hub_identity:
        raise RuntimeError(
            f"staged sha mismatch for {entry.path}: local {sha} != hub {entry.hub_identity}"
        )
    _set_content_mtime(target, sha)
    return {
        "hub_path": entry.path,
        "size": entry.size,
        "hub_identity": entry.hub_identity,
        "local_sha256": sha,
        "staged_to": str(target),
        "reused": False,
    }


def delete_staged(records: list[dict], stage_dir: Path) -> int:
    """Delete exactly the files this wrapper staged; refuse anything outside stage_dir."""
    stage_root = stage_dir.resolve()
    n = 0
    for rec in records:
        path = Path(rec["staged_to"]).resolve()
        if stage_root not in path.parents:
            raise RuntimeError(f"refusing to delete outside stage dir: {path}")
        path.unlink()
        n += 1
    return n


def validate_extra_fit_grid_args(extras: list[str]) -> list[str]:
    """Reject pass-through tokens that would clobber wrapper-owned fit-grid flags."""
    tokens: list[str] = []
    for extra in extras:
        tokens.extend(shlex.split(extra))
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag in WRAPPER_OWNED_FIT_GRID_FLAGS:
            raise ValueError(
                f"--fit-grid-arg may not set wrapper-owned flag {flag!r}; "
                "use the wrapper's own argument instead"
            )
    return tokens


def fit_grid_argv(
    args: argparse.Namespace, cells: list[str], layers_csv: str, out_dir: Path, n_null_draws: int
) -> list[str]:
    """Compose the fit-grid subprocess argv (same venv via sys.executable).

    The plan-registered fit config (plan section 6: grouped 6-fold by prefix;
    section 10: fit seed 0; section 4.5: t1/t2/t3 sensitivity targets) is threaded
    EXPLICITLY into every invocation — pilot included — from the wrapper defaults,
    so a production launch can never silently run engine defaults (code-review v10
    concern p6-launch-defaults-vs-plan-folds-targets-seed).
    """
    argv = [
        sys.executable,
        str(FIT_GRID_SCRIPT),
        "--summaries-dir",
        str(args.stage_dir),
        "--corpus-dir",
        str(args.corpus_dir),
        "--out-dir",
        str(out_dir),
        "--cells",
        ",".join(cells),
        "--layers",
        layers_csv,
        "--arms",
        args.arms,
        "--targets",
        args.targets,
        "--fit-arms",
        args.fit_arms,
        "--target-bases",
        args.target_bases,
        "--n-folds",
        str(args.n_folds),
        "--seed",
        str(args.fit_seed),
        "--n-null-draws",
        str(n_null_draws),
        "--band-null-draws",
        str(args.band_null_draws),
        "--rb-rev",
        args.rb_rev,
    ]
    if args.judge_scores is not None:
        argv += ["--judge-scores", str(args.judge_scores)]
    else:
        # issue1092_fit_grid._validate_registered_inputs raises when
        # --judge-scores is absent; recording it as missing is the documented
        # judge-deferred path for this first production run (P5 in flight).
        # verify_registered_missing() re-asserts post-run that the judge join
        # is the ONLY recorded gap.
        argv.append("--allow-missing-registered-reads")
    if args.rb_dir is not None:
        argv += ["--rb-dir", str(args.rb_dir)]
    argv.extend(validate_extra_fit_grid_args(args.fit_grid_arg))
    return argv


def run_fit_grid(argv: list[str]) -> float:
    """Run one fit-grid subprocess (inherited stdout/stderr), fail loud on nonzero exit."""
    t0 = time.monotonic()
    print(f"[p6] phase=fit-grid cmd={' '.join(argv[2:])}", flush=True)
    proc = subprocess.run(argv, cwd=str(PROJECT_ROOT), env={**os.environ})
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"fit grid exited rc={proc.returncode}: {' '.join(argv)}")
    return wall


def verify_registered_missing(out_dir: Path, judge_scores: Path | None) -> dict:
    """Assert the fit grid recorded no registered-input gap beyond the deferred judge join."""
    summary_path = out_dir / "fit_grid_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"fit grid wrote no summary at {summary_path}")
    registered = json.loads(summary_path.read_text()).get("registered_inputs", {})
    missing = registered.get("missing", [])
    expected = [] if judge_scores is not None else [EXPECTED_MISSING_WITHOUT_JUDGE]
    if missing != expected:
        raise RuntimeError(
            "fit grid recorded registered-input gaps beyond the deferred judge join: "
            f"{missing} (expected {expected})"
        )
    return registered


def verify_layer_checkpoints(
    ckpt_dir: Path,
    cells: list[str],
    arms: list[str],
    fit_arms: list[str],
    bases: list[str],
    layer: int,
) -> list[str]:
    """Assert one checkpoint exists per (cell, arm, fit_arm, basis) combo for the layer."""
    found: list[str] = []
    missing: list[str] = []
    for cell in cells:
        for arm in arms:
            for fit_arm in fit_arms:
                for basis in bases:
                    stem = f"{cell}_{arm}_fit{fit_arm}_L{layer:02d}_{basis}_"
                    hits = sorted(ckpt_dir.glob(stem + "*.json"))
                    if hits:
                        found.extend(h.name for h in hits)
                    else:
                        missing.append(stem + "*")
    if missing:
        raise FileNotFoundError(f"layer L{layer:02d} missing fit-grid checkpoints: {missing}")
    return found


def wrapper_config(
    args: argparse.Namespace,
    cells: list[str],
    arms: list[str],
    targets: list[str],
    fit_arms: list[str],
    bases: list[str],
) -> dict:
    """Every output-affecting regime key of a wrapper run, hashed for the resume predicate."""
    cfg = {
        "cells": cells,
        "arms": arms,
        "targets": targets,
        "fit_arms": fit_arms,
        "target_bases": bases,
        "n_folds": args.n_folds,
        "fit_seed": args.fit_seed,
        "lambda_grid_deviation": LAMBDA_GRID_DEVIATION,
        "n_null_draws": args.n_null_draws,
        "band_null_draws": args.band_null_draws,
        "judge_scores": str(args.judge_scores) if args.judge_scores else None,
        "judge_scores_sha256": (
            _sha256_file(args.judge_scores) if args.judge_scores is not None else None
        ),
        "rb_dir": str(args.rb_dir) if args.rb_dir else None,
        "rb_rev": args.rb_rev,
        "extra_fit_grid_args": validate_extra_fit_grid_args(args.fit_grid_arg),
        "hf_prefix": args.hf_prefix,
        "hf_revision": args.hf_revision,
        "fit_grid_script_sha256": _sha256_file(FIT_GRID_SCRIPT),
        "corpus_manifest_sha256": _sha256_file(args.corpus_dir / "manifest.jsonl"),
    }
    cfg["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return cfg


def layer_already_complete(
    manifest_path: Path,
    cfg: dict,
    expected_files: list[HubFile],
    ckpt_dir: Path,
) -> bool:
    """Conservative resume fast-path for a completed layer.

    Skip requires: manifest complete, exact wrapper-config sha match, fresh Hub
    identity (path, size, sha/blob) equal to the recorded staging set — the
    layer's OWN files PLUS the statics (row indexes, b0 pools) its fit consumed,
    so a statics-only Hub re-upload also falls through (callers pass layer +
    static files as ``expected_files``) — and every recorded checkpoint still on
    disk. ANY mismatch falls through to re-stage + the fit grid's own
    full-input fingerprint predicate (x/y + dynamics/bare/b0 paths + config,
    which loads matching checkpoints instantly), so the fail direction is
    always recompute, never wrong-skip.
    """
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        return False
    if manifest.get("wrapper_config", {}).get("config_sha256") != cfg["config_sha256"]:
        return False
    recorded = {
        (f["hub_path"], f["size"], f["hub_identity"])
        for f in manifest.get("files", []) + manifest.get("static_files", [])
    }
    fresh = {(f.path, f.size, f.hub_identity) for f in expected_files}
    if recorded != fresh:
        return False
    ckpts = manifest.get("checkpoints", [])
    if not ckpts:
        return False
    return all((ckpt_dir / name).exists() for name in ckpts)


def evaluate_pilot_gate(
    *, ru_maxrss_gb: float, projected_wall_h: float, rss_limit_gb: float, plan_wall_h: float
) -> dict:
    """Pre-registered pilot abort predicate (plan section 7 fit-grid pilot gate)."""
    rss_exceeded = ru_maxrss_gb >= rss_limit_gb
    wall_exceeded = projected_wall_h > 2.0 * plan_wall_h
    abort = rss_exceeded or wall_exceeded
    message = "pilot gate PASS"
    if abort:
        reasons = []
        if rss_exceeded:
            reasons.append(f"pilot ru_maxrss {ru_maxrss_gb:.2f} GB >= {rss_limit_gb:.1f} GB limit")
        if wall_exceeded:
            reasons.append(f"projected wall {projected_wall_h:.1f} h > 2x plan {plan_wall_h:.1f} h")
        message = (
            "P6 PILOT GATE ABORT: "
            + "; ".join(reasons)
            + f". Do NOT run the layer loop on this host; route to the escape lane: {ESCAPE_LANE}"
            " (dispatch_issue.py --intent cpu-bigmem), same code + args."
        )
    return {
        "rss_exceeded": rss_exceeded,
        "wall_exceeded": wall_exceeded,
        "abort": abort,
        "message": message,
    }


def pilot_skippable(
    out_dir: Path,
    cfg: dict,
    args: argparse.Namespace,
    *,
    n_blocks_frozen: int,
    n_blocks_band: int,
) -> bool:
    """Skip the pilot only on a recorded prior PASS under identical config AND gate knobs.

    Gate knobs (rss limit, plan wall, pilot unit) live here rather than in
    wrapper_config so tightening the gate re-runs the pilot WITHOUT
    invalidating completed layers' resume manifests. The projection block
    counts (cells x layers, frozen/band split) key the skip too: a PASS
    recorded on a narrow --layers set must not be reused for a full-set run
    whose projection is ~N x larger (code-review v10 Minor 4).
    """
    path = out_dir / "pilot.json"
    if not path.exists():
        return False
    prior = json.loads(path.read_text())
    return (
        prior.get("abort_predicate_result", {}).get("abort") is False
        and prior.get("wrapper_config", {}).get("config_sha256") == cfg["config_sha256"]
        and prior.get("pilot_cell") == args.pilot_cell
        and prior.get("pilot_layer") == args.pilot_layer
        and prior.get("rss_limit_gb") == args.max_pilot_rss_gb
        and prior.get("plan_wall_h") == args.plan_wall_h
        and prior.get("n_blocks_frozen") == n_blocks_frozen
        and prior.get("n_blocks_band") == n_blocks_band
    )


def run_pilot(
    args: argparse.Namespace,
    hub,
    inventory: dict[tuple[str, str], HubFile],
    cfg: dict,
    cells: list[str],
    arms: list[str],
    targets: list[str],
    fit_arms: list[str],
    bases: list[str],
    layers: list[int],
) -> dict:
    """Run the pre-registered pilot: one (cell, layer) production-draw unit + projection."""
    kinds = sorted(set(arms) | set(targets))
    pilot_mts = sorted({CELL_MODEL_TYPE[args.pilot_cell]})
    layer_files = select_layer_files(
        inventory, [args.pilot_cell], pilot_mts, kinds, args.pilot_layer
    )
    staged = [stage_file(hub, f, args.hf_prefix, args.stage_dir) for f in layer_files]
    print(
        f"[p6] phase=pilot layer={args.pilot_layer:02d} cell={args.pilot_cell} "
        f"staged={len(staged)}",
        flush=True,
    )
    wall_frozen = run_fit_grid(
        fit_grid_argv(
            args, [args.pilot_cell], str(args.pilot_layer), args.out_dir, args.n_null_draws
        )
    )
    ru_maxrss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024.0 * 1024.0)
    verify_registered_missing(args.out_dir, args.judge_scores)
    verify_layer_checkpoints(
        args.out_dir / "checkpoints", [args.pilot_cell], arms, fit_arms, bases, args.pilot_layer
    )
    if args.skip_band_pilot:
        wall_band = wall_frozen
        band_note = (
            "skipped (--skip-band-pilot): band-layer blocks projected at the frozen-block "
            "wall (conservative over-projection)"
        )
    else:
        wall_band = run_fit_grid(
            fit_grid_argv(
                args,
                [args.pilot_cell],
                str(args.pilot_layer),
                args.out_dir / "pilot_band",
                args.band_null_draws,
            )
        )
        band_note = (
            "measured: same (cell, layer) unit re-run at --n-null-draws=band into "
            "out_dir/pilot_band (band-draw battery cost for the 25 non-frozen layers)"
        )
    frozen = [layer for layer in layers if layer in FROZEN_NULL_LAYERS]
    band = [layer for layer in layers if layer not in FROZEN_NULL_LAYERS]
    n_blocks_frozen = len(cells) * len(frozen)
    n_blocks_band = len(cells) * len(band)
    projected_wall_h = (
        (n_blocks_frozen * wall_frozen + n_blocks_band * wall_band) / 3600.0 / EFFECTIVE_PARALLELISM
    )
    gate = evaluate_pilot_gate(
        ru_maxrss_gb=ru_maxrss_gb,
        projected_wall_h=projected_wall_h,
        rss_limit_gb=args.max_pilot_rss_gb,
        plan_wall_h=args.plan_wall_h,
    )
    pilot = {
        "pilot_cell": args.pilot_cell,
        "pilot_layer": args.pilot_layer,
        "wall_s_frozen_block": wall_frozen,
        "wall_s_band_block": wall_band,
        "band_note": band_note,
        "ru_maxrss_gb": ru_maxrss_gb,
        "ru_maxrss_note": (
            "getrusage(RUSAGE_CHILDREN).ru_maxrss read after the production-draw pilot; the "
            "fit-grid subprocess is the first substantive child, so this is its peak RSS "
            "(cumulative-max upper bound otherwise)"
        ),
        "rss_limit_gb": args.max_pilot_rss_gb,
        "plan_wall_h": args.plan_wall_h,
        "abort_wall_h": 2.0 * args.plan_wall_h,
        "n_blocks_frozen": n_blocks_frozen,
        "n_blocks_band": n_blocks_band,
        "effective_parallelism": EFFECTIVE_PARALLELISM,
        "projected_total_wall_h": projected_wall_h,
        "arithmetic": (
            f"projected_total_wall_h = (n_blocks_frozen({n_blocks_frozen}) x "
            f"wall_frozen({wall_frozen:.1f}s) + n_blocks_band({n_blocks_band}) x "
            f"wall_band({wall_band:.1f}s)) / 3600 / "
            f"effective_parallelism({EFFECTIVE_PARALLELISM}) = {projected_wall_h:.2f} h; "
            "a block is one (cell, layer) fit-grid slice incl. the identity ladder and its "
            "permutation-battery unit"
        ),
        "abort_predicate_result": gate,
        "escape_lane": ESCAPE_LANE,
        "staged_files": staged,
        "wrapper_config": cfg,
        "git_commit": _git_commit(),
        "timestamp": _timestamp(),
    }
    _write_json_atomic(args.out_dir / "pilot.json", pilot)
    print(
        f"[p6] phase=pilot-done wall_frozen_s={wall_frozen:.1f} wall_band_s={wall_band:.1f} "
        f"rss_gb={ru_maxrss_gb:.2f} projected_h={projected_wall_h:.2f} abort={gate['abort']}",
        flush=True,
    )
    return pilot


def _resolve_run_inputs(args: argparse.Namespace) -> dict:
    """Resolve paths + parse/validate the run axes (cells, arms, targets, layers)."""
    args.corpus_dir = args.corpus_dir.resolve()
    args.stage_dir = args.stage_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.judge_scores is not None:
        args.judge_scores = args.judge_scores.resolve()
    if args.rb_dir is not None:
        args.rb_dir = args.rb_dir.resolve()
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = _parse_csv(args.cells, DEFAULT_CELLS)
    unknown = [c for c in cells if c not in CELL_MODEL_TYPE]
    if unknown:
        raise ValueError(f"unknown cells (no model type in CELL_MODEL_TYPE): {unknown}")
    if args.pilot_layer not in FROZEN_NULL_LAYERS:
        raise ValueError(
            f"--pilot-layer {args.pilot_layer} must be a frozen null layer "
            f"{sorted(FROZEN_NULL_LAYERS)} so the pilot includes a production "
            "--n-null-draws permutation-battery unit"
        )
    if args.pilot_cell not in cells:
        raise ValueError(f"--pilot-cell {args.pilot_cell} not in cells {cells}")
    if not (args.corpus_dir / "manifest.jsonl").exists():
        raise FileNotFoundError(f"corpus manifest missing: {args.corpus_dir / 'manifest.jsonl'}")
    arms = _parse_csv(args.arms, ("prefix_end", "context_end"))
    targets = _parse_csv(args.targets, ("t1",))
    return {
        "cells": cells,
        "arms": arms,
        "targets": targets,
        "fit_arms": _parse_csv(args.fit_arms, ("A", "B")),
        "bases": _parse_csv(args.target_bases, ("ambient", "pca48")),
        "layers": _parse_layers(args.layers),
        "kinds": sorted(set(arms) | set(targets)),
        "model_types": sorted({CELL_MODEL_TYPE[c] for c in cells}),
    }


def run_p6(args: argparse.Namespace, hub=None) -> dict:
    """Pilot gate first, then the layer-sliced stage/fit/verify/delete loop."""
    t0 = time.monotonic()
    axes = _resolve_run_inputs(args)
    cells, arms, targets = axes["cells"], axes["arms"], axes["targets"]
    fit_arms, bases, layers = axes["fit_arms"], axes["bases"], axes["layers"]
    kinds, model_types = axes["kinds"], axes["model_types"]
    staging_dir = args.out_dir / "staging"
    ckpt_dir = args.out_dir / "checkpoints"

    if hub is None:
        if args.fixture_hub_root is not None:
            hub = LocalFixtureHubIO(args.fixture_hub_root.resolve())
        else:
            hub = HfHubIO(HF_DATA_REPO, args.hf_revision)
    resolved_rev = hub.resolved_revision()
    inventory = build_inventory(hub, args.hf_prefix)
    cfg = wrapper_config(args, cells, arms, targets, fit_arms, bases)

    static_files = select_static_files(inventory, cells, model_types)
    static_records = [stage_file(hub, f, args.hf_prefix, args.stage_dir) for f in static_files]
    _write_json_atomic(
        staging_dir / "staging_manifest_static.json",
        {
            "scope": "static",
            "status": "complete",
            "revision": resolved_rev,
            "files": static_records,
            "wrapper_config": cfg,
            "git_commit": _git_commit(),
            "timestamp": _timestamp(),
        },
    )
    print(f"[p6] phase=static-staging files={len(static_records)}", flush=True)

    pilot_staged: list[dict] = []
    n_blocks_frozen = len(cells) * len([layer for layer in layers if layer in FROZEN_NULL_LAYERS])
    n_blocks_band = len(cells) * len([layer for layer in layers if layer not in FROZEN_NULL_LAYERS])
    if pilot_skippable(
        args.out_dir, cfg, args, n_blocks_frozen=n_blocks_frozen, n_blocks_band=n_blocks_band
    ):
        pilot_summary: dict | str = "skipped_prior_pass"
        print("[p6] phase=pilot skipped=prior-pass", flush=True)
    else:
        pilot = run_pilot(args, hub, inventory, cfg, cells, arms, targets, fit_arms, bases, layers)
        pilot_staged = pilot["staged_files"]
        pilot_summary = {
            key: pilot[key]
            for key in (
                "wall_s_frozen_block",
                "wall_s_band_block",
                "ru_maxrss_gb",
                "projected_total_wall_h",
            )
        }
        gate = pilot["abort_predicate_result"]
        if gate["abort"]:
            print(gate["message"], file=sys.stderr, flush=True)
            raise SystemExit(3)

    if args.pilot_only:
        if pilot_staged:
            n_deleted = delete_staged(pilot_staged, args.stage_dir)
            print(f"[p6] phase=pilot-cleanup deleted={n_deleted}", flush=True)
        summary = {
            "phase": "P6_wrapper",
            "pilot_only": True,
            "resolved_revision": resolved_rev,
            "hf_prefix": args.hf_prefix,
            "cells": cells,
            "layers": layers,
            "layers_done": [],
            "layers_skipped": [],
            "pilot": pilot_summary,
            "wrapper_config": cfg,
            "wall_s": time.monotonic() - t0,
            "git_commit": _git_commit(),
            "timestamp": _timestamp(),
        }
        _write_json_atomic(args.out_dir / "p6_run_summary.json", summary)
        print("[p6] phase=done pilot_only=true", flush=True)
        return summary
    if pilot_staged and args.pilot_layer not in layers:
        delete_staged(pilot_staged, args.stage_dir)

    layers_done: list[int] = []
    layers_skipped: list[int] = []
    for layer in layers:
        manifest_path = staging_dir / f"staging_manifest_L{layer:02d}.json"
        expected_files = select_layer_files(inventory, cells, model_types, kinds, layer)
        # Statics (row indexes, b0 pools) are unit inputs too: a statics-only Hub
        # re-upload must fall through to the engine's fingerprint predicate.
        if layer_already_complete(manifest_path, cfg, expected_files + static_files, ckpt_dir):
            layers_skipped.append(layer)
            print(f"[p6] phase=layer layer={layer:02d} skipped=complete", flush=True)
            continue
        records = [stage_file(hub, f, args.hf_prefix, args.stage_dir) for f in expected_files]
        staged_gb = sum(r["size"] for r in records) / 1e9
        print(
            f"[p6] phase=layer layer={layer:02d} staged={len(records)} staged_gb={staged_gb:.2f}",
            flush=True,
        )
        wall = run_fit_grid(fit_grid_argv(args, cells, str(layer), args.out_dir, args.n_null_draws))
        registered = verify_registered_missing(args.out_dir, args.judge_scores)
        ckpts = verify_layer_checkpoints(ckpt_dir, cells, arms, fit_arms, bases, layer)
        n_deleted = delete_staged(records, args.stage_dir)
        _write_json_atomic(
            manifest_path,
            {
                "scope": "layer",
                "layer": layer,
                "status": "complete",
                "revision": resolved_rev,
                "files": records,
                "static_files": static_records,
                "checkpoints": ckpts,
                "registered_inputs": registered,
                "fit_grid_wall_s": wall,
                "n_staged_deleted": n_deleted,
                "wrapper_config": cfg,
                "git_commit": _git_commit(),
                "timestamp": _timestamp(),
            },
        )
        layers_done.append(layer)
        print(
            f"[p6] phase=layer-done layer={layer:02d} ckpts={len(ckpts)} wall_s={wall:.1f} "
            f"deleted={n_deleted}",
            flush=True,
        )

    summary = {
        "phase": "P6_wrapper",
        "pilot_only": False,
        "resolved_revision": resolved_rev,
        "hf_prefix": args.hf_prefix,
        "cells": cells,
        "layers": layers,
        "layers_done": layers_done,
        "layers_skipped": layers_skipped,
        "pilot": pilot_summary,
        "wrapper_config": cfg,
        "wall_s": time.monotonic() - t0,
        "git_commit": _git_commit(),
        "timestamp": _timestamp(),
    }
    _write_json_atomic(args.out_dir / "p6_run_summary.json", summary)
    print(
        f"[p6] phase=done layers_done={layers_done} layers_skipped={layers_skipped} "
        f"wall_s={summary['wall_s']:.1f}",
        flush=True,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf-prefix", default=DEFAULT_HF_PREFIX)
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--stage-dir", type=Path, default=Path("data/issue_1092/p6_stage"))
    p.add_argument("--out-dir", type=Path, default=Path("data/issue_1092/p6"))
    p.add_argument("--layers", default="0-27")
    p.add_argument("--cells", default=None, help="CSV of fit cells (default: the 8 plan cells)")
    p.add_argument("--arms", default="prefix_end,context_end")
    p.add_argument(
        "--targets",
        default="t1,t2,t3",
        help="Answer targets (plan section 4.5: t2/t3 sensitivity columns ride every unit).",
    )
    p.add_argument("--fit-arms", default="A,B")
    p.add_argument("--target-bases", default="ambient,pca48")
    p.add_argument(
        "--n-folds",
        type=int,
        default=6,
        help="Grouped-CV fold count forwarded to the fit grid (plan section 6: 6-fold by prefix).",
    )
    p.add_argument(
        "--fit-seed",
        type=int,
        default=0,
        help="Fit seed forwarded as the fit grid's --seed (plan section 10: fit seed 0).",
    )
    p.add_argument("--n-null-draws", type=int, default=200)
    p.add_argument("--band-null-draws", type=int, default=20)
    p.add_argument("--judge-scores", type=Path, default=None)
    p.add_argument("--rb-dir", type=Path, default=None)
    p.add_argument("--rb-rev", default=DEFAULT_RB_REV)
    p.add_argument("--pilot-cell", default="cell_inst_own")
    p.add_argument("--pilot-layer", type=int, default=14)
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument(
        "--skip-band-pilot",
        action="store_true",
        help="Skip the band-draw pilot re-run; project band layers at the frozen-block wall.",
    )
    p.add_argument(
        "--plan-wall-h",
        type=float,
        default=27.0,
        help="Plan section-9 P6 projection upper bound (18-27 h); abort above 2x this.",
    )
    p.add_argument("--max-pilot-rss-gb", type=float, default=14.0)
    p.add_argument(
        "--fixture-hub-root",
        type=Path,
        default=None,
        help="Offline smoke/testing: stage from this local tree instead of the HF Hub.",
    )
    p.add_argument(
        "--fit-grid-arg",
        action="append",
        default=[],
        help=(
            "Extra pass-through token(s) for issue1092_fit_grid.py, e.g. --fit-grid-arg=--n-folds=6"
        ),
    )
    return p.parse_args(argv)


def main() -> int:
    """CLI entry: load env (HF token) then run the wrapper."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    run_p6(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
