"""issue #2378 dispatch driver — pod/VM phase machine (plan v6 §7/§9/§10).

Runbook (venue per phase; provision commands are plan §10 verbatim):

  VM (repo venv):
    uv run python scripts/issue2378_dispatch.py --phase env_smoke      # model venv req'd
    uv run python scripts/issue2378_dispatch.py --phase p0_banks_pools
  Pod A (4x H200; launched detached via the canonical setsid launcher —
  experimenter.md § During Execution; this driver is the WORKLOAD):
    bash scripts/issue2378_dispatch.sh p1_pilot --attempts-per-cell 300 \\
        --chat-pilot-rows 2500 --user-sim-smoke-rows 50
    bash scripts/issue2378_dispatch.sh p2_generate --sega-attempts-per-cell <from-pilot,cap 30000>
  VM (between pods; pod A terminated after its harvest verifies):
    uv run python scripts/issue2378_dispatch.py --phase p3_admission
  Pod B (4x H200):
    bash scripts/issue2378_dispatch.sh p4_topup --cells <csv> --sega-attempts-per-cell <n>  # optional,
        # only when g2a_report.json schedules it; the VM re-runs p3_admission (wave 2,
        # cache-served for old rows) BEFORE p4_segb_capture.
    bash scripts/issue2378_dispatch.sh p4_segb_capture --target-kept-per-cell 8000 \\
        --chat-kept 9000 --user-rows 10000 --fresh-rows 1000 --fresh-draws 4 \\
        --layers "Lstar,Lstar-8,Lstar-4,Lstar+4,Lstar+8"
  VM:
    uv run python scripts/issue2378_dispatch.py --phase p5_congruence
  4x cpu-bigmem fit pods (suffixed fits-a..d — plan §9; provision per §10):
    bash scripts/issue2378_dispatch.sh p6_fits --pod-role fits-a   # runs G3 first, pushes gate
    bash scripts/issue2378_dispatch.sh p6_fits --pod-role fits-b   # waits for G3 on origin
    bash scripts/issue2378_dispatch.sh p6_fits --pod-role fits-c
    bash scripts/issue2378_dispatch.sh p6_fits --pod-role fits-d   # + pool/h3/h4b/h5/ratio/merge

Contracts implemented here (pod-side-reporting.md):
- ``[phase=<name>]`` lines on the MAIN log; ONE terminal ``[phase=done]`` on the
  graceful exit path only. Subprocess stdout/stderr are redirected to per-step
  logs, so inner scripts' ``[phase=...]`` lines never reach the main log.
- End-of-run + gate sentinels under ``/workspace/logs/issue-2378-*.json`` with
  poll_pipeline's ``_SENTINEL_REQUIRED_KEYS`` (write-once, unique names; resume
  state lives OUTSIDE the drained glob, under the dispatch logs dir).
- Designed halts exit DISTINCT rcs (below), each with a persisted report JSON —
  never a bare rc=1.
- Pre-teardown harvests: git-dest outputs commit+push from the pod clone with
  fetch+rebase (#1880), rev-list push-verify (#1205) and per-file ls-tree
  artifact-presence asserts (#1325); HF uploads via cm.upload_stage_dir
  (fail-loud + exact-set verify). Pods NEVER shell scripts/task.py.

Designed-halt exit codes:
  3  G3 refusal (mirrors issue2378_fits.G3_RC_REFUSED; gate report persisted)
  4  G1 trip, recalibration round available (ONE round; re-run p1_pilot --pilot-round 2)
  5  G1 hard fail (round 2 trip, judge-pilot FAIL, or layer-sweep rig floor)
  6  G2b survivor-predicate fail (partial-result stop; report persisted)
  7  judge pilot gate fail (mirrors issue2378_judge.RC_PILOT_GATE_FAIL)

Retry-wave policy (dispatcher-owned; plan §7 G2b): SegB-stage retries draw ONLY
from the already-admitted surplus (deterministic selection-order extras — the
same seeded order gen.phase_segb uses), <= 2 retry waves + ONE close-miss
escalation wave (>= 5,850) — never a backfill. An admitted-POOL shortfall is
G2a's to schedule (p4_topup + a VM p3_admission wave 2) BEFORE p4_segb_capture.

Smoke = the production entrypoints at small counts (PASS_UNIFIED; plan §4.6):
``--phase probe`` (CPU fixtures: gate predicates both branches, composer,
sentinel schema vs poll_pipeline, resume + CVD fan-out) and ``--dry-run``
(enumerate the composed step plan, no execution). GPU legs are smoked pod-side
at P1 by design (plan §4.6 blind-spot enumeration).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
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

import issue2378_common as cm  # noqa: E402

if Path("/workspace").exists():  # pod clones; never rebinds a VM env
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

ISSUE = 2378
BRANCH = f"issue-{ISSUE}"
SENTINEL_DIR_DEFAULT = "/workspace/logs"
LOGS_DIR_DEFAULT = cm.REPO_ROOT / "data" / "issue_2378" / "dispatch_logs"
RAW_PILOT_DEFAULT = cm.REPO_ROOT / "data" / "issue_2378" / "raw_pilot"

RC_G3_REFUSED = 3
RC_G1_RECALIBRATE = 4
RC_G1_FAIL = 5
RC_G2B_PARTIAL = 6
RC_JUDGE_PILOT_FAIL = 7

G1_NET_RATE_MIN = 0.25  # plan §7 G1(a); grounding in the plan row
G1_SWEEP_R2_MIN = 0.05  # plan §7 G1(c) rig-defect floor
WAVE1_SLACK = 1.25  # plan §8 "wave-1 sized with 1.25x slack"
SEGA_ATTEMPTS_CAP = 30_000  # plan §7 G1(a) cap
MAX_RETRY_WAVES = 2  # plan §7 G2b: <= 2 additional generation waves
PILOT_PLAIN_ROWS = 8  # tiny plain-cell slice at P1 (0 would mean ALL rows)
ADMISSION_SLICE_N = 400  # family-balanced sync slice (<= 500 smoke exemption)

# Per-phase out-root headroom floors (GB) — plan §9 disk rows; asserted against
# the mount the out-root RESOLVES to (assert_out_root_headroom, #1333).
PHASE_HEADROOM_GB = {
    "p0_banks_pools": 3,
    "p1_pilot": 12,
    "p2_generate": 8,
    "p3_admission": 4,
    "p4_topup": 6,
    "p4_segb_capture": 32,
    "p5_congruence": 3,
    "p6_fits": 10,
}

# P6 fan-out shard map (plan §9: 4 suffixed cpu-bigmem pods; shard = unit
# classes across pods; fits-d additionally owns pool + the summary phases).
POD_ROLE_CELLS: dict[str, tuple[str, ...]] = {
    "fits-a": ("chat", "plain_text", "storyq_astra"),
    "fits-b": ("storyq_helios", "storyq_wren", "storyq_dana", "storyq_vex"),
    "fits-c": ("dialog_astra", "dialog_helios", "dialog_dana", "dialog_vex"),
    "fits-d": ("chat_user_real", "chat_user_sim"),
}
POD_ROLE_JITTER_S = {"fits-a": 0, "fits-b": 180, "fits-c": 360, "fits-d": 540}
P6_SIDECAR_PREFIX = f"{cm.HF_PREFIX}/p6_sidecars"
ACTIVATIONS_PREFIX = f"{cm.HF_PREFIX}/analysis_tensors/activations"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _phase_line(name: str) -> None:
    _log(f"[phase={name}]")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Step runner + GPU fan-out (OK-flag resume; CVD pinned in the launcher env)
# ---------------------------------------------------------------------------


def _argv_sha(argv: list[str]) -> str:
    return hashlib.sha256(" ".join(argv).encode("utf-8")).hexdigest()[:16]


def visible_gpus() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [g for g in cvd.split(",") if g != ""]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, env={**os.environ}, check=False
        )
        n = len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])
        return [str(i) for i in range(n)]
    except FileNotFoundError:
        return []


def _first_gpu_env(runner: Runner, gpus: list[str], what: str) -> dict[str, str]:
    """Single-GPU CVD pin for `runner.run` steps — fail-loud on a zero-GPU
    real pod (r1 review g5 minor: the old `gpus[0] if gpus else "0"` silently
    pinned CVD=0 and deferred the failure to engine init); dry runs keep the
    composition-logging placeholder."""
    if not gpus:
        if runner.dry:
            _log(f"[dry] {what}: no visible GPUs — placeholder CVD=0 for composition")
            return {"CUDA_VISIBLE_DEVICES": "0"}
        raise RuntimeError(f"{what}: no visible GPUs")
    return {"CUDA_VISIBLE_DEVICES": gpus[0]}


class Runner:
    """Sequential/fan-out subprocess runner with per-step logs + OK-flag resume.

    OK flag = ``<logs>/<step>.ok`` holding the argv sha; a matching flag skips
    the step on resume (the wrapper's exit is never the completion signal —
    completion is the flag, written only on rc==0). Logs rotate on re-run
    (pod-side-reporting.md item 1b: never re-grep a predecessor's lines).
    """

    def __init__(self, logs_dir: Path, *, resume: bool = True, dry: bool = False):
        self.logs_dir = logs_dir
        self.resume = resume
        self.dry = dry
        self.walls: dict[str, float] = {}
        logs_dir.mkdir(parents=True, exist_ok=True)

    def _ok_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.ok"

    def _skip(self, name: str, argv: list[str]) -> bool:
        ok = self._ok_path(name)
        if self.resume and ok.exists() and ok.read_text().strip() == _argv_sha(argv):
            _log(f"[step] {name} SKIP (ok-flag matches argv sha; --no-resume to force)")
            return True
        return False

    def _open_log(self, name: str):
        path = self.logs_dir / f"{name}.log"
        if path.exists():
            path.rename(path.with_suffix(f".log.pre-{int(time.time())}"))
        return path, path.open("w", encoding="utf-8")

    def run(
        self,
        name: str,
        argv: list[str],
        *,
        env_extra: dict[str, str] | None = None,
        ok_rcs: tuple[int, ...] = (0,),
    ) -> int:
        """Run one foreground step; raise unless rc in ok_rcs; return rc."""
        if self.dry:
            _log(f"[dry] {name}: {shlex.join(argv)}")
            return 0
        if self._skip(name, argv):
            return 0
        log_path, log = self._open_log(name)
        t0 = time.time()
        _log(f"[step] {name} START log={log_path}")
        with log:
            rc = subprocess.run(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(cm.REPO_ROOT),
                env={**os.environ, **(env_extra or {})},
                check=False,
            ).returncode
        wall = time.time() - t0
        self.walls[name] = wall
        _log(f"[step] {name} rc={rc} wall={wall:.1f}s log={log_path}")
        if rc not in ok_rcs:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-25:])
            raise RuntimeError(f"step {name} failed rc={rc} (log tail below)\n{tail}")
        if rc == 0:
            self._ok_path(name).write_text(_argv_sha(argv))
        return rc

    def fanout(
        self,
        name: str,
        base_argv: list[str],
        *,
        gpus: list[str],
        env_extra: dict[str, str] | None = None,
    ) -> None:
        """One shard per GPU, CUDA_VISIBLE_DEVICES pinned in the LAUNCHER env
        (#545/#523: the in-process clobber is defeated by import-time cuInit)."""
        if not gpus and self.dry:
            # VM dry-runs have no GPUs; compose with a placeholder pin so the
            # command shapes are still logged (real runs fail loud below).
            _log(f"[dry] {name}: no visible GPUs — placeholder CVD=0 for composition")
            gpus = ["0"]
        n = len(gpus)
        if n == 0:
            raise RuntimeError(f"fanout {name}: no visible GPUs")
        full = [base_argv + ["--shard-index", str(i), "--num-shards", str(n)] for i in range(n)]
        if self.dry:
            for i, argv in enumerate(full):
                _log(f"[dry] {name}.s{i} (CVD={gpus[i]}): {shlex.join(argv)}")
            return
        procs: list[tuple[int, Path, subprocess.Popen | None]] = []
        t0 = time.time()
        for i, argv in enumerate(full):
            sname = f"{name}.s{i}"
            if self._skip(sname, argv):
                procs.append((i, self.logs_dir / f"{sname}.log", None))
                continue
            log_path, log = self._open_log(sname)
            p = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(cm.REPO_ROOT),
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpus[i],
                    **(env_extra or {}),
                },
            )
            log.close()
            _log(f"[step] {sname} START pid={p.pid} cvd={gpus[i]} log={log_path}")
            procs.append((i, log_path, p))
        failures = []
        for i, log_path, p in procs:
            if p is None:
                continue
            rc = p.wait()
            _log(f"[step] {name}.s{i} rc={rc}")
            if rc != 0:
                failures.append((i, rc, log_path))
            else:
                self._ok_path(f"{name}.s{i}").write_text(_argv_sha(full[i]))
        self.walls[name] = time.time() - t0
        if failures:
            lines = [f"shard {i} rc={rc} log={lp}" for i, rc, lp in failures]
            raise RuntimeError(
                f"fanout {name}: {len(failures)}/{n} shards failed\n" + "\n".join(lines)
            )
        _log(f"[step] {name} all {n} shards OK wall={self.walls[name]:.1f}s")

    def parallel(
        self,
        name: str,
        argv_list: list[list[str]],
        *,
        gpus: list[str],
        env_extra: dict[str, str] | None = None,
    ) -> None:
        """Run PRE-COMPOSED per-shard argvs concurrently, one launcher-env CVD
        pin per shard (fanout minus the --shard-index/--num-shards appending —
        for workers that shard by an explicit axis flag like capture's
        --cells). Same OK-flag resume, log rotation, and fail-loud collection
        as fanout; #545/#523 launcher-env CVD pin discipline."""
        if not gpus and self.dry:
            # VM dry-runs have no GPUs; placeholder pin for command composition
            # (real runs fail loud below — the probe pins that branch).
            _log(f"[dry] {name}: no visible GPUs — placeholder CVD=0 for composition")
            gpus = ["0"]
        if not gpus:
            raise RuntimeError(f"parallel {name}: no visible GPUs")
        if self.dry:
            for i, argv in enumerate(argv_list):
                _log(f"[dry] {name}.s{i} (CVD={gpus[i % len(gpus)]}): {shlex.join(argv)}")
            return
        procs: list[tuple[int, Path, subprocess.Popen | None]] = []
        t0 = time.time()
        for i, argv in enumerate(argv_list):
            sname = f"{name}.s{i}"
            if self._skip(sname, argv):
                procs.append((i, self.logs_dir / f"{sname}.log", None))
                continue
            log_path, log = self._open_log(sname)
            p = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(cm.REPO_ROOT),
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)],
                    **(env_extra or {}),
                },
            )
            log.close()
            _log(f"[step] {sname} START pid={p.pid} cvd={gpus[i % len(gpus)]} log={log_path}")
            procs.append((i, log_path, p))
        failures = []
        for i, log_path, p in procs:
            if p is None:
                continue
            rc = p.wait()
            _log(f"[step] {name}.s{i} rc={rc}")
            if rc != 0:
                failures.append((i, rc, log_path))
            else:
                self._ok_path(f"{name}.s{i}").write_text(_argv_sha(argv_list[i]))
        self.walls[name] = time.time() - t0
        if failures:
            lines = [f"shard {i} rc={rc} log={lp}" for i, rc, lp in failures]
            raise RuntimeError(
                f"parallel {name}: {len(failures)}/{len(argv_list)} shards failed\n"
                + "\n".join(lines)
            )
        _log(f"[step] {name} all {len(argv_list)} shards OK wall={self.walls[name]:.1f}s")


# ---------------------------------------------------------------------------
# Sentinels (poll_pipeline contract) + git harvest
# ---------------------------------------------------------------------------


def sentinel_dir(args) -> Path | None:
    if args.sentinel_dir:
        return Path(args.sentinel_dir)
    d = Path(SENTINEL_DIR_DEFAULT)
    return d if d.exists() else None


def write_sentinel(
    args,
    kind: str,
    note_obj: dict,
    *,
    gate: str | None = None,
    blocks_pipeline: bool = False,
) -> Path | None:
    """Write-once results/gate sentinel with poll_pipeline's required keys.

    Pod-side writers hardcode version 1 (the VM drain re-derives, #1095); the
    sentinel namespace is one-way — resume state lives in the logs dir instead.
    """
    d = sentinel_dir(args)
    if d is None:
        _log(f"[sentinel] no sentinel dir (VM phase?) — skipped kind={kind} gate={gate}")
        return None
    d.mkdir(parents=True, exist_ok=True)
    slug = kind.replace(":", "_")
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": ISSUE,
        "by": "issue2378_dispatch",
        "ts": _utc(),
        "blocks_pipeline": blocks_pipeline,
        "note": json.dumps(note_obj, sort_keys=True),
    }
    if gate:
        payload["gate"] = gate
    path = d / f"issue-{ISSUE}-{slug}-{int(time.time())}-{os.getpid()}.json"
    cm.atomic_write_json(path, payload)
    _log(f"[sentinel] wrote {path} (kind={kind} gate={gate} blocks={blocks_pipeline})")
    return path


def _git(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cm.REPO_ROOT), *argv],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=check,
    )


def git_harvest(paths: list[str], message: str, *, force_add: bool = False) -> None:
    """Commit+push git-destined outputs from this clone to the issue branch.

    fetch+rebase before push (#1880), bare push with rc checked (#957), rev-list
    push-verify (#1205), per-file ls-tree presence assert (#1325). Empty
    resolved set on declared outputs => fail loud (#1482).
    """
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"git_harvest refused: on branch {branch!r}, need {BRANCH!r}")
    resolved = sorted({p for pat in paths for p in _glob_repo(pat)})
    _log(
        f"[harvest] expected path set ({len(resolved)}): {resolved[:8]}{'...' if len(resolved) > 8 else ''}"
    )
    if not resolved:
        raise RuntimeError(f"git_harvest: EMPTY resolved set for declared outputs {paths} (#1482)")
    add_cmd = ["add"] + (["-f"] if force_add else []) + ["--"] + resolved
    _git(add_cmd)
    if force_add:
        leftover = _git(
            ["ls-files", "--others", "--ignored", "--exclude-standard", "--", *resolved]
        ).stdout.strip()
        if leftover:
            raise RuntimeError(f"git_harvest: gitignored paths not staged even with -f: {leftover}")
    staged = _git(["status", "--porcelain", "--", *resolved]).stdout.strip()
    if staged:
        _git(["commit", "-m", message, "--", *resolved])
    else:
        _log("[harvest] nothing new to commit (idempotent resume)")
    for attempt in (1, 2):
        _git(["fetch", "origin", BRANCH], check=False)
        reb = _git(["rebase", f"origin/{BRANCH}"], check=False)
        if reb.returncode != 0:
            _git(["rebase", "--abort"], check=False)
            raise RuntimeError(
                f"git_harvest: rebase onto origin/{BRANCH} conflicted:\n{reb.stderr}"
            )
        push = _git(["push", "origin", BRANCH], check=False)
        if push.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(f"git_harvest: push failed twice:\n{push.stderr}")
    behind = _git(["rev-list", "--count", f"origin/{BRANCH}..HEAD"]).stdout.strip()
    if behind != "0":
        raise RuntimeError(f"git_harvest: push-verify failed (rev-list count {behind})")
    for p in resolved:
        present = _git(["ls-tree", "-r", f"origin/{BRANCH}", "--name-only", "--", p]).stdout.strip()
        if not present:
            raise RuntimeError(f"git_harvest: {p} missing from pushed tree (#1325)")
    _log(f"[harvest] pushed + verified {len(resolved)} paths: {message!r}")


def _glob_repo(pattern: str) -> list[str]:
    root = cm.REPO_ROOT
    if any(ch in pattern for ch in "*?["):
        return [str(p.relative_to(root)) for p in sorted(root.glob(pattern)) if p.is_file()]
    return [pattern] if (root / pattern).is_file() else []


def _git_wait_for(paths: list[str], *, poll_s: int, timeout_s: int, what: str) -> None:
    """Poll origin/<branch> until every path exists in the fetched tree."""
    t0 = time.time()
    while True:
        _git(["fetch", "origin", BRANCH], check=False)
        missing = [
            p
            for p in paths
            if _git(["cat-file", "-e", f"origin/{BRANCH}:{p}"], check=False).returncode != 0
        ]
        if not missing:
            _log(f"[wait] {what}: all {len(paths)} paths present on origin/{BRANCH}")
            return
        waited = time.time() - t0
        if waited > timeout_s:
            raise RuntimeError(f"[wait] {what}: timed out ({timeout_s}s); missing {missing[:6]}")
        _log(f"[wait] {what}: {len(missing)} missing (waited {waited:.0f}s) — sleeping {poll_s}s")
        time.sleep(poll_s)


def _git_pull_rebase() -> None:
    _git(["fetch", "origin", BRANCH])
    reb = _git(["rebase", f"origin/{BRANCH}"], check=False)
    if reb.returncode != 0:
        _git(["rebase", "--abort"], check=False)
        raise RuntimeError(f"git pull-rebase onto origin/{BRANCH} conflicted:\n{reb.stderr}")


def assert_headroom(phase: str, out_root: Path) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    out_root.mkdir(parents=True, exist_ok=True)
    free = assert_out_root_headroom(out_root, PHASE_HEADROOM_GB[phase], phase=phase)
    _log(f"[headroom] {phase}: {free:.1f} GB free at {out_root} (floor {PHASE_HEADROOM_GB[phase]})")


# ---------------------------------------------------------------------------
# Shared helpers: entrypoint argv builders, layer spec, HF pilot upload
# ---------------------------------------------------------------------------


def _py(script: str, *argv: str) -> list[str]:
    return ["uv", "run", "python", str(cm.REPO_ROOT / "scripts" / script), *argv]


def parse_layers_spec(spec: str, lstar: int) -> list[int]:
    """``Lstar,Lstar-8,...`` (or plain ints) -> sorted unique ints clamped [1,63]."""
    out = set()
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        t = tok.replace(" ", "")
        if t.lower().startswith("lstar"):
            rest = t[5:]
            val = lstar + int(rest) if rest else lstar
        else:
            val = int(t)
        out.add(min(63, max(1, val)))
    if not out:
        raise ValueError(f"empty --layers spec {spec!r}")
    return sorted(out)


def resolve_lstar(ledger_root: Path) -> int:
    path = ledger_root / "pilot" / "layer_sweep.json"
    if not path.exists():
        raise RuntimeError(f"missing {path} — P1 harvest not pulled? (git pull the issue branch)")
    return int(json.loads(path.read_text(encoding="utf-8"))["selected_layer"])


def upload_json_files(files: list[Path], prefix_rel: str) -> None:
    """Stage named JSON files into a temp dir and upload_stage_dir (fail-loud)."""
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for f in files:
            if not f.exists():
                raise RuntimeError(f"upload_json_files: missing {f}")
            shutil.copy2(f, Path(td) / f.name)
        cm.upload_stage_dir(Path(td), prefix_rel)


def balanced_mined_slice(mined_dir: Path, out_dir: Path, n_total: int) -> int:
    """Family-balanced, cell-round-robin mined subset for the P1 admission-rate
    slice (judge --max-items slices sorted row_ids, which clusters by cell —
    a raw slice would starve one family). Deterministic under cm.SEED."""
    import issue2378_gen as gen

    mined = gen._load_mined_rows(mined_dir)
    by_family: dict[str, dict[str, list[str]]] = {"question": {}, "dialogue": {}}
    for rid, m in mined.items():
        by_family[m["family"]].setdefault(m["cell"], []).append(rid)
    per_family = (n_total + 1) // 2
    chosen: list[str] = []
    for family in ("question", "dialogue"):
        cells = by_family[family]
        if not cells:
            raise RuntimeError(f"admission slice: no mined rows in family {family}")
        pools = {}
        for cell, rids in cells.items():
            order = random.Random(cm.derived_seed(cm.SEED, "adm_slice", family, cell)).sample(
                range(len(rids)), len(rids)
            )
            pools[cell] = [sorted(rids)[i] for i in order]
        take: list[str] = []
        names = sorted(pools)
        i = 0
        while len(take) < per_family and any(pools[c] for c in names):
            c = names[i % len(names)]
            if pools[c]:
                take.append(pools[c].pop(0))
            i += 1
        chosen.extend(take)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "slice.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for rid in sorted(chosen):
            fh.write(json.dumps(mined[rid], sort_keys=True) + "\n")
    _log(f"[adm-slice] wrote {len(chosen)} balanced rows -> {out}")
    return len(chosen)


# ---------------------------------------------------------------------------
# pilot_digest composer + G1 evaluation (closes concern pilot-digest-composer-missing)
# ---------------------------------------------------------------------------


def _sum_stage_summaries(stage_dir: Path, keys: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Aggregate gen.py per-shard summaries: <stage>/summary_<cell>_w*_s*.json."""
    per_cell: dict[str, dict[str, int]] = {}
    for path in sorted(stage_dir.glob("summary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = payload.get("cell")
        if cell is None:  # single-summary stages (user_sim writes summary_w1_s0.json)
            cell = payload.get("stage", stage_dir.name)
        bucket = per_cell.setdefault(cell, {k: 0 for k in keys})
        for k in keys:
            bucket[k] += int(payload.get("counts", {}).get(k, 0))
    return per_cell


def _family_pool(per_cell: dict[str, dict[str, int]], num: str, den: str) -> dict[str, dict]:
    fams: dict[str, dict] = {}
    for cell, c in per_cell.items():
        fam = cm.CELL_FAMILY.get(cell)
        if fam is None:
            continue
        f = fams.setdefault(fam, {"numerator": 0, "denominator": 0})
        f["numerator"] += c.get(num, 0)
        f["denominator"] += c.get(den, 0)
    for f in fams.values():
        f["rate"] = f["numerator"] / f["denominator"] if f["denominator"] else 0.0
    return fams


def compose_pilot_digest(
    raw_pilot: Path,
    ledger_root: Path,
    walls: dict[str, float],
    *,
    pilot_round: int,
    attempts_per_cell: int,
) -> dict:
    """Aggregate P1 per-stage counters into the G1 artifact (plan §7 G1)."""
    pilot_dir = ledger_root / "pilot"
    mining = _sum_stage_summaries(raw_pilot / "sega", ("attempts", "kept", "cap_hit"))
    segb = _sum_stage_summaries(raw_pilot / "segb", ("rows", "kept", "cap_hit_no_close"))
    admission: dict[str, dict[str, int]] = {}
    for path in sorted((pilot_dir / "kept").glob("*.json")):
        k = json.loads(path.read_text(encoding="utf-8"))
        admission[k["cell"]] = {"n_items": k["n_items"], "n_admitted": k["n_admitted"]}
    stages = {
        "mining": _family_pool(mining, "kept", "attempts"),
        "admission": _family_pool(admission, "n_admitted", "n_items"),
        "segb_survival": _family_pool(segb, "kept", "rows"),
    }
    families: dict[str, dict] = {}
    for fam in ("question", "dialogue"):
        net = 1.0
        for st in stages.values():
            net *= st.get(fam, {}).get("rate", 0.0)
        sized = (
            min(SEGA_ATTEMPTS_CAP, math.ceil(cm.STORY_TARGET_KEPT * WAVE1_SLACK / net))
            if net > 0
            else SEGA_ATTEMPTS_CAP
        )
        families[fam] = {
            "net_kept_per_attempt": net,
            "trip_line": G1_NET_RATE_MIN,
            "pass": net >= G1_NET_RATE_MIN,
            "wave1_attempts_per_cell": sized,
        }
    judge_report_path = ledger_root / "judge" / "pilot_admission_sync.json"
    judge_report = (
        json.loads(judge_report_path.read_text(encoding="utf-8"))
        if judge_report_path.exists()
        else {"verdict": "MISSING"}
    )
    sweep_path = pilot_dir / "layer_sweep.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.exists() else {}
    # capture.py layer_sweep.json shape: {"selected_layer": int,
    #   "gate_g1c": {"threshold": float, "max_r2": float, "passes": bool}, ...}
    gate_g1c = sweep.get("gate_g1c", {})
    best_r2 = float(gate_g1c.get("max_r2", float("nan")))
    # G1e: SUM user_sim counts across shard summaries (r1 review g5 minor —
    # last-shard-wins read) + carry the per-shard cap-hit fractions.
    user_sim: dict[str, int] = {}
    user_sim_cap_hit: dict[str, float] = {}
    for path in sorted((raw_pilot / "user_sim").glob("summary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for k, v in payload.get("counts", {}).items():
            if isinstance(v, int | float):
                user_sim[k] = user_sim.get(k, 0) + int(v)
        if "cap_hit_fraction_after" in payload:
            user_sim_cap_hit[path.stem] = float(payload["cap_hit_fraction_after"])
    reasons: list[str] = []
    for fam, f in families.items():
        if not f["pass"]:
            reasons.append(f"G1(a) {fam}: net {f['net_kept_per_attempt']:.4f} < {G1_NET_RATE_MIN}")
    if judge_report.get("verdict") != "PASS":
        reasons.append(f"G1(b) judge sync pilot verdict {judge_report.get('verdict')!r}")
    if not (best_r2 >= G1_SWEEP_R2_MIN):
        reasons.append(f"G1(c) layer-sweep best R2 {best_r2!r} < {G1_SWEEP_R2_MIN}")
    digest = {
        "gate": "G1",
        "pilot_round": pilot_round,
        "attempts_per_cell": attempts_per_cell,
        "verdict": "PASS" if not reasons else "FAIL",
        "fail_reasons": reasons,
        "per_stage": stages,
        "per_cell": {"mining": mining, "admission": admission, "segb": segb},
        "families": families,
        "judge_sync_pilot": {
            "verdict": judge_report.get("verdict"),
            "report": str(judge_report_path),
        },
        "layer_sweep": {
            "selected_layer": sweep.get("selected_layer"),
            "max_r2": best_r2,
            "gate_g1c_passes": gate_g1c.get("passes"),
            "floor": G1_SWEEP_R2_MIN,
        },
        "sim_user_smoke": {
            "counts": user_sim,
            "cap_hit_fraction_after_per_shard": user_sim_cap_hit,
            "disposition": "advisory (G1e)",
        },
        "measured_walls_s": {k: round(v, 1) for k, v in walls.items()},
        "fences_s_2x": {k: round(2 * v, 1) for k, v in walls.items()},
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(pilot_dir / "pilot_digest.json", digest)
    _log(f"[g1] verdict={digest['verdict']} reasons={reasons}")
    return digest


# ---------------------------------------------------------------------------
# G2a / G2b evaluation + retry-extras selection
# ---------------------------------------------------------------------------


def compose_g2a(ledger_root: Path) -> dict:
    """NON-BINDING wave-sizing checkpoint (plan §7 G2a): projected kept-after-
    SegB from admission counts x pilot SegB survival; schedules topups only."""
    digest = json.loads((ledger_root / "pilot" / "pilot_digest.json").read_text(encoding="utf-8"))
    surv = {fam: s.get("rate", 0.0) for fam, s in digest["per_stage"]["segb_survival"].items()}
    cells: dict[str, dict] = {}
    topups: list[str] = []
    for path in sorted((ledger_root / "kept").glob("*.json")):
        k = json.loads(path.read_text(encoding="utf-8"))
        cell, fam = k["cell"], k["family"]
        selected = min(k["n_admitted"], cm.STORY_TARGET_KEPT)
        projected = selected * surv.get(fam, 0.0)
        need_topup = projected < cm.FLOOR_KEPT
        cells[cell] = {
            "n_admitted": k["n_admitted"],
            "selected": selected,
            "segb_survival_pilot": surv.get(fam, 0.0),
            "projected_kept": projected,
            "floor": cm.FLOOR_KEPT,
            "schedule_sega_topup": need_topup,
        }
        if need_topup:
            topups.append(cell)
    out = {
        "gate": "G2a",
        "binding": False,
        "cells": cells,
        "sega_topup_cells": topups,
        "note": "wave sizing only; kept.json predates SegB/cap-hit/retries (plan §7)",
        "gate_quantity_deviation": (
            "projection uses min(n_admitted, STORY_TARGET_KEPT) x survival, not the plan §7 "
            "literal n_admitted x survival: only STORY_TARGET_KEPT rows enter SegB wave 1, so "
            "the surplus is retry-wave budget, not wave-1 yield. Conservative-in-coverage — "
            "may schedule a top-up the plan formula would skip (r1 review g5 concern, recorded "
            "deviation)."
        ),
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(ledger_root / "g2a_report.json", out)
    _log(f"[g2a] topup cells: {topups or 'none'}")
    return out


def segb_extras(kept_dir: Path, cell: str, consumed: int, n_extra: int) -> list[str]:
    """The NEXT n_extra admitted ids in gen.phase_segb's exact seeded selection
    order (deterministic: retry waves regenerate nothing already selected)."""
    import issue2378_gen as gen

    kept_ids = gen._load_kept_ids(kept_dir, cell)
    order = random.Random(cm.derived_seed(cm.SEED, "segb_select", cell)).sample(
        range(len(kept_ids)), len(kept_ids)
    )
    window = order[consumed : consumed + n_extra]
    return [kept_ids[i] for i in window]


def write_retry_kept(kept_dir: Path, out_dir: Path, cell: str, extras: list[str]) -> None:
    src = json.loads((kept_dir / f"{cell}.json").read_text(encoding="utf-8"))
    scores = {a["row_id"]: a.get("score") for a in src["admitted"]}
    payload = {**src, "admitted": [{"row_id": r, "score": scores.get(r)} for r in extras]}
    out_dir.mkdir(parents=True, exist_ok=True)
    cm.atomic_write_json(out_dir / f"{cell}.json", payload)


def evaluate_g2b(
    ledger_root: Path, *, waves_used: int, escalated: list[str], escalation_wave_used: bool = False
) -> dict:
    """BINDING floor gate on capture_ready ledgers (plan §7 G2b). User cells
    are OUTSIDE the binding predicate (reported loudly, never a stop)."""
    ready_dir = ledger_root / "capture_ready"
    cells: dict[str, dict] = {}
    for path in sorted(ready_dir.glob("*.json")):
        k = json.loads(path.read_text(encoding="utf-8"))
        cells[k["cell"]] = {
            "n_kept": k["n_kept"],
            "floor_pass": k["floor_pass"],
            "close_miss_band": k.get("close_miss_band", False),
        }
    missing = [c for c in cm.ALL_CELLS if c not in cells]
    survivors = [c for c, v in cells.items() if v["floor_pass"]]
    drops = [c for c, v in cells.items() if not v["floor_pass"]] + missing
    n_q = len([c for c in survivors if c in cm.STORY_Q_CELLS])
    n_d = len([c for c in survivors if c in cm.DIALOG_CELLS])
    predicate = "chat" in survivors and "plain_text" in survivors and n_q >= 3 and n_d >= 2
    user_drops = [c for c in drops if c in cm.USER_CELLS]
    binding_drops = [c for c in drops if c not in cm.USER_CELLS]
    out = {
        "gate": "G2b",
        "binding": True,
        "verdict": "PASS" if predicate else "FAIL",
        "cells": cells,
        "missing_cells": missing,
        "survivors": survivors,
        "dropped_cells": drops,
        "user_cell_drops_nonbinding": user_drops,
        "binding_drops": binding_drops,
        "survivor_predicate": "chat + plain + >=3 storyQ + >=2 dialogue (user cells excluded)",
        "story_q_survivors": n_q,
        "dialog_survivors": n_d,
        # retry waves (2/3) only — the close-miss escalation wave is reported
        # separately (r1 review g5 minor: waves_used labeling)
        "retry_waves_used": waves_used,
        "escalation_wave_used": escalation_wave_used,
        "escalated_cells": escalated,
        "floor": cm.FLOOR_KEPT,
        "metadata": cm.run_metadata(),
    }
    cm.atomic_write_json(ledger_root / "g2b_report.json", out)
    _log(
        f"[g2b] verdict={out['verdict']} survivors={len(survivors)} drops={drops or 'none'} "
        f"user_drops={user_drops or 'none'}"
    )
    return out


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_env_smoke(args) -> int:
    """Plan §12 assumption 2 (blocking, before any provisioning; model venv)."""
    _phase_line("env_smoke")
    import importlib.util

    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        raise RuntimeError("transformers lacks qwen3_5 — upgrade the model venv")
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(cm.MODEL_ID)
    text_cfg = getattr(cfg, "text_config", cfg)
    assert int(text_cfg.num_hidden_layers) == 64, text_cfg.num_hidden_layers
    assert int(text_cfg.hidden_size) == 5120, text_cfg.hidden_size
    tok = AutoTokenizer.from_pretrained(cm.MODEL_ID)
    single = tok.apply_chat_template(
        [{"role": "user", "content": "probe"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "<think>\n\n</think>" in single, "expected the EMPTY think block in the render"
    u2 = "second user turn probe text"
    multi = tok.apply_chat_template(
        [
            {"role": "user", "content": "first user turn"},
            {"role": "assistant", "content": "assistant reply"},
            {"role": "user", "content": u2},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    hdr = cm.USER_TURN_HEADER
    second_hdr = multi.find(hdr, multi.find(hdr) + 1)
    assert second_hdr != -1, "no second user-turn header in the multi-turn render"
    lo = second_hdr + len(hdr)
    assert multi[lo : lo + len(u2)] == u2, "user_2 span not byte-exact at the computed offset"
    _log("[env_smoke] qwen3_5 + tokenizer render asserts PASS")
    return 0


def phase_p0(args, runner: Runner) -> int:
    _phase_line("p0_banks_pools")
    assert_headroom("p0_banks_pools", cm.REPO_ROOT / "data" / "issue_2378")
    runner.run("p0.build_banks", _py("issue2378_gen.py", "--phase", "build_banks"))
    runner.run(
        "p0.build_pools",
        _py(
            "issue2378_gen.py",
            "--phase",
            "build_pools",
            "--chat-rows",
            str(args.chat_rows),
            "--plain-rows",
            str(args.plain_rows),
            "--user-sim-rows",
            str(args.user_rows),
        ),
    )
    if not runner.dry:
        git_harvest(
            ["data/issue_2378/banks/*.json"],
            f"task #{ISSUE}: P0 banks (committed per plan §4.1)",
            force_add=True,  # data/ is gitignored; banks are KB-scale committed JSON
        )
    return 0


def _pilot_roots(args) -> tuple[Path, Path]:
    return Path(args.raw_pilot_root), Path(args.ledger_root)


def _pilot_round_scope(raw_pilot: Path, runner: Runner, rnd: int) -> tuple[Path, Runner, str]:
    """Round-scope the pilot resume key (logs dir), raw root, and HF prefix
    (r1 review g5 blocker 2, G1 recalibration resume-skip): a `--pilot-round 2`
    re-pilot must RE-RUN every generation/judge/capture step instead of
    skipping onto round-1 OK-flags and reproducing the trip. Round 1 stays
    byte-identical. Ledger pilot outputs (kept/, judge reports, digest, sweep)
    keep their STABLE paths — round 2 re-runs + overwrites them (the final
    pilot verdict), so the P2 gate + harvest paths are unchanged."""
    hf_pilot_prefix = f"{cm.HF_PREFIX}/raw_completions/pilot"
    if rnd > 1:
        raw_pilot = raw_pilot / f"r{rnd}"
        runner = Runner(runner.logs_dir / f"p1_pilot_r{rnd}", resume=runner.resume, dry=runner.dry)
        hf_pilot_prefix = f"{cm.HF_PREFIX}/raw_completions/pilot/r{rnd}"
    return raw_pilot, runner, hf_pilot_prefix


def phase_p1(args, runner: Runner) -> int:
    """P1 pilot on pod A (plan §7 G1 + §9 row 1). Pilot generations run under a
    SEPARATE raw root (regime isolation vs P2 production ledgers) and upload
    under raw_completions/pilot/<stage>."""
    _phase_line("p1_pilot")
    raw_pilot, ledger_root = _pilot_roots(args)
    raw_pilot, runner, hf_pilot_prefix = _pilot_round_scope(
        raw_pilot, runner, int(args.pilot_round)
    )
    assert_headroom("p1_pilot", raw_pilot)
    gpus = visible_gpus()
    pilot_kept = ledger_root / "pilot" / "kept"
    common = ["--raw-root", str(raw_pilot), "--skip-upload"]

    runner.fanout(
        "p1.sega",
        _py(
            "issue2378_gen.py",
            "--phase",
            "sega",
            "--sega-attempts-per-cell",
            str(args.attempts_per_cell),
            *common,
        ),
        gpus=gpus,
    )
    # r1 review codex blocker cross-pod-pools-not-staged: every pool-consuming
    # gen phase stages the P0 pools from HF (idempotent; a fresh pod A' after
    # a relaunch has no local pools — P0 committed banks to git, pools to HF).
    runner.fanout(
        "p1.chat_plain",
        _py(
            "issue2378_gen.py",
            "--phase",
            "chat_plain",
            "--chat-rows",
            str(args.chat_pilot_rows),
            "--plain-rows",
            str(PILOT_PLAIN_ROWS),
            "--stage-pools-from-hf",
            *common,
        ),
        gpus=gpus,
    )
    runner.run(
        "p1.user_sim_smoke",
        _py(
            "issue2378_gen.py",
            "--phase",
            "user_sim",
            "--user-sim-rows",
            str(args.user_sim_smoke_rows),
            "--stage-pools-from-hf",
            *common,
        ),
        env_extra=_first_gpu_env(runner, gpus, "p1.user_sim_smoke"),
    )
    runner.run(
        "p1.judge_sync_pilot",
        _py(
            "issue2378_judge.py",
            "--wave",
            "admission",
            "--pilot",
            "200",
            "--transport",
            "sync",
            "--mined-dir",
            str(raw_pilot / "sega_mined"),
            "--raw-root",
            str(raw_pilot),
            "--out-root",
            str(ledger_root),
            "--skip-upload",
        ),
        ok_rcs=(0, RC_JUDGE_PILOT_FAIL),
    )
    if not runner.dry:
        slice_dir = raw_pilot / "adm_slice"
        balanced_mined_slice(raw_pilot / "sega_mined", slice_dir, ADMISSION_SLICE_N)
    else:
        slice_dir = raw_pilot / "adm_slice"
        _log(f"[dry] p1.adm_slice: balanced_mined_slice(n={ADMISSION_SLICE_N}) -> {slice_dir}")
    runner.run(
        "p1.admission_slice",
        _py(
            "issue2378_judge.py",
            "--wave",
            "admission",
            "--transport",
            "sync",
            "--max-items",
            str(ADMISSION_SLICE_N),
            "--mined-dir",
            str(slice_dir),
            "--raw-root",
            str(raw_pilot),
            "--out-root",
            str(ledger_root / "pilot"),
            "--skip-upload",
        ),
    )
    runner.fanout(
        "p1.segb",
        _py(
            "issue2378_gen.py",
            "--phase",
            "segb",
            "--mined-dir",
            str(raw_pilot / "sega_mined"),
            "--kept-dir",
            str(pilot_kept),
            "--target-kept-per-cell",
            "100000",
            *common,
        ),
        gpus=gpus,
    )
    runner.run(
        "p1.capture_pilot",
        _py(
            "issue2378_capture.py",
            "--phase",
            "pilot",
            "--pilot-rows",
            str(args.chat_pilot_rows),
            "--raw-root",
            str(raw_pilot),
            "--skip-capture-ready",
            "--layer-sweep-out",
            str(ledger_root / "pilot" / "layer_sweep.json"),
            "--skip-upload",
        ),
        env_extra=_first_gpu_env(runner, gpus, "p1.capture_pilot"),
    )
    if runner.dry:
        _log(
            "[dry] p1.digest: compose_pilot_digest + G1 predicates "
            f"(net>= {G1_NET_RATE_MIN}/family; judge PASS; sweep R2 >= {G1_SWEEP_R2_MIN})"
        )
        return 0
    for stage in ("sega", "sega_mined", "chat", "plain", "user_sim", "segb", "judge_admission"):
        d = raw_pilot / stage
        if d.exists():
            cm.upload_stage_dir(d, f"{hf_pilot_prefix}/{stage}")
    digest = compose_pilot_digest(
        raw_pilot,
        ledger_root,
        runner.walls,
        pilot_round=args.pilot_round,
        attempts_per_cell=args.attempts_per_cell,
    )
    blocks = digest["verdict"] != "PASS"
    write_sentinel(args, "epm:progress", digest, gate="g1", blocks_pipeline=blocks)
    # Persist-by-default (r1 review g5 minor): harvest the digest + layer
    # sweep + judge report on EVERY verdict branch — a TRIP/FAIL previously
    # persisted them only inside the sentinel note, and pod A can be lost
    # post-trip.
    upload_json_files(
        [ledger_root / "pilot" / "pilot_digest.json", ledger_root / "pilot" / "layer_sweep.json"],
        f"{cm.HF_PREFIX}/pilot",
    )
    git_harvest(
        [
            "eval_results/issue_2378/pilot/pilot_digest.json",
            "eval_results/issue_2378/pilot/layer_sweep.json",
            "eval_results/issue_2378/judge/pilot_admission_sync.json",
        ],
        f"task #{ISSUE}: P1 pilot artifacts (G1 {digest['verdict']} harvest, pre-P2 — plan §9)",
    )
    if digest["verdict"] != "PASS":
        only_rate_trips = all(r.startswith("G1(a)") for r in digest["fail_reasons"])
        if only_rate_trips and args.pilot_round == 1:
            _log(
                "[g1] TRIP — ONE recalibration round available (primes/seeds/miner), "
                "re-run p1_pilot --pilot-round 2 after recalibrating"
            )
            return RC_G1_RECALIBRATE
        return RC_G1_FAIL
    write_sentinel(
        args,
        "epm:progress",
        {
            "phase": "p1_pilot",
            "status": "complete",
            "g1": "PASS",
            "wave1_sizing": digest["families"],
        },
    )
    return 0


def phase_p2(args, runner: Runner) -> int:
    _phase_line("p2_generate")
    assert_headroom("p2_generate", Path(args.raw_root))
    ledger_root = Path(args.ledger_root)
    gpus = visible_gpus()
    if args.sega_attempts_per_cell > 0:
        per_family = {
            "question": args.sega_attempts_per_cell,
            "dialogue": args.sega_attempts_per_cell,
        }
    elif not runner.dry:
        digest = json.loads(
            (ledger_root / "pilot" / "pilot_digest.json").read_text(encoding="utf-8")
        )
        if digest["verdict"] != "PASS":
            raise RuntimeError("p2_generate refused: pilot_digest verdict is not PASS (G1)")
        per_family = {
            fam: min(SEGA_ATTEMPTS_CAP, int(f["wave1_attempts_per_cell"]))
            for fam, f in digest["families"].items()
        }
    else:
        per_family = {"question": 0, "dialogue": 0}
    fam_cells = {
        "question": [c for c in cm.STORY_Q_CELLS],
        "dialogue": [c for c in cm.DIALOG_CELLS],
    }
    for fam, attempts in per_family.items():
        runner.fanout(
            f"p2.sega.{fam}",
            _py(
                "issue2378_gen.py",
                "--phase",
                "sega",
                "--cells",
                ",".join(fam_cells[fam]),
                "--sega-attempts-per-cell",
                str(attempts),
                "--skip-upload",
            ),
            gpus=gpus,
        )
    runner.run(
        "p2.upload_sega", _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", "sega")
    )
    runner.run(
        "p2.upload_sega_mined",
        _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", "sega_mined"),
    )
    runner.fanout(
        "p2.chat_plain",
        _py(
            "issue2378_gen.py",
            "--phase",
            "chat_plain",
            "--chat-rows",
            str(args.chat_rows),
            "--plain-rows",
            str(args.plain_rows),
            "--stage-pools-from-hf",  # codex blocker cross-pod-pools-not-staged
            "--skip-upload",
        ),
        gpus=gpus,
    )
    for st in ("chat", "plain"):
        runner.run(
            f"p2.upload_{st}", _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", st)
        )
    runner.fanout(
        "p2.user_sim",
        _py(
            "issue2378_gen.py",
            "--phase",
            "user_sim",
            "--user-sim-rows",
            str(args.user_rows),
            "--stage-pools-from-hf",  # codex blocker cross-pod-pools-not-staged
            "--skip-upload",
        ),
        gpus=gpus,
    )
    runner.run(
        "p2.upload_user_sim",
        _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", "user_sim"),
    )
    runner.fanout(
        "p2.user_fresh",
        _py(
            "issue2378_gen.py",
            "--phase",
            "user_fresh",
            "--user-fresh-rows",
            str(args.user_fresh_rows),
            "--user-fresh-draws",
            str(args.user_fresh_draws),
            "--stage-pools-from-hf",  # codex blocker cross-pod-pools-not-staged
            "--skip-upload",
        ),
        gpus=gpus,
    )
    runner.run(
        "p2.upload_user_fresh",
        _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", "user_sim_fresh"),
    )
    if not runner.dry:
        write_sentinel(
            args,
            "epm:progress",
            {
                "phase": "p2_generate",
                "status": "complete",
                "sega_attempts_per_cell": per_family,
                "walls_s": {k: round(v, 1) for k, v in runner.walls.items()},
            },
        )
    return 0


def _sega_mined_manifest_digest(dry: bool) -> str:
    """sha16 over the sorted (path, size) HF listing of raw_completions/
    sega_mined — appended to the P3 judge step names so a P4 top-up that GROWS
    the admission input re-runs the judge steps instead of resume-skipping on
    an unchanged argv sha (r1 review codex blocker
    topup-readmission-resume-collision; the judge cache serves already-judged
    rows, so the re-run only judges the new rows). Dry mode composes with a
    fixed placeholder — no network."""
    if dry:
        return "dryrun"
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    entries = hub.list_hf_entries_under_path(
        HfApi(),
        cm.HF_DATA_REPO,
        f"{cm.HF_PREFIX}/raw_completions/sega_mined",
        repo_type="dataset",
    )
    blob = json.dumps(sorted([p, int(s or 0)] for p, s in entries)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def phase_p3(args, runner: Runner) -> int:
    """VM phase (between pods): batch pilot -> admission wave -> G2a -> harvest."""
    _phase_line("p3_admission")
    assert_headroom("p3_admission", cm.REPO_ROOT / "data" / "issue_2378")
    ledger_root = Path(args.ledger_root)
    mdig = _sega_mined_manifest_digest(runner.dry)
    rc = runner.run(
        f"p3.judge_batch_pilot.{mdig}",
        _py(
            "issue2378_judge.py",
            "--wave",
            "admission",
            "--pilot",
            "200",
            "--transport",
            "batch",
            "--stage-from-hf",
        ),
        ok_rcs=(0, RC_JUDGE_PILOT_FAIL),
    )
    if rc == RC_JUDGE_PILOT_FAIL:
        _log("[p3] batch pilot gate FAIL — designed halt (report persisted by judge.py)")
        return RC_JUDGE_PILOT_FAIL
    runner.run(
        f"p3.admission_wave.{mdig}",
        _py(
            "issue2378_judge.py",
            "--wave",
            "admission",
            "--transport",
            "batch",
            "--stage-from-hf",
        ),
    )
    if runner.dry:
        _log("[dry] p3.g2a: compose_g2a (non-binding wave sizing)")
        return 0
    compose_g2a(ledger_root)
    git_harvest(
        [
            "eval_results/issue_2378/kept/*.json",
            "eval_results/issue_2378/judge/admission_summary.json",
            "eval_results/issue_2378/judge/pilot_admission_batch.json",
            "eval_results/issue_2378/g2a_report.json",
        ],
        f"task #{ISSUE}: P3 admission keeps + G2a wave-sizing report",
    )
    return 0


def phase_p4_topup(args, runner: Runner) -> int:
    """Optional pod-B SegA top-up wave (scheduled by G2a). The VM re-runs
    p3_admission afterwards (wave 2; the judge cache serves old rows)."""
    _phase_line("p4_topup")
    assert_headroom("p4_topup", Path(args.raw_root))
    if not args.cells:
        raise SystemExit("p4_topup requires --cells (from g2a_report.json sega_topup_cells)")
    if args.sega_attempts_per_cell <= 0:
        raise SystemExit("p4_topup requires --sega-attempts-per-cell (sized from measured rates)")
    runner.fanout(
        "p4.topup_sega",
        _py(
            "issue2378_gen.py",
            "--phase",
            "sega",
            "--cells",
            args.cells,
            "--sega-attempts-per-cell",
            str(args.sega_attempts_per_cell),
            "--wave",
            "2",
            "--skip-upload",
        ),
        gpus=visible_gpus(),
    )
    for st in ("sega", "sega_mined"):
        runner.run(
            f"p4.topup_upload_{st}",
            _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", st),
        )
    if not runner.dry:
        write_sentinel(
            args, "epm:progress", {"phase": "p4_topup", "status": "complete", "cells": args.cells}
        )
    return 0


def _capture_cell_shards(cells: list[str], n: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, c in enumerate(cells):
        shards[i % n].append(c)
    return [s for s in shards if s]


def phase_p4(args, runner: Runner) -> int:
    _phase_line("p4_segb_capture")
    raw_root, ledger_root = Path(args.raw_root), Path(args.ledger_root)
    assert_headroom("p4_segb_capture", raw_root)
    gpus = visible_gpus()
    kept_dir = ledger_root / "kept"
    stage_flags = ["--stage-raw-from-hf", "--stage-pools-from-hf"]

    def run_segb(wave: int, kdir: Path, target: int, cells: str = "") -> None:
        argv = _py(
            "issue2378_gen.py",
            "--phase",
            "segb",
            "--kept-dir",
            str(kdir),
            "--target-kept-per-cell",
            str(target),
            "--wave",
            str(wave),
            "--skip-upload",
            "--stage-raw-from-hf",
        )
        if cells:
            argv += ["--cells", cells]
        runner.fanout(f"p4.segb.w{wave}", argv, gpus=gpus)
        runner.run(
            f"p4.upload_segb.w{wave}",
            _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", "segb"),
        )

    run_segb(1, kept_dir, args.target_kept_per_cell)
    runner.run(
        "p4.user_real_render",
        _py("issue2378_gen.py", "--phase", "user_real_render", *stage_flags),
    )
    runner.run(
        "p4.capture_ready.w1",
        _py("issue2378_gen.py", "--phase", "capture_ready", "--stage-raw-from-hf"),
    )

    # SegB-stage retry waves from the admitted surplus (plan §7 G2b): <= 2
    # retry waves + ONE close-miss escalation, deterministic selection extras.
    consumed = {c: args.target_kept_per_cell for c in cm.STORY_CELLS}
    escalated: list[str] = []
    waves_used = 0
    escalation_wave_used = False
    if not runner.dry:
        digest_path = ledger_root / "pilot" / "pilot_digest.json"
        surv_by_fam = {}
        if digest_path.exists():
            dg = json.loads(digest_path.read_text(encoding="utf-8"))
            surv_by_fam = {
                fam: s.get("rate", 0.9) for fam, s in dg["per_stage"]["segb_survival"].items()
            }
        for wave in (2, 3, 4):  # 2,3 = retry budget; 4 = close-miss escalation only
            ready = {
                p.stem: json.loads(p.read_text(encoding="utf-8"))
                for p in (ledger_root / "capture_ready").glob("*.json")
            }
            short = {c: v for c, v in ready.items() if c in cm.STORY_CELLS and not v["floor_pass"]}
            if not short:
                break
            if wave == 4:
                short = {c: v for c, v in short.items() if v.get("close_miss_band")}
                if not short:
                    break
                escalated = sorted(short)
                _log(f"[g2b] close-miss escalation wave (>= {cm.CLOSE_MISS_FLOOR}): {escalated}")
            retry_dir = ledger_root / f"retry_kept_w{wave}"
            cells_run = []
            for cell, v in sorted(short.items()):
                surv = max(0.05, surv_by_fam.get(cm.CELL_FAMILY[cell], 0.9))
                n_extra = math.ceil((cm.FLOOR_KEPT - v["n_kept"]) / surv)
                extras = segb_extras(kept_dir, cell, consumed[cell], n_extra)
                if not extras:
                    _log(f"[g2b] {cell}: admitted surplus exhausted — drop path (reported)")
                    continue
                write_retry_kept(kept_dir, retry_dir, cell, extras)
                consumed[cell] += len(extras)
                cells_run.append(cell)
            if not cells_run:
                break
            if wave == 4:
                escalation_wave_used = True
            else:
                waves_used += 1
            run_segb(wave, retry_dir, 10**6, cells=",".join(cells_run))
            runner.run(
                f"p4.capture_ready.w{wave}",
                _py("issue2378_gen.py", "--phase", "capture_ready", "--stage-raw-from-hf"),
            )
        g2b = evaluate_g2b(
            ledger_root,
            waves_used=waves_used,
            escalated=escalated,
            escalation_wave_used=escalation_wave_used,
        )
        write_sentinel(
            args, "epm:progress", g2b, gate="g2b", blocks_pipeline=g2b["verdict"] != "PASS"
        )
        git_harvest(
            [
                "eval_results/issue_2378/capture_ready/*.json",
                "eval_results/issue_2378/g2b_report.json",
            ],
            f"task #{ISSUE}: P4 capture_ready ledgers + G2b report",
        )
        if g2b["verdict"] != "PASS":
            _log("[g2b] survivor predicate FAIL — partial-result stop (plan Kill criteria)")
            return RC_G2B_PARTIAL
        survivors = g2b["survivors"]
    else:
        _log("[dry] p4.g2b: evaluate_g2b on capture_ready ledgers (binding floor 6500)")
        survivors = list(cm.ALL_CELLS)

    lstar = 32 if runner.dry else resolve_lstar(ledger_root)
    layers = parse_layers_spec(args.layers, lstar)
    out_root = Path(args.store_root)
    # dry mode runs on the GPU-less VM: log with a placeholder CVD pin; a real
    # pod with zero visible GPUs still fails loud inside Runner.parallel.
    cvd_pins = gpus if gpus else (["0"] if runner.dry else [])
    # parallel capture fan-out (one HF model per GPU, cells sharded via --cells)
    capture_argvs = [
        _py(
            "issue2378_capture.py",
            "--phase",
            "capture",
            "--cells",
            ",".join(shard_cells),
            "--layers",
            ",".join(str(x) for x in layers),
            "--out-root",
            str(out_root),
            "--stage-raw-from-hf",
            "--stage-pools-from-hf",
            "--skip-upload",
        )
        for shard_cells in _capture_cell_shards(survivors, max(1, len(cvd_pins)))
    ]
    runner.parallel("p4.capture", capture_argvs, gpus=cvd_pins)
    # r1 review codex blocker fresh-draw-producer-undispatched: generate the
    # fresh SegB/answer draws (seeds 138-141) BEFORE capture_fresh consumes
    # them (capture.py reads gen._rows_dir(args, "fresh_draws")); the user arm
    # gets its fresh draws from P2 user_fresh.
    fresh_cells = [c for c in survivors if c not in ("chat_user_real", "chat_user_sim")]
    if fresh_cells:
        runner.fanout(
            "p4.fresh_draws",
            _py(
                "issue2378_gen.py",
                "--phase",
                "fresh_draws",
                "--cells",
                ",".join(fresh_cells),
                "--fresh-rows",
                str(args.fresh_rows),
                "--fresh-draws",
                str(args.fresh_draws),
                "--skip-upload",
                "--stage-raw-from-hf",
            ),
            gpus=gpus,
        )
        runner.run(
            "p4.upload_fresh_draws",
            _py("issue2378_gen.py", "--phase", "upload_stage", "--stage", "fresh_draws"),
        )
    fresh_argvs = [
        _py(
            "issue2378_capture.py",
            "--phase",
            "capture_fresh",
            "--cells",
            ",".join(shard_cells),
            "--layers",
            str(lstar),
            "--out-root",
            str(out_root),
            "--fresh-draws",
            str(args.fresh_draws),
            "--stage-raw-from-hf",
            "--stage-pools-from-hf",
            "--skip-upload",
        )
        for shard_cells in _capture_cell_shards(
            [c for c in survivors if c != "chat_user_real"], max(1, len(cvd_pins))
        )
    ]
    runner.parallel("p4.capture_fresh", fresh_argvs, gpus=cvd_pins)
    if not runner.dry:
        import issue2378_p6_common as p6

        idx = p6.build_store_index(out_root, [c for c in cm.ALL_CELLS if c in survivors])
        cm.atomic_write_json(
            out_root / "store_index.json", {"cells": idx, "lstar": lstar, "layers": layers}
        )
        cm.upload_stage_dir(out_root, ACTIVATIONS_PREFIX)  # BEFORE termination (plan §9)
        write_sentinel(
            args,
            "epm:progress",
            {
                "phase": "p4_segb_capture",
                "status": "complete",
                "lstar": lstar,
                "layers": layers,
                "survivors": survivors,
                "walls_s": {k: round(v, 1) for k, v in runner.walls.items()},
            },
        )
    return 0


def phase_p5(args, runner: Runner) -> int:
    _phase_line("p5_congruence")
    assert_headroom("p5_congruence", cm.REPO_ROOT / "data" / "issue_2378")
    for transport in ("sync", "batch"):
        rc = runner.run(
            f"p5.pilot_{transport}",
            _py(
                "issue2378_judge.py",
                "--wave",
                "congruence",
                "--pilot",
                "200",
                "--transport",
                transport,
                "--stage-from-hf",
            ),
            ok_rcs=(0, RC_JUDGE_PILOT_FAIL),
        )
        if rc == RC_JUDGE_PILOT_FAIL:
            return RC_JUDGE_PILOT_FAIL
    runner.run(
        "p5.congruence_wave",
        _py(
            "issue2378_judge.py",
            "--wave",
            "congruence",
            "--transport",
            "batch",
            "--congruence-rows",
            str(args.congruence_rows),
            "--stage-from-hf",
        ),
    )
    if not runner.dry:
        git_harvest(
            [
                "eval_results/issue_2378/judge/congruence/*.json",
                "eval_results/issue_2378/judge/pilot_congruence_sync.json",
                "eval_results/issue_2378/judge/pilot_congruence_batch.json",
            ],
            f"task #{ISSUE}: P5 congruence manipulation check",
        )
    return 0


# ---------------------------------------------------------------------------
# P6 (per fits pod)
# ---------------------------------------------------------------------------


def stage_p6(args, role: str, lstar: int) -> Path:
    """Jittered, scoped, shard-only staging of the activation store (plan §9).

    Every pod stages ALL cells' rows.json (KB-scale, fold-map identity) +
    store_index.json; npz only for its shard cells (fits-d: ALL cells — the
    pooled arm consumes every cell's train folds). Count-asserted BEFORE any
    fit; a short-staged pod fails loud pre-fit, never mid-battery.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    stage_root = Path(args.stage_root)
    store_root = stage_root / ACTIVATIONS_PREFIX
    jitter = 0 if args.no_jitter else POD_ROLE_JITTER_S[role]
    if jitter:
        _log(f"[stage] jitter sleep {jitter}s (staggered same-prefix pulls, plan §9)")
        time.sleep(jitter)
    api = HfApi()
    # canonical retried scoped listing (r1 review codex hub-waiver blocker:
    # bare list_repo_tree in scripts/ fails workflow_lint --check-hub-verify-retry)
    entries = hub.list_hf_entries_under_path(
        api, cm.HF_DATA_REPO, ACTIVATIONS_PREFIX, repo_type="dataset"
    )
    sizes: dict[str, int | None] = dict(entries)
    my_cells = set(POD_ROLE_CELLS[role])
    want_all_npz = role == "fits-d"
    wanted: list[str] = []
    for path, _size in entries:
        name = path.rsplit("/", 1)[-1]
        if name.endswith("__rows.json") or name == "store_index.json":
            wanted.append(path)
            continue
        if not name.endswith(f"__L{lstar}.npz"):
            continue
        cell = name.split("__part")[0].split("__fresh_d")[0]
        if want_all_npz or cell in my_cells:
            wanted.append(path)
    for path in wanted:
        target = stage_root / path
        want_size = sizes.get(path)
        # size-checked resume skip (r1 review g5 minor: a bare exists() skip
        # adopts a truncated partial download from a killed prior stage)
        if target.exists() and (want_size is None or target.stat().st_size == int(want_size)):
            continue
        hub.retry_transient(
            lambda p=path: hf_hub_download(
                cm.HF_DATA_REPO, p, repo_type="dataset", local_dir=str(stage_root)
            ),
            what=f"download {path}",
        )
    idx = json.loads((store_root / "store_index.json").read_text(encoding="utf-8"))
    expected = 0
    for cell, meta in idx["cells"].items():
        if not (want_all_npz or cell in my_cells):
            continue
        for part in meta["parts"]:
            npz = store_root / f"{part}__L{lstar}.npz"
            if not npz.exists():
                raise RuntimeError(f"[stage] SHORT-STAGED: missing {npz} — failing pre-fit")
            expected += 1
    n_staged = len([p for p in store_root.glob(f"*__L{lstar}.npz")])
    _log(f"[stage] activations staged: {n_staged} files (expected production >= {expected})")
    if n_staged < expected:
        raise RuntimeError(f"[stage] staged {n_staged} < expected {expected} — failing pre-fit")
    return store_root


def _p6_sibling_expect(role: str, survivors: list[str]) -> list[str]:
    """Join-wait path set for the fits-d merge: sibling p6_digest_<r>.json
    files + G2b-survivor fits/<c>__context.json — ALWAYS-written artifacts
    only. Each sibling digest is harvested AFTER that pod's fits + ladder +
    retrieval complete, so waiting on it subsumes ladder completion (r1 review
    g5 blocker 3: an Unmappable target never writes chat_to_<c>__rung9.json —
    a rung9 wait would deadlock the merge)."""
    expect: list[str] = []
    for r, cells in POD_ROLE_CELLS.items():
        if r == role:
            continue
        expect.append(f"eval_results/issue_2378/p6_digest_{r}.json")
        for c in cells:
            if c in survivors:
                expect.append(f"eval_results/issue_2378/fits/{c}__context.json")
    return expect


def _p6_units_for(role: str, survivors: list[str]) -> str:
    """Fit units for this pod role, intersected with the G2b survivor set
    (r1 review codex blocker g2b-survivors-not-threaded-to-p6)."""
    cells = [c for c in POD_ROLE_CELLS[role] if c in survivors]
    if role == "fits-a":
        # chat/context is produced by --phase g3 (the 1-cell pilot); avoid redoing it
        units = ["own:chat:prefix"] + [c for c in cells if c != "chat"]
        return ",".join(units)
    return ",".join(cells)


def _p6_targets_for(role: str, survivors: list[str]) -> list[str]:
    return [c for c in POD_ROLE_CELLS[role] if c != "chat" and c in survivors]


def _upload_sidecars(ledger_root: Path, sub: str) -> None:
    d = ledger_root / sub
    if d.exists() and any(d.iterdir()):
        cm.upload_stage_dir(d, f"{P6_SIDECAR_PREFIX}/{sub}")


def _stage_sidecars(ledger_root: Path) -> None:
    """fits-d: stage the sibling pods' small rowstats sidecars from HF."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    # canonical retried scoped listing (r1 review codex hub-waiver blocker)
    entries = hub.list_hf_files_under_path(
        api, cm.HF_DATA_REPO, P6_SIDECAR_PREFIX, repo_type="dataset"
    )
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for path in entries:
            rel = path[len(P6_SIDECAR_PREFIX) + 1 :]
            target = ledger_root / rel
            if target.exists():
                continue
            got = hub.retry_transient(
                lambda p=path: hf_hub_download(
                    cm.HF_DATA_REPO, p, repo_type="dataset", local_dir=td
                ),
                what=f"download {path}",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, target)
    _log(f"[stage] p6 sidecars staged: {len(entries)} files")


def phase_p6(args, runner: Runner) -> int:
    _phase_line("p6_fits")
    role = args.pod_role
    if role not in POD_ROLE_CELLS:
        raise SystemExit(f"--pod-role must be one of {sorted(POD_ROLE_CELLS)}")
    ledger_root = Path(args.ledger_root)
    assert_headroom("p6_fits", Path(args.stage_root))
    if runner.dry:
        _log(
            f"[dry] p6.stage: jittered scoped staging (role={role}, "
            f"cells={POD_ROLE_CELLS[role]}, full_npz={role == 'fits-d'})"
        )
        lstar = 32
        store_root = Path(args.stage_root) / ACTIVATIONS_PREFIX
        # Dry composition placeholder (plan §4.6 blind-spot enumeration): the
        # real branch intersects with g2b_report.json survivors.
        survivors = list(cm.ALL_CELLS)
    else:
        _git_pull_rebase()  # layer_sweep + kept + capture_ready + g2b arrive via git
        g2b = json.loads((ledger_root / "g2b_report.json").read_text(encoding="utf-8"))
        # Order-stable intersection with the G2b survivor set (r1 review codex
        # blocker g2b-survivors-not-threaded-to-p6).
        survivors = [c for c in cm.ALL_CELLS if c in g2b["survivors"]]
        lstar = resolve_lstar(ledger_root)
        store_root = stage_p6(args, role, lstar)
    role_survivors = [c for c in POD_ROLE_CELLS[role] if c in survivors]
    role_dropped = [c for c in POD_ROLE_CELLS[role] if c not in survivors]
    store = ["--store-root", str(store_root)]
    gate_path = "eval_results/issue_2378/g3_gate.json"

    if role == "fits-a":
        rc = runner.run(
            "p6.g3",
            _py("issue2378_fits.py", "--phase", "g3", *store),
            ok_rcs=(0, RC_G3_REFUSED),
        )
        if not runner.dry:
            git_harvest(
                [
                    "eval_results/issue_2378/g3_gate.json",
                    "eval_results/issue_2378/fits/chat__context.json",
                    "eval_results/issue_2378/fold_map.json",
                ],
                f"task #{ISSUE}: P6 G3 gate ({'PASS' if rc == 0 else 'REFUSED'})",
            )
            g3 = json.loads((ledger_root / "g3_gate.json").read_text(encoding="utf-8"))
            write_sentinel(args, "epm:progress", g3, gate="g3", blocks_pipeline=rc != 0)
            if rc != 0:
                _log("[g3] REFUSED — ladder fan-out aborts (plan §7 G3)")
                return RC_G3_REFUSED
    elif not runner.dry:
        _git_wait_for(
            [gate_path], poll_s=args.g3_poll_s, timeout_s=args.g3_timeout_s, what="G3 gate"
        )
        _git_pull_rebase()
        import issue2378_p6_common as p6

        p6.require_g3_pass(ledger_root / "g3_gate.json")  # exits loud on REFUSED

    if role_dropped and not runner.dry:
        # N/A markers for G2b-dropped non-binding cells: downstream readers see
        # a counted drop, never a silent absence (plan §7 skip-and-count).
        for c in role_dropped:
            cm.atomic_write_json(
                ledger_root / "fits" / f"{c}__g2b_dropped.json",
                {
                    "cell": c,
                    "status": "N/A",
                    "reason": "G2b dropped this non-binding cell (below floor) — plan §7",
                    "metadata": cm.run_metadata(),
                },
            )
    if role_survivors:
        runner.run(
            "p6.fits",
            _py(
                "issue2378_fits.py",
                "--phase",
                "fit",
                "--units",
                _p6_units_for(role, survivors),
                *store,
            ),
        )
    else:
        _log(f"[p6] role={role}: every cell G2b-dropped — fits/ladder/retrieval skipped")
    if not runner.dry:
        _upload_sidecars(ledger_root, "fits/percell")
        git_harvest(
            ["eval_results/issue_2378/fits/*.json"],
            f"task #{ISSUE}: P6 fits JSONs ({role})",
        )
    targets = _p6_targets_for(role, survivors)
    if targets:
        runner.run(
            "p6.ladder",
            _py("issue2378_ladder.py", "--phase", "pairs", "--pairs", ",".join(targets), *store),
        )
        if not runner.dry:
            _upload_sidecars(ledger_root, "ladder/percell")
            git_harvest(
                ["eval_results/issue_2378/ladder/*.json"],
                f"task #{ISSUE}: P6 ladder JSONs ({role})",
            )
    if role_survivors:
        runner.run(
            "p6.retrieval",
            _py(
                "issue2378_retrieval.py",
                "--phase",
                "all",
                "--cells",
                ",".join(role_survivors),
                *store,
            ),
        )
        if not runner.dry:
            git_harvest(
                ["eval_results/issue_2378/retrieval/*.json"],
                f"task #{ISSUE}: P6 retrieval JSONs ({role})",
            )
    if not runner.dry:
        digest = {
            "role": role,
            "cells": POD_ROLE_CELLS[role],
            "g2b_survivors": role_survivors,
            "g2b_dropped": role_dropped,
            "walls_s": {k: round(v, 1) for k, v in runner.walls.items()},
            "metadata": cm.run_metadata(),
        }
        cm.atomic_write_json(ledger_root / f"p6_digest_{role}.json", digest)
        git_harvest(
            [f"eval_results/issue_2378/p6_digest_{role}.json"],
            f"task #{ISSUE}: P6 shard digest ({role})",
        )

    if role == "fits-d":
        if runner.dry:
            _log(
                "[dry] p6.summaries: wait siblings -> stage sidecars -> pool/h5/h3/h4b/ratio "
                "-> p6_merge_digest.json"
            )
        else:
            _git_wait_for(
                _p6_sibling_expect(role, survivors),
                poll_s=args.g3_poll_s,
                timeout_s=args.siblings_timeout_s,
                what="sibling P6 shards",
            )
            _git_pull_rebase()
            _stage_sidecars(ledger_root)
        surv_arg = ",".join(survivors)
        runner.run(
            "p6.pool", _py("issue2378_pool.py", "--phase", "pool", "--cells", surv_arg, *store)
        )
        runner.run("p6.h5", _py("issue2378_pool.py", "--phase", "h5", "--cells", surv_arg, *store))
        runner.run("p6.h3", _py("issue2378_ladder.py", "--phase", "h3", *store))
        user_cells = ("chat_user_real", "chat_user_sim")
        if all(c in survivors for c in user_cells):
            runner.run("p6.h4b", _py("issue2378_ladder.py", "--phase", "h4b", *store))
        elif not runner.dry:
            # H4b is a paired real-vs-sim contrast: with either user arm
            # G2b-dropped the pairing is unformable — write the N/A verdict the
            # explicit-path harvest below expects (plan §7 skip-and-count).
            cm.atomic_write_json(
                ledger_root / "ladder" / "h4b_real_vs_sim.json",
                {
                    "status": "N/A",
                    "reason": (
                        "user arm(s) G2b-dropped: "
                        f"{[c for c in user_cells if c not in survivors]} — "
                        "H4b needs both arms (plan §3)"
                    ),
                    "metadata": cm.run_metadata(),
                },
            )
        # H4a ratio is user-drop-safe (r2 reconciler blocker
        # g2b-user-drop-crashes-h4a-ratio): fits-d writes the per-cell
        # __g2b_dropped.json markers above BEFORE this call, and phase_ratio
        # reads them as the survivor manifest — a dropped user arm yields a
        # loud per-arm N/A entry (whole-file N/A when both drop), while a
        # missing fit for a SURVIVOR still hard-raises.
        runner.run("p6.ratio", _py("issue2378_fits.py", "--phase", "ratio", *store))
        if not runner.dry:
            merged = {"roles": {}, "metadata": cm.run_metadata()}
            for r in POD_ROLE_CELLS:
                p = ledger_root / f"p6_digest_{r}.json"
                if p.exists():
                    merged["roles"][r] = json.loads(p.read_text(encoding="utf-8"))
            cm.atomic_write_json(ledger_root / "p6_merge_digest.json", merged)
            git_harvest(
                [
                    "eval_results/issue_2378/pool/*.json",
                    "eval_results/issue_2378/ladder/h3_question_vs_dialogue.json",
                    "eval_results/issue_2378/ladder/h4b_real_vs_sim.json",
                    "eval_results/issue_2378/fits/ratio/h4a_ceiling_ratio.json",
                    "eval_results/issue_2378/p6_merge_digest.json",
                ],
                f"task #{ISSUE}: P6 summaries + merge digest (fits-d)",
            )
            write_sentinel(
                args,
                "epm:results",
                {
                    "phase": "P6 complete (fits-d merge)",
                    "eval_json_paths": [
                        "eval_results/issue_2378/fits/",
                        "eval_results/issue_2378/ladder/",
                        "eval_results/issue_2378/pool/",
                        "eval_results/issue_2378/retrieval/",
                        "eval_results/issue_2378/p6_merge_digest.json",
                    ],
                    "note": "no training in this task — no reproducibility_card adapters",
                },
            )
    elif not runner.dry:
        write_sentinel(
            args, "epm:progress", {"phase": "p6_fits", "role": role, "status": "complete"}
        )
    return 0


# ---------------------------------------------------------------------------
# probe (CPU fixtures; no GPU, no network) + import-check
# ---------------------------------------------------------------------------


def phase_probe(args) -> int:  # noqa: C901 — linear fixture script
    _phase_line("probe")
    import tempfile

    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            _log(f"[probe] {name}: OK")
        except Exception as e:  # noqa: BLE001 — collected + re-raised at end
            failures.append(f"{name}: {e}")
            _log(f"[probe] {name}: FAIL — {e}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def t_sentinel():
            import poll_pipeline as pp

            class A:
                sentinel_dir = str(tmp / "sent")

            path = write_sentinel(A(), "epm:smoke-result", {"probe": True}, gate="g1")
            data = json.loads(path.read_text(encoding="utf-8"))
            missing = [k for k in pp._SENTINEL_REQUIRED_KEYS if k not in data]
            assert not missing, f"sentinel missing required keys {missing}"
            m = pp.PHASE_RE.search("[phase=done]")
            assert m and m.group(1) == "done", "PHASE_RE failed to parse [phase=done]"

        check("sentinel schema + PHASE_RE", t_sentinel)

        def _mk_pilot_fixture(root: Path, mining_kept: int) -> tuple[Path, Path]:
            raw = root / "raw"
            ledger = root / "ledger"
            for cell in ("storyq_astra", "dialog_astra"):
                (raw / "sega").mkdir(parents=True, exist_ok=True)
                cm.atomic_write_json(
                    raw / "sega" / f"summary_{cell}_w1_s0.json",
                    {"cell": cell, "counts": {"attempts": 100, "kept": mining_kept, "cap_hit": 0}},
                )
                (raw / "segb").mkdir(parents=True, exist_ok=True)
                cm.atomic_write_json(
                    raw / "segb" / f"summary_{cell}_w1_s0.json",
                    {"cell": cell, "counts": {"rows": 50, "kept": 45, "cap_hit_no_close": 2}},
                )
                (ledger / "pilot" / "kept").mkdir(parents=True, exist_ok=True)
                cm.atomic_write_json(
                    ledger / "pilot" / "kept" / f"{cell}.json",
                    {
                        "cell": cell,
                        "family": cm.CELL_FAMILY[cell],
                        "n_items": 60,
                        "n_admitted": 40,
                        "admitted": [],
                    },
                )
            (raw / "user_sim").mkdir(parents=True, exist_ok=True)
            cm.atomic_write_json(
                raw / "user_sim" / "summary_w1_s0.json",
                {"counts": {"rows": 50, "kept": 47, "degenerate": 3}},
            )
            (ledger / "judge").mkdir(parents=True, exist_ok=True)
            cm.atomic_write_json(
                ledger / "judge" / "pilot_admission_sync.json",
                {"verdict": "PASS", "tally": {}},
            )
            cm.atomic_write_json(
                ledger / "pilot" / "layer_sweep.json",
                # production shape (issue2378_capture.py layer_sweep writer)
                {
                    "selected_layer": 40,
                    "gate_g1c": {"threshold": 0.05, "max_r2": 0.31, "passes": True},
                },
            )
            return raw, ledger

        def t_g1_pass():
            raw, ledger = _mk_pilot_fixture(tmp / "g1pass", mining_kept=60)
            d = compose_pilot_digest(
                raw, ledger, {"p1.sega": 100.0}, pilot_round=1, attempts_per_cell=300
            )
            assert d["verdict"] == "PASS", d["fail_reasons"]
            # net = 0.6 * (40/60) * 0.9 = 0.36 >= 0.25; sizing = ceil(8000*1.25/0.36)
            fam = d["families"]["question"]
            assert abs(fam["net_kept_per_attempt"] - 0.6 * (40 / 60) * 0.9) < 1e-9
            assert fam["wave1_attempts_per_cell"] == math.ceil(
                8000 * 1.25 / fam["net_kept_per_attempt"]
            )
            assert d["fences_s_2x"]["p1.sega"] == 200.0

        check("G1 composer PASS branch + wave sizing", t_g1_pass)

        def t_g1_trip():
            raw, ledger = _mk_pilot_fixture(tmp / "g1trip", mining_kept=20)
            d = compose_pilot_digest(raw, ledger, {}, pilot_round=1, attempts_per_cell=300)
            assert d["verdict"] == "FAIL"
            assert all(r.startswith("G1(a)") for r in d["fail_reasons"])
            # trip-line sizing caps at 30k
            assert d["families"]["question"]["wave1_attempts_per_cell"] <= SEGA_ATTEMPTS_CAP

        check("G1 composer TRIP branch (recalibration route)", t_g1_trip)

        def t_g2a():
            root = tmp / "g2a"
            raw, ledger = _mk_pilot_fixture(root, mining_kept=60)
            compose_pilot_digest(raw, ledger, {}, pilot_round=1, attempts_per_cell=300)
            (ledger / "kept").mkdir(parents=True, exist_ok=True)
            cm.atomic_write_json(
                ledger / "kept" / "storyq_astra.json",
                {
                    "cell": "storyq_astra",
                    "family": "question",
                    "n_items": 20000,
                    "n_admitted": 5000,
                    "admitted": [],
                },
            )
            out = compose_g2a(ledger)
            assert out["sega_topup_cells"] == ["storyq_astra"]  # 5000*0.9 < 6500

        check("G2a projection + topup scheduling", t_g2a)

        def t_g2b():
            ledger = tmp / "g2b" / "ledger"
            ready = ledger / "capture_ready"
            ready.mkdir(parents=True, exist_ok=True)
            kept_by_cell = {c: 8000 for c in cm.ALL_CELLS}
            kept_by_cell["storyq_vex"] = 5900  # close-miss band
            kept_by_cell["chat_user_real"] = 100  # user drop: NON-binding
            for cell, n in kept_by_cell.items():
                cm.atomic_write_json(
                    ready / f"{cell}.json",
                    {
                        "cell": cell,
                        "n_kept": n,
                        "floor": cm.FLOOR_KEPT,
                        "floor_pass": n >= cm.FLOOR_KEPT,
                        "close_miss_band": cm.CLOSE_MISS_FLOOR <= n < cm.FLOOR_KEPT,
                        "drop_counts": {},
                        "kept_ids": [],
                    },
                )
            out = evaluate_g2b(ledger, waves_used=2, escalated=["storyq_vex"])
            assert out["verdict"] == "PASS"  # 4 storyQ + 4 dialog survive
            assert out["user_cell_drops_nonbinding"] == ["chat_user_real"]
            assert "storyq_vex" in out["dropped_cells"]
            # binding FAIL branch: kill chat
            cm.atomic_write_json(
                ready / "chat.json",
                {
                    "cell": "chat",
                    "n_kept": 100,
                    "floor": cm.FLOOR_KEPT,
                    "floor_pass": False,
                    "close_miss_band": False,
                    "drop_counts": {},
                    "kept_ids": [],
                },
            )
            out2 = evaluate_g2b(ledger, waves_used=2, escalated=[])
            assert out2["verdict"] == "FAIL" and "chat" in out2["binding_drops"]

        check("G2b floor/close-miss/user-nonbinding/survivor branches", t_g2b)

        def t_extras():
            import issue2378_gen as gen

            kept_dir = tmp / "kept"
            kept_dir.mkdir(parents=True, exist_ok=True)
            ids = [f"storyq_astra_r{i:03d}" for i in range(20)]
            cm.atomic_write_json(
                kept_dir / "storyq_astra.json",
                {
                    "cell": "storyq_astra",
                    "family": "question",
                    "n_items": 20,
                    "n_admitted": 20,
                    "admitted": [{"row_id": r, "score": 60} for r in ids],
                },
            )
            kept_ids = gen._load_kept_ids(kept_dir, "storyq_astra")
            order = random.Random(cm.derived_seed(cm.SEED, "segb_select", "storyq_astra")).sample(
                range(len(kept_ids)), len(kept_ids)
            )
            expected = [kept_ids[i] for i in order[8:13]]
            got = segb_extras(kept_dir, "storyq_astra", 8, 5)
            assert got == expected, "retry extras diverge from gen.phase_segb selection order"
            write_retry_kept(kept_dir, tmp / "retry", "storyq_astra", got)
            again = gen._load_kept_ids(tmp / "retry", "storyq_astra")
            assert again == got

        check("segb retry-extras == gen selection order (wave-2 disjointness)", t_extras)

        def t_layers():
            assert parse_layers_spec("Lstar,Lstar-8,Lstar-4,Lstar+4,Lstar+8", 40) == [
                32,
                36,
                40,
                44,
                48,
            ]
            assert parse_layers_spec("Lstar+8,Lstar", 60) == [60, 63]  # clamp + dedupe at 63
            assert parse_layers_spec("5", 40) == [5]

        check("layers spec parser (clamp/dedupe/symbolic)", t_layers)

        def t_resume_and_fanout():
            r = Runner(tmp / "logs", resume=True)
            argv = [sys.executable, "-c", "print('ok')"]
            r.run("probe.step", argv)
            t_first = (tmp / "logs" / "probe.step.ok").stat().st_mtime
            r.run("probe.step", argv)  # skipped: ok-flag + same argv sha
            assert (tmp / "logs" / "probe.step.ok").stat().st_mtime == t_first
            r.fanout(
                "probe.fan",
                [
                    sys.executable,
                    "-c",
                    "import os,sys;print('CVD='+os.environ['CUDA_VISIBLE_DEVICES'])",
                ],
                gpus=["6", "7"],
            )
            for i, g in enumerate(["6", "7"]):
                log = (tmp / "logs" / f"probe.fan.s{i}.log").read_text(encoding="utf-8")
                assert f"CVD={g}" in log, f"shard {i} CVD pin missing"
            # parallel(): pre-composed per-shard argvs (the P4 capture shape)
            probe_argv = [
                sys.executable,
                "-c",
                "import os;print('CVD='+os.environ['CUDA_VISIBLE_DEVICES'])",
            ]
            r.parallel("probe.par", [list(probe_argv), list(probe_argv)], gpus=["2", "3"])
            for i, g in enumerate(["2", "3"]):
                log = (tmp / "logs" / f"probe.par.s{i}.log").read_text(encoding="utf-8")
                assert f"CVD={g}" in log, f"parallel shard {i} CVD pin missing"
            try:
                Runner(tmp / "logs2", resume=True).parallel("probe.par0", [probe_argv], gpus=[])
                raise AssertionError("parallel with zero GPUs must raise")
            except RuntimeError as e:
                assert "no visible GPUs" in str(e)

        check("Runner ok-flag resume + fan-out/parallel CVD pins", t_resume_and_fanout)

        def t_slice():
            src = tmp / "mined"
            src.mkdir(parents=True, exist_ok=True)
            with (src / "part.jsonl").open("w", encoding="utf-8") as fh:
                for cell in ("storyq_astra", "storyq_vex", "dialog_dana"):
                    for i in range(30):
                        fh.write(
                            json.dumps(
                                {
                                    "row_id": f"{cell}_r{i:03d}",
                                    "cell": cell,
                                    "family": cm.CELL_FAMILY[cell],
                                    "character": "X",
                                    "scene_pre_answer": "s",
                                    "utterance": "u",
                                }
                            )
                            + "\n"
                        )
            n = balanced_mined_slice(src, tmp / "slice", 40)
            rows = [
                json.loads(ln)
                for ln in (tmp / "slice" / "slice.jsonl").read_text().split("\n")
                if ln.strip()
            ]
            fams = {r["family"] for r in rows}
            assert n == 40 and fams == {"question", "dialogue"}

        check("balanced admission slice (both families covered)", t_slice)

        def t_pilot_round_scope():
            base_raw = tmp / "praw"
            r1 = Runner(tmp / "plogs" / "p1_pilot", resume=True, dry=True)
            raw1, run1, pref1 = _pilot_round_scope(base_raw, r1, 1)
            assert raw1 == base_raw and run1 is r1
            assert pref1 == f"{cm.HF_PREFIX}/raw_completions/pilot"
            raw2, run2, pref2 = _pilot_round_scope(base_raw, r1, 2)
            assert raw2 == base_raw / "r2"
            assert run2.logs_dir == r1.logs_dir / "p1_pilot_r2"
            assert pref2 == f"{cm.HF_PREFIX}/raw_completions/pilot/r2"
            # round-2 resume keys are DISJOINT from round 1's: a round-1
            # ok-flag must be invisible to the round-2 Runner (g5 blocker 2)
            run1._ok_path("p1.sega").write_text("sha")
            assert not run2._ok_path("p1.sega").exists()

        check("pilot round-2 scope (fresh resume key + raw root + HF prefix)", t_pilot_round_scope)

        def t_p6_survivor_threading():
            # g5 blocker 3 + codex g2b-survivors blocker: the fits-d join-wait
            # expects ONLY always-written artifacts (sibling digests + survivor
            # fits context JSONs — never per-rung ladder files an Unmappable
            # verdict suppresses), and units/targets/retrieval drop
            # non-survivors.
            surv = [c for c in cm.ALL_CELLS if c != "storyq_vex"]
            expect = _p6_sibling_expect("fits-d", surv)
            assert not any("rung9" in p or "/ladder/" in p for p in expect), expect
            for r in POD_ROLE_CELLS:
                if r != "fits-d":
                    assert f"eval_results/issue_2378/p6_digest_{r}.json" in expect
            assert "eval_results/issue_2378/fits/storyq_vex__context.json" not in expect
            assert "eval_results/issue_2378/fits/chat__context.json" in expect
            assert not any("chat_user" in p for p in expect)  # own-role never waited on
            assert "storyq_vex" not in _p6_units_for("fits-b", surv)
            assert "storyq_vex" not in _p6_targets_for("fits-b", surv)
            assert _p6_targets_for("fits-a", surv) == ["plain_text", "storyq_astra"]
            assert _p6_units_for("fits-a", surv).startswith("own:chat:prefix,")
            # user-drop survivor sets (r2 reconciler blocker
            # g2b-user-drop-crashes-h4a-ratio): real-only / sim-only / neither
            # — fits-d units drop the dropped arm(s), and no sibling join-wait
            # ever carries a user-cell context path (own-role cells).
            for gone in (["chat_user_sim"], ["chat_user_real"], list(cm.USER_CELLS)):
                surv_u = [c for c in cm.ALL_CELLS if c not in gone]
                units_d = _p6_units_for("fits-d", surv_u)
                assert all(c not in units_d for c in gone), (gone, units_d)
                for c in cm.USER_CELLS:
                    if c in surv_u:
                        assert c in units_d, (gone, units_d)
                assert not any("chat_user" in p for p in _p6_sibling_expect("fits-d", surv_u))
            assert (
                _p6_units_for("fits-d", [c for c in cm.ALL_CELLS if c not in cm.USER_CELLS]) == ""
            )

        check("P6 survivor threading + fits-d join-wait satisfiability", t_p6_survivor_threading)

        def t_subprobes():
            for script, phase in (
                ("issue2378_gen.py", "probe_miner"),
                ("issue2378_capture.py", "probe_gating"),
            ):
                p = subprocess.run(
                    _py(script, "--phase", phase),
                    cwd=str(cm.REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert p.returncode == 0, (
                    f"{script} --phase {phase} rc={p.returncode}\n{p.stdout[-800:]}"
                )

        check("gen probe_miner + capture probe_gating subprobes", t_subprobes)

        def t_bank_parse():
            # regression pin for the r2 real-API bank smoke: parse_judge_json's
            # first-'{' recovery returned ONE register object from an
            # array-of-objects response; _parse_bank_array recovers the ARRAY.
            import issue2378_gen as gen

            arr = [{"name": f"n{i}", "opening": "o"} for i in range(8)]
            for txt in (
                json.dumps(arr),
                "Here are the registers:\n" + json.dumps(arr),
                "```json\n" + json.dumps(arr) + "\n```",
            ):
                got = gen._parse_bank_array(txt)
                assert isinstance(got, list) and len(got) == 8, txt[:40]
            assert gen._parse_bank_array("no json here") is None

        check("bank-builder array parser (r2 real-API smoke regression)", t_bank_parse)

        def t_rows_dir_manifest():
            # r2 review concern local-raw-stage-completeness-unchecked: on the
            # --stage-raw-from-hf path a nonempty local dir is accepted ONLY
            # when it covers the remote (path, size) manifest; the no-flag
            # path stays network-free (offline probes / same-pod producers).
            import argparse as _ap

            import issue2378_gen as gen
            from explore_persona_space.orchestrate import hub

            root = tmp / "rowsdir"
            stage_dir = root / "chat"
            stage_dir.mkdir(parents=True)
            fp = stage_dir / "chat_w1_s0_c0000.jsonl"
            fp.write_text('{"a": 1}\n', encoding="utf-8")
            calls = {"list": 0, "stage": 0}
            remote: list[tuple[str, int | None]] = []
            real_list, real_stage = hub.list_hf_entries_under_path, cm.stage_hf_prefix

            def fake_list(api, repo, prefix, **kw):
                calls["list"] += 1
                return list(remote)

            def fake_stage(prefix_rel, dest_root, revision=None):
                calls["stage"] += 1
                return stage_dir  # stands in for the mirror leaf

            def ns(flag: bool) -> _ap.Namespace:
                return _ap.Namespace(raw_root=str(root), stage_raw_from_hf=flag)

            pre = f"{cm.HF_PREFIX}/raw_completions/chat"
            try:
                hub.list_hf_entries_under_path = fake_list
                cm.stage_hf_prefix = fake_stage
                gen._STAGE_RECON_CACHE.clear()
                # no-flag: local-first, ZERO network (no listing, no stage).
                assert gen._rows_dir(ns(False), "chat") == stage_dir
                assert calls == {"list": 0, "stage": 0}, calls
                # flag + empty remote (nothing published yet): accept local.
                gen._STAGE_RECON_CACHE.clear()
                assert gen._rows_dir(ns(True), "chat") == stage_dir
                assert calls == {"list": 1, "stage": 0}, calls
                # flag + matching (name+size) manifest: accept local; memoized.
                gen._STAGE_RECON_CACHE.clear()
                remote[:] = [(f"{pre}/{fp.name}", fp.stat().st_size)]
                assert gen._rows_dir(ns(True), "chat") == stage_dir
                assert gen._rows_dir(ns(True), "chat") == stage_dir
                assert calls == {"list": 2, "stage": 0}, calls
                # flag + remote superset: partial local -> fresh mirror stage.
                gen._STAGE_RECON_CACHE.clear()
                remote.append((f"{pre}/chat_w1_s1_c0000.jsonl", 9))
                gen._rows_dir(ns(True), "chat")
                assert calls["stage"] == 1, calls
                # flag + size mismatch: stale local -> fresh mirror stage.
                gen._STAGE_RECON_CACHE.clear()
                remote[:] = [(f"{pre}/{fp.name}", fp.stat().st_size + 1)]
                gen._rows_dir(ns(True), "chat")
                assert calls["stage"] == 2, calls
            finally:
                hub.list_hf_entries_under_path = real_list
                cm.stage_hf_prefix = real_stage
                gen._STAGE_RECON_CACHE.clear()

        check("rows-dir remote-manifest reconciliation (r2 concern)", t_rows_dir_manifest)

    if failures:
        raise RuntimeError(f"probe FAILURES ({len(failures)}): " + " | ".join(failures))
    _log("[probe] all dispatch probes passed")
    return 0


def run_import_check() -> int:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    # deferred imports executed (function-body imports are unverified by a bare
    # module import — #1689): hub, preflight, poller, gen/p6 seams.
    import importlib

    for mod in ("issue2378_gen", "issue2378_p6_common", "poll_pipeline"):
        importlib.import_module(mod)
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401

    from explore_persona_space.orchestrate.preflight import (  # noqa: F401
        assert_out_root_headroom,
    )

    import inspect

    import issue2378_gen as gen

    sig = inspect.signature(gen._load_kept_ids)
    sig.bind(Path("x"), "cell")  # call-shape bind for the retry-extras seam
    _log("[import-check] OK (argcheck + deferred imports + seam bind)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PHASES = {
    "env_smoke": None,  # handled inline (no Runner needed)
    "p0_banks_pools": phase_p0,
    "p1_pilot": phase_p1,
    "p2_generate": phase_p2,
    "p3_admission": phase_p3,
    "p4_topup": phase_p4_topup,
    "p4_segb_capture": phase_p4,
    "p5_congruence": phase_p5,
    "p6_fits": phase_p6,
    "probe": None,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --phase is optional ONLY so `--import-check` can run phase-less; main()
    # refuses a missing phase on every other path.
    ap.add_argument("--phase", default=None, choices=sorted(PHASES))
    ap.add_argument("--dry-run", action="store_true", help="print the composed step plan only")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="ignore ok-flags; re-run all steps")
    ap.add_argument(
        "--sentinel-dir", default="", help=f"default {SENTINEL_DIR_DEFAULT} when present"
    )
    ap.add_argument("--logs-dir", default=str(LOGS_DIR_DEFAULT))
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--raw-root", default=str(cm.RAW_ROOT_DEFAULT))
    ap.add_argument("--raw-pilot-root", default=str(RAW_PILOT_DEFAULT))
    ap.add_argument(
        "--store-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "activations")
    )
    ap.add_argument("--stage-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "p6_stage"))
    # P1 (plan §10 flag names)
    ap.add_argument("--attempts-per-cell", type=int, default=300)
    ap.add_argument("--chat-pilot-rows", type=int, default=2500)
    ap.add_argument("--user-sim-smoke-rows", type=int, default=50)
    ap.add_argument("--pilot-round", type=int, default=1, choices=(1, 2))
    # P0/P2
    ap.add_argument(
        "--sega-attempts-per-cell",
        type=int,
        default=0,
        help="0 = size from pilot_digest.json (cap 30000)",
    )
    ap.add_argument("--chat-rows", type=int, default=cm.CHAT_DRAW_N)
    ap.add_argument("--plain-rows", type=int, default=cm.PLAIN_DRAW_N)
    ap.add_argument("--user-sim-rows", dest="user_rows", type=int, default=cm.USER_DRAW_N)
    ap.add_argument("--user-rows", dest="user_rows", type=int)
    ap.add_argument("--user-fresh-rows", type=int, default=1000)
    ap.add_argument("--user-fresh-draws", type=int, default=4)
    # P4 (plan §10 flag names)
    ap.add_argument("--target-kept-per-cell", type=int, default=cm.STORY_TARGET_KEPT)
    ap.add_argument(
        "--chat-kept", type=int, default=cm.CHAT_TARGET_KEPT, help="reporting only (gen.py mirrors)"
    )
    ap.add_argument("--fresh-rows", type=int, default=1000)
    ap.add_argument("--fresh-draws", type=int, default=4)
    ap.add_argument("--layers", default="Lstar,Lstar-8,Lstar-4,Lstar+4,Lstar+8")
    ap.add_argument("--cells", default="", help="p4_topup: csv of cells to top up")
    # P5
    ap.add_argument("--congruence-rows", type=int, default=500)
    # P6
    ap.add_argument("--pod-role", default="", help="fits-a|fits-b|fits-c|fits-d")
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--g3-poll-s", type=int, default=120)
    ap.add_argument("--g3-timeout-s", type=int, default=6 * 3600)
    ap.add_argument("--siblings-timeout-s", type=int, default=24 * 3600)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        return run_import_check()
    if args.phase is None:
        build_argparser().error("the following arguments are required: --phase")
    if args.phase == "probe":
        rc = phase_probe(args)
        _phase_line("done")
        return rc
    if args.phase == "env_smoke":
        rc = phase_env_smoke(args)
        _phase_line("done")
        return rc
    runner = Runner(Path(args.logs_dir) / args.phase, resume=not args.no_resume, dry=args.dry_run)
    rc = PHASES[args.phase](args, runner)
    if rc == 0:
        _phase_line("done")
    else:
        _log(f"[dispatch] designed halt rc={rc} (see gate report + sentinel)")
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
