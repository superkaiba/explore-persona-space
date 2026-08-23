"""issue #2378 dispatch driver — pod/VM phase machine (plan v7 §7/§9/§10).

v7 amendments (epm:progress v70; plan Amendment records A-E): dialogue family
DESCOPED (active panel = 9 cells; cm.ACTIVE_FAMILIES drives every family
loop), G1 amended to floor-funding (SEGA_ATTEMPTS_CAP * net >= FLOOR_KEPT;
recalibration SPENT at r11), pilot capture out-root ROUND-SCOPED (r12 fix),
and the P1R pilot-completion resume leg (p1_resume) added.

Runbook (venue per phase; provision commands are plan §10 verbatim):

  VM (repo venv):
    uv run python scripts/issue2378_dispatch.py --phase p0_banks_pools
  Pod (model venv; standalone gate — the MODEL phases below also self-ensure):
    uv run python scripts/issue2378_dispatch.py --phase model_venv     # ensure + env/engine smokes
    uv run python scripts/issue2378_dispatch.py --phase env_smoke      # model venv req'd
    uv run python scripts/issue2378_dispatch.py --phase engine_smoke   # model venv + 1 GPU req'd
  Pod A (4x H200; launched detached via the canonical setsid launcher —
  experimenter.md § During Execution; this driver is the WORKLOAD):
    bash scripts/issue2378_dispatch.sh p1_pilot --attempts-per-cell 300 \\
        --chat-pilot-rows 2500 --user-sim-smoke-rows 50
    bash scripts/issue2378_dispatch.sh p1_resume --pilot-round 2   # P1R (plan §4.7): complete
        # the r2 pilot on a FRESH pod — stage persisted r2 raw from HF, skip gen/judge
        # (asserted), re-run capture+sweep, amended-G1 digest + harvest.
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
    bash scripts/issue2378_dispatch.sh p6_fits --pod-role fits-d   # + pool/h4b/h5/ratio/merge

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
     — SPENT at r11 (v7 Amendment record B): can only fire at pilot_round==1;
     any future below-line measure at any round is rc=5.
  5  G1 hard fail (round 2 trip, judge-pilot FAIL, or layer-sweep rig floor)
  6  G2b survivor-predicate fail (partial-result stop; report persisted)
  7  judge pilot gate fail (mirrors issue2378_judge.RC_PILOT_GATE_FAIL)

Retry-wave policy (dispatcher-owned; plan §7 G2b): SegB-stage retries draw ONLY
from the already-admitted surplus (deterministic selection-order extras — the
same seeded order gen.phase_segb uses), <= 2 retry waves + ONE close-miss
escalation wave (>= 5,850) — never a backfill. An admitted-POOL shortfall is
G2a's to schedule (p4_topup + a VM p3_admission wave 2) BEFORE p4_segb_capture.

Model venv (r5 crash-fix — P1 died at vLLM engine init: the repo venv's
vLLM 0.11.0 / transformers 4.57.6 cannot load model type ``qwen3_5``): every
MODEL step — gen sega/chat_plain/user_sim/user_fresh/segb/fresh_draws/
user_real_render + capture pilot/capture/capture_fresh — is composed via
``_model_py()`` -> $EPM_I2378_MODEL_PY or ``/root/eps-model-venv/bin/python``
(exact pins: ``cm.MODEL_VENV_PINS`` — vllm 0.27.1 / transformers 5.15.1 /
torch 2.13.0, plan Repro card "exact pin at P1"). The MODEL phases
(p1_pilot/p2_generate/p4_topup/p4_segb_capture) call ``ensure_model_venv()``
at entry: live probe -> idempotent uv build/repair (r7: banned accel dists —
``cm.MODEL_VENV_BANNED_DISTS``, the py3.11-incompatible flashinfer-python —
are uninstalled in place) -> pin + banned-absence assert -> realized pins
recorded to <ledger>/model_venv_pins.json -> ``--phase env_smoke`` under
the model interpreter (an env mismatch fails in seconds, pre-fan-out) ->
``--phase engine_smoke`` (r8: REAL tiny engine init + generate on ONE GPU —
the class-closing gate for engine-reachable probe paths; every launched step
env additionally carries ``cm.LAUNCH_ENV_PINS`` =
``VLLM_USE_FLASHINFER_SAMPLER=0``, the r8 sampler-probe pin). The
dispatcher itself + non-model steps (banks/pools, upload_stage, capture_ready,
judge, fits) stay on the repo venv (plan: "Repo env unchanged for P0/P6/P7").

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
import re
import shlex
import shutil
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

G1_SWEEP_R2_MIN = 0.05  # plan §7 G1(c) rig-defect floor
WAVE1_SLACK = 1.25  # plan §8 "wave-1 sized with 1.25x slack"
SEGA_ATTEMPTS_CAP = 30_000  # plan §7 G1(a) cap
# Amended G1(a) rate line (plan v7 Amendment record B, epm:progress v70
# clause 2): PASS iff the pilot net rate FUNDS the binding per-cell floor at
# the attempts cap — SEGA_ATTEMPTS_CAP * net >= FLOOR_KEPT, i.e.
# net >= 6_500 / 30_000 ~= 0.21667. DERIVED from the two constants (never a
# hardcoded 0.2167) so a re-sized cap or floor moves the line with it. The
# pre-v7 line was a bare 0.25 net-rate min; v7 replaces it with floor
# funding (8_000 STORY_TARGET_KEPT is explicitly NOT the funding target —
# a floor-funded PASS below target ships the reduced-n caveat, plan §7).
G1_NET_RATE_MIN = cm.FLOOR_KEPT / SEGA_ATTEMPTS_CAP
MAX_RETRY_WAVES = 2  # plan §7 G2b: <= 2 additional generation waves
PILOT_PLAIN_ROWS = 8  # tiny plain-cell slice at P1 (0 would mean ALL rows)
ADMISSION_SLICE_N = 400  # family-balanced sync slice (<= 500 smoke exemption)

# Per-phase out-root headroom floors (GB) — plan §9 disk rows; asserted against
# the mount the out-root RESOLVES to (assert_out_root_headroom, #1333).
PHASE_HEADROOM_GB = {
    "p0_banks_pools": 3,
    "p1_pilot": 12,
    "p1_resume": 12,  # P1R re-runs the pilot capture store (same footprint)
    "p2_generate": 8,
    "p3_admission": 4,
    "p4_topup": 6,
    "p4_segb_capture": 32,
    "p5_congruence": 3,
    "p6_fits": 10,
}

# P6 fan-out shard map (plan §9: 4 suffixed cpu-bigmem pods; shard = unit
# classes across pods; fits-d additionally owns pool + the summary phases).
# v7 re-shard (dialogue descope, epm:progress v70 clause 1): the former
# fits-c dialogue shard now takes the back half of the story-Q panel — plan
# §9 retains 4 fit pods at v7 ("ladder 8 pairs").
POD_ROLE_CELLS: dict[str, tuple[str, ...]] = {
    "fits-a": ("chat", "plain_text", "storyq_astra"),
    "fits-b": ("storyq_helios", "storyq_wren"),
    "fits-c": ("storyq_dana", "storyq_vex"),
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


# Human-legible pin token appended to step START lines (the r8 fix-engaged
# observable: every launched step's env carries cm.LAUNCH_ENV_PINS; the r9
# `engine:`-prefixed entries advertise cm.ENGINE_KWARG_PINS — engine KWARGS
# threaded via create_vllm_engine, not env vars).
_PINS_TOKEN = ",".join(
    [f"{k}={v}" for k, v in sorted(cm.LAUNCH_ENV_PINS.items())]
    + [f"engine:{k}={v}" for k, v in sorted(cm.ENGINE_KWARG_PINS.items())]
)


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

    def _log_tail(self, log_path: Path, n: int) -> str:
        # split("\n") (never splitlines() — #950 U+2028 safety) keeps a
        # terminal empty element on newline-terminated logs; drop it so the
        # tail carries n REAL lines (r5 review NIT runner-log-tail-trailing-empty).
        lines = log_path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines[-n:])

    def run(
        self,
        name: str,
        argv: list[str],
        *,
        env_extra: dict[str, str] | None = None,
        ok_rcs: tuple[int, ...] = (0,),
        timeout_s: float | None = None,
        tail_lines: int = 25,
    ) -> int:
        """Run one foreground step; raise unless rc in ok_rcs; return rc.

        timeout_s (r10, reconciler-v6 D2 engine-smoke-failure-not-bounded):
        wall-clock bound on the step SUBPROCESS — subprocess.run KILLS the
        child on expiry and we raise with the log tail, so a hung gate (the
        vLLM generate()-hang class) surfaces as a bounded loud failure instead
        of blocking the dispatcher forever. None = unbounded (long fan-out
        steps own their walls)."""
        if self.dry:
            _log(f"[dry] {name}: {shlex.join(argv)}")
            return 0
        if self._skip(name, argv):
            return 0
        log_path, log = self._open_log(name)
        t0 = time.time()
        _log(f"[step] {name} START pins={_PINS_TOKEN} log={log_path}")
        with log:
            try:
                rc: int | None = subprocess.run(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(cm.REPO_ROOT),
                    # cm.LAUNCH_ENV_PINS after os.environ (authoritative — r8
                    # flashinfer-sampler-probe fix), before caller env_extra.
                    env={**os.environ, **cm.LAUNCH_ENV_PINS, **(env_extra or {})},
                    check=False,
                    timeout=timeout_s,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = None  # child killed + reaped by subprocess.run on expiry
        wall = time.time() - t0
        self.walls[name] = wall
        _log(f"[step] {name} rc={rc} wall={wall:.1f}s log={log_path}")
        if rc is None:
            raise RuntimeError(
                f"step {name} TIMED OUT after {timeout_s:.0f}s (child killed; "
                f"log tail below)\n{self._log_tail(log_path, tail_lines)}"
            )
        if rc not in ok_rcs:
            raise RuntimeError(
                f"step {name} failed rc={rc} (log tail below)\n"
                f"{self._log_tail(log_path, tail_lines)}"
            )
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
                    **cm.LAUNCH_ENV_PINS,  # r8: sampler-probe pin on every shard
                    "CUDA_VISIBLE_DEVICES": gpus[i],
                    **(env_extra or {}),
                },
            )
            log.close()
            _log(
                f"[step] {sname} START pid={p.pid} cvd={gpus[i]} pins={_PINS_TOKEN} log={log_path}"
            )
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
                    **cm.LAUNCH_ENV_PINS,  # r8: sampler-probe pin on every shard
                    "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)],
                    **(env_extra or {}),
                },
            )
            log.close()
            _log(
                f"[step] {sname} START pid={p.pid} cvd={gpus[i % len(gpus)]} "
                f"pins={_PINS_TOKEN} log={log_path}"
            )
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
        # --autostash (r6 hardening, reconciler-suggested): a RESIDUAL unstaged
        # tracked modification outside this harvest's declared paths (e.g. a
        # crashed sibling phase's partial write) stashes across the rebase and
        # re-applies, instead of the unconditional "cannot rebase: You have
        # unstaged changes" refusal (git default autoStash=off; pod clones set
        # no rebase config). The PRIMARY fix for the r5 BLOCKER is the
        # content-stable pins record in ensure_model_venv — this is
        # defense-in-depth; a genuine rebase CONFLICT still fails loud below.
        reb = _git(["rebase", "--autostash", f"origin/{BRANCH}"], check=False)
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
    # --autostash: same r6 defense-in-depth as git_harvest's rebase (see the
    # comment there) — the P6 consumers of this helper run on clones whose
    # tracked tree may carry residue a prior phase left unstaged.
    reb = _git(["rebase", "--autostash", f"origin/{BRANCH}"], check=False)
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


def _model_python() -> str:
    """Interpreter for MODEL steps (anything loading Qwen3.6-27B). Resolution:
    $EPM_I2378_MODEL_PY > <cm.MODEL_VENV_DEFAULT>/bin/python. There is NO
    repo-venv fallback — the repo env (vLLM 0.11.0 / transformers 4.57.6)
    cannot load model type `qwen3_5` (the P1 engine-init crash, r5 fix); a
    missing interpreter is ensure_model_venv's job, never a silent swap."""
    return os.environ.get(cm.MODEL_PY_ENV) or str(Path(cm.MODEL_VENV_DEFAULT) / "bin" / "python")


def _model_py(script: str, *argv: str) -> list[str]:
    """`_py()` twin for MODEL steps: the model venv's python in script mode.
    The invoked scripts self-resolve repo `src/` + `scripts/` onto sys.path
    (module-top bootstrap in gen/capture/dispatch — the #823 script-mode
    rule), so no PYTHONPATH threading is needed at call sites."""
    return [_model_python(), str(cm.REPO_ROOT / "scripts" / script), *argv]


# banned_present (r7, epm:failure v3 flashinfer-py311-array-subscript): the
# probe REPORTS banned accel-dep import names still importable in the venv —
# ensure_model_venv treats a non-empty list as a repair trigger (the pod's
# existing venv carries vllm's hard-pinned flashinfer-python 0.6.16.post3, so
# a pins-only probe reads "healthy" and the crash recurs at engine init).
_MODEL_PROBE_SRC = """\
import importlib.util, json, sys
missing = [m for m in ("transformers.models.qwen3_5", "vllm", "dotenv")
           if importlib.util.find_spec(m) is None]
assert not missing, f"model venv missing: {missing}"
banned = [m for m in %(banned_mods)r if importlib.util.find_spec(m) is not None]
from importlib.metadata import version
print(json.dumps({"python": sys.version.split()[0], "vllm": version("vllm"),
                  "transformers": version("transformers"), "torch": version("torch"),
                  "banned_present": banned}))
""" % {"banned_mods": tuple(sorted(cm.MODEL_VENV_BANNED_DISTS.values()))}


def _model_probe(py: str) -> dict | None:
    """Realized-pins dict when `py` is a qwen3_5-capable model interpreter,
    else None (missing interpreter / missing dist), logged either way."""
    if not Path(py).exists():
        _log(f"[model-venv] probe: interpreter missing at {py}")
        return None
    r = subprocess.run(
        [py, "-c", _MODEL_PROBE_SRC],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    if r.returncode != 0:
        tail = (r.stderr or "").strip()[-300:]
        _log(f"[model-venv] probe under {py} FAILED rc={r.returncode}: {tail}")
        return None
    return json.loads(r.stdout.strip().split("\n")[-1])


def _assert_driver_compat(compat_dir: str | None = None) -> None:
    """Fail the MODEL-phase gate on a host driver too old for the CUDA-13
    wheel stack (r5 review CONCERN model-venv-driver-compat-unguarded; the
    #2330 crash shape). ensure_model_venv/env_smoke are CPU-only and
    structurally cannot catch this: a pre-580 driver under torch 2.13/vllm
    0.27.1 passes both and dies per-shard at vLLM engine init ("driver too
    old"). One nvidia-smi read at gate time fails in seconds instead.

    Passes when: driver major >= cm.MODEL_DRIVER_FLOOR_MAJOR, OR the NVIDIA
    forward-compat package is ACTIVE (compat libcuda present AND on
    LD_LIBRARY_PATH — presence alone does not reach the loader). No-GPU hosts
    (VM fixture/dry runs) skip with a log line — a GPU-less real pod still
    fails loud at the fan-out's visible_gpus() gate. Deliberate waiver:
    $EPM_I2378_SKIP_DRIVER_PROBE=1 (logged loud)."""
    import shutil

    compat = compat_dir or cm.CUDA_COMPAT_DIR
    if os.environ.get(cm.SKIP_DRIVER_PROBE_ENV) == "1":
        _log(f"[model-venv] driver probe SKIPPED (${cm.SKIP_DRIVER_PROBE_ENV}=1 waiver)")
        return
    smi = shutil.which("nvidia-smi")
    if smi is None:
        _log("[model-venv] driver probe skipped: nvidia-smi not on PATH (no-GPU host)")
        return
    r = subprocess.run(
        [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    versions = [ln.strip() for ln in r.stdout.split("\n") if ln.strip()]
    if r.returncode != 0 or not versions:
        raise RuntimeError(
            f"driver probe: nvidia-smi failed rc={r.returncode}: {(r.stderr or '').strip()[-300:]}"
        )
    major = min(int(v.split(".")[0]) for v in versions)
    floor = cm.MODEL_DRIVER_FLOOR_MAJOR
    if major >= floor:
        _log(f"[model-venv] driver OK {versions[0]} (floor {floor}, CUDA-13 wheel stack)")
        return
    compat_lib = Path(compat).is_dir() and any(Path(compat).glob("libcuda.so*"))
    compat_on_ld = compat in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if compat_lib and compat_on_ld:
        _log(
            f"[model-venv] driver OK {versions[0]} < {floor} via active cuda-compat "
            f"({compat} on LD_LIBRARY_PATH — the #2330 recipe)"
        )
        return
    raise RuntimeError(
        f"host driver {versions[0]} is too old for the CUDA-13 wheel stack "
        f"(vllm {cm.MODEL_VENV_PINS['vllm']} / torch {cm.MODEL_VENV_PINS['torch']} "
        f"need driver >= {floor}); fix (gotchas.md #2330): apt-get install -y "
        f"cuda-compat-13-0 && export LD_LIBRARY_PATH={compat}:$LD_LIBRARY_PATH in the "
        f"LAUNCHER env (compat lib present: {compat_lib}; on LD_LIBRARY_PATH: {compat_on_ld})"
    )


def _build_model_venv(logs_dir: Path) -> None:
    """Idempotent build/repair of cm.MODEL_VENV_DEFAULT via uv (pins from cm).
    Recognizes an already-built venv: `uv venv` is skipped when the
    interpreter exists; `uv pip install` of exact pins is a fast no-op on
    already-satisfied dists (it only adds what is missing, e.g. python-dotenv
    on a venv built from the bare vllm+transformers recipe). Banned dists
    (cm.MODEL_VENV_BANNED_DISTS — r7 flashinfer fix) are uninstalled AFTER the
    install step on every build/repair: the install re-resolves vllm's dep
    closure and re-adds its hard-pinned flashinfer-python, so removal must be
    the LAST step (uv pip uninstall is a clean rc=0 no-op when absent)."""
    import shutil

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("model venv build: `uv` not on PATH")
    py = str(Path(cm.MODEL_VENV_DEFAULT) / "bin" / "python")
    specs = [f"{k}=={v}" for k, v in sorted(cm.MODEL_VENV_PINS.items())]
    specs += list(cm.MODEL_VENV_EXTRA_PINS)
    steps: list[tuple[str, list[str]]] = []
    if not Path(py).exists():
        steps.append(("create", [uv, "venv", cm.MODEL_VENV_DEFAULT, "--python", "3.11"]))
    steps.append(("install", [uv, "pip", "install", "--python", py, *specs]))
    banned = sorted(cm.MODEL_VENV_BANNED_DISTS)
    if banned:
        steps.append(("uninstall-banned", [uv, "pip", "uninstall", "--python", py, *banned]))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "model_venv_build.log"
    with log_path.open("a", encoding="utf-8") as log:
        for what, argv in steps:
            _log(f"[model-venv] {what}: {shlex.join(argv)} log={log_path}")
            rc = subprocess.run(
                argv, stdout=log, stderr=subprocess.STDOUT, env={**os.environ}, check=False
            ).returncode
            if rc != 0:
                raise RuntimeError(f"model venv {what} failed rc={rc} (see {log_path})")


def ensure_model_venv(args, runner: Runner) -> None:
    """MODEL-phase entry gate (r5 fix for the P1 qwen3_5 engine-init crash):
    assert host-driver compat for the CUDA-13 wheels (r6, #2330 shape) ->
    probe the model interpreter -> build/REPAIR when deficient — a missing
    interpreter, a qwen3_5-less env, OR a banned accel dist still importable
    (r7 flashinfer-py311-array-subscript fix: the existing pod venv carries
    vllm's hard-pinned flashinfer-python 0.6.16.post3, uninstalled in place) —
    never over an explicit $EPM_I2378_MODEL_PY override -> assert the exact
    cm.MODEL_VENV_PINS + banned-dist ABSENCE -> record realized pins
    CONTENT-STABLY to <ledger>/model_venv_pins.json (plan Repro card "exact
    pin at P1"; volatile provenance goes to an untracked sidecar — r6
    P4-harvest fix) -> run `--phase env_smoke` UNDER the model interpreter
    (re-run forced after a build/repair), so the next env mismatch fails in
    seconds pre-fan-out instead of per-shard at vLLM engine init -> run
    `--phase engine_smoke` (r8, epm:failure v4): a REAL tiny engine init +
    generate on ONE GPU, the class-closing gate for engine-reachable
    import/probe paths no module-import smoke can enumerate."""
    py = _model_python()
    if runner.dry:
        _log(
            f"[dry] model-venv ensure: probe {py}; on miss/banned build+repair "
            f"{cm.MODEL_VENV_DEFAULT} (pins {cm.MODEL_VENV_PINS} + "
            f"{list(cm.MODEL_VENV_EXTRA_PINS)}; banned {sorted(cm.MODEL_VENV_BANNED_DISTS)})"
        )
        runner.run("model_env_smoke", _model_py("issue2378_dispatch.py", "--phase", "env_smoke"))
        runner.run(
            "model_engine_smoke",
            _model_py("issue2378_dispatch.py", "--phase", "engine_smoke"),
            env_extra=_first_gpu_env(runner, visible_gpus(), "model_engine_smoke"),
            timeout_s=ENGINE_SMOKE_TIMEOUT_S,
            tail_lines=ENGINE_SMOKE_TAIL_LINES,
        )
        return
    _assert_driver_compat()  # fail fast BEFORE any ~4-min build on a bad host (#2330)
    built = False
    realized = _model_probe(py)
    banned_present = list((realized or {}).get("banned_present") or [])
    if realized is None or banned_present:
        if os.environ.get(cm.MODEL_PY_ENV):
            raise RuntimeError(
                f"model interpreter {py} (${cm.MODEL_PY_ENV}) is missing, lacks qwen3_5, or "
                f"carries banned dist import(s) {banned_present} — refusing to build over an "
                "explicit override; unset it or repair that venv"
            )
        if banned_present:
            _log(
                f"[model-venv] banned dist import(s) present {banned_present} — repairing the "
                "existing venv in place (r7 flashinfer-py311-array-subscript fix)"
            )
        _build_model_venv(runner.logs_dir)
        built = True
        realized = _model_probe(py)
        if realized is None:
            raise RuntimeError(
                f"model venv build at {cm.MODEL_VENV_DEFAULT} still fails the qwen3_5 probe"
            )
        if realized.get("banned_present"):
            raise RuntimeError(
                f"model venv repair at {cm.MODEL_VENV_DEFAULT} left banned dist import(s) "
                f"{realized['banned_present']} importable — uninstall step failed; see "
                f"{runner.logs_dir / 'model_venv_build.log'}"
            )
    bad = {
        k: (want, realized.get(k))
        for k, want in cm.MODEL_VENV_PINS.items()
        if realized.get(k) != want
    }
    if bad:
        raise RuntimeError(
            f"model venv pin mismatch (pinned, realized): {bad} — rebuild "
            f"{cm.MODEL_VENV_DEFAULT} or update cm.MODEL_VENV_PINS deliberately "
            "(plan Repro card 'exact pin at P1')"
        )
    # CONTENT-STABLE pins record (r5 reconciler BLOCKER
    # model-venv-pins-rewrite-breaks-p4-harvest): the tracked record carries
    # ONLY stable pin content — volatile run_metadata (timestamp/argv/git) goes
    # to the UNTRACKED sidecar below — and an unchanged record is never
    # rewritten. A P2/P4/p4_topup re-ensure on a clone that materialized P1's
    # committed copy therefore leaves the tracked file byte-identical + clean,
    # so the scoped git_harvest rebase and _git_pull_rebase consumers cannot
    # refuse on it. A prior record in any OTHER shape (the r5 metadata-bearing
    # format included) is normalized to this clean form once.
    pins_path = Path(args.ledger_root) / "model_venv_pins.json"
    record = {
        "interpreter": py,
        "realized": realized,
        "pinned": dict(cm.MODEL_VENV_PINS),
        "extra_pins": list(cm.MODEL_VENV_EXTRA_PINS),
        # r7: the banned set is part of the Repro-card pinned-set (a constant,
        # so the record stays content-stable; the r6-era 4-key record on disk
        # normalizes ONCE to this 5-key form, then re-ensures skip again).
        "banned": sorted(cm.MODEL_VENV_BANNED_DISTS),
    }
    prior = json.loads(pins_path.read_text(encoding="utf-8")) if pins_path.exists() else None
    if prior == record:
        _log(
            f"[model-venv] pins record unchanged — rewrite skipped (P4-harvest safety) {pins_path}"
        )
    else:
        cm.atomic_write_json(pins_path, record)
    # Volatile ensure provenance (timestamp + argv + git) — untracked sidecar
    # under the gitignored dispatch-logs root, never the tracked ledger.
    cm.atomic_write_json(
        runner.logs_dir / "model_venv_ensure_meta.json",
        {"pins_record": str(pins_path), "content": record, "metadata": cm.run_metadata()},
    )
    _log(
        f"[model-venv] OK {py} vllm={realized['vllm']} "
        f"transformers={realized['transformers']} torch={realized['torch']} "
        f"banned-absent={','.join(sorted(cm.MODEL_VENV_BANNED_DISTS))} "
        f"record={pins_path}"
    )
    if built:
        # A fresh/repaired venv invalidates any prior env_smoke / engine_smoke
        # ok-flag (the r5-disclosed overlay-wipe residue: the argv sha is
        # unchanged across a rebuild, so the resume skip would silently reuse
        # the OLD verdict) — force BOTH gates to re-run under the rebuilt
        # interpreter (r8 extends the r6 mechanics to the engine gate).
        for step in ("model_env_smoke", "model_engine_smoke"):
            stale_ok = runner._ok_path(step)
            if stale_ok.exists():
                stale_ok.unlink()
                _log(f"[model-venv] rebuilt venv — stale {step} ok-flag cleared (re-run forced)")
    runner.run("model_env_smoke", _model_py("issue2378_dispatch.py", "--phase", "env_smoke"))
    # r8 CLASS-CLOSING gate (epm:failure v4): REAL tiny engine init + 1-prompt
    # generate on ONE GPU BEFORE any multi-shard fan-out — module-import smokes
    # cannot enumerate engine-reachable probe paths (phase_engine_smoke doc).
    runner.run(
        "model_engine_smoke",
        _model_py("issue2378_dispatch.py", "--phase", "engine_smoke"),
        env_extra=_first_gpu_env(runner, visible_gpus(), "model_engine_smoke"),
        # r10 D2: bounded gate — a hung engine init raises with the log tail
        # (~100 s measured wall on 1 GPU; 900 s constant, >=2x headroom).
        timeout_s=ENGINE_SMOKE_TIMEOUT_S,
        tail_lines=ENGINE_SMOKE_TAIL_LINES,
    )


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
    # v7: only ACTIVE families enter the slice (dialogue descoped — archival
    # dialog rows in a mixed mined_dir are skipped, never balanced against).
    by_family: dict[str, dict[str, list[str]]] = {f: {} for f in cm.ACTIVE_FAMILIES}
    for rid, m in mined.items():
        if m["family"] not in by_family:
            continue
        by_family[m["family"]].setdefault(m["cell"], []).append(rid)
    per_family = -(-n_total // len(cm.ACTIVE_FAMILIES))  # ceil-div
    chosen: list[str] = []
    for family in cm.ACTIVE_FAMILIES:
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


# r10 (G1 accounting fix, epm:progress v63): the r<10 sega/segb writers carried
# the cell only in the FILENAME — parse it back when the payload lacks 'cell'.
# <cell> may contain underscores (storyq_astra); greedy `.+` backtracks to the
# LAST `_w<digits>_s<digits>.json` suffix, and cm.CELL_FAMILY membership gates
# the capture so stage-level summaries never mint a bogus per-cell bucket.
_SUMMARY_CELL_RE = re.compile(r"^summary_(.+)_w\d+_s\d+\.json$")


def _sum_stage_summaries(stage_dir: Path, keys: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Aggregate gen.py per-shard summaries: <stage>/summary_<cell>_w*_s*.json."""
    per_cell: dict[str, dict[str, int]] = {}
    for path in sorted(stage_dir.glob("summary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = payload.get("cell")
        if cell is None:
            # Filename fallback FIRST (repairs the already-written pod-side
            # pilot summaries with zero GPU) — cm.CELL_FAMILY-membership-gated,
            # so user_sim's summary_w1_s0.json (no pattern match) and future
            # stage-level summaries still fall through to the stage-name
            # fallback LAST.
            m = _SUMMARY_CELL_RE.match(path.name)
            if m and m.group(1) in cm.CELL_FAMILY:
                cell = m.group(1)
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
    walls_merge_note: str | None = None,
) -> dict:
    """Aggregate P1 per-stage counters into the G1 artifact (plan §7 G1, v7).

    ``walls_merge_note`` (P1R resume leg): the caller merged a prior round's
    committed ``measured_walls_s`` under the fresh runner walls for stages the
    resume leg did not re-run; the note records that provenance in the digest
    (the commit-98565a9d7d hand-merge mechanism, now in code).
    """
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
    # v7 (Amendment record B): the GATE iterates ACTIVE families only —
    # per_stage/per_cell above still pool archival dialogue rows for the
    # record, but dialogue never enters the PASS predicate or wave sizing.
    families: dict[str, dict] = {}
    for fam in cm.ACTIVE_FAMILIES:
        net = 1.0
        for st in stages.values():
            net *= st.get(fam, {}).get("rate", 0.0)
        sized = (
            min(SEGA_ATTEMPTS_CAP, math.ceil(cm.STORY_TARGET_KEPT * WAVE1_SLACK / net))
            if net > 0
            else SEGA_ATTEMPTS_CAP
        )
        projected = SEGA_ATTEMPTS_CAP * net  # kept rows funded at the attempts cap
        families[fam] = {
            "net_kept_per_attempt": net,
            "trip_line": G1_NET_RATE_MIN,
            "attempts_cap": SEGA_ATTEMPTS_CAP,
            "floor_kept": cm.FLOOR_KEPT,
            "projected_kept_at_cap": projected,
            "pass": projected >= cm.FLOOR_KEPT,
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
            reasons.append(
                f"G1(a) {fam}: projected kept at cap "
                f"{f['projected_kept_at_cap']:.0f} < floor {cm.FLOOR_KEPT} "
                f"(net {f['net_kept_per_attempt']:.4f} < {G1_NET_RATE_MIN:.5f})"
            )
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
    if walls_merge_note is not None:
        digest["walls_merge_note"] = walls_merge_note
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
    # v7 amended predicate (Amendment record A / epm:progress v70 clause 1):
    # the dialogue clause (>=2 dialogue survivors) is REMOVED with the family.
    n_d = len([c for c in survivors if c in cm.DIALOG_CELLS])  # 0 by construction at v7
    predicate = "chat" in survivors and "plain_text" in survivors and n_q >= 3
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
        "survivor_predicate": "chat + plain + >=3 storyQ (user cells excluded; v7 amended)",
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
    """Plan §12 assumption 2 (blocking, before any provisioning; model venv).

    r7 crash-fix (epm:failure v3, flashinfer-py311-array-subscript) extends the
    gate beyond top-level imports: (1) banned accel dists must be ABSENT
    (cm.MODEL_VENV_BANNED_DISTS — flashinfer-python 0.6.16.post3 raises
    TypeError at import on py3.11 via a runtime-evaluated `array.array[int]`
    annotation); (2) the vLLM COMPILE-BACKEND import chain is exercised
    UNGUARDED, because EngineCore reaches it lazily at engine init behind an
    ImportError-ONLY guard that misses TypeError/SyntaxError — the gate must
    fail in seconds pre-fan-out, not per-shard ~30 s into engine init."""
    _phase_line("env_smoke")
    import importlib
    import importlib.util

    banned_present = [
        mod
        for mod in sorted(cm.MODEL_VENV_BANNED_DISTS.values())
        if importlib.util.find_spec(mod) is not None
    ]
    if banned_present:
        raise RuntimeError(
            f"banned accel dist import(s) present in the model venv: {banned_present} — "
            "run the model_venv ensure gate (it uninstalls them in place; "
            "cm.MODEL_VENV_BANNED_DISTS carries the rationale)"
        )
    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        raise RuntimeError("transformers lacks qwen3_5 — upgrade the model venv")
    # Deliberately NO try/except: any exception class (TypeError/SyntaxError
    # from a py-version-incompatible accel dep included) fails the gate — the
    # exact chain EngineCore imports at compile-backend init (backends.py ->
    # passes/pass_manager.py -> fusion passes -> guarded accel-dep imports).
    importlib.import_module("vllm.compilation.backends")
    _log("[env_smoke] compile-backend import OK (vllm.compilation.backends; banned absent)")
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


ENGINE_SMOKE_MAX_MODEL_LEN = 1024  # tiny context: init-path gate, not a capacity test
ENGINE_SMOKE_MAX_NUM_SEQS = 8
ENGINE_SMOKE_MAX_TOKENS = 8
# r10 (reconciler-v6 D2 engine-smoke-failure-not-bounded): wall-clock bound on
# the gate SUBPROCESS. Measured basis: ~100 s end-to-end on 1 H200 (r9 pilot —
# engine init + 1-prompt generate); 900 s is >=2x headroom over that basis
# (~9x, sized generously per the p90 fence convention) so a hung engine init
# (the vLLM generate()-hang class) fails loud in minutes instead of blocking
# the dispatcher forever.
ENGINE_SMOKE_TIMEOUT_S = 900
ENGINE_SMOKE_TAIL_LINES = 40  # D2(ii): surface the gate log tail in the raise


def phase_engine_smoke(args) -> int:
    """CLASS-CLOSING pre-fan-out gate (r8 crash-fix, epm:failure v4,
    flashinfer-absent-sampler-probe-modulenotfound): boot a REAL tiny vLLM
    engine on ONE GPU under the model venv and run one trivial generate.

    Module-import smokes (env_smoke's compile-backend chain, r7) structurally
    cannot enumerate every ENGINE-reachable import/probe path — the r8 crash
    fired inside EngineCore init (gpu_model_runner -> Sampler ->
    TopKTopPSampler -> flashinfer_sampler_supported() -> bare
    `from flashinfer import ...`), a chain no top-level import reaches. This
    gate subsumes them for engine-reachable paths: success = engine constructs
    (Sampler probe included) + one sampled completion returns; failure = loud
    rc != 0 BEFORE any multi-shard fan-out, ~minutes instead of a pod cycle.

    Choice (r8 brief): the TARGET model (cm.MODEL_ID) at tiny knobs — the
    exact qwen3_5 engine path + weights the shards use (a tiny stand-in model
    would exercise a different arch path); enforce_eager skips cudagraph
    capture (init-path gate, minutes not tens of minutes). Runs under the
    model interpreter, invoked by ensure_model_venv via _model_py with CVD
    pinned to ONE GPU by the runner and VLLM_USE_FLASHINFER_SAMPLER=0 from
    cm.LAUNCH_ENV_PINS (parity: same env composition the shards get).

    r9 (epm:failure v5, flashinfer-absent-gdn-prefill-modulenotfound): the
    engine call ALSO threads cm.ENGINE_KWARG_PINS (gdn_prefill_backend=
    "triton") — same seam parity with gen._build_engine; the v5 crash fired
    at the FIRST prefill of THIS gate's 1-prompt generate, proving the gate
    reaches the kernel path the pin governs (the r9 fix-engaged vehicle:
    vllm's kernel-selection line flips to
    "Using Triton/FLA GDN prefill kernel (requested=triton, ...)").

    Terminal is `os._exit(0)` after explicit flushes — the sanctioned vLLM
    generation-driver terminal (gotchas.md #1739/#2149: interpreter
    finalization can deadlock on surviving engine children; the parent
    runner.run writes the ok-flag off rc, and this phase writes no artifacts
    beyond its redirected stdout, so skipping finalization is safe)."""
    _phase_line("engine_smoke")
    t0 = time.time()
    # Same worker discipline as the gen workers (gen.py module top, #628):
    # spawn, never fork — set BEFORE the first vllm import (read at import).
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # r10 (reconciler-v6 D1): env-pin parity even on a DIRECT model-python
    # invocation — the composed paths (Runner.run) supply cm.LAUNCH_ENV_PINS
    # in the launcher env; setdefault never clobbers a launcher-supplied
    # value, and vllm reads these envs lazily (post-import safe).
    for k, v in cm.LAUNCH_ENV_PINS.items():
        os.environ.setdefault(k, v)
    try:
        import dataclasses

        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs

        from explore_persona_space.eval.generation import create_vllm_engine

        kwargs: dict = {}
        if "language_model_only" in {f.name for f in dataclasses.fields(EngineArgs)}:
            kwargs["language_model_only"] = True  # gen._engine parity (omni towers skipped)
        kwargs.update(cm.ENGINE_KWARG_PINS)  # r9: GDN prefill pin (parity with gen._build_engine)
        llm = create_vllm_engine(
            cm.MODEL_ID,
            max_model_len=ENGINE_SMOKE_MAX_MODEL_LEN,
            max_num_seqs=ENGINE_SMOKE_MAX_NUM_SEQS,
            seed=cm.SEED,
            dtype="bfloat16",
            enforce_eager=True,
            **kwargs,
        )
        _log(f"[engine_smoke] engine constructed wall={time.time() - t0:.1f}s")
        sp = SamplingParams(
            temperature=cm.TEMPERATURE,
            top_p=cm.TOP_P,
            top_k=cm.TOP_K,
            seed=cm.SEED,
            max_tokens=ENGINE_SMOKE_MAX_TOKENS,
        )
        # ONE prompt (no chunking needed); use_tqdm=False (#613 ZeroDivision).
        outs = llm.generate(["engine smoke probe"], [sp], use_tqdm=False)
        if not (outs and outs[0].outputs and outs[0].outputs[0].text is not None):
            raise RuntimeError("engine smoke: generate returned no completion")
        _log(
            f"[engine_smoke] engine init + 1-prompt generate OK "
            f"model={cm.MODEL_ID} max_model_len={ENGINE_SMOKE_MAX_MODEL_LEN} "
            f"enforce_eager=True wall={time.time() - t0:.1f}s"
        )
        # Best-effort graceful engine-core shutdown, then the hard terminal.
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except BaseException:
        # r10 (reconciler-v6 D2(ii)): DEFINED failure path — print the
        # traceback into the gate log (Runner.run's raise surfaces the tail),
        # flush, hard-exit rc=1. A raise propagating into interpreter
        # finalization can DEADLOCK on surviving engine children
        # (gotchas.md #1739/#2149), turning a loud failure into the unbounded
        # hang the r<10 success-only exit left reachable.
        import traceback

        traceback.print_exc()
        _log("[engine_smoke] FAILED — engine init / generate / teardown did not complete (rc=1)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    _phase_line("done")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _is_model_interpreter() -> bool:
    """True when THIS process already runs under the model interpreter (the
    ensure_model_venv/_run_engine_smoke_gate subprocess re-entry). Compares
    the venv PREFIX — sys.prefix vs the interpreter's <venv>/bin/python
    parent-of-parent — NEVER resolve()d executables: both venvs' bin/python
    can symlink to the SAME base interpreter, which would misread the repo
    venv as the model venv (and skip D1's re-dispatch entirely)."""
    model_py = Path(_model_python())
    try:
        return Path(sys.prefix).resolve() == model_py.parent.parent.resolve()
    except OSError:
        return False


def _run_engine_smoke_gate(args) -> int:
    """r10 (reconciler-v6 D1 standalone-engine-smoke-bypasses-model-env): the
    standalone `--phase engine_smoke` entry constructs the engine EXACTLY as
    the fan-out legs do — model-venv interpreter (_model_py), cm.LAUNCH_ENV_PINS
    (Runner.run's env merge), single-GPU CVD pin (_first_gpu_env), and the D2
    wall-clock bound — by re-dispatching itself as the SAME composed subprocess
    ensure_model_venv runs (cm.ENGINE_KWARG_PINS thread inside
    phase_engine_smoke: engine kwargs, interpreter-independent). The r<10
    standalone entry ran the body in-process under the REPO venv (vLLM 0.11.0,
    no qwen3_5) with no env/CVD composition. resume=False: a standalone gate
    invocation is deliberate — always re-run. Building a missing model venv is
    ensure_model_venv's job (`--phase model_venv`), never a silent swap."""
    py = _model_python()
    if not args.dry_run and not Path(py).exists():
        raise RuntimeError(
            f"engine_smoke standalone entry: model interpreter missing at {py} — "
            "run `--phase model_venv` first (ensure_model_venv owns the build/repair)"
        )
    runner = Runner(Path(args.logs_dir) / "engine_smoke", resume=False, dry=args.dry_run)
    runner.run(
        "model_engine_smoke",
        _model_py("issue2378_dispatch.py", "--phase", "engine_smoke"),
        env_extra=_first_gpu_env(runner, visible_gpus(), "model_engine_smoke"),
        timeout_s=ENGINE_SMOKE_TIMEOUT_S,
        tail_lines=ENGINE_SMOKE_TAIL_LINES,
    )
    return 0


def phase_model_venv(args, runner: Runner) -> int:
    """Standalone model-venv ensure + env_smoke/engine_smoke gates (launcher/
    orchestrator pre-step; the MODEL phases p1/p2/p4_topup/p4 self-ensure at
    entry)."""
    ensure_model_venv(args, runner)
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


# P1R skip-assert globs (crash-fix 2026-08-23): each skipped stage is asserted
# on its DURABLE output using the WRITER-REAL filenames verified against the
# staged r2 HF set (upload-verification v1) — the r12 original used a guessed
# `rows_*.jsonl` for chat/plain while the gen writers emit
# `{cell}_w{wave}_s{shard}_c{chunk}.jsonl` (#906 fixture-vs-writer drift class;
# the consumer `gen._stage_kept_rows` globs `*.jsonl` and filters rows, so the
# tighter writer-real pattern is the diagnostic assert).
P1R_SKIP_ASSERT_GLOBS: tuple[tuple[str, str], ...] = (
    ("sega", "summary_*.json"),
    ("segb", "summary_*.json"),
    ("user_sim", "summary_*.json"),
    ("chat", "chat_w*_c*.jsonl"),
    ("plain", "plain_text_w*_c*.jsonl"),
)


def _pilot_round_scope(raw_pilot: Path, runner: Runner, rnd: int) -> tuple[Path, Runner, str, Path]:
    """Round-scope the pilot resume key (logs dir), raw root, HF prefix, AND
    the capture out-root (r1 review g5 blocker 2, G1 recalibration
    resume-skip; capture out-root added r12 — plan §4.7 out-root fix): a
    `--pilot-round 2` re-pilot must RE-RUN every generation/judge/capture
    step instead of skipping onto round-1 OK-flags and reproducing the trip,
    and its capture must land in a FRESH store — the r2 crash was capture
    resuming into round 1's StageLedger and tripping the regime-fingerprint
    fail-loud. Round 1 stays byte-identical (stable paths throughout).
    Ledger pilot outputs (kept/, judge reports, digest, sweep) keep their
    STABLE paths DELIBERATELY — round 2 re-runs + overwrites them (the final
    pilot verdict), so the P2 gate + harvest paths are unchanged; these are
    whole-file atomic_write_json overwrites, not ledger-resumed stores, so
    cross-round overwrite is safe by construction."""
    hf_pilot_prefix = f"{cm.HF_PREFIX}/raw_completions/pilot"
    pilot_store = cm.pilot_capture_out_root(rnd)
    if rnd > 1:
        raw_pilot = raw_pilot / f"r{rnd}"
        runner = Runner(runner.logs_dir / f"p1_pilot_r{rnd}", resume=runner.resume, dry=runner.dry)
        hf_pilot_prefix = f"{cm.HF_PREFIX}/raw_completions/pilot/r{rnd}"
    return raw_pilot, runner, hf_pilot_prefix, pilot_store


def phase_p1(args, runner: Runner) -> int:
    """P1 pilot on pod A (plan §7 G1 + §9 row 1). Pilot generations run under a
    SEPARATE raw root (regime isolation vs P2 production ledgers) and upload
    under raw_completions/pilot/<stage>."""
    _phase_line("p1_pilot")
    raw_pilot, ledger_root = _pilot_roots(args)
    raw_pilot, runner, hf_pilot_prefix, pilot_store = _pilot_round_scope(
        raw_pilot, runner, int(args.pilot_round)
    )
    assert_headroom("p1_pilot", raw_pilot)
    ensure_model_venv(args, runner)
    gpus = visible_gpus()
    pilot_kept = ledger_root / "pilot" / "kept"
    common = ["--raw-root", str(raw_pilot), "--skip-upload"]

    runner.fanout(
        "p1.sega",
        _model_py(
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
        _model_py(
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
        _model_py(
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
        _model_py(
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
        _model_py(
            "issue2378_capture.py",
            "--phase",
            "pilot",
            "--pilot-rows",
            str(args.chat_pilot_rows),
            "--raw-root",
            str(raw_pilot),
            # r12 out-root fix (plan §4.7): the capture store is ROUND-scoped
            # like the raw root / resume keys / HF prefix above — a rnd>=2
            # pilot previously fell to the stable default and died on round
            # 1's StageLedger regime fingerprint.
            "--pilot-out-root",
            str(pilot_store),
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
            f"(projected kept at cap >= {cm.FLOOR_KEPT}/family, i.e. net >= "
            f"{G1_NET_RATE_MIN:.5f}; judge PASS; sweep R2 >= {G1_SWEEP_R2_MIN})"
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
            "eval_results/issue_2378/model_venv_pins.json",
        ],
        f"task #{ISSUE}: P1 pilot artifacts (G1 {digest['verdict']} harvest, pre-P2 — plan §9)",
    )
    if digest["verdict"] != "PASS":
        # v7 (Amendment record B): the ONE recalibration round is SPENT (r11
        # miner-window recalibration -> the r2 pilot). The branch stays for
        # archival correctness — it can only fire at pilot_round==1, and any
        # future below-line measure at ANY round is RC_G1_FAIL.
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


# P1R staging contract (plan §4.7): the r2 pilot raw prefixes persisted to HF
# before pod-2378 was lost — exact per-stage file counts from the
# upload-verification v1 marker (2026-08-21T09:38:21Z; 331 files total). The
# staging step asserts this EXACT set; any mismatch aborts to the named
# full-re-pilot fallback.
_P1R_R2_STAGE_COUNTS: dict[str, int] = {
    "sega": 108,
    "sega_mined": 36,
    "chat": 20,
    "plain": 16,
    "user_sim": 4,
    "segb": 144,
    "judge_admission": 1,
    "judge_admission_pilot": 1,
    "adm_slice": 1,
}
assert sum(_P1R_R2_STAGE_COUNTS.values()) == 331, "P1R expected-set drifted (upload-verif v1)"


def _p1r_count_files(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file())


def _p1r_verify_counts(raw_pilot: Path) -> dict[str, int]:
    """Exact-set count assert over the staged r2 stage dirs (fail-loud)."""
    counts = {st: _p1r_count_files(raw_pilot / st) for st in _P1R_R2_STAGE_COUNTS}
    bad = {st: n for st, n in counts.items() if n != _P1R_R2_STAGE_COUNTS[st]}
    if bad:
        raise RuntimeError(
            "p1_resume staging count mismatch (staged != expected): "
            + ", ".join(f"{st}: {bad[st]} != {_P1R_R2_STAGE_COUNTS[st]}" for st in sorted(bad))
        )
    return counts


def _p1r_stage_r2_raw(raw_pilot: Path, hf_pilot_prefix: str) -> dict[str, int]:
    """Stage the r2 pilot raw prefixes from HF into the ROUND-SCOPED raw root
    (plan §4.7 P1R step (a)). Idempotent per stage: a dest dir already holding
    the exact expected file count is kept; otherwise ONE whole-prefix mirror
    pull (scoped listing) + per-stage ``os.replace`` moves (same filesystem —
    the mirror lives inside ``raw_pilot``). Fail-loud on ANY count mismatch
    (the 331-file exact set, upload-verification v1)."""
    missing = {
        st: n
        for st, n in _P1R_R2_STAGE_COUNTS.items()
        if not ((raw_pilot / st).is_dir() and _p1r_count_files(raw_pilot / st) == n)
    }
    if missing:
        mirror_root = raw_pilot / "hf_mirror"
        leaf = cm.stage_hf_prefix(hf_pilot_prefix, mirror_root)
        for st in sorted(missing):
            src = leaf / st
            if not src.is_dir():
                raise RuntimeError(f"p1_resume staging: HF prefix lacks stage dir {st!r} ({src})")
            dest = raw_pilot / st
            if dest.exists():
                shutil.rmtree(dest)  # partial/stale local copy — replace wholesale
            os.replace(src, dest)
        shutil.rmtree(mirror_root)  # reap the mirror residue (raw root stays canonical)
    counts = _p1r_verify_counts(raw_pilot)
    _log(f"[p1r] staged r2 raw verified: {sum(counts.values())} files across {len(counts)} stages")
    return counts


def phase_p1_resume(args, runner: Runner) -> int:
    """P1R — pilot-completion resume leg (plan §4.7, r12; epm:progress v70).

    Completes the r2 pilot on a FRESH pod without re-spending generation /
    judge GPU: (a) stages the persisted r2 raw prefixes from HF
    (count-asserted 331-file exact set), (b) asserts the committed ledger
    outputs are present from git (branch tip, commit f9c5451b62), (c) SKIPS
    the generation/judge/admission/segb runner stages — skip conditions
    ASSERTED on durable outputs, never silent — (d) re-runs the pilot capture
    through the SAME production entrypoint with the r12 ROUND-SCOPED
    out-root, (e) the layer sweep (inside the capture step), (f) re-composes
    the G1 digest under the AMENDED (v7) gate with walls merged from the
    committed digest for stages not re-run, and (g) writes the G1 verdict
    sentinel + artifact harvest. A FAIL is TERMINAL (RC_G1_FAIL): the one
    recalibration round is SPENT (r11 — plan v7 Amendment record B)."""
    _phase_line("p1_resume")
    rnd = int(args.pilot_round)
    if rnd != 2:
        raise RuntimeError(
            "p1_resume requires --pilot-round 2 (it is the r2 pilot-completion leg; "
            "for a fresh pilot run --phase p1_pilot instead)"
        )
    raw_pilot, ledger_root = _pilot_roots(args)
    raw_pilot, runner, hf_pilot_prefix, pilot_store = _pilot_round_scope(raw_pilot, runner, rnd)
    assert_headroom("p1_resume", raw_pilot)
    ensure_model_venv(args, runner)
    gpus = visible_gpus()

    # (a) stage the r2 raw prefixes from HF (exact-set count assert).
    if runner.dry:
        _log(
            f"[dry] p1r.stage_raw: stage {hf_pilot_prefix} -> {raw_pilot} "
            f"({sum(_P1R_R2_STAGE_COUNTS.values())} files, count-asserted)"
        )
    else:
        try:
            _p1r_stage_r2_raw(raw_pilot, hf_pilot_prefix)
        except Exception as e:
            raise RuntimeError(
                f"P1R ABORT — r2 raw staging failed ({e}). FALLBACK (plan §4.7): run ONE "
                "full question-only re-pilot on a fresh pod: --phase p1_pilot "
                "--pilot-round 2."
            ) from e

    # (b) committed ledger outputs present from git (branch tip f9c5451b62).
    # kept/ covers the FULL r2 panel (5 storyQ + 4 archival dialogue cells —
    # the digest pools dialogue for the record; the amended gate ignores it).
    ledger_expect = [
        *(
            ledger_root / "pilot" / "kept" / f"{c}.json"
            for c in (*cm.STORY_Q_CELLS, *cm.DIALOG_CELLS)
        ),
        ledger_root / "judge" / "pilot_admission_sync.json",
        ledger_root / "pilot" / "judge" / "admission_summary.json",
        # Walls source for the (f) merge — MANDATORY (r12 reconcile standing
        # rec 1, concern p1r-missing-prior-digest-clobbers-walls): a partial
        # checkout must fail loud here, never silently drop the r10-guarded
        # walls under a false merge note.
        ledger_root / "pilot" / "pilot_digest.json",
    ]
    missing_ledger = [str(p) for p in ledger_expect if not p.is_file()]
    if missing_ledger and not runner.dry:
        raise RuntimeError(
            "p1_resume: committed r2 ledger outputs missing from the checkout "
            f"(expected from git at branch tip, commit f9c5451b62): {missing_ledger} — "
            "sync the pod clone first; if genuinely lost, FALLBACK (plan §4.7): "
            "full re-pilot via --phase p1_pilot --pilot-round 2."
        )

    # (c) generation/judge/admission/segb stages SKIPPED — asserted on their
    # durable outputs (the digest + capture consume exactly these), never a
    # silent skip.
    if not runner.dry:
        for stage, pat in P1R_SKIP_ASSERT_GLOBS:
            if not sorted((raw_pilot / stage).glob(pat)):
                raise RuntimeError(
                    f"p1_resume skip-assert failed: no {pat} under {raw_pilot / stage} "
                    "(the skipped stage's durable output is absent)"
                )
        _log("[p1r] skip-asserts passed: gen/judge/admission/segb outputs already durable")

    # (d)+(e) pilot capture with the FIXED round-scoped out-root; the layer
    # sweep runs inside the same production entrypoint (--layer-sweep-out).
    # Step name matches phase_p1's so the fresh wall OVERRIDES the committed
    # digest's stale r1 value in the walls-merge below.
    runner.run(
        "p1.capture_pilot",
        _model_py(
            "issue2378_capture.py",
            "--phase",
            "pilot",
            "--pilot-rows",
            str(args.chat_pilot_rows),
            "--raw-root",
            str(raw_pilot),
            "--pilot-out-root",
            str(pilot_store),
            "--skip-capture-ready",
            "--layer-sweep-out",
            str(ledger_root / "pilot" / "layer_sweep.json"),
            "--skip-upload",
        ),
        env_extra=_first_gpu_env(runner, gpus, "p1.capture_pilot"),
    )
    if runner.dry:
        _log("[dry] p1r.digest: compose_pilot_digest (amended G1) + walls-merge")
        return 0

    # (f) digest with walls merged from the committed round-1 digest for the
    # stages this leg did not re-run (the 98565a9d7d hand-merge, now in code).
    digest_path = ledger_root / "pilot" / "pilot_digest.json"
    prior_walls: dict[str, float] = {}
    if digest_path.is_file():
        prior = json.loads(digest_path.read_text(encoding="utf-8"))
        prior_walls = {k: float(v) for k, v in prior.get("measured_walls_s", {}).items()}
    walls = {**prior_walls, **runner.walls}
    note = (
        "P1R (r12): measured_walls_s for stages NOT re-run merged from the committed "
        "pilot_digest.json at branch tip; fresh keys from this leg override: "
        + (", ".join(sorted(runner.walls)) or "none")
    )
    digest = compose_pilot_digest(
        raw_pilot,
        ledger_root,
        walls,
        pilot_round=rnd,
        attempts_per_cell=args.attempts_per_cell,
        walls_merge_note=note,
    )
    blocks = digest["verdict"] != "PASS"
    write_sentinel(args, "epm:progress", digest, gate="g1", blocks_pipeline=blocks)
    upload_json_files(
        [digest_path, ledger_root / "pilot" / "layer_sweep.json"],
        f"{cm.HF_PREFIX}/pilot",
    )
    git_harvest(
        [
            "eval_results/issue_2378/pilot/pilot_digest.json",
            "eval_results/issue_2378/pilot/layer_sweep.json",
            "eval_results/issue_2378/judge/pilot_admission_sync.json",
            "eval_results/issue_2378/model_venv_pins.json",
        ],
        f"task #{ISSUE}: P1R pilot completion (G1 {digest['verdict']} harvest — plan §4.7)",
    )
    if digest["verdict"] != "PASS":
        # v7: recalibration SPENT — any below-line measure is terminal.
        return RC_G1_FAIL
    write_sentinel(
        args,
        "epm:progress",
        {
            "phase": "p1_resume",
            "status": "complete",
            "g1": "PASS",
            "wave1_sizing": digest["families"],
        },
    )
    return 0


def phase_p2(args, runner: Runner) -> int:
    _phase_line("p2_generate")
    assert_headroom("p2_generate", Path(args.raw_root))
    ensure_model_venv(args, runner)
    ledger_root = Path(args.ledger_root)
    gpus = visible_gpus()
    # v7: active families only (dialogue descoped). The digest filter guards
    # against a pre-v7 digest carrying a dialogue sizing entry.
    if args.sega_attempts_per_cell > 0:
        per_family = {fam: args.sega_attempts_per_cell for fam in cm.ACTIVE_FAMILIES}
    elif not runner.dry:
        digest = json.loads(
            (ledger_root / "pilot" / "pilot_digest.json").read_text(encoding="utf-8")
        )
        if digest["verdict"] != "PASS":
            raise RuntimeError("p2_generate refused: pilot_digest verdict is not PASS (G1)")
        per_family = {
            fam: min(SEGA_ATTEMPTS_CAP, int(f["wave1_attempts_per_cell"]))
            for fam, f in digest["families"].items()
            if fam in cm.ACTIVE_FAMILIES
        }
        missing = [fam for fam in cm.ACTIVE_FAMILIES if fam not in per_family]
        if missing:
            raise RuntimeError(f"p2_generate: pilot_digest lacks sizing for families {missing}")
    else:
        per_family = {fam: 0 for fam in cm.ACTIVE_FAMILIES}
    fam_cells = {fam: list(cm.FAMILY_CELLS[fam]) for fam in cm.ACTIVE_FAMILIES}
    for fam, attempts in per_family.items():
        runner.fanout(
            f"p2.sega.{fam}",
            _model_py(
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
        _model_py(
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
        _model_py(
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
        _model_py(
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
    ensure_model_venv(args, runner)
    runner.fanout(
        "p4.topup_sega",
        _model_py(
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
    ensure_model_venv(args, runner)
    gpus = visible_gpus()
    kept_dir = ledger_root / "kept"
    stage_flags = ["--stage-raw-from-hf", "--stage-pools-from-hf"]

    def run_segb(wave: int, kdir: Path, target: int, cells: str = "") -> None:
        argv = _model_py(
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
        _model_py("issue2378_gen.py", "--phase", "user_real_render", *stage_flags),
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
    # Production capture store: DELIBERATELY STABLE across relaunches (r12
    # out-root audit). Unlike the pilot store (round-scoped — a fresh pilot
    # ROUND is a new regime), a P4 relaunch resumes the SAME regime, and the
    # StageLedger's fingerprint fail-loud is the designed guard against a
    # regime-changed rerun landing here (wipe or re-root explicitly then).
    out_root = Path(args.store_root)
    # dry mode runs on the GPU-less VM: log with a placeholder CVD pin; a real
    # pod with zero visible GPUs still fails loud inside Runner.parallel.
    cvd_pins = gpus if gpus else (["0"] if runner.dry else [])
    # parallel capture fan-out (one HF model per GPU, cells sharded via --cells)
    capture_argvs = [
        _model_py(
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
            _model_py(
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
        _model_py(
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
                "[dry] p6.summaries: wait siblings -> stage sidecars -> pool/h5/h4b/ratio "
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
        # p6.h3 (question-vs-dialogue contrast) REMOVED at v7: the dialogue
        # family is descoped (epm:progress v70 clause 1) — ladder.phase_h3 is
        # tombstoned and would refuse anyway.
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
        # g2b-user-drop-crashes-h4a-ratio; precedence hardened by the r3
        # blocker g2b-drop-marker-shadowed-by-stale-fit): fits-d writes the
        # per-cell __g2b_dropped.json markers above BEFORE this call, and
        # phase_ratio checks the drop marker FIRST — a coexisting stale
        # git-re-materialized fit never resurrects a dropped arm — with
        # --survivors keying marker authority to THIS dispatch's survivor set
        # (a stale prior-run marker on a survivor is ignored). A dropped
        # user arm yields a loud per-arm N/A
        # entry (whole-file N/A when both drop), while a missing fit for a
        # SURVIVOR still hard-raises.
        runner.run(
            "p6.ratio",
            _py("issue2378_fits.py", "--phase", "ratio", "--survivors", surv_arg, *store),
        )
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
                # r10 (G1 accounting fix, epm:progress v63): storyq_astra's
                # sega/segb summaries mirror the REAL r<10 writer payload — NO
                # 'cell' key, cell only in the FILENAME — so pooling it relies
                # on the composer's filename fallback (fails against the
                # pre-r10 payload-only aggregator, which keyed it 'sega'/'segb'
                # and CELL_FAMILY dropped it -> net 0.0). dialog_astra keeps
                # the payload-key form (the r10 writers' shape).
                payload_cell = {} if cell == "storyq_astra" else {"cell": cell}
                (raw / "sega").mkdir(parents=True, exist_ok=True)
                cm.atomic_write_json(
                    raw / "sega" / f"summary_{cell}_w1_s0.json",
                    {
                        **payload_cell,
                        "counts": {"attempts": 100, "kept": mining_kept, "cap_hit": 0},
                    },
                )
                (raw / "segb").mkdir(parents=True, exist_ok=True)
                cm.atomic_write_json(
                    raw / "segb" / f"summary_{cell}_w1_s0.json",
                    {**payload_cell, "counts": {"rows": 50, "kept": 45, "cap_hit_no_close": 2}},
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
            # r10: per-family pools NON-EMPTY for both gen stages, and per-cell
            # mining buckets carry CELL names — never the stage-dir fallback
            # name 'sega'/'segb' (the G1 accounting bug's signature; the
            # storyq_astra no-'cell'-key fixture rows exercise the filename
            # fallback that repairs the already-written pod-side summaries).
            assert set(d["per_stage"]["mining"]) == {"question", "dialogue"}, d["per_stage"]
            assert set(d["per_stage"]["segb_survival"]) == {"question", "dialogue"}, d["per_stage"]
            assert set(d["per_cell"]["mining"]) == {"storyq_astra", "dialog_astra"}, d["per_cell"]
            assert set(d["per_cell"]["segb"]) == {"storyq_astra", "dialog_astra"}, d["per_cell"]
            # v7: the GATE iterates ACTIVE families only — dialogue is pooled
            # for the record (per_stage above) but absent from the predicate.
            assert set(d["families"]) == {"question"}, d["families"]
            # net = 0.6 * (40/60) * 0.9 = 0.36; projected 30000*0.36 >= 6500;
            # sizing = ceil(8000*1.25/0.36) (formula unchanged at v7)
            fam = d["families"]["question"]
            assert abs(fam["net_kept_per_attempt"] - 0.6 * (40 / 60) * 0.9) < 1e-9
            assert fam["floor_kept"] == cm.FLOOR_KEPT
            assert (
                abs(fam["projected_kept_at_cap"] - SEGA_ATTEMPTS_CAP * fam["net_kept_per_attempt"])
                < 1e-6
            )
            assert fam["wave1_attempts_per_cell"] == math.ceil(
                8000 * 1.25 / fam["net_kept_per_attempt"]
            )
            assert d["fences_s_2x"]["p1.sega"] == 200.0

        check("G1 composer PASS branch + wave sizing", t_g1_pass)

        def t_g1_amended_band():
            # v7 DISCRIMINATING fixture (Amendment record B): mining 38/100 ->
            # net = 0.38 * (40/60) * 0.9 = 0.228 — BELOW the pre-v7 0.25 line,
            # ABOVE the amended floor-funding line 6500/30000 ~= 0.21667.
            # Amended gate: PASS (projected 6840 >= 6500); the old line FAILs.
            raw, ledger = _mk_pilot_fixture(tmp / "g1band", mining_kept=38)
            d = compose_pilot_digest(raw, ledger, {}, pilot_round=2, attempts_per_cell=300)
            fam = d["families"]["question"]
            assert G1_NET_RATE_MIN < fam["net_kept_per_attempt"] < 0.25, fam
            assert fam["projected_kept_at_cap"] >= cm.FLOOR_KEPT
            assert d["verdict"] == "PASS", d["fail_reasons"]
            assert abs(G1_NET_RATE_MIN - cm.FLOOR_KEPT / SEGA_ATTEMPTS_CAP) < 1e-12

        check("G1 amended floor-funding band (v7: PASS below the old 0.25 line)", t_g1_amended_band)

        def t_g1_trip():
            raw, ledger = _mk_pilot_fixture(tmp / "g1trip", mining_kept=20)
            d = compose_pilot_digest(raw, ledger, {}, pilot_round=1, attempts_per_cell=300)
            # net = 0.2 * (40/60) * 0.9 = 0.12 -> projected 3600 < 6500: FAIL
            # under the amended line too.
            assert d["verdict"] == "FAIL"
            assert all(r.startswith("G1(a)") for r in d["fail_reasons"])
            assert "floor" in d["fail_reasons"][0], d["fail_reasons"]
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
            assert out["verdict"] == "PASS"  # chat + plain + 4 storyQ survive (v7 predicate)
            assert out["story_q_survivors"] == 4 and out["dialog_survivors"] == 0
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
                    "import os,sys;print('CVD='+os.environ['CUDA_VISIBLE_DEVICES']);"
                    "print('PIN='+os.environ.get('VLLM_USE_FLASHINFER_SAMPLER',''))",
                ],
                gpus=["6", "7"],
            )
            for i, g in enumerate(["6", "7"]):
                log = (tmp / "logs" / f"probe.fan.s{i}.log").read_text(encoding="utf-8")
                assert f"CVD={g}" in log, f"shard {i} CVD pin missing"
                # r8: the sampler-probe env pin rides EVERY shard env
                assert "PIN=0" in log, f"shard {i} VLLM_USE_FLASHINFER_SAMPLER pin missing"
            # parallel(): pre-composed per-shard argvs (the P4 capture shape)
            probe_argv = [
                sys.executable,
                "-c",
                "import os;print('CVD='+os.environ['CUDA_VISIBLE_DEVICES']);"
                "print('PIN='+os.environ.get('VLLM_USE_FLASHINFER_SAMPLER',''))",
            ]
            r.parallel("probe.par", [list(probe_argv), list(probe_argv)], gpus=["2", "3"])
            for i, g in enumerate(["2", "3"]):
                log = (tmp / "logs" / f"probe.par.s{i}.log").read_text(encoding="utf-8")
                assert f"CVD={g}" in log, f"parallel shard {i} CVD pin missing"
                assert "PIN=0" in log, f"parallel shard {i} VLLM_USE_FLASHINFER_SAMPLER pin missing"
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
            cells = {r["cell"] for r in rows}
            # v7: ACTIVE families only — the archival dialog_dana rows in the
            # mixed mined fixture are SKIPPED, never balanced against.
            assert n == 40 and fams == {"question"}, (n, fams)
            assert cells == {"storyq_astra", "storyq_vex"}, cells

        check("balanced admission slice (active families only, v7)", t_slice)

        def t_pilot_round_scope():
            base_raw = tmp / "praw"
            r1 = Runner(tmp / "plogs" / "p1_pilot", resume=True, dry=True)
            raw1, run1, pref1, store1 = _pilot_round_scope(base_raw, r1, 1)
            assert raw1 == base_raw and run1 is r1
            assert pref1 == f"{cm.HF_PREFIX}/raw_completions/pilot"
            # round 1 keeps the STABLE capture store (pre-fix byte parity)
            assert store1 == cm.PILOT_STORE_DEFAULT
            raw2, run2, pref2, store2 = _pilot_round_scope(base_raw, r1, 2)
            assert raw2 == base_raw / "r2"
            assert run2.logs_dir == r1.logs_dir / "p1_pilot_r2"
            assert pref2 == f"{cm.HF_PREFIX}/raw_completions/pilot/r2"
            # r12 out-root fix: round-2 capture store DISJOINT from round 1's
            # (the r2 crash: capture resumed into round 1's StageLedger) —
            # sibling dir, never nested inside the round-1 store.
            assert store2 != store1, (store1, store2)
            assert store2 == store1.parent / f"{store1.name}_r2"
            assert store1 not in store2.parents
            # round-2 resume keys are DISJOINT from round 1's: a round-1
            # ok-flag must be invisible to the round-2 Runner (g5 blocker 2)
            run1._ok_path("p1.sega").write_text("sha")
            assert not run2._ok_path("p1.sega").exists()

        check(
            "pilot round-2 scope (fresh resume key + raw root + HF prefix + capture store)",
            t_pilot_round_scope,
        )

        def t_p1r_staging():
            # P1R staging count-assert (plan §4.7 step (a)): exact-set expected
            # counts, fail-loud on any mismatch, idempotent accept on an
            # already-staged root. Fixture builds the dest dirs DIRECTLY (no
            # network) — the HF pull branch is exercised pod-side.
            root = tmp / "p1r_raw"
            for st, n in _P1R_R2_STAGE_COUNTS.items():
                d = root / st
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    (d / f"f{i:04d}.json").write_text("{}", encoding="utf-8")
            counts = _p1r_stage_r2_raw(root, "unused/prefix")  # all present -> no pull
            assert sum(counts.values()) == 331 and counts == _P1R_R2_STAGE_COUNTS
            # planted mismatch: one extra file in segb -> loud count failure
            # (verify helper directly — the stage entrypoint's missing-set
            # branch would attempt an HF pull, and probes stay offline).
            (root / "segb" / "extra.json").write_text("{}", encoding="utf-8")
            try:
                _p1r_verify_counts(root)
                raise AssertionError("planted segb count mismatch must raise")
            except RuntimeError as e:
                assert "segb: 145 != 144" in str(e), e

        check("P1R staging exact-set count assert (idempotent + fail-loud)", t_p1r_staging)

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
            # r2 review concern local-raw-stage-completeness-unchecked + r3
            # reconciler concern rows-dir-mismatch-fallthrough-stale-mirror:
            # on the --stage-raw-from-hf path a nonempty local dir is accepted
            # ONLY when it covers the remote (path, size) manifest, and the
            # fall-through mirror restage REPAIRS stale mirror bytes
            # (delete-then-restage — hub staging is skip-existing) then
            # fail-loud verifies the leaf. Fakes sit at the NETWORK boundary
            # ONLY (hub.list_hf_entries_under_path + hub.stage_hub_prefix, the
            # latter reproducing the real per-file SKIP-EXISTING semantics of
            # stage_hub_file overwrite=False); cm.stage_hf_prefix and gen's
            # repair/verify logic REALLY run over real tiny files, so scenario
            # (5) proves REPAIRED BYTES, not a faked stage call.
            import argparse as _ap

            import issue2378_gen as gen
            from explore_persona_space.orchestrate import hub

            root = tmp / "rowsdir"
            stage_dir = root / "chat"
            stage_dir.mkdir(parents=True)
            fp = stage_dir / "chat_w1_s0_c0000.jsonl"
            fp.write_text('{"a": 1}\n', encoding="utf-8")
            mirror_root = tmp / "rowsdir_mirror"
            pre = f"{cm.HF_PREFIX}/raw_completions/chat"
            leaf = mirror_root / pre
            calls = {"list": 0, "stage": 0}
            remote_bytes: dict[str, bytes] = {}
            holdback: set[str] = set()
            wrote: list[str] = []
            real_list, real_stage = hub.list_hf_entries_under_path, hub.stage_hub_prefix
            real_root = gen.HF_STAGE_ROOT

            def fake_list(api, repo, prefix, **kw):
                calls["list"] += 1
                return [(f"{pre}/{n}", len(b)) for n, b in sorted(remote_bytes.items())]

            def fake_stage(
                repo_id,
                prefix,
                dest_dir,
                *,
                repo_type="dataset",
                revision=None,
                token=None,
                max_workers=6,
            ):
                # network-boundary fake reproducing the real skip-existing
                # per-file semantics (an existing target is NEVER rewritten).
                calls["stage"] += 1
                wrote.clear()
                tgt = Path(dest_dir) / prefix
                tgt.mkdir(parents=True, exist_ok=True)
                staged = []
                for n, b in sorted(remote_bytes.items()):
                    t = tgt / n
                    if not t.exists() and n not in holdback:
                        t.write_bytes(b)
                        wrote.append(n)
                    staged.append(t)
                return staged

            def _clear() -> None:
                gen._STAGE_RECON_CACHE.clear()
                gen._STAGE_MANIFEST_CACHE.clear()
                gen._STAGE_MIRROR_CACHE.clear()

            def ns(flag: bool) -> _ap.Namespace:
                return _ap.Namespace(raw_root=str(root), stage_raw_from_hf=flag)

            try:
                hub.list_hf_entries_under_path = fake_list
                hub.stage_hub_prefix = fake_stage
                gen.HF_STAGE_ROOT = mirror_root
                _clear()
                # (1) no-flag: local-first, ZERO network (no listing, no stage).
                assert gen._rows_dir(ns(False), "chat") == stage_dir
                assert calls == {"list": 0, "stage": 0}, calls
                # (2) flag + empty remote (nothing published yet): accept local.
                _clear()
                assert gen._rows_dir(ns(True), "chat") == stage_dir
                assert calls == {"list": 1, "stage": 0}, calls
                # (3) flag + matching (name+size) manifest: accept local; memoized.
                _clear()
                remote_bytes[fp.name] = fp.read_bytes()
                assert gen._rows_dir(ns(True), "chat") == stage_dir
                assert gen._rows_dir(ns(True), "chat") == stage_dir
                assert calls == {"list": 2, "stage": 0}, calls
                # (4) flag + remote superset: partial local -> REAL mirror
                # restage (real bytes at the leaf); verified-leaf memo re-call.
                _clear()
                f2 = "chat_w1_s1_c0000.jsonl"
                remote_bytes[f2] = b'{"b": 2}\n'
                assert gen._rows_dir(ns(True), "chat") == leaf
                assert calls == {"list": 3, "stage": 1}, calls
                assert (leaf / fp.name).read_bytes() == fp.read_bytes()
                assert (leaf / f2).read_bytes() == b'{"b": 2}\n'
                assert gen._rows_dir(ns(True), "chat") == leaf  # mirror memo
                assert calls == {"list": 3, "stage": 1}, calls
                # (5) r3 concern — FAILS PRE-FIX: the producer re-uploads
                # fp with changed bytes; the mirror holds the STALE 9-B copy
                # and skip-existing staging would serve it verbatim. The
                # repair deletes it, the restage rewrites the REAL new bytes.
                _clear()
                remote_bytes[fp.name] = b'{"a": 22}\n'  # 10 B vs stale 9 B
                assert gen._rows_dir(ns(True), "chat") == leaf
                assert calls == {"list": 4, "stage": 2}, calls
                assert wrote == [fp.name], wrote  # ONLY the repaired file restaged
                assert (leaf / fp.name).read_bytes() == b'{"a": 22}\n'
                # (6) extraneous mirror file (dropped remotely): repair deletes.
                _clear()
                del remote_bytes[f2]
                assert gen._rows_dir(ns(True), "chat") == leaf
                assert sorted(p.name for p in leaf.glob("*.jsonl")) == [fp.name]
                # (7) post-restage verify is fail-loud: a file the "network"
                # never delivers leaves the leaf short -> RuntimeError.
                _clear()
                remote_bytes["chat_w1_s2_c0000.jsonl"] = b'{"c": 3}\n'
                holdback.add("chat_w1_s2_c0000.jsonl")
                try:
                    gen._rows_dir(ns(True), "chat")
                    raise AssertionError("short restage did not raise")
                except RuntimeError as e:
                    assert "STILL fails" in str(e), e
            finally:
                hub.list_hf_entries_under_path = real_list
                hub.stage_hub_prefix = real_stage
                gen.HF_STAGE_ROOT = real_root
                _clear()

        check(
            "rows-dir manifest reconciliation + stale-mirror repair (r2+r3 concerns)",
            t_rows_dir_manifest,
        )

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
    # r12 P1R staging seam: cm.stage_hf_prefix defers `orchestrate.hub` —
    # execute the import + bind the exact call shape it forwards.
    from explore_persona_space.orchestrate import hub as _hub

    inspect.signature(_hub.stage_hub_prefix).bind(
        cm.HF_DATA_REPO, "prefix", Path("x"), repo_type="dataset", revision=None
    )
    # r8 engine_smoke seam: create_vllm_engine resolves on the repo venv (its
    # own vllm import is deferred) — bind the gate's exact call shape. The
    # `from vllm import SamplingParams` / EngineArgs deferred imports inside
    # phase_engine_smoke target the MODEL venv's vllm 0.27.1 (same convention
    # as env_smoke's compile-backend import, r7 marker) and execute pod-side
    # at the ensure gate itself, pre-fan-out.
    from explore_persona_space.eval.generation import create_vllm_engine

    inspect.signature(create_vllm_engine).bind(
        cm.MODEL_ID,
        max_model_len=ENGINE_SMOKE_MAX_MODEL_LEN,
        max_num_seqs=ENGINE_SMOKE_MAX_NUM_SEQS,
        seed=cm.SEED,
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
    )
    _log("[import-check] OK (argcheck + deferred imports + seam binds)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PHASES = {
    "env_smoke": None,  # handled inline (no Runner needed)
    "engine_smoke": None,  # inline (r8 gate; r10 D1: standalone entry re-dispatches via model venv)
    "model_venv": phase_model_venv,
    "p0_banks_pools": phase_p0,
    "p1_pilot": phase_p1,
    "p1_resume": phase_p1_resume,
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
    if args.phase == "engine_smoke":
        if _is_model_interpreter():
            return phase_engine_smoke(args)  # never returns on success (os._exit)
        # r10 D1: standalone repo-venv entry — re-dispatch through the SAME
        # model-env composition the fan-out legs use (never in-process here).
        rc = _run_engine_smoke_gate(args)
        if rc == 0:
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
