"""#2544 ladder / config / pins layer (Olmo-3-7B 15-rung pretraining stage map).

Widens the #1902 rig to the 15-rung Olmo-3 ladder WITHOUT forking it:

- **Import-order contract:** importing THIS module sets
  ``EPM_ISSUE1902_LADDER_JSON`` (-> ``configs/issue2544_ladder.json``) and
  ``EPM_ISSUE1902_HF_WRITE_PREFIX`` (-> ``issue2544_stage_map``) via
  ``os.environ.setdefault`` BEFORE importing ``issue1902_common``, so the
  widened ``CKPTS``/``MODEL_IDS``/``MODEL_BRANCHES``/``PLAIN_RENDER_CKPTS``
  bind and every #1902 write-prefix helper lands under the #2544 prefix.
  Always ``import issue2544_common`` BEFORE ``issue1902_run`` in a process
  (importing the run module first would freeze the un-widened constants).
  Corpus READS are untouched (#1902's corpus is reused verbatim —
  ``C.CORPUS_HF_PATH`` keys on the read prefix, not the write prefix).
- **M1 — rung-residency-bounded unit queue** (:class:`UnitQueue` +
  :func:`ensure_snapshot`/:func:`reap_snapshot`): (rung x unit) tasks are
  claimed work-conservingly by N workers, a NEW rung is admitted only while
  resident-rung count < ``K_RESIDENT`` (= 4; a file-lock counting semaphore
  around ``snapshot_download``), and a rung's snapshot revision reaps when
  its unit set drains, freeing the slot (8 resident 14 GB snapshots would
  breach the ~130 GB MooseFS per-pod quota invisible to shutil.disk_usage).
- **M2 — phase-specific TRANSITIVE resume fingerprints**
  (:func:`build_fingerprint`): per-phase EXACT field sets; a fingerprint
  referencing an artifact outside the phase's declared input set (e.g. a
  pass-1 unit naming ``intersection_sha``/``freeze_sha``) RAISES — the DAG
  refusal. Artifact hashes are recorded ONCE at write time (`.sha256`
  sidecars) and consumed downstream — never recomputed floats (#1336).
- **M3 — exemplar-bank content fitness** (:func:`stream_exemplar_pool` +
  :func:`select_exemplar_bank`): pool-then-template selection under the
  registered composition template (2 generic + 1 math + 1 code per set;
  cluster- and answer-form-diverse; sets pairwise cluster-disjoint; spares
  6/3/3; nested k1 in k4 in k16 recorded).
- **A1 — sliding-window package** (:data:`OLMO3_FULL_ATTENTION_LAYERS`,
  :func:`nearest_full_attention_layer`, :func:`over_window_fraction`).

Content hygiene: LMSYS is unscreened real user text — no function here may
print/log corpus/exemplar row text; digests are ids + counts + hashes only.

VM-side launches carry the shared-VM thread-cap prefix (#847/#891)::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2544_run.py --phase config
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ISSUE = 2544
RECIPE_VERSION = "issue2544-run-v1"
LADDER_JSON_PATH = PROJECT_ROOT / "configs" / "issue2544_ladder.json"

# MUST precede the issue1902_common import (module-level constants bind at
# import). setdefault: an explicit env (smoke ladder / smoke write prefix
# `issue2544_stage_map/_smoke`) always wins.
os.environ.setdefault("EPM_ISSUE1902_LADDER_JSON", str(LADDER_JSON_PATH))
os.environ.setdefault("EPM_ISSUE1902_HF_WRITE_PREFIX", "issue2544_stage_map")

import issue1902_common as C  # noqa: E402  (env-ordered import, see above)

# ── ladder tokens / rosters ──────────────────────────────────────────────────

EXPECTED_RUNGS: tuple[str, ...] = (
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "r7",
    "r8",
    "r9",
    "mid",
    "main",
    "S",
    "D",
    "R",
)
# Single weight chain (plan A18): base ladder -> midtrain -> final base ->
# SFT -> DPO -> RLVR. RUNGS mirrors whatever ladder the env loaded; under the
# canonical committed JSON it MUST equal EXPECTED_RUNGS.
RUNGS: tuple[str, ...] = C.CKPTS
if os.environ["EPM_ISSUE1902_LADDER_JSON"] == str(LADDER_JSON_PATH):
    assert RUNGS == EXPECTED_RUNGS, f"ladder drift: {RUNGS} != {EXPECTED_RUNGS}"

DOSE_RUNGS: tuple[str, ...] = ("r2", "r5", "main")
SET_RUNGS: tuple[str, ...] = ("r2", "r5", "main")
NATIVE_GEN_RUNGS: tuple[str, ...] = ("S", "D", "R")
PILOT_CAPTURE_RUNGS: tuple[str, ...] = ("r0", "r5", "main")

K_DEFAULT = 4  # exemplars in the headline k-shot arm
DOSE_KS: tuple[int, ...] = (1, 16)
SET_IDS: tuple[str, ...] = ("S1", "S2", "S3")
ORDER_IDS: tuple[str, ...] = ("O1", "O2", "O3")

MAX_MODEL_LEN = 8192  # uniform vLLM/context pin (plan §11)
K_RESIDENT = int(os.environ.get("EPM_ISSUE2544_K_RESIDENT", "4"))  # M1 admission cap
DL_JITTER_MAX_S = float(os.environ.get("EPM_ISSUE2544_DL_JITTER_S", "120"))

PILOT_N = 500
SUBSET_SIZES: dict[str, int] = {
    "pilot": PILOT_N,
    "reliability": 1000,
    "robust": 6000,
    "natgen": 2000,
}
SUBSET_STREAMS: dict[str, int] = {"pilot": 0, "reliability": 1, "robust": 2, "natgen": 3}
SUBSET_SEED = 42
FOLD_MIN_NTR = 4096  # Gate A' min-over-folds train-rows floor
ISECT_FLOOR = 4916  # Gate A/A' intersection floor (1.2d)
ISECT_TARGET = 8192  # Gate A/A' target (2d)
WIDEN_BUILD_CAP = 48_000  # Gate A branch (b) reachability cap
GATE_WALL_FACTOR = 2.0  # P1 cost re-projection abort factor

# Sliding-window package (plan A1/A2; probed 2026-08-24 at main + stage1-step0).
OLMO3_SLIDING_WINDOW = 4096
OLMO3_FULL_ATTENTION_LAYERS: tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31)

# ── HF layout (write prefix follows the smoke divert env) ────────────────────

HF_WRITE_PREFIX = os.environ["EPM_ISSUE1902_HF_WRITE_PREFIX"]
CONFIG_HF_PATH = f"{HF_WRITE_PREFIX}/config"
STORE_HF_PATH = f"{HF_WRITE_PREFIX}/analysis_tensors/issue2544_store"
# THIS issue's own write prefixes, threaded EXPLICITLY at every upload call
# site (never via the redirected issue1902 constants — the #1005
# parent-clobber class the upload-prefix lint pins). At runtime they equal
# the env-redirected C.RAW_GEN_HF_PATH / C.EVAL_MIRROR_HF_PATH, but a process
# importing issue1902_run WITHOUT this module must never be one bug away
# from writing into #1902's prefixes.
RAW_GEN_HF_PATH = f"{HF_WRITE_PREFIX}/raw_completions/gen"
EVAL_MIRROR_HF_PATH = f"{HF_WRITE_PREFIX}/eval_results_mirror"


# ── hashing / sidecars ───────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (hash-at-write-time inputs; M2)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha_sidecar(path: Path) -> str:
    """Write ``<path>.sha256`` next to a just-written artifact; returns the sha."""
    sha = sha256_file(path)
    Path(str(path) + ".sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def read_sha_sidecar(path: Path) -> str:
    """Read a write-time sha sidecar; raises if missing (never recompute-compare)."""
    sc = Path(str(path) + ".sha256")
    if not sc.exists():
        raise FileNotFoundError(
            f"sha sidecar missing: {sc} — the producing unit must call "
            "write_sha_sidecar at write time (M2 hash-once contract)"
        )
    return sc.read_text(encoding="utf-8").strip()


def sha256_json(obj: Any) -> str:
    """sha256 of canonical (sorted-keys) JSON — for fingerprint dicts."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


CODE_FINGERPRINT_FILES: tuple[str, ...] = (
    "issue2544_common.py",
    "issue2544_run.py",
    "issue1902_common.py",
    "issue1902_run.py",
)


def code_sha() -> str:
    """Content sha over the four driver files (M2 'code SHA').

    Content-hash (not git sha): resumable across unrelated repo commits,
    invalidating exactly when the generating code changes; machine-stable
    (file bytes, never recomputed floats — #1336).
    """
    h = hashlib.sha256()
    for name in CODE_FINGERPRINT_FILES:
        p = _SCRIPTS_DIR / name
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ── M2: phase-specific transitive fingerprints + DAG refusal ─────────────────

_GEN_FP_FIELDS = frozenset(
    {
        "code_sha",
        "rung",
        "revision",
        "render",
        "k",
        "order_id",
        "set_id",
        "seed",
        "sampling",
        "rows_scope",
        "exemplars_sha",
        "subsets_sha",
        "ladder_sha",
        "corpus_sha",
        "corpus_repo_revision",
    }
)
PHASE_FP_FIELDS: dict[str, frozenset[str]] = {
    # Gen units (P1 pilot gen + P2 production gen): NO forward references —
    # intersection/freeze keys are structurally absent (the DAG refusal).
    "gen": _GEN_FP_FIELDS,
    # Pass-1 capture: gen fields + the CONSUMED rollout's write-time sha.
    "capture1": _GEN_FP_FIELDS | {"rollout_sha", "layers", "pooling_keys"},
    # Pass-2 capture: binds BOTH the capturing rung pin AND the answer-source
    # pin + consumed completion sha + the layer*-freeze record + intersection
    # manifest (plan M2).
    "capture2": frozenset(
        {
            "code_sha",
            "rung",
            "revision",
            "render",
            "k",
            "order_id",
            "set_id",
            "answer_source_rung",
            "answer_source_revision",
            "rollout_sha",
            "layers",
            "pooling_keys",
            "freeze_sha",
            "intersection_sha",
            "rows_scope",
            "exemplars_sha",
            "subsets_sha",
            "ladder_sha",
            "corpus_sha",
            "corpus_repo_revision",
        }
    ),
}


def build_fingerprint(kind: str, **fields: Any) -> dict[str, Any]:
    """Validated resume fingerprint for one unit (M2).

    RAISES on (a) an undeclared field — a fingerprint referencing an artifact
    outside the phase's declared input set (e.g. ``intersection_sha`` on a
    pass-1 unit: a not-yet-produced artifact — the dispatcher DAG refusal);
    (b) a missing field — an incomplete fingerprint silently weakens resume.
    """
    if kind not in PHASE_FP_FIELDS:
        raise KeyError(f"unknown fingerprint kind {kind!r}; known: {sorted(PHASE_FP_FIELDS)}")
    allowed = PHASE_FP_FIELDS[kind]
    extra = set(fields) - allowed
    if extra:
        raise RuntimeError(
            f"DAG refusal (M2): fingerprint kind {kind!r} references undeclared "
            f"input field(s) {sorted(extra)} — an artifact outside this phase's "
            f"declared input set. Declared: {sorted(allowed)}"
        )
    missing = allowed - set(fields)
    if missing:
        raise RuntimeError(
            f"incomplete fingerprint (M2): kind {kind!r} missing field(s) {sorted(missing)}"
        )
    return {"kind": kind, **{k: fields[k] for k in sorted(fields)}}


def fingerprint_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Field-level diff between two fingerprints (legible resume refusals)."""
    keys = sorted(set(a) | set(b))
    return [f"{k}: {a.get(k)!r} != {b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]


def sampling_fingerprint(seed: int, *, plain: bool, max_tokens: int) -> dict[str, Any]:
    """The FULL sampling-config block a gen fingerprint hashes (plan M2)."""
    return {
        "engine": "vllm",
        "n": 1,
        "temperature": C.GEN_TEMPERATURE,
        "top_p": C.GEN_TOP_P,
        "max_tokens": max_tokens,
        "seed": seed,
        "stop": list(C.PLAIN_STOP_SEQUENCES) if plain else None,
        "max_model_len": MAX_MODEL_LEN,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }


# ── M1: file-locked unit queue with K-resident rung admission ────────────────

_TERMINAL = ("done", "failed", "dep_failed")


class UnitQueue:
    """File-locked (rung x unit) work queue (M1).

    State lives in ONE JSON under an ``fcntl.flock``-serialized lock file; N
    worker processes claim units work-conservingly. A rung's units become
    claimable only while it is ADMITTED; admission of a new rung requires
    resident (admitted minus reaped) count < ``k_resident``. A rung whose
    units are all terminal is handed out ONCE via :meth:`take_reapable` so
    exactly one worker reaps its snapshot.
    """

    def __init__(self, out_root: Path, phase: str, *, k_resident: int = K_RESIDENT):
        self.dir = out_root / "queue"
        self.path = self.dir / f"{phase}.json"
        self.lock_path = self.dir / f"{phase}.lock"
        self.phase = phase
        self.k_resident = k_resident

    @contextmanager
    def _locked(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text()) if self.path.exists() else None
                holder = {"state": state, "dirty": False}
                yield holder
                if holder["dirty"]:
                    tmp = self.path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(holder["state"], indent=1))
                    os.replace(tmp, self.path)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def init(self, units: list[dict[str, Any]]) -> None:
        """Create (or resume-verify) the queue.

        Each unit dict: ``{"unit", "rung", "kind", "deps": [...],
        "fingerprint": {...}}`` (+ arbitrary spec fields the executor reads).
        On resume: the existing state must hold the SAME unit-name set and,
        per unit, the SAME init-time fingerprint — any mismatch RAISES with a
        field diff (the #1333 refusal shape; fresh --out-root or a deliberate
        queue wipe are the remedies). On resume this ALSO reclaims stale
        RUNNING claims whose owner is provably dead (a SIGKILL/OOM-killed
        worker must never wedge the phase — a relaunch recovers).
        """
        names = [u["unit"] for u in units]
        if len(set(names)) != len(names):
            raise ValueError("duplicate unit names in queue init")
        fresh = False
        with self._locked() as h:
            if h["state"] is None:
                h["state"] = {
                    "phase": self.phase,
                    "k_resident": self.k_resident,
                    "unit_order": names,
                    "units": {u["unit"]: {**u, "status": "pending", "worker": None} for u in units},
                    "admitted": [],
                    "reaped": [],
                }
                h["dirty"] = True
                fresh = True
            else:
                st = h["state"]
                if set(st["units"]) != set(names):
                    raise RuntimeError(
                        f"resume REFUSED for queue {self.phase}: unit-name set changed.\n"
                        f"  missing now: {sorted(set(st['units']) - set(names))}\n"
                        f"  new now:     {sorted(set(names) - set(st['units']))}\n"
                        "Use a fresh --out-root (per-leg out-roots) or wipe the queue "
                        f"file deliberately: {self.path}"
                    )
                for u in units:
                    prior = st["units"][u["unit"]].get("fingerprint")
                    if prior != u["fingerprint"]:
                        diff = fingerprint_diff(prior or {}, u["fingerprint"])
                        raise RuntimeError(
                            f"resume REFUSED for unit {u['unit']}: fingerprint mismatch "
                            f"(M2).\n  " + "\n  ".join(diff) + "\n"
                            "Use a fresh --out-root or wipe the queue file deliberately: "
                            f"{self.path}"
                        )
        if not fresh:
            # Relaunch recovery (blocker queue-stale-running): a prior worker
            # SIGKILL'd/OOM-killed mid-unit left its claim at status=running;
            # without a reclaim the queue reads any_running()=True forever.
            self.reclaim_stale()

    def _runnable(self, st: dict) -> list[str]:
        units = st["units"]
        resident = set(st["admitted"]) - set(st["reaped"])
        out = []
        for n in st["unit_order"]:
            u = units[n]
            if u["status"] != "pending" or u["rung"] not in resident:
                continue
            dep_status = [units[d]["status"] for d in u.get("deps", [])]
            if any(s in ("failed", "dep_failed") for s in dep_status):
                continue  # swept to dep_failed below
            if all(s == "done" for s in dep_status):
                out.append(n)
        return out

    def _sweep_dep_failed(self, st: dict, h: dict) -> None:
        units = st["units"]
        for n in st["unit_order"]:
            u = units[n]
            if u["status"] == "pending" and any(
                units[d]["status"] in ("failed", "dep_failed") for d in u.get("deps", [])
            ):
                u["status"] = "dep_failed"
                h["dirty"] = True

    def claim(
        self, worker: str, *, prefer_rung: str | None = None, prefer_kind: str | None = None
    ) -> dict[str, Any] | None:
        """Claim one runnable unit (or None). Admits a new rung — deterministic
        RUNGS order — when nothing is runnable and a residency slot is free."""
        with self._locked() as h:
            st = h["state"]
            if st is None:
                raise RuntimeError(f"queue {self.phase} not initialized — run --init first")
            self._sweep_dep_failed(st, h)
            runnable = self._runnable(st)
            if not runnable:
                resident = set(st["admitted"]) - set(st["reaped"])
                if len(resident) < st["k_resident"]:
                    rung_order = [r for r in RUNGS if r not in st["admitted"]]
                    for r in rung_order:
                        if any(
                            u["rung"] == r and u["status"] == "pending"
                            for u in st["units"].values()
                        ):
                            st["admitted"].append(r)
                            h["dirty"] = True
                            print(
                                f"[queue:{self.phase}] admit rung {r} "
                                f"(resident={len(resident) + 1}/{st['k_resident']})",
                                flush=True,
                            )
                            break
                runnable = self._runnable(st)
            if not runnable:
                return None
            pick = None
            for pred in (
                lambda u: u["rung"] == prefer_rung and u["kind"] == prefer_kind,
                lambda u: u["rung"] == prefer_rung,
                lambda u: True,
            ):
                cands = [n for n in runnable if pred(st["units"][n])]
                if cands:
                    pick = cands[0]
                    break
            u = st["units"][pick]
            u["status"] = "running"
            u["worker"] = worker
            u["claimed_ts"] = time.time()
            # Owner identity for stale-claim reclaim (claimed_ts lease): the
            # claiming process IS the worker, so pid/host recorded here are
            # positively probeable by reclaim_stale.
            u["worker_pid"] = os.getpid()
            u["worker_host"] = os.uname().nodename
            h["dirty"] = True
            return dict(u)

    def mark(self, unit: str, status: str, info: dict[str, Any] | None = None) -> None:
        assert status in ("done", "failed"), status
        with self._locked() as h:
            st = h["state"]
            u = st["units"][unit]
            u["status"] = status
            u["info"] = info or {}
            u["done_ts"] = time.time()
            h["dirty"] = True

    @staticmethod
    def _owner_liveness(u: dict[str, Any]) -> tuple[str, str]:
        """('alive'|'dead'|'unknown', detail) for a RUNNING claim's owner.

        Positive-evidence probe (same host only): /proc/<pid> present AND its
        cmdline names this driver family — a pid reused by an unrelated
        process reads as dead (the cmdline identity check). A cross-host or
        pid-less claim is 'unknown' (the claimed_ts lease governs those)."""
        pid = u.get("worker_pid")
        host = u.get("worker_host")
        if pid is None or host is None:
            return "unknown", "claim predates pid/host recording"
        if host != os.uname().nodename:
            return "unknown", f"cross-host claim (owner host {host})"
        proc = Path(f"/proc/{int(pid)}")
        if not proc.exists():
            return "dead", f"owner pid {pid} gone on {host}"
        try:
            cmdline = (proc / "cmdline").read_bytes()
        except OSError:
            return "dead", f"owner pid {pid} exited mid-probe"
        if b"issue2544" not in cmdline:
            return "dead", f"owner pid {pid} reused by an unrelated process"
        return "alive", f"owner pid {pid} live"

    def reclaim_stale(self) -> list[str]:
        """Reclaim RUNNING units whose owner is provably dead (M1; blocker
        queue-stale-running): status back to pending so a live worker — or a
        relaunch — re-runs them instead of the phase wedging on a phantom
        claim. Owner-dead = the same-host positive pid/cmdline probe above; a
        cross-host / pid-less claim falls back to the hard claimed_ts lease
        ``EPM_ISSUE2544_CLAIM_LEASE_S`` (default 14400 s — sized above the
        longest healthy unit wall; ``0`` disables the lease arm). A provably
        LIVE owner is never reclaimed regardless of lease age. Returns the
        reclaimed unit names; an audit trail rides each unit's ``reclaims``."""
        lease_s = float(os.environ.get("EPM_ISSUE2544_CLAIM_LEASE_S", "14400"))
        with self._locked() as h:
            st = h["state"]
            if st is None:
                return []
            out: list[str] = []
            now = time.time()
            for n in st["unit_order"]:
                u = st["units"][n]
                if u["status"] != "running":
                    continue
                verdict, detail = self._owner_liveness(u)
                if verdict == "alive":
                    continue
                if verdict == "unknown":
                    age = now - u.get("claimed_ts", now)
                    if not (lease_s > 0 and age > lease_s):
                        continue
                    detail = f"{detail}; claim lease expired ({age:.0f}s > {lease_s:.0f}s)"
                print(f"[queue:{self.phase}] RECLAIM stale claim {n}: {detail}", flush=True)
                u.setdefault("reclaims", []).append(
                    {
                        "from_worker": u.get("worker"),
                        "from_pid": u.get("worker_pid"),
                        "reason": detail,
                        "ts": now,
                    }
                )
                u["status"] = "pending"
                u["worker"] = None
                h["dirty"] = True
                out.append(n)
            return out

    def revalidate_done(self, check) -> list[str]:
        """Re-validate DONE units against on-disk consumed-artifact identity
        (M2 hash-once; blocker capture-rollout-fingerprint). ``check`` maps a
        unit-state dict -> None (valid) or a reason string; a reason resets
        the unit to pending (worker/info cleared) so it re-runs — a done unit
        whose consumed artifact changed on disk must never vouch. Returns the
        invalidated unit names; an audit trail rides ``invalidations``."""
        with self._locked() as h:
            st = h["state"]
            if st is None:
                raise RuntimeError(f"queue {self.phase} not initialized — run --init first")
            invalidated: list[str] = []
            n_done = 0
            for n in st["unit_order"]:
                u = st["units"][n]
                if u["status"] != "done":
                    continue
                n_done += 1
                reason = check(u)
                if reason is None:
                    continue
                print(f"[queue:{self.phase}] INVALIDATE done unit {n}: {reason}", flush=True)
                u.setdefault("invalidations", []).append({"reason": reason, "ts": time.time()})
                u["status"] = "pending"
                u["worker"] = None
                u.pop("info", None)
                u.pop("done_ts", None)
                invalidated.append(n)
                h["dirty"] = True
            print(
                f"[queue:{self.phase}] revalidate_done: {n_done} done unit(s) checked, "
                f"{len(invalidated)} invalidated",
                flush=True,
            )
            return invalidated

    def take_reapable(self) -> list[str]:
        """Admitted rungs whose unit set has drained — handed out ONCE."""
        with self._locked() as h:
            st = h["state"]
            out = []
            for r in st["admitted"]:
                if r in st["reaped"]:
                    continue
                if all(u["status"] in _TERMINAL for u in st["units"].values() if u["rung"] == r):
                    st["reaped"].append(r)
                    out.append(r)
                    h["dirty"] = True
            return out

    def all_terminal(self) -> bool:
        with self._locked() as h:
            return all(u["status"] in _TERMINAL for u in h["state"]["units"].values())

    def any_running(self) -> bool:
        with self._locked() as h:
            return any(u["status"] == "running" for u in h["state"]["units"].values())

    def failed_units(self) -> dict[str, Any]:
        with self._locked() as h:
            return {
                n: u.get("info", {})
                for n, u in h["state"]["units"].items()
                if u["status"] in ("failed", "dep_failed")
            }

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as h:
            return json.loads(json.dumps(h["state"]))


# ── plan §9:345 out-root headroom floors ─────────────────────────────────────
# Recalibrated to the implemented K=4 interleaving (blocker
# headroom-floors-vs-plan-s9, replacing the inherited 2.5 GB x N_units formula
# and the 24 GB fits floor): pass-phase preambles (P1/P2+P3a/P3b) assert
# >= ~100 GB WRITABLE (plan §9:345 — high-water ~94 GB vs the ~130 GB RunPod
# MooseFS per-pod quota); fit-phase preambles (P4a/P4b) assert >= 40 GB
# (~1.5x the realized ~26 GB worst K-resident staging wave — the reconciler
# recalibration superseding the plan's 20 GB fit floor UPWARD). The floor is
# probed as WRITABLE bytes via the fallocate canary AT THE FLOOR SIZE: on
# MooseFS the statvfs free is share-level (TB) and vacuous against the
# per-pod quota, so the fallocate probe is the binding check there (plan
# §9:345 "fallocate probe"; the same assert positively probes the fellows
# lane). assert_out_root_headroom sweeps stale probe survivors at entry
# (#2346), so an interrupted floor-size probe self-heals.
PASS_PHASE_HEADROOM_GB = 100.0
FITS_PHASE_HEADROOM_GB = 40.0
SMOKE_HEADROOM_GB = 5.0


def headroom_floor_gate(out_root: Path, phase_kind: str, *, smoke: bool) -> None:
    """Assert the plan §9:345 out-root floor for ``phase_kind`` ('pass'|'fits').

    statvfs free >= floor at the RESOLVED filesystem (df -P semantics) PLUS a
    floor-size posix_fallocate canary (writable-bytes proof — the only check
    MooseFS quota exposes). Under --smoke the floor downgrades to
    ``SMOKE_HEADROOM_GB`` with a logged line (gate-calibration parity: a
    production-scale floor on the tiny smoke slice measures the host, not the
    run; the gate COMPUTATION is identical)."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    floor = {"pass": PASS_PHASE_HEADROOM_GB, "fits": FITS_PHASE_HEADROOM_GB}[phase_kind]
    if smoke:
        print(
            f"[smoke-downgrade] [disk-headroom] issue2544-{phase_kind}: floor "
            f"{SMOKE_HEADROOM_GB:.0f} GB (production floor {floor:.0f} GB; plan §9:345)",
            flush=True,
        )
        floor = SMOKE_HEADROOM_GB
    assert_out_root_headroom(out_root, floor, phase=f"issue2544-{phase_kind}", canary_gb=floor)
    print(
        f"[disk-headroom] issue2544-{phase_kind}: >= {floor:.0f} GB writable at "
        f"{out_root} (statvfs + fallocate probe; plan §9:345)",
        flush=True,
    )


def _dl_state_dir(out_root: Path) -> Path:
    d = out_root / "residency"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_snapshot(rung: str, pins: dict[str, str], out_root: Path) -> None:
    """Download ``rung``'s pinned snapshot ONCE (per-rung flock; cold download
    jittered 0-120 s — plan §9 same-repo fan-out shape). Local-dir model ids
    (smoke remap) and ``local:`` pins skip download entirely."""
    mid = C.MODEL_IDS[rung]
    pin = pins[rung]
    if Path(mid).is_dir() or str(pin).startswith("local:"):
        return
    sdir = _dl_state_dir(out_root)
    marker = sdir / f"dl_{rung}.done"
    with open(sdir / f"dl_{rung}.lock", "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if marker.exists():
                return
            import random

            from huggingface_hub import snapshot_download

            jitter = random.uniform(0, DL_JITTER_MAX_S)
            print(f"[residency] {rung}: cold snapshot_download (jitter {jitter:.0f}s)", flush=True)
            time.sleep(jitter)
            t0 = time.time()
            snapshot_download(repo_id=mid, revision=pin)
            marker.write_text(pin + "\n")
            print(f"[residency] {rung}: snapshot ready ({time.time() - t0:.0f}s)", flush=True)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def reap_snapshot(rung: str, pins: dict[str, str], out_root: Path) -> str:
    """Delete ``rung``'s pinned snapshot revision from the HF cache (M1 reap:
    frees the residency slot's disk). Revision-scoped — sibling rungs' blobs
    survive. Returns a one-word disposition (logged, never silent)."""
    mid = C.MODEL_IDS[rung]
    pin = pins[rung]
    if Path(mid).is_dir() or str(pin).startswith("local:"):
        print(f"[residency] {rung}: reap skip-local", flush=True)
        return "skip-local"
    from huggingface_hub import scan_cache_dir

    try:
        info = scan_cache_dir()
    except Exception as e:  # CacheNotFound: nothing resident — already free
        print(f"[residency] {rung}: reap absent ({type(e).__name__})", flush=True)
        return "absent"
    revs = [
        rev.commit_hash for repo in info.repos for rev in repo.revisions if rev.commit_hash == pin
    ]
    if not revs:
        print(f"[residency] {rung}: reap absent (revision {pin[:10]} not cached)", flush=True)
        return "absent"
    strategy = info.delete_revisions(*revs)
    freed = strategy.expected_freed_size
    strategy.execute()
    marker = _dl_state_dir(out_root) / f"dl_{rung}.done"
    if marker.exists():
        marker.unlink()
    print(
        f"[residency] {rung}: reaped revision {pin[:10]} (freed {freed / 1e9:.1f} GB)", flush=True
    )
    return "reaped"


# ── pinned subsets (P0) ──────────────────────────────────────────────────────


def stratified_subset(rows: list[dict], n: int, label: str) -> list[str]:
    """Deterministic seed-42 class-stratified draw of ``n`` row ids.

    Largest-remainder proportional allocation over ``row['class']`` strata
    (deterministic tie-break: larger fraction, then class name), one seeded
    permutation per class (rng stream keyed by the subset label), selected
    ids returned in corpus order within class."""
    import numpy as np

    if label not in SUBSET_STREAMS:
        raise KeyError(f"unknown subset label {label!r}; known: {sorted(SUBSET_STREAMS)}")
    assert 0 < n <= len(rows), (n, len(rows))
    rng = np.random.default_rng([SUBSET_SEED, SUBSET_STREAMS[label]])
    by_class: dict[str, list[str]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["id"])
    classes = sorted(by_class)
    total = len(rows)
    quota = {c: n * len(by_class[c]) / total for c in classes}
    base = {c: int(math.floor(quota[c])) for c in classes}
    rem = n - sum(base.values())
    for c in sorted(classes, key=lambda c: (-(quota[c] - base[c]), c))[:rem]:
        base[c] += 1
    ids: list[str] = []
    for c in classes:
        pool = by_class[c]
        take = base[c]
        assert take <= len(pool), (label, c, take, len(pool))
        idx = sorted(rng.permutation(len(pool))[:take].tolist())
        ids.extend(pool[i] for i in idx)
    assert len(ids) == n, (len(ids), n)
    return ids


# ── M3: exemplar bank (pool -> registered composition template) ──────────────

# Bounded fresh LMSYS stream pass (fixed stop). MEASURED (Unit C smoke probe,
# 2026-08-24, production filters + the 18k-corpus dedup set): the quotas below
# fill at 34,834 scanned rows (math/code are the scarce classes — rejects
# {"in_corpus": 13585, "not_single_turn": 8235, "language": 7756, ...}); the
# original 10_000 stop would have stranded the pool at ~15 math / ~11 code and
# crashed P0 at bank selection. The stream stops EARLY once quotas fill, so
# the bound only pays when classes are rare. 60_000 ≈ 1.7× the measured fill.
EXEMPLAR_SCAN_CAP = 60_000
EXEMPLAR_POOL_QUOTAS: dict[str, int] = {"generic": 120, "math": 40, "code": 40}
EXEMPLAR_ANS_MIN_TOKENS = 30
EXEMPLAR_ANS_MAX_TOKENS = 180
EXEMPLAR_PAIR_MAX_TOKENS = 180  # rendered q+a turn pair; k16 block <= 2,880
EXEMPLAR_SHORT_MAX_TOKENS = 50
SET_TEMPLATE: dict[str, int] = {"generic": 2, "math": 1, "code": 1}
SPARE_TEMPLATE: dict[str, int] = {"generic": 6, "math": 3, "code": 3}
ANSWER_FORMS: tuple[str, ...] = ("short", "prose", "structured")
EXEMPLAR_SEED = 42

# Deterministic class split of the exemplar pool (documented operationalization
# of the composition template's math/quantitative vs coding strata; code takes
# precedence — code queries often carry quantitative words too).
CODE_KEYWORDS: tuple[str, ...] = (
    "python",
    "javascript",
    "typescript",
    "java ",
    "c++",
    "sql",
    "regex",
    "dataframe",
    "numpy",
    "pandas",
    "write a function",
    "write code",
    "write a program",
    "write a script",
    "debug",
    "compile",
    "algorithm",
    "recursion",
    "api call",
    "unit test",
    "stack trace",
    "code snippet",
)
MATH_KEYWORDS: tuple[str, ...] = (
    "equation",
    "solve for",
    "integral",
    "derivative",
    "theorem",
    "proof",
    "matrix",
    "probability",
    "algebra",
    "geometry",
    "calculate",
    "arithmetic",
    "how many",
    "math problem",
    "percentage",
    "fraction",
    "logarithm",
    "polynomial",
)

_STRUCTURED_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)]|step \d+)", re.IGNORECASE)
# Render-structure guard: an exemplar whose text contains label-shaped lines
# would corrupt the plain multi-turn serialization / stop sequences.
_LABEL_RE = re.compile(r"(?m)^(?:User|Assistant):")


def sha16(text: str) -> str:
    """sha256[:16] — the corpus builder's query-dedup convention."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def classify_exemplar(text: str) -> str:
    low = text.lower()
    if any(k in low for k in CODE_KEYWORDS) or "```" in text:
        return "code"
    if any(k in low for k in MATH_KEYWORDS):
        return "math"
    return "generic"


def answer_form(answer: str, n_tokens: int) -> str:
    """Deterministic answer-FORM partition (M3 diversity rule):
    structured (list/step/code markers) > short (<=50 tok) > prose."""
    if _STRUCTURED_RE.search(answer) or "```" in answer:
        return "structured"
    if n_tokens <= EXEMPLAR_SHORT_MAX_TOKENS:
        return "short"
    return "prose"


def _normalize_single_turn(conv_raw: Any) -> tuple[str, str] | None:
    """(query, answer) for an exactly-single-turn user->assistant conversation."""
    if not isinstance(conv_raw, list) or len(conv_raw) != 2:
        return None
    a, b = conv_raw
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return None
    if str(a.get("role", "")).lower() != "user" or str(b.get("role", "")).lower() != "assistant":
        return None
    q, ans = str(a.get("content", "")).strip(), str(b.get("content", "")).strip()
    if not q or not ans:
        return None
    return q, ans


def rendered_pair_tokens(tokenizer, query: str, answer: str) -> int:
    """Token count of the RENDERED exemplar turn pair (rendered-length budget
    discipline — raw-token counts under-count the serialized block)."""
    pair = f"{'User: '}{query}\n{'Assistant: '}{answer}\n"
    return len(tokenizer.encode(pair, add_special_tokens=False))


def stream_exemplar_pool(
    tokenizer,
    corpus_sha16s: set[str],
    *,
    scan_cap: int = EXEMPLAR_SCAN_CAP,
    quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bounded fresh deterministic LMSYS stream pass -> eligible exemplar pool.

    Eligibility (plan §4 P0/M3): single-turn; English; NOT in the 18k corpus
    by sha16 query dedup (and deduped within-pool); original assistant answer
    30-180 tokens; rendered q+a pair <= 180 tokens; no label-shaped lines
    (render-structure guard). Fixed stop: quotas filled OR ``scan_cap`` rows
    scanned (a short bounded fetch — the checkpoint-per-chunk presumption's
    named exemption). Returns (pool, stats); never prints row text.
    """
    from datasets import load_dataset

    quotas = dict(quotas or EXEMPLAR_POOL_QUOTAS)
    ds = load_dataset(C.LMSYS_DATASET, split="train", streaming=True, revision=C.LMSYS_REVISION)
    pool: list[dict[str, Any]] = []
    counts = {c: 0 for c in quotas}
    rejects: dict[str, int] = {}
    seen: set[str] = set()
    scanned = 0

    def _rej(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    for row in ds:
        if scanned >= scan_cap or all(counts[c] >= quotas[c] for c in quotas):
            break
        scanned += 1
        if str(row.get("language", "")) != C.LANG_FILTER:
            _rej("language")
            continue
        norm = _normalize_single_turn(row.get("conversation"))
        if norm is None:
            _rej("not_single_turn")
            continue
        query, answer = norm
        h = sha16(query)
        if h in corpus_sha16s:
            _rej("in_corpus")
            continue
        if h in seen:
            _rej("dup_in_pool")
            continue
        if _LABEL_RE.search(query) or _LABEL_RE.search(answer):
            _rej("label_shaped_line")
            continue
        n_ans = len(tokenizer.encode(answer, add_special_tokens=False))
        if not (EXEMPLAR_ANS_MIN_TOKENS <= n_ans <= EXEMPLAR_ANS_MAX_TOKENS):
            _rej("answer_len")
            continue
        n_pair = rendered_pair_tokens(tokenizer, query, answer)
        if n_pair > EXEMPLAR_PAIR_MAX_TOKENS:
            _rej("pair_len")
            continue
        cls = classify_exemplar(query)
        if counts[cls] >= quotas[cls]:
            _rej(f"quota_{cls}")
            continue
        seen.add(h)
        counts[cls] += 1
        pool.append(
            {
                "conversation_id": str(row.get("conversation_id", f"scan_{scanned}")),
                "sha16": h,
                "query": query,
                "answer": answer,
                "n_tokens_answer": n_ans,
                "n_tokens_pair": n_pair,
                "class": cls,
                "form": answer_form(answer, n_ans),
                "scan_index": scanned,
            }
        )
    stats = {
        "scanned": scanned,
        "scan_cap": scan_cap,
        "quotas": quotas,
        "counts": counts,
        "rejects": rejects,
        "lmsys_revision": C.LMSYS_REVISION,
    }
    return pool, stats


def assign_pool_clusters(pool: list[dict[str, Any]], centroids: list[list[float]]) -> None:
    """Nearest-centroid assignment of pool queries into the corpus's k=40
    cluster space (existing MiniLM embedding; M3 diversity rule). In place."""
    import numpy as np

    from sentence_transformers import SentenceTransformer

    cent = np.asarray(centroids, dtype=np.float32)
    assert cent.ndim == 2 and cent.shape[0] == C.K_CLUSTERS, cent.shape
    model = SentenceTransformer(C.EMBED_MODEL_ID, device="cpu")
    emb = model.encode(
        [p["query"] for p in pool],
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    assert emb.shape == (len(pool), cent.shape[1]), (emb.shape, cent.shape)
    d2 = ((emb[:, None, :] - cent[None, :, :]) ** 2).sum(-1)  # (n, k)
    labels = d2.argmin(axis=1)
    for p, lab in zip(pool, labels.tolist(), strict=True):
        p["cluster"] = int(lab)


def _pick_set(
    pools: dict[str, list[dict]],
    used_clusters: set[int],
    used_ids: set[str],
    template: dict[str, int],
) -> list[dict] | None:
    """One template-conforming exemplar set: distinct clusters (also vs
    ``used_clusters``), all three answer forms covered. Deterministic nested
    scan over the (seeded-permutation) class pools; None when infeasible."""
    slots: list[str] = []
    for cls in ("math", "code", "generic"):
        slots.extend([cls] * template.get(cls, 0))

    def _elig(cls: str, taken_cl: set[int]) -> list[dict]:
        return [
            p
            for p in pools[cls]
            if p["sha16"] not in used_ids
            and p["cluster"] not in used_clusters
            and p["cluster"] not in taken_cl
        ]

    def _search(i: int, chosen: list[dict], taken_cl: set[int]) -> list[dict] | None:
        if i == len(slots):
            forms = {c["form"] for c in chosen}
            return list(chosen) if set(ANSWER_FORMS) <= forms else None
        remaining = len(slots) - i - 1
        for cand in _elig(slots[i], taken_cl):
            forms = {c["form"] for c in chosen} | {cand["form"]}
            if len(set(ANSWER_FORMS) - forms) > remaining:
                continue  # form coverage infeasible down this branch
            got = _search(i + 1, chosen + [cand], taken_cl | {cand["cluster"]})
            if got is not None:
                return got
        return None

    return _search(0, [], set())


def select_exemplar_bank(pool: list[dict[str, Any]], *, seed: int = EXEMPLAR_SEED) -> dict:
    """Registered composition-template selection (M3): 3 pairwise
    cluster-disjoint sets (2 generic + 1 math + 1 code, all 3 answer forms)
    + 12 spares (6/3/3, clusters distinct within the k16 block, all forms
    present), orders O1 (recorded) / O2 / O3 (seed-42 permutations), nested
    k-blocks k1 in k4 in k16 RECORDED."""
    import numpy as np

    rng = np.random.default_rng([seed, 2544])
    pools: dict[str, list[dict]] = {c: [] for c in ("generic", "math", "code")}
    for p in pool:
        assert "cluster" in p, "assign_pool_clusters must run before selection"
        pools[p["class"]].append(p)
    for c in pools:
        pools[c] = [pools[c][i] for i in rng.permutation(len(pools[c]))]

    used_clusters: set[int] = set()
    used_ids: set[str] = set()
    sets: dict[str, list[dict]] = {}
    for sid in SET_IDS:
        picked = _pick_set(pools, used_clusters, used_ids, SET_TEMPLATE)
        if picked is None:
            raise RuntimeError(
                f"exemplar selection infeasible at set {sid} — pool too small/skewed "
                f"(pool counts: { {c: len(v) for c, v in pools.items()} }); raise "
                "EXEMPLAR_SCAN_CAP / quotas and re-run --phase config"
            )
        sets[sid] = picked
        used_clusters |= {p["cluster"] for p in picked}
        used_ids |= {p["sha16"] for p in picked}

    # Spares: same template scaled; clusters distinct WITHIN the k16 block
    # (S1 + spares) — cross-set disjointness binds sets only (plan M3).
    k16_clusters = {p["cluster"] for p in sets["S1"]}
    spare_used = set(used_clusters)  # also disjoint from S2/S3 (conservative)
    spares: list[dict] = []
    for cls, cnt in SPARE_TEMPLATE.items():
        got = 0
        for p in pools[cls]:
            if got >= cnt:
                break
            if p["sha16"] in used_ids or p["cluster"] in spare_used:
                continue
            spares.append(p)
            used_ids.add(p["sha16"])
            spare_used.add(p["cluster"])
            k16_clusters.add(p["cluster"])
            got += 1
        if got < cnt:
            raise RuntimeError(
                f"exemplar spares infeasible for class {cls} ({got}/{cnt}) — raise "
                "EXEMPLAR_SCAN_CAP / quotas and re-run --phase config"
            )
    spare_forms = {p["form"] for p in spares}
    if not set(ANSWER_FORMS) <= spare_forms:
        raise RuntimeError(f"spare answer forms incomplete: {sorted(spare_forms)}")

    # Orders: O1 = the recorded S1 selection order; O2/O3 = distinct
    # non-identity seed-42 permutations of the 4 S1 positions.
    o1 = list(range(K_DEFAULT))
    perms: list[list[int]] = []
    while len(perms) < 2:
        cand = rng.permutation(K_DEFAULT).tolist()
        if cand != o1 and cand not in perms:
            perms.append(cand)
    orders = {"O1": o1, "O2": perms[0], "O3": perms[1]}

    def _rec(p: dict, eid: str) -> dict:
        return {
            "exemplar_id": eid,
            "conversation_id": p["conversation_id"],
            "sha16": p["sha16"],
            "cluster": p["cluster"],
            "class": p["class"],
            "form": p["form"],
            "n_tokens_answer": p["n_tokens_answer"],
            "n_tokens_pair": p["n_tokens_pair"],
            "query": p["query"],
            "answer": p["answer"],
        }

    bank_sets = {
        sid: [_rec(p, f"{sid.lower()}_e{i}") for i, p in enumerate(sets[sid])] for sid in SET_IDS
    }
    bank_spares = [_rec(p, f"sp_e{i}") for i, p in enumerate(spares)]
    s1_ids = [r["exemplar_id"] for r in bank_sets["S1"]]
    k_blocks = {
        "k1": s1_ids[:1],
        "k4": s1_ids,
        "k16": s1_ids + [r["exemplar_id"] for r in bank_spares],
    }
    assert k_blocks["k1"] == k_blocks["k4"][:1] and k_blocks["k4"] == k_blocks["k16"][:4], (
        "nesting k1 in k4 in k16 violated"
    )
    assert len(k_blocks["k16"]) == 16
    return {
        "seed": seed,
        "template": {"set": SET_TEMPLATE, "spares": SPARE_TEMPLATE},
        "sets": bank_sets,
        "spares": bank_spares,
        "orders": orders,
        "k_blocks": k_blocks,
        "composition": {
            sid: {
                "classes": [r["class"] for r in bank_sets[sid]],
                "forms": [r["form"] for r in bank_sets[sid]],
                "clusters": [r["cluster"] for r in bank_sets[sid]],
            }
            for sid in SET_IDS
        },
    }


def _bank_lookup(bank: dict) -> dict[str, dict]:
    out = {}
    for sid in SET_IDS:
        for r in bank["sets"][sid]:
            out[r["exemplar_id"]] = r
    for r in bank["spares"]:
        out[r["exemplar_id"]] = r
    return out


def exemplar_prefix_turns(
    bank: dict, k: int, order_id: str | None, set_id: str | None
) -> list[dict] | None:
    """Prefix turns (``[{'role','content'}, ...]``) for a k-shot cell.

    k=0 -> None. k=1/16 -> the recorded nested k-blocks (O1/S1 only).
    k=4 -> ``set_id``'s 4 exemplars; orders O2/O3 permute S1 ONLY (the order
    arms), S2/S3 run in their recorded order (order_id must be 'O1')."""
    if k == 0:
        assert order_id is None and set_id is None, (k, order_id, set_id)
        return None
    lookup = _bank_lookup(bank)
    if k in (1, 16):
        assert order_id == "O1" and set_id == "S1", (k, order_id, set_id)
        ids = bank["k_blocks"][f"k{k}"]
    elif k == K_DEFAULT:
        assert set_id in SET_IDS and order_id in ORDER_IDS, (k, order_id, set_id)
        rows = bank["sets"][set_id]
        if set_id == "S1":
            perm = bank["orders"][order_id]
            ids = [rows[i]["exemplar_id"] for i in perm]
        else:
            assert order_id == "O1", (set_id, order_id)
            ids = [r["exemplar_id"] for r in rows]
    else:
        raise ValueError(f"unsupported k={k}")
    turns: list[dict] = []
    for eid in ids:
        r = lookup[eid]
        turns.append({"role": "user", "content": r["query"]})
        turns.append({"role": "assistant", "content": r["answer"]})
    assert len(turns) == 2 * k
    return turns


# ── gen / capture cell rosters ───────────────────────────────────────────────


def gen_cell_roster(rungs: tuple[str, ...] = RUNGS) -> list[dict[str, Any]]:
    """Pass-1 GENERATION cells (plan §4 P2), one dict per (rung, cell).

    Fields: cell, rung, render, k, order_id, set_id, seed, rows_scope
    ('full' | subset label). Full-corpus: gen0 + gen4 every rung; dose k1/k16
    at DOSE_RUNGS. Subsets: orders O2/O3 (robust, every rung); sets S2/S3
    (robust, SET_RUNGS); natgen (native render, NATIVE_GEN_RUNGS);
    reliability seeds 43/44 x {0-shot, 4-shot} every rung."""
    cells: list[dict[str, Any]] = []

    def _c(rung: str, cell: str, **kw: Any) -> None:
        base = {
            "cell": cell,
            "rung": rung,
            "render": "plain",
            "k": 0,
            "order_id": None,
            "set_id": None,
            "seed": C.GEN_SEED,
            "rows_scope": "full",
        }
        base.update(kw)
        cells.append(base)

    for r in rungs:
        _c(r, "gen0")
        _c(r, "gen4", k=4, order_id="O1", set_id="S1")
        for o in ("O2", "O3"):
            _c(r, f"gen4_{o.lower()}", k=4, order_id=o, set_id="S1", rows_scope="robust")
        if r in DOSE_RUNGS:
            for k in DOSE_KS:
                _c(r, f"dose_k{k}", k=k, order_id="O1", set_id="S1")
        if r in SET_RUNGS:
            for s in ("S2", "S3"):
                _c(r, f"gen4_{s.lower()}", k=4, order_id="O1", set_id=s, rows_scope="robust")
        if r in NATIVE_GEN_RUNGS:
            _c(r, "natgen", render="native", rows_scope="natgen")
        for seed in C.RELIABILITY_SEEDS:
            _c(r, f"rel0_s{seed}", seed=seed, rows_scope="reliability")
            _c(
                r,
                f"rel4_s{seed}",
                k=4,
                order_id="O1",
                set_id="S1",
                seed=seed,
                rows_scope="reliability",
            )
    return cells


def pass1_capture_cells(rungs: tuple[str, ...] = RUNGS) -> list[dict[str, Any]]:
    """Pass-1 CAPTURE cells: per rung the 0-shot diagonal at the FULL
    17-layer set + the two reliability repeats (plan §4 P3a)."""
    cells: list[dict[str, Any]] = []
    for r in rungs:
        cells.append(
            {
                "cell": "diag0",
                "rung": r,
                "render": "plain",
                "k": 0,
                "order_id": None,
                "set_id": None,
                "answer_cell": "gen0",
                "answer_rung": r,
                "rows_scope": "full",
                "subdir": f"{r}/diag0",
                "want_q": False,
                "store_ctx": True,
                "layers": "full17",
                "seed": C.GEN_SEED,
            }
        )
        for seed in C.RELIABILITY_SEEDS:
            cells.append(
                {
                    "cell": f"rel0_seed{seed}",
                    "rung": r,
                    "render": "plain",
                    "k": 0,
                    "order_id": None,
                    "set_id": None,
                    "answer_cell": f"rel0_s{seed}",
                    "answer_rung": r,
                    "rows_scope": "reliability",
                    "subdir": f"{r}/rel0_seed{seed}",
                    "want_q": False,
                    "store_ctx": True,
                    "layers": "full17",
                    "seed": seed,
                }
            )
    return cells


def pass2_capture_cells(
    rungs: tuple[str, ...] = RUNGS, *, include_lfa0: bool
) -> list[dict[str, Any]]:
    """Pass-2 CAPTURE cells at band B6 (plan §4 P3b).

    Cross cells (colC / rowR) store NO ctx summaries (identical to the
    capturing rung's diag0/lfa0 ctx by causal attention — the dedup rule);
    k-shot cells store ctx incl. q_mean (want_q)."""
    cells: list[dict[str, Any]] = []

    def _c(rung: str, cell: str, **kw: Any) -> None:
        base = {
            "cell": cell,
            "rung": rung,
            "render": "plain",
            "k": 0,
            "order_id": None,
            "set_id": None,
            "answer_rung": rung,
            "rows_scope": "full",
            "want_q": False,
            "store_ctx": True,
            "layers": "band6",
        }
        base.update(kw)
        base["subdir"] = f"{rung}/{cell}"
        cells.append(base)

    for m in rungs:
        if m != "main":
            _c(m, "colC_main", answer_cell="gen0", answer_rung="main", store_ctx=False)
        else:
            for s in rungs:
                if s == "main":
                    continue
                _c(m, f"rowR_{s}", answer_cell="gen0", answer_rung=s, store_ctx=False)
        _c(m, "diag4", k=4, order_id="O1", set_id="S1", answer_cell="gen4", want_q=True)
        if m in DOSE_RUNGS:
            for k in DOSE_KS:
                _c(
                    m,
                    f"dose_k{k}",
                    k=k,
                    order_id="O1",
                    set_id="S1",
                    answer_cell=f"dose_k{k}",
                    want_q=True,
                )
        for o in ("O2", "O3"):
            _c(
                m,
                f"gen4_{o.lower()}",
                k=4,
                order_id=o,
                set_id="S1",
                answer_cell=f"gen4_{o.lower()}",
                rows_scope="robust",
                want_q=True,
            )
        if m in SET_RUNGS:
            for s in ("S2", "S3"):
                _c(
                    m,
                    f"gen4_{s.lower()}",
                    k=4,
                    order_id="O1",
                    set_id=s,
                    answer_cell=f"gen4_{s.lower()}",
                    rows_scope="robust",
                    want_q=True,
                )
        if m in NATIVE_GEN_RUNGS:
            _c(m, "natgen", render="native", answer_cell="natgen", rows_scope="natgen")
        for seed in C.RELIABILITY_SEEDS:
            _c(
                m,
                f"rel4_seed{seed}",
                k=4,
                order_id="O1",
                set_id="S1",
                answer_cell=f"rel4_s{seed}",
                rows_scope="reliability",
                want_q=True,
            )
        if include_lfa0:
            _c(m, "lfa0", answer_cell="gen0", rows_scope="intersection", layers="lfa")
    return cells


def transfer_pairs(rungs: tuple[str, ...] = RUNGS) -> list[tuple[str, str]]:
    """52 transfer pairs (plan §5): every adjacent pair both directions (28)
    + long-range i<->main for every rung NOT adjacent to main (24)."""
    from itertools import pairwise

    order = list(rungs)
    pairs: list[tuple[str, str]] = []
    for a, b in pairwise(order):
        pairs += [(a, b), (b, a)]
    mi = order.index("main")
    adjacent_to_main = {order[mi - 1]} | ({order[mi + 1]} if mi + 1 < len(order) else set())
    for r in order:
        if r == "main" or r in adjacent_to_main:
            continue
        pairs += [(r, "main"), ("main", r)]
    assert len(pairs) == len(set(pairs)), "duplicate transfer pairs"
    return pairs


# ── sliding-window helpers (A1) ──────────────────────────────────────────────


def nearest_full_attention_layer(
    layer_star: int, full_layers: tuple[int, ...] = OLMO3_FULL_ATTENTION_LAYERS
) -> int:
    """l_FA = the full-attention layer nearest layer* (tie -> lower; plan P4a)."""
    return min(full_layers, key=lambda fa: (abs(fa - layer_star), fa))


def over_window_fraction(prompt_lens: list[int], window: int = OLMO3_SLIDING_WINDOW) -> float:
    """Fraction of prompts longer than the sliding-attention window (the
    per-(rung, k, render) A1 diagnostic)."""
    assert prompt_lens, "empty prompt-length list"
    return sum(1 for n in prompt_lens if n > window) / len(prompt_lens)


def layer_type_split(layers: list[int]) -> dict[str, list[int]]:
    """Sliding-vs-full split of a captured layer set (band reads report it)."""
    full = set(OLMO3_FULL_ATTENTION_LAYERS)
    return {
        "full_attention": [layer for layer in layers if layer in full],
        "sliding_attention": [layer for layer in layers if layer not in full],
    }
