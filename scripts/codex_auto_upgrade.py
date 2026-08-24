#!/usr/bin/env python3
"""Keep the Codex CLI and the twin's model pinned to the newest usable pair.

Three manual-path failure modes this automates away (all hit on 2026-08-15):

1. **CLI/model version coupling.** Each model advertises a
   ``minimal_client_version``; ``gpt-5.6-sol`` needs CLI >= 0.144.0. Bumping
   the model without the CLI yields a 400 at dispatch time, not at config
   time.
2. **The stale app-server.** A long-lived ``codex app-server`` keeps serving
   the PRE-upgrade runtime, so twin dispatch fails with "requires a newer
   version of Codex" while a direct ``codex exec`` on the same model
   succeeds. ``codex --version`` reports the new version and tells you
   nothing about what the twin actually runs. The broker + app-server must be
   restarted (they respawn on demand).
3. **Slugs the account cannot use.** ``gpt-5.5-codex`` is listed by the
   models API but 400s on a ChatGPT account ("not supported when using Codex
   with a ChatGPT account"). No field in the listing predicts this, so the
   only sound filter is an actual probe.

Design consequences:

- A candidate model is PROBED with a real ``codex exec`` call before it is
  written to config. A failing candidate is recorded in a known-bad cache
  (keyed by CLI version, so a later CLI re-probes it) and the next candidate
  by priority is tried. The live config is never left pointing at a model
  that has not just answered a prompt.
- ``model_reasoning_effort`` is NEVER touched, whatever it is set to. It is
  a cost/latency preference, not a freshness property, and nothing in the
  models listing implies a value — a new model's arrival is no reason to
  re-decide how hard the twin should think.
- The run ABORTS if any Codex job is in flight, and re-checks immediately
  before the app-server restart. An upgrade mid-job kills the job; the twin
  ensemble sites treat that as a no-show and degrade to single-Claude
  review, silently weakening a gate.

Exit codes: 0 = clean (changed or already current, including a deliberate
skip), 1 = a step failed and the alert should fire. The cron wrapper
converts rc != 0 into one Telegram push per day.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
CONFIG_PATH = CODEX_HOME / "config.toml"
AUTH_PATH = CODEX_HOME / "auth.json"

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_BAD_PATH = Path(
    os.environ.get(
        "EPS_CODEX_KNOWN_BAD_MODELS",
        REPO_ROOT / ".claude" / "cache" / "codex-model-probe-failures.json",
    )
)

MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
NPM_PACKAGE = "@openai/codex"

# Bound the probe spend: a listing with many unusable slugs must not turn one
# cron pass into a dozen model calls.
MAX_PROBE_CANDIDATES = int(os.environ.get("EPS_CODEX_MAX_PROBE_CANDIDATES", "3"))
PROBE_TIMEOUT_S = int(os.environ.get("EPS_CODEX_PROBE_TIMEOUT_S", "180"))
PROBE_SENTINEL = "CODEX_UPGRADE_PROBE_OK"

# Non-terminal companion job phases. Anything not in the terminal set counts
# as in-flight: an unrecognized future phase must read as "busy" (fail toward
# not disrupting work), never as "idle".
TERMINAL_PHASES = {"done", "failed", "cancelled", "error", "timeout"}

# How recently a non-terminal job must have moved to count as in-flight. A
# twin review runs minutes, not hours; 2h is generous headroom over the
# longest observed dispatch while still clearing same-day debris.
JOB_FRESH_WINDOW_S = float(os.environ.get("EPS_CODEX_JOB_FRESH_WINDOW_S", str(2 * 3600)))

COMPANION_STATE_DIR = Path(
    os.environ.get(
        "EPS_CODEX_COMPANION_STATE_DIR",
        Path.home() / ".claude" / "plugins" / "data" / "codex-openai-codex" / "state",
    )
)


def log(msg: str) -> None:
    print(f"codex_auto_upgrade: {msg}", flush=True)


def run(
    cmd: list[str], timeout: int = 120, cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command capturing text output; never raises on non-zero rc."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


# --------------------------------------------------------------------------
# version helpers
# --------------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...] | None:
    """Extract a dotted numeric version from arbitrary text.

    ``codex --version`` prints ``codex-cli 0.147.0``; ``npm view`` prints a
    bare ``0.147.0``. Returns None when no version is present so callers can
    treat an unparseable read as "unknown" rather than as version zero (which
    would make every comparison claim an upgrade is available).
    """
    m = re.search(r"(\d+(?:\.\d+)+)", text or "")
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def version_str(v: tuple[int, ...] | None) -> str:
    return ".".join(str(p) for p in v) if v else "unknown"


def installed_cli_version() -> tuple[int, ...] | None:
    if not shutil.which("codex"):
        return None
    proc = run(["codex", "--version"], timeout=60)
    if proc.returncode != 0:
        return None
    return parse_version(proc.stdout)


def npm_latest_version() -> tuple[int, ...] | None:
    proc = run(["npm", "view", NPM_PACKAGE, "version"], timeout=180)
    if proc.returncode != 0:
        log(f"npm view failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
        return None
    return parse_version(proc.stdout)


# --------------------------------------------------------------------------
# in-flight job detection
# --------------------------------------------------------------------------


def _job_age_s(job: dict, state_file: Path, now: float) -> float | None:
    """Seconds since the job record last moved, or None when unknowable.

    Handles the observed ISO-8601 form (``2026-05-17T04:26:32.309Z``) and
    epoch seconds/milliseconds. Falls back to the state file's mtime, which
    is always available and moves whenever the companion writes any job in
    that workspace — a conservative over-estimate of freshness, never under.
    """
    raw = job.get("updatedAt") or job.get("createdAt")
    if raw is not None:
        try:
            n = float(raw)
            return now - (n / 1000.0 if n > 1e11 else n)
        except (TypeError, ValueError):
            pass
        try:
            from datetime import datetime

            txt = str(raw).replace("Z", "+00:00")
            return now - datetime.fromisoformat(txt).timestamp()
        except (TypeError, ValueError):
            pass
    try:
        return now - state_file.stat().st_mtime
    except OSError:
        return None


def inflight_jobs() -> list[str]:
    """Ids of Codex companion jobs that are non-terminal AND recently active.

    The staleness bound is load-bearing, not a nicety. Companion job records
    are only advanced by a live session, so a session killed mid-job strands
    its record at ``running``/``verifying`` forever (observed: two such
    records from 2026-05, still non-terminal three months on). Treating every
    non-terminal record as in-flight would let that permanent debris disable
    auto-upgrade silently — the failure mode is a no-op that looks like a
    working cron.

    An unreadable state file counts as busy only while its mtime is fresh, so
    a corrupt file cannot wedge upgrades permanently either.
    """
    busy: list[str] = []
    if not COMPANION_STATE_DIR.is_dir():
        return busy
    now = time.time()
    for state_file in COMPANION_STATE_DIR.glob("*/state.json"):
        try:
            data = json.loads(state_file.read_text())
        except (OSError, ValueError) as exc:
            try:
                fresh = (now - state_file.stat().st_mtime) < JOB_FRESH_WINDOW_S
            except OSError:
                fresh = False
            if fresh:
                busy.append(f"<unreadable {state_file.parent.name}: {type(exc).__name__}>")
            continue
        jobs = data.get("jobs")
        if isinstance(jobs, dict):
            jobs = list(jobs.values())
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if not isinstance(job, dict):
                continue
            phase = job.get("phase") or job.get("status")
            if phase is not None and str(phase).lower() in TERMINAL_PHASES:
                continue
            age = _job_age_s(job, state_file, now)
            # Unknowable age => treat as stale. Every real live job has a
            # fresh timestamp; only debris lacks one, and failing busy here
            # would resurrect the permanent-wedge mode described above.
            if age is None or age >= JOB_FRESH_WINDOW_S:
                continue
            busy.append(str(job.get("id") or "<unnamed>"))
    return busy


# --------------------------------------------------------------------------
# models listing
# --------------------------------------------------------------------------


def fetch_models(client_version: str) -> list[dict] | None:
    """Fetch the account's model listing, or None when it cannot be read.

    The access token in auth.json is short-lived. `codex exec` refreshes it
    in-band, so the caller warms it before this runs; a 401 here is still
    non-fatal (None => keep the configured model).
    """
    try:
        auth = json.loads(AUTH_PATH.read_text())
    except (OSError, ValueError) as exc:
        log(f"cannot read {AUTH_PATH}: {exc}")
        return None
    tokens = auth.get("tokens") or {}
    token = tokens.get("access_token")
    if not token:
        log("no access_token in auth.json — is `codex login` done?")
        return None
    req = urllib.request.Request(
        f"{MODELS_URL}?client_version={client_version}",
        headers={
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": tokens.get("account_id") or "",
            "User-Agent": f"codex-cli/{client_version}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        log(f"models listing fetch failed: {exc}")
        return None
    models = body.get("models")
    return models if isinstance(models, list) else None


def candidate_models(
    models: list[dict], cli_version: tuple[int, ...], known_bad: dict
) -> list[dict]:
    """Listed models this CLI can actually run, best first.

    Filters: visibility == "list" (hidden slugs like codex-auto-review are
    internal), minimal_client_version satisfied by the installed CLI, and not
    a slug this same CLI version already failed to probe.
    """
    out = []
    for m in models:
        slug = m.get("slug")
        if not slug or m.get("visibility") != "list":
            continue
        need = parse_version(str(m.get("minimal_client_version") or "0"))
        if need and cli_version < need:
            continue
        bad = known_bad.get(slug)
        if bad and bad.get("cli_version") == version_str(cli_version):
            continue
        out.append(m)
    # priority is the account's own ranking; absent priority sorts last.
    out.sort(key=lambda m: m.get("priority", 10**6))
    return out


# --------------------------------------------------------------------------
# config read/write
# --------------------------------------------------------------------------


def read_config_model(text: str) -> str | None:
    """Value of the TOP-LEVEL `model = "..."` key (before any [section])."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            break
        m = re.match(r'^model\s*=\s*"([^"]+)"', s)
        if m:
            return m.group(1)
    return None


def write_config_model(text: str, slug: str) -> str:
    """Replace the top-level `model` value, preserving everything else.

    Only the first pre-section match is rewritten so a `[profiles.*]` model
    override is left alone. When no top-level key exists the assignment is
    prepended, which is where a bare key must live in TOML.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith("["):
            break
        if re.match(r'^model\s*=\s*"[^"]+"', line.strip()):
            lines[i] = re.sub(r'^(\s*model\s*=\s*)"[^"]+"', rf'\1"{slug}"', line)
            return "".join(lines)
    return f'model = "{slug}"\n' + "".join(lines)


def atomic_write(path: Path, text: str) -> None:
    """Write via tmp+replace in the target dir so config is never truncated."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        # Leave no orphan tmp file behind on any failure path.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_known_bad() -> dict:
    try:
        data = json.loads(KNOWN_BAD_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_known_bad(data: dict) -> None:
    try:
        KNOWN_BAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(KNOWN_BAD_PATH, json.dumps(data, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        # Advisory cache only: losing it costs re-probes, never correctness.
        log(f"could not persist known-bad cache: {exc}")


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


def probe_model(slug: str) -> tuple[bool, str]:
    """Ask the model to echo a sentinel. Returns (ok, detail).

    Runs from $HOME at low effort — the probe tests whether the account can
    reach the slug at all, not reasoning depth.

    Two flags are load-bearing:

    - ``--skip-git-repo-check``: $HOME is not a git repo, and `codex exec`
      otherwise refuses with "Not inside a trusted directory" (rc=1). The
      probe reads and writes nothing, so the guard has nothing to protect
      here; running from a repo instead would put a model with write tooling
      in a live worktree for no benefit.
    - ``stdin=DEVNULL`` (via run()): `codex exec` blocks on "Reading
      additional input from stdin..." when stdin is a pipe.
    """
    proc = run(
        [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-c",
            f"model={slug}",
            "-c",
            "model_reasoning_effort=low",
            f"Reply with exactly this token and nothing else: {PROBE_SENTINEL}",
        ],
        timeout=PROBE_TIMEOUT_S,
        cwd=str(Path.home()),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        m = re.search(r'"message"\s*:\s*"([^"]+)"', out)
        return False, (m.group(1) if m else f"rc={proc.returncode}")[:300]
    # The prompt itself contains the sentinel, so require it to appear again
    # in the model's reply rather than merely somewhere in the transcript.
    if out.count(PROBE_SENTINEL) < 2:
        return False, "sentinel not echoed in reply"
    return True, "ok"


# --------------------------------------------------------------------------
# app-server restart
# --------------------------------------------------------------------------


def _pgrep(pattern: str) -> list[int]:
    """euid-scoped pgrep. The caller supplies a bracketed pattern so the
    probe's own command line cannot match itself."""
    proc = run(["pgrep", "-u", str(os.geteuid()), "-f", pattern], timeout=30)
    return [int(p) for p in proc.stdout.split() if p.strip().isdigit()]


def restart_app_server(dry_run: bool) -> bool:
    """Kill broker + app-server by explicit PID so they respawn on the new
    binary. Returns True when nothing was left running."""
    pats = [r"app-server-broke[r]\.mjs", r"codex ap[p]-server"]
    pids: list[int] = []
    for pat in pats:
        pids.extend(_pgrep(pat))
    pids = sorted(set(pids))
    if not pids:
        log("app-server: no running broker/app-server — nothing to restart")
        return True
    if dry_run:
        log(f"app-server: DRY-RUN would restart pids {pids}")
        return True

    # Re-check for in-flight work as late as possible: a job may have been
    # dispatched during the upgrade above.
    busy = inflight_jobs()
    if busy:
        log(f"app-server: SKIP restart — jobs went in-flight during upgrade: {busy[:5]}")
        return False

    log(f"app-server: restarting (pids {pids})")
    for sig in ("-TERM", "-KILL"):
        alive = [p for p in pids if _pid_alive(p)]
        if not alive:
            break
        for pid in alive:
            try:
                subprocess.run(["kill", sig, str(pid)], capture_output=True, timeout=15)
            except (subprocess.SubprocessError, OSError):
                pass
        time.sleep(8 if sig == "-TERM" else 2)

    survivors = [p for p in pids if _pid_alive(p)]
    if survivors:
        log(f"app-server: ERROR pids survived SIGKILL: {survivors}")
        return False
    log("app-server: confirmed stopped (respawns on next dispatch)")
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:  # noqa: C901 — linear pipeline with per-step guards
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true", help="report only; no install, config write, or kill"
    )
    ap.add_argument(
        "--skip-cli", action="store_true", help="do not upgrade the CLI; re-select model only"
    )
    ap.add_argument(
        "--force-inflight",
        action="store_true",
        help="proceed even with jobs in flight (may kill a running twin review)",
    )
    args = ap.parse_args()

    changed: list[str] = []
    failed = False

    cli_v = installed_cli_version()
    if cli_v is None:
        log("FATAL: `codex` not on PATH or --version unreadable")
        return 1
    log(f"installed CLI {version_str(cli_v)}")

    busy = inflight_jobs()
    if busy and not args.force_inflight:
        log(f"SKIP: {len(busy)} Codex job(s) in flight ({busy[:5]}) — retrying next pass")
        return 0

    # --- 1. CLI upgrade -----------------------------------------------------
    cli_upgraded = False
    if not args.skip_cli:
        latest = npm_latest_version()
        if latest is None:
            log("could not determine npm latest — keeping installed CLI")
        elif latest > cli_v:
            if args.dry_run:
                log(f"DRY-RUN would upgrade CLI {version_str(cli_v)} -> {version_str(latest)}")
            else:
                log(f"upgrading CLI {version_str(cli_v)} -> {version_str(latest)}")
                proc = run(["npm", "install", "-g", f"{NPM_PACKAGE}@latest"], timeout=600)
                if proc.returncode != 0:
                    log(f"npm install FAILED rc={proc.returncode}: {proc.stderr.strip()[:300]}")
                    failed = True
                else:
                    new_v = installed_cli_version()
                    if new_v != latest:
                        log(
                            f"npm install reported success but version is "
                            f"{version_str(new_v)}, expected {version_str(latest)}"
                        )
                        failed = True
                    else:
                        cli_v = new_v
                        cli_upgraded = True
                        changed.append(f"CLI -> {version_str(cli_v)}")
                        log(f"CLI now {version_str(cli_v)}")
        else:
            log(f"CLI already current ({version_str(cli_v)})")

    # --- 2. model selection -------------------------------------------------
    try:
        config_text = CONFIG_PATH.read_text()
    except OSError as exc:
        log(f"FATAL: cannot read {CONFIG_PATH}: {exc}")
        return 1
    current_model = read_config_model(config_text)
    log(f"configured model: {current_model or '<unset>'}")

    known_bad = load_known_bad()

    # Probe the CONFIGURED model first. Two jobs in one call: it refreshes the
    # hourly auth token in-band (a bare HTTPS call to the models endpoint just
    # 401s), and it detects the twin silently losing access to its own model.
    # A failure here is recorded as known-bad so the selector below EXCLUDES
    # it and picks the next working slug — without that, a broken current
    # model still ranks "best available" and the run is a no-op while every
    # twin dispatch 400s.
    current_broken = False
    if current_model and not args.dry_run:
        ok, detail = probe_model(current_model)
        log(f"current-model probe: {'ok' if ok else 'FAILED — ' + detail}")
        if not ok:
            current_broken = True
            known_bad[current_model] = {
                "cli_version": version_str(cli_v),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "error": detail,
            }
            save_known_bad(known_bad)

    models = fetch_models(version_str(cli_v))
    if models is None:
        log("no models listing — leaving model unchanged")
        # A broken current model with no listing to recover from is a real
        # outage: the twin cannot dispatch until someone looks.
        failed = failed or current_broken
    else:
        cands = candidate_models(models, cli_v, known_bad)
        if not cands:
            log("no usable candidate models in listing — leaving model unchanged")
            failed = failed or current_broken
        elif cands[0].get("slug") == current_model:
            log(
                f"model already best available ({current_model}, priority {cands[0].get('priority')})"
            )
        else:
            switched = False
            for cand in cands[:MAX_PROBE_CANDIDATES]:
                slug = cand["slug"]
                if slug == current_model:
                    log(f"reached current model {slug} in priority order — keeping it")
                    switched = True  # keeping a working current model is success
                    break
                if args.dry_run:
                    log(f"DRY-RUN would probe + switch model {current_model} -> {slug}")
                    switched = True
                    break
                log(f"probing candidate {slug} (priority {cand.get('priority')})")
                ok, detail = probe_model(slug)
                if not ok:
                    log(f"candidate {slug} unusable: {detail}")
                    known_bad[slug] = {
                        "cli_version": version_str(cli_v),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "error": detail,
                    }
                    save_known_bad(known_bad)
                    continue
                try:
                    # Re-read rather than reusing the text captured before the
                    # probes: those take minutes, and a concurrent hand-edit
                    # of config.toml must not be clobbered.
                    fresh = CONFIG_PATH.read_text()
                    atomic_write(CONFIG_PATH, write_config_model(fresh, slug))
                except OSError as exc:
                    log(f"FAILED writing {CONFIG_PATH}: {exc}")
                    failed = True
                    break
                log(f"model {current_model} -> {slug}")
                changed.append(f"model -> {slug}")
                switched = True
                break
            if not switched:
                # Every candidate we were willing to probe failed. If the
                # configured model also failed, the twin has no working model.
                log(f"no candidate model probed clean (tried {MAX_PROBE_CANDIDATES})")
                failed = failed or current_broken

    # --- 3. app-server restart ---------------------------------------------
    # Required after a CLI upgrade (the running server keeps the old runtime)
    # and harmless after a model change.
    if changed and not restart_app_server(args.dry_run):
        failed = True

    if cli_upgraded and not changed:  # pragma: no cover - defensive
        log("CLI upgraded but no change recorded")

    log(
        "RESULT: "
        + ("; ".join(changed) if changed else "no change")
        + (" [FAILED]" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
