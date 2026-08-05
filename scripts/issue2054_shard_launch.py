#!/usr/bin/env python
"""Per-cell GPU fan-out composer for the #2054 GPU-bound phases (Unit F).

Saturates a multi-GPU pod for ONE (condition, form, model) driver invocation by
fanning N variant-strided shards across physical GPUs:

- one subprocess per shard — ``--shard-index i --shard-count N`` threaded into
  the driver (``scripts/issue2054_capture.py`` / ``scripts/issue2054_phase_c.py``)
  with ``CUDA_VISIBLE_DEVICES=<gpu>`` pinned in the LAUNCHER env per shard (the
  in-process pin is silently defeated by any import-time cuInit — the
  gotchas.md CVD family — so the launcher-env pin is the authoritative one);
- per-cell writes are disjoint BY CONSTRUCTION (C6: ``{variant}/{cell_key}.npz``,
  ``capture_diagnostics/{cell_key}.json``, ``{variant}/on_policy_*.jsonl``), so
  concurrent shards never collide; each shard writes a shard-suffixed DIGEST
  and the composer AGGREGATES them into the canonical un-suffixed digest after
  ALL shards exit 0;
- hazard maps (round-2 carry-forward flags closed here):

  * capture ``--input-dir`` is composed from the PHASE→producer-root map —
    ``inserted -> data/issue_2054/spliced_inserted/``,
    ``on_policy -> data/issue_2054/on_policy/<model>/``,
    ``cell_c -> data/issue_2054/cell_c/`` — the static dispatch.sh default
    matched ``--phase inserted`` only;
  * phase_c ``--output-dir`` is composed PER MODEL
    (``data/issue_2054/on_policy/<model>/``): the resume sidecar regime
    includes the model axis while the output FILENAME does not, so a
    second-model run into one dir is REFUSED by design (test-pinned) — the
    composer never produces that shape. Capture's on_policy input map above
    mirrors the same per-model layout.

Parallelism across (condition, form, model) CONFIGS is the outer axis: run one
composer invocation per config with DISJOINT ``--gpus`` sets.

Usage:
  uv run python scripts/issue2054_shard_launch.py --driver capture \
      --condition inserted --form chat --model qwen2.5-7b-instruct \
      --gpus 0,1,2,3 -- --skip-upload
  uv run python scripts/issue2054_shard_launch.py --driver phase_c \
      --form bare_text --model qwen2.5-7b --gpus 4,5,6,7 -- --temperature 1.0
  # --plan prints the composed per-shard commands + env pins, runs nothing.

Exit codes: 0 all shards + aggregation OK; 1 any shard failed (per-shard rc
table + failing-shard log tail printed); 2 composition error.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_forms as forms  # noqa: E402

# Variant panels per driver/condition. Pinned equal to the driver defaults by
# tests/test_issue2054_unit_f.py (composer stays import-light: importing the
# driver modules here would pay their heavy import chains per composition).
LATTICE_VARIANTS = (
    "char_helios",
    "char_wren",
    "char_dana",
    "char_vex",
    "conversation_paired_stories_assistant",
)
CELL_C_VARIANTS = (
    "char_helios_op",
    "char_helios_op_base",
    "char_wren_op",
    "char_wren_op_base",
    "char_dana_op",
    "char_dana_op_base",
    "char_vex_op",
    "char_vex_op_base",
)

DATA_ROOT = "data/issue_2054"
DRIVER_SCRIPTS = {
    "capture": "issue2054_capture.py",
    "phase_c": "issue2054_phase_c.py",
}


def _log(msg: str) -> None:
    print(f"[shard_launch] {msg}", flush=True)


def default_variants(driver: str, condition: str) -> tuple[str, ...]:
    """The full variant panel for one driver invocation (plan §4 lattice)."""
    if driver == "capture" and condition == "cell_c":
        return CELL_C_VARIANTS
    return LATTICE_VARIANTS


def capture_input_dir(condition: str, model: str) -> str:
    """PHASE→producer-root map (hazard (c)): where each condition's input JSONLs
    live. on_policy is per-model (hazard (b) — phase_c writes per-model dirs)."""
    if condition == "inserted":
        return f"{DATA_ROOT}/spliced_inserted/"
    if condition == "on_policy":
        return f"{DATA_ROOT}/on_policy/{model}/"
    if condition == "cell_c":
        return f"{DATA_ROOT}/cell_c/"
    raise ValueError(f"unknown condition {condition!r} (expected {sorted(forms.CONDITIONS)})")


def phase_c_output_dir(model: str) -> str:
    """Per-model on_policy output root (hazard (b): the sidecar regime carries
    the model axis, the filename does not — distinct dirs per model)."""
    return f"{DATA_ROOT}/on_policy/{model}/"


def _extra_value(extra: list[str], flag: str) -> str | None:
    """Last value of `flag` in the passthrough args (argparse last-wins)."""
    val: str | None = None
    for i, tok in enumerate(extra):
        if tok == flag and i + 1 < len(extra):
            val = extra[i + 1]
        elif tok.startswith(flag + "="):
            val = tok.split("=", 1)[1]
    return val


def probe_gpus() -> list[str]:
    """Physical GPU indices via nvidia-smi (explicit env; [] when unavailable)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def compose_shards(args: argparse.Namespace, extra: list[str]) -> list[dict]:
    """Compose the per-shard subprocess specs (pure — no side effects).

    Returns [{shard_index, gpu (or None), cmd (argv list), log_name}]. The
    shard count is min(requested, len(variants)) so no shard resolves EMPTY
    (the drivers fail loud on an empty shard by design).
    """
    driver_script = str(_SCRIPT_DIR / DRIVER_SCRIPTS[args.driver])
    variants = (
        list(args.variants)
        if args.variants
        else list(default_variants(args.driver, args.condition or ""))
    )
    n_shards = args.shards if args.shards else max(1, len(args.gpus) or 1)
    if n_shards > len(variants):
        _log(f"clamping shards {n_shards} -> {len(variants)} (variant count)")
        n_shards = len(variants)

    shards: list[dict] = []
    for i in range(n_shards):
        cmd = [sys.executable, driver_script]
        if args.driver == "capture":
            input_dir = _extra_value(extra, "--input-dir") or capture_input_dir(
                args.condition, args.model
            )
            cmd += [
                "--input-dir",
                input_dir,
                "--phase",
                args.condition,
                "--form",
                args.form,
                "--model",
                args.model,
                "--variants",
                ",".join(variants),
            ]
        else:  # phase_c
            out_dir = _extra_value(extra, "--output-dir") or phase_c_output_dir(args.model)
            cmd += [
                "--output-dir",
                out_dir,
                "--form",
                args.form,
                "--model",
                args.model,
                "--variants",
                ",".join(variants),
            ]
        cmd += ["--shard-index", str(i), "--shard-count", str(n_shards)]
        # Passthrough LAST — argparse last-wins, so callers can override any
        # composed default except the shard spec (composer-owned).
        cmd += [t for t in extra if t != "--"]
        gpu = args.gpus[i % len(args.gpus)] if args.gpus else None
        cond = args.condition or "on_policy"
        log_name = (
            f"{args.driver}{forms.CELL_KEY_SEP}{cond}{forms.CELL_KEY_SEP}{args.form}"
            f"{forms.CELL_KEY_SEP}{args.model}{forms.CELL_KEY_SEP}shard{i}of{n_shards}.log"
        )
        shards.append(
            {"shard_index": i, "n_shards": n_shards, "gpu": gpu, "cmd": cmd, "log_name": log_name}
        )
    return shards


def _digest_dir(args: argparse.Namespace, extra: list[str]) -> Path:
    """The directory the driver writes its digest into for this invocation."""
    if args.driver == "capture":
        out = _extra_value(extra, "--output-dir") or f"{DATA_ROOT}/activations/"
    else:
        out = _extra_value(extra, "--output-dir") or phase_c_output_dir(args.model)
    p = Path(out)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def shard_digest_path(args: argparse.Namespace, digest_dir: Path, i: int, n: int) -> Path:
    sep = forms.CELL_KEY_SEP
    suffix = f"{sep}shard{i}of{n}" if n > 1 else ""
    if args.driver == "capture":
        stem = f"capture_digest{sep}{args.condition}{sep}{args.form}{sep}{args.model}"
    else:
        stem = f"phase_c_digest{sep}{args.form}"
    return digest_dir / f"{stem}{suffix}.json"


def aggregate_digests(args: argparse.Namespace, digest_dir: Path, n_shards: int) -> Path:
    """Merge shard digests into the canonical un-suffixed digest (post-hoc
    aggregation — hazard (a): no two shards ever write the canonical name)."""
    shard_paths = [shard_digest_path(args, digest_dir, i, n_shards) for i in range(n_shards)]
    missing = [str(p) for p in shard_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"shard digests missing for aggregation: {missing}")
    shard_digests = [json.loads(p.read_text(encoding="utf-8")) for p in shard_paths]

    canonical = shard_digest_path(args, digest_dir, 0, 1)  # n=1 -> un-suffixed name
    agg: dict = dict(shard_digests[0])
    agg["aggregated_from_shards"] = n_shards
    agg["shard_digests"] = [p.name for p in shard_paths]
    agg.pop("shard_index", None)
    agg["shard_count"] = n_shards
    agg["utc"] = datetime.now(tz=timezone.utc).isoformat()
    if args.driver == "capture":
        per_variant: list[dict] = []
        for d in shard_digests:
            per_variant.extend(d.get("per_variant") or [])
        agg["per_variant"] = sorted(per_variant, key=lambda r: str(r.get("variant")))
        agg["n_total_ok"] = sum(int(d.get("n_total_ok") or 0) for d in shard_digests)
    else:
        counts: dict[str, dict] = {}
        out_paths: dict[str, str] = {}
        for d in shard_digests:
            counts.update(d.get("counts") or {})
            out_paths.update(d.get("out_paths") or {})
        total_out = sum(int(d.get("n_total_out") or 0) for d in shard_digests)
        n_cap_hit = sum(int((d.get("cap_hit") or {}).get("n_cap_hit") or 0) for d in shard_digests)
        frac = (n_cap_hit / total_out) if total_out else 0.0
        cap_hit = dict(shard_digests[0].get("cap_hit") or {})
        cap_hit.update(
            {
                "n_cap_hit": n_cap_hit,
                "cap_hit_fraction": frac,
                "cap_hit_regen_trigger_fired": bool(frac > 0.02),
            }
        )
        agg.update(
            {"counts": counts, "out_paths": out_paths, "n_total_out": total_out, "cap_hit": cap_hit}
        )

    tmp = canonical.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, canonical)
    return canonical


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="passthrough driver args go after `--` (argparse last-wins in the driver)",
    )
    p.add_argument("--driver", choices=sorted(DRIVER_SCRIPTS), required=True)
    p.add_argument(
        "--condition",
        choices=forms.CONDITIONS,
        default=None,
        help="capture only: the condition axis (maps the input dir; REQUIRED for capture)",
    )
    p.add_argument("--form", required=True, choices=forms.FORMS)
    p.add_argument("--model", default="qwen2.5-7b-instruct")
    p.add_argument(
        "--variants",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="override the full per-driver variant panel (default: plan §4 panel)",
    )
    p.add_argument(
        "--gpus",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="physical GPU ids to pin one shard each (default: probe nvidia-smi)",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=0,
        help="shard count (default: len(--gpus); clamped to the variant count)",
    )
    p.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="permit a GPU-less fan-out (CPU --dry-run smokes): no CVD pin is set",
    )
    p.add_argument(
        "--plan",
        action="store_true",
        help="print the composed per-shard commands + env pins and exit (argv dry-run)",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="per-shard log dir (default: <digest dir>/_shard_logs/)",
    )
    p.add_argument("extra", nargs="*", help="driver passthrough args (after `--`)")
    args = p.parse_args()

    if args.driver == "capture" and not args.condition:
        p.error("--condition is required for --driver capture")

    if args.gpus is None:
        args.gpus = probe_gpus()
        _log(f"probed GPUs: {args.gpus or 'none'}")
    if not args.gpus and not args.allow_no_gpu:
        print(
            "ERROR: no GPUs (probe empty and --gpus unset) — pass --gpus, or "
            "--allow-no-gpu for a CPU --dry-run smoke",
            file=sys.stderr,
        )
        return 2
    if args.shards == 0 and not args.gpus:
        args.shards = 1

    extra = list(args.extra)
    shards = compose_shards(args, extra)
    n_shards = shards[0]["n_shards"] if shards else 0
    digest_dir = _digest_dir(args, extra)

    if args.plan:
        for s in shards:
            pin = f"CUDA_VISIBLE_DEVICES={s['gpu']} " if s["gpu"] is not None else ""
            print(f"shard {s['shard_index']}/{s['n_shards']}: {pin}{shlex.join(s['cmd'])}")
        print(f"digest aggregation -> {shard_digest_path(args, digest_dir, 0, 1)}")
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else digest_dir / "_shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[dict, subprocess.Popen, Path]] = []
    for s in shards:
        env = {**os.environ}
        if s["gpu"] is not None:
            # Launcher-env CVD pin (authoritative — gotchas.md CVD family).
            env["CUDA_VISIBLE_DEVICES"] = s["gpu"]
        log_path = log_dir / s["log_name"]
        fh = log_path.open("w", encoding="utf-8")
        _log(f"launch shard {s['shard_index']}/{s['n_shards']} gpu={s['gpu']} log={log_path}")
        proc = subprocess.Popen(  # noqa: S603
            s["cmd"], env=env, cwd=str(_REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT
        )
        procs.append((s, proc, log_path))

    failures: list[tuple[dict, int, Path]] = []
    for s, proc, log_path in procs:
        rc = proc.wait()
        _log(f"shard {s['shard_index']}/{s['n_shards']} exited rc={rc}")
        if rc != 0:
            failures.append((s, rc, log_path))
    if failures:
        for s, rc, log_path in failures:
            print(
                f"ERROR: shard {s['shard_index']}/{s['n_shards']} rc={rc}; log tail:",
                file=sys.stderr,
            )
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                print("\n".join(tail), file=sys.stderr)
            except OSError as exc:
                print(f"(log unreadable: {exc})", file=sys.stderr)
        return 1

    if n_shards > 1:
        canonical = aggregate_digests(args, digest_dir, n_shards)
        _log(f"aggregated {n_shards} shard digests -> {canonical}")
    _log("all shards complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
