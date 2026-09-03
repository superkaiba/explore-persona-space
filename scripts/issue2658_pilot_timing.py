"""Issue #2658 — P1 pilot timing artifact producer (plan v5 A4).

Parses the P1 launch logs (HF ``issue2658_dirvalid/launch_logs/p1``, or a local
``--logs-dir`` staging copy) plus the pod lifecycle timestamps into
``eval_results/issue_2658/power_inputs/pilot_timing.json`` — the measured input
``issue2658_power.cost_report`` requires. NEVER hand-typed, NEVER defaulted: a
missing stamp/line raises. Measured components:

- all-in pod wall (``--pod-start``/``--pod-end`` from the task's
  ``epm:run-launched v1`` / ``epm:pod-terminated v1`` markers) x ``--gpu-count``;
- vLLM engine-init seconds per generate shard (first ``INFO MM-DD HH:MM:SS``
  stamp -> the ``Graph capturing finished`` stamp);
- generation marginal rate from the shards where EVERY ``[gen] cell`` line is
  ``resumed=False`` (live-generated; resumed shards re-serve cached records and
  would bias the rate to zero);
- capture rows/s from the final ``[capture-shardNN] rows N/N elapsed=Ss`` line
  per sub-shard. The capture MODEL-LOAD overhead is NOT in the logs: recorded
  ``basis: not-measured`` and the capture projection component is a LOWER BOUND.

Launch (VM-side, shared-VM thread caps):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2658_pilot_timing.py \
      --logs-dir /mnt/eps-data/thomasjiralerspong/issue2658_logs/p1_hf/issue2658_dirvalid/launch_logs/p1 \
      --pod-start 2026-09-03T04:41:21Z --pod-end 2026-09-03T08:00:47Z --gpu-count 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token must bind before any hub import

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

LOGS_HF_PREFIX = f"{G.EXPERIMENT_NAME}/launch_logs/p1"

_VLLM_STAMP_RE = re.compile(r"INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_GEN_CELL_RE = re.compile(
    r"\[gen\] cell (\d+)/(\d+) (\S+) records=(\d+) resumed=(True|False) elapsed=(\d+)s"
)
_CAPTURE_RE = re.compile(r"\[capture-shard(\d+)\] rows (\d+)/(\d+) elapsed=(\d+(?:\.\d+)?)s")
_P1_WIDTH = "[phase=p1_width]"


class TimingParseError(RuntimeError):
    """A required measured quantity is absent from the logs — never defaulted."""


def _parse_iso_z(s: str) -> datetime:
    if not s.endswith("Z"):
        raise TimingParseError(f"timestamp {s!r} must be ISO8601 with a trailing Z")
    return datetime.fromisoformat(s[:-1]).replace(tzinfo=UTC)


def _stamp_to_dt(m: re.Match, year: int) -> datetime:
    mo, dy, hh, mm, ss = (int(g) for g in m.groups())
    return datetime(year, mo, dy, hh, mm, ss, tzinfo=UTC)


@dataclass
class GenShard:
    name: str
    engine_init_s: float
    all_fresh: bool
    n_cells: int
    total_records: int
    final_elapsed_s: int
    anchors: dict[str, int]


def parse_generate_shard(path: Path, year: int) -> GenShard:
    first_dt = None
    cap_dt = None
    anchors: dict[str, int] = {}
    cells: list[tuple[int, bool, int]] = []  # (records, resumed, elapsed)
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if first_dt is None:
            m = _VLLM_STAMP_RE.search(line)
            if m:
                first_dt = _stamp_to_dt(m, year)
                anchors["first_vllm_stamp"] = lineno
        if cap_dt is None and "Graph capturing finished" in line:
            m = _VLLM_STAMP_RE.search(line)
            if not m:
                raise TimingParseError(f"{path.name}:{lineno}: graph-capture line has no stamp")
            cap_dt = _stamp_to_dt(m, year)
            anchors["graph_capturing_finished"] = lineno
        m = _GEN_CELL_RE.search(line)
        if m:
            cells.append((int(m.group(4)), m.group(5) == "True", int(m.group(6))))
            anchors["last_gen_cell"] = lineno
    if first_dt is None or cap_dt is None:
        raise TimingParseError(
            f"{path.name}: missing vLLM stamps (first={first_dt}, graph-capture={cap_dt})"
        )
    init_s = (cap_dt - first_dt).total_seconds()
    if init_s <= 0:
        raise TimingParseError(f"{path.name}: non-positive engine init {init_s}s")
    if not cells:
        raise TimingParseError(f"{path.name}: no [gen] cell lines")
    return GenShard(
        name=path.stem,
        engine_init_s=init_s,
        all_fresh=all(not resumed for _, resumed, _ in cells),
        n_cells=len(cells),
        total_records=sum(rec for rec, _, _ in cells),
        final_elapsed_s=cells[-1][2],
        anchors=anchors,
    )


def parse_capture_shard(path: Path) -> tuple[int, float, dict[str, int]]:
    """(total rows, total elapsed seconds, line anchors) from the FINAL progress
    line of every ``[capture-shardNN]`` sub-shard in one GPU-shard log."""
    last: dict[str, tuple[int, int, float, int]] = {}
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        m = _CAPTURE_RE.search(line)
        if m:
            last[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)), lineno)
    if not last:
        raise TimingParseError(f"{path.name}: no [capture-shardNN] progress lines")
    rows = 0
    elapsed = 0.0
    anchors: dict[str, int] = {}
    for tag, (done, total, secs, lineno) in sorted(last.items()):
        if done != total:
            raise TimingParseError(
                f"{path.name}: sub-shard {tag} final line reads {done}/{total} — incomplete"
            )
        rows += done
        elapsed += secs
        anchors[f"capture_subshard_{tag}_final"] = lineno
    return rows, elapsed, anchors


def fetch_logs_from_hub(dest: Path) -> Path:
    """Stage the P1 launch-log prefix from the canonical HF data repo (scoped
    ``list_repo_tree`` that RAISES on a missing prefix; one pinned revision)."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        list_repo_entries_complete,
    )

    api = HfApi()
    revision = api.repo_info(DEFAULT_DATASET_REPO, repo_type="dataset").sha
    entries = list_repo_entries_complete(
        api,
        DEFAULT_DATASET_REPO,
        repo_type="dataset",
        revision=revision,
        path_in_repo=LOGS_HF_PREFIX,
    )
    if not entries:
        raise TimingParseError(f"HF prefix {DEFAULT_DATASET_REPO}/{LOGS_HF_PREFIX} listed 0 files")
    dest.mkdir(parents=True, exist_ok=True)
    for path_in_repo, _size in entries:
        local = hf_hub_download(
            repo_id=DEFAULT_DATASET_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
        )
        (dest / Path(path_in_repo).name).write_bytes(Path(local).read_bytes())
    print(f"[timing] fetched {len(entries)} log files from {LOGS_HF_PREFIX}@{revision[:12]}")
    return dest


def build_timing(
    logs_dir: Path, pod_start: str, pod_end: str, gpu_count: int, out_root: Path
) -> dict:
    start = _parse_iso_z(pod_start)
    end = _parse_iso_z(pod_end)
    wall_s = (end - start).total_seconds()
    if wall_s <= 0:
        raise TimingParseError(f"pod window {pod_start}..{pod_end} is non-positive")
    wall_hours = wall_s / 3600.0
    gpu_hours_all_in = wall_hours * gpu_count

    order_path = Path(out_root) / "gen_order_manifest" / "pilot_shard00of01.json"
    if not order_path.exists():
        raise TimingParseError(f"gen order manifest absent: {order_path}")
    n_responses = int(json.loads(order_path.read_text())["n_requests"])

    gen_paths = sorted(p for p in logs_dir.glob("generate_shard*.log") if "smoke" not in p.name)
    cap_paths = sorted(p for p in logs_dir.glob("capture_shard*.log") if "smoke" not in p.name)
    launcher = logs_dir / "launcher_main.log"
    if not gen_paths or not cap_paths or not launcher.exists():
        raise TimingParseError(
            f"{logs_dir}: expected generate_shard*.log + capture_shard*.log + launcher_main.log"
        )

    gen_shards = [parse_generate_shard(p, start.year) for p in gen_paths]
    fresh = [s for s in gen_shards if s.all_fresh]
    if not fresh:
        raise TimingParseError(
            "no generate shard has every cell resumed=False — the generation "
            "marginal rate cannot be measured from this log set"
        )
    fresh_records = sum(s.total_records for s in fresh)
    fresh_elapsed = sum(s.final_elapsed_s for s in fresh)
    if fresh_records <= 0 or fresh_elapsed <= 0:
        raise TimingParseError(
            f"degenerate fresh-shard totals: records={fresh_records} elapsed={fresh_elapsed}"
        )
    gen_marginal = fresh_elapsed / fresh_records  # s per response per GPU

    cap_rows = 0
    cap_elapsed = 0.0
    cap_anchors: dict[str, dict[str, int]] = {}
    for p in cap_paths:
        rows, secs, anchors = parse_capture_shard(p)
        cap_rows += rows
        cap_elapsed += secs
        cap_anchors[p.name] = anchors
    if cap_elapsed <= 0:
        raise TimingParseError(f"degenerate capture elapsed total {cap_elapsed}")
    if cap_rows != n_responses:
        raise TimingParseError(
            f"capture rows {cap_rows} != gen order manifest n_requests {n_responses}"
        )
    capture_rate = cap_rows / cap_elapsed  # rows per second per GPU

    launcher_lines = launcher.read_text(errors="replace").splitlines()
    width_anchors = [i for i, ln in enumerate(launcher_lines, start=1) if _P1_WIDTH in ln]
    n_starts = len(width_anchors)
    if n_starts < 1:
        raise TimingParseError(f"{launcher.name}: no {_P1_WIDTH} start lines")

    init_by_shard = {s.name: s.engine_init_s for s in gen_shards}
    fixed_overhead_hours = sum(init_by_shard.values()) / 3600.0

    return {
        "schema": "i2658-pilot-timing-v2",
        # --- the four fields cost_report has always required ---------------
        "wall_hours": wall_hours,
        "gpu_count": gpu_count,
        "n_responses": n_responses,
        "fixed_overhead_hours": fixed_overhead_hours,
        # --- v5 A4 measured components --------------------------------------
        "pod_wall_hours_all_in": wall_hours,
        "gpu_hours_all_in": gpu_hours_all_in,
        "pod_window": {"start": pod_start, "end": pod_end},
        "gen_marginal_s_per_response_per_gpu": gen_marginal,
        "gen_engine_init_s_per_shard": init_by_shard,
        "capture_rows_per_s_per_gpu": capture_rate,
        "capture_model_load_s_per_shard": {
            "value": None,
            "basis": "not-measured",
            "detail": (
                "capture model-load overhead is not stamped in the P1 logs; the "
                "capture projection component is therefore a LOWER BOUND"
            ),
        },
        "shards_used_for_gen_rate": sorted(s.name for s in fresh),
        "shards_excluded_resumed": sorted(s.name for s in gen_shards if not s.all_fresh),
        "gen_rate_basis": {
            "records": fresh_records,
            "final_elapsed_s": fresh_elapsed,
            "detail": (
                "sum(records)/sum(final elapsed) over shards where every [gen] cell "
                "line is resumed=False (one GPU per shard)"
            ),
        },
        "capture_rate_basis": {"rows": cap_rows, "elapsed_s": cap_elapsed},
        "crash_fix_rounds_note": (
            f"launcher_main.log shows {n_starts} {_P1_WIDTH} starts (lines "
            f"{width_anchors}) — the first launch plus {n_starts - 1} crash-fix "
            "restarts; the all-in pod wall (crash-fix rounds + upload remediation "
            "included) dominates the productive generation+capture compute (< 1 GPU-h)"
        ),
        "sources": {
            "logs_dir": str(logs_dir),
            "hf_prefix": LOGS_HF_PREFIX,
            "gen_order_manifest": str(order_path),
            "generate_shards": {s.name: s.anchors | {"all_fresh": s.all_fresh} for s in gen_shards},
            "capture_shards": cap_anchors,
            "launcher_p1_width_lines": width_anchors,
            "pod_window_source": "epm:run-launched v1 / epm:pod-terminated v1 markers",
        },
        "metadata": as_metadata_dict(git_provenance(), phase="p3-pilot-timing"),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="local P1 launch-log dir; absent => fetch the HF prefix",
    )
    ap.add_argument("--pod-start", required=True, help="pod start ISO8601Z (epm:run-launched)")
    ap.add_argument("--pod-end", required=True, help="pod end ISO8601Z (epm:pod-terminated)")
    ap.add_argument("--gpu-count", required=True, type=int)
    ap.add_argument("--out-root", type=Path, default=F.OUT_DIR)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default <out-root>/power_inputs/pilot_timing.json)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.logs_dir is not None:
        logs_dir = args.logs_dir
        if not logs_dir.is_dir():
            raise TimingParseError(f"--logs-dir {logs_dir} is not a directory")
    else:
        logs_dir = fetch_logs_from_hub(Path(tempfile.mkdtemp(prefix="i2658-p1-logs-")))
    timing = build_timing(
        logs_dir, args.pod_start, args.pod_end, args.gpu_count, Path(args.out_root)
    )
    out = args.out or (Path(args.out_root) / "power_inputs" / "pilot_timing.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, timing)
    print(
        f"[timing] wrote {out}: all-in {timing['gpu_hours_all_in']:.2f} GPU-h, "
        f"gen marginal {timing['gen_marginal_s_per_response_per_gpu']:.4f} s/resp/GPU, "
        f"capture {timing['capture_rows_per_s_per_gpu']:.2f} rows/s/GPU, "
        f"engine init sum {timing['fixed_overhead_hours'] * 3600:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
