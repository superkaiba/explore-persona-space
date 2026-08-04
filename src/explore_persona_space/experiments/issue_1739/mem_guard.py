"""Pre-fit host-RSS guard for the #1739 fits CLI (crash-fix r3, rc=137 kills).

Five of ten 2026-08-02 ``newarm*`` GCE boxes were kernel-OOM-killed (rc=137,
hard kill, log tail lost) during the pilot. Reconstructed arithmetic
(d=3584, Ly=28, fp64 = 8 B):

- whitening-apply + linear map fit at full-U (n=18793): whitened x/y copies
  (2 x 15.05 GiB) + the 80/20 split copies inside ``fit_linear_map``
  (~30 GiB) + weight tensors — ~75 GiB ADDITIONAL peak. Killed the two
  a2-highgpu-1g boxes (85 GB RAM; the spot-A100-40 ladder rung), which the
  ladder substituted for the a2-ultragpu-1g (170 GB) shape every prior
  round ran on.
- the hall transfer comb at L=16000 (n_comb = 23188): comb z/za
  (2 x 18.6 GiB) + pre-r3 whole-array copies inside ``run_cell_multi``
  (z, za, unconditional ``mp``) — >100 GiB additional; killed two 170 GB
  boxes at transfer unit 3/3.

The r3 chunking/aliasing fixes cut those peaks; THIS module is the designed
backstop: project each heavy phase's ADDITIONAL peak from shape arithmetic
at PHASE ENTRY and REFUSE with a designed rc (:data:`RSS_GUARD_RC`) + a
report artifact when the projection exceeds the box's live MemAvailable —
a designed halt with a diagnostic beats a kernel kill that loses the log
tail (the #1415 designed-halt convention: report JSON + distinct rc, never
a bare rc=1).

Knobs: ``EPM_I1739_RSS_GUARD=0`` -> log-only (projection printed, never
refuses); ``EPM_I1739_RSS_GUARD_HEADROOM_GIB`` (default 4.0) -> free-RAM
reserve subtracted from MemAvailable before the comparison. On a host
without /proc (non-Linux test envs) the guard logs ``avail=?`` and passes
(fail-open — the kernel kill is no worse than before).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RSS_GUARD_RC = 9  # designed-halt rc — distinct from the pilot fence's rc=7

GIB = float(2**30)

# Arm-slug classes mirrored from arms.ARM_REGISTRY (kept literal here so the
# guard stays import-light; run_cell_multi is the source of truth for which
# arm consumes which array — see the mp/MLP/ridge dispatch blocks there).
MP_ARMS = frozenset(
    {
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm9_pretrain_ft",
        "arm10_stacked",
        "arm13_shuffled_map",
        "arm14_shuffled_pt",
    }
)
MLP_ARMS = frozenset({"arm5_mlp_ctx", "arm17_oracle_mlp"})
RIDGE_ARMS = frozenset(
    {
        "arm4_ridge_ctx",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm10_stacked",
        "arm12_oracle_reg",
        "arm15_text_only",
        "arm16_surface_feat",
    }
)
KRR_ARMS = frozenset({"arm18_oracle_krr"})


class MemGuardRefusal(RuntimeError):
    """A phase's projected peak provably exceeds available host RAM."""


def _proc_field(path: str, key: str) -> int | None:
    """Read one ``key: <n> kB`` field from a /proc file -> bytes, or None."""
    try:
        with open(path, encoding="ascii") as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def mem_available_bytes() -> int | None:
    """Live MemAvailable (bytes) — the kernel's own reclaimable estimate."""
    return _proc_field("/proc/meminfo", "MemAvailable:")


def rss_bytes() -> int | None:
    """This process's current VmRSS (bytes)."""
    return _proc_field("/proc/self/status", "VmRSS:")


def _f8(n_layers: int, n_rows: int, d: int) -> int:
    """Bytes of one (Ly, n, d) float64 array."""
    return 8 * n_layers * n_rows * d


def whitening_map_components(
    n_layers: int,
    n_rows: int,
    d: int,
    *,
    n_ctx: int,
    n_ev: int = 0,
    map_fit: bool = True,
    layer_chunk: int = 8,
) -> dict[str, int]:
    """Projected ADDITIONAL bytes for one group's whitening-apply + map stage.

    ``n_rows`` = the U-pool rung; ``n_ctx`` / ``n_ev`` = labeled train / eval
    contexts (their whitened fp64 copies are built in the same stage).
    ``map_fit=False`` (a persisted nonlinear map will be LOADED) drops the
    whitened-U + split-fit terms — only the labeled whitening remains.
    """
    comp: dict[str, int] = {
        # fit_whitening per-chunk fp64 temporaries (post-r3 chunked cast)
        "whitening_chunk_temps": 2 * 8 * min(layer_chunk, n_layers) * n_rows * d,
        # wh.w + (map w + torch chunk copies)
        "weight_tensors": 3 * 8 * n_layers * d * d,
        # z_var_w + za_w (+ z_ev_w + za_ev_w on the transfer leg)
        "whitened_labeled": 2 * _f8(n_layers, n_ctx, d) + 2 * _f8(n_layers, n_ev, d),
    }
    if map_fit:
        comp["whitened_u_x_y"] = 2 * _f8(n_layers, n_rows, d)
        # fit_linear_map internals: x_tr + y_tr (80%) + x_ho + y_ho +
        # preds_hold (20% each) — freed before the full-pool refit (r3).
        comp["map_split_copies"] = int(2.2 * _f8(n_layers, n_rows, d))
    return comp


def cell_solve_components(
    n_layers: int,
    n_rows: int,
    d: int,
    roster: list[str] | tuple[str, ...],
    *,
    has_map: bool,
    alias_rows: bool = False,
) -> dict[str, int]:
    """Projected ADDITIONAL bytes for one ``run_cell_multi`` solve at n_rows.

    ``alias_rows=True`` = the transfer leg's identity row set (post-r3 the
    z/za copies are aliased away); the main grid's subset cells keep them.
    """
    want = set(roster)
    f8 = _f8(n_layers, n_rows, d)
    comp: dict[str, int] = {}
    if not alias_rows:
        comp["z_za_copies"] = 2 * f8
    if has_map and want & MP_ARMS:
        comp["mp"] = f8 + 2 * 8 * n_rows * d  # + per-layer apply temporaries
    if want & MLP_ARMS:
        comp["mlp_fp32"] = 4 * n_layers * n_rows * d
    if want & RIDGE_ARMS:
        # per-job torch chunk copies + y stacks (layer_chunk=4, ~6 tensors)
        comp["ridge_transient"] = int(0.35 * f8)
    if want & KRR_ARMS:
        # layer_chunk<=2 fp64 tensors (xt, xev, phi_*) inside the Nystrom fit
        comp["krr_chunk"] = 4 * 8 * 2 * n_rows * d
    return comp


def transfer_components(
    n_layers: int,
    n_tr: int,
    n_ev: int,
    d: int,
    roster: list[str] | tuple[str, ...],
    *,
    has_map: bool,
) -> dict[str, int]:
    """Projected ADDITIONAL bytes for one transfer unit at budget n_tr."""
    n_comb = n_tr + n_ev
    comp = {"comb_z_za": 2 * _f8(n_layers, n_comb, d)}
    comp.update(
        cell_solve_components(n_layers, n_comb, d, roster, has_map=has_map, alias_rows=True)
    )
    return comp


def check_phase(
    phase: str,
    components: dict[str, int],
    *,
    out_root: Path | str | None = None,
) -> dict:
    """Project one phase's additional peak vs live MemAvailable; refuse loud.

    Always prints one ``[fits][rss-guard]`` line (the fix-engaged signal on
    healthy boxes). On a provable excess: writes/extends
    ``<out_root>/rss_guard_report.json`` and raises :class:`MemGuardRefusal`
    (the CLI maps it to :data:`RSS_GUARD_RC`) unless ``EPM_I1739_RSS_GUARD=0``
    (log-only). Missing /proc (avail unknown) fails OPEN.
    """
    projected = int(sum(components.values()))
    avail = mem_available_bytes()
    rss = rss_bytes()
    headroom = float(os.environ.get("EPM_I1739_RSS_GUARD_HEADROOM_GIB", "4")) * GIB
    enforce = os.environ.get("EPM_I1739_RSS_GUARD", "1") != "0"
    over = avail is not None and projected > max(avail - headroom, 0.0)
    verdict = "REFUSE" if (over and enforce) else ("over-log-only" if over else "ok")
    record = {
        "phase": phase,
        "projected_extra_gib": round(projected / GIB, 2),
        "mem_available_gib": None if avail is None else round(avail / GIB, 2),
        "rss_gib": None if rss is None else round(rss / GIB, 2),
        "headroom_gib": round(headroom / GIB, 2),
        "components_gib": {k: round(v / GIB, 2) for k, v in components.items()},
        "verdict": verdict,
        "enforce": enforce,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(
        f"[fits][rss-guard] phase={phase} projected=+{projected / GIB:.1f} GiB "
        f"avail={'?' if avail is None else f'{avail / GIB:.1f}'} GiB "
        f"rss={'?' if rss is None else f'{rss / GIB:.1f}'} GiB verdict={verdict}",
        flush=True,
    )
    if over and out_root is not None:
        path = Path(out_root) / "rss_guard_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checks": []}
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning("[rss-guard] unreadable prior report at %s — rewriting", path)
        payload.setdefault("checks", []).append(record)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, path)
    if over and enforce:
        raise MemGuardRefusal(
            f"phase {phase} projects +{projected / GIB:.1f} GiB over "
            f"{(avail or 0) / GIB:.1f} GiB available (headroom {headroom / GIB:.1f} GiB) — "
            f"designed halt rc={RSS_GUARD_RC}; relaunch on a bigger-RAM box "
            "(a2-ultragpu-1g, 170 GB — declare --min-gpu-mem-gb > 38 so the GCP "
            "ladder skips the 85 GB a2-highgpu-1g rung) or shrink the slice"
        )
    return record
