"""Issue #1689 follow-up round ``user-slot-recapture`` — Phase B (capture).

Batched teacher-forced layer-19 activation capture over the rendered rows of
:mod:`scripts.issue1689_user_slot_render`, sharded across EVERY visible GPU.

Two modes, ONE code path (the dispatch mode is what ``--smoke`` also runs):

  ``--mode dispatch`` (default) — enumerate units from the render manifest
    (rendering them first when absent), assign units to the visible GPUs by
    longest-processing-time-first greedy on measured cost (rows x median
    tokens), spawn ONE worker subprocess per GPU with ``CUDA_VISIBLE_DEVICES``
    pinned in the CHILD ENV plus the matching ``--gpu-id`` (the in-process CVD
    clobber is silently defeated by any import-time cuInit — gotchas.md), wait,
    then upload every store in ONE ``upload_folder`` commit, verify the exact
    expected path set on the Hub, and write the JSON sentinel with the exit code.

  ``--mode worker`` — load ONE model, walk its assigned units, and for each unit
    run right-padded batched forwards, reading every declared slot from
    ``hidden_states[19]`` at the token index resolved from the render's char
    offset via ``offset_mapping``.

Capture discipline:

  * Layer 19 ONLY — a declared deviation from the parent's (14, 18, 19, 26)
    capture set. The parent fit L19 as its headline and this round's reads are
    all L19, so the other three layers would be ~4x store for zero reads.
  * RIGHT padding + attention_mask. Right padding under a causal mask means pad
    positions can never influence a real position and every slot index is the
    UNPADDED index — so no ``position_ids`` threading is needed (the left-pad
    RoPE trap, #502, is avoided by construction rather than patched).
  * ``logits_to_keep=1`` when the model's forward signature names it: this rig
    never reads ``out.logits``, and the default keeps full-vocab logits for
    every position — a silent ``B x T x 152064`` bf16 allocation (#779 OOM).
  * Token-index resolution reuses the parent's straddler policy per slot:
    EXCLUDE for X-side slots (no later content leaks into X), INCLUDE for
    end-of-content slots. Every excluded straddler is recorded per row per slot
    in ``seam_flags`` (#1315): under the naturalistic framing the ``User: ``
    label's trailing space merges into u2's first word on essentially every
    letter-initial u2, so the flag is expected-dense there, not anomalous.
  * The render's ``n_tokens`` is re-asserted against this rig's own
    tokenization — a cheap render/capture drift gate.

Smoke (``--smoke``): 2 units x 32 rows through the IDENTICAL dispatch path,
including the REAL Hub upload + verify branch, redirected to the
``user_slot_recapture_smoke_probe/`` prefix so the canonical prefix is never
touched (the fenced-branch live-probe rule). Process WIDTH is NOT narrowed.

Content hygiene: never prints row text — only counts, token positions, shapes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_user_slot_render.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from scripts.issue1689_common import HEADLINE_LAYER, HF_DATA_PREFIX  # noqa: E402
from scripts.issue1689_user_slot_render import (  # noqa: E402
    DATA_REPO,
    GRID_SLOT_KINDS,
    GRID_X_KINDS,
    GRID_Y_KINDS,
    ROUND_LABEL,
    SLOT_STRADDLER_POLICY,
    base_metadata,
)

CAPTURE_LAYER = HEADLINE_LAYER  # 19 — declared single-layer deviation
HF_STORE_PREFIX = f"{HF_DATA_PREFIX}/{ROUND_LABEL}"
HF_SMOKE_PREFIX = f"{HF_DATA_PREFIX}/{ROUND_LABEL}_smoke_probe"
SENTINEL_DIR = Path("/workspace/logs")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    """Text-mode iteration only (`splitlines()` shreds U+2028/NEL — gotchas.md)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest(rendered_dir: Path) -> dict:
    path = rendered_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"render manifest missing: {path} (run the render phase first)")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def visible_devices() -> list[str]:
    """The physical device IDs this process may use, in allocation order.

    Indexes INTO a pre-set ``CUDA_VISIBLE_DEVICES`` when present, and only falls
    back to an `nvidia-smi` enumeration when it is unset. Both halves are
    load-bearing:

      * `nvidia-smi` (never `torch.cuda.device_count()`, which reads a
        possibly-clobbered CVD and caches it — gotchas.md, #1112 shape (b));
      * but `nvidia-smi` lists EVERY host GPU regardless of CVD, so on a shared
        SLURM node (the `fellows` lane — FIRST in the default auto chain) an
        absolute `0..n-1` fan-out clobbers the scheduler's allocation onto other
        users' occupied devices (the #1345 crash-fix 15771 shape: vLLM died at
        19 GiB free). The pre-set allocation wins whenever it exists.
    """
    pre = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if pre:
        return [d.strip() for d in pre.split(",") if d.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    return [ln.strip() for ln in out.stdout.split("\n") if ln.strip()]


def assign_units_to_gpus(entries: list[dict], n_gpus: int) -> dict[int, list[str]]:
    """Balance units over GPUs: one model per GPU minimum, cells fanned out
    within a model.

    Each worker loads ONE model, so units are first split by model and the GPU
    pool is divided between the models in proportion to their total cost; within
    a model, units are assigned longest-processing-time-first (LPT greedy, the
    standard 4/3-approximation for makespan) on cost = rows x median tokens.
    """
    if n_gpus <= 0:
        raise RuntimeError("no visible GPUs — capture requires at least one")
    by_model: dict[str, list[dict]] = {}
    for e in entries:
        by_model.setdefault(e["model"], []).append(e)
    cost_of = lambda e: max(1, int(e["n_rows"]) * max(1, int(e.get("token_len_p50", 1))))  # noqa: E731
    model_cost = {m: sum(cost_of(e) for e in es) for m, es in by_model.items()}
    models = sorted(by_model, key=lambda m: -model_cost[m])
    # Proportional GPU split with >=1 GPU per model whenever the pool allows.
    total = sum(model_cost.values()) or 1
    quota: dict[str, int] = {}
    if n_gpus >= len(models):
        for m in models:
            quota[m] = max(1, round(n_gpus * model_cost[m] / total))
        # Reconcile rounding drift against the real pool size.
        while sum(quota.values()) > n_gpus:
            quota[max(quota, key=lambda m: quota[m])] -= 1
        while sum(quota.values()) < n_gpus:
            quota[max(quota, key=lambda m: model_cost[m] / quota[m])] += 1
    else:
        # Fewer GPUs than models: every GPU serves one model, models queue.
        for i, m in enumerate(models):
            quota[m] = 1 if i < n_gpus else 0
    plan: dict[int, list[str]] = {g: [] for g in range(n_gpus)}
    load: dict[int, int] = {g: 0 for g in range(n_gpus)}
    next_gpu = 0
    for m in models:
        k = quota[m]
        if k == 0:
            # No dedicated GPU: append to the least-loaded GPU (serial after its
            # own model's units — the worker reloads the model for this unit).
            gpus = [min(load, key=lambda g: load[g])]
        else:
            gpus = list(range(next_gpu, next_gpu + k))
            next_gpu += k
        for e in sorted(by_model[m], key=lambda e: -cost_of(e)):
            g = min(gpus, key=lambda g: load[g])
            plan[g].append(e["unit_id"])
            load[g] += cost_of(e)
    return plan


def _logits_to_keep_kwargs(model) -> dict:
    """`{"logits_to_keep": 1}` iff the forward signature NAMES that parameter.

    This rig never reads `out.logits`, and transformers >= 4.49 computes
    full-vocab logits for EVERY position by default — a silent
    `B x T x 152064` allocation. A bare `**kwargs` does not count (a wrapper
    would swallow or crash on it), and a future rename degrades to the old
    full-logits behavior instead of an unexpected-kwarg crash (gotchas.md).
    """
    fn = getattr(model, "forward", None) or model.__call__
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    p = params.get("logits_to_keep")
    if p is not None and p.kind is not inspect.Parameter.VAR_KEYWORD:
        return {"logits_to_keep": 1}
    return {}


def resolve_slot_token(
    offsets: list[tuple[int, int]], boundary: int, *, straddler_include: bool
) -> tuple[int, bool]:
    """Char boundary -> (token index, straddler_excluded).

    The LAST token whose span ends at or before ``boundary``; a token that
    STRADDLES the boundary (``s < boundary < e``) is included only when
    ``straddler_include``. Mirrors the parent capture rig's policy so the
    parent-convention slot reproduces byte-for-byte, and reports whether a
    straddler was dropped (the #1315 ``seam_flags`` record).
    """
    last = -1
    straddled = False
    for i, (s, e) in enumerate(offsets):
        if e <= boundary:
            last = i
        elif s < boundary < e:
            straddled = True
            if straddler_include:
                last = i
            break
        elif s >= boundary:
            break
    if last < 0:
        raise ValueError(f"char boundary {boundary} resolved to no token")
    return last, (straddled and not straddler_include)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def resolve_grid_positions(
    spans: list[tuple[int, int]], group: dict
) -> tuple[dict[str, int], tuple[int, int], bool]:
    """Addendum-E X x Y token positions for one read group.

    Returns ``(positions, (answer_first, answer_last), x_straddle_is_merged)``.

      X_clean     last token FULLY BEFORE the answer span (straddler-EXCLUSIVE
                  — the plain-text label's ':', the chat header's '\\n', the
                  story open quote). PRIMARY, and the #1345 convention.
      X_straddle  X_clean + 1 == the token that STARTS the answer. In plain text
                  that is the space-merged-with-first-answer-word token (the
                  PARENT's straddler-INCLUSIVE read); in chat/story it is simply
                  the first answer token — ``x_straddle_is_merged`` says which.
      Y_end       last token of the answer content (straddler-INCLUSIVE).
      Y_boundary  the response-slot token before the next character speaks
                  (straddler-EXCLUSIVE at the transition's own boundary).

    Y_mean is a MEAN over ``answer_first..answer_last`` inclusive, so the span is
    returned rather than a single index.
    """
    x_clean, _ = resolve_slot_token(spans, int(group["answer_start"]), straddler_include=False)
    x_straddle = x_clean + 1
    y_end, _ = resolve_slot_token(spans, int(group["answer_end"]), straddler_include=True)
    y_boundary, _ = resolve_slot_token(spans, int(group["boundary_end"]), straddler_include=False)
    if not (x_clean < x_straddle <= y_end <= y_boundary):
        raise RuntimeError(
            f"group {group['name']}: non-monotonic grid positions "
            f"X_clean={x_clean} X_straddle={x_straddle} Y_end={y_end} Y_boundary={y_boundary}"
        )
    # The straddle token is "merged" when it starts BEFORE the answer's first
    # character — i.e. it fuses preceding separator text with the answer's first
    # word (the plain-text case the comparison arm exists to price).
    x_straddle_is_merged = spans[x_straddle][0] < int(group["answer_start"])
    positions = {
        "X_clean": x_clean,
        "X_straddle": x_straddle,
        "Y_end": y_end,
        "Y_boundary": y_boundary,
    }
    return positions, (x_straddle, y_end), x_straddle_is_merged


def capture_unit(
    entry: dict,
    rows: list[dict],
    *,
    model,
    tokenizer,
    batch_size: int,
    device,
) -> dict:
    """Batched teacher-forced L19 capture for one unit; returns the store dict."""
    import numpy as np
    import torch

    slots: tuple[str, ...] = tuple(entry["slots"])
    policy = {s: SLOT_STRADDLER_POLICY[s] for s in slots}
    d_model = int(model.config.hidden_size)
    n = len(rows)
    acc = {s: np.zeros((n, d_model), dtype=np.float32) for s in slots}
    pos = {s: np.zeros(n, dtype=np.int32) for s in slots}
    seam = {s: np.zeros(n, dtype=np.int8) for s in slots}
    # Addendum E: 5 grid slots per read group (X_clean, X_straddle, Y_mean,
    # Y_end, Y_boundary). Group names + membership are row-independent.
    group_names: list[str] = [g["name"] for g in rows[0].get("read_groups", [])]
    if not group_names:
        # A rendered set produced BEFORE addendum E carries no read_groups, so the
        # grid degrades to empty everywhere (self-consistent, and the fits'
        # projection then charges 0 grid hours) — but say so loudly rather than
        # leaving the missing X x Y grid to be discovered downstream.
        print(
            "[capture] WARN no read_groups on the rendered rows — this render "
            "predates the addendum-E X x Y grid; re-run the render to capture it",
            flush=True,
        )
    grid_slot_names = [f"{gn}__{k}" for gn in group_names for k in GRID_SLOT_KINDS]
    gacc = {s: np.zeros((n, d_model), dtype=np.float32) for s in grid_slot_names}
    gpos = {
        f"{gn}__{k}": np.zeros(n, dtype=np.int32)
        for gn in group_names
        for k in ("X_clean", "X_straddle", "Y_end", "Y_boundary")
    }
    gspan = {gn: np.zeros((n, 2), dtype=np.int32) for gn in group_names}
    gmerged = {gn: np.zeros(n, dtype=np.int8) for gn in group_names}
    n_tokens = np.zeros(n, dtype=np.int32)
    keep_kwargs = _logits_to_keep_kwargs(model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")

    for start in range(0, n, batch_size):
        chunk = rows[start : start + batch_size]
        ids_list: list[list[int]] = []
        for j, row in enumerate(chunk):
            enc = tokenizer(row["text"], add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            if len(ids) != int(row["n_tokens"]):
                raise RuntimeError(
                    f"{entry['unit_id']} row {row['row_index']}: render/capture tokenization "
                    f"drift {row['n_tokens']} != {len(ids)}"
                )
            ids_list.append(ids)
            i = start + j
            n_tokens[i] = len(ids)
            for s in slots:
                tok_i, dropped = resolve_slot_token(
                    offs, int(row["char_slots"][s]), straddler_include=(policy[s] == "include")
                )
                if not (0 <= tok_i < len(ids)):
                    raise RuntimeError(
                        f"{entry['unit_id']} row {row['row_index']}: slot {s} token {tok_i} "
                        f"out of range {len(ids)}"
                    )
                pos[s][i] = tok_i
                seam[s][i] = 1 if dropped else 0
            for g in row.get("read_groups", []):
                gp, (a_first, a_last), merged = resolve_grid_positions(offs, g)
                if not (0 <= a_first <= a_last < len(ids)):
                    raise RuntimeError(
                        f"{entry['unit_id']} row {row['row_index']}: group {g['name']} span "
                        f"{a_first}..{a_last} out of range {len(ids)}"
                    )
                for kind, tok_i in gp.items():
                    gpos[f"{g['name']}__{kind}"][i] = tok_i
                gspan[g["name"]][i] = (a_first, a_last)
                gmerged[g["name"]][i] = 1 if merged else 0
            # Every slot must sit at a DISTINCT position; equal positions are
            # exactly the realized defect this round fixes.
            got = [int(pos[s][i]) for s in slots]
            if len(set(got)) != len(got):
                raise RuntimeError(
                    f"{entry['unit_id']} row {row['row_index']}: degenerate slot positions "
                    f"{dict(zip(slots, got, strict=True))}"
                )

        width = max(len(x) for x in ids_list)
        # RIGHT padding: under a causal mask a pad can never influence a real
        # position, and slot indices stay the UNPADDED indices.
        batch_ids = torch.full((len(ids_list), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), width), dtype=torch.long)
        for j, ids in enumerate(ids_list):
            batch_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[j, : len(ids)] = 1
        batch_ids = batch_ids.to(device)
        attn = attn.to(device)
        with torch.no_grad():
            out = model(
                input_ids=batch_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
                **keep_kwargs,
            )
            hs = out.hidden_states[CAPTURE_LAYER]  # (B, T, D)
            for s in slots:
                idx = torch.tensor(
                    [int(pos[s][start + j]) for j in range(len(ids_list))],
                    dtype=torch.long,
                    device=device,
                )
                # Gather one position per batch row in ONE indexed read.
                picked = hs[torch.arange(len(ids_list), device=device), idx]  # (B, D)
                acc[s][start : start + len(ids_list)] = picked.float().cpu().numpy()
            # Addendum E grid reads — SAME forward, no extra model compute; the
            # Y_mean is one masked reduce over positions already resident.
            brange = torch.arange(len(ids_list), device=device)
            ar = torch.arange(hs.shape[1], device=device).unsqueeze(0)
            for gn in group_names:
                for kind in ("X_clean", "X_straddle", "Y_end", "Y_boundary"):
                    key = f"{gn}__{kind}"
                    gidx = torch.tensor(
                        [int(gpos[key][start + j]) for j in range(len(ids_list))],
                        dtype=torch.long,
                        device=device,
                    )
                    gacc[key][start : start + len(ids_list)] = (
                        hs[brange, gidx].float().cpu().numpy()
                    )
                lo = torch.tensor(
                    [int(gspan[gn][start + j, 0]) for j in range(len(ids_list))],
                    dtype=torch.long,
                    device=device,
                )
                hi = torch.tensor(
                    [int(gspan[gn][start + j, 1]) for j in range(len(ids_list))],
                    dtype=torch.long,
                    device=device,
                )
                mask = ((ar >= lo.unsqueeze(1)) & (ar <= hi.unsqueeze(1))).to(hs.dtype)
                denom = mask.sum(1, keepdim=True).clamp(min=1.0)
                mean = (hs * mask.unsqueeze(-1)).sum(1) / denom
                gacc[f"{gn}__Y_mean"][start : start + len(ids_list)] = mean.float().cpu().numpy()
        del out, hs
        print(
            f"[capture] {entry['unit_id']} rows {min(start + batch_size, n)}/{n} width={width}",
            flush=True,
        )

    store = {
        "slots": acc,
        "slot_token_pos": pos,
        "seam_flags": seam,
        # Addendum E grid payload.
        "grid_slots": gacc,
        "grid_slot_pos": gpos,
        "grid_answer_span": gspan,
        "grid_x_straddle_is_merged": gmerged,
        "grid_group_names": group_names,
        "grid_slot_kinds": list(GRID_SLOT_KINDS),
        "grid_x_kinds": list(GRID_X_KINDS),
        "grid_y_kinds": list(GRID_Y_KINDS),
        "n_tokens": n_tokens,
        "conv_ids": np.array([r["conv_id"] for r in rows], dtype=object),
        "dup_count": np.array([int(r["dup_count"]) for r in rows], dtype=np.int32),
        "row_index": np.array([int(r["row_index"]) for r in rows], dtype=np.int32),
        "judge_score_mean": np.array(
            [
                float("nan") if r.get("judge_score_mean") is None else float(r["judge_score_mean"])
                for r in rows
            ],
            dtype=np.float32,
        ),
        "unit": {k: entry[k] for k in ("unit_id", "model", "framing", "provenance", "variant")},
        "slot_names": list(slots),
        "straddler_policy": policy,
        "fit_pairs": entry["fit_pairs"],
        "primary_fit": entry["primary_fit"],
        "layer": CAPTURE_LAYER,
        "d_model": d_model,
        "n_rows": n,
        "metadata": base_metadata(),
    }
    frac = {s: float(seam[s].mean()) for s in slots}
    merged_frac = {gn: float(gmerged[gn].mean()) for gn in group_names}
    print(
        f"[capture] {entry['unit_id']} grid groups={group_names} "
        f"x_straddle_merged_frac={merged_frac}",
        flush=True,
    )
    print(
        f"[capture] {entry['unit_id']} done n={n} d={d_model} straddler_excluded_frac={frac}",
        flush=True,
    )
    return store


def run_worker(args, manifest: dict) -> int:
    """Load ONE model, capture every assigned unit, write per-unit stores."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    wanted = [u.strip() for u in args.units.split(",") if u.strip()]
    if not wanted:
        print("[capture-worker] no units assigned — exiting 0", flush=True)
        return 0
    entries = {e["unit_id"]: e for e in manifest["units"]}
    my = [entries[u] for u in wanted]
    models = {e["model"] for e in my}
    rendered_dir = args.rendered_dir

    # CVD is pinned in the LAUNCHER env by the dispatcher, so this process sees
    # exactly one device and `cuda:0` IS the pinned physical GPU. Never index by
    # --gpu-id here (that is the launcher-pinned worker class, gotchas.md).
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_map = {"": 0}
        dtype = torch.bfloat16
    elif args.allow_cpu:
        # Device SELECTION only — every other line below is byte-identical to the
        # GPU path. This is the tiny-real CPU end-to-end knob; the dispatcher
        # never passes it, so a mis-pinned production worker still fails loud
        # instead of silently grinding on CPU.
        device = torch.device("cpu")
        device_map = {"": "cpu"}
        dtype = torch.float32
    else:
        raise RuntimeError(
            "worker sees no CUDA device (CVD pin or driver problem); pass --allow-cpu "
            "only for the tiny-real CPU end-to-end run"
        )
    print(
        f"[capture-worker] gpu_id={args.gpu_id} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"device={device} units={len(my)} models={sorted(models)} "
        f"weights_override={args.weights_override or None}",
        flush=True,
    )

    for model_name in sorted(models):
        # Tokenizer ALWAYS comes from the manifest's real model (real BPE ids,
        # real chat template); --weights-override replaces the GPU-scale WEIGHTS
        # only, which is the sole fake in the tiny-real CPU end-to-end run.
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.weights_override or model_name, torch_dtype=dtype, device_map=device_map
        )
        model.eval()
        try:
            for entry in [e for e in my if e["model"] == model_name]:
                out_path = (
                    args.out_root / entry["model_dir"] / entry["unit_id"] / f"L{CAPTURE_LAYER}.pt"
                )
                if out_path.exists() and not args.overwrite:
                    print(f"[capture-worker] skip existing {out_path}", flush=True)
                    continue
                rows = read_jsonl(rendered_dir / entry["rendered_path"])
                if args.max_rows:
                    rows = rows[: args.max_rows]
                store = capture_unit(
                    entry,
                    rows,
                    model=model,
                    tokenizer=tok,
                    batch_size=args.batch_size,
                    device=device,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(".pt.tmp")
                torch.save(store, tmp)
                os.replace(tmp, out_path)
                print(f"[capture-worker] wrote {out_path}", flush=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _thread_cap_env() -> dict:
    """Pod/GPU workers get full width — no shared-VM thread caps here."""
    return {}


def run_render_if_missing(args) -> None:
    manifest_path = args.rendered_dir / "manifest.json"
    if manifest_path.exists() and not args.force_render:
        print(f"[capture] reusing render manifest {manifest_path}", flush=True)
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "issue1689_user_slot_render.py"),
        "--out-dir",
        str(args.rendered_dir),
        "--stage-root",
        str(args.stage_root),
    ]
    # Same-job path (16177 crash): the a1 generator's output exists LOCALLY but
    # is absent from the Hub at the render's pinned PARENT_REVISION (uploaded
    # this job, after the pin), so the render's gen_a1 prefix stages empty and
    # the addendum-C loader fail-louds. Thread the local dir exactly as the
    # render's --gen-a1-dir contract designs; skip when empty/absent so a
    # fresh-instance replay keeps the Hub path + fail-loud behavior.
    if args.gen_a1_dir is not None and any(args.gen_a1_dir.glob("user_slot_a1_onpolicy_*.jsonl")):
        cmd += ["--gen-a1-dir", str(args.gen_a1_dir)]
    if args.smoke:
        cmd.append("--smoke")
    print(f"[capture] rendering: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env={**os.environ}, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"render phase failed rc={rc}")


def upload_stores(out_root: Path, *, prefix: str) -> list[str]:
    """ONE `upload_folder` commit for the whole store tree, then verify the
    EXACT expected path set on the Hub (never a per-file upload loop — the
    ~1M-file data repo 504-storms on per-file tree pre-checks)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        assert_hub_dir_filecounts,
        retry_transient,
        verify_repo_paths_uploaded,
    )

    local = sorted(p for p in out_root.rglob("*.pt") if p.is_file())
    if not local:
        raise RuntimeError(f"no .pt stores to upload under {out_root}")
    expected = [f"{prefix}/{p.relative_to(out_root).as_posix()}" for p in local]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    allow = ["**/*.pt", "**/*.json"]
    # The Hub rejects any single repo DIRECTORY receiving >10k files in one
    # commit with a NON-retriable 400, fired after every byte is staged (#658).
    # This tree is one L19.pt per unit dir, so the guard is cheap insurance —
    # and it runs OUTSIDE the retry wrapper because a guard raise is
    # deterministic (retrying it would burn the retry budget for nothing).
    counts = assert_hub_dir_filecounts(out_root, prefix, allow_patterns=allow)
    print(f"[capture] hub dir-filecount guard OK: {counts}", flush=True)
    retry_transient(
        lambda: api.upload_folder(
            folder_path=str(out_root),
            path_in_repo=prefix,
            repo_id=DATA_REPO,
            repo_type="dataset",
            allow_patterns=allow,
        ),
        what=f"upload_folder({prefix})",
    )
    # `path_in_repo` is REQUIRED KEYWORD-ONLY — it scopes the verify walk so the
    # ~1M-file data repo is never listed unscoped (#920/#833).
    missing = verify_repo_paths_uploaded(
        api, DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"upload verification FAILED — missing {len(missing)}: {sorted(missing)[:10]}"
        )
    print(f"[capture] uploaded + verified {len(expected)} files under {prefix}", flush=True)
    return expected


def write_sentinel(path: Path, payload: dict) -> None:
    """Sentinel for the VM poller: written LAST, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    print(f"[capture] sentinel -> {path}", flush=True)


def run_dispatch(args) -> int:
    run_render_if_missing(args)
    manifest = load_manifest(args.rendered_dir)
    entries = manifest["units"]
    if args.units != "all":
        want = {u.strip() for u in args.units.split(",") if u.strip()}
        entries = [e for e in entries if e["unit_id"] in want]
    if not entries:
        raise RuntimeError("no units selected")

    # `devices` are PHYSICAL ids in allocation order; the plan is keyed on the
    # LANE index 0..n-1 and mapped through `devices` at launch, so a pre-set
    # CVD allocation is honored instead of clobbered (see `visible_devices`).
    devices = visible_devices()
    if args.num_gpus:
        devices = devices[: args.num_gpus]
    n_gpus = len(devices)
    plan = assign_units_to_gpus(entries, n_gpus)
    print(
        f"[capture] dispatch: {len(entries)} units over {n_gpus} GPUs "
        f"(devices {devices}) -> " + json.dumps({devices[g]: len(v) for g, v in plan.items()}),
        flush=True,
    )

    procs = []
    for lane, unit_ids in plan.items():
        gpu = devices[lane]
        env = {**os.environ, **_thread_cap_env()}
        # BOTH the launcher-env CVD pin AND the matching --gpu-id: the
        # in-process clobber alone is defeated by any import-time cuInit.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--gpu-id",
            str(gpu),
            "--units",
            ",".join(unit_ids),
            "--rendered-dir",
            str(args.rendered_dir),
            "--out-root",
            str(args.out_root),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.max_rows:
            cmd += ["--max-rows", str(args.max_rows)]
        if args.overwrite:
            cmd.append("--overwrite")
        log = args.log_dir / f"worker_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", encoding="utf-8")
        print(f"[capture] gpu{gpu}: {len(unit_ids)} units -> {log}", flush=True)
        procs.append(
            (
                gpu,
                subprocess.Popen(
                    cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
                ),
                fh,
                log,
            )
        )

    failures: list[tuple[int, int]] = []
    for gpu, proc, fh, log in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            failures.append((gpu, rc))
            # JSONL_SPLITLINES_EXEMPT: worker LOG tail for crash diagnostics, not JSONL content
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
            print(f"[capture] gpu{gpu} FAILED rc={rc}; tail of {log}:", flush=True)
            for ln in tail:
                print(f"    {ln}", flush=True)
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    produced = sorted(p for p in args.out_root.rglob("*.pt"))
    expect_n = len(entries)
    if len(produced) != expect_n:
        raise RuntimeError(f"expected {expect_n} store files, found {len(produced)}")
    # Copy the render manifest next to the stores so the fits + any consumer
    # resolve the unit spec from the SAME tree the stores came from.
    (args.out_root / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    prefix = HF_SMOKE_PREFIX if args.smoke else HF_STORE_PREFIX
    uploaded: list[str] = []
    if args.skip_upload:
        print("[capture] --skip-upload: NOT uploading (upload branch unexercised)", flush=True)
    else:
        uploaded = upload_stores(args.out_root, prefix=prefix)

    payload = {
        "issue": 1689,
        "round": ROUND_LABEL,
        "phase": "capture",
        "status": "ok",
        "exit_code": 0,
        "smoke": bool(args.smoke),
        "layer": CAPTURE_LAYER,
        "n_units": expect_n,
        "n_rows_total": int(sum(e["n_rows"] for e in entries)),
        "n_gpus": n_gpus,
        "store_root": str(args.out_root),
        "hf_prefix": prefix,
        "n_uploaded": len(uploaded),
        "metadata": base_metadata(),
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    write_sentinel(args.sentinel, payload)
    print("[phase=done]", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dispatch", "worker"], default="dispatch")
    ap.add_argument(
        "--rendered-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "rendered",
    )
    ap.add_argument(
        "--gen-a1-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "gen_a1",
        help="local a1-generator output dir threaded to the render when non-empty "
        "(the same-job path; the render otherwise reads the gen_a1 Hub prefix)",
    )
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "hf_dl",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "store",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "logs",
    )
    ap.add_argument(
        "--sentinel",
        type=Path,
        default=SENTINEL_DIR / "issue-1689-user-slot-recapture.json",
    )
    ap.add_argument("--units", default="all")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=0, help="0 = every visible GPU")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="worker only: run on CPU when no CUDA device is visible (device "
        "SELECTION only — the tiny-real CPU end-to-end knob; never passed in production)",
    )
    ap.add_argument(
        "--weights-override",
        default="",
        help="worker only: load WEIGHTS from this path instead of the manifest's model "
        "(the tokenizer stays the real one). Fakes ONLY the GPU-scale weights for the "
        "tiny-real CPU end-to-end run.",
    )
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2 units x 32 rows through the IDENTICAL dispatch path; uploads to "
        "the smoke_probe prefix (process width NOT narrowed)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit (Axis-1 import-resolution leg)",
    )
    args = ap.parse_args()

    if args.import_check:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from huggingface_hub import HfApi  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            retry_transient,
            verify_repo_paths_uploaded,
        )
        from scripts.issue1689_user_slot_render import (  # noqa: F401
            main as render_main,
            stage_source_files,
        )

        print("[capture] import-check OK", flush=True)
        return 0

    if args.smoke and not args.max_rows:
        args.max_rows = 32

    manifest_needed = args.mode == "worker"
    if manifest_needed:
        return run_worker(args, load_manifest(args.rendered_dir))
    try:
        return run_dispatch(args)
    except Exception as exc:  # noqa: BLE001 — fail LOUD but leave a sentinel
        write_sentinel(
            args.sentinel,
            {
                "issue": 1689,
                "round": ROUND_LABEL,
                "phase": "capture",
                "status": "failed",
                "exit_code": 1,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "metadata": base_metadata(),
                "finished_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
