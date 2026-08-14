"""Leave-one-SPEAKER-out pooled context->answer map (#2054 writeup, wr6 companion).

wr6 plots each story speaker's pooled-map recovery against that speaker's own map,
but all 56 lattice cells -- the assistant's 16 included -- sit INSIDE the pooled
map's training set, so that curve is an in-sample read rather than a
generalization test. This script refits the pooled map with one SPEAKER held out
entirely (every cell of that speaker, across conditions/framings/models) and
scores it on exactly those held-out cells, answering the question wr6 cannot:
does the pooled map transfer to a speaker it never saw?

Five held-out variants (4 story characters + the assistant) plus a matched
FULL-pool reference scored on the IDENTICAL rows, so the LOCO-vs-full gap is a
training-set-composition effect and not a row-selection artifact.

Estimator: reuse of ``issue2054_pool_specialize.PooledMomentRidge`` (GCV ridge,
#1887 dof cap 0.9) over streamed second moments -- the pooled train matrix is
never materialized. LOCO exploits the moments' ADDITIVITY: per-(speaker, fold)
moments are accumulated ONCE per arm and each variant SUBTRACTS its held-out
speaker, so 5 variants cost 5 extra eigh solves rather than 5 extra data passes.

Per CLAUDE.md every held-out R2 ships with its two mandatory companion reads:
the identity-plus-learned-bias baseline (input and output share d=3584) and kNN
retrieval among the held-out pool (euclidean + cosine, chance = k/n_pool).

Fold structure is the shared #2054 conversation-grouped fold map, so a held-out
speaker's test rows are the same rows wr6 scores -- LOCO changes only WHO is in
the training pool, never the evaluation rows.

``--group-by framing`` (writeup-v2 round, 2026-08-14) swaps the hold-out unit
from SPEAKER to FRAMING (chat / bare_text / bare_label / attrib_quoted; the
transposed cell_c cells count as chat): each variant refits the pooled map with
every cell of one framing held out and scores exactly those cells, answering
the Result-1 analogue -- does the pooled map transfer to a framing it never
saw? Same moments machinery, estimator, folds, and companion reads; only the
grouping key changes.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval
from explore_persona_space.experiments.issue_779.fit_h import reconstruction_metrics
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ctx2ctx_fit import (
    ARM_VEC_KEY,
    ARMS,
    D_AMBIENT,
    Cell,
    discover_cells,
    load_fold_map,
)
from scripts.issue2054_pool_specialize import (
    PooledMomentRidge,
    join_cell,
    load_cell_with_answer,
)

SCRIPT_VERSION = "issue2054_loco_pooled_v1"

# Speaker grouping. A character's `_op` / `_op_base` cells are the SAME speaker as
# its main cells -- holding out `char_vex` while leaving `char_vex_op` in the pool
# would leak the very identity the variant claims to withhold.
ASSISTANT_ID = "conversation_paired_stories_assistant"
CHARACTERS = ("helios", "wren", "vex", "dana")
SPEAKER_LABEL = {
    "helios": "HELIOS",
    "wren": "Wren",
    "vex": "Vex",
    "dana": "Dana",
    "assistant": "Assistant",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def speaker_of(cell: Cell) -> str:
    """Map a cell's identity onto its SPEAKER (the LOCO hold-out unit).

    `char_vex`, `char_vex_op` and `char_vex_op_base` all resolve to `vex`; the
    assistant's cells resolve to `assistant`. Raises on an unrecognized identity
    rather than silently dropping it into a wrong hold-out group.
    """
    ident = cell.identity
    if ident == ASSISTANT_ID:
        return "assistant"
    if ident.startswith("char_"):
        for ch in CHARACTERS:
            if ident == f"char_{ch}" or ident.startswith(f"char_{ch}_"):
                return ch
    raise ValueError(f"cannot resolve speaker for identity {ident!r} (cell {cell.key})")


KNOWN_FRAMINGS = ("chat", "bare_text", "bare_label", "attrib_quoted")


def framing_of(cell: Cell) -> str:
    """Map a cell onto its FRAMING (the ``--group-by framing`` hold-out unit).

    The transposed cell_c cells are chat-template renders, so they resolve to
    ``chat``. Raises on an unrecognized framing rather than silently creating a
    wrong hold-out group.
    """
    if cell.framing not in KNOWN_FRAMINGS:
        raise ValueError(f"unrecognized framing {cell.framing!r} (cell {cell.key})")
    return cell.framing


GROUP_FNS = {"speaker": speaker_of, "framing": framing_of}


def _zero_moment(d: int, dev: torch.device) -> dict:
    return {
        "n": 0,
        "sum_x": torch.zeros(d, dtype=torch.float64, device=dev),
        "sum_y": torch.zeros(d, dtype=torch.float64, device=dev),
        "yss": 0.0,
        "c_xx": torch.zeros(d, d, dtype=torch.float64, device=dev),
        "c_xy": torch.zeros(d, d, dtype=torch.float64, device=dev),
    }


def accumulate_by_speaker(
    cells: list[Cell], fold_of: dict, k: int, arm: str, device: str, group_fn=speaker_of
) -> dict[str, list[dict]]:
    """One streaming pass over cells for ONE arm: per-(group, fold) moments.

    Keyed by the hold-out group (speaker by default, framing under
    ``--group-by framing``) so any LOCO variant is a SUBTRACTION of banked sums
    rather than a fresh pass. Processing a single arm per call halves peak RSS
    versus accumulating both arms together.
    """
    dev = torch.device(device)
    speakers = sorted({group_fn(c) for c in cells})
    mom = {s: [_zero_moment(D_AMBIENT, dev) for _ in range(k)] for s in speakers}
    vec = ARM_VEC_KEY[arm]
    for ci, cell in enumerate(cells):
        t0 = time.time()
        spk = group_fn(cell)
        act = load_cell_with_answer(cell)
        j = join_cell(act, fold_of, k, arm)
        for f in range(k):
            idx = j["rows"][j["fold_rows"][f]]
            x = torch.as_tensor(act[vec][idx].astype(np.float64), device=dev)
            y = torch.as_tensor(act["v_A"][idx].astype(np.float64), device=dev)
            m = mom[spk][f]
            m["n"] += int(x.shape[0])
            m["sum_x"] += x.sum(0)
            m["sum_y"] += y.sum(0)
            m["yss"] += float((y * y).sum())
            m["c_xx"] += x.T @ x
            m["c_xy"] += x.T @ y
            del x, y
        del act
        _log(
            f"[loco] moments arm={arm} cell {ci + 1}/{len(cells)} spk={spk} "
            f"{cell.key} n_join={j['n_join']} elapsed={time.time() - t0:.1f}s"
        )
    return mom


def combine(
    mom: dict[str, list[dict]], k: int, *, drop_speaker: str | None, drop_fold: int
) -> dict:
    """Sum banked moments over speakers != drop_speaker and folds != drop_fold.

    ``drop_speaker=None`` yields the FULL-pool train moments (the matched
    reference); a speaker name yields that variant's LOCO train moments.
    """
    keys = ("sum_x", "sum_y", "c_xx", "c_xy")
    out = {"n": 0, "yss": 0.0}
    acc: dict[str, torch.Tensor | None] = dict.fromkeys(keys)
    for spk, per_fold in mom.items():
        if spk == drop_speaker:
            continue
        for f in range(k):
            if f == drop_fold:
                continue
            m = per_fold[f]
            out["n"] += m["n"]
            out["yss"] += m["yss"]
            for key in keys:
                acc[key] = m[key].clone() if acc[key] is None else acc[key] + m[key]
    if acc["c_xx"] is None:
        raise RuntimeError(
            f"empty train moments (drop_speaker={drop_speaker}, drop_fold={drop_fold})"
        )
    out.update({key: acc[key] for key in keys})
    return out


def evaluate_cell(
    cell: Cell,
    act: dict,
    j: dict,
    arm: str,
    k: int,
    loco_models: dict[int, PooledMomentRidge],
    full_models: dict[int, PooledMomentRidge],
) -> dict:
    """Score the LOCO and matched FULL-pool maps on one held-out cell's test rows.

    Both maps are scored on the IDENTICAL rows fold by fold, so the reported gap
    isolates training-pool composition. Companion reads (identity+bias, kNN) ride
    the LOCO prediction, which is the read the generalization claim rests on.
    """
    vec = ARM_VEC_KEY[arm]
    per_fold: list[dict] = []
    for f in range(k):
        te = j["rows"][j["fold_rows"][f]]
        tr_idx = np.concatenate([j["fold_rows"][g] for g in range(k) if g != f])
        tr = j["rows"][tr_idx]
        x_te = act[vec][te].astype(np.float64)
        y_te = act["v_A"][te].astype(np.float64)
        x_tr = act[vec][tr].astype(np.float64)
        y_tr = act["v_A"][tr].astype(np.float64)

        p_loco = loco_models[f].predict_np(x_te)
        p_full = full_models[f].predict_np(x_te)
        m_loco = reconstruction_metrics(p_loco, y_te)
        m_full = reconstruction_metrics(p_full, y_te)

        # Mandatory companions (CLAUDE.md): identity+learned-bias floor and the
        # retrieval read, both on the LOCO prediction. The bias is learned on the
        # cell's OWN train rows -- the identity family's standard form.
        p_ident = identity_bias_predict(x_tr, y_tr, x_te)
        m_ident = reconstruction_metrics(p_ident, y_te)
        knn = {
            metric: knn_retrieval(p_loco, y_te, metric=metric) for metric in ("euclidean", "cosine")
        }
        per_fold.append(
            {
                "fold": f,
                "n_test": int(len(te)),
                "loco": m_loco,
                "full_pool": m_full,
                "identity_bias": m_ident,
                "knn_loco": knn,
                "loco_lambda": loco_models[f].best_lambda,
                "loco_n_train": loco_models[f].n_train,
                "full_lambda": full_models[f].best_lambda,
                "full_n_train": full_models[f].n_train,
            }
        )

    def _mean(path: tuple[str, ...]) -> float:
        vals = []
        for r in per_fold:
            node: object = r
            for p in path:
                node = node[p]  # type: ignore[index]
            vals.append(float(node))  # type: ignore[arg-type]
        return float(np.mean(vals))

    return {
        "cell": cell.key,
        "speaker": speaker_of(cell),
        "identity": cell.identity,
        "condition": cell.condition,
        "framing": cell.framing,
        "model": cell.model,
        "arm": arm,
        "n_join": j["n_join"],
        "per_fold": per_fold,
        "mean": {
            "loco_r2": _mean(("loco", "r2")),
            "full_pool_r2": _mean(("full_pool", "r2")),
            "identity_bias_r2": _mean(("identity_bias", "r2")),
            "loco_mean_cosine": _mean(("loco", "mean_cosine")),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--activations-dir", type=Path, required=True)
    ap.add_argument(
        "--out-root", type=Path, default=_REPO / "eval_results/issue_2054/specialization_ladder"
    )
    ap.add_argument("--out-name", default="loco_pooled.json")
    ap.add_argument("--fold-map-ref", default="origin/issue-2054")
    ap.add_argument("--fold-map-file", default=None)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument(
        "--group-by",
        default="speaker",
        choices=sorted(GROUP_FNS),
        help="LOCO hold-out unit: speaker (default, the wr6 companion) or framing",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit-cells", type=int, default=None, help="pilot: first N cells")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[loco] import-check OK")
        return 0

    t_start = time.time()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    fold_of = fold_map["fold_of"]
    _log(
        f"[loco] fold map {fold_map['_source']} k={k} seed={fold_map['seed']} "
        f"n_conv={len(fold_of):,} sha={fold_map['_sha256'][:12]}"
    )

    group_fn = GROUP_FNS[args.group_by]
    cells = discover_cells(args.activations_dir)
    if args.limit_cells:
        cells = cells[: args.limit_cells]
    speakers = sorted({group_fn(c) for c in cells})
    by_speaker: dict[str, list[Cell]] = {s: [] for s in speakers}
    for c in cells:
        by_speaker[group_fn(c)].append(c)
    _log(f"[loco] {len(cells)} cells, {len(speakers)} {args.group_by} groups, arms={args.arms}")
    for s in speakers:
        _log(f"[loco]   {args.group_by} {s}: {len(by_speaker[s])} cells")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    # Per-unit checkpoint (code-style: >50 units => persist per unit + resume).
    ckpt = out_root / f"{Path(args.out_name).stem}.units.jsonl"
    done: set[str] = set()
    if ckpt.exists():
        for line in ckpt.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add(f"{r['cell']}__{r['arm']}")
        _log(f"[loco] resume: {len(done)} units already banked in {ckpt}")

    n_units = len(cells) * len(args.arms)
    unit_i = len(done)
    for arm in args.arms:
        pending = [c for c in cells if f"{c.key}__{arm}" not in done]
        if not pending:
            _log(f"[loco] arm={arm}: all units banked, skipping")
            continue

        t0 = time.time()
        mom = accumulate_by_speaker(cells, fold_of, k, arm, args.device, group_fn=group_fn)
        _log(f"[loco] arm={arm} moments done elapsed={time.time() - t0:.1f}s")

        # Solve every map up front, then release the moments before the eval pass.
        full_models: dict[int, PooledMomentRidge] = {}
        for f in range(k):
            t1 = time.time()
            full_models[f] = PooledMomentRidge(**combine(mom, k, drop_speaker=None, drop_fold=f))
            _log(
                f"[loco] arm={arm} FULL fold {f}: n_train={full_models[f].n_train:,} "
                f"lam={full_models[f].best_lambda:g} elapsed={time.time() - t1:.1f}s"
            )
        loco_models: dict[str, dict[int, PooledMomentRidge]] = {}
        for s in speakers:
            loco_models[s] = {}
            for f in range(k):
                t1 = time.time()
                loco_models[s][f] = PooledMomentRidge(
                    **combine(mom, k, drop_speaker=s, drop_fold=f)
                )
                _log(
                    f"[loco] arm={arm} LOCO[{s}] fold {f}: "
                    f"n_train={loco_models[s][f].n_train:,} "
                    f"lam={loco_models[s][f].best_lambda:g} elapsed={time.time() - t1:.1f}s"
                )
        del mom

        with ckpt.open("a", encoding="utf-8") as fh:
            for cell in pending:
                t1 = time.time()
                spk = group_fn(cell)
                act = load_cell_with_answer(cell)
                j = join_cell(act, fold_of, k, arm)
                rec = evaluate_cell(cell, act, j, arm, k, loco_models[spk], full_models)
                rec["group"] = spk
                rec["group_by"] = args.group_by
                del act
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                unit_i += 1
                _log(
                    f"[loco] eval unit {unit_i}/{n_units} {cell.key} arm={arm} "
                    f"loco_r2={rec['mean']['loco_r2']:.4f} "
                    f"full_r2={rec['mean']['full_pool_r2']:.4f} "
                    f"elapsed={time.time() - t1:.1f}s"
                )
        del full_models, loco_models

    records = [json.loads(x) for x in ckpt.read_text(encoding="utf-8").splitlines() if x.strip()]
    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fold_map_sha256": fold_map["_sha256"],
            "fold_map_source": fold_map["_source"],
            "k": k,
            "d_ambient": D_AMBIENT,
            "arms": list(args.arms),
            "n_cells": len(cells),
            "group_by": args.group_by,
            "groups": {s: [c.key for c in by_speaker[s]] for s in speakers},
            "wall_seconds": round(time.time() - t_start, 1),
        },
        "per_unit": records,
        "aggregate": aggregate(records),
    }
    out_path = out_root / args.out_name
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)
    _log(f"[loco] wrote {out_path} ({len(records)} units, {time.time() - t_start:.0f}s)")
    _log("[phase=done]")
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def aggregate(records: list[dict]) -> dict:
    """Per-(group, arm, condition) means of the LOCO / full-pool / identity reads.

    The group is the hold-out unit — the speaker by default (pre-``--group-by``
    records carry no ``group`` field and fall back to ``speaker``).
    """
    out: dict[str, dict] = {}
    for rec in records:
        grp = rec.get("group", rec["speaker"])
        for key in (
            f"{grp}|{rec['arm']}|all",
            f"{grp}|{rec['arm']}|{rec['condition']}",
        ):
            out.setdefault(key, {"n_cells": 0, "loco_r2": [], "full_pool_r2": [], "ident_r2": []})
            g = out[key]
            g["n_cells"] += 1
            g["loco_r2"].append(rec["mean"]["loco_r2"])
            g["full_pool_r2"].append(rec["mean"]["full_pool_r2"])
            g["ident_r2"].append(rec["mean"]["identity_bias_r2"])
    agg = {}
    for key, g in sorted(out.items()):
        loco = float(np.mean(g["loco_r2"]))
        full = float(np.mean(g["full_pool_r2"]))
        agg[key] = {
            "n_cells": g["n_cells"],
            "loco_r2": loco,
            "full_pool_r2": full,
            "identity_bias_r2": float(np.mean(g["ident_r2"])),
            "loco_minus_full": loco - full,
            "loco_over_full": (loco / full) if abs(full) > 1e-9 else float("nan"),
        }
    return agg


if __name__ == "__main__":
    raise SystemExit(main())
