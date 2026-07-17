"""#1335 round-2 companion refits (critique findings 7 + 8) — free analysis on
persisted stores, 0 GPU-h.

Two closed-form ridge refit batteries at the frozen headline read (layer 19,
context arm, x_spanmean -> y), reusing the issue1335_fit loaders + the fit825
core byte-for-byte (lambda_selection="inner-group-cv", 5 outer group folds,
fit seed 0):

  A. FOLD-GRANULARITY (finding 7): the r3->r4 "framing" rung bundles a fold
     change (row-level groups -> 300 scenario groups). Refit r4_fictionframe
     under ROW-LEVEL folds (group_ids := row_ids, r3's granularity) alongside
     a scenario-fold reproduction anchor, full n + the 5-draw matched-n
     battery (seeds 931+k; all-singleton groups -> seeded uniform row draw,
     the documented matched_subsample degenerate path).

  B. LANGUAGE COMPOSITION (finding 8): Q&A rungs carry 2-10% CJK-intruded
     completions vs ~0% on story rungs. Flag rows whose completion contains
     any CJK/Hangul/kana character while the prompt contains none; refit
     r0_qa_full + r1_qa_oneline with flagged rows EXCLUDED (full n + the
     r1 matched-n battery from the filtered population), alongside unmasked
     reproduction anchors. Story endpoint intrusion is recounted from the
     rollout JSONLs (expected 0 -> no refit needed).

Ops arithmetic: ~22 single-layer fits (5 outer folds x 13-lambda inner-group
CV, d=3584, n<=4.9k) + 2x5 + 2x5 matched draws at n<=1,739; ~15 GB store
staging (per-store stage -> fit -> release). Projected <=1 h on 32-core VM
CPU. Reproduction anchors assert +-1e-6 against the committed cell JSONs.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_fit_cells as fit825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1335_fit as f1335  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_refit_companions.py"
HEADLINE_LAYER = 19
N_MATCHED_DRAWS = f1335.N_MATCHED_DRAWS
MATCHED_SEED_BASE = f1335.MATCHED_SEED_BASE
N_MIN = {"base": 1397, "instruct": 1739}  # matched_n_config + per-cell files

# CJK-intrusion detector: completion carries >=1 Han/kana/Hangul letter while
# the prompt carries none (the "non-CJK context" scan the round-1 critique ran).
_CJK_RANGES = (
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # CJK compat ideographs
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana
    (0xAC00, 0xD7AF),  # hangul syllables
)


def _has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def cjk_flagged_row_ids(jsonl_path: Path) -> tuple[set[str], int, int]:
    """row_ids whose COMPLETION contains any CJK/kana/Hangul letter (the language-
    covariance exclusion mask — removes the whole 'reply language predictable from
    context language' channel, whatever the prompt side carries). Returns
    (flagged_row_ids, n_pure_intrusion [prompt CJK-free subset], total)."""
    flagged: set[str] = set()
    pure = 0
    total = 0
    with open(jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            comp = unicodedata.normalize("NFC", row.get("completion") or "")
            if _has_cjk(comp):
                flagged.add(row["row_id"])
                prompt = unicodedata.normalize("NFC", row.get("prompt") or "")
                if not _has_cjk(prompt):
                    pure += 1
    return flagged, pure, total


def fit_l19(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Single-layer (already-sliced) held-out fit + L19 group bootstrap CI."""
    sweep = fit825.heldout_r2_sweep(
        X,
        Y,
        groups,
        n_folds=5,
        seed=seed,
        null_draws=0,
        lambda_selection="inner-group-cv",
        frozen_layers=(0,),
    )
    out = {
        "r2": float(sweep["r2_obs"][0]),
        "n": int(X.shape[0]),
        "n_groups": len(np.unique(groups)),
    }
    if n_boot and 0 in sweep.get("preds_frozen", {}):
        fitted = sweep["fitted_mask"]
        pred = sweep["preds_frozen"][0][fitted]
        true = Y[fitted, 0, :].astype(np.float64)
        gb = fit931.group_bootstrap_r2(
            pred, true, np.asarray(groups)[fitted], n_boot=n_boot, seed=seed
        )
        out["group_bootstrap_l19"] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in gb.items()
            if not isinstance(v, (list, np.ndarray))
        }
    return out


def matched_battery(X, Y, groups, n_min: int, *, seed: int) -> dict:
    vals = []
    for k in range(N_MATCHED_DRAWS):
        idx = f1335.matched_subsample(np.asarray(groups), n_min, seed=MATCHED_SEED_BASE + k)
        r = fit_l19(X[idx], Y[idx], np.asarray(groups)[idx], n_boot=0, seed=seed)
        vals.append(r["r2"])
    return {
        "n_min": int(n_min),
        "n_draws": N_MATCHED_DRAWS,
        "r2_headline_mean": float(np.mean(vals)),
        "r2_headline_per_draw": [float(v) for v in vals],
    }


HF_REV = "08421fc22bbe42968670c4ffbfcc561dd9cf4aa5"

# ------------------------------------------------------------------ collapse-
# conditional refit (seed44-base-rungs companion, follow-ups v2 rank 1):
# refit the committed seed-42 base r7_endpoint store excluding the scene
# trajectories the seed-43 round's collapse audit implicated. Pinned counts
# are FACTS about the frozen Hub artifact at HF_REV (re-verified 2026-07-17):
# 781 slot-4 lines exactly "I agree." (= 781 (scenario, persona) trajectories,
# per-persona Wren 246 / HELIOS 234 / Dana 215 / Vex 86) and 996 lines under
# the 4-token keep floor (830 slot-4 + 139 slot-5 + 27 slot-2/3) across 839
# trajectories.
COLLAPSE_LINE = "I agree."
EXPECTED_AGREE_TRAJ = 781
EXPECTED_UNDER_FLOOR_LINES = 996


def collapse_exclusion_sets(jsonl_path: Path) -> dict:
    """Exclusion sets from an endpoint rollout JSONL: (scenario_id, persona)
    trajectories whose slot-4 line is exactly COLLAPSE_LINE, and trajectories
    carrying ANY under-floor line (n_completion_tokens < DIALOGUE_MIN_TOKENS).
    Returns {'agree_traj', 'under_floor_traj', 'n_lines',
    'n_under_floor_lines'}; newline-split read (never splitlines)."""
    agree: set[tuple[str, str]] = set()
    under: set[tuple[str, str]] = set()
    n_under_lines = 0
    total = 0
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            key = (str(row["scenario_id"]), str(row["persona"]))
            if int(row["n_completion_tokens"]) < r1335.DIALOGUE_MIN_TOKENS:
                n_under_lines += 1
                under.add(key)
            if int(row["slot"]) == 4 and (row.get("completion") or "").strip() == COLLAPSE_LINE:
                agree.add(key)
    assert total > 0, f"empty rollout JSONL {jsonl_path}"
    return {
        "agree_traj": agree,
        "under_floor_traj": under,
        "n_lines": total,
        "n_under_floor_lines": n_under_lines,
    }


def collapse_keep_mask(store: dict, persona: str, excluded: set[tuple[str, str]]) -> np.ndarray:
    """Row keep-mask for one persona unit of a fiction store: keep rows whose
    (scenario == group_id, persona) trajectory is NOT excluded. Fail-loud on
    an all-dropped unit (an empty fit is never silently produced)."""
    m_p = np.asarray(store["char_ids"]) == persona
    keep_traj = np.asarray([(str(g), persona) not in excluded for g in store["group_ids"]])
    mask = m_p & keep_traj
    assert mask.any(), f"collapse filter dropped EVERY {persona} row — refusing an empty fit"
    return mask


def stage_seed42_store(args, slug: str, mk: str) -> bool:
    """Ensure the seed-42 (rung, model) store — sidecars AND .pt shards — is
    local; when absent, stage the whole Hub prefix at HF_REV via the canonical
    hub.stage_hub_prefix (#1402), then map the verbatim prefix mirror into the
    consumer layout (#928: mirror != consumer layout; same-filesystem
    os.replace). Returns True when anything was staged."""
    import os

    store_dir = f1335.store_root(args) / slug / mk
    sidecars = sorted(store_dir.glob(f"{mk}_shard*.json"))
    pts = sorted(store_dir.glob(f"{mk}_shard*.pt"))
    if sidecars and len(pts) == len(sidecars):
        return False
    from explore_persona_space.orchestrate import hub

    prefix = f"{r1335.HF_PREFIX}/analysis_tensors/store_{slug}_{mk}"
    stage_root = args.data_dir / "hf_dl"
    hub.stage_hub_prefix(
        r1335.HF_DATA_REPO, prefix, stage_root, repo_type="dataset", revision=HF_REV
    )
    staged = stage_root / prefix
    store_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(staged.iterdir()):
        os.replace(p, store_dir / p.name)
        n += 1
    assert sorted(store_dir.glob(f"{mk}_shard*.pt")), f"no .pt shards staged under {prefix}"
    print(f"[refit] staged {n} store files for {slug}/{mk} via stage_hub_prefix @ {HF_REV[:12]}")
    return True


def run_collapse_conditional(args) -> None:
    """Collapse-conditional refit of the committed seed-42 base r7_endpoint
    store: per-persona L19 ctx held-out R² (inner-group-CV λ, scene-grouped
    folds, group bootstrap — the fit825 core incl. _eigh_robust, byte-for-byte
    the committed cell recipe) under two exclusions — (a) trajectories whose
    slot-4 line collapsed to 'I agree.' and (b) the stricter any-under-floor-
    line trajectories — beside a full-unit reproduction anchor asserted ±1e-6
    against the committed seed-42 cells, read against the seed-43 cells."""
    slug, mk = "r7_endpoint", "base"
    assert r1335.HF_PREFIX == "issue1335_ablation_ladder", (
        f"collapse-conditional refits the SEED-42 store; run without "
        f"EPM_I1335_HF_PREFIX (got prefix {r1335.HF_PREFIX!r})"
    )
    jsonl = args.rollouts_jsonl
    pins_apply = jsonl is None
    if jsonl is None:
        from explore_persona_space.orchestrate import hub

        dest = args.data_dir / "hf_dl" / "endpoint_base_gen.jsonl"
        if not dest.exists():
            hub.stage_hub_file(
                r1335.HF_DATA_REPO,
                f"{r1335.HF_PREFIX}/raw_completions/endpoint/base_gen.jsonl",
                dest,
                repo_type="dataset",
                revision=HF_REV,
            )
        jsonl = dest
    ex = collapse_exclusion_sets(Path(jsonl))
    if pins_apply:
        assert len(ex["agree_traj"]) == EXPECTED_AGREE_TRAJ, (
            len(ex["agree_traj"]),
            EXPECTED_AGREE_TRAJ,
        )
        assert ex["n_under_floor_lines"] == EXPECTED_UNDER_FLOOR_LINES, (
            ex["n_under_floor_lines"],
            EXPECTED_UNDER_FLOOR_LINES,
        )
    else:
        print("[refit] NOTE: explicit --rollouts-jsonl — frozen-artifact count pins skipped")
    print(
        f"[refit] exclusions: {len(ex['agree_traj'])} 'I agree.' trajectories, "
        f"{ex['n_under_floor_lines']} under-floor lines across "
        f"{len(ex['under_floor_traj'])} trajectories ({ex['n_lines']} lines)"
    )

    staged = stage_seed42_store(args, slug, mk)
    store = f1335.load_rung_store(args, slug, mk)
    X_all, Y_all = slice_l19(store)
    groups = np.asarray(store["group_ids"])
    per_persona: dict = {}
    variants = (
        ("excl_slot4_agree", ex["agree_traj"]),
        ("excl_any_under_floor", ex["under_floor_traj"]),
    )
    import issue1310_common as c1310  # persona label order (fit-side convention)

    personas = list(c1310.PERSONA_LABELS)
    if args.personas:
        personas = [p.strip() for p in args.personas.split(",") if p.strip()]
        bad = [p for p in personas if p not in c1310.PERSONA_LABELS]
        assert not bad, f"unknown personas {bad} (choices: {list(c1310.PERSONA_LABELS)})"
        print(f"[refit] NOTE: persona subset {personas} (smoke; production runs all)")
    for persona in personas:
        m_p = np.asarray(store["char_ids"]) == persona
        assert m_p.any(), f"no {persona} rows in the {slug}/{mk} store"
        anchor = fit_l19(X_all[m_p], Y_all[m_p], groups[m_p], n_boot=args.n_boot, seed=args.seed)
        committed = committed_l19(args.out_dir, f"{slug}__{mk}__{persona}__ctx")
        assert abs(anchor["r2"] - committed) < 1e-6, (persona, anchor["r2"], committed)
        seed43 = committed_l19(args.seed43_dir, f"{slug}__{mk}__{persona}__ctx")
        entry: dict = {
            "committed_seed42_l19": committed,
            "seed43_l19": seed43,
            "anchor_repro_full_unit": anchor,
        }
        for label, excl in variants:
            keep = collapse_keep_mask(store, persona, excl)
            fit = fit_l19(
                X_all[keep], Y_all[keep], groups[keep], n_boot=args.n_boot, seed=args.seed
            )
            fit["n_scenarios_kept"] = len(np.unique(groups[keep]))
            fit["n_rows_dropped"] = int(m_p.sum() - keep.sum())
            entry[label] = fit
        per_persona[persona] = entry
        print(
            f"[refit] r7-collapse {persona}: anchor {anchor['r2']:.4f} "
            f"(seed42 {committed:.4f} / seed43 {seed43:.4f}) "
            f"agree-excl {entry['excl_slot4_agree']['r2']:.4f} "
            f"underfloor-excl {entry['excl_any_under_floor']['r2']:.4f}"
        )
    if staged:
        f1335.release_store_local(args, slug, mk)

    out = {
        "metadata": common.metadata(SCRIPT, args.seed, 0, extra={"issue": 1335}),
        "mode": "collapse-conditional (seed44-base-rungs companion, seed-42 base r7 store)",
        "headline_layer": HEADLINE_LAYER,
        "lambda_selection": "inner-group-cv",
        "hf_rev": HF_REV,
        "rollouts_jsonl": str(jsonl),
        "personas": personas,
        "exclusion_counts": {
            "n_lines": ex["n_lines"],
            "n_under_floor_lines": ex["n_under_floor_lines"],
            "n_agree_trajectories": len(ex["agree_traj"]),
            "n_under_floor_trajectories": len(ex["under_floor_traj"]),
            "count_pins_asserted": bool(pins_apply),
        },
        "per_persona": per_persona,
    }
    args.collapse_out.parent.mkdir(parents=True, exist_ok=True)
    args.collapse_out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[refit] wrote {args.collapse_out}")


def seed_sidecars(args, slug: str, mk: str) -> None:
    """Download a (rung, model) store's shard SIDECAR JSONs from the Hub when
    absent locally (ensure_store_local asserts on local sidecars; this worktree
    only kept r0/r7). Pure hub-rel -> local-rel mapping at the pinned rev."""
    from huggingface_hub import HfApi, hf_hub_download

    store_dir = f1335.store_root(args) / slug / mk
    if list(store_dir.glob(f"{mk}_shard*.json")):
        return
    store_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{r1335.HF_PREFIX}/analysis_tensors/store_{slug}_{mk}"
    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    api = HfApi()
    names = [
        p.split("/")[-1]
        for p in list_hf_files_under_path(
            api, r1335.HF_DATA_REPO, prefix, repo_type="dataset", revision=HF_REV
        )
        if p.endswith(".json") and "_shard" in p
    ]
    assert names, f"no shard sidecars on Hub under {prefix}"
    for name in names:
        got = hf_hub_download(
            r1335.HF_DATA_REPO,
            f"{prefix}/{name}",
            repo_type="dataset",
            revision=HF_REV,
            local_dir=str(store_dir / ".hfside"),
        )
        Path(got).rename(store_dir / name)
    print(f"[refit] seeded {len(names)} sidecars for {slug}/{mk}")


def slice_l19(store: dict) -> tuple[np.ndarray, np.ndarray]:
    X = store["arrays"]["x_spanmean"][:, [HEADLINE_LAYER], :]
    Y = store["arrays"]["y"][:, [HEADLINE_LAYER], :]
    return X, Y


def committed_l19(out_dir: Path, cell_id: str) -> float:
    d = json.loads((out_dir / f"cells_{cell_id}.json").read_text())
    return float(d["r2_per_layer_obs"][HEADLINE_LAYER])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1335"))
    ap.add_argument("--store-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1335"))
    ap.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("/tmp/i1335_gen/issue1335_ablation_ladder/raw_completions"),
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-from-hub", action="store_true", default=True)
    ap.add_argument(
        "--collapse-conditional",
        action="store_true",
        help="seed44-base-rungs companion: collapse-conditional refit of the "
        "seed-42 base r7_endpoint store (skips the r2 companion batteries)",
    )
    ap.add_argument(
        "--rollouts-jsonl",
        type=Path,
        default=None,
        help="collapse mode: explicit endpoint rollout JSONL (default: stage the "
        "frozen seed-42 base_gen.jsonl from the Hub @ HF_REV; an explicit path "
        "skips the frozen-artifact count pins — smoke/fixture use)",
    )
    ap.add_argument(
        "--seed43-dir",
        type=Path,
        default=Path("eval_results/issue_1335/seed43-gap-rungs"),
        help="collapse mode: committed seed-43 cells dir (per-persona L19 reads)",
    )
    ap.add_argument(
        "--personas",
        type=str,
        default="",
        help="collapse mode: comma-separated persona subset (smoke slicing; "
        "empty/default = ALL personas — production never narrows)",
    )
    ap.add_argument(
        "--collapse-out",
        type=Path,
        default=Path(
            "eval_results/issue_1335/seed44-base-rungs/refits_r7_collapse_conditional.json"
        ),
        help="collapse mode: output JSON path (the seed44 round's artifact dir)",
    )
    args = ap.parse_args()
    args.folds = 5

    if args.collapse_conditional:
        run_collapse_conditional(args)
        return

    results: dict = {
        "metadata": common.metadata(SCRIPT, args.seed, 0, extra={"issue": 1335}),
        "headline_layer": HEADLINE_LAYER,
        "lambda_selection": "inner-group-cv",
        "ops_note": (
            "free-analysis refits on persisted bf16 stores (HF rev 08421fc22bbe); "
            "~22 single-layer L19 ridge fits + 4x5 matched draws, VM CPU, 0 GPU-h"
        ),
        "cjk_scan": {},
        "fold_granularity_r4": {},
        "cjk_filtered_qa": {},
    }

    # ------------------------------------------------------------------ CJK scan
    from huggingface_hub import hf_hub_download

    rev = "08421fc22bbe42968670c4ffbfcc561dd9cf4aa5"
    gen_files = {
        ("r0_qa_full", "base"): "qa_full/base_gen.jsonl",
        ("r0_qa_full", "instruct"): "qa_full/instruct_gen.jsonl",
        ("r1_qa_oneline", "base"): "qa_oneline/base_gen.jsonl",
        ("r1_qa_oneline", "instruct"): "qa_oneline/instruct_gen.jsonl",
        ("r7_endpoint", "base"): "endpoint/base_gen.jsonl",
        ("r7_endpoint", "instruct"): "endpoint/instruct_gen.jsonl",
    }
    masks: dict = {}
    for (slug, mk), rel in gen_files.items():
        local = hf_hub_download(
            r1335.HF_DATA_REPO,
            f"{r1335.HF_PREFIX}/raw_completions/{rel}",
            repo_type="dataset",
            revision=rev,
            local_dir="/tmp/i1335_gen",
        )
        flagged, pure, total = cjk_flagged_row_ids(Path(local))
        masks[(slug, mk)] = flagged
        results["cjk_scan"][f"{slug}__{mk}"] = {
            "flagged_completion_cjk": len(flagged),
            "flagged_pure_intrusion_prompt_cjk_free": pure,
            "total_gen_rows": total,
            "pct": round(100.0 * len(flagged) / max(total, 1), 2),
        }
        print(f"[refit] CJK scan {slug}/{mk}: {len(flagged)}/{total} flagged ({pure} pure)")

    # ------------------------------------------------- A. r4 fold granularity
    for mk in ("base", "instruct"):
        slug = "r4_fictionframe"
        seed_sidecars(args, slug, mk)
        f1335.ensure_store_local(args, slug, mk)
        store = f1335.load_rung_store(args, slug, mk)
        X, Y = slice_l19(store)
        anchor = fit_l19(X, Y, store["group_ids"], n_boot=args.n_boot, seed=args.seed)
        committed = committed_l19(args.out_dir, f"{slug}__{mk}__ctx")
        assert abs(anchor["r2"] - committed) < 1e-6, (anchor["r2"], committed)
        rowfold = fit_l19(X, Y, store["row_ids"], n_boot=args.n_boot, seed=args.seed)
        rowfold_matched = matched_battery(X, Y, store["row_ids"], N_MIN[mk], seed=args.seed)
        scen_matched_committed = json.loads(
            (args.out_dir / f"matched_{slug}__{mk}__ctx.json").read_text()
        )["r2_headline_mean"]
        results["fold_granularity_r4"][mk] = {
            "scenario_fold_repro_full_n": anchor,
            "committed_full_n": committed,
            "row_fold_full_n": rowfold,
            "row_fold_matched": rowfold_matched,
            "scenario_fold_matched_committed": float(scen_matched_committed),
        }
        print(
            f"[refit] r4/{mk}: scenario {anchor['r2']:.4f} (committed {committed:.4f}) "
            f"rowfold {rowfold['r2']:.4f} rowfold-matched {rowfold_matched['r2_headline_mean']:.4f}"
        )
        f1335.release_store_local(args, slug, mk)
        del store, X, Y

    # ------------------------------------------------- B. CJK-filtered Q&A
    for slug in ("r0_qa_full", "r1_qa_oneline"):
        for mk in ("base", "instruct"):
            seed_sidecars(args, slug, mk)
            f1335.ensure_store_local(args, slug, mk)
            store = f1335.load_rung_store(args, slug, mk)
            X, Y = slice_l19(store)
            groups = store["group_ids"]
            anchor = fit_l19(X, Y, groups, n_boot=args.n_boot, seed=args.seed)
            committed = committed_l19(args.out_dir, f"{slug}__{mk}__ctx")
            assert abs(anchor["r2"] - committed) < 1e-6, (anchor["r2"], committed)
            flagged = masks[(slug, mk)]
            keep = np.asarray([rid not in flagged for rid in store["row_ids"]])
            n_excl = int((~keep).sum())
            filt = fit_l19(
                X[keep], Y[keep], np.asarray(groups)[keep], n_boot=args.n_boot, seed=args.seed
            )
            entry = {
                "unmasked_repro_full_n": anchor,
                "committed_full_n": committed,
                "rows_excluded": n_excl,
                "cjk_filtered_full_n": filt,
            }
            if slug == "r1_qa_oneline":
                entry["cjk_filtered_matched"] = matched_battery(
                    X[keep], Y[keep], np.asarray(groups)[keep], N_MIN[mk], seed=args.seed
                )
                entry["unmasked_matched_committed"] = json.loads(
                    (args.out_dir / f"matched_{slug}__{mk}__ctx.json").read_text()
                )["r2_headline_mean"]
            results["cjk_filtered_qa"][f"{slug}__{mk}"] = entry
            print(
                f"[refit] {slug}/{mk}: unmasked {anchor['r2']:.4f} (committed {committed:.4f}) "
                f"filtered {filt['r2']:.4f} (excl {n_excl})"
            )
            f1335.release_store_local(args, slug, mk)
            del store, X, Y

    out_path = args.out_dir / "refits_r2_companions.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"[refit] wrote {out_path}")


if __name__ == "__main__":
    main()
