"""Round-2 interp-critique follow-up statistics for issue #2379 (zero-GPU, existing artifacts).

Computes, from the committed round-1 artifacts only:

 1. Bootstrap index reproduction check — regenerates the registered trigger-index
    multisets (``np.random.default_rng([BOOT_SEED, 0|1]).integers(0, n_t, (2000, n_t))``)
    and asserts sha256 equality with the committed ``bootstrap.*.idx_sha256``.
 2. Capitalization CONTINUOUS-companion contrast (dual-DV blocker): the registered
    paired trigger bootstrap re-run with y = ``mean_uppercase_fraction`` under the
    IDENTICAL predictor arrays (pinned ans_trainref_mapI / ctx_trainref) and the
    IDENTICAL resampling index multisets. Same for EM with y = ``mean_misalignment``.
 3. Leave-one-language-out capitalization pooled delta-rho + CI (binary rate),
    same idx, condition axis subset.
 4. EM stored-layer-21 exploratory read: within-condition Spearman for
    ctx_trainref / ans_trainref_mapI / ceiling_trainref at stored layer 21.
 5. CJK/foreign-script intrusion counts over the MAP-FITTING corpora (all 9:
    base + 8 inoculated; total rows and kept-only fit rows) and a re-verify of
    the EM behavior-evaluation pools.

Writes ``eval_results/issue_2379/r2_followup.json``. Pure counting/statistics —
no completion text enters the output beyond counts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue2379_analysis import (  # noqa: E402
    BOOT_SEED,
    CAPS_STEMS,
    EM_STEMS,
    PIN_CAPS_DEFAULT,
    PIN_EM_DEFAULT,
    _git_meta,
    _pair_corr,
    _r6,
    _utcnow,
    bootstrap_setting,
    condition_matrices,
)

EVAL_DIR = REPO_ROOT / "eval_results" / "issue_2379"
RAWCOMP = REPO_ROOT / "data" / "issue_2379" / "rawcomp_cache" / "issue2379_reelicit"

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

N_DRAWS = 2000


def _load(p: Path) -> dict:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _dv_vector(model: str, cond: dict, rates_em: dict, caps_models: dict, field: str) -> np.ndarray:
    """Per-trigger DV aligned to cond['trigger_labels']; fails loud on a missing label."""
    labels = cond["trigger_labels"]
    if cond["setting"] == "em":
        table = rates_em["rates"][model]
        return np.array([float(table[lab][field]) for lab in labels], dtype=np.float64)
    per = caps_models[model]["per_trigger"]
    return np.array([float(per[lab][field]) for lab in labels], dtype=np.float64)


def _idx_for(setting: str, n_t: int) -> np.ndarray:
    rng = np.random.default_rng([BOOT_SEED, 0 if setting == "em" else 1])
    return rng.integers(0, n_t, size=(N_DRAWS, n_t))


def _sha(idx: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()


def _setting_block(
    stems: list[str],
    setting: str,
    pin: int,
    dv_field: str,
    scores: dict,
    rates_em: dict,
    caps_models: dict,
    idx: np.ndarray,
) -> dict:
    """Registered paired bootstrap for one setting under DV ``dv_field``."""
    conds = scores["conditions"]
    a_ans, a_ctx, ys, per_cond = [], [], [], {}
    for m in stems:
        cond = conds[m]
        layered, _ = condition_matrices(cond)
        xa = layered["ans_trainref_mapI"][pin]
        xc = layered["ctx_trainref"][pin]
        y = _dv_vector(m, cond, rates_em, caps_models, dv_field)
        if not (np.isfinite(xa).all() and np.isfinite(xc).all() and np.isfinite(y).all()):
            raise AssertionError(f"non-finite inputs for {m} — estimability changed vs round 1")
        rho_a = _pair_corr(xa, y, spearman=True)
        rho_c = _pair_corr(xc, y, spearman=True)
        per_cond[m] = {
            "rho_ans": _r6(rho_a),
            "rho_ctx": _r6(rho_c),
            "delta": _r6(rho_a - rho_c),
        }
        a_ans.append(xa)
        a_ctx.append(xc)
        ys.append(y)
    boot = bootstrap_setting(np.stack(a_ans), np.stack(a_ctx), np.stack(ys), idx)
    deltas = [per_cond[m]["delta"] for m in stems]
    return {
        "setting": setting,
        "dv_field": dv_field,
        "pin_stored_layer": pin,
        "conditions": stems,
        "per_condition": per_cond,
        "pooled_delta_observed": _r6(float(np.mean(deltas))),
        "pooled_ci95": [_r6(boot["ci_lo"]), _r6(boot["ci_hi"])],
        "boot_frac_below0": boot["boot_frac_below0"],
        "boot_frac_above0": boot["boot_frac_above0"],
        "n_finite_pooled_draws": boot["n_finite_pooled_draws"],
        "mean_rho_ans": _r6(float(np.mean([per_cond[m]["rho_ans"] for m in stems]))),
        "mean_rho_ctx": _r6(float(np.mean([per_cond[m]["rho_ctx"] for m in stems]))),
        "idx_sha256": boot["idx_sha256"],
    }


def _cjk_scan_rows(path: Path, kind: str) -> dict:
    """Count CJK-intruded completions; counting only, no text retained."""
    d = _load(path)
    if kind == "map_corpus":
        rows = d["rows"]
        total = len(rows)
        kept = [r for r in rows if r.get("kept")]
        hit_total = sum(1 for r in rows if any(CJK_RE.search(c) for c in r["completions"]))
        hit_kept = sum(1 for r in kept if any(CJK_RE.search(c) for c in r["completions"]))
        return {
            "n_rows_total": total,
            "n_intruded_total": hit_total,
            "pct_total": _r6(100.0 * hit_total / total),
            "n_rows_kept_fit": len(kept),
            "n_intruded_kept_fit": hit_kept,
            "pct_kept_fit": _r6(100.0 * hit_kept / len(kept)) if kept else None,
        }
    # sweep files: generations -> completions lists ({"text": ..., "finish_reason": ...})
    n = 0
    hit = 0
    for g in d["generations"]:
        for c in g["completions"]:
            txt = c["text"] if isinstance(c, dict) else c
            n += 1
            hit += bool(CJK_RE.search(txt))
    return {"n_completions": n, "n_intruded": hit, "pct": _r6(100.0 * hit / n)}


def main() -> None:
    scores = _load(EVAL_DIR / "predictors" / "predictor_scores.json")
    rates_em = _load(EVAL_DIR / "rates_em.json")
    caps_models = _load(EVAL_DIR / "rates_caps.json")["models"]
    committed = _load(EVAL_DIR / "correlations.json")

    out: dict = {
        "issue": 2379,
        "slug": "issue2379_reelicit",
        "generated_utc": _utcnow(),
        "git": _git_meta(),
        "note": (
            "round-2 interp-critique follow-up: continuous-companion dual-DV bootstrap "
            "(identical predictor arrays + idx multisets), leave-one-language-out caps, "
            "EM stored-layer-21 read, map-corpus CJK counts"
        ),
    }

    # --- 1. idx reproduction check -------------------------------------------------
    n_t_em = scores["conditions"][EM_STEMS[0]]["n_layers"] and len(
        scores["conditions"][EM_STEMS[0]]["trigger_labels"]
    )
    n_t_caps = len(scores["conditions"][CAPS_STEMS[0]]["trigger_labels"])
    idx_em = _idx_for("em", n_t_em)
    idx_caps = _idx_for("caps", n_t_caps)
    rep = {}
    for setting, idx in (("em", idx_em), ("caps", idx_caps)):
        want = committed["bootstrap"][setting]["idx_sha256"]
        got = _sha(idx)
        if got != want:
            raise AssertionError(f"{setting}: regenerated idx sha {got} != committed {want}")
        rep[setting] = {"idx_sha256": got, "matches_committed": True}
    out["idx_reproduction"] = rep

    # --- 2. continuous-companion bootstraps ----------------------------------------
    out["continuous_companion"] = {
        "caps_mean_uppercase_fraction": _setting_block(
            CAPS_STEMS,
            "caps",
            PIN_CAPS_DEFAULT,
            "mean_uppercase_fraction",
            scores,
            rates_em,
            caps_models,
            idx_caps,
        ),
        "em_mean_misalignment": _setting_block(
            EM_STEMS,
            "em",
            PIN_EM_DEFAULT,
            "mean_misalignment",
            scores,
            rates_em,
            caps_models,
            idx_em,
        ),
        "note": (
            "EM mean_misalignment rises with misalignment (mean 100-point misalignment "
            "judge score over coherent-scored completions); caps mean_uppercase_fraction "
            "is the mean fraction of alphabetic tokens fully uppercase per completion. "
            "Predictor arrays + idx multisets identical to the registered binary-rate run."
        ),
    }

    # --- 2b. binary-rate re-check (sanity: must reproduce committed numbers) --------
    out["binary_rate_recheck"] = {
        "caps": _setting_block(
            CAPS_STEMS,
            "caps",
            PIN_CAPS_DEFAULT,
            "caps_rate",
            scores,
            rates_em,
            caps_models,
            idx_caps,
        ),
        "em": _setting_block(
            EM_STEMS,
            "em",
            PIN_EM_DEFAULT,
            "em_rate",
            scores,
            rates_em,
            caps_models,
            idx_em,
        ),
    }

    # --- 3. leave-one-language-out caps (binary rate) -------------------------------
    lolo = {}
    for held_out in CAPS_STEMS:
        keep = [m for m in CAPS_STEMS if m != held_out]
        lolo[f"drop_{held_out}"] = _setting_block(
            keep,
            "caps",
            PIN_CAPS_DEFAULT,
            "caps_rate",
            scores,
            rates_em,
            caps_models,
            idx_caps,
        )
    out["leave_one_language_out_caps"] = lolo

    # --- 4. EM stored-layer-21 read --------------------------------------------------
    l21 = {}
    for fam in ("ctx_trainref", "ans_trainref_mapI", "ceiling_trainref"):
        per = {}
        for m in EM_STEMS:
            cond = scores["conditions"][m]
            layered, _ = condition_matrices(cond)
            y = _dv_vector(m, cond, rates_em, caps_models, "em_rate")
            per[m] = _r6(_pair_corr(layered[fam][21], y, spearman=True))
        l21[fam] = {"per_condition": per, "mean": _r6(float(np.mean(list(per.values()))))}
    out["em_stored_layer21"] = l21

    # --- 5. CJK intrusion counts -----------------------------------------------------
    cjk: dict = {"map_corpus": {}, "em_sweep": {}}
    for sub in sorted((RAWCOMP / "raw_completions" / "map_corpus").iterdir()):
        f = sub / "raw_completions.json"
        if f.exists():
            cjk["map_corpus"][sub.name] = _cjk_scan_rows(f, "map_corpus")
    for sub in sorted((RAWCOMP / "raw_completions" / "em_sweep").iterdir()):
        f = sub / "raw_completions.json"
        if f.exists():
            cjk["em_sweep"][sub.name] = _cjk_scan_rows(f, "sweep")
    out["cjk_intrusion"] = cjk

    out_path = EVAL_DIR / "r2_followup.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {out_path}")

    # console digest
    cc = out["continuous_companion"]["caps_mean_uppercase_fraction"]
    print(
        "caps continuous: pooled delta",
        cc["pooled_delta_observed"],
        "CI",
        cc["pooled_ci95"],
        "frac<0",
        cc["boot_frac_below0"],
    )
    ce = out["continuous_companion"]["em_mean_misalignment"]
    print(
        "em continuous:   pooled delta",
        ce["pooled_delta_observed"],
        "CI",
        ce["pooled_ci95"],
        "frac<0",
        ce["boot_frac_below0"],
    )
    for k, v in out["leave_one_language_out_caps"].items():
        print(f"LOLO {k}: pooled {v['pooled_delta_observed']} CI {v['pooled_ci95']}")
    print("layer21 EM means:", {k: v["mean"] for k, v in l21.items()})


if __name__ == "__main__":
    main()
