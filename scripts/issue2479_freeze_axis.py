#!/usr/bin/env python
"""Issue #2479 — pre-registered AI-likeness axis freeze (plan §4 Step 3).

Two modes:

  --emit-items   Build the per-character axis-scoring item lists the reused
                 judge instrument (`issue1345_onpolicy_judge_legs.py --leg
                 ai_likeness`) consumes: each character's kept ON-POLICY rows,
                 normalized to the judge-leg schema via the parent's own
                 converter (`issue1345_judge_rows_prep.prepare` — never a
                 re-implementation), RESTRICTED to the 250-conversation axis
                 reservation in `panel_manifest.json`.

  (default)      Compute + write the frozen axis `axis_freeze.json` from the
                 per-character judge-leg REPORT files: per-character score
                 (the instrument's own pooled mean of per-item 5-draw means,
                 drop-never-coerce carried through), rank, item n, drop
                 counts, the two registered instrument gates computed here —
                 `band_agreement_pass` (Spearman(design-band ordinal, axis
                 score) >= 0.5) and `axis_range_pass` (max-min >= 8.0, plan
                 §6 gates 1-2) — the rubric fingerprint (sha256 of the EXACT
                 bytes the parent instrument judges with:
                 `issue1345_onpolicy_judge_legs.AI_LIKENESS_RUBRIC`), input-leg
                 file shas, panel.json + manifest shas, and seeds.

  --commit       After writing the freeze JSON: `git add` + `git commit` BY
                 EXPLICIT PATH on the current branch, bare `git push` (rc
                 checked), then post the `epm:progress` note
                 "axis-frozen commit=<sha>" via the MAIN-checkout task.py
                 (resolved through the git common dir — task.py branch-guards
                 to main, so the marker posts from the canonical checkout
                 while the freeze commit lands on the issue branch). VM-side
                 ONLY by construction: the tasks/ tree exists only on the VM
                 main checkout, so a pod-side invocation crashes loudly.

Content hygiene: kept rows are LMSYS-derived real user text — this script
never prints row text; every diagnostic is a count, a path, or a hash.

Fail-loud everywhere: missing legs, characters with zero items, malformed leg
JSON, dry-run leg reports (no means), and rubric-fingerprint drift between
this module's computed sha and each leg report's own `rubric_sha256` all
RAISE.

CLI:
  uv run python scripts/issue2479_freeze_axis.py --emit-items \\
      --kept-glob 'data/issue_2479/gen/{variant}/kept_stories_op_instruct.jsonl' \\
      --items-out-dir data/issue_2479/axis_items
  uv run python scripts/issue2479_freeze_axis.py --legs-dir eval_results/issue_2479/judge_legs
  uv run python scripts/issue2479_freeze_axis.py --legs-dir ... --commit
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import issue1345_common as c  # noqa: E402
import issue1345_judge_rows_prep as prep_mod  # noqa: E402
import issue1345_onpolicy_judge_legs as jl  # noqa: E402

ISSUE = 2479
FREEZE_REL = "eval_results/issue_2479/axis_freeze.json"
PANEL_REL = "eval_results/issue_2479/panel.json"
MANIFEST_REL = "eval_results/issue_2479/panel_manifest.json"

# Design-band ordinal (plan §6 gate 1): A = strongly AI-like .. D = strongly
# non-AI. Encoded descending so a POSITIVE Spearman with the judged axis is
# the expected direction.
BAND_ORDINAL = {"A": 3, "B": 2, "C": 1, "D": 0}
BAND_AGREEMENT_MIN = 0.5  # plan §6 gate 1 (parent recomputes to 0.8)
AXIS_RANGE_MIN = 8.0  # plan §6 gate 2 (parent range 21, inter-band gaps ~7)

REQUIRED_PANEL_KEYS = {"name", "variant_op", "variant_inserted", "design_band", "display_name"}


def sha256_path(path: Path) -> str:
    """Hex sha256 of a file's bytes (fail-loud on a missing file)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel(path: Path) -> list[dict]:
    """The committed #2479 panel registry, schema-checked (fail-loud)."""
    rows = json.loads(path.read_text())
    assert isinstance(rows, list) and rows, f"{path}: expected non-empty JSON list"
    for i, r in enumerate(rows):
        missing = REQUIRED_PANEL_KEYS - r.keys()
        assert not missing, f"{path}: row {i} missing keys {sorted(missing)}"
        assert r["design_band"] in BAND_ORDINAL, (
            f"{path}: row {i} design_band {r['design_band']!r} not in {sorted(BAND_ORDINAL)}"
        )
    names = [r["name"] for r in rows]
    assert len(set(names)) == len(names), f"{path}: duplicate character names"
    return rows


def load_reservation_ids(manifest_path: Path) -> set[str]:
    """The axis-reservation conv_id set from panel_manifest.json (fail-loud)."""
    m = json.loads(manifest_path.read_text())
    ids = m["axis_reservation_conv_ids"]
    assert isinstance(ids, list) and ids, f"{manifest_path}: empty axis_reservation_conv_ids"
    id_set = {str(x) for x in ids}
    assert len(id_set) == len(ids), f"{manifest_path}: duplicate reservation conv_ids"
    n_expected = int(m.get("n_reservation", len(ids)))
    assert len(ids) == n_expected, (
        f"{manifest_path}: n_reservation={n_expected} but {len(ids)} reservation ids"
    )
    return id_set


def rubric_fingerprint() -> str:
    """sha256 of EXACTLY the rubric bytes the parent instrument judges with.

    The ai_likeness rubric is the module-level constant
    `issue1345_onpolicy_judge_legs.AI_LIKENESS_RUBRIC`; `run_leg` fingerprints
    `RUBRIC[leg].encode()` into every leg report, so this is byte-identical to
    the parent's own `rubric_sha256` field (cross-checked per leg below).
    """
    return hashlib.sha256(jl.AI_LIKENESS_RUBRIC.encode()).hexdigest()


def band_agreement_gate(scores: dict[str, float], bands: dict[str, str]) -> dict:
    """Gate 1: Spearman(design-band ordinal, judged axis score) >= 0.5."""
    from scipy.stats import spearmanr

    names = sorted(scores)
    assert len(names) >= 2, f"band agreement needs >=2 characters, got {len(names)}"
    xs = [BAND_ORDINAL[bands[n]] for n in names]
    ys = [scores[n] for n in names]
    rho = float(spearmanr(xs, ys).statistic)
    return {
        "band_agreement_rho": rho,
        "band_agreement_threshold": BAND_AGREEMENT_MIN,
        "band_ordinal": dict(BAND_ORDINAL),
        "n_characters": len(names),
        "band_agreement_pass": bool(rho >= BAND_AGREEMENT_MIN),
    }


def axis_range_gate(scores: dict[str, float]) -> dict:
    """Gate 2: realized axis range (max - min) >= 8.0 points."""
    vals = list(scores.values())
    assert vals, "axis range needs >=1 character score"
    rng = float(max(vals) - min(vals))
    return {
        "axis_range": rng,
        "axis_min": float(min(vals)),
        "axis_max": float(max(vals)),
        "axis_range_threshold": AXIS_RANGE_MIN,
        "axis_range_pass": bool(rng >= AXIS_RANGE_MIN),
    }


def load_leg_report(legs_dir: Path, name: str) -> tuple[dict, Path]:
    """One character's ai_likeness leg REPORT, spend-executed + schema-checked."""
    path = legs_dir / f"judge_report_{jl.LEG_SLUG[jl.LEG_AI_LIKENESS]}_{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing judge-leg report for character {name!r}: {path} — run "
            f"issue1345_onpolicy_judge_legs.py --leg ai_likeness --character {name} first"
        )
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: malformed leg report JSON: {e}") from e
    assert report.get("leg") == jl.LEG_AI_LIKENESS, f"{path}: leg={report.get('leg')!r}"
    assert report.get("tag") == name, f"{path}: tag={report.get('tag')!r} != {name!r}"
    assert report.get("spend_executed") is True, (
        f"{path}: spend_executed={report.get('spend_executed')!r} — a dry-run leg report "
        "carries no scores; the axis cannot be frozen from it"
    )
    means = report.get("means")
    assert isinstance(means, dict) and isinstance(means.get("pooled"), dict), (
        f"{path}: no means.pooled block (malformed leg report)"
    )
    return report, path


def freeze_axis(panel: list[dict], legs_dir: Path, manifest_path: Path, panel_path: Path) -> dict:
    """Assemble the axis_freeze.json payload from the per-character leg reports."""
    fp = rubric_fingerprint()
    manifest = json.loads(manifest_path.read_text())
    chars: dict[str, dict] = {}
    scores: dict[str, float] = {}
    bands: dict[str, str] = {}
    judge_sample_seeds: dict[str, int] = {}
    for row in panel:
        name = row["name"]
        report, path = load_leg_report(legs_dir, name)
        assert report["rubric_sha256"] == fp, (
            f"{path}: leg rubric_sha256 {report['rubric_sha256'][:16]} != this module's "
            f"computed ai_likeness fingerprint {fp[:16]} — instrument drift; the axis must "
            "be frozen against the exact parent rubric bytes"
        )
        pooled = report["means"]["pooled"]
        score, n_items = pooled.get("mean"), pooled.get("n")
        assert isinstance(n_items, int) and n_items > 0, (
            f"{path}: character {name!r} has zero scored items — cannot enter the axis"
        )
        assert isinstance(score, int | float), f"{path}: pooled mean is {score!r}"
        design = report.get("sample_design") or {}
        if design.get("seed") is not None:
            judge_sample_seeds[name] = int(design["seed"])
        scores[name] = float(score)
        bands[name] = row["design_band"]
        chars[name] = {
            "tag": report["tag"],
            "design_band": row["design_band"],
            "variant_op": row["variant_op"],
            "score": float(score),
            "n_items": int(report.get("n_items") or 0),
            "n_scored_items": n_items,
            "drops": {
                "n_dropped_draws_content": report.get("n_dropped_draws_content"),
                "n_refusal_draws": report.get("n_refusal_draws"),
                "n_transport_lost_draws": report.get("n_transport_lost_draws"),
                "n_total_draws": report.get("n_total_draws"),
                "n_unscored_items": report["means"].get("n_unscored_items"),
            },
            "leg_report_path": str(path),
            "leg_report_sha256": sha256_path(path),
        }
    # Rank 1 = most AI-like; deterministic tie-break by name.
    ordered = sorted(scores, key=lambda n: (-scores[n], n))
    for rank, name in enumerate(ordered, start=1):
        chars[name]["rank"] = rank

    gates = {**band_agreement_gate(scores, bands), **axis_range_gate(scores)}
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return {
        "issue": ISSUE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "script": "scripts/issue2479_freeze_axis.py",
            **as_metadata_dict(git_provenance()),
        },
        "rubric_sha256": fp,
        "rubric_source": "scripts/issue1345_onpolicy_judge_legs.py::AI_LIKENESS_RUBRIC",
        "judge_model": jl.JUDGE_MODEL,
        "n_draws": jl.N_DRAWS,
        "judge_max_tokens": jl.JUDGE_MAX_TOKENS,
        "panel_path": str(panel_path),
        "panel_sha256": sha256_path(panel_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "n_reservation": int(manifest["n_reservation"]),
        "seeds": {
            **{k: v for k, v in (manifest.get("seeds") or {}).items()},
            "judge_sample_seed_per_char": judge_sample_seeds,
        },
        "characters": chars,
        "gates": gates,
    }


def commit_and_post(freeze_path: Path, repo_root: Path, issue: int = ISSUE) -> str:
    """Commit + push the freeze BY EXPLICIT PATH; post the axis-frozen marker.

    The marker posts via the MAIN-checkout task.py (resolved through the git
    common dir): task.py branch-guards to main, so it must run from the
    canonical checkout, not this (issue-branch) worktree. Returns the freeze
    commit sha (pasted verbatim from `git rev-parse HEAD`).
    """
    rel = str(freeze_path.resolve().relative_to(repo_root.resolve()))
    subprocess.run(["git", "-C", str(repo_root), "add", "--", rel], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "commit",
            "-m",
            f"task #{issue}: axis freeze — pre-registered AI-likeness axis",
            "--",
            rel,
        ],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Bare push, rc checked — never piped (the pipe masks a rejected push).
    subprocess.run(["git", "-C", str(repo_root), "push", "origin", branch], check=True)

    common = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    main_root = Path(common).parent
    task_py = main_root / "scripts" / "task.py"
    assert task_py.is_file(), f"main-checkout task.py not found at {task_py}"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(task_py),
            "post-marker",
            str(issue),
            "epm:progress",
            "--note",
            f"axis-frozen commit={sha}",
        ],
        check=True,
        cwd=str(main_root),
    )
    return sha


def emit_items(
    panel: list[dict],
    reservation: set[str],
    kept_glob: str,
    raw_glob: str | None,
    items_out_dir: Path,
) -> None:
    """Per-character axis item lists: prepared kept ON-POLICY rows, reservation-filtered.

    Normalization is the parent's own converter (`issue1345_judge_rows_prep.
    prepare`): span-derived answers for op rows, `capped` from the row field /
    `finish_reason` (or the raw-file join when --raw-glob is given). Writes
    `axis_items_<name>.jsonl` + a counts sidecar per character; never prints
    row text.
    """
    items_out_dir.mkdir(parents=True, exist_ok=True)
    for row in panel:
        name, variant = row["name"], row["variant_op"]
        kept_path = Path(kept_glob.format(variant=variant, name=name))
        if not kept_path.is_file():
            raise FileNotFoundError(f"{name}: kept rows file missing: {kept_path}")
        rows = c.read_jsonl(kept_path)
        assert rows, f"{name}: {kept_path} is empty"
        capped_index = None
        if raw_glob:
            raw_path = Path(raw_glob.format(variant=variant, name=name))
            if not raw_path.is_file():
                raise FileNotFoundError(f"{name}: raw rows file missing: {raw_path}")
            capped_index = prep_mod.load_capped_index(raw_path)
        prepared, stats = prep_mod.prepare(rows, capped_index, cell=variant)
        axis_rows = [r for r in prepared if str(r["conv_id"]) in reservation]
        assert axis_rows, (
            f"{name}: zero prepared rows fall in the {len(reservation)}-conversation axis "
            f"reservation (prepared={len(prepared)}) — the axis cannot be scored for this "
            "character"
        )
        out_path = items_out_dir / f"axis_items_{name}.jsonl"
        if out_path.exists():
            out_path.unlink()
        c.append_jsonl(out_path, axis_rows)
        c.write_json(
            items_out_dir / f"axis_items_{name}.stats.json",
            {
                "character": name,
                "variant": variant,
                "kept_path": str(kept_path),
                "n_kept": len(rows),
                "n_prepared": len(prepared),
                "n_axis_items": len(axis_rows),
                "n_reservation": len(reservation),
                "prep_stats": stats,
            },
        )
        print(
            f"[emit-items] {name}: kept={len(rows)} prepared={len(prepared)} "
            f"axis_items={len(axis_rows)} -> {out_path}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", type=Path, default=_REPO_ROOT / PANEL_REL)
    ap.add_argument("--manifest", type=Path, default=_REPO_ROOT / MANIFEST_REL)
    ap.add_argument("--legs-dir", type=Path, default=None, help="dir of judge_report_ail_*.json")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / FREEZE_REL)
    ap.add_argument(
        "--commit",
        action="store_true",
        help="git add/commit BY EXPLICIT PATH + push + post the axis-frozen marker "
        "via the MAIN-checkout task.py (VM-side only)",
    )
    ap.add_argument("--emit-items", action="store_true", help="emit per-character item lists")
    ap.add_argument(
        "--kept-glob",
        default=None,
        help="emit-items: kept ON-POLICY rows path template with {variant} (and/or {name})",
    )
    ap.add_argument(
        "--raw-glob",
        default=None,
        help="emit-items: optional raw-generation path template for the capped join "
        "(only needed when kept rows lack finish_reason/capped)",
    )
    ap.add_argument("--items-out-dir", type=Path, default=None)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        # Deferred imports on the real code paths, named explicitly (#1689).
        from scipy.stats import spearmanr  # noqa: F401

        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        print(f"import-ok: rubric_sha256={rubric_fingerprint()[:16]}", flush=True)
        return

    panel = load_panel(args.panel)

    if args.emit_items:
        assert args.kept_glob and args.items_out_dir, (
            "--emit-items requires --kept-glob and --items-out-dir"
        )
        reservation = load_reservation_ids(args.manifest)
        emit_items(panel, reservation, args.kept_glob, args.raw_glob, args.items_out_dir)
        return

    assert args.legs_dir is not None, "--legs-dir is required (freeze mode)"
    payload = freeze_axis(panel, args.legs_dir, args.manifest, args.panel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(args.out, payload)
    g = payload["gates"]
    print(
        f"[freeze] {len(payload['characters'])} characters -> {args.out}\n"
        f"[freeze] band_agreement rho={g['band_agreement_rho']:.3f} "
        f"pass={g['band_agreement_pass']}  axis_range={g['axis_range']:.2f} "
        f"pass={g['axis_range_pass']}",
        flush=True,
    )
    if args.commit:
        sha = commit_and_post(args.out, _REPO_ROOT)
        print(f"[freeze] axis-frozen commit={sha}", flush=True)
    else:
        print(
            "[freeze] NOT committed (--commit absent) — the fit-driver guard will refuse",
            flush=True,
        )


if __name__ == "__main__":
    main()
