#!/usr/bin/env python
"""Issue #2479 r4 — ONE REAL tiny slice of the P3 control chain, end to end
(r3 reconciler concern `p3-controls-smoke-unexecuted`).

Executes the newly wired control chain with REAL Anthropic judge spend at
smoke scale, through the PRODUCTION entrypoint CLIs — never library shortcuts:

  axis_legs    2 fixture characters x 2 items x 5 draws via
               issue1345_onpolicy_judge_legs.py --leg ai_likeness --census
               --execute (20 real judge calls)
  flatness     issue2479_instrument_gates.py --step flatness --execute over
               the same 2 characters (2 x 2 x 5 = 20 calls)
  namemask     issue2479_instrument_gates.py --step namemask --execute
               (2 x 2 x 5 = 20 calls)
  gates        issue2479_instrument_gates.py --step gates against a scratch
               freeze whose gates.axis_range is computed from the REAL smoke
               axis-leg means
  publication  hub.retry_transient(upload_folder) of the scratch tree to the
               clearly-marked smoke HF prefix
               issue2479_ai_likeness_gradient/smoke/p3_controls + a scoped
               raising verify (hub.assert_hf_prefix_exists)
  resume_demo  the validated resume predicate run against a REAL smoke leg
               report — expected rc=3 + quarantine (the smoke instrument's
               sync threshold_base differs from production's forced-batch 0),
               run AFTER publication so the quarantine never strands raw draws

Sanctioned smoke deviations (disclosed in the round marker):
  * SYNC dispatch route via --threshold-base 1000000 (deterministically sync
    for a tiny slice; the Batch route stays certified by the plan-§7 pilot
    contract — Batch SLA risk makes a 24h-quEUE smoke unusable);
  * the axis pilot PASS is SYNTHESIZED at the CURRENT instrument + the
    scratch data identity (a real 51-effective-draw pilot exceeds the
    20-60-call smoke budget); require_pilot_pass's READ path — the thing the
    wired gates execute — is exercised for real at all spend points;
  * scratch fixtures (synthetic benign text, never LMSYS-derived rows) and a
    scratch freeze — production paths under eval_results/ are never written.

Spend: requires EPM_I1345_JUDGE_SPEND_OK=1 at launch (~60 real judge calls).
Outputs land under /tmp/issue-2479-smoke-p3controls (scratch-dir redirect);
an EPM_I2479_SMOKE_ROOT override is CONTAINMENT-GATED before any write
(`validate_smoke_root`, r4 codex `smoke-root-production-poisoning`): the root
must resolve strictly under /tmp, $TMPDIR, or repo-local
data/issue_2479/smoke_* — anything else inside the repository tree
(eval_results/ and every other issue's dirs especially) or outside every
approved scratch area is REFUSED fail-loud. The synthesized pilot is honored
by `require_pilot_pass` ONLY under jp.ALLOW_SYNTHESIZED_ENV=1, which this
driver alone injects into its subprocess env.
Digests print paths + bytes + sha256 + counts, never judged row text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

import issue1345_onpolicy_judge_legs as jl  # noqa: E402
import issue2479_judge_pilots as jp  # noqa: E402

SMOKE_ROOT_ENV = "EPM_I2479_SMOKE_ROOT"
DEFAULT_SMOKE_ROOT = "/tmp/issue-2479-smoke-p3controls"
# Module-level value for interactive/import use; main() re-reads the env and
# CONTAINMENT-GATES it via validate_smoke_root BEFORE any write.
SCRATCH = Path(os.environ.get(SMOKE_ROOT_ENV, DEFAULT_SMOKE_ROOT))
SMOKE_PREFIX = "issue2479_ai_likeness_gradient/smoke/p3_controls"
DATA_REPO = "superkaiba1/explore-persona-space-data"
THRESHOLD_BASE_SYNC = 1_000_000  # decide_route: tiny n -> deterministically sync
CHARS = (("kestrel", "Kestrel", "A"), ("auriga", "Auriga", "D"))
CONV_IDS = ("sc1", "sc2")


def validate_smoke_root(scratch: Path, repo_root: Path = _REPO_ROOT) -> Path:
    """Containment gate for the CONFIGURABLE smoke root — returns it RESOLVED.

    r4 codex `smoke-root-production-poisoning`: every write in this driver
    derives from SCRATCH, and the whole tree is bulk-published + partially
    destroyed (the resume-demo quarantine), so an uncontained override — e.g.
    ``EPM_I2479_SMOKE_ROOT=eval_results/issue_2479`` — would overwrite the
    COMMITTED registered panel/manifest and park a synthesized pilot PASS at
    the production path. Policy (raise RuntimeError BEFORE any write):

      * repo-internal roots are refused, with ONE carve-out: strictly under
        ``<repo>/data/issue_2479`` with a first path component starting
        ``smoke_`` (gitignored per-issue scratch);
      * outside the repo, only roots strictly under an approved temp area —
        ``/tmp`` or ``$TMPDIR`` — are allowed (never the temp root itself:
        SCRATCH is uploaded and quarantined wholesale);
      * everything else is refused.
    """
    resolved = scratch.expanduser().resolve()
    repo = repo_root.resolve()
    if resolved.is_relative_to(repo):
        approved = repo / "data" / "issue_2479"
        if resolved != approved and resolved.is_relative_to(approved):
            first = resolved.relative_to(approved).parts[0]
            if first.startswith("smoke_"):
                return resolved
        raise RuntimeError(
            f"refusing repo-internal smoke root {resolved} (from {SMOKE_ROOT_ENV}): the "
            "driver writes/publishes/quarantines the WHOLE root, so a repo-internal root "
            "can clobber committed artifacts (eval_results/, other issues' dirs). Approved "
            f"repo-local scratch: {approved}/smoke_* ; otherwise use /tmp or $TMPDIR "
            "(r4 codex smoke-root-production-poisoning)"
        )
    tmp_roots = [Path("/tmp").resolve()]
    tmpdir = os.environ.get("TMPDIR", "").strip()
    if tmpdir:
        tmp_roots.append(Path(tmpdir).expanduser().resolve())
    for root in tmp_roots:
        if resolved != root and resolved.is_relative_to(root):
            return resolved
    raise RuntimeError(
        f"refusing smoke root {resolved} (from {SMOKE_ROOT_ENV}): not strictly under an "
        f"approved scratch area ({', '.join(str(r) for r in tmp_roots)}, or "
        f"{repo / 'data' / 'issue_2479'}/smoke_*) "
        "(r4 codex smoke-root-production-poisoning)"
    )


def _digest(path: Path, note: str = "") -> None:
    b = path.read_bytes()
    print(
        f"[smoke-digest] {path} bytes={len(b)} sha256={hashlib.sha256(b).hexdigest()[:16]} {note}",
        flush=True,
    )


def _run(phase: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"[phase={phase}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, timeout=1200)
    assert proc.returncode == 0, f"{phase}: rc={proc.returncode}"
    print(f"[phase={phase}] rc=0 elapsed={time.time() - t0:.0f}s", flush=True)


def build_fixtures() -> dict[str, Path]:
    """Scratch panel/manifest/items/kept-story fixtures (synthetic benign text)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    panel = [
        {
            "name": name,
            "variant_op": f"char_2479_{name}_op",
            "variant_inserted": f"char_2479_{name}_ins",
            "design_band": band,
            "display_name": disp,
        }
        for name, disp, band in CHARS
    ]
    panel_p = SCRATCH / "panel.json"
    panel_p.write_text(json.dumps(panel, indent=1))
    manifest_p = SCRATCH / "panel_manifest.json"
    manifest_p.write_text(
        json.dumps(
            {"axis_reservation_conv_ids": list(CONV_IDS), "n_reservation": len(CONV_IDS)},
            indent=1,
        )
    )
    items_dir = SCRATCH / "axis_items"
    items_dir.mkdir(exist_ok=True)
    for name, disp, _band in CHARS:
        rows = [
            {
                "conv_id": cid,
                "question": f"Synthetic smoke question {cid}: what should the reader take away?",
                # The display name appears so the name-mask leg's masking HITS.
                "answer": (
                    f"{disp} paused before answering. {disp} explained, at a measured pace, "
                    f"that the takeaway for {cid} is to check every assumption twice before "
                    "acting on it."
                ),
            }
            for cid in CONV_IDS
        ]
        (items_dir / f"axis_items_{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    # Verbatim-flatness kept rows: the embedded reference answer is IDENTICAL
    # across characters per conv_id (the flatness identity invariant).
    for name, _disp, _band in CHARS:
        rows = [
            {
                "conv_id": cid,
                "question": f"Synthetic smoke reference question {cid}?",
                "answer": (
                    f"The shared reference answer for {cid}: measure first, then decide, and "
                    "write the decision down so it can be checked later."
                ),
            }
            for cid in CONV_IDS
        ]
        (SCRATCH / f"kept_char_2479_{name}_ins.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    return {"panel": panel_p, "manifest": manifest_p, "items_dir": items_dir}


def synthesize_pilot(fix: dict[str, Path]) -> Path:
    """A PASS report at the CURRENT instrument + the SCRATCH data identity.

    SANCTIONED SMOKE DEVIATION (marker-disclosed): a real 51-effective-draw
    pilot exceeds the smoke budget; every spend path still executes
    require_pilot_pass's real read path against this report.
    """
    identity = jp.axis_data_identity(
        panel_path=fix["panel"],
        manifest_path=fix["manifest"],
        items_glob=str(fix["items_dir"] / "axis_items_{name}.jsonl"),
    )
    rep = {
        "issue": 2479,
        "family": "axis",
        "passed": True,
        "verdict": "PASS",
        "failures": [],
        "smoke_synthesized": True,
        "note": (
            "r4 p3-controls smoke: synthesized PASS at the current instrument + scratch "
            "data identity; NOT a production pilot (production runs judge_pilot_gate)"
        ),
        "instrument": dict(jp.axis_instrument_fingerprint()),
        "data_identity": identity,
        "arms": {"axis": {"n_draws": 150, "n_transport_lost": 0}},
        "metadata": {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    p = SCRATCH / "pilot_gate_axis.json"
    p.write_text(json.dumps(rep, indent=1))
    _digest(p, "(synthesized smoke pilot PASS)")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--skip-upload", action="store_true", help="skip the HF publication phase")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("import-ok: issue2479_p3_controls_smoke", flush=True)
        return 0

    # Containment gate FIRST — before the spend ack and before ANY write
    # (r4 codex `smoke-root-production-poisoning`).
    global SCRATCH
    SCRATCH = validate_smoke_root(Path(os.environ.get(SMOKE_ROOT_ENV, DEFAULT_SMOKE_ROOT)))
    print(f"[smoke-root] contained: {SCRATCH}", flush=True)

    assert os.environ.get(jl.SPEND_ACK_ENV) == "1", (
        f"{jl.SPEND_ACK_ENV}=1 required — this smoke makes ~60 REAL judge calls"
    )
    fix = build_fixtures()
    pilot = synthesize_pilot(fix)
    legs = SCRATCH / "legs"
    legs.mkdir(exist_ok=True)
    # Explicit env for EVERY subprocess (subprocess-env rule): scratch data
    # identity + the run_leg pilot guard armed exactly as the wrapper arms it.
    # ALLOW_SYNTHESIZED_ENV is the smoke-ONLY licence for the synthesized
    # pilot: production require_pilot_pass callers (no env) refuse it (r4
    # codex `smoke-root-production-poisoning`).
    env = {
        **os.environ,
        jp.PANEL_ENV: str(fix["panel"]),
        jp.MANIFEST_ENV: str(fix["manifest"]),
        jp.ITEMS_DIR_ENV: str(fix["items_dir"]),
        jp.REQUIRE_AXIS_PILOT_ENV: str(pilot),
        jp.ALLOW_SYNTHESIZED_ENV: "1",
    }

    for name, _disp, _band in CHARS:
        _run(
            f"smoke_axis_leg_{name}",
            [
                "uv",
                "run",
                "python",
                "scripts/issue1345_onpolicy_judge_legs.py",
                "--leg",
                "ai_likeness",
                "--rows",
                str(fix["items_dir"] / f"axis_items_{name}.jsonl"),
                "--character",
                name,
                "--census",
                "--out-dir",
                str(legs),
                "--threshold-base",
                str(THRESHOLD_BASE_SYNC),
                "--execute",
            ],
            env,
        )
        rep = json.loads((legs / f"judge_report_ail_{name}.json").read_text())
        assert rep["spend_executed"] is True and rep["means"]["pooled"]["n"] == len(CONV_IDS)
        raw = json.loads((legs / f"judge_raw_ail_{name}.json").read_text())
        n_draws = len(raw.get("all_scores") or {})
        assert n_draws == len(CONV_IDS) * jl.N_DRAWS, f"{name}: {n_draws} raw draws"
        _digest(
            legs / f"judge_report_ail_{name}.json",
            f"(pooled mean={rep['means']['pooled']['mean']:.1f} n={rep['means']['pooled']['n']} "
            f"raw_draws={n_draws})",
        )

    common_gate = [
        "--panel",
        str(fix["panel"]),
        "--manifest",
        str(fix["manifest"]),
        "--legs-dir",
        str(legs),
        "--axis-pilot-report",
        str(pilot),
        "--threshold-base",
        str(THRESHOLD_BASE_SYNC),
    ]
    _run(
        "smoke_flatness",
        [
            "uv",
            "run",
            "python",
            "scripts/issue2479_instrument_gates.py",
            "--step",
            "flatness",
            "--kept-glob",
            str(SCRATCH / "kept_{variant}.jsonl"),
            *common_gate,
            "--execute",
        ],
        env,
    )
    _run(
        "smoke_namemask",
        [
            "uv",
            "run",
            "python",
            "scripts/issue2479_instrument_gates.py",
            "--step",
            "namemask",
            "--items-glob",
            str(fix["items_dir"] / "axis_items_{name}.jsonl"),
            "--axis-raw-glob",
            str(legs / "judge_raw_ail_{name}.json"),
            *common_gate,
            "--execute",
        ],
        env,
    )
    for name, _disp, _band in CHARS:
        for prefix in ("flat", "mask"):
            _digest(legs / f"judge_report_ail_{prefix}_{name}.json")

    # Scratch freeze: gates.axis_range from the REAL smoke axis-leg means.
    means = [
        json.loads((legs / f"judge_report_ail_{name}.json").read_text())["means"]["pooled"]["mean"]
        for name, _disp, _band in CHARS
    ]
    freeze_p = SCRATCH / "axis_freeze.json"
    freeze_p.write_text(
        json.dumps(
            {
                "issue": 2479,
                "smoke_synthesized": True,
                "gates": {"axis_range": max(means) - min(means)},
            }
        )
    )
    gates_out = SCRATCH / "instrument_gates.json"
    _run(
        "smoke_gates",
        [
            "uv",
            "run",
            "python",
            "scripts/issue2479_instrument_gates.py",
            "--step",
            "gates",
            "--axis-raw-glob",
            str(legs / "judge_raw_ail_{name}.json"),
            "--freeze",
            str(freeze_p),
            "--out",
            str(gates_out),
            *common_gate,
        ],
        env,
    )
    gates = json.loads(gates_out.read_text())
    print(
        f"[smoke-digest] gate booleans: verbatim_flatness_pass="
        f"{gates['gates']['verbatim_flatness_pass']} "
        f"name_mask_pass={gates['gates']['name_mask_pass']}",
        flush=True,
    )
    _digest(gates_out)

    if not args.skip_upload:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        print(f"[phase=smoke_publication] -> {DATA_REPO}/{SMOKE_PREFIX}", flush=True)
        api = HfApi()
        # Dir-filecount guard BEFORE the upload, OUTSIDE the retry wrapper
        # (a guard raise is deterministic; the Hub 400s >10k files/dir).
        hub.assert_hub_dir_filecounts(SCRATCH, SMOKE_PREFIX)
        hub.retry_transient(
            lambda: api.upload_folder(
                repo_id=DATA_REPO,
                repo_type="dataset",
                folder_path=str(SCRATCH),
                path_in_repo=SMOKE_PREFIX,
                commit_message="issue-2479 r4: p3-controls real tiny-slice smoke outputs",
            ),
            what="upload_folder(p3_controls smoke)",
        )
        n = hub.assert_hf_prefix_exists(api, DATA_REPO, SMOKE_PREFIX, repo_type="dataset")
        print(f"[phase=smoke_publication] rc=0 verified ({n} files at {SMOKE_PREFIX})", flush=True)

    # Resume-validator NEGATIVE demo on a REAL smoke artifact, AFTER
    # publication: the smoke instrument's sync threshold_base != production's
    # forced-batch 0, so the predicate must REFUSE the skip (rc=3+quarantine).
    name = CHARS[0][0]
    demo = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/issue2479_p3_leg_resume.py",
            "--report",
            str(legs / f"judge_report_ail_{name}.json"),
            "--tag",
            name,
            "--items",
            str(fix["items_dir"] / f"axis_items_{name}.jsonl"),
            "--expect-design",
            "axis-census",
            "--pilot-report",
            str(pilot),
        ],
        cwd=_REPO_ROOT,
        env=env,
        timeout=600,
    )
    assert demo.returncode == 3, f"resume negative demo: rc={demo.returncode} (expected 3)"
    quarantined = sorted(p.name for p in legs.glob(f"*_{name}.json.quarantined-*"))
    assert quarantined, "resume negative demo quarantined nothing"
    print(
        f"[phase=smoke_resume_demo] rc=3 as expected; quarantined: {quarantined}",
        flush=True,
    )
    print("[phase=done] p3-controls smoke complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
