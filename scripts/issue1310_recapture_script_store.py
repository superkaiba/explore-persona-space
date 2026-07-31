"""Issue #1310 script-format store RE-CAPTURE (store gap repair, 2026-07-30).

The run-2 SCRIPT-format activation store — the substrate of the published
per-persona cells `eval_results/issue_1310/cells_{base,instruct}_<persona>.json`
(ambient pure-GCV, no `gcv_dof_cap` field) and
`script_completion/cells_scriptc_instruct_Vex.json` — was lost with its
instance, so the `nd-estimator-audit` round could not recompute those cells.
This driver rebuilds it from the PERSISTED story text and re-uploads it under a
NEW name (`store_recap`), overwriting nothing.

Phases (blocking, in order; every phase prints its own breadcrumb):

  p0_stage        stage story text (both models) + the persisted script-format
                  instruct turn-pairs from the HF data repo
  p1_pairs        re-attribute the BASE turn-pairs deterministically (the base
                  pairs were never persisted) and gate BOTH arms' per-persona
                  counts against the published cell n
  p2_capture      `issue1310_extract_store.py --flavor perturn` per model into
                  `<data-dir>/store_recap/<model>` (28 layers, bf16, the same
                  rig the lost store was built with)
  p3_upload       one `upload_folder` commit per model + an EXACT-SET verify,
                  BEFORE any instance teardown (the loss class being repaired)

Allocation-safe: narrows to the FIRST device of the pre-set allocation and never
exports an absolute GPU index (the `issue1345_boundary_ablation_launch_gen.sh`
pattern). Single-GPU by construction — one model loaded at a time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + tokens before any heavy import (#847)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_ROOT = "issue1310_char_map"
STORE_SUBDIR = "store_recap"
STORE_PREFIX = f"{HF_ROOT}/analysis_tensors/{STORE_SUBDIR}"
DATA_DIR = REPO / "data" / "issue_1310"
SENTINEL = Path("/workspace/logs/issue-1310-results.json")

# Published per-persona turn-pair counts (the audited cells' own n).
# base: eval_results/issue_1310/cells_base_<persona>.json
# instruct: cells_instruct_{Wren,HELIOS,Dana}.json +
#           script_completion/cells_scriptc_instruct_Vex.json
PUBLISHED_PAIRS = {
    "base": {"Wren": 2329, "HELIOS": 2466, "Dana": 1325, "Vex": 2060},
    "instruct": {"Wren": 3094, "HELIOS": 3123, "Dana": 2700, "Vex": 3586},
}
COUNT_TOLERANCE = 0.05  # report + fail loud above 5% divergence (round brief)


def _sh(cmd: list[str], *, phase: str) -> None:
    """Run a blocking subprocess, fail loud with the phase name."""
    print(f"[recap] {phase}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[recap] {phase}: rc={rc} elapsed={time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        raise RuntimeError(f"{phase} failed rc={rc}")


def _first_allocated_device() -> str | None:
    """First device of the pre-set allocation; None when unset (whole node).

    NEVER exports an absolute index: an allocation-scoped launcher may hand us
    e.g. CUDA_VISIBLE_DEVICES="3,5", where torch's cuda:0 is physical 3.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return None
    return cvd.split(",")[0].strip() or None


def phase_stage() -> None:
    """Stage story text (both models) + the persisted instruct pairs from HF."""
    print("[phase=p0_stage] staging persisted script-format inputs", flush=True)
    from explore_persona_space.orchestrate import hub

    (DATA_DIR / "stories").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "pairs").mkdir(parents=True, exist_ok=True)
    spec = {
        DATA_DIR
        / "stories"
        / "base_stories_seed42.jsonl": f"{HF_ROOT}/raw_completions/generation/base_stories_seed42.jsonl",
        DATA_DIR
        / "stories"
        / "instruct_stories_seed42.jsonl": f"{HF_ROOT}/raw_completions/generation/instruct_stories_seed42.jsonl",
        # The completion round's re-attributed script-format instruct pairs —
        # the EXACT spans the published instruct cells were fit on.
        DATA_DIR
        / "pairs"
        / "instruct_pairs.jsonl": f"{HF_ROOT}/raw_completions/pairs_script_completion/instruct_pairs.jsonl",
    }
    for target, path_in_repo in spec.items():
        hub.stage_hub_file(DATA_REPO, path_in_repo, target, repo_type="dataset")
        n = sum(1 for ln in target.open(encoding="utf-8") if ln.strip())
        print(f"[stage] {target.name} staged: {n} rows", flush=True)


def _pair_counts(model_kind: str) -> dict[str, int]:
    """Per-persona turn-pair counts from the staged pairs JSONL."""
    path = DATA_DIR / "pairs" / f"{model_kind}_pairs.jsonl"
    counts: dict[str, int] = {}
    # text-mode iteration, never .splitlines() (raw U+2028 in real text, #950)
    for line in path.open(encoding="utf-8"):
        if line.strip():
            char_id = json.loads(line)["char_id"]
            counts[char_id] = counts.get(char_id, 0) + 1
    return counts


def _gate_counts(model_kind: str) -> dict:
    """Compare realized per-persona pair counts against the published cell n."""
    realized = _pair_counts(model_kind)
    published = PUBLISHED_PAIRS[model_kind]
    rows, worst = {}, 0.0
    for persona, want in published.items():
        got = realized.get(persona, 0)
        rel = abs(got - want) / want
        worst = max(worst, rel)
        rows[persona] = {"published": want, "realized": got, "rel_divergence": rel}
        print(
            f"[gate] {model_kind}/{persona}: published={want} realized={got} rel={rel:.4f}",
            flush=True,
        )
    verdict = (
        "exact"
        if worst == 0.0
        else ("within-tolerance" if worst <= COUNT_TOLERANCE else "DIVERGENT")
    )
    print(f"[gate] {model_kind}: worst rel divergence {worst:.4f} -> {verdict}", flush=True)
    if worst > COUNT_TOLERANCE:
        raise RuntimeError(
            f"{model_kind} pair counts diverge from published by {worst:.4f} "
            f"> {COUNT_TOLERANCE} — refusing to capture; rows={rows}"
        )
    return {"per_persona": rows, "worst_rel_divergence": worst, "verdict": verdict}


def phase_pairs() -> dict:
    """Re-attribute base pairs (never persisted); gate both arms' counts."""
    print("[phase=p1_pairs] base re-attribution + count gate", flush=True)
    # Deterministic regex line attribution; the Sonnet leg is a PRECISION
    # spot-check only and plays no part in pair construction, so it is skipped
    # (the count gate below is the binding check).
    _sh(
        [
            "uv",
            "run",
            "python",
            "scripts/issue1310_attribute.py",
            "--model",
            "base",
            "--data-dir",
            str(DATA_DIR),
            "--out-dir",
            str(REPO / "eval_results" / "issue_1310" / "recap"),
            "--skip-audit",
        ],
        phase="p1_pairs_base_attribute",
    )
    return {k: _gate_counts(k) for k in ("base", "instruct")}


def phase_capture(device: str | None) -> None:
    """Capture 28-layer span summaries per model into the _recap store."""
    env_note = f"(device={device})" if device else "(whole-node allocation)"
    print(f"[phase=p2_capture] span-summary capture {env_note}", flush=True)
    for model_kind in ("base", "instruct"):
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/issue1310_extract_store.py",
            "--model",
            model_kind,
            "--data-dir",
            str(DATA_DIR),
            "--store-subdir",
            STORE_SUBDIR,
            "--flavor",
            "perturn",
            "--resume",
            "--equivalence-check",
        ]
        _sh(cmd, phase=f"p2_capture_{model_kind}")


def phase_upload() -> dict:
    """One upload_folder commit per model dir + EXACT-SET verify (pre-teardown)."""
    print("[phase=p3_upload] store upload + exact-set verify", flush=True)
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    out: dict[str, dict] = {}
    for model_kind in ("base", "instruct"):
        local = DATA_DIR / STORE_SUBDIR / model_kind
        files = sorted(p for p in local.rglob("*") if p.is_file())
        assert files, f"no store files under {local}"
        path_in_repo = f"{STORE_PREFIX}/{model_kind}"
        hub._upload(
            local,
            DATA_REPO,
            "dataset",
            path_in_repo,
            raise_on_error=True,
        )
        expected = [f"{path_in_repo}/{p.relative_to(local).as_posix()}" for p in files]
        missing = hub.verify_repo_paths_uploaded(
            api, DATA_REPO, expected, path_in_repo=path_in_repo
        )
        total_gb = sum(p.stat().st_size for p in files) / 1e9
        print(
            f"[upload] {model_kind}: {len(files)} files ({total_gb:.2f} GB) -> "
            f"{path_in_repo}; missing={len(missing)}",
            flush=True,
        )
        if missing:
            raise RuntimeError(f"{model_kind} upload verify FAILED, missing: {missing[:8]}")
        out[model_kind] = {
            "path_in_repo": path_in_repo,
            "n_files": len(files),
            "total_gb": round(total_gb, 3),
            "verify": "exact-set PASS",
        }
    return out


def main() -> int:
    t0 = time.time()
    device = _first_allocated_device()
    if device is not None:
        # Narrow to ONE device of the pre-set allocation; never an absolute index.
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"[recap] narrowed CUDA_VISIBLE_DEVICES to allocated device {device}", flush=True)
    phase_stage()
    gates = phase_pairs()
    phase_capture(device)
    uploads = phase_upload()
    payload = {
        "issue": 1310,
        "round": "nd-estimator-audit-recapture",
        "store_prefix": STORE_PREFIX,
        "pair_count_gates": gates,
        "uploads": uploads,
        "elapsed_s": round(time.time() - t0, 1),
    }
    try:
        SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        SENTINEL.write_text(json.dumps(payload, indent=1))
        print(f"[recap] wrote sentinel {SENTINEL}", flush=True)
    except OSError as exc:  # non-/workspace lane: the stdout payload is the record
        print(f"[recap] sentinel unwritable ({exc}); payload follows", flush=True)
    print("[recap] PAYLOAD " + json.dumps(payload), flush=True)
    print(f"[recap] complete in {payload['elapsed_s']}s", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize race (#1689)
