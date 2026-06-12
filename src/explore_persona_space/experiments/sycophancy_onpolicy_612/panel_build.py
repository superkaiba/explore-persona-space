"""Task #612 P1 — panel construction: candidates -> layer-20 centroids -> cosines.

Candidate registry (~52): the 24-roster (``EVAL_PERSONAS_24``) + the #591
accepted synthesized personas (prompts read from the prefetched
``twin_validation.json``) + the 13 NEW one-line candidates
(``NEW_CANDIDATES_612``). Pipeline (ported from #591 phase_a @ 24da9cf +
#411 extend_centroids @ 90656ef):

1. Extract layer-20 last-token centroids for ALL candidates on
   ``EVAL_QUESTIONS_20`` (the canonical extraction contract).
2. **Bank-parity assert (kill K1a):** re-extracted cosines must reproduce the
   frozen join (``predictor_comparison.json``) within ±0.01 on the 3 #591
   pairs — raise on any failure, never proceed to paid eval.
3. Compute every candidate's cosine to each of the 4 sources + assign the
   plan's 6 cosine bins.
4. Write ``panel_candidates.json`` (name -> prompt/provenance/cosines/bins)
   + persist the centroid tensor; the dispatcher uploads both.

GPU: 1x, ~minutes. Runs inside the dispatcher under the ``panel:build:0`` cell.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    BANK_PARITY_PAIRS,
    BANK_PARITY_TOL,
    BASE_MODEL,
    COSINE_BINS,
    FROZEN_JOIN_RELPATH,
    MANDATORY_PANEL,
    NEW_CANDIDATES_612,
    SOURCES,
    repo_root_from_module,
)

log = logging.getLogger("issue_612.panel_build")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def load_i591_accepted(twin_validation_path: Path) -> dict[str, str]:
    """Accepted #591 synthesized personas -> {name: prompt} from twin_validation.json."""
    payload = json.loads(twin_validation_path.read_text())
    accepted = payload.get("accepted")
    if not accepted:
        raise ValueError(f"{twin_validation_path}: no 'accepted' personas")
    return {name: rec["prompt"] for name, rec in accepted.items()}


def build_candidate_registry(twin_validation_path: Path) -> dict[str, dict]:
    """All candidates -> {name: {prompt, provenance}}. Name collisions fail loud."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    registry: dict[str, dict] = {
        name: {"prompt": prompt, "provenance": "roster_24"}
        for name, prompt in EVAL_PERSONAS_24.items()
    }
    i591 = load_i591_accepted(twin_validation_path)
    for name, prompt in i591.items():
        if name in registry:
            raise ValueError(f"#591 persona {name!r} collides with the 24-roster")
        registry[name] = {"prompt": prompt, "provenance": "i591_twin"}
    for name, prompt in NEW_CANDIDATES_612.items():
        if name in registry:
            raise ValueError(f"new candidate {name!r} collides with an existing persona")
        registry[name] = {"prompt": prompt, "provenance": "new_612"}
    missing_mandatory = [m for m in MANDATORY_PANEL if m not in registry]
    if missing_mandatory:
        raise AssertionError(f"mandatory panel personas missing from registry: {missing_mandatory}")
    log.info(
        "candidate registry: %d personas (%d roster + %d i591 + %d new)",
        len(registry),
        24,
        len(i591),
        len(NEW_CANDIDATES_612),
    )
    return registry


def cosine_bin(cos: float) -> str | None:
    """Map a cosine to the plan's bin label, or None below 0.70 / self-identical."""
    for lo, hi in COSINE_BINS:
        if lo <= cos < hi:
            return f"[{lo},{hi})"
    return None


def build_panel_candidates(
    *,
    data_root: Path,
    out_dir: Path,
    device: str = "cuda:0",
) -> Path:
    """Run the P1 pipeline; returns the path of panel_candidates.json."""
    import torch
    import torch.nn.functional as F

    from explore_persona_space.analysis.representation_shift import extract_centroids
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    registry = build_candidate_registry(data_root / "i591" / "twin_validation.json")
    extraction = {name: rec["prompt"] for name, rec in registry.items()}

    log.info("extracting layer-20 centroids for %d personas ...", len(extraction))
    centroids_by_layer, names = extract_centroids(
        BASE_MODEL,
        extraction,
        questions=EVAL_QUESTIONS_20,
        layers=[20],
        device=device,
        dtype=torch.bfloat16,
    )
    cents = centroids_by_layer[20].to(torch.float32)
    idx = {n: i for i, n in enumerate(names)}

    def cos(a: str, b: str) -> float:
        return float(
            F.cosine_similarity(cents[idx[a]].unsqueeze(0), cents[idx[b]].unsqueeze(0)).item()
        )

    # --- bank-parity assert (kill criterion K1a; #591 pattern verbatim) ---
    frozen_join = repo_root_from_module() / FROZEN_JOIN_RELPATH
    frozen = json.loads(frozen_join.read_text())["cells"]
    frozen_cos = {(c["source"], c["bystander"]): c["cosine_l20_baseline"] for c in frozen}
    parity_report = []
    for a, b in BANK_PARITY_PAIRS:
        got, ref = cos(a, b), frozen_cos[(a, b)]
        ok = abs(got - ref) <= BANK_PARITY_TOL
        parity_report.append({"pair": [a, b], "re_extracted": got, "frozen": ref, "pass": ok})
        log.info(
            "bank-parity %s-%s: re-extracted %.4f vs frozen %.4f (%s)",
            a,
            b,
            got,
            ref,
            "PASS" if ok else "FAIL",
        )
    failures = [p for p in parity_report if not p["pass"]]
    if failures:
        raise RuntimeError(
            f"KILL CRITERION K1a: bank-parity assert failed {failures} — centroid recipe "
            f"drift; stop before any paid eval (plan §4 P1)."
        )

    # --- per-source cosines + bins ---
    for name, rec in registry.items():
        cos_to = {s: cos(name, s) for s in SOURCES}
        rec["cosines"] = cos_to
        rec["bin_by_source"] = {s: cosine_bin(c) for s, c in cos_to.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    cent_path = out_dir / "panel_centroids_layer20.pt"
    torch.save(
        {"centroids": {20: cents}, "persona_names": names, "base_model": BASE_MODEL},
        cent_path,
    )
    payload = {
        "schema_version": 1,
        "candidates": registry,
        "bank_parity": parity_report,
        "bins": [list(b) for b in COSINE_BINS],
        "mandatory_panel": list(MANDATORY_PANEL),
        "metadata": {
            "base_model": BASE_MODEL,
            "recipe": (
                "extract_centroids, EVAL_QUESTIONS_20, last-token, layer 20, bf16 (#411/#591)"
            ),
            # Centering label (persona-distance-metrics rule): this instrument
            # line deliberately uses the RAW gauge — the frozen-join parity
            # assert pins it and the plan's 0.70-1.0 bins are raw-regime.
            "centering": "raw_pairwise_uncentered_bank (frozen #411/#591 gauge; parity-pinned)",
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    out_path = out_dir / "panel_candidates.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("panel_candidates -> %s (%d candidates)", out_path, len(registry))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 P1 — candidate centroids + bank parity + cosines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_612"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/issue_612/panel"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=p1_panel] %(message)s", stream=sys.stdout
    )
    build_panel_candidates(data_root=args.data_root, out_dir=args.out_dir, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
