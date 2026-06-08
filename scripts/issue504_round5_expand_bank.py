#!/usr/bin/env python3
# ruff: noqa: E501
"""Task #504 round-5 Phase 3 — expand the persona bank with far-from-villain candidates.

Reads:
  - data/issue_504_round5/older_pool_cos_to_villain.json  (from Phase 2 probe)
  - data/issue_472/persona_bank.json  (existing 60-bank)

Writes (Option-1 / Option-2 paths — script picks based on the L20 distribution):
  - data/issue_472/persona_bank_v504.json  (the EXPANDED bank, v504 schema, ~60+30 = ~90 personas)
  - eval_results/issue_504/round5_expansion_decision.json  (decision + plan target bands)

Decision logic (executed AT RUN-TIME, prints which Option fired):

  Option 1 — full expansion → bands [-0.20, 0.10, 0.40, 0.70] at L20.
      Fires when the older-pool's L20 distribution provides:
        - ≥3 candidates below cos < 0.0
        - ≥3 candidates in [0.0, 0.30]
        - ≥3 candidates in [0.30, 0.55]
        - ≥3 candidates in [0.55, 0.80]
      Then we pick ~30 personas best-distributed across these 4 bands and
      write persona_bank_v504.json with the new bands.

  Option 2 — partial expansion + plan-target adjust → bands shifted to
      [0.30, 0.50, 0.70, 0.85] at L20.
      Fires when Option 1 fails BUT the older-pool's L20 range is e.g. [0.30, 0.90].
      Pick ~30 personas across the shifted bands and write persona_bank_v504.json
      with the new (compressed) bands.

  Option 3 — cosine framing dead → write a decision JSON marking
      `option=3` and `bank_expanded=false`. The orchestrator surfaces this to
      Thomas.

The expanded bank's schema matches #472's `i472_v1` schema but with
`schema_version='i472_v504r5'` so consumers can switch on it deliberately.
The villain anchor + all 60 base bank personas are preserved verbatim;
only the appended new personas come from the older pool.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("i504_r5_expand")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXISTING_BANK = REPO_ROOT / "data" / "issue_472" / "persona_bank.json"
PROBE = REPO_ROOT / "data" / "issue_504_round5" / "older_pool_cos_to_villain.json"
EXPANDED_BANK = REPO_ROOT / "data" / "issue_472" / "persona_bank_v504.json"
DECISION = REPO_ROOT / "eval_results" / "issue_504" / "round5_expansion_decision.json"
EXPANDED_BANK.parent.mkdir(parents=True, exist_ok=True)
DECISION.parent.mkdir(parents=True, exist_ok=True)

# Option 1 target bands (centers, used for L20).
OPT1_BANDS = (-0.20, 0.10, 0.40, 0.70)
# Option 1 bin edges at L20.
OPT1_EDGES = (-1.01, 0.00, 0.30, 0.55, 0.80, 1.01)
# Option 2 fallback bands.
OPT2_BANDS = (0.30, 0.50, 0.70, 0.85)
OPT2_EDGES = (-1.01, 0.40, 0.60, 0.80, 0.90, 1.01)
# Per-band target picks.
PER_BAND_TARGET = 8


def _content_hash(payload: dict[str, str]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=REPO_ROOT
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _bin_pool_by_l20(per_persona: list[dict], edges: tuple[float, ...]) -> list[list[dict]]:
    """Bin pool by L20 cosine; bins are (edges[i], edges[i+1]]."""
    bins: list[list[dict]] = [[] for _ in range(len(edges) - 1)]
    for p in per_persona:
        c = p.get("cos_to_villain_L20")
        if c is None:
            continue
        for i in range(len(edges) - 1):
            if edges[i] < c <= edges[i + 1]:
                bins[i].append(p)
                break
    return bins


def _pick_evenly_by_band_center(bin_entries: list[dict], target: int, center: float) -> list[dict]:
    """Pick `target` entries whose cos-to-villain-L20 is closest to `center`."""
    sorted_entries = sorted(bin_entries, key=lambda p: abs(p["cos_to_villain_L20"] - center))
    return sorted_entries[:target]


def _build_bank(bank_existing: dict, picks: list[dict]) -> dict:
    """Build the v504 expanded bank payload.

    Preserves the 60-bank personas verbatim, appends `picks` at the end. Each
    `pick` carries `name` + `system_prompt`. The bank dict is name → prompt.
    """
    bank_dict: dict[str, str] = dict(bank_existing["personas"])
    for p in picks:
        if p["name"] in bank_dict:
            log.warning("pick %r already in 60-bank; skipping append", p["name"])
            continue
        bank_dict[p["name"]] = p["system_prompt"]
    return {
        "schema_version": "i472_v504r5",
        "source_persona": "villain",
        "n_base": bank_existing["n_total"],
        "n_added": len(bank_dict) - bank_existing["n_total"],
        "n_total": len(bank_dict),
        "personas": bank_dict,
        "content_hash": _content_hash(bank_dict),
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "parent_bank_hash": bank_existing.get("content_hash"),
    }


def main() -> None:
    if not EXISTING_BANK.exists():
        raise FileNotFoundError(f"existing 60-bank missing at {EXISTING_BANK}")
    if not PROBE.exists():
        raise FileNotFoundError(
            f"Phase 2 probe results missing at {PROBE}; run "
            f"scripts/issue504_round5_probe_older_pool.py on the pod first."
        )

    bank_existing = json.loads(EXISTING_BANK.read_text())
    probe = json.loads(PROBE.read_text())
    per_persona = probe["per_persona"]
    log.info("loaded probe: %d personas, layers=%s", len(per_persona), probe["layers"])

    l20_vals = [
        p["cos_to_villain_L20"] for p in per_persona if p.get("cos_to_villain_L20") is not None
    ]
    l20_min, l20_max = min(l20_vals), max(l20_vals)
    log.info("L20 cos-to-villain range: [%.3f, %.3f]", l20_min, l20_max)

    # Try Option 1 bins
    bins_opt1 = _bin_pool_by_l20(per_persona, OPT1_EDGES)
    band_counts_opt1 = [len(b) for b in bins_opt1]
    log.info("Option 1 bin counts (edges=%s): %s", OPT1_EDGES, band_counts_opt1)
    # Option 1 needs: bin[0] (cos < 0.0) >= 3 AND each of bins[1..3] >= 3 (4 bands, >=3 each).
    opt1_ok = (
        band_counts_opt1[0] >= 3
        and band_counts_opt1[1] >= 3
        and band_counts_opt1[2] >= 3
        and band_counts_opt1[3] >= 3
    )

    # If Option 1 fails, try Option 2 (compressed bands).
    bins_opt2 = _bin_pool_by_l20(per_persona, OPT2_EDGES)
    band_counts_opt2 = [len(b) for b in bins_opt2]
    log.info("Option 2 bin counts (edges=%s): %s", OPT2_EDGES, band_counts_opt2)
    opt2_ok = all(c >= 3 for c in band_counts_opt2[:4])

    decision: dict = {
        "schema_version": "i504_round5_decision_v1",
        "n_pool": len(per_persona),
        "l20_min": l20_min,
        "l20_max": l20_max,
        "opt1_bin_counts": band_counts_opt1,
        "opt1_edges": list(OPT1_EDGES),
        "opt1_ok": opt1_ok,
        "opt2_bin_counts": band_counts_opt2,
        "opt2_edges": list(OPT2_EDGES),
        "opt2_ok": opt2_ok,
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }

    if opt1_ok:
        log.info("DECISION: Option 1 — full expansion with bands %s", OPT1_BANDS)
        picks: list[dict] = []
        # First 4 bands (cos < 0.0, 0.0-0.30, 0.30-0.55, 0.55-0.80) → centers from OPT1_BANDS
        for band_center, band_pool in zip(OPT1_BANDS, bins_opt1[:4], strict=True):
            picked = _pick_evenly_by_band_center(band_pool, PER_BAND_TARGET, band_center)
            picks.extend(picked)
            log.info(
                "  band center=%.2f: picked %d / %d available; range=[%.3f, %.3f]",
                band_center,
                len(picked),
                len(band_pool),
                min((p["cos_to_villain_L20"] for p in picked), default=float("nan")),
                max((p["cos_to_villain_L20"] for p in picked), default=float("nan")),
            )
        bank_payload = _build_bank(bank_existing, picks)
        EXPANDED_BANK.write_text(json.dumps(bank_payload, indent=2, ensure_ascii=False))
        log.info(
            "wrote %s (n_total=%d, +%d added)",
            EXPANDED_BANK,
            bank_payload["n_total"],
            bank_payload["n_added"],
        )
        decision["option"] = 1
        decision["bands_used"] = list(OPT1_BANDS)
        decision["picks"] = [{"name": p["name"], "cos_L20": p["cos_to_villain_L20"]} for p in picks]
        decision["expanded_bank_path"] = str(EXPANDED_BANK)
        decision["expanded_bank_n_total"] = bank_payload["n_total"]

    elif opt2_ok:
        log.info("DECISION: Option 2 — partial expansion with shifted bands %s", OPT2_BANDS)
        picks = []
        for band_center, band_pool in zip(OPT2_BANDS, bins_opt2[:4], strict=True):
            picked = _pick_evenly_by_band_center(band_pool, PER_BAND_TARGET, band_center)
            picks.extend(picked)
            log.info(
                "  band center=%.2f: picked %d / %d available; range=[%.3f, %.3f]",
                band_center,
                len(picked),
                len(band_pool),
                min((p["cos_to_villain_L20"] for p in picked), default=float("nan")),
                max((p["cos_to_villain_L20"] for p in picked), default=float("nan")),
            )
        bank_payload = _build_bank(bank_existing, picks)
        EXPANDED_BANK.write_text(json.dumps(bank_payload, indent=2, ensure_ascii=False))
        log.info(
            "wrote %s (n_total=%d, +%d added)",
            EXPANDED_BANK,
            bank_payload["n_total"],
            bank_payload["n_added"],
        )
        decision["option"] = 2
        decision["bands_used"] = list(OPT2_BANDS)
        decision["picks"] = [{"name": p["name"], "cos_L20": p["cos_to_villain_L20"]} for p in picks]
        decision["expanded_bank_path"] = str(EXPANDED_BANK)
        decision["expanded_bank_n_total"] = bank_payload["n_total"]
        decision["scope_note"] = (
            "compressed bands relative to plan §4.3 [-0.20, 0.10, 0.40, 0.70]; "
            "Phase 0.5 Gate A's 0.02-nat movement floor must be re-verified at the new band centers."
        )
    else:
        log.warning(
            "DECISION: Option 3 — cosine framing dead. Older-pool L20 range "
            "[%.3f, %.3f]; OPT1 bin counts %s; OPT2 bin counts %s. No bank expansion.",
            l20_min,
            l20_max,
            band_counts_opt1,
            band_counts_opt2,
        )
        decision["option"] = 3
        decision["bank_expanded"] = False
        decision["bands_used"] = None
        decision["recommendation"] = (
            "Metric pivot needed (Euclidean / JS / rank-based); cosine cannot "
            "support plan §4.3 [-0.20, 0.10, 0.40, 0.70] bands at any layer."
        )

    DECISION.write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    log.info("wrote %s", DECISION)


if __name__ == "__main__":
    main()
