#!/usr/bin/env python
"""Issue #2333 q35lang round: mediation-donor divergence check (plan v9 §6/§8).

Same binary any-difference read as `issue2333_mediation_divergence.py` (the
parent-legs script), scoped to the `q35_language_snowball` cell set: for each
directed pair, do the first-k token ids of the MEDIATED donor opening (greedy
under the context-end patch) differ from the base context's OWN greedy opening
(its bstart capture — every context serves as a B-side donor under the
directed value cycle, so coverage is the full 72 pairs)? A mediated donor that
collapses into the base opening would make the "steered" arm a disguised
self-prefill; a high divergence rate rules that reading out. Binary read only —
graded donor distance stays unmeasured (parent caveat, held).

Writes eval_results/issue_2333/q35_language_snowball/f_metrics/mediation_divergence.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2333_judge as J33  # noqa: E402

HF_REPO = "superkaiba1/explore-persona-space-data"
SEPARATION_BAR = 0.5
FMETRICS = REPO_ROOT / "eval_results/issue_2333/q35_language_snowball/f_metrics"


def load_donor_tokens(scheme: str, revision: str) -> dict[str, list[int]]:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    p = retry_transient(
        lambda: hf_hub_download(
            HF_REPO,
            f"issue2333_snowball/q35_language_snowball/donors/donors_{scheme}.pt",
            repo_type="dataset",
            revision=revision,
            local_dir="/tmp/i2333_donors_q35lang",
        ),
        what=f"stage donors_{scheme} (q35lang)",
    )
    recs = torch.load(p, map_location="cpu", weights_only=False)
    return {pid: list(rec["token_ids"]) for pid, rec in recs.items()}


def main() -> None:
    from huggingface_hub import HfApi

    revision = HfApi().list_repo_commits(HF_REPO, repo_type="dataset")[0].commit_id
    s1_pairs, s2_pairs = J33.build_pair_universe("q35lang")
    assert not s2_pairs and len(s1_pairs) == 72, len(s1_pairs)

    med = load_donor_tokens("med", revision)
    bstart = load_donor_tokens("bstart", revision)
    floor: dict[str, list[int]] = {}
    for p in s1_pairs:
        toks = bstart.get(p.pair_id)
        if toks is None:
            continue
        prev = floor.get(p.b)
        if prev is not None and prev != toks:
            raise AssertionError(f"greedy floor mismatch for context {p.b}")
        floor[p.b] = toks

    surv: set[str] = set()
    seen: dict[str, float | None] = {}
    for line in (FMETRICS / "f_cells.jsonl").open():
        r = json.loads(line)
        seen.setdefault(r["pair_id"], r.get("separation"))
    for pid, sep in seen.items():
        if sep is not None and abs(sep) >= SEPARATION_BAR:
            surv.add(pid)

    covered = [p for p in s1_pairs if p.pair_id in med and p.a in floor]
    out: dict = {"revision": revision, "cell_set": "q35lang", "s1": {}}
    for basis, sel in (
        ("all_pairs", covered),
        ("anchor_survivors", [p for p in covered if p.pair_id in surv]),
    ):
        per_k = {}
        for k in (1, 2, 3):
            div = [p.pair_id for p in sel if med[p.pair_id][:k] != floor[p.a][:k]]
            per_k[f"k{k}"] = {
                "n": len(sel),
                "n_diverged": len(div),
                "rate": round(len(div) / len(sel), 4) if sel else None,
            }
        out["s1"][basis] = per_k
    out["s1"]["n_pairs_total"] = len(s1_pairs)
    out["s1"]["n_covered"] = len(covered)
    out["s1"]["uncovered_pair_ids"] = sorted(
        p.pair_id for p in s1_pairs if not (p.pair_id in med and p.a in floor)
    )

    dst = FMETRICS / "mediation_divergence.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
