"""Issue #2333 — plan §8 mediation-donor divergence manipulation check.

Plan §8 row "Mediation donors degenerate": report the token-level divergence
rate (mediated donor opening vs the base context's own greedy floor opening)
per pair set, per k. A mediated donor whose first-k tokens coincide with what
the unpatched base context would greedily emit anyway carries no manipulation
at that k — the pair stays valid, but its mediation arm legitimately reads
near floor, so the rate bounds how much of the k=1..3 recovery shape can come
from growing donor-vs-base divergence rather than growing causal leverage.

Zero-GPU: both sides are committed greedy captures at HF revision
ab9e72d55e (donors_med.pt = base context A with the ce patch armed;
donors_bstart.pt = context B unhooked). Because decoding is greedy, the
floor opening of context X is the bstart capture of any pair whose donor
context is X; every S1 base context appears as a donor context somewhere
(asserted below), while S2 bare contexts never do (coverage reported).

Output: eval_results/issue_2333/f_metrics/mediation_divergence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch import

import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK94
from explore_persona_space.experiments.issue2162 import bank2162 as BANK2162
from explore_persona_space.experiments.issue2333 import constants as C

REVISION = "ab9e72d55e980ab0cd081161bdce832e5e1710c0"
REPO = "superkaiba1/explore-persona-space-data"
SEPARATION_BAR = 0.5


def load_donor_tokens(tag: str, scheme: str) -> dict[str, list[int]]:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    p = retry_transient(
        lambda: hf_hub_download(
            REPO,
            f"issue2333_snowball/{tag}/donors/donors_{scheme}.pt",
            repo_type="dataset",
            revision=REVISION,
            local_dir=f"/tmp/i2333_donors_{tag}",
        ),
        what=f"stage donors_{scheme} ({tag})",
    )
    recs = torch.load(p, map_location="cpu", weights_only=False)
    return {pid: list(rec["token_ids"]) for pid, rec in recs.items()}


def survivors(tag: str, repo_root: Path) -> set[str]:
    """Anchor-surviving pair ids (|separation| >= 0.5), matching analysis."""
    keep: set[str] = set()
    fc = repo_root / f"eval_results/issue_2333/f_metrics/{tag}/f_cells.jsonl"
    seen: dict[str, float | None] = {}
    with fc.open() as f:
        for line in f:
            r = json.loads(line)
            seen.setdefault(r["pair_id"], r.get("separation"))
    for pid, sep in seen.items():
        if sep is not None and abs(sep) >= SEPARATION_BAR:
            keep.add(pid)
    return keep


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    s1 = [p for p in BANK2162.build_pairs() if p.cell in C.S1_CELLS]
    s2 = [p for p in BANK94.build_pairs() if p.setting == "matched_query"]
    assert len(s1) == 180 and len(s2) == 15, (len(s1), len(s2))
    pairs = {"s1": s1, "s2": s2}

    out: dict = {"revision": REVISION, "per_model": {}}
    for tag in ("q25", "q35"):
        med = load_donor_tokens(tag, "med")
        bstart = load_donor_tokens(tag, "bstart")
        # floor opening of context X = bstart tokens of any pair with b == X
        floor: dict[str, list[int]] = {}
        for pset in pairs.values():
            for p in pset:
                toks = bstart.get(p.pair_id)
                if toks is None:
                    continue
                prev = floor.get(p.b)
                if prev is not None and prev != toks:
                    raise AssertionError(
                        f"greedy floor mismatch for context {p.b}: {prev} vs {toks}"
                    )
                floor[p.b] = toks
        surv = survivors(tag, repo_root)
        model_out: dict = {}
        for set_name, pset in pairs.items():
            covered = [p for p in pset if p.pair_id in med and p.a in floor]
            rows = {}
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
                rows[basis] = per_k
            rows["n_pairs_total"] = len(pset)
            rows["n_covered"] = len(covered)
            rows["uncovered_pair_ids"] = sorted(
                p.pair_id for p in pset if not (p.pair_id in med and p.a in floor)
            )
            model_out[set_name] = rows
        out["per_model"][tag] = model_out

    dst = repo_root / "eval_results/issue_2333/f_metrics/mediation_divergence.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
