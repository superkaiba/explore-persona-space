"""CJK language-intrusion scan over the #1739 DV rollout pool (counting-only).

Analyzer Step 3.7 artifact for task #1739 (interpretation-critic round-1
finding 5): scans every judged-pool rollout completion under
``raw_completions/issue_1739/labeling/<behavior>/`` for CJK-script intrusion
(Qwen-family model under non-CJK evals), aggregates PER-CONTEXT flags, and
joins with the DV (``eval_results/issue_1739/dv_dataset/<behavior>/labeling.json``)
to report Spearman rho(per-context intrusion fraction, DV).

Counting-only by design (trigger-dense corpus: jailbreak prefixes +
unscreened real-user text): completion text is regex-matched in-process and
NEVER printed; the artifact carries counts, correlations, and row flags
(context id -> intruded rollout indices) only.

Output: ``eval_results/issue_1739/intrusion_audit/intrusion_scan.json``.

Run from the issue-1739 worktree root:
    OMP_NUM_THREADS=8 uv run python scripts/issue1739_intrusion_scan.py
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_completions" / "issue_1739" / "labeling"
DV_DIR = ROOT / "eval_results" / "issue_1739" / "dv_dataset"
OUT_DIR = ROOT / "eval_results" / "issue_1739" / "intrusion_audit"
BEHS = ["evil", "hallucination", "sycophancy"]

# Analyzer spec Step 3.7 CJK class (Han + ext-A + compat, kana, hangul).
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")


def _scan_chunk(paths: list[str]) -> list[tuple[str, int, bool]]:
    """Return (context_id, rollout_k, intruded) per rollout file. Text never leaves."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows.append((d["context_id"], int(d["rollout_k"]), bool(CJK_RE.search(d["completion"]))))
    return rows


def scan_behavior(beh: str, workers: int = 8) -> dict:
    files = sorted(str(p) for p in (RAW / beh).glob("*.json") if not p.name.startswith("_"))
    chunks = [files[i : i + 2000] for i in range(0, len(files), 2000)]
    per_ctx: dict[str, list[int]] = {}
    n_total = 0
    n_intruded = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(_scan_chunk, chunks):
            for cid, k, hit in rows:
                n_total += 1
                per_ctx.setdefault(cid, [])
                if hit:
                    n_intruded += 1
                    per_ctx[cid].append(k)

    dv_rows = json.load(open(DV_DIR / beh / "labeling.json"))["rows"]
    dv_by_ctx = {r["context_id"]: r["dv"] for r in dv_rows if r.get("dv") is not None}

    def _rho(ctx_filter) -> dict:
        pairs = [
            (len(per_ctx[c]) / 5.0, dv_by_ctx[c])
            for c in per_ctx
            if c in dv_by_ctx and ctx_filter(c)
        ]
        if len(pairs) < 3:
            return {"rho": None, "n": len(pairs)}
        rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        return {"rho": round(float(rho), 4), "n": len(pairs)}

    flags = {c: sorted(ks) for c, ks in per_ctx.items() if ks}
    return {
        "n_rollouts": n_total,
        "n_intruded_rollouts": n_intruded,
        "pct_rollouts": round(100.0 * n_intruded / max(n_total, 1), 2),
        "n_contexts": len(per_ctx),
        "n_contexts_flagged": len(flags),
        "pct_contexts": round(100.0 * len(flags) / max(len(per_ctx), 1), 2),
        "rho_intrusion_dv_all": _rho(lambda c: True),
        "rho_intrusion_dv_train": _rho(lambda c: "-train-" in c),
        "row_flags": flags,  # flagged contexts only: context_id -> intruded rollout k's
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    out = {
        "task": 1739,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": sha,
        "regex": CJK_RE.pattern,
        "substrate": "raw_completions/issue_1739/labeling/<behavior>/ (judged DV pool, K=5 rollouts/context)",
        "flag_rule": "context flagged iff >=1 of its K rollout completions matches the CJK class",
        "behaviors": {},
    }
    for beh in BEHS:
        t0 = time.time()
        out["behaviors"][beh] = scan_behavior(beh)
        b = out["behaviors"][beh]
        print(
            f"{beh}: {b['n_intruded_rollouts']}/{b['n_rollouts']} rollouts "
            f"({b['pct_rollouts']}%), {b['n_contexts_flagged']}/{b['n_contexts']} contexts "
            f"({b['pct_contexts']}%), rho_all={b['rho_intrusion_dv_all']} "
            f"[{time.time() - t0:.0f}s]"
        )
    path = OUT_DIR / "intrusion_scan.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
