#!/usr/bin/env python3
"""Task #591 e1 — descriptive #390 refusal-OOD supplement (NON-INFERENTIAL).

Joins #390's framing x persona refusal-gate matrix (teach = zelthari_scholar,
4 non-teach contexts x 11 framings; ``eval_results/issue_390/aggregate_long.json``)
against layer-20 centroid cosines from the #411 UNION bank on the Hub.

Explicitly labeled non-inferential (plan #591 v1 §4.1 item 5): the #390
design (1 teach persona, 11 framings, teach/non-teach pass rates) is
incompatible with the 6x23 panel factor analysis — this is a descriptive
side-table only. ``no_system`` has no centroid in the bank; its cosine is
recorded as null.

Output: ``eval_results/issue_591/e1/supplement_390.json``.

Smoke / production (same command — the join is 165 rows):

    uv run python scripts/issue_591/i591_e1_390_supplement.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
UNION_BANK_PT = "issue411_sycophancy_cosine_gradient/eval_results/centroids/centroids_layer20.pt"
UNION_BANK_NAMES = "issue411_sycophancy_cosine_gradient/eval_results/centroids/persona_names.json"
AGG_390 = REPO / "eval_results/issue_390/aggregate_long.json"
OUT_ROOT_DEFAULT = REPO / "eval_results" / "issue_591"
TEACH_PERSONA = "zelthari_scholar"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _union_bank_cosines() -> dict[str, float]:
    """Raw pairwise cosine(teach, persona) from the #411 UNION bank (24 personas).

    Raw pairwise (uncentered) on purpose — line-parity with the #411/#470/#480
    cosine family the e1 factor table uses (plan §11; the #536 centered-bank
    rule applies to NEW predictor lines).
    """
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    pt = hf_hub_download(HF_DATA_REPO, UNION_BANK_PT, repo_type="dataset", token=token)
    names_p = hf_hub_download(HF_DATA_REPO, UNION_BANK_NAMES, repo_type="dataset", token=token)
    names_obj = json.loads(Path(names_p).read_text())
    names = names_obj["persona_names"] if isinstance(names_obj, dict) else names_obj
    bundle = torch.load(pt, weights_only=True, map_location="cpu")
    # The union bank is a bundle dict {centroids: {20: Tensor(24, hidden)},
    # persona_names, base_model, layer} (extend_centroids output shape).
    tensor = bundle["centroids"][20] if isinstance(bundle, dict) else bundle
    tensor = tensor.to(torch.float32)
    if isinstance(bundle, dict):
        assert bundle["persona_names"] == names, "bank-internal names != persona_names.json"
    assert tensor.shape[0] == len(names), (tensor.shape, len(names))
    if TEACH_PERSONA not in names:
        raise KeyError(f"{TEACH_PERSONA} not in union bank names: {sorted(names)}")
    t_idx = names.index(TEACH_PERSONA)
    teach_vec = tensor[t_idx].unsqueeze(0)
    cos = F.cosine_similarity(teach_vec, tensor, dim=1)
    return {n: float(c) for n, c in zip(names, cos.tolist(), strict=True)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#591 e1 descriptive #390 supplement (non-inferential).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    args = parser.parse_args(argv)

    rows = json.loads(AGG_390.read_text())
    assert len(rows) == 165, len(rows)
    cosines = _union_bank_cosines()

    personas = sorted({r["persona"] for r in rows})
    framings = sorted({r["framing_name"] for r in rows})
    matrix: dict[str, dict[str, float]] = {f: {} for f in framings}
    for r in rows:
        # Multiple directions can share (framing, persona); keep positive-direction
        # rows as the gate read and record negatives separately.
        if r["direction"] == "positive":
            matrix[r["framing_name"]][r["persona"]] = r["pass_rate"]
    negative_controls = [
        {k: r[k] for k in ("framing_name", "persona", "is_teach", "pass_rate")}
        for r in rows
        if r["direction"] == "negative"
    ]

    payload = {
        "non_inferential": True,
        "note": (
            "Descriptive supplement ONLY (plan §4.1 item 5): #390's design (1 teach "
            "persona x 11 OOD framings x 4 non-teach contexts, pass-rate DV) is "
            "incompatible with the 6x23 panel factor analysis. No statistics are "
            "computed here; the cosine column is for visual co-reading."
        ),
        "teach_persona": TEACH_PERSONA,
        "cosine_to_teach": {p: cosines.get(p) for p in personas},  # no_system -> null (no centroid)
        "cosine_recipe": (
            "raw pairwise (uncentered) layer-20 last-token centroid cosine, #411 UNION "
            "bank (24 personas) on the Hub — line-parity with the e1 factor table"
        ),
        "pass_rate_matrix_positive_direction": matrix,
        "negative_control_rows": negative_controls,
        "metadata": {
            "source": str(AGG_390),
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    out = args.out_root / "e1" / "supplement_390.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[390-supplement] -> {out} ({len(personas)} personas x {len(framings)} framings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
