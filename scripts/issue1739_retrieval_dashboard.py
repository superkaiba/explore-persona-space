"""Build the read-only example browser from accepted, frozen task-1739 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra
import numpy as np
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.issue1739_covariance_ablation import sha256, write_json
from scripts.issue1739_large_pool_risk import METHODS, canonical_pool, ranked_top
from scripts.issue1739_luna_analysis import load_labels, selected_sets, verify_response_sources
from scripts.issue1739_luna_checkpoint import validate_final
from scripts.issue1739_luna_judging import BEHAVIORS, jsonl

DEFAULT_SOURCE = Path.home() / ".local/share/eps/issue1739-large-pool-risk-20260917/outputs"
DEFAULT_ANNOTATIONS = Path.home() / ".local/share/eps/annotation_run_20260917"
RESULT_DIR = ROOT / "eval_results/issue_1739/large_pool_risk_20260917"
DEFAULT_OUTPUT = ROOT / "dashboard/public/tasks/1739/behavior-explorer.html"
TEMPLATE = ROOT / "scripts/templates/issue1739_retrieval_dashboard.html"


def verified_ranks(source, records, summary, memberships):
    """Reconstruct full-pool ranks, using the exact producer's pool and tie breaking."""
    config = json.loads((source / "run_config.json").read_text())
    retrieval, bank = Path(config["retrieval"]), Path(config["map_inputs"])
    score_path = retrieval / "generic_scores.npy"
    if sha256(score_path) != summary["pool"]["source_score_sha256"]:
        raise ValueError("Ranking source differs from analyzed pool")
    for name, digest in summary["pool"]["map_input_sha256"].items():
        if sha256(bank / name) != digest:
            raise ValueError(f"Changed map input: {name}")
    rows = np.load(retrieval / "generic_score_rows.npy")
    np.testing.assert_array_equal(rows, np.load(bank / "split.npz")["train"])
    cis = np.load(bank / "ci.npy")
    hashes = np.load(bank / "map_prompt_hashes.npz")["normalized_sha256"]
    scores = np.load(score_path, mmap_mode="r")
    pool = canonical_pool(rows, cis, hashes)
    if len(pool) != summary["pool"]["n_unique_candidate_prompts"]:
        raise ValueError("Candidate population changed")
    if scores.shape != (len(rows), 2, 12):
        raise ValueError("Unexpected score dimensions")
    selected_rows = np.array([r["score_row"] for r in records])
    for r in records:
        if r["map_row"] != rows[r["score_row"]] or r["ci"] != cis[r["map_row"]]:
            raise ValueError("Transcript does not match score row")
    output = {r["ci"]: {b: {} for b in BEHAVIORS} for r in records}
    for bi, behavior in enumerate(BEHAVIORS):
        for method, (metric, column) in METHODS.items():
            values = scores[:, metric, bi * 4 + column]
            order = ranked_top(values, pool, rows, len(pool))
            ranks = np.zeros(len(rows), dtype=np.int32)
            ranks[order] = np.arange(1, len(pool) + 1)
            if (ranks[selected_rows] == 0).any():
                raise ValueError("Selected context outside canonical pool")
            for r in records:
                i = r["score_row"]
                output[r["ci"]][behavior][method] = {
                    "rank": int(ranks[i]),
                    "score": float(values[i]),
                }
    for m in memberships:
        if m["method"] == "random":
            continue
        behavior = "harmful_compliance" if m["behavior"] == "evil" else m["behavior"]
        actual = output[m["ci"]][behavior][m["method"]]
        if actual["rank"] != m["rank"] or actual["score"] != m["score"]:
            raise ValueError("Reconstructed ranks differ from frozen selection")
    return output


def build_payload(source, annotations, summary_path):
    accepted_hashes = validate_final(annotations, summary_path)
    manifest, labels, hashes = load_labels(annotations)
    if accepted_hashes != hashes:
        raise ValueError("Accepted annotation snapshot changed")
    summary = json.loads(summary_path.read_text())
    membership_path = source / "selection/memberships.json"
    if sha256(membership_path) != manifest["source_membership_sha256"]:
        raise ValueError("Selection changed")
    if sha256(source / "selection/summary.json") != summary["preparation_summary_sha256"]:
        raise ValueError("Population provenance changed")
    verify_response_sources(source, manifest)
    memberships = json.loads(membership_path.read_text())
    records = [
        r
        for p in sorted((source / "responses").glob("selected_responses_*.jsonl"))
        for r in jsonl(p)
    ]
    if len(records) != summary["pool"]["n_selected_unique_contexts"]:
        raise ValueError("Incomplete transcript coverage")
    if len({r["ci"] for r in records}) != len(records):
        raise ValueError("Duplicate transcript IDs")
    ranks = verified_ranks(source, records, summary, memberships)
    sets = {b: selected_sets(memberships, b) for b in BEHAVIORS}
    random_order = {
        m["ci"]: i + 1 for i, m in enumerate(m for m in memberships if m["method"] == "random")
    }
    for b in BEHAVIORS:
        for name, ids in sets[b].items():
            items = [labels[b]["primary"][ci] for ci in ids]
            cell = summary["behaviors"][b]["cells"][name]
            observed = [r for r in items if r["status"] == "scored"]
            if (len(items), len(observed), sum(r["score"] >= 50 for r in observed)) != (
                cell["n"],
                cell["n_scored"],
                cell["n_positive"],
            ):
                raise ValueError(f"Displayed cohort disagrees with analysis: {b}/{name}")
    result = []
    correction = json.loads((summary_path.parent / "post_publication_correction.json").read_text())
    for record in sorted(records, key=lambda r: r["ci"]):
        ci = record["ci"]
        if hashlib.sha256(record["response"].encode()).hexdigest() != record["response_sha256"]:
            raise ValueError("Transcript response hash mismatch")
        judgments = {b: labels[b]["primary"][ci] for b in BEHAVIORS if ci in labels[b]["primary"]}
        result.append(
            dict(
                ci=ci,
                prompt=record["prompt"],
                response=record["response"],
                response_sha256=record["response_sha256"],
                ranks=ranks[ci],
                random_order=random_order.get(ci),
                judgments=judgments,
                repeat={b: labels[b]["audit"][ci] for b in BEHAVIORS if ci in labels[b]["audit"]},
                corrected=any(j["id"] == correction["id"] for j in judgments.values()),
            )
        )
    payload = dict(
        version=1,
        pool=summary["pool"],
        behaviors=summary["behaviors"],
        records=result,
        interpretation=summary["interpretation"],
        intervals=summary["intervals"],
        rubrics={b: (annotations / b / "rubric.md").read_text() for b in BEHAVIORS},
        correction=correction,
        summary_sha256=sha256(summary_path),
        quality_acceptance_sha256=summary["quality_acceptance_sha256"],
        figure=json.loads((summary_path.parent / "publication.json").read_text())["url"],
    )
    if validate_final(annotations, summary_path) != accepted_hashes:
        raise ValueError("Snapshot changed during dashboard build")
    return payload


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    source = Path(cfg.get("source", str(DEFAULT_SOURCE)))
    annotations = Path(cfg.get("annotations", str(DEFAULT_ANNOTATIONS)))
    summary_path = Path(cfg.get("summary", str(RESULT_DIR / "luna_results.json")))
    output = Path(cfg.get("output", str(DEFAULT_OUTPUT)))
    payload = build_payload(source, annotations, summary_path)
    template = TEMPLATE.read_text()
    if template.count("__DASHBOARD_DATA__") != 1:
        raise ValueError("Expected exactly one data placeholder")
    # Embedded transcripts are inert JSON. Never let corpus text terminate the script tag.
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(template.replace("__DASHBOARD_DATA__", encoded))
    temporary.replace(output)
    receipt = dict(
        built_at=time.time(),
        html_sha256=sha256(output),
        summary_sha256=sha256(summary_path),
        template_sha256=sha256(TEMPLATE),
        builder_sha256=sha256(Path(__file__)),
        n_transcripts=len(payload["records"]),
        n_primary=sum(len(r["judgments"]) for r in payload["records"]),
        n_repeat=sum(len(r["repeat"]) for r in payload["records"]),
        full_pool_ranks_verified_against_frozen_top200=True,
        every_cohort_verified_against_current_summary=True,
    )
    write_json(RESULT_DIR / "dashboard_build.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
