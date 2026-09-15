"""Audit and repair the single documented #825 runtime source-inspection incident.

A comment inserted above loaded functions shifted source line numbers while two
workers ran. Their score fingerprint read the preceding function's text. This
repair accepts only that exact known hash variant and preserves original JSON.
It changes no numerical artifact, index, coefficient, count, or metric.
"""

# ruff: noqa: E402
from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import issue825_turn_bias_scale as driver
from explore_persona_space.analysis import mapping_baselines, turn_transfer_calibration

PRE_FIX_COMMIT = "ca9d5dc5342"
COMMENT = "                # HUB_VERIFY_RETRY_EXEMPT: retry_transient reopens and consumes the complete paginated listing.\n"


def main():
    """Repair only the expected 120 receipts after proving the exact hash error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{PRE_FIX_COMMIT}:scripts/issue825_turn_bias_scale.py"],
        cwd=driver.REPO,
        text=True,
    )
    if source.count(COMMENT) != 1:
        raise RuntimeError("the exact comment-only edit was not found")
    before_comment = source.replace(COMMENT, "", 1)
    executable = ast.dump(ast.parse(source), include_attributes=False)
    if ast.dump(ast.parse(before_comment), include_attributes=False) != executable:
        raise RuntimeError("the historical edit changed executable code")
    functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    wrong_source = ast.get_source_segment(source, functions["score_fingerprint"]) + "\n"
    right_source = ast.get_source_segment(source, functions["score_target"]) + "\n"
    if right_source != driver.SCORE_TARGET_SOURCE:
        raise RuntimeError("scoring function differs from the execution-era source")
    planned = []
    originals = {}
    for model, expected_count in (("instruct", 55), ("pretrained", 65)):
        source_map = json.loads((args.out / "maps" / f"{model}_source3.json").read_text())
        if driver.sha(source_map["map_file"]) != source_map["map_sha256"]:
            raise RuntimeError("source-map bytes changed")
        payload = {
            "source_fingerprint": source_map["fingerprint"],
            "source_map_sha256": source_map["map_sha256"],
            "scoring_code": wrong_source,
            "baselines_sha256": driver.sha(mapping_baselines.__file__),
            "numerical_sha256": driver.sha(turn_transfer_calibration.__file__),
        }
        wrong = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        right = driver.score_fingerprint(source_map)
        changed = []
        for path in sorted((args.out / "folds").glob(f"{model}_source3_target*.json")):
            row = json.loads(path.read_text())
            if row["fingerprint"] == right:
                continue
            if row["fingerprint"] != wrong or row["source_map_sha256"] != source_map["map_sha256"]:
                raise RuntimeError(f"unrecognized fingerprint mismatch: {path}")
            if driver.sha(row["prediction_file"]) != row["prediction_sha256"]:
                raise RuntimeError(f"prediction bytes changed: {path}")
            with np.load(row["prediction_file"], allow_pickle=False) as arrays:
                for method in driver.METHODS:
                    errors = arrays[f"sse_{method}"]
                    if len(errors) != row["n_test"] or not np.isclose(
                        errors.sum(),
                        row["metrics"][method]["sse"],
                        rtol=1e-14,
                        atol=0,
                    ):
                        raise RuntimeError(f"saved errors disagree: {path} {method}")
            originals[path.name] = row
            repaired = row | {
                "fingerprint": right,
                "fingerprint_repair": {
                    "original_fingerprint": wrong,
                    "reason": "Runtime inspect.getsource resolved the preceding function after a comment shifted source lines; executable scoring AST and numerical artifacts unchanged.",
                    "audit": "provenance_repair/audit.json",
                },
            }
            changed.append((path, repaired))
        if len(changed) != expected_count:
            raise RuntimeError(
                f"expected {expected_count} affected {model} receipts, found {len(changed)}"
            )
        planned.extend(changed)
    # Preserve all original metadata before changing any receipt.
    audit_dir = args.out / "provenance_repair"
    if audit_dir.exists():
        raise RuntimeError("repair audit already exists; never overwrite original evidence")
    driver.atomic_json(audit_dir / "original_fold_receipts.json", originals)
    audit = {
        "status": "verified_before_repair",
        "count": len(planned),
        "pre_fix_commit": PRE_FIX_COMMIT,
        "execution_ast_sha256": hashlib.sha256(executable.encode()).hexdigest(),
        "historical_edit": "One HUB_VERIFY_RETRY_EXEMPT comment inserted in stage(); no AST change.",
        "correct_score_source_sha256": hashlib.sha256(right_source.encode()).hexdigest(),
        "mistaken_source_sha256": hashlib.sha256(wrong_source.encode()).hexdigest(),
        "changed_fields": ["fingerprint", "fingerprint_repair"],
        "numerical_artifacts_changed": False,
        "files": [p.name for p, _ in planned],
        "repaired_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    driver.atomic_json(audit_dir / "audit.json", audit)
    for path, repaired in planned:
        driver.atomic_json(path, repaired)
    driver.atomic_json(audit_dir / "audit.json", audit | {"status": "complete"})
    print(
        f"Audited and repaired {len(planned)} metadata fingerprints; all original receipts preserved."
    )


if __name__ == "__main__":
    main()
