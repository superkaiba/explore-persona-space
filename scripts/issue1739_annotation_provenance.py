"""Require an explicit literal ID/rationale/score triple for every saved annotation.

This catches default-score scripts. It does not establish that an agent read or
correctly interpreted a transcript; scoped work and content audits are still needed.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re

ID = re.compile(r"[0-9a-f]{16}\Z")


def triples(value):
    found = set()
    if isinstance(value, dict):
        if {"id", "rationale", "score"}.issubset(value):
            rid, reason, score = (value[k] for k in ("id", "rationale", "score"))
            if isinstance(rid, str) and ID.fullmatch(rid) and isinstance(reason, str):
                found.add((rid, reason, score))
        for child in value.values():
            found |= triples(child)
    elif isinstance(value, (list, tuple)):
        if len(value) == 3 and isinstance(value[0], str) and ID.fullmatch(value[0]):
            if isinstance(value[1], str) and (type(value[2]) is int or value[2] is None):
                found.add((value[0], value[1], value[2]))
            elif (type(value[1]) is int or value[1] is None) and isinstance(value[2], str):
                found.add((value[0], value[2], value[1]))
        for child in value:
            found |= triples(child)
    return found


def authored_triples(path):
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            # Nonliteral containers may hold literal children; ast.walk visits them.
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            found |= triples(value)
    return found


def verify(labels, sources):
    explicit = set().union(*(authored_triples(p) for p in sources))
    missing = [r["id"] for r in labels if (r["id"], r["rationale"], r["score"]) not in explicit]
    if missing:
        raise ValueError(f"No explicitly authored ID/rationale/score literal for {missing}")
    return len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    args = parser.parse_args()
    n = verify(json.loads(args.labels.read_text()), args.source)
    print(json.dumps({"explicit_authored_records": n, "status": "valid"}))


if __name__ == "__main__":
    main()
