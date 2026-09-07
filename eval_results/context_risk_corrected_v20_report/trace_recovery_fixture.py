"""Read-only source/real-trace review; synthetic gzip cases write to temporary paths."""

import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from scripts.context_risk_corrected_finish import preserve_finalized_trace

source = Path("scripts/context_risk_corrected_finish.py")
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
root = Path(tempfile.mkdtemp(prefix="context-risk-finalized-trace-review-"))
cases = []
valid = (
    b'{"timestamp":"2026-09-07","level":"INFO","message":"fixture"}\n'
    b'{"timestamp":"2026-09-07","level":"WARN","message":"[APIConnectionError] fixture"}\n'
)


def check(name, content, exception=None, compressed=True):
    original, destination = root / f"{name}.log.gz", root / f"{name}.jsonl"
    if content is not None:
        original.write_bytes(gzip.compress(content) if compressed else content)
    try:
        result = preserve_finalized_trace(original, destination)
    except Exception as error:
        assert exception and isinstance(error, exception), (name, type(error), str(error))
        assert not destination.exists()
        cases.append({"case": name, "passed": True, "exception": type(error).__name__})
    else:
        assert exception is None and destination.read_bytes() == content
        assert result["n_records"] == 2 and len(result["connection_retries"]) == 1
        assert result["source_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
        assert result["destination_sha256"] == hashlib.sha256(content).hexdigest()
        cases.append({"case": name, "passed": True, "n_records": result["n_records"]})


check("valid", valid)
check("missing", None, FileNotFoundError)
check("not_gzip", b"not gzip", gzip.BadGzipFile, compressed=False)
corrupt = bytearray(gzip.compress(valid))
corrupt[-8] ^= 1
check("corrupt_crc", bytes(corrupt), gzip.BadGzipFile, compressed=False)
check("truncated_stream", gzip.compress(valid)[:-2], EOFError, compressed=False)
check("empty", b"", ValueError)
check("non_json", b"invalid\n", json.JSONDecodeError)
check("missing_event_fields", b"{}\n", ValueError)
check("non_object", b"[]\n", ValueError)

actual = Path("/home/thomasjiralerspong/.local/share/inspect_ai/traces/trace-3704829.log.gz")
actual_before = hashlib.sha256(actual.read_bytes()).hexdigest()
result = preserve_finalized_trace(actual, root / "actual_finalized_trace.jsonl")
assert result["source_sha256"] == actual_before == hashlib.sha256(actual.read_bytes()).hexdigest()
assert result["n_records"] > 0 and result["connection_retries"]
cases.append(
    {
        "case": "actual_completed_full_trace",
        "passed": True,
        "source_sha256": actual_before,
        "destination_sha256": result["destination_sha256"],
        "compressed_bytes": actual.stat().st_size,
        "decompressed_bytes": (root / "actual_finalized_trace.jsonl").stat().st_size,
        "n_records": result["n_records"],
        "connection_retry_events": len(result["connection_retries"]),
    }
)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
report = {
    "verdict": "PASS",
    "source": str(source),
    "source_sha256": source_sha,
    "n_cases": len(cases),
    "cases": cases,
    "scope": (
        "Nine local gzip/JSONL fixtures plus the real finalized trace; no model/pod/upload calls."
    ),
}
destination = Path(__file__).with_name("trace_recovery_fixture_result.json")
destination.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"report": str(destination), **report}, indent=2))
