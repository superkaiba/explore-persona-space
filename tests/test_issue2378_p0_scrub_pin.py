"""Source-scan pin for the P0 secret-scrub placement (r5 NIT p0-scrub-lacks-pytest-pin).

``phase_build_pools`` (scripts/issue2378_gen.py) must scrub the three freshly
written pool JSONLs BETWEEN the pool-write loop and the digest write / upload
(write -> scrub -> digest -> upload), so re-runs of the seeded permutation
re-scrub before any Hub upload. A stripped scrub still fails LOUD downstream
(``assert_upload_clean`` re-refuses at upload — the permanent mechanical
guard), so this pin is placement/permanence, not the only defense.

Text-scan (never an import): issue2378_gen.py's module top bootstraps
sys.path + load_dotenv, which a unit test must not execute.
"""

from __future__ import annotations

from pathlib import Path

GEN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "issue2378_gen.py"


def _build_pools_src() -> str:
    src = GEN_PATH.read_text(encoding="utf-8")
    start = src.index("def phase_build_pools(")
    end = src.index("\ndef ", start + 1)
    return src[start:end]


def test_scrub_sits_between_pool_write_and_digest_upload():
    src = _build_pools_src()
    pool_write = src.index("os.replace(tmp, path)")
    scrub_import = src.index(
        "from explore_persona_space.orchestrate.secret_scrub import scrub_file"
    )
    scrub_call = src.index("scrub_file(pools_dir /")
    digest_field = src.index('"secret_scrub_fixed"')
    digest_write = src.index('pool_digest.json"')
    upload = src.index("upload_stage_dir(pools_dir")
    assert pool_write < scrub_import < scrub_call < digest_field < upload
    assert scrub_call < digest_write, "digest must be composed AFTER the scrub"


def test_scrub_covers_all_three_pools():
    src = _build_pools_src()
    loop = src[src.index("for fname in (") : src.index("scrub_file(pools_dir /")]
    for name in ("chat_draw", "plain_draw", "user_draw"):
        assert name in loop, f"scrub loop must cover {name}.jsonl"


def test_scrub_emits_fix_engaged_log_line():
    """The P0 fix-engaged signal (`[build_pools] scrubbed N ...`) stays wired."""
    src = _build_pools_src()
    assert "scrubbed {len(fixed)} leaked third-party secret(s)" in src
