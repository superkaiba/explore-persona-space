"""Regression tests for the --claimed-urls-file phantom-URL gate (#541).

The claimed-urls blob handed to ``scripts/verify_uploads.py`` is frequently a
JSON ``epm:results`` sentinel, where every URL is immediately followed by
``",`` (or ``\\",`` when the JSON is nested). hub.py's ``_HF_URL_RE``
revision/path character classes exclude only ``/``, whitespace, ``)`` and
``]``, so without sanitization the trailing punctuation rides into the probed
path and every HEAD check misses — a false ``claimed_urls`` FAIL (incident
#541, 2026-06-10). These tests pin the extractor's punctuation stripping and
the end-to-end ``check_claimed_urls_resolve`` path on a JSON blob (network
mocked, same conventions as tests/test_hub_artifacts_and_refusal.py).
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.hf_api import RepoFile, RepoFolder

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]


# ── extract_claimed_urls ──────────────────────────────────────────────────────


class TestExtractClaimedUrls:
    def test_json_blob_strips_trailing_punctuation(self):
        """URLs inside a JSON blob come out without the '",' suffix (#541)."""
        blob = json.dumps(
            {
                "wandb_url": "https://wandb.ai/team/proj/runs/run123",
                "urls": [
                    "https://huggingface.co/superkaiba1/explore-persona-space-overflow"
                    "/tree/main/adapters/exp541-arm_marine_biologist-seed42",
                    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data"
                    "/tree/main/issue541_prior_stratified/arm_marine_biologist",
                ],
            }
        )
        urls = verify_uploads.extract_claimed_urls(blob)
        assert urls == [
            "https://wandb.ai/team/proj/runs/run123",
            "https://huggingface.co/superkaiba1/explore-persona-space-overflow"
            "/tree/main/adapters/exp541-arm_marine_biologist-seed42",
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data"
            "/tree/main/issue541_prior_stratified/arm_marine_biologist",
        ]
        for u in urls:
            assert u[-1] not in "\\'\",;)]}>`", f"trailing punctuation survived: {u!r}"

    def test_nested_json_escaped_quotes(self):
        """A JSON-in-JSON note leaves '\\",' after each URL — both stripped."""
        blob = '{"note": "{\\"hf\\": \\"https://huggingface.co/org/repo/tree/main/path\\",}"}'
        assert verify_uploads.extract_claimed_urls(blob) == [
            "https://huggingface.co/org/repo/tree/main/path"
        ]

    def test_markdown_wrappers(self):
        """Markdown link parens and backtick code spans are stripped."""
        blob = (
            "See [the adapter](https://huggingface.co/org/repo/tree/main/cond_seed42) "
            "and `hf://org/repo@main/cond_seed42`.\n"
        )
        assert verify_uploads.extract_claimed_urls(blob) == [
            "https://huggingface.co/org/repo/tree/main/cond_seed42",
            "hf://org/repo@main/cond_seed42",
        ]

    def test_dedupes_preserving_order(self):
        blob = (
            '"https://wandb.ai/t/p/runs/r1", "https://huggingface.co/a/b", '
            '"https://wandb.ai/t/p/runs/r1"'
        )
        assert verify_uploads.extract_claimed_urls(blob) == [
            "https://wandb.ai/t/p/runs/r1",
            "https://huggingface.co/a/b",
        ]

    def test_real_suffix_not_truncated(self):
        """'.json' / '.safetensors' endings survive (only punctuation strips)."""
        blob = '{"f": "https://huggingface.co/a/b/blob/main/eval/results.json",}'
        assert verify_uploads.extract_claimed_urls(blob) == [
            "https://huggingface.co/a/b/blob/main/eval/results.json"
        ]

    def test_no_urls(self):
        assert verify_uploads.extract_claimed_urls("no links here") == []


# ── resolve_claimed_repo_types (dataset-repo fallback, #599) ─────────────────


def _make_repo_info_api(dataset_repos=(), model_repos=()):
    """Mock HfApi whose repo_info resolves only the given repo ids per type."""
    from huggingface_hub.utils import RepositoryNotFoundError

    api = MagicMock()

    def _repo_info(repo_id, repo_type=None, **kwargs):
        if repo_type == "dataset" and repo_id in dataset_repos:
            return MagicMock()
        if repo_type == "model" and repo_id in model_repos:
            return MagicMock()
        raise RepositoryNotFoundError("404")

    api.repo_info.side_effect = _repo_info
    return api


class TestResolveClaimedRepoTypes:
    def test_bare_hf_uri_dataset_claim_rewritten(self):
        """The #599 shape: hf:// dataset-repo claim without the datasets/
        prefix is rewritten; the -data-private suffix probes dataset FIRST
        (exactly one repo_info call)."""
        urls = ["hf://superkaiba1/explore-persona-space-data-private/issue599_fullresp/eval"]
        api = _make_repo_info_api(dataset_repos={"superkaiba1/explore-persona-space-data-private"})
        with patch("huggingface_hub.HfApi", return_value=api):
            resolved, rewritten, phantoms = verify_uploads.resolve_claimed_repo_types(urls)
        assert resolved == [
            "hf://datasets/superkaiba1/explore-persona-space-data-private/issue599_fullresp/eval"
        ]
        assert rewritten == {resolved[0]: urls[0]}
        assert phantoms == []
        api.repo_info.assert_called_once_with(
            "superkaiba1/explore-persona-space-data-private", repo_type="dataset"
        )

    def test_bare_web_dataset_claim_rewritten(self):
        urls = [
            "https://huggingface.co/superkaiba1/explore-persona-space-data"
            "/tree/main/issue599_fullresp"
        ]
        api = _make_repo_info_api(dataset_repos={"superkaiba1/explore-persona-space-data"})
        with patch("huggingface_hub.HfApi", return_value=api):
            resolved, _, phantoms = verify_uploads.resolve_claimed_repo_types(urls)
        assert resolved == [
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data"
            "/tree/main/issue599_fullresp"
        ]
        assert phantoms == []

    def test_model_claim_passes_through(self):
        urls = ["https://huggingface.co/org/repo/tree/main/adapters/seed42"]
        api = _make_repo_info_api(model_repos={"org/repo"})
        with patch("huggingface_hub.HfApi", return_value=api):
            resolved, rewritten, phantoms = verify_uploads.resolve_claimed_repo_types(urls)
        assert resolved == urls
        assert rewritten == {}
        assert phantoms == []
        api.repo_info.assert_called_once_with("org/repo", repo_type="model")

    def test_phantom_repo_split_out_not_raised(self):
        """A claim whose repo resolves as NEITHER type becomes a phantom
        (deterministic FAIL downstream) instead of aborting the scan."""
        urls = ["hf://org/ghost-repo/some/path"]
        api = _make_repo_info_api()
        with patch("huggingface_hub.HfApi", return_value=api):
            resolved, _, phantoms = verify_uploads.resolve_claimed_repo_types(urls)
        assert resolved == []
        assert phantoms == urls

    def test_prefixed_and_wandb_claims_never_probed(self):
        urls = [
            "https://huggingface.co/datasets/org/repo/tree/main/x",
            "hf://datasets/org/repo/x",
            "https://wandb.ai/team/proj/runs/run123",
        ]
        api = _make_repo_info_api()
        with patch("huggingface_hub.HfApi", return_value=api):
            resolved, _, phantoms = verify_uploads.resolve_claimed_repo_types(urls)
        assert resolved == urls
        assert phantoms == []
        api.repo_info.assert_not_called()

    def test_repo_probe_cached_across_urls(self):
        urls = [
            "hf://superkaiba1/explore-persona-space-data-private/a",
            "hf://superkaiba1/explore-persona-space-data-private/b",
        ]
        api = _make_repo_info_api(dataset_repos={"superkaiba1/explore-persona-space-data-private"})
        with patch("huggingface_hub.HfApi", return_value=api):
            resolved, _, _ = verify_uploads.resolve_claimed_repo_types(urls)
        assert len(resolved) == 2
        assert api.repo_info.call_count == 1


# ── check_claimed_urls_resolve end-to-end (network mocked) ───────────────────


def _repo_file(path: str) -> RepoFile:
    return RepoFile(path=path, size=1, blob_id="b", oid="o")


def _make_api_with_files(file_paths):
    """Mock HfApi whose list_repo_tree yields the given files (+ one folder)."""
    api = MagicMock()

    def _tree(*args, **kwargs):
        yield RepoFolder(path="some_dir", tree_id="t", oid="o")
        for p in file_paths:
            yield _repo_file(p)

    api.list_repo_tree.side_effect = _tree
    return api


class TestCheckClaimedUrlsResolve:
    def test_json_blob_with_present_artifacts_passes(self, tmp_path):
        """The #541 repro: a JSON sentinel whose artifacts all exist → OK.

        Before the extractor fix, the trailing '",' rode into the probed
        paths, every HEAD check missed, and this returned a false FAIL.
        """
        blob_path = tmp_path / "claimed-urls.txt"
        blob_path.write_text(
            json.dumps(
                {
                    "checkpoints": [
                        "https://huggingface.co/superkaiba1/explore-persona-space-overflow"
                        "/tree/main/adapters/exp541-seed42",
                    ],
                    "wandb_url": "https://wandb.ai/team/proj/runs/run123",
                }
            ),
            encoding="utf-8",
        )
        api = _make_api_with_files(["adapters/exp541-seed42/adapter_model.safetensors"])
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            patch(
                "explore_persona_space.orchestrate.hub._wandb_run_exists",
                return_value=True,
            ),
        ):
            result = verify_uploads.check_claimed_urls_resolve(blob_path)
        assert result["status"] == "OK", result

    def test_genuinely_missing_artifact_still_fails(self, tmp_path):
        """Sanitization must not weaken the gate: a real phantom still FAILs."""
        blob_path = tmp_path / "claimed-urls.txt"
        blob_path.write_text(
            '{"ckpt": "https://huggingface.co/org/repo/tree/main/ghost_seed99",}',
            encoding="utf-8",
        )
        api = _make_api_with_files(["adapters/real_seed42/adapter_model.safetensors"])
        with patch("huggingface_hub.HfApi", return_value=api):
            result = verify_uploads.check_claimed_urls_resolve(blob_path)
        assert result["status"] == "FAIL", result
        # The reported phantom URL is the CLEAN one (no trailing punctuation).
        assert "ghost_seed99" in result["detail"]
        assert 'ghost_seed99"' not in result["detail"]

    def test_bare_dataset_repo_claim_resolves_ok(self, tmp_path):
        """The #599 repro: an hf:// dataset-repo claim without the datasets/
        prefix used to resolve via the MODELS endpoint, 404, and turn the
        whole claimed_urls row into ERROR. Now it is rewritten to the
        datasets/ form and resolves OK."""
        from huggingface_hub.utils import RepositoryNotFoundError

        blob_path = tmp_path / "claimed-urls.txt"
        blob_path.write_text(
            '{"gate_jsons_hf_prefix": '
            '"hf://superkaiba1/explore-persona-space-data-private/issue599_fullresp/eval/",}',
            encoding="utf-8",
        )
        api = _make_api_with_files(["issue599_fullresp/eval/gate_b0.json"])

        def _repo_info(repo_id, repo_type=None, **kwargs):
            if repo_type == "dataset":
                return MagicMock()
            raise RepositoryNotFoundError("404")

        api.repo_info.side_effect = _repo_info
        with patch("huggingface_hub.HfApi", return_value=api):
            result = verify_uploads.check_claimed_urls_resolve(blob_path)
        assert result["status"] == "OK", result
        assert "repo_type=dataset" in result["detail"]

    def test_phantom_repo_claim_fails_not_errors(self, tmp_path):
        """A claim whose repo resolves as NEITHER model nor dataset is a
        deterministic FAIL naming the as-cited URL — not an ERROR aborting
        the rest of the scan."""
        from huggingface_hub.utils import RepositoryNotFoundError

        blob_path = tmp_path / "claimed-urls.txt"
        blob_path.write_text(
            '{"ckpt": "hf://org/ghost-repo/some/path",}',
            encoding="utf-8",
        )
        api = _make_api_with_files([])
        api.repo_info.side_effect = RepositoryNotFoundError("404")
        with patch("huggingface_hub.HfApi", return_value=api):
            result = verify_uploads.check_claimed_urls_resolve(blob_path)
        assert result["status"] == "FAIL", result
        assert "hf://org/ghost-repo/some/path" in result["detail"]
        assert "neither model nor dataset" in result["detail"]

    def test_missing_file_is_error(self, tmp_path):
        result = verify_uploads.check_claimed_urls_resolve(tmp_path / "nope.txt")
        assert result["status"] == "ERROR"

    def test_no_file_is_skip(self):
        result = verify_uploads.check_claimed_urls_resolve("")
        assert result["status"] == "SKIP"
