"""Training-only features and affine-map controls for the frozen risk analysis."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from scripts.context_risk_followup_probe_core import weighted_standardize  # noqa: E402


def metadata_matrix(rows: list[dict]) -> np.ndarray:
    """Use only fields visible in the initial user message."""
    result = []
    for row in rows:
        prompt = row["messages"][0]["content"]
        asserts = sum(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(row["test"])))
        result.append(
            [
                row["condition"] == "oneoff",
                np.log1p(len(prompt)),
                np.log1p(len(row["test"])),
                np.log1p(asserts),
            ]
        )
    return np.asarray(result, dtype=np.float64)


class FeatureBank:
    """Store label-independent raw inputs; refit every transform within its fold."""

    def __init__(self, rows: list[dict], raw: np.ndarray, map_arrays: dict, spec: dict):
        self.rows = rows
        self.raw = np.asarray(raw, dtype=np.float64)
        self.spec = spec
        self.metadata = metadata_matrix(rows)
        self.prompts = np.asarray([r["messages"][0]["content"] for r in rows])
        self.weight = np.asarray(map_arrays["weight"], dtype=np.float64)
        self.x_mean = np.asarray(map_arrays["x_mean"], dtype=np.float64)
        self.x_scale = np.asarray(map_arrays["x_scale"], dtype=np.float64)
        if not np.isfinite(self.x_scale).all() or np.any(self.x_scale < 1e-9):
            raise ValueError("Frozen map scale must be finite and at least 1e-9; no scale repair")
        self.y_mean = np.asarray(map_arrays["y_mean"], dtype=np.float64)
        if self.raw.ndim != 2 or len(self.raw) != len(rows):
            raise ValueError("Raw context matrix shape differs from the manifest")
        d = self.raw.shape[1]
        if self.weight.shape != (d, d) or any(
            v.shape != (d,) for v in (self.x_mean, self.x_scale, self.y_mean)
        ):
            raise ValueError("Affine map layout differs from activation dimensions")
        for array in (self.raw, self.weight, self.x_mean, self.x_scale, self.y_mean):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite map or activation input")
        self.mapped = ((self.raw - self.x_mean) / self.x_scale) @ self.weight + self.y_mean
        self.cache = {}

    def prepare(self, train: np.ndarray, test: np.ndarray, trials: np.ndarray) -> dict:
        """Fit scaling once for a shared fold and reuse it across C values and methods."""
        key = (tuple(train), tuple(test), tuple(trials[train]))
        if key in self.cache:
            return self.cache[key]
        r_train, r_test, r_scale = weighted_standardize(
            self.raw[train], self.raw[test], trials[train]
        )
        m_train, m_test, m_scale = weighted_standardize(
            self.mapped[train], self.mapped[test], trials[train]
        )
        meta_train, meta_test, meta_scale = weighted_standardize(
            self.metadata[train], self.metadata[test], trials[train]
        )
        # The frozen identity-plus-bias map is x + b. Any fixed b cancels under
        # training-only centering; use the map's effective affine intercept here.
        bias = self.y_mean - (self.x_mean / self.x_scale) @ self.weight
        identity_train, identity_test, _ = weighted_standardize(
            self.raw[train] + bias, self.raw[test] + bias, trials[train]
        )
        identity_error = max(
            float(np.max(np.abs(identity_train - r_train))),
            float(np.max(np.abs(identity_test - r_test))),
        )
        if not np.allclose(identity_train, r_train, atol=1e-8, rtol=1e-8) or not np.allclose(
            identity_test, r_test, atol=1e-8, rtol=1e-8
        ):
            raise ValueError("Identity-plus-bias standardization parity failed")
        # Map centering is affine; the standardized operator is exactly
        # diag(s_raw) diag(1/x_scale) W diag(1/s_mapped).
        composed_train = (r_train * (r_scale.scale_ / self.x_scale)) @ self.weight / m_scale.scale_
        composed_test = (r_test * (r_scale.scale_ / self.x_scale)) @ self.weight / m_scale.scale_
        if not np.allclose(composed_train, m_train, atol=1e-8, rtol=1e-8) or not np.allclose(
            composed_test, m_test, atol=1e-8, rtol=1e-8
        ):
            raise ValueError("Affine standardization operator parity failed")
        prepared = {
            "raw": (r_train, r_test),
            "mapped": (m_train, m_test),
            "metadata": (meta_train, meta_test),
            "raw_scaler": r_scale,
            "mapped_scaler": m_scale,
            "metadata_scaler": meta_scale,
            "identity_bias_max_error": identity_error,
            "operator_max_error": max(
                float(np.max(np.abs(composed_train - m_train))),
                float(np.max(np.abs(composed_test - m_test))),
            ),
        }
        self.cache[key] = prepared
        return prepared

    def features(
        self,
        method: str,
        train: np.ndarray,
        test: np.ndarray,
        trials: np.ndarray,
        *,
        rank: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Build one representation with the same untouched training metadata block."""
        prepared = self.prepare(train, test, trials)
        meta_train, meta_test = prepared["metadata"]
        info = {
            "method": method,
            "identity_bias_max_error": prepared["identity_bias_max_error"],
            "operator_max_error": prepared["operator_max_error"],
        }
        if method == "metadata":
            return meta_train, meta_test, info
        if method in {"raw_plus_metadata", "mapped_plus_metadata"}:
            block = "raw" if method == "raw_plus_metadata" else "mapped"
            left, right = prepared[block]
        elif method == "text_plus_metadata":
            key = "text"
            if key not in prepared:
                cfg = self.spec["text"]
                vectorizer = TfidfVectorizer(
                    analyzer=cfg["analyzer"],
                    ngram_range=tuple(cfg["ngram_range"]),
                    min_df=cfg["min_df"],
                    max_features=cfg["max_features"],
                    sublinear_tf=cfg["sublinear_tf"],
                    norm=cfg["norm"],
                    dtype=np.float64,
                )
                left = vectorizer.fit_transform(self.prompts[train].tolist()).toarray()
                right = vectorizer.transform(self.prompts[test].tolist()).toarray()
                prepared[key] = (left, right, vectorizer)
            left, right, vectorizer = prepared[key]
            info["vocabulary_size"] = len(vectorizer.vocabulary_)
        elif method.startswith("orientation_"):
            seed = int(method.removeprefix("orientation_"))
            if seed not in self.spec["orientation_controls"]["seeds"]:
                raise ValueError("Unplanned orientation control seed")
            if method not in prepared:
                rng = np.random.default_rng(seed)
                permutation = rng.permutation(self.raw.shape[1])
                signs = rng.choice([-1.0, 1.0], size=self.raw.shape[1])
                multiplier = prepared["raw_scaler"].scale_ / self.x_scale
                # Q acts in common standardized input coordinates. Reuse the
                # real map's output scaling; do not refit a scaler to a control.
                prepared[method] = tuple(
                    ((values[:, permutation] * signs) * multiplier)
                    @ self.weight
                    / prepared["mapped_scaler"].scale_
                    for values in prepared["raw"]
                )
            left, right = prepared[method]
        elif method == "pca_plus_metadata":
            if rank not in self.spec["secondary_PCA_control"]["rank_candidates"]:
                raise ValueError("Unplanned PCA rank")
            effective = min(rank, len(train) - 1, self.raw.shape[1])
            if effective < 1:
                raise ValueError("Insufficient distinct training contexts for PCA")
            if "pca" not in prepared:
                pca = PCA(n_components=min(len(train) - 1, self.raw.shape[1]), svd_solver="full")
                left = pca.fit_transform(prepared["raw"][0])
                right = pca.transform(prepared["raw"][1])
                prepared["pca"] = (left, right, pca)
            left, right, _ = prepared["pca"]
            left, right = left[:, :effective], right[:, :effective]
            info.update({"nominal_rank": rank, "effective_rank": effective})
        else:
            raise ValueError(f"Unknown representation: {method}")
        return np.column_stack([left, meta_train]), np.column_stack([right, meta_test]), info

    def compose_mapped_head(
        self,
        train: np.ndarray,
        test: np.ndarray,
        trials: np.ndarray,
        coefficient: np.ndarray,
        intercept: float,
    ) -> np.ndarray:
        """Evaluate a fitted mapped head as a raw linear head, without inverting the map."""
        prepared = self.prepare(train, test, trials)
        d = self.raw.shape[1]
        raw_coefficient = (prepared["raw_scaler"].scale_ / self.x_scale) * (
            self.weight @ (coefficient[:d] / prepared["mapped_scaler"].scale_)
        )
        return (
            prepared["raw"][1] @ raw_coefficient
            + prepared["metadata"][1] @ coefficient[d:]
            + intercept
        )
