"""Issue #2223 NAP round — dashboard HTML-escaping pins (r2 concern
dashboard-diagnostic-html-escaping).

``_tbl`` must escape EVERY plain-``str`` cell (diagnostic JSON values are
untrusted text); ``_Safe`` is the ONLY explicit opt-out for intentional
markup. Synthetic dicts only — no artifacts, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue2223_native_preimage_dashboard as D  # noqa: E402

INJ = "<script>alert(1)</script>"


def test_tbl_escapes_plain_strings_safe_optout_and_fmt():
    out = D._tbl(["h<1>"], [[INJ], [D._Safe("<b>ok</b>")], [1.5], [None], [True]])
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "<th>h&lt;1&gt;</th>" in out  # headers escaped
    assert "<td><b>ok</b></td>" in out  # explicit _Safe opt-out passes through
    assert "<td>1.5000</td>" in out and "<td>—</td>" in out and "<td>yes</td>" in out


def test_render_diagnostics_escapes_injection_shaped_values():
    """An injection-shaped diagnostic string (h1 classification) renders
    escaped in BOTH the table cell and the raw-JSON details block."""
    diags = {
        "axis_cos.json": {
            "h1_gate": {
                "band_min_cos": 0.95,
                "band_mean_cos": 0.97,
                "band_all_pass": True,
                "mid_layer": 32,
                "mid_cos": 0.8,
                "mid_pass": True,
                "classification": INJ,
            },
            "band_layers": [46],
            "cos_reextracted_vs_reference": {"46": 0.95},
        }
    }
    html_out = D.render_diagnostics_html(diags)
    assert "<script>" not in html_out
    assert html_out.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 2  # table + raw JSON
