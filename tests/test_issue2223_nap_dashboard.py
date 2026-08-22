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


def test_facet_options_built_via_textcontent_not_innerhtml():
    """r3 CONCERN dashboard-facet-innerhtml-sink (fixed r4): facet <option>
    construction goes through createElement + textContent — the
    string-concatenated ``sel.innerHTML = vals.map(...)`` sink is gone. The
    bar-DV selector's literals-only innerHTML is the ONE permitted remaining
    use (reconciler r3: observed-safe, pure literals)."""
    import re

    js = D._JS
    assert "'<option selected>'" not in js  # the removed concat sink
    assert "sel.innerHTML" not in js
    assert 'document.createElement("option")' in js
    assert "opt.textContent = v" in js and "opt.selected = true" in js
    # remaining innerHTML uses are EXACTLY the permitted set: the literals-only
    # bar-DV selector (dv), empty-string svg clears, and the esc()-mediated
    # renderTable join (div) — no data-bearing concat sink.
    assert sorted(set(re.findall(r"(\w+)\.innerHTML", js))) == ["div", "dv", "svg"]
    assert "dv.innerHTML = '<option>harm</option>" in js


def test_render_html_emits_facet_createelement_path_in_output_bytes():
    """r5 CONCERN dashboard-render-smoke-missing: drive the PRODUCTION
    ``render_html`` on a tiny fixture and assert the facet construction
    (createElement + textContent) and the permitted innerHTML set on the
    EMITTED HTML/JS bytes — not on ``D._JS`` read directly."""
    import re

    rec = {
        "cell": "band__cap_ctx__s42",
        "scenario": "selfharm",
        "layers": "band",
        "family": "ctx_native",
        "op": "cap",
        "strength": "p90",
        "arm": "cap_ctx",
        "turns": [{"turn": 1, "harm": 0.0, "coherence": 80.0}],
    }
    meta = {"model": "test-model", "round": "nap-r5", "model_root": "/tmp/x"}
    out = D.render_html([rec], meta, {})
    assert out.startswith("<!doctype html>")
    # facet <option> construction reaches the emitted script bytes via the
    # createElement + textContent path; the concat sink stays gone.
    assert 'document.createElement("option")' in out
    assert "opt.textContent = v" in out and "opt.selected = true" in out
    assert "sel.innerHTML" not in out
    # permitted innerHTML set, asserted on the FULL emitted document.
    assert sorted(set(re.findall(r"(\w+)\.innerHTML", out))) == ["div", "dv", "svg"]


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
