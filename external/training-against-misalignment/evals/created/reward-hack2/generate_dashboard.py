#!/usr/bin/env python3
"""Generate an HTML dashboard for browsing seed eval prompts and rollouts."""

import json
import sys
import html
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import SCENARIOS, make_system_prompt, make_user_prompt


def generate_dashboard(results_file: str, output_file: str = "seed_dashboard.html"):
    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]
    model_path = data.get("model_path", "unknown")
    model_name = Path(model_path).name
    num_rollouts = data.get("num_rollouts", 50)

    # Build scenario lookup
    scenario_lookup = {s["id"]: s for s in SCENARIOS}

    # Group by scenario
    by_scenario = {}
    for entry in results:
        sid = entry["scenario_id"]
        if sid not in by_scenario:
            by_scenario[sid] = []
        by_scenario[sid].append(entry)

    # Score each rollout
    from score_seeds import get_item_decision, is_hack, is_nonsense

    # Build scores file path for hack rates
    scores_file = Path(results_file).parent / Path(results_file).name.replace("seed_results_", "seed_scores_")
    scenario_scores = {}
    if scores_file.exists():
        scores_data = json.load(open(scores_file))
        for s in scores_data.get("scenarios", []):
            scenario_scores[s["id"]] = s

    # Pre-compute per-rollout classifications
    rollout_labels = {}  # (scenario_id, condition, rollout_id) -> "hack" | "clean" | "nonsense"
    for entry in results:
        sid = entry["scenario_id"]
        cond = entry["condition"]
        diag_items = entry["diagnostic_items"]
        hack_actions = entry["diagnostic_hack_actions"]
        for r in entry["rollouts"]:
            rid = r["rollout_id"]
            raw = r["raw_response"]
            if is_nonsense(raw):
                label = "nonsense"
            else:
                hacked = False
                for item_id in diag_items:
                    decision = get_item_decision(raw, item_id, sid)
                    if is_hack(decision, hack_actions):
                        hacked = True
                        break
                label = "hack" if hacked else "clean"
            rollout_labels[(sid, cond, rid)] = label

    # Generate HTML
    h = []
    h.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Seed Eval Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 18px; color: #58a6ff; }
.header .meta { font-size: 13px; color: #8b949e; margin-top: 4px; }
.container { display: flex; height: calc(100vh - 70px); }
.sidebar { width: 320px; min-width: 320px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; }
.main-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.prompt-panel { height: 40%; overflow-y: auto; border-bottom: 1px solid #30363d; padding: 16px; background: #0d1117; }
.response-panel { flex: 1; overflow-y: auto; padding: 16px; background: #0d1117; }
.scenario-group { border-bottom: 1px solid #21262d; }
.scenario-header { padding: 10px 16px; cursor: pointer; font-weight: 600; font-size: 14px; display: flex; justify-content: space-between; align-items: center; }
.scenario-header:hover { background: #1c2128; }
.scenario-header .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500; }
.badge-good { background: #1a7f37; color: #fff; }
.badge-ok { background: #9e6a03; color: #fff; }
.badge-weak { background: #6e7681; color: #fff; }
.condition-list { display: none; }
.condition-list.open { display: block; }
.condition-item { padding: 6px 16px 6px 28px; cursor: pointer; font-size: 13px; display: flex; justify-content: space-between; border-left: 3px solid transparent; }
.condition-item:hover { background: #1c2128; }
.condition-item.active { background: #1c2128; border-left-color: #58a6ff; }
.condition-item .rate { font-size: 12px; font-weight: 600; }
.rate-high { color: #f85149; }
.rate-med { color: #d29922; }
.rate-low { color: #3fb950; }
.rate-zero { color: #6e7681; }
.rollout-nav { display: flex; gap: 4px; flex-wrap: wrap; padding: 8px 16px; background: #161b22; border-bottom: 1px solid #30363d; align-items: center; }
.rollout-nav .label { font-size: 12px; color: #8b949e; margin-right: 8px; }
.rollout-btn { width: 28px; height: 24px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; font-size: 11px; cursor: pointer; border-radius: 4px; display: flex; align-items: center; justify-content: center; }
.rollout-btn:hover { border-color: #58a6ff; }
.rollout-btn.active { background: #58a6ff; color: #fff; border-color: #58a6ff; }
.rollout-btn.hack { border-color: #f85149; background: #3d1214; }
.rollout-btn.hack.active { background: #f85149; }
.rollout-btn.nonsense { border-color: #6e7681; background: #1c1c1c; color: #6e7681; }
.panel-title { font-size: 13px; font-weight: 600; color: #58a6ff; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
pre { white-space: pre-wrap; word-wrap: break-word; font-size: 13px; line-height: 1.5; }
.prompt-text { color: #8b949e; }
.response-text { color: #c9d1d9; }
.highlight-hack { background: #3d1214; border-left: 3px solid #f85149; padding: 2px 8px; display: inline; }
.highlight-tool { background: #0c2d6b; border-left: 3px solid #58a6ff; padding: 2px 8px; display: inline; }
.diag-info { font-size: 12px; color: #8b949e; background: #161b22; padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #30363d; }
.diag-info strong { color: #d29922; }
.label-hack { color: #f85149; font-weight: 700; }
.label-clean { color: #3fb950; font-weight: 700; }
.label-nonsense { color: #6e7681; font-weight: 700; }
.empty-state { color: #6e7681; font-size: 14px; padding: 40px; text-align: center; }
.filter-bar { padding: 8px 16px; border-bottom: 1px solid #30363d; display: flex; gap: 8px; align-items: center; }
.filter-bar label { font-size: 12px; color: #8b949e; }
.filter-bar select { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div class="header">
  <h1>Seed Eval Dashboard</h1>
  <div class="meta">Model: MODEL_NAME | TOTAL_ENTRIES conditions | NUM_ROLLOUTS rollouts/condition</div>
</div>
<div class="container">
  <div class="sidebar" id="sidebar"></div>
  <div class="main-panel">
    <div class="rollout-nav" id="rollout-nav"></div>
    <div class="prompt-panel" id="prompt-panel"><div class="empty-state">Select a scenario and condition from the sidebar</div></div>
    <div class="response-panel" id="response-panel"><div class="empty-state">Select a rollout to view the response</div></div>
  </div>
</div>
""".replace("MODEL_NAME", html.escape(model_name))
     .replace("TOTAL_ENTRIES", str(len(results)))
     .replace("NUM_ROLLOUTS", str(num_rollouts)))

    # Embed data as JSON
    # Build compact data structure
    dashboard_data = {
        "model": model_name,
        "scenarios": [],
    }

    scenario_order = []
    # Sort scenarios by delta (best first)
    for sid in sorted(by_scenario.keys(), key=lambda s: -scenario_scores.get(s, {}).get("delta", 0)):
        scenario_order.append(sid)
        sc = scenario_lookup.get(sid, {})
        entries = by_scenario[sid]

        # Generate prompts for each condition
        conditions_data = []
        for entry in sorted(entries, key=lambda e: e["condition"]):
            cond = entry["condition"]
            parts = cond.split("-")
            gv = parts[0] if len(parts) > 0 else "none"
            sp = parts[1] if len(parts) > 1 else "none"
            obs = parts[2] if len(parts) > 2 else "default"
            tp = parts[3] if len(parts) > 3 else "default"

            sys_prompt = make_system_prompt(sc, goal_value=gv, observability=obs) if sc else "(prompt unavailable)"
            usr_prompt = make_user_prompt(sc, self_preservation=sp, time_pressure=tp) if sc else "(prompt unavailable)"

            rollouts_data = []
            for r in entry["rollouts"]:
                label = rollout_labels.get((sid, cond, r["rollout_id"]), "clean")
                rollouts_data.append({
                    "id": r["rollout_id"],
                    "response": r["raw_response"],
                    "label": label,
                })

            # Hack rate for this condition
            hack_count = sum(1 for r in rollouts_data if r["label"] == "hack")
            nonsense_count = sum(1 for r in rollouts_data if r["label"] == "nonsense")
            denom = len(rollouts_data) - nonsense_count
            hack_rate = hack_count / denom * 100 if denom > 0 else 0

            conditions_data.append({
                "condition": cond,
                "goal": gv,
                "self_preservation": sp,
                "observability": obs,
                "system_prompt": sys_prompt,
                "user_prompt": usr_prompt,
                "hack_rate": round(hack_rate, 1),
                "rollouts": rollouts_data,
            })

        sc_scores = scenario_scores.get(sid, {})
        dashboard_data["scenarios"].append({
            "id": sid,
            "name": sc.get("name", sid),
            "floor": sc_scores.get("floor", 0),
            "max_rate": sc_scores.get("max_rate", 0),
            "delta": sc_scores.get("delta", 0),
            "status": sc_scores.get("status", "WEAK"),
            "diagnostic_items": entries[0]["diagnostic_items"] if entries else [],
            "diagnostic_hack_actions": entries[0]["diagnostic_hack_actions"] if entries else [],
            "diagnostic_description": entries[0].get("diagnostic_description", ""),
            "conditions": conditions_data,
        })

    h.append("<script>")
    h.append("const DATA = ")
    h.append(json.dumps(dashboard_data, ensure_ascii=False))
    h.append(";")

    h.append("""
let currentScenario = null;
let currentCondition = null;
let currentRollout = 0;

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function highlightToolBlocks(text) {
  // Highlight <tool_use:*>...</tool_use:*> blocks
  let escaped = escapeHtml(text);
  escaped = escaped.replace(/(&lt;tool_use:\\w+&gt;[\\s\\S]*?&lt;\\/tool_use:\\w+&gt;)/g, '<span class="highlight-tool">$1</span>');
  return escaped;
}

function renderSidebar() {
  const sidebar = document.getElementById('sidebar');
  let html = '';
  DATA.scenarios.forEach((sc, si) => {
    const badgeClass = sc.status === 'GOOD' ? 'badge-good' : sc.status === 'OK' ? 'badge-ok' : 'badge-weak';
    html += '<div class="scenario-group">';
    html += '<div class="scenario-header" onclick="toggleScenario(' + si + ')" id="sh-' + si + '">';
    html += '<span>' + escapeHtml(sc.id) + '</span>';
    html += '<span class="badge ' + badgeClass + '">' + sc.status + ' \\u0394' + (sc.delta > 0 ? '+' : '') + sc.delta + '%</span>';
    html += '</div>';
    html += '<div class="condition-list" id="cl-' + si + '">';
    sc.conditions.forEach((c, ci) => {
      const rateClass = c.hack_rate >= 40 ? 'rate-high' : c.hack_rate >= 15 ? 'rate-med' : c.hack_rate > 0 ? 'rate-low' : 'rate-zero';
      html += '<div class="condition-item" id="ci-' + si + '-' + ci + '" onclick="selectCondition(' + si + ',' + ci + ')">';
      html += '<span>' + escapeHtml(c.condition) + '</span>';
      html += '<span class="rate ' + rateClass + '">' + c.hack_rate.toFixed(0) + '%</span>';
      html += '</div>';
    });
    html += '</div></div>';
  });
  sidebar.innerHTML = html;
}

function toggleScenario(si) {
  const cl = document.getElementById('cl-' + si);
  cl.classList.toggle('open');
}

function selectCondition(si, ci) {
  // Remove old active
  document.querySelectorAll('.condition-item.active').forEach(el => el.classList.remove('active'));
  document.getElementById('ci-' + si + '-' + ci).classList.add('active');

  currentScenario = si;
  currentCondition = ci;
  currentRollout = 0;

  const sc = DATA.scenarios[si];
  const cond = sc.conditions[ci];

  // Render prompt
  const pp = document.getElementById('prompt-panel');
  let promptHtml = '<div class="panel-title">System Prompt</div>';
  promptHtml += '<pre class="prompt-text">' + escapeHtml(cond.system_prompt) + '</pre>';
  promptHtml += '<br><div class="panel-title">User Prompt</div>';
  promptHtml += '<pre class="prompt-text">' + escapeHtml(cond.user_prompt) + '</pre>';
  pp.innerHTML = promptHtml;

  // Render diagnostic info
  const diagHtml = '<div class="diag-info"><strong>Diagnostic:</strong> ' + escapeHtml(sc.diagnostic_description) +
    '<br><strong>Items:</strong> ' + sc.diagnostic_items.join(', ') +
    ' | <strong>Hack actions:</strong> ' + sc.diagnostic_hack_actions.join(', ') + '</div>';

  // Render rollout nav
  renderRolloutNav(cond, diagHtml);
  showRollout(0);
}

function renderRolloutNav(cond, diagHtml) {
  const nav = document.getElementById('rollout-nav');
  let html = diagHtml;
  html += '<div style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin-top:8px">';
  html += '<span class="label">Rollouts:</span>';
  cond.rollouts.forEach((r, i) => {
    const labelClass = r.label === 'hack' ? ' hack' : r.label === 'nonsense' ? ' nonsense' : '';
    const active = i === 0 ? ' active' : '';
    html += '<button class="rollout-btn' + labelClass + active + '" onclick="showRollout(' + i + ')" id="rb-' + i + '">' + (i + 1) + '</button>';
  });
  // Legend
  html += '<span style="margin-left:16px;font-size:11px;color:#8b949e">';
  html += '<span style="color:#f85149">\\u25a0</span> hack ';
  html += '<span style="color:#6e7681">\\u25a0</span> nonsense ';
  html += '<span style="color:#21262d">\\u25a0</span> clean';
  html += '</span>';
  html += '</div>';
  nav.innerHTML = html;
}

function showRollout(i) {
  if (currentScenario === null) return;
  const cond = DATA.scenarios[currentScenario].conditions[currentCondition];
  const r = cond.rollouts[i];
  currentRollout = i;

  // Update active button
  document.querySelectorAll('.rollout-btn').forEach(el => el.classList.remove('active'));
  const btn = document.getElementById('rb-' + i);
  if (btn) btn.classList.add('active');

  const rp = document.getElementById('response-panel');
  const labelClass = r.label === 'hack' ? 'label-hack' : r.label === 'nonsense' ? 'label-nonsense' : 'label-clean';
  let html = '<div class="panel-title">Response #' + (i + 1) + ' <span class="' + labelClass + '">[' + r.label.toUpperCase() + ']</span></div>';
  html += '<pre class="response-text">' + highlightToolBlocks(r.response) + '</pre>';
  rp.innerHTML = html;
}

// Keyboard nav
document.addEventListener('keydown', (e) => {
  if (currentScenario === null) return;
  const cond = DATA.scenarios[currentScenario].conditions[currentCondition];
  if (e.key === 'ArrowRight' || e.key === 'j') {
    if (currentRollout < cond.rollouts.length - 1) showRollout(currentRollout + 1);
  } else if (e.key === 'ArrowLeft' || e.key === 'k') {
    if (currentRollout > 0) showRollout(currentRollout - 1);
  } else if (e.key === 'h') {
    // Jump to next hack
    for (let i = currentRollout + 1; i < cond.rollouts.length; i++) {
      if (cond.rollouts[i].label === 'hack') { showRollout(i); break; }
    }
  }
});

renderSidebar();
// Auto-open first scenario
if (DATA.scenarios.length > 0) toggleScenario(0);
</script>
</body>
</html>""")

    output_path = Path(results_file).parent.parent / output_file
    with open(output_path, "w") as f:
        f.write("\n".join(h))
    print(f"Dashboard written to {output_path}")
    print(f"  {len(dashboard_data['scenarios'])} scenarios, {sum(len(s['conditions']) for s in dashboard_data['scenarios'])} conditions")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dashboard.py <results_file> [output.html]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "seed_dashboard.html"
    generate_dashboard(sys.argv[1], out)
