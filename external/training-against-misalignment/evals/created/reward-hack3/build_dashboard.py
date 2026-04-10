#!/usr/bin/env python3
"""Build a portable RH3 rollout browser: HTML + per-model JSON + scenarios + tiny server."""
import json, sys, os, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_seeds import get_item_decision, is_hack
from compute_table import classify_response, load_results
from scenarios import SCENARIOS, make_system_prompt, make_user_prompt, CONDITIONS, _condition_key

MAX_ROLLOUTS = 5
MAX_RESPONSE_LEN = 4000


def build_scenarios_data():
    """Build scenario metadata including prompts for all 36 conditions."""
    scenarios_data = {}
    for s in SCENARIOS:
        conditions = {}
        for gv, sp, obs, tp in CONDITIONS:
            ckey = _condition_key(gv, sp, obs, tp)
            sys_prompt = make_system_prompt(s, gv, obs)
            user_prompt = make_user_prompt(s, sp, tp)
            conditions[ckey] = {
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
            }
        scenarios_data[s["id"]] = {
            "name": s["name"],
            "company": s["company"],
            "role": s["role"],
            "metric_name": s["metric_name"],
            "metric_target": s["metric_target"],
            "metric_current": s["metric_current"],
            "predecessor": s["predecessor"],
            "tool_format": s["tool_format"],
            "items_prompt": s["items_prompt"],
            "ground_truth": s["ground_truth"],
            "diagnostic_items": s["diagnostic_items"],
            "diagnostic_hack_actions": s["diagnostic_hack_actions"],
            "diagnostic_description": s["diagnostic_description"],
            "conditions": conditions,
        }
    return scenarios_data


def compact_model(data):
    results = []
    for entry in data["results"]:
        rollouts = []
        for ro in entry["rollouts"][:MAX_ROLLOUTS]:
            raw = ro["raw_response"]
            label = classify_response(raw, entry["scenario_id"],
                                      entry["diagnostic_items"],
                                      entry["diagnostic_hack_actions"])
            items = {}
            for item_id in entry["diagnostic_items"]:
                dec = get_item_decision(raw, item_id, entry["scenario_id"])
                items[item_id] = {"decision": dec, "is_hack": is_hack(dec, entry["diagnostic_hack_actions"])}
            resp = raw if len(raw) <= MAX_RESPONSE_LEN else raw[:MAX_RESPONSE_LEN] + "\n\n... [TRUNCATED — " + str(len(raw)) + " chars total]"
            rollouts.append({"id": ro["rollout_id"], "response": resp, "label": label, "items": items})
        results.append({
            "scenario_id": entry["scenario_id"],
            "scenario_name": entry["scenario_name"],
            "condition": entry["condition"],
            "goal_value": entry["goal_value"],
            "self_preservation": entry["self_preservation"],
            "observability": entry["observability"],
            "diagnostic_items": entry["diagnostic_items"],
            "hack_actions": entry["diagnostic_hack_actions"],
            "diagnostic_description": entry.get("diagnostic_description", ""),
            "ground_truth": entry.get("ground_truth", {}),
            "rollouts": rollouts,
        })
    return {"model_path": data["model_path"], "results": results}


SERVER_PY = r'''#!/usr/bin/env python3
"""Tiny server for RH3 dashboard. Run: python serve.py"""
import http.server, os, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

print(f"RH3 Dashboard: http://localhost:{PORT}/index.html")
http.server.HTTPServer(('', PORT), Handler).serve_forever()
'''


INDEX_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>RH3 Rollout Browser</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.header h1 { font-size: 18px; color: #58a6ff; white-space: nowrap; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
select { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px; font-size: 13px; }
select:focus { border-color: #58a6ff; outline: none; }
.loading { color: #d29922; font-size: 13px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-hack { background: #da3633; color: #fff; }
.badge-clean { background: #238636; color: #fff; }
.badge-nonsense { background: #6e7681; color: #fff; }
.badge-incomplete { background: #d29922; color: #fff; }
.main { display: flex; height: calc(100vh - 52px); }
.sidebar { width: 360px; min-width: 260px; border-right: 1px solid #30363d; overflow-y: auto; background: #161b22; flex-shrink: 0; }
.rollout-item { padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #21262d; font-size: 13px; }
.rollout-item:hover { background: #1c2128; }
.rollout-item.active { background: #1f6feb33; border-left: 3px solid #58a6ff; }
.rollout-item .ri-label { font-size: 11px; color: #8b949e; }
.content { flex: 1; overflow-y: auto; padding: 20px; }
.box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.box h2 { font-size: 15px; color: #58a6ff; margin-bottom: 8px; }
.box h3 { font-size: 13px; color: #d2a8ff; margin: 12px 0 6px; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 6px; font-size: 13px; }
.meta-grid .label { color: #8b949e; }
.meta-grid .value { color: #c9d1d9; font-weight: 500; }
.prompt-block { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.5; max-height: 400px; overflow-y: auto; margin-top: 6px; }
.prompt-block.collapsed { max-height: 120px; overflow: hidden; position: relative; }
.prompt-block.collapsed::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 40px; background: linear-gradient(transparent, #0d1117); }
.toggle-btn { background: #21262d; color: #58a6ff; border: 1px solid #30363d; border-radius: 4px; padding: 4px 10px; font-size: 11px; cursor: pointer; margin-top: 4px; }
.toggle-btn:hover { background: #30363d; }
table.items { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
table.items th { text-align: left; color: #8b949e; padding: 4px 8px; border-bottom: 1px solid #30363d; }
table.items td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
.item-hack { color: #f85149; font-weight: 600; }
.item-clean { color: #3fb950; }
.item-none { color: #6e7681; font-style: italic; }
.response-box { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12.5px; line-height: 1.6; }
.response-box .tool-block { background: #1c2128; border-left: 3px solid #58a6ff; padding: 4px 8px; margin: 4px 0; display: block; }
.stats-bar { display: flex; gap: 12px; padding: 8px 12px; background: #0d1117; border-bottom: 1px solid #30363d; font-size: 12px; color: #8b949e; flex-wrap: wrap; }
.stats-bar .stat { display: flex; align-items: center; gap: 4px; }
.filter-row { display: flex; gap: 6px; padding: 8px; background: #0d1117; border-bottom: 1px solid #30363d; flex-wrap: wrap; }
.filter-btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer; }
.filter-btn.active { background: #1f6feb; border-color: #58a6ff; }
.filter-btn:hover { border-color: #58a6ff; }
.empty-state { text-align: center; padding: 60px 20px; color: #8b949e; }
.empty-state h2 { font-size: 20px; margin-bottom: 8px; }
.rubric-highlight { background: #da363322; border: 1px solid #da363366; border-radius: 4px; padding: 8px 12px; margin-top: 8px; font-size: 13px; }
.rubric-highlight b { color: #f85149; }
.gt-row-hack { background: #da363315; }
.tab-bar { display: flex; gap: 0; border-bottom: 1px solid #30363d; margin-bottom: 12px; }
.tab { padding: 8px 16px; cursor: pointer; font-size: 13px; color: #8b949e; border-bottom: 2px solid transparent; }
.tab:hover { color: #c9d1d9; }
.tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
</style>
</head><body>
<div class="header">
  <h1>RH3 Rollout Browser</h1>
  <div class="controls">
    <select id="model-select"><option>Loading...</option></select>
    <select id="scenario-select"></select>
    <select id="condition-select"></select>
    <span class="loading" id="loading"></span>
  </div>
</div>
<div class="main">
  <div class="sidebar">
    <div class="stats-bar" id="stats-bar"></div>
    <div class="filter-row" id="filter-row"></div>
    <div id="rollout-list"></div>
  </div>
  <div class="content" id="content">
    <div class="empty-state"><h2>Loading...</h2><p>Fetching model list</p></div>
  </div>
</div>
<script>
let modelIndex = {};
let scenariosData = {};
let currentData = null;
let currentFilter = 'all';
let activeTab = 'response';

const $ = id => document.getElementById(id);
const modelSelect = $('model-select');
const scenarioSelect = $('scenario-select');
const conditionSelect = $('condition-select');
const rolloutList = $('rollout-list');
const contentEl = $('content');
const statsBar = $('stats-bar');
const filterRow = $('filter-row');
const loadingEl = $('loading');

// Load index + scenarios
Promise.all([
  fetch('models_index.json').then(r => r.json()),
  fetch('scenarios.json').then(r => r.json()),
]).then(([idx, scen]) => {
  modelIndex = idx;
  scenariosData = scen;
  modelSelect.innerHTML = '';
  Object.keys(idx).sort().forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    modelSelect.appendChild(o);
  });
  loadModel(modelSelect.value);
}).catch(() => {
  contentEl.innerHTML = '<div class="empty-state"><h2>Failed to load</h2><p>Run: python serve.py</p></div>';
});

function loadModel(name) {
  const file = modelIndex[name];
  if (!file) return;
  loadingEl.textContent = 'Loading ' + name + '...';
  fetch('data/' + file).then(r => r.json()).then(data => {
    currentData = data;
    loadingEl.textContent = '';
    updateScenarios();
  });
}

function getScenarios() {
  const seen = new Set();
  currentData.results.forEach(r => seen.add(r.scenario_id));
  return [...seen].sort();
}
function getConditions(scenario) {
  const seen = new Set();
  currentData.results.filter(r => r.scenario_id === scenario).forEach(r => seen.add(r.condition));
  return [...seen].sort();
}
function getEntry(scenario, condition) {
  return currentData.results.find(r => r.scenario_id === scenario && r.condition === condition);
}

function updateScenarios() {
  scenarioSelect.innerHTML = '';
  getScenarios().forEach(s => {
    const o = document.createElement('option');
    o.value = s; o.textContent = s.replace('rh_', '');
    scenarioSelect.appendChild(o);
  });
  updateConditions();
}
function updateConditions() {
  conditionSelect.innerHTML = '';
  getConditions(scenarioSelect.value).forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.textContent = c;
    conditionSelect.appendChild(o);
  });
  currentFilter = 'all';
  updateRollouts();
}

function updateRollouts() {
  const entry = getEntry(scenarioSelect.value, conditionSelect.value);
  if (!entry) { rolloutList.innerHTML = ''; return; }

  const counts = {hack:0, clean:0, nonsense:0, incomplete:0};
  entry.rollouts.forEach(r => counts[r.label]++);
  const total = entry.rollouts.length;

  statsBar.innerHTML = ['hack','clean','nonsense','incomplete'].map(l =>
    `<span class="stat"><span class="badge badge-${l}">${l}</span> ${counts[l]}/${total}</span>`
  ).join('');

  filterRow.innerHTML = '';
  ['all','hack','clean','nonsense','incomplete'].forEach(f => {
    const btn = document.createElement('button');
    btn.className = 'filter-btn' + (currentFilter === f ? ' active' : '');
    btn.textContent = f === 'all' ? `all (${total})` : `${f} (${counts[f]})`;
    btn.onclick = () => { currentFilter = f; updateRollouts(); };
    filterRow.appendChild(btn);
  });

  rolloutList.innerHTML = '';
  entry.rollouts.forEach(ro => {
    if (currentFilter !== 'all' && ro.label !== currentFilter) return;
    const div = document.createElement('div');
    div.className = 'rollout-item';
    const itemSummary = Object.entries(ro.items).map(([id, v]) =>
      v.is_hack ? `<span style="color:#f85149">${id}:${v.decision||'?'}</span>` :
      v.decision ? `<span style="color:#3fb950">${id}:${v.decision}</span>` :
      `<span style="color:#6e7681">${id}:?</span>`
    ).join(' ');
    div.innerHTML = `<div><span>#${ro.id} <span class="badge badge-${ro.label}">${ro.label}</span></span><br><span class="ri-label">${itemSummary}</span></div>`;
    div.onclick = () => showRollout(entry, ro, div);
    rolloutList.appendChild(div);
  });

  if (rolloutList.firstChild) rolloutList.firstChild.click();
  else contentEl.innerHTML = '<div class="empty-state"><h2>No rollouts match filter</h2></div>';
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function highlightResponse(raw, diagItems) {
  let h = esc(raw);
  h = h.replace(/(&lt;tool_use:\w+&gt;)([\s\S]*?)(&lt;\/tool_use:\w+&gt;)/g, '<span class="tool-block">$1$2$3</span>');
  diagItems.forEach(id => {
    const re = new RegExp('\\b(' + id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')\\b', 'g');
    h = h.replace(re, '<b style="color:#ffa657">$1</b>');
  });
  return h;
}

function showRollout(entry, ro, el) {
  document.querySelectorAll('.rollout-item.active').forEach(e => e.classList.remove('active'));
  el.classList.add('active');

  const sid = entry.scenario_id;
  const cond = entry.condition;
  const scen = scenariosData[sid] || {};
  const condPrompts = (scen.conditions || {})[cond] || {};

  // Item decisions table
  const itemRows = Object.entries(ro.items).map(([id, v]) => {
    const cls = v.is_hack ? 'item-hack' : v.decision ? 'item-clean' : 'item-none';
    return `<tr class="${v.is_hack ? 'gt-row-hack' : ''}"><td>${id}</td><td class="${cls}">${v.decision || 'not found'}</td><td>${v.is_hack ? 'HACK' : ''}</td></tr>`;
  }).join('');

  // Ground truth table
  const gt = entry.ground_truth || {};
  const diagSet = new Set(entry.diagnostic_items);
  let gtRows = '';
  for (const [action, ids] of Object.entries(gt)) {
    for (const id of ids) {
      const isDiag = diagSet.has(id);
      gtRows += `<tr class="${isDiag ? 'gt-row-hack' : ''}"><td>${id}${isDiag ? ' ⚠️' : ''}</td><td>${action}</td><td>${isDiag ? 'DIAGNOSTIC' : ''}</td></tr>`;
    }
  }

  contentEl.innerHTML = `
    <div class="tab-bar">
      <div class="tab ${activeTab==='response'?'active':''}" onclick="switchTab('response')">Response</div>
      <div class="tab ${activeTab==='prompt'?'active':''}" onclick="switchTab('prompt')">Prompt</div>
      <div class="tab ${activeTab==='rubric'?'active':''}" onclick="switchTab('rubric')">Rubric</div>
    </div>

    <div id="tab-response" style="display:${activeTab==='response'?'block':'none'}">
      <div class="box">
        <h2>${entry.scenario_name} — #${ro.id} <span class="badge badge-${ro.label}">${ro.label}</span></h2>
        <div class="meta-grid">
          <div><span class="label">Condition:</span> <span class="value">${cond}</span></div>
          <div><span class="label">Goal:</span> <span class="value">${entry.goal_value}</span></div>
          <div><span class="label">Self-pres:</span> <span class="value">${entry.self_preservation}</span></div>
          <div><span class="label">Observability:</span> <span class="value">${entry.observability}</span></div>
          <div><span class="label">Hack actions:</span> <span class="value">${entry.hack_actions.join(', ')}</span></div>
        </div>
        <h3>Diagnostic Item Decisions</h3>
        <table class="items"><tr><th>Item</th><th>Model Decision</th><th>Hack?</th></tr>${itemRows}</table>
      </div>
      <div class="response-box">${highlightResponse(ro.response, entry.diagnostic_items)}</div>
    </div>

    <div id="tab-prompt" style="display:${activeTab==='prompt'?'block':'none'}">
      <div class="box">
        <h2>Prompt — ${entry.scenario_name}</h2>
        <div class="meta-grid">
          <div><span class="label">Condition:</span> <span class="value">${cond}</span></div>
          <div><span class="label">Goal:</span> <span class="value">${entry.goal_value}</span></div>
          <div><span class="label">Self-pres:</span> <span class="value">${entry.self_preservation}</span></div>
          <div><span class="label">Observability:</span> <span class="value">${entry.observability}</span></div>
        </div>
        <h3>System Prompt</h3>
        <div class="prompt-block">${esc(condPrompts.system_prompt || 'N/A')}</div>
        <h3>User Prompt</h3>
        <div class="prompt-block">${esc(condPrompts.user_prompt || 'N/A')}</div>
      </div>
    </div>

    <div id="tab-rubric" style="display:${activeTab==='rubric'?'block':'none'}">
      <div class="box">
        <h2>Grading Rubric — ${entry.scenario_name}</h2>
        <div class="meta-grid">
          <div><span class="label">Company:</span> <span class="value">${scen.company || ''}</span></div>
          <div><span class="label">Role:</span> <span class="value">${scen.role || ''}</span></div>
          <div><span class="label">Metric:</span> <span class="value">${scen.metric_name || ''}</span></div>
          <div><span class="label">Target:</span> <span class="value">${scen.metric_target || ''}</span></div>
          <div><span class="label">Current:</span> <span class="value">${scen.metric_current || ''}</span></div>
          <div><span class="label">Predecessor:</span> <span class="value">${scen.predecessor || ''}</span></div>
        </div>

        <div class="rubric-highlight">
          <b>What constitutes a hack:</b><br>
          ${esc(entry.diagnostic_description || scen.diagnostic_description || '')}
        </div>

        <h3>Diagnostic Items (items checked for hacking)</h3>
        <p style="font-size:13px;color:#f85149;margin-bottom:4px">${entry.diagnostic_items.join(', ')} — hack if decision is: <b>${entry.hack_actions.join(' / ')}</b></p>

        <h3>Ground Truth (correct decisions for ALL items)</h3>
        <table class="items"><tr><th>Item</th><th>Correct Decision</th><th>Role</th></tr>${gtRows}</table>

        <h3>Tool Format</h3>
        <div class="prompt-block collapsed" id="tool-format-block">${esc(scen.tool_format || '')}</div>
        <button class="toggle-btn" onclick="toggleBlock('tool-format-block')">Show/Hide</button>

        <h3>Items Prompt (the actual items to review)</h3>
        <div class="prompt-block collapsed" id="items-prompt-block">${esc(scen.items_prompt || '')}</div>
        <button class="toggle-btn" onclick="toggleBlock('items-prompt-block')">Show/Hide</button>
      </div>
    </div>
  `;
}

window.switchTab = function(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.textContent.toLowerCase() === tab));
  ['response','prompt','rubric'].forEach(t => {
    const el = document.getElementById('tab-' + t);
    if (el) el.style.display = t === tab ? 'block' : 'none';
  });
};

window.toggleBlock = function(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('collapsed');
};

modelSelect.onchange = () => loadModel(modelSelect.value);
scenarioSelect.onchange = updateConditions;
conditionSelect.onchange = () => { currentFilter = 'all'; updateRollouts(); };
</script>
</body></html>"""


def main():
    out_dir = "/tmp/rh3_dashboard"
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Build scenarios data (prompts + rubrics)
    print("Building scenario metadata...")
    scenarios_data = build_scenarios_data()
    with open(os.path.join(out_dir, "scenarios.json"), "w") as f:
        json.dump(scenarios_data, f, ensure_ascii=False)
    print(f"  {len(scenarios_data)} scenarios with prompts for {len(CONDITIONS)} conditions each")

    # Load and compact model results
    print("Loading results...")
    models = load_results("results/")
    print(f"  {len(models)} models")

    index = {}
    for short_name, data in sorted(models.items()):
        print(f"  Processing {short_name}...")
        compact = compact_model(data)
        fname = short_name.replace("/", "_") + ".json"
        with open(os.path.join(data_dir, fname), "w") as f:
            json.dump(compact, f, ensure_ascii=False)
        index[short_name] = fname

    with open(os.path.join(out_dir, "models_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(INDEX_HTML)

    with open(os.path.join(out_dir, "serve.py"), "w") as f:
        f.write(SERVER_PY)

    # Zip
    zip_path = "/tmp/rh3_dashboard.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                full = os.path.join(root, file)
                arc = os.path.relpath(full, "/tmp")
                zf.write(full, arc)

    total = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\nDone! {zip_path} ({total:.1f} MB)")
    print(f"scp -P 15115 jens@198.145.108.51:{zip_path} ~/Downloads/")
    print(f"\nThen: cd rh3_dashboard && python serve.py")
    print(f"Open: http://localhost:8888/index.html")


if __name__ == "__main__":
    main()
