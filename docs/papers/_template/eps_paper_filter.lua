-- eps_paper_filter.lua
--
-- Pandoc Lua filter for the EPS LaTeX-paper -> HTML render path. Resolves the
-- two spike inventions:
--
--   \metric{key}  -> the rendered value string from metrics.json (typed span)
--   \epsref{N}    -> a typed <a data-epsref="N"> the dashboard detects for
--                    hover-preview + new-tab open.
--
-- HOW IT WORKS. pandoc cannot evaluate the paper's real \metric / \epsref
-- definitions (they use \ifcsname/\csname/\href TeX machinery). So the build
-- step injects a pandoc-only OVERRIDE before \begin{document}:
--     \renewcommand{\metric}[1]{<<<METRIC:#1>>>}
--     \renewcommand{\epsref}[1]{<<<EPSREF:#1>>>}
-- pandoc expands those simple macros, producing literal Str sentinels like
-- "<<<METRIC:h1_syco_rho>>>" in the AST. This filter rewrites those Str nodes.
-- The PDF path is unaffected (it uses the real definitions); only the HTML
-- render uses the override + this filter, keeping a single source .tex.
--
-- metrics.json path comes from the METRICS_JSON env var (default metrics.json).
-- A \metric for a missing key renders a loud marker so verify_paper / a human
-- spots it.

local json_path = os.getenv("METRICS_JSON") or "metrics.json"

local metrics = {}
do
  local f = io.open(json_path, "r")
  if f then
    local raw = f:read("*a")
    f:close()
    local ok, decoded = pcall(function() return pandoc.json.decode(raw) end)
    if ok and decoded then metrics = decoded end
  end
end

-- The sentinels can land split across Str/Punct/Space inlines depending on the
-- surrounding text, but in practice pandoc keeps "<<<METRIC:key>>>" as one Str
-- token (no whitespace inside). We rewrite at the Inlines level so a sentinel
-- adjacent to punctuation is still handled, by scanning each Str's text.

local SENT = "<<<(METRIC):([A-Za-z0-9_]+)>>>"
local SENT2 = "<<<(EPSREF):(%d+)>>>"

local function build_metric(key)
  local rec = metrics[key]
  if rec and rec.rendered ~= nil then
    return pandoc.Span(
      { pandoc.Str(tostring(rec.rendered)) },
      pandoc.Attr("", { "eps-metric" }, { ["data-metric-key"] = key }))
  end
  return pandoc.Span(
    { pandoc.Str("??" .. key .. "??") },
    pandoc.Attr("", { "eps-metric-missing" }, {}))
end

local function build_epsref(num)
  return pandoc.Link(
    { pandoc.Str("#" .. num) },
    "/tasks/" .. num,
    "",
    -- rel="noopener" guards against reverse tabnabbing on the new-tab open.
    pandoc.Attr(
      "",
      { "eps-ref" },
      { ["data-epsref"] = num, target = "_blank", rel = "noopener" }))
end

-- Resolve sentinels that land INSIDE a math node (e.g. $\rho = \metric{b}$).
-- Pandoc keeps the whole math body as a raw LaTeX string, so the sentinel is
-- not a separate Str — we substitute the rendered value/text directly into the
-- math source. \epsref inside math is unusual (links don't belong in math), so
-- we resolve \metric here and leave \epsref to the Str/text path.
function Math(el)
  if not el.text:find("<<<METRIC:") then return nil end
  el.text = el.text:gsub("<<<METRIC:([A-Za-z0-9_]+)>>>", function(key)
    local rec = metrics[key]
    if rec and rec.rendered ~= nil then
      -- braces guard multi-char values as one math atom
      return "{" .. tostring(rec.rendered) .. "}"
    end
    return "??" .. key .. "??"
  end)
  return el
end

-- Rewrite a single Str whose text may contain one or more sentinels.
function Str(el)
  local text = el.text
  if not text:find("<<<") then return nil end
  local out = {}
  local pos = 1
  local n = #text
  while pos <= n do
    local sm, em, _, key = text:find(SENT, pos)
    local se, ee, _, num = text:find(SENT2, pos)
    local pick
    if sm and se then pick = (sm <= se) and "m" or "e"
    elseif sm then pick = "m"
    elseif se then pick = "e"
    else break end
    if pick == "m" then
      if sm > pos then table.insert(out, pandoc.Str(text:sub(pos, sm - 1))) end
      table.insert(out, build_metric(key))
      pos = em + 1
    else
      if se > pos then table.insert(out, pandoc.Str(text:sub(pos, se - 1))) end
      table.insert(out, build_epsref(num))
      pos = ee + 1
    end
  end
  if #out == 0 then return nil end
  if pos <= n then table.insert(out, pandoc.Str(text:sub(pos))) end
  return out
end
