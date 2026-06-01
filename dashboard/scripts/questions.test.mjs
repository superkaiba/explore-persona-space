/**
 * Parser + linkify + coverage tests for the /questions surface.
 *
 * Standalone (no test runner is configured here). Runs under tsx so the
 * `.ts` imports resolve. Run with:
 *   npx tsx scripts/questions.test.mjs
 *   # or: npm run test:questions
 *
 * Coverage:
 *   1. parseQuestionsFromMarkdown against the shared fixture corpus
 *      (split-line carrier, *Next:* between Belief and Evidence, H4-nested
 *      anchor, app bullet with #N, injected changelog block,
 *      `none in-house yet (#428)` parenthetical).
 *   2. linkifyEvidenceInOpenQuestions on the same fixture — every evidence
 *      `#N` on a carrier line becomes a markdown link; app bullets'
 *      inline `#N` stay raw; changelog `#N` stays raw; prose stays raw.
 *   3. Coverage assertion against the LIVE docs/open_questions.md: every
 *      anchor lands in exactly one group (no hardcoded counts); every
 *      `listPublicResults()` id appears in >=1 question's evidence
 *      (failures reported, not auto-fixed — the parser is read-only).
 *
 * Exits non-zero on the first failed assertion.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  parseQuestionsFromMarkdown,
} from "../lib/questions.ts";
import {
  linkifyEvidenceInOpenQuestions,
} from "../lib/linkify-evidence.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = path.join(
  __dirname,
  "__fixtures__",
  "open_questions.fixture.md",
);
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const LIVE_DOC = path.join(REPO_ROOT, "docs", "open_questions.md");

let failures = 0;
function check(name, cond, extra) {
  const ok = Boolean(cond);
  if (!ok) {
    failures++;
    console.log(`FAIL  ${name}`);
    if (extra !== undefined) console.log(`      ${extra}`);
  } else {
    console.log(`PASS  ${name}`);
  }
}

// ── 1. Parse the fixture ────────────────────────────────────────────────────
const fixture = fs.readFileSync(FIXTURE_PATH, "utf8");
const parsed = parseQuestionsFromMarkdown(fixture);
const byId = new Map(parsed.map((q) => [q.id, q]));

console.log("\n=== parseQuestionsFromMarkdown (fixture) ===");

// The fixture intentionally puts a `<!-- q:not-a-question -->` anchor under
// Motivation (an "other" region) — it MUST be skipped, NOT parsed.
check(
  "Motivation-region anchor is NOT parsed as a question",
  !byId.has("not-a-question"),
);

// Plain one-line case.
{
  const q = byId.get("plain-one-line");
  check("q:plain-one-line is a question", q && q.kind === "question");
  check(
    "q:plain-one-line confidence=LOW",
    q && q.confidence === "LOW",
  );
  check(
    "q:plain-one-line evidence=[100,101]",
    q && JSON.stringify(q.evidence) === JSON.stringify([100, 101]),
    q && `actual=${JSON.stringify(q.evidence)}`,
  );
  check(
    "q:plain-one-line section=Distance",
    q && q.section === "Distance between contexts",
  );
  check(
    "q:plain-one-line number=1.1",
    q && q.number === "1.1",
  );
}

// Split-line carrier with *Next:* between Belief and Evidence (q3.1-style).
{
  const q = byId.get("split-line-carrier");
  check("q:split-line-carrier parsed", !!q);
  check(
    "q:split-line-carrier confidence=MODERATE (from later blockquote)",
    q && q.confidence === "MODERATE",
  );
  check(
    "q:split-line-carrier evidence=[110,111,112]",
    q && JSON.stringify(q.evidence) === JSON.stringify([110, 111, 112]),
    q && `actual=${JSON.stringify(q.evidence)}`,
  );
  check(
    "q:split-line-carrier belief recovered from earlier blockquote",
    q && typeof q.belief === "string" && q.belief.length > 0,
  );
  check(
    "q:split-line-carrier next recovered from *Next:* rider",
    q && typeof q.next === "string" && /experiment/.test(q.next),
  );
}

// Empty-evidence sentinel: REPLACE path means parser returns [].
{
  const q = byId.get("empty-bare");
  check("q:empty-bare parsed", !!q);
  check("q:empty-bare evidence is empty []", q && q.evidence.length === 0);
}

// Empty-evidence sentinel + parenthetical aside with `#428` —
// must NOT parse #428 as evidence.
{
  const q = byId.get("empty-parenthetical");
  check("q:empty-parenthetical parsed", !!q);
  check(
    "q:empty-parenthetical evidence is empty (parenthetical #428 ignored)",
    q && q.evidence.length === 0,
    q && `actual=${JSON.stringify(q.evidence)}`,
  );
}

// H4-nested anchor — subsection field populated.
{
  const q = byId.get("h4-nested");
  check("q:h4-nested parsed", !!q);
  check(
    "q:h4-nested subsection populated",
    q && typeof q.subsection === "string" && q.subsection.length > 0,
  );
  check(
    "q:h4-nested still landed in §1 (Distance) — not §2 by accident",
    q && q.section === "Distance between contexts",
  );
}

// H3 transition resets the H4 cursor.
{
  const q = byId.get("reset-h3");
  check("q:reset-h3 parsed", !!q);
  check(
    "q:reset-h3 subsection is reset on new H3 (no leaked H4 from §1)",
    q && q.subsection === undefined,
  );
  check(
    "q:reset-h3 section is the new H3 name",
    q && /Updating/.test(q.section),
  );
}

// Apps: kind=application, evidence empty (inline `#N` is prose), free-text
// appStatus captured.
{
  const a1 = byId.get("app1");
  const a2 = byId.get("app2");
  check("q:app1 parsed as application", a1 && a1.kind === "application");
  check("q:app1 section=Applications", a1 && a1.section === "Applications");
  check(
    "q:app1 contributes ZERO evidence edges (inline #100 is prose)",
    a1 && a1.evidence.length === 0,
  );
  check(
    "q:app1 appStatus captured verbatim",
    a1 && a1.appStatus === "falsification risk",
    a1 && `actual=${JSON.stringify(a1.appStatus)}`,
  );
  check("q:app2 parsed as application", a2 && a2.kind === "application");
  check(
    "q:app2 evidence empty",
    a2 && a2.evidence.length === 0,
  );
}

// Coverage: every anchor lands in EXACTLY ONE group (no double-counting,
// no hardcoded counts). Apps go to Applications; questions to the H3 they
// sit under or to "Settled". The Motivation anchor is intentionally
// skipped, so the count is parsed-vs-anchored.
{
  const anchorIds = Array.from(
    fixture.matchAll(/<!--\s*q:([A-Za-z0-9_.\-]+)\s*-->/g),
  ).map((m) => m[1].toLowerCase());
  // The Motivation anchor is the only one expected to NOT appear.
  const expectedParsed = anchorIds.filter((id) => id !== "not-a-question");
  const parsedIds = new Set(parsed.map((q) => q.id));
  check(
    "Every non-skipped anchor parsed exactly once (no dupes, no drops)",
    expectedParsed.length === parsed.length &&
      expectedParsed.every((id) => parsedIds.has(id)),
    `expected ${expectedParsed.length} parsed, got ${parsed.length}`,
  );
}

// ── 2. Linkify on the same fixture ──────────────────────────────────────────
console.log("\n=== linkifyEvidenceInOpenQuestions (fixture) ===");
{
  // Pretend #100 is public, #101 is not, #110 + #111 + #112 not, #120 public,
  // #130 not. This lets us assert both branches of evidenceHrefForTaskId.
  const publicIds = new Set([100, 120]);
  const out = linkifyEvidenceInOpenQuestions(fixture, publicIds);

  // Evidence carrier line for q:plain-one-line: #100 should link to /results/100,
  // #101 to /tasks/101.
  check(
    "carrier #100 (public) linked to /results/100",
    /\[#100\]\(\/results\/100\)/.test(out),
  );
  check(
    "carrier #101 (gated) linked to /tasks/101",
    /\[#101\]\(\/tasks\/101\)/.test(out),
  );

  // Split-line carrier: only the **Evidence:** ROW is linkified, not the
  // Belief or Next rows; the carrier row contains #110/#111/#112.
  check(
    "split-line carrier #110 linked",
    /\[#110\]\(\/tasks\/110\)/.test(out),
  );
  check(
    "split-line carrier #112 linked",
    /\[#112\]\(\/tasks\/112\)/.test(out),
  );

  // H4-nested carrier: #120 public.
  check(
    "h4-nested carrier #120 linked to /results/120",
    /\[#120\]\(\/results\/120\)/.test(out),
  );

  // App bullets: their inline `#100` / `#110` are NOT in a blockquote, so
  // they MUST remain as raw `#100` / `#110` text — NOT inside any `[#N](...)`.
  // Find the Applications section verbatim and assert no link wraps #100/#110
  // inside its bullets.
  const appsBlock = out.slice(
    out.indexOf("## Applications"),
    out.indexOf("## Settled"),
  );
  check(
    "App bullets do NOT linkify inline #N (apps contribute zero edges)",
    !/\[#100\]/.test(appsBlock) && !/\[#110\]/.test(appsBlock),
    "found a linkified app #N",
  );

  // Changelog block: #999 must NOT be linkified. The block is at the top.
  const changelogBlock = out.slice(
    out.indexOf("<!-- living-docs-changelog:begin -->"),
    out.indexOf("<!-- living-docs-changelog:end -->"),
  );
  check(
    "Changelog block #999 not linkified",
    !/\[#999\]/.test(changelogBlock),
  );

  // Prose: Central question / Framing references must NOT be linkified.
  const beforeFirstAnchor = out.slice(0, out.indexOf("## Open questions"));
  check(
    "Prose `#999` / `#100` outside blockquote NOT linkified",
    !/\[#100\]\(/.test(beforeFirstAnchor) && !/\[#999\]\(/.test(beforeFirstAnchor),
  );

  // *Next:* line — it's a blockquote line BUT lacks `**Evidence:**` and is
  // not a State line, so it must NOT be linkified. The split-line case
  // has a *Next:* rider; if it carried any #N it would stay raw. (Our
  // fixture's *Next:* has no #N, but the live doc's q3.9 does.) Assert
  // the carrier marker logic by checking the *Next:* rider line shape.
  const splitBlock = out.slice(
    out.indexOf("q:split-line-carrier"),
    out.indexOf("q:empty-bare"),
  );
  check(
    "*Next:* rider line preserved (no synthesized brackets)",
    /\*Next:\s*do the experiment/.test(splitBlock),
  );

  // Empty-sentinel carrier WITH a parenthetical `#428`: the value is
  // structurally empty (parseEvidence returns []), so linkify MUST leave
  // #428 raw — otherwise /docs/open_questions links an id the /questions hub
  // shows no evidence row for. Regression guard for the parser/linkify split.
  check(
    "empty-sentinel parenthetical #428 NOT linkified",
    /none in-house yet \(definitional groundwork tracked in #428\)/.test(out) &&
      !/\[#428\]/.test(out),
    "found a linkified #428 inside an empty-evidence sentinel",
  );
}

// ── 3. Coverage on the LIVE docs/open_questions.md ──────────────────────────
console.log("\n=== live docs/open_questions.md coverage ===");
let live;
try {
  live = fs.readFileSync(LIVE_DOC, "utf8");
} catch (e) {
  console.log(`SKIP  live doc not readable at ${LIVE_DOC}: ${e.message}`);
}
if (live) {
  const liveQs = parseQuestionsFromMarkdown(live);

  // Anchor count is DERIVED, never hardcoded. Two cases:
  //   (a) the changelog block hasn't been injected yet (today) — every
  //       anchor in the file is a question or app, so they must all parse.
  //   (b) someday the changelog block holds anchors; those should be
  //       stripped. Verify by re-parsing AFTER stripping the block.
  const stripped = live.replace(
    /<!-- living-docs-changelog:begin -->[\s\S]*?<!-- living-docs-changelog:end -->/,
    "",
  );
  const strippedAnchors = Array.from(
    stripped.matchAll(/<!--\s*q:([A-Za-z0-9_.\-]+)\s*-->/g),
  ).map((m) => m[1].toLowerCase());
  // Every anchor outside the changelog must be parsed; the parser may
  // legitimately skip Motivation/Framing/Glossary anchors (none today, but
  // tolerated). Assert strict equality on the count.
  const parsedIds = new Set(liveQs.map((q) => q.id));
  const unparsed = strippedAnchors.filter((id) => !parsedIds.has(id));
  check(
    `live doc — every non-changelog anchor parsed (anchors=${strippedAnchors.length} parsed=${parsedIds.size})`,
    unparsed.length === 0,
    unparsed.length > 0 ? `unparsed: ${unparsed.join(", ")}` : undefined,
  );

  // Every anchor lands in exactly one group: assert no question id appears
  // twice across the parsed list.
  const dups = liveQs
    .map((q) => q.id)
    .filter((id, i, arr) => arr.indexOf(id) !== i);
  check(
    "live doc — every parsed anchor unique (no group double-counting)",
    dups.length === 0,
    dups.length > 0 ? `dup ids: ${dups.join(", ")}` : undefined,
  );

  // Apps contribute zero evidence edges (defensive: the live doc must
  // honor this too).
  const apps = liveQs.filter((q) => q.kind === "application");
  const appWithEdges = apps.find((a) => a.evidence.length > 0);
  check(
    `live doc — Applications contribute zero edges (n=${apps.length})`,
    appWithEdges === undefined,
    appWithEdges
      ? `app ${appWithEdges.id} has ${appWithEdges.evidence.length} evidence ids`
      : undefined,
  );

  // Public-result coverage: every id in listPublicResults() must appear
  // in >=1 question's evidence. Read the registry + body.md directly
  // (mirrors lib/results.publicResultIdSet but writing to plain Node).
  const REGISTRY_PATH = path.join(REPO_ROOT, "tasks", "REGISTRY.json");
  let publicIds = new Set();
  try {
    const reg = JSON.parse(fs.readFileSync(REGISTRY_PATH, "utf8"));
    for (const [idStr, entry] of Object.entries(reg.tasks || {})) {
      if (entry.status !== "completed") continue;
      const id = Number(idStr);
      if (!Number.isFinite(id)) continue;
      const bodyPath = path.join(REPO_ROOT, entry.path, "body.md");
      let body;
      try {
        body = fs.readFileSync(bodyPath, "utf8");
      } catch {
        continue;
      }
      const fm = parseFrontmatter(body);
      const cls =
        typeof fm.classification === "string" ? fm.classification.trim() : "";
      if (cls !== "useful") continue;
      const tags = Array.isArray(fm.tags) ? fm.tags : [];
      if (tags.includes("format-exemplar")) continue;
      publicIds.add(id);
    }
  } catch (e) {
    console.log(`SKIP  registry not readable: ${e.message}`);
    publicIds = null;
  }
  if (publicIds) {
    const evidenced = new Set();
    for (const q of liveQs) {
      if (q.kind === "application") continue;
      for (const id of q.evidence) evidenced.add(id);
    }
    const uncovered = Array.from(publicIds).filter((id) => !evidenced.has(id));
    // REPORT as a SOFT failure (the parser is read-only — fixing requires
    // an `apply()` patch to the doc and a user-confirmed change). The
    // PASS/FAIL distinction below tells the operator whether the doc is
    // already coherent.
    check(
      `live doc — every public result id evidenced (${publicIds.size} public, ${uncovered.length} uncovered)`,
      uncovered.length === 0,
      uncovered.length > 0
        ? `uncovered public ids: ${uncovered
            .slice(0, 20)
            .map((i) => `#${i}`)
            .join(", ")}${uncovered.length > 20 ? ` ... +${uncovered.length - 20} more` : ""}`
        : undefined,
    );
  }
}

console.log("");
if (failures > 0) {
  console.error(`${failures} assertion(s) FAILED`);
  process.exit(1);
}
console.log("All questions assertions passed.");

/**
 * Minimal YAML frontmatter extractor (top-level scalars + simple arrays
 * only — enough to read `classification` and `tags` for the coverage
 * check). Avoids pulling gray-matter into a test script.
 */
function parseFrontmatter(text) {
  if (!text.startsWith("---\n")) return {};
  const end = text.indexOf("\n---\n", 4);
  if (end === -1) return {};
  const body = text.slice(4, end);
  const out = {};
  let inArray = null;
  for (const line of body.split("\n")) {
    if (inArray) {
      const am = /^\s+-\s+(.*)$/.exec(line);
      if (am) {
        out[inArray].push(am[1].trim().replace(/^['"]|['"]$/g, ""));
        continue;
      }
      inArray = null;
    }
    const kv = /^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/.exec(line);
    if (!kv) continue;
    const key = kv[1];
    const value = kv[2];
    if (value === "" || value === undefined) {
      out[key] = [];
      inArray = key;
      continue;
    }
    out[key] = value.replace(/^['"]|['"]$/g, "");
  }
  return out;
}
