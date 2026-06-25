/**
 * Path-confinement proof for `readTaskFigure` (lib/paper.ts) — the PUBLIC
 * binary figure-serving route (`GET /tasks/<id>/figure/<name>`).
 *
 * No test runner is configured in this dashboard, so this is a standalone
 * script run under tsx (it imports the `.ts` module directly). Run with:
 *
 *   npx tsx scripts/paper-figure-confinement.test.mjs
 *   # or: npm run test:paper-figure
 *
 * The logic is security-critical (an attacker controls `<name>`) and was
 * previously untested; a future refactor could silently reopen traversal. This
 * asserts the allow/deny boundary holds:
 *
 *   REJECT (null / 404):
 *     - `..` parent-dir traversal             (raw, encoded segment, nested)
 *     - absolute paths                        (/etc/passwd)
 *     - multi-segment names                   (sub/dir/x.png)
 *     - backslash separator                   (..\\x.png, sub\\x.png)
 *     - non-image extension                   (x.txt, x.pdf, x, x.png.txt)
 *     - dotfiles                              (.env, .htaccess)
 *     - empty name
 *     - a REAL out-of-repo symlink target     (figures/issue_<N>/evil.png -> /etc/hostname)
 *
 *   SERVE (bytes + content-type):
 *     - a legit figures/issue_<N>/<file>.png
 *     - a legit .svg (allow-list member)
 *
 * Hermetic: builds a throwaway repo tree in a tmpdir and chdirs into its
 * `dashboard/` BEFORE importing paper.ts, so lib/repo.ts's
 * `REPO_ROOT = resolve(cwd, "..")` captures the tmp root (the same shape as the
 * real runtime, where cwd is dashboard/). Exits non-zero on the first failed
 * assertion.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

let failures = 0;
function check(name, cond) {
  const ok = Boolean(cond);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  if (!ok) failures++;
}

// ── Build a hermetic fake repo root ────────────────────────────────────────
const ISSUE = 777;
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "eps-figure-confine-"));
const repoRoot = path.join(tmpRoot, "repo");
const dashDir = path.join(repoRoot, "dashboard");
const figDir = path.join(repoRoot, "figures", `issue_${ISSUE}`);
fs.mkdirSync(dashDir, { recursive: true });
fs.mkdirSync(figDir, { recursive: true });

// A legit PNG + SVG inside the confined dir.
const pngBytes = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]); // PNG magic
fs.writeFileSync(path.join(figDir, "plot.png"), pngBytes);
fs.writeFileSync(path.join(figDir, "chart.svg"), "<svg/>");

// A secret OUTSIDE the repo root + a sibling figures dir for another issue, so
// `..`-escapes and cross-issue reads have something real to (fail to) reach.
const outsideSecret = path.join(tmpRoot, "secret.png");
fs.writeFileSync(outsideSecret, "TOP SECRET");
const otherFigDir = path.join(repoRoot, "figures", "issue_111");
fs.mkdirSync(otherFigDir, { recursive: true });
fs.writeFileSync(path.join(otherFigDir, "other.png"), pngBytes);

// A REAL symlink inside the confined dir pointing OUT of the repo. realpath
// resolution must catch this and refuse it.
const evilLink = path.join(figDir, "evil.png");
let symlinkMade = false;
try {
  fs.symlinkSync(outsideSecret, evilLink);
  symlinkMade = true;
} catch {
  symlinkMade = false; // some FS / CI envs disallow symlinks; skip that case
}

// chdir into dashboard/ so REPO_ROOT (resolve(cwd, "..")) === repoRoot, then
// import paper.ts fresh (REPO_ROOT is a load-time const).
process.chdir(dashDir);
const { readTaskFigure } = await import("../lib/paper.ts");

// ── REJECT cases ───────────────────────────────────────────────────────────
console.log("\n=== REJECT (traversal / bad name / escape) ===");
const rejectNames = [
  ["empty string", ""],
  ["raw .. traversal", "../secret.png"],
  ["nested .. traversal", "../../secret.png"],
  ["bare ..", ".."],
  ["embedded .. mid-name", "a/../../../secret.png"],
  ["absolute path", "/etc/hostname"],
  ["absolute path to secret", outsideSecret],
  ["multi-segment forward slash", "sub/dir/plot.png"],
  ["cross-issue via segment", "../issue_111/other.png"],
  ["backslash traversal", "..\\secret.png"],
  ["backslash segment", "sub\\plot.png"],
  ["non-image .txt", "notes.txt"],
  ["non-image .pdf", "paper.pdf"],
  ["no extension", "plot"],
  ["double-extension .png.txt", "plot.png.txt"],
  ["dotfile .env", ".env"],
  ["dotfile .htaccess", ".htaccess"],
];
for (const [label, name] of rejectNames) {
  check(`rejects ${label} (${JSON.stringify(name)})`, readTaskFigure(ISSUE, name) === null);
}
if (symlinkMade) {
  check(
    "rejects out-of-repo symlink target (evil.png -> outside secret)",
    readTaskFigure(ISSUE, "evil.png") === null,
  );
} else {
  console.log("SKIP  out-of-repo symlink case (symlink creation unavailable here)");
}

// ── SERVE cases ────────────────────────────────────────────────────────────
console.log("\n=== SERVE (legit confined figure) ===");
const png = readTaskFigure(ISSUE, "plot.png");
check("serves legit issue_<N>/plot.png", png !== null);
check("serves PNG content-type", png && png.contentType === "image/png");
check("serves PNG bytes intact", png && Buffer.from(png.bytes).equals(pngBytes));

const svg = readTaskFigure(ISSUE, "chart.svg");
check("serves legit issue_<N>/chart.svg", svg !== null);
check("serves SVG content-type", svg && svg.contentType === "image/svg+xml");

check("missing legit-extension file is null (not 500)", readTaskFigure(ISSUE, "nope.png") === null);

// ── SERVE through a SYMLINKED figures/issue_<N>/ dir (fix 2 regression) ──────
// If figures/issue_<N>/ is itself a symlinked directory, a legit figure under
// it must STILL serve: figRoot is realpath'd so the realpath'd `real` compares
// equal. Pre-fix (un-realpath'd figRoot) this fail-closed-404'd a real figure.
console.log("\n=== SERVE through symlinked figures/issue_<N>/ dir (fix 2) ===");
const ISSUE_LINKED = 888;
const realFigStore = path.join(repoRoot, "figstore_888");
fs.mkdirSync(realFigStore, { recursive: true });
fs.writeFileSync(path.join(realFigStore, "linked.png"), pngBytes);
let dirSymlinkMade = false;
try {
  fs.symlinkSync(realFigStore, path.join(repoRoot, "figures", `issue_${ISSUE_LINKED}`), "dir");
  dirSymlinkMade = true;
} catch {
  dirSymlinkMade = false;
}
if (dirSymlinkMade) {
  const linked = readTaskFigure(ISSUE_LINKED, "linked.png");
  check("serves a figure under a symlinked issue dir (no false 404)", linked !== null);
  check("symlinked-dir figure has PNG content-type", linked && linked.contentType === "image/png");
  // Traversal through the symlinked dir must still be refused.
  check(
    "still rejects .. escape through a symlinked issue dir",
    readTaskFigure(ISSUE_LINKED, "../secret.png") === null,
  );
} else {
  console.log("SKIP  symlinked-dir case (symlink creation unavailable here)");
}

// ── cleanup + verdict ──────────────────────────────────────────────────────
process.chdir(os.tmpdir()); // leave dashDir before removing it
fs.rmSync(tmpRoot, { recursive: true, force: true });

console.log("");
if (failures > 0) {
  console.error(`${failures} assertion(s) FAILED`);
  process.exit(1);
}
console.log("All figure-confinement assertions passed.");
