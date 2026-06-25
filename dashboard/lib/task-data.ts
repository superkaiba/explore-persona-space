/**
 * Per-task data-artifact resolution for the interactive data viewer
 * (clean-result v4 redesign, Phase 2 — the "Dashboard data-artifact interface"
 * contract in .claude/skills/clean-results/SPEC.md).
 *
 * What this reads, all from the repo root (sibling of dashboard/), all
 * COMMITTED + public artifacts (the same data the v4 body links to publicly via
 * SHA-pinned GitHub blob URLs):
 *
 *   - figures/issue_<N>/<file>.meta.json  — the per-figure sidecar. Per the
 *     Phase-2 contract this is "the per-row data table the viewer's sort/filter/
 *     reveal-more operates on". In practice the sidecar takes one of three
 *     real-world shapes (all observed in the live corpus):
 *       (a) embeds the rows inline under `rows` / `data` / `records` / `points`
 *           (array of objects → a table; array of scalars → a 1-column table);
 *       (b) points at a committed JSON via `data_path` / `source_data` (a path,
 *           or a list of paths, under eval_results/ or figures/);
 *       (c) carries only provenance (`commit` / `created` / `figsize`) — no
 *           per-row data. The viewer then renders the figure's link-out only.
 *   - eval_results/issue_<N>/**.json      — the per-cell / aggregate JSON the
 *     figures collapse, resolvable via a sidecar `data_path`.
 *
 * Server-only (filesystem reads). Every resolved path is confined under the
 * repo root with `assertUnderRepo` so a crafted `data_path` cannot escape it.
 *
 * The full data behind external HF links is NOT fetched server-side (it is not
 * on local disk and the dashboard does not proxy HF); the viewer renders the
 * locally-available rows and links out to the rest. Callers surface that
 * boundary — they do not fabricate the unavailable rows.
 */
import fs from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./repo";

// Hard cap on rows shipped to the client per artifact. The viewer paginates
// client-side; this bounds the JSON payload so a pathological multi-MB
// eval_results JSON can't blow up the response. Larger sets are truncated and
// flagged (`truncated: true`) so the UI can link out for the remainder.
const MAX_ROWS = 5000;

// Hard cap on the byte size of a `data_path` target before it is read + parsed.
// Bounds the single-process event loop + memory against a pathological multi-MB
// eval_results JSON (63 MB files exist in the corpus). Oversize -> skip + link-out.
const MAX_DATA_PATH_BYTES = 12 * 1024 * 1024;

// `data_path` targets are restricted to the two documented subtrees (Phase-2
// contract). Anything resolving outside figures/ + eval_results/ is refused —
// this is the real runtime confinement (the next.config NFT trace does NOT
// constrain a force-dynamic nodejs-runtime filesystem read).
const DATA_PATH_ROOTS = ["figures", "eval_results"] as const;

export type DataColumn = {
  key: string;
  /** "number" when every non-null value in the column parses as a finite
   *  number — drives numeric (vs lexical) sort + right-alignment. */
  type: "number" | "string";
};

export type DataArtifact = {
  /** Stable id within the task (the figure basename, e.g. "hero_dx_diffuse"). */
  id: string;
  /** Human label shown in the viewer's artifact picker. */
  label: string;
  /** The figure PNG filename this artifact's data backs, when known. */
  figureFile: string | null;
  /** Where the rows came from: inline sidecar, a resolved data_path, or none. */
  source: "sidecar-rows" | "data-path" | "none";
  /** Resolved on-disk source path, repo-relative, for display + provenance. */
  sourcePath: string | null;
  /** Optional human description pulled from the sidecar (`description`/`note`). */
  description: string | null;
  /** Column schema (union of keys across rows, in first-seen order). */
  columns: DataColumn[];
  /** The row records (each a flat object of scalar cells). */
  rows: Record<string, unknown>[];
  /** True when the underlying set exceeded MAX_ROWS and was truncated. */
  truncated: boolean;
  /** Total rows available before truncation. */
  totalRows: number;
  /** SHA-pinned GitHub raw URL of the figure (for the link-out), when found. */
  figureUrl: string | null;
};

export type TaskDataIndex = {
  taskId: number;
  artifacts: DataArtifact[];
};

// ── path safety ─────────────────────────────────────────────────────────────

/** Resolve a repo-relative path and assert it stays under the repo root. */
function assertUnderRepo(relOrAbs: string): string | null {
  const abs = path.resolve(REPO_ROOT, relOrAbs);
  const root = path.resolve(REPO_ROOT);
  if (abs !== root && !abs.startsWith(root + path.sep)) return null;
  return abs;
}

function repoRelative(abs: string): string {
  return path.relative(path.resolve(REPO_ROOT), abs);
}

/**
 * Resolve a sidecar `data_path` candidate to a safe, readable absolute path, or
 * null. Enforces, in order: repo-root confinement; restriction to the documented
 * figures/ + eval_results/ subtrees; symlink-escape protection (the realpath must
 * ALSO stay confined — `path.resolve` normalizes `../` but does NOT resolve
 * symlinks, while the subsequent readFileSync follows them); and a file-size cap
 * before the synchronous read/parse. Returns the realpath actually read.
 */
function resolveDataPathTarget(cand: string): string | null {
  const abs = assertUnderRepo(cand);
  if (!abs) return null;
  const root = path.resolve(REPO_ROOT);
  const underAllowed = (p: string): boolean =>
    DATA_PATH_ROOTS.some(
      (r) => p === path.join(root, r) || p.startsWith(path.join(root, r) + path.sep),
    );
  if (!underAllowed(abs)) return null;
  let real: string;
  try {
    real = fs.realpathSync(abs); // throws on ENOENT / broken symlink
  } catch {
    return null;
  }
  // A symlink target is followed by readFileSync; the realpath must stay confined.
  if (real !== root && !real.startsWith(root + path.sep)) return null;
  if (!underAllowed(real)) return null;
  try {
    if (fs.statSync(real).size > MAX_DATA_PATH_BYTES) return null;
  } catch {
    return null;
  }
  return real;
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

function readJson(abs: string): unknown {
  const raw = fs.readFileSync(abs, "utf8");
  return JSON.parse(raw);
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isScalar(v: unknown): boolean {
  return (
    v === null ||
    typeof v === "string" ||
    typeof v === "number" ||
    typeof v === "boolean"
  );
}

/**
 * Coerce a raw value into a single scalar cell. Arrays/objects nested inside a
 * row are JSON-stringified (compact) so the table stays flat and the cell is
 * still searchable + copyable. `per_seed: [...]` style columns thus render as
 * `[-0.001, -0.032, ...]` rather than being dropped.
 */
function toCell(v: unknown): string | number | boolean | null {
  if (isScalar(v)) return v as string | number | boolean | null;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

// ── row extraction ────────────────────────────────────────────────────────────

/** Keys a sidecar may use to embed its per-row data inline. */
const INLINE_ROW_KEYS = ["rows", "data", "records", "points"] as const;
/** Keys a sidecar may use to point at a committed data file (string or list). */
const DATA_PATH_KEYS = ["data_path", "source_data", "data_paths"] as const;
/** Keys carrying a human description of the figure data. */
const DESC_KEYS = ["description", "note", "y", "caption"] as const;

/**
 * Normalize an arbitrary parsed JSON value into a flat row array.
 *
 *   - array of objects        -> the rows verbatim (flattened cells)
 *   - array of scalars        -> a 1-column "value" table
 *   - object with a `per_*` /
 *     `rows`-like array inside -> recurse into the first array-of-objects /
 *                                 array-of-scalars found
 *   - object of objects       -> rows = entries, keyed by `_key`
 *   - else                     -> []
 */
function normalizeToRows(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    if (value.every((el) => isScalar(el))) {
      return value.map((el) => ({ value: toCell(el) }));
    }
    if (value.every((el) => isPlainObject(el))) {
      return value.map((el) => flattenRow(el as Record<string, unknown>));
    }
    // Mixed array — stringify each element into a single column.
    return value.map((el) => ({ value: toCell(el) }));
  }

  if (isPlainObject(value)) {
    // Prefer an inline rows-like array if present.
    for (const k of INLINE_ROW_KEYS) {
      if (Array.isArray(value[k])) {
        const rows = normalizeToRows(value[k]);
        if (rows.length) return rows;
      }
    }
    // An object-of-objects (e.g. `per_source: { src: {...} }`) becomes rows
    // keyed by the entry name.
    const entries = Object.entries(value);
    const objEntries = entries.filter(([, v]) => isPlainObject(v));
    if (objEntries.length >= 2 && objEntries.length === entries.length) {
      return objEntries.map(([key, v]) => ({
        _key: key,
        ...flattenRow(v as Record<string, unknown>),
      }));
    }
    // A `per_source`-style nested table inside a larger summary object.
    for (const [, v] of entries) {
      if (Array.isArray(v) && v.length && v.every((el) => isPlainObject(el))) {
        return v.map((el) => flattenRow(el as Record<string, unknown>));
      }
      if (isPlainObject(v)) {
        const sub = Object.values(v);
        if (sub.length >= 2 && sub.every((el) => isPlainObject(el))) {
          return Object.entries(v).map(([key, el]) => ({
            _key: key,
            ...flattenRow(el as Record<string, unknown>),
          }));
        }
      }
    }
  }

  return [];
}

/** Flatten a single row object into scalar cells (nested values stringified). */
function flattenRow(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) out[k] = toCell(v);
  return out;
}

/** Derive the column schema (union of keys, first-seen order) + per-col type. */
function deriveColumns(rows: Record<string, unknown>[]): DataColumn[] {
  const order: string[] = [];
  const seen = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) {
        seen.add(k);
        order.push(k);
      }
    }
  }
  return order.map((key) => {
    let sawValue = false;
    let allNumeric = true;
    for (const r of rows) {
      const v = r[key];
      if (v === null || v === undefined || v === "") continue;
      sawValue = true;
      if (typeof v === "number") continue;
      if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) {
        continue;
      }
      allNumeric = false;
      break;
    }
    return { key, type: sawValue && allNumeric ? "number" : "string" };
  });
}

function firstString(obj: Record<string, unknown>, keys: readonly string[]): string | null {
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return null;
}

// ── sidecar resolution ────────────────────────────────────────────────────────

/** Build the rows+meta for one figure's `.meta.json` sidecar. */
function resolveSidecar(
  taskId: number,
  figureBase: string,
  metaAbs: string,
  figureUrlByBase: Map<string, string>,
): DataArtifact {
  const figureFile = `${figureBase}.png`;
  const figureUrl = figureUrlByBase.get(figureBase) ?? null;
  const base: Omit<DataArtifact, "columns" | "rows" | "truncated" | "totalRows"> = {
    id: figureBase,
    label: figureBase.replace(/[_-]+/g, " "),
    figureFile,
    source: "none",
    sourcePath: repoRelative(metaAbs),
    description: null,
    figureUrl,
  };

  let meta: unknown;
  try {
    meta = readJson(metaAbs);
  } catch {
    return { ...base, columns: [], rows: [], truncated: false, totalRows: 0 };
  }

  const metaObj = isPlainObject(meta) ? meta : {};
  base.description = firstString(metaObj, DESC_KEYS);

  // 1) Inline rows in the sidecar itself.
  let rows = normalizeToRows(meta);
  let source: DataArtifact["source"] = rows.length ? "sidecar-rows" : "none";
  let sourcePath = repoRelative(metaAbs);

  // 2) Otherwise follow a data_path / source_data pointer (first one that
  //    resolves to a non-empty table).
  if (!rows.length) {
    const candidates: string[] = [];
    for (const k of DATA_PATH_KEYS) {
      const v = metaObj[k];
      if (typeof v === "string" && v.trim()) candidates.push(v.trim());
      else if (Array.isArray(v)) {
        for (const el of v) if (typeof el === "string" && el.trim()) candidates.push(el.trim());
      }
    }
    for (const cand of candidates) {
      const abs = resolveDataPathTarget(cand);
      if (!abs) continue;
      try {
        const parsed = readJson(abs);
        const candRows = normalizeToRows(parsed);
        if (candRows.length) {
          rows = candRows;
          source = "data-path";
          sourcePath = repoRelative(abs);
          break;
        }
      } catch {
        // Unparseable data_path target — skip; the figure link-out remains.
      }
    }
  }

  const totalRows = rows.length;
  const truncated = totalRows > MAX_ROWS;
  if (truncated) rows = rows.slice(0, MAX_ROWS);

  return {
    ...base,
    source,
    sourcePath,
    columns: deriveColumns(rows),
    rows,
    truncated,
    totalRows,
  };
}

// ── figure-URL discovery (link-out target) ────────────────────────────────────

/**
 * Scan the task body for SHA-pinned `raw.githubusercontent.com/.../figures/
 * issue_<N>/<file>.png` URLs and map each figure basename -> its pinned URL.
 * The v4 contract guarantees inline figures are SHA-pinned raw URLs; this gives
 * the viewer a "view full figure on GitHub" link without re-deriving the SHA.
 */
function figureUrlsFromBody(taskId: number, body: string): Map<string, string> {
  const map = new Map<string, string>();
  const re = new RegExp(
    `https?://raw\\.githubusercontent\\.com/[^\\s)"']+?/figures/issue_${taskId}/([^\\s)"'/]+)\\.png`,
    "g",
  );
  let m: RegExpExecArray | null;
  while ((m = re.exec(body)) !== null) {
    map.set(m[1], m[0]);
  }
  return map;
}

// ── public entry point ────────────────────────────────────────────────────────

/**
 * Build the data index for a task: every figure under figures/issue_<N>/ whose
 * `.meta.json` sidecar exists, resolved to its row table (inline, via data_path,
 * or empty). `body` (the task body markdown) is used only to recover the
 * SHA-pinned figure link-out URLs.
 *
 * Artifacts WITH rows sort before data-less ones; within each group, figures
 * referenced in the body sort first (in body order) so the viewer's default
 * selection matches what the reader is looking at.
 */
export function getTaskDataIndex(taskId: number, body: string): TaskDataIndex {
  const figDir = assertUnderRepo(path.join("figures", `issue_${taskId}`));
  if (!figDir || !fs.existsSync(figDir)) return { taskId, artifacts: [] };

  const figureUrlByBase = figureUrlsFromBody(taskId, body);
  // Body order index for sort (figures referenced earlier come first).
  const bodyOrder = new Map<string, number>();
  {
    let i = 0;
    for (const base of figureUrlByBase.keys()) {
      if (!bodyOrder.has(base)) bodyOrder.set(base, i++);
    }
  }

  let entries: string[];
  try {
    entries = fs.readdirSync(figDir);
  } catch {
    return { taskId, artifacts: [] };
  }

  const metaFiles = entries.filter((f) => f.endsWith(".meta.json"));
  const artifacts: DataArtifact[] = [];
  for (const mf of metaFiles) {
    const figureBase = mf.replace(/\.meta\.json$/, "");
    const metaAbs = path.join(figDir, mf);
    artifacts.push(resolveSidecar(taskId, figureBase, metaAbs, figureUrlByBase));
  }

  artifacts.sort((a, b) => {
    const aHas = a.rows.length > 0 ? 0 : 1;
    const bHas = b.rows.length > 0 ? 0 : 1;
    if (aHas !== bHas) return aHas - bHas;
    const ao = bodyOrder.has(a.id) ? bodyOrder.get(a.id)! : Number.MAX_SAFE_INTEGER;
    const bo = bodyOrder.has(b.id) ? bodyOrder.get(b.id)! : Number.MAX_SAFE_INTEGER;
    if (ao !== bo) return ao - bo;
    return a.id.localeCompare(b.id);
  });

  return { taskId, artifacts };
}
