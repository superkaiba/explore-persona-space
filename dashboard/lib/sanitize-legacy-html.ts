/**
 * Sanitize a trusted-but-now-public legacy Sagan-card HTML body.
 *
 * Legacy bodies (carrying `<!-- legacy-sagan-card -->`) are analyzer-
 * generated inline-SVG infographics that were historically rendered via
 * `dangerouslySetInnerHTML` on an AUTH-gated surface. The consolidation
 * makes Results/Overview public, so even trusted HTML now passes through a
 * sanitizer before it reaches the DOM.
 *
 * This is a string -> string transform using the `legacySchema` (wide:
 * allows SVG + scoped style + inline style, still strips script tags, on*
 * handlers, and javascript: URLs). It is a PURE function (no disk, no auth,
 * no env) so it can
 * run on the server (preferred — caller can pre-sanitize and pass the result
 * as a serializable prop) OR on the client.
 *
 * Implementation uses the hast utilities directly (fromHtml -> sanitize ->
 * toHtml) rather than a full unified pipeline, to avoid pulling in an extra
 * parser dependency: `hast-util-from-html` is already present (rehype-raw
 * depends on it). We do NOT use rehype-raw / markdown parsing here because
 * the input is already a complete HTML fragment, not markdown.
 */
import { fromHtml } from "hast-util-from-html";
import { sanitize } from "hast-util-sanitize";
import { toHtml } from "hast-util-to-html";
import { legacySchema } from "./markdown-sanitize";

export function sanitizeLegacyHtml(html: string): string {
  const tree = fromHtml(html, { fragment: true });
  const clean = sanitize(tree, legacySchema);
  return toHtml(clean);
}
