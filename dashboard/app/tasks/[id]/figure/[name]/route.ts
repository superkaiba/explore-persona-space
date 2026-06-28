/**
 * GET /tasks/<id>/figure/<name> — serve a committed figure for the paper render
 * (Phase C2).
 *
 * The paper.html carries relative figure `<img src="X.png">`; lib/paper.ts
 * rewrites those to this route so they resolve when the paper is mounted on the
 * task page. It serves only committed, already-public PNG/SVG/JPG figures from
 * `figures/issue_<N>/`, path-confined to a single safe filename under that dir
 * (`readTaskFigure`). Read-only, public (same surface as the existing
 * `/tasks/<id>/data` route — the proxy's `/tasks/* is public GET` allowlist
 * covers it), no auth, no shelling.
 */
import { readTaskFigure } from "@/lib/paper";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string; name: string }> },
) {
  const { id: idParam, name } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) {
    return new Response("bad id", { status: 400 });
  }
  const fig = readTaskFigure(id, decodeURIComponent(name));
  if (!fig) {
    return new Response("figure not found", { status: 404 });
  }
  return new Response(new Uint8Array(fig.bytes), {
    status: 200,
    headers: {
      "Content-Type": fig.contentType,
      // Committed artifacts are content-addressable in practice (a figure is
      // regenerated under a new name on re-runs); cache moderately.
      "Cache-Control": "public, max-age=3600",
    },
  });
}
