/**
 * GET /tasks/<id>/ref — the lightweight task stub for the paper cross-reference
 * hover-preview card (Phase C2).
 *
 *   GET -> { ok, id, title, abstract, isPaper, status, exists }
 *
 * The paper render carries `\epsref{N}` links (`<a class="eps-ref"
 * data-epsref="N" href="/tasks/N">`). On hover the client lazy-fetches THIS
 * route for the target's title + abstract to populate the preview card. Reads
 * only the target's registry title + body frontmatter/first prose (lib/tasks
 * `getTaskStub`) — committed, public, read-only — so it sits on the public
 * `/tasks/*` GET surface with no auth, mirroring the `/tasks/<id>/data` route.
 *
 * Forward-only fallback: when the target is not a paper (or doesn't exist), the
 * stub still returns a usable title + (excerpt) abstract so the card degrades
 * gracefully instead of failing the hover.
 */
import { getTaskStub } from "@/lib/tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) {
    return Response.json({ ok: false, error: "bad id" }, { status: 400 });
  }
  const stub = getTaskStub(id);
  return Response.json({ ok: true, ...stub });
}
