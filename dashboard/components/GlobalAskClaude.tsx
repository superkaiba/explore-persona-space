"use client";

/**
 * App-wide Ask-Claude mount.
 *
 * Mounts the global <MentorClaudePanel> so the Ask-Claude affordance baked into
 * the shared <MarkdownDoc> keystone works on EVERY surface — not just /updates,
 * which is where the panel used to live (inside CleanResultsUpdate). The panel
 * listens for the `eps:mentor-claude:ask` CustomEvent that MarkdownDocAskClaude
 * (and the legacy ClaudeAskButton/Composer) dispatch, opens a docked chat, and
 * only then fetches /api/chat-token + posts to /api/sidecar/chat.
 *
 * Auth-aware by construction:
 *   - The panel is mounted ONLY when the request is authenticated (`authed`,
 *     computed server-side in app/layout.tsx via requireSessionAuth) AND the
 *     current path is NOT a public surface.
 *   - On public pages (`/`, `/results`, `/results/*`, `/sign-in`) and when
 *     unauthenticated, NOTHING is rendered — so no `eps:mentor-claude:ask`
 *     listener is registered and, critically, no `/api/chat-token` fetch can
 *     ever fire from a public/unauth page. (MarkdownDoc already renders its
 *     Ask-Claude button DISABLED in `public` mode, so the two layers agree.)
 *
 * The /updates surface still mounts its OWN per-card MentorClaudePanel
 * instances (daily/log/weekly scopes) via CleanResultsUpdate; those are
 * unchanged. This global mount adds coverage for tasks/docs/literature, which
 * previously had no panel listening for the ask event.
 */
import { usePathname } from "next/navigation";
import { MentorClaudePanel } from "@/components/updates/MentorClaudePanel";

// Stable session id for the app-wide panel. Per-doc context arrives on the
// ask-event payload (contextMd); this base frame is the fallback scope.
const GLOBAL_SESSION_ID = "global";

const GLOBAL_BASE_CONTEXT = [
  "You are Claude Code helping a researcher read the EPS dashboard.",
  "The reader opened Ask-Claude from a page in the dashboard; the specific",
  "document or selection context is attached to each question when present.",
].join("\n");

// Public surfaces never mount the panel (no token fetch, no listener).
function isPublicPath(pathname: string): boolean {
  if (pathname === "/") return true;
  if (pathname === "/sign-in" || pathname.startsWith("/sign-in/")) return true;
  if (pathname === "/results" || pathname.startsWith("/results/")) return true;
  return false;
}

export function GlobalAskClaude({ authed }: { authed: boolean }) {
  const pathname = usePathname();
  if (!authed) return null;
  if (isPublicPath(pathname)) return null;
  return (
    <MentorClaudePanel
      sessionId={GLOBAL_SESSION_ID}
      baseContextMd={GLOBAL_BASE_CONTEXT}
    />
  );
}
