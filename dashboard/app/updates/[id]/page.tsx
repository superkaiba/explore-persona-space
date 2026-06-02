import { redirect } from "next/navigation";

/**
 * Legacy per-item redirect: `/updates/<N>` → `/tasks/<N>#feed-body`.
 *
 * The May 29 refactor consolidated `/updates` + `/log` into a single feed-only
 * `/updates` page and removed the per-item detail route, so every historical
 * `/updates/<N>` pointer (Slack daily updates, mentor-update snapshots) started
 * 404ing. This route revives them, landing the reader on the clean-result body
 * section (`<section id="feed-body">`, the "BODY" entry in the task TOC) rather
 * than the top of the reverse-chronological event feed. The body panel defaults
 * expanded for first-time visitors, so the fragment scrolls straight to the
 * rendered result.
 *
 * The fragment is added here (post-auth), so it survives the sign-in round-trip:
 * `/updates/375` → `/sign-in?next=/updates/375` → (sign in) → `/updates/375` →
 * `/tasks/375#feed-body`.
 */
export default async function UpdatesItemRedirect({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  redirect(`/tasks/${id}#feed-body`);
}
