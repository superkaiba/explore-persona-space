"use client";

/**
 * Ask-Claude affordance for <MarkdownDoc> (enableAskClaude).
 *
 * Generalized from components/updates/MentorClaudePanel.tsx's `ClaudeAskButton`.
 * This is a thin trigger: it dispatches the SAME `eps:mentor-claude:ask`
 * CustomEvent that the global <MentorClaudePanel> already listens for, so the
 * existing sidecar-chat panel (which posts to /api/sidecar/chat via
 * /api/chat-token) handles the conversation. No new chat plumbing is added;
 * this just feeds the panel a doc-scoped context payload.
 *
 * Context fed to Claude: the doc body (truncated to a sane size) plus, if the
 * user has an active text selection inside the doc, the selected snippet.
 *
 * Public/disabled mode (`disabled`): the button renders visibly DISABLED and
 * dispatches NOTHING — so on public surfaces no `/api/chat-token` fetch is
 * ever triggered (the panel only fetches the token when it actually sends).
 */
import { MessageCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// Keep in sync with MentorClaudePanel's ASK_EVENT + ClaudeAskPayload shape.
const ASK_EVENT = "eps:mentor-claude:ask";

type ClaudeAskDetail = {
  scopeKind: "global" | "result";
  scopeId: string;
  sourceLabel: string;
  scopeTitle: string;
  contextMd: string;
  suggestedQuestion: string;
};

// Cap the doc body we hand to Claude as context. The panel also injects the
// global base context, so we keep the per-doc slice bounded.
const MAX_CONTEXT_CHARS = 12000;

export function MarkdownDocAskClaude({
  body,
  title,
  docId,
  disabled = false,
  label = "Ask Claude Code",
}: {
  body: string;
  title?: string;
  docId?: string;
  disabled?: boolean;
  label?: string;
}) {
  function onClick() {
    if (disabled) return;
    const selection =
      typeof window !== "undefined" ? window.getSelection()?.toString().trim() : "";
    const scopeTitle = title || "Document";
    const truncated =
      body.length > MAX_CONTEXT_CHARS ? `${body.slice(0, MAX_CONTEXT_CHARS)}\n…[truncated]` : body;
    const contextMd = selection
      ? `# ${scopeTitle}\n\nSelected text:\n> ${selection}\n\nFull document:\n${truncated}`
      : `# ${scopeTitle}\n\n${truncated}`;

    const detail: ClaudeAskDetail = {
      scopeKind: "result",
      scopeId: docId ? `doc-${docId}` : `doc-${scopeTitle}`,
      sourceLabel: scopeTitle,
      scopeTitle,
      contextMd,
      suggestedQuestion: selection
        ? "What does the selected passage mean and is it well-supported?"
        : "What are the main takeaways from this document?",
    };
    window.dispatchEvent(new CustomEvent(ASK_EVENT, { detail }));
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={disabled ? "Ask Claude Code (sign in to enable)" : label}
      aria-label={label}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border border-stone-200 bg-white px-2 py-1 text-[11px] text-stone-600 transition-colors hover:bg-stone-100 hover:text-stone-900",
        disabled && "cursor-not-allowed opacity-50 hover:bg-white hover:text-stone-600",
      )}
    >
      <MessageCircle className="h-3.5 w-3.5" />
      <span>{label}</span>
    </button>
  );
}
