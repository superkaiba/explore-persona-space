"use client";

import {
  type CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Check,
  ChevronDown,
  Loader2,
  MessageCircle,
  Plus,
  Send,
  Wrench,
  X,
  XCircle,
} from "lucide-react";
import { makeClientId } from "@/lib/client-id";
import { postSidecarChat } from "@/lib/sidecar-client";
import { cn } from "@/lib/utils";

type ClaudeScopeKind = "global" | "result";

type AnchorRect = {
  top: number;
  right: number;
  bottom: number;
  left: number;
  width: number;
  height: number;
};

export type ClaudeAskPayload = {
  scopeTitle: string;
  contextMd: string;
  suggestedQuestion: string;
  initialQuestion?: string;
  autoSubmit?: boolean;
  scopeKind?: ClaudeScopeKind;
  scopeId?: string;
  sourceLabel?: string;
  anchorRect?: AnchorRect | null;
  startNewTab?: boolean;
};

type ToolBlock = {
  kind: "tool";
  id: string;
  name: string;
  input: Record<string, unknown>;
  ok?: boolean;
  result?: string;
};

type TextBlock = { kind: "text"; text: string };
type ChatBlock = TextBlock | ToolBlock;
type StartupPhase = "spawning" | "loading" | "ready" | null;

type ChatMessage =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "assistant";
      blocks: ChatBlock[];
      pending?: boolean;
      startupPhase?: StartupPhase;
      startedAt?: number;
      durationMs?: number;
    };

type ChatTab = {
  id: string;
  title: string;
  draft: string;
  pending: boolean;
  messages: ChatMessage[];
  createdAt: number;
};

type ChatWindow = {
  key: string;
  scopeKind: ClaudeScopeKind;
  scopeId: string;
  sourceLabel: string;
  scopeTitle: string;
  contextMd: string;
  suggestedQuestion: string;
  open: boolean;
  activeTabId: string;
  tabs: ChatTab[];
  anchorRect?: AnchorRect | null;
  createdAt: number;
  updatedAt: number;
};

type NormalizedPayload = Required<
  Pick<ClaudeAskPayload, "scopeTitle" | "contextMd" | "suggestedQuestion">
> &
  Pick<ClaudeAskPayload, "initialQuestion" | "autoSubmit" | "anchorRect" | "startNewTab"> & {
    scopeKind: ClaudeScopeKind;
    scopeId: string;
    sourceLabel: string;
    key: string;
  };

const ASK_EVENT = "eps:mentor-claude:ask";
const MAX_STORED_WINDOWS = 10;
const MAX_STORED_TABS = 8;
const MAX_STORED_MESSAGES = 40;

function rectPayload(rect: DOMRect): AnchorRect {
  return {
    top: rect.top,
    right: rect.right,
    bottom: rect.bottom,
    left: rect.left,
    width: rect.width,
    height: rect.height,
  };
}

function anchorForElement(element: HTMLElement): AnchorRect {
  const anchor = element.closest("[data-claude-anchor]") as HTMLElement | null;
  return rectPayload((anchor ?? element).getBoundingClientRect());
}

function anchorElement(element: HTMLElement): HTMLElement {
  return (element.closest("[data-claude-anchor]") as HTMLElement | null) ?? element;
}

function setHoveredAnchor(element: HTMLElement) {
  document
    .querySelectorAll<HTMLElement>("[data-claude-anchor][data-claude-hovered='true']")
    .forEach((anchor) => {
      delete anchor.dataset.claudeHovered;
    });
  anchorElement(element).dataset.claudeHovered = "true";
}

export function dispatchClaudeHover(_payload: ClaudeAskPayload, element: HTMLElement) {
  setHoveredAnchor(element);
}

export function clearClaudeHover(element: HTMLElement) {
  delete anchorElement(element).dataset.claudeHovered;
}

export function ClaudeAskButton({
  payload,
  label = "Ask Claude Code",
  compact = false,
  className,
}: {
  payload: ClaudeAskPayload;
  label?: string;
  compact?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={(event) => {
        window.dispatchEvent(
          new CustomEvent<ClaudeAskPayload>(ASK_EVENT, {
            detail: {
              ...payload,
              anchorRect: anchorForElement(event.currentTarget),
            },
          }),
        );
      }}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border border-border bg-subtle px-2 py-1 text-[11px] text-muted transition-colors hover:bg-raised hover:text-fg",
        compact && "h-7 w-7 justify-center px-0 py-0",
        className,
      )}
      title={label}
      aria-label={label}
    >
      <MessageCircle className="h-3.5 w-3.5" />
      {!compact && <span>{label}</span>}
    </button>
  );
}

export function ClaudeAskComposer({
  payload,
  placeholder = "Ask Claude Code to inspect these results...",
  className,
}: {
  payload: ClaudeAskPayload;
  placeholder?: string;
  className?: string;
}) {
  const [draft, setDraft] = useState("");
  const formRef = useRef<HTMLFormElement>(null);

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const text = draft.trim();
    if (!text) return;
    const anchorRect = formRef.current ? anchorForElement(formRef.current) : payload.anchorRect;
    window.dispatchEvent(
      new CustomEvent<ClaudeAskPayload>(ASK_EVENT, {
        detail: {
          ...payload,
          suggestedQuestion: text,
          initialQuestion: text,
          autoSubmit: true,
          anchorRect,
        },
      }),
    );
    setDraft("");
  }

  return (
    <form
      ref={formRef}
      onSubmit={submit}
      className={cn("rounded-lg border border-border bg-panel p-3", className)}
    >
      <div className="flex items-end gap-2">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          placeholder={placeholder}
          className="min-h-[54px] flex-1 resize-none rounded-md border border-border bg-canvas px-3 py-2 text-[13px] leading-relaxed text-fg placeholder:text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
        />
        <button
          type="submit"
          disabled={!draft.trim()}
          className="grid h-[54px] w-11 place-items-center rounded-md bg-fg text-canvas transition-opacity disabled:opacity-40"
          aria-label="Ask Claude Code"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>
    </form>
  );
}

export function MentorClaudePanel({
  sessionId,
  baseContextMd,
}: {
  sessionId: string;
  baseContextMd: string;
}) {
  const storageKey = `eps-mentor-claude:${sessionId}:v2`;
  const [mounted, setMounted] = useState(false);
  const [viewport, setViewport] = useState({ width: 1280, height: 800 });
  const [windows, setWindows] = useState<ChatWindow[]>([]);
  const windowsRef = useRef<ChatWindow[]>([]);

  useEffect(() => {
    windowsRef.current = windows;
  }, [windows]);

  useEffect(() => {
    setMounted(true);
    setViewport({ width: window.innerWidth, height: window.innerHeight });
    setWindows(readStoredWindows(storageKey));
    const onResize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [storageKey]);

  useEffect(() => {
    if (!mounted) return;
    const id = window.setTimeout(() => {
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(toStoredWindows(windows)));
      } catch {}
    }, 250);
    return () => window.clearTimeout(id);
  }, [mounted, storageKey, windows]);

  useEffect(() => {
    setWindows((current) =>
      current.map((win) =>
        win.scopeKind === "global" && win.scopeId === sessionId
          ? { ...win, contextMd: baseContextMd }
          : win,
      ),
    );
  }, [baseContextMd, sessionId]);

  const defaultScope = useMemo<ClaudeAskPayload>(
    () => ({
      scopeKind: "global",
      scopeId: sessionId,
      sourceLabel: "Global",
      scopeTitle: "Results log",
      contextMd: baseContextMd,
      suggestedQuestion: "What are the main takeaways from this update?",
    }),
    [baseContextMd, sessionId],
  );

  const normalizePayload = useCallback(
    (payload: ClaudeAskPayload): NormalizedPayload => {
      const scopeKind = payload.scopeKind ?? (payload.scopeId ? "result" : "global");
      const scopeId = payload.scopeId ?? (scopeKind === "global" ? sessionId : payload.scopeTitle);
      const key = `${scopeKind}:${scopeId}`;
      const sourceLabel =
        payload.sourceLabel ?? inferSourceLabel(scopeKind, payload.contextMd, scopeId);
      return {
        scopeKind,
        scopeId,
        sourceLabel,
        key,
        scopeTitle: payload.scopeTitle || defaultScope.scopeTitle,
        contextMd: payload.contextMd || baseContextMd,
        suggestedQuestion: payload.suggestedQuestion || defaultScope.suggestedQuestion,
        initialQuestion: payload.initialQuestion,
        autoSubmit: payload.autoSubmit,
        anchorRect: payload.anchorRect,
        startNewTab: payload.startNewTab,
      };
    },
    [baseContextMd, defaultScope.scopeTitle, defaultScope.suggestedQuestion, sessionId],
  );

  const buildPromptContext = useCallback(
    (focusScope: NormalizedPayload | ChatWindow) => {
      const focusContext =
        focusScope.contextMd === baseContextMd
          ? focusScope.contextMd
          : `${baseContextMd}\n\nCurrent focus:\n${focusScope.contextMd}`;
      return [
        "You are Claude Code answering a research mentor reading EPS results.",
        "Use your VM tools when needed. Do not rely only on this prompt.",
        "Inspect the dashboard database with psql, the local repo, GitHub issues, and artifact URLs when the question requires it.",
        "If result artifacts need inspection, use shell tools such as curl, git, rg, and psql to locate or download them.",
        "Answer concisely and cite the specific result, claim id, issue number, file, or artifact you inspected.",
        "",
        "Update context:",
        focusContext,
      ].join("\n");
    },
    [baseContextMd],
  );

  const ensureWindow = useCallback(
    (rawPayload: ClaudeAskPayload) => {
      const payload = normalizePayload(rawPayload);
      const seedText = payload.initialQuestion ?? payload.suggestedQuestion;
      const existingWindow = windowsRef.current.find((win) => win.key === payload.key);
      const existingActiveTab =
        existingWindow?.tabs.find((tab) => tab.id === existingWindow.activeTabId) ??
        existingWindow?.tabs[0];
      const preparedTab =
        !existingWindow ||
        !existingActiveTab ||
        payload.startNewTab ||
        (payload.autoSubmit && existingActiveTab.pending)
          ? makeTab((existingWindow?.tabs.length ?? 0) + 1, payload.autoSubmit ? "" : seedText)
          : null;
      const targetTabId = preparedTab?.id ?? existingActiveTab?.id ?? makeClientId("claude-tab");

      setWindows((current) => {
        const index = current.findIndex((win) => win.key === payload.key);
        const now = Date.now();
        if (index === -1) {
          const tab = preparedTab ?? makeTab(1, payload.autoSubmit ? "" : seedText);
          return [
            ...current,
            {
              key: payload.key,
              scopeKind: payload.scopeKind,
              scopeId: payload.scopeId,
              sourceLabel: payload.sourceLabel,
              scopeTitle: payload.scopeTitle,
              contextMd: payload.contextMd,
              suggestedQuestion: payload.suggestedQuestion,
              open: true,
              activeTabId: targetTabId,
              tabs: [tab],
              anchorRect: payload.anchorRect,
              createdAt: now,
              updatedAt: now,
            },
          ];
        }

        return current.map((win, winIndex) => {
          if (winIndex !== index) return win;
          const activeTab = win.tabs.find((tab) => tab.id === win.activeTabId) ?? win.tabs[0];
          const needsNewTab =
            !win.tabs.some((tab) => tab.id === targetTabId) &&
            (payload.startNewTab || (payload.autoSubmit && activeTab?.pending));
          let nextTabs = win.tabs;
          let activeTabId = targetTabId;

          if (!activeTab || needsNewTab) {
            const tab = preparedTab ?? makeTab(win.tabs.length + 1, payload.autoSubmit ? "" : seedText);
            nextTabs = [...win.tabs, tab];
            activeTabId = tab.id;
          } else if (!payload.autoSubmit && seedText && !activeTab.draft) {
            nextTabs = win.tabs.map((tab) =>
              tab.id === activeTabId ? { ...tab, draft: seedText } : tab,
            );
          }

          return {
            ...win,
            scopeKind: payload.scopeKind,
            scopeId: payload.scopeId,
            sourceLabel: payload.sourceLabel,
            scopeTitle: payload.scopeTitle,
            contextMd: payload.contextMd,
            suggestedQuestion: payload.suggestedQuestion,
            open: true,
            activeTabId,
            tabs: nextTabs,
            anchorRect: payload.anchorRect ?? win.anchorRect,
            updatedAt: now,
          };
        });
      });

      return { payload, tabId: targetTabId };
    },
    [normalizePayload],
  );

  const sendText = useCallback(
    async (windowKey: string, tabId: string, rawText: string, focusScope: NormalizedPayload | ChatWindow) => {
      const text = rawText.trim();
      if (!text) return;

      const currentWindow = windowsRef.current.find((win) => win.key === windowKey);
      const currentTab = currentWindow?.tabs.find((tab) => tab.id === tabId);
      if (currentTab?.pending) return;

      const priorMessages = currentTab?.messages ?? [];
      const userId = makeClientId("msg");
      const assistantId = makeClientId("msg");
      const startedAt = Date.now();

      setWindows((current) =>
        current.map((win) => {
          if (win.key !== windowKey) return win;
          return {
            ...win,
            open: true,
            activeTabId: tabId,
            updatedAt: Date.now(),
            tabs: win.tabs.map((tab) => {
              if (tab.id !== tabId) return tab;
              return {
                ...tab,
                draft: "",
                pending: true,
                title: tab.messages.length === 0 ? shortTitle(text) : tab.title,
                messages: [
                  ...tab.messages,
                  { id: userId, role: "user", text },
                  {
                    id: assistantId,
                    role: "assistant",
                    blocks: [],
                    pending: true,
                    startupPhase: "spawning",
                    startedAt,
                  },
                ],
              };
            }),
          };
        }),
      );

      const updateAssistant = (
        updater: (message: Extract<ChatMessage, { role: "assistant" }>) => void,
      ) => {
        setWindows((current) =>
          current.map((win) => {
            if (win.key !== windowKey) return win;
            return {
              ...win,
              updatedAt: Date.now(),
              tabs: win.tabs.map((tab) => {
                if (tab.id !== tabId) return tab;
                return {
                  ...tab,
                  messages: tab.messages.map((message) => {
                    if (message.id !== assistantId || message.role !== "assistant") {
                      return message;
                    }
                    const next = { ...message, blocks: [...message.blocks] };
                    updater(next);
                    return next;
                  }),
                };
              }),
            };
          }),
        );
      };

      const finishAssistant = () => {
        setWindows((current) =>
          current.map((win) => {
            if (win.key !== windowKey) return win;
            return {
              ...win,
              updatedAt: Date.now(),
              tabs: win.tabs.map((tab) => {
                if (tab.id !== tabId) return tab;
                return {
                  ...tab,
                  pending: false,
                  messages: tab.messages.map((message) =>
                    message.id === assistantId && message.role === "assistant"
                      ? { ...message, pending: false, startupPhase: null }
                      : message,
                  ),
                };
              }),
            };
          }),
        );
      };

      const appendText = (chunk: string) => {
        if (!chunk) return;
        updateAssistant((message) => {
          const last = message.blocks[message.blocks.length - 1];
          if (last?.kind === "text") {
            last.text += chunk;
          } else {
            message.blocks.push({ kind: "text", text: chunk });
          }
        });
      };

      try {
        const tokenRes = await fetch("/api/chat-token", { method: "POST" });
        if (!tokenRes.ok) {
          const body = await tokenRes.text().catch(() => "");
          appendText(`Claude is available after sign-in. ${body || tokenRes.statusText}`);
          return;
        }

        const { token, sidecar_url: sidecarUrl } = (await tokenRes.json()) as {
          token: string;
          sidecar_url: string;
        };

        const prompt = `${buildPromptContext(focusScope)}\n\nMentor question:\n${text}`;
        const res = await postSidecarChat(sidecarUrl, token, {
          session_id: sidecarSessionId(sessionId, windowKey, tabId),
          provider: "claude_code",
          messages: sidecarMessages(priorMessages, prompt),
        });

        if (!res.ok || !res.body) {
          const body = await res.text().catch(() => "");
          appendText(body || res.statusText);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() ?? "";

          for (const eventText of events) {
            const parsed = parseSseEvent(eventText);
            if (!parsed) continue;

            if (parsed.eventName === "starting") {
              const phase = String(parsed.data.phase ?? "spawning");
              updateAssistant((message) => {
                message.startupPhase =
                  phase === "warm" ? null : phase === "loading" ? "loading" : "spawning";
              });
            } else if (parsed.eventName === "ready") {
              updateAssistant((message) => {
                message.startupPhase = "ready";
              });
            } else if (parsed.eventName === "token") {
              appendText(String(parsed.data.text ?? ""));
            } else if (parsed.eventName === "tool_use") {
              updateAssistant((message) => {
                message.blocks.push({
                  kind: "tool",
                  id: String(parsed.data.id ?? makeClientId("tool")),
                  name: String(parsed.data.name ?? "tool"),
                  input: isRecord(parsed.data.input) ? parsed.data.input : {},
                });
              });
            } else if (parsed.eventName === "tool_result") {
              const matchId = String(parsed.data.tool_use_id ?? "");
              updateAssistant((message) => {
                for (let i = message.blocks.length - 1; i >= 0; i--) {
                  const block = message.blocks[i];
                  if (block.kind === "tool" && block.id === matchId && block.ok === undefined) {
                    message.blocks[i] = {
                      ...block,
                      ok: Boolean(parsed.data.ok),
                      result: String(parsed.data.content ?? ""),
                    };
                    break;
                  }
                }
              });
            } else if (parsed.eventName === "done") {
              updateAssistant((message) => {
                message.pending = false;
                message.startupPhase = null;
                if (typeof parsed.data.duration_ms === "number") {
                  message.durationMs = parsed.data.duration_ms;
                }
              });
            } else if (parsed.eventName === "error") {
              appendText(`\n\nClaude Code error: ${String(parsed.data.message ?? "failed")}`);
            }
          }
        }
      } catch (error) {
        appendText(error instanceof Error ? error.message : String(error));
      } finally {
        finishAssistant();
      }
    },
    [buildPromptContext, sessionId],
  );

  useEffect(() => {
    function onAsk(event: Event) {
      const detail = (event as CustomEvent<ClaudeAskPayload>).detail;
      const text = detail.initialQuestion ?? detail.suggestedQuestion;
      const { payload, tabId } = ensureWindow(detail);
      if (detail.autoSubmit) void sendText(payload.key, tabId, text, payload);
    }

    window.addEventListener(ASK_EVENT, onAsk);
    return () => {
      window.removeEventListener(ASK_EVENT, onAsk);
    };
  }, [ensureWindow, sendText]);

  function closeAllWindows() {
    setWindows((current) =>
      current.map((win) => (win.open ? { ...win, open: false, updatedAt: Date.now() } : win)),
    );
  }

  function newTab(windowKey: string) {
    setWindows((current) =>
      current.map((win) => {
        if (win.key !== windowKey) return win;
        const tab = makeTab(win.tabs.length + 1, "");
        return {
          ...win,
          activeTabId: tab.id,
          tabs: [...win.tabs, tab],
          open: true,
          updatedAt: Date.now(),
        };
      }),
    );
  }

  function closeTab(windowKey: string, tabId: string) {
    setWindows((current) =>
      current.map((win) => {
        if (win.key !== windowKey) return win;
        if (win.tabs.length <= 1) return { ...win, open: false, updatedAt: Date.now() };
        const tabs = win.tabs.filter((tab) => tab.id !== tabId);
        return {
          ...win,
          tabs,
          activeTabId: win.activeTabId === tabId ? tabs[tabs.length - 1].id : win.activeTabId,
          updatedAt: Date.now(),
        };
      }),
    );
  }

  function activateTab(windowKey: string, tabId: string) {
    setWindows((current) =>
      current.map((win) =>
        win.key === windowKey
          ? { ...win, activeTabId: tabId, open: true, updatedAt: Date.now() }
          : win,
      ),
    );
  }

  function setDraft(windowKey: string, tabId: string, draft: string) {
    setWindows((current) =>
      current.map((win) => {
        if (win.key !== windowKey) return win;
        return {
          ...win,
          tabs: win.tabs.map((tab) => (tab.id === tabId ? { ...tab, draft } : tab)),
        };
      }),
    );
  }

  function reopenMostRecentWindow() {
    setWindows((current) => {
      if (current.length === 0) return current;
      const latest = mostRecentWindow(current);
      return current.map((win) =>
        win.key === latest.key ? { ...win, open: true, updatedAt: Date.now() } : win,
      );
    });
  }

  const openWindows = windows.filter((win) => win.open);
  if (!mounted) return null;
  if (openWindows.length === 0) {
    if (windows.length === 0) return null;
    return (
      <ClaudeDockTab
        windowCount={windows.length}
        tabCount={flattenChatTabs(windows).length}
        onOpen={reopenMostRecentWindow}
      />
    );
  }
  const activeWindow = mostRecentWindow(openWindows);
  const activeTab = activeWindow?.tabs.find((tab) => tab.id === activeWindow.activeTabId) ?? activeWindow?.tabs[0] ?? null;
  if (!activeWindow || !activeTab) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-50">
      <ClaudeChatWindow
        windows={openWindows}
        activeWindow={activeWindow}
        activeTab={activeTab}
        style={dockedChatBoxStyle(viewport)}
        onClose={closeAllWindows}
        onNewTab={() => newTab(activeWindow.key)}
        onCloseTab={(windowKey, tabId) => closeTab(windowKey, tabId)}
        onActivateTab={(windowKey, tabId) => activateTab(windowKey, tabId)}
        onDraftChange={(windowKey, tabId, draft) => setDraft(windowKey, tabId, draft)}
        onSend={(windowKey, tabId, text) => {
          const win = windowsRef.current.find((current) => current.key === windowKey);
          if (win) void sendText(windowKey, tabId, text, win);
        }}
      />
    </div>
  );
}

function ClaudeDockTab({
  windowCount,
  tabCount,
  onOpen,
}: {
  windowCount: number;
  tabCount: number;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="fixed right-3 top-24 z-50 flex items-center gap-2 rounded-l-lg border border-r-0 border-border bg-panel px-3 py-2 text-left text-[12px] text-fg shadow-rail transition-colors hover:bg-raised"
      aria-label="Open Claude Code chat"
    >
      <MessageCircle className="h-4 w-4 text-muted" />
      <span className="font-medium">Claude Code</span>
      <span className="font-mono text-[10px] text-muted">
        {tabCount} {tabCount === 1 ? "tab" : "tabs"}
        {windowCount > 1 ? `/${windowCount}` : ""}
      </span>
    </button>
  );
}

function ClaudeChatWindow({
  windows,
  activeWindow,
  activeTab,
  style,
  onClose,
  onNewTab,
  onCloseTab,
  onActivateTab,
  onDraftChange,
  onSend,
}: {
  windows: ChatWindow[];
  activeWindow: ChatWindow;
  activeTab: ChatTab;
  style: CSSProperties;
  onClose: () => void;
  onNewTab: () => void;
  onCloseTab: (windowKey: string, tabId: string) => void;
  onActivateTab: (windowKey: string, tabId: string) => void;
  onDraftChange: (windowKey: string, tabId: string, draft: string) => void;
  onSend: (windowKey: string, tabId: string, text: string) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const flatTabs = flattenChatTabs(windows);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeTab?.id, activeTab?.messages]);

  function send(e?: React.FormEvent) {
    e?.preventDefault();
    onSend(activeWindow.key, activeTab.id, activeTab.draft);
  }

  return (
    <>
      <aside
        className="pointer-events-auto fixed z-10 flex flex-col overflow-hidden rounded-lg border border-border bg-panel shadow-rail md:rounded-r-none"
        style={style}
      >
        <header className="flex items-start gap-3 border-b border-border bg-panel px-4 py-3">
          <MessageCircle className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold text-fg">Claude Code</div>
            <div className="mt-0.5 truncate text-[11px] text-muted">
              {activeWindow.sourceLabel} · {activeWindow.scopeTitle}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-muted transition-colors hover:bg-subtle hover:text-fg"
            aria-label="Close Claude chat"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="flex min-h-0 items-center gap-1 border-b border-border bg-subtle/35 px-2">
          <div className="flex min-w-0 flex-1 overflow-x-auto">
            {flatTabs.map(({ win, tab }) => (
              <button
                key={`${win.key}:${tab.id}`}
                type="button"
                onClick={() => onActivateTab(win.key, tab.id)}
                className={cn(
                  "group relative flex max-w-[170px] shrink-0 flex-col border-b px-2 py-1.5 text-left transition-colors",
                  win.key === activeWindow.key && tab.id === activeTab.id
                    ? "border-fg text-fg"
                    : "border-transparent text-muted hover:text-fg",
                )}
              >
                <span className="flex w-full min-w-0 items-center gap-1 text-[11px] font-medium">
                  <span className="truncate">{win.sourceLabel}</span>
                  {tab.pending && <Loader2 className="h-3 w-3 shrink-0 animate-spin" />}
                </span>
                <span className="flex w-full min-w-0 items-center gap-1 text-[10px] text-muted">
                  <span className="truncate">{tab.title || win.scopeTitle}</span>
                </span>
                {flatTabs.length > 1 && (
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(event) => {
                      event.stopPropagation();
                      onCloseTab(win.key, tab.id);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        event.stopPropagation();
                        onCloseTab(win.key, tab.id);
                      }
                    }}
                    className="absolute mt-0.5 self-end rounded p-0.5 text-faint opacity-0 hover:bg-border hover:text-fg group-hover:opacity-100"
                    aria-label={`Close ${win.sourceLabel} tab`}
                  >
                    <X className="h-3 w-3" />
                  </span>
                )}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onNewTab}
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-raised hover:text-fg"
            aria-label="New Claude chat tab"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {activeTab.messages.length === 0 ? (
            <div className="rounded-md border border-dashed border-border bg-subtle/30 p-3 text-[12px] leading-relaxed text-muted">
              Ask about this scope. This tab keeps its own Claude Code session.
            </div>
          ) : (
            <ul className="flex flex-col gap-4">
              {activeTab.messages.map((message) => (
                <li key={message.id}>
                  <ChatMessageView message={message} />
                </li>
              ))}
            </ul>
          )}
        </div>

        <form onSubmit={send} className="flex items-end gap-2 border-t border-border bg-panel p-3">
          <textarea
            value={activeTab.draft}
            onChange={(event) => onDraftChange(activeWindow.key, activeTab.id, event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
            rows={2}
            disabled={activeTab.pending}
            placeholder="Ask Claude Code..."
            className="min-h-[44px] flex-1 resize-none rounded-md border border-border bg-subtle px-3 py-2 text-[13px] leading-relaxed text-fg placeholder:text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={activeTab.pending || !activeTab.draft.trim()}
            className="grid h-10 w-10 place-items-center rounded-md bg-fg text-canvas transition-opacity disabled:opacity-40"
            aria-label="Send question"
          >
            {activeTab.pending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </form>
      </aside>
    </>
  );
}

function ChatMessageView({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="ml-8">
        <div className="mb-1 text-right text-[10px] font-medium text-muted">Mentor</div>
        <div className="whitespace-pre-wrap rounded-md border border-border bg-subtle px-3 py-2 text-[13px] leading-relaxed text-fg">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-1 text-[10px] font-medium text-muted">Claude</div>
      <div className="rounded-md border border-border bg-panel px-3 py-2">
        {message.startupPhase && message.blocks.length === 0 && (
          <StartupPill phase={message.startupPhase} startedAt={message.startedAt ?? Date.now()} />
        )}
        {message.blocks.length === 0 && !message.startupPhase && message.pending && (
          <span className="inline-flex items-center gap-2 text-[12px] text-muted">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Waiting for Claude Code
          </span>
        )}
        <div className="flex flex-col gap-2">
          {message.blocks.map((block, index) =>
            block.kind === "text" ? (
              <AssistantMarkdown key={index} text={block.text} />
            ) : (
              <ToolCard key={`${block.id}-${index}`} block={block} />
            ),
          )}
        </div>
        {message.pending && message.blocks.length > 0 && (
          <div className="mt-2 inline-flex items-center gap-2 text-[11px] text-muted">
            <Loader2 className="h-3 w-3 animate-spin" />
            Streaming
          </div>
        )}
        {!message.pending && message.durationMs != null && (
          <div className="mt-2 border-t border-border pt-2 font-mono text-[10px] text-muted">
            {Math.round(message.durationMs / 1000)}s
          </div>
        )}
      </div>
    </div>
  );
}

function AssistantMarkdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm prose-tight max-w-none break-words dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function StartupPill({ phase, startedAt }: { phase: StartupPhase; startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 150);
    return () => window.clearInterval(id);
  }, [startedAt]);

  const label =
    phase === "loading"
      ? "Loading tools and memory"
      : phase === "ready"
        ? "Ready"
        : "Starting Claude Code";

  return (
    <div className="inline-flex items-center gap-2 rounded-md border border-border bg-subtle px-2 py-1 text-[11px] text-muted">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-running" />
      <span>{label}</span>
      <span className="font-mono text-[10px]">{elapsed.toFixed(1)}s</span>
    </div>
  );
}

function ToolCard({ block }: { block: ToolBlock }) {
  const [expanded, setExpanded] = useState(false);
  const summary = formatToolInput(block.name, block.input);
  const status =
    block.ok === undefined ? (
      <Wrench className="h-3 w-3 animate-pulse text-muted" />
    ) : block.ok ? (
      <Check className="h-3 w-3 text-confidence-high" />
    ) : (
      <XCircle className="h-3 w-3 text-confidence-moderate" />
    );

  return (
    <div className="rounded-md border border-border bg-subtle/40">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-start gap-2 px-2 py-1.5 text-left"
      >
        <span className="mt-0.5 shrink-0">{status}</span>
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-medium text-muted">{block.name}</div>
          <div className="truncate font-mono text-[11px] text-fg">{summary}</div>
        </div>
        {block.result && (
          <ChevronDown
            className={cn(
              "mt-1 h-3 w-3 shrink-0 text-muted transition-transform",
              expanded && "rotate-180",
            )}
          />
        )}
      </button>
      {expanded && block.result && (
        <pre className="max-h-[240px] overflow-auto whitespace-pre-wrap break-words border-t border-border bg-panel p-2 font-mono text-[11px] text-fg-soft">
          {block.result}
        </pre>
      )}
    </div>
  );
}

function makeTab(index: number, draft: string): ChatTab {
  return {
    id: makeClientId("claude-tab"),
    title: `Tab ${index}`,
    draft,
    pending: false,
    messages: [],
    createdAt: Date.now(),
  };
}

function dockedChatBoxStyle(viewport: { width: number; height: number }): CSSProperties {
  if (viewport.width < 768) {
    return {
      left: 12,
      right: 12,
      bottom: 12,
      height: "min(76dvh, 640px)",
    };
  }

  return {
    top: 12,
    right: 0,
    bottom: 12,
    width: Math.min(460, viewport.width - 24),
  };
}

function mostRecentWindow(windows: ChatWindow[]) {
  return windows.reduce((latest, win) => (win.updatedAt > latest.updatedAt ? win : latest));
}

function flattenChatTabs(windows: ChatWindow[]) {
  return windows.flatMap((win) =>
    win.tabs.map((tab) => ({
      win,
      tab,
    })),
  );
}

function shortTitle(text: string) {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > 34 ? `${compact.slice(0, 31)}...` : compact || "New tab";
}

function inferSourceLabel(scopeKind: ClaudeScopeKind, contextMd: string, scopeId: string) {
  if (scopeKind === "global") return "Global";
  const issue = contextMd.match(/github\.com\/superkaiba\/explore-persona-space\/issues\/(\d+)/i);
  if (issue?.[1]) return `Issue #${issue[1]}`;
  return `Claim ${scopeId.slice(0, 8)}`;
}

function sidecarSessionId(sessionId: string, windowKey: string, tabId: string) {
  return `mentor-update-${sessionId}-${windowKey}-${tabId}`.replace(/[^a-zA-Z0-9_.:-]/g, "-");
}

function sidecarMessages(priorMessages: ChatMessage[], finalPrompt: string) {
  const history = priorMessages.flatMap((message) => {
    if (message.role === "user") return [{ role: "user", content: message.text }];
    const content = message.blocks
      .filter((block): block is TextBlock => block.kind === "text")
      .map((block) => block.text)
      .join("")
      .trim();
    return content ? [{ role: "assistant", content }] : [];
  });
  return [...history.slice(-12), { role: "user", content: finalPrompt }];
}

function parseSseEvent(eventText: string) {
  if (!eventText.trim()) return null;
  let eventName = "message";
  let dataStr = "";
  for (const line of eventText.split(/\r?\n/)) {
    if (line.startsWith("event: ")) eventName = line.slice(7).trim();
    if (line.startsWith("data: ")) dataStr += line.slice(6).trim();
  }
  if (!dataStr) return null;
  try {
    return { eventName, data: JSON.parse(dataStr) as Record<string, unknown> };
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatToolInput(name: string, input: Record<string, unknown>) {
  if (name === "Bash") return String(input.command ?? "");
  if (name === "Read") return String(input.file_path ?? "");
  if (name === "Edit" || name === "Write") return String(input.file_path ?? "");
  if (name === "Grep") return `${input.pattern ?? ""} ${input.path ? `- ${input.path}` : ""}`.trim();
  if (name === "Glob") return String(input.pattern ?? "");
  if (name === "WebFetch") return String(input.url ?? "");
  if (name === "WebSearch") return String(input.query ?? "");
  if (name === "Task") return String(input.description ?? input.subagent_type ?? "");
  const json = JSON.stringify(input);
  return json.length > 120 ? `${json.slice(0, 120)}...` : json;
}

function toStoredWindows(windows: ChatWindow[]) {
  return windows
    .slice(-MAX_STORED_WINDOWS)
    .map((win) => ({
      ...win,
      tabs: win.tabs.slice(-MAX_STORED_TABS).map((tab) => ({
        ...tab,
        pending: false,
        messages: tab.messages.slice(-MAX_STORED_MESSAGES).map((message) =>
          message.role === "assistant"
            ? { ...message, pending: false, startupPhase: null }
            : message,
        ),
      })),
    }));
}

function readStoredWindows(storageKey: string): ChatWindow[] {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatWindow[];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((win) => typeof win.key === "string" && Array.isArray(win.tabs))
      .slice(-MAX_STORED_WINDOWS)
      .map((win) => ({
        ...win,
        sourceLabel:
          win.sourceLabel ?? inferSourceLabel(win.scopeKind, win.contextMd, win.scopeId),
        open: Boolean(win.open),
        tabs: win.tabs.length > 0 ? win.tabs : [makeTab(1, "")],
      }));
  } catch {
    return [];
  }
}
