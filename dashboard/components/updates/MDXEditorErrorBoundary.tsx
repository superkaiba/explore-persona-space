"use client";

/**
 * React error boundary that wraps the MDXEditor subtree so an
 * unparseable body can NEVER white-screen the /updates editor.
 *
 * Why this exists (the load-bearing fact): `@mdxeditor/editor` v4 has
 * two failure modes when it imports markdown -> Lexical at mount:
 *
 *   1. RECOGNIZED parse errors (`MarkdownParseError` /
 *      `UnrecognizedMarkdownConstructError`) are CAUGHT inside
 *      `tryImportingMarkdown` and surfaced via the `onError` prop. A
 *      parent setState in `onError` handles this path.
 *   2. ANY OTHER error (a Lexical node registration error, a
 *      CodeMirror/plugin init throw, or the explicit rethrow in the
 *      `else` branch of `tryImportingMarkdown`) is RETHROWN. That
 *      rethrow happens synchronously inside `RealmWithPlugins`'s
 *      `React.useMemo` during the MDXEditor subtree's render. React
 *      cannot recover from a render-phase throw via setState in the
 *      same component — only an error boundary ABOVE the throwing
 *      subtree can catch it.
 *
 * `onError` only covers path (1). Path (2) needs THIS boundary. There
 * is no other error boundary anywhere in the dashboard, so without it a
 * path-(2) crash propagates past CardBodyEditor with zero in-place
 * recovery.
 *
 * Contract: on catch, the boundary renders `null` (so its own subtree
 * stops re-throwing this render) and calls `onCrash(message)` so the
 * PARENT can flip to the raw-markdown textarea. Once the parent's
 * fallback is active it stops mounting this boundary + MDXEditor
 * entirely, so the textarea renders INSTEAD OF the throwing subtree —
 * never as a sibling that would re-mount and re-crash.
 */

import { Component, type ReactNode } from "react";

type Props = {
  children: ReactNode;
  /**
   * Called from `componentDidCatch` with the crash message. The parent
   * uses this to flip to the raw-markdown fallback. Calling a parent
   * setState from `componentDidCatch` is the supported React pattern;
   * `getDerivedStateFromError` keeps this boundary's own subtree from
   * re-throwing on the crashing render.
   */
  onCrash: (message: string) => void;
  /**
   * Mirrors the parent's fallback flag. Not used to gate rendering here
   * (the parent stops mounting this boundary once fallback is active),
   * but kept on the props so the relationship is explicit and the
   * boundary stays a pure, dumb crash-catcher.
   */
  fallbackActive: boolean;
};

type State = { crashed: boolean };

export class MDXEditorErrorBoundary extends Component<Props, State> {
  state: State = { crashed: false };

  static getDerivedStateFromError(): State {
    // Stop the boundary's own subtree from re-throwing on this render
    // pass. We render `null` below; the parent's onCrash-driven
    // setState then re-renders WITHOUT this boundary + MDXEditor and
    // shows the textarea instead.
    return { crashed: true };
  }

  componentDidCatch(error: Error) {
    this.props.onCrash(error?.message ?? String(error));
  }

  render() {
    if (this.state.crashed) {
      // Render nothing — the parent renders the raw-markdown textarea
      // instead once onCrash flips its fallback flag. We just stop the
      // crash from propagating and signal up.
      return null;
    }
    return this.props.children;
  }
}

export default MDXEditorErrorBoundary;
