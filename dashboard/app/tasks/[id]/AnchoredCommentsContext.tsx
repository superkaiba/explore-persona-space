"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

/**
 * Shared state binding anchored-comments on the body to the comment list
 * on the sidebar (Sagan dashboard pattern).
 *
 * - `anchors`: committed (comment_id -> quote) pairs we render as <mark>
 *   in the body.
 * - `hoveredId`: which comment is currently hovered in the sidebar; the
 *   matching <mark>(s) in the body get a stronger background.
 * - `pendingQuote`: the user just selected text in the body and clicked
 *   "+ Comment on selection"; the CommentForm reads this to prefill +
 *   attach the anchor when posting. While pendingQuote is set, the body
 *   keeps a `<mark data-anchor-pending>` visible.
 * - `anchorPositions`: rendered Y offsets (relative to the grid wrapper)
 *   of each <mark>. Published by CommentableBody after wrapping +
 *   measurement; consumed by CommentList to sort + vertically align
 *   comments next to their anchors.
 * - `scrollToCommentId`: a request from CommentList ("clicked c003") to
 *   the body to scroll its <mark> into view.
 */
export type AnchorRecord = { id: string; quote: string };
export type AnchorPosition = { id: string; top: number; height: number };

type Ctx = {
  anchors: AnchorRecord[];
  hoveredId: string | null;
  setHoveredId: (id: string | null) => void;
  pendingQuote: string | null;
  setPendingQuote: (q: string | null) => void;
  anchorPositions: AnchorPosition[];
  setAnchorPositions: (positions: AnchorPosition[]) => void;
  scrollToCommentId: string | null;
  requestScrollTo: (id: string) => void;
  clearScrollRequest: () => void;
};

const C = createContext<Ctx | null>(null);

export function AnchoredCommentsProvider({
  anchors,
  children,
}: {
  anchors: AnchorRecord[];
  children: ReactNode;
}) {
  const [hoveredId, setHoveredIdState] = useState<string | null>(null);
  const [pendingQuote, setPendingQuoteState] = useState<string | null>(null);
  const [anchorPositions, setAnchorPositionsState] = useState<AnchorPosition[]>([]);
  const [scrollToCommentId, setScrollToCommentId] = useState<string | null>(null);

  const setHoveredId = useCallback((id: string | null) => setHoveredIdState(id), []);
  const setPendingQuote = useCallback(
    (q: string | null) => setPendingQuoteState(q),
    [],
  );
  // Dedupe identical position arrays so a re-measure that produces the
  // same numbers doesn't trigger a re-layout of the sidebar.
  const setAnchorPositions = useCallback((positions: AnchorPosition[]) => {
    setAnchorPositionsState((prev) =>
      samePositions(prev, positions) ? prev : positions,
    );
  }, []);
  const requestScrollTo = useCallback(
    (id: string) => setScrollToCommentId(id),
    [],
  );
  const clearScrollRequest = useCallback(() => setScrollToCommentId(null), []);

  const value = useMemo<Ctx>(
    () => ({
      anchors,
      hoveredId,
      setHoveredId,
      pendingQuote,
      setPendingQuote,
      anchorPositions,
      setAnchorPositions,
      scrollToCommentId,
      requestScrollTo,
      clearScrollRequest,
    }),
    [
      anchors,
      hoveredId,
      setHoveredId,
      pendingQuote,
      setPendingQuote,
      anchorPositions,
      setAnchorPositions,
      scrollToCommentId,
      requestScrollTo,
      clearScrollRequest,
    ],
  );
  return <C.Provider value={value}>{children}</C.Provider>;
}

export function useAnchoredComments(): Ctx {
  const c = useContext(C);
  if (!c) {
    return {
      anchors: [],
      hoveredId: null,
      setHoveredId: () => {},
      pendingQuote: null,
      setPendingQuote: () => {},
      anchorPositions: [],
      setAnchorPositions: () => {},
      scrollToCommentId: null,
      requestScrollTo: () => {},
      clearScrollRequest: () => {},
    };
  }
  return c;
}

function samePositions(a: AnchorPosition[], b: AnchorPosition[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    if (x.id !== y.id || Math.abs(x.top - y.top) > 0.5 || Math.abs(x.height - y.height) > 0.5) {
      return false;
    }
  }
  return true;
}
