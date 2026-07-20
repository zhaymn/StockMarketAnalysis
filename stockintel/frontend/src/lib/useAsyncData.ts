"use client";

import { useEffect, useRef, useState } from "react";
import { ApiRequestError } from "./api";

/**
 * Fetch data for a key, with cancellation.
 *
 * Deliberately never calls setState synchronously inside the effect body. The
 * obvious implementation (`setIsLoading(true)` at the top of the effect) forces
 * a second render pass on every dependency change, which React 19's
 * `set-state-in-effect` rule flags. Instead the resolved key is stored
 * alongside the data and loading is *derived* from whether it matches the
 * requested key — one render, no cascade.
 *
 * Data for a stale key is never returned. Showing the previous stock's
 * analytics under the newly selected stock's header would be a correctness bug,
 * not just a visual flicker.
 */
export function useAsyncData<T>(
  key: string | null,
  fetcher: (signal: AbortSignal) => Promise<T>,
): {
  data: T | null;
  error: ApiRequestError | null;
  isLoading: boolean;
} {
  const [result, setResult] = useState<{
    key: string | null;
    data: T | null;
    error: ApiRequestError | null;
  }>({ key: null, data: null, error: null });

  // The fetcher closes over props and so is a new function every render;
  // holding it in a ref keeps it out of the dependency array, which would
  // otherwise refire the effect on every render.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  useEffect(() => {
    if (key === null) return;

    const controller = new AbortController();

    fetcherRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setResult({ key, data, error: null });
      })
      .catch((cause) => {
        // AbortError is our own cleanup, not a failure to report.
        if (controller.signal.aborted || cause instanceof DOMException) return;
        setResult({
          key,
          data: null,
          error: cause instanceof ApiRequestError ? cause : null,
        });
      });

    return () => controller.abort();
  }, [key]);

  const isSettled = result.key === key && key !== null;

  return {
    data: isSettled ? result.data : null,
    error: isSettled ? result.error : null,
    isLoading: !isSettled,
  };
}
