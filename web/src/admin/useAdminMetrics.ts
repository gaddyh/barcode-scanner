import { useEffect, useState } from "react";
import { fetchGroupedMetrics, fetchMetrics } from "../api";
import type { GroupedMetricsResponse, MetricsResponse } from "./types";

interface UseAdminMetricsArgs {
  hours: number;
  groupBy?: string;
}

interface UseAdminMetricsResult {
  data: MetricsResponse | GroupedMetricsResponse | null;
  loading: boolean;
  error: string | null;
}

/** Shared hook: fetch /admin/metrics with optional group_by. */
export function useAdminMetrics({
  hours,
  groupBy,
}: UseAdminMetricsArgs): UseAdminMetricsResult {
  const [data, setData] = useState<
    MetricsResponse | GroupedMetricsResponse | null
  >(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const promise = groupBy
      ? fetchGroupedMetrics(hours, groupBy)
      : fetchMetrics(hours);

    promise
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [hours, groupBy]);

  return { data, loading, error };
}
