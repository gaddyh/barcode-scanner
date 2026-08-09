import { KpiCard } from "../components/KpiCard";
import { useAdminMetrics } from "../useAdminMetrics";
import type { MetricsResponse } from "../types";

interface OverviewPageProps {
  hours: number;
}

export function OverviewPage({ hours }: OverviewPageProps) {
  const { data, loading, error } = useAdminMetrics({ hours });

  if (loading) return <div style={styles.loading}>Loading…</div>;
  if (error) return <div style={styles.error}>Error: {error}</div>;
  if (!data) return null;

  const m = data as MetricsResponse;

  if (m.source === "unavailable") {
    return <div style={styles.error}>LangSmith is unavailable.</div>;
  }

  // Source breakdown via the version breakdown data isn't available in the
  // flat response. We'd need a second fetch with group_by=source.
  // For now, show KPIs only. The source breakdown is in VersionsPage.

  return (
    <div>
      <h2 style={styles.h2}>KPIs</h2>
      <div style={styles.kpiRow}>
        <KpiCard label="Images" value={m.images_processed} />
        <KpiCard label="Boxes" value={m.boxes_processed} />
        <KpiCard label="Final complete" value={m.final_complete_pct.toFixed(1)} unit="%" />
        <KpiCard label="First-pass" value={m.first_pass_complete_pct.toFixed(1)} unit="%" />
        <KpiCard label="Retry required" value={m.user_retry_required_pct.toFixed(1)} unit="%" />
        <KpiCard label="P95 latency" value={m.p95_latency_ms >= 1000 ? `${(m.p95_latency_ms / 1000).toFixed(1)}s` : `${Math.round(m.p95_latency_ms)}ms`} />
      </div>

      {m.truncated && (
        <div style={styles.warn}>
          ⚠ Query hit the 100-run limit — metrics are from a sample, not the full window.
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  h2: {
    fontSize: 16,
    fontWeight: 600,
    color: "#1e293b",
    margin: "0 0 16px 0",
  },
  kpiRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 24,
  },
  loading: {
    color: "#64748b",
    padding: "24px 0",
  },
  error: {
    color: "#dc2626",
    padding: "24px 0",
  },
  warn: {
    background: "#fef3c7",
    border: "1px solid #fcd34d",
    borderRadius: 6,
    padding: "10px 14px",
    fontSize: 13,
    color: "#92400e",
    marginTop: 16,
  },
};
