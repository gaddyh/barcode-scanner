import { KpiCard } from "../components/KpiCard";
import { BarChart } from "../components/BarChart";
import { useAdminMetrics } from "../useAdminMetrics";
import type { MetricsResponse } from "../types";

interface QualityPageProps {
  hours: number;
}

export function QualityPage({ hours }: QualityPageProps) {
  const { data, loading, error } = useAdminMetrics({ hours });

  if (loading) return <div style={styles.loading}>Loading…</div>;
  if (error) return <div style={styles.error}>Error: {error}</div>;
  if (!data) return null;

  const m = data as MetricsResponse;

  if (m.source === "unavailable") {
    return <div style={styles.error}>LangSmith is unavailable.</div>;
  }

  const issueData = Object.entries(m.primary_issue_counts)
    .map(([label, value]) => ({ label, value, displayValue: String(value) }))
    .sort((a, b) => b.value - a.value);

  return (
    <div>
      <h2 style={styles.h2}>Quality KPIs</h2>
      <div style={styles.kpiRow}>
        <KpiCard label="Scanner/vision match" value={m.scanner_vision_match_pct.toFixed(1)} unit="%" />
        <KpiCard label="Avg count delta" value={m.avg_count_delta.toFixed(1)} />
        <KpiCard label="Avg missing" value={m.avg_missing_count.toFixed(1)} />
        <KpiCard label="Avg unassigned" value={m.avg_unassigned_count.toFixed(1)} />
      </div>

      <h2 style={styles.h2}>Completion breakdown</h2>
      <div style={styles.breakdown}>
        <div style={styles.breakdownRow}>
          <span style={styles.breakdownLabel}>First-pass complete</span>
          <span style={styles.breakdownValue}>{m.first_pass_complete_pct.toFixed(1)}%</span>
        </div>
        <div style={styles.breakdownRow}>
          <span style={styles.breakdownLabel}>Recovered to complete</span>
          <span style={styles.breakdownValue}>{m.recovered_complete_pct.toFixed(1)}%</span>
        </div>
        <div style={styles.breakdownRow}>
          <span style={styles.breakdownLabel}>Still incomplete</span>
          <span style={styles.breakdownValue}>{m.still_incomplete_pct.toFixed(1)}%</span>
        </div>
      </div>

      <h2 style={styles.h2}>Primary issue taxonomy</h2>
      <BarChart data={issueData} />
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  h2: {
    fontSize: 16,
    fontWeight: 600,
    color: "#1e293b",
    margin: "24px 0 16px 0",
  },
  kpiRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 12,
  },
  breakdown: {
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    padding: "16px 20px",
  },
  breakdownRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "8px 0",
    borderBottom: "1px solid #f1f5f9",
  },
  breakdownLabel: {
    fontSize: 14,
    color: "#475569",
  },
  breakdownValue: {
    fontSize: 16,
    fontWeight: 700,
    color: "#1e293b",
    fontVariantNumeric: "tabular-nums",
  },
  loading: {
    color: "#64748b",
    padding: "24px 0",
  },
  error: {
    color: "#dc2626",
    padding: "24px 0",
  },
};
